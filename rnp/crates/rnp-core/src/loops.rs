//! The strided/contiguous inner-loop drivers shared by every ufunc.
//!
//! Three shapes of walk, in the order numpy's own iterator would pick them:
//!  1. both operands contiguous and already the output shape — a flat typed
//!     loop LLVM can vectorise;
//!  2. every operand a single arithmetic progression (numpy's nditer coalesces
//!     the same axes) — two stepped pointers;
//!  3. anything else — the odometer in [`crate::iter::offsets`].
//!
//! Each driver comes in a plain and a `_flagged` variant. The flagged one
//! runs the *same* full-speed loop and then scans the output with a cheap
//! predicate; only if that fires does it walk the operands again to attribute
//! the exact `divide`/`overflow`/`invalid` mask (see [`crate::fpe`]). Doing
//! the attribution inside the loop instead costs 3-4x on `add_f64`, because
//! the NaN/infinity tests block vectorisation.

use crate::array::NdArray;
use crate::element::Element;
use crate::fpe;
use crate::iter::offsets;

/// Element count above which a loop is split across rayon. Element-wise
/// results are independent, so any split is bit-identical.
pub(crate) const PAR_THRESHOLD: usize = 1 << 16;

/// Chunk handed to each rayon task.
pub(crate) const PAR_CHUNK: usize = 1 << 14;

/// True when the array's elements sit at `byte_offset + i * itemsize` in
/// order, i.e. a flat typed slice can be formed.
#[inline]
pub(crate) fn flat_ptr<T>(a: &NdArray) -> Option<*const T> {
    if !a.flags.c_contiguous {
        return None;
    }
    // SAFETY: `byte_offset` is always a multiple of the itemsize and the
    // allocation is 64-byte aligned, so the result is well aligned for T; the
    // array's `size()` elements from here are inside the allocation.
    unsafe { Some(a.buffer.as_ptr().offset(a.byte_offset) as *const T) }
}

/// A raw pointer that rayon is allowed to move into a task.
///
/// Safety is argued at each use: the pointed-at buffers are read-only for the
/// duration of the parallel section (or, for the output, split into disjoint
/// chunks by `par_chunks_mut`).
#[derive(Copy, Clone)]
pub(crate) struct SendPtr(pub *const u8);

impl SendPtr {
    /// A method, not a field access: edition-2021 closures capture disjoint
    /// *fields*, so `p.0` inside a rayon task would capture the bare raw
    /// pointer (which is not `Sync`) instead of this wrapper.
    #[inline]
    pub(crate) fn ptr(self) -> *const u8 {
        self.0
    }
}
// SAFETY: see the type comment; the pointer is only dereferenced inside
// bounds already validated on the owning thread.
unsafe impl Send for SendPtr {}
// SAFETY: as above; shared access is read-only.
unsafe impl Sync for SendPtr {}

/// If every element of `a` sits at `base + i * stride` in C order, return
/// `(stride, base)`.
///
/// This is true for any array whose axes, read from the innermost out, keep
/// the product `stride_k * shape_k == stride_{k-1}` invariant -- i.e. exactly
/// the shapes numpy's iterator coalesces into one dimension. Axes of length
/// one impose no constraint, and a zero stride (a broadcast axis) only
/// collapses when the whole walk is degenerate.
#[inline]
pub(crate) fn linear_walk(a: &NdArray) -> Option<(isize, isize)> {
    if a.ndim() == 0 {
        return Some((0, a.byte_offset));
    }
    let mut stride = 0isize;
    let mut expect = 0isize;
    let mut first = true;
    for ax in (0..a.ndim()).rev() {
        if a.shape[ax] == 1 {
            continue;
        }
        if a.shape[ax] == 0 {
            return Some((0, a.byte_offset));
        }
        if first {
            stride = a.strides[ax];
            expect = stride * a.shape[ax];
            first = false;
        } else {
            if a.strides[ax] != expect {
                return None;
            }
            expect *= a.shape[ax];
        }
    }
    Some((stride, a.byte_offset))
}

// ---------------------------------------------------------------------------
// Binary drivers
// ---------------------------------------------------------------------------

/// The inner loop, generic over the element type *and* the operation.
///
/// `F` is a distinct zero-sized type per call site, so the operation inlines
/// into the loop body and LLVM can vectorise the contiguous path. Passing a
/// `fn(T, T) -> U` pointer instead costs roughly 4x on f64 adds.
#[inline]
pub(crate) fn binary2<T, U, F>(a: &NdArray, b: &NdArray, o: *mut U, n: usize, f: F)
where
    T: Element + Sync,
    U: Copy + Send,
    F: Fn(T, T) -> U + Sync + Send,
{
    binary2_watch::<T, U, _, _>(a, b, o, n, f, |_, _, _| false);
}

/// As [`binary2`], additionally OR-folding a cheap per-element predicate and
/// reporting whether it ever fired.
///
/// `watch` is a single compare for the ops that use it and a constant `false`
/// for the ops that do not, so the accumulator disappears entirely from the
/// loops that cannot raise. Folding it here rather than re-scanning the output
/// afterwards matters: a separate pass over a 1e6-element `f64` result costs
/// more than the arithmetic did.
///
/// The predicate is handed the operands as well as the result, because a few
/// ops can raise while returning a perfectly ordinary number -- `remainder`
/// reports the overflow of the quotient it computed on the way, and the complex
/// comparisons report the NaN their signalling compare tripped over. Closures
/// that ignore the operands compile to exactly what they did before.
#[inline]
pub(crate) fn binary2_watch<T, U, F, S>(
    a: &NdArray,
    b: &NdArray,
    o: *mut U,
    n: usize,
    f: F,
    watch: S,
) -> bool
where
    T: Element + Sync,
    U: Copy + Send,
    F: Fn(T, T) -> U + Sync + Send,
    S: Fn(T, T, U) -> bool + Sync + Send + Copy,
{
    let mut hit = false;
    if let (Some(pa), Some(pb)) = (flat_ptr::<T>(a), flat_ptr::<T>(b)) {
        if n >= PAR_THRESHOLD {
            use rayon::prelude::*;
            // SAFETY: `o` owns `n` freshly allocated elements that nothing
            // else refers to yet, and `pa`/`pb` are contiguous runs of `n`
            // elements of a live buffer.
            let (dst, sa, sb) = unsafe {
                (
                    std::slice::from_raw_parts_mut(o, n),
                    std::slice::from_raw_parts(pa, n),
                    std::slice::from_raw_parts(pb, n),
                )
            };
            return dst
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .map(|(c, chunk)| {
                    let base = c * PAR_CHUNK;
                    let mut local = false;
                    for (k, slot) in chunk.iter_mut().enumerate() {
                        let (x, y) = (sa[base + k], sb[base + k]);
                        let r = f(x, y);
                        local |= watch(x, y, r);
                        *slot = r;
                    }
                    local
                })
                .reduce(|| false, |p, q| p | q);
        }
        for i in 0..n {
            // SAFETY: i < n and all three buffers hold at least n elements.
            unsafe {
                let (x, y) = (*pa.add(i), *pb.add(i));
                let r = f(x, y);
                hit |= watch(x, y, r);
                *o.add(i) = r;
            }
        }
        return hit;
    }
    if let (Some((sa, base_a)), Some((sb, base_b))) = (linear_walk(a), linear_walk(b)) {
        let pa = a.buffer.as_ptr();
        let pb = b.buffer.as_ptr();
        // A broadcast scalar (stride 0) is handled below, before the generic
        // two-pointer walk, because hoisting its load is worth more than the
        // rayon split for the sizes where both apply.
        if n >= PAR_THRESHOLD && sa != 0 && sb != 0 {
            use rayon::prelude::*;
            // SAFETY: `o` owns `n` freshly allocated elements nothing else
            // refers to, and each chunk touches a disjoint slice of them.
            let dst = unsafe { std::slice::from_raw_parts_mut(o, n) };
            let (pa, pb) = (SendPtr(pa), SendPtr(pb));
            return dst
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .map(|(c, chunk)| {
                    let start = (c * PAR_CHUNK) as isize;
                    let mut local = false;
                    for (k, slot) in chunk.iter_mut().enumerate() {
                        let i = start + k as isize;
                        // SAFETY: as the serial path below.
                        unsafe {
                            let x = std::ptr::read_unaligned(
                                pa.ptr().offset(base_a + i * sa) as *const T,
                            );
                            let y = std::ptr::read_unaligned(
                                pb.ptr().offset(base_b + i * sb) as *const T,
                            );
                            let r = f(x, y);
                            local |= watch(x, y, r);
                            *slot = r;
                        }
                    }
                    local
                })
                .reduce(|| false, |p, q| p | q);
        }
        // A broadcast scalar (stride 0) on one side is worth its own loop:
        // hoisting the load lets LLVM vectorise the other operand.
        if sa == 0 || sb == 0 {
            // SAFETY: each scalar operand's single element lives at its base.
            let (scalar_first, sv, step, base) = unsafe {
                if sa == 0 {
                    (
                        true,
                        std::ptr::read_unaligned(pa.offset(base_a) as *const T),
                        sb,
                        base_b,
                    )
                } else {
                    (
                        false,
                        std::ptr::read_unaligned(pb.offset(base_b) as *const T),
                        sa,
                        base_a,
                    )
                }
            };
            let other = SendPtr(if sa == 0 { pb } else { pa });
            // The (x, y) pair the op sees, in operand order.
            let pair = move |v: T| if scalar_first { (sv, v) } else { (v, sv) };
            // The overwhelmingly common shape of `array OP python-number`:
            // the non-scalar side is contiguous. Walking it with a byte
            // pointer and a *runtime* step, as the general loop below does,
            // hides that fact from LLVM and costs the vectoriser entirely --
            // `bitwise_and(i32_array, 255)` ran 2.6x numpy purely because of
            // the pointer shape. Forming real slices puts it back.
            if step == std::mem::size_of::<T>() as isize
                && base % std::mem::size_of::<T>() as isize == 0
            {
                // SAFETY: the operand is contiguous over the `n` elements
                // starting at `base`, `base` is a multiple of the itemsize (so
                // the pointer is aligned for T), and `o` owns `n` freshly
                // allocated elements nothing else refers to.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(other.ptr().offset(base) as *const T, n),
                        std::slice::from_raw_parts_mut(o, n),
                    )
                };
                if n >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    return dst
                        .par_chunks_mut(PAR_CHUNK)
                        .zip(src.par_chunks(PAR_CHUNK))
                        .map(|(chunk, s)| {
                            let mut local = false;
                            for (slot, &v) in chunk.iter_mut().zip(s) {
                                let (x, y) = pair(v);
                                let r = f(x, y);
                                local |= watch(x, y, r);
                                *slot = r;
                            }
                            local
                        })
                        .reduce(|| false, |p, q| p | q);
                }
                for (slot, &v) in dst.iter_mut().zip(src) {
                    let (x, y) = pair(v);
                    let r = f(x, y);
                    hit |= watch(x, y, r);
                    *slot = r;
                }
                return hit;
            }
            let apply = move |v: T| {
                let (x, y) = pair(v);
                f(x, y)
            };
            if n >= PAR_THRESHOLD {
                use rayon::prelude::*;
                // SAFETY: `o` owns `n` freshly allocated elements nothing else
                // refers to, and each chunk touches a disjoint slice.
                let dst = unsafe { std::slice::from_raw_parts_mut(o, n) };
                return dst
                    .par_chunks_mut(PAR_CHUNK)
                    .enumerate()
                    .map(|(c, chunk)| {
                        let start = (c * PAR_CHUNK) as isize;
                        let mut local = false;
                        for (k, slot) in chunk.iter_mut().enumerate() {
                            let i = start + k as isize;
                            // SAFETY: base + i*step is in bounds for i < n.
                            unsafe {
                                let v = std::ptr::read_unaligned(
                                    other.ptr().offset(base + i * step) as *const T,
                                );
                                let (x, y) = pair(v);
                                let r = apply(v);
                                local |= watch(x, y, r);
                                *slot = r;
                            }
                        }
                        local
                    })
                    .reduce(|| false, |p, q| p | q);
            }
            for i in 0..n {
                // SAFETY: base + i*step stays inside the array for i < n.
                unsafe {
                    let v = std::ptr::read_unaligned(
                        other.ptr().offset(base + i as isize * step) as *const T,
                    );
                    let (x, y) = pair(v);
                    let r = apply(v);
                    hit |= watch(x, y, r);
                    *o.add(i) = r;
                }
            }
            return hit;
        }
        for i in 0..n {
            // SAFETY: base + i*stride stays within a's (and b's) elements for
            // i < n, since both were broadcast to the output shape.
            unsafe {
                let x = std::ptr::read_unaligned(pa.offset(base_a + i as isize * sa) as *const T);
                let y = std::ptr::read_unaligned(pb.offset(base_b + i as isize * sb) as *const T);
                let r = f(x, y);
                hit |= watch(x, y, r);
                *o.add(i) = r;
            }
        }
        return hit;
    }

    let ia = offsets(&a.shape, &a.strides, a.byte_offset);
    let ib = offsets(&b.shape, &b.strides, b.byte_offset);
    for (i, (oa, ob)) in ia.zip(ib).enumerate() {
        // SAFETY: offsets come from in-bounds strided iteration over a and b,
        // and i < n because the iterators yield exactly n elements each.
        unsafe {
            let x = std::ptr::read_unaligned(a.buffer.as_ptr().offset(oa) as *const T);
            let y = std::ptr::read_unaligned(b.buffer.as_ptr().offset(ob) as *const T);
            let r = f(x, y);
            hit |= watch(x, y, r);
            *o.add(i) = r;
        }
    }
    hit
}

/// As [`binary2`], but watching for the results that *could* carry a
/// floating-point error condition.
///
/// The watch predicate is deliberately cheap (one bit-mask compare, which LLVM
/// keeps inside the vectorised body as a compare-and-or), so the common case
/// -- a loop whose results are all ordinary finite numbers -- costs nothing
/// beyond that. Only when the watch fires does the second, expensive pass run
/// to attribute the exact `divide`/`overflow`/`invalid` mask, and only over
/// the elements that were flagged.
#[inline]
pub(crate) fn binary2_flagged<T, U, F, S, G>(
    a: &NdArray,
    b: &NdArray,
    o: *mut U,
    n: usize,
    f: F,
    watch: S,
    g: G,
) where
    T: Element + Sync,
    U: Copy + Send,
    F: Fn(T, T) -> U + Sync + Send,
    S: Fn(T, T, U) -> bool + Sync + Send + Copy,
    G: Fn(T, T, U) -> u8 + Sync + Send,
{
    if !binary2_watch::<T, U, _, _>(a, b, o, n, f, watch) {
        return;
    }
    // SAFETY: the loop above wrote `n` elements of U at `o`.
    let out = unsafe { std::slice::from_raw_parts(o, n) };
    let mut flags = 0u8;
    let (pa, pb) = (a.buffer.as_ptr(), b.buffer.as_ptr());
    // The attribution pass has to reach the operands of the elements that
    // fired, so it needs the same walk the main loop used -- going through the
    // odometer unconditionally would make a single zero product (say
    // `arange(1e6) * arange(1e6)`) cost more than the whole multiply.
    if let (Some((sa, base_a)), Some((sb, base_b))) = (linear_walk(a), linear_walk(b)) {
        for (i, &r) in out.iter().enumerate() {
            // SAFETY: base + i*stride is in bounds for i < n.
            unsafe {
                let x = std::ptr::read_unaligned(pa.offset(base_a + i as isize * sa) as *const T);
                let y = std::ptr::read_unaligned(pb.offset(base_b + i as isize * sb) as *const T);
                if watch(x, y, r) {
                    flags |= g(x, y, r);
                }
            }
        }
        fpe::raise(flags);
        return;
    }
    let ia = offsets(&a.shape, &a.strides, a.byte_offset);
    let ib = offsets(&b.shape, &b.strides, b.byte_offset);
    for (i, (oa, ob)) in ia.zip(ib).enumerate() {
        // SAFETY: the offsets come from in-bounds strided iteration over the
        // (already broadcast) operands.
        unsafe {
            let x = std::ptr::read_unaligned(pa.offset(oa) as *const T);
            let y = std::ptr::read_unaligned(pb.offset(ob) as *const T);
            if watch(x, y, out[i]) {
                flags |= g(x, y, out[i]);
            }
        }
    }
    fpe::raise(flags);
}

// ---------------------------------------------------------------------------
// Unary drivers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn unary1<T, U, F>(a: &NdArray, o: *mut U, n: usize, f: F)
where
    T: Element + Sync,
    U: Copy + Send,
    F: Fn(T) -> U + Sync + Send,
{
    unary1_watch::<T, U, _, _>(a, o, n, f, |_, _| false);
}

/// As [`unary1`], OR-folding a cheap per-element predicate.
#[inline]
pub(crate) fn unary1_watch<T, U, F, S>(
    a: &NdArray,
    o: *mut U,
    n: usize,
    f: F,
    watch: S,
) -> bool
where
    T: Element + Sync,
    U: Copy + Send,
    F: Fn(T) -> U + Sync + Send,
    S: Fn(T, U) -> bool + Sync + Send + Copy,
{
    let mut hit = false;
    if let Some(pa) = flat_ptr::<T>(a) {
        if n >= PAR_THRESHOLD {
            use rayon::prelude::*;
            // SAFETY: `o` owns `n` freshly allocated elements; `pa` is a
            // contiguous run of `n` elements of a live buffer.
            let (dst, sa) = unsafe {
                (
                    std::slice::from_raw_parts_mut(o, n),
                    std::slice::from_raw_parts(pa, n),
                )
            };
            return dst
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .map(|(c, chunk)| {
                    let base = c * PAR_CHUNK;
                    let mut local = false;
                    for (k, slot) in chunk.iter_mut().enumerate() {
                        let x = sa[base + k];
                        let r = f(x);
                        local |= watch(x, r);
                        *slot = r;
                    }
                    local
                })
                .reduce(|| false, |p, q| p | q);
        }
        for i in 0..n {
            // SAFETY: i < n and both buffers hold at least n elements.
            unsafe {
                let x = *pa.add(i);
                let r = f(x);
                hit |= watch(x, r);
                *o.add(i) = r;
            }
        }
        return hit;
    }
    if let Some((sa, base_a)) = linear_walk(a) {
        let pa = a.buffer.as_ptr();
        for i in 0..n {
            // SAFETY: base + i*stride stays inside `a` for i < n.
            unsafe {
                let x = std::ptr::read_unaligned(pa.offset(base_a + i as isize * sa) as *const T);
                let r = f(x);
                hit |= watch(x, r);
                *o.add(i) = r;
            }
        }
        return hit;
    }
    for (i, oa) in offsets(&a.shape, &a.strides, a.byte_offset).enumerate() {
        // SAFETY: offsets come from in-bounds strided iteration over `a`.
        unsafe {
            let x = std::ptr::read_unaligned(a.buffer.as_ptr().offset(oa) as *const T);
            let r = f(x);
            hit |= watch(x, r);
            *o.add(i) = r;
        }
    }
    hit
}

/// As [`binary2`], taking the error flags from the CPU's status register.
///
/// The binary counterpart of [`unary1_hw`]; used by complex `power`, whose
/// conditions come partly from the platform's `cpow` and partly from the
/// complex division `npy_cpow` performs for a negative exponent.
#[inline]
pub(crate) fn binary2_hw<T, F>(a: &NdArray, b: &NdArray, o: *mut T, n: usize, f: F)
where
    T: Element + Sync + Send,
    F: Fn(T, T) -> T + Sync + Send,
{
    if n >= PAR_THRESHOLD {
        if let (Some(pa), Some(pb)) = (flat_ptr::<T>(a), flat_ptr::<T>(b)) {
            use rayon::prelude::*;
            // SAFETY: `o` owns `n` freshly allocated elements that nothing else
            // refers to; `pa`/`pb` are contiguous runs of `n` live elements.
            let (dst, sa, sb) = unsafe {
                (
                    std::slice::from_raw_parts_mut(o, n),
                    std::slice::from_raw_parts(pa, n),
                    std::slice::from_raw_parts(pb, n),
                )
            };
            let flags = dst
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .map(|(c, chunk)| {
                    fpe::hw_clear();
                    let base = c * PAR_CHUNK;
                    for (k, slot) in chunk.iter_mut().enumerate() {
                        *slot = f(sa[base + k], sb[base + k]);
                    }
                    fpe::hw_take()
                })
                .reduce(|| 0u8, |p, q| p | q);
            fpe::raise(flags);
            return;
        }
    }
    fpe::hw_clear();
    binary2::<T, T, _>(a, b, o, n, f);
    fpe::raise(fpe::hw_take());
}

/// As [`unary1`], taking the error flags from the CPU's status register rather
/// than from the values.
///
/// This is the driver for the complex transcendentals, which call the platform
/// libm exactly as numpy does; see [`crate::fpe::hw_take`] for why the hardware
/// is the only faithful source there. The register is per thread, so a loop
/// that rayon splits has to clear and read it inside each chunk -- which is
/// still one pair of calls per 16k elements, not per element.
#[inline]
pub(crate) fn unary1_hw<T, F>(a: &NdArray, o: *mut T, n: usize, f: F)
where
    T: Element + Sync + Send,
    F: Fn(T) -> T + Sync + Send,
{
    if n >= PAR_THRESHOLD {
        if let Some(pa) = flat_ptr::<T>(a) {
            use rayon::prelude::*;
            // SAFETY: `o` owns `n` freshly allocated elements that nothing else
            // refers to, and `pa` is a contiguous run of `n` live elements.
            let (dst, sa) = unsafe {
                (
                    std::slice::from_raw_parts_mut(o, n),
                    std::slice::from_raw_parts(pa, n),
                )
            };
            let flags = dst
                .par_chunks_mut(PAR_CHUNK)
                .enumerate()
                .map(|(c, chunk)| {
                    fpe::hw_clear();
                    let base = c * PAR_CHUNK;
                    for (k, slot) in chunk.iter_mut().enumerate() {
                        *slot = f(sa[base + k]);
                    }
                    fpe::hw_take()
                })
                .reduce(|| 0u8, |p, q| p | q);
            fpe::raise(flags);
            return;
        }
    }
    // Every remaining path in `unary1` is serial, so one bracket covers it.
    fpe::hw_clear();
    unary1::<T, T, _>(a, o, n, f);
    fpe::raise(fpe::hw_take());
}

/// As [`unary1`], with the same two-phase error-flag scheme as
/// [`binary2_flagged`].
#[inline]
pub(crate) fn unary1_flagged<T, U, F, S, G>(
    a: &NdArray,
    o: *mut U,
    n: usize,
    f: F,
    watch: S,
    g: G,
) where
    T: Element + Sync,
    U: Copy + Send,
    F: Fn(T) -> U + Sync + Send,
    S: Fn(T, U) -> bool + Sync + Send + Copy,
    G: Fn(T, U) -> u8 + Sync + Send,
{
    if !unary1_watch::<T, U, _, _>(a, o, n, f, watch) {
        return;
    }
    // SAFETY: the loop above wrote `n` elements of U at `o`.
    let out = unsafe { std::slice::from_raw_parts(o, n) };
    let mut flags = 0u8;
    let pa = a.buffer.as_ptr();
    if let Some((sa, base_a)) = linear_walk(a) {
        for (i, &r) in out.iter().enumerate() {
            // SAFETY: base + i*stride is in bounds for i < n.
            unsafe {
                let x = std::ptr::read_unaligned(pa.offset(base_a + i as isize * sa) as *const T);
                if watch(x, r) {
                    flags |= g(x, r);
                }
            }
        }
        fpe::raise(flags);
        return;
    }
    for (i, oa) in offsets(&a.shape, &a.strides, a.byte_offset).enumerate() {
        // SAFETY: the offset comes from in-bounds strided iteration over `a`.
        unsafe {
            let x = std::ptr::read_unaligned(pa.offset(oa) as *const T);
            if watch(x, out[i]) {
                flags |= g(x, out[i]);
            }
        }
    }
    fpe::raise(flags);
}
