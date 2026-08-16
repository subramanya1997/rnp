//! Reductions: `sum`, `prod`, `min`, `max`, `argmin`, `argmax`.
//!
//! The float paths reproduce numpy's pairwise summation *exactly*, including
//! the 8 accumulators, the `PW_BLOCKSIZE = 128` cutoff, the `-0.0` seed of
//! the short tail and the "round the split down to a multiple of 8" rule
//! (`upstream/numpy/_core/src/umath/loops_utils.h.src`). Results are
//! therefore bit-identical to numpy's, which `harness/dev_check.py` asserts
//! on random arrays.
//!
//! Which grouping applies also follows numpy: a reduction whose axis is the
//! innermost one runs the pairwise inner loop, while a reduction across an
//! outer axis accumulates whole slices in order, because that is how numpy's
//! iterator lays the loops out.

use crate::array::NdArray;
use crate::dtype::{DType, Kind};
use crate::element::{Element, Scalar, C32, C64v, F16};
use crate::error::{Error, Result};
use crate::ops::Arith;

/// numpy's `PW_BLOCKSIZE`.
const PW_BLOCKSIZE: usize = 128;

/// Element count above which a reduction is split across rayon's pool.
///
/// For the float sums the split happens *on numpy's own pairwise-tree
/// boundaries*, so the parallel result is bit-identical to the serial one --
/// the recursion already brackets the additions, and rayon only decides which
/// thread evaluates each bracket.
const PAR_THRESHOLD: usize = 1 << 16;

/// A raw pointer that may cross a rayon `join`.
///
/// Sound because every parallel branch only ever *reads* a disjoint slice of
/// one immutable buffer.
#[derive(Copy, Clone)]
struct SendPtr(*const u8);
// SAFETY: see above -- read-only access to a live, shared allocation.
unsafe impl Send for SendPtr {}
// SAFETY: as above.
unsafe impl Sync for SendPtr {}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ReduceOp {
    Sum,
    Prod,
    Min,
    Max,
    ArgMin,
    ArgMax,
}

impl ReduceOp {
    pub fn name(self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Prod => "prod",
            ReduceOp::Min => "min",
            ReduceOp::Max => "max",
            ReduceOp::ArgMin => "argmin",
            ReduceOp::ArgMax => "argmax",
        }
    }

    fn is_arg(self) -> bool {
        matches!(self, ReduceOp::ArgMin | ReduceOp::ArgMax)
    }

    /// True when the reduction has no identity and must seed from the first
    /// element (numpy raises on empty input for these).
    fn needs_element(self) -> bool {
        !matches!(self, ReduceOp::Sum | ReduceOp::Prod)
    }
}

/// The dtype numpy gives the result of a reduction.
///
/// `sum`/`prod` accumulate small integers in the platform integer type
/// (`np.sum(np.int8 array).dtype == int64`); everything else keeps the
/// operand dtype, and the `arg*` family returns `intp`.
pub fn reduce_dtype(op: ReduceOp, src: DType) -> DType {
    match op {
        ReduceOp::ArgMin | ReduceOp::ArgMax => DType::I64,
        ReduceOp::Min | ReduceOp::Max => src,
        ReduceOp::Sum | ReduceOp::Prod => match src.category() {
            Kind::Bool => DType::I64,
            Kind::Int if src.itemsize() < 8 => DType::I64,
            Kind::Uint if src.itemsize() < 8 => DType::U64,
            _ => src,
        },
    }
}

// ---------------------------------------------------------------------------
// Pairwise summation
// ---------------------------------------------------------------------------

macro_rules! pairwise_float {
    ($name:ident, $store:ty, $acc:ty, $load:expr) => {
        /// numpy's `@TYPE@_pairwise_sum`, transcribed.
        ///
        /// # Safety
        /// `a` must point at `n` elements of `$store` spaced `stride` bytes
        /// apart, all inside one allocation.
        unsafe fn $name(a: *const u8, n: usize, stride: isize) -> $acc {
            let load = |i: usize| -> $acc {
                // SAFETY: caller guarantees element `i` is in bounds.
                let v: $store =
                    unsafe { std::ptr::read_unaligned(a.offset(i as isize * stride) as *const $store) };
                #[allow(clippy::redundant_closure_call)]
                $load(v)
            };
            if n < 8 {
                // numpy starts at -0.0 so that summing only -0.0 keeps the
                // sign (`0 + -0 == 0` but `-0 + -0 == -0`).
                let mut res: $acc = -0.0;
                for i in 0..n {
                    res += load(i);
                }
                return res;
            }
            if n <= PW_BLOCKSIZE {
                let mut r: [$acc; 8] = [
                    load(0),
                    load(1),
                    load(2),
                    load(3),
                    load(4),
                    load(5),
                    load(6),
                    load(7),
                ];
                let mut i = 8usize;
                while i < n - (n % 8) {
                    for k in 0..8 {
                        r[k] += load(i + k);
                    }
                    i += 8;
                }
                let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                while i < n {
                    res += load(i);
                    i += 1;
                }
                return res;
            }
            // Split in half, rounded down to a multiple of the unroll factor.
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            // SAFETY: both halves lie inside the caller's range.
            let right = unsafe { a.offset(n2 as isize * stride) };
            if n >= PAR_THRESHOLD {
                // Same brackets, different threads: bit-identical.
                let (l, r) = (SendPtr(a), SendPtr(right));
                let (x, y) = rayon::join(
                    // SAFETY: disjoint read-only halves of the same buffer.
                    // (`let p = l` captures the whole `SendPtr`, not the raw
                    // pointer field, which is what makes the closure `Send`.)
                    || {
                        let p = l;
                        unsafe { $name(p.0, n2, stride) }
                    },
                    // SAFETY: as above.
                    || {
                        let p = r;
                        unsafe { $name(p.0, n - n2, stride) }
                    },
                );
                return x + y;
            }
            // SAFETY: both halves lie inside the caller's range.
            unsafe { $name(a, n2, stride) + $name(right, n - n2, stride) }
        }
    };
}

pairwise_float!(pairwise_f32, f32, f32, |v: f32| v);
pairwise_float!(pairwise_f64, f64, f64, |v: f64| v);
// numpy sums halves in `float` and rounds once at the very end.
pairwise_float!(pairwise_f16, F16, f32, |v: F16| v.to_f32());

macro_rules! pairwise_complex {
    ($name:ident, $ftype:ty) => {
        /// numpy's complex `@TYPE@_pairwise_sum`. `n` counts *components*
        /// (twice the number of complex elements) and `stride` is the
        /// component stride, exactly as numpy calls it.
        ///
        /// # Safety
        /// As the real version.
        unsafe fn $name(a: *const u8, n: usize, stride: isize) -> ($ftype, $ftype) {
            let re = |i: usize| -> $ftype {
                // SAFETY: caller guarantees the component is in bounds.
                unsafe {
                    std::ptr::read_unaligned(a.offset(i as isize * stride) as *const $ftype)
                }
            };
            let im = |i: usize| -> $ftype {
                // SAFETY: as above; the imaginary part follows the real one.
                unsafe {
                    std::ptr::read_unaligned(
                        a.offset(i as isize * stride + std::mem::size_of::<$ftype>() as isize)
                            as *const $ftype,
                    )
                }
            };
            if n < 8 {
                let (mut rr, mut ri): ($ftype, $ftype) = (-0.0, -0.0);
                let mut i = 0usize;
                while i < n {
                    rr += re(i);
                    ri += im(i);
                    i += 2;
                }
                return (rr, ri);
            }
            if n <= PW_BLOCKSIZE {
                let mut r: [$ftype; 8] = [
                    re(0),
                    im(0),
                    re(2),
                    im(2),
                    re(4),
                    im(4),
                    re(6),
                    im(6),
                ];
                let mut i = 8usize;
                while i < n - (n % 8) {
                    r[0] += re(i);
                    r[1] += im(i);
                    r[2] += re(i + 2);
                    r[3] += im(i + 2);
                    r[4] += re(i + 4);
                    r[5] += im(i + 4);
                    r[6] += re(i + 6);
                    r[7] += im(i + 6);
                    i += 8;
                }
                let mut rr = (r[0] + r[2]) + (r[4] + r[6]);
                let mut ri = (r[1] + r[3]) + (r[5] + r[7]);
                while i < n {
                    rr += re(i);
                    ri += im(i);
                    i += 2;
                }
                return (rr, ri);
            }
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            // SAFETY: both halves lie inside the caller's range.
            let right = unsafe { a.offset(n2 as isize * stride) };
            if n >= PAR_THRESHOLD {
                let (l, r) = (SendPtr(a), SendPtr(right));
                let ((rr1, ri1), (rr2, ri2)) = rayon::join(
                    // SAFETY: disjoint read-only halves of the same buffer.
                    || {
                        let p = l;
                        unsafe { $name(p.0, n2, stride) }
                    },
                    // SAFETY: as above.
                    || {
                        let p = r;
                        unsafe { $name(p.0, n - n2, stride) }
                    },
                );
                return (rr1 + rr2, ri1 + ri2);
            }
            // SAFETY: both halves lie inside the caller's range.
            let (rr1, ri1) = unsafe { $name(a, n2, stride) };
            let (rr2, ri2) = unsafe { $name(right, n - n2, stride) };
            (rr1 + rr2, ri1 + ri2)
        }
    };
}

pairwise_complex!(pairwise_c64, f32);
pairwise_complex!(pairwise_c128, f64);

// ---------------------------------------------------------------------------
// Partial accumulator
// ---------------------------------------------------------------------------

/// A running reduction result, held in the accumulation type.
#[derive(Copy, Clone, Debug)]
enum Acc {
    Int(i64),
    Uint(u64),
    F16(f32),
    F32(f32),
    F64(f64),
    C64(C32),
    C128(C64v),
}

impl Acc {
    fn zero(dt: DType) -> Acc {
        match dt {
            DType::F16 => Acc::F16(0.0),
            DType::F32 => Acc::F32(0.0),
            DType::F64 => Acc::F64(0.0),
            DType::C64 => Acc::C64(C32::new(0.0, 0.0)),
            DType::C128 => Acc::C128(C64v::new(0.0, 0.0)),
            d if d.is_unsigned() => Acc::Uint(0),
            _ => Acc::Int(0),
        }
    }

    fn add(self, o: Acc) -> Acc {
        match (self, o) {
            (Acc::Int(a), Acc::Int(b)) => Acc::Int(a.wrapping_add(b)),
            (Acc::Uint(a), Acc::Uint(b)) => Acc::Uint(a.wrapping_add(b)),
            (Acc::F16(a), Acc::F16(b)) => Acc::F16(a + b),
            (Acc::F32(a), Acc::F32(b)) => Acc::F32(a + b),
            (Acc::F64(a), Acc::F64(b)) => Acc::F64(a + b),
            (Acc::C64(a), Acc::C64(b)) => Acc::C64(a + b),
            (Acc::C128(a), Acc::C128(b)) => Acc::C128(a + b),
            _ => self,
        }
    }

    fn to_scalar(self) -> Scalar {
        match self {
            Acc::Int(v) => Scalar::Int(v),
            Acc::Uint(v) => Scalar::Uint(v),
            // The float accumulator is rounded to the storage type once, at
            // the end, exactly as numpy's HALF loop does.
            Acc::F16(v) => Scalar::Float(F16::from_f32(v).to_f64()),
            Acc::F32(v) => Scalar::Float(v as f64),
            Acc::F64(v) => Scalar::Float(v),
            Acc::C64(v) => Scalar::Complex(C64v::new(v.re as f64, v.im as f64)),
            Acc::C128(v) => Scalar::Complex(v),
        }
    }
}

/// Sum one strided run, using numpy's grouping for the dtype.
///
/// # Safety
/// `off` must address `n` in-bounds elements spaced `stride` bytes apart.
unsafe fn sum_run(arr: &NdArray, off: isize, n: usize, stride: isize) -> Acc {
    // SAFETY: guaranteed by the caller.
    let p = unsafe { arr.buffer.as_ptr().offset(off) };
    match arr.dtype {
        // SAFETY: `p`/`n`/`stride` describe an in-bounds run of this dtype.
        DType::F16 => Acc::F16(unsafe { pairwise_f16(p, n, stride) }),
        DType::F32 => Acc::F32(unsafe { pairwise_f32(p, n, stride) }),
        DType::F64 => Acc::F64(unsafe { pairwise_f64(p, n, stride) }),
        DType::C64 => {
            let (re, im) = unsafe { pairwise_c64(p, 2 * n, stride / 2) };
            Acc::C64(C32::new(re, im))
        }
        DType::C128 => {
            let (re, im) = unsafe { pairwise_c128(p, 2 * n, stride / 2) };
            Acc::C128(C64v::new(re, im))
        }
        d if d.is_unsigned() => {
            let mut acc = 0u64;
            if stride == d.itemsize() as isize
                && (p as usize) % d.alignment().max(1) == 0
            {
                crate::dispatch_dtype!(d, T, {
                    // SAFETY: `n` contiguous, aligned, in-bounds elements.
                    let s: &[T] = unsafe { std::slice::from_raw_parts(p as *const T, n) };
                    acc = sum_int_contig(
                        s,
                        |v: T| match v.to_scalar() {
                            Scalar::Uint(u) => u,
                            Scalar::Int(x) => x as u64,
                            Scalar::Bool(b) => b as u64,
                            _ => 0,
                        },
                        |x: u64, y: u64| x.wrapping_add(y),
                        0u64,
                    );
                });
                return Acc::Uint(acc);
            }
            crate::dispatch_dtype!(d, T, {
                for i in 0..n {
                    // SAFETY: in-bounds by the caller's contract.
                    let v: T =
                        unsafe { std::ptr::read_unaligned(p.offset(i as isize * stride) as *const T) };
                    acc = acc.wrapping_add(match v.to_scalar() {
                        Scalar::Uint(u) => u,
                        Scalar::Int(x) => x as u64,
                        Scalar::Bool(b) => b as u64,
                        _ => 0,
                    });
                }
            });
            Acc::Uint(acc)
        }
        d => {
            let mut acc = 0i64;
            if stride == d.itemsize() as isize
                && (p as usize) % d.alignment().max(1) == 0
            {
                crate::dispatch_dtype!(d, T, {
                    // SAFETY: `n` contiguous, aligned, in-bounds elements.
                    let s: &[T] = unsafe { std::slice::from_raw_parts(p as *const T, n) };
                    acc = sum_int_contig(
                        s,
                        |v: T| match v.to_scalar() {
                            Scalar::Int(x) => x,
                            Scalar::Uint(u) => u as i64,
                            Scalar::Bool(b) => b as i64,
                            _ => 0,
                        },
                        |x: i64, y: i64| x.wrapping_add(y),
                        0i64,
                    );
                });
                return Acc::Int(acc);
            }
            crate::dispatch_dtype!(d, T, {
                for i in 0..n {
                    // SAFETY: in-bounds by the caller's contract.
                    let v: T =
                        unsafe { std::ptr::read_unaligned(p.offset(i as isize * stride) as *const T) };
                    acc = acc.wrapping_add(match v.to_scalar() {
                        Scalar::Int(x) => x,
                        Scalar::Uint(u) => u as i64,
                        Scalar::Bool(b) => b as i64,
                        _ => 0,
                    });
                }
            });
            Acc::Int(acc)
        }
    }
}

// ---------------------------------------------------------------------------
// Ordering helpers (numpy's NaN semantics)
// ---------------------------------------------------------------------------

/// numpy's `npy_isnan`, plus the signed-zero test the min/max reductions
/// need, for one element type.
pub trait MaybeNan: Copy {
    fn elem_is_nan(self) -> bool;
    /// True for `-0.0` only.
    fn is_neg_zero(self) -> bool {
        false
    }
    /// True for `+0.0` only.
    fn is_pos_zero(self) -> bool {
        false
    }
}

macro_rules! never_nan {
    ($($t:ty),*) => {$(impl MaybeNan for $t { #[inline] fn elem_is_nan(self) -> bool { false } })*};
}
never_nan!(i8, i16, i32, i64, u8, u16, u32, u64, crate::element::NpBool);

macro_rules! float_nan {
    ($($t:ty),*) => {$(
        impl MaybeNan for $t {
            #[inline] fn elem_is_nan(self) -> bool { self.is_nan() }
            #[inline] fn is_neg_zero(self) -> bool { self == 0.0 && self.is_sign_negative() }
            #[inline] fn is_pos_zero(self) -> bool { self == 0.0 && self.is_sign_positive() }
        }
    )*};
}
float_nan!(f32, f64);

impl MaybeNan for F16 {
    #[inline]
    fn elem_is_nan(self) -> bool {
        F16::is_nan(self)
    }
    #[inline]
    fn is_neg_zero(self) -> bool {
        self.0 == 0x8000
    }
    #[inline]
    fn is_pos_zero(self) -> bool {
        self.0 == 0
    }
}
impl MaybeNan for C32 {
    #[inline]
    fn elem_is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}
impl MaybeNan for C64v {
    #[inline]
    fn elem_is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

/// The ordered max/min of two elements, ignoring NaN and signed zero (both
/// are corrected afterwards in `extreme_contig`).
///
/// The float impls call `f64::max`/`f64::min`, which lower to a single
/// `fmaxnm`/`fminnm` instruction and therefore vectorise; the generic
/// comparison version does not.
pub trait Extreme: Copy {
    fn omax(self, o: Self) -> Self;
    fn omin(self, o: Self) -> Self;
}

macro_rules! extreme_by_cmp {
    ($($t:ty),*) => {$(
        impl Extreme for $t {
            #[inline]
            fn omax(self, o: Self) -> Self {
                if crate::ops::Cmp::c_lt(self, o) { o } else { self }
            }
            #[inline]
            fn omin(self, o: Self) -> Self {
                if crate::ops::Cmp::c_lt(o, self) { o } else { self }
            }
        }
    )*};
}
extreme_by_cmp!(i8, i16, i32, i64, u8, u16, u32, u64, crate::element::NpBool, F16, C32, C64v);

macro_rules! extreme_by_intrinsic {
    ($($t:ty),*) => {$(
        impl Extreme for $t {
            #[inline]
            fn omax(self, o: Self) -> Self { <$t>::max(self, o) }
            #[inline]
            fn omin(self, o: Self) -> Self { <$t>::min(self, o) }
        }
    )*};
}
extreme_by_intrinsic!(f32, f64);

/// The pure ordered min/max fold over a contiguous slice.
///
/// Sixteen independent accumulators: a floating-point max is not associative
/// in LLVM's eyes, so a plain fold stays scalar, and splitting the reduction
/// by hand is what lets it use vector lanes. That is safe here because
/// ordered min/max *is* associative once NaN and signed zero are handled
/// separately (see `extreme_contig`), and rayon may therefore split it too.
fn ordered_extreme<T>(s: &[T], want_max: bool) -> T
where
    T: Element + Extreme + Send + Sync,
{
    const LANES: usize = 16;
    if s.len() >= PAR_THRESHOLD {
        let mid = s.len() / 2;
        let (l, r) = s.split_at(mid);
        let (a, b) = rayon::join(
            || ordered_extreme(l, want_max),
            || ordered_extreme(r, want_max),
        );
        return if want_max { a.omax(b) } else { a.omin(b) };
    }
    let mut r: [T; LANES] = [s[0]; LANES];
    let mut it = s.chunks_exact(LANES);
    // `try_into` on a `chunks_exact` block gives the optimiser a fixed-size
    // array, which removes the bounds checks and lets it use vector lanes.
    if want_max {
        for c in &mut it {
            let c: &[T; LANES] = c.try_into().expect("chunks_exact yields LANES");
            for k in 0..LANES {
                r[k] = r[k].omax(c[k]);
            }
        }
    } else {
        for c in &mut it {
            let c: &[T; LANES] = c.try_into().expect("chunks_exact yields LANES");
            for k in 0..LANES {
                r[k] = r[k].omin(c[k]);
            }
        }
    }
    let tail = it.remainder();
    let mut acc = r[0];
    for k in 1..LANES {
        acc = if want_max { acc.omax(r[k]) } else { acc.omin(r[k]) };
    }
    for &v in tail {
        acc = if want_max { acc.omax(v) } else { acc.omin(v) };
    }
    acc
}

/// The contiguous min/max fast path: an ordered reduction plus numpy's NaN
/// and signed-zero rules, both resolved in passes that only cost anything
/// when they actually apply (and vanish entirely for integer dtypes).
///
/// # Safety
/// `p` must point at `n` contiguous, aligned, in-bounds elements of `T`.
unsafe fn extreme_contig<T>(p: *const T, n: usize, want_max: bool) -> T
where
    T: Element + Extreme + MaybeNan + Send + Sync,
{
    // SAFETY: guaranteed by the caller.
    let s: &[T] = unsafe { std::slice::from_raw_parts(p, n) };
    let acc = ordered_extreme(s, want_max);
    // A branchless fold: `any` would short-circuit, which keeps the scan
    // scalar and costs more than the reduction itself. For integer dtypes
    // `elem_is_nan` is a constant `false`, so this disappears entirely.
    let saw_nan = s.iter().fold(false, |a, &v| a | v.elem_is_nan());
    if saw_nan {
        // numpy returns the NaN itself, and the first one at that.
        for &v in s {
            if v.elem_is_nan() {
                return v;
            }
        }
    }
    // numpy reduces min/max with the SIMD fmin/fmax, which prefer `-0.0` for
    // a minimum and `+0.0` for a maximum.
    if (want_max && acc.is_neg_zero()) || (!want_max && acc.is_pos_zero()) {
        for &v in s {
            if (want_max && v.is_pos_zero()) || (!want_max && v.is_neg_zero()) {
                return v;
            }
        }
    }
    acc
}

/// Contiguous integer/bool sum: sixteen widening accumulators, split across
/// rayon above the threshold. Wrapping integer addition is associative and
/// commutative, so any split gives bit-identical results.
fn sum_int_contig<T, A, W, P>(s: &[T], widen: W, add: P, zero: A) -> A
where
    T: Copy + Send + Sync,
    A: Copy + Send + Sync,
    // Generic (not `fn` pointers): the closures must inline for the inner
    // loop to vectorise at all.
    W: Fn(T) -> A + Copy + Send + Sync,
    P: Fn(A, A) -> A + Copy + Send + Sync,
{
    const LANES: usize = 16;
    if s.len() >= PAR_THRESHOLD {
        let (l, r) = s.split_at(s.len() / 2);
        let (a, b) = rayon::join(
            || sum_int_contig(l, widen, add, zero),
            || sum_int_contig(r, widen, add, zero),
        );
        return add(a, b);
    }
    let mut acc = [zero; LANES];
    let mut it = s.chunks_exact(LANES);
    for c in &mut it {
        let c: &[T; LANES] = c.try_into().expect("chunks_exact yields LANES");
        for k in 0..LANES {
            acc[k] = add(acc[k], widen(c[k]));
        }
    }
    let mut total = zero;
    for a in acc {
        total = add(total, a);
    }
    for &v in it.remainder() {
        total = add(total, widen(v));
    }
    total
}

/// Reduce one strided run with min/max/argmin/argmax.
///
/// Returns the extreme value and the index at which it was found.
///
/// # Safety
/// As `sum_run`.
unsafe fn extreme_run(
    arr: &NdArray,
    off: isize,
    n: usize,
    stride: isize,
    op: ReduceOp,
    seed: Option<(Scalar, usize)>,
) -> (Scalar, usize) {
    // SAFETY: guaranteed by the caller.
    let p = unsafe { arr.buffer.as_ptr().offset(off) };
    let want_max = matches!(op, ReduceOp::Max | ReduceOp::ArgMax);
    // min/max over a contiguous run needs no index, so it can take the
    // vectorisable path.
    if !op.is_arg()
        && seed.is_none()
        && stride == arr.itemsize() as isize
        && n > 0
        && (p as usize) % arr.dtype.alignment().max(1) == 0
    {
        #[allow(unused_assignments)]
        let mut out = Scalar::Int(0);
        crate::dispatch_dtype!(arr.dtype, T, {
            // SAFETY: the caller guarantees `n` contiguous in-bounds elements.
            out = unsafe { extreme_contig::<T>(p as *const T, n, want_max) }.to_scalar();
        });
        return (out, 0);
    }
    let mut best = seed;
    crate::dispatch_dtype!(arr.dtype, T, {
        let mut cur: Option<(T, usize)> = best.map(|(s, i)| (T::from_scalar(s), i));
        for i in 0..n {
            // SAFETY: in-bounds by the caller's contract.
            let v: T =
                unsafe { std::ptr::read_unaligned(p.offset(i as isize * stride) as *const T) };
            match cur {
                None => cur = Some((v, i)),
                Some((b, _)) => {
                    if b.elem_is_nan() {
                        // numpy stops at the first NaN: it is already the
                        // answer for both min and max.
                        break;
                    }
                    // `argmin`/`argmax` use numpy's strict `!(v <= best)` /
                    // `!(v >= best)` tests, which keep the *first* extreme
                    // and make a NaN win. `min`/`max` instead follow the
                    // SIMD `fmin`/`fmax` numpy reduces with, which prefer
                    // `+0.0` for a maximum and `-0.0` for a minimum.
                    let take = if op.is_arg() {
                        if want_max {
                            !crate::ops::Cmp::c_le(v, b)
                        } else {
                            !crate::ops::Cmp::c_le(b, v) || v.elem_is_nan()
                        }
                    } else if want_max {
                        v.elem_is_nan()
                            || !crate::ops::Cmp::c_le(v, b)
                            || (b.is_neg_zero() && v.is_pos_zero())
                    } else {
                        v.elem_is_nan()
                            || !crate::ops::Cmp::c_le(b, v)
                            || (b.is_pos_zero() && v.is_neg_zero())
                    };
                    if take {
                        cur = Some((v, i));
                    }
                }
            }
        }
        best = cur.map(|(v, i)| (v.to_scalar(), i));
    });
    best.expect("extreme_run over an empty run with no seed")
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

fn check_numeric(arr: &NdArray, op: ReduceOp) -> Result<()> {
    if !arr.dtype.is_numeric() {
        return Err(Error::NotImplemented(format!(
            "{} is not implemented for dtype {}",
            op.name(),
            arr.dtype
        )));
    }
    Ok(())
}

/// Iterate the array as a sequence of `(offset, len, stride)` runs along its
/// innermost axis, collapsing to one run when the array is contiguous.
fn runs(arr: &NdArray) -> Vec<(isize, usize, isize)> {
    if arr.ndim() == 0 {
        return vec![(arr.byte_offset, 1, arr.itemsize() as isize)];
    }
    if arr.flags.c_contiguous {
        return vec![(arr.byte_offset, arr.size(), arr.itemsize() as isize)];
    }
    let last = arr.ndim() - 1;
    let inner = arr.shape[last].max(0) as usize;
    let stride = arr.strides[last];
    let outer_shape: Vec<isize> = arr.shape[..last].to_vec();
    let outer_strides: Vec<isize> = arr.strides[..last].to_vec();
    crate::iter::offsets(&outer_shape, &outer_strides, arr.byte_offset)
        .map(|o| (o, inner, stride))
        .collect()
}

/// Reduce the whole array (`axis=None`).
pub fn reduce_all(arr: &NdArray, op: ReduceOp) -> Result<Scalar> {
    check_numeric(arr, op)?;
    let n = arr.size();
    if n == 0 {
        if op.needs_element() {
            return Err(Error::ValueError(format!(
                "zero-size array to reduction operation {} which has no identity",
                op.name()
            )));
        }
        return Ok(identity_scalar(op, reduce_dtype(op, arr.dtype)));
    }
    match op {
        ReduceOp::Sum => {
            let mut acc = Acc::zero(reduce_dtype(op, arr.dtype));
            for (off, len, stride) in runs(arr) {
                if len == 0 {
                    continue;
                }
                // SAFETY: `runs` only yields in-bounds runs of this array.
                acc = acc.add(unsafe { sum_run(arr, off, len, stride) });
            }
            Ok(acc.to_scalar())
        }
        ReduceOp::Prod => {
            let out_dt = reduce_dtype(op, arr.dtype);
            let mut acc = Scalar::Int(1).cast(out_dt);
            crate::dispatch_dtype!(out_dt, A, {
                let mut a = A::from_scalar(acc);
                for off in crate::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset) {
                    a = a.a_mul(A::from_scalar(arr.read_at(off)));
                }
                acc = a.to_scalar();
            });
            Ok(acc)
        }
        ReduceOp::Min | ReduceOp::Max | ReduceOp::ArgMin | ReduceOp::ArgMax => {
            let mut best: Option<(Scalar, usize)> = None;
            let mut base = 0usize;
            for (off, len, stride) in runs(arr) {
                if len == 0 {
                    continue;
                }
                // Run-local indices are shifted into the flat C-order index
                // space by `base`.
                // SAFETY: `runs` only yields in-bounds runs of this array.
                let (v, i) = unsafe { extreme_run(arr, off, len, stride, op, None) };
                best = match best {
                    None => Some((v, base + i)),
                    Some((b, bi)) => {
                        let better = better_of(op, b, v);
                        if better {
                            Some((v, base + i))
                        } else {
                            Some((b, bi))
                        }
                    }
                };
                base += len;
            }
            let (v, i) = best.expect("non-empty array");
            Ok(if op.is_arg() {
                Scalar::Int(i as i64)
            } else {
                v
            })
        }
    }
}

/// Would `candidate` replace `current` for this op? Mirrors numpy's `!(v <= b)`.
fn better_of(op: ReduceOp, current: Scalar, candidate: Scalar) -> bool {
    let want_max = matches!(op, ReduceOp::Max | ReduceOp::ArgMax);
    let nan = |s: Scalar| match s {
        Scalar::Float(f) => f.is_nan(),
        Scalar::Complex(c) => c.re.is_nan() || c.im.is_nan(),
        _ => false,
    };
    if nan(current) {
        return false;
    }
    if nan(candidate) {
        return true;
    }
    let le = |a: Scalar, b: Scalar| -> bool {
        match (a, b) {
            (Scalar::Int(x), Scalar::Int(y)) => x <= y,
            (Scalar::Uint(x), Scalar::Uint(y)) => x <= y,
            (Scalar::Bool(x), Scalar::Bool(y)) => !x | y,
            (Scalar::Float(x), Scalar::Float(y)) => x <= y,
            (Scalar::Complex(x), Scalar::Complex(y)) => {
                x.re < y.re || (x.re == y.re && x.im <= y.im)
            }
            _ => false,
        }
    };
    if want_max {
        !le(candidate, current)
    } else {
        !le(current, candidate)
    }
}

fn identity_scalar(op: ReduceOp, dt: DType) -> Scalar {
    match op {
        ReduceOp::Sum => Scalar::Int(0).cast(dt),
        ReduceOp::Prod => Scalar::Int(1).cast(dt),
        _ => Scalar::Int(0).cast(dt),
    }
}

/// Reduce along a single axis.
pub fn reduce_axis(arr: &NdArray, axis: usize, op: ReduceOp, keepdims: bool) -> Result<NdArray> {
    check_numeric(arr, op)?;
    if axis >= arr.ndim() {
        return Err(Error::AxisError(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis,
            arr.ndim()
        )));
    }
    let n = arr.shape[axis].max(0) as usize;
    if n == 0 && op.needs_element() {
        return Err(Error::ValueError(format!(
            "zero-size array to reduction operation {} which has no identity",
            op.name()
        )));
    }
    let out_dt = reduce_dtype(op, arr.dtype);
    let mut out_shape: Vec<isize> = arr.shape.clone();
    if keepdims {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }
    let out = NdArray::zeros(out_shape, out_dt)?;

    // The view of `arr` with `axis` removed gives one output element per
    // position; the reduction runs along `arr.strides[axis]`.
    #[allow(clippy::needless_late_init)]
    let mut rest_shape: Vec<isize> = arr.shape.clone();
    let mut rest_strides: Vec<isize> = arr.strides.clone();
    rest_shape.remove(axis);
    rest_strides.remove(axis);
    let axis_stride = arr.strides[axis];

    // numpy runs its pairwise inner loop whenever the reduction axis ends up
    // innermost in the iterator: either it already is, or every other axis is
    // degenerate so the iterator collapses onto it.
    let rest: usize = rest_shape.iter().map(|&d| d.max(0) as usize).product();
    let innermost = axis + 1 == arr.ndim() || rest == 1;
    let out_offsets: Vec<isize> =
        crate::iter::offsets(&out.shape, &out.strides, out.byte_offset).collect();
    let src_offsets: Vec<isize> =
        crate::iter::offsets(&rest_shape, &rest_strides, arr.byte_offset).collect();

    for (k, &src_off) in src_offsets.iter().enumerate() {
        let dst = out_offsets[k];
        let value = match op {
            ReduceOp::Sum if innermost || n == 0 => {
                if n == 0 {
                    identity_scalar(op, out_dt)
                } else {
                    // SAFETY: the run lies inside `arr`.
                    Acc::zero(out_dt)
                        .add(unsafe { sum_run(arr, src_off, n, axis_stride) })
                        .to_scalar()
                }
            }
            ReduceOp::Sum => {
                // An outer axis: numpy accumulates whole slices in order, so
                // the grouping is strictly sequential.
                let mut acc = Acc::zero(out_dt);
                for i in 0..n {
                    let off = src_off + i as isize * axis_stride;
                    // SAFETY: single in-bounds element.
                    acc = acc.add(unsafe { sum_run(arr, off, 1, axis_stride) });
                }
                acc.to_scalar()
            }
            ReduceOp::Prod => {
                let mut acc = Scalar::Int(1).cast(out_dt);
                crate::dispatch_dtype!(out_dt, A, {
                    let mut a = A::from_scalar(acc);
                    for i in 0..n {
                        let off = src_off + i as isize * axis_stride;
                        a = a.a_mul(A::from_scalar(arr.read_at(off)));
                    }
                    acc = a.to_scalar();
                });
                acc
            }
            _ => {
                // SAFETY: the run lies inside `arr`.
                let (v, i) = unsafe { extreme_run(arr, src_off, n, axis_stride, op, None) };
                if op.is_arg() {
                    Scalar::Int(i as i64)
                } else {
                    v
                }
            }
        };
        out.write_at(dst, value);
    }
    Ok(out)
}

/// `np.mean`: numpy sums in `float64` for exact inputs and in the operand's
/// own float type otherwise, then divides by the count.
pub fn mean_dtype(src: DType) -> DType {
    if src.is_exact() {
        DType::F64
    } else {
        src
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f64], dt: DType) -> NdArray {
        let s: Vec<Scalar> = v.iter().map(|&x| Scalar::Float(x)).collect();
        NdArray::from_scalars(&s, dt).unwrap()
    }

    #[test]
    fn sum_of_small_arrays() {
        let a = arr(&[1.0, 2.0, 3.0], DType::F64);
        assert_eq!(reduce_all(&a, ReduceOp::Sum).unwrap(), Scalar::Float(6.0));
        assert_eq!(reduce_all(&a, ReduceOp::Prod).unwrap(), Scalar::Float(6.0));
        assert_eq!(reduce_all(&a, ReduceOp::Min).unwrap(), Scalar::Float(1.0));
        assert_eq!(reduce_all(&a, ReduceOp::Max).unwrap(), Scalar::Float(3.0));
        assert_eq!(reduce_all(&a, ReduceOp::ArgMin).unwrap(), Scalar::Int(0));
        assert_eq!(reduce_all(&a, ReduceOp::ArgMax).unwrap(), Scalar::Int(2));
    }

    #[test]
    fn integer_sums_widen_like_numpy() {
        let a = NdArray::from_scalars(
            &[Scalar::Int(100), Scalar::Int(100)],
            DType::I8,
        )
        .unwrap();
        // np.sum(np.array([100, 100], np.int8)) == 200 (int64), no overflow.
        assert_eq!(reduce_dtype(ReduceOp::Sum, DType::I8), DType::I64);
        assert_eq!(reduce_all(&a, ReduceOp::Sum).unwrap(), Scalar::Int(200));
        assert_eq!(reduce_dtype(ReduceOp::Sum, DType::U8), DType::U64);
        assert_eq!(reduce_dtype(ReduceOp::Sum, DType::Bool), DType::I64);
        assert_eq!(reduce_dtype(ReduceOp::Sum, DType::F32), DType::F32);
        assert_eq!(reduce_dtype(ReduceOp::Max, DType::I8), DType::I8);
        assert_eq!(reduce_dtype(ReduceOp::ArgMax, DType::F32), DType::I64);
    }

    #[test]
    fn pairwise_grouping_matches_the_reference() {
        // A hand-run of numpy's algorithm for n = 9: r[0..7] seeded from the
        // first 8 elements, then the tail added to the tree sum.
        let v: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let a = arr(&v, DType::F64);
        let want = {
            let r: Vec<f64> = v[..8].to_vec();
            let tree = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            0.0 + (tree + v[8])
        };
        match reduce_all(&a, ReduceOp::Sum).unwrap() {
            Scalar::Float(f) => assert_eq!(f.to_bits(), want.to_bits()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn signed_zero_extremes_match_numpy() {
        // np.array([-0., 0.]).max() is +0.0 and .min() is -0.0, in either
        // order, because numpy reduces with the SIMD fmin/fmax.
        for v in [[-0.0, 0.0], [0.0, -0.0]] {
            let a = arr(&v, DType::F64);
            match reduce_all(&a, ReduceOp::Max).unwrap() {
                Scalar::Float(f) => assert!(f == 0.0 && f.is_sign_positive()),
                other => panic!("{other:?}"),
            }
            match reduce_all(&a, ReduceOp::Min).unwrap() {
                Scalar::Float(f) => assert!(f == 0.0 && f.is_sign_negative()),
                other => panic!("{other:?}"),
            }
            // argmin/argmax still keep the first element.
            assert_eq!(reduce_all(&a, ReduceOp::ArgMax).unwrap(), Scalar::Int(0));
            assert_eq!(reduce_all(&a, ReduceOp::ArgMin).unwrap(), Scalar::Int(0));
        }
    }

    #[test]
    fn nan_propagates_like_numpy() {
        let a = arr(&[1.0, f64::NAN, 3.0], DType::F64);
        match reduce_all(&a, ReduceOp::Max).unwrap() {
            Scalar::Float(f) => assert!(f.is_nan()),
            other => panic!("{other:?}"),
        }
        // np.argmax stops at the first NaN.
        assert_eq!(reduce_all(&a, ReduceOp::ArgMax).unwrap(), Scalar::Int(1));
        assert_eq!(reduce_all(&a, ReduceOp::ArgMin).unwrap(), Scalar::Int(1));
    }

    #[test]
    fn axis_reductions() {
        let a = NdArray::arange(0.0, 6.0, 1.0, DType::F64)
            .unwrap()
            .reshape(&[2, 3])
            .unwrap();
        // np.sum(a, axis=0) -> [3., 5., 7.]
        let s0 = reduce_axis(&a, 0, ReduceOp::Sum, false).unwrap();
        assert_eq!(s0.shape, vec![3]);
        assert_eq!(
            s0.to_vec(),
            vec![Scalar::Float(3.0), Scalar::Float(5.0), Scalar::Float(7.0)]
        );
        // np.sum(a, axis=1) -> [3., 12.]
        let s1 = reduce_axis(&a, 1, ReduceOp::Sum, false).unwrap();
        assert_eq!(s1.to_vec(), vec![Scalar::Float(3.0), Scalar::Float(12.0)]);
        // keepdims
        let k = reduce_axis(&a, 1, ReduceOp::Sum, true).unwrap();
        assert_eq!(k.shape, vec![2, 1]);
        // argmax along axis 0
        let am = reduce_axis(&a, 0, ReduceOp::ArgMax, false).unwrap();
        assert_eq!(am.dtype, DType::I64);
        assert_eq!(am.to_vec(), vec![Scalar::Int(1); 3]);
    }

    #[test]
    fn empty_reductions() {
        let e = NdArray::zeros(vec![0], DType::F64).unwrap();
        assert_eq!(reduce_all(&e, ReduceOp::Sum).unwrap(), Scalar::Float(0.0));
        assert_eq!(reduce_all(&e, ReduceOp::Prod).unwrap(), Scalar::Float(1.0));
        assert!(reduce_all(&e, ReduceOp::Max).is_err());
    }
}
