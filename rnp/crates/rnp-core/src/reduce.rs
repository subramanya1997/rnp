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
use crate::element::{C64v, Element, Scalar, C32, F16};
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
                let v: $store = unsafe {
                    std::ptr::read_unaligned(a.offset(i as isize * stride) as *const $store)
                };
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
                unsafe { std::ptr::read_unaligned(a.offset(i as isize * stride) as *const $ftype) }
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
                let mut r: [$ftype; 8] = [re(0), im(0), re(2), im(2), re(4), im(4), re(6), im(6)];
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

    /// The accumulator numpy starts a reduction from: the identity, or the
    /// caller's `initial=` when there is one.
    fn seeded(dt: DType, seed: Option<Scalar>) -> Acc {
        let Some(s) = seed else { return Acc::zero(dt) };
        match Acc::zero(dt) {
            Acc::Int(_) => Acc::Int(match s.cast(DType::I64) {
                Scalar::Int(v) => v,
                _ => 0,
            }),
            Acc::Uint(_) => Acc::Uint(match s.cast(DType::U64) {
                Scalar::Uint(v) => v,
                _ => 0,
            }),
            Acc::F16(_) => Acc::F16(match s.cast(DType::F16) {
                Scalar::Float(v) => v as f32,
                _ => 0.0,
            }),
            Acc::F32(_) => Acc::F32(match s.cast(DType::F32) {
                Scalar::Float(v) => v as f32,
                _ => 0.0,
            }),
            Acc::F64(_) => Acc::F64(match s.cast(DType::F64) {
                Scalar::Float(v) => v,
                _ => 0.0,
            }),
            Acc::C64(_) => Acc::C64(match s.cast(DType::C64) {
                Scalar::Complex(c) => C32::new(c.re as f32, c.im as f32),
                _ => C32::new(0.0, 0.0),
            }),
            Acc::C128(_) => Acc::C128(match s.cast(DType::C128) {
                Scalar::Complex(c) => c,
                _ => C64v::new(0.0, 0.0),
            }),
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

    /// Round a running total back through the storage dtype, which is what a
    /// buffered reduce does every time it folds a buffer into its output
    /// operand. Only `float16` actually loses anything: every other
    /// accumulator is already held at the output's precision.
    fn round_to_storage(self, dt: DType) -> Acc {
        match (self, dt) {
            (Acc::F16(v), DType::F16) => Acc::F16(F16::from_f32(v).to_f32()),
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
    match arr.dtype() {
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
            if stride == d.itemsize() as isize && (p as usize) % d.alignment().max(1) == 0 {
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
                    let v: T = unsafe {
                        std::ptr::read_unaligned(p.offset(i as isize * stride) as *const T)
                    };
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
            if stride == d.itemsize() as isize && (p as usize) % d.alignment().max(1) == 0 {
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
                    let v: T = unsafe {
                        std::ptr::read_unaligned(p.offset(i as isize * stride) as *const T)
                    };
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
extreme_by_cmp!(
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    crate::element::NpBool,
    F16,
    C32,
    C64v
);

macro_rules! extreme_by_intrinsic {
    ($($t:ty),*) => {$(
        impl Extreme for $t {
            #[inline]
            fn omax(self, o: Self) -> Self {
                if cfg!(all(target_os = "linux", target_arch = "x86_64"))
                    && self == 0.0 && o == 0.0 { o } else { <$t>::max(self, o) }
            }
            #[inline]
            fn omin(self, o: Self) -> Self {
                if cfg!(all(target_os = "linux", target_arch = "x86_64"))
                    && self == 0.0 && o == 0.0 { o } else { <$t>::min(self, o) }
            }
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
fn ordered_extreme<T>(s: &[T], want_max: bool) -> (T, bool)
where
    T: Element + Extreme + MaybeNan + Send + Sync,
{
    const LANES: usize = 16;
    if s.len() >= PAR_THRESHOLD {
        let mid = s.len() / 2;
        let (l, r) = s.split_at(mid);
        let ((a, na), (b, nb)) = rayon::join(
            || ordered_extreme(l, want_max),
            || ordered_extreme(r, want_max),
        );
        return (if want_max { a.omax(b) } else { a.omin(b) }, na | nb);
    }
    let mut r: [T; LANES] = [s[0]; LANES];
    // NaN detection rides along in the same pass: min/max is memory-bound at
    // these sizes, so a second scan over the data doubled the cost. The fold
    // is branchless, and `elem_is_nan` is a constant `false` for the integer
    // dtypes, so the whole thing disappears for them.
    let mut nan = [false; LANES];
    let mut it = s.chunks_exact(LANES);
    // `try_into` on a `chunks_exact` block gives the optimiser a fixed-size
    // array, which removes the bounds checks and lets it use vector lanes.
    if want_max {
        for c in &mut it {
            let c: &[T; LANES] = c.try_into().expect("chunks_exact yields LANES");
            for k in 0..LANES {
                r[k] = r[k].omax(c[k]);
                nan[k] |= c[k].elem_is_nan();
            }
        }
    } else {
        for c in &mut it {
            let c: &[T; LANES] = c.try_into().expect("chunks_exact yields LANES");
            for k in 0..LANES {
                r[k] = r[k].omin(c[k]);
                nan[k] |= c[k].elem_is_nan();
            }
        }
    }
    let tail = it.remainder();
    let mut acc = r[0];
    let mut saw_nan = nan[0];
    for k in 1..LANES {
        acc = if want_max {
            acc.omax(r[k])
        } else {
            acc.omin(r[k])
        };
        saw_nan |= nan[k];
    }
    for &v in tail {
        acc = if want_max { acc.omax(v) } else { acc.omin(v) };
        saw_nan |= v.elem_is_nan();
    }
    (acc, saw_nan)
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
    let (acc, saw_nan) = ordered_extreme(s, want_max);
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
    if !cfg!(all(target_os = "linux", target_arch = "x86_64"))
        && ((want_max && acc.is_neg_zero()) || (!want_max && acc.is_pos_zero()))
    {
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

/// Sum one strided run under numpy's `where=` mask.
///
/// numpy does *not* reduce a masked operand by substituting the identity into
/// the pairwise tree, and it does not compress-then-reduce either. Its masked
/// strided loop (`generic_masked_strided_loop`, `array_method.c`) walks the
/// mask with `npy_memchr` and hands the *unmasked* inner loop each maximal
/// contiguous run of set mask values in turn. So the pairwise tree is rebuilt
/// per run and the run totals are folded into the accumulator sequentially,
/// which is exactly why a masked sum differs from either of the two obvious
/// models in the last bits.
///
/// `m` holds the mask value of each of the `n` elements, in run order.
///
/// # Safety
/// As `sum_run`.
unsafe fn sum_run_masked(
    arr: &NdArray,
    off: isize,
    n: usize,
    stride: isize,
    start: Acc,
    m: &[bool],
) -> Acc {
    let mut acc = start;
    let mut i = 0usize;
    while i < n {
        if !m[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && m[i] {
            i += 1;
        }
        // SAFETY: a sub-range of the caller's run, so still in bounds.
        acc = acc.add(unsafe { sum_run(arr, off + start as isize * stride, i - start, stride) });
    }
    acc
}

/// The mask values of `mask`, in C order. `mask` must already be broadcast to
/// the shape of the array being reduced, so this visits its elements in the
/// same order `runs` visits the data's.
fn mask_bits(mask: &NdArray) -> Vec<bool> {
    crate::iter::offsets(&mask.shape, &mask.strides, mask.byte_offset)
        .map(|o| !matches!(mask.read_at(o), Scalar::Bool(false) | Scalar::Int(0)))
        .collect()
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
        && (p as usize) % arr.dtype().alignment().max(1) == 0
    {
        #[allow(unused_assignments)]
        let mut out = Scalar::Int(0);
        crate::dispatch_dtype!(arr.dtype(), T, {
            // SAFETY: the caller guarantees `n` contiguous in-bounds elements.
            out = unsafe { extreme_contig::<T>(p as *const T, n, want_max) }.to_scalar();
        });
        return (out, 0);
    }
    let mut best = seed;
    crate::dispatch_dtype!(arr.dtype(), T, {
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
                            || if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
                                (b.is_neg_zero() || b.is_pos_zero())
                                    && (v.is_neg_zero() || v.is_pos_zero())
                            } else {
                                b.is_neg_zero() && v.is_pos_zero()
                            }
                    } else {
                        v.elem_is_nan()
                            || !crate::ops::Cmp::c_le(b, v)
                            || if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
                                (b.is_neg_zero() || b.is_pos_zero())
                                    && (v.is_neg_zero() || v.is_pos_zero())
                            } else {
                                b.is_pos_zero() && v.is_neg_zero()
                            }
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
    if !arr.dtype().is_numeric() {
        return Err(Error::NotImplemented(format!(
            "{} is not implemented for dtype {}",
            op.name(),
            arr.dtype()
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

/// numpy's default `NPY_BUFSIZE`: the element count one buffered `nditer`
/// pass copies out before the inner loop is called on it.
const NPY_BUFSIZE: usize = 8192;

/// numpy's iteration plan for a whole-array reduction operand.
///
/// `add.reduce` drives a **buffered** `nditer` in `NPY_KEEPORDER`, so before
/// any arithmetic happens the iterator rewrites the operand's layout:
///
/// 1. size-1 axes are removed (`npyiter_remove_axis` / the shape fixup),
/// 2. `npyiter_find_best_axis_ordering` sorts what is left so the axis with
///    the smallest absolute stride ends up innermost,
/// 3. `npyiter_coalesce_axes` merges a neighbour pair whose strides already
///    describe one continuous run.
///
/// Negative strides are *not* flipped: a read-only operand is simply walked
/// backwards through memory. Probed against real numpy on 3000 random
/// arrays (every dtype x C/F/transposed/strided/reversed layouts): the sorted
/// C-order walk of these dims is exactly the element order numpy's pairwise
/// tree sees, and `[::-1]` really does sum from the last row to the first.
///
/// Returns the coalesced `(len, stride)` dims, **outermost first**.
fn nditer_dims(arr: &NdArray) -> Vec<(usize, isize)> {
    let mut dims: Vec<(usize, isize)> = arr
        .shape
        .iter()
        .zip(arr.strides.iter())
        .filter(|&(&s, _)| s != 1)
        .map(|(&s, &st)| (s.max(0) as usize, st))
        .collect();
    if dims.is_empty() {
        return vec![(1, arr.itemsize() as isize)];
    }
    // Stable sort by descending |stride|: the smallest stride lands last, and
    // C-order iteration over the result runs it innermost.
    dims.sort_by_key(|&(_, st)| std::cmp::Reverse(st.unsigned_abs()));
    let mut out: Vec<(usize, isize)> = vec![dims[dims.len() - 1]];
    for &(s, st) in dims.iter().rev().skip(1) {
        let (cl, cst) = out[0];
        if st == cst * cl as isize {
            out[0] = (cl * s, cst);
        } else {
            out.insert(0, (s, st));
        }
    }
    out
}

/// numpy's buffered `add.reduce` over every element of `arr`.
///
/// When the iterator plan collapses to one genuinely contiguous run there is
/// nothing to buffer and numpy sums the whole block with a single pairwise
/// tree. Otherwise each pass copies up to `NPY_BUFSIZE` elements -- rounded
/// down to a whole number of innermost runs, and never fewer than one --
/// into a contiguous buffer, sums *that*, and folds the result into the
/// output operand. The fold goes through the output's own dtype, which is why
/// a multi-pass `float16` sum rounds to half between passes.
fn sum_all_nditer(arr: &NdArray, mut acc: Acc) -> Acc {
    let dims = nditer_dims(arr);
    let isz = arr.itemsize();
    let n = arr.size();
    if dims.len() == 1 && dims[0].1 == isz as isize {
        // SAFETY: one contiguous in-bounds run of `n` elements from the base.
        return acc.add(unsafe { sum_run(arr, arr.byte_offset, n, isz as isize) });
    }
    let inner = dims[dims.len() - 1].0.max(1);
    let chunk = inner.max((NPY_BUFSIZE / inner) * inner);
    let shape: Vec<isize> = dims.iter().map(|&(l, _)| l as isize).collect();
    let strides: Vec<isize> = dims.iter().map(|&(_, s)| s).collect();
    let tmp = match NdArray::zeros(vec![chunk as isize], arr.dtype()) {
        Ok(t) => t,
        // A buffer this small cannot fail to allocate; fall back to the plain
        // C-order walk rather than panicking if it somehow does.
        Err(_) => {
            for (off, len, stride) in runs(arr) {
                if len > 0 {
                    // SAFETY: `runs` only yields in-bounds runs.
                    acc = acc.add(unsafe { sum_run(arr, off, len, stride) });
                }
            }
            return acc;
        }
    };
    let mut filled = 0usize;
    for off in crate::iter::offsets(&shape, &strides, arr.byte_offset) {
        tmp.write_raw_at(
            tmp.byte_offset + (filled * isz) as isize,
            arr.raw_bytes_at(off),
        );
        filled += 1;
        if filled == chunk {
            // SAFETY: `tmp` holds `chunk` contiguous elements of this dtype.
            acc = acc
                .add(unsafe { sum_run(&tmp, tmp.byte_offset, chunk, isz as isize) })
                .round_to_storage(arr.dtype());
            filled = 0;
        }
    }
    if filled > 0 {
        // SAFETY: the first `filled` elements of `tmp` were just written.
        acc = acc
            .add(unsafe { sum_run(&tmp, tmp.byte_offset, filled, isz as isize) })
            .round_to_storage(arr.dtype());
    }
    acc
}

/// The extras numpy's `reduce` carries besides the operand and the axis.
#[derive(Copy, Clone, Default)]
pub struct ReduceOpts<'a> {
    /// numpy's `where=`, already broadcast to the operand's shape.
    ///
    /// Only the float sum path consults it: for every other op numpy's inner
    /// loop is a plain sequential fold, so skipping a masked element and
    /// folding in the identity the caller substituted for it give
    /// bit-identical results.
    pub mask: Option<&'a NdArray>,
    /// numpy's `initial=`.
    ///
    /// It *seeds the accumulator* before the first element is folded in; it
    /// is not combined with the finished result. For a reduction over a
    /// non-innermost axis, where the fold is strictly sequential, the two
    /// differ in the last bits.
    pub seed: Option<Scalar>,
}

/// Reduce the whole array (`axis=None`).
pub fn reduce_all(arr: &NdArray, op: ReduceOp) -> Result<Scalar> {
    reduce_all_with(arr, op, ReduceOpts::default())
}

/// Reduce the whole array, honouring `where=` and `initial=`.
pub fn reduce_all_with(arr: &NdArray, op: ReduceOp, opts: ReduceOpts<'_>) -> Result<Scalar> {
    if !arr.is_native() {
        return reduce_all_swapped(arr, op, opts);
    }
    reduce_all_native(arr, op, opts)
}

#[cold]
#[inline(never)]
fn reduce_all_swapped(arr: &NdArray, op: ReduceOp, opts: ReduceOpts<'_>) -> Result<Scalar> {
    reduce_all_native(&arr.to_native(), op, opts)
}

fn reduce_all_native(arr: &NdArray, op: ReduceOp, opts: ReduceOpts<'_>) -> Result<Scalar> {
    let ReduceOpts { mask, seed } = opts;
    if arr.dtype().is_datetime_like() {
        return crate::datetime_ops::reduce_all(arr, op, seed);
    }
    check_numeric(arr, op)?;
    let n = arr.size();
    if n == 0 {
        if op.needs_element() {
            return Err(Error::ValueError(format!(
                "zero-size array to reduction operation {} which has no identity",
                op.name()
            )));
        }
        let dt = reduce_dtype(op, arr.dtype());
        return Ok(seed.map_or_else(|| identity_scalar(op, dt), |s| s.cast(dt)));
    }
    match op {
        ReduceOp::Sum => {
            let acc_dt = reduce_dtype(op, arr.dtype());
            let mut acc = Acc::seeded(acc_dt, seed);
            // An inexact whole-array sum has to follow numpy's *iterator*
            // order, which is not the C order `runs` walks. Integer sums are
            // order-independent (wrapping add is associative), and the masked
            // path consumes its mask in C order, so both stay below.
            if mask.is_none() && matches!(arr.dtype().category(), Kind::Float | Kind::Complex) {
                return Ok(sum_all_nditer(arr, acc).to_scalar());
            }
            // `runs` and `mask_bits` both walk the array in C order, so the
            // mask can be consumed run by run.
            let bits = mask.map(mask_bits);
            let mut seen = 0usize;
            for (off, len, stride) in runs(arr) {
                if len == 0 {
                    continue;
                }
                acc = match &bits {
                    // SAFETY: `runs` only yields in-bounds runs of this array.
                    None => acc.add(unsafe { sum_run(arr, off, len, stride) }),
                    // SAFETY: as above; the mask slice covers exactly this run.
                    Some(b) => unsafe {
                        sum_run_masked(arr, off, len, stride, acc, &b[seen..seen + len])
                    },
                };
                seen += len;
            }
            Ok(acc.to_scalar())
        }
        ReduceOp::Prod => {
            let out_dt = reduce_dtype(op, arr.dtype());
            if out_dt == DType::F16 {
                let mut acc = seed.map_or(1.0f32, |s| s.as_f64() as f32);
                for off in crate::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset) {
                    acc *= arr.read_at(off).as_f64() as f32;
                }
                return Ok(Scalar::Float(F16::from_f32(acc).to_f64()));
            }
            let mut acc = seed.unwrap_or(Scalar::Int(1)).cast(out_dt);
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
            // `initial=` seeds the accumulator; the index that rides along is
            // never read, because `arg*` has no `initial`.
            let mut best: Option<(Scalar, usize)> = seed
                .filter(|_| !op.is_arg())
                .map(|s| (s.cast(arr.dtype()), usize::MAX));
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
    reduce_axis_with(arr, axis, op, keepdims, ReduceOpts::default())
}

/// Reduce several axes with the same single-axis kernels numpy uses.
///
/// Axes are collapsed from highest to lowest so their original numbers stay
/// valid. Reducing every axis is one whole-array collapse: for floating sums
/// this matters because numpy coalesces a contiguous operand into one
/// pairwise tree rather than chaining per-axis trees.
pub fn reduce_axes(arr: &NdArray, axes: &[usize], op: ReduceOp, keepdims: bool) -> Result<NdArray> {
    let mut removed = axes.to_vec();
    removed.sort_unstable();
    if removed.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::ValueError("duplicate value in 'axis'".into()));
    }
    if removed.iter().any(|&axis| axis >= arr.ndim()) {
        let axis = removed
            .iter()
            .copied()
            .find(|&axis| axis >= arr.ndim())
            .unwrap();
        return Err(Error::AxisError(format!(
            "axis {axis} is out of bounds for array of dimension {}",
            arr.ndim()
        )));
    }

    if removed.is_empty() {
        let dt = reduce_dtype(op, arr.dtype());
        return Ok(if dt == arr.dtype() {
            arr.copy()
        } else {
            arr.astype(dt)
        });
    }

    if removed.len() == arr.ndim() {
        let value = reduce_all(arr, op)?;
        let shape = if keepdims {
            vec![1; arr.ndim()]
        } else {
            vec![]
        };
        let out = NdArray::zeros(shape, reduce_dtype(op, arr.dtype()))?;
        out.write_at(out.byte_offset, value);
        return Ok(out);
    }

    if removed.len() == 1 {
        return reduce_axis(arr, removed[0], op, keepdims);
    }

    if op == ReduceOp::Sum {
        // `add.reduce` has pairwise inner loops. NpyIter coalesces only
        // neighbouring selected axes whose input strides form one dimension;
        // each remaining group is a distinct buffered collapse (and thus a
        // distinct rounding point for float16).
        let mut groups: Vec<(usize, usize)> = Vec::new();
        let mut start = removed[0];
        let mut end = start;
        for &axis in removed.iter().skip(1) {
            let coalesces =
                axis == end + 1 && arr.strides[end] == arr.strides[axis] * arr.shape[axis];
            if coalesces {
                end = axis;
            } else {
                groups.push((start, end));
                start = axis;
                end = axis;
            }
        }
        groups.push((start, end));

        if arr.dtype() == DType::F16 && groups.len() > 1 {
            // HALF keeps the float accumulator alive across the separately
            // iterated outer reduction groups and rounds only when storing
            // the final output cell. Each innermost group still uses numpy's
            // pairwise HALF loop.
            let (inner_start, inner_end) = *groups.last().expect("non-empty groups");
            let inner_n = arr.shape[inner_start..=inner_end]
                .iter()
                .map(|&n| n.max(0) as usize)
                .product::<usize>();
            let inner_stride = arr.strides[inner_end];
            let outer_axes: Vec<usize> = removed
                .iter()
                .copied()
                .filter(|&axis| axis < inner_start)
                .collect();
            let outer_shape: Vec<isize> = outer_axes.iter().map(|&axis| arr.shape[axis]).collect();
            let outer_strides: Vec<isize> =
                outer_axes.iter().map(|&axis| arr.strides[axis]).collect();
            let mut out_shape = Vec::with_capacity(arr.ndim() - removed.len());
            let mut rest_strides = Vec::with_capacity(arr.ndim() - removed.len());
            for axis in 0..arr.ndim() {
                if !removed.contains(&axis) {
                    out_shape.push(arr.shape[axis]);
                    rest_strides.push(arr.strides[axis]);
                }
            }
            let out = NdArray::zeros(out_shape, DType::F16)?;
            let bases = crate::iter::offsets(&out.shape, &rest_strides, arr.byte_offset);
            let dsts = crate::iter::offsets(&out.shape, &out.strides, out.byte_offset);
            for (base, dst) in bases.zip(dsts) {
                let mut acc = Acc::F16(0.0);
                for outer in crate::iter::offsets(&outer_shape, &outer_strides, base) {
                    // SAFETY: the coalesced innermost group is an in-bounds
                    // strided run beginning at this outer-group offset.
                    acc = acc
                        .add(unsafe { sum_run(arr, outer, inner_n, inner_stride) })
                        .round_to_storage(DType::F16);
                }
                out.write_at(dst, acc.to_scalar());
            }
            let mut cur = out;
            if keepdims {
                let mut shape = cur.shape.clone();
                for &axis in &removed {
                    shape.insert(axis, 1);
                }
                cur = cur.reshape(&shape)?;
            }
            return Ok(cur);
        }

        // HALF uses its float accumulator across separately buffered groups;
        // packing all selected axes reproduces that one final rounding.
        if arr.dtype() != DType::F16 || groups.len() == 1 {
            let mut cur = arr.clone();
            for &(start, end) in groups.iter().rev() {
                if start == end {
                    cur = reduce_axis(&cur, start, op, false)?;
                    continue;
                }
                let mut merged = cur.clone();
                merged.shape[start] = cur.shape[start..=end]
                    .iter()
                    .map(|&n| n.max(0) as usize)
                    .product::<usize>() as isize;
                merged.strides[start] = cur.strides[end];
                merged.shape.drain(start + 1..=end);
                merged.strides.drain(start + 1..=end);
                merged.update_flags();
                cur = reduce_axis(&merged, start, op, false)?;
            }
            if keepdims {
                let mut shape = cur.shape.clone();
                for &axis in &removed {
                    shape.insert(axis, 1);
                }
                cur = cur.reshape(&shape)?;
            }
            return Ok(cur);
        }
    }

    if op == ReduceOp::Prod && arr.dtype() == DType::F16 {
        let mut groups: Vec<(usize, usize)> = Vec::new();
        let mut start = removed[0];
        let mut end = start;
        for &axis in removed.iter().skip(1) {
            let coalesces =
                axis == end + 1 && arr.strides[end] == arr.strides[axis] * arr.shape[axis];
            if coalesces {
                end = axis;
            } else {
                groups.push((start, end));
                start = axis;
                end = axis;
            }
        }
        groups.push((start, end));

        if groups.len() > 1 {
            // The HALF multiply loop holds a float accumulator within each
            // contiguous iterator buffer, then stores (and therefore rounds)
            // it through the half output before folding the next outer run.
            let (inner_start, inner_end) = *groups.last().expect("non-empty groups");
            let inner_n = arr.shape[inner_start..=inner_end]
                .iter()
                .map(|&n| n.max(0) as usize)
                .product::<usize>();
            let inner_stride = arr.strides[inner_end];
            let outer_axes: Vec<usize> = removed
                .iter()
                .copied()
                .filter(|&axis| axis < inner_start)
                .collect();
            let outer_shape: Vec<isize> = outer_axes.iter().map(|&axis| arr.shape[axis]).collect();
            let outer_strides: Vec<isize> =
                outer_axes.iter().map(|&axis| arr.strides[axis]).collect();
            let mut out_shape = Vec::with_capacity(arr.ndim() - removed.len());
            let mut rest_strides = Vec::with_capacity(arr.ndim() - removed.len());
            for axis in 0..arr.ndim() {
                if !removed.contains(&axis) {
                    out_shape.push(arr.shape[axis]);
                    rest_strides.push(arr.strides[axis]);
                }
            }
            let out = NdArray::zeros(out_shape, DType::F16)?;
            let bases = crate::iter::offsets(&out.shape, &rest_strides, arr.byte_offset);
            let dsts = crate::iter::offsets(&out.shape, &out.strides, out.byte_offset);
            for (base, dst) in bases.zip(dsts) {
                let mut acc = 1.0f32;
                for outer in crate::iter::offsets(&outer_shape, &outer_strides, base) {
                    for i in 0..inner_n {
                        acc *= arr
                            .read_at(outer + i as isize * inner_stride)
                            .as_f64() as f32;
                    }
                    acc = F16::from_f32(acc).to_f32();
                }
                out.write_at(dst, Scalar::Float(F16::from_f32(acc).to_f64()));
            }
            let mut cur = out;
            if keepdims {
                let mut shape = cur.shape.clone();
                for &axis in &removed {
                    shape.insert(axis, 1);
                }
                cur = cur.reshape(&shape)?;
            }
            return Ok(cur);
        }

        if groups.len() == 1 {
            let (start, end) = groups[0];
            let mut merged = arr.clone();
            merged.shape[start] = arr.shape[start..=end]
                .iter()
                .map(|&n| n.max(0) as usize)
                .product::<usize>() as isize;
            merged.strides[start] = arr.strides[end];
            merged.shape.drain(start + 1..=end);
            merged.strides.drain(start + 1..=end);
            merged.update_flags();
            let mut cur = reduce_axis(&merged, start, op, false)?;
            if keepdims {
                let mut shape = cur.shape.clone();
                for &axis in &removed {
                    shape.insert(axis, 1);
                }
                cur = cur.reshape(&shape)?;
            }
            return Ok(cur);
        }
    }

    // NpyIter packs the selected reduction axes into its inner buffer in
    // C-order before invoking the ufunc loop. Pack that exact logical run for
    // each output cell, then use the already bit-exact whole-array kernel.
    let reduced_shape: Vec<isize> = removed.iter().map(|&axis| arr.shape[axis]).collect();
    let reduced_strides: Vec<isize> = removed.iter().map(|&axis| arr.strides[axis]).collect();
    let reduced_size: usize = reduced_shape.iter().map(|&n| n.max(0) as usize).product();
    let tmp = NdArray::empty(vec![reduced_size as isize], arr.dtype())?;

    let mut out_shape = Vec::with_capacity(arr.ndim() - removed.len());
    let mut rest_strides = Vec::with_capacity(arr.ndim() - removed.len());
    for axis in 0..arr.ndim() {
        if !removed.contains(&axis) {
            out_shape.push(arr.shape[axis]);
            rest_strides.push(arr.strides[axis]);
        }
    }
    let out = NdArray::zeros(out_shape, reduce_dtype(op, arr.dtype()))?;
    let src_bases = crate::iter::offsets(&out.shape, &rest_strides, arr.byte_offset);
    let dst_offsets = crate::iter::offsets(&out.shape, &out.strides, out.byte_offset);
    for (base, dst) in src_bases.zip(dst_offsets) {
        for (i, src) in crate::iter::offsets(&reduced_shape, &reduced_strides, base).enumerate() {
            tmp.write_raw_at(
                tmp.byte_offset + (i * arr.itemsize()) as isize,
                arr.raw_bytes_at(src),
            );
        }
        out.write_at(dst, reduce_all(&tmp, op)?);
    }

    let mut cur = out;
    if keepdims {
        let mut shape = cur.shape.clone();
        for &axis in &removed {
            shape.insert(axis, 1);
        }
        cur = cur.reshape(&shape)?;
    }
    Ok(cur)
}

/// Reduce along a single axis, honouring `where=` and `initial=`.
pub fn reduce_axis_with(
    arr: &NdArray,
    axis: usize,
    op: ReduceOp,
    keepdims: bool,
    opts: ReduceOpts<'_>,
) -> Result<NdArray> {
    if !arr.is_native() {
        return reduce_axis_swapped(arr, axis, op, keepdims, opts);
    }
    reduce_axis_native(arr, axis, op, keepdims, opts)
}

#[cold]
#[inline(never)]
fn reduce_axis_swapped(
    arr: &NdArray,
    axis: usize,
    op: ReduceOp,
    keepdims: bool,
    opts: ReduceOpts<'_>,
) -> Result<NdArray> {
    reduce_axis_native(&arr.to_native(), axis, op, keepdims, opts)
}

fn reduce_axis_native(
    arr: &NdArray,
    axis: usize,
    op: ReduceOp,
    keepdims: bool,
    opts: ReduceOpts<'_>,
) -> Result<NdArray> {
    let ReduceOpts { mask, seed } = opts;
    if arr.dtype().is_datetime_like() {
        return crate::datetime_ops::reduce_axis(arr, axis, op, keepdims, seed);
    }
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
    let out_dt = reduce_dtype(op, arr.dtype());
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
    // innermost after the iterator's stride-based axis ordering. Numerical
    // axis position is insufficient for transposed/F-order views.
    let rest: usize = rest_shape.iter().map(|&d| d.max(0) as usize).product();
    let axis_stride_abs = axis_stride.unsigned_abs();
    let innermost = rest == 1
        || rest_shape
            .iter()
            .zip(rest_strides.iter())
            .filter(|&(&n, _)| n > 1)
            .all(|(_, &stride)| axis_stride_abs <= stride.unsigned_abs());
    let out_offsets: Vec<isize> =
        crate::iter::offsets(&out.shape, &out.strides, out.byte_offset).collect();
    let src_offsets: Vec<isize> =
        crate::iter::offsets(&rest_shape, &rest_strides, arr.byte_offset).collect();

    // The mask, walked with the same index arithmetic but its own strides.
    // Only the innermost (pairwise) sum consults it -- for an outer axis
    // numpy's inner loop is elementwise `out[j] += in[j]`, a sequential fold
    // in which skipping a masked element and adding the identity the caller
    // substituted for it agree bit-for-bit.
    let mask_walk = mask.filter(|_| op == ReduceOp::Sum && innermost).map(|m| {
        let mut m_rest: Vec<isize> = m.strides.clone();
        let m_axis = m.strides[axis];
        m_rest.remove(axis);
        let offs: Vec<isize> = crate::iter::offsets(&rest_shape, &m_rest, m.byte_offset).collect();
        (m, offs, m_axis)
    });

    for (k, &src_off) in src_offsets.iter().enumerate() {
        let dst = out_offsets[k];
        let value = match op {
            ReduceOp::Sum if innermost || n == 0 => {
                if n == 0 {
                    seed.map_or_else(|| identity_scalar(op, out_dt), |s| s.cast(out_dt))
                } else if let Some((m, offs, m_axis)) = &mask_walk {
                    let bits: Vec<bool> = (0..n)
                        .map(|i| {
                            let o = offs[k] + i as isize * m_axis;
                            !matches!(m.read_at(o), Scalar::Bool(false) | Scalar::Int(0))
                        })
                        .collect();
                    // SAFETY: the run lies inside `arr`.
                    unsafe {
                        sum_run_masked(
                            arr,
                            src_off,
                            n,
                            axis_stride,
                            Acc::seeded(out_dt, seed),
                            &bits,
                        )
                    }
                    .to_scalar()
                } else {
                    // SAFETY: the run lies inside `arr`.
                    Acc::seeded(out_dt, seed)
                        .add(unsafe { sum_run(arr, src_off, n, axis_stride) })
                        .to_scalar()
                }
            }
            ReduceOp::Sum => {
                // An outer axis: numpy accumulates whole slices in order, so
                // the grouping is strictly sequential.
                let mut acc = Acc::seeded(out_dt, seed);
                for i in 0..n {
                    let off = src_off + i as isize * axis_stride;
                    // SAFETY: single in-bounds element.
                    acc = acc
                        .add(unsafe { sum_run(arr, off, 1, axis_stride) })
                        .round_to_storage(out_dt);
                }
                acc.to_scalar()
            }
            ReduceOp::Prod => {
            if out_dt == DType::F16 && innermost {
                    let mut acc = seed.map_or(1.0f32, |s| s.as_f64() as f32);
                    for i in 0..n {
                        let off = src_off + i as isize * axis_stride;
                        acc *= arr.read_at(off).as_f64() as f32;
                    }
                    Scalar::Float(F16::from_f32(acc).to_f64())
                } else {
                    let mut acc = seed.unwrap_or(Scalar::Int(1)).cast(out_dt);
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
            }
            _ => {
                // `initial=` seeds the accumulator, as it does for sum/prod.
                let start = seed
                    .filter(|_| !op.is_arg())
                    .map(|s| (s.cast(arr.dtype()), usize::MAX));
                // SAFETY: the run lies inside `arr`.
                let (v, i) = unsafe { extreme_run(arr, src_off, n, axis_stride, op, start) };
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
        let a = NdArray::from_scalars(&[Scalar::Int(100), Scalar::Int(100)], DType::I8).unwrap();
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
        // macOS resolves zero ties by sign; Linux x86 keeps the later value.
        for v in [[-0.0, 0.0], [0.0, -0.0]] {
            let a = arr(&v, DType::F64);
            match reduce_all(&a, ReduceOp::Max).unwrap() {
                Scalar::Float(f) => {
                    let negative = cfg!(all(target_os = "linux", target_arch = "x86_64"))
                        && v[1].is_sign_negative();
                    assert!(f == 0.0 && f.is_sign_negative() == negative)
                }
                other => panic!("{other:?}"),
            }
            match reduce_all(&a, ReduceOp::Min).unwrap() {
                Scalar::Float(f) => {
                    let negative = !cfg!(all(target_os = "linux", target_arch = "x86_64"))
                        || v[1].is_sign_negative();
                    assert!(f == 0.0 && f.is_sign_negative() == negative)
                }
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
        assert_eq!(am.dtype(), DType::I64);
        assert_eq!(am.to_vec(), vec![Scalar::Int(1); 3]);
    }

    #[test]
    fn a_masked_sum_rebuilds_the_tree_per_run() {
        // numpy's masked strided loop calls the unmasked inner loop once per
        // maximal run of set mask values, so a masked sum is
        // `((0 + pw(run0)) + pw(run1)) + ...` -- *not* one pairwise tree over
        // the whole operand with the identity substituted, which is what a
        // naive implementation produces and which differs in the last bits.
        let v: Vec<f64> = (0..12).map(|i| (i as f64 + 1.0) / 7.0).collect();
        let a = arr(&v, DType::F64);
        let keep = [
            true, true, true, false, true, true, true, true, true, true, false, true,
        ];
        let mask = NdArray::from_scalars(
            &keep.iter().map(|&b| Scalar::Bool(b)).collect::<Vec<_>>(),
            DType::Bool,
        )
        .unwrap();
        // Runs are [0,3), [4,10), [11,12); each is shorter than 8, so each
        // pairwise call is `-0.0` plus a sequential add.
        let pw = |lo: usize, hi: usize| v[lo..hi].iter().fold(-0.0f64, |r, &x| r + x);
        let want = ((0.0f64 + pw(0, 3)) + pw(4, 10)) + pw(11, 12);
        let opts = ReduceOpts {
            mask: Some(&mask),
            seed: None,
        };
        match reduce_all_with(&a, ReduceOp::Sum, opts).unwrap() {
            Scalar::Float(f) => assert_eq!(f.to_bits(), want.to_bits()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn initial_seeds_the_accumulator() {
        // numpy pre-fills the output with `initial=` and then folds, so an
        // outer-axis sum is `((init + a0) + a1) + ...`. Combining `init` with
        // the finished result instead rounds differently.
        let a = NdArray::from_scalars(
            &(0..8)
                .map(|i| Scalar::Float(1.0 / (3.0 + i as f64)))
                .collect::<Vec<_>>(),
            DType::F64,
        )
        .unwrap()
        .reshape(&[4, 2])
        .unwrap();
        let init = 1e16f64;
        let opts = ReduceOpts {
            mask: None,
            seed: Some(Scalar::Float(init)),
        };
        // axis 0 is the outer axis, so numpy's fold is strictly sequential.
        let got = reduce_axis_with(&a, 0, ReduceOp::Sum, false, opts).unwrap();
        for col in 0..2 {
            let want = (0..4).fold(init, |acc, r| acc + 1.0 / (3.0 + (2 * r + col) as f64));
            match got.to_vec()[col] {
                Scalar::Float(f) => assert_eq!(f.to_bits(), want.to_bits()),
                other => panic!("{other:?}"),
            }
        }
        // The seed is also what an all-empty reduction returns.
        let e = NdArray::zeros(vec![0], DType::F64).unwrap();
        assert_eq!(
            reduce_all_with(&e, ReduceOp::Sum, opts).unwrap(),
            Scalar::Float(init)
        );
    }

    #[test]
    fn empty_reductions() {
        let e = NdArray::zeros(vec![0], DType::F64).unwrap();
        assert_eq!(reduce_all(&e, ReduceOp::Sum).unwrap(), Scalar::Float(0.0));
        assert_eq!(reduce_all(&e, ReduceOp::Prod).unwrap(), Scalar::Float(1.0));
        assert!(reduce_all(&e, ReduceOp::Max).is_err());
    }
}
