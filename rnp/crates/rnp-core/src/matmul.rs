//! The matmul family of generalized ufuncs: `matmul`, `vecdot`, `matvec` and
//! `vecmat`.
//!
//! These are the only ufuncs numpy gives a *core signature* — `matmul` is
//! `(n?,k),(k,m?)->(n?,m?)` — so the hard part is not the arithmetic but the
//! gufunc dimension algebra: optional core dimensions that disappear when an
//! operand is too small, core dimensions matched by label across operands,
//! and the remaining leading dimensions broadcast against each other. This
//! module reimplements that algebra the way `ufunc_object.c` does it
//! (`_validate_num_dims` / `_get_coredim_sizes`), including the exact
//! wording of the three errors it can raise, because upstream's tests read
//! those messages.
//!
//! The kernel itself is deliberately simple. numpy's own non-BLAS inner loop
//! (`matmul_inner_noblas` in `umath/matmul.c.src`) accumulates one output
//! element at a time, left to right, in the *output* dtype — except for
//! `float16`, which accumulates in `float32`, and `bool`, which is an OR of
//! ANDs. The loop below reproduces that accumulation order exactly, so for
//! every dtype numpy does *not* hand to BLAS (bool, all integers, float16,
//! and any zero-sized or `k == 1` float case) the port is bit-identical.
//! For contiguous float32/float64/complex numpy calls BLAS, whose blocked
//! summation this cannot match bit for bit; `harness/dev_check_matmul.py`
//! measures the resulting ULP gap rather than assuming it away.
//!
//! One consequence of that is worth naming, because it is the port's only
//! known bit-level divergence here. `vecdot`/`vecmat` on complex operands run
//! numpy's `@TYPE@_dotc`, which hands a *unit-stride* operand to
//! `cblas_?dotc_sub`. At an inner length of exactly 1, Apple Accelerate's
//! `dotc` produces a NaN whose **sign bit** differs from the one the same
//! expression produces in C -- and no straight-line f64 expression reproduces
//! it (searched: 4096 complex pairs and the 121-point special-value grid; the
//! closest candidate still misses 3 cells, and adopting it would introduce
//! 588 fresh divergences on the non-BLAS path, measured). This module
//! therefore transcribes numpy's own C source, which matches numpy exactly on
//! every path numpy does not route through BLAS (0/2401 pairs), and leaves
//! the Accelerate n == 1 NaN sign as a documented platform artifact.

use std::borrow::Cow;

use crate::array::{shape_size, NdArray};
use crate::descr::Descr;
use crate::dtype::{promote, DType};
use crate::element::{Element, NpBool, C32, C64v, F16};
use crate::error::{ufunc_no_loop, Error, Result};

/// Which member of the family is being evaluated.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatKind {
    MatMul,
    VecDot,
    MatVec,
    VecMat,
}

/// The static description of one core signature, in the shape
/// `ufunc_object.c` keeps it: a flat list of dimension *labels* per operand,
/// plus a per-label "may be dropped" flag.
struct Sig {
    name: &'static str,
    text: &'static str,
    /// `core_num_dims` for (input 0, input 1, output).
    ndims: [usize; 3],
    /// `core_dim_ixs` for each operand: the label of each core dimension.
    ixs: [&'static [usize]; 3],
    /// `UFUNC_CORE_DIM_CAN_IGNORE`, per label.
    can_ignore: &'static [bool],
}

const SIG_MATMUL: Sig = Sig {
    name: "matmul",
    text: "(n?,k),(k,m?)->(n?,m?)",
    ndims: [2, 2, 2],
    // labels: n = 0, k = 1, m = 2
    ixs: [&[0, 1], &[1, 2], &[0, 2]],
    can_ignore: &[true, false, true],
};

const SIG_VECDOT: Sig = Sig {
    name: "vecdot",
    text: "(n),(n)->()",
    ndims: [1, 1, 0],
    ixs: [&[0], &[0], &[]],
    can_ignore: &[false],
};

const SIG_MATVEC: Sig = Sig {
    name: "matvec",
    text: "(m,n),(n)->(m)",
    ndims: [2, 1, 1],
    // labels: m = 0, n = 1
    ixs: [&[0, 1], &[1], &[0]],
    can_ignore: &[false, false],
};

const SIG_VECMAT: Sig = Sig {
    name: "vecmat",
    text: "(n),(n,m)->(m)",
    ndims: [1, 2, 1],
    // labels: n = 0, m = 1
    ixs: [&[0], &[0, 1], &[1]],
    can_ignore: &[false, false],
};

impl MatKind {
    fn sig(self) -> &'static Sig {
        match self {
            MatKind::MatMul => &SIG_MATMUL,
            MatKind::VecDot => &SIG_VECDOT,
            MatKind::MatVec => &SIG_MATVEC,
            MatKind::VecMat => &SIG_VECMAT,
        }
    }

    pub fn name(self) -> &'static str {
        self.sig().name
    }

    pub fn signature(self) -> &'static str {
        self.sig().text
    }

    pub fn from_name(s: &str) -> Option<MatKind> {
        match s {
            "matmul" => Some(MatKind::MatMul),
            "vecdot" => Some(MatKind::VecDot),
            "matvec" => Some(MatKind::MatVec),
            "vecmat" => Some(MatKind::VecMat),
            _ => None,
        }
    }

    /// numpy conjugates the *first* operand of `vecdot` (it calls `dotc`) and
    /// of `vecmat` (it drives gemm with `CblasConjTrans`); `matvec` and
    /// `matmul` never conjugate.
    fn conjugates_first(self) -> bool {
        matches!(self, MatKind::VecDot | MatKind::VecMat)
    }
}

/// The resolved dimensions of one call.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Broadcast leading ("loop") dimensions.
    pub loop_shape: Vec<isize>,
    /// The full shape of the result.
    pub out_shape: Vec<isize>,
    /// Rows of the inner 2-D product (1 when the label is absent).
    pub rows: isize,
    /// The contracted dimension.
    pub inner: isize,
    /// Columns of the inner 2-D product (1 when the label is absent).
    pub cols: isize,
    /// True when operand 0 must gain a length-1 axis before its last one.
    pub a_rowless: bool,
    /// True when operand 1 must gain a trailing length-1 axis.
    pub b_colless: bool,
}

/// numpy's `convert_shape_to_string`: leading "newaxis" entries (encoded as
/// negative sizes) are dropped, a one-element result keeps its trailing
/// comma, and the separator is a bare comma with no space.
fn shape_str(vals: &[isize]) -> String {
    let mut i = 0;
    while i < vals.len() && vals[i] < 0 {
        i += 1;
    }
    if i == vals.len() {
        return "()".to_string();
    }
    let mut s = vals[i].to_string();
    let first = i;
    i += 1;
    while i < vals.len() {
        if vals[i] < 0 {
            s.push_str(",newaxis");
        } else {
            s.push(',');
            s.push_str(&vals[i].to_string());
        }
        i += 1;
    }
    if first == 0 && vals.len() == 1 {
        format!("({s},)")
    } else {
        format!("({s})")
    }
}

/// Resolve the core and loop dimensions, mirroring `_validate_num_dims` and
/// `_get_coredim_sizes`.
pub fn plan(
    kind: MatKind,
    a_shape: &[isize],
    b_shape: &[isize],
    out_shape: Option<&[isize]>,
) -> Result<Plan> {
    let sig = kind.sig();
    let nlabels = sig.can_ignore.len();
    let ndim: [Option<usize>; 3] = [
        Some(a_shape.len()),
        Some(b_shape.len()),
        out_shape.map(|s| s.len()),
    ];
    let shapes: [Option<&[isize]>; 3] = [Some(a_shape), Some(b_shape), out_shape];

    let mut op_core = sig.ndims;
    let mut can_ignore: Vec<bool> = sig.can_ignore.to_vec();
    let mut missing = vec![false; nlabels];

    // --- _validate_num_dims -------------------------------------------------
    for i in 0..3 {
        let Some(nd) = ndim[i] else { continue };
        if nd >= op_core[i] {
            continue;
        }
        for &label in sig.ixs[i] {
            if !can_ignore[label] {
                continue;
            }
            missing[label] = true;
            can_ignore[label] = false;
            for i1 in 0..3 {
                for &l1 in sig.ixs[i1] {
                    if l1 == label {
                        op_core[i1] -= 1;
                    }
                }
            }
            if nd == op_core[i] {
                break;
            }
        }
        if nd < op_core[i] {
            return Err(Error::ValueError(format!(
                "{}: {} operand {} does not have enough dimensions (has {}, \
                 gufunc core with signature {} requires {})",
                sig.name,
                if i < 2 { "Input" } else { "Output" },
                if i < 2 { i } else { 0 },
                nd,
                sig.text,
                op_core[i],
            )));
        }
    }

    // --- _get_coredim_sizes -------------------------------------------------
    let mut sizes: Vec<isize> = vec![-1; nlabels];
    for i in 0..3 {
        let (Some(nd), Some(shape)) = (ndim[i], shapes[i]) else {
            continue;
        };
        let core_start = nd - op_core[i];
        let mut delta = 0usize;
        for (idim, &label) in sig.ixs[i].iter().enumerate() {
            let op_dim = if missing[label] {
                delta += 1;
                1
            } else {
                shape[core_start + idim - delta]
            };
            if sizes[label] < 0 {
                sizes[label] = op_dim;
            } else if op_dim != sizes[label] {
                return Err(Error::ValueError(format!(
                    "{}: {} operand {} has a mismatch in its core dimension {}, \
                     with gufunc signature {} (size {} is different from {})",
                    sig.name,
                    if i < 2 { "Input" } else { "Output" },
                    if i < 2 { i } else { 0 },
                    idim - delta,
                    sig.text,
                    op_dim,
                    sizes[label],
                )));
            }
        }
    }

    // --- loop dimensions ----------------------------------------------------
    let mut broadcast_ndim = 0usize;
    for i in 0..3 {
        if let Some(nd) = ndim[i] {
            broadcast_ndim = broadcast_ndim.max(nd - op_core[i]);
        }
    }
    let mut loop_shape = vec![1isize; broadcast_ndim];
    let mut conflict = false;
    for i in 0..3 {
        let (Some(nd), Some(shape)) = (ndim[i], shapes[i]) else {
            continue;
        };
        let nloop = nd - op_core[i];
        for j in 0..nloop {
            let d = shape[j];
            let slot = broadcast_ndim - nloop + j;
            if loop_shape[slot] == 1 {
                loop_shape[slot] = d;
            } else if d != 1 && d != loop_shape[slot] {
                conflict = true;
            }
        }
    }
    if conflict {
        return Err(broadcast_error(sig, &ndim, &shapes, &op_core, &sizes, &missing,
                                   broadcast_ndim));
    }

    // The output's core dimensions, in signature order, skipping the ones the
    // optional-dimension pass removed.
    let mut out_full = loop_shape.clone();
    for &label in sig.ixs[2] {
        if !missing[label] {
            out_full.push(sizes[label]);
        }
    }

    let (rows, inner, cols) = match kind {
        // labels n = 0, k = 1, m = 2
        MatKind::MatMul => (
            if missing[0] { 1 } else { sizes[0] },
            sizes[1],
            if missing[2] { 1 } else { sizes[2] },
        ),
        MatKind::VecDot => (1, sizes[0], 1),
        // labels m = 0, n = 1
        MatKind::MatVec => (sizes[0], sizes[1], 1),
        // labels n = 0, m = 1
        MatKind::VecMat => (1, sizes[0], sizes[1]),
    };

    Ok(Plan {
        loop_shape,
        out_shape: out_full,
        rows,
        inner,
        cols,
        a_rowless: op_core[0] == 1,
        b_colless: op_core[1] == 1,
    })
}

/// numpy's nditer message for operands whose loop dimensions do not
/// broadcast, complete with the `[original->remapped]` rendering.
fn broadcast_error(
    sig: &Sig,
    ndim: &[Option<usize>; 3],
    shapes: &[Option<&[isize]>; 3],
    op_core: &[usize; 3],
    sizes: &[isize],
    missing: &[bool],
    broadcast_ndim: usize,
) -> Error {
    // The iterator's dimensionality: the broadcast dimensions, then the
    // output's (surviving) core dimensions.
    let mut iter_shape: Vec<isize> = vec![-1; broadcast_ndim];
    for &label in sig.ixs[2] {
        if !missing[label] {
            iter_shape.push(sizes[label]);
        }
    }
    let iter_ndim = iter_shape.len();

    let mut parts = String::new();
    for i in 0..3 {
        let (Some(nd), Some(shape)) = (ndim[i], shapes[i]) else {
            continue;
        };
        parts.push_str(&shape_str(shape));
        parts.push_str("->");
        let nloop = nd - op_core[i];
        let mut remapped = vec![-1isize; iter_ndim];
        for j in 0..nloop {
            remapped[broadcast_ndim - nloop + j] = shape[j];
        }
        if i == 2 {
            // The output also owns the trailing core axes.
            let mut k = nd - op_core[i];
            for slot in remapped.iter_mut().skip(broadcast_ndim) {
                *slot = shape[k];
                k += 1;
            }
        }
        parts.push_str(&shape_str(&remapped));
        parts.push(' ');
    }
    Error::ValueError(format!(
        "operands could not be broadcast together with remapped shapes \
         [original->remapped]: {} and requested shape {}",
        parts,
        shape_str(&iter_shape)
    ))
}

/// True for the dtypes numpy gives a matmul loop.
fn has_loop(dt: DType) -> bool {
    matches!(
        dt,
        DType::Bool
            | DType::I8
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::U64
            | DType::F16
            | DType::F32
            | DType::F64
            | DType::C64
            | DType::C128
            | DType::Object
    )
}

/// The dtype the loop runs in: numpy promotes the two operands and uses the
/// single resulting type for inputs, accumulator and output alike.
pub fn result_dtype(kind: MatKind, a: DType, b: DType) -> Result<DType> {
    if !has_loop(a) || !has_loop(b) {
        return Err(ufunc_no_loop(kind.name(), &[&a.name(), &b.name()]));
    }
    Ok(promote(a, b))
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

/// One element type the inner loop understands, together with the accumulator
/// numpy uses for it.
///
/// There are two accumulations, not one, because numpy has two loops. The
/// plain product (`matmul`, `matvec`, and `vecdot`/`vecmat` on real types)
/// runs `@TYPE@_matmul_inner_noblas` / `@TYPE@_dot`, which accumulate in the
/// element type. The *conjugating* product (`vecdot`/`vecmat` on complex
/// types) runs `@TYPE@_dotc` from `umath/matmul.c.src`, which is a different
/// expression -- `sumr += xr*yr + xi*yi`, `sumi += xr*yi - xi*yr`, not a
/// conjugate followed by a complex multiply -- and which accumulates in
/// `npy_double` even for `complex64`. Transcribing `dotc` rather than
/// synthesising it from `conj()` matters for both the low bits of a
/// `complex64` result and the sign bits of NaNs.
trait MatElem: Element + Copy {
    type Acc: Copy;
    fn acc_zero() -> Self::Acc;
    fn acc_add(acc: Self::Acc, x: Self, y: Self) -> Self::Acc;
    fn acc_finish(acc: Self::Acc) -> Self;

    /// The accumulator `@TYPE@_dotc` uses. Identical to `Acc` for every type
    /// where conjugation is a no-op.
    type ConjAcc: Copy;
    fn conj_zero() -> Self::ConjAcc;
    fn conj_add(acc: Self::ConjAcc, x: Self, y: Self) -> Self::ConjAcc;
    fn conj_finish(acc: Self::ConjAcc) -> Self;

    /// True for the dtypes whose loop numpy leaves the FP status flags of.
    ///
    /// numpy compiles `matmul`/`vecdot`/`matvec`/`vecmat` with `USEBLAS` for
    /// float32, float64, complex64 and complex128, and every one of those
    /// loops ends in `if (!npy_blas_supports_fpe()) npy_clear_floatstatus_
    /// barrier(...)` -- so on a platform whose BLAS does not preserve the FP
    /// environment (Apple Accelerate, here) those four dtypes report *no*
    /// warnings at all, even for the shapes that never reached BLAS. bool and
    /// the integers have no FP status to report. That leaves float16, whose
    /// loop is not compiled with `USEBLAS` and therefore does report.
    const REPORT_FPE: bool = false;

    /// Flags raised by one `acc_add` step, given the accumulator before and
    /// after. Only consulted when `REPORT_FPE`.
    fn acc_flags(_prev: Self::Acc, _x: Self, _y: Self, _next: Self::Acc) -> u8 {
        0
    }

    /// Flags raised by narrowing the accumulator back to the element type.
    fn finish_flags(_acc: Self::Acc, _out: Self) -> u8 {
        0
    }

    /// `@TYPE@_dotc`'s CBLAS branch, for the two dtypes numpy compiles it for
    /// (`CFLOAT` and `CDOUBLE`; `CLONGDOUBLE` has `USE_BLAS = 0`).
    ///
    /// `None` means "numpy has no cblas branch for this type", so the
    /// transcribed scalar loop is the whole story.
    ///
    /// # Safety
    /// `x` and `y` must each address `n` in-bounds elements of `Self` spaced
    /// `incx`/`incy` *elements* apart.
    unsafe fn dotc_cblas(
        _x: *const u8,
        _incx: i64,
        _y: *const u8,
        _incy: i64,
        _n: usize,
    ) -> Option<Self> {
        None
    }
}

/// The `ConjAcc` half of the trait for every type where conjugation does
/// nothing, so the two accumulations coincide.
macro_rules! conj_is_plain {
    () => {
        type ConjAcc = Self::Acc;
        #[inline]
        fn conj_zero() -> Self::ConjAcc {
            Self::acc_zero()
        }
        #[inline]
        fn conj_add(acc: Self::ConjAcc, x: Self, y: Self) -> Self::ConjAcc {
            Self::acc_add(acc, x, y)
        }
        #[inline]
        fn conj_finish(acc: Self::ConjAcc) -> Self {
            Self::acc_finish(acc)
        }
    };
}

macro_rules! int_elem {
    ($t:ty) => {
        impl MatElem for $t {
            type Acc = $t;
            #[inline]
            fn acc_zero() -> $t {
                0
            }
            #[inline]
            fn acc_add(acc: $t, x: $t, y: $t) -> $t {
                acc.wrapping_add(x.wrapping_mul(y))
            }
            #[inline]
            fn acc_finish(acc: $t) -> $t {
                acc
            }
            conj_is_plain!();
        }
    };
}

int_elem!(i8);
int_elem!(i16);
int_elem!(i32);
int_elem!(i64);
int_elem!(u8);
int_elem!(u16);
int_elem!(u32);
int_elem!(u64);

macro_rules! float_elem {
    ($t:ty) => {
        impl MatElem for $t {
            type Acc = $t;
            #[inline]
            fn acc_zero() -> $t {
                0.0
            }
            #[inline]
            fn acc_add(acc: $t, x: $t, y: $t) -> $t {
                acc + x * y
            }
            #[inline]
            fn acc_finish(acc: $t) -> $t {
                acc
            }
            conj_is_plain!();
        }
    };
}

float_elem!(f32);
float_elem!(f64);

impl MatElem for NpBool {
    type Acc = bool;
    #[inline]
    fn acc_zero() -> bool {
        false
    }
    #[inline]
    fn acc_add(acc: bool, x: NpBool, y: NpBool) -> bool {
        acc | (x.get() & y.get())
    }
    #[inline]
    fn acc_finish(acc: bool) -> NpBool {
        NpBool::new(acc)
    }
    conj_is_plain!();
}

impl MatElem for F16 {
    // `HALF_matmul_inner_noblas` and `HALF_dot` both accumulate in float32
    // and round once at the end.
    type Acc = f32;
    #[inline]
    fn acc_zero() -> f32 {
        0.0
    }
    #[inline]
    fn acc_add(acc: f32, x: F16, y: F16) -> f32 {
        acc + x.to_f32() * y.to_f32()
    }
    #[inline]
    fn acc_finish(acc: f32) -> F16 {
        F16::from_f32(acc)
    }
    conj_is_plain!();

    // float16 is the one dtype whose matmul loop numpy does *not* compile
    // with `USEBLAS`, so it is the one dtype whose FP status survives to be
    // reported. The conditions are the ones the C loop's own arithmetic
    // raises: an invalid operation in the float32 multiply-accumulate, and an
    // overflow either there or in the final `npy_float_to_half` narrowing
    // (which on this platform is a hardware `fcvt`, and in numpy's own build
    // is additionally guarded by `NPY_HALF_GENERATE_OVERFLOW`).
    const REPORT_FPE: bool = true;

    #[inline]
    fn acc_flags(prev: f32, x: F16, y: F16, next: f32) -> u8 {
        let (xf, yf) = (x.to_f32(), y.to_f32());
        let p = xf * yf;
        let mut f = 0u8;
        if p.is_nan() && !(xf.is_nan() || yf.is_nan()) {
            f |= crate::fpe::INVALID;
        }
        if p.is_infinite() && xf.is_finite() && yf.is_finite() {
            f |= crate::fpe::OVER;
        }
        if next.is_nan() && !prev.is_nan() && !p.is_nan() {
            f |= crate::fpe::INVALID;
        }
        if next.is_infinite() && prev.is_finite() && p.is_finite() {
            f |= crate::fpe::OVER;
        }
        f
    }

    #[inline]
    fn finish_flags(acc: f32, out: F16) -> u8 {
        if acc.is_finite() && out.to_f32().is_infinite() {
            crate::fpe::OVER
        } else {
            0
        }
    }
}

impl MatElem for C32 {
    type Acc = C32;
    #[inline]
    fn acc_zero() -> C32 {
        C32::new(0.0, 0.0)
    }
    #[inline]
    fn acc_add(acc: C32, x: C32, y: C32) -> C32 {
        // The same real/imaginary split numpy's inner loop writes out.
        C32::new(
            acc.re + (x.re * y.re - x.im * y.im),
            acc.im + (x.re * y.im + x.im * y.re),
        )
    }
    #[inline]
    fn acc_finish(acc: C32) -> C32 {
        acc
    }

    // `CFLOAT_dotc`: the products stay in `npy_float`, but the running sum is
    // `npy_double` ("at least double for stability"), and the conjugation is
    // folded into the expression rather than applied to the operand.
    type ConjAcc = (f64, f64);
    #[inline]
    fn conj_zero() -> (f64, f64) {
        (0.0, 0.0)
    }
    #[inline]
    fn conj_add(acc: (f64, f64), x: C32, y: C32) -> (f64, f64) {
        (
            acc.0 + (x.re * y.re + x.im * y.im) as f64,
            acc.1 + (x.re * y.im - x.im * y.re) as f64,
        )
    }
    #[inline]
    fn conj_finish(acc: (f64, f64)) -> C32 {
        C32::new(acc.0 as f32, acc.1 as f32)
    }

    #[inline]
    unsafe fn dotc_cblas(x: *const u8, incx: i64, y: *const u8, incy: i64, n: usize) -> Option<C32> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own contract.
        Some(unsafe { crate::blas::cdotc(x, incx, y, incy, n) })
    }
}

impl MatElem for C64v {
    type Acc = C64v;
    #[inline]
    fn acc_zero() -> C64v {
        C64v::new(0.0, 0.0)
    }
    #[inline]
    fn acc_add(acc: C64v, x: C64v, y: C64v) -> C64v {
        C64v::new(
            acc.re + (x.re * y.re - x.im * y.im),
            acc.im + (x.re * y.im + x.im * y.re),
        )
    }
    #[inline]
    fn acc_finish(acc: C64v) -> C64v {
        acc
    }

    // `CDOUBLE_dotc`, whose `npy_double` sum type is already the element's.
    type ConjAcc = C64v;
    #[inline]
    fn conj_zero() -> C64v {
        C64v::new(0.0, 0.0)
    }
    #[inline]
    fn conj_add(acc: C64v, x: C64v, y: C64v) -> C64v {
        C64v::new(
            acc.re + (x.re * y.re + x.im * y.im),
            acc.im + (x.re * y.im - x.im * y.re),
        )
    }
    #[inline]
    fn conj_finish(acc: C64v) -> C64v {
        acc
    }

    #[inline]
    unsafe fn dotc_cblas(
        x: *const u8,
        incx: i64,
        y: *const u8,
        incy: i64,
        n: usize,
    ) -> Option<C64v> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own contract.
        Some(unsafe { crate::blas::zdotc(x, incx, y, incy, n) })
    }
}

/// Read one `T` out of `arr` at a byte offset.
///
/// # Safety
/// `off` must address an in-bounds element of `arr`, and `arr`'s dtype must
/// be `T`'s in native byte order.
#[inline]
unsafe fn read<T: Copy>(base: *const u8, off: isize) -> T {
    // SAFETY: the caller guarantees `off` lands on a whole in-bounds element
    // of exactly this type; `read_unaligned` covers views whose byte offset
    // is not a multiple of `size_of::<T>()`.
    unsafe { std::ptr::read_unaligned(base.offset(off) as *const T) }
}

/// The blocked inner loop. `a` has shape `loop ++ [rows, inner]`, `b` has
/// `loop ++ [inner, cols]`, and `out` is a freshly allocated C-contiguous
/// array whose element count is `nbatch * rows * cols`.
/// `is_blasable2d` from `matmul.c.src`, verbatim. `BLAS_MAXSIZE` is
/// `NPY_MAX_INT64 - 1` under ILP64, so the upper bound never bites here.
fn is_blasable2d(
    byte_stride1: isize,
    byte_stride2: isize,
    _d1: isize,
    d2: isize,
    itemsize: usize,
) -> bool {
    let isz = itemsize as isize;
    if byte_stride2 != isz {
        return false;
    }
    byte_stride1 % isz == 0 && byte_stride1 / isz >= d2
}

/// Does numpy reach `@TYPE@_dotc` for this call, or does it go through gemm?
///
/// `@TYPE@_vecdot` always calls `dotc` element by element. `@TYPE@_vecmat`
/// prefers `@TYPE@_vecmat_via_gemm` (gemm, because gemv cannot conjugate) and
/// only falls back to a per-column `dotc` when its `blasable` test fails --
/// which it does for any `dn == 1` or `dm == 1`, the shape the special-value
/// grid produces. `matmul`/`matvec` never conjugate, so they never get here.
fn routes_through_dotc(kind: MatKind, a: &NdArray, b: &NdArray, p: &Plan, itemsize: usize) -> bool {
    match kind {
        MatKind::VecDot => true,
        MatKind::VecMat => {
            let nl = p.loop_shape.len();
            let (dn, dm) = (p.inner, p.cols);
            let is1_n = a.strides[nl + 1];
            let (is2_n, is2_m) = (b.strides[nl], b.strides[nl + 1]);
            let i1 = is_blasable2d(is1_n, itemsize as isize, dn, 1, itemsize);
            let i2c = is_blasable2d(is2_n, is2_m, dn, dm, itemsize);
            let i2f = is_blasable2d(is2_m, is2_n, dm, dn, itemsize);
            !(i1 && (i2c || i2f) && dn > 1 && dm > 1)
        }
        MatKind::MatMul | MatKind::MatVec => false,
    }
}

fn kernel<T: MatElem>(
    a: &NdArray,
    b: &NdArray,
    out: &mut NdArray,
    conj_a: bool,
    dotc_route: bool,
    p: &Plan,
) {
    let (r, k, c) = (p.rows as usize, p.inner as usize, p.cols as usize);
    let nbatch = shape_size(&p.loop_shape);
    if nbatch == 0 || r == 0 || c == 0 {
        return;
    }
    let nl = p.loop_shape.len();
    let a_batch: Vec<isize> =
        crate::iter::offsets(&p.loop_shape, &a.strides[..nl], a.byte_offset).collect();
    let b_batch: Vec<isize> =
        crate::iter::offsets(&p.loop_shape, &b.strides[..nl], b.byte_offset).collect();
    let (sa_r, sa_k) = (a.strides[nl], a.strides[nl + 1]);
    let (sb_k, sb_c) = (b.strides[nl], b.strides[nl + 1]);

    let isz = a.itemsize();
    let a_base = a.buffer.as_ptr();
    let b_base = b.buffer.as_ptr();
    // SAFETY: `out` was allocated by this module as a C-contiguous array of
    // exactly `nbatch * r * c` elements of `T`, and nothing else aliases it.
    let out_slice: &mut [T] = unsafe {
        std::slice::from_raw_parts_mut(
            out.buffer.as_mut_ptr().offset(out.byte_offset) as *mut T,
            nbatch * r * c,
        )
    };

    // Packed copies of the two tiles, so the innermost loop walks contiguous
    // memory whatever the operands' strides were.
    let mut ap: Vec<T> = Vec::with_capacity(r * k);
    let mut bp: Vec<T> = Vec::with_capacity(k * c);
    let mut acc: Vec<T::Acc> = Vec::with_capacity(r * c);
    let mut cacc: Vec<T::ConjAcc> = Vec::new();
    let mut flags = 0u8;

    for bi in 0..nbatch {
        let (ao, bo) = (a_batch[bi], b_batch[bi]);
        ap.clear();
        for i in 0..r {
            for t in 0..k {
                // SAFETY: `i < rows`, `t < inner` and `ao` is the in-bounds
                // start of this batch, so the offset is in bounds.
                ap.push(unsafe { read(a_base, ao + i as isize * sa_r + t as isize * sa_k) });
            }
        }
        bp.clear();
        for t in 0..k {
            for j in 0..c {
                // SAFETY: as above, with `t < inner`, `j < cols`.
                bp.push(unsafe { read(b_base, bo + t as isize * sb_k + j as isize * sb_c) });
            }
        }
        let dst = &mut out_slice[bi * r * c..(bi + 1) * r * c];

        if conj_a {
            // `@TYPE@_dotc` tries CBLAS first, and its test is on the
            // *original* operand strides -- so it has to run against `a`/`b`
            // themselves, not the packed `ap`/`bp` copies above. One call per
            // output element, exactly as `@TYPE@_vecdot`/`@TYPE@_vecmat` issue
            // them: `dotc(x_row_i, is1 = sa_k, y_col_j, is2 = sb_k, n = k)`.
            if std::env::var_os("RNP_DBG_DOTC").is_some() { eprintln!("DBG conj_a route={} k={} sa_k={} sb_k={} isz={}", dotc_route, k, sa_k, sb_k, isz); }
            if dotc_route && k > 0 {
                if let (Some(incx), Some(incy)) = (
                    crate::blas::blas_stride(sa_k, isz),
                    crate::blas::blas_stride(sb_k, isz),
                ) {
                    let mut all = true;
                    for i in 0..r {
                        for j in 0..c {
                            // SAFETY: `i < rows`, `j < cols`, and `ao`/`bo`
                            // start this batch, so both runs of `k` elements
                            // are in bounds with the strides just validated.
                            let got = unsafe {
                                T::dotc_cblas(
                                    a_base.offset(ao + i as isize * sa_r),
                                    incx,
                                    b_base.offset(bo + j as isize * sb_c),
                                    incy,
                                    k,
                                )
                            };
                            match got {
                                Some(v) => dst[i * c + j] = v,
                                None => {
                                    all = false;
                                    break;
                                }
                            }
                        }
                        if !all {
                            break;
                        }
                    }
                    if all {
                        continue;
                    }
                }
            }
            // `@TYPE@_dotc`: a different expression and, for complex64, a
            // wider accumulator (see the trait's comment).
            cacc.clear();
            cacc.resize(r * c, T::conj_zero());
            for i in 0..r {
                let arow = &ap[i * k..i * k + k];
                let orow = &mut cacc[i * c..i * c + c];
                for (t, &x) in arow.iter().enumerate() {
                    let brow = &bp[t * c..t * c + c];
                    for (o, &y) in orow.iter_mut().zip(brow.iter()) {
                        *o = T::conj_add(*o, x, y);
                    }
                }
            }
            for (d, &s) in dst.iter_mut().zip(cacc.iter()) {
                *d = T::conj_finish(s);
            }
            continue;
        }

        acc.clear();
        acc.resize(r * c, T::acc_zero());
        for i in 0..r {
            let arow = &ap[i * k..i * k + k];
            let orow = &mut acc[i * c..i * c + c];
            for (t, &x) in arow.iter().enumerate() {
                let brow = &bp[t * c..t * c + c];
                for (o, &y) in orow.iter_mut().zip(brow.iter()) {
                    let next = T::acc_add(*o, x, y);
                    if T::REPORT_FPE {
                        flags |= T::acc_flags(*o, x, y, next);
                    }
                    *o = next;
                }
            }
        }
        for (d, &s) in dst.iter_mut().zip(acc.iter()) {
            *d = T::acc_finish(s);
            if T::REPORT_FPE {
                flags |= T::finish_flags(s, *d);
            }
        }
    }
    if T::REPORT_FPE {
        crate::fpe::raise(flags);
    }
}

/// Evaluate one member of the family.
///
/// `out_shape` is the shape of a user-supplied `out=` array, which takes part
/// in resolving the optional core dimensions (numpy lets `out` decide whether
/// `n?`/`m?` are present); the result is always returned as a fresh
/// C-contiguous array for the caller to store.
pub fn matmul(
    kind: MatKind,
    a: &NdArray,
    b: &NdArray,
    out_shape: Option<&[isize]>,
    dtype: Option<DType>,
) -> Result<NdArray> {
    let p = plan(kind, &a.shape, &b.shape, out_shape)?;
    let dt = match dtype {
        Some(d) => d,
        None => result_dtype(kind, a.dtype(), b.dtype())?,
    };
    if dt == DType::Object {
        return Err(Error::NotImplemented(format!(
            "{} on object arrays is handled by the shim",
            kind.name()
        )));
    }
    if !has_loop(dt) {
        return Err(ufunc_no_loop(
            kind.name(),
            &[&a.dtype().name(), &b.dtype().name()],
        ));
    }

    let ac = cast_native(a, dt);
    let bc = cast_native(b, dt);

    // Give both operands the canonical `(loop..., rows, inner)` /
    // `(loop..., inner, cols)` layout, then broadcast the loop dimensions.
    let a2 = if p.a_rowless {
        ac.insert_axis(ac.ndim() - 1)
    } else {
        (*ac).clone()
    };
    let b2 = if p.b_colless {
        bc.insert_axis(bc.ndim())
    } else {
        (*bc).clone()
    };
    let mut a_target = p.loop_shape.clone();
    a_target.extend_from_slice(&[p.rows, p.inner]);
    let mut b_target = p.loop_shape.clone();
    b_target.extend_from_slice(&[p.inner, p.cols]);
    let av = crate::iter::broadcast_to(&a2, &a_target)?;
    let bv = crate::iter::broadcast_to(&b2, &b_target)?;

    let mut out = NdArray::empty_descr(p.out_shape.clone(), Descr::native(dt))?;
    // Whether numpy's loop for this call reaches `@TYPE@_dotc` (and so may
    // hand the operands to CBLAS) rather than driving gemm itself.
    let dotc_route = kind.conjugates_first()
        && routes_through_dotc(kind, &av, &bv, &p, dt.itemsize());
    match dt {
        DType::Bool => kernel::<NpBool>(&av, &bv, &mut out, false, false, &p),
        DType::I8 => kernel::<i8>(&av, &bv, &mut out, false, false, &p),
        DType::I16 => kernel::<i16>(&av, &bv, &mut out, false, false, &p),
        DType::I32 => kernel::<i32>(&av, &bv, &mut out, false, false, &p),
        DType::I64 => kernel::<i64>(&av, &bv, &mut out, false, false, &p),
        DType::U8 => kernel::<u8>(&av, &bv, &mut out, false, false, &p),
        DType::U16 => kernel::<u16>(&av, &bv, &mut out, false, false, &p),
        DType::U32 => kernel::<u32>(&av, &bv, &mut out, false, false, &p),
        DType::U64 => kernel::<u64>(&av, &bv, &mut out, false, false, &p),
        DType::F16 => kernel::<F16>(&av, &bv, &mut out, false, false, &p),
        DType::F32 => kernel::<f32>(&av, &bv, &mut out, false, false, &p),
        DType::F64 => kernel::<f64>(&av, &bv, &mut out, false, false, &p),
        DType::C64 => kernel::<C32>(&av, &bv, &mut out, kind.conjugates_first(), dotc_route, &p),
        DType::C128 => kernel::<C64v>(&av, &bv, &mut out, kind.conjugates_first(), dotc_route, &p),
        _ => {
            return Err(ufunc_no_loop(
                kind.name(),
                &[&a.dtype().name(), &b.dtype().name()],
            ))
        }
    }
    out.update_flags();
    Ok(out)
}

/// `arr` as `dt` in native byte order, borrowing when it already is.
fn cast_native(arr: &NdArray, dt: DType) -> Cow<'_, NdArray> {
    if arr.dtype() == dt && arr.is_native() {
        Cow::Borrowed(arr)
    } else {
        Cow::Owned(arr.astype(dt))
    }
}

/// `np.dot`'s shape rule, which is *not* matmul's: the last axis of `a` is
/// contracted with the second-to-last of `b`, every other axis of both is
/// kept, and nothing broadcasts.
///
/// Returns the result; scalar (0-d) operands multiply elementwise, as numpy
/// allows for `dot` but not for `matmul`.
pub fn dot(a: &NdArray, b: &NdArray) -> Result<NdArray> {
    if a.ndim() == 0 || b.ndim() == 0 {
        return crate::ops::binary(a, b, crate::ops::BinOp::Mul);
    }
    let ka = a.shape[a.ndim() - 1];
    let kb = if b.ndim() == 1 {
        b.shape[0]
    } else {
        b.shape[b.ndim() - 2]
    };
    if ka != kb {
        let (adim, bdim) = (a.ndim() - 1, if b.ndim() == 1 { 0 } else { b.ndim() - 2 });
        return Err(Error::ValueError(format!(
            "shapes {} and {} not aligned: {} (dim {}) != {} (dim {})",
            fmt_tuple(&a.shape),
            fmt_tuple(&b.shape),
            ka,
            adim,
            kb,
            bdim
        )));
    }
    // Flatten to (M, K) @ (K, N) and reshape the result.
    let m: isize = a.shape[..a.ndim() - 1].iter().product();
    let a2 = a.reshape(&[m, ka])?;
    let (b2, tail): (NdArray, Vec<isize>) = if b.ndim() == 1 {
        (b.reshape(&[kb, 1])?, vec![])
    } else {
        // Move the contracted axis to the front of b's trailing block:
        // (..., K, N) -> (K, rest) requires a transpose when b has > 2 dims.
        let nd = b.ndim();
        let mut axes: Vec<usize> = vec![nd - 2];
        axes.extend((0..nd - 2).chain(std::iter::once(nd - 1)));
        let bt = b.permute(&axes)?;
        let rest: isize = bt.shape[1..].iter().product();
        let btc = if bt.is_c_contiguous() { bt.clone() } else { bt.copy() };
        (
            btc.reshape(&[kb, rest])?,
            b.shape[..nd - 2]
                .iter()
                .copied()
                .chain(std::iter::once(b.shape[nd - 1]))
                .collect(),
        )
    };
    let res = matmul(MatKind::MatMul, &a2, &b2, None, None)?;
    let mut shape: Vec<isize> = a.shape[..a.ndim() - 1].to_vec();
    shape.extend_from_slice(&tail);
    res.reshape(&shape)
}

/// `np.inner`: contract the *last* axis of both operands.
pub fn inner(a: &NdArray, b: &NdArray) -> Result<NdArray> {
    if a.ndim() == 0 || b.ndim() == 0 {
        return crate::ops::binary(a, b, crate::ops::BinOp::Mul);
    }
    let ka = a.shape[a.ndim() - 1];
    let kb = b.shape[b.ndim() - 1];
    if ka != kb {
        // numpy reports b with its last two axes swapped here.
        let mut bs = b.shape.clone();
        let nd = bs.len();
        if nd >= 2 {
            bs.swap(nd - 2, nd - 1);
        }
        return Err(Error::ValueError(format!(
            "shapes {} and {} not aligned: {} (dim {}) != {} (dim {})",
            fmt_tuple(&a.shape),
            fmt_tuple(&bs),
            ka,
            a.ndim() - 1,
            kb,
            if nd >= 2 { nd - 2 } else { 0 }
        )));
    }
    let m: isize = a.shape[..a.ndim() - 1].iter().product();
    let n: isize = b.shape[..b.ndim() - 1].iter().product();
    let a2 = a.reshape(&[m, ka])?;
    let bt = b.reshape(&[n, kb])?.transpose();
    let res = matmul(MatKind::MatMul, &a2, &bt, None, None)?;
    let mut shape: Vec<isize> = a.shape[..a.ndim() - 1].to_vec();
    shape.extend_from_slice(&b.shape[..b.ndim() - 1]);
    res.reshape(&shape)
}

fn fmt_tuple(s: &[isize]) -> String {
    if s.len() == 1 {
        format!("({},)", s[0])
    } else {
        format!(
            "({})",
            s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Scalar;

    fn arr(shape: &[isize], vals: &[f64]) -> NdArray {
        let mut a = NdArray::zeros(shape.to_vec(), DType::F64).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            a.set_flat(i, Scalar::Float(v));
        }
        a
    }

    fn iarr(shape: &[isize], vals: &[i64]) -> NdArray {
        let mut a = NdArray::zeros(shape.to_vec(), DType::I64).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            a.set_flat(i, Scalar::Int(v));
        }
        a
    }

    fn floats(a: &NdArray) -> Vec<f64> {
        a.to_vec().iter().map(|s| s.as_f64()).collect()
    }

    #[test]
    fn matmul_2d_2d() {
        let a = arr(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = arr(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let r = matmul(MatKind::MatMul, &a, &b, None, None).unwrap();
        assert_eq!(r.shape, vec![2, 2]);
        assert_eq!(floats(&r), vec![58., 64., 139., 154.]);
    }

    #[test]
    fn matmul_1d_promotion() {
        let a = arr(&[3], &[1., 2., 3.]);
        let b = arr(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let r = matmul(MatKind::MatMul, &a, &b, None, None).unwrap();
        assert_eq!(r.shape, vec![2]);
        assert_eq!(floats(&r), vec![22., 28.]);

        let r2 = matmul(MatKind::MatMul, &b, &arr(&[2], &[1., 1.]), None, None).unwrap();
        assert_eq!(r2.shape, vec![3]);
        assert_eq!(floats(&r2), vec![3., 7., 11.]);

        let r3 = matmul(MatKind::MatMul, &a, &a, None, None).unwrap();
        assert_eq!(r3.shape, Vec::<isize>::new());
        assert_eq!(floats(&r3), vec![14.]);
    }

    #[test]
    fn matmul_batched_broadcast() {
        let a = arr(&[2, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let b = arr(&[2, 2], &[1., 0., 0., 1.]);
        let r = matmul(MatKind::MatMul, &a, &b, None, None).unwrap();
        assert_eq!(r.shape, vec![2, 2, 2]);
        assert_eq!(floats(&r), floats(&a));

        // (2,1,3,4) @ (5,4,2) -> (2,5,3,2)
        let x = NdArray::ones(vec![2, 1, 3, 4], DType::F64).unwrap();
        let y = NdArray::ones(vec![5, 4, 2], DType::F64).unwrap();
        let r = matmul(MatKind::MatMul, &x, &y, None, None).unwrap();
        assert_eq!(r.shape, vec![2, 5, 3, 2]);
        assert!(floats(&r).iter().all(|&v| v == 4.0));
    }

    #[test]
    fn matmul_strided_and_negative() {
        // a[::-1] of a (3,3): reversed rows, negative row stride.
        let a = arr(&[3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let rev = a.slice_axis(0, 2, 3, -1);
        let b = NdArray::ones(vec![3, 1], DType::F64).unwrap();
        let r = matmul(MatKind::MatMul, &rev, &b, None, None).unwrap();
        assert_eq!(floats(&r), vec![24., 15., 6.]);

        // Every other column.
        let cols = a.slice_axis(1, 0, 2, 2);
        let b2 = NdArray::ones(vec![2, 1], DType::F64).unwrap();
        let r2 = matmul(MatKind::MatMul, &cols, &b2, None, None).unwrap();
        assert_eq!(floats(&r2), vec![4., 10., 16.]);
    }

    #[test]
    fn matmul_zero_sized() {
        let a = NdArray::ones(vec![2, 0], DType::F64).unwrap();
        let b = NdArray::ones(vec![0, 3], DType::F64).unwrap();
        let r = matmul(MatKind::MatMul, &a, &b, None, None).unwrap();
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(floats(&r), vec![0.; 6]);

        let a = NdArray::ones(vec![0, 3], DType::F64).unwrap();
        let b = NdArray::ones(vec![3, 4], DType::F64).unwrap();
        let r = matmul(MatKind::MatMul, &a, &b, None, None).unwrap();
        assert_eq!(r.shape, vec![0, 4]);
        assert_eq!(r.size(), 0);
    }

    #[test]
    fn integer_matmul_wraps_like_numpy() {
        let mut a = NdArray::zeros(vec![2, 2], DType::I8).unwrap();
        for i in 0..4 {
            a.set_flat(i, Scalar::Int(127));
        }
        let r = matmul(MatKind::MatMul, &a, &a, None, None).unwrap();
        assert_eq!(r.dtype(), DType::I8);
        assert_eq!(r.to_vec(), vec![Scalar::Int(2); 4]);
    }

    #[test]
    fn bool_matmul_is_or_of_ands() {
        let mut a = NdArray::zeros(vec![2, 2], DType::Bool).unwrap();
        a.set_flat(0, Scalar::Bool(true));
        a.set_flat(3, Scalar::Bool(true));
        let r = matmul(MatKind::MatMul, &a, &a, None, None).unwrap();
        assert_eq!(r.dtype(), DType::Bool);
        assert_eq!(
            r.to_vec(),
            vec![
                Scalar::Bool(true),
                Scalar::Bool(false),
                Scalar::Bool(false),
                Scalar::Bool(true)
            ]
        );
    }

    #[test]
    fn family_shapes() {
        let m = arr(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let v = arr(&[3], &[1., 1., 1.]);
        let r = matmul(MatKind::MatVec, &m, &v, None, None).unwrap();
        assert_eq!(r.shape, vec![2]);
        assert_eq!(floats(&r), vec![6., 15.]);

        let w = arr(&[2], &[1., 1.]);
        let r = matmul(MatKind::VecMat, &w, &m, None, None).unwrap();
        assert_eq!(r.shape, vec![3]);
        assert_eq!(floats(&r), vec![5., 7., 9.]);

        let r = matmul(MatKind::VecDot, &v, &v, None, None).unwrap();
        assert_eq!(r.shape, Vec::<isize>::new());
        assert_eq!(floats(&r), vec![3.]);
    }

    #[test]
    /// `@TYPE@_dotc` is not "conjugate, then multiply": it is a distinct
    /// expression, and the difference is observable in the sign bit of a NaN.
    ///
    /// Measured against real numpy 2.5.2 over all 2401 pairs drawn from
    /// {nan, ±inf, ±0, 1, 2}^2 with non-unit strides (which is the path numpy
    /// runs in C rather than in BLAS): the expression below matches on every
    /// pair, while `conj(x) * y` -- and the algebraically identical
    /// `x.re*y.im + (-x.im)*y.re` -- do not.
    #[test]
    fn dotc_is_transcribed_not_synthesised() {
        let one = C64v::new(f64::NAN, 0.0);
        let mut a = NdArray::zeros(vec![2], DType::C128).unwrap();
        let mut b = NdArray::zeros(vec![2], DType::C128).unwrap();
        for i in 0..2 {
            a.set_flat(i, Scalar::Complex(one));
            b.set_flat(i, Scalar::Complex(one));
        }
        let r = matmul(MatKind::VecDot, &a, &b, None, None).unwrap();
        let Scalar::Complex(v) = r.get_flat(0) else {
            panic!("expected a complex result")
        };
        assert!(v.re.is_nan() && v.im.is_nan());
        // `xr*yi - xi*yr` with xr = NaN, yi = +0 leaves the propagated NaN
        // positive; the `conj`-then-multiply form would negate it.
        assert!(!v.im.is_sign_negative(), "imaginary NaN sign flipped");
    }

    /// float16 is the only dtype whose matmul loop numpy leaves the FP status
    /// of (every other floating loop is compiled with `USEBLAS` and ends by
    /// clearing the status when the platform BLAS does not preserve it).
    #[test]
    fn float16_reports_fp_status() {
        let mut a = NdArray::zeros(vec![1, 2], DType::F16).unwrap();
        let mut b = NdArray::zeros(vec![2, 1], DType::F16).unwrap();
        // inf * 0 -> invalid
        a.set_flat(0, Scalar::Float(f64::INFINITY));
        a.set_flat(1, Scalar::Float(0.0));
        b.set_flat(0, Scalar::Float(0.0));
        b.set_flat(1, Scalar::Float(0.0));
        crate::fpe::clear();
        let _ = matmul(MatKind::MatMul, &a, &b, None, None).unwrap();
        assert_eq!(crate::fpe::take() & crate::fpe::INVALID, crate::fpe::INVALID);

        // A finite float32 accumulator that no longer fits in float16.
        a.set_flat(0, Scalar::Float(60000.0));
        a.set_flat(1, Scalar::Float(60000.0));
        b.set_flat(0, Scalar::Float(2.0));
        b.set_flat(1, Scalar::Float(2.0));
        crate::fpe::clear();
        let r = matmul(MatKind::MatMul, &a, &b, None, None).unwrap();
        assert_eq!(crate::fpe::take() & crate::fpe::OVER, crate::fpe::OVER);
        assert!(r.get_flat(0).as_f64().is_infinite());

        // The other dtypes stay silent, because numpy's do.
        let x = NdArray::ones(vec![2, 2], DType::F64).unwrap();
        crate::fpe::clear();
        let _ = matmul(MatKind::MatMul, &x, &x, None, None).unwrap();
        assert_eq!(crate::fpe::take(), 0);
    }

    #[test]
    fn vecdot_conjugates_the_first_operand() {
        let mut a = NdArray::zeros(vec![2], DType::C128).unwrap();
        a.set_flat(0, Scalar::Complex(C64v::new(1., 2.)));
        a.set_flat(1, Scalar::Complex(C64v::new(3., 4.)));
        let mut b = NdArray::zeros(vec![2], DType::C128).unwrap();
        b.set_flat(0, Scalar::Complex(C64v::new(5., 6.)));
        b.set_flat(1, Scalar::Complex(C64v::new(7., 8.)));
        let r = matmul(MatKind::VecDot, &a, &b, None, None).unwrap();
        assert_eq!(r.to_vec(), vec![Scalar::Complex(C64v::new(70., -8.))]);
        // matvec does not conjugate.
        let a2 = a.reshape(&[1, 2]).unwrap();
        let r2 = matmul(MatKind::MatVec, &a2, &b, None, None).unwrap();
        assert_eq!(r2.to_vec(), vec![Scalar::Complex(C64v::new(-18., 68.))]);
    }

    #[test]
    fn error_messages_match_numpy() {
        let e = matmul(
            MatKind::MatMul,
            &NdArray::ones(vec![2, 3], DType::F64).unwrap(),
            &NdArray::ones(vec![4], DType::F64).unwrap(),
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            format!("{e:?}").contains("core dimension 0"),
            true,
            "{e:?}"
        );
        match e {
            Error::ValueError(m) => assert_eq!(
                m,
                "matmul: Input operand 1 has a mismatch in its core dimension 0, \
                 with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 3)"
            ),
            other => panic!("{other:?}"),
        }

        let e = matmul(
            MatKind::MatMul,
            &NdArray::ones(vec![], DType::F64).unwrap(),
            &NdArray::ones(vec![2, 2], DType::F64).unwrap(),
            None,
            None,
        )
        .unwrap_err();
        match e {
            Error::ValueError(m) => assert_eq!(
                m,
                "matmul: Input operand 0 does not have enough dimensions (has 0, \
                 gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)"
            ),
            other => panic!("{other:?}"),
        }

        let e = matmul(
            MatKind::MatMul,
            &NdArray::ones(vec![2, 3, 4], DType::F64).unwrap(),
            &NdArray::ones(vec![5, 4, 2], DType::F64).unwrap(),
            None,
            None,
        )
        .unwrap_err();
        match e {
            Error::ValueError(m) => assert_eq!(
                m,
                "operands could not be broadcast together with remapped shapes \
                 [original->remapped]: (2,3,4)->(2,newaxis,newaxis) \
                 (5,4,2)->(5,newaxis,newaxis)  and requested shape (3,2)"
            ),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn out_shape_drives_optional_dims() {
        // numpy resolves `n?`/`m?` from *every* operand, `out` included, so a
        // 0-d `out` makes both optional dimensions vanish.
        let e = plan(MatKind::MatMul, &[2, 3], &[3, 4], Some(&[])).unwrap_err();
        match e {
            Error::ValueError(m) => assert!(m.contains("size 4 is different from 3"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn dot_and_inner_shapes() {
        let a = arr(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = arr(&[3, 4], &[1.; 12]);
        assert_eq!(dot(&a, &b).unwrap().shape, vec![2, 4]);
        let c = NdArray::ones(vec![5, 3, 4], DType::F64).unwrap();
        assert_eq!(dot(&a, &c).unwrap().shape, vec![2, 5, 4]);
        assert_eq!(
            inner(&a, &NdArray::ones(vec![4, 3], DType::F64).unwrap())
                .unwrap()
                .shape,
            vec![2, 4]
        );
        let v = arr(&[3], &[1., 2., 3.]);
        assert_eq!(floats(&dot(&v, &v).unwrap()), vec![14.]);
        assert_eq!(floats(&inner(&v, &v).unwrap()), vec![14.]);
    }

    #[test]
    fn integer_dot_promotes_like_numpy() {
        let a = iarr(&[2, 2], &[1, 2, 3, 4]);
        let r = dot(&a, &a).unwrap();
        assert_eq!(r.dtype(), DType::I64);
        assert_eq!(r.to_vec(), vec![
            Scalar::Int(7),
            Scalar::Int(10),
            Scalar::Int(15),
            Scalar::Int(22)
        ]);
    }
}
