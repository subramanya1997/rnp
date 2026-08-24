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
//! Complex `vecdot` and `vecmat` also preserve numpy's BLAS split. Positive,
//! element-aligned iterator strides reach Accelerate's ILP64 `dotc`; a
//! BLAS-layout `vecmat` reaches `gemm` with `CblasConjTrans`. Length-1 core
//! axes are a subtle exception: numpy's gufunc iterator supplies stride zero,
//! so they use the scalar loop. On AArch64 clang contracts that loop's
//! `xr*yi - xi*yr` to `fnmsub`; the helper below emits that same instruction
//! so NaN signs match too.

use std::borrow::Cow;

use crate::array::{shape_size, NdArray};
use crate::descr::Descr;
use crate::dtype::{promote, DType};
use crate::element::{C64v, Element, NpBool, C32, F16};
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
        return Err(broadcast_error(
            sig,
            &ndim,
            &shapes,
            &op_core,
            &sizes,
            &missing,
            broadcast_ndim,
        ));
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
    /// Whether numpy compiles this dtype's matmul family with `USEBLAS`.
    const HAS_BLAS: bool = false;

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

    /// `@TYPE@_dot`'s CBLAS branch. `None` means this dtype has no BLAS loop.
    ///
    /// # Safety
    /// `x` and `y` must each address `n` in-bounds elements of `Self` spaced
    /// by the positive element strides `incx` and `incy`.
    unsafe fn dot_blas(
        _x: *const u8,
        _incx: i64,
        _y: *const u8,
        _incy: i64,
        _n: usize,
    ) -> Option<Self> {
        None
    }

    /// `@TYPE@_gemv`'s CBLAS branch.
    ///
    /// # Safety
    /// The pointers must describe the complete BLAS extents encoded by the
    /// dimensions, leading dimension, and positive increments.
    unsafe fn gemv_blas(
        _matrix: *const u8,
        _matrix_col_major: bool,
        _lda: i64,
        _vector: *const u8,
        _incx: i64,
        _out: *mut u8,
        _incy: i64,
        _m: usize,
        _n: usize,
    ) -> Option<()> {
        None
    }

    /// `@TYPE@_matmul_matrixmatrix`'s CBLAS branch.
    ///
    /// # Safety
    /// The pointers must describe the complete BLAS matrix extents encoded by
    /// the dimensions, transpose flags, and leading dimensions.
    unsafe fn gemm_blas(
        _a: *const u8,
        _transpose_a: bool,
        _lda: i64,
        _b: *const u8,
        _transpose_b: bool,
        _ldb: i64,
        _out: *mut u8,
        _ldc: i64,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Option<()> {
        None
    }

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

    /// Complex `@TYPE@_vecmat_via_gemm`; `None` for every dtype without that
    /// CBLAS loop.
    ///
    /// # Safety
    /// `x` and `y` must describe the matrices encoded by `n`, `m`, `lda`, and
    /// `ldb`, and `out` must address `m` writable elements of `Self`.
    unsafe fn vecmat_cblas(
        _x: *const u8,
        _lda: i64,
        _y: *const u8,
        _ldb: i64,
        _transpose_y: bool,
        _out: *mut u8,
        _n: usize,
        _m: usize,
    ) -> Option<()> {
        None
    }
}

/// clang's contraction of `a * b - c` in numpy's complex `dotc` fallback.
/// Keeping this as one hardware operation is observable for NaN sign bits.
#[inline(always)]
fn numpy_mul_sub_f32(mut a: f32, b: f32, c: f32) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        // SAFETY: `fnmsub` is a scalar floating-point instruction with no
        // memory access or control-flow effects. Its operands and result stay
        // entirely in compiler-allocated FP registers.
        unsafe {
            std::arch::asm!(
                "fnmsub {a:s}, {a:s}, {b:s}, {c:s}",
                a = inout(vreg) a,
                b = in(vreg) b,
                c = in(vreg) c,
                options(nomem, nostack, preserves_flags),
            );
        }
        a
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        a * b - c
    }
}

#[inline(always)]
fn numpy_mul_sub_f64(mut a: f64, b: f64, c: f64) -> f64 {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        // SAFETY: as in `numpy_mul_sub_f32`; the `d` register view selects
        // the double-precision form of the same side-effect-free instruction.
        unsafe {
            std::arch::asm!(
                "fnmsub {a:d}, {a:d}, {b:d}, {c:d}",
                a = inout(vreg) a,
                b = in(vreg) b,
                c = in(vreg) c,
                options(nomem, nostack, preserves_flags),
            );
        }
        a
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        a * b - c
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
    ($t:ty, $dot:ident, $gemv:ident, $gemm:ident) => {
        impl MatElem for $t {
            const HAS_BLAS: bool = true;

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

            #[inline]
            unsafe fn dot_blas(
                x: *const u8,
                incx: i64,
                y: *const u8,
                incy: i64,
                n: usize,
            ) -> Option<Self> {
                if !crate::blas::HAVE_CBLAS {
                    return None;
                }
                // SAFETY: forwarded from this method's own strided-run
                // contract.
                Some(unsafe { crate::blas::$dot(x, incx, y, incy, n) })
            }

            #[inline]
            unsafe fn gemv_blas(
                matrix: *const u8,
                matrix_col_major: bool,
                lda: i64,
                vector: *const u8,
                incx: i64,
                out: *mut u8,
                incy: i64,
                m: usize,
                n: usize,
            ) -> Option<()> {
                if !crate::blas::HAVE_CBLAS {
                    return None;
                }
                // SAFETY: forwarded from this method's own matrix/vector
                // extent contract.
                unsafe {
                    crate::blas::$gemv(matrix, matrix_col_major, lda, vector, incx, out, incy, m, n)
                };
                Some(())
            }

            #[inline]
            unsafe fn gemm_blas(
                a: *const u8,
                transpose_a: bool,
                lda: i64,
                b: *const u8,
                transpose_b: bool,
                ldb: i64,
                out: *mut u8,
                ldc: i64,
                m: usize,
                n: usize,
                k: usize,
            ) -> Option<()> {
                if !crate::blas::HAVE_CBLAS {
                    return None;
                }
                // SAFETY: forwarded from this method's own matrix-extent
                // contract.
                unsafe {
                    crate::blas::$gemm(a, transpose_a, lda, b, transpose_b, ldb, out, ldc, m, n, k)
                };
                Some(())
            }
        }
    };
}

float_elem!(f32, sdot, sgemv, sgemm);
float_elem!(f64, ddot, dgemv, dgemm);

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
    const HAS_BLAS: bool = true;

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
            acc.1 + numpy_mul_sub_f32(x.re, y.im, x.im * y.re) as f64,
        )
    }
    #[inline]
    fn conj_finish(acc: (f64, f64)) -> C32 {
        C32::new(acc.0 as f32, acc.1 as f32)
    }

    #[inline]
    unsafe fn dot_blas(x: *const u8, incx: i64, y: *const u8, incy: i64, n: usize) -> Option<C32> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own strided-run contract.
        Some(unsafe { crate::blas::cdotu(x, incx, y, incy, n) })
    }

    #[inline]
    unsafe fn gemv_blas(
        matrix: *const u8,
        matrix_col_major: bool,
        lda: i64,
        vector: *const u8,
        incx: i64,
        out: *mut u8,
        incy: i64,
        m: usize,
        n: usize,
    ) -> Option<()> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own matrix/vector contract.
        unsafe { crate::blas::cgemv(matrix, matrix_col_major, lda, vector, incx, out, incy, m, n) };
        Some(())
    }

    #[inline]
    unsafe fn gemm_blas(
        a: *const u8,
        transpose_a: bool,
        lda: i64,
        b: *const u8,
        transpose_b: bool,
        ldb: i64,
        out: *mut u8,
        ldc: i64,
        m: usize,
        n: usize,
        k: usize,
    ) -> Option<()> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own matrix-extent contract.
        unsafe { crate::blas::cgemm(a, transpose_a, lda, b, transpose_b, ldb, out, ldc, m, n, k) };
        Some(())
    }

    #[inline]
    unsafe fn dotc_cblas(
        x: *const u8,
        incx: i64,
        y: *const u8,
        incy: i64,
        n: usize,
    ) -> Option<C32> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own contract.
        Some(unsafe { crate::blas::cdotc(x, incx, y, incy, n) })
    }

    #[inline]
    unsafe fn vecmat_cblas(
        x: *const u8,
        lda: i64,
        y: *const u8,
        ldb: i64,
        transpose_y: bool,
        out: *mut u8,
        n: usize,
        m: usize,
    ) -> Option<()> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own matrix-extent contract.
        unsafe { crate::blas::cgemm_vecmat(x, lda, y, ldb, transpose_y, out, n, m) };
        Some(())
    }
}

impl MatElem for C64v {
    const HAS_BLAS: bool = true;

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
            acc.im + numpy_mul_sub_f64(x.re, y.im, x.im * y.re),
        )
    }
    #[inline]
    fn conj_finish(acc: C64v) -> C64v {
        acc
    }

    #[inline]
    unsafe fn dot_blas(x: *const u8, incx: i64, y: *const u8, incy: i64, n: usize) -> Option<C64v> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own strided-run contract.
        Some(unsafe { crate::blas::zdotu(x, incx, y, incy, n) })
    }

    #[inline]
    unsafe fn gemv_blas(
        matrix: *const u8,
        matrix_col_major: bool,
        lda: i64,
        vector: *const u8,
        incx: i64,
        out: *mut u8,
        incy: i64,
        m: usize,
        n: usize,
    ) -> Option<()> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own matrix/vector contract.
        unsafe { crate::blas::zgemv(matrix, matrix_col_major, lda, vector, incx, out, incy, m, n) };
        Some(())
    }

    #[inline]
    unsafe fn gemm_blas(
        a: *const u8,
        transpose_a: bool,
        lda: i64,
        b: *const u8,
        transpose_b: bool,
        ldb: i64,
        out: *mut u8,
        ldc: i64,
        m: usize,
        n: usize,
        k: usize,
    ) -> Option<()> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own matrix-extent contract.
        unsafe { crate::blas::zgemm(a, transpose_a, lda, b, transpose_b, ldb, out, ldc, m, n, k) };
        Some(())
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

    #[inline]
    unsafe fn vecmat_cblas(
        x: *const u8,
        lda: i64,
        y: *const u8,
        ldb: i64,
        transpose_y: bool,
        out: *mut u8,
        n: usize,
        m: usize,
    ) -> Option<()> {
        if !crate::blas::HAVE_CBLAS {
            return None;
        }
        // SAFETY: forwarded from this method's own matrix-extent contract.
        unsafe { crate::blas::zgemm_vecmat(x, lda, y, ldb, transpose_y, out, n, m) };
        Some(())
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
    let blas_max = if isize::BITS == 64 {
        isize::MAX - 1
    } else {
        isize::MAX
    };
    if byte_stride2 != isz {
        return false;
    }
    let unit_stride1 = byte_stride1 / isz;
    byte_stride1 % isz == 0 && unit_stride1 >= d2 && unit_stride1 <= blas_max
}

/// The gufunc iterator normalises a size-1 core axis to stride zero. These are
/// the `steps[...]` values the generated loops receive and run through
/// `blas_stride`, not the source ndarray's cosmetic stride for that axis.
#[inline]
fn gufunc_core_stride(byte_stride: isize, dim: isize) -> isize {
    if dim == 1 {
        0
    } else {
        byte_stride
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ConjRoute {
    Dotc,
    Gemm {
        lda: i64,
        ldb: i64,
        transpose_y: bool,
    },
    Scalar,
}

/// Select the generated complex inner loop numpy actually registers.
fn conjugating_route(
    kind: MatKind,
    a: &NdArray,
    b: &NdArray,
    p: &Plan,
    itemsize: usize,
) -> ConjRoute {
    match kind {
        MatKind::VecDot => ConjRoute::Dotc,
        MatKind::VecMat => {
            let nl = p.loop_shape.len();
            let (dn, dm) = (p.inner, p.cols);
            let is1_n = gufunc_core_stride(a.strides[nl + 1], dn);
            let is2_n = gufunc_core_stride(b.strides[nl], dn);
            let is2_m = gufunc_core_stride(b.strides[nl + 1], dm);
            let i1 = is_blasable2d(is1_n, itemsize as isize, dn, 1, itemsize);
            let i2c = is_blasable2d(is2_n, is2_m, dn, dm, itemsize);
            let i2f = is_blasable2d(is2_m, is2_n, dm, dn, itemsize);
            let blas_max = if isize::BITS == 64 {
                isize::MAX - 1
            } else {
                isize::MAX
            };
            let too_big = dn > blas_max || dm > blas_max;
            if i1 && (i2c || i2f) && !too_big && dn > 1 && dm > 1 {
                let isz = itemsize as isize;
                ConjRoute::Gemm {
                    lda: (is1_n / isz) as i64,
                    ldb: (if i2c { is2_n } else { is2_m } / isz) as i64,
                    transpose_y: !i2c,
                }
            } else {
                ConjRoute::Dotc
            }
        }
        MatKind::MatMul | MatKind::MatVec => ConjRoute::Scalar,
    }
}

/// Run the `@TYPE@_dot` CBLAS branch for an `r`-by-`c` grid of dot products.
/// This is the fallback used by standalone `vecdot`, `matvec`, and `vecmat`,
/// and by `matmul`'s scalar-output special case.
#[allow(clippy::too_many_arguments)]
fn try_plain_dot_grid<T: MatElem>(
    a_base: *const u8,
    ao: isize,
    sa_r: isize,
    sa_k: isize,
    b_base: *const u8,
    bo: isize,
    sb_k: isize,
    sb_c: isize,
    dst: &mut [T],
    r: usize,
    k: usize,
    c: usize,
    itemsize: usize,
) -> bool {
    if k == 0 {
        return false;
    }
    let is1 = gufunc_core_stride(sa_k, k as isize);
    let is2 = gufunc_core_stride(sb_k, k as isize);
    let (Some(incx), Some(incy)) = (
        crate::blas::blas_stride(is1, itemsize),
        crate::blas::blas_stride(is2, itemsize),
    ) else {
        return false;
    };
    for i in 0..r {
        for j in 0..c {
            // SAFETY: `ao`/`bo` start this batch; the row/column offsets stay
            // inside the logical matrices, and `blas_stride` validated both
            // complete positive-stride runs of `k` elements.
            let got = unsafe {
                T::dot_blas(
                    a_base.offset(ao + i as isize * sa_r),
                    incx,
                    b_base.offset(bo + j as isize * sb_c),
                    incy,
                    k,
                )
            };
            let Some(value) = got else {
                return false;
            };
            dst[i * c + j] = value;
        }
    }
    true
}

/// Run numpy's `_gemv` wrapper after the caller has selected the same logical
/// matrix/vector orientation as the generated C loop.
#[allow(clippy::too_many_arguments)]
fn try_gemv<T: MatElem>(
    matrix_base: *const u8,
    matrix_offset: isize,
    matrix_stride_m: isize,
    matrix_stride_n: isize,
    vector_base: *const u8,
    vector_offset: isize,
    vector_stride_n: isize,
    out: *mut u8,
    out_stride_m: isize,
    m: usize,
    n: usize,
    itemsize: usize,
) -> bool {
    let c_blasable = is_blasable2d(
        matrix_stride_m,
        matrix_stride_n,
        m as isize,
        n as isize,
        itemsize,
    );
    let f_blasable = is_blasable2d(
        matrix_stride_n,
        matrix_stride_m,
        n as isize,
        m as isize,
        itemsize,
    );
    let Some(incx) = crate::blas::blas_stride(vector_stride_n, itemsize) else {
        return false;
    };
    let Some(incy) = crate::blas::blas_stride(out_stride_m, itemsize) else {
        return false;
    };
    if !(c_blasable || f_blasable) {
        return false;
    }
    let lda = if c_blasable {
        matrix_stride_m / itemsize as isize
    } else {
        matrix_stride_n / itemsize as isize
    } as i64;
    // SAFETY: the two `is_blasable2d` predicates and both `blas_stride`
    // conversions validate the full matrix/vector/output extents; offsets are
    // the in-bounds starts of this batch.
    unsafe {
        T::gemv_blas(
            matrix_base.offset(matrix_offset),
            c_blasable,
            lda,
            vector_base.offset(vector_offset),
            incx,
            out,
            incy,
            m,
            n,
        )
    }
    .is_some()
}

/// Copy one logical matrix into the C- or Fortran-contiguous temporary numpy
/// chooses when the original strides are not BLASable.
#[allow(clippy::too_many_arguments)]
fn copy_blas_matrix<T: Copy>(
    base: *const u8,
    offset: isize,
    stride_row: isize,
    stride_col: isize,
    rows: usize,
    cols: usize,
    transpose: bool,
) -> Vec<T> {
    let mut out = Vec::with_capacity(rows * cols);
    if transpose {
        for j in 0..cols {
            for i in 0..rows {
                // SAFETY: `offset` is the in-bounds matrix start and both
                // indices stay inside its logical extent.
                out.push(unsafe {
                    read(
                        base,
                        offset + i as isize * stride_row + j as isize * stride_col,
                    )
                });
            }
        }
    } else {
        for i in 0..rows {
            for j in 0..cols {
                // SAFETY: as above, in row-major temporary order.
                out.push(unsafe {
                    read(
                        base,
                        offset + i as isize * stride_row + j as isize * stride_col,
                    )
                });
            }
        }
    }
    out
}

/// Matrix-matrix BLAS routing, including numpy's exact C/F layout tests and
/// its copy orientation for non-BLASable operands.
#[allow(clippy::too_many_arguments)]
fn try_gemm<T: MatElem>(
    a_base: *const u8,
    ao: isize,
    sa_r: isize,
    sa_k: isize,
    b_base: *const u8,
    bo: isize,
    sb_k: isize,
    sb_c: isize,
    out: *mut u8,
    r: usize,
    k: usize,
    c: usize,
    itemsize: usize,
) -> bool {
    let a_c = is_blasable2d(sa_r, sa_k, r as isize, k as isize, itemsize);
    let a_f = is_blasable2d(sa_k, sa_r, k as isize, r as isize, itemsize);
    let b_c = is_blasable2d(sb_k, sb_c, k as isize, c as isize, itemsize);
    let b_f = is_blasable2d(sb_c, sb_k, c as isize, k as isize, itemsize);

    let mut a_copy: Option<Vec<T>> = None;
    let (a_ptr, transpose_a, lda) = if a_c {
        // SAFETY: `ao` is the in-bounds start of this batch.
        (
            unsafe { a_base.offset(ao) },
            false,
            (sa_r / itemsize as isize) as i64,
        )
    } else if a_f {
        // SAFETY: `ao` is the in-bounds start of this batch.
        (
            unsafe { a_base.offset(ao) },
            true,
            (sa_k / itemsize as isize) as i64,
        )
    } else {
        let transpose = sa_r.unsigned_abs() < sa_k.unsigned_abs();
        a_copy = Some(copy_blas_matrix::<T>(
            a_base, ao, sa_r, sa_k, r, k, transpose,
        ));
        let copy = a_copy.as_ref().expect("matrix copy was just initialized");
        (
            copy.as_ptr() as *const u8,
            transpose,
            if transpose { r as i64 } else { k as i64 },
        )
    };

    let mut b_copy: Option<Vec<T>> = None;
    let (b_ptr, transpose_b, ldb) = if b_c {
        // SAFETY: `bo` is the in-bounds start of this batch.
        (
            unsafe { b_base.offset(bo) },
            false,
            (sb_k / itemsize as isize) as i64,
        )
    } else if b_f {
        // SAFETY: `bo` is the in-bounds start of this batch.
        (
            unsafe { b_base.offset(bo) },
            true,
            (sb_c / itemsize as isize) as i64,
        )
    } else {
        let transpose = sb_k.unsigned_abs() < sb_c.unsigned_abs();
        b_copy = Some(copy_blas_matrix::<T>(
            b_base, bo, sb_k, sb_c, k, c, transpose,
        ));
        let copy = b_copy.as_ref().expect("matrix copy was just initialized");
        (
            copy.as_ptr() as *const u8,
            transpose,
            if transpose { k as i64 } else { c as i64 },
        )
    };

    // Keep the backing temporary matrices alive across the FFI call.
    let _keep_alive = (&a_copy, &b_copy);
    // SAFETY: direct operands passed `is_blasable2d`; copied operands are
    // densely laid out in the exact orientation represented by their
    // transpose flag and leading dimension. `out` holds `r*c` writable
    // elements in row-major order.
    unsafe {
        T::gemm_blas(
            a_ptr,
            transpose_a,
            lda,
            b_ptr,
            transpose_b,
            ldb,
            out,
            c as i64,
            r,
            c,
            k,
        )
    }
    .is_some()
}

/// Transcribe numpy's generated float/complex routing for one gufunc outer
/// iteration.
#[allow(clippy::too_many_arguments)]
fn try_plain_blas_single<T: MatElem>(
    kind: MatKind,
    a: &NdArray,
    b: &NdArray,
    p: &Plan,
    ao: isize,
    bo: isize,
    dst: &mut [T],
) -> bool {
    if !crate::blas::HAVE_CBLAS || !T::HAS_BLAS {
        return false;
    }
    let (r, k, c) = (p.rows as usize, p.inner as usize, p.cols as usize);
    let nl = p.loop_shape.len();
    let itemsize = a.itemsize();
    let is1_m = gufunc_core_stride(a.strides[nl], p.rows);
    let is1_n = gufunc_core_stride(a.strides[nl + 1], p.inner);
    let is2_n = gufunc_core_stride(b.strides[nl], p.inner);
    let is2_p = gufunc_core_stride(b.strides[nl + 1], p.cols);
    let a_base = a.buffer.as_ptr();
    let b_base = b.buffer.as_ptr();
    let out = dst.as_mut_ptr() as *mut u8;
    let blas_max = if isize::BITS == 64 {
        isize::MAX - 1
    } else {
        isize::MAX
    };

    match kind {
        MatKind::VecDot => try_plain_dot_grid(
            a_base, ao, is1_m, is1_n, b_base, bo, is2_n, is2_p, dst, r, k, c, itemsize,
        ),
        MatKind::MatVec => {
            let too_big = p.rows > blas_max || p.inner > blas_max;
            let matrix_blasable = is_blasable2d(is1_m, is1_n, p.rows, p.inner, itemsize)
                || is_blasable2d(is1_n, is1_m, p.inner, p.rows, itemsize);
            let vector_blasable = is_blasable2d(is2_n, itemsize as isize, p.inner, 1, itemsize);
            if matrix_blasable && vector_blasable && !too_big && p.inner > 1 && p.rows > 1 {
                try_gemv::<T>(
                    a_base,
                    ao,
                    is1_m,
                    is1_n,
                    b_base,
                    bo,
                    is2_n,
                    out,
                    itemsize as isize,
                    r,
                    k,
                    itemsize,
                )
            } else {
                try_plain_dot_grid(
                    a_base, ao, is1_m, is1_n, b_base, bo, is2_n, is2_p, dst, r, k, c, itemsize,
                )
            }
        }
        MatKind::VecMat => {
            let too_big = p.cols > blas_max || p.inner > blas_max;
            let vector_blasable = is_blasable2d(is1_n, itemsize as isize, p.inner, 1, itemsize);
            let matrix_blasable = is_blasable2d(is2_n, is2_p, p.inner, p.cols, itemsize)
                || is_blasable2d(is2_p, is2_n, p.cols, p.inner, itemsize);
            if vector_blasable && matrix_blasable && !too_big && p.inner > 1 && p.cols > 1 {
                try_gemv::<T>(
                    b_base,
                    bo,
                    is2_p,
                    is2_n,
                    a_base,
                    ao,
                    is1_n,
                    out,
                    itemsize as isize,
                    c,
                    k,
                    itemsize,
                )
            } else {
                try_plain_dot_grid(
                    a_base, ao, is1_m, is1_n, b_base, bo, is2_n, is2_p, dst, r, k, c, itemsize,
                )
            }
        }
        MatKind::MatMul => {
            let special_case = p.rows == 1 || p.inner == 1 || p.cols == 1;
            let any_zero = p.rows == 0 || p.inner == 0 || p.cols == 0;
            let too_big = p.rows > blas_max || p.inner > blas_max || p.cols > blas_max;
            if any_zero || too_big {
                return false;
            }
            if !special_case {
                return try_gemm::<T>(
                    a_base, ao, is1_m, is1_n, b_base, bo, is2_n, is2_p, out, r, k, c, itemsize,
                );
            }
            if p.rows == 1 && p.cols == 1 {
                return try_plain_dot_grid(
                    a_base, ao, is1_m, is1_n, b_base, bo, is2_n, is2_p, dst, r, k, c, itemsize,
                );
            }
            if p.inner == 1 && (p.cols == 1 || p.rows == 1) {
                return false;
            }
            let i1_blasable = is_blasable2d(is1_m, is1_n, p.rows, p.inner, itemsize)
                || is_blasable2d(is1_n, is1_m, p.inner, p.rows, itemsize);
            let i2_blasable = is_blasable2d(is2_n, is2_p, p.inner, p.cols, itemsize)
                || is_blasable2d(is2_p, is2_n, p.cols, p.inner, itemsize);
            let vector_matrix = p.rows == 1
                && i2_blasable
                && is_blasable2d(is1_n, itemsize as isize, p.inner, 1, itemsize);
            if vector_matrix {
                return try_gemv::<T>(
                    b_base,
                    bo,
                    is2_p,
                    is2_n,
                    a_base,
                    ao,
                    is1_n,
                    out,
                    itemsize as isize,
                    c,
                    k,
                    itemsize,
                );
            }
            let matrix_vector = p.cols == 1
                && i1_blasable
                && is_blasable2d(is2_n, itemsize as isize, p.inner, 1, itemsize);
            if matrix_vector {
                return try_gemv::<T>(
                    a_base,
                    ao,
                    is1_m,
                    is1_n,
                    b_base,
                    bo,
                    is2_n,
                    out,
                    itemsize as isize,
                    r,
                    k,
                    itemsize,
                );
            }
            false
        }
    }
}

fn kernel<T: MatElem>(
    kind: MatKind,
    a: &NdArray,
    b: &NdArray,
    out: &mut NdArray,
    conj_a: bool,
    conj_route: ConjRoute,
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

    if !conj_a {
        let mut all_blas = true;
        for bi in 0..nbatch {
            let dst = &mut out_slice[bi * r * c..(bi + 1) * r * c];
            if !try_plain_blas_single(kind, a, b, p, a_batch[bi], b_batch[bi], dst) {
                all_blas = false;
                break;
            }
        }
        if all_blas {
            return;
        }
    }

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
            if let ConjRoute::Gemm {
                lda,
                ldb,
                transpose_y,
            } = conj_route
            {
                debug_assert_eq!(r, 1);
                // SAFETY: the route is selected only after numpy's exact
                // `is_blasable2d` checks on the original operand strides.
                // `ao`/`bo` start this batch, and `dst` is `c` writable
                // contiguous elements.
                let used = unsafe {
                    T::vecmat_cblas(
                        a_base.offset(ao),
                        lda,
                        b_base.offset(bo),
                        ldb,
                        transpose_y,
                        dst.as_mut_ptr() as *mut u8,
                        k,
                        c,
                    )
                };
                if used.is_some() {
                    continue;
                }
            }

            // `@TYPE@_dotc` tests the iterator's original inner strides and
            // calls CBLAS on the original memory, not the packed copies.
            if conj_route == ConjRoute::Dotc && k > 0 {
                let is1 = gufunc_core_stride(sa_k, p.inner);
                let is2 = gufunc_core_stride(sb_k, p.inner);
                if let (Some(incx), Some(incy)) = (
                    crate::blas::blas_stride(is1, isz),
                    crate::blas::blas_stride(is2, isz),
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
    let conj_route = if kind.conjugates_first() {
        conjugating_route(kind, &av, &bv, &p, dt.itemsize())
    } else {
        ConjRoute::Scalar
    };
    match dt {
        DType::Bool => kernel::<NpBool>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::I8 => kernel::<i8>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::I16 => kernel::<i16>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::I32 => kernel::<i32>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::I64 => kernel::<i64>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::U8 => kernel::<u8>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::U16 => kernel::<u16>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::U32 => kernel::<u32>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::U64 => kernel::<u64>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::F16 => kernel::<F16>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::F32 => kernel::<f32>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::F64 => kernel::<f64>(kind, &av, &bv, &mut out, false, ConjRoute::Scalar, &p),
        DType::C64 => kernel::<C32>(
            kind,
            &av,
            &bv,
            &mut out,
            kind.conjugates_first(),
            conj_route,
            &p,
        ),
        DType::C128 => kernel::<C64v>(
            kind,
            &av,
            &bv,
            &mut out,
            kind.conjugates_first(),
            conj_route,
            &p,
        ),
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
        let btc = if bt.is_c_contiguous() {
            bt.clone()
        } else {
            bt.copy()
        };
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
            s.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
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

    #[cfg(all(target_os = "macos", target_vendor = "apple"))]
    #[test]
    fn contiguous_f64_family_matches_direct_accelerate_bits() {
        let n = 33usize;
        let av: Vec<f64> = (0..n * n)
            .map(|i| ((i * 37 % 101) as f64 - 50.0) / 7.0)
            .collect();
        let bv: Vec<f64> = (0..n * n)
            .map(|i| ((i * 53 % 97) as f64 - 48.0) / 11.0)
            .collect();
        let a = arr(&[n as isize, n as isize], &av);
        let b = arr(&[n as isize, n as isize], &bv);

        let got = matmul(MatKind::MatMul, &a, &b, None, None).unwrap();
        let got_bits = floats(&got).iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        // SAFETY: both inputs and the result are dense `n`-by-`n` matrices.
        // Reusing the result buffer also holds pointer alignment constant, as
        // Accelerate may select a different microkernel for another address.
        unsafe {
            crate::blas::dgemm(
                a.buffer.as_ptr().wrapping_offset(a.byte_offset),
                false,
                n as i64,
                b.buffer.as_ptr().wrapping_offset(b.byte_offset),
                false,
                n as i64,
                got.buffer.as_mut_ptr().offset(got.byte_offset),
                n as i64,
                n,
                n,
                n,
            );
        }
        assert_eq!(
            got_bits,
            floats(&got).iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );

        let vvals: Vec<f64> = (0..n)
            .map(|i| ((i * 29 % 43) as f64 - 21.0) / 5.0)
            .collect();
        let v = arr(&[n as isize], &vvals);
        let got_mv = matmul(MatKind::MatVec, &a, &v, None, None).unwrap();
        let got_mv_bits = floats(&got_mv)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        // SAFETY: `a` is a dense `n`-by-`n` matrix and `v`/`got_mv` are
        // dense length-`n` vectors. The output address stays unchanged.
        unsafe {
            crate::blas::dgemv(
                a.buffer.as_ptr().wrapping_offset(a.byte_offset),
                true,
                n as i64,
                v.buffer.as_ptr().wrapping_offset(v.byte_offset),
                1,
                got_mv.buffer.as_mut_ptr().offset(got_mv.byte_offset),
                1,
                n,
                n,
            );
        }
        assert_eq!(
            got_mv_bits,
            floats(&got_mv)
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );

        let got_dot = matmul(MatKind::VecDot, &v, &v, None, None).unwrap();
        // SAFETY: both arguments are the same live dense length-`n` vector.
        let expected_dot = unsafe {
            crate::blas::ddot(
                v.buffer.as_ptr().wrapping_offset(v.byte_offset),
                1,
                v.buffer.as_ptr().wrapping_offset(v.byte_offset),
                1,
                n,
            )
        };
        assert_eq!(floats(&got_dot)[0].to_bits(), expected_dot.to_bits());

        let batches = 5usize;
        let side = 8usize;
        let batch_av: Vec<f64> = (0..batches * side * side)
            .map(|i| ((i * 31 % 113) as f64 - 56.0) / 9.0)
            .collect();
        let batch_bv: Vec<f64> = (0..batches * side * side)
            .map(|i| ((i * 47 % 127) as f64 - 63.0) / 13.0)
            .collect();
        let batch_a = arr(&[batches as isize, side as isize, side as isize], &batch_av);
        let batch_b = arr(&[batches as isize, side as isize, side as isize], &batch_bv);
        let got_batch = matmul(MatKind::MatMul, &batch_a, &batch_b, None, None).unwrap();
        let got_batch_bits = floats(&got_batch)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        for bi in 0..batches {
            let byte_offset = (bi * side * side * std::mem::size_of::<f64>()) as isize;
            // SAFETY: every offset identifies a dense, disjoint `side`-by-`side`
            // tile in the two inputs and output. The output addresses are reused
            // so Accelerate sees the same alignment as the routed implementation.
            unsafe {
                crate::blas::dgemm(
                    batch_a
                        .buffer
                        .as_ptr()
                        .wrapping_offset(batch_a.byte_offset + byte_offset),
                    false,
                    side as i64,
                    batch_b
                        .buffer
                        .as_ptr()
                        .wrapping_offset(batch_b.byte_offset + byte_offset),
                    false,
                    side as i64,
                    got_batch
                        .buffer
                        .as_mut_ptr()
                        .offset(got_batch.byte_offset + byte_offset),
                    side as i64,
                    side,
                    side,
                    side,
                );
            }
        }
        assert_eq!(
            got_batch_bits,
            floats(&got_batch)
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }

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

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn length_one_complex_core_uses_numpy_fnmsub_nan_sign() {
        for dtype in [DType::C64, DType::C128] {
            let value = C64v::new(f64::NAN, 0.0);
            let mut vector = NdArray::zeros(vec![1], dtype).unwrap();
            vector.set_flat(0, Scalar::Complex(value));
            let matrix = vector.reshape(&[1, 1]).unwrap();

            for (kind, rhs) in [(MatKind::VecDot, &vector), (MatKind::VecMat, &matrix)] {
                let result = matmul(kind, &vector, rhs, None, None).unwrap();
                let Scalar::Complex(got) = result.get_flat(0) else {
                    panic!("expected a complex result")
                };
                assert!(got.re.is_nan());
                assert!(got.im.is_nan());
                assert!(got.im.is_sign_negative(), "{kind:?} {dtype:?}");
            }
        }
        assert_eq!(gufunc_core_stride(16, 1), 0);
        assert_eq!(gufunc_core_stride(-16, 2), -16);
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
        assert_eq!(
            crate::fpe::take() & crate::fpe::INVALID,
            crate::fpe::INVALID
        );

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
        assert_eq!(format!("{e:?}").contains("core dimension 0"), true, "{e:?}");
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
        assert_eq!(
            r.to_vec(),
            vec![
                Scalar::Int(7),
                Scalar::Int(10),
                Scalar::Int(15),
                Scalar::Int(22)
            ]
        );
    }
}
