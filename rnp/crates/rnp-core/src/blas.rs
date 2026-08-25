//! The CBLAS entry points numpy's `matmul.c.src` reaches for complex `vecdot`
//! and `vecmat`, bound to the same library and ABI as numpy.
//!
//! Everything else in this crate is from-scratch Rust, and deliberately so.
//! This module is the one exception, for a reason that is not a shortcut:
//! `@TYPE@_dotc` routes to `cblas_?dotc_sub` whenever both iterator-provided
//! operand strides are usable. Complex `vecmat` first tries
//! `cblas_?gemm(..., CblasConjTrans, ...)` and reaches `dotc` only when that
//! 2-D predicate fails. The iterator detail matters: a length-1 core axis has
//! stride zero even when the source ndarray reports an item-sized stride, so
//! that case stays in numpy's scalar fallback.
//!
//! On macOS, the numpy wheel in `.venv` links
//! `_cblas_zdotc_sub$NEWLAPACK$ILP64` (checked with `nm -u` on
//! `_multiarray_umath`), i.e. Accelerate's *new* LAPACK interface in its
//! ILP64 flavour, so `CBLAS_INT` is 64 bits wide. `npy_cblas.h` then makes
//! `CBLAS_INT_MAX` `NPY_MAX_INT64` and `NPY_CBLAS_CHUNK` `NPY_MAX_INTP`, which
//! is why the chunking loop in `dotc` never actually splits a call there.
//!
//! On Linux x86-64, [`initialize_linux_openblas`] must be called with the exact
//! `numpy.libs/libscipy_openblas*.so` bundled beside the imported manylinux
//! wheel. The loader resolves that wheel's LP64 `scipy_cblas_*` symbols and
//! only publishes the backend after every symbol used below is present. Until
//! then, [`have_cblas`] is false and `matmul` uses its transcribed Rust paths.

use crate::element::{C64v, C32};

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_COL_MAJOR: i32 = 102;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;
const CBLAS_CONJ_TRANS: i32 = 113;

/// numpy's `NPY_CBLAS_CHUNK` for the selected ABI. Accelerate uses ILP64;
/// the manylinux backend requested by the runtime initializer uses LP64.
#[cfg(target_os = "macos")]
pub const NPY_CBLAS_CHUNK: usize = isize::MAX as usize;
#[cfg(target_os = "linux")]
pub const NPY_CBLAS_CHUNK: usize = 1usize << 30;
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub const NPY_CBLAS_CHUNK: usize = 0;

/// Largest matrix dimension/leading dimension accepted by the active ABI.
/// NumPy subtracts one from the ILP64 maximum on 64-bit platforms.
#[cfg(target_os = "macos")]
pub const CBLAS_MAX_SIZE: isize = isize::MAX - 1;
#[cfg(target_os = "linux")]
pub const CBLAS_MAX_SIZE: isize = i32::MAX as isize;
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub const CBLAS_MAX_SIZE: isize = 0;

/// numpy's `blas_stride`: a byte stride converted to an *element* stride, or
/// `None` when CBLAS cannot be used (BLAS will not walk a zero or negative
/// stride the way numpy wants).
#[inline]
pub fn blas_stride(byte_stride: isize, itemsize: usize) -> Option<i64> {
    if byte_stride > 0 && byte_stride % itemsize as isize == 0 {
        let stride = (byte_stride / itemsize as isize) as i64;
        if !cfg!(target_os = "linux") || stride <= i32::MAX as i64 {
            return Some(stride);
        }
    }
    None
}

/// Whether the process currently has a usable CBLAS backend.
///
/// This is always true for the linked Accelerate backend. On Linux it becomes
/// true only after [`initialize_linux_openblas`] has loaded and validated all
/// required LP64 symbols. Other platforms remain on the Rust fallback.
#[inline]
pub fn have_cblas() -> bool {
    #[cfg(target_os = "macos")]
    {
        true
    }
    #[cfg(target_os = "linux")]
    {
        sys::is_loaded()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

/// Load NumPy's bundled manylinux OpenBLAS by exact filesystem path.
///
/// The Linux Python shim calls this once during import after scanning each
/// `sys.path` entry for `numpy.libs/libscipy_openblas*.so`. A failed load does
/// not poison the initializer, so another candidate can be tried. On success,
/// the library remains loaded for the process lifetime.
#[cfg(target_os = "linux")]
pub fn initialize_linux_openblas(path: &std::path::Path) -> Result<(), String> {
    sys::initialize(path)
}

#[cfg(not(target_os = "linux"))]
pub fn initialize_linux_openblas(_path: &std::path::Path) -> Result<(), String> {
    Err("the NumPy OpenBLAS runtime backend is Linux-only".into())
}

#[cfg(target_os = "macos")]
mod sys {
    use std::ffi::c_void;

    // SAFETY: these are the C signatures from `npy_cblas_base.h`
    //     void cblas_?dotc_sub(const CBLAS_INT N, const void *X,
    //                          const CBLAS_INT incX, const void *Y,
    //                          const CBLAS_INT incY, void *dotc);
    // with `CBLAS_INT = npy_int64` (the ILP64 interface, see the module
    // comment). The `$NEWLAPACK$ILP64` suffix is what `BLAS_SYMBOL_SUFFIX`
    // expands to in `npy_cblas.h` on macOS >= 13.3, and is exactly the symbol
    // the installed numpy imports. Both `x`/`y` are read-only arrays of `n`
    // elements spaced `incx`/`incy` *elements* apart and `out` points at one
    // writable element; every caller below upholds that.
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[link_name = "cblas_sdot$NEWLAPACK$ILP64"]
        pub fn cblas_sdot(n: i64, x: *const f32, incx: i64, y: *const f32, incy: i64) -> f32;

        #[link_name = "cblas_ddot$NEWLAPACK$ILP64"]
        pub fn cblas_ddot(n: i64, x: *const f64, incx: i64, y: *const f64, incy: i64) -> f64;

        #[link_name = "cblas_cdotu_sub$NEWLAPACK$ILP64"]
        pub fn cblas_cdotu_sub(
            n: i64,
            x: *const c_void,
            incx: i64,
            y: *const c_void,
            incy: i64,
            out: *mut c_void,
        );

        #[link_name = "cblas_zdotu_sub$NEWLAPACK$ILP64"]
        pub fn cblas_zdotu_sub(
            n: i64,
            x: *const c_void,
            incx: i64,
            y: *const c_void,
            incy: i64,
            out: *mut c_void,
        );

        #[link_name = "cblas_cdotc_sub$NEWLAPACK$ILP64"]
        pub fn cblas_cdotc_sub(
            n: i64,
            x: *const c_void,
            incx: i64,
            y: *const c_void,
            incy: i64,
            out: *mut c_void,
        );

        #[link_name = "cblas_zdotc_sub$NEWLAPACK$ILP64"]
        pub fn cblas_zdotc_sub(
            n: i64,
            x: *const c_void,
            incx: i64,
            y: *const c_void,
            incy: i64,
            out: *mut c_void,
        );

        #[link_name = "cblas_sgemv$NEWLAPACK$ILP64"]
        pub fn cblas_sgemv(
            order: i32,
            trans: i32,
            m: i64,
            n: i64,
            alpha: f32,
            a: *const f32,
            lda: i64,
            x: *const f32,
            incx: i64,
            beta: f32,
            y: *mut f32,
            incy: i64,
        );

        #[link_name = "cblas_dgemv$NEWLAPACK$ILP64"]
        pub fn cblas_dgemv(
            order: i32,
            trans: i32,
            m: i64,
            n: i64,
            alpha: f64,
            a: *const f64,
            lda: i64,
            x: *const f64,
            incx: i64,
            beta: f64,
            y: *mut f64,
            incy: i64,
        );

        #[link_name = "cblas_cgemv$NEWLAPACK$ILP64"]
        pub fn cblas_cgemv(
            order: i32,
            trans: i32,
            m: i64,
            n: i64,
            alpha: *const c_void,
            a: *const c_void,
            lda: i64,
            x: *const c_void,
            incx: i64,
            beta: *const c_void,
            y: *mut c_void,
            incy: i64,
        );

        #[link_name = "cblas_zgemv$NEWLAPACK$ILP64"]
        pub fn cblas_zgemv(
            order: i32,
            trans: i32,
            m: i64,
            n: i64,
            alpha: *const c_void,
            a: *const c_void,
            lda: i64,
            x: *const c_void,
            incx: i64,
            beta: *const c_void,
            y: *mut c_void,
            incy: i64,
        );

        #[link_name = "cblas_sgemm$NEWLAPACK$ILP64"]
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i64,
            n: i64,
            k: i64,
            alpha: f32,
            a: *const f32,
            lda: i64,
            b: *const f32,
            ldb: i64,
            beta: f32,
            c: *mut f32,
            ldc: i64,
        );

        #[link_name = "cblas_dgemm$NEWLAPACK$ILP64"]
        pub fn cblas_dgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i64,
            n: i64,
            k: i64,
            alpha: f64,
            a: *const f64,
            lda: i64,
            b: *const f64,
            ldb: i64,
            beta: f64,
            c: *mut f64,
            ldc: i64,
        );

        #[link_name = "cblas_cgemm$NEWLAPACK$ILP64"]
        pub fn cblas_cgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i64,
            n: i64,
            k: i64,
            alpha: *const c_void,
            a: *const c_void,
            lda: i64,
            b: *const c_void,
            ldb: i64,
            beta: *const c_void,
            c: *mut c_void,
            ldc: i64,
        );

        #[link_name = "cblas_zgemm$NEWLAPACK$ILP64"]
        pub fn cblas_zgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i64,
            n: i64,
            k: i64,
            alpha: *const c_void,
            a: *const c_void,
            lda: i64,
            b: *const c_void,
            ldb: i64,
            beta: *const c_void,
            c: *mut c_void,
            ldc: i64,
        );
    }
}

#[cfg(target_os = "linux")]
#[path = "blas_linux.rs"]
mod sys;

/// `FLOAT_dot`'s CBLAS branch, including numpy's double chunk accumulator.
///
/// # Safety
/// `x` and `y` must each address `n` in-bounds `f32` elements spaced by the
/// positive element strides `incx` and `incy`.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn sdot(x: *const u8, incx: i64, y: *const u8, incy: i64, n: usize) -> f32 {
    let mut sum = 0.0f64;
    let (mut xp, mut yp, mut left) = (x, y, n);
    while left > 0 {
        let chunk = left.min(NPY_CBLAS_CHUNK);
        // SAFETY: the caller guarantees both strided runs are in bounds.
        sum += unsafe {
            sys::cblas_sdot(chunk as i64, xp as *const f32, incx, yp as *const f32, incy) as f64
        };
        left -= chunk;
        if left > 0 {
            // SAFETY: a further chunk remains, so these advances stay within
            // the caller-provided runs.
            unsafe {
                xp = xp.offset(chunk as isize * incx as isize * 4);
                yp = yp.offset(chunk as isize * incy as isize * 4);
            }
        }
    }
    sum as f32
}

/// `DOUBLE_dot`'s CBLAS branch.
///
/// # Safety
/// As [`sdot`], for `f64` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn ddot(x: *const u8, incx: i64, y: *const u8, incy: i64, n: usize) -> f64 {
    let mut sum = 0.0f64;
    let (mut xp, mut yp, mut left) = (x, y, n);
    while left > 0 {
        let chunk = left.min(NPY_CBLAS_CHUNK);
        // SAFETY: the caller guarantees both strided runs are in bounds.
        sum += unsafe {
            sys::cblas_ddot(chunk as i64, xp as *const f64, incx, yp as *const f64, incy)
        };
        left -= chunk;
        if left > 0 {
            // SAFETY: as in `sdot`, with an 8-byte element.
            unsafe {
                xp = xp.offset(chunk as isize * incx as isize * 8);
                yp = yp.offset(chunk as isize * incy as isize * 8);
            }
        }
    }
    sum
}

/// `CFLOAT_dot`'s unconjugated CBLAS branch.
///
/// # Safety
/// As [`sdot`], for `npy_cfloat` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn cdotu(x: *const u8, incx: i64, y: *const u8, incy: i64, n: usize) -> C32 {
    let mut sum = [0.0f64, 0.0f64];
    let (mut xp, mut yp, mut left) = (x, y, n);
    while left > 0 {
        let chunk = left.min(NPY_CBLAS_CHUNK);
        let mut tmp = [0.0f32, 0.0f32];
        // SAFETY: the caller guarantees both runs; `tmp` is one writable
        // complex32 value.
        unsafe {
            sys::cblas_cdotu_sub(
                chunk as i64,
                xp as *const _,
                incx,
                yp as *const _,
                incy,
                tmp.as_mut_ptr() as *mut _,
            );
        }
        sum[0] += tmp[0] as f64;
        sum[1] += tmp[1] as f64;
        left -= chunk;
        if left > 0 {
            // SAFETY: as in `sdot`, with an 8-byte complex element.
            unsafe {
                xp = xp.offset(chunk as isize * incx as isize * 8);
                yp = yp.offset(chunk as isize * incy as isize * 8);
            }
        }
    }
    C32::new(sum[0] as f32, sum[1] as f32)
}

/// `CDOUBLE_dot`'s unconjugated CBLAS branch.
///
/// # Safety
/// As [`sdot`], for `npy_cdouble` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn zdotu(x: *const u8, incx: i64, y: *const u8, incy: i64, n: usize) -> C64v {
    let mut sum = [0.0f64, 0.0f64];
    let (mut xp, mut yp, mut left) = (x, y, n);
    while left > 0 {
        let chunk = left.min(NPY_CBLAS_CHUNK);
        let mut tmp = [0.0f64, 0.0f64];
        // SAFETY: the caller guarantees both runs; `tmp` is one writable
        // complex64 value.
        unsafe {
            sys::cblas_zdotu_sub(
                chunk as i64,
                xp as *const _,
                incx,
                yp as *const _,
                incy,
                tmp.as_mut_ptr() as *mut _,
            );
        }
        sum[0] += tmp[0];
        sum[1] += tmp[1];
        left -= chunk;
        if left > 0 {
            // SAFETY: as in `sdot`, with a 16-byte complex element.
            unsafe {
                xp = xp.offset(chunk as isize * incx as isize * 16);
                yp = yp.offset(chunk as isize * incy as isize * 16);
            }
        }
    }
    C64v::new(sum[0], sum[1])
}

/// `CDOUBLE_dotc`'s CBLAS branch, transcribed including the `npy_double`
/// accumulator it keeps "at least double for stability" across chunks.
///
/// # Safety
/// `x` and `y` must each address `n` in-bounds `npy_cdouble` elements spaced
/// `incx`/`incy` elements apart, and `incx`/`incy` must be positive.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn zdotc(x: *const u8, incx: i64, y: *const u8, incy: i64, n: usize) -> C64v {
    let mut sum = [0.0f64, 0.0f64];
    let (mut xp, mut yp, mut left) = (x, y, n);
    while left > 0 {
        let chunk = left.min(NPY_CBLAS_CHUNK);
        let mut tmp = [0.0f64, 0.0f64];
        // SAFETY: the caller guarantees the run; `tmp` is two live f64s.
        unsafe {
            sys::cblas_zdotc_sub(
                chunk as i64,
                xp as *const _,
                incx,
                yp as *const _,
                incy,
                tmp.as_mut_ptr() as *mut _,
            );
        }
        sum[0] += tmp[0];
        sum[1] += tmp[1];
        left -= chunk;
        if left > 0 {
            // numpy advances with the *byte* strides here. The caller's
            // in-bounds-run contract guarantees the next chunk starts inside
            // each allocation.
            // SAFETY: `left > 0`, so neither pointer advances past the final
            // element of its run.
            unsafe {
                xp = xp.offset(chunk as isize * incx as isize * 16);
                yp = yp.offset(chunk as isize * incy as isize * 16);
            }
        }
    }
    C64v::new(sum[0], sum[1])
}

/// `CFLOAT_dotc`'s CBLAS branch. numpy accumulates the chunk results in
/// `npy_double` for this type too, and rounds to `npy_float` only on store.
///
/// # Safety
/// As [`zdotc`], for `npy_cfloat` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn cdotc(x: *const u8, incx: i64, y: *const u8, incy: i64, n: usize) -> C32 {
    let mut sum = [0.0f64, 0.0f64];
    let (mut xp, mut yp, mut left) = (x, y, n);
    while left > 0 {
        let chunk = left.min(NPY_CBLAS_CHUNK);
        let mut tmp = [0.0f32, 0.0f32];
        // SAFETY: the caller guarantees the run; `tmp` is two live f32s.
        unsafe {
            sys::cblas_cdotc_sub(
                chunk as i64,
                xp as *const _,
                incx,
                yp as *const _,
                incy,
                tmp.as_mut_ptr() as *mut _,
            );
        }
        sum[0] += tmp[0] as f64;
        sum[1] += tmp[1] as f64;
        left -= chunk;
        if left > 0 {
            // SAFETY: as in `zdotc`, with an 8-byte element.
            unsafe {
                xp = xp.offset(chunk as isize * incx as isize * 8);
                yp = yp.offset(chunk as isize * incy as isize * 8);
            }
        }
    }
    C32::new(sum[0] as f32, sum[1] as f32)
}

/// numpy's `_gemv` wrapper for a logical `m`-by-`n` matrix and length-`n`
/// vector. `matrix_col_major` selects the same CBLAS order inferred by
/// `is_blasable2d`; the CBLAS dimensions are deliberately flipped because
/// numpy always requests `CblasTrans` here.
///
/// # Safety
/// `matrix`, `vector`, and `out` must describe the complete BLAS extents
/// encoded by the dimensions, leading dimension, and positive increments.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn sgemv(
    matrix: *const u8,
    matrix_col_major: bool,
    lda: i64,
    vector: *const u8,
    incx: i64,
    out: *mut u8,
    incy: i64,
    m: usize,
    n: usize,
) {
    // SAFETY: the caller guarantees the matrix, input, and output extents;
    // the argument order exactly transcribes numpy's `FLOAT_gemv`.
    unsafe {
        sys::cblas_sgemv(
            if matrix_col_major {
                CBLAS_COL_MAJOR
            } else {
                CBLAS_ROW_MAJOR
            },
            CBLAS_TRANS,
            n as i64,
            m as i64,
            1.0,
            matrix as *const f32,
            lda,
            vector as *const f32,
            incx,
            0.0,
            out as *mut f32,
            incy,
        );
    }
}

/// `DOUBLE_gemv`; see [`sgemv`].
///
/// # Safety
/// As [`sgemv`], for `f64` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn dgemv(
    matrix: *const u8,
    matrix_col_major: bool,
    lda: i64,
    vector: *const u8,
    incx: i64,
    out: *mut u8,
    incy: i64,
    m: usize,
    n: usize,
) {
    // SAFETY: as in `sgemv`, for `f64` elements.
    unsafe {
        sys::cblas_dgemv(
            if matrix_col_major {
                CBLAS_COL_MAJOR
            } else {
                CBLAS_ROW_MAJOR
            },
            CBLAS_TRANS,
            n as i64,
            m as i64,
            1.0,
            matrix as *const f64,
            lda,
            vector as *const f64,
            incx,
            0.0,
            out as *mut f64,
            incy,
        );
    }
}

/// `CFLOAT_gemv`; see [`sgemv`].
///
/// # Safety
/// As [`sgemv`], for `npy_cfloat` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn cgemv(
    matrix: *const u8,
    matrix_col_major: bool,
    lda: i64,
    vector: *const u8,
    incx: i64,
    out: *mut u8,
    incy: i64,
    m: usize,
    n: usize,
) {
    let alpha = [1.0f32, 0.0f32];
    let beta = [0.0f32, 0.0f32];
    // SAFETY: as in `sgemv`; both scalar arrays are live complex values.
    unsafe {
        sys::cblas_cgemv(
            if matrix_col_major {
                CBLAS_COL_MAJOR
            } else {
                CBLAS_ROW_MAJOR
            },
            CBLAS_TRANS,
            n as i64,
            m as i64,
            alpha.as_ptr() as *const _,
            matrix as *const _,
            lda,
            vector as *const _,
            incx,
            beta.as_ptr() as *const _,
            out as *mut _,
            incy,
        );
    }
}

/// `CDOUBLE_gemv`; see [`sgemv`].
///
/// # Safety
/// As [`sgemv`], for `npy_cdouble` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn zgemv(
    matrix: *const u8,
    matrix_col_major: bool,
    lda: i64,
    vector: *const u8,
    incx: i64,
    out: *mut u8,
    incy: i64,
    m: usize,
    n: usize,
) {
    let alpha = [1.0f64, 0.0f64];
    let beta = [0.0f64, 0.0f64];
    // SAFETY: as in `sgemv`; both scalar arrays are live complex values.
    unsafe {
        sys::cblas_zgemv(
            if matrix_col_major {
                CBLAS_COL_MAJOR
            } else {
                CBLAS_ROW_MAJOR
            },
            CBLAS_TRANS,
            n as i64,
            m as i64,
            alpha.as_ptr() as *const _,
            matrix as *const _,
            lda,
            vector as *const _,
            incx,
            beta.as_ptr() as *const _,
            out as *mut _,
            incy,
        );
    }
}

/// Row-major matrix-matrix multiplication with numpy's transpose and leading
/// dimension choices already resolved by the caller.
///
/// # Safety
/// `a`, `b`, and `out` must describe the complete BLAS matrix extents encoded
/// by `m`, `n`, `k`, the transpose flags, and the leading dimensions.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn sgemm(
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
) {
    // SAFETY: the caller guarantees all matrix extents and leading
    // dimensions; the enum constants match numpy's `npy_cblas.h`.
    unsafe {
        sys::cblas_sgemm(
            CBLAS_ROW_MAJOR,
            if transpose_a {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            if transpose_b {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            m as i64,
            n as i64,
            k as i64,
            1.0,
            a as *const f32,
            lda,
            b as *const f32,
            ldb,
            0.0,
            out as *mut f32,
            ldc,
        );
    }
}

/// `DOUBLE_matmul_matrixmatrix`; see [`sgemm`].
///
/// # Safety
/// As [`sgemm`], for `f64` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn dgemm(
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
) {
    // SAFETY: as in `sgemm`, for `f64` elements.
    unsafe {
        sys::cblas_dgemm(
            CBLAS_ROW_MAJOR,
            if transpose_a {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            if transpose_b {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            m as i64,
            n as i64,
            k as i64,
            1.0,
            a as *const f64,
            lda,
            b as *const f64,
            ldb,
            0.0,
            out as *mut f64,
            ldc,
        );
    }
}

/// `CFLOAT_matmul_matrixmatrix`; see [`sgemm`].
///
/// # Safety
/// As [`sgemm`], for `npy_cfloat` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn cgemm(
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
) {
    let alpha = [1.0f32, 0.0f32];
    let beta = [0.0f32, 0.0f32];
    // SAFETY: as in `sgemm`; both scalar arrays are live complex values.
    unsafe {
        sys::cblas_cgemm(
            CBLAS_ROW_MAJOR,
            if transpose_a {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            if transpose_b {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            m as i64,
            n as i64,
            k as i64,
            alpha.as_ptr() as *const _,
            a as *const _,
            lda,
            b as *const _,
            ldb,
            beta.as_ptr() as *const _,
            out as *mut _,
            ldc,
        );
    }
}

/// `CDOUBLE_matmul_matrixmatrix`; see [`sgemm`].
///
/// # Safety
/// As [`sgemm`], for `npy_cdouble` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn zgemm(
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
) {
    let alpha = [1.0f64, 0.0f64];
    let beta = [0.0f64, 0.0f64];
    // SAFETY: as in `sgemm`; both scalar arrays are live complex values.
    unsafe {
        sys::cblas_zgemm(
            CBLAS_ROW_MAJOR,
            if transpose_a {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            if transpose_b {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            m as i64,
            n as i64,
            k as i64,
            alpha.as_ptr() as *const _,
            a as *const _,
            lda,
            b as *const _,
            ldb,
            beta.as_ptr() as *const _,
            out as *mut _,
            ldc,
        );
    }
}

/// `CFLOAT_vecmat_via_gemm`, with `x` as a conjugate-transposed 1-by-`n`
/// row and `y` as an `n`-by-`m` matrix.
///
/// # Safety
/// `x`, `y`, and `out` must satisfy the CBLAS matrix extents described by
/// `n`, `m`, `lda`, and `ldb`; `out` must hold `m` writable `npy_cfloat`s.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn cgemm_vecmat(
    x: *const u8,
    lda: i64,
    y: *const u8,
    ldb: i64,
    transpose_y: bool,
    out: *mut u8,
    n: usize,
    m: usize,
) {
    let alpha = [1.0f32, 0.0f32];
    let beta = [0.0f32, 0.0f32];
    // SAFETY: the caller guarantees all three matrix extents. The scalar
    // arrays are live complex values for the duration of the call, and the
    // enum values are the definitions in numpy's `npy_cblas.h`.
    unsafe {
        sys::cblas_cgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_CONJ_TRANS,
            if transpose_y {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            1,
            m as i64,
            n as i64,
            alpha.as_ptr() as *const _,
            x as *const _,
            lda,
            y as *const _,
            ldb,
            beta.as_ptr() as *const _,
            out as *mut _,
            m as i64,
        );
    }
}

/// `CDOUBLE_vecmat_via_gemm`; see [`cgemm_vecmat`].
///
/// # Safety
/// As [`cgemm_vecmat`], for `npy_cdouble` elements.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub unsafe fn zgemm_vecmat(
    x: *const u8,
    lda: i64,
    y: *const u8,
    ldb: i64,
    transpose_y: bool,
    out: *mut u8,
    n: usize,
    m: usize,
) {
    let alpha = [1.0f64, 0.0f64];
    let beta = [0.0f64, 0.0f64];
    // SAFETY: the caller guarantees the matrix extents and writable output;
    // the remaining arguments exactly transcribe `CDOUBLE_vecmat_via_gemm`.
    unsafe {
        sys::cblas_zgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_CONJ_TRANS,
            if transpose_y {
                CBLAS_TRANS
            } else {
                CBLAS_NO_TRANS
            },
            1,
            m as i64,
            n as i64,
            alpha.as_ptr() as *const _,
            x as *const _,
            lda,
            y as *const _,
            ldb,
            beta.as_ptr() as *const _,
            out as *mut _,
            m as i64,
        );
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn sdot(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> f32 {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn ddot(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> f64 {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn cdotu(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> C32 {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn zdotu(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> C64v {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub unsafe fn zdotc(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> C64v {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub unsafe fn cdotc(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> C32 {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn sgemv(
    _matrix: *const u8,
    _matrix_col_major: bool,
    _lda: i64,
    _vector: *const u8,
    _incx: i64,
    _out: *mut u8,
    _incy: i64,
    _m: usize,
    _n: usize,
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn dgemv(
    _matrix: *const u8,
    _matrix_col_major: bool,
    _lda: i64,
    _vector: *const u8,
    _incx: i64,
    _out: *mut u8,
    _incy: i64,
    _m: usize,
    _n: usize,
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn cgemv(
    _matrix: *const u8,
    _matrix_col_major: bool,
    _lda: i64,
    _vector: *const u8,
    _incx: i64,
    _out: *mut u8,
    _incy: i64,
    _m: usize,
    _n: usize,
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn zgemv(
    _matrix: *const u8,
    _matrix_col_major: bool,
    _lda: i64,
    _vector: *const u8,
    _incx: i64,
    _out: *mut u8,
    _incy: i64,
    _m: usize,
    _n: usize,
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn sgemm(
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
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn dgemm(
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
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn cgemm(
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
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
/// Unavailable off Apple platforms.
///
/// # Safety
/// This function never returns.
pub unsafe fn zgemm(
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
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub unsafe fn cgemm_vecmat(
    _x: *const u8,
    _lda: i64,
    _y: *const u8,
    _ldb: i64,
    _transpose_y: bool,
    _out: *mut u8,
    _n: usize,
    _m: usize,
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub unsafe fn zgemm_vecmat(
    _x: *const u8,
    _lda: i64,
    _y: *const u8,
    _ldb: i64,
    _transpose_y: bool,
    _out: *mut u8,
    _n: usize,
    _m: usize,
) {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_blas_stride_rejects_nonpositive_and_unaligned_strides() {
        assert_eq!(blas_stride(16, 16), Some(1));
        assert_eq!(blas_stride(32, 16), Some(2));
        assert_eq!(blas_stride(0, 16), None);
        assert_eq!(blas_stride(-16, 16), None);
        assert_eq!(blas_stride(17, 16), None);
    }
}
