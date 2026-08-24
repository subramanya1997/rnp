//! The Accelerate CBLAS entry points numpy's `matmul.c.src` reaches for
//! complex `vecdot` and `vecmat`, bound to the same library and ABI as numpy.
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
//! Which symbol: the numpy wheel in `.venv` links
//! `_cblas_zdotc_sub$NEWLAPACK$ILP64` (checked with `nm -u` on
//! `_multiarray_umath`), i.e. Accelerate's *new* LAPACK interface in its
//! ILP64 flavour, so `CBLAS_INT` is 64 bits wide. `npy_cblas.h` then makes
//! `CBLAS_INT_MAX` `NPY_MAX_INT64` and `NPY_CBLAS_CHUNK` `NPY_MAX_INTP`, which
//! is why the chunking loop in `dotc` never actually splits a call here.

use crate::element::{C32, C64v};

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;
const CBLAS_CONJ_TRANS: i32 = 113;

/// numpy's `NPY_CBLAS_CHUNK`. With ILP64 `CBLAS_INT_MAX == NPY_MAX_INTP`, so
/// `npy_cblas.h` takes the `#else` branch and the chunk is the whole range.
pub const NPY_CBLAS_CHUNK: usize = isize::MAX as usize;

/// numpy's `blas_stride`: a byte stride converted to an *element* stride, or
/// `None` when CBLAS cannot be used (BLAS will not walk a zero or negative
/// stride the way numpy wants).
#[inline]
pub fn blas_stride(byte_stride: isize, itemsize: usize) -> Option<i64> {
    if byte_stride > 0 && byte_stride % itemsize as isize == 0 {
        // `stride <= CBLAS_INT_MAX` is vacuous at 64-bit CBLAS_INT.
        return Some((byte_stride / itemsize as isize) as i64);
    }
    None
}

/// True when this build can call Accelerate at all.
pub const HAVE_CBLAS: bool = cfg!(all(target_os = "macos", target_vendor = "apple"));

#[cfg(all(target_os = "macos", target_vendor = "apple"))]
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

/// `CDOUBLE_dotc`'s CBLAS branch, transcribed including the `npy_double`
/// accumulator it keeps "at least double for stability" across chunks.
///
/// # Safety
/// `x` and `y` must each address `n` in-bounds `npy_cdouble` elements spaced
/// `incx`/`incy` elements apart, and `incx`/`incy` must be positive.
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
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
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
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

/// `CFLOAT_vecmat_via_gemm`, with `x` as a conjugate-transposed 1-by-`n`
/// row and `y` as an `n`-by-`m` matrix.
///
/// # Safety
/// `x`, `y`, and `out` must satisfy the CBLAS matrix extents described by
/// `n`, `m`, `lda`, and `ldb`; `out` must hold `m` writable `npy_cfloat`s.
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
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
            if transpose_y { CBLAS_TRANS } else { CBLAS_NO_TRANS },
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
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
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
            if transpose_y { CBLAS_TRANS } else { CBLAS_NO_TRANS },
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

#[cfg(not(all(target_os = "macos", target_vendor = "apple")))]
pub unsafe fn zdotc(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> C64v {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(all(target_os = "macos", target_vendor = "apple")))]
pub unsafe fn cdotc(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> C32 {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(all(target_os = "macos", target_vendor = "apple")))]
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

#[cfg(not(all(target_os = "macos", target_vendor = "apple")))]
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
