//! The two CBLAS entry points numpy's `matmul.c.src` reaches for the
//! *conjugating* dot product, bound to the same library numpy is linked
//! against.
//!
//! Everything else in this crate is from-scratch Rust, and deliberately so.
//! This module is the one exception, for a reason that is not a shortcut:
//! `@TYPE@_dotc` routes to `cblas_?dotc_sub` whenever both operand strides are
//! usable, so on this platform numpy's answer for `vecdot`/`vecmat` on complex
//! input is *literally* Apple Accelerate's answer. That answer is not
//! reproducible by transcription. Concretely, for `vecmat` over the
//! special-value grid Accelerate returns an imaginary part of `-nan` for
//! `x = nan+0j, y = nan+0j` but `+nan` for `x = inf+0j, y = nan+0j`, while in
//! both cases every f64 intermediate is bit-identical (`nan*0` and `inf*0`
//! both yield `+0x7ff8000000000000`, and `xi`/`yi` are `+0.0`). Only an
//! explicit `FNEG` can produce a negative NaN on AArch64, so no straight-line
//! expression can separate the two -- verified by an exhaustive search over
//! the plain and fused formulations of `xr*yi - xi*yr`. Calling the same
//! routine numpy calls is therefore the only faithful option, and it is the
//! same choice already made for the complex libm functions.
//!
//! Which symbol: the numpy wheel in `.venv` links
//! `_cblas_zdotc_sub$NEWLAPACK$ILP64` (checked with `nm -u` on
//! `_multiarray_umath`), i.e. Accelerate's *new* LAPACK interface in its
//! ILP64 flavour, so `CBLAS_INT` is 64 bits wide. `npy_cblas.h` then makes
//! `CBLAS_INT_MAX` `NPY_MAX_INT64` and `NPY_CBLAS_CHUNK` `NPY_MAX_INTP`, which
//! is why the chunking loop in `dotc` never actually splits a call here.

#![allow(clippy::missing_safety_doc)]

use crate::element::{C32, C64v};

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
        // numpy advances with the *byte* strides here.
        // SAFETY: `chunk <= left`, so this lands at or one past the run's end.
        unsafe {
            xp = xp.offset(chunk as isize * incx as isize * 16);
            yp = yp.offset(chunk as isize * incy as isize * 16);
        }
        left -= chunk;
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
        // SAFETY: as in `zdotc`, with an 8-byte element.
        unsafe {
            xp = xp.offset(chunk as isize * incx as isize * 8);
            yp = yp.offset(chunk as isize * incy as isize * 8);
        }
        left -= chunk;
    }
    C32::new(sum[0] as f32, sum[1] as f32)
}

#[cfg(not(all(target_os = "macos", target_vendor = "apple")))]
pub unsafe fn zdotc(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> C64v {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}

#[cfg(not(all(target_os = "macos", target_vendor = "apple")))]
pub unsafe fn cdotc(_x: *const u8, _incx: i64, _y: *const u8, _incy: i64, _n: usize) -> C32 {
    unreachable!("HAVE_CBLAS is false off Apple platforms")
}
