//! Runtime bindings for NumPy's bundled manylinux OpenBLAS.

use std::ffi::{c_char, c_void, CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::OnceLock;

use crate::element::{C64v, C32};

type Sdot = unsafe extern "C" fn(i32, *const f32, i32, *const f32, i32) -> f32;
type Ddot = unsafe extern "C" fn(i32, *const f64, i32, *const f64, i32) -> f64;
type Cdot = unsafe extern "C" fn(i32, *const c_void, i32, *const c_void, i32, *mut c_void);

type Sgemv = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    f32,
    *const f32,
    i32,
    *const f32,
    i32,
    f32,
    *mut f32,
    i32,
);
type Dgemv = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    f64,
    *const f64,
    i32,
    *const f64,
    i32,
    f64,
    *mut f64,
    i32,
);
type Cgemv = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    *const c_void,
    i32,
    *const c_void,
    *mut c_void,
    i32,
);

type Sgemm = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f32,
    *const f32,
    i32,
    *const f32,
    i32,
    f32,
    *mut f32,
    i32,
);
type Dgemm = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    f64,
    *const f64,
    i32,
    *const f64,
    i32,
    f64,
    *mut f64,
    i32,
);
type Cgemm = unsafe extern "C" fn(
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    *const c_void,
    i32,
    *const c_void,
    i32,
    *const c_void,
    *mut c_void,
    i32,
);

macro_rules! lapack_api {
    ($(
        $field:ident: $function_type:ident = $symbol:literal (
            $($argument:ident: $argument_type:ty),* $(,)?
        );
    )*) => {
        $(type $function_type = unsafe extern "C" fn($($argument_type),*);)*

        struct LapackApi {
            $($field: $function_type,)*
        }

        impl LapackApi {
            unsafe fn load(handle: *mut c_void) -> Result<Self, String> {
                Ok(Self {
                    $($field: {
                        // SAFETY: each entry lists the exact ILP64 Fortran
                        // signature exported by NumPy's scipy OpenBLAS build.
                        unsafe {
                            symbol::<$function_type>(
                                handle,
                                concat!($symbol, "\0").as_bytes(),
                            )
                        }?
                    },)*
                })
            }
        }

        $(
            pub unsafe fn $field($($argument: $argument_type),*) {
                // SAFETY: the caller guarantees the LAPACK dimensions and
                // pointer extents documented at the kernel call site.
                unsafe { (api().lapack.$field)($($argument),*) }
            }
        )*
    };
}

lapack_api! {
    sgesv: Sgesv = "scipy_sgesv_64_" (
        n: *const i64, nrhs: *const i64, a: *mut f32, lda: *const i64,
        ipiv: *mut i64, b: *mut f32, ldb: *const i64, info: *mut i64
    );
    dgesv: Dgesv = "scipy_dgesv_64_" (
        n: *const i64, nrhs: *const i64, a: *mut f64, lda: *const i64,
        ipiv: *mut i64, b: *mut f64, ldb: *const i64, info: *mut i64
    );
    cgesv: Cgesv = "scipy_cgesv_64_" (
        n: *const i64, nrhs: *const i64, a: *mut C32, lda: *const i64,
        ipiv: *mut i64, b: *mut C32, ldb: *const i64, info: *mut i64
    );
    zgesv: Zgesv = "scipy_zgesv_64_" (
        n: *const i64, nrhs: *const i64, a: *mut C64v, lda: *const i64,
        ipiv: *mut i64, b: *mut C64v, ldb: *const i64, info: *mut i64
    );

    sgetrf: Sgetrf = "scipy_sgetrf_64_" (
        m: *const i64, n: *const i64, a: *mut f32, lda: *const i64,
        ipiv: *mut i64, info: *mut i64
    );
    dgetrf: Dgetrf = "scipy_dgetrf_64_" (
        m: *const i64, n: *const i64, a: *mut f64, lda: *const i64,
        ipiv: *mut i64, info: *mut i64
    );
    cgetrf: Cgetrf = "scipy_cgetrf_64_" (
        m: *const i64, n: *const i64, a: *mut C32, lda: *const i64,
        ipiv: *mut i64, info: *mut i64
    );
    zgetrf: Zgetrf = "scipy_zgetrf_64_" (
        m: *const i64, n: *const i64, a: *mut C64v, lda: *const i64,
        ipiv: *mut i64, info: *mut i64
    );

    spotrf: Spotrf = "scipy_spotrf_64_" (
        uplo: *const u8, n: *const i64, a: *mut f32, lda: *const i64,
        info: *mut i64
    );
    dpotrf: Dpotrf = "scipy_dpotrf_64_" (
        uplo: *const u8, n: *const i64, a: *mut f64, lda: *const i64,
        info: *mut i64
    );
    cpotrf: Cpotrf = "scipy_cpotrf_64_" (
        uplo: *const u8, n: *const i64, a: *mut C32, lda: *const i64,
        info: *mut i64
    );
    zpotrf: Zpotrf = "scipy_zpotrf_64_" (
        uplo: *const u8, n: *const i64, a: *mut C64v, lda: *const i64,
        info: *mut i64
    );

    sgelsd: Sgelsd = "scipy_sgelsd_64_" (
        m: *const i64, n: *const i64, nrhs: *const i64, a: *mut f32,
        lda: *const i64, b: *mut f32, ldb: *const i64, s: *mut f32,
        rcond: *const f32, rank: *mut i64, work: *mut f32,
        lwork: *const i64, iwork: *mut i64, info: *mut i64
    );
    dgelsd: Dgelsd = "scipy_dgelsd_64_" (
        m: *const i64, n: *const i64, nrhs: *const i64, a: *mut f64,
        lda: *const i64, b: *mut f64, ldb: *const i64, s: *mut f64,
        rcond: *const f64, rank: *mut i64, work: *mut f64,
        lwork: *const i64, iwork: *mut i64, info: *mut i64
    );
    cgelsd: Cgelsd = "scipy_cgelsd_64_" (
        m: *const i64, n: *const i64, nrhs: *const i64, a: *mut C32,
        lda: *const i64, b: *mut C32, ldb: *const i64, s: *mut f32,
        rcond: *const f32, rank: *mut i64, work: *mut C32,
        lwork: *const i64, rwork: *mut f32, iwork: *mut i64, info: *mut i64
    );
    zgelsd: Zgelsd = "scipy_zgelsd_64_" (
        m: *const i64, n: *const i64, nrhs: *const i64, a: *mut C64v,
        lda: *const i64, b: *mut C64v, ldb: *const i64, s: *mut f64,
        rcond: *const f64, rank: *mut i64, work: *mut C64v,
        lwork: *const i64, rwork: *mut f64, iwork: *mut i64, info: *mut i64
    );

    dgesdd: Dgesdd = "scipy_dgesdd_64_" (
        jobz: *const u8, m: *const i64, n: *const i64, a: *mut f64,
        lda: *const i64, s: *mut f64, u: *mut f64, ldu: *const i64,
        vt: *mut f64, ldvt: *const i64, work: *mut f64,
        lwork: *const i64, iwork: *mut i64, info: *mut i64
    );
    zgesdd: Zgesdd = "scipy_zgesdd_64_" (
        jobz: *const u8, m: *const i64, n: *const i64, a: *mut C64v,
        lda: *const i64, s: *mut f64, u: *mut C64v, ldu: *const i64,
        vt: *mut C64v, ldvt: *const i64, work: *mut C64v,
        lwork: *const i64, rwork: *mut f64, iwork: *mut i64, info: *mut i64
    );

    dsyevd: Dsyevd = "scipy_dsyevd_64_" (
        jobz: *const u8, uplo: *const u8, n: *const i64, a: *mut f64,
        lda: *const i64, w: *mut f64, work: *mut f64, lwork: *const i64,
        iwork: *mut i64, liwork: *const i64, info: *mut i64
    );
    zheevd: Zheevd = "scipy_zheevd_64_" (
        jobz: *const u8, uplo: *const u8, n: *const i64, a: *mut C64v,
        lda: *const i64, w: *mut f64, work: *mut C64v, lwork: *const i64,
        rwork: *mut f64, lrwork: *const i64, iwork: *mut i64,
        liwork: *const i64, info: *mut i64
    );

    dgeev: Dgeev = "scipy_dgeev_64_" (
        jobvl: *const u8, jobvr: *const u8, n: *const i64, a: *mut f64,
        lda: *const i64, wr: *mut f64, wi: *mut f64, vl: *mut f64,
        ldvl: *const i64, vr: *mut f64, ldvr: *const i64, work: *mut f64,
        lwork: *const i64, info: *mut i64
    );
    zgeev: Zgeev = "scipy_zgeev_64_" (
        jobvl: *const u8, jobvr: *const u8, n: *const i64, a: *mut C64v,
        lda: *const i64, w: *mut C64v, vl: *mut C64v, ldvl: *const i64,
        vr: *mut C64v, ldvr: *const i64, work: *mut C64v,
        lwork: *const i64, rwork: *mut f64, info: *mut i64
    );

    dgeqrf: Dgeqrf = "scipy_dgeqrf_64_" (
        m: *const i64, n: *const i64, a: *mut f64, lda: *const i64,
        tau: *mut f64, work: *mut f64, lwork: *const i64, info: *mut i64
    );
    zgeqrf: Zgeqrf = "scipy_zgeqrf_64_" (
        m: *const i64, n: *const i64, a: *mut C64v, lda: *const i64,
        tau: *mut C64v, work: *mut C64v, lwork: *const i64, info: *mut i64
    );
    dorgqr: Dorgqr = "scipy_dorgqr_64_" (
        m: *const i64, n: *const i64, k: *const i64, a: *mut f64,
        lda: *const i64, tau: *mut f64, work: *mut f64,
        lwork: *const i64, info: *mut i64
    );
    zungqr: Zungqr = "scipy_zungqr_64_" (
        m: *const i64, n: *const i64, k: *const i64, a: *mut C64v,
        lda: *const i64, tau: *mut C64v, work: *mut C64v,
        lwork: *const i64, info: *mut i64
    );
}

struct Api {
    handle: usize,
    sdot: Sdot,
    ddot: Ddot,
    cdotu: Cdot,
    zdotu: Cdot,
    cdotc: Cdot,
    zdotc: Cdot,
    sgemv: Sgemv,
    dgemv: Dgemv,
    cgemv: Cgemv,
    zgemv: Cgemv,
    sgemm: Sgemm,
    dgemm: Dgemm,
    cgemm: Cgemm,
    zgemm: Cgemm,
    lapack: LapackApi,
}

static API: OnceLock<Api> = OnceLock::new();

fn dlerror_message() -> String {
    // SAFETY: `dlerror` returns either null or a process-owned NUL-terminated
    // diagnostic string that remains valid until the next loader operation.
    let error = unsafe { libc::dlerror() };
    if error.is_null() {
        return "dynamic loader returned no diagnostic".into();
    }
    // SAFETY: the non-null pointer is the NUL-terminated string promised by
    // `dlerror`; it is copied before another loader operation can invalidate it.
    unsafe { CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned()
}

unsafe fn symbol<T: Copy>(handle: *mut c_void, name: &'static [u8]) -> Result<T, String> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    // SAFETY: clearing prior loader state is required before `dlsym`; it does
    // not dereference application memory.
    unsafe {
        libc::dlerror();
    }
    // SAFETY: `handle` came from a successful `dlopen` and `name` is a static
    // NUL-terminated byte string.
    let pointer = unsafe { libc::dlsym(handle, name.as_ptr().cast::<c_char>()) };
    if pointer.is_null() {
        let printable = CStr::from_bytes_with_nul(name)
            .expect("static BLAS symbol must be NUL-terminated")
            .to_string_lossy();
        return Err(format!(
            "missing OpenBLAS symbol {printable}: {}",
            dlerror_message()
        ));
    }
    // SAFETY: every caller supplies the exact C signature for the named LP64
    // CBLAS function. POSIX specifies that a `dlsym` result for a function can
    // be converted to the corresponding function pointer and invoked.
    Ok(unsafe { std::mem::transmute_copy(&pointer) })
}

impl Api {
    unsafe fn load(handle: *mut c_void) -> Result<Self, String> {
        macro_rules! get {
            ($name:literal, $ty:ty) => {{
                // SAFETY: the literal identifies the `$ty` CBLAS signature,
                // and `handle` is live for the process lifetime on success.
                unsafe { symbol::<$ty>(handle, concat!($name, "\0").as_bytes()) }?
            }};
        }

        Ok(Self {
            handle: handle as usize,
            sdot: get!("scipy_cblas_sdot", Sdot),
            ddot: get!("scipy_cblas_ddot", Ddot),
            cdotu: get!("scipy_cblas_cdotu_sub", Cdot),
            zdotu: get!("scipy_cblas_zdotu_sub", Cdot),
            cdotc: get!("scipy_cblas_cdotc_sub", Cdot),
            zdotc: get!("scipy_cblas_zdotc_sub", Cdot),
            sgemv: get!("scipy_cblas_sgemv", Sgemv),
            dgemv: get!("scipy_cblas_dgemv", Dgemv),
            cgemv: get!("scipy_cblas_cgemv", Cgemv),
            zgemv: get!("scipy_cblas_zgemv", Cgemv),
            sgemm: get!("scipy_cblas_sgemm", Sgemm),
            dgemm: get!("scipy_cblas_dgemm", Dgemm),
            cgemm: get!("scipy_cblas_cgemm", Cgemm),
            zgemm: get!("scipy_cblas_zgemm", Cgemm),
            lapack: {
                // SAFETY: `handle` is live and `LapackApi::load` validates
                // every required ILP64 symbol before returning.
                unsafe { LapackApi::load(handle) }?
            },
        })
    }
}

pub(super) fn initialize(path: &Path) -> Result<(), String> {
    if API.get().is_some() {
        return Ok(());
    }
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| "OpenBLAS path contains a NUL byte".to_string())?;
    // SAFETY: `path` is NUL-terminated and lives through the call. RTLD_LOCAL
    // keeps NumPy's private BLAS symbols out of the global lookup namespace.
    let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        return Err(format!(
            "could not load NumPy OpenBLAS: {}",
            dlerror_message()
        ));
    }
    // SAFETY: `handle` is live, and `Api::load` validates every required symbol
    // before exposing any function pointer to callers.
    let api = match unsafe { Api::load(handle) } {
        Ok(api) => api,
        Err(error) => {
            // SAFETY: this handle was returned by `dlopen` above and has not
            // been published or closed.
            unsafe {
                libc::dlclose(handle);
            }
            return Err(error);
        }
    };
    if let Err(api) = API.set(api) {
        // Another thread won initialization. This duplicate handle has no
        // published function pointers and can be released safely.
        // SAFETY: `api.handle` is the still-live duplicate `dlopen` handle.
        unsafe {
            libc::dlclose(api.handle as *mut c_void);
        }
    }
    Ok(())
}

#[inline]
pub(super) fn is_loaded() -> bool {
    API.get().is_some()
}

#[inline]
fn api() -> &'static Api {
    API.get()
        .expect("Linux CBLAS call requires initialize_linux_openblas")
}

#[inline]
fn lp64(value: i64) -> i32 {
    debug_assert!((0..=i32::MAX as i64).contains(&value));
    value as i32
}

pub unsafe fn cblas_sdot(n: i64, x: *const f32, incx: i64, y: *const f32, incy: i64) -> f32 {
    // SAFETY: the public BLAS wrapper guarantees the pointer extents; LP64
    // chunking and stride validation keep every integer representable.
    unsafe { (api().sdot)(lp64(n), x, lp64(incx), y, lp64(incy)) }
}

pub unsafe fn cblas_ddot(n: i64, x: *const f64, incx: i64, y: *const f64, incy: i64) -> f64 {
    // SAFETY: as `cblas_sdot`, for f64 elements.
    unsafe { (api().ddot)(lp64(n), x, lp64(incx), y, lp64(incy)) }
}

macro_rules! complex_dot {
    ($function:ident, $field:ident) => {
        pub unsafe fn $function(
            n: i64,
            x: *const c_void,
            incx: i64,
            y: *const c_void,
            incy: i64,
            out: *mut c_void,
        ) {
            // SAFETY: the public BLAS wrapper guarantees both strided inputs
            // and one writable complex output; LP64 values were validated.
            unsafe { (api().$field)(lp64(n), x, lp64(incx), y, lp64(incy), out) }
        }
    };
}

complex_dot!(cblas_cdotu_sub, cdotu);
complex_dot!(cblas_zdotu_sub, zdotu);
complex_dot!(cblas_cdotc_sub, cdotc);
complex_dot!(cblas_zdotc_sub, zdotc);

macro_rules! real_gemv {
    ($function:ident, $field:ident, $ty:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $function(
            order: i32,
            trans: i32,
            m: i64,
            n: i64,
            alpha: $ty,
            a: *const $ty,
            lda: i64,
            x: *const $ty,
            incx: i64,
            beta: $ty,
            y: *mut $ty,
            incy: i64,
        ) {
            // SAFETY: the public wrapper guarantees the BLAS extents and
            // writable output; every integer fits the LP64 ABI.
            unsafe {
                (api().$field)(
                    order,
                    trans,
                    lp64(m),
                    lp64(n),
                    alpha,
                    a,
                    lp64(lda),
                    x,
                    lp64(incx),
                    beta,
                    y,
                    lp64(incy),
                )
            }
        }
    };
}

real_gemv!(cblas_sgemv, sgemv, f32);
real_gemv!(cblas_dgemv, dgemv, f64);

macro_rules! complex_gemv {
    ($function:ident, $field:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $function(
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
        ) {
            // SAFETY: the public wrapper guarantees all buffers and live
            // complex scalars; every integer fits the LP64 ABI.
            unsafe {
                (api().$field)(
                    order,
                    trans,
                    lp64(m),
                    lp64(n),
                    alpha,
                    a,
                    lp64(lda),
                    x,
                    lp64(incx),
                    beta,
                    y,
                    lp64(incy),
                )
            }
        }
    };
}

complex_gemv!(cblas_cgemv, cgemv);
complex_gemv!(cblas_zgemv, zgemv);

macro_rules! real_gemm {
    ($function:ident, $field:ident, $ty:ty) => {
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $function(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i64,
            n: i64,
            k: i64,
            alpha: $ty,
            a: *const $ty,
            lda: i64,
            b: *const $ty,
            ldb: i64,
            beta: $ty,
            c: *mut $ty,
            ldc: i64,
        ) {
            // SAFETY: the public wrapper guarantees all matrix extents and
            // writable output; every integer fits the LP64 ABI.
            unsafe {
                (api().$field)(
                    order,
                    trans_a,
                    trans_b,
                    lp64(m),
                    lp64(n),
                    lp64(k),
                    alpha,
                    a,
                    lp64(lda),
                    b,
                    lp64(ldb),
                    beta,
                    c,
                    lp64(ldc),
                )
            }
        }
    };
}

real_gemm!(cblas_sgemm, sgemm, f32);
real_gemm!(cblas_dgemm, dgemm, f64);

macro_rules! complex_gemm {
    ($function:ident, $field:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $function(
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
        ) {
            // SAFETY: the public wrapper guarantees matrix and scalar pointer
            // validity; every integer fits the LP64 ABI.
            unsafe {
                (api().$field)(
                    order,
                    trans_a,
                    trans_b,
                    lp64(m),
                    lp64(n),
                    lp64(k),
                    alpha,
                    a,
                    lp64(lda),
                    b,
                    lp64(ldb),
                    beta,
                    c,
                    lp64(ldc),
                )
            }
        }
    };
}

complex_gemm!(cblas_cgemm, cgemm);
complex_gemm!(cblas_zgemm, zgemm);
