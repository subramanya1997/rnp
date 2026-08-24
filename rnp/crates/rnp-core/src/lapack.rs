//! Apple Accelerate LAPACK kernels used by `numpy.linalg`.
//!
//! NumPy 2.5.2's wheel on this host imports the `$NEWLAPACK$ILP64`
//! variants, so every Fortran integer below is an `i64`.  Matrices are copied
//! into column-major scratch buffers before crossing the FFI boundary; this
//! also handles arbitrary and byte-swapped ndarray inputs without exposing
//! their storage directly to LAPACK.

use num_complex::Complex;

use crate::array::NdArray;
use crate::dtype::DType;
use crate::element::{C32, C64v, Scalar};
use crate::error::{Error, Result};
use crate::iter::{broadcast_shapes, broadcast_to, offsets};

/// True when this build can call Accelerate LAPACK.
pub const HAVE_LAPACK: bool = cfg!(all(target_os = "macos", target_vendor = "apple"));

#[cfg(all(target_os = "macos", target_vendor = "apple"))]
mod sys {
    use crate::element::{C32, C64v};

    // SAFETY: these declarations transcribe the Fortran LAPACK signatures in
    // NumPy's `linalg/umath_linalg.cpp`. The installed NumPy extension imports
    // these exact `$NEWLAPACK$ILP64` symbols, making every integer argument an
    // `i64`. Callers provide live column-major buffers sized from m/n/nrhs,
    // valid scalar/character arguments, and workspace arrays returned by a
    // LAPACK workspace query before invoking any routine below.
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[link_name = "sgesv$NEWLAPACK$ILP64"]
        pub fn sgesv(n: *const i64, nrhs: *const i64, a: *mut f32, lda: *const i64,
                     ipiv: *mut i64, b: *mut f32, ldb: *const i64, info: *mut i64);
        #[link_name = "dgesv$NEWLAPACK$ILP64"]
        pub fn dgesv(n: *const i64, nrhs: *const i64, a: *mut f64, lda: *const i64,
                     ipiv: *mut i64, b: *mut f64, ldb: *const i64, info: *mut i64);
        #[link_name = "cgesv$NEWLAPACK$ILP64"]
        pub fn cgesv(n: *const i64, nrhs: *const i64, a: *mut C32, lda: *const i64,
                     ipiv: *mut i64, b: *mut C32, ldb: *const i64, info: *mut i64);
        #[link_name = "zgesv$NEWLAPACK$ILP64"]
        pub fn zgesv(n: *const i64, nrhs: *const i64, a: *mut C64v, lda: *const i64,
                     ipiv: *mut i64, b: *mut C64v, ldb: *const i64, info: *mut i64);

        #[link_name = "sgetrf$NEWLAPACK$ILP64"]
        pub fn sgetrf(m: *const i64, n: *const i64, a: *mut f32, lda: *const i64,
                      ipiv: *mut i64, info: *mut i64);
        #[link_name = "dgetrf$NEWLAPACK$ILP64"]
        pub fn dgetrf(m: *const i64, n: *const i64, a: *mut f64, lda: *const i64,
                      ipiv: *mut i64, info: *mut i64);
        #[link_name = "cgetrf$NEWLAPACK$ILP64"]
        pub fn cgetrf(m: *const i64, n: *const i64, a: *mut C32, lda: *const i64,
                      ipiv: *mut i64, info: *mut i64);
        #[link_name = "zgetrf$NEWLAPACK$ILP64"]
        pub fn zgetrf(m: *const i64, n: *const i64, a: *mut C64v, lda: *const i64,
                      ipiv: *mut i64, info: *mut i64);

        #[link_name = "spotrf$NEWLAPACK$ILP64"]
        pub fn spotrf(uplo: *const u8, n: *const i64, a: *mut f32,
                      lda: *const i64, info: *mut i64);
        #[link_name = "dpotrf$NEWLAPACK$ILP64"]
        pub fn dpotrf(uplo: *const u8, n: *const i64, a: *mut f64,
                      lda: *const i64, info: *mut i64);
        #[link_name = "cpotrf$NEWLAPACK$ILP64"]
        pub fn cpotrf(uplo: *const u8, n: *const i64, a: *mut C32,
                      lda: *const i64, info: *mut i64);
        #[link_name = "zpotrf$NEWLAPACK$ILP64"]
        pub fn zpotrf(uplo: *const u8, n: *const i64, a: *mut C64v,
                      lda: *const i64, info: *mut i64);

        #[link_name = "sgelsd$NEWLAPACK$ILP64"]
        pub fn sgelsd(m: *const i64, n: *const i64, nrhs: *const i64,
                      a: *mut f32, lda: *const i64, b: *mut f32, ldb: *const i64,
                      s: *mut f32, rcond: *const f32, rank: *mut i64,
                      work: *mut f32, lwork: *const i64, iwork: *mut i64,
                      info: *mut i64);
        #[link_name = "dgelsd$NEWLAPACK$ILP64"]
        pub fn dgelsd(m: *const i64, n: *const i64, nrhs: *const i64,
                      a: *mut f64, lda: *const i64, b: *mut f64, ldb: *const i64,
                      s: *mut f64, rcond: *const f64, rank: *mut i64,
                      work: *mut f64, lwork: *const i64, iwork: *mut i64,
                      info: *mut i64);
        #[link_name = "cgelsd$NEWLAPACK$ILP64"]
        pub fn cgelsd(m: *const i64, n: *const i64, nrhs: *const i64,
                      a: *mut C32, lda: *const i64, b: *mut C32, ldb: *const i64,
                      s: *mut f32, rcond: *const f32, rank: *mut i64,
                      work: *mut C32, lwork: *const i64, rwork: *mut f32,
                      iwork: *mut i64, info: *mut i64);
        #[link_name = "zgelsd$NEWLAPACK$ILP64"]
        pub fn zgelsd(m: *const i64, n: *const i64, nrhs: *const i64,
                      a: *mut C64v, lda: *const i64, b: *mut C64v, ldb: *const i64,
                      s: *mut f64, rcond: *const f64, rank: *mut i64,
                      work: *mut C64v, lwork: *const i64, rwork: *mut f64,
                      iwork: *mut i64, info: *mut i64);

        #[link_name = "dgesdd$NEWLAPACK$ILP64"]
        pub fn dgesdd(jobz: *const u8, m: *const i64, n: *const i64,
                      a: *mut f64, lda: *const i64, s: *mut f64,
                      u: *mut f64, ldu: *const i64, vt: *mut f64,
                      ldvt: *const i64, work: *mut f64, lwork: *const i64,
                      iwork: *mut i64, info: *mut i64);
        #[link_name = "zgesdd$NEWLAPACK$ILP64"]
        pub fn zgesdd(jobz: *const u8, m: *const i64, n: *const i64,
                      a: *mut C64v, lda: *const i64, s: *mut f64,
                      u: *mut C64v, ldu: *const i64, vt: *mut C64v,
                      ldvt: *const i64, work: *mut C64v, lwork: *const i64,
                      rwork: *mut f64, iwork: *mut i64, info: *mut i64);

        #[link_name = "dsyevd$NEWLAPACK$ILP64"]
        pub fn dsyevd(jobz: *const u8, uplo: *const u8, n: *const i64,
                      a: *mut f64, lda: *const i64, w: *mut f64,
                      work: *mut f64, lwork: *const i64, iwork: *mut i64,
                      liwork: *const i64, info: *mut i64);
        #[link_name = "zheevd$NEWLAPACK$ILP64"]
        pub fn zheevd(jobz: *const u8, uplo: *const u8, n: *const i64,
                      a: *mut C64v, lda: *const i64, w: *mut f64,
                      work: *mut C64v, lwork: *const i64, rwork: *mut f64,
                      lrwork: *const i64, iwork: *mut i64,
                      liwork: *const i64, info: *mut i64);

        #[link_name = "dgeev$NEWLAPACK$ILP64"]
        pub fn dgeev(jobvl: *const u8, jobvr: *const u8, n: *const i64,
                     a: *mut f64, lda: *const i64, wr: *mut f64, wi: *mut f64,
                     vl: *mut f64, ldvl: *const i64, vr: *mut f64,
                     ldvr: *const i64, work: *mut f64, lwork: *const i64,
                     info: *mut i64);
        #[link_name = "zgeev$NEWLAPACK$ILP64"]
        pub fn zgeev(jobvl: *const u8, jobvr: *const u8, n: *const i64,
                     a: *mut C64v, lda: *const i64, w: *mut C64v,
                     vl: *mut C64v, ldvl: *const i64, vr: *mut C64v,
                     ldvr: *const i64, work: *mut C64v, lwork: *const i64,
                     rwork: *mut f64, info: *mut i64);

        #[link_name = "dgeqrf$NEWLAPACK$ILP64"]
        pub fn dgeqrf(m: *const i64, n: *const i64, a: *mut f64,
                      lda: *const i64, tau: *mut f64, work: *mut f64,
                      lwork: *const i64, info: *mut i64);
        #[link_name = "zgeqrf$NEWLAPACK$ILP64"]
        pub fn zgeqrf(m: *const i64, n: *const i64, a: *mut C64v,
                      lda: *const i64, tau: *mut C64v, work: *mut C64v,
                      lwork: *const i64, info: *mut i64);
        #[link_name = "dorgqr$NEWLAPACK$ILP64"]
        pub fn dorgqr(m: *const i64, n: *const i64, k: *const i64,
                      a: *mut f64, lda: *const i64, tau: *mut f64,
                      work: *mut f64, lwork: *const i64, info: *mut i64);
        #[link_name = "zungqr$NEWLAPACK$ILP64"]
        pub fn zungqr(m: *const i64, n: *const i64, k: *const i64,
                      a: *mut C64v, lda: *const i64, tau: *mut C64v,
                      work: *mut C64v, lwork: *const i64, info: *mut i64);
    }
}

trait LapackScalar: Copy + Default {
    const DTYPE: DType;
    fn from_scalar(value: Scalar) -> Self;
    fn to_scalar(self) -> Scalar;
    fn one() -> Self;
    fn nan() -> Self;
    fn gesv(n: i64, nrhs: i64, a: &mut [Self], piv: &mut [i64], b: &mut [Self]) -> i64;
    fn getrf(n: i64, a: &mut [Self], piv: &mut [i64]) -> i64;
    fn potrf(uplo: u8, n: i64, a: &mut [Self]) -> i64;
    fn slog_diagonal(a: &[Self], piv: &[i64], n: usize) -> (Self, f64);
    fn det_from_slog(sign: Self, logabs: f64) -> Self;
}

macro_rules! impl_real_lapack {
    ($t:ty, $dtype:expr, $gesv:path, $getrf:path, $potrf:path) => {
        impl LapackScalar for $t {
            const DTYPE: DType = $dtype;
            fn from_scalar(value: Scalar) -> Self { value.as_f64() as $t }
            fn to_scalar(self) -> Scalar { Scalar::Float(self as f64) }
            fn one() -> Self { 1.0 }
            fn nan() -> Self { <$t>::NAN }
            fn gesv(n: i64, nrhs: i64, a: &mut [Self], piv: &mut [i64], b: &mut [Self]) -> i64 {
                let lda = n.max(1);
                let ldb = n.max(1);
                let mut info = 0;
                // SAFETY: `a`, `piv`, and `b` are column-major scratch arrays
                // sized n*n, n, and n*nrhs by the generic caller.
                unsafe { $gesv(&n, &nrhs, a.as_mut_ptr(), &lda, piv.as_mut_ptr(),
                               b.as_mut_ptr(), &ldb, &mut info) };
                info
            }
            fn getrf(n: i64, a: &mut [Self], piv: &mut [i64]) -> i64 {
                let lda = n.max(1);
                let mut info = 0;
                // SAFETY: `a` holds n*n elements and `piv` holds n elements.
                unsafe { $getrf(&n, &n, a.as_mut_ptr(), &lda, piv.as_mut_ptr(), &mut info) };
                info
            }
            fn potrf(uplo: u8, n: i64, a: &mut [Self]) -> i64 {
                let lda = n.max(1);
                let mut info = 0;
                // SAFETY: `a` holds n*n elements and `uplo` is `L` or `U`.
                unsafe { $potrf(&uplo, &n, a.as_mut_ptr(), &lda, &mut info) };
                info
            }
            fn slog_diagonal(a: &[Self], piv: &[i64], n: usize) -> (Self, f64) {
                let mut sign: $t = if piv.iter().enumerate().filter(|&(i, p)| *p != i as i64 + 1).count() % 2 == 0 { 1.0 } else { -1.0 };
                let mut logabs: $t = 0.0;
                for i in 0..n {
                    let mut d = a[i * (n + 1)];
                    if d < 0.0 { sign = -sign; d = -d; }
                    logabs += d.ln();
                }
                (sign, logabs as f64)
            }
            fn det_from_slog(sign: Self, logabs: f64) -> Self {
                sign * (logabs as $t).exp()
            }
        }
    };
}

macro_rules! impl_complex_lapack {
    ($t:ty, $real:ty, $dtype:expr, $gesv:path, $getrf:path, $potrf:path) => {
        impl LapackScalar for $t {
            const DTYPE: DType = $dtype;
            fn from_scalar(value: Scalar) -> Self {
                match value {
                    Scalar::Complex(z) => Complex::new(z.re as $real, z.im as $real),
                    other => Complex::new(other.as_f64() as $real, 0.0),
                }
            }
            fn to_scalar(self) -> Scalar {
                Scalar::Complex(C64v::new(self.re as f64, self.im as f64))
            }
            fn one() -> Self { Complex::new(1.0, 0.0) }
            fn nan() -> Self { Complex::new(<$real>::NAN, <$real>::NAN) }
            fn gesv(n: i64, nrhs: i64, a: &mut [Self], piv: &mut [i64], b: &mut [Self]) -> i64 {
                let lda = n.max(1);
                let ldb = n.max(1);
                let mut info = 0;
                // SAFETY: `a`, `piv`, and `b` are column-major scratch arrays
                // sized n*n, n, and n*nrhs by the generic caller.
                unsafe { $gesv(&n, &nrhs, a.as_mut_ptr(), &lda, piv.as_mut_ptr(),
                               b.as_mut_ptr(), &ldb, &mut info) };
                info
            }
            fn getrf(n: i64, a: &mut [Self], piv: &mut [i64]) -> i64 {
                let lda = n.max(1);
                let mut info = 0;
                // SAFETY: `a` holds n*n elements and `piv` holds n elements.
                unsafe { $getrf(&n, &n, a.as_mut_ptr(), &lda, piv.as_mut_ptr(), &mut info) };
                info
            }
            fn potrf(uplo: u8, n: i64, a: &mut [Self]) -> i64 {
                let lda = n.max(1);
                let mut info = 0;
                // SAFETY: `a` holds n*n elements and `uplo` is `L` or `U`.
                unsafe { $potrf(&uplo, &n, a.as_mut_ptr(), &lda, &mut info) };
                info
            }
            fn slog_diagonal(a: &[Self], piv: &[i64], n: usize) -> (Self, f64) {
                let parity = piv.iter().enumerate().filter(|&(i, p)| *p != i as i64 + 1).count();
                let mut sign: $t = Complex::new(if parity % 2 == 0 { 1.0 } else { -1.0 }, 0.0);
                let mut logabs: $real = 0.0;
                for i in 0..n {
                    let d = a[i * (n + 1)];
                    let mag = d.norm();
                    sign *= d / mag;
                    logabs += mag.ln();
                }
                (sign, logabs as f64)
            }
            fn det_from_slog(sign: Self, logabs: f64) -> Self {
                sign * (logabs as $real).exp()
            }
        }
    };
}

#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl_real_lapack!(f32, DType::F32, sys::sgesv, sys::sgetrf, sys::spotrf);
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl_real_lapack!(f64, DType::F64, sys::dgesv, sys::dgetrf, sys::dpotrf);
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl_complex_lapack!(C32, f32, DType::C64, sys::cgesv, sys::cgetrf, sys::cpotrf);
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl_complex_lapack!(C64v, f64, DType::C128, sys::zgesv, sys::zgetrf, sys::zpotrf);

trait GelsdScalar: LapackScalar {
    fn gelsd(m: i64, n: i64, nrhs: i64, a: &mut [Self], b: &mut [Self], rcond: f64)
        -> (i64, i64, Vec<f64>);
    fn abs2(self) -> f64;
}

macro_rules! impl_real_gelsd {
    ($t:ty, $call:path) => {
        impl GelsdScalar for $t {
            fn gelsd(m: i64, n: i64, nrhs: i64, a: &mut [Self], b: &mut [Self], rcond: f64)
                -> (i64, i64, Vec<f64>) {
                let lda = m.max(1);
                let ldb = m.max(n).max(1);
                let mut s = vec![0 as $t; (m.min(n) as usize).max(1)];
                let rcond = rcond as $t;
                let mut rank = 0i64;
                let mut info = 0i64;
                let mut work_query = 0 as $t;
                let mut iwork_query = 0i64;
                let query = -1i64;
                // SAFETY: all fixed-size arrays satisfy LAPACK's leading
                // dimensions; lwork=-1 makes this a workspace query.
                unsafe { $call(&m, &n, &nrhs, a.as_mut_ptr(), &lda, b.as_mut_ptr(),
                               &ldb, s.as_mut_ptr(), &rcond, &mut rank,
                               &mut work_query, &query, &mut iwork_query, &mut info) };
                if info != 0 { return (info, rank, Vec::new()); }
                let lwork = (work_query as i64).max(1);
                let mut work = vec![0 as $t; lwork as usize];
                let mut iwork = vec![0i64; iwork_query.max(1) as usize];
                // SAFETY: workspace sizes are the values returned by the
                // preceding query and all data buffers retain their sizes.
                unsafe { $call(&m, &n, &nrhs, a.as_mut_ptr(), &lda, b.as_mut_ptr(),
                               &ldb, s.as_mut_ptr(), &rcond, &mut rank,
                               work.as_mut_ptr(), &lwork, iwork.as_mut_ptr(), &mut info) };
                s.truncate(m.min(n) as usize);
                (info, rank, s.into_iter().map(|v| v as f64).collect())
            }
            fn abs2(self) -> f64 { let v = self as f64; v * v }
        }
    };
}

macro_rules! impl_complex_gelsd {
    ($t:ty, $real:ty, $call:path) => {
        impl GelsdScalar for $t {
            fn gelsd(m: i64, n: i64, nrhs: i64, a: &mut [Self], b: &mut [Self], rcond: f64)
                -> (i64, i64, Vec<f64>) {
                let lda = m.max(1);
                let ldb = m.max(n).max(1);
                let mut s = vec![0 as $real; (m.min(n) as usize).max(1)];
                let rcond = rcond as $real;
                let mut rank = 0i64;
                let mut info = 0i64;
                let mut work_query: $t = Complex::new(0.0, 0.0);
                let mut rwork_query = 0 as $real;
                let mut iwork_query = 0i64;
                let query = -1i64;
                // SAFETY: all fixed-size arrays satisfy LAPACK's leading
                // dimensions; lwork=-1 makes this a workspace query.
                unsafe { $call(&m, &n, &nrhs, a.as_mut_ptr(), &lda, b.as_mut_ptr(),
                               &ldb, s.as_mut_ptr(), &rcond, &mut rank,
                               &mut work_query, &query, &mut rwork_query,
                               &mut iwork_query, &mut info) };
                if info != 0 { return (info, rank, Vec::new()); }
                let lwork = (work_query.re as i64).max(1);
                let mut work = vec![Complex::new(0.0, 0.0); lwork as usize];
                let mut rwork = vec![0 as $real; (rwork_query as i64).max(1) as usize];
                let mut iwork = vec![0i64; iwork_query.max(1) as usize];
                // SAFETY: workspace sizes are the values returned by the
                // preceding query and all data buffers retain their sizes.
                unsafe { $call(&m, &n, &nrhs, a.as_mut_ptr(), &lda, b.as_mut_ptr(),
                               &ldb, s.as_mut_ptr(), &rcond, &mut rank,
                               work.as_mut_ptr(), &lwork, rwork.as_mut_ptr(),
                               iwork.as_mut_ptr(), &mut info) };
                s.truncate(m.min(n) as usize);
                (info, rank, s.into_iter().map(|v| v as f64).collect())
            }
            fn abs2(self) -> f64 { self.re as f64 * self.re as f64 + self.im as f64 * self.im as f64 }
        }
    };
}

#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl_real_gelsd!(f32, sys::sgelsd);
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl_real_gelsd!(f64, sys::dgelsd);
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl_complex_gelsd!(C32, f32, sys::cgelsd);
#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl_complex_gelsd!(C64v, f64, sys::zgelsd);

trait QrScalar: LapackScalar {
    fn geqrf(m: i64, n: i64, a: &mut [Self], tau: &mut [Self]) -> i64;
    fn gqr(m: i64, n: i64, k: i64, a: &mut [Self], tau: &mut [Self]) -> i64;
}

#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl QrScalar for f64 {
    fn geqrf(m: i64, n: i64, a: &mut [Self], tau: &mut [Self]) -> i64 {
        let lda=m.max(1);let mut query=0.0;let q=-1i64;let mut info=0i64;
        // SAFETY: lwork=-1 query with m*n matrix and min(m,n) tau buffers.
        unsafe{sys::dgeqrf(&m,&n,a.as_mut_ptr(),&lda,tau.as_mut_ptr(),&mut query,&q,&mut info)};
        if info!=0{return info;}let lw=(query as i64).max(n.max(1));let mut work=vec![0.0;lw as usize];
        // SAFETY: `work` has the queried size and all data extents are unchanged.
        unsafe{sys::dgeqrf(&m,&n,a.as_mut_ptr(),&lda,tau.as_mut_ptr(),work.as_mut_ptr(),&lw,&mut info)};info
    }
    fn gqr(m:i64,n:i64,k:i64,a:&mut[Self],tau:&mut[Self])->i64{
        let lda=m.max(1);let mut query=0.0;let q=-1i64;let mut info=0i64;
        // SAFETY: lwork=-1 query with m*n Q storage and k reflector scalars.
        unsafe{sys::dorgqr(&m,&n,&k,a.as_mut_ptr(),&lda,tau.as_mut_ptr(),&mut query,&q,&mut info)};
        if info!=0{return info;}let lw=(query as i64).max(n.max(1));let mut work=vec![0.0;lw as usize];
        // SAFETY: `work` has the queried size and all other extents are unchanged.
        unsafe{sys::dorgqr(&m,&n,&k,a.as_mut_ptr(),&lda,tau.as_mut_ptr(),work.as_mut_ptr(),&lw,&mut info)};info
    }
}

#[cfg(all(target_os = "macos", target_vendor = "apple"))]
impl QrScalar for C64v {
    fn geqrf(m:i64,n:i64,a:&mut[Self],tau:&mut[Self])->i64{
        let lda=m.max(1);let mut query=C64v::new(0.0,0.0);let q=-1i64;let mut info=0i64;
        // SAFETY: lwork=-1 query with m*n matrix and min(m,n) tau buffers.
        unsafe{sys::zgeqrf(&m,&n,a.as_mut_ptr(),&lda,tau.as_mut_ptr(),&mut query,&q,&mut info)};
        if info!=0{return info;}let lw=(query.re as i64).max(n.max(1));let mut work=vec![C64v::new(0.0,0.0);lw as usize];
        // SAFETY: `work` has the queried size and all data extents are unchanged.
        unsafe{sys::zgeqrf(&m,&n,a.as_mut_ptr(),&lda,tau.as_mut_ptr(),work.as_mut_ptr(),&lw,&mut info)};info
    }
    fn gqr(m:i64,n:i64,k:i64,a:&mut[Self],tau:&mut[Self])->i64{
        let lda=m.max(1);let mut query=C64v::new(0.0,0.0);let q=-1i64;let mut info=0i64;
        // SAFETY: lwork=-1 query with m*n Q storage and k reflector scalars.
        unsafe{sys::zungqr(&m,&n,&k,a.as_mut_ptr(),&lda,tau.as_mut_ptr(),&mut query,&q,&mut info)};
        if info!=0{return info;}let lw=(query.re as i64).max(n.max(1));let mut work=vec![C64v::new(0.0,0.0);lw as usize];
        // SAFETY: `work` has the queried size and all other extents are unchanged.
        unsafe{sys::zungqr(&m,&n,&k,a.as_mut_ptr(),&lda,tau.as_mut_ptr(),work.as_mut_ptr(),&lw,&mut info)};info
    }
}

fn unavailable() -> Error {
    Error::NotImplemented("numpy.linalg requires Apple Accelerate LAPACK".into())
}

fn matrix_to_col_major<T: LapackScalar>(a: &NdArray, base: isize, rows: usize, cols: usize) -> Vec<T> {
    let rs = a.strides[a.ndim() - 2];
    let cs = a.strides[a.ndim() - 1];
    let mut out = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            out.push(T::from_scalar(a.read_at(base + row as isize * rs + col as isize * cs)));
        }
    }
    out
}

fn rhs_to_col_major<T: LapackScalar>(b: &NdArray, base: isize, rows: usize, cols: usize, vector: bool) -> Vec<T> {
    if vector {
        let stride = b.strides[b.ndim() - 1];
        return (0..rows).map(|row| T::from_scalar(b.read_at(base + row as isize * stride))).collect();
    }
    matrix_to_col_major::<T>(b, base, rows, cols)
}

fn solve_impl<T: LapackScalar>(
    a: &NdArray,
    b: &NdArray,
    vector: bool,
    singular_nan: bool,
) -> Result<NdArray> {
    let n = a.shape[a.ndim() - 1] as usize;
    let b_rows = if vector { b.shape[b.ndim() - 1] } else { b.shape[b.ndim() - 2] };
    if b_rows != n as isize {
        return Err(Error::ValueError(format!(
            "solve: Input operand 1 has a mismatch in its core dimension 0 (size {b_rows} is different from {n})"
        )));
    }
    let nrhs = if vector { 1 } else { b.shape[b.ndim() - 1] as usize };
    let a_batch = &a.shape[..a.ndim() - 2];
    let b_batch = if vector { &b.shape[..b.ndim() - 1] } else { &b.shape[..b.ndim() - 2] };
    let batch = broadcast_shapes(a_batch, b_batch)?;

    let mut ashape = batch.clone();
    ashape.extend([n as isize, n as isize]);
    let mut bshape = batch.clone();
    bshape.push(n as isize);
    if !vector { bshape.push(nrhs as isize); }
    let av = broadcast_to(a, &ashape)?;
    let bv = broadcast_to(b, &bshape)?;

    let mut out = NdArray::zeros(bshape, T::DTYPE)?;
    let ao = offsets(&batch, &av.strides[..batch.len()], av.byte_offset);
    let bo = offsets(&batch, &bv.strides[..batch.len()], bv.byte_offset);
    for (batch_i, (abase, bbase)) in ao.zip(bo).enumerate() {
        if n == 0 { continue; }
        let mut aa = matrix_to_col_major::<T>(&av, abase, n, n);
        let mut bb = rhs_to_col_major::<T>(&bv, bbase, n, nrhs, vector);
        let mut piv = vec![0i64; n];
        let info = T::gesv(n as i64, nrhs as i64, &mut aa, &mut piv, &mut bb);
        if info != 0 {
            if !singular_nan {
                return Err(Error::ValueError("Singular matrix".into()));
            }
            for row in 0..n {
                for col in 0..nrhs {
                    let flat = if vector { batch_i * n + row } else { batch_i * n * nrhs + row * nrhs + col };
                    out.set_flat(flat, T::nan().to_scalar());
                }
            }
            continue;
        }
        for row in 0..n {
            for col in 0..nrhs {
                let src = bb[row + col * n];
                let flat = if vector { batch_i * n + row } else { batch_i * n * nrhs + row * nrhs + col };
                out.set_flat(flat, src.to_scalar());
            }
        }
    }
    Ok(out)
}

/// Solve `a @ x = b`, including gufunc-style broadcasting over matrix stacks.
pub fn solve(a: &NdArray, b: &NdArray, vector: bool, dtype: DType) -> Result<NdArray> {
    if !HAVE_LAPACK { return Err(unavailable()); }
    match dtype {
        DType::F32 => solve_impl::<f32>(a, b, vector, false),
        DType::F64 => solve_impl::<f64>(a, b, vector, false),
        DType::C64 => solve_impl::<C32>(a, b, vector, false),
        DType::C128 => solve_impl::<C64v>(a, b, vector, false),
        _ => Err(Error::TypeError("unsupported LAPACK dtype".into())),
    }
}

/// Invert square matrices by solving against an identity matrix.
pub fn inv(a: &NdArray, dtype: DType, singular_nan: bool) -> Result<NdArray> {
    let n = a.shape[a.ndim() - 1];
    let mut identity = NdArray::zeros(vec![n, n], dtype)?;
    for i in 0..n as usize { identity.set_flat(i * n as usize + i, Scalar::Int(1)); }
    if !HAVE_LAPACK { return Err(unavailable()); }
    match dtype {
        DType::F32 => solve_impl::<f32>(a, &identity, false, singular_nan),
        DType::F64 => solve_impl::<f64>(a, &identity, false, singular_nan),
        DType::C64 => solve_impl::<C32>(a, &identity, false, singular_nan),
        DType::C128 => solve_impl::<C64v>(a, &identity, false, singular_nan),
        _ => Err(Error::TypeError("unsupported LAPACK dtype".into())),
    }
}

fn slogdet_impl<T: LapackScalar>(a: &NdArray) -> Result<(NdArray, NdArray)> {
    let n = a.shape[a.ndim() - 1] as usize;
    let batch = a.shape[..a.ndim() - 2].to_vec();
    let mut sign = NdArray::zeros(batch.clone(), T::DTYPE)?;
    let mut logabs = NdArray::zeros(batch.clone(), if T::DTYPE == DType::F32 || T::DTYPE == DType::C64 { DType::F32 } else { DType::F64 })?;
    let ao = offsets(&batch, &a.strides[..batch.len()], a.byte_offset);
    for (batch_i, abase) in ao.enumerate() {
        if n == 0 {
            sign.set_flat(batch_i, T::one().to_scalar());
            continue;
        }
        let mut aa = matrix_to_col_major::<T>(a, abase, n, n);
        let mut piv = vec![0i64; n];
        if T::getrf(n as i64, &mut aa, &mut piv) != 0 {
            sign.set_flat(batch_i, Scalar::Int(0));
            logabs.set_flat(batch_i, Scalar::Float(f64::NEG_INFINITY));
            continue;
        }
        let (s, l) = T::slog_diagonal(&aa, &piv, n);
        sign.set_flat(batch_i, s.to_scalar());
        logabs.set_flat(batch_i, Scalar::Float(l));
    }
    Ok((sign, logabs))
}

pub fn slogdet(a: &NdArray, dtype: DType) -> Result<(NdArray, NdArray)> {
    if !HAVE_LAPACK { return Err(unavailable()); }
    match dtype {
        DType::F32 => slogdet_impl::<f32>(a), DType::F64 => slogdet_impl::<f64>(a),
        DType::C64 => slogdet_impl::<C32>(a), DType::C128 => slogdet_impl::<C64v>(a),
        _ => Err(Error::TypeError("unsupported LAPACK dtype".into())),
    }
}

fn det_impl<T: LapackScalar>(a: &NdArray) -> Result<NdArray> {
    let (sign, logabs) = slogdet_impl::<T>(a)?;
    let mut out = NdArray::zeros(sign.shape.clone(), T::DTYPE)?;
    for i in 0..out.size() {
        let s = T::from_scalar(sign.get_flat(i));
        out.set_flat(i, T::det_from_slog(s, logabs.get_flat(i).as_f64()).to_scalar());
    }
    Ok(out)
}

pub fn det(a: &NdArray, dtype: DType) -> Result<NdArray> {
    if !HAVE_LAPACK { return Err(unavailable()); }
    match dtype {
        DType::F32 => det_impl::<f32>(a), DType::F64 => det_impl::<f64>(a),
        DType::C64 => det_impl::<C32>(a), DType::C128 => det_impl::<C64v>(a),
        _ => Err(Error::TypeError("unsupported LAPACK dtype".into())),
    }
}

fn cholesky_impl<T: LapackScalar>(a: &NdArray, upper: bool) -> Result<NdArray> {
    let n = a.shape[a.ndim() - 1] as usize;
    let batch = a.shape[..a.ndim() - 2].to_vec();
    let mut out = NdArray::zeros(a.shape.clone(), T::DTYPE)?;
    let ao = offsets(&batch, &a.strides[..batch.len()], a.byte_offset);
    for (batch_i, abase) in ao.enumerate() {
        if n == 0 { continue; }
        let mut aa = matrix_to_col_major::<T>(a, abase, n, n);
        if T::potrf(if upper { b'U' } else { b'L' }, n as i64, &mut aa) != 0 {
            return Err(Error::ValueError("Matrix is not positive definite".into()));
        }
        for row in 0..n {
            for col in 0..n {
                let value = if (upper && col >= row) || (!upper && row >= col) {
                    aa[row + col * n].to_scalar()
                } else { Scalar::Int(0) };
                out.set_flat(batch_i * n * n + row * n + col, value);
            }
        }
    }
    Ok(out)
}

pub fn cholesky(a: &NdArray, upper: bool, dtype: DType) -> Result<NdArray> {
    if !HAVE_LAPACK { return Err(unavailable()); }
    match dtype {
        DType::F32 => cholesky_impl::<f32>(a, upper), DType::F64 => cholesky_impl::<f64>(a, upper),
        DType::C64 => cholesky_impl::<C32>(a, upper), DType::C128 => cholesky_impl::<C64v>(a, upper),
        _ => Err(Error::TypeError("unsupported LAPACK dtype".into())),
    }
}

pub struct LstsqResult {
    pub x: NdArray,
    pub residuals: NdArray,
    pub rank: i64,
    pub singular_values: NdArray,
}

fn lstsq_impl<T: GelsdScalar>(a: &NdArray, b: &NdArray, rcond: f64) -> Result<LstsqResult> {
    let m = a.shape[0] as usize;
    let n = a.shape[1] as usize;
    let nrhs = b.shape[1] as usize;
    let ldb = m.max(n).max(1);
    let mut aa = matrix_to_col_major::<T>(a, a.byte_offset, m, n);
    if aa.is_empty() { aa.push(T::default()); }
    let mut bb = vec![T::default(); (ldb * nrhs).max(1)];
    for col in 0..nrhs {
        for row in 0..m {
            bb[row + col * ldb] = T::from_scalar(
                b.read_at(b.byte_offset + row as isize * b.strides[0] + col as isize * b.strides[1]));
        }
    }
    let (info, rank, values) = T::gelsd(m as i64, n as i64, nrhs as i64, &mut aa, &mut bb, rcond);
    if info != 0 {
        return Err(Error::ValueError("SVD did not converge in Linear Least Squares".into()));
    }
    let mut x = NdArray::zeros(vec![n as isize, nrhs as isize], T::DTYPE)?;
    for row in 0..n {
        for col in 0..nrhs {
            x.set_flat(row * nrhs + col, bb[row + col * ldb].to_scalar());
        }
    }
    let real_dtype = if T::DTYPE == DType::F32 || T::DTYPE == DType::C64 { DType::F32 } else { DType::F64 };
    let mut residuals = NdArray::zeros(vec![nrhs as isize], real_dtype)?;
    if m >= n && rank == n as i64 {
        for col in 0..nrhs {
            let value = (n..m).map(|row| bb[row + col * ldb].abs2()).sum();
            residuals.set_flat(col, Scalar::Float(value));
        }
    } else {
        for col in 0..nrhs { residuals.set_flat(col, Scalar::Float(f64::NAN)); }
    }
    let mut singular_values = NdArray::zeros(vec![values.len() as isize], real_dtype)?;
    for (i, value) in values.into_iter().enumerate() {
        singular_values.set_flat(i, Scalar::Float(value));
    }
    Ok(LstsqResult { x, residuals, rank, singular_values })
}

pub fn lstsq(a: &NdArray, b: &NdArray, rcond: f64, dtype: DType) -> Result<LstsqResult> {
    if !HAVE_LAPACK { return Err(unavailable()); }
    match dtype {
        DType::F32 => lstsq_impl::<f32>(a, b, rcond), DType::F64 => lstsq_impl::<f64>(a, b, rcond),
        DType::C64 => lstsq_impl::<C32>(a, b, rcond), DType::C128 => lstsq_impl::<C64v>(a, b, rcond),
        _ => Err(Error::TypeError("unsupported LAPACK dtype".into())),
    }
}

pub struct SvdResult {
    pub u: Option<NdArray>,
    pub singular_values: NdArray,
    pub vh: Option<NdArray>,
}

fn svd_real_one(mut a: Vec<f64>, m: usize, n: usize, jobz: u8)
    -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let k = m.min(n);
    let (ucols, vtrows) = match jobz { b'A' => (m, n), b'S' => (k, k), _ => (0, 0) };
    if k == 0 {
        let mut u = vec![0.0; m * ucols];
        let mut vt = vec![0.0; vtrows * n];
        if jobz == b'A' {
            for i in 0..m { u[i + i * m] = 1.0; }
            for i in 0..n { vt[i + i * n] = 1.0; }
        }
        return Ok((u, Vec::new(), vt));
    }
    let lda = (m as i64).max(1);
    let ldu = lda;
    let ldvt = (vtrows as i64).max(1);
    let mut s = vec![0.0; k];
    let mut u = vec![0.0; (m * ucols).max(1)];
    let mut vt = vec![0.0; (vtrows * n).max(1)];
    let mut iwork = vec![0i64; (8 * k).max(1)];
    let mut query = 0.0;
    let mut info = 0i64;
    let query_len = -1i64;
    // SAFETY: all data arrays use the LAPACK dimensions above; lwork=-1 is
    // the documented workspace query and the query scalar is writable.
    unsafe { sys::dgesdd(&jobz, &(m as i64), &(n as i64), a.as_mut_ptr(), &lda,
                         s.as_mut_ptr(), u.as_mut_ptr(), &ldu, vt.as_mut_ptr(),
                         &ldvt, &mut query, &query_len, iwork.as_mut_ptr(), &mut info) };
    if info != 0 { return Err(Error::ValueError("SVD did not converge".into())); }
    let lwork = (query as i64).max(1);
    let mut work = vec![0.0; lwork as usize];
    // SAFETY: `work` has the queried size and every other buffer is unchanged.
    unsafe { sys::dgesdd(&jobz, &(m as i64), &(n as i64), a.as_mut_ptr(), &lda,
                         s.as_mut_ptr(), u.as_mut_ptr(), &ldu, vt.as_mut_ptr(),
                         &ldvt, work.as_mut_ptr(), &lwork, iwork.as_mut_ptr(), &mut info) };
    if info != 0 { return Err(Error::ValueError("SVD did not converge".into())); }
    u.truncate(m * ucols);
    vt.truncate(vtrows * n);
    Ok((u, s, vt))
}

fn svd_complex_one(mut a: Vec<C64v>, m: usize, n: usize, jobz: u8)
    -> Result<(Vec<C64v>, Vec<f64>, Vec<C64v>)> {
    let k = m.min(n);
    let (ucols, vtrows) = match jobz { b'A' => (m, n), b'S' => (k, k), _ => (0, 0) };
    if k == 0 {
        let mut u = vec![C64v::new(0.0, 0.0); m * ucols];
        let mut vt = vec![C64v::new(0.0, 0.0); vtrows * n];
        if jobz == b'A' {
            for i in 0..m { u[i + i * m] = C64v::new(1.0, 0.0); }
            for i in 0..n { vt[i + i * n] = C64v::new(1.0, 0.0); }
        }
        return Ok((u, Vec::new(), vt));
    }
    let lda = (m as i64).max(1);
    let ldu = lda;
    let ldvt = (vtrows as i64).max(1);
    let mut s = vec![0.0; k];
    let mut u = vec![C64v::new(0.0, 0.0); (m * ucols).max(1)];
    let mut vt = vec![C64v::new(0.0, 0.0); (vtrows * n).max(1)];
    let mut iwork = vec![0i64; (8 * k).max(1)];
    let rcount = if jobz == b'N' { 7 * k } else { 5 * k * k + 5 * k };
    let mut rwork = vec![0.0; (2 * rcount).max(1)];
    let mut query = C64v::new(0.0, 0.0);
    let mut info = 0i64;
    let query_len = -1i64;
    // SAFETY: as the real query, with complex element and real workspace.
    unsafe { sys::zgesdd(&jobz, &(m as i64), &(n as i64), a.as_mut_ptr(), &lda,
                         s.as_mut_ptr(), u.as_mut_ptr(), &ldu, vt.as_mut_ptr(),
                         &ldvt, &mut query, &query_len, rwork.as_mut_ptr(),
                         iwork.as_mut_ptr(), &mut info) };
    if info != 0 { return Err(Error::ValueError("SVD did not converge".into())); }
    let lwork = (query.re as i64).max(1);
    let mut work = vec![C64v::new(0.0, 0.0); lwork as usize];
    // SAFETY: `work` has the queried size and all other buffers retain their extents.
    unsafe { sys::zgesdd(&jobz, &(m as i64), &(n as i64), a.as_mut_ptr(), &lda,
                         s.as_mut_ptr(), u.as_mut_ptr(), &ldu, vt.as_mut_ptr(),
                         &ldvt, work.as_mut_ptr(), &lwork, rwork.as_mut_ptr(),
                         iwork.as_mut_ptr(), &mut info) };
    if info != 0 { return Err(Error::ValueError("SVD did not converge".into())); }
    u.truncate(m * ucols);
    vt.truncate(vtrows * n);
    Ok((u, s, vt))
}

fn write_col_major<T: LapackScalar>(out: &mut NdArray, batch_i: usize,
                                     data: &[T], rows: usize, cols: usize) {
    for row in 0..rows {
        for col in 0..cols {
            out.set_flat(batch_i * rows * cols + row * cols + col,
                         data[row + col * rows].to_scalar());
        }
    }
}

pub fn svd(a: &NdArray, full: bool, vectors: bool, complex: bool) -> Result<SvdResult> {
    if !HAVE_LAPACK { return Err(unavailable()); }
    let m = a.shape[a.ndim() - 2] as usize;
    let n = a.shape[a.ndim() - 1] as usize;
    let k = m.min(n);
    let batch = a.shape[..a.ndim() - 2].to_vec();
    let jobz = if !vectors { b'N' } else if full { b'A' } else { b'S' };
    let (ucols, vtrows) = if full { (m, n) } else { (k, k) };
    let mut sshape = batch.clone(); sshape.push(k as isize);
    let mut singular_values = NdArray::zeros(sshape, DType::F64)?;
    let mut u = if vectors { let mut sh=batch.clone(); sh.extend([m as isize, ucols as isize]); Some(NdArray::zeros(sh, if complex {DType::C128}else{DType::F64})?) } else { None };
    let mut vh = if vectors { let mut sh=batch.clone(); sh.extend([vtrows as isize, n as isize]); Some(NdArray::zeros(sh, if complex {DType::C128}else{DType::F64})?) } else { None };
    let ao = offsets(&batch, &a.strides[..batch.len()], a.byte_offset);
    for (batch_i, abase) in ao.enumerate() {
        if complex {
            let aa = matrix_to_col_major::<C64v>(a, abase, m, n);
            let (uu, ss, vv) = svd_complex_one(aa, m, n, jobz)?;
            for (i, value) in ss.into_iter().enumerate() { singular_values.set_flat(batch_i*k+i, Scalar::Float(value)); }
            if let Some(out) = &mut u { write_col_major(out, batch_i, &uu, m, ucols); }
            if let Some(out) = &mut vh { write_col_major(out, batch_i, &vv, vtrows, n); }
        } else {
            let aa = matrix_to_col_major::<f64>(a, abase, m, n);
            let (uu, ss, vv) = svd_real_one(aa, m, n, jobz)?;
            for (i, value) in ss.into_iter().enumerate() { singular_values.set_flat(batch_i*k+i, Scalar::Float(value)); }
            if let Some(out) = &mut u { write_col_major(out, batch_i, &uu, m, ucols); }
            if let Some(out) = &mut vh { write_col_major(out, batch_i, &vv, vtrows, n); }
        }
    }
    Ok(SvdResult { u, singular_values, vh })
}

pub struct EigResult {
    pub values: NdArray,
    pub vectors: Option<NdArray>,
}

fn eig_real_one(mut a: Vec<f64>, n: usize, vectors: bool) -> Result<(Vec<C64v>, Vec<C64v>)> {
    if n == 0 { return Ok((Vec::new(), Vec::new())); }
    let jobvl = b'N'; let jobvr = if vectors { b'V' } else { b'N' };
    let ni = n as i64; let lda=ni.max(1); let ldvl=1i64; let ldvr=if vectors {lda}else{1};
    let mut wr=vec![0.0;n]; let mut wi=vec![0.0;n]; let mut vl=vec![0.0;1];
    let mut vr=vec![0.0;if vectors {n*n}else{1}]; let mut query=0.0; let mut info=0i64; let q=-1i64;
    // SAFETY: lwork=-1 workspace query with all LAPACK matrix/vector extents live.
    unsafe { sys::dgeev(&jobvl,&jobvr,&ni,a.as_mut_ptr(),&lda,wr.as_mut_ptr(),wi.as_mut_ptr(),vl.as_mut_ptr(),&ldvl,vr.as_mut_ptr(),&ldvr,&mut query,&q,&mut info) };
    if info != 0 { return Err(Error::ValueError("Eigenvalues did not converge".into())); }
    let lwork=(query as i64).max(1); let mut work=vec![0.0;lwork as usize];
    // SAFETY: `work` has the queried size; all other buffers retain their extents.
    unsafe { sys::dgeev(&jobvl,&jobvr,&ni,a.as_mut_ptr(),&lda,wr.as_mut_ptr(),wi.as_mut_ptr(),vl.as_mut_ptr(),&ldvl,vr.as_mut_ptr(),&ldvr,work.as_mut_ptr(),&lwork,&mut info) };
    if info != 0 { return Err(Error::ValueError("Eigenvalues did not converge".into())); }
    let values=(0..n).map(|i| C64v::new(wr[i],wi[i])).collect();
    let mut out=vec![C64v::new(0.0,0.0);if vectors {n*n}else{0}];
    if vectors { let mut j=0; while j<n { if wi[j]==0.0 { for r in 0..n { out[r+j*n]=C64v::new(vr[r+j*n],0.0); } j+=1; } else { for r in 0..n { let z=C64v::new(vr[r+j*n],vr[r+(j+1)*n]); out[r+j*n]=z; out[r+(j+1)*n]=z.conj(); } j+=2; } } }
    Ok((values,out))
}

fn eig_complex_one(mut a: Vec<C64v>, n: usize, vectors: bool) -> Result<(Vec<C64v>, Vec<C64v>)> {
    if n == 0 { return Ok((Vec::new(), Vec::new())); }
    let jobvl=b'N'; let jobvr=if vectors {b'V'} else {b'N'}; let ni=n as i64;
    let lda=ni.max(1); let ldvl=1i64; let ldvr=if vectors {lda}else{1};
    let mut w=vec![C64v::new(0.0,0.0);n]; let mut vl=vec![C64v::new(0.0,0.0);1];
    let mut vr=vec![C64v::new(0.0,0.0);if vectors {n*n}else{1}]; let mut rw=vec![0.0;2*n];
    let mut query=C64v::new(0.0,0.0); let mut info=0i64; let q=-1i64;
    // SAFETY: lwork=-1 workspace query with valid complex and real work arrays.
    unsafe { sys::zgeev(&jobvl,&jobvr,&ni,a.as_mut_ptr(),&lda,w.as_mut_ptr(),vl.as_mut_ptr(),&ldvl,vr.as_mut_ptr(),&ldvr,&mut query,&q,rw.as_mut_ptr(),&mut info) };
    if info != 0 { return Err(Error::ValueError("Eigenvalues did not converge".into())); }
    let lwork=(query.re as i64).max(1); let mut work=vec![C64v::new(0.0,0.0);lwork as usize];
    // SAFETY: `work` has the queried size and all other extents are unchanged.
    unsafe { sys::zgeev(&jobvl,&jobvr,&ni,a.as_mut_ptr(),&lda,w.as_mut_ptr(),vl.as_mut_ptr(),&ldvl,vr.as_mut_ptr(),&ldvr,work.as_mut_ptr(),&lwork,rw.as_mut_ptr(),&mut info) };
    if info != 0 { return Err(Error::ValueError("Eigenvalues did not converge".into())); }
    if !vectors { vr.clear(); }
    Ok((w,vr))
}

pub fn eig(a: &NdArray, vectors: bool, complex_input: bool) -> Result<EigResult> {
    if !HAVE_LAPACK { return Err(unavailable()); }
    let n=a.shape[a.ndim()-1] as usize; let batch=a.shape[..a.ndim()-2].to_vec();
    let mut vshape=batch.clone(); vshape.push(n as isize); let mut values=NdArray::zeros(vshape,DType::C128)?;
    let mut vectors_out=if vectors { let mut sh=batch.clone(); sh.extend([n as isize,n as isize]); Some(NdArray::zeros(sh,DType::C128)?) } else {None};
    for (bi,base) in offsets(&batch,&a.strides[..batch.len()],a.byte_offset).enumerate() {
        let (w,v)=if complex_input {eig_complex_one(matrix_to_col_major::<C64v>(a,base,n,n),n,vectors)?} else {eig_real_one(matrix_to_col_major::<f64>(a,base,n,n),n,vectors)?};
        for (i,z) in w.into_iter().enumerate(){values.set_flat(bi*n+i,Scalar::Complex(z));}
        if let Some(out)=&mut vectors_out {write_col_major(out,bi,&v,n,n);}
    }
    Ok(EigResult{values,vectors:vectors_out})
}

fn eigh_real_one(mut a: Vec<f64>, n: usize, upper: bool, vectors: bool) -> Result<(Vec<f64>,Vec<f64>)> {
    if n==0{return Ok((Vec::new(),Vec::new()));} let job=if vectors{b'V'}else{b'N'}; let uplo=if upper{b'U'}else{b'L'};
    let ni=n as i64;let lda=ni.max(1);let mut w=vec![0.0;n];let mut qw=0.0;let mut qi=0i64;let mut info=0i64;let q=-1i64;
    // SAFETY: both workspace lengths are -1 queries and all matrix buffers are live.
    unsafe{sys::dsyevd(&job,&uplo,&ni,a.as_mut_ptr(),&lda,w.as_mut_ptr(),&mut qw,&q,&mut qi,&q,&mut info)};
    if info!=0{return Err(Error::ValueError("Eigenvalues did not converge".into()));}let lw=(qw as i64).max(1);let li=qi.max(1);let mut work=vec![0.0;lw as usize];let mut iw=vec![0i64;li as usize];
    // SAFETY: queried workspace sizes and unchanged matrix extents.
    unsafe{sys::dsyevd(&job,&uplo,&ni,a.as_mut_ptr(),&lda,w.as_mut_ptr(),work.as_mut_ptr(),&lw,iw.as_mut_ptr(),&li,&mut info)};
    if info!=0{return Err(Error::ValueError("Eigenvalues did not converge".into()));}if !vectors{a.clear();}Ok((w,a))
}

fn eigh_complex_one(mut a: Vec<C64v>, n: usize, upper: bool, vectors: bool) -> Result<(Vec<f64>,Vec<C64v>)> {
    if n==0{return Ok((Vec::new(),Vec::new()));}let job=if vectors{b'V'}else{b'N'};let uplo=if upper{b'U'}else{b'L'};let ni=n as i64;let lda=ni.max(1);
    let mut w=vec![0.0;n];let mut qw=C64v::new(0.0,0.0);let mut qr=0.0;let mut qi=0i64;let mut info=0i64;let q=-1i64;
    // SAFETY: all three workspace lengths are -1 queries with writable scalars.
    unsafe{sys::zheevd(&job,&uplo,&ni,a.as_mut_ptr(),&lda,w.as_mut_ptr(),&mut qw,&q,&mut qr,&q,&mut qi,&q,&mut info)};
    if info!=0{return Err(Error::ValueError("Eigenvalues did not converge".into()));}let lw=(qw.re as i64).max(1);let lr=(qr as i64).max(1);let li=qi.max(1);
    let mut work=vec![C64v::new(0.0,0.0);lw as usize];let mut rw=vec![0.0;lr as usize];let mut iw=vec![0i64;li as usize];
    // SAFETY: queried workspace sizes and unchanged matrix extents.
    unsafe{sys::zheevd(&job,&uplo,&ni,a.as_mut_ptr(),&lda,w.as_mut_ptr(),work.as_mut_ptr(),&lw,rw.as_mut_ptr(),&lr,iw.as_mut_ptr(),&li,&mut info)};
    if info!=0{return Err(Error::ValueError("Eigenvalues did not converge".into()));}if !vectors{a.clear();}Ok((w,a))
}

pub fn eigh(a:&NdArray,upper:bool,vectors:bool,complex:bool)->Result<EigResult>{
    if !HAVE_LAPACK{return Err(unavailable());}let n=a.shape[a.ndim()-1] as usize;let batch=a.shape[..a.ndim()-2].to_vec();let mut sh=batch.clone();sh.push(n as isize);let mut values=NdArray::zeros(sh,DType::F64)?;
    let mut vo=if vectors{let mut s=batch.clone();s.extend([n as isize,n as isize]);Some(NdArray::zeros(s,if complex{DType::C128}else{DType::F64})?)}else{None};
    for(bi,base)in offsets(&batch,&a.strides[..batch.len()],a.byte_offset).enumerate(){if complex{let(w,v)=eigh_complex_one(matrix_to_col_major::<C64v>(a,base,n,n),n,upper,vectors)?;for(i,x)in w.into_iter().enumerate(){values.set_flat(bi*n+i,Scalar::Float(x));}if let Some(out)=&mut vo{write_col_major(out,bi,&v,n,n);}}else{let(w,v)=eigh_real_one(matrix_to_col_major::<f64>(a,base,n,n),n,upper,vectors)?;for(i,x)in w.into_iter().enumerate(){values.set_flat(bi*n+i,Scalar::Float(x));}if let Some(out)=&mut vo{write_col_major(out,bi,&v,n,n);}}}
    Ok(EigResult{values,vectors:vo})
}

fn qr_raw_impl<T:QrScalar>(a:&NdArray)->Result<NdArray>{
    let m=a.shape[a.ndim()-2] as usize;let n=a.shape[a.ndim()-1] as usize;let k=m.min(n);let batch=a.shape[..a.ndim()-2].to_vec();let mut tshape=batch.clone();tshape.push(k as isize);let mut tau_out=NdArray::zeros(tshape,T::DTYPE)?;
    for(bi,base)in offsets(&batch,&a.strides[..batch.len()],a.byte_offset).enumerate(){if k==0{continue;}let mut aa=matrix_to_col_major::<T>(a,base,m,n);let mut tau=vec![T::default();k];if T::geqrf(m as i64,n as i64,&mut aa,&mut tau)!=0{return Err(Error::ValueError("Incorrect argument found while performing QR factorization".into()));}
        let rs=a.strides[a.ndim()-2];let cs=a.strides[a.ndim()-1];for row in 0..m{for col in 0..n{a.write_at(base+row as isize*rs+col as isize*cs,aa[row+col*m].to_scalar());}}for(i,value)in tau.into_iter().enumerate(){tau_out.set_flat(bi*k+i,value.to_scalar());}}
    Ok(tau_out)
}

pub fn qr_raw(a:&NdArray,complex:bool)->Result<NdArray>{if !HAVE_LAPACK{return Err(unavailable());}if complex{qr_raw_impl::<C64v>(a)}else{qr_raw_impl::<f64>(a)}}

fn qr_q_impl<T:QrScalar>(a:&NdArray,tau:&NdArray,complete:bool)->Result<NdArray>{
    let m=a.shape[a.ndim()-2] as usize;let n=a.shape[a.ndim()-1] as usize;let k=m.min(n);let mc=if complete{m}else{k};let batch=a.shape[..a.ndim()-2].to_vec();let mut shape=batch.clone();shape.extend([m as isize,mc as isize]);let mut out=NdArray::zeros(shape,T::DTYPE)?;
    let ao=offsets(&batch,&a.strides[..batch.len()],a.byte_offset);let to=offsets(&batch,&tau.strides[..batch.len()],tau.byte_offset);
    for(bi,(base,tbase))in ao.zip(to).enumerate(){if mc==0{continue;}let mut q=vec![T::default();m*mc];let rs=a.strides[a.ndim()-2];let cs=a.strides[a.ndim()-1];for col in 0..n.min(mc){for row in 0..m{q[row+col*m]=T::from_scalar(a.read_at(base+row as isize*rs+col as isize*cs));}}let ts=tau.strides[tau.ndim()-1];let mut tv=(0..k).map(|i|T::from_scalar(tau.read_at(tbase+i as isize*ts))).collect::<Vec<_>>();if T::gqr(m as i64,mc as i64,k as i64,&mut q,&mut tv)!=0{return Err(Error::ValueError("Incorrect argument found while performing QR factorization".into()));}write_col_major(&mut out,bi,&q,m,mc);}
    Ok(out)
}

pub fn qr_q(a:&NdArray,tau:&NdArray,complete:bool,complex:bool)->Result<NdArray>{if !HAVE_LAPACK{return Err(unavailable());}if complex{qr_q_impl::<C64v>(a,tau,complete)}else{qr_q_impl::<f64>(a,tau,complete)}}
