//! Portable fallback surface for platforms without the Accelerate LAPACK ABI.
//!
//! The transcribed ndarray engine does not yet contain pure-Rust LAPACK
//! kernels. Keeping this module separate prevents Apple framework linkage and
//! `$NEWLAPACK$ILP64` symbol names from entering non-macOS builds while
//! preserving the public API used by the Python bindings.

use crate::array::NdArray;
use crate::dtype::DType;
use crate::error::{Error, Result};

pub const HAVE_LAPACK: bool = false;

pub struct LstsqResult {
    pub x: NdArray,
    pub residuals: NdArray,
    pub rank: i64,
    pub singular_values: NdArray,
}

pub struct SvdResult {
    pub u: Option<NdArray>,
    pub singular_values: NdArray,
    pub vh: Option<NdArray>,
}

pub struct EigResult {
    pub values: NdArray,
    pub vectors: Option<NdArray>,
}

fn unavailable<T>() -> Result<T> {
    Err(Error::NotImplemented(
        "numpy.linalg requires a platform LAPACK backend".into(),
    ))
}

pub fn solve(_a: &NdArray, _b: &NdArray, _vector: bool, _dtype: DType) -> Result<NdArray> {
    unavailable()
}

pub fn inv(_a: &NdArray, _dtype: DType, _singular_nan: bool) -> Result<NdArray> {
    unavailable()
}

pub fn slogdet(_a: &NdArray, _dtype: DType) -> Result<(NdArray, NdArray)> {
    unavailable()
}

pub fn det(_a: &NdArray, _dtype: DType) -> Result<NdArray> {
    unavailable()
}

pub fn cholesky(_a: &NdArray, _upper: bool, _dtype: DType) -> Result<NdArray> {
    unavailable()
}

pub fn lstsq(_a: &NdArray, _b: &NdArray, _rcond: f64, _dtype: DType) -> Result<LstsqResult> {
    unavailable()
}

pub fn svd(_a: &NdArray, _full: bool, _vectors: bool, _complex: bool) -> Result<SvdResult> {
    unavailable()
}

pub fn eig(_a: &NdArray, _vectors: bool, _complex_input: bool) -> Result<EigResult> {
    unavailable()
}

pub fn eigh(_a: &NdArray, _upper: bool, _vectors: bool, _complex: bool) -> Result<EigResult> {
    unavailable()
}

pub fn qr_raw(_a: &NdArray, _complex: bool) -> Result<NdArray> {
    unavailable()
}

pub fn qr_q(_a: &NdArray, _tau: &NdArray, _complete: bool, _complex: bool) -> Result<NdArray> {
    unavailable()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_lapack_backend_unavailable() {
        let matrix = NdArray::zeros(vec![1, 1], DType::F64).unwrap();
        assert!(matches!(
            det(&matrix, DType::F64),
            Err(Error::NotImplemented(_))
        ));
    }
}
