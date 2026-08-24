//! Python bindings for the FFT kernels.

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use rnp_core::DType;

use crate::convert::array_from_any;
use crate::pyarray::{store_or_wrap, PyNdArray};

fn output_dtype(input: DType) -> DType {
    match input {
        DType::F32 | DType::C64 => DType::C64,
        _ => DType::C128,
    }
}

fn validate_out(out: Option<&Bound<'_, PyAny>>, shape: &[isize], dtype: DType) -> PyResult<()> {
    let Some(out) = out.filter(|o| !o.is_none()) else {
        return Ok(());
    };
    let cell = out
        .cast::<PyNdArray>()
        .map_err(|_| PyTypeError::new_err("return arrays must be of ArrayType"))?;
    let arr = &cell.borrow().arr;
    if arr.shape != shape {
        return Err(PyValueError::new_err("output array has wrong shape."));
    }
    if arr.dtype() != dtype {
        return Err(PyTypeError::new_err(format!(
            "Cannot cast ufunc 'fft' output from dtype('{}') to dtype('{}') with casting rule 'same_kind'",
            dtype.name(), arr.dtype().name()
        )));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (a, n, axis, forward, scale, out = None))]
fn _fft_c2c<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: usize,
    axis: isize,
    forward: bool,
    scale: f64,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let input = array_from_any(a, None, false)?;
    let ndim = input.ndim() as isize;
    let axis = if axis < 0 { axis + ndim } else { axis };
    if axis < 0 || axis >= ndim {
        return Err(PyIndexError::new_err("tuple index out of range"));
    }
    let dtype = output_dtype(input.dtype());
    let mut shape = input.shape.clone();
    shape[axis as usize] = n as isize;
    validate_out(out, &shape, dtype)?;
    let result = rnp_core::fft::c2c_axis(&input, n, axis as usize, forward, scale, dtype)
        .map_err(crate::err)?;
    store_or_wrap(py, result, out)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_fft_c2c, module)?)?;
    Ok(())
}
