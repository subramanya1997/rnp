//! `_rnp` — the CPython extension module exposing `rnp-core`.

use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use rnp_core::{BinOp, DType, NdArray, Scalar};

mod convert;
mod pyarray;
mod pydtype;

use convert::{array_from_any, scalar_from_py};
use pyarray::{dtype_or_default, shape_from_any, ufunc2, PyNdArray};
use pydtype::{dtype_from_any, PyDType};

/// Map a core error onto the Python exception numpy would raise.
pub(crate) fn err(e: rnp_core::Error) -> PyErr {
    match e {
        rnp_core::Error::ValueError(m) => PyValueError::new_err(m),
        rnp_core::Error::TypeError(m) => PyTypeError::new_err(m),
        rnp_core::Error::IndexError(m) => PyIndexError::new_err(m),
        rnp_core::Error::NotImplemented(m) => PyNotImplementedError::new_err(m),
    }
}

fn wrap(py: Python<'_>, a: NdArray) -> PyResult<Py<PyNdArray>> {
    PyNdArray::into_py_any(a, py)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None))]
fn zeros(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    let d = dtype_or_default(dtype, DType::F64)?;
    wrap(py, NdArray::zeros(shape_from_any(shape)?, d).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None))]
fn ones(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    let d = dtype_or_default(dtype, DType::F64)?;
    wrap(py, NdArray::ones(shape_from_any(shape)?, d).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None))]
fn empty(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    let d = dtype_or_default(dtype, DType::F64)?;
    wrap(py, NdArray::empty(shape_from_any(shape)?, d).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype = None))]
fn full(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    let v = scalar_from_py(fill_value)
        .ok_or_else(|| PyTypeError::new_err("full() fill_value must be a scalar"))?;
    // numpy infers the dtype from the fill value when none is given.
    let d = match dtype {
        None => v.natural_dtype(),
        Some(o) if o.is_none() => v.natural_dtype(),
        Some(o) => dtype_from_any(o)?,
    };
    wrap(py, NdArray::full(shape_from_any(shape)?, d, v).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (start, stop = None, step = None, dtype = None))]
fn arange(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    let to_f = |o: &Bound<'_, PyAny>| -> PyResult<(f64, bool)> {
        let s = scalar_from_py(o)
            .ok_or_else(|| PyTypeError::new_err("arange() arguments must be numbers"))?;
        let exact = matches!(s, Scalar::Int(_) | Scalar::Uint(_) | Scalar::Bool(_));
        let v = match s {
            Scalar::Bool(b) => b as u8 as f64,
            Scalar::Int(i) => i as f64,
            Scalar::Uint(u) => u as f64,
            Scalar::Float(f) => f,
            Scalar::Complex(c) => c.re,
        };
        Ok((v, exact))
    };

    let (a, a_exact) = to_f(start)?;
    // arange(stop) means arange(0, stop).
    let (start_v, stop_v, all_exact) = match stop {
        Some(s) if !s.is_none() => {
            let (b, b_exact) = to_f(s)?;
            (a, b, a_exact && b_exact)
        }
        _ => (0.0, a, a_exact),
    };
    let (step_v, step_exact) = match step {
        None => (1.0, true),
        Some(s) if s.is_none() => (1.0, true),
        Some(s) => to_f(s)?,
    };
    // numpy: integral arguments give int64, otherwise float64.
    let d = match dtype {
        None => {
            if all_exact && step_exact {
                DType::I64
            } else {
                DType::F64
            }
        }
        Some(o) if o.is_none() => {
            if all_exact && step_exact {
                DType::I64
            } else {
                DType::F64
            }
        }
        Some(o) => dtype_from_any(o)?,
    };
    wrap(py, NdArray::arange(start_v, stop_v, step_v, d).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (obj, dtype = None, *, copy = true))]
fn array(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    copy: bool,
) -> PyResult<Py<PyNdArray>> {
    let d = match dtype {
        None => None,
        Some(o) if o.is_none() => None,
        Some(o) => Some(dtype_from_any(o)?),
    };
    wrap(py, array_from_any(obj, d, copy)?)
}

#[pyfunction]
#[pyo3(signature = (obj, dtype = None))]
fn asarray(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    let d = match dtype {
        None => None,
        Some(o) if o.is_none() => None,
        Some(o) => Some(dtype_from_any(o)?),
    };
    wrap(py, array_from_any(obj, d, false)?)
}

macro_rules! binary_ufunc {
    ($name:ident, $op:expr) => {
        #[pyfunction]
        fn $name(
            py: Python<'_>,
            a: &Bound<'_, PyAny>,
            b: &Bound<'_, PyAny>,
        ) -> PyResult<Py<PyAny>> {
            ufunc2(py, a, b, $op)
        }
    };
}

binary_ufunc!(add, BinOp::Add);
binary_ufunc!(subtract, BinOp::Sub);
binary_ufunc!(multiply, BinOp::Mul);
binary_ufunc!(divide, BinOp::Div);
binary_ufunc!(true_divide, BinOp::Div);
binary_ufunc!(equal, BinOp::Eq);
binary_ufunc!(not_equal, BinOp::Ne);
binary_ufunc!(less, BinOp::Lt);
binary_ufunc!(less_equal, BinOp::Le);
binary_ufunc!(greater, BinOp::Gt);
binary_ufunc!(greater_equal, BinOp::Ge);

#[pyfunction]
fn promote_types(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyDType> {
    Ok(PyDType::new(rnp_core::promote(
        dtype_from_any(a)?,
        dtype_from_any(b)?,
    )))
}

#[pyfunction]
#[pyo3(signature = (*shapes))]
fn broadcast_shapes<'py>(
    py: Python<'py>,
    shapes: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut acc: Vec<isize> = vec![];
    for s in shapes.iter() {
        let sh = shape_from_any(&s)?;
        acc = rnp_core::iter::broadcast_shapes(&acc, &sh).map_err(err)?;
    }
    PyTuple::new(py, acc.iter().map(|&d| d as usize))
}

#[pyfunction]
fn shape<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let a = array_from_any(obj, None, false)?;
    PyTuple::new(py, a.shape.iter().map(|&d| d as usize))
}

#[pyfunction]
fn result_type(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<PyDType> {
    let _ = py;
    let mut acc: Option<DType> = None;
    for a in args.iter() {
        let d = if let Ok(arr) = a.cast::<PyNdArray>() {
            arr.borrow().arr.dtype
        } else if let Some(s) = scalar_from_py(&a) {
            s.natural_dtype()
        } else {
            dtype_from_any(&a)?
        };
        acc = Some(match acc {
            None => d,
            Some(p) => rnp_core::promote(p, d),
        });
    }
    Ok(PyDType::new(acc.unwrap_or(DType::F64)))
}

/// A dict of every supported dtype, keyed by name, for the shim to expose as
/// `np.int8`, `np.float64`, ...
#[pyfunction]
fn _dtype_table(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for dt in rnp_core::ALL_DTYPES {
        d.set_item(dt.name(), PyDType::new(dt))?;
    }
    Ok(d)
}

#[pymodule]
fn _rnp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNdArray>()?;
    m.add_class::<PyDType>()?;
    m.add_class::<pyarray::PyFlags>()?;

    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(asarray, m)?)?;

    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(divide, m)?)?;
    m.add_function(wrap_pyfunction!(true_divide, m)?)?;
    m.add_function(wrap_pyfunction!(equal, m)?)?;
    m.add_function(wrap_pyfunction!(not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(less, m)?)?;
    m.add_function(wrap_pyfunction!(less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(greater, m)?)?;
    m.add_function(wrap_pyfunction!(greater_equal, m)?)?;

    m.add_function(wrap_pyfunction!(promote_types, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_shapes, m)?)?;
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(result_type, m)?)?;
    m.add_function(wrap_pyfunction!(_dtype_table, m)?)?;

    m.add("__version__", "0.1.0")?;
    Ok(())
}
