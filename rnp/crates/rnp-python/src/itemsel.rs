//! Module-level item-selection and searching functions: `take`, `put`,
//! `putmask`, `compress`, `choose`, `nonzero`, `where`, `flatnonzero`.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use rnp_core::indexing::TakeMode;
use rnp_core::{DType, NdArray, Scalar};

use crate::convert::array_from_any;
use crate::pyarray::{store_or_wrap, PyNdArray};

fn asarr(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    array_from_any(obj, None, false)
}

fn mode_of(mode: &str) -> PyResult<TakeMode> {
    TakeMode::from_str(mode).ok_or_else(|| {
        PyValueError::new_err(format!(
            "clipmode not understood; expected 'raise', 'wrap' or 'clip' (got '{mode}')"
        ))
    })
}

#[pyfunction]
#[pyo3(signature = (a, indices, axis = None, out = None, mode = "raise"))]
pub fn take<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyNdArray::into_py_any(asarr(a)?, py)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("axis", axis)?;
    kwargs.set_item("out", out)?;
    kwargs.set_item("mode", mode)?;
    arr.bind(py).call_method("take", (indices,), Some(&kwargs))
}

#[pyfunction]
#[pyo3(signature = (a, indices, values, mode = "raise"))]
pub fn put(
    a: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    mode: &str,
) -> PyResult<()> {
    let cell = a
        .cast::<PyNdArray>()
        .map_err(|_| PyTypeError::new_err("argument 1 must be numpy.ndarray, not list"))?;
    let m = mode_of(mode)?;
    let idx = asarr(indices)?;
    let ivals: Vec<i64> = int_values(&idx);
    let arr = cell.borrow().arr.clone();
    let vals = array_from_any(values, Some(arr.dtype()), false)?;
    rnp_core::indexing::put(&arr, &ivals, &vals, m).map_err(crate::err)
}

#[pyfunction]
pub fn putmask(
    a: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let cell = a
        .cast::<PyNdArray>()
        .map_err(|_| PyTypeError::new_err("putmask: first argument must be an array"))?;
    let arr = cell.borrow().arr.clone();
    let m = asarr(mask)?;
    let v = array_from_any(values, Some(arr.dtype()), false)?;
    rnp_core::indexing::putmask(&arr, &m, &v).map_err(crate::err)
}

#[pyfunction]
#[pyo3(signature = (condition, a, axis = None, out = None))]
pub fn compress<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    a: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyNdArray::into_py_any(asarr(a)?, py)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("axis", axis)?;
    kwargs.set_item("out", out)?;
    arr.bind(py).call_method("compress", (condition,), Some(&kwargs))
}

#[pyfunction]
#[pyo3(signature = (a, choices, out = None, mode = "raise"))]
pub fn choose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    choices: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let sel = asarr(a)?;
    let m = mode_of(mode)?;
    let mut arrays = Vec::new();
    for c in choices.try_iter()? {
        arrays.push(asarr(&c?)?);
    }
    let res = rnp_core::indexing::choose(&sel, &arrays, m).map_err(crate::err)?;
    store_or_wrap(py, res, out)
}

#[pyfunction]
pub fn nonzero<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let arr = asarr(a)?;
    if arr.ndim() == 0 {
        return Err(PyValueError::new_err(
            "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.",
        ));
    }
    let cols = rnp_core::indexing::nonzero(&arr);
    let mut out = Vec::with_capacity(cols.len());
    for c in cols {
        out.push(PyNdArray::into_py_any(c, py)?);
    }
    PyTuple::new(py, out)
}

#[pyfunction]
pub fn flatnonzero<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Py<PyNdArray>> {
    let arr = asarr(a)?;
    let n = arr.size() as isize;
    let flat = if arr.flags.c_contiguous {
        arr.reshape(&[n]).map_err(crate::err)?
    } else {
        arr.copy().reshape(&[n]).map_err(crate::err)?
    };
    let cols = rnp_core::indexing::nonzero(&flat);
    PyNdArray::into_py_any(cols.into_iter().next().unwrap(), py)
}

fn int_values(a: &NdArray) -> Vec<i64> {
    a.to_vec()
        .into_iter()
        .map(|s| match s {
            Scalar::Int(i) => i,
            Scalar::Uint(u) => u as i64,
            Scalar::Bool(b) => b as i64,
            Scalar::Float(f) => f as i64,
            Scalar::Complex(c) => c.re as i64,
        })
        .collect()
}

/// `np.where(cond)` == `np.nonzero(cond)`; `np.where(cond, x, y)` selects.
#[pyfunction]
#[pyo3(signature = (condition, x = None, y = None))]
pub fn where_<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    x: Option<&Bound<'py, PyAny>>,
    y: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let cond = asarr(condition)?;
    let (x, y) = match (x, y) {
        (None, None) => {
            return Ok(nonzero(py, condition)?.into_any());
        }
        (Some(a), Some(b)) if !a.is_none() && !b.is_none() => (a, b),
        _ => {
            return Err(PyValueError::new_err(
                "either both or neither of x and y should be given",
            ))
        }
    };
    let xa = asarr(x)?;
    let ya = asarr(y)?;
    let dt = rnp_core::promote(xa.dtype(), ya.dtype());
    let mut shape = rnp_core::iter::broadcast_shapes(&cond.shape, &xa.shape).map_err(crate::err)?;
    shape = rnp_core::iter::broadcast_shapes(&shape, &ya.shape).map_err(crate::err)?;
    let bc = rnp_core::iter::broadcast_to(&cond, &shape).map_err(crate::err)?;
    let bx = rnp_core::iter::broadcast_to(&xa, &shape).map_err(crate::err)?;
    let by = rnp_core::iter::broadcast_to(&ya, &shape).map_err(crate::err)?;
    let out = NdArray::empty(shape.clone(), dt).map_err(crate::err)?;
    let co: Vec<isize> = rnp_core::iter::offsets(&bc.shape, &bc.strides, bc.byte_offset).collect();
    let xo: Vec<isize> = rnp_core::iter::offsets(&bx.shape, &bx.strides, bx.byte_offset).collect();
    let yo: Vec<isize> = rnp_core::iter::offsets(&by.shape, &by.strides, by.byte_offset).collect();
    let isz = out.itemsize() as isize;
    for i in 0..co.len() {
        let t = match bc.read_at(co[i]) {
            Scalar::Bool(b) => b,
            Scalar::Int(v) => v != 0,
            Scalar::Uint(v) => v != 0,
            Scalar::Float(v) => v != 0.0,
            Scalar::Complex(c) => c.re != 0.0 || c.im != 0.0,
        };
        let v = if t { bx.read_at(xo[i]) } else { by.read_at(yo[i]) };
        out.write_at(i as isize * isz, v);
    }
    Ok(PyNdArray::into_py_any(out, py)?.into_bound(py).into_any())
}

/// `np.broadcast_to` — a read-only, zero-copy stretched view.
#[pyfunction]
#[pyo3(signature = (array, shape, subok = false))]
pub fn broadcast_to<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
    subok: bool,
) -> PyResult<Py<PyNdArray>> {
    let _ = subok;
    let a = asarr(array)?;
    let want = crate::pyarray::shape_from_any(shape)?;
    let out = rnp_core::iter::broadcast_to(&a, &want).map_err(|_| {
        PyValueError::new_err(format!(
            "operands could not be broadcast together with remapped shapes \
             [original->remapped]: {:?} and requested shape {:?}",
            a.shape, want
        ))
    })?;
    PyNdArray::into_py_any(out, py)
}

/// `numpy.lib.stride_tricks.as_strided`: a view with caller-chosen strides.
#[pyfunction]
#[pyo3(signature = (x, shape = None, strides = None, writeable = true))]
pub fn _as_strided<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    shape: Option<&Bound<'py, PyAny>>,
    strides: Option<&Bound<'py, PyAny>>,
    writeable: bool,
) -> PyResult<Py<PyNdArray>> {
    let a = asarr(x)?;
    let mut out = a.clone();
    if let Some(s) = shape {
        if !s.is_none() {
            out.shape = crate::pyarray::shape_from_any(s)?;
        }
    }
    if let Some(s) = strides {
        if !s.is_none() {
            out.strides = crate::pyarray::shape_from_any(s)?;
        }
    }
    if out.strides.len() != out.shape.len() {
        return Err(PyValueError::new_err(
            "strides must have the same length as shape",
        ));
    }
    out.flags.owndata = false;
    out.flags.writeable = writeable;
    out.update_flags();
    out.flags.writeable = writeable;
    PyNdArray::into_py_any(out, py)
}

/// A dtype constant the shim uses for its bool-typed helpers.
#[pyfunction]
pub fn _bool_dtype() -> crate::pydtype::PyDType {
    crate::pydtype::PyDType::new(DType::Bool)
}
