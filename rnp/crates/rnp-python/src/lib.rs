//! `_rnp` — the CPython extension module exposing `rnp-core`.

use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use rnp_core::casting::{Casting, TypeArg, WeakKind};
use rnp_core::reduce::ReduceOp;
use rnp_core::{BinOp, DType, Descr, NdArray, Scalar};

mod convert;
mod index;
mod itemsel;
mod pyarray;
mod objects;
mod pydtype;
mod ufuncs;

use convert::{any_scalar, array_from_any, array_from_any_descr, scalar_from_py};
use pyarray::{descr_or_default, shape_from_any, ufunc2, PyNdArray};
use pydtype::{descr_from_any, dtype_from_any, PyDType};

/// Python exception constructors the shim installs at import time, keyed by
/// the engine error they build. The engine only knows names and dtypes; the
/// shim owns the classes (`numpy._core._exceptions`) and the ufunc objects.
static ERROR_FACTORIES: std::sync::OnceLock<Py<PyDict>> = std::sync::OnceLock::new();

/// Install the shim's exception factories. Called once from
/// `rnp_numpy._core._exceptions`.
#[pyfunction]
fn _set_error_factories(d: Py<PyDict>) {
    let _ = ERROR_FACTORIES.set(d);
}

/// Build numpy's `_UFuncNoLoopError` through the shim's factory, falling back
/// to the engine's own message when the shim has not registered one (which
/// happens only if `_rnp` is used without the shim).
fn ufunc_no_loop_err(ufunc: String, dtypes: Vec<String>, fallback: String) -> PyErr {
    Python::attach(|py| {
        let Some(d) = ERROR_FACTORIES.get() else {
            return PyTypeError::new_err(fallback);
        };
        match d.bind(py).get_item("ufunc_no_loop") {
            Ok(Some(f)) => match f.call1((ufunc, dtypes)) {
                Ok(exc) => PyErr::from_value(exc),
                Err(e) => e,
            },
            _ => PyTypeError::new_err(fallback),
        }
    })
}

/// Map a core error onto the Python exception numpy would raise.
pub(crate) fn err(e: rnp_core::Error) -> PyErr {
    match e {
        rnp_core::Error::ValueError(m) => PyValueError::new_err(m),
        rnp_core::Error::TypeError(m) => PyTypeError::new_err(m),
        rnp_core::Error::IndexError(m) => PyIndexError::new_err(m),
        rnp_core::Error::UFuncNoLoop {
            ufunc,
            dtypes,
            message,
        } => ufunc_no_loop_err(ufunc, dtypes, message),
        // The shim re-raises these as np.exceptions.AxisError /
        // DTypePromotionError, both of which numpy derives from ValueError.
        rnp_core::Error::AxisError(m) => PyValueError::new_err(m),
        rnp_core::Error::DTypePromotionError(m) => PyValueError::new_err(m),
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
    let d = descr_or_default(dtype, DType::F64)?;
    wrap(py, NdArray::zeros_descr(shape_from_any(shape)?, d).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None))]
fn ones(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    let d = descr_or_default(dtype, DType::F64)?;
    wrap(py, NdArray::ones_descr(shape_from_any(shape)?, d).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = None))]
fn empty(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    let d = descr_or_default(dtype, DType::F64)?;
    wrap(py, NdArray::empty_descr(shape_from_any(shape)?, d).map_err(err)?)
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype = None))]
fn full(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    // A Python int too wide for any integer dtype cannot fill an integer
    // array at all: probed, `np.full((2,), 2**100, dtype=np.uint8)` is
    // `OverflowError: Python int too large to convert to C long`.
    if convert::huge_int(fill_value)?.is_some() {
        if let Some(o) = dtype {
            if !o.is_none() && descr_from_any(o)?.dt.is_integer() {
                return Err(convert::too_large());
            }
        }
    }
    let v = any_scalar(fill_value)?
        .ok_or_else(|| PyTypeError::new_err("full() fill_value must be a scalar"))?;
    // numpy infers the dtype from the fill value when none is given; a numpy
    // scalar contributes its own dtype, a Python number its natural one.
    let np = convert::np_scalar(fill_value)?;
    let inferred = match np {
        Some((d, _)) => d,
        None => v.natural_dtype(),
    };
    let d = match dtype {
        None => Descr::native(inferred),
        Some(o) if o.is_none() => Descr::native(inferred),
        Some(o) => descr_from_any(o)?,
    };
    if dtype.is_some_and(|o| !o.is_none()) {
        convert::check_int_store(d.dt, np.map(|(t, _)| t), v)?;
    }
    wrap(py, NdArray::full_descr(shape_from_any(shape)?, d, v).map_err(err)?)
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
        let s = any_scalar(o)?
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
        Some(o) => descr_from_any(o)?.dt,
    };
    let bo = match dtype {
        Some(o) if !o.is_none() => descr_from_any(o)?,
        _ => Descr::native(d),
    };
    wrap(
        py,
        NdArray::arange(start_v, stop_v, step_v, d)
            .map_err(err)?
            .into_descr(bo),
    )
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
        Some(o) => Some(descr_from_any(o)?),
    };
    wrap(py, array_from_any_descr(obj, d, copy)?)
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
        Some(o) => Some(descr_from_any(o)?),
    };
    wrap(py, array_from_any_descr(obj, d, false)?)
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
    let (da, db) = (descr_from_any(a)?, descr_from_any(b)?);
    if da.dt.is_flexible() || db.dt.is_flexible() {
        let p = rnp_core::dtype::promote_flexible(da.dt, db.dt).ok_or_else(|| {
            PyValueError::new_err(format!(
                "The DType {} could not be promoted by {}.",
                da.str_code(),
                db.str_code()
            ))
        })?;
        return Ok(PyDType::from_descr(Descr::native(p)));
    }
    Ok(PyDType::new(rnp_core::promote(da.dt, db.dt)))
}

/// Classify one `result_type` argument under NEP 50.
fn type_arg(a: &Bound<'_, PyAny>) -> PyResult<TypeArg> {
    if let Ok(arr) = a.cast::<PyNdArray>() {
        return Ok(TypeArg::Concrete(arr.borrow().arr.dtype()));
    }
    // A bare Python number is *weak*: it contributes its kind, not its value.
    if a.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(TypeArg::Weak(WeakKind::Bool));
    }
    if a.is_instance_of::<pyo3::types::PyInt>() {
        // Probed: `np.result_type(2**100)` is `object` (it is the dtype
        // `np.array(2**100)` would have), but `np.result_type(np.int8,
        // 2**100)` is `int8` -- the huge int is still a *weak* integer.
        if convert::huge_int(a)?.is_some() {
            return Ok(TypeArg::HugeInt);
        }
        return Ok(TypeArg::Weak(WeakKind::Int));
    }
    if a.is_instance_of::<pyo3::types::PyFloat>() {
        return Ok(TypeArg::Weak(WeakKind::Float));
    }
    if a.is_instance_of::<pyo3::types::PyComplex>() {
        return Ok(TypeArg::Weak(WeakKind::Complex));
    }
    Ok(TypeArg::Concrete(dtype_from_any(a)?))
}

#[pyfunction]
#[pyo3(signature = (from_, to, casting = "safe"))]
fn can_cast(from_: &Bound<'_, PyAny>, to: &Bound<'_, PyAny>, casting: &str) -> PyResult<bool> {
    // numpy 2.x removed the value-based path, and says so loudly.
    if from_.is_instance_of::<pyo3::types::PyInt>()
        || from_.is_instance_of::<pyo3::types::PyFloat>()
        || from_.is_instance_of::<pyo3::types::PyComplex>()
    {
        return Err(PyTypeError::new_err(
            "can_cast() does not support Python ints, floats, and complex \
             because the result used to depend on the value.\nThis change \
             was part of adopting NEP 50, we may explicitly allow them again \
             in the future.",
        ));
    }
    let kind = Casting::from_str(casting).ok_or_else(|| {
        PyValueError::new_err(format!(
            "casting must be one of 'no', 'equiv', 'safe', 'same_kind', or \
             'unsafe' (got '{casting}')"
        ))
    })?;
    let src = if let Ok(arr) = from_.cast::<PyNdArray>() {
        Descr::native(arr.borrow().arr.dtype())
    } else {
        descr_from_any(from_)?
    };
    Ok(rnp_core::can_cast(src, descr_from_any(to)?, kind))
}

#[pyfunction]
fn min_scalar_type(value: &Bound<'_, PyAny>) -> PyResult<PyDType> {
    if let Ok(arr) = value.cast::<PyNdArray>() {
        let a = &arr.borrow().arr;
        // Only 0-d arrays and scalars get the value-based treatment.
        if a.ndim() > 0 {
            return Ok(PyDType::new(a.dtype()));
        }
        return Ok(PyDType::new(rnp_core::min_scalar_type(a.get_flat(0))));
    }
    // Probed: `np.min_scalar_type(2**100)` is `object` -- no integer dtype
    // holds it, so numpy falls back to the one that holds anything.
    if convert::huge_int(value)?.is_some() {
        return Ok(PyDType::new(DType::Object));
    }
    let s = scalar_from_py(value)
        .ok_or_else(|| PyTypeError::new_err("min_scalar_type() needs a scalar or an array"))?;
    Ok(PyDType::new(rnp_core::min_scalar_type(s)))
}

#[pyfunction]
#[pyo3(signature = (*arrays))]
fn common_type(arrays: &Bound<'_, PyTuple>) -> PyResult<PyDType> {
    let mut dts = Vec::with_capacity(arrays.len());
    for a in arrays.iter() {
        let arr = array_from_any(&a, None, false)?;
        if !arr.dtype().is_numeric() {
            return Err(PyTypeError::new_err("can't get common type for non-numeric array"));
        }
        dts.push(arr.dtype());
    }
    rnp_core::common_type(&dts)
        .map(PyDType::new)
        .ok_or_else(|| PyTypeError::new_err("can't get common type for non-numeric array"))
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
#[pyo3(signature = (*args))]
fn result_type(args: &Bound<'_, PyTuple>) -> PyResult<PyDType> {
    let mut parsed = Vec::with_capacity(args.len());
    for a in args.iter() {
        parsed.push(type_arg(&a)?);
    }
    let d = rnp_core::result_type(&parsed).ok_or_else(|| {
        if parsed.is_empty() {
            PyValueError::new_err("at least one array or dtype is required")
        } else {
            // numpy raises `np.exceptions.DTypePromotionError`, a TypeError.
            PyTypeError::new_err(
                "The DTypes do not have a common DType. For example they \
                 cannot be stored in a single array unless the dtype is \
                 `object`.",
            )
        }
    })?;
    Ok(PyDType::new(d))
}

/// Reductions as free functions (`np.sum(a, axis=0)`), delegating to the
/// ndarray methods so there is exactly one implementation.
#[pyfunction]
#[pyo3(signature = (a, axis = None, out = None, keepdims = false))]
fn argmin<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    reduce_free(py, a, ReduceOp::ArgMin, axis, None, out, keepdims)
}

#[pyfunction]
#[pyo3(signature = (a, axis = None, out = None, keepdims = false))]
fn argmax<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    reduce_free(py, a, ReduceOp::ArgMax, axis, None, out, keepdims)
}

fn reduce_free<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    op: ReduceOp,
    axis: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyNdArray::into_py_any(array_from_any(a, None, false)?, py)?;
    let kwargs = PyDict::new(py);
    if let Some(x) = axis {
        kwargs.set_item("axis", x)?;
    }
    if let Some(x) = dtype {
        kwargs.set_item("dtype", x)?;
    }
    if let Some(x) = out {
        kwargs.set_item("out", x)?;
    }
    kwargs.set_item("keepdims", keepdims)?;
    let name = match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Prod => "prod",
        ReduceOp::Min => "min",
        ReduceOp::Max => "max",
        ReduceOp::ArgMin => "argmin",
        ReduceOp::ArgMax => "argmax",
    };
    arr.bind(py).call_method(name, (), Some(&kwargs))
}

#[pyfunction]
#[pyo3(signature = (a, axis = None, dtype = None, out = None, keepdims = false))]
fn sum<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    reduce_free(py, a, ReduceOp::Sum, axis, dtype, out, keepdims)
}

#[pyfunction]
#[pyo3(signature = (a, axis = None, dtype = None, out = None, keepdims = false))]
fn prod<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    reduce_free(py, a, ReduceOp::Prod, axis, dtype, out, keepdims)
}

#[pyfunction]
#[pyo3(signature = (a, axis = None, out = None, keepdims = false))]
fn amin<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    reduce_free(py, a, ReduceOp::Min, axis, None, out, keepdims)
}

#[pyfunction]
#[pyo3(signature = (a, axis = None, out = None, keepdims = false))]
fn amax<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    reduce_free(py, a, ReduceOp::Max, axis, None, out, keepdims)
}

#[pyfunction]
#[pyo3(signature = (a, axis = None, dtype = None, out = None, keepdims = false))]
fn mean<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyNdArray::into_py_any(array_from_any(a, None, false)?, py)?;
    let kwargs = PyDict::new(py);
    if let Some(x) = axis {
        kwargs.set_item("axis", x)?;
    }
    if let Some(x) = dtype {
        kwargs.set_item("dtype", x)?;
    }
    if let Some(x) = out {
        kwargs.set_item("out", x)?;
    }
    kwargs.set_item("keepdims", keepdims)?;
    arr.bind(py).call_method("mean", (), Some(&kwargs))
}

/// Install the shim's per-dtype `_wrap` callables for the scalar fast path.
#[pyfunction]
fn _register_scalar_wraps(wraps: Vec<Py<PyAny>>) {
    pydtype::register_scalar_wraps(wraps);
}

/// Install the shim's `name -> scalar class` map behind `dtype.type`.
#[pyfunction]
fn _register_scalar_types(types: &Bound<'_, PyDict>) {
    pydtype::register_scalar_types(types.clone());
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
    m.add_class::<pyarray::PyFlatIter>()?;

    m.add_function(wrap_pyfunction!(itemsel::take, m)?)?;
    m.add_function(wrap_pyfunction!(itemsel::put, m)?)?;
    m.add_function(wrap_pyfunction!(itemsel::putmask, m)?)?;
    m.add_function(wrap_pyfunction!(itemsel::compress, m)?)?;
    m.add_function(wrap_pyfunction!(itemsel::choose, m)?)?;
    m.add_function(wrap_pyfunction!(itemsel::nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(itemsel::flatnonzero, m)?)?;
    m.add_function(wrap_pyfunction!(itemsel::where_, m)?)?;
    m.add_function(wrap_pyfunction!(itemsel::broadcast_to, m)?)?;
    m.add_function(wrap_pyfunction!(itemsel::_as_strided, m)?)?;

    m.add_function(wrap_pyfunction!(_set_error_factories, m)?)?;

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
    m.add_function(wrap_pyfunction!(can_cast, m)?)?;
    m.add_function(wrap_pyfunction!(min_scalar_type, m)?)?;
    m.add_function(wrap_pyfunction!(common_type, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(prod, m)?)?;
    m.add_function(wrap_pyfunction!(amin, m)?)?;
    m.add_function(wrap_pyfunction!(amax, m)?)?;
    m.add_function(wrap_pyfunction!(argmin, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_shapes, m)?)?;
    m.add_function(wrap_pyfunction!(shape, m)?)?;
    m.add_function(wrap_pyfunction!(result_type, m)?)?;
    m.add_function(wrap_pyfunction!(_dtype_table, m)?)?;
    m.add_function(wrap_pyfunction!(_register_scalar_types, m)?)?;
    m.add_function(wrap_pyfunction!(_register_scalar_wraps, m)?)?;
    ufuncs::register(m)?;

    m.add("__version__", "0.1.0")?;
    Ok(())
}
