//! `_rnp` — the CPython extension module exposing `rnp-core`.

use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use rnp_core::casting::{Casting, TypeArg, WeakKind};
use rnp_core::reduce::ReduceOp;
use rnp_core::{BinOp, DType, Descr, NdArray, Scalar};

mod adopt;
mod convert;
mod fields;
mod index;
mod itemsel;
mod linalgops;
mod pyarray;
mod objects;
mod objloops;
mod pydtype;
mod straggler;
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
        rnp_core::Error::OverflowError(m) => pyo3::exceptions::PyOverflowError::new_err(m),
        rnp_core::Error::RuntimeError(m) => pyo3::exceptions::PyRuntimeError::new_err(m),
        rnp_core::Error::UFuncBinaryResolution {
            ufunc,
            dtypes,
            message,
        } => binary_resolution_err(ufunc, dtypes, message),
    }
}

/// numpy's `_UFuncBinaryResolutionError`, built through the shim's factory so
/// that the exception really is a `UFuncTypeError` subclass.
fn binary_resolution_err(ufunc: String, dtypes: Vec<String>, fallback: String) -> PyErr {
    Python::attach(|py| {
        let Some(d) = ERROR_FACTORIES.get() else {
            return PyTypeError::new_err(fallback);
        };
        match d.bind(py).get_item("ufunc_binary_resolution") {
            Ok(Some(f)) => match f.call1((ufunc, dtypes)) {
                Ok(exc) => PyErr::from_value(exc),
                Err(e) => e,
            },
            _ => PyTypeError::new_err(fallback),
        }
    })
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

/// `np.frombuffer(buffer, dtype=float, count=-1, offset=0)` — zero-copy.
#[pyfunction]
#[pyo3(signature = (buffer, dtype = None, count = -1, offset = 0))]
fn frombuffer(
    py: Python<'_>,
    buffer: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    count: i64,
    offset: i64,
) -> PyResult<Py<PyNdArray>> {
    Py::new(py, adopt::frombuffer(py, buffer, dtype, count, offset)?)
}

/// The `np.array`/`np.asarray` front door, shared by both.
///
/// Two things happen here that the generic `array_from_any` cannot do because
/// it deals in bare `NdArray`s rather than Python objects: an *existing*
/// array is handed back (or re-typed as a plain `ndarray` view of a subclass,
/// the way numpy's `subok=False` does), and foreign objects get a chance to
/// convert themselves through the array protocol.
fn array_front(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    copy: Option<bool>,
) -> PyResult<Py<PyNdArray>> {
    let d = match dtype {
        None => None,
        Some(o) if o.is_none() => None,
        Some(o) => Some(descr_from_any(o)?),
    };
    if let Ok(cell) = obj.cast::<PyNdArray>() {
        let same_dtype = {
            let me = cell.borrow();
            d.is_none_or(|want| want == me.arr.descr)
        };
        if copy != Some(true) && same_dtype {
            let is_exact = obj.get_type().is(&py.get_type::<PyNdArray>());
            if is_exact {
                // numpy returns the very same object.
                return Ok(cell.clone().unbind());
            }
            // A subclass instance: `asarray` yields a base-class *view* whose
            // base is the subclass instance (`np.asarray(memmap).base is fp`).
            let arr = cell.borrow().arr.clone();
            return Py::new(
                py,
                PyNdArray { arr, base: Some(obj.clone().unbind()) },
            );
        }
        return wrap(py, array_from_any_descr(obj, d, copy == Some(true))?);
    }
    if let Some(res) = protocol_array(py, obj, d, copy)? {
        return Ok(res);
    }
    // PEP 3118: anything exporting a buffer is adopted with the dtype the
    // buffer itself declares (`np.array(bytearray(b"12")).dtype` is uint8,
    // not the int64 a sequence of ints would produce).
    if let Some(built) = adopt::from_buffer_protocol(py, obj)? {
        return finish_adopted(py, built, d, copy);
    }
    if copy == Some(false) {
        // Nothing below this point can be done without allocating.
        return Err(pyo3::exceptions::PyValueError::new_err(adopt::NO_COPY_MSG));
    }
    wrap(py, array_from_any_descr(obj, d, copy == Some(true))?)
}

/// Apply the caller's `dtype=`/`copy=` to a freshly adopted array.
///
/// A cast or an explicit `copy=True` produces a fresh, owning array (numpy's
/// `np.array(bytearray(...)).base` is None); otherwise the adoption is handed
/// back untouched and stays zero-copy.
fn finish_adopted(
    py: Python<'_>,
    built: PyNdArray,
    d: Option<Descr>,
    copy: Option<bool>,
) -> PyResult<Py<PyNdArray>> {
    let arr = built.arr.clone();
    match d {
        Some(want) if want != arr.descr => {
            if copy == Some(false) {
                return Err(pyo3::exceptions::PyValueError::new_err(adopt::NO_COPY_MSG));
            }
            wrap(py, arr.astype_descr(want))
        }
        _ if copy == Some(true) => wrap(py, arr.copy()),
        _ => Py::new(py, built),
    }
}

/// Try the array protocol (`__array__`, then `__array_interface__`) on a
/// foreign object. `Ok(None)` means "not an array-protocol object".
fn protocol_array(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    d: Option<Descr>,
    copy: Option<bool>,
) -> PyResult<Option<Py<PyNdArray>>> {
    if obj.get_type().hasattr("__array__")? {
        if let Some(res) = adopt::call_array_protocol(py, obj, d, copy)? {
            // A third-party `__array__` may hand back an array of *another*
            // library. That is still an array, so adopt it through the buffer
            // protocol rather than refusing; only a non-array result is the
            // error numpy reports.
            let cell = match res.cast::<PyNdArray>() {
                Ok(c) => c.clone(),
                Err(_) => {
                    if let Some(built) = adopt::from_buffer_protocol(py, &res)? {
                        return Ok(Some(finish_adopted(py, built, d, copy)?));
                    }
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "object __array__ method not producing an array",
                    ));
                }
            };
            let cell = &cell;
            let arr = cell.borrow().arr.clone();
            let arr = match d {
                Some(want) if want != arr.descr => {
                    if copy == Some(false) {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            adopt::NO_COPY_MSG,
                        ));
                    }
                    arr.astype_descr(want)
                }
                _ if copy == Some(true) => arr.copy(),
                _ => return Ok(Some(cell.clone().unbind())),
            };
            return Ok(Some(wrap(py, arr)?));
        }
    }
    if obj.hasattr("__array_interface__")? {
        if let Some(built) = adopt::from_array_interface(py, obj)? {
            let arr = built.arr.clone();
            let base = built.base;
            let out = match d {
                Some(want) if want != arr.descr => PyNdArray { arr: arr.astype_descr(want), base: None },
                _ if copy == Some(true) => PyNdArray { arr: arr.copy(), base: None },
                _ => PyNdArray { arr, base },
            };
            return Ok(Some(Py::new(py, out)?));
        }
    }
    Ok(None)
}

/// `copy` is numpy 2.x's tri-state: `True` always copies, `False` refuses to,
/// and `None` copies only when it must.
#[pyfunction]
#[pyo3(signature = (obj, dtype = None, *, copy = Some(true)))]
fn array(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    copy: Option<bool>,
) -> PyResult<Py<PyNdArray>> {
    array_front(py, obj, dtype, copy)
}

#[pyfunction]
#[pyo3(signature = (obj, dtype = None))]
fn asarray(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    array_front(py, obj, dtype, None)
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
    if da.dt.is_datetime_like() || db.dt.is_datetime_like() {
        // Datetime promotion has its own failure modes (nonlinear units,
        // multiplier overflow) with numpy's own exception types.
        if da.dt.is_datetime_like() && db.dt.is_datetime_like() {
            let p = rnp_core::datetime::promote_meta(da.dt, db.dt).map_err(err)?;
            return Ok(PyDType::from_descr(Descr::native(p)));
        }
        let p = rnp_core::dtype::promote_datetime(da.dt, db.dt).ok_or_else(|| {
            promotion_error(da.dt, db.dt)
        })?;
        return Ok(PyDType::from_descr(Descr::native(p)));
    }
    Ok(PyDType::new(rnp_core::promote(da.dt, db.dt)))
}

/// numpy's `DTypePromotionError` text for two DTypes with no common type.
fn promotion_error(a: rnp_core::dtype::DType, b: rnp_core::dtype::DType) -> PyErr {
    err(rnp_core::Error::DTypePromotionError(format!(
        "The DTypes <class 'numpy.dtypes.{}'> and <class 'numpy.dtypes.{}'> do \
         not have a common DType. For example they cannot be stored in a \
         single array unless the dtype is `object`.",
        dtype_class_name(a),
        dtype_class_name(b)
    )))
}

fn dtype_class_name(d: rnp_core::dtype::DType) -> String {
    use rnp_core::dtype::DType;
    match d {
        DType::DateTime(_) => "DateTime64DType".into(),
        DType::TimeDelta(_) => "TimeDelta64DType".into(),
        DType::Bool => "BoolDType".into(),
        DType::Object => "ObjectDType".into(),
        DType::Bytes(_) => "BytesDType".into(),
        DType::Str(_) => "StrDType".into(),
        DType::Void(_) | DType::Struct(_) | DType::SubArray(_) => "VoidDType".into(),
        other => {
            let n = other.name();
            let mut s = String::new();
            if n.starts_with("uint") {
                s.push_str("UInt");
                s.push_str(&n[4..]);
            } else {
                let mut c = n.chars();
                if let Some(f) = c.next() {
                    s.extend(f.to_uppercase());
                }
                s.extend(c);
            }
            s.push_str("DType");
            s
        }
    }
}

/// Classify one `result_type` argument under NEP 50.
fn type_arg(a: &Bound<'_, PyAny>) -> PyResult<TypeArg> {
    if let Ok(arr) = a.cast::<PyNdArray>() {
        return Ok(TypeArg::Concrete(arr.borrow().arr.dtype()));
    }
    // A numpy scalar is *strong*: `np.result_type(np.float64(1), np.float16)`
    // is float64. It has to be recognised before the Python-number tests
    // below, because our `float64` / `complex128` scalars subclass `float` /
    // `complex` (and `bool_` is not a `bool`, but `int64` is not an `int`).
    if let Some((d, _)) = convert::np_scalar(a)? {
        return Ok(TypeArg::Concrete(d));
    }
    // A bare Python number is *weak*: it contributes its kind, not its value.
    if a.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(TypeArg::Weak(WeakKind::Bool));
    }
    if a.is_instance_of::<pyo3::types::PyInt>() {
        // A weak integer, carrying the dtype it would have *alone*, which is
        // still value-based: `np.result_type(2**63)` is uint64 and
        // `np.result_type(2**100)` is object, but both are weak `int` as soon
        // as anything else joins them.
        let alone = if a.extract::<i64>().is_ok() {
            DType::I64
        } else if a.extract::<u64>().is_ok() {
            DType::U64
        } else {
            DType::Object
        };
        return Ok(TypeArg::WeakInt(alone));
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

// ---------------------------------------------------------------------------
// datetime64 / timedelta64 support
// ---------------------------------------------------------------------------

/// `np.datetime_data(dtype)`: the `(unit, count)` pair.
#[pyfunction]
fn datetime_data(dtype: &Bound<'_, PyAny>) -> PyResult<(String, u32)> {
    let d = descr_from_any(dtype)?;
    let m = rnp_core::datetime::meta_of(d.dt).ok_or_else(|| {
        PyTypeError::new_err(format!("cannot get datetime metadata from non-datetime {d:?}"))
    })?;
    Ok((
        rnp_core::datetime::UNIT_NAMES[m.base as usize].to_string(),
        m.num,
    ))
}

/// `np.isnat`.
#[pyfunction]
fn isnat(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyNdArray>> {
    let arr = array_from_any(a, None, false)?;
    wrap(py, rnp_core::datetime_ops::isnat(&arr).map_err(err)?)
}

/// Every element of a datetime-like array as its numpy `str()` rendering.
///
/// This is the engine half of `datetime_as_string`, of `astype('U')` and of
/// `str(np.datetime64(...))`; `unit` of `None` means "the array's own unit",
/// and `"auto"` is numpy's shortest-lossless choice.
#[pyfunction]
#[pyo3(signature = (a, unit = None, timezone = "naive", casting = "same_kind"))]
fn _datetime_strings(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    unit: Option<&str>,
    timezone: &str,
    casting: &str,
) -> PyResult<Py<PyAny>> {
    use rnp_core::datetime as dtm;
    let arr = array_from_any(a, None, false)?;
    let dt = arr.dtype();
    let meta = dtm::meta_of(dt).ok_or_else(|| {
        PyTypeError::new_err("cannot render a non-datetime array as datetime strings")
    })?;
    let cast = rnp_core::casting::Casting::from_str(casting)
        .ok_or_else(|| PyValueError::new_err(format!("casting must be one of ... got {casting}")))?;
    let utc = match timezone {
        "naive" => false,
        "UTC" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unsupported timezone input {other:?}: only 'naive' and 'UTC' are supported"
            )))
        }
    };
    let base = match unit {
        None => Some(meta.base),
        Some("auto") => None,
        Some(u) => Some(dtm::parse_unit(u).ok_or_else(|| {
            PyValueError::new_err(format!("Invalid datetime unit {u:?} in metadata"))
        })?),
    };
    let n = arr.to_native();
    let out = PyList::empty(py);
    for off in rnp_core::iter::offsets(&n.shape, &n.strides, n.byte_offset) {
        let v = match n.read_at(off) {
            rnp_core::Scalar::Int(i) => i,
            s => s.as_f64() as i64,
        };
        let text = if dt.is_timedelta() {
            dtm::timedelta_str(meta, v)
        } else if v == dtm::NAT {
            "NaT".to_string()
        } else {
            let dts = dtm::dt64_to_dts(meta, v).map_err(err)?;
            dtm::make_iso8601(&dts, base, utc, cast).map_err(err)?
        };
        out.append(text)?;
    }
    Ok(out.into_any().unbind())
}

/// The `S`/`U` width numpy gives `arr.astype('U')` for a datetime-like array.
#[pyfunction]
fn _datetime_string_len(dtype: &Bound<'_, PyAny>) -> PyResult<usize> {
    let d = descr_from_any(dtype)?;
    Ok(rnp_core::datetime::string_cast_len(d.dt))
}

/// `arr.astype(object)` for a datetime-like array: python `date`/`datetime`
/// objects for datetime64, `timedelta` for the units it can express, plain
/// ints for the rest, and `None` for NaT — exactly what numpy hands back.
#[pyfunction]
fn _datetime_objects(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use rnp_core::datetime as dtm;
    let arr = array_from_any(a, None, false)?;
    let dt = arr.dtype();
    let meta = dtm::meta_of(dt)
        .ok_or_else(|| PyTypeError::new_err("not a datetime array"))?;
    let dtmod = py.import("datetime")?;
    let n = arr.to_native();
    let out = PyList::empty(py);
    for off in rnp_core::iter::offsets(&n.shape, &n.strides, n.byte_offset) {
        let v = match n.read_at(off) {
            rnp_core::Scalar::Int(i) => i,
            s => s.as_f64() as i64,
        };
        if v == dtm::NAT {
            out.append(py.None())?;
            continue;
        }
        if dt.is_timedelta() {
            match dtm::timedelta_parts(meta, v) {
                Some((days, secs, us)) => {
                    let td = dtmod.getattr("timedelta")?.call1((days, secs, us))?;
                    out.append(td)?;
                }
                // Y/M and the sub-microsecond units have no `timedelta`
                // representation, so numpy hands back the raw integer.
                None => out.append(v)?,
            }
            continue;
        }
        // Years and months are out of `datetime`'s range as *offsets*, and
        // anything finer than microseconds cannot be represented, so numpy
        // returns the raw integer for those.
        if meta.base < dtm::UNIT_D || meta.base > dtm::UNIT_US {
            out.append(v)?;
            continue;
        }
        let dts = dtm::dt64_to_dts(meta, v).map_err(err)?;
        if dts.year < 1 || dts.year > 9999 {
            out.append(v)?;
            continue;
        }
        let obj = if meta.base <= dtm::UNIT_D {
            dtmod
                .getattr("date")?
                .call1((dts.year, dts.month, dts.day))?
        } else {
            dtmod.getattr("datetime")?.call1((
                dts.year, dts.month, dts.day, dts.hour, dts.min, dts.sec, dts.us,
            ))?
        };
        out.append(obj)?;
    }
    Ok(out.into_any().unbind())
}

/// The broken-down calendar fields of one datetime-like value.
///
/// For `M8` this is numpy's `npy_datetimestruct`
/// `(year, month, day, hour, min, sec, us, ps, as)`; for `m8` it is
/// `npy_timedeltastruct` padded into the same shape
/// `(0, 0, day, 0, 0, sec, us, ps, as)`. `None` for NaT, and for a
/// timedelta whose unit has no struct form.
#[pyfunction]
fn _datetime_struct(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    use rnp_core::datetime as dtm;
    let arr = array_from_any(a, None, false)?;
    let dt = arr.dtype();
    let meta =
        dtm::meta_of(dt).ok_or_else(|| PyTypeError::new_err("not a datetime array"))?;
    let n = arr.to_native();
    let out = PyList::empty(py);
    for off in rnp_core::iter::offsets(&n.shape, &n.strides, n.byte_offset) {
        let v = match n.read_at(off) {
            rnp_core::Scalar::Int(i) => i,
            s => s.as_f64() as i64,
        };
        if v == rnp_core::datetime::NAT {
            out.append(py.None())?;
            continue;
        }
        if dt.is_timedelta() {
            match dtm::timedelta_struct(meta, v) {
                Some((d, s, u, p, a_)) => {
                    out.append((0i64, 0i32, d, 0i32, 0i32, s, u, p, a_))?
                }
                None => out.append(py.None())?,
            }
            continue;
        }
        let s = dtm::dt64_to_dts(meta, v).map_err(err)?;
        out.append((
            s.year, s.month, s.day, s.hour, s.min, s.sec, s.us, s.ps, s.as_,
        ))?;
    }
    Ok(out.into_any().unbind())
}

/// Install the shim's `datetime64`/`timedelta64` scalar builder.
#[pyfunction]
fn _register_datetime_factory(f: Py<PyAny>) {
    pydtype::register_datetime_factory(f);
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
    adopt::mark_ndarray_as_sequence(m.py());
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
    m.add_function(wrap_pyfunction!(straggler::lexsort, m)?)?;
    m.add_function(wrap_pyfunction!(straggler::_reconstruct, m)?)?;
    m.add_function(wrap_pyfunction!(straggler::_frombuffer, m)?)?;
    m.add_function(wrap_pyfunction!(straggler::_c_concat, m)?)?;

    m.add_function(wrap_pyfunction!(_set_error_factories, m)?)?;

    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(asarray, m)?)?;
    m.add_function(wrap_pyfunction!(frombuffer, m)?)?;

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
    m.add_function(wrap_pyfunction!(datetime_data, m)?)?;
    m.add_function(wrap_pyfunction!(isnat, m)?)?;
    m.add_function(wrap_pyfunction!(_datetime_strings, m)?)?;
    m.add_function(wrap_pyfunction!(_datetime_string_len, m)?)?;
    m.add_function(wrap_pyfunction!(_datetime_objects, m)?)?;
    m.add_function(wrap_pyfunction!(_datetime_struct, m)?)?;
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
    m.add_function(wrap_pyfunction!(_register_datetime_factory, m)?)?;
    ufuncs::register(m)?;
    linalgops::register(m)?;

    m.add("__version__", "0.1.0")?;
    Ok(())
}
