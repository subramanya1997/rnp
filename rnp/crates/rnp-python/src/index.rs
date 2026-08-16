//! Parsing Python index expressions into `rnp_core::IndexItem`s.

use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PySlice, PyString, PyTuple};

use rnp_core::indexing::{IndexItem, SliceSpec};
use rnp_core::{DType, NdArray};

use crate::convert::array_from_any;
use crate::pyarray::PyNdArray;

/// numpy's message for an index element it does not understand.
pub const BAD_INDEX: &str = "only integers, slices (`:`), ellipsis (`...`), \
                             numpy.newaxis (`None`) and integer or boolean arrays are valid indices";

/// Classify one already-materialised array as an index.
fn array_item(a: NdArray) -> PyResult<IndexItem> {
    if a.dtype == DType::Bool {
        if a.ndim() == 0 {
            let v = matches!(a.get_flat(0), rnp_core::Scalar::Bool(true));
            return Ok(IndexItem::ZeroDBool(v));
        }
        return Ok(IndexItem::BoolArray(a));
    }
    if a.dtype.is_integer() {
        return Ok(IndexItem::IntArray(a));
    }
    Err(PyIndexError::new_err(
        "arrays used as indices must be of integer (or boolean) type",
    ))
}

/// Is this object a list/tuple that should become an index *array*?
fn is_index_sequence(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>()
}

fn slice_spec(s: &Bound<'_, PySlice>) -> PyResult<SliceSpec> {
    let get = |name: &str| -> PyResult<Option<isize>> {
        let v = s.getattr(name)?;
        if v.is_none() {
            return Ok(None);
        }
        match index_of(&v) {
            Some(i) => Ok(Some(i)),
            None => Err(PyTypeError::new_err(
                "slice indices must be integers or None or have an __index__ method",
            )),
        }
    };
    Ok(SliceSpec {
        start: get("start")?,
        stop: get("stop")?,
        step: get("step")?,
    })
}

/// `operator.index(obj)` without raising: `None` when the object is not an
/// integer (floats and numpy floats included).
fn index_of(obj: &Bound<'_, PyAny>) -> Option<isize> {
    if obj.is_instance_of::<PyBool>() {
        return None;
    }
    if obj.is_instance_of::<pyo3::types::PyInt>() {
        return obj.extract::<isize>().ok();
    }
    if obj.is_instance_of::<pyo3::types::PyFloat>() || obj.is_instance_of::<PyString>() {
        return None;
    }
    // Anything else must implement __index__ (numpy accepts those).
    if obj.hasattr("__index__").unwrap_or(false) {
        if let Ok(v) = obj.call_method0("__index__") {
            return v.extract::<isize>().ok();
        }
    }
    None
}

/// Parse a single element of an index expression.
fn parse_item(obj: &Bound<'_, PyAny>) -> PyResult<IndexItem> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(IndexItem::NewAxis);
    }
    if obj.is(&py.Ellipsis()) {
        return Ok(IndexItem::Ellipsis);
    }
    if let Ok(s) = obj.cast::<PySlice>() {
        return Ok(IndexItem::Slice(slice_spec(s)?));
    }
    // A bare Python bool is a 0-d boolean index, never an integer.
    if obj.is_instance_of::<PyBool>() {
        return Ok(IndexItem::ZeroDBool(obj.extract::<bool>()?));
    }
    if let Ok(a) = obj.cast::<PyNdArray>() {
        let inner = a.borrow().arr.clone();
        return array_item(inner);
    }
    if let Some(i) = index_of(obj) {
        return Ok(IndexItem::Int(i));
    }
    if is_index_sequence(obj) {
        let a = array_from_any(obj, None, false)
            .map_err(|_| PyIndexError::new_err(BAD_INDEX))?;
        // `a[[]]` is a legal empty fancy index even though the empty list
        // coerces to float64.
        if a.size() == 0 && !a.dtype.is_integer() && a.dtype != DType::Bool {
            return Ok(IndexItem::IntArray(a.astype(DType::I64)));
        }
        if !a.dtype.is_integer() && a.dtype != DType::Bool {
            return Err(PyIndexError::new_err(BAD_INDEX));
        }
        return array_item(a);
    }
    Err(PyIndexError::new_err(BAD_INDEX))
}

/// Parse a whole `a[key]` expression.
pub fn parse_index(key: &Bound<'_, PyAny>) -> PyResult<Vec<IndexItem>> {
    if let Ok(t) = key.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(t.len());
        for it in t.iter() {
            out.push(parse_item(&it)?);
        }
        return Ok(out);
    }
    Ok(vec![parse_item(key)?])
}
