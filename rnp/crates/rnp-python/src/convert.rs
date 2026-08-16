//! Python <-> `rnp-core` conversions: scalars, nested sequences, NEP 50
//! weak-scalar promotion.

use num_complex::Complex;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PyList, PySequence, PyString, PyTuple};

use rnp_core::{promote, DType, NdArray, Scalar};

use crate::pyarray::PyNdArray;

/// Convert a Python scalar to a core `Scalar`. Returns `None` for anything
/// that is not a recognised scalar (sequences, arrays, ...).
pub fn scalar_from_py(obj: &Bound<'_, PyAny>) -> Option<Scalar> {
    // bool first: it is a subclass of int.
    if obj.is_instance_of::<PyBool>() {
        return Some(Scalar::Bool(obj.extract::<bool>().ok()?));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Some(Scalar::Int(i));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Some(Scalar::Uint(u));
        }
        return None;
    }
    if obj.is_instance_of::<PyFloat>() {
        return Some(Scalar::Float(obj.extract::<f64>().ok()?));
    }
    if obj.is_instance_of::<PyComplex>() {
        let c = obj.cast::<PyComplex>().ok()?;
        return Some(Scalar::Complex(Complex::new(c.real(), c.imag())));
    }
    None
}

/// Turn a core `Scalar` into the natural Python object.
pub fn scalar_to_py<'py>(py: Python<'py>, s: Scalar) -> PyResult<Bound<'py, PyAny>> {
    Ok(match s {
        Scalar::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Scalar::Int(i) => i.into_pyobject(py)?.into_any(),
        Scalar::Uint(u) => u.into_pyobject(py)?.into_any(),
        Scalar::Float(f) => f.into_pyobject(py)?.into_any(),
        Scalar::Complex(c) => PyComplex::from_doubles(py, c.re, c.im).into_any(),
    })
}

/// True for objects that should be treated as nested sequences by `array()`.
fn is_sequence(obj: &Bound<'_, PyAny>) -> bool {
    if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBool>() {
        return false;
    }
    obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>()
}

fn discover_shape(obj: &Bound<'_, PyAny>, shape: &mut Vec<isize>) -> PyResult<()> {
    if !is_sequence(obj) {
        return Ok(());
    }
    let seq = obj.cast::<PySequence>()?;
    let n = seq.len()?;
    shape.push(n as isize);
    if n > 0 {
        discover_shape(&seq.get_item(0)?, shape)?;
    }
    Ok(())
}

fn flatten(obj: &Bound<'_, PyAny>, depth: usize, shape: &[isize], out: &mut Vec<Scalar>) -> PyResult<()> {
    if depth == shape.len() {
        let s = scalar_from_py(obj).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "unsupported element type in array(): {}",
                obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?;
        out.push(s);
        return Ok(());
    }
    if !is_sequence(obj) {
        return Err(PyValueError::new_err(
            "setting an array element with a sequence. The requested array has \
             an inhomogeneous shape.",
        ));
    }
    let seq = obj.cast::<PySequence>()?;
    let n = seq.len()? as isize;
    if n != shape[depth] {
        return Err(PyValueError::new_err(
            "setting an array element with a sequence. The requested array has \
             an inhomogeneous shape.",
        ));
    }
    for i in 0..n {
        flatten(&seq.get_item(i as usize)?, depth + 1, shape, out)?;
    }
    Ok(())
}

/// numpy's dtype discovery over a flat list of Python scalars.
fn infer_dtype(values: &[Scalar]) -> DType {
    if values.is_empty() {
        return DType::F64;
    }
    let mut d = values[0].natural_dtype();
    for v in &values[1..] {
        d = promote(d, v.natural_dtype());
    }
    d
}

/// `np.array` / `np.asarray` core: build an `NdArray` from any Python object.
pub fn array_from_any(
    obj: &Bound<'_, PyAny>,
    dtype: Option<DType>,
    copy: bool,
) -> PyResult<NdArray> {
    // Already one of ours.
    if let Ok(a) = obj.cast::<PyNdArray>() {
        let inner = a.borrow().arr.clone();
        return Ok(match dtype {
            Some(d) if d != inner.dtype => inner.astype(d),
            _ if copy => inner.copy(),
            _ => inner,
        });
    }
    // A bare scalar becomes a 0-d array.
    if let Some(s) = scalar_from_py(obj) {
        let d = dtype.unwrap_or_else(|| s.natural_dtype());
        let mut a = NdArray::zeros(vec![], d).map_err(crate::err)?;
        a.set(&[], s).map_err(crate::err)?;
        return Ok(a);
    }
    if is_sequence(obj) {
        let mut shape = Vec::new();
        discover_shape(obj, &mut shape)?;
        let mut values = Vec::new();
        flatten(obj, 0, &shape, &mut values)?;
        let d = dtype.unwrap_or_else(|| infer_dtype(&values));
        let flat = NdArray::from_scalars(&values, d).map_err(crate::err)?;
        return flat.reshape(&shape).map_err(crate::err);
    }
    // Anything else exposing __iter__ / __len__: go through list().
    if let Ok(seq) = obj.cast::<PySequence>() {
        let list = PyList::new(obj.py(), seq.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        return array_from_any(list.as_any(), dtype, copy);
    }
    Err(PyTypeError::new_err(format!(
        "could not convert {} to an array",
        obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()
    )))
}

/// NEP 50 promotion for `array OP python-scalar`: the Python scalar is
/// "weak" and adopts the array's dtype unless its own kind is higher.
///
/// Probed from numpy 2.5.2: `int8 + 1 -> int8`, `int8 + 1.0 -> float64`,
/// `float32 + 1j -> complex64`, `bool + 1 -> int64`.
pub fn weak_promote(arr: DType, s: Scalar) -> DType {
    let arr_level = match arr {
        DType::Bool => 0,
        d if d.is_integer() => 1,
        d if d.is_float() => 2,
        _ => 3,
    };
    let (scalar_level, _) = match s {
        Scalar::Bool(_) => (0, ()),
        Scalar::Int(_) | Scalar::Uint(_) => (1, ()),
        Scalar::Float(_) => (2, ()),
        Scalar::Complex(_) => (3, ()),
    };
    if scalar_level <= arr_level {
        return arr;
    }
    match scalar_level {
        1 => DType::I64,
        2 => DType::F64,
        // A complex scalar keeps the array's float precision when it has one.
        _ if arr == DType::F32 => DType::C64,
        _ => DType::C128,
    }
}

/// Coerce the right-hand operand of a binary op into an array, applying weak
/// scalar promotion against `self_dtype`.
pub fn operand(obj: &Bound<'_, PyAny>, self_dtype: DType) -> PyResult<Option<NdArray>> {
    if let Ok(a) = obj.cast::<PyNdArray>() {
        return Ok(Some(a.borrow().arr.clone()));
    }
    if let Some(s) = scalar_from_py(obj) {
        let d = weak_promote(self_dtype, s);
        let mut a = NdArray::zeros(vec![], d).map_err(crate::err)?;
        a.set(&[], s).map_err(crate::err)?;
        return Ok(Some(a));
    }
    if is_sequence(obj) {
        return Ok(Some(array_from_any(obj, None, false)?));
    }
    Ok(None)
}
