//! Python <-> `rnp-core` conversions: scalars, nested sequences, NEP 50
//! weak-scalar promotion.

use num_complex::Complex;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyComplex, PyFloat, PyInt, PyList, PySequence, PyString, PyTuple};

use rnp_core::descr::Descr;
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

/// Any scalar the port understands: a numpy scalar keeps its own value, a
/// Python number gives its natural one.
pub fn any_scalar(obj: &Bound<'_, PyAny>) -> PyResult<Option<Scalar>> {
    if let Some((_, s)) = np_scalar(obj)? {
        return Ok(Some(s));
    }
    Ok(scalar_from_py(obj))
}

/// Recognise one of the shim's numpy scalar objects.
///
/// The protocol is deliberately narrow: a numpy scalar carries both a `_v`
/// attribute (its Python-native payload) and a `dtype` that is one of our
/// `dtype` objects. `ndarray` has a `dtype` but no `_v`, and Python numbers
/// have neither, so nothing else is mistaken for one.
pub fn np_scalar(obj: &Bound<'_, PyAny>) -> PyResult<Option<(DType, Scalar)>> {
    Ok(np_scalar_descr(obj)?.map(|(d, s)| (d.dt, s)))
}

/// As [`np_scalar`], keeping the scalar's full descriptor — which is what
/// carries the C-type alias that makes
/// `np.array(np.longlong(2)).dtype.type is np.longlong` true.
pub fn np_scalar_descr(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Descr, Scalar)>> {
    let v = match obj.getattr(pyo3::intern!(obj.py(), "_v")) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let d = match obj.getattr(pyo3::intern!(obj.py(), "dtype")) {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let descr = match d.cast::<crate::pydtype::PyDType>() {
        Ok(p) => p.get().d,
        Err(_) => return Ok(None),
    };
    let dt = descr.dt;
    match scalar_from_py(&v) {
        Some(s) => Ok(Some((descr, s.cast(dt)))),
        None => Ok(None),
    }
}

/// Read one element of any dtype back as the Python object numpy would give,
/// *without* wrapping it in a numpy scalar type.
pub fn element_to_py<'py>(
    py: Python<'py>,
    arr: &NdArray,
    off: isize,
) -> PyResult<Bound<'py, PyAny>> {
    if arr.dtype().is_object() {
        return Ok(crate::objects::read(py, arr, off));
    }
    if arr.dtype().is_flexible() {
        return flexible_to_py(py, arr, off);
    }
    scalar_to_py(py, arr.read_at(off))
}

/// Build the numpy scalar object for one element, falling back to the plain
/// Python value when the shim has not registered a class for that dtype.
pub fn npscalar_to_py<'py>(
    py: Python<'py>,
    dt: DType,
    s: Scalar,
) -> PyResult<Bound<'py, PyAny>> {
    if dt.is_object() {
        let h = match s {
            Scalar::Uint(u) => u,
            _ => 0,
        };
        return Ok(crate::objects::resolve(py, h));
    }
    if let Some(w) = crate::pydtype::scalar_wrap(py, dt) {
        return w.call1((scalar_to_py(py, s)?,));
    }
    match crate::pydtype::scalar_class(py, dt) {
        Some(cls) => cls.call_method1(pyo3::intern!(py, "_wrap"), (scalar_to_py(py, s)?,)),
        None => scalar_to_py(py, s),
    }
}

/// As [`npscalar_to_py`], for the flexible dtypes whose payload is bytes/str.
pub fn npflexible_to_py<'py>(
    py: Python<'py>,
    arr: &NdArray,
    off: isize,
) -> PyResult<Bound<'py, PyAny>> {
    if arr.dtype().is_object() {
        return Ok(crate::objects::read(py, arr, off));
    }
    let v = flexible_to_py(py, arr, off)?;
    match crate::pydtype::scalar_class(py, arr.dtype()) {
        Some(cls) => cls.call1((v,)),
        None => Ok(v),
    }
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

/// One leaf of a nested sequence.
#[derive(Copy, Clone)]
struct Leaf {
    /// For a numpy scalar, the dtype it contributes *strongly* to the
    /// inferred result; `None` for a bare Python number.
    src: Option<DType>,
    val: Scalar,
    /// True when the leaf was a Python `int` too wide for any integer dtype.
    /// `val` then holds `float(obj)`, which is all the inexact dtypes need.
    huge: bool,
}

fn flatten(obj: &Bound<'_, PyAny>, depth: usize, shape: &[isize], out: &mut Vec<Leaf>) -> PyResult<()> {
    if depth == shape.len() {
        // A numpy scalar keeps its own dtype: `np.array([np.int8(1)])` is
        // int8, not int64.
        if let Some((d, v)) = np_scalar(obj)? {
            out.push(Leaf { src: Some(d), val: v, huge: false });
            return Ok(());
        }
        if let Some(f) = huge_int(obj)? {
            out.push(Leaf { src: None, val: Scalar::Float(f), huge: true });
            return Ok(());
        }
        let s = scalar_from_py(obj).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "unsupported element type in array(): {}",
                obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?;
        out.push(Leaf { src: None, val: s, huge: false });
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

/// numpy's dtype discovery over a flat list of leaves.
///
/// Array *coercion* is not NEP 50 promotion: a bare Python number contributes
/// its default dtype (int64 / float64 / complex128), not a weak one. Probed
/// against numpy 2.5.2: `np.array([np.float32(1), 2])` is float64, while
/// `np.float32(1) + 2` is float32.
fn infer_dtype(values: &[Leaf]) -> DType {
    if values.is_empty() {
        return DType::F64;
    }
    // A Python int too wide for uint64/int64 drags the whole array to object:
    // probed, `np.array([1, 2**100])` and `np.array([2**100, 1.0])` are both
    // `dtype('O')`.
    if values.iter().any(|l| l.huge) {
        return DType::Object;
    }
    let dt = |l: &Leaf| l.src.unwrap_or_else(|| l.val.natural_dtype());
    let mut d = dt(&values[0]);
    for v in &values[1..] {
        d = promote(d, dt(v));
    }
    d
}

/// `float(obj)` for a Python `int` too wide for `Scalar` (outside
/// `i64::MIN ..= u64::MAX`), or `None` when `obj` is not such an int.
///
/// numpy carries these as `object` when it can and rejects them with an
/// `OverflowError` when it cannot; `Scalar` cannot hold them at all, so the
/// float value is kept for the inexact dtypes that *can* take them
/// (`np.array(2**100, dtype=np.float32)` is `1.2676506e+30`).
pub fn huge_int(obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if obj.is_instance_of::<PyBool>() || !obj.is_instance_of::<PyInt>() {
        return Ok(None);
    }
    if obj.extract::<i64>().is_ok() || obj.extract::<u64>().is_ok() {
        return Ok(None);
    }
    // `float(huge)` itself overflows past ~1.8e308; numpy lets the value
    // saturate to an infinity there, so we do too.
    let negative = obj.lt(0i64)?;
    Ok(Some(obj.extract::<f64>().unwrap_or_else(|e| {
        e.restore(obj.py());
        PyErr::fetch(obj.py());
        if negative { f64::NEG_INFINITY } else { f64::INFINITY }
    })))
}

/// The inclusive value range of an integer dtype.
fn int_bounds(d: DType) -> Option<(i128, i128)> {
    Some(match d {
        DType::I8 => (i8::MIN as i128, i8::MAX as i128),
        DType::I16 => (i16::MIN as i128, i16::MAX as i128),
        DType::I32 => (i32::MIN as i128, i32::MAX as i128),
        DType::I64 => (i64::MIN as i128, i64::MAX as i128),
        DType::U8 => (0, u8::MAX as i128),
        DType::U16 => (0, u16::MAX as i128),
        DType::U32 => (0, u32::MAX as i128),
        DType::U64 => (0, u64::MAX as i128),
        _ => return None,
    })
}

/// numpy's `OverflowError` for a Python integer that cannot even reach the
/// C conversion function. Probed message, verbatim, for every target dtype:
/// `np.array(2**63, dtype=np.int64)`, `np.array([2**64], dtype=np.uint64)`,
/// `np.int64(2**100)` all say exactly this.
pub fn too_large() -> PyErr {
    pyo3::exceptions::PyOverflowError::new_err(
        "Python int too large to convert to C long",
    )
}

/// numpy's range check when a value is *stored into* an integer array by
/// `np.array(..., dtype=D)` (and the assignment paths that share it).
///
/// The rule is not symmetric and was probed one cell at a time (see
/// `harness/dev_check_nep50.py`, which asserts the whole matrix):
///
/// | leaf                     | signed target | unsigned target |
/// |--------------------------|---------------|-----------------|
/// | Python `int` / `float`   | OverflowError | OverflowError   |
/// | numpy scalar             | OverflowError | wraps silently  |
///
/// and, either way, a value outside the C conversion type raises
/// [`too_large`] instead. Python `bool` and non-finite floats are never
/// checked.
pub fn check_int_store(target: DType, src: Option<DType>, v: Scalar) -> PyResult<()> {
    let Some((lo, hi)) = int_bounds(target) else {
        return Ok(());
    };
    let n: i128 = match v {
        // `np.array([True], dtype=np.int8)` is 1, never an error.
        Scalar::Bool(_) => return Ok(()),
        Scalar::Int(i) => i as i128,
        Scalar::Uint(u) => u as i128,
        // numpy converts through `int(obj)`, which truncates toward zero.
        Scalar::Float(f) => {
            if !f.is_finite() {
                return Ok(());
            }
            let t = f.trunc();
            if t >= 1.7e38 {
                i128::MAX
            } else if t <= -1.7e38 {
                i128::MIN
            } else {
                t as i128
            }
        }
        Scalar::Complex(_) => return Ok(()),
    };
    let signed = target.category() == rnp_core::dtype::Kind::Int;
    // Signed targets convert through a C `long`; unsigned ones accept the
    // whole `[i64::MIN, u64::MAX]` span before wrapping.
    let reaches_c = if signed {
        n >= i64::MIN as i128 && n <= i64::MAX as i128
    } else {
        n >= i64::MIN as i128 && n <= u64::MAX as i128
    };
    if !reaches_c {
        return Err(too_large());
    }
    if n < lo || n > hi {
        if signed || src.is_none() {
            return Err(pyo3::exceptions::PyOverflowError::new_err(format!(
                "Python integer {n} out of bounds for {}",
                target.name()
            )));
        }
    }
    Ok(())
}

/// Does the first leaf of a (possibly nested) sequence look like text?
fn first_leaf_is_text(obj: &Bound<'_, PyAny>) -> bool {
    if is_text(obj) {
        return true;
    }
    let mut cur = obj.clone();
    loop {
        if !is_sequence(&cur) {
            return is_text(&cur);
        }
        let seq = match cur.cast::<PySequence>() {
            Ok(s) => s,
            Err(_) => return false,
        };
        match seq.len() {
            Ok(n) if n > 0 => match seq.get_item(0) {
                Ok(next) => cur = next,
                Err(_) => return false,
            },
            _ => return false,
        }
    }
}

/// True for the Python objects that become elements of an `S`/`U` array.
fn is_text(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBytes>()
}

/// The element bytes and logical length of one `S`/`U` value.
fn text_bytes(obj: &Bound<'_, PyAny>, dt: DType) -> PyResult<(Vec<u8>, usize)> {
    match dt {
        DType::Bytes(_) => {
            let b: Vec<u8> = if let Ok(s) = obj.cast::<PyBytes>() {
                s.as_bytes().to_vec()
            } else if let Ok(s) = obj.cast::<PyString>() {
                // numpy encodes str into an S array as ASCII/latin-1 bytes.
                s.to_str()?.as_bytes().to_vec()
            } else {
                return Err(PyTypeError::new_err(
                    "only str and bytes can fill a bytes ('S') array",
                ));
            };
            let n = b.len();
            Ok((b, n))
        }
        DType::Str(_) => {
            let s: String = if let Ok(s) = obj.cast::<PyString>() {
                s.to_str()?.to_string()
            } else if let Ok(b) = obj.cast::<PyBytes>() {
                String::from_utf8_lossy(b.as_bytes()).into_owned()
            } else {
                return Err(PyTypeError::new_err(
                    "only str and bytes can fill a str ('U') array",
                ));
            };
            let mut out = Vec::with_capacity(s.chars().count() * 4);
            let mut n = 0usize;
            for c in s.chars() {
                out.extend_from_slice(&(c as u32).to_le_bytes());
                n += 1;
            }
            Ok((out, n))
        }
        _ => Err(PyTypeError::new_err("not a flexible dtype")),
    }
}

/// Read one `S`/`U`/`V` element back as the Python object numpy would give.
pub fn flexible_to_py<'py>(
    py: Python<'py>,
    arr: &NdArray,
    off: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let raw = arr.element_bytes_at(off);
    let raw: &[u8] = &raw;
    match arr.dtype() {
        DType::Bytes(_) => {
            let end = raw.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
            Ok(PyBytes::new(py, &raw[..end]).into_any())
        }
        DType::Str(_) => {
            let mut s = String::new();
            for c in raw.chunks_exact(4) {
                let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                if v == 0 {
                    continue;
                }
                s.push(char::from_u32(v).unwrap_or('\u{fffd}'));
            }
            // numpy strips only trailing NULs; interior ones are kept, but
            // the common case has none at all.
            Ok(PyString::new(py, &s).into_any())
        }
        _ => Ok(PyBytes::new(py, raw).into_any()),
    }
}

/// Build an `S`/`U` array from nested Python sequences of str/bytes.
fn array_from_text(obj: &Bound<'_, PyAny>, dtype: Option<DType>) -> PyResult<NdArray> {
    let mut shape = Vec::new();
    discover_shape(obj, &mut shape)?;
    let mut items: Vec<Bound<'_, PyAny>> = Vec::new();
    collect_objects(obj, 0, &shape, &mut items)?;

    // An unsized (or absent) dtype takes its width from the data, exactly as
    // `np.array(['ab', 'cde'])` gives `<U3`.
    let base = match dtype {
        Some(d @ (DType::Bytes(_) | DType::Str(_))) => d,
        _ => {
            if items.iter().all(|o| o.is_instance_of::<PyBytes>()) && !items.is_empty() {
                DType::Bytes(0)
            } else {
                DType::Str(0)
            }
        }
    };
    let mut encoded = Vec::with_capacity(items.len());
    let mut width = 0usize;
    for it in &items {
        let (bytes, n) = text_bytes(it, base)?;
        width = width.max(n);
        encoded.push(bytes);
    }
    let dt = match base {
        DType::Bytes(0) => DType::Bytes(width as u32),
        DType::Str(0) => DType::Str(width as u32),
        other => other,
    };
    let out = NdArray::zeros(shape, dt).map_err(crate::err)?;
    let itemsize = out.itemsize() as isize;
    for (i, bytes) in encoded.iter().enumerate() {
        out.write_raw_at(out.byte_offset + i as isize * itemsize, bytes);
    }
    Ok(out)
}

/// Flatten nested sequences into the leaf Python objects.
fn collect_objects<'py>(
    obj: &Bound<'py, PyAny>,
    depth: usize,
    shape: &[isize],
    out: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    if depth == shape.len() {
        out.push(obj.clone());
        return Ok(());
    }
    let seq = obj.cast::<PySequence>()?;
    for i in 0..shape[depth] {
        collect_objects(&seq.get_item(i as usize)?, depth + 1, shape, out)?;
    }
    Ok(())
}

/// `np.array` / `np.asarray` core: build an `NdArray` from any Python object.
/// [`array_from_any`] with a full descriptor: build in the host's byte order,
/// then relabel (and swap) once, at the end.
///
/// This is the whole byte-swap policy in one place — nothing below ever sees
/// non-native bytes, and the native path pays a single equality test.
pub fn array_from_any_descr(
    obj: &Bound<'_, PyAny>,
    descr: Option<Descr>,
    copy: bool,
) -> PyResult<NdArray> {
    let a = array_from_any(obj, descr.map(|d| d.dt), copy)?;
    match descr {
        // The *resolved* storage dtype is what the array actually has: an
        // unsized `'>U'` request comes back as `U1`, so the byte order and
        // alias are re-applied to that, not to the request.
        Some(d) => {
            let target = rnp_core::descr::Descr::with_alias(a.dtype(), d.bo, d.alias);
            Ok(a.into_descr(target))
        }
        None => Ok(a),
    }
}

pub fn array_from_any(
    obj: &Bound<'_, PyAny>,
    dtype: Option<DType>,
    copy: bool,
) -> PyResult<NdArray> {
    // Already one of ours.
    if let Ok(a) = obj.cast::<PyNdArray>() {
        let inner = a.borrow().arr.clone();
        return Ok(match dtype {
            Some(d) if d != inner.dtype() => inner.astype(d),
            _ if copy => inner.copy(),
            _ => inner,
        });
    }
    // Object arrays store handles into the interning slab.
    if dtype == Some(DType::Object) {
        return crate::objects::array_from_objects(obj);
    }
    // A numpy scalar is a strong 0-d operand.
    if let Some((sd, sv)) = np_scalar_descr(obj)? {
        let d = dtype.unwrap_or(sd.dt);
        let mut a = NdArray::zeros(vec![], d).map_err(crate::err)?;
        a.set(&[], sv.cast(d)).map_err(crate::err)?;
        if dtype.is_none() {
            // Keep the scalar's own descriptor, so `np.longlong` stays `'q'`.
            a.descr = sd;
        }
        return Ok(a);
    }
    // Structured dtypes: the "scalar" of the array is a Python tuple, one
    // entry per field.
    if let Some(DType::Struct(id)) = dtype {
        return array_from_records(obj, id);
    }
    // Flexible dtypes take a completely separate path: their elements are
    // Python str/bytes, which `Scalar` cannot carry.
    let wants_text = matches!(dtype, Some(DType::Bytes(_)) | Some(DType::Str(_)));
    if wants_text || (dtype.is_none() && first_leaf_is_text(obj)) {
        return array_from_text(obj, dtype);
    }
    // A Python int too wide for any integer dtype: object, or an error.
    if let Some(f) = huge_int(obj)? {
        match dtype {
            None => return crate::objects::array_from_objects(obj),
            Some(d) if int_bounds(d).is_some() => return Err(too_large()),
            Some(d) => {
                let mut a = NdArray::zeros(vec![], d).map_err(crate::err)?;
                a.set(&[], Scalar::Float(f)).map_err(crate::err)?;
                return Ok(a);
            }
        }
    }
    // A bare scalar becomes a 0-d array.
    if let Some(s) = scalar_from_py(obj) {
        let d = dtype.unwrap_or_else(|| s.natural_dtype());
        if dtype.is_some() {
            check_int_store(d, None, s)?;
        }
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
        if d == DType::Object {
            return crate::objects::array_from_objects(obj);
        }
        if values.iter().any(|l| l.huge) && int_bounds(d).is_some() {
            return Err(too_large());
        }
        if dtype.is_some() {
            for l in &values {
                check_int_store(d, l.src, l.val)?;
            }
        }
        let plain: Vec<Scalar> = values.iter().map(|l| l.val).collect();
        let flat = NdArray::from_scalars(&plain, d).map_err(crate::err)?;
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

/// Build a structured array from nested sequences of record tuples.
fn array_from_records(obj: &Bound<'_, PyAny>, id: u32) -> PyResult<NdArray> {
    let def = rnp_core::descr::registry::struct_def(id);
    let nfields = def.fields.len();
    // Descend list/tuple nesting until the first record tuple (a tuple whose
    // length matches the field count) is reached.
    let mut shape: Vec<isize> = Vec::new();
    let mut cur = obj.clone();
    loop {
        let is_record = cur
            .cast::<PyTuple>()
            .map(|t| t.len() == nfields)
            .unwrap_or(false);
        if is_record || !is_sequence(&cur) {
            break;
        }
        let seq = cur.cast::<PySequence>()?;
        let n = seq.len()?;
        shape.push(n as isize);
        if n == 0 {
            break;
        }
        cur = seq.get_item(0)?;
    }
    let mut records: Vec<Bound<'_, PyAny>> = Vec::new();
    collect_objects(obj, 0, &shape, &mut records)?;

    let out = NdArray::zeros(shape, DType::Struct(id)).map_err(crate::err)?;
    let isz = out.itemsize() as isize;
    for (i, rec) in records.iter().enumerate() {
        let base = i as isize * isz;
        let items: Vec<Bound<'_, PyAny>> = if let Ok(t) = rec.cast::<PyTuple>() {
            t.iter().collect()
        } else if let Ok(l) = rec.cast::<PyList>() {
            l.iter().collect()
        } else {
            return Err(PyTypeError::new_err(
                "a structured array element must be a tuple of field values",
            ));
        };
        if items.len() != nfields {
            return Err(PyValueError::new_err(format!(
                "could not assign tuple of length {} to structure with {} fields.",
                items.len(),
                nfields
            )));
        }
        for (f, val) in def.fields.iter().zip(items.iter()) {
            let off = base + f.offset as isize;
            let fdt = f.descr.dt;
            if matches!(fdt, DType::Bytes(_) | DType::Str(_)) {
                let (bytes, _) = text_bytes(val, fdt)?;
                let n = fdt.itemsize().min(bytes.len());
                // SAFETY: `off` addresses this field inside a zeroed record.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.buffer.as_mut_ptr().offset(off),
                        n,
                    );
                }
            } else {
                let sv = scalar_from_py(val).ok_or_else(|| {
                    PyTypeError::new_err("unsupported field value in a structured array")
                })?;
                let mut field = out.clone();
                field.descr = f.descr;
                field.write_at(off, sv);
            }
        }
    }
    Ok(out)
}

/// NEP 50 promotion for `array OP python-scalar`, taking the scalar's *value*
/// into account when its kind alone would not hold it.
///
/// The kind rule is [`weak_kind_promote`]. On top of it, an integer that does
/// not fit the dtype that rule picks widens to one that does:
/// `np.uint8(10) < -1` has to answer `False`, not `10 < 255`, and the only way
/// to get there is to compare in a dtype that holds `-1`.
///
/// numpy raises an `OverflowError` instead for the *arithmetic* ops (see
/// [`weak_dtype`], which is what the array path uses); the widening here is
/// the answer for everything that cannot raise.
pub fn weak_promote(arr: DType, s: Scalar) -> DType {
    let d = weak_kind_promote(arr, s);
    let v: i128 = match s {
        Scalar::Int(i) => i as i128,
        Scalar::Uint(u) => u as i128,
        _ => return d,
    };
    match int_bounds(d) {
        Some((lo, hi)) if v < lo || v > hi => promote(d, s.natural_dtype()),
        _ => d,
    }
}

/// The kind-only half of NEP 50 weak promotion: the Python scalar adopts the
/// array's dtype unless its own kind is higher.
///
/// Probed from numpy 2.5.2: `int8 + 1 -> int8`, `int8 + 1.0 -> float64`,
/// `float32 + 1j -> complex64`, `bool + 1 -> int64`.
pub fn weak_kind_promote(arr: DType, s: Scalar) -> DType {
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
        // A complex scalar keeps the array's float precision when it has one;
        // there is no complex32, so float16 also lands on complex64.
        _ if arr == DType::F32 || arr == DType::F16 => DType::C64,
        _ => DType::C128,
    }
}

/// The dtype a weak Python scalar adopts, or numpy's OverflowError when the
/// value does not fit.
///
/// NEP 50: `np.uint8(1) + 300` is an OverflowError, while `np.uint8(200) < -1`
/// still answers correctly -- so a comparison widens the *weak* operand to its
/// own natural dtype instead of raising.
pub fn weak_dtype(arr: DType, s: Scalar, comparison: bool) -> PyResult<DType> {
    let d = weak_kind_promote(arr, s);
    let v: i128 = match s {
        Scalar::Int(i) => i as i128,
        Scalar::Uint(u) => u as i128,
        _ => return Ok(d),
    };
    let Some((lo, hi)) = int_bounds(d) else {
        return Ok(d);
    };
    if v >= lo && v <= hi {
        return Ok(d);
    }
    if comparison {
        return Ok(s.natural_dtype());
    }
    Err(pyo3::exceptions::PyOverflowError::new_err(format!(
        "Python integer {v} out of bounds for {d}"
    )))
}

/// Coerce the right-hand operand of a binary op into an array, applying weak
/// scalar promotion against `self_dtype`.
pub fn operand(obj: &Bound<'_, PyAny>, self_dtype: DType) -> PyResult<Option<NdArray>> {
    operand_for(obj, self_dtype, false)
}

/// As [`operand`], telling the weak-scalar rule whether the op is a
/// comparison.
pub fn operand_for(
    obj: &Bound<'_, PyAny>,
    self_dtype: DType,
    comparison: bool,
) -> PyResult<Option<NdArray>> {
    if let Ok(a) = obj.cast::<PyNdArray>() {
        return Ok(Some(a.borrow().arr.clone()));
    }
    // numpy scalars are *strong* under NEP 50: they keep their own dtype.
    if let Some((d, s)) = np_scalar(obj)? {
        let mut a = NdArray::zeros(vec![], d).map_err(crate::err)?;
        a.set(&[], s).map_err(crate::err)?;
        return Ok(Some(a));
    }
    // A Python int too wide for any integer dtype.
    if let Some(f) = huge_int(obj)? {
        let exact = self_dtype.is_float() || self_dtype.is_complex();
        if exact {
            // Probed: `np.arange(3.0) + 2**100` is float64 `1.2676506e+30`.
            let mut a = NdArray::zeros(vec![], self_dtype).map_err(crate::err)?;
            a.set(&[], Scalar::Float(f)).map_err(crate::err)?;
            return Ok(Some(a));
        }
        if !(self_dtype.is_integer() || self_dtype == DType::Bool) {
            return Ok(None);
        }
        if !comparison {
            // Probed: `np.arange(3) + 2**100` is this OverflowError.
            return Err(too_large());
        }
        // A comparison still has to answer, and correctly: probed,
        // `np.arange(3, dtype=np.uint8) < 2**100` is all-True. Every value an
        // integer array can hold lies inside (-2**63, 2**64), so replacing the
        // scalar with +-2**65 gives every one of the six comparisons the same
        // answer as the true value would -- exactly, with no rounding to worry
        // about, since 2**65 is a power of two.
        const BEYOND: f64 = 36893488147419103232.0; // 2**65
        let v = Scalar::Float(if f < 0.0 { -BEYOND } else { BEYOND });
        let mut a = NdArray::zeros(vec![], DType::F64).map_err(crate::err)?;
        a.set(&[], v).map_err(crate::err)?;
        return Ok(Some(a));
    }
    if let Some(s) = scalar_from_py(obj) {
        let d = weak_dtype(self_dtype, s, comparison)?;
        let mut a = NdArray::zeros(vec![], d).map_err(crate::err)?;
        a.set(&[], s).map_err(crate::err)?;
        return Ok(Some(a));
    }
    if is_sequence(obj) {
        return Ok(Some(array_from_any(obj, None, false)?));
    }
    Ok(None)
}
