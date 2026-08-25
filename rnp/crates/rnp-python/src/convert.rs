//! Python <-> `rnp-core` conversions: scalars, nested sequences, NEP 50
//! weak-scalar promotion.

use num_complex::Complex;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyComplex, PyFloat, PyInt, PyList, PySequence, PyString, PyTuple};

use rnp_core::descr::Descr;
use rnp_core::{promote, C160, DType, F80, NdArray, Scalar};

use crate::pyarray::PyNdArray;

#[pyclass(name = "_F80Value", module = "_rnp", frozen, skip_from_py_object)]
#[derive(Copy, Clone)]
pub struct PyF80Value { pub value: F80 }

#[pymethods]
impl PyF80Value {
    fn __float__(&self) -> f64 { self.value.to_f64() }
    fn __int__(&self) -> i64 { self.value.to_f64() as i64 }
    fn __bool__(&self) -> bool { !self.value.is_zero() }
    fn __str__(&self) -> String { self.value.to_shortest_string() }
    fn __repr__(&self) -> String { self.value.to_shortest_string() }
    #[getter] fn real(&self) -> Self { *self }
    #[getter] fn imag(&self) -> Self { Self { value: F80::ZERO } }
}

#[pyclass(name = "_C160Value", module = "_rnp", frozen, skip_from_py_object)]
#[derive(Copy, Clone)]
pub struct PyC160Value { pub value: C160 }

#[pymethods]
impl PyC160Value {
    fn __complex__<'py>(&self, py: Python<'py>) -> Bound<'py, PyComplex> {
        PyComplex::from_doubles(py, self.value.re.to_f64(), self.value.im.to_f64())
    }
    fn __bool__(&self) -> bool { !self.value.re.is_zero() || !self.value.im.is_zero() }
    #[getter] fn real(&self) -> PyF80Value { PyF80Value { value: self.value.re } }
    #[getter] fn imag(&self) -> PyF80Value { PyF80Value { value: self.value.im } }
    fn conjugate(&self) -> Self { Self { value: C160 { re: self.value.re, im: self.value.im.neg() } } }
}

/// Convert a Python scalar to a core `Scalar`. Returns `None` for anything
/// that is not a recognised scalar (sequences, arrays, ...).
pub fn scalar_from_py(obj: &Bound<'_, PyAny>) -> Option<Scalar> {
    if let Ok(v) = obj.extract::<PyRef<'_, PyF80Value>>() { return Some(Scalar::Float80(v.value)); }
    if let Ok(v) = obj.extract::<PyRef<'_, PyC160Value>>() { return Some(Scalar::Complex160(v.value)); }
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
        Scalar::Float80(f) => Py::new(py, PyF80Value { value: f })?.into_bound(py).into_any(),
        Scalar::Complex(c) => PyComplex::from_doubles(py, c.re, c.im).into_any(),
        Scalar::Complex160(c) => Py::new(py, PyC160Value { value: c })?.into_bound(py).into_any(),
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

/// A scalar used as the right-hand side of assignment.  Unlike direct 0-d
/// construction (`np.array(np.float64(np.nan), dtype=int)`), assignment uses
/// NumPy's scalar-to-C conversion rules and therefore performs the same
/// integer validation as a scalar nested inside a sequence.
pub fn assignment_scalar(obj: &Bound<'_, PyAny>, target: DType) -> PyResult<Option<Scalar>> {
    if let Some((src, value)) = np_scalar(obj)? {
        check_int_store(target, Some(src), value)?;
        return Ok(Some(value));
    }
    if let Some(value) = scalar_from_py(obj) {
        check_int_store(target, None, value)?;
        return Ok(Some(value));
    }
    Ok(None)
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
    if arr.dtype().is_string() {
        return Ok(crate::objects::read_string(py, arr, off));
    }
    if arr.dtype().is_object() {
        return Ok(crate::objects::read(py, arr, off));
    }
    if arr.dtype().is_flexible() {
        return flexible_to_py(py, arr, off);
    }
    if arr.dtype().is_datetime_like() {
        let v = match arr.read_at(off) {
            Scalar::Int(i) => i,
            s => s.as_f64() as i64,
        };
        return datetime_object(py, arr.dtype(), v);
    }
    scalar_to_py(py, arr.read_at(off))
}

/// numpy's `DATETIME_getitem` / `TIMEDELTA_getitem`: the Python object one
/// datetime-like element hands back — a `datetime.date` / `datetime.datetime` /
/// `datetime.timedelta` where that is exact, `None` for NaT, and the raw
/// integer for every unit or magnitude `datetime` cannot express.
pub fn datetime_object<'py>(
    py: Python<'py>,
    dt: DType,
    v: i64,
) -> PyResult<Bound<'py, PyAny>> {
    use rnp_core::datetime::PyDtObj;
    let Some(what) = rnp_core::datetime::value_to_pyobj(dt, v) else {
        return scalar_to_py(py, Scalar::Int(v));
    };
    let dtmod = || py.import(pyo3::intern!(py, "datetime"));
    match what {
        PyDtObj::Nothing => Ok(py.None().into_bound(py)),
        PyDtObj::Int(i) => Ok(i.into_pyobject(py)?.into_any()),
        PyDtObj::Date { year, month, day } => dtmod()?
            .getattr(pyo3::intern!(py, "date"))?
            .call1((year, month, day)),
        PyDtObj::DateTime {
            year,
            month,
            day,
            hour,
            min,
            sec,
            us,
        } => dtmod()?
            .getattr(pyo3::intern!(py, "datetime"))?
            .call1((year, month, day, hour, min, sec, us)),
        PyDtObj::Delta { days, secs, us } => dtmod()?
            .getattr(pyo3::intern!(py, "timedelta"))?
            .call1((days, secs, us)),
    }
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
    if dt.is_datetime_like() {
        if let Some(r) = crate::pydtype::datetime_scalar(
            py,
            dt,
            match s {
                Scalar::Int(i) => i,
                other => other.as_f64() as i64,
            },
        ) {
            return r;
        }
    }
    if let Some(result) = crate::ufuncs::native_builtin_scalar(py, dt, s) {
        return result;
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
    if arr.dtype().is_string() {
        return Ok(crate::objects::read_string(py, arr, off));
    }
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
    // numpy's `NPY_MAXDIMS`. A co-recursive list (gh-11154) is unbounded, and
    // numpy answers with this ValueError rather than overflowing the stack.
    if shape.len() >= 64 {
        return Err(PyValueError::new_err(
            "setting an array element with a sequence. The requested array \
             would exceed the maximum number of dimension of 64.",
        ));
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
    /// Set when the leaf was a Python `int` too wide for any integer dtype.
    huge: Option<Huge>,
}

fn flatten(obj: &Bound<'_, PyAny>, depth: usize, shape: &[isize], out: &mut Vec<Leaf>) -> PyResult<()> {
    if depth == shape.len() {
        // A numpy scalar keeps its own dtype: `np.array([np.int8(1)])` is
        // int8, not int64.
        if let Some((d, v)) = np_scalar(obj)? {
            out.push(Leaf { src: Some(d), val: v, huge: None });
            return Ok(());
        }
        if let Some(h) = huge_int(obj)? {
            out.push(Leaf { src: None, val: h.placeholder(), huge: Some(h) });
            return Ok(());
        }
        let s = scalar_from_py(obj).ok_or_else(|| {
            PyTypeError::new_err(format!(
                "unsupported element type in array(): {}",
                obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?;
        out.push(Leaf { src: None, val: s, huge: None });
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
    if values.iter().any(|l| l.huge.is_some()) {
        return DType::Object;
    }
    let dt = |l: &Leaf| l.src.unwrap_or_else(|| l.val.natural_dtype());
    let mut d = dt(&values[0]);
    for v in &values[1..] {
        d = promote(d, dt(v));
        if d == DType::Void(0) {
            // `promote` uses unsized void as its no-common-dtype sentinel.
            // Array discovery boxes such heterogeneous values instead of
            // materialising a meaningless V0 array.
            return DType::Object;
        }
    }
    d
}

/// A Python `int` too wide for `Scalar` (outside `i64::MIN ..= u64::MAX`).
///
/// numpy carries these as `object` when it can, converts them to a C `double`
/// for the inexact dtypes (`np.array(2**100, dtype=np.float32)` is
/// `1.2676506e+30`), and raises an `OverflowError` when neither works.
#[derive(Copy, Clone, Debug)]
pub struct Huge {
    /// `float(obj)`, or `None` when even that overflowed (past ~1.8e308).
    /// Probed: numpy then raises `OverflowError: int too large to convert to
    /// float`, but *only* if the target dtype is inexact --
    /// `np.array(2**2000, dtype=bool)` is `True` and `np.array(2**2000)` is
    /// an object array.
    pub as_f64: Option<f64>,
    pub negative: bool,
}

impl Huge {
    /// The value to store in an inexact array, or numpy's `OverflowError`.
    fn float(self) -> PyResult<f64> {
        self.as_f64.ok_or_else(|| {
            pyo3::exceptions::PyOverflowError::new_err("int too large to convert to float")
        })
    }

    /// The `Scalar` this contributes to a *non*-inexact target, where the
    /// exact value never matters: `bool` only asks whether it is zero (it
    /// never is), and integer targets reject it outright.
    fn placeholder(self) -> Scalar {
        Scalar::Float(self.as_f64.unwrap_or(if self.negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }))
    }
}

/// Classify `obj` as a [`Huge`] Python integer, or `None`.
pub fn huge_int(obj: &Bound<'_, PyAny>) -> PyResult<Option<Huge>> {
    if obj.is_instance_of::<PyBool>() || !obj.is_instance_of::<PyInt>() {
        return Ok(None);
    }
    if obj.extract::<i64>().is_ok() || obj.extract::<u64>().is_ok() {
        return Ok(None);
    }
    let negative = obj.lt(0i64)?;
    let as_f64 = match obj.extract::<f64>() {
        Ok(f) => Some(f),
        Err(e) => {
            // Swallow the CPython error; whether it is fatal depends on the
            // target dtype, which the caller decides.
            e.restore(obj.py());
            PyErr::fetch(obj.py());
            None
        }
    };
    Ok(Some(Huge { as_f64, negative }))
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
/// [`too_large`] instead. Python `bool` is never checked. Non-finite floats
/// raise through integer conversion, except that numpy float scalars assigned
/// to unsigned arrays use the unsigned casting loop and wrap.
pub fn check_int_store(target: DType, src: Option<DType>, v: Scalar) -> PyResult<()> {
    let Some((lo, hi)) = int_bounds(target) else {
        return Ok(());
    };
    let signed = target.category() == rnp_core::dtype::Kind::Int;
    let n: i128 = match v {
        // `np.array([True], dtype=np.int8)` is 1, never an error.
        Scalar::Bool(_) => return Ok(()),
        Scalar::Int(i) => i as i128,
        Scalar::Uint(u) => u as i128,
        // numpy converts through `int(obj)`, which truncates toward zero.
        Scalar::Float(f) => {
            if !f.is_finite() {
                if src.is_some() && !signed {
                    return Ok(());
                }
                return Err(if f.is_nan() {
                    PyValueError::new_err("cannot convert float NaN to integer")
                } else {
                    pyo3::exceptions::PyOverflowError::new_err(
                        "cannot convert float infinity to integer",
                    )
                });
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
        Scalar::Float80(f) => {
            let f = f.to_f64();
            if !f.is_finite() {
                if src.is_some() && !signed { return Ok(()); }
                return Err(if f.is_nan() {
                    PyValueError::new_err("cannot convert float NaN to integer")
                } else {
                    pyo3::exceptions::PyOverflowError::new_err("cannot convert float infinity to integer")
                });
            }
            f.trunc() as i128
        }
        Scalar::Complex(_) | Scalar::Complex160(_) => return Ok(()),
    };
    // An *unsigned* target does not check a numpy scalar at all: probed,
    // `np.array([np.int64(-1)], dtype=np.uint8)` is 255 and
    // `np.array([np.uint64(2**63)], dtype=np.uint8)` is 0, while the same
    // values written as Python ints both raise.
    if src.is_some() && !signed {
        return Ok(());
    }
    // Every dtype first converts the Python object through one C integer
    // type, and a value that does not fit *that* never reaches the range
    // check. numpy picks the C type per dtype (`arraytypes.c.src`): only
    // `uint`/`ulong` use an unsigned conversion, so `uint8` and `uint16`
    // reject 2**63 with "too large" while `uint32` calls it out of bounds.
    // Probed for all eight integer dtypes x {2**63, 2**64-1, 2**64,
    // -2**63, -2**63-1}.
    let unsigned_conversion = matches!(target, DType::U32 | DType::U64);
    let hi_c = if unsigned_conversion {
        u64::MAX as i128
    } else {
        i64::MAX as i128
    };
    if !(n >= i64::MIN as i128 && n <= hi_c) {
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
///
/// A value that is neither `str` nor `bytes` is rendered with `str()`, which
/// is what numpy does: probed on 2.5.2, `np.zeros(2, 'S3')[...] = 0` gives
/// `b'0'`, `= 1.5` gives `b'1.5'`, `= True` gives `b'Tru'` (truncated) and
/// `= None` gives `b'Non'`.
fn text_bytes(obj: &Bound<'_, PyAny>, dt: DType) -> PyResult<(Vec<u8>, usize)> {
    match dt {
        DType::Bytes(_) => {
            let b: Vec<u8> = if let Ok(s) = obj.cast::<PyBytes>() {
                s.as_bytes().to_vec()
            } else if let Ok(s) = obj.cast::<PyString>() {
                // numpy encodes str into an S array as ASCII/latin-1 bytes.
                s.to_str()?.as_bytes().to_vec()
            } else {
                obj.str()?.to_str()?.as_bytes().to_vec()
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
                obj.str()?.to_str()?.to_string()
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
        // Legacy fixed-width strings resolve an all-empty input to one code
        // unit, not the internal S0/U0 discovery sentinel.
        DType::Bytes(0) => DType::Bytes(width.max(1) as u32),
        DType::Str(0) => DType::Str(width.max(1) as u32),
        other => other,
    };
    let out = NdArray::zeros(shape, dt).map_err(crate::err)?;
    let itemsize = out.itemsize() as isize;
    for (i, bytes) in encoded.iter().enumerate() {
        out.write_raw_at(out.byte_offset + i as isize * itemsize, bytes);
    }
    Ok(out)
}

fn string_na_matches(item: &Bound<'_, PyAny>, na: &Bound<'_, PyAny>) -> bool {
    if item.is(na) {
        return true;
    }
    if item.extract::<f64>().is_ok_and(f64::is_nan)
        && na.extract::<f64>().is_ok_and(f64::is_nan)
    {
        return true;
    }
    item.eq(na).unwrap_or(false)
}

/// Build a variable-width StringDType array. The 16-byte core cell carries a
/// slab handle; the pointed-to object is always a builtin `str`, except for a
/// configured NA sentinel whose exact Python identity is retained.
fn array_from_string(obj: &Bound<'_, PyAny>, dtype: DType) -> PyResult<NdArray> {
    let py = obj.py();
    let (coerce, na_object) = crate::pydtype::string_config(py, dtype)
        .ok_or_else(|| PyTypeError::new_err("invalid StringDType descriptor"))?;
    let mut shape = Vec::new();
    discover_shape(obj, &mut shape)?;
    let mut items: Vec<Bound<'_, PyAny>> = Vec::new();
    collect_objects(obj, 0, &shape, &mut items)?;
    let out = NdArray::zeros(shape, dtype).map_err(crate::err)?;
    let isz = dtype.itemsize() as isize;
    for (i, item) in items.iter().enumerate() {
        if let Some(na) = &na_object {
            if string_na_matches(item, na.bind(py)) {
                crate::objects::write_string(&out, i as isize * isz, na.bind(py));
                continue;
            }
        }
        let value = if let Ok(s) = item.cast::<PyString>() {
            PyString::new(py, s.to_str()?).into_any()
        } else if !coerce {
            return Err(PyValueError::new_err(
                "StringDType only allows string data when string coercion is disabled.",
            ));
        } else if let Ok(b) = item.cast::<PyBytes>() {
            b.call_method0("decode")?
        } else {
            item.str()?.into_any()
        };
        crate::objects::write_string(&out, i as isize * isz, &value);
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
    if let Some(d) = dtype.filter(|d| d.is_string()) {
        return array_from_string(obj, d);
    }
    // datetime64 / timedelta64 elements can be strings, `datetime` objects
    // or plain ints, and the unit is metadata, so they need their own path.
    if let Some(d) = dtype.filter(|d| d.is_datetime_like()) {
        return array_from_datetime(obj, d);
    }
    // Flexible dtypes take a completely separate path: their elements are
    // rendered Python values. This must precede numpy-scalar extraction:
    // datetime64(123, "s") -> "1970-01-01T00:02:03", not byte 123 (`'{'`).
    let wants_text = matches!(dtype, Some(DType::Bytes(_)) | Some(DType::Str(_)));
    if wants_text || (dtype.is_none() && first_leaf_is_text(obj)) {
        return array_from_text(obj, dtype);
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
    // A Python int too wide for any integer dtype: object, or an error.
    if let Some(h) = huge_int(obj)? {
        match dtype {
            None => return crate::objects::array_from_objects(obj),
            Some(d) if int_bounds(d).is_some() => return Err(too_large()),
            Some(d) => {
                let v = if d.is_float() || d.is_complex() {
                    Scalar::Float(h.float()?)
                } else {
                    h.placeholder()
                };
                let mut a = NdArray::zeros(vec![], d).map_err(crate::err)?;
                a.set(&[], v).map_err(crate::err)?;
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
        if values.iter().any(|l| l.huge.is_some()) {
            if int_bounds(d).is_some() {
                return Err(too_large());
            }
            if d.is_float() || d.is_complex() {
                for l in &values {
                    if let Some(h) = l.huge {
                        h.float()?;
                    }
                }
            }
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
///
/// numpy's discovery rule for a structured dtype, probed on 2.5.2: a **tuple
/// is always one element** of the array (whatever its length -- a wrong length
/// is a `ValueError`, not a new dimension), a **list is always a dimension**,
/// and anything else is a single element too. So
/// `np.array([(1, 2), (3, 4)], 'i4,f8')` has shape `(2,)`, while
/// `np.array([[1, 2], [3, 4]], 'i4,f8')` has shape `(2, 2)` with each *scalar*
/// filling both fields of its record.
fn array_from_records(obj: &Bound<'_, PyAny>, id: u32) -> PyResult<NdArray> {
    // Descend *lists* (and other non-tuple sequences) only; a tuple stops the
    // descent because it is an element.
    let mut shape: Vec<isize> = Vec::new();
    let mut cur = obj.clone();
    while is_sequence(&cur) && cur.cast::<PyTuple>().is_err() {
        if shape.len() >= 64 {
            return Err(PyValueError::new_err(
                "setting an array element with a sequence. The requested \
                 array would exceed the maximum number of dimension of 64.",
            ));
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
    let descr = out.descr;
    for (i, rec) in records.iter().enumerate() {
        write_element(&out, i as isize * isz, descr, rec)?;
    }
    Ok(out)
}

/// Store one Python object into the element of `descr` at absolute byte
/// offset `base` inside `out`'s buffer.
///
/// This is numpy's `VOID_setitem` (see
/// `upstream/numpy/_core/src/multiarray/arraytypes.c.src`), probed on 2.5.2:
///
/// * a **tuple** whose length matches the field count is distributed across
///   the fields, one component each, recursively;
/// * a tuple of any *other* length is
///   `ValueError: could not assign tuple of length N to structure with M
///   fields.`;
/// * **anything else** is assigned to *every* field, so
///   `np.zeros(2, 'i4,f8,u1')[...] = 5` is `(5, 5.0, 5)` and a nested
///   structured field gets 5 in each of *its* leaves in turn.
///
/// Errors surface from the first field that cannot take the value, which is
/// how `np.zeros(2, 'i4,f8')[...] = b'ab'` ends up raising `int`'s own
/// `invalid literal for int() with base 10: b'ab'`.
fn write_element(
    out: &NdArray,
    base: isize,
    descr: Descr,
    val: &Bound<'_, PyAny>,
) -> PyResult<()> {
    // A field value may be a *foreign* array or scalar -- numpy's own
    // `tolist()` renders a subarray field as a real `np.ndarray`, and feeding
    // those tuples straight back is exactly what `np.array(rows, dtype=...)`
    // supports. Render them as plain Python objects first; the target dtype is
    // already pinned by the field, so nothing is inferred from them.
    if let Some(plain) = as_plain_python(val)? {
        return write_element(out, base, descr, &plain);
    }
    let Some(def) = descr.struct_def() else {
        // A leaf (or a subarray field): hand the bytes to the ordinary
        // assignment machinery through a view, so broadcasting, casting and
        // every error message match `view[...] = val` exactly.
        let elem = element_view(out, base, descr);
        let py = val.py();
        let holder = Py::new(py, PyNdArray::wrap(elem))?.into_bound(py);
        return holder.set_item(py.Ellipsis(), val);
    };
    if let Ok(t) = val.cast::<PyTuple>() {
        if t.len() != def.fields.len() {
            return Err(PyValueError::new_err(format!(
                "could not assign tuple of length {} to structure with {} fields.",
                t.len(),
                def.fields.len()
            )));
        }
        for (f, v) in def.fields.iter().zip(t.iter()) {
            write_element(out, base + f.offset as isize, f.descr, &v)?;
        }
        return Ok(());
    }
    for f in def.fields.iter() {
        write_element(out, base + f.offset as isize, f.descr, val)?;
    }
    Ok(())
}

/// `Some(plain)` when `val` is an array-like from *another* library (a real
/// `numpy.ndarray`, a numpy scalar) that has to be rendered as plain Python
/// objects before it can be stored. `None` means "already plain".
fn as_plain_python<'py>(val: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if val.cast::<PyNdArray>().is_ok()
        || is_sequence(val)
        || is_text(val)
        || val.is_none()
        || scalar_from_py(val).is_some()
    {
        return Ok(None);
    }
    if val.hasattr(pyo3::intern!(val.py(), "tolist"))? {
        return Ok(Some(val.call_method0(pyo3::intern!(val.py(), "tolist"))?));
    }
    Ok(None)
}

/// A writeable view of the single element of `descr` at `base`: 0-d for a
/// leaf, and the subarray's own shape for a subarray field.
pub fn element_view(out: &NdArray, base: isize, descr: Descr) -> NdArray {
    let zero = NdArray {
        buffer: out.buffer.clone(),
        byte_offset: base,
        shape: Vec::new(),
        strides: Vec::new(),
        descr,
        flags: rnp_core::array::Flags {
            owndata: false,
            writeable: true,
            ..out.flags
        },
    };
    // `field_view` splices a subarray field's shape onto the (empty) shape and
    // replaces the descriptor with the subarray's base.
    zero.field_view(descr, 0)
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
    if arr.is_datetime_like() {
        // A weak Python number keeps its own dtype against a datetime-like
        // array: the datetime type resolvers are what decide whether it is
        // read as a count in the array's unit (`m8 + 3`, `m8 < 3`) or as a
        // plain multiplier/divisor (`m8 * 3`, `m8 / 3` -> m8, not float64).
        return s.natural_dtype();
    }
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
        Scalar::Float80(_) => (2, ()),
        Scalar::Complex(_) | Scalar::Complex160(_) => (3, ()),
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
    // A comparison against an *integer* array always answers, in whatever
    // dtype holds both sides -- probed over every integer dtype crossed with
    // every out-of-range value. A `bool` array is the exception: numpy has no
    // loop for it and lets the conversion error out (`np.ones(3, bool) <
    // 2**63` is an OverflowError, while `np.ones(3, np.uint8) < 2**63` is
    // not), which `check_int_store` below reproduces.
    if comparison && arr != DType::Bool {
        return Ok(promote(d, s.natural_dtype()));
    }
    check_int_store(d, None, s)?;
    Ok(d)
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
    if let Some(h) = huge_int(obj)? {
        if self_dtype.is_float() || self_dtype.is_complex() {
            // Probed: `np.arange(3.0) + 2**100` is float64 `1.2676506e+30`,
            // and `np.arange(3.0) + 2**2000` is an OverflowError even for a
            // comparison.
            let mut a = NdArray::zeros(vec![], self_dtype).map_err(crate::err)?;
            a.set(&[], Scalar::Float(h.float()?)).map_err(crate::err)?;
            return Ok(Some(a));
        }
        if !(self_dtype.is_integer() || self_dtype == DType::Bool) {
            return Ok(None);
        }
        // A `bool` array has no large-integer loop at all, comparison or not.
        if !comparison || self_dtype == DType::Bool {
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
        let v = Scalar::Float(if h.negative { -BEYOND } else { BEYOND });
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

// ---------------------------------------------------------------------------
// datetime64 / timedelta64 coercion
// ---------------------------------------------------------------------------

use rnp_core::datetime as dtm;

/// numpy's warning when an ISO string carries a timezone. It is a plain
/// `UserWarning`, emitted once per parsed string.
fn warn_timezone(py: Python<'_>) -> PyResult<()> {
    pyo3::PyErr::warn(
        py,
        &py.get_type::<pyo3::exceptions::PyUserWarning>(),
        std::ffi::CString::new(
            "no explicit representation of timezones available for np.datetime64",
        )
        .unwrap()
        .as_c_str(),
        1,
    )
}

/// One datetime-like element, plus the unit the source suggests (numpy's
/// `out_bestunit`), which unit discovery for a generic `M8`/`m8` uses.
fn datetime_element(
    obj: &Bound<'_, PyAny>,
    meta: dtm::DtMeta,
    is_td: bool,
) -> PyResult<(i64, Option<u8>)> {
    let py = obj.py();
    if obj.is_none() {
        return Ok((dtm::NAT, None));
    }
    // One of our own datetime scalars, or a 0-d datetime array.
    if let Some((d, s)) = np_scalar_descr(obj)? {
        if d.dt.is_datetime_like() {
            let raw = match s {
                Scalar::Int(i) => i,
                other => other.as_f64() as i64,
            };
            let src = dtm::meta_of(d.dt).unwrap();
            let v = if src.is_generic() || meta.is_generic() {
                raw
            } else {
                dtm::cast_value(d.dt, rnp_core::datetime::with_meta(d.dt, meta), raw)
                    .map_err(crate::err)?
            };
            return Ok((v, Some(src.base)));
        }
    }
    if let Ok(s) = obj.cast::<PyString>() {
        let text = s.to_str()?;
        return parse_datetime_text(py, text, meta, is_td);
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        let text = String::from_utf8_lossy(b.as_bytes()).into_owned();
        return parse_datetime_text(py, &text, meta, is_td);
    }
    // datetime.timedelta / datetime.date / datetime.datetime, recognised by
    // attribute so that duck types work as they do in numpy.
    if is_td && obj.hasattr(pyo3::intern!(py, "total_seconds"))? {
        let days: i64 = obj.getattr(pyo3::intern!(py, "days"))?.extract()?;
        let secs: i64 = obj.getattr(pyo3::intern!(py, "seconds"))?.extract()?;
        let us: i64 = obj.getattr(pyo3::intern!(py, "microseconds"))?.extract()?;
        let total_us = ((days * 86400 + secs) as i128) * 1_000_000 + us as i128;
        let src = dtm::DtMeta::unit(dtm::UNIT_US);
        let raw = total_us as i64;
        let v = if meta.is_generic() {
            raw
        } else {
            dtm::cast_timedelta(src, meta, raw).map_err(crate::err)?
        };
        return Ok((v, Some(dtm::UNIT_US)));
    }
    if !is_td && obj.hasattr(pyo3::intern!(py, "year"))? {
        let mut dts = dtm::Dts::epoch();
        dts.year = obj.getattr(pyo3::intern!(py, "year"))?.extract()?;
        dts.month = obj.getattr(pyo3::intern!(py, "month"))?.extract()?;
        dts.day = obj.getattr(pyo3::intern!(py, "day"))?.extract()?;
        let mut best = dtm::UNIT_D;
        if obj.hasattr(pyo3::intern!(py, "hour"))? {
            dts.hour = obj.getattr(pyo3::intern!(py, "hour"))?.extract()?;
            dts.min = obj.getattr(pyo3::intern!(py, "minute"))?.extract()?;
            dts.sec = obj.getattr(pyo3::intern!(py, "second"))?.extract()?;
            dts.us = obj.getattr(pyo3::intern!(py, "microsecond"))?.extract()?;
            best = dtm::UNIT_US;
        }
        if meta.is_generic() {
            // Unit discovery pass: the caller re-runs with the resolved unit.
            return Ok((dtm::NAT, Some(best)));
        }
        let v = dtm::dts_to_dt64(meta, &dts).map_err(crate::err)?;
        return Ok((v, Some(best)));
    }
    // A plain integer (or bool) is the raw count in the target's own unit.
    if let Some(s) = scalar_from_py(obj) {
        return Ok((
            match s {
                Scalar::Int(i) => i,
                Scalar::Uint(u) => u as i64,
                Scalar::Bool(b) => b as i64,
                Scalar::Float(f) => {
                    if f.is_finite() {
                        f as i64
                    } else {
                        dtm::NAT
                    }
                }
                Scalar::Float80(f) => if f.is_finite() { f.to_f64() as i64 } else { dtm::NAT },
                Scalar::Complex(c) => c.re as i64,
                Scalar::Complex160(c) => c.re.to_f64() as i64,
            },
            None,
        ));
    }
    Err(PyValueError::new_err(if is_td {
        "Could not convert object to NumPy timedelta"
    } else {
        "Could not convert object to NumPy datetime"
    }))
}

fn parse_datetime_text(
    py: Python<'_>,
    text: &str,
    meta: dtm::DtMeta,
    is_td: bool,
) -> PyResult<(i64, Option<u8>)> {
    if is_td {
        if text.is_empty() || text.eq_ignore_ascii_case("nat") {
            return Ok((dtm::NAT, None));
        }
        let v: i64 = text.trim().parse().map_err(|_| {
            PyValueError::new_err("Could not convert object to NumPy timedelta")
        })?;
        return Ok((v, None));
    }
    let (dts, best, special, tz) = special_or_parse(py, text)?;
    if tz {
        warn_timezone(py)?;
    }
    if dts.is_nat() {
        return Ok((dtm::NAT, Some(dtm::UNIT_GENERIC)));
    }
    if meta.is_generic() {
        // Unit discovery pass: the caller re-runs with the resolved unit.
        return Ok((dtm::NAT, Some(best)));
    }
    let _ = special;
    let v = dtm::dts_to_dt64(meta, &dts).map_err(crate::err)?;
    Ok((v, Some(best)))
}

/// `parse_iso8601` plus the two clock-reading special strings numpy accepts.
pub fn special_or_parse(
    py: Python<'_>,
    text: &str,
) -> PyResult<(dtm::Dts, u8, bool, bool)> {
    let t = text.trim();
    if t.eq_ignore_ascii_case("today") || t.eq_ignore_ascii_case("now") {
        let time = py.import("time")?;
        let secs: f64 = time.call_method0("time")?.extract()?;
        if t.eq_ignore_ascii_case("today") {
            // numpy takes *local* midnight for 'today'.
            let lt = time.call_method1("localtime", (secs,))?;
            let mut dts = dtm::Dts::epoch();
            dts.year = lt.getattr("tm_year")?.extract()?;
            dts.month = lt.getattr("tm_mon")?.extract()?;
            dts.day = lt.getattr("tm_mday")?.extract()?;
            return Ok((dts, dtm::UNIT_D, true, false));
        }
        let dts = dtm::dt64_to_dts(dtm::DtMeta::unit(dtm::UNIT_S), secs as i64)
            .map_err(crate::err)?;
        return Ok((dts, dtm::UNIT_S, true, false));
    }
    let p = dtm::parse_iso8601(text).map_err(crate::err)?;
    Ok((p.dts, p.bestunit, p.special, p.had_timezone))
}

/// Build a datetime64 / timedelta64 array from nested Python sequences.
fn array_from_datetime(obj: &Bound<'_, PyAny>, dt: DType) -> PyResult<NdArray> {
    let mut shape = Vec::new();
    discover_shape(obj, &mut shape)?;
    let mut items: Vec<Bound<'_, PyAny>> = Vec::new();
    if shape.is_empty() {
        items.push(obj.clone());
    } else {
        collect_objects(obj, 0, &shape, &mut items)?;
    }
    let is_td = dt.is_timedelta();
    let mut meta = dtm::meta_of(dt).unwrap();
    if meta.is_generic() {
        // numpy's unit discovery: the finest unit any element needs.
        let mut best: Option<u8> = None;
        for it in &items {
            if let (_, Some(b)) = datetime_element(it, meta, is_td)? {
                if b != dtm::UNIT_GENERIC {
                    best = Some(best.map_or(b, |x| x.max(b)));
                }
            }
        }
        if let Some(b) = best {
            meta = dtm::DtMeta::unit(b);
        }
    }
    let out_dt = rnp_core::datetime::with_meta(dt, meta);
    let mut vals = Vec::with_capacity(items.len());
    for it in &items {
        vals.push(Scalar::Int(datetime_element(it, meta, is_td)?.0));
    }
    let flat = NdArray::from_scalars(&vals, out_dt).map_err(crate::err)?;
    flat.reshape(&shape).map_err(crate::err)
}
