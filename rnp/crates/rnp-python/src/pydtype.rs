//! The `dtype` pyclass, backed by `rnp_core::Descr`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyAttributeError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::sync::{OnceLock, RwLock};
use pyo3::types::{
    PyBool, PyComplex, PyDict, PyFloat, PyInt, PyList, PyMappingProxy, PyString, PyTuple, PyType,
};
use pyo3::PyTypeInfo;

use rnp_core::descr::{make_struct, make_subarray, FieldSpec};
use rnp_core::{DType, Descr};

/// The `name -> scalar class` map the shim installs, so that `dtype.type`
/// can hand back `np.float64` and friends without rnp-core knowing about
/// Python classes.
static SCALAR_TYPES: OnceLock<Py<PyDict>> = OnceLock::new();

/// Python owns dtype metadata values.  Core descriptors carry only a compact
/// identity, keeping `Descr: Copy` while arrays and views propagate the
/// decoration with their descriptor.  Entries intentionally live for the
/// process lifetime, just like the compound-dtype interners in rnp-core.
static METADATA: OnceLock<RwLock<Vec<Py<PyDict>>>> = OnceLock::new();

/// Python-owned parameters for NEP 55 StringDType descriptors. Core keeps the
/// compact id in `DType::String`, just as it does for interned structured
/// descriptors, while the NA sentinel retains its Python identity here.
struct StringParams {
    coerce: bool,
    na_object: Option<Py<PyAny>>,
}

static STRING_PARAMS: OnceLock<RwLock<Vec<StringParams>>> = OnceLock::new();

fn string_params_store() -> &'static RwLock<Vec<StringParams>> {
    STRING_PARAMS.get_or_init(|| RwLock::new(Vec::new()))
}

fn string_params(py: Python<'_>, id: u32) -> Option<(bool, Option<Py<PyAny>>)> {
    if id == 0 {
        return Some((true, None));
    }
    string_params_store()
        .read()
        .ok()?
        .get(id as usize - 1)
        .map(|p| (p.coerce, p.na_object.as_ref().map(|o| o.clone_ref(py))))
}

pub(crate) fn string_config(
    py: Python<'_>,
    dt: DType,
) -> Option<(bool, Option<Py<PyAny>>)> {
    match dt {
        DType::String(id) => string_params(py, id),
        _ => None,
    }
}

pub fn new_string_dtype(
    py: Python<'_>,
    coerce: bool,
    na_object: Option<&Bound<'_, PyAny>>,
) -> PyDType {
    if coerce && na_object.is_none() {
        return PyDType::new(DType::String(0));
    }
    let mut store = string_params_store().write().unwrap();
    store.push(StringParams {
        coerce,
        na_object: na_object.map(|o| o.clone().unbind()),
    });
    let _ = py;
    PyDType::new(DType::String(store.len() as u32))
}

fn string_params_equal(py: Python<'_>, a: u32, b: u32) -> PyResult<bool> {
    let Some((ac, an)) = string_params(py, a) else { return Ok(false) };
    let Some((bc, bn)) = string_params(py, b) else { return Ok(false) };
    if ac != bc {
        return Ok(false);
    }
    string_na_equal(py, an, bn)
}

fn string_na_equal(
    py: Python<'_>,
    an: Option<Py<PyAny>>,
    bn: Option<Py<PyAny>>,
) -> PyResult<bool> {
    match (an, bn) {
        (None, None) => Ok(true),
        (Some(a), Some(b)) => {
            let (a, b) = (a.bind(py), b.bind(py));
            if a.is(b) {
                return Ok(true);
            }
            // All floating NaNs are one logical NA category in NumPy's
            // StringDType equality, even though their ordinary equality and
            // hashes remain object-specific.
            if a.extract::<f64>().is_ok_and(f64::is_nan)
                && b.extract::<f64>().is_ok_and(f64::is_nan)
            {
                return Ok(true);
            }
            match a.eq(b) {
                Ok(eq) => Ok(eq),
                Err(_) => Ok(false),
            }
        }
        _ => Ok(false),
    }
}

fn string_repr(py: Python<'_>, id: u32) -> PyResult<String> {
    let (coerce, na) = string_params(py, id)
        .ok_or_else(|| PyTypeError::new_err("invalid StringDType descriptor"))?;
    let mut args = Vec::new();
    if let Some(na) = na {
        args.push(format!("na_object={}", na.bind(py).repr()?.to_str()?));
    }
    if !coerce {
        args.push("coerce=False".to_string());
    }
    Ok(format!("StringDType({})", args.join(", ")))
}

pub(crate) fn string_repr_for(py: Python<'_>, dt: DType) -> PyResult<String> {
    match dt {
        DType::String(id) => string_repr(py, id),
        _ => Err(PyTypeError::new_err("not a StringDType descriptor")),
    }
}

pub(crate) fn promote_string_descr(
    py: Python<'_>,
    a: Descr,
    b: Descr,
) -> PyResult<Descr> {
    match (a.dt, b.dt) {
        (DType::String(ai), DType::String(bi)) => {
            let (ac, an) = string_params(py, ai)
                .ok_or_else(|| PyTypeError::new_err("invalid StringDType descriptor"))?;
            let (bc, bn) = string_params(py, bi)
                .ok_or_else(|| PyTypeError::new_err("invalid StringDType descriptor"))?;
            let na = match (an, bn) {
                (None, None) => None,
                (Some(na), None) | (None, Some(na)) => Some(na),
                (Some(an), Some(bn)) => {
                    if !string_na_equal(
                        py,
                        Some(an.clone_ref(py)),
                        Some(bn.clone_ref(py)),
                    )? {
                        return Err(PyTypeError::new_err(
                            "StringDType instances with distinct na_object values cannot be promoted",
                        ));
                    }
                    let _ = bn;
                    Some(an)
                }
            };
            Ok(new_string_dtype(py, ac && bc, na.as_ref().map(|o| o.bind(py))).d)
        }
        (DType::String(_), DType::Str(_)) => Ok(a),
        (DType::Str(_), DType::String(_)) => Ok(b),
        (DType::String(_), DType::Object) | (DType::Object, DType::String(_)) => {
            Ok(Descr::native(DType::Object))
        }
        _ => Err(PyTypeError::new_err(
            "The StringDType could not be promoted by the other DType",
        )),
    }
}

fn metadata_store() -> &'static RwLock<Vec<Py<PyDict>>> {
    METADATA.get_or_init(|| RwLock::new(Vec::new()))
}

fn metadata_dict(py: Python<'_>, id: u32) -> Option<Py<PyDict>> {
    if id == 0 {
        return None;
    }
    metadata_store()
        .read()
        .ok()?
        .get(id as usize - 1)
        .map(|d| d.clone_ref(py))
}

fn intern_metadata(py: Python<'_>, source: &Bound<'_, PyDict>) -> PyResult<u32> {
    let copied = PyDict::new(py);
    copied.update(source.as_mapping())?;
    let mut store = metadata_store()
        .write()
        .map_err(|_| PyTypeError::new_err("dtype metadata registry is poisoned"))?;
    store.push(copied.unbind());
    Ok(store.len() as u32)
}

/// Compare compound storage recursively while ignoring metadata decoration.
/// This cannot live in `Descr::Hash`: compound interning hashes definitions
/// while holding its write lock, so recursively re-entering that registry
/// would deadlock.
pub(crate) fn storage_eq(a: Descr, b: Descr) -> bool {
    match (a.struct_def(), b.struct_def()) {
        (Some(a), Some(b)) => {
            a.itemsize == b.itemsize
                && a.alignment == b.alignment
                && a.aligned == b.aligned
                && a.fields.len() == b.fields.len()
                && a.fields.iter().zip(&b.fields).all(|(a, b)| {
                    a.name == b.name
                        && storage_eq(a.descr, b.descr)
                        && a.offset == b.offset
                        && a.title == b.title
                })
        }
        (Some(_), None) | (None, Some(_)) => false,
        (None, None) => match (a.subarray_def(), b.subarray_def()) {
            (Some(a), Some(b)) => a.shape == b.shape && storage_eq(a.base, b.base),
            (Some(_), None) | (None, Some(_)) => false,
            (None, None) => a == b,
        },
    }
}

pub fn register_scalar_types(d: Bound<'_, PyDict>) {
    let _ = SCALAR_TYPES.set(d.unbind());
}

/// The shim's per-dtype `_wrap` callables, in `SCALAR_DTYPES` order.
///
/// `scalar_class` builds a `String` key and does a dict lookup, then the
/// caller has to look `_wrap` up on the class -- three dictionary probes and a
/// heap allocation before anything happens. Every `a[i]` that returns a numpy
/// scalar paid them. This table replaces all of it with one index.
static SCALAR_WRAPS: OnceLock<Vec<Py<PyAny>>> = OnceLock::new();

pub fn register_scalar_wraps(v: Vec<Py<PyAny>>) {
    let _ = SCALAR_WRAPS.set(v);
}

/// The shim's `datetime64`/`timedelta64` builder, `f(raw_int, dtype)`.
///
/// The datetime scalars cannot go through `SCALAR_WRAPS`: that table is
/// indexed by numeric dtype code, and a datetime scalar needs its *unit* as
/// well as its value.
static DATETIME_FACTORY: OnceLock<Py<PyAny>> = OnceLock::new();

pub fn register_datetime_factory(f: Py<PyAny>) {
    let _ = DATETIME_FACTORY.set(f);
}

/// Build the shim's datetime scalar for `(dt, value)`, if it registered one.
pub fn datetime_scalar<'py>(
    py: Python<'py>,
    dt: DType,
    v: i64,
) -> Option<PyResult<Bound<'py, PyAny>>> {
    DATETIME_FACTORY
        .get()
        .map(|f| f.bind(py).call1((v, PyDType::new(dt))))
}

/// The `_wrap` for a storage dtype, if the shim registered one.
#[inline]
pub fn scalar_wrap<'py>(py: Python<'py>, dt: DType) -> Option<&'py Bound<'py, PyAny>> {
    let code = crate::ufuncs::dtype_code(dt)?;
    SCALAR_WRAPS.get().map(|v| v[code].bind(py))
}

/// The Python scalar class the shim registered for a storage dtype, if any.
pub fn scalar_class<'py>(py: Python<'py>, dt: DType) -> Option<Bound<'py, PyAny>> {
    let key: String = match dt {
        DType::Bytes(_) => "bytes_".into(),
        DType::Str(_) => "str_".into(),
        DType::Void(_) | DType::Struct(_) | DType::SubArray(_) => "void".into(),
        DType::Object => "object_".into(),
        DType::DateTime(_) => "datetime64".into(),
        DType::TimeDelta(_) => "timedelta64".into(),
        d => d.name(),
    };
    SCALAR_TYPES
        .get()
        .and_then(|m| m.bind(py).get_item(key).ok().flatten())
}

#[pyclass(name = "dtype", module = "_rnp", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyDType {
    pub d: Descr,
}

impl PyDType {
    pub fn new(dt: DType) -> Self {
        PyDType {
            d: Descr::native(dt),
        }
    }

    pub fn from_descr(d: Descr) -> Self {
        PyDType { d }
    }

    pub fn dt(&self) -> DType {
        self.d.dt
    }
}

/// Resolve any of numpy's dtype spellings to a full `Descr`.
pub fn descr_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Descr> {
    descr_from_any_aligned(obj, false)
}

/// As `descr_from_any`, with numpy's `align=` flag for the struct forms.
pub fn descr_from_any_aligned(obj: &Bound<'_, PyAny>, align: bool) -> PyResult<Descr> {
    if obj.is_none() {
        return Ok(Descr::native(DType::F64));
    }
    if let Ok(d) = obj.extract::<PyDType>() {
        return Ok(d.d);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        let name = s.to_str()?;
        return Descr::parse(name)
            .ok_or_else(|| PyTypeError::new_err(format!("data type '{}' not understood", name)));
    }
    if let Ok(t) = obj.cast::<PyType>() {
        // Builtin Python types map to numpy's defaults. Order matters: bool
        // is checked before int because it is a subclass.
        if t.is(&PyBool::type_object(obj.py())) {
            return Ok(Descr::native(DType::Bool));
        }
        // `np.dtype(object)` — a descriptor only; arrays of it are rejected
        // at creation time.
        if t.is(&PyAny::type_object(obj.py())) {
            return Ok(Descr::native(DType::Object));
        }
        if t.is(&PyInt::type_object(obj.py())) {
            return Ok(Descr::native(DType::I64));
        }
        if t.is(&PyFloat::type_object(obj.py())) {
            return Ok(Descr::native(DType::F64));
        }
        if t.is(&PyComplex::type_object(obj.py())) {
            return Ok(Descr::native(DType::C128));
        }
        if t.is(&PyString::type_object(obj.py())) {
            return Ok(Descr::native(DType::Str(0)));
        }
        if t.is(&pyo3::types::PyBytes::type_object(obj.py())) {
            return Ok(Descr::native(DType::Bytes(0)));
        }
        // A scalar-alias class from the shim carries its dtype.
        if let Ok(attr) = t.getattr("dtype") {
            if let Ok(d) = attr.extract::<PyDType>() {
                return Ok(d.d);
            }
        }
    }
    if let Ok(l) = obj.cast::<PyList>() {
        return struct_from_list(l, align);
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        return descr_from_tuple(t, align);
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        return struct_from_dict(d, align);
    }
    // Anything exposing a `.dtype` (e.g. our own ndarray).
    if let Ok(attr) = obj.getattr("dtype") {
        if !attr.is(obj) {
            if let Ok(d) = attr.extract::<PyDType>() {
                return Ok(d.d);
            }
        }
    }
    Err(PyTypeError::new_err(format!(
        "Cannot interpret '{}' as a data type",
        obj.repr()?.to_str()?
    )))
}

/// Just the storage dtype.
///
/// Byte order is dropped here, which is right for every caller that is about
/// to *compute* (the engine computes in the host order and stores the result
/// natively, as numpy's own ufuncs do); callers that build or relabel storage
/// use [`descr_from_any`] instead.
pub fn dtype_from_any(obj: &Bound<'_, PyAny>) -> PyResult<DType> {
    Ok(descr_from_any(obj)?.dt)
}

/// A shape argument inside a dtype spec: an int or a tuple of ints.
fn shape_arg(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(i) = obj.extract::<isize>() {
        return Ok(vec![i]);
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        return t.iter().map(|x| x.extract::<isize>()).collect();
    }
    if let Ok(l) = obj.cast::<PyList>() {
        return l.iter().map(|x| x.extract::<isize>()).collect();
    }
    Err(PyTypeError::new_err("invalid shape in a dtype spec"))
}

/// `('f4', (2, 2))` -> subarray; `('f4', 3)` -> subarray of 3.
fn descr_from_tuple(t: &Bound<'_, PyTuple>, align: bool) -> PyResult<Descr> {
    if t.len() != 2 {
        return Err(PyTypeError::new_err(
            "a dtype tuple must be (base, shape) or (base, itemsize)",
        ));
    }
    let first = t.get_item(0)?;
    let second = t.get_item(1)?;
    // `(np.void, dtype)` is NumPy's legacy way to retain a descriptor's
    // structure/metadata while expressing opaque scalar storage.
    if first
        .getattr("__name__")
        .ok()
        .and_then(|n| n.extract::<String>().ok())
        .as_deref()
        == Some("void")
    {
        if let Ok(source) = second.extract::<PyDType>() {
            if source.d.is_struct() {
                return Ok(source.d);
            }
            return Ok(Descr::native(DType::Void(source.d.itemsize() as u32))
                .with_metadata(source.d.metadata));
        }
    }
    let base = descr_from_any_aligned(&first, align)?;
    let shape = shape_arg(&second)?;
    if shape.is_empty() {
        return Ok(base);
    }
    Ok(make_subarray(base, shape))
}

/// Parse one entry of the list form: `(name, format)`,
/// `(name, format, shape)` or `((title, name), format[, shape])`.
fn field_from_entry(entry: &Bound<'_, PyAny>, index: usize, align: bool) -> PyResult<FieldSpec> {
    let t = entry
        .cast::<PyTuple>()
        .map_err(|_| PyTypeError::new_err("a structured dtype entry must be a tuple"))?;
    if t.len() < 2 || t.len() > 3 {
        return Err(PyTypeError::new_err(
            "a structured dtype entry must have 2 or 3 items",
        ));
    }
    let first = t.get_item(0)?;
    let (title, name) = if let Ok(pair) = first.cast::<PyTuple>() {
        if pair.len() != 2 {
            return Err(PyTypeError::new_err("a (title, name) pair must have 2 items"));
        }
        (
            Some(pair.get_item(0)?.extract::<String>()?),
            pair.get_item(1)?.extract::<String>()?,
        )
    } else {
        (None, first.extract::<String>().unwrap_or_else(|_| format!("f{index}")))
    };
    let mut descr = descr_from_any_aligned(&t.get_item(1)?, align)?;
    if t.len() == 3 {
        let shape = shape_arg(&t.get_item(2)?)?;
        if !shape.is_empty() && !(shape.len() == 1 && shape[0] == 1) {
            descr = make_subarray(descr, shape);
        } else if shape.len() == 1 && shape[0] == 1 {
            // numpy collapses a trailing length-1 subarray to the base type.
            descr = make_subarray(descr, shape);
        }
    }
    Ok(FieldSpec {
        name,
        descr,
        title,
        offset: None,
    })
}

fn struct_from_list(l: &Bound<'_, PyList>, align: bool) -> PyResult<Descr> {
    let mut specs = Vec::with_capacity(l.len());
    for (i, entry) in l.iter().enumerate() {
        specs.push(field_from_entry(&entry, i, align)?);
    }
    make_struct(specs, None, align).map_err(crate::err)
}

fn struct_from_dict(d: &Bound<'_, PyDict>, align: bool) -> PyResult<Descr> {
    let get = |k: &str| -> PyResult<Option<Bound<'_, PyAny>>> { d.get_item(k) };
    let names = get("names")?;
    let formats = get("formats")?;
    if names.is_none() || formats.is_none() {
        return Err(PyTypeError::new_err(
            "a dtype dict must provide 'names' and 'formats'",
        ));
    }
    let names: Vec<String> = names.unwrap().extract()?;
    let formats = formats.unwrap();
    let formats: Vec<Bound<'_, PyAny>> = formats.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    if names.len() != formats.len() {
        return Err(PyValueError::new_err(
            "'names' and 'formats' must have the same length",
        ));
    }
    let offsets: Option<Vec<usize>> = match get("offsets")? {
        Some(o) if !o.is_none() => Some(o.extract()?),
        _ => None,
    };
    let titles: Option<Vec<Option<String>>> = match get("titles")? {
        Some(o) if !o.is_none() => Some(o.extract()?),
        _ => None,
    };
    let itemsize: Option<usize> = match get("itemsize")? {
        Some(o) if !o.is_none() => Some(o.extract()?),
        _ => None,
    };
    let aligned = match get("aligned")? {
        Some(o) if !o.is_none() => o.extract::<bool>()?,
        _ => align,
    };

    let mut specs = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        specs.push(FieldSpec {
            name: name.clone(),
            descr: descr_from_any_aligned(&formats[i], aligned)?,
            title: titles.as_ref().and_then(|t| t[i].clone()),
            offset: offsets.as_ref().map(|o| o[i]),
        });
    }
    make_struct(specs, itemsize, aligned).map_err(crate::err)
}

#[pymethods]
impl PyDType {
    #[new]
    #[pyo3(signature = (obj, align = false, copy = false, **kwargs))]
    fn py_new(
        obj: &Bound<'_, PyAny>,
        align: bool,
        copy: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let _ = copy;
        let mut d = descr_from_any_aligned(obj, align)?;
        if let Some(kwargs) = kwargs {
            for (key, _) in kwargs.iter() {
                let key: String = key.extract()?;
                if key != "metadata" {
                    return Err(PyTypeError::new_err(format!(
                        "dtype() got an unexpected keyword argument '{key}'"
                    )));
                }
            }
            if let Some(value) = kwargs.get_item("metadata")? {
                let supplied = value.cast::<PyDict>().map_err(|_| {
                    let type_name = value
                        .get_type()
                        .name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| "object".into());
                    PyTypeError::new_err(format!(
                        "dtype() argument 4 must be dict, not {type_name}"
                    ))
                })?;
                let merged = PyDict::new(obj.py());
                if let Some(old) = metadata_dict(obj.py(), d.metadata) {
                    merged.update(old.bind(obj.py()).as_mapping())?;
                }
                merged.update(supplied.as_mapping())?;
                d.metadata = intern_metadata(obj.py(), &merged)?;
            }
        }
        Ok(PyDType { d })
    }

    #[getter]
    fn name(&self) -> String {
        self.d.name()
    }

    #[getter]
    fn kind(&self) -> String {
        self.d.kind().to_string()
    }

    #[getter]
    fn char(&self) -> String {
        self.d.char_code().to_string()
    }

    #[getter]
    fn itemsize(&self) -> usize {
        self.d.itemsize()
    }

    #[getter]
    fn alignment(&self) -> usize {
        self.d.alignment()
    }

    #[getter]
    fn num(&self) -> i32 {
        self.d.num()
    }

    #[getter]
    fn byteorder(&self) -> String {
        self.d.bo.as_char().to_string()
    }

    #[getter]
    fn str(&self) -> String {
        match self.d.dt {
            DType::String(id) => Python::attach(|py| {
                string_repr(py, id).unwrap_or_else(|_| "StringDType()".into())
            }),
            _ => self.d.str_code(),
        }
    }

    #[getter]
    fn base(&self) -> PyDType {
        PyDType::from_descr(self.d.base())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.d.shape().len()
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.d.shape().iter().map(|&d| d as usize))
    }

    #[getter]
    fn isnative(&self) -> bool {
        self.d.isnative()
    }

    #[getter]
    fn isbuiltin(&self) -> i32 {
        if self.d.is_struct() || self.d.subarray_def().is_some() || self.d.dt.is_string() {
            0
        } else {
            1
        }
    }

    #[getter]
    fn isalignedstruct(&self) -> bool {
        self.d.isalignedstruct()
    }

    #[getter]
    fn hasobject(&self) -> bool {
        self.d.dt.is_object() || self.d.dt.is_string()
    }

    #[getter]
    fn coerce(&self, py: Python<'_>) -> PyResult<bool> {
        match self.d.dt {
            DType::String(id) => string_params(py, id)
                .map(|p| p.0)
                .ok_or_else(|| PyAttributeError::new_err("coerce")),
            _ => Err(PyAttributeError::new_err("coerce")),
        }
    }

    #[getter]
    fn na_object<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.d.dt {
            DType::String(id) => match string_params(py, id).and_then(|p| p.1) {
                Some(na) => Ok(na.into_bound(py)),
                None => Err(PyAttributeError::new_err("na_object")),
            },
            _ => Err(PyAttributeError::new_err("na_object")),
        }
    }

    /// `dtype.type`: the scalar class, looked up in the shim's registry.
    #[getter]
    fn r#type<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if self.d.dt.is_string() {
            return Ok(Some(PyString::type_object(py).into_any()));
        }
        // The C-type aliases keep their own scalar class: `np.dtype('q').type`
        // is `np.longlong`, not `np.int64`, even though the two dtypes compare
        // equal.
        let alias_key = match self.d.alias {
            rnp_core::descr::Alias::LongLong => Some("longlong"),
            rnp_core::descr::Alias::ULongLong => Some("ulonglong"),
            rnp_core::descr::Alias::LongDouble => Some("longdouble"),
            rnp_core::descr::Alias::CLongDouble => Some("clongdouble"),
            _ => None,
        };
        if let Some(k) = alias_key {
            if let Some(map) = SCALAR_TYPES.get() {
                if let Some(t) = map.bind(py).get_item(k)? {
                    return Ok(Some(t));
                }
            }
        }
        let key: String = match self.d.dt {
            DType::Bytes(_) => "bytes_".into(),
            DType::Str(_) => "str_".into(),
            DType::Void(_) | DType::Struct(_) | DType::SubArray(_) => "void".into(),
            // The datetime classes are one per *family*, not per unit.
            DType::DateTime(_) => "datetime64".into(),
            DType::TimeDelta(_) => "timedelta64".into(),
            DType::String(_) => "str_".into(),
            d => d.name(),
        };
        match SCALAR_TYPES.get() {
            Some(map) => Ok(map.bind(py).get_item(key)?),
            None => Ok(None),
        }
    }

    #[getter]
    fn flags(&self) -> i32 {
        // NPY_LIST_PICKLE etc.; nothing the port models yet.
        if self.d.dt.is_string() { 107 } else { 0 }
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(dict) = metadata_dict(py, self.d.metadata) else {
            return Ok(None);
        };
        Ok(Some(
            PyMappingProxy::new(py, dict.bind(py).as_mapping()).into_any(),
        ))
    }

    /// `dtype.fields`: `{name: (dtype, offset[, title])}`, or None.
    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let def = match self.d.struct_def() {
            Some(d) => d,
            None => return Ok(None),
        };
        let out = PyDict::new(py);
        for f in &def.fields {
            let value: Bound<'py, PyTuple> = match &f.title {
                Some(t) => PyTuple::new(
                    py,
                    [
                        PyDType::from_descr(f.descr).into_pyobject(py)?.into_any(),
                        f.offset.into_pyobject(py)?.into_any(),
                        t.into_pyobject(py)?.into_any(),
                    ],
                )?,
                None => PyTuple::new(
                    py,
                    [
                        PyDType::from_descr(f.descr).into_pyobject(py)?.into_any(),
                        f.offset.into_pyobject(py)?.into_any(),
                    ],
                )?,
            };
            out.set_item(&f.name, &value)?;
            // numpy also keys the mapping by title.
            if let Some(t) = &f.title {
                out.set_item(t, &value)?;
            }
        }
        Ok(Some(out))
    }

    #[getter]
    fn names<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        match self.d.struct_def() {
            Some(def) => Ok(Some(PyTuple::new(
                py,
                def.fields.iter().map(|f| f.name.clone()),
            )?)),
            None => Ok(None),
        }
    }

    #[getter]
    fn subdtype<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        match self.d.subarray_def() {
            Some(sub) => {
                let base = PyDType::from_descr(sub.base).into_pyobject(py)?.into_any();
                let shape = PyTuple::new(py, sub.shape.iter().map(|&d| d as usize))?.into_any();
                Ok(Some(PyTuple::new(py, [base, shape])?))
            }
            None => Ok(None),
        }
    }

    /// `dtype.descr`: the `__array_interface__` description list.
    #[getter]
    pub fn descr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyAny>> = match self.d.struct_def() {
            Some(def) => {
                let mut out = Vec::new();
                for f in &def.fields {
                    out.push(
                        PyTuple::new(
                            py,
                            [
                                f.name.clone().into_pyobject(py)?.into_any(),
                                f.descr.str_code().into_pyobject(py)?.into_any(),
                            ],
                        )?
                        .into_any(),
                    );
                }
                out
            }
            None => vec![PyTuple::new(
                py,
                [
                    "".into_pyobject(py)?.into_any(),
                    self.d.str_code().into_pyobject(py)?.into_any(),
                ],
            )?
            .into_any()],
        };
        PyList::new(py, items)
    }

    #[pyo3(signature = (new_order = None))]
    fn newbyteorder(&self, new_order: Option<&str>) -> PyResult<PyDType> {
        let c = match new_order {
            None => None,
            Some(s) => s.chars().next(),
        };
        Ok(PyDType::from_descr(
            self.d.newbyteorder(c).map_err(crate::err)?,
        ))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        match self.d.dt {
            DType::String(id) => string_repr(py, id),
            _ => Ok(self.d.repr_string()),
        }
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        match self.d.dt {
            DType::String(id) => string_repr(py, id),
            _ => Ok(self.d.str_string()),
        }
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        if let DType::String(id) = self.d.dt {
            let (coerce, na) = string_params(py, id)
                .ok_or_else(|| PyTypeError::new_err("invalid StringDType descriptor"))?;
            let args = PyList::empty(py);
            args.append(coerce)?;
            if let Some(na) = na {
                args.append(na.bind(py))?;
            }
            return args.to_tuple().hash();
        }
        let mut h = DefaultHasher::new();
        if self.d.is_struct() || self.d.subarray_def().is_some() {
            // repr omits metadata but captures compound storage recursively.
            self.d.repr_string().hash(&mut h);
        } else {
            self.d.hash(&mut h);
        }
        Ok(h.finish() as isize)
    }

    fn __len__(&self) -> usize {
        self.d.struct_def().map(|d| d.fields.len()).unwrap_or(0)
    }

    /// `dtype['name']` and `dtype[i]` for structured dtypes.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyDType> {
        let def = self
            .d
            .struct_def()
            .ok_or_else(|| PyKeyError::new_err("there are no fields in this dtype"))?;
        if let Ok(name) = key.extract::<String>() {
            return def
                .fields
                .iter()
                .find(|f| f.name == name || f.title.as_deref() == Some(name.as_str()))
                .map(|f| PyDType::from_descr(f.descr))
                .ok_or_else(|| PyKeyError::new_err(name));
        }
        let i: isize = key.extract()?;
        let n = def.fields.len() as isize;
        let i = if i < 0 { i + n } else { i };
        if i < 0 || i >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Field index out of range.",
            ));
        }
        Ok(PyDType::from_descr(def.fields[i as usize].descr))
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let eq = match descr_from_any(other) {
            Ok(d) => match (self.d.dt, d.dt) {
                (DType::String(a), DType::String(b)) => string_params_equal(py, a, b)?,
                _ => storage_eq(d, self.d),
            },
            // numpy returns False rather than raising for junk comparisons.
            Err(_) => false,
        };
        match op {
            CompareOp::Eq => Ok(eq.into_pyobject(py)?.to_owned().unbind().into_any()),
            CompareOp::Ne => Ok((!eq).into_pyobject(py)?.to_owned().unbind().into_any()),
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        if let DType::String(id) = slf.borrow().d.dt {
            let (coerce, na) = string_params(py, id)
                .ok_or_else(|| PyTypeError::new_err("invalid StringDType descriptor"))?;
            let helper = py.import("numpy.dtypes")?.getattr("_reconstruct_string_dtype")?;
            let has_na = na.is_some();
            let args = PyTuple::new(
                py,
                [
                    coerce.into_pyobject(py)?.to_owned().into_any(),
                    has_na.into_pyobject(py)?.to_owned().into_any(),
                    na.unwrap_or_else(|| py.None()).into_bound(py),
                ],
            )?;
            return PyTuple::new(py, [helper, args.into_any()]);
        }
        let cls = slf.get_type().into_any();
        let args = PyTuple::new(py, [slf.borrow().d.str_code()])?.into_any();
        let metadata = metadata_dict(py, slf.borrow().d.metadata);
        match metadata {
            None => PyTuple::new(py, [cls, args]),
            Some(metadata) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("metadata", metadata.bind(py))?;
                let callable = py
                    .import("functools")?
                    .getattr("partial")?
                    .call((cls,), Some(&kwargs))?;
                PyTuple::new(py, [callable, args])
            }
        }
    }
}
