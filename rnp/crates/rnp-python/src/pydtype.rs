//! The `dtype` pyclass, backed by `rnp_core::Descr`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::sync::OnceLock;
use pyo3::types::{PyBool, PyComplex, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};
use pyo3::PyTypeInfo;

use rnp_core::descr::{make_struct, make_subarray, FieldSpec};
use rnp_core::{DType, Descr};

/// The `name -> scalar class` map the shim installs, so that `dtype.type`
/// can hand back `np.float64` and friends without rnp-core knowing about
/// Python classes.
static SCALAR_TYPES: OnceLock<Py<PyDict>> = OnceLock::new();

pub fn register_scalar_types(d: Bound<'_, PyDict>) {
    let _ = SCALAR_TYPES.set(d.unbind());
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

/// The storage dtype, rejecting the byte-swapped descriptors the engine
/// cannot compute on yet.
pub fn dtype_from_any(obj: &Bound<'_, PyAny>) -> PyResult<DType> {
    let d = descr_from_any(obj)?;
    require_native(d)?;
    Ok(d.dt)
}

pub fn require_native(d: Descr) -> PyResult<()> {
    if !d.isnative() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "rnp cannot yet create or compute on byte-swapped arrays ({})",
            d.str_code()
        )));
    }
    Ok(())
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
    let base = descr_from_any_aligned(&t.get_item(0)?, align)?;
    let second = t.get_item(1)?;
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
    #[pyo3(signature = (obj, align = false, copy = false))]
    fn py_new(obj: &Bound<'_, PyAny>, align: bool, copy: bool) -> PyResult<Self> {
        let _ = copy;
        Ok(PyDType {
            d: descr_from_any_aligned(obj, align)?,
        })
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
        self.d.str_code()
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
        if self.d.is_struct() || self.d.subarray_def().is_some() {
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
        false
    }

    /// `dtype.type`: the scalar class, looked up in the shim's registry.
    #[getter]
    fn r#type<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let key: String = match self.d.dt {
            DType::Bytes(_) => "bytes_".into(),
            DType::Str(_) => "str_".into(),
            DType::Void(_) | DType::Struct(_) | DType::SubArray(_) => "void".into(),
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
        0
    }

    #[getter]
    fn metadata(&self) -> Option<()> {
        None
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
    fn descr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
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

    fn __repr__(&self) -> String {
        self.d.repr_string()
    }

    fn __str__(&self) -> String {
        self.d.str_string()
    }

    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.d.hash(&mut h);
        h.finish()
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
            Ok(d) => d == self.d,
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
        let cls = slf.get_type().into_any();
        let args = PyTuple::new(py, [slf.borrow().d.str_code()])?.into_any();
        PyTuple::new(py, [cls, args])
    }
}
