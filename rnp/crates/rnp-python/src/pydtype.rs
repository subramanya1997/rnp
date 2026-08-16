//! The `dtype` pyclass.

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::PyTypeInfo;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PyString, PyType};

use rnp_core::DType;

#[pyclass(name = "dtype", module = "_rnp", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyDType {
    pub dt: DType,
}

impl PyDType {
    pub fn new(dt: DType) -> Self {
        PyDType { dt }
    }
}

/// Resolve any of numpy's dtype spellings to a `DType`.
///
/// Accepts: `dtype` instances, strings (`"int64"`, `"i8"`, `"<f8"`), the
/// Python builtin types (`int`, `float`, `bool`, `complex`), and `None`
/// (which numpy resolves to `float64`).
pub fn dtype_from_any(obj: &Bound<'_, PyAny>) -> PyResult<DType> {
    if obj.is_none() {
        return Ok(DType::F64);
    }
    if let Ok(d) = obj.extract::<PyDType>() {
        return Ok(d.dt);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        let name = s.to_str()?;
        return DType::from_str(name).ok_or_else(|| {
            PyTypeError::new_err(format!("data type '{}' not understood", name))
        });
    }
    if let Ok(t) = obj.cast::<PyType>() {
        // Builtin Python types map to numpy's defaults. Order matters: bool
        // is checked before int because it is a subclass.
        if t.is(&PyBool::type_object(obj.py())) {
            return Ok(DType::Bool);
        }
        if t.is(&PyInt::type_object(obj.py())) {
            return Ok(DType::I64);
        }
        if t.is(&PyFloat::type_object(obj.py())) {
            return Ok(DType::F64);
        }
        if t.is(&PyComplex::type_object(obj.py())) {
            return Ok(DType::C128);
        }
        // A scalar-alias class from the shim carries its dtype.
        if let Ok(attr) = t.getattr("dtype") {
            if let Ok(d) = attr.extract::<PyDType>() {
                return Ok(d.dt);
            }
        }
    }
    // Anything exposing a `.dtype` (e.g. our own ndarray).
    if let Ok(attr) = obj.getattr("dtype") {
        if !attr.is(obj) {
            if let Ok(d) = attr.extract::<PyDType>() {
                return Ok(d.dt);
            }
        }
    }
    Err(PyTypeError::new_err(format!(
        "Cannot interpret '{}' as a data type",
        obj.repr()?.to_str()?
    )))
}

#[pymethods]
impl PyDType {
    #[new]
    #[pyo3(signature = (obj))]
    fn py_new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(PyDType {
            dt: dtype_from_any(obj)?,
        })
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.dt.name()
    }

    #[getter]
    fn kind(&self) -> String {
        self.dt.kind().to_string()
    }

    #[getter]
    fn char(&self) -> String {
        self.dt.char_code().to_string()
    }

    #[getter]
    fn itemsize(&self) -> usize {
        self.dt.itemsize()
    }

    #[getter]
    fn alignment(&self) -> usize {
        self.dt.alignment()
    }

    #[getter]
    fn num(&self) -> i32 {
        self.dt.num()
    }

    #[getter]
    fn byteorder(&self) -> String {
        self.dt.byteorder().to_string()
    }

    #[getter]
    fn str(&self) -> String {
        self.dt.str_code()
    }

    #[getter]
    fn base(&self) -> PyDType {
        *self
    }

    #[getter]
    fn ndim(&self) -> usize {
        0
    }

    #[getter]
    fn shape(&self) -> () {}

    #[getter]
    fn isnative(&self) -> bool {
        true
    }

    #[getter]
    fn hasobject(&self) -> bool {
        false
    }

    #[getter]
    fn fields(&self) -> Option<()> {
        None
    }

    #[getter]
    fn names(&self) -> Option<()> {
        None
    }

    #[getter]
    fn subdtype(&self) -> Option<()> {
        None
    }

    fn newbyteorder(&self) -> PyDType {
        *self
    }

    fn __repr__(&self) -> String {
        format!("dtype('{}')", self.dt.name())
    }

    fn __str__(&self) -> &'static str {
        self.dt.name()
    }

    fn __hash__(&self) -> u64 {
        self.dt.num() as u64
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let eq = match dtype_from_any(other) {
            Ok(d) => d == self.dt,
            // numpy returns NotImplemented-ish False for junk comparisons.
            Err(_) => false,
        };
        match op {
            CompareOp::Eq => Ok(eq.into_pyobject(py)?.to_owned().unbind().into_any()),
            CompareOp::Ne => Ok((!eq).into_pyobject(py)?.to_owned().unbind().into_any()),
            _ => Ok(py.NotImplemented()),
        }
    }
}
