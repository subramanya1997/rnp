//! The `ndarray` pyclass: properties, indexing, operators, buffer protocol.

use std::ffi::{c_int, CString};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyBufferError, PyIndexError, PyKeyError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PySliceMethods, PyTuple};

use rnp_core::{binary, BinOp, DType, NdArray, Scalar};

use crate::convert::{array_from_any, operand, scalar_from_py, scalar_to_py};
use rnp_core::printing;
use crate::pydtype::{dtype_from_any, PyDType};

#[pyclass(name = "ndarray", module = "_rnp", subclass)]
pub struct PyNdArray {
    pub arr: NdArray,
}

impl PyNdArray {
    pub fn wrap(arr: NdArray) -> PyNdArray {
        PyNdArray { arr }
    }

    pub fn into_py_any(arr: NdArray, py: Python<'_>) -> PyResult<Py<PyNdArray>> {
        Py::new(py, PyNdArray::wrap(arr))
    }
}

/// One parsed element of an index expression.
enum Idx {
    Int(isize),
    Slice(Py<PySlice>),
    NewAxis,
    Ellipsis,
}

fn parse_index_items(key: &Bound<'_, PyAny>) -> PyResult<Vec<Idx>> {
    let py = key.py();
    let items: Vec<Bound<'_, PyAny>> = if let Ok(t) = key.cast::<PyTuple>() {
        t.iter().collect()
    } else {
        vec![key.clone()]
    };
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        if it.is_none() {
            out.push(Idx::NewAxis);
        } else if it.is(&py.Ellipsis()) {
            out.push(Idx::Ellipsis);
        } else if let Ok(s) = it.cast::<PySlice>() {
            out.push(Idx::Slice(s.clone().unbind()));
        } else if let Ok(i) = it.extract::<isize>() {
            out.push(Idx::Int(i));
        } else if it.cast::<PyNdArray>().is_ok() || it.cast::<PyList>().is_ok() {
            return Err(PyNotImplementedError::new_err(
                "advanced (fancy/boolean) indexing is not implemented yet",
            ));
        } else {
            return Err(PyIndexError::new_err(
                "only integers, slices (`:`), ellipsis (`...`) and newaxis \
                 (`None`) are valid indices",
            ));
        }
    }
    Ok(out)
}

/// Apply a basic-indexing expression, returning the resulting view and
/// whether every axis was consumed by an integer (numpy returns a scalar in
/// that case).
fn index_view(arr: &NdArray, key: &Bound<'_, PyAny>) -> PyResult<(NdArray, bool)> {
    let py = key.py();
    let items = parse_index_items(key)?;
    let consuming = items
        .iter()
        .filter(|i| matches!(i, Idx::Int(_) | Idx::Slice(_)))
        .count();
    if consuming > arr.ndim() {
        return Err(PyIndexError::new_err(format!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            arr.ndim(),
            consuming
        )));
    }
    let mut ellipsis_seen = false;
    let mut cur = arr.clone();
    let mut ax = 0usize;
    let mut all_int = true;

    for item in items {
        match item {
            Idx::NewAxis => {
                all_int = false;
                cur = cur.insert_axis(ax);
                ax += 1;
            }
            Idx::Ellipsis => {
                if ellipsis_seen {
                    return Err(PyIndexError::new_err(
                        "an index can only have a single ellipsis ('...')",
                    ));
                }
                ellipsis_seen = true;
                all_int = false;
                ax += arr.ndim() - consuming;
            }
            Idx::Int(i) => {
                let n = cur.shape[ax];
                let i = if i < 0 { i + n } else { i };
                if i < 0 || i >= n {
                    return Err(PyIndexError::new_err(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        if i < 0 { i - n } else { i },
                        ax,
                        n
                    )));
                }
                cur = cur.slice_axis(ax, i, 1, 1).remove_axis(ax);
            }
            Idx::Slice(s) => {
                all_int = false;
                let ind = s.bind(py).indices(cur.shape[ax])?;
                cur = cur.slice_axis(ax, ind.start, ind.slicelength as isize, ind.step);
                ax += 1;
            }
        }
    }
    let scalarize = all_int && cur.ndim() == 0;
    Ok((cur, scalarize))
}

fn nested_list<'py>(
    py: Python<'py>,
    arr: &NdArray,
    index: &mut Vec<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    if index.len() == arr.ndim() {
        let v = arr.get(index).map_err(crate::err)?;
        return scalar_to_py(py, v);
    }
    let n = arr.shape[index.len()];
    let mut items = Vec::with_capacity(n.max(0) as usize);
    for i in 0..n {
        index.push(i);
        items.push(nested_list(py, arr, index)?);
        index.pop();
    }
    Ok(PyList::new(py, items)?.into_any())
}

/// numpy's `a.flags` object: attribute *and* item access.
#[pyclass(name = "flagsobj", module = "_rnp")]
pub struct PyFlags {
    c: bool,
    f: bool,
    w: bool,
    o: bool,
    a: bool,
}

#[pymethods]
impl PyFlags {
    #[getter]
    fn c_contiguous(&self) -> bool {
        self.c
    }
    #[getter]
    fn f_contiguous(&self) -> bool {
        self.f
    }
    #[getter]
    fn contiguous(&self) -> bool {
        self.c
    }
    #[getter]
    fn fortran(&self) -> bool {
        self.f
    }
    #[getter]
    fn writeable(&self) -> bool {
        self.w
    }
    #[getter]
    fn owndata(&self) -> bool {
        self.o
    }
    #[getter]
    fn aligned(&self) -> bool {
        self.a
    }
    #[getter]
    fn behaved(&self) -> bool {
        self.a && self.w
    }

    fn __getitem__(&self, key: &str) -> PyResult<bool> {
        match key {
            "C_CONTIGUOUS" | "C" => Ok(self.c),
            "F_CONTIGUOUS" | "F" => Ok(self.f),
            "WRITEABLE" | "W" => Ok(self.w),
            "OWNDATA" | "O" => Ok(self.o),
            "ALIGNED" | "A" => Ok(self.a),
            "BEHAVED" | "B" => Ok(self.a && self.w),
            other => Err(PyKeyError::new_err(other.to_string())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "  C_CONTIGUOUS : {}\n  F_CONTIGUOUS : {}\n  OWNDATA : {}\n  \
             WRITEABLE : {}\n  ALIGNED : {}",
            pybool(self.c),
            pybool(self.f),
            pybool(self.o),
            pybool(self.w),
            pybool(self.a)
        )
    }
}

fn pybool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// Buffer-protocol bookkeeping kept alive for the lifetime of a `Py_buffer`.
struct BufInfo {
    shape: Vec<ffi::Py_ssize_t>,
    strides: Vec<ffi::Py_ssize_t>,
    format: CString,
}

#[pymethods]
impl PyNdArray {
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.arr.shape.iter().map(|&d| d as usize))
    }

    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.arr.strides.iter().copied())
    }

    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType::new(self.arr.dtype)
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.arr.ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.arr.size()
    }

    #[getter]
    fn itemsize(&self) -> usize {
        self.arr.itemsize()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.arr.nbytes()
    }

    #[getter]
    fn flags(&self) -> PyFlags {
        PyFlags {
            c: self.arr.flags.c_contiguous,
            f: self.arr.flags.f_contiguous,
            w: self.arr.flags.writeable,
            o: self.arr.flags.owndata,
            a: self.arr.flags.aligned,
        }
    }

    #[getter]
    fn base(&self) -> Option<()> {
        None
    }

    #[getter]
    #[pyo3(name = "T")]
    fn t(&self, py: Python<'_>) -> PyResult<Py<PyNdArray>> {
        PyNdArray::into_py_any(self.arr.transpose(), py)
    }

    #[pyo3(signature = (*axes))]
    fn transpose(&self, py: Python<'_>, axes: &Bound<'_, PyTuple>) -> PyResult<Py<PyNdArray>> {
        if axes.is_empty() {
            return PyNdArray::into_py_any(self.arr.transpose(), py);
        }
        let spec = shape_from_args(axes)?;
        let perm: Vec<usize> = spec
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + self.arr.ndim() as isize) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        PyNdArray::into_py_any(self.arr.permute(&perm).map_err(crate::err)?, py)
    }

    #[pyo3(signature = (*shape))]
    fn reshape(&self, py: Python<'_>, shape: &Bound<'_, PyTuple>) -> PyResult<Py<PyNdArray>> {
        let spec = shape_from_args(shape)?;
        PyNdArray::into_py_any(self.arr.reshape(&spec).map_err(crate::err)?, py)
    }

    fn ravel(&self, py: Python<'_>) -> PyResult<Py<PyNdArray>> {
        let n = self.arr.size() as isize;
        PyNdArray::into_py_any(self.arr.reshape(&[n]).map_err(crate::err)?, py)
    }

    fn flatten(&self, py: Python<'_>) -> PyResult<Py<PyNdArray>> {
        let n = self.arr.size() as isize;
        let c = self.arr.copy();
        PyNdArray::into_py_any(c.reshape(&[n]).map_err(crate::err)?, py)
    }

    #[pyo3(signature = (dtype, copy = true))]
    fn astype(
        &self,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        copy: bool,
    ) -> PyResult<Py<PyNdArray>> {
        let d = dtype_from_any(dtype)?;
        let out = if d == self.arr.dtype && !copy {
            self.arr.clone()
        } else {
            self.arr.astype(d)
        };
        PyNdArray::into_py_any(out, py)
    }

    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyNdArray>> {
        PyNdArray::into_py_any(self.arr.copy(), py)
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Py<PyNdArray>> {
        self.copy(py)
    }

    fn fill(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let s = scalar_from_py(value)
            .ok_or_else(|| PyTypeError::new_err("fill() requires a scalar value"))?;
        self.arr.fill(s);
        Ok(())
    }

    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut index = Vec::new();
        nested_list(py, &self.arr, &mut index)
    }

    #[pyo3(signature = (*args))]
    fn item<'py>(&self, py: Python<'py>, args: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyAny>> {
        let v = if args.is_empty() {
            if self.arr.size() != 1 {
                return Err(PyValueError::new_err(
                    "can only convert an array of size 1 to a Python scalar",
                ));
            }
            self.arr.get_flat(0)
        } else if args.len() == 1 {
            let i: isize = args.get_item(0)?.extract()?;
            let n = self.arr.size() as isize;
            let i = if i < 0 { i + n } else { i };
            if i < 0 || i >= n {
                return Err(PyIndexError::new_err("index out of bounds"));
            }
            self.arr.get_flat(i as usize)
        } else {
            let idx = shape_from_args(args)?;
            self.arr.get(&idx).map_err(crate::err)?
        };
        scalar_to_py(py, v)
    }

    fn __len__(&self) -> PyResult<usize> {
        if self.arr.ndim() == 0 {
            return Err(PyTypeError::new_err("len() of unsized object"));
        }
        Ok(self.arr.shape[0] as usize)
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (view, scalarize) = index_view(&self.arr, key)?;
        if scalarize {
            return scalar_to_py(py, view.get(&[]).map_err(crate::err)?);
        }
        Ok(PyNdArray::into_py_any(view, py)?.into_bound(py).into_any())
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let (mut view, _) = index_view(&self.arr, key)?;
        if !view.flags.writeable {
            return Err(PyValueError::new_err(
                "assignment destination is read-only",
            ));
        }
        if let Some(s) = scalar_from_py(value) {
            view.fill(s);
            return Ok(());
        }
        // Array (or nested-sequence) assignment: broadcast the source.
        let src = array_from_any(value, Some(view.dtype), false)?;
        let src = rnp_core::iter::broadcast_to(&src, &view.shape).map_err(crate::err)?;
        let dst_offsets: Vec<isize> =
            rnp_core::iter::offsets(&view.shape, &view.strides, view.byte_offset).collect();
        let src_offsets: Vec<isize> =
            rnp_core::iter::offsets(&src.shape, &src.strides, src.byte_offset).collect();
        for (&d, &s) in dst_offsets.iter().zip(src_offsets.iter()) {
            view.write_at(d, src.read_at(s));
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        printing::repr(&self.arr)
    }

    fn __str__(&self) -> String {
        printing::to_str(&self.arr)
    }

    fn __bool__(&self) -> PyResult<bool> {
        match self.arr.size() {
            1 => Ok(match self.arr.get_flat(0) {
                Scalar::Bool(b) => b,
                Scalar::Int(i) => i != 0,
                Scalar::Uint(u) => u != 0,
                Scalar::Float(f) => f != 0.0,
                Scalar::Complex(c) => c.re != 0.0 || c.im != 0.0,
            }),
            0 => Err(PyValueError::new_err(
                "The truth value of an empty array is ambiguous.",
            )),
            _ => Err(PyValueError::new_err(
                "The truth value of an array with more than one element is \
                 ambiguous. Use a.any() or a.all()",
            )),
        }
    }

    fn __float__(&self) -> PyResult<f64> {
        if self.arr.size() != 1 {
            return Err(PyTypeError::new_err(
                "only length-1 arrays can be converted to Python scalars",
            ));
        }
        Ok(match self.arr.get_flat(0) {
            Scalar::Bool(b) => b as u8 as f64,
            Scalar::Int(i) => i as f64,
            Scalar::Uint(u) => u as f64,
            Scalar::Float(f) => f,
            Scalar::Complex(c) => c.re,
        })
    }

    fn __int__(&self) -> PyResult<i64> {
        if self.arr.size() != 1 {
            return Err(PyTypeError::new_err(
                "only length-1 arrays can be converted to Python scalars",
            ));
        }
        Ok(match self.arr.get_flat(0) {
            Scalar::Bool(b) => b as i64,
            Scalar::Int(i) => i,
            Scalar::Uint(u) => u as i64,
            Scalar::Float(f) => f as i64,
            Scalar::Complex(c) => c.re as i64,
        })
    }

    // ---- operators -----------------------------------------------------

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Add, false)
    }
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Add, true)
    }
    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Sub, false)
    }
    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Sub, true)
    }
    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Mul, false)
    }
    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Mul, true)
    }
    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Div, false)
    }
    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Div, true)
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let bop = match op {
            CompareOp::Eq => BinOp::Eq,
            CompareOp::Ne => BinOp::Ne,
            CompareOp::Lt => BinOp::Lt,
            CompareOp::Le => BinOp::Le,
            CompareOp::Gt => BinOp::Gt,
            CompareOp::Ge => BinOp::Ge,
        };
        self.binop(py, other, bop, false)
    }

    // ---- buffer protocol -----------------------------------------------

    /// Expose the array through the C buffer protocol so real numpy can wrap
    /// it zero-copy (`np.asarray(rnp_array)`), which is what the
    /// cross-verification harness relies on.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        let me = slf.borrow();
        let arr = &me.arr;
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE && !arr.flags.writeable {
            return Err(PyBufferError::new_err("Object is not writable"));
        }
        if (flags & ffi::PyBUF_STRIDES) != ffi::PyBUF_STRIDES && !arr.flags.c_contiguous {
            return Err(PyBufferError::new_err(
                "Object is not C-contiguous; a strided buffer request is required",
            ));
        }

        let info = Box::new(BufInfo {
            shape: arr.shape.iter().map(|&d| d as ffi::Py_ssize_t).collect(),
            strides: arr.strides.iter().map(|&s| s as ffi::Py_ssize_t).collect(),
            format: CString::new(arr.dtype.buffer_format()).unwrap(),
        });

        // SAFETY: `view` is a valid, writable Py_buffer supplied by CPython.
        // The pointers we store into it (buffer bytes, and the shape/strides/
        // format owned by `info`) stay alive until __releasebuffer__ frees
        // `info`; the data pointer is kept alive by the `obj` reference below.
        unsafe {
            (*view).buf = arr.buffer.as_ptr().offset(arr.byte_offset) as *mut std::ffi::c_void;
            (*view).len = arr.nbytes() as ffi::Py_ssize_t;
            (*view).readonly = (!arr.flags.writeable) as c_int;
            (*view).itemsize = arr.itemsize() as ffi::Py_ssize_t;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                info.format.as_ptr() as *mut std::os::raw::c_char
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = arr.ndim() as c_int;
            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                info.shape.as_ptr() as *mut ffi::Py_ssize_t
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                info.strides.as_ptr() as *mut ffi::Py_ssize_t
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = Box::into_raw(info) as *mut std::ffi::c_void;
            (*view).obj = slf.clone().into_any().into_ptr();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        // SAFETY: `internal` was set by __getbuffer__ to a Box<BufInfo>.
        unsafe {
            if !(*view).internal.is_null() {
                drop(Box::from_raw((*view).internal as *mut BufInfo));
                (*view).internal = std::ptr::null_mut();
            }
        }
    }
}

impl PyNdArray {
    fn binop(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: BinOp,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        let rhs = match operand(other, self.arr.dtype)? {
            Some(a) => a,
            None => return Ok(py.NotImplemented()),
        };
        let (a, b) = if reflected {
            (&rhs, &self.arr)
        } else {
            (&self.arr, &rhs)
        };
        let out = binary(a, b, op).map_err(crate::err)?;
        Ok(PyNdArray::into_py_any(out, py)?.into_any())
    }
}

/// Accept `reshape(2, 3)` and `reshape((2, 3))` alike.
pub fn shape_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Vec<isize>> {
    if args.len() == 1 {
        let first = args.get_item(0)?;
        if let Ok(t) = first.cast::<PyTuple>() {
            return t.iter().map(|x| x.extract::<isize>()).collect();
        }
        if let Ok(l) = first.cast::<PyList>() {
            return l.iter().map(|x| x.extract::<isize>()).collect();
        }
        return Ok(vec![first.extract::<isize>()?]);
    }
    args.iter().map(|x| x.extract::<isize>()).collect()
}

/// Shape argument for the creation functions (`zeros(3)`, `zeros((2, 3))`).
pub fn shape_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(i) = obj.extract::<isize>() {
        return Ok(vec![i]);
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        return t.iter().map(|x| x.extract::<isize>()).collect();
    }
    if let Ok(l) = obj.cast::<PyList>() {
        return l.iter().map(|x| x.extract::<isize>()).collect();
    }
    Err(PyTypeError::new_err(
        "shape must be an int or a sequence of ints",
    ))
}

/// Helper used by the module-level ufunc wrappers.
pub fn ufunc2(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    op: BinOp,
) -> PyResult<Py<PyAny>> {
    // Weak-scalar promotion applies when exactly one side is an array.
    let a_is_arr = a.cast::<PyNdArray>().is_ok();
    let b_is_arr = b.cast::<PyNdArray>().is_ok();
    let (lhs, rhs) = if a_is_arr && !b_is_arr {
        let lhs = array_from_any(a, None, false)?;
        let rhs = operand(b, lhs.dtype)?
            .ok_or_else(|| PyTypeError::new_err("unsupported operand for ufunc"))?;
        (lhs, rhs)
    } else if b_is_arr && !a_is_arr {
        let rhs = array_from_any(b, None, false)?;
        let lhs = operand(a, rhs.dtype)?
            .ok_or_else(|| PyTypeError::new_err("unsupported operand for ufunc"))?;
        (lhs, rhs)
    } else {
        (
            array_from_any(a, None, false)?,
            array_from_any(b, None, false)?,
        )
    };
    let out = binary(&lhs, &rhs, op).map_err(crate::err)?;
    Ok(PyNdArray::into_py_any(out, py)?.into_any())
}

/// `a.flags` as a plain dict, for code that wants mapping access.
#[allow(dead_code)]
pub fn flags_dict<'py>(py: Python<'py>, arr: &NdArray) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("C_CONTIGUOUS", arr.flags.c_contiguous)?;
    d.set_item("F_CONTIGUOUS", arr.flags.f_contiguous)?;
    d.set_item("WRITEABLE", arr.flags.writeable)?;
    d.set_item("OWNDATA", arr.flags.owndata)?;
    d.set_item("ALIGNED", arr.flags.aligned)?;
    Ok(d)
}

/// Build a `DType` for creation functions, defaulting to float64 like numpy.
pub fn dtype_or_default(obj: Option<&Bound<'_, PyAny>>, default: DType) -> PyResult<DType> {
    match obj {
        None => Ok(default),
        Some(o) if o.is_none() => Ok(default),
        Some(o) => dtype_from_any(o),
    }
}
