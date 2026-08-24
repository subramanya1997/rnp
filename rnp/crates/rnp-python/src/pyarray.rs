//! The `ndarray` pyclass: properties, indexing, operators, buffer protocol.

use std::ffi::{c_int, CString};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyBufferError, PyIndexError, PyKeyError, PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PySliceMethods, PyTuple};

use rnp_core::reduce::{mean_dtype, reduce_all, reduce_axis, reduce_dtype, ReduceOp};
use rnp_core::element::Element;
use rnp_core::indexing::{Indexed, TakeMode};
use rnp_core::{binary, BinOp, DType, NdArray, Scalar};

use crate::convert::{array_from_any, flexible_to_py, npflexible_to_py, npscalar_to_py,
                     operand_for, scalar_from_py, scalar_to_py};
use rnp_core::printing;
use crate::pydtype::{descr_from_any, dtype_from_any, PyDType};

#[pyclass(name = "ndarray", module = "_rnp", subclass)]
pub struct PyNdArray {
    pub arr: NdArray,
    /// numpy's `a.base`: the object that actually owns the memory. Views of
    /// views collapse onto the owner, exactly as numpy does.
    pub base: Option<Py<PyAny>>,
}

impl PyNdArray {
    pub fn wrap(arr: NdArray) -> PyNdArray {
        PyNdArray { arr, base: None }
    }

    pub fn into_py_any(arr: NdArray, py: Python<'_>) -> PyResult<Py<PyNdArray>> {
        Py::new(py, PyNdArray::wrap(arr))
    }

    /// Wrap a *view* of `parent`, propagating the base chain.
    pub fn view_of(
        arr: NdArray,
        parent: &Bound<'_, PyNdArray>,
    ) -> PyResult<Py<PyNdArray>> {
        // numpy's base-collapsing rule (`PyArray_SetBaseObject`): walk up the
        // chain only while the next link is *another ndarray of the same
        // type*. An array that adopted foreign memory has a non-array base
        // (`bytes`, an `mmap`, ...), and numpy stops there — `a[1:3].base` is
        // `a`, not the bytes object.
        let py = parent.py();
        let base = match &parent.borrow().base {
            Some(b) if b.bind(py).is_instance_of::<PyNdArray>() => b.clone_ref(py),
            _ => parent.clone().into_any().unbind(),
        };
        Py::new(
            parent.py(),
            PyNdArray {
                arr,
                base: Some(base),
            },
        )
    }
}

/// A deep copy of an array cell's contents, for the methods that must not
/// write through a shared buffer.
fn self_arr_copy(slf: &Bound<'_, PyNdArray>) -> NdArray {
    slf.borrow().arr.copy()
}

fn nested_list<'py>(
    py: Python<'py>,
    arr: &NdArray,
    index: &mut Vec<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    if index.len() == arr.ndim() {
        return crate::convert::element_to_py(py, arr, arr.byte_index(index));
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

/// `np.flatiter`: a 1-D, C-order view of an array's elements that supports
/// iteration, indexing and assignment (writing through to the base array).
#[pyclass(name = "flatiter", module = "_rnp")]
pub struct PyFlatIter {
    pub arr: NdArray,
    pos: usize,
    base: Py<PyAny>,
}

const FLAT_BAD_INDEX: &str = "only integers, slices (`:`), ellipsis (`...`) and \
                              integer or boolean arrays are valid indices";

/// What a flatiter index resolves to.
enum FlatKey {
    /// Every element (`...`, `()`).
    All,
    One(isize),
    Range(isize, isize, isize),
    /// Explicit flat positions plus the shape the result should take.
    Gather(Vec<i64>, Vec<isize>),
}

impl PyFlatIter {
    fn n(&self) -> isize {
        self.arr.size() as isize
    }

    fn offset(&self, i: usize) -> isize {
        rnp_core::indexing::flat_offset(&self.arr, i)
    }

    /// Materialise the iterator as a fresh 1-D array.
    fn to_array(&self) -> NdArray {
        let n = self.arr.size();
        let out = NdArray::empty_descr(vec![n as isize], self.arr.descr).expect("flat alloc");
        let isz = self.arr.itemsize() as isize;
        for i in 0..n {
            let s = self.offset(i);
            if self.arr.dtype().is_flexible() {
                out.write_raw_at(i as isize * isz, self.arr.raw_bytes_at(s));
            } else {
                out.write_at(i as isize * isz, self.arr.read_at(s));
            }
        }
        out
    }

    fn gather(&self, positions: &[i64], shape: &[isize]) -> PyResult<NdArray> {
        let n = self.n();
        let out = NdArray::empty_descr(shape.to_vec(), self.arr.descr).map_err(crate::err)?;
        let isz = self.arr.itemsize() as isize;
        for (k, &raw) in positions.iter().enumerate() {
            let v = if raw < 0 { raw + n as i64 } else { raw };
            if v < 0 || v >= n as i64 {
                return Err(PyIndexError::new_err(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    raw, n
                )));
            }
            let s = self.offset(v as usize);
            if self.arr.dtype().is_flexible() {
                out.write_raw_at(k as isize * isz, self.arr.raw_bytes_at(s));
            } else {
                out.write_at(k as isize * isz, self.arr.read_at(s));
            }
        }
        Ok(out)
    }

    /// Classify a flatiter index expression.
    fn parse(&self, key: &Bound<'_, PyAny>, assigning: bool) -> PyResult<FlatKey> {
        let py = key.py();
        if let Ok(t) = key.cast::<PyTuple>() {
            return match t.len() {
                0 => {
                    if assigning {
                        Err(PyIndexError::new_err(
                            "Assigning to a flat iterator with a 0-D index is not supported",
                        ))
                    } else {
                        Ok(FlatKey::All)
                    }
                }
                1 => self.parse(&t.get_item(0)?, assigning),
                k => Err(PyIndexError::new_err(format!(
                    "too many indices for flat iterator: flat iterator is \
                     1-dimensional, but {} were indexed",
                    k
                ))),
            };
        }
        if key.is(&py.Ellipsis()) {
            return Ok(FlatKey::All);
        }
        if key.is_none() {
            return Err(PyIndexError::new_err(FLAT_BAD_INDEX));
        }
        if key.is_instance_of::<pyo3::types::PyBool>() {
            PyErr::warn(
                py,
                &py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
                std::ffi::CString::new(
                    "In the future, 0-dimensional boolean index will be treated as \
                     a mask, not as a scalar. (Deprecated NumPy 2.5)",
                )
                .unwrap()
                .as_c_str(),
                1,
            )?;
            let v: bool = key.extract()?;
            return Ok(FlatKey::Range(0, if v { 1 } else { 0 }, 1));
        }
        if let Ok(s) = key.cast::<PySlice>() {
            let ind = s.indices(self.n())?;
            return Ok(FlatKey::Range(ind.start, ind.slicelength as isize, ind.step));
        }
        // A flatiter used as an index behaves like its underlying array.
        let owned;
        let as_arr: Option<&NdArray> = if let Ok(f) = key.cast::<PyFlatIter>() {
            owned = f.borrow().to_array();
            Some(&owned)
        } else if let Ok(a) = key.cast::<PyNdArray>() {
            owned = a.borrow().arr.clone();
            Some(&owned)
        } else {
            None
        };
        if let Some(a) = as_arr {
            if a.dtype() == DType::Bool {
                if a.size() as isize != self.n() {
                    return Err(PyIndexError::new_err(format!(
                        "boolean index did not match indexed flat iterator; size of \
                         iterator is {} but size of corresponding boolean is {}",
                        self.n(),
                        a.size()
                    )));
                }
                let pos: Vec<i64> = a
                    .to_vec()
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| matches!(s, Scalar::Bool(true)))
                    .map(|(i, _)| i as i64)
                    .collect();
                let n = pos.len() as isize;
                return Ok(FlatKey::Gather(pos, vec![n]));
            }
            if a.dtype().is_flexible() || !a.dtype().is_integer() {
                // An empty non-integer index is accepted (it matches the array
                // path, where `arr[[]]` must work).
                if a.size() == 0 {
                    return Ok(FlatKey::Gather(vec![], vec![0]));
                }
                return Err(PyIndexError::new_err(FLAT_BAD_INDEX));
            }
            return Ok(FlatKey::Gather(int_values(a), a.shape.clone()));
        }
        if let Ok(i) = key.extract::<isize>() {
            return Ok(FlatKey::One(i));
        }
        if key.is_instance_of::<PyList>() {
            let a = array_from_any(key, None, false)
                .map_err(|_| PyIndexError::new_err(FLAT_BAD_INDEX))?;
            if a.dtype() == DType::Bool {
                return Err(PyIndexError::new_err(
                    "boolean indices for iterators are not supported",
                ));
            }
            if a.size() == 0 {
                return Ok(FlatKey::Gather(vec![], vec![0]));
            }
            if !a.dtype().is_integer() {
                if !a.dtype().is_float() {
                    return Err(PyIndexError::new_err(FLAT_BAD_INDEX));
                }
                PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
                    std::ffi::CString::new(
                        "Invalid non-array indices for iterator objects are \
                         deprecated. (Deprecated NumPy 2.5)",
                    )
                    .unwrap()
                    .as_c_str(),
                    1,
                )?;
            }
            return Ok(FlatKey::Gather(int_values(&a), a.shape.clone()));
        }
        Err(PyIndexError::new_err(FLAT_BAD_INDEX))
    }
}

fn int_values(a: &NdArray) -> Vec<i64> {
    a.to_vec()
        .into_iter()
        .map(|s| match s {
            Scalar::Int(i) => i,
            Scalar::Uint(u) => u as i64,
            Scalar::Bool(b) => b as i64,
            Scalar::Float(f) => f as i64,
            Scalar::Complex(c) => c.re as i64,
        })
        .collect()
}

#[pymethods]
impl PyFlatIter {
    fn __len__(&self) -> usize {
        self.arr.size()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(mut slf: PyRefMut<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if slf.pos >= slf.arr.size() {
            return Ok(None);
        }
        let py = slf.py();
        let off = slf.offset(slf.pos);
        slf.pos += 1;
        if slf.arr.dtype().is_flexible() || slf.arr.dtype().is_object() {
            return Ok(Some(npflexible_to_py(py, &slf.arr, off)?));
        }
        Ok(Some(npscalar_to_py(py, slf.arr.dtype(), slf.arr.read_at(off))?))
    }

    #[getter]
    fn base(&self, py: Python<'_>) -> Py<PyAny> {
        self.base.clone_ref(py)
    }

    #[getter]
    fn index(&self) -> usize {
        self.pos
    }

    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let mut rem = self.pos;
        let mut out = vec![0usize; self.arr.ndim()];
        for ax in (0..self.arr.ndim()).rev() {
            let d = self.arr.shape[ax].max(1) as usize;
            out[ax] = rem % d;
            rem /= d;
        }
        PyTuple::new(py, out)
    }

    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyNdArray>> {
        PyNdArray::into_py_any(self.to_array(), py)
    }

    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&Bound<'_, PyAny>>,
        copy: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyNdArray>> {
        let _ = copy;
        let a = self.to_array();
        let a = match dtype {
            Some(d) if !d.is_none() => a.astype_descr(descr_from_any(d)?),
            _ => a,
        };
        PyNdArray::into_py_any(a, py)
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let out = match self.parse(key, false)? {
            FlatKey::All => self.to_array(),
            FlatKey::One(i) => {
                let n = self.n();
                let v = if i < 0 { i + n } else { i };
                if v < 0 || v >= n {
                    return Err(PyIndexError::new_err(format!(
                        "index {} is out of bounds for axis 0 with size {}",
                        i, n
                    )));
                }
                let off = self.offset(v as usize);
                if self.arr.dtype().is_flexible() {
                    return npflexible_to_py(py, &self.arr, off);
                }
                return npscalar_to_py(py, self.arr.dtype(), self.arr.read_at(off));
            }
            FlatKey::Range(start, len, step) => {
                let pos: Vec<i64> = (0..len).map(|k| (start + k * step) as i64).collect();
                self.gather(&pos, &[len])?
            }
            FlatKey::Gather(pos, shape) => self.gather(&pos, &shape)?,
        };
        Ok(PyNdArray::into_py_any(out, py)?.into_bound(py).into_any())
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if !self.arr.flags.writeable {
            return Err(PyValueError::new_err("assignment destination is read-only"));
        }
        let positions: Vec<i64> = match self.parse(key, true)? {
            FlatKey::All => (0..self.n() as i64).collect(),
            FlatKey::One(i) => vec![i as i64],
            FlatKey::Range(start, len, step) => {
                (0..len).map(|k| (start + k * step) as i64).collect()
            }
            FlatKey::Gather(pos, _) => pos,
        };
        let n = self.n();
        let mut resolved = Vec::with_capacity(positions.len());
        for &raw in &positions {
            let v = if raw < 0 { raw + n as i64 } else { raw };
            if v < 0 || v >= n as i64 {
                return Err(PyIndexError::new_err(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    raw, n
                )));
            }
            resolved.push(v as usize);
        }
        if !self.arr.dtype().is_flexible() {
            if let Some(s) = scalar_from_py(value) {
                for &v in &resolved {
                    self.arr.write_at(self.offset(v), s);
                }
                return Ok(());
            }
        }
        let src = array_from_any(value, Some(self.arr.dtype()), false)?;
        let want = [resolved.len() as isize];
        let src = match rnp_core::iter::broadcast_to(&src, &want) {
            Ok(s) => s,
            Err(_) => {
                // A multi-dimensional value is flattened first, as numpy does
                // for flatiter assignment.
                let n = src.size() as isize;
                let flat = src.copy().reshape(&[n]).map_err(crate::err)?;
                rnp_core::iter::broadcast_to(&flat, &want).map_err(crate::err)?
            }
        };
        let src = src.in_order_of(&self.arr);
        let src_offs: Vec<isize> =
            rnp_core::iter::offsets(&src.shape, &src.strides, src.byte_offset).collect();
        for (k, &v) in resolved.iter().enumerate() {
            let d = self.offset(v);
            if self.arr.dtype().is_flexible() {
                self.arr.write_raw_at(d, src.raw_bytes_at(src_offs[k]));
            } else {
                self.arr.write_at(d, src.read_at(src_offs[k]));
            }
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("<numpy.flatiter object at 0x{:x}>", self as *const _ as usize)
    }
}

/// numpy's `a.flags` object: attribute *and* item access.
///
/// It is a live *proxy* onto the owning array, not a snapshot, because
/// `x.flags.writeable = False` has to change `x`.
#[pyclass(name = "flagsobj", module = "_rnp")]
pub struct PyFlags {
    owner: Py<PyNdArray>,
}

impl PyFlags {
    fn get<T>(&self, py: Python<'_>, f: impl FnOnce(&rnp_core::array::Flags) -> T) -> T {
        f(&self.owner.borrow(py).arr.flags)
    }
}

#[pymethods]
impl PyFlags {
    #[getter]
    fn c_contiguous(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.c_contiguous)
    }
    #[getter]
    fn f_contiguous(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.f_contiguous)
    }
    #[getter]
    fn contiguous(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.c_contiguous)
    }
    #[getter]
    fn fortran(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.f_contiguous)
    }
    #[getter]
    fn writeable(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.writeable)
    }
    #[setter]
    fn set_writeable(&self, py: Python<'_>, value: bool) -> PyResult<()> {
        self.set_flag(py, "WRITEABLE", value)
    }
    #[getter]
    fn owndata(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.owndata)
    }
    #[getter]
    fn aligned(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.aligned)
    }
    #[setter]
    fn set_aligned(&self, py: Python<'_>, value: bool) -> PyResult<()> {
        self.set_flag(py, "ALIGNED", value)
    }
    #[getter]
    fn behaved(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.aligned && f.writeable)
    }
    #[getter]
    fn carray(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.c_contiguous && f.aligned && f.writeable)
    }
    #[getter]
    fn farray(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.f_contiguous && !f.c_contiguous && f.aligned && f.writeable)
    }
    #[getter]
    fn forc(&self, py: Python<'_>) -> bool {
        self.get(py, |f| f.f_contiguous || f.c_contiguous)
    }
    #[getter]
    fn writebackifcopy(&self) -> bool {
        false
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<bool> {
        match key {
            "C_CONTIGUOUS" | "C" => Ok(self.c_contiguous(py)),
            "F_CONTIGUOUS" | "F" => Ok(self.f_contiguous(py)),
            "WRITEABLE" | "W" => Ok(self.writeable(py)),
            "OWNDATA" | "O" => Ok(self.owndata(py)),
            "ALIGNED" | "A" => Ok(self.aligned(py)),
            "BEHAVED" | "B" => Ok(self.behaved(py)),
            "CARRAY" | "CA" => Ok(self.carray(py)),
            "FARRAY" | "FA" => Ok(self.farray(py)),
            "FORC" => Ok(self.forc(py)),
            "WRITEBACKIFCOPY" => Ok(false),
            other => Err(PyKeyError::new_err(other.to_string())),
        }
    }

    fn __setitem__(&self, py: Python<'_>, key: &str, value: bool) -> PyResult<()> {
        self.set_flag(py, key, value)
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        match other.cast::<PyFlags>() {
            Ok(o) => {
                let a = self.owner.borrow(py).arr.flags;
                let b = o.borrow().owner.borrow(py).arr.flags;
                a == b
            }
            Err(_) => false,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "  C_CONTIGUOUS : {}\n  F_CONTIGUOUS : {}\n  OWNDATA : {}\n  \
             WRITEABLE : {}\n  ALIGNED : {}\n  WRITEBACKIFCOPY : {}",
            pybool(self.c_contiguous(py)),
            pybool(self.f_contiguous(py)),
            pybool(self.owndata(py)),
            pybool(self.writeable(py)),
            pybool(self.aligned(py)),
            pybool(false)
        )
    }
}

impl PyFlags {
    /// numpy only lets three flags be set from Python, and only
    /// `WRITEABLE`/`ALIGNED` mean anything here.
    fn set_flag(&self, py: Python<'_>, key: &str, value: bool) -> PyResult<()> {
        match key {
            "WRITEABLE" | "W" => {
                if value {
                    // numpy refuses to re-enable writing on an array that
                    // neither owns its data nor has a writeable base.
                    let ok = {
                        let me = self.owner.borrow(py);
                        me.arr.flags.owndata
                            || match &me.base {
                                None => true,
                                Some(b) => match b.bind(py).cast::<PyNdArray>() {
                                    Ok(bb) => bb.borrow().arr.flags.writeable,
                                    Err(_) => true,
                                },
                            }
                    };
                    if !ok {
                        return Err(PyValueError::new_err(
                            "cannot set WRITEABLE flag to True of this array",
                        ));
                    }
                }
                self.owner.borrow_mut(py).arr.flags.writeable = value;
                Ok(())
            }
            "ALIGNED" | "A" => {
                if value && !self.owner.borrow(py).arr.flags.aligned {
                    return Err(PyValueError::new_err(
                        "cannot set ALIGNED flag to True of this array",
                    ));
                }
                Ok(())
            }
            "WRITEBACKIFCOPY" => {
                if value {
                    return Err(PyValueError::new_err(
                        "cannot set WRITEBACKIFCOPY flag to True",
                    ));
                }
                Ok(())
            }
            "C_CONTIGUOUS" | "C" | "F_CONTIGUOUS" | "F" | "OWNDATA" | "O" => Err(
                PyKeyError::new_err(format!("Cannot set flag {key}")),
            ),
            other => Err(PyKeyError::new_err(other.to_string())),
        }
    }
}

/// The text of a `kind=` argument. numpy accepts `str` and `bytes` (the C
/// converter runs the bytes through the same path); anything else is a
/// `TypeError` naming the operation. Probed against numpy 2.5.2.
fn kind_text(kind: &Bound<'_, PyAny>, what: &str) -> PyResult<String> {
    if let Ok(s) = kind.extract::<String>() {
        return Ok(s);
    }
    if let Ok(b) = kind.extract::<Vec<u8>>() {
        if kind.is_instance_of::<pyo3::types::PyBytes>() {
            return Ok(String::from_utf8_lossy(&b).into_owned());
        }
    }
    let ty = kind.get_type().name()?;
    Err(PyTypeError::new_err(format!(
        "{what} kind must be str, not {ty}"
    )))
}

/// `repr()` of the offending `kind=`, for numpy's verbatim error text.
fn kind_repr(kind: &Bound<'_, PyAny>) -> String {
    kind.repr()
        .map(|r| r.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "?".into())
}

/// numpy accepts `kind=` and `stable=` but not both; only the stability
/// matters here.
///
/// numpy matches `kind=` on its **first character, case-insensitively**:
/// `q`uicksort, `h`eapsort, `m`ergesort, `s`table — so `'quick'`, `'Q'` and
/// `'sfoobar'` are all accepted, while `'timsort'`/`'radixsort'`/`'introsort'`
/// are rejected. Probed from numpy 2.5.2, error text included.
fn sort_stable(kind: Option<&Bound<'_, PyAny>>, stable: Option<bool>) -> PyResult<bool> {
    let k = match kind {
        Some(o) if !o.is_none() => Some((kind_text(o, "sort")?, o)),
        _ => None,
    };
    if k.is_some() && stable.is_some() {
        return Err(PyValueError::new_err(
            "`kind` and keyword parameters can't be provided at the same time. Use only one of them.",
        ));
    }
    let Some((text, obj)) = k else {
        return Ok(stable.unwrap_or(false));
    };
    match text.chars().next().map(|c| c.to_ascii_lowercase()) {
        // quicksort and heapsort are not stable; mergesort is numpy's alias
        // for the stable kind.
        Some('q') | Some('h') => Ok(false),
        Some('m') | Some('s') => Ok(true),
        _ => Err(PyValueError::new_err(format!(
            "sort kind must be one of 'quick', 'heap', or 'stable' (got {})",
            kind_repr(obj)
        ))),
    }
}

/// `partition`/`argpartition` accept only the exact string `'introselect'`
/// — no prefix matching, case-sensitive, and `None` is a `TypeError` (unlike
/// `sort`, whose `kind=None` means "default"). Probed from numpy 2.5.2.
fn check_select_kind(kind: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(obj) = kind else { return Ok(()) };
    let text = kind_text(obj, "select")?;
    if text == "introselect" {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "select kind must be 'introselect' (got {})",
        kind_repr(obj)
    )))
}

/// The length of `arr` along `axis`. A 0-d array behaves as length 1, which
/// is the shape `norm_sort_axis` already pretends it has.
fn axis_len(arr: &NdArray, axis: usize) -> usize {
    arr.shape.get(axis).copied().unwrap_or(1) as usize
}

/// Normalise the `kth=` argument of `partition`/`argpartition` into in-range
/// ranks. `kth` is a single index or a sequence of them; each may be negative
/// and is taken modulo the axis length.
///
/// Two probed details numpy gets subtly right (verified against 2.5.2):
///   * a non-integer index is `TypeError: Partition index must be integer`,
///     raised before any bounds check — and it is the `__index__` protocol,
///     so a `np.int64` or a `bool` is accepted while a float is not;
///   * an out-of-bounds *negative* index is reported by its **normalised**
///     value, not the one the caller passed: on a length-5 axis `kth=-6`
///     raises `kth(=-1) out of bounds (5)`, while `kth=5` raises `kth(=5)`.
fn norm_kths(kth: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<usize>> {
    let items: Vec<Bound<'_, PyAny>> = if let Ok(seq) = kth.try_iter() {
        seq.collect::<PyResult<Vec<_>>>()?
    } else {
        vec![kth.clone()]
    };
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let raw = it
            .call_method0("__index__")
            .and_then(|i| i.extract::<isize>())
            .map_err(|_| PyTypeError::new_err("Partition index must be integer"))?;
        let k = if raw < 0 { raw + n as isize } else { raw };
        if k < 0 || k >= n as isize {
            return Err(PyValueError::new_err(format!(
                "kth(={k}) out of bounds ({n})"
            )));
        }
        out.push(k as usize);
    }
    Ok(out)
}

/// Normalise a (possibly negative) sort axis.
fn norm_sort_axis(arr: &NdArray, axis: isize) -> PyResult<usize> {
    let nd = arr.ndim().max(1) as isize;
    let ax = if axis < 0 { axis + nd } else { axis };
    if ax < 0 || ax >= nd {
        return Err(PyValueError::new_err(format!(
            "axis {axis} is out of bounds for array of dimension {}",
            arr.ndim()
        )));
    }
    Ok(ax as usize)
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
    /// `ndarray(shape, dtype=float, buffer=None, offset=0, strides=None,
    /// order=None)` — numpy's low-level constructor. With `buffer=` the
    /// exporter's memory is adopted zero-copy; see `adopt.rs`.
    #[new]
    #[pyo3(signature = (shape, dtype = None, buffer = None, offset = 0, strides = None, order = None))]
    fn py_new(
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        dtype: Option<&Bound<'_, PyAny>>,
        buffer: Option<&Bound<'_, PyAny>>,
        offset: i64,
        strides: Option<&Bound<'_, PyAny>>,
        order: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyNdArray> {
        crate::adopt::ndarray_new(py, shape, dtype, buffer, offset, strides, order)
    }

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
        PyDType::from_descr(self.arr.descr)
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
    fn flags(slf: &Bound<'_, Self>) -> PyFlags {
        PyFlags {
            owner: slf.clone().unbind(),
        }
    }

    /// `a.real` — a view for real dtypes, a strided view of the real parts
    /// for complex ones (numpy returns a view in both cases).
    #[getter]
    fn real(slf: &Bound<'_, Self>) -> PyResult<Py<PyNdArray>> {
        Self::component(slf, 0)
    }

    #[getter]
    fn imag(slf: &Bound<'_, Self>) -> PyResult<Py<PyNdArray>> {
        Self::component(slf, 1)
    }

    #[getter]
    fn base(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.base.as_ref().map(|b| b.clone_ref(py))
    }

    /// numpy's `base` is read-only; this port lets it be assigned so that
    /// pure-Python subclasses written against the shim (`memmap`) can record
    /// the object that owns their memory, which is what numpy's C-level
    /// `PyArray_SetBaseObject` does for them.
    #[setter]
    fn set_base(&mut self, obj: Option<Py<PyAny>>) {
        self.base = obj;
    }

    #[getter]
    #[pyo3(name = "T")]
    fn t(slf: &Bound<'_, Self>) -> PyResult<Py<PyNdArray>> {
        let arr = slf.borrow().arr.transpose();
        PyNdArray::view_of(arr, slf)
    }

    #[pyo3(signature = (*axes))]
    fn transpose(slf: &Bound<'_, Self>, axes: &Bound<'_, PyTuple>) -> PyResult<Py<PyNdArray>> {
        let me = slf.borrow().arr.clone();
        if axes.is_empty() || (axes.len() == 1 && axes.get_item(0)?.is_none()) {
            return PyNdArray::view_of(me.transpose(), slf);
        }
        let spec = shape_from_args(axes)?;
        let perm: Vec<usize> = spec
            .iter()
            .map(|&a| {
                if a < 0 {
                    (a + me.ndim() as isize) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        PyNdArray::view_of(me.permute(&perm).map_err(crate::err)?, slf)
    }

    #[pyo3(signature = (*shape))]
    fn reshape(slf: &Bound<'_, Self>, shape: &Bound<'_, PyTuple>) -> PyResult<Py<PyNdArray>> {
        let me = slf.borrow().arr.clone();
        let spec = shape_from_args(shape)?;
        let out = me.reshape(&spec).map_err(crate::err)?;
        if std::sync::Arc::ptr_eq(&out.buffer, &me.buffer) {
            PyNdArray::view_of(out, slf)
        } else {
            PyNdArray::into_py_any(out, slf.py())
        }
    }

    fn ravel(slf: &Bound<'_, Self>) -> PyResult<Py<PyNdArray>> {
        let me = slf.borrow().arr.clone();
        let n = me.size() as isize;
        let out = me.reshape(&[n]).map_err(crate::err)?;
        if std::sync::Arc::ptr_eq(&out.buffer, &me.buffer) {
            PyNdArray::view_of(out, slf)
        } else {
            PyNdArray::into_py_any(out, slf.py())
        }
    }

    #[pyo3(signature = (order = "C"))]
    fn flatten(&self, py: Python<'_>, order: &str) -> PyResult<Py<PyNdArray>> {
        let _ = order;
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
        let d = descr_from_any(dtype)?;
        let out = if d == self.arr.descr && d.alias == self.arr.descr.alias {
            if copy {
                self.arr.copy()
            } else {
                self.arr.clone()
            }
        } else if d.dt == self.arr.dtype() {
            // Same storage, different byte order (or C-type spelling): a
            // straight swap-and-relabel, no value cast involved.
            self.arr.copy().into_descr(d)
        } else if d.is_struct() || self.arr.descr.is_struct() {
            // Structured casts are field-by-field; see `fields.rs`.
            crate::fields::struct_astype(&self.arr, d)?
        } else if d.dt.is_flexible() || self.arr.dtype().is_flexible() {
            return Err(PyNotImplementedError::new_err(format!(
                "astype from {} to {} is not implemented yet",
                self.arr.dtype(), d.dt
            )));
        } else {
            self.arr.astype_descr(d)
        };
        PyNdArray::into_py_any(out, py)
    }

    fn copy(&self, py: Python<'_>) -> PyResult<Py<PyNdArray>> {
        PyNdArray::into_py_any(self.arr.copy(), py)
    }

    /// numpy's array interface (version 3). `strides` is `None` exactly when
    /// the array is C-contiguous, which is what numpy reports.
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", PyTuple::new(py, self.arr.shape.iter().map(|&x| x as usize))?)?;
        d.set_item("typestr", self.arr.descr.str_code())?;
        d.set_item(
            "descr",
            crate::pydtype::PyDType::from_descr(self.arr.descr).descr(py)?,
        )?;
        if self.arr.flags.c_contiguous {
            d.set_item("strides", py.None())?;
        } else {
            d.set_item("strides", PyTuple::new(py, self.arr.strides.iter().copied())?)?;
        }
        let ptr = self.arr.buffer.as_ptr() as usize;
        let addr = (ptr as isize + self.arr.byte_offset) as usize;
        d.set_item("data", (addr, !self.arr.flags.writeable))?;
        d.set_item("version", 3)?;
        Ok(d)
    }

    /// `a.sort(axis=-1, kind=None, order=None, *, stable=None)`, in place.
    #[pyo3(signature = (axis = -1, kind = None, order = None, *, stable = None))]
    fn sort(
        &mut self,
        axis: isize,
        kind: Option<&Bound<'_, PyAny>>,
        order: Option<&Bound<'_, PyAny>>,
        stable: Option<bool>,
    ) -> PyResult<()> {
        if order.is_some_and(|o| !o.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "sort(order=) is not implemented yet",
            ));
        }
        let stable = sort_stable(kind, stable)?;
        let ax = norm_sort_axis(&self.arr, axis)?;
        rnp_core::sort::sort_inplace(&mut self.arr, ax, stable).map_err(crate::err)
    }

    #[pyo3(signature = (axis = Some(-1), kind = None, order = None, *, stable = None))]
    fn argsort(
        &self,
        py: Python<'_>,
        axis: Option<isize>,
        kind: Option<&Bound<'_, PyAny>>,
        order: Option<&Bound<'_, PyAny>>,
        stable: Option<bool>,
    ) -> PyResult<Py<PyNdArray>> {
        if order.is_some_and(|o| !o.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "argsort(order=) is not implemented yet",
            ));
        }
        let stable = sort_stable(kind, stable)?;
        let (arr, ax) = match axis {
            None => {
                let flat = self.arr.reshape(&[-1]).map_err(crate::err)?;
                (flat, 0)
            }
            Some(a) => {
                let ax = norm_sort_axis(&self.arr, a)?;
                (self.arr.clone(), ax)
            }
        };
        let out = rnp_core::sort::argsort(&arr, ax, stable).map_err(crate::err)?;
        PyNdArray::into_py_any(out, py)
    }

    /// `a.partition(kth, axis=-1, kind='introselect', order=None)`, in place.
    #[pyo3(signature = (kth, axis = -1, kind = None, order = None))]
    fn partition(
        &mut self,
        kth: &Bound<'_, PyAny>,
        axis: isize,
        kind: Option<&Bound<'_, PyAny>>,
        order: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if order.is_some_and(|o| !o.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "partition(order=) is not implemented yet",
            ));
        }
        check_select_kind(kind)?;
        let ax = norm_sort_axis(&self.arr, axis)?;
        let n = axis_len(&self.arr, ax);
        let kths = norm_kths(kth, n)?;
        rnp_core::sort::partition_inplace(&mut self.arr, &kths, ax).map_err(crate::err)
    }

    /// `a.argpartition(kth, axis=-1, kind='introselect', order=None)`.
    #[pyo3(signature = (kth, axis = Some(-1), kind = None, order = None))]
    fn argpartition(
        &self,
        py: Python<'_>,
        kth: &Bound<'_, PyAny>,
        axis: Option<isize>,
        kind: Option<&Bound<'_, PyAny>>,
        order: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyNdArray>> {
        if order.is_some_and(|o| !o.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "argpartition(order=) is not implemented yet",
            ));
        }
        check_select_kind(kind)?;
        let (arr, ax) = match axis {
            None => (self.arr.reshape(&[-1]).map_err(crate::err)?, 0),
            Some(a) => {
                let ax = norm_sort_axis(&self.arr, a)?;
                (self.arr.clone(), ax)
            }
        };
        let n = axis_len(&arr, ax);
        let kths = norm_kths(kth, n)?;
        let out = rnp_core::sort::argpartition(&arr, &kths, ax).map_err(crate::err)?;
        PyNdArray::into_py_any(out, py)
    }

    #[pyo3(signature = (v, side = "left", sorter = None))]
    fn searchsorted(
        &self,
        py: Python<'_>,
        v: &Bound<'_, PyAny>,
        side: &str,
        sorter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if sorter.is_some_and(|o| !o.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "searchsorted(sorter=) is not implemented yet",
            ));
        }
        let right = match side {
            "left" => false,
            "right" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "'{other}' is an invalid value for keyword 'side'"
                )))
            }
        };
        let vv = array_from_any(v, Some(self.arr.dtype()), false)?;
        let out = rnp_core::sort::searchsorted(&self.arr, &vv, right).map_err(crate::err)?;
        if out.ndim() == 0 {
            return crate::convert::npscalar_to_py(py, out.dtype(), out.get_flat(0))
                .map(|b| b.unbind());
        }
        Ok(PyNdArray::into_py_any(out, py)?.into_any())
    }

    /// `ndarray.setflags(write=None, align=None, uic=None)`.
    #[pyo3(signature = (write = None, align = None, uic = None))]
    fn setflags(
        slf: &Bound<'_, Self>,
        write: Option<bool>,
        align: Option<bool>,
        uic: Option<bool>,
    ) -> PyResult<()> {
        let flags = PyFlags {
            owner: slf.clone().unbind(),
        };
        let py = slf.py();
        if let Some(w) = write {
            flags.set_flag(py, "WRITEABLE", w)?;
        }
        if let Some(a) = align {
            flags.set_flag(py, "ALIGNED", a)?;
        }
        if let Some(u) = uic {
            flags.set_flag(py, "WRITEBACKIFCOPY", u)?;
        }
        Ok(())
    }

    /// `ndarray.byteswap(inplace=False)`: reverse the bytes of every element,
    /// leaving the *descriptor* alone, so the values change.
    #[pyo3(signature = (inplace = false))]
    fn byteswap(slf: &Bound<'_, Self>, inplace: bool) -> PyResult<Py<PyAny>> {
        if inplace {
            {
                let mut me = slf.borrow_mut();
                if !me.arr.flags.writeable {
                    return Err(PyValueError::new_err(
                        "assignment destination is read-only",
                    ));
                }
                me.arr.byteswap_inplace();
            }
            return Ok(slf.clone().into_any().unbind());
        }
        let mut out = self_arr_copy(slf);
        out.byteswap_inplace();
        Ok(PyNdArray::into_py_any(out, slf.py())?.into_any())
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
        if self.arr.dtype().is_object() {
            if self.arr.size() != 1 {
                return Err(PyValueError::new_err(
                    "can only convert an array of size 1 to a Python scalar",
                ));
            }
            return Ok(crate::objects::read(py, &self.arr, self.arr.byte_offset));
        }
        if self.arr.dtype().is_flexible() {
            if self.arr.size() != 1 || !args.is_empty() {
                return Err(PyNotImplementedError::new_err(
                    "item() on flexible dtypes is only implemented for size-1 arrays",
                ));
            }
            return flexible_to_py(py, &self.arr, self.arr.byte_offset);
        }
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


    /// `a.getfield(dtype, offset)` — a field-typed view at a byte offset.
    #[pyo3(signature = (dtype, offset = 0))]
    fn getfield<'py>(
        slf: &Bound<'py, Self>,
        dtype: &Bound<'py, PyAny>,
        offset: isize,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::fields::getfield(slf, dtype, offset)
    }

    /// `a.setfield(value, dtype, offset)`.
    #[pyo3(signature = (value, dtype, offset = 0))]
    fn setfield(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        dtype: &Bound<'_, PyAny>,
        offset: isize,
    ) -> PyResult<()> {
        crate::fields::setfield(slf, value, dtype, offset)
    }

    /// `a.view(dtype)` — reinterpret the same bytes under another dtype.
    #[pyo3(signature = (dtype = None, type_ = None))]
    fn view(
        slf: &Bound<'_, Self>,
        dtype: Option<&Bound<'_, PyAny>>,
        type_: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        // `a.view(SomeSubclass)` may spell the type in either argument.
        let mut want_type: Option<Bound<'_, PyAny>> = type_
            .filter(|t| !t.is_none())
            .map(|t| t.clone());
        let mut dtype = dtype;
        if let Some(o) = dtype {
            if !o.is_none()
                && o.cast::<pyo3::types::PyType>().is_ok()
                && descr_from_any(o).is_err()
            {
                want_type = Some(o.clone());
                dtype = None;
            }
        }
        // numpy's `a.view()` on a subclass instance keeps the subclass.
        if want_type.is_none() && !slf.get_type().is(&py.get_type::<PyNdArray>()) {
            want_type = Some(slf.get_type().into_any());
        }
        // Whatever the result's dtype, the *type* of the object is decided
        // once, at the end: `retype` turns a plain view into an instance of
        // the requested subclass.
        let retype = |v: Py<PyNdArray>| -> PyResult<Py<PyAny>> {
            let Some(t) = &want_type else {
                return Ok(v.into_any());
            };
            let ty = t.cast::<pyo3::types::PyType>().map_err(|_| {
                PyTypeError::new_err("type must be a sub-type of ndarray type")
            })?;
            let b = v.bind(py);
            let (arr, base) = {
                let me = b.borrow();
                (me.arr.clone(), me.base.as_ref().map(|x| x.clone_ref(py)))
            };
            crate::adopt::new_of_type(py, ty, arr, base, Some(slf.as_any()))
        };
        let me = slf.borrow().arr.clone();
        let d = match dtype {
            None => return retype(PyNdArray::view_of(me, slf)?),
            Some(o) if o.is_none() => return retype(PyNdArray::view_of(me, slf)?),
            Some(o) => descr_from_any(o)?,
        };
        let old = me.itemsize();
        let new = d.itemsize();
        let mut out = me.clone();
        out.descr = d;
        if old != new {
            if me.ndim() == 0 {
                return Err(PyValueError::new_err(
                    "Changing the dtype of a 0d array is only supported if the \
                     itemsize is unchanged",
                ));
            }
            let last = me.ndim() - 1;
            if me.strides[last] != old as isize {
                return Err(PyValueError::new_err(
                    "To change to a dtype of a different size, the last axis \
                     must be contiguous",
                ));
            }
            let bytes = me.shape[last] * old as isize;
            if bytes % new as isize != 0 {
                return Err(PyValueError::new_err(
                    "When changing to a larger dtype, its size must be a \
                     divisor of the total size in bytes of the last axis of \
                     the array.",
                ));
            }
            out.shape[last] = bytes / new as isize;
            out.strides[last] = new as isize;
        }
        out.flags.owndata = false;
        out.update_flags();
        retype(PyNdArray::view_of(out, slf)?)
    }

    #[pyo3(signature = (axis = None, out = None, keepdims = false))]
    fn all<'py>(
        &self,
        py: Python<'py>,
        axis: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.bool_reduce(py, true, axis, out, keepdims)
    }

    #[pyo3(signature = (axis = None, out = None, keepdims = false))]
    fn any<'py>(
        &self,
        py: Python<'py>,
        axis: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.bool_reduce(py, false, axis, out, keepdims)
    }

    /// `a.repeat(repeats, axis=None)`.
    #[pyo3(signature = (repeats, axis = None))]
    fn repeat<'py>(
        &self,
        py: Python<'py>,
        repeats: &Bound<'py, PyAny>,
        axis: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ax = resolve_axis(&self.arr, axis)?;
        let n = match ax {
            None => self.arr.size(),
            Some(a) => self.arr.shape[a].max(0) as usize,
        };
        let reps = array_from_any(repeats, None, false)?;
        let counts: Vec<i64> = if reps.size() == 1 {
            let v = as_i64(reps.get_flat(0));
            vec![v; n]
        } else {
            if reps.size() != n {
                return Err(PyValueError::new_err(format!(
                    "operands could not be broadcast together with shape ({},) ({},)",
                    n,
                    reps.size()
                )));
            }
            (0..reps.size()).map(|i| as_i64(reps.get_flat(i))).collect()
        };
        let mut idx = Vec::new();
        for (i, &c) in counts.iter().enumerate() {
            if c < 0 {
                return Err(PyValueError::new_err("negative dimensions are not allowed"));
            }
            for _ in 0..c {
                idx.push(Scalar::Int(i as i64));
            }
        }
        let iarr = NdArray::from_scalars(&idx, DType::I64).map_err(crate::err)?;
        let res = rnp_core::indexing::take(&self.arr, &iarr, ax, TakeMode::Raise)
            .map_err(crate::err)?;
        Ok(PyNdArray::into_py_any(res, py)?.into_bound(py).into_any())
    }

    #[pyo3(signature = (axis = None))]
    fn squeeze(slf: &Bound<'_, Self>, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyNdArray>> {
        let me = slf.borrow().arr.clone();
        let drop: Vec<usize> = match axis {
            None => (0..me.ndim()).filter(|&i| me.shape[i] == 1).collect(),
            Some(o) if o.is_none() => (0..me.ndim()).filter(|&i| me.shape[i] == 1).collect(),
            Some(o) => {
                let mut v = Vec::new();
                for a in shape_from_any(o)? {
                    let a = if a < 0 { a + me.ndim() as isize } else { a };
                    if a < 0 || a as usize >= me.ndim() {
                        return Err(PyValueError::new_err(format!(
                            "axis {} is out of bounds for array of dimension {}",
                            a,
                            me.ndim()
                        )));
                    }
                    if me.shape[a as usize] != 1 {
                        return Err(PyValueError::new_err(
                            "cannot select an axis to squeeze out which has size \
                             not equal to one",
                        ));
                    }
                    v.push(a as usize);
                }
                v
            }
        };
        let mut out = me.clone();
        for &a in drop.iter().rev() {
            out.shape.remove(a);
            out.strides.remove(a);
        }
        out.flags.owndata = false;
        out.update_flags();
        PyNdArray::view_of(out, slf)
    }

    fn swapaxes(slf: &Bound<'_, Self>, a: isize, b: isize) -> PyResult<Py<PyNdArray>> {
        let me = slf.borrow().arr.clone();
        let nd = me.ndim() as isize;
        let (i, j) = (
            if a < 0 { a + nd } else { a },
            if b < 0 { b + nd } else { b },
        );
        if i < 0 || i >= nd || j < 0 || j >= nd {
            return Err(PyValueError::new_err(format!(
                "axis {} is out of bounds for array of dimension {}",
                a, nd
            )));
        }
        let mut perm: Vec<usize> = (0..me.ndim()).collect();
        perm.swap(i as usize, j as usize);
        PyNdArray::view_of(me.permute(&perm).map_err(crate::err)?, slf)
    }

    // ---- item selection --------------------------------------------------

    #[pyo3(signature = (indices, axis = None, out = None, mode = "raise"))]
    fn take<'py>(
        &self,
        py: Python<'py>,
        indices: &Bound<'py, PyAny>,
        axis: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = TakeMode::from_str(mode).ok_or_else(|| {
            PyValueError::new_err(format!(
                "clipmode not understood; expected 'raise', 'wrap' or 'clip' (got '{mode}')"
            ))
        })?;
        let idx = array_from_any(indices, None, false)?;
        if !idx.dtype().is_integer() && idx.dtype() != DType::Bool {
            if idx.size() != 0 {
                return Err(PyIndexError::new_err(
                    "arrays used as indices must be of integer (or boolean) type",
                ));
            }
        }
        let ax = resolve_axis(&self.arr, axis)?;
        let res = rnp_core::indexing::take(&self.arr, &idx, ax, m).map_err(crate::err)?;
        store_or_wrap(py, res, out)
    }

    #[pyo3(signature = (indices, values, mode = "raise"))]
    fn put(
        &mut self,
        indices: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        mode: &str,
    ) -> PyResult<()> {
        let m = TakeMode::from_str(mode).ok_or_else(|| {
            PyValueError::new_err(format!(
                "clipmode not understood; expected 'raise', 'wrap' or 'clip' (got '{mode}')"
            ))
        })?;
        let idx = array_from_any(indices, None, false)?;
        let ivals: Vec<i64> = idx
            .to_vec()
            .into_iter()
            .map(|s| match s {
                Scalar::Int(i) => i,
                Scalar::Uint(u) => u as i64,
                Scalar::Bool(b) => b as i64,
                Scalar::Float(f) => f as i64,
                Scalar::Complex(c) => c.re as i64,
            })
            .collect();
        let vals = array_from_any(values, Some(self.arr.dtype()), false)?;
        rnp_core::indexing::put(&self.arr, &ivals, &vals, m).map_err(crate::err)
    }

    #[pyo3(signature = (condition, axis = None, out = None))]
    fn compress<'py>(
        &self,
        py: Python<'py>,
        condition: &Bound<'py, PyAny>,
        axis: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cond = array_from_any(condition, None, false)?;
        let ax = resolve_axis(&self.arr, axis)?;
        let res = rnp_core::indexing::compress(&self.arr, &cond, ax).map_err(crate::err)?;
        store_or_wrap(py, res, out)
    }

    #[pyo3(signature = (choices, out = None, mode = "raise"))]
    fn choose<'py>(
        &self,
        py: Python<'py>,
        choices: &Bound<'py, PyAny>,
        out: Option<&Bound<'py, PyAny>>,
        mode: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m = TakeMode::from_str(mode)
            .ok_or_else(|| PyValueError::new_err("mode must be 'raise', 'wrap' or 'clip'"))?;
        let mut arrays = Vec::new();
        for c in choices.try_iter()? {
            arrays.push(array_from_any(&c?, None, false)?);
        }
        let res = rnp_core::indexing::choose(&self.arr, &arrays, m).map_err(crate::err)?;
        store_or_wrap(py, res, out)
    }

    fn nonzero<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let cols = rnp_core::indexing::nonzero(&self.arr);
        let mut out = Vec::with_capacity(cols.len());
        for c in cols {
            out.push(PyNdArray::into_py_any(c, py)?);
        }
        PyTuple::new(py, out)
    }

    #[getter]
    fn flat(slf: &Bound<'_, Self>) -> PyResult<Py<PyFlatIter>> {
        let base = match &slf.borrow().base {
            Some(b) => b.clone_ref(slf.py()),
            None => slf.clone().into_any().unbind(),
        };
        Py::new(
            slf.py(),
            PyFlatIter {
                arr: slf.borrow().arr.clone(),
                pos: 0,
                base,
            },
        )
    }

    // ---- reductions ----------------------------------------------------

    #[pyo3(signature = (axis = None, dtype = None, out = None, keepdims = false))]
    fn sum<'py>(
        &self,
        py: Python<'py>,
        axis: Option<&Bound<'py, PyAny>>,
        dtype: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.reduce(py, ReduceOp::Sum, axis, dtype, out, keepdims)
    }

    #[pyo3(signature = (axis = None, dtype = None, out = None, keepdims = false))]
    fn prod<'py>(
        &self,
        py: Python<'py>,
        axis: Option<&Bound<'py, PyAny>>,
        dtype: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.reduce(py, ReduceOp::Prod, axis, dtype, out, keepdims)
    }

    #[pyo3(signature = (axis = None, out = None, keepdims = false))]
    fn min<'py>(
        &self,
        py: Python<'py>,
        axis: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.reduce(py, ReduceOp::Min, axis, None, out, keepdims)
    }

    #[pyo3(signature = (axis = None, out = None, keepdims = false))]
    fn max<'py>(
        &self,
        py: Python<'py>,
        axis: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.reduce(py, ReduceOp::Max, axis, None, out, keepdims)
    }

    #[pyo3(signature = (axis = None, out = None, keepdims = false))]
    fn argmin<'py>(
        &self,
        py: Python<'py>,
        axis: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.reduce(py, ReduceOp::ArgMin, axis, None, out, keepdims)
    }

    #[pyo3(signature = (axis = None, out = None, keepdims = false))]
    fn argmax<'py>(
        &self,
        py: Python<'py>,
        axis: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.reduce(py, ReduceOp::ArgMax, axis, None, out, keepdims)
    }

    /// `a.mean()`.
    ///
    /// Transcribed from `numpy/_core/_methods.py::_mean`: bool and integer
    /// operands accumulate in float64, float16 accumulates in float32 and is
    /// converted back at the end, and the division happens *in the
    /// accumulator's own type* (so a complex mean goes through numpy's
    /// complex divide, not a component-wise one).
    #[pyo3(signature = (axis = None, dtype = None, out = None, keepdims = false))]
    fn mean<'py>(
        &self,
        py: Python<'py>,
        axis: Option<&Bound<'py, PyAny>>,
        dtype: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if out.is_some_and(|o| !o.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "mean(out=) is not implemented yet",
            ));
        }
        let is_half = self.arr.dtype() == DType::F16;
        let acc_dt = match dtype {
            Some(d) if !d.is_none() => dtype_from_any(d)?,
            _ if is_half => DType::F32,
            _ => mean_dtype(self.arr.dtype()),
        };
        let promoted = self.arr.astype(acc_dt);
        let ax = resolve_axis(&self.arr, axis)?;
        let n = match ax {
            None => self.arr.size(),
            Some(a) => self.arr.shape[a].max(0) as usize,
        };
        let out_dt = if is_half { DType::F16 } else { acc_dt };
        let divide = |total: Scalar| -> Scalar {
            if n == 0 {
                // numpy warns and yields NaN (NaN+NaNj for complex).
                return match acc_dt {
                    DType::C64 | DType::C128 => Scalar::Complex(
                        num_complex::Complex::new(f64::NAN, f64::NAN),
                    ),
                    d => Scalar::Float(f64::NAN).cast(d),
                };
            }
            #[allow(unused_assignments)]
            // complex64 is the odd one out: numpy's scalar divide widens to
            // double and rounds once, so it is a true component-wise
            // division, while complex128 goes through the complex divide
            // (multiply by the reciprocal). Both were probed against numpy
            // over thousands of random values.
            if acc_dt == DType::C64 {
                if let Scalar::Complex(c) = total {
                    let re = (c.re / n as f64) as f32 as f64;
                    let im = (c.im / n as f64) as f32 as f64;
                    return Scalar::Complex(num_complex::Complex::new(re, im)).cast(out_dt);
                }
            }
            // `dispatch_dtype!` expands to a match, so the result is
            // assigned from every arm rather than returned.
            let v: Scalar = rnp_core::dispatch_dtype!(acc_dt, A, {
                let count = A::from_scalar(Scalar::Float(n as f64));
                rnp_core::ops::Arith::a_div(A::from_scalar(total), count).to_scalar()
            });
            v.cast(out_dt)
        };
        match ax {
            None => {
                let total = if n == 0 {
                    Scalar::Float(0.0).cast(acc_dt)
                } else {
                    reduce_all(&promoted, ReduceOp::Sum).map_err(crate::err)?
                };
                let v = divide(total);
                if keepdims {
                    let shape = vec![1isize; self.arr.ndim()];
                    let mut a = NdArray::zeros(shape, out_dt).map_err(crate::err)?;
                    a.fill(v);
                    return Ok(PyNdArray::into_py_any(a, py)?.into_bound(py).into_any());
                }
                npscalar_to_py(py, out_dt, v)
            }
            Some(a) => {
                let sums = reduce_axis(&promoted, a, ReduceOp::Sum, keepdims)
                    .map_err(crate::err)?;
                let res = NdArray::zeros(sums.shape.clone(), out_dt).map_err(crate::err)?;
                let src: Vec<isize> =
                    rnp_core::iter::offsets(&sums.shape, &sums.strides, sums.byte_offset)
                        .collect();
                let dst: Vec<isize> =
                    rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset)
                        .collect();
                for (&s, &d) in src.iter().zip(dst.iter()) {
                    res.write_at(d, divide(sums.read_at(s)));
                }
                Ok(PyNdArray::into_py_any(res, py)?.into_bound(py).into_any())
            }
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        if self.arr.ndim() == 0 {
            return Err(PyTypeError::new_err("len() of unsized object"));
        }
        Ok(self.arr.shape[0] as usize)
    }

    /// Iterating a 0-d array is numpy's own TypeError; without this the
    /// `__getitem__` fallback protocol would stop at the first IndexError and
    /// look like an empty iterator.
    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.borrow().arr.ndim() == 0 {
            return Err(PyTypeError::new_err("iteration over a 0-d array"));
        }
        // A lazy sequence iterator over `__getitem__`, which is what CPython
        // would have used had we not defined `__iter__` at all.
        // SAFETY: `slf` is a live object; PySeqIter_New borrows it and
        // returns a new strong reference (or NULL with an exception set).
        let it = unsafe { ffi::PySeqIter_New(slf.as_ptr()) };
        if it.is_null() {
            return Err(PyErr::fetch(py));
        }
        // SAFETY: `it` is a new strong reference we now own.
        Ok(unsafe { Py::from_owned_ptr(py, it) })
    }

    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let me = slf.borrow().arr.clone();
        // `a[i]` on a 1-d numeric array is the single most common indexing
        // expression there is, and going through the general machinery for it
        // costs a `Vec<IndexItem>`, a view `NdArray` (two more `Vec`s and an
        // `Arc` bump) and a dtype-name lookup, all to read eight bytes. A
        // bare `int` on a 1-d simple-dtype array skips straight to the read.
        if me.ndim() == 1
            && !me.dtype().is_flexible()
            && !me.dtype().is_object()
            && key.is_exact_instance_of::<pyo3::types::PyInt>()
        {
            if let Ok(i) = key.extract::<isize>() {
                let n = me.shape[0];
                let v = if i < 0 { i + n } else { i };
                if v < 0 || v >= n {
                    return Err(PyIndexError::new_err(format!(
                        "index {} is out of bounds for axis 0 with size {}",
                        i, n
                    )));
                }
                let off = me.byte_offset + v * me.strides[0];
                return npscalar_to_py(py, me.dtype(), me.read_at(off));
            }
        }
        // Structured field access: `a['f0']` / `a[['f0','f2']]`.
        if let Some(out) = crate::fields::getitem(slf, key)? {
            return Ok(out);
        }
        let items = crate::index::parse_index(key)?;
        match rnp_core::indexing::index(&me, &items).map_err(crate::err)? {
            Indexed::View { arr: view, scalarize } => {
                if scalarize {
                    if crate::fields::is_struct_element(view.descr) {
                        return crate::fields::struct_scalar(py, view, slf);
                    }
                    if view.dtype().is_flexible() || view.dtype().is_object() {
                        return npflexible_to_py(py, &view, view.byte_offset);
                    }
                    return npscalar_to_py(py, view.dtype(), view.get(&[]).map_err(crate::err)?);
                }
                Ok(PyNdArray::view_of(view, slf)?.into_bound(py).into_any())
            }
            Indexed::Fancy(plan) => {
                let out = rnp_core::indexing::gather(&me, &plan).map_err(crate::err)?;
                if plan.scalarize {
                    if crate::fields::is_struct_element(out.descr) {
                        let off = out.byte_offset;
                        return crate::fields::struct_scalar_owned(py, &out, off);
                    }
                    if out.dtype().is_flexible() || out.dtype().is_object() {
                        return npflexible_to_py(py, &out, out.byte_offset);
                    }
                    return npscalar_to_py(py, out.dtype(), out.get_flat(0));
                }
                Ok(PyNdArray::into_py_any(out, py)?.into_bound(py).into_any())
            }
        }
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // Borrow immutably and clone the header: the buffer is shared, and a
        // mutable borrow would deadlock when the key or the value *is* this
        // same array (`a[a > 0] = a`).
        let me = slf.borrow().arr.clone();
        if !me.flags.writeable {
            return Err(PyValueError::new_err(
                "assignment destination is read-only",
            ));
        }
        // Structured field assignment: `a['f0'] = v` / `a[['f0','f1']] = v`.
        if crate::fields::setitem(slf, key, value)? {
            return Ok(());
        }
        let items = crate::index::parse_index(key)?;
        match rnp_core::indexing::index(&me, &items).map_err(crate::err)? {
            Indexed::View { arr: mut view, .. } => {
                if view.dtype().is_object() {
                    let src = crate::objects::array_from_objects(value)?;
                    let src = if src.shape == view.shape {
                        src
                    } else {
                        rnp_core::iter::broadcast_to(&src, &view.shape).map_err(crate::err)?
                    };
                    let dst: Vec<isize> = rnp_core::iter::offsets(
                        &view.shape, &view.strides, view.byte_offset).collect();
                    let so: Vec<isize> = rnp_core::iter::offsets(
                        &src.shape, &src.strides, src.byte_offset).collect();
                    for (&d, &s) in dst.iter().zip(so.iter()) {
                        view.write_at(d, src.read_at(s));
                    }
                    return Ok(());
                }
                if !view.dtype().is_flexible() {
                    if let Some(s) = scalar_from_py(value) {
                        view.fill(s.cast(view.dtype()));
                        return Ok(());
                    }
                }
                let src = assignment_source(value, &view.shape, view.dtype())?;
                let src = src.in_order_of(&view);
                let dst_offsets: Vec<isize> =
                    rnp_core::iter::offsets(&view.shape, &view.strides, view.byte_offset).collect();
                let src_offsets: Vec<isize> =
                    rnp_core::iter::offsets(&src.shape, &src.strides, src.byte_offset).collect();
                if view.dtype().is_flexible() {
                    for (&d, &s) in dst_offsets.iter().zip(src_offsets.iter()) {
                        view.write_raw_at(d, src.raw_bytes_at(s));
                    }
                } else {
                    for (&d, &s) in dst_offsets.iter().zip(src_offsets.iter()) {
                        view.write_at(d, src.read_at(s));
                    }
                }
                Ok(())
            }
            Indexed::Fancy(plan) => {
                let src = assignment_source(value, &plan.shape, me.dtype())?;
                rnp_core::indexing::scatter(&me, &plan, &src).map_err(crate::err)
            }
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if self.arr.dtype().is_object() {
            return Ok(format!("array({}, dtype=object)",
                              object_body(py, &self.arr, &mut Vec::new())?));
        }
        Ok(printing::repr(&self.arr))
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        if self.arr.dtype().is_object() {
            return object_body(py, &self.arr, &mut Vec::new());
        }
        Ok(printing::to_str(&self.arr))
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

    /// `operator.index()`: numpy accepts only a 0-d array of an integer
    /// dtype — not bool, not a one-element 1-d array — and raises this exact
    /// TypeError for everything else.
    fn __index__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.arr.ndim() != 0 || !self.arr.dtype().is_integer() {
            return Err(PyTypeError::new_err(
                "only integer scalar arrays can be converted to a scalar index",
            ));
        }
        match self.arr.get_flat(0) {
            Scalar::Int(i) => Ok(i.into_pyobject(py)?.into_any().unbind()),
            Scalar::Uint(u) => Ok(u.into_pyobject(py)?.into_any().unbind()),
            _ => Err(PyTypeError::new_err(
                "only integer scalar arrays can be converted to a scalar index",
            )),
        }
    }

    // ---- operators -----------------------------------------------------

    fn __matmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        crate::linalgops::matmul_operator(py, &self.arr, other, false)
    }
    fn __rmatmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        crate::linalgops::matmul_operator(py, &self.arr, other, true)
    }

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
    fn __floordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::FloorDiv, false)
    }
    fn __rfloordiv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::FloorDiv, true)
    }
    fn __mod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Mod, false)
    }
    fn __rmod__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::Mod, true)
    }
    fn __pow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if modulo.is_some_and(|m| !m.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "pow() with a modulus is not supported for arrays",
            ));
        }
        self.binop(py, other, BinOp::Pow, false)
    }
    fn __rpow__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = modulo;
        self.binop(py, other, BinOp::Pow, true)
    }
    fn __and__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::BitAnd, false)
    }
    fn __rand__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::BitAnd, true)
    }
    fn __or__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::BitOr, false)
    }
    fn __ror__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::BitOr, true)
    }
    fn __xor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::BitXor, false)
    }
    fn __rxor__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::BitXor, true)
    }
    fn __lshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::LShift, false)
    }
    fn __rlshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::LShift, true)
    }
    fn __rshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::RShift, false)
    }
    fn __rrshift__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binop(py, other, BinOp::RShift, true)
    }
    fn __divmod__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::divmod_pair(slf, other, false)
    }
    fn __rdivmod__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::divmod_pair(slf, other, true)
    }

    // ---- unary operators -----------------------------------------------

    fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.unop(py, rnp_core::UnOp::Negative)
    }
    fn __pos__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.unop(py, rnp_core::UnOp::Positive)
    }
    fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.unop(py, rnp_core::UnOp::Absolute)
    }
    fn __invert__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.unop(py, rnp_core::UnOp::Invert)
    }

    // ---- in-place operators ---------------------------------------------

    fn __imatmul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        crate::linalgops::imatmul(slf, other)
    }

    fn __iadd__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::Add)
    }
    fn __isub__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::Sub)
    }
    fn __imul__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::Mul)
    }
    fn __itruediv__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::Div)
    }
    fn __ifloordiv__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::FloorDiv)
    }
    fn __imod__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::Mod)
    }
    fn __ipow__(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let _ = modulo;
        Self::inplace(slf, other, BinOp::Pow)
    }
    fn __iand__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::BitAnd)
    }
    fn __ior__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::BitOr)
    }
    fn __ixor__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::BitXor)
    }
    fn __ilshift__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::LShift)
    }
    fn __irshift__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::inplace(slf, other, BinOp::RShift)
    }

    fn __richcmp__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        // Structured/void comparison has its own rules; see `fields.rs`.
        crate::fields::check_comparable(slf, other)?;
        let this = slf.borrow();
        let bop = match op {
            CompareOp::Eq => BinOp::Eq,
            CompareOp::Ne => BinOp::Ne,
            CompareOp::Lt => BinOp::Lt,
            CompareOp::Le => BinOp::Le,
            CompareOp::Gt => BinOp::Gt,
            CompareOp::Ge => BinOp::Ge,
        };
        this.binop(py, other, bop, false)
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
            format: CString::new(arr.descr.buffer_format()).unwrap(),
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

/// numpy prints an object array by `repr`-ing each element, so the body has
/// to be built on the Python side.
fn object_body(py: Python<'_>, arr: &NdArray, index: &mut Vec<isize>) -> PyResult<String> {
    if index.len() == arr.ndim() {
        let o = crate::objects::read(py, arr, arr.byte_index(index));
        return Ok(o.repr()?.to_string());
    }
    let n = arr.shape[index.len()];
    let mut parts = Vec::with_capacity(n.max(0) as usize);
    for i in 0..n {
        index.push(i);
        parts.push(object_body(py, arr, index)?);
        index.pop();
    }
    Ok(format!("[{}]", parts.join(", ")))
}

/// Return `res`, or copy it into a caller-supplied `out=` array (casting to
/// the output dtype, as numpy does).
pub fn store_or_wrap<'py>(
    py: Python<'py>,
    res: NdArray,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let dest = match out {
        None => return Ok(PyNdArray::into_py_any(res, py)?.into_bound(py).into_any()),
        Some(o) if o.is_none() => {
            return Ok(PyNdArray::into_py_any(res, py)?.into_bound(py).into_any())
        }
        Some(o) => o,
    };
    let cell = dest.cast::<PyNdArray>().map_err(|_| {
        PyTypeError::new_err("return arrays must be of ArrayType")
    })?;
    let target = cell.borrow().arr.clone();
    if target.shape != res.shape {
        return Err(PyValueError::new_err(format!(
            "could not broadcast input array from shape {} into shape {}",
            fmt_shape(&res.shape),
            fmt_shape(&target.shape)
        )));
    }
    let res = res.in_order_of(&target);
    let src: Vec<isize> =
        rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset).collect();
    let dst: Vec<isize> =
        rnp_core::iter::offsets(&target.shape, &target.strides, target.byte_offset).collect();
    for (&s, &d) in src.iter().zip(dst.iter()) {
        if target.dtype().is_flexible() {
            target.write_raw_at(d, res.raw_bytes_at(s));
        } else {
            target.write_at(d, res.read_at(s));
        }
    }
    Ok(dest.clone())
}

/// Coerce the right-hand side of an assignment: build an array of `dtype`
/// and broadcast it to `shape`, with numpy's message when it does not fit.
fn assignment_source(
    value: &Bound<'_, PyAny>,
    shape: &[isize],
    dtype: DType,
) -> PyResult<NdArray> {
    let src = array_from_any(value, Some(dtype), false)?;
    rnp_core::iter::broadcast_to(&src, shape).map_err(|_| {
        PyValueError::new_err(format!(
            "shape mismatch: value array of shape {} could not be broadcast \
             to indexing result of shape {}",
            fmt_shape(&src.shape),
            fmt_shape(shape)
        ))
    })
}

fn fmt_shape(s: &[isize]) -> String {
    if s.len() == 1 {
        format!("({},)", s[0])
    } else {
        format!(
            "({})",
            s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")
        )
    }
}

/// Normalise an `axis=` argument: `None` means "reduce everything".
fn resolve_axis(arr: &NdArray, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Option<usize>> {
    let a = match axis {
        None => return Ok(None),
        Some(o) if o.is_none() => return Ok(None),
        Some(o) => o,
    };
    let i: isize = a.extract().map_err(|_| {
        PyNotImplementedError::new_err("only a single integer axis= is implemented yet")
    })?;
    let nd = arr.ndim() as isize;
    let i = if i < 0 { i + nd } else { i };
    if i < 0 || i >= nd {
        return Err(PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            i, nd
        )));
    }
    Ok(Some(i as usize))
}

fn as_i64(s: Scalar) -> i64 {
    match s {
        Scalar::Int(i) => i,
        Scalar::Uint(u) => u as i64,
        Scalar::Bool(b) => b as i64,
        Scalar::Float(f) => f as i64,
        Scalar::Complex(c) => c.re as i64,
    }
}

impl PyNdArray {
    /// `all()`/`any()` over the whole array or one axis.
    fn bool_reduce<'py>(
        &self,
        py: Python<'py>,
        want_all: bool,
        axis: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let truthy = |s: Scalar| -> bool {
            match s {
                Scalar::Bool(b) => b,
                Scalar::Int(i) => i != 0,
                Scalar::Uint(u) => u != 0,
                Scalar::Float(f) => f != 0.0,
                Scalar::Complex(c) => c.re != 0.0 || c.im != 0.0,
            }
        };
        match resolve_axis(&self.arr, axis)? {
            None => {
                let mut acc = want_all;
                for o in rnp_core::iter::offsets(&self.arr.shape, &self.arr.strides,
                                                 self.arr.byte_offset) {
                    let t = truthy(self.arr.read_at(o));
                    if want_all {
                        acc &= t;
                        if !acc {
                            break;
                        }
                    } else {
                        acc |= t;
                        if acc {
                            break;
                        }
                    }
                }
                if keepdims {
                    let mut a = NdArray::zeros(vec![1; self.arr.ndim()], DType::Bool)
                        .map_err(crate::err)?;
                    a.fill(Scalar::Bool(acc));
                    return store_or_wrap(py, a, out);
                }
                npscalar_to_py(py, DType::Bool, Scalar::Bool(acc))
            }
            Some(ax) => {
                let bools = self.arr.astype(DType::Bool);
                let op = if want_all {
                    rnp_core::ReduceOp::Min
                } else {
                    rnp_core::ReduceOp::Max
                };
                let res = rnp_core::reduce_axis(&bools, ax, op, keepdims).map_err(crate::err)?;
                store_or_wrap(py, res, out)
            }
        }
    }

    /// The shared body of every reduction method.
    fn reduce<'py>(
        &self,
        py: Python<'py>,
        op: ReduceOp,
        axis: Option<&Bound<'py, PyAny>>,
        dtype: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if out.is_some_and(|o| !o.is_none()) {
            return Err(PyNotImplementedError::new_err(
                "reductions with out= are not implemented yet",
            ));
        }
        // `dtype=` casts the operand first, which is what numpy does for the
        // accumulation type of sum/prod.
        let owned;
        let src = match dtype {
            Some(d) if !d.is_none() => {
                let dt = dtype_from_any(d)?;
                owned = self.arr.astype(dt);
                &owned
            }
            _ => &self.arr,
        };
        match resolve_axis(src, axis)? {
            None => {
                let v = reduce_all(src, op).map_err(crate::err)?;
                if keepdims {
                    let shape = vec![1isize; src.ndim()];
                    let mut a =
                        NdArray::zeros(shape, reduce_dtype(op, src.dtype())).map_err(crate::err)?;
                    a.fill(v);
                    return Ok(PyNdArray::into_py_any(a, py)?.into_bound(py).into_any());
                }
                npscalar_to_py(py, reduce_dtype(op, src.dtype()), v)
            }
            Some(ax) => {
                let a = reduce_axis(src, ax, op, keepdims).map_err(crate::err)?;
                Ok(PyNdArray::into_py_any(a, py)?.into_bound(py).into_any())
            }
        }
    }

    /// The real (`k == 0`) or imaginary (`k == 1`) component view.
    fn component(slf: &Bound<'_, Self>, k: isize) -> PyResult<Py<PyNdArray>> {
        let a = slf.borrow().arr.clone();
        if !a.dtype().is_complex() {
            if k == 0 {
                return PyNdArray::view_of(a, slf);
            }
            // numpy's `.imag` on a real array is a read-only zero array.
            let z = NdArray::zeros(a.shape.clone(), a.dtype()).map_err(crate::err)?;
            return PyNdArray::into_py_any(z, slf.py());
        }
        let comp = if a.dtype() == DType::C64 { DType::F32 } else { DType::F64 };
        let mut v = a.clone();
        // The component view keeps the parent's byte order: a `'>c16'` array
        // has `'>f8'` halves.
        v.descr = rnp_core::descr::Descr::new(comp, a.descr.bo);
        v.byte_offset += k * comp.itemsize() as isize;
        v.update_flags();
        PyNdArray::view_of(v, slf)
    }

    fn binop(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: BinOp,
        reflected: bool,
    ) -> PyResult<Py<PyAny>> {
        let rhs = match operand_for(other, self.arr.dtype(), op.is_comparison())? {
            Some(a) => a,
            None => return Ok(py.NotImplemented()),
        };
        let (a, b) = if reflected {
            (&rhs, &self.arr)
        } else {
            (&self.arr, &rhs)
        };
        rnp_core::fpe::clear();
        let out = binary(a, b, op).map_err(crate::err)?;
        crate::ufuncs::report_fpe(py, op.name())?;
        Ok(PyNdArray::into_py_any(out, py)?.into_any())
    }

    fn unop(&self, py: Python<'_>, op: rnp_core::UnOp) -> PyResult<Py<PyAny>> {
        rnp_core::fpe::clear();
        let out = rnp_core::unary(&self.arr, op).map_err(crate::err)?;
        crate::ufuncs::report_fpe(py, op.name())?;
        Ok(PyNdArray::into_py_any(out, py)?.into_any())
    }

    /// `divmod(a, b)` on arrays: one pass, two outputs.
    fn divmod_pair<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        reflected: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let me = slf.borrow().arr.clone();
        let rhs = match operand_for(other, me.dtype(), false)? {
            Some(a) => a,
            None => return Ok(py.NotImplemented().into_bound(py)),
        };
        let (a, b) = if reflected { (&rhs, &me) } else { (&me, &rhs) };
        rnp_core::fpe::clear();
        let (q, r) = rnp_core::divmod(a, b).map_err(crate::err)?;
        crate::ufuncs::report_fpe(py, "divmod")?;
        let q = PyNdArray::into_py_any(q, py)?;
        let r = PyNdArray::into_py_any(r, py)?;
        Ok(PyTuple::new(py, [q.into_any(), r.into_any()])?.into_any())
    }

    /// `a op= b`, writing the result back into `a`'s own buffer with numpy's
    /// same-kind cast.
    fn inplace(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>, op: BinOp) -> PyResult<()> {
        let py = slf.py();
        let me = slf.borrow().arr.clone();
        if !me.flags.writeable {
            return Err(PyValueError::new_err(
                "output array is read-only",
            ));
        }
        let rhs = operand_for(other, me.dtype(), false)?
            .ok_or_else(|| PyTypeError::new_err("unsupported operand for in-place op"))?;
        rnp_core::fpe::clear();
        let res = binary(&me, &rhs, op).map_err(crate::err)?;
        crate::ufuncs::report_fpe(py, op.name())?;
        if res.shape != me.shape {
            return Err(PyValueError::new_err(format!(
                "non-broadcastable output operand with shape {} doesn't match                  the broadcast shape {}",
                fmt_shape(&me.shape),
                fmt_shape(&res.shape)
            )));
        }
        let src: Vec<isize> =
            rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset).collect();
        let dst: Vec<isize> =
            rnp_core::iter::offsets(&me.shape, &me.strides, me.byte_offset).collect();
        for (&s, &d) in src.iter().zip(dst.iter()) {
            me.write_at(d, res.read_at(s));
        }
        Ok(())
    }
}

/// Accept `reshape(2, 3)` and `reshape((2, 3))` alike.
pub fn shape_from_args(args: &Bound<'_, PyTuple>) -> PyResult<Vec<isize>> {
    if args.len() == 1 {
        return shape_from_any(&args.get_item(0)?);
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
    // An integer array is a valid axis/shape sequence too
    // (`a.transpose(np.random.permutation(5))`).
    if let Ok(a) = obj.cast::<PyNdArray>() {
        let arr = a.borrow().arr.clone();
        if arr.dtype().is_integer() && arr.ndim() <= 1 {
            return Ok(arr.to_vec().into_iter().map(as_i64).map(|v| v as isize).collect());
        }
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
        let rhs = operand_for(b, lhs.dtype(), op.is_comparison())?
            .ok_or_else(|| PyTypeError::new_err("unsupported operand for ufunc"))?;
        (lhs, rhs)
    } else if b_is_arr && !a_is_arr {
        let rhs = array_from_any(b, None, false)?;
        let lhs = operand_for(a, rhs.dtype(), op.is_comparison())?
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

/// As [`dtype_or_default`], keeping the byte order and C-type alias.
pub fn descr_or_default(
    obj: Option<&Bound<'_, PyAny>>,
    default: DType,
) -> PyResult<rnp_core::descr::Descr> {
    match obj {
        None => Ok(rnp_core::descr::Descr::native(default)),
        Some(o) if o.is_none() => Ok(rnp_core::descr::Descr::native(default)),
        Some(o) => descr_from_any(o),
    }
}
