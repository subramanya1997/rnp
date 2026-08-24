//! Structured-array field access: `a['f0']`, `a[['f0','f2']]`, the matching
//! assignments, and the `np.void` scalar an element of a structured array
//! indexes to.
//!
//! Every behaviour here was probed directly against real numpy 2.5.2:
//!
//! * `a['f0']` is a **view**: the field's byte offset is folded into the
//!   array's `byte_offset` and the field's descriptor replaces the array's,
//!   so writes through the view show up in the parent. A *subarray* field
//!   splices its shape onto the array's shape.
//! * `a[['f0','f2']]` is **also a view** in numpy 2.x (it used to be a copy).
//!   The result keeps the *parent's* itemsize and the fields' *original*
//!   offsets, is writeable, and its `.base` is the parent array:
//!   `np.zeros(4,'i4,f8,u1')[['f0','f2']].dtype` is
//!   `{'names':['f0','f2'], 'formats':['<i4','u1'], 'offsets':[0,12],
//!    'itemsize':13}`.
//! * A single element of a structured array is an `np.void` that writes
//!   *through* to the parent (`a[1]['f0'] = 7` changes `a`).

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};

use rnp_core::descr::{select_fields, Descr};
use rnp_core::NdArray;

use crate::pyarray::PyNdArray;

/// numpy's message for a key that is not a valid array index at all.
pub const NOT_AN_INDEX: &str = "only integers, slices (`:`), ellipsis (`...`), \
                                numpy.newaxis (`None`) and integer or boolean \
                                arrays are valid indices";

/// What a `__getitem__`/`__setitem__` key turned out to be, as far as field
/// access is concerned.
pub enum FieldKey {
    /// A single field name (a bare `str` key).
    One(String),
    /// A non-empty list of field names.
    Many(Vec<String>),
    /// Not a field key at all — fall through to the normal index machinery.
    NotAField,
}

/// Classify a key.
///
/// A bare `str` is always a field key (on a non-structured array numpy raises
/// `IndexError`, which the callers below produce). A `list` counts only when
/// it is non-empty and *every* element is a `str`: `a[[]]` and `a[[0, 1]]` are
/// ordinary fancy indexing, and `a[['f0', 1]]` is numpy's `IndexError`.
pub fn classify(key: &Bound<'_, PyAny>) -> PyResult<FieldKey> {
    if let Ok(s) = key.cast::<PyString>() {
        return Ok(FieldKey::One(s.to_cow()?.into_owned()));
    }
    let list = match key.cast::<PyList>() {
        Ok(l) => l,
        Err(_) => return Ok(FieldKey::NotAField),
    };
    let n = list.len();
    if n == 0 {
        return Ok(FieldKey::NotAField);
    }
    // numpy only treats a list as a field selection when the *first* item is
    // a string; a mixed list is then an error rather than a fancy index.
    if !list.get_item(0)?.is_instance_of::<PyString>() {
        return Ok(FieldKey::NotAField);
    }
    let mut names = Vec::with_capacity(n);
    for item in list.iter() {
        match item.cast::<PyString>() {
            Ok(s) => names.push(s.to_cow()?.into_owned()),
            Err(_) => return Err(PyIndexError::new_err(NOT_AN_INDEX)),
        }
    }
    Ok(FieldKey::Many(names))
}

/// `a['name']` as an `NdArray` view of `arr`.
pub fn one_field(arr: &NdArray, name: &str) -> PyResult<NdArray> {
    if !arr.descr.is_struct() {
        return Err(PyIndexError::new_err(NOT_AN_INDEX));
    }
    match arr.descr.field(name) {
        Some((d, off)) => Ok(arr.field_view(d, off)),
        None => Err(PyValueError::new_err(format!("no field of name {name}"))),
    }
}

/// `a[['a','b']]` as an `NdArray` view of `arr`.
pub fn many_fields(arr: &NdArray, names: &[String]) -> PyResult<NdArray> {
    if !arr.descr.is_struct() {
        return Err(PyIndexError::new_err(NOT_AN_INDEX));
    }
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for n in names {
        if !seen.insert(n.as_str()) {
            return Err(PyValueError::new_err(format!(
                "duplicate field of name '{n}'"
            )));
        }
        if arr.descr.field(n).is_none() {
            // numpy raises a *KeyError* here, not the ValueError a bare
            // string key gets.
            return Err(PyKeyError::new_err(n.clone()));
        }
    }
    let sel = select_fields(arr.descr, names)
        .ok_or_else(|| PyIndexError::new_err(NOT_AN_INDEX))?;
    Ok(arr.with_descr_same_itemsize(sel))
}

/// The `__getitem__` hook. `Ok(None)` means "not a field key, carry on".
pub fn getitem<'py>(
    slf: &Bound<'py, PyNdArray>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = slf.py();
    let arr = slf.borrow().arr.clone();
    let view = match classify(key)? {
        FieldKey::One(name) => one_field(&arr, &name)?,
        FieldKey::Many(names) => many_fields(&arr, &names)?,
        FieldKey::NotAField => return Ok(None),
    };
    Ok(Some(PyNdArray::view_of(view, slf)?.into_bound(py).into_any()))
}

/// The `__setitem__` hook. `Ok(false)` means "not a field key, carry on".
pub fn setitem(
    slf: &Bound<'_, PyNdArray>,
    key: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let py = slf.py();
    let arr = slf.borrow().arr.clone();
    match classify(key)? {
        FieldKey::One(name) => {
            if !arr.flags.writeable {
                return Err(PyValueError::new_err("assignment destination is read-only"));
            }
            let view = one_field(&arr, &name)?;
            assign_into(py, slf, view, value)?;
            Ok(true)
        }
        FieldKey::Many(names) => {
            if !arr.flags.writeable {
                return Err(PyValueError::new_err("assignment destination is read-only"));
            }
            // Assigning to a multi-field selection writes each selected field
            // in turn, so a plain tuple `a[['f0','f2']] = (5, 6)` fills each
            // field with the matching component -- exactly what numpy does.
            let sub = many_fields(&arr, &names)?;
            let holder = PyNdArray::view_of(sub, slf)?.into_bound(py);
            assign_struct(py, &holder, value, &names)?;
            Ok(true)
        }
        FieldKey::NotAField => Ok(false),
    }
}

/// Assign `value` into an existing field view, going back through the array's
/// own `__setitem__` with an all-encompassing `Ellipsis` key so that
/// broadcasting, casting and every value form are handled by exactly the same
/// code path as `view[...] = value`.
fn assign_into(
    py: Python<'_>,
    _parent: &Bound<'_, PyNdArray>,
    view: NdArray,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let holder = Py::new(py, PyNdArray::wrap(view))?.into_bound(py);
    let ell = py.Ellipsis();
    holder.set_item(ell, value)
}

/// `a[['f0','f2']] = value`, field by field.
fn assign_struct(
    py: Python<'_>,
    dest: &Bound<'_, PyNdArray>,
    value: &Bound<'_, PyAny>,
    names: &[String],
) -> PyResult<()> {
    let arr = dest.borrow().arr.clone();
    // A structured *array* source is matched up by position, as numpy does.
    if let Ok(src) = value.cast::<PyNdArray>() {
        let sdescr = src.borrow().arr.descr;
        if let Some(src_names) = sdescr.field_names() {
            if src_names.len() != names.len() {
                return Err(PyValueError::new_err(format!(
                    "could not broadcast input array from shape ({},) into shape ({},)",
                    src_names.len(),
                    names.len()
                )));
            }
            for (dst_name, src_name) in names.iter().zip(src_names.iter()) {
                let v = one_field(&arr, dst_name)?;
                let sv = src.get_item(PyString::new(py, src_name))?;
                assign_into(py, dest, v, &sv)?;
            }
            return Ok(());
        }
    }
    // A tuple/list source distributes its components across the fields.
    if let Ok(t) = value.cast::<PyTuple>() {
        if t.len() == names.len() {
            for (name, v) in names.iter().zip(t.iter()) {
                let view = one_field(&arr, name)?;
                assign_into(py, dest, view, &v)?;
            }
            return Ok(());
        }
    }
    // Anything else broadcasts into every selected field.
    for name in names {
        assign_one(py, dest, &arr, name, value)?;
    }
    Ok(())
}

/// Assign `value` into a single named field, recursing into a *nested*
/// structured field so that `a[['f0','f1']] = 0` reaches its leaves -- which
/// is what numpy does; a bare scalar is not a "tuple of field values", so the
/// straight assignment would be refused.
fn assign_one(
    py: Python<'_>,
    dest: &Bound<'_, PyNdArray>,
    arr: &NdArray,
    name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let view = one_field(arr, name)?;
    if view.descr.is_struct() && value.cast::<PyTuple>().is_err() {
        let inner = view.descr.field_names().unwrap_or_default();
        let holder = Py::new(py, PyNdArray::wrap(view.clone()))?.into_bound(py);
        for sub in &inner {
            assign_one(py, &holder, &view, sub, value)?;
        }
        return Ok(());
    }
    assign_into(py, dest, view, value)
}

/// numpy refuses to compare a structured/void array with anything that is not
/// also void. Probed on 2.5.2: `np.zeros(2,'i4,f8') == 1` and
/// `np.zeros(2,'V4') == np.zeros(2,'i4')` are both `TypeError`, for every
/// comparison operator, while `V == V` compares bytes and structured vs
/// unstructured void gets its own (numpy-internal-sounding) message.
pub fn check_comparable(slf: &Bound<'_, PyNdArray>, other: &Bound<'_, PyAny>) -> PyResult<()> {
    use rnp_core::dtype::Kind;
    let a = slf.borrow().arr.descr;
    if a.dt.category() != Kind::Void {
        return Ok(());
    }
    let b = other
        .cast::<PyNdArray>()
        .ok()
        .map(|o| o.borrow().arr.descr)
        .filter(|d| d.dt.category() == Kind::Void);
    match b {
        None => Err(PyTypeError::new_err(
            "Cannot compare structured or void to non-void arrays.",
        )),
        Some(b) if a.is_struct() != b.is_struct() => Err(PyTypeError::new_err(
            "Cannot compare structured with unstructured void arrays. \
             (unreachable error, please report to NumPy devs.)",
        )),
        Some(_) => Ok(()),
    }
}

/// numpy's message when a cast is refused outright.
fn cannot_cast(from: Descr, to: Descr) -> PyErr {
    PyTypeError::new_err(format!(
        "Cannot cast array data from {} to {} according to the rule 'unsafe'",
        from.repr_string(),
        to.repr_string()
    ))
}

/// Copy `src` into `dst` element by element; both must already share a dtype
/// and a shape.
fn copy_into(dst: &NdArray, src: &NdArray) {
    let d: Vec<isize> =
        rnp_core::iter::offsets(&dst.shape, &dst.strides, dst.byte_offset).collect();
    let s: Vec<isize> =
        rnp_core::iter::offsets(&src.shape, &src.strides, src.byte_offset).collect();
    for (&a, &b) in d.iter().zip(s.iter()) {
        dst.write_raw_at(a, src.raw_bytes_at(b));
    }
}

/// `astype` where either side is a structured dtype.
///
/// Probed from numpy 2.5.2:
/// * structured -> structured is matched up **by position**, not by name, and
///   the two dtypes must have the *same number of fields*
///   (`np.zeros(2,'i4,f8').astype([('x','f8'),('y','i4')])` swaps the types
///   over, and adding a third field is a `TypeError`);
/// * unstructured -> structured broadcasts the source into *every* field;
/// * structured -> unstructured is refused.
pub fn struct_astype(arr: &NdArray, to: Descr) -> PyResult<NdArray> {
    let from = arr.descr;
    let out = NdArray::zeros_descr(arr.shape.clone(), to).map_err(crate::err)?;
    match (from.struct_def(), to.struct_def()) {
        (Some(sd), Some(dd)) => {
            if sd.fields.len() != dd.fields.len() {
                return Err(cannot_cast(from, to));
            }
            for (sf, df) in sd.fields.iter().zip(dd.fields.iter()) {
                let sv = arr.field_view(sf.descr, sf.offset);
                let dv = out.field_view(df.descr, df.offset);
                if sv.shape != dv.shape {
                    return Err(cannot_cast(from, to));
                }
                let cast = cast_leaf(&sv, dv.descr, from, to)?;
                copy_into(&dv, &cast);
            }
        }
        (None, Some(dd)) => {
            for df in dd.fields.iter() {
                let dv = out.field_view(df.descr, df.offset);
                let src = if dv.shape == arr.shape {
                    arr.clone()
                } else {
                    rnp_core::iter::broadcast_to(arr, &dv.shape).map_err(crate::err)?
                };
                let cast = cast_leaf(&src, dv.descr, from, to)?;
                copy_into(&dv, &cast);
            }
        }
        _ => return Err(cannot_cast(from, to)),
    }
    Ok(out)
}

/// Cast one field's worth of data, recursing for nested structured fields.
fn cast_leaf(src: &NdArray, to: Descr, whole_from: Descr, whole_to: Descr) -> PyResult<NdArray> {
    if src.descr == to {
        return Ok(src.copy());
    }
    if src.descr.is_struct() || to.is_struct() {
        return struct_astype(src, to);
    }
    if src.descr.dt.is_flexible() != to.dt.is_flexible() {
        return Err(cannot_cast(whole_from, whole_to));
    }
    if src.descr.dt.is_flexible() {
        // S/U/V leaves: the engine has no value cast for these yet.
        if src.descr.dt != to.dt {
            return Err(cannot_cast(whole_from, whole_to));
        }
        return Ok(src.copy());
    }
    Ok(src.astype_descr(to))
}

/// `a.getfield(dtype, offset)`: reinterpret each item at a byte offset.
///
/// numpy keeps the shape and strides and only changes the descriptor and the
/// starting offset, so the result is a writeable view of the same buffer.
pub fn getfield<'py>(
    slf: &Bound<'py, PyNdArray>,
    dtype: &Bound<'py, PyAny>,
    offset: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let arr = slf.borrow().arr.clone();
    let d = crate::pydtype::descr_from_any(dtype)?;
    if offset < 0 {
        return Err(PyValueError::new_err("offset is negative"));
    }
    // `array_getfield` reports two *different* messages, and which one it
    // picks does not depend on the offset at all: a new type wider than the
    // original is rejected on its own terms first, and only a new type that
    // *would* fit at offset 0 gets the "plus offset" wording. Probed on an
    // `int32` array: `getfield('i8', 0)`, `('i8', 4)` and `('i8', 8)` all say
    // "new type is larger than original type", while `('i4', 4)`, `('i2', 3)`
    // and `('i1', 4)` say "new type plus offset is larger than original type".
    if arr.itemsize() < d.itemsize() {
        return Err(PyValueError::new_err(
            "new type is larger than original type",
        ));
    }
    if offset as usize > arr.itemsize() - d.itemsize() {
        return Err(PyValueError::new_err(
            "new type plus offset is larger than original type",
        ));
    }
    let view = arr.field_view(d, offset as usize);
    Ok(PyNdArray::view_of(view, slf)?.into_bound(py).into_any())
}

/// `a.setfield(value, dtype, offset)`: the assignment counterpart.
pub fn setfield(
    slf: &Bound<'_, PyNdArray>,
    value: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
    offset: isize,
) -> PyResult<()> {
    let py = slf.py();
    let target = getfield(slf, dtype, offset)?;
    target.set_item(py.Ellipsis(), value)
}

/// Build the `np.void` scalar for a single element of a structured array.
///
/// The scalar is backed by a genuine 0-d *view* of the element, so field
/// access, `setfield` and `v['f0'] = x` all write through to the parent —
/// which is what numpy does.
pub fn struct_scalar<'py>(
    py: Python<'py>,
    view: NdArray,
    parent: &Bound<'py, PyNdArray>,
) -> PyResult<Bound<'py, PyAny>> {
    let zero = NdArray {
        buffer: view.buffer.clone(),
        byte_offset: view.byte_offset,
        shape: Vec::new(),
        strides: Vec::new(),
        descr: view.descr,
        flags: rnp_core::array::Flags {
            owndata: false,
            ..view.flags
        },
    };
    let holder = PyNdArray::view_of(zero, parent)?.into_bound(py);
    match crate::pydtype::scalar_class(py, view.dtype()) {
        Some(cls) => cls.call_method1(pyo3::intern!(py, "_from_array"), (holder,)),
        None => Ok(holder.into_any()),
    }
}

/// As [`struct_scalar`], for an element of an array that has no Python-side
/// parent object (a freshly gathered result).
pub fn struct_scalar_owned<'py>(
    py: Python<'py>,
    view: &NdArray,
    off: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let zero = NdArray {
        buffer: view.buffer.clone(),
        byte_offset: off,
        shape: Vec::new(),
        strides: Vec::new(),
        descr: view.descr,
        flags: rnp_core::array::Flags {
            owndata: false,
            ..view.flags
        },
    };
    let holder = Py::new(py, PyNdArray::wrap(zero))?.into_bound(py);
    match crate::pydtype::scalar_class(py, view.dtype()) {
        Some(cls) => cls.call_method1(pyo3::intern!(py, "_from_array"), (holder,)),
        None => Ok(holder.into_any()),
    }
}

/// True when an element of this dtype should surface as a structured
/// `np.void` rather than a bytes-backed one.
pub fn is_struct_element(d: Descr) -> bool {
    d.is_struct() || d.subarray_def().is_some()
}
