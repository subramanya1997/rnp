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

use crate::convert::{array_from_any_descr, element_to_py, npflexible_to_py};
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

/// `astype` where either side is a structured dtype.
///
/// Probed from numpy 2.5.2:
/// * structured -> structured is matched up **by position**, not by name, and
///   the two dtypes must have the *same number of fields*
///   (`np.zeros(2,'i4,f8').astype([('x','f8'),('y','i4')])` swaps the types
///   over, and adding a third field is a `TypeError`);
/// * unstructured -> structured broadcasts the source into *every* field;
/// * structured -> unstructured unwraps a single field, recursively, and is
///   refused when there is more than one field.
pub fn struct_astype(py: Python<'_>, arr: &NdArray, to: Descr) -> PyResult<NdArray> {
    let from = arr.descr;
    let out = NdArray::zeros_descr(arr.shape.clone(), to).map_err(crate::err)?;
    let src_offsets: Vec<isize> =
        rnp_core::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset).collect();
    let dst_offsets: Vec<isize> =
        rnp_core::iter::offsets(&out.shape, &out.strides, out.byte_offset).collect();
    for (&src_off, &dst_off) in src_offsets.iter().zip(dst_offsets.iter()) {
        transfer_value(py, arr, src_off, from, &out, dst_off, to, from, to)?;
    }
    Ok(out)
}

/// Transfer one record-local value. This mirrors numpy's VOID transfer stack:
/// structured fields are paired by position, a scalar is broadcast into all
/// destination fields, and subarrays are handled inside each record rather
/// than by comparing the expanded ndarray field-view shapes.
#[allow(clippy::too_many_arguments)]
fn transfer_value(
    py: Python<'_>,
    src: &NdArray,
    src_off: isize,
    from: Descr,
    dst: &NdArray,
    dst_off: isize,
    to: Descr,
    whole_from: Descr,
    whole_to: Descr,
) -> PyResult<()> {
    match (from.struct_def(), to.struct_def()) {
        (Some(sd), Some(dd)) => {
            if sd.fields.len() != dd.fields.len() {
                return Err(cannot_cast(whole_from, whole_to));
            }
            for (sf, df) in sd.fields.iter().zip(dd.fields.iter()) {
                transfer_value(
                    py,
                    src,
                    src_off + sf.offset as isize,
                    sf.descr,
                    dst,
                    dst_off + df.offset as isize,
                    df.descr,
                    whole_from,
                    whole_to,
                )?;
            }
            return Ok(());
        }
        (Some(sd), None) => {
            if sd.fields.len() != 1 {
                return Err(cannot_cast(whole_from, whole_to));
            }
            let field = &sd.fields[0];
            return transfer_value(
                py,
                src,
                src_off + field.offset as isize,
                field.descr,
                dst,
                dst_off,
                to,
                whole_from,
                whole_to,
            );
        }
        (None, Some(dd)) => {
            for field in &dd.fields {
                transfer_value(
                    py,
                    src,
                    src_off,
                    from,
                    dst,
                    dst_off + field.offset as isize,
                    field.descr,
                    whole_from,
                    whole_to,
                )?;
            }
            return Ok(());
        }
        (None, None) => {}
    }

    let from_sub = from.subarray_def();
    let to_sub = to.subarray_def();
    if from_sub.is_some() || to_sub.is_some() {
        let (from_base, from_shape): (Descr, &[isize]) = match from_sub.as_deref() {
            Some(sub) => (sub.base, &sub.shape),
            None => (from, &[]),
        };
        let (to_base, to_shape): (Descr, &[isize]) = match to_sub.as_deref() {
            Some(sub) => (sub.base, &sub.shape),
            None => (to, &[]),
        };
        let from_size = from_shape.iter().product::<isize>().max(1) as usize;
        let to_size = to_shape.iter().product::<isize>().max(1) as usize;
        for dst_index in 0..to_size {
            let src_index = subarray_source_index(dst_index, from_shape, to_shape);
            let Some(src_index) = src_index else {
                // `out` starts zeroed, which is numpy's zero-padding rule.
                continue;
            };
            debug_assert!(src_index < from_size);
            transfer_value(
                py,
                src,
                src_off + (src_index * from_base.itemsize()) as isize,
                from_base,
                dst,
                dst_off + (dst_index * to_base.itemsize()) as isize,
                to_base,
                whole_from,
                whole_to,
            )?;
        }
        return Ok(());
    }

    cast_leaf(
        py, src, src_off, from, dst, dst_off, to, whole_from, whole_to,
    )
}

/// Map a flat destination subarray index to numpy's source index. Missing
/// leading dimensions select coordinate zero; short source dimensions are
/// truncated and out-of-range destination coordinates are zero-filled.
fn subarray_source_index(
    dst_flat: usize,
    src_shape: &[isize],
    dst_shape: &[isize],
) -> Option<usize> {
    let src_size = src_shape.iter().product::<isize>().max(1) as usize;
    let dst_size = dst_shape.iter().product::<isize>().max(1) as usize;
    if (src_size == 1 && dst_size == 1) || src_shape == dst_shape {
        return Some(dst_flat);
    }
    if src_size == 1 {
        return Some(0);
    }

    let ndim = src_shape.len().max(dst_shape.len());
    let mut dst_index = dst_flat;
    let mut src_index = 0usize;
    let mut src_factor = 1usize;
    for i in (0..ndim).rev() {
        let mut coord = 0usize;
        if i >= ndim - dst_shape.len() {
            let shape = dst_shape[i - (ndim - dst_shape.len())] as usize;
            coord = dst_index % shape;
            dst_index /= shape;
        }
        if i >= ndim - src_shape.len() {
            let shape = src_shape[i - (ndim - src_shape.len())] as usize;
            if shape == 1 {
                // A length-one source dimension broadcasts coordinate zero.
            } else if coord < shape {
                src_index += src_factor * coord;
                src_factor *= shape;
            } else {
                return None;
            }
        }
    }
    Some(src_index)
}

fn scalar_view(owner: &NdArray, byte_offset: isize, descr: Descr) -> NdArray {
    let mut view = owner.clone();
    view.byte_offset = byte_offset;
    view.shape.clear();
    view.strides.clear();
    view.descr = descr;
    view.flags.owndata = false;
    view
}

#[allow(clippy::too_many_arguments)]
fn cast_leaf(
    py: Python<'_>,
    src: &NdArray,
    src_off: isize,
    from: Descr,
    dst: &NdArray,
    dst_off: isize,
    to: Descr,
    _whole_from: Descr,
    _whole_to: Descr,
) -> PyResult<()> {
    let src_view = scalar_view(src, src_off, from);
    let dst_view = scalar_view(dst, dst_off, to);
    if from == to {
        dst_view.write_raw_at(dst_off, src_view.raw_bytes_at(src_off));
        return Ok(());
    }
    if from.dt.is_flexible() || to.dt.is_flexible() {
        // Flexible values live outside the core `Scalar` enum. Round-trip one
        // leaf through Python, just as NumPy's text transfer loops do. Keep a
        // numpy scalar wrapper for a text source: CPython then produces the
        // exact public errors (`np.str_('x')` / `np.bytes_(b'x')`) when its
        // numeric constructors reject the value.
        let value = if from.dt.is_flexible() {
            npflexible_to_py(py, &src_view, src_off)?
        } else {
            element_to_py(py, &src_view, src_off)?
        };
        let value = if from.dt.is_flexible() {
            let constructor = match to.dt.kind() {
                'b' => Some("bool"),
                'i' | 'u' => Some("int"),
                'f' => Some("float"),
                'c' => Some("complex"),
                _ => None,
            };
            match constructor {
                Some(name) => py.import("builtins")?.getattr(name)?.call1((value,))?,
                None => value,
            }
        } else {
            value
        };
        let cast = array_from_any_descr(&value, Some(to), false)?;
        dst_view.write_raw_at(dst_off, cast.raw_bytes_at(cast.byte_offset));
        return Ok(());
    }
    let cast = src_view.astype_descr(to);
    dst_view.write_raw_at(dst_off, cast.raw_bytes_at(cast.byte_offset));
    Ok(())
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
    if offset as usize + d.itemsize() > arr.itemsize() {
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
