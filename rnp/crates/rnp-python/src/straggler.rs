//! The straggler cluster: `lexsort`, and the `ndarray` methods that were
//! still missing after M5 — `tobytes`, `clip`, `resize`, `conj`/`conjugate`,
//! `std`/`var`, `dump`/`dumps`, `getfield`/`setfield` — plus pickle support
//! (`__reduce__` / `__setstate__` and the `_reconstruct` entry point).
//!
//! Every contract here was probed against real numpy 2.5.2; the surprising
//! ones are called out at their implementation.

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBytes, PyList, PyTuple};

use rnp_core::descr::Descr;
use rnp_core::{BinOp, DType, NdArray, Scalar};

use crate::convert::{array_from_any, operand_for};
use crate::pyarray::{store_or_wrap, PyNdArray};
use crate::pydtype::{descr_from_any, PyDType};

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// numpy raises `np.exceptions.AxisError` (a `ValueError`/`IndexError`
/// subclass) for an out-of-range axis, and the tests assert that exact class.
/// The engine only carries a message, so build the shim's class directly and
/// fall back to `ValueError` if the shim is not importable.
fn axis_err(axis: isize, ndim: usize) -> PyErr {
    let msg = format!("axis {axis} is out of bounds for array of dimension {ndim}");
    Python::attach(|py| {
        match py
            .import("rnp_numpy.exceptions")
            .and_then(|m| m.getattr("AxisError"))
            .and_then(|c| c.call1((axis, ndim)))
        {
            Ok(exc) => PyErr::from_value(exc),
            Err(_) => PyValueError::new_err(msg),
        }
    })
}

/// Validate `axis` against `arr` without computing anything, so callers that
/// have their own reduction path still raise numpy's `AxisError`.
pub fn check_axis(arr: &NdArray, axis: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    resolve_axis_pub(arr, axis).map(|_| ())
}

/// Normalise an axis against `ndim`, with numpy's 0-d special case: a 0-d
/// array behaves as if it had one axis, so `0` and `-1` are both accepted.
fn norm_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let n = ndim.max(1) as isize;
    let a = if axis < 0 { axis + n } else { axis };
    if a < 0 || a >= n {
        return Err(axis_err(axis, ndim));
    }
    Ok(a as usize)
}

/// A fresh C-contiguous array holding `arr`'s elements in the order that
/// `order` selects along `axis` (a dtype-agnostic `take_along_axis`).
fn permute_along_axis(arr: &NdArray, order: &[Vec<usize>], axis: usize) -> NdArray {
    let out = NdArray::zeros_descr(arr.shape.clone(), arr.descr).expect("permute alloc");
    for (lane_idx, lane) in lane_offsets(arr, axis).into_iter().enumerate() {
        let out_lane = lane_offsets(&out, axis);
        let dst = &out_lane[lane_idx];
        for (k, &src) in order[lane_idx].iter().enumerate() {
            out.write_raw_at(dst[k], arr.raw_bytes_at(lane[src]));
        }
    }
    out
}

/// The byte offsets of every 1-D lane along `axis`. Mirrors `sort.rs::lanes`,
/// which is private to `rnp-core`.
fn lane_offsets(arr: &NdArray, axis: usize) -> Vec<Vec<isize>> {
    if arr.ndim() == 0 {
        return vec![vec![arr.byte_offset]];
    }
    let n = arr.shape[axis].max(0);
    let step = arr.strides[axis];
    let mut shape = arr.shape.clone();
    let mut strides = arr.strides.clone();
    shape.remove(axis);
    strides.remove(axis);
    rnp_core::iter::offsets(&shape, &strides, arr.byte_offset)
        .map(|base| (0..n).map(|k| base + k * step).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// lexsort
// ---------------------------------------------------------------------------

/// One stable argsort pass over `arr` along `axis`, returned as per-lane
/// permutations. Object arrays sort through Python's own comparison; every
/// other dtype delegates to `rnp_core::sort::argsort`, which is stable.
fn stable_order(py: Python<'_>, arr: &NdArray, axis: usize) -> PyResult<Vec<Vec<usize>>> {
    if arr.dtype().is_object() {
        return object_order(py, arr, axis);
    }
    let idx = rnp_core::sort::argsort(arr, axis, true).map_err(crate::err)?;
    Ok(lane_offsets(&idx, axis)
        .into_iter()
        .map(|lane| {
            lane.into_iter()
                .map(|o| match idx.read_at(o) {
                    Scalar::Int(i) => i as usize,
                    Scalar::Uint(u) => u as usize,
                    s => s.as_f64() as usize,
                })
                .collect()
        })
        .collect())
}

/// Stable argsort of an object array using Python's `<`.
fn object_order(py: Python<'_>, arr: &NdArray, axis: usize) -> PyResult<Vec<Vec<usize>>> {
    let mut out = Vec::new();
    for lane in lane_offsets(arr, axis) {
        let objs: Vec<Bound<'_, PyAny>> = lane
            .iter()
            .map(|&o| crate::objects::read(py, arr, o))
            .collect();
        let mut order: Vec<usize> = (0..objs.len()).collect();
        // `sort_by` cannot carry a `Result`, so the first comparison error is
        // parked and re-raised once the sort finishes.
        let mut failure: Option<PyErr> = None;
        order.sort_by(|&i, &j| {
            if failure.is_some() {
                return std::cmp::Ordering::Equal;
            }
            match objs[i].lt(&objs[j]) {
                Ok(true) => std::cmp::Ordering::Less,
                Ok(false) => match objs[j].lt(&objs[i]) {
                    Ok(true) => std::cmp::Ordering::Greater,
                    Ok(false) => std::cmp::Ordering::Equal,
                    Err(e) => {
                        failure = Some(e);
                        std::cmp::Ordering::Equal
                    }
                },
                Err(e) => {
                    failure = Some(e);
                    std::cmp::Ordering::Equal
                }
            }
        });
        if let Some(e) = failure {
            return Err(e);
        }
        out.push(order);
    }
    Ok(out)
}

/// `np.lexsort(keys, axis=-1)`.
///
/// Probed facts: the keys are **not** broadcast — mismatched shapes raise
/// `ValueError("all keys need to be the same shape")`; an empty (or
/// non-sequence) `keys` raises `TypeError`; 0-d keys are legal and yield a
/// 0-d result; and the last key is the primary one.
#[pyfunction]
#[pyo3(signature = (keys, axis = -1))]
pub fn lexsort<'py>(
    py: Python<'py>,
    keys: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'py, PyAny>> = match keys.try_iter() {
        Ok(it) => it.collect::<PyResult<Vec<_>>>()?,
        Err(_) => {
            return Err(PyTypeError::new_err(
                "need sequence of keys with len > 0 in lexsort",
            ))
        }
    };
    if items.is_empty() {
        return Err(PyTypeError::new_err(
            "need sequence of keys with len > 0 in lexsort",
        ));
    }
    let mut arrays = Vec::with_capacity(items.len());
    for it in &items {
        arrays.push(array_from_any(it, None, false)?);
    }
    let shape = arrays[0].shape.clone();
    if arrays.iter().any(|a| a.shape != shape) {
        return Err(PyValueError::new_err("all keys need to be the same shape"));
    }
    let ndim = shape.len();
    let ax = norm_axis(axis, ndim)?;

    // Least-significant-digit pass: sort by the first key, then stably
    // re-sort by each later key, so the last key ends up dominant.
    let n_lanes = lane_offsets(&arrays[0], if ndim == 0 { 0 } else { ax }).len();
    let lane_len = if ndim == 0 { 1 } else { shape[ax].max(0) as usize };
    let mut order: Vec<Vec<usize>> = vec![(0..lane_len).collect(); n_lanes];

    for key in &arrays {
        let permuted = if ndim == 0 {
            key.clone()
        } else {
            permute_along_axis(key, &order, ax)
        };
        let pass = stable_order(py, &permuted, if ndim == 0 { 0 } else { ax })?;
        for (lane, p) in order.iter_mut().zip(pass.iter()) {
            *lane = p.iter().map(|&i| lane[i]).collect();
        }
    }

    let out = NdArray::zeros(shape.clone(), DType::I64).map_err(crate::err)?;
    if ndim == 0 {
        return crate::convert::npscalar_to_py(py, DType::I64, Scalar::Int(0));
    }
    for (lane, off) in lane_offsets(&out, ax).into_iter().zip(order.iter()) {
        for (k, &v) in off.iter().enumerate() {
            out.write_at(lane[k], Scalar::Int(v as i64));
        }
    }
    Ok(PyNdArray::into_py_any(out, py)?.into_bound(py).into_any())
}

// ---------------------------------------------------------------------------
// tobytes
// ---------------------------------------------------------------------------

/// `a.tobytes(order)` / `a.tostring(order)`.
///
/// `'A'` and `'K'` mean "F if the array is F-contiguous (and not C), else C",
/// which is what numpy's `NPY_ANYORDER`/`NPY_KEEPORDER` collapse to for a
/// flat byte dump.
pub fn tobytes_impl<'py>(
    py: Python<'py>,
    arr: &NdArray,
    order: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    let want_f = match order.unwrap_or("C") {
        "C" | "c" => false,
        "F" | "f" => true,
        "A" | "a" | "K" | "k" => arr.flags.f_contiguous && !arr.flags.c_contiguous,
        other => {
            return Err(PyValueError::new_err(format!(
                "order must be one of 'C', 'F', 'A', or 'K' (got '{other}')"
            )))
        }
    };
    let isz = arr.itemsize();
    let mut buf = Vec::with_capacity(arr.size() * isz);
    // F order == C order of the transpose, so one offset walk covers both.
    let src = if want_f { arr.transpose() } else { arr.clone() };
    for off in rnp_core::iter::offsets(&src.shape, &src.strides, src.byte_offset) {
        buf.extend_from_slice(arr.raw_bytes_at(off));
    }
    Ok(PyBytes::new(py, &buf))
}

// ---------------------------------------------------------------------------
// clip
// ---------------------------------------------------------------------------

/// The inclusive integer range a dtype can hold, for clamping out-of-range
/// Python-int bounds (see `clip_impl`).
fn int_range(dt: DType) -> Option<(f64, f64)> {
    Some(match dt {
        DType::Bool => (0.0, 1.0),
        DType::I8 => (i8::MIN as f64, i8::MAX as f64),
        DType::I16 => (i16::MIN as f64, i16::MAX as f64),
        DType::I32 => (i32::MIN as f64, i32::MAX as f64),
        DType::I64 => (i64::MIN as f64, i64::MAX as f64),
        DType::U8 => (0.0, u8::MAX as f64),
        DType::U16 => (0.0, u16::MAX as f64),
        DType::U32 => (0.0, u32::MAX as f64),
        DType::U64 => (0.0, u64::MAX as f64),
        _ => return None,
    })
}

/// Coerce one clip bound. A Python `int` that does not fit the array's dtype
/// is clamped to that dtype's range rather than raising: numpy's `clip`
/// accepts `uint8_arr.clip(-1, 300)` and returns the array unchanged, and
/// clamping the bound to `[0, 255]` reproduces that exactly.
fn clip_bound(obj: &Bound<'_, PyAny>, dt: DType) -> PyResult<NdArray> {
    match operand_for(obj, dt, false) {
        Ok(Some(a)) => Ok(a),
        Ok(None) => Err(PyTypeError::new_err("unsupported operand for clip")),
        Err(e) => {
            let is_int = obj.is_instance_of::<pyo3::types::PyInt>()
                && !obj.is_instance_of::<pyo3::types::PyBool>();
            let Some((lo, hi)) = int_range(dt) else {
                return Err(e);
            };
            if !is_int {
                return Err(e);
            }
            let v: f64 = match obj.extract::<i128>() {
                Ok(i) => i as f64,
                Err(_) => {
                    // Beyond i128: only the sign matters for the clamp.
                    if obj.lt(0i64).unwrap_or(false) {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                }
            };
            let clamped = Scalar::Float(v.clamp(lo, hi)).cast(dt);
            let mut a = NdArray::zeros(vec![], dt).map_err(crate::err)?;
            a.set(&[], clamped).map_err(crate::err)?;
            Ok(a)
        }
    }
}

/// `a.clip(min=None, max=None, out=None, **kwargs)`.
///
/// Probed fact: in numpy 2.5.2 both bounds being `None` does **not** raise —
/// it returns a plain copy. (`casting=` is accepted and ignored, as the only
/// thing it changes upstream is whether a warning is emitted.)
pub fn clip_impl<'py>(
    py: Python<'py>,
    arr: &NdArray,
    min: Option<&Bound<'py, PyAny>>,
    max: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let lo = min.filter(|b| !b.is_none());
    let hi = max.filter(|b| !b.is_none());
    let mut res = arr.clone();
    let mut dt = arr.dtype();
    // First pass fixes the result dtype (weak Python scalars adopt the
    // array's dtype, numpy scalars and arrays promote against it).
    for b in [lo, hi].into_iter().flatten() {
        dt = rnp_core::promote(dt, clip_bound(b, arr.dtype())?.dtype());
    }
    if dt != arr.dtype() || !arr.is_native() {
        res = arr.astype(dt);
    }
    if let Some(b) = lo {
        let bound = clip_bound(b, dt)?;
        res = rnp_core::binary(&res, &bound, BinOp::Maximum).map_err(crate::err)?;
    }
    if let Some(b) = hi {
        let bound = clip_bound(b, dt)?;
        res = rnp_core::binary(&res, &bound, BinOp::Minimum).map_err(crate::err)?;
    }
    if lo.is_none() && hi.is_none() {
        res = res.copy();
    }
    store_or_wrap(py, res, out)
}

// ---------------------------------------------------------------------------
// conjugate
// ---------------------------------------------------------------------------

/// `a.conj()` / `a.conjugate([out])`.
///
/// Probed facts: for a real numeric dtype numpy returns `self` itself (the
/// tests assert `a.conj() is a`); flexible and datetime dtypes raise
/// `TypeError("cannot conjugate non-numeric dtype")`; object arrays call
/// `.conjugate()` on every element.
pub fn conjugate_impl<'py>(
    slf: &Bound<'py, PyNdArray>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let arr = slf.borrow().arr.clone();
    let dt = arr.dtype();
    if arr.descr.is_struct() || dt.is_flexible() || matches!(dt, DType::DateTime(_) | DType::TimeDelta(_)) {
        return Err(PyTypeError::new_err(format!(
            "cannot conjugate non-numeric dtype {}",
            arr.descr.name()
        )));
    }
    if dt.is_object() {
        let res = NdArray::zeros_descr(arr.shape.clone(), arr.descr).map_err(crate::err)?;
        let src: Vec<isize> =
            rnp_core::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset).collect();
        let dst: Vec<isize> =
            rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset).collect();
        for (&s, &d) in src.iter().zip(dst.iter()) {
            let o = crate::objects::read(py, &arr, s);
            let c = o.call_method0("conjugate")?;
            crate::objects::write(&res, d, &c);
        }
        return store_or_wrap(py, res, out);
    }
    if !matches!(dt, DType::C64 | DType::C128) {
        // numpy hands back the very same object for real numeric dtypes.
        if out.is_none_or(|o| o.is_none()) {
            return Ok(slf.clone().into_any());
        }
        return store_or_wrap(py, arr, out);
    }
    let res = NdArray::zeros_descr(arr.shape.clone(), arr.descr).map_err(crate::err)?;
    let src: Vec<isize> =
        rnp_core::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset).collect();
    let dst: Vec<isize> =
        rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset).collect();
    for (&s, &d) in src.iter().zip(dst.iter()) {
        let v = match arr.read_at(s) {
            Scalar::Complex(c) => Scalar::Complex(num_complex::Complex::new(c.re, -c.im)),
            other => other,
        };
        res.write_at(d, v);
    }
    store_or_wrap(py, res, out)
}

// ---------------------------------------------------------------------------
// pickle
// ---------------------------------------------------------------------------

/// numpy hands `__setstate__` a `bytes` buffer as the array's `base` once the
/// payload is big enough to be worth sharing; below the threshold it copies.
/// Probed: the switch happens above 1000 bytes.
const SETSTATE_BASE_THRESHOLD: usize = 1000;

/// `numpy._core.multiarray._reconstruct(subtype, shape, dtype)` — builds the
/// empty shell that `__setstate__` then fills.
#[pyfunction]
#[pyo3(signature = (subtype, shape, dtype = None))]
pub fn _reconstruct<'py>(
    py: Python<'py>,
    subtype: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = subtype;
    let want = crate::pyarray::shape_from_any(shape)?;
    let d = match dtype {
        Some(o) if !o.is_none() => {
            // The pickle payload carries the one-byte type code `b'b'`.
            match o.extract::<Vec<u8>>() {
                Ok(bytes) if bytes.len() == 1 => {
                    descr_from_any((bytes[0] as char).to_string().into_pyobject(py)?.as_any())?
                }
                _ => descr_from_any(o)?,
            }
        }
        _ => Descr::native(DType::U8),
    };
    let arr = NdArray::zeros_descr(want, d).map_err(crate::err)?;
    Ok(PyNdArray::into_py_any(arr, py)?.into_bound(py).into_any())
}

/// The 5-tuple numpy stores as an array's pickle state:
/// `(version, shape, dtype, is_fortran, data)`, where `data` is the raw
/// C-order (or F-order) bytes, or a list of objects for `object` dtype.
pub fn reduce_state<'py>(py: Python<'py>, arr: &NdArray) -> PyResult<Bound<'py, PyTuple>> {
    let is_f = arr.flags.f_contiguous && !arr.flags.c_contiguous && arr.ndim() > 1;
    let data: Py<PyAny> = if arr.dtype().is_object() {
        let items = PyList::empty(py);
        let src = if is_f { arr.transpose() } else { arr.clone() };
        for off in rnp_core::iter::offsets(&src.shape, &src.strides, src.byte_offset) {
            items.append(crate::objects::read(py, arr, off))?;
        }
        items.into_any().unbind()
    } else {
        tobytes_impl(py, arr, Some(if is_f { "F" } else { "C" }))?
            .into_any()
            .unbind()
    };
    PyTuple::new(
        py,
        [
            1i64.into_pyobject(py)?.into_any().unbind(),
            PyTuple::new(py, arr.shape.iter().copied())?
                .into_any()
                .unbind(),
            Py::new(py, PyDType::from_descr(arr.descr))?.into_any(),
            is_f.into_pyobject(py)?.to_owned().into_any().unbind(),
            data,
        ],
    )
}

/// `a.__reduce__()` — `(_reconstruct, (ndarray, (0,), b'b'), state)`.
pub fn reduce_impl<'py>(slf: &Bound<'py, PyNdArray>) -> PyResult<Bound<'py, PyTuple>> {
    let py = slf.py();
    // `_rnp._reconstruct`, not the shim's `rnp_numpy._core.multiarray`
    // spelling: pickle has to resolve the name on load, and only the
    // extension's own entry point understands the `b'b'` type code numpy
    // stores in the reconstructor arguments.
    let recon = py.import("_rnp")?.getattr("_reconstruct")?;
    let args = PyTuple::new(
        py,
        [
            slf.get_type().into_any().unbind(),
            PyTuple::new(py, [0i64])?.into_any().unbind(),
            PyBytes::new(py, b"b").into_any().unbind(),
        ],
    )?;
    let state = reduce_state(py, &slf.borrow().arr)?;
    PyTuple::new(
        py,
        [
            recon.unbind(),
            args.into_any().unbind(),
            state.into_any().unbind(),
        ],
    )
}

/// `a.__setstate__(state)` — rebuild the array in place from the 5-tuple.
pub fn setstate_impl(slf: &Bound<'_, PyNdArray>, state: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = slf.py();
    let t = state
        .cast::<PyTuple>()
        .map_err(|_| PyValueError::new_err("invalid pickle state"))?;
    // numpy accepted a 4-tuple in its very first pickle version; every state
    // this port writes is a 5-tuple.
    let (shape_i, dtype_i, fortran_i, data_i) = match t.len() {
        5 => (1, 2, 3, 4),
        4 => (0, 1, 2, 3),
        _ => return Err(PyValueError::new_err("invalid pickle state")),
    };
    let shape = crate::pyarray::shape_from_any(&t.get_item(shape_i)?)?;
    let descr = descr_from_any(&t.get_item(dtype_i)?)?;
    let is_f: bool = t.get_item(fortran_i)?.extract().unwrap_or(false);
    let data = t.get_item(data_i)?;

    let mut arr = NdArray::zeros_descr(shape.clone(), descr).map_err(crate::err)?;
    if is_f && arr.ndim() > 1 {
        arr = arr.transpose().copy().transpose();
        arr.descr = descr;
    }
    let mut base: Option<Py<PyAny>> = None;
    if descr.dt.is_object() {
        let items = data.cast::<PyList>().map_err(|_| {
            PyValueError::new_err("object pickle state must be a list")
        })?;
        let walk = if is_f { arr.transpose() } else { arr.clone() };
        let offs: Vec<isize> =
            rnp_core::iter::offsets(&walk.shape, &walk.strides, walk.byte_offset).collect();
        for (k, &off) in offs.iter().enumerate() {
            crate::objects::write(&arr, off, &items.get_item(k)?);
        }
    } else {
        let bytes: Vec<u8> = match data.extract::<Vec<u8>>() {
            Ok(b) => b,
            Err(_) => data.str()?.to_string().into_bytes(),
        };
        if bytes.len() != arr.nbytes() {
            return Err(PyValueError::new_err(
                "buffer size does not match array size",
            ));
        }
        let walk = if is_f { arr.transpose() } else { arr.clone() };
        let isz = arr.itemsize();
        for (k, off) in
            rnp_core::iter::offsets(&walk.shape, &walk.strides, walk.byte_offset).enumerate()
        {
            arr.write_raw_at(off, &bytes[k * isz..(k + 1) * isz]);
        }
        if bytes.len() > SETSTATE_BASE_THRESHOLD && data.is_instance_of::<PyBytes>() {
            base = Some(data.clone().unbind());
        }
    }
    arr.update_flags();
    let mut me = slf.borrow_mut();
    me.arr = arr;
    me.base = base;
    let _ = py;
    Ok(())
}

// ---------------------------------------------------------------------------
// resize
// ---------------------------------------------------------------------------

/// `a.resize(*shape, refcheck=True)`.
///
/// Probed facts: there is **no** `order=` keyword (passing one is a
/// `TypeError`); the ownership and refcount checks only run when the total
/// byte count actually changes, so `view.resize(same_total)` succeeds; and
/// growing zero-fills the tail.
pub fn resize_impl(
    slf: &Bound<'_, PyNdArray>,
    shape: Vec<isize>,
    refcheck: bool,
) -> PyResult<()> {
    let py = slf.py();
    if shape.iter().any(|&d| d < 0) {
        return Err(PyValueError::new_err("negative dimensions not allowed"));
    }
    let arr = slf.borrow().arr.clone();
    let newsize = rnp_core::array::shape_size(&shape);
    let isz = arr.itemsize();
    let newbytes = newsize * isz;

    if newbytes != arr.nbytes() {
        // numpy tests the OWNDATA flag here. This port's `array()` leaves
        // OWNDATA false on freshly built arrays (see the note in the report),
        // so the reliable signal for "this is a view" is a non-empty base
        // chain, which `PyNdArray::view_of` maintains.
        if slf.borrow().base.is_some() {
            return Err(PyValueError::new_err(
                "cannot resize this array: it does not own its data",
            ));
        }
        if refcheck && referenced(slf)? {
            return Err(PyValueError::new_err(
                "cannot resize an array that references or is referenced\n\
                 by another object in this way.\n\
                 Use the np.resize function to get a new resized copy or\n \
                 set refcheck=False to disable this check",
            ));
        }
    }
    // Rebuild rather than realloc: the engine's buffers are `Arc`-shared, so
    // growing in place is not an option. The kept prefix is the array's own
    // C-order element sequence, and the tail is left zeroed.
    let out = NdArray::zeros_descr(shape, arr.descr).map_err(crate::err)?;
    let keep = arr.size().min(newsize);
    let src: Vec<isize> =
        rnp_core::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset).collect();
    for k in 0..keep {
        out.write_raw_at(out.byte_offset + (k * isz) as isize, arr.raw_bytes_at(src[k]));
    }
    let mut me = slf.borrow_mut();
    me.arr = out;
    me.base = None;
    let _ = py;
    Ok(())
}

/// numpy refuses to resize an array that anything else still points at. It
/// tests the CPython refcount and the weakref list; the calling convention
/// here adds a fixed number of temporary references, calibrated against the
/// upstream tests (`x.resize(...)` alone must succeed, `y = x` must not).
fn referenced(slf: &Bound<'_, PyNdArray>) -> PyResult<bool> {
    let py = slf.py();
    let wr = py.import("weakref")?;
    let n: usize = wr.call_method1("getweakrefcount", (slf,))?.extract()?;
    if n > 0 {
        return Ok(true);
    }
    Ok(slf.get_refcnt() > RESIZE_BASE_REFCNT)
}

/// The refcount a lone `x.resize(...)` call site produces; anything above it
/// means a second name (or container) also holds the array. Measured at 2 for
/// `x.resize(...)` and 3 once `y = x` exists — the same threshold numpy's own
/// C check uses.
const RESIZE_BASE_REFCNT: isize = 2;

// ---------------------------------------------------------------------------
// var / std
// ---------------------------------------------------------------------------

/// Transcribed from `numpy/_core/_methods.py::_var`: the mean is taken in the
/// accumulator dtype, the deviations are squared through `x * conj(x)` for
/// complex operands (so the result is real), and the final division is by
/// `max(count - ddof, 0)`.
#[allow(clippy::too_many_arguments)]
pub fn var_impl<'py>(
    py: Python<'py>,
    arr: &NdArray,
    axis: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    ddof: f64,
    keepdims: bool,
    where_: Option<&Bound<'py, PyAny>>,
    sqrt: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let dt = arr.dtype();
    if dt.is_object() || dt.is_flexible() || arr.descr.is_struct() {
        return Err(PyTypeError::new_err(format!(
            "cannot perform var with type {}",
            arr.descr.name()
        )));
    }
    // Accumulator dtype: an explicit `dtype=` wins, bool/int accumulate in
    // float64, float16 in float32, everything else keeps its own type.
    let acc = match dtype {
        Some(d) if !d.is_none() => crate::pydtype::dtype_from_any(d)?,
        _ => match dt {
            DType::F16 => DType::F32,
            DType::F32 => DType::F32,
            DType::C64 => DType::C64,
            DType::C128 => DType::C128,
            DType::F64 => DType::F64,
            _ => DType::F64,
        },
    };
    // `var`/`std` of a complex *operand* are real-valued, but an explicit
    // complex `dtype=` still wins: numpy sums the (real) squared deviations
    // with that dtype, so `np.var(float_mat, dtype='F')` is complex64.
    let explicit = dtype.is_some_and(|d| !d.is_none());
    let res_dt = if explicit {
        acc
    } else {
        match acc {
            DType::C64 => DType::F32,
            DType::C128 => DType::F64,
            d => d,
        }
    };
    // The squared deviations themselves are always real.
    let sq_dt = match res_dt {
        DType::C64 => DType::F32,
        DType::C128 => DType::F64,
        d => d,
    };
    let out_dt = if dt == DType::F16 && !explicit {
        DType::F16
    } else {
        res_dt
    };

    let ax = resolve_axis_pub(arr, axis)?;
    let mask = match where_ {
        Some(w) if !w.is_none() => {
            let m = array_from_any(w, Some(DType::Bool), false)?;
            Some(rnp_core::iter::broadcast_to(&m, &arr.shape).map_err(crate::err)?)
        }
        _ => None,
    };

    let promoted = arr.astype(acc);
    let count = counts(arr, ax, mask.as_ref());
    let mean = masked_sum(&promoted, ax, mask.as_ref(), true)?;
    let mean = divide_by(&mean, &count, acc);

    // x = arr - mean, broadcast back over the reduced axis.
    let diff = rnp_core::binary(&promoted, &mean, BinOp::Sub).map_err(crate::err)?;
    let sq = NdArray::zeros(diff.shape.clone(), sq_dt).map_err(crate::err)?;
    let d_off: Vec<isize> =
        rnp_core::iter::offsets(&diff.shape, &diff.strides, diff.byte_offset).collect();
    let s_off: Vec<isize> =
        rnp_core::iter::offsets(&sq.shape, &sq.strides, sq.byte_offset).collect();
    for (&s, &d) in d_off.iter().zip(s_off.iter()) {
        let v = match diff.read_at(s) {
            Scalar::Complex(c) => Scalar::Float(c.re * c.re + c.im * c.im),
            other => {
                let f = other.as_f64();
                Scalar::Float(f * f)
            }
        };
        sq.write_at(d, v.cast(sq_dt));
    }

    let total = masked_sum(&sq.astype(res_dt), ax, mask.as_ref(), keepdims)?;
    let denom = counts(arr, ax, mask.as_ref());
    let denom = if keepdims || ax.is_none() {
        denom
    } else {
        drop_axis(&denom, ax.unwrap())
    };
    let denom = sub_clamped(&denom, ddof);
    if any_nonpositive(&denom) {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            std::ffi::CString::new("Degrees of freedom <= 0 for slice")
                .unwrap()
                .as_c_str(),
            1,
        )?;
    }
    let mut res = divide_by(&total, &denom, res_dt);
    if sqrt {
        let r2 = NdArray::zeros(res.shape.clone(), res_dt).map_err(crate::err)?;
        let a_off: Vec<isize> =
            rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset).collect();
        let b_off: Vec<isize> =
            rnp_core::iter::offsets(&r2.shape, &r2.strides, r2.byte_offset).collect();
        for (&a, &b) in a_off.iter().zip(b_off.iter()) {
            r2.write_at(b, Scalar::Float(res.read_at(a).as_f64().sqrt()).cast(res_dt));
        }
        res = r2;
    }
    if out_dt != res_dt {
        res = res.astype(out_dt);
    }
    if ax.is_none() && !keepdims && out.is_none_or(|o| o.is_none()) {
        let v = res.get_flat(0);
        return crate::convert::npscalar_to_py(py, out_dt, v);
    }
    store_or_wrap(py, res, out)
}

/// `a.mean(...)` for the two argument shapes `pyarray.rs`'s own `mean` does
/// not cover: an `out=` destination and a `where=` mask. The accumulator
/// rules are the same ones `_methods._mean` uses.
pub fn mean_impl<'py>(
    py: Python<'py>,
    arr: &NdArray,
    axis: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
    where_: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let dt = arr.dtype();
    let is_half = dt == DType::F16;
    let acc = match dtype {
        Some(d) if !d.is_none() => crate::pydtype::dtype_from_any(d)?,
        _ if is_half => DType::F32,
        _ => rnp_core::reduce::mean_dtype(dt),
    };
    let out_dt = if is_half && dtype.is_none_or(|d| d.is_none()) {
        DType::F16
    } else {
        acc
    };
    let ax = resolve_axis_pub(arr, axis)?;
    let mask = match where_ {
        Some(w) if !w.is_none() => {
            let m = array_from_any(w, Some(DType::Bool), false)?;
            Some(rnp_core::iter::broadcast_to(&m, &arr.shape).map_err(crate::err)?)
        }
        _ => None,
    };
    let promoted = arr.astype(acc);
    let total = masked_sum(&promoted, ax, mask.as_ref(), keepdims)?;
    let mut count = counts(arr, ax, mask.as_ref());
    if !keepdims && ax.is_some() {
        count = drop_axis(&count, ax.unwrap());
    }
    if any_nonpositive(&count) {
        warn_empty_mean(py)?;
    }
    let mut res = divide_by(&total, &count, acc);
    if out_dt != acc {
        res = res.astype(out_dt);
    }
    if ax.is_none() && !keepdims && out.is_none_or(|o| o.is_none()) {
        let v = res.get_flat(0);
        return crate::convert::npscalar_to_py(py, out_dt, v);
    }
    store_or_wrap(py, res, out)
}

/// `resolve_axis` lives in `pyarray.rs` and is private there; this is the
/// same normalisation for the two callers in this module.
fn resolve_axis_pub(arr: &NdArray, axis: Option<&Bound<'_, PyAny>>) -> PyResult<Option<usize>> {
    match axis {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => {
            let a: isize = o.extract()?;
            Ok(Some(norm_axis(a, arr.ndim())?))
        }
    }
}

/// The number of elements each output cell reduces over, honouring `where`.
/// Always keepdims-shaped, so it lines up with the keepdims mean.
fn counts(arr: &NdArray, axis: Option<usize>, mask: Option<&NdArray>) -> NdArray {
    let shape: Vec<isize> = match axis {
        None => vec![1; arr.ndim()],
        Some(a) => {
            let mut s = arr.shape.clone();
            s[a] = 1;
            s
        }
    };
    let out = NdArray::zeros(shape.clone(), DType::F64).expect("count alloc");
    match mask {
        None => {
            let n = match axis {
                None => arr.size() as f64,
                Some(a) => arr.shape[a].max(0) as f64,
            };
            for off in rnp_core::iter::offsets(&out.shape, &out.strides, out.byte_offset) {
                out.write_at(off, Scalar::Float(n));
            }
        }
        Some(m) => {
            let keep = keepdims_view(&out, arr, axis);
            for (src, dst) in rnp_core::iter::offsets(&m.shape, &m.strides, m.byte_offset)
                .zip(rnp_core::iter::offsets(&keep.shape, &keep.strides, keep.byte_offset))
            {
                if m.read_at(src).as_f64() != 0.0 {
                    let prev = out.read_at(dst).as_f64();
                    out.write_at(dst, Scalar::Float(prev + 1.0));
                }
            }
        }
    }
    out
}

/// `out` stretched back over `arr`'s full shape, so a keepdims accumulator
/// can be walked in lockstep with the operand.
fn keepdims_view(out: &NdArray, arr: &NdArray, axis: Option<usize>) -> NdArray {
    let mut v = out.clone();
    v.shape = arr.shape.clone();
    match axis {
        None => v.strides = vec![0; arr.ndim()],
        Some(a) => v.strides[a] = 0,
    }
    for (k, &d) in arr.shape.iter().enumerate() {
        if out.shape[k] == 1 && d != 1 {
            v.strides[k] = 0;
        }
    }
    v
}

/// numpy's `_methods._mean` warning for a reduction over nothing.
pub fn warn_empty_mean(py: Python<'_>) -> PyResult<()> {
    PyErr::warn(
        py,
        &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
        std::ffi::CString::new("Mean of empty slice.")
            .unwrap()
            .as_c_str(),
        1,
    )
}

/// True when any reduction slice would divide by zero (or fewer) elements.
fn any_nonpositive(denom: &NdArray) -> bool {
    rnp_core::iter::offsets(&denom.shape, &denom.strides, denom.byte_offset)
        .any(|o| denom.read_at(o).as_f64() <= 0.0)
}

/// Sum over `axis`, skipping the elements `mask` excludes.
///
/// With no mask this delegates to the engine's own reduction so that the
/// accumulation order — and therefore the last ULP — matches numpy's
/// pairwise summation exactly; the hand-rolled loop below is only for the
/// `where=` path, which numpy also accumulates sequentially.
fn masked_sum(
    arr: &NdArray,
    axis: Option<usize>,
    mask: Option<&NdArray>,
    keepdims: bool,
) -> PyResult<NdArray> {
    if mask.is_none() {
        return match axis {
            Some(a) => rnp_core::reduce_axis(arr, a, rnp_core::ReduceOp::Sum, keepdims)
                .map_err(crate::err),
            None => {
                let total = if arr.size() == 0 {
                    Scalar::Float(0.0).cast(arr.dtype())
                } else {
                    rnp_core::reduce_all(arr, rnp_core::ReduceOp::Sum).map_err(crate::err)?
                };
                let shape = if keepdims {
                    vec![1isize; arr.ndim()]
                } else {
                    vec![]
                };
                let mut out = NdArray::zeros(shape, arr.dtype()).map_err(crate::err)?;
                out.fill(total);
                Ok(out)
            }
        };
    }
    let acc_shape: Vec<isize> = match axis {
        None => vec![1; arr.ndim()],
        Some(a) => {
            let mut s = arr.shape.clone();
            s[a] = 1;
            s
        }
    };
    let out = NdArray::zeros(acc_shape, arr.dtype()).map_err(crate::err)?;
    let keep = keepdims_view(&out, arr, axis);
    let src: Vec<isize> =
        rnp_core::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset).collect();
    let dst: Vec<isize> =
        rnp_core::iter::offsets(&keep.shape, &keep.strides, keep.byte_offset).collect();
    let msk: Option<Vec<isize>> =
        mask.map(|m| rnp_core::iter::offsets(&m.shape, &m.strides, m.byte_offset).collect());
    for i in 0..src.len() {
        if let (Some(m), Some(offs)) = (mask, msk.as_ref()) {
            if m.read_at(offs[i]).as_f64() == 0.0 {
                continue;
            }
        }
        let prev = out.read_at(dst[i]);
        let add = arr.read_at(src[i]);
        out.write_at(dst[i], add_scalar(prev, add));
    }
    if keepdims || axis.is_none() {
        return Ok(out);
    }
    Ok(drop_axis(&out, axis.unwrap()))
}

fn add_scalar(a: Scalar, b: Scalar) -> Scalar {
    match (a, b) {
        (Scalar::Complex(x), Scalar::Complex(y)) => Scalar::Complex(x + y),
        (Scalar::Complex(x), y) => Scalar::Complex(x + num_complex::Complex::new(y.as_f64(), 0.0)),
        (x, Scalar::Complex(y)) => Scalar::Complex(num_complex::Complex::new(x.as_f64(), 0.0) + y),
        (Scalar::Int(x), Scalar::Int(y)) => Scalar::Int(x.wrapping_add(y)),
        (Scalar::Uint(x), Scalar::Uint(y)) => Scalar::Uint(x.wrapping_add(y)),
        (x, y) => Scalar::Float(x.as_f64() + y.as_f64()),
    }
}

fn drop_axis(arr: &NdArray, axis: usize) -> NdArray {
    let mut out = arr.clone();
    out.shape.remove(axis);
    out.strides.remove(axis);
    out.update_flags();
    out
}

/// `count - ddof`, floored at zero (numpy's `max(rcount - ddof, 0)`).
fn sub_clamped(count: &NdArray, ddof: f64) -> NdArray {
    let out = NdArray::zeros(count.shape.clone(), DType::F64).expect("denom alloc");
    for (s, d) in rnp_core::iter::offsets(&count.shape, &count.strides, count.byte_offset)
        .zip(rnp_core::iter::offsets(&out.shape, &out.strides, out.byte_offset))
    {
        out.write_at(d, Scalar::Float((count.read_at(s).as_f64() - ddof).max(0.0)));
    }
    out
}

/// Elementwise divide of a (possibly keepdims) accumulator by a count array
/// of the same shape, producing `dt`.
fn divide_by(total: &NdArray, count: &NdArray, dt: DType) -> NdArray {
    // Materialise the counts in the accumulator's own dtype and shape and let
    // the engine's divide do the work: numpy divides a complex accumulator
    // through its *complex* divide, which is not a component-wise one, and
    // reusing `binary` keeps this bit-identical with it.
    let denom = NdArray::zeros(total.shape.clone(), dt).expect("divide alloc");
    let c: Vec<isize> =
        rnp_core::iter::offsets(&count.shape, &count.strides, count.byte_offset).collect();
    for (i, o) in
        rnp_core::iter::offsets(&denom.shape, &denom.strides, denom.byte_offset).enumerate()
    {
        let n = count.read_at(c[i % c.len()]).as_f64();
        denom.write_at(o, Scalar::Float(n).cast(dt));
    }
    let num = total.astype(dt);
    rnp_core::binary(&num, &denom, BinOp::Div).expect("divide")
}

// ---------------------------------------------------------------------------
// getfield / setfield
// ---------------------------------------------------------------------------

/// `a.getfield(dtype, offset=0)` — reinterpret the bytes at `offset` within
/// each element under another dtype.
pub fn getfield_impl(
    slf: &Bound<'_, PyNdArray>,
    dtype: &Bound<'_, PyAny>,
    offset: isize,
) -> PyResult<Py<PyNdArray>> {
    let arr = slf.borrow().arr.clone();
    let d = descr_from_any(dtype)?;
    if offset < 0 || offset + d.itemsize() as isize > arr.itemsize() as isize {
        return Err(PyValueError::new_err(format!(
            "Need 0 <= offset <= {} for requested type but received offset = {}, \
             required size = {}",
            arr.itemsize() as isize - d.itemsize() as isize,
            offset,
            d.itemsize()
        )));
    }
    let mut out = arr.clone();
    out.descr = d;
    out.byte_offset += offset;
    out.flags.owndata = false;
    out.update_flags();
    PyNdArray::view_of(out, slf)
}

/// `a.setfield(value, dtype, offset=0)` — the in-place inverse of `getfield`.
pub fn setfield_impl(
    slf: &Bound<'_, PyNdArray>,
    value: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
    offset: isize,
) -> PyResult<()> {
    let field = getfield_impl(slf, dtype, offset)?;
    let py = slf.py();
    let target = field.borrow(py).arr.clone();
    let src = array_from_any(value, Some(target.dtype()), false)?;
    let bc = rnp_core::iter::broadcast_to(&src, &target.shape).map_err(crate::err)?;
    for (s, d) in rnp_core::iter::offsets(&bc.shape, &bc.strides, bc.byte_offset).zip(
        rnp_core::iter::offsets(&target.shape, &target.strides, target.byte_offset),
    ) {
        if target.dtype().is_flexible() {
            target.write_raw_at(d, bc.raw_bytes_at(s));
        } else {
            target.write_at(d, bc.read_at(s));
        }
    }
    Ok(())
}

/// `np.c_[...]` — column-wise concatenation with `r_`'s 2-D upgrade rule:
/// every operand is turned into at least a 2-D array by appending an axis,
/// then the pieces are joined on the last axis.
#[pyfunction]
pub fn _c_concat<'py>(py: Python<'py>, items: &Bound<'py, PyAny>) -> PyResult<Py<PyNdArray>> {
    let mut parts: Vec<NdArray> = Vec::new();
    for it in items.try_iter()? {
        let a = array_from_any(&it?, None, false)?;
        let a = match a.ndim() {
            0 => a.reshape(&[1, 1]).map_err(crate::err)?,
            1 => {
                let n = a.shape[0];
                a.reshape(&[n, 1]).map_err(crate::err)?
            }
            _ => a,
        };
        parts.push(a);
    }
    if parts.is_empty() {
        return Err(PyValueError::new_err(
            "need at least one array to concatenate",
        ));
    }
    let mut dt = parts[0].dtype();
    for p in &parts[1..] {
        dt = rnp_core::promote(dt, p.dtype());
    }
    let rows = parts[0].shape[0];
    let mut cols = 0isize;
    for p in &parts {
        if p.ndim() != 2 || p.shape[0] != rows {
            return Err(PyValueError::new_err(
                "all the input array dimensions except for the concatenation \
                 axis must match exactly",
            ));
        }
        cols += p.shape[1];
    }
    let out = NdArray::zeros(vec![rows, cols], dt).map_err(crate::err)?;
    let mut base = 0isize;
    for p in &parts {
        let c = p.astype(dt);
        for r in 0..rows {
            for j in 0..p.shape[1] {
                let v = c.get(&[r, j]).map_err(crate::err)?;
                out.write_at(out.byte_index(&[r, base + j]), v);
            }
        }
        base += p.shape[1];
    }
    let _ = py;
    PyNdArray::into_py_any(out, py)
}

/// Index helper used by the `IndexError` message for `getfield` on a struct.
#[allow(dead_code)]
fn _unused(_: PyIndexError) {}
