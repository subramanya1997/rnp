//! Object-dtype storage.
//!
//! An `object` array's elements are 8-byte handles into a process-wide slab
//! of `Py<PyAny>`; handle 0 is `None`, so a zeroed (or `np.empty`) object
//! array reads back as `None`, exactly as numpy's does.
//!
//! The slab is append-only: entries are never released. That trades a bounded
//! leak for the guarantee that no handle can ever dangle, which is the right
//! side of the trade for a port whose object-array support exists to let the
//! upstream test modules construct and inspect these arrays. Dropping to real
//! per-element refcounting is tracked as an M4 item in PLAN.md.

use std::cmp::Ordering;
use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::pyclass::CompareOp;
use pyo3::prelude::*;

use rnp_core::{DType, Descr, NdArray, Scalar};

static SLAB: Mutex<Vec<Py<PyAny>>> = Mutex::new(Vec::new());

/// Intern one Python object and return its handle. `None` is always 0.
pub fn intern(obj: &Bound<'_, PyAny>) -> u64 {
    if obj.is_none() {
        return 0;
    }
    let mut slab = SLAB.lock().unwrap();
    slab.push(obj.clone().unbind());
    slab.len() as u64
}

/// Intern even `None` as a nonzero handle. StringDType needs this because its
/// all-zero cell is the empty string while `None` may be an explicit NA.
fn intern_nonzero(obj: &Bound<'_, PyAny>) -> u64 {
    let mut slab = SLAB.lock().unwrap();
    slab.push(obj.clone().unbind());
    slab.len() as u64
}

/// Resolve a handle back to its Python object.
pub fn resolve<'py>(py: Python<'py>, handle: u64) -> Bound<'py, PyAny> {
    if handle == 0 {
        return py.None().into_bound(py);
    }
    let slab = SLAB.lock().unwrap();
    match slab.get(handle as usize - 1) {
        Some(o) => o.bind(py).clone(),
        None => py.None().into_bound(py),
    }
}

/// The object stored at a byte offset of an `object` array.
pub fn read<'py>(py: Python<'py>, arr: &NdArray, off: isize) -> Bound<'py, PyAny> {
    match arr.read_at(off) {
        Scalar::Uint(h) => resolve(py, h),
        _ => py.None().into_bound(py),
    }
}

/// The object stored in a StringDType cell. A zeroed descriptor is the empty
/// string (not None), matching both `np.empty(..., dtype="T")` and zeros.
pub fn read_string<'py>(py: Python<'py>, arr: &NdArray, off: isize) -> Bound<'py, PyAny> {
    match arr.read_at(off) {
        Scalar::Uint(0) => pyo3::types::PyString::new(py, "").into_any(),
        Scalar::Uint(h) => resolve(py, h),
        _ => pyo3::types::PyString::new(py, "").into_any(),
    }
}

/// Store one Python object at a byte offset of an `object` array.
pub fn write(arr: &NdArray, off: isize, obj: &Bound<'_, PyAny>) {
    arr.write_at(off, Scalar::Uint(intern(obj)));
}

pub fn write_string(arr: &NdArray, off: isize, obj: &Bound<'_, PyAny>) {
    arr.write_at(off, Scalar::Uint(intern_nonzero(obj)));
}

/// Compare two object-array elements with Python's rich comparison protocol.
/// NumPy's object comparator asks `a < b`, then `a > b`; equality is the
/// fallback when both are false.  `rich_compare` preserves Python's reflected
/// operation and exact TypeError when the pair is unorderable.
fn compare_at(py: Python<'_>, arr: &NdArray, a: isize, b: isize) -> PyResult<Ordering> {
    let lhs = read(py, arr, a);
    let rhs = read(py, arr, b);
    if lhs.rich_compare(&rhs, CompareOp::Lt)?.is_truthy()? {
        return Ok(Ordering::Less);
    }
    if lhs.rich_compare(&rhs, CompareOp::Gt)?.is_truthy()? {
        return Ok(Ordering::Greater);
    }
    Ok(Ordering::Equal)
}

fn lanes(arr: &NdArray, axis: usize) -> Vec<Vec<isize>> {
    if arr.ndim() == 0 && axis == 0 {
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

/// A stable, fallible ordering of one lane.  Rust's slice sort comparator
/// cannot propagate a Python exception, so object lanes use insertion sort;
/// object arrays in NumPy's own tests are small and this preserves the first
/// rich-comparison failure verbatim.
fn lane_order(py: Python<'_>, arr: &NdArray, lane: &[isize]) -> PyResult<Vec<usize>> {
    let mut order: Vec<usize> = (0..lane.len()).collect();
    for i in 1..order.len() {
        let mut j = i;
        while j > 0
            && compare_at(py, arr, lane[order[j]], lane[order[j - 1]])? == Ordering::Less
        {
            order.swap(j, j - 1);
            j -= 1;
        }
    }
    Ok(order)
}

pub fn sort_inplace(py: Python<'_>, arr: &NdArray, axis: usize) -> PyResult<()> {
    if !arr.flags.writeable {
        return Err(PyValueError::new_err(
            "assignment destination is read-only",
        ));
    }
    for lane in lanes(arr, axis) {
        let order = lane_order(py, arr, &lane)?;
        let values: Vec<Scalar> = lane.iter().map(|&off| arr.read_at(off)).collect();
        for (dst, &src) in lane.iter().zip(order.iter()) {
            arr.write_at(*dst, values[src]);
        }
    }
    Ok(())
}

pub fn argsort(py: Python<'_>, arr: &NdArray, axis: usize) -> PyResult<NdArray> {
    let out = NdArray::zeros(arr.shape.clone(), DType::I64).map_err(crate::err)?;
    if arr.ndim() == 0 {
        out.write_at(out.byte_offset, Scalar::Int(0));
        return Ok(out);
    }
    let n = arr.shape[axis].max(0);
    let mut shape = arr.shape.clone();
    let mut strides = out.strides.clone();
    shape.remove(axis);
    let out_step = strides.remove(axis);
    let bases: Vec<isize> =
        rnp_core::iter::offsets(&shape, &strides, out.byte_offset).collect();
    for (lane, base) in lanes(arr, axis).iter().zip(bases) {
        let order = lane_order(py, arr, lane)?;
        for k in 0..n as usize {
            out.write_at(
                base + k as isize * out_step,
                Scalar::Int(order[k] as i64),
            );
        }
    }
    Ok(out)
}

/// Build an `object` array from nested Python sequences of arbitrary values.
pub fn array_from_objects(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let mut shape = Vec::new();
    discover(obj, &mut shape);
    let mut items: Vec<Bound<'_, PyAny>> = Vec::new();
    collect(obj, 0, &shape, &mut items)?;
    let out = NdArray::zeros(shape, DType::Object).map_err(crate::err)?;
    for (i, it) in items.iter().enumerate() {
        write(&out, out.byte_offset + (i * 8) as isize, it);
    }
    Ok(out)
}

/// Cast any ndarray to object dtype by interning each converted Python value.
pub fn astype_object(py: Python<'_>, arr: &NdArray) -> PyResult<NdArray> {
    let out = NdArray::zeros(arr.shape.clone(), DType::Object).map_err(crate::err)?;
    let src = rnp_core::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset);
    let dst = rnp_core::iter::offsets(&out.shape, &out.strides, out.byte_offset);
    for (source, target) in src.zip(dst) {
        let value = crate::convert::element_to_py(py, arr, source)?;
        write(&out, target, &value);
    }
    Ok(out)
}

/// Allocate an object array whose every cell owns the same Python value.
pub fn full_object(shape: Vec<isize>, descr: Descr, value: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let out = NdArray::zeros_descr(shape, descr).map_err(crate::err)?;
    for target in rnp_core::iter::offsets(&out.shape, &out.strides, out.byte_offset) {
        write(&out, target, value);
    }
    Ok(out)
}

/// Object arrays only descend through lists and tuples; everything else --
/// including a numpy array, a dict or a generator -- is one element, which is
/// the behaviour `np.array([{...}], dtype=object)` relies on.
fn is_seq(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<pyo3::types::PyList>() || obj.is_instance_of::<pyo3::types::PyTuple>()
}

fn discover(obj: &Bound<'_, PyAny>, shape: &mut Vec<isize>) {
    // The ragged check re-descends every sibling, so the walk is exponential
    // in the nesting depth. Real inputs are a handful of levels deep; a
    // co-recursive list (gh-11154) is infinitely deep, so the walk also
    // carries a hard budget on the number of nodes it will visit.
    let mut budget: u32 = 1_000_000;
    discover_depth(obj, shape, 0, &mut budget)
}

/// numpy's `NPY_MAXDIMS` is a hard recursion bound: a co-recursive list
/// (gh-11154) is infinitely deep, and without the cap the ragged check --
/// which restarts `discover` with a fresh vector at every level -- runs the
/// C stack out.
fn discover_depth(
    obj: &Bound<'_, PyAny>,
    shape: &mut Vec<isize>,
    depth: usize,
    budget: &mut u32,
) {
    if !is_seq(obj) || depth >= 64 || *budget == 0 {
        return;
    }
    *budget -= 1;
    let seq = match obj.cast::<pyo3::types::PySequence>() {
        Ok(s) => s,
        Err(_) => return,
    };
    let n = match seq.len() {
        Ok(n) => n,
        Err(_) => return,
    };
    shape.push(n as isize);
    if n > 0 {
        if let Ok(first) = seq.get_item(0) {
            // Ragged input stops the shape here, as numpy's does.
            let mut sub = Vec::new();
            discover_depth(&first, &mut sub, depth + 1, budget);
            if !sub.is_empty() && ragged(&seq, n, &sub, depth + 1, budget) {
                return;
            }
            shape.extend(sub);
        }
    }
}

fn ragged(
    seq: &Bound<'_, pyo3::types::PySequence>,
    n: usize,
    sub: &[isize],
    depth: usize,
    budget: &mut u32,
) -> bool {
    for i in 0..n {
        let item = match seq.get_item(i) {
            Ok(x) => x,
            Err(_) => return true,
        };
        let mut s = Vec::new();
        discover_depth(&item, &mut s, depth, budget);
        if s != sub {
            return true;
        }
    }
    false
}

fn collect<'py>(
    obj: &Bound<'py, PyAny>,
    depth: usize,
    shape: &[isize],
    out: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    if depth == shape.len() {
        out.push(obj.clone());
        return Ok(());
    }
    let seq = obj.cast::<pyo3::types::PySequence>()?;
    for i in 0..shape[depth] {
        collect(&seq.get_item(i as usize)?, depth + 1, shape, out)?;
    }
    Ok(())
}
