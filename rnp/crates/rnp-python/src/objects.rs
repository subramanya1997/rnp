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

use std::sync::Mutex;

use pyo3::prelude::*;

use rnp_core::{DType, NdArray, Scalar};

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
