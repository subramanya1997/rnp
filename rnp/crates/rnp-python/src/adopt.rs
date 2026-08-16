//! Adopting foreign memory: `ndarray(..., buffer=...)`, `np.frombuffer`, and
//! the array protocol (`__array__` / `__array_interface__`).
//!
//! # Lifetime model
//!
//! An adopted array's `Buffer` does not own its bytes; it owns a *keep-alive
//! token* ([`KeepAlive`]) holding one strong reference to a Python object that
//! guarantees the memory. Because `NdArray` clones share the same
//! `Arc<Buffer>`, every view of an adopted array transitively keeps that
//! Python object alive, which is exactly the invariant
//! `Buffer::from_foreign` demands.
//!
//! There are two grades of guarantee, and they are numpy's, not ours:
//!
//! * **`ndarray(buffer=X)`** — numpy calls `PyObject_GetBuffer`, records the
//!   pointer, immediately calls `PyBuffer_Release`, and keeps only a reference
//!   to `X` (see `PyArray_BufferConverter` in `conversion_utils.c`, whose
//!   comment says the exporter is expected to keep the buffer around). We do
//!   the same, so `mmap.close()` on an adopted mapping is permitted — and
//!   dereferencing afterwards segfaults, in this port exactly as in numpy.
//! * **`frombuffer(X)`** — when `X`'s type has a `bf_releasebuffer` slot,
//!   numpy wraps it in a `memoryview` first so the *export* is held for the
//!   array's lifetime. We do the same, and the `memoryview` is the object we
//!   keep alive, so the pointer is guaranteed by CPython for as long as any
//!   view exists (this is what makes `mmap.close()` raise `BufferError`).
//!
//! In both cases the `Py_buffer` we obtain is released exactly once, on the
//! statement after we read `.buf`/`.len` out of it; no `Py_buffer` is ever
//! stored, so there is no path on which one leaks or is released twice.

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};

use rnp_core::array::{c_strides, f_strides, Flags};
use rnp_core::buffer::{Buffer, ForeignOwner};
use rnp_core::descr::Descr;
use rnp_core::{DType, NdArray};

use crate::pyarray::PyNdArray;
use crate::pydtype::descr_from_any;

// ---------------------------------------------------------------------------
// keep-alive token
// ---------------------------------------------------------------------------

/// One strong reference to the object that guarantees an adopted allocation.
///
/// Dropping it decrefs; when that object is a `memoryview` the decref is also
/// what releases the underlying `Py_buffer` export.
struct KeepAlive(#[allow(dead_code)] Py<PyAny>);

// SAFETY: `Py<PyAny>` is `Send + Sync` (dropping without the GIL is deferred
// to pyo3's pending-decref queue), and `KeepAlive` adds no other state.
impl ForeignOwner for KeepAlive {}

/// What a successful `PyObject_GetBuffer` told us, after the view was released.
struct RawBuf {
    ptr: *mut u8,
    len: usize,
    writable: bool,
}

/// Ask `obj` for a contiguous buffer, preferring a writable one.
///
/// The `Py_buffer` is released before this returns; only the address, the
/// length and the writability survive, mirroring numpy's
/// `PyArray_BufferConverter`.
fn request_buffer(obj: &Bound<'_, PyAny>) -> PyResult<RawBuf> {
    let mut view = std::mem::MaybeUninit::<ffi::Py_buffer>::zeroed();
    let want = ffi::PyBUF_ANY_CONTIGUOUS | ffi::PyBUF_SIMPLE;
    // SAFETY: `obj` is a live, non-null Python object and `view` is a
    // correctly-sized, writable `Py_buffer` slot we own. `PyObject_GetBuffer`
    // either fills it in and returns 0 (in which case we must release it
    // exactly once, which the `PyBuffer_Release` below does on every path) or
    // returns -1 with an exception set and leaves it untouched.
    let (rc, writable) = unsafe {
        let rc = ffi::PyObject_GetBuffer(obj.as_ptr(), view.as_mut_ptr(), want | ffi::PyBUF_WRITABLE);
        if rc == 0 {
            (0, true)
        } else {
            ffi::PyErr_Clear();
            (ffi::PyObject_GetBuffer(obj.as_ptr(), view.as_mut_ptr(), want), false)
        }
    };
    if rc != 0 {
        return Err(PyErr::fetch(obj.py()));
    }
    // SAFETY: `rc == 0` means the view was filled in, so it is initialised.
    // We copy the two scalars we need and release the view immediately; from
    // here on the memory is guaranteed by the keep-alive object the caller
    // stores, exactly as numpy documents in `PyArray_BufferConverter`.
    let (ptr, len) = unsafe {
        let v = view.assume_init_mut();
        let out = (v.buf as *mut u8, v.len as usize);
        ffi::PyBuffer_Release(v);
        out
    };
    Ok(RawBuf { ptr, len, writable })
}

/// The object numpy records as `arr.base` for `ndarray(buffer=obj)`:
/// a `memoryview`'s underlying object, or `obj` itself.
fn buffer_base<'py>(obj: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
    if let Ok(mv) = obj.cast::<pyo3::types::PyMemoryView>() {
        if let Ok(inner) = mv.getattr("obj") {
            if !inner.is_none() {
                return inner;
            }
        }
    }
    obj.clone()
}

/// Build the `Arc<Buffer>` that adopts `raw`, keeping `keep` alive.
///
/// `raw.ptr` may legitimately be null for a zero-length buffer (`b""`), which
/// `Buffer::from_foreign` refuses; a dangling-but-aligned address is used
/// instead, and nothing may ever read from it because `len == 0`.
fn adopt(raw: &RawBuf, keep: &Bound<'_, PyAny>) -> Arc<Buffer> {
    let ptr = if raw.ptr.is_null() {
        std::ptr::NonNull::<u8>::dangling().as_ptr()
    } else {
        raw.ptr
    };
    let owner = Box::new(KeepAlive(keep.clone().unbind()));
    // SAFETY: `ptr`/`len` came from a `Py_buffer` that CPython filled in for
    // `keep` (or, when `len == 0`, from a dangling pointer that is never
    // dereferenced). `KeepAlive` holds a strong reference to `keep` for as
    // long as this `Buffer` lives, and the buffer protocol requires an
    // exporter to keep an exported region valid and at a fixed address while
    // it is referenced this way — the same contract numpy relies on.
    Arc::new(unsafe { Buffer::from_foreign(ptr, raw.len, owner) })
}

// ---------------------------------------------------------------------------
// shape / order / stride helpers
// ---------------------------------------------------------------------------

/// numpy's `PyArray_IntpConverter`: an int, or any sequence of ints.
pub fn dims_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(v) = crate::pyarray::shape_from_any(obj) {
        return Ok(v);
    }
    // Anything else iterable (a generator, a range, ...).
    let mut out = Vec::new();
    for item in obj.try_iter().map_err(|_| {
        PyTypeError::new_err("expected a sequence of integers or a single integer")
    })? {
        out.push(item?.extract::<isize>().map_err(|_| {
            PyTypeError::new_err("expected a sequence of integers or a single integer")
        })?);
    }
    Ok(out)
}

/// numpy's `PyArray_OrderConverter`. Only "is this Fortran order" survives.
fn is_f_order(order: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    let Some(o) = order else { return Ok(false) };
    if o.is_none() {
        return Ok(false);
    }
    let s: String = o.extract().map_err(|_| {
        PyTypeError::new_err("order must be str, not ".to_string() + &type_name(o))
    })?;
    match s.chars().next() {
        Some('C') | Some('c') => Ok(false),
        Some('F') | Some('f') => Ok(true),
        Some('A') | Some('a') | Some('K') | Some('k') => Ok(false),
        _ => Err(PyValueError::new_err(format!(
            "order must be one of 'C', 'F', 'A', or 'K' (got '{s}')"
        ))),
    }
}

fn type_name(o: &Bound<'_, PyAny>) -> String {
    o.get_type().name().map(|n| n.to_string()).unwrap_or_default()
}

/// The lowest and highest byte offsets a strided walk can reach, relative to
/// element (0, 0, ...). Mirrors numpy's `offset_bounds_from_strides`.
fn offset_bounds(itemsize: usize, dims: &[isize], strides: &[isize]) -> (isize, isize) {
    let mut lower = 0isize;
    let mut upper = itemsize as isize;
    for (i, &d) in dims.iter().enumerate() {
        if d == 0 {
            return (0, 0);
        }
        let step = (d - 1) * strides[i];
        if step < 0 {
            lower += step;
        } else {
            upper += step;
        }
    }
    (lower, upper)
}

/// numpy's `PyArray_CheckStrides`: can this stride set ever walk outside the
/// `numbytes` region that starts `offset` bytes before the first element?
pub fn check_strides(
    itemsize: usize,
    numbytes: isize,
    offset: isize,
    dims: &[isize],
    strides: &[isize],
) -> bool {
    let numbytes = if numbytes == 0 {
        dims.iter().map(|&d| d.max(0)).product::<isize>() * itemsize as isize
    } else {
        numbytes
    };
    let (lower, upper) = offset_bounds(itemsize, dims, strides);
    !(upper > numbytes - offset || lower < -offset)
}

// ---------------------------------------------------------------------------
// ndarray.__new__
// ---------------------------------------------------------------------------

/// `ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)`.
///
/// A faithful port of `array_new` in `arrayobject.c`, including its quirks:
/// a negative `offset` is *not* rejected outright (only the resulting span is
/// checked), a misaligned offset merely clears `ALIGNED`, and `shape=-1` with
/// a buffer means "as many items as fit".
pub fn ndarray_new(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    buffer: Option<&Bound<'_, PyAny>>,
    offset: i64,
    strides: Option<&Bound<'_, PyAny>>,
    order: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyNdArray> {
    let mut dims = dims_from_any(shape)?;
    let descr = match dtype {
        None => Descr::native(DType::F64),
        Some(o) if o.is_none() => Descr::native(DType::F64),
        Some(o) => descr_from_any(o)?,
    };
    let itemsize = descr.itemsize();
    let f_order = is_f_order(order)?;

    let buffer = match buffer {
        Some(b) if !b.is_none() => Some(b),
        _ => None,
    };

    let strides = match strides {
        Some(s) if !s.is_none() => Some(dims_from_any(s)?),
        _ => None,
    };

    // numpy validates strides *before* looking at the buffer contents.
    if let Some(st) = &strides {
        if st.len() != dims.len() {
            return Err(PyValueError::new_err(
                "strides, if given, must be the same length as shape",
            ));
        }
    }

    let Some(bufobj) = buffer else {
        // No buffer: allocate, exactly like `np.empty` (numpy leaves the
        // memory uninitialised here; object slots are set to None).
        if let Some(st) = &strides {
            if !check_strides(itemsize, 0, 0, &dims, st) {
                return Err(PyValueError::new_err(
                    "strides is incompatible with shape of requested array and size of buffer",
                ));
            }
        }
        let mut arr = NdArray::empty_descr(dims.clone(), descr).map_err(crate::err)?;
        if let Some(st) = strides {
            arr.strides = st;
        } else if dims.iter().any(|&d| d == 0) {
            // numpy hands an empty *self-allocated* array all-zero strides
            // (`np.ndarray((0, 4)).strides == (0, 0)`); an empty array over a
            // caller-supplied buffer keeps the usual C/F strides.
            arr.strides = vec![0; dims.len()];
        } else if f_order {
            arr.strides = f_strides(&dims, itemsize);
        }
        set_layout_flags(&mut arr);
        return Ok(PyNdArray { arr, base: None });
    };

    let raw = request_buffer(bufobj)?;
    let base = buffer_base(bufobj);

    if let Some(st) = &strides {
        if !check_strides(itemsize, raw.len as isize, offset as isize, &dims, st) {
            return Err(PyValueError::new_err(
                "strides is incompatible with shape of requested array and size of buffer",
            ));
        }
    }

    if dims.len() == 1 && dims[0] == -1 {
        if itemsize == 0 {
            return Err(PyValueError::new_err("itemsize cannot be zero in type"));
        }
        dims[0] = (raw.len as isize - offset as isize) / itemsize as isize;
    } else {
        for &d in &dims {
            if d < 0 {
                return Err(PyValueError::new_err("negative dimensions are not allowed"));
            }
        }
        if strides.is_none() {
            let need = offset as i128
                + itemsize as i128 * dims.iter().map(|&d| d as i128).product::<i128>();
            if (raw.len as i128) < need {
                return Err(PyTypeError::new_err("buffer is too small for requested array"));
            }
        }
    }
    let strides = strides.unwrap_or_else(|| {
        if f_order {
            f_strides(&dims, itemsize)
        } else {
            c_strides(&dims, itemsize)
        }
    });
    let mut arr = NdArray {
        buffer: adopt(&raw, &base),
        byte_offset: offset as isize,
        shape: dims,
        strides,
        descr,
        flags: Flags::default(),
    };
    set_layout_flags(&mut arr);
    arr.flags.writeable = raw.writable;
    arr.flags.owndata = false;
    Ok(PyNdArray {
        arr,
        base: Some(base.unbind()),
    })
}

// ---------------------------------------------------------------------------
// frombuffer
// ---------------------------------------------------------------------------

/// True when `obj`'s type implements `bf_releasebuffer`, i.e. when holding the
/// export actually matters for the memory's lifetime.
fn has_release_buffer(obj: &Bound<'_, PyAny>) -> bool {
    // One of our own arrays needs no memoryview: holding the `PyNdArray`
    // holds its `Arc<Buffer>`, so the pointer is guaranteed anyway. numpy
    // reasons identically about its own arrays ("NumPy arrays will never get
    // wrapped here"), and it is what makes `frombuffer(arr).base is arr`.
    if obj.cast::<PyNdArray>().is_ok() {
        return false;
    }
    // SAFETY: `obj` is a live object, so `Py_TYPE` is a valid type object and
    // `tp_as_buffer` is either null or a valid `PyBufferProcs`.
    unsafe {
        let ty = ffi::Py_TYPE(obj.as_ptr());
        let procs = (*ty).tp_as_buffer;
        !procs.is_null() && (*procs).bf_releasebuffer.is_some()
    }
}

/// `np.frombuffer(buffer, dtype=float, count=-1, offset=0)` — a port of
/// `PyArray_FromBuffer`.
pub fn frombuffer(
    py: Python<'_>,
    buffer: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
    count: i64,
    offset: i64,
) -> PyResult<PyNdArray> {
    let descr = match dtype {
        None => Descr::native(DType::F64),
        Some(o) if o.is_none() => Descr::native(DType::F64),
        Some(o) => descr_from_any(o)?,
    };
    if descr.dt.is_object() {
        return Err(PyValueError::new_err(
            "cannot create an OBJECT array from memory buffer",
        ));
    }
    let itemsize = descr.itemsize();
    if itemsize == 0 {
        return Err(PyValueError::new_err("itemsize cannot be zero in type"));
    }

    // Wrap in a memoryview when the exporter has a release slot, so the export
    // — not just the object — is held for the array's lifetime. Otherwise keep
    // the object itself, which is what makes `frombuffer(b).base is b`.
    let keep = if has_release_buffer(buffer) {
        pyo3::types::PyMemoryView::from(buffer)?.into_any()
    } else {
        buffer.clone()
    };

    let raw = request_buffer(&keep)?;
    let ts = raw.len as i64;
    if offset < 0 || offset > ts {
        return Err(PyValueError::new_err(format!(
            "offset must be non-negative and no greater than buffer length ({ts})"
        )));
    }
    let s = ts - offset;
    let n = if count < 0 {
        if s % itemsize as i64 != 0 {
            return Err(PyValueError::new_err(
                "buffer size must be a multiple of element size",
            ));
        }
        s / itemsize as i64
    } else {
        if s < count * itemsize as i64 {
            return Err(PyValueError::new_err("buffer is smaller than requested size"));
        }
        count
    };

    let _ = py;
    let mut arr = NdArray {
        buffer: adopt(&raw, &keep),
        byte_offset: offset as isize,
        shape: vec![n as isize],
        strides: vec![itemsize as isize],
        descr,
        flags: Flags::default(),
    };
    set_layout_flags(&mut arr);
    arr.flags.writeable = raw.writable;
    arr.flags.owndata = false;
    Ok(PyNdArray {
        arr,
        base: Some(keep.unbind()),
    })
}

// ---------------------------------------------------------------------------
// subclass-aware allocation
// ---------------------------------------------------------------------------

/// Build `arr` as an instance of `ty`, which must be `ndarray` or a Python
/// subclass of it.
///
/// A subclass instance cannot be produced with `Py::new` (its layout carries
/// the subclass's `__dict__`), so `ndarray.__new__(ty, ())` allocates it —
/// deliberately bypassing any `__new__` the subclass defines, exactly as
/// numpy's view machinery bypasses it — and its contents are then replaced.
/// `__array_finalize__` is invoked afterwards when the subclass defines one.
pub fn new_of_type(
    py: Python<'_>,
    ty: &Bound<'_, PyType>,
    arr: NdArray,
    base: Option<Py<PyAny>>,
    parent: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let nd = py.get_type::<PyNdArray>();
    if ty.is(&nd) {
        return Ok(Py::new(py, PyNdArray { arr, base })?.into_any());
    }
    if !ty.is_subclass(&nd)? {
        return Err(PyTypeError::new_err(
            "type must be a sub-type of ndarray type",
        ));
    }
    let obj = nd
        .getattr("__new__")?
        .call1((ty, PyTuple::empty(py)))?;
    {
        let cell = obj.cast::<PyNdArray>()?;
        let mut me = cell.borrow_mut();
        me.arr = arr;
        me.base = base;
    }
    if let Ok(fin) = obj.getattr("__array_finalize__") {
        if !fin.is_none() {
            let p: Py<PyAny> = match parent {
                Some(p) => p.clone().unbind(),
                None => py.None(),
            };
            fin.call1((p,))?;
        }
    }
    Ok(obj.unbind())
}

/// `Py_TPFLAGS_SEQUENCE` (CPython 3.10+), the bit `match arr: case [a, b]`
/// consults. It has no dedicated pyo3 binding.
const PY_TPFLAGS_SEQUENCE: std::os::raw::c_ulong = 1 << 5;

/// Tell CPython's structural pattern matching that `ndarray` is a sequence.
///
/// numpy's `PyArray_Type` sets this bit, so `match np.array([1, 2, 3])`
/// destructures; a pyclass does not get it for free. Called once, from module
/// init, before any instance exists.
pub fn mark_ndarray_as_sequence(py: Python<'_>) {
    let ty = py.get_type::<PyNdArray>();
    // SAFETY: `ty` is the live, heap-allocated type object pyo3 just created
    // for `PyNdArray`; we own it exclusively at module-init time (no instance
    // and no subclass exists yet). Setting a tp_flags bit that only affects
    // pattern matching cannot invalidate any layout invariant, and
    // `PyType_Modified` invalidates the attribute caches afterwards, which is
    // what CPython requires after mutating a type in place.
    unsafe {
        let tp = ty.as_type_ptr();
        (*tp).tp_flags |= PY_TPFLAGS_SEQUENCE;
        ffi::PyType_Modified(tp);
    }
}

// ---------------------------------------------------------------------------
// the array protocol
// ---------------------------------------------------------------------------

/// numpy's message when `np.array(..., copy=False)` cannot avoid a copy.
pub const NO_COPY_MSG: &str = "Unable to avoid copy while creating an array as requested.\nIf using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).\nFor more details, see https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.";

const COPY_DEPRECATION: &str = "__array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword";

/// Call `obj.__array__(dtype=..., copy=...)`, handling numpy 2.x's fallback
/// for third-party implementations that predate the `copy` keyword.
///
/// Returns `None` when `obj` has no `__array__` at all.
pub fn call_array_protocol<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dtype: Option<Descr>,
    copy: Option<bool>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    // numpy looks the hook up on the *type*, so an instance attribute named
    // `__array__` does not count.
    let Ok(hook) = obj.get_type().getattr("__array__") else {
        return Ok(None);
    };
    let _ = &hook;
    let hook = obj.getattr("__array__")?;
    let kwargs = pyo3::types::PyDict::new(py);
    let dt: Py<PyAny> = match dtype {
        Some(d) => crate::pydtype::PyDType::from_descr(d).into_pyobject(py)?.into_any().unbind(),
        None => py.None(),
    };
    kwargs.set_item("dtype", &dt)?;
    kwargs.set_item(
        "copy",
        match copy {
            None => py.None(),
            Some(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        },
    )?;
    match hook.call((), Some(&kwargs)) {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.is_instance_of::<PyTypeError>(py) => {
            // Legacy `__array__(self, dtype=None)`. numpy retries without the
            // keyword, warning only when a copy was explicitly refused.
            if copy == Some(false) {
                pyo3::PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
                    std::ffi::CString::new(COPY_DEPRECATION).unwrap().as_c_str(),
                    1,
                )?;
                return Err(PyValueError::new_err(NO_COPY_MSG));
            }
            let kw = pyo3::types::PyDict::new(py);
            kw.set_item("dtype", &dt)?;
            match hook.call((), Some(&kw)) {
                Ok(v) => Ok(Some(v)),
                Err(_) => Ok(Some(hook.call0()?)),
            }
        }
        Err(e) => Err(e),
    }
}

/// Build an array from `obj.__array_interface__` (version 3), zero-copy when
/// the exporter hands us a raw address.
pub fn from_array_interface(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Option<PyNdArray>> {
    let Ok(iface) = obj.getattr("__array_interface__") else {
        return Ok(None);
    };
    let Ok(d) = iface.cast::<pyo3::types::PyDict>() else {
        return Err(PyValueError::new_err(
            "Invalid __array_interface__ value, must be a dict",
        ));
    };
    if let Some(mask) = d.get_item("mask")? {
        if !mask.is_none() {
            return Err(PyValueError::new_err(
                "__array_interface__ masked arrays are not supported",
            ));
        }
    }
    let shape: Vec<isize> = match d.get_item("shape")? {
        Some(s) => dims_from_any(&s)?,
        None => return Err(PyValueError::new_err("Missing __array_interface__ shape")),
    };
    let typestr = match d.get_item("typestr")? {
        Some(t) => t,
        None => return Err(PyValueError::new_err("Missing __array_interface__ typestr")),
    };
    let descr = descr_from_any(&typestr)?;
    let itemsize = descr.itemsize();
    let strides = match d.get_item("strides")? {
        Some(s) if !s.is_none() => dims_from_any(&s)?,
        _ => c_strides(&shape, itemsize),
    };
    let data = match d.get_item("data")? {
        Some(x) => x,
        None => return Err(PyValueError::new_err("Missing __array_interface__ data")),
    };
    if let Ok(t) = data.cast::<PyTuple>() {
        let addr: usize = t.get_item(0)?.extract()?;
        let readonly: bool = t.get_item(1)?.extract()?;
        // The exporter promises the address stays valid while it is alive; we
        // hold a reference to it (numpy holds the object itself the same way).
        let span = span_bytes(itemsize, &shape, &strides);
        let owner = Box::new(KeepAlive(obj.clone().unbind()));
        // SAFETY: the __array_interface__ contract states that `data[0]` is a
        // valid address for the described array and remains valid for as long
        // as the exporting object lives; `KeepAlive` holds that object. This
        // is the same trust numpy places in the protocol.
        let buf = unsafe { Buffer::from_foreign(addr as *mut u8, span, owner) };
        let mut arr = NdArray {
            buffer: Arc::new(buf),
            byte_offset: 0,
            shape,
            strides,
            descr,
            flags: Flags::default(),
        };
        set_layout_flags(&mut arr);
        arr.flags.writeable = !readonly;
        arr.flags.owndata = false;
        return Ok(Some(PyNdArray {
            arr,
            base: Some(obj.clone().unbind()),
        }));
    }
    // A buffer-exporting object in the `data` slot.
    let raw = request_buffer(&data)?;
    let mut arr = NdArray {
        buffer: adopt(&raw, &data),
        byte_offset: 0,
        shape,
        strides,
        descr,
        flags: Flags::default(),
    };
    set_layout_flags(&mut arr);
    arr.flags.writeable = raw.writable;
    arr.flags.owndata = false;
    let _ = py;
    Ok(Some(PyNdArray {
        arr,
        base: Some(data.unbind()),
    }))
}

/// numpy's `_IsAligned`: the data pointer *and* every stride must be a
/// multiple of the dtype's alignment. `NdArray::update_flags` only checks the
/// pointer, so an adopted array with hand-written strides needs this pass.
fn set_layout_flags(arr: &mut NdArray) {
    arr.update_flags();
    let align = arr.dtype().alignment() as isize;
    if align > 1 && arr.strides.iter().any(|s| s % align != 0) {
        arr.flags.aligned = false;
    }
}

/// Total byte span a strided array occupies from its first element.
fn span_bytes(itemsize: usize, shape: &[isize], strides: &[isize]) -> usize {
    let (lower, upper) = offset_bounds(itemsize, shape, strides);
    (upper - lower).max(0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_strides_matches_numpy() {
        // 4 int32 in a 64-byte buffer at offset 0 with stride 100: escapes.
        assert!(!check_strides(4, 64, 0, &[4], &[100]));
        // Negative strides walking back from offset 48 stay inside.
        assert!(check_strides(4, 64, 48, &[4], &[-4]));
        // ... but not from offset 0.
        assert!(!check_strides(4, 64, 0, &[4], &[-4]));
        // A zero stride never escapes.
        assert!(check_strides(4, 64, 0, &[10], &[0]));
        // numbytes == 0 means "derive it from the shape".
        assert!(!check_strides(4, 0, 0, &[4], &[8]));
        assert!(check_strides(4, 0, 0, &[4], &[4]));
        // Empty arrays touch nothing.
        assert!(check_strides(4, 0, 0, &[0], &[1 << 40]));
    }

    #[test]
    fn offset_bounds_are_inclusive_of_the_item() {
        assert_eq!(offset_bounds(8, &[3], &[8]), (0, 24));
        assert_eq!(offset_bounds(8, &[3], &[-8]), (-16, 8));
        assert_eq!(offset_bounds(8, &[2, 2], &[16, 8]), (0, 32));
    }
}
