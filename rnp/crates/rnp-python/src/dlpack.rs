//! CPU DLPack producer and consumer support.
//!
//! The ABI definitions below mirror the vendored DLPack 1.0 header used by
//! NumPy 2.5.2.  Export capsules own a Python reference to the source array;
//! imported arrays own an internal capsule which invokes the producer's
//! deleter exactly once when the last array/view releases the foreign buffer.

use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyString, PyTuple};

use rnp_core::array::{c_strides, is_c_contiguous, is_f_contiguous, Flags};
use rnp_core::buffer::{Buffer, ForeignOwner};
use rnp_core::{DType, Descr, NdArray};

use crate::pyarray::PyNdArray;
use crate::pydtype::PyDType;

const K_DL_CPU: i32 = 1;
const K_DL_INT: u8 = 0;
const K_DL_UINT: u8 = 1;
const K_DL_FLOAT: u8 = 2;
const K_DL_COMPLEX: u8 = 5;
const K_DL_BOOL: u8 = 6;
const READ_ONLY: u64 = 1;
const IS_COPIED: u64 = 2;
const MAX_DIMS: usize = 64;

const LEGACY_NAME: &CStr = c"dltensor";
const LEGACY_USED_NAME: &CStr = c"used_dltensor";
const LEGACY_INTERNAL_NAME: &CStr = c"numpy_dltensor";
const VERSIONED_NAME: &CStr = c"dltensor_versioned";
const VERSIONED_USED_NAME: &CStr = c"used_dltensor_versioned";
const VERSIONED_INTERNAL_NAME: &CStr = c"numpy_dltensor_versioned";

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

// The managed tensor is first so a pointer to this allocation is also the ABI
// pointer placed in the capsule.
#[repr(C)]
struct LegacyExport {
    managed: DLManagedTensor,
    shape: Vec<i64>,
    strides: Vec<i64>,
    owner: *mut ffi::PyObject,
}

#[repr(C)]
struct VersionedExport {
    managed: DLManagedTensorVersioned,
    shape: Vec<i64>,
    strides: Vec<i64>,
    owner: *mut ffi::PyObject,
}

unsafe fn drop_legacy_export(ptr: *mut LegacyExport) {
    // SAFETY: `ptr` is the unique Box allocation made by the exporter.
    let boxed = unsafe { Box::from_raw(ptr) };
    let owner = boxed.owner;
    drop(boxed);
    // SAFETY: a DLPack deleter may run on a non-Python thread. Acquiring the
    // GIL makes the synchronous decref of the exported Python owner valid.
    unsafe {
        let state = ffi::PyGILState_Ensure();
        ffi::Py_DECREF(owner);
        ffi::PyGILState_Release(state);
    }
}

unsafe fn drop_versioned_export(ptr: *mut VersionedExport) {
    // SAFETY: `ptr` is the unique Box allocation made by the exporter.
    let boxed = unsafe { Box::from_raw(ptr) };
    let owner = boxed.owner;
    drop(boxed);
    // SAFETY: as in `drop_legacy_export`, the GIL is required for the
    // synchronous decref expected by CPython's capsule ownership contract.
    unsafe {
        let state = ffi::PyGILState_Ensure();
        ffi::Py_DECREF(owner);
        ffi::PyGILState_Release(state);
    }
}

unsafe extern "C" fn legacy_export_deleter(ptr: *mut DLManagedTensor) {
    if !ptr.is_null() {
        // SAFETY: exporter capsules contain a pointer produced by
        // `Box::into_raw(Box<LegacyExport>)`, whose first field is `managed`.
        unsafe { drop_legacy_export(ptr.cast::<LegacyExport>()) };
    }
}

unsafe extern "C" fn versioned_export_deleter(ptr: *mut DLManagedTensorVersioned) {
    if !ptr.is_null() {
        // SAFETY: exporter capsules contain a pointer produced by
        // `Box::into_raw(Box<VersionedExport>)`, whose first field is `managed`.
        unsafe { drop_versioned_export(ptr.cast::<VersionedExport>()) };
    }
}

unsafe extern "C" fn legacy_capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: CPython calls this only with the live capsule being destroyed;
    // the static names are NUL-terminated for the duration of the process.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, LEGACY_USED_NAME.as_ptr()) != 0 {
            return;
        }
        let ptr =
            ffi::PyCapsule_GetPointer(capsule, LEGACY_NAME.as_ptr()).cast::<DLManagedTensor>();
        if !ptr.is_null() {
            if let Some(deleter) = (*ptr).deleter {
                deleter(ptr);
            }
        }
    }
}

unsafe extern "C" fn versioned_capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: CPython calls this only with the live capsule being destroyed;
    // the static names are NUL-terminated for the duration of the process.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, VERSIONED_USED_NAME.as_ptr()) != 0 {
            return;
        }
        let ptr = ffi::PyCapsule_GetPointer(capsule, VERSIONED_NAME.as_ptr())
            .cast::<DLManagedTensorVersioned>();
        if !ptr.is_null() {
            if let Some(deleter) = (*ptr).deleter {
                deleter(ptr);
            }
        }
    }
}

unsafe extern "C" fn legacy_internal_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: only `from_dlpack` creates capsules with this name, and their
    // pointer is the still-live producer-owned `DLManagedTensor`.
    unsafe {
        let ptr = ffi::PyCapsule_GetPointer(capsule, LEGACY_INTERNAL_NAME.as_ptr())
            .cast::<DLManagedTensor>();
        if !ptr.is_null() {
            if let Some(deleter) = (*ptr).deleter {
                deleter(ptr);
            }
        }
    }
}

unsafe extern "C" fn versioned_internal_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: only `from_dlpack` creates capsules with this name, and their
    // pointer is the still-live producer-owned `DLManagedTensorVersioned`.
    unsafe {
        let ptr = ffi::PyCapsule_GetPointer(capsule, VERSIONED_INTERNAL_NAME.as_ptr())
            .cast::<DLManagedTensorVersioned>();
        if !ptr.is_null() {
            if let Some(deleter) = (*ptr).deleter {
                deleter(ptr);
            }
        }
    }
}

#[derive(Default)]
struct Registry {
    import: HashMap<(u8, u8), Descr>,
    export: HashMap<Descr, (u8, u8)>,
}

static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();

fn registry() -> &'static Mutex<Registry> {
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

fn builtin_dl_dtype(dtype: DType) -> Option<(u8, u8)> {
    Some(match dtype {
        DType::Bool => (K_DL_BOOL, 8),
        DType::I8 => (K_DL_INT, 8),
        DType::I16 => (K_DL_INT, 16),
        DType::I32 => (K_DL_INT, 32),
        DType::I64 => (K_DL_INT, 64),
        DType::U8 => (K_DL_UINT, 8),
        DType::U16 => (K_DL_UINT, 16),
        DType::U32 => (K_DL_UINT, 32),
        DType::U64 => (K_DL_UINT, 64),
        DType::F16 => (K_DL_FLOAT, 16),
        DType::F32 => (K_DL_FLOAT, 32),
        DType::F64 => (K_DL_FLOAT, 64),
        DType::C64 => (K_DL_COMPLEX, 64),
        DType::C128 => (K_DL_COMPLEX, 128),
        _ => return None,
    })
}

fn builtin_descr(code: u8, bits: u8) -> Option<Descr> {
    let dt = match (code, bits) {
        (K_DL_BOOL, 8) => DType::Bool,
        (K_DL_INT, 8) => DType::I8,
        (K_DL_INT, 16) => DType::I16,
        (K_DL_INT, 32) => DType::I32,
        (K_DL_INT, 64) => DType::I64,
        (K_DL_UINT, 8) => DType::U8,
        (K_DL_UINT, 16) => DType::U16,
        (K_DL_UINT, 32) => DType::U32,
        (K_DL_UINT, 64) => DType::U64,
        (K_DL_FLOAT, 16) => DType::F16,
        (K_DL_FLOAT, 32) => DType::F32,
        (K_DL_FLOAT, 64) => DType::F64,
        (K_DL_COMPLEX, 64) => DType::C64,
        (K_DL_COMPLEX, 128) => DType::C128,
        _ => return None,
    };
    Some(Descr::native(dt))
}

fn export_dtype(descr: Descr) -> PyResult<DLDataType> {
    if !descr.isnative() {
        return Err(PyBufferError::new_err(
            "DLPack only supports native byte order.",
        ));
    }
    let pair = builtin_dl_dtype(descr.dt)
        .or_else(|| registry().lock().unwrap().export.get(&descr).copied());
    let Some((code, bits)) = pair else {
        return Err(PyBufferError::new_err(
            "DLPack only supports signed/unsigned integers, float and complex dtypes (or dtypes registered by third-party packages).",
        ));
    };
    Ok(DLDataType {
        code,
        bits,
        lanes: 1,
    })
}

fn parse_copy(copy: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    let Some(copy) = copy else { return Ok(false) };
    if copy.is_none() {
        return Ok(false);
    }
    if copy.cast::<PyString>().is_ok() {
        return Err(PyValueError::new_err(
            "strings are not allowed for 'copy' keyword. Use True/False/None instead.",
        ));
    }
    copy.is_truthy()
}

fn parse_device(device: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(device) = device else { return Ok(()) };
    if device.is_none() {
        return Ok(());
    }
    let tuple = device
        .cast::<PyTuple>()
        .map_err(|_| PyTypeError::new_err("dl_device must be a tuple"))?;
    if tuple.len() != 2 {
        return Err(PyTypeError::new_err(format!(
            "function takes exactly 2 arguments ({} given)",
            tuple.len()
        )));
    }
    let ty: i32 = tuple.get_item(0)?.extract()?;
    let id: i32 = tuple.get_item(1)?.extract()?;
    if (ty, id) != (K_DL_CPU, 0) {
        return Err(PyBufferError::new_err("unsupported device requested"));
    }
    Ok(())
}

fn tensor_for(arr: &NdArray, dtype: DLDataType) -> PyResult<(DLTensor, Vec<i64>, Vec<i64>)> {
    let itemsize = arr.itemsize() as isize;
    if !arr.flags.c_contiguous && arr.size() != 1 {
        for (&dim, &stride) in arr.shape.iter().zip(&arr.strides) {
            if dim != 1 && stride % itemsize != 0 {
                return Err(PyBufferError::new_err(
                    "DLPack only supports strides which are a multiple of itemsize.",
                ));
            }
        }
    }
    let shape: Vec<i64> = arr.shape.iter().map(|&x| x as i64).collect();
    let strides: Vec<i64> = arr.strides.iter().map(|&x| (x / itemsize) as i64).collect();
    // SAFETY: `byte_offset` is the in-bounds address of logical element zero;
    // the exported owner reference keeps the underlying buffer alive.
    let data = unsafe { arr.buffer.as_ptr().offset(arr.byte_offset) } as *mut c_void;
    let tensor = DLTensor {
        data,
        device: DLDevice {
            device_type: K_DL_CPU,
            device_id: 0,
        },
        ndim: arr.ndim() as i32,
        dtype,
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    };
    Ok((tensor, shape, strides))
}

pub fn export(
    slf: &Bound<'_, PyNdArray>,
    stream: Option<&Bound<'_, PyAny>>,
    max_version: Option<&Bound<'_, PyAny>>,
    dl_device: Option<&Bound<'_, PyAny>>,
    copy: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if stream.is_some_and(|s| !s.is_none()) {
        return Err(PyValueError::new_err("NumPy only supports stream=None."));
    }
    parse_device(dl_device)?;
    let major = match max_version {
        None => 0,
        Some(v) if v.is_none() => 0,
        Some(v) => {
            let tuple = v.cast::<PyTuple>().map_err(|_| {
                PyTypeError::new_err("max_version must be None or a tuple with two elements.")
            })?;
            if tuple.len() != 2 {
                return Err(PyTypeError::new_err(
                    "max_version must be None or a tuple with two elements.",
                ));
            }
            tuple.get_item(0)?.extract::<i64>()?
        }
    };
    let copied = parse_copy(copy)?;
    let py = slf.py();
    let (arr, owner): (NdArray, Py<PyAny>) = if copied {
        let arr = slf.borrow().arr.copy();
        let owner = Py::new(py, PyNdArray::wrap(arr.clone()))?.into_any();
        (arr, owner)
    } else {
        (slf.borrow().arr.clone(), slf.clone().into_any().unbind())
    };
    if major < 1 && !arr.flags.writeable {
        return Err(PyBufferError::new_err(
            "Cannot export readonly array since signalling readonly is unsupported by DLPack (supported by newer DLPack version).",
        ));
    }
    let dtype = export_dtype(arr.descr)?;
    let (tensor, shape, strides) = tensor_for(&arr, dtype)?;

    if major >= 1 {
        let mut boxed = Box::new(VersionedExport {
            managed: DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(versioned_export_deleter),
                flags: (if arr.flags.writeable { 0 } else { READ_ONLY })
                    | (if copied { IS_COPIED } else { 0 }),
                dl_tensor: tensor,
            },
            shape,
            strides,
            owner: owner.into_ptr(),
        });
        if !boxed.shape.is_empty() {
            boxed.managed.dl_tensor.shape = boxed.shape.as_mut_ptr();
            boxed.managed.dl_tensor.strides = boxed.strides.as_mut_ptr();
        }
        let raw = NonNull::new(Box::into_raw(boxed).cast::<c_void>()).unwrap();
        // SAFETY: `raw` is a live Box allocation whose first field is the
        // versioned ABI struct; the registered destructor owns and frees it.
        let capsule = unsafe {
            PyCapsule::new_with_pointer_and_destructor(
                py,
                raw,
                VERSIONED_NAME,
                Some(versioned_capsule_destructor),
            )
        };
        match capsule {
            Ok(c) => Ok(c.into_any().unbind()),
            Err(e) => {
                // SAFETY: capsule construction failed, so ownership of `raw`
                // never transferred to CPython and must be reclaimed here.
                unsafe { drop_versioned_export(raw.as_ptr().cast::<VersionedExport>()) };
                Err(e)
            }
        }
    } else {
        let mut boxed = Box::new(LegacyExport {
            managed: DLManagedTensor {
                dl_tensor: tensor,
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(legacy_export_deleter),
            },
            shape,
            strides,
            owner: owner.into_ptr(),
        });
        if !boxed.shape.is_empty() {
            boxed.managed.dl_tensor.shape = boxed.shape.as_mut_ptr();
            boxed.managed.dl_tensor.strides = boxed.strides.as_mut_ptr();
        }
        let raw = NonNull::new(Box::into_raw(boxed).cast::<c_void>()).unwrap();
        // SAFETY: `raw` is a live Box allocation whose first field is the
        // legacy ABI struct; the registered destructor owns and frees it.
        let capsule = unsafe {
            PyCapsule::new_with_pointer_and_destructor(
                py,
                raw,
                LEGACY_NAME,
                Some(legacy_capsule_destructor),
            )
        };
        match capsule {
            Ok(c) => Ok(c.into_any().unbind()),
            Err(e) => {
                // SAFETY: capsule construction failed, so ownership of `raw`
                // never transferred to CPython and must be reclaimed here.
                unsafe { drop_legacy_export(raw.as_ptr().cast::<LegacyExport>()) };
                Err(e)
            }
        }
    }
}

struct DLPackOwner(#[allow(dead_code)] Py<PyAny>);
impl ForeignOwner for DLPackOwner {}

fn import_descr(dtype: DLDataType) -> PyResult<Descr> {
    if dtype.lanes != 1 {
        return Err(PyBufferError::new_err(
            "Unsupported lanes in DLTensor dtype.",
        ));
    }
    builtin_descr(dtype.code, dtype.bits)
        .or_else(|| {
            registry()
                .lock()
                .unwrap()
                .import
                .get(&(dtype.code, dtype.bits))
                .copied()
        })
        .ok_or_else(|| PyBufferError::new_err("Unsupported dtype in DLTensor."))
}

fn foreign_array(
    tensor: DLTensor,
    readonly: bool,
    internal: &Bound<'_, PyCapsule>,
) -> PyResult<NdArray> {
    if tensor.device.device_type != K_DL_CPU {
        return Err(PyBufferError::new_err("Unsupported device in DLTensor."));
    }
    if tensor.ndim < 0 || tensor.ndim as usize > MAX_DIMS {
        return Err(PyBufferError::new_err(
            "maxdims of DLPack tensor is higher than the supported maxdims.",
        ));
    }
    let ndim = tensor.ndim as usize;
    let descr = import_descr(tensor.dtype)?;
    let itemsize = descr.itemsize();
    let shape = if ndim == 0 {
        Vec::new()
    } else {
        if tensor.shape.is_null() {
            return Err(PyBufferError::new_err("DLPack tensor shape is NULL."));
        }
        // SAFETY: the producer promises `shape` points to at least `ndim`
        // int64 values for the managed tensor's full lifetime.
        unsafe { std::slice::from_raw_parts(tensor.shape, ndim) }
            .iter()
            .map(|&d| d as isize)
            .collect::<Vec<_>>()
    };
    if shape.iter().any(|&d| d < 0) {
        return Err(PyBufferError::new_err("negative dimension in DLTensor."));
    }
    let strides = if ndim == 0 {
        Vec::new()
    } else if tensor.strides.is_null() {
        c_strides(&shape, itemsize)
    } else {
        // SAFETY: the producer promises `strides` points to at least `ndim`
        // int64 values for the managed tensor's full lifetime.
        unsafe { std::slice::from_raw_parts(tensor.strides, ndim) }
            .iter()
            .map(|&s| s as isize * itemsize as isize)
            .collect::<Vec<_>>()
    };

    let empty = shape.iter().any(|&d| d == 0);
    let mut lower = 0isize;
    let mut upper = if empty { 0 } else { itemsize as isize };
    if !empty {
        for (&dim, &stride) in shape.iter().zip(&strides) {
            let step = (dim - 1) * stride;
            if step < 0 {
                lower += step;
            } else {
                upper += step;
            }
        }
    }
    let first = if tensor.data.is_null() {
        if !empty {
            return Err(PyBufferError::new_err("DLPack tensor data is NULL."));
        }
        NonNull::<u8>::dangling().as_ptr()
    } else {
        let base = tensor.data.cast::<u8>();
        // SAFETY: `byte_offset` is producer-supplied and denotes the first
        // logical element within the allocation promised by the DLTensor.
        unsafe { base.add(tensor.byte_offset as usize) }
    };
    // SAFETY: the DLTensor contract guarantees every address reachable from
    // `first` by `shape`/`strides`; `lower..upper` is exactly that byte span.
    let span_start = unsafe { first.offset(lower) };
    let owner = Box::new(DLPackOwner(internal.clone().into_any().unbind()));
    // SAFETY: `internal` owns the producer's managed tensor and deleter, so
    // the computed foreign span remains valid until this Buffer is dropped.
    let buffer =
        Arc::new(unsafe { Buffer::from_foreign(span_start, (upper - lower) as usize, owner) });
    let align = descr.alignment().max(1) as isize;
    let mut flags = Flags {
        c_contiguous: is_c_contiguous(&shape, &strides, itemsize),
        f_contiguous: is_f_contiguous(&shape, &strides, itemsize),
        writeable: !readonly,
        owndata: false,
        aligned: (first as usize).is_multiple_of(align as usize)
            && strides.iter().all(|s| s % align == 0),
    };
    if empty {
        flags.c_contiguous = true;
        flags.f_contiguous = true;
    }
    Ok(NdArray {
        buffer,
        byte_offset: -lower,
        shape,
        strides,
        descr,
        flags,
    })
}

#[pyfunction]
#[pyo3(signature = (obj, *, device = None, copy = None))]
pub fn from_dlpack(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    device: Option<&Bound<'_, PyAny>>,
    copy: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyNdArray>> {
    let dl_device = match device {
        None => py.None(),
        Some(d) if d.is_none() => py.None(),
        Some(d) => {
            if d.extract::<String>().ok().as_deref() != Some("cpu") {
                return Err(PyValueError::new_err(format!(
                    "Device not understood. Only \"cpu\" is allowed, but received: {}",
                    d.str()?
                )));
            }
            (K_DL_CPU, 0).into_pyobject(py)?.into_any().unbind()
        }
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("dl_device", dl_device)?;
    kwargs.set_item(
        "copy",
        copy.map_or_else(|| py.None(), |c| c.clone().unbind()),
    )?;
    kwargs.set_item("max_version", (1, 0))?;
    let capsule = match obj.call_method("__dlpack__", (), Some(&kwargs)) {
        Ok(c) => c,
        Err(e) if e.is_instance_of::<PyTypeError>(py) && device.is_none() && copy.is_none() => {
            obj.call_method0("__dlpack__")?
        }
        Err(e) => return Err(e),
    };

    let (managed_ptr, tensor, readonly, versioned) = {
        let ptr = capsule.as_ptr();
        // SAFETY: the validity checks establish both the capsule name and the
        // managed tensor ABI before any pointer is dereferenced.
        unsafe {
            if ffi::PyCapsule_IsValid(ptr, VERSIONED_NAME.as_ptr()) != 0 {
                let managed = ffi::PyCapsule_GetPointer(ptr, VERSIONED_NAME.as_ptr())
                    .cast::<DLManagedTensorVersioned>();
                if managed.is_null() {
                    return Err(PyErr::fetch(py));
                }
                if (*managed).version.major > 1 {
                    return Err(PyBufferError::new_err(
                        "from_dlpack(): the exported DLPack major version is too high to be imported by this version of NumPy.",
                    ));
                }
                (
                    managed.cast::<c_void>(),
                    (*managed).dl_tensor,
                    (*managed).flags & READ_ONLY != 0,
                    true,
                )
            } else if ffi::PyCapsule_IsValid(ptr, LEGACY_NAME.as_ptr()) != 0 {
                let managed =
                    ffi::PyCapsule_GetPointer(ptr, LEGACY_NAME.as_ptr()).cast::<DLManagedTensor>();
                if managed.is_null() {
                    return Err(PyErr::fetch(py));
                }
                (managed.cast::<c_void>(), (*managed).dl_tensor, true, false)
            } else {
                return Err(PyTypeError::new_err(
                    "PyCapsule_GetPointer called with incorrect name",
                ));
            }
        }
    };

    let raw = NonNull::new(managed_ptr).unwrap();
    let (internal_name, internal_destructor, used_name) = if versioned {
        (
            VERSIONED_INTERNAL_NAME,
            versioned_internal_destructor as ffi::PyCapsule_Destructor,
            VERSIONED_USED_NAME,
        )
    } else {
        (
            LEGACY_INTERNAL_NAME,
            legacy_internal_destructor as ffi::PyCapsule_Destructor,
            LEGACY_USED_NAME,
        )
    };
    // SAFETY: `raw` was obtained from a valid DLPack capsule. Ownership is
    // transferred to this internal capsule after the source is renamed used.
    let internal = unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            raw,
            internal_name,
            Some(internal_destructor),
        )?
    };
    // SAFETY: `capsule` is live and was validated above; `used_name` is a
    // process-static C string. Renaming is the DLPack ownership transfer.
    let rename_rc = unsafe { ffi::PyCapsule_SetName(capsule.as_ptr(), used_name.as_ptr()) };
    if rename_rc != 0 {
        // Avoid having both the source and internal capsule call the deleter.
        // SAFETY: `internal` is a live capsule created immediately above.
        unsafe { ffi::PyCapsule_SetDestructor(internal.as_ptr(), None) };
        return Err(PyErr::fetch(py));
    }
    let arr = foreign_array(tensor, readonly, &internal)?;
    let base = internal.clone().into_any().unbind();
    Ok(Py::new(
        py,
        PyNdArray {
            arr,
            base: Some(base),
        },
    )?)
}

#[pyfunction]
pub fn _register_dlpack_dtype(key: &Bound<'_, PyAny>, dtype: &Bound<'_, PyAny>) -> PyResult<()> {
    let tuple = key
        .cast::<PyTuple>()
        .map_err(|_| PyTypeError::new_err("dlpack_key must be a tuple of (code, bits)"))?;
    if tuple.len() != 2 {
        return Err(PyTypeError::new_err(
            "dlpack_key must be a tuple of (code, bits)",
        ));
    }
    let code: i64 = tuple.get_item(0)?.extract()?;
    let bits: i64 = tuple.get_item(1)?.extract()?;
    if !(0..=255).contains(&code) {
        return Err(PyValueError::new_err(
            "register_dlpack_dtype: DLPack code must be in 0..255.",
        ));
    }
    if !(0..=255).contains(&bits) {
        return Err(PyValueError::new_err(
            "register_dlpack_dtype: DLPack bits must be in 0..255.",
        ));
    }
    let dtype = dtype.cast::<PyDType>().map_err(|_| {
        PyTypeError::new_err("register_dlpack_dtype: dtype must be a numpy.dtype instance")
    })?;
    let descr = dtype.borrow().d;
    let pair = (code as u8, bits as u8);
    if descr.itemsize() * 8 != bits as usize {
        return Err(PyValueError::new_err(
            "register_dlpack_dtype: DLPack bits must match the dtype itemsize.",
        ));
    }
    if let Some(existing) = builtin_descr(pair.0, pair.1) {
        if existing != descr {
            return Err(PyValueError::new_err(
                "register_dlpack_dtype: DLPack (code, bits) already maps to a different dtype.",
            ));
        }
    }
    let mut reg = registry().lock().unwrap();
    if reg.export.get(&descr).is_some_and(|&p| p != pair) {
        return Err(PyValueError::new_err(
            "register_dlpack_dtype: dtype is already exported with a different DLPack (code, bits).",
        ));
    }
    // NumPy installs the export direction before validating the reverse
    // mapping.  A reverse-map conflict therefore raises but deliberately
    // leaves this dtype exportable (test_register_conflict depends on it).
    reg.export.insert(descr, pair);
    if reg.import.get(&pair).is_some_and(|&d| d != descr) {
        return Err(PyValueError::new_err(
            "register_dlpack_dtype: DLPack (code, bits) already maps to a different dtype.",
        ));
    }
    reg.import.insert(pair, descr);
    Ok(())
}

#[pyfunction]
pub fn _dlpack_registry_replace<'py>(
    py: Python<'py>,
    import: &Bound<'py, PyDict>,
    export: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let mut new_import = HashMap::new();
    for (key, value) in import.iter() {
        let tuple = key.cast::<PyTuple>()?;
        let code: u8 = tuple.get_item(0)?.extract()?;
        let bits: u8 = tuple.get_item(1)?.extract()?;
        new_import.insert((code, bits), value.cast::<PyDType>()?.borrow().d);
    }
    let mut new_export = HashMap::new();
    for (key, value) in export.iter() {
        let descr = key.cast::<PyDType>()?.borrow().d;
        let tuple = value.cast::<PyTuple>()?;
        let code: u8 = tuple.get_item(0)?.extract()?;
        let bits: u8 = tuple.get_item(1)?.extract()?;
        new_export.insert(descr, (code, bits));
    }
    let mut reg = registry().lock().unwrap();
    let old = std::mem::replace(
        &mut *reg,
        Registry {
            import: new_import,
            export: new_export,
        },
    );
    let old_import = PyDict::new(py);
    for ((code, bits), descr) in old.import {
        old_import.set_item((code, bits), PyDType::from_descr(descr))?;
    }
    let old_export = PyDict::new(py);
    for (descr, pair) in old.export {
        old_export.set_item(PyDType::from_descr(descr), pair)?;
    }
    Ok((old_import, old_export))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(from_dlpack, module)?)?;
    module.add_function(wrap_pyfunction!(_register_dlpack_dtype, module)?)?;
    module.add_function(wrap_pyfunction!(_dlpack_registry_replace, module)?)?;
    Ok(())
}
