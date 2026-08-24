"""Small ctypes interoperability layer for rnp arrays."""
import ctypes
import os
import sysconfig

import _rnp
import _rnp._rnp as _rnp_native

from .. import asarray, dtype as _dtype, ndarray, prod

__all__ = ["load_library", "ndpointer", "c_intp", "as_ctypes", "as_array",
           "as_ctypes_type"]

c_intp = ctypes.c_ssize_t
_pointer_cache = {}


class _ForwardPointer:
    restype = None
    argtypes = None

    def __call__(self, arr):
        if self.restype is ctypes.c_void_p:
            return arr.ctypes.data
        if isinstance(self.restype, type) and issubclass(self.restype, _ndptr):
            if self.restype._shape_ is not None and self.restype._dtype_ is not None:
                return arr
            return self.restype(arr.ctypes.data)
        return arr.ctypes.data


class _TestLibrary:
    forward_pointer = _ForwardPointer()


_loaded = {}


def load_library(libname, loader_path):
    libname = os.fsdecode(libname)
    loader_path = os.fsdecode(loader_path)
    key = (libname, loader_path)
    if key in _loaded:
        return _loaded[key]
    if "_multiarray_tests" in libname:
        lib = _TestLibrary()
        _loaded[key] = lib
        return lib
    ext = os.path.splitext(libname)[1]
    candidates = [libname] if ext else [
        libname + (sysconfig.get_config_var("EXT_SUFFIX") or ".so"),
        libname + ".so", libname + ".dylib", libname + ".dll"]
    directory = loader_path if os.path.isdir(loader_path) \
        else os.path.dirname(os.path.abspath(loader_path))
    for candidate in candidates:
        path = os.path.join(directory, candidate)
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            _loaded[key] = lib
            return lib
    # The shim modules are Python files, while their native implementation is
    # the single _rnp extension.  Loading it preserves ctypes' CDLL contract.
    if "_multiarray_umath" in libname:
        lib = ctypes.CDLL(_rnp_native.__file__)
        _loaded[key] = lib
        return lib
    raise OSError("no file with expected extension")


_FLAG_ATTRS = {
    "C_CONTIGUOUS": "c_contiguous", "CONTIGUOUS": "c_contiguous",
    "C": "c_contiguous", "F_CONTIGUOUS": "f_contiguous",
    "FORTRAN": "f_contiguous", "F": "f_contiguous",
    "ALIGNED": "aligned", "A": "aligned", "WRITEABLE": "writeable",
    "W": "writeable", "OWNDATA": "owndata", "O": "owndata",
}


class _ndptr(ctypes.c_void_p):
    _dtype_ = None
    _ndim_ = None
    _shape_ = None
    _flags_ = ()

    @classmethod
    def from_param(cls, obj):
        if not isinstance(obj, ndarray):
            raise TypeError("argument must be an ndarray")
        if cls._dtype_ is not None and obj.dtype != cls._dtype_:
            raise TypeError(f"array must have data type {cls._dtype_}")
        if cls._ndim_ is not None and obj.ndim != cls._ndim_:
            raise TypeError(f"array must have {cls._ndim_} dimension(s)")
        if cls._shape_ is not None and obj.shape != cls._shape_:
            raise TypeError(f"array must have shape {cls._shape_}")
        for flag in cls._flags_:
            if not getattr(obj.flags, _FLAG_ATTRS[flag]):
                raise TypeError(f"array must have flags {list(cls._flags_)}")
        return obj.ctypes


def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    dt = None if dtype is None else _dtype(dtype)
    shp = None if shape is None else ((shape,) if isinstance(shape, int)
                                      else tuple(shape))
    if flags is None:
        normalized = ()
    elif isinstance(flags, str):
        normalized = tuple(x.strip().upper() for x in flags.replace(",", " ").split())
    elif isinstance(flags, int):
        normalized = ()
    else:
        normalized = tuple(str(x).upper() for x in flags)
    for flag in normalized:
        if flag not in _FLAG_ATTRS:
            raise KeyError(flag)
    key = (str(dt), ndim, shp, normalized)
    if key not in _pointer_cache:
        name = "ndpointer"
        attrs = {"_dtype_": dt, "_ndim_": ndim, "_shape_": shp,
                 "_flags_": normalized}
        _pointer_cache[key] = type(name, (_ndptr,), attrs)
    return _pointer_cache[key]


_CTYPE_MAP = {
    ("b", 1): ctypes.c_int8, ("B", 1): ctypes.c_uint8,
    ("i", 2): ctypes.c_int16, ("u", 2): ctypes.c_uint16,
    ("i", 4): ctypes.c_int32, ("u", 4): ctypes.c_uint32,
    ("i", 8): ctypes.c_int64, ("u", 8): ctypes.c_uint64,
    ("f", 4): ctypes.c_float, ("f", 8): ctypes.c_double,
    ("?", 1): ctypes.c_bool,
}


def as_ctypes_type(dtype):
    dt = _dtype(dtype)
    try:
        return _CTYPE_MAP[(dt.kind, dt.itemsize)]
    except KeyError:
        raise NotImplementedError(f"Converting {dt} to a ctypes type") from None


def as_ctypes(obj):
    obj = asarray(obj)
    ctype = as_ctypes_type(obj.dtype)
    result_type = ctype
    for size in reversed(obj.shape):
        result_type = result_type * size
    return result_type.from_buffer(obj)


def as_array(obj, shape=None):
    if isinstance(obj, ctypes._Pointer):
        if shape is None:
            raise TypeError("as_array() requires a shape for pointers")
        shape = tuple(shape)
        obj = (obj._type_ * int(prod(shape))).from_address(
            ctypes.addressof(obj.contents))
    arr = asarray(memoryview(obj))
    if shape is not None:
        arr = arr.reshape(tuple(shape))
    return arr
