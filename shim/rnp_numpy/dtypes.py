"""Concrete DType classes backed by the Rust descriptor storage class.

The extension owns descriptor data and operations in ``_rnp.dtype``.  This
module supplies NumPy's public class graph: an abstract ``numpy.dtype`` base
and one final ``numpy.dtypes.*DType`` subclass for each builtin family.
Instances remain genuine ``_rnp.dtype`` objects, so Rust extraction does not
need a proxy or an unwrap path.
"""

import inspect as _inspect

import _rnp
from _rnp import _string_dtype

_raw_dtype = _rnp.dtype

__all__ = ["register_dlpack_dtype"]


def register_dlpack_dtype(dlpack_key, dtype, /):
    """Register a third-party ``(DLPack code, bits)`` dtype mapping."""
    from _rnp import _register_dlpack_dtype
    return _register_dlpack_dtype(dlpack_key, dtype)


class _DTypeMeta(type):
    """Metaclass shared by the abstract base and all builtin DType classes."""

    def __new__(mcls, name, bases, namespace, **kwargs):
        if any(getattr(base, "_final_dtype", False) for base in bases):
            raise TypeError("Preliminary-API: Cannot subclass DType.")
        return super().__new__(mcls, name, bases, namespace, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls is dtype:
            if (len(args) == 1 and isinstance(args[0], dtype)
                    and not kwargs):
                return args[0]
            return _concrete(_raw_dtype(*args, **kwargs))
        return super().__call__(*args, **kwargs)


_DTypeMeta.__module__ = "numpy"


def _signature(*parameters):
    return _inspect.Signature(parameters)


class dtype(_raw_dtype, metaclass=_DTypeMeta):
    """Public abstract dtype base; descriptor storage lives in ``_rnp``."""

    __module__ = "numpy"
    _abstract = True
    _parametric = False
    _is_numeric = False

    def newbyteorder(self, new_order="S", /):
        return _allocate(type(self), _raw_dtype.newbyteorder(self, new_order))

    @property
    def base(self):
        return _concrete(_raw_dtype.base.__get__(self, _raw_dtype))

    @property
    def fields(self):
        fields = _raw_dtype.fields.__get__(self, _raw_dtype)
        if fields is None:
            return None
        out = {}
        for key, value in fields.items():
            out[key] = (_concrete(value[0]), *value[1:])
        return out

    @property
    def subdtype(self):
        subdtype = _raw_dtype.subdtype.__get__(self, _raw_dtype)
        if subdtype is None:
            return None
        return _concrete(subdtype[0]), subdtype[1]

    def __getitem__(self, key):
        return _concrete(_raw_dtype.__getitem__(self, key))

    def __reduce__(self):
        metadata = self.metadata
        metadata = None if metadata is None else dict(metadata)
        if self.char in "qQgG":
            prefix = "" if self.byteorder == "|" else self.byteorder
            spec = prefix + self.char
        else:
            spec = self.str
        return _reconstruct_dtype, (spec, metadata)


dtype.__signature__ = _signature(
    _inspect.Parameter("dtype", _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    _inspect.Parameter("align", _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       default=False),
    _inspect.Parameter("copy", _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                       default=False),
    _inspect.Parameter("kwargs", _inspect.Parameter.VAR_KEYWORD),
)


def _reconstruct_dtype(spec, metadata):
    if metadata is None:
        return dtype(spec)
    return dtype(spec, metadata=metadata)


_CLASS_BY_CHAR = {}


def _allocate(cls, raw):
    return _raw_dtype.__new__(cls, raw)


def _simple_new(cls):
    return cls._singleton


def _make_simple(name, spec, *, numeric):
    raw = _raw_dtype(spec)
    cls = _DTypeMeta(name, (dtype,), {
        "__module__": "numpy.dtypes",
        "__new__": _simple_new,
        "_abstract": False,
        "_parametric": False,
        "_is_numeric": numeric,
        "_raw_default": raw,
        "dtype": raw,
        "type": raw.type,
    })
    cls.__signature__ = _signature()
    cls._singleton = _allocate(cls, raw)
    cls._final_dtype = True
    globals()[name] = cls
    __all__.append(name)
    _CLASS_BY_CHAR[raw.char] = cls
    return cls


def _make_parametric(name, spec, parameter, constructor, *, discovery=None):
    raw = _raw_dtype(spec)

    def __new__(cls, value):
        return _allocate(cls, _raw_dtype(constructor(value)))

    cls = _DTypeMeta(name, (dtype,), {
        "__module__": "numpy.dtypes",
        "__new__": __new__,
        "_abstract": False,
        "_parametric": True,
        "_is_numeric": False,
        "_raw_default": raw,
        "_default_discovery": (_raw_dtype(discovery)
                               if discovery is not None else raw),
        "dtype": raw,
        "type": raw.type,
    })
    cls.__signature__ = _signature(
        _inspect.Parameter(parameter, _inspect.Parameter.POSITIONAL_ONLY))
    cls._final_dtype = True
    globals()[name] = cls
    __all__.append(name)
    _CLASS_BY_CHAR[raw.char] = cls
    return cls


# Keep the same order as NumPy's public ``numpy.dtypes.__all__``.
BoolDType = _make_simple("BoolDType", "?", numeric=True)
Int8DType = _make_simple("Int8DType", "b", numeric=True)
ByteDType = Int8DType
__all__.append("ByteDType")
UInt8DType = _make_simple("UInt8DType", "B", numeric=True)
UByteDType = UInt8DType
__all__.append("UByteDType")
Int16DType = _make_simple("Int16DType", "h", numeric=True)
ShortDType = Int16DType
__all__.append("ShortDType")
UInt16DType = _make_simple("UInt16DType", "H", numeric=True)
UShortDType = UInt16DType
__all__.append("UShortDType")
Int32DType = _make_simple("Int32DType", "i", numeric=True)
IntDType = Int32DType
__all__.append("IntDType")
UInt32DType = _make_simple("UInt32DType", "I", numeric=True)
UIntDType = UInt32DType
__all__.append("UIntDType")
Int64DType = _make_simple("Int64DType", "l", numeric=True)
LongDType = Int64DType
__all__.append("LongDType")
UInt64DType = _make_simple("UInt64DType", "L", numeric=True)
ULongDType = UInt64DType
__all__.append("ULongDType")
LongLongDType = _make_simple("LongLongDType", "q", numeric=True)
ULongLongDType = _make_simple("ULongLongDType", "Q", numeric=True)
Float16DType = _make_simple("Float16DType", "e", numeric=True)
Float32DType = _make_simple("Float32DType", "f", numeric=True)
Float64DType = _make_simple("Float64DType", "d", numeric=True)
LongDoubleDType = _make_simple("LongDoubleDType", "g", numeric=True)
Complex64DType = _make_simple("Complex64DType", "F", numeric=True)
Complex128DType = _make_simple("Complex128DType", "D", numeric=True)
CLongDoubleDType = _make_simple("CLongDoubleDType", "G", numeric=True)
ObjectDType = _make_simple("ObjectDType", "O", numeric=False)

BytesDType = _make_parametric(
    "BytesDType", "S", "size", lambda n: f"S{n}", discovery="S1")
StrDType = _make_parametric(
    "StrDType", "U", "size", lambda n: f"U{n}", discovery="U1")
VoidDType = _make_parametric(
    "VoidDType", "V", "length", lambda n: f"V{n}", discovery="V8")
DateTime64DType = _make_parametric(
    "DateTime64DType", "M8", "unit", lambda unit: f"M8[{unit}]")
TimeDelta64DType = _make_parametric(
    "TimeDelta64DType", "m8", "unit", lambda unit: f"m8[{unit}]")


class StringDType(dtype):
    """Variable-width UTF-8 string dtype (NEP 55)."""

    __module__ = "numpy.dtypes"
    _abstract = False
    _parametric = True
    _is_numeric = False
    dtype = _raw_dtype("T")
    type = str

    def __new__(cls, *, coerce=True, **kwargs):
        unknown = set(kwargs) - {"na_object"}
        if unknown:
            name = next(iter(unknown))
            raise TypeError(f"StringDType() got an unexpected keyword argument {name!r}")
        has_na = "na_object" in kwargs
        raw = _string_dtype(bool(coerce), has_na,
                            None if not has_na else kwargs["na_object"])
        return _allocate(cls, raw)

    def __reduce__(self):
        try:
            return _reconstruct_string_dtype, (self.coerce, True, self.na_object)
        except AttributeError:
            return _reconstruct_string_dtype, (self.coerce, False, None)


StringDType.__signature__ = _signature(
    _inspect.Parameter("coerce", _inspect.Parameter.KEYWORD_ONLY,
                       default=True),
    _inspect.Parameter("kwargs", _inspect.Parameter.VAR_KEYWORD),
)
StringDType._raw_default = StringDType.dtype
StringDType._default_discovery = StringDType.dtype
StringDType._final_dtype = True
__all__.append("StringDType")
_CLASS_BY_CHAR[StringDType.dtype.char] = StringDType

# Engine error payloads use canonical storage names. Keep this compatibility
# table for the exception factory while concrete identity itself is keyed by
# descriptor ``char`` so C aliases remain distinct.
_CLASS_NAMES = {
    "bool": "BoolDType",
    "int8": "Int8DType", "int16": "Int16DType",
    "int32": "Int32DType", "int64": "Int64DType",
    "uint8": "UInt8DType", "uint16": "UInt16DType",
    "uint32": "UInt32DType", "uint64": "UInt64DType",
    "float16": "Float16DType", "float32": "Float32DType",
    "float64": "Float64DType", "float128": "LongDoubleDType",
    "complex64": "Complex64DType", "complex128": "Complex128DType",
    "complex256": "CLongDoubleDType", "bytes": "BytesDType",
    "str": "StrDType", "void": "VoidDType", "object": "ObjectDType",
    "datetime64": "DateTime64DType", "timedelta64": "TimeDelta64DType",
}


def _reconstruct_string_dtype(coerce, has_na, na_object):
    if has_na:
        return StringDType(na_object=na_object, coerce=coerce)
    return StringDType(coerce=coerce)


def _concrete(value):
    """Return *value* with the concrete builtin DType Python class."""
    if isinstance(value, dtype):
        return value
    if not isinstance(value, _raw_dtype):
        value = _raw_dtype(value)
    cls = _CLASS_BY_CHAR.get(value.char, VoidDType)
    if (not cls._parametric and value.byteorder in ("=", "|")
            and value.metadata is None and value.names is None
            and value.subdtype is None):
        return cls._singleton
    return _allocate(cls, value)


def _default_for_discovery(cls):
    raw = getattr(cls, "_default_discovery", None)
    if raw is None:
        return cls()
    return _concrete(raw)
