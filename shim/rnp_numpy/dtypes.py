"""numpy.dtypes equivalent: the per-dtype DType classes.

Real numpy exposes one concrete class per dtype (`np.dtypes.Int64DType`).
Until the port grows a class hierarchy, each name here is a thin factory that
produces the corresponding `dtype` instance.
"""

from . import _dtype_table, dtype

__all__ = []


def register_dlpack_dtype(dlpack_key, dtype, /):
    """Register a third-party ``(DLPack code, bits)`` dtype mapping."""
    from _rnp import _register_dlpack_dtype
    return _register_dlpack_dtype(dlpack_key, dtype)


__all__.append("register_dlpack_dtype")


class _DTypeMeta(type):
    def __repr__(cls):
        return f"<class 'numpy.dtypes.{cls.__name__}'>"


def _make(name, dt):
    def __new__(cls):
        return dt

    cls = _DTypeMeta(name, (), {"_dtype": dt, "__new__": __new__,
                                "name": dt.name, "type": dt})
    globals()[name] = cls
    __all__.append(name)
    return cls


_CLASS_NAMES = {
    "bool": "BoolDType",
    "int8": "Int8DType", "int16": "Int16DType",
    "int32": "Int32DType", "int64": "Int64DType",
    "uint8": "UInt8DType", "uint16": "UInt16DType",
    "uint32": "UInt32DType", "uint64": "UInt64DType",
    "float16": "Float16DType",
    "float32": "Float32DType", "float64": "Float64DType",
    "complex64": "Complex64DType", "complex128": "Complex128DType",
}

for _name, _dt in _dtype_table().items():
    _make(_CLASS_NAMES[_name], _dt)

# Aliases numpy also exports on 64-bit platforms.
ByteDType = Int8DType  # noqa: F821
ShortDType = Int16DType  # noqa: F821
IntDType = Int32DType  # noqa: F821
LongDType = Int64DType  # noqa: F821
LongLongDType = Int64DType  # noqa: F821
UByteDType = UInt8DType  # noqa: F821
UShortDType = UInt16DType  # noqa: F821
UIntDType = UInt32DType  # noqa: F821
ULongDType = UInt64DType  # noqa: F821
ULongLongDType = UInt64DType  # noqa: F821
HalfDType = Float16DType  # noqa: F821

# The flexible dtypes are parameterised, so their classes take a size.
_make("BytesDType", dtype("S"))
_make("StrDType", dtype("U"))
_make("VoidDType", dtype("V"))
_make("ObjectDType", dtype("O"))
# The datetime classes are parameterised by unit, so the class stands for the
# whole family; the engine reports the family name in its error payloads.
_make("DateTime64DType", dtype("M8"))
_make("TimeDelta64DType", dtype("m8"))
_CLASS_NAMES.update({
    "datetime64": "DateTime64DType",
    "timedelta64": "TimeDelta64DType",
    "bytes": "BytesDType", "str": "StrDType", "void": "VoidDType",
    "object": "ObjectDType",
})
