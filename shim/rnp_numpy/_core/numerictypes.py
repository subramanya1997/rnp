"""`numpy._core.numerictypes`: the scalar-type registry helpers.

Only the pieces the port's dtype system can answer honestly live here; the
rest arrives with the real scalar hierarchy in a later milestone.
"""
import rnp_numpy as _np
from rnp_numpy import (  # noqa: F401
    bool_ as bool,
    byte,
    bytes_,
    cdouble,
    complex64,
    complex128,
    complexfloating,
    csingle,
    double,
    dtype,
    float16,
    float32,
    float64,
    floating,
    generic,
    half,
    inexact,
    int8,
    int16,
    int32,
    int64,
    int_,
    intc,
    integer,
    intp,
    long,
    longdouble,
    longlong,
    number,
    short,
    signedinteger,
    single,
    str_,
    ubyte,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
    uintc,
    uintp,
    ulong,
    ulonglong,
    clongdouble,
    unsignedinteger,
    ushort,
    void,
    typecodes,
    datetime64,
    timedelta64,
    object_,
)

sctypeDict = _np.sctypeDict

#: numpy's grouping of the concrete scalar types by category.
sctypes = _np.sctypes

ScalarType = _np.ScalarType


def issctype(rep):
    """True if `rep` names a valid scalar data type."""
    if not isinstance(rep, (type, dtype, str)):
        return False
    try:
        res = dtype(rep)
    except Exception:  # noqa: BLE001
        return False
    return res.kind != 'O'


def sctype2char(sctype):
    """The single-character type code for a scalar type."""
    d = dtype(sctype)
    return d.char


def obj2sctype(rep, default=None):
    try:
        return dtype(rep).type
    except Exception:  # noqa: BLE001
        return default


def issubdtype(arg1, arg2):
    return _np.issubdtype(arg1, arg2)
