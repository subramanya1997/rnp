"""rnp_numpy — a numpy-shaped surface backed by the Rust engine in `_rnp`.

The harness maps ``import numpy`` onto this package so NumPy's own tests can
run unmodified against the port. At M0 only a small fraction of the surface
exists; everything present is backed by `_rnp` rather than reimplemented in
Python, except for a few thin conveniences at the bottom of this file.
"""

import builtins as _builtins

from _rnp import (
    add,
    arange,
    array,
    asarray,
    broadcast_shapes,
    divide,
    dtype,
    empty,
    equal,
    full,
    greater,
    greater_equal,
    less,
    less_equal,
    multiply,
    ndarray,
    not_equal,
    ones,
    promote_types,
    result_type,
    shape,
    subtract,
    true_divide,
    zeros,
)
from _rnp import _dtype_table as _dtype_table

from . import exceptions
from .exceptions import AxisError, ComplexWarning, DTypePromotionError, VisibleDeprecationWarning

__version__ = "2.5.2"

newaxis = None
pi = 3.141592653589793
e = 2.718281828459045
euler_gamma = 0.5772156649015329
inf = float("inf")
nan = float("nan")
Inf = inf
NAN = nan
NaN = nan


# --------------------------------------------------------------------------
# Scalar type aliases.
#
# Real numpy exposes these as scalar *classes*. Until the port grows real
# scalar types (a later milestone), each alias is a callable class carrying a
# `dtype` attribute, which is enough for `dtype=np.int32`, `np.int32(3)` and
# `np.dtype(np.int32)` to behave.
# --------------------------------------------------------------------------

_DTYPES = _dtype_table()


class _ScalarMeta(type):
    def __repr__(cls):
        return f"<class 'numpy.{cls.__name__}'>"

    def __eq__(cls, other):
        own = getattr(cls, "dtype", None)
        if own is not None and isinstance(other, (dtype, str)):
            return own == other
        return type.__eq__(cls, other)

    def __ne__(cls, other):
        return not cls.__eq__(other)

    def __hash__(cls):
        return type.__hash__(cls)


# The abstract scalar hierarchy. Real numpy scalars land in a later
# milestone; these exist so that `np.integer`-style checks and third-party
# code (hypothesis.extra.numpy) can at least import.
class generic(metaclass=_ScalarMeta):
    pass


class number(generic):
    pass


class integer(number):
    pass


class signedinteger(integer):
    pass


class unsignedinteger(integer):
    pass


class inexact(number):
    pass


class floating(inexact):
    pass


class complexfloating(inexact):
    pass


class flexible(generic):
    pass


class character(flexible):
    pass


def _make_scalar_type(name, dt, base=generic):
    def __new__(cls, value=0):
        # numpy scalars are not implemented yet; return the Python value the
        # cast produces so arithmetic still works.
        return array(value, dtype=cls.dtype).item()

    return _ScalarMeta(name, (base,), {"dtype": dt, "__new__": __new__})


bool_ = _make_scalar_type("bool_", _DTYPES["bool"], generic)
int8 = _make_scalar_type("int8", _DTYPES["int8"], signedinteger)
int16 = _make_scalar_type("int16", _DTYPES["int16"], signedinteger)
int32 = _make_scalar_type("int32", _DTYPES["int32"], signedinteger)
int64 = _make_scalar_type("int64", _DTYPES["int64"], signedinteger)
uint8 = _make_scalar_type("uint8", _DTYPES["uint8"], unsignedinteger)
uint16 = _make_scalar_type("uint16", _DTYPES["uint16"], unsignedinteger)
uint32 = _make_scalar_type("uint32", _DTYPES["uint32"], unsignedinteger)
uint64 = _make_scalar_type("uint64", _DTYPES["uint64"], unsignedinteger)
float32 = _make_scalar_type("float32", _DTYPES["float32"], floating)
float64 = _make_scalar_type("float64", _DTYPES["float64"], floating)
complex64 = _make_scalar_type("complex64", _DTYPES["complex64"], complexfloating)
complex128 = _make_scalar_type("complex128", _DTYPES["complex128"], complexfloating)

# Platform aliases, matching numpy on 64-bit Linux/macOS.
byte = int8
short = int16
intc = int32
int_ = int64
long = int64
longlong = int64
intp = int64
ubyte = uint8
ushort = uint16
uintc = uint32
uint = uint64
ulong = uint64
ulonglong = uint64
uintp = uint64
single = float32
double = float64
float_ = float64
csingle = complex64
cdouble = complex128
complex_ = complex128
bool = bool_


def _unsupported_scalar_type(name, base=generic):
    """A scalar type the port does not implement yet.

    The name exists (module-level lookups in the test files succeed), but it
    carries no dtype, so `np.dtype(np.str_)` fails loudly rather than
    silently doing the wrong thing.
    """

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(f"numpy scalar type {name!r} is not "
                                  f"implemented by rnp yet")

    return _ScalarMeta(name, (base,), {"__new__": __new__})


str_ = _unsupported_scalar_type("str_", character)
unicode_ = str_
bytes_ = _unsupported_scalar_type("bytes_", character)
void = _unsupported_scalar_type("void", flexible)
object_ = _unsupported_scalar_type("object_")
datetime64 = _unsupported_scalar_type("datetime64")
timedelta64 = _unsupported_scalar_type("timedelta64", signedinteger)
float16 = _unsupported_scalar_type("float16", floating)
half = float16
longdouble = _unsupported_scalar_type("longdouble", floating)
clongdouble = _unsupported_scalar_type("clongdouble", complexfloating)
longfloat = longdouble
clongfloat = clongdouble

sctypeDict = {t.__name__: t for t in (
    bool_, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64, complex64, complex128,
)}


# --------------------------------------------------------------------------
# Thin Python-level conveniences.
# --------------------------------------------------------------------------

from ._core.shape_base import (  # noqa: E402
    atleast_1d,
    atleast_2d,
    atleast_3d,
    block,
    concatenate,
    hstack,
    stack,
    vstack,
)


def _asarr(a):
    return a if isinstance(a, ndarray) else asarray(a)


def unstack(x, /, *, axis=0):
    x = _asarr(x)
    if x.ndim == 0:
        raise ValueError("Input array must be at least 1-d.")
    index = [slice(None)] * x.ndim
    out = []
    for i in range(x.shape[axis]):
        index[axis] = i
        out.append(x[tuple(index)])
    return tuple(out)


def eye(N, M=None, k=0, dtype=float64):
    M = N if M is None else M
    out = zeros((N, M), dtype)
    for i in range(N):
        j = i + k
        if 0 <= j < M:
            out[i, j] = 1
    return out


def identity(n, dtype=float64):
    return eye(n, dtype=dtype)


def reshape(a, /, shape=None, **kwargs):
    return _asarr(a).reshape(shape if shape is not None else kwargs.pop("newshape"))


def transpose(a, axes=None):
    a = _asarr(a)
    return a.transpose() if axes is None else a.transpose(axes)


def ravel(a):
    return _asarr(a).ravel()


def copy(a):
    return _asarr(a).copy()


def zeros_like(a, dtype=None):
    a = _asarr(a)
    return zeros(a.shape, a.dtype if dtype is None else dtype)


def ones_like(a, dtype=None):
    a = _asarr(a)
    return ones(a.shape, a.dtype if dtype is None else dtype)


def empty_like(a, dtype=None):
    a = _asarr(a)
    return empty(a.shape, a.dtype if dtype is None else dtype)


def full_like(a, fill_value, dtype=None):
    a = _asarr(a)
    return full(a.shape, fill_value, a.dtype if dtype is None else dtype)


def size(a, axis=None):
    a = _asarr(a)
    return a.size if axis is None else a.shape[axis]


def ndim(a):
    return _asarr(a).ndim


def _flat_values(a):
    """Every element of `a` in C order, as Python scalars."""
    out = []
    stack = [_asarr(a).tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def sum(a, axis=None, dtype=None):
    """Whole-array sum only. Axis-wise reductions arrive in M4."""
    if axis is not None:
        raise NotImplementedError("axis= reductions are not implemented yet")
    return _builtins.sum(_flat_values(a))


def prod(a, axis=None):
    if axis is not None:
        raise NotImplementedError("axis= reductions are not implemented yet")
    acc = 1
    for v in _flat_values(a):
        acc *= v
    return acc


def max(a, axis=None):
    if axis is not None:
        raise NotImplementedError("axis= reductions are not implemented yet")
    return _builtins.max(_flat_values(a))


def min(a, axis=None):
    if axis is not None:
        raise NotImplementedError("axis= reductions are not implemented yet")
    return _builtins.min(_flat_values(a))


amax = max
amin = min


def all(a, axis=None):
    if axis is not None:
        raise NotImplementedError("axis= reductions are not implemented yet")
    return _builtins.all(_flat_values(a))


def any(a, axis=None):
    if axis is not None:
        raise NotImplementedError("axis= reductions are not implemented yet")
    return _builtins.any(_flat_values(a))


def array_equal(a1, a2):
    a1, a2 = _asarr(a1), _asarr(a2)
    if a1.shape != a2.shape:
        return False
    return _builtins.all(_flat_values(equal(a1, a2)))


def isscalar(x):
    return isinstance(x, (int, float, complex, str, bytes, _builtins.bool))


def can_cast(from_, to, casting="safe"):
    # Placeholder until the real casting table lands in M1.
    return dtype(to) == promote_types(from_, to)


class iinfo:
    """Machine limits for integer dtypes."""

    def __init__(self, int_type):
        self.dtype = dtype(int_type)
        if self.dtype.kind not in "iu":
            raise ValueError(f"Invalid integer data type {self.dtype.name!r}.")
        self.bits = self.dtype.itemsize * 8
        self.key = f"{self.dtype.kind}{self.dtype.itemsize}"

    @property
    def min(self):
        return 0 if self.dtype.kind == "u" else -(2 ** (self.bits - 1))

    @property
    def max(self):
        if self.dtype.kind == "u":
            return 2 ** self.bits - 1
        return 2 ** (self.bits - 1) - 1

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype.name})"


class finfo:
    """Machine limits for floating point dtypes."""

    _PARAMS = {
        "float32": dict(bits=32, eps=1.1920929e-07, max=3.4028235e+38,
                        min=-3.4028235e+38, tiny=1.1754944e-38, nmant=23,
                        nexp=8, precision=6, resolution=1e-06,
                        epsneg=5.9604645e-08, iexp=8, machep=-23, negep=-24,
                        maxexp=128, minexp=-126),
        "float64": dict(bits=64, eps=2.220446049250313e-16,
                        max=1.7976931348623157e+308,
                        min=-1.7976931348623157e+308,
                        tiny=2.2250738585072014e-308, nmant=52, nexp=11,
                        precision=15, resolution=1e-15,
                        epsneg=1.1102230246251565e-16, iexp=11, machep=-52,
                        negep=-53, maxexp=1024, minexp=-1022),
    }

    def __init__(self, float_type):
        d = dtype(float_type)
        if d.kind == "c":
            d = dtype("float32" if d.itemsize == 8 else "float64")
        if d.name not in self._PARAMS:
            raise ValueError(f"data type {d.name!r} not inexact")
        self.dtype = d
        for k, v in self._PARAMS[d.name].items():
            setattr(self, k, v)
        self.smallest_normal = self.tiny
        self.smallest_subnormal = 1e-45 if d.name == "float32" else 5e-324

    def __repr__(self):
        return f"finfo(resolution={self.resolution}, dtype={self.dtype.name})"


class errstate:
    """No-op stand-in: the port does not raise FP warnings yet."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, func):
        return func


def seterr(**kwargs):
    return {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


def geterr():
    return seterr()


__all__ = [n for n in dir() if not n.startswith("_")]
