"""rnp_numpy — a numpy-shaped surface backed by the Rust engine in `_rnp`.

The harness maps ``import numpy`` onto this package so NumPy's own tests can
run unmodified against the port. At M0 only a small fraction of the surface
exists; everything present is backed by `_rnp` rather than reimplemented in
Python, except for a few thin conveniences at the bottom of this file.
"""

import builtins as _builtins
import enum as _enum

from _rnp import (
    add,
    amax,
    amin,
    arange,
    argmax,
    argmin,
    array,
    asarray,
    broadcast_shapes,
    broadcast_to,
    can_cast,
    choose,
    common_type,
    compress,
    divide,
    dtype,
    empty,
    equal,
    flatiter,
    flatnonzero,
    full,
    greater,
    greater_equal,
    less,
    less_equal,
    mean,
    min_scalar_type,
    multiply,
    ndarray,
    nonzero,
    not_equal,
    ones,
    prod,
    promote_types,
    put,
    putmask,
    result_type,
    shape,
    subtract,
    sum,
    take,
    true_divide,
    zeros,
)
from _rnp import where_ as where
from _rnp import _dtype_table as _dtype_table
import _rnp

from ._stubs import ufunc

# The implemented ufuncs, wrapped so that `isinstance(np.add, np.ufunc)` and
# `np.add.reduce(...)` behave the way the upstream tests expect.
add = ufunc("add", 2, 1, add, identity=0)
subtract = ufunc("subtract", 2, 1, subtract)
multiply = ufunc("multiply", 2, 1, multiply, identity=1)
divide = ufunc("divide", 2, 1, divide)
true_divide = ufunc("true_divide", 2, 1, true_divide)
equal = ufunc("equal", 2, 1, equal)
not_equal = ufunc("not_equal", 2, 1, not_equal)
less = ufunc("less", 2, 1, less)
less_equal = ufunc("less_equal", 2, 1, less_equal)
greater = ufunc("greater", 2, 1, greater)
greater_equal = ufunc("greater_equal", 2, 1, greater_equal)

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
float16 = _make_scalar_type("float16", _DTYPES["float16"], floating)
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
half = float16
single = float32
double = float64
float_ = float64
csingle = complex64
cdouble = complex128
complex_ = complex128
bool = bool_


class _InertScalar:
    """An instance of a scalar type the port does not implement.

    Upstream test modules *construct* these at import time (inside class
    bodies and parametrize lists), so construction has to succeed; every
    operation on the result raises NotImplementedError instead.
    """

    __slots__ = ("_args", "_kwargs")

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def _nope(self, *args, **kwargs):
        raise NotImplementedError(
            f"numpy scalar type {type(self).__name__!r} is not implemented "
            f"by rnp yet")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _nope
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = _nope
    __mod__ = __rmod__ = __divmod__ = __pow__ = __neg__ = __abs__ = _nope
    __int__ = __float__ = __complex__ = __index__ = _nope
    __array__ = _nope

    def __repr__(self):
        inner = ", ".join(repr(a) for a in self._args)
        return f"numpy.{type(self).__name__}({inner})"


def _unsupported_scalar_type(name, base=generic):
    """A scalar type the port does not implement yet.

    The name exists and can be instantiated (upstream modules do so at import
    time), but it carries no dtype and every operation on an instance raises,
    so nothing silently produces a wrong answer.
    """

    def __new__(cls, *args, **kwargs):
        return _InertScalar.__new__(_ScalarMeta(name, (_InertScalar,), {}))

    def __init__(self, *args, **kwargs):
        _InertScalar.__init__(self, *args, **kwargs)

    return _ScalarMeta(name, (base,), {"__new__": __new__,
                                       "__init__": __init__})


# The flexible scalar types exist as dtype *descriptors* (M1); their scalar
# behaviour (indexing an array of them returns a plain str/bytes for now)
# arrives with the real scalar hierarchy.
str_ = _make_scalar_type("str_", dtype("U"), character)
unicode_ = str_
bytes_ = _make_scalar_type("bytes_", dtype("S"), character)
void = _make_scalar_type("void", dtype("V"), flexible)
# `object` exists as a dtype descriptor only: arrays of it are rejected
# at creation time rather than silently mis-stored.
object_ = _make_scalar_type("object_", dtype("O"), generic)
datetime64 = _unsupported_scalar_type("datetime64")
timedelta64 = _unsupported_scalar_type("timedelta64", signedinteger)
# On this platform (macOS/arm64) numpy's long double *is* an IEEE double; it
# keeps a distinct type number (13) that the port does not model yet, so the
# aliases below behave like float64/complex128.
longdouble = _make_scalar_type("longdouble", _DTYPES["float64"], floating)
clongdouble = _make_scalar_type("clongdouble", _DTYPES["complex128"], complexfloating)
longfloat = longdouble
clongfloat = clongdouble

sctypeDict = {t.__name__: t for t in (
    bool_, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float16, float32, float64, complex64, complex128,
)}


_rnp._register_scalar_types({
    **{t.__name__: t for t in (
        bool_, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
        float16, float32, float64, complex64, complex128,
    )},
    "str_": str_, "bytes_": bytes_, "void": void,
})

sctypeDict.update({d.char: t for d, t in
                   ((t.dtype, t) for t in sctypeDict.values())})

ScalarType = (int, float, complex, _builtins.bool, bytes, str, memoryview)

True_ = bool_(True)
False_ = bool_(False)


class _CopyMode(_enum.Enum):
    """numpy's `copy=` enum (NEP 50 / array-API `copy` semantics)."""

    ALWAYS = True
    NEVER = False
    IF_NEEDED = 2

    def __bool__(self):
        if self == _CopyMode.ALWAYS:
            return True
        if self == _CopyMode.NEVER:
            return False
        raise ValueError(f"{self} is neither True nor False.")

# numpy's typecode groups, minus the codes the port has no dtype for yet
# (longdouble 'g'/'G', object 'O', datetime 'M'/'m').
typecodes = {
    'Character': 'c',
    'Integer': 'bhilqnp',
    'UnsignedInteger': 'BHILQNP',
    'Float': 'efd',
    'Complex': 'FD',
    'AllInteger': 'bBhHiIlLqQnNpP',
    'AllFloat': 'efdFD',
    'Datetime': '',
    'All': '?bhilqnpBHILQNPefdFDSUVO',
}


def issubdtype(arg1, arg2):
    """numpy's `issubdtype`, over the abstract scalar hierarchy above."""
    if not isinstance(arg1, type) or not issubclass(arg1, generic):
        arg1 = dtype(arg1).type
    if not isinstance(arg2, type) or not issubclass(arg2, generic):
        try:
            arg2 = dtype(arg2).type
        except Exception:  # noqa: BLE001
            pass
    return issubclass(arg1, arg2)


def isdtype(dtype_, kind):
    """Array-API `isdtype`."""
    d = dtype(dtype_)
    kinds = kind if isinstance(kind, tuple) else (kind,)
    groups = {
        "bool": lambda x: x.kind == "b",
        "signed integer": lambda x: x.kind == "i",
        "unsigned integer": lambda x: x.kind == "u",
        "integral": lambda x: x.kind in "iu",
        "real floating": lambda x: x.kind == "f",
        "complex floating": lambda x: x.kind == "c",
        "numeric": lambda x: x.kind in "biufc",
    }
    for k in kinds:
        if isinstance(k, str):
            if k not in groups:
                raise ValueError(f"unsupported kind: {k!r}")
            if groups[k](d):
                return True
        elif dtype(k) == d:
            return True
    return False


def result_type_(*args):
    return result_type(*args)


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


# `sum`, `prod`, `mean`, `amin`/`amax`, `argmin`/`argmax` are native Rust
# reductions imported from `_rnp` at the top of this module. numpy also
# exposes the builtins-shadowing aliases:
max = amax
min = amin


def all(a, axis=None, out=None, keepdims=False):
    return _asarr(a).all(axis=axis, out=out, keepdims=keepdims)


def any(a, axis=None, out=None, keepdims=False):
    return _asarr(a).any(axis=axis, out=out, keepdims=keepdims)


def repeat(a, repeats, axis=None):
    return _asarr(a).repeat(repeats, axis=axis)


def squeeze(a, axis=None):
    return _asarr(a).squeeze(axis)


def swapaxes(a, axis1, axis2):
    return _asarr(a).swapaxes(axis1, axis2)


class _IndexExpression:
    """numpy's `np.s_` / `np.index_exp`."""

    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        return item


s_ = _IndexExpression(maketuple=False)
index_exp = _IndexExpression(maketuple=True)


class ndindex:
    """Iterate over the C-order multi-indices of a shape."""

    def __init__(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], int):
            shape = tuple(shape[0])
        self._shape = tuple(_builtins.int(s) for s in shape)
        self._total = 1
        for s in self._shape:
            self._total *= s
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self._total:
            raise StopIteration
        rem, out = self._i, [0] * len(self._shape)
        for ax in range(len(self._shape) - 1, -1, -1):
            d = self._shape[ax]
            out[ax] = rem % d
            rem //= d
        self._i += 1
        return tuple(out)

    def ndincr(self):
        next(self)


def ix_(*args):
    """Open mesh from N 1-D sequences (numpy's `np.ix_`)."""
    out = []
    nd = len(args)
    for k, seq in enumerate(args):
        a = _asarr(seq)
        if a.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if a.dtype == dtype("bool"):
            a = nonzero(a)[0]
        elif a.size == 0:
            a = a.astype("intp")
        shape = (1,) * k + (a.size,) + (1,) * (nd - k - 1)
        out.append(a.reshape(shape))
    return tuple(out)


def indices(dimensions, dtype=int_):
    dimensions = tuple(dimensions)
    n = len(dimensions)
    out = empty((n,) + dimensions, dtype)
    for i, d in enumerate(dimensions):
        shape = (1,) * i + (d,) + (1,) * (n - i - 1)
        out[i] = arange(d, dtype=dtype).reshape(shape)
    return out



def array_equal(a1, a2):
    a1, a2 = _asarr(a1), _asarr(a2)
    if a1.shape != a2.shape:
        return False
    return _builtins.all(_flat_values(equal(a1, a2)))


def nextafter(x, y):
    """Scalar-only `nextafter`; the ufunc version lands with M3."""
    import math
    return math.nextafter(_builtins.float(x), _builtins.float(y))


def isscalar(x):
    return isinstance(x, (int, float, complex, str, bytes, _builtins.bool))


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
        "float16": dict(bits=16, eps=0.000977, max=65500.0, min=-65500.0,
                        tiny=6.104e-05, nmant=10, nexp=5, precision=3,
                        resolution=0.001, epsneg=0.0004885, iexp=5,
                        machep=-10, negep=-11, maxexp=16, minexp=-14),
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
        self.smallest_subnormal = {
            "float16": 6e-08, "float32": 1e-45, "float64": 5e-324,
        }[d.name]

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


# --------------------------------------------------------------------------
# The rest of the ufunc namespace.
#
# `np.add` and friends are real (native `_rnp` functions); every other ufunc
# name exists so that upstream test modules import and collect, but raises
# NotImplementedError the moment it is called. Real ufuncs land in M3.
# --------------------------------------------------------------------------

from ._core import umath as _umath  # noqa: E402

for _name in _umath._ALL:
    globals().setdefault(_name, getattr(_umath, _name))
del _name

from . import __config__, lib, ma, random  # noqa: E402,F401
from ._core import multiarray  # noqa: E402,F401
from ._core.numerictypes import sctypes  # noqa: E402,F401
from ._stubs import inert_class as _inert_class  # noqa: E402
from ._stubs import not_implemented as _not_implemented  # noqa: E402

# Classes upstream test modules *instantiate* at import time.
for _name in ("matrix", "poly1d", "vectorize", "broadcast"):
    globals().setdefault(_name, _inert_class(_name))
del _name


# Names the upstream tests reference at module level but that belong to later
# milestones. Each raises NotImplementedError when used.
for _name in (
    "argsort", "sort", "searchsorted", "partition", "argpartition", "dot",
    "vdot", "inner", "outer", "tensordot", "einsum", "cross", "trace",
    "diagonal", "cumsum", "cumprod", "diff", "gradient", "histogram",
    "linspace", "logspace", "geomspace", "meshgrid", "roll", "rot90", "flip",
    "tile", "unique", "sort_complex", "clip", "round", "around", "ptp",
    "var", "std", "median", "average", "correlate", "convolve", "count_nonzero",
    "isclose", "allclose", "array_equiv", "moveaxis", "rollaxis", "expand_dims",
    "split", "array_split", "dsplit", "hsplit", "vsplit", "dstack", "column_stack",
    "append", "insert", "delete", "resize", "trim_zeros", "fromfunction",
    "frombuffer", "fromfile", "fromiter", "fromstring", "loadtxt", "savetxt",
    "save", "load", "matmul", "vecdot", "packbits", "unpackbits", "digitize",
    "select", "piecewise", "extract", "place", "copyto", "shares_memory",
    "may_share_memory", "apply_along_axis", "asanyarray", "ascontiguousarray",
    "asfortranarray", "require", "nan_to_num", "real", "imag", "angle",
    "asmatrix", "bmat", "poly1d", "recarray", "record", "chararray",
    "vectorize", "frompyfunc", "busday_count", "busday_offset", "is_busday",
    "datetime_data", "printoptions", "set_printoptions", "get_printoptions",
    "array2string", "array_repr", "array_str", "format_float_positional",
    "format_float_scientific", "base_repr", "binary_repr", "info", "who",
    "setdiff1d", "union1d", "intersect1d", "in1d", "isin", "genfromtxt",
    "memmap", "nditer", "broadcast", "errstate_unavailable",
):
    globals().setdefault(_name, _not_implemented(f"numpy.{_name}"))
del _name


__all__ = [n for n in dir() if not n.startswith("_")]
