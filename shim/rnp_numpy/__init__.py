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

from . import exceptions
from .exceptions import AxisError, ComplexWarning, DTypePromotionError, VisibleDeprecationWarning

__version__ = "2.5.2"

# `ndarray.astype`/`.copy`/`.__setitem__` gain their NumPy signatures here.
# This has to happen before `._scalars` mirrors ndarray's method signatures
# onto `generic`, hence the early import.
from . import _arraycompat as _arraycompat  # noqa: E402

_arraycompat.install()


# --------------------------------------------------------------------------
# `order=` support for the allocating constructors.
#
# The Rust engine only ever allocates C-contiguous buffers.  An F-contiguous
# array of shape ``s`` is obtained by allocating the reversed shape and
# reversing the axes, which yields a genuine Fortran-ordered view.
# --------------------------------------------------------------------------

_rnp_zeros = zeros
_rnp_empty = empty
_rnp_ones = ones


def _normalize_alloc_order(order):
    if order is None:
        return "C"
    if isinstance(order, str) and len(order) == 1 and order.upper() in "CFAK":
        return order.upper()
    raise ValueError("order not understood")


def _alloc_with_order(_fn, shape, dtype, order):
    if order in (None, "C", "c"):
        # Fast path: nothing to do beyond the raw Rust constructor.
        return _fn(shape, dtype)
    order = _normalize_alloc_order(order)
    if hasattr(shape, "__len__"):
        shape = tuple(shape)
    else:
        shape = (shape,)
    if order == "F" and len(shape) > 1:
        arr = _fn(shape[::-1], dtype)
        return arr.transpose(tuple(_builtins.range(len(shape) - 1, -1, -1)))
    return _fn(shape, dtype)


def zeros(shape, dtype=None, order="C", *, like=None):
    return _alloc_with_order(_rnp_zeros, shape, dtype, order)


def empty(shape, dtype=None, order="C", *, like=None):
    return _alloc_with_order(_rnp_empty, shape, dtype, order)


def ones(shape, dtype=None, order="C", *, like=None):
    return _alloc_with_order(_rnp_ones, shape, dtype, order)


# --------------------------------------------------------------------------
# `array()` on sequences that contain ndarrays.
#
# The Rust constructor only understands sequences of Python scalars.  When it
# reports an unsupported ndarray element we flatten the nested arrays into
# plain lists and re-run, carrying the promoted dtype of the nested arrays so
# the result dtype is not degraded by the round trip through Python objects.
# --------------------------------------------------------------------------

_rnp_array = array

_NESTED_NDARRAY_MSG = "unsupported element type in array(): ndarray"


def _arraycompat_mod():
    """`._arraycompat`, imported lazily (it is bound at the bottom of here)."""
    import importlib
    import sys
    mod = sys.modules.get(__name__ + "._arraycompat")
    return mod if mod is not None else importlib.import_module(
        __name__ + "._arraycompat")


def _flatten_nested_ndarrays(obj, dtypes):
    if isinstance(obj, ndarray):
        dtypes.append(obj.dtype)
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_flatten_nested_ndarrays(x, dtypes) for x in obj]
    return obj


#: Spellings of numpy's "character" typecode.  `np.dtype('c')` compares equal
#: to `S1`, but the `'c'` typecode additionally means "unpack each string into
#: its individual characters", which `S1` does not.  The engine's dtype object
#: does not carry the distinction, so it has to be caught on the spelling the
#: caller passed in, before the dtype is normalised.
_CHAR_TYPECODES = frozenset(("c", "|c", "=c"))


def _char_typecode_elements(obj):
    """Unpack every string in `obj` into a list of one-character strings.

    This mirrors what numpy does for `dtype='c'`: `'abc1'` becomes
    `[b'a', b'b', b'c', b'1']`, so the result gains a trailing axis of length
    equal to the string.  Splitting each element by *its own* length (rather
    than padding to a common itemsize) is deliberate — it is what makes
    ragged input such as `['a', 'bc']` raise numpy's usual inhomogeneous-shape
    ValueError instead of silently NUL-padding.
    """
    base = _rnp_array(obj, "S", copy=True)

    def split(value):
        if isinstance(value, list):
            return [split(v) for v in value]
        return [value[i:i + 1] for i in range(len(value))]

    return split(base.tolist())


def _string_fill_fallback(exc, dtype, obj):
    """Render numbers as text when a `U`/`S` array was asked for.

    The engine only fills string arrays from `str`/`bytes`; numpy also accepts
    numbers and formats them.  Returns the built array, or None when `exc` was
    something else entirely (in which case the caller keeps its usual paths).
    """
    if dtype is None or hasattr(type(obj), "__array__"):
        # Objects carrying `__array__` have their own conversion path, which
        # must win — this fallback is only for plain numeric input.
        return None
    import importlib
    _strcast = importlib.import_module(__name__ + "._core._strcast")
    if not _strcast.is_string_fill_error(exc):
        return None
    _dt = _rnp.dtype(dtype)
    if _dt.kind not in ("U", "S"):
        return None
    # `dtype=str`/`dtype='U'` is unsized and lets numpy choose the width;
    # an explicit `U7` pins it.
    _size = _dt.itemsize // (4 if _dt.kind == "U" else 1) or None
    try:
        return _strcast.to_string_array(obj, _dt, _dt.kind, _size)
    except TypeError:
        # Source is not a numeric type we know a width for; let the caller
        # fall through to its normal handling (and its normal error).
        return None


def array(obj, dtype=None, *, copy=True, order="K", subok=False, ndmin=0,
          like=None):
    _copy = False if copy is None else copy
    if isinstance(dtype, str) and dtype in _CHAR_TYPECODES:
        obj, dtype = _char_typecode_elements(obj), "S1"
    try:
        res = _rnp_array(obj, dtype, copy=_copy)
    except TypeError as exc:
        _ac = _arraycompat_mod()
        _as_text = _string_fill_fallback(exc, dtype, obj)
        if _as_text is not None:
            res = _as_text
        elif _NESTED_NDARRAY_MSG in str(exc) and isinstance(obj, (list, tuple)):
            res = _rnp_array(*_unnest(obj, dtype), copy=_copy)
        elif _ac._has_none_error(exc):
            res = _ac.array_none_fallback(obj, dtype)
        elif hasattr(type(obj), "__array__"):
            res = _ac.array_from_protocol(obj, dtype, copy, exc)
        else:
            from . import _textparse as _tp
            parsed = (_tp.parse_text(obj, dtype if dtype is None
                                     else _rnp.dtype(dtype))
                      if _tp.has_text_error(exc) else None)
            if parsed is None:
                raise
            res = _rnp_array(parsed, dtype, copy=_copy)
    if order in (None, "K", "k") and not ndmin:
        return res
    return _arraycompat_mod().finish_array(res, order, ndmin, copy)


def _unnest(obj, dtype):
    dtypes = []
    plain = _flatten_nested_ndarrays(obj, dtypes)
    if dtype is None and dtypes:
        dtype = dtypes[0]
        for _d in dtypes[1:]:
            dtype = promote_types(dtype, _d)
    return plain, dtype


_rnp_broadcast_to = broadcast_to


def broadcast_to(array, shape, subok=False):
    """`broadcast_to` over anything array-like, not just an existing array.

    The engine's version needs a real array; numpy accepts any array-like and
    converts first, which is what makes `np.broadcast_to(None, (2,))` an
    object array of `None` rather than a TypeError.
    """
    if not isinstance(array, ndarray):
        array = globals()["array"](array)
    return _rnp_broadcast_to(array, shape)


_rnp_asarray = asarray


def asarray(obj, dtype=None):
    try:
        return _rnp_asarray(obj, dtype)
    except TypeError as exc:
        _as_text = _string_fill_fallback(exc, dtype, obj)
        if _as_text is not None:
            return _as_text
        if hasattr(obj, "__array_interface__"):
            import importlib
            _mo = importlib.import_module(__name__ + "._core._memoverlap")
            try:
                return _mo.array_from_interface(obj, dtype)
            except TypeError:
                # Not every object exposing `__array_interface__` can be
                # rebuilt from it (the composition wrappers forward the
                # attribute but have no base array to point at).  Fall
                # through to `__array__` rather than failing outright.
                if not hasattr(type(obj), "__array__"):
                    raise
        if hasattr(type(obj), "__array__"):
            # Same protocol fallback `array()` already honours.  Objects that
            # are array-like only via `__array__` (the composition-based
            # `memmap`/`chararray` wrappers, and any user class implementing
            # just this hook) must convert here too, not only through
            # `np.array`.
            return _arraycompat_mod().array_from_protocol(
                obj, dtype, False, exc
            )
        if _NESTED_NDARRAY_MSG not in str(exc) or not isinstance(
            obj, (list, tuple)
        ):
            raise
    return _rnp_asarray(*_unnest(obj, dtype))

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
# The scalar type hierarchy (M3) — real Python types with numpy's MRO.
# --------------------------------------------------------------------------

from . import _scalars as _sc  # noqa: E402
from ._scalars import (  # noqa: E402,F401
    ScalarType,
    bool_,
    byte,
    bytes_,
    cdouble,
    character,
    clongdouble,
    complex64,
    complex128,
    complexfloating,
    csingle,
    datetime64,
    double,
    flexible,
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
    object_,
    sctypeDict,
    sctypes,
    short,
    signedinteger,
    single,
    str_,
    timedelta64,
    typecodes,
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
    unicode_,
    unsignedinteger,
    ushort,
    void,
)
from ._scalars import False_, True_  # noqa: E402

bool = bool_
float_ = float64
complex_ = complex128
longfloat = longdouble
clongfloat = clongdouble
string_ = bytes_

_DTYPES = _dtype_table()
_rnp._register_scalar_types(_sc.registry())


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
    if not -x.ndim <= axis < x.ndim:
        raise AxisError(axis, x.ndim)
    if axis < 0:
        axis += x.ndim
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


def _normalize_axis_tuple(axis, ndim, argname="axis"):
    if not isinstance(axis, (tuple, list)):
        try:
            axis = [_builtins.int(axis)]
        except TypeError:
            pass
    out = []
    for ax in axis:
        if not -ndim <= ax < ndim:
            raise AxisError(ax, ndim, argname)
        out.append(ax + ndim if ax < 0 else ax)
    out = tuple(out)
    if len(set(out)) != len(out):
        raise ValueError(f"repeated axis in `{argname}` argument")
    return out


def moveaxis(a, source, destination):
    a = _asarr(a)
    source = _normalize_axis_tuple(source, a.ndim, "source")
    destination = _normalize_axis_tuple(destination, a.ndim, "destination")
    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` arguments must have the same number "
            "of elements"
        )
    order = [n for n in _builtins.range(a.ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    return a.transpose(tuple(order))


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

    #: Attributes numpy exposes as scalars *of the inspected dtype* rather
    #: than as Python floats.  The distinction is observable — `str()` of a
    #: `float16` scalar is `'6.55e+04'` where the Python float prints
    #: `'65500.0'` — so the counts below stay Python ints and only these are
    #: wrapped.
    _SCALAR_FIELDS = frozenset(
        "eps max min tiny resolution epsneg "
        "smallest_normal smallest_subnormal".split())

    def __init__(self, float_type):
        d = dtype(float_type)
        if d.kind == "c":
            d = dtype("float32" if d.itemsize == 8 else "float64")
        if d.name not in self._PARAMS:
            raise ValueError(f"data type {d.name!r} not inexact")
        self.dtype = d
        _scalar = d.type
        for k, v in self._PARAMS[d.name].items():
            setattr(self, k, _scalar(v) if k in self._SCALAR_FIELDS else v)
        self.smallest_normal = self.tiny
        self.smallest_subnormal = _scalar({
            "float16": 6e-08, "float32": 1e-45, "float64": 5e-324,
        }[d.name])

    def __repr__(self):
        return f"finfo(resolution={self.resolution}, dtype={self.dtype.name})"


from ._errstate import (  # noqa: E402,F401
    errstate,
    geterr,
    geterrcall,
    seterr,
    seterrcall,
)


# --------------------------------------------------------------------------
# The ufunc namespace: one real `ufunc` object per name numpy exposes.
# --------------------------------------------------------------------------

from ._ufunc import ALL as _UFUNCS  # noqa: E402
from ._ufunc import ufunc  # noqa: E402

globals().update(_UFUNCS)

# Installs the ufunc exception factories the Rust engine raises through, so it
# has to come after `_ufunc` (the factory looks ufunc objects up by name).
from ._core import _exceptions as _core_exceptions  # noqa: E402,F401
from ._core.records import record  # noqa: E402,F401

from . import __config__, lib, ma, random  # noqa: E402,F401
from ._core import multiarray  # noqa: E402,F401
from . import dtypes  # noqa: E402,F401
from ._stubs import inert_class as _inert_class  # noqa: E402
from ._stubs import not_implemented as _not_implemented  # noqa: E402
from ._printing import (  # noqa: E402,F401
    format_float_positional, format_float_scientific,
)

# --------------------------------------------------------------------------
# Array printing.
#
# `_core.arrayprint` is numpy's own (pure-Python) printer, ported onto the
# port's primitives, and — as in numpy — it *is* `ndarray.__repr__`/`__str__`
# rather than a parallel implementation, so `repr()` and `np.array2string`
# can never disagree and `printoptions` reaches every array in the suite.
# --------------------------------------------------------------------------

from ._core import arrayprint as _arrayprint  # noqa: E402
from ._core.arrayprint import (  # noqa: E402,F401
    array2string, array_repr, array_str, get_printoptions, printoptions,
    set_printoptions,
)

def _ndarray_repr(self):
    return _arrayprint._default_array_repr(self)


def _ndarray_str(self):
    return _arrayprint._default_array_str(self)


ndarray.__repr__ = _ndarray_repr
ndarray.__str__ = _ndarray_str

from ._arraycompat import (  # noqa: E402,F401
    broadcast_arrays,
    can_cast,
    copy,
    copyto,
    full,
    may_share_memory,
    shares_memory,
)


# --------------------------------------------------------------------------
# Pure-Python subsystems ported onto the engine's primitives.  Each is wired
# defensively: a subsystem that is not present (or that fails to import while
# it is being landed) leaves its names as the NotImplementedError stubs below
# rather than breaking the whole package.  `_subsystem_import_errors` records
# what went wrong so a missing name is always diagnosable.
# --------------------------------------------------------------------------

_subsystem_import_errors = {}


def _wire_subsystem(module, names):
    try:
        _mod = _importlib.import_module(module, __name__)
    except Exception as _exc:  # pragma: no cover - diagnostic path
        _subsystem_import_errors[module] = _exc
        return
    for _n in names:
        try:
            globals()[_n] = getattr(_mod, _n)
        except AttributeError as _exc:  # pragma: no cover - diagnostic path
            _subsystem_import_errors[f"{module}.{_n}"] = _exc


import importlib as _importlib  # noqa: E402

_wire_subsystem("._core._numeric_close", ("isclose", "allclose"))
_wire_subsystem("._core.defchararray", ("chararray", "compare_chararrays"))
_wire_subsystem("._core.memmap", ("memmap",))
_wire_subsystem(
    "._core._textio",
    ("fromstring", "fromfile", "loadtxt", "genfromtxt", "savetxt"),
)

for _name in ("char", "strings"):
    try:
        globals()[_name] = _importlib.import_module(f".{_name}", __name__)
    except Exception as _exc:  # pragma: no cover - diagnostic path
        _subsystem_import_errors[_name] = _exc
del _name


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
    "asfortranarray", "require", "nan_to_num",  "angle",
    "asmatrix", "bmat", "poly1d", "recarray", "chararray",
    "vectorize", "frompyfunc", "busday_count", "busday_offset", "is_busday",
    "datetime_data", "base_repr", "binary_repr", "info", "who",
    "setdiff1d", "union1d", "intersect1d", "in1d", "isin", "genfromtxt",
    "memmap", "nditer", "broadcast", "errstate_unavailable",
):
    globals().setdefault(_name, _not_implemented(f"numpy.{_name}"))
del _name


__all__ = [n for n in dir() if not n.startswith("_")]
