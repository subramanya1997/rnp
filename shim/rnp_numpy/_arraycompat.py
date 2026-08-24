"""Python-level compatibility layer for array construction, casting and copying.

The Rust engine allocates C-contiguous buffers and performs casts without the
bookkeeping NumPy's public API promises (``order=``, ``casting=``, the
``ComplexWarning``, object-dtype fallbacks).  Everything in this module is a
thin wrapper that re-expresses those promises in terms of the primitives the
engine already provides -- transposed views, ``where``, ``__setitem__`` -- so
no numeric kernel is reimplemented here.
"""

import builtins as _b
import warnings as _warnings

import _rnp
from _rnp import (
    broadcast_to as _broadcast_to,
    can_cast as _can_cast,
    dtype as _dtype,
    empty as _empty,
    ndarray,
    where_ as _where,
)
from _rnp import array as _raw_array
from _rnp import full as _raw_full

from .exceptions import ComplexWarning

_orig_astype = ndarray.astype
_orig_copy = ndarray.copy
_orig_setitem = ndarray.__setitem__

_ORDERS = ("C", "F", "A", "K")

_NAN = float("nan")
_CNAN = complex(_NAN, _NAN)
def _pkg():
    import sys
    return sys.modules[__name__.rsplit(".", 1)[0]]


# ---------------------------------------------------------------------------
# memory order
# ---------------------------------------------------------------------------

def _norm_order(order, default="K"):
    if order is None:
        return default
    if isinstance(order, str) and len(order) == 1 and order.upper() in _ORDERS:
        return order.upper()
    raise ValueError("order not understood")


def _perm_for(a, order):
    """Axis permutation whose C-contiguous copy realises `order` for `a`."""
    n = a.ndim
    ident = tuple(_b.range(n))
    rev = tuple(_b.range(n - 1, -1, -1))
    if order == "C":
        return ident
    if order == "F":
        return rev
    flags = a.flags
    if order == "A":
        return rev if (flags.f_contiguous and not flags.c_contiguous) else ident
    # 'K': keep the existing axis ordering, strongest stride first.
    strides = a.strides
    return tuple(sorted(ident, key=lambda i: -abs(strides[i])))


def _invert(perm):
    out = [0] * len(perm)
    for pos, ax in enumerate(perm):
        out[ax] = pos
    return tuple(out)


def _ordered_copy(a, dt, order):
    """A fresh array with dtype `dt` and the memory order `order`."""
    if a.ndim < 2:
        return _orig_astype(a, dt, copy=True)
    perm = _perm_for(a, order)
    if perm == tuple(_b.range(a.ndim)):
        return _orig_astype(a, dt, copy=True)
    return _orig_astype(a.transpose(perm), dt, copy=True).transpose(_invert(perm))


def _order_ok(a, order):
    if order in ("K", "A"):
        return True
    flags = a.flags
    return flags.c_contiguous if order == "C" else flags.f_contiguous


# ---------------------------------------------------------------------------
# object-dtype fallbacks (element-wise, via Python)
# ---------------------------------------------------------------------------

def _build(values, shape, dt):
    out = _empty(shape, dt)
    if out.size:
        out[...] = _raw_array(values, dt)
    return out


def _to_python(value, dt):
    kind = dt.kind
    if value is None and kind in "fc":
        return _NAN if kind == "f" else _CNAN
    if isinstance(value, (bytes, bytearray)) and kind in "iufc":
        value = value.decode("latin1")
    if kind == "b":
        return _b.bool(value)
    if kind in "iu":
        return _b.int(value)
    if kind == "f":
        return _b.float(value)
    if kind == "c":
        return complex(value)
    if kind == "U":
        return value if isinstance(value, str) else (
            value.decode("latin1") if isinstance(value, bytes) else str(value))
    if kind == "S":
        if isinstance(value, bytes):
            return value
        return str(value).encode("latin1") if not isinstance(value, str) \
            else value.encode("latin1")
    return value


def _map_nested(obj, fn):
    if isinstance(obj, _b.list):
        return [_map_nested(x, fn) for x in obj]
    return fn(obj)


def _object_astype(self, dt):
    """`astype` out of an object array, converting element by element."""
    values = _map_nested(self.tolist(), lambda v: _to_python(v, dt))
    if dt.kind in "SU" and dt.itemsize == 0:
        flat = []
        _flatten(values, flat)
        width = _b.max([len(x) for x in flat], default=0) or 1
        dt = _dtype(f"{dt.char}{width}")
    if self.size == 0:
        return _empty(self.shape, dt)
    return _build(values, self.shape, dt)


def _flatten(obj, out):
    if isinstance(obj, _b.list):
        for x in obj:
            _flatten(x, out)
    else:
        out.append(obj)


def _to_object(self):
    values = self.tolist()
    out = _empty(self.shape, _dtype("O"))
    if self.size == 0:
        return out
    if self.ndim == 0:
        out[()] = values
        return out
    _fill_object(out, values, ())
    return out


def _fill_object(out, values, prefix):
    if isinstance(values, _b.list):
        for i, v in enumerate(values):
            _fill_object(out, v, prefix + (i,))
    else:
        out[prefix] = values


# ---------------------------------------------------------------------------
# void ("V") targets
# ---------------------------------------------------------------------------

def _resolve_void(self, dt):
    if dt.itemsize:
        return dt
    if self.dtype.kind == "O":
        flat = []
        _flatten(self.tolist(), flat)
        width = 8
        for v in flat:
            if isinstance(v, (bytes, bytearray)):
                width = _b.max(width, len(v))
        return _dtype(f"V{width}")
    return _dtype(f"V{self.dtype.itemsize}")


# ---------------------------------------------------------------------------
# can_cast
# ---------------------------------------------------------------------------

def _dtype_of(x):
    if isinstance(x, ndarray):
        return x.dtype
    try:
        return _dtype(x)
    except Exception:  # noqa: BLE001
        return None


def can_cast(from_, to, casting="safe"):
    """`np.can_cast`, with the object-dtype rules the engine does not model."""
    src, dst = _dtype_of(from_), _dtype_of(to)
    if src is not None and dst is not None and (src.kind == "O") != (
            dst.kind == "O"):
        # Anything may be stored in an object array; nothing may come back
        # out of one without an unsafe cast.
        return casting == "unsafe" if src.kind == "O" else \
            casting in ("safe", "same_kind", "unsafe")
    return _can_cast(from_, to, casting)


# ---------------------------------------------------------------------------
# ndarray.astype
# ---------------------------------------------------------------------------

_COMPLEX_MSG = "Casting complex values to real discards the imaginary part"


def _check_copy_flag(copy):
    mode = getattr(_pkg(), "_CopyMode", None)
    if mode is not None and isinstance(copy, mode):
        raise ValueError(
            "_CopyMode enum is not allowed for astype function. "
            "Use true/false instead.")


def astype(self, /, dtype, order="K", casting="unsafe", subok=True,
           copy=True):
    _check_copy_flag(copy)
    dt = _dtype(dtype)
    order = _norm_order(order, "K")
    src = self.dtype
    if not can_cast(src, dt, casting):
        raise TypeError(
            f"Cannot cast array data from {src!r} to {dt!r} "
            f"according to the rule {casting!r}")
    if src.kind == "c" and dt.kind in "iufmM":
        _warnings.warn(_COMPLEX_MSG, ComplexWarning, stacklevel=2)
    if dt.kind == "V" and dt.names is None and dt.subdtype is None:
        dt = _resolve_void(self, dt)
        if dt.itemsize == src.itemsize and src.kind != "O":
            return _ordered_copy(self, src, "C").view(dt)
        if self.size == 0:
            return _empty(self.shape, dt)
    if not copy and dt == src and _order_ok(self, order):
        return self
    if src.kind == "O" and dt.kind in "mM":
        # The engine's datetime coercion handles every object form numpy
        # accepts (and rejects the rest with numpy's own ValueError), so the
        # generic element-by-element path -- which would recurse into a
        # co-recursive list -- is bypassed.
        return _pkg().array(self.tolist(), dtype=dt).reshape(self.shape)
    if src.kind == "O" and dt.kind != "O":
        return _object_astype(self, dt)
    if src.kind in "SU" and dt.kind in "biufc":
        return _object_astype(self, dt)
    if src.kind in "mM" and dt.kind == "O":
        return _pkg().array(_rnp._datetime_objects(self), dtype=object
                            ).reshape(self.shape)
    if dt.kind == "O" and src.kind != "O":
        return _to_object(self)
    if dt.kind in "SU" and src.kind in "mM":
        # datetime -> text: the engine renders each element the way numpy's
        # `datetime_as_string` (for M8) and `str(scalar)` (for m8) do, and an
        # unsized target takes numpy's own per-unit width.
        from . import _datetime as _dtmod  # noqa: F401
        strs = _rnp._datetime_strings(self, casting="unsafe")
        width = dt.itemsize // (4 if dt.kind == "U" else 1)
        if width == 0:
            width = _rnp._datetime_string_len(src)
        out = _empty(self.shape, _dtype(f"{dt.char}{width}"))
        flat = out.reshape(-1)
        for i, text in enumerate(strs):
            flat[i] = text if dt.kind == "U" else text.encode()
        return out
    if src.kind in "SU" and dt.kind in "mM":
        # text -> datetime: the array constructor already parses ISO strings.
        return _pkg().array(self.tolist(), dtype=dt)
    if dt.kind in "SU" and src.kind in "biufc":
        # Numbers rendered as text.  Note this must come before the
        # `itemsize == 0` branch below: that one sizes the result from the
        # source's *bytes*, which is right for S<->U but not here (float64 is
        # 8 bytes yet numpy gives it `U32`).
        from ._core import _strcast
        _size = dt.itemsize // (4 if dt.kind == "U" else 1) or None
        return _strcast.to_string_array(self, dt, dt.kind, _size)
    if dt.kind in "SU" and src.kind in "SU":
        from ._core import _strcast
        return _strcast.restring(self, dt)
    if dt.kind in "SU" and dt.itemsize == 0:
        dt = _dtype(f"{dt.char}{_b.max(src.itemsize, 1)}")
    return _ordered_copy(self, dt, order)


# ---------------------------------------------------------------------------
# ndarray.copy / np.copy
# ---------------------------------------------------------------------------

def copy_method(self, /, order="C"):
    return _ordered_copy(self, self.dtype, _norm_order(order, "C"))


def copy(a, order="K", subok=False):
    a = a if isinstance(a, ndarray) else _pkg().asarray(a)
    return _ordered_copy(a, a.dtype, _norm_order(order, "K"))


# ---------------------------------------------------------------------------
# memory overlap
# ---------------------------------------------------------------------------

# Imported lazily: `._core` pulls names back out of the package, and this
# module is imported before the package finishes initialising.

def _overlap_module():
    import importlib
    return importlib.import_module(
        __name__.rsplit(".", 1)[0] + "._core._memoverlap")


def shares_memory(a, b, /, max_work=None):
    return _overlap_module().shares_memory(a, b, max_work=max_work)


def may_share_memory(a, b, /, max_work=None):
    return _overlap_module().may_share_memory(a, b, max_work=max_work)


# ---------------------------------------------------------------------------
# cast safety for Python scalars (NEP 50 "weak" scalars)
# ---------------------------------------------------------------------------

_WEAK = {
    _b.bool: ("bool", "biufc", "Python bool"),
    _b.int: ("int64", "iufc", "Python int"),
    _b.float: ("float64", "fc", "Python float"),
    complex: ("complex128", "c", "Python complex"),
}


def _weak_kind(value):
    return _WEAK.get(type(value))


def _check_scalar_cast(value, dst_dt, casting):
    """Validate assigning the Python scalar `value` into `dst_dt`."""
    info = _weak_kind(value)
    default = _dtype(info[0])
    if dst_dt.kind == "O":
        return
    if dst_dt.kind in info[1]:
        if casting == "equiv" and dst_dt != default:
            raise TypeError(
                f"cannot cast {info[2]} to {dst_dt.name} under the casting "
                f"rule {casting!r}")
        if dst_dt.kind in "iu":
            _check_int_range(value, dst_dt)
        return
    if not can_cast(default, dst_dt, casting):
        raise TypeError(
            f"Cannot cast scalar from {default!r} to {dst_dt!r} "
            f"according to the rule {casting!r}")


def _check_int_range(value, dst_dt):
    bits = dst_dt.itemsize * 8
    if dst_dt.kind == "u":
        lo, hi = 0, (1 << bits) - 1
    else:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    value = _b.int(value)
    if not lo <= value <= hi:
        raise OverflowError(
            f"Python integer {value} out of bounds for {dst_dt.name}")


# ---------------------------------------------------------------------------
# np.copyto
# ---------------------------------------------------------------------------

def copyto(dst, src, casting="same_kind", where=True):
    if not isinstance(dst, ndarray):
        raise TypeError(
            "copyto() argument 1 must be a numpy.ndarray, "
            f"not {type(dst).__name__}")
    if casting not in ("no", "equiv", "safe", "same_kind", "unsafe"):
        raise ValueError(
            f"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or "
            f"'unsafe' (got {casting!r})")
    dt = dst.dtype

    if _weak_kind(src) is not None:
        _check_scalar_cast(src, dt, casting)
        if isinstance(src, complex) and dt.kind in "iufmM":
            _warnings.warn(_COMPLEX_MSG, ComplexWarning, stacklevel=2)
        value = _raw_array(src, dt if dt.kind != "O" else None)
    else:
        value = src if isinstance(src, ndarray) else _pkg().asarray(src)
        if not can_cast(value.dtype, dt, casting):
            what = "scalar" if value.ndim == 0 else "array data"
            raise TypeError(
                f"Cannot cast {what} from {value.dtype!r} to {dt!r} "
                f"according to the rule {casting!r}")
        if value.dtype.kind == "c" and dt.kind in "iufmM":
            _warnings.warn(_COMPLEX_MSG, ComplexWarning, stacklevel=2)

    if where is True or where is None:
        mask = None
    elif where is False:
        return
    else:
        mask = where if isinstance(where, ndarray) else _pkg().asarray(where)
        if mask.dtype.kind != "b":
            mask = _orig_astype(mask, _dtype("bool"), copy=True)
        if mask.ndim == 0:
            if not mask.tolist():
                return
            mask = None
        else:
            try:
                mask = _broadcast_to(mask, dst.shape)
            except Exception:
                raise ValueError(
                    f"could not broadcast where mask from shape "
                    f"{mask.shape} into shape {dst.shape}") from None

    try:
        value = _broadcast_to(value, dst.shape) if value.shape != dst.shape \
            else value
    except Exception:
        raise ValueError(
            f"could not broadcast input array from shape {value.shape} "
            f"into shape {dst.shape}") from None

    # Materialise a temporary so overlapping source/destination is safe.
    if value.dtype != dt:
        # Use the public compatibility path: object and text sources need its
        # element-wise conversion before the Rust numeric cast loop.
        value = value.astype(dt, copy=True)
    else:
        value = _ordered_copy(value, dt, "C")

    if mask is None:
        dst[...] = value
    else:
        dst[...] = _where(_ordered_copy(mask, mask.dtype, "C"), value, dst)


# ---------------------------------------------------------------------------
# np.full with a non-scalar fill value
# ---------------------------------------------------------------------------

def full(shape, fill_value, dtype=None, order="C", *, like=None):
    try:
        out = _raw_full(shape, fill_value, dtype)
    except TypeError:
        fv = fill_value if isinstance(fill_value, ndarray) \
            else _pkg().asarray(fill_value, dtype)
        if dtype is None:
            dtype = fv.dtype
        shape = tuple(shape) if hasattr(shape, "__len__") else (shape,)
        out = _empty(shape, dtype)
        if fv.ndim == 0 and fv.dtype == object:
            # Broadcasting a 0-d object array would store the *array* in every
            # cell; numpy stores the object it holds.
            out[...] = fv[()]
        else:
            out[...] = fv
        return out
    order = _norm_order(order, "C")
    if order in ("F",) and out.ndim > 1:
        return _ordered_copy(out, out.dtype, "F")
    return out


# ---------------------------------------------------------------------------
# None -> NaN in array construction and item assignment
# ---------------------------------------------------------------------------

_NONE_MSGS = ("could not convert NoneType to an array",
              "unsupported element type in array(): NoneType")


def _has_none_error(exc):
    text = str(exc)
    return any(m in text for m in _NONE_MSGS)


def _replace_none(obj, fill):
    if obj is None:
        return fill
    if isinstance(obj, (_b.list, _b.tuple)):
        return [_replace_none(x, fill) for x in obj]
    return obj


def array_none_fallback(obj, dt):
    """Build an array from `obj` when it contains ``None``."""
    if dt is None:
        return _object_array(obj)
    d = _dtype(dt)
    if d.kind == "f":
        return _raw_array(_replace_none(obj, _b.float("nan")), d)
    if d.kind == "c":
        return _raw_array(_replace_none(obj, _CNAN), d)
    if d.kind == "O":
        return _object_array(obj)
    raise TypeError(f"float() argument must be a string or a real number, "
                    f"not 'NoneType'")


def _object_array(obj):
    od = _dtype("O")
    if obj is None or not isinstance(obj, (_b.list, _b.tuple)):
        out = _empty((), od)
        out[()] = obj
        return out
    shape, flat = _shape_of(obj), []
    _flatten(_as_lists(obj), flat)
    out = _empty(shape, od)
    _fill_object(out, _as_lists(obj), ())
    return out


#: numpy's `NPY_MAXDIMS`. Nesting deeper than this is how a co-recursive
#: list (gh-11154) announces itself; numpy answers with a ValueError rather
#: than blowing the C stack, and so must the port.
_MAXDIMS = 64


def _too_deep(found):
    return ValueError(
        f"maximum supported dimension for an ndarray is {_MAXDIMS}, "
        f"found {found}")


def _as_lists(obj, _depth=0):
    if isinstance(obj, (_b.list, _b.tuple)):
        if _depth >= _MAXDIMS:
            raise _too_deep(_depth + 1)
        return [_as_lists(x, _depth + 1) for x in obj]
    return obj


def _shape_of(obj):
    shape = []
    while isinstance(obj, (_b.list, _b.tuple)):
        if len(shape) >= _MAXDIMS:
            raise _too_deep(len(shape) + 1)
        shape.append(len(obj))
        if not obj:
            break
        obj = obj[0]
    return tuple(shape)


def setitem(self, key, value):
    # `a[dst] = a[src]` must behave as if the right-hand side were read in
    # full before anything is written, exactly as numpy's PyArray_AssignArray
    # does: when source and destination alias, materialise the source first.
    if isinstance(value, ndarray) and value.size and self.size:
        if may_share_memory(self, value):
            value = _ordered_copy(value, value.dtype, "C")
    if isinstance(value, str) and self.dtype.kind in "fciub":
        value = _b.float(value) if self.dtype.kind in "fc" else _b.int(value)
    if isinstance(value, ndarray) and value.dtype.kind in "OSU" \
            and self.dtype.kind in "fciub":
        value = value.astype(self.dtype, copy=True)
    try:
        result = _orig_setitem(self, key, value)
    except TypeError as exc:
        if not _has_none_error(exc) or self.dtype.kind not in "fc":
            raise
        fill = _NAN if self.dtype.kind == "f" else _CNAN
        result = _orig_setitem(self, key, _replace_none(value, fill))
    from . import _errstate
    _errstate.drain("cast", stacklevel=3)
    return result


# ---------------------------------------------------------------------------
# the `__array__` protocol
# ---------------------------------------------------------------------------

_NO_COPY_MSG = (
    "Unable to avoid copy while creating an array as requested.\n"
    "If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` "
    "to allow a copy when needed (no behavior change in NumPy 1.x).\n"
    "For more details, see "
    "https://numpy.org/devdocs/numpy_2_0_migration_guide.html"
    "#adapting-to-changes-in-the-copy-keyword.")


def array_protocol(self, dtype=None, copy=None):
    """``ndarray.__array__``."""
    dt = _dtype(dtype) if dtype is not None else self.dtype
    if dt == self.dtype:
        return _ordered_copy(self, dt, "K") if copy else self
    if copy is False:
        raise ValueError(_NO_COPY_MSG)
    return _orig_astype(self, dt, copy=True)


def array_from_protocol(obj, dtype, copy, original=None):
    """Build an array from an object exposing ``__array__``.

    Objects that merely *declare* the hook without honouring it (stand-ins for
    dtypes the engine has not grown yet) fall back to `original`, the error the
    engine itself raised.
    """
    dt = _dtype(dtype) if dtype is not None else None
    # An *unsized* flexible dtype (`np.dtype(str)` is `<U` with itemsize 0)
    # only pins down the kind; the length is discovered from the data. numpy
    # therefore hands `__array__` a plain `None` and adapts afterwards.
    unsized = dt is not None and dt.itemsize == 0 and dt.kind in "USV"
    requested = None if unsized else dt
    try:
        try:
            res = obj.__array__(dtype=requested, copy=copy)
        except TypeError:
            res = obj.__array__()
    except (TypeError, NotImplementedError, AttributeError):
        if original is None:
            raise
        raise original from None
    if not isinstance(res, ndarray):
        if original is not None:
            raise original from None
        raise TypeError("object __array__ method not producing an array")
    if unsized and res.dtype.kind == dt.kind:
        # Already the right kind: its own itemsize *is* the adapted one.
        return _ordered_copy(res, res.dtype, "K") if copy else res
    if dt is not None and res.dtype != dt:
        if copy is False:
            raise ValueError(_NO_COPY_MSG)
        return _orig_astype(res, dt, copy=True)
    return _ordered_copy(res, res.dtype, "K") if copy else res


def object_scalar_array(obj, dtype):
    """0-d object array for a value the engine has no dtype for."""
    if dtype is not None and _dtype(dtype).kind != "O":
        raise TypeError(
            f"Cannot cast array data from {_dtype('O')!r} to "
            f"{_dtype(dtype)!r} according to the rule 'unsafe'")
    out = _empty((), _dtype("O"))
    out[()] = obj
    return out


# ---------------------------------------------------------------------------
# np.array post-processing (`order=` / `ndmin=`)
# ---------------------------------------------------------------------------

def finish_array(res, order, ndmin, copy=True):
    order = _norm_order(order, "K")
    if ndmin:
        while res.ndim < ndmin:
            res = res[None]
    if order in ("K", "A") or _order_ok(res, order):
        return res
    return _ordered_copy(res, res.dtype, order)


def broadcast_arrays(*args, subok=False):
    pkg = _pkg()
    arrays = [a if isinstance(a, ndarray) else pkg.asarray(a) for a in args]
    shape = _rnp.broadcast_shapes(*[a.shape for a in arrays])
    return tuple(_broadcast_to(a, shape) for a in arrays)


#: Rich-comparison slots that must accept a bare `str`/`bytes` operand.
_STR_COMPARISONS = ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__")


def _make_str_comparison(name):
    """Wrap a comparison so a `str`/`bytes` scalar is treated as array-like.

    Against a string array the engine only understands another *array*: given
    a bare `'x'` it returns NotImplemented, Python falls back to identity, and
    `arr == 'x'` collapses to a single `False` instead of comparing
    elementwise.  Promoting the scalar first restores numpy's broadcasting.
    """
    _orig = getattr(ndarray, name)

    def compare(self, other):
        if isinstance(other, (str, bytes)) and self.dtype.kind in "SU":
            return _orig(self, _pkg().array(other))
        return _orig(self, other)

    compare.__name__ = name
    compare.__qualname__ = f"ndarray.{name}"
    return compare


#: Arithmetic slots numpy gives meaning over string arrays: concatenation,
#: repetition and printf-formatting.  Value is the `numpy.strings` function
#: and whether the operands are reversed (the `__r*__` forms).
_STR_BINOPS = {
    "__add__": ("add", False),
    "__radd__": ("add", True),
    "__mul__": ("multiply", False),
    "__rmul__": ("multiply", True),
    "__mod__": ("mod", False),
}


def _make_str_binop(name, fn, reflected):
    """Route `+`, `*` and `%` on string arrays into `numpy.strings`.

    The engine has no string loops for these, and the operator slots go
    straight to it rather than through the ufunc object, so the fallback
    installed on `ufunc.__call__` never sees them.  `arr1 + arr2` on string
    arrays therefore has to be intercepted here as well.
    """
    _orig = getattr(ndarray, name)

    def binop(self, other):
        if self.dtype.kind in "SU":
            strings = _pkg().strings
            if fn == "multiply":
                # Repetition takes (string, count) whichever side the count
                # was written on, so `2 * arr` and `arr * 2` agree.
                a, b = self, other
            else:
                a, b = (other, self) if reflected else (self, other)
            try:
                return getattr(strings, fn)(a, b)
            except (TypeError, ValueError):
                # Not a combination `numpy.strings` accepts — defer to the
                # engine so its own error (or NotImplemented) is what shows.
                pass
        return _orig(self, other)

    binop.__name__ = name
    binop.__qualname__ = f"ndarray.{name}"
    return binop


#: In-place forms.  These keep the array's own dtype, so the result is
#: truncated back to its itemsize rather than widening it.
_STR_INPLACE = {"__iadd__": "add", "__imul__": "multiply"}


def _make_str_inplace(name, fn):
    """In-place `+=`/`*=` on a string array.

    Unlike the out-of-place forms these cannot widen the dtype, so the result
    is written back through the existing itemsize and simply truncates —
    `np.array(['foo ', 'bar']) *= 2` gives `['foo ', 'barb']`, not the full
    concatenation.
    """
    _orig = getattr(ndarray, name)

    def inplace(self, other):
        if self.dtype.kind in "SU":
            strings = _pkg().strings
            try:
                res = getattr(strings, fn)(self, other)
            except (TypeError, ValueError):
                return _orig(self, other)
            except OverflowError:
                raise OverflowError(
                    "Overflow detected in string multiply") from None
            self[...] = res.astype(self.dtype)
            return self
        return _orig(self, other)

    inplace.__name__ = name
    inplace.__qualname__ = f"ndarray.{name}"
    return inplace


def install():
    ndarray.astype = astype
    ndarray.__array__ = array_protocol
    ndarray.copy = copy_method
    ndarray.__setitem__ = setitem

    def array_function(self, func, types, args, kwargs):
        if not isinstance(args, tuple):
            raise TypeError("args must be a tuple")
        if not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dict")
        if not all(issubclass(t, ndarray) for t in types):
            return NotImplemented
        implementation = getattr(func, "_implementation", func)
        return implementation(*args, **kwargs)

    ndarray.__array_function__ = array_function
    def ctypes_property(self):
        from ._core._internal import _ctypes
        return _ctypes(self, self.__array_interface__["data"][0])

    ndarray.ctypes = property(ctypes_property)
    for _name in _STR_COMPARISONS:
        setattr(ndarray, _name, _make_str_comparison(_name))
    for _name, (_fn, _rev) in _STR_BINOPS.items():
        setattr(ndarray, _name, _make_str_binop(_name, _fn, _rev))
    for _name, _fn in _STR_INPLACE.items():
        setattr(ndarray, _name, _make_str_inplace(_name, _fn))
