"""Object-dtype behaviour that has to be installed on `ndarray` itself.

The Rust engine's object loops live in `rnp-python/src/objloops.rs` and are
reached through the ufunc entry points.  Two surfaces do *not* go through
those entry points:

* the operator slots (`a + b`, `a == b`, `-a`, ...), which `_rnp.ndarray`
  implements directly against the numeric engine, and
* the reduction *methods* (`a.sum()`, `a.all()`, ...), which call the engine's
  native reductions, and which have no object kernels by design -- folding
  Python objects is the ufunc machinery's job.

Both are therefore intercepted here and rerouted to the object ufuncs, in the
same way `_arraycompat` already reroutes the string-array operators.  The
check is one `dtype.kind` read on a path that is not itself a hot loop (the
benchmarks all call the ufuncs directly), so nothing numeric pays for it.
"""

import _rnp


def _pkg():
    import importlib
    return importlib.import_module(__name__.rsplit(".", 1)[0])


#: Operator slot -> (ufunc name, operands reversed).
BINOPS = {
    "__add__": ("add", False), "__radd__": ("add", True),
    "__sub__": ("subtract", False), "__rsub__": ("subtract", True),
    "__mul__": ("multiply", False), "__rmul__": ("multiply", True),
    "__truediv__": ("divide", False), "__rtruediv__": ("divide", True),
    "__floordiv__": ("floor_divide", False),
    "__rfloordiv__": ("floor_divide", True),
    "__mod__": ("remainder", False), "__rmod__": ("remainder", True),
    "__pow__": ("power", False), "__rpow__": ("power", True),
    "__and__": ("bitwise_and", False), "__rand__": ("bitwise_and", True),
    "__or__": ("bitwise_or", False), "__ror__": ("bitwise_or", True),
    "__xor__": ("bitwise_xor", False), "__rxor__": ("bitwise_xor", True),
    "__lshift__": ("left_shift", False), "__rlshift__": ("left_shift", True),
    "__rshift__": ("right_shift", False), "__rrshift__": ("right_shift", True),
    "__eq__": ("equal", False), "__ne__": ("not_equal", False),
    "__lt__": ("less", False), "__le__": ("less_equal", False),
    "__gt__": ("greater", False), "__ge__": ("greater_equal", False),
}

UNOPS = {
    "__neg__": "negative", "__pos__": "positive",
    "__invert__": "invert", "__abs__": "absolute",
}

#: Reduction method -> (ufunc name, forced result dtype or None).
#: `any`/`all` fold in `bool`, exactly as numpy's `umr_any`/`umr_all` do, so
#: they hand back a `np.bool` rather than one of the operands.
REDUCTIONS = {
    "sum": ("add", None),
    "prod": ("multiply", None),
    "min": ("minimum", None),
    "max": ("maximum", None),
    "any": ("logical_or", "bool"),
    "all": ("logical_and", "bool"),
}

ACCUMULATIONS = {"cumsum": "add", "cumprod": "multiply"}


def _is_object(a):
    try:
        return a.dtype.kind == "O"
    except AttributeError:
        return False


def _make_binop(ndarray, name, fn, reflected):
    orig = getattr(ndarray, name)

    def binop(self, other):
        if self.dtype.kind == "O":
            f = getattr(_pkg(), fn)
            return f(other, self) if reflected else f(self, other)
        return orig(self, other)

    binop.__name__ = name
    binop.__qualname__ = f"ndarray.{name}"
    return binop


def _make_unop(ndarray, name, fn):
    orig = getattr(ndarray, name)

    def unop(self):
        if self.dtype.kind == "O":
            return getattr(_pkg(), fn)(self)
        return orig(self)

    unop.__name__ = name
    unop.__qualname__ = f"ndarray.{name}"
    return unop


def _make_reduction(ndarray, name, fn, force_dtype):
    orig = getattr(ndarray, name)

    def reduce(self, axis=None, dtype=None, out=None, keepdims=False,
               initial=None, where=True, **kw):
        if self.dtype.kind != "O":
            return orig(self, axis=axis, dtype=dtype, out=out,
                        keepdims=keepdims, **kw) if dtype is not None else \
                orig(self, axis=axis, out=out, keepdims=keepdims, **kw)
        uf = getattr(_pkg(), fn)
        kwargs = {"axis": axis, "out": out, "keepdims": keepdims,
                  "where": where}
        if dtype is not None:
            kwargs["dtype"] = dtype
        elif force_dtype is not None:
            kwargs["dtype"] = force_dtype
        if initial is not None:
            kwargs["initial"] = initial
        return uf.reduce(self, **kwargs)

    reduce.__name__ = name
    reduce.__qualname__ = f"ndarray.{name}"
    return reduce


def _make_accumulation(ndarray, name, fn):
    orig = getattr(ndarray, name, None)

    def acc(self, axis=None, dtype=None, out=None, **kw):
        if self.dtype.kind == "O":
            uf = getattr(_pkg(), fn)
            a = self if axis is not None else self.reshape(-1)
            res = uf.accumulate(a, axis=0 if axis is None else axis, out=out)
            return res
        if orig is None:
            raise NotImplementedError(f"numpy.{name} is not implemented by rnp yet")
        return orig(self, axis=axis, dtype=dtype, out=out, **kw)

    acc.__name__ = name
    acc.__qualname__ = f"ndarray.{name}"
    return acc


def install(ndarray):
    """Wrap the operator slots and reduction methods of `ndarray`."""
    for name, (fn, rev) in BINOPS.items():
        setattr(ndarray, name, _make_binop(ndarray, name, fn, rev))
    for name, fn in UNOPS.items():
        setattr(ndarray, name, _make_unop(ndarray, name, fn))
    for name, (fn, forced) in REDUCTIONS.items():
        if hasattr(ndarray, name):
            setattr(ndarray, name, _make_reduction(ndarray, name, fn, forced))
    for name, fn in ACCUMULATIONS.items():
        setattr(ndarray, name, _make_accumulation(ndarray, name, fn))


# ---------------------------------------------------------------------------
# The module-level spellings
# ---------------------------------------------------------------------------

def cumsum(a, axis=None, dtype=None, out=None):
    return _pkg().asarray(a).cumsum(axis=axis, dtype=dtype, out=out)


def cumprod(a, axis=None, dtype=None, out=None):
    return _pkg().asarray(a).cumprod(axis=axis, dtype=dtype, out=out)


def _wrap_free(fn, method):
    """`np.<method>(a, ...)` routed through the (already wrapped) method."""
    def free(a, axis=None, dtype=None, out=None, keepdims=False,
             initial=None, where=True, **kw):
        if isinstance(a, _rnp.ndarray) and type(a) is not _rnp.ndarray:
            return getattr(a, method)(axis, out)
        arr = _pkg().asarray(a)
        if arr.dtype.kind != "O":
            return fn(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
                      **kw) if dtype is not None else \
                fn(a, axis=axis, out=out, keepdims=keepdims, **kw)
        return getattr(arr, method)(
            axis=axis, dtype=dtype, out=out, keepdims=keepdims,
            initial=initial, where=where)

    free.__name__ = method
    return free


def _wrap_free_simple(fn, method):
    """As `_wrap_free`, for the reductions whose free form takes no `dtype`."""
    def free(a, axis=None, out=None, keepdims=False, initial=None,
             where=True, **kw):
        arr = _pkg().asarray(a)
        if arr.dtype.kind != "O":
            return fn(a, axis=axis, out=out, keepdims=keepdims, **kw)
        return getattr(arr, method)(axis=axis, out=out, keepdims=keepdims,
                                    initial=initial, where=where)

    free.__name__ = method
    return free


# ---------------------------------------------------------------------------
# Exception payloads the engine cannot build itself
# ---------------------------------------------------------------------------

def _cast_error(ufunc_name, casting, from_, to, i):
    """`_UFuncInputCastingError` for an object operand under an explicit
    non-object `dtype=`.  The engine knows only names, so it calls this."""
    pkg = _pkg()
    from ._core._exceptions import _UFuncInputCastingError
    return _UFuncInputCastingError(
        pkg._ufunc.ALL[ufunc_name], casting,
        pkg.dtype(from_), pkg.dtype(to), i)


_register = getattr(_rnp, "_register_object_cast_error", None)
if _register is not None:
    _register(_cast_error)

# `numpy.exceptions.AxisError` for the engine's axis bounds checks.
_register_axis = getattr(_rnp, "_register_axis_error", None)
if _register_axis is not None:
    from .exceptions import AxisError as _AxisError
    _register_axis(_AxisError)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------

def _is_object_dtype(x):
    try:
        return _pkg().dtype(x).kind == "O"
    except Exception:  # noqa: BLE001 - not a dtype at all; let the engine say so
        return False


def wrap_promote(orig):
    def promote_types(a, b):
        if a is b:
            return orig(a, b)
        if _is_object_dtype(a) or _is_object_dtype(b):
            return _pkg().dtype(object)
        return orig(a, b)
    return promote_types


def wrap_result_type(orig):
    def result_type(*arrays_and_dtypes):
        for x in arrays_and_dtypes:
            dt = getattr(x, "dtype", None)
            if dt is not None:
                if dt.kind == "O":
                    return _pkg().dtype(object)
            elif _is_object_dtype(x):
                return _pkg().dtype(object)
        return orig(*arrays_and_dtypes)
    return result_type
