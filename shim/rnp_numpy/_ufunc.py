"""`numpy.ufunc` — the object model, backed by the Rust engine.

Metadata (`nin`, `nout`, `identity`, `types`, `ntypes`) comes from
`_ufunc_table.py`, generated from real numpy 2.5.2, so introspection matches
exactly even where the port has no loop yet. Calling a ufunc the port has not
implemented raises NotImplementedError rather than returning a wrong answer.
"""

import builtins as _builtins

import _rnp

from . import _errstate
from ._ufunc_table import TABLE

ndarray = _rnp.ndarray

#: Names the Rust engine can evaluate. Anything else is metadata-only.
_IMPLEMENTED = frozenset("""
add subtract multiply divide true_divide floor_divide remainder mod fmod
power pow float_power divmod
equal not_equal less less_equal greater greater_equal
maximum minimum fmax fmin
arctan2 atan2 hypot copysign nextafter logaddexp logaddexp2 heaviside ldexp
bitwise_and bitwise_or bitwise_xor left_shift right_shift
bitwise_left_shift bitwise_right_shift gcd lcm
logical_and logical_or logical_xor logical_not
negative positive absolute abs fabs sign rint floor ceil trunc
sqrt cbrt square reciprocal
exp exp2 expm1 log log2 log10 log1p
sin cos tan arcsin arccos arctan asin acos atan
sinh cosh tanh arcsinh arccosh arctanh asinh acosh atanh
deg2rad rad2deg degrees radians
conj conjugate invert bitwise_invert bitwise_not bitwise_count
isnan isinf isfinite signbit spacing
frexp modf _ones_like
""".split())


def _is_scalarish(x):
    """True for the inputs that make a ufunc return a scalar rather than a
    0-d array (numpy's rule: every input is a scalar or a 0-d array)."""
    if isinstance(x, ndarray):
        return x.ndim == 0
    return not isinstance(x, (list, tuple))


class ufunc:
    """A numpy ufunc."""

    __module__ = "numpy"
    __slots__ = ("__name__", "_qualname", "nin", "nout", "nargs", "types",
                 "ntypes", "identity", "signature", "_ok")

    def __init__(self, name):
        meta = TABLE.get(name)
        if meta is None:
            meta = (2, 1, None, [])
        nin, nout, identity, types = meta
        self.__name__ = name
        self._qualname = f"numpy.{name}"
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self.types = list(types)
        self.ntypes = len(types)
        self.identity = identity
        self.signature = None
        self._ok = name in _IMPLEMENTED

    # -- introspection -----------------------------------------------------

    def __repr__(self):
        return f"<ufunc '{self.__name__}'>"

    @property
    def __doc__(self):
        return f"{self.__name__}(x1, x2, /, out=None, *, where=True, ...)"

    def _nope(self, what=""):
        suffix = f".{what}" if what else ""
        raise NotImplementedError(
            f"{self._qualname}{suffix} is not implemented by rnp yet")

    # -- the call ----------------------------------------------------------

    def __call__(self, *args, out=None, where=True, casting="same_kind",
                 order="K", dtype=None, subok=True, signature=None,
                 axes=None, axis=None, keepdims=None):
        if not self._ok:
            self._nope()
        if len(args) > self.nin:
            # numpy allows the outputs to be passed positionally.
            out = args[self.nin] if out is None else out
            args = args[: self.nin]
        if len(args) != self.nin:
            raise TypeError(
                f"invalid number of arguments to ufunc {self.__name__!r}")
        if isinstance(out, tuple):
            out = out[0] if len(out) == 1 else out
        scalar_out = out is None and _builtins.all(
            _is_scalarish(a) for a in args)
        res = _rnp._ufunc_call(self.__name__, args, out=out, where_=where,
                               casting=casting, dtype=dtype)
        _errstate.drain(self.__name__)
        if isinstance(res, tuple):
            return tuple(_maybe_scalar(r, scalar_out) for r in res)
        return _maybe_scalar(res, scalar_out)

    # -- methods -----------------------------------------------------------

    def reduce(self, array, axis=0, dtype=None, out=None, keepdims=False,
               initial=None, where=True):
        if not self._ok or self.nin != 2:
            self._nope("reduce")
        if isinstance(out, tuple):
            out = out[0]
        res = _rnp._ufunc_reduce(self.__name__, array, axis=axis, dtype=dtype,
                                 out=out, keepdims=keepdims, initial=initial,
                                 where_=where)
        _errstate.drain(self.__name__)
        return _maybe_scalar(res, out is None)

    def accumulate(self, array, axis=0, dtype=None, out=None):
        if not self._ok or self.nin != 2:
            self._nope("accumulate")
        if isinstance(out, tuple):
            out = out[0]
        res = _rnp._ufunc_accumulate(self.__name__, array, axis=axis,
                                     dtype=dtype, out=out)
        _errstate.drain(self.__name__)
        return res

    def reduceat(self, array, indices, axis=0, dtype=None, out=None):
        if not self._ok or self.nin != 2:
            self._nope("reduceat")
        a = _rnp.asarray(array)
        idx = [_builtins.int(i) for i in _rnp.asarray(indices).tolist()] \
            if not isinstance(indices, int) else [indices]
        n = a.shape[axis]
        pieces = []
        for k, start in enumerate(idx):
            if start < 0:
                start += n
            if not 0 <= start < n:
                raise IndexError(
                    f"index {idx[k]} out-of-bounds in {self.__name__}.reduceat")
            nxt = idx[k + 1] if k + 1 < len(idx) else n
            if nxt < 0:
                nxt += n
            stop = nxt if nxt > start else start + 1
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(start, min(stop, n))
            piece = a[tuple(sl)]
            red = self.reduce(piece, axis=axis, dtype=dtype)
            pieces.append(_rnp.asarray(red))
        from ._core.shape_base import stack
        res = stack(pieces, axis=axis)
        if out is not None:
            out[...] = res
            return out
        return res

    def outer(self, a, b, **kwargs):
        if not self._ok or self.nin != 2:
            self._nope("outer")
        a = _rnp.asarray(a)
        b = _rnp.asarray(b)
        a2 = a.reshape(a.shape + (1,) * b.ndim)
        return self(a2, b, **kwargs)

    def at(self, a, indices, b=None):
        if not self._ok:
            self._nope("at")
        if self.nin == 1:
            a[indices] = self(a[indices])
            return None
        # Unbuffered: repeated indices must each take effect, so the update is
        # applied one element at a time.
        idx = _rnp.asarray(indices)
        if idx.dtype.kind == "b" or idx.ndim == 0:
            a[indices] = self(a[indices], b)
            return None
        flat = idx.tolist()
        vals = b
        b_arr = _rnp.asarray(b) if b is not None else None
        for k, i in enumerate(flat):
            v = b_arr[k] if b_arr is not None and b_arr.ndim > 0 else vals
            a[i] = self(a[i], v)
        return None

    def resolve_dtypes(self, dtypes, *, signature=None, casting=None,
                       reduction=False):
        self._nope("resolve_dtypes")


def _maybe_scalar(res, want_scalar):
    """numpy returns a scalar when every input was one."""
    if want_scalar and isinstance(res, ndarray) and res.ndim == 0:
        return res[()]
    return res


#: numpy binds several names to the *same* ufunc object, so `np.acos is
#: np.arccos` and `np.acos.__name__ == "arccos"`. Probed from numpy 2.5.2.
_ALIASES = {
    "abs": "absolute",
    "acos": "arccos", "acosh": "arccosh",
    "asin": "arcsin", "asinh": "arcsinh",
    "atan": "arctan", "atan2": "arctan2", "atanh": "arctanh",
    "bitwise_invert": "invert", "bitwise_not": "invert",
    "bitwise_left_shift": "left_shift",
    "bitwise_right_shift": "right_shift",
    "conj": "conjugate",
    "true_divide": "divide",
    "mod": "remainder",
    "pow": "power",
}

#: One ufunc object per *canonical* name numpy exposes, plus the aliases
#: pointing at the same object.
ALL = {name: ufunc(name) for name in TABLE if name not in _ALIASES}
for _alias, _canon in _ALIASES.items():
    ALL[_alias] = ALL[_canon]
del _alias, _canon
