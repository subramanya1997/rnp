"""`numpy.ufunc` — the object model, backed by the Rust engine.

Metadata (`nin`, `nout`, `identity`, `types`, `ntypes`) comes from
`_ufunc_table.py`, generated from real numpy 2.5.2, so introspection matches
exactly even where the port has no loop yet. Calling a ufunc the port has not
implemented raises NotImplementedError rather than returning a wrong answer.
"""

import builtins as _builtins
import operator as _operator

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
isnan isinf isfinite isnat signbit spacing
frexp modf _ones_like
matmul vecdot matvec vecmat
""".split())


#: The four ufuncs numpy gives a *core signature*. They are dispatched
#: through `_rnp._matmul` rather than the elementwise loop selector, because
#: the operands' trailing dimensions are consumed by the signature rather
#: than broadcast. Probed from numpy 2.5.2.
GUFUNC_SIGNATURES = {
    "matmul": "(n?,k),(k,m?)->(n?,m?)",
    "vecdot": "(n),(n)->()",
    "matvec": "(m,n),(n)->(m)",
    "vecmat": "(n),(n,m)->(m)",
}

#: How many core dimensions each operand of each gufunc has, in the order
#: (input 0, input 1, output). Used only by the `axes=` rewrite.
_GUFUNC_CORE_NDIM = {
    "matmul": (2, 2, 2),
    "vecdot": (1, 1, 0),
    "matvec": (2, 1, 1),
    "vecmat": (1, 2, 1),
}


class _Unset:
    """Marker for "the caller did not pass this keyword at all"."""

    def __repr__(self):
        return "<unset>"


_UNSET = _Unset()


#: Ufuncs numpy gives string loops that the engine has no loop for.  Each
#: maps to the equivalent `numpy.strings` implementation, which is where the
#: real work (and numpy's itemsize arithmetic) already lives.
_STRING_LOOPS = {
    "add": "add", "multiply": "multiply",
    "equal": "equal", "not_equal": "not_equal", "less": "less",
    "less_equal": "less_equal", "greater": "greater",
    "greater_equal": "greater_equal",
    "maximum": "maximum", "minimum": "minimum",
}


def _string_kinds(args):
    """Dtype kinds of `args`, or None if any operand is not array-like."""
    kinds = []
    for a in args:
        dt = getattr(a, "dtype", None)
        if dt is None:
            try:
                dt = _rnp.asarray(a).dtype
            except Exception:
                return None
        kinds.append(dt.kind)
    return kinds


#: Comparisons numpy gives `SS->?` and `UU->?` loops, but no mixed loop.
_STRING_COMPARISONS = frozenset(
    "equal not_equal less less_equal greater greater_equal".split())


def _mixed_string_comparison(name, args, signature, casting):
    """Comparison of a bytes array against a str array.

    numpy has no mixed loop, so an unqualified call is a "did not contain a
    loop" TypeError.  An explicit `signature` plus a permissive `casting`
    is the documented way to ask for one side to be cast, and that does
    work — so honour it rather than reporting no loop.
    """
    if name not in _STRING_COMPARISONS:
        return None
    kinds = _string_kinds(args)
    if kinds is None or sorted(kinds) != ["S", "U"]:
        return None

    target = None
    if isinstance(signature, str) and "->" in signature:
        target = signature.split("->")[0].strip()[:1]
    elif isinstance(signature, (tuple, list)) and signature:
        first = signature[0]
        target = getattr(first, "kind", None) or str(first)[:1]
    if target in ("S", "U") and casting in ("unsafe", "same_kind"):
        cast = [_rnp.asarray(a).astype(target) for a in args]
        return _rnp._ufunc_call(name, tuple(cast), out=None, where_=True,
                                casting=casting, dtype=None)

    from ._core._exceptions import _UFuncNoLoopError
    from . import dtypes as _dtypes
    classes = tuple(
        _dtypes.BytesDType if k == "S" else _dtypes.StrDType for k in kinds)
    from . import _ufunc as _self
    raise _UFuncNoLoopError(getattr(_self, "ALL")[name], classes + (None,))


def _string_loop_fallback(name, args):
    """Evaluate a string ufunc through `numpy.strings`, or return None.

    Returning None means "no string loop applies" and the caller re-raises
    the engine's error — which is what keeps genuinely absent loops (notably
    `S` against `U`, which numpy also refuses) reporting as errors rather
    than being silently coerced.
    """
    fn = _STRING_LOOPS.get(name)
    if fn is None:
        return None
    kinds = _string_kinds(args)
    if kinds is None or not any(k in "SUT" for k in kinds):
        return None
    if name == "add":
        # numpy has S+S and U+U loops but deliberately no mixed S/U loop.
        if (len(kinds) != 2
                or ("T" not in kinds and set(kinds) not in ({"S"}, {"U"}))):
            return None
    elif name == "multiply":
        # One string operand repeated an integer number of times.
        if len(kinds) != 2 or sorted(
            "s" if k in "SUT" else "i" if k in "iub" else "?" for k in kinds
        ) != ["i", "s"]:
            return None
        if kinds[0] not in "SUT":  # numpy accepts either order
            args = (args[1], args[0])
    from . import strings as _strings
    return getattr(_strings, fn)(*args)


def _string_isnan(args):
    """StringDType's ``isnan`` loop classifies only its configured NA.

    A string-valued, ``None``, or pandas-style sentinel is not NaN.  NumPy
    classifies a floating NaN sentinel once on the descriptor and then uses
    identity when reading array elements; the storage layer likewise retains
    that exact Python object.
    """
    if len(args) != 1 or _string_kinds(args) != ["T"]:
        return None
    arr = _rnp.asarray(args[0])
    na = getattr(arr.dtype, "na_object", _UNSET)
    nan_sentinel = isinstance(na, float) and na != na
    if na is not _UNSET and not nan_sentinel:
        try:
            bool(na)
        except TypeError:
            nan_sentinel = True
    vals = arr.reshape(-1).tolist() if arr.ndim else [arr.tolist()]
    return _rnp.asarray([nan_sentinel and v is na for v in vals]).reshape(
        arr.shape)


def _index_product(shape):
    """Yield index tuples without depending on numpy's Python iterators."""
    if not shape:
        yield ()
        return
    tail = shape[1:]
    for i in range(shape[0]):
        for rest in _index_product(tail):
            yield (i,) + rest


def _string_add_reduce(array, axis, dtype, out, keepdims, initial, where):
    """Reference-safe StringDType add reduction used by ``reduce``."""
    if where is not True:
        return None
    arr = _rnp.asarray(array)
    target = _rnp.dtype(dtype) if dtype is not None else arr.dtype
    if target.kind != "T" and arr.dtype.kind != "T":
        return None
    if arr.dtype != target:
        arr = arr.astype(target)
    if axis is None:
        arr = arr.reshape((-1,))
        axis = 0
    axis = int(axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        return None
    outer_shape = arr.shape[:axis] + arr.shape[axis + 1:]
    result_shape = (arr.shape[:axis] + (1,) + arr.shape[axis + 1:]
                    if keepdims else outer_shape)
    from . import empty
    result = empty(result_shape, dtype=target)
    for outer in _index_product(outer_shape):
        values = []
        for i in range(arr.shape[axis]):
            idx = outer[:axis] + (i,) + outer[axis:]
            values.append(arr[idx])
        if initial is None:
            if not values:
                raise ValueError(
                    "zero-size array to reduction operation add which has no identity")
            acc, values = values[0], values[1:]
        else:
            acc = initial
        for value in values:
            if isinstance(acc, str) and isinstance(value, str):
                acc += value
            else:
                # Descriptor nulls that are NaN-like propagate; all other
                # non-string nulls cannot participate in concatenation.
                from ._core.strings import _is_nan_na
                if _is_nan_na(arr, acc):
                    continue
                if _is_nan_na(arr, value):
                    acc = value
                    continue
                raise ValueError("Cannot compare null that is not a nan-like value")
        dest = (outer[:axis] + (0,) + outer[axis:]
                if keepdims else outer)
        result[dest] = acc
    if out is not None:
        out[...] = result.tolist()
        return out
    return result


def _string_add_accumulate(array, axis, dtype, out):
    """Reference-safe StringDType add accumulation along one axis."""
    arr = _rnp.asarray(array)
    target = _rnp.dtype(dtype) if dtype is not None else arr.dtype
    if target.kind != "T" and arr.dtype.kind != "T":
        return None
    if arr.dtype != target:
        arr = arr.astype(target)
    from . import empty
    result = empty(arr.shape, dtype=target)
    outer_shape = arr.shape[:axis] + arr.shape[axis + 1:]
    for outer in _index_product(outer_shape):
        acc = None
        for i in range(arr.shape[axis]):
            idx = outer[:axis] + (i,) + outer[axis:]
            value = arr[idx]
            if i == 0:
                acc = value
            elif isinstance(acc, str) and isinstance(value, str):
                acc += value
            else:
                from ._core.strings import _is_nan_na
                if not _is_nan_na(arr, acc):
                    if _is_nan_na(arr, value):
                        acc = value
                    else:
                        raise ValueError(
                            "Cannot compare null that is not a nan-like value")
            result[idx] = acc
    if out is not None:
        out[...] = result.tolist()
        return out
    return result


def _broadcast_against_out(args, oshape):
    """numpy broadcasts the *output* together with the inputs, so an input of
    shape (1,) against ``out`` of shape (1000,) is legal.  The engine expects
    the inputs to already match, so widen them here."""
    shapes = [a.shape for a in args if isinstance(a, ndarray)]
    if not shapes or _builtins.all(s == oshape for s in shapes):
        return args
    try:
        target = _rnp.broadcast_shapes(*shapes, oshape)
    except Exception:  # noqa: BLE001 - let the engine report the real error
        return args
    if target != oshape:
        return args
    return tuple(
        _rnp.broadcast_to(a, target)
        if isinstance(a, ndarray) and a.shape != target else a
        for a in args)


def _single_axis(array, axis, what):
    """``axis=None`` means "every axis", which `accumulate`/`reduceat` only
    accept when there is at most one."""
    if axis is not None:
        return axis
    ndim = array.ndim if isinstance(array, ndarray) else _rnp.asarray(array).ndim
    if ndim > 1:
        raise ValueError(f"{what} does not allow multiple axes")
    return 0


def _detached(x, target):
    """`x`, guaranteed not to alias `target`."""
    if not isinstance(x, ndarray) or not isinstance(target, ndarray):
        return x
    if not x.size or not target.size:
        return x
    import importlib
    mod = importlib.import_module(__name__.rsplit(".", 1)[0]
                                  + "._core._memoverlap")
    return x.copy() if mod.may_share_memory(x, target) else x


def _obj_kind(x):
    """The dtype kind of `x` as an array operand, or None if it is not one."""
    dt = getattr(x, "dtype", None)
    if dt is None:
        try:
            dt = _rnp.asarray(x).dtype
        except Exception:  # noqa: BLE001
            return None
    return dt.kind


def _object_gufunc(name, a, b, out):
    """`matmul` and friends over `object` arrays.

    numpy's `OBJECT_matmul_inner_noblas` builds each output element as
    ``a[i,0]*b[0,j]`` and then repeatedly ``+=`` the remaining products, so
    every operation goes through the *elements'* own `__mul__`/`__add__`, in
    `k` order, and an empty `k` yields the integer 0. That is reproduced here
    literally, element by element: the engine has no object-dtype elementwise
    loops to delegate to, and even if it did, whole-array slicing would not
    guarantee this exact left-to-right association for user-defined types.

    The dimension algebra (which optional core dimensions survive, what the
    loop shape is) is the same as for every other dtype, so it is asked of
    the Rust planner rather than duplicated.
    """
    from . import array, asarray, broadcast_to

    aa, bb = asarray(a), asarray(b)
    loop, out_shape, rows, inner, cols, a_rowless, b_colless = _rnp._matmul_plan(
        name, aa.shape, bb.shape,
        None if out is None else out.shape)
    loop, out_shape = tuple(loop), tuple(out_shape)
    a2 = aa.reshape(aa.shape[:-1] + (1, aa.shape[-1])) if a_rowless else aa
    b2 = bb.reshape(bb.shape + (1,)) if b_colless else bb
    a2 = broadcast_to(a2, loop + (rows, inner))
    b2 = broadcast_to(b2, loop + (inner, cols))

    nbatch = 1
    for d in loop:
        nbatch *= d
    flat = []
    for batch in range(nbatch):
        rem, prefix = batch, []
        for d in reversed(loop):
            prefix.append(rem % d)
            rem //= d
        prefix = tuple(reversed(prefix))
        for i in range(rows):
            for j in range(cols):
                acc = 0
                for t in range(inner):
                    term = a2[prefix + (i, t)] * b2[prefix + (t, j)]
                    acc = term if t == 0 else acc + term
                flat.append(acc)
    res = array(flat, dtype=object).reshape(out_shape) if flat else \
        array([], dtype=object).reshape(out_shape)
    if out is not None:
        out[...] = res
        return out
    return res


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
                 "ntypes", "identity", "signature", "_ok", "_pyfunc",
                 "_doc", "_identity_set")

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
        self.signature = GUFUNC_SIGNATURES.get(name)
        self._ok = name in _IMPLEMENTED
        self._pyfunc = None
        self._doc = None
        self._identity_set = True

    @classmethod
    def _from_pyfunc(cls, func, nin, nout, identity, identity_set):
        name = f"{getattr(func, '__name__', '?')} (vectorized)"
        obj = cls(name)
        obj.nin = nin
        obj.nout = nout
        obj.nargs = nin + nout
        obj.types = ["O" * nin + "->" + "O" * nout]
        obj.ntypes = 1
        obj.identity = identity
        obj.signature = None
        obj._ok = True
        obj._pyfunc = func
        args = "x" if nin == 1 else ", ".join(
            f"x{i}" for i in range(1, nin + 1))
        obj._doc = (
            f"{name}({args}, /, out=None, *, where=True, "
            "casting='same_kind', order='K', dtype=None, subok=True"
            "[, signature])\n\ndynamic ufunc based on a python function"
        )
        obj._identity_set = identity_set
        return obj

    # -- introspection -----------------------------------------------------

    def __repr__(self):
        return f"<ufunc '{self.__name__}'>"

    @property
    def __doc__(self):
        if self._doc is not None:
            return self._doc
        return f"{self.__name__}(x1, x2, /, out=None, *, where=True, ...)"

    def _no_signature_method(self, what):
        """numpy refuses every reduction method on a ufunc with a signature,
        with one shared RuntimeError."""
        if self.signature is not None:
            raise RuntimeError("Reduction not defined on ufunc with signature")

    def _nope(self, what=""):
        suffix = f".{what}" if what else ""
        raise NotImplementedError(
            f"{self._qualname}{suffix} is not implemented by rnp yet")

    # -- the call ----------------------------------------------------------

    def __call__(self, *args, out=None, where=_UNSET, casting="same_kind",
                 order="K", dtype=None, subok=True, signature=None,
                 axes=None, axis=None, keepdims=None):
        if not self._ok:
            self._nope()
        if not isinstance(casting, str):
            raise TypeError(
                f"casting must be str, not {type(casting).__name__}")
        if casting not in ("no", "equiv", "safe", "same_kind", "unsafe"):
            raise ValueError(
                "casting must be one of 'no', 'equiv', 'safe', "
                f"'same_kind', 'unsafe' (got {casting!r})")
        if self.signature is not None:
            # A gufunc has no `where=`: numpy's argument parser does not even
            # define the keyword for it.
            if where is not _UNSET:
                raise TypeError(
                    f"{self.__name__}() got an unexpected keyword argument "
                    "'where'")
            if len(args) > self.nin:
                out = args[self.nin] if out is None else out
                args = args[:self.nin]
            if len(args) != self.nin:
                raise TypeError(
                    f"invalid number of arguments to ufunc {self.__name__!r}")
            return self._call_gufunc(args, out, casting, dtype, axes, axis)
        where = True if where is _UNSET else where
        if len(args) > self.nin:
            # numpy allows the outputs to be passed positionally.
            out = args[self.nin] if out is None else out
            args = args[: self.nin]
        if len(args) != self.nin:
            raise TypeError(
                f"invalid number of arguments to ufunc {self.__name__!r}")
        if isinstance(out, tuple):
            out = out[0] if len(out) == 1 else out
        # NumPy's private quantile path uses ``out=...`` to request a newly
        # allocated ndarray.  It differs from omitted ``out`` only for scalar
        # inputs, where the result must remain a 0-d array rather than being
        # unboxed to a NumPy scalar.
        allocate_out = out is Ellipsis
        if allocate_out:
            out = None
        scalar_out = out is None and not allocate_out and _builtins.all(
            _is_scalarish(a) for a in args)
        if isinstance(out, ndarray):
            args = _broadcast_against_out(args, out.shape)
            # Real NumPy has a concrete DType metaclass per dtype, so
            # ``type(out.dtype)`` identifies the loop dtype.  The engine uses
            # one generic ``dtype`` class; recover the intended concrete dtype
            # from the output array when upstream code passes that class.
            if dtype is _rnp.dtype:
                dtype = out.dtype
        override = self._dispatch_pyfunc_override(args, out)
        if override is not _UNSET:
            return override
        if self._pyfunc is not None:
            return self._call_pyfunc(
                args, out, where, casting, dtype, allocate_out, scalar_out)
        res = None
        if (self.__name__ == "isnan" and out is None and dtype is None
                and where is True):
            res = _string_isnan(args)
        kinds = _string_kinds(args)
        if (res is None and dtype is None and where is True
                and kinds is not None and "T" in kinds):
            res = _string_loop_fallback(self.__name__, args)
            if res is not None and out is not None:
                out[...] = res.tolist()
                res = out
        if res is not None:
            return _maybe_scalar(res, scalar_out)
        try:
            res = _rnp._ufunc_call(self.__name__, args, out=out, where_=where,
                                   casting=casting, dtype=dtype)
        except NotImplementedError:
            res = None
            if out is None and dtype is None:
                res = _mixed_string_comparison(
                    self.__name__, args, signature, casting)
                if res is None and signature is None:
                    res = _string_loop_fallback(self.__name__, args)
            if res is None:
                raise
        _errstate.drain(self.__name__)
        if isinstance(res, tuple):
            return tuple(_maybe_scalar(r, scalar_out) for r in res)
        return _maybe_scalar(res, scalar_out)

    def _dispatch_pyfunc_override(self, args, out):
        """NEP 13 override ordering for dynamically-created ufuncs."""
        candidates = []
        types = []
        operands = list(args)
        if isinstance(out, tuple):
            operands.extend(x for x in out if x is not None)
        elif out is not None:
            operands.append(out)
        for operand in operands:
            typ = type(operand)
            method = getattr(typ, "__array_ufunc__", None)
            if method is None or typ in types or typ is ndarray:
                continue
            insert = len(types)
            for i, old in enumerate(types):
                if issubclass(typ, old):
                    insert = i
                    break
            types.insert(insert, typ)
            candidates.insert(insert, (operand, method))
        if not candidates:
            return _UNSET
        kwargs = {} if out is None else {
            "out": out if isinstance(out, tuple) else (out,)}
        for operand, method in candidates:
            result = method(operand, self, "__call__", *args, **kwargs)
            if result is not NotImplemented:
                return result
        names = ", ".join(t.__name__ for t in types)
        raise TypeError(
            f"operand type(s) all returned NotImplemented from "
            f"__array_ufunc__({self!r}, '__call__', ...): {names}")

    def _call_pyfunc(self, args, out, where, casting, dtype,
                     allocate_out, scalar_out):
        """Run a `frompyfunc` loop over broadcast object operands."""
        if dtype is not None and _rnp.dtype(dtype).kind != "O":
            from ._core._exceptions import _UFuncNoLoopError
            raise _UFuncNoLoopError(
                self, tuple([_rnp.dtype(object)] * self.nargs))

        from . import asarray as _asarray
        arrays = [_asarray(a, dtype=object) for a in args]
        if self.nout == 1:
            destinations = [out]
        elif out is None:
            destinations = [None] * self.nout
        elif isinstance(out, tuple) and len(out) == self.nout:
            destinations = list(out)
        else:
            raise ValueError(
                "The 'out' tuple must have exactly one entry per ufunc output")

        concrete = [o for o in destinations if o is not None]
        if _builtins.any(not isinstance(o, ndarray) for o in concrete):
            raise TypeError("return arrays must be of ArrayType")
        shapes = [a.shape for a in arrays]
        shapes.extend(o.shape for o in concrete)
        shape = tuple(_rnp.broadcast_shapes(*shapes)) if shapes else ()
        if _builtins.any(o.shape != shape for o in concrete):
            raise ValueError("operands could not be broadcast together")
        arrays = [_rnp.broadcast_to(a, shape) if a.shape != shape else a
                  for a in arrays]

        mask = _asarray(where, dtype=bool)
        if mask.shape != shape:
            mask = _rnp.broadcast_to(mask, shape)

        object_dtype = _rnp.dtype(object)
        for i, dest in enumerate(destinations):
            if dest is not None and not _rnp.can_cast(
                    object_dtype, dest.dtype, casting=casting):
                from ._core._exceptions import _UFuncOutputCastingError
                raise _UFuncOutputCastingError(
                    self, casting, object_dtype, dest.dtype, i)

        size = 1
        for dim in shape:
            size *= dim
        values = [[None] * size for _ in range(self.nout)]

        def c_index(flat):
            if not shape:
                return ()
            index = [0] * len(shape)
            for axis in range(len(shape) - 1, -1, -1):
                index[axis] = flat % shape[axis]
                flat //= shape[axis]
            return tuple(index)

        for flat in range(size):
            index = c_index(flat)
            if not bool(mask[index]):
                continue
            result = self._pyfunc(*(a[index] for a in arrays))
            if self.nout == 1:
                values[0][flat] = result
            elif isinstance(result, tuple) and len(result) == self.nout:
                for k, value in enumerate(result):
                    values[k][flat] = value
            # NumPy fills every output with None when a multi-output callback
            # returns anything other than the exact expected tuple.

        results = []
        for k, dest in enumerate(destinations):
            if dest is None:
                # Building from the nested Python values would interpret a
                # tuple/list returned by the callback as another array axis.
                # Allocate the object container first, then store each result
                # as one opaque Python object.
                res = _rnp.empty(shape, dtype=object)
                for flat, value in enumerate(values[k]):
                    res[c_index(flat)] = value
            else:
                for flat, value in enumerate(values[k]):
                    index = c_index(flat)
                    if bool(mask[index]):
                        dest[index] = value
                res = dest
            results.append(res)

        if self.nout == 1:
            return _maybe_scalar(results[0], scalar_out and not allocate_out)
        if scalar_out and not allocate_out:
            return tuple(_maybe_scalar(r, True) for r in results)
        return tuple(results)

    # -- the generalized (core-signature) call ------------------------------

    def _call_gufunc(self, args, out, casting, dtype, axes, axis):
        name = self.__name__
        if isinstance(out, tuple):
            if len(out) != 1:
                raise ValueError(
                    f"The 'out' tuple must have exactly one entry per ufunc "
                    f"output")
            out = out[0]
        if out is not None and not isinstance(out, ndarray):
            raise TypeError("return arrays must be of ArrayType")
        if axis is not None:
            # numpy only accepts `axis=` for gufuncs whose operands share a
            # single core dimension; none of this family does.
            raise TypeError(
                f"{name}: axis can only be used with a single shared core "
                "dimension, not with the 2 distinct ones implied by "
                f"signature {self.signature}.")
        if axes is not None:
            return self._call_with_axes(args, out, casting, dtype, axes)

        a, b = args
        if _obj_kind(a) == "O" or _obj_kind(b) == "O":
            return _object_gufunc(name, a, b, out)

        if out is not None:
            loop_dt = (_rnp.dtype(dtype) if dtype is not None
                       else _rnp.dtype(_rnp._matmul_dtype(name, a, b)))
            if not _rnp.can_cast(loop_dt, out.dtype, casting):
                from ._core._exceptions import _UFuncOutputCastingError
                raise _UFuncOutputCastingError(
                    self, casting, loop_dt, out.dtype, 0)

        res = _rnp._matmul(name, a, b, out=out, dtype=dtype)
        _errstate.drain(name)
        if out is None and isinstance(res, ndarray) and res.ndim == 0:
            # Like every ufunc, a 0-d result comes back as a numpy scalar.
            return res[()]
        return res

    def _call_with_axes(self, args, out, casting, dtype, axes):
        """`axes=[(..), (..), (..)]`: run the core dimensions from wherever
        the caller put them, and put the result's back where asked.

        numpy implements this by remapping the iterator's axes; moving the
        core dimensions to the end with a transpose and undoing it afterwards
        gives the same answer without a second iterator.
        """
        axes = list(axes)
        if len(axes) not in (self.nin, self.nin + self.nout):
            raise ValueError(
                "axes don't match ufunc: must have entries for all inputs "
                "(and outputs if they have core dimensions)")

        def norm(entry, nd):
            """One `axes` entry as a tuple of non-negative axis numbers.

            numpy accepts a bare int wherever the operand has exactly one core
            dimension, which is how `axes=[(1, 0), (0), (0)]` -- the spelling
            upstream's own test uses -- gets parsed.
            """
            if isinstance(entry, int):
                entry = (entry,)
            entry = tuple(int(x) for x in entry)
            return tuple(x + nd if x < 0 else x for x in entry)

        moved = []
        for i, operand in enumerate(args):
            arr = _rnp.asarray(operand)
            src = norm(axes[i], arr.ndim)
            if not src:
                moved.append(arr)
                continue
            rest = [d for d in range(arr.ndim) if d not in src]
            moved.append(arr.transpose(tuple(rest + list(src))))
        res = self._call_gufunc(tuple(moved), None, casting, dtype, None, None)

        if len(axes) > self.nin:
            arr = _rnp.asarray(res)
            dest = norm(axes[self.nin], arr.ndim)
            want_out = len(dest)
            order = [None] * arr.ndim
            core_start = arr.ndim - want_out
            for j, pos in enumerate(dest):
                order[pos] = core_start + j
            it = iter(range(core_start))
            order = [x if x is not None else next(it) for x in order]
            res = arr.transpose(tuple(order))
        if out is not None:
            out[...] = res
            return out
        return res

    # -- methods -----------------------------------------------------------

    def reduce(self, array, axis=0, dtype=None, out=None, keepdims=False,
               initial=None, where=True):
        return self._reduce_impl(
            array, axis, dtype, out, keepdims, initial, where, "reduce")

    def _reduce_impl(self, array, axis, dtype, out, keepdims, initial, where,
                     error_context):
        if self._pyfunc is not None:
            return self._reduce_pyfunc(
                array, axis, dtype, out, keepdims, initial, where)
        self._no_signature_method("reduce")
        if not self._ok or self.nin != 2:
            self._nope("reduce")
        if isinstance(out, tuple):
            out = out[0]
        if self.__name__ == "add":
            if dtype is None and _rnp.asarray(array).dtype.kind in "SU":
                raise TypeError(
                    "the resolved dtypes are not compatible with add.reduce")
            res = _string_add_reduce(
                array, axis, dtype, out, keepdims, initial, where)
            if res is not None:
                return _maybe_scalar(res, out is None)
        res = _rnp._ufunc_reduce(self.__name__, array, axis=axis, dtype=dtype,
                                 out=out, keepdims=keepdims, initial=initial,
                                 where_=where)
        _errstate.drain(error_context)
        return _maybe_scalar(res, out is None)

    def _reduce_pyfunc(self, array, axis, dtype, out, keepdims,
                       initial, where):
        """Object reduction used by binary frompyfunc ufuncs."""
        if self.nin != 2 or self.nout != 1:
            self._nope("reduce")
        if dtype is not None and _rnp.dtype(dtype).kind != "O":
            from ._core._exceptions import _UFuncNoLoopError
            raise _UFuncNoLoopError(
                self, tuple([_rnp.dtype(object)] * self.nargs))
        if isinstance(out, tuple):
            out = out[0]

        from . import asarray as _asarray
        arr = _asarray(array, dtype=object)
        ndim = arr.ndim
        if axis is None:
            axes = tuple(range(ndim))
        elif isinstance(axis, tuple):
            axes = tuple(axis)
        else:
            axes = (axis,)
        normalized = []
        for original in axes:
            value = _operator.index(original)
            value = value + ndim if value < 0 else value
            if ndim == 0 and value in (0, -1):
                value = 0
            if value < 0 or value >= max(ndim, 1):
                from .exceptions import AxisError
                raise AxisError(
                    f"axis {original} is out of bounds for array of dimension {ndim}")
            if value in normalized:
                raise ValueError("duplicate value in 'axis'")
            normalized.append(value)
        axes = tuple(normalized)
        if len(axes) > 1 and not self._identity_set:
            raise ValueError(
                "reduction operation is not reorderable, so at most one axis may be specified")

        if not axes:
            res = arr.copy()
            if out is not None:
                out[...] = res
                return out
            return res

        out_shape = tuple(
            1 if keepdims and i in axes else dim
            for i, dim in enumerate(arr.shape) if keepdims or i not in axes)
        mask = _asarray(where, dtype=bool)
        if mask.shape != arr.shape:
            mask = _rnp.broadcast_to(mask, arr.shape)

        def indices(shape):
            size = 1
            for dim in shape:
                size *= dim
            for flat in range(size):
                index = [0] * len(shape)
                rem = flat
                for i in range(len(shape) - 1, -1, -1):
                    index[i] = rem % shape[i]
                    rem //= shape[i]
                yield tuple(index)

        groups = {index: [] for index in indices(out_shape)}
        for index in indices(arr.shape):
            if not bool(mask[index]):
                continue
            if keepdims:
                key = tuple(0 if i in axes else value
                            for i, value in enumerate(index))
            else:
                key = tuple(value for i, value in enumerate(index)
                            if i not in axes)
            groups[key].append(arr[index])

        res = _rnp.empty(out_shape, dtype=object)
        for index, values in groups.items():
            if initial is not None:
                acc = initial
                rest = values
            elif self._identity_set and self.identity is not None:
                acc = self.identity
                rest = values
            elif values:
                acc = values[0]
                rest = values[1:]
            else:
                raise ValueError(
                    f"zero-size array to reduction operation {self.__name__} "
                    "which has no identity")
            for value in rest:
                acc = self._pyfunc(acc, value)
            res[index] = acc

        if out is not None:
            if not isinstance(out, ndarray):
                raise TypeError("return arrays must be of ArrayType")
            if out.shape != out_shape:
                raise ValueError("operands could not be broadcast together")
            object_dtype = _rnp.dtype(object)
            if not _rnp.can_cast(object_dtype, out.dtype, casting="same_kind"):
                from ._core._exceptions import _UFuncOutputCastingError
                raise _UFuncOutputCastingError(
                    self, "same_kind", object_dtype, out.dtype, 0)
            out[...] = res
            return out
        return _maybe_scalar(res, not keepdims and not out_shape)

    def accumulate(self, array, axis=0, dtype=None, out=None):
        self._no_signature_method("accumulate")
        if not self._ok or self.nin != 2:
            self._nope("accumulate")
        if isinstance(out, tuple):
            out = out[0]
        axis = _single_axis(array, axis, "accumulate")
        if self.__name__ == "add":
            norm_axis = axis
            arr = _rnp.asarray(array)
            if norm_axis < 0:
                norm_axis += arr.ndim
            res = _string_add_accumulate(arr, norm_axis, dtype, out)
            if res is not None:
                return res
        res = _rnp._ufunc_accumulate(self.__name__, array, axis=axis,
                                     dtype=dtype, out=out)
        _errstate.drain("accumulate")
        return res

    def reduceat(self, array, indices, axis=0, dtype=None, out=None):
        self._no_signature_method("reduceat")
        if not self._ok or self.nin != 2:
            self._nope("reduceat")
        axis = _single_axis(array, axis, "reduceat")
        a = _rnp.asarray(array)
        idx = [_builtins.int(i) for i in _rnp.asarray(indices).tolist()] \
            if not isinstance(indices, int) else [indices]
        n = a.shape[axis]
        # See the comment on the branch below.
        # Object `add` folds sequentially, not with a pairwise tree, so the
        # head/tail split below would only change the bracketing for nothing.
        _split_head = self.__name__ == "add" and a.dtype.kind != "O"
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
            stop = _builtins.min(nxt if nxt > start else start + 1, n)
            sl = [slice(None)] * a.ndim
            if _split_head:
                # numpy's `PyUFunc_Reduceat` copies `a[start]` into the output
                # and then runs the inner loop over the *remaining* count - 1
                # elements, so the segment fold is
                # `a[start] + pairwise(a[start+1:stop])`. Only `add` reduces
                # with the pairwise tree (`PW = 1, 0, 0, 0` in
                # `loops_arithm_fp.dispatch.c.src`); for every other ufunc the
                # inner loop is a sequential accumulator, where seeding it
                # with `a[start]` and seeding it with the identity give
                # bit-identical results, so the plain segment reduce below is
                # already right. For `add` they disagree from segment length
                # 3 upwards.
                sl[axis] = slice(start, start + 1)
                head = _rnp.asarray(self._reduce_impl(
                    a[tuple(sl)], axis, dtype, None, False, None, True,
                    "reduceat"))
                if stop - start > 1:
                    sl[axis] = slice(start + 1, stop)
                    tail = self._reduce_impl(
                        a[tuple(sl)], axis, dtype, None, False, None, True,
                        "reduceat")
                    red = self(head, _rnp.asarray(tail))
                else:
                    red = head
            else:
                sl[axis] = slice(start, stop)
                # Not `self.reduce`: that unwraps a 0-d result to the Python
                # object it holds, which for an object array loses the dtype.
                red = _rnp._ufunc_reduce(
                    self.__name__, a[tuple(sl)], axis=axis, dtype=dtype,
                    out=None, keepdims=False, initial=None, where_=True)
            pieces.append(_rnp.asarray(red))
        _errstate.drain("reduceat")
        if a.dtype.kind == "O":
            # `stack` goes through `array(..., dtype=object)`, whose element
            # discovery does not descend into arrays -- each segment would
            # become one *element*.  Assemble the object result directly.
            oshape = list(a.shape)
            oshape[axis] = len(pieces)
            res = _rnp.empty(tuple(oshape), a.dtype)
            for i, p in enumerate(pieces):
                sl = [slice(None)] * a.ndim
                sl[axis] = i
                res[tuple(sl)] = p[()] if p.ndim == 0 else p
        else:
            from ._core.shape_base import stack
            res = stack(pieces, axis=axis)
        if out is not None:
            out[...] = res
            return out
        return res

    def outer(self, a, b, **kwargs):
        if self.signature is not None:
            raise TypeError(
                "method outer is not allowed in ufunc with non-trivial "
                "signature")
        if not self._ok or self.nin != 2:
            self._nope("outer")
        a = _rnp.asarray(a)
        b = _rnp.asarray(b)
        a2 = a.reshape(a.shape + (1,) * b.ndim)
        return self(a2, b, **kwargs)

    def at(self, a, indices, b=None):
        if self.signature is not None:
            raise TypeError(
                f"{self.__name__}.at does not support ufunc with non-trivial "
                f"signature: {self.__name__} has signature {self.signature}.")
        if not self._ok:
            self._nope("at")
        # The *operand* is unbuffered, but the index array and the second
        # operand are read in full before the updates start; when either
        # aliases `a` they have to be materialised first (gh-6272).
        indices = _detached(indices, a)
        b = _detached(b, a)
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


def frompyfunc(func, nin, nout, **kwargs):
    """Create an object-loop ufunc from an arbitrary Python callable."""
    identity = kwargs.pop("identity", _UNSET)
    if kwargs:
        name = next(iter(kwargs))
        raise TypeError(f"'{name}' is an invalid keyword to frompyfunc")
    if not callable(func):
        raise TypeError("function must be callable")
    nin = _operator.index(nin)
    nout = _operator.index(nout)
    if nin + nout > 64:
        raise ValueError(
            "Cannot construct a ufunc with more than 64 operands "
            f"(requested number were: inputs = {nin} and outputs = {nout})")
    identity_set = identity is not _UNSET
    value = None if identity is _UNSET else identity
    return ufunc._from_pyfunc(func, nin, nout, value, identity_set)


frompyfunc.__text_signature__ = "(func, nin, nout, **kwargs)"


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
