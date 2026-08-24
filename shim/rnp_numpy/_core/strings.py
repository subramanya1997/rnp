"""Pure-Python re-implementation of ``numpy.strings`` for the rnp port.

Upstream this module is a thin wrapper over a family of C string ufuncs
(``numpy._core.umath._center``, ``_replace``, ``str_len``, ...).  The Rust
port exposes none of those, so every operation here is done by pulling the
elements out of the ``S``/``U`` array as Python ``bytes``/``str``, applying
the corresponding Python method, and rebuilding an array.

The *values* are the easy part; the part that is deliberately faithful to
upstream is the **result dtype and itemsize**, which numpy computes from
explicit "buffer size" formulas rather than from the realized strings.  Those
formulas are reproduced verbatim in each function below.
"""

import rnp_numpy as np

__all__ = [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "maximum", "minimum",
    "add", "multiply", "isalpha", "isdigit", "isspace", "isalnum", "islower",
    "isupper", "istitle", "isdecimal", "isnumeric", "str_len", "find",
    "rfind", "index", "rindex", "count", "startswith", "endswith", "lstrip",
    "rstrip", "strip", "replace", "expandtabs", "center", "ljust", "rjust",
    "zfill", "partition", "rpartition", "slice",
    "upper", "lower", "swapcase", "capitalize", "title",
    "mod", "decode", "encode", "translate",
]

MAX = 2 ** 63 - 1

_ND = np.ndarray


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def _unwrap(obj):
    """Peel a ``chararray`` wrapper down to the ndarray it owns."""
    inner = getattr(obj, "_rnp_chararray_data", None)
    return obj if inner is None else inner


def asarray(obj, dtype=None):
    """``np.asarray`` that also understands this port's ``chararray``."""
    obj = _unwrap(obj)
    if type(obj) is _ND and dtype is None:
        return obj
    if type(obj) is _ND:
        want = np.dtype(dtype)
        if obj.dtype == want:
            return obj
        return astype(obj, want)
    # numpy 2.x changed `copy=False` from "avoid a copy if you can" to "raise
    # if a copy would be needed"; `asarray` is the spelling that still means
    # the former, and is what numpy's own migration guide points at.
    return np.asarray(obj, dtype=dtype)


def _shape(a):
    return tuple(a.shape)


def _size(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _elems(arr, shape=None):
    """Elements of ``arr`` (broadcast to ``shape``) as Python/numpy scalars.

    String elements are decoded straight out of the array's buffer rather
    than through ``tolist()``/indexing: the port drops embedded NUL
    characters when it materialises a ``U`` element, which loses data that
    numpy keeps (only *trailing* NULs are padding).
    """
    if shape is not None and _shape(arr) != tuple(shape):
        arr = np.broadcast_to(arr, tuple(shape))
    if arr.dtype.kind == "T":
        if arr.ndim == 0:
            return [arr.tolist()]
        return arr.reshape(-1).tolist()
    if arr.dtype.kind in "SU":
        return _string_elems(arr)
    if arr.ndim == 0:
        return [arr[()]]
    if arr.size == 0:
        return []
    try:
        flat = arr.ravel()
    except Exception:
        flat = np.array(arr).ravel()
    return [flat[i] for i in range(flat.size)]


def _string_elems(arr):
    """Exact ``bytes``/``str`` values of a string array, C order."""
    n = arr.dtype.itemsize
    size = arr.size
    if size == 0 or n == 0:
        empty = b"" if arr.dtype.kind == "S" else ""
        return [empty] * size
    flat = arr.reshape(-1) if arr.ndim != 1 else arr
    buf = bytes(memoryview(flat.copy()))
    if arr.dtype.kind == "S":
        return [buf[i * n:(i + 1) * n].rstrip(b"\0") for i in range(size)]
    codec = "utf-32-be" if arr.dtype.str[0] == ">" else "utf-32-le"
    return [buf[i * n:(i + 1) * n].decode(codec).rstrip("\0")
            for i in range(size)]


def _nest(flat, shape):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = _size(shape[1:])
    return [_nest(flat[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def _bcast(*arrs):
    shapes = [_shape(a) for a in arrs]
    if not shapes:
        return ()
    return tuple(np.broadcast_shapes(*shapes))


def _num_chars(a):
    """Number of characters (not bytes) an element of ``a`` can hold."""
    if a.dtype.kind == "U":
        return a.dtype.itemsize // 4
    if a.dtype.kind == "T":
        return max((len(v) for v in _elems(a) if isinstance(v, str)), default=0)
    return a.dtype.itemsize


def _check_string(a, name="a"):
    if a.dtype.kind not in "SUT":
        raise TypeError(
            f"string operation on non-string array")
    return a


class _StringUFunc:
    """Stand-in for the private string ufuncs upstream implements in C.

    Only the parts the suite reaches are modelled: the object exists, carries
    a `__name__`, and reports numpy's "did not contain a loop" error for
    non-string input through either `__call__` or `reduce`.
    """

    nin = 1
    nout = 1

    def __init__(self, name):
        self.__name__ = name

    def __repr__(self):
        return f"<ufunc {self.__name__!r}>"

    def _no_loop(self, args):
        from ._exceptions import _UFuncNoLoopError
        from .. import dtypes as _dtypes
        kinds = []
        for a in args:
            dt = getattr(a, "dtype", None)
            name = getattr(dt, "name", None)
            cls = getattr(_dtypes, _dtypes._CLASS_NAMES.get(name, ""), None)
            kinds.append(cls)
        return _UFuncNoLoopError(self, tuple(kinds) + (None,) * self.nout)

    def __call__(self, *args, **kwargs):
        raise self._no_loop(args)

    def reduce(self, array, *a, **k):
        raise self._no_loop((array,))


_expandtabs_length = _StringUFunc("_expandtabs_length")
_str_len_ufunc = _StringUFunc("str_len")
_slice_ufunc = _StringUFunc("_slice")
_slice_ufunc.nin = 4

_STRING_UFUNCS = {
    "_expandtabs_length": _expandtabs_length,
    "str_len": _str_len_ufunc,
    "_slice": _slice_ufunc,
}


def _no_loop_unless_string(a, ufunc_name):
    """Like `_check_string`, but reporting numpy's ufunc no-loop error.

    Upstream splits these too: `upper` and friends are Python functions that
    say "string operation on non-string array", while `expandtabs`, `str_len`
    and `multiply` are ufunc-backed and surface a `_UFuncNoLoopError`.
    """
    if a.dtype.kind not in "SUT":
        raise _STRING_UFUNCS[ufunc_name]._no_loop((a,))
    return a


def _string_dtype(kind, nchars):
    return np.dtype(f"{kind}{max(int(nchars), 0)}")


def _make_string(flat, shape, kind, nchars):
    """Build an ``S``/``U`` array of exactly ``nchars`` characters."""
    if getattr(kind, "kind", kind) == "T":
        return _make(flat, shape, kind)
    kind = getattr(kind, "kind", kind)
    dt = _string_dtype(kind, nchars)
    shape = tuple(shape)
    if _size(shape) == 0:
        return np.empty(shape, dtype=dt)
    if kind == "S":
        flat = [v.encode("ascii") if isinstance(v, str) else bytes(v)
                for v in flat]
    else:
        flat = [v.decode("ascii") if isinstance(v, (bytes, bytearray))
                else str(v) for v in flat]
    if not shape:
        return np.array(flat[0], dtype=dt)
    return np.array(_nest(flat, shape), dtype=dt)


def _make(flat, shape, dtype):
    shape = tuple(shape)
    dt = np.dtype(dtype)
    if _size(shape) == 0:
        return np.empty(shape, dtype=dt)
    if not shape:
        return np.array(flat[0], dtype=dt)
    return np.array(_nest(flat, shape), dtype=dt)


def _make_object(flat, shape):
    """Build an object array; the values may themselves be lists."""
    shape = tuple(shape)
    if _size(shape) == 0:
        return np.empty(shape, dtype="O")
    # A trailing ``None`` sentinel keeps the outer list ragged so the array
    # constructor cannot absorb the per-element lists as extra dimensions.
    flat = list(flat)
    out = np.array(flat + [None], dtype="O")[:len(flat)]
    if not shape:
        return out.reshape(())
    return out.reshape(shape)


def _obj_flat(arr):
    """Flat list of the Python objects held by an object array."""
    if arr.ndim == 0:
        return [arr.tolist()]
    if arr.size == 0:
        return []
    lst = arr.tolist()
    out = []

    def rec(x, d):
        if d == 0:
            out.append(x)
        else:
            for y in x:
                rec(y, d - 1)

    rec(lst, arr.ndim)
    return out


def astype(a, dtype):
    """``ndarray.astype`` restricted to (and working for) string dtypes."""
    a = _unwrap(a)
    dt = np.dtype(dtype)
    if a.dtype == dt:
        return a.copy()
    if a.dtype.kind in "SU" and dt.kind in "SU":
        nchars = dt.itemsize // 4 if dt.kind == "U" else dt.itemsize
        if nchars == 0:
            # An unsized 'S'/'U' descriptor keeps the source's width.
            nchars = _num_chars(a)
        vals = [v[:nchars] if nchars else type(v)() for v in _elems(a)]
        return _make_string(vals, _shape(a), dt.kind, nchars)
    return a.astype(dt)


def _to_bytes_or_str_array(flat, shape, like):
    """Mirror of upstream ``_to_bytes_or_str_array``.

    ``flat`` holds the raw Python results; ``like`` fixes the output *kind*
    while the itemsize comes from the natural width of the results.
    """
    like = asarray(like)
    kind = like.dtype.kind
    if kind == "T":
        return _make(flat, shape, like.dtype)
    if kind not in "SU":
        kind = "U"
    if _size(tuple(shape)) == 0:
        return np.empty(tuple(shape), dtype=_string_dtype(kind, 0))
    nchars = max(len(v) for v in flat)
    return _make_string(flat, shape, kind, nchars)


# ---------------------------------------------------------------------------
# _vec_string
# ---------------------------------------------------------------------------

def _vec_string(char_array, dtype, method, args=()):
    """``numpy._core.multiarray._vec_string``, in Python."""
    out_dtype = np.dtype(dtype)  # TypeError for a bogus descriptor

    arr = _unwrap(char_array)
    if type(arr) is not _ND:
        arr = np.array(arr)
    if arr.dtype.kind not in "SUT":
        raise TypeError("string operation on non-string array")

    if not isinstance(args, (list, tuple)):
        raise TypeError("'args' must be a sequence of arguments")

    arg_arrays = []
    for a in args:
        arg_arrays.append(asarray(a) if not isinstance(a, (int, type(None)))
                          else np.array(a) if a is not None else None)

    shapes = [_shape(arr)]
    for a in arg_arrays:
        if a is not None:
            shapes.append(_shape(a))
    shape = tuple(np.broadcast_shapes(*shapes))  # ValueError on mismatch

    if out_dtype.kind not in "SUTO":
        raise TypeError(
            "return array must be of type string, unicode or object")

    base = _elems(arr, shape)
    if arr.dtype.kind == "T" and any(not isinstance(v, str) for v in base):
        raise ValueError("Cannot apply string operation to non-string NA")
    cols = []
    for a in arg_arrays:
        cols.append(None if a is None else _elems(a, shape))

    results = []
    for i, elem in enumerate(base):
        f = getattr(elem, method)  # AttributeError for a bogus method
        call_args = []
        for j, a in enumerate(arg_arrays):
            call_args.append(None if a is None else cols[j][i])
        results.append(f(*call_args))

    if out_dtype.kind == "O":
        return _make_object(results, shape)
    nchars = (out_dtype.itemsize // 4 if out_dtype.kind == "U"
              else out_dtype.itemsize)
    if nchars == 0 and results:
        nchars = max(len(r) for r in results)
    return _make_string(results, shape, out_dtype.kind, nchars)


def _clean_args(*args):
    newargs = []
    for chk in args:
        if chk is None:
            break
        newargs.append(chk)
    return newargs


# ---------------------------------------------------------------------------
# generic elementwise drivers
# ---------------------------------------------------------------------------

def _unary(a, fn, kind_of_result):
    a = _check_string(asarray(a))
    vals = []
    for v in _elems(a):
        if a.dtype.kind == "T" and not isinstance(v, str):
            raise ValueError("Cannot apply string operation to non-string NA")
        vals.append(fn(v))
    return _make(vals, _shape(a), kind_of_result)


def _same_shape_string(a, fn):
    """Elementwise op that keeps ``a``'s dtype exactly."""
    a = _check_string(asarray(a))
    vals = []
    for v in _elems(a):
        if a.dtype.kind == "T" and not isinstance(v, str):
            raise ValueError("Cannot apply string operation to non-string NA")
        vals.append(fn(v))
    return _make_string(vals, _shape(a), a.dtype, _num_chars(a))


def _is_nan_na(arr, value):
    """Whether *value* is this StringDType's floating-NaN sentinel."""
    if arr.dtype.kind != "T" or not hasattr(arr.dtype, "na_object"):
        return False
    na = arr.dtype.na_object
    return value is na and isinstance(na, float) and na != na


def _int_arg(x, name):
    arr = asarray(x)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"unsupported type {arr.dtype} for operand {name!r}")
    return arr


def _coerce_like(value, kind):
    if kind == "S":
        if isinstance(value, str):
            return value.encode("ascii")
        return bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("ascii")
    return str(value)


# ---------------------------------------------------------------------------
# comparisons (plain, non-stripping - these are the numpy ufuncs)
# ---------------------------------------------------------------------------

def _cmp(x1, x2, op):
    a = _check_string(asarray(x1))
    b = _check_string(asarray(x2))
    shape = _bcast(a, b)
    kind = "U" if "T" in (a.dtype.kind, b.dtype.kind) or "U" in (
        a.dtype.kind, b.dtype.kind) else "S"
    va = [_coerce_like(v, kind) for v in _elems(a, shape)]
    vb = [_coerce_like(v, kind) for v in _elems(b, shape)]
    return _make([op(x, y) for x, y in zip(va, vb)], shape, np.bool_)


def equal(x1, x2):
    return _cmp(x1, x2, lambda a, b: a == b)


def not_equal(x1, x2):
    return _cmp(x1, x2, lambda a, b: a != b)


def less(x1, x2):
    return _cmp(x1, x2, lambda a, b: a < b)


def less_equal(x1, x2):
    return _cmp(x1, x2, lambda a, b: a <= b)


def greater(x1, x2):
    return _cmp(x1, x2, lambda a, b: a > b)


def greater_equal(x1, x2):
    return _cmp(x1, x2, lambda a, b: a >= b)


def _extreme(x1, x2, take_greater):
    a = _check_string(asarray(x1))
    b = _check_string(asarray(x2))
    shape = _bcast(a, b)
    va = _elems(a, shape)
    vb = _elems(b, shape)
    vals = [x if (x >= y if take_greater else x <= y) else y
            for x, y in zip(va, vb)]
    out_dtype = (np.promote_types(a.dtype, b.dtype)
                 if "T" in (a.dtype.kind, b.dtype.kind)
                 else "U" if "U" in (a.dtype.kind, b.dtype.kind) else "S")
    return _make_string(vals, shape, out_dtype,
                        max(_num_chars(a), _num_chars(b)))


def maximum(x1, x2):
    return _extreme(x1, x2, True)


def minimum(x1, x2):
    return _extreme(x1, x2, False)


# ---------------------------------------------------------------------------
# size / search
# ---------------------------------------------------------------------------

def str_len(a):
    _no_loop_unless_string(asarray(a), "str_len")
    return _unary(a, len, np.intp)


def _search(a, sub, start, end, name):
    arr = _check_string(asarray(a))
    subarr = _check_string(asarray(sub))
    start = _int_arg(0 if start is None else start, "start")
    end = _int_arg(MAX if end is None else end, "end")
    shape = _bcast(arr, subarr, start, end)
    kind = "U" if "T" in (arr.dtype.kind, subarr.dtype.kind) or "U" in (
        arr.dtype.kind, subarr.dtype.kind) else "S"
    raw_a = _elems(arr, shape)
    if arr.dtype.kind == "T" and any(
            not isinstance(v, str) for v in raw_a):
        raise ValueError("Cannot apply string operation to non-string NA")
    va = [_coerce_like(v, kind) for v in raw_a]
    vb = [_coerce_like(v, kind) for v in _elems(subarr, shape)]
    vs = _elems(start, shape)
    ve = _elems(end, shape)
    try:
        vals = [getattr(x, name)(y, int(s), int(e))
                for x, y, s, e in zip(va, vb, vs, ve)]
    except ValueError:
        # `index`/`rindex` raise when the substring is absent.  Python words
        # this differently per type -- `str` says "substring not found" but
        # `bytes` says "subsection not found" -- while numpy always reports
        # the former, whatever the dtype.
        raise ValueError("substring not found") from None
    return _make(vals, shape, np.intp)


def find(a, sub, start=0, end=None):
    return _search(a, sub, start, end, "find")


def rfind(a, sub, start=0, end=None):
    return _search(a, sub, start, end, "rfind")


def index(a, sub, start=0, end=None):
    return _search(a, sub, start, end, "index")


def rindex(a, sub, start=0, end=None):
    return _search(a, sub, start, end, "rindex")


def count(a, sub, start=0, end=None):
    return _search(a, sub, start, end, "count")


def _affix(a, affix, start, end, name):
    arr = _check_string(asarray(a))
    other = _check_string(asarray(affix))
    start = _int_arg(0 if start is None else start, "start")
    end = _int_arg(MAX if end is None else end, "end")
    shape = _bcast(arr, other, start, end)
    kind = "U" if "T" in (arr.dtype.kind, other.dtype.kind) or "U" in (
        arr.dtype.kind, other.dtype.kind) else "S"
    raw_a = _elems(arr, shape)
    vb = [_coerce_like(v, kind) for v in _elems(other, shape)]
    vs = _elems(start, shape)
    ve = _elems(end, shape)
    vals = []
    for raw, y, s, e in zip(raw_a, vb, vs, ve):
        if arr.dtype.kind == "T" and not isinstance(raw, str):
            if _is_nan_na(arr, raw):
                vals.append(False)
                continue
            raise ValueError("Cannot apply string operation to non-string NA")
        vals.append(getattr(_coerce_like(raw, kind), name)(y, int(s), int(e)))
    return _make(vals, shape, np.bool_)


def startswith(a, prefix, start=0, end=None):
    return _affix(a, prefix, start, end, "startswith")


def endswith(a, suffix, start=0, end=None):
    return _affix(a, suffix, start, end, "endswith")


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------

def _predicate(a, name, unicode_only=False):
    arr = _check_string(asarray(a))
    if unicode_only and arr.dtype.kind not in "UT":
        raise TypeError(
            f"'{name}' is not supported for the input types, and the inputs "
            "could not be safely coerced to any supported types")
    vals = []
    for v in _elems(arr):
        if arr.dtype.kind == "T" and not isinstance(v, str):
            if _is_nan_na(arr, v):
                vals.append(False)
                continue
            raise ValueError("Cannot apply string operation to non-string NA")
        vals.append(getattr(v, name)())
    return _make(vals, _shape(arr), np.bool_)


def isalpha(a):
    return _predicate(a, "isalpha")


def isalnum(a):
    return _predicate(a, "isalnum")


def isdigit(a):
    return _predicate(a, "isdigit")


def isspace(a):
    return _predicate(a, "isspace")


def islower(a):
    return _predicate(a, "islower")


def isupper(a):
    return _predicate(a, "isupper")


def istitle(a):
    return _predicate(a, "istitle")


def isnumeric(a):
    return _predicate(a, "isnumeric", unicode_only=True)


def isdecimal(a):
    return _predicate(a, "isdecimal", unicode_only=True)


# ---------------------------------------------------------------------------
# case conversion (itemsize preserved, exactly like the ufuncs)
# ---------------------------------------------------------------------------

def upper(a):
    return _same_shape_string(a, lambda v: v.upper())


def lower(a):
    return _same_shape_string(a, lambda v: v.lower())


def swapcase(a):
    return _same_shape_string(a, lambda v: v.swapcase())


def capitalize(a):
    return _same_shape_string(a, lambda v: v.capitalize())


def title(a):
    return _same_shape_string(a, lambda v: v.title())


# ---------------------------------------------------------------------------
# strip
# ---------------------------------------------------------------------------

# The trailing NUL of a fixed-width string element is padding, so numpy's
# whitespace-stripping loops trim NULs alongside real whitespace.
_WS = " \t\n\r\v\f\0"
_WS_VARIABLE = " \t\n\r\v\f"


def _strip(a, chars, name):
    arr = _check_string(asarray(a))
    if chars is None:
        ws = (_WS.encode("ascii") if arr.dtype.kind == "S" else
              _WS_VARIABLE if arr.dtype.kind == "T" else _WS)
        vals = []
        for v in _elems(arr):
            if arr.dtype.kind == "T" and not isinstance(v, str):
                if _is_nan_na(arr, v):
                    vals.append(v)
                    continue
                raise ValueError(
                    "Cannot apply string operation to non-string NA")
            vals.append(getattr(v, name)(ws))
        return _make_string(vals, _shape(arr), arr.dtype, _num_chars(arr))
    ch = _check_string(asarray(chars))
    shape = _bcast(arr, ch)
    kind = "U" if arr.dtype.kind == "T" else arr.dtype.kind
    raw_a = _elems(arr, shape)
    vb = [_coerce_like(v, kind) for v in _elems(ch, shape)]
    vals = []
    for raw, y in zip(raw_a, vb):
        if arr.dtype.kind == "T" and not isinstance(raw, str):
            if _is_nan_na(arr, raw):
                vals.append(raw)
                continue
            raise ValueError("Cannot apply string operation to non-string NA")
        vals.append(getattr(_coerce_like(raw, kind), name)(y))
    return _make_string(vals, shape, arr.dtype, _num_chars(arr))


def lstrip(a, chars=None):
    return _strip(a, chars, "lstrip")


def rstrip(a, chars=None):
    return _strip(a, chars, "rstrip")


def strip(a, chars=None):
    return _strip(a, chars, "strip")


# ---------------------------------------------------------------------------
# arithmetic-ish
# ---------------------------------------------------------------------------

def add(x1, x2):
    a = _check_string(asarray(x1))
    b = _check_string(asarray(x2))
    kind = "U" if "T" in (a.dtype.kind, b.dtype.kind) or "U" in (
        a.dtype.kind, b.dtype.kind) else "S"
    out_dtype = (np.promote_types(a.dtype, b.dtype)
                 if "T" in (a.dtype.kind, b.dtype.kind) else kind)
    shape = _bcast(a, b)
    raw_a = _elems(a, shape)
    raw_b = _elems(b, shape)
    nchars = _num_chars(a) + _num_chars(b)
    vals = []
    for x, y in zip(raw_a, raw_b):
        bad_x = a.dtype.kind == "T" and not isinstance(x, str)
        bad_y = b.dtype.kind == "T" and not isinstance(y, str)
        if bad_x or bad_y:
            if ((not bad_x or _is_nan_na(a, x))
                    and (not bad_y or _is_nan_na(b, y))):
                vals.append(x if bad_x else y)
                continue
            raise ValueError("Cannot apply string operation to non-string NA")
        vals.append(_coerce_like(x, kind) + _coerce_like(y, kind))
    return _make_string(vals, shape, out_dtype, nchars)


def multiply(a, i):
    arr = _no_loop_unless_string(asarray(a), "str_len")
    try:
        iarr = asarray(i)
    except Exception:
        raise TypeError("unsupported type for operand 'i'") from None
    if iarr.dtype.kind not in "iub":
        raise TypeError(f"unsupported type {iarr.dtype} for operand 'i'")
    shape = _bcast(arr, iarr)
    va = _elems(arr, shape)
    vi = [max(int(v), 0) for v in _elems(iarr, shape)]
    for x in va:
        if arr.dtype.kind == "T" and not isinstance(x, str):
            if not _is_nan_na(arr, x):
                raise ValueError(
                    "Cannot apply string operation to non-string NA")
    lengths = [(len(x) if isinstance(x, (str, bytes)) else 0) * n
               for x, n in zip(va, vi)]
    nchars = max(lengths) if lengths else 0
    # Checked before the strings are built: the product is what overflows,
    # and materialising it first would exhaust memory rather than raise.
    if nchars > MAX:
        raise OverflowError("Overflow encountered in string multiply")
    vals = [x if _is_nan_na(arr, x) else x * n for x, n in zip(va, vi)]
    return _make_string(vals, shape, arr.dtype, nchars)


def mod(a, values):
    arr = asarray(a)
    res = _vec_string(arr, np.dtype("O"), "__mod__", (values,))
    return _to_bytes_or_str_array(_obj_flat(res), _shape(res), arr)


# ---------------------------------------------------------------------------
# padding
# ---------------------------------------------------------------------------

def _just(a, width, fillchar, name):
    warr = _int_arg(width, "width")
    arr = _check_string(asarray(a))
    fill = _check_string(asarray(fillchar))
    if any(len(v) != 1 for v in _elems(fill)):
        raise TypeError("The fill character must be exactly one character long")
    kind = "U" if arr.dtype.kind == "T" else arr.dtype.kind
    shape = _bcast(arr, warr, fill)
    va = _elems(arr, shape)
    vw = [int(v) for v in _elems(warr, shape)]
    vf = [_coerce_like(v, kind) for v in _elems(fill, shape)]
    for s in va:
        if arr.dtype.kind == "T" and not isinstance(s, str):
            if not _is_nan_na(arr, s):
                raise ValueError(
                    "Cannot apply string operation to non-string NA")
    widths = [max(len(s) if isinstance(s, (str, bytes)) else 0, w)
              for s, w in zip(va, vw)]
    nchars = max(widths) if widths else 0
    vals = [s if _is_nan_na(arr, s) else getattr(s, name)(w, f)
            for s, w, f in zip(va, widths, vf)]
    return _make_string(vals, shape, arr.dtype, nchars)


def center(a, width, fillchar=' '):
    return _just(a, width, fillchar, "center")


def ljust(a, width, fillchar=' '):
    return _just(a, width, fillchar, "ljust")


def rjust(a, width, fillchar=' '):
    return _just(a, width, fillchar, "rjust")


def zfill(a, width):
    warr = _int_arg(width, "width")
    arr = _check_string(asarray(a))
    shape = _bcast(arr, warr)
    va = _elems(arr, shape)
    vw = [int(v) for v in _elems(warr, shape)]
    for s in va:
        if arr.dtype.kind == "T" and not isinstance(s, str):
            if not _is_nan_na(arr, s):
                raise ValueError(
                    "Cannot apply string operation to non-string NA")
    widths = [max(len(s) if isinstance(s, (str, bytes)) else 0, w)
              for s, w in zip(va, vw)]
    nchars = max(widths) if widths else 0
    return _make_string([s if _is_nan_na(arr, s) else s.zfill(w)
                         for s, w in zip(va, widths)], shape,
                        arr.dtype, nchars)


def expandtabs(a, tabsize=8):
    arr = _no_loop_unless_string(asarray(a), "_expandtabs_length")
    tarr = _int_arg(tabsize, "tabsize")
    shape = _bcast(arr, tarr)
    va = _elems(arr, shape)
    if arr.dtype.kind == "T" and any(not isinstance(v, str) for v in va):
        raise ValueError("Cannot apply string operation to non-string NA")
    vt = [int(v) for v in _elems(tarr, shape)]
    try:
        vals = [s.expandtabs(t) for s, t in zip(va, vt)]
    except OverflowError:
        # Python reports the C-level conversion failure; numpy words the
        # same condition as a statement about the result.
        raise OverflowError("new string is too long") from None
    nchars = max((len(v) for v in vals), default=0)
    return _make_string(vals, shape, arr.dtype, nchars)


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------

def replace(a, old, new, count=-1):
    carr = _int_arg(count, "count")
    arr = _check_string(asarray(a))
    oldarr = _check_string(asarray(old))
    newarr = _check_string(asarray(new))
    kind = "U" if arr.dtype.kind == "T" else arr.dtype.kind
    shape = _bcast(arr, oldarr, newarr, carr)
    raw_a = _elems(arr, shape)
    for value in raw_a:
        if arr.dtype.kind == "T" and not isinstance(value, str):
            if not _is_nan_na(arr, value):
                raise ValueError(
                    "Cannot apply string operation to non-string NA")
    va = [_coerce_like(v, kind) if isinstance(v, (str, bytes)) else v
          for v in raw_a]
    vo = [_coerce_like(v, kind) for v in _elems(oldarr, shape)]
    vn = [_coerce_like(v, kind) for v in _elems(newarr, shape)]
    vc = [int(v) for v in _elems(carr, shape)]

    counts = []
    for s, o, c in zip(va, vo, vc):
        n = 0 if _is_nan_na(arr, s) else s.count(o)
        counts.append(n if c < 0 else min(n, c))
    buffersizes = [(len(s) if isinstance(s, (str, bytes)) else 0)
                   + n * (len(x) - len(o))
                   for s, n, o, x in zip(va, counts, vo, vn)]
    nchars = max(buffersizes) if buffersizes else 0
    vals = [s if _is_nan_na(arr, s) else s.replace(o, x, n)
            for s, o, x, n in zip(va, vo, vn, counts)]
    return _make_string(vals, shape, arr.dtype, nchars)


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------

def _partition(a, sep, right):
    arr = _check_string(asarray(a))
    sarr = _check_string(asarray(sep))
    kind = "U" if arr.dtype.kind == "T" else arr.dtype.kind
    shape = _bcast(arr, sarr)
    va = [_coerce_like(v, kind) for v in _elems(arr, shape)]
    vs = [_coerce_like(v, kind) for v in _elems(sarr, shape)]
    parts = [s.rpartition(p) if right else s.partition(p)
             for s, p in zip(va, vs)]
    found = [bool(p[1]) for p in parts]
    n1 = max((len(p[0]) for p in parts), default=0)
    n3 = max((len(p[2]) for p in parts), default=0)
    n2 = 1 if not any(found) else max((len(p) for p in vs), default=1)
    return (_make_string([p[0] for p in parts], shape, arr.dtype, n1),
            _make_string([p[1] for p in parts], shape, arr.dtype, n2),
            _make_string([p[2] for p in parts], shape, arr.dtype, n3))


def partition(a, sep):
    return _partition(a, sep, right=False)


def rpartition(a, sep):
    return _partition(a, sep, right=True)


# ---------------------------------------------------------------------------
# codecs / translate
# ---------------------------------------------------------------------------

def decode(a, encoding=None, errors=None):
    res = _vec_string(a, np.dtype("O"), "decode",
                      _clean_args(encoding, errors))
    return _to_bytes_or_str_array(_obj_flat(res), _shape(res), np.str_(""))


def encode(a, encoding=None, errors=None):
    res = _vec_string(a, np.dtype("O"), "encode",
                      _clean_args(encoding, errors))
    return _to_bytes_or_str_array(_obj_flat(res), _shape(res), np.bytes_(b""))


def translate(a, table, deletechars=None):
    arr = asarray(a)
    if arr.dtype.kind == "U":
        return _vec_string(arr, arr.dtype, "translate", (table,))
    return _vec_string(arr, arr.dtype, "translate",
                       [table] + _clean_args(deletechars))


# ---------------------------------------------------------------------------
# slice
# ---------------------------------------------------------------------------

_NO_VALUE = object()


def slice(a, start=None, stop=_NO_VALUE, step=None, /):
    # Upstream `_slice` is a ufunc whose index operands have an integer
    # signature, so a string passed as start/stop/step is an input *casting*
    # failure rather than a missing loop.  The reported operand number counts
    # the arguments as the caller wrote them, so it is computed before the
    # one-argument `stop`/`start` swap below.
    _supplied = [v for v in (start, stop, step) if v is not _NO_VALUE
                 and v is not None]
    if stop is _NO_VALUE:
        stop, start = start, None
    arr = _no_loop_unless_string(asarray(a), "_slice")
    if step is None:
        step = 1
    for _i, _v in enumerate(_supplied, start=2):
        _dt = getattr(asarray(_v), "dtype", None)
        if _dt is not None and _dt.kind in "SU":
            from ._exceptions import _UFuncInputCastingError
            raise _UFuncInputCastingError(
                _slice_ufunc, "same_kind", _dt, np.dtype(np.intp), _i)
    steparr = _int_arg(step, "step")
    if any(int(v) == 0 for v in _elems(steparr)):
        raise ValueError("slice step cannot be zero")
    pieces = [arr, steparr]
    startarr = None if start is None else _int_arg(start, "start")
    stoparr = None if stop is None else _int_arg(stop, "stop")
    for p in (startarr, stoparr):
        if p is not None:
            pieces.append(p)
    shape = _bcast(*pieces)
    va = _elems(arr, shape)
    vstep = [int(v) for v in _elems(steparr, shape)]
    vstart = ([None] * len(va) if startarr is None
              else [int(v) for v in _elems(startarr, shape)])
    vstop = ([None] * len(va) if stoparr is None
             else [int(v) for v in _elems(stoparr, shape)])
    import builtins
    vals = [s[builtins.slice(b, e, t)]
            for s, b, e, t in zip(va, vstart, vstop, vstep)]
    return _make_string(vals, shape, arr.dtype, _num_chars(arr))


# ---------------------------------------------------------------------------
# split family (object output)
# ---------------------------------------------------------------------------

def _split(a, sep=None, maxsplit=None):
    return _vec_string(a, np.dtype("O"), "split",
                       [sep] + _clean_args(maxsplit))


def _rsplit(a, sep=None, maxsplit=None):
    return _vec_string(a, np.dtype("O"), "rsplit",
                       [sep] + _clean_args(maxsplit))


def _splitlines(a, keepends=None):
    return _vec_string(a, np.dtype("O"), "splitlines", _clean_args(keepends))


def _join(sep, seq):
    res = _vec_string(sep, np.dtype("O"), "join", (seq,))
    return _to_bytes_or_str_array(_obj_flat(res), _shape(res), seq)


# ---------------------------------------------------------------------------
# override protocols
#
# Upstream these functions are split by *implementation*: some are true
# ufuncs, the rest are Python functions carrying `array_function_dispatch`.
# That split is observable, because an argument defining `__array_ufunc__`
# must be honoured by the first group and one defining `__array_function__`
# by the second.  The port implements both groups in Python, so the
# distinction is re-imposed here, following upstream's grouping exactly.
# ---------------------------------------------------------------------------

#: Implemented upstream as ufuncs -> honour `__array_ufunc__`.
_UFUNC_BACKED = (
    "add", "lstrip", "rstrip", "strip", "equal", "not_equal", "greater_equal",
    "less_equal", "greater", "less", "count", "endswith", "find", "index",
    "isalnum", "isalpha", "isdecimal", "isdigit", "islower", "isnumeric",
    "isspace", "istitle", "isupper", "rfind", "rindex", "startswith",
    "str_len", "replace",
)

#: Implemented upstream as dispatched Python functions.
_FUNCTION_BACKED = (
    "center", "capitalize", "decode", "encode", "expandtabs", "ljust",
    "lower", "mod", "multiply", "partition", "rjust", "rpartition",
    "swapcase", "title", "translate", "upper", "zfill",
)


def _overrides(args, kwargs, attr):
    """Arguments that implement `attr` themselves rather than being arrays."""
    seen = []
    for value in (*args, *kwargs.values()):
        if isinstance(value, _ND) or type(value) in seen:
            continue
        if hasattr(type(value), attr):
            seen.append(type(value))
    return seen


def _dispatched(fn, attr):
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        types = _overrides(args, kwargs, attr)
        if types:
            handler = getattr(type(args[0]), attr, None) if args else None
            if handler is not None:
                if attr == "__array_ufunc__":
                    return handler(args[0], wrapper, "__call__",
                                   *args, **kwargs)
                return handler(args[0], wrapper, types, args, kwargs)
        return fn(*args, **kwargs)

    return wrapper


for _name in _UFUNC_BACKED:
    globals()[_name] = _dispatched(globals()[_name], "__array_ufunc__")
for _name in _FUNCTION_BACKED:
    globals()[_name] = _dispatched(globals()[_name], "__array_function__")
del _name
