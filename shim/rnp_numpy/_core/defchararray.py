"""``numpy._core.defchararray`` for the rnp port.

Upstream ``chararray`` is ``class chararray(ndarray)``.  The port's
``_rnp.ndarray`` cannot be instantiated from Python at all -- neither
``ndarray.__new__`` nor ``object.__new__`` on a Python subclass works
("cannot create ... instances") -- so ``chararray`` here is a *composition*
wrapper around a private ``_rnp.ndarray``.

Two consequences are worked around explicitly:

* ``isinstance(x, np.ndarray)`` has to keep answering ``True`` (numpy's own
  test helpers gate on it).  ``chararray`` therefore reports a ``__class__``
  of ``_chararray_nd``, a real (never instantiated) ndarray subclass.
  ``type(x)`` and ``isinstance(x, chararray)`` are unaffected, because
  CPython's isinstance tries the real type first and only consults
  ``__class__`` when that fails.
* ``ndarray.view(some_type)`` silently ignores the type argument in the
  port, so ``ndarray.view`` is wrapped (below) to recognise ``chararray``.
  Every other call signature is delegated to the original method untouched.
"""

import rnp_numpy as np
from rnp_numpy._core import strings as _s
from rnp_numpy._core.strings import (  # noqa: F401
    add,
    capitalize,
    center,
    count,
    decode,
    encode,
    endswith,
    expandtabs,
    find,
    index,
    isalnum,
    isalpha,
    isdecimal,
    isdigit,
    islower,
    isnumeric,
    isspace,
    istitle,
    isupper,
    ljust,
    lower,
    lstrip,
    replace,
    rfind,
    rindex,
    rjust,
    rstrip,
    slice,
    startswith,
    str_len,
    strip,
    swapcase,
    title,
    translate,
    upper,
    zfill,
)
from rnp_numpy._core.strings import _join as join
from rnp_numpy._core.strings import _rsplit as rsplit
from rnp_numpy._core.strings import _split as split
from rnp_numpy._core.strings import _splitlines as splitlines
from rnp_numpy._core.strings import mod  # noqa: F401
from rnp_numpy._core.strings import multiply as _strings_multiply
from rnp_numpy._core.strings import partition as _strings_partition
from rnp_numpy._core.strings import rpartition as _strings_rpartition

__all__ = [
    'equal', 'not_equal', 'greater_equal', 'less_equal',
    'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize',
    'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs',
    'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
    'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition',
    'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
    'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase',
    'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal',
    'array', 'asarray', 'compare_chararrays', 'chararray'
    ]

_ND = np.ndarray
_WHITESPACE = " \t\n\r\v\f"


# ---------------------------------------------------------------------------
# compare_chararrays
# ---------------------------------------------------------------------------

_CMP_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _rstrip_pad(value):
    if isinstance(value, bytes):
        return value.rstrip(_WHITESPACE.encode("ascii") + b"\0")
    return value.rstrip(_WHITESPACE + "\0")


def compare_chararrays(a1, a2, cmp, rstrip):
    """Element-wise comparison of two string arrays.

    With ``rstrip`` true, trailing whitespace (and NULs) are removed from
    both operands first -- the numarray-compatible behaviour that backs
    ``np.char.equal`` and friends.
    """
    try:
        op = _CMP_OPS[cmp]
    except (KeyError, TypeError):
        raise ValueError(
            f"comparison must be '==', '!=', '<', '>', '<=', '>='") from None

    a = _s.asarray(a1)
    b = _s.asarray(a2)
    if a.dtype.kind not in "SU" or b.dtype.kind not in "SU":
        raise TypeError("comparison of non-string arrays")
    kind = "U" if "U" in (a.dtype.kind, b.dtype.kind) else "S"
    shape = _s._bcast(a, b)
    va = [_s._coerce_like(v, kind) for v in _s._elems(a, shape)]
    vb = [_s._coerce_like(v, kind) for v in _s._elems(b, shape)]
    if rstrip:
        va = [_rstrip_pad(v) for v in va]
        vb = [_rstrip_pad(v) for v in vb]
    return _s._make([op(x, y) for x, y in zip(va, vb)], shape, np.bool_)


def equal(x1, x2):
    return compare_chararrays(x1, x2, '==', True)


def not_equal(x1, x2):
    return compare_chararrays(x1, x2, '!=', True)


def greater_equal(x1, x2):
    return compare_chararrays(x1, x2, '>=', True)


def less_equal(x1, x2):
    return compare_chararrays(x1, x2, '<=', True)


def greater(x1, x2):
    return compare_chararrays(x1, x2, '>', True)


def less(x1, x2):
    return compare_chararrays(x1, x2, '<', True)


def multiply(a, i):
    """``np.strings.multiply`` but raising ``ValueError`` on a non-integer."""
    try:
        return _strings_multiply(a, i)
    except TypeError:
        raise ValueError("Can only multiply by integers") from None


def _stack_last(parts):
    """``np.stack(parts, axis=-1)`` with a common string itemsize."""
    kind = parts[0].dtype.kind
    nchars = max(_s._num_chars(p) for p in parts)
    parts = [_s.astype(p, _s._string_dtype(kind, nchars)) for p in parts]
    return np.stack(parts, axis=-1)


def partition(a, sep):
    return _stack_last(_strings_partition(a, sep))


def rpartition(a, sep):
    return _stack_last(_strings_rpartition(a, sep))


# ---------------------------------------------------------------------------
# chararray
# ---------------------------------------------------------------------------

class _chararray_nd(np.ndarray):
    """Marker type only -- never instantiated.

    It exists so ``isinstance(chararray_instance, np.ndarray)`` is true.
    """


_chararray_nd.__name__ = "chararray"
_chararray_nd.__qualname__ = "chararray"
_chararray_nd.__module__ = "numpy.char"


def _is_char_scalar(value):
    return isinstance(value, (bytes, str)) and not isinstance(value, _ND)


class _cmp_bytes(bytes):
    """``bytes`` that compares the way a `chararray` element does."""

    __hash__ = bytes.__hash__

    def __eq__(self, other):
        if isinstance(other, (bytes, bytearray)):
            return _rstrip_pad(bytes(self)) == _rstrip_pad(bytes(other))
        return NotImplemented

    def __ne__(self, other):
        res = self.__eq__(other)
        return res if res is NotImplemented else not res

    def __lt__(self, other):
        if isinstance(other, (bytes, bytearray)):
            return _rstrip_pad(bytes(self)) < _rstrip_pad(bytes(other))
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, (bytes, bytearray)):
            return _rstrip_pad(bytes(self)) <= _rstrip_pad(bytes(other))
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (bytes, bytearray)):
            return _rstrip_pad(bytes(self)) > _rstrip_pad(bytes(other))
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (bytes, bytearray)):
            return _rstrip_pad(bytes(self)) >= _rstrip_pad(bytes(other))
        return NotImplemented


class _cmp_str(str):
    """``str`` that compares the way a `chararray` element does."""

    __hash__ = str.__hash__

    def __eq__(self, other):
        if isinstance(other, str):
            return _rstrip_pad(str(self)) == _rstrip_pad(str(other))
        return NotImplemented

    def __ne__(self, other):
        res = self.__eq__(other)
        return res if res is NotImplemented else not res

    def __lt__(self, other):
        if isinstance(other, str):
            return _rstrip_pad(str(self)) < _rstrip_pad(str(other))
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, str):
            return _rstrip_pad(str(self)) <= _rstrip_pad(str(other))
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, str):
            return _rstrip_pad(str(self)) > _rstrip_pad(str(other))
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, str):
            return _rstrip_pad(str(self)) >= _rstrip_pad(str(other))
        return NotImplemented


def _cmp_scalar(value):
    if isinstance(value, (bytes, bytearray)):
        return _cmp_bytes(value)
    return _cmp_str(value)


class chararray:
    """A convenient view on arrays of string and bytes values.

    .. deprecated:: 2.5
    """

    _rnp_chararray_data = None

    # -- construction -----------------------------------------------------
    def __new__(cls, shape, itemsize=1, unicode=False, buffer=None,
                offset=0, strides=None, order='C'):
        itemsize = int(itemsize)
        kind = "U" if unicode else "S"
        dtype = _s._string_dtype(kind, itemsize)

        if isinstance(shape, (int, np.integer)):
            shape_t = (int(shape),)
        else:
            shape_t = tuple(int(s) for s in shape)

        if isinstance(buffer, (bytes, bytearray, str)):
            n = _s._size(shape_t)
            chunks = [buffer[i * itemsize:(i + 1) * itemsize]
                      for i in range(n)]
            if unicode and isinstance(buffer, (bytes, bytearray)):
                chunks = [c.decode("ascii") for c in chunks]
            elif not unicode and isinstance(buffer, str):
                chunks = [c.encode("ascii") for c in chunks]
            arr = _s._make_string(chunks, shape_t, kind, itemsize)
        elif buffer is None:
            arr = np.zeros(shape_t, dtype=dtype)
        else:
            arr = np.frombuffer(buffer, dtype=dtype,
                                offset=offset).reshape(shape_t)
        return cls._wrap(arr, None)

    @classmethod
    def _wrap(cls, arr, base):
        self = object.__new__(cls)
        if arr.dtype.kind not in "VSU":
            raise ValueError("Can only create a chararray from string data.")
        object.__setattr__(self, "_rnp_chararray_data", arr)
        object.__setattr__(self, "_rnp_chararray_base", base)
        return self

    # -- identity ---------------------------------------------------------
    @property
    def __class__(self):
        # Makes ``isinstance(self, np.ndarray)`` true; ``type(self)`` and
        # ``isinstance(self, chararray)`` are unaffected.
        return _chararray_nd

    @property
    def base(self):
        return self._rnp_chararray_base

    def __array__(self, dtype=None, copy=None):
        arr = self._rnp_chararray_data
        if dtype is not None:
            return _s.astype(arr, dtype)
        return arr

    # -- ndarray surface --------------------------------------------------
    def __getattr__(self, name):
        if name.startswith("_rnp_chararray"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_rnp_chararray_data"),
                       name)

    def __dir__(self):
        return sorted(set(object.__dir__(self))
                      | set(dir(self._rnp_chararray_data)))

    def __len__(self):
        return len(self._rnp_chararray_data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, obj):
        val = self._rnp_chararray_data[obj]
        if _is_char_scalar(val):
            return val.rstrip()
        if type(val) is _ND:
            return type(self)._wrap(val, self)
        return val

    def __setitem__(self, obj, value):
        self._rnp_chararray_data[obj] = _s._unwrap(value)

    def __repr__(self):
        return repr(self._rnp_chararray_data).replace(
            "array(", "chararray(", 1)

    def __str__(self):
        return str(self._rnp_chararray_data)

    def __bool__(self):
        return bool(self._rnp_chararray_data)

    @property
    def T(self):
        return type(self)._wrap(self._rnp_chararray_data.T, self)

    @staticmethod
    def _as_char(arr):
        """Re-attach the chararray wrapper to a ufunc-style result."""
        return chararray._wrap(arr, None)

    def tobytes(self, order='C'):
        return bytes(memoryview(self._rnp_chararray_data))

    tostring = tobytes

    def tolist(self):
        """Elements as (comparison-aware) Python scalars.

        A ``chararray`` compares whitespace-insensitively -- ``A == B`` runs
        through `compare_chararrays` with ``rstrip=True``.  Upstream that
        stays true however deeply an expression nests, because the operator
        is always the chararray's own.  Here the elements have to carry the
        rule themselves, so the scalars handed out keep their exact bytes
        but compare with trailing padding removed, exactly as the array
        would.
        """
        arr = self._rnp_chararray_data
        vals = [_cmp_scalar(v) for v in _s._elems(arr)]
        return _s._nest(vals, _s._shape(arr))

    def astype(self, dtype, order='K', casting='unsafe', subok=True,
               copy=True):
        return type(self)._wrap(
            _s.astype(self._rnp_chararray_data, dtype), None)

    def copy(self, order='C'):
        return type(self)._wrap(self._rnp_chararray_data.copy(), None)

    def reshape(self, *args, **kwargs):
        return type(self)._wrap(
            self._rnp_chararray_data.reshape(*args, **kwargs), self)

    def ravel(self, *args, **kwargs):
        return type(self)._wrap(
            self._rnp_chararray_data.ravel(*args, **kwargs), self)

    def flatten(self, *args, **kwargs):
        return type(self)._wrap(
            self._rnp_chararray_data.flatten(*args, **kwargs), None)

    def transpose(self, *args, **kwargs):
        return type(self)._wrap(
            self._rnp_chararray_data.transpose(*args, **kwargs), self)

    def squeeze(self, *args, **kwargs):
        return type(self)._wrap(
            self._rnp_chararray_data.squeeze(*args, **kwargs), self)

    def take(self, *args, **kwargs):
        return type(self)._wrap(
            self._rnp_chararray_data.take(*args, **kwargs), None)

    def repeat(self, *args, **kwargs):
        return type(self)._wrap(
            self._rnp_chararray_data.repeat(*args, **kwargs), None)

    def view(self, *args, **kwargs):
        if args and isinstance(args[0], type):
            cls = args[0]
            if issubclass(cls, (chararray, _chararray_nd)):
                return chararray._wrap(self._rnp_chararray_data, self)
            raise TypeError(f"cannot view a chararray as {cls!r}")
        return self._rnp_chararray_data.view(*args, **kwargs)

    def argsort(self, axis=-1, kind=None, order=None, *, stable=None,
                descending=None):
        """Indices that sort the array lexicographically.

        Upstream this delegates to ``ndarray.argsort``; the port has no
        ``argsort`` for any dtype, so the (stable) sort is done here.
        """
        arr = self._rnp_chararray_data
        if axis is None:
            arr = arr.ravel()
            axis = -1
        nd = arr.ndim
        if nd == 0:
            return np.array(0, dtype=np.intp)
        axis = axis % nd
        shape = _s._shape(arr)
        vals = _s._elems(arr)
        # Walk every 1-D lane along ``axis``.
        strides = [1] * nd
        for i in range(nd - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        out = [0] * len(vals)
        n = shape[axis]
        outer = [range(s) for i, s in enumerate(shape) if i != axis]

        def _iter(dims):
            if not dims:
                yield ()
                return
            for head in dims[0]:
                for rest in _iter(dims[1:]):
                    yield (head,) + rest

        other_axes = [i for i in range(nd) if i != axis]
        for combo in _iter(outer):
            base = 0
            for ax, v in zip(other_axes, combo):
                base += v * strides[ax]
            lane = [base + j * strides[axis] for j in range(n)]
            order_ = sorted(range(n), key=lambda j: vals[lane[j]],
                            reverse=bool(descending))
            for j, k in enumerate(order_):
                out[lane[j]] = k
        return _s._make(out, shape, np.intp)

    def sort(self, axis=-1, kind=None, order=None, *, stable=None,
             descending=None):
        idx = self.argsort(axis=axis, stable=stable, descending=descending)
        arr = self._rnp_chararray_data
        vals = _s._elems(arr)
        shape = _s._shape(arr)
        nd = arr.ndim
        axis = axis % nd
        strides = [1] * nd
        for i in range(nd - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        flat_idx = _s._elems(idx)
        new = list(vals)
        n = shape[axis]
        other_axes = [i for i in range(nd) if i != axis]

        def _iter(dims):
            if not dims:
                yield ()
                return
            for head in dims[0]:
                for rest in _iter(dims[1:]):
                    yield (head,) + rest

        for combo in _iter([range(shape[i]) for i in other_axes]):
            base = 0
            for ax, v in zip(other_axes, combo):
                base += v * strides[ax]
            lane = [base + j * strides[axis] for j in range(n)]
            for j in range(n):
                new[lane[j]] = vals[lane[int(flat_idx[lane[j]])]]
        res = _s._make_string(new, shape, arr.dtype.kind, _s._num_chars(arr))
        self._rnp_chararray_data[...] = res

    # -- comparisons ------------------------------------------------------
    def __eq__(self, other):
        return equal(self, other)

    def __ne__(self, other):
        return not_equal(self, other)

    def __ge__(self, other):
        return greater_equal(self, other)

    def __le__(self, other):
        return less_equal(self, other)

    def __gt__(self, other):
        return greater(self, other)

    def __lt__(self, other):
        return less(self, other)

    __hash__ = None

    # -- operators --------------------------------------------------------
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, i):
        return asarray(multiply(self, i))

    def __rmul__(self, i):
        return asarray(multiply(self, i))

    def __mod__(self, i):
        return asarray(mod(self, i))

    def __rmod__(self, other):
        return NotImplemented

    # -- string methods ---------------------------------------------------
    def capitalize(self):
        return asarray(capitalize(self))

    def center(self, width, fillchar=' '):
        return asarray(center(self, width, fillchar))

    def count(self, sub, start=0, end=None):
        return count(self, sub, start, end)

    def decode(self, encoding=None, errors=None):
        return decode(self, encoding, errors)

    def encode(self, encoding=None, errors=None):
        return encode(self, encoding, errors)

    def endswith(self, suffix, start=0, end=None):
        return endswith(self, suffix, start, end)

    def expandtabs(self, tabsize=8):
        return asarray(expandtabs(self, tabsize))

    def find(self, sub, start=0, end=None):
        return find(self, sub, start, end)

    def index(self, sub, start=0, end=None):
        return index(self, sub, start, end)

    def isalnum(self):
        return isalnum(self)

    def isalpha(self):
        return isalpha(self)

    def isdigit(self):
        return isdigit(self)

    def islower(self):
        return islower(self)

    def isspace(self):
        return isspace(self)

    def istitle(self):
        return istitle(self)

    def isupper(self):
        return isupper(self)

    def join(self, seq):
        return join(self, seq)

    def ljust(self, width, fillchar=' '):
        return asarray(ljust(self, width, fillchar))

    def lower(self):
        return asarray(lower(self))

    def lstrip(self, chars=None):
        # Upstream this is a plain ufunc call: the chararray subclass is
        # carried through by ``__array_wrap__``, not by ``asarray``.
        return self._as_char(lstrip(self, chars))

    def partition(self, sep):
        return asarray(partition(self, sep))

    def replace(self, old, new, count=None):
        return self._as_char(
            replace(self, old, new, count if count is not None else -1))

    def rfind(self, sub, start=0, end=None):
        return rfind(self, sub, start, end)

    def rindex(self, sub, start=0, end=None):
        return rindex(self, sub, start, end)

    def rjust(self, width, fillchar=' '):
        return asarray(rjust(self, width, fillchar))

    def rpartition(self, sep):
        return asarray(rpartition(self, sep))

    def rsplit(self, sep=None, maxsplit=None):
        return rsplit(self, sep, maxsplit)

    def rstrip(self, chars=None):
        return self._as_char(rstrip(self, chars))

    def split(self, sep=None, maxsplit=None):
        return split(self, sep, maxsplit)

    def splitlines(self, keepends=None):
        return splitlines(self, keepends)

    def startswith(self, prefix, start=0, end=None):
        return startswith(self, prefix, start, end)

    def strip(self, chars=None):
        return self._as_char(strip(self, chars))

    def swapcase(self):
        return asarray(swapcase(self))

    def title(self):
        return asarray(title(self))

    def translate(self, table, deletechars=None):
        return asarray(translate(self, table, deletechars))

    def upper(self):
        return asarray(upper(self))

    def zfill(self, width):
        return asarray(zfill(self, width))

    def isnumeric(self):
        return isnumeric(self)

    def isdecimal(self):
        return isdecimal(self)


chararray.__module__ = "numpy.char"
chararray.__qualname__ = "chararray"


# ---------------------------------------------------------------------------
# ndarray.view(chararray) support
# ---------------------------------------------------------------------------

def _install_view_hook():
    if getattr(_ND, "_rnp_char_view_patched", False):
        return
    original = _ND.view

    def view(self, *args, **kwargs):
        if args and isinstance(args[0], type) and issubclass(
                args[0], (chararray, _chararray_nd)):
            return chararray._wrap(self, self.base)
        return original(self, *args, **kwargs)

    view.__doc__ = original.__doc__
    _ND.view = view
    _ND._rnp_char_view_patched = True


_install_view_hook()


# ---------------------------------------------------------------------------
# array / asarray
# ---------------------------------------------------------------------------

def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    """Create a `chararray`.

    .. deprecated:: 2.5
    """
    if isinstance(obj, (bytes, str)):
        if unicode is None:
            unicode = isinstance(obj, str)
        if itemsize is None:
            itemsize = len(obj)
        shape = len(obj) // itemsize if itemsize else 0
        return chararray(shape, itemsize=itemsize, unicode=unicode,
                         buffer=obj, order=order)

    if isinstance(obj, (list, tuple)):
        obj = np.asarray(obj)

    inner = _s._unwrap(obj)
    is_char_wrapper = inner is not obj

    if type(inner) is _ND and inner.dtype.kind in "SU":
        if itemsize is None:
            itemsize = _s._num_chars(inner)
        if unicode is None:
            unicode = inner.dtype.kind == "U"
        kind = "U" if unicode else "S"
        if copy or itemsize != _s._num_chars(inner) or kind != inner.dtype.kind:
            return chararray._wrap(
                _s.astype(inner, _s._string_dtype(kind, itemsize)), None)
        if is_char_wrapper:
            return obj
        return chararray._wrap(inner, inner.base)

    kind = "U" if unicode else "S"
    if type(inner) is _ND:
        obj = inner.tolist()
    return chararray._wrap(_from_pyobject(obj, kind, itemsize), None)


def _coerce_leaves(obj, kind):
    """Recursively cast a nested Python structure to `kind`'s scalar type.

    numpy's object -> ``S``/``U`` cast stringifies whatever it finds, so
    ``2`` becomes ``b'2'``; a non-ASCII ``str`` cast to ``S`` raises
    ``UnicodeEncodeError`` (a ``ValueError``), which is what the caller
    relies on.
    """
    if isinstance(obj, (list, tuple)):
        return [_coerce_leaves(x, kind) for x in obj]
    if kind == "S":
        if isinstance(obj, (bytes, bytearray)):
            return bytes(obj)
        if isinstance(obj, str):
            return obj.encode("ascii")
        return str(obj).encode("ascii")
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("ascii")
    return str(obj)


def _from_pyobject(obj, kind, itemsize):
    nested = _coerce_leaves(obj, kind)
    leaves = []

    def walk(x):
        if isinstance(x, list):
            for y in x:
                walk(y)
        else:
            leaves.append(x)

    walk(nested)
    if itemsize is None:
        itemsize = max((len(v) for v in leaves), default=0)
    return np.array(nested, dtype=_s._string_dtype(kind, itemsize))


def asarray(obj, itemsize=None, unicode=None, order=None):
    """Convert `obj` to a `chararray`, copying only when necessary.

    .. deprecated:: 2.5
    """
    return array(obj, itemsize, copy=False, unicode=unicode, order=order)
