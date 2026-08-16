"""NumPy's scalar type hierarchy.

Every fact encoded here was probed from real numpy 2.5.2 in `.venv`:

  * the abstract tower is
    ``generic -> number -> {integer -> {signed,unsigned}, inexact ->
    {floating, complexfloating}}`` plus ``flexible -> character``;
  * only *four* concrete types inherit a Python builtin --
    ``float64(floating, float)``, ``complex128(complexfloating, complex)``,
    ``bytes_(bytes, character)`` and ``str_(str, character)``.  ``int64`` does
    **not** inherit ``int`` in numpy 2.x;
  * ``np.bool_.__name__`` is ``"bool"``, and ``np.True_`` / ``np.False_`` are
    singletons;
  * ``longlong``/``ulonglong``/``longdouble``/``clongdouble`` are distinct
    classes even where their storage matches another type.

The numeric behaviour is *not* reimplemented here: each operator hands the
operands to the Rust engine (`_rnp._scalar_binop` / `_scalar_unop`), which runs
the same inner loop an array of that dtype would, applies NEP 50 weak-scalar
promotion, and reports the floating-point error flags that this module turns
into numpy's RuntimeWarnings.
"""

import builtins as _builtins
import inspect as _inspect
import math as _math
import numbers as _numbers
import types as _types
import warnings as _warnings
from decimal import Decimal as _Decimal

import _rnp

from . import _errstate

dtype = _rnp.dtype
ndarray = _rnp.ndarray

_SCALAR_BY_NAME = {}


def _void_key(v):
    """A comparable rendering of a structured void.

    `item()` hands subarray fields back as *arrays*, exactly as numpy does, so
    it cannot be compared with `==` directly -- the array comparison would be
    elementwise. Equality is defined on the values, so the arrays are
    flattened to lists first.
    """
    return tuple(x.tolist() if hasattr(x, "tolist") and hasattr(x, "shape")
                 else x for x in v.item())


def _raw_bytes(arr):
    """The raw item bytes of a (0-d) array; the port has no `ndarray.tobytes`."""
    try:
        return bytes(memoryview(arr))
    except (TypeError, BufferError, ValueError):
        return bytes(memoryview(arr.copy()))


# ---------------------------------------------------------------------------
# repr helpers
# ---------------------------------------------------------------------------

#: The decimal exponent at (or above) which numpy switches a float scalar's
#: str/repr from positional to scientific. Probed from numpy 2.5.2: the
#: positional window is `-4 <= exp10 <= {2, 5, 15}` for half/single/double.
_SCI_AT = {"float16": 3, "float32": 6, "float64": 16}


def _shortest(v, name):
    """`(digits, exp10)` of the shortest decimal that round-trips.

    ``repr(np.float32(0.1))`` is ``np.float32(0.1)``, not the float64 spelling
    of the same bits, so the search runs against the *narrow* type's bytes.
    """
    from ._printing import _unique_digits
    return _unique_digits(_builtins.abs(v), name)


def _sci_at(name):
    """Where a scalar's str/repr switches to scientific notation.

    numpy 2.3 gave float16/float32 their own (much lower) cutoffs; under
    ``printoptions(legacy='2.2')`` and older they keep float64's, which is why
    `str(np.half(65504))` is `65500.0` in legacy mode but `6.55e+04` now.
    """
    from ._printing import _legacy_level
    if name != "float64" and _legacy_level() <= 202:
        return _SCI_AT["float64"]
    return _SCI_AT[name]


def _float_str(v, name):
    """numpy's scalar float formatting.

    Two rules, both probed: the digits are the shortest decimal that
    round-trips through *this* float type, and the positional/scientific
    choice is made on the decimal exponent of the stored value (which is why
    `str(np.float32(1e-4))` is `1e-04` -- the stored value is 9.9999997e-05,
    whose exponent is -5).
    """
    if v != v:
        return "nan"
    if v == _math.inf:
        return "inf"
    if v == -_math.inf:
        return "-inf"
    if v == 0.0:
        return "-0.0" if _math.copysign(1.0, v) < 0 else "0.0"
    from ._printing import _split
    digits, exp = _shortest(v, name)
    sign = "-" if _math.copysign(1.0, v) < 0 else ""
    # The positional/scientific choice is made on the *stored* value's
    # exponent, but the digits printed are the shortest form's -- which is why
    # `str(np.float32(1e-4))` is `1e-04` even though 9.9999997e-05 is stored.
    adjusted = _Decimal(v).adjusted()
    if adjusted < -4 or adjusted >= _sci_at(name):
        mant = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
        return "%s%se%s%02d" % (sign, mant, "+" if exp >= 0 else "-",
                                _builtins.abs(exp))
    intstr, fracstr = _split(digits, exp - len(digits) + 1)
    return "%s%s.%s" % (sign, intstr, fracstr or "0")


def _strip_parens(s):
    return s[1:-1] if s.startswith("(") else s


def _complex_str(v, name):
    """numpy formats a complex scalar the way Python does, except that each
    component is rendered with *its own* float type's rules -- which is why
    `str(np.complex64(2147483647))` is `(2.1474836e+09+0j)`."""
    comp = "float32" if name == "complex64" else "float64"

    def part(x):
        # Python's complex repr drops a trailing `.0`: `str(1+2j) == '(1+2j)'`.
        t = _float_str(x, comp)
        return t[:-2] if t.endswith(".0") else t

    re, im = part(v.real), part(v.imag)
    if not im.startswith("-"):
        im = "+" + im
    if v.real == 0.0 and _math.copysign(1.0, v.real) > 0:
        return f"{im.lstrip('+')}j"
    return f"({re}{im}j)"


# ---------------------------------------------------------------------------
# The abstract tower
# ---------------------------------------------------------------------------


class _ScalarMeta(type):
    def __repr__(cls):
        return f"<class 'numpy.{cls.__name__}'>"


class generic(metaclass=_ScalarMeta):
    """Base of every numpy scalar type."""

    __slots__ = ()
    #: Overridden on the concrete subclasses.
    dtype = None

    def __new__(cls, *args, **kwargs):
        if cls is generic or cls.dtype is None:
            raise TypeError(
                f"cannot create 'numpy.{cls.__name__}' instances")
        return super().__new__(cls)

    def __radd__(self, other):
        # numpy fills `generic`'s nb_add slot, so every scalar type -- even the
        # character ones, which have no arithmetic -- exposes `__radd__`. It
        # yields NotImplemented (gh-9620) so that `b'def' + np.bytes_('abc')`
        # falls back to `bytes.__add__` instead of raising.
        return NotImplemented

    # -- array-like surface ------------------------------------------------

    @property
    def shape(self):
        return ()

    @property
    def strides(self):
        return ()

    @property
    def ndim(self):
        return 0

    @property
    def size(self):
        return 1

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.dtype.itemsize

    @property
    def base(self):
        return None

    @property
    def flags(self):
        return _rnp.array(self._v, dtype=self.dtype).flags

    @property
    def T(self):
        return self

    @property
    def flat(self):
        return self.__array__().flat

    def __array__(self, dtype=None, copy=None):
        a = _rnp.array(self._v, dtype=self.dtype)
        return a if dtype is None else a.astype(dtype)

    def __len__(self):
        raise TypeError(f"len() of unsized object")

    def __getitem__(self, key):
        # `x[()]` returns the scalar itself; anything else is an error, just
        # as it is on a 0-d array.
        if key == () or key is Ellipsis:
            return self
        if key == (Ellipsis,):
            return self.__array__()
        raise IndexError(
            "invalid index to scalar variable.")

    def __iter__(self):
        raise TypeError("iteration over a 0-d array")

    def item(self, *args):
        if args and args != (0,) and args != ((),):
            raise ValueError("can only convert an array of size 1 to a "
                             "Python scalar")
        return self._v

    def tolist(self):
        return self._v

    def astype(self, dt, *a, **k):
        d = dtype(dt)
        cls = _SCALAR_BY_NAME.get(_type_key(d))
        if cls is None:
            return self.__array__().astype(d)
        return cls(self._v)

    def copy(self, *a, **k):
        return self

    def reshape(self, *shape, **k):
        return self.__array__().reshape(*shape)

    def ravel(self, *a, **k):
        return self.__array__().ravel()

    def flatten(self, *a, **k):
        return self.__array__().flatten()

    def squeeze(self, *a, **k):
        return self

    def transpose(self, *a):
        return self

    def view(self, dt=None, type_=None):
        # numpy's scalar `.view` gives back a *scalar*, not a 0-d array.
        if dt is None:
            return self
        v = self.__array__().view(dt)
        return v[()] if getattr(v, "ndim", 1) == 0 else v

    def fill(self, value):
        raise ValueError("cannot write to a numpy scalar")

    def tobytes(self, order="C"):
        return self.__array__().tobytes() if hasattr(
            self.__array__(), "tobytes") else bytes(memoryview(self.__array__()))

    def __array_wrap__(self, arr, context=None, return_scalar=True):
        # Probed: with the third argument omitted, None or True a 0-d result
        # "decays" back to a scalar; only an explicit False (or a result that
        # is not 0-d) keeps the array.
        if return_scalar is False or getattr(arr, "ndim", 1) != 0:
            return arr
        return arr[()]

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (type(self), (self._v,))

    def __format__(self, spec):
        if spec == "":
            return str(self)
        return format(self._v, spec)

    # -- array-API surface -------------------------------------------------

    @property
    def device(self):
        return "cpu"

    def to_device(self, device, /, *, stream=None):
        if device != "cpu":
            raise ValueError(f"Unsupported device: {device!r}.")
        if stream is not None:
            raise ValueError("The stream argument in to_device() is not "
                             "supported")
        return self

    def __array_namespace__(self, /, *, api_version=None):
        import rnp_numpy
        if api_version is not None and api_version not in (
                "2021.12", "2022.12", "2023.12", "2024.12"):
            raise ValueError(f"Version {api_version!r} of the Array API "
                             "Standard is not supported.")
        return rnp_numpy

    # -- PEP 585 subscription ----------------------------------------------

    def __class_getitem__(cls, item):
        """numpy's abstract numeric scalar types are subscriptable.

        Probed against numpy 2.5.2: the ``number`` tower takes exactly one
        argument, ``complexfloating`` takes one or two, ``bool``/``datetime64``
        are unrestricted, and everything else (``generic``, ``flexible``,
        ``character`` and every other concrete type) raises ``TypeError``.
        """
        arity = _GENERIC_ALIAS_ARITY.get(cls)
        if arity is None:
            raise TypeError(
                f"type 'numpy.{cls.__name__}' is not subscriptable")
        if arity is not True:
            n = len(item) if isinstance(item, tuple) else 1
            if n < arity[0]:
                raise TypeError(f"Too few arguments for numpy.{cls.__name__}")
            if n > arity[1]:
                raise TypeError(f"Too many arguments for numpy.{cls.__name__}")
        return _types.GenericAlias(cls, item)


class number(generic):
    __slots__ = ()


class integer(number):
    __slots__ = ()

    def is_integer(self, /):
        return True


class signedinteger(integer):
    __slots__ = ()


class unsignedinteger(integer):
    __slots__ = ()


class inexact(number):
    __slots__ = ()


class floating(inexact):
    __slots__ = ()

    def is_integer(self, /):
        return _builtins.float(self._v).is_integer()

    def as_integer_ratio(self, /):
        return _builtins.float(self._v).as_integer_ratio()


class complexfloating(inexact):
    __slots__ = ()


class flexible(generic):
    __slots__ = ()


class character(flexible):
    __slots__ = ()


#: Which scalar classes accept ``cls[...]``, and with how many arguments.
#: ``True`` means "any number" (that is what ``np.bool`` and ``np.datetime64``
#: do); a ``(min, max)`` pair is the ``number`` tower's restriction.
_GENERIC_ALIAS_ARITY = {
    number: (1, 1),
    integer: (1, 1),
    signedinteger: (1, 1),
    unsignedinteger: (1, 1),
    inexact: (1, 1),
    floating: (1, 1),
    complexfloating: (1, 2),
}


# ---------------------------------------------------------------------------
# ndarray-compatible methods on ``generic``
# ---------------------------------------------------------------------------

#: Every method numpy's scalars share with ``ndarray``.  Each one has to
#: report *the same* signature as the ``ndarray`` method of the same name, so
#: rather than duplicating the parameter lists they are generated from
#: ``ndarray``'s own introspected signature.
_ARRAY_METHOD_NAMES = (
    "__array_namespace__", "__copy__", "__deepcopy__", "all", "any", "argmax",
    "argmin", "argsort", "astype", "byteswap", "choose", "clip", "compress",
    "conj", "conjugate", "copy", "cumprod", "cumsum", "diagonal", "dump",
    "dumps", "fill", "flatten", "getfield", "item", "max", "mean", "min",
    "nonzero", "prod", "put", "ravel", "repeat", "reshape", "resize", "round",
    "searchsorted", "setfield", "setflags", "sort", "squeeze", "std", "sum",
    "swapaxes", "take", "to_device", "tobytes", "tofile", "tolist", "trace",
    "transpose", "var", "view",
)


def _array_forwarder(name):
    """Fallback body: run the method on the equivalent 0-d array."""
    def f(self, *a, **k):
        return getattr(self.__array__(), name)(*a, **k)
    f.__name__ = name
    return f


def _mirror_ndarray_signature(name, impl):
    """Re-declare `impl` with the parameter list `ndarray.<name>` reports.

    ``inspect.signature(np.generic.take) == inspect.signature(np.ndarray.take)``
    is part of numpy's documented scalar/array parity, so the source of truth
    is the array method itself; if the port's ``ndarray`` has no such method
    (or no introspectable signature) the scalar method is left alone.
    """
    target = getattr(ndarray, name, None)
    if target is None:
        return None
    try:
        sig = _inspect.signature(target)
    except (TypeError, ValueError):
        return None

    P = _inspect.Parameter
    params = list(sig.parameters.values())
    ns = {"_impl": impl}
    decl, call = [], []
    pos_only, star_seen = [], False
    for i, p in enumerate(params):
        if p.default is not p.empty:
            ns[f"_d{i}"] = p.default
            spelling = f"{p.name}=_d{i}"
        else:
            spelling = p.name
        if p.kind is P.POSITIONAL_ONLY:
            pos_only.append(spelling)
            if i:  # `self` is bound by the descriptor, never forwarded
                call.append(p.name)
            continue
        if pos_only:
            decl.extend(pos_only + ["/"])
            pos_only = []
        if p.kind is P.POSITIONAL_OR_KEYWORD:
            decl.append(spelling)
            call.append(p.name)
        elif p.kind is P.VAR_POSITIONAL:
            decl.append("*" + p.name)
            call.append("*" + p.name)
            star_seen = True
        elif p.kind is P.KEYWORD_ONLY:
            if not star_seen:
                decl.append("*")
                star_seen = True
            decl.append(spelling)
            call.append(f"{p.name}={p.name}")
        else:  # VAR_KEYWORD
            decl.append("**" + p.name)
            call.append("**" + p.name)
    if pos_only:
        decl.extend(pos_only + ["/"])

    src = (f"def {name}({', '.join(decl)}):\n"
           f"    return _impl({', '.join(['self'] + call)})\n")
    exec(compile(src, "<rnp scalar methods>", "exec"), ns)
    fn = ns[name]
    fn.__qualname__ = f"generic.{name}"
    fn.__module__ = "numpy"
    return fn


def _install_array_methods():
    for name in _ARRAY_METHOD_NAMES:
        impl = getattr(generic, name, None)
        fn = _mirror_ndarray_signature(
            name, impl if impl is not None else _array_forwarder(name))
        if fn is not None:
            setattr(generic, name, fn)
        # If the port's ndarray has no such method there is nothing to mirror
        # and nothing to delegate to, so `generic` keeps whatever it had.


_install_array_methods()


# numpy registers its abstract numeric tower with the `numbers` ABCs, so
# `issubclass(np.floating, numbers.Real)` and friends hold (test_abc.py).
# Only the abstract bases are registered: registration is inherited by the
# concrete types, and nothing is registered with `numbers.Rational`, which is
# what makes `issubclass(np.float64, numbers.Rational)` correctly False.
_numbers.Number.register(number)
_numbers.Complex.register(inexact)
_numbers.Complex.register(complexfloating)
_numbers.Real.register(floating)
_numbers.Integral.register(integer)
_numbers.Integral.register(signedinteger)
_numbers.Integral.register(unsignedinteger)


# ---------------------------------------------------------------------------
# Operator plumbing
# ---------------------------------------------------------------------------

#: numpy's wording for the four floating-point error conditions.
_FLAG_NAMES = ((1, "divide"), (2, "over"), (4, "under"), (8, "invalid"))
_FLAG_TEXT = {
    "divide": "divide by zero encountered",
    "over": "overflow encountered",
    "under": "underflow encountered",
    "invalid": "invalid value encountered",
}


def _report(flags, opname):
    # Frames above this one: _report, _wrap_result, _binary/_unary, the
    # operator method, then the caller.
    if flags:
        _errstate.report(flags, f"scalar {opname}", stacklevel=6)


def _wrap_result(res, opname):
    d, value, flags = res
    _report(flags, opname)
    cls = _SCALAR_BY_NAME.get(_type_key(d))
    if cls is None:  # pragma: no cover - every dtype we produce is registered
        return value
    return cls._wrap(value)


def _type_key(d):
    k = d.kind
    if k == "S":
        return "bytes_"
    if k == "U":
        return "str_"
    if k == "V":
        return "void"
    if k == "O":
        return "object_"
    return d.name


def _binary(opname, a, b):
    if isinstance(a, ndarray) or isinstance(b, ndarray):
        return NotImplemented
    for x in (a, b):
        if not isinstance(x, (generic, int, float, complex, _builtins.bool)):
            return NotImplemented
    res = _rnp._scalar_binop(opname, a, b)
    if opname == "divmod":
        return tuple(_wrap_result(r, "divmod") for r in res)
    return _wrap_result(res, opname)


def _unary(opname, a):
    return _wrap_result(_rnp._scalar_unop(opname, a), opname)


#: `dtype code -> the `_wrap` of the scalar class for it`, in the order
#: `_rnp._scalar_dtype_names()` reports. The Rust fast path hands back that
#: code instead of building a `dtype` object, so the whole result-typing step
#: on this side is one list index.
_WRAPS = [None] * 14

_sb2 = _rnp._scalar_binop2
_report_fpe = _errstate.report


def _install_operators(cls):
    """Give one concrete scalar class its full operator surface.

    Each method is one call into the Rust scalar path (no 0-d arrays, no
    `dtype` object) plus one list index to find the wrapper for the result
    dtype. `_binary` below is the fallback for the few ops the fast path has
    no opcode for (`divmod`).
    """

    def make(op, name):
        code = _rnp._scalar_binop_code(name)
        if code is None:  # pragma: no cover - every operator name has one
            def fwd(self, other):
                return _binary(name, self, other)

            def rev(self, other):
                return _binary(name, other, self)
        else:
            where = "scalar " + name

            # The defaults are keyword-only on purpose: `pow(a, b, m)` calls
            # `__pow__` with *three* positional arguments, and a plain default
            # would silently bind the modulus to the opcode. Keyword-only
            # keeps numpy's TypeError while still compiling to a LOAD_FAST.
            def fwd(self, other, *, _c=code, _w=where):
                r = _sb2(_c, self, other)
                if r is None:
                    return NotImplemented
                d, v, flags = r
                if flags:
                    # Frames: report, this method, the caller.
                    _report_fpe(flags, _w, stacklevel=3)
                return _WRAPS[d](v)

            def rev(self, other, *, _c=code, _w=where):
                r = _sb2(_c, other, self)
                if r is None:
                    return NotImplemented
                d, v, flags = r
                if flags:
                    _report_fpe(flags, _w, stacklevel=3)
                return _WRAPS[d](v)

        fwd.__name__ = f"__{op}__"
        rev.__name__ = f"__r{op}__"
        return fwd, rev

    for op, name in (
        ("add", "add"), ("sub", "subtract"), ("mul", "multiply"),
        ("truediv", "divide"), ("floordiv", "floor_divide"),
        ("mod", "remainder"), ("pow", "power"), ("divmod", "divmod"),
        ("and", "bitwise_and"), ("or", "bitwise_or"), ("xor", "bitwise_xor"),
        ("lshift", "left_shift"), ("rshift", "right_shift"),
    ):
        fwd, rev = make(op, name)
        setattr(cls, f"__{op}__", fwd)
        setattr(cls, f"__r{op}__", rev)

    def make_cmp(op, name):
        f = make(op, name)[0]
        f.__name__ = f"__{op}__"
        return f

    for op, name in (("lt", "less"), ("le", "less_equal"),
                     ("gt", "greater"), ("ge", "greater_equal"),
                     ("eq", "equal"), ("ne", "not_equal")):
        setattr(cls, f"__{op}__", make_cmp(op, name))
    cls.__hash__ = lambda self: hash(self._v)

    cls.__neg__ = lambda self: _unary("negative", self)
    cls.__pos__ = lambda self: _unary("positive", self)
    cls.__abs__ = lambda self: _unary("absolute", self)
    cls.__invert__ = lambda self: _unary("invert", self)
    return cls


# ---------------------------------------------------------------------------
# Concrete types
# ---------------------------------------------------------------------------


def _make_numeric(name, spec, base, builtin=None, clsname=None):
    d = dtype(spec)
    ns = {
        "__slots__": ("_v",),
        "dtype": d,
        "__module__": "numpy",
    }
    bases = (base,) if builtin is None else (base, builtin)
    # The type whose *exact* instances need no conversion at all: `float` for
    # the 8-byte floats, `complex` for the 16-byte complexes, `int` for the
    # integers (after the same range check `_check_range` would make). Values
    # of that type take a straight-line constructor; everything else -- narrow
    # floats, strings, arrays, out-of-range ints -- goes through `_cast`,
    # which is where all of numpy's coercion rules live.
    #
    # This matters because `np.float64(1.5)` is ~1.15us of a 3.9us
    # `np.float64(1.5) + np.float64(2.5)`, and almost all of it is the generic
    # coercion machinery deciding it has nothing to do.
    k, isize = d.kind, d.itemsize
    if k == "f" and isize == 8:
        _fast_type, _fast_range = _builtins.float, None
    elif k == "c" and isize == 16:
        _fast_type, _fast_range = complex, None
    elif k in "iu":
        _fast_type, _fast_range = int, _INT_RANGE[d.name]
    else:
        _fast_type, _fast_range = None, None

    if builtin is not None:
        # `float64(floating, float)` -- the builtin has to carry the payload,
        # so `__new__` routes through it.
        def __new__(cls, value=0, *extra, _raw=None, **kw):
            # numpy's scalar constructors take the value plus the same
            # trailing arguments the corresponding builtin accepts.
            if _raw is None:
                if type(value) is _fast_type and not extra and not kw:
                    self = builtin.__new__(cls, value)
                    self._v = value
                    return self
                arr = _as_array_arg(cls, value, extra, kw)
                if arr is not None:
                    return arr
            v = _raw if _raw is not None else _cast(cls, value, *extra, **kw)
            self = builtin.__new__(cls, v)
            self._v = v
            return self
    else:
        def __new__(cls, value=0, *extra, _raw=None, **kw):
            if _raw is None:
                if type(value) is _fast_type and not extra and not kw \
                        and (_fast_range is None
                             or _fast_range[0] <= value <= _fast_range[1]):
                    self = object.__new__(cls)
                    self._v = value
                    return self
                arr = _as_array_arg(cls, value, extra, kw)
                if arr is not None:
                    return arr
            v = _raw if _raw is not None else _cast(cls, value, *extra, **kw)
            self = object.__new__(cls)
            self._v = v
            return self

    ns["__new__"] = __new__
    cls = _ScalarMeta(clsname or name, bases, ns)

    # `_wrap` is on the hot path of every scalar operator, so it constructs
    # the object directly rather than going back through `__new__`'s signature
    # machinery: the value it is handed is always already of the right type
    # (it came out of the engine as this dtype).
    if builtin is None:
        def _wrap(value, _cls=cls, _new=object.__new__):
            self = _new(_cls)
            self._v = value
            return self
    else:
        def _wrap(value, _cls=cls, _new=builtin.__new__):
            self = _new(_cls, value)
            self._v = value
            return self

    cls._wrap = staticmethod(_wrap)
    _install_operators(cls)
    _install_numeric_extras(cls, d)
    _SCALAR_BY_NAME.setdefault(d.name, cls)
    # Lets the Rust fast path recognise an operand by its type pointer instead
    # of two `getattr`s and a downcast.
    _rnp._register_scalar_class(cls, d)
    return cls


def _as_array_arg(cls, value, extra, kw):
    """`np.float64([1, 2])` is an *array* constructor, not a scalar one.

    Probed against numpy 2.5.2: a numeric scalar type called with anything
    that is at least 1-d -- list, tuple or ndarray -- returns
    ``array(value, dtype=cls.dtype)``, even when it holds a single element
    (``np.float64([1])`` is ``array([1.])``).  A 0-d array still yields a
    scalar.  Returns `None` when the scalar path applies.
    """
    if extra or kw or cls.dtype.kind not in "biufc":
        return None
    if isinstance(value, (list, tuple)):
        from . import array as _array
        return _array(value, cls.dtype)
    if isinstance(value, ndarray) and value.ndim > 0:
        return value.astype(cls.dtype)
    return None


def _cast(cls, value, *extra, **kw):
    """numpy's scalar constructor coercion."""
    if extra or kw:
        raise TypeError(
            f"{cls.__name__}() takes at most 1 argument "
            f"({1 + len(extra) + len(kw)} given)")
    if isinstance(value, (str, bytes)) and cls.dtype.kind in "biufc":
        text = value.decode() if isinstance(value, bytes) else value
        k = cls.dtype.kind
        if k in "iu":
            value = int(text, 0)
        elif k == "f":
            value = _builtins.float(text)
        elif k == "c":
            value = complex(text)
        else:
            value = _builtins.bool(int(text))
    elif isinstance(value, ndarray):
        if value.size != 1:
            raise ValueError("only length-1 arrays can be converted to "
                             "Python scalars")
        value = value.item()
    value = _coerce_huge_int(cls, value)
    _check_range(cls.dtype, value)
    return _rnp._scalar_cast(cls.dtype, value)


def _coerce_huge_int(cls, value):
    """Handle a Python int too large for the target float type.

    numpy splits here, and the two halves really do differ (probed on 2.5.2):
    `np.float64(2**1024)` raises ``OverflowError: int too large to convert to
    float``, while `np.longdouble(2**1024)` saturates to `inf` and emits a
    ``RuntimeWarning``.  That is not an accident of precision -- the long
    double path converts through a widening routine that saturates instead of
    raising -- so it stays correct here even though this port aliases
    `longdouble` to `float64`.

    Anything that is not an oversized int is returned untouched.
    """
    if cls.dtype.kind not in "fc" or not isinstance(value, int):
        return value
    if isinstance(value, _builtins.bool):
        return value
    try:
        return _builtins.float(value)
    except OverflowError:
        if cls.__name__ not in ("longdouble", "clongdouble"):
            raise
        _warnings.warn(
            "overflow encountered in conversion from python long",
            RuntimeWarning, stacklevel=2,
        )
        return _builtins.float("-inf" if value < 0 else "inf")


#: (min, max) for each integer dtype, used by the constructor checks.
_INT_RANGE = {
    "int8": (-128, 127), "int16": (-32768, 32767),
    "int32": (-2**31, 2**31 - 1), "int64": (-2**63, 2**63 - 1),
    "uint8": (0, 255), "uint16": (0, 65535),
    "uint32": (0, 2**32 - 1), "uint64": (0, 2**64 - 1),
}


def _check_range(d, value):
    """numpy's scalar constructors refuse values they cannot represent.

    Probed: `np.int8(128)` is an OverflowError, `np.int8(float('nan'))` a
    ValueError, and `np.int8(1 + 2j)` a TypeError -- the port used to wrap
    silently in all three cases.
    """
    kind = d.kind
    # `np.bool_(1+2j)` is fine (it is a truth test); the numeric types are the
    # ones that refuse a complex argument.
    if isinstance(value, complex) and not isinstance(value, generic) \
            and kind in "iuf":
        raise TypeError(
            f"{'int' if kind in 'iub' else 'float'}() argument must be a "
            f"string, a bytes-like object or a real number, not 'complex'")
    if kind not in "iu":
        return
    if isinstance(value, _builtins.float):
        if value != value:
            raise ValueError("cannot convert float NaN to integer")
        if value in (_math.inf, -_math.inf):
            raise OverflowError("cannot convert float infinity to integer")
        value = _math.trunc(value)
    elif isinstance(value, generic) and value.dtype.kind == "f":
        v = _builtins.float(value)
        if v != v:
            raise ValueError("cannot convert float NaN to integer")
        if v in (_math.inf, -_math.inf):
            raise OverflowError("cannot convert float infinity to integer")
        value = _math.trunc(v)
    elif isinstance(value, generic):
        value = value.item()
    if not isinstance(value, int):
        return
    lo, hi = _INT_RANGE[d.name]
    if not lo <= value <= hi:
        raise OverflowError(
            f"Python integer {value} out of bounds for {d.name}")


def _install_numeric_extras(cls, d):
    kind = d.kind

    cls.__bool__ = lambda self: _builtins.bool(self._v)
    cls.__int__ = lambda self: _builtins.int(self._v.real if kind == "c"
                                             else self._v)
    cls.__float__ = lambda self: _builtins.float(
        self._v.real if kind == "c" else self._v)
    cls.__complex__ = lambda self: complex(self._v)

    if kind in "biu":
        cls.__index__ = lambda self: _builtins.int(self._v)
        cls.bit_count = lambda self: _builtins.int(self._v).bit_count()
        cls.__lshift__ = cls.__lshift__
    if kind in "iu":
        # numpy declares these with a positional-only `self` (they are C
        # methods upstream), which `inspect.signature` reports and
        # test_scalar_methods asserts.
        def is_integer(self, /):
            return True

        def as_integer_ratio(self, /):
            return (_builtins.int(self._v), 1)

        cls.is_integer = is_integer
        cls.as_integer_ratio = as_integer_ratio
        cls.numerator = property(lambda self: type(self)(self._v))
        cls.denominator = property(lambda self: type(self)(1))

    if kind == "f":
        def is_integer(self, /):
            return _builtins.float(self._v).is_integer()

        def as_integer_ratio(self, /):
            return _builtins.float(self._v).as_integer_ratio()

        cls.is_integer = is_integer
        cls.as_integer_ratio = as_integer_ratio
        cls.hex = lambda self: _builtins.float(self._v).hex()
        cls.fromhex = classmethod(
            lambda c, s: c(_builtins.float.fromhex(s)))

    if kind == "c":
        cls.real = property(
            lambda self: _real_type(d)(self._v.real))
        cls.imag = property(
            lambda self: _real_type(d)(self._v.imag))
        cls.conjugate = lambda self: type(self)(self._v.conjugate())
    else:
        cls.real = property(lambda self: self)
        cls.imag = property(lambda self: type(self)(0))
        cls.conjugate = lambda self: self
    cls.conj = cls.conjugate

    # repr / str
    name = d.name
    if kind == "b":
        cls.__repr__ = lambda self: "np.True_" if self._v else "np.False_"
        cls.__str__ = lambda self: "True" if self._v else "False"
    elif kind == "f":
        cls.__repr__ = lambda self, n=name, c=cls: \
            f"np.{c.__name__}({_float_str(self._v, n)})"
        cls.__str__ = lambda self, n=name: _float_str(self._v, n)
    elif kind == "c":
        # `str` parenthesises a complex with a real part but not a pure
        # imaginary one, so the parens are stripped only when present.
        cls.__repr__ = lambda self, n=name: \
            f"np.{n}({_strip_parens(_complex_str(self._v, n))})"
        cls.__str__ = lambda self, n=name: _complex_str(self._v, n)
    else:
        # numpy reprs a scalar by its *dtype* name, so `np.longlong(0)` shows
        # as `np.int64(0)`. A *subclass* is not a builtin scalar type, so
        # `genint_type_repr` falls back to the type's own name (gh-27106).
        cls.__repr__ = lambda self, n=name, c=cls: (
            f"np.{n}({self._v})" if type(self) is c
            else f"{type(self).__name__}({self._v})")
        cls.__str__ = lambda self: str(self._v)

    _install_legacy_repr(cls)

    # reductions, for parity with 0-d arrays
    for m in ("sum", "prod", "min", "max", "mean", "all", "any", "argmin",
              "argmax", "cumsum", "cumprod", "ptp", "std", "var", "round"):
        if not hasattr(cls, m):
            setattr(cls, m, _forward(m))


def _legacy():
    from ._printing import _legacy_level
    return _legacy_level()


def _install_legacy_repr(cls):
    """NEP 51 gave scalars their `np.<type>(value)` repr in numpy 2.0.

    ``printoptions(legacy='1.25')`` and older restore the pre-NEP-51 repr,
    which for every numeric, bool and text scalar is simply its `str`.
    """
    nep51_repr = cls.__repr__

    def __repr__(self):
        from ._printing import _legacy_level
        if _legacy_level() <= 125:
            return self.__str__()
        return nep51_repr(self)

    cls.__repr__ = __repr__


def _forward(method):
    def f(self, *a, **k):
        return getattr(self.__array__(), method)(*a, **k)
    f.__name__ = method
    return f


def _real_type(d):
    return complex64_real if d.name == "complex64" else float64


# ---- the concrete numeric types -------------------------------------------

bool_ = _make_numeric("bool_", "?", generic, clsname="bool")
int8 = _make_numeric("int8", "int8", signedinteger)
int16 = _make_numeric("int16", "int16", signedinteger)
int32 = _make_numeric("int32", "int32", signedinteger)
int64 = _make_numeric("int64", "int64", signedinteger)
longlong = _make_numeric("longlong", "q", signedinteger)
uint8 = _make_numeric("uint8", "uint8", unsignedinteger)
uint16 = _make_numeric("uint16", "uint16", unsignedinteger)
uint32 = _make_numeric("uint32", "uint32", unsignedinteger)
uint64 = _make_numeric("uint64", "uint64", unsignedinteger)
ulonglong = _make_numeric("ulonglong", "Q", unsignedinteger)
float16 = _make_numeric("float16", "float16", floating)
float32 = _make_numeric("float32", "float32", floating)
float64 = _make_numeric("float64", "float64", floating, _builtins.float)
complex64 = _make_numeric("complex64", "complex64", complexfloating)
complex128 = _make_numeric("complex128", "complex128", complexfloating,
                           complex)
# On macOS/arm64 numpy's long double *is* an IEEE double, but it keeps its own
# type number (13/16) and its own scalar class.
longdouble = _make_numeric("longdouble", "g", floating)
clongdouble = _make_numeric("clongdouble", "G", complexfloating)

#: `np.complex64(1+2j).real` is a float32.
complex64_real = float32

_SCALAR_BY_NAME.update({
    "bool": bool_, "int8": int8, "int16": int16, "int32": int32,
    "int64": int64, "uint8": uint8, "uint16": uint16, "uint32": uint32,
    "uint64": uint64, "float16": float16, "float32": float32,
    "float64": float64, "complex64": complex64, "complex128": complex128,
})

True_ = bool_(True)
False_ = bool_(False)


def _bool_singleton(value, _t=True_, _f=False_):
    return _t if value else _f


bool_._wrap = staticmethod(_bool_singleton)

# The scalar fast path returns a dtype *code*; this is the table it indexes.
# Built after `bool_._wrap` is replaced by the singleton lookup, so that
# `np.float64(1) < np.float64(2)` still comes back as `np.True_` itself.
_rnp._register_generic_class(generic)
for _i, _nm in enumerate(_rnp._scalar_dtype_names()):
    _WRAPS[_i] = _SCALAR_BY_NAME[_nm]._wrap
del _i, _nm
# The engine uses the same table whenever it has to hand back a scalar --
# `a[i]`, `a.max()`, `a.item()` -- instead of looking the class up by name.
_rnp._register_scalar_wraps(_WRAPS)


# ---- the flexible types ----------------------------------------------------


class _FlexMeta(_ScalarMeta):
    """Gives the flexible scalar *classes* a `dtype` (`np.dtype(np.bytes_)`
    is `dtype('S')`) while instances keep their own sized one."""

    def __new__(mcls, name, bases, ns, char=None):
        cls = super().__new__(mcls, name, bases, ns)
        cls._char = char
        return cls

    @property
    def dtype(cls):
        return dtype(cls._char)


class bytes_(_builtins.bytes, character, metaclass=_FlexMeta, char="S"):
    __module__ = "numpy"
    __slots__ = ()

    def __new__(cls, value=b"", encoding=None, errors=None):
        if isinstance(value, str):
            b = value.encode(encoding or "utf-8", errors or "strict")
        elif isinstance(value, (bytes, bytearray, memoryview)):
            b = bytes(value)
        elif isinstance(value, int):
            b = b"\x00" * value
        else:
            b = str(value).encode()
        return _builtins.bytes.__new__(cls, b)

    @property
    def _v(self):
        return _builtins.bytes(self)

    @property
    def dtype(self):
        return dtype(f"S{len(self) or 1}")

    def __repr__(self):
        # Pre-NEP-51 (`legacy='1.25'`) the text scalars simply inherited the
        # builtin's repr.
        if _legacy() <= 125:
            return _builtins.bytes.__repr__(self)
        return f"np.bytes_({_builtins.bytes.__repr__(self)})"

    def __str__(self):
        return _builtins.bytes.__str__(self)

    def __array__(self, dtype_=None, copy=None):
        return _rnp.array(_builtins.bytes(self))


class str_(str, character, metaclass=_FlexMeta, char="U"):
    __module__ = "numpy"
    __slots__ = ()

    def __new__(cls, value="", *a, **k):
        if isinstance(value, (bytes, bytearray)):
            value = value.decode(*a, **k) if a or k else value.decode()
        return str.__new__(cls, value)

    @property
    def _v(self):
        return str(self)

    @property
    def dtype(self):
        return dtype(f"U{len(self) or 1}")

    def __repr__(self):
        if _legacy() <= 125:
            return str.__repr__(self)
        return f"np.str_({str.__repr__(self)})"

    def __str__(self):
        return str.__str__(self)

    def __array__(self, dtype_=None, copy=None):
        return _rnp.array(str(self))


class void(flexible, metaclass=_FlexMeta, char="V"):
    """`np.void`: either an opaque run of bytes (`V<n>`) or a **structured
    scalar** — one element of a structured array.

    A structured void is backed by a genuine 0-d *view* of the element it came
    from (`_arr`), which is how numpy's own behaviour falls out for free:
    `a[1]['f0'] = 7` writes through to `a`, `v.base` is the parent array, and
    `setfield` mutates shared memory.
    """

    __module__ = "numpy"
    __slots__ = ("_b", "_arr")

    def __new__(cls, value=b"", dtype=None):
        self = object.__new__(cls)
        self._arr = None
        if dtype is not None:
            d = _rnp.dtype(dtype)
            if d.names is not None:
                # `np.void((1, 2.0), dtype='i4,f8')` builds a real structured
                # scalar, as numpy 2.x does.
                a = _rnp.array([value], dtype=d)
                self._arr = a[0]._arr
                self._b = None
                return self
        if isinstance(value, int):
            self._b = b"\x00" * value
        else:
            self._b = bytes(value)
        return self

    @classmethod
    def _from_array(cls, arr):
        """Wrap a 0-d array *view* of one element. Called from Rust."""
        self = object.__new__(cls)
        self._arr = arr
        self._b = None
        return self

    # -- the byte payload --------------------------------------------------

    @property
    def _v(self):
        if self._arr is None:
            return self._b
        if self._arr.dtype.names is None:
            return _raw_bytes(self._arr)
        return self.item()

    @property
    def dtype(self):
        if self._arr is not None:
            return self._arr.dtype
        return _rnp.dtype(f"V{len(self._b)}")

    @property
    def base(self):
        return None if self._arr is None else self._arr.base

    @property
    def flags(self):
        if self._arr is not None:
            return self._arr.flags
        return _rnp.array(self._b, dtype=self.dtype).flags

    def __array__(self, dtype=None, copy=None):
        a = self._arr if self._arr is not None else _rnp.array(
            self._b, dtype=self.dtype)
        return a if dtype is None else a.astype(dtype)

    def __bytes__(self):
        return self.tobytes()

    def tobytes(self, order="C"):
        if self._arr is None:
            return self._b
        return _raw_bytes(self._arr)

    def __len__(self):
        # Probed: `len(np.void(b'abc'))` is 0 -- an unstructured void has no
        # fields, and numpy reports the *field* count.
        names = self.dtype.names
        return 0 if names is None else len(names)

    # -- field access ------------------------------------------------------

    def _field_name(self, indx):
        names = self.dtype.names
        if names is None:
            raise IndexError(
                "too many indices for array: array is 0-dimensional, "
                "but 1 were indexed")
        n = len(names)
        i = indx + n if indx < 0 else indx
        if i < 0 or i >= n:
            # Probed: numpy reports the *normalised* index, so `v[-99]` on a
            # two-field void says `invalid index (-97)`.
            raise IndexError(f"invalid index ({i})")
        return names[i]

    def __getitem__(self, indx):
        if indx == () or indx is Ellipsis:
            return self
        if self._arr is None:
            if isinstance(indx, str):
                raise TypeError("void data-type with no fields")
            raise IndexError(
                "too many indices for array: array is 0-dimensional, "
                "but 1 were indexed")
        if isinstance(indx, _numbers.Integral) and not isinstance(indx, _builtins.bool):
            indx = self._field_name(int(indx))
        if isinstance(indx, str) or isinstance(indx, list):
            v = self._arr[indx]
            # A *subarray* field of a 0-d void is a real array, not a scalar:
            # `tuple(np.zeros(1,[('x','i4',(2,2))])[0])` yields an ndarray.
            return v[()] if getattr(v, "ndim", 0) == 0 else v
        raise IndexError(
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
            "(`None`) and integer or boolean arrays are valid indices")

    def __setitem__(self, indx, value):
        if self._arr is None:
            raise TypeError("void data-type with no fields")
        if isinstance(indx, _numbers.Integral) and not isinstance(indx, _builtins.bool):
            indx = self._field_name(int(indx))
        self._arr[indx] = value

    def __iter__(self):
        names = self.dtype.names
        if names is None:
            raise TypeError("iteration over a 0-d array")
        for name in names:
            yield self[name]

    def __contains__(self, item):
        names = self.dtype.names
        return bool(names) and item in names

    def getfield(self, dt, offset=0):
        return self.__array__().getfield(dt, offset)[()]

    def setfield(self, val, dt, offset=0):
        if self._arr is None:
            raise TypeError("Cannot set fields on an unstructured void")
        self._arr.setfield(val, dt, offset)

    def view(self, dt=None, type_=None):
        # numpy's `(scalar_class, dtype)` spelling: `v.view((np.record, dt))`
        # is how `numpy._core.records` re-labels a structured void as a
        # `record`. Reuse the same backing view so writeback still works.
        if isinstance(dt, tuple) and len(dt) == 2 and isinstance(dt[0], type):
            cls, want = dt
            if issubclass(cls, void) and self._arr is not None:
                arr = self._arr
                if _rnp.dtype(want) != arr.dtype:
                    arr = arr.astype(_rnp.dtype(want))
                return cls._from_array(arr)
        if isinstance(dt, type) and issubclass(dt, void):
            return dt._from_array(self._arr) if self._arr is not None \
                else dt(self._b)
        if dt is None:
            return self
        v = self.__array__().view(dt)
        return v[()] if getattr(v, "ndim", 1) == 0 else v

    # -- value surface -----------------------------------------------------

    def item(self, *args):
        if args and args not in ((0,), ((),)):
            raise ValueError("can only convert an array of size 1 to a "
                             "Python scalar")
        if self._arr is None:
            return self._b
        names = self.dtype.names
        if names is None:
            return _raw_bytes(self._arr)
        out = []
        for name in names:
            f = self._arr[name]
            # Probed: a *subarray* field comes back from `item()` as a real
            # ndarray, not as nested lists -- unlike every scalar field.
            out.append(f[()].item() if f.ndim == 0 else f)
        return tuple(out)

    def tolist(self):
        return self.item()

    def __eq__(self, other):
        # numpy hands back `np.True_`/`np.False_`, not Python bools.
        if isinstance(other, void):
            if (self.dtype.names is None) != (other.dtype.names is None):
                return NotImplemented
            return bool_(_void_key(self) == _void_key(other))
        if isinstance(other, (bytes, bytearray)) and self.dtype.names is None:
            return bool_(self.tobytes() == bytes(other))
        if self.dtype.names is not None:
            raise TypeError(
                "Cannot compare structured or void to non-void arrays.")
        return NotImplemented

    def __ne__(self, other):
        r = self.__eq__(other)
        return r if r is NotImplemented else bool_(not r)

    def __hash__(self):
        if self.dtype.names is None:
            return hash(self.tobytes())
        return hash(_void_key(self))

    def __reduce__(self):
        if self.dtype.names is None:
            return (type(self), (self.tobytes(),))
        return (type(self), (self.item(), self.dtype))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        if self.dtype.names is None:
            inner = "".join("\\x%02x" % b for b in self.tobytes())
            return f"np.void(b'{inner}')"
        return self._struct_repr()

    def _struct_repr(self):
        # numpy's own `_void_scalar_to_string(x, is_repr=True)` wraps the value
        # in `np.dtype((np.void, x.dtype))`, which for a structured dtype is
        # just that dtype again. The wrapper is rebuilt here rather than taken
        # from arrayprint because the port's `dtype((void, struct))` spelling
        # is not yet accepted by the dtype constructor.
        cls = type(self)
        fqn = cls.__module__.replace("numpy", "np") + "." + cls.__name__
        return f"{fqn}({self._struct_str(False)}, dtype={self.dtype!s})"

    def _struct_str(self, is_repr):
        # Delegates to the faithful port of numpy's own arrayprint, which is
        # exactly what numpy's C `scalartypes.c.src` does for a structured
        # void: per-field elementwise formatters, `float_kind` forced to str.
        from ._core.arrayprint import _void_scalar_to_string
        return _void_scalar_to_string(self, is_repr=is_repr)

    def __str__(self):
        if self.dtype.names is None:
            inner = "".join("\\x%02x" % b for b in self.tobytes())
            return f"b'{inner}'"
        return self._struct_str(False)


class object_(generic, metaclass=_ScalarMeta):
    """`np.object_` is a descriptor-level type only: calling it returns the
    argument unchanged, which is what numpy does."""

    __module__ = "numpy"
    __slots__ = ()
    dtype = dtype("O")

    def __new__(cls, value=None, *a, **k):
        return value


class datetime64(generic, metaclass=_FlexMeta, char="M8"):
    """Descriptor-level only: `np.dtype('M8[ns]')` constructs, but the port
    has no datetime storage, so instances carry their argument unchanged."""

    __module__ = "numpy"
    __slots__ = ("_v", "_unit")

    def __new__(cls, value=None, unit=None):
        self = object.__new__(cls)
        self._v = value
        self._unit = unit
        return self

    @property
    def dtype(self):
        return dtype("M8" if self._unit is None else f"M8[{self._unit}]")

    def __repr__(self):
        return f"numpy.datetime64({self._v!r})"

    def __repr__(self):
        return f"numpy.datetime64({self._v!r})"


class timedelta64(signedinteger, metaclass=_FlexMeta, char="m8"):
    __module__ = "numpy"
    __slots__ = ("_v", "_unit")

    def __new__(cls, value=None, unit=None):
        self = object.__new__(cls)
        self._v = value
        self._unit = unit
        return self

    @property
    def dtype(self):
        return dtype("m8" if self._unit is None else f"m8[{self._unit}]")

    def __repr__(self):
        return f"numpy.timedelta64({self._v!r})"


_SCALAR_BY_NAME.update({
    "bytes_": bytes_, "str_": str_, "void": void, "object_": object_,
})

# `np.bool` and `np.datetime64` are the two *concrete* types numpy makes
# subscriptable, and unlike the `number` tower they accept any number of
# arguments (probed: `np.bool[Any, Any, Any]` is a valid GenericAlias).
_GENERIC_ALIAS_ARITY[bool_] = True
_GENERIC_ALIAS_ARITY[datetime64] = True


# ---------------------------------------------------------------------------
# Aliases, sctypeDict and ScalarType (all probed from numpy 2.5.2)
# ---------------------------------------------------------------------------

byte = int8
short = int16
intc = int32
int_ = int64
long = int64
intp = int64
ubyte = uint8
ushort = uint16
uintc = uint32
uint = uint64
ulong = uint64
uintp = uint64
half = float16
single = float32
double = float64
csingle = complex64
cdouble = complex128
unicode_ = str_

CONCRETE = (
    bool_, int8, int16, int32, int64, longlong, uint8, uint16, uint32,
    uint64, ulonglong, float16, float32, float64, longdouble, complex64,
    complex128, clongdouble, bytes_, str_, void, object_, datetime64,
    timedelta64,
)

sctypeDict = {}
for _t in CONCRETE:
    sctypeDict[_t.__name__] = _t
sctypeDict.update({
    "bool": bool_, "bool_": bool_,
    "byte": int8, "short": int16, "intc": int32, "int_": int64,
    "long": int64, "intp": int64, "int": int64,
    "ubyte": uint8, "ushort": uint16, "uintc": uint32, "uint": uint64,
    "ulong": uint64, "uintp": uint64,
    "half": float16, "single": float32, "double": float64, "float": float64,
    "csingle": complex64, "cdouble": complex128, "complex": complex128,
    "str": str_, "str_": str_, "unicode": str_,
    "bytes": bytes_, "bytes_": bytes_,
    "object": object_, "object_": object_,
    "longdouble": longdouble, "clongdouble": clongdouble,
})
# numpy's `sctypeDict` is keyed by *name* only -- the single-character codes
# are not registered (probed: `'?' not in np.sctypeDict`).

ScalarType = (
    int, _builtins.float, complex, _builtins.bool, _builtins.bytes,
    str, memoryview,
    bool_, complex64, complex128, clongdouble, float16, float32, float64,
    longdouble, int8, int16, int32, int64, longlong, datetime64,
    timedelta64, object_, bytes_, str_, uint8, uint16, uint32, uint64,
    ulonglong, void,
)

#: numpy's typecode groups, verbatim from `np.typecodes`.
typecodes = {
    "Character": "c",
    "Integer": "bhilqnp",
    "UnsignedInteger": "BHILQNP",
    "Float": "efdg",
    "Complex": "FDG",
    "AllInteger": "bBhHiIlLqQnNpP",
    "AllFloat": "efdgFDG",
    "Datetime": "Mm",
    "All": "?bhilqnpBHILQNPefdgFDGSUVOMm",
}

#: Verbatim from numpy 2.5.2's `numpy._core.sctypes` -- including the C-type
#: aliases (`longlong`, `ulonglong`, `clongdouble`) and the scalar (not
#: builtin) types under "others".
sctypes = {
    "int": [int8, int16, int32, int64, longlong],
    "uint": [uint8, uint16, uint32, uint64, ulonglong],
    "float": [float16, float32, float64, longdouble],
    "complex": [complex64, clongdouble, complex128],
    "others": [bytes_, str_, void, bool_, object_],
}


#: The map handed to the Rust side so `dtype.type` and element extraction can
#: build the right scalar class.
def registry():
    d = dict(_SCALAR_BY_NAME)
    d["object_"] = object_
    # The engine looks scalar classes up by `dtype.name`, and the object
    # dtype's name is "object" (no trailing underscore) while the scalar class
    # is `object_`.  Without this alias `np.dtype('O').type` comes back None,
    # so `issubclass(arr.dtype.type, np.object_)` raises instead of answering.
    d["object"] = object_
    # The C-type aliases are distinct classes; `np.dtype('q').type` must find
    # `np.longlong` rather than `np.int64`.
    d.update({"longlong": longlong, "ulonglong": ulonglong,
              "longdouble": longdouble, "clongdouble": clongdouble,
              "datetime64": datetime64, "timedelta64": timedelta64})
    return d
