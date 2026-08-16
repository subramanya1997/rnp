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
import math as _math
import struct as _struct
from decimal import Decimal as _Decimal

import _rnp

from . import _errstate

dtype = _rnp.dtype
ndarray = _rnp.ndarray

_SCALAR_BY_NAME = {}


# ---------------------------------------------------------------------------
# repr helpers
# ---------------------------------------------------------------------------

_PACK = {"float16": "e", "float32": "f", "float64": "d"}

#: The decimal exponent at (or above) which numpy switches a float scalar's
#: str/repr from positional to scientific. Probed from numpy 2.5.2: the
#: positional window is `-4 <= exp10 <= {2, 5, 15}` for half/single/double.
_SCI_AT = {"float16": 3, "float32": 6, "float64": 16}


def _shortest(v, name):
    """`(value, significant digits)` of the shortest decimal that round-trips.

    ``repr(np.float32(0.1))`` is ``np.float32(0.1)``, not the float64 spelling
    of the same bits, so the search runs against the *narrow* type's bytes.
    """
    if not _math.isfinite(v):
        return v, 1
    code = _PACK[name]
    target = _struct.pack(code, v)
    for prec in range(1, 18):
        cand = _builtins.float("%.*e" % (prec - 1, v))
        try:
            if _struct.pack(code, cand) == target:
                return cand, prec
        except (OverflowError, ValueError):
            continue
    return v, 17


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
    short, nd = _shortest(v, name)
    if _Decimal(v).adjusted() < -4 or _Decimal(v).adjusted() >= _SCI_AT[name]:
        return "%.*e" % (nd - 1, short)
    # The digit count is measured against the *rounded* value: float16 0.1 is
    # stored as 0.09997..., whose exponent is -2, but it prints as `0.1`.
    return "%.*f" % (max(1, nd - 1 - _Decimal(short).adjusted()), short)


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
        if dt is None:
            return self
        return self.__array__().view(dt)

    def fill(self, value):
        raise ValueError("cannot write to a numpy scalar")

    def tobytes(self, order="C"):
        return self.__array__().tobytes() if hasattr(
            self.__array__(), "tobytes") else bytes(memoryview(self.__array__()))

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        return arr

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


class number(generic):
    __slots__ = ()


class integer(number):
    __slots__ = ()


class signedinteger(integer):
    __slots__ = ()


class unsignedinteger(integer):
    __slots__ = ()


class inexact(number):
    __slots__ = ()


class floating(inexact):
    __slots__ = ()


class complexfloating(inexact):
    __slots__ = ()


class flexible(generic):
    __slots__ = ()


class character(flexible):
    __slots__ = ()


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


def _install_operators(cls):
    """Give one concrete scalar class its full operator surface."""

    def make(op, name):
        def fwd(self, other):
            return _binary(name, self, other)

        def rev(self, other):
            return _binary(name, other, self)

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

    def make_cmp(name):
        def f(self, other):
            return _binary(name, self, other)
        return f

    for op, name in (("lt", "less"), ("le", "less_equal"),
                     ("gt", "greater"), ("ge", "greater_equal")):
        setattr(cls, f"__{op}__", make_cmp(name))

    def __eq__(self, other):
        r = _binary("equal", self, other)
        return NotImplemented if r is NotImplemented else r

    def __ne__(self, other):
        r = _binary("not_equal", self, other)
        return NotImplemented if r is NotImplemented else r

    cls.__eq__ = __eq__
    cls.__ne__ = __ne__
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
    if builtin is not None:
        # `float64(floating, float)` -- the builtin has to carry the payload,
        # so `__new__` routes through it.
        def __new__(cls, value=0, *extra, _raw=None, **kw):
            # numpy's scalar constructors take the value plus the same
            # trailing arguments the corresponding builtin accepts.
            v = _raw if _raw is not None else _cast(cls, value, *extra, **kw)
            self = builtin.__new__(cls, v)
            self._v = v
            return self
    else:
        def __new__(cls, value=0, *extra, _raw=None, **kw):
            v = _raw if _raw is not None else _cast(cls, value, *extra, **kw)
            self = object.__new__(cls)
            self._v = v
            return self

    ns["__new__"] = __new__
    cls = _ScalarMeta(clsname or name, bases, ns)

    def _wrap(value, _cls=cls):
        return _cls(_raw=value)

    cls._wrap = staticmethod(_wrap)
    _install_operators(cls)
    _install_numeric_extras(cls, d)
    _SCALAR_BY_NAME.setdefault(d.name, cls)
    return cls


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
    _check_range(cls.dtype, value)
    return _rnp._scalar_cast(cls.dtype, value)


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
        cls.is_integer = lambda self: True
        cls.as_integer_ratio = lambda self: (_builtins.int(self._v), 1)
        cls.numerator = property(lambda self: type(self)(self._v))
        cls.denominator = property(lambda self: type(self)(1))

    if kind == "f":
        cls.is_integer = lambda self: _builtins.float(self._v).is_integer()
        cls.as_integer_ratio = lambda self: _builtins.float(
            self._v).as_integer_ratio()
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
        # as `np.int64(0)`.
        cls.__repr__ = lambda self, n=name: f"np.{n}({self._v})"
        cls.__str__ = lambda self: str(self._v)

    # reductions, for parity with 0-d arrays
    for m in ("sum", "prod", "min", "max", "mean", "all", "any", "argmin",
              "argmax", "cumsum", "cumprod", "ptp", "std", "var", "round"):
        if not hasattr(cls, m):
            setattr(cls, m, _forward(m))


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
        return f"np.str_({str.__repr__(self)})"

    def __str__(self):
        return str.__str__(self)

    def __array__(self, dtype_=None, copy=None):
        return _rnp.array(str(self))


class void(flexible, metaclass=_FlexMeta, char="V"):
    __module__ = "numpy"
    __slots__ = ("_v",)

    def __new__(cls, value=b"", dtype=None):
        self = object.__new__(cls)
        if isinstance(value, int):
            self._v = b"\x00" * value
        elif isinstance(value, (bytes, bytearray, memoryview)):
            self._v = bytes(value)
        else:
            self._v = bytes(value)
        return self

    @property
    def dtype(self):
        return dtype(f"V{len(self._v)}")

    def __bytes__(self):
        return self._v

    def __len__(self):
        return len(self._v)

    def __repr__(self):
        inner = "".join("\\x%02x" % b for b in self._v)
        return f"np.void(b'{inner}')"

    __str__ = __repr__


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

sctypes = {
    "int": [int8, int16, int32, int64],
    "uint": [uint8, uint16, uint32, uint64],
    "float": [float16, float32, float64, longdouble],
    "complex": [complex64, complex128, clongdouble],
    "others": [_builtins.bool, object, _builtins.bytes, str, _builtins.memoryview],
}


#: The map handed to the Rust side so `dtype.type` and element extraction can
#: build the right scalar class.
def registry():
    d = dict(_SCALAR_BY_NAME)
    d["object_"] = object_
    # The C-type aliases are distinct classes; `np.dtype('q').type` must find
    # `np.longlong` rather than `np.int64`.
    d.update({"longlong": longlong, "ulonglong": ulonglong,
              "longdouble": longdouble, "clongdouble": clongdouble,
              "datetime64": datetime64, "timedelta64": timedelta64})
    return d
