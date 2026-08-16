#!/usr/bin/env python3
"""Differential check for *object-dtype* ufuncs: the port vs real numpy.

Same idiom as `dev_check.py`: real numpy and the port's shim (`rnp_numpy`) are
both imported normally into this one process -- no import redirection -- the
identical Python objects are fed to each, and the two answers are compared.
Object arrays hold real Python objects, so the comparison is on the objects
themselves (`repr` plus exact type), not on bytes.

Errors count as results: for every case where numpy raises, the port must
raise the same exception type with the same message.

Usage: .venv/bin/python harness/dev_check_object.py
"""
import os
import sys
import traceback
from decimal import Decimal
from fractions import Fraction

import numpy as np

_SHIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shim")
if _SHIM_DIR not in sys.path:
    sys.path.insert(0, _SHIM_DIR)

import rnp_numpy as rnp  # noqa: E402

CHECKS = 0
FAILURES = []


# ---------------------------------------------------------------------------
# Element types
# ---------------------------------------------------------------------------

class Plain:
    """A class with no numeric protocol at all: every operator must fail."""

    def __repr__(self):
        return "Plain()"


class Boxed:
    """A full numeric type, so every operator loop has something to call."""

    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return f"Boxed({self.v})"

    def __eq__(self, o):
        return isinstance(o, Boxed) and self.v == o.v

    def __hash__(self):
        return hash(self.v)

    def _val(self, o):
        return o.v if isinstance(o, Boxed) else o

    def __add__(self, o):
        return Boxed(self.v + self._val(o))

    def __radd__(self, o):
        return Boxed(self._val(o) + self.v)

    def __sub__(self, o):
        return Boxed(self.v - self._val(o))

    def __mul__(self, o):
        return Boxed(self.v * self._val(o))

    def __truediv__(self, o):
        return Boxed(self.v / self._val(o))

    def __floordiv__(self, o):
        return Boxed(self.v // self._val(o))

    def __mod__(self, o):
        return Boxed(self.v % self._val(o))

    def __pow__(self, o, m=None):
        return Boxed(self.v ** self._val(o))

    def __neg__(self):
        return Boxed(-self.v)

    def __pos__(self):
        return Boxed(+self.v)

    def __abs__(self):
        return Boxed(abs(self.v))

    def __lt__(self, o):
        return self.v < self._val(o)

    def __le__(self, o):
        return self.v <= self._val(o)

    def __gt__(self, o):
        return self.v > self._val(o)

    def __ge__(self, o):
        return self.v >= self._val(o)

    def __bool__(self):
        return bool(self.v)

    def __floor__(self):
        return Boxed(int(self.v // 1))

    def __ceil__(self):
        return Boxed(int(-((-self.v) // 1)))

    def __trunc__(self):
        return Boxed(int(self.v))


class Mathy:
    """Carries the same-named methods numpy's `TD(P, ...)` loops call."""

    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return f"Mathy({self.v})"

    def __eq__(self, o):
        return isinstance(o, Mathy) and self.v == o.v

    def __hash__(self):
        return hash(self.v)


def _mk_method(name):
    def m(self, *a):
        return f"{name}({self.v}{''.join(',' + repr(x) for x in a)})"
    return m


for _n in ("sqrt", "cbrt", "exp", "exp2", "expm1", "log", "log2", "log10",
           "log1p", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh",
           "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "rint", "fabs",
           "degrees", "radians", "deg2rad", "rad2deg", "conjugate",
           "bit_count"):
    setattr(Mathy, _n, _mk_method(_n))
for _n in ("fmod", "arctan2", "hypot", "logical_xor"):
    setattr(Mathy, _n, _mk_method(_n))
del _n


class Raiser:
    """Every operator raises, so error propagation can be checked exactly."""

    def __repr__(self):
        return "Raiser()"

    def __add__(self, o):
        raise ZeroDivisionError("raiser add")

    def __radd__(self, o):
        raise ZeroDivisionError("raiser radd")

    def __mul__(self, o):
        raise KeyError("raiser mul")

    def __lt__(self, o):
        raise RuntimeError("raiser lt")

    def sqrt(self):
        raise ValueError("raiser sqrt")


class Reflected:
    """Only reflected operators, exercising the `NotImplemented` protocol."""

    def __repr__(self):
        return "Reflected()"

    def __radd__(self, o):
        return ("radd", o)

    def __rmul__(self, o):
        return ("rmul", o)


class Notimpl:
    """Returns `NotImplemented`, so Python's own fallback must run."""

    def __repr__(self):
        return "Notimpl()"

    def __add__(self, o):
        return NotImplemented

    def __sub__(self, o):
        return NotImplemented


# ---------------------------------------------------------------------------
# Comparison plumbing
# ---------------------------------------------------------------------------

def describe(x):
    """A representation that distinguishes value *and* type."""
    if isinstance(x, (np.ndarray, rnp.ndarray)):
        return f"array[{x.dtype}]{list(x.shape)}={describe(x.tolist())}"
    if isinstance(x, list):
        return "[" + ", ".join(describe(v) for v in x) + "]"
    if isinstance(x, tuple):
        return "(" + ", ".join(describe(v) for v in x) + ")"
    if isinstance(x, float) and x != x:
        return "nan"
    return f"{type(x).__name__}:{x!r}"


def run(fn, mod, mkarray):
    """Evaluate `fn` against one library, returning ("ok", v) or ("exc", ...)."""
    try:
        return ("ok", describe(fn(mod, mkarray)))
    except Exception as e:  # noqa: BLE001
        return ("exc", type(e).__name__, str(e))


def np_array(values, shape=None):
    a = np.array(values, dtype=object)
    return a.reshape(shape) if shape is not None else a


def rnp_array(values, shape=None):
    a = rnp.array(values, dtype=object)
    return a.reshape(shape) if shape is not None else a


def compare(label, fn):
    """Run `fn` on both libraries and record any divergence."""
    global CHECKS
    CHECKS += 1
    want = run(fn, np, np_array)
    got = run(fn, rnp, rnp_array)
    if want != got:
        FAILURES.append((label, f"port {got!r} != numpy {want!r}"))


# ---------------------------------------------------------------------------
# The ufunc tables (probed from real numpy 2.5.2)
# ---------------------------------------------------------------------------

#: `numpy._core.umath._ones_like` is deliberately absent: the shim's generated
#: ufunc table does not carry the private ufuncs at all, so there is nothing to
#: compare against.  Its object loop exists and is covered by the Rust unit
#: test `classify_covers_the_probed_object_loops`.
UNARY = [
    "negative", "positive", "absolute", "invert", "sign", "square",
    "reciprocal", "conjugate", "logical_not",
    "floor", "ceil", "trunc", "rint", "fabs",
    "sqrt", "cbrt", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "degrees", "radians", "deg2rad", "rad2deg", "bitwise_count",
]

#: Unary ufuncs numpy gives no object loop at all.
UNARY_NO_LOOP = ["isnan", "isinf", "isfinite", "signbit", "spacing"]

BINARY = [
    "add", "subtract", "multiply", "divide", "true_divide", "floor_divide",
    "remainder", "mod", "power", "bitwise_and", "bitwise_or", "bitwise_xor",
    "left_shift", "right_shift", "logical_and", "logical_or", "logical_xor",
    "maximum", "minimum", "fmax", "fmin", "gcd", "lcm",
    "fmod", "arctan2", "hypot",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
]

BINARY_NO_LOOP = ["copysign", "nextafter", "ldexp", "logaddexp", "logaddexp2",
                  "heaviside", "float_power", "divmod"]

REDUCTIONS = ["add", "multiply", "minimum", "maximum", "logical_and",
              "logical_or", "logical_xor", "bitwise_and", "bitwise_or",
              "bitwise_xor", "subtract", "gcd", "lcm", "fmax", "fmin"]


def uf(mod, name):
    """The ufunc `name` on `mod`, or None if that module has no such name."""
    f = getattr(mod, name, None)
    if f is None:
        f = getattr(getattr(mod, "_core", None), "umath", None)
        f = getattr(f, name, None) if f is not None else None
    return f


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

#: Element sets fed to the unary loops.  Kept as *thunks* so each library
#: builds its own fresh objects.
UNARY_OPERANDS = [
    ("ints", lambda: [1, 2, 3]),
    ("negints", lambda: [-5, 0, 7]),
    ("floats", lambda: [1.5, -2.25, 0.0]),
    ("bools", lambda: [True, False]),
    ("fractions", lambda: [Fraction(3, 2), Fraction(-1, 4)]),
    ("decimals", lambda: [Decimal("1.5"), Decimal("-2")]),
    ("strings", lambda: ["ab", ""]),
    ("lists", lambda: [[1, 2], [3]]),
    ("none", lambda: [None]),
    ("boxed", lambda: [Boxed(3), Boxed(-2)]),
    ("mathy", lambda: [Mathy(1), Mathy(2)]),
    ("plain", lambda: [Plain()]),
    ("raiser", lambda: [Raiser()]),
    ("mixed", lambda: [1, "a", Boxed(2)]),
]

BINARY_OPERANDS = [
    ("int/int", lambda: ([6, 7, 8], [2, 3, 4])),
    ("int/negint", lambda: ([6, -7], [-2, 3])),
    ("float/float", lambda: ([1.5, 6.0], [0.5, 4.0])),
    ("bool/bool", lambda: ([True, False], [False, True])),
    ("frac/frac", lambda: ([Fraction(3, 2), Fraction(4, 1)],
                           [Fraction(1, 2), Fraction(6, 1)])),
    ("dec/dec", lambda: ([Decimal("6"), Decimal("7.5")],
                         [Decimal("2"), Decimal("2.5")])),
    ("str/str", lambda: (["ab", "c"], ["de", "f"])),
    ("str/int", lambda: (["ab", "c"], [3, 2])),
    ("list/list", lambda: ([[1], [2]], [[3], [4]])),
    ("boxed/boxed", lambda: ([Boxed(6), Boxed(7)], [Boxed(2), Boxed(3)])),
    ("boxed/int", lambda: ([Boxed(6), Boxed(7)], [2, 3])),
    ("mathy/int", lambda: ([Mathy(1), Mathy(2)], [3, 4])),
    ("plain/plain", lambda: ([Plain()], [Plain()])),
    ("plain/int", lambda: ([Plain()], [1])),
    ("raiser/int", lambda: ([Raiser()], [1])),
    ("int/raiser", lambda: ([1], [Raiser()])),
    ("reflected", lambda: ([1], [Reflected()])),
    ("notimpl/int", lambda: ([Notimpl()], [1])),
    ("none/int", lambda: ([None], [1])),
    ("mixed", lambda: ([1, "a"], [2, "b"])),
]


def section_unary():
    for name in UNARY + UNARY_NO_LOOP:
        for label, make in UNARY_OPERANDS:
            def fn(mod, mk, name=name, make=make):
                f = uf(mod, name)
                if f is None:
                    return "NO-SUCH-UFUNC"
                return f(mk(make()))
            compare(f"{name}(object[{label}])", fn)


def section_binary():
    for name in BINARY + BINARY_NO_LOOP:
        for label, make in BINARY_OPERANDS:
            def fn(mod, mk, name=name, make=make):
                f = uf(mod, name)
                if f is None:
                    return "NO-SUCH-UFUNC"
                a, b = make()
                return f(mk(a), mk(b))
            compare(f"{name}(object[{label}])", fn)


def section_mixed_dtype():
    """One object operand, one operand of a concrete dtype."""
    others = [
        ("int64", lambda m: m.array([10, 20], dtype="int64")),
        ("int8", lambda m: m.array([10, 20], dtype="int8")),
        ("uint16", lambda m: m.array([10, 20], dtype="uint16")),
        ("float32", lambda m: m.array([1.5, 2.5], dtype="float32")),
        ("float64", lambda m: m.array([1.5, 2.5], dtype="float64")),
        ("bool", lambda m: m.array([True, False], dtype="bool")),
        ("pyint", lambda m: 5),
        ("pyfloat", lambda m: 2.5),
        ("npscalar", lambda m: m.float64(2.5)),
        ("list", lambda m: [3, 4]),
    ]
    for name in ["add", "subtract", "multiply", "divide", "floor_divide",
                 "remainder", "power", "maximum", "minimum", "equal", "less",
                 "greater_equal", "logical_and", "bitwise_or"]:
        for olabel, other in others:
            for swap in (False, True):
                def fn(mod, mk, name=name, other=other, swap=swap):
                    f = uf(mod, name)
                    a = mk([6, 7])
                    b = other(mod)
                    return f(b, a) if swap else f(a, b)
                compare(f"{name}(object,{olabel},swap={swap})", fn)


def section_promotion():
    for spec in ["int64", "float64", "bool", "int8", "complex128"]:
        def fn(mod, mk, spec=spec):
            return str(mod.result_type(mod.dtype(object), mod.dtype(spec)))
        compare(f"result_type(object,{spec})", fn)

        def fn2(mod, mk, spec=spec):
            return str(mod.promote_types(mod.dtype(object), mod.dtype(spec)))
        compare(f"promote_types(object,{spec})", fn2)

    def dt(mod, mk):
        return str(mod.add(mk([1, 2]), mod.array([1, 2])).dtype)
    compare("add(object,int64).dtype", dt)

    for name in ["equal", "less", "greater", "not_equal", "less_equal",
                 "greater_equal"]:
        def cmpdt(mod, mk, name=name):
            return str(uf(mod, name)(mk([1, 2]), mk([1, 3])).dtype)
        compare(f"{name}(object,object).dtype", cmpdt)

        def cmpobj(mod, mk, name=name):
            out = mod.empty(2, dtype=object)
            r = uf(mod, name)(mk([1, 2]), mk([1, 3]), out=out)
            return (str(r.dtype), r.tolist())
        compare(f"{name}(object,object,out=object)", cmpobj)


def section_broadcast_out_where():
    shapes = [
        ((2, 3), (3,)),
        ((2, 3), (2, 1)),
        ((1, 3), (2, 1)),
        ((3,), ()),
        ((2, 2), (2, 2)),
    ]
    for name in ["add", "multiply", "subtract", "maximum", "less", "equal"]:
        for sa, sb in shapes:
            def fn(mod, mk, name=name, sa=sa, sb=sb):
                na = int(np.prod(sa)) if sa else 1
                nb = int(np.prod(sb)) if sb else 1
                a = mk(list(range(1, na + 1)), sa)
                b = mk(list(range(1, nb + 1)), sb)
                return uf(mod, name)(a, b)
            compare(f"{name} broadcast {sa}x{sb}", fn)

    masks = [[True, False, True], [False, False, False], [True, True, True],
             True]
    for name in ["add", "multiply", "subtract", "less"]:
        for mi, mask in enumerate(masks):
            def fnw(mod, mk, name=name, mask=mask):
                a = mk([1, 2, 3])
                b = mk([10, 20, 30])
                out = mod.empty(3, dtype=object)
                r = uf(mod, name)(a, b, out=out, where=mask)
                return (r.tolist(), out.tolist(), str(out.dtype))
            compare(f"{name} where={mi} out=object", fnw)

            def fnw2(mod, mk, name=name, mask=mask):
                a = mk([1, 2, 3])
                b = mk([10, 20, 30])
                out = mod.array([100, 200, 300], dtype=object)
                uf(mod, name)(a, b, out=out, where=mask)
                return out.tolist()
            compare(f"{name} where={mi} out=filled", fnw2)

    for name in ["add", "multiply"]:
        def fno(mod, mk, name=name):
            out = mod.empty(3, dtype=object)
            r = uf(mod, name)(mk([1, 2, 3]), mk([4, 5, 6]), out=out)
            return (r is out, out.tolist())
        compare(f"{name} out identity", fno)

    def fn0(mod, mk):
        r = mod.add(mk(1), mk(2))
        return (type(r).__name__, r)
    compare("add 0-d object returns scalar", fn0)


def section_reductions():
    data = [
        ("ints", [1, 2, 3, 4]),
        ("negints", [3, -1, 4, -1]),
        ("floats", [1.5, 2.5, 3.0]),
        ("bools", [True, False, True]),
        ("fractions", [Fraction(1, 2), Fraction(1, 3)]),
        ("decimals", [Decimal("1.5"), Decimal("2")]),
        ("strings", ["a", "b", "c"]),
        ("boxed", [Boxed(1), Boxed(2), Boxed(3)]),
        ("mathy", [Mathy(1), Mathy(2)]),
        ("plain", [Plain(), Plain()]),
        ("single", [7]),
        ("empty", []),
    ]
    for name in REDUCTIONS:
        for label, values in data:
            def fn(mod, mk, name=name, values=values):
                return uf(mod, name).reduce(mk(list(values)))
            compare(f"{name}.reduce(object[{label}])", fn)

            def fnacc(mod, mk, name=name, values=values):
                return uf(mod, name).accumulate(mk(list(values)))
            compare(f"{name}.accumulate(object[{label}])", fnacc)

    # initial= / where= / keepdims= / axis=
    for name in ["add", "multiply", "minimum", "maximum"]:
        for initial in (None, 0, 10):
            def fni(mod, mk, name=name, initial=initial):
                kw = {} if initial is None else {"initial": initial}
                return uf(mod, name).reduce(mk([1, 2, 3]), **kw)
            compare(f"{name}.reduce initial={initial}", fni)

            def fne(mod, mk, name=name, initial=initial):
                kw = {} if initial is None else {"initial": initial}
                return uf(mod, name).reduce(mk([]), **kw)
            compare(f"{name}.reduce empty initial={initial}", fne)

        for mask in ([True, False, True], [False, False, False]):
            for initial in (None, 1):
                def fnw(mod, mk, name=name, mask=mask, initial=initial):
                    kw = {} if initial is None else {"initial": initial}
                    return uf(mod, name).reduce(mk([1, 2, 3]), where=mask, **kw)
                compare(f"{name}.reduce where initial={initial}", fnw)

    for name in ["add", "multiply", "minimum", "maximum", "logical_or"]:
        for axis in (0, 1, -1, None, (0, 1)):
            for keep in (False, True):
                def fna(mod, mk, name=name, axis=axis, keep=keep):
                    a = mk([1, 2, 3, 4, 5, 6], (2, 3))
                    return uf(mod, name).reduce(a, axis=axis, keepdims=keep)
                compare(f"{name}.reduce 2d axis={axis} keepdims={keep}", fna)

    for name in ["add", "multiply", "maximum"]:
        for axis in (0, 1):
            def fnac(mod, mk, name=name, axis=axis):
                a = mk([1, 2, 3, 4, 5, 6], (2, 3))
                return uf(mod, name).accumulate(a, axis=axis)
            compare(f"{name}.accumulate 2d axis={axis}", fnac)

    for name in ["add", "multiply", "maximum", "minimum"]:
        for idx in ([0, 2], [0], [1, 1, 3], [0, 3, 1]):
            def fnr(mod, mk, name=name, idx=idx):
                return uf(mod, name).reduceat(mk([1, 2, 3, 4, 5]), idx)
            compare(f"{name}.reduceat({idx})", fnr)

    # The `np.*` spellings that go through the reductions.
    for label, values in data:
        for fname in ["sum", "prod", "min", "max", "any", "all", "cumsum",
                      "cumprod"]:
            def fn(mod, mk, fname=fname, values=values):
                r = getattr(mod, fname)(mk(list(values)))
                return (type(r).__name__, describe(r))
            compare(f"np.{fname}(object[{label}])", fn)

    # An explicit non-object dtype= must leave the object loop.
    for dt in ["int64", "float64", "bool"]:
        def fnd(mod, mk, dt=dt):
            r = mod.add.reduce(mk([1, 2, 3]), dtype=dt)
            return (str(mod.asarray(r).dtype), describe(r))
        compare(f"add.reduce(object, dtype={dt})", fnd)


def section_errors():
    """Exception type and message must match exactly, including the index."""
    cases = []
    for name in ["sqrt", "exp", "log", "conjugate", "rint", "fabs", "cbrt",
                 "degrees", "arcsin", "bitwise_count"]:
        for pos, vals in [(0, [Plain(), Mathy(1)]),
                          (1, [Mathy(1), Plain()]),
                          (2, [Mathy(1), Mathy(2), 3])]:
            cases.append((f"{name} missing-method at {pos}", name, vals))
    for label, name, vals in cases:
        def fn(mod, mk, name=name, vals=vals):
            return uf(mod, name)(mk(list(vals)))
        compare(label, fn)

    # Operators that are simply absent.
    for name in ["negative", "positive", "absolute", "invert", "square",
                 "reciprocal", "sign", "floor", "ceil", "trunc"]:
        def fn(mod, mk, name=name):
            return uf(mod, name)(mk([Plain()]))
        compare(f"{name}(Plain) error", fn)

    # An exception raised *inside* an element's operator propagates unchanged.
    for name, vals in [("add", [Raiser()]), ("multiply", [Raiser()]),
                       ("less", [Raiser()]), ("sqrt", [Raiser()])]:
        def fn(mod, mk, name=name, vals=vals):
            f = uf(mod, name)
            a = mk(list(vals))
            return f(a) if f.nin == 1 else f(a, mk([1]))
        compare(f"{name} propagates element exception", fn)

    # Reductions with no identity over an empty array.
    for name in REDUCTIONS:
        def fn(mod, mk, name=name):
            return uf(mod, name).reduce(mk([]))
        compare(f"{name}.reduce(empty) identity", fn)

    # Bad broadcasts and bad axes.
    def bad_bcast(mod, mk):
        return mod.add(mk([1, 2, 3]), mk([1, 2]))
    compare("object bad broadcast", bad_bcast)

    def bad_axis(mod, mk):
        return mod.add.reduce(mk([1, 2, 3]), axis=3)
    compare("object bad axis", bad_axis)

    def scalar_reduce(mod, mk):
        return mod.add.reduce(mk(1))
    compare("object reduce on scalar", scalar_reduce)

    def bad_out(mod, mk):
        return mod.add(mk([1, 2]), mk([1, 2]), out=mod.empty(3, dtype=object))
    compare("object out shape mismatch", bad_out)


def section_sign():
    """`np.sign` has its own three-way probe; the "unorderable" case included."""
    class NoOrder:
        def __lt__(self, o):
            return False

        def __gt__(self, o):
            return False

        def __eq__(self, o):
            return False

        def __hash__(self):
            return 0

        def __repr__(self):
            return "NoOrder()"

    sets = [
        [-3, 0, 7], [0], [-1], [1.5, -1.5, 0.0], [True, False],
        [Fraction(-3, 2), Fraction(0), Fraction(5, 2)],
        [Decimal("-1.5"), Decimal("0"), Decimal("2")],
        [Boxed(-2), Boxed(0), Boxed(2)],
        ["a"], [None], [NoOrder()], [Plain()],
    ]
    for i, vals in enumerate(sets):
        def fn(mod, mk, vals=vals):
            return mod.sign(mk(list(vals)))
        compare(f"sign case {i}", fn)


def section_gcd_lcm():
    sets = [
        ([6, 8], [4, 12]), ([-4], [6]), ([0], [5]), ([0], [0]),
        ([Fraction(4, 1)], [Fraction(6, 1)]),
        ([Decimal(4)], [Decimal(6)]),
        ([4.0], [6.0]), ([Plain()], [1]), (["a"], ["b"]),
        ([2 ** 70], [2 ** 35]),
    ]
    for name in ("gcd", "lcm"):
        for i, (a, b) in enumerate(sets):
            def fn(mod, mk, name=name, a=a, b=b):
                return uf(mod, name)(mk(list(a)), mk(list(b)))
            compare(f"{name} case {i}", fn)


def section_python_operators():
    """The `ndarray` operators must reach the same object loops."""
    ops = [
        ("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b), ("/", lambda a, b: a / b),
        ("//", lambda a, b: a // b), ("%", lambda a, b: a % b),
        ("**", lambda a, b: a ** b), ("&", lambda a, b: a & b),
        ("|", lambda a, b: a | b), ("^", lambda a, b: a ^ b),
        ("<<", lambda a, b: a << b), (">>", lambda a, b: a >> b),
        ("<", lambda a, b: a < b), ("<=", lambda a, b: a <= b),
        (">", lambda a, b: a > b), (">=", lambda a, b: a >= b),
        ("==", lambda a, b: a == b), ("!=", lambda a, b: a != b),
    ]
    operands = [
        ([6, 7], [2, 3]), ([6, 7], 2), ([Boxed(6)], [Boxed(2)]),
        ([Fraction(3, 2)], [Fraction(1, 2)]),
    ]
    for label, op in ops:
        for i, (a, b) in enumerate(operands):
            def fn(mod, mk, op=op, a=a, b=b):
                x = mk(list(a))
                y = mk(list(b)) if isinstance(b, list) else b
                return op(x, y)
            compare(f"operator {label} case {i}", fn)

    for label, op in [("-x", lambda a: -a), ("+x", lambda a: +a),
                      ("~x", lambda a: ~a), ("abs", lambda a: abs(a))]:
        for i, vals in enumerate([[1, 2], [Boxed(3)], [Plain()], [1.5]]):
            def fn(mod, mk, op=op, vals=vals):
                return op(mk(list(vals)))
            compare(f"unary operator {label} case {i}", fn)


def section_dtype_kwarg():
    for name in ["add", "multiply", "less"]:
        for dt in [None, "object", "int64", "float64"]:
            def fn(mod, mk, name=name, dt=dt):
                kw = {} if dt is None else {"dtype": dt}
                r = uf(mod, name)(mk([1, 2]), mk([3, 4]), **kw)
                return (str(r.dtype), describe(r))
            compare(f"{name} dtype={dt}", fn)

    # A non-object array with dtype=object goes to the object loop.
    for name in ["add", "multiply", "sqrt", "negative"]:
        def fn(mod, mk, name=name):
            f = uf(mod, name)
            a = mod.array([1, 2])
            return f(a, a, dtype=object) if f.nin == 2 else f(a, dtype=object)
        compare(f"{name} numeric with dtype=object", fn)


def section_scalars_and_zero_d():
    """0-d object arrays and bare Python objects as operands."""
    values = [1, 2.5, Fraction(1, 2), Decimal("3"), Boxed(4), "s", True]
    for name in ["add", "multiply", "subtract", "less", "equal", "maximum"]:
        for v in values:
            def fn(mod, mk, name=name, v=v):
                a = mod.array(v, dtype=object)
                r = uf(mod, name)(a, a)
                return (type(r).__name__, describe(r))
            compare(f"{name} 0-d {type(v).__name__}", fn)

    for name in ["negative", "absolute", "sign", "square"]:
        for v in values:
            def fn(mod, mk, name=name, v=v):
                r = uf(mod, name)(mod.array(v, dtype=object))
                return (type(r).__name__, describe(r))
            compare(f"{name} 0-d unary {type(v).__name__}", fn)


def section_leaks():
    """A big reduction must not need one interned slab entry per intermediate.

    This is a behavioural check only -- it asserts the *answer*, which is what
    the differential harness can see; the allocation claim is in PLAN.md.
    """
    def fn(mod, mk):
        return mod.add.reduce(mk(list(range(2000))))
    compare("large object reduce", fn)

    def fn2(mod, mk):
        return mod.multiply.reduce(mk([1] * 5000))
    compare("large object multiply reduce", fn2)

    def fn3(mod, mk):
        a = mk(list(range(500)))
        return mod.add(a, a).tolist()[-5:]
    compare("large object binary", fn3)


SECTIONS = [
    ("unary", section_unary),
    ("binary", section_binary),
    ("mixed dtype", section_mixed_dtype),
    ("promotion", section_promotion),
    ("broadcast/out/where", section_broadcast_out_where),
    ("reductions", section_reductions),
    ("errors", section_errors),
    ("sign", section_sign),
    ("gcd/lcm", section_gcd_lcm),
    ("python operators", section_python_operators),
    ("dtype kwarg", section_dtype_kwarg),
    ("scalars/0-d", section_scalars_and_zero_d),
    ("large", section_leaks),
]


def main():
    for label, fn in SECTIONS:
        before = len(FAILURES)
        try:
            fn()
        except Exception:  # noqa: BLE001
            FAILURES.append((f"section {label}",
                             traceback.format_exc().strip().splitlines()[-1]))
        new = len(FAILURES) - before
        if new:
            print(f"  section {label}: {new} divergences")
    print(f"{CHECKS} comparisons, {len(FAILURES)} divergences")
    for name, msg in FAILURES:
        print(f"  FAIL {name}: {msg}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
