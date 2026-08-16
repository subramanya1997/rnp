"""`numpy._core.umath` — the ufunc namespace.

The arithmetic and comparison ufuncs are real (they are `_rnp` functions);
everything else is a loud stub until M3 lands the full ufunc machinery.
"""

from .. import (  # noqa: F401
    add,
    divide,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    multiply,
    not_equal,
    subtract,
    true_divide,
)
from .._stubs import ufunc_stub

_UNARY = [
    "absolute", "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh",
    "cbrt", "ceil", "conj", "conjugate", "cos", "cosh", "deg2rad", "degrees",
    "exp", "exp2", "expm1", "fabs", "floor", "invert", "isfinite", "isinf",
    "isnan", "isnat", "log", "log10", "log1p", "log2", "logical_not",
    "negative", "positive", "rad2deg", "radians", "reciprocal", "rint",
    "sign", "signbit", "sin", "sinh", "spacing", "sqrt", "square", "tan",
    "tanh", "trunc", "_ones_like", "bitwise_count",
]

_BINARY = [
    "arctan2", "bitwise_and", "bitwise_or", "bitwise_xor", "copysign",
    "float_power", "floor_divide", "fmax", "fmin", "fmod", "gcd", "heaviside",
    "hypot", "lcm", "ldexp", "left_shift", "logaddexp", "logaddexp2",
    "logical_and", "logical_or", "logical_xor", "maximum", "minimum", "mod",
    "nextafter", "power", "remainder", "right_shift", "matmul", "vecdot",
    "matvec", "vecmat", "bitwise_left_shift", "bitwise_right_shift",
    "bitwise_invert", "pow", "divmod",
]

_MISC = ["frexp", "modf"]

for _n in _UNARY:
    globals()[_n] = ufunc_stub(f"numpy.{_n}", nin=1)
for _n in _BINARY:
    globals()[_n] = ufunc_stub(f"numpy.{_n}", nin=2)
for _n in _MISC:
    globals()[_n] = ufunc_stub(f"numpy.{_n}", nin=1, nout=2)
del _n

# Constants numpy exposes from the umath module.
pi = 3.141592653589793
e = 2.718281828459045
euler_gamma = 0.5772156649015329

FLOATING_POINT_SUPPORT = 1
UFUNC_BUFSIZE_DEFAULT = 8192
UFUNC_PYVALS_NAME = "UFUNC_PYVALS"
ERR_IGNORE, ERR_WARN, ERR_RAISE, ERR_CALL, ERR_PRINT, ERR_LOG, ERR_DEFAULT = \
    0, 1, 2, 3, 4, 5, 521

_ALL = _UNARY + _BINARY + _MISC
