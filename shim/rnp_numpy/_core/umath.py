"""`numpy._core.umath` — the ufunc namespace.

Every name is the same `ufunc` object `numpy.<name>` is bound to; the module
exists because upstream tests import from it directly.
"""

from .._ufunc import ALL as _ALL_UFUNCS

globals().update(_ALL_UFUNCS)

_ALL = list(_ALL_UFUNCS)

# Constants numpy exposes from the umath module.
pi = 3.141592653589793
e = 2.718281828459045
euler_gamma = 0.5772156649015329

FLOATING_POINT_SUPPORT = 1
UFUNC_BUFSIZE_DEFAULT = 8192
UFUNC_PYVALS_NAME = "UFUNC_PYVALS"
ERR_IGNORE, ERR_WARN, ERR_RAISE, ERR_CALL, ERR_PRINT, ERR_LOG, ERR_DEFAULT = \
    0, 1, 2, 3, 4, 5, 521


def seterrobj(*a, **k):
    from .._errstate import seterr
    return seterr()


def geterrobj(*a, **k):
    return [UFUNC_BUFSIZE_DEFAULT, ERR_DEFAULT, None]


def _get_promotion_state():
    return "weak"


def _set_promotion_state(state):
    if state != "weak":
        raise ValueError(f"unsupported promotion state {state!r}")

# The float constants numpy's umath module exposes (upstream tests read them).
PZERO = 0.0
NZERO = -0.0
PINF = float("inf")
NINF = float("-inf")
NAN = float("nan")
