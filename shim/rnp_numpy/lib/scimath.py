# ---------------------------------------------------------------------------
# Ported from upstream numpy lib/scimath.py (v2.5.2), verbatim except for the
# import rewrites listed below, which stand in for numpy internals the port
# does not expose in the same shape.  Regenerate with harness-side port.py.
#
#   (no rewrites)
# ---------------------------------------------------------------------------
from ._scimath_impl import (  # noqa: F401
    __all__,
    __doc__,
    arccos,
    arcsin,
    arctanh,
    log,
    log2,
    log10,
    logn,
    power,
    sqrt,
)
