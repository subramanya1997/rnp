# ---------------------------------------------------------------------------
# Ported from upstream numpy lib/array_utils.py (v2.5.2), verbatim except for the
# import rewrites listed below, which stand in for numpy internals the port
# does not expose in the same shape.  Regenerate with harness-side port.py.
#
#   (no rewrites)
# ---------------------------------------------------------------------------
from ._array_utils_impl import (  # noqa: F401
    __all__,
    __doc__,
    byte_bounds,
    normalize_axis_index,
    normalize_axis_tuple,
)
