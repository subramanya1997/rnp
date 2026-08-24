# ---------------------------------------------------------------------------
# Ported from upstream numpy lib/npyio.py (v2.5.2), verbatim except for the
# import rewrites listed below, which stand in for numpy internals the port
# does not expose in the same shape.  Regenerate with harness-side port.py.
#
#   (no rewrites)
# ---------------------------------------------------------------------------
from ._npyio_impl import DataSource, NpzFile, __doc__  # noqa: F401
