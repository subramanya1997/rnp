# ---------------------------------------------------------------------------
# Ported from upstream numpy lib/format.py (v2.5.2), verbatim except for the
# import rewrites listed below, which stand in for numpy internals the port
# does not expose in the same shape.  Regenerate with harness-side port.py.
#
#   (no rewrites)
# ---------------------------------------------------------------------------
from ._format_impl import (  # noqa: F401
    ARRAY_ALIGN,
    BUFFER_SIZE,
    EXPECTED_KEYS,
    GROWTH_AXIS_MAX_DIGITS,
    MAGIC_LEN,
    MAGIC_PREFIX,
    __all__,
    __doc__,
    descr_to_dtype,
    drop_metadata,
    dtype_to_descr,
    header_data_from_array_1_0,
    isfileobj,
    magic,
    open_memmap,
    read_array,
    read_array_header_1_0,
    read_array_header_2_0,
    read_magic,
    write_array,
    write_array_header_1_0,
    write_array_header_2_0,
)
