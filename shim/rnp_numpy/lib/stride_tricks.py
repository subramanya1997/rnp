"""`numpy.lib.stride_tricks` — the public alias of `_stride_tricks_impl`."""

from ._stride_tricks_impl import (  # noqa: F401
    DummyArray,
    as_strided,
    broadcast_arrays,
    broadcast_shapes,
    broadcast_to,
    sliding_window_view,
)

__all__ = ["as_strided", "sliding_window_view"]
