"""`numpy.lib.stride_tricks`."""

from .. import broadcast_to  # noqa: F401
from .._stubs import not_implemented

def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    """A view of `x` with caller-chosen shape and strides."""
    from _rnp import _as_strided
    return _as_strided(x, shape, strides, writeable)
sliding_window_view = not_implemented(
    "numpy.lib.stride_tricks.sliding_window_view")


def broadcast_arrays(*args, subok=False):
    from .. import asarray, broadcast_shapes
    arrays = [asarray(a) for a in args]
    shape = broadcast_shapes(*[a.shape for a in arrays])
    return tuple(broadcast_to(a, shape) for a in arrays)
