"""`numpy.lib._stride_tricks_impl` — the private module behind
`numpy.lib.stride_tricks`.

Upstream test modules import `as_strided` and `DummyArray` from here directly,
so the name has to exist even though the public alias is what user code uses.
"""

__all__ = ["broadcast_to", "broadcast_arrays", "broadcast_shapes"]


class DummyArray:
    """Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """

    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base


def as_strided(x, shape=None, strides=None, subok=False, writeable=True,
               *, check_bounds=None):
    """Create a view into the array with the given shape and strides."""
    from _rnp import _as_strided
    return _as_strided(x, shape, strides, writeable)


def sliding_window_view(x, window_shape, axis=None, *, subok=False,
                        writeable=False):
    from .._stubs import not_implemented
    return not_implemented(
        "numpy.lib.stride_tricks.sliding_window_view")(
            x, window_shape, axis, subok=subok, writeable=writeable)


def broadcast_to(array, shape, subok=False):
    from .. import broadcast_to as _broadcast_to
    return _broadcast_to(array, shape)


def broadcast_shapes(*args):
    from .. import broadcast_shapes as _broadcast_shapes
    return _broadcast_shapes(*args)


def broadcast_arrays(*args, subok=False):
    from .. import asarray
    arrays = [asarray(a) for a in args]
    shape = broadcast_shapes(*[a.shape for a in arrays])
    return tuple(broadcast_to(a, shape) for a in arrays)
