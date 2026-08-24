"""`numpy._core.multiarray` — the names upstream tests import from the C
module. Everything real is re-exported from the top-level shim; the rest are
loud stubs (they raise NotImplementedError when *used*, never silently)."""

from .. import (  # noqa: F401
    arange,
    array,
    asarray,
    broadcast_to,
    can_cast,
    choose,
    compress,
    dtype,
    empty,
    flatiter,
    flatnonzero,
    frombuffer,
    full,
    ndarray,
    nonzero,
    ones,
    promote_types,
    put,
    putmask,
    result_type,
    take,
    where,
    zeros,
)
from .._stubs import not_implemented

# numpy exposes the ABI/API version numbers of the C extension here. The port
# has no C ABI, so it reports the versions of the numpy it targets (2.5.2).
_ARRAY_API_VERSION = 0x00000012


def _get_ndarray_c_version():
    return _ARRAY_API_VERSION


def _reconstruct(subtype, shape, dtype_):
    return ndarray.__new__(subtype, shape, dtype_)


dot = not_implemented("numpy._core.multiarray.dot")
inner = not_implemented("numpy._core.multiarray.inner")
vdot = not_implemented("numpy._core.multiarray.vdot")
matmul = not_implemented("numpy._core.multiarray.matmul")
lexsort = not_implemented("numpy._core.multiarray.lexsort")
bincount = not_implemented("numpy._core.multiarray.bincount")
c_einsum = not_implemented("numpy._core.multiarray.c_einsum")
copyto = not_implemented("numpy._core.multiarray.copyto")
concatenate = not_implemented("numpy._core.multiarray.concatenate")
correlate = not_implemented("numpy._core.multiarray.correlate")
correlate2 = not_implemented("numpy._core.multiarray.correlate2")


def _vec_string(char_array, dtype, method, args=()):
    """Element-wise method call over a string array (see `_core.strings`)."""
    from .strings import _vec_string as _impl
    return _impl(char_array, dtype, method, args)


scalar = not_implemented("numpy._core.multiarray.scalar")
set_datetimeparse_function = not_implemented(
    "numpy._core.multiarray.set_datetimeparse_function")
from .._datetime import (  # noqa: E402
    datetime_data, datetime_as_string, is_busday, busday_count, busday_offset,
    busdaycalendar,
)
# NB: `frombuffer` is deliberately NOT stubbed here -- the buffer-adoption
# lane implements it for real against the Py_buffer protocol.
fromfile = not_implemented("numpy._core.multiarray.fromfile")
fromiter = not_implemented("numpy._core.multiarray.fromiter")
fromstring = not_implemented("numpy._core.multiarray.fromstring")
nested_iters = not_implemented("numpy._core.multiarray.nested_iters")
shares_memory = not_implemented("numpy._core.multiarray.shares_memory")
may_share_memory = not_implemented("numpy._core.multiarray.may_share_memory")
_get_madvise_hugepage = not_implemented(
    "numpy._core.multiarray._get_madvise_hugepage")
_set_madvise_hugepage = not_implemented(
    "numpy._core.multiarray._set_madvise_hugepage")


def normalize_axis_index(axis, ndim, msg_prefix=None):
    """Transcribed from numpy's `normalize_axis_index`."""
    from ..exceptions import AxisError
    if axis < -ndim or axis >= ndim:
        raise AxisError(axis, ndim, msg_prefix)
    return axis % ndim
