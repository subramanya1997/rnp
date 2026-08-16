"""`numpy.ma` — masked arrays are not ported (their own milestone)."""

from .._stubs import not_implemented

MaskedArray = not_implemented("numpy.ma.MaskedArray")
masked_array = not_implemented("numpy.ma.masked_array")
array = not_implemented("numpy.ma.array")
masked = None
nomask = False
getmask = not_implemented("numpy.ma.getmask")
getmaskarray = not_implemented("numpy.ma.getmaskarray")
masked_equal = not_implemented("numpy.ma.masked_equal")
masked_where = not_implemented("numpy.ma.masked_where")
filled = not_implemented("numpy.ma.filled")


def is_masked(x):
    """True when `x` is a MaskedArray with at least one masked entry.

    Masked arrays are not ported yet, so nothing is ever masked.  The name has
    to exist because `numpy.lib._arraysetops_impl.unique` consults it on every
    call.
    """
    return False


def is_mask(m):
    return False
