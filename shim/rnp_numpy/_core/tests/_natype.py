"""Minimal pandas.NA stand-in used by NumPy's StringDType tests.

Adapted from NumPy 2.5.2's vendored ``numpy._core.tests._natype`` helper.
"""

import numbers

import numpy as np

__all__ = ["pd_NA"]


def _binary(name, is_divmod=False):
    is_cmp = name.strip("_") in {"eq", "ne", "le", "lt", "ge", "gt"}

    def method(self, other):
        if (other is pd_NA or isinstance(other, (str, bytes, numbers.Number,
                                                  np.bool))
                or (isinstance(other, np.ndarray) and not other.shape)):
            return (pd_NA, pd_NA) if is_divmod else pd_NA
        if isinstance(other, np.ndarray):
            out = np.empty(other.shape, dtype=object)
            out[:] = pd_NA
            return (out, out.copy()) if is_divmod else out
        if is_cmp and isinstance(other, (np.datetime64, np.timedelta64)):
            return pd_NA
        if isinstance(other, np.datetime64) and name in {"__sub__", "__rsub__"}:
            return pd_NA
        if (isinstance(other, np.timedelta64)
                and name in {"__sub__", "__rsub__", "__add__", "__radd__"}):
            return pd_NA
        return NotImplemented

    method.__name__ = name
    return method


def _unary(name):
    def method(self):
        return pd_NA
    method.__name__ = name
    return method


class NAType:
    def __repr__(self):
        return "<NA>"

    def __format__(self, format_spec):
        try:
            return self.__repr__().__format__(format_spec)
        except ValueError:
            return self.__repr__()

    def __bool__(self):
        raise TypeError("boolean value of NA is ambiguous")

    def __hash__(self):
        return 2**61 - 1

    def __reduce__(self):
        return "pd_NA"

    __add__ = _binary("__add__")
    __radd__ = _binary("__radd__")
    __sub__ = _binary("__sub__")
    __rsub__ = _binary("__rsub__")
    __mul__ = _binary("__mul__")
    __rmul__ = _binary("__rmul__")
    __matmul__ = _binary("__matmul__")
    __rmatmul__ = _binary("__rmatmul__")
    __truediv__ = _binary("__truediv__")
    __rtruediv__ = _binary("__rtruediv__")
    __floordiv__ = _binary("__floordiv__")
    __rfloordiv__ = _binary("__rfloordiv__")
    __mod__ = _binary("__mod__")
    __rmod__ = _binary("__rmod__")
    __divmod__ = _binary("__divmod__", True)
    __rdivmod__ = _binary("__rdivmod__", True)
    __eq__ = _binary("__eq__")
    __ne__ = _binary("__ne__")
    __le__ = _binary("__le__")
    __lt__ = _binary("__lt__")
    __gt__ = _binary("__gt__")
    __ge__ = _binary("__ge__")
    __neg__ = _unary("__neg__")
    __pos__ = _unary("__pos__")
    __abs__ = _unary("__abs__")
    __invert__ = _unary("__invert__")

    def __and__(self, other):
        if other is False:
            return False
        if other is True or other is pd_NA:
            return pd_NA
        return NotImplemented

    __rand__ = __and__

    def __or__(self, other):
        if other is True:
            return True
        if other is False or other is pd_NA:
            return pd_NA
        return NotImplemented

    __ror__ = __or__

    def __xor__(self, other):
        if other is False or other is True or other is pd_NA:
            return pd_NA
        return NotImplemented

    __rxor__ = __xor__


pd_NA = NAType()
