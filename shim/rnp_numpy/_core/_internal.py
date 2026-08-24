"""`numpy._core._internal` — the pure-Python helpers numpy keeps here."""

import ctypes
import re

from .._stubs import not_implemented

format_re = re.compile(r'(?P<order1>[<>|=]?)'
                       r'(?P<repeats> *[(]?[ ,0-9]*[)]? *)'
                       r'(?P<order2>[<>|=]?)'
                       r'(?P<dtype>[A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?)')
sep_re = re.compile(r'\s*,\s*')
space_re = re.compile(r'\s+$')

_pep3118_native_map = {
    '?': '?', 'c': 'S1', 'b': 'b', 'B': 'B', 'h': 'h', 'H': 'H', 'i': 'i',
    'I': 'I', 'l': 'l', 'L': 'L', 'q': 'q', 'Q': 'Q', 'e': 'e', 'f': 'f',
    'd': 'd', 'g': 'g', 'Zf': 'F', 'Zd': 'D', 'Zg': 'G', 's': 'S', 'w': 'U',
    'O': 'O', 'x': 'V',
}

_dtype_from_pep3118 = not_implemented("numpy._core._internal._dtype_from_pep3118")
_view_is_safe = not_implemented("numpy._core._internal._view_is_safe")
_gcd = not_implemented("numpy._core._internal._gcd")
def _getintp_ctype():
    return ctypes.c_ssize_t


class _ctypes:
    """The pointer/shape facade exposed as ``ndarray.ctypes``."""

    def __init__(self, array, ptr=None):
        self._arr = array
        if ptr is None:
            ptr = array.__array_interface__["data"][0]
        self._data = ctypes.c_void_p(ptr)

    def data_as(self, obj):
        ptr = ctypes.cast(ctypes.c_void_p(self.data), obj)
        ptr._arr = self._arr
        return ptr

    def shape_as(self, obj):
        if self._arr.ndim == 0:
            return None
        return (obj * self._arr.ndim)(*self._arr.shape)

    def strides_as(self, obj):
        if self._arr.ndim == 0:
            return None
        return (obj * self._arr.ndim)(*self._arr.strides)

    @property
    def data(self):
        return self._data.value

    @property
    def shape(self):
        return self.shape_as(_getintp_ctype())

    @property
    def strides(self):
        return self.strides_as(_getintp_ctype())

    @property
    def _as_parameter_(self):
        return self.data_as(ctypes.c_void_p)
array_function_errmsg_formatter = not_implemented(
    "numpy._core._internal.array_function_errmsg_formatter")
array_ufunc_errmsg_formatter = not_implemented(
    "numpy._core._internal.array_ufunc_errmsg_formatter")
npy_ctypes_check = not_implemented("numpy._core._internal.npy_ctypes_check")


def _makenames_list(adict, align):
    raise NotImplementedError(
        "numpy._core._internal._makenames_list is not implemented by rnp")


def _commastring(astr):
    """Transcribed from numpy: split a comma-separated dtype string."""
    startindex = 0
    result = []
    islist = False
    while startindex < len(astr):
        mo = format_re.match(astr, pos=startindex)
        try:
            (order1, repeats, order2, dtype) = mo.groups()
        except (TypeError, AttributeError):
            raise ValueError(
                f'format number {len(result) + 1} of "{astr}" is not recognized'
            ) from None
        startindex = mo.end()
        if startindex < len(astr):
            if space_re.match(astr, pos=startindex):
                startindex = len(astr)
            else:
                mo = sep_re.match(astr, pos=startindex)
                if not mo:
                    raise ValueError(
                        'format number %d of "%s" is not recognized' %
                        (len(result) + 1, astr))
                startindex = mo.end()
                islist = True
        if order2 == '':
            order = order1
        elif order1 == '':
            order = order2
        else:
            order1 = _convorder.get(order1, order1)
            order2 = _convorder.get(order2, order2)
            if order1 != order2:
                raise ValueError(
                    f'inconsistent byte-order specification {order1} and {order2}')
            order = order1
        if order in ('|', '=', '<' if _NATIVE_LITTLE else '>'):
            order = ''
        dtype = order + dtype
        if repeats == '':
            newitem = dtype
        else:
            if (repeats[0] == "(" and repeats[-1] == ")"
                    and repeats[1:-1].strip() != ""
                    and "," not in repeats):
                pass
            newitem = (dtype, eval(repeats))
        result.append(newitem)
    return result if (islist or len(result) > 1) else result[0]


_convorder = {'=': '<'}
_NATIVE_LITTLE = True
