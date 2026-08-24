"""A lightweight two-dimensional ``matrix`` compatibility facade.

The Rust ndarray type cannot be subclassed.  The metaclass therefore treats
two-dimensional ndarrays as matrix instances while construction and
``asmatrix`` enforce matrix's two-dimensional shape rule.
"""
from .. import asanyarray

__all__ = ['matrix', 'bmat', 'asmatrix']

class _MatrixMeta(type):
    def __instancecheck__(cls, obj):
        from .. import ndarray
        return isinstance(obj, ndarray) and obj.ndim == 2

    def __call__(cls, data, dtype=None, copy=True):
        arr = asanyarray(data, dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, arr.shape[0])
        elif arr.ndim > 2:
            raise ValueError("shape too large to be a matrix.")
        return arr.copy() if copy else arr


class matrix(metaclass=_MatrixMeta):
    pass


def asmatrix(data, dtype=None):
    return matrix(data, dtype=dtype, copy=False)


def bmat(obj, ldict=None, gdict=None):
    raise NotImplementedError("numpy.bmat is not implemented by rnp")
