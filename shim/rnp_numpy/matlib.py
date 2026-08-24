"""Matrix-oriented convenience functions, matching ``numpy.matlib``."""
import warnings

warnings.warn(
    "Importing from numpy.matlib is deprecated since 1.19.0. The matrix "
    "subclass is not the recommended way to represent matrices or deal "
    "with linear algebra (see https://docs.scipy.org/doc/numpy/user/"
    "numpy-for-matlab-users.html). Please adjust your code to use regular "
    "ndarray. ", PendingDeprecationWarning, stacklevel=2)

from . import *  # noqa: F401,F403
from . import __version__, random
from . import empty as _empty, eye as _eye, ones as _ones, zeros as _zeros
from . import tile as _tile
from .matrixlib.defmatrix import asmatrix, matrix

__all__ = ["rand", "randn", "repmat"]


def _shape2(shape):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return (1, shape[0]) if len(shape) == 1 else shape


def empty(shape, dtype=None, order="C"):
    return asmatrix(_empty(_shape2(shape), dtype=dtype, order=order))


def ones(shape, dtype=None, order="C"):
    return asmatrix(_ones(_shape2(shape), dtype=dtype, order=order))


def zeros(shape, dtype=None, order="C"):
    return asmatrix(_zeros(_shape2(shape), dtype=dtype, order=order))


def identity(n, dtype=None):
    return asmatrix(_eye(n, dtype=dtype))


def eye(n, M=None, k=0, dtype=float, order="C"):
    return asmatrix(_eye(n, M=M, k=k, dtype=dtype, order=order))


def rand(*args):
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(random.rand(*args))


def randn(*args):
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(random.randn(*args))


def repmat(a, m, n):
    a = asanyarray(a)
    if a.ndim == 0:
        origrows, origcols = 1, 1
    elif a.ndim == 1:
        origrows, origcols = 1, a.shape[0]
    else:
        origrows, origcols = a.shape
    return _tile(a.reshape(origrows, origcols), (m, n))
