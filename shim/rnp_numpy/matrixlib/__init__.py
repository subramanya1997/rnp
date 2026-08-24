"""`numpy.matrixlib` — see `.defmatrix` for why this placeholder exists."""

from . import defmatrix  # noqa: F401
from .defmatrix import asmatrix, bmat, matrix  # noqa: F401

__all__ = ['matrix', 'bmat', 'asmatrix', 'defmatrix']
