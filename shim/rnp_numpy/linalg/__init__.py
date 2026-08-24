"""NumPy-compatible linear algebra namespace."""

from . import _linalg
from ._linalg import *  # noqa: F401,F403

__all__ = _linalg.__all__.copy()
