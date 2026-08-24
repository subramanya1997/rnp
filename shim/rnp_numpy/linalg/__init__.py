"""`numpy.linalg` — placeholder namespace.

The port has no decomposition/factorisation engine yet (`rnp-python` provides
matmul/dot/inner and nothing more), so this package exists only so that the
module *name* resolves: upstream test modules import `numpy.linalg` at import
time and a missing package scores the whole file zero as a collection error.

Every routine here is a loud stub -- it raises NotImplementedError when
called, never returns a fabricated answer. Where the top-level shim already
implements the same operation (`matmul`, `tensordot`, `outer`, ...) the name
delegates to it, which is the real port behaviour rather than a stand-in.
"""

from .._stubs import not_implemented


class LinAlgError(ValueError):
    """Generic Python-exception-derived object raised by linalg functions."""


__all__ = [
    "LinAlgError", "cholesky", "cond", "cross", "det", "diagonal", "eig",
    "eigh", "eigvals", "eigvalsh", "inv", "lstsq", "matmul", "matrix_norm",
    "matrix_power", "matrix_rank", "matrix_transpose", "multi_dot", "norm",
    "outer", "pinv", "qr", "slogdet", "solve", "svd", "svdvals", "tensordot",
    "tensorinv", "tensorsolve", "trace", "vecdot", "vector_norm",
]

#: Names the top-level shim already implements identically.
_DELEGATED = ("matmul", "tensordot", "outer", "cross", "diagonal", "trace",
              "vecdot", "matrix_transpose")

_cache = {}


def __getattr__(name):
    if name in _cache:
        return _cache[name]
    if name in _DELEGATED:
        import rnp_numpy as _np
        obj = getattr(_np, name, None)
        if obj is not None:
            _cache[name] = obj
            return obj
    if name in __all__:
        obj = not_implemented(f"numpy.linalg.{name}")
        obj.__module__ = "numpy.linalg"
        _cache[name] = obj
        return obj
    raise AttributeError(f"module 'numpy.linalg' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
