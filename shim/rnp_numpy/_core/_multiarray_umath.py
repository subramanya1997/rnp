"""`numpy._core._multiarray_umath` — the single C extension module numpy
builds `numpy._core.multiarray` and `numpy._core.umath` out of.

Upstream tests import it directly (`import numpy._core._multiarray_umath as
ncu`), so the shim presents the union of the two namespaces plus the handful
of private helpers those tests reach for. Nothing here fabricates a result:
what the port cannot answer raises NotImplementedError when *called*.
"""

from .. import _stubs as _stubs_mod
from .._stubs import not_implemented, ufunc

# The union of the two public halves. `umath` goes on top of `multiarray`
# because that is the order numpy's own `_core/__init__` uses, and the only
# names in both are the ufuncs themselves.
from . import multiarray as _multiarray
from . import umath as _umath
from ..lib._rnp_compat import _ArrayFunctionDispatcher, _get_implementing_args

for _mod in (_multiarray, _umath):
    for _name in dir(_mod):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_mod, _name)
del _mod, _name

#: numpy 2.x caps ndim at 64 (NPY_MAXDIMS).
MAXDIMS = 64

#: numpy publishes the runtime CPU-feature table it dispatches on here. This
#: port has no SIMD dispatch table at all, so it reports no features rather
#: than claiming ones it never uses. Tests read it to *enable* extra checks;
#: an empty mapping only ever makes them skip, never pass vacuously.
__cpu_features__ = {}
__cpu_baseline__ = []
__cpu_dispatch__ = []


def _arg_impl(x):
    """Complex argument, elementwise — numpy's private `_arg` ufunc."""
    import rnp_numpy as np
    a = np.asarray(x)
    return np.arctan2(a.imag, a.real)


_arg = ufunc("_arg", nin=1, nout=1, impl=_arg_impl,
             qualname="numpy._core._multiarray_umath._arg")


def _discover_array_parameters(obj, dtype=None):
    """Return the ``(descr, shape)`` array coercion would produce for *obj*.

    numpy discovers these without allocating; the port answers the same
    question by actually performing the coercion, which gives the same
    ``(dtype, shape)`` for everything the port can coerce at all.
    """
    import rnp_numpy as np
    if dtype is not None and isinstance(dtype, type):
        # Callers pass a DType *class* (``type(np.dtype("f8"))``); numpy
        # accepts that and discovers the concrete descriptor from it.
        if issubclass(dtype, np.dtype):
            dtype = np.dtypes._default_for_discovery(dtype)
        else:
            dtype = dtype()
    arr = np.asarray(obj, dtype=dtype)
    return arr.dtype, arr.shape


#: numpy returns a `numpy._BoundArrayMethod` wrapping the C ArrayMethod that
#: implements one dtype-to-dtype cast. The port's casting machinery is not
#: reified as first-class method objects, so this is a loud stub: the tests
#: that use it fail rather than pass on a fake.
_get_castingimpl = not_implemented(
    "numpy._core._multiarray_umath._get_castingimpl")

#: The "scaled float" demo DType is defined in numpy's C test-support code and
#: has no port equivalent.
_get_sfloat_dtype = not_implemented(
    "numpy._core._multiarray_umath._get_sfloat_dtype")


def __getattr__(name):
    if name.startswith("_") and not name.startswith("__"):
        return not_implemented(f"numpy._core._multiarray_umath.{name}")
    raise AttributeError(
        f"module 'numpy._core._multiarray_umath' has no attribute {name!r}")
