"""Stand-ins for the numpy internals the ported pure-Python modules import.

The upstream files under `numpy/lib/` and `numpy/polynomial/` are ported into
this shim verbatim apart from their import lines (see `port.py`'s rewrite
table, echoed in each ported file's header).  Everything those files reach for
that lives in a C extension module (`numpy._core._multiarray_umath`) or in
numpy's `__array_function__` dispatch machinery (`numpy._core.overrides`) is
provided here instead.

Nothing in this module is a numpy *behaviour* re-implementation: the dispatch
decorators are no-ops (the port has no `__array_function__` protocol, so a
plain function is the correct degenerate form), and the axis normalisers are
upstream's own algorithm.
"""
import functools
import operator

import rnp_numpy as np

__all__ = [
    "set_module", "array_function_dispatch", "finalize_array_function_like",
    "normalize_axis_index", "normalize_axis_tuple", "asbytes", "asunicode",
    "add_docstring", "tracemalloc_domain", "ARRAY_FUNCTION_ENABLED",
]

#: numpy exposes this so tests can tell whether dispatch is compiled in.  The
#: port has no dispatch machinery at all, which is what `False` means here.
ARRAY_FUNCTION_ENABLED = False

tracemalloc_domain = 389047


# ---------------------------------------------------------------------------
# numpy._utils
# ---------------------------------------------------------------------------

def set_module(module):
    """Upstream `numpy._utils.set_module`, verbatim."""
    def decorator(func):
        if module is not None:
            if isinstance(func, type):
                try:
                    func._module_source = func.__module__
                except AttributeError:
                    pass
            func.__module__ = module
        return func
    return decorator


def asunicode(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')


# ---------------------------------------------------------------------------
# numpy._core.overrides
#
# The port implements no `__array_function__` protocol, so dispatch collapses
# to "call the implementation".  `array_function_dispatch` keeps upstream's
# signature (it is used both bare and with a dispatcher argument) and keeps
# the `_implementation` attribute the tests introspect.
# ---------------------------------------------------------------------------

def array_function_dispatch(dispatcher=None, module=None, verify=True,
                            docs_from_dispatcher=False):
    def decorator(implementation):
        if module is not None:
            implementation.__module__ = module
        implementation._implementation = implementation
        implementation.__wrapped__ = implementation
        return implementation
    return decorator


def finalize_array_function_like(public_api):
    return public_api


def set_array_function_like_doc(public_api):
    return public_api


def array_function_from_dispatcher(implementation, module=None,
                                   verify=True, docs_from_dispatcher=True):
    def decorator(dispatcher):
        return array_function_dispatch(
            dispatcher, module, verify=verify,
            docs_from_dispatcher=docs_from_dispatcher)(implementation)
    return decorator


def get_array_function_like_doc(*a, **k):
    return ""


def verify_matching_signatures(*a, **k):
    return None


# ---------------------------------------------------------------------------
# numpy._core.numeric axis helpers (upstream algorithm, upstream messages)
# ---------------------------------------------------------------------------

def normalize_axis_index(axis, ndim, msg_prefix=None):
    try:
        axis = operator.index(axis)
    except TypeError:
        raise TypeError(
            f"'{type(axis).__name__}' object cannot be interpreted as an "
            f"integer") from None
    if axis < -ndim or axis >= ndim:
        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        if msg_prefix:
            msg = f"{msg_prefix}: {msg}"
        raise np.exceptions.AxisError(axis, ndim, msg_prefix)
    if axis < 0:
        axis += ndim
    return axis


def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    if not isinstance(axis, (tuple, list)):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    axis = tuple(normalize_axis_index(ax, ndim, argname) for ax in axis)
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError(f'repeated axis in `{argname}` argument')
        else:
            raise ValueError('repeated axis')
    return axis


# ---------------------------------------------------------------------------
# numpy._core._multiarray_umath entry points reached from pure Python.
#
# These are the ones the ported modules import at module scope; each raises
# only when actually called, so a module that merely imports one still loads.
# ---------------------------------------------------------------------------

def add_docstring(obj, docstring):
    """numpy attaches docstrings to C objects with this; a no-op here."""
    try:
        obj.__doc__ = docstring
    except (AttributeError, TypeError):
        pass


def _not_available(name):
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"numpy._core._multiarray_umath.{name} is not implemented by rnp "
            f"yet")
    _fn.__name__ = name
    return _fn


_load_from_filelike = _not_available("_load_from_filelike")


def _unique_hash(ar, equal_nan=True, return_index=False,
                 return_inverse=False, return_counts=False):
    """numpy's hash-based `unique` fast path.

    Returning `NotImplemented` is the documented "this dtype has no hash
    path" answer, and is exactly what `_unique1d` tests for before falling
    back to its sort-based implementation.  The port always takes the sort
    path.
    """
    return NotImplemented
dragon4_positional = _not_available("dragon4_positional")
dragon4_scientific = _not_available("dragon4_scientific")


class _array_converter:
    """Minimal stand-in for the C `_array_converter` helper.

    Upstream uses it in `_type_check_impl` and `_function_base_impl` to
    convert a group of inputs together and then map results back to the
    caller's array-or-scalar-ness.  The port only needs the two operations
    those call sites use: iteration over the converted arrays, and
    `wrap`/`scalar_input`.
    """

    def __init__(self, *args):
        self.scalar_input = tuple(
            not isinstance(a, np.ndarray) and np.ndim(a) == 0 for a in args)
        self._arrays = tuple(np.asanyarray(a) for a in args)

    def __iter__(self):
        return iter(self._arrays)

    def __getitem__(self, i):
        return self._arrays[i]

    def __len__(self):
        return len(self._arrays)

    @property
    def result_type(self):
        return np.result_type(*self._arrays)

    def as_arrays(self, subok=True, pyscalars="convert_if_no_array"):
        return self._arrays

    def wrap(self, arr, to_scalar=None):
        if to_scalar is None:
            to_scalar = all(self.scalar_input)
        if to_scalar and np.ndim(arr) == 0:
            return arr[()] if isinstance(arr, np.ndarray) else arr
        return arr


# `numpy.linalg` is out of scope for this lane (it needs real LAPACK kernels).
# `_polynomial_impl` imports these three at module scope, so they have to
# exist as names or the whole module — and with it `np.poly1d` — fails to
# import.  Each raises only when called.
eigvals = _not_available("linalg.eigvals")
inv = _not_available("linalg.inv")
lstsq = _not_available("linalg.lstsq")


def _monotonicity(xp):
    """numpy's C `_monotonicity`: +1 strictly increasing, -1 strictly
    decreasing, 0 otherwise."""
    vals = np.asarray(xp).reshape(-1).tolist()
    if len(vals) < 2:
        return 1
    if all(b > a for a, b in zip(vals, vals[1:])):
        return 1
    if all(b < a for a, b in zip(vals, vals[1:])):
        return -1
    return 0


def _vec_string(*a, **k):
    raise NotImplementedError("_vec_string is not implemented by rnp yet")


# ---------------------------------------------------------------------------
# Fallback resolution.
#
# The ported modules import a long tail of names from `numpy._core.numeric`,
# `numpy._core.multiarray`, `numpy._core.umath` and `numpy._core` itself.  The
# port keeps all of those in one flat namespace (`rnp_numpy`) plus the
# `_core`-level fill-in in `._rnp_corecompat`, so rather than enumerate the
# tail here, every `numpy._core.*` import is rewritten to come from this
# module and anything not defined above is resolved against those two.
#
# Order matters: `_rnp_corecompat` first, so that a name it implements (e.g.
# `interp`, `bincount`) is preferred over a same-named stub in the top-level
# package.
# ---------------------------------------------------------------------------

#: Names resolved to a raise-on-call placeholder, in import order.
_missing = {}


def __getattr__(name):
    if not name.startswith("__"):
        from . import _rnp_corecompat
        from rnp_numpy._core import numerictypes, shape_base
        for source in (_rnp_corecompat, np, shape_base, numerictypes):
            try:
                return getattr(source, name)
            except AttributeError:
                continue
        # A `numpy._core` name nothing in the port provides.  Returning a
        # raise-on-call placeholder rather than raising here is deliberate:
        # these are almost all imported at module scope, and an ImportError
        # would cost the whole module (and every test in the file that
        # imports it) rather than just the calls that actually need the
        # missing primitive.
        stub = _not_available(name)
        _missing[name] = stub
        return stub
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Submodule aliases.
#
# Several ported modules import a `numpy._core` *submodule* rather than a name
# out of one (`from numpy._core import overrides, getlimits`, then
# `overrides.ARRAY_FUNCTION_ENABLED` / `getlimits.finfo`).  Since every
# `numpy._core.*` import is rewritten to this module, those submodules are
# bound here to this module itself; the `__getattr__` fallback above then
# resolves whatever attribute is read off them.
# ---------------------------------------------------------------------------

import sys as _sys

_self = _sys.modules[__name__]

overrides = _self
numeric = _self
numerictypes = _self
fromnumeric = _self
multiarray = _self
umath = _self
getlimits = _self
_multiarray_umath = _self


# `functools` re-export: several ported modules do
# `array_function_dispatch = functools.partial(array_function_dispatch, module=...)`
_ = functools
