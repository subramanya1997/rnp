"""Stand-ins for the numpy internals the ported pure-Python modules import.

The upstream files under `numpy/lib/` and `numpy/polynomial/` are ported into
this shim verbatim apart from their import lines (see `port.py`'s rewrite
table, echoed in each ported file's header).  Everything those files reach for
that lives in a C extension module (`numpy._core._multiarray_umath`) or in
numpy's `__array_function__` dispatch machinery (`numpy._core.overrides`) is
provided here instead.

The NEP 18 dispatch layer is implemented in Python; numeric work continues to
be delegated to the Rust engine.
"""
import collections
import functools
import inspect
import operator
import types

import rnp_numpy as np

__all__ = [
    "set_module", "array_function_dispatch", "finalize_array_function_like",
    "normalize_axis_index", "normalize_axis_tuple", "asbytes", "asunicode",
    "add_docstring", "tracemalloc_domain", "ARRAY_FUNCTION_ENABLED",
]

#: The shim implements NEP 18 dispatch in Python.
ARRAY_FUNCTION_ENABLED = True
ARRAY_FUNCTIONS = set()

array_function_like_doc = """like : array_like, optional
    Reference object to allow the creation of arrays which are not NumPy
    arrays. If an array-like passed in as ``like`` supports the
    ``__array_function__`` protocol, the result will be defined by it."""

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
# Pure-Python equivalent of NumPy's `_ArrayFunctionDispatcher`.
# ---------------------------------------------------------------------------

def _get_implementing_args(relevant_args):
    """Collect one argument per overriding type, subclasses first."""
    implementing_args = []
    implementing_types = []
    for argument in relevant_args:
        argument_type = type(argument)
        if getattr(argument_type, "__array_function__", None) is None:
            continue
        if argument_type in implementing_types:
            continue
        if len(implementing_types) >= 64:
            raise TypeError(
                "maximum number (64) of distinct argument types implementing "
                "__array_function__ exceeded")
        insert_at = len(implementing_types)
        for i, old_type in enumerate(implementing_types):
            if issubclass(argument_type, old_type):
                insert_at = i
                break
        implementing_types.insert(insert_at, argument_type)
        implementing_args.insert(insert_at, argument)
    return implementing_args


def _restore_dispatcher(module_name, name):
    import importlib
    return getattr(importlib.import_module(module_name), name)


class _ArrayFunctionDispatcher:
    """Callable descriptor wrapping an implementation and its dispatcher."""

    def __init__(self, dispatcher, implementation):
        self._dispatcher = dispatcher
        self._implementation = implementation
        functools.update_wrapper(self, implementation)

    def __call__(self, *args, **kwargs):
        try:
            inspect.signature(self._implementation).bind(*args, **kwargs)
        except TypeError:
            # Calling the implementation reproduces CPython's function-name
            # prefix exactly; Signature.bind omits it.
            return self._implementation(*args, **kwargs)
        except (ValueError, AttributeError):
            pass
        relevant_args = self._dispatcher(*args, **kwargs)
        implementing_args = _get_implementing_args(relevant_args)
        if not implementing_args:
            return self._implementation(*args, **kwargs)

        # Base rnp ndarrays use the default implementation.  This preserves
        # the normal-call fast path while duck arrays take protocol dispatch.
        if all(type(arg) is np.ndarray for arg in implementing_args):
            return self._implementation(*args, **kwargs)

        types_ = tuple(type(arg) for arg in implementing_args)
        for argument in implementing_args:
            method = type(argument).__array_function__
            result = method(argument, self, types_, args, kwargs)
            if result is not NotImplemented:
                return result
        type_list = ", ".join(t.__name__ for t in types_)
        raise TypeError(
            f"no implementation found for '{self.__module__}.{self.__name__}' "
            "on types that implement __array_function__: "
            f"[{type_list}]")

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return types.MethodType(self, obj)

    def __repr__(self):
        return repr(self._implementation)

    def __str__(self):
        return str(self._implementation)

    def __reduce__(self):
        return _restore_dispatcher, (self.__module__, self.__name__)


ArgSpec = collections.namedtuple("ArgSpec", "args varargs keywords defaults")


def _argspec(func):
    spec = inspect.getfullargspec(func)
    return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)


def verify_matching_signatures(implementation, dispatcher):
    implementation_spec = _argspec(implementation)
    dispatcher_spec = _argspec(dispatcher)
    if (implementation_spec.args != dispatcher_spec.args or
            implementation_spec.varargs != dispatcher_spec.varargs or
            implementation_spec.keywords != dispatcher_spec.keywords or
            bool(implementation_spec.defaults) != bool(dispatcher_spec.defaults) or
            (implementation_spec.defaults is not None and
             len(implementation_spec.defaults) != len(dispatcher_spec.defaults))):
        raise RuntimeError(
            f"implementation and dispatcher for {implementation} have "
            "different function signatures")
    if (implementation_spec.defaults is not None and
            dispatcher_spec.defaults != (None,) * len(dispatcher_spec.defaults)):
        raise RuntimeError(
            "dispatcher functions can only use None for default argument values")

def array_function_dispatch(dispatcher=None, module=None, verify=True,
                            docs_from_dispatcher=False):
    def decorator(implementation):
        if dispatcher is None:
            if verify:
                co = implementation.__code__
                last_arg = co.co_argcount + co.co_kwonlyargcount - 1
                if (co.co_kwonlyargcount == 0 or
                        co.co_varnames[last_arg] != "like"):
                    raise RuntimeError(
                        "__array_function__ expects `like=` to be the last "
                        "argument and a keyword-only argument. "
                        f"{implementation} does not seem to comply.")
            # NumPy's no-dispatcher form is an internal `like=` building
            # block.  It expects the reference object as an extra positional
            # argument and deliberately errors when called as a public API.
            def like_dispatch(like, /, *args, **kwargs):
                return (like,)
            public_api = _ArrayFunctionDispatcher(like_dispatch, implementation)
        else:
            if verify:
                verify_matching_signatures(implementation, dispatcher)
            if docs_from_dispatcher and dispatcher.__doc__ is not None:
                implementation.__doc__ = inspect.cleandoc(dispatcher.__doc__)
            public_api = _ArrayFunctionDispatcher(dispatcher, implementation)
        if module is not None:
            public_api.__module__ = module
        ARRAY_FUNCTIONS.add(public_api)
        return public_api
    return decorator


def finalize_array_function_like(public_api):
    ARRAY_FUNCTIONS.add(public_api)
    public_api.__doc__ = get_array_function_like_doc(public_api)
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
    public_api = a[0] if a else None
    if public_api is not None:
        ARRAY_FUNCTIONS.add(public_api)
        doc = public_api.__doc__ or (a[1] if len(a) > 1 else "")
        return doc.replace("${ARRAY_FUNCTION_LIKE}", array_function_like_doc)
    return array_function_like_doc


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


# `_polynomial_impl` imports these at module load time, before the top-level
# package has finished wiring its linalg namespace. Resolve them only when a
# legacy polynomial operation actually calls one; the backed linalg wrappers
# are fully available by then.
def _linalg_function(name):
    def call(*args, **kwargs):
        from rnp_numpy import linalg
        return getattr(linalg, name)(*args, **kwargs)
    call.__name__ = name
    return call


eigvals = _linalg_function("eigvals")
inv = _linalg_function("inv")
lstsq = _linalg_function("lstsq")


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
