"""
``numpy.lib`` is mostly a space for implementing functions that don't
belong in core or in another NumPy submodule with a clear purpose
(e.g. ``random``, ``fft``, ``linalg``, ``ma``).

``numpy.lib``'s private submodules contain basic functions that are used by
other public modules and are useful to have in the main name-space.

--- rnp note ---------------------------------------------------------------

Ported from upstream `numpy/lib/__init__.py`, with two adaptations:

*   `add_docstring` / `tracemalloc_domain` / `add_newdoc` come from
    `._rnp_compat` rather than from the C extension module.

*   **Submodule import is lazy.**  Upstream imports its private submodules
    eagerly at package import.  Here it cannot: `rnp_numpy/__init__.py` pulls
    names *out of* these submodules while it is still executing, and those
    submodules in turn do `from rnp_numpy import ...` at module scope, so an
    eager import would deadlock on the partially-initialised package.  A
    module-level `__getattr__` imports each submodule on first access, which
    is also what keeps one broken submodule from taking down `import numpy`
    for the whole suite.  Anything that failed to import is recorded in
    `_import_errors`.
"""

import importlib as _importlib

from ._rnp_compat import add_docstring, tracemalloc_domain  # noqa: F401

#: {submodule name: exception} for anything that failed to import.
_import_errors = {}

_SUBMODULES = frozenset({
    "_rnp_compat",
    "_rnp_corecompat",
    "_array_utils_impl",
    "array_utils",
    "_version",
    "_utils_impl",
    "mixins",
    "introspect",
    "_type_check_impl",
    "_ufunclike_impl",
    "_stride_tricks_impl",
    "stride_tricks",
    "_twodim_base_impl",
    "_shape_base_impl",
    "_index_tricks_impl",
    "_arraysetops_impl",
    "_function_base_impl",
    "_histograms_impl",
    "_nanfunctions_impl",
    "_arraypad_impl",
    "_arrayterator_impl",
    "_scimath_impl",
    "scimath",
    "_polynomial_impl",
    "_format_impl",
    "format",
    "_iotools",
    "_datasource",
    "_npyio_impl",
    "npyio",
    "_user_array_impl",
    "user_array",
})

#: Names `numpy.lib` itself exports, and the submodule each comes from.
_MEMBERS = {
    "Arrayterator": "_arrayterator_impl",
    "NumpyVersion": "_version",
}


def add_newdoc(place, obj, doc, warn_on_python=True):
    """`numpy._core.function_base.add_newdoc`; a no-op for the port."""
    return None


add_newdoc.__module__ = "numpy.lib"

__all__ = [
    "Arrayterator", "add_docstring", "add_newdoc", "array_utils",
    "format", "introspect", "mixins", "NumpyVersion", "npyio", "scimath",
    "stride_tricks", "tracemalloc_domain",
]

test = None


def __getattr__(attr):
    if attr in _SUBMODULES:
        try:
            mod = _importlib.import_module(f".{attr}", __name__)
        except Exception as exc:
            _import_errors[attr] = exc
            raise AttributeError(
                f"numpy.lib.{attr} failed to import in this port: {exc!r}"
            ) from exc
        globals()[attr] = mod
        return mod

    if attr in _MEMBERS:
        mod = __getattr__(_MEMBERS[attr])
        value = getattr(mod, attr)
        globals()[attr] = value
        return value

    # Warn for deprecated/removed aliases
    if attr == "emath":
        raise AttributeError(
            "numpy.lib.emath was an alias for emath module that was removed "
            "in NumPy 2.0. Replace usages of numpy.lib.emath with "
            "numpy.emath.",
            name=None
        )
    elif attr in (
        "histograms", "type_check", "nanfunctions", "function_base",
        "arraypad", "arraysetops", "ufunclike", "utils", "twodim_base",
        "shape_base", "polynomial", "index_tricks",
    ):
        raise AttributeError(
            f"numpy.lib.{attr} is now private. If you are using a public "
            "function, it should be available in the main numpy namespace, "
            "otherwise check the NumPy 2.0 migration guide.",
            name=None
        )
    elif attr == "arrayterator":
        raise AttributeError(
            "numpy.lib.arrayterator submodule is now private. To access "
            "Arrayterator class use numpy.lib.Arrayterator.",
            name=None
        )
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
