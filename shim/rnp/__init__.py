"""rnp — the NumPy-compatible surface of the Rust engine, under its own name.

``import rnp as np`` is the supported spelling for using the port directly:
the module presents exactly the surface of ``rnp_numpy`` (which itself mirrors
``numpy``), submodules included, so any code written against ``numpy`` runs
unchanged with only the import swapped.
"""

import importlib as _importlib
import sys as _sys

import rnp_numpy as _base

# Adopt the full top-level namespace.
_this = _sys.modules[__name__]
for _name in dir(_base):
    if not _name.startswith("__"):
        setattr(_this, _name, getattr(_base, _name))
__version__ = _base.__version__
__all__ = getattr(_base, "__all__", [n for n in dir(_base) if not n.startswith("_")])

# Submodules already imported by rnp_numpy's own init become importable as
# ``rnp.<name>`` immediately; anything else resolves lazily via __getattr__.
for _full, _mod in list(_sys.modules.items()):
    if _full.startswith("rnp_numpy.") and _mod is not None:
        _sys.modules["rnp." + _full[len("rnp_numpy."):]] = _mod


def __getattr__(name):
    try:
        return getattr(_base, name)
    except AttributeError:
        pass
    mod = _importlib.import_module(f"rnp_numpy.{name}")
    _sys.modules[f"rnp.{name}"] = mod
    return mod


def __dir__():
    return sorted(set(__all__) | set(dir(_base)))
