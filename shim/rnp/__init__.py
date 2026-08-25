"""rnp — the NumPy-compatible surface of the Rust engine, under its own name.

``import rnp as np`` is the supported spelling for using the port directly:
the module presents exactly the surface of ``rnp_numpy`` (which itself mirrors
``numpy``), submodules included, so any code written against ``numpy`` runs
unchanged with only the import swapped.
"""

import importlib as _importlib
import importlib.abc as _abc
import importlib.util as _ilu
import sys as _sys

# The shim's internals (modules ported nearly verbatim from numpy) import
# `numpy.*` by absolute name, so importing rnp claims the `numpy` name for
# this process: a meta-path finder maps the whole `numpy` namespace onto
# `rnp_numpy`, returning the same module objects so identity checks hold.
# Mixing engines in one process is not supported — fail loudly rather than
# hand back a chimera of real-numpy and rnp arrays.
if "numpy" in _sys.modules and not getattr(
        _sys.modules["numpy"], "__rnp__", False):
    raise ImportError(
        "cannot import rnp after real numpy is already imported: rnp takes "
        "over the 'numpy' module namespace for the whole process. Import rnp "
        "first (or drop the real-numpy import).")


class _Redirector(_abc.MetaPathFinder, _abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        # `numpy` and `numpy.*` — and `rnp.*` submodule imports — all map
        # onto the corresponding rnp_numpy module (same object, so identity
        # checks hold across spellings).
        if fullname == "numpy" or fullname.startswith("numpy."):
            return _ilu.spec_from_loader(fullname, self)
        if fullname.startswith("rnp."):
            return _ilu.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        stem = spec.name.split(".", 1)
        return _importlib.import_module(
            "rnp_numpy" + ("." + stem[1] if len(stem) > 1 else ""))

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Redirector) for f in _sys.meta_path):
    _sys.meta_path.insert(0, _Redirector())

import rnp_numpy as _base

_base.__rnp__ = True
_sys.modules.setdefault("numpy", _base)

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
