"""
=============
Masked Arrays
=============

`numpy.ma`, ported from upstream.  See `.core` and `.extras`, which are
upstream's own source with only their import lines rewritten (each file's
header lists the rewrites).

--- rnp note ---------------------------------------------------------------

Upstream's `__init__` is ``from .core import *`` / ``from .extras import *``
at module scope.  This port loads both **lazily and defensively** instead,
for two reasons:

*   *Ordering.*  `rnp_numpy/__init__.py` imports this package while it is
    still executing, and `ma.core` does ``from rnp_numpy import angle,
    expand_dims, iscomplexobj, _NoValue`` — names bound later in that file.
    An eager import here fails on the partially-initialised package.

*   *Blast radius.*  `ma.core` is 9k lines standing on a lot of engine
    surface.  If it raises on import, an eager ``from .core import *`` would
    propagate out of ``import numpy`` itself and score **every** test file in
    the whole suite zero.  Loading on first attribute access confines the
    damage to `numpy.ma` users, and the fallback table below keeps even those
    collecting: a test module that does ``from numpy.ma import masked_array``
    still imports, and fails at the point of use rather than at collection.

`_import_error` holds the exception when the real modules could not be
loaded, so a fallback name is always diagnosable.
"""

from .._stubs import not_implemented

#: Set when `.core`/`.extras` fail to import; None while they are fine.
_import_error = None

#: Minimal surface kept alive when the real modules cannot load.  These are
#: the names upstream test modules and `numpy.lib` touch at *import* time.
_FALLBACK = {
    "MaskedArray": lambda: not_implemented("numpy.ma.MaskedArray"),
    "masked_array": lambda: not_implemented("numpy.ma.masked_array"),
    "array": lambda: not_implemented("numpy.ma.array"),
    "masked": lambda: None,
    "nomask": lambda: False,
    "getmask": lambda: not_implemented("numpy.ma.getmask"),
    "getmaskarray": lambda: not_implemented("numpy.ma.getmaskarray"),
    "masked_equal": lambda: not_implemented("numpy.ma.masked_equal"),
    "masked_where": lambda: not_implemented("numpy.ma.masked_where"),
    "filled": lambda: not_implemented("numpy.ma.filled"),
    "is_masked": lambda: (lambda x: False),
    "is_mask": lambda: (lambda m: False),
}

_loaded = False


def _load():
    """Import `.core` and `.extras` and hoist their exports up here."""
    global _loaded, _import_error
    if _loaded:
        return _import_error is None
    _loaded = True
    try:
        from . import core, extras
    except Exception as exc:  # pragma: no cover - diagnostic path
        _import_error = exc
        return False
    g = globals()
    g["core"] = core
    g["extras"] = extras
    names = ["core", "extras"]
    for mod in (core, extras):
        for name in getattr(mod, "__all__", ()):
            try:
                g[name] = getattr(mod, name)
            except AttributeError:
                continue
            names.append(name)
    g["__all__"] = names
    return True


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    if _load():
        try:
            return globals()[name]
        except KeyError:
            pass
        # Submodules `_load` does not pull in (mrecords, testutils).
        import importlib
        try:
            mod = importlib.import_module(f".{name}", __name__)
        except ImportError:
            pass
        else:
            globals()[name] = mod
            return mod
    if name in _FALLBACK:
        value = _FALLBACK[name]()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    _load()
    return sorted(set(globals()) | set(_FALLBACK))
