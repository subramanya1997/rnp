"""Import redirection for the upstream-test subprocess.

Installed via PYTHONPATH by harness/run.py. Makes `import numpy` (and any
`numpy.*` submodule import) resolve to the rnp_numpy shim package, so the
upstream test files run against the Rust port WITHOUT any modification.

The finder implements `find_spec` (the `find_module`/`load_module` protocol
was removed in Python 3.12), and returns the *same module object* for
`numpy.X` and `rnp_numpy.X` so identity checks keep working.
"""
import importlib
import importlib.abc
import importlib.util
import sys

_PREFIX = "numpy"
_TARGET = "rnp_numpy"


class _ShimLoader(importlib.abc.Loader):
    def __init__(self, target_name):
        self.target_name = target_name

    def create_module(self, spec):
        # Raises ModuleNotFoundError when the shim does not provide this
        # submodule yet, which is the honest answer for the tests.
        return importlib.import_module(self.target_name)

    def exec_module(self, module):
        # Already executed as part of importing the shim module.
        pass


class _NumpyRedirector(importlib.abc.MetaPathFinder):
    """Meta-path finder mapping the 'numpy' namespace onto 'rnp_numpy'."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _PREFIX and not fullname.startswith(_PREFIX + "."):
            return None
        target_name = _TARGET + fullname[len(_PREFIX):]
        spec = importlib.util.spec_from_loader(
            fullname, _ShimLoader(target_name), is_package=True
        )
        return spec


if not any(isinstance(f, _NumpyRedirector) for f in sys.meta_path):
    sys.meta_path.insert(0, _NumpyRedirector())


# ---------------------------------------------------------------------------
# Bind `numpy` in sys.modules eagerly.
#
# The meta-path finder above is enough for `_core`, whose test directory is not
# a package: pytest imports those files under a bare module name and every
# `import numpy` inside them goes through the finder.
#
# `lib/`, `ma/` and `polynomial/` tests DO have an `__init__.py`, so pytest
# resolves e.g. `polynomial/tests/test_polyutils.py` to the dotted name
# `numpy.polynomial.tests.test_polyutils`.  In `--import-mode=importlib`,
# pytest materialises each *parent* of that name itself
# (`_pytest.pathlib._import_module_using_spec`): for a parent not already in
# `sys.modules` it builds a spec with `spec_from_file_location` against the
# `__init__.py` it found on disk, which never consults `sys.meta_path`.  The
# real `upstream/numpy/__init__.py` therefore gets executed, dies on its first
# `from . import version`, and the whole file is scored zero as a collection
# error.
#
# pytest checks `sys.modules.get(parent_module_name)` first, so binding the
# alias here — before pytest starts — makes it reuse the shim instead.  This
# has to happen at interpreter startup; doing it at the end of the shim's own
# `__init__` is too late, because by then pytest has already begun importing
# the upstream package.
# ---------------------------------------------------------------------------

if _PREFIX not in sys.modules:
    sys.modules[_PREFIX] = importlib.import_module(_TARGET)
