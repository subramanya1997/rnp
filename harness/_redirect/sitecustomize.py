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
