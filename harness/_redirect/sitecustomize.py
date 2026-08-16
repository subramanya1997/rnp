"""Import redirection for the upstream-test subprocess.

Installed via PYTHONPATH by harness/run.py. Makes `import numpy` (and any
`numpy.*` submodule import) resolve to the rnp_numpy shim package, so the
upstream test files run against the Rust port WITHOUT any modification.
"""
import importlib
import sys


class _NumpyRedirector:
    """Meta-path finder mapping the 'numpy' namespace onto 'rnp_numpy'."""

    def find_module(self, fullname, path=None):  # legacy protocol, pytest-safe
        if fullname == "numpy" or fullname.startswith("numpy."):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        target = "rnp_numpy" + fullname[len("numpy"):]
        mod = importlib.import_module(target)
        sys.modules[fullname] = mod
        return mod


sys.meta_path.insert(0, _NumpyRedirector())
