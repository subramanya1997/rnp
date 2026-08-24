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
# Make the *packaged* upstream test suites resolvable.
#
# `upstream/numpy/_core/tests/` has no `__init__.py`, so pytest imports those
# files under a bare module name and every `import numpy` inside them goes
# through the finder above. But `lib/`, `linalg/`, `fft/`, `random/`, `ma/`,
# `polynomial/` and `matrixlib/` test directories DO have one, so pytest
# resolves e.g. `fft/tests/test_helper.py` to the dotted name
# `numpy.fft.tests.test_helper` and imports its parent packages first.
#
# Two distinct things go wrong there, and both have to be fixed or the whole
# file scores zero as a collection error:
#
#  1. In `--import-mode=importlib`, pytest materialises each *parent* of that
#     dotted name itself (`_pytest.pathlib._import_module_using_spec`): for a
#     parent not already in `sys.modules` it builds a spec with
#     `spec_from_file_location` against the `__init__.py` it found on disk,
#     which never consults `sys.meta_path`. The REAL
#     `upstream/numpy/__init__.py` therefore executes and dies on its first
#     `from . import version`. pytest checks `sys.modules` first, so binding
#     the alias here -- before pytest starts -- makes it reuse the shim. This
#     has to happen at interpreter startup; doing it at the end of the shim's
#     own `__init__` is too late.
#
#  2. The parent packages then have to be importable at all. `numpy.fft.tests`
#     has no shim counterpart and never will -- it is a test directory, not
#     API -- so the finder's honest ModuleNotFoundError is the wrong answer
#     here. Synthesise an empty package for any `numpy.<sub>.tests[...]` name
#     the shim does not provide. pytest still loads the test module from the
#     upstream file path (verified: nodeids come back as
#     `upstream/numpy/fft/tests/test_helper.py::...`), so this changes which
#     tests can be COLLECTED, never what they assert or how they are scored.
# ---------------------------------------------------------------------------

class _TestPkgLoader(importlib.abc.Loader):
    """Loader for `numpy.<sub>.tests` packages the shim has no analogue for."""

    def create_module(self, spec):
        return None  # default module creation

    def exec_module(self, module):
        module.__path__ = []


class _TestPkgShim(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        if parts[0] != _PREFIX or "tests" not in parts[1:]:
            return None
        # Only step in when the shim genuinely has nothing to offer, so a real
        # shim module always wins.
        target_name = _TARGET + fullname[len(_PREFIX):]
        try:
            if importlib.util.find_spec(target_name) is not None:
                return None
        except (ImportError, AttributeError, ValueError):
            pass
        return importlib.util.spec_from_loader(
            fullname, _TestPkgLoader(), is_package=True
        )


if not any(isinstance(f, _TestPkgShim) for f in sys.meta_path):
    # After the redirector, so the shim keeps priority.
    sys.meta_path.insert(1, _TestPkgShim())

if _PREFIX not in sys.modules:
    sys.modules[_PREFIX] = importlib.import_module(_TARGET)
