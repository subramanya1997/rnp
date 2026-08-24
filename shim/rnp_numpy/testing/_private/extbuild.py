"""Small compatible surface for NumPy's test-only extension builder.

The rnp harness never builds NumPy's C test extensions.  Keeping real module
functions here lets tests import and introspect the helper; a call fails at
the point where it would cross the shim-only lane into a native build.
"""

__all__ = ["build_and_import_extension", "compile_extension_module"]


def _unsupported():
    raise NotImplementedError(
        "building NumPy C test extensions is unavailable in the rnp shim")


def build_and_import_extension(
        modname, functions, *, prologue="", build_dir=None,
        include_dirs=None, more_init=""):
    _unsupported()


def compile_extension_module(
        name, builddir, include_dirs, source_string, libraries=None,
        library_dirs=None):
    _unsupported()
