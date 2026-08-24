"""numpy.testing equivalent, backed by our own minimal assertion helpers."""

from ._private.utils import *  # noqa: F401,F403
from ._private.utils import __all__ as _utils_all
from ._private import extbuild, utils  # noqa: F401

from unittest import SkipTest, TestCase  # noqa: F401

from . import overrides  # noqa: F401

__all__ = list(_utils_all) + ["SkipTest", "TestCase", "overrides"]


def test(*args, **kwargs):
    raise NotImplementedError("numpy.testing.test() is not supported")


def run_module_suite(file_to_run=None, argv=None):
    raise NotImplementedError("run_module_suite is not supported")
