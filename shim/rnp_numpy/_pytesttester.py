"""Compatibility wrapper for packages that expose a ``test`` callable."""

__all__ = ["PytestTester"]


class PytestTester:
    def __init__(self, module_name):
        self.module_name = module_name

    def __call__(self, *args, **kwargs):
        return self.test(*args, **kwargs)

    def test(self, *args, **kwargs):
        raise NotImplementedError(
            "in-process numpy.test() is unavailable; use harness/run.py")
