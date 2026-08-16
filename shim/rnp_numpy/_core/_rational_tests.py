"""Stand-in for numpy._core._rational_tests (a C user-dtype example).

User-defined dtypes are far beyond M0; these names exist only so that test
modules importing them can be collected.
"""


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "the rational user dtype is not implemented by rnp")


class rational(_Unavailable):
    pass


class rational2(_Unavailable):
    pass
