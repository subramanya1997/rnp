"""Stand-in for numpy._core._rational_tests (a C user-dtype example).

User-defined dtypes are far beyond M1. These names exist so that test modules
importing them can be *collected*; each carries a placeholder `dtype` (an
opaque 8-byte void) so that module-level `np.dtype(rational)` succeeds, and
any test that actually exercises rational arithmetic fails loudly.
"""
from .. import dtype as _dtype


class _Unavailable:
    #: Placeholder descriptor so `np.dtype(cls)` resolves during collection.
    dtype = _dtype("V8")

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "the rational user dtype is not implemented by rnp")


class rational(_Unavailable):
    pass


class rational2(_Unavailable):
    pass
