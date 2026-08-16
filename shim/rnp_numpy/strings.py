"""``numpy.strings`` -- re-export of the port's pure-Python implementation."""

from rnp_numpy._core.strings import *  # noqa: F401,F403
from rnp_numpy._core.strings import __all__  # noqa: F401
from rnp_numpy._core.strings import (  # noqa: F401
    _join,
    _rsplit,
    _split,
    _splitlines,
)
