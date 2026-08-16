"""numpy._core equivalent: the private namespace upstream tests import from."""

from .. import (  # noqa: F401
    add,
    arange,
    array,
    asarray,
    divide,
    dtype,
    empty,
    equal,
    full,
    greater,
    greater_equal,
    less,
    less_equal,
    multiply,
    ndarray,
    newaxis,
    not_equal,
    ones,
    promote_types,
    result_type,
    subtract,
    zeros,
)
from . import multiarray, numerictypes, shape_base, umath  # noqa: F401
from . import _internal, _umath_tests  # noqa: F401
from .numerictypes import sctypes  # noqa: F401
from .shape_base import (  # noqa: F401
    atleast_1d,
    atleast_2d,
    atleast_3d,
    block,
    concatenate,
    hstack,
    stack,
    vstack,
)
