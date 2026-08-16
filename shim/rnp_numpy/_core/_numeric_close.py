"""`isclose` / `allclose`, ported from `numpy/_core/numeric.py`.

These are pure Python in numpy — they are built entirely out of ufuncs the
engine already provides (`abs`, `less_equal`, `isfinite`, `isnan`, `equal`,
`logical_and`, `logical_or`) — so they belong in the shim rather than in Rust.

The structure below deliberately mirrors upstream statement for statement,
including the `result_type(y, 1.)` promotion that keeps `abs(MIN_INT)` from
wrapping, the `errstate(invalid='ignore')` around the comparison, and the
`result[()]` unwrap that turns a 0-d result into a scalar.
"""
import builtins
import warnings

import _rnp

from .. import _errstate as _errstate_mod

__all__ = ["isclose", "allclose"]


def _asanyarray(a, dtype=None):
    from .. import asarray
    if dtype is None:
        return asarray(a)
    return asarray(a, dtype=dtype)


def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """Return a boolean array where `a` and `b` are equal within a tolerance."""
    from .. import (
        abs as _abs, all as _all, errstate, geterr, isfinite, isnan,
        less_equal, result_type,
    )

    # Turn all but python scalars into arrays.
    x, y, atol, rtol = (
        v if isinstance(v, (int, float, complex)) else _asanyarray(v)
        for v in (a, b, atol, rtol)
    )

    # Make sure `y` is an inexact type, to avoid bad behaviour on
    # `abs(MIN_INT)`.  This also drives the casting of `x` further down.
    dtype = getattr(y, "dtype", None)
    if dtype is not None and dtype.kind != "m":
        dt = result_type(y, 1.)
        y = _asanyarray(y, dtype=dt)
    elif isinstance(y, int):
        y = float(y)

    # `atol` and `rtol` may themselves be arrays.
    if not (_all(isfinite(atol)) and _all(isfinite(rtol))):
        err_s = geterr()["invalid"]
        err_msg = (
            f"One of rtol or atol is not valid, atol: {atol}, rtol: {rtol}"
        )
        if err_s == "warn":
            warnings.warn(err_msg, RuntimeWarning, stacklevel=2)
        elif err_s == "raise":
            raise FloatingPointError(err_msg)
        elif err_s == "print":
            print(err_msg)

    with errstate(invalid="ignore"):
        result = (
            less_equal(_abs(x - y), atol + rtol * _abs(y))
            & isfinite(y)
            | (x == y)
        )
        if equal_nan:
            result = result | (isnan(x) & isnan(y))

    # Flatten 0-d results to scalars, exactly as numpy does.
    try:
        return result[()]
    except (TypeError, IndexError):
        return result


def allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """Return True if two arrays are elementwise equal within a tolerance."""
    from .. import all as _all

    res = _all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))
    return builtins.bool(res)
