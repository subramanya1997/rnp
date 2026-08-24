"""`isclose` / `allclose` / `array_equal` / `array_equiv`, ported from
`numpy/_core/numeric.py`.

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

__all__ = ["isclose", "allclose", "array_equal", "array_equiv"]


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


# --------------------------------------------------------------------------
# `array_equal` / `array_equiv`, ported statement for statement from
# `numpy/_core/numeric.py`.
#
# The only substitution is `array_equiv`'s shape test: upstream builds a
# `multiarray.broadcast` object purely to find out whether the two operands
# broadcast at all, and the port has no `broadcast` object yet, so the same
# question is asked of `broadcast_shapes`.
# --------------------------------------------------------------------------

#: The dtypes numpy considers incapable of holding a NaN, letting
#: `array_equal(..., equal_nan=True)` skip the isnan pass. Upstream spells
#: this as a set of dtype *classes* (`type(np.dtype(nt.int8))` and friends,
#: which in numpy are distinct `numpy.dtypes.*DType` subclasses) and lists
#: exactly bool plus the four signed integer widths -- not the unsigned ones,
#: which is a quirk of upstream's list, not of the arithmetic.
#:
#: The port has a single `dtype` class for every dtype, so `type(dt)` cannot
#: discriminate; the same set is spelled by name instead. Since this only
#: selects between two routes to the same answer, matching upstream's exact
#: membership (unsigned excluded) keeps the behaviour identical either way.
_NO_NAN_DTYPE_NAMES = frozenset(
    ["bool", "int8", "int16", "int32", "int64"])


def _dtype_cannot_hold_nan(dtype):
    return getattr(dtype, "name", None) in _NO_NAN_DTYPE_NAMES


def array_equal(a1, a2, equal_nan=False):
    """True if two arrays have the same shape and elements."""
    from .. import isnan
    try:
        a1, a2 = _asanyarray(a1), _asanyarray(a2)
    except Exception:  # noqa: BLE001
        return False
    if a1.shape != a2.shape:
        return False
    if not equal_nan:
        return builtins.bool((_asanyarray(a1 == a2)).all())

    if a1 is a2:
        # nan will compare equal so an array will compare equal to itself.
        return True

    cannot_have_nan = (_dtype_cannot_hold_nan(a1.dtype)
                       and _dtype_cannot_hold_nan(a2.dtype))
    if cannot_have_nan:
        return builtins.bool(_asanyarray(a1 == a2).all())

    # Fast path for a1 and a2 being all NaN arrays.
    a1nan = isnan(a1)
    if a1nan.all():
        return builtins.bool(isnan(a2).all())

    equal_or_both_nan = (a1 == a2) | (a1nan & isnan(a2))
    return builtins.bool(equal_or_both_nan.all())


def array_equiv(a1, a2):
    """True if the inputs broadcast together and are equal everywhere."""
    try:
        a1, a2 = _asanyarray(a1), _asanyarray(a2)
    except Exception:  # noqa: BLE001
        return False
    try:
        _rnp.broadcast_shapes(a1.shape, a2.shape)
    except Exception:  # noqa: BLE001
        return False

    return builtins.bool(_asanyarray(a1 == a2).all())
