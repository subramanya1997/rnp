"""datetime64 / timedelta64 helpers that live above the engine.

`datetime_data`, `isnat` and `datetime_as_string` are thin wrappers over the
Rust entry points; the business-day family is a documented gap that raises
`NotImplementedError` while still existing, so that code (and numpy's own test
module) can import it.
"""

import _rnp

ndarray = _rnp.ndarray
dtype = _rnp.dtype

datetime_data = _rnp.datetime_data


def isnat(x, /, out=None, **kwargs):
    """`np.isnat`: True exactly where a datetime64/timedelta64 is NaT."""
    r = _rnp.isnat(x)
    if out is not None:
        out[...] = r
        return out
    if getattr(x, "ndim", None) == 0 or not hasattr(x, "ndim"):
        if r.ndim == 0:
            return r[()]
    return r


def datetime_as_string(arr, unit=None, timezone="naive", casting="same_kind"):
    """`np.datetime_as_string`.

    Only the `'naive'` and `'UTC'` timezones are supported; a `tzinfo` object
    (numpy's `'local'` mode) is a documented gap.
    """
    a = arr if isinstance(arr, ndarray) else _rnp.asarray(arr)
    if a.dtype.kind != "M":
        raise TypeError(
            "Input must have datetime64 dtype, "
            f"got {a.dtype!r} instead")
    if not isinstance(timezone, str):
        raise NotImplementedError(
            "rnp does not implement datetime_as_string(timezone=<tzinfo>)")
    strs = _rnp._datetime_strings(a, unit=unit, timezone=timezone,
                                  casting=casting)
    width = max((len(s) for s in strs), default=1) or 1
    from . import _core  # noqa: F401  (package init ordering)
    out = _rnp.array(strs, dtype=dtype(f"U{width}"))
    return out.reshape(a.shape)


class busdaycalendar:
    """numpy's `busdaycalendar` — present so `test_datetime.py` collects."""

    def __init__(self, weekmask="1111100", holidays=None):
        raise NotImplementedError(
            "rnp does not implement numpy.busdaycalendar yet")


def _busday_gap(name):
    def fn(*args, **kwargs):
        raise NotImplementedError(
            f"rnp does not implement numpy.{name} yet")
    fn.__name__ = name
    return fn


is_busday = _busday_gap("is_busday")
busday_offset = _busday_gap("busday_offset")
busday_count = _busday_gap("busday_count")
