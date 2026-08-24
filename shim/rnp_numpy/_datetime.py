"""datetime64 / timedelta64 helpers that live above the engine.

`datetime_data`, `isnat` and `datetime_as_string` are thin wrappers over the
Rust entry points; the business-day family is a documented gap that raises
`NotImplementedError` while still existing, so that code (and numpy's own test
module) can import it.
"""

import _rnp

ndarray = _rnp.ndarray
dtype = _rnp.dtype
_dtype = _rnp.dtype
_raw_arange = _rnp.arange

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


# ---------------------------------------------------------------------------
# arange over datetime64 / timedelta64
# ---------------------------------------------------------------------------

def _time_kind(obj):
    """`'M'`/`'m'` if `obj` is a datetime-like value, else None."""
    dt = getattr(obj, "dtype", None)
    if dt is not None and dt.kind in "mM":
        return dt.kind
    return None


def _delta_dtype(obj):
    """The timedelta64 dtype whose unit `obj` contributes to the range."""
    dt = getattr(obj, "dtype", None)
    if dt is None or dt.kind not in "mM":
        return None
    unit, num = datetime_data(dt)
    return dtype("m8" if unit == "generic" else f"m8[{num}{unit}]")


def arange(start, stop=None, step=None, dtype=None, *, device=None,
           like=None):
    """`np.arange`, extended to datetime64 / timedelta64 ranges.

    numpy resolves a single common unit across start/stop/step (which is what
    makes `arange(m8[D], m8[M])` a TypeError, since days and months have no
    common divisor), builds the integer range in that unit, and relabels the
    result. Anything without a datetime-like operand goes straight to the
    engine's numeric `arange`.
    """
    kind = None
    for v in (start, stop, step):
        k = _time_kind(v)
        if k is not None:
            kind = "M" if "M" in (kind, k) else "m"
    if dtype is not None:
        d = _dtype(dtype)
        if d.kind in "mM":
            kind = "M" if "M" in (kind, d.kind) else "m"
        elif kind is None:
            return _raw_arange(start, stop, step, dtype)
    if kind is None:
        return _raw_arange(start, stop, step, dtype)

    # `arange(stop)` means `arange(0, stop)`.
    if stop is None:
        start, stop = None, start

    def coerce(v, as_delta):
        """One bound as a datetime-like 0-d array (strings included)."""
        if v is None:
            return None
        if _time_kind(v) is not None:
            return _rnp.asarray(v)
        if isinstance(v, (str, bytes)):
            k = "m" if as_delta else kind
            return _rnp.array(v, _dtype("M8" if k == "M" else "m8"))
        return v  # a plain number: it counts in the resolved unit

    a, b, s = (coerce(start, kind == "m"), coerce(stop, kind == "m"),
               coerce(step, True))

    # The common unit: every datetime-like operand contributes, as a timedelta
    # (the *step* of a datetime range is a duration, not an instant).
    common = None
    for v in (a, b, s):
        d = _delta_dtype(v)
        if d is not None:
            common = d if common is None else _rnp.promote_types(common, d)
    if dtype is not None and _dtype(dtype).kind in "mM":
        u, num = datetime_data(_dtype(dtype))
        d = _dtype("m8") if u == "generic" else _dtype(f"m8[{num}{u}]")
        common = d if common is None else _rnp.promote_types(common, d)
    if common is None or datetime_data(common)[0] == "generic":
        raise ValueError(
            "Cannot create a NumPy datetime other than NaT with generic units")
    unit, num = datetime_data(common)
    spec = f"{num}{unit}"
    target = _dtype(f"{kind}8[{spec}]")
    delta = _dtype(f"m8[{spec}]")

    def raw(v, as_delta, default):
        if v is None:
            return default
        if _time_kind(v) is not None:
            return int(v.astype(delta if as_delta else target)
                       .astype("int64")[()])
        return int(v)

    lo = raw(a, kind == "m", 0)
    hi = raw(b, kind == "m", 0)
    inc = raw(s, True, 1)
    if inc == 0:
        raise ValueError("step may not be zero")
    span = hi - lo
    n = 0 if (span > 0) != (inc > 0) or span == 0 else -(-span // inc)
    return _rnp.array([lo + i * inc for i in range(n)],
                      dtype="int64").astype(target)
