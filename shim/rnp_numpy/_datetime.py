"""datetime64 / timedelta64 helpers that live above the engine.

`datetime_data` and `datetime_as_string` are thin wrappers over the Rust
entry points (`np.isnat` is a real ufunc, so it lives in the ufunc table).
The business-day family performs Python argument parsing and broadcasting
here, then calls the direct Rust ports of NumPy's scalar calendar kernels.
"""

import _rnp

ndarray = _rnp.ndarray
dtype = _rnp.dtype
_dtype = _rnp.dtype
_raw_arange = _rnp.arange

datetime_data = _rnp.datetime_data


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


_DEFAULT_WEEKMASK = (True, True, True, True, True, False, False)
_UNSET = object()
_DAY_DTYPE = dtype("M8[D]")


def _parse_weekmask(value):
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        if len(value) == 7 and all(c in "01" for c in value):
            return tuple(c == "1" for c in value)
        names = {
            "Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3,
            "Fri": 4, "Sat": 5, "Sun": 6,
        }
        mask = [False] * 7
        i = 0
        while i < len(value):
            while i < len(value) and value[i].isspace():
                i += 1
            if i == len(value):
                break
            token = value[i:i + 3]
            if len(token) != 3 or token not in names:
                raise ValueError(
                    f'Invalid business day weekmask string "{value}"')
            mask[names[token]] = True
            i += 3
        return tuple(mask)
    if isinstance(value, ndarray):
        if value.ndim != 1:
            raise ValueError(
                "A business day weekmask array must have length 7")
        value = value.tolist()
    try:
        values = list(value)
    except TypeError:
        raise ValueError(
            "Couldn't convert object into a business day weekmask") from None
    if len(values) != 7:
        raise ValueError("A business day weekmask array must have length 7")
    return tuple(bool(v) for v in values)


def _date_array(value):
    return _rnp.array(value, dtype=_DAY_DTYPE)


def _flat_ints(value):
    return value.astype("int64").reshape((-1,)).tolist()


def _holiday_values(value, weekmask):
    if value is None:
        return ()
    holidays = _date_array(value)
    if holidays.ndim != 1:
        raise ValueError("holidays must be a provided as a one-dimensional array")
    return tuple(_rnp._busday_normalize(
        list(weekmask), _flat_ints(holidays)))


class busdaycalendar:
    """A normalized, immutable business-day calendar."""

    __slots__ = ("_weekmask", "_holidays")

    def __init__(self, weekmask="1111100", holidays=None):
        parsed = _parse_weekmask(weekmask)
        if not any(parsed):
            raise ValueError(
                "Cannot construct a numpy.busdaycal with a weekmask of all zeros")
        self._weekmask = parsed
        self._holidays = _holiday_values(holidays, parsed)

    @property
    def weekmask(self):
        return _rnp.array(list(self._weekmask), dtype="bool")

    @property
    def holidays(self):
        return _rnp.array(list(self._holidays), dtype="int64").astype(_DAY_DTYPE)


def _calendar_args(name, weekmask, holidays, busdaycal):
    if busdaycal is not None:
        if not isinstance(busdaycal, globals()["busdaycalendar"]):
            raise TypeError(
                f"busdaycal must be a numpy.busdaycalendar object, got {type(busdaycal)!r}")
        if weekmask is not _UNSET or holidays is not _UNSET:
            raise ValueError(
                "Cannot supply both the weekmask/holidays and the "
                f"busdaycal parameters to {name}()")
        return busdaycal._weekmask, busdaycal._holidays
    parsed = (_DEFAULT_WEEKMASK if weekmask is _UNSET
              else _parse_weekmask(weekmask))
    if not any(parsed):
        raise ValueError(
            "the business day weekmask must have at least one valid business day")
    normalized = _holiday_values(
        None if holidays is _UNSET else holidays, parsed)
    return parsed, normalized


def _broadcast(*arrays):
    shape = _rnp.broadcast_shapes(*[a.shape for a in arrays])
    return shape, tuple(_rnp.broadcast_to(a, shape) for a in arrays)


def _finish_busday(values, shape, result_dtype, out, name):
    result = _rnp.array(values, dtype=result_dtype).reshape(shape)
    if out is not None:
        if not isinstance(out, ndarray):
            raise ValueError(f"{name}: must provide a NumPy array for 'out'")
        out[...] = result
        return out
    return result[()] if shape == () else result


def busday_offset(dates, offsets, roll="raise", weekmask=_UNSET,
                  holidays=_UNSET, busdaycal=None, out=None):
    mask, days_off = _calendar_args(
        "busday_offset", weekmask, holidays, busdaycal)
    date_arr = _date_array(dates)
    offset_arr = _rnp.array(offsets, dtype="int64")
    shape, (date_arr, offset_arr) = _broadcast(date_arr, offset_arr)
    values = _rnp._busday_offset(
        _flat_ints(date_arr), _flat_ints(offset_arr),
        roll.decode() if isinstance(roll, bytes) else roll,
        list(mask), list(days_off))
    return _finish_busday(values, shape, _DAY_DTYPE, out, "busday_offset")


def busday_count(begindates, enddates, weekmask=_UNSET, holidays=_UNSET,
                 busdaycal=None, out=None):
    mask, days_off = _calendar_args(
        "busday_count", weekmask, holidays, busdaycal)
    begin = _date_array(begindates)
    end = _date_array(enddates)
    shape, (begin, end) = _broadcast(begin, end)
    values = _rnp._busday_count(
        _flat_ints(begin), _flat_ints(end), list(mask), list(days_off))
    return _finish_busday(values, shape, "int64", out, "busday_count")


def is_busday(dates, weekmask=_UNSET, holidays=_UNSET,
              busdaycal=None, out=None):
    mask, days_off = _calendar_args(
        "is_busday", weekmask, holidays, busdaycal)
    dates = _date_array(dates)
    values = _rnp._is_busday(
        _flat_ints(dates), list(mask), list(days_off))
    return _finish_busday(values, dates.shape, "bool", out, "is_busday")


# ---------------------------------------------------------------------------
# Scalar metadata and Python datetime conversion
# ---------------------------------------------------------------------------

# These patches live in the datetime-only shim module so the shared scalar
# module stays outside this lane. NumPy accepts bracketed and byte metadata in
# scalar constructors even though dtype strings use the unbracketed Unicode
# spelling internally.
from ._scalars import datetime64 as _datetime64, timedelta64 as _timedelta64


def _normalize_scalar_unit(unit):
    if isinstance(unit, bytes):
        unit = unit.decode()
    elif isinstance(unit, tuple) and len(unit) == 2:
        base, number = unit
        if isinstance(base, bytes):
            base = base.decode()
        unit = (base, number)
    if isinstance(unit, str) and len(unit) >= 2 \
            and unit[0] == "[" and unit[-1] == "]":
        unit = unit[1:-1]
    return unit


_datetime64_new = _datetime64.__new__
_timedelta64_new = _timedelta64.__new__
_datetime64_astype = _datetime64.astype
_timedelta64_astype = _timedelta64.astype


def _new_datetime64(cls, value=None, unit=None):
    if isinstance(value, int) and not -(2 ** 63) <= value < 2 ** 63:
        raise OverflowError("int too big to convert")
    return _datetime64_new(cls, value, _normalize_scalar_unit(unit))


def _new_timedelta64(cls, value=None, unit=None):
    if isinstance(value, int) and not -(2 ** 63) <= value < 2 ** 63:
        raise OverflowError("int too big to convert")
    return _timedelta64_new(cls, value, _normalize_scalar_unit(unit))


def _astype_datetime64(self, target, *args, **kwargs):
    import datetime as _pydatetime
    if target in (_pydatetime.date, _pydatetime.datetime, object):
        return self.item()
    return _datetime64_astype(self, target, *args, **kwargs)


def _astype_timedelta64(self, target, *args, **kwargs):
    import datetime as _pydatetime
    if target in (_pydatetime.timedelta, object):
        return self.item()
    return _timedelta64_astype(self, target, *args, **kwargs)


_datetime64.__new__ = _new_datetime64
_timedelta64.__new__ = _new_timedelta64
_datetime64.astype = _astype_datetime64
_timedelta64.astype = _astype_timedelta64


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
        elif d.kind == "O" and kind is None:
            return _raw_arange(start, stop, step, None).astype(d)
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
