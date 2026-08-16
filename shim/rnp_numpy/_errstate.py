"""`np.seterr` / `np.errstate` and the RuntimeWarnings the ufuncs raise.

The Rust inner loops accumulate a four-bit mask (divide / over / under /
invalid) in `rnp_core::fpe`; every ufunc entry point drains it and calls
[`report`], which applies the current error state.  numpy's defaults are
``divide='warn', over='warn', under='ignore', invalid='warn'``.
"""

import warnings

import _rnp

DIVIDE, OVER, UNDER, INVALID = 1, 2, 4, 8

_ORDER = (("divide", DIVIDE), ("over", OVER), ("under", UNDER),
          ("invalid", INVALID))

_TEXT = {
    "divide": "divide by zero encountered",
    "over": "overflow encountered",
    "under": "underflow encountered",
    "invalid": "invalid value encountered",
}

_STATE = {"divide": "warn", "over": "warn", "under": "ignore",
          "invalid": "warn"}

_CALL = {"call": None}

_VALID = ("ignore", "warn", "raise", "call", "print", "log")


def geterr():
    return dict(_STATE)


def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    old = dict(_STATE)
    for key, value in (("divide", divide), ("over", over), ("under", under),
                       ("invalid", invalid)):
        v = value if value is not None else all
        if v is None:
            continue
        if v not in _VALID:
            raise ValueError(f"{v!r} is not a valid error handler name")
        _STATE[key] = v
    # The engine only pays for underflow detection when someone asks for it.
    _rnp._watch_underflow(_STATE["under"] != "ignore")
    return old


def geterrcall():
    return _CALL["call"]


def seterrcall(func):
    old = _CALL["call"]
    if func is not None and not (callable(func) or hasattr(func, "write")):
        raise ValueError("Only callable can be used as callback")
    _CALL["call"] = func
    return old


def clear():
    """Drop any flags the engine has accumulated but not yet reported."""
    _rnp._fpe_clear()


def report(flags, where, stacklevel=4):
    """Apply the current error state to `flags` raised inside `where`."""
    for name, bit in _ORDER:
        if not (flags & bit):
            continue
        action = _STATE[name]
        message = f"{_TEXT[name]} in {where}"
        if action == "ignore":
            continue
        if action == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=stacklevel)
        elif action == "raise":
            raise FloatingPointError(message)
        elif action == "print":
            print(f"Warning: {message}")
        elif action in ("call", "log"):
            cb = _CALL["call"]
            if cb is None:
                continue
            if action == "log" or hasattr(cb, "write"):
                cb.write(f"Warning: {message}\n")
            else:
                cb(message, flags)


def _from_engine(flags, where):
    """Called by the Rust engine after any op that touched the FP flags."""
    # Frames: report, _from_engine, then (the Rust frame is invisible) the
    # Python caller.
    report(flags, where, stacklevel=3)


_rnp._register_fpe_reporter(_from_engine)


def drain(where, stacklevel=4):
    """Read the engine's flags and report them."""
    flags = _rnp._fpe_take()
    if flags:
        report(flags, where, stacklevel)


class errstate:
    """Context manager / decorator form of `seterr`."""

    def __init__(self, *, call=None, **kwargs):
        self._kwargs = kwargs
        self._call = call
        self._old = None
        self._oldcall = None

    def __enter__(self):
        clear()
        self._old = seterr(**self._kwargs)
        if self._call is not None:
            self._oldcall = seterrcall(self._call)
        return self

    def __exit__(self, *exc):
        seterr(**self._old)
        if self._call is not None:
            seterrcall(self._oldcall)
        return False

    def __call__(self, func):
        import functools

        @functools.wraps(func)
        def wrapper(*a, **k):
            with type(self)(call=self._call, **self._kwargs):
                return func(*a, **k)

        return wrapper
