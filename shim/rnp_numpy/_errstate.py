"""`np.seterr` / `np.errstate` and the RuntimeWarnings the ufuncs raise.

The Rust inner loops accumulate a four-bit mask (divide / over / under /
invalid) in `rnp_core::fpe`; every ufunc entry point drains it and calls
[`report`], which applies the current error state.  numpy's defaults are
``divide='warn', over='warn', under='ignore', invalid='warn'``.
"""

import contextvars as _contextvars
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

_DEFAULT = {"divide": "warn", "over": "warn", "under": "ignore",
            "invalid": "warn"}

_VALID = ("ignore", "warn", "raise", "call", "print", "log")

#: numpy 2.x keeps the error state in a `contextvars.ContextVar` rather than a
#: process global, which is what makes `np.errstate` safe under asyncio: every
#: task runs in its own copy of the context, so a `with np.errstate(...)` inside
#: one coroutine is invisible to the others (test_errstate.test_asyncio_safe).
#: The value is the immutable pair `(state, call)`.
_EXTOBJ = _contextvars.ContextVar(
    "rnp_errstate", default=(_DEFAULT.copy(), None))


def _get():
    return _EXTOBJ.get()


def _sync_engine(state):
    # The engine only pays for underflow detection when someone asks for it.
    _rnp._watch_underflow(state["under"] != "ignore")


def geterr():
    return dict(_get()[0])


def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    state, call = _get()
    old = dict(state)
    new = dict(state)
    for key, value in (("divide", divide), ("over", over), ("under", under),
                       ("invalid", invalid)):
        v = value if value is not None else all
        if v is None:
            continue
        if v not in _VALID:
            raise ValueError(f"{v!r} is not a valid error handler name")
        new[key] = v
    _EXTOBJ.set((new, call))
    _sync_engine(new)
    return old


def geterrcall():
    return _get()[1]


def seterrcall(func):
    state, old = _get()
    if func is not None and not (callable(func) or hasattr(func, "write")):
        raise ValueError("Only callable can be used as callback")
    _EXTOBJ.set((state, func))
    return old


def clear():
    """Drop any flags the engine has accumulated but not yet reported."""
    _rnp._fpe_clear()


def report(flags, where, stacklevel=4):
    """Apply the current error state to `flags` raised inside `where`."""
    state, cb = _get()
    for name, bit in _ORDER:
        if not (flags & bit):
            continue
        action = state[name]
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


#: Distinguishes `errstate()` from `errstate(call=None)`: the latter really does
#: install `None` as the callback, which `np.geterrcall()` must then report.
_UNSET = object()


class errstate:
    """Context manager / decorator form of `seterr`.

    Entering pushes a new value onto the `_EXTOBJ` contextvar and exiting
    resets the token, so the state unwinds exactly with the `with` block even
    across `await` points. A single instance may only be entered once: numpy
    refuses the second entry because one stored token cannot describe two
    concurrent unwinds.
    """

    def __init__(self, *, call=_UNSET, **kwargs):
        self._kwargs = kwargs
        self._call = call
        self._token = None

    def __enter__(self):
        if self._token is not None:
            raise TypeError("Cannot enter `np.errstate` twice.")
        clear()
        state, call = _get()
        new = dict(state)
        for key, value in self._kwargs.items():
            if key == "all":
                continue
            if key not in new:
                raise TypeError(
                    f"errstate() got an unexpected keyword argument {key!r}")
        allv = self._kwargs.get("all")
        for key in new:
            v = self._kwargs.get(key, None)
            v = v if v is not None else allv
            if v is None:
                continue
            if v not in _VALID:
                raise ValueError(f"{v!r} is not a valid error handler name")
            new[key] = v
        if self._call is not _UNSET:
            call = self._call
            if call is not None and not (callable(call)
                                         or hasattr(call, "write")):
                raise ValueError("Only callable can be used as callback")
        self._token = _EXTOBJ.set((new, call))
        _sync_engine(new)
        return self

    def __exit__(self, *exc):
        _EXTOBJ.reset(self._token)
        _sync_engine(_get()[0])
        return False

    def __call__(self, func):
        import functools

        @functools.wraps(func)
        def wrapper(*a, **k):
            # A fresh instance per call: the decorated function may recurse or
            # run concurrently, and each activation needs its own token.
            with type(self)(call=self._call, **self._kwargs):
                return func(*a, **k)

        return wrapper
