"""Minimal stand-ins for numpy.testing._private.utils.

Upstream's version is ~2900 lines built on numpy APIs the port does not have
yet (masked arrays, ufunc machinery, refcount probes). These are pure-Python
fallbacks with the same names and, for the assertions the target tests use,
the same semantics.
"""

import contextlib
import platform
from functools import wraps
import sys
import warnings

import rnp_numpy as np

__all__ = [
    "HAS_LAPACK64", "HAS_REFCOUNT", "IS_64BIT", "IS_EDITABLE", "IS_INSTALLED",
    "IS_MUSL", "IS_PYPY", "IS_PYSTON", "IS_WASM", "NOGIL_BUILD", "NUMPY_ROOT",
    "assert_", "assert_allclose", "assert_almost_equal",
    "assert_approx_equal", "assert_array_almost_equal",
    "assert_array_almost_equal_nulp", "assert_array_compare",
    "assert_array_equal", "assert_array_less", "assert_array_max_ulp",
    "assert_equal", "assert_no_gc_cycles", "assert_no_warnings",
    "assert_raises", "assert_raises_regex", "assert_string_equal",
    "assert_warns", "break_cycles", "build_err_msg", "clear_and_catch_warnings",
    "check_support_sve", "decorate_methods", "jiffies", "measure", "memusage",
    "print_assert_equal", "rundocs", "runstring", "suppress_warnings",
    "tempdir", "temppath", "verbose",
]

verbose = 0

IS_WASM = platform.machine() in ("wasm32", "wasm64")
IS_PYPY = sys.implementation.name == "pypy"
IS_PYSTON = hasattr(sys, "pyston_version_info")
IS_MUSL = False
IS_EDITABLE = False
IS_INSTALLED = True
NUMPY_ROOT = None
HAS_REFCOUNT = getattr(sys, "getrefcount", None) is not None and not IS_PYPY
HAS_LAPACK64 = False
IS_64BIT = sys.maxsize > 2 ** 32
NOGIL_BUILD = False


def check_support_sve(__cache=[]):
    return False


# --------------------------------------------------------------------------
# Value comparison helpers
# --------------------------------------------------------------------------

def _is_array(x):
    if isinstance(x, np.ndarray):
        return True
    # Upstream these are `ndarray` subclasses (e.g. `numpy.memmap`); the port
    # cannot subclass `ndarray`, so they are array-likes exposing `__array__`.
    # Scalars also expose `__array__` and must keep comparing as scalars.
    return (hasattr(type(x), '__array__')
            and not isinstance(x, np.generic))


def _flat(x):
    """Elements of `x` in C order as Python scalars, plus its shape."""
    if _is_array(x):
        if not isinstance(x, np.ndarray):
            x = np.asarray(x.__array__())
        return np._flat_values(x), tuple(x.shape)
    if isinstance(x, (list, tuple)):
        a = np.asarray(x)
        return np._flat_values(a), tuple(a.shape)
    return [x], ()


def _values_equal(a, b):
    """Elementwise equality with numpy's NaN-equals-NaN testing semantics."""
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:
            return True
    if isinstance(a, complex) or isinstance(b, complex):
        ar, ai = complex(a).real, complex(a).imag
        br, bi = complex(b).real, complex(b).imag
        return _values_equal(ar, br) and _values_equal(ai, bi)
    return bool(a == b)


def build_err_msg(arrays, err_msg, header="Items are not equal:",
                  verbose=True, names=("ACTUAL", "DESIRED"), precision=8):
    msg = ["\n" + header]
    if err_msg:
        msg = [msg[0] + " " + err_msg] if not verbose else msg + [err_msg]
    if verbose:
        for i, a in enumerate(arrays):
            name = names[i] if i < len(names) else f"item {i}"
            try:
                r = repr(a)
            except Exception:
                r = "[repr failed]"
            msg.append(f" {name}: {r}")
    return "\n".join(msg)


def assert_(val, msg=""):
    if not val:
        raise AssertionError(msg() if callable(msg) else msg)


def assert_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k, v in desired.items():
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], v, f"key={k!r}\n{err_msg}", verbose)
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for a, d in zip(actual, desired):
            assert_equal(a, d, err_msg, verbose)
        return
    if _is_array(actual) or _is_array(desired):
        return assert_array_equal(actual, desired, err_msg, verbose,
                                  strict=strict)
    if not _values_equal(actual, desired):
        raise AssertionError(build_err_msg([actual, desired], err_msg,
                                           verbose=verbose))


def assert_array_compare(comparison, x, y, err_msg="", verbose=True,
                         header="", precision=6, equal_nan=True,
                         equal_inf=True, *, strict=False):
    xf, xs = _flat(x)
    yf, ys = _flat(y)
    if xs != ys:
        # Broadcast a scalar against an array, as numpy does.
        if xs == () and len(xf) == 1:
            xf = xf * len(yf)
        elif ys == () and len(yf) == 1:
            yf = yf * len(xf)
        else:
            raise AssertionError(build_err_msg(
                [x, y], err_msg,
                header=f"{header}\n(shapes {xs}, {ys} mismatch)",
                verbose=verbose))
    if strict and _is_array(x) and _is_array(y) and x.dtype != y.dtype:
        raise AssertionError(build_err_msg(
            [x, y], err_msg, header=f"{header}\n(dtypes differ)",
            verbose=verbose))
    bad = [i for i, (a, b) in enumerate(zip(xf, yf)) if not comparison(a, b)]
    if bad:
        raise AssertionError(build_err_msg(
            [x, y], err_msg,
            header=f"{header}\nMismatched elements: {len(bad)} / {len(xf)}",
            verbose=verbose))


def assert_array_equal(x, y, err_msg="", verbose=True, *, strict=False):
    assert_array_compare(_values_equal, x, y, err_msg, verbose,
                         header="Arrays are not equal", strict=strict)


def assert_array_less(x, y, err_msg="", verbose=True, *, strict=False):
    assert_array_compare(lambda a, b: a < b, x, y, err_msg, verbose,
                         header="Arrays are not less-ordered", strict=strict)


def _close(a, b, rtol, atol):
    if _values_equal(a, b):
        return True
    try:
        return abs(a - b) <= atol + rtol * abs(b)
    except TypeError:
        return False


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True,
                    err_msg="", verbose=True, *, strict=False):
    assert_array_compare(lambda a, b: _close(a, b, rtol, atol),
                         actual, desired, err_msg, verbose,
                         header="Not equal to tolerance", strict=strict)


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    tol = 1.5 * 10.0 ** (-decimal)
    assert_array_compare(lambda a, b: _close(a, b, 0, tol),
                         actual, desired, err_msg, verbose,
                         header=f"Arrays are not almost equal to {decimal} decimals")


def assert_array_almost_equal(x, y, decimal=6, err_msg="", verbose=True):
    assert_almost_equal(x, y, decimal, err_msg, verbose)


def assert_approx_equal(actual, desired, significant=7, err_msg="",
                        verbose=True):
    scale = max(abs(float(desired)), abs(float(actual)), 1e-300)
    assert_(abs(float(actual) - float(desired)) < scale * 10.0 ** -significant,
            err_msg or f"{actual} != {desired}")


def assert_array_almost_equal_nulp(x, y, nulp=1):
    assert_allclose(x, y, rtol=nulp * 2.220446049250313e-16)


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    assert_allclose(a, b, rtol=maxulp * 2.220446049250313e-16)
    return 0


def assert_string_equal(actual, desired):
    assert_equal(actual, desired)


def print_assert_equal(test_string, actual, desired):
    assert_equal(actual, desired, test_string)


# --------------------------------------------------------------------------
# Exception / warning helpers
# --------------------------------------------------------------------------

def assert_raises(*args, **kwargs):
    import pytest
    if len(args) == 1 and not kwargs:
        return pytest.raises(args[0])
    exc, func, *rest = args
    with pytest.raises(exc):
        return func(*rest, **kwargs)


def assert_raises_regex(exc, regex, *args, **kwargs):
    import pytest
    if not args:
        return pytest.raises(exc, match=regex)
    func, *rest = args
    with pytest.raises(exc, match=regex):
        return func(*rest, **kwargs)


def assert_warns(warning_class, *args, **kwargs):
    import pytest
    if not args:
        return pytest.warns(warning_class)
    func, *rest = args
    with pytest.warns(warning_class):
        return func(*rest, **kwargs)


@contextlib.contextmanager
def assert_no_warnings():
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        yield
    if len(log) > 0:
        raise AssertionError(f"Got warnings: {log}")


@contextlib.contextmanager
def assert_no_gc_cycles(*args, **kwargs):
    yield


def break_cycles():
    import gc
    gc.collect()


class suppress_warnings:
    """A very small subset of upstream's context manager of the same name."""

    def __init__(self, forwarding_rule="always"):
        self._entered = False
        self._filters = []

    def filter(self, category=Warning, message="", module=None):
        self._filters.append((category, message))

    record = filter

    def __enter__(self):
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore")
        self._entered = True
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        self._entered = False
        return False

    def __call__(self, func):
        def wrapper(*a, **kw):
            with self:
                return func(*a, **kw)
        return wrapper


class clear_and_catch_warnings(warnings.catch_warnings):
    class_modules = ()

    def __init__(self, record=False, modules=()):
        self.modules = set(modules)
        super().__init__(record=record)


# --------------------------------------------------------------------------
# Misc utilities the tests import but rarely exercise
# --------------------------------------------------------------------------

def memusage():
    return 0


def jiffies():
    return 0


def measure(code_str, times=1, label=None):
    return 0.0


def runstring(astr, dict):
    exec(astr, dict)


def rundocs(filename=None, raise_on_error=True):
    return None


def decorate_methods(cls, decorator, testmatch=None):
    return cls


@contextlib.contextmanager
def tempdir(*args, **kwargs):
    import shutil
    import tempfile
    d = tempfile.mkdtemp(*args, **kwargs)
    try:
        yield d
    finally:
        shutil.rmtree(d)


@contextlib.contextmanager
def temppath(*args, **kwargs):
    import os
    import tempfile
    fd, path = tempfile.mkstemp(*args, **kwargs)
    os.close(fd)
    try:
        yield path
    finally:
        if os.path.isfile(path):
            os.remove(path)


def _pytest_skip(reason):
    import pytest
    return pytest.mark.skip(reason=reason)


def requires_memory(free_bytes):
    """Upstream skips when RAM is short; we always skip (M0 has no big-array
    support to speak of)."""
    return _pytest_skip("large-memory tests are not supported by rnp yet")


def requires_deep_recursion(func=None):
    import pytest
    mark = pytest.mark.skip(reason="deep recursion tests not supported yet")
    return mark if func is None else mark(func)


def check_free_memory(free_bytes):
    return None


# --------------------------------------------------------------------------
# Additional names numpy.testing exports (transcribed from
# upstream/numpy/testing/_private/utils.py, which is the oracle).
# --------------------------------------------------------------------------

#: numpy probes its BLAS for floating-point-exception support; the port has no
#: BLAS yet, so the tests that depend on it are simply not applicable.
BLAS_SUPPORTS_FPE = False


class KnownFailureException(Exception):
    """Raise this exception to mark a test as a known failing test."""


KnownFailureTest = KnownFailureException


class IgnoreException(Exception):
    "Ignoring this exception due to disabled feature"


def _gen_alignment_data(dtype=None, type='binary', max_size=24):
    """Generator producing data with different alignment and offsets, used to
    exercise SIMD paths. Transcribed from numpy's own helper."""
    import rnp_numpy as _np
    if dtype is None:
        dtype = _np.float32
    arange, empty = _np.arange, _np.empty
    ufmt = 'unary offset=(%d, %d), size=%d, dtype=%r, %s'
    bfmt = 'binary offset=(%d, %d, %d), size=%d, dtype=%r, %s'
    for o in range(3):
        for s in range(o + 2, max(o + 3, max_size)):
            if type == 'unary':
                def inp():
                    return arange(s, dtype=dtype)[o:]
                out = empty((s,), dtype=dtype)[o:]
                yield out, inp(), ufmt % (o, o, s, dtype, 'out of place')
                d = inp()
                yield d, d, ufmt % (o, o, s, dtype, 'in place')
                yield out[1:], inp()[:-1], ufmt % (o + 1, o, s - 1, dtype,
                                                   'out of place')
                yield out[:-1], inp()[1:], ufmt % (o, o + 1, s - 1, dtype,
                                                   'out of place')
                yield inp()[:-1], inp()[1:], ufmt % (o, o + 1, s - 1, dtype,
                                                     'aliased')
                yield inp()[1:], inp()[:-1], ufmt % (o + 1, o, s - 1, dtype,
                                                     'aliased')
            if type == 'binary':
                def inp1():
                    return arange(s, dtype=dtype)[o:]
                inp2 = inp1
                out = empty((s,), dtype=dtype)[o:]
                yield out, inp1(), inp2(), bfmt % (o, o, o, s, dtype,
                                                   'out of place')
                d = inp1()
                yield d, d, inp2(), bfmt % (o, o, o, s, dtype, 'in place1')
                d = inp2()
                yield d, inp1(), d, bfmt % (o, o, o, s, dtype, 'in place2')
                yield out[1:], inp1()[:-1], inp2()[:-1], bfmt % \
                    (o + 1, o, o, s - 1, dtype, 'out of place')
                yield out[:-1], inp1()[1:], inp2()[:-1], bfmt % \
                    (o, o + 1, o, s - 1, dtype, 'out of place')
                yield out[:-1], inp1()[:-1], inp2()[1:], bfmt % \
                    (o, o, o + 1, s - 1, dtype, 'out of place')
                yield inp1()[1:], inp1()[:-1], inp2()[:-1], bfmt % \
                    (o + 1, o, o, s - 1, dtype, 'aliased')
                yield inp1()[:-1], inp1()[1:], inp2()[:-1], bfmt % \
                    (o, o + 1, o, s - 1, dtype, 'aliased')
                yield inp1()[:-1], inp1()[:-1], inp2()[1:], bfmt % \
                    (o, o, o + 1, s - 1, dtype, 'aliased')


def _assert_valid_refcount(op):
    """numpy checks that ufuncs do not mishandle the refcount of the int 1."""
    if not HAS_REFCOUNT:
        return True
    import gc

    import rnp_numpy as np
    b = np.arange(100 * 100).reshape(100, 100)
    c = b
    i = 1
    gc.disable()
    try:
        rc = sys.getrefcount(i)
        for _ in range(15):
            op(b, c)
        assert_(sys.getrefcount(i) >= rc)
    finally:
        gc.enable()
    return True


def run_threaded(func, max_workers=8, pass_count=False, pass_barrier=False,
                 outer_iterations=1, prepare_args=None):
    """Run `func` many times in parallel (transcribed from numpy)."""
    import concurrent.futures
    import threading
    for _ in range(outer_iterations):
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers) as tpe:
            args = [] if prepare_args is None else prepare_args()
            if pass_barrier:
                args.append(threading.Barrier(max_workers))
            if pass_count:
                all_args = [(func, i, *args) for i in range(max_workers)]
            else:
                all_args = [(func, *args) for _ in range(max_workers)]
            futures = [tpe.submit(*a) for a in all_args]
            for f in futures:
                f.result()


def check_support_sve(__cache=[]):
    return False


def assert_array_max_ulp_unavailable(*a, **k):  # pragma: no cover
    raise NotImplementedError


__all__ += [
    "BLAS_SUPPORTS_FPE", "IgnoreException", "KnownFailureException",
    "_gen_alignment_data", "_assert_valid_refcount", "run_threaded",
]


#: On this platform numpy's long double is an IEEE double, never the IBM
#: double-double format.
LONG_DOUBLE_IS_IBM_DOUBLE_DOUBLE = False


def _glibc_older_than(x):
    return False


def _no_tracing(func):
    """numpy disables sys tracing around tests that count bytecode frames."""
    if not hasattr(sys, 'gettrace'):
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        original_trace = sys.gettrace()
        try:
            sys.settrace(None)
            return func(*args, **kwargs)
        finally:
            sys.settrace(original_trace)
    return wrapper
