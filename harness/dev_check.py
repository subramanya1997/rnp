#!/usr/bin/env python3
"""Differential check: the Rust port (`_rnp`) vs real numpy, side by side.

Both libraries are imported normally in this process (no import redirection),
identical inputs are fed to each, and the results are compared with real
numpy's own assert_array_equal -- exact for integers and bool, exact for the
simple float ops M0 implements (NaN is treated as equal to NaN).

The port's arrays are read back through the buffer protocol, so the bytes
compared are the bytes the Rust engine actually wrote.

Usage: .venv/bin/python harness/dev_check.py [--seed N]
"""
import argparse
import itertools
import math
import os
import random
import sys
import traceback
import warnings

import numpy as np

import _rnp

# The M3 surface -- scalar types, ufunc objects, errstate -- lives in the pure
# Python shim package `rnp_numpy`, not in the `_rnp` extension. It is a normal
# package named `rnp_numpy`, so it can be imported alongside real numpy in this
# same process; no import redirection is involved.
#
# The import is deliberately *lazy* (see load_shim(), called from
# run_m3_sections() after every M0-M2 section has finished). Importing the shim
# registers the scalar types with the extension, which changes what `_rnp`
# reductions and element access hand back -- Python floats become numpy-style
# scalars. The M0-M2 sections above were written against the pre-M3 behaviour
# and must keep testing exactly what they tested before, so the shim is not
# allowed to exist until they are done.
_SHIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shim")
if _SHIM_DIR not in sys.path:
    sys.path.insert(0, _SHIM_DIR)

rnp = None
SHIM_IMPORT_ERROR = None


def load_shim():
    """Import the shim, once, returning True on success."""
    global rnp, SHIM_IMPORT_ERROR
    if rnp is not None:
        return True
    try:
        import rnp_numpy
    except Exception:  # noqa: BLE001
        SHIM_IMPORT_ERROR = traceback.format_exc().strip().splitlines()[-1]
        return False
    rnp = rnp_numpy
    return True

DTYPES = [
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "complex64", "complex128",
]

#: Every dtype spelling the M1 constructor is expected to round-trip.
DTYPE_SPECS = (
    DTYPES
    + ["?", "b", "h", "i", "l", "q", "p", "n", "B", "H", "I", "L", "Q", "P", "N",
       "e", "f", "d", "F", "D", "c",
       "b1", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8",
       "f2", "f4", "f8", "c8", "c16",
       "int", "float", "double", "single", "half", "complex", "bool_",
       "byte", "short", "intc", "long", "longlong", "intp", "uintp",
       "ubyte", "ushort", "uintc", "ulong", "ulonglong", "csingle", "cdouble"]
    # byte-order prefixes
    + [p + c for p in "<>=|" for c in
       ["b1", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4",
        "f8", "c8", "c16"]]
    # flexible descriptors
    + ["S", "U", "V", "a"]
    + [f"S{n}" for n in (1, 3, 5, 17, 64)]
    + [f"U{n}" for n in (1, 3, 5, 17, 64)]
    + [f"V{n}" for n in (1, 3, 5, 16, 100)]
    + ["<U3", ">U3", "|S5", "|V8"]
    # comma-separated and subarray format strings
    + ["i4,f8", "f8,i4,u1", "i4, f8", "S3,U2", "(2,2)f4", "3f8", "2i4",
       "(3,)f4", "(2,3)i2", "i4,(2,2)f4"]
)

#: Structured / subarray dtypes given as Python objects.
DTYPE_OBJECTS = [
    [("a", "i4"), ("b", "f8")],
    [("a", "i4"), ("b", "f8"), ("c", "S3")],
    [("a", "i1"), ("b", "f8")],
    [("x", [("p", "i4"), ("q", "f4")]), ("y", "f8")],
    [("a", "f4", (2, 3)), ("b", "i8")],
    [("a", "f4", (2,))],
    [(("A", "a"), "i4")],
    [(("A", "a"), "i4"), ("b", "u2")],
    [],
    {"names": ["a", "b"], "formats": ["i4", "f8"]},
    {"names": ["a", "b"], "formats": ["i4", "f8"], "offsets": [0, 8]},
    {"names": ["a", "b"], "formats": ["i4", "f8"], "offsets": [0, 8],
     "itemsize": 32},
    {"names": ["a"], "formats": ["i4"], "titles": ["A"]},
    ("f4", (2, 2)),
    ("i4", 3),
    (np.dtype(("i4", (2,))), (3,)),
]

#: Dtype properties compared attribute-by-attribute against real numpy.
DTYPE_ATTRS = ["str", "kind", "char", "num", "itemsize", "alignment",
               "byteorder", "name", "isnative", "hasobject", "names",
               "isalignedstruct", "ndim", "shape"]

FAILURES = []
CHECKS = 0


def as_np(port_array):
    """Wrap one of our arrays as a real numpy array, zero-copy."""
    return np.asarray(port_array)


def check(name, want, got_port):
    """Compare a real-numpy result with a port result."""
    global CHECKS
    CHECKS += 1
    got = as_np(got_port)
    try:
        if want.dtype != got.dtype:
            raise AssertionError(f"dtype {got.dtype} != numpy's {want.dtype}")
        if want.shape != got.shape:
            raise AssertionError(f"shape {got.shape} != numpy's {want.shape}")
        np.testing.assert_array_equal(got, want, strict=True)
    except Exception as exc:
        FAILURES.append((name, f"{exc}".strip().splitlines()[0][:300]))


def sample(rng, dtype, size):
    """Deterministic sample data for a dtype, identical for both libraries."""
    if dtype == "bool":
        return np.array([rng.random() > 0.5 for _ in range(size)], dtype=bool)
    if dtype.startswith("uint"):
        hi = min(2 ** (8 * np.dtype(dtype).itemsize) - 1, 10_000)
        return np.array([rng.randrange(0, hi) for _ in range(size)], dtype=dtype)
    if dtype.startswith("int"):
        lim = min(2 ** (8 * np.dtype(dtype).itemsize - 1) - 1, 10_000)
        return np.array([rng.randrange(-lim, lim) for _ in range(size)], dtype=dtype)
    if dtype.startswith("float"):
        return np.array([rng.uniform(-1e3, 1e3) for _ in range(size)], dtype=dtype)
    return np.array(
        [complex(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(size)],
        dtype=dtype,
    )


def to_port(a):
    """Rebuild a numpy array as a port array through nested lists.

    The bytes are then compared against numpy's own, so any divergence in the
    element-by-element conversion shows up as a failure rather than hiding.
    """
    flat = _rnp.array(a.ravel().tolist(), str(a.dtype)) if a.size else \
        _rnp.zeros(a.shape, str(a.dtype))
    return flat.reshape(a.shape) if a.size else flat


def _reduction_repr(value, ref_dtype):
    """Bytes of a reduction result, in a form comparable across libraries.

    Whole-array reductions come back from the port as plain Python numbers
    (real numpy scalar types are a later milestone), so the *value* is
    compared bit-exactly after re-encoding it in numpy's result dtype, and
    the result *dtype* is checked separately through a `keepdims=True` call,
    where both libraries hand back a real array.
    """
    if hasattr(value, "dtype") or isinstance(value, np.ndarray):
        a = np.asarray(value)
        return (str(a.dtype), a.shape, a.tobytes())
    a = np.asarray(value, dtype=ref_dtype)
    return (str(a.dtype), a.shape, a.tobytes())


def _bool_or_exc(fn):
    """Run `fn`, returning its value or the exception type name."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


def _port_spec(spec):
    """Translate a numpy dtype spec into the equivalent port spec."""
    if isinstance(spec, np.dtype):
        if spec.names is not None:
            return _port_spec([(n, spec.fields[n][0]) for n in spec.names])
        if spec.subdtype is not None:
            base, shape = spec.subdtype
            return (_port_spec(base), shape)
        return _rnp.dtype(spec.str)
    if isinstance(spec, list):
        return [tuple(_port_spec(x) for x in entry) for entry in spec]
    if isinstance(spec, tuple):
        return tuple(_port_spec(x) for x in spec)
    if isinstance(spec, dict):
        return {k: ([_port_spec(f) for f in v] if k == "formats" else v)
                for k, v in spec.items()}
    return spec


def _compare_dtype(label, want, got):
    """Compare every dtype attribute the port models against numpy's."""
    for attr in DTYPE_ATTRS:
        w = getattr(want, attr)
        g = getattr(got, attr)
        if attr == "shape":
            w, g = tuple(w), tuple(g)
        if w != g:
            FAILURES.append((f"{label}.{attr}", f"{g!r} != numpy's {w!r}"))
    if repr(want) != repr(got):
        FAILURES.append((f"{label} repr", f"{got!r} != numpy's {want!r}"))
    if str(want) != str(got):
        FAILURES.append((f"{label} str", f"{str(got)!r} != numpy's {str(want)!r}"))
    if (want.fields is None) != (got.fields is None):
        FAILURES.append((f"{label}.fields", "presence differs from numpy"))
    elif want.fields is not None:
        w = {k: (str(v[0]), v[1]) for k, v in want.fields.items()}
        g = {k: (str(v[0]), v[1]) for k, v in got.fields.items()}
        if w != g:
            FAILURES.append((f"{label}.fields", f"{g!r} != numpy's {w!r}"))
    if (want.subdtype is None) != (got.subdtype is None):
        FAILURES.append((f"{label}.subdtype", "presence differs from numpy"))
    elif want.subdtype is not None:
        if (repr(want.subdtype[0]), want.subdtype[1]) != (
                repr(got.subdtype[0]), tuple(got.subdtype[1])):
            FAILURES.append((f"{label}.subdtype",
                             f"{got.subdtype!r} != numpy's {want.subdtype!r}"))


# --------------------------------------------------------------------------
# Differential indexing (M2)
#
# A few hundred random index expressions are built once and applied to
# identical arrays in both libraries. Values, dtype, shape, error type *and*
# view-vs-copy semantics (observed through write-back) must all agree.
# --------------------------------------------------------------------------

#: The shapes indexed against, with the dtype used for each.
INDEX_SHAPES = [(7,), (5, 4), (3, 4, 5), (2, 3, 2, 3), (0, 3), (), (1, 1)]

INDEX_DTYPES = ["int32", "int64", "float64", "float32", "bool", "complex128"]


def _random_index(rng, shape, allow_advanced=True):
    """One random index expression, as a tuple of Python-level pieces.

    Returned as a list of *descriptors* so the very same expression can be
    rebuilt separately for numpy and for the port (index arrays must not be
    shared between the two libraries).
    """
    nd = len(shape)
    items = []
    axis = 0
    n_adv = 0
    while axis < nd:
        r = rng.random()
        if r < 0.05:
            items.append(("newaxis", None))
            continue
        if r < 0.10 and not any(k == "ellipsis" for k, _ in items):
            # An ellipsis swallows the remaining axes.
            items.append(("ellipsis", None))
            axis = nd
            break
        n = shape[axis]
        if r < 0.35:
            if n == 0:
                items.append(("slice", (None, None, None)))
            else:
                items.append(("int", rng.randrange(-n, n)))
            axis += 1
        elif r < 0.70:
            lo = rng.choice([None, rng.randrange(-n - 1, n + 1)]) if n else None
            hi = rng.choice([None, rng.randrange(-n - 1, n + 1)]) if n else None
            st = rng.choice([None, 1, 2, 3, -1, -2])
            items.append(("slice", (lo, hi, st)))
            axis += 1
        elif allow_advanced and r < 0.85 and n > 0 and n_adv < 2:
            k = rng.randrange(1, 4)
            vals = [rng.randrange(-n, n) for _ in range(k)]
            if rng.random() < 0.3:
                items.append(("intarray2d", (vals, n)))
            else:
                items.append(("intarray", vals))
            n_adv += 1
            axis += 1
        elif allow_advanced and n_adv < 1:
            # A boolean mask over 1 or 2 axes.
            ax = 2 if (rng.random() < 0.3 and axis + 1 < nd) else 1
            mask_shape = tuple(shape[axis:axis + ax])
            mask = [rng.random() < 0.5 for _ in range(int(np.prod(mask_shape)))]
            items.append(("boolmask", (mask, mask_shape)))
            n_adv += 1
            axis += ax
        else:
            items.append(("slice", (None, None, None)))
            axis += 1
    if rng.random() < 0.15:
        items.append(("newaxis", None))
    return items


def _build_index(items, mod, mk):
    """Rebuild an index expression for one library."""
    out = []
    for kind, payload in items:
        if kind == "newaxis":
            out.append(None)
        elif kind == "ellipsis":
            out.append(Ellipsis)
        elif kind == "int":
            out.append(payload)
        elif kind == "slice":
            out.append(slice(*payload))
        elif kind == "intarray":
            out.append(mk(np.array(payload, dtype=np.intp)))
        elif kind == "intarray2d":
            vals, _n = payload
            out.append(mk(np.array([vals, vals[::-1]], dtype=np.intp)))
        elif kind == "boolmask":
            mask, mask_shape = payload
            out.append(mk(np.array(mask, dtype=bool).reshape(mask_shape)))
        else:  # pragma: no cover
            raise AssertionError(kind)
    return tuple(out)


def _describe(items):
    parts = []
    for kind, payload in items:
        if kind == "newaxis":
            parts.append("None")
        elif kind == "ellipsis":
            parts.append("...")
        elif kind == "int":
            parts.append(str(payload))
        elif kind == "slice":
            lo, hi, st = payload
            parts.append(":".join("" if v is None else str(v)
                                  for v in (lo, hi, st)).rstrip(":") or ":")
        elif kind == "intarray":
            parts.append(str(payload))
        elif kind == "intarray2d":
            parts.append(f"2d{payload[0]}")
        else:
            parts.append(f"mask{payload[1]}")
    return "[" + ", ".join(parts) + "]"


def check_indexing(rng, n_cases=400):
    """Differential test of __getitem__/__setitem__ against real numpy."""
    global CHECKS

    for _ in range(n_cases):
        shape = rng.choice(INDEX_SHAPES)
        dt = rng.choice(INDEX_DTYPES)
        size = int(np.prod(shape)) if shape else 1
        base = sample(rng, dt, max(size, 1)).reshape(shape) if size else \
            np.zeros(shape, dt)
        want_arr = base.copy()
        got_arr = to_port(base)

        items = _random_index(rng, shape)
        desc = f"{dt}{shape}{_describe(items)}"

        np_key = _build_index(items, np, lambda a: a)
        rp_key = _build_index(items, _rnp, to_port)

        CHECKS += 1
        try:
            want = want_arr[np_key]
            np_exc = None
        except Exception as e:  # noqa: BLE001
            want, np_exc = None, type(e).__name__
        try:
            got = got_arr[rp_key]
            rp_exc = None
        except Exception as e:  # noqa: BLE001
            got, rp_exc = None, type(e).__name__

        if np_exc is not None or rp_exc is not None:
            if np_exc != rp_exc:
                FAILURES.append((f"getitem {desc}",
                                 f"port raised {rp_exc}, numpy raised {np_exc}"))
            continue

        # numpy returns a scalar where the port still returns a Python number
        # (no scalar types until M3), so compare through np.asarray.
        want_a = np.asarray(want)
        got_a = np.asarray(got) if not isinstance(got, (int, float, complex, bool)) \
            else np.asarray(got, dtype=want_a.dtype)
        try:
            if want_a.shape != got_a.shape:
                raise AssertionError(f"shape {got_a.shape} != numpy's {want_a.shape}")
            if want_a.dtype != got_a.dtype:
                raise AssertionError(f"dtype {got_a.dtype} != numpy's {want_a.dtype}")
            np.testing.assert_array_equal(got_a, want_a, strict=True)
        except Exception as exc:  # noqa: BLE001
            FAILURES.append((f"getitem {desc}",
                             f"{exc}".strip().splitlines()[0][:200]))
            continue

        # View-vs-copy parity, observed through write-back: mutate the result
        # in place in both libraries and compare the *base* arrays.
        if want_a.size and not isinstance(got, (int, float, complex, bool)):
            CHECKS += 1
            try:
                want[...] = 0
                probe_ok = True
            except Exception:  # noqa: BLE001
                probe_ok = False
            if probe_ok:
                try:
                    got[...] = 0
                except Exception as e:  # noqa: BLE001
                    FAILURES.append((f"writeback {desc}",
                                     f"port raised {type(e).__name__} on `r[...] = 0`"))
                    continue
                try:
                    np.testing.assert_array_equal(np.asarray(got_arr), want_arr,
                                                  strict=True)
                except Exception as exc:  # noqa: BLE001
                    FAILURES.append((
                        f"view-vs-copy {desc}",
                        "writing through the result changed the base "
                        f"differently: {exc}".strip().splitlines()[0][:200]))
                    continue

        # ---- setitem parity ------------------------------------------------
        want_arr2 = base.copy()
        got_arr2 = to_port(base)
        value = rng.choice([0, 1, -3, 2.5])
        CHECKS += 1
        try:
            want_arr2[np_key] = value
            np_exc = None
        except Exception as e:  # noqa: BLE001
            np_exc = type(e).__name__
        try:
            got_arr2[rp_key] = value
            rp_exc = None
        except Exception as e:  # noqa: BLE001
            rp_exc = type(e).__name__
        if np_exc != rp_exc:
            FAILURES.append((f"setitem {desc} = {value}",
                             f"port raised {rp_exc}, numpy raised {np_exc}"))
            continue
        if np_exc is None:
            try:
                np.testing.assert_array_equal(np.asarray(got_arr2), want_arr2,
                                              strict=True)
            except Exception as exc:  # noqa: BLE001
                FAILURES.append((f"setitem {desc} = {value}",
                                 f"{exc}".strip().splitlines()[0][:200]))

    # ---- item selection --------------------------------------------------
    for _ in range(60):
        shape = rng.choice([(7,), (5, 4), (3, 4, 5)])
        dt = rng.choice(["int32", "int64", "float64"])
        base = sample(rng, dt, int(np.prod(shape))).reshape(shape)
        port = to_port(base)
        axis = rng.choice([None] + list(range(len(shape))))
        mode = rng.choice(["raise", "wrap", "clip"])
        n = base.size if axis is None else shape[axis]
        idx = np.array([rng.randrange(-2 * n, 2 * n)
                        for _ in range(rng.randrange(1, 5))], dtype=np.intp)
        label = f"take({dt}{shape}, {idx.tolist()}, axis={axis}, mode={mode!r})"
        CHECKS += 1
        try:
            want = base.take(idx, axis=axis, mode=mode)
            np_exc = None
        except Exception as e:  # noqa: BLE001
            want, np_exc = None, type(e).__name__
        try:
            got = port.take(to_port(idx), axis=axis, mode=mode)
            rp_exc = None
        except Exception as e:  # noqa: BLE001
            got, rp_exc = None, type(e).__name__
        if np_exc != rp_exc:
            FAILURES.append((label, f"port raised {rp_exc}, numpy raised {np_exc}"))
        elif np_exc is None:
            check(label, want, got)

    # ---- nonzero / where / flatnonzero -----------------------------------
    for _ in range(40):
        shape = rng.choice([(7,), (5, 4), (3, 4, 5), (0, 3)])
        dt = rng.choice(["int32", "float64", "bool"])
        size = int(np.prod(shape))
        base = sample(rng, dt, max(size, 1))[:size].reshape(shape)
        # Force a decent mix of zeros.
        base = np.where(np.arange(base.size).reshape(shape) % 3 == 0,
                        np.zeros_like(base), base)
        port = to_port(base)
        for k, (w, g) in enumerate(zip(np.nonzero(base), _rnp.nonzero(port))):
            check(f"nonzero({dt}{shape})[{k}]", w, g)
        check(f"flatnonzero({dt}{shape})", np.flatnonzero(base),
              _rnp.flatnonzero(port))
        mask = base != 0
        check(f"where({dt}{shape}, 1, 2)", np.where(mask, 1, 2),
              _rnp.where_(to_port(mask), 1, 2))


# ==========================================================================
# M3: numpy scalar types + ufunc machinery
#
# Everything below talks to the shim (`rnp`) rather than to `_rnp` directly,
# because scalar types, ufunc objects and errstate are implemented in Python
# on top of the Rust engine. Real numpy is always the oracle: every expected
# value is read out of numpy at run time, never written down by hand.
# ==========================================================================

#: Ufuncs (and other entry points) the port has not implemented yet. These are
#: reported separately from FAILURES so the scoreboard stays honest: a missing
#: feature is a "not yet", not a wrong answer.
NOT_YET = []

#: (function, dtype, sample-kind, max ULP) rows produced by check_ulp().
ULP_ROWS = []

#: Concrete scalar types, in numpy's own hierarchy order.
CONCRETE_SCALARS = [
    "bool_", "int8", "int16", "int32", "int64", "longlong",
    "uint8", "uint16", "uint32", "uint64", "ulonglong",
    "float16", "float32", "float64", "longdouble",
    "complex64", "complex128", "clongdouble",
    "bytes_", "str_", "void", "object_", "datetime64", "timedelta64",
]

#: Abstract bases. issubclass() over these is where most of the hierarchy
#: information lives, so they are exercised in the same cross product.
ABSTRACT_SCALARS = [
    "generic", "number", "integer", "signedinteger", "unsignedinteger",
    "inexact", "floating", "complexfloating", "flexible", "character",
]

ALL_SCALARS = CONCRETE_SCALARS + ABSTRACT_SCALARS

#: The numeric scalar types whose *values* are compared in check_scalar_values.
VALUE_SCALARS = [
    "bool_", "int8", "int16", "int32", "int64", "longlong",
    "uint8", "uint16", "uint32", "uint64", "ulonglong",
    "float16", "float32", "float64", "complex64", "complex128",
]

#: Values fed to every scalar constructor. Real numpy decides which of these
#: are legal; where it raises, the port must raise the same exception type.
SCALAR_VALUES = [
    0, 1, -1, 2, 127, 128, 255, 256, -128,
    2 ** 31 - 1, 2 ** 31, 2 ** 63 - 1, 2 ** 63,
    0.5, -0.5, 0.1, 1e300, 1e-300,
    float("inf"), float("-inf"), float("nan"), 1 + 2j,
]

#: The dtypes the ufunc special-value tables are evaluated in.
UFUNC_DTYPES = ["float64", "float32", "float16", "int32", "int64", "uint8",
                "complex128", "bool"]

#: Transcendental ufuncs: the ones for which a NaN result only has to *be* a
#: NaN, not carry the same payload bits. Everything else is compared bit for
#: bit including the NaN payload.
TRANSCENDENTAL = {
    "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "cbrt", "sqrt", "square", "reciprocal", "hypot", "power", "float_power",
    "rint", "logaddexp", "logaddexp2", "spacing", "nextafter", "ldexp",
    "frexp", "modf", "divide", "true_divide", "floor_divide", "remainder",
    "mod", "fmod", "divmod", "absolute", "positive", "negative",
    "deg2rad", "rad2deg", "degrees", "radians", "conjugate", "sign",
    # numpy 2's C99 spellings are aliases of the same objects.
    "acos", "asin", "atan", "atan2", "acosh", "asinh", "atanh", "pow",
}


def eq(name, want, got):
    """One comparison of two directly comparable Python values."""
    global CHECKS
    CHECKS += 1
    if want != got:
        FAILURES.append((name, f"port {got!r} != numpy {want!r}"))
        return False
    return True


def attempt(fn):
    """Run `fn`, returning ("ok", value), ("exc", name) or ("notyet", name)."""
    try:
        return ("ok", fn())
    except NotImplementedError as exc:  # noqa: BLE001
        return ("notyet", f"{type(exc).__name__}: {exc}"[:120])
    except Exception as exc:  # noqa: BLE001
        return ("exc", type(exc).__name__)


def raw(x):
    """(dtype, shape, bytes) for anything either library hands back.

    numpy scalars, port scalars and both libraries' arrays all expose
    `.dtype`/`.tobytes()`, so the *bytes the engine actually produced* are what
    gets compared -- no float() round trip that could hide a payload or a
    low-bit difference.
    """
    if hasattr(x, "tobytes") and hasattr(x, "dtype"):
        return (str(x.dtype), tuple(np.shape(x)), x.tobytes())
    a = np.asarray(x)
    return (str(a.dtype), a.shape, a.tobytes())


def _tobytes(x):
    """Raw bytes of a scalar or array from either library.

    The port's arrays expose the buffer protocol but not `.tobytes()`, and its
    scalars expose `.tobytes()` but not a working `__array__`, so both routes
    are needed.
    """
    if hasattr(x, "tobytes"):
        return x.tobytes()
    return np.asarray(x).tobytes()


def _nan_tolerant_equal(want, got):
    """True when two same-dtype byte-identical-modulo-NaN-payload results agree."""
    w = np.asarray(want)
    g = np.frombuffer(_tobytes(got), dtype=w.dtype).reshape(w.shape)
    if w.dtype.kind not in "fc":
        return bool(np.array_equal(w, g))
    both_nan = np.isnan(w) & np.isnan(g)
    same_bits = (w.tobytes() == g.tobytes())
    if same_bits:
        return True
    # Bits differ: allow it only where both sides are NaN.
    wv = w.view(_int_view(w.dtype)) if w.dtype.kind == "f" else None
    if wv is None:
        # complex: compare real/imag halves separately
        w2 = w.view(np.float64 if w.dtype.itemsize == 16 else np.float32)
        g2 = g.view(w2.dtype)
        ok = (w2.view(_int_view(w2.dtype)) == g2.view(_int_view(w2.dtype)))
        return bool(np.all(ok | (np.isnan(w2) & np.isnan(g2))))
    ok = (wv == g.view(_int_view(g.dtype)))
    return bool(np.all(ok | both_nan))


def _int_view(dt):
    return {2: np.int16, 4: np.int32, 8: np.int64}[np.dtype(dt).itemsize]


def _record_warnings(caught):
    """Warnings in a comparable, order-independent form."""
    return sorted((w.category.__name__, str(w.message)) for w in caught)


# --------------------------------------------------------------------------
# 1. Scalar type identity, MRO and hierarchy
# --------------------------------------------------------------------------

def check_scalar_types():
    """Names, MROs, issubclass cross product, aliases, dtypes, registries."""
    global CHECKS

    present = []
    for name in ALL_SCALARS:
        if not hasattr(np, name):
            continue
        CHECKS += 1
        if not hasattr(rnp, name):
            FAILURES.append((f"scalar type {name}", "missing from the shim"))
            continue
        present.append(name)
        t, s = getattr(np, name), getattr(rnp, name)
        eq(f"{name}.__name__", t.__name__, getattr(s, "__name__", None))
        eq(f"{name}.__mro__",
           tuple(c.__name__ for c in t.__mro__),
           tuple(c.__name__ for c in getattr(s, "__mro__", ())))

    # The full cross product of issubclass. This is where the hierarchy is
    # actually pinned down -- names and MRO order alone would not catch, say,
    # a missing `bool_ is not integer` relation.
    for a in present:
        for b in present:
            eq(f"issubclass({a}, {b})",
               issubclass(getattr(np, a), getattr(np, b)),
               issubclass(getattr(rnp, a), getattr(rnp, b)))

    # Every alias numpy exposes at the top level that is a scalar type.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aliases = []
        for name in dir(np):
            try:
                obj = getattr(np, name)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, type) and issubclass(obj, np.generic):
                aliases.append(name)
    for name in sorted(aliases):
        CHECKS += 1
        got = getattr(rnp, name, None)
        if not isinstance(got, type):
            FAILURES.append((f"alias np.{name}",
                             f"shim has {got!r}, numpy has a scalar type"))
            continue
        eq(f"alias np.{name}.__name__", getattr(np, name).__name__, got.__name__)

    # np.dtype(scalar_type) parity.
    for name in CONCRETE_SCALARS:
        if name not in present:
            continue
        want = np.dtype(getattr(np, name))
        kind, got = attempt(lambda: rnp.dtype(getattr(rnp, name)))
        CHECKS += 1
        if kind != "ok":
            # Recorded as a divergence on purpose: numpy answers, the port
            # does not, and that is visible work remaining.
            FAILURES.append((f"dtype({name})", f"port raised {got}"))
            continue
        for attr in ["num", "char", "kind", "itemsize", "name"]:
            eq(f"dtype({name}).{attr}", getattr(want, attr),
               getattr(got, attr, None))
        eq(f"str(dtype({name}))", str(want), str(got))

    # sctypeDict: same keys, same class per key.
    CHECKS += 1
    want_keys = set(np.sctypeDict)
    got_keys = set(getattr(rnp, "sctypeDict", {}))
    if want_keys != got_keys:
        FAILURES.append((
            "sctypeDict keys",
            f"port has {len(got_keys)} keys, numpy {len(want_keys)}; "
            f"extra={sorted(got_keys - want_keys)[:12]} "
            f"missing={sorted(want_keys - got_keys)[:12]}"))
    for key in sorted(want_keys & got_keys, key=str):
        eq(f"sctypeDict[{key!r}]", np.sctypeDict[key].__name__,
           getattr(rnp.sctypeDict[key], "__name__", None))

    eq("typecodes", dict(np.typecodes), dict(getattr(rnp, "typecodes", {})))

    want_st = [t.__name__ for t in np.ScalarType]
    got_st = [getattr(t, "__name__", None) for t in getattr(rnp, "ScalarType", ())]
    eq("len(ScalarType)", len(want_st), len(got_st))
    eq("ScalarType", want_st, got_st)


# --------------------------------------------------------------------------
# 2. Scalar value parity
# --------------------------------------------------------------------------

def check_scalar_values():
    """repr/str/hash/item/dtype/... for every (type, value) numpy accepts."""
    global CHECKS

    for name in VALUE_SCALARS:
        if not hasattr(np, name) or not hasattr(rnp, name):
            continue
        nt, st = getattr(np, name), getattr(rnp, name)
        for v in SCALAR_VALUES:
            label = f"{name}({v!r})"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"):
                    wkind, want = attempt(lambda: nt(v))
                    gkind, got = attempt(lambda: st(v))
            CHECKS += 1
            if wkind != "ok":
                # numpy refuses the value: the port must refuse it identically.
                if gkind == "notyet":
                    NOT_YET.append((label, got))
                elif gkind != "exc" or want != got:
                    FAILURES.append((label,
                                     f"port {gkind}:{got!r}, numpy raised {want}"))
                continue
            if gkind == "notyet":
                NOT_YET.append((label, got))
                continue
            if gkind != "ok":
                FAILURES.append((label, f"port raised {got}, numpy returned "
                                        f"{want!r}"))
                continue

            eq(f"repr({label})", repr(want), repr(got))
            eq(f"str({label})", str(want), str(got))
            # CPython hashes NaN by object identity (3.10+), so a NaN hash is
            # address-dependent and says nothing about the port.
            if not _is_nan(want):
                eq(f"hash({label})", _safe(lambda: hash(want)),
                   _safe(lambda: hash(got)))
            # Compared through repr so NaN == NaN and -0.0 != 0.0.
            eq(f"{label}.item()", _safe(lambda: repr(want.item())),
               _safe(lambda: repr(got.item())))
            eq(f"type({label}.item())",
               _safe(lambda: type(want.item()).__name__),
               _safe(lambda: type(got.item()).__name__))
            eq(f"{label}.dtype", str(want.dtype), str(getattr(got, "dtype", None)))
            eq(f"{label}.itemsize", want.itemsize, getattr(got, "itemsize", None))
            eq(f"{label}.ndim", want.ndim, getattr(got, "ndim", None))
            eq(f"{label}.shape", want.shape, getattr(got, "shape", None))
            eq(f"bool({label})", _safe(lambda: bool(want)), _safe(lambda: bool(got)))
            eq(f"type({label}[()])", _safe(lambda: type(want[()]).__name__),
               _safe(lambda: type(got[()]).__name__))
            # The stored bits themselves.
            eq(f"bits({label})", raw(want), _safe(lambda: raw(got)))


def _safe(fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}>"


def _is_nan(x):
    try:
        return bool(np.isnan(x))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# 3. 0-d and element extraction rules
# --------------------------------------------------------------------------

def check_element_extraction():
    """What *type* comes out of indexing, iteration and whole-array reductions."""
    global CHECKS

    dtypes = ["bool", "int8", "int32", "int64", "uint8", "uint32", "uint64",
              "float16", "float32", "float64", "complex64", "complex128"]
    rng = random.Random(4242)
    for dt in dtypes:
        base2 = sample(rng, dt, 12).reshape(3, 4)
        cases = [
            ("0d", np.array(base2[0, 0]), to_port(base2.ravel()[:1]).reshape(())),
            ("1d", base2[0].copy(), to_port(base2[0].copy())),
            ("2d", base2, to_port(base2.ravel()).reshape(3, 4)),
        ]
        for shape_name, w, g in cases:
            probes = [
                ("a[0]", lambda a: type(a[0]).__name__),
                ("a[0,0]", lambda a: type(a[0, 0]).__name__),
                ("a[()]", lambda a: type(a[()]).__name__),
                ("a.item()", lambda a: type(a.item()).__name__),
                ("a.sum()", lambda a: type(a.sum()).__name__),
                ("a.max()", lambda a: type(a.max()).__name__),
                ("a.all()", lambda a: type(a.all()).__name__),
                ("a.argmax()", lambda a: type(a.argmax()).__name__),
                ("iter(a)", lambda a: type(next(iter(a))).__name__),
            ]
            for pname, fn in probes:
                label = f"type of {pname} on {dt} {shape_name}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with np.errstate(all="ignore"):
                        wk, wv = attempt(lambda: fn(w))
                        gk, gv = attempt(lambda: fn(g))
                CHECKS += 1
                if gk == "notyet":
                    NOT_YET.append((label, gv))
                    continue
                if (wk, wv) != (gk, gv):
                    FAILURES.append((label, f"port {gk}:{gv!r}, numpy {wk}:{wv!r}"))


# --------------------------------------------------------------------------
# 4. ufunc special-value tables
# --------------------------------------------------------------------------

def _specials(dt):
    """The special-value vector for a dtype, built once in numpy.

    Both libraries are then handed the *same* cast result, so a divergence in
    the float->int cast cannot be mistaken for a divergence in the ufunc.
    """
    d = np.dtype(dt)
    if d.kind == "f":
        fi = np.finfo(d)
        tiny, huge = float(fi.tiny), float(fi.max)
    elif d.kind == "c":
        fi = np.finfo(np.dtype(f"f{d.itemsize // 2}"))
        tiny, huge = float(fi.tiny), float(fi.max)
    elif d.kind in "iu":
        ii = np.iinfo(d)
        tiny, huge = 1.0, float(min(ii.max, 2 ** 53))
    else:
        tiny, huge = 1.0, 1.0
    base = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0,
            1.0, -1.0, tiny, huge, 2.0, -2.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            return np.array(base, dtype=np.float64).astype(d)


def check_ufunc_tables():
    """Every ufunc, every dtype, the full special-value cross product."""
    global CHECKS

    names = sorted(n for n in dir(np) if isinstance(getattr(np, n, None), np.ufunc))
    for name in names:
        nf = getattr(np, name)
        gf = getattr(rnp, name, None)
        CHECKS += 1
        if gf is None:
            FAILURES.append((f"ufunc {name}", "missing from the shim"))
            continue
        for dt in UFUNC_DTYPES:
            vals = _specials(dt)
            pvals = to_port(vals)
            label = f"{name}({dt})"
            if nf.nin == 1:
                np_args = (vals,)
                rp_args = (pvals,)
            else:
                # The full 11x11 cross product in one broadcast call.
                np_args = (vals[:, None], vals[None, :])
                rp_args = (pvals[:, None], pvals[None, :])
                if nf.nin > 2:
                    np_args = np_args + (vals[None, :],)
                    rp_args = rp_args + (pvals[None, :],)

            with warnings.catch_warnings(record=True) as wlog:
                warnings.simplefilter("always")
                wkind, want = attempt(lambda: nf(*np_args))
            want_warn = _record_warnings(wlog)

            with warnings.catch_warnings(record=True) as glog:
                warnings.simplefilter("always")
                gkind, got = attempt(lambda: gf(*rp_args))
            got_warn = _record_warnings(glog)

            if gkind == "notyet":
                NOT_YET.append((f"ufunc {name}", got))
                break  # the whole ufunc is missing; no point going per-dtype

            CHECKS += 1
            if wkind != gkind:
                FAILURES.append((label,
                                 f"port {gkind}:{got!r}, numpy {wkind}:{want!r}"))
                continue
            if wkind == "exc":
                eq(f"{label} raises", want, got)
                continue

            wouts = want if nf.nout > 1 else (want,)
            gouts = tuple(got) if nf.nout > 1 else (got,)
            if len(wouts) != len(gouts):
                FAILURES.append((label, f"port returned {len(gouts)} outputs, "
                                        f"numpy {len(wouts)}"))
                continue
            for k, (w, g) in enumerate(zip(wouts, gouts)):
                suffix = f"[out{k}]" if nf.nout > 1 else ""
                wr, gr = raw(w), _safe(lambda: raw(g))
                if not isinstance(gr, tuple):
                    FAILURES.append((label + suffix, f"result unreadable: {gr}"))
                    continue
                if wr[:2] != gr[:2]:
                    FAILURES.append((
                        label + suffix,
                        f"dtype/shape {gr[:2]} != numpy's {wr[:2]}"))
                    continue
                if wr[2] == gr[2]:
                    continue
                nan_ok = name in TRANSCENDENTAL
                if not nan_ok or not _nan_tolerant_equal(w, g):
                    FAILURES.append((label + suffix,
                                     _bit_diff(w, g, vals, nf.nin, nan_ok)))

            # Emitted warnings must match exactly, message text included.
            CHECKS += 1
            if want_warn != got_warn:
                FAILURES.append((f"{label} warnings",
                                 f"port {got_warn} != numpy {want_warn}"))


def _bit_diff(want, got, vals, nin, nan_ok=False):
    """Point at the first differing element, with the inputs that produced it.

    With `nan_ok`, positions where both sides are NaN are skipped, so the
    message points at a real numeric divergence rather than a payload bit.
    """
    w = np.asarray(want)
    g = np.frombuffer(_tobytes(got), dtype=w.dtype).reshape(w.shape)
    flat_w, flat_g = w.ravel(), g.ravel()
    both_nan = (np.isnan(flat_w) & np.isnan(flat_g)) if w.dtype.kind in "fc" \
        else np.zeros(flat_w.shape, bool)
    for i in range(flat_w.size):
        if nan_ok and both_nan[i]:
            continue
        if flat_w[i:i + 1].tobytes() != flat_g[i:i + 1].tobytes():
            idx = np.unravel_index(i, w.shape)
            if nin == 1:
                args = f"x={vals[idx[0]]!r}"
            else:
                args = f"x={vals[idx[0]]!r}, y={vals[idx[1]]!r}"
            return (f"{args}: port {flat_g[i]!r} "
                    f"(0x{flat_g[i:i + 1].tobytes().hex()}) != numpy "
                    f"{flat_w[i]!r} (0x{flat_w[i:i + 1].tobytes().hex()})")
    return "bytes differ"


# --------------------------------------------------------------------------
# 5. ULP measurement for the transcendentals
# --------------------------------------------------------------------------

#: (function, (lo, hi) domain per argument). Binary functions get two ranges.
ULP_FUNCS = [
    ("exp", [(-700.0, 700.0)]),
    ("exp2", [(-1000.0, 1000.0)]),
    ("expm1", [(-30.0, 30.0)]),
    ("log", [(1e-300, 1e300)]),
    ("log2", [(1e-300, 1e300)]),
    ("log10", [(1e-300, 1e300)]),
    ("log1p", [(-0.999, 1e6)]),
    ("sin", [(-100.0, 100.0)]),
    ("cos", [(-100.0, 100.0)]),
    ("tan", [(-100.0, 100.0)]),
    ("arcsin", [(-1.0, 1.0)]),
    ("arccos", [(-1.0, 1.0)]),
    ("arctan", [(-1e6, 1e6)]),
    ("sinh", [(-700.0, 700.0)]),
    ("cosh", [(-700.0, 700.0)]),
    ("tanh", [(-20.0, 20.0)]),
    ("arcsinh", [(-1e6, 1e6)]),
    ("arccosh", [(1.0, 1e6)]),
    ("arctanh", [(-1.0, 1.0)]),
    ("cbrt", [(-1e6, 1e6)]),
    ("sqrt", [(0.0, 1e300)]),
    ("hypot", [(-1e150, 1e150), (-1e150, 1e150)]),
    ("arctan2", [(-1e6, 1e6), (-1e6, 1e6)]),
    ("power", [(0.0, 100.0), (-20.0, 20.0)]),
    ("float_power", [(0.0, 100.0), (-20.0, 20.0)]),
]


def _ordered_bits(a):
    """Map IEEE floats to a monotonically increasing unsigned integer.

    Adjacent representable floats map to adjacent integers, so |difference| is
    the ULP distance. -0.0 and +0.0 map to the same integer, as they should.
    """
    n = a.dtype.itemsize * 8
    ut = {16: np.uint16, 32: np.uint32, 64: np.uint64}[n]
    u = a.view(ut).astype(np.uint64)
    sign = np.uint64(1) << np.uint64(n - 1)
    mag = u & (sign - np.uint64(1))
    neg = (u & sign) != 0
    return np.where(neg, sign - mag, sign + mag)


def _ulp_distance(want, got):
    """Elementwise ULP distance, with NaN/inf mismatches flagged separately."""
    w = np.asarray(want)
    g = np.frombuffer(_tobytes(got), dtype=w.dtype).reshape(w.shape)
    nan_w, nan_g = np.isnan(w), np.isnan(g)
    mismatched_nan = nan_w != nan_g
    ow, og = _ordered_bits(w), _ordered_bits(g)
    d = np.where(ow > og, ow - og, og - ow)
    d = np.where(nan_w & nan_g, np.uint64(0), d)
    return d, mismatched_nan, w, g


def _edge_values(lo, hi, dtype):
    """A dense edge-case vector inside [lo, hi] for a dtype."""
    fi = np.finfo(dtype)
    lo = max(lo, -float(fi.max))
    hi = min(hi, float(fi.max))
    cand = [lo, hi, 0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0,
            float(fi.tiny), -float(fi.tiny), float(fi.eps), -float(fi.eps),
            1.0 + float(fi.eps), 1.0 - float(fi.eps),
            math.pi, -math.pi, math.pi / 2, -math.pi / 2, math.pi / 4,
            2 * math.pi, 3 * math.pi / 2, math.e, 1 / math.e,
            1e-6, 1e-3, 10.0, 100.0, 1e3, 1e6]
    # A dense sweep of the domain, plus the two representable neighbours of
    # each endpoint (where the last bit of the implementation usually shows).
    for k in range(257):
        cand.append(lo + (hi - lo) * k / 256.0)
    for k in range(-30, 31):
        cand.append(math.ldexp(1.0, k))
        cand.append(-math.ldexp(1.0, k))
    vals = sorted({float(v) for v in cand
                   if lo <= v <= hi and math.isfinite(v)})
    a = np.array(vals, dtype=dtype)
    extra = np.array([np.nextafter(a.dtype.type(lo), a.dtype.type(hi)),
                      np.nextafter(a.dtype.type(hi), a.dtype.type(lo))],
                     dtype=dtype)
    return np.unique(np.concatenate([a, extra]))


def check_ulp(n_random=200_000):
    """Max ULP difference vs real numpy over random and edge-case inputs."""
    global CHECKS

    for dtype_name in ["float64", "float32"]:
        dt = np.dtype(dtype_name)
        rng = np.random.default_rng(20260816)
        for fname, domains in ULP_FUNCS:
            nf, gf = getattr(np, fname, None), getattr(rnp, fname, None)
            if nf is None or gf is None:
                continue
            worst = 0
            worst_at = None
            bad_nan = 0
            skipped = False
            for kind in ("random", "edge"):
                if kind == "random":
                    args = []
                    # Clamp to the dtype's own range so float32 sampling does
                    # not simply saturate to inf and stop measuring anything.
                    fmax = float(np.finfo(dt).max)
                    for lo, hi in domains:
                        lo, hi = max(lo, -fmax), min(hi, fmax)
                        # Log-uniform where the domain spans many decades, so
                        # the small end is not swamped by the large end.
                        if hi > 0 and lo >= 0 and hi / max(lo, 1e-320) > 1e6:
                            u = rng.uniform(math.log(max(lo, 1e-300)),
                                            math.log(hi), n_random)
                            v = np.exp(u)
                        elif lo < 0 < hi and hi > 1e6:
                            u = rng.uniform(0.0, math.log(hi), n_random)
                            v = np.exp(u) * rng.choice([-1.0, 1.0], n_random)
                        else:
                            v = rng.uniform(lo, hi, n_random)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            with np.errstate(all="ignore"):
                                args.append(v.astype(dt))
                else:
                    edges = [_edge_values(lo, hi, dt) for lo, hi in domains]
                    if len(edges) == 1:
                        args = edges
                    else:
                        # Cross product, trimmed so the table stays small.
                        a = edges[0][::max(1, len(edges[0]) // 64)]
                        b = edges[1][::max(1, len(edges[1]) // 64)]
                        args = [np.repeat(a, len(b)),
                                np.tile(b, len(a))]
                port_args = [to_port(a) for a in args]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with np.errstate(all="ignore"):
                        want = nf(*args)
                        gkind, got = attempt(lambda: gf(*port_args))
                if gkind == "notyet":
                    NOT_YET.append((f"ulp {fname}", got))
                    skipped = True
                    break
                if gkind != "ok":
                    FAILURES.append((f"ulp {fname} {dtype_name} ({kind})",
                                     f"port raised {got}"))
                    skipped = True
                    break
                d, mismatched_nan, w, g = _ulp_distance(want, got)
                bad_nan += int(mismatched_nan.sum())
                if mismatched_nan.any():
                    i = int(np.argmax(mismatched_nan))
                    FAILURES.append((
                        f"ulp {fname} {dtype_name} ({kind}) NaN",
                        f"x={args[0].ravel()[i]!r}: port {g.ravel()[i]!r} != "
                        f"numpy {w.ravel()[i]!r}"))
                if d.size:
                    i = int(np.argmax(d))
                    if int(d.ravel()[i]) > worst:
                        worst = int(d.ravel()[i])
                        worst_at = (kind, [float(a.ravel()[i]) for a in args],
                                    float(g.ravel()[i]), float(w.ravel()[i]))
            if skipped:
                continue
            ULP_ROWS.append((fname, dtype_name, worst, worst_at))
            CHECKS += 1
            if worst > 4:
                at = ""
                if worst_at:
                    at = (f" at {worst_at[0]} x={worst_at[1]}: port "
                          f"{worst_at[2]!r} != numpy {worst_at[3]!r}")
                FAILURES.append((f"ulp {fname} {dtype_name}",
                                 f"max ULP {worst} > 4{at}"))


# --------------------------------------------------------------------------
# 6. ufunc method parity
# --------------------------------------------------------------------------

METHOD_UFUNCS = ["add", "multiply", "minimum", "maximum",
                 "logical_and", "logical_or", "bitwise_and", "bitwise_or"]


def check_ufunc_methods():
    """reduce/accumulate/outer/reduceat, plus every ufunc's introspection."""
    global CHECKS

    # ---- introspection attributes, for every ufunc numpy has ------------
    for name in sorted(n for n in dir(np)
                       if isinstance(getattr(np, n, None), np.ufunc)):
        nf = getattr(np, name)
        gf = getattr(rnp, name, None)
        if gf is None:
            continue
        # numpy's C99 spellings (acos, pow, ...) are aliases of the same ufunc
        # object, so `np.acos.__name__` is "arccos"; the port must agree.
        eq(f"np.{name}.__name__", nf.__name__, getattr(gf, "__name__", "<missing>"))
        for attr in ["nin", "nout", "nargs", "identity", "ntypes"]:
            eq(f"{name}.{attr}", getattr(nf, attr), getattr(gf, attr, "<missing>"))
        eq(f"{name}.types", list(nf.types), list(getattr(gf, "types", [])))

    rng = random.Random(97531)
    shapes = [(12,), (4, 5), (3, 4, 2), (1, 7), (6, 1)]
    dtypes = ["int32", "int64", "float64", "float32", "bool"]

    for name in METHOD_UFUNCS:
        nf, gf = getattr(np, name), getattr(rnp, name, None)
        CHECKS += 1
        if gf is None:
            FAILURES.append((f"ufunc {name}", "missing from the shim"))
            continue
        for dt in dtypes:
            if name.startswith("bitwise") and dt.startswith("float"):
                continue  # numpy has no float loop; covered by the table above
            for shape in shapes:
                base = sample(rng, dt, int(np.prod(shape))).reshape(shape)
                port = to_port(base.ravel()).reshape(shape)
                other = sample(rng, dt, 5)
                p_other = to_port(other)

                axes = [None] + list(range(len(shape)))
                for axis in axes:
                    for keepdims in (False, True):
                        kws = [{}]
                        if name in ("add", "multiply", "minimum", "maximum") \
                                and dt != "bool":
                            kws.append({"initial": 1})
                        for extra in kws:
                            kw = dict(axis=axis, keepdims=keepdims, **extra)
                            _cmp_method(f"{name}.reduce({dt}{shape}, {kw})",
                                        lambda k=kw: nf.reduce(base, **k),
                                        lambda k=kw: gf.reduce(port, **k))
                    # `where` needs an identity or an explicit initial.
                    mask = np.array(
                        [rng.random() < 0.6 for _ in range(base.size)]
                    ).reshape(shape)
                    kw = dict(axis=axis, where=mask)
                    if nf.identity is None:
                        kw["initial"] = 0 if dt != "bool" else False
                    _cmp_method(
                        f"{name}.reduce({dt}{shape}, axis={axis}, where=...)",
                        lambda k=kw, m=mask: nf.reduce(base, **k),
                        lambda k=kw, m=mask: gf.reduce(
                            port, **{**k, "where": to_port(m)}))

                for axis in range(len(shape)):
                    _cmp_method(f"{name}.accumulate({dt}{shape}, axis={axis})",
                                lambda a=axis: nf.accumulate(base, axis=a),
                                lambda a=axis: gf.accumulate(port, axis=a))

                flat = base.ravel()
                p_flat = to_port(flat)
                _cmp_method(f"{name}.outer({dt}{shape})",
                            lambda: nf.outer(flat, other),
                            lambda: gf.outer(p_flat, p_other))

                n = shape[0]
                idx = np.array(sorted({rng.randrange(0, n) for _ in range(3)}),
                               dtype=np.intp)
                _cmp_method(f"{name}.reduceat({dt}{shape}, {idx.tolist()})",
                            lambda i=idx: nf.reduceat(base, i),
                            lambda i=idx: gf.reduceat(port, to_port(i)))


def _cmp_method(label, want_fn, got_fn):
    """Run a ufunc method on both sides and compare value, dtype and shape."""
    global CHECKS
    CHECKS += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            wkind, want = attempt(want_fn)
            gkind, got = attempt(got_fn)
    if gkind == "notyet":
        NOT_YET.append((label, got))
        return
    if wkind != gkind:
        FAILURES.append((label, f"port {gkind}:{got!r}, numpy {wkind}:{want!r}"))
        return
    if wkind == "exc":
        eq(f"{label} raises", want, got)
        return
    wr, gr = raw(want), _safe(lambda: raw(got))
    if not isinstance(gr, tuple):
        FAILURES.append((label, f"result unreadable: {gr}"))
        return
    if wr[:2] != gr[:2]:
        FAILURES.append((label, f"dtype/shape {gr[:2]} != numpy's {wr[:2]}"))
        return
    if wr[2] != gr[2] and not _nan_tolerant_equal(want, got):
        # Decode the port's bytes through numpy so the message is readable even
        # when the port object itself does not support the array protocol.
        gv = np.frombuffer(gr[2], dtype=wr[0]).reshape(wr[1])
        FAILURES.append((label, f"values differ: port {gv.ravel()[:6]} "
                                f"vs numpy {np.asarray(want).ravel()[:6]}"))


# --------------------------------------------------------------------------
# 7. errstate parity
# --------------------------------------------------------------------------

#: (label, category, expression). Each expression is built separately for each
#: library so no array is ever shared between them.
ERRSTATE_CASES = [
    ("1/0 float64", "divide",
     lambda m, mk: m.divide(mk([1.0]), mk([0.0]))),
    ("-1/0 float64", "divide",
     lambda m, mk: m.divide(mk([-1.0]), mk([0.0]))),
    ("1//0 int64", "divide",
     lambda m, mk: m.floor_divide(mk([1], "int64"), mk([0], "int64"))),
    ("1%0 int64", "divide",
     lambda m, mk: m.remainder(mk([1], "int64"), mk([0], "int64"))),
    ("log(0)", "divide", lambda m, mk: m.log(mk([0.0]))),
    ("exp(1000)", "over", lambda m, mk: m.exp(mk([1000.0]))),
    ("3e38*3e38 f32", "over",
     lambda m, mk: m.multiply(mk([3e38], "float32"), mk([3e38], "float32"))),
    ("1e308+1e308", "over", lambda m, mk: m.add(mk([1e308]), mk([1e308]))),
    ("sqrt(-1)", "invalid", lambda m, mk: m.sqrt(mk([-1.0]))),
    ("0/0", "invalid", lambda m, mk: m.divide(mk([0.0]), mk([0.0]))),
    ("inf-inf", "invalid",
     lambda m, mk: m.subtract(mk([float("inf")]), mk([float("inf")]))),
    ("log(-1)", "invalid", lambda m, mk: m.log(mk([-1.0]))),
    ("arcsin(2)", "invalid", lambda m, mk: m.arcsin(mk([2.0]))),
    ("inf*0", "invalid",
     lambda m, mk: m.multiply(mk([float("inf")]), mk([0.0]))),
    ("1e-320*1e-10", "under",
     lambda m, mk: m.multiply(mk([1e-320]), mk([1e-10]))),
]


def check_errstate():
    """Warn / raise / ignore behaviour and the exact message text."""
    global CHECKS

    def run(mod, mk, expr, category, setting):
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter("always")
            try:
                with mod.errstate(**{category: setting}):
                    value = expr(mod, mk)
                outcome = ("value", raw(value))
            except NotImplementedError as exc:  # noqa: BLE001
                return ("notyet", f"{exc}"[:120]), []
            except Exception as exc:  # noqa: BLE001
                outcome = ("raised", type(exc).__name__, str(exc))
        return outcome, _record_warnings(log)

    def np_mk(values, dt="float64"):
        return np.array(values, dtype=dt)

    def rp_mk(values, dt="float64"):
        return to_port(np.array(values, dtype=dt))

    for label, category, expr in ERRSTATE_CASES:
        for setting in ["warn", "raise", "ignore"]:
            name = f"errstate({category}={setting!r}) {label}"
            want, want_warn = run(np, np_mk, expr, category, setting)
            got, got_warn = run(rnp, rp_mk, expr, category, setting)
            CHECKS += 1
            if got[0] == "notyet":
                NOT_YET.append((name, got[1]))
                continue
            if want[0] != got[0]:
                FAILURES.append((name, f"port {got[0]}, numpy {want[0]}"))
                continue
            if want[0] == "raised":
                eq(f"{name} exception", want[1:], got[1:])
            else:
                eq(f"{name} result", want[1], got[1])
            eq(f"{name} warnings", want_warn, got_warn)

    # The aggregate settings (all=..., seterr/geterr round trip) too.
    for setting in ["warn", "raise", "ignore", "print"]:
        CHECKS += 1
        try:
            with np.errstate(all=setting):
                want = dict(np.geterr())
        except Exception as exc:  # noqa: BLE001
            want = type(exc).__name__
        try:
            with rnp.errstate(all=setting):
                got = dict(rnp.geterr())
        except Exception as exc:  # noqa: BLE001
            got = type(exc).__name__
        if want != got:
            FAILURES.append((f"errstate(all={setting!r}) geterr",
                             f"port {got!r} != numpy {want!r}"))


def run_m3_sections():
    """All the M3 sections, each isolated so one crash cannot hide the rest."""
    if not load_shim():
        FAILURES.append(("M3 shim import", SHIM_IMPORT_ERROR or "unavailable"))
        return
    sections = [
        ("scalar types", check_scalar_types),
        ("scalar values", check_scalar_values),
        ("element extraction", check_element_extraction),
        ("ufunc tables", check_ufunc_tables),
        ("ufunc methods", check_ufunc_methods),
        ("errstate", check_errstate),
        ("ulp", check_ulp),
    ]
    for name, fn in sections:
        try:
            fn()
        except Exception:  # noqa: BLE001
            FAILURES.append((f"M3 section {name}",
                             "crashed: " + traceback.format_exc()
                             .strip().splitlines()[-1]))


def print_m3_report():
    """The ULP table and the not-yet-implemented list, ready for PLAN.md."""
    if ULP_ROWS:
        print()
        print("ULP table (port vs numpy 2.5.2; max over 200k random + dense edges)")
        print("| function    | dtype   |        max_ulp |")
        print("|-------------|---------|----------------|")
        for fname, dtype_name, worst, _at in ULP_ROWS:
            print(f"| {fname:<11} | {dtype_name:<7} | {worst:>14} |")
        worst_rows = [r for r in ULP_ROWS if r[2] > 4]
        if worst_rows:
            print()
            print("ULP outliers (> 4 ULP):")
            for fname, dtype_name, worst, at in worst_rows:
                loc = "" if at is None else (f" worst at {at[0]} x={at[1]}: "
                                             f"port {at[2]!r} vs numpy {at[3]!r}")
                print(f"  {fname} {dtype_name}: {worst} ULP{loc}")
    if NOT_YET:
        print()
        seen = {}
        for label, reason in NOT_YET:
            seen.setdefault(label, reason)
        print(f"not-yet-implemented in the port ({len(seen)}):")
        for label in sorted(seen):
            print(f"  SKIP {label}: {seen[label]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- creation ------------------------------------------------------
    for dt in DTYPES:
        for shape in [(5,), (2, 3), (3, 1, 2), (0,), ()]:
            check(f"zeros({shape}, {dt})", np.zeros(shape, dt), _rnp.zeros(shape, dt))
            check(f"ones({shape}, {dt})", np.ones(shape, dt), _rnp.ones(shape, dt))

    # ---- arange --------------------------------------------------------
    for _ in range(15):
        start = rng.randrange(-20, 20)
        stop = start + rng.randrange(1, 40)
        step = rng.choice([1, 2, 3, 7])
        for dt in ["int8", "int32", "int64", "uint16", "float32", "float64"]:
            if dt.startswith("uint") and start < 0:
                continue
            check(
                f"arange({start},{stop},{step},{dt})",
                np.arange(start, stop, step, dtype=dt),
                _rnp.arange(start, stop, step, dt),
            )
    for _ in range(5):
        a, b, s = rng.uniform(-5, 5), rng.uniform(6, 20), rng.uniform(0.1, 1.5)
        check(f"arange float {a},{b},{s}", np.arange(a, b, s), _rnp.arange(a, b, s))

    # ---- astype: all 13x13 pairs ---------------------------------------
    for src, dst in itertools.product(DTYPES, DTYPES):
        data = sample(rng, src, 8)
        check(
            f"astype {src}->{dst}",
            data.astype(dst),
            to_port(data).astype(dst),
        )

    # ---- binary ops with promotion and broadcasting ---------------------
    ops = [
        ("add", np.add, _rnp.add),
        ("subtract", np.subtract, _rnp.subtract),
        ("multiply", np.multiply, _rnp.multiply),
        ("divide", np.divide, _rnp.divide),
        ("equal", np.equal, _rnp.equal),
        ("less", np.less, _rnp.less),
    ]
    for _ in range(40):
        da, db = rng.choice(DTYPES), rng.choice(DTYPES)
        name, np_op, port_op = rng.choice(ops)
        if name == "subtract" and da == "bool" and db == "bool":
            continue  # numpy raises TypeError; checked separately below
        a = sample(rng, da, 6)
        b = sample(rng, db, 6)
        with np.errstate(all="ignore"):
            want = np_op(a, b)
        check(f"{name}({da},{db})", want, port_op(to_port(a), to_port(b)))

    # Broadcasting shapes.
    for sa, sb in [((3, 1), (1, 4)), ((5, 4), (4,)), ((2, 3, 4), (3, 4)),
                   ((2, 1, 4), (1, 3, 1)), ((1,), (6,))]:
        a = np.arange(int(np.prod(sa)), dtype="int32").reshape(sa)
        b = np.arange(int(np.prod(sb)), dtype="float32").reshape(sb)
        pa = to_port(a.ravel()).reshape(sa)
        pb = to_port(b.ravel()).reshape(sb)
        with np.errstate(all="ignore"):
            check(f"broadcast add {sa}+{sb}", np.add(a, b), _rnp.add(pa, pb))
            check(f"broadcast mul {sa}*{sb}", np.multiply(a, b), _rnp.multiply(pa, pb))

    # ---- strided views --------------------------------------------------
    base = np.arange(24, dtype="int64").reshape(4, 6)
    pbase = to_port(base.ravel()).reshape(4, 6)
    views = [
        ("[::2]", lambda x: x[::2]),
        ("[:, ::3]", lambda x: x[:, ::3]),
        ("[1:3, 1:5]", lambda x: x[1:3, 1:5]),
        ("[::-1]", lambda x: x[::-1]),
        ("[:, ::-2]", lambda x: x[:, ::-2]),
        (".T", lambda x: x.T),
        ("[2]", lambda x: x[2]),
        ("[..., 1]", lambda x: x[..., 1]),
        ("[None]", lambda x: x[None]),
    ]
    for label, fn in views:
        want, got = fn(base), fn(pbase)
        check(f"view {label}", want, got)
        check(f"view {label} .copy()", want.copy(), got.copy())
        check(f"view {label} +1", want + 1, got + 1)
        check(f"view {label} astype f32", want.astype("float32"),
              got.astype("float32"))
        if tuple(got.strides) != want.strides:
            FAILURES.append((f"view {label} strides",
                             f"{tuple(got.strides)} != numpy's {want.strides}"))

    # ---- NEP 50 weak scalars -------------------------------------------
    for dt in DTYPES:
        a = sample(rng, dt, 4)
        pa = to_port(a)
        for scalar in [2, 2.5, 1 + 1j, True]:
            if dt == "bool" and scalar is True:
                pass
            with np.errstate(all="ignore"):
                try:
                    want = a + scalar
                except TypeError:
                    continue
            check(f"{dt} + {scalar!r}", want, pa + scalar)

    # ---- repr -----------------------------------------------------------
    repr_cases = [
        np.zeros((2, 3)),
        np.arange(6, dtype="int32").reshape(2, 3),
        np.array([True, False]),
        np.array([0.0, 0.5]),
        np.arange(3.0),
        np.array([1 + 2j]),
        np.arange(2000),
        np.array([], dtype="float64"),
        np.arange(24, dtype="int64").reshape(4, 6),
    ]
    for want in repr_cases:
        global CHECKS
        CHECKS += 1
        port = to_port(want.ravel()).reshape(want.shape) if want.size else \
            _rnp.zeros(want.shape, str(want.dtype))
        if repr(port) != repr(want):
            FAILURES.append(
                (f"repr {want.dtype}{want.shape}",
                 f"{repr(port)!r} != numpy's {repr(want)!r}"))

    # ---- dtype constructor round-trips ----------------------------------
    for spec in DTYPE_SPECS:
        CHECKS += 1
        want = _bool_or_exc(lambda: np.dtype(spec))
        got = _bool_or_exc(lambda: _rnp.dtype(spec))
        if isinstance(want, str) or isinstance(got, str):
            if want != got:
                FAILURES.append((f"dtype({spec!r})",
                                 f"port {got!r} != numpy {want!r}"))
            continue
        _compare_dtype(f"dtype({spec!r})", want, got)

    for spec in DTYPE_OBJECTS:
        CHECKS += 1
        port_spec = _port_spec(spec)
        try:
            want, got = np.dtype(spec), _rnp.dtype(port_spec)
        except Exception as exc:  # noqa: BLE001
            FAILURES.append((f"dtype({spec!r})", f"construction raised {exc!r}"))
            continue
        _compare_dtype(f"dtype({spec!r})", want, got)
        # align=True changes the layout, so check it separately.
        if isinstance(spec, (list, dict)) and spec:
            CHECKS += 1
            try:
                _compare_dtype(f"dtype({spec!r}, align=True)",
                               np.dtype(spec, align=True),
                               _rnp.dtype(port_spec, align=True))
            except Exception as exc:  # noqa: BLE001
                FAILURES.append((f"dtype({spec!r}, align=True)", repr(exc)))

    # newbyteorder round-trips
    for spec in ["i4", "f8", "u2", ">i4", "<f4", "b", "S5", "U3",
                 "i4,f8"]:
        for order in [None, "S", "<", ">", "=", "|"]:
            CHECKS += 1
            args = () if order is None else (order,)
            try:
                want = np.dtype(spec).newbyteorder(*args)
                got = _rnp.dtype(spec).newbyteorder(*args)
            except Exception as exc:  # noqa: BLE001
                FAILURES.append((f"{spec!r}.newbyteorder({order!r})", repr(exc)))
                continue
            if repr(want) != repr(got):
                FAILURES.append((f"{spec!r}.newbyteorder({order!r})",
                                 f"{got!r} != numpy's {want!r}"))

    # dtype hashing and equality
    equiv_groups = [["i4", "<i4", "=i4", "int32"], ["f8", "float64", "<f8"],
                    ["S5", "|S5"], ["U3", "<U3"], ["i4,f8"], ["?", "bool", "b1"]]
    for group in equiv_groups:
        for a in group:
            for b in group:
                CHECKS += 1
                want_eq = np.dtype(a) == np.dtype(b)
                got_eq = _rnp.dtype(a) == _rnp.dtype(b)
                want_h = hash(np.dtype(a)) == hash(np.dtype(b))
                got_h = hash(_rnp.dtype(a)) == hash(_rnp.dtype(b))
                if (want_eq, want_h) != (got_eq, got_h):
                    FAILURES.append((f"dtype eq/hash {a!r} vs {b!r}",
                                     f"({got_eq}, {got_h}) != numpy's "
                                     f"({want_eq}, {want_h})"))
    for a, b in [("i4", "i8"), ("i4", ">i4"), ("S3", "S5"), ("U3", "S3"),
                 ("i4,f8", "i4,f4")]:
        CHECKS += 1
        if (np.dtype(a) == np.dtype(b)) != (_rnp.dtype(a) == _rnp.dtype(b)):
            FAILURES.append((f"dtype eq {a!r} vs {b!r}", "differs from numpy"))

    # dtype == string / type comparisons
    for spec, other in [("i4", "i4"), ("i4", "int32"), ("i4", "f8"),
                        ("f8", "float64"), ("S5", "S5"), ("S5", "S3")]:
        CHECKS += 1
        if (np.dtype(spec) == other) != (_rnp.dtype(spec) == other):
            FAILURES.append((f"dtype({spec!r}) == {other!r}", "differs"))

    # ---- promotion / casting fixtures -----------------------------------
    castable = DTYPES + ["S0", "S3", "S5", "U0", "U3", "U5", "V0", "V3", "V8"]
    for a, b in itertools.product(castable, castable):
        for casting in ["no", "equiv", "safe", "same_kind", "unsafe"]:
            CHECKS += 1
            want = _bool_or_exc(lambda: np.can_cast(a, b, casting))
            got = _bool_or_exc(lambda: _rnp.can_cast(a, b, casting))
            if want != got:
                FAILURES.append((f"can_cast({a}, {b}, {casting})",
                                 f"{got} != numpy's {want}"))
    for a, b in itertools.product(DTYPES, DTYPES):
        CHECKS += 1
        want = str(np.promote_types(a, b))
        got = str(_rnp.promote_types(a, b))
        if want != got:
            FAILURES.append((f"promote_types({a}, {b})",
                             f"{got} != numpy's {want}"))
    for a, b in itertools.product(["S3", "S5", "U3", "U5"], repeat=2):
        CHECKS += 1
        want = _bool_or_exc(lambda: str(np.promote_types(a, b)))
        got = _bool_or_exc(lambda: str(_rnp.promote_types(a, b)))
        if want != got:
            FAILURES.append((f"promote_types({a}, {b})",
                             f"{got} != numpy's {want}"))

    # result_type with NEP 50 weak python scalars
    weak_values = [0, 1, 300, -5, 2 ** 40, True, 2.5, 1e300, 1 + 1j]
    for dt in DTYPES:
        for v in weak_values:
            CHECKS += 1
            want = _bool_or_exc(lambda: str(np.result_type(np.dtype(dt), v)))
            got = _bool_or_exc(lambda: str(_rnp.result_type(_rnp.dtype(dt), v)))
            if want != got:
                FAILURES.append((f"result_type({dt}, {v!r})",
                                 f"{got} != numpy's {want}"))
    for a, b in itertools.product(DTYPES, DTYPES):
        CHECKS += 1
        want = str(np.result_type(np.dtype(a), np.dtype(b)))
        got = str(_rnp.result_type(_rnp.dtype(a), _rnp.dtype(b)))
        if want != got:
            FAILURES.append((f"result_type({a}, {b})", f"{got} != numpy's {want}"))
    for v in weak_values:
        for w in weak_values:
            CHECKS += 1
            want = str(np.result_type(v, w))
            got = str(_rnp.result_type(v, w))
            if want != got:
                FAILURES.append((f"result_type({v!r}, {w!r})",
                                 f"{got} != numpy's {want}"))

    # min_scalar_type
    scalars = [0, 1, 127, 128, 255, 256, 65535, 65536, 2 ** 31, 2 ** 32,
               2 ** 63, 2 ** 64 - 1, -1, -128, -129, -32768, -32769,
               -2 ** 31, -2 ** 31 - 1, True, False,
               0.0, -0.0, 0.1, 3.0, 64999.9, 65000.0, 65504.0, 3.3e38,
               3.4028235e38, 1e40, 1e300, float("inf"), float("-inf"),
               float("nan"), 3 + 4j, 1e300 + 0j, 65001 + 0j]
    for v in scalars:
        CHECKS += 1
        want = str(np.min_scalar_type(v))
        got = str(_rnp.min_scalar_type(v))
        if want != got:
            FAILURES.append((f"min_scalar_type({v!r})", f"{got} != numpy's {want}"))

    # common_type
    for combo in [("i4",), ("f4",), ("f8",), ("c8",), ("f4", "f8"),
                  ("i4", "f4"), ("f2",), ("c8", "f8"), ("i8", "i4")]:
        CHECKS += 1
        arrays_np = [np.ones(3, dt) for dt in combo]
        arrays_port = [_rnp.ones(3, dt) for dt in combo]
        want = np.common_type(*arrays_np).__name__
        got = str(_rnp.common_type(*arrays_port))
        if want != got:
            FAILURES.append((f"common_type{combo}", f"{got} != numpy's {want}"))

    # can_cast rejects python numbers under NEP 50
    for v in [3, 3.0, 3j]:
        CHECKS += 1
        want = _bool_or_exc(lambda: np.can_cast(v, "i1"))
        got = _bool_or_exc(lambda: _rnp.can_cast(v, "i1"))
        if want != got:
            FAILURES.append((f"can_cast({v!r}, i1)", f"{got} != numpy's {want}"))

    # ---- reductions, bit for bit ----------------------------------------
    red_dtypes = ["bool", "int8", "int16", "int32", "int64", "uint8",
                  "uint16", "uint32", "uint64", "float16", "float32",
                  "float64", "complex64", "complex128"]
    for dt in red_dtypes:
        for size in [0, 1, 2, 7, 8, 9, 16, 100, 127, 128, 129, 200, 1000,
                     1031, 5000]:
            data = sample(rng, dt, size)
            port = to_port(data)
            for name in ["sum", "prod", "min", "max", "argmin", "argmax",
                         "mean"]:
                CHECKS += 1
                w = _bool_or_exc(lambda: getattr(data, name)())
                g = _bool_or_exc(lambda: getattr(port, name)())
                ref = getattr(w, "dtype", None)
                want = w if isinstance(w, str) else _reduction_repr(w, ref)
                got = g if isinstance(g, str) else _reduction_repr(g, ref)
                if want != got:
                    FAILURES.append((f"{name}({dt}, n={size})",
                                     f"{got} != numpy's {want}"))
                # The result dtype, checked where both sides give an array.
                CHECKS += 1
                wd = _bool_or_exc(
                    lambda: str(getattr(data.reshape(1, -1), name)(
                        axis=1, keepdims=True).dtype))
                gd = _bool_or_exc(
                    lambda: str(getattr(port.reshape(1, -1), name)(
                        axis=1, keepdims=True).dtype))
                if wd != gd:
                    FAILURES.append((f"{name}({dt}, n={size}).dtype",
                                     f"{gd} != numpy's {wd}"))

    # Signed zero and NaN edge cases.
    for values, dt in [([-0.0, -0.0], "float64"), ([-0.0, 0.0], "float64"),
                       ([1.0, float("nan"), 3.0], "float64"),
                       ([float("nan")], "float32"),
                       ([float("inf"), float("-inf")], "float64")]:
        data = np.array(values, dt)
        port = to_port(data)
        for name in ["sum", "min", "max", "argmin", "argmax"]:
            CHECKS += 1
            w = getattr(data, name)()
            want = _reduction_repr(w, w.dtype)
            got = _reduction_repr(getattr(port, name)(), w.dtype)
            if want != got:
                FAILURES.append((f"{name}({values}, {dt})",
                                 f"{got} != numpy's {want}"))

    # Axis reductions on 2-D and 3-D arrays, contiguous and strided.
    for dt in ["int32", "int64", "uint8", "float32", "float64", "complex128"]:
        for shape in [(3, 5), (4, 6), (2, 3, 4), (17, 9), (1, 130), (130, 1)]:
            base = sample(rng, dt, int(np.prod(shape))).reshape(shape)
            pbase = to_port(base.ravel()).reshape(shape)
            for view_name, view in [("", lambda x: x),
                                    ("[::2]", lambda x: x[::2]),
                                    ("[..., ::2]", lambda x: x[..., ::2])]:
                a, pa = view(base), view(pbase)
                for axis in range(len(shape)):
                    for keepdims in (False, True):
                        for name in ["sum", "min", "max", "argmin", "argmax",
                                     "mean"]:
                            CHECKS += 1
                            kw = {"axis": axis, "keepdims": keepdims}
                            w = _bool_or_exc(lambda: getattr(a, name)(**kw))
                            g = _bool_or_exc(lambda: getattr(pa, name)(**kw))
                            ref = getattr(w, "dtype", None)
                            want = w if isinstance(w, str) else \
                                _reduction_repr(w, ref)
                            got = g if isinstance(g, str) else \
                                _reduction_repr(g, ref)
                            if want != got:
                                FAILURES.append((
                                    f"{name}({dt}{shape}{view_name}, "
                                    f"axis={axis}, keepdims={keepdims})",
                                    f"{got} != numpy's {want}"))

    # ---- flexible (S/U) arrays -------------------------------------------
    for values in [[b"ab", b"cde"], ["ab", "cde"], ["", "x"],
                   [b"", b"zzzz"], ["\u00e9", "ab"]]:
        CHECKS += 1
        want = np.array(values)
        got = _rnp.array(values)
        if str(want.dtype) != str(got.dtype):
            FAILURES.append((f"array({values!r}).dtype",
                             f"{got.dtype} != numpy's {want.dtype}"))
        CHECKS += 1
        if want.tolist() != got.tolist():
            FAILURES.append((f"array({values!r}).tolist()",
                             f"{got.tolist()!r} != numpy's {want.tolist()!r}"))
        CHECKS += 1
        want_eq = (want == want).tolist()
        got_eq = (got == got).tolist()
        if want_eq != got_eq:
            FAILURES.append((f"array({values!r}) == itself",
                             f"{got_eq} != numpy's {want_eq}"))

    # ---- indexing (M2) ---------------------------------------------------
    check_indexing(rng)

    # ---- error parity ---------------------------------------------------
    error_cases = [
        ("bool subtract", lambda m, mk: m.subtract(mk([True]), mk([True]))),
        ("bad broadcast", lambda m, mk: m.add(mk([1, 2, 3]), mk([1, 2, 3, 4]))),
    ]
    for label, fn in error_cases:
        CHECKS += 1
        try:
            fn(np, lambda x: np.array(x))
            np_exc = None
        except Exception as e:  # noqa: BLE001
            np_exc = type(e).__name__
        try:
            fn(_rnp, lambda x: _rnp.array(x))
            port_exc = None
        except Exception as e:  # noqa: BLE001
            port_exc = type(e).__name__
        if np_exc != port_exc:
            FAILURES.append((f"error {label}",
                             f"port raised {port_exc}, numpy raised {np_exc}"))

    # ---- scalar types and ufunc machinery (M3) ---------------------------
    run_m3_sections()

    print(f"{CHECKS} comparisons, {len(FAILURES)} divergences")
    for name, msg in FAILURES:
        print(f"  FAIL {name}: {msg}")
    print_m3_report()
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
