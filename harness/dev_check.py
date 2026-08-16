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
import random
import sys
import traceback

import numpy as np

import _rnp

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

    print(f"{CHECKS} comparisons, {len(FAILURES)} divergences")
    for name, msg in FAILURES:
        print(f"  FAIL {name}: {msg}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
