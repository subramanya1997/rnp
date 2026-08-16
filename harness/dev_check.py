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
    "float32", "float64", "complex64", "complex128",
]

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
    """Rebuild a numpy array as a port array through nested lists."""
    return _rnp.array(a.tolist(), str(a.dtype))


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
