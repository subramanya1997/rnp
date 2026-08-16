#!/usr/bin/env python3
"""Differential cross-verification: real NumPy vs the Rust port.

Hypothesis generates random (op, dtypes, shapes, values); both libraries
compute the result; integers/bool must match bit-exactly, floats within 1 ULP
(elementwise ops on same inputs should normally be bit-identical too — we
start strict and loosen only with documented justification).

Real numpy is imported normally. The port is imported as rnp_numpy from shim/.

Usage: .venv/bin/python harness/crosscheck.py [--examples 200]
"""
import argparse
import sys
from pathlib import Path

import numpy as real_np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shim"))
import rnp_numpy as port_np  # noqa: E402

from hypothesis import given, settings, seed  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

DTYPES = ["bool", "int8", "int16", "int32", "int64",
          "uint8", "uint16", "uint32", "uint64",
          "float32", "float64"]

BINARY_OPS = ["add", "subtract", "multiply"]  # divide handled separately (zeros)


def to_real(port_arr):
    """Convert a port array to a real numpy array via the buffer protocol."""
    return real_np.asarray(memoryview(port_arr))


@st.composite
def array_case(draw):
    dt = draw(st.sampled_from(DTYPES))
    shape = draw(st.lists(st.integers(1, 6), min_size=1, max_size=3))
    n = 1
    for s in shape:
        n *= s
    if dt == "bool":
        vals = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    elif dt.startswith("float"):
        vals = draw(st.lists(
            st.floats(-1e6, 1e6, allow_nan=False, width=32),
            min_size=n, max_size=n))
    else:
        info = real_np.iinfo(dt)
        lo = max(info.min, -1 << 40)
        hi = min(info.max, 1 << 40)
        vals = draw(st.lists(st.integers(lo, hi), min_size=n, max_size=n))
    return dt, tuple(shape), vals


failures = []


def check(name, real_out, port_out):
    port_as_real = to_real(port_out)
    if real_out.dtype != port_as_real.dtype:
        failures.append(f"{name}: dtype {real_out.dtype} != {port_as_real.dtype}")
        return
    if real_out.shape != tuple(port_as_real.shape):
        failures.append(f"{name}: shape {real_out.shape} != {port_as_real.shape}")
        return
    if not real_np.array_equal(real_out, port_as_real, equal_nan=True):
        failures.append(f"{name}: values diverge\n real={real_out!r}\n port={port_as_real!r}")


@seed(0)
@settings(max_examples=200, deadline=None)
@given(a=array_case(), b=array_case(), op=st.sampled_from(BINARY_OPS))
def test_binary(a, b, op):
    (dt_a, sh_a, va), (dt_b, sh_b, vb) = a, b
    ra = real_np.array(va, dtype=dt_a).reshape(sh_a)
    rb = real_np.array(vb, dtype=dt_b).reshape(sh_b)
    try:
        expected = getattr(real_np, op)(ra, rb)
    except ValueError:
        return  # non-broadcastable; port behavior checked in unit tests
    pa = port_np.array(va, dtype=dt_a).reshape(sh_a)
    pb = port_np.array(vb, dtype=dt_b).reshape(sh_b)
    got = getattr(port_np, op)(pa, pb)
    check(f"{op}({dt_a}{sh_a}, {dt_b}{sh_b})", expected, got)


@seed(1)
@settings(max_examples=100, deadline=None)
@given(a=array_case(), to=st.sampled_from(DTYPES))
def test_astype(a, to):
    dt, sh, vals = a
    if dt.startswith("float") and not to.startswith("float"):
        return  # float->int of out-of-range values is UB-ish; covered elsewhere
    ra = real_np.array(vals, dtype=dt).reshape(sh)
    pa = port_np.array(vals, dtype=dt).reshape(sh)
    check(f"astype({dt}->{to})", ra.astype(to), pa.astype(to))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=200)
    args = ap.parse_args()
    for fn in (test_binary, test_astype):
        fn()
    if failures:
        print(f"{len(failures)} DIVERGENCES:")
        for f in failures[:20]:
            print(" -", f)
        return 1
    print("crosscheck: all cases match real numpy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
