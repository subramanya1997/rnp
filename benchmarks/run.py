#!/usr/bin/env python3
"""Paired benchmarks: Rust port (rnp_numpy) vs real NumPy, same venv.

Each case is timed in BOTH libraries with identical inputs; output is a table
with the ratio port/numpy (lower is better for the port). Real numpy is
imported normally; the port is imported as rnp_numpy (no redirection needed
here since benchmarks are ours, not upstream's).

Usage: .venv/bin/python benchmarks/run.py [--sizes 1000,1000000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as real_np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shim"))

try:
    import rnp_numpy as port_np
except ImportError as e:
    print(f"port not importable yet: {e}", file=sys.stderr)
    sys.exit(1)

RESULTS = ROOT / "benchmarks" / "results.json"


def bench(fn, *args, reps=25, inner=1):
    fn(*args)  # warmup
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn(*args)
        best = min(best, (time.perf_counter() - t0) / inner)
    return best


def cases(np_mod, size):
    a = np_mod.arange(size, dtype=np_mod.float64)
    b = np_mod.arange(size, dtype=np_mod.float64)
    i32 = np_mod.arange(size, dtype=np_mod.int32)
    yield "add_f64", lambda: np_mod.add(a, b)
    yield "mul_f64", lambda: np_mod.multiply(a, b)
    yield "add_i32", lambda: np_mod.add(i32, i32)
    yield "sum_f64", lambda: np_mod.sum(a)
    yield "max_f64", lambda: np_mod.max(a)
    yield "astype", lambda: a.astype(np_mod.float32)
    yield "copy", lambda: a.copy()
    yield "strided_add", lambda: np_mod.add(a[::2], b[::2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,1000000")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    rows = []
    print(f"{'case':16s} {'size':>10s} {'numpy (us)':>12s} {'port (us)':>12s} {'ratio':>7s}")
    for size in sizes:
        inner = max(1, 100_000 // size)
        real = dict(cases(real_np, size))
        port = dict(cases(port_np, size))
        for name in real:
            try:
                t_port = bench(port[name], inner=inner) * 1e6
            except Exception as e:
                print(f"{name:16s} {size:>10d}  port FAILED: {e}")
                continue
            t_real = bench(real[name], inner=inner) * 1e6
            ratio = t_port / t_real
            rows.append({"case": name, "size": size, "numpy_us": t_real,
                         "port_us": t_port, "ratio": ratio})
            print(f"{name:16s} {size:>10d} {t_real:12.2f} {t_port:12.2f} {ratio:7.2f}x")
    RESULTS.write_text(json.dumps(rows, indent=2))
    print(f"\n-> {RESULTS}")


if __name__ == "__main__":
    main()
