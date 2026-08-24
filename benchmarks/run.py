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


def _time_once(fn, inner):
    t0 = time.perf_counter()
    for _ in range(inner):
        fn()
    return (time.perf_counter() - t0) / inner


def bench_pair(fn_a, fn_b, reps=25, inner=1):
    """Time two implementations of the same case, interleaved.

    Interleaving matters: running one library's rounds to completion before
    the other's hands the second one a warm allocator and warm caches, which
    was worth up to 3x on the 1e6 rows.
    """
    for _ in range(3):  # warmup, both sides
        fn_a()
        fn_b()
    best_a = best_b = float("inf")
    for _ in range(reps):
        best_a = min(best_a, _time_once(fn_a, inner))
        best_b = min(best_b, _time_once(fn_b, inner))
    return best_a, best_b


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

    # ---- M2: indexing ----------------------------------------------------
    idx = np_mod.arange(0, size, 3, dtype=np_mod.int64)
    mask = np_mod.zeros(size, dtype=np_mod.bool_)
    mask[idx] = True
    rows_n = max(1, size // 100)
    m = np_mod.arange(rows_n * 100, dtype=np_mod.float64).reshape(rows_n, 100)
    row_idx = np_mod.arange(0, rows_n, 2, dtype=np_mod.int64)
    yield "slice_view", lambda: a[::2]
    yield "slice_copy", lambda: a[1:size - 1].copy()
    yield "fancy_1d", lambda: a[idx]
    yield "fancy_2d_rows", lambda: m[row_idx]
    yield "bool_mask", lambda: a[mask]
    yield "fancy_setitem", lambda: b.__setitem__(idx, 1.0)
    yield "bool_setitem", lambda: b.__setitem__(mask, 2.0)
    yield "take_axis0", lambda: m.take(row_idx, axis=0)

    # ---- M3: ufuncs ------------------------------------------------------
    # Scaled into each function's ordinary range: `exp(arange(1e6))` would
    # overflow on almost every element, which measures the special-value path
    # rather than the loop.
    small = np_mod.multiply(a, 1e-5)
    f32 = small.astype(np_mod.float32)
    pos = np_mod.add(small, 1.0)
    yield "exp_f64", lambda: np_mod.exp(small)
    yield "exp_f32", lambda: np_mod.exp(f32)
    yield "sin_f64", lambda: np_mod.sin(small)
    yield "sqrt_f64", lambda: np_mod.sqrt(a)  # arange is non-negative
    yield "log_f64", lambda: np_mod.log(pos)
    yield "power_f64", lambda: np_mod.power(pos, 2.5)
    yield "abs_f64", lambda: np_mod.absolute(a)
    yield "negative_f64", lambda: np_mod.negative(a)
    yield "maximum_f64", lambda: np_mod.maximum(a, b)
    yield "floor_divide_i32", lambda: np_mod.floor_divide(i32 + 1, 7)
    yield "bitwise_and_i32", lambda: np_mod.bitwise_and(i32, 255)
    yield "add_reduce_f64", lambda: np_mod.add.reduce(a)
    yield "scalar_add_f64", lambda: np_mod.float64(1.5) + np_mod.float64(2.5)
    yield "scalar_add_i64", lambda: np_mod.int64(3) + np_mod.int64(4)
    yield "scalar_extract", lambda: a[7]


def matmul_cases(np_mod):
    """The matmul family, sized in matrix dimensions rather than elements.

    numpy hands contiguous float32/float64/complex matrix-matrix products to
    BLAS (`cblas_?gemm`), which is blocked, hand-tuned and vectorised; the
    port's kernel is a plain packed triple loop. The float rows below are
    therefore expected to lose, and are kept in the table at full size so the
    gap is visible rather than hidden. The integer and batched-small rows are
    the ones where numpy also runs a scalar C loop, so they are the fair
    like-for-like comparison.
    """
    def sq(n, dt):
        a = np_mod.arange(n * n, dtype=dt).reshape(n, n)
        b = np_mod.arange(n * n, dtype=dt).reshape(n, n)
        return a, b

    for n in (32, 128, 256):
        for dt_name in ("float64", "float32", "int32"):
            a, b = sq(n, getattr(np_mod, dt_name))
            yield f"matmul_{dt_name}_{n}", (lambda a=a, b=b: np_mod.matmul(a, b))
    a, b = sq(512, np_mod.float64)
    yield "matmul_float64_512", lambda: np_mod.matmul(a, b)

    # Batched small matrices: 4096 independent 8x8 products, where numpy's
    # per-call BLAS overhead is at its worst and the port's loop at its best.
    ba = np_mod.arange(4096 * 8 * 8, dtype=np_mod.float64).reshape(4096, 8, 8)
    bb = np_mod.arange(4096 * 8 * 8, dtype=np_mod.float64).reshape(4096, 8, 8)
    yield "matmul_batched_4096x8x8", lambda: np_mod.matmul(ba, bb)

    # Vector shapes.
    v = np_mod.arange(1_000_000, dtype=np_mod.float64)
    w = np_mod.arange(1_000_000, dtype=np_mod.float64)
    yield "vecdot_float64_1e6", lambda: np_mod.vecdot(v, w)
    mv, vv = sq(512, np_mod.float64)[0], np_mod.arange(512, dtype=np_mod.float64)
    yield "matvec_float64_512", lambda: np_mod.matvec(mv, vv)
    dot_a, dot_b = sq(256, np_mod.float64)
    yield "dot_float64_256", lambda: np_mod.dot(dot_a, dot_b)


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
                t_real, t_port = bench_pair(real[name], port[name], inner=inner)
            except Exception as e:
                print(f"{name:16s} {size:>10d}  port FAILED: {e}")
                continue
            t_real *= 1e6
            t_port *= 1e6
            ratio = t_port / t_real
            rows.append({"case": name, "size": size, "numpy_us": t_real,
                         "port_us": t_port, "ratio": ratio})
            print(f"{name:16s} {size:>10d} {t_real:12.2f} {t_port:12.2f} {ratio:7.2f}x")

    # ---- the matmul family, sized in matrix dimensions -------------------
    print()
    print(f"{'case':24s} {'numpy (us)':>12s} {'port (us)':>12s} {'ratio':>7s}")
    real = dict(matmul_cases(real_np))
    port = dict(matmul_cases(port_np))
    for name in real:
        try:
            t_real, t_port = bench_pair(real[name], port[name], reps=5, inner=1)
        except Exception as e:  # noqa: BLE001
            print(f"{name:24s}  port FAILED: {e}")
            continue
        t_real *= 1e6
        t_port *= 1e6
        ratio = t_port / t_real
        rows.append({"case": name, "size": None, "numpy_us": t_real,
                     "port_us": t_port, "ratio": ratio})
        print(f"{name:24s} {t_real:12.2f} {t_port:12.2f} {ratio:7.2f}x")

    RESULTS.write_text(json.dumps(rows, indent=2))
    print(f"\n-> {RESULTS}")


if __name__ == "__main__":
    main()
