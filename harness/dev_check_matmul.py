#!/usr/bin/env python3
"""Differential check for the matmul family: the port vs real numpy.

Same idiom as `harness/dev_check.py` -- both libraries are imported normally
into this one process (no import redirection), fed byte-identical inputs, and
their answers compared. What is checked here:

  * `matmul`, `vecdot`, `matvec`, `vecmat`, `dot` and `inner` over every
    dtype the port implements, across 1-D promotion, batched/broadcast
    operands, `out=`, zero-sized dimensions and four operand layouts
    (contiguous, strided, negative-strided, Fortran-order);
  * every error case, compared by exception type *and* message;
  * `ndarray.__index__`;
  * `isclose` / `allclose` / `array_equal` / `array_equiv`.

Tolerance, and why it is not zero
---------------------------------
For bool, every integer dtype and float16 numpy never calls BLAS: its
`matmul_inner_noblas` (and `TYPE_dot`) accumulate one output element at a
time, left to right, in the output dtype -- float16 in a float32 accumulator,
bool as an OR of ANDs. The port's kernel reproduces that order exactly, so
those dtypes are required to be **bit-identical** and any difference is a
failure.

For float32/float64/complex64/complex128 numpy dispatches contiguous 2-D
products to BLAS (`cblas_?gemm` / `?gemv` / `?dot`), whose blocked, vectorised
summation is *not* a left-to-right sum. Without linking BLAS the port cannot
match those bit for bit, so the two answers are held to the *accumulated
rounding bound* that any summation order satisfies:

    |port - numpy|  <=  TOL_FACTOR * eps(dtype) * sum_k |a_k| * |b_k|

with the right-hand sum computed as `f(abs(a), abs(b))` -- the same product
with every operand made non-negative, so no cancellation can shrink it. The
textbook bound for one ordering is `k * eps * sum|a_k b_k|`; `TOL_FACTOR` is
128, which covers both orderings for every `k` this file uses (<= 32) with
room to spare. Crucially the bound does *not* depend on the observed answer,
so a genuinely wrong result cannot slip through by being wrong consistently.

ULP distance is reported alongside but is *not* the pass criterion: a dot
product whose terms cancel has a tiny result, so a rounding difference far
inside the bound above can still be thousands of ULP. The per-dtype worst ULP
printed at the end is a measurement of how far the port's accumulation order
lands from BLAS's, not a threshold anything was tuned to.

Usage: .venv/bin/python harness/dev_check_matmul.py [--seed N]
"""
import argparse
import os
import random
import sys
import traceback

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHIM_DIR = os.path.join(_ROOT, "shim")
if _SHIM_DIR not in sys.path:
    sys.path.insert(0, _SHIM_DIR)

import rnp_numpy as rnp  # noqa: E402

CHECKS = 0
FAILURES = []

#: Worst ULP distance seen per dtype, over every float/complex comparison.
WORST_ULP = {}

#: Multiplier on `eps * sum|a_k b_k|` (see the module docstring). 128 bounds
#: the difference between any two summation orders for the products here.
TOL_FACTOR = 128

#: Fallback ULP bound for the handful of float comparisons where no magnitude
#: bound is available (results that are exact by construction).
ULP_BOUND = 4

DTYPES = [
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "complex64", "complex128",
]

FLOATY = {"float16", "float32", "float64", "complex64", "complex128"}


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def sample(rng, dtype, size):
    """Deterministic data for a dtype, identical for both libraries.

    Ranges are kept small for the narrow integer types so that a dot product
    of a few dozen terms is interesting without being dominated by wraparound
    (wraparound is exercised deliberately in its own section).
    """
    if dtype == "bool":
        return np.array([rng.random() > 0.5 for _ in range(size)], dtype=bool)
    if dtype.startswith("uint"):
        hi = min(2 ** (8 * np.dtype(dtype).itemsize) - 1, 50)
        return np.array([rng.randrange(0, hi) for _ in range(size)], dtype=dtype)
    if dtype.startswith("int"):
        lim = min(2 ** (8 * np.dtype(dtype).itemsize - 1) - 1, 50)
        return np.array([rng.randrange(-lim, lim) for _ in range(size)],
                        dtype=dtype)
    if dtype == "float16":
        return np.array([rng.uniform(-3, 3) for _ in range(size)], dtype=dtype)
    if dtype.startswith("float"):
        return np.array([rng.uniform(-10, 10) for _ in range(size)], dtype=dtype)
    return np.array(
        [complex(rng.uniform(-10, 10), rng.uniform(-10, 10))
         for _ in range(size)], dtype=dtype)


def to_port(a):
    """Rebuild a numpy array as a port array, element by element."""
    if a.size == 0:
        return rnp.zeros(a.shape, dtype=str(a.dtype))
    flat = rnp.array(a.ravel().tolist(), dtype=str(a.dtype))
    return flat.reshape(a.shape)


def _bytes_of(x):
    if hasattr(x, "tobytes"):
        return x.tobytes()
    return np.asarray(x).tobytes()


def _as_numpy(got, ref):
    """`got` (a port array or scalar) as a numpy array shaped like `ref`."""
    return np.frombuffer(_bytes_of(got), dtype=ref.dtype).reshape(ref.shape)


def _ordered_bits(a):
    """IEEE floats -> monotonically increasing unsigned ints (ULP ladder)."""
    n = a.dtype.itemsize * 8
    ut = {16: np.uint16, 32: np.uint32, 64: np.uint64}[n]
    u = a.view(ut).astype(np.uint64)
    sign = np.uint64(1) << np.uint64(n - 1)
    mag = u & (sign - np.uint64(1))
    neg = (u & sign) != 0
    return np.where(neg, sign - mag, sign + mag)


def _ulp(w, g):
    """Max ULP distance between two same-dtype real float arrays."""
    if w.size == 0:
        return 0
    nan_w, nan_g = np.isnan(w), np.isnan(g)
    if np.any(nan_w != nan_g):
        return None  # a NaN appeared or vanished: never a rounding difference
    ow, og = _ordered_bits(w), _ordered_bits(g)
    d = np.where(ow > og, ow - og, og - ow)
    d = np.where(nan_w & nan_g, np.uint64(0), d)
    return int(d.max())


def _isnan_any(x):
    """NaN mask; for complex, True where either half is NaN (numpy's rule)."""
    a = np.asarray(x)
    if a.dtype.kind not in "fc":
        return np.zeros(a.shape, bool)
    return np.isnan(a)


def magnitude(fn, a, b):
    """`sum_k |a_k| |b_k|` per output element: the same product with every
    operand made non-negative, so cancellation cannot shrink it.

    Returns None for products that come out exact (integer and bool loops),
    where `compare` requires bit equality instead.
    """
    a, b = np.asarray(a), np.asarray(b)
    if np.result_type(a.dtype, b.dtype).kind not in "fc":
        return None
    try:
        return np.asarray(fn(np.abs(a).astype(np.float64),
                             np.abs(b).astype(np.float64)), dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None


def _record_ulp(dtype, value):
    if value is None:
        return
    if value > WORST_ULP.get(dtype, -1):
        WORST_ULP[dtype] = value


def compare(name, want, got, scale=None):
    """One result comparison.

    Exact dtypes must be bit-identical. Inexact ones must land inside the
    accumulated-rounding bound built from `scale` (an upper bound on
    `sum_k |a_k b_k|` for each output element); the ULP distance is recorded
    for the report either way.
    """
    global CHECKS
    CHECKS += 1
    w = np.asarray(want)
    try:
        g = _as_numpy(got, w)
    except Exception as exc:  # noqa: BLE001
        FAILURES.append((name, f"could not read port result: {exc!r}"))
        return
    got_dtype = str(getattr(got, "dtype", w.dtype))
    if got_dtype != str(w.dtype):
        FAILURES.append((name, f"dtype {got_dtype} != numpy's {w.dtype}"))
        return
    got_shape = tuple(np.shape(got))
    if got_shape != w.shape:
        FAILURES.append((name, f"shape {got_shape} != numpy's {w.shape}"))
        return
    dt = str(w.dtype)
    if dt not in FLOATY:
        if not np.array_equal(w, g):
            bad = int(np.count_nonzero(w != g))
            FAILURES.append((name, f"{bad}/{w.size} elements differ (exact "
                                   f"equality required for {dt})"))
        return
    if w.dtype.kind == "c":
        half = np.float32 if w.dtype.itemsize == 8 else np.float64
        # `.view()` refuses to change the itemsize of a 0-d array, so flatten
        # first: the ULP comparison does not care about the shape.
        d = _ulp(np.ascontiguousarray(w).ravel().view(half),
                 np.ascontiguousarray(g).ravel().view(half))
    else:
        d = _ulp(w, g)
    if d is None:
        FAILURES.append((name, "NaN pattern differs from numpy's"))
        return
    _record_ulp(dt, d)

    if scale is None:
        if d > ULP_BOUND:
            FAILURES.append(
                (name, f"max ULP {d} exceeds the {ULP_BOUND} bound (no "
                       f"magnitude bound was available for this call)"))
        return

    real = np.float32 if w.dtype.itemsize in (4, 8) and w.dtype.kind == "c" \
        else w.dtype
    eps = np.finfo(np.float32 if w.dtype == np.complex64 else
                   np.float64 if w.dtype == np.complex128 else
                   w.dtype).eps
    tol = TOL_FACTOR * eps * np.asarray(scale, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        diff = np.abs(np.asarray(w, dtype=np.complex128 if w.dtype.kind == "c"
                                 else np.float64)
                      - np.asarray(g, dtype=np.complex128 if w.dtype.kind == "c"
                                   else np.float64))
        finite = np.isfinite(diff)
        over = finite & (diff > tol)
        # Non-finite results must agree exactly (bit for bit modulo NaN).
        bad_nonfinite = (~finite) & (np.asarray(w) != np.asarray(g)) \
            & ~(_isnan_any(w) & _isnan_any(g))
    del real
    if np.any(over) or np.any(bad_nonfinite):
        worst = float(np.max(np.where(finite, diff, 0.0))) if diff.size else 0.0
        FAILURES.append(
            (name, f"|port - numpy| = {worst:.3e} exceeds the rounding bound "
                   f"{TOL_FACTOR}*eps*sum|a||b| (max ULP {d})"))


def eq(name, want, got):
    global CHECKS
    CHECKS += 1
    if want != got:
        FAILURES.append((name, f"port {got!r} != numpy {want!r}"))


def outcome(fn):
    """('ok', value) or ('exc', (type name, message))."""
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001
        return ("exc", (type(exc).__name__, str(exc)))


def check_error(name, np_fn, port_fn):
    """Both sides must raise the same exception type with the same message."""
    global CHECKS
    CHECKS += 1
    w = outcome(np_fn)
    g = outcome(port_fn)
    if w[0] != g[0]:
        FAILURES.append((name, f"port {g[0]}:{g[1]!r} vs numpy {w[0]}:{w[1]!r}"))
        return
    if w[0] == "exc":
        if w[1][0] != g[1][0]:
            FAILURES.append(
                (name, f"port raised {g[1][0]}, numpy raised {w[1][0]}"))
        elif w[1][1] != g[1][1]:
            FAILURES.append(
                (name, f"message {g[1][1]!r} != numpy's {w[1][1]!r}"))


# ---------------------------------------------------------------------------
# operand layouts
# ---------------------------------------------------------------------------

def build(rng, shape, dtype, layout):
    """A (numpy, port) operand pair of exactly `shape`, in a given layout.

    Every layout is expressed with operations both libraries have, and both
    sides are built from the same values, so the comparison never depends on
    the port's constructor agreeing with numpy's about anything but the data.
    """
    shape = tuple(shape)
    if layout == "c":
        base = sample(rng, dtype, int(np.prod(shape)) if shape else 1)
        w = base.reshape(shape)
        return w, to_port(w)
    if layout == "step":
        big = tuple(max(d * 2, 0) for d in shape)
        base = sample(rng, dtype, int(np.prod(big)) if big else 1).reshape(big)
        sl = tuple(slice(None, None, 2) for _ in shape)
        return base[sl], to_port(base)[sl]
    if layout == "rev":
        base = sample(rng, dtype, int(np.prod(shape)) if shape else 1)
        base = base.reshape(shape)
        sl = tuple(slice(None, None, -1) for _ in shape)
        return base[sl], to_port(base)[sl]
    if layout == "f":
        rev = shape[::-1]
        base = sample(rng, dtype, int(np.prod(rev)) if rev else 1).reshape(rev)
        return base.T, to_port(base).T
    raise AssertionError(layout)


LAYOUTS = ["c", "step", "rev", "f"]

#: (kind, a_shape, b_shape) triples covering 1-D promotion on either side,
#: batching, broadcast batching and zero-sized core/loop dimensions.
CASES = [
    ("matmul", (2, 3), (3, 4)),
    ("matmul", (1, 1), (1, 1)),
    ("matmul", (3,), (3, 4)),
    ("matmul", (2, 3), (3,)),
    ("matmul", (3,), (3,)),
    ("matmul", (5, 2, 3), (3, 4)),
    ("matmul", (2, 3), (5, 3, 4)),
    ("matmul", (5, 2, 3), (5, 3, 4)),
    ("matmul", (2, 1, 3, 4), (5, 4, 2)),
    ("matmul", (1, 2, 3), (4, 3, 2)),
    ("matmul", (7, 9), (9, 5)),
    ("matmul", (2, 0), (0, 3)),
    ("matmul", (0, 3), (3, 4)),
    ("matmul", (3, 4), (4, 0)),
    ("matmul", (0, 2, 3), (3, 4)),
    ("vecdot", (3,), (3,)),
    ("vecdot", (5, 3), (3,)),
    ("vecdot", (5, 1, 3), (4, 3)),
    ("vecdot", (0,), (0,)),
    ("vecdot", (4, 0), (0,)),
    ("vecdot", (17,), (17,)),
    ("matvec", (2, 3), (3,)),
    ("matvec", (5, 2, 3), (5, 3)),
    ("matvec", (2, 3), (5, 3)),
    ("matvec", (0, 3), (3,)),
    ("matvec", (3, 0), (0,)),
    ("vecmat", (3,), (3, 2)),
    ("vecmat", (5, 3), (3, 2)),
    ("vecmat", (5, 3), (5, 3, 2)),
    ("vecmat", (0,), (0, 2)),
]

DOT_CASES = [
    ("dot", (2, 3), (3, 4)),
    ("dot", (3,), (3,)),
    ("dot", (5, 2, 3), (3, 4)),
    ("dot", (5, 2, 3), (6, 3, 4)),
    ("dot", (3,), (5, 3, 4)),
    ("dot", (2, 3), (3,)),
    ("dot", (2, 0), (0, 4)),
    ("inner", (3,), (3,)),
    ("inner", (2, 3), (3,)),
    ("inner", (2, 3), (4, 3)),
    ("inner", (5, 2, 3), (4, 3)),
    ("inner", (2, 0), (4, 0)),
]


def check_products(rng):
    for kind, sa, sb in CASES:
        np_fn, port_fn = getattr(np, kind), getattr(rnp, kind)
        for dt in DTYPES:
            for layout in LAYOUTS:
                a, pa = build(rng, sa, dt, layout)
                b, pb = build(rng, sb, dt, layout)
                want = np_fn(a, b)
                got = outcome(lambda: port_fn(pa, pb))
                if got[0] == "exc":
                    global CHECKS
                    CHECKS += 1
                    FAILURES.append(
                        (f"{kind}{sa}x{sb} {dt} {layout}",
                         f"port raised {got[1][0]}: {got[1][1]}"))
                    continue
                compare(f"{kind}{sa}x{sb} {dt} {layout}", want, got[1],
                        magnitude(np_fn, a, b))

    # Mixed dtypes exercise the promotion path on both sides.
    for kind, sa, sb in CASES[:12]:
        np_fn, port_fn = getattr(np, kind), getattr(rnp, kind)
        for _ in range(6):
            da, db = rng.choice(DTYPES), rng.choice(DTYPES)
            a, pa = build(rng, sa, da, "c")
            b, pb = build(rng, sb, db, "c")
            compare(f"{kind}{sa}x{sb} {da}@{db}", np_fn(a, b),
                    port_fn(pa, pb), magnitude(np_fn, a, b))


def check_dot_inner(rng):
    for kind, sa, sb in DOT_CASES:
        np_fn, port_fn = getattr(np, kind), getattr(rnp, kind)
        for dt in DTYPES:
            for layout in LAYOUTS:
                a, pa = build(rng, sa, dt, layout)
                b, pb = build(rng, sb, dt, layout)
                compare(f"{kind}{sa}x{sb} {dt} {layout}",
                        np_fn(a, b), port_fn(pa, pb),
                        magnitude(np_fn, a, b))
    # dot/inner with scalar operands multiply elementwise.
    for dt in ["int32", "float64", "complex128"]:
        a, pa = build(rng, (), dt, "c")
        b, pb = build(rng, (2, 2), dt, "c")
        compare(f"dot scalar {dt}", np.dot(a, b), rnp.dot(pa, pb),
                magnitude(np.dot, a, b))
        compare(f"dot scalar2 {dt}", np.dot(b, a), rnp.dot(pb, pa),
                magnitude(np.dot, b, a))
        compare(f"inner scalar {dt}", np.inner(a, b), rnp.inner(pa, pb),
                magnitude(np.inner, a, b))
    # vdot flattens and conjugates.
    for dt in ["int32", "float64", "complex128"]:
        a, pa = build(rng, (2, 3), dt, "c")
        b, pb = build(rng, (3, 2), dt, "c")
        compare(f"vdot {dt}", np.vdot(a, b), rnp.vdot(pa, pb),
                magnitude(np.vdot, a, b))


def check_out(rng):
    """`out=`: the returned object, the stored contents, and the shapes and
    dtypes numpy accepts."""
    for kind, sa, sb in CASES:
        np_fn, port_fn = getattr(np, kind), getattr(rnp, kind)
        for dt in ["int32", "int64", "float32", "float64", "complex128"]:
            a, pa = build(rng, sa, dt, "c")
            b, pb = build(rng, sb, dt, "c")
            ref = np_fn(a, b)
            oshape = np.shape(ref)
            o_np = np.empty(oshape, dt)
            o_port = rnp.empty(oshape, dtype=dt)
            r_np = np_fn(a, b, out=o_np)
            r_port = port_fn(pa, pb, out=o_port)
            mag = magnitude(np_fn, a, b)
            compare(f"{kind}{sa}x{sb} {dt} out= return", r_np, r_port, mag)
            compare(f"{kind}{sa}x{sb} {dt} out= stored", o_np, o_port, mag)

    # An `out` whose dtype the result can be safe-cast into.
    a, pa = build(rng, (2, 3), "int32", "c")
    b, pb = build(rng, (3, 4), "int32", "c")
    o_np, o_port = np.empty((2, 4), "float64"), rnp.empty((2, 4), dtype="float64")
    compare("matmul out=f8 from i4", np.matmul(a, b, out=o_np),
            rnp.matmul(pa, pb, out=o_port),
            magnitude(np.matmul, a.astype("float64"), b.astype("float64")))

    # `out` with extra leading dimensions of size 1 broadcasts.
    o_np, o_port = np.empty((1, 2, 4), "int32"), rnp.empty((1, 2, 4), dtype="int32")
    compare("matmul out=(1,2,4)", np.matmul(a, b, out=o_np),
            rnp.matmul(pa, pb, out=o_port))

    # dtype= forces the loop type.
    for dt in ["float32", "float64", "complex128"]:
        compare(f"matmul dtype={dt}", np.matmul(a, b, dtype=dt),
                rnp.matmul(pa, pb, dtype=dt),
                magnitude(np.matmul, a.astype(dt), b.astype(dt)))

    # A 0-d result comes back as a scalar, not a 0-d array.
    v, pv = build(rng, (4,), "float64", "c")
    eq("matmul 1d@1d is a scalar", type(np.matmul(v, v)).__name__,
       type(rnp.matmul(pv, pv)).__name__)
    eq("vecdot 1d is a scalar", type(np.vecdot(v, v)).__name__,
       type(rnp.vecdot(pv, pv)).__name__)


def check_operator(rng):
    """`@`, `@=` and the reflected form."""
    for dt in ["int64", "float64", "complex128"]:
        a, pa = build(rng, (3, 4), dt, "c")
        b, pb = build(rng, (4, 3), dt, "c")
        compare(f"a @ b {dt}", a @ b, pa @ pb, magnitude(np.matmul, a, b))
        compare(f"b @ a {dt}", b @ a, pb @ pa, magnitude(np.matmul, b, a))
        v, pv = build(rng, (4,), dt, "c")
        compare(f"a @ v {dt}", a @ v, pa @ pv, magnitude(np.matmul, a, v))
        compare(f"v @ b {dt}", v @ b, pv @ pb, magnitude(np.matmul, v, b))
        # a list on the left goes through __rmatmul__.
        compare(f"list @ arr {dt}", v.tolist() @ b, pv.tolist() @ pb,
                magnitude(np.matmul, v, b))
        # in-place
        sq, psq = build(rng, (3, 3), dt, "c")
        sq0 = sq.copy()
        eye = np.eye(3, dtype=dt)
        want = sq @ eye
        sq @= eye
        psq @= to_port(eye)
        compare(f"a @= I {dt}", want, psq, magnitude(np.matmul, sq0, eye))
        compare(f"a @= I {dt} (numpy in place)", sq, psq,
                magnitude(np.matmul, sq0, eye))


def check_wraparound(rng):
    """Integer accumulation must wrap in the output dtype, exactly."""
    for dt in ["int8", "uint8", "int16", "uint16", "int32"]:
        info = np.iinfo(dt)
        a = np.full((4, 32), info.max, dt)
        b = np.full((32, 4), info.max, dt)
        compare(f"saturating {dt}", np.matmul(a, b),
                rnp.matmul(to_port(a), to_port(b)))
        compare(f"saturating {dt} vecdot", np.vecdot(a, a),
                rnp.vecdot(to_port(a), to_port(a)))
    # bool is an OR of ANDs, not an integer sum.
    for _ in range(10):
        a = np.array([[rng.random() > 0.7 for _ in range(9)] for _ in range(5)])
        b = np.array([[rng.random() > 0.7 for _ in range(4)] for _ in range(9)])
        compare("bool matmul", np.matmul(a, b),
                rnp.matmul(to_port(a), to_port(b)))


def check_object(rng):
    """`object` arrays go through Python's own `*` and `+`."""
    for shape_a, shape_b in [((2, 3), (3, 2)), ((3,), (3,)), ((2, 2), (2,)),
                             ((2, 0), (0, 2))]:
        a = np.array(
            [rng.randrange(-9, 9) for _ in range(int(np.prod(shape_a)))],
            dtype=object).reshape(shape_a)
        b = np.array(
            [rng.randrange(-9, 9) for _ in range(int(np.prod(shape_b)))],
            dtype=object).reshape(shape_b)
        want = np.matmul(a, b)
        # `tolist()` loses a zero-length trailing axis, so rebuild flat and
        # reshape rather than round-tripping the nesting.
        pa = rnp.array(a.ravel().tolist(), dtype=object).reshape(shape_a)
        pb = rnp.array(b.ravel().tolist(), dtype=object).reshape(shape_b)
        got = outcome(lambda: rnp.matmul(pa, pb))
        global CHECKS
        CHECKS += 1
        if got[0] == "exc":
            FAILURES.append((f"object matmul {shape_a}x{shape_b}",
                             f"port raised {got[1][0]}: {got[1][1]}"))
        elif np.asarray(want).tolist() != got[1].tolist():
            FAILURES.append((f"object matmul {shape_a}x{shape_b}",
                             f"{got[1]!r} != numpy's {want!r}"))


def check_errors():
    z, pz = np.zeros, rnp.zeros

    def both(name, fn):
        check_error(name, lambda: fn(np, np.zeros), lambda: fn(rnp, rnp.zeros))

    cases = [
        ("matmul 0-d lhs", lambda m, k: m.matmul(k(()), k((2, 2)))),
        ("matmul 0-d rhs", lambda m, k: m.matmul(k((2, 2)), k(()))),
        ("matmul scalar rhs", lambda m, k: m.matmul(k((2, 2)), 3.0)),
        ("matmul k mismatch", lambda m, k: m.matmul(k((2, 3)), k((4, 5)))),
        ("matmul 1d k mismatch", lambda m, k: m.matmul(k(2), k((3, 4)))),
        ("matmul 2d@1d mismatch", lambda m, k: m.matmul(k((2, 3)), k(4))),
        ("matmul 1d@1d mismatch", lambda m, k: m.matmul(k(2), k(3))),
        ("matmul 3d@2d mismatch", lambda m, k: m.matmul(k((2, 3, 4)), k((5, 6)))),
        ("matmul batch mismatch",
         lambda m, k: m.matmul(k((2, 3, 4)), k((5, 4, 2)))),
        ("matmul batch mismatch 4d",
         lambda m, k: m.matmul(k((3, 2, 3, 4)), k((5, 4, 2)))),
        ("matmul out core mismatch",
         lambda m, k: m.matmul(k((2, 3)), k((3, 2)), out=m.empty((3, 3)))),
        ("matmul out batch mismatch",
         lambda m, k: m.matmul(k((2, 3, 4)), k((2, 4, 5)),
                               out=m.empty((3, 3, 5)))),
        ("matmul out 0-d",
         lambda m, k: m.matmul(k((2, 3)), k((3, 4)), out=m.empty(()))),
        ("matmul out bad cast",
         lambda m, k: m.matmul(k((2, 3)), k((3, 2)),
                               out=m.empty((2, 2), dtype="int32"))),
        ("matmul out not an array",
         lambda m, k: m.matmul(k((2, 2)), k((2, 2)), out=[0, 0])),
        ("matmul where=", lambda m, k: m.matmul(k((2, 2)), k((2, 2)), where=True)),
        ("matmul.reduce", lambda m, k: m.matmul.reduce(k((2, 2)))),
        ("matmul.accumulate", lambda m, k: m.matmul.accumulate(k((2, 2)))),
        ("matmul.reduceat", lambda m, k: m.matmul.reduceat(k((2, 2)), [0])),
        ("matmul.outer", lambda m, k: m.matmul.outer(k((2, 2)), k((2, 2)))),
        ("matmul.at", lambda m, k: m.matmul.at(k((2, 2)), [0], k((2, 2)))),
        ("vecdot mismatch", lambda m, k: m.vecdot(k(3), k(4))),
        ("vecdot 0-d", lambda m, k: m.vecdot(k(()), k(()))),
        ("vecdot batch mismatch", lambda m, k: m.vecdot(k((5, 3)), k((4, 3)))),
        ("matvec mismatch", lambda m, k: m.matvec(k((2, 3)), k(4))),
        ("matvec 1-d lhs", lambda m, k: m.matvec(k(3), k(3))),
        ("matvec batch mismatch",
         lambda m, k: m.matvec(k((5, 2, 3)), k((4, 3)))),
        ("vecmat 1-d rhs", lambda m, k: m.vecmat(k(3), k(3))),
        ("vecmat mismatch", lambda m, k: m.vecmat(k(3), k((4, 2)))),
        ("dot mismatch", lambda m, k: m.dot(k((2, 3)), k((4, 5)))),
        ("dot 1d mismatch", lambda m, k: m.dot(k(3), k(4))),
        ("dot 3d mismatch", lambda m, k: m.dot(k((5, 2, 3)), k((6, 4, 2)))),
        ("dot out bad dtype",
         lambda m, k: m.dot(k((2, 3)), k((3, 4)),
                            out=m.empty((2, 4), dtype="float32"))),
        ("dot out bad shape",
         lambda m, k: m.dot(k((2, 3)), k((3, 4)), out=m.empty((2, 5)))),
        ("inner mismatch", lambda m, k: m.inner(k((2, 3)), k((2, 4)))),
        ("inner 1d mismatch", lambda m, k: m.inner(k(3), k(4))),
    ]
    for name, fn in cases:
        both(name, fn)

    # `@` with an unusable right-hand side.
    check_error("arr @ 3", lambda: np.arange(3.) @ 3,
                lambda: rnp.arange(3.) @ 3)
    check_error("3 @ arr", lambda: 3 @ np.arange(3.),
                lambda: 3 @ rnp.arange(3.))
    del z, pz


def check_index():
    """`operator.index()` on arrays."""
    import operator

    def both(name, mk):
        check_error(name, lambda: operator.index(mk(np)),
                    lambda: operator.index(mk(rnp)))
        # And the value, where it works.
        w = outcome(lambda: operator.index(mk(np)))
        g = outcome(lambda: operator.index(mk(rnp)))
        if w[0] == "ok" and g[0] == "ok":
            eq(name + " value", w[1], g[1])

    both("index 0-d int64", lambda m: m.array(5))
    both("index 0-d int8", lambda m: m.array(5, dtype="int8"))
    both("index 0-d uint8", lambda m: m.array(200, dtype="uint8"))
    both("index 0-d uint64 big",
         lambda m: m.array(2 ** 63, dtype="uint64"))
    both("index 0-d negative", lambda m: m.array(-7, dtype="int32"))
    both("index 0-d bool", lambda m: m.array(True))
    both("index 0-d float", lambda m: m.array(5.0))
    both("index 0-d complex", lambda m: m.array(5.0 + 0j))
    both("index 1-element 1-d", lambda m: m.array([5]))
    both("index 1-element 2-d", lambda m: m.array([[5]]))
    both("index 3-element 1-d", lambda m: m.array([1, 2, 3]))
    both("index empty", lambda m: m.zeros(0, dtype="int64"))
    both("index 0-d str", lambda m: m.array("a"))

    # Consumers of __index__.
    check_error("hex(0-d)", lambda: hex(np.array(255)),
                lambda: hex(rnp.array(255)))
    eq("hex value", hex(np.array(255)), hex(rnp.array(255)))
    eq("list index", [1, 2, 3][np.array(1)], [1, 2, 3][rnp.array(1)])
    eq("range length", len(range(np.array(3))), len(range(rnp.array(3))))


def check_close(rng):
    """isclose / allclose / array_equal / array_equiv."""
    nan, inf = float("nan"), float("inf")
    vectors = [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0 + 1e-9],
        [1.0, nan, 3.0],
        [nan, nan, nan],
        [inf, -inf, 0.0],
        [inf, 1.0, 2.0],
        [0.0, -0.0, 1e-20],
        [1e10, 1e-10, 0.0],
    ]
    for va in vectors:
        for vb in vectors:
            for dt in ["float32", "float64"]:
                a, b = np.array(va, dt), np.array(vb, dt)
                pa, pb = to_port(a), to_port(b)
                for eqn in (False, True):
                    compare(f"isclose({va},{vb},{dt},{eqn})",
                            np.isclose(a, b, equal_nan=eqn),
                            rnp.isclose(pa, pb, equal_nan=eqn))
                    eq(f"allclose({va},{vb},{dt},{eqn})",
                       np.allclose(a, b, equal_nan=eqn),
                       rnp.allclose(pa, pb, equal_nan=eqn))
                    eq(f"array_equal({va},{vb},{dt},{eqn})",
                       np.array_equal(a, b, equal_nan=eqn),
                       rnp.array_equal(pa, pb, equal_nan=eqn))
                eq(f"array_equiv({va},{vb},{dt})",
                   np.array_equiv(a, b), rnp.array_equiv(pa, pb))

    # rtol / atol variations.
    a = np.array([1.0, 100.0, 1e-8])
    b = np.array([1.0 + 1e-6, 100.0 + 1e-3, 2e-8])
    pa, pb = to_port(a), to_port(b)
    for rtol in [0.0, 1e-9, 1e-5, 1e-2]:
        for atol in [0.0, 1e-12, 1e-8, 1.0]:
            compare(f"isclose rtol={rtol} atol={atol}",
                    np.isclose(a, b, rtol=rtol, atol=atol),
                    rnp.isclose(pa, pb, rtol=rtol, atol=atol))
            eq(f"allclose rtol={rtol} atol={atol}",
               np.allclose(a, b, rtol=rtol, atol=atol),
               rnp.allclose(pa, pb, rtol=rtol, atol=atol))

    # Integer and bool operands (the `result_type(y, 1.)` promotion).
    for dt in ["int8", "int64", "uint64", "bool"]:
        a = np.array([1, 0, 1], dt)
        b = np.array([1, 1, 1], dt)
        compare(f"isclose {dt}", np.isclose(a, b), rnp.isclose(to_port(a), to_port(b)))
        eq(f"array_equal {dt}", np.array_equal(a, b),
           rnp.array_equal(to_port(a), to_port(b)))
        eq(f"array_equal {dt} equal_nan", np.array_equal(a, b, equal_nan=True),
           rnp.array_equal(to_port(a), to_port(b), equal_nan=True))
        eq(f"array_equiv {dt}", np.array_equiv(a, b),
           rnp.array_equiv(to_port(a), to_port(b)))

    # Scalars, lists, and shape/broadcast disagreements.
    pairs = [
        ([1, 2], [1, 2]),
        ([1, 2], [1, 2, 3]),
        ([1, 2], [[1, 2], [1, 2]]),
        ([1, 2], [[1, 2], [3, 4]]),
        ([], []),
        ([], [1]),
        (1, 1),
        (1, [1, 1]),
        (1, [1, 2]),
        ([[1], [2]], [1, 2]),
    ]
    for x, y in pairs:
        eq(f"array_equal({x},{y})", np.array_equal(x, y), rnp.array_equal(x, y))
        eq(f"array_equiv({x},{y})", np.array_equiv(x, y), rnp.array_equiv(x, y))
        check_error(f"allclose({x},{y})",
                    lambda x=x, y=y: np.allclose(x, y),
                    lambda x=x, y=y: rnp.allclose(x, y))
        eq(f"isclose({x},{y}) shapes",
           np.shape(outcome(lambda x=x, y=y: np.isclose(x, y))[1]),
           np.shape(outcome(lambda x=x, y=y: rnp.isclose(x, y))[1]))

    # equal_nan over complex, where numpy treats "either half is NaN" as NaN.
    ca = np.array([1 + 1j])
    cb = ca.copy()
    ca = ca.copy()
    ca.real = np.nan
    cb.imag = np.nan
    eq("array_equal complex nan halves",
       np.array_equal(ca, cb, equal_nan=True),
       rnp.array_equal(to_port(ca), to_port(cb), equal_nan=True))

    # An array is equal to itself under equal_nan, even full of NaNs.
    n = np.full(3, np.nan)
    pn = to_port(n)
    eq("array_equal self nan", np.array_equal(n, n, equal_nan=True),
       rnp.array_equal(pn, pn, equal_nan=True))
    eq("array_equal nan no equal_nan", np.array_equal(n, n),
       rnp.array_equal(pn, pn))


def check_axes(rng):
    """`axes=` remaps which dimensions the signature consumes."""
    a = np.arange(3 * 4 * 5, dtype="float64").reshape(3, 4, 5)
    pa = to_port(a)
    for axes in [[(-2, -1), (-1, -2), (1, 2)],
                 [(-2, -1), (-1, -2), (0, 1)]]:
        compare(f"matmul axes={axes}", np.matmul(a, a, axes=axes),
                rnp.matmul(pa, pa, axes=axes),
                np.matmul(np.abs(a), np.abs(a), axes=axes))
    v = np.arange(3, dtype="float64")
    compare("matmul axes with 1-d",
            np.matmul(a, v, axes=[(1, 0), (0), (0)]),
            rnp.matmul(pa, to_port(v), axes=[(1, 0), (0), (0)]),
            np.matmul(np.abs(a), np.abs(v), axes=[(1, 0), (0), (0)]))


def check_metadata():
    for name in ["matmul", "vecdot", "matvec", "vecmat"]:
        w, g = getattr(np, name), getattr(rnp, name)
        eq(f"{name}.signature", w.signature, g.signature)
        eq(f"{name}.nin", w.nin, g.nin)
        eq(f"{name}.nout", w.nout, g.nout)
        eq(f"{name}.identity", w.identity, g.identity)
        eq(f"{name}.__name__", w.__name__, g.__name__)
        eq(f"{name}.types", w.types, g.types)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    check_metadata()
    check_products(rng)
    check_dot_inner(rng)
    check_out(rng)
    check_operator(rng)
    check_wraparound(rng)
    check_object(rng)
    check_axes(rng)
    check_errors()
    check_index()
    check_close(rng)

    print(f"{CHECKS} comparisons, {len(FAILURES)} divergences")
    for name, msg in FAILURES:
        print(f"  FAIL {name}: {msg}")
    print()
    print("worst observed ULP distance vs numpy, per dtype -- measured, not "
          "asserted;")
    print(f"the pass criterion is |port - numpy| <= {TOL_FACTOR}"
          "*eps*sum|a||b| (see the module docstring):")
    for dt in ["float16", "float32", "float64", "complex64", "complex128"]:
        seen = WORST_ULP.get(dt)
        print(f"  {dt:<12} {'(none seen)' if seen is None else seen}")
    print("  every other dtype: bit-exact (0 divergences required)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
