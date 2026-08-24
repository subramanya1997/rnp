#!/usr/bin/env python3
"""Differential check for NEP 50 promotion: the Rust port vs real numpy.

Same idea and idiom as `harness/dev_check.py` -- both libraries live in this
one process, identical inputs are handed to each, and the two answers are
compared -- but the subject here is *promotion*, which is mostly about dtypes
and exceptions rather than array contents:

  * `result_type` / `promote_types` / `can_cast` over 2, 3 and 4 participants,
    concrete and weak mixed together (NEP 50 promotion is neither associative
    nor commutative, so a 3-operand answer is not the 2-operand one folded);
  * array construction with out-of-range integers for every integer dtype,
    from every kind of source object, comparing the exception *type and
    message* verbatim;
  * Python integers too wide for any integer dtype;
  * weak-scalar arithmetic and comparison promotion.

Real numpy is imported normally; the port is imported as `rnp_numpy` from
`shim/`, exactly as `harness/crosscheck.py` does it. No import redirection is
involved, so `np` below is always genuinely numpy 2.5.2.

Usage: .venv/bin/python harness/dev_check_nep50.py [-v]
"""
import argparse
import itertools
import random
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shim"))
import rnp_numpy as rnp  # noqa: E402

FAILURES = []
CHECKS = 0

#: The 14 numeric dtypes, by name.
DTYPES = ["bool", "int8", "int16", "int32", "int64",
          "uint8", "uint16", "uint32", "uint64",
          "float16", "float32", "float64", "complex64", "complex128"]

INT_DTYPES = ["int8", "int16", "int32", "int64",
              "uint8", "uint16", "uint32", "uint64"]

CASTING = ["no", "equiv", "safe", "same_kind", "unsafe"]


# ---------------------------------------------------------------------------
# Outcome capture and comparison
# ---------------------------------------------------------------------------

def normalise(v):
    """A module-independent description of one result.

    Dtypes become their `str()` (identical text in both libraries), arrays
    become dtype + shape + values, and numpy scalars become dtype + value, so
    a port answer and a numpy answer are directly comparable.
    """
    if isinstance(v, (np.dtype, rnp.dtype)):
        return ("dtype", str(v))
    if isinstance(v, (bool, int, float, complex, str, bytes, type(None))):
        return ("py", repr(v))
    if isinstance(v, tuple):
        return ("tuple", tuple(normalise(x) for x in v))
    dt = getattr(v, "dtype", None)
    if dt is not None:
        # Read the values through the object's own `tolist`, never through
        # `np.asarray`: real numpy cannot adopt a port array or scalar, and
        # the point is to compare what each library produced on its own.
        shape = tuple(getattr(v, "shape", ()))
        try:
            vals = v.tolist()
        except Exception:  # noqa: BLE001
            vals = v
        return ("arr", str(dt), shape, repr(vals))
    return ("other", repr(v))


def outcome(fn, mod, with_message):
    """Run `fn(mod)`, capturing either its normalised value or its error."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ("ok",) + normalise(fn(mod))
    except Exception as exc:  # noqa: BLE001
        if with_message:
            return ("err", type(exc).__name__, str(exc))
        return ("err", type(exc).__name__)


def compare(label, fn, with_message=False):
    """Compare `fn(numpy)` with `fn(port)`."""
    global CHECKS
    CHECKS += 1
    want = outcome(fn, np, with_message)
    got = outcome(fn, rnp, with_message)
    if want != got:
        FAILURES.append((label, f"port {got!r} != numpy {want!r}"))


# ---------------------------------------------------------------------------
# Operand specs: a description both libraries can build their own operand from
# ---------------------------------------------------------------------------

def build(mod, spec):
    kind, payload = spec
    if kind == "dt":
        return mod.dtype(payload)
    if kind == "arr":
        return mod.zeros(2, dtype=payload)
    if kind == "arr0d":
        return mod.array(1, dtype=payload)
    if kind == "sc":
        return mod.dtype(payload).type(1)
    if kind == "py":
        return payload
    raise AssertionError(kind)


def spec_name(spec):
    kind, payload = spec
    return f"{kind}({payload!r})"


DTYPE_SPECS = [("dt", d) for d in DTYPES]
ARRAY_SPECS = [("arr", d) for d in DTYPES]
SCALAR_SPECS = [("sc", d) for d in DTYPES]
#: Weak Python literals, including two that no integer dtype can hold.
WEAK_SPECS = [("py", v) for v in (True, False, 1, -1, 300, 2**63, 2**63 + 1,
                                  2**64 - 1, 2**100, -2**100, -2**63 - 1,
                                  1.0, -2.5, 1j)]


# ---------------------------------------------------------------------------
# Section 1: result_type / promote_types / can_cast
# ---------------------------------------------------------------------------

def section_promotion(rng):
    # Every concrete pair, all three functions.
    for a, b in itertools.product(DTYPE_SPECS, DTYPE_SPECS):
        compare(f"result_type({spec_name(a)}, {spec_name(b)})",
                lambda m, a=a, b=b: m.result_type(build(m, a), build(m, b)))
        compare(f"promote_types({spec_name(a)}, {spec_name(b)})",
                lambda m, a=a, b=b: m.promote_types(build(m, a), build(m, b)))
        for c in CASTING:
            compare(f"can_cast({spec_name(a)}, {spec_name(b)}, {c})",
                    lambda m, a=a, b=b, c=c:
                        m.can_cast(build(m, a), build(m, b), casting=c))

    # Single-argument result_type, including the huge int that answers
    # `object` only when it is alone.
    for s in DTYPE_SPECS + ARRAY_SPECS + SCALAR_SPECS + WEAK_SPECS:
        compare(f"result_type({spec_name(s)})",
                lambda m, s=s: m.result_type(build(m, s)))

    # Concrete + weak pairs, both orders.
    for a, w in itertools.product(DTYPE_SPECS + ARRAY_SPECS + SCALAR_SPECS,
                                  WEAK_SPECS):
        compare(f"result_type({spec_name(a)}, {spec_name(w)})",
                lambda m, a=a, w=w: m.result_type(build(m, a), build(m, w)))
        compare(f"result_type({spec_name(w)}, {spec_name(a)})",
                lambda m, a=a, w=w: m.result_type(build(m, w), build(m, a)))

    # Three and four participants, every ordering of a hand-picked set that is
    # known to separate whole-sequence promotion from a pairwise left fold,
    # plus a large random sample over the whole pool.
    tricky = [
        ("uint8", "int8", "float16"),
        ("int8", "uint16", "float16"),
        ("uint16", "int8", "float16"),
        ("int16", "uint16", "float32"),
        ("float16", "int64", "uint64"),
        ("int64", "uint64", "float16"),
        ("uint8", "int8", "float32"),
        ("int32", "uint32", "float16"),
        ("uint32", "int16", "float16"),
        ("int8", "uint8", "int16"),
    ]
    for triple in tricky:
        for order in itertools.permutations(triple):
            specs = tuple(("dt", d) for d in order)
            compare("result_type(" + ", ".join(spec_name(s) for s in specs) + ")",
                    lambda m, specs=specs:
                        m.result_type(*[build(m, s) for s in specs]))

    pool = DTYPE_SPECS + ARRAY_SPECS + SCALAR_SPECS + WEAK_SPECS
    for n in (3, 4, 5):
        for _ in range(250):
            specs = tuple(rng.choice(pool) for _ in range(n))
            compare("result_type(" + ", ".join(spec_name(s) for s in specs) + ")",
                    lambda m, specs=specs:
                        m.result_type(*[build(m, s) for s in specs]))

    # can_cast also has to keep rejecting Python scalars outright.
    for w in WEAK_SPECS:
        for d in DTYPE_SPECS:
            compare(f"can_cast({spec_name(w)}, {spec_name(d)})",
                    lambda m, w=w, d=d: m.can_cast(build(m, w), build(m, d)))

    # min_scalar_type, which is where the huge ints land on `object`.
    for v in (0, 1, -1, 127, 128, 255, 256, 2**31, 2**63, 2**64 - 1,
              2**64, 2**100, -2**63, -2**63 - 1, -2**100, 0.5, 1e40, 3j):
        compare(f"min_scalar_type({v!r})",
                lambda m, v=v: m.min_scalar_type(v))


# ---------------------------------------------------------------------------
# Section 2: out-of-range construction, exception type *and* message
# ---------------------------------------------------------------------------

def oob_values(name):
    info = np.iinfo(name)
    return [info.min, info.max, info.min - 1, info.max + 1,
            0, 1, -1, 300, -300, 2**63, 2**64 - 1, 2**64, 2**100, -2**100]


def section_oob_construction():
    for name in INT_DTYPES:
        for v in oob_values(name):
            # A Python int, the case NEP 50 tightened.
            compare(f"array([{v}], dtype={name})",
                    lambda m, v=v, n=name: m.array([v], dtype=n),
                    with_message=True)
            compare(f"array({v}, dtype={name})",
                    lambda m, v=v, n=name: m.array(v, dtype=n),
                    with_message=True)
            compare(f"array([[{v}]], dtype={name})",
                    lambda m, v=v, n=name: m.array([[v]], dtype=n),
                    with_message=True)
            compare(f"array(({v},), dtype={name})",
                    lambda m, v=v, n=name: m.array((v,), dtype=n),
                    with_message=True)
            compare(f"asarray([{v}], dtype={name})",
                    lambda m, v=v, n=name: m.asarray([v], dtype=n),
                    with_message=True)
            compare(f"full((2,), {v}, dtype={name})",
                    lambda m, v=v, n=name: m.full((2,), v, dtype=n),
                    with_message=True)
            # The scalar constructor, which already had the rule.
            compare(f"{name}({v})",
                    lambda m, v=v, n=name: m.dtype(n).type(v),
                    with_message=True)
            # A decimal string, which numpy parses and then range-checks the
            # same way (`np.array(['-129'], dtype=np.int8)` overflows).
            compare(f"array(['{v}'], dtype={name})",
                    lambda m, v=v, n=name: m.array([str(v)], dtype=n),
                    with_message=True)
            # A Python float, converted with int() first.
            if abs(v) < 2**53:
                compare(f"array([{v}.0], dtype={name})",
                        lambda m, v=v, n=name: m.array([float(v)], dtype=n),
                        with_message=True)
                compare(f"array({v}.0, dtype={name})",
                        lambda m, v=v, n=name: m.array(float(v), dtype=n),
                        with_message=True)
            # A *numpy* scalar leaf: signed targets range-check it, unsigned
            # ones wrap it. Only build the ones the source dtype can hold.
            for src in INT_DTYPES:
                si = np.iinfo(src)
                if not (si.min <= v <= si.max):
                    continue
                compare(f"array([{src}({v})], dtype={name})",
                        lambda m, v=v, n=name, s=src:
                            m.array([m.dtype(s).type(v)], dtype=n),
                        with_message=True)
                compare(f"array({src}({v}), dtype={name})",
                        lambda m, v=v, n=name, s=src:
                            m.array(m.dtype(s).type(v), dtype=n),
                        with_message=True)
        # bool and the inexact dtypes never range-check.
        for other in ["bool", "float16", "float32", "float64", "complex128"]:
            for v in (300, -300, 2**64, 2**100, -2**100):
                compare(f"array([{v}], dtype={other})",
                        lambda m, v=v, n=other: m.array([v], dtype=n),
                        with_message=True)
                compare(f"array({v}, dtype={other})",
                        lambda m, v=v, n=other: m.array(v, dtype=n),
                        with_message=True)


# ---------------------------------------------------------------------------
# Section 3: Python integers too wide for any integer dtype
# ---------------------------------------------------------------------------

HUGE = [2**64, -2**64, 2**100, -2**100, 2**200, 2**1000, 2**2000, -2**2000,
        2**63 + 1, -(2**63) - 1]


def section_huge_ints():
    for v in HUGE:
        compare(f"array({v!r})", lambda m, v=v: m.array(v), with_message=True)
        compare(f"array([{v!r}])", lambda m, v=v: m.array([v]),
                with_message=True)
        compare(f"array([1, {v!r}])", lambda m, v=v: m.array([1, v]),
                with_message=True)
        compare(f"array([{v!r}, 1.5])", lambda m, v=v: m.array([v, 1.5]),
                with_message=True)
        compare(f"array({v!r}, dtype=object)",
                lambda m, v=v: m.array(v, dtype=object), with_message=True)
        compare(f"min_scalar_type({v!r})", lambda m, v=v: m.min_scalar_type(v))
        compare(f"result_type({v!r})", lambda m, v=v: m.result_type(v))
        for d in DTYPES:
            compare(f"array({v!r}, dtype={d})",
                    lambda m, v=v, d=d: m.array(v, dtype=d), with_message=True)
            compare(f"array([{v!r}], dtype={d})",
                    lambda m, v=v, d=d: m.array([v], dtype=d),
                    with_message=True)
            compare(f"result_type({d}, {v!r})",
                    lambda m, v=v, d=d: m.result_type(m.dtype(d), v))
            # Arithmetic and comparison against an array of every dtype.
            compare(f"zeros({d}) + {v!r}",
                    lambda m, v=v, d=d: m.zeros(3, dtype=d) + v,
                    with_message=True)
            compare(f"zeros({d}) < {v!r}",
                    lambda m, v=v, d=d: m.zeros(3, dtype=d) < v,
                    with_message=True)
            compare(f"zeros({d}) >= {v!r}",
                    lambda m, v=v, d=d: m.zeros(3, dtype=d) >= v,
                    with_message=True)
            compare(f"zeros({d}) == {v!r}",
                    lambda m, v=v, d=d: m.zeros(3, dtype=d) == v,
                    with_message=True)


# ---------------------------------------------------------------------------
# Section 4: weak-scalar arithmetic and comparison promotion
# ---------------------------------------------------------------------------

WEAK_VALUES = [True, 1, 0, -1, 3, 300, -300, 70000, 2**31, 2**63,
               2**64 - 1, 1.0, 0.5, -2.5, 3e100, 1j, 2 + 3j]

ARITH = ["add", "subtract", "multiply", "power"]
COMPARE_OPS = ["equal", "not_equal", "less", "less_equal", "greater",
               "greater_equal"]


def section_weak_scalars():
    for d in DTYPES:
        for v in WEAK_VALUES:
            for op in ARITH:
                compare(f"{op}(ones({d}), {v!r})",
                        lambda m, d=d, v=v, op=op:
                            getattr(m, op)(m.ones(3, dtype=d), v),
                        with_message=True)
                compare(f"{op}({v!r}, ones({d}))",
                        lambda m, d=d, v=v, op=op:
                            getattr(m, op)(v, m.ones(3, dtype=d)),
                        with_message=True)
            for op in COMPARE_OPS:
                compare(f"{op}(full({d}, 10), {v!r})",
                        lambda m, d=d, v=v, op=op:
                            getattr(m, op)(m.full(3, 10, dtype=d), v),
                        with_message=True)
                compare(f"{op}({v!r}, full({d}, 10))",
                        lambda m, d=d, v=v, op=op:
                            getattr(m, op)(v, m.full(3, 10, dtype=d)),
                        with_message=True)
            # 0-d arrays take the same path as n-d ones under NEP 50. The
            # result goes back through `asarray` because numpy hands back a
            # *scalar* for a 0-d operation while the port hands back a 0-d
            # array -- a separate, pre-existing packaging difference that has
            # nothing to do with promotion; dtype and value are still compared
            # in full.
            compare(f"asarray(array(10, {d}) + {v!r})",
                    lambda m, d=d, v=v: m.asarray(m.array(10, dtype=d) + v),
                    with_message=True)
            compare(f"asarray(array(10, {d}) < {v!r})",
                    lambda m, d=d, v=v: m.asarray(m.array(10, dtype=d) < v),
                    with_message=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    section_promotion(rng)
    section_oob_construction()
    section_huge_ints()
    section_weak_scalars()

    print(f"{CHECKS} comparisons, {len(FAILURES)} divergences")
    shown = FAILURES if args.verbose else FAILURES[:60]
    for name, msg in shown:
        print(f"  FAIL {name}: {msg}")
    if len(shown) < len(FAILURES):
        print(f"  ... and {len(FAILURES) - len(shown)} more (-v for all)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
