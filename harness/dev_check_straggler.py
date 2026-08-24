#!/usr/bin/env python3
"""Differential check for the straggler cluster: the port vs real numpy.

Real numpy and the port's shim (`rnp_numpy`) are both imported normally in
this one process -- no import redirection -- identical inputs are fed to each
and the results are compared exactly: values, dtype spelling and shape for a
success, exception type *and* message for a failure.

Covers `np.lexsort`, and the `ndarray` methods `tobytes`, `clip`, `resize`,
`conj`/`conjugate`, `std`/`var`, `dump`/`dumps`, `getfield`/`setfield`,
`__sizeof__`, pickling (`__reduce__`/`__setstate__`) at every protocol,
weakref support, `np.frombuffer` and `np.c_`.

Usage: .venv/bin/python harness/dev_check_straggler.py [--seed N]
"""
import argparse
import io
import math
import os
import pickle
import random
import sys
import traceback
import weakref

import numpy as np

_SHIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shim")
if _SHIM_DIR not in sys.path:
    sys.path.insert(0, _SHIM_DIR)

import rnp_numpy as rnp  # noqa: E402

CHECKS = 0
FAILURES = []

NUMERIC = [
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "complex64", "complex128",
]
REAL = [d for d in NUMERIC if not d.startswith("complex")]
FLOATS = ["float16", "float32", "float64"]
SHAPES = [(6,), (3, 4), (2, 3, 4), (5, 1), (1, 5), (0,), (0, 3), (), (7,)]


# ---------------------------------------------------------------------------
# comparison plumbing
# ---------------------------------------------------------------------------

def _canon(v):
    """A hashable, NaN-stable rendering of any result either side produces.

    dtype-carrying values are matched *before* the plain Python types: numpy's
    `float64`/`complex128` scalars subclass `float`/`complex`, so testing the
    builtins first would hide a scalar-vs-0-d-array difference.
    """
    if isinstance(v, (bytes, bytearray)):
        return ("bytes", bytes(v))
    if v is None:
        return ("none",)
    if hasattr(v, "dtype") and hasattr(v, "shape") and hasattr(v, "tolist"):
        kind = "arr" if hasattr(v, "flags") else "scalar"
        return (kind, str(v.dtype), tuple(v.shape), _canon(v.tolist()))
    if hasattr(v, "dtype"):
        return ("scalar", str(v.dtype), _canon(v.item()))
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, float):
        return ("float", "nan" if math.isnan(v) else repr(v))
    if isinstance(v, complex):
        return ("complex", _canon(v.real), _canon(v.imag))
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, (list, tuple)):
        return ("seq", tuple(_canon(x) for x in v))
    return ("repr", repr(v))


def _run(fn, mod):
    try:
        return ("ok", _canon(fn(mod)))
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__, str(e))


def check(label, fn):
    """Run `fn(module)` against both libraries and compare the outcomes."""
    global CHECKS
    CHECKS += 1
    got_np = _run(fn, np)
    got_port = _run(fn, rnp)
    if got_np != got_port:
        FAILURES.append((label, f"port={got_port!r} numpy={got_np!r}"))


def make(mod, data, dtype=None, order=None):
    """Build the same array in whichever library `mod` is."""
    a = mod.array(data, dtype=dtype)
    if order == "F" and a.ndim > 1:
        a = a.T.copy().T
    return a


# ---------------------------------------------------------------------------
# lexsort
# ---------------------------------------------------------------------------

def section_lexsort(rng):
    for dt in NUMERIC:
        for nkeys in (1, 2, 3):
            vals = [[rng.randrange(0, 4) for _ in range(8)] for _ in range(nkeys)]
            check(f"lexsort {dt} x{nkeys}",
                  lambda m, v=vals, d=dt: m.lexsort(
                      tuple(m.array(x, dtype=d) for x in v)))
    # multi-dimensional keys and every legal axis
    for shape in [(3, 4), (2, 3, 4), (5, 1)]:
        n = 1
        for d in shape:
            n *= d
        vals = [rng.randrange(0, 3) for _ in range(n)]
        vals2 = [rng.randrange(0, 3) for _ in range(n)]
        for axis in list(range(-len(shape), len(shape))):
            check(f"lexsort {shape} axis={axis}",
                  lambda m, s=shape, a=vals, b=vals2, ax=axis: m.lexsort(
                      (m.array(a).reshape(s), m.array(b).reshape(s)), axis=ax))
    # string keys
    names = ["Hertz", "Galilei", "Hertz", "Aa", "zz", "Hertz"]
    firsts = ["Heinrich", "Galileo", "Gustav", "b", "a", "Aaron"]
    for dt in ("U8", "S8"):
        check(f"lexsort strings {dt}",
              lambda m, d=dt: m.lexsort((m.array(firsts, dtype=d),
                                         m.array(names, dtype=d))))
    # 0-d and empty keys
    check("lexsort 0d", lambda m: m.lexsort((m.array(1), m.array(2))))
    check("lexsort 0d axis0", lambda m: m.lexsort((m.array(1),), axis=0))
    check("lexsort empty", lambda m: m.lexsort((m.array([], dtype="int64"),)))
    check("lexsort 1d-as-keys", lambda m: m.lexsort(m.array([3, 1, 2])))
    check("lexsort lists", lambda m: m.lexsort([[1, 2, 1], [3, 1, 2]]))
    # negative and float keys, NaN ordering
    check("lexsort nan",
          lambda m: m.lexsort((m.array([1.0, float("nan"), 0.5, float("nan")]),)))
    check("lexsort neg", lambda m: m.lexsort((m.array([-3, 2, -3, 0]),)))
    # errors
    check("lexsort mismatch",
          lambda m: m.lexsort((m.arange(3), m.arange(4))))
    check("lexsort mismatch ndim",
          lambda m: m.lexsort((m.arange(3), m.arange(6).reshape(2, 3))))
    check("lexsort empty tuple", lambda m: m.lexsort(()))
    check("lexsort scalar arg", lambda m: m.lexsort(5))
    check("lexsort axis oob", lambda m: m.lexsort((m.arange(3),), axis=2))
    check("lexsort axis -2", lambda m: m.lexsort((m.arange(3),), axis=-2))
    check("lexsort 0d axis1", lambda m: m.lexsort((m.array(1),), axis=1))


# ---------------------------------------------------------------------------
# tobytes
# ---------------------------------------------------------------------------

def section_tobytes(rng):
    for dt in NUMERIC + ["S3", "U2"]:
        for shape in [(6,), (3, 4), (2, 3, 4), (0,), ()]:
            n = 1
            for d in shape:
                n *= d
            if dt in ("S3", "U2"):
                data = [f"x{rng.randrange(9)}" for _ in range(n)]
            else:
                data = [rng.randrange(0, 9) for _ in range(n)]
            if shape == ():
                data = data[0] if data else 0
            for order in (None, "C", "F", "A", "K"):
                check(f"tobytes {dt} {shape} {order}",
                      lambda m, d=data, t=dt, s=shape, o=order: (
                          m.array(d, dtype=t).reshape(s).tobytes()
                          if o is None else
                          m.array(d, dtype=t).reshape(s).tobytes(o)))
    # non-contiguous sources: transposes and strided slices
    base = [rng.randrange(0, 100) for _ in range(24)]
    for order in ("C", "F", "A", "K"):
        check(f"tobytes transpose {order}",
              lambda m, o=order: m.array(base).reshape(2, 3, 4).transpose(
                  1, 2, 0).tobytes(o))
        check(f"tobytes step2 {order}",
              lambda m, o=order: m.array(base).reshape(4, 6)[:, ::2].tobytes(o))
        check(f"tobytes reverse {order}",
              lambda m, o=order: m.array(base).reshape(4, 6)[::-1].tobytes(o))
        check(f"tobytes Forder {order}",
              lambda m, o=order: make(m, base, order="F").reshape(
                  4, 6).tobytes(o))
    # byte-swapped storage must dump its own bytes, not native ones
    for dt in ("int32", "float64", "complex64"):
        check(f"tobytes swapped {dt}",
              lambda m, d=dt: m.array([1, 2, 3], dtype=d).astype(
                  m.dtype(d).newbyteorder(">")).tobytes())
    check("tobytes bad order", lambda m: m.arange(4).tobytes("Q"))


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------

def section_clip(rng):
    bounds = [
        (None, None), (2, None), (None, 5), (2, 5), (5, 2), (0, 0),
        (-1, 300), (1.5, 3.5), (-2.5, None), (None, 1.5),
    ]
    for dt in NUMERIC:
        data = [rng.randrange(0, 9) for _ in range(8)]
        for lo, hi in bounds:
            check(f"clip {dt} {lo},{hi}",
                  lambda m, d=data, t=dt, a=lo, b=hi:
                      m.array(d, dtype=t).clip(a, b))
    # numpy scalars as bounds keep their own (strong) dtype
    for dt in ("uint8", "int16", "float32"):
        for bdt in ("int32", "float64", "int8"):
            check(f"clip {dt} npscalar {bdt}",
                  lambda m, t=dt, b=bdt: m.array([0, 4, 9], dtype=t).clip(
                      m.dtype(b).type(1), m.dtype(b).type(6)))
    # array-valued bounds broadcast
    check("clip array bounds",
          lambda m: m.arange(6).clip(m.array([1, 1, 1, 4, 4, 4]), 5))
    check("clip 2d array bounds",
          lambda m: m.arange(12).reshape(3, 4).clip(m.array([0, 2, 4, 6]), 9))
    # NaN propagation
    check("clip nan",
          lambda m: m.array([-2.0, float("nan"), 0.5, 3.0]).clip(-1, 1))
    check("clip nan bound",
          lambda m: m.array([1.0, 2.0]).clip(float("nan"), None))
    # out=
    for dt in ("uint8", "int32", "float64"):
        check(f"clip out {dt}",
              lambda m, t=dt: (lambda a, o: (a.clip(1, 5, o), o))(
                  m.arange(8, dtype=t), m.zeros(8, dtype=t)))
        check(f"clip out is returned {dt}",
              lambda m, t=dt: (lambda a, o: a.clip(1, 5, o) is o)(
                  m.arange(8, dtype=t), m.zeros(8, dtype=t)))
    check("clip out wrong shape",
          lambda m: m.arange(8).clip(1, 5, m.zeros(3)))
    check("clip out bswap",
          lambda m: (lambda a, o: (a.clip(1, 3, o), str(o.dtype)))(
              m.arange(5).astype(m.dtype("int32").newbyteorder(">")),
              m.zeros(5, dtype=m.dtype("int32").newbyteorder(">"))))
    # keyword spellings
    check("clip min kw", lambda m: m.arange(8).clip(min=3))
    check("clip max kw", lambda m: m.arange(8).clip(max=4))
    check("clip module both", lambda m: m.clip(m.arange(8), 2, 6))
    check("clip module none", lambda m: m.clip(m.arange(8), None, None))
    # 0-d and empty
    check("clip 0d", lambda m: m.array(7).clip(1, 5))
    check("clip empty", lambda m: m.array([], dtype="float64").clip(0, 1))
    # non-contiguous operand
    check("clip strided",
          lambda m: m.arange(12).reshape(3, 4)[:, ::2].clip(2, 8))
    check("clip transposed",
          lambda m: m.arange(12).reshape(3, 4).T.clip(2, 8))


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------

def section_resize(rng):
    cases = [
        ((3, 3), (5, 5)), ((3, 3), (3,)), ((3, 3), (2, 3, 3)),
        ((3, 3), (3, 2, 1)), ((4,), (2, 2)), ((4,), (0,)),
        ((1,), ()), ((), (1,)), ((6,), (2, 3)), ((6,), (10,)),
        ((2, 3), (6,)), ((2, 3), (1,)),
    ]
    for src, dst in cases:
        n = 1
        for d in src:
            n *= d
        data = list(range(n))
        check(f"resize {src}->{dst}",
              lambda m, s=src, t=dst, d=data: (
                  lambda a: (a.resize(t, refcheck=False), a))(
                      m.array(d, dtype="int64").reshape(s)))
        check(f"resize varargs {src}->{dst}",
              lambda m, s=src, t=dst, d=data: (
                  lambda a: (a.resize(*t, refcheck=False), a))(
                      m.array(d, dtype="int64").reshape(s)))
    for dt in NUMERIC:
        check(f"resize grow {dt}",
              lambda m, t=dt: (lambda a: (a.resize(9, refcheck=False), a))(
                  m.array([1, 2, 3], dtype=t)))
    # no-op spellings
    check("resize None", lambda m: (lambda a: (a.resize(None), a))(m.eye(3)))
    check("resize ()", lambda m: (lambda a: (a.resize(), a))(m.eye(3)))
    # errors
    check("resize str", lambda m: m.eye(3).resize("hi"))
    check("resize neg", lambda m: m.eye(3).resize(-1))
    check("resize order kw", lambda m: m.eye(3).resize(3, order="C"))
    check("resize refcheck str", lambda m: m.eye(3).resize(3, refcheck="hi"))
    check("resize view", lambda m: (lambda a: a[:2].resize(9))(m.arange(4)))
    check("resize view same size",
          lambda m: (lambda v: (v.resize((4, 1), refcheck=False), v))(
              m.arange(4)[...]))
    check("resize empty view",
          lambda m: (lambda v: (v.resize((0, 10)), v.shape))(
              m.zeros((10, 0), int)[...]))
    check("resize aliased", lambda m: (lambda a, b: a.resize((5, 5)))(
        *(lambda x: (x, x))(m.zeros((2, 2)))))
    check("resize weakref", lambda m: (lambda a, r: a.resize((5, 1)))(
        *(lambda x: (x, weakref.ref(x)))(m.eye(3))))
    check("resize aliased refcheck=False",
          lambda m: (lambda a, b: (a.resize((5, 5), refcheck=False), a.shape))(
              *(lambda x: (x, x))(m.zeros((2, 2)))))
    # np.resize (the module-level tiling version)
    for src, dst in [((4,), (8,)), ((4,), (2, 3)), ((6,), (2, 2)),
                     ((2, 3), (3, 3))]:
        n = 1
        for d in src:
            n *= d
        check(f"np.resize {src}->{dst}",
              lambda m, s=src, t=dst, k=n: m.resize(
                  m.arange(k).reshape(s), t))


# ---------------------------------------------------------------------------
# conjugate
# ---------------------------------------------------------------------------

def section_conjugate(rng):
    for dt in NUMERIC:
        data = [1, 2, 3]
        check(f"conj {dt}", lambda m, t=dt, d=data: m.array(d, dtype=t).conj())
        check(f"conjugate {dt}",
              lambda m, t=dt, d=data: m.array(d, dtype=t).conjugate())
        check(f"conj identity {dt}",
              lambda m, t=dt, d=data: (lambda a: a.conj() is a)(
                  m.array(d, dtype=t)))
    for dt in ("complex64", "complex128"):
        check(f"conj values {dt}",
              lambda m, t=dt: m.array([1 - 1j, 1 + 1j, 23 + 23j], dtype=t).conj())
        check(f"conj strided {dt}",
              lambda m, t=dt: m.array(
                  [1 - 1j, 2 + 2j, 3 - 3j, 4 + 4j], dtype=t)[::2].conj())
        check(f"conj out {dt}",
              lambda m, t=dt: (lambda a, o: (a.conjugate(o), o))(
                  m.array([1 - 1j, 1 + 1j], dtype=t),
                  m.zeros(2, dtype=t)))
        check(f"conj out identity {dt}",
              lambda m, t=dt: (lambda a, o: a.conjugate(o) is o)(
                  m.array([1 - 1j, 1 + 1j], dtype=t),
                  m.zeros(2, dtype=t)))
    check("conj 0d complex", lambda m: m.array(5j).conjugate())
    check("conj 0d int", lambda m: m.array(5).conjugate())
    for dt in ("S3", "U3"):
        check(f"conj error {dt}",
              lambda m, t=dt: m.array(["ab", "cd"], dtype=t).conj())


# ---------------------------------------------------------------------------
# var / std
# ---------------------------------------------------------------------------

def section_varstd(rng):
    for dt in NUMERIC:
        data = [rng.randrange(0, 20) for _ in range(12)]
        for axis in (None, 0, 1, -1):
            for ddof in (0, 1, 2):
                check(f"var {dt} ax={axis} ddof={ddof}",
                      lambda m, d=data, t=dt, a=axis, k=ddof: m.array(
                          d, dtype=t).reshape(3, 4).var(axis=a, ddof=k))
                check(f"std {dt} ax={axis} ddof={ddof}",
                      lambda m, d=data, t=dt, a=axis, k=ddof: m.array(
                          d, dtype=t).reshape(3, 4).std(axis=a, ddof=k))
    for dt in FLOATS + ["complex64", "complex128"]:
        for out_dt in FLOATS + ["complex64", "complex128"]:
            check(f"var {dt} dtype={out_dt}",
                  lambda m, t=dt, o=out_dt: m.eye(3, dtype=t).var(
                      axis=1, dtype=o))
            check(f"std {dt} dtype={out_dt} axisNone",
                  lambda m, t=dt, o=out_dt: m.eye(3, dtype=t).std(
                      axis=None, dtype=o))
    for axis in (None, 0, 1):
        check(f"var keepdims ax={axis}",
              lambda m, a=axis: m.arange(24, dtype="float64").reshape(
                  2, 3, 4).var(axis=a, keepdims=True))
        check(f"std keepdims ax={axis}",
              lambda m, a=axis: m.arange(24, dtype="float64").reshape(
                  2, 3, 4).std(axis=a, keepdims=True))
        check(f"mean keepdims ax={axis}",
              lambda m, a=axis: m.arange(24, dtype="float64").reshape(
                  2, 3, 4).mean(axis=a, keepdims=True))
    # out=
    for fn in ("var", "std", "mean"):
        check(f"{fn} out",
              lambda m, f=fn: (lambda a, o: (getattr(a, f)(axis=1, out=o), o))(
                  m.eye(3), m.zeros(3)))
        check(f"{fn} out identity",
              lambda m, f=fn: (lambda a, o: getattr(a, f)(axis=1, out=o) is o)(
                  m.eye(3), m.zeros(3)))
        check(f"{fn} out bad shape",
              lambda m, f=fn: getattr(m.eye(3), f)(axis=1, out=m.empty(2)))
        check(f"{fn} out bad ndim",
              lambda m, f=fn: getattr(m.eye(3), f)(axis=1, out=m.empty((2, 2))))
        check(f"{fn} axis error", lambda m, f=fn: getattr(m.arange(10), f)(axis=2))
        check(f"{fn} axis error type",
              lambda m, f=fn: _axis_error_class(m, f))
    # where=
    mask = [[True, False, True, True], [True, True, False, False],
            [False, True, True, True]]
    for fn in ("var", "std", "mean"):
        for axis in (None, 0, 1):
            check(f"{fn} where ax={axis}",
                  lambda m, f=fn, a=axis, w=mask: getattr(
                      m.arange(12, dtype="float64").reshape(3, 4), f)(
                          axis=a, where=m.array(w)))
            check(f"np.{fn} where ax={axis}",
                  lambda m, f=fn, a=axis, w=mask: getattr(m, f)(
                      m.arange(12, dtype="float64").reshape(3, 4),
                      axis=a, where=m.array(w)))
    # correction= is numpy 2's alias for ddof
    check("var correction",
          lambda m: m.arange(12, dtype="float64").reshape(3, 4).var(
              axis=1, correction=1))
    # module-level spellings
    for fn in ("var", "std"):
        check(f"np.{fn} list", lambda m, f=fn: getattr(m, f)([1, 2, 3, 4]))
        check(f"np.{fn} ddof", lambda m, f=fn: getattr(m, f)([1.0, 2, 3], ddof=1))
    # non-contiguous
    check("var transposed",
          lambda m: m.arange(12, dtype="float64").reshape(3, 4).T.var(axis=0))
    check("std strided",
          lambda m: m.arange(12, dtype="float64").reshape(3, 4)[:, ::2].std(axis=1))
    # errors
    check("var on strings", lambda m: m.array(["a", "b"]).var())


def _axis_error_class(m, fn):
    exc = m.exceptions.AxisError
    try:
        getattr(m.arange(10), fn)(axis=2)
    except exc:
        return "AxisError"
    except Exception as e:  # noqa: BLE001
        return f"other:{type(e).__name__}"
    return "no raise"


# ---------------------------------------------------------------------------
# pickling, dump/dumps, weakref
# ---------------------------------------------------------------------------

def _arrays_for_pickle(m, rng):
    """A spread of layouts to round-trip: contiguous, strided, F, 0-d, empty."""
    base = m.arange(24, dtype="int64").reshape(2, 3, 4)
    out = [
        m.arange(6),
        base,
        base.transpose(1, 2, 0),
        base[:, ::2],
        m.arange(12).reshape(3, 4).T,
        make(m, list(range(12)), order="F").reshape(3, 4),
        m.array(5),
        m.array([], dtype="float32"),
        m.arange(1000),
        m.array([1.5, float("nan"), -0.0]),
        m.array([1 + 2j, 3 - 4j]),
        m.array(["ab", "cde"], dtype="U4"),
        m.array([b"ab", b"cde"], dtype="S4"),
        m.arange(5).astype(m.dtype("int32").newbyteorder(">")),
        m.arange(5).astype(m.dtype("float64").newbyteorder(">")),
        m.zeros((0, 3)),
    ]
    return out


def section_pickle(rng):
    n = len(_arrays_for_pickle(np, rng))
    for i in range(n):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            check(f"pickle[{i}] proto={proto}",
                  lambda m, k=i, p=proto: pickle.loads(
                      pickle.dumps(_arrays_for_pickle(m, rng)[k], p)))
        check(f"reduce shape[{i}]",
              lambda m, k=i: (lambda r: (len(r), r[1][1], r[1][2],
                                         r[2][0], r[2][1], str(r[2][2]),
                                         r[2][3], r[2][4]))(
                  _arrays_for_pickle(m, rng)[k].__reduce__()))
        check(f"dumps roundtrip[{i}]",
              lambda m, k=i: pickle.loads(
                  _arrays_for_pickle(m, rng)[k].dumps()))
        check(f"setstate roundtrip[{i}]",
              lambda m, k=i: (lambda a: (lambda b: (b.__setstate__(
                  a.__reduce__()[2]), b))(m.empty(0, dtype=a.dtype)))(
                      _arrays_for_pickle(m, rng)[k]))
    # base is the pickle bytes above numpy's copy threshold, None below it
    for size in (10, 100, 125, 126, 500, 1000):
        check(f"pickle base kind n={size}",
              lambda m, s=size: type(
                  pickle.loads(pickle.dumps(m.arange(s), 4)).base).__name__)
        check(f"pickle writeable n={size}",
              lambda m, s=size: pickle.loads(
                  pickle.dumps(m.arange(s), 4)).flags.writeable)
    # dump / load through a file
    for i in range(4):
        check(f"dump file[{i}]",
              lambda m, k=i: _dump_roundtrip(m, _arrays_for_pickle(m, rng)[k]))
        check(f"dump fileobj[{i}]",
              lambda m, k=i: _dump_fileobj(m, _arrays_for_pickle(m, rng)[k]))
        check(f"save load[{i}]",
              lambda m, k=i: _save_roundtrip(m, _arrays_for_pickle(m, rng)[k]))


def _dump_roundtrip(m, a):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.pkl")
        a.dump(p)
        with open(p, "rb") as fh:
            return pickle.load(fh)


def _dump_fileobj(m, a):
    buf = io.BytesIO()
    a.dump(buf)
    buf.seek(0)
    return pickle.load(buf)


def _save_roundtrip(m, a):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.npy")
        m.save(p, a)
        return m.load(p, allow_pickle=True)


def section_weakref(rng):
    for i in range(6):
        check(f"weakref alive[{i}]",
              lambda m, k=i: (lambda a: weakref.ref(a)() is a)(m.arange(k + 1)))
        check(f"weakref dead[{i}]",
              lambda m, k=i: _weakref_dies(m, k + 1))
        check(f"weakref count[{i}]",
              lambda m, k=i: (lambda a: (weakref.ref(a),
                                         weakref.getweakrefcount(a))[1])(
                  m.arange(k + 1)))
    check("weakref callback", lambda m: _weakref_callback(m))


def _weakref_dies(m, n):
    a = m.arange(n)
    r = weakref.ref(a)
    del a
    return r() is None


def _weakref_callback(m):
    seen = []
    a = m.arange(3)
    weakref.ref(a, lambda _r: seen.append(1))
    del a
    return len(seen)


# ---------------------------------------------------------------------------
# getfield / setfield, __sizeof__, frombuffer, c_
# ---------------------------------------------------------------------------

def section_fields(rng):
    pairs = [
        ("int64", "int32", 0), ("int64", "int32", 4), ("int64", "int8", 0),
        ("int64", "int8", 7), ("complex128", "float64", 0),
        ("complex128", "float64", 8), ("float64", "int64", 0),
        ("int32", "int16", 0), ("int32", "int16", 2), ("int32", "uint8", 1),
    ]
    for base, field, off in pairs:
        check(f"getfield {base}->{field}@{off}",
              lambda m, b=base, f=field, o=off: m.arange(
                  6, dtype=b).getfield(m.dtype(f), o))
        check(f"setfield {base}->{field}@{off}",
              lambda m, b=base, f=field, o=off: (
                  lambda a: (a.setfield(3, m.dtype(f), o), a))(
                      m.arange(6, dtype=b)))
    check("getfield too big",
          lambda m: m.arange(4, dtype="int32").getfield(m.dtype("int64"), 0))
    check("getfield neg offset",
          lambda m: m.arange(4, dtype="int32").getfield(m.dtype("int8"), -1))
    check("getfield offset past end",
          lambda m: m.arange(4, dtype="int32").getfield(m.dtype("int16"), 3))


def section_sizeof(rng):
    # Absolute byte counts are implementation-defined; the *relations* numpy's
    # own tests assert are not.
    for dt, elem in [("int32", 4), ("int64", 8), ("float32", 4), ("float64", 8)]:
        for length in (10, 50, 100, 500):
            check(f"sizeof>data {dt} {length}",
                  lambda m, t=dt, n=length, e=elem: sys.getsizeof(
                      m.arange(n, dtype=t)) > n * e)
    check("sizeof view < owner",
          lambda m: (lambda d: sys.getsizeof(d[...]) < sys.getsizeof(d))(
              m.ones(100)))
    check("sizeof reshape grows",
          lambda m: (lambda d: sys.getsizeof(d) < sys.getsizeof(
              d.reshape(100, 1, 1).copy()))(m.ones(100)))
    check("sizeof resize shrinks",
          lambda m: (lambda d, o: (d.resize(50, refcheck=False),
                                   o > sys.getsizeof(d))[1])(
              *(lambda x: (x, sys.getsizeof(x)))(m.ones(100))))
    check("sizeof extra arg", lambda m: m.ones(4).__sizeof__("a"))


def section_frombuffer(rng):
    for dt in NUMERIC:
        data = [rng.randrange(0, 9) for _ in range(6)]
        check(f"frombuffer {dt}",
              lambda m, d=data, t=dt: m.frombuffer(
                  m.array(d, dtype=t).tobytes(), dtype=t))
        check(f"frombuffer count {dt}",
              lambda m, d=data, t=dt: m.frombuffer(
                  m.array(d, dtype=t).tobytes(), dtype=t, count=3))
        check(f"frombuffer offset {dt}",
              lambda m, d=data, t=dt: m.frombuffer(
                  m.array(d, dtype=t).tobytes(),
                  dtype=t, offset=m.dtype(t).itemsize))
    check("frombuffer ragged",
          lambda m: m.frombuffer(b"abc", dtype="int32"))
    check("frombuffer too few",
          lambda m: m.frombuffer(b"abcd", dtype="int32", count=4))
    check("frombuffer F roundtrip",
          lambda m: m.frombuffer(
              m.arange(12).reshape(3, 4).tobytes("F"), dtype="int64"))


def section_c_(rng):
    cases = [
        ([1, 2, 3], [4, 5, 6]),
        ([1.0, 2.0], [3, 4]),
        ([[1, 2], [3, 4]], [[5], [6]]),
        ([1, 2, 3], [4, 5, 6], [7, 8, 9]),
    ]
    for i, parts in enumerate(cases):
        check(f"c_[{i}]",
              lambda m, p=parts: m.c_[tuple(m.array(x) for x in p)])
    check("c_ single", lambda m: m.c_[m.array([1, 2, 3])])
    check("c_ mismatch",
          lambda m: m.c_[m.array([1, 2, 3]), m.array([1, 2])])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260823)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for section in (
        section_lexsort, section_tobytes, section_clip, section_resize,
        section_conjugate, section_varstd, section_pickle, section_weakref,
        section_fields, section_sizeof, section_frombuffer, section_c_,
    ):
        section(rng)

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
