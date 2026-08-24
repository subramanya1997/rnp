#!/usr/bin/env python3
"""Differential check for *structured* arrays: the port vs real numpy.

Same idiom as `dev_check.py`: both libraries are imported normally into this
one process (no import redirection), identical inputs are built on each side
from the *same* Python spec, and the results are compared.

Structured arrays cannot be compared through the buffer protocol the way
`dev_check.py` compares numeric ones (a field view is not contiguous, and the
port has no `ndarray.tobytes`), so the comparison is done on the *observable
surface* instead: dtype spelling, field offsets, shape, strides, `.base`
identity, `repr`/`str`, and the per-field `tolist()` values. Every one of those
is a thing numpy's own test suite asserts on, and a divergence in the stored
bytes shows up in the values.

Covered: field get/set on randomly generated structured dtypes (varying field
count, types, nesting, subarray fields, alignment, explicit offsets, titles),
multi-field selection, writeback through field views, structured scalars,
`astype` between structured dtypes, sorting/comparison, and `repr`.

Usage: .venv/bin/python harness/dev_check_struct.py [--seed N] [--rounds N]
"""
import argparse
import os
import random
import sys
import traceback

import numpy as np

_SHIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shim")
if _SHIM_DIR not in sys.path:
    sys.path.insert(0, _SHIM_DIR)

import rnp_numpy as rnp  # noqa: E402

FAILURES = []
CHECKS = 0


def cmp(name, want, got):
    """One comparison. `want` is numpy's answer, `got` is the port's."""
    global CHECKS
    CHECKS += 1
    if want != got:
        FAILURES.append((name, f"{got!r} != numpy's {want!r}"))


def call(fn, *a, **k):
    """Run `fn`, returning either its value or a `('!', ExcName, msg)` tag.

    Exceptions are compared as data so that error *parity* is checked with the
    same machinery as values -- which is the point of a differential harness.
    """
    try:
        return fn(*a, **k)
    except Exception as exc:  # noqa: BLE001
        return ("!", type(exc).__name__, str(exc))


# ---------------------------------------------------------------------------
# random structured dtype specs
# ---------------------------------------------------------------------------

#: Leaf formats the generator draws from. Deliberately mixes widths so that
#: natural offsets are irregular and alignment actually bites.
LEAF = ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8",
        "f4", "f8", "c8", "c16", "?", "S3", "U2"]

#: Leaves that `astype` between two structured dtypes can always convert.
NUMERIC_LEAF = ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "f4", "f8"]


def rand_leaf(rng, numeric_only=False):
    return rng.choice(NUMERIC_LEAF if numeric_only else LEAF)


def rand_spec(rng, depth=0, numeric_only=False):
    """A dtype *spec* as a plain Python object, identical for both libraries.

    Returns `(spec, align)` where `spec` is a list of field tuples suitable for
    `np.dtype(spec, align=align)`.
    """
    n = rng.randint(1, 4)
    fields = []
    for i in range(n):
        name = f"f{i}"
        roll = rng.random()
        if roll < 0.12 and depth < 2 and not numeric_only:
            # a nested structured field
            sub, _ = rand_spec(rng, depth + 1)
            fields.append((name, sub))
        elif roll < 0.28 and not numeric_only:
            # a subarray field
            shape = rng.choice([(2,), (3,), (2, 2), (1, 3)])
            fields.append((name, rand_leaf(rng), shape))
        elif roll < 0.36 and depth == 0 and not numeric_only:
            # a titled field: numpy's `(title, name)` spelling
            fields.append(((f"T{i}", name), rand_leaf(rng)))
        else:
            fields.append((name, rand_leaf(rng, numeric_only)))
    return fields, rng.random() < 0.3


def offsets_spec(rng):
    """A dict-form spec with explicit offsets and an oversized itemsize."""
    n = rng.randint(1, 3)
    formats = [rand_leaf(rng) for _ in range(n)]
    sizes = [np.dtype(f).itemsize for f in formats]
    offs, cur = [], 0
    for s in sizes:
        cur += rng.choice([0, 1, 4])
        offs.append(cur)
        cur += s
    return {"names": [f"f{i}" for i in range(n)], "formats": formats,
            "offsets": offs, "itemsize": cur + rng.choice([0, 3, 8])}


def leaf_names(dt, prefix=()):
    """Every *leaf* field path of a dtype, as tuples of names."""
    out = []
    for name in dt.names or ():
        sub = dt[name]
        base = sub.base if sub.shape else sub
        if base.names is not None:
            out.extend(leaf_names(base, prefix + (name,)))
        else:
            out.append(prefix + (name,))
    return out


def fill(a, rng):
    """Deterministically fill every leaf field of `a` in place."""
    for path in leaf_names(a.dtype):
        v = a
        for p in path:
            v = v[p]
        base = v.dtype
        size = int(np.prod(v.shape)) if v.shape else 1
        if base.kind in "iu":
            vals = [rng.randrange(0, 100) for _ in range(size)]
        elif base.kind in "fc":
            vals = [round(rng.uniform(-50, 50), 3) for _ in range(size)]
        elif base.kind == "b":
            vals = [rng.random() > 0.5 for _ in range(size)]
        elif base.kind == "S":
            vals = [bytes(rng.choice([b"ab", b"c", b""])) for _ in range(size)]
        else:
            vals = [rng.choice(["x", "yz", ""]) for _ in range(size)]
        # Plain Python values only: the *same* object graph must be handed to
        # both libraries, so no numpy array may be involved here.
        v[...] = _nest(vals, v.shape) if v.shape else vals[0]


def _nest(flat, shape):
    """Reshape a flat Python list into nested lists of `shape`."""
    if len(shape) <= 1:
        return list(flat)
    step = len(flat) // shape[0]
    return [_nest(flat[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def build(mod, spec, align, n, rng_seed):
    """The same array on either library: `mod` is `np` or `rnp`."""
    rng = random.Random(rng_seed)
    dt = mod.dtype(spec, align=align) if not isinstance(spec, dict) \
        else mod.dtype(spec)
    a = mod.zeros(n, dtype=dt)
    fill(a, rng)
    return a


def values(a):
    """A library-independent rendering of every leaf field's contents."""
    out = []
    for path in leaf_names(a.dtype):
        v = a
        for p in path:
            v = v[p]
        out.append((path, v.tolist()))
    return out


def dtype_key(dt):
    """Everything about a dtype that a view has to preserve, as plain data."""
    fields = None
    if dt.names is not None:
        fields = [(n, str(dt.fields[n][0]), dt.fields[n][1]) for n in dt.names]
    return (str(dt), dt.itemsize, dt.names, fields, dt.isalignedstruct,
            dt.kind, dt.alignment)


# ---------------------------------------------------------------------------
# the sections
# ---------------------------------------------------------------------------


def check_dtype_and_repr(spec, align, seed):
    wa = build(np, spec, align, 4, seed)
    ga = build(rnp, spec, align, 4, seed)
    tag = f"dtype({spec!r}, align={align})"
    cmp(f"{tag}.dtype", dtype_key(wa.dtype), dtype_key(ga.dtype))
    cmp(f"{tag} values", values(wa), values(ga))
    cmp(f"{tag} repr", repr(wa), repr(ga))
    cmp(f"{tag} str", str(wa), str(ga))
    cmp(f"{tag} shape/strides", (wa.shape, wa.strides), (ga.shape, ga.strides))
    return wa, ga


def check_single_fields(wa, ga, tag):
    for name in wa.dtype.names:
        wf, gf = wa[name], ga[name]
        cmp(f"{tag}[{name!r}].dtype", dtype_key(wf.dtype), dtype_key(gf.dtype))
        cmp(f"{tag}[{name!r}] shape/strides",
            (wf.shape, wf.strides), (gf.shape, gf.strides))
        cmp(f"{tag}[{name!r}].base is parent", wf.base is wa, gf.base is ga)
        cmp(f"{tag}[{name!r}] writeable",
            wf.flags.writeable, gf.flags.writeable)
        cmp(f"{tag}[{name!r}] values", values(wf), values(gf))
        cmp(f"{tag}[{name!r}] repr", repr(wf), repr(gf))
    # unknown / bad keys
    for key in ("nope", "", "F0"):
        cmp(f"{tag}[{key!r}] error",
            call(lambda: repr(wa[key])), call(lambda: repr(ga[key])))


def check_multi_fields(wa, ga, tag, rng):
    names = list(wa.dtype.names)
    for _ in range(min(3, len(names))):
        k = rng.randint(1, len(names))
        sel = rng.sample(names, k)
        wm, gm = call(lambda: wa[sel]), call(lambda: ga[sel])
        if isinstance(wm, tuple):
            cmp(f"{tag}[{sel!r}] error", wm, gm)
            continue
        cmp(f"{tag}[{sel!r}].dtype", dtype_key(wm.dtype), dtype_key(gm.dtype))
        cmp(f"{tag}[{sel!r}].itemsize preserved",
            wm.dtype.itemsize == wa.dtype.itemsize,
            gm.dtype.itemsize == ga.dtype.itemsize)
        cmp(f"{tag}[{sel!r}].base is parent", wm.base is wa, gm.base is ga)
        cmp(f"{tag}[{sel!r}] writeable",
            wm.flags.writeable, gm.flags.writeable)
        cmp(f"{tag}[{sel!r}] shape/strides",
            (wm.shape, wm.strides), (gm.shape, gm.strides))
        cmp(f"{tag}[{sel!r}] values", values(wm), values(gm))
        cmp(f"{tag}[{sel!r}] repr", repr(wm), repr(gm))
    # duplicate, unknown and mixed-type selections
    if names:
        for bad in ([names[0], names[0]], [names[0], "nope"], ["nope"],
                    [names[0], 1], [0, 1]):
            cmp(f"{tag}[{bad!r}] error",
                call(lambda: repr(wa[bad])), call(lambda: repr(ga[bad])))


def check_field_setitem(spec, align, seed, rng):
    """`a['f'] = v` and `a[['f','g']] = v`, then compare the whole array."""
    wa = build(np, spec, align, 4, seed)
    ga = build(rnp, spec, align, 4, seed)
    tag = f"setitem({spec!r}, align={align})"
    for name in wa.dtype.names:
        sub = wa.dtype[name]
        base = sub.base if sub.shape else sub
        if base.names is not None:
            continue
        if base.kind in "iu":
            vals = [rng.randrange(0, 50), [rng.randrange(0, 50)] * 4]
        elif base.kind in "fc":
            vals = [rng.uniform(-9, 9), [1.5, 2.5, 3.5, 4.5]]
        elif base.kind == "b":
            vals = [True, [True, False, True, False]]
        elif base.kind == "S":
            vals = [b"zz", [b"a", b"bb", b"", b"cc"]]
        else:
            vals = ["zz", ["a", "bb", "", "cc"]]
        for v in vals:
            w = call(lambda: wa.__setitem__(name, v))
            g = call(lambda: ga.__setitem__(name, v))
            cmp(f"{tag}[{name!r}] = {v!r}", w, g)
            cmp(f"{tag}[{name!r}] = {v!r} -> array", values(wa), values(ga))
            cmp(f"{tag}[{name!r}] = {v!r} -> repr", repr(wa), repr(ga))
    # writeback through a field *view* must reach the parent
    for name in wa.dtype.names:
        wf, gf = wa[name], ga[name]
        if wf.dtype.names is not None:
            continue
        blank = {"i": 0, "u": 0, "f": 0.0, "c": 0j, "b": False,
                 "S": b"", "U": ""}.get(wf.dtype.base.kind)
        if blank is None:
            continue
        w = call(lambda: wf.__setitem__(Ellipsis, blank))
        g = call(lambda: gf.__setitem__(Ellipsis, blank))
        cmp(f"{tag} writeback {name!r}", w, g)
        cmp(f"{tag} writeback {name!r} -> parent", values(wa), values(ga))
    # multi-field assignment
    names = list(wa.dtype.names)
    if len(names) >= 2:
        sel = names[:2]
        for v in (0, (1, 2)):
            w = call(lambda: wa.__setitem__(sel, v))
            g = call(lambda: ga.__setitem__(sel, v))
            cmp(f"{tag}[{sel!r}] = {v!r}", w, g)
            cmp(f"{tag}[{sel!r}] = {v!r} -> array", values(wa), values(ga))


def check_scalars(wa, ga, tag):
    for i in (0, 2, -1):
        wv, gv = wa[i], ga[i]
        cmp(f"{tag}[{i}] type name", type(wv).__name__, type(gv).__name__)
        cmp(f"{tag}[{i}] repr", repr(wv), repr(gv))
        cmp(f"{tag}[{i}] str", str(wv), str(gv))
        cmp(f"{tag}[{i}] len", call(len, wv), call(len, gv))
        cmp(f"{tag}[{i}] item", call(lambda: repr(wv.item())),
            call(lambda: repr(gv.item())))
        cmp(f"{tag}[{i}] tolist", call(lambda: repr(wv.tolist())),
            call(lambda: repr(gv.tolist())))
        cmp(f"{tag}[{i}] tuple",
            call(lambda: [repr(x) for x in tuple(wv)]),
            call(lambda: [repr(x) for x in tuple(gv)]))
        cmp(f"{tag}[{i}] dtype", dtype_key(wv.dtype), dtype_key(gv.dtype))
        cmp(f"{tag}[{i}] base is parent", wv.base is wa, gv.base is ga)
        cmp(f"{tag}[{i}] == itself", call(lambda: repr(wv == wa[i])),
            call(lambda: repr(gv == ga[i])))
        cmp(f"{tag}[{i}] == other", call(lambda: repr(wv == wa[1])),
            call(lambda: repr(gv == ga[1])))
        cmp(f"{tag}[{i}] != tuple", call(lambda: repr(wv == (1, 2))),
            call(lambda: repr(gv == (1, 2))))
        for j, name in enumerate(wa.dtype.names):
            cmp(f"{tag}[{i}][{name!r}]", call(lambda: repr(wv[name])),
                call(lambda: repr(gv[name])))
            cmp(f"{tag}[{i}][{j}]", call(lambda: repr(wv[j])),
                call(lambda: repr(gv[j])))
        cmp(f"{tag}[{i}]['nope']", call(lambda: repr(wv["nope"])),
            call(lambda: repr(gv["nope"])))
        cmp(f"{tag}[{i}][99]", call(lambda: repr(wv[99])),
            call(lambda: repr(gv[99])))
        # getfield / setfield
        cmp(f"{tag}[{i}].getfield(u1,0)",
            call(lambda: repr(wv.getfield(np.uint8, 0))),
            call(lambda: repr(gv.getfield(rnp.uint8, 0))))
        # writing through the scalar must reach the parent array
        name = wa.dtype.names[0]
        sub = wa.dtype[name]
        if not sub.shape and sub.names is None and sub.kind in "iufb":
            w = call(lambda: repr(wv.__setitem__(name, 3)))
            g = call(lambda: repr(gv.__setitem__(name, 3)))
            cmp(f"{tag}[{i}][{name!r}] = 3", w, g)
            cmp(f"{tag}[{i}][{name!r}] = 3 -> parent",
                values(wa), values(ga))


def check_astype(spec, align, seed, rng):
    """`astype` between two structured dtypes is field-by-field by position."""
    src, _ = spec, align
    dst, dalign = rand_spec(random.Random(seed ^ 0x5f5f), numeric_only=True)
    wa = build(np, src, align, 3, seed)
    ga = build(rnp, src, align, 3, seed)
    tag = f"astype({src!r} -> {dst!r})"
    w = call(lambda: repr(np.dtype(dst, align=dalign)))
    g = call(lambda: repr(rnp.dtype(dst, align=dalign)))
    cmp(f"{tag} dst dtype", w, g)
    wr = call(lambda: repr(wa.astype(np.dtype(dst, align=dalign))))
    gr = call(lambda: repr(ga.astype(rnp.dtype(dst, align=dalign))))
    cmp(f"{tag}", wr, gr)
    # astype back to itself, and to the parent's own dtype
    cmp(f"{tag} self", call(lambda: repr(wa.astype(wa.dtype))),
        call(lambda: repr(ga.astype(ga.dtype))))
    # a plain numeric target
    for t in ("i4", "f8"):
        cmp(f"{tag} -> {t}", call(lambda: repr(wa.astype(t))),
            call(lambda: repr(ga.astype(t))))
    # can_cast / promote_types on structured dtypes
    wd, gd = np.dtype(dst, align=dalign), rnp.dtype(dst, align=dalign)
    cmp(f"{tag} can_cast", call(lambda: np.can_cast(wa.dtype, wd)),
        call(lambda: rnp.can_cast(ga.dtype, gd)))
    cmp(f"{tag} promote_types",
        call(lambda: repr(np.promote_types(wa.dtype, wd))),
        call(lambda: repr(rnp.promote_types(ga.dtype, gd))))
    cmp(f"{tag} promote_types self",
        call(lambda: repr(np.promote_types(wa.dtype, wa.dtype))),
        call(lambda: repr(rnp.promote_types(ga.dtype, ga.dtype))))


def check_compare_and_sort(spec, align, seed):
    wa = build(np, spec, align, 4, seed)
    ga = build(rnp, spec, align, 4, seed)
    tag = f"cmp({spec!r})"
    wb = build(np, spec, align, 4, seed + 1)
    gb = build(rnp, spec, align, 4, seed + 1)
    cmp(f"{tag} a == a", call(lambda: repr(wa == wa)),
        call(lambda: repr(ga == ga)))
    cmp(f"{tag} a == b", call(lambda: repr(wa == wb)),
        call(lambda: repr(ga == gb)))
    cmp(f"{tag} a != b", call(lambda: repr(wa != wb)),
        call(lambda: repr(ga != gb)))
    cmp(f"{tag} a == 1", call(lambda: repr(wa == 1)),
        call(lambda: repr(ga == 1)))
    cmp(f"{tag} sort", call(lambda: repr(np.sort(wa))),
        call(lambda: repr(rnp.sort(ga))))
    cmp(f"{tag} argsort", call(lambda: repr(np.argsort(wa))),
        call(lambda: repr(rnp.argsort(ga))))


def check_construction(spec, align, seed):
    """`np.array([tuple, ...], dtype=...)` and friends."""
    wd = np.dtype(spec, align=align)
    tag = f"construct({spec!r})"
    zero = np.zeros(2, wd)
    rows = [tuple(x) for x in zero.tolist()]
    cmp(f"{tag} from tuples",
        call(lambda: repr(np.array(rows, dtype=wd))),
        call(lambda: repr(rnp.array(rows, dtype=rnp.dtype(spec, align=align)))))
    cmp(f"{tag} zeros repr", repr(np.zeros(3, wd)),
        repr(rnp.zeros(3, rnp.dtype(spec, align=align))))
    cmp(f"{tag} empty repr", repr(np.zeros(0, wd)),
        repr(rnp.zeros(0, rnp.dtype(spec, align=align))))


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--rounds", type=int, default=90)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    #: Hand-written specs that pin the cases the generator would only reach by
    #: luck; every one was cross-checked against numpy by hand first.
    FIXED = [
        ([("f0", "i4"), ("f1", "f8")], False),
        ([("f0", "i4"), ("f1", "f8"), ("f2", "u1")], False),
        ([("f0", "S3"), ("f1", "u4")], False),
        ([("f0", "i4"), ("f1", "f8")], True),
        ([("f0", "i1"), ("f1", "i8"), ("f2", "i2")], True),
        ([("f0", "i4", (2, 2)), ("f1", "f4")], False),
        ([("f0", "f8", (3,))], False),
        ([("f0", [("p", "i4"), ("q", "f8")]), ("f1", "i2")], False),
        ([("f0", [("p", "i4")]), ("f1", [("q", "u1"), ("r", "f4")])], True),
        ([(("T0", "f0"), "i4"), ("f1", "u2")], False),
        ([("f0", "U2"), ("f1", "?")], False),
        ([("f0", "c16"), ("f1", "c8")], False),
        ([("f0", "i4")], False),
        ([("f0", ">i4"), ("f1", "<f8")], False),
    ]

    specs = list(FIXED)
    for _ in range(args.rounds):
        specs.append(rand_spec(rng))
    for _ in range(args.rounds // 6):
        specs.append((offsets_spec(rng), False))

    for k, (spec, align) in enumerate(specs):
        seed = args.seed + k
        try:
            wa, ga = check_dtype_and_repr(spec, align, seed)
        except Exception:  # noqa: BLE001
            FAILURES.append((f"build({spec!r}, align={align})",
                             traceback.format_exc().strip().splitlines()[-1]))
            continue
        tag = f"a{k}"
        for fn, a in ((check_single_fields, (wa, ga, tag)),
                      (check_multi_fields, (wa, ga, tag, rng)),
                      (check_scalars, (wa, ga, tag)),
                      (check_field_setitem, (spec, align, seed, rng)),
                      (check_astype, (spec, align, seed, rng)),
                      (check_compare_and_sort, (spec, align, seed)),
                      (check_construction, (spec, align, seed))):
            try:
                fn(*a)
            except Exception:  # noqa: BLE001
                FAILURES.append((
                    f"{fn.__name__}({spec!r}, align={align})",
                    traceback.format_exc().strip().splitlines()[-1]))

    print(f"{CHECKS} comparisons, {len(FAILURES)} divergences")
    seen = {}
    for name, msg in FAILURES:
        seen.setdefault(msg.split("!=")[0][:80], []).append(name)
    for name, msg in FAILURES[:60]:
        print(f"  FAIL {name}: {msg}")
    if len(FAILURES) > 60:
        print(f"  ... and {len(FAILURES) - 60} more")
        print("  distinct messages:")
        for msg, names in list(seen.items())[:25]:
            print(f"    {len(names):5d}x {msg}  e.g. {names[0]}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
