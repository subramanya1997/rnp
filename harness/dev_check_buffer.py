#!/usr/bin/env python3
"""Differential check for buffer adoption: the Rust port vs real numpy.

Both libraries are imported normally in this process (no import redirection)
and handed the *same* source objects, so every check below is a statement
about two implementations wrapping identical bytes.

The sections are:

* ``ndarray(shape, dtype, buffer, offset, strides, order)`` over a grid of
  shapes, strides, offsets and dtypes -- contents, flags, ``.base`` identity,
  writability, and the exact exception type+message for every documented
  failure mode.
* ``frombuffer`` including its ``count``/``offset`` edge cases.
* Writeback in both directions: mutate through the adopted array and require
  the *source object* to see it, and mutate the source and require the array
  to see it.  This is what "zero-copy" actually means, so it is checked for
  the port and for numpy with the same assertions.
* The array protocol: ``__array__`` (with and without numpy 2.x's ``copy``
  keyword) and ``__array_interface__``.

Usage: .venv/bin/python harness/dev_check_buffer.py [--seed N]
"""
import argparse
import array as _pyarray
import mmap
import os
import sys
import tempfile
import traceback
import warnings

import numpy as np

import _rnp

_SHIM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shim")
if _SHIM_DIR not in sys.path:
    sys.path.insert(0, _SHIM_DIR)

FAILURES = []
CHECKS = 0


def _fail(name, msg):
    FAILURES.append((name, str(msg).strip().splitlines()[0][:300]))


def eq(name, want, got):
    """Compare two plain Python values."""
    global CHECKS
    CHECKS += 1
    if want != got:
        _fail(name, f"port {got!r} != numpy {want!r}")


def eq_array(name, want, got_port):
    """Compare a numpy array with a port array, read back through its bytes."""
    global CHECKS
    CHECKS += 1
    try:
        got = np.asarray(got_port)
        if want.dtype != got.dtype:
            raise AssertionError(f"dtype {got.dtype} != numpy's {want.dtype}")
        if want.shape != got.shape:
            raise AssertionError(f"shape {got.shape} != numpy's {want.shape}")
        np.testing.assert_array_equal(got, want, strict=True)
    except Exception as exc:  # noqa: BLE001
        _fail(name, exc)


def _outcome(fn):
    """Run `fn`, returning ("ok", value) or ("exc", type name, message)."""
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001
        return ("exc", type(exc).__name__, str(exc))


def _describe(outcome):
    """Summarise an outcome without ever repr-ing array *contents*.

    An adopted array may legitimately point outside its buffer (numpy permits
    a negative offset), so printing one can segfault the printer.
    """
    if outcome[0] == "exc":
        return f"raised {outcome[1]}: {outcome[2]}"
    v = outcome[1]
    if isinstance(v, (np.ndarray, _rnp.ndarray)):
        return f"ok <{type(v).__name__} shape={v.shape} dtype={v.dtype}>"
    return f"ok {v!r}"


def same_outcome(name, np_fn, port_fn, compare=None):
    """Require both libraries to succeed identically, or fail identically."""
    global CHECKS
    CHECKS += 1
    a = _outcome(np_fn)
    b = _outcome(port_fn)
    if a[0] != b[0]:
        _fail(name, f"numpy {_describe(a)} but port {_describe(b)}")
        return None
    if a[0] == "exc":
        if a[1] != b[1]:
            _fail(name, f"port raised {b[1]}, numpy raised {a[1]}")
        elif a[2] != b[2]:
            _fail(name, f"port message {b[2]!r} != numpy {a[2]!r}")
        return None
    if compare is not None:
        compare(name, a[1], b[1])
    return (a[1], b[1])


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

DTYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32",
          "int64", "uint64", "float32", "float64", "complex64", "complex128",
          "bool", "float16"]

#: (label, factory, is_writable). The factory is called once per library so
#: neither implementation can see the other's mutations.
PAYLOAD = bytes((i * 7 + 3) % 256 for i in range(256))


def sources():
    """(label, factory, writable). `factory(lib)` builds a *fresh* source.

    The factory takes the library so that an ndarray source is native to the
    implementation under test: `.base` collapsing only ever walks onto arrays
    of the same type, so handing the port a real numpy array as its buffer
    would compare two structurally different situations.
    """
    def native(lib):
        buf = bytearray(PAYLOAD)
        if lib is np:
            return np.frombuffer(buf, "uint8")
        return _rnp.frombuffer(buf, _rnp.dtype("uint8"))

    return [
        ("bytes", lambda lib: PAYLOAD, False),
        ("bytearray", lambda lib: bytearray(PAYLOAD), True),
        ("memoryview(bytearray)", lambda lib: memoryview(bytearray(PAYLOAD)), True),
        ("array.array('B')", lambda lib: _pyarray.array('B', PAYLOAD), True),
        ("ndarray-uint8", native, True),
    ]


def make(lib, shape, dtype, buf, offset=0, strides=None, order=None):
    if lib is np:
        return np.ndarray(shape, dtype, buf, offset, strides, order)
    return _rnp.ndarray(shape, _rnp.dtype(dtype), buf, offset, strides, order)


def base_kind(arr, src):
    """A comparable description of `.base` that works across libraries."""
    b = arr.base
    if b is None:
        return "None"
    if b is src:
        return "src"
    if isinstance(b, (np.ndarray, _rnp.ndarray)):
        return "ndarray"
    return type(b).__name__


def flags_of(arr):
    f = arr.flags
    return (bool(f.c_contiguous), bool(f.f_contiguous),
            bool(f.writeable), bool(f.owndata), bool(f.aligned))


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def check_constructor_grid():
    """ndarray(shape, dtype, buffer, offset, strides) across many layouts."""
    cases = []
    for shape in [(4,), (8,), (2, 3), (3, 2), (2, 2, 2), (0,), (), (1, 5), (5, 1)]:
        cases.append((shape, None, None))
    # explicit offsets
    for shape in [(4,), (2, 3)]:
        for off in [0, 1, 8, 16, 24]:
            cases.append((shape, off, None))
    # explicit strides, including negative, zero and over-large ones
    cases += [
        ((4,), 0, (8,)),
        ((4,), 0, (16,)),
        ((4,), 0, (0,)),
        ((4,), 48, (-8,)),
        ((3,), 64, (-16,)),
        ((2, 3), 0, (48, 8)),
        ((2, 3), 0, (8, 16)),
        ((3, 2), 8, (16, 8)),
        ((4,), 0, (1,)),
        ((4,), 0, (3,)),
    ]
    for label, factory, _w in sources():
        for dtype in DTYPES:
            for shape, off, strides in cases:
                nm = f"ndarray({shape}, {dtype}, {label}, off={off}, st={strides})"
                s_np, s_pt = factory(np), factory(_rnp)
                for order in (None, "C", "F"):
                    if order is not None and strides is not None:
                        continue
                    res = same_outcome(
                        nm + f" order={order}",
                        lambda: make(np, shape, dtype, s_np, off or 0, strides, order),
                        lambda: make(_rnp, shape, dtype, s_pt, off or 0, strides, order),
                    )
                    if res is None:
                        continue
                    a, b = res
                    eq_array(nm + " contents", a, b)
                    eq(nm + " flags", flags_of(a), flags_of(b))
                    eq(nm + " base", base_kind(a, s_np), base_kind(b, s_pt))
                    eq(nm + " strides", a.strides, b.strides)


def check_constructor_errors():
    """Every documented failure mode, message included."""
    b = bytes(range(64))
    bad = [
        ("too small", (100,), "int32", 0, None),
        ("offset past end", (2,), "int32", 100, None),
        ("negative offset ok", (2,), "int32", -4, None),
        ("misaligned offset", (2,), "int32", 3, None),
        ("strides escape", (4,), "int32", 0, (100,)),
        ("strides escape back", (4,), "int32", 0, (-4,)),
        ("strides length", (4,), "int32", 0, (4, 4)),
        ("huge shape", (1 << 40,), "int32", 0, None),
        ("negative dim", (-2,), "int32", 0, None),
        ("auto -1", (-1,), "int32", 0, None),
        ("object dtype", (2,), object, 0, None),
        ("zero shape", (0,), "int32", 0, None),
    ]
    for name, shape, dt, off, st in bad:
        same_outcome(
            f"err {name}",
            lambda: np.ndarray(shape, dt, b, off, st),
            lambda: _rnp.ndarray(shape, _rnp.dtype(dt) if dt is not object
                                 else _rnp.dtype("O"), b, off, st),
        )
    # non-buffer objects and bad orders
    for name, obj in [("int", 5), ("str", "abc"), ("None-ish", object())]:
        same_outcome(
            f"err buffer={name}",
            lambda: np.ndarray((2,), "int32", obj),
            lambda: _rnp.ndarray((2,), _rnp.dtype("int32"), obj),
        )
    for o in ["Q", "", "zzz"]:
        same_outcome(
            f"err order={o!r}",
            lambda: np.ndarray((2,), "int32", None, 0, None, o),
            lambda: _rnp.ndarray((2,), _rnp.dtype("int32"), None, 0, None, o),
        )
    # strides without a buffer are validated against the implied allocation
    for st in [(8,), (4,), (0,), (-4,)]:
        same_outcome(
            f"err no-buffer strides {st}",
            lambda: np.ndarray((4,), "int32", None, 0, st),
            lambda: _rnp.ndarray((4,), _rnp.dtype("int32"), None, 0, st),
        )


def check_allocating_constructor():
    """`ndarray(shape)` with no buffer: shape/dtype/flags, not contents."""
    for shape in [(), (3,), (2, 3), (0, 4), (2, 3, 4)]:
        for dtype in [None, "float64", "int32", "complex128", "bool"]:
            nm = f"alloc ndarray({shape}, {dtype})"
            a = np.ndarray(shape) if dtype is None else np.ndarray(shape, dtype)
            b = (_rnp.ndarray(shape) if dtype is None
                 else _rnp.ndarray(shape, _rnp.dtype(dtype)))
            eq(nm + " shape", a.shape, b.shape)
            eq(nm + " dtype", str(a.dtype), str(b.dtype))
            eq(nm + " flags", flags_of(a), flags_of(b))
            eq(nm + " base", base_kind(a, None), base_kind(b, None))
            eq(nm + " strides", a.strides, b.strides)
        for order in ("C", "F"):
            nm = f"alloc ndarray({shape}, order={order})"
            a = np.ndarray(shape, "float64", None, 0, None, order)
            b = _rnp.ndarray(shape, _rnp.dtype("float64"), None, 0, None, order)
            eq(nm + " strides", a.strides, b.strides)
            eq(nm + " flags", flags_of(a), flags_of(b))


def check_base_chain():
    """`.base` for adopted arrays and for views of them."""
    for label, factory, _w in sources():
        s_np, s_pt = factory(np), factory(_rnp)
        a = np.ndarray((8,), "int32", s_np)
        b = _rnp.ndarray((8,), _rnp.dtype("int32"), s_pt)
        eq(f"base {label}", base_kind(a, s_np), base_kind(b, s_pt))
        for nm, va, vb in [
            ("slice", a[1:5], b[1:5]),
            ("step", a[::2], b[::2]),
            ("reshape", a.reshape(2, 4), b.reshape(2, 4)),
            ("T", a.reshape(2, 4).T, b.reshape(2, 4).T),
            ("view", a.view("uint8"), b.view(_rnp.dtype("uint8"))),
        ]:
            eq(f"base {label} {nm}", base_kind(va, s_np), base_kind(vb, s_pt))
            eq(f"base {label} {nm} is-parent",
               va.base is a, vb.base is b)
            # a view of a view collapses onto the same array in both
            eq(f"base {label} {nm} nested",
               base_kind(va[...], s_np), base_kind(vb[...], s_pt))
            eq(f"base {label} {nm} writeable",
               bool(va.flags.writeable), bool(vb.flags.writeable))


def check_writeback():
    """Mutating the array must be visible in the source, and vice versa."""
    for label, factory, writable in sources():
        for dtype, val in [("int32", 12345), ("float64", -2.5),
                           ("uint8", 200), ("int64", -9999)]:
            nm = f"writeback {label} {dtype}"
            s_np, s_pt = factory(np), factory(_rnp)
            a = np.ndarray((4,), dtype, s_np, 0)
            b = _rnp.ndarray((4,), _rnp.dtype(dtype), s_pt, 0)
            eq(nm + " writeable", bool(a.flags.writeable), bool(b.flags.writeable))
            if not writable:
                # Writing must fail identically.
                same_outcome(nm + " readonly write",
                             lambda: a.__setitem__(0, val),
                             lambda: b.__setitem__(0, val))
                continue
            a[1] = val
            b[1] = val
            # ...and the *source objects* must now agree, byte for byte.
            eq(nm + " src bytes",
               bytes(memoryview(s_np).cast("B")),
               bytes(memoryview(s_pt).cast("B")))
            # The other direction: poke the source, read through the array.
            memoryview(s_np).cast("B")[8:12] = b"\xde\xad\xbe\xef"
            memoryview(s_pt).cast("B")[8:12] = b"\xde\xad\xbe\xef"
            eq_array(nm + " src->arr", a, b)
            # A view writes through too.
            a[::2] = 0
            b[::2] = 0
            eq(nm + " view writeback",
               bytes(memoryview(s_np).cast("B")),
               bytes(memoryview(s_pt).cast("B")))
            eq_array(nm + " after view write", a, b)


def check_frombuffer():
    payload = bytes((i * 11 + 5) % 256 for i in range(96))
    for label, factory, writable in sources():
        for dtype in DTYPES:
            for count, offset in [(-1, 0), (2, 0), (0, 0), (1, 8), (-1, 8),
                                  (-1, 3), (3, 16), (-1, 96), (100, 0),
                                  (-1, -1), (-1, 200), (2, 90)]:
                nm = f"frombuffer({label}, {dtype}, {count}, {offset})"
                s_np, s_pt = factory(np), factory(_rnp)
                res = same_outcome(
                    nm,
                    lambda: np.frombuffer(s_np, dtype, count, offset),
                    lambda: _rnp.frombuffer(s_pt, _rnp.dtype(dtype), count, offset),
                )
                if res is None:
                    continue
                a, b = res
                eq_array(nm + " contents", a, b)
                eq(nm + " flags", flags_of(a), flags_of(b))
                eq(nm + " base", base_kind(a, s_np), base_kind(b, s_pt))
                eq(nm + " slice base",
                   base_kind(a[...], s_np), base_kind(b[...], s_pt))
    # errors that do not depend on the source object
    for name, fn_np, fn_pt in [
        ("object dtype",
         lambda: np.frombuffer(payload, object),
         lambda: _rnp.frombuffer(payload, _rnp.dtype("O"))),
        ("str source",
         lambda: np.frombuffer("abc", "int8"),
         lambda: _rnp.frombuffer("abc", _rnp.dtype("int8"))),
        ("int source",
         lambda: np.frombuffer(7, "int8"),
         lambda: _rnp.frombuffer(7, _rnp.dtype("int8"))),
        ("empty",
         lambda: np.frombuffer(b"", "int32"),
         lambda: _rnp.frombuffer(b"", _rnp.dtype("int32"))),
    ]:
        same_outcome(f"frombuffer err {name}", fn_np, fn_pt)
    # frombuffer writeback
    for dtype in ["int32", "float64", "uint16"]:
        nm = f"frombuffer writeback {dtype}"
        s_np, s_pt = bytearray(payload), bytearray(payload)
        a = np.frombuffer(s_np, dtype)
        b = _rnp.frombuffer(s_pt, _rnp.dtype(dtype))
        eq(nm + " writeable", bool(a.flags.writeable), bool(b.flags.writeable))
        a[2] = 7
        b[2] = 7
        eq(nm + " src", bytes(s_np), bytes(s_pt))


def check_mmap():
    """A real mmap: adoption, writeback to disk, and lifetime."""
    global CHECKS
    for lib, ctor in [("numpy", np), ("port", _rnp)]:
        with tempfile.NamedTemporaryFile(prefix="rnpbuf", delete=False) as fh:
            path = fh.name
            fh.write(b"\0" * 64)
        try:
            f = open(path, "r+b")
            mm = mmap.mmap(f.fileno(), 64)
            if ctor is np:
                arr = np.ndarray((8,), "int64", mm)
            else:
                arr = _rnp.ndarray((8,), _rnp.dtype("int64"), mm)
            CHECKS += 1
            if arr.base is not mm:
                _fail(f"mmap {lib} base", f"base is {type(arr.base).__name__}")
            arr[3] = 0x1122334455667788
            mm.flush()
            del arr
            mm.close()
            f.close()
            with open(path, "rb") as rf:
                data = rf.read()
            CHECKS += 1
            expect = (0x1122334455667788).to_bytes(8, sys.byteorder)
            if data[24:32] != expect:
                _fail(f"mmap {lib} writeback", f"disk has {data[24:32].hex()}")
        finally:
            os.unlink(path)

    # frombuffer holds the export, so closing must raise for both.
    for lib, fn in [("numpy", lambda m: np.frombuffer(m, np.uint8)),
                    ("port", lambda m: _rnp.frombuffer(m, _rnp.dtype("uint8")))]:
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            tmp.write(b"asdf")
            tmp.flush()
            mm = mmap.mmap(tmp.fileno(), 0)
            arr = fn(mm)
            CHECKS += 1
            try:
                mm.close()
                _fail(f"frombuffer {lib} mmap close", "close() did not raise")
            except BufferError:
                pass
            del arr
            mm.close()

    # A view outliving its parent keeps the mapping alive.
    for lib, ctor in [("numpy", np), ("port", _rnp)]:
        with tempfile.NamedTemporaryFile(prefix="rnpbuf", delete=False) as fh:
            path = fh.name
            fh.write(bytes(range(64)))
        try:
            f = open(path, "r+b")
            mm = mmap.mmap(f.fileno(), 64)
            if ctor is np:
                parent = np.ndarray((8,), "int64", mm)
            else:
                parent = _rnp.ndarray((8,), _rnp.dtype("int64"), mm)
            child = parent[2:5]
            del parent, mm
            f.close()
            CHECKS += 1
            if int(child[0]) != int.from_bytes(bytes(range(16, 24)), sys.byteorder):
                _fail(f"mmap {lib} orphan view", f"got {int(child[0])}")
            del child
        finally:
            os.unlink(path)


def check_array_protocol():
    """__array__ (with and without `copy=`) and __array_interface__."""
    # `__array__` must hand back an array of the library that is asking, which
    # is exactly what a real third-party object does when only one array
    # library is installed.
    def payload(lib, dtype):
        if lib is np:
            a = np.arange(6.0)
            return a if dtype is None else a.astype(dtype)
        a = _rnp.arange(0.0, 6.0, 1.0)
        return a if dtype is None else a.astype(dtype)

    class Modern:
        def __init__(self, lib):
            self.lib = lib

        def __array__(self, dtype=None, copy=None):
            return payload(self.lib, dtype)

    class Legacy:
        def __init__(self, lib):
            self.lib = lib

        def __array__(self, dtype=None):
            return payload(self.lib, dtype)

    class Bad:
        def __init__(self, lib):
            self.lib = lib

        def __array__(self, dtype=None, copy=None):
            return 5

    for cls in (Modern, Legacy):
        for dtype in [None, "float64", "int32", "float32"]:
            nm = f"asarray({cls.__name__}, {dtype})"
            same_outcome(
                nm,
                lambda: np.asarray(cls(np), dtype),
                lambda: _rnp.asarray(cls(_rnp), None if dtype is None
                                     else _rnp.dtype(dtype)),
                compare=lambda n, a, b: (eq_array(n, a, b),
                                         eq(n + " base", base_kind(a, None),
                                            base_kind(b, None))),
            )
            same_outcome(
                nm + " copy=True",
                lambda: np.array(cls(np), dtype, copy=True),
                lambda: _rnp.array(cls(_rnp), None if dtype is None
                                   else _rnp.dtype(dtype), copy=True),
                compare=eq_array,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            same_outcome(
                f"array({cls.__name__}, copy=False)",
                lambda: np.array(cls(np), copy=False),
                lambda: _rnp.array(cls(_rnp), copy=False),
                compare=eq_array,
            )
        same_outcome(
            f"array({cls.__name__}, copy=None)",
            lambda: np.array(cls(np), copy=None),
            lambda: _rnp.array(cls(_rnp), copy=None),
            compare=eq_array,
        )
    same_outcome("asarray(Bad)",
                 lambda: np.asarray(Bad(np)),
                 lambda: _rnp.asarray(Bad(_rnp)))

    # __array_interface__ over live memory, both directions of writeback.
    for dtype in ["int32", "float64", "uint8", "int64", "float32"]:
        holder_np = np.arange(8, dtype=dtype)
        holder_pt = _rnp.arange(0, 8, 1, _rnp.dtype(dtype))

        class Iface:
            def __init__(self, arr):
                self._a = arr

            @property
            def __array_interface__(self):
                return self._a.__array_interface__

        nm = f"__array_interface__ {dtype}"
        res = same_outcome(
            nm,
            lambda: np.asarray(Iface(holder_np)),
            lambda: _rnp.asarray(Iface(holder_pt)),
        )
        if res is None:
            continue
        a, b = res
        eq_array(nm, a, b)
        eq(nm + " flags", flags_of(a), flags_of(b))
        # zero-copy: poking the holder shows through
        holder_np[3] = 42
        holder_pt[3] = 42
        eq_array(nm + " zero-copy", a, b)


def check_ndarray_subclassing():
    """`ndarray.__new__(subtype, ...)` and `view(type=)`."""
    class Sub(_rnp.ndarray):
        pass

    class NpSub(np.ndarray):
        pass

    buf_np, buf_pt = bytearray(range(64)), bytearray(range(64))
    a = NpSub.__new__(NpSub, (8,), "int64", buf_np)
    b = Sub.__new__(Sub, (8,), _rnp.dtype("int64"), buf_pt)
    eq("subclass type", type(a).__name__ == "NpSub", type(b) is Sub)
    eq("subclass isinstance", isinstance(a, np.ndarray), isinstance(b, _rnp.ndarray))
    eq_array("subclass contents", np.asarray(a), b)
    eq("subclass base", base_kind(a, buf_np), base_kind(b, buf_pt))
    eq("subclass view() keeps the subclass",
       type(a.view()) is NpSub, type(b.view()) is Sub)
    eq("subclass view(ndarray)",
       type(a.view(np.ndarray)) is np.ndarray,
       type(b.view(_rnp.ndarray)) is _rnp.ndarray)
    eq("subclass view base is self", a.view().base is a, b.view().base is b)
    eq("asarray(subclass).base is obj",
       np.asarray(a).base is a, _rnp.asarray(b).base is b)
    # writing through the subclass reaches the source
    a[0] = 5
    b[0] = 5
    eq("subclass writeback", bytes(buf_np), bytes(buf_pt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260816)
    ap.parse_args()

    check_allocating_constructor()
    check_constructor_grid()
    check_constructor_errors()
    check_base_chain()
    check_writeback()
    check_frombuffer()
    check_mmap()
    check_array_protocol()
    check_ndarray_subclassing()

    print(f"{CHECKS} comparisons, {len(FAILURES)} divergences")
    for name, msg in FAILURES:
        print(f"  FAIL {name}: {msg}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
