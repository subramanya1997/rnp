"""Stand-in for numpy._core._multiarray_tests (a C test-support extension).

The real module exposes internals the port does not have. Every entry point
raises NotImplementedError so that test *collection* succeeds and only the
tests that actually need these helpers fail.
"""


def _missing(name):
    def fn(*args, **kwargs):
        raise NotImplementedError(
            f"_multiarray_tests.{name} is not implemented by rnp")
    fn.__name__ = name
    return fn


_NAMES = [
    "create_custom_field_dtype", "get_fpu_mode", "getset_numericops",
    "format_float_OSprintf_g", "get_buffer_info", "get_c_wrapping_array",
    "internal_overlap", "solve_diophantine", "array_indexing",
    "test_nditer_too_large", "npy_char_deprecation", "run_byteorder_converter",
    "run_casting_converter", "run_clipmode_converter",
    "run_intp_converter", "run_scalar_intp_converter", "run_selectkind_converter",
    "run_sortkind_converter", "run_searchside_converter", "identityhash_tester",
    "incref_elide", "incref_elide_l", "npy_ensurenocopy", "get_struct_alignments",
    "get_all_cast_information", "corrupt_or_fix_bufferinfo",
]

for _n in _NAMES:
    globals()[_n] = _missing(_n)


def get_buffer_info(obj, flags):
    """Request a buffer and return its ``(shape, strides)`` pair.

    This is the observable part of NumPy's C test helper.  A writable request
    against a scalar has NumPy's scalar-specific error text.
    """
    from .. import ndarray
    known = {
        "SIMPLE", "WRITABLE", "STRIDES", "ND", "C_CONTIGUOUS",
        "F_CONTIGUOUS", "ANY_CONTIGUOUS", "INDIRECT", "FORMAT",
        "STRIDED", "STRIDED_RO", "RECORDS", "RECORDS_RO", "FULL",
        "FULL_RO", "CONTIG", "CONTIG_RO",
    }
    if any(flag not in known for flag in flags):
        raise ValueError("invalid flag used.")
    if "WRITABLE" in flags and not isinstance(obj, ndarray):
        raise BufferError("scalar buffer is readonly")
    view = memoryview(obj)
    if "WRITABLE" in flags and view.readonly:
        raise BufferError("Object is not writable")
    return view.shape, view.strides


# ---------------------------------------------------------------------------
# npy_argparse — the C fast-call argument parser, exercised by test_argparse.py
# ---------------------------------------------------------------------------
#
# `npy_parse_arguments` (numpy/_core/src/common/npy_argparse.c) is NumPy's own
# METH_FASTCALL argument parser. Argument specs are strings: a bare name is a
# *required* positional-or-keyword parameter, `|name` an optional one, `$name`
# a keyword-only one, and the empty name marks a positional-only slot. The
# order in which it reports problems is observable (converters run *before*
# the missing-argument sweep, for instance), so the checks below follow the C
# control flow rather than Python's own calling convention.

_INT_MIN, _INT_MAX = -(2 ** 31), 2 ** 31 - 1
_LONG_MIN, _LONG_MAX = -(2 ** 63), 2 ** 63 - 1


def _python_int_from_int(obj):
    """`PyArray_PythonPyIntFromInt` — CPython's ``"i"`` conversion."""
    # "Pythons behaviour is to check only for float explicitly..."
    if isinstance(obj, float):
        raise TypeError("integer argument expected, got float")
    # PyLong_AsLong: exact ints pass through, everything else goes via
    # __index__ (which raises the "cannot be interpreted as an integer"
    # TypeError for objects that have none).
    if isinstance(obj, int):
        result = int(obj)
    else:
        index = getattr(type(obj), "__index__", None)
        if index is None:
            raise TypeError(
                f"'{type(obj).__name__}' object cannot be interpreted "
                f"as an integer")
        result = index(obj)
    if not (_LONG_MIN <= result <= _LONG_MAX):
        raise OverflowError("Python int too large to convert to C long")
    if not (_INT_MIN <= result <= _INT_MAX):
        raise OverflowError("Python int too large to convert to C int")
    return result


class _ArgParser:
    """Python transcription of `_npy_parse_arguments`."""

    _UNSET = object()

    def __init__(self, funcname, specs):
        self.funcname = funcname
        self.converters = [conv for _, conv in specs]
        self.nargs = len(specs)
        self.nrequired = 0
        self.npositional = 0
        self.npositional_only = 0
        names = []
        for name, _conv in specs:
            if name[:1] == "|":
                name = name[1:]
                self.npositional += 1
            elif name[:1] == "$":
                name = name[1:]
            else:
                self.nrequired += 1
                self.npositional += 1
            if name == "":
                self.npositional_only += 1
            names.append(name)
        # Keyword slots follow the positional-only ones; duplicate names are
        # resolved to the *first* matching slot, as the C linear scan does.
        self.kw_strings = names[self.npositional_only:]

    def _raise_too_many_positional(self, len_args):
        verb = "was" if len_args == 1 else "were"
        if self.npositional == self.nrequired:
            raise TypeError(
                f"{self.funcname}() takes {self.npositional} positional "
                f"arguments but {len_args} {verb} given")
        raise TypeError(
            f"{self.funcname}() takes from {self.nrequired} to "
            f"{self.npositional} positional arguments but "
            f"{len_args} {verb} given")

    def _raise_missing(self, i):
        if i < self.npositional_only:
            raise TypeError(
                f"{self.funcname}() missing required positional argument {i}")
        kw = self.kw_strings[i - self.npositional_only]
        raise TypeError(
            f"{self.funcname}() missing required argument '{kw}' (pos {i})")

    def parse(self, args, kwargs):
        unset = self._UNSET
        len_args = len(args)
        if len_args > self.npositional:
            self._raise_too_many_positional(len_args)

        all_arguments = [unset] * max(self.nargs, len_args)
        all_arguments[:len_args] = args
        max_nargs = len_args

        if kwargs:
            max_nargs = self.nargs
            for key, value in kwargs.items():
                for idx, name in enumerate(self.kw_strings):
                    if name == key:
                        break
                else:
                    raise TypeError(
                        f"{self.funcname}() got an unexpected keyword "
                        f"argument '{key}'")
                param_pos = idx + self.npositional_only
                if all_arguments[param_pos] is not unset:
                    raise TypeError(
                        f"argument for {self.funcname}() given by name "
                        f"('{key}') and position (position {param_pos})")
                all_arguments[param_pos] = value

        out = [unset] * self.nargs
        for i in range(max_nargs):
            if all_arguments[i] is unset:
                continue
            converter = self.converters[i]
            out[i] = (all_arguments[i] if converter is None
                      else converter(all_arguments[i]))

        if len_args < self.nrequired:
            if max_nargs < self.nrequired:
                self._raise_missing(max_nargs)
            for i in range(self.nrequired):
                if all_arguments[i] is unset:
                    self._raise_missing(i)
        return out


# def func(arg1, /, arg2, *, arg3):  (see _multiarray_tests.c.src)
_ARGPARSE_EXAMPLE = _ArgParser("func", [
    ("", _python_int_from_int),
    ("arg2", None),
    ("|arg3", None),
    ("$arg3", None),
])

_THREADED_ARGPARSE_EXAMPLE = _ArgParser("thread_func", [
    ("$arg1", _python_int_from_int),
    ("$arg2", None),
])


def argparse_example_function(*args, **kwargs):
    _ARGPARSE_EXAMPLE.parse(args, kwargs)
    return None


def threaded_argparse_example_function(*args, **kwargs):
    _THREADED_ARGPARSE_EXAMPLE.parse(args, kwargs)
    return None


# ---------------------------------------------------------------------------
# extint128 — the checked 128-bit helpers exercised by test_extint128.py
# ---------------------------------------------------------------------------
#
# numpy's `npy_extint128` is *sign-magnitude*, not two's complement: the sign
# bit is stored separately from a 128-bit magnitude, so the representable range
# is symmetric, [-(2**128 - 1), 2**128 - 1], and "negative zero" exists. Python
# ints model the values directly; what has to be reproduced is the range
# checking (OverflowError) and the truncate-toward-zero division, which is what
# a sign-magnitude representation gives you for free and which differs from
# Python's floor division on negative operands.

_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1
_INT128_MAX = 2 ** 128 - 1
_INT128_MIN = -_INT128_MAX


def _check_128(v):
    if not (_INT128_MIN <= v <= _INT128_MAX):
        raise OverflowError("overflow in 128-bit integer")
    return v


def extint_to_128(a):
    return int(a)


def extint_to_64(a):
    a = int(a)
    if not (_INT64_MIN <= a <= _INT64_MAX):
        raise OverflowError("cannot convert to 64-bit integer")
    return a


def extint_mul_64_64(a, b):
    # A 64x64 product always fits the 128-bit magnitude; no check needed.
    return int(a) * int(b)


def extint_add_128(a, b):
    return _check_128(int(a) + int(b))


def extint_sub_128(a, b):
    return _check_128(int(a) - int(b))


def extint_neg_128(a):
    return -int(a)


def extint_shl_128(a):
    # Shifting the *magnitude* left by one, truncated to 128 bits, and
    # reattaching the sign.
    a = int(a)
    if a < 0:
        return -(((-a) << 1) & _INT128_MAX)
    return (a << 1) & _INT128_MAX


def extint_shr_128(a):
    a = int(a)
    if a < 0:
        return -((-a) >> 1)
    return a >> 1


def extint_gt_128(a, b):
    return int(a) > int(b)


def extint_divmod_128_64(a, b):
    # Truncating division: the quotient rounds toward zero and the remainder
    # carries the sign of the dividend (b is always positive here).
    a, b = int(a), int(b)
    if a >= 0:
        q, r = divmod(a, b)
    else:
        q, r = divmod(-a, b)
        q, r = -q, -r
    return q, r


def extint_floordiv_128_64(a, b):
    return int(a) // int(b)


def extint_ceildiv_128_64(a, b):
    a, b = int(a), int(b)
    return -((-a) // b)


def extint_safe_binop(a, b, op):
    a, b = int(a), int(b)
    if op == 1:
        c = a + b
    elif op == 2:
        c = a - b
    elif op == 3:
        c = a * b
    else:
        raise ValueError(f"invalid op {op}")
    if not (_INT64_MIN <= c <= _INT64_MAX):
        raise OverflowError("overflow in safe binop")
    return c


# ---------------------------------------------------------------------------
# The identity hash table exercised by test_hashtable.py
# ---------------------------------------------------------------------------
#
# numpy's `PyArrayIdentityHash` keys on object *identity*, not equality, and
# `set_item_default` has set-default semantics: the first writer for a key wins
# and every later writer gets that first value back. The tests hammer it from
# eight threads and assert exactly that, so the whole thing sits behind one
# lock. Key objects are stored alongside the value: keying on `id()` alone
# would let a collected key's address be reused by an unrelated object.

import threading as _threading


class _IdentityHash:
    __slots__ = ("key_length", "_lock", "_table")

    def __init__(self, key_length):
        self.key_length = key_length
        self._lock = _threading.Lock()
        self._table = {}

    @staticmethod
    def _k(key):
        return tuple(id(o) for o in key)

    def set_item_default(self, key, value):
        k = self._k(key)
        with self._lock:
            entry = self._table.get(k)
            if entry is not None:
                return entry[1]
            # `key` is kept alive by the entry so its ids stay unique.
            self._table[k] = (tuple(key), value)
            return value

    def get_item(self, key):
        with self._lock:
            entry = self._table.get(self._k(key))
        return None if entry is None else entry[1]


def create_identity_hash(key_length):
    return _IdentityHash(key_length)


def identity_hash_set_item_default(ht, key, value):
    return ht.set_item_default(key, value)


def identity_hash_get_item(ht, key):
    return ht.get_item(key)


# ---------------------------------------------------------------------------
# Memory-overlap solving (mem_overlap.c) — real implementations.
# ---------------------------------------------------------------------------

from ._memoverlap import tests_internal_overlap as internal_overlap  # noqa: E402
from ._memoverlap import tests_solve_diophantine as solve_diophantine  # noqa: E402

internal_overlap.__name__ = "internal_overlap"
solve_diophantine.__name__ = "solve_diophantine"


def __getattr__(name):
    return _missing(name)
