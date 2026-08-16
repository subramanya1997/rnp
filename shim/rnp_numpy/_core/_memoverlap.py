"""Memory-overlap solving — a Python port of numpy's ``mem_overlap.c``.

This backs ``np.shares_memory`` / ``np.may_share_memory`` and the
``_multiarray_tests`` entry points ``solve_diophantine`` / ``internal_overlap``.

The algorithm is Ramachandran's bounded-Diophantine depth-first Euclid search,
exactly as implemented in ``numpy/_core/src/common/mem_overlap.c``.  Asking
whether two strided arrays overlap is equivalent to asking whether::

    sum(a[i] * x[i] for i in range(n)) == b,   a[i] > 0,   0 <= x[i] <= ub[i]

has a solution.  Fixed-width overflow behaviour is reproduced faithfully
(``safe_add``/``safe_mul`` wrap and raise a flag; the 128-bit intermediates of
``npy_extint128.h`` are modelled with Python ints), so the decision results and
the ``OverflowError`` cases match numpy's.

The one thing numpy gets from C and this port has to work for: the *data
pointer* of an array view.  The Rust ``_rnp.ndarray`` exposes no ``.data`` and
no ``__array_interface__``, but it does implement the buffer protocol, and
``PyObject_GetBuffer`` (reached through ``ctypes``) hands back ``Py_buffer.buf``
— which is precisely ``PyArray_DATA``, for arbitrary strides including negative
and out-of-bounds ``as_strided`` views.
"""

import ctypes

INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1
UINT64_MASK = 2 ** 64 - 1
INT128_MAX = 2 ** 128 - 1

# mem_overlap_t
MEM_OVERLAP_NO = 0
MEM_OVERLAP_YES = 1
MEM_OVERLAP_TOO_HARD = -1
MEM_OVERLAP_OVERFLOW = -2
MEM_OVERLAP_ERROR = -3

MAY_SHARE_BOUNDS = 0
MAY_SHARE_EXACT = -1

SSIZE_MIN = INT64_MIN
SSIZE_MAX = INT64_MAX


# ---------------------------------------------------------------------------
# Fixed-width integer helpers (npy_extint128.h)
# ---------------------------------------------------------------------------

def _wrap64(v):
    """Two's-complement wrap to int64, matching __builtin_*_overflow results."""
    return ((v + 2 ** 63) & UINT64_MASK) - 2 ** 63


def safe_add(a, b, ov):
    r = a + b
    if r < INT64_MIN or r > INT64_MAX:
        ov[0] = True
        return _wrap64(r)
    return r


def safe_sub(a, b, ov):
    r = a - b
    if r < INT64_MIN or r > INT64_MAX:
        ov[0] = True
        return _wrap64(r)
    return r


def safe_mul(a, b, ov):
    r = a * b
    if r < INT64_MIN or r > INT64_MAX:
        ov[0] = True
        return _wrap64(r)
    return r


def _to_64(x, ov):
    """``to_64`` of npy_extint128.h: sign-magnitude 128-bit -> int64."""
    if x < INT64_MIN or x > INT64_MAX:
        ov[0] = True
        return _wrap64(x)
    return x


def _add_128(x, y, ov):
    r = x + y
    if r > INT128_MAX or r < -INT128_MAX:
        ov[0] = True
    return r


def _sub_128(x, y, ov):
    return _add_128(x, -y, ov)


def _floordiv_128_64(a, b):
    # b > 0; Python floor division already rounds down.
    return a // b


def _ceildiv_128_64(a, b):
    # b > 0
    return -((-a) // b)


def _truncdiv(a, b):
    """C integer division/remainder (truncating), for b > 0."""
    if a < 0:
        q = -((-a) // b)
    else:
        q = a // b
    return q, a - q * b


# ---------------------------------------------------------------------------
# Bounded Diophantine solver
# ---------------------------------------------------------------------------

def _euclid(a1, a2):
    """Solve gamma*a1 + epsilon*a2 == gcd(a1, a2); a1, a2 > 0."""
    gamma1, gamma2 = 1, 0
    epsilon1, epsilon2 = 0, 1

    while True:
        if a2 > 0:
            r = a1 // a2
            a1 -= r * a2
            gamma1 -= r * gamma2
            epsilon1 -= r * epsilon2
        else:
            return a1, gamma1, epsilon1

        if a1 > 0:
            r = a2 // a1
            a2 -= r * a1
            gamma2 -= r * gamma1
            epsilon2 -= r * epsilon1
        else:
            return a2, gamma2, epsilon2


def _diophantine_precompute(n, A, U):
    """Returns (EpA, EpU, Gamma, Epsilon) or None on integer overflow."""
    EpA = [0] * n
    EpU = [0] * n
    Gamma = [0] * n
    Epsilon = [0] * n
    ov = [False]

    a_gcd, gamma, epsilon = _euclid(A[0], A[1])
    EpA[0] = a_gcd
    Gamma[0] = gamma
    Epsilon[0] = epsilon

    if n > 2:
        c1 = A[0] // a_gcd
        c2 = A[1] // a_gcd
        EpU[0] = safe_add(safe_mul(U[0], c1, ov), safe_mul(U[1], c2, ov), ov)
        if ov[0]:
            return None

    for j in range(2, n):
        a_gcd, gamma, epsilon = _euclid(EpA[j - 2], A[j])
        EpA[j - 1] = a_gcd
        Gamma[j - 1] = gamma
        Epsilon[j - 1] = epsilon

        if j < n - 1:
            c1 = EpA[j - 2] // a_gcd
            c2 = A[j] // a_gcd
            EpU[j - 1] = safe_add(safe_mul(c1, EpU[j - 2], ov),
                                  safe_mul(c2, U[j], ov), ov)
            if ov[0]:
                return None

    return EpA, EpU, Gamma, Epsilon


def _diophantine_dfs(n, v, A, U, EpA, EpU, Gamma, Epsilon,
                     b, max_work, require_ub_nontrivial, x, count):
    if max_work >= 0 and count[0] >= max_work:
        return MEM_OVERLAP_TOO_HARD

    if v == 1:
        a1 = A[0]
        u1 = U[0]
    else:
        a1 = EpA[v - 2]
        u1 = EpU[v - 2]

    a2 = A[v]
    u2 = U[v]

    a_gcd = EpA[v - 1]
    gamma = Gamma[v - 1]
    epsilon = Epsilon[v - 1]

    c, r = _truncdiv(b, a_gcd)
    if r != 0:
        count[0] += 1
        return MEM_OVERLAP_NO

    c1 = a2 // a_gcd
    c2 = a1 // a_gcd

    # x1 = gamma*c + c1*t ; x2 = epsilon*c - c2*t ; 0 <= x1 <= u1, 0 <= x2 <= u2
    ov = [False]
    x10 = gamma * c
    x20 = epsilon * c

    t_l1 = _ceildiv_128_64(-x10, c1)
    t_l2 = _ceildiv_128_64(_sub_128(x20, u2, ov), c2)

    t_u1 = _floordiv_128_64(_sub_128(u1, x10, ov), c1)
    t_u2 = _floordiv_128_64(x20, c2)

    if ov[0]:
        return MEM_OVERLAP_OVERFLOW

    if t_l2 > t_l1:
        t_l1 = t_l2
    if t_u1 > t_u2:
        t_u1 = t_u2

    if t_l1 > t_u1:
        count[0] += 1
        return MEM_OVERLAP_NO

    t_l = _to_64(t_l1, ov)
    t_u = _to_64(t_u1, ov)

    x10 = _add_128(x10, c1 * t_l, ov)
    x20 = _sub_128(x20, c2 * t_l, ov)

    t_u = safe_sub(t_u, t_l, ov)
    t_l = 0
    x1 = _to_64(x10, ov)
    x2 = _to_64(x20, ov)

    if ov[0]:
        return MEM_OVERLAP_OVERFLOW

    if v == 1:
        # Base case
        if t_u >= t_l:
            x[0] = x1 + c1 * t_l
            x[1] = x2 - c2 * t_l
            if require_ub_nontrivial:
                for j in range(n):
                    if x[j] != U[j] // 2:
                        break
                else:
                    # Ignore the 'trivial' solution
                    count[0] += 1
                    return MEM_OVERLAP_NO
            return MEM_OVERLAP_YES
        count[0] += 1
        return MEM_OVERLAP_NO

    for t in range(t_l, t_u + 1):
        xv = x2 - c2 * t
        x[v] = xv

        ov[0] = False
        b2 = safe_sub(b, safe_mul(a2, xv, ov), ov)
        if ov[0]:
            return MEM_OVERLAP_OVERFLOW

        res = _diophantine_dfs(n, v - 1, A, U, EpA, EpU, Gamma, Epsilon,
                               b2, max_work, require_ub_nontrivial, x, count)
        if res != MEM_OVERLAP_NO:
            return res

    count[0] += 1
    return MEM_OVERLAP_NO


def solve_diophantine(n, A, U, b, max_work, require_ub_nontrivial, x):
    """Solve ``sum(A[i]*x[i]) == b``, ``0 <= x[i] <= U[i]``, ``A[i] > 0``.

    ``x`` is a caller-supplied list of length >= n that receives the solution.
    """
    for j in range(n):
        if A[j] <= 0:
            return MEM_OVERLAP_ERROR
        elif U[j] < 0:
            return MEM_OVERLAP_NO

    if require_ub_nontrivial:
        ub_sum = 0
        ov = [False]
        for j in range(n):
            if U[j] % 2 != 0:
                return MEM_OVERLAP_ERROR
            ub_sum = safe_add(ub_sum, safe_mul(A[j], U[j] // 2, ov), ov)
        if ov[0]:
            return MEM_OVERLAP_ERROR
        b = ub_sum

    if b < 0:
        return MEM_OVERLAP_NO

    if n == 0:
        if require_ub_nontrivial:
            return MEM_OVERLAP_NO
        if b == 0:
            return MEM_OVERLAP_YES
        return MEM_OVERLAP_NO
    elif n == 1:
        if require_ub_nontrivial:
            return MEM_OVERLAP_NO
        if b % A[0] == 0:
            x[0] = b // A[0]
            if 0 <= x[0] <= U[0]:
                return MEM_OVERLAP_YES
        return MEM_OVERLAP_NO
    else:
        pre = _diophantine_precompute(n, A, U)
        if pre is None:
            return MEM_OVERLAP_OVERFLOW
        EpA, EpU, Gamma, Epsilon = pre
        return _diophantine_dfs(n, n - 1, A, U, EpA, EpU, Gamma, Epsilon,
                                b, max_work, require_ub_nontrivial, x, [0])


def diophantine_simplify(A, U, b):
    """Combine identical coefficients, drop unneeded variables, trim bounds.

    ``A`` and ``U`` are mutated in place and truncated; returns ``-1`` on
    integer overflow, else ``0``.
    """
    n = len(A)
    for j in range(n):
        if U[j] < 0:
            return 0
    if b < 0:
        return 0

    ov = [False]

    # Sort vs. coefficients, descending
    order = sorted(range(n), key=lambda j: -A[j])
    A[:] = [A[j] for j in order]
    U[:] = [U[j] for j in order]

    # Combine identical coefficients
    m = n
    i = 0
    for j in range(1, m):
        if A[i] == A[j]:
            U[i] = safe_add(U[i], U[j], ov)
            n -= 1
        else:
            i += 1
            if i != j:
                A[i] = A[j]
                U[i] = U[j]
    del A[n:]
    del U[n:]

    # Trim bounds and remove unnecessary variables
    m = n
    i = 0
    for j in range(m):
        ub = b // A[j]
        if U[j] > ub:
            U[j] = ub
        if U[j] == 0:
            n -= 1
        else:
            if i != j:
                A[i] = A[j]
                U[i] = U[j]
            i += 1
    del A[n:]
    del U[n:]

    return -1 if ov[0] else 0


# ---------------------------------------------------------------------------
# Array introspection: data pointer, shape, strides, itemsize
# ---------------------------------------------------------------------------

class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_PyObject_GetBuffer = ctypes.pythonapi.PyObject_GetBuffer
_PyObject_GetBuffer.argtypes = [ctypes.py_object,
                                ctypes.POINTER(_PyBuffer), ctypes.c_int]
_PyObject_GetBuffer.restype = ctypes.c_int

_PyBuffer_Release = ctypes.pythonapi.PyBuffer_Release
_PyBuffer_Release.argtypes = [ctypes.POINTER(_PyBuffer)]
_PyBuffer_Release.restype = None

_PyBUF_STRIDES = 0x0018  # PyBUF_ND | strides


def data_pointer(arr):
    """The equivalent of ``PyArray_DATA(arr)``, via the buffer protocol."""
    view = _PyBuffer()
    if _PyObject_GetBuffer(arr, ctypes.byref(view), _PyBUF_STRIDES) != 0:
        # ctypes does not propagate the C-level exception; drop it and report
        # the failure the way the caller expects.
        ctypes.pythonapi.PyErr_Clear()
        raise TypeError("cannot obtain a data pointer for this array")
    try:
        return view.buf or 0
    finally:
        _PyBuffer_Release(ctypes.byref(view))


class ArrayInfo:
    """The strided-memory description the solver needs."""

    __slots__ = ("ptr", "itemsize", "shape", "strides")

    def __init__(self, ptr, itemsize, shape, strides):
        self.ptr = ptr
        self.itemsize = itemsize
        self.shape = shape
        self.strides = strides

    @property
    def ndim(self):
        return len(self.shape)


def _itemsize_from_typestr(typestr):
    return int(typestr[2:] or 0)


def array_info(obj):
    """Build an :class:`ArrayInfo` for anything array-like.

    Mirrors ``PyArray_FROM_O`` in ``array_shares_memory_impl``: real arrays are
    used directly, objects exposing ``__array_interface__`` or ``__array__``
    are honoured (gh-5604), everything else goes through ``asarray``.
    """
    from .. import asarray, ndarray

    if isinstance(obj, ndarray):
        return ArrayInfo(data_pointer(obj), obj.itemsize,
                         tuple(obj.shape), tuple(obj.strides))

    interface = getattr(obj, "__array_interface__", None)
    if isinstance(interface, dict):
        shape = tuple(interface["shape"])
        typestr = interface["typestr"]
        itemsize = _itemsize_from_typestr(typestr)
        strides = interface.get("strides")
        if strides is None:
            strides = _c_contiguous_strides(shape, itemsize)
        else:
            strides = tuple(strides)
        data = interface["data"]
        if isinstance(data, tuple):
            ptr = data[0]
        else:  # buffer-exposing object
            ptr = data_pointer(data)
        return ArrayInfo(ptr, itemsize, shape, strides)

    conv = getattr(obj, "__array__", None)
    if conv is not None:
        try:
            arr = conv(dtype=None, copy=None)
        except TypeError:
            arr = conv()
        if isinstance(arr, ndarray):
            return array_info(arr)

    return array_info(asarray(obj))


def _array_interface(self):
    """``ndarray.__array_interface__`` (version 3).

    The Rust array type does not carry one; it is grafted on here because the
    overlap machinery is the thing that needs a data pointer, and because
    upstream tests hand numpy objects that merely *proxy* this attribute
    (gh-5604) or rebuild an array from it (``DummyArray``).
    """
    typestr = self.dtype.str
    strides = tuple(self.strides)
    info = ArrayInfo(0, self.itemsize, tuple(self.shape), strides)
    return {
        "data": (data_pointer(self), not self.flags.writeable),
        "strides": None if is_c_contiguous(info) else strides,
        "descr": [("", typestr)],
        "typestr": typestr,
        "shape": tuple(self.shape),
        "version": 3,
    }


def _install_array_interface():
    from _rnp import ndarray
    if "__array_interface__" not in vars(ndarray):
        ndarray.__array_interface__ = property(_array_interface)


_install_array_interface()


def _c_contiguous_strides(shape, itemsize):
    strides = [0] * len(shape)
    sd = itemsize
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = sd
        sd *= shape[i]
    return tuple(strides)


def array_from_interface(obj, dtype=None):
    """Build an array from an object exposing ``__array_interface__``.

    numpy can wrap a bare pointer; the Rust engine cannot, so the array is
    rebuilt as a strided view of the interface's *base* array — which is what
    `DummyArray` (and every in-tree user of it) supplies.
    """
    from .. import asarray, ndarray
    from ..lib._stride_tricks_impl import as_strided

    interface = obj.__array_interface__
    if not isinstance(interface, dict):
        raise TypeError("__array_interface__ must be a dict")

    shape = tuple(interface["shape"])
    typestr = interface["typestr"]
    itemsize = _itemsize_from_typestr(typestr)
    strides = interface.get("strides")
    strides = (_c_contiguous_strides(shape, itemsize) if strides is None
               else tuple(strides))

    data = interface["data"]
    ptr = data[0] if isinstance(data, tuple) else data_pointer(data)

    base = getattr(obj, "base", None)
    if not isinstance(base, ndarray):
        raise TypeError(
            "cannot build an array from an __array_interface__ without a "
            "base array")

    root = base
    while getattr(root, "base", None) is not None:
        root = root.base

    offset = ptr - data_pointer(root)
    if offset < 0 or offset % itemsize:
        raise TypeError("__array_interface__ data pointer is not addressable")

    from .. import dtype as _dtype
    flat = root.ravel().view(_dtype(typestr))
    out = as_strided(flat[offset // itemsize:], shape, strides)
    if dtype is not None and _dtype(dtype) != out.dtype:
        out = asarray(out).astype(_dtype(dtype))
    return out


def is_c_contiguous(info):
    """``PyArray_ISCONTIGUOUS`` — computed, not read off a flags object."""
    sd = info.itemsize
    shape = info.shape
    strides = info.strides
    for i in range(len(shape) - 1, -1, -1):
        dim = shape[i]
        if dim == 0:
            return True
        if dim != 1:
            if strides[i] != sd:
                return False
            sd *= dim
    return True


def offset_bounds_from_strides(itemsize, shape, strides):
    """Half-open range [lower, upper) of offsets from the data pointer."""
    lower = 0
    upper = 0
    for dim, stride in zip(shape, strides):
        if dim == 0:
            return 0, 0
        max_axis_offset = stride * (dim - 1)
        if max_axis_offset > 0:
            upper += max_axis_offset
        else:
            lower += max_axis_offset
    return lower, upper + itemsize


def _memory_extents(info):
    low, upper = offset_bounds_from_strides(info.itemsize, info.shape,
                                            info.strides)
    return info.ptr + low, info.ptr + upper


def _strides_to_terms(info, A, U, skip_empty):
    for i in range(len(info.shape)):
        dim = info.shape[i]
        stride = info.strides[i]
        if skip_empty and (dim <= 1 or stride == 0):
            continue
        a = -stride if stride < 0 else stride
        if a < 0:
            return True  # integer overflow (stride == INT64_MIN)
        A.append(a)
        U.append(dim - 1)
    return False


# ---------------------------------------------------------------------------
# The two public solvers
# ---------------------------------------------------------------------------

def solve_may_share_memory(a, b, max_work):
    """``a``, ``b``: :class:`ArrayInfo`.  Returns a ``mem_overlap_t``."""
    start1, end1 = _memory_extents(a)
    start2, end2 = _memory_extents(b)

    if not (start1 < end2 and start2 < end1 and start1 < end1 and start2 < end2):
        return MEM_OVERLAP_NO

    if max_work == 0:
        return MEM_OVERLAP_TOO_HARD

    rhs = min(end2 - 1 - start1, end1 - 1 - start2)
    if rhs > INT64_MAX:
        return MEM_OVERLAP_OVERFLOW

    A = []
    U = []
    if _strides_to_terms(a, A, U, True):
        return MEM_OVERLAP_OVERFLOW
    if _strides_to_terms(b, A, U, True):
        return MEM_OVERLAP_OVERFLOW
    if a.itemsize > 1:
        A.append(1)
        U.append(a.itemsize - 1)
    if b.itemsize > 1:
        A.append(1)
        U.append(b.itemsize - 1)

    if diophantine_simplify(A, U, rhs):
        return MEM_OVERLAP_OVERFLOW

    n = len(A)
    return solve_diophantine(n, A, U, rhs, max_work, 0, [0] * (n + 2))


def solve_may_have_internal_overlap(info, max_work):
    if is_c_contiguous(info):
        return MEM_OVERLAP_NO

    A = []
    U = []
    if _strides_to_terms(info, A, U, False):
        return MEM_OVERLAP_OVERFLOW
    if info.itemsize > 1:
        A.append(1)
        U.append(info.itemsize - 1)

    # Get rid of zero coefficients and empty terms
    A2 = []
    U2 = []
    for a, ub in zip(A, U):
        if ub == 0:
            continue
        elif ub < 0:
            return MEM_OVERLAP_NO
        elif a == 0:
            return MEM_OVERLAP_YES
        A2.append(a)
        U2.append(ub * 2)  # double bounds -> internal overlap problem

    # Sort vs. coefficients (descending); diophantine_simplify must not be
    # used here, as it would change the inequality part of the problem.
    order = sorted(range(len(A2)), key=lambda j: -A2[j])
    A2 = [A2[j] for j in order]
    U2 = [U2[j] for j in order]

    n = len(A2)
    return solve_diophantine(n, A2, U2, -1, max_work, 1, [0] * (n + 2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _coerce_max_work(max_work, default):
    if max_work is None:
        return default
    if not isinstance(max_work, int):
        raise ValueError("max_work must be an integer")
    max_work = int(max_work)
    if max_work < SSIZE_MIN or max_work > SSIZE_MAX:
        raise OverflowError("Python int too large to convert to C ssize_t")
    if max_work < -2:
        raise ValueError("Invalid value for max_work")
    return max_work


def _shares_memory_impl(a, b, max_work, default_max_work, raise_exceptions):
    max_work = _coerce_max_work(max_work, default_max_work)

    result = solve_may_share_memory(array_info(a), array_info(b), max_work)

    if result == MEM_OVERLAP_NO:
        return False
    elif result == MEM_OVERLAP_YES:
        return True
    elif result == MEM_OVERLAP_OVERFLOW:
        if raise_exceptions:
            raise OverflowError("Integer overflow in computing overlap")
        return True
    elif result == MEM_OVERLAP_TOO_HARD:
        if raise_exceptions:
            from ..exceptions import TooHardError
            raise TooHardError("Exceeded max_work")
        return True
    raise RuntimeError("Error in computing overlap")


def shares_memory(a, b, /, max_work=None):
    """Determine if two arrays share memory."""
    return _shares_memory_impl(a, b, max_work, MAY_SHARE_EXACT, True)


def may_share_memory(a, b, /, max_work=None):
    """Determine if two arrays might share memory."""
    return _shares_memory_impl(a, b, max_work, MAY_SHARE_BOUNDS, False)


# ---- _multiarray_tests entry points ---------------------------------------

def tests_solve_diophantine(A, U, b, max_work=-1, simplify=0,
                            require_ub_nontrivial=0):
    if not isinstance(A, tuple) or not isinstance(U, tuple):
        raise TypeError("argument must be a tuple")

    if len(U) != len(A):
        raise ValueError("A, U must be tuples of equal length")

    b = _as_ssize_t(b)
    max_work = _as_ssize_t(max_work)

    A = [_as_ssize_t(v) for v in A]
    U = [_as_ssize_t(v) for v in U]

    result = MEM_OVERLAP_YES
    if simplify and not require_ub_nontrivial:
        if diophantine_simplify(A, U, b):
            result = MEM_OVERLAP_OVERFLOW

    nterms = len(A)
    x = [0] * (nterms + 2)
    if result == MEM_OVERLAP_YES:
        result = solve_diophantine(nterms, A, U, b, max_work,
                                   require_ub_nontrivial, x)

    if result == MEM_OVERLAP_YES:
        return tuple(x[:nterms])
    elif result == MEM_OVERLAP_NO:
        return None
    elif result == MEM_OVERLAP_ERROR:
        raise ValueError("Invalid arguments")
    elif result == MEM_OVERLAP_OVERFLOW:
        raise OverflowError("Integer overflow")
    elif result == MEM_OVERLAP_TOO_HARD:
        raise RuntimeError("Too much work done")
    raise RuntimeError("Unknown error")


def tests_internal_overlap(self, max_work=MAY_SHARE_EXACT):
    max_work = _as_ssize_t(max_work)
    if max_work < -2:
        raise ValueError("Invalid value for max_work")

    result = solve_may_have_internal_overlap(array_info(self), max_work)

    if result == MEM_OVERLAP_NO:
        return False
    elif result == MEM_OVERLAP_YES:
        return True
    elif result == MEM_OVERLAP_OVERFLOW:
        raise OverflowError("Integer overflow in computing overlap")
    elif result == MEM_OVERLAP_TOO_HARD:
        raise ValueError("Exceeded max_work")
    raise RuntimeError("Error in computing overlap")


def _as_ssize_t(v):
    v = int(v)
    if v < SSIZE_MIN or v > SSIZE_MAX:
        raise OverflowError("Python int too large to convert to C ssize_t")
    return v
