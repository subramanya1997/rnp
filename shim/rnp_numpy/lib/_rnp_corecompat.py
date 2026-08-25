"""Pure-Python implementations of the `numpy._core` surface `numpy.lib` and
`numpy.polynomial` are written against.

Why this module exists
----------------------

The ported upstream modules under `rnp_numpy/lib/` and `rnp_numpy/polynomial/`
are numpy's own pure-Python source.  They stand on a layer of `numpy._core`
functions (`linspace`, `clip`, `cumsum`, `dot`, `partition`, ...) that in real
numpy live in C or in `numpy/_core/*.py`.  The port's `_core` package is owned
elsewhere, so the functions are built here, on top of the primitives the Rust
engine does expose: ndarray methods (`sum`/`prod`/`sort`/`argsort`/`take`/
`reshape`/`transpose`/...), the ufunc objects (including `reduce`,
`accumulate` and `outer`), and `_core.shape_base.concatenate`.

These are correctness-first implementations, not performance work: they are
here so the `lib`/`polynomial` suites have a floor to stand on.  Where the
engine grows a native equivalent, the wiring in `rnp_numpy/__init__.py` will
prefer it — that wiring only fills names still bound to a not-implemented
stub, so nothing here shadows a real implementation.
"""
import builtins
import operator

import rnp_numpy as np
from rnp_numpy._core.shape_base import concatenate, stack, atleast_1d

from ._rnp_compat import normalize_axis_index, normalize_axis_tuple


# ---------------------------------------------------------------------------
# Sentinels and predicates
# ---------------------------------------------------------------------------

class _NoValueType:
    """`numpy._globals._NoValue`: "argument not given" for kwargs whose
    absence is distinguishable from every value they accept."""

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __repr__(self):
        return "<no value>"


_NoValue = _NoValueType()


def iterable(y):
    try:
        iter(y)
    except TypeError:
        return False
    return True


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def asanyarray(a, dtype=None, order=None, *, device=None, copy=None, like=None):
    # The port has no ndarray subclasses that survive a conversion, so
    # `asanyarray` and `asarray` coincide except that an ndarray input is
    # passed through untouched.
    if isinstance(a, np.ndarray) and dtype is None and order in (None, 'K'):
        return a
    return np.asarray(a, dtype, order=order)


def ascontiguousarray(a, dtype=None, *, like=None):
    arr = np.asarray(a, dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = arr.copy()
    return arr


def asfortranarray(a, dtype=None, *, like=None):
    arr = np.asarray(a, dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if not arr.flags["F_CONTIGUOUS"]:
        if arr.ndim <= 1:
            arr = arr.copy()
        else:
            axes = tuple(builtins.range(arr.ndim - 1, -1, -1))
            arr = arr.transpose(axes).copy().transpose(axes)
    return arr


def require(a, dtype=None, requirements=None, *, like=None):
    possible = {
        'C': 'C', 'C_CONTIGUOUS': 'C', 'CONTIGUOUS': 'C',
        'F': 'F', 'F_CONTIGUOUS': 'F', 'FORTRAN': 'F',
        'A': 'A', 'ALIGNED': 'A',
        'W': 'W', 'WRITEABLE': 'W',
        'O': 'O', 'OWNDATA': 'O',
        'E': 'E', 'ENSUREARRAY': 'E',
    }
    if not requirements:
        return asanyarray(a, dtype=dtype)
    requirements = {possible[x.upper()] for x in requirements}
    if 'E' in requirements:
        requirements.discard('E')
        arr = np.asarray(a, dtype)
    else:
        arr = asanyarray(a, dtype=dtype)
    if 'C' in requirements and 'F' in requirements:
        raise ValueError('Cannot specify both "C" and "F" order')
    if 'F' in requirements:
        arr = asfortranarray(arr)
    elif 'C' in requirements:
        arr = ascontiguousarray(arr)
    if 'O' in requirements and not arr.flags["OWNDATA"]:
        arr = arr.copy()
    return arr


# ---------------------------------------------------------------------------
# Range constructors
# ---------------------------------------------------------------------------

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0, *, device=None):
    num = operator.index(num)
    if num < 0:
        raise ValueError(f"Number of samples, {num}, must be non-negative.")
    div = (num - 1) if endpoint else num

    start = np.asarray(start) * 1.0
    stop = np.asarray(stop) * 1.0
    dt = np.result_type(start, stop, float(num))
    integer_dtype = dtype is not None and np.dtype(dtype).kind in "iub"
    if dtype is None:
        dtype = dt

    delta = stop - start
    y = np.arange(0, num, dtype=dt).reshape((-1,) + (1,) * np.ndim(delta))
    if div > 0:
        step = delta / div
        any_step_zero = not builtins.all(_flat(step != 0))
        if any_step_zero:
            y = y / div
            y = y * delta
        else:
            y = y * step
    else:
        step = np.asarray(np.nan)
        y = y * delta

    y = y + start

    if endpoint and num > 1:
        y[-1, ...] = stop

    if axis != 0:
        y = np.moveaxis(y, 0, axis)

    if integer_dtype:
        y = _floor(y)

    y = y.astype(dtype, copy=False)
    if retstep:
        return y, (step if np.ndim(step) else step[()])
    return y


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None,
             axis=0):
    if builtins.any(np.ndim(x) for x in (start, stop, base)):
        start, stop, base = np.broadcast_arrays(
            np.asarray(start), np.asarray(stop), np.asarray(base))
        base = np.expand_dims(base, axis=0) if np.ndim(base) else base
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    result = np.power(base, y)
    if dtype is None:
        return result
    return result.astype(np.dtype(dtype), copy=False)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    start = np.asarray(start) * 1.0
    stop = np.asarray(stop) * 1.0
    if builtins.any(_flat(start == 0)) or builtins.any(_flat(stop == 0)):
        raise ValueError('Geometric sequence cannot include zero')
    dt = np.result_type(start, stop, float(num))
    if dtype is None:
        dtype = dt
    out_sign = np.ones(np.broadcast_shapes(start.shape, stop.shape), dt)
    both_neg = _flat_bool(np.logical_and(start < 0, stop < 0))
    start, stop = np.broadcast_arrays(start, stop)
    start = start.astype(dt).copy()
    stop = stop.astype(dt).copy()
    if builtins.any(both_neg):
        neg = np.logical_and(start < 0, stop < 0)
        start = np.where(neg, -start, start)
        stop = np.where(neg, -stop, stop)
        out_sign = np.where(neg, -out_sign, out_sign)
    log_start = np.log10(start)
    log_stop = np.log10(stop)
    result = logspace(log_start, log_stop, num=num, endpoint=endpoint,
                      base=10.0, dtype=dt)
    if num > 0:
        result[0] = start
        if num > 1 and endpoint:
            result[-1] = stop
    result = out_sign * result
    if axis != 0:
        result = np.moveaxis(result, 0, axis)
    return result.astype(np.dtype(dtype), copy=False)


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------

def _flat(a):
    """Iterate the elements of an array-or-scalar as Python objects."""
    a = np.asarray(a)
    return iter(a.reshape(-1).tolist())


def _flat_bool(a):
    return list(_flat(a))


def _floor(a):
    return np.floor(a)


# ---------------------------------------------------------------------------
# Reductions the engine does not expose as ndarray methods
# ---------------------------------------------------------------------------

def count_nonzero(a, axis=None, *, keepdims=False):
    a = np.asanyarray(a)
    if a.dtype.kind == "b":
        nz = a
    else:
        nz = np.not_equal(a, np.zeros((), a.dtype)[()])
    return np.add.reduce(nz.astype(np.intp), axis=axis, keepdims=keepdims)


def ptp(a, axis=None, out=None, keepdims=_NoValue):
    a = np.asanyarray(a)
    kw = {} if keepdims is _NoValue else {"keepdims": keepdims}
    return np.subtract(a.max(axis=axis, **kw), a.min(axis=axis, **kw))


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=_NoValue, *,
        where=_NoValue, mean=_NoValue, correction=_NoValue):
    if correction is not _NoValue:
        ddof = correction
    a = np.asanyarray(a)
    if a.dtype.kind in "biu":
        a = a.astype(np.float64)
    if dtype is not None:
        a = a.astype(dtype)
    kd = False if keepdims is _NoValue else keepdims
    n = _reduce_count(a, axis)
    m = a.mean(axis=axis, keepdims=True)
    dev = np.subtract(a, m)
    if a.dtype.kind == "c":
        sq = np.multiply(dev, np.conjugate(dev)).real
    else:
        sq = np.multiply(dev, dev)
    total = np.add.reduce(sq, axis=axis, keepdims=kd)
    denom = n - ddof
    res = np.divide(total, denom if denom > 0 else np.nan)
    if denom <= 0:
        import warnings
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning,
                      stacklevel=2)
    return res


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=_NoValue, *,
        where=_NoValue, mean=_NoValue, correction=_NoValue):
    return np.sqrt(var(a, axis=axis, dtype=dtype, ddof=ddof,
                       keepdims=keepdims, correction=correction))


def _reduce_count(a, axis):
    if axis is None:
        return a.size
    axes = normalize_axis_tuple(axis, a.ndim)
    n = 1
    for ax in axes:
        n *= a.shape[ax]
    return n


def cumsum(a, axis=None, dtype=None, out=None):
    return _accumulate(np.add, a, axis, dtype)


def cumprod(a, axis=None, dtype=None, out=None):
    return _accumulate(np.multiply, a, axis, dtype)


def _accumulate(uf, a, axis, dtype):
    a = np.asanyarray(a)
    if axis is None:
        a = a.reshape(-1)
        axis = 0
    if dtype is not None:
        a = a.astype(dtype)
    elif a.dtype.kind == "b":
        a = a.astype(np.int_)
    return uf.accumulate(a, axis=axis)


# ---------------------------------------------------------------------------
# Element-wise shaping helpers
# ---------------------------------------------------------------------------

def clip(a, a_min=_NoValue, a_max=_NoValue, out=None, **kwargs):
    if a_min is _NoValue:
        a_min = None
    if a_max is _NoValue:
        a_max = None
    if a_min is None and a_max is None:
        raise ValueError("One of max or min must be given")
    res = np.asanyarray(a)
    if a_min is not None:
        res = np.maximum(res, a_min)
    if a_max is not None:
        res = np.minimum(res, a_max)
    if out is not None:
        out[...] = res
        return out
    return res


def round(a, decimals=0, out=None):
    a = np.asanyarray(a)
    if a.dtype.kind in "biu":
        if decimals >= 0:
            return a.copy() if out is None else _to_out(a, out)
        scale = 10.0 ** (-decimals)
        res = (np.rint(np.divide(a, scale)) * scale).astype(a.dtype)
        return res if out is None else _to_out(res, out)
    if decimals == 0:
        res = np.rint(a)
    else:
        scale = 10.0 ** decimals
        res = np.divide(np.rint(np.multiply(a, scale)), scale)
    res = res.astype(a.dtype, copy=False)
    return res if out is None else _to_out(res, out)


around = round


def _to_out(res, out):
    out[...] = res
    return out


def flip(m, axis=None):
    m = np.asanyarray(m)
    if axis is None:
        indexer = (slice(None, None, -1),) * m.ndim
    else:
        axes = normalize_axis_tuple(axis, m.ndim)
        indexer = tuple(slice(None, None, -1) if ax in axes else slice(None)
                        for ax in builtins.range(m.ndim))
    return m[indexer]


def fliplr(m):
    m = np.asanyarray(m)
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return m[:, ::-1]


def flipud(m):
    m = np.asanyarray(m)
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return m[::-1, ...]


def roll(a, shift, axis=None):
    a = np.asanyarray(a)
    if axis is None:
        return roll(a.reshape(-1), shift, 0).reshape(a.shape)
    axes = normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
    broadcasted = np.broadcast_arrays(np.asarray(shift), np.asarray(axes))
    shifts_arr, axes_arr = broadcasted[0], broadcasted[1]
    if shifts_arr.ndim > 1:
        raise ValueError("'shift' and 'axis' should be scalars or 1D "
                         "sequences")
    shifts = {ax: 0 for ax in builtins.range(a.ndim)}
    for sh, ax in zip(_flat(shifts_arr), _flat(axes_arr)):
        shifts[int(ax)] += int(sh)

    result = a
    for ax, sh in shifts.items():
        n = a.shape[ax]
        if n == 0:
            continue
        sh %= n
        if sh == 0:
            continue
        idx_lo = tuple(slice(None) if i != ax else slice(n - sh, None)
                       for i in builtins.range(a.ndim))
        idx_hi = tuple(slice(None) if i != ax else slice(0, n - sh)
                       for i in builtins.range(a.ndim))
        result = concatenate((result[idx_lo], result[idx_hi]), axis=ax)
    if result is a:
        result = a.copy()
    return result


def rot90(m, k=1, axes=(0, 1)):
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    m = np.asanyarray(m)
    if axes[0] == axes[1] or builtins.abs(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")
    if (axes[0] >= m.ndim or axes[0] < -m.ndim
            or axes[1] >= m.ndim or axes[1] < -m.ndim):
        raise ValueError(
            f"Axes={axes} out of range for array of ndim={m.ndim}.")
    k %= 4
    if k == 0:
        return m[:]
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])
    axes_list = list(builtins.range(m.ndim))
    axes_list[axes[0]], axes_list[axes[1]] = (axes_list[axes[1]],
                                              axes_list[axes[0]])
    if k == 1:
        return np.transpose(flip(m, axes[1]), axes_list)
    return flip(np.transpose(m, axes_list), axes[1])


def rollaxis(a, axis, start=0):
    n = a.ndim
    axis = normalize_axis_index(axis, n)
    if start < 0:
        start += n
    if not (0 <= start < n + 1):
        raise np.exceptions.AxisError(
            f"start arg requires -{n} <= start <= {n}, but {start} was passed "
            f"in")
    if axis < start:
        start -= 1
    if axis == start:
        return a[...]
    axes = list(builtins.range(0, n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.transpose(tuple(axes))


def expand_dims(a, axis):
    a = np.asanyarray(a)
    if type(axis) not in (tuple, list):
        axis = (axis,)
    out_ndim = len(axis) + a.ndim
    axis = normalize_axis_tuple(axis, out_ndim)
    shape_it = iter(a.shape)
    shape = tuple(1 if ax in axis else next(shape_it)
                  for ax in builtins.range(out_ndim))
    return a.reshape(shape)


def array_equiv(a1, a2):
    try:
        a1, a2 = np.asarray(a1), np.asarray(a2)
    except Exception:
        return False
    try:
        np.broadcast_shapes(a1.shape, a2.shape)
    except ValueError:
        return False
    return builtins.bool(np.all(np.equal(a1, a2)))


# ---------------------------------------------------------------------------
# Diagonals and linear algebra kernels
# ---------------------------------------------------------------------------

def diagonal(a, offset=0, axis1=0, axis2=1):
    a = np.asanyarray(a)
    if a.ndim < 2:
        raise ValueError("diag requires an array of at least two dimensions")
    axis1 = normalize_axis_index(axis1, a.ndim)
    axis2 = normalize_axis_index(axis2, a.ndim)
    if axis1 == axis2:
        raise ValueError("axis1 and axis2 cannot be the same")
    a = np.moveaxis(a, (axis1, axis2), (-2, -1))
    n, m = a.shape[-2], a.shape[-1]
    if offset >= 0:
        length = builtins.max(builtins.min(n, m - offset), 0)
        cropped = a[..., :length, offset:offset + length]
    else:
        length = builtins.max(builtins.min(n + offset, m), 0)
        cropped = a[..., -offset:-offset + length, :length]
    # A diagonal is one strided axis whose step is the sum of the two source
    # strides.  The native helper preserves the collapsed ndarray base chain,
    # so einsum can return this as a write-through view just like NumPy.
    from _rnp import _as_strided
    return _as_strided(
        cropped,
        cropped.shape[:-2] + (length,),
        cropped.strides[:-2] + (cropped.strides[-2] + cropped.strides[-1],),
        cropped.flags.writeable,
    )


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    d = diagonal(a, offset, axis1, axis2)
    return np.add.reduce(d if dtype is None else d.astype(dtype), axis=-1)


def dot(a, b, out=None):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        res = np.multiply(a, b)
        return _to_out(res, out) if out is not None else res
    if a.shape[-1] != b.shape[-2 if b.ndim > 1 else -1]:
        raise ValueError(
            f"shapes {a.shape} and {b.shape} not aligned: "
            f"{a.shape[-1]} (dim {a.ndim - 1}) != "
            f"{b.shape[-2 if b.ndim > 1 else -1]} "
            f"(dim {builtins.max(b.ndim - 2, 0)})")
    n = a.shape[-1]
    if n == 0:
        shape = a.shape[:-1]
        if b.ndim > 1:
            shape += b.shape[:-2] + b.shape[-1:]
        res = np.zeros(shape, dtype=np.result_type(a, b))
        if res.ndim == 0:
            res = res[()]
        return _to_out(res, out) if out is not None else res
    a2 = a.reshape(-1, n)
    if b.ndim == 1:
        res = np.add.reduce(np.multiply(a2, b.reshape(1, n)), axis=1)
        res = res.reshape(a.shape[:-1])
    else:
        bt = np.moveaxis(b, -2, 0).reshape(n, -1)
        res = np.add.reduce(
            np.multiply(a2.reshape(a2.shape[0], n, 1),
                        bt.reshape(1, n, bt.shape[1])), axis=1)
        res = res.reshape(a.shape[:-1] + b.shape[:-2] + b.shape[-1:])
    if res.ndim == 0:
        res = res[()]
    return _to_out(res, out) if out is not None else res


def vdot(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    if a.dtype.kind == "c":
        a = np.conjugate(a)
    return np.add.reduce(np.multiply(a, b))[()]


def inner(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        return np.multiply(a, b)
    n = a.shape[-1]
    if b.shape[-1] != n:
        raise ValueError(
            f"shapes {a.shape} and {b.shape} not aligned: {n} (dim "
            f"{a.ndim - 1}) != {b.shape[-1]} (dim {b.ndim - 1})")
    if n == 0:
        res = np.zeros(a.shape[:-1] + b.shape[:-1],
                       dtype=np.result_type(a, b))
        return res[()] if res.ndim == 0 else res
    a2 = a.reshape(-1, n)
    b2 = b.reshape(-1, n)
    res = np.add.reduce(
        np.multiply(a2.reshape(a2.shape[0], 1, n),
                    b2.reshape(1, b2.shape[0], n)), axis=2)
    res = res.reshape(a.shape[:-1] + b.shape[:-1])
    return res[()] if res.ndim == 0 else res


def outer(a, b, out=None):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    res = np.multiply(a.reshape(-1, 1), b.reshape(1, -1))
    return _to_out(res, out) if out is not None else res


def matmul(a, b, out=None):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("matmul: Input operand does not have enough "
                         "dimensions")
    a1d, b1d = a.ndim == 1, b.ndim == 1
    if a1d:
        a = a.reshape(1, -1)
    if b1d:
        b = b.reshape(-1, 1)
    n, k, m = a.shape[-2], a.shape[-1], b.shape[-1]
    if b.shape[-2] != k:
        raise ValueError(
            f"matmul: Input operand 1 has a mismatch in its core dimension 0")
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = np.broadcast_to(a, batch + (n, k))
    b = np.broadcast_to(b, batch + (k, m))
    res = np.add.reduce(
        np.multiply(a.reshape(batch + (n, k, 1)),
                    b.reshape(batch + (1, k, m))), axis=len(batch) + 1)
    if a1d and b1d:
        res = res.reshape(batch)
    elif a1d:
        res = res.reshape(batch + (m,))
    elif b1d:
        res = res.reshape(batch + (n,))
    if res.ndim == 0:
        res = res[()]
    return _to_out(res, out) if out is not None else res


def tensordot(a, b, axes=2):
    a, b = np.asarray(a), np.asarray(b)
    try:
        iter(axes)
    except Exception:
        axes_a = list(builtins.range(-axes, 0))
        axes_b = list(builtins.range(0, axes))
    else:
        axes_a, axes_b = axes
        axes_a = [axes_a] if np.ndim(axes_a) == 0 else list(axes_a)
        axes_b = [axes_b] if np.ndim(axes_b) == 0 else list(axes_b)
    na, nb = len(axes_a), len(axes_b)
    as_, nda = a.shape, a.ndim
    bs, ndb = b.shape, b.ndim
    if na != nb:
        raise ValueError("shape-mismatch for sum")
    axes_a = [ax + nda if ax < 0 else ax for ax in axes_a]
    axes_b = [ax + ndb if ax < 0 else ax for ax in axes_b]
    for k in builtins.range(na):
        if as_[axes_a[k]] != bs[axes_b[k]]:
            raise ValueError("shape-mismatch for sum")
    notin_a = [k for k in builtins.range(nda) if k not in axes_a]
    notin_b = [k for k in builtins.range(ndb) if k not in axes_b]
    newaxes_a = notin_a + axes_a
    newaxes_b = axes_b + notin_b
    N2a = 1
    for ax in axes_a:
        N2a *= as_[ax]
    olda = [as_[ax] for ax in notin_a]
    oldb = [bs[ax] for ax in notin_b]
    if N2a == 0:
        out = np.empty(olda + oldb, dtype=np.result_type(a, b))
        out[...] = 0
        return out
    at = a.transpose(tuple(newaxes_a)).reshape(-1, N2a)
    bt = b.transpose(tuple(newaxes_b)).reshape(N2a, -1)
    res = np.dot(at, bt)
    return np.asarray(res).reshape(olda + oldb)


def kron(a, b):
    b = np.asanyarray(b)
    a = np.asarray(a)
    ndb, nda = b.ndim, a.ndim
    nd = builtins.max(ndb, nda)
    if nda == 0 or ndb == 0:
        return np.multiply(a, b)
    as_ = (1,) * (nd - nda) + a.shape
    bs = (1,) * (nd - ndb) + b.shape
    a = a.reshape(as_)
    b = b.reshape(bs)
    a_arr = a.reshape(sum(((s, 1) for s in as_), ()))
    b_arr = b.reshape(sum(((1, s) for s in bs), ()))
    res = np.multiply(a_arr, b_arr)
    return res.reshape(tuple(x * y for x, y in zip(as_, bs)))


# ---------------------------------------------------------------------------
# Selection / ordering
# ---------------------------------------------------------------------------

def partition(a, kth, axis=-1, kind='introselect', order=None):
    # A full sort satisfies the partition postcondition (every element before
    # `kth` is <= it and every one after is >= it); it is simply stronger than
    # required.  Correct, not optimal.
    a = np.asanyarray(a).copy()
    if axis is None:
        a = a.reshape(-1)
        axis = 0
    _check_kth(kth, a.shape[normalize_axis_index(axis, a.ndim)])
    a.sort(axis=axis)
    return a


def argpartition(a, kth, axis=-1, kind='introselect', order=None):
    a = np.asanyarray(a)
    if axis is None:
        a = a.reshape(-1)
        axis = 0
    _check_kth(kth, a.shape[normalize_axis_index(axis, a.ndim)])
    return a.argsort(axis=axis)


def _check_kth(kth, n):
    for k in (kth if isinstance(kth, (list, tuple)) else [kth]):
        k = operator.index(k)
        if not (-n <= k < n):
            raise ValueError(f"kth(={k}) out of bounds ({n})")


def sort_complex(a):
    b = np.asarray(a).copy()
    b.sort()
    if b.dtype.kind != "c":
        if b.dtype.itemsize in (1, 2):
            return b.astype(np.complex64)
        return b.astype(np.complex128)
    return b


def lexsort(keys, axis=-1):
    keys = list(keys)
    arrays = [np.asarray(k) for k in keys]
    if not arrays:
        raise ValueError("need sequence of keys with len > 0 in lexsort")
    shape = arrays[0].shape
    n = shape[axis]
    idx = list(builtins.range(n))
    if len(shape) == 1:
        cols = [a.tolist() for a in arrays]
        idx.sort(key=lambda i: tuple(c[i] for c in builtins.reversed(cols)))
        return np.array(idx, np.intp)
    raise NotImplementedError("lexsort on >1-D keys is not implemented by rnp")


def bincount(x, weights=None, minlength=0):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("object too deep for desired array")
    if x.dtype.kind not in "iub":
        raise TypeError("Cannot cast array data from "
                        f"{x.dtype} to dtype('int64') according to the rule "
                        "'safe'")
    vals = [int(v) for v in _flat(x)]
    if vals and builtins.min(vals) < 0:
        raise ValueError("'list' argument must have no negative elements")
    size = builtins.max(minlength, (builtins.max(vals) + 1) if vals else 0)
    if weights is None:
        out = [0] * size
        for v in vals:
            out[v] += 1
        return np.array(out, np.intp)
    w = np.asarray(weights)
    if w.shape != x.shape:
        raise ValueError("The weights and list don't have the same length.")
    wl = list(_flat(w))
    out = [0.0] * size
    for v, wv in zip(vals, wl):
        out[v] += wv
    return np.array(out, np.float64)


# ---------------------------------------------------------------------------
# Flat-index conversions
# ---------------------------------------------------------------------------

def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    dims = tuple(dims)
    arrays = [np.asarray(m) for m in multi_index]
    if len(arrays) != len(dims):
        raise ValueError(
            "parameter multi_index must be a sequence of length "
            f"{len(dims)}")
    shape = np.broadcast_shapes(*[a.shape for a in arrays])
    arrays = [np.broadcast_to(a, shape) for a in arrays]
    flat = [list(_flat(a)) for a in arrays]
    modes = mode if isinstance(mode, (tuple, list)) else [mode] * len(dims)
    out = []
    for i in builtins.range(int(np.prod(np.array(shape))) if shape else 1):
        idxs = []
        for d, (col, m) in enumerate(zip(flat, modes)):
            v = int(col[i])
            v = _apply_mode(v, dims[d], m)
            idxs.append(v)
        acc = 0
        rng = builtins.range(len(dims)) if order == 'C' else builtins.range(
            len(dims) - 1, -1, -1)
        for d in rng:
            acc = acc * dims[d] + idxs[d]
        out.append(acc)
    res = np.array(out, np.intp)
    return res.reshape(shape) if shape else res.reshape(())[()]


def _apply_mode(v, n, mode):
    if mode == 'raise':
        if not (-n <= v < n):
            raise ValueError(
                f"invalid entry in coordinates array")
        return v % n
    if mode == 'wrap':
        return v % n
    if mode == 'clip':
        return builtins.min(builtins.max(v, 0), n - 1)
    raise ValueError(f"Invalid mode {mode!r}")


def unravel_index(indices, shape, order='C'):
    shape = tuple(shape) if np.iterable(shape) else (shape,)
    arr = np.asarray(indices)
    total = 1
    for s in shape:
        total *= s
    vals = list(_flat(arr))
    for v in vals:
        if not (0 <= int(v) < total):
            raise ValueError(
                f"index {int(v)} is out of bounds for array with size {total}")
    cols = [[] for _ in shape]
    order_range = (builtins.range(len(shape) - 1, -1, -1) if order == 'C'
                   else builtins.range(len(shape)))
    for v in vals:
        v = int(v)
        for d in order_range:
            cols[d].append(v % shape[d])
            v //= shape[d]
    out = tuple(np.array(c, np.intp).reshape(arr.shape) for c in cols)
    if arr.ndim == 0:
        out = tuple(o[()] for o in out)
    return out


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------

def packbits(a, axis=None, bitorder='big'):
    a = np.asarray(a)
    if a.dtype.kind not in "biu":
        raise TypeError("Expected an input array of integer or boolean "
                        "data type")
    if bitorder not in ('big', 'little'):
        raise ValueError("'order' must be either 'little' or 'big'")
    if axis is None:
        bits = [1 if v else 0 for v in _flat(a)]
        return np.array(_pack(bits, bitorder), np.uint8)
    axis = normalize_axis_index(axis, a.ndim)
    a = np.moveaxis(a, axis, -1)
    outer_shape = a.shape[:-1]
    flat = a.reshape(-1, a.shape[-1]).tolist()
    packed = [_pack([1 if v else 0 for v in row], bitorder) for row in flat]
    n = len(packed[0]) if packed else (a.shape[-1] + 7) // 8
    res = np.array(packed, np.uint8).reshape(outer_shape + (n,))
    return np.moveaxis(res, -1, axis)


def _pack(bits, bitorder):
    out = []
    for i in builtins.range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        chunk = chunk + [0] * (8 - len(chunk))
        if bitorder == 'little':
            chunk = chunk[::-1]
        byte = 0
        for b in chunk:
            byte = (byte << 1) | b
        out.append(byte)
    return out


def unpackbits(a, axis=None, count=None, bitorder='big'):
    a = np.asarray(a)
    if a.dtype != np.dtype(np.uint8):
        raise TypeError("Expected an input array of unsigned byte data type")
    if bitorder not in ('big', 'little'):
        raise ValueError("'order' must be either 'little' or 'big'")

    def expand(row):
        bits = []
        for byte in row:
            b = [(int(byte) >> (7 - i)) & 1 for i in builtins.range(8)]
            if bitorder == 'little':
                b = b[::-1]
            bits.extend(b)
        if count is not None:
            bits = bits[:count] if count >= 0 else bits[:count]
        return bits

    if axis is None:
        return np.array(expand(list(_flat(a))), np.uint8)
    axis = normalize_axis_index(axis, a.ndim)
    a = np.moveaxis(a, axis, -1)
    outer_shape = a.shape[:-1]
    rows = a.reshape(-1, a.shape[-1]).tolist()
    out = [expand(r) for r in rows]
    n = len(out[0]) if out else (a.shape[-1] * 8 if count is None else count)
    res = np.array(out, np.uint8).reshape(outer_shape + (n,))
    return np.moveaxis(res, -1, axis)


# ---------------------------------------------------------------------------
# Iterator / buffer constructors
# ---------------------------------------------------------------------------

def fromiter(iterable_, dtype, count=-1, *, like=None):
    dt = np.dtype(dtype)
    vals = []
    for i, v in enumerate(iterable_):
        if count >= 0 and i >= count:
            break
        vals.append(v)
    if count >= 0 and len(vals) < count:
        raise ValueError("iterator too short")
    return np.array(vals, dt) if vals else np.empty((0,), dt)


def frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None):
    dt = np.dtype(dtype)
    data = memoryview(buffer).tobytes()[offset:]
    if count >= 0:
        data = data[:count * dt.itemsize]
    if len(data) % dt.itemsize:
        raise ValueError("buffer size must be a multiple of element size")
    return np.frombuffer_bytes(data, dt) if hasattr(np, "frombuffer_bytes") \
        else _frombytes(data, dt)


def _frombytes(data, dt):
    from rnp_numpy._core._textio import fromstring as _fs  # noqa: F401
    n = len(data) // dt.itemsize
    out = np.empty((n,), dt)
    mv = memoryview(out.__array_interface__ and out)  # placeholder
    raise NotImplementedError(
        "np.frombuffer is not implemented by rnp yet (needs an engine-level "
        "buffer constructor)")


# ---------------------------------------------------------------------------
# Convolution / correlation
# ---------------------------------------------------------------------------

def correlate(a, v, mode='valid'):
    a, v = np.asarray(a), np.asarray(v)
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError("object too deep for desired array")
    if a.size == 0 or v.size == 0:
        raise ValueError("a cannot be empty")
    if v.dtype.kind == "c":
        v = np.conjugate(v)
    return _corr_full(a, v[::-1], mode)[::1] if False else _convolve_impl(
        a, v[::-1], mode)


def convolve(a, v, mode='full'):
    a, v = np.asarray(a), np.asarray(v)
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError("object too deep for desired array")
    if a.size == 0 or v.size == 0:
        raise ValueError("v cannot be empty" if v.size == 0
                         else "a cannot be empty")
    if len(v) > len(a):
        a, v = v, a
    return _convolve_impl(a, v, mode)


def _convolve_impl(a, v, mode):
    n, m = a.size, v.size
    dt = np.result_type(a, v)
    av = a.astype(dt).tolist()
    vv = v.astype(dt).tolist()
    full = [0] * (n + m - 1)
    for i, x in enumerate(av):
        for j, y in enumerate(vv):
            full[i + j] += x * y
    if mode in ('full', 2):
        out = full
    elif mode in ('same', 1):
        start = (m - 1) // 2
        out = full[start:start + builtins.max(n, m)]
    elif mode in ('valid', 0):
        out = full[m - 1:n]
    else:
        raise ValueError("mode must be one of 'valid', 'same', or 'full'")
    return np.array(out, dt)


def _corr_full(a, v, mode):  # pragma: no cover - superseded
    return _convolve_impl(a, v, mode)


# ---------------------------------------------------------------------------
# Interpolation (numpy's `compiled_interp`)
# ---------------------------------------------------------------------------

def interp(x, xp, fp, left=None, right=None, period=None):
    xa = np.asarray(x)
    xp_a = np.asarray(xp, np.float64)
    complex_fp = np.asarray(fp).dtype.kind == "c"
    fp_a = np.asarray(fp, np.complex128 if complex_fp else np.float64)
    if xp_a.ndim != 1 or fp_a.ndim != 1:
        raise ValueError("object too deep for desired array")
    if xp_a.size != fp_a.size:
        raise ValueError("fp and xp are not of the same length")
    if xp_a.size == 0:
        raise ValueError("array of sample points is empty")

    xs = xp_a.tolist()
    ys = fp_a.tolist()
    if period is not None:
        if period == 0:
            raise ValueError("period must be a non-zero value")
        period = builtins.abs(period)
        xvals = [v % period for v in _flat(xa)]
        pairs = sorted(zip([v % period for v in xs], ys))
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        xs = [xs[-1] - period] + xs + [xs[0] + period]
        ys = [ys[-1]] + ys + [ys[0]]
        left = right = None
    else:
        xvals = list(_flat(xa))

    lo = ys[0] if left is None else left
    hi = ys[-1] if right is None else right

    out = []
    for xv in xvals:
        if xv != xv:  # NaN
            out.append(xv if not complex_fp else complex(xv, xv))
            continue
        if xv < xs[0]:
            out.append(lo)
        elif xv > xs[-1]:
            out.append(hi)
        else:
            j = _bisect_right(xs, xv) - 1
            if j >= len(xs) - 1:
                out.append(ys[-1])
                continue
            x0, x1 = xs[j], xs[j + 1]
            y0, y1 = ys[j], ys[j + 1]
            if x1 == x0:
                out.append(y0)
            else:
                slope = (y1 - y0) / (x1 - x0)
                res = slope * (xv - x0) + y0
                if res != res:
                    res = slope * (xv - x1) + y1
                    if res != res and y0 == y1:
                        res = y0
                out.append(res)
    dt = np.complex128 if complex_fp else np.float64
    arr = np.array(out, dt)
    if xa.ndim == 0:
        return arr.reshape(())[()]
    return arr.reshape(xa.shape)


def _bisect_right(seq, value):
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if value < seq[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# `np.place` / `np.insert` C helpers used by _function_base_impl
# ---------------------------------------------------------------------------

def _place(arr, mask, vals):
    mask = np.asarray(mask)
    idx = [i for i, v in enumerate(_flat(mask)) if v]
    if not idx:
        return
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0:
        raise ValueError("Cannot insert from an empty array!")
    flat = arr.reshape(-1)
    vl = vals.tolist()
    for k, i in enumerate(idx):
        flat[i] = vl[k % len(vl)]


def _insert(arr, mask, vals):  # pragma: no cover - alias kept for parity
    return _place(arr, mask, vals)


#: Names the top-level package wires in (only over not-implemented stubs).
WIRED = (
    "iterable", "asanyarray", "ascontiguousarray", "asfortranarray",
    "require", "linspace", "logspace", "geomspace", "count_nonzero", "ptp",
    "var", "std", "cumsum", "cumprod", "clip", "round", "around", "flip",
    "fliplr", "flipud", "roll", "rot90", "rollaxis", "expand_dims",
    "array_equiv", "diagonal", "trace", "dot", "vdot", "inner", "outer",
    "matmul", "tensordot", "kron", "partition", "argpartition",
    "sort_complex", "lexsort", "bincount", "ravel_multi_index",
    "unravel_index", "packbits", "unpackbits", "fromiter", "correlate",
    "convolve", "interp", "argwhere", "isfortran", "einsum", "einsum_path",
    "cumulative_sum", "cumulative_prod",
)


def argwhere(a):
    a = np.asanyarray(a)
    if a.ndim == 0:
        return np.asarray(a.astype(builtins.bool)).reshape(1)[
            :1 if a else 0].reshape(-1, 0)
    idx = np.nonzero(a)
    return stack(idx, axis=-1) if idx[0].size else np.empty(
        (0, a.ndim), np.intp)


def isfortran(a):
    return builtins.bool(a.flags["F_CONTIGUOUS"] and not
                         a.flags["C_CONTIGUOUS"])


# ---------------------------------------------------------------------------
# Reduction wrappers: tuple `axis=` and `where=`.
#
# The engine's reductions take a single integer axis and no `where=`, but
# `ufunc.reduce` already accepts a tuple axis.  These wrappers delegate to the
# engine *unchanged* whenever neither extension is in play, so the existing
# `_core` behaviour on the common path is bit-for-bit what it was; they only
# add a path for the two forms upstream's `lib` code uses that the engine
# rejects.
# ---------------------------------------------------------------------------

def _make_reduction(name, ufunc, bool_result=False):
    engine = getattr(np, name)

    def reduction(a, axis=None, dtype=None, out=None, keepdims=_NoValue,
                  initial=_NoValue, where=_NoValue, **kwargs):
        has_where = where is not _NoValue and where is not True
        has_initial = initial is not _NoValue
        # Tuple axes and `out=` are native now.  Keep the ufunc fallback only
        # for the two options the ndarray reduction signature still does not
        # expose (`where` and `initial`).
        if not has_where and not has_initial:
            kw = dict(kwargs)
            if keepdims is not _NoValue:
                kw["keepdims"] = keepdims
            if dtype is not None:
                kw["dtype"] = dtype
            if out is not None:
                kw["out"] = out
            return engine(a, axis=axis, **kw)

        arr = np.asanyarray(a)
        kw = {}
        if keepdims is not _NoValue:
            kw["keepdims"] = keepdims
        if initial is not _NoValue:
            kw["initial"] = initial
        if has_where:
            kw["where"] = where
        if dtype is not None and not bool_result:
            kw["dtype"] = dtype
        if axis is None and arr.ndim > 1:
            arr = arr.reshape(-1)
            if has_where:
                kw["where"] = np.asarray(where).reshape(-1)
            axis = 0
        # The engine's multiply reduction currently folds ``initial`` into
        # more than one internal partial product.  Apply it once to the final
        # product, which is NumPy's contract for every output slice.
        deferred_initial = _NoValue
        if ufunc is np.multiply and has_initial:
            deferred_initial = kw.pop("initial")
        res = ufunc.reduce(arr, axis=axis, **kw)
        if deferred_initial is not _NoValue:
            res = ufunc(res, deferred_initial)
        if out is not None:
            out[...] = res
            return out
        return res

    reduction.__name__ = name
    return reduction


def _make_minmax_reduction(name, ufunc):
    """Add NumPy's omitted/``initial``/``where`` semantics to min/max."""
    engine = getattr(np, name)

    def reduction(a, axis=None, out=None, keepdims=_NoValue,
                  initial=_NoValue, where=_NoValue):
        has_where = where is not _NoValue and where is not True
        has_initial = initial is not _NoValue
        if not has_where and not has_initial:
            kw = {}
            if keepdims is not _NoValue:
                kw["keepdims"] = keepdims
            if out is not None:
                kw["out"] = out
            return engine(a, axis=axis, **kw)

        arr = np.asanyarray(a)
        kw = {}
        if keepdims is not _NoValue:
            kw["keepdims"] = keepdims
        if has_initial:
            kw["initial"] = initial
        if has_where:
            kw["where"] = where
        if axis is None and arr.ndim > 1:
            arr = arr.reshape(-1)
            if has_where:
                kw["where"] = np.asarray(where).reshape(-1)
            axis = 0
        res = ufunc.reduce(arr, axis=axis, **kw)
        if out is not None:
            out[...] = res
            return out
        return res

    reduction.__name__ = name
    return reduction


def _install_reductions(namespace):
    """Bind the wrapped reductions into `namespace` (the top-level package)."""
    for name, uf, is_bool in (
        ("sum", np.add, False),
        ("prod", np.multiply, False),
        ("all", np.logical_and, True),
        ("any", np.logical_or, True),
    ):
        namespace[name] = _make_reduction(name, uf, is_bool)
    for name, uf in (("amin", np.minimum), ("amax", np.maximum)):
        namespace[name] = _make_minmax_reduction(name, uf)
    namespace["min"] = namespace["amin"]
    namespace["max"] = namespace["amax"]


# ---------------------------------------------------------------------------
# `*_like` constructors gain numpy's `shape=` override.
# ---------------------------------------------------------------------------

def _make_like(alloc, fill_value):
    def like(a, dtype=None, order='K', subok=True, shape=None, *, device=None):
        a = np.asanyarray(a)
        shape = a.shape if shape is None else shape
        dtype = a.dtype if dtype is None else dtype
        if subok and type(a) is not np.ndarray:
            return np._new_like_subclass(
                a, dtype, fill_value, shape=shape)
        return alloc(shape, dtype)
    return like


def _install_likes(namespace):
    namespace["zeros_like"] = _make_like(np.zeros, 0)
    namespace["ones_like"] = _make_like(np.ones, 1)
    namespace["empty_like"] = _make_like(np.empty, np._LIKE_EMPTY)

    def full_like(a, fill_value, dtype=None, order='K', subok=True,
                  shape=None, *, device=None):
        a = np.asanyarray(a)
        shape = a.shape if shape is None else shape
        dtype = a.dtype if dtype is None else dtype
        if subok and type(a) is not np.ndarray:
            return np._new_like_subclass(
                a, dtype, fill_value, shape=shape)
        return np.full(shape, fill_value, dtype)

    namespace["full_like"] = full_like


# ---------------------------------------------------------------------------
# ndarray methods.
#
# numpy exposes most of the above as methods too, and upstream's `lib` source
# uses the method spelling freely (`a.cumsum(...)`, `a.clip(...)`,
# `a.partition(...)`).  The engine's ndarray does not carry them, so they are
# attached here as thin adaptors over the functions in this module.  Only
# names the type does *not* already have are installed, so nothing the engine
# implements natively is shadowed.
# ---------------------------------------------------------------------------

def _install_ndarray_methods():
    installed = []

    def _m(name, fn):
        if not hasattr(np.ndarray, name):
            setattr(np.ndarray, name, fn)
            installed.append(name)

    _m("cumsum", lambda self, axis=None, dtype=None, out=None:
       cumsum(self, axis, dtype, out))
    _m("cumprod", lambda self, axis=None, dtype=None, out=None:
       cumprod(self, axis, dtype, out))
    _m("clip", lambda self, min=None, max=None, out=None, **k:
       clip(self, min, max, out, **k))
    _m("round", lambda self, decimals=0, out=None: round(self, decimals, out))
    _m("var", lambda self, axis=None, dtype=None, out=None, ddof=0,
       keepdims=False, **k: var(self, axis, dtype, out, ddof, keepdims, **k))
    _m("std", lambda self, axis=None, dtype=None, out=None, ddof=0,
       keepdims=False, **k: std(self, axis, dtype, out, ddof, keepdims, **k))
    _m("ptp", lambda self, axis=None, out=None, keepdims=False:
       ptp(self, axis, out, keepdims))
    _m("dot", lambda self, b, out=None: np.dot(self, b, out=out))
    _m("diagonal", lambda self, offset=0, axis1=0, axis2=1:
       diagonal(self, offset, axis1, axis2))
    _m("trace", lambda self, offset=0, axis1=0, axis2=1, dtype=None,
       out=None: trace(self, offset, axis1, axis2, dtype, out))
    _m("conj", lambda self: np.conjugate(self))
    _m("conjugate", lambda self: np.conjugate(self))

    def _partition(self, kth, axis=-1, kind='introselect', order=None):
        # in-place, like numpy's method
        self[...] = partition(self, kth, axis, kind, order)

    _m("partition", _partition)
    _m("argpartition", lambda self, kth, axis=-1, kind='introselect',
       order=None: argpartition(self, kth, axis, kind, order))
    return installed


#: numpy dispatches `np.interp` to one of two C loops depending on whether
#: `fp` is complex; the Python implementation above handles both, so the
#: complex entry point is the same function.
interp_complex = interp
compiled_interp = interp
compiled_interp_complex = interp


# ---------------------------------------------------------------------------
# einsum
#
# A correctness-first evaluator: every operand is expanded to the full index
# space, the product is formed, and the indices absent from the output are
# summed away.  That is the definition of the notation rather than an
# optimised contraction order, which is the right trade here — several
# upstream test modules call `np.einsum` at *module* scope, so without it the
# whole file is a collection error.
# ---------------------------------------------------------------------------

def _parse_einsum(subscripts, operands):
    if "->" in subscripts:
        lhs, rhs = subscripts.split("->")
        explicit = True
    else:
        lhs, rhs = subscripts, None
        explicit = False
    terms = [t.strip().replace(" ", "") for t in lhs.split(",")]
    if len(terms) != len(operands):
        raise ValueError(
            "more operands provided to einstein sum function than specified "
            "in the subscripts string")

    # Ellipsis expansion: give the broadcast axes private index letters.
    max_ell = 0
    for term, op in zip(terms, operands):
        if "..." in term:
            max_ell = builtins.max(max_ell, np.ndim(op) - (len(term) - 3))
    pool = [c for c in "ZYXWVUTSRQPONMLKJIHGFEDCBA" if c not in subscripts]
    ell_idx = "".join(pool[:max_ell][::-1])
    new_terms = []
    for term, op in zip(terms, operands):
        if "..." in term:
            n = np.ndim(op) - (len(term) - 3)
            term = term.replace("...", ell_idx[len(ell_idx) - n:])
        new_terms.append(term)
    terms = new_terms

    if explicit:
        out = rhs.strip().replace(" ", "").replace("...", ell_idx)
    else:
        counts = {}
        for term in terms:
            for ch in set(term):
                counts[ch] = counts.get(ch, 0) + term.count(ch)
        out = ell_idx + "".join(
            sorted(ch for ch, n in counts.items()
                   if n == 1 and ch not in ell_idx))
    return terms, out


def einsum(*operands, out=None, dtype=None, order='K', casting='safe',
           optimize=False):
    from rnp_numpy._core.einsumfunc import einsum as _einsum
    return _einsum(*operands, out=out, optimize=optimize, dtype=dtype,
                   order=order, casting=casting)


def einsum_path(*operands, optimize='greedy', einsum_call=False):
    # Keep NumPy's pure-Python parser and planner intact in its canonical
    # module.  This wrapper avoids duplicating a 900-line compatibility
    # surface here and preserves the existing late wiring during package init.
    from rnp_numpy._core.einsumfunc import einsum_path as _einsum_path
    return _einsum_path(*operands, optimize=optimize,
                        einsum_call=einsum_call)


# ---------------------------------------------------------------------------
# Array-API spellings numpy 2 added alongside the classic names.
# ---------------------------------------------------------------------------

def cumulative_sum(x, /, *, axis=None, dtype=None, out=None,
                   include_initial=False):
    return _cumulative(np.add, x, axis, dtype, out, include_initial, 0)


def cumulative_prod(x, /, *, axis=None, dtype=None, out=None,
                    include_initial=False):
    return _cumulative(np.multiply, x, axis, dtype, out, include_initial, 1)


def _cumulative(uf, x, axis, dtype, out, include_initial, ident):
    x = np.asanyarray(x)
    if axis is None:
        if x.ndim > 1:
            raise ValueError(
                "The `axis` argument is required for arrays with more than "
                "one dimension")
        axis = 0
    axis = normalize_axis_index(axis, builtins.max(x.ndim, 1))
    res = _accumulate(uf, x, axis, dtype)
    if include_initial:
        pad_shape = list(res.shape)
        pad_shape[axis] = 1
        pad = np.full(tuple(pad_shape), ident, res.dtype)
        res = concatenate((pad, res), axis=axis)
    if out is not None:
        out[...] = res
        return out
    return res
