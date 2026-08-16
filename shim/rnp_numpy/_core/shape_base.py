"""Shape-joining routines, built on the Rust primitives.

These are our own implementations (upstream's `shape_base.py` is not copied),
covering the cases M0 can support. `block` and its private helpers exist so
that `test_shape_base.py` imports successfully; they raise
`NotImplementedError` until M4.
"""

import builtins

from .. import asarray, empty, ndarray, promote_types

__all__ = [
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "block",
    "concatenate",
    "hstack",
    "stack",
    "vstack",
]


def _asarr(a):
    return a if isinstance(a, ndarray) else asarray(a)


def _normalize_axis(axis, ndim):
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        from ..exceptions import AxisError

        raise AxisError(axis, ndim)
    return axis


def concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    arrs = [_asarr(a) for a in arrays]
    if not arrs:
        raise ValueError("need at least one array to concatenate")
    if axis is None:
        arrs = [a.copy().ravel() for a in arrs]
        axis = 0
    nd = arrs[0].ndim
    if nd == 0:
        raise ValueError("zero-dimensional arrays cannot be concatenated")
    axis = _normalize_axis(axis, nd)

    for a in arrs:
        if a.ndim != nd:
            raise ValueError(
                "all the input array dimensions except for the concatenation "
                "axis must match exactly"
            )
        for ax in range(nd):
            if ax != axis and a.shape[ax] != arrs[0].shape[ax]:
                raise ValueError(
                    "all the input array dimensions except for the "
                    "concatenation axis must match exactly"
                )

    dt = arrs[0].dtype if dtype is None else dtype
    if dtype is None:
        for a in arrs[1:]:
            dt = promote_types(dt, a.dtype)

    shape = list(arrs[0].shape)
    shape[axis] = builtins.sum(a.shape[axis] for a in arrs)
    result = empty(tuple(shape), dt) if out is None else out

    pos = 0
    for a in arrs:
        n = a.shape[axis]
        if n:
            index = [slice(None)] * nd
            index[axis] = slice(pos, pos + n)
            result[tuple(index)] = a
        pos += n
    return result


def stack(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    arrs = [_asarr(a) for a in arrays]
    if not arrs:
        raise ValueError("need at least one array to stack")
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError("all input arrays must have the same shape")
    nd = arrs[0].ndim + 1
    axis = _normalize_axis(axis, nd)
    expanded = [a[(slice(None),) * axis + (None,)] for a in arrs]
    return concatenate(expanded, axis=axis, out=out, dtype=dtype)


def atleast_1d(*arys):
    res = []
    for a in arys:
        a = _asarr(a)
        res.append(a.reshape(1) if a.ndim == 0 else a)
    return res[0] if len(res) == 1 else res


def atleast_2d(*arys):
    res = []
    for a in arys:
        a = _asarr(a)
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a[None, :]
        res.append(a)
    return res[0] if len(res) == 1 else res


def atleast_3d(*arys):
    res = []
    for a in arys:
        a = _asarr(a)
        if a.ndim == 0:
            a = a.reshape(1, 1, 1)
        elif a.ndim == 1:
            a = a[None, :, None]
        elif a.ndim == 2:
            a = a[:, :, None]
        res.append(a)
    return res[0] if len(res) == 1 else res


def vstack(tup, *, dtype=None, casting="same_kind"):
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return concatenate(arrs, 0, dtype=dtype)


def hstack(tup, *, dtype=None, casting="same_kind"):
    arrs = atleast_1d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    axis = 0 if arrs and arrs[0].ndim == 1 else 1
    return concatenate(arrs, axis, dtype=dtype)


# `block` and its private helpers land in M4; the names exist so that
# `test_shape_base.py` can be imported and collected.
def _block_setup(arrays):
    raise NotImplementedError("np.block is not implemented yet")


def _block_dispatcher(arrays):
    raise NotImplementedError("np.block is not implemented yet")


def _block_slicing(arrays, list_ndim, result_ndim):
    raise NotImplementedError("np.block is not implemented yet")


def _block_concatenate(arrays, list_ndim, result_ndim):
    raise NotImplementedError("np.block is not implemented yet")


def block(arrays):
    raise NotImplementedError("np.block is not implemented yet")
