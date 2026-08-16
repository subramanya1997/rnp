"""Shape-joining routines, built on the Rust primitives.

These are our own implementations (upstream's `shape_base.py` is not copied),
except for `block` and its private helpers, whose decomposition mirrors
upstream exactly because the tests exercise `_block_setup`, `_block_slicing`
and `_block_concatenate` individually.
"""

import builtins
import functools
import itertools
import operator

from .. import asarray, can_cast, empty, ndarray, promote_types, result_type
from .. import dtype as _dtype
from ..exceptions import AxisError

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

_CASTING_RULES = ("no", "equiv", "safe", "same_kind", "unsafe")


def _asarr(a):
    return a if isinstance(a, ndarray) else asarray(a)


def _ndim_of(a):
    return a.ndim if isinstance(a, ndarray) else _asarr(a).ndim


def _size_of(a):
    return a.size if isinstance(a, ndarray) else _asarr(a).size


def _normalize_axis(axis, ndim):
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise AxisError(axis, ndim)
    return axis


def _check_casting(casting):
    if casting not in _CASTING_RULES:
        raise ValueError(
            "casting must be one of 'no', 'equiv', 'safe', 'same_kind', "
            f"'unsafe' (got '{casting}')"
        )


def _arrays_for_stack_dispatcher(arrays):
    if not hasattr(arrays, "__getitem__"):
        raise TypeError(
            'arrays to stack must be passed as a "sequence" type '
            "such as list or tuple."
        )
    return tuple(arrays)


def _cast_check(arrs, dt, casting):
    for a in arrs:
        if not can_cast(a.dtype, dt, casting=casting):
            raise TypeError(
                f"Cannot cast array data from {a.dtype!r} to {dt!r} "
                f"according to the rule {casting!r}"
            )


def concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
    _check_casting(casting)
    if out is not None and dtype is not None:
        raise TypeError(
            "concatenate() only takes `out` or `dtype` as an argument, "
            "but both were provided."
        )
    if not hasattr(arrays, "__getitem__"):
        raise TypeError("The first input argument needs to be a sequence")

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

    for i, a in enumerate(arrs):
        if a.ndim != nd:
            raise ValueError(
                "all the input arrays must have same number of dimensions, "
                f"but the array at index 0 has {nd} dimension(s) and the "
                f"array at index {i} has {a.ndim} dimension(s)"
            )
    for ax in range(nd):
        if ax == axis:
            continue
        for i, a in enumerate(arrs):
            if a.shape[ax] != arrs[0].shape[ax]:
                raise ValueError(
                    "all the input array dimensions except for the "
                    "concatenation axis must match exactly, but along "
                    f"dimension {ax}, the array at index 0 has size "
                    f"{arrs[0].shape[ax]} and the array at index {i} has "
                    f"size {a.shape[ax]}"
                )

    if out is not None:
        dt = out.dtype
    elif dtype is not None:
        dt = dtype if hasattr(dtype, "kind") else _dtype(dtype)
        if getattr(dt, "subdtype", None) is not None:
            raise TypeError(
                "concatenate() does not support subarray dtype "
                f"{dt!r} as the requested output dtype."
            )
    else:
        dt = arrs[0].dtype
        for a in arrs[1:]:
            dt = promote_types(dt, a.dtype)

    if casting != "unsafe" or out is not None or dtype is not None:
        _cast_check(arrs, dt, casting)

    shape = list(arrs[0].shape)
    shape[axis] = builtins.sum(a.shape[axis] for a in arrs)
    shape = tuple(shape)

    if out is None:
        # Match numpy: an all-Fortran-ordered input set yields an
        # F-ordered result.
        f_order = builtins.all(a.flags["F_CONTIGUOUS"] for a in arrs)
        c_order = builtins.all(a.flags["C_CONTIGUOUS"] for a in arrs)
        order = "F" if f_order and not c_order else "C"
        result = empty(shape, dt, order=order)
    else:
        if tuple(out.shape) != shape:
            raise ValueError("Output array is the wrong shape")
        result = out

    pos = 0
    for a in arrs:
        n = a.shape[axis]
        if n:
            index = [slice(None)] * nd
            index[axis] = slice(pos, pos + n)
            result[tuple(index)] = a
        pos += n
    return result


def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    arrays = _arrays_for_stack_dispatcher(arrays)
    arrs = [_asarr(a) for a in arrays]
    if not arrs:
        raise ValueError("need at least one array to stack")
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError("all input arrays must have the same shape")
    nd = arrs[0].ndim + 1
    axis = _normalize_axis(axis, nd)
    expanded = [a[(slice(None),) * axis + (None,)] for a in arrs]
    return concatenate(
        expanded, axis=axis, out=out, dtype=dtype, casting=casting
    )


def atleast_1d(*arys):
    res = []
    for a in arys:
        a = _asarr(a)
        res.append(a.reshape(1) if a.ndim == 0 else a)
    return res[0] if len(res) == 1 else tuple(res)


def atleast_2d(*arys):
    res = []
    for a in arys:
        a = _asarr(a)
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a[None, :]
        res.append(a)
    return res[0] if len(res) == 1 else tuple(res)


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
    return res[0] if len(res) == 1 else tuple(res)


def vstack(tup, *, dtype=None, casting="same_kind"):
    tup = _arrays_for_stack_dispatcher(tup)
    arrs = atleast_2d(*tup)
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    return concatenate(arrs, 0, dtype=dtype, casting=casting)


def hstack(tup, *, dtype=None, casting="same_kind"):
    tup = _arrays_for_stack_dispatcher(tup)
    arrs = atleast_1d(*tup)
    if not isinstance(arrs, tuple):
        arrs = (arrs,)
    axis = 0 if arrs and arrs[0].ndim == 1 else 1
    return concatenate(arrs, axis, dtype=dtype, casting=casting)


# --------------------------------------------------------------------------
# `block` — the decomposition below mirrors upstream `numpy/_core/shape_base.py`
# because `test_shape_base.py` imports and drives the private helpers directly.
# --------------------------------------------------------------------------


def _block_format_index(index):
    """Convert a list of indices ``[0, 1, 2]`` into ``"arrays[0][1][2]"``."""
    idx_str = "".join(f"[{i}]" for i in index if i is not None)
    return "arrays" + idx_str


def _block_check_depths_match(arrays, parent_index=[]):
    """Recursively check that the depths of nested lists in `arrays` match.

    Returns ``(first_index, max_arr_ndim, final_size)``.
    """
    if isinstance(arrays, tuple):
        raise TypeError(
            f"{_block_format_index(parent_index)} is a tuple. "
            "Only lists can be used to arrange blocks, and np.block does "
            "not allow implicit conversion from tuple to ndarray."
        )
    elif isinstance(arrays, list) and len(arrays) > 0:
        idxs_ndims = (
            _block_check_depths_match(arr, parent_index + [i])
            for i, arr in enumerate(arrays)
        )

        first_index, max_arr_ndim, final_size = next(idxs_ndims)
        for index, ndim, size in idxs_ndims:
            final_size += size
            if ndim > max_arr_ndim:
                max_arr_ndim = ndim
            if len(index) != len(first_index):
                raise ValueError(
                    "List depths are mismatched. First element was at "
                    f"depth {len(first_index)}, but there is an element at "
                    f"depth {len(index)} ({_block_format_index(index)})"
                )
            # propagate our flag that indicates an empty list at the bottom
            if index[-1] is None:
                first_index = index

        return first_index, max_arr_ndim, final_size
    elif isinstance(arrays, list) and len(arrays) == 0:
        # We've 'bottomed out' on an empty list
        return parent_index + [None], 0, 0
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        size = _size_of(arrays)
        return parent_index, _ndim_of(arrays), size


def _atleast_nd(a, ndim):
    # Ensures `a` has at least `ndim` dimensions by prepending
    # ones to `a.shape` as necessary
    a = _asarr(a)
    if a.ndim < ndim:
        a = a.reshape((1,) * (ndim - a.ndim) + tuple(a.shape))
    return a


def _accumulate(values):
    return list(itertools.accumulate(values))


def _concatenate_shapes(shapes, axis):
    """Given array shapes, return the resulting shape and slice prefixes."""
    # Cache a result that will be reused.
    shape_at_axis = [shape[axis] for shape in shapes]

    # Take a shape, any shape
    first_shape = shapes[0]
    first_shape_pre = first_shape[:axis]
    first_shape_post = first_shape[axis + 1:]

    if builtins.any(
        shape[:axis] != first_shape_pre or shape[axis + 1:] != first_shape_post
        for shape in shapes
    ):
        raise ValueError(f"Mismatched array shapes in block along axis {axis}.")

    shape = first_shape_pre + (builtins.sum(shape_at_axis),) + first_shape[axis + 1:]

    offsets_at_axis = _accumulate(shape_at_axis)
    slice_prefixes = [
        (slice(start, end),)
        for start, end in zip([0] + offsets_at_axis, offsets_at_axis)
    ]
    return shape, slice_prefixes


def _block_info_recursion(arrays, max_depth, result_ndim, depth=0):
    """Return ``(shape, slices, arrays)`` for the slicing-based algorithm."""
    if depth < max_depth:
        shapes, slices, arrays = zip(
            *[
                _block_info_recursion(arr, max_depth, result_ndim, depth + 1)
                for arr in arrays
            ]
        )

        axis = result_ndim - max_depth + depth
        shape, slice_prefixes = _concatenate_shapes(shapes, axis)

        # Prepend the slice prefix and flatten the slices
        slices = [
            slice_prefix + the_slice
            for slice_prefix, inner_slices in zip(slice_prefixes, slices)
            for the_slice in inner_slices
        ]

        # Flatten the array list
        arrays = functools.reduce(operator.add, arrays)

        return shape, slices, arrays
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        arr = _atleast_nd(arrays, result_ndim)
        return tuple(arr.shape), [()], [arr]


def _block(arrays, max_depth, result_ndim, depth=0):
    """Internal implementation of block based on repeated concatenation."""
    if depth < max_depth:
        arrs = [_block(arr, max_depth, result_ndim, depth + 1) for arr in arrays]
        return concatenate(arrs, axis=-(max_depth - depth))
    else:
        # We've 'bottomed out' - arrays is either a scalar or an array
        return _atleast_nd(arrays, result_ndim)


def _block_dispatcher(arrays):
    # Use isinstance(..., list) to match the behavior of np.block(), which
    # special cases list specifically rather than allowing for generic
    # iterables or tuple.
    if isinstance(arrays, list):
        for subarrays in arrays:
            yield from _block_dispatcher(subarrays)
    else:
        yield arrays


def _block_setup(arrays):
    """Returns ``(arrays, list_ndim, result_ndim, final_size)``."""
    bottom_index, arr_ndim, final_size = _block_check_depths_match(arrays)
    list_ndim = len(bottom_index)
    if bottom_index and bottom_index[-1] is None:
        raise ValueError(
            f"List at {_block_format_index(bottom_index)} cannot be empty"
        )
    result_ndim = builtins.max(arr_ndim, list_ndim)
    return arrays, list_ndim, result_ndim, final_size


def _block_slicing(arrays, list_ndim, result_ndim):
    shape, slices, arrays = _block_info_recursion(arrays, list_ndim, result_ndim)
    dtype = result_type(*[arr.dtype for arr in arrays])

    # Test preferring F only in the case that all input arrays are F
    F_order = builtins.all(arr.flags["F_CONTIGUOUS"] for arr in arrays)
    C_order = builtins.all(arr.flags["C_CONTIGUOUS"] for arr in arrays)
    order = "F" if F_order and not C_order else "C"
    result = empty(shape, dtype, order=order)

    for the_slice, arr in zip(slices, arrays):
        result[(Ellipsis,) + the_slice] = arr
    return result


def _block_concatenate(arrays, list_ndim, result_ndim):
    result = _block(arrays, list_ndim, result_ndim)
    if list_ndim == 0:
        # Catch an edge case where _block returns a view because
        # `arrays` is a single numpy array and not a list of numpy arrays.
        result = result.copy()
    return result


def block(arrays):
    arrays, list_ndim, result_ndim, final_size = _block_setup(arrays)

    # It was found through benchmarking that making an array of final size
    # around 256x256 was faster by straight concatenation.
    if list_ndim * final_size > (2 * 512 * 512):
        return _block_slicing(arrays, list_ndim, result_ndim)
    else:
        return _block_concatenate(arrays, list_ndim, result_ndim)
