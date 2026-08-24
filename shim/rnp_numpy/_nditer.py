"""A correctness-first implementation of :class:`numpy.nditer`.

The storage and view operations are supplied by ``_rnp.ndarray``.  This
module is intentionally concerned only with iterator policy: broadcasting,
axis order, iterator state, and (in later sections) buffering/casting.
"""

from __future__ import annotations

import builtins


def _shape_tuple(shape):
    return tuple(int(x) for x in shape)


def _shape_size(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def _broadcast_shape(shapes):
    ndim = max((len(shape) for shape in shapes), default=0)
    result = [1] * ndim
    for shape in shapes:
        for j, dim in enumerate(reversed(shape)):
            axis = ndim - 1 - j
            old = result[axis]
            if old != 1 and dim != 1 and old != dim:
                compact = " ".join(str(tuple(s)) for s in shapes)
                raise ValueError(
                    "operands could not be broadcast together with shapes " + compact
                )
            result[axis] = max(old, dim)
    return tuple(result)


def _normalize_operands(op, ndarray_type, asarray):
    # An ndarray is one operand, while list/tuple is numpy's multiple-operand
    # constructor spelling.  Python scalars and numpy scalar objects are also
    # a single operand.
    if isinstance(op, (list, tuple)):
        raw = list(op)
    else:
        raw = [op]
    if not raw:
        raise ValueError("Must provide at least one operand")
    arrays = []
    scalar = []
    for value in raw:
        if value is None:
            raise TypeError("Iterator operand required copying or buffering")
        scalar.append(not isinstance(value, ndarray_type))
        arrays.append(asarray(value))
    return raw, arrays, scalar


def _normalize_op_flags(op_flags, nop):
    if op_flags is None:
        return [frozenset(("readonly",)) for _ in range(nop)]
    if isinstance(op_flags, str):
        return [frozenset((op_flags,)) for _ in range(nop)]
    flags = list(op_flags)
    # A flat list is applied to every operand; a nested list is per operand.
    if not flags or isinstance(flags[0], str):
        one = frozenset(flags)
        return [one for _ in range(nop)]
    if len(flags) != nop:
        raise ValueError(
            "op_flags must be a tuple or array of per-op flag-tuples"
        )
    return [frozenset(x) for x in flags]


class nditer:
    """Iterate one or more rnp arrays using NumPy's nditer protocol."""

    def __init__(
        self,
        op,
        flags=None,
        op_flags=None,
        op_dtypes=None,
        order="K",
        casting="safe",
        op_axes=None,
        itershape=None,
        buffersize=0,
    ):
        # Imports are delayed to avoid a cycle while rnp_numpy initializes.
        from . import asarray, ndarray

        self._raw_operands, arrays, self._scalar_operands = _normalize_operands(
            op, ndarray, asarray
        )
        self._op_flags = _normalize_op_flags(op_flags, len(arrays))
        self._flags = frozenset(() if flags is None else flags)
        self._closed = False
        self._buffersize = int(buffersize)
        self._order = str(order).upper()
        if self._order not in ("C", "F", "A", "K"):
            raise ValueError("order must be one of 'C', 'F', 'A', or 'K'")
        if op_dtypes is not None or op_axes is not None or itershape is not None:
            raise NotImplementedError(
                "op_dtypes, op_axes, and itershape require the extended nditer lane"
            )

        self._operands = tuple(arrays)
        self._logical_shape = _broadcast_shape(
            [_shape_tuple(a.shape) for a in arrays]
        )
        self._removed_axes = set()
        self._axis_fast, self._axis_reverse = self._choose_axis_order()
        self._start = 0
        self._stop = _shape_size(self._logical_shape)
        self._pos = self._start
        if self._stop == 0 and "zerosize_ok" not in self._flags:
            raise ValueError("Iteration of zero-sized operands is not enabled")

    def _mapped_stride(self, arr, axis):
        shift = len(self._logical_shape) - arr.ndim
        source_axis = axis - shift
        if source_axis < 0 or arr.shape[source_axis] == 1:
            return 0
        return int(arr.strides[source_axis])

    def _choose_axis_order(self):
        ndim = len(self._logical_shape)
        if self._order == "C":
            return list(range(ndim - 1, -1, -1)), [False] * ndim
        if self._order == "F":
            return list(range(ndim)), [False] * ndim
        if self._order == "A":
            use_f = all(a.flags.f_contiguous for a in self._operands)
            if use_f:
                return list(range(ndim)), [False] * ndim
            return list(range(ndim - 1, -1, -1)), [False] * ndim

        # KEEPORDER: rank axes by the first operand that has a meaningful
        # stride for both axes.  This reproduces the memory-increasing order
        # for ordinary views, transposes, and negative-stride slices.
        def key(axis):
            strides = [
                abs(self._mapped_stride(arr, axis))
                for arr in self._operands
                if self._mapped_stride(arr, axis) != 0
            ]
            return (min(strides) if strides else 1 << 62, -axis)

        fast = sorted(range(ndim), key=key)
        reverse = [False] * ndim
        for axis in range(ndim):
            for arr in self._operands:
                stride = self._mapped_stride(arr, axis)
                if stride:
                    reverse[axis] = stride < 0
                    break
        return fast, reverse

    def _coord_at(self, pos):
        coord = [0] * len(self._logical_shape)
        value = pos
        for axis in self._axis_fast:
            dim = self._logical_shape[axis]
            digit = value % dim if dim else 0
            value //= max(dim, 1)
            coord[axis] = dim - 1 - digit if self._axis_reverse[axis] else digit
        for axis in self._removed_axes:
            coord[axis] = 0
        return tuple(coord)

    def _operand_coord(self, arr, coord):
        shift = len(self._logical_shape) - arr.ndim
        out = []
        for axis, dim in enumerate(arr.shape):
            logical = coord[axis + shift]
            out.append(0 if dim == 1 else logical)
        return tuple(out)

    @staticmethod
    def _cell_view(arr, coord, writeable):
        if arr.ndim == 0:
            view = arr
        else:
            slices = tuple(slice(i, i + 1) for i in coord)
            view = arr[slices].reshape(())
        if not writeable:
            # This changes only the yielded view header, never the operand.
            view.flags.writeable = False
        return view

    def _value_for_operand(self, operand):
        if self.finished:
            raise ValueError("Iterator is past the end")
        arr = self._operands[operand]
        coord = self._operand_coord(arr, self._coord_at(self._pos))
        mode = self._op_flags[operand]
        return self._cell_view(arr, coord, "readonly" not in mode)

    @property
    def operands(self):
        return self._operands

    @property
    def dtypes(self):
        return tuple(a.dtype for a in self._operands)

    @property
    def nop(self):
        return len(self._operands)

    @property
    def itersize(self):
        return _shape_size(self._active_shape())

    @property
    def ndim(self):
        # Multi-index tracking preserves the public broadcast dimensions.
        if "multi_index" in self._flags:
            return len(self._active_shape())
        return 0 if not self._active_shape() else 1

    @property
    def shape(self):
        if "multi_index" in self._flags:
            return self._active_shape()
        if not self._active_shape():
            return ()
        return (self.itersize,)

    def _active_shape(self):
        return tuple(
            dim
            for axis, dim in enumerate(self._logical_shape)
            if axis not in self._removed_axes
        )

    @property
    def finished(self):
        return self._pos >= self._stop

    @property
    def iterindex(self):
        return self._pos

    @iterindex.setter
    def iterindex(self, value):
        value = int(value)
        if value < self._start or value > self._stop:
            raise ValueError("Iterator index out of bounds")
        self._pos = value

    @property
    def value(self):
        values = tuple(self._value_for_operand(i) for i in range(self.nop))
        return values[0] if self.nop == 1 else values

    @property
    def iterationneedsapi(self):
        return any(a.dtype.hasobject for a in self._operands)

    @property
    def has_delayed_bufalloc(self):
        return False

    @property
    def itviews(self):
        from . import broadcast_to

        return tuple(broadcast_to(a, self._logical_shape) for a in self._operands)

    def __len__(self):
        return self.itersize

    def __iter__(self):
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        value = self.value
        self._pos += 1
        return value

    def __getitem__(self, key):
        indices = list(range(self.nop))[key]
        if isinstance(indices, int):
            return self._value_for_operand(indices)
        return tuple(self._value_for_operand(i) for i in indices)

    def __setitem__(self, key, value):
        targets = self[key]
        if isinstance(key, slice):
            values = list(value)
            if len(values) != len(targets):
                raise ValueError("mismatched iterator assignment")
            for target, item in zip(targets, values):
                target[...] = item
        else:
            targets[...] = value

    def iternext(self):
        if not self.finished:
            self._pos += 1
        return not self.finished

    def reset(self):
        self._pos = self._start

    def close(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError("close() takes no arguments")
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def nested_iters(*args, **kwargs):
    raise NotImplementedError("numpy.nested_iters is not implemented yet")

