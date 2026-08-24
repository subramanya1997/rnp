"""Correctness-first implementation of :class:`numpy.nditer`.

The ndarray storage and view operations come from ``_rnp.ndarray``.  Iterator
policy deliberately lives here: broadcasting, traversal order, index state,
temporary operands, and external loops.
"""

from __future__ import annotations

import copy as _copy


_GLOBAL_FLAGS = frozenset((
    "buffered", "c_index", "common_dtype", "copy_if_overlap",
    "delay_bufalloc", "external_loop", "f_index", "grow_inner", "growinner",
    "multi_index", "ranged", "reduce_ok", "refs_ok", "zerosize_ok",
))
_OP_FLAGS = frozenset((
    "aligned", "allocate", "arraymask", "contig", "copy", "nbo",
    "no_broadcast", "no_subtype", "overlap_assume_elementwise", "readonly",
    "readwrite", "updateifcopy", "writemasked", "writeonly",
    "writebackifcopy",
))
_ACCESS_FLAGS = frozenset(("readonly", "readwrite", "writeonly"))
_CASTING = frozenset(("no", "equiv", "safe", "same_kind", "unsafe"))


def _shape_tuple(shape):
    return tuple(int(x) for x in shape)


def _shape_size(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def _compact_shape(shape):
    return "(" + ",".join(str(x) for x in shape) + ")"


def _broadcast_shape(shapes):
    ndim = max((len(shape) for shape in shapes), default=0)
    result = [1] * ndim
    for shape in shapes:
        for j, dim in enumerate(reversed(shape)):
            axis = ndim - 1 - j
            old = result[axis]
            if old != 1 and dim != 1 and old != dim:
                compact = " ".join(_compact_shape(s) for s in shapes)
                raise ValueError(
                    "operands could not be broadcast together with shapes " + compact
                )
            if old == 1:
                result[axis] = dim
            elif dim != 1:
                result[axis] = old
    return tuple(result)


def _raw_operands(op, ndarray_type):
    if isinstance(op, (list, tuple)) and not isinstance(op, ndarray_type):
        raw = list(op)
    else:
        raw = [op]
    if not raw:
        raise ValueError("Must provide at least one operand")
    return raw


def _normalize_op_flags(op_flags, nop):
    if op_flags is None:
        result = [frozenset(("readonly",)) for _ in range(nop)]
    elif isinstance(op_flags, str):
        result = [frozenset((op_flags,)) for _ in range(nop)]
    else:
        flags = list(op_flags)
        if not flags or isinstance(flags[0], str):
            result = [frozenset(flags) for _ in range(nop)]
        else:
            if len(flags) != nop:
                raise ValueError(
                    "op_flags must be a tuple or array of per-op flag-tuples"
                )
            result = [frozenset(x) for x in flags]
    for mode in result:
        unknown = mode - _OP_FLAGS
        if unknown:
            raise ValueError(f"Unexpected per-op iterator flag {next(iter(unknown))!r}")
        if len(mode & _ACCESS_FLAGS) != 1:
            raise ValueError(
                "Only one of the iterator flags READWRITE, READONLY, and "
                "WRITEONLY may be specified for an operand"
            )
    return result


def _normalize_dtypes(op_dtypes, nop, dtype):
    if op_dtypes is None:
        return [None] * nop
    if isinstance(op_dtypes, (str, type)) or hasattr(op_dtypes, "kind"):
        return [dtype(op_dtypes)] * nop
    if nop == 1:
        try:
            return [dtype(op_dtypes)]
        except (TypeError, ValueError):
            pass
    specs = list(op_dtypes)
    if len(specs) != nop:
        raise ValueError("op_dtypes must be a tuple or array of per-op dtypes")
    return [None if spec is None else dtype(spec) for spec in specs]


class nditer:
    """Iterate one or more rnp arrays using NumPy's nditer protocol."""

    def __init__(self, op, flags=None, op_flags=None, op_dtypes=None,
                 order="K", casting="safe", op_axes=None, itershape=None,
                 buffersize=0):
        from . import asarray, can_cast, dtype, empty, ndarray, result_type

        raw = _raw_operands(op, ndarray)
        self._raw_operands = tuple(raw)
        self._op_flags = _normalize_op_flags(op_flags, len(raw))
        self._flags = set(() if flags is None else flags)
        unknown = self._flags - _GLOBAL_FLAGS
        if unknown:
            raise ValueError(f"Unexpected iterator global flag {next(iter(unknown))!r}")
        if "c_index" in self._flags and "f_index" in self._flags:
            raise ValueError("Iterator cannot track both a C and an F index")
        if "external_loop" in self._flags and self._flags & {
                "c_index", "f_index", "multi_index"}:
            raise ValueError("Iterator flag EXTERNAL_LOOP cannot be used with an index")

        self._closed = False
        self._buffersize = int(buffersize)
        self._casting = str(casting)
        if self._casting not in _CASTING:
            raise ValueError(f"casting must be one of {sorted(_CASTING)}")
        self._order = str(order).upper()
        if self._order not in ("C", "F", "A", "K"):
            raise ValueError("order must be one of 'C', 'F', 'A', or 'K'")
        self._op_axes = op_axes
        self._itershape_arg = itershape
        self._requested_dtypes = _normalize_dtypes(op_dtypes, len(raw), dtype)
        if op_flags is None:
            self._op_flags = [
                frozenset(("writeonly", "allocate"))
                if value is None else mode
                for value, requested, mode in zip(
                    raw, self._requested_dtypes, self._op_flags
                )
            ]

        arrays = []
        scalar = []
        for value, mode in zip(raw, self._op_flags):
            if value is None:
                if "allocate" not in mode:
                    raise TypeError("Iterator operand required copying or buffering")
                arrays.append(None)
                scalar.append(False)
                continue
            is_scalar = not isinstance(value, ndarray)
            if is_scalar and "readonly" not in mode:
                raise TypeError("Iterator operand is flagged as writeable, but is an object")
            arr = asarray(value)
            if "readonly" not in mode and not arr.flags.writeable:
                raise ValueError("operand array with iterator write flag set is read-only")
            if (arr.dtype.hasobject or arr.dtype.kind == "O") \
                    and "refs_ok" not in self._flags:
                raise TypeError(
                    "Iterator operand or requested dtype holds references, "
                    "but the REFS_OK flag was not enabled"
                )
            arrays.append(arr)
            scalar.append(is_scalar)
        self._scalar_operands = tuple(scalar)
        self._validate_writemasked(arrays)

        if "common_dtype" in self._flags:
            common_inputs = [
                requested if requested is not None else arr.dtype
                for arr, requested in zip(arrays, self._requested_dtypes)
                if arr is not None
            ]
            if common_inputs:
                common = result_type(*common_inputs)
                self._requested_dtypes = [
                    common if arr is not None or requested is None else requested
                    for arr, requested in zip(arrays, self._requested_dtypes)
                ]

        source_arrays = list(arrays)
        self._has_op_axes = op_axes is not None
        if op_axes is not None:
            arrays, logical_shape = self._apply_op_axes(arrays, op_axes, itershape)
        else:
            input_shapes = [_shape_tuple(a.shape) for a in arrays if a is not None]
            logical_shape = _broadcast_shape(input_shapes)
            if itershape is not None:
                requested = tuple(int(x) for x in itershape)
                if any(x < -1 for x in requested):
                    raise ValueError("invalid itershape dimension")
                if len(requested) != len(logical_shape):
                    if input_shapes:
                        raise ValueError("operands could not be broadcast to requested shape")
                    logical_shape = tuple(1 if x == -1 else x for x in requested)
                else:
                    logical_shape = tuple(
                        inferred if asked == -1 else asked
                        for inferred, asked in zip(logical_shape, requested)
                    )

        inputs = [a for a in arrays if a is not None]
        promotion_inputs = [
            a for a, mode in zip(arrays, self._op_flags)
            if a is not None and "writeonly" not in mode
        ]
        inferred_dtype = (result_type(*promotion_inputs)
                          if promotion_inputs else None)
        allocation_order = self._allocation_order(inputs)
        public_arrays = list(source_arrays)
        for i, (arr, mode, requested) in enumerate(
                zip(arrays, self._op_flags, self._requested_dtypes)):
            if arr is not None:
                continue
            if "readonly" in mode:
                raise ValueError("An iterator operand was NULL, but was flagged READONLY")
            target = requested if requested is not None else inferred_dtype
            if target is None:
                raise TypeError("cannot allocate an iterator output without a dtype")
            allocation_shape = (self._allocation_shapes[i]
                                if self._has_op_axes else logical_shape)
            allocated = self._allocate_output(
                empty, allocation_shape, target, inputs, i, allocation_order
            )
            public_arrays[i] = allocated
            arrays[i] = (self._remap_array(allocated, self._axis_mappings[i])
                         if self._has_op_axes else allocated)

        self._original_operands = tuple(arrays)
        self._writebacks = []
        arrays = self._prepare_temporary_operands(arrays, can_cast)
        self._operands = tuple(arrays)
        self._public_operands = (tuple(public_arrays) if self._has_op_axes
                                 else self._operands)
        self._logical_shape = tuple(logical_shape)
        for arr, mode in zip(self._operands, self._op_flags):
            if "no_broadcast" in mode:
                padded = (1,) * (len(self._logical_shape) - arr.ndim) + tuple(arr.shape)
                if padded != self._logical_shape:
                    raise ValueError(
                        f"non-broadcastable operand with shape {_compact_shape(arr.shape)} "
                        f"doesn't match the broadcast shape "
                        f"{_compact_shape(self._logical_shape)}"
                    )

        self._removed_axes = set()
        self._multi_index_removed = False
        self._axis_fast, self._axis_reverse = self._choose_axis_order()
        self._start = 0
        self._stop = _shape_size(self._logical_shape)
        self._pos = self._start
        self._chunk_cache = None
        self._chunk_start = None
        self._chunk_len = 0
        if self._stop == 0 and "zerosize_ok" not in self._flags:
            raise ValueError("Iteration of zero-sized operands is not enabled")

    def _allocation_order(self, inputs):
        if self._order == "F":
            return "F"
        if self._order in ("A", "K") and inputs and all(
                a.flags.f_contiguous and not a.flags.c_contiguous for a in inputs):
            return "F"
        return "C"

    def _allocate_output(self, empty, shape, target, inputs, operand,
                         fallback_order):
        if not shape or len(shape) == 1:
            return empty(shape, dtype=target, order=fallback_order)
        if not self._has_op_axes:
            return empty(shape, dtype=target, order=fallback_order)
        mapping = self._axis_mappings[operand]
        if self._order == "C":
            logical_fast = list(range(len(mapping) - 1, -1, -1))
        elif self._order == "F":
            logical_fast = list(range(len(mapping)))
        else:
            first = next((a for a in inputs if a is not None), None)
            if first is None:
                logical_fast = list(range(len(mapping) - 1, -1, -1))
            else:
                logical_fast = sorted(
                    range(len(mapping)),
                    key=lambda axis: (
                        -1 if int(first.shape[axis]) == 1
                        else abs(int(first.strides[axis])), -axis
                    ),
                )
        raw_fast = []
        for logical_axis in logical_fast:
            raw_axis = mapping[logical_axis]
            if raw_axis >= 0 and raw_axis not in raw_fast:
                raw_fast.append(raw_axis)
        raw_fast.extend(axis for axis in range(len(shape)) if axis not in raw_fast)
        slow_to_fast = list(reversed(raw_fast))
        base = empty(tuple(shape[axis] for axis in slow_to_fast), dtype=target)
        permutation = tuple(slow_to_fast.index(axis) for axis in range(len(shape)))
        return base.transpose(permutation)

    def _validate_writemasked(self, arrays):
        masks = [i for i, mode in enumerate(self._op_flags)
                 if "arraymask" in mode]
        writers = [i for i, mode in enumerate(self._op_flags)
                   if "writemasked" in mode]
        if not masks and not writers:
            return
        if len(masks) != 1 or not writers:
            raise ValueError("WRITEMASKED requires exactly one ARRAYMASK operand")
        mask_index = masks[0]
        mask_mode = self._op_flags[mask_index]
        if "writemasked" in mask_mode or "readonly" not in mask_mode:
            raise ValueError("ARRAYMASK must be a separate readonly operand")
        mask = arrays[mask_index]
        if mask is None:
            raise ValueError("ARRAYMASK cannot be an allocated operand")
        if not (mask.dtype.kind == "b"
                or (mask.dtype.kind == "u" and mask.dtype.itemsize == 1)):
            raise TypeError("ARRAYMASK must have boolean or uint8 dtype")
        for writer_index in writers:
            mode = self._op_flags[writer_index]
            writer = arrays[writer_index]
            if "readonly" in mode:
                raise ValueError("a WRITEMASKED operand must be writeable")
            if writer is None or mask.ndim > writer.ndim:
                raise ValueError("ARRAYMASK shape must match the WRITEMASKED operand")
            padded = (1,) * (writer.ndim - mask.ndim) + tuple(mask.shape)
            if any(mdim not in (1, wdim)
                   for mdim, wdim in zip(padded, writer.shape)):
                raise ValueError("ARRAYMASK shape must match the WRITEMASKED operand")
        # Large masked buffering is not yet emulated.  Fail promptly instead
        # of performing an unmasked write over millions of scalar iterations.
        if max((int(a.size) for a in arrays if a is not None), default=0) > 1000:
            raise ValueError("large WRITEMASKED buffering is not implemented")

    def _apply_op_axes(self, arrays, op_axes, itershape):
        if len(op_axes) != len(arrays):
            raise ValueError("op_axes must have one entry per operand")
        explicit = [tuple(x) for x in op_axes if x is not None]
        if explicit:
            iterator_ndim = len(explicit[0])
            if any(len(x) != iterator_ndim for x in explicit):
                raise ValueError("Each entry of op_axes must have the same size")
        elif itershape is not None:
            iterator_ndim = len(itershape)
        else:
            iterator_ndim = max((a.ndim for a in arrays if a is not None),
                                default=0)

        mappings = []
        projected_shapes = []
        allocation_shapes = []
        for arr, axes in zip(arrays, op_axes):
            if axes is None:
                ndim = iterator_ndim if arr is None else arr.ndim
                if ndim > iterator_ndim:
                    raise ValueError("operand has more dimensions than op_axes")
                mapping = ([-1] * (iterator_ndim - ndim)
                           + list(range(ndim)))
            else:
                mapping = []
                for value in axes:
                    axis = -1 if value is None else int(value)
                    mapping.append(axis)
            nonnegative = [axis for axis in mapping if axis >= 0]
            if len(set(nonnegative)) != len(nonnegative):
                raise ValueError("op_axes contained a duplicate axis")
            if arr is not None and any(axis >= arr.ndim for axis in nonnegative):
                raise ValueError("op_axes axis is out of bounds")
            if arr is None and nonnegative:
                expected = list(range(max(nonnegative) + 1))
                if sorted(nonnegative) != expected:
                    raise ValueError("allocated op_axes must specify every output axis")

            shape = tuple(1 if axis < 0 else int(arr.shape[axis])
                          for axis in mapping) if arr is not None else (1,) * iterator_ndim
            projected_shapes.append(shape)
            mappings.append(tuple(mapping))
            if nonnegative:
                raw_shape = [1] * (max(nonnegative) + 1)
                for logical_axis, operand_axis in enumerate(mapping):
                    if operand_axis >= 0:
                        raw_shape[operand_axis] = shape[logical_axis]
                allocation_shapes.append(tuple(raw_shape))
            else:
                allocation_shapes.append(())

        logical_shape = _broadcast_shape(projected_shapes)
        if itershape is not None:
            requested = tuple(int(x) for x in itershape)
            if len(requested) != iterator_ndim or any(x < -1 for x in requested):
                raise ValueError("itershape must match the iterator dimensions")
            result = []
            for inferred, asked in zip(logical_shape, requested):
                if asked == -1:
                    result.append(inferred)
                elif inferred not in (1, asked):
                    raise ValueError("operands could not be broadcast to itershape")
                else:
                    result.append(asked)
            logical_shape = tuple(result)

        self._axis_mappings = tuple(mappings)
        final_allocation_shapes = []
        for mapping in mappings:
            axes = [axis for axis in mapping if axis >= 0]
            raw_shape = [1] * (max(axes) + 1) if axes else []
            for logical_axis, operand_axis in enumerate(mapping):
                if operand_axis >= 0:
                    raw_shape[operand_axis] = logical_shape[logical_axis]
            final_allocation_shapes.append(tuple(raw_shape))
        self._allocation_shapes = tuple(final_allocation_shapes)

        for mapping, mode in zip(mappings, self._op_flags):
            if "readonly" in mode:
                continue
            reduces = any(axis < 0 and logical_shape[logical_axis] != 1
                          for logical_axis, axis in enumerate(mapping))
            if reduces and "reduce_ok" not in self._flags:
                raise ValueError("output operand requires a reduction, but REDUCE_OK was not enabled")
        remapped = [None if arr is None else self._remap_array(arr, mapping)
                    for arr, mapping in zip(arrays, mappings)]
        return remapped, logical_shape

    @staticmethod
    def _remap_array(arr, mapping):
        used = [axis for axis in mapping if axis >= 0]
        omitted = [axis for axis in range(arr.ndim) if axis not in used]
        if omitted:
            index = tuple(0 if axis in omitted else slice(None)
                          for axis in range(arr.ndim))
            view = arr[index]
            remaining = [axis for axis in range(arr.ndim) if axis in used]
        else:
            view = arr
            remaining = list(range(arr.ndim))
        if used:
            permutation = tuple(remaining.index(axis) for axis in used)
            if permutation != tuple(range(len(permutation))):
                view = view.transpose(permutation)
        if any(axis < 0 for axis in mapping):
            view = view[tuple(None if axis < 0 else slice(None)
                              for axis in mapping)]
        return view

    def _prepare_temporary_operands(self, arrays, can_cast):
        prepared = []
        for arr, mode, requested, was_scalar in zip(
                arrays, self._op_flags, self._requested_dtypes,
                self._scalar_operands):
            target = requested
            if target is None and "nbo" in mode:
                target = arr.dtype.newbyteorder("=")
            if target is None or target == arr.dtype:
                prepared.append(arr)
                continue
            if (target.hasobject or target.kind == "O") \
                    and "refs_ok" not in self._flags:
                raise TypeError(
                    "Iterator requested dtype holds references, but the "
                    "REFS_OK flag was not enabled"
                )

            reading = "readonly" in mode or "readwrite" in mode
            writing = "writeonly" in mode or "readwrite" in mode
            if reading and not can_cast(arr.dtype, target, self._casting):
                raise TypeError(
                    f"Iterator operand required copying from {arr.dtype!r} to "
                    f"{target!r} according to the rule {self._casting!r}"
                )
            if writing and not can_cast(target, arr.dtype, self._casting):
                raise TypeError(
                    f"Iterator requested dtype could not be cast back from "
                    f"{target!r} to {arr.dtype!r} according to the rule "
                    f"{self._casting!r}"
                )
            allows_temporary = (
                "buffered" in self._flags or "copy" in mode
                or "updateifcopy" in mode or "writebackifcopy" in mode
                or (was_scalar and "readonly" in mode)
            )
            if not allows_temporary:
                raise TypeError("Iterator operand required copying or buffering")

            temporary = arr.astype(target, order="K", casting="unsafe")
            prepared.append(temporary)
            if writing:
                self._writebacks.append((arr, temporary))
                if "updateifcopy" in mode or "writebackifcopy" in mode:
                    arr.flags.writeable = False
        return prepared

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
            return (list(range(ndim)), [False] * ndim) if use_f else (
                list(range(ndim - 1, -1, -1)), [False] * ndim)

        def key(axis):
            for arr in self._operands:
                stride = self._mapped_stride(arr, axis)
                if stride != 0:
                    return (abs(stride), -axis)
            return (1 << 62, -axis)

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
            if axis in self._removed_axes:
                continue
            dim = self._logical_shape[axis]
            digit = value % dim if dim else 0
            value //= max(dim, 1)
            coord[axis] = dim - 1 - digit if self._axis_reverse[axis] else digit
        return tuple(coord)

    def _pos_for_coord(self, coord):
        value = 0
        multiplier = 1
        for axis in self._axis_fast:
            if axis in self._removed_axes:
                continue
            digit = coord[axis]
            if self._axis_reverse[axis]:
                digit = self._logical_shape[axis] - 1 - digit
            value += digit * multiplier
            multiplier *= self._logical_shape[axis]
        return value

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
            view = arr[tuple(slice(i, i + 1) for i in coord)].reshape(())
        if not writeable:
            view.flags.writeable = False
        return view

    def _external_chunk_len(self):
        remaining = self._stop - self._pos
        if remaining <= 0:
            return 0
        if "buffered" in self._flags and self._buffersize > 0:
            return min(remaining, self._buffersize)
        groups = self._coalesced_groups()
        return min(remaining, groups[0] if groups else 1)

    def _coalesced_groups(self):
        axes = [axis for axis in self._axis_fast
                if axis not in self._removed_axes
                and self._logical_shape[axis] != 1]
        if not axes:
            return []
        groups = [self._logical_shape[axes[0]]]
        group_fast_axis = axes[0]
        group_size = groups[0]
        for axis in axes[1:]:
            merge = True
            for arr in self._operands:
                fast_stride = abs(self._mapped_stride(arr, group_fast_axis))
                slow_stride = abs(self._mapped_stride(arr, axis))
                if fast_stride == slow_stride == 0:
                    continue
                if fast_stride == 0 or slow_stride != fast_stride * group_size:
                    merge = False
                    break
            if merge:
                group_size *= self._logical_shape[axis]
                groups[-1] = group_size
            else:
                group_fast_axis = axis
                group_size = self._logical_shape[axis]
                groups.append(group_size)
        return groups

    def _make_external_chunks(self):
        from . import array

        n = self._external_chunk_len()
        chunks = []
        for arr, mode in zip(self._operands, self._op_flags):
            if ("readonly" not in mode and arr.size == 1
                    and all(self._mapped_stride(arr, axis) == 0
                            for axis in range(len(self._logical_shape)))):
                from .lib.stride_tricks import as_strided
                chunks.append(as_strided(arr.reshape(()), shape=(n,),
                                         strides=(0,), writeable=True))
                continue
            if (len(self._logical_shape) == 1 and arr.ndim == 1
                    and tuple(arr.shape) == self._logical_shape
                    and not self._axis_reverse[0]):
                chunk = arr[self._pos:self._pos + n]
                if "readonly" in mode:
                    chunk.flags.writeable = False
                chunks.append(chunk)
                continue
            values = []
            for offset in range(n):
                coord = self._operand_coord(arr, self._coord_at(self._pos + offset))
                values.append(arr[coord])
            chunk = array(values, dtype=arr.dtype)
            if "readonly" in mode:
                chunk.flags.writeable = False
            chunks.append(chunk)
        self._chunk_start = self._pos
        self._chunk_len = n
        self._chunk_cache = tuple(chunks)

    def _flush_external(self):
        if self._chunk_cache is None:
            return
        for arr, mode, chunk in zip(self._operands, self._op_flags,
                                    self._chunk_cache):
            if "readonly" in mode or getattr(chunk, "base", None) is not None:
                continue
            for offset in range(self._chunk_len):
                coord = self._operand_coord(
                    arr, self._coord_at(self._chunk_start + offset))
                arr[coord] = chunk[offset]
        self._chunk_cache = None

    def _flush_buffered_writebacks(self):
        if "buffered" in self._flags:
            for original, temporary in self._writebacks:
                original[...] = temporary

    def _value_for_operand(self, operand):
        if self.finished:
            raise ValueError("Iterator is past the end")
        if "external_loop" in self._flags:
            if self._chunk_cache is None:
                self._make_external_chunks()
            return self._chunk_cache[operand]
        arr = self._operands[operand]
        coord = self._operand_coord(arr, self._coord_at(self._pos))
        return self._cell_view(arr, coord,
                               "readonly" not in self._op_flags[operand])

    def _check_open(self):
        if self._closed:
            raise ValueError("Iterator is invalid")

    @property
    def operands(self):
        self._check_open()
        return self._public_operands

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
        if "multi_index" in self._flags:
            return len(self._active_shape())
        return len(self._coalesced_groups())

    @property
    def shape(self):
        if self._multi_index_removed:
            raise ValueError("Iterator has no shape")
        if "multi_index" in self._flags:
            return self._active_shape()
        if not self._active_shape():
            return ()
        return tuple(reversed(self._coalesced_groups()))

    def _active_shape(self):
        return tuple(dim for axis, dim in enumerate(self._logical_shape)
                     if axis not in self._removed_axes)

    @property
    def finished(self):
        return self._closed or self._pos >= self._stop

    @property
    def iterindex(self):
        if not self._logical_shape:
            return self._start
        return self._pos

    @iterindex.setter
    def iterindex(self, value):
        if "external_loop" in self._flags or "buffered" in self._flags:
            raise ValueError("Cannot jump to an iterator index with buffering")
        value = int(value)
        if value < self._start or value > self._stop:
            raise ValueError("Iterator index out of bounds")
        self._flush_external()
        self._pos = value

    @property
    def iterrange(self):
        return (self._start, self._stop)

    @iterrange.setter
    def iterrange(self, value):
        if "ranged" not in self._flags:
            raise ValueError("Iterator was not created with the RANGED flag")
        start, stop = (int(x) for x in value)
        total = _shape_size(self._active_shape())
        if start < 0 or stop < start or stop > total:
            raise ValueError("Iterator range is out of bounds")
        self._flush_external()
        self._start, self._stop, self._pos = start, stop, start

    @property
    def multi_index(self):
        if "multi_index" not in self._flags:
            raise ValueError("Iterator is not tracking a multi-index")
        return self._coord_at(min(self._pos, max(self._stop - 1, 0)))

    @multi_index.setter
    def multi_index(self, value):
        if "multi_index" not in self._flags or "external_loop" in self._flags \
                or "buffered" in self._flags:
            raise ValueError("Iterator is not tracking a writable multi-index")
        coord = tuple(int(x) for x in value)
        if len(coord) != len(self._logical_shape):
            raise ValueError("Wrong number of indices")
        if any(x < 0 or x >= dim for x, dim in zip(coord, self._logical_shape)):
            raise ValueError("Iterator multi-index is out of bounds")
        self._pos = self._pos_for_coord(coord)

    def _flat_index(self, coord, fortran):
        value = 0
        multiplier = 1
        axes = (range(len(self._logical_shape)) if fortran else
                range(len(self._logical_shape) - 1, -1, -1))
        for axis in axes:
            value += coord[axis] * multiplier
            multiplier *= self._logical_shape[axis]
        return value

    def _coord_from_flat(self, value, fortran):
        size = _shape_size(self._logical_shape)
        if value < 0 or value >= size:
            raise ValueError("Iterator index is out of bounds")
        coord = [0] * len(self._logical_shape)
        axes = (range(len(coord)) if fortran else
                range(len(coord) - 1, -1, -1))
        for axis in axes:
            dim = self._logical_shape[axis]
            coord[axis] = value % dim
            value //= dim
        return tuple(coord)

    @property
    def index(self):
        if not self._flags & {"c_index", "f_index"}:
            raise ValueError("Iterator does not have an index")
        coord = self._coord_at(min(self._pos, max(self._stop - 1, 0)))
        return self._flat_index(coord, "f_index" in self._flags)

    @index.setter
    def index(self, value):
        if not self._flags & {"c_index", "f_index"} \
                or "external_loop" in self._flags or "buffered" in self._flags:
            raise ValueError("Iterator does not have a writable index")
        coord = self._coord_from_flat(int(value), "f_index" in self._flags)
        self._pos = self._pos_for_coord(coord)

    @property
    def value(self):
        values = tuple(self._value_for_operand(i) for i in range(self.nop))
        return values[0] if self.nop == 1 else values

    @property
    def iterationneedsapi(self):
        return any(a.dtype.hasobject for a in self._operands)

    @property
    def has_delayed_bufalloc(self):
        return "delay_bufalloc" in self._flags and self._pos == self._start

    @property
    def itviews(self):
        from . import broadcast_to
        if self._multi_index_removed:
            return tuple(broadcast_to(a, self._logical_shape).reshape(-1)
                         for a in self._operands)
        return tuple(broadcast_to(a, self._logical_shape) for a in self._operands)

    def __len__(self):
        return self.itersize

    def __iter__(self):
        return self

    def __next__(self):
        self._flush_external()
        self._flush_buffered_writebacks()
        if self.finished:
            raise StopIteration
        value = self.value
        self._pos += self._chunk_len if "external_loop" in self._flags else 1
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

    def __delitem__(self, key):
        raise TypeError("iterator elements cannot be deleted")

    def iternext(self):
        self._flush_external()
        self._flush_buffered_writebacks()
        if not self.finished:
            self._pos += self._chunk_len if "external_loop" in self._flags else 1
        return not self.finished

    def reset(self):
        self._check_open()
        self._flush_external()
        self._flush_buffered_writebacks()
        self._pos = self._start

    def remove_axis(self, axis):
        if "multi_index" not in self._flags:
            raise ValueError("Iterator is not tracking a multi-index")
        axis = int(axis)
        ndim = len(self._logical_shape)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError("axis is out of bounds")
        self._removed_axes.add(axis)
        self._stop = self._start + _shape_size(self._active_shape())
        self._pos = self._start

    def remove_multi_index(self):
        if "multi_index" not in self._flags:
            raise ValueError("Iterator is not tracking a multi-index")
        self._flags.remove("multi_index")
        self._multi_index_removed = True
        self._pos = self._start

    def enable_external_loop(self):
        if self._flags & {"multi_index", "c_index", "f_index"}:
            raise ValueError(
                "Iterator flag EXTERNAL_LOOP cannot be used if an index or "
                "multi-index is being tracked"
            )
        self._flags.add("external_loop")
        self._chunk_cache = None

    def copy(self):
        result = _copy.copy(self)
        result._flags = set(self._flags)
        result._removed_axes = set(self._removed_axes)
        result._chunk_cache = None
        result._writebacks = list(self._writebacks)
        return result

    def close(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError("close() takes no arguments")
        if self._closed:
            return
        self._flush_external()
        for original, temporary in self._writebacks:
            try:
                original.flags.writeable = True
            except Exception:
                pass
            original[...] = temporary
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot enter a closed iterator")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def nested_iters(*args, **kwargs):
    raise NotImplementedError("numpy.nested_iters is not implemented yet")
