"""The unoptimized ``c_einsum`` execution contract, implemented on rnp arrays.

The public parser and contraction planner live in :mod:`einsumfunc` and are a
near-verbatim port of NumPy 2.5.2.  This module owns the part NumPy implements
in ``multiarray/einsum.cpp``: diagonal views, common-dtype selection, and the
ordered sum-of-products loop.
"""

import itertools

import rnp_numpy as np


_ORDERS = {"C", "F", "A", "K"}
_CASTINGS = {"no", "equiv", "safe", "same_kind", "unsafe"}
_MISSING = object()


def _combined_view(term, arr):
    """Combine repeated labels as diagonal axes, preserving a view."""
    while len(set(term)) != len(term):
        for label in term:
            if term.count(label) > 1:
                axis1 = term.index(label)
                axis2 = term.index(label, axis1 + 1)
                if arr.shape[axis1] != arr.shape[axis2]:
                    raise ValueError(
                        f"dimensions in operand 0 for collapsing index "
                        f"{label!r} don't match ({arr.shape[axis1]} != "
                        f"{arr.shape[axis2]})"
                    )
                arr = np.diagonal(arr, 0, axis1, axis2)
                term = "".join(
                    ch for i, ch in enumerate(term)
                    if i not in (axis1, axis2)
                ) + label
                break
    return term, arr


def _single_operand_view(term, output, arr):
    """Return NumPy's no-computation diagonal/transpose view when possible."""
    term, arr = _combined_view(term, arr)
    if set(term) != set(output) or len(term) != len(output):
        return None
    axes = tuple(term.index(label) for label in output)
    return arr.transpose(axes)


def _dimension_dict(terms, arrays):
    sizes = {}
    for operand_num, (term, arr) in enumerate(zip(terms, arrays)):
        if len(term) != arr.ndim:
            raise ValueError(
                f"Einstein sum subscript {term} does not contain the correct "
                f"number of indices for operand {operand_num}."
            )
        local = {}
        for axis, (label, size) in enumerate(zip(term, arr.shape)):
            if label in local and local[label] != size:
                raise ValueError(
                    f"dimensions in operand {operand_num} for collapsing "
                    f"index {label!r} don't match ({local[label]} != {size})"
                )
            local[label] = size
            previous = sizes.get(label)
            if previous is None or previous == 1:
                sizes[label] = size
            elif size not in (1, previous):
                raise ValueError(
                    f"Size of label {label!r} for operand {operand_num} "
                    f"({size}) does not match previous terms ({previous})."
                )
    return sizes


def _can_cast(src, dst, casting):
    try:
        return bool(np.can_cast(src, dst, casting=casting))
    except TypeError:
        # ``same_value`` is value-sensitive and is handled by astype below.
        return casting == "same_value"


def _coerce_operands(arrays, result_dtype, casting):
    coerced = []
    for arr in arrays:
        if arr.dtype == result_dtype:
            coerced.append(arr)
            continue
        if not _can_cast(arr.dtype, result_dtype, casting):
            raise TypeError(
                f"Iterator operand 0 dtype could not be cast from "
                f"{arr.dtype!r} to {result_dtype!r} according to the rule "
                f"{casting!r}"
            )
        coerced.append(arr.astype(result_dtype))
    return coerced


def _allocate(shape, dtype, order, arrays, terms, output):
    if order == "A":
        order = "F" if all(a.flags.f_contiguous for a in arrays) else "C"
    if order in ("C", "F"):
        return np.zeros(shape, dtype=dtype, order=order)
    # NpyIter's KEEPORDER chooses axes from the smallest available operand
    # strides. Missing axes do not constrain the comparison; ties prefer the
    # later iterator axis.
    scores = []
    for output_axis, label in enumerate(output):
        strides = [
            abs(arr.strides[term.index(label)])
            for term, arr in zip(terms, arrays) if label in term
            if arr.strides[term.index(label)] != 0
        ]
        scores.append((min(strides) if strides else 0, -output_axis,
                       output_axis))
    fast_axes = [item[2] for item in sorted(scores)]
    slow_axes = list(reversed(fast_axes))
    base = np.zeros(tuple(shape[axis] for axis in slow_axes), dtype=dtype)
    return base.transpose(tuple(slow_axes.index(axis)
                                for axis in range(len(shape))))


def _iter_indices(shape):
    if not shape:
        yield ()
        return
    yield from itertools.product(*(range(n) for n in shape))


def _sum_of_products(terms, output, arrays, sizes, result):
    """Run the C core's left-to-right product and scalar accumulation order."""
    reduction = "".join(sorted(set("".join(terms)) - set(output)))
    labels = output + reduction
    shape = tuple(sizes[label] for label in labels)
    output_ndim = len(output)

    # Convert each term to label positions once; singleton axes use index 0.
    operand_maps = []
    for term, arr in zip(terms, arrays):
        positions = tuple(labels.index(label) for label in term)
        operand_maps.append((arr, positions))

    if result.dtype.kind != "O":
        from _rnp import _einsum_numeric
        _einsum_numeric(arrays, [positions for _, positions in operand_maps],
                        shape, output_ndim, result)
        return

    for index in _iter_indices(shape):
        product = _MISSING
        for arr, positions in operand_maps:
            arr_index = tuple(
                0 if arr.shape[axis] == 1 else index[position]
                for axis, position in enumerate(positions)
            )
            value = arr[arr_index]
            product = value if product is _MISSING else product * value
        out_index = index[:output_ndim]
        result[out_index] = result[out_index] + product


def c_einsum(*operands, out=None, dtype=None, order="K", casting="safe"):
    """Evaluate one unoptimized Einstein sum with NumPy-compatible parsing."""
    from .einsumfunc import _parse_einsum_input

    order = str(order).upper()
    if order not in _ORDERS:
        raise ValueError(
            f"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')"
        )
    if casting not in _CASTINGS:
        raise ValueError(
            "casting must be one of 'no', 'equiv', 'safe', 'same_kind', "
            f"or 'unsafe' (got '{casting}')"
        )
    if out is not None and not isinstance(out, np.ndarray):
        raise TypeError("keyword parameter out must be an array")

    input_subscripts, output, arrays = _parse_einsum_input(operands)
    terms = input_subscripts.split(",")
    sizes = _dimension_dict(terms, arrays)

    result_dtype = np.dtype(dtype) if dtype is not None else np.result_type(*arrays)
    arrays = _coerce_operands(arrays, result_dtype, casting)

    # Exactly NumPy's early get_single_op_view path.  A dtype conversion or
    # explicit output necessarily turns this into a computation/copy.
    if len(arrays) == 1 and out is None and arrays[0].dtype == result_dtype:
        view = _single_operand_view(terms[0], output, arrays[0])
        if view is not None:
            return view

    combined = [_combined_view(term, arr) for term, arr in zip(terms, arrays)]
    terms = [item[0] for item in combined]
    arrays = [item[1] for item in combined]

    output_shape = tuple(sizes[label] for label in output)
    if out is not None:
        if out.shape != output_shape:
            raise ValueError(
                f"out parameter does not have the correct dimensions, "
                f"has {out.ndim} but should have {len(output_shape)}"
            )
        if not _can_cast(result_dtype, out.dtype, casting):
            raise TypeError(
                f"Iterator requested dtype could not be cast to output dtype "
                f"according to the rule {casting!r}"
            )

    # Always compute into a fresh buffer.  This matches COPY_IF_OVERLAP when
    # ``out`` aliases an operand and keeps zero-initialization deterministic.
    result = _allocate(output_shape, result_dtype, order, arrays, terms, output)
    if result.dtype.kind == "O":
        result[...] = 0
    _sum_of_products(terms, output, arrays, sizes, result)

    if out is not None:
        # The object assignment loop must receive one scalar at a time.  In
        # particular, assigning a numeric ndarray wholesale to an object
        # ndarray would otherwise store the entire RHS array in every slot.
        if out.dtype.kind == "O" and result.dtype.kind != "O":
            for index in _iter_indices(output_shape):
                out[index] = result[index]
        else:
            out[...] = result
        return out
    if result.ndim == 0:
        return result[()]
    return result
