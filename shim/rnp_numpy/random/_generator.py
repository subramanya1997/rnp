"""Bit-exact high-level Generator operations."""

import bisect
from collections.abc import Sequence
import itertools
import math
import operator
import warnings

from .. import (
    arange,
    array,
    asarray,
    broadcast_to,
    can_cast,
    dtype as _dtype,
    empty,
    float32,
    float64,
    int64,
    ndarray,
    uint32,
)
from .._arraycompat import broadcast_arrays
from ..exceptions import AxisError
from ._bit_generators import BitGenerator
from ._distributions import DistributionKernels


_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1


def _shape(size):
    if size is None:
        return ()
    try:
        shape = (operator.index(size),)
    except TypeError:
        shape = tuple(operator.index(v) for v in size)
    if any(v < 0 for v in shape):
        raise ValueError("negative dimensions are not allowed")
    return shape


def _count(shape):
    count = 1
    for dim in shape:
        count *= dim
    return count


def _dtype_name(value):
    return str(_dtype(value))


def _gen_mask(maximum):
    mask = maximum
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16
    mask |= mask >> 32
    return mask


def _indices(shape):
    """Yield C-order index tuples without depending on ndarray iteration."""
    if not shape:
        yield ()
        return
    yield from itertools.product(*(range(dim) for dim in shape))


def _flat_indices(shape, fortran=False):
    """Yield flat-iteration coordinates in C or Fortran memory order."""
    if not fortran:
        yield from _indices(shape)
        return
    for reversed_index in _indices(tuple(reversed(shape))):
        yield tuple(reversed(reversed_index))


class Generator:
    _poisson_lam_max = (1 << 63) - 1 - math.sqrt((1 << 63) - 1) * 10
    _distribution_names = {
        "beta", "binomial", "chisquare", "dirichlet", "exponential", "f",
        "gamma", "geometric", "gumbel", "hypergeometric", "laplace",
        "logistic", "lognormal", "logseries", "multinomial",
        "multivariate_hypergeometric", "multivariate_normal",
        "negative_binomial", "noncentral_chisquare", "noncentral_f", "normal",
        "pareto", "permuted", "poisson", "power", "rayleigh",
        "standard_cauchy", "standard_exponential", "standard_gamma",
        "standard_normal", "standard_t", "triangular", "vonmises", "wald",
        "weibull", "zipf",
    }

    def __init__(self, bit_generator):
        if isinstance(bit_generator, type) and issubclass(bit_generator, BitGenerator):
            raise ValueError("Generator requires a BitGenerator instance")
        if not isinstance(bit_generator, BitGenerator):
            raise TypeError("Generator requires a BitGenerator instance")
        self._bit_generator = bit_generator
        self._distributions = DistributionKernels(bit_generator)

    def __getattr__(self, name):
        # Keep the complete public surface collectable while distribution
        # algorithms are landed in the next stage.
        if name in self._distribution_names:
            def pending(*args, **kwargs):
                raise NotImplementedError(f"Generator.{name} is not implemented")
            return pending
        raise AttributeError(name)

    @property
    def bit_generator(self):
        return self._bit_generator

    def spawn(self, n_children):
        return [type(self)(bg) for bg in self._bit_generator.spawn(n_children)]

    def __repr__(self):
        return (
            f"Generator({type(self._bit_generator).__name__}) at "
            f"{id(self):#X}".replace("0X", "0x")
        )

    def __str__(self):
        return f"Generator({type(self._bit_generator).__name__})"

    def __reduce__(self):
        from . import _pickle
        constructor = getattr(_pickle, "__generator_ctor")
        return constructor, (self._bit_generator,), None

    def random(self, size=None, dtype=float64, out=None):
        name = _dtype_name(dtype)
        if name == "float64":
            draw = self._bit_generator.next_double
        elif name == "float32":
            draw = lambda: (self._bit_generator.next_uint32() >> 8) * (1.0 / 16777216.0)
        else:
            raise TypeError(f"Unsupported dtype {str(_dtype(dtype))!r} for random")

        if size is None and out is None:
            value = draw()
            if name == "float32":
                return array([value], dtype=float32)[0]
            return value

        shape = tuple(out.shape) if out is not None and size is None else _shape(size)
        if out is not None:
            if tuple(out.shape) != shape or str(out.dtype) != name:
                raise ValueError("size must match out.shape when used together")
            target = out
        else:
            target = empty(shape, dtype=dtype)
        flat = target.flat
        for i in range(_count(shape)):
            flat[i] = draw()
        return target

    def _lemire32(self, inclusive_range):
        if inclusive_range == 0:
            return 0
        if inclusive_range == _MASK32:
            return self._bit_generator.next_uint32()
        bound = inclusive_range + 1
        product = self._bit_generator.next_uint32() * bound
        leftover = product & _MASK32
        if leftover < bound:
            threshold = (_MASK32 - inclusive_range) % bound
            while leftover < threshold:
                product = self._bit_generator.next_uint32() * bound
                leftover = product & _MASK32
        return product >> 32

    def _lemire64(self, inclusive_range):
        if inclusive_range == 0:
            return 0
        if inclusive_range <= _MASK32:
            return self._lemire32(inclusive_range)
        if inclusive_range == _MASK64:
            return self._bit_generator.next_uint64()
        bound = inclusive_range + 1
        product = self._bit_generator.next_uint64() * bound
        leftover = product & _MASK64
        if leftover < bound:
            threshold = (_MASK64 - inclusive_range) % bound
            while leftover < threshold:
                product = self._bit_generator.next_uint64() * bound
                leftover = product & _MASK64
        return product >> 64

    def _buffered_drawer(self, bits, inclusive_range):
        if inclusive_range == 0:
            return lambda: 0
        if bits == 32:
            return lambda: self._lemire32(inclusive_range)
        if bits == 64:
            return lambda: self._lemire64(inclusive_range)

        chunks = 32 // bits
        mask = (1 << bits) - 1
        remaining = 0
        buffer = 0

        def next_piece():
            nonlocal remaining, buffer
            if remaining == 0:
                buffer = self._bit_generator.next_uint32()
                remaining = chunks - 1
            else:
                buffer >>= bits
                remaining -= 1
            return buffer & mask

        if bits == 1:
            return next_piece
        if inclusive_range == mask:
            return next_piece
        bound = inclusive_range + 1

        def draw():
            product = next_piece() * bound
            leftover = product & mask
            if leftover < bound:
                threshold = (mask - inclusive_range) % bound
                while leftover < threshold:
                    product = next_piece() * bound
                    leftover = product & mask
            return product >> bits

        return draw

    def _small_piece_drawer(self, bits):
        chunks = 32 // bits
        mask = (1 << bits) - 1
        remaining = 0
        buffer = 0

        def next_piece():
            nonlocal remaining, buffer
            if remaining == 0:
                buffer = self._bit_generator.next_uint32()
                remaining = chunks - 1
            else:
                buffer >>= bits
                remaining -= 1
            return buffer & mask

        return next_piece

    @staticmethod
    def _bounded_small(bits, inclusive_range, next_piece):
        if inclusive_range == 0:
            return 0
        if bits == 1:
            return next_piece()
        mask = (1 << bits) - 1
        if inclusive_range == mask:
            return next_piece()
        bound = inclusive_range + 1
        product = next_piece() * bound
        leftover = product & mask
        if leftover < bound:
            threshold = (mask - inclusive_range) % bound
            while leftover < threshold:
                product = next_piece() * bound
                leftover = product & mask
        return product >> bits

    def integers(self, low, high=None, size=None, dtype=int64, endpoint=False):
        single_argument = high is None
        if high is None:
            low, high = 0, low
        name = _dtype_name(dtype)
        if name.startswith((">", "<")):
            raise ValueError(
                "Providing a dtype with a non-native byteorder is not supported. "
                "If you require platform-independent byteorder, call byteswap when required."
            )
        info = {
            "bool": (1, 0, 1),
            "int8": (8, -(1 << 7), (1 << 7) - 1),
            "uint8": (8, 0, (1 << 8) - 1),
            "int16": (16, -(1 << 15), (1 << 15) - 1),
            "uint16": (16, 0, (1 << 16) - 1),
            "int32": (32, -(1 << 31), (1 << 31) - 1),
            "uint32": (32, 0, _MASK32),
            "int64": (64, -(1 << 63), (1 << 63) - 1),
            "uint64": (64, 0, _MASK64),
        }
        if name not in info:
            raise TypeError(f"Unsupported dtype {str(_dtype(dtype))!r} for integers")
        bits, dtype_low, dtype_high = info[name]

        scalar_bounds = True
        try:
            low_value = operator.index(low)
            high_value = operator.index(high)
        except TypeError:
            scalar_bounds = False

        if scalar_bounds:
            shape = _shape(size)
            if size is not None and _count(shape) == 0:
                return empty(shape, dtype=dtype)
            last = high_value if endpoint else high_value - 1
            if low_value < dtype_low:
                raise ValueError(f"low is out of bounds for {name}")
            if last > dtype_high:
                raise ValueError(f"high is out of bounds for {name}")
            if last < low_value:
                if single_argument:
                    raise ValueError("high <= 0" if not endpoint else "high < 0")
                raise ValueError("low >= high" if not endpoint else "low > high")
            count = 1 if size is None else _count(shape)
            draw = self._buffered_drawer(bits, last - low_value)
            values = [low_value + draw() for _ in range(count)]
        else:
            low_arr, high_arr = broadcast_arrays(asarray(low), asarray(high))
            shape = tuple(low_arr.shape) if size is None else _shape(size)
            low_arr = broadcast_to(low_arr, shape)
            high_arr = broadcast_to(high_arr, shape)
            values = []
            next_piece = self._small_piece_drawer(bits) if bits < 32 else None
            # NumPy keeps one small-dtype bit buffer for the whole fill, but
            # range-dependent Lemire parameters can change per element.
            for lo, hi in zip(low_arr.flat, high_arr.flat):
                lo, hi = operator.index(lo), operator.index(hi)
                last = hi if endpoint else hi - 1
                if lo < dtype_low:
                    raise ValueError(f"low is out of bounds for {name}")
                if last > dtype_high:
                    raise ValueError(f"high is out of bounds for {name}")
                if last < lo:
                    if single_argument:
                        raise ValueError("high <= 0" if not endpoint else "high < 0")
                    raise ValueError("low >= high" if not endpoint else "low > high")
                inclusive_range = last - lo
                if bits < 32:
                    value = self._bounded_small(bits, inclusive_range, next_piece)
                else:
                    value = self._buffered_drawer(bits, inclusive_range)()
                values.append(lo + value)

        if size is None and scalar_bounds:
            if dtype is int:
                return int(values[0])
            if dtype is bool:
                return bool(values[0])
            return array(values, dtype=dtype)[0]
        return array(values, dtype=dtype).reshape(shape)

    def bytes(self, length):
        length = operator.index(length)
        if length < 0:
            raise ValueError("negative dimensions are not allowed")
        count = (length + 3) // 4
        return b"".join(
            self._bit_generator.next_uint32().to_bytes(4, "little")
            for _ in range(count)
        )[:length]

    def _random_interval(self, maximum):
        if maximum == 0:
            return 0
        mask = _gen_mask(maximum)
        if maximum <= _MASK32:
            while True:
                value = self._bit_generator.next_uint32() & mask
                if value <= maximum:
                    return value
        while True:
            value = self._bit_generator.next_uint64() & mask
            if value <= maximum:
                return value

    def _shuffle_indices(self, values, stop=1):
        for i in range(len(values) - 1, stop - 1, -1):
            j = self._random_interval(i)
            values[i], values[j] = values[j], values[i]

    def _shuffle_bounded(self, values, stop=1):
        for i in range(len(values) - 1, stop - 1, -1):
            j = self._lemire64(i)
            values[i], values[j] = values[j], values[i]

    def shuffle(self, x, axis=0):
        if hasattr(x, "ndim"):
            values = x if isinstance(x, ndarray) else asarray(x)
            ndim = values.ndim
            if ndim == 0:
                raise TypeError("len() of unsized object")
            if not values.flags.writeable:
                raise ValueError("array is read-only")
            axis = operator.index(axis)
            if axis < 0:
                axis += ndim
            if axis < 0 or axis >= ndim:
                raise AxisError(axis, ndim=ndim)
            n = values.shape[axis]
            order = list(range(n))
            for i in range(n - 1, 0, -1):
                j = self._random_interval(i)
                order[i], order[j] = order[j], order[i]
            if n:
                if values.dtype.names:
                    for field_name in values.dtype.names:
                        field = values[field_name]
                        original = field.copy()
                        for index in _indices(tuple(field.shape)):
                            base = index[:values.ndim]
                            source_base = (base[:axis]
                                           + (order[base[axis]],)
                                           + base[axis + 1:])
                            source = source_base + index[values.ndim:]
                            field[index] = original[source]
                else:
                    original = values.copy()
                    for index in _indices(tuple(values.shape)):
                        source = (index[:axis] + (order[index[axis]],)
                                  + index[axis + 1:])
                        values[index] = original[source]
            return None
        if axis != 0:
            raise NotImplementedError("Axis argument is only supported on ndarray objects")
        if not isinstance(x, Sequence):
            warnings.warn(
                f"you are shuffling a '{type(x).__name__}' object which is "
                "not a subclass of 'Sequence'; `shuffle` is not guaranteed "
                "to behave correctly.",
                UserWarning,
                stacklevel=2,
            )
        for i in range(len(x) - 1, 0, -1):
            j = self._random_interval(i)
            x[i], x[j] = x[j], x[i]
        return None

    def permuted(self, x, *, axis=None, out=None):
        x = asarray(x)
        if out is None:
            out = x.copy(order="K")
        else:
            if not isinstance(out, ndarray):
                raise TypeError("out must be a numpy array")
            if not out.flags.writeable:
                raise ValueError("array is read-only")
            if tuple(out.shape) != tuple(x.shape):
                raise ValueError("out must have the same shape as x")
            if not can_cast(x.dtype, out.dtype, casting="safe"):
                raise TypeError(
                    f"Cannot cast array data from dtype('{x.dtype}') to "
                    f"dtype('{out.dtype}') according to the rule 'safe'"
                )
            if out is not x:
                # ``out`` may overlap ``x`` (for example reversed views).
                # Preserve all source values before writing any destination.
                x = x.copy(order="K")
                for index in _indices(tuple(x.shape)):
                    out[index] = x[index]

        if axis is None:
            coordinates = list(_flat_indices(
                tuple(out.shape),
                fortran=out.flags.f_contiguous and not out.flags.c_contiguous,
            ))
            for i in range(len(coordinates) - 1, 0, -1):
                j = self._random_interval(i)
                left, right = coordinates[i], coordinates[j]
                temporary = out[left]
                out[left] = out[right]
                out[right] = temporary
            return out

        ndim = out.ndim
        axis = operator.index(axis)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise AxisError(axis, ndim=ndim)
        other_shape = tuple(out.shape[:axis]) + tuple(out.shape[axis + 1:])
        for other in _indices(other_shape):
            for i in range(out.shape[axis] - 1, 0, -1):
                j = self._random_interval(i)
                left = other[:axis] + (i,) + other[axis:]
                right = other[:axis] + (j,) + other[axis:]
                temporary = out[left]
                out[left] = out[right]
                out[right] = temporary
        return out

    def permutation(self, x, axis=0):
        try:
            n = operator.index(x)
        except TypeError:
            result = asarray(x).copy()
        else:
            result = arange(n)
        if result.ndim == 0:
            raise AxisError(axis, ndim=0)
        self.shuffle(result, axis=axis)
        return result

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        original = a
        source = asarray(a)
        if source.ndim == 0:
            try:
                pop_size = operator.index(source.item())
            except TypeError as exc:
                raise ValueError(
                    "a must be a sequence or an integer, "
                    f"not {type(original)}"
                ) from exc
            source = None
        else:
            axis = operator.index(axis)
            if axis < 0:
                axis += source.ndim
            if axis < 0 or axis >= source.ndim:
                raise IndexError("axis is out of bounds")
            pop_size = source.shape[axis]

        shape = () if size is None else _shape(size)
        count = 1 if size is None else _count(shape)
        if pop_size <= 0 and count:
            raise ValueError(
                "a must be a positive integer unless no samples are taken"
                if source is None else
                "a cannot be empty unless no samples are taken"
            )

        weights = None
        if p is not None:
            p_array = asarray(p)
            if p_array.ndim != 1:
                raise ValueError("p must be 1-dimensional")
            try:
                weights = [float(v) for v in p_array.flat]
            except (TypeError, ValueError):
                raise ValueError("Probabilities contain NaN") from None
            if len(weights) != pop_size:
                raise ValueError("a and p must have same size")
            if any(math.isnan(v) for v in weights):
                raise ValueError("Probabilities contain NaN")
            if any(v < 0 for v in weights):
                raise ValueError("Probabilities are not non-negative")
            total = math.fsum(weights)
            eps = {
                "float16": 0.0009765625,
                "float32": 1.1920928955078125e-07,
                "float64": 2.220446049250313e-16,
            }.get(str(p_array.dtype), 2.220446049250313e-16)
            if abs(total - 1.0) > math.sqrt(eps):
                raise ValueError("Probabilities do not sum to 1. See Notes section of docstring for more information.")

        if replace:
            if weights is None:
                indices = [int(self.integers(0, pop_size)) for _ in range(count)]
            else:
                cdf = []
                running = 0.0
                for weight in weights:
                    running += weight
                    cdf.append(running)
                cdf = [v / cdf[-1] for v in cdf]
                indices = [bisect.bisect_right(cdf, self.random()) for _ in range(count)]
        else:
            if count > pop_size:
                raise ValueError("Cannot take a larger sample than population when replace is False")
            if weights is not None:
                if sum(v > 0 for v in weights) < count:
                    raise ValueError("Fewer non-zero entries in p than size")
                indices = []
                active = list(weights)
                while len(indices) < count:
                    needed = count - len(indices)
                    for found in indices:
                        active[found] = 0.0
                    cdf = []
                    running = 0.0
                    for weight in active:
                        running += weight
                        cdf.append(running)
                    cdf = [v / cdf[-1] for v in cdf]
                    batch = [bisect.bisect_right(cdf, self.random()) for _ in range(needed)]
                    seen = set()
                    for value in batch:
                        if value not in seen:
                            seen.add(value)
                            indices.append(value)
            elif pop_size > 10000 and count > pop_size // (50 if shuffle else 20):
                indices = list(range(pop_size))
                self._shuffle_bounded(indices, max(pop_size - count, 1))
                indices = indices[-count:]
            else:
                selected = set()
                indices = []
                for j in range(pop_size - count, pop_size):
                    value = self._lemire64(j)
                    if value in selected:
                        selected.add(j)
                        indices.append(j)
                    else:
                        selected.add(value)
                        indices.append(value)
                if shuffle:
                    self._shuffle_bounded(indices)

        if size is None:
            index = indices[0]
            if source is None:
                return int(index)
            result = source.take(index, axis=axis)
            return result[()] if result.ndim == 0 else result
        index_array = array(indices, dtype=int64).reshape(shape)
        if source is None:
            return index_array
        return source.take(index_array, axis=axis)

    def uniform(self, low=0.0, high=1.0, size=None):
        def validate(values):
            span = values[1] - values[0]
            if not math.isfinite(span):
                raise OverflowError("high - low range exceeds valid bounds")
            if span < 0.0:
                raise ValueError("high - low < 0")

        return self._broadcast_kernel(
            (low, high), size, validate,
            lambda v: math.fma(v[1] - v[0], self.random(), v[0]),
        )

    def _broadcast_kernel(self, params, size, validate, draw, dtype=float64):
        arrays = [asarray(value) for value in params]
        scalar = all(value.ndim == 0 for value in arrays)
        if scalar:
            scalar_values = tuple(float(value) for value in params)
            validate(scalar_values)
            if size is None:
                return draw(scalar_values)
            shape = _shape(size)
            return array([draw(scalar_values) for _ in range(_count(shape))],
                         dtype=dtype).reshape(shape)
        arrays = list(broadcast_arrays(*arrays))
        shape = tuple(arrays[0].shape) if size is None else _shape(size)
        try:
            arrays = [broadcast_to(value, shape) for value in arrays]
        except ValueError:
            raise ValueError(
                f"Output size {shape} is not compatible with broadcast "
                "dimensions of inputs."
            ) from None
        rows = [tuple(float(value.flat[i]) for value in arrays)
                for i in range(_count(shape))]
        for values in rows:
            validate(values)
        results = [draw(values) for values in rows]
        if scalar and size is None:
            return results[0]
        return array(results, dtype=dtype).reshape(shape)

    def _continuous(self, draw, size=None, dtype=float64, out=None):
        name = _dtype_name(dtype)
        if name not in ("float64", "float32"):
            raise TypeError(f"Unsupported dtype {str(_dtype(dtype))!r} for distribution")
        if size is None and out is None:
            return draw()
        shape = tuple(out.shape) if out is not None and size is None else _shape(size)
        if out is not None:
            if str(out.dtype) != name:
                raise TypeError(
                    f"Supplied output array has the wrong type. Expected {name}, "
                    f"got {out.dtype}"
                )
            if tuple(out.shape) != shape:
                raise ValueError("size must match out.shape when used together")
            if not (out.flags.c_contiguous or out.flags.f_contiguous):
                raise ValueError("Supplied output array must be contiguous, writable, aligned, and in machine byte-order.")
            if not out.flags.writeable:
                raise ValueError("Supplied output array must be contiguous, writable, aligned, and in machine byte-order.")
            target = out
        else:
            target = empty(shape, dtype=dtype)
        for i in range(_count(shape)):
            target.flat[i] = draw()
        return target

    def standard_normal(self, size=None, dtype=float64, out=None):
        draw = (self._distributions.standard_normal_f
                if _dtype_name(dtype) == "float32"
                else self._distributions.standard_normal)
        return self._continuous(
            draw, size=size, dtype=dtype, out=out
        )

    def standard_exponential(self, size=None, dtype=float64, method="zig", out=None):
        if method not in ("zig", "inv"):
            raise ValueError(f"Method {method} is not supported")
        if method == "inv":
            if _dtype_name(dtype) == "float32":
                draw = lambda: -math.log1p(
                    -((self._bit_generator.next_uint32() >> 8)
                      * (1.0 / 16777216.0))
                )
            else:
                draw = lambda: -math.log1p(-self._bit_generator.next_double())
        else:
            draw = (self._distributions.standard_exponential_f
                    if _dtype_name(dtype) == "float32"
                    else self._distributions.standard_exponential)
        return self._continuous(draw, size=size, dtype=dtype, out=out)

    def standard_gamma(self, shape, size=None, dtype=float64, out=None):
        shape_arr = asarray(shape)
        if shape_arr.ndim != 0:
            def validate(values):
                value = values[0]
                if (math.isnan(value) or value < 0.0
                        or (value == 0.0 and math.copysign(1.0, value) < 0.0)):
                    raise ValueError("shape < 0")
            kernel = (self._distributions.standard_gamma_f
                      if _dtype_name(dtype) == "float32"
                      else self._distributions.standard_gamma)
            if out is not None:
                name = _dtype_name(dtype)
                if str(out.dtype) != name:
                    raise TypeError(
                        f"Supplied output array has the wrong type. Expected {name}, "
                        f"got {out.dtype}"
                    )
                if not out.flags.c_contiguous or not out.flags.writeable:
                    raise ValueError("Supplied output array must be contiguous, writable, aligned, and in machine byte-order.")
            effective_size = tuple(out.shape) if out is not None and size is None else size
            result = self._broadcast_kernel(
                (shape,), effective_size, validate, lambda v: kernel(v[0]), dtype=dtype
            )
            if out is None:
                return result
            if tuple(out.shape) != tuple(result.shape):
                raise ValueError("size must match out.shape when used together")
            for i in range(result.size):
                out.flat[i] = result.flat[i]
            return out
        shape_value = float(shape)
        if (math.isnan(shape_value) or shape_value < 0.0
                or (shape_value == 0.0 and math.copysign(1.0, shape_value) < 0.0)):
            raise ValueError("shape < 0")
        if _dtype_name(dtype) == "float32":
            draw = lambda: self._distributions.standard_gamma_f(shape_value)
        else:
            draw = lambda: self._distributions.standard_gamma(shape_value)
        return self._continuous(
            draw,
            size=size, dtype=dtype, out=out,
        )

    def normal(self, loc=0.0, scale=1.0, size=None):
        def validate(v):
            if (v[1] < 0.0 or math.isnan(v[1])
                    or (v[1] == 0.0 and math.copysign(1.0, v[1]) < 0.0)):
                raise ValueError("scale < 0")
        return self._broadcast_kernel(
            (loc, scale), size, validate,
            lambda v: math.fma(v[1], self._distributions.standard_normal(), v[0]),
        )

    def exponential(self, scale=1.0, size=None):
        def validate(v):
            self._positive(v[0], "scale", allow_zero=True)
        return self._broadcast_kernel(
            (scale,), size, validate,
            lambda v: v[0] * self._distributions.standard_exponential(),
        )

    def gamma(self, shape, scale=1.0, size=None):
        def validate(v):
            self._positive(v[0], "shape", allow_zero=True)
            self._positive(v[1], "scale", allow_zero=True)
        return self._broadcast_kernel(
            (shape, scale), size, validate,
            lambda v: v[1] * self._distributions.standard_gamma(v[0]),
        )

    def beta(self, a, b, size=None):
        def validate(v):
            self._positive(v[0], "a")
            self._positive(v[1], "b")
        return self._broadcast_kernel(
            (a, b), size, validate, lambda v: self._distributions.beta(*v)
        )

    @staticmethod
    def _positive(value, name, allow_zero=False):
        value = float(value)
        bad = value < 0.0 if allow_zero else value <= 0.0
        if allow_zero and value == 0.0 and math.copysign(1.0, value) < 0.0:
            bad = True
        if bad or math.isnan(value):
            raise ValueError(f"{name} <= 0" if not allow_zero else f"{name} < 0")
        return value

    def chisquare(self, df, size=None):
        return self._broadcast_kernel(
            (df,), size, lambda v: self._positive(v[0], "df"),
            lambda v: self._distributions.chisquare(v[0]),
        )

    def f(self, dfnum, dfden, size=None):
        def validate(v):
            self._positive(v[0], "dfnum")
            self._positive(v[1], "dfden")
        return self._broadcast_kernel(
            (dfnum, dfden), size, validate, lambda v: self._distributions.f(*v)
        )

    def standard_cauchy(self, size=None):
        return self._continuous(self._distributions.standard_cauchy, size=size)

    def pareto(self, a, size=None):
        return self._broadcast_kernel(
            (a,), size, lambda v: self._positive(v[0], "a"),
            lambda v: self._distributions.pareto(v[0]),
        )

    def weibull(self, a, size=None):
        return self._broadcast_kernel(
            (a,), size, lambda v: self._positive(v[0], "a", allow_zero=True),
            lambda v: self._distributions.weibull(v[0]),
        )

    def power(self, a, size=None):
        return self._broadcast_kernel(
            (a,), size, lambda v: self._positive(v[0], "a"),
            lambda v: self._distributions.power(v[0]),
        )

    def laplace(self, loc=0.0, scale=1.0, size=None):
        return self._location_scale(loc, scale, size, self._distributions.laplace)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        return self._location_scale(loc, scale, size, self._distributions.gumbel)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        return self._location_scale(loc, scale, size, self._distributions.logistic)

    def _location_scale(self, loc, scale, size, kernel):
        def validate(v):
            self._positive(v[1], "scale", allow_zero=True)
        return self._broadcast_kernel(
            (loc, scale), size, validate, lambda v: kernel(*v)
        )

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        def validate(v):
            self._positive(v[1], "sigma", allow_zero=True)
        return self._broadcast_kernel(
            (mean, sigma), size, validate, lambda v: self._distributions.lognormal(*v)
        )

    def rayleigh(self, scale=1.0, size=None):
        return self._broadcast_kernel(
            (scale,), size, lambda v: self._positive(v[0], "scale", allow_zero=True),
            lambda v: self._distributions.rayleigh(v[0]),
        )

    def standard_t(self, df, size=None):
        return self._broadcast_kernel(
            (df,), size, lambda v: self._positive(v[0], "df"),
            lambda v: self._distributions.standard_t(v[0]),
        )

    def triangular(self, left, mode, right, size=None):
        def validate(v):
            if v[0] > v[1]:
                raise ValueError("left > mode")
            if v[1] > v[2]:
                raise ValueError("mode > right")
            if v[0] == v[2]:
                raise ValueError("left == right")
        return self._broadcast_kernel(
            (left, mode, right), size, validate,
            lambda v: self._distributions.triangular(*v),
        )

    def _discrete(self, draw, size=None):
        if size is None:
            return int(draw())
        shape = _shape(size)
        values = [draw() for _ in range(_count(shape))]
        return array(values, dtype=int64).reshape(shape)

    def geometric(self, p, size=None):
        def validate(v):
            if v[0] <= 0.0 or v[0] > 1.0 or math.isnan(v[0]):
                raise ValueError("p <= 0, p > 1 or p contains NaNs")
        return self._broadcast_kernel(
            (p,), size, validate, lambda v: self._distributions.geometric(v[0]),
            dtype=int64,
        )

    def logseries(self, p, size=None):
        def validate(v):
            if v[0] < 0.0 or v[0] >= 1.0 or math.isnan(v[0]):
                raise ValueError("p < 0, p >= 1 or p is NaN")
        return self._broadcast_kernel(
            (p,), size, validate, lambda v: self._distributions.logseries(v[0]),
            dtype=int64,
        )

    def zipf(self, a, size=None):
        def validate(v):
            if v[0] <= 1.0 or math.isnan(v[0]):
                raise ValueError("a <= 1 or a is NaN")
        return self._broadcast_kernel(
            (a,), size, validate, lambda v: self._distributions.zipf(v[0]),
            dtype=int64,
        )

    def poisson(self, lam=1.0, size=None):
        def validate(v):
            if v[0] < 0.0 or math.isnan(v[0]) or v[0] > self._poisson_lam_max:
                raise ValueError("lam < 0 or lam is too large")
        return self._broadcast_kernel(
            (lam,), size, validate, lambda v: self._distributions.poisson(v[0]),
            dtype=int64,
        )

    def binomial(self, n, p, size=None):
        def validate(v):
            if v[0] < 0.0:
                raise ValueError("n < 0")
            if v[1] < 0.0 or v[1] > 1.0 or math.isnan(v[1]):
                raise ValueError("p < 0, p > 1 or p is NaN")
        return self._broadcast_kernel(
            (n, p), size, validate,
            lambda v: self._distributions.binomial(int(v[0]), v[1]), dtype=int64,
        )

    def negative_binomial(self, n, p, size=None):
        def validate(v):
            if v[0] <= 0.0 or math.isnan(v[0]):
                raise ValueError("n <= 0")
            if v[1] <= 0.0 or v[1] > 1.0 or math.isnan(v[1]):
                raise ValueError("p <= 0, p > 1 or p contains NaNs")
            max_lam = (1.0 - v[1]) / v[1] * (v[0] + 10.0 * math.sqrt(v[0]))
            if max_lam > self._poisson_lam_max:
                raise ValueError("n too large or p too small")
        return self._broadcast_kernel(
            (n, p), size, validate,
            lambda v: self._distributions.negative_binomial(*v), dtype=int64,
        )

    def multinomial(self, n, pvals, size=None):
        p_array = asarray(pvals)
        if p_array.ndim == 0 or p_array.shape[-1] == 0:
            raise ValueError(
                "pvals must have at least 1 dimension and the last dimension "
                "of pvals must be greater than 0."
            )
        d = p_array.shape[-1]
        try:
            p_rows = [[float(p_array.flat[offset + j]) for j in range(d)]
                      for offset in range(0, p_array.size, d)]
        except (TypeError, ValueError):
            raise ValueError("pvals < 0, pvals > 1 or pvals contains NaNs") from None
        if any(v < 0.0 or v > 1.0 or math.isnan(v)
               for row in p_rows for v in row):
            raise ValueError("pvals < 0, pvals > 1 or pvals contains NaNs")
        for row in p_rows:
            if math.fsum(row[:-1]) > 1.0 + 1e-12:
                if (str(p_array.dtype) in ("float16", "float32")
                        and math.fsum(row) < 1.0001):
                    raise ValueError(
                        "sum(pvals[:-1].astype(np.float64)) > 1.0. The pvals "
                        "array is cast to 64-bit floating point prior to "
                        "checking the sum. Precision changes when casting may "
                        "cause problems even if the sum of the original pvals "
                        "is valid."
                    )
                raise ValueError("sum(pvals[:-1]) > 1.0")

        n_array = asarray(n)
        p_param_shape = tuple(p_array.shape[:-1])
        if size is None:
            n_broadcast, marker = broadcast_arrays(n_array, empty(p_param_shape))
            row_shape = tuple(n_broadcast.shape)
        else:
            row_shape = _shape(size)
        try:
            n_broadcast = broadcast_to(n_array, row_shape)
            p_broadcast = broadcast_to(p_array, row_shape + (d,))
        except ValueError:
            raise ValueError(
                f"Output size {row_shape} is not compatible with broadcast "
                "dimensions of inputs."
            ) from None
        try:
            n_values = [operator.index(v) for v in n_broadcast.flat]
        except TypeError:
            raise TypeError("n must be an integer") from None
        if any(value < 0 for value in n_values):
            raise ValueError("n < 0")

        shape = row_shape + (d,)
        rows = _count(row_shape)
        values = []
        for row_index in range(rows):
            remaining = n_values[row_index]
            remaining_p = 1.0
            probs = [float(p_broadcast.flat[row_index * d + j])
                     for j in range(d)]
            for p in probs[:-1]:
                if remaining <= 0:
                    draw = 0
                else:
                    draw = self._distributions.binomial(remaining, p / remaining_p)
                values.append(draw)
                remaining -= draw
                remaining_p -= p
            values.append(remaining)
        return array(values, dtype=int64).reshape(shape)

    def noncentral_chisquare(self, df, nonc, size=None):
        def validate(v):
            self._positive(v[0], "df")
            if v[1] < 0.0:
                raise ValueError("nonc < 0")
        return self._broadcast_kernel(
            (df, nonc), size, validate,
            lambda v: self._distributions.noncentral_chisquare(*v),
        )

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        def validate(v):
            self._positive(v[0], "dfnum")
            self._positive(v[1], "dfden")
            if v[2] < 0.0:
                raise ValueError("nonc < 0")
        return self._broadcast_kernel(
            (dfnum, dfden, nonc), size, validate,
            lambda v: self._distributions.noncentral_f(*v),
        )

    def wald(self, mean, scale, size=None):
        def validate(v):
            self._positive(v[0], "mean")
            self._positive(v[1], "scale")
        return self._broadcast_kernel(
            (mean, scale), size, validate, lambda v: self._distributions.wald(*v)
        )

    def vonmises(self, mu, kappa, size=None):
        def validate(v):
            if v[1] < 0.0:
                raise ValueError("kappa < 0")
        return self._broadcast_kernel(
            (mu, kappa), size, validate, lambda v: self._distributions.vonmises(*v)
        )

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        if all(asarray(value).ndim == 0 for value in (ngood, nbad, nsample)):
            ngood, nbad, nsample = int(ngood), int(nbad), int(nsample)
        def validate(v):
            good, bad, sample = (int(x) for x in v)
            if good < 0 or bad < 0 or sample < 0:
                raise ValueError("ngood, nbad, and nsample must be nonnegative")
            if good >= 10**9 or bad >= 10**9:
                raise ValueError("both ngood and nbad must be less than 1000000000")
            if sample > good + bad:
                raise ValueError("nsample > ngood + nbad")
        return self._broadcast_kernel(
            (ngood, nbad, nsample), size, validate,
            lambda v: self._distributions.hypergeometric(*(int(x) for x in v)),
            dtype=int64,
        )

    def multivariate_hypergeometric(self, colors, nsample, size=None,
                                    method="marginals"):
        colors_arr = asarray(colors)
        if colors_arr.ndim != 1:
            raise ValueError("colors must be a one-dimensional sequence")
        values = [operator.index(v) for v in colors_arr.flat]
        nsample = operator.index(nsample)
        if method not in ("count", "marginals"):
            raise ValueError("method must be 'count' or 'marginals'")
        if any(v < 0 for v in values):
            raise ValueError("colors must be nonnegative")
        total = sum(values)
        if method == "marginals" and total >= 10**9:
            raise ValueError(
                "When method is 'marginals', the sum of colors must be less "
                "than 1000000000"
            )
        if method == "count" and total > ((1 << 63) - 1) // 8:
            raise ValueError("colors is too large for the count method")
        if nsample < 0 or nsample > total:
            raise ValueError("nsample must be nonnegative and no greater than colors.sum()")
        out_shape = _shape(size) + (len(values),)
        rows = 1 if size is None else _count(_shape(size))
        result = []
        more_than_half = nsample > total // 2
        computed = total - nsample if more_than_half else nsample
        if method == "count":
            choices = []
            for color, count in enumerate(values):
                choices.extend([color] * count)
            for _ in range(rows):
                row = [0] * len(values)
                for j in range(computed):
                    k = j + self._random_interval(total - j - 1)
                    choices[j], choices[k] = choices[k], choices[j]
                for j in range(computed):
                    row[choices[j]] += 1
                if more_than_half:
                    row = [count - draw for count, draw in zip(values, row)]
                result.extend(row)
        else:
            for _ in range(rows):
                row = [0] * len(values)
                num_to_sample = computed
                remaining = total
                for j in range(max(len(values) - 1, 0)):
                    if num_to_sample <= 0:
                        break
                    remaining -= values[j]
                    draw = self._distributions.hypergeometric(
                        values[j], remaining, num_to_sample
                    )
                    row[j] = draw
                    num_to_sample -= draw
                if num_to_sample > 0 and values:
                    row[-1] = num_to_sample
                if more_than_half:
                    row = [count - draw for count, draw in zip(values, row)]
                result.extend(row)
        return array(result, dtype=int64).reshape(out_shape)

    def multivariate_normal(self, mean, cov, size=None, check_valid="warn",
                            tol=1e-8, *, method="svd"):
        from .. import dot, sqrt
        from ..linalg import cholesky, eigh, svd

        mean = asarray(mean)
        cov = asarray(cov)
        if mean.dtype.kind == "c" or cov.dtype.kind == "c":
            raise TypeError("mean and cov must not be complex")
        if mean.ndim != 1:
            raise ValueError("mean must be 1 dimensional")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")
        if method not in ("svd", "eigh", "cholesky"):
            raise ValueError("method must be 'svd', 'eigh', or 'cholesky'")
        final_shape = _shape(size) + (mean.shape[0],)
        x = self.standard_normal(final_shape).reshape(-1, mean.shape[0])
        cov = cov.astype(float64)
        if method == "cholesky":
            factor = cholesky(cov)
        elif method == "eigh":
            eigenvalues, u = eigh(cov)
            if check_valid != "ignore" and any(v < -tol for v in eigenvalues.flat):
                if check_valid == "raise":
                    raise ValueError("covariance is not symmetric positive-semidefinite")
                warnings.warn("covariance is not symmetric positive-semidefinite.", RuntimeWarning)
            factor = u * sqrt(abs(eigenvalues))
        else:
            u, singular, _ = svd(cov)
            factor = u * sqrt(singular)
        if check_valid not in ("warn", "raise", "ignore"):
            raise ValueError("check_valid must equal 'warn', 'raise', or 'ignore'")
        if check_valid != "ignore" and method != "cholesky":
            symmetric = all(
                abs(float(cov[i, j]) - float(cov[j, i])) <= tol
                for i in range(cov.shape[0]) for j in range(cov.shape[1])
            )
            evals, _ = eigh(cov)
            valid = symmetric and all(float(v) >= -tol for v in evals.flat)
            if not valid:
                if check_valid == "raise":
                    raise ValueError("covariance is not symmetric positive-semidefinite")
                warnings.warn("covariance is not symmetric positive-semidefinite.", RuntimeWarning)
        return (mean + dot(x, factor.T)).reshape(final_shape)

    def dirichlet(self, alpha, size=None):
        values = [float(v) for v in asarray(alpha).flat]
        if asarray(alpha).ndim != 1:
            raise ValueError("alpha must be 1-dimensional")
        if any(v < 0.0 or math.isnan(v) for v in values):
            raise ValueError("alpha < 0")
        rows = 1 if size is None else _count(_shape(size))
        result = []
        if values and max(values) < 0.1:
            cumulative = [0.0] * len(values)
            total = 0.0
            for j in range(len(values) - 1, -1, -1):
                total += values[j]
                cumulative[j] = total
            for _ in range(rows):
                row = [0.0] * len(values)
                acc = 1.0
                if total > 0.0:
                    for j in range(len(values) - 1):
                        if cumulative[j + 1] == 0.0:
                            v = 1.0
                        else:
                            v = self._distributions.beta(values[j], cumulative[j + 1])
                        row[j] = acc * v
                        acc *= 1.0 - v
                        if cumulative[j + 1] == 0.0:
                            break
                    row[-1] = acc
                result.extend(row)
        else:
            for _ in range(rows):
                row = [self._distributions.standard_gamma(v) for v in values]
                total = 0.0
                for v in row:
                    total += v
                invtotal = 1.0 / total
                result.extend(v * invtotal for v in row)
        shape = _shape(size) + (len(values),)
        return array(result, dtype=float64).reshape(shape)


__all__ = ["Generator"]
