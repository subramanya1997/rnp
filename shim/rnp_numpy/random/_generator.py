"""Bit-exact high-level Generator operations."""

import bisect
import math
import operator

from .. import (
    arange,
    array,
    asarray,
    broadcast_to,
    dtype as _dtype,
    empty,
    float32,
    float64,
    int64,
    uint32,
)
from .._arraycompat import broadcast_arrays
from ..exceptions import AxisError
from ._bit_generators import BitGenerator


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
                if low_value == 0:
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
            ndim = x.ndim
            axis = operator.index(axis)
            if axis < 0:
                axis += ndim
            if axis < 0 or axis >= ndim:
                raise AxisError(axis, ndim=ndim)
            n = x.shape[axis]
            order = list(range(n))
            for i in range(n - 1, 0, -1):
                j = self._random_interval(i)
                order[i], order[j] = order[j], order[i]
            if n:
                original = x.copy()
                shuffled = original.take(array(order, dtype=int64), axis=axis)
                x[...] = shuffled
            return None
        if axis != 0:
            raise NotImplementedError("Axis argument is only supported on ndarray objects")
        for i in range(len(x) - 1, 0, -1):
            j = self._random_interval(i)
            x[i], x[j] = x[j], x[i]
        return None

    def permutation(self, x, axis=0):
        try:
            n = operator.index(x)
        except TypeError:
            result = asarray(x).copy()
        else:
            result = arange(n)
        self.shuffle(result, axis=axis)
        return result

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        try:
            pop_size = operator.index(a)
            source = None
        except TypeError:
            source = asarray(a)
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
            weights = [float(v) for v in asarray(p).flat]
            if len(weights) != pop_size:
                raise ValueError("a and p must have same size")
            if any(math.isnan(v) for v in weights):
                raise ValueError("Probabilities contain NaN")
            if any(v < 0 for v in weights):
                raise ValueError("Probabilities are not non-negative")
            total = math.fsum(weights)
            if abs(total - 1.0) > math.sqrt(2.220446049250313e-16):
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
            return source.take(index, axis=axis)
        index_array = array(indices, dtype=int64).reshape(shape)
        if source is None:
            return index_array
        return source.take(index_array, axis=axis)

    def uniform(self, low=0.0, high=1.0, size=None):
        low = float(low)
        high = float(high)
        span = high - low
        if not math.isfinite(span):
            raise OverflowError("high - low range exceeds valid bounds")
        if span < 0:
            raise ValueError("high - low < 0")
        if size is None:
            return low + span * self.random()
        values = [low + span * self.random() for _ in range(_count(_shape(size)))]
        return array(values, dtype=float64).reshape(_shape(size))


__all__ = ["Generator"]
