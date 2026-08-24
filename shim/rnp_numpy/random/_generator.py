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
        arrays = list(broadcast_arrays(*arrays))
        shape = tuple(arrays[0].shape) if size is None else _shape(size)
        try:
            arrays = [broadcast_to(value, shape) for value in arrays]
        except ValueError:
            raise ValueError("shape mismatch: objects cannot be broadcast to a single shape") from None
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
            if tuple(out.shape) != shape or str(out.dtype) != name:
                raise ValueError("size must match out.shape when used together")
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
                raise TypeError("Unsupported dtype float32 for method inv")
            draw = lambda: -math.log1p(-self._bit_generator.next_double())
        else:
            draw = (self._distributions.standard_exponential_f
                    if _dtype_name(dtype) == "float32"
                    else self._distributions.standard_exponential)
        return self._continuous(draw, size=size, dtype=dtype, out=out)

    def standard_gamma(self, shape, size=None, dtype=float64, out=None):
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
        loc = float(loc)
        scale = float(scale)
        if (scale < 0.0 or math.isnan(scale)
                or (scale == 0.0 and math.copysign(1.0, scale) < 0.0)):
            raise ValueError("scale < 0")
        return self._continuous(lambda: math.fma(
            scale, self._distributions.standard_normal(), loc
        ), size=size)

    def exponential(self, scale=1.0, size=None):
        scale = float(scale)
        if (scale < 0.0 or math.isnan(scale)
                or (scale == 0.0 and math.copysign(1.0, scale) < 0.0)):
            raise ValueError("scale < 0")
        return self._continuous(
            lambda: scale * self._distributions.standard_exponential(), size=size
        )

    def gamma(self, shape, scale=1.0, size=None):
        shape = float(shape)
        scale = float(scale)
        if (shape < 0.0 or math.isnan(shape)
                or (shape == 0.0 and math.copysign(1.0, shape) < 0.0)):
            raise ValueError("shape < 0")
        if (scale < 0.0 or math.isnan(scale)
                or (scale == 0.0 and math.copysign(1.0, scale) < 0.0)):
            raise ValueError("scale < 0")
        return self._continuous(
            lambda: scale * self._distributions.standard_gamma(shape), size=size
        )

    def beta(self, a, b, size=None):
        a = float(a)
        b = float(b)
        if a <= 0.0 or math.isnan(a):
            raise ValueError("a <= 0")
        if b <= 0.0 or math.isnan(b):
            raise ValueError("b <= 0")
        return self._continuous(lambda: self._distributions.beta(a, b), size=size)

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
        df = self._positive(df, "df")
        return self._continuous(lambda: self._distributions.chisquare(df), size=size)

    def f(self, dfnum, dfden, size=None):
        dfnum = self._positive(dfnum, "dfnum")
        dfden = self._positive(dfden, "dfden")
        return self._continuous(lambda: self._distributions.f(dfnum, dfden), size=size)

    def standard_cauchy(self, size=None):
        return self._continuous(self._distributions.standard_cauchy, size=size)

    def pareto(self, a, size=None):
        a = self._positive(a, "a")
        return self._continuous(lambda: self._distributions.pareto(a), size=size)

    def weibull(self, a, size=None):
        a = self._positive(a, "a", allow_zero=True)
        return self._continuous(lambda: self._distributions.weibull(a), size=size)

    def power(self, a, size=None):
        a = self._positive(a, "a")
        return self._continuous(lambda: self._distributions.power(a), size=size)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        loc = float(loc)
        scale = self._positive(scale, "scale", allow_zero=True)
        return self._continuous(lambda: self._distributions.laplace(loc, scale), size=size)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        loc = float(loc)
        scale = self._positive(scale, "scale", allow_zero=True)
        return self._continuous(lambda: self._distributions.gumbel(loc, scale), size=size)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        loc = float(loc)
        scale = self._positive(scale, "scale", allow_zero=True)
        return self._continuous(lambda: self._distributions.logistic(loc, scale), size=size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        mean = float(mean)
        sigma = self._positive(sigma, "sigma", allow_zero=True)
        return self._continuous(lambda: self._distributions.lognormal(mean, sigma), size=size)

    def rayleigh(self, scale=1.0, size=None):
        scale = self._positive(scale, "scale", allow_zero=True)
        return self._continuous(lambda: self._distributions.rayleigh(scale), size=size)

    def standard_t(self, df, size=None):
        df = self._positive(df, "df")
        return self._continuous(lambda: self._distributions.standard_t(df), size=size)

    def triangular(self, left, mode, right, size=None):
        left, mode, right = float(left), float(mode), float(right)
        if left > mode:
            raise ValueError("left > mode")
        if mode > right:
            raise ValueError("mode > right")
        if left == right:
            raise ValueError("left == right")
        return self._continuous(
            lambda: self._distributions.triangular(left, mode, right), size=size
        )

    def _discrete(self, draw, size=None):
        if size is None:
            return int(draw())
        shape = _shape(size)
        values = [draw() for _ in range(_count(shape))]
        return array(values, dtype=int64).reshape(shape)

    def geometric(self, p, size=None):
        p = float(p)
        if p <= 0.0 or p > 1.0 or math.isnan(p):
            raise ValueError("p <= 0, p > 1 or p contains NaNs")
        return self._discrete(lambda: self._distributions.geometric(p), size=size)

    def logseries(self, p, size=None):
        p = float(p)
        if p < 0.0 or p >= 1.0 or math.isnan(p):
            raise ValueError("p < 0, p >= 1 or p is NaN")
        return self._discrete(lambda: self._distributions.logseries(p), size=size)

    def zipf(self, a, size=None):
        a = float(a)
        if a <= 1.0 or math.isnan(a):
            raise ValueError("a <= 1 or a is NaN")
        return self._discrete(lambda: self._distributions.zipf(a), size=size)

    def poisson(self, lam=1.0, size=None):
        lam = float(lam)
        if lam < 0.0 or math.isnan(lam) or lam > self._poisson_lam_max:
            raise ValueError("lam < 0 or lam is too large")
        return self._discrete(lambda: self._distributions.poisson(lam), size=size)

    def binomial(self, n, p, size=None):
        n = int(float(n))
        p = float(p)
        if n < 0:
            raise ValueError("n < 0")
        if p < 0.0 or p > 1.0 or math.isnan(p):
            raise ValueError("p < 0, p > 1 or p is NaN")
        return self._discrete(lambda: self._distributions.binomial(n, p), size=size)

    def negative_binomial(self, n, p, size=None):
        n = float(n)
        p = float(p)
        if n <= 0.0 or math.isnan(n):
            raise ValueError("n <= 0")
        if p <= 0.0 or p > 1.0 or math.isnan(p):
            raise ValueError("p <= 0, p > 1 or p contains NaNs")
        return self._discrete(
            lambda: self._distributions.negative_binomial(n, p), size=size
        )

    def multinomial(self, n, pvals, size=None):
        n = operator.index(n)
        probs = [float(v) for v in asarray(pvals).flat]
        if n < 0:
            raise ValueError("n < 0")
        if not probs:
            raise ValueError("pvals must have at least 1 dimension")
        if any(v < 0.0 or math.isnan(v) for v in probs):
            raise ValueError("pvals < 0, pvals > 1 or pvals contains NaNs")
        if math.fsum(probs[:-1]) > 1.0 + math.sqrt(2.220446049250313e-16):
            raise ValueError("sum(pvals[:-1]) > 1.0")
        shape = _shape(size) + (len(probs),)
        rows = 1 if size is None else _count(_shape(size))
        values = []
        for _ in range(rows):
            remaining = n
            remaining_p = 1.0
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
        df = self._positive(df, "df")
        nonc = float(nonc)
        if nonc < 0.0:
            raise ValueError("nonc < 0")
        return self._continuous(
            lambda: self._distributions.noncentral_chisquare(df, nonc), size=size
        )

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        dfnum = self._positive(dfnum, "dfnum")
        dfden = self._positive(dfden, "dfden")
        nonc = float(nonc)
        if nonc < 0.0:
            raise ValueError("nonc < 0")
        return self._continuous(
            lambda: self._distributions.noncentral_f(dfnum, dfden, nonc), size=size
        )

    def wald(self, mean, scale, size=None):
        mean = self._positive(mean, "mean")
        scale = self._positive(scale, "scale")
        return self._continuous(lambda: self._distributions.wald(mean, scale), size=size)

    def vonmises(self, mu, kappa, size=None):
        mu = float(mu)
        kappa = float(kappa)
        if kappa < 0.0:
            raise ValueError("kappa < 0")
        return self._continuous(lambda: self._distributions.vonmises(mu, kappa), size=size)

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
