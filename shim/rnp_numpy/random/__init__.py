"""`numpy.random` — enough of the API for the upstream core tests to run.

These generators are *not* bit-compatible with numpy's MT19937/PCG64 streams
(that is its own milestone); they are ordinary pseudo-random values with the
right shapes and dtypes, which is all the `_core` tests use them for.  What
*is* matched here is the API surface: the shape/dtype contract, the accepted
argument types (anything with `__index__`, not just `int` — numpy takes
`np.int64` sizes everywhere), and the error type and message for bad input.
Every message below was probed from real numpy 2.5.2.
"""

import operator as _operator
import random as _random
import warnings as _warnings
import functools as _functools
import inspect as _inspect
import math
import types as _types

from .. import arange, array, asarray, empty, float64, int_, zeros
from .bit_generator import (
    ISeedSequence,
    ISpawnableSeedSequence,
    SeedlessSeedSequence,
    SeedSequence,
)
from ._bit_generators import (
    BitGenerator,
    MT19937,
    PCG64,
    PCG64DXSM,
    Philox,
    SFC64,
)
from ._generator import Generator
from ._legacy import LegacyDistributions, LegacyMT19937
from ._distributions import DistributionKernels

_rng = LegacyMT19937()


def seed(s=None):
    _rng.seed(s)


def get_state(legacy=True):
    """Snapshot the global generator.

    numpy hands back the MT19937 key tuple; this port's generator is the
    stdlib Mersenne Twister rather than numpy's own stream (see the module
    docstring), so its state is what gets returned.  Callers -- hypothesis's
    `deterministic_PRNG` among them -- only ever round-trip the value through
    `set_state`, which is what makes the substitution safe.
    """
    return _rng.state


def set_state(state):
    _rng.state = state


def _bad_size(size):
    return TypeError(
        "expected a sequence of integers or a single integer, got '%r'" % (size,)
    )


def _shape(size):
    """numpy's `size=` conversion.

    A single value is accepted if it implements `__index__` (so `np.int64(3)`
    works, and `3.0` does not); otherwise it must be a sequence of such
    values.  `isinstance(size, int)` would reject every numpy integer scalar,
    which is the bug this replaces.
    """
    if size is None:
        return ()
    try:
        shape = (_operator.index(size),)
        if shape[0] < 0:
            raise ValueError("negative dimensions are not allowed")
        return shape
    except TypeError:
        pass
    try:
        items = tuple(size)
    except TypeError:
        raise _bad_size(size) from None
    # A non-integer *element* propagates `__index__`'s own TypeError
    # ("'float' object cannot be interpreted as an integer"), which is what
    # numpy does; only a non-sequence gets the "expected a sequence" message.
    shape = tuple(_operator.index(d) for d in items)
    if any(dim < 0 for dim in shape):
        raise ValueError("negative dimensions are not allowed")
    return shape


def _count(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _fill(shape, gen, dtype=float64):
    n = _count(shape)
    out = empty(shape, dtype)
    flat = out.flat
    for i in range(n):
        flat[i] = gen()
    return out


def _float_values(value):
    """Flatten an array-like parameter for vector-path validation."""
    return [float(item) for item in asarray(value).reshape(-1).tolist()]


# --- the implementations, each parameterised by the generator instance so
# --- that `RandomState(seed)` is actually independent of the global stream.


def _random_sample(r, size=None):
    shape = _shape(size)
    if size is None:
        return r.random()
    return _fill(shape, r.random)


def _randn(r, *args):
    shape = _shape(args if args else None)
    if not args:
        return r.gauss(0.0, 1.0)
    return _fill(shape, lambda: r.gauss(0.0, 1.0))


def _standard_normal(r, size=None):
    if size is None:
        return r.gauss(0.0, 1.0)
    return _fill(_shape(size), lambda: r.gauss(0.0, 1.0))


def _randint(r, low, high=None, size=None, dtype=int_):
    low = _operator.index(low)
    if high is None:
        low, high = 0, low
    else:
        high = _operator.index(high)
    if low >= high:
        raise ValueError("low >= high")
    if size is None:
        return r.randrange(low, high)
    return _fill(_shape(size), lambda: r.randrange(low, high), dtype)


def _random_integers(r, low, high=None, size=None):
    low = _operator.index(low)
    if high is None:
        low, high = 1, low
    else:
        high = _operator.index(high)
    _warnings.warn(
        "This function is deprecated. Please call randint(%s, %s + 1) instead"
        % (low, high),
        DeprecationWarning,
        stacklevel=3,
    )
    return _randint(r, low, high + 1, size)


def _permutation(r, x):
    try:
        n = _operator.index(x)
    except TypeError:
        items = asarray(x).tolist()
    else:
        items = list(range(n))
    r.shuffle(items)
    return array(items)


def _shuffle(r, x):
    n = len(x)
    for i in range(n - 1, 0, -1):
        j = r.randrange(i + 1)
        tmp = x[i].copy() if hasattr(x[i], "copy") else x[i]
        x[i] = x[j]
        x[j] = tmp


def _choice(r, a, size=None, replace=True, p=None):
    from_int = True
    try:
        n_a = _operator.index(a)
    except TypeError:
        from_int = False
        try:
            pool = asarray(a)
        except TypeError:
            pool = array(a, dtype=object)
        if pool.ndim != 1:
            raise ValueError("a must be 1-dimensional") from None
        pop = pool.size
        empty_msg = "'a' cannot be empty unless no samples are taken"
    else:
        if n_a <= 0:
            # numpy checks this before it looks at `size`, but only rejects
            # when samples are actually requested.
            pop = 0
            pool = arange(0)
            empty_msg = "a must be greater than 0 unless no samples are taken"
        else:
            pop = n_a
            pool = arange(n_a)
            empty_msg = "a must be greater than 0 unless no samples are taken"

    shape = _shape(size)
    n = _count(shape)

    if p is not None:
        try:
            p_array = asarray(p)
        except TypeError:
            raise ValueError("probabilities must be numeric") from None
        if p_array.ndim != 1:
            raise ValueError("'p' must be 1-dimensional")
        weights = [float(w) for w in p_array.tolist()]
        if len(weights) != pop:
            raise ValueError("'a' and 'p' must have same size")
        if any(not math.isfinite(w) for w in weights):
            raise ValueError("probabilities contain NaN")
        if any(w < 0 for w in weights):
            raise ValueError("probabilities are not non-negative")
        total = sum(weights)
        if abs(total - 1.0) > 1e-8:
            raise ValueError("probabilities do not sum to 1")

    if pop == 0:
        if n > 0:
            raise ValueError(empty_msg)
        picks = []
    elif replace:
        if p is None:
            picks = [r.randrange(pop) for _ in range(n)]
        else:
            picks = _weighted(r, weights, n)
    else:
        if n > pop:
            raise ValueError(
                "Cannot take a larger sample than population when 'replace=False'"
            )
        if p is None:
            picks = r.sample(range(pop), n)
        else:
            if sum(1 for w in weights if w > 0) < n:
                raise ValueError("Fewer non-zero entries in p than size")
            picks = _weighted_no_replace(r, weights, n)

    if size is None:
        if not picks:
            raise ValueError(empty_msg)
        item = pool[picks[0]]
        # numpy returns a plain `int` when `a` was given as an integer, but a
        # numpy scalar when `a` was an array.
        return int(item) if from_int else item
    out = pool[array(picks, dtype=int_)] if picks else pool[array([], dtype=int_)]
    return out.reshape(shape)


def _weighted(r, weights, n):
    """`n` independent draws from the categorical distribution `weights`."""
    cum = []
    running = 0.0
    for w in weights:
        running += w
        cum.append(running)
    picks = []
    for _ in range(n):
        u = r.random() * running
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if u < cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        picks.append(lo)
    return picks


def _weighted_no_replace(r, weights, n):
    """Legacy NumPy's batched renormalise, redraw and stable-unique loop."""
    remaining = list(weights)
    picks = []
    while len(picks) < n:
        for idx in picks:
            remaining[idx] = 0.0
        total = sum(remaining)
        batch = []
        for _ in range(n - len(picks)):
            u = r.random() * total
            running = 0.0
            idx = len(remaining) - 1
            for i, w in enumerate(remaining):
                running += w
                if u < running:
                    idx = i
                    break
            if idx not in batch:
                batch.append(idx)
        picks.extend(batch)
    return picks


def _uniform(r, low=0.0, high=1.0, size=None):
    if size is None:
        return r.uniform(low, high)
    return _fill(_shape(size), lambda: r.uniform(low, high))


def _normal(r, loc=0.0, scale=1.0, size=None):
    if size is None:
        return r.gauss(loc, scale)
    return _fill(_shape(size), lambda: r.gauss(loc, scale))


# --- module-level API, bound to the global generator ---


def random_sample(size=None):
    return _random_sample(_rng, size)


random = random_sample
ranf = random_sample
sample = random_sample


def rand(*args):
    return random_sample(args if args else None)


def randn(*args):
    return _randn(_rng, *args)


def standard_normal(size=None):
    return _standard_normal(_rng, size)


def randint(low, high=None, size=None, dtype=int_):
    return _randint(_rng, low, high, size, dtype)


def random_integers(low, high=None, size=None):
    return _random_integers(_rng, low, high, size)


def permutation(x):
    return _permutation(_rng, x)


def shuffle(x):
    return _shuffle(_rng, x)


def choice(a, size=None, replace=True, p=None):
    return _choice(_rng, a, size, replace, p)


def uniform(low=0.0, high=1.0, size=None):
    return _uniform(_rng, low, high, size)


def normal(loc=0.0, scale=1.0, size=None):
    return _normal(_rng, loc, scale, size)


class RandomState:
    """numpy's legacy `RandomState`.

    `.sample` and `.ranf` are deliberately absent: numpy removed them from
    `RandomState` in 2.0 (they survive only as module-level aliases), and the
    upstream tests check for their absence.  Probed on 2.5.2.
    """

    _poisson_lam_max = 9.223372006484771e18

    def __init__(self, s=None):
        if isinstance(s, type) and issubclass(s, BitGenerator):
            raise ValueError("RandomState requires an instantiated BitGenerator")
        self._bit_generator = s if isinstance(s, BitGenerator) else LegacyMT19937(s)
        self._legacy = LegacyDistributions(self._bit_generator)
        self._modern = DistributionKernels(self._bit_generator)

    def seed(self, s=None):
        if isinstance(self._bit_generator, LegacyMT19937):
            self._bit_generator.seed(s)
        else:
            replacement = type(self._bit_generator)(s)
            self._bit_generator.state = replacement.state
        self._legacy.has_gauss = False
        self._legacy.gauss_value = 0.0

    def _fill(self, size, draw, dtype=float64):
        if size is None:
            return draw()
        return _fill(_shape(size), draw, dtype)

    def random_sample(self, size=None):
        return self._fill(size, self._bit_generator.next_double)

    def random(self, size=None):
        return self.random_sample(size)

    def rand(self, *args):
        return self.random_sample(args if args else None)

    def randn(self, *args):
        return self.standard_normal(args if args else None)

    def standard_normal(self, size=None):
        return self._fill(size, self._legacy.gauss)

    def randint(self, low, high=None, size=None, dtype=int_):
        low = _operator.index(low)
        if high is None:
            low, high = 0, low
        else:
            high = _operator.index(high)
        shape = _shape(size)
        if size is not None and _count(shape) == 0:
            return _fill(shape, lambda: 0, dtype)
        if low >= high:
            raise ValueError("low >= high")
        maximum = high - low - 1
        def draw():
            mask = maximum
            mask |= mask >> 1
            mask |= mask >> 2
            mask |= mask >> 4
            mask |= mask >> 8
            mask |= mask >> 16
            mask |= mask >> 32
            while True:
                value = (self._bit_generator.next_uint32() if maximum <= 0xFFFFFFFF
                         else self._bit_generator.next_uint64()) & mask
                if value <= maximum:
                    return low + value
        return self._fill(size, draw, dtype)

    def random_integers(self, low, high=None, size=None):
        _warnings.warn("This function is deprecated", DeprecationWarning, stacklevel=2)
        if high is None:
            low, high = 1, low
        return self.randint(low, _operator.index(high) + 1, size)

    def permutation(self, x):
        try:
            result = arange(_operator.index(x))
        except TypeError:
            result = asarray(x).copy()
            if result.ndim == 0:
                raise IndexError("x must be an integer or at least 1-dimensional")
        self.shuffle(result)
        return result

    def shuffle(self, x):
        for i in range(len(x) - 1, 0, -1):
            j = self.randint(0, i + 1)
            try:
                x[[i, j]] = x[[j, i]]
            except TypeError:
                tmp = x[i]
                x[i] = x[j]
                x[j] = tmp

    def choice(self, a, size=None, replace=True, p=None):
        return _choice(self._bit_generator, a, size, replace, p)

    def bytes(self, length):
        length = _operator.index(length)
        return b"".join(self._bit_generator.next_uint32().to_bytes(4, "little")
                        for _ in range((length + 3) // 4))[:length]

    def uniform(self, low=0.0, high=1.0, size=None):
        try:
            low, high = float(low), float(high)
        except TypeError:
            lows, highs = _float_values(low), _float_values(high)
            if len(lows) == 1:
                lows *= len(highs)
            if len(highs) == 1:
                highs *= len(lows)
            spans = [hi - lo for lo, hi in zip(lows, highs)]
            if any(not math.isfinite(span) for span in spans):
                raise OverflowError("high - low range exceeds valid bounds")
            if any(span < 0.0 for span in spans):
                raise ValueError("high - low < 0")
            raise
        span = high - low
        if not math.isfinite(span):
            raise OverflowError("high - low range exceeds valid bounds")
        if span < 0.0:
            raise ValueError("high - low < 0")
        return self._fill(size, lambda: math.fma(high - low,
                          self._bit_generator.next_double(), low))

    def normal(self, loc=0.0, scale=1.0, size=None):
        loc, scale = float(loc), float(scale)
        if scale < 0.0 or (scale == 0.0 and math.copysign(1.0, scale) < 0.0):
            raise ValueError("scale < 0")
        return self._fill(size, lambda: math.fma(scale, self._legacy.gauss(), loc))

    def standard_exponential(self, size=None):
        return self._fill(size, self._legacy.standard_exponential)

    def exponential(self, scale=1.0, size=None):
        scale = float(scale)
        if scale < 0.0 or (scale == 0.0 and math.copysign(1.0, scale) < 0.0):
            raise ValueError("scale < 0")
        return self._fill(size, lambda: scale * self._legacy.standard_exponential())

    def standard_gamma(self, shape, size=None):
        shape = float(shape)
        if shape < 0.0 or (shape == 0.0 and math.copysign(1.0, shape) < 0.0):
            raise ValueError("shape < 0")
        return self._fill(size, lambda: self._legacy.standard_gamma(shape))

    def gamma(self, shape, scale=1.0, size=None):
        shape, scale = float(shape), float(scale)
        if (shape < 0.0 or scale < 0.0 or
                (shape == 0.0 and math.copysign(1.0, shape) < 0.0) or
                (scale == 0.0 and math.copysign(1.0, scale) < 0.0)):
            raise ValueError("shape < 0 or scale < 0")
        return self._fill(size, lambda: scale * self._legacy.standard_gamma(shape))

    def beta(self, a, b, size=None):
        a, b = float(a), float(b)
        def draw():
            if a <= 1.0 and b <= 1.0:
                while True:
                    u = self._bit_generator.next_double()
                    v = self._bit_generator.next_double()
                    x = math.pow(u, 1.0 / a)
                    y = math.pow(v, 1.0 / b)
                    if x + y <= 1.0:
                        if x + y > 0.0:
                            return x / (x + y)
                        lx, ly = math.log(u) / a, math.log(v) / b
                        lm = max(lx, ly)
                        lx, ly = lx - lm, ly - lm
                        return math.exp(lx - math.log(math.exp(lx) + math.exp(ly)))
            ga = self._legacy.standard_gamma(a)
            gb = self._legacy.standard_gamma(b)
            return ga / (ga + gb)
        return self._fill(size, draw)

    def chisquare(self, df, size=None):
        df = float(df)
        return self._fill(size, lambda: 2.0 * self._legacy.standard_gamma(df / 2.0))

    def f(self, dfnum, dfden, size=None):
        dfnum, dfden = float(dfnum), float(dfden)
        return self._fill(size, lambda: (2.0 * self._legacy.standard_gamma(dfnum / 2.0) * dfden) /
                          (2.0 * self._legacy.standard_gamma(dfden / 2.0) * dfnum))

    def pareto(self, a, size=None):
        a = float(a)
        return self._fill(size, lambda: math.exp(self._legacy.standard_exponential() / a) - 1.0)

    def weibull(self, a, size=None):
        a = float(a)
        if a < 0.0 or (a == 0.0 and math.copysign(1.0, a) < 0.0):
            raise ValueError("a < 0")
        return self._fill(size, lambda: 0.0 if a == 0.0 else
                          math.pow(self._legacy.standard_exponential(), 1.0 / a))

    def power(self, a, size=None):
        a = float(a)
        return self._fill(size, lambda: math.pow(
            1.0 - math.exp(-self._legacy.standard_exponential()), 1.0 / a))

    def standard_cauchy(self, size=None):
        return self._fill(size, lambda: self._legacy.gauss() / self._legacy.gauss())

    def standard_t(self, df, size=None):
        df = float(df)
        return self._fill(size, lambda: math.sqrt(df / 2.0) * self._legacy.gauss() /
                          math.sqrt(self._legacy.standard_gamma(df / 2.0)))

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        mean, sigma = float(mean), float(sigma)
        if sigma < 0.0 or (sigma == 0.0 and math.copysign(1.0, sigma) < 0.0):
            raise ValueError("sigma < 0")
        return self._fill(size, lambda: math.exp(math.fma(sigma, self._legacy.gauss(), mean)))

    def rayleigh(self, scale=1.0, size=None):
        scale = float(scale)
        if scale < 0.0 or (scale == 0.0 and math.copysign(1.0, scale) < 0.0):
            raise ValueError("scale < 0")
        return self._fill(size, lambda: scale * math.sqrt(
            -2.0 * math.log1p(-self._bit_generator.next_double())))

    def poisson(self, lam=1.0, size=None):
        try:
            lam = float(lam)
        except TypeError:
            values = _float_values(lam)
            if any(v < 0.0 or math.isnan(v) or
                   v > Generator._poisson_lam_max for v in values):
                raise ValueError("lam < 0 or lam is too large")
            raise
        if lam < 0.0 or math.isnan(lam) or lam > Generator._poisson_lam_max:
            raise ValueError("lam < 0 or lam is too large")
        return self._fill(size, lambda: self._modern.poisson(lam), int_)

    def geometric(self, p, size=None):
        try:
            p = float(p)
        except TypeError:
            values = _float_values(p)
            if any(v <= 0.0 or v > 1.0 or math.isnan(v) for v in values):
                raise ValueError("p <= 0, p > 1 or p is NaN")
            raise
        if p <= 0.0 or p > 1.0 or math.isnan(p):
            raise ValueError("p <= 0, p > 1 or p is NaN")
        def draw():
            if p >= 1.0 / 3.0:
                x = 1
                total = product = p
                q = 1.0 - p
                u = self._bit_generator.next_double()
                while u > total:
                    product *= q
                    total += product
                    x += 1
                return x
            return math.ceil(math.log1p(-self._bit_generator.next_double()) /
                             math.log(1.0 - p))
        return self._fill(size, draw, int_)

    def logseries(self, p, size=None):
        try:
            p = float(p)
        except TypeError:
            values = _float_values(p)
            if any(v < 0.0 or v >= 1.0 or math.isnan(v) for v in values):
                raise ValueError("p < 0, p >= 1 or p is NaN")
            raise
        if p < 0.0 or p >= 1.0 or math.isnan(p):
            raise ValueError("p < 0, p >= 1 or p is NaN")
        return self._fill(size, lambda: self._modern.logseries(p), int_)

    def zipf(self, a, size=None):
        a = float(a)
        def draw():
            am1 = a - 1.0
            b = math.pow(2.0, am1)
            while True:
                u = 1.0 - self._bit_generator.next_double()
                v = self._bit_generator.next_double()
                x = math.floor(math.pow(u, -1.0 / am1))
                if x > (1 << 63) - 1 or x < 1.0:
                    continue
                t = math.pow(1.0 + 1.0 / x, am1)
                if v * x * (t - 1.0) / (b - 1.0) <= t / b:
                    return int(x)
        return self._fill(size, draw, int_)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        ngood, nbad, nsample = int(ngood), int(nbad), int(nsample)
        def draw():
            if nsample <= 0:
                return 0
            if nsample <= 10:
                d1 = nbad + ngood - nsample
                d2 = min(nbad, ngood)
                y = float(d2)
                k = nsample
                while y > 0.0:
                    u = self._bit_generator.next_double()
                    y -= math.floor(u + y / (d1 + k))
                    k -= 1
                    if k == 0:
                        break
                z = int(d2 - y)
                return nsample - z if ngood > nbad else z
            mingb, maxgb = min(ngood, nbad), max(ngood, nbad)
            pop = ngood + nbad
            m = min(nsample, pop - nsample)
            d4 = mingb / pop
            d5 = 1.0 - d4
            d6 = m * d4 + 0.5
            d7 = math.sqrt((pop - m) * nsample * d4 * d5 / (pop - 1) + 0.5)
            d8 = math.fma(1.7155277699214135, d7, 0.8989161620588988)
            d9 = math.floor((m + 1.0) * (mingb + 1.0) / (pop + 2.0))
            lg = self._modern._loggam
            d10 = (lg(d9 + 1.0) + lg(mingb - d9 + 1.0) + lg(m - d9 + 1.0)
                   + lg(maxgb - m + d9 + 1.0))
            d11 = min(min(m, mingb) + 1.0, math.floor(d6 + 16.0 * d7))
            while True:
                x = self._bit_generator.next_double()
                y = self._bit_generator.next_double()
                w = math.fma(d8 / x, y - 0.5, d6)
                if w < 0.0 or w >= d11:
                    continue
                z = math.floor(w)
                t = d10 - (lg(z + 1.0) + lg(mingb - z + 1.0) + lg(m - z + 1.0)
                           + lg(maxgb - m + z + 1.0))
                if math.fma(x, 4.0 - x, -3.0) <= t:
                    break
                if x * (x - t) >= 1.0:
                    continue
                if 2.0 * math.log(x) <= t:
                    break
            if ngood > nbad:
                z = m - z
            if m < nsample:
                z = ngood - z
            return z
        return self._fill(size, draw, int_)

    def triangular(self, left, mode, right, size=None):
        left, mode, right = float(left), float(mode), float(right)
        return self._fill(size, lambda: self._modern.triangular(left, mode, right))

    def laplace(self, loc=0.0, scale=1.0, size=None):
        loc, scale = float(loc), float(scale)
        if scale < 0.0 or (scale == 0.0 and math.copysign(1.0, scale) < 0.0):
            raise ValueError("scale < 0")
        return self._fill(size, lambda: self._modern.laplace(loc, scale))

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        loc, scale = float(loc), float(scale)
        if scale < 0.0 or (scale == 0.0 and math.copysign(1.0, scale) < 0.0):
            raise ValueError("scale < 0")
        return self._fill(size, lambda: self._modern.gumbel(loc, scale))

    def logistic(self, loc=0.0, scale=1.0, size=None):
        loc, scale = float(loc), float(scale)
        return self._fill(size, lambda: self._modern.logistic(loc, scale))

    def wald(self, mean, scale, size=None):
        mean, scale = float(mean), float(scale)
        def draw():
            mu_2l = mean / (2.0 * scale)
            y = self._legacy.gauss()
            y = mean * y * y
            radical = math.fma(4.0 * scale, y, y * y)
            x = math.fma(mu_2l, y - math.sqrt(radical), mean)
            if self._bit_generator.next_double() <= mean / (mean + x):
                return x
            return mean * mean / x
        return self._fill(size, draw)

    def binomial(self, n, p, size=None):
        n, p = int(float(n)), float(p)
        return self._fill(size, lambda: self._legacy_binomial(n, p), int_)

    def _legacy_binomial(self, n, p):
        use_p = p if p <= 0.5 else 1.0 - p
        if use_p * n > 30.0:
            return self._modern.binomial(n, p)
        q = 1.0 - use_p
        qn = math.exp(n * math.log(q))
        np_ = n * use_p
        bound = min(n, int(np_ + 10.0 * math.sqrt(np_ * q + 1.0)))
        x, px = 0, qn
        u = self._bit_generator.next_double()
        while u > px:
            x += 1
            if x > bound:
                x, px = 0, qn
                u = self._bit_generator.next_double()
            else:
                u -= px
                px = ((n - x + 1) * use_p * px) / (x * q)
        return n - x if p > 0.5 else x

    def negative_binomial(self, n, p, size=None):
        n = float(n)
        try:
            p = float(p)
        except TypeError:
            values = _float_values(p)
            if any(not math.isfinite(v) or v <= 0.0 or v > 1.0 for v in values):
                raise ValueError("p <= 0, p > 1 or p is NaN")
            raise
        if not math.isfinite(p) or p <= 0.0 or p > 1.0:
            raise ValueError("p <= 0, p > 1 or p is NaN")
        return self._fill(size, lambda: self._modern.poisson(
            self._legacy.standard_gamma(n) * ((1.0 - p) / p)), int_)

    def dirichlet(self, alpha, size=None):
        values = [float(v) for v in asarray(alpha).flat]
        if any(v <= 0.0 or math.isnan(v) for v in values):
            raise ValueError("alpha <= 0")
        rows = 1 if size is None else _count(_shape(size))
        out = []
        for _ in range(rows):
            row = [self._legacy.standard_gamma(v) for v in values]
            total = sum(row)
            out.extend(v / total for v in row)
        return array(out).reshape(_shape(size) + (len(values),))

    def multinomial(self, n, pvals, size=None):
        n = _operator.index(n)
        probs = [float(v) for v in asarray(pvals).flat]
        rows = 1 if size is None else _count(_shape(size))
        out = []
        for _ in range(rows):
            remaining, remaining_p = n, 1.0
            row = [0] * len(probs)
            for j, p in enumerate(probs[:-1]):
                row[j] = self._legacy_binomial(remaining, p / remaining_p)
                remaining -= row[j]
                if remaining <= 0:
                    break
                remaining_p -= p
            if remaining > 0:
                row[-1] = remaining
            out.extend(row)
        return array(out, dtype=int_).reshape(_shape(size) + (len(probs),))

    def noncentral_chisquare(self, df, nonc, size=None):
        df, nonc = float(df), float(nonc)
        def draw():
            if nonc == 0.0:
                return 2.0 * self._legacy.standard_gamma(df / 2.0)
            if df > 1.0:
                chi2 = 2.0 * self._legacy.standard_gamma((df - 1.0) / 2.0)
                value = self._legacy.gauss() + math.sqrt(nonc)
                return math.fma(value, value, chi2)
            count = self._modern.poisson(nonc / 2.0)
            return 2.0 * self._legacy.standard_gamma((df + 2.0 * count) / 2.0)
        return self._fill(size, draw)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        dfnum, dfden, nonc = float(dfnum), float(dfden), float(nonc)
        return self._fill(size, lambda: self.noncentral_chisquare(dfnum, nonc) * dfden /
                          (2.0 * self._legacy.standard_gamma(dfden / 2.0) * dfnum))

    def vonmises(self, mu, kappa, size=None):
        mu, kappa = float(mu), float(kappa)
        def draw():
            if math.isnan(kappa):
                return math.nan
            if kappa < 1e-8:
                return math.pi * (2.0 * self._bit_generator.next_double() - 1.0)
            if kappa < 1e-5:
                s = 1.0 / kappa + kappa
            else:
                r = 1.0 + math.sqrt(math.fma(4.0 * kappa, kappa, 1.0))
                rho = (r - math.sqrt(2.0 * r)) / (2.0 * kappa)
                s = math.fma(rho, rho, 1.0) / (2.0 * rho)
            while True:
                u = self._bit_generator.next_double()
                z = math.cos(math.pi * u)
                w = math.fma(s, z, 1.0) / (s + z)
                y = kappa * (s - w)
                v = self._bit_generator.next_double()
                if math.fma(y, 2.0 - y, -v) >= 0.0 or math.log(y / v) + 1.0 - y >= 0.0:
                    break
            result = math.acos(w)
            if self._bit_generator.next_double() < 0.5:
                result = -result
            result += mu
            negative = result < 0.0
            mod = math.fmod(abs(result) + math.pi, 2.0 * math.pi) - math.pi
            return -mod if negative else mod
        return self._fill(size, draw)

    def multivariate_normal(self, mean, cov, size=None, check_valid="warn", tol=1e-8):
        # Use the legacy polar normal stream, then the same SVD transform.
        if check_valid not in ("warn", "raise", "ignore"):
            raise ValueError("check_valid must equal 'warn', 'raise', or 'ignore'")
        from .. import allclose, dot, sqrt
        from ..linalg import svd
        mean, cov = asarray(mean), asarray(cov)
        final_shape = _shape(size) + (mean.shape[0],)
        x = self.standard_normal(final_shape).reshape(-1, mean.shape[0])
        u, singular, vh = svd(cov.astype(float64))
        valid = allclose(dot(vh.T * singular, vh), cov, rtol=tol, atol=tol)
        if not valid and check_valid != "ignore":
            if check_valid == "raise":
                raise ValueError("covariance is not symmetric positive-semidefinite")
            _warnings.warn("covariance is not symmetric positive-semidefinite",
                           RuntimeWarning, stacklevel=2)
        return (mean + dot(x, (u * sqrt(singular)).T)).reshape(final_shape)

    def get_state(self, legacy=True):
        if isinstance(self._bit_generator, LegacyMT19937):
            state = list(self._bit_generator.state)
            state[3] = int(self._legacy.has_gauss)
            state[4] = self._legacy.gauss_value
            if legacy:
                return tuple(state)
            return {
                "bit_generator": "MT19937",
                "state": {"key": state[1], "pos": state[2]},
                "has_gauss": int(self._legacy.has_gauss),
                "gauss": self._legacy.gauss_value,
            }
        if legacy:
            _warnings.warn("get_state legacy output is unavailable for this BitGenerator",
                           RuntimeWarning, stacklevel=2)
        state = dict(self._bit_generator.state)
        state["has_gauss"] = int(self._legacy.has_gauss)
        state["gauss"] = self._legacy.gauss_value
        return state

    def set_state(self, state):
        if not isinstance(state, (tuple, dict)):
            raise TypeError("state must be a dict or a tuple")
        if isinstance(state, dict):
            name = state.get("bit_generator")
            if name is None:
                raise ValueError("state dictionary is missing bit_generator")
            if not hasattr(self, "_bit_generator"):
                classes = {"MT19937": LegacyMT19937, "PCG64": PCG64,
                           "PCG64DXSM": PCG64DXSM, "Philox": Philox,
                           "SFC64": SFC64}
                if name not in classes:
                    raise ValueError("unknown bit generator")
                self._bit_generator = classes[name]()
                self._legacy = LegacyDistributions(self._bit_generator)
                self._modern = DistributionKernels(self._bit_generator)
            if isinstance(self._bit_generator, LegacyMT19937):
                self._bit_generator.state = (
                    "MT19937", state["state"]["key"], state["state"]["pos"]
                )
            else:
                bitgen_state = {key: value for key, value in state.items()
                                if key not in ("has_gauss", "gauss")}
                self._bit_generator.state = bitgen_state
            self._legacy.has_gauss = bool(state.get("has_gauss", 0))
            self._legacy.gauss_value = float(state.get("gauss", 0.0))
            return
        if not isinstance(self._bit_generator, LegacyMT19937):
            raise ValueError(
                f"state must be for a {type(self._bit_generator).__name__} RNG"
            )
        self._bit_generator.state = state
        if isinstance(state, tuple) and len(state) > 4:
            self._legacy.has_gauss = bool(state[3])
            self._legacy.gauss_value = float(state[4])

    def __getstate__(self):
        return self.get_state(legacy=False)

    def __setstate__(self, state):
        self.set_state(state)

    def __repr__(self):
        return f"RandomState({type(self._bit_generator).__name__.replace('Legacy', '')}) at {id(self):#X}"

    def tomaxint(self, size=None):
        return self._fill(size, lambda: self._bit_generator.next_uint64() >> 1, int_)


def _broadcast_distribution(method):
    """Vectorize a legacy scalar distribution over broadcast parameters.

    NumPy's legacy ``RandomState`` broadcasts distribution parameters first,
    then consumes the scalar kernel once per output element in C order.  The
    scalar kernels above already carry the validation and exact stream logic;
    this adapter supplies only that shared outer loop.
    """
    signature = _inspect.signature(method)

    @_functools.wraps(method)
    def broadcasted(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self", None)
        size = values.pop("size", None)
        arrays = {name: asarray(value) for name, value in values.items()}
        if all(value.ndim == 0 for value in arrays.values()):
            _validate_broadcast_distribution(
                method.__name__,
                {name: value.reshape(-1)[0] for name, value in arrays.items()},
            )
            return method(self, *args, **kwargs)

        names = list(arrays)
        from .. import broadcast_arrays, broadcast_to
        if size is None:
            expanded = broadcast_arrays(*(arrays[name] for name in names))
            shape = expanded[0].shape
        else:
            shape = _shape(size)
            expanded = [broadcast_to(arrays[name], shape) for name in names]

        # Repeated parameter arrays are common (and the thread-safety tests
        # use a 100x1000 array of ones).  Preserve the same scalar-kernel
        # stream while letting the existing bulk fill avoid a Python call per
        # element.
        constants = []
        for value in expanded:
            first = value.reshape(-1)[0]
            if not bool((value == first).all()):
                break
            constants.append(first)
        else:
            scalar_args = dict(zip(names, constants))
            _validate_broadcast_distribution(method.__name__, scalar_args)
            return method(self, **scalar_args, size=shape)

        flat = [value.reshape(-1) for value in expanded]
        result = []
        for index in range(_count(shape)):
            scalar_args = {name: flat[pos][index]
                           for pos, name in enumerate(names)}
            _validate_broadcast_distribution(method.__name__, scalar_args)
            result.append(method(self, **scalar_args, size=None))
        return array(result).reshape(shape)

    return broadcasted


def _validate_broadcast_distribution(name, values):
    """Validate scalar parameters before entering rejection-sampling kernels."""
    value = {key: float(item) for key, item in values.items()}

    def reject(condition, message):
        if condition:
            raise ValueError(message)

    if name == "beta":
        reject(value["a"] <= 0, "a <= 0")
        reject(value["b"] <= 0, "b <= 0")
    elif name == "f":
        reject(value["dfnum"] <= 0, "dfnum <= 0")
        reject(value["dfden"] <= 0, "dfden <= 0")
    elif name == "noncentral_f":
        reject(value["dfnum"] <= 0, "dfnum <= 0")
        reject(value["dfden"] <= 0, "dfden <= 0")
        reject(value["nonc"] < 0, "nonc < 0")
    elif name == "chisquare":
        reject(value["df"] <= 0, "df <= 0")
    elif name == "noncentral_chisquare":
        reject(value["df"] <= 0, "df <= 0")
        reject(value["nonc"] < 0, "nonc < 0")
    elif name == "vonmises":
        reject(value["kappa"] < 0, "kappa < 0")
    elif name in ("pareto", "power"):
        reject(value["a"] <= 0, "a <= 0")
    elif name == "logistic":
        reject(value["scale"] < 0, "scale < 0")
    elif name == "wald":
        reject(value["mean"] <= 0, "mean <= 0")
        reject(value["scale"] <= 0, "scale <= 0")
    elif name == "triangular":
        reject(value["left"] > value["mode"], "left > mode")
        reject(value["mode"] > value["right"], "mode > right")
        reject(value["left"] == value["right"], "left == right")
    elif name == "binomial":
        reject(value["n"] < 0, "n < 0")
        reject(not 0 <= value["p"] <= 1, "p < 0, p > 1 or p is NaN")
    elif name == "negative_binomial":
        reject(value["n"] <= 0, "n <= 0")
        reject(not 0 < value["p"] <= 1, "p <= 0, p > 1 or p contains NaNs")
    elif name == "poisson":
        lam = value["lam"]
        reject(not 0 <= lam <= RandomState._poisson_lam_max,
               "lam < 0 or lam contains NaNs")
    elif name == "zipf":
        reject(not value["a"] > 1, "a <= 1 or a contains NaNs")
    elif name == "geometric":
        reject(not 0 < value["p"] <= 1,
               "p <= 0, p > 1 or p contains NaNs")
    elif name == "hypergeometric":
        reject(value["ngood"] < 0, "ngood < 0")
        reject(value["nbad"] < 0, "nbad < 0")
        reject(value["nsample"] < 1, "nsample < 1")
        reject(value["ngood"] + value["nbad"] < value["nsample"],
               "ngood + nbad < nsample")
    elif name == "logseries":
        reject(not 0 < value["p"] < 1,
               "p < 0, p >= 1 or p contains NaNs")


for _broadcast_name in (
    "uniform", "normal", "beta", "exponential", "standard_gamma", "gamma",
    "f", "noncentral_f", "chisquare", "noncentral_chisquare", "standard_t",
    "vonmises", "pareto", "weibull", "power", "laplace", "gumbel",
    "logistic", "lognormal", "rayleigh", "wald", "triangular", "binomial",
    "negative_binomial", "poisson", "zipf", "geometric", "hypergeometric",
    "logseries",
):
    setattr(RandomState, _broadcast_name,
            _broadcast_distribution(getattr(RandomState, _broadcast_name)))


def default_rng(seed=None):
    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, RandomState):
        bit_generator = seed._bit_generator
        if isinstance(bit_generator, LegacyMT19937):
            modern = MT19937()
            modern.state = {
                "bit_generator": "MT19937",
                "state": {"key": array(bit_generator._key, dtype="uint32"),
                          "pos": bit_generator._pos},
            }
            seed._bit_generator = modern
            seed._legacy = LegacyDistributions(modern)
            seed._modern = DistributionKernels(modern)
            bit_generator = modern
        return Generator(bit_generator)
    if isinstance(seed, BitGenerator):
        return Generator(seed)
    return Generator(PCG64(seed))


# NumPy's module-level API is a singleton RandomState.
_global_random_state = RandomState(_rng)
mtrand = _types.SimpleNamespace(_rand=_global_random_state)


def get_bit_generator():
    return _global_random_state._bit_generator


def set_bit_generator(bit_generator):
    if not isinstance(bit_generator, BitGenerator):
        raise ValueError("bit_generator must be an instantiated BitGenerator")
    _global_random_state._bit_generator = bit_generator
    _global_random_state._legacy = LegacyDistributions(bit_generator)
    _global_random_state._modern = DistributionKernels(bit_generator)


for _legacy_name in (
    "seed", "get_state", "set_state", "random_sample", "random", "rand",
    "randn", "standard_normal", "randint", "random_integers", "permutation",
    "shuffle", "choice", "uniform", "normal", "bytes", "standard_exponential",
    "exponential", "standard_gamma", "gamma", "beta", "binomial", "chisquare",
    "dirichlet", "f", "geometric", "gumbel", "hypergeometric", "laplace",
    "logistic", "lognormal", "logseries", "multinomial", "multivariate_normal",
    "negative_binomial", "noncentral_chisquare", "noncentral_f", "pareto",
    "poisson", "power", "rayleigh", "standard_cauchy", "standard_t",
    "triangular", "vonmises", "wald", "weibull", "zipf", "tomaxint",
):
    globals()[_legacy_name] = getattr(_global_random_state, _legacy_name)
ranf = sample = random_sample


__all__ = [n for n in dir() if not n.startswith("_")]
