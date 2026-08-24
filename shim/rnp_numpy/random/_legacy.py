"""NumPy RandomState's legacy MT19937 stream and preserved distributions."""

import math
import operator
import threading

from .. import array, asarray, uint32
from ._bit_generators import BitGenerator


_MASK32 = (1 << 32) - 1


class LegacyMT19937(BitGenerator):
    def __init__(self, seed=None):
        self.lock = threading.Lock()
        self._seed_seq = None
        self._ctypes_interface = None
        self._cffi_interface = None
        self._key = [0] * 624
        self._pos = 624
        self._has_gauss = False
        self._gauss = 0.0
        self.seed(seed)

    def seed(self, seed=None):
        if seed is None:
            # NumPy uses OS entropy. This fallback is intentionally local to
            # the no-explicit-seed case; all reproducibility tests pass a seed.
            import os
            seed = int.from_bytes(os.urandom(4), "little")
        try:
            value = operator.index(seed)
        except TypeError:
            ndim = getattr(seed, "ndim", None)
            if ndim is not None and ndim != 1:
                raise ValueError("Seed array must be 1-d")
            try:
                raw_keys = list(seed)
            except TypeError:
                raise TypeError("Seed must be an integer") from None
            keys = []
            for item in raw_keys:
                try:
                    keys.append(operator.index(item))
                except TypeError:
                    if isinstance(item, (list, tuple)) or getattr(item, "ndim", 0) > 0:
                        raise ValueError("Seed array must be 1-d integer values") from None
                    raise TypeError("Seed array values must be integers") from None
            if not keys:
                raise ValueError("Seed must be non-empty")
            if any(v < 0 or v > _MASK32 for v in keys):
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            self._init_by_array(keys)
        else:
            if value < 0 or value > _MASK32:
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            current = value & _MASK32
            for pos in range(624):
                self._key[pos] = current
                current = (1812433253 * (current ^ (current >> 30)) + pos + 1) & _MASK32
            self._pos = 624
        self._has_gauss = False
        self._gauss = 0.0

    def _init_by_array(self, keys):
        current = 19650218
        for pos in range(624):
            self._key[pos] = current
            current = (1812433253 * (current ^ (current >> 30)) + pos + 1) & _MASK32
        i, j = 1, 0
        for _ in range(max(624, len(keys))):
            prev = self._key[i - 1]
            self._key[i] = ((self._key[i] ^ ((prev ^ (prev >> 30)) * 1664525))
                            + keys[j] + j) & _MASK32
            i += 1
            j += 1
            if i >= 624:
                self._key[0] = self._key[623]
                i = 1
            if j >= len(keys):
                j = 0
        for _ in range(623):
            prev = self._key[i - 1]
            self._key[i] = ((self._key[i] ^ ((prev ^ (prev >> 30)) * 1566083941))
                            - i) & _MASK32
            i += 1
            if i >= 624:
                self._key[0] = self._key[623]
                i = 1
        self._key[0] = 0x80000000
        self._pos = 624

    def _twist(self):
        for i in range(227):
            y = (self._key[i] & 0x80000000) | (self._key[i + 1] & 0x7FFFFFFF)
            self._key[i] = self._key[i + 397] ^ (y >> 1) ^ ((-(y & 1)) & 0x9908B0DF)
        for i in range(227, 623):
            y = (self._key[i] & 0x80000000) | (self._key[i + 1] & 0x7FFFFFFF)
            self._key[i] = self._key[i - 227] ^ (y >> 1) ^ ((-(y & 1)) & 0x9908B0DF)
        y = (self._key[623] & 0x80000000) | (self._key[0] & 0x7FFFFFFF)
        self._key[623] = self._key[396] ^ (y >> 1) ^ ((-(y & 1)) & 0x9908B0DF)
        self._key = [v & _MASK32 for v in self._key]
        self._pos = 0

    def next_uint32(self):
        if self._pos >= 624:
            self._twist()
        value = self._key[self._pos]
        self._pos += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & _MASK32

    def next_uint64(self):
        return (self.next_uint32() << 32) | self.next_uint32()

    def next_raw(self):
        return self.next_uint32()

    def next_double(self):
        a = self.next_uint32() >> 5
        b = self.next_uint32() >> 6
        return (a * 67108864.0 + b) / 9007199254740992.0

    random = next_double

    def uniform(self, low=0.0, high=1.0):
        return math.fma(high - low, self.next_double(), low)

    def _interval(self, maximum):
        mask = maximum
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        while True:
            value = self.next_uint32() & mask
            if value <= maximum:
                return value

    def randrange(self, start, stop=None):
        if stop is None:
            start, stop = 0, start
        return start + self._interval(stop - start - 1)

    def shuffle(self, values):
        for i in range(len(values) - 1, 0, -1):
            j = self._interval(i)
            values[i], values[j] = values[j], values[i]

    def sample(self, population, count):
        values = list(population)
        self.shuffle(values)
        return values[:count]

    def gauss(self, loc=0.0, scale=1.0):
        if self._has_gauss:
            value = self._gauss
            self._has_gauss = False
            self._gauss = 0.0
        else:
            while True:
                x1 = math.fma(2.0, self.next_double(), -1.0)
                x2 = math.fma(2.0, self.next_double(), -1.0)
                r2 = math.fma(x1, x1, x2 * x2)
                if 0.0 < r2 < 1.0:
                    break
            factor = math.sqrt(-2.0 * math.log(r2) / r2)
            self._gauss = factor * x1
            self._has_gauss = True
            value = factor * x2
        return math.fma(scale, value, loc)

    @property
    def state(self):
        return ("MT19937", array(self._key, dtype=uint32), self._pos,
                int(self._has_gauss), self._gauss)

    @state.setter
    def state(self, state):
        if state[0] != "MT19937":
            raise ValueError("state must be for MT19937")
        self._key = [operator.index(v) & _MASK32 for v in state[1]]
        self._pos = operator.index(state[2])
        self._has_gauss = bool(state[3]) if len(state) > 3 else False
        self._gauss = float(state[4]) if len(state) > 4 else 0.0


class LegacyDistributions:
    def __init__(self, bit_generator):
        self.bit_generator = bit_generator
        self.has_gauss = False
        self.gauss_value = 0.0

    def gauss(self):
        if self.has_gauss:
            self.has_gauss = False
            value = self.gauss_value
            self.gauss_value = 0.0
            return value
        while True:
            x1 = math.fma(2.0, self.bit_generator.next_double(), -1.0)
            x2 = math.fma(2.0, self.bit_generator.next_double(), -1.0)
            r2 = math.fma(x1, x1, x2 * x2)
            if 0.0 < r2 < 1.0:
                break
        factor = math.sqrt(-2.0 * math.log(r2) / r2)
        self.gauss_value = factor * x1
        self.has_gauss = True
        return factor * x2

    def standard_exponential(self):
        return -math.log(1.0 - self.bit_generator.next_double())

    def standard_gamma(self, shape):
        if shape == 1.0:
            return self.standard_exponential()
        if shape == 0.0:
            return 0.0
        if shape < 1.0:
            while True:
                u = self.bit_generator.next_double()
                v = self.standard_exponential()
                if u <= 1.0 - shape:
                    x = math.pow(u, 1.0 / shape)
                    if x <= v:
                        return x
                else:
                    y = -math.log((1.0 - u) / shape)
                    x = math.pow(math.fma(shape, y, 1.0 - shape), 1.0 / shape)
                    if x <= v + y:
                        return x
        b = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * b)
        while True:
            while True:
                x = self.gauss()
                v = math.fma(c, x, 1.0)
                if v > 0.0:
                    break
            v = v * v * v
            u = self.bit_generator.next_double()
            x2 = x * x
            if u < math.fma(-0.0331, x2 * x2, 1.0):
                return b * v
            if math.log(u) < math.fma(b, 1.0 - v + math.log(v), 0.5 * x2):
                return b * v


__all__ = ["LegacyMT19937", "LegacyDistributions"]
