"""Bit-exact core random-number engines used by :mod:`rnp_numpy.random`."""

import copy
import ctypes as _ctypes
import operator
import threading

from .. import array, uint32, uint64
from .bit_generator import ISeedSequence, ISpawnableSeedSequence, SeedSequence
from ._common import interface


_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1
_MASK128 = (1 << 128) - 1


class BitGenerator:
    """Python implementation of NumPy's public BitGenerator contract."""

    def __init__(self, seed=None):
        if not isinstance(seed, ISeedSequence):
            seed = SeedSequence(seed)
        self._seed_seq = seed
        self.lock = threading.Lock()
        self._ctypes_interface = None
        self._cffi_interface = None

    @property
    def seed_seq(self):
        return self._seed_seq

    def spawn(self, n_children):
        n_children = operator.index(n_children)
        if n_children < 0:
            raise ValueError("n_children must be non-negative")
        if not isinstance(self._seed_seq, ISpawnableSeedSequence):
            raise TypeError("The underlying SeedSequence does not implement spawning.")
        return [type(self)(seed=seq) for seq in self._seed_seq.spawn(n_children)]

    def random_raw(self, size=None, output=True):
        if size is None:
            value = self.next_raw()
            return value if output else None
        try:
            shape = (operator.index(size),)
        except TypeError:
            shape = tuple(operator.index(v) for v in size)
        count = 1
        for dim in shape:
            if dim < 0:
                raise ValueError("negative dimensions are not allowed")
            count *= dim
        values = [self.next_raw() for _ in range(count)]
        if not output:
            return None
        return array(values, dtype=uint64).reshape(shape)

    def next_double(self):
        return (self.next_uint64() >> 11) * (1.0 / 9007199254740992.0)

    def next_raw(self):
        return self.next_uint64()

    def _benchmark(self, count, method="uint64"):
        count = operator.index(count)
        if method == "uint64":
            for _ in range(count):
                self.next_uint64()
        elif method == "double":
            for _ in range(count):
                self.next_double()
        else:
            raise ValueError("Unknown method")

    @property
    def ctypes(self):
        if self._ctypes_interface is None:
            self._ctypes_interface = interface(
                id(self), _ctypes.c_void_p(id(self)), None, None, None, None
            )
        return self._ctypes_interface

    @property
    def cffi(self):
        if self._cffi_interface is None:
            self._cffi_interface = interface(id(self), None, None, None, None, None)
        return self._cffi_interface

    def __getstate__(self):
        return self.state, self._seed_seq

    def __setstate__(self, value):
        state, seed_seq = value
        self.lock = threading.Lock()
        self._ctypes_interface = None
        self._cffi_interface = None
        self._seed_seq = seed_seq
        self.state = state

    def __repr__(self):
        return f"{type(self).__module__.replace('rnp_numpy', 'numpy')}.{type(self).__name__}"

    __str__ = __repr__


def _rotr64(value, rotation):
    rotation &= 63
    return ((value >> rotation) | (value << ((-rotation) & 63))) & _MASK64


def _advance_lcg(state, delta, multiplier, increment, mask):
    delta &= mask
    acc_mult = 1
    acc_plus = 0
    cur_mult = multiplier
    cur_plus = increment
    while delta:
        if delta & 1:
            acc_mult = (acc_mult * cur_mult) & mask
            acc_plus = (acc_plus * cur_mult + cur_plus) & mask
        cur_plus = ((cur_mult + 1) * cur_plus) & mask
        cur_mult = (cur_mult * cur_mult) & mask
        delta >>= 1
    return (acc_mult * state + acc_plus) & mask


class _PCG64Base(BitGenerator):
    _seed_multiplier = (2549297995355413924 << 64) | 4865540595714422341
    _multiplier = _seed_multiplier
    _jump = 0x9E3779B97F4A7C15F39CC0605CEDC835

    def __init__(self, seed=None):
        super().__init__(seed)
        words = [int(v) for v in self._seed_seq.generate_state(4, uint64)]
        initstate = (words[0] << 64) | words[1]
        initseq = (words[2] << 64) | words[3]
        self._state = 0
        self._inc = ((initseq << 1) | 1) & _MASK128
        self._state = (self._state * self._seed_multiplier + self._inc) & _MASK128
        self._state = (self._state + initstate) & _MASK128
        self._state = (self._state * self._seed_multiplier + self._inc) & _MASK128
        self._has_uint32 = 0
        self._uinteger = 0

    def _step(self):
        self._state = (
            self._state * self._multiplier + self._inc
        ) & _MASK128

    def next_uint32(self):
        if self._has_uint32:
            self._has_uint32 = 0
            return self._uinteger
        value = self.next_uint64()
        self._has_uint32 = 1
        self._uinteger = (value >> 32) & _MASK32
        return value & _MASK32

    @property
    def state(self):
        return {
            "bit_generator": type(self).__name__,
            "state": {"state": self._state, "inc": self._inc},
            "has_uint32": self._has_uint32,
            "uinteger": self._uinteger,
        }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        if value.get("bit_generator", "") != type(self).__name__:
            raise ValueError(f"state must be for a {type(self).__name__} RNG")
        self._state = operator.index(value["state"]["state"]) & _MASK128
        self._inc = operator.index(value["state"]["inc"]) & _MASK128
        self._has_uint32 = operator.index(value["has_uint32"])
        self._uinteger = operator.index(value["uinteger"]) & _MASK32

    def advance(self, delta):
        self._state = _advance_lcg(
            self._state,
            operator.index(delta),
            self._multiplier,
            self._inc,
            _MASK128,
        )
        self._has_uint32 = 0
        self._uinteger = 0
        return self

    def jumped(self, jumps=1):
        other = type(self)()
        other._seed_seq = copy.deepcopy(self._seed_seq)
        other.state = self.state
        other.advance(self._jump * operator.index(jumps))
        return other


class PCG64(_PCG64Base):
    def next_uint64(self):
        self._step()
        high = self._state >> 64
        low = self._state & _MASK64
        return _rotr64(high ^ low, high >> 58)


class PCG64DXSM(_PCG64Base):
    _multiplier = 0xDA942042E4DD58B5

    def next_uint64(self):
        high = self._state >> 64
        low = (self._state & _MASK64) | 1
        high ^= high >> 32
        high = (high * 0xDA942042E4DD58B5) & _MASK64
        high ^= high >> 48
        result = (high * low) & _MASK64
        self._step()
        return result


class MT19937(BitGenerator):
    def __init__(self, seed=None):
        super().__init__(seed)
        words = [int(v) for v in self._seed_seq.generate_state(624, uint32)]
        self._key = [0x80000000] + words[1:]
        # This is deliberately 623, matching NumPy's Cython loop variable
        # after ``for i in range(1, RK_STATE_LEN)``.
        self._pos = 623

    def _twist(self):
        key = self._key
        for i in range(624 - 397):
            y = (key[i] & 0x80000000) | (key[i + 1] & 0x7FFFFFFF)
            key[i] = key[i + 397] ^ (y >> 1) ^ ((-(y & 1)) & 0x9908B0DF)
        for i in range(624 - 397, 623):
            y = (key[i] & 0x80000000) | (key[i + 1] & 0x7FFFFFFF)
            key[i] = key[i - 227] ^ (y >> 1) ^ ((-(y & 1)) & 0x9908B0DF)
        y = (key[623] & 0x80000000) | (key[0] & 0x7FFFFFFF)
        key[623] = key[396] ^ (y >> 1) ^ ((-(y & 1)) & 0x9908B0DF)
        self._key = [v & _MASK32 for v in key]
        self._pos = 0

    def next_uint32(self):
        if self._pos == 624:
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

    @staticmethod
    def _jump_gen_next(key, pos):
        if pos < 227:
            other = pos + 397
            next_pos = pos + 1
            following = pos + 1
        elif pos < 623:
            other = pos - 227
            next_pos = pos + 1
            following = pos + 1
        else:
            other = 396
            next_pos = 0
            following = 0
        y = (key[pos] & 0x80000000) | (key[following] & 0x7FFFFFFF)
        key[pos] = (key[other] ^ (y >> 1) ^ ((-(y & 1)) & 0x9908B0DF)) & _MASK32
        return next_pos

    @staticmethod
    def _jump_add(target_key, target_pos, source_key, source_pos):
        for offset in range(624):
            target = (target_pos + offset) % 624
            source = (source_pos + offset) % 624
            target_key[target] ^= source_key[source]

    def _jump_once(self):
        from ._mt_jump import POLY_COEF

        source_key = list(self._key)
        source_pos = 0 if self._pos >= 624 else self._pos
        index = 19936
        while not ((POLY_COEF[index >> 5] >> (index & 31)) & 1):
            index -= 1
        temp_key = list(source_key)
        temp_pos = self._jump_gen_next(temp_key, source_pos)
        index -= 1
        while index > 0:
            if (POLY_COEF[index >> 5] >> (index & 31)) & 1:
                self._jump_add(temp_key, temp_pos, source_key, source_pos)
            temp_pos = self._jump_gen_next(temp_key, temp_pos)
            index -= 1
        if POLY_COEF[0] & 1:
            self._jump_add(temp_key, temp_pos, source_key, source_pos)
        self._key = [value & _MASK32 for value in temp_key]
        self._pos = temp_pos

    def jumped(self, jumps=1):
        jumps = operator.index(jumps)
        other = type(self)()
        other._seed_seq = copy.deepcopy(self._seed_seq)
        other.state = self.state
        for _ in range(jumps):
            other._jump_once()
        return other

    @property
    def state(self):
        return {
            "bit_generator": "MT19937",
            "state": {
                "key": array(self._key, dtype=uint32),
                "pos": self._pos,
            },
        }

    @state.setter
    def state(self, value):
        if isinstance(value, tuple) and len(value) >= 2:
            if isinstance(value[0], str):
                key, pos = value[1:3]
            else:
                key, pos = value[:2]
        else:
            if not isinstance(value, dict):
                raise TypeError("state must be a dict or a tuple")
            if value.get("bit_generator", "MT19937") != "MT19937":
                raise ValueError("state must be for a MT19937 RNG")
            key = value["state"]["key"]
            pos = value["state"]["pos"]
        key = [operator.index(v) for v in key]
        if len(key) != 624:
            raise ValueError("state vector is the wrong size")
        self._key = [v & _MASK32 for v in key]
        self._pos = operator.index(pos)


def _int_to_words(value, count, name):
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        words = [operator.index(v) for v in value]
        if len(words) != count:
            raise ValueError(f"{name} must have {count} elements when using array form")
        if any(v < 0 or v > _MASK64 for v in words):
            raise ValueError(f"{name} values must be between 0 and 2**64 - 1")
        return words
    value = operator.index(value)
    if value < 0 or value >= 1 << (64 * count):
        raise ValueError(f"{name} must be between 0 and 2**{64 * count} - 1")
    return [(value >> (64 * i)) & _MASK64 for i in range(count)]


class Philox(BitGenerator):
    def __init__(self, seed=None, counter=None, key=None):
        if seed is not None and key is not None:
            raise ValueError("seed and key cannot be both used")
        super().__init__(seed)
        if key is None:
            self._key = [int(v) for v in self._seed_seq.generate_state(2, uint64)]
        else:
            self._key = _int_to_words(key, 2, "key")
            self._seed_seq = None
        self._counter = _int_to_words(0 if counter is None else counter, 4, "counter")
        self._buffer = [0, 0, 0, 0]
        self._buffer_pos = 4
        self._has_uint32 = 0
        self._uinteger = 0

    @staticmethod
    def _round(counter, key):
        product0 = 0xD2E7470EE14C6C93 * counter[0]
        product1 = 0xCA5A826395121157 * counter[2]
        lo0, hi0 = product0 & _MASK64, (product0 >> 64) & _MASK64
        lo1, hi1 = product1 & _MASK64, (product1 >> 64) & _MASK64
        return [
            (hi1 ^ counter[1] ^ key[0]) & _MASK64,
            lo1,
            (hi0 ^ counter[3] ^ key[1]) & _MASK64,
            lo0,
        ]

    @classmethod
    def _generate_block(cls, counter, key):
        counter = list(counter)
        round_key = list(key)
        for round_no in range(10):
            counter = cls._round(counter, round_key)
            if round_no != 9:
                round_key[0] = (round_key[0] + 0x9E3779B97F4A7C15) & _MASK64
                round_key[1] = (round_key[1] + 0xBB67AE8584CAA73B) & _MASK64
        return counter

    def _increment_counter(self, amount=1):
        carry = amount
        for i in range(4):
            total = self._counter[i] + (carry & _MASK64)
            self._counter[i] = total & _MASK64
            carry = (carry >> 64) + (total >> 64)

    def next_uint64(self):
        if self._buffer_pos < 4:
            value = self._buffer[self._buffer_pos]
            self._buffer_pos += 1
            return value
        self._increment_counter()
        self._buffer = self._generate_block(self._counter, self._key)
        self._buffer_pos = 1
        return self._buffer[0]

    def next_uint32(self):
        if self._has_uint32:
            self._has_uint32 = 0
            return self._uinteger
        value = self.next_uint64()
        self._has_uint32 = 1
        self._uinteger = (value >> 32) & _MASK32
        return value & _MASK32

    @property
    def state(self):
        return {
            "bit_generator": "Philox",
            "state": {
                "counter": array(self._counter, dtype=uint64),
                "key": array(self._key, dtype=uint64),
            },
            "buffer": array(self._buffer, dtype=uint64),
            "buffer_pos": self._buffer_pos,
            "has_uint32": self._has_uint32,
            "uinteger": self._uinteger,
        }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        if value.get("bit_generator", "") != "Philox":
            raise ValueError("state must be for a Philox PRNG")
        self._counter = [operator.index(v) & _MASK64 for v in value["state"]["counter"]]
        self._key = [operator.index(v) & _MASK64 for v in value["state"]["key"]]
        self._buffer = [operator.index(v) & _MASK64 for v in value["buffer"]]
        self._buffer_pos = operator.index(value["buffer_pos"])
        self._has_uint32 = operator.index(value["has_uint32"])
        self._uinteger = operator.index(value["uinteger"]) & _MASK32

    def advance(self, delta):
        self._increment_counter(operator.index(delta) & ((1 << 256) - 1))
        self._buffer = [0, 0, 0, 0]
        self._buffer_pos = 4
        self._has_uint32 = 0
        self._uinteger = 0
        return self

    def jumped(self, jumps=1):
        other = type(self)()
        other._seed_seq = copy.deepcopy(self._seed_seq)
        other.state = self.state
        other.advance(operator.index(jumps) << 128)
        return other


class SFC64(BitGenerator):
    def __init__(self, seed=None):
        super().__init__(seed)
        words = [int(v) for v in self._seed_seq.generate_state(3, uint64)]
        self._s = words + [1]
        self._has_uint32 = 0
        self._uinteger = 0
        for _ in range(12):
            self.next_uint64()

    def next_uint64(self):
        a, b, c, counter = self._s
        result = (a + b + counter) & _MASK64
        self._s[3] = (counter + 1) & _MASK64
        self._s[0] = b ^ (b >> 11)
        self._s[1] = (c + (c << 3)) & _MASK64
        rotated = ((c << 24) | (c >> 40)) & _MASK64
        self._s[2] = (rotated + result) & _MASK64
        return result

    def next_uint32(self):
        if self._has_uint32:
            self._has_uint32 = 0
            return self._uinteger
        value = self.next_uint64()
        self._has_uint32 = 1
        self._uinteger = (value >> 32) & _MASK32
        return value & _MASK32

    @property
    def state(self):
        return {
            "bit_generator": "SFC64",
            "state": {"state": array(self._s, dtype=uint64)},
            "has_uint32": self._has_uint32,
            "uinteger": self._uinteger,
        }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        if value.get("bit_generator", "") != "SFC64":
            raise ValueError("state must be for a SFC64 RNG")
        self._s = [operator.index(v) & _MASK64 for v in value["state"]["state"]]
        self._has_uint32 = operator.index(value["has_uint32"])
        self._uinteger = operator.index(value["uinteger"]) & _MASK32


__all__ = [
    "BitGenerator", "MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"
]
