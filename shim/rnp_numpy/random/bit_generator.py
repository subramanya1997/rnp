"""Seed material handling and base interfaces for :mod:`numpy.random`.

The mixing operations here are a direct integer transcription of NumPy's
``random/bit_generator.pyx``.  Every arithmetic operation is explicitly
reduced to 32 bits, matching Cython's ``uint32_t`` overflow semantics.
"""

import abc
import operator
import secrets

from .. import array, uint32, uint64


_MASK32 = (1 << 32) - 1
_INIT_A = 0x43B0D7E5
_MULT_A = 0x931E8875
_INIT_B = 0x8B51F9DD
_MULT_B = 0x58F38DED
_MIX_MULT_L = 0xCA01F9DD
_MIX_MULT_R = 0x4973F715
_XSHIFT = 16
_DEFAULT_POOL_SIZE = 4


def _u32(value):
    return value & _MASK32


def _int_to_uint32_words(value):
    value = operator.index(value)
    if value < 0:
        raise ValueError("expected non-negative integer")
    if value == 0:
        return [0]
    words = []
    while value:
        words.append(value & _MASK32)
        value >>= 32
    return words


def _coerce_to_uint32_words(value, _seen=None):
    """Normalize entropy exactly as NumPy's ``_coerce_to_uint32_array``."""
    if isinstance(value, str):
        if value.startswith("0x"):
            return _int_to_uint32_words(int(value, 16))
        if value.isdecimal():
            return _int_to_uint32_words(int(value, 10))
        raise ValueError("unrecognized seed string")

    try:
        return _int_to_uint32_words(value)
    except TypeError:
        pass

    if isinstance(value, float):
        raise TypeError("seed must be integer")

    if _seen is None:
        _seen = set()
    ident = id(value)
    if ident in _seen:
        raise TypeError("SeedSequence does not accept nested sequences.")
    _seen.add(ident)
    try:
        values = list(value)
    except TypeError:
        raise TypeError("seed must be integer") from None
    finally:
        _seen.discard(ident)

    words = []
    for item in values:
        if not isinstance(item, str) and hasattr(item, "__len__"):
            raise TypeError("SeedSequence does not accept nested sequences.")
        words.extend(_coerce_to_uint32_words(item, _seen))
    return words


def _hashmix(value, hash_const):
    value = _u32(value ^ hash_const)
    hash_const = _u32(hash_const * _MULT_A)
    value = _u32(value * hash_const)
    value ^= value >> _XSHIFT
    return _u32(value), hash_const


def _mix(x, y):
    result = _u32(_MIX_MULT_L * x - _MIX_MULT_R * y)
    return _u32(result ^ (result >> _XSHIFT))


class ISeedSequence(abc.ABC):
    @abc.abstractmethod
    def generate_state(self, n_words, dtype=uint32):
        raise NotImplementedError


class ISpawnableSeedSequence(ISeedSequence):
    @abc.abstractmethod
    def spawn(self, n_children):
        raise NotImplementedError


class SeedlessSeedSequence(ISpawnableSeedSequence):
    def generate_state(self, n_words, dtype=uint32):
        raise NotImplementedError("seedless SeedSequences cannot generate state")

    def spawn(self, n_children):
        n_children = operator.index(n_children)
        if n_children < 0:
            raise ValueError("n_children must be non-negative")
        return [self] * n_children


class SeedSequence(ISpawnableSeedSequence):
    def __init__(self, entropy=None, *, spawn_key=(), pool_size=4,
                 n_children_spawned=0):
        pool_size = operator.index(pool_size)
        if pool_size < _DEFAULT_POOL_SIZE:
            raise ValueError(
                "The size of the entropy pool should be at least "
                f"{_DEFAULT_POOL_SIZE}"
            )
        if entropy is None:
            entropy = secrets.randbits(pool_size * 32)
        elif isinstance(entropy, SeedSequence):
            raise TypeError(
                "SeedSequence expects int or sequence of ints for entropy "
                f"not {entropy}"
            )

        self.entropy = entropy
        self.spawn_key = tuple(spawn_key)
        self.pool_size = pool_size
        self.n_children_spawned = operator.index(n_children_spawned)
        assembled = _coerce_to_uint32_words(entropy)
        spawn_words = _coerce_to_uint32_words(self.spawn_key)
        if spawn_words and len(assembled) < pool_size:
            assembled += [0] * (pool_size - len(assembled))
        assembled += spawn_words
        self.pool = self._mix_entropy(assembled)

    def _mix_entropy(self, entropy):
        mixer = [0] * self.pool_size
        hash_const = _INIT_A
        for i in range(self.pool_size):
            mixer[i], hash_const = _hashmix(
                entropy[i] if i < len(entropy) else 0, hash_const
            )
        for src in range(self.pool_size):
            for dst in range(self.pool_size):
                if src != dst:
                    hashed, hash_const = _hashmix(mixer[src], hash_const)
                    mixer[dst] = _mix(mixer[dst], hashed)
        for src in range(self.pool_size, len(entropy)):
            for dst in range(self.pool_size):
                hashed, hash_const = _hashmix(entropy[src], hash_const)
                mixer[dst] = _mix(mixer[dst], hashed)
        return mixer

    @property
    def state(self):
        return {
            "entropy": self.entropy,
            "spawn_key": self.spawn_key,
            "pool_size": self.pool_size,
            "n_children_spawned": self.n_children_spawned,
        }

    def generate_state(self, n_words, dtype=uint32):
        n_words = operator.index(n_words)
        dtype_name = str(dtype)
        is_u64 = dtype is uint64 or dtype_name in ("uint64", "<u8", "=u8")
        is_u32 = dtype is uint32 or dtype_name in ("uint32", "<u4", "=u4")
        if not (is_u32 or is_u64):
            raise ValueError("only support uint32 or uint64")
        count = n_words * 2 if is_u64 else n_words
        words = []
        hash_const = _INIT_B
        for i in range(count):
            value = self.pool[i % self.pool_size] ^ hash_const
            hash_const = _u32(hash_const * _MULT_B)
            value = _u32(value * hash_const)
            value ^= value >> _XSHIFT
            words.append(_u32(value))
        if is_u64:
            values = [words[i] | (words[i + 1] << 32)
                      for i in range(0, count, 2)]
            return array(values, dtype=uint64)
        return array(words, dtype=uint32)

    def spawn(self, n_children):
        n_children = operator.index(n_children)
        if n_children < 0:
            raise ValueError("n_children must be non-negative")
        children = [
            type(self)(
                self.entropy,
                spawn_key=self.spawn_key + (i,),
                pool_size=self.pool_size,
            )
            for i in range(
                self.n_children_spawned,
                self.n_children_spawned + n_children,
            )
        ]
        self.n_children_spawned += n_children
        return children

    def __repr__(self):
        lines = [f"{type(self).__name__}(", f"    entropy={self.entropy!r},"]
        if self.spawn_key:
            lines.append(f"    spawn_key={self.spawn_key!r},")
        if self.pool_size != _DEFAULT_POOL_SIZE:
            lines.append(f"    pool_size={self.pool_size!r},")
        if self.n_children_spawned:
            lines.append(f"    n_children_spawned={self.n_children_spawned!r},")
        lines.append(")")
        return "\n".join(lines)


__all__ = [
    "ISeedSequence",
    "ISpawnableSeedSequence",
    "SeedlessSeedSequence",
    "SeedSequence",
]
