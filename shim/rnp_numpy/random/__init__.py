"""`numpy.random` — enough of the API for the upstream core tests to run.

These generators are *not* bit-compatible with numpy's MT19937/PCG64 streams
(that is its own milestone); they are ordinary pseudo-random values with the
right shapes and dtypes, which is all the `_core` tests use them for.
"""

import random as _random

from .. import arange, array, asarray, empty, float64, int_, zeros

_rng = _random.Random()


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
    return _rng.getstate()


def set_state(state):
    _rng.setstate(state)


def _shape(size):
    if size is None:
        return ()
    if isinstance(size, int):
        return (size,)
    return tuple(size)


def _fill(shape, gen, dtype=float64):
    n = 1
    for d in shape:
        n *= d
    out = empty(shape, dtype)
    flat = out.flat
    for i in range(n):
        flat[i] = gen()
    return out


def random_sample(size=None):
    shape = _shape(size)
    if shape == ():
        return _rng.random()
    return _fill(shape, _rng.random)


random = random_sample
ranf = random_sample
sample = random_sample


def rand(*args):
    return random_sample(args if args else None)


def randn(*args):
    shape = args if args else None
    shape = _shape(shape)
    if shape == ():
        return _rng.gauss(0.0, 1.0)
    return _fill(shape, lambda: _rng.gauss(0.0, 1.0))


def standard_normal(size=None):
    return randn(*(_shape(size)))


def randint(low, high=None, size=None, dtype=int_):
    if high is None:
        low, high = 0, low
    shape = _shape(size)
    if shape == ():
        return _rng.randrange(low, high)
    return _fill(shape, lambda: _rng.randrange(low, high), dtype)


random_integers = randint


def permutation(x):
    if isinstance(x, int):
        items = list(range(x))
    else:
        items = asarray(x).tolist()
    _rng.shuffle(items)
    return array(items)


def shuffle(x):
    n = len(x)
    for i in range(n - 1, 0, -1):
        j = _rng.randrange(i + 1)
        tmp = x[i].copy() if hasattr(x[i], "copy") else x[i]
        x[i] = x[j]
        x[j] = tmp


def choice(a, size=None, replace=True, p=None):
    if isinstance(a, int):
        a = arange(a)
    a = asarray(a)
    shape = _shape(size)
    n = 1
    for d in shape:
        n *= d
    picks = [_rng.randrange(a.size) for _ in range(n)]
    out = a[array(picks)]
    return out.reshape(shape) if shape else out[0] if False else out.reshape(shape)


def uniform(low=0.0, high=1.0, size=None):
    shape = _shape(size)
    if shape == ():
        return _rng.uniform(low, high)
    return _fill(shape, lambda: _rng.uniform(low, high))


def normal(loc=0.0, scale=1.0, size=None):
    shape = _shape(size)
    if shape == ():
        return _rng.gauss(loc, scale)
    return _fill(shape, lambda: _rng.gauss(loc, scale))


class RandomState:
    def __init__(self, s=None):
        self._r = _random.Random(s)

    def seed(self, s=None):
        self._r.seed(s)

    random_sample = staticmethod(random_sample)
    rand = staticmethod(rand)
    randn = staticmethod(randn)
    randint = staticmethod(randint)
    permutation = staticmethod(permutation)
    shuffle = staticmethod(shuffle)
    uniform = staticmethod(uniform)
    normal = staticmethod(normal)
    choice = staticmethod(choice)


class Generator:
    """A stand-in for `np.random.Generator` (not stream-compatible)."""

    def __init__(self, seed=None):
        self._r = _random.Random(seed)

    def random(self, size=None, dtype=float64):
        shape = _shape(size)
        if shape == ():
            return self._r.random()
        return _fill(shape, self._r.random, dtype)

    def integers(self, low, high=None, size=None, dtype=int_, endpoint=False):
        if high is None:
            low, high = 0, low
        hi = high + 1 if endpoint else high
        shape = _shape(size)
        if shape == ():
            return self._r.randrange(low, hi)
        return _fill(shape, lambda: self._r.randrange(low, hi), dtype)

    def standard_normal(self, size=None, dtype=float64):
        shape = _shape(size)
        if shape == ():
            return self._r.gauss(0.0, 1.0)
        return _fill(shape, lambda: self._r.gauss(0.0, 1.0), dtype)

    normal = normal
    uniform = uniform
    permutation = staticmethod(permutation)
    shuffle = staticmethod(shuffle)
    choice = staticmethod(choice)


def default_rng(seed=None):
    return Generator(seed)


class PCG64:
    def __init__(self, seed=None):
        self.seed = seed


class MT19937:
    def __init__(self, seed=None):
        self.seed = seed


__all__ = [n for n in dir() if not n.startswith("_")]
