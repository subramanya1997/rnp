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
        return (_operator.index(size),)
    except TypeError:
        pass
    try:
        items = tuple(size)
    except TypeError:
        raise _bad_size(size) from None
    # A non-integer *element* propagates `__index__`'s own TypeError
    # ("'float' object cannot be interpreted as an integer"), which is what
    # numpy does; only a non-sequence gets the "expected a sequence" message.
    return tuple(_operator.index(d) for d in items)


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
        pool = asarray(a)
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
        weights = [float(w) for w in asarray(p).tolist()]
        if len(weights) != pop:
            raise ValueError("'a' and 'p' must have same size")
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
    """Sequential weighted sampling without replacement (numpy does the same
    renormalise-and-redraw; only the stream differs)."""
    remaining = list(weights)
    picks = []
    for _ in range(n):
        total = sum(remaining)
        u = r.random() * total
        running = 0.0
        idx = len(remaining) - 1
        for i, w in enumerate(remaining):
            running += w
            if u < running:
                idx = i
                break
        while remaining[idx] == 0.0:
            idx = (idx + 1) % len(remaining)
        picks.append(idx)
        remaining[idx] = 0.0
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

    def __init__(self, s=None):
        self._r = _random.Random(s)

    def seed(self, s=None):
        self._r.seed(s)

    def random_sample(self, size=None):
        return _random_sample(self._r, size)

    def random(self, size=None):
        return _random_sample(self._r, size)

    def rand(self, *args):
        return _random_sample(self._r, args if args else None)

    def randn(self, *args):
        return _randn(self._r, *args)

    def standard_normal(self, size=None):
        return _standard_normal(self._r, size)

    def randint(self, low, high=None, size=None, dtype=int_):
        return _randint(self._r, low, high, size, dtype)

    def random_integers(self, low, high=None, size=None):
        return _random_integers(self._r, low, high, size)

    def permutation(self, x):
        return _permutation(self._r, x)

    def shuffle(self, x):
        return _shuffle(self._r, x)

    def choice(self, a, size=None, replace=True, p=None):
        return _choice(self._r, a, size, replace, p)

    def uniform(self, low=0.0, high=1.0, size=None):
        return _uniform(self._r, low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return _normal(self._r, loc, scale, size)

    def tomaxint(self, size=None):
        return _randint(self._r, 0, 2 ** 63 - 1, size)


class Generator:
    """A stand-in for `np.random.Generator` (not stream-compatible)."""

    def __init__(self, seed=None):
        self._r = _random.Random(seed)

    def random(self, size=None, dtype=float64):
        if size is None:
            return self._r.random()
        return _fill(_shape(size), self._r.random, dtype)

    def integers(self, low, high=None, size=None, dtype=int_, endpoint=False):
        low = _operator.index(low)
        if high is None:
            low, high = 0, low
        else:
            high = _operator.index(high)
        hi = high + 1 if endpoint else high
        if size is None:
            return self._r.randrange(low, hi)
        return _fill(_shape(size), lambda: self._r.randrange(low, hi), dtype)

    def standard_normal(self, size=None, dtype=float64):
        if size is None:
            return self._r.gauss(0.0, 1.0)
        return _fill(_shape(size), lambda: self._r.gauss(0.0, 1.0), dtype)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return _normal(self._r, loc, scale, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return _uniform(self._r, low, high, size)

    def permutation(self, x):
        return _permutation(self._r, x)

    def shuffle(self, x):
        return _shuffle(self._r, x)

    def choice(self, a, size=None, replace=True, p=None):
        return _choice(self._r, a, size, replace, p)


def default_rng(seed=None):
    return Generator(seed)


class PCG64:
    def __init__(self, seed=None):
        self.seed = seed


class MT19937:
    def __init__(self, seed=None):
        self.seed = seed


__all__ = [n for n in dir() if not n.startswith("_")]
