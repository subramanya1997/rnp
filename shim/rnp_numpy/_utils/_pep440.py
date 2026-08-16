"""PEP 440 version parsing and ordering.

numpy ships its own copy so that version comparisons work without
`packaging`; the upstream tests only use `parse`, `Version` and the
comparison operators, which is what this implements.
"""

import itertools
import re

__all__ = ["parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN"]

VERSION_PATTERN = r"""
    v?
    (?:(?P<epoch>[0-9]+)!)?
    (?P<release>[0-9]+(?:\.[0-9]+)*)
    (?P<pre>[-_\.]?(?P<pre_l>a|b|c|rc|alpha|beta|pre|preview)[-_\.]?
        (?P<pre_n>[0-9]+)?)?
    (?P<post>(?:-(?P<post_n1>[0-9]+))|(?:[-_\.]?(?P<post_l>post|rev|r)
        [-_\.]?(?P<post_n2>[0-9]+)?))?
    (?P<dev>[-_\.]?(?P<dev_l>dev)[-_\.]?(?P<dev_n>[0-9]+)?)?
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?
"""

_REGEX = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$",
                    re.VERBOSE | re.IGNORECASE)

_PRE_ALIASES = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc",
                "preview": "rc"}
_POST_ALIASES = {"rev": "post", "r": "post"}

class _Sentinel:
    """Orders before (or after) every other comparison key.

    The sort key mixes tuples with "absent" markers, so the markers have to
    compare against tuples rather than being plain floats.
    """

    def __init__(self, name, less):
        self._name = name
        self._less = less

    def __repr__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        return isinstance(other, _Sentinel) and other._name == self._name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return True if self._less else not self.__eq__(other) and False

    def __le__(self, other):
        return self._less or self.__eq__(other)

    def __gt__(self, other):
        return (not self._less) and not self.__eq__(other)

    def __ge__(self, other):
        return (not self._less) or self.__eq__(other)


#: A missing pre-release sorts *after* one that is present (`1.0 > 1.0rc1`);
#: a missing post/dev segment sorts before/after respectively.
_NEG_INF = _Sentinel("-inf", less=True)
_INF = _Sentinel("inf", less=False)


class InvalidVersion(ValueError):
    pass


class _BaseVersion:
    _key = ()

    def __hash__(self):
        return hash(self._key)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._compare(other, lambda a, b: a != b)

    def _compare(self, other, op):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return op(self._key, other._key)


class Version(_BaseVersion):
    def __init__(self, version):
        match = _REGEX.search(str(version))
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")
        self.public = str(version).strip()
        self._epoch = int(match.group("epoch") or 0)
        self._release = tuple(int(p) for p in match.group("release").split("."))
        self._pre = _norm_letter_number(match.group("pre_l"),
                                        match.group("pre_n"), _PRE_ALIASES)
        self._post = (
            ("post", int(match.group("post_n1") or match.group("post_n2") or 0))
            if match.group("post") else None
        )
        self._dev = (("dev", int(match.group("dev_n") or 0))
                     if match.group("dev") else None)
        local = match.group("local")
        self._local = tuple(
            int(p) if p.isdigit() else p.lower()
            for p in re.split(r"[-_\.]", local)
        ) if local else None
        self._key = _cmp_key(self._epoch, self._release, self._pre,
                             self._post, self._dev, self._local)

    @property
    def base_version(self):
        return ".".join(str(p) for p in self._release)

    def __repr__(self):
        return f"<Version({self.public!r})>"

    def __str__(self):
        return self.public


class LegacyVersion(_BaseVersion):
    """A version string that is not PEP 440 compliant; sorts before all of
    them, which is what numpy's copy does."""

    def __init__(self, version):
        self.public = str(version)
        self._key = (-1, tuple(self.public.split(".")), _NEG_INF, _NEG_INF,
                     _NEG_INF, _NEG_INF)

    def __repr__(self):
        return f"<LegacyVersion({self.public!r})>"

    def __str__(self):
        return self.public


def _norm_letter_number(letter, number, aliases):
    if letter is None and number is None:
        return None
    letter = aliases.get((letter or "").lower(), (letter or "").lower())
    return (letter, int(number or 0))


def _cmp_key(epoch, release, pre, post, dev, local):
    # Trailing zeros in the release segment are not significant.
    trimmed = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )
    if pre is None and post is None and dev is not None:
        pre_key = _NEG_INF          # 1.0.dev0 < 1.0a0
    elif pre is None:
        pre_key = _INF              # 1.0 > 1.0rc1
    else:
        pre_key = pre
    post_key = _NEG_INF if post is None else post
    dev_key = _INF if dev is None else dev
    local_key = _NEG_INF if local is None else tuple(
        (i, "") if isinstance(i, int) else (-1, i) for i in local
    )
    return (epoch, trimmed, pre_key, post_key, dev_key, local_key)


def parse(version):
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)
