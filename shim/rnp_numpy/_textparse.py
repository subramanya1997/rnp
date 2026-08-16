"""Parsing text elements when `array()` is given a numeric dtype.

``np.array(['1e0', '1e1'], dtype=np.float64)`` is ``array([1., 10.])`` in
numpy: with an explicit numeric dtype the constructor parses each string (or
bytes) element rather than rejecting it.  The Rust constructor only accepts
numeric elements, so the strings are converted here first.

The conversions are Python's own, which is also where numpy's error messages
come from -- ``np.array(['abc'], dtype=float)`` raises the very
``ValueError: could not convert string to float: 'abc'`` that ``float('abc')``
raises, and ``dtype=int`` on ``'1.5'`` raises ``int``'s
``invalid literal for int() with base 10: '1.5'``.
"""

import builtins as _builtins

#: What the Rust constructor says when it meets a text element.
TEXT_MSGS = ("unsupported element type in array(): str",
             "unsupported element type in array(): bytes")

_CONVERTERS = {
    "f": _builtins.float,
    "i": _builtins.int,
    "u": _builtins.int,
    "c": complex,
    "b": _builtins.bool,
}


def has_text_error(exc):
    text = str(exc)
    return any(msg in text for msg in TEXT_MSGS)


def parse_text(obj, dt):
    """`obj` with every text leaf converted for `dt`, or `None` if N/A."""
    if dt is None:
        return None
    conv = _CONVERTERS.get(dt.kind)
    if conv is None:
        return None

    def walk(x):
        if isinstance(x, (list, tuple)):
            return [walk(y) for y in x]
        if isinstance(x, (bytes, bytearray)):
            x = x.decode()
        if isinstance(x, str):
            return conv(x)
        return x

    return walk(obj)
