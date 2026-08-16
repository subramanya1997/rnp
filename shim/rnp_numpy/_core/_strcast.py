"""Casting numeric arrays to the string dtypes (`U`/`S`).

The engine can fill a `U`/`S` array from `str`/`bytes` but not from numbers,
so `np.array([1.5], dtype=str)` and `arr.astype(str)` both fail.  numpy
supports both, and the conversion is entirely a formatting question, so it
lives here rather than in Rust.

Two things have to match numpy exactly:

*   **The element text.**  It is `str()` of the *scalar*, not of a Python
    float — `str(np.float32(1.1))` is ``'1.1'`` (shortest repr that round
    trips in that precision) while `str(float(np.float32(1.1)))` is
    ``'1.100000023841858'``.  So elements are formatted through the port's
    own scalar objects.

*   **The itemsize.**  When the target dtype is unsized (`dtype=str`), numpy
    picks a width per source dtype that is wide enough for any value of that
    type.  Those widths are the table below; they are not derived from the
    data, so an array of small numbers still lands in `<U32` if it is float.
"""

__all__ = [
    "width_for", "to_string_array", "is_string_fill_error",
    "char_count", "restring",
]

#: Width numpy gives an unsized `U`/`S` target, per source dtype name.
_WIDTH = {
    "bool": 5,
    "int8": 4, "int16": 6, "int32": 11, "int64": 21,
    "byte": 4, "short": 6, "intc": 11, "long": 21, "longlong": 21, "intp": 21,
    "uint8": 3, "uint16": 5, "uint32": 10, "uint64": 20,
    "ubyte": 3, "ushort": 5, "uintc": 10, "ulong": 20, "ulonglong": 20,
    "uintp": 20,
    "float16": 32, "float32": 32, "float64": 32, "longdouble": 32,
    "half": 32, "single": 32, "double": 32,
    "complex64": 64, "complex128": 64, "clongdouble": 64,
    "csingle": 64, "cdouble": 64,
}

#: Substrings of the engine's error when a numeric value reaches a `U`/`S`
#: array. Matching on this keeps the fallback narrow.
_FILL_ERRORS = (
    "only str and bytes can fill a str ('U') array",
    "only str and bytes can fill a bytes ('S') array",
)


def is_string_fill_error(exc):
    """True if `exc` is the engine refusing a number for a `U`/`S` array."""
    return any(m in str(exc) for m in _FILL_ERRORS)


def width_for(dtype):
    """Unsized-target width numpy would use for a source `dtype`, or None."""
    return _WIDTH.get(getattr(dtype, "name", None))


def _elements(arr):
    """Nested lists of `str()`/`bytes` per element, preserving shape."""
    if arr.ndim == 0:
        return str(arr[()])
    return [_elements(arr[i]) for i in range(arr.shape[0])]


def to_string_array(obj, dtype, kind, itemsize=None):
    """Build a `U`/`S` array of `obj`'s values rendered as text.

    `kind` is ``'U'`` or ``'S'``.  `itemsize` is the requested character
    count, or None to use numpy's per-dtype default width.
    """
    from .. import asarray

    src = obj if hasattr(obj, "dtype") else asarray(obj)
    if itemsize is None:
        itemsize = width_for(src.dtype)
        if itemsize is None:
            raise TypeError(
                f"cannot cast {src.dtype} to a string dtype"
            )

    text = _elements(src)
    if kind == "S":
        def encode(v):
            if isinstance(v, list):
                return [encode(x) for x in v]
            return v.encode("ascii")
        text = encode(text)

    from .. import array as _array
    return _array(text, f"{kind}{itemsize}")


def char_count(dtype):
    """Characters a string dtype holds (`U` stores 4 bytes per character)."""
    return dtype.itemsize // (4 if dtype.kind == "U" else 1)


def restring(src, dt):
    """Cast between the string dtypes: `S`<->`U` and either one resized.

    Probed against numpy 2.5.2: the result is truncated to the target's
    character count; `U`->`S` encodes as ASCII and `S`->`U` decodes as ASCII,
    both raising the ordinary `UnicodeEncodeError`/`UnicodeDecodeError` on
    anything outside that range.  An unsized target (`astype('U')`) keeps the
    source's *character* count -- not its byte count, which is what makes
    `U5 -> U` stay `U5` rather than widening to `U20`.
    """
    kind = dt.kind
    n = char_count(dt) or char_count(src.dtype) or 1

    def conv(v):
        if isinstance(v, list):
            return [conv(x) for x in v]
        if kind == "U":
            v = v.decode("ascii") if isinstance(v, bytes) else v
        else:
            v = v.encode("ascii") if isinstance(v, str) else v
        return v[:n]

    from .. import array as _array
    return _array(conv(src.tolist()), f"{kind}{n}")
