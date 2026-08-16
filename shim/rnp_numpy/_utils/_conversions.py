"""str/bytes coercion helpers."""

__all__ = ["asbytes", "asunicode"]


def asunicode(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)


def asbytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')
