"""Text I/O: ``fromstring``, ``fromfile``, ``loadtxt``, ``genfromtxt``,
``savetxt`` (and the ``ndarray.tofile`` method they round-trip with).

``fromstring``/``fromfile`` are ports of the C machinery in
``numpy/_core/src/multiarray/ctors.c`` (``array_from_text``, together with
``fromstr_next_element``/``fromstr_skip_separator`` for strings and
``fromfile_next_element``/``fromfile_skip_separator`` for files) and of the
per-dtype ``fromstr``/``scan`` functions in ``arraytypes.c.src``.

The port is deliberately literal: numpy's separator matching has a number of
quirks (``swab_separator`` normalising whitespace runs and appending a
trailing space, the whitespace wildcard that consumes *one or more* spaces,
the complex parser that pushes a bogus ``'a'`` back into the stream to force a
parse error) that user code and the test-suite depend on, and the only
reliable way to reproduce the exact set of accepted/rejected inputs is to
follow the original control flow.
"""

import os
import re
import struct

import rnp_numpy as _np


__all__ = ["fromstring", "fromfile", "loadtxt", "genfromtxt", "savetxt"]


# ---------------------------------------------------------------------------
# Low level scalar parsers (the C library functions numpy leans on)
# ---------------------------------------------------------------------------

_C_SPACE = " \t\n\r\v\f"


def _isspace(c):
    return c in _C_SPACE


_STRTOD_RE = re.compile(
    r"""[ \t\n\r\v\f]*
        (?P<num>
            [+-]?
            (?:
                0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)
                     (?:[pP][+-]?[0-9]+)?
              | (?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?
              | [iI][nN][fF](?:[iI][nN][iI][tT][yY])?
              | [nN][aA][nN](?:\([0-9a-zA-Z_]*\))?
            )
        )""",
    re.VERBOSE,
)

_STRTOL_RE = re.compile(r"[ \t\n\r\v\f]*(?P<num>[+-]?[0-9]+)")


def _text_to_float(text):
    """``float()`` restricted to what C's ``strtod`` spells."""
    low = text.lower().lstrip("+-")
    sign = -1.0 if text.lstrip()[:1] == "-" else 1.0
    if low.startswith("nan"):
        return float("nan")
    if low.startswith("inf"):
        return sign * float("inf")
    if low.startswith("0x"):
        return float.fromhex(text)
    return float(text)


def _strtod(s, i):
    """C ``strtod``: ``(value, endpos)``; ``endpos == i`` means no conversion."""
    m = _STRTOD_RE.match(s, i)
    if m is None:
        return 0.0, i
    return _text_to_float(m.group("num")), m.end()


def _strtol(s, i):
    m = _STRTOL_RE.match(s, i)
    if m is None:
        return 0, i
    return int(m.group("num")), m.end()


def _c_fromstr(s, i):
    """Port of ``CDOUBLE_fromstr`` from ``arraytypes.c.src``."""
    result, end = _strtod(s, i)
    ch = s[end] if end < len(s) else "\0"
    if ch in "+-":
        real = result
        imag_val, end2 = _strtod(s, end)
        ch2 = s[end2] if end2 < len(s) else "\0"
        if ch2 == "j":
            return complex(real, imag_val), end2 + 1
        # numpy leaves the cursor past the failed imaginary parse, which makes
        # the separator match fail and produces the "unmatched data" error.
        return complex(real, 0.0), end2
    if ch == "j":
        return complex(0.0, result), end + 1
    return complex(result, 0.0), end


def _fromstr_for(kind):
    if kind == "f":
        return _strtod
    if kind == "c":
        return _c_fromstr
    if kind in "iu":
        return _strtol
    if kind == "b":
        def _b(s, i):
            v, e = _strtod(s, i)
            return v != 0.0, e
        return _b
    return None


# ---------------------------------------------------------------------------
# array_from_text over a string
# ---------------------------------------------------------------------------

def _swab_separator(sep):
    """Port of ``swab_separator``: collapse whitespace runs to single spaces,
    prepend a space when the separator does not start with one, and append a
    space when the result already ends in one."""
    out = []
    skip_space = False
    if sep and not _isspace(sep[0]):
        out.append(" ")
    for ch in sep:
        if _isspace(ch):
            if not skip_space:
                out.append(" ")
                skip_space = True
        else:
            out.append(ch)
            skip_space = False
    if out and out[-1] == " ":
        out.append(" ")
    return "".join(out)


def _fromstr_skip_separator(s, i, sep, end):
    """Port of ``fromstr_skip_separator``. Returns ``(newpos, flag)``."""
    string = i
    si = 0
    nsep = len(sep)
    while True:
        c = s[string] if string < end else "\0"
        if string >= end:
            return string, -1
        if si >= nsep:
            if string != i:
                return string, 0
            return string, -2
        if sep[si] == " ":
            if not _isspace(c):
                si += 1
                continue
        elif sep[si] != c:
            return string, -2
        else:
            si += 1
        string += 1


_UNMATCHED = ("string or file could not be read to its end "
              "due to unmatched data")


def _finish(vals, num, dtype):
    if num >= 0 and len(vals) < num:
        # numpy leaves the tail of the (uninitialised) buffer alone; zeros are
        # the closest deterministic stand-in.
        out = _np.zeros(num, dtype=dtype)
        if vals:
            out[:len(vals)] = _np.array(vals, dtype=dtype)
        return out
    return _np.array(vals, dtype=dtype)


def _array_from_text_string(dtype, num, sep, s):
    fromstr = _fromstr_for(dtype.kind)
    if fromstr is None:
        raise ValueError("don't know how to read character strings with "
                         "that array type")
    clean_sep = _swab_separator(sep)
    end = len(s)
    i = 0
    vals = []
    stop = 0
    while num < 0 or len(vals) < num:
        val, e = fromstr(s, i)
        if e == i:
            stop = -1 if i >= end else -2
            break
        if e > end:
            stop = -1
            break
        vals.append(val)
        i = e
        i, stop = _fromstr_skip_separator(s, i, clean_sep, end)
        if stop < 0:
            if num == len(vals):
                stop = -1
            break
    if stop == -2:
        raise ValueError(_UNMATCHED)
    return _finish(vals, num, dtype)


# ---------------------------------------------------------------------------
# array_from_text over a stream (fromfile text mode)
# ---------------------------------------------------------------------------

_EOF = object()


class _Stream:
    """The subset of ``FILE *`` that ``array_from_text`` uses."""

    __slots__ = ("buf", "pos", "pb")

    def __init__(self, buf):
        self.buf = buf
        self.pos = 0
        self.pb = []

    def getc(self):
        if self.pb:
            return self.pb.pop()
        if self.pos >= len(self.buf):
            return None
        c = self.buf[self.pos]
        self.pos += 1
        return c

    def ungetc(self, c):
        if c is not None:
            self.pb.append(c)


class _EndMatch(Exception):
    pass


def _read_numberlike_string(st):
    """Port of ``read_numberlike_string``. ``None`` means EOF before anything."""
    buf = []
    c = st.getc()
    if c is None:
        return None
    while c is not None and _isspace(c):
        c = st.getc()

    cur = [c]

    def nxt():
        if cur[0] is None:
            raise _EndMatch
        buf.append(cur[0])
        cur[0] = st.getc()

    def alpha_nocase(word):
        for p in word:
            c = cur[0]
            if c is not None and (c == p or c == p.upper()):
                nxt()
            else:
                raise _EndMatch

    try:
        if cur[0] in ("+", "-"):
            nxt()
        c = cur[0]
        if c in ("n", "N"):
            nxt()
            alpha_nocase("an")
            if cur[0] == "(":
                nxt()
                while cur[0] is not None and (cur[0].isalnum() or cur[0] == "_"):
                    nxt()
                if cur[0] == ")":
                    nxt()
            raise _EndMatch
        elif c in ("i", "I"):
            nxt()
            alpha_nocase("nfinity")
            raise _EndMatch
        while cur[0] is not None and "0" <= cur[0] <= "9":
            nxt()
        if cur[0] == ".":
            nxt()
            ok = False
            while cur[0] is not None and "0" <= cur[0] <= "9":
                nxt()
                ok = True
            if not ok:
                raise _EndMatch
        if cur[0] in ("e", "E"):
            nxt()
            if cur[0] in ("+", "-"):
                nxt()
            ok = False
            while cur[0] is not None and "0" <= cur[0] <= "9":
                nxt()
                ok = True
            if not ok:
                raise _EndMatch
    except _EndMatch:
        pass

    st.ungetc(cur[0])
    return "".join(buf)


def _ftolf(st):
    """Port of ``NumPyOS_ascii_ftolf``: ``(value, r)`` with r in {-1, 0, 1}."""
    text = _read_numberlike_string(st)
    if text is None:
        return 0.0, -1          # EOF
    if not text:
        return 0.0, 0
    val, p = _strtod(text, 0)
    if p == 0:
        return 0.0, 0
    return val, 1


def _read_intlike_string(st):
    buf = []
    c = st.getc()
    if c is None:
        return None
    while c is not None and _isspace(c):
        c = st.getc()
    if c is not None and c in "+-":
        buf.append(c)
        c = st.getc()
    while c is not None and "0" <= c <= "9":
        buf.append(c)
        c = st.getc()
    st.ungetc(c)
    return "".join(buf)


def _int_scan(st):
    text = _read_intlike_string(st)
    if text is None:
        return 0, -1
    val, p = _strtol(text, 0)
    if p == 0:
        return 0, 0
    return val, 1


def _double_scan(st):
    return _ftolf(st)


def _bool_scan(st):
    val, r = _ftolf(st)
    return (val != 0.0), r


def _complex_scan(st):
    """Port of ``CDOUBLE_scan``."""
    result, ret_real = _ftolf(st)
    nxt = st.getc()
    if nxt in ("+", "-"):
        real = result
        st.ungetc(nxt)
        imag, ret_imag = _ftolf(st)
        nxt = st.getc()
        if ret_imag == 1 and nxt == "j":
            return complex(real, imag), ret_real
        st.ungetc("a")     # numpy's deliberate poison pill
        return complex(real, 0.0), ret_real
    if nxt == "j":
        return complex(0.0, result), ret_real
    st.ungetc(nxt)
    return complex(result, 0.0), ret_real


def _scan_for(kind):
    if kind == "f":
        return _double_scan
    if kind == "c":
        return _complex_scan
    if kind in "iu":
        return _int_scan
    if kind == "b":
        return _bool_scan
    return None


def _fromfile_skip_separator(st, sep):
    """Port of ``fromfile_skip_separator``."""
    sepi = 0
    start = 0
    nsep = len(sep)
    while True:
        c = st.getc()
        if c is None:
            return -1
        if sepi >= nsep:
            st.ungetc(c)
            return 0 if sepi != start else -2
        if sep[sepi] == " ":
            if not _isspace(c):
                sepi += 1
                start += 1
                st.ungetc(c)
            elif sepi == start:
                start -= 1
        elif sep[sepi] != c:
            st.ungetc(c)
            return -2
        else:
            sepi += 1


def _array_from_text_file(dtype, num, sep, st):
    scan = _scan_for(dtype.kind)
    if scan is None:
        raise ValueError("don't know how to read character strings with "
                         "that array type")
    clean_sep = _swab_separator(sep)
    vals = []
    stop = 0
    while num < 0 or len(vals) < num:
        val, r = scan(st)
        if r == 1:
            stop = 0
        elif r == -1:
            stop = -1
            break
        else:
            stop = -2
            break
        vals.append(val)
        stop = _fromfile_skip_separator(st, clean_sep)
        if stop < 0:
            if num == len(vals):
                stop = -1
            break
    if stop == -2:
        raise ValueError(_UNMATCHED)
    return _finish(vals, num, dtype)


# ---------------------------------------------------------------------------
# Binary helpers (fromfile with sep='')
# ---------------------------------------------------------------------------

_STRUCT_CODE = {
    ("b", 1): "?",
    ("i", 1): "b", ("i", 2): "h", ("i", 4): "i", ("i", 8): "q",
    ("u", 1): "B", ("u", 2): "H", ("u", 4): "I", ("u", 8): "Q",
    ("f", 2): "e", ("f", 4): "f", ("f", 8): "d",
}


def _from_bytes(data, dtype, count):
    kind, size = dtype.kind, dtype.itemsize
    if kind == "c":
        half = size // 2
        code = _STRUCT_CODE.get(("f", half))
        if code is None:
            raise ValueError(f"cannot read dtype {dtype!r} from a binary file")
        n = len(data) // size
        if count >= 0:
            n = min(n, count)
        flat = struct.unpack(f"<{2 * n}{code}", data[:n * size])
        vals = [complex(flat[2 * k], flat[2 * k + 1]) for k in range(n)]
        return _np.array(vals, dtype=dtype) if vals else _np.array(
            [], dtype=dtype)
    code = _STRUCT_CODE.get((kind, size))
    if code is None:
        raise ValueError(f"cannot read dtype {dtype!r} from a binary file")
    n = len(data) // size
    if count >= 0:
        n = min(n, count)
    vals = list(struct.unpack(f"<{n}{code}", data[:n * size]))
    return _np.array(vals, dtype=dtype) if vals else _np.array([], dtype=dtype)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _as_dtype(dtype):
    return _np.dtype(float) if dtype is None else _np.dtype(dtype)


def fromstring(string, dtype=float, count=-1, *, sep, like=None):
    """A new 1-D array initialized from text data in a string."""
    if like is not None:
        raise NotImplementedError("fromstring: `like=` is not supported")
    dt = _as_dtype(dtype)
    if sep == "" or sep is None:
        raise ValueError(
            "The binary mode of fromstring is removed, use frombuffer instead")
    if isinstance(string, (bytes, bytearray, memoryview)):
        string = bytes(string).decode("latin-1")
    elif not isinstance(string, str):
        raise TypeError("fromstring() argument 1 must be str or bytes")
    return _array_from_text_string(dt, int(count), sep, string)


def _open_binary(file):
    """Return ``(fileobj, should_close)`` for a path-or-file argument."""
    if hasattr(file, "read"):
        return file, False
    if hasattr(file, "__fspath__") or isinstance(file, (str, bytes)):
        return open(os.fspath(file), "rb"), True
    raise TypeError("fromfile: `file` must be a path or a file object")


def fromfile(file, dtype=float, count=-1, sep="", offset=0, *, like=None):
    """Construct an array from data in a text or binary file."""
    if like is not None:
        raise NotImplementedError("fromfile: `like=` is not supported")
    dt = _as_dtype(dtype)
    count = int(count)

    fh, close = _open_binary(file)
    try:
        if offset:
            if sep != "":
                raise TypeError("'offset' argument only permitted for "
                                "binary files")
            try:
                fh.seek(offset) if close else fh.seek(offset, 1)
            except (OSError, ValueError):
                fh.read(offset)
        raw = fh.read()
    finally:
        if close:
            fh.close()

    if isinstance(raw, str):
        raw = raw.encode("latin-1")

    if sep == "":
        return _from_bytes(raw, dt, count)

    return _array_from_text_file(dt, count, sep, _Stream(raw.decode("latin-1")))


# ---------------------------------------------------------------------------
# ndarray.tofile -- the write side of the same text format
# ---------------------------------------------------------------------------

def _tofile(self, fid, sep="", format="%s"):
    close = False
    if hasattr(fid, "write"):
        fh = fid
    else:
        fh = open(os.fspath(fid), "wb")
        close = True
    try:
        if sep == "":
            fh.write(self.tobytes(order="C"))
            return
        items = _np.asarray(self).reshape(-1).tolist()
        text = sep.join(format % v for v in items)
        try:
            fh.write(text.encode("latin-1"))
        except TypeError:
            fh.write(text)
    finally:
        if close:
            fh.close()


if not hasattr(_np.ndarray, "tofile"):
    try:
        _np.ndarray.tofile = _tofile
    except (AttributeError, TypeError):  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# loadtxt / genfromtxt / savetxt
# ---------------------------------------------------------------------------

def _line_source(fname, encoding, newline_ok=True):
    """Yield decoded lines from a path, file object or iterable of lines."""
    if isinstance(fname, (str, bytes)) or hasattr(fname, "__fspath__"):
        path = os.fspath(fname)
        if isinstance(path, bytes):
            path = path.decode("latin-1")
        fh = open(path, encoding=encoding or "utf-8")
        try:
            yield from fh
        finally:
            fh.close()
        return
    for line in fname:
        if isinstance(line, bytes):
            line = line.decode(encoding or "utf-8")
        yield line


def _strip_comments(line, comments):
    if not comments:
        return line
    for c in comments:
        idx = line.find(c)
        if idx >= 0:
            line = line[:idx]
    return line


def _split(line, delimiter):
    if delimiter is None:
        return line.split()
    if isinstance(delimiter, int):
        return [line[i:i + delimiter]
                for i in range(0, len(line), delimiter)]
    return line.split(delimiter)


def _dtype_converter(dt):
    kind = dt.kind
    if kind == "f":
        return float
    if kind in "iu":
        return lambda s: int(float(s)) if ("." in s or "e" in s or "E" in s) \
            else int(s)
    if kind == "c":
        return lambda s: complex(s.replace(" ", ""))
    if kind == "b":
        return lambda s: bool(_bool_from_text(s))
    if kind in "US":
        return lambda s: s
    return lambda s: s


def _bool_from_text(s):
    t = s.strip().lower()
    if t in ("true", "1", "t", "yes"):
        return True
    if t in ("false", "0", "f", "no", ""):
        return False
    return float(t) != 0.0


def _ensure_ndmin(a, ndmin):
    if ndmin not in (0, 1, 2):
        raise ValueError(f"Illegal value of ndmin keyword: {ndmin}")
    if a.ndim > ndmin:
        a = _np.squeeze(a)
    if a.ndim < ndmin:
        if ndmin == 1:
            a = _np.atleast_1d(a)
        elif ndmin == 2:
            a = _np.atleast_2d(a).T
    return a


def _read_rows(fname, comments, delimiter, skiprows, usecols, max_rows,
               encoding, skip_footer=0, quotechar=None):
    if comments is None:
        comment_list = ()
    elif isinstance(comments, str):
        comment_list = (comments,)
    else:
        comment_list = tuple(comments)

    rows = []
    seen = 0
    for line in _line_source(fname, encoding):
        if seen < skiprows:
            seen += 1
            continue
        seen += 1
        line = _strip_comments(line.rstrip("\r\n"), comment_list)
        if quotechar:
            line = line.replace(quotechar, "")
        fields = _split(line, delimiter)
        fields = [f for f in fields] if delimiter is None else fields
        if not fields or (len(fields) == 1 and not fields[0].strip()):
            continue
        rows.append(fields)
        if (max_rows is not None and skip_footer == 0
                and len(rows) >= max_rows):
            break
    if skip_footer:
        rows = rows[:len(rows) - skip_footer] if skip_footer < len(rows) else []
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows]
    if usecols is not None:
        cols = [usecols] if isinstance(usecols, int) else list(usecols)
        rows = [[r[c] for c in cols] for r in rows]
    return rows


def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding=None, max_rows=None, *, quotechar=None,
            like=None):
    """Load data from a text file."""
    if like is not None:
        raise NotImplementedError("loadtxt: `like=` is not supported")
    dt = _as_dtype(dtype)
    if dt.names is not None:
        raise NotImplementedError(
            "loadtxt: structured dtypes are not supported by rnp yet")

    rows = _read_rows(fname, comments, delimiter, skiprows, usecols,
                      max_rows, encoding, quotechar=quotechar)

    conv = _dtype_converter(dt)
    if converters is not None:
        if callable(converters):
            colconv = None
            allconv = converters
        else:
            colconv = {int(k): v for k, v in converters.items()}
            allconv = None
    else:
        colconv = allconv = None

    data = []
    ncols = None
    for r in rows:
        if ncols is None:
            ncols = len(r)
        elif len(r) != ncols:
            raise ValueError(
                f"the number of columns changed from {ncols} to {len(r)} "
                "at some point")
        out = []
        for j, field in enumerate(r):
            if allconv is not None:
                field = allconv(field)
            elif colconv is not None and j in colconv:
                field = colconv[j](field)
            out.append(field if not isinstance(field, str) else
                       conv(field.strip()))
        data.append(out)

    if not data:
        import warnings
        warnings.warn("loadtxt: input contained no data", UserWarning,
                      stacklevel=2)
        arr = _np.array([], dtype=dt)
        arr = _ensure_ndmin(arr, ndmin)
        return arr.T if unpack else arr

    arr = _np.array(data, dtype=dt)
    arr = _ensure_ndmin(arr, ndmin)
    return arr.T if unpack else arr


def _guess_column_dtype(values):
    for conv, dt in ((int, _np.dtype(int)),
                     (float, _np.dtype(float)),
                     (complex, _np.dtype(complex))):
        try:
            for v in values:
                conv(v)
        except (ValueError, TypeError):
            continue
        return dt
    width = max((len(v) for v in values), default=1)
    return _np.dtype(f"U{max(width, 1)}")


def genfromtxt(fname, dtype=float, comments='#', delimiter=None,
               skip_header=0, skip_footer=0, converters=None,
               missing_values=None, filling_values=None, usecols=None,
               names=None, excludelist=None,
               deletechars=None, replace_space='_', autostrip=False,
               case_sensitive=True, defaultfmt="f%i", unpack=None,
               usemask=False, loose=True, invalid_raise=True, max_rows=None,
               encoding=None, *, ndmin=0, like=None):
    """Load data from a text file, with missing values handled as specified."""
    if like is not None:
        raise NotImplementedError("genfromtxt: `like=` is not supported")
    if usemask:
        raise NotImplementedError(
            "genfromtxt: `usemask=True` is not supported by rnp yet")

    if names is True or names == "":
        names = None
        take_names = True
    else:
        take_names = False

    rows = _read_rows(fname, comments, delimiter, skip_header, None,
                      max_rows, encoding, skip_footer=skip_footer)

    if take_names and rows:
        names = [f.strip() for f in rows[0]]
        rows = rows[1:]
    elif isinstance(names, str):
        names = [n.strip() for n in names.split(",")]

    if usecols is not None:
        cols = [usecols] if isinstance(usecols, int) else list(usecols)
        rows = [[r[c] for c in cols] for r in rows]
        if names is not None:
            names = [names[c] for c in cols]

    if autostrip or delimiter is not None:
        rows = [[f.strip() for f in r] for r in rows]

    # Missing / filling values.
    if missing_values is None:
        missing = {""}
    elif isinstance(missing_values, str):
        missing = {m.strip() for m in missing_values.split(",")}
    else:
        missing = set(missing_values)
    missing.add("")

    dt = None if dtype is None else _np.dtype(dtype)
    structured = dt is not None and dt.names is not None
    if dt is None and names is not None:
        raise NotImplementedError(
            "genfromtxt: `names=` requires structured array support, which "
            "rnp does not have yet")

    ncols = len(rows[0]) if rows else 0
    if invalid_raise:
        for r in rows:
            if len(r) != ncols:
                raise ValueError(
                    f"Some errors were detected !\n    Line #? "
                    f"(got {len(r)} columns instead of {ncols})")
    else:
        rows = [r for r in rows if len(r) == ncols]

    if dt is None:
        cols = [[r[j] for r in rows] for j in range(ncols)]
        guessed = [_guess_column_dtype([v for v in c if v not in missing])
                   for c in cols]
        dt = guessed[0] if guessed else _np.dtype(float)
        for g in guessed[1:]:
            dt = _np.promote_types(dt, g)

    conv = _dtype_converter(dt)
    if filling_values is None:
        fill = 0 if dt.kind in "iub" else (
            float("nan") if dt.kind == "f" else
            complex("nan") if dt.kind == "c" else "")
    else:
        fill = filling_values

    colconv = None
    allconv = None
    if converters is not None:
        if callable(converters):
            allconv = converters
        else:
            colconv = {int(k): v for k, v in converters.items()}

    field_converters = None
    if structured:
        field_converters = [
            _dtype_converter(dt.fields[name][0]) for name in dt.names]

    data = []
    for r in rows:
        out = []
        for j, field in enumerate(r):
            if field in missing:
                out.append(fill)
                continue
            if allconv is not None:
                field = allconv(field)
            elif colconv is not None and j in colconv:
                field = colconv[j](field)
            if isinstance(field, str):
                try:
                    converter = (field_converters[j] if field_converters
                                 is not None else conv)
                    field = converter(field.strip())
                except ValueError:
                    if not loose:
                        raise
                    field = fill
            out.append(field)
        data.append(tuple(out) if structured else out)

    if not data:
        import warnings
        warnings.warn("genfromtxt: Empty input file: " + repr(fname),
                      UserWarning, stacklevel=2)
        arr = _np.array([], dtype=dt)
    else:
        arr = _np.array(data, dtype=dt)
    arr = _ensure_ndmin(arr, ndmin)
    return arr.T if unpack else arr


def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
            footer='', comments='# ', encoding=None):
    """Save an array to a text file."""
    arr = _np.asarray(X)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        rows = [[v] for v in arr.tolist()]
    elif arr.ndim == 2:
        rows = arr.tolist()
    else:
        raise ValueError(
            "Expected 1D or 2D array, got %dD array instead" % arr.ndim)

    ncol = len(rows[0]) if rows else 0

    if isinstance(fmt, (list, tuple)):
        if len(fmt) != ncol:
            raise AttributeError(
                f"fmt has wrong shape.  {fmt}")
        fmt_row = delimiter.join(fmt)
    elif isinstance(fmt, str):
        n_spec = fmt.count('%') - 2 * fmt.count('%%')
        if n_spec == 1:
            fmt_row = delimiter.join([fmt] * ncol)
        elif n_spec != ncol:
            raise AttributeError(f"fmt has wrong number of % formats:  {fmt}")
        else:
            fmt_row = fmt
    else:
        raise ValueError("invalid fmt: %r" % (fmt,))

    close = False
    if hasattr(fname, "write"):
        fh = fname
    else:
        path = os.fspath(fname)
        if isinstance(path, bytes):
            path = path.decode("latin-1")
        fh = open(path, "w", encoding=encoding or "utf-8")
        close = True
    try:
        if header:
            fh.write(comments + header.replace("\n", "\n" + comments) + newline)
        for row in rows:
            fh.write(fmt_row % tuple(row) + newline)
        if footer:
            fh.write(comments + footer.replace("\n", "\n" + comments) + newline)
    finally:
        if close:
            fh.close()
