"""``numpy.memmap`` — a memory-mapped array.

Upstream this is ``class memmap(ndarray)``: ``ndarray.__new__`` is handed the
``mmap`` object as ``buffer=``, so the array *is* the mapped pages and every
write lands in the file with no further work.

The port's ``_rnp.ndarray`` cannot do either half of that: it is not
instantiable from Python at all (``ndarray.__new__``/``obj.__class__ =``
both raise ``TypeError: cannot create ... instances``) and it has no way to
build an array over foreign memory (no ``buffer=`` constructor, no
``frombuffer``, no ``__array_interface__``).  So ``memmap`` here is a
*composition* wrapper: it owns a private ``_rnp.ndarray`` holding a copy of
the mapped bytes and copies the bytes back into the mapping whenever the
array is mutated (and on ``flush``/``__del__``).  ``_rnp.ndarray`` does
export the buffer protocol, so both copies are a single ``memoryview`` blit.

The observable difference from real numpy: two *live* memmap objects over the
same file do not see each other's un-flushed writes, and mutations performed
through a raw ``_rnp.ndarray`` obtained from the wrapper (rather than through
the wrapper's own API) are only propagated at the next sync point.
Everything else — persistence after ``flush()``/``del``, views sharing the
mapping, ``mode``/``offset``/``filename`` bookkeeping — is real.
"""

import operator
import os
from collections.abc import Sequence
from contextlib import nullcontext

import _rnp

__all__ = ['memmap']

dtypedescr = _rnp.dtype
ndarray = _rnp.ndarray
valid_filemodes = ["r", "c", "r+", "w+"]
writeable_filemodes = ["r+", "w+"]

mode_equivalents = {
    "readonly": "r",
    "copyonwrite": "c",
    "readwrite": "r+",
    "write": "w+"
    }

_BASIC = (slice, type(Ellipsis), type(None))


def _is_basic_element(index):
    if isinstance(index, _BASIC):
        return True
    if isinstance(index, (bool, tuple, list, ndarray, memmap)):
        return False
    try:
        operator.index(index)
    except TypeError:
        return False
    return True


def _is_basic_index(index):
    """True when ``arr[index]`` is a view in numpy's indexing rules."""
    if isinstance(index, tuple):
        return all(_is_basic_element(i) for i in index)
    return _is_basic_element(index)


class _flags:
    """The subset of ``ndarray.flags`` a memmap has an opinion about."""

    def __init__(self, arr, writeable):
        self._arr = arr
        self.writeable = writeable

    def __getattr__(self, name):
        return getattr(self._arr.flags, name)

    def __getitem__(self, key):
        if key in ('WRITEABLE', 'W'):
            return self.writeable
        return self._arr.flags[key]

    def __repr__(self):
        return (f"  C_CONTIGUOUS : {self._arr.flags.c_contiguous}\n"
                f"  F_CONTIGUOUS : {self._arr.flags.f_contiguous}\n"
                f"  OWNDATA : False\n"
                f"  WRITEABLE : {self.writeable}\n"
                f"  ALIGNED : {self._arr.flags.aligned}\n")


def _unwrap(obj):
    return obj._arr if isinstance(obj, memmap) else obj


class memmap(Sequence):
    """Create a memory-map to an array stored in a *binary* file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  NumPy's memmaps
    are array-like objects.
    """

    __module__ = "numpy"
    __array_priority__ = -100.0

    _mmap = None
    filename = None
    offset = None
    mode = None

    def __new__(cls, filename, dtype=_rnp.dtype('uint8'), mode='r+', offset=0,
                shape=None, order='C'):
        import mmap
        import os.path
        try:
            mode = mode_equivalents[mode]
        except KeyError:
            if mode not in valid_filemodes:
                all_modes = valid_filemodes + list(mode_equivalents.keys())
                raise ValueError(
                    f"mode must be one of {all_modes!r} (got {mode!r})"
                ) from None

        if mode == 'w+' and shape is None:
            raise ValueError("shape must be given if mode == 'w+'")

        if hasattr(filename, 'read'):
            f_ctx = nullcontext(filename)
        else:
            f_ctx = open(
                os.fspath(filename),
                ('r' if mode == 'c' else mode) + 'b'
            )

        with f_ctx as fid:
            fid.seek(0, 2)
            flen = fid.tell()
            descr = dtypedescr(dtype)
            _dbytes = descr.itemsize

            if shape is None:
                nbytes = flen - offset
                if nbytes % _dbytes:
                    raise ValueError("Size of available data is not a "
                                     "multiple of the data-type size.")
                size = nbytes // _dbytes
                shape = (size,)
            else:
                shape = _normalize_shape(shape)
                size = 1
                for k in shape:
                    size *= k

            nbytes = int(offset + size * _dbytes)

            if mode in ('w+', 'r+'):
                # gh-27723: if nbytes == 0, write one byte so that an empty
                # memmap can still be mapped.
                nbytes = max(nbytes, 1)
                if flen < nbytes:
                    fid.seek(nbytes - 1, 0)
                    fid.write(b'\0')
                    fid.flush()

            if mode == 'c':
                acc = mmap.ACCESS_COPY
            elif mode == 'r':
                acc = mmap.ACCESS_READ
            else:
                acc = mmap.ACCESS_WRITE

            start = offset - offset % mmap.ALLOCATIONGRANULARITY
            nbytes -= start
            # length=0 maps the whole file, see gh-27723.
            if nbytes == 0 and start > 0:
                nbytes += mmap.ALLOCATIONGRANULARITY
                start -= mmap.ALLOCATIONGRANULARITY
            array_offset = offset - start
            mm = mmap.mmap(fid.fileno(), nbytes, access=acc, offset=start)

            self = object.__new__(cls)
            self._mmap = mm
            self._array_offset = array_offset
            self._dtype = descr
            self._order = order
            self._root = self
            self._writeable = mode != 'r'
            self._arr = _rnp.empty(shape, dtype=descr)
            self._load()
            self.offset = offset
            self.mode = mode

            if isinstance(filename, os.PathLike):
                self.filename = filename.resolve()
            elif hasattr(fid, "name") and isinstance(fid.name, str):
                self.filename = os.path.abspath(fid.name)
            else:
                self.filename = None

        return self

    def __init__(self, *args, **kwargs):
        pass

    # -- the mapping <-> array bridge --------------------------------------

    def _span(self):
        arr = self._arr
        start = self._array_offset
        return start, start + arr.nbytes

    def _load(self):
        """Fill the backing array from the mapped bytes."""
        arr = self._arr
        if arr.nbytes == 0:
            return
        start, stop = self._span()
        memoryview(arr).cast('B')[:] = self._mmap[start:stop]

    def _store(self):
        """Push the backing array back into the mapping."""
        root = self._root
        if root._mmap is None or root._mmap.closed:
            return
        if not root._writeable:
            return
        arr = root._arr
        if arr.nbytes == 0:
            return
        start, stop = root._span()
        try:
            root._mmap[start:stop] = bytes(memoryview(arr).cast('B'))
        except (ValueError, TypeError):
            pass

    def flush(self):
        """Write any changes in the array to the file on disk."""
        self._store()
        root = self._root
        if root._mmap is not None and not root._mmap.closed:
            try:
                root._mmap.flush()
            except ValueError:
                pass

    def __del__(self):
        try:
            if self._root is self:
                self._store()
                if self._mmap is not None and not self._mmap.closed:
                    try:
                        self._mmap.flush()
                    except ValueError:
                        pass
                    self._mmap.close()
        except Exception:  # noqa: BLE001 - never raise out of __del__
            pass

    # -- constructing derived memmaps --------------------------------------

    def _wrap_view(self, arr):
        new = object.__new__(type(self))
        new._arr = arr
        new._root = self._root
        new._mmap = self._mmap
        new._array_offset = self._array_offset
        new._dtype = self._dtype
        new._order = self._order
        new._writeable = self._writeable
        new.offset = self.offset
        new.mode = self.mode
        new.filename = self.filename
        return new

    def _wrap_copy(self, arr):
        """A memmap-typed result that does not share the mapping.

        ``memmap`` itself hands these back as plain ndarrays (numpy's
        ``__array_wrap__``/``__getitem__`` rule); subclasses keep their type.
        """
        if type(self) is memmap:
            return arr
        new = object.__new__(type(self))
        new._arr = arr
        new._root = new
        new._mmap = None
        new._array_offset = 0
        new._dtype = arr.dtype
        new._order = self._order
        new._writeable = True
        new.offset = None
        new.mode = None
        new.filename = None
        return new

    # -- array protocol ----------------------------------------------------

    def __getitem__(self, index):
        res = self._arr[_unwrap(index)]
        if not isinstance(res, ndarray):
            return res
        if _is_basic_index(index):
            return self._wrap_view(res)
        return self._wrap_copy(res)

    def __setitem__(self, index, value):
        self._arr[_unwrap(index)] = _unwrap(value)
        self._store()

    def __len__(self):
        return len(self._arr)

    def __iter__(self):
        for i in range(len(self._arr)):
            yield self[i]

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            return self._arr
        return self._arr.astype(dtype)

    def __buffer__(self, flags):
        return memoryview(self._arr)

    def view(self, dtype=None, type=None):
        if dtype is not None or type is not None:
            return self._arr.view(dtype) if dtype is not None else self._arr
        new = self._wrap_view(self._arr[...])
        new._base = self if getattr(self, '_base', None) is None \
            else self._base
        return new

    @property
    def base(self):
        b = getattr(self, '_base', None)
        if b is not None:
            return b
        if self._root is not self:
            return self._root
        return self._mmap

    @property
    def flags(self):
        return _flags(self._arr, self._writeable)

    def __getattr__(self, name):
        # Only reached for names not found on the class/instance.
        if name in ('_arr', '_root', '_mmap'):
            raise AttributeError(name)
        return getattr(self._arr, name)

    def __repr__(self):
        body = repr(self._arr)
        if body.startswith('array('):
            body = body[len('array('):-1]
        return f"memmap({body})"

    def __str__(self):
        return str(self._arr)

    # -- operators ---------------------------------------------------------

    def __eq__(self, other):
        return self._arr == _unwrap(other)

    def __ne__(self, other):
        return self._arr != _unwrap(other)

    def __hash__(self):
        raise TypeError(f"unhashable type: '{type(self).__name__}'")

    def __bool__(self):
        return bool(self._arr)

    def __int__(self):
        return int(self._arr)

    def __float__(self):
        return float(self._arr)

    def __index__(self):
        return operator.index(self._arr)


def _normalize_shape(shape):
    if isinstance(shape, memmap):
        shape = shape._arr
    if isinstance(shape, ndarray):
        shape = shape.tolist()
    if not isinstance(shape, (tuple, list)):
        try:
            shape = [operator.index(shape)]
        except TypeError:
            pass
    return tuple(operator.index(k) for k in shape)


def _binop(name, op):
    def f(self, other):
        return op(self._arr, _unwrap(other))
    f.__name__ = name
    return f


def _rbinop(name, op):
    def f(self, other):
        return op(_unwrap(other), self._arr)
    f.__name__ = name
    return f


def _inplace(name, op):
    def f(self, other):
        self._arr = op(self._arr, _unwrap(other))
        self._store()
        return self
    f.__name__ = name
    return f


def _unop(name, op):
    def f(self):
        return op(self._arr)
    f.__name__ = name
    return f


for _nm, _op in [
        ('add', operator.add), ('sub', operator.sub), ('mul', operator.mul),
        ('truediv', operator.truediv), ('floordiv', operator.floordiv),
        ('mod', operator.mod), ('pow', operator.pow),
        ('and', operator.and_), ('or', operator.or_), ('xor', operator.xor),
        ('lshift', operator.lshift), ('rshift', operator.rshift),
        ('matmul', operator.matmul),
        ('lt', operator.lt), ('le', operator.le),
        ('gt', operator.gt), ('ge', operator.ge)]:
    setattr(memmap, f'__{_nm}__', _binop(f'__{_nm}__', _op))

for _nm, _op in [
        ('add', operator.add), ('sub', operator.sub), ('mul', operator.mul),
        ('truediv', operator.truediv), ('floordiv', operator.floordiv),
        ('mod', operator.mod), ('pow', operator.pow),
        ('and', operator.and_), ('or', operator.or_), ('xor', operator.xor),
        ('lshift', operator.lshift), ('rshift', operator.rshift),
        ('matmul', operator.matmul)]:
    setattr(memmap, f'__r{_nm}__', _rbinop(f'__r{_nm}__', _op))
    setattr(memmap, f'__i{_nm}__', _inplace(f'__i{_nm}__', _op))

for _nm, _op in [('neg', operator.neg), ('pos', operator.pos),
                 ('abs', operator.abs), ('invert', operator.invert)]:
    setattr(memmap, f'__{_nm}__', _unop(f'__{_nm}__', _op))

del _nm, _op
