"""``numpy.memmap`` — a memory-mapped array.

This is the real design, not a wrapper: ``memmap`` is a genuine subclass of
``_rnp.ndarray`` and its buffer *is* the ``mmap`` object.  ``ndarray.__new__``
adopts the mapped pages zero-copy (see ``rnp-python/src/adopt.rs``), so every
write through the array lands in the mapping — and therefore in the file —
with no copying or syncing anywhere.  The ``mmap`` is kept alive by the
array's buffer, so a view outliving its parent still has valid pages.

The one place this port still differs from upstream: ``ndarray`` methods that
return a *new* array (indexing, ufuncs, reductions) hand back a plain
``ndarray`` rather than re-running ``__array_finalize__`` on the subclass, so
``__getitem__`` re-wraps its result here explicitly.  ``__array_wrap__`` is
therefore not consulted by the ufunc machinery, which is why reductions over a
``memmap`` *subclass* come back as ``ndarray`` instead of the subclass.
"""

import operator
import os
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
    if isinstance(index, (bool, tuple, list, ndarray)):
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


class memmap(ndarray):
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

    def __new__(subtype, filename, dtype=_rnp.dtype('uint8'), mode='r+',
                offset=0, shape=None, order='C'):
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

            self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,
                                   offset=array_offset, order=order)
            self._mmap = mm
            self.offset = offset
            self.mode = mode

            if isinstance(filename, os.PathLike):
                self.filename = filename.resolve()
            elif hasattr(fid, "name") and isinstance(fid.name, str):
                self.filename = os.path.abspath(fid.name)
            else:
                self.filename = None

        return self

    def __array_finalize__(self, obj):
        if getattr(obj, '_mmap', None) is not None:
            self._mmap = obj._mmap
            self.filename = obj.filename
            self.offset = obj.offset
            self.mode = obj.mode
        else:
            self._mmap = None
            self.filename = None
            self.offset = None
            self.mode = None

    def flush(self):
        """Write any changes in the array to the file on disk."""
        if self.base is not None and hasattr(self.base, 'flush'):
            self.base.flush()

    def _inherit(self, other):
        """Copy this memmap's mapping bookkeeping onto `other`."""
        other._mmap = self._mmap
        other.filename = self.filename
        other.offset = self.offset
        other.mode = self.mode
        return other

    def __getitem__(self, index):
        res = ndarray.__getitem__(self, index)
        if not isinstance(res, ndarray):
            return res
        if _is_basic_index(index):
            # A view: it shares the mapping, so it stays a memmap.
            return self._inherit(res.view(type(self)))
        # Fancy indexing copies. numpy hands the copy back as ``ndarray`` for
        # ``memmap`` itself and keeps the type for subclasses of it.
        if type(self) is memmap:
            return res
        out = res.view(type(self))
        out._mmap = None
        out.filename = None
        out.offset = None
        out.mode = None
        return out


def _normalize_shape(shape):
    if isinstance(shape, ndarray):
        shape = shape.tolist()
    if not isinstance(shape, (tuple, list)):
        try:
            shape = [operator.index(shape)]
        except TypeError:
            pass
    return tuple(operator.index(k) for k in shape)
