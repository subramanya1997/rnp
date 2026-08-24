"""`numpy._core.records` — record arrays and the `np.rec` namespace.

Ported from `upstream/numpy/_core/records.py` (numpy 2.5.2). The algorithms
(`format_parser`, `fromarrays`, `fromrecords`, `fromstring`, `fromfile`,
`array`) are transcribed from upstream; only the places where upstream reaches
for C-level machinery the port does not have are adapted, and every such spot
is called out below.

**The one structural deviation.** Upstream's `recarray` is an `ndarray`
*subclass*, and its `record`-ness is carried in `dtype.type` via
`np.dtype((record, descr))`. The port's `ndarray` is a PyO3 class with no
Python-level constructor and no `__array_finalize__` / `view(type=...)`
support, so it cannot be subclassed from Python. `recarray` here is therefore a
**delegating wrapper** around a real structured array: it forwards every
attribute, index and operator to the wrapped array and re-wraps structured
results. Everything that goes through the documented `np.rec` API behaves the
same; what differs is `isinstance(r, np.ndarray)` and `arr.view(np.recarray)`,
which need ndarray subclassing in the Rust layer.
"""

import os
import warnings
from collections import Counter
from contextlib import nullcontext

import rnp_numpy as sb

from .._utils import set_module
from . import numerictypes as nt
from .arrayprint import _get_legacy_print_mode
from .._scalars import void

__all__ = [
    'record', 'recarray', 'format_parser', 'fromarrays', 'fromrecords',
    'fromstring', 'fromfile', 'array', 'find_duplicate',
]


ndarray = sb.ndarray

_byteorderconv = {'b': '>',
                  'l': '<',
                  'n': '=',
                  'B': '>',
                  'L': '<',
                  'N': '=',
                  'S': 's',
                  's': 's',
                  '>': '>',
                  '<': '<',
                  '=': '=',
                  '|': '|',
                  'I': '|',
                  'i': '|'}

numfmt = nt.sctypeDict


@set_module('numpy.rec')
def find_duplicate(list):
    """Find duplication in a list, return a list of duplicated elements"""
    return [
        item
        for item, counts in Counter(list).items()
        if counts > 1
    ]


@set_module('numpy.rec')
class format_parser:
    """Convert formats, names, titles description to a dtype.

    ``dtype = format_parser(formats, names, titles).dtype``
    """

    def __init__(self, formats, names, titles, aligned=False, byteorder=None):
        self._parseFormats(formats, aligned)
        self._setfieldnames(names, titles)
        self._createdtype(byteorder)

    def _parseFormats(self, formats, aligned=False):
        """Parse the field formats."""
        if formats is None:
            raise ValueError("Need formats argument")
        if isinstance(formats, list):
            dtype = sb.dtype(
                [(f'f{i}', format_) for i, format_ in enumerate(formats)],
                aligned,
            )
        else:
            dtype = sb.dtype(formats, aligned)
        fields = dtype.fields
        if fields is None:
            dtype = sb.dtype([('f1', dtype)], aligned)
            fields = dtype.fields
        keys = dtype.names
        self._f_formats = [fields[key][0] for key in keys]
        self._offsets = [fields[key][1] for key in keys]
        self._nfields = len(keys)

    def _setfieldnames(self, names, titles):
        """Convert input field names into a list."""
        if names:
            if type(names) in [list, tuple]:
                pass
            elif isinstance(names, str):
                names = names.split(',')
            else:
                raise NameError(f"illegal input names {repr(names)}")

            self._names = [n.strip() for n in names[:self._nfields]]
        else:
            self._names = []

        self._names += [f'f{i}' for i in range(len(self._names),
                                               self._nfields)]
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError(f"Duplicate field names: {_dup}")

        if titles:
            self._titles = [n.strip() for n in titles[:self._nfields]]
        else:
            self._titles = []
            titles = []

        if self._nfields > len(titles):
            self._titles += [None] * (self._nfields - len(titles))

    def _createdtype(self, byteorder):
        dtype = sb.dtype({
            'names': self._names,
            'formats': self._f_formats,
            'offsets': self._offsets,
            'titles': self._titles,
        })
        if byteorder is not None:
            byteorder = _byteorderconv[byteorder[0]]
            dtype = dtype.newbyteorder(byteorder)

        self.dtype = dtype


class record(void):
    """A data-type scalar that allows field access as attribute lookup."""

    # numpy sets these by hand so the class prints as `numpy.record`.
    __name__ = 'record'
    __module__ = 'numpy'
    __slots__ = ()

    def __repr__(self):
        if _get_legacy_print_mode() <= 113:
            return self.__str__()
        return super().__repr__()

    def __str__(self):
        if _get_legacy_print_mode() <= 113:
            return str(self.item())
        return super().__str__()

    def __getattribute__(self, attr):
        # The port's `void` keeps its backing 0-d view in `_arr` / `_b`; those
        # must never be routed through the field lookup below (upstream's
        # `void` has no Python-level state, so it has no equivalent).
        if attr in ('setfield', 'getfield', 'dtype') or attr.startswith('_'):
            return void.__getattribute__(self, attr)
        try:
            return void.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None) if fielddict else None
        if res:
            obj = self.getfield(*res[:2])
            try:
                dt = obj.dtype
            except AttributeError:
                # happens if field is Object type
                return obj
            if dt.names is not None:
                return obj.view((self.__class__, obj.dtype))
            return obj
        raise AttributeError(f"'record' object has no attribute '{attr}'")

    def __setattr__(self, attr, val):
        if attr in ('setfield', 'getfield', 'dtype'):
            raise AttributeError(f"Cannot set '{attr}' attribute")
        if attr.startswith('_'):
            return void.__setattr__(self, attr, val)
        fielddict = void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None) if fielddict else None
        if res:
            return self.setfield(val, *res[:2])
        if getattr(self, attr, None):
            return void.__setattr__(self, attr, val)
        raise AttributeError(f"'record' object has no attribute '{attr}'")

    def __getitem__(self, indx):
        obj = void.__getitem__(self, indx)
        # Mirror __getattribute__: a nested structured field is a record too.
        if isinstance(obj, void) and obj.dtype.names is not None:
            return obj.view((self.__class__, obj.dtype))
        return obj

    def pprint(self):
        """Pretty-print all fields."""
        names = self.dtype.names or ()
        maxlen = max((len(n) for n in names), default=0)
        rows = [f"{name:>{maxlen}}: {getattr(self, name)}" for name in names]
        return "\n".join(rows)


def _asarray(x):
    return x._arr if isinstance(x, recarray) else sb.asarray(x)


@set_module("numpy.rec")
class recarray:
    """An array that allows field access using attributes.

    See the module docstring for the one way this differs from numpy's: it
    wraps a structured array rather than subclassing `ndarray`, because the
    port's `ndarray` cannot be subclassed from Python.
    """

    #: Set on the instance in `__new__`; named with a leading underscore so it
    #: can never collide with a field name.
    __slots__ = ("_arr",)

    def __new__(cls, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False, order='C'):
        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(
                formats, names, titles, aligned, byteorder).dtype

        if buf is None:
            arr = sb.zeros(shape, descr, order=order)
        else:
            arr = sb.frombuffer(buf, dtype=descr, offset=offset)
            if strides is not None:
                raise NotImplementedError(
                    "recarray(strides=) needs ndarray stride surgery, which "
                    "the port does not expose yet")
            if shape is not None:
                arr = arr.reshape(shape)
        return cls._wrap(arr)

    @classmethod
    def _wrap(cls, arr):
        self = object.__new__(cls)
        object.__setattr__(self, "_arr", arr)
        return self

    # -- the ndarray surface -----------------------------------------------

    def __array__(self, dtype=None, copy=None):
        a = object.__getattribute__(self, "_arr")
        return a if dtype is None else a.astype(dtype)

    def __getattribute__(self, attr):
        if attr in ("_arr", "_wrap", "field", "view", "__array__",
                    "__class__", "__dict__"):
            return object.__getattribute__(self, attr)
        arr = object.__getattribute__(self, "_arr")
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            pass
        # A field name wins over nothing; a real ndarray attribute wins over a
        # field, exactly as upstream documents.
        fielddict = arr.dtype.fields
        if not (fielddict and attr in fielddict) and hasattr(arr, attr):
            return _rewrap(getattr(arr, attr))
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(f"recarray has no attribute {attr}") from e
        obj = arr.getfield(*res)
        if obj.dtype.names is not None:
            return recarray._wrap(obj)
        return obj

    def __setattr__(self, attr, val):
        if attr == "_arr":
            object.__setattr__(self, attr, val)
            return
        arr = object.__getattribute__(self, "_arr")
        fielddict = arr.dtype.fields or {}
        if attr in fielddict:
            return arr.setfield(val, *fielddict[attr][:2])
        if attr == "dtype":
            raise AttributeError(
                "the port's recarray cannot have its dtype reassigned")
        raise AttributeError(f"record array has no attribute {attr}")

    def __getitem__(self, indx):
        arr = object.__getattribute__(self, "_arr")
        obj = arr[indx]
        if isinstance(obj, ndarray):
            if obj.dtype.names is not None:
                return recarray._wrap(obj)
            return obj
        if isinstance(obj, void) and obj.dtype.names is not None:
            return obj.view((record, obj.dtype))
        return obj

    def __setitem__(self, indx, value):
        arr = object.__getattribute__(self, "_arr")
        arr[indx] = _asarray(value) if isinstance(value, recarray) else value

    def __len__(self):
        return len(object.__getattribute__(self, "_arr"))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other):
        return _rewrap(
            object.__getattribute__(self, "_arr") == _cmp_operand(other))

    def __ne__(self, other):
        return _rewrap(
            object.__getattribute__(self, "_arr") != _cmp_operand(other))

    def __hash__(self):
        raise TypeError("unhashable type: 'numpy.recarray'")

    def __repr__(self):
        arr = object.__getattribute__(self, "_arr")
        repr_dtype = arr.dtype
        prefix = "rec.array("
        fmt = 'rec.array(%s,%sdtype=%s)'
        if arr.size > 0 or arr.shape == (0,):
            lst = sb.array2string(
                arr, separator=', ', prefix=prefix, suffix=',')
        else:
            lst = f"[], shape={repr(arr.shape)}"
        lf = '\n' + ' ' * len(prefix)
        if _get_legacy_print_mode() <= 113:
            lf = ' ' + lf  # trailing space
        return fmt % (lst, lf, repr_dtype)

    def __str__(self):
        return str(object.__getattribute__(self, "_arr"))

    def view(self, dtype=None, type=None):
        arr = object.__getattribute__(self, "_arr")
        if dtype is None and type is None:
            return recarray._wrap(arr)
        return arr.view(dtype, type)

    def field(self, attr, val=None):
        arr = object.__getattribute__(self, "_arr")
        if isinstance(attr, int):
            attr = arr.dtype.names[attr]
        res = arr.dtype.fields[attr][:2]
        if val is None:
            obj = arr.getfield(*res)
            if obj.dtype.names is not None:
                return recarray._wrap(obj)
            return obj
        return arr.setfield(val, *res)


def _cmp_operand(other):
    return object.__getattribute__(other, "_arr") \
        if isinstance(other, recarray) else other


def _rewrap(value):
    """Re-wrap a structured ndarray result as a recarray."""
    if isinstance(value, ndarray) and value.dtype.names is not None:
        return recarray._wrap(value)
    return value


def _deprecate_shape_0_as_None(shape):
    if shape == 0:
        warnings.warn(
            "Passing `shape=0` to have the shape be inferred is deprecated, "
            "and in future will be equivalent to `shape=(0,)`. To infer "
            "the shape and suppress this warning, pass `shape=None` instead.",
            FutureWarning, stacklevel=3)
        return None
    return shape


@set_module("numpy.rec")
def fromarrays(arrayList, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    """Create a record array from a (flat) list of arrays."""
    arrayList = [_asarray(x) for x in arrayList]

    shape = _deprecate_shape_0_as_None(shape)

    if shape is None:
        shape = arrayList[0].shape
    elif isinstance(shape, int):
        shape = (shape,)

    if formats is None and dtype is None:
        formats = [obj.dtype for obj in arrayList]

    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    _names = descr.names

    if len(descr) != len(arrayList):
        raise ValueError("mismatch between the number of fields "
                         "and the number of arrays")

    d0 = descr[0].shape
    nn = len(d0)
    if nn > 0:
        shape = shape[:-nn]

    _array = recarray(shape, descr)

    for k, obj in enumerate(arrayList):
        nn = descr[k].ndim
        testshape = obj.shape[:obj.ndim - nn]
        name = _names[k]
        if testshape != shape:
            raise ValueError(f'array-shape mismatch in array {k} ("{name}")')
        _array[name] = obj

    return _array


@set_module("numpy.rec")
def fromrecords(recList, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None):
    """Create a recarray from a list of records in text form."""
    if formats is None and dtype is None:  # slower
        obj = sb.array(recList, dtype=object)
        arrlist = [
            sb.array(obj[..., i].tolist()) for i in range(obj.shape[-1])
        ]
        return fromarrays(arrlist, formats=formats, shape=shape, names=names,
                          titles=titles, aligned=aligned, byteorder=byteorder)

    if dtype is not None:
        # Upstream writes `sb.dtype((record, dtype))`; the port carries the
        # record-ness on the wrapper class instead of in `dtype.type`.
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(
            formats, names, titles, aligned, byteorder).dtype

    try:
        retval = sb.array(recList, dtype=descr)
    except (TypeError, ValueError):
        shape = _deprecate_shape_0_as_None(shape)
        if shape is None:
            shape = len(recList)
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) > 1:
            raise ValueError("Can only deal with 1-d array.")
        _array = recarray(shape, descr)
        for k in range(len(recList)):
            _array[k] = tuple(recList[k])
        warnings.warn(
            "fromrecords expected a list of tuples, may have received a list "
            "of lists instead. In the future that will raise an error",
            FutureWarning, stacklevel=2)
        return _array
    else:
        if shape is not None and retval.shape != shape:
            retval = retval.reshape(shape)

    return recarray._wrap(retval)


@set_module("numpy.rec")
def fromstring(datastring, dtype=None, shape=None, offset=0, formats=None,
               names=None, titles=None, aligned=False, byteorder=None):
    """Create a record array from binary data."""
    if dtype is None and formats is None:
        raise TypeError("fromstring() needs a 'dtype' or 'formats' argument")

    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype

    itemsize = descr.itemsize

    shape = _deprecate_shape_0_as_None(shape)

    if shape in (None, -1):
        shape = (len(datastring) - offset) // itemsize

    _array = recarray(shape, descr,
                      buf=datastring, offset=offset)
    return _array


def get_remaining_size(fd):
    pos = fd.tell()
    try:
        fd.seek(0, 2)
        return fd.tell() - pos
    finally:
        fd.seek(pos, 0)


@set_module("numpy.rec")
def fromfile(fd, dtype=None, shape=None, offset=0, formats=None,
             names=None, titles=None, aligned=False, byteorder=None):
    """Create an array from binary file data."""
    if dtype is None and formats is None:
        raise TypeError("fromfile() needs a 'dtype' or 'formats' argument")

    shape = _deprecate_shape_0_as_None(shape)

    if shape is None:
        shape = (-1,)
    elif isinstance(shape, int):
        shape = (shape,)

    if hasattr(fd, 'readinto'):
        ctx = nullcontext(fd)
    else:
        ctx = open(os.fspath(fd), 'rb')

    with ctx as fd:
        if offset > 0:
            fd.seek(offset, 1)
        size = get_remaining_size(fd)

        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(
                formats, names, titles, aligned, byteorder).dtype

        itemsize = descr.itemsize

        shapeprod = 1
        for s in shape:
            shapeprod *= s
        shapesize = shapeprod * itemsize
        if shapesize < 0:
            shape = list(shape)
            shape[shape.index(-1)] = size // -shapesize
            shape = tuple(shape)
            shapeprod = 1
            for s in shape:
                shapeprod *= s

        nbytes = shapeprod * itemsize

        if nbytes > size:
            raise ValueError(
                "Not enough bytes left in file for specified "
                "shape and type.")

        data = fd.read(int(nbytes))
        if len(data) != nbytes:
            raise OSError("Didn't read as many bytes as expected")

    _array = recarray(shape, descr, buf=data)
    return _array


@set_module("numpy.rec")
def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None,
          names=None, titles=None, aligned=False, byteorder=None, copy=True):
    """Construct a record array from a wide variety of objects."""
    if ((isinstance(obj, (type(None), str)) or hasattr(obj, 'readinto')) and
            formats is None and dtype is None):
        raise ValueError("Must define formats (or dtype) if object is "
                         "None, string, or an open file")

    kwds = {}
    if dtype is not None:
        dtype = sb.dtype(dtype)
    elif formats is not None:
        dtype = format_parser(formats, names, titles,
                              aligned, byteorder).dtype
    else:
        kwds = {'formats': formats,
                'names': names,
                'titles': titles,
                'aligned': aligned,
                'byteorder': byteorder}

    if obj is None:
        if shape is None:
            raise ValueError("Must define a shape if obj is None")
        return recarray(shape, dtype, buf=obj, offset=offset, strides=strides)

    elif isinstance(obj, bytes):
        return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)

    elif isinstance(obj, (list, tuple)):
        if isinstance(obj[0], (tuple, list)):
            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
        return fromarrays(obj, dtype=dtype, shape=shape, **kwds)

    elif isinstance(obj, recarray):
        arr = object.__getattribute__(obj, "_arr")
        if dtype is not None and (arr.dtype != dtype):
            new = arr.astype(dtype)
        else:
            new = arr
        if copy:
            new = new.copy()
        return recarray._wrap(new)

    elif hasattr(obj, 'readinto'):
        return fromfile(obj, dtype=dtype, shape=shape, offset=offset)

    elif isinstance(obj, ndarray):
        if dtype is not None and (obj.dtype != dtype):
            new = obj.astype(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return recarray._wrap(new)

    else:
        interface = getattr(obj, "__array_interface__", None)
        if interface is None or not isinstance(interface, dict):
            raise ValueError("Unknown input type")
        obj = sb.array(obj)
        if dtype is not None and (obj.dtype != dtype):
            obj = obj.astype(dtype)
        return recarray._wrap(obj)
