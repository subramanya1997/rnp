"""`numpy._core._multiarray_umath` — the single C extension module numpy
builds `numpy._core.multiarray` and `numpy._core.umath` out of.

Upstream tests import it directly (`import numpy._core._multiarray_umath as
ncu`), so the shim presents the union of the two namespaces plus the handful
of private helpers those tests reach for. Nothing here fabricates a result:
what the port cannot answer raises NotImplementedError when *called*.
"""

from .. import _stubs as _stubs_mod
from .._stubs import not_implemented, ufunc

# The union of the two public halves. `umath` goes on top of `multiarray`
# because that is the order numpy's own `_core/__init__` uses, and the only
# names in both are the ufuncs themselves.
from . import multiarray as _multiarray
from . import umath as _umath
from ..lib._rnp_compat import _ArrayFunctionDispatcher, _get_implementing_args

for _mod in (_multiarray, _umath):
    for _name in dir(_mod):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_mod, _name)
del _mod, _name

#: numpy 2.x caps ndim at 64 (NPY_MAXDIMS).
MAXDIMS = 64

#: numpy publishes the runtime CPU-feature table it dispatches on here. This
#: port has no SIMD dispatch table at all, so it reports no features rather
#: than claiming ones it never uses. Tests read it to *enable* extra checks;
#: an empty mapping only ever makes them skip, never pass vacuously.
__cpu_features__ = {}
__cpu_baseline__ = []
__cpu_dispatch__ = []


def _arg_impl(x):
    """Complex argument, elementwise — numpy's private `_arg` ufunc."""
    import rnp_numpy as np
    a = np.asarray(x)
    return np.arctan2(a.imag, a.real)


_arg = ufunc("_arg", nin=1, nout=1, impl=_arg_impl,
             qualname="numpy._core._multiarray_umath._arg")


def _discover_array_parameters(obj, dtype=None):
    """Return the ``(descr, shape)`` array coercion would produce for *obj*.

    numpy discovers these without allocating; the port answers the same
    question by actually performing the coercion, which gives the same
    ``(dtype, shape)`` for everything the port can coerce at all.
    """
    import rnp_numpy as np
    if dtype is not None and isinstance(dtype, type):
        # Callers pass a DType *class* (``type(np.dtype("f8"))``); numpy
        # accepts that and discovers the concrete descriptor from it.
        if issubclass(dtype, np.dtype):
            dtype = np.dtypes._default_for_discovery(dtype)
        else:
            dtype = dtype()
    if dtype is not None and np.dtype(dtype).kind == "O":
        from .. import _arraycompat
        return np.dtype(dtype), _arraycompat._discovery_shape(obj)
    arr = np.asarray(obj, dtype=dtype)
    return arr.dtype, arr.shape


def _casting_level(np, from_dt, to_dt):
    """Return NumPy's integer casting level for two resolved descriptors."""
    for level, name in enumerate(("no", "equiv", "safe", "same_kind",
                                  "unsafe")):
        if np.can_cast(from_dt, to_dt, casting=name):
            return level
    return 4


def _string_length_for(dtype):
    if dtype.kind == "b":
        return 5
    if dtype.kind in "iu":
        digits = {1: 3, 2: 5, 4: 10, 8: 20}[dtype.itemsize]
        return digits + (dtype.kind == "i")
    if dtype.char in "gG":
        return 48 * (2 if dtype.char == "G" else 1)
    if dtype.kind == "f":
        return 32
    if dtype.kind == "c":
        return 64
    if dtype.kind in "SUV":
        return dtype.itemsize // (4 if dtype.kind == "U" else 1)
    raise TypeError(f"cannot resolve a string length for {dtype}")


class _BoundArrayMethod:
    """Python equivalent of NumPy's bound two-descriptor cast ArrayMethod."""

    __module__ = "numpy"

    def __init__(self, from_dtype, to_dtype):
        import rnp_numpy as np
        if not (isinstance(from_dtype, type) and
                issubclass(from_dtype, np.dtype)):
            raise TypeError("from_dtype must be a DType class")
        if not (isinstance(to_dtype, type) and
                issubclass(to_dtype, np.dtype)):
            raise TypeError("to_dtype must be a DType class")
        self.dtypes = (from_dtype, to_dtype)
        self._supports_unaligned = False

    def __repr__(self):
        names = ", ".join(repr(cls) for cls in self.dtypes)
        return f"<np._BoundArrayMethod `cast` for dtypes ({names})>"

    def _default_to(self, from_dt):
        import rnp_numpy as np
        to_cls = self.dtypes[1]
        char = to_cls.dtype.char
        if char in "SU":
            if from_dt.kind == "O":
                raise TypeError(
                    "casting from object to the parametric DType requires a "
                    "specified output descriptor")
            length = _string_length_for(from_dt)
            return np.dtype(f"{char}{length}")
        if char == "V":
            return np.dtype(f"V{from_dt.itemsize}")
        if char in "Mm" and from_dt.kind == char:
            return from_dt.newbyteorder("=")
        if to_cls._parametric:
            raise TypeError(
                "casting from object to the parametric DType requires a "
                "specified output descriptor")
        return to_cls()

    def _view_offset(self, from_dt, to_dt, safety):
        if from_dt.fields is not None and to_dt.fields is not None:
            from_fields = [from_dt.fields[name] for name in from_dt.names]
            to_fields = [to_dt.fields[name] for name in to_dt.names]
            if len(from_fields) != len(to_fields):
                return None
            if any(left[0] != right[0]
                   for left, right in zip(from_fields, to_fields)):
                return None
            offsets = [left[1] - right[1]
                       for left, right in zip(from_fields, to_fields)]
            return offsets[0] if offsets and offsets[0] >= 0 \
                and len(set(offsets)) == 1 else None
        if from_dt.fields is not None and to_dt.fields is None:
            if len(from_dt.names) == 1:
                field_dt, offset, *_ = from_dt.fields[from_dt.names[0]]
                if field_dt == to_dt:
                    return offset
            return None
        if from_dt.subdtype is not None and to_dt.subdtype is None:
            base, shape = from_dt.subdtype
            size = 1
            for dim in shape:
                size *= dim
            if size == 1 and base == to_dt:
                return 0
        if from_dt.kind in "SUV" and to_dt.kind == from_dt.kind:
            return (0 if (to_dt.itemsize <= from_dt.itemsize
                          and to_dt.byteorder == from_dt.byteorder) else None)
        if (from_dt == to_dt
                and from_dt.byteorder == to_dt.byteorder):
            return 0
        if from_dt.kind in "Mm" and to_dt.kind == from_dt.kind:
            # A generic-unit descriptor can be relabelled with a concrete unit.
            if from_dt.name == from_dt.type.__name__ or from_dt == to_dt:
                return 0
        if to_dt.kind == "V" and from_dt.kind != "O":
            return 0 if to_dt.itemsize <= from_dt.itemsize else None
        if safety == 0 and from_dt.itemsize == to_dt.itemsize:
            return 0
        return None

    def _resolve_descriptors(self, descriptors):
        import rnp_numpy as np
        if not isinstance(descriptors, tuple) or len(descriptors) != 2:
            raise TypeError("descriptors must be a 2-tuple")
        from_dt, to_dt = descriptors
        if from_dt is None:
            raise TypeError("the input descriptor cannot be None")
        from_dt = np.dtype(from_dt)
        if not isinstance(from_dt, self.dtypes[0]):
            raise TypeError("input descriptor does not match its DType class")
        if to_dt is None:
            to_dt = self._default_to(from_dt)
        else:
            to_dt = np.dtype(to_dt)
            if not isinstance(to_dt, self.dtypes[1]):
                raise TypeError("output descriptor does not match its DType class")

        safety = _casting_level(np, from_dt, to_dt)
        if from_dt.fields is not None and to_dt.fields is not None:
            safety = 1 if from_dt.names == to_dt.names else 2
        if from_dt.kind in "biufc" and to_dt.kind in "biufc":
            safety |= 64  # NPY_METH_SUPPORTS_SAME_VALUE
        view_offset = self._view_offset(from_dt, to_dt, safety & ~64)
        return safety, (from_dt, to_dt), view_offset

    @property
    def casting(self):
        """The descriptor-independent lower bound for this cast."""
        try:
            return self._resolve_descriptors(
                (self.dtypes[0](), None))[0]
        except TypeError:
            return 4

    def _simple_strided_call(self, arrays):
        if not isinstance(arrays, tuple) or len(arrays) != 2:
            raise TypeError("arrays must be a 2-tuple")
        source, destination = arrays
        if source.shape != destination.shape:
            raise ValueError("input and output shapes must match")
        destination[...] = source.astype(destination.dtype)
        return None


def _get_castingimpl(from_dtype, to_dtype):
    return _BoundArrayMethod(from_dtype, to_dtype)

#: The "scaled float" demo DType is defined in numpy's C test-support code and
#: has no port equivalent.
_get_sfloat_dtype = not_implemented(
    "numpy._core._multiarray_umath._get_sfloat_dtype")


def __getattr__(name):
    if name.startswith("_") and not name.startswith("__"):
        return not_implemented(f"numpy._core._multiarray_umath.{name}")
    raise AttributeError(
        f"module 'numpy._core._multiarray_umath' has no attribute {name!r}")
