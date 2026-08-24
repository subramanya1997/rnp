"""`numpy._core._exceptions` — the richly-typed ufunc exceptions.

These are ordinary Python classes in real numpy too, so the port defines them
the same way: `UFuncTypeError` is the public base (a `TypeError`), and the
concrete subclasses are implementation details renamed to display as the base,
exactly as numpy's `_display_as_base` decorator does.

The Rust engine raises these through the factory registered by
`_install_error_factories()` at the bottom of this module.
"""

import _rnp


def _unpack_tuple(tup):
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def _display_as_base(cls):
    """Make an exception class look like its base.

    Subclasses are implementation details — the user should catch the base
    type, which is what the traceback shows them.
    """
    assert issubclass(cls, Exception)
    cls.__name__ = cls.__base__.__name__
    return cls


class UFuncTypeError(TypeError):
    """ Base class for all ufunc exceptions """
    __module__ = "numpy._core._exceptions"

    def __init__(self, ufunc):
        self.ufunc = ufunc


@_display_as_base
class _UFuncNoLoopError(UFuncTypeError):
    """ Thrown when a ufunc loop cannot be found """
    __module__ = "numpy._core._exceptions"

    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc)
        self.dtypes = tuple(dtypes)

    def __str__(self):
        return (
            f"ufunc {self.ufunc.__name__!r} did not contain a loop with "
            f"signature matching types "
            f"{_unpack_tuple(self.dtypes[:self.ufunc.nin])!r} "
            f"-> {_unpack_tuple(self.dtypes[self.ufunc.nin:])!r}"
        )


@_display_as_base
class _UFuncBinaryResolutionError(_UFuncNoLoopError):
    """ Thrown when a binary resolution fails """
    __module__ = "numpy._core._exceptions"

    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc, dtypes)
        assert len(self.dtypes) == 2

    def __str__(self):
        return (
            "ufunc {!r} cannot use operands with types {!r} and {!r}"
        ).format(
            self.ufunc.__name__, *self.dtypes
        )


@_display_as_base
class _UFuncCastingError(UFuncTypeError):
    __module__ = "numpy._core._exceptions"

    def __init__(self, ufunc, casting, from_, to):
        super().__init__(ufunc)
        self.casting = casting
        self.from_ = from_
        self.to = to


@_display_as_base
class _UFuncInputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc input cannot be casted """
    __module__ = "numpy._core._exceptions"

    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.in_i = i

    def __str__(self):
        # only show the number if more than one input exists
        i_str = f"{self.in_i} " if self.ufunc.nin != 1 else ""
        return (
            f"Cannot cast ufunc {self.ufunc.__name__!r} input {i_str}from "
            f"{self.from_!r} to {self.to!r} with casting rule {self.casting!r}"
        )


@_display_as_base
class _UFuncOutputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc output cannot be casted """
    __module__ = "numpy._core._exceptions"

    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.out_i = i

    def __str__(self):
        # only show the number if more than one output exists
        i_str = f"{self.out_i} " if self.ufunc.nout != 1 else ""
        return (
            f"Cannot cast ufunc {self.ufunc.__name__!r} output {i_str}from "
            f"{self.from_!r} to {self.to!r} with casting rule {self.casting!r}"
        )


@_display_as_base
class _ArrayMemoryError(MemoryError):
    """ Thrown when an array cannot be allocated"""
    __module__ = "numpy._core._exceptions"

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def _total_size(self):
        num_bytes = self.dtype.itemsize
        for dim in self.shape:
            num_bytes *= dim
        return num_bytes

    @staticmethod
    def _size_to_string(num_bytes):
        """ Convert a number of bytes into a binary size string """
        LOG2_STEP = 10
        STEP = 1024
        units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']

        unit_i = max(num_bytes.bit_length() - 1, 1) // LOG2_STEP
        unit_val = 1 << (unit_i * LOG2_STEP)
        n_units = num_bytes / unit_val
        del unit_val

        # ensure we pick a unit that is correct after rounding
        if round(n_units) == STEP:
            unit_i += 1
            n_units /= STEP

        # deal with sizes so large that we don't have units for them
        if unit_i >= len(units):
            new_unit_i = len(units) - 1
            n_units *= 1 << ((unit_i - new_unit_i) * LOG2_STEP)
            unit_i = new_unit_i

        unit_name = units[unit_i]
        # format with a sensible number of digits
        if unit_i == 0:
            # no decimal point on bytes
            return f'{n_units:.0f} {unit_name}'
        elif round(n_units) < 1000:
            # 3 significant figures, if none are dropped to the left of the .
            return f'{n_units:#.3g} {unit_name}'
        else:
            # just give all the digits otherwise
            return f'{n_units:#.0f} {unit_name}'

    def __str__(self):
        size_str = self._size_to_string(self._total_size)
        return (f"Unable to allocate {size_str} for an array with shape "
                f"{self.shape} and data type {self.dtype}")


def _no_loop_error(ufunc_name, dtype_names):
    """Build `_UFuncNoLoopError` from the plain strings the engine hands up.

    The engine has no view of the Python ufunc objects or the `numpy.dtypes`
    classes, so it reports names and this rebuilds numpy's exact payload.
    """
    from .. import dtypes as _dtypes
    from .._ufunc import ALL as _UFUNCS
    ufunc = _UFUNCS[ufunc_name]
    classes = tuple(getattr(_dtypes, _dtypes._CLASS_NAMES[n]) for n in dtype_names)
    return _UFuncNoLoopError(ufunc, classes + (None,) * ufunc.nout)


def _binary_resolution_error(ufunc_name, dtype_strs):
    """Build `_UFuncBinaryResolutionError` from the engine's dtype strings.

    This is what the datetime type resolvers raise when no loop combination
    fits, e.g. ``np.datetime64(1, 's') + np.datetime64(1, 's')``.
    """
    from .. import dtype as _dtype
    from .._ufunc import ALL as _UFUNCS
    ufunc = _UFUNCS[ufunc_name]
    return _UFuncBinaryResolutionError(ufunc, tuple(_dtype(s) for s in dtype_strs))


def _input_casting_error(ufunc_name, casting, from_str, to_str, i):
    """Build `_UFuncInputCastingError` from the engine's plain payload.

    Reached when a datetime ufunc's units are incommensurate: numpy resolves
    the loop to the datetime operand's unit and then fails casting the other
    input into it, so the user sees a casting error rather than a metadata one.
    """
    from .._ufunc import ALL as _UFUNCS
    from .. import dtype as _dtype
    ufunc = _UFUNCS[ufunc_name]
    return _UFuncInputCastingError(
        ufunc, casting, _dtype(from_str), _dtype(to_str), i)


def _install_error_factories():
    # The engine-side hook is optional: an older/mid-rebuild `_rnp` may not
    # export it yet.  Registering is a pure enhancement (it upgrades the
    # engine's plain-string error into numpy's exact `_UFuncNoLoopError`
    # payload), so its absence must not take down the whole package.
    setter = getattr(_rnp, "_set_error_factories", None)
    if setter is None:
        return
    setter({"ufunc_no_loop": _no_loop_error,
            "ufunc_binary_resolution": _binary_resolution_error,
            "ufunc_input_casting": _input_casting_error})


_install_error_factories()
