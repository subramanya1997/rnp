"""numpy.exceptions equivalents."""


class ComplexWarning(RuntimeWarning):
    """Cast of a complex value to a real one discards the imaginary part."""


class VisibleDeprecationWarning(UserWarning):
    """A deprecation the user should see by default."""


class RankWarning(UserWarning):
    """Ill-conditioned fit."""


class DTypePromotionError(TypeError):
    """Two dtypes have no common promoted type."""


class TooHardError(RuntimeError):
    pass


class AxisError(ValueError, IndexError):
    """Axis out of range, or otherwise invalid."""

    def __init__(self, axis, ndim=None, msg_prefix=None):
        if ndim is None and msg_prefix is None:
            msg = axis
        else:
            msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
            if msg_prefix is not None:
                msg = f"{msg_prefix}: {msg}"
        self.axis = axis
        self.ndim = ndim
        super().__init__(msg)


__all__ = [
    "AxisError",
    "ComplexWarning",
    "DTypePromotionError",
    "RankWarning",
    "TooHardError",
    "VisibleDeprecationWarning",
]
