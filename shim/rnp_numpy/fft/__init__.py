"""Discrete Fourier transforms backed by the Rust FFT kernels."""

import math as _math
import warnings as _warnings

from _rnp import _fft_c2c, _fft_c2r, _fft_r2c
from rnp_numpy import arange, asarray, conjugate, empty, integer, roll

__all__ = [
    "fft", "ifft", "rfft", "irfft", "hfft", "ihfft", "fftn", "ifftn",
    "fft2", "ifft2", "rfftn", "irfftn", "rfft2", "irfft2", "fftfreq",
    "rfftfreq", "fftshift", "ifftshift",
]


def _scale(n, norm, forward):
    if norm is None or norm == "backward":
        return 1.0 if forward else 1.0 / n
    if norm == "ortho":
        return 1.0 / _math.sqrt(n)
    if norm == "forward":
        return 1.0 / n if forward else 1.0
    raise ValueError(
        f'Invalid norm value {norm}; should be "backward","ortho" or "forward".'
    )


def _c2c(a, n, axis, norm, forward, out):
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    n = int(n)
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
    return _fft_c2c(a, n, axis, forward, _scale(n, norm, forward), out)


def fft(a, n=None, axis=-1, norm=None, out=None):
    return _c2c(a, n, axis, norm, True, out)


def ifft(a, n=None, axis=-1, norm=None, out=None):
    return _c2c(a, n, axis, norm, False, out)


def rfft(a, n=None, axis=-1, norm=None, out=None):
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    n = int(n)
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
    return _fft_r2c(a, n, axis, _scale(n, norm, True), out)


def irfft(a, n=None, axis=-1, norm=None, out=None):
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    n = int(n)
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
    return _fft_c2r(a, n, axis, _scale(n, norm, False), out)


_SWAP_DIRECTION_MAP = {"backward": "forward", None: "forward",
                       "ortho": "ortho", "forward": "backward"}


def _swap_direction(norm):
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        raise ValueError(
            f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".'
        ) from None


def hfft(a, n=None, axis=-1, norm=None, out=None):
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    return irfft(conjugate(a), n, axis, norm=_swap_direction(norm), out=out)


def ihfft(a, n=None, axis=-1, norm=None, out=None):
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    result = rfft(a, n, axis, norm=_swap_direction(norm), out=out)
    return conjugate(result, out=result)


def _cook_nd_args(a, s=None, axes=None, invreal=False):
    if s is None:
        shapeless = True
        if axes is None:
            s = list(a.shape)
        else:
            s = [a.shape[axis] for axis in axes]
    else:
        shapeless = False
    s = list(s)
    if axes is None:
        if not shapeless:
            _warnings.warn(
                "`axes` should not be `None` if `s` is not `None` "
                "(Deprecated in NumPy 2.0).",
                DeprecationWarning, stacklevel=3,
            )
        axes = list(range(-len(s), 0))
    else:
        axes = list(axes)
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    if None in s:
        _warnings.warn(
            "Passing an array containing `None` values to `s` is deprecated in NumPy 2.0.",
            DeprecationWarning, stacklevel=3,
        )
    s = [a.shape[axis] if size == -1 else size for size, axis in zip(s, axes)]
    return s, axes


def _raw_fftnd(a, s=None, axes=None, function=fft, norm=None, out=None):
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    for ii in range(len(axes) - 1, -1, -1):
        a = function(a, n=s[ii], axis=axes[ii], norm=norm, out=out)
    return a


def fftn(a, s=None, axes=None, norm=None, out=None):
    return _raw_fftnd(a, s, axes, fft, norm, out=out)


def ifftn(a, s=None, axes=None, norm=None, out=None):
    return _raw_fftnd(a, s, axes, ifft, norm, out=out)


def fft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return _raw_fftnd(a, s, axes, fft, norm, out=out)


def ifft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return _raw_fftnd(a, s, axes, ifft, norm, out=out)


def rfftn(a, s=None, axes=None, norm=None, out=None):
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes)
    a = rfft(a, s[-1], axes[-1], norm, out=out)
    for ii in range(len(axes) - 2, -1, -1):
        a = fft(a, s[ii], axes[ii], norm, out=out)
    return a


def rfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return rfftn(a, s, axes, norm, out=out)


def irfftn(a, s=None, axes=None, norm=None, out=None):
    a = asarray(a)
    s, axes = _cook_nd_args(a, s, axes, invreal=True)
    for ii in range(len(axes) - 1):
        a = ifft(a, s[ii], axes[ii], norm)
    return irfft(a, s[-1], axes[-1], norm, out=out)


def irfft2(a, s=None, axes=(-2, -1), norm=None, out=None):
    return irfftn(a, s, axes, norm, out=out)


_INTEGER_TYPES = (int, integer)


def fftshift(x, axes=None):
    x = asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, _INTEGER_TYPES):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return roll(x, shift, axes)


def ifftshift(x, axes=None):
    x = asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, _INTEGER_TYPES):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return roll(x, shift, axes)


def _validate_device(device):
    if device is not None and device != "cpu":
        raise ValueError(f'Device not understood. Only "cpu" is allowed, but received: {device}')


def fftfreq(n, d=1.0, device=None):
    if not isinstance(n, _INTEGER_TYPES):
        raise ValueError("n should be an integer")
    _validate_device(device)
    val = 1.0 / (n * d)
    result = empty(n, int)
    split = (n - 1) // 2 + 1
    result[:split] = arange(0, split, dtype=int)
    result[split:] = arange(-(n // 2), 0, dtype=int)
    return result * val


def rfftfreq(n, d=1.0, device=None):
    if not isinstance(n, _INTEGER_TYPES):
        raise ValueError("n should be an integer")
    _validate_device(device)
    val = 1.0 / (n * d)
    n = n // 2 + 1
    result = arange(0, n, dtype=int)
    return result * val


for _name in __all__:
    globals()[_name].__module__ = "numpy.fft"
