"""Discrete Fourier transforms backed by the Rust FFT kernels."""

import math as _math

from _rnp import _fft_c2c, _fft_c2r, _fft_r2c
from rnp_numpy import asarray, conjugate

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


def _pending(*args, **kwargs):
    raise NotImplementedError("FFT transform cluster not implemented yet")


fftn = ifftn = fft2 = ifft2 = _pending
rfftn = irfftn = rfft2 = irfft2 = _pending
fftfreq = rfftfreq = fftshift = ifftshift = _pending

for _name in __all__:
    globals()[_name].__module__ = "numpy.fft"
