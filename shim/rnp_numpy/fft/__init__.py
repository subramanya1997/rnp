"""Discrete Fourier transforms backed by the Rust FFT kernels."""

import math as _math

from _rnp import _fft_c2c
from rnp_numpy import asarray

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


def _pending(*args, **kwargs):
    raise NotImplementedError("FFT transform cluster not implemented yet")


rfft = irfft = hfft = ihfft = _pending
fftn = ifftn = fft2 = ifft2 = _pending
rfftn = irfftn = rfft2 = irfft2 = _pending
fftfreq = rfftfreq = fftshift = ifftshift = _pending

for _name in __all__:
    globals()[_name].__module__ = "numpy.fft"
