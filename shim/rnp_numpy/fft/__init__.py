"""Minimal ``numpy.fft`` namespace used by public API introspection."""

__all__ = ["fft"]


def fft(a, n=None, axis=-1, norm=None, out=None):
    raise NotImplementedError("numpy.fft.fft requires a Rust FFT kernel")


fft.__module__ = "numpy.fft"
