#!/usr/bin/env python3
"""Byte-for-byte differential probe for NumPy's FFT family."""

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "shim"))

import rnp_numpy as rnp  # noqa: E402


SIZES = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 15, 16, 17, 25,
         32, 49, 64, 97, 100, 227, 1000, 1024]
NORMS = [None, "backward", "ortho", "forward"]


def port_array(value):
    return rnp.array(value.tolist(), dtype=str(value.dtype))


def same_bytes(label, expected, actual, failures):
    expected_bytes = expected.tobytes()
    actual_bytes = actual.tobytes()
    if (str(expected.dtype), expected.shape, expected_bytes) != (
        str(actual.dtype), actual.shape, actual_bytes
    ):
        failures.append(
            f"{label}: expected {expected.dtype}{expected.shape}, "
            f"got {actual.dtype}{actual.shape}"
        )


def main():
    rng = np.random.default_rng(0x50C4E7)
    failures = []
    checks = 0

    for dtype in (np.complex128, np.complex64):
        for n in SIZES:
            data = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
            port = port_array(data)
            for norm in NORMS:
                for name in ("fft", "ifft"):
                    same_bytes(
                        f"{name} dtype={dtype.__name__} n={n} norm={norm}",
                        getattr(np.fft, name)(data, norm=norm),
                        getattr(rnp.fft, name)(port, norm=norm),
                        failures,
                    )
                    checks += 1

    for real_dtype, complex_dtype in (
        (np.float64, np.complex128),
        (np.float32, np.complex64),
    ):
        for n in SIZES:
            real = rng.standard_normal(n).astype(real_dtype)
            half = (
                rng.standard_normal(n // 2 + 1)
                + 1j * rng.standard_normal(n // 2 + 1)
            ).astype(complex_dtype)
            port_real = port_array(real)
            port_half = port_array(half)
            for norm in NORMS:
                cases = (
                    ("rfft", (real,), (port_real,), {}),
                    ("irfft", (half,), (port_half,), {"n": n}),
                    ("hfft", (half,), (port_half,), {"n": n}),
                    ("ihfft", (real,), (port_real,), {"n": n}),
                )
                for name, oracle_args, port_args, kwargs in cases:
                    same_bytes(
                        f"{name} dtype={real_dtype.__name__} n={n} norm={norm}",
                        getattr(np.fft, name)(*oracle_args, norm=norm, **kwargs),
                        getattr(rnp.fft, name)(*port_args, norm=norm, **kwargs),
                        failures,
                    )
                    checks += 1

    nd_cases = (
        ("fftn", (3, 4, 5), None, None),
        ("ifftn", (3, 4, 5), (5, 3), (2, 0)),
        ("fft2", (4, 3, 7), (6, 2), (0, 2)),
        ("ifft2", (4, 3, 7), None, (0, 2)),
        ("rfftn", (3, 4, 5), None, None),
        ("irfftn", (3, 4, 3), (3, 4, 5), (0, 1, 2)),
        ("rfft2", (4, 3, 7), (6, 2), (0, 2)),
        ("irfft2", (4, 3, 2), (4, 3), (0, 2)),
    )
    for dtype in (np.float64, np.float32):
        for name, shape, transform_shape, axes in nd_cases:
            if name.startswith("i"):
                complex_dtype = np.complex64 if dtype is np.float32 else np.complex128
                data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
                    complex_dtype
                )
            else:
                data = rng.standard_normal(shape).astype(dtype)
            port = port_array(data)
            kwargs = {"s": transform_shape, "axes": axes}
            for norm in NORMS:
                same_bytes(
                    f"{name} dtype={dtype.__name__} shape={shape} norm={norm}",
                    getattr(np.fft, name)(data, norm=norm, **kwargs),
                    getattr(rnp.fft, name)(port, norm=norm, **kwargs),
                    failures,
                )
                checks += 1

    if failures:
        print(f"fft bit-exact: {checks - len(failures)}/{checks}")
        print("\n".join(failures[:20]))
        raise SystemExit(1)
    print(f"fft bit-exact: {checks}/{checks}")


if __name__ == "__main__":
    main()
