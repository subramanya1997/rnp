"""Synthetic-image convolution, downsampling, and range normalization."""

import numpy as np


TOLERANCES = {"convolved": (1e-12, 1e-12), "downsampled": (1e-12, 1e-12), "normalized": (1e-12, 1e-12)}


def results():
    y, x = np.mgrid[0:8, 0:10]
    image = 0.25 * x + 0.5 * y
    image[2:6, 3:7] += 5.0
    kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
    windows = np.lib.stride_tricks.sliding_window_view(image, (3, 3))
    convolved = np.einsum("ijxy,xy->ij", windows, kernel)
    downsampled = convolved[::2, ::2]
    normalized = (downsampled - downsampled.min()) / np.ptp(downsampled)
    return {
        "convolved": convolved,
        "downsampled": downsampled,
        "normalized": normalized,
    }


def main():
    out = results()
    print("convolution shape:", out["convolved"].shape)
    print("downsampled image:\n", out["downsampled"])
    print("normalized range:", out["normalized"].min(), out["normalized"].max())
    expected = [
        [0.75, 2.1875, 3.0, 2.5625],
        [1.75, 6.0, 7.75, 4.5],
        [2.75, 6.0625, 7.5, 5.1875],
    ]
    assert np.allclose(out["downsampled"], expected, rtol=0.0, atol=0.0)
    assert np.array_equal(np.array([out["normalized"].min(), out["normalized"].max()]), [0.0, 1.0])


if __name__ == "__main__":
    main()
