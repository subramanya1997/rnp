"""Missing-data analysis with NumPy masked arrays."""

import numpy as np


TOLERANCES = {"column_means": (1e-12, 1e-12), "imputed": (1e-12, 1e-12)}


def results():
    readings = np.array([
        [18.0, 45.0, 1012.0],
        [19.5, np.nan, 1011.0],
        [np.nan, 48.0, 1010.0],
        [21.0, 52.0, np.nan],
    ])
    masked = np.ma.masked_invalid(readings)
    column_means = masked.mean(axis=0)
    imputed = masked.filled(column_means)
    hot_and_valid = np.ma.masked_where(masked[:, 0] < 20.0, masked[:, 0])
    return {
        "mask": np.ma.getmaskarray(masked),
        "valid_counts": masked.count(axis=0),
        "column_means": column_means.filled(np.nan),
        "imputed": imputed,
        "hot_values": hot_and_valid.compressed(),
    }


def main():
    out = results()
    print("valid counts:", out["valid_counts"])
    print("column means:", np.round(out["column_means"], 4))
    print("hot valid values:", out["hot_values"])
    assert np.array_equal(out["valid_counts"], [3, 3, 3])
    assert np.allclose(out["column_means"], [19.5, 145.0 / 3.0, 1011.0], rtol=0.0, atol=1e-12)
    assert np.array_equal(out["hot_values"], [21.0])


if __name__ == "__main__":
    main()
