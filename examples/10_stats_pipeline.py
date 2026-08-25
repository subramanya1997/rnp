"""A compact descriptive-statistics and polynomial-trend pipeline."""

import numpy as np


TOLERANCES = {
    "percentiles": (1e-12, 1e-12),
    "quantiles": (1e-12, 1e-12),
    "nan_summary": (1e-12, 1e-12),
    "polyfit": (1e-12, 1e-12),
}


def results():
    measurements = np.array([
        1.0, 2.0, np.nan, 4.0, 5.0, 7.0, 8.0, np.nan, 10.0, 12.0,
    ])
    valid = measurements[~np.isnan(measurements)]
    histogram, edges = np.histogram(valid, bins=np.array([0.0, 3.0, 6.0, 9.0, 12.1]))
    percentiles = np.nanpercentile(measurements, [10.0, 50.0, 90.0])
    quantiles = np.nanquantile(measurements, [0.25, 0.75])
    summary = np.array([np.nanmean(measurements), np.nanstd(measurements), np.nanmin(measurements), np.nanmax(measurements)])

    x = np.arange(7, dtype=np.float64)
    y = 1.5 * x * x - 0.5 * x + 2.0
    coefficients = np.polynomial.polynomial.polyfit(x, y, deg=2)[::-1]
    return {
        "histogram": histogram,
        "bin_edges": edges,
        "percentiles": percentiles,
        "quantiles": quantiles,
        "nan_summary": summary,
        "polyfit": coefficients,
    }


def main():
    out = results()
    print("histogram:", out["histogram"])
    print("10/50/90 percentiles:", out["percentiles"])
    print("nan-aware mean/std/min/max:", np.round(out["nan_summary"], 6))
    print("quadratic coefficients:", np.round(out["polyfit"], 6))
    assert np.array_equal(out["histogram"], [2, 2, 2, 2])
    assert np.allclose(out["percentiles"], [1.7, 6.0, 10.6], rtol=0.0, atol=1e-12)
    assert np.allclose(out["polyfit"], [1.5, -0.5, 2.0], rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    main()
