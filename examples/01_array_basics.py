"""Array creation, dtypes, indexing, reshaping, and broadcasting."""

import numpy as np


TOLERANCES = {}


def results():
    grid = np.arange(12, dtype=np.int64).reshape(3, 4)
    unit_interval = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    sliced = grid[::2, 1::2]
    fancy = grid[[2, 0], [1, 3]]
    broadcast_sum = grid[:, :2] + np.array([10, 100], dtype=np.int64)
    return {
        "grid": grid,
        "unit_interval": unit_interval,
        "sliced": sliced,
        "fancy": fancy,
        "broadcast_sum": broadcast_sum,
    }


def main():
    out = results()
    print("grid shape/dtype:", out["grid"].shape, out["grid"].dtype)
    print("slice:", out["sliced"])
    print("fancy index:", out["fancy"])
    print("broadcast row 0:", out["broadcast_sum"][0])
    assert np.array_equal(out["sliced"], [[1, 3], [9, 11]])
    assert np.array_equal(out["fancy"], [9, 3])
    assert np.array_equal(out["broadcast_sum"][-1], [18, 109])
    assert np.array_equal(out["unit_interval"], [0.0, 0.25, 0.5, 0.75, 1.0])


if __name__ == "__main__":
    main()
