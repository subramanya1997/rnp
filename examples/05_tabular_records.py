"""Structured records, CSV parsing, sorting, and key-based joins."""

import io

import numpy as np


TOLERANCES = {"loaded_numeric": (1e-12, 1e-12), "joined_bonus": (1e-12, 1e-12)}


def _numeric_csv():
    table = np.array([[101.0, 8.5], [102.0, 9.25], [103.0, 7.75]])
    buffer = io.StringIO()
    np.savetxt(buffer, table, delimiter=",", fmt=["%.0f", "%.2f"])
    buffer.seek(0)
    return buffer


def _record_csv():
    rows = ["id,name,score", "103,Linus,7.75", "101,Ada,8.50", "102,Grace,9.25"]
    return io.StringIO("\n".join(rows))


def results():
    loaded = np.loadtxt(_numeric_csv(), delimiter=",")
    records = np.genfromtxt(_record_csv(), delimiter=",", names=True, dtype=None, encoding=None)
    sorted_records = records[np.argsort(records["score"])[::-1]]
    bonuses = np.array([(101, 1.5), (102, 2.0), (103, 1.0)], dtype=[("id", "i8"), ("bonus", "f8")])
    left = records[np.argsort(records["id"])]
    right = bonuses[np.argsort(bonuses["id"])]
    right_rows = np.searchsorted(right["id"], left["id"])
    matched = right["id"][right_rows] == left["id"]
    return {
        "loaded_numeric": loaded,
        "sorted_ids": sorted_records["id"],
        "sorted_names": sorted_records["name"],
        "joined_ids": left["id"][matched],
        "joined_bonus": right["bonus"][right_rows[matched]],
    }


def main():
    out = results()
    print("loaded numeric rows:", out["loaded_numeric"])
    print("ranked names:", out["sorted_names"])
    print("joined bonuses:", out["joined_bonus"])
    assert np.allclose(out["loaded_numeric"], [[101.0, 8.5], [102.0, 9.25], [103.0, 7.75]], rtol=0.0, atol=0.0)
    assert np.array_equal(out["sorted_ids"], [102, 101, 103])
    assert np.array_equal(out["sorted_names"], ["Grace", "Ada", "Linus"])
    assert np.array_equal(out["joined_ids"], [101, 102, 103])


if __name__ == "__main__":
    main()
