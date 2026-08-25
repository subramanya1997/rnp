# Known gaps

The complete playbook passes on rnp, but building it exposed these narrower API
gaps. Each example uses an equivalent public NumPy formulation so the intended
workload still runs, and the alternatives are recorded here rather than silently
hidden.

## Structured-record convenience operations

- `np.sort(records, order="score")` raises `NotImplementedError: sort(order=) is
  not implemented yet`. `05_tabular_records.py` sorts the same structured records
  with `records[np.argsort(records["score"])]`.
- `np.lib.recfunctions.join_by(...)` produced correct payload columns but filled
  the joined integer key with `999999` in the exercised inner join. The example
  performs the same sorted inner join with `argsort` and `searchsorted`.

## Legacy polynomial-fit wrapper

- `np.polyfit(x, y, 2)` reaches an unimplemented private
  `numpy._core._multiarray_umath.linalg.lstsq` binding in rnp. `np.linalg.lstsq`
  itself works, as shown in `02_linear_algebra.py`, and `10_stats_pipeline.py`
  uses the supported `np.polynomial.polynomial.polyfit` API.

## Lazy testing namespace

- Accessing `np.testing` immediately after only `import numpy as np` raises
  `AttributeError` under rnp. An explicit `import numpy.testing` works and attaches
  the namespace. The examples use ordinary final `assert` statements; `run_all.py`
  performs cross-engine comparisons with the oracle process's `numpy.testing`
  helpers.

These gaps are outside the result paths compared by `run_all.py`; they do not
weaken any oracle comparison reported as passing.
