# Closed known gaps

The complete playbook exposed four narrower API gaps. They are now fixed; the
examples retain their equivalent public NumPy formulations so their compared
result paths remain unchanged.

## Structured-record convenience operations

- Fixed: `np.sort(records, order="score")` and structured `argsort` now honor a
  single field or a multi-field sequence. Unnamed fields remain declaration-order
  tie breakers, and stable sorting preserves fully equal records. The alternative
  field-`argsort` formulation remains in `05_tabular_records.py`.
- Fixed: `np.lib.recfunctions.join_by(...)` now writes joined integer keys into
  both the masked data and its parent field mask, rather than replacing them with
  the integer fill value `999999`. Inner, outer, and left-outer joins retain the
  same key ordering and missing-side masks as NumPy.

## Legacy polynomial-fit wrapper

- Fixed: legacy `np.polyfit(x, y, 2)` now resolves its private least-squares
  dependency to rnp's implemented `np.linalg.lstsq`. Explicit `rcond` values,
  full diagnostics, weighting, and covariance modes follow NumPy. The newer
  polynomial API remains in `10_stats_pipeline.py`.

## Lazy testing namespace

- Fixed: accessing `np.testing` immediately after only `import numpy as np` now
  imports and caches `rnp_numpy.testing`, matching NumPy's lazy namespace. The
  examples still use ordinary final `assert` statements; `run_all.py` performs
  cross-engine comparisons with the oracle process's `numpy.testing` helpers.

All four original repros now execute through rnp without their documented
failures.
