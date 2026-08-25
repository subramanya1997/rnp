# benchmarks

Paired micro-benchmarks: every case is timed on **real NumPy** and on **the rnp port**
in the same process group, same venv, same machine. No synthetic baselines — the
comparison target is always the installed `numpy==2.5.2` wheel.

## Running

```bash
.venv/bin/python benchmarks/run.py
```

Prints the full table and writes machine-readable results to `results.json`.

## Methodology

- Each case runs a warmup, then takes the median of repeated timed batches
  (batch sizes auto-scaled so each measurement is well above timer resolution).
- Operands are preconstructed outside the timed closure; the timed region is the
  operation only.
- `ratio` = port time / NumPy time. **Below 1.0 = the port is faster.**
- Correctness is enforced separately: `harness/dev_check.py` must report zero
  divergences on the same build before any benchmark number is considered valid.
  A faster wrong answer is a failure, not a win.

## Cases

- **Elementwise / indexing** at 1K and 1M elements: slicing, fancy and boolean
  indexing (get and set), `take`, copies, casts, transcendentals (`exp`, `sin`,
  `log`, `sqrt`, `power`), integer ops, reductions, scalar-op overhead probes.
- **Matmul family**: `matmul` (f64/f32/i32 at 32²–512², batched), `dot`, `vecdot`,
  `matvec` — the float paths route through the same Accelerate BLAS NumPy uses,
  so parity is the expected result there; integer matmul has no BLAS route and
  runs the port's own blocked kernels.

## Latest results

See [`REPORT.md`](REPORT.md) for the current full table with commentary, and
`results.json` for the raw numbers behind it.
