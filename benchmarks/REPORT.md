# Benchmark report — rnp vs. real NumPy 2.5.2

Measured 2026-08-24 on macOS arm64 (Apple Silicon), `numpy==2.5.2` wheel with
Accelerate BLAS, rnp built `--release` with fat LTO and `codegen-units=1`.
Ratio = port µs / NumPy µs — **below 1.0 means the port is faster**. Raw numbers:
`results.json`. Correctness gate on the same build: `harness/dev_check.py` =
36,504 comparisons, 0 divergences.

## Elementwise, indexing, reductions (1M elements)

| Case | NumPy µs | Port µs | Ratio |
|---|---:|---:|---:|
| max (f64) | 80.7 | 79.9 | 0.99× |
| astype | 80.2 | 47.3 | **0.59×** |
| copy | 99.4 | 101.5 | 1.02× |
| strided add | 160.2 | 133.4 | **0.83×** |
| slice view | 0.08 | 0.58 | 7.02× |
| slice copy | 102.2 | 104.4 | 1.02× |
| fancy 1-d gather | 129.1 | 105.3 | **0.82×** |
| fancy 2-d row gather | 116.9 | 70.6 | **0.60×** |
| boolean mask | 1450.8 | 531.1 | **0.37×** |
| fancy setitem | 221.7 | 238.1 | 1.07× |
| boolean setitem | 1077.5 | 527.6 | **0.49×** |
| take (axis 0) | 83.0 | 90.7 | 1.09× |
| exp (f64) | 1428.3 | 352.0 | **0.25×** |
| exp (f32) | 1195.3 | 308.7 | **0.26×** |
| sin (f64) | 1947.5 | 367.6 | **0.19×** |
| sqrt (f64) | 245.5 | 133.6 | **0.54×** |
| log (f64) | 1719.5 | 343.5 | **0.20×** |
| power (f64) | 4478.8 | 692.8 | **0.15×** |
| abs (f64) | 106.5 | 104.9 | 0.98× |
| negative (f64) | 107.8 | 106.3 | 0.99× |
| maximum (f64) | 182.1 | 151.5 | **0.83×** |
| floor_divide (i32) | 262.7 | 274.2 | 1.04× |
| bitwise_and (i32) | 51.8 | 62.7 | 1.21× |
| add.reduce (f64) | 115.4 | 73.5 | **0.64×** |
| scalar add (f64) | 0.17 | 0.54 | 3.27× |
| scalar add (i64) | 0.37 | 0.58 | 1.55× |
| scalar extract | 0.08 | 0.21 | 2.50× |

## Matmul family

| Case | NumPy µs | Port µs | Ratio |
|---|---:|---:|---:|
| matmul f64 32² | 1.54 | 2.75 | 1.78× |
| matmul f32 32² | 1.25 | 2.54 | 2.03× |
| matmul i32 32² | 10.3 | 6.3 | **0.61×** |
| matmul f64 128² | 12.4 | 13.7 | 1.11× |
| matmul f32 128² | 4.5 | 5.6 | 1.24× |
| matmul i32 128² | 671.1 | 127.8 | **0.19×** |
| matmul f64 256² | 77.7 | 78.8 | 1.01× |
| matmul f32 256² | 21.6 | 25.8 | 1.19× |
| matmul i32 256² | 9618.3 | 1116.4 | **0.12×** |
| matmul f64 512² | 333.7 | 335.0 | 1.00× |
| batched 4096×8×8 | 300.0 | 316.9 | 1.06× |
| vecdot f64 1M | 109.0 | 110.5 | 1.01× |
| matvec f64 512 | 5.7 | 7.5 | 1.31× |
| dot f64 256² | 79.3 | 92.8 | 1.17× |

## Reading the table

- **Where the port wins big** — transcendentals (4–6.5×), boolean masking (~2.5×),
  reductions (1.6×), integer matmul (up to 8.6×): NumPy is single-threaded for
  elementwise loops and has no BLAS route for integer matmul. The port parallelizes
  with Rayon above a size threshold, and reductions keep NumPy's exact pairwise
  summation tree so the speedup costs zero bits of accuracy.
- **Parity rows** — float matmul, `vecdot`, large `dot`: both sides call the same
  Accelerate GEMM under the same routing predicate, so parity (and bit-identity)
  is the designed outcome, not a coincidence.
- **Where the port loses** — scalar ops, slice views, tiny matmuls: a fixed
  ~0.3–0.5 µs of Python-side dispatch per call, visible only when the operation
  itself is sub-microsecond. Array-scale workloads never see it; scalar-heavy
  inner loops in Python will. This is the known remaining performance work.

## History

- First baseline (pre-optimization) had float matmul 20–55× slower (own packed
  loops vs Accelerate) and transcendentals ~1.6× faster (no parallelism). The
  optimization pass routed the float matmul family through Accelerate under
  NumPy's exact blasability predicate and added thresholded Rayon parallelism,
  producing the table above with the differential checkers still at zero.
