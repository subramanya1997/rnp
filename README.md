# rnp — a Rust implementation of NumPy's core

A from-scratch Rust port of NumPy, validated against **NumPy v2.5.2's own unmodified
test suite** and benchmarked against the real NumPy wheel on the same machine.

- **95.5% of NumPy's entire test suite passing** — 46,330 / 48,511 tests
- **Bit-for-bit identical to real NumPy** across 73,879 differential byte-level checks (0 divergences)
- **Faster than NumPy** on most large-array operations (transcendentals 4–6.5×, integer matmul up to 8.6×, boolean masking ~2.5×), at parity on BLAS paths

Measured 2026-08-24 on macOS arm64 against `numpy==2.5.2` (Accelerate BLAS).

---

## Scorecard

Every number comes from running NumPy's own test files, unmodified, against the Rust
engine through an import shim. Real NumPy passes these by definition, so the pass rate
is literal distance-to-parity.

| Suite | Passing | Share |
|---|---:|---:|
| `linalg` | 485 / 485 | **100%** |
| `fft` | 159 / 159 | **100%** |
| `polynomial` | 609 / 610 | 99.8% |
| `random` | 1,341 / 1,349 | 99.4% |
| `ma` (masked arrays) | 4,219 / 4,376 | 96.4% |
| `_core` (the engine) | 34,924 / 36,357 | 96.1% |
| `matrixlib` | 189 / 209 | 90.4% |
| `lib` | 4,272 / 4,759 | 89.8% |
| top-level / testing | 132 / 207 | 63.8% |
| **Grand total** | **46,330 / 48,511** | **95.5%** |

Out of scope by design: `f2py` (tests the Fortran toolchain), `typing` stubs, and
PyInstaller packaging tests. The realistic ceiling without implementing NumPy's C ABI
is ~96–97%; most remaining failures are C-extension fixtures a Rust port cannot pass
by definition.

## Correctness: bit-for-bit, not approximately

Stricter than the test suite, a differential harness compares the port against the
installed real NumPy element-by-element at the byte level:

| Checker | Comparisons | Divergences |
|---|---:|---:|
| General ops (indexing, ufuncs, reductions) | 36,504 | 0 |
| Structured dtypes | 16,922 | 0 |
| NEP 50 promotion | 12,094 | 0 |
| Matmul family | 3,862 | 0 |
| Object dtype | 2,641 | 0 |
| Stragglers | 1,697 | 0 |

This holds because the port calls the **same libraries NumPy calls** where NumPy
delegates: Apple Accelerate ILP64 BLAS for float matmul/dot, Accelerate LAPACK for all
of `linalg` (raw-bytes probes across solve/eig/SVD/QR/Cholesky match exactly), a
faithful Rust transcription of NumPy's vendored pocketfft for FFT, and exact
transcriptions of the random BitGenerators and distributions — every seed produces
NumPy's exact stream, including legacy `RandomState`.

## Performance

Paired timings against the real NumPy wheel — see [`benchmarks/REPORT.md`](benchmarks/REPORT.md)
for the full table and methodology. Highlights at 1M elements (ratio = port / NumPy;
lower is faster):

| Operation | Ratio | |
|---|---:|---|
| `power` (f64) | 0.15× | 6.5× faster |
| `sin` / `log` / `exp` (f64) | 0.19–0.25× | 4–5× faster |
| matmul int32 256² | 0.12× | 8.6× faster |
| boolean mask / setitem | 0.37–0.49× | ~2.5× faster |
| `add.reduce` (f64) | 0.64× | 1.6× faster |
| matmul f64 512², dot 256², vecdot 1M | 0.97–1.02× | BLAS parity |
| matmul 32² / matvec | 1.05–1.25× | small-size dispatch, ~0.3 µs |
| scalar ops / slice views | 1.5–7× | fixed sub-µs overhead; needs native scalar types |

NumPy is single-threaded for elementwise work; the port parallelizes with Rayon above
a size threshold **without changing any result bit** (reductions keep NumPy's exact
pairwise summation tree).

## Is it production-ready?

**Production-quality core, beta product.** For pure-Python numerical workloads inside
the tested surface — arrays, ufuncs, linalg, fft, random, masked arrays — the port is
functionally interchangeable with NumPy at the 95.5% level, bit-exact where measured,
and frequently faster. It is **not** yet a drop-in replacement for arbitrary
third-party code:

- **No C ABI.** SciPy, pandas, and scikit-learn reach NumPy through its C API; the
  port exposes only the Python surface.
- **Object/string interning leak.** Object-dtype and StringDType cells intern into an
  append-only slab (deliberate correctness-first trade); long-running churn of object
  arrays leaks memory until per-element refcounting lands.
- **Threading unaudited.** The multithreading test file does not collect yet.
- **Linux x86_64: 99.975% bit-exact.** The Linux build (dlopening numpy's own
  bundled OpenBLAS) passes the differential harness at 36,506 comparisons with
  9 remaining divergences — all 80-bit x87 `longdouble` storage items,
  documented in [`harness/LINUX_PARITY.md`](harness/LINUX_PARITY.md). macOS
  arm64 remains fully bit-exact.

## Layout

```
rnp/                Rust workspace
  crates/rnp-core     pure-Rust ndarray engine (dtypes, strides, ufuncs, FFT, busday…)
  crates/rnp-python   PyO3 extension exposing the NumPy-compatible surface
shim/               Python package presenting the numpy API, backed by the engine
harness/            runs numpy's unmodified tests against the shim + differential checkers
benchmarks/         paired port-vs-NumPy timings (see benchmarks/README.md)
examples/           runnable real-world NumPy workloads (see examples/README.md)
upstream/           shallow clone of numpy v2.5.2 — the read-only oracle (not committed)
PLAN.md             the full milestone log (M0–M8)
```

## Reproducing

```bash
# one-time: the oracle and a venv
git clone --depth 1 --branch v2.5.2 https://github.com/numpy/numpy upstream
python3.13 -m venv .venv && .venv/bin/pip install numpy==2.5.2 pytest pytest-json-report maturin hypothesis

# build the extension
cd rnp && ../.venv/bin/maturin develop --release -m crates/rnp-python/Cargo.toml && cd ..

# the full suite, per-suite scoreboards
.venv/bin/python harness/run.py --full

# bit-exactness gates
.venv/bin/python harness/dev_check.py

# benchmarks
.venv/bin/python benchmarks/run.py
```

## Rules the project was built under

- `upstream/` is read-only. **The tests are never modified** — the port is made to pass
  the tests, never the other way around.
- When port and oracle disagree, the port is wrong.
- Zero divergences on the differential checkers is the merge gate; a faster wrong
  answer is a failure.
- No `unsafe` in `rnp-core` without a `// SAFETY:` justification.

## Attribution

This project ports and validates against [NumPy](https://github.com/numpy/numpy)
(BSD-3-Clause, © 2005-2025 NumPy Developers). The FFT implementation transcribes
[pocketfft](https://github.com/mreineck/pocketfft); random-number generation
transcribes NumPy's vendored generator and distribution algorithms. Issue templates
are adapted from numpy/numpy.
