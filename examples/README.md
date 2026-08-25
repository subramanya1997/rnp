# rnp example playbook

The executed notebooks are the showcase for RNP's NumPy-compatible workflows.
They redirect an ordinary `import numpy as np` to the RNP shim, print the active
engine, use deterministic data, and store every real output and final passing
assertion for GitHub to render.

The matching self-contained scripts remain the verification layer. Each script
can run unchanged against either NumPy or RNP, while `run_all.py` proves that the
two engines produce matching results.

| Showcase notebook | Verification script | Workload |
|---|---|---|
| [`01_array_basics.ipynb`](notebooks/01_array_basics.ipynb) | `01_array_basics.py` | Creation, explicit dtypes, reshape, slicing, fancy indexing, and broadcasting |
| [`02_linear_algebra.ipynb`](notebooks/02_linear_algebra.ipynb) | `02_linear_algebra.py` | Linear solve, least-squares polynomial fitting, eigendecomposition/PCA, SVD, and QR |
| [`03_signal_fft.ipynb`](notebooks/03_signal_fft.ipynb) | `03_signal_fft.py` | FFT low-pass filtering and overlapping-window spectral analysis |
| [`04_monte_carlo.ipynb`](notebooks/04_monte_carlo.ipynb) | `04_monte_carlo.py` | Seeded Monte Carlo, bootstrap confidence intervals, `default_rng`, and `RandomState` |
| [`05_tabular_records.ipynb`](notebooks/05_tabular_records.ipynb) | `05_tabular_records.py` | Structured arrays, generated CSV input, `loadtxt`/`genfromtxt`, sorting, and record joins |
| [`06_masked_data.ipynb`](notebooks/06_masked_data.ipynb) | `06_masked_data.py` | Masking invalid sensor readings, reductions, filling, and compressed selections |
| [`07_strings_text.ipynb`](notebooks/07_strings_text.ipynb) | `07_strings_text.py` | `StringDType` arrays and vectorized `np.strings` cleanup |
| [`08_einsum_ml.ipynb`](notebooks/08_einsum_ml.ipynb) | `08_einsum_ml.py` | Attention-style contractions, stable softmax, and batched matrix multiplication |
| [`09_image_ops.ipynb`](notebooks/09_image_ops.ipynb) | `09_image_ops.py` | Stride-trick convolution, downsampling, and image normalization |
| [`10_stats_pipeline.ipynb`](notebooks/10_stats_pipeline.ipynb) | `10_stats_pipeline.py` | Histograms, percentiles/quantiles, NaN-aware statistics, and polynomial fitting |

## Run the playbook

From the repository root:

```bash
.venv/bin/python examples/run_all.py
```

The runner is the dual-engine proof: it launches every script in isolated
subprocesses twice. The NumPy run uses the installed `numpy==2.5.2` wheel as the
oracle. The rnp run prepends
`shim/` and `harness/_redirect/` to `PYTHONPATH`, exactly like `harness/run.py`,
so an ordinary `import numpy` resolves to rnp. It captures each script's printed
output and `results()` dictionary, then checks exact operations with
`numpy.testing.assert_array_equal` and explicitly marked numerical routines with
`numpy.testing.assert_allclose`.

The `RNP` column also verifies that `import numpy` resolved to the
`rnp_numpy` module, preventing a false pass caused by accidentally running the
oracle twice.

You can also run any script directly against real NumPy:

```bash
.venv/bin/python examples/03_signal_fft.py
```

See [`KNOWN_GAPS.md`](KNOWN_GAPS.md) for any workload that cannot currently be represented faithfully on rnp.
