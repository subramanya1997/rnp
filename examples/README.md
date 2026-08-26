# examples

Ten real-world NumPy workloads, each an **executed Jupyter notebook** running on
the rnp engine — every cell's output is stored, so the results are visible right
here on GitHub. Each notebook opens with the one-line swap:

```python
import rnp as np
```

and everything after that is ordinary NumPy code, ending with assertions on
concrete values (all passing, and verified to match real NumPy 2.5.2 exactly).

| Notebook | Workload |
|---|---|
| [01_array_basics](01_array_basics.ipynb) | creation, dtypes, reshape, slicing, fancy indexing, broadcasting |
| [02_linear_algebra](02_linear_algebra.ipynb) | least-squares fitting, PCA via eig/SVD, QR |
| [03_signal_fft](03_signal_fft.ipynb) | filtering a noisy signal, windowed spectra |
| [04_monte_carlo](04_monte_carlo.ipynb) | seeded simulation, bootstrap CIs, `default_rng` + legacy `RandomState` |
| [05_tabular_records](05_tabular_records.ipynb) | structured arrays, loadtxt/genfromtxt, sorting, joins |
| [06_masked_data](06_masked_data.ipynb) | `np.ma` workflows with missing data |
| [07_strings_text](07_strings_text.ipynb) | `np.strings` + StringDType cleanup pipeline |
| [08_einsum_ml](08_einsum_ml.ipynb) | einsum attention-style contractions, softmax, batched matmul |
| [09_image_ops](09_image_ops.ipynb) | 2-d convolution via stride tricks, downsampling, normalization |
| [10_stats_pipeline](10_stats_pipeline.ipynb) | histograms, quantiles, nan-handling, polyfit |

To re-execute locally: build rnp (see the repo README), then open any notebook
with the project venv's Jupyter — the first cell adds `shim/` to `sys.path`.
