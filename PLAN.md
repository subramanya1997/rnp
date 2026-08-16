# PLAN.md — rnp milestone plan (owner: Fable 5)

Goal: a Rust implementation of NumPy's core that passes NumPy v2.5.2's own
unmodified tests, cross-verified and benchmarked against real NumPy 2.5.2.

Strategy: NumPy's full surface is enormous (69 test files in `_core/tests`
alone, plus linalg/fft/random/lib). We do NOT try to pass everything at once.
We build the engine bottom-up, run the *whole* upstream core test suite every
milestone, and drive the scoreboard (percent of tests passing) monotonically
upward. A test file is "adopted" when ≥95% of its tests pass; adopted files
become regression gates.

## Architecture

- `rnp-core` (pure Rust, no Python):
  - `dtype`: descriptor model mirroring NumPy — kind, itemsize, byteorder,
    type-promotion lattice (NEP 50 semantics, since 2.x tests assume it).
    Start: bool, int8/16/32/64, uint8/16/32/64, float32/64, complex64/128.
    Later: float16, datetime64/timedelta64, str_/bytes_, structured dtypes.
  - `ndarray`: header = {data: Arc<Buffer>, offset, shape, strides, dtype,
    flags}. Views are first-class (slicing never copies). C/F-order aware.
  - `iter`: strided iteration + broadcasting (nditer-lite), inner-loop
    specialization for contiguous cases.
  - `ufunc`: typed inner loops + dispatch table + promotion; unary/binary,
    reductions (add.reduce etc.), `out=`, `where=`, casting rules.
  - `casting`: same-kind/safe/unsafe/equiv rules + cast loops.
- `rnp-python` (PyO3): `ndarray` pyclass, `dtype` pyclass, scalar types,
  buffer protocol + `__array_interface__` + DLPack (lets real numpy read our
  arrays during cross-verification), rich indexing, operator protocol.
- `shim/rnp_numpy`: pure-Python package assembling the `numpy`-shaped module
  (np.add, np.sum, np.zeros, numpy.testing passthrough where possible...).
  The harness installs an import hook mapping `numpy` -> `rnp_numpy` for the
  upstream-test subprocess only. Tests are never edited.

## Cross-verification (beyond the upstream tests)

`harness/crosscheck.py`: property-based differential tester. Hypothesis
generates (op, shapes, dtypes, values); runs both real numpy (in-process,
normal import) and the port (subprocess with redirected import); compares
bit-exact for integers/bool, ULP-tolerance for floats. Runs on every milestone.

## Benchmarks

`benchmarks/run.py`: paired timings (median of N, warmup) for: elementwise
add/mul (contiguous + strided, sizes 1e3/1e6/1e8), reductions (sum/max, axis
and full), broadcasting ops, matmul, sort, copy/astype. Output: table with
ratio port/numpy. Criterion micro-benches live in `rnp/` for inner loops.

## Milestones

- **M0 — scaffold (this session)**: workspace builds, maturin develop works,
  `rnp.ndarray` exists with shape/strides/dtype, zeros/ones/arange/asarray for
  the 12 base dtypes, buffer protocol, repr. Harness runs upstream
  `test_shape_base`-class files and reports a scoreboard (mostly failing — fine).
- **M1 — dtype system + promotion**: NEP 50 promotion table, casting rules,
  `np.dtype()` constructor forms. Target file: `test_dtype.py` (subset),
  `test_numerictypes.py`.
- **M2 — indexing/views**: basic+advanced indexing, slicing views, boolean
  masks, `test_indexing.py` adoption.
- **M3 — ufuncs core**: arithmetic/comparison/logical ufuncs with broadcasting,
  `out=`, NEP 50 scalar behavior. Targets: `test_umath.py` (subset),
  `test_scalarmath.py` (subset), crosscheck suite green.
- **M4 — reductions + shape ops**: sum/prod/min/max/argmin/argmax/mean/std,
  reshape/transpose/concatenate/stack. Targets: `test_shape_base.py`,
  `test_multiarray.py` (subset).
- **M5 — printing + creation breadth**: arrayprint, linspace/eye/full/like-
  variants, `test_arrayprint.py`, `test_array_coercion.py` (subset).
- **M6 — sorting/searching, matmul**: sort/argsort/searchsorted, matmul via
  faer or hand-rolled blocked kernel. `test_matmul` portions of multiarray.
- **M7+ — breadth**: einsum, fft, linalg (LAPACK via faer), random (bit-exact
  MT19937/PCG64 streams), datetime64, strings, structured dtypes. Each gets its
  own milestone written when reached.

Every milestone ends with: full-suite scoreboard run, crosscheck run,
benchmark run, Fable review, git commit.

## Status log

- 2026-08-16: repo initialized, upstream v2.5.2 cloned, venv ready. M0 started.
- 2026-08-16: M0 complete (Opus build, Fable verified). cargo test 46/46;
  crosscheck 537/537 vs real numpy 2.5.2; test_shape_base 20/212; elementwise
  benchmarks ~1.0-1.6x numpy (reductions still Python placeholders). Notable
  fidelity work: FMA-contracted arange, Smith's algorithm for complex divide,
  NEP 50 promotion computed from rules and tested against the generated table.
- 2026-08-16: M1 started (Opus). Scope: dtype constructor breadth (byte order,
  S/U/V, structured basics so test_dtype.py collects), promote_types/
  result_type/can_cast/min_scalar_type, and native Rust full reductions
  (sum/prod/min/max) to retire the placeholder 550x benchmark rows.
