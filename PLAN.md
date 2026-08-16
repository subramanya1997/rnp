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
- 2026-08-16: M1 complete (Opus build). cargo test 70/70; dev_check 9971
  comparisons / 0 divergences; crosscheck green. Scoreboard: test_dtype
  660/890 (74.2%, was a collection crash), test_numerictypes 76/138 (55.1%,
  was a collection crash), test_shape_base 20/212 (unchanged); full suite
  968/2109 (45.9%).

  What landed:
  * `descr.rs` — the real descriptor model. `DType` keeps the storage type
    (now including `float16` and the flexible `S`/`U`/`V` kinds, plus interned
    ids for structured and subarray dtypes, so it stays `Copy`); `Descr` adds
    byte order and the `q`/`Q`/`c` C-type aliases that carry their own `num`
    and `char`. repr/str are transcriptions of `upstream/numpy/_core/_dtype.py`
    (packed-vs-dict struct forms, `align=True`, titles, nested and subarray
    fields). Equality/hash normalise `<` vs `=` and ignore the alias, as numpy
    does.
  * `casting.rs` — `can_cast` (all five kinds, numeric table generated from
    real numpy by `harness/gen_tables.py`, flexible rules probed), NEP 50
    `result_type` with weak python scalars, `min_scalar_type` transcribed from
    numpy's `min_scalar_type_num`, `common_type`, flexible `promote_types`.
  * `reduce.rs` — native sum/prod/min/max/argmin/argmax with `axis=` and
    `keepdims=`, plus `mean`. numpy's pairwise summation is reproduced exactly
    (8 accumulators, `PW_BLOCKSIZE` 128, `-0.0` tail seed, split rounded down
    to a multiple of 8), including the complex variant, so float sums are
    bit-identical. Which grouping applies also follows numpy's iterator:
    pairwise when the reduced axis ends up innermost, sequential slice
    accumulation otherwise.
  * Fidelity work found by the differential check: FMA-contracted complex
    multiply, `fmin`/`fmax` signed-zero preference in min/max (but not in
    argmin/argmax), numpy's float16 mean accumulating in float32, and the
    complex64 mean widening to double before dividing.
  * Performance: `lto = "fat"`, rayon-parallel reductions and elementwise
    loops above 2^16 elements. The float sums split *on numpy's own pairwise
    tree boundaries*, so parallelism does not change a single rounding.
    1e6-element ratios (port/numpy, lower is better): sum_f64 0.49x,
    max_f64 0.80x, add_f64 0.77x, mul_f64 0.66x, astype 0.92x, copy 0.99x,
    add_i32 1.15x, strided_add 1.41x. Integer sums run 3-4x faster than
    numpy. (The host is noisy; ratios move +-50% between runs.)

  Known gaps carried into later milestones: byte-swapped *arrays* raise
  NotImplementedError (the dtype objects round-trip correctly); structured
  arrays are limited to construction; `longdouble`/`clongdouble` alias
  float64/complex128; no real numpy scalar types yet, so whole-array
  reductions return Python numbers; `S`/`U` support comparisons but no
  string ufuncs.

- 2026-08-16: M2 started (Opus). Scope: basic + advanced indexing, views,
  item selection, flatiter, and the shim surface needed to unlock more
  upstream files for collection.
- 2026-08-16: M2 complete (Opus build). cargo test 77/77; dev_check 11215
  comparisons / 0 divergences (of which ~1250 are the new differential
  indexing section); crosscheck green. Full suite **1462/3565 (41.0%)**, up
  from the M1 baseline of 968; no adopted file regressed.

  Scoreboard deltas (passed/collected):
  * test_indexing        17/106 -> **60/106**
  * test_item_selection   0/244 -> **204/254**
  * test_indexerrors       2/8  -> **5/8**
  * test_dtype           660/894 -> 711/944
  * test_numerictypes     76/138 -> 77/138
  * newly collecting: test_umath 81/910, test_scalar_methods 97/215,
    test_api 2/54, test_print 8/22, test_scalarprint 3/30,
    test_mem_overlap 1/25.
  * unchanged: test_shape_base 20/213, test_unicode 45/76,
    test_scalar_ctors 142/201.

  What landed:
  * `indexing.rs` — numpy's whole index model. Basic indexing (int / slice /
    `...` / `None` / tuple, negative steps, 0-d results) always produces a
    view; advanced indexing produces a gather/scatter plan. The layout rules
    are transcribed from `mapping.c` and were probed case by case against
    numpy 2.5.2: plain integers become 0-d index arrays as soon as any
    advanced index is present, a `k`-d boolean becomes the `k` arrays
    `nonzero()` returns, a 0-d boolean contributes a length-0/1 index array
    and consumes no axis, and the broadcast shape is spliced into the
    subspace at the first advanced index only when the advanced indices are
    *consecutive* — where a newaxis counts as a separator just like a slice
    (verified: `a[:, [0,1], None, [0,1]]` on (7,3,4,5) is `(2,7,1,5)`, while
    `a[:, [0,1], [0,1], None]` is `(7,2,1,5)`).
  * Error messages are the real ones (probed, not guessed): out-of-bounds
    reports the *original* signed index, "too many indices for array: array
    is N-dimensional, but M were indexed", the boolean-shape mismatch, the
    index-array broadcast mismatch, and numpy's "only integers, slices ..."
    text for both arrays and flat iterators.
  * `a.base` is now real: `PyNdArray` carries the owning object and view
    chains collapse onto it, as numpy's do. Every view-producing method
    (`__getitem__`, `T`, `transpose`, `reshape`, `ravel`, `squeeze`,
    `swapaxes`, `view`, `.flat`) propagates it.
  * `np.flatiter` (`a.flat`): iteration, integer/slice/ellipsis/bool-mask/
    fancy indexing and assignment writing through to the base, `.base`,
    `.index`, `.coords`, `.copy()`, `__array__`, and numpy's two
    DeprecationWarnings (0-d boolean index, non-array float indices).
  * Item selection: `take` (axis=, mode=raise/wrap/clip, out=), `put`,
    `putmask`, `compress`, `choose`, `repeat`, `nonzero`, `flatnonzero`,
    `where` (1- and 3-argument), `ndarray.item`, `.fill`, plus `all`/`any`,
    `squeeze`, `swapaxes`, `view(dtype)`, `as_strided`, `broadcast_to`,
    `np.s_`/`index_exp`, `ndindex`, `ix_`, `indices`.
  * Collection unlock: `numpy._core.multiarray`, `numpy._core.umath`,
    `numpy._core._internal`, `numpy._core._umath_tests`,
    `numpy._core.tests._locales`, `numpy.random` (shape-correct but *not*
    stream-compatible), `numpy.lib.stride_tricks`, `numpy.lib.recfunctions`,
    `numpy.ma`, `numpy.__config__`, and the missing `numpy.testing` names
    (`_gen_alignment_data`, `run_threaded`, `_no_tracing`, ...). A `ufunc`
    class now backs `np.add` & co. so `isinstance(f, np.ufunc)` and
    `np.add.reduce` work; every unimplemented ufunc is a `ufunc` instance
    that raises NotImplementedError when *called*.
  * dtype breadth needed by those files: an `object` descriptor (`np.dtype(
    object)`, `'O'`, `O` fields inside structured dtypes — arrays of it are
    refused at creation, never silently mis-stored), `g`/`G` longdouble
    aliases carrying numpy's own num/char, and structured-array construction
    from tuples (`np.array([('a', 1)], dtype='S1,u4')`).
  * Verification: `harness/dev_check.py` grew a differential indexing
    section — a few hundred random index expressions (ints, slices with
    negative steps, ellipsis, newaxis, boolean masks over 1 and 2 axes, 1-d
    and 2-d integer arrays) are built once and replayed against both
    libraries, comparing values, dtype, shape, error type, *and* view-vs-copy
    semantics by writing through the result and diffing the base array;
    plus take/nonzero/where/flatnonzero parity.

  Performance (ratio port/numpy, lower is better; the benchmark harness now
  interleaves the two libraries' timing rounds — running one to completion
  first handed the second a warm allocator and was worth up to 3x):

    case             1e3      1e6
    add_f64         1.39x    0.70x
    mul_f64         1.48x    0.94x
    add_i32         1.54x    1.74x
    sum_f64         0.46x    0.99x
    max_f64         0.64x    1.61x
    astype          1.07x    1.00x
    copy            0.99x    0.99x
    strided_add     2.56x    0.91x   (was 1.34-1.50x at 1e6)
    slice_view      5.14x    3.00x   (0.4-1.0us absolute: PyO3 call cost)
    slice_copy      1.83x    1.01x
    fancy_1d        2.92x    2.13x   (was 7.8x / 10.9x)
    fancy_2d_rows   1.43x    1.08x   (was 1.7x / 2.8x)
    bool_mask       0.66x    0.42x   (was 3.8x / 3.5x)
    fancy_setitem   2.19x    1.40x   (was 7.7x / 10.5x)
    bool_setitem    0.84x    0.54x   (was 5.2x / 5.3x)
    take_axis0      2.31x    1.12x   (was 7.1x / 11.4x)

  What made the indexing rows move: the gather/scatter plan is now walked by
  a callback instead of materialising an offset vector; the single-index-array
  case folds the index array straight into the offset accumulator with a
  typed loop (no `Vec<i64>` of widened indices); `nonzero` has a contiguous
  1-D boolean fast path; scatter has a broadcast-scalar path; and `take`
  memcpys whole contiguous runs. The strided binary op now coalesces axes
  that form a single arithmetic progression (numpy's nditer does the same)
  and splits across rayon above 2^16 elements. min/max folds its NaN scan
  into the reduction loop instead of making a second pass.

  Known gaps carried forward:
  * No numpy scalar types yet (M3): `a[0, 0]` returns a Python number, so
    tests asserting `type(a[i]) is np.int64` fail. `a.base` chains are right,
    but subclass (`__array_finalize__`) plumbing is absent.
  * `test_multiarray.py`, `test_numeric.py` and `test_nep50_promotions.py`
    still fail to collect. The remaining blockers are, respectively, object
    *arrays* (`np.array([None], dtype=object)` at module scope),
    `datetime64`/`timedelta64` dtypes, and byte-swapped arrays
    (`np.ones(20000, dtype='>u4')`) — all three are storage-model work, not
    indexing work, and belong to their own milestones.
  * `add_i32` and `max_f64` at 1e6 sit above 1.0x; both are pre-M2 reduction/
    elementwise paths that are ALU- rather than bandwidth-bound. The host is
    noisy (repeat runs move any single row by +-50%), so these were confirmed
    with isolated repeat measurements before being recorded here.
  * Small-array (1e3) rows still carry 0.2-0.5us of PyO3 boundary cost per
    call, which is most of the ratio on `slice_view`.
