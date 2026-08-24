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

## M5 FINAL (2026-08-24)

M5 closed with the zero-divergence gate fully restored on merged main
(commit 3d014ee): dev_check 36504/0, dev_check_struct 16922/0,
dev_check_matmul 3862/0, dev_check_nep50 12094/0, dev_check_object 2641/0,
dev_check_straggler 1697/0 — 73,720 bit-exact comparisons total; cargo
test --release 154 green. Landed via three fix lanes: datetime
(tolist/item object conversion, negative-year %04d repr, UFuncInputCasting
errors), message-parity (OverflowError/getfield messages, tobytes order
bug, complex std summation order, and bit-exact complex vecdot/vecmat by
calling Accelerate ILP64 dotc/gemm under numpy's exact blas_stride
predicate — new rnp-core/src/blas.rs), and structured dtypes
(multi-field setitem, subarray fields, VOID sort, field-by-position
astype, structured can_cast/promote_types/result_type +
DTypePromotionError). The last two lanes were implemented by Codex
(gpt-5.6-sol) during a sustained Anthropic API 529 outage, spec'd and
verified by Fable. Full-suite scoreboard rerun in progress.

## Revised ladder after M5 (2026-08-23)

The original M6 (sort/matmul) landed early — sort/argsort/searchsorted in M4,
the matmul family in M5. The scoreboard so far covers only `_core/tests`
(66 files); the full upstream tree also has lib (25 files / ~1375 test fns),
random (10/673), ma (8/526), polynomial (10/345), linalg (3/126), matrixlib
(7/89), fft (2/52), and ~12 top-level/testing files. f2py, typing and
_pyinstaller tests are out of scope (they test the Fortran toolchain and type
stubs, not the array engine).

- **M6 — finish the core**: nditer object (test_nditer 13.5%),
  `__array_function__` overrides (test_overrides 21.9%), test_numeric
  (17.9%), `__dlpack__` (0/95), scalar buffer protocol (0/79),
  casting-FP-error warnings (0/210), collection for test_ufunc/test_einsum/
  test_casting_unittests/test_deprecations, StringDType storage model
  (662 failures in test_strings). Harness extended to run the whole
  upstream suite (per-suite scoreboards) — measurement lane running.
- **M7 — lib + ma**: mostly pure-Python breadth over the engine.
- **M8 — linalg + fft + random**: LAPACK/FFT (faer or Accelerate, consistent
  with the BLAS-parity precedent), bit-exact generator streams, polynomial,
  matrixlib.

- 2026-08-23: whole-suite baseline measured (harness/run.py grew --suite/
  --full, commit c6d3f20; per-suite scoreboards in harness/scoreboard_*.json).
  Against the current build: lib 2259/4575 (49%), ma 3666/4352 (84%),
  polynomial 383/610 (63%), matrixlib 2/89, random 21/363 (6 of 10 files
  blocked: PCG64DXSM/SeedSequence missing), top 3/6, linalg 0 (no
  rnp_numpy.linalg._linalg), fft 0 (no rnp_numpy.fft at all). Combined with
  core: ~32,114 passing of ~41,304 currently collecting. Top collection
  blockers by files blocked: testing._private.utils.run_subprocess (4),
  random.PCG64DXSM (3), linalg._linalg (3), random.SeedSequence (2), fft (2),
  structured "unsupported field value" (2 — the in-flight struct lane's
  territory). REAL BUG found: upstream lib/test_regression.py collects then
  dies in a native abort inside _arraycompat.setitem — needs a crash triage
  lane before M7.

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

- 2026-08-16: M3 started (Opus). Scope: real numpy scalar types, the ufunc
  object model, ufunc breadth with numpy-exact special values and error
  flags, and `np.errstate`.
- 2026-08-16: M3 complete (Opus build). cargo test 86/86; dev_check
  **22089 comparisons / 119 divergences** (up from 11215/0 — the M3 sections
  are ~10900 new comparisons and every remaining divergence is listed below);
  crosscheck green. Full suite **8343/10493 (79.5%)**, up from the M2
  baseline of 1462/3565. The collected total grew because five large files
  (`test_scalarmath`, `test_umath`'s parametrised bulk, `test_regression`,
  `test_dtype`'s pickling block, `test_half`) only collect once scalars
  exist.

  Scoreboard deltas (passed/collected):
  * test_umath          81/801  -> **4881/5209** (93.7%)
  * test_scalarmath      0/0    -> **1463/1583** (92.4%)
  * test_dtype         711/940  -> **952/1160**  (82.1%)
  * test_regression      0/0    -> **221/419**
  * test_item_selection 204/254 -> 250/294
  * test_finfo           0/30   -> **30/30**
  * test_print           8/22   -> **22/22**
  * test_numerictypes   77/138  -> 96/138
  * test_scalar_methods 97/215  -> 115/227
  * test_longdouble      0/0    -> 11/33
  * test_half            0/39   -> 8/39
  * test_scalarprint     3/29   -> 10/29
  * test_errstate        0/6    -> 3/6
  * test_indexerrors     5/8    -> 7/8
  * regressed: test_scalar_ctors 142/197 -> 138/197 (see gaps)
  * still not collecting: test_multiarray, test_numeric,
    test_nep50_promotions, test_ufunc, test_datetime

  What landed:
  * `shim/rnp_numpy/_scalars.py` — the real scalar hierarchy, as *Python*
    classes so the MRO can be numpy's exactly. Probed and reproduced: only
    four concrete types inherit a builtin (`float64(floating, float)`,
    `complex128(complexfloating, complex)`, `bytes_(bytes, character)`,
    `str_(str, character)`) — `int64` does **not** inherit `int` in numpy
    2.x; `np.bool_.__name__` is `"bool"`; `True_`/`False_` are singletons;
    `longlong`/`ulonglong`/`longdouble`/`clongdouble` are distinct classes.
    Every scalar carries `.dtype`, `.itemsize`, `.shape`, `.ndim`, `.size`,
    `.real`/`.imag`, `.item()`, `x[()]`, `__array__`, `__index__`, hashes
    equal to the Python equivalent, and numpy's repr/str. The float
    formatting is numpy's own rule, derived by probing: the digits are the
    shortest decimal that round-trips through *that* float type, and the
    positional/scientific switch is made on the decimal exponent of the
    *stored* value (which is why `str(np.float32(1e-4))` is `1e-04`).
  * `shim/rnp_numpy/_ufunc.py` + `_ufunc_table.py` — one `ufunc` object per
    name, with `nin/nout/nargs/identity/ntypes/types` generated from real
    numpy so introspection matches even where the port has no loop. numpy's
    aliasing is reproduced (`np.acos is np.arccos`, `np.abs is np.absolute`,
    ...). `__call__(out=, where=, casting=, dtype=, order=)`, `.reduce`,
    `.accumulate`, `.reduceat`, `.outer`, `.at` all work.
  * `rnp-core/src/ufunc.rs` + an extended `ops::BinOp` — Rust inner loops for
    the whole M3 list: the transcendental family, `power`/`float_power`,
    `floor_divide`/`remainder`/`fmod`/`divmod`, `minimum`/`maximum`/`fmin`/
    `fmax`, `arctan2`/`hypot`/`copysign`/`nextafter`/`spacing`/`logaddexp`/
    `logaddexp2`/`heaviside`/`ldexp`, the bitwise and shift ops, `gcd`/`lcm`,
    the logical ops, `isnan`/`isinf`/`isfinite`/`signbit`, `frexp`/`modf`,
    `conjugate`, `real`/`imag`, and the sign/round family. Type resolution
    follows numpy's own loop tables: the float-only ufuncs pick the
    *smallest* loop that fits (`np.exp(np.uint8(1))` is float16), `rint` has
    no integer loop while `floor`/`ceil`/`trunc` do, `absolute` on complex
    drops to the real component type.
  * `rnp-core/src/fpe.rs` + `shim/rnp_numpy/_errstate.py` — the
    divide/over/under/invalid model. The Rust loops accumulate a 4-bit mask
    and the shim turns it into numpy's `RuntimeWarning` /
    `FloatingPointError` under the current `np.seterr` state, with numpy's
    exact wording (`"divide by zero encountered in log"`, and the
    `"... in scalar multiply"` form for scalar ops). Array operators report
    through the same path, so `np.errstate` works on `a / b` as well as on
    `np.divide(a, b)`.
  * NEP 50 completed for the scalar/array boundary: numpy scalars are strong
    operands, Python numbers weak, an out-of-range Python integer is an
    `OverflowError` in arithmetic but answers *correctly* in a comparison,
    and `uint64` vs signed comparisons go through an exact 128-bit loop
    rather than the float64 the two would otherwise promote to.
  * Object arrays: `np.array([...], dtype=object)` stores 8-byte handles into
    an append-only slab of `Py<PyAny>` on the Python side (handle 0 is
    `None`, so `np.empty(3, object)` reads back as numpy's does). Indexing,
    assignment, `tolist`, `item` and `repr` all work. The slab never frees,
    which is a bounded leak traded for the guarantee that no handle dangles.
  * datetime64 / timedelta64 as *descriptors*: `np.dtype('M8[ns]')`,
    `'m8[D]'`, `'datetime64[us]'`, the byte-order forms and all of
    `np.typecodes['All']` now construct, with numpy's `num`/`char`/`name`/
    `str`/`repr`. There is no datetime storage or arithmetic yet.
  * Half-precision conversion bugs found by `test_half`: `f16 -> f32` was
    off by a factor of two for subnormals (numpy's
    `npy_halfbits_to_floatbits` shifts once *before* the renormalising loop),
    and `f32/f64 -> f16` discarded NaN payloads instead of keeping the top
    ten mantissa bits. A new cargo test round-trips all 65536 half patterns
    through both float types.
  * Allocation guard: `np.zeros([975] * 7, np.int8)` used to overflow the
    element-count product, hand a wrapped length to the allocator and
    **abort the process**; it now raises numpy's ValueError. This was a
    latent M0 bug that only became visible once `test_regression` collected.

  ULP accuracy vs real numpy (max over 200k random values in-domain plus a
  dense edge-case list, per `harness/dev_check.py`; policy is <= 4 ULP):

    function                                       f64   f32
    exp exp2 expm1 log log2 log10 log1p              0     0
    sin cos                                          0     1
    tan arcsin arccos arctan                         0     0
    sinh cosh                                        0     0
    tanh                                             1     1
    arcsinh                                          1     1
    arccosh                                          1     1
    arctanh                                          2     1
    cbrt sqrt hypot arctan2 power float_power        0     0

  `arcsinh`/`arccosh`/`arctanh` needed numpy's own formulations: the library
  forms overflow for huge arguments and cancel catastrophically at the edges
  of their domains (they measured 2.5e7 and 9.8e13 ULP before).

  Performance (ratio port/numpy, lower is better). The FP-error flags are
  *not* computed per element: each loop OR-folds a one-compare watch
  predicate, and only if that fires does a second pass attribute the exact
  mask. Computing the flags inline cost 3-4x on `add_f64`. Underflow
  detection additionally has to treat every zero result as a candidate, so
  it is gated on `np.seterr(under=...)` being something other than the
  default 'ignore'.

  (The host is noisy: `mul_f64` and `max_f64` at 1e6 moved between 1.15x and
  2.15x across back-to-back runs, and numpy's own timings moved +-20%.)

    case                 1e3      1e6
    add_f64             2.82x    1.18x
    mul_f64             4.08x    2.15x
    add_i32             3.45x    1.39x
    sum_f64             0.78x    1.02x
    max_f64             0.97x    2.08x
    exp_f64             1.38x    0.59x
    exp_f32             1.26x    0.61x
    sin_f64             1.23x    0.47x
    log_f64             1.34x    0.32x
    power_f64           1.20x    0.27x
    sqrt_f64            2.24x    1.05x
    abs_f64             3.05x    0.77x
    negative_f64        2.68x    0.74x
    maximum_f64         3.86x    1.15x
    floor_divide_i32    2.33x    3.35x
    bitwise_and_i32     3.19x    3.80x
    add_reduce_f64      2.60x    0.63x
    scalar_add_f64     17.69x   12.20x
    scalar_add_i64      8.73x    6.32x
    scalar_extract     12.46x    5.18x

  Known gaps carried into M4:
  * **Scalar-op overhead**: a scalar binary op is ~2.4us against numpy's
    0.15us. Every one crosses the PyO3 boundary and builds two 0-d arrays;
    the fix is a dedicated scalar path in Rust that never allocates.
  * `bitwise_and_i32` and `floor_divide_i32` at 1e6 are 3.3-3.8x: the
    broadcast-scalar operand takes the stepped-pointer loop, which does not
    vectorise even after the scalar load was hoisted out of it. `maximum` is
    ~1.2x because the signed-zero tie-break costs several selects.
  * Every 1e3 row carries the same 0.5-0.8us of PyO3 + `ufunc.__call__`
    dispatch cost, which is most of the small-array ratios.
  * **119 dev_check divergences remain**, all enumerated by
    `harness/dev_check.py`: complex functions at infinite arguments (numpy
    follows C99 Annex G, `num_complex` does not) — about 30; last-bit
    ordering differences in `add.reduce`/`multiply.reduce` with `axis=None`
    or `where=` (the fold order is not numpy's pairwise tree) — about 20;
    a handful of 1-ULP float32 transcendental bit differences (inside the
    ULP policy but flagged by the bit-exact table); `lcm` at the signed
    boundary; `gcd`/`lcm` raising `TypeError` where numpy raises
    `UFuncTypeError`; and `np.record` missing.
  * **`test_scalar_ctors` lost 4 tests**, and honestly so: `longlong` and
    `int64` used to be the *same* class, which made
    `np.array(np.longlong(2)).dtype.type is np.longlong` trivially true.
    They are now distinct, and `NdArray` carries a `DType` rather than a
    `Descr`, so an array loses the C-type alias. Threading `Descr` through
    the array header is M4 work.
  * **Byte-swapped arrays** still raise, so `test_nep50_promotions` does not
    collect. **Object-array ufuncs** are not implemented (the storage is),
    so `test_multiarray` still does not collect. **datetime storage** does
    not exist, so `test_numeric` does not collect. All three are storage-model
    milestones.
  * `matmul`/`vecdot`/`matvec`/`vecmat`/`isnat` are the five ufuncs with no
    loop at all (M6 for the matmul family).
  * Underflow is only detected under a non-default `errstate(under=...)`,
    and float16 `str`/`repr` still differs from numpy on a few values.

- 2026-08-16: M4 started (Opus). Scope, in the priority order set by Fable:
  (1) kill all 119 dev_check divergences; (2) storage-model completion
  (`Descr` in the array header, byte-swapped arrays, datetime64/timedelta64,
  object-dtype ufuncs, structured field access); (3) a collection sweep over
  the files that still fail to import; (4) the scalar fast path and the
  remaining benchmark debt.

- 2026-08-16: M4 correctness gate met. **`harness/dev_check.py`: 22096
  comparisons / 0 divergences** (was 22089 / 119). `cargo test --release`
  97 passing (was 86). No check in `dev_check.py` was weakened or skipped at
  any point; the count went to zero by fixing the port.

  How the 119 fell, by cluster:

  * **~45 complex-at-infinity/NaN cases.** The premise that numpy implements
    the C99 Annex G tables in `npy_math_complex.c.src` turned out to be false
    on this platform: every one of those bodies sits behind `#ifndef
    HAVE_CSIN@C@`, and `numpy/_core/meson.build` defines the whole `HAVE_*`
    C99 complex family whenever libc provides it -- which macOS does. **numpy's
    complex ufuncs are the system libm's `csin`/`cacos`/`csqrt`/...**, which was
    confirmed by compiling a C probe whose output matches numpy bit for bit.
    The port now calls the same libm entry points through `extern "C"`
    (`Complex<T>` is `#[repr(C)]`, the same parameter class as
    `double _Complex` under AAPCS64 and x86-64 SysV), with numpy's own `nc_*`
    wrappers from `umath/funcs.inc.src` transcribed around them:
    `exp2 = cexp(z*ln2)`, `log10 = clog(z)*log10(e)` -- a *multiply*, which is
    what fixed the 1-ULP `log10`/`log2` rows -- plus `expm1`/`log1p` built from
    the real routines and `CDOUBLE_sign`/`_reciprocal` from `loops.c.src`.
    `complex64` was silently broken along the way (23 ufuncs diverging, hidden
    because `UFUNC_DTYPES` carries no `complex64`); it now uses the
    single-precision libm entries as numpy does.
  * That forced the error-flag design. Conditions like
    `arctanh(1.8e308+0j)` -> divide-by-zero *and* invalid with a finite result
    are artifacts of the libm's internals; no value-level rule reproduces them.
    Since the port now calls the same libm, it reads the same source numpy
    reads: `fpe::hw_clear`/`hw_take` mirror `npy_get_floatstatus_barrier`,
    scoped to the complex loops only, cleared and folded per rayon chunk. The
    consequence, recorded deliberately: rnp's complex warnings are now as
    platform-dependent as numpy's own are.
  * **~15 real-dtype flag mismatches.** Probing `npy_divmod` showed only two
    of its divisions can signal and that `npy_remainder` discards the
    quotient -- which is why `remainder(1.0, 0.0)` is silent while
    `divmod(1.0, 0.0)` is divide-by-zero, and why float16 `remainder` reports
    nothing at all (half computes in `float`, so its quotient never
    overflows). `logaddexp`'s invalid comes from C's signalling `>`/`<=` on a
    NaN. `INT_MIN // -1` overflow was added for `floor_divide`/`divmod`.
  * **4 float16 min/max.** The half loops carry no signed-zero fixup, so the
    *first* operand wins ties -- the opposite of float32/float64, which were
    already right and are unchanged.
  * **2 `lcm` signed-boundary.** numpy takes both magnitudes in the *unsigned*
    type (`(utype)0 - (utype)a`), runs the unsigned algorithm and casts back.
  * **2 float32 `sin`/`cos`.** Transcribed the Cody-Waite reduction plus
    polynomial from `loops_trigonometric.dispatch.cpp` scalar-wise (hex
    constants via `f32::from_bits`, every `MulAdd` a real FMA). The result is
    **bit-exact, 0 ULP** -- not the "within 1 ULP" the plan allowed for -- and
    the dev_check ULP table now reads `sin float32 0`, `cos float32 0`.
  * **23 reduction-ordering rows.** Three distinct model errors, each pinned
    by bit-level experiment against numpy before being implemented.
    `axis=None` was a *chain* of per-axis reductions; numpy's iterator
    coalesces a contiguous operand into one run and does a single pairwise
    tree, bit-identical to `reduce(a.reshape(-1))`. `where=` is neither
    compress-then-reduce nor identity-substitution: numpy's
    `generic_masked_strided_loop` hands the unmasked inner loop each maximal
    contiguous run of set mask values, so the pairwise tree is rebuilt per run
    and the run totals fold in sequentially. `reduceat` copies `a[start]` into
    the output and loops over the remaining `count-1`, so the fold is
    `a[start] + pairwise(a[start+1:stop])` -- and that is keyed on `add`,
    the only loop with `PW = 1`; a naive "always split" broke
    `multiply.reduceat`, which is what caught it. Found in passing:
    `initial=` *seeds the accumulator* before the fold rather than combining
    with the finished result, which differs for an outer-axis sum.
  * **14 type-resolution rows.** `numpy._core._exceptions` is now ported
    faithfully (correct MRO, `_display_as_base` renaming, exact `__str__`) and
    raised from Rust through a `_set_error_factories` hook, so
    `pytest.raises(np._core._exceptions._UFuncNoLoopError)` works.
    `gcd`/`lcm`/`positive`/`sign` reject their common dtype exactly where
    numpy's `SimpleUniformOperationTypeResolver` does, including refusing bool
    rather than lifting it to int8; `negative` on bool raises a plain
    `TypeError` because `PyUFunc_NegativeTypeResolver` errors *before* loop
    lookup -- the asymmetry is real. Done in Rust, so `+arr`/`-arr` are right
    too. `ldexp` now accepts bool, and the NEP-50 weak-scalar bug
    (`np.ldexp(np.float64(0.5), 2)` wrongly raising) is fixed with a per-slot
    resolution table.
  * **`np.record`** is a real `void` subclass with numpy's
    `(record, void, flexible, generic, object)` MRO.

- 2026-08-16: M4 complete (Opus build). Full suite **10769/14105 (76.3%)**,
  up from the M3 baseline of 8344/10493. The *percentage* fell because the
  denominator grew by 3612: `test_multiarray`, `test_strings`,
  `test_nep50_promotions`, `test_arrayprint`, `test_nditer` and
  `test__exceptions` only collect once this milestone's storage-model work
  exists, and they arrive with their failures attached. Passed is the number
  that matters and it is up 2425. **No adopted file regressed** (verified
  file-by-file against the committed M3 scoreboard).

  Gates: `cargo test --release` **100 unit + 1 integration**;
  `harness/dev_check.py` **22814 comparisons / 0 divergences**;
  `harness/crosscheck.py` green.

  Scoreboard deltas (passed/collected):
  * test_strings          0/0     -> **1346/2066** (was a collection error)
  * test_shape_base       20/213  -> **182/213**
  * test_nditer           0/0     -> **122/926**
  * test_umath            4881/5235 -> **4991/5315**
  * test_arrayprint       0/0     -> **103/134**
  * test_defchararray     0/100   -> **100/100**
  * test_scalar_methods   115/227 -> **211/233**
  * test_api              2/57    -> **53/59**
  * test_scalar_ctors     138/201 -> **177/201** (the M3 regression is repaid
                                     nearly 10x over; the gate was >=142)
  * test_regression       221/422 -> **257/422**
  * test_hashtable        0/36    -> **36/36**
  * test_unicode          45/76   -> **75/76**
  * test_nep50_promotions 0/0     -> **26/443**
  * test_longdouble       11/34   -> **33/34**
  * test_memmap           0/24    -> **21/24**
  * test_half             8/39    -> **29/39**
  * test_mem_overlap      4/25    -> **24/25**
  * test_scalarprint      10/30   -> **29/30**
  * test_item_selection   250/294 -> **266/294**
  * test_extint128        0/13    -> **13/13**
  * test__exceptions      0/0     -> **10/11**
  * test_indexing         60/106  -> **65/106**
  * test_argparse 2->7/7, test_abc 0->5/5, test_errstate 3->6/6,
    test_scalarinherit 4->6/6, test_protocols 0->2/2,
    test_scalarmath 1463->1466, test_dtype 952->953

  Storage model:
  * **`Descr` now lives in the `NdArray` header** (it was a bare `DType`), so
    arrays carry byte order and the C-type aliases. `np.array(np.longlong(2))
    .dtype.type is np.longlong` is True again -- the M3 regression that cost
    `test_scalar_ctors` 4 tests.
  * **Byte-swapped arrays** work end to end: storage, casting, ufuncs,
    reductions, printing, indexing. The design normalises **at the operand
    boundary, not in the inner loop**: every compute entry point starts with
    one `if !a.is_native()`, and when it fires a cold `#[inline(never)]`
    trampoline swaps the whole operand into a native temporary and calls the
    *same* kernel. That keeps the ~400 monomorphised cast/ufunc loops from
    doubling and leaves the native path paying one predictable byte compare
    outside every loop. The first attempt put the fallback *inside* the hot
    functions, making them recursive -- `astype` at 1e6 went 72us -> 910us and
    `exp`/`log`/`sin`/`power` regressed 2-4x. Splitting each into a thin gate
    plus a cold trampoline plus a non-recursive `*_native` kernel restored it.
  * `ndarray.byteswap(inplace=)`, `newbyteorder`, `__array_interface__` (v3,
    `strides: None` iff C-contiguous), and an assignable `flags.writeable`
    (`PyFlags` is now a live proxy holding the array, not a snapshot) with
    numpy's refusal to re-enable WRITEABLE on a non-owning array whose base is
    read-only.
  * New `sort.rs`: `sort`/`argsort`/`searchsorted` with numpy's total order
    (NaN last and NaN-ties-equal, complex lexicographic on `(re, im)`,
    flexible dtypes by logical code points).
  * `dev_check.py` gained a `check_byteorder` section (+718 comparisons):
    creation/copy/slice/fancy/reshape/byteswap/add/astype/sum/item/repr/
    setitem over `>`/`<` x 10 numeric codes, `>U3`/`<U3`, `dtype.newbyteorder`
    and swapped `view`. Getting it to zero exposed a wrong PEP-3118 format
    string; `Descr::buffer_format` now emits `>i`, `3w`, `3s`, `4x` as numpy
    does.

  Breadth landed in the shim (pure Python, no engine changes):
  * `np.strings` + `defchararray` + `np.char` + `chararray`, reproducing
    numpy's result-itemsize formulas; this is what took `test_strings` from a
    collection error to 1346 and `test_defchararray` to 100/100.
  * `_core/arrayprint.py` -- upstream's file with 11 diff hunks, wired as the
    *actual* `ndarray.__repr__`/`__str__` (`_rnp.ndarray` is a heap type, so
    the assignment is possible), plus upstream's `printoptions` ContextVar.
    673 randomized repr/str cases cross-checked against real numpy: 0
    divergences.
  * `format_float_positional`/`format_float_scientific` (403k-case
    differential run vs numpy: 0 divergences). Found and fixed a real bug: the
    shortest-round-trip search rounded only one way, which is wrong for floats
    on a power of two, whose rounding interval is asymmetric.
  * `memmap`, `fromstring`/`fromfile`/`loadtxt`/`genfromtxt`/`savetxt`,
    `copyto`, `shares_memory`/`may_share_memory` (a real port of
    `mem_overlap.c`'s Diophantine solver, 106 cases cross-checked),
    `np.block` and the private `_block_*` helpers, `np.record`,
    `_core/_exceptions.py`, and the `_multiarray_tests` extint128 and
    identity-hashtable helpers.
  * `errstate` now keeps its state in a `contextvars.ContextVar` as numpy 2.x
    does, which is what makes it asyncio-safe, and refuses a second `__enter__`.
  * The numeric tower is registered with the `numbers` ABCs.

  Performance (ratio port/numpy, lower is better; **the host carried an
  external load average of 25-35 throughout, so every row was measured over
  repeated interleaved runs and these are minima**):

    scalar_add_f64      21.65x -> **3.28x**     scalar_add_i64  9.97x -> 1.64x
    scalar_extract      12.29x -> **2.00x**     bitwise_and_i32 2.64x -> 0.69x
    floor_divide_i32     2.19x -> **1.35x**     mul_f64         1.54x -> 1.08x
    maximum_f64          1.12x -> 1.44x(noise)  abs_f64         0.67x -> 0.66x
    add_i32              0.88x -> 0.76x         sum_f64                  0.56x
    copy 1.00x  astype 1.01x  add_reduce_f64 0.85x  exp_f64 0.32x
    sin_f64 0.36x  power_f64 0.31x  bool_mask 0.39x

  * The **scalar fast path** is the headline: `rnp_core::ops::binary_scalar`
    computes `scalar op scalar` over `Scalar` values with zero allocation --
    no 0-d `NdArray`, no `Arc`, no shape/stride vectors -- calling the *same*
    element kernels and flag-attribution helpers the array loops use. The
    bridge takes an int opcode (no ufunc name lookup), classifies operands by
    type-object pointer, and returns `(dtype_code, value, flags)` with no
    `dtype` object constructed. NEP 50 weak/strong promotion, the exact-128-bit
    `uint64`-vs-signed comparison, and the `NotImplemented`-vs-`TypeError`
    distinction are all preserved. A 700k-triple integration test
    (`tests/scalar_path.rs`) pins result dtype, result *bits*, FP flags and
    error against the array driver.
  * The broadcast-scalar binary loop formed a byte pointer with a runtime
    step, hiding contiguity from LLVM; it now forms real slices when the step
    equals the itemsize. That is what moved `bitwise_and_i32` and
    `floor_divide_i32`, and it applies to the whole binary-op family.
  * `multiply`/`divide` had lost vectorisation entirely because `watch_scale`
    read a relaxed **atomic** load inside the loop body, which LLVM will not
    hoist. Now read once before the loop.

  Known gaps carried into M5:
  * **`test_multiarray.py` collects (14272 tests, verified by `--co -q`) but
    exceeds the harness's 900s per-file timeout**, so it contributes 0 to the
    scoreboard above and **its pass count is UNMEASURED**. An out-of-harness
    run with a 90-minute budget was started but was killed before producing
    any output, so there is no number for this file -- do not assume one. It
    is the largest single unknown in the scoreboard, and M5 should first raise
    the harness timeout or shard the file, then measure it.
  * **datetime64/timedelta64 storage and arithmetic were not attempted.**
    `test_datetime.py` and `test_numeric.py` still fail to collect on a
    `dispatch_dtype: TimeDelta(n) is not a numeric dtype` panic. The unit
    promotion table was probed and is saved for M5.
  * **Object-dtype ufuncs** were not attempted; this is 384 of the 417
    `test_nep50_promotions` failures (a single `test_integer_comparison`
    parametrisation) and blocks `test_records`.
  * **Structured-array field access** (`a['f0']` returning a view) does not
    exist, so `test_records` stays at 0/44 and structured reprs are wrong.
  * `test_strings`' remaining 681 failures are 662 `StringDType` (a
    variable-width heap-arena storage model, genuinely a new dtype), 12
    byte-swapped `>U`, 6 dtype-size validation, 1 `astype` from void.
  * `test_casting_floatingpoint_errors` (0/210) needs casts to raise FP
    warnings; `test_dlpack` (0/95) needs `__dlpack__`; `test_nditer` is at
    122/926; `test_ufunc` still does not collect.
  * `floor_divide_i32` at 1.35x is the one benchmark row left above parity:
    numpy specialises a loop-invariant integer divisor with libdivide. With an
    *array* divisor the ratio inverts to 0.41x. Closing it needs
    Granlund-Montgomery magic numbers, which was judged too subtle to land
    safely in this pass.
  * `memmap`'s last 3 failures need real ndarray subclassing: `_rnp.ndarray`
    has no `tp_new` and `.view()` ignores its type argument.

- 2026-08-23: M5 build ran (Opus lanes: pylib, struct, datetime, objloops,
  matmul, buf-fix, collect, nep50, straggler). All branches merged into main,
  but the orchestrating session was cut off (two lanes died on API 529s and
  their residue was committed as unverified WIP snapshots) before the
  post-merge rebuild + verification ever ran. Landed per git log: matmul/dot
  family (numpy's dotc transcribed), datetime64/timedelta64 storage +
  arithmetic + repr, structured field access, object-dtype ufunc loops,
  buffer/__array__ adoption fixes, NEP 50 C-conversion bounds (unfinished),
  harness sharding for oversized files (test_multiarray now measurable),
  lexsort/tobytes/clip/std/var/pickle/frombuffer/getfield breadth,
  numpy._core._multiarray_umath, linalg skeleton (WIP residue).

- 2026-08-23: post-merge verification (Fable). Rebuilt in the main venv
  (the installed extension predated the last two merges). cargo test
  --release: 150/150 green. dev_check gates are NOT met on the merged tree:
  dev_check 36504/105 (was 0 pre-M5; ~104 datetime: tolist not returning
  datetime/timedelta objects or None for NaT, negative-year repr '-0001' vs
  numpy's '-001', TypeError instead of UFuncTypeError on mismatched-unit
  M8±m8[Y]; plus 1 vecmat complex-NaN), dev_check_nep50 12094/39
  (OverflowError message parity — the crashed lane's unfinished work),
  dev_check_object 2641/4 (dtype('O') vs dtype('object') in messages),
  dev_check_straggler 1697/3 (getfield message; a REAL tobytes order='F'
  value bug; std complex128 last-ULP), dev_check_struct 16922/339
  (multi-field setitem, struct→struct astype, subarray-field construction,
  VOID sort/argsort ordering). dev_check_buffer 27405/0 and dev_check_matmul
  3862/0 are green. Three Opus fix lanes launched in fresh worktrees:
  m5/dt-fix2 (datetime cluster), m5/struct-fix (struct cluster), m5/msgs
  (nep50 messages + object text + straggler + vecmat). Full-suite scoreboard
  baseline running in parallel.
