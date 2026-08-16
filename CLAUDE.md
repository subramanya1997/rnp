# rnp — Rust port of NumPy

A from-scratch Rust implementation of NumPy's core, validated against NumPy's own
unmodified test suite and benchmarked against real NumPy.

## Model policy (who does what)

- **Fable 5 (the main session model) is the planner and final reviewer.** Fable
  writes and maintains the detailed plan (`PLAN.md`), decomposes milestones,
  makes all architecture/design decisions, and does final correctness review of
  anything that ships. Do NOT delegate planning or final review to cheaper models.
- **Opus 4.8 subagents build and improve the code.** All substantive Rust
  implementation, PyO3 binding work, debugging, and improvement passes are done
  by subagents launched with `model: "opus"`. Opus is also used for design-level
  improvement passes on existing code.
- **Sonnet 5 subagents** may be used only for routine mechanical work (bulk
  search, boilerplate generation from a precise spec, running/collecting test
  results).
- Never use Haiku.

## Standing directives (from the user, 2026-08-16)

- **Keep going.** When a milestone lands and is verified, immediately plan and
  launch the next one. Do not stop to ask permission between milestones.
- **Performance is a first-class goal.** Match or beat real NumPy on every
  benchmark row. Use LTO + codegen-units=1, SIMD-friendly inner loops, and
  rayon parallelism above a size threshold (numpy is single-threaded for
  elementwise/reductions — beating it there is expected, not aspirational).
  Bit-exact crosscheck (0 divergences) is the gate before any optimization
  counts. Scalability (large arrays, multi-core) matters.
- **Validation is against NumPy's actual GitHub test suite** (`upstream/`,
  tag v2.5.2), run unmodified. Every milestone ends with a full-suite
  scoreboard run (`harness/run.py --all`), not just the targeted files.

## Hard rules

- **`upstream/` is read-only. Never edit anything under `upstream/`, especially
  test files. The tests must not change at all — they are the oracle.** The port
  is made to pass the tests; the tests are never made to pass the port.
- Cross-verification: behavior is checked against real NumPy 2.5.2 installed in
  `.venv` (the same version as the `upstream/` clone, tag v2.5.2).
- Benchmarks compare the Rust port against real NumPy (same venv), never against
  synthetic baselines.
- No `unsafe` in `rnp-core` without a `// SAFETY:` justification comment.

## Layout

- `upstream/` — shallow clone of numpy at tag v2.5.2. READ-ONLY oracle.
- `.venv/` — Python 3.13 venv with real `numpy==2.5.2`, pytest, hypothesis, maturin.
- `rnp/` — Rust workspace
  - `crates/rnp-core` — pure-Rust ndarray engine (dtypes, strides, broadcasting, ufuncs).
  - `crates/rnp-python` — PyO3 extension module exposing the NumPy-compatible Python API.
- `shim/` — Python package `rnp_numpy` presenting the `numpy` API surface backed by
  the Rust engine, plus the import redirection used by the harness.
- `harness/` — pytest conftest/runner that executes `upstream/numpy/**/tests` against
  the shim (via import redirection — tests unmodified) and writes a pass-rate
  scoreboard to `harness/scoreboard.json`.
- `benchmarks/` — paired benchmarks: each op timed on real numpy vs the Rust port.
- `PLAN.md` — Fable's living milestone plan. Update it as milestones land.

## Workflow

1. Fable updates `PLAN.md` with the next milestone, precisely scoped.
2. Opus subagent implements it in `rnp/` + `shim/`.
3. Harness runs the relevant upstream test files; scoreboard updated.
4. Divergences are cross-checked against `.venv` real numpy before deciding
   whether the port is wrong (it almost always is — the tests are the oracle).
5. Fable reviews, then next milestone.

## Commands

- Build the extension into the venv: `cd rnp && ../.venv/bin/maturin develop --release -m crates/rnp-python/Cargo.toml`
- Run harness: `.venv/bin/python harness/run.py [test_file ...]`
- Benchmarks: `.venv/bin/python benchmarks/run.py`
