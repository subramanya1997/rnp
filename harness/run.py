#!/usr/bin/env python3
"""Run upstream numpy test files against the Rust port (rnp_numpy shim).

Usage:
    .venv/bin/python harness/run.py                    # default target set
    .venv/bin/python harness/run.py test_indexing.py   # specific file(s)
    .venv/bin/python harness/run.py --all              # every _core test file
    .venv/bin/python harness/run.py --suite lib        # every file in a suite
    .venv/bin/python harness/run.py --full             # every suite

Tests under upstream/ are NEVER modified. We run pytest in a subprocess whose
PYTHONPATH injects (a) the rnp_numpy shim and (b) a sitecustomize that
redirects `import numpy` to the shim. Results land in harness/scoreboard.json
(the "core" suite / --all / default-target-set runs); other suites land in
harness/scoreboard_<suite>.json.

Very large files (test_multiarray.py collects ~14k tests) do not finish inside
a sane per-file timeout, which used to make them score 0. They are now *sharded*
across several subprocesses by `harness/_shard/rnp_shard.py` and the shards'
results are summed. Sharding is scheduling only: the union of the shards is the
file's whole collection.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_TESTS = ROOT / "upstream" / "numpy" / "_core" / "tests"
VENV_PY = ROOT / ".venv" / "bin" / "python"
SCOREBOARD = ROOT / "harness" / "scoreboard.json"
UPSTREAM_NUMPY = ROOT / "upstream" / "numpy"

# Suite name -> list of upstream test directories that make it up. "core" is
# kept identical to the historical UPSTREAM_TESTS default (single dir); "top"
# covers the two small top-level test dirs that don't have their own suite.
SUITE_DIRS = {
    "core": [UPSTREAM_NUMPY / "_core" / "tests"],
    "lib": [UPSTREAM_NUMPY / "lib" / "tests"],
    "ma": [UPSTREAM_NUMPY / "ma" / "tests"],
    "linalg": [UPSTREAM_NUMPY / "linalg" / "tests"],
    "fft": [UPSTREAM_NUMPY / "fft" / "tests"],
    "random": [UPSTREAM_NUMPY / "random" / "tests"],
    "polynomial": [UPSTREAM_NUMPY / "polynomial" / "tests"],
    "matrixlib": [UPSTREAM_NUMPY / "matrixlib" / "tests"],
    "top": [UPSTREAM_NUMPY / "tests", UPSTREAM_NUMPY / "testing" / "tests"],
}
# Order used for --full and for the "run everything" report.
SUITE_ORDER = ["core", "lib", "ma", "linalg", "fft", "random", "polynomial",
               "matrixlib", "top"]

# Files targeted by current/adopted milestones (see PLAN.md).
DEFAULT_TARGETS = [
    "test_dtype.py",
    "test_numerictypes.py",
    "test_indexing.py",
    "test_shape_base.py",
    "test_umath.py",
    "test_scalarmath.py",
    "test_multiarray.py",
]

# Per-file shard counts for files too large to finish in one subprocess.
# Kept deliberately minimal: only files that actually exceed the timeout are
# sharded, so a scoreboard comparison against an older unsharded run is not
# confounded by the split. (Sharding was measured to be score-neutral --
# test_indexing.py scores 65/106 both ways -- but the whole point of the
# regression gate is not to have to take that on trust.)
SHARDS = {
    "test_multiarray.py": 16,
}

TIMEOUT = 900


def display_name(test_file: Path) -> str:
    # Disambiguate same-named files across suites (e.g. test_regression.py
    # exists in _core, lib, and ma).
    sub = test_file.parent.parent.name
    return test_file.name if sub == "_core" else f"{sub}/{test_file.name}"


def _blank(name: str) -> dict:
    return {"file": name, "passed": 0, "failed": 0, "errors": 0,
            "skipped": 0, "total": 0}


def run_shard(test_file: Path, shard: int, nshards: int,
              timeout: int = TIMEOUT) -> dict:
    """Run one subprocess: either the whole file (nshards == 1) or one shard."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "shim"), str(ROOT / "harness" / "_redirect"),
         str(ROOT / "harness" / "_shard")]
    )
    tag = test_file.stem if nshards == 1 else f"{test_file.stem}-{shard}"
    report = ROOT / "harness" / f".report-{tag}.json"
    cmd = [
        str(VENV_PY), "-m", "pytest", str(test_file),
        "-q", "-p", "no:cacheprovider", "--continue-on-collection-errors",
        "--json-report", f"--json-report-file={report}",
        "-c", str(ROOT / "harness" / "pytest.ini"),
        "--rootdir", str(test_file.parent),
        # importlib mode keeps the test module out of the redirected `numpy.*`
        # namespace (otherwise pytest would try to import the test file itself
        # as `numpy._core.tests.<name>`, which the shim does not provide).
        "--import-mode=importlib",
        # upstream/numpy/conftest.py imports C extension modules the port has
        # no equivalent for; cut conftest discovery off below it.
        f"--confcutdir={test_file.parent}",
    ]
    if nshards > 1:
        cmd += ["-p", "rnp_shard", f"--rnp-shard={shard}/{nshards}"]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout, cwd=ROOT / "harness")
    except subprocess.TimeoutExpired:
        r = _blank(display_name(test_file))
        r["error"] = f"timeout (shard {shard}/{nshards})"
        return r
    summary = _blank(display_name(test_file))
    if report.exists():
        data = json.loads(report.read_text())
        s = data.get("summary", {})
        summary.update(
            passed=s.get("passed", 0),
            failed=s.get("failed", 0),
            errors=s.get("error", 0),
            skipped=s.get("skipped", 0) + s.get("xfailed", 0)
            + s.get("xpassed", 0),
            total=s.get("collected", s.get("total", 0)),
        )
        report.unlink()
    else:
        # Catastrophic failure (collection crash before json report).
        summary["error"] = (proc.stderr or proc.stdout)[-2000:]
    return summary


def merge(name: str, parts: list) -> dict:
    out = _blank(name)
    errs = []
    for p in parts:
        for k in ("passed", "failed", "errors", "skipped", "total"):
            out[k] += p[k]
        if "error" in p:
            errs.append(p["error"])
    if errs:
        out["error"] = " | ".join(errs)[-4000:]
    if len(parts) > 1:
        out["shards"] = len(parts)
    return out


def plan_units(files, shard_override=None):
    """Expand files into (file, shard, nshards) units of work."""
    units = []
    for f in files:
        n = shard_override or SHARDS.get(f.name, 1)
        for i in range(n):
            units.append((f, i, n))
    return units


def suite_files(suite: str) -> list:
    """Every test_*.py file across a suite's upstream test dir(s), sorted."""
    files = []
    for d in SUITE_DIRS[suite]:
        if d.is_dir():
            files.extend(sorted(d.glob("test_*.py")))
    return files


def scoreboard_path_for(suite: str) -> Path:
    # "core" (== the historical --all target set) keeps the original,
    # un-suffixed scoreboard.json so existing workflows/tooling don't break.
    if suite == "core":
        return SCOREBOARD
    return ROOT / "harness" / f"scoreboard_{suite}.json"


def run_files(files: list, jobs: int, shards: int, timeout: int,
              out: Path, quiet: bool = False) -> dict:
    """Run `files` through the shard/timeout machinery and write a scoreboard.

    Returns the board dict (also written to `out`).
    """
    files = [f for f in files if f.exists() or print(f"!! missing: {f}",
                                                     file=sys.stderr)]
    units = plan_units(files, shards)

    parts = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(run_shard, f, i, n, timeout): (f, i, n)
                for (f, i, n) in units}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            f, i, n = futs[fut]
            parts.setdefault(f, []).append(fut.result())
            done += 1
            print(f"  [{done}/{len(units)}] {display_name(f)}"
                  + (f" shard {i}/{n}" if n > 1 else ""), file=sys.stderr)

    results, tot_pass, tot_all = [], 0, 0
    for f in files:
        r = merge(display_name(f), parts.get(f, []))
        results.append(r)
        tot_pass += r["passed"]
        tot_all += max(r["total"] - r["skipped"], 0)
        if not quiet:
            denom = max(r["total"] - r["skipped"], 0)
            pct = 100 * r["passed"] / max(denom, 1)
            print(f"{r['file']:38s} {r['passed']:5d}/{denom:5d}  ({pct:5.1f}%)"
                  + ("  [CRASH]" if "error" in r else ""))

    board = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "overall_passed": tot_pass,
        "overall_total": tot_all,
        "files": results,
    }
    out.write_text(json.dumps(board, indent=2))
    if not quiet:
        print(f"\nOVERALL: {tot_pass}/{tot_all} "
              f"({100 * tot_pass / max(tot_all, 1):.1f}%)  -> {out}")
    return board


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="upstream test file names")
    ap.add_argument("--all", action="store_true",
                    help="run every file in upstream/numpy/_core/tests")
    ap.add_argument("--suite", choices=sorted(SUITE_DIRS),
                    help="run every file in an upstream suite's tests dir "
                         "(core, lib, ma, linalg, fft, random, polynomial, "
                         "matrixlib, top); writes harness/scoreboard_<suite>"
                         ".json (core writes harness/scoreboard.json)")
    ap.add_argument("--full", action="store_true",
                    help="run every suite (core, lib, ma, linalg, fft, "
                         "random, polynomial, matrixlib, top), each writing "
                         "its own harness/scoreboard[_<suite>].json")
    ap.add_argument("--jobs", "-j", type=int, default=4,
                    help="parallel pytest subprocesses (default 4)")
    ap.add_argument("--shards", type=int, default=None,
                    help="override the per-file shard count for every file")
    ap.add_argument("--timeout", type=int, default=TIMEOUT,
                    help="per-subprocess timeout in seconds (default 900)")
    ap.add_argument("--out", default=None, help="scoreboard path override")
    args = ap.parse_args()

    if args.full:
        grand_pass, grand_all = 0, 0
        for suite in SUITE_ORDER:
            files = suite_files(suite)
            out = Path(args.out) if args.out else scoreboard_path_for(suite)
            print(f"\n=== suite: {suite} ({len(files)} files) ===",
                  file=sys.stderr)
            board = run_files(files, args.jobs, args.shards, args.timeout, out)
            grand_pass += board["overall_passed"]
            grand_all += board["overall_total"]
        print(f"\nGRAND TOTAL: {grand_pass}/{grand_all} "
              f"({100 * grand_pass / max(grand_all, 1):.1f}%)")
        return 0

    if args.suite:
        files = suite_files(args.suite)
        out = Path(args.out) if args.out else scoreboard_path_for(args.suite)
        run_files(files, args.jobs, args.shards, args.timeout, out)
        return 0

    # --all / explicit files / default target set: identical to historical
    # behavior, always against upstream/numpy/_core/tests, always writing
    # harness/scoreboard.json unless --out overrides it.
    if args.all:
        files = sorted(UPSTREAM_TESTS.glob("test_*.py"))
    elif args.files:
        files = [UPSTREAM_TESTS / f for f in args.files]
    else:
        files = [UPSTREAM_TESTS / f for f in DEFAULT_TARGETS]

    out = Path(args.out) if args.out else SCOREBOARD
    run_files(files, args.jobs, args.shards, args.timeout, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
