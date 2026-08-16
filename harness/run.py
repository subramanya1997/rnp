#!/usr/bin/env python3
"""Run upstream numpy test files against the Rust port (rnp_numpy shim).

Usage:
    .venv/bin/python harness/run.py                    # default target set
    .venv/bin/python harness/run.py test_indexing.py   # specific file(s)
    .venv/bin/python harness/run.py --all              # every _core test file

Tests under upstream/ are NEVER modified. We run pytest in a subprocess whose
PYTHONPATH injects (a) the rnp_numpy shim and (b) a sitecustomize that
redirects `import numpy` to the shim. Results land in harness/scoreboard.json.
"""
import argparse
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


def run_file(test_file: Path, timeout: int = 900) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "shim"), str(ROOT / "harness" / "_redirect")]
    )
    report = ROOT / "harness" / f".report-{test_file.stem}.json"
    cmd = [
        str(VENV_PY), "-m", "pytest", str(test_file),
        "-q", "-p", "no:cacheprovider", "--continue-on-collection-errors",
        "--json-report", f"--json-report-file={report}",
        "--rootdir", str(ROOT / "harness"),
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=timeout, cwd=ROOT / "harness")
    except subprocess.TimeoutExpired:
        return {"file": test_file.name, "error": "timeout", "passed": 0,
                "failed": 0, "errors": 0, "skipped": 0, "total": 0}
    summary = {"file": test_file.name, "passed": 0, "failed": 0, "errors": 0,
               "skipped": 0, "total": 0}
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="upstream test file names")
    ap.add_argument("--all", action="store_true",
                    help="run every file in upstream/numpy/_core/tests")
    args = ap.parse_args()

    if args.all:
        files = sorted(UPSTREAM_TESTS.glob("test_*.py"))
    elif args.files:
        files = [UPSTREAM_TESTS / f for f in args.files]
    else:
        files = [UPSTREAM_TESTS / f for f in DEFAULT_TARGETS]

    results, tot_pass, tot_all = [], 0, 0
    for f in files:
        if not f.exists():
            print(f"!! missing: {f}", file=sys.stderr)
            continue
        r = run_file(f)
        results.append(r)
        tot_pass += r["passed"]
        tot_all += max(r["total"] - r["skipped"], 0)
        pct = 100 * r["passed"] / max(r["total"] - r["skipped"], 1)
        print(f"{r['file']:38s} {r['passed']:5d}/{max(r['total']-r['skipped'],0):5d}  ({pct:5.1f}%)"
              + ("  [CRASH]" if "error" in r else ""))

    board = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "overall_passed": tot_pass,
        "overall_total": tot_all,
        "files": results,
    }
    SCOREBOARD.write_text(json.dumps(board, indent=2))
    print(f"\nOVERALL: {tot_pass}/{tot_all} "
          f"({100 * tot_pass / max(tot_all, 1):.1f}%)  -> {SCOREBOARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
