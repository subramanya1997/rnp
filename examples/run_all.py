#!/usr/bin/env python3
"""Run every playbook example against NumPy and rnp, then compare results."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
EXAMPLE_FILES = sorted(EXAMPLES.glob("[0-9][0-9]_*.py"))
PAYLOAD_MARKER = "__RNP_EXAMPLE_PAYLOAD__="


def _json_value(value):
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, bytes):
        return {"__bytes__": list(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except ValueError:
            pass
    return value


def _encode_array(np, value):
    array = np.asanyarray(value)
    if np.ma.isMaskedArray(array):
        return {
            "kind": "masked",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": _json_value(array.data.tolist()),
            "mask": _json_value(np.ma.getmaskarray(array).tolist()),
        }
    return {
        "kind": "array",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": _json_value(array.tolist()),
    }


def _load_module(path):
    spec = importlib.util.spec_from_file_location(f"playbook_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(path):
    import numpy as np

    module = _load_module(path)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        module.main()
    result_dict = module.results()
    if not isinstance(result_dict, dict):
        raise TypeError(f"{path.name}.results() did not return a dict")
    payload = {
        "engine_module": np.__name__,
        "printed": output.getvalue(),
        "results": {key: _encode_array(np, value) for key, value in result_dict.items()},
        "tolerances": getattr(module, "TOLERANCES", {}),
    }
    print(PAYLOAD_MARKER + json.dumps(payload, sort_keys=True, allow_nan=False))


def _engine_environment(engine):
    env = os.environ.copy()
    if engine == "rnp":
        env["PYTHONPATH"] = os.pathsep.join([
            str(ROOT / "shim"),
            str(ROOT / "harness" / "_redirect"),
        ])
    else:
        env.pop("PYTHONPATH", None)
    return env


def run_one(path, engine):
    command = [str(VENV_PYTHON), str(Path(__file__).resolve()), "--worker", str(path)]
    completed = subprocess.run(
        command,
        cwd=EXAMPLES,
        env=_engine_environment(engine),
        capture_output=True,
        text=True,
        timeout=120,
    )
    marker_lines = [line for line in completed.stdout.splitlines() if line.startswith(PAYLOAD_MARKER)]
    if completed.returncode != 0 or not marker_lines:
        detail = (completed.stderr or completed.stdout).strip()
        return None, detail[-4000:]
    payload = json.loads(marker_lines[-1][len(PAYLOAD_MARKER):])
    expected_module = "rnp_numpy" if engine == "rnp" else "numpy"
    if payload["engine_module"] != expected_module:
        return None, (
            f"expected {expected_module!r}, but import numpy resolved to "
            f"{payload['engine_module']!r}"
        )
    return payload, None


def _restore_json(value):
    if isinstance(value, dict) and "__complex__" in value:
        return complex(*value["__complex__"])
    if isinstance(value, dict) and "__bytes__" in value:
        return bytes(value["__bytes__"])
    if isinstance(value, list):
        return [_restore_json(item) for item in value]
    return value


def _decode(np, encoded):
    data = _restore_json(encoded["data"])
    dtype = object if encoded["dtype"].startswith("StringDType(") else encoded["dtype"]
    array = np.array(data, dtype=dtype).reshape(encoded["shape"])
    if encoded["kind"] == "masked":
        mask = np.array(encoded["mask"], dtype=bool).reshape(encoded["shape"])
        return np.ma.array(array, mask=mask)
    return array


def compare_payloads(oracle, candidate):
    import numpy as np

    if set(oracle["results"]) != set(candidate["results"]):
        raise AssertionError("results() dictionaries have different keys")
    for key, oracle_encoded in oracle["results"].items():
        candidate_encoded = candidate["results"][key]
        if oracle_encoded["dtype"] != candidate_encoded["dtype"]:
            raise AssertionError(
                f"{key}: dtype {candidate_encoded['dtype']} != oracle {oracle_encoded['dtype']}"
            )
        expected = _decode(np, oracle_encoded)
        actual = _decode(np, candidate_encoded)
        tolerance = oracle["tolerances"].get(key)
        if tolerance is None:
            np.testing.assert_array_equal(actual, expected, err_msg=key)
        else:
            rtol, atol = tolerance
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=key)


def run_suite():
    rows = []
    diagnostics = []
    for path in EXAMPLE_FILES:
        oracle, oracle_error = run_one(path, "numpy")
        candidate, candidate_error = run_one(path, "rnp")
        oracle_status = "PASS" if oracle is not None else "FAIL"
        candidate_status = "PASS" if candidate is not None else "FAIL"
        match_status = "FAIL"
        if oracle is not None and candidate is not None:
            try:
                compare_payloads(oracle, candidate)
                match_status = "PASS"
            except Exception:
                diagnostics.append((path.name, "comparison", traceback.format_exc()))
        if oracle_error:
            diagnostics.append((path.name, "numpy", oracle_error))
        if candidate_error:
            diagnostics.append((path.name, "rnp", candidate_error))
        rows.append((path.name, oracle_status, candidate_status, match_status))

    name_width = max(len("EXAMPLE"), *(len(row[0]) for row in rows))
    print(f"{'EXAMPLE':<{name_width}}  NUMPY  RNP   MATCH")
    print(f"{'-' * name_width}  -----  ----  -----")
    for name, oracle_status, candidate_status, match_status in rows:
        print(f"{name:<{name_width}}  {oracle_status:<5}  {candidate_status:<4}  {match_status:<5}")

    if diagnostics:
        print("\nDiagnostics:")
        for name, stage, detail in diagnostics:
            print(f"\n[{name} / {stage}]\n{detail.rstrip()}")
    return 1 if diagnostics else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())
