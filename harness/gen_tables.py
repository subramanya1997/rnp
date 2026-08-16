#!/usr/bin/env python3
"""Regenerate the dtype fact tables that rnp-core's unit tests assert against.

Everything written here is probed from the real numpy 2.5.2 in `.venv`; no
entry is written by hand. Run:

    .venv/bin/python harness/gen_tables.py
"""
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "rnp" / "crates" / "rnp-core" / "src"

# numpy name -> rnp-core `DType` variant.
DTYPES = {
    "bool": "Bool",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "uint8": "U8",
    "uint16": "U16",
    "uint32": "U32",
    "uint64": "U64",
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
    "complex64": "C64",
    "complex128": "C128",
}


def gen_promotion() -> str:
    lines = [
        "// Generated from real numpy 2.5.2 via numpy.promote_types. "
        "Do not edit by hand.",
        "[",
    ]
    for a, b in itertools.product(DTYPES, DTYPES):
        r = str(np.promote_types(a, b))
        lines.append(f"    (DType::{DTYPES[a]}, DType::{DTYPES[b]}, DType::{DTYPES[r]}),")
    lines.append("]")
    return "\n".join(lines) + "\n"


def gen_casting() -> str:
    """(from, to, safe, same_kind) for every numeric pair."""
    lines = [
        "// Generated from real numpy 2.5.2 via numpy.can_cast. "
        "Do not edit by hand.",
        "// (from, to, safe, same_kind)",
        "[",
    ]
    for a, b in itertools.product(DTYPES, DTYPES):
        safe = str(bool(np.can_cast(a, b, "safe"))).lower()
        same = str(bool(np.can_cast(a, b, "same_kind"))).lower()
        lines.append(f"    (DType::{DTYPES[a]}, DType::{DTYPES[b]}, {safe}, {same}),")
    lines.append("]")
    return "\n".join(lines) + "\n"


def gen_string_lengths() -> str:
    """Smallest `S<n>`/`U<n>` each numeric dtype casts safely into."""
    lines = [
        "// Generated from real numpy 2.5.2: the smallest n for which "
        "can_cast(dtype, 'S<n>') holds.",
        "// Do not edit by hand.",
        "[",
    ]
    for name, variant in DTYPES.items():
        n = next(n for n in range(1, 512) if np.can_cast(name, f"S{n}"))
        m = next(n for n in range(1, 512) if np.can_cast(name, f"U{n}"))
        assert n == m, (name, n, m)
        lines.append(f"    (DType::{variant}, {n}),")
    lines.append("]")
    return "\n".join(lines) + "\n"


def main() -> None:
    (SRC / "promotion_table.inc").write_text(gen_promotion())
    (SRC / "casting_table.inc").write_text(gen_casting())
    (SRC / "string_lengths.inc").write_text(gen_string_lengths())
    print(f"wrote promotion/casting/string-length tables to {SRC}")


if __name__ == "__main__":
    main()
