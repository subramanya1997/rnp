"""Small public helpers exposed by NumPy's compiled random common module."""

from collections import namedtuple


interface = namedtuple(
    "interface",
    [
        "state_address",
        "state",
        "next_uint64",
        "next_uint32",
        "next_double",
        "bit_generator",
    ],
)


__all__ = ["interface"]
