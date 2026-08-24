"""Stand-in for numpy._core._operand_flag_tests (a C test-support extension).

It exposes exactly one ufunc, built to exercise operand-flag handling.
"""

from .._stubs import ufunc_stub

inplace_add = ufunc_stub("numpy._core._operand_flag_tests.inplace_add")


def __getattr__(name):
    return ufunc_stub(f"numpy._core._operand_flag_tests.{name}")
