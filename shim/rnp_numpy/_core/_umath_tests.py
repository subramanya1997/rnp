"""Stand-in for numpy._core._umath_tests (a C test-support extension)."""

from .._stubs import ufunc_stub

inner1d = ufunc_stub("numpy._core._umath_tests.inner1d")
innerwt = ufunc_stub("numpy._core._umath_tests.innerwt", nin=3)
matrix_multiply = ufunc_stub("numpy._core._umath_tests.matrix_multiply")
matmul = ufunc_stub("numpy._core._umath_tests.matmul")
euclidean_pdist = ufunc_stub("numpy._core._umath_tests.euclidean_pdist", nin=1)
cumsum = ufunc_stub("numpy._core._umath_tests.cumsum", nin=1)
cross1d = ufunc_stub("numpy._core._umath_tests.cross1d")
conjugate = ufunc_stub("numpy._core._umath_tests.conjugate", nin=1)
always_error = ufunc_stub("numpy._core._umath_tests.always_error")
always_error_gufunc = ufunc_stub("numpy._core._umath_tests.always_error_gufunc")
indexed_negative = ufunc_stub("numpy._core._umath_tests.indexed_negative", nin=1)
_pickleable_module_global = ufunc_stub(
    "numpy._core._umath_tests._pickleable_module_global")


def __getattr__(name):
    return ufunc_stub(f"numpy._core._umath_tests.{name}")
