"""Introspection helpers for NumPy override protocols."""


def get_overridable_numpy_ufuncs():
    return set()


def allows_array_ufunc_override(func):
    return False


def get_overridable_numpy_array_functions():
    from .._core.overrides import ARRAY_FUNCTIONS
    return set(ARRAY_FUNCTIONS)


def allows_array_function_override(func):
    from .._core.overrides import ARRAY_FUNCTIONS
    return func in ARRAY_FUNCTIONS
