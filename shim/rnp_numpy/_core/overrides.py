"""Implementation of ``__array_function__`` overrides from NEP 18."""

from ..lib._rnp_compat import (
    ARRAY_FUNCTIONS,
    _ArrayFunctionDispatcher,
    _get_implementing_args,
    array_function_dispatch,
    array_function_from_dispatcher,
    array_function_like_doc,
    finalize_array_function_like,
    get_array_function_like_doc,
    set_array_function_like_doc,
    set_module,
    verify_matching_signatures,
)

__all__ = [
    "ARRAY_FUNCTIONS", "_ArrayFunctionDispatcher", "_get_implementing_args",
    "array_function_dispatch", "array_function_from_dispatcher",
    "array_function_like_doc", "finalize_array_function_like",
    "get_array_function_like_doc", "set_array_function_like_doc",
    "set_module", "verify_matching_signatures",
]
