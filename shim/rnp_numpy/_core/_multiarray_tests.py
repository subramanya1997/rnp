"""Stand-in for numpy._core._multiarray_tests (a C test-support extension).

The real module exposes internals the port does not have. Every entry point
raises NotImplementedError so that test *collection* succeeds and only the
tests that actually need these helpers fail.
"""


def _missing(name):
    def fn(*args, **kwargs):
        raise NotImplementedError(
            f"_multiarray_tests.{name} is not implemented by rnp")
    fn.__name__ = name
    return fn


_NAMES = [
    "create_custom_field_dtype", "get_fpu_mode", "getset_numericops",
    "format_float_OSprintf_g", "get_buffer_info", "get_c_wrapping_array",
    "internal_overlap", "solve_diophantine", "array_indexing",
    "test_nditer_too_large", "npy_char_deprecation", "run_byteorder_converter",
    "run_casting_converter", "run_clipmode_converter",
    "run_intp_converter", "run_scalar_intp_converter", "run_selectkind_converter",
    "run_sortkind_converter", "run_searchside_converter", "identityhash_tester",
    "incref_elide", "incref_elide_l", "npy_ensurenocopy", "get_struct_alignments",
    "get_all_cast_information", "corrupt_or_fix_bufferinfo",
]

for _n in _NAMES:
    globals()[_n] = _missing(_n)


def __getattr__(name):
    return _missing(name)
