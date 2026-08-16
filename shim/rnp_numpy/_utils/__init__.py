"""`numpy._utils` — private helpers with no dependency on the rest of numpy.

Written from the documented behaviour, not copied: `set_module` rewrites
`__module__`, and the conversion helpers coerce between `str` and `bytes`.
"""

import functools  # noqa: F401  (numpy re-exports it via this module's users)

from ._conversions import asbytes, asunicode  # noqa: F401


def set_module(module):
    """Override `__module__` on a function or class."""

    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func

    return decorator


def _rename_parameter(old_names, new_names, dep_version=None):
    """numpy's decorator for renamed keyword arguments."""

    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            for old, new in zip(old_names, new_names):
                if old in kwargs:
                    kwargs[new] = kwargs.pop(old)
            return fun(*args, **kwargs)

        return wrapper

    return decorator
