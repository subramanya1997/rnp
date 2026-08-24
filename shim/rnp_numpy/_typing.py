"""Runtime-only typing aliases needed by copied NumPy Python modules."""

from typing import Any


class NDArray:
    """Typing-only stand-in; runtime values are the extension ndarray."""

    def __class_getitem__(cls, item):
        return Any

__all__ = ["NDArray"]
