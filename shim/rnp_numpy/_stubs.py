"""The ufunc object model, plus loud placeholders for the parts of the numpy
surface the port has not reached.

A stub must *exist* (so upstream test modules import and collect) but must
never silently produce a wrong answer: calling one raises NotImplementedError.
"""


class ufunc:
    """A minimal stand-in for `numpy.ufunc`.

    `impl` is the native `_rnp` function when the port implements the ufunc;
    otherwise every call raises NotImplementedError. The attributes numpy's
    tests introspect (`nin`, `nout`, `nargs`, `types`, `identity`,
    `signature`) are present in both cases.
    """

    def __init__(self, name, nin=2, nout=1, impl=None, identity=None,
                 qualname=None):
        self.__name__ = name
        self._qualname = qualname or f"numpy.{name}"
        self._impl = impl
        self.nin = nin
        self.nout = nout
        self.nargs = nin + nout
        self.types = []
        self.ntypes = 0
        self.identity = identity
        self.signature = None

    def __call__(self, *args, **kwargs):
        if self._impl is None:
            raise NotImplementedError(
                f"{self._qualname} is not implemented by rnp yet")
        return self._impl(*args, **kwargs)

    def __repr__(self):
        return f"<ufunc '{self.__name__}'>"

    def _unsupported(self, what):
        raise NotImplementedError(
            f"{self._qualname}.{what} is not implemented by rnp yet")

    def reduce(self, array, axis=0, dtype=None, out=None, keepdims=False,
               initial=None, where=True):
        import rnp_numpy as np
        method = _REDUCE_METHOD.get(self.__name__)
        if method is None or initial is not None or where is not True:
            self._unsupported("reduce")
        kwargs = {"axis": axis, "out": out, "keepdims": keepdims}
        if method in ("sum", "prod"):
            kwargs["dtype"] = dtype
        return getattr(np.asarray(array), method)(**kwargs)

    def accumulate(self, *a, **k):
        self._unsupported("accumulate")

    def reduceat(self, *a, **k):
        self._unsupported("reduceat")

    def outer(self, *a, **k):
        self._unsupported("outer")

    def at(self, *a, **k):
        self._unsupported("at")

    def resolve_dtypes(self, *a, **k):
        self._unsupported("resolve_dtypes")


#: Reductions the port can answer natively.
_REDUCE_METHOD = {
    "add": "sum",
    "multiply": "prod",
    "maximum": "max",
    "minimum": "min",
    "logical_and": "all",
    "logical_or": "any",
}


class _NotImplementedCallable:
    def __init__(self, name):
        self.__name__ = name.rsplit(".", 1)[-1]
        self._qualname = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self._qualname} is not implemented by rnp yet")

    def __repr__(self):
        return f"<not-implemented {self._qualname}>"


def not_implemented(name):
    return _NotImplementedCallable(name)


def ufunc_stub(name, nin=2, nout=1):
    return ufunc(name.rsplit(".", 1)[-1], nin=nin, nout=nout, qualname=name)


class _InertObject:
    """An instance of a class the port does not implement.

    Upstream test modules construct these at import time; every operation on
    the result raises NotImplementedError.
    """

    def __init__(self, *args, **kwargs):
        pass

    def _nope(self, *args, **kwargs):
        raise NotImplementedError(
            f"numpy.{type(self).__name__} is not implemented by rnp yet")

    __array__ = __add__ = __radd__ = __sub__ = __rsub__ = _nope
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _nope
    __getitem__ = __setitem__ = __len__ = __iter__ = _nope

    def __repr__(self):
        return f"<not-implemented numpy.{type(self).__name__} instance>"


def inert_class(name):
    return type(name, (_InertObject,), {})
