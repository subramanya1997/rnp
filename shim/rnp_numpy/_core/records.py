"""`numpy._core.records` — the record scalar type.

Only `record` lives here so far; `recarray` and the `np.rec` namespace are a
later milestone. `record` is a genuine subclass of `np.void`, matching numpy's
`(record, void, flexible, generic, object)` MRO, so `issubclass`/`isinstance`
checks and dtype-hierarchy walks see the same thing numpy shows them.
"""

from .._scalars import void

__all__ = ["record"]


class record(void):
    """A data-type scalar that allows field access as attribute lookup.
    """

    # numpy sets these by hand so the class prints as `numpy.record`.
    __name__ = "record"
    __module__ = "numpy"
    __slots__ = ()

    def __getattribute__(self, attr):
        if attr in ("setfield", "getfield", "dtype"):
            return void.__getattribute__(self, attr)
        try:
            return void.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = void.__getattribute__(self, "dtype").fields
        res = fielddict.get(attr, None) if fielddict else None
        if res:
            obj = self.getfield(*res[:2])
            # A field that itself has fields comes back as a record.
            try:
                dt = obj.dtype
            except AttributeError:
                # happens if field is Object type
                return obj
            if dt.names is not None:
                return obj.view((self.__class__, obj.dtype))
            return obj
        raise AttributeError(f"'record' object has no attribute '{attr}'")

    def __setattr__(self, attr, val):
        if attr in ("setfield", "getfield", "dtype"):
            raise AttributeError(f"Cannot set '{attr}' attribute")
        fielddict = void.__getattribute__(self, "dtype").fields
        res = fielddict.get(attr, None) if fielddict else None
        if res:
            return self.setfield(val, *res[:2])
        if getattr(self, attr, None):
            return void.__setattr__(self, attr, val)
        raise AttributeError(f"'record' object has no attribute '{attr}'")

    def __getitem__(self, indx):
        obj = void.__getitem__(self, indx)
        # Mirror __getattribute__: a nested structured field is a record too.
        if isinstance(obj, void) and obj.dtype.names is not None:
            return obj.view((self.__class__, obj.dtype))
        return obj

    def pprint(self):
        """Pretty-print all fields."""
        names = self.dtype.names or ()
        maxlen = max((len(n) for n in names), default=0)
        fmt = '%% %ds: %%s' % maxlen
        rows = [fmt % (name, getattr(self, name)) for name in names]
        return "\n".join(rows)
