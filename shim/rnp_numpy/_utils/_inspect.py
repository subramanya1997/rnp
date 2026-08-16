"""A tiny `inspect.getargspec` stand-in, as numpy carries one."""

import collections
import inspect

__all__ = ["getargspec", "formatargspec"]

ArgSpec = collections.namedtuple(
    "ArgSpec", "args varargs varkw defaults")


def getargspec(func):
    sig = inspect.signature(func)
    args, varargs, varkw, defaults = [], None, None, []
    for name, p in sig.parameters.items():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            varargs = name
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            varkw = name
        else:
            args.append(name)
            if p.default is not inspect.Parameter.empty:
                defaults.append(p.default)
    return ArgSpec(args, varargs, varkw, tuple(defaults) or None)


def formatargspec(args, varargs=None, varkw=None, defaults=None, **kwargs):
    parts = list(args)
    if defaults:
        for i, d in enumerate(defaults):
            parts[len(args) - len(defaults) + i] += f"={d!r}"
    if varargs:
        parts.append("*" + varargs)
    if varkw:
        parts.append("**" + varkw)
    return "(" + ", ".join(parts) + ")"
