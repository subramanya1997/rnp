import numpy as np
class Bare:
    def __init__(s,v): s.v=v
    def __repr__(s): return f"Bare({s.v})"

unaries = ["negative","positive","absolute","invert","sign","square","reciprocal","conjugate",
  "sqrt","exp","exp2","expm1","log","log2","log10","log1p","sin","cos","tan","arcsin","arccos",
  "arctan","sinh","cosh","tanh","arcsinh","arccosh","arctanh","floor","ceil","trunc","rint",
  "fabs","isnan","isinf","isfinite","signbit","spacing","cbrt","degrees","radians","deg2rad",
  "rad2deg","logical_not","abs","fabs","_ones_like" ]
import numpy._core.umath as um
for name in unaries:
    f = getattr(np, name, None) or getattr(um, name, None)
    if f is None:
        print(f"{name}: NOT A UFUNC"); continue
    a = np.array([Bare(1)], dtype=object)
    try:
        r = f(a)
        print(f"{name}: OK -> {r!r}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {e}")
