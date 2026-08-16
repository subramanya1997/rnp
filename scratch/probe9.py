import numpy as np
o=lambda *xs: np.array(list(xs),dtype=object)
import fractions, decimal
for v in (1, 1.5, np.float64(2.0), fractions.Fraction(1,2), "s", b"b", [1], np.int8(3)):
    try: np.sqrt(o(v))
    except Exception as e: print(type(v).__name__, "->", e)
print("---- where on object with identity ----")
for f in ("multiply","logical_and","bitwise_or","minimum"):
    try: print(f, repr(getattr(np,f).reduce(o(1,2,3), where=[True,False,True])))
    except Exception as e: print(f, type(e).__name__, e)
print("--- cmp with object out ---")
try:
    ob=np.empty(1,dtype=object); print(repr(np.less(o(1),o(2),out=ob)), ob.dtype)
except Exception as e: print(type(e).__name__, e)
print("--- asarray of Fraction ---")
print(np.asarray([fractions.Fraction(1,2)]).dtype, np.add(fractions.Fraction(1,2), fractions.Fraction(1,3)))
print(np.result_type(fractions.Fraction(1,2)))
print("--- object scalar ---")
print(repr(np.sqrt(np.array(4.0,dtype=object))) if False else "")
class Q:
    def sqrt(s): return 2
print("0d:", repr(np.sqrt(np.array(Q(),dtype=object))), type(np.sqrt(np.array(Q(),dtype=object))))
print("--- exception chaining ---")
class Bad:
    @property
    def sqrt(self): raise RuntimeError("inner")
try: np.sqrt(o(Bad()))
except Exception as e: print(type(e).__name__, e, "| cause:", repr(e.__cause__))
