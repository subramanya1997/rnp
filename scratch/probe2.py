import numpy as np, math
from fractions import Fraction
from decimal import Decimal

class M:
    def __init__(s,v=1): s.v=v
    def __repr__(s): return f"M({s.v})"
    def sqrt(s): return "sqrt-called"
    def conjugate(s): return "conj-called"
    def rint(s): return "rint-called"
    def __floor__(s): return "floor-called"
    def __ceil__(s): return "ceil-called"
    def __trunc__(s): return "trunc-called"
    def __abs__(s): return "abs-called"
print("sqrt", np.sqrt(np.array([M()],dtype=object)))
print("conj", np.conjugate(np.array([M()],dtype=object)))
print("rint", np.rint(np.array([M()],dtype=object)))
for n in ("floor","ceil","trunc"):
    try: print(n, getattr(np,n)(np.array([M()],dtype=object)))
    except Exception as e: print(n, type(e).__name__, e)
print("abs", np.absolute(np.array([M()],dtype=object)))

# sign probing behaviour
class S:
    def __init__(s,v): s.v=v
    def __repr__(s): return f"S({s.v})"
    def __lt__(s,o): print("  __lt__",o); return s.v < 0
    def __gt__(s,o): print("  __gt__",o); return s.v > 0
    def __eq__(s,o): print("  __eq__",o); return s.v == 0
for v in (-5,0,5):
    print("sign S(%d)"%v, repr(np.sign(np.array([S(v)],dtype=object))))
print("sign fractions", repr(np.sign(np.array([Fraction(-3,2),Fraction(0),Fraction(5,2)],dtype=object))))
print("sign decimals", repr(np.sign(np.array([Decimal("-1.5"),Decimal(0)],dtype=object))))
print("sign ints", repr(np.sign(np.array([-3,0,7],dtype=object))))
print("sign strs", end=" ")
try: print(repr(np.sign(np.array(["a"],dtype=object))))
except Exception as e: print(type(e).__name__, e)
print("dtype of results:", np.sqrt(np.array([M()],dtype=object)).dtype, np.sign(np.array([1],dtype=object)).dtype)
print("logical_not", repr(np.logical_not(np.array([0,1,"",[],[1]],dtype=object))))
print("isnan? ", end="")
