import numpy as np
class B:
    def __init__(s,v=1): s.v=v
    def __repr__(s): return f"B({s.v})"
o=lambda *xs: np.array(list(xs),dtype=object)
for n in ["fmod","hypot","arctan2","gcd","lcm","logical_xor","maximum","minimum","fmax","fmin","matmul","add"]:
    try: print(n, repr(getattr(np,n)(o(B(1)),o(B(2)))))
    except Exception as e: print(n, type(e).__name__+":", e)
print("---gcd with class having gcd method")
class G:
    def gcd(s,o): return "gcd!"
    def lcm(s,o): return "lcm!"
    def fmod(s,o): return "fmod!"
print(repr(np.gcd(o(G()),o(1))), repr(np.lcm(o(G()),o(1))), repr(np.fmod(o(G()),o(1))))
print("--- comparison return values")
class C:
    def __lt__(s,o): return "LT"
    def __eq__(s,o): return "EQ"
print("less:", repr(np.less(o(C()),o(1))), np.less(o(C()),o(1)).dtype)
print("equal:", repr(np.equal(o(C()),o(1))))
print("--- maximum with custom cmp")
class Mx:
    def __init__(s,v):s.v=v
    def __repr__(s):return f"Mx({s.v})"
    def __gt__(s,o): print("   gt"); return s.v>o.v
    def __lt__(s,o): print("   lt"); return s.v<o.v
    def __ge__(s,o): print("   ge"); return s.v>=o.v
    def __le__(s,o): print("   le"); return s.v<=o.v
print("max:",repr(np.maximum(o(Mx(1)),o(Mx(2)))))
print("min:",repr(np.minimum(o(Mx(1)),o(Mx(2)))))
print("fmax:",repr(np.fmax(o(Mx(1)),o(Mx(2)))))
print("--- promotion")
print(np.result_type(np.dtype(object), np.int64))
print(repr(np.add(o(1,2), np.array([10,20]))))
print(repr(np.add(o(1,2), 5)), repr(np.add(o(1,2), np.float64(2.5))))
print(repr(np.equal(o(1,2), np.array([1,5]))))
print("--- no-loop error text")
try: np.isnan(o(1))
except Exception as e: print(type(e).__mro__, repr(str(e)))
