import numpy as np
o=lambda *xs: np.array(list(xs),dtype=object)
class C:
    def __lt__(s,o): return "LT"
    def __eq__(s,o): return "EQ"
r=np.less(o(C()),o(1)); print("less:", repr(r), r.dtype)
r=np.equal(o(C()),o(1)); print("equal:", repr(r), r.dtype)
class Mx:
    def __init__(s,v):s.v=v
    def __repr__(s):return f"Mx({s.v})"
    def __ge__(s,o): print("   ge"); return s.v>=o.v
    def __le__(s,o): print("   le"); return s.v<=o.v
print("max:",repr(np.maximum(o(Mx(1)),o(Mx(2)))))
print("min:",repr(np.minimum(o(Mx(1)),o(Mx(2)))))
print("--- promotion")
print(np.result_type(np.dtype(object), np.int64), np.promote_types(object,np.int64))
print(repr(np.add(o(1,2), np.array([10,20]))))
print(repr(np.add(o(1,2), 5)), repr(np.add(o(1,2), np.float64(2.5))))
print(repr(np.equal(o(1,2), np.array([1,5]))))
print(repr(np.add(np.array([10,20]), o(1,2))))
print("elementwise conv:", repr(np.add(o(1), np.array([[1,2],[3,4]]))))
print("nonobj type:", [type(x) for x in np.add(o(1,2), np.array([10,20],dtype=np.int8))])
print("float:", [type(x) for x in np.add(o(1,2), np.array([1.5,2.5],dtype=np.float32))])
print("str+obj:", repr(np.add(o("a"), np.array(["b"]))))
print("--- no-loop error text")
try: np.isnan(o(1))
except Exception as e: print(type(e).__name__, type(e).__mro__, repr(str(e)))
import numpy._core._exceptions as ex
print(ex._UFuncNoLoopError)
