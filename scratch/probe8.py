import numpy as np
o=lambda *xs: np.array(list(xs),dtype=object)
try: np.minimum.reduce(np.array([1.,2.]),where=[True,False])
except Exception as e: print("float min where:", type(e).__name__, e)
print("obj add where+initial:", repr(np.add.reduce(o(1,2,3),where=[True,False,True],initial=0)))
pass
out=np.empty(3,dtype=object); np.add(o(1,2,3),o(1,1,1),out=out); print("out:",repr(out))
out2=np.empty(3,dtype=object); print("wh ret:",repr(np.add(o(1,2,3),o(10,10,10),out=out2,where=[True,False,True])))
print("bcast:", repr(np.add(o(1,2,3), np.array([[10],[20]],dtype=object))))
r=np.add(np.array(1,dtype=object), 2); print("0d:", repr(r), type(r))
print("where no out:", repr(np.add(o(1,2,3),o(10,10,10),where=[True,False,True])))
print("cmp where:", repr(np.less(o(1,2,3),o(2,2,2),where=[True,False,True])))
print("obj sum dtype kwarg:", repr(np.add.reduce(o(1,2,3),dtype=object)))
print("np.sum on obj 2d:", repr(np.sum(np.array([[1,2],[3,4]],dtype=object),axis=1)))
# error propagation
class E:
    def __add__(s,x): raise ZeroDivisionError("boom-from-elem")
try: np.add(o(E(),E()),o(1,1))
except Exception as e: print("err prop:", type(e).__name__, e)
try: np.add(o(1,2),o(1,"a"))
except Exception as e: print("mixed err:", type(e).__name__, e)
# NotImplemented handling
class NI:
    def __add__(s,x): return NotImplemented
    def __repr__(s): return "NI()"
try: print(repr(np.add(o(NI()),o(1))))
except Exception as e: print("NI:", type(e).__name__, e)
# reflected
class R:
    def __radd__(s,x): return "reflected!"
print("refl:", repr(np.add(o(1),o(R()))))
# sqrt error multi-element index
class M:
    def sqrt(s): return 1
try: np.sqrt(o(M(),M(),3))
except Exception as e: print("sqrt idx:", type(e).__name__, e)
# str
print("str mul:", repr(np.multiply(o("ab"),o(3))))
print("list add:", repr(np.add(o([1,2]),o([3]))))
print("None elem:", end=" ")
try: print(repr(np.add(o(None),o(1))))
except Exception as e: print(type(e).__name__, e)
print("np.empty obj add:", end=" ")
try: print(repr(np.add(np.empty(2,dtype=object), 1)))
except Exception as e: print(type(e).__name__, e)
