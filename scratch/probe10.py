import numpy as np
from fractions import Fraction
from decimal import Decimal
o=lambda *xs: np.array(list(xs),dtype=object)
class Bad:
    @property
    def sqrt(self): raise RuntimeError("inner")
try: np.sqrt(o(Bad()))
except Exception as e: print(type(e).__name__, e, "| cause:", repr(e.__cause__), "| ctx:", repr(e.__context__))
class NotCallable:
    sqrt = 5
try: np.sqrt(o(NotCallable()))
except Exception as e: print(type(e).__name__, e, "| cause:", repr(e.__cause__))
print("gcd frac:", end=" ")
try: print(repr(np.gcd(o(Fraction(4,1)),o(Fraction(6,1)))))
except Exception as e: print(type(e).__name__, e)
print("gcd dec:", end=" ")
try: print(repr(np.gcd(o(Decimal(4)),o(Decimal(6)))))
except Exception as e: print(type(e).__name__, e)
print("gcd float:", end=" ")
try: print(repr(np.gcd(o(4.0),o(6.0))))
except Exception as e: print(type(e).__name__, e)
print("gcd neg:", repr(np.gcd(o(-4),o(6))), "lcm neg:", repr(np.lcm(o(-4),o(6))))
print("lcm dec:", end=" ")
try: print(repr(np.lcm(o(Decimal(4)),o(Decimal(6)))))
except Exception as e: print(type(e).__name__, e)
print("pow3:", repr(np.power(o(2),o(3))))
print("bool out cmp:", np.equal(o(1),o(1)).dtype)
print("obj vs obj ne identity:", repr(np.equal(o(float('nan')),o(float('nan')))))
x=o(float('nan')); print("self eq:", repr(np.equal(x,x)))
