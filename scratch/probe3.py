import numpy as np
class NoCmp:
    def __lt__(s,o): return False
    def __gt__(s,o): return False
    def __eq__(s,o): return False
    def __repr__(s): return "NoCmp()"
try: print("sign none:", repr(np.sign(np.array([NoCmp()],dtype=object))))
except Exception as e: print("sign none:", type(e).__name__, e)

# binary ufuncs
bins = ["add","subtract","multiply","divide","true_divide","floor_divide","remainder","mod","fmod",
 "power","float_power","divmod","bitwise_and","bitwise_or","bitwise_xor","left_shift","right_shift",
 "less","less_equal","greater","greater_equal","equal","not_equal","logical_and","logical_or",
 "logical_xor","maximum","minimum","fmax","fmin","hypot","arctan2","copysign","nextafter","ldexp",
 "logaddexp","logaddexp2","heaviside","gcd","lcm","matmul"]
a = np.array([6,7],dtype=object); b=np.array([2,3],dtype=object)
for n in bins:
    f = getattr(np, n, None)
    if f is None: print(n,"MISSING"); continue
    try: print(f"{n}: {f(a,b)!r}")
    except Exception as e: print(f"{n}: {type(e).__name__}: {e}")
