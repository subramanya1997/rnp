import numpy as np, sys

def t(label, fn):
    try:
        r = fn()
        print(f"{label:60s} -> OK {r!r}")
    except Exception as e:
        print(f"{label:60s} -> {type(e).__name__}: {e}")

print("=== construction with out-of-range int ===")
for dt in [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.uint32, np.int64, np.uint64]:
    n = dt.__name__
    t(f"np.array([300], dtype={n})", lambda dt=dt: np.array([300], dtype=dt).tolist())
    t(f"np.array(300, dtype={n})", lambda dt=dt: np.array(300, dtype=dt).tolist())
    t(f"{n}(300)", lambda dt=dt: dt(300))
    t(f"np.array([-300], dtype={n})", lambda dt=dt: np.array([-300], dtype=dt).tolist())
    t(f"np.full((2,), 300, dtype={n})", lambda dt=dt: np.full((2,),300,dtype=dt).tolist())
    t(f"zeros({n}).fill(300)", lambda dt=dt: (np.zeros(2,dtype=dt).fill(300)))
    def assign(dt=dt):
        a = np.zeros(2, dtype=dt); a[0] = 300; return a.tolist()
    t(f"a[0]=300 ({n})", assign)
    def assign_slice(dt=dt):
        a = np.zeros(2, dtype=dt); a[:] = 300; return a.tolist()
    t(f"a[:]=300 ({n})", assign_slice)
    t(f"np.asarray([300]).astype({n})", lambda dt=dt: np.asarray([300]).astype(dt).tolist())
    print()

print("=== huge ints ===")
h = 2**100
t("np.array(2**100)", lambda: (np.array(h).dtype, np.array(h)))
t("np.array([2**100])", lambda: (np.array([h]).dtype,))
t("np.array(2**100, dtype=np.int64)", lambda: np.array(h, dtype=np.int64))
t("np.array([2**100], dtype=np.int64)", lambda: np.array([h], dtype=np.int64))
t("np.int64(2**100)", lambda: np.int64(h))
t("np.uint64(2**100)", lambda: np.uint64(h))
t("np.float64(2**100)", lambda: np.float64(h))
t("np.array(2**100, dtype=np.float64)", lambda: np.array(h,dtype=np.float64))
t("np.array(2**100, dtype=object)", lambda: np.array(h,dtype=object))
t("np.arange(3) + 2**100", lambda: np.arange(3)+h)
t("np.arange(3).astype(np.float64) + 2**100", lambda: np.arange(3.0)+h)
t("np.array([1,2**100])", lambda: (np.array([1,h]).dtype,))
t("np.array(2**64)", lambda: (np.array(2**64).dtype,))
t("np.array(2**64-1)", lambda: (np.array(2**64-1).dtype,))
t("np.array(2**63)", lambda: (np.array(2**63).dtype,))
t("np.array(-2**63)", lambda: (np.array(-2**63).dtype,))
t("np.array(-2**64)", lambda: (np.array(-2**64).dtype,))
t("np.array([2**64-1, -1])", lambda: (np.array([2**64-1,-1]).dtype,))
t("np.result_type(2**100)", lambda: np.result_type(h))
t("np.result_type(np.int8, 2**100)", lambda: np.result_type(np.int8, h))
t("np.uint8(1) + 2**100", lambda: np.uint8(1)+h)
t("np.arange(3, dtype=np.uint8) < 2**100", lambda: np.arange(3,dtype=np.uint8) < h)
t("np.min_scalar_type(2**100)", lambda: np.min_scalar_type(h))

print("=== result_type multi ===")
cases = [
 (np.int8, np.uint8, np.int8),
 (np.uint8, np.int8, np.int8),
 (np.int64, np.uint64, np.float16),
 (np.uint8, np.int8, np.float16),
 (np.int8, np.uint16, np.float16),
 (np.uint16, np.int8, np.float16),
 (np.int16, np.uint16, np.float32),
 (np.float16, np.int64, np.uint64),
]
for c in cases:
    names = ",".join(x.__name__ for x in c)
    fold = np.promote_types(np.promote_types(c[0], c[1]), c[2])
    try:
        rt = np.result_type(*c)
    except Exception as e:
        rt = f"{type(e).__name__}"
    print(f"result_type({names}) = {rt}   leftfold={fold}")
