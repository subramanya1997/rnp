import numpy as np, operator, sys

def t(label, fn):
    try:
        r = fn()
        print(f"{label:58s} -> OK {r!r}")
    except Exception as e:
        print(f"{label:58s} -> {type(e).__name__}: {e}")

print("=== scalar comparisons out of bound ===")
for sct in [np.int8, np.uint8, np.uint64, np.int64]:
    for other in [-200, -1, 0, 9, 10, 11, 2**63, 200]:
        for comp in [operator.eq, operator.lt, operator.ge]:
            t(f"{comp.__name__}({sct.__name__}(10), {other})", lambda s=sct,o=other,c=comp: (c(s(10),o), type(c(s(10),o)).__name__))

print("=== strings to int dtype ===")
t("np.array(['-129'], dtype=np.int8)", lambda: np.array(['-129'],dtype=np.int8))
t("np.array(['-129'], dtype=np.int16)", lambda: np.array(['-129'],dtype=np.int16))
t("np.int8('-129')", lambda: np.int8('-129'))
t("np.int8('-128')", lambda: np.int8('-128'))
t("np.array(['-128'], dtype=np.int8)", lambda: np.array(['-128'],dtype=np.int8))
t("np.uint8('256')", lambda: np.uint8('256'))
t("np.array(['256'], dtype=np.uint8)", lambda: np.array(['256'],dtype=np.uint8))
t("np.array(['12'], dtype=np.float32)", lambda: np.array(['12'],dtype=np.float32))
t("np.int64('9223372036854775808')", lambda: np.int64('9223372036854775808'))
t("np.array(['9223372036854775808'],dtype=np.int64)", lambda: np.array(['9223372036854775808'],dtype=np.int64))
t("np.uint64('-1')", lambda: np.uint64('-1'))
t("np.array(['-1'],dtype=np.uint64)", lambda: np.array(['-1'],dtype=np.uint64))
t("np.uint64('18446744073709551616')", lambda: np.uint64('18446744073709551616'))
t("np.array(['18446744073709551616'],dtype=np.uint64)", lambda: np.array(['18446744073709551616'],dtype=np.uint64))
t("np.array(2**63, dtype=np.int64)", lambda: np.array(2**63,dtype=np.int64))
t("np.array([2**63], dtype=np.int64)", lambda: np.array([2**63],dtype=np.int64))
t("np.int64(2**63)", lambda: np.int64(2**63))
t("np.array([-2**63-1], dtype=np.int64)", lambda: np.array([-2**63-1],dtype=np.int64))
t("np.array([2**64], dtype=np.uint64)", lambda: np.array([2**64],dtype=np.uint64))

print("=== huge int ufunc paths ===")
for uf in [np.add, np.power]:
    n = uf.__name__
    t(f"{n}(np.int64(0), 2**63)", lambda uf=uf: uf(np.int64(0), 2**63))
    t(f"{n}(np.uint64(0), 2**64)", lambda uf=uf: uf(np.uint64(0), 2**64))
    t(f"{n}(np.uint64(1), 2**63)", lambda uf=uf: (uf(np.uint64(1), 2**63), uf(np.uint64(1),2**63).dtype))
    t(f"{n}(np.int64(1), 2**63)", lambda uf=uf: uf(np.int64(1), 2**63))
    t(f"{n}(np.int64(1), 2**100)", lambda uf=uf: uf(np.int64(1), 2**100))
    t(f"{n}(1.0, 2**100)", lambda uf=uf: (uf(1.0, 2**100), type(uf(1.0,2**100)).__name__))
    t(f"{n}(1, 2**63, dtype=object)", lambda uf=uf: uf(1, 2**63, dtype=object))
for comp in [np.equal, np.not_equal, np.less_equal, np.less, np.greater_equal, np.greater]:
    t(f"{comp.__name__}(2**200, -2**200)", lambda c=comp: c(2**200, -2**200))
    t(f"{comp.__name__}(2**200,-2**200,dtype=object)", lambda c=comp: c(2**200, -2**200, dtype=object))

print("=== weak int with inexact ===")
for dt in "efdgFDG":
    try:
        sct = np.dtype(dt).type
        big = int(np.finfo(dt).max)*2
    except Exception as e:
        print(dt, "skip", e); continue
    t(f"{np.dtype(dt).name}(1) + too_big", lambda s=sct,b=big: (s(1)+b, (s(1)+b).dtype))
    t(f"np.add(np.array(1,{np.dtype(dt).name}), big, dtype={dt})", lambda d=dt,b=big: np.add(np.array(1,dtype=d), b, dtype=d))
    t(f"np.array(1,{np.dtype(dt).name}) + too_big", lambda d=dt,b=big: np.array(1,dtype=d)+b)

print("=== misc ===")
t("np.uint8(100) + 200", lambda: np.uint8(100)+200)
t("np.float32(1) + 3e100", lambda: np.float32(1)+3e100)
t("np.complex64(3) + complex(2**300)", lambda: np.complex64(3)+complex(2**300))
t("np.array([1],np.uint8) + 300", lambda: np.array([1],np.uint8)+300)
t("np.uint8(1) + 300", lambda: np.uint8(1)+300)
t("np.uint8(1) + -1", lambda: np.uint8(1)+-1)
t("np.uint8(3) ** 1000", lambda: np.uint8(3)**1000)
t("np.concatenate([np.float32(1), 1.], axis=None).dtype", lambda: np.concatenate([np.float32(1),1.],axis=None).dtype)
t("np.choose(1,[np.float32(1),1.]).dtype", lambda: np.choose(1,[np.float32(1),1.]).dtype)
t("np.r_[np.arange(5,dtype=np.int8), 255]", lambda: np.r_[np.arange(5,dtype=np.int8),255])
t("np.ones(100,'>u4') >= -1", lambda: (np.ones(100,'>u4')>=-1).all())
t("np.ones(100,'>u4') < -1", lambda: (np.ones(100,'>u4')<-1).any())
