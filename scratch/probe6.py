import numpy as np, numpy._core.umath as um
names=set()
for mod in (np,um):
    for n in dir(mod):
        v=getattr(mod,n,None)
        if type(v) is np.ufunc: names.add(n)
for n in sorted(names):
    f=getattr(np,n,None)
    if type(f) is not np.ufunc: f=getattr(um,n)
    ol=[t for t in f.types if 'O' in t]
    if ol: print(f"{n:20s} {f.nin}->{f.nout} ident={f.identity!r:8s} {ol}")
print("=== NO object loop ===")
print(sorted(n for n in names if not any('O' in t for t in (getattr(np,n,None) if type(getattr(np,n,None)) is np.ufunc else getattr(um,n)).types)))
