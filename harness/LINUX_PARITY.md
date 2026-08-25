# Linux parity status

## Verified remainder

On the x86-64 Linux parity host, NumPy 2.5.2 and RNP currently produce:

```text
36506 comparisons, 9 divergences
```

All nine divergences are the unsupported x87 extended-precision dtype surface:

| Count | Surface | NumPy on x86-64 Linux | RNP today |
|---:|---|---|---|
| 2 | `np.float128`, `np.complex256` aliases | distinct scalar types | absent |
| 3 | `dtype(longdouble)` itemsize, name, and string | 16 bytes, `float128` | 8 bytes, `float64` |
| 3 | `dtype(clongdouble)` itemsize, name, and string | 32 bytes, `complex256` | 16 bytes, `complex128` |
| 1 | `sctypeDict` keys | includes `float128`, `complex256` | both absent |

The exact failing checks are:

```text
alias np.complex256
alias np.float128
dtype(longdouble).itemsize
dtype(longdouble).name
str(dtype(longdouble))
dtype(clongdouble).itemsize
dtype(clongdouble).name
str(dtype(clongdouble))
sctypeDict keys
```

## Why this is documented rather than aliased

On this platform NumPy's `longdouble` is an 80-bit x87 extended-precision
value stored in a 16-byte dtype slot; `clongdouble` stores two such values in
32 bytes. RNP has no 80-bit element representation, descriptor, cast matrix,
scalar type, or arithmetic/reduction/ufunc loop family. Advertising the NumPy
names while storing `float64`/`complex128` would make itemsize, strides, buffer
layout, precision, promotion, and arithmetic semantics incorrect.

Closing these nine checks therefore requires an end-to-end extended-precision
dtype implementation. The aliases and metadata should be added only with that
real storage and loop support. The remaining Linux harness surface is exact;
the nine checks above are not hidden or excluded from `dev_check.py`.
