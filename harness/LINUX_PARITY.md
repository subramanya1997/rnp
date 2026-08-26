# Linux parity status

## Verified result

On the x86-64 Linux parity host, NumPy 2.5.2 and RNP currently produce:

```text
36510 comparisons, 0 divergences
```

The previous nine divergences are closed:

| Count | Surface | Verified RNP result |
|---:|---|---|
| 2 | `np.float128`, `np.complex256` aliases | present and identical to `longdouble` / `clongdouble` |
| 3 | `dtype(longdouble)` itemsize, name, and string | 16 bytes, `float128`, `<f16` |
| 3 | `dtype(clongdouble)` itemsize, name, and string | 32 bytes, `complex256`, `<c32` |
| 1 | `sctypeDict` keys | all 51 NumPy keys, including `float128` and `complex256` |

A direct post-harness probe also verifies:

- 16-byte alignment and the x87 payload layout (64-bit significand followed
  by the 15-bit biased exponent/sign word and six zero padding bytes);
- lossless parsing and repr of `1.0000000000000000001`;
- subtraction from `1.0` produces `1.084202172485504434e-19`, below binary64
  epsilon;
- every exposed `finfo` field matches NumPy for both `longdouble` and
  `clongdouble`, including max, tiny, and smallest subnormal.

## Implementation boundary

Linux x86-64 uses the software `F80` / `C160` element types. `F80` is an x87
80-bit value in NumPy's 16-byte storage slot, and implements exact binary64
conversion, decimal parsing/formatting, comparisons, and software add,
subtract, multiply, and divide over its 64-bit significand and 15-bit
exponent. `C160` stores two `F80` components in 32 bytes.

The wider transcendental ufunc surface currently routes through binary64; it
was not required by the differential checker or the upstream regression gate.
Storage, dtype metadata, scalar repr, machine limits, promotion, and elementary
arithmetic remain in the extended representation.

The implementation is selected only for Linux x86-64. Other platforms keep
the existing behavior; in particular, macOS continues to model
`longdouble`/`clongdouble` as `float64`/`complex128` storage with their distinct
scalar aliases.

## Full-suite scoreboard, Linux x86_64 (2026-08-25, main @ f036983)

core 32316/34229 (94.4%), lib 4153/4759, ma 4177/4376, fft 157/159,
random 1324/1349, polynomial 566/610, top 129/207, matrixlib 189/209,
linalg 485/485 (100%) — GRAND TOTAL 43938/46383 (94.7%).

The linalg gap (originally 158/485) was closed by extending the dlopen
backend to the bundled library's LAPACK entry points. Symbol scheme note:
numpy's scipy-openblas64 exports Fortran LAPACK as scipy_<routine>_64_
(e.g. scipy_dgesv_64_) and CBLAS as scipy_cblas_<routine>64_ — plain,
LAPACKE, and unsuffixed forms are absent. 26 routines resolved; Linux
routes through the same linalg kernels as macOS with runtime backend
gating, and the F80/C160 long-double paths cover linalg on Linux too.
