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
random 1324/1349, polynomial 566/610, top 129/207, matrixlib 99/209,
linalg 158/485 — GRAND TOTAL 43079/46383 (92.9%).

Known gap driving linalg (32.6% vs 100% on macOS) and matrixlib: the Linux
runtime backend dlopens numpy's bundled scipy-openblas64 for BLAS (gemm/dotc)
but does not yet resolve its LAPACK entry points; macOS links Accelerate
LAPACK. Extending the dlopen backend to LAPACK closes it.
