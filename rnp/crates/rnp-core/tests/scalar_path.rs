//! `binary_scalar` must agree with the array driver, bit for bit.
//!
//! This lives in its own integration test — its own process — on purpose: the
//! engine's FP-flag accumulator is a process-global, so a unit test reading it
//! would race every other test cargo runs in parallel.

use rnp_core::element::Scalar;
use rnp_core::fpe;
use rnp_core::ops::{binary, binary_scalar, BinOp};
use rnp_core::{DType, NdArray};

/// The whole justification for `binary_scalar`: it must agree with the array
/// driver on the result dtype, the result *bits*, the FP flags and the error,
/// for every (dtype, dtype, op) triple over a value grid.
#[test]
fn scalar_matches_array() {
    let dts = [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::C64,
        DType::C128,
    ];
    use BinOp::*;
    let ops = [
        Add, Sub, Mul, Div, Eq, Ne, Lt, Le, Gt, Ge, Pow, FloatPower, FloorDiv, Mod, Fmod, Minimum,
        Maximum, Fmin, Fmax, Arctan2, Hypot, Copysign, Nextafter, Logaddexp, Logaddexp2, Heaviside,
        Ldexp, BitAnd, BitOr, BitXor, LShift, RShift, Gcd, Lcm, LogicalAnd, LogicalOr, LogicalXor,
    ];
    let vals: [Scalar; 11] = [
        Scalar::Int(0),
        Scalar::Int(1),
        Scalar::Int(2),
        Scalar::Int(-1),
        Scalar::Int(3),
        Scalar::Int(127),
        Scalar::Float(0.5),
        Scalar::Float(-0.0),
        Scalar::Float(f64::INFINITY),
        Scalar::Float(f64::NAN),
        Scalar::Float(1.5e30),
    ];
    let mut checked = 0usize;
    for &da in &dts {
        for &db in &dts {
            for &op in &ops {
                for &x in &vals {
                    for &y in &vals {
                        let mut a0 = NdArray::zeros(vec![], da).unwrap();
                        a0.set(&[], x).unwrap();
                        let mut b0 = NdArray::zeros(vec![], db).unwrap();
                        b0.set(&[], y).unwrap();
                        fpe::clear();
                        let want = binary(&a0, &b0, op);
                        let want_flags = fpe::take();
                        fpe::clear();
                        let got = binary_scalar(x.cast(da), da, y.cast(db), db, op)
                            .expect("simple dtypes are always handled");
                        let got_flags = fpe::take();
                        checked += 1;
                        match (want, got) {
                            (Ok(w), Ok((gd, gv))) => {
                                assert_eq!(w.dtype, gd, "{da:?} {op:?} {db:?} dtype");
                                let wv = w.get_flat(0);
                                assert!(
                                    same_bits(wv, gv),
                                    "{da:?}({x:?}) {op:?} {db:?}({y:?}): array {wv:?} vs scalar {gv:?}"
                                );
                                assert_eq!(
                                    want_flags, got_flags,
                                    "{da:?}({x:?}) {op:?} {db:?}({y:?}) flags"
                                );
                            }
                            (Err(w), Err(g)) => {
                                assert_eq!(
                                    format!("{w:?}"),
                                    format!("{g:?}"),
                                    "{da:?} {op:?} {db:?} error"
                                );
                            }
                            (w, g) => panic!("{da:?} {op:?} {db:?}: {w:?} vs {g:?}"),
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 500_000, "grid shrank: {checked}");
}

/// Bit equality, so `-0.0` and NaN payloads are compared as numpy's storage
/// would compare them rather than by `==`.
fn same_bits(a: Scalar, b: Scalar) -> bool {
    match (a, b) {
        (Scalar::Float(x), Scalar::Float(y)) => x.to_bits() == y.to_bits(),
        (Scalar::Complex(x), Scalar::Complex(y)) => {
            x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()
        }
        _ => a == b,
    }
}
