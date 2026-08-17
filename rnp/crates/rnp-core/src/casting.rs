//! `np.can_cast`, `np.min_scalar_type` and the NEP 50 `np.result_type`
//! machinery.
//!
//! The numeric part of the casting lattice is a table generated straight from
//! real numpy (`casting_table.inc`); the flexible (`S`/`U`/`V`) rules were
//! probed from numpy and are asserted against it in `harness/dev_check.py`.

use crate::descr::Descr;
use crate::dtype::{DType, Kind};
use crate::element::Scalar;

/// numpy's `casting=` keyword.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Casting {
    No,
    Equiv,
    Safe,
    SameKind,
    Unsafe,
}

impl Casting {
    pub fn from_str(s: &str) -> Option<Casting> {
        Some(match s {
            "no" => Casting::No,
            "equiv" => Casting::Equiv,
            "safe" => Casting::Safe,
            "same_kind" => Casting::SameKind,
            "unsafe" => Casting::Unsafe,
            _ => return None,
        })
    }
}

/// `(from, to, safe, same_kind)` for every numeric pair, from real numpy.
const NUMERIC_CASTS: &[(DType, DType, bool, bool)] = &include!("casting_table.inc");

/// The smallest `S<n>` / `U<n>` each numeric dtype casts safely into.
const STRING_LENGTHS: &[(DType, u32)] = &include!("string_lengths.inc");

fn numeric_cast(from: DType, to: DType, same_kind: bool) -> bool {
    for &(a, b, safe, sk) in NUMERIC_CASTS {
        if a == from && b == to {
            return if same_kind { sk } else { safe };
        }
    }
    false
}

/// The number of characters numpy needs to render any value of `d`.
fn string_length(d: DType) -> Option<u32> {
    STRING_LENGTHS
        .iter()
        .find(|(a, _)| *a == d)
        .map(|&(_, n)| n)
}

/// The `safe`/`same_kind` half of `can_cast`, on storage types alone (byte
/// order does not affect castability in numpy).
fn cast_ok(from: DType, to: DType, same_kind: bool) -> bool {
    if from == to {
        return true;
    }
    if (from.is_datetime_like() || to.is_datetime_like())
        && to.category() != Kind::Void
    {
        return datetime_cast_ok(
            from,
            to,
            if same_kind {
                Casting::SameKind
            } else {
                Casting::Safe
            },
        );
    }
    if from.is_numeric() && to.is_numeric() {
        return numeric_cast(from, to, same_kind);
    }
    match (from.category(), to.category()) {
        // Anything -> void: fits when the target is unsized or at least as
        // wide. Structured/subarray targets only accept themselves.
        (_, Kind::Void) => match to {
            // void -> void always works; anything else must fit.
            DType::Void(n) => {
                same_kind && from.category() == Kind::Void
                    || n == 0
                    || n as usize >= from.itemsize()
            }
            _ => false,
        },
        // Flexible -> numeric is never safe, only `unsafe`.
        (Kind::Bytes | Kind::Str | Kind::Void, _) if to.is_numeric() => false,
        // Void -> S/U is never safe.
        (Kind::Void, _) => false,
        // str -> bytes loses information; numpy rejects it even same_kind.
        (Kind::Str, Kind::Bytes) => false,
        (Kind::Bytes, Kind::Bytes) | (Kind::Bytes, Kind::Str) | (Kind::Str, Kind::Str) => {
            if same_kind {
                return true;
            }
            let m = match from {
                DType::Bytes(m) | DType::Str(m) => m,
                _ => return false,
            };
            let n = match to {
                DType::Bytes(n) | DType::Str(n) => n,
                _ => return false,
            };
            n == 0 || n >= m
        }
        // Numeric -> S/U: any width works for same_kind; a safe cast needs
        // room for the widest rendering of the source dtype.
        (_, Kind::Bytes | Kind::Str) => {
            if same_kind {
                return true;
            }
            let n = match to {
                DType::Bytes(n) | DType::Str(n) => n,
                _ => return false,
            };
            if n == 0 {
                return true;
            }
            match string_length(from) {
                Some(need) => n >= need,
                None => false,
            }
        }
        _ => false,
    }
}

/// The datetime half of the casting lattice.
///
/// This is numpy's `time_to_time_resolve_descriptors`
/// (`multiarray/datetime.c`), not `can_cast_datetime64_metadata`: the
/// resolver is what `PyArray_CanCastTypeTo` actually consults, and it knows
/// about the "same duration spelled with a metric prefix" cases -- `M8[7s]`
/// and `M8[7000ms]` are *equivalent*, so the cast is safe in both directions
/// even though `ms` is the finer base.
fn datetime_cast_ok(from: DType, to: DType, casting: Casting) -> bool {
    use crate::datetime::{meta_of, metadata_divides, UNIT_M};
    match (from.is_datetime_like(), to.is_datetime_like()) {
        (true, true) => {
            if from.is_datetime() != to.is_datetime() {
                // datetime64 <-> timedelta64 is `unsafe` only.
                return false;
            }
            let (m1, m2) = (meta_of(from).unwrap(), meta_of(to).unwrap());
            let is_td = from.is_timedelta();
            // Equal metadata, or one of the 10^3 / 10^6 / 10^9 metric-prefix
            // pairs, which denote exactly the same duration.
            let step = m1.base as i32 - m2.base as i32;
            let prefix_equiv = m2.base >= crate::datetime::UNIT_S
                && (1..=3).contains(&step)
                && m2.num != 0
                && m1.num % m2.num == 0
                && (m1.num / m2.num) as u64 == 1000u64.pow(step as u32);
            if (m1.base == m2.base && m1.num == m2.num) || prefix_equiv {
                return true;
            }
            if m1.is_generic() {
                // A generic source is only ever NaT, so it fits anywhere.
                return true;
            }
            if m2.is_generic() {
                // Converting *to* generic is an error, not a cast.
                return false;
            }
            // A timedelta may not cross the nonlinear-unit barrier at all.
            if is_td && ((m1.base <= UNIT_M) != (m2.base <= UNIT_M)) {
                return false;
            }
            if m1.base <= m2.base && metadata_divides(m1, m2, is_td) {
                return true; // safe
            }
            casting == Casting::SameKind
        }
        // Probed: datetime64/timedelta64 -> S/U and -> any numeric dtype is
        // `unsafe` only, in both directions.
        (true, false) => false,
        (false, true) => {
            if !to.is_timedelta() {
                // Nothing casts into datetime64 short of `unsafe`.
                return false;
            }
            if casting == Casting::SameKind {
                return from.is_integer() || from.is_bool();
            }
            // A *safe* cast needs the integer to fit in the int64 storage,
            // which `uint64` does not (probed: can_cast('u8', 'm8') is False
            // while can_cast('u4', 'm8') is True).
            cast_ok(from, DType::I64, false)
        }
        (false, false) => unreachable!(),
    }
}

/// `np.can_cast(from, to, casting)`.
pub fn can_cast(from: Descr, to: Descr, casting: Casting) -> bool {
    // An unsized flexible target ('S', 'U', 'V') adapts to the source, so
    // even 'no' casting accepts it as long as the kinds agree.
    let unsized_target = matches!(
        to.dt,
        DType::Bytes(0) | DType::Str(0) | DType::Void(0)
    ) && from.dt.category() == to.dt.category();
    match casting {
        Casting::No => from == to || unsized_target,
        Casting::Equiv => from.dt == to.dt || unsized_target,
        Casting::Unsafe => true,
        Casting::Safe => cast_ok(from.dt, to.dt, false),
        Casting::SameKind => cast_ok(from.dt, to.dt, true),
    }
}

/// `np.min_scalar_type` for a Python/array scalar value.
///
/// Transcribed from `min_scalar_type_num` in
/// `upstream/numpy/_core/src/multiarray/convert_datatype.c`; the odd-looking
/// `65000` / `3.4e38` bounds and the `!isfinite -> half` special case are
/// numpy's, not ours.
pub fn min_scalar_type(value: Scalar) -> DType {
    match value {
        Scalar::Bool(_) => DType::Bool,
        Scalar::Uint(v) => min_unsigned(v),
        Scalar::Int(v) => {
            if v >= 0 {
                min_unsigned(v as u64)
            } else if v >= i8::MIN as i64 {
                DType::I8
            } else if v >= i16::MIN as i64 {
                DType::I16
            } else if v >= i32::MIN as i64 {
                DType::I32
            } else {
                DType::I64
            }
        }
        Scalar::Float(v) => {
            if (v > -65000.0 && v < 65000.0) || !v.is_finite() {
                DType::F16
            } else if v > -3.4e38 && v < 3.4e38 {
                DType::F32
            } else {
                DType::F64
            }
        }
        Scalar::Complex(c) => {
            if c.re > -3.4e38 && c.re < 3.4e38 && c.im > -3.4e38 && c.im < 3.4e38 {
                DType::C64
            } else {
                DType::C128
            }
        }
    }
}

fn min_unsigned(v: u64) -> DType {
    if v <= u8::MAX as u64 {
        DType::U8
    } else if v <= u16::MAX as u64 {
        DType::U16
    } else if v <= u32::MAX as u64 {
        DType::U32
    } else {
        DType::U64
    }
}

/// One argument to `np.result_type`: either a concrete dtype (from an array,
/// a dtype object or a numpy scalar) or a *weak* Python scalar, which under
/// NEP 50 contributes only its kind, never its value.
#[derive(Copy, Clone, Debug)]
pub enum TypeArg {
    Concrete(DType),
    /// A Python `bool`/`int`/`float`/`complex` literal.
    Weak(WeakKind),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum WeakKind {
    Bool,
    Int,
    Float,
    Complex,
}

impl WeakKind {
    /// The dtype a weak scalar falls back to when there is nothing concrete
    /// to attach to.
    pub fn default_dtype(self) -> DType {
        match self {
            WeakKind::Bool => DType::Bool,
            WeakKind::Int => DType::I64,
            WeakKind::Float => DType::F64,
            WeakKind::Complex => DType::C128,
        }
    }

    fn of(d: DType) -> WeakKind {
        match d.category() {
            Kind::Bool => WeakKind::Bool,
            Kind::Int | Kind::Uint => WeakKind::Int,
            Kind::Float => WeakKind::Float,
            _ => WeakKind::Complex,
        }
    }
}

/// `np.result_type(*args)` under NEP 50.
///
/// Concrete dtypes promote among themselves as usual. A Python scalar only
/// bumps the *kind* of the result (`int8` + `300` is still `int8`), and only
/// when its kind is higher than the concrete result's.
pub fn result_type(args: &[TypeArg]) -> Option<DType> {
    let mut concrete: Option<DType> = None;
    let mut weak: Option<WeakKind> = None;
    for a in args {
        match *a {
            TypeArg::Concrete(d) => {
                concrete = Some(match concrete {
                    None => d,
                    Some(p) => crate::dtype::promote(p, d),
                });
            }
            TypeArg::Weak(k) => {
                weak = Some(match weak {
                    None => k,
                    Some(p) => p.max(k),
                });
            }
        }
    }
    match (concrete, weak) {
        (None, None) => None,
        (None, Some(k)) => Some(k.default_dtype()),
        (Some(d), None) => Some(d),
        (Some(d), Some(k)) => {
            if k <= WeakKind::of(d) {
                Some(d)
            } else {
                // The scalar's kind wins, at the narrowest width that holds
                // the concrete operand: int8 + 1.0 -> float64 (numpy uses the
                // default precision for the kind), float32 + 1j -> complex64.
                Some(match k {
                    WeakKind::Bool => d,
                    WeakKind::Int => crate::dtype::promote(d, DType::I64),
                    WeakKind::Float => crate::dtype::promote(d, DType::F64),
                    WeakKind::Complex => match d {
                        DType::F32 | DType::F16 => DType::C64,
                        _ => crate::dtype::promote(d, DType::C128),
                    },
                })
            }
        }
    }
}

/// `np.common_type(*arrays)`: the inexact dtype every argument fits into.
/// Integers count as `float64`, and only float/complex results are legal.
pub fn common_type(dtypes: &[DType]) -> Option<DType> {
    let mut acc = DType::F16;
    let mut any = false;
    for &d in dtypes {
        any = true;
        let as_inexact = if d.is_exact() { DType::F64 } else { d };
        if !as_inexact.is_float() && !as_inexact.is_complex() {
            return None;
        }
        acc = crate::dtype::promote(acc, as_inexact);
    }
    if !any {
        return Some(DType::F64);
    }
    // numpy never returns float16 from common_type: an integer array maps to
    // float64 and a float16 array maps to float16 only if it is alone, which
    // the promotion above already handles.
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::ALL_DTYPES;

    #[test]
    fn numeric_can_cast_matches_numpy_table() {
        assert_eq!(NUMERIC_CASTS.len(), 196);
        for &(a, b, safe, same) in NUMERIC_CASTS {
            let (da, db) = (Descr::native(a), Descr::native(b));
            assert_eq!(can_cast(da, db, Casting::Safe), safe, "safe {a} -> {b}");
            assert_eq!(
                can_cast(da, db, Casting::SameKind),
                same,
                "same_kind {a} -> {b}"
            );
            assert!(can_cast(da, db, Casting::Unsafe));
            assert_eq!(can_cast(da, db, Casting::No), a == b);
            assert_eq!(can_cast(da, db, Casting::Equiv), a == b);
        }
    }

    #[test]
    fn byteorder_only_affects_no_casting() {
        let native = Descr::parse("i4").unwrap();
        let big = Descr::parse(">i4").unwrap();
        // Probed: [False, True, True, True, True] for >i4 -> <i4.
        assert!(!can_cast(big, native, Casting::No));
        assert!(can_cast(big, native, Casting::Equiv));
        assert!(can_cast(big, native, Casting::Safe));
        assert!(can_cast(big, native, Casting::SameKind));
        assert!(can_cast(big, native, Casting::Unsafe));
    }

    #[test]
    fn flexible_can_cast_matches_numpy() {
        let d = |s: &str| Descr::parse(s).unwrap();
        // Probed from numpy 2.5.2.
        assert!(can_cast(d("S3"), d("S5"), Casting::Safe));
        assert!(!can_cast(d("S5"), d("S3"), Casting::Safe));
        assert!(can_cast(d("S5"), d("S3"), Casting::SameKind));
        assert!(can_cast(d("S3"), d("U3"), Casting::Safe));
        assert!(!can_cast(d("U3"), d("S3"), Casting::Safe));
        assert!(!can_cast(d("U3"), d("S3"), Casting::SameKind));
        assert!(can_cast(d("i4"), d("S"), Casting::Safe));
        assert!(!can_cast(d("i4"), d("S3"), Casting::Safe));
        assert!(can_cast(d("i4"), d("S11"), Casting::Safe));
        assert!(can_cast(d("i4"), d("U"), Casting::Safe));
        assert!(!can_cast(d("S3"), d("i4"), Casting::Safe));
        assert!(can_cast(d("S3"), d("i4"), Casting::Unsafe));
        assert!(can_cast(d("S3"), d("V3"), Casting::Safe));
        assert!(!can_cast(d("S5"), d("V3"), Casting::Safe));
        assert!(can_cast(d("V3"), d("V0"), Casting::Safe));
        assert!(can_cast(d("V8"), d("V3"), Casting::SameKind));
        assert!(!can_cast(d("V8"), d("V3"), Casting::Safe));
        // An unsized target is accepted even by 'no'.
        assert!(can_cast(d("S3"), d("S"), Casting::No));
        assert!(can_cast(d("V8"), d("V"), Casting::Equiv));
        assert!(!can_cast(d("i4"), d("S"), Casting::No));
    }

    #[test]
    fn min_scalar_type_matches_numpy() {
        use num_complex::Complex;
        // Probed from numpy 2.5.2.
        assert_eq!(min_scalar_type(Scalar::Int(3)), DType::U8);
        assert_eq!(min_scalar_type(Scalar::Int(300)), DType::U16);
        assert_eq!(min_scalar_type(Scalar::Int(-3)), DType::I8);
        assert_eq!(min_scalar_type(Scalar::Uint(u64::MAX)), DType::U64);
        assert_eq!(min_scalar_type(Scalar::Bool(true)), DType::Bool);
        assert_eq!(min_scalar_type(Scalar::Float(3.0)), DType::F16);
        assert_eq!(min_scalar_type(Scalar::Float(0.1)), DType::F16);
        assert_eq!(min_scalar_type(Scalar::Float(65504.0)), DType::F32);
        assert_eq!(min_scalar_type(Scalar::Float(3.3e38)), DType::F32);
        assert_eq!(min_scalar_type(Scalar::Float(1e40)), DType::F64);
        assert_eq!(min_scalar_type(Scalar::Float(f64::NAN)), DType::F16);
        assert_eq!(min_scalar_type(Scalar::Float(f64::INFINITY)), DType::F16);
        assert_eq!(
            min_scalar_type(Scalar::Complex(Complex::new(3.0, 4.0))),
            DType::C64
        );
        assert_eq!(
            min_scalar_type(Scalar::Complex(Complex::new(1e300, 0.0))),
            DType::C128
        );
    }

    #[test]
    fn result_type_is_nep50_weak() {
        use TypeArg::*;
        // Probed: np.result_type(np.int8, 300) is int8 under NEP 50.
        assert_eq!(
            result_type(&[Concrete(DType::I8), Weak(WeakKind::Int)]),
            Some(DType::I8)
        );
        assert_eq!(
            result_type(&[Concrete(DType::I8), Weak(WeakKind::Float)]),
            Some(DType::F64)
        );
        assert_eq!(
            result_type(&[Concrete(DType::F32), Weak(WeakKind::Complex)]),
            Some(DType::C64)
        );
        assert_eq!(
            result_type(&[Weak(WeakKind::Int), Weak(WeakKind::Int)]),
            Some(DType::I64)
        );
        assert_eq!(
            result_type(&[Weak(WeakKind::Int), Weak(WeakKind::Float)]),
            Some(DType::F64)
        );
        assert_eq!(
            result_type(&[Concrete(DType::F32), Weak(WeakKind::Float)]),
            Some(DType::F32)
        );
    }

    #[test]
    fn safe_casting_agrees_with_promotion() {
        // numpy's definition: a safe cast is one that promotion already
        // reaches, for the numeric lattice.
        for a in ALL_DTYPES {
            for b in ALL_DTYPES {
                let want = crate::dtype::promote(a, b) == b;
                assert_eq!(
                    can_cast(Descr::native(a), Descr::native(b), Casting::Safe),
                    want,
                    "{a} -> {b}"
                );
            }
        }
    }

    #[test]
    fn common_type_matches_numpy() {
        // np.common_type(np.array([1,2],'i4')) -> float64
        assert_eq!(common_type(&[DType::I32]), Some(DType::F64));
        assert_eq!(common_type(&[DType::F32]), Some(DType::F32));
        assert_eq!(common_type(&[DType::F32, DType::F64]), Some(DType::F64));
        assert_eq!(common_type(&[DType::C64]), Some(DType::C64));
        assert_eq!(common_type(&[]), Some(DType::F64));
    }
}
