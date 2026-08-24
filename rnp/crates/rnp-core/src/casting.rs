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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TypeArg {
    Concrete(DType),
    /// A Python `bool`/`int`/`float`/`complex` literal.
    Weak(WeakKind),
    /// A Python `int`, carrying the dtype `np.array(v)` would give it.
    ///
    /// It promotes exactly like `Weak(Int)` against anything else, but numpy
    /// short-circuits a *single* argument straight to the array's own dtype,
    /// which for an integer is still value-based. Probed:
    /// `np.result_type(2**63)` is `uint64` and `np.result_type(2**100)` is
    /// `object`, yet `np.result_type(np.int8, 2**100)` is `int8`.
    WeakInt(DType),
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

    #[allow(dead_code)]
    fn of(d: DType) -> WeakKind {
        match d.category() {
            Kind::Bool => WeakKind::Bool,
            Kind::Int | Kind::Uint => WeakKind::Int,
            Kind::Float => WeakKind::Float,
            _ => WeakKind::Complex,
        }
    }
}

// ---------------------------------------------------------------------------
// numpy's promotion of a *sequence* of DTypes
// ---------------------------------------------------------------------------
//
// Promotion is neither associative nor commutative in numpy (unsigned and
// signed integers see to that), so `np.result_type(a, b, c)` is deliberately
// *not* `promote(promote(a, b), c)`. Probed from numpy 2.5.2:
//
//     result_type(uint8,  int8,   float16) -> float16   (left fold: float32)
//     result_type(int8,   uint16, float16) -> float32   (left fold: float64)
//     result_type(int16,  uint16, float32) -> float32   (left fold: float64)
//
// The algorithm below is `PyArray_PromoteDTypeSequence` from
// `upstream/numpy/_core/src/multiarray/common_dtype.c`, transcribed. Its two
// moving parts are:
//
//   * `common_dtype(a, b)` is one-sided and may answer "not implemented":
//     `default_builtin_common_dtype` defers whenever `b`'s *type number* is
//     larger than `a`'s (and `float16`'s type number, 23, is the largest of
//     all, which is exactly why it wins above).
//   * the reduction first sorts the "most knowledgeable" DType to the front
//     by swapping on every deferral, then promotes every remaining
//     participant against *that* one rather than against the running result.
//
// A weak Python scalar takes part as one of numpy's abstract DTypes
// (`PyLongDType` / `PyFloatDType` / `PyComplexDType`), whose one-sided
// `common_dtype` rules are transcribed from `abstractdtypes.c`.

/// One participant of the promotion: a real dtype, or the abstract dtype of a
/// weak Python scalar.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Part {
    Concrete(DType),
    /// `PyLongDType` / `PyFloatDType` / `PyComplexDType`. A Python `bool` is
    /// *not* abstract in numpy -- it coerces to `np.bool_` -- so `WeakKind`
    /// only reaches here as `Int`, `Float` or `Complex`.
    Abstract(WeakKind),
}

impl Part {
    /// The concrete dtype an abstract participant falls back to when it is
    /// the final answer (numpy's `default_descr`).
    fn resolve(self) -> DType {
        match self {
            Part::Concrete(d) => d,
            Part::Abstract(k) => k.default_dtype(),
        }
    }
}

/// numpy's one-sided `NPY_DT_CALL_common_dtype(a, b)`.
///
/// `None` is numpy's `NotImplemented`: "`a` does not know how to promote with
/// `b`; ask `b`".
fn one_sided_common(a: Part, b: Part) -> Option<Part> {
    use Part::*;
    match (a, b) {
        // `object_common_dtype`: object swallows everything.
        (Concrete(DType::Object), _) => Some(Concrete(DType::Object)),
        // `default_builtin_common_dtype`, the abstract half.
        (Concrete(c), Abstract(k)) => match k {
            WeakKind::Complex => match c {
                d if d.is_complex() => Some(Concrete(d)),
                DType::F16 | DType::F32 => Some(Concrete(DType::C64)),
                DType::F64 => Some(Concrete(DType::C128)),
                _ => None,
            },
            WeakKind::Float => {
                if c.is_complex() || c.is_float() {
                    Some(Concrete(c))
                } else {
                    None
                }
            }
            WeakKind::Int => {
                if c.is_complex()
                    || c.is_float()
                    || c.is_integer()
                    || matches!(c, DType::TimeDelta(_))
                {
                    Some(Concrete(c))
                } else {
                    None
                }
            }
            // Never constructed; a Python bool is concrete `np.bool_`.
            WeakKind::Bool => Some(Concrete(c)),
        },
        // `default_builtin_common_dtype`, the legacy half: defer to whoever
        // has the larger type number.
        (Concrete(c), Concrete(o)) => {
            if o.num() > c.num() {
                None
            } else if o == DType::Object {
                Some(Concrete(DType::Object))
            } else {
                Some(Concrete(crate::dtype::promote(c, o)))
            }
        }
        // `int_common_dtype`: a Python int knows only how to lift a bool to
        // the default integer; everything else is the other side's problem.
        (Abstract(WeakKind::Int), Concrete(DType::Bool)) => Some(Concrete(DType::I64)),
        (Abstract(WeakKind::Int), _) => None,
        // `float_common_dtype`.
        (Abstract(WeakKind::Float), Concrete(c)) => {
            if c == DType::Bool || c.is_integer() {
                Some(Concrete(DType::F64))
            } else {
                None
            }
        }
        (Abstract(WeakKind::Float), Abstract(WeakKind::Int)) => Some(Abstract(WeakKind::Float)),
        (Abstract(WeakKind::Float), _) => None,
        // `complex_common_dtype`.
        (Abstract(WeakKind::Complex), Concrete(c)) => {
            if c == DType::Bool || c.is_integer() {
                Some(Concrete(DType::C128))
            } else {
                None
            }
        }
        (Abstract(WeakKind::Complex), Abstract(WeakKind::Int | WeakKind::Float)) => {
            Some(Abstract(WeakKind::Complex))
        }
        (Abstract(WeakKind::Complex), _) => None,
        (Abstract(WeakKind::Bool), _) => None,
    }
}

/// numpy's symmetric `PyArray_CommonDType`: try both directions.
fn two_sided_common(a: Part, b: Part) -> Option<Part> {
    if a == b {
        return Some(a);
    }
    one_sided_common(a, b).or_else(|| one_sided_common(b, a))
}

/// `reduce_dtypes_to_most_knowledgeable`: partially sort `parts[..length]` so
/// that `parts[0]` is the participant every other one defers to, clearing the
/// entries that provably cannot influence the answer. Returns the last
/// pairwise result (`None` for numpy's `NotImplemented`).
fn reduce_to_most_knowledgeable(parts: &mut [Option<Part>], length: usize) -> Option<Part> {
    debug_assert!(length >= 2);
    let half = length / 2;
    let mut res = None;
    for low in 0..half {
        let high = length - 1 - low;
        // Entries are only ever cleared at an index at or above
        // `length - half`, which is past the prefix any recursive call looks
        // at, so both ends are still present here.
        let (a, b) = (parts[low].unwrap(), parts[high].unwrap());
        res = if a == b { Some(a) } else { one_sided_common(a, b) };
        match res {
            // "Guess at the other being more knowledgeable."
            None => parts.swap(low, high),
            // `parts[high]` cannot influence the result any more.
            Some(p) if p == a => parts[high] = None,
            Some(_) => {}
        }
    }
    if length == 2 {
        return res;
    }
    reduce_to_most_knowledgeable(parts, length - half)
}

/// `PyArray_PromoteDTypeSequence`. `None` is numpy's `DTypePromotionError`.
fn promote_sequence(input: &[Part]) -> Option<Part> {
    if input.len() == 1 {
        return Some(input[0]);
    }
    let n = input.len();
    let mut parts: Vec<Option<Part>> = input.iter().map(|&p| Some(p)).collect();
    let mut result = reduce_to_most_knowledgeable(&mut parts, n);
    let main = parts[0].unwrap();
    // When the reduction ended in "not implemented" its result is unusable and
    // `parts[0]` itself has not been folded in yet.
    let reduce_start = if result.is_some() { 2 } else { 1 };
    for slot in parts.iter().take(n).skip(reduce_start) {
        let Some(part) = *slot else { continue };
        let promotion = one_sided_common(main, part)?;
        result = Some(match result {
            None => promotion,
            Some(r) => two_sided_common(r, promotion)?,
        });
    }
    result
}

/// `np.result_type(*args)` under NEP 50.
///
/// Concrete dtypes promote among themselves as a whole sequence (see the note
/// above `Part`), while a Python scalar contributes only its *kind*, never its
/// value: `np.result_type(np.int8, 300)` is `int8`.
pub fn result_type(args: &[TypeArg]) -> Option<DType> {
    match args {
        [] => return None,
        // numpy short-circuits a single argument entirely: it is handed back
        // as-is, which is why `np.result_type(2**100)` is `object` even though
        // `np.result_type(np.int8, 2**100)` is `int8`.
        [TypeArg::Concrete(d)] => return Some(*d),
        [TypeArg::Weak(k)] => return Some(k.default_dtype()),
        [TypeArg::WeakInt(d)] => return Some(*d),
        _ => {}
    }
    let parts: Vec<Part> = args
        .iter()
        .map(|a| match *a {
            TypeArg::Concrete(d) => Part::Concrete(d),
            TypeArg::Weak(WeakKind::Bool) => Part::Concrete(DType::Bool),
            TypeArg::Weak(k) => Part::Abstract(k),
            TypeArg::WeakInt(_) => Part::Abstract(WeakKind::Int),
        })
        .collect();
    promote_sequence(&parts).map(Part::resolve)
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
    fn result_type_promotes_the_whole_sequence() {
        use TypeArg::Concrete as C;
        let d = |s: &str| Descr::parse(s).unwrap().dt;
        // Every one of these differs from `promote(promote(a, b), c)`, and
        // every one was probed from numpy 2.5.2. The left fold is given in
        // the comment.
        let cases: &[(&[&str], &str, &str)] = &[
            (&["uint8", "int8", "float16"], "float16", "float32"),
            (&["int8", "uint16", "float16"], "float32", "float64"),
            (&["uint16", "int8", "float16"], "float32", "float64"),
            (&["int16", "uint16", "float32"], "float32", "float64"),
            (&["int32", "uint32", "float16"], "float64", "float64"),
            (&["float16", "int64", "uint64"], "float64", "float64"),
            (&["int8", "uint8", "int8"], "int16", "int16"),
        ];
        for (args, want, left_fold) in cases {
            let parts: Vec<TypeArg> = args.iter().map(|s| C(d(s))).collect();
            assert_eq!(
                result_type(&parts),
                Some(d(want)),
                "result_type{args:?}"
            );
            let mut fold = d(args[0]);
            for a in &args[1..] {
                fold = crate::dtype::promote(fold, d(a));
            }
            assert_eq!(fold, d(left_fold), "left fold of {args:?} moved");
        }
    }

    #[test]
    fn result_type_is_order_independent_for_the_probed_sets() {
        use TypeArg::Concrete as C;
        let d = |s: &str| Descr::parse(s).unwrap().dt;
        // numpy guarantees a stable answer whatever the argument order; the
        // reduction is what buys that, so assert it directly.
        for set in [
            ["uint8", "int8", "float16"],
            ["int8", "uint16", "float16"],
            ["int16", "uint16", "float32"],
        ] {
            let want = result_type(&set.map(|s| C(d(s))));
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                let mut p = set;
                p.swap(i, j);
                assert_eq!(result_type(&p.map(|s| C(d(s)))), want, "{p:?}");
            }
        }
    }

    #[test]
    fn result_type_weak_int_is_value_based_only_when_alone() {
        use TypeArg::*;
        // Probed: np.result_type(2**63) is uint64, np.result_type(2**100) is
        // object, but as soon as anything else joins in they are plain weak
        // integers again.
        assert_eq!(result_type(&[WeakInt(DType::U64)]), Some(DType::U64));
        assert_eq!(result_type(&[WeakInt(DType::Object)]), Some(DType::Object));
        assert_eq!(
            result_type(&[Concrete(DType::I8), WeakInt(DType::Object)]),
            Some(DType::I8)
        );
        assert_eq!(
            result_type(&[WeakInt(DType::Object), WeakInt(DType::U64)]),
            Some(DType::I64)
        );
    }

    #[test]
    fn result_type_object_swallows_everything() {
        use TypeArg::*;
        for d in ALL_DTYPES {
            assert_eq!(
                result_type(&[Concrete(d), Concrete(DType::Object)]),
                Some(DType::Object),
                "{d} + object"
            );
            assert_eq!(
                result_type(&[Concrete(DType::Object), Concrete(d)]),
                Some(DType::Object),
                "object + {d}"
            );
        }
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
