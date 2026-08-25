//! `np.can_cast`, `np.min_scalar_type` and the NEP 50 `np.result_type`
//! machinery.
//!
//! The numeric part of the casting lattice is a table generated straight from
//! real numpy (`casting_table.inc`); the flexible (`S`/`U`/`V`) rules were
//! probed from numpy and are asserted against it in `harness/dev_check.py`.

use crate::descr::Descr;
use crate::dtype::{DType, Kind};
use crate::element::Scalar;

/// Why a concrete value cannot be preserved by a `same_value` cast.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SameValueError {
    /// The real value changes through the destination representation.
    Changed,
    /// A complex-to-real cast would discard a non-zero imaginary component.
    Imaginary,
}

/// Check NumPy's value-sensitive `casting='same_value'` rule for one element.
///
/// This is the policy in `lowlevel_strided_loops.c.src`: casts to bool are
/// accepted unconditionally, non-finite values survive only through floating
/// components, integer destinations are range checked, and every remaining
/// conversion must round-trip exactly.
pub fn same_value_preserved(
    value: Scalar,
    from: DType,
    to: DType,
) -> std::result::Result<(), SameValueError> {
    if to == DType::Bool {
        // NumPy deliberately exposes bool as a non-zero test, even in
        // same-value mode (the generated C loop omits the check for bool2).
        return Ok(());
    }

    let from_component = component_dtype(from);
    let to_component = component_dtype(to);
    match value {
        Scalar::Complex(value) => {
            if real_value_preserved(value.re, from_component, to_component).is_err() {
                return Err(SameValueError::Changed);
            }
            if to.is_complex() {
                real_value_preserved(value.im, from_component, to_component)
            } else if value.im == 0.0 {
                Ok(())
            } else {
                Err(SameValueError::Imaginary)
            }
        }
        Scalar::Float(value) => real_value_preserved(value, from_component, to_component),
        Scalar::Int(value) => integer_value_preserved(value as i128, to_component),
        Scalar::Uint(value) => unsigned_value_preserved(value as u128, to_component),
        Scalar::Bool(_) => Ok(()),
    }
}

fn component_dtype(dtype: DType) -> DType {
    match dtype {
        DType::C64 => DType::F32,
        DType::C128 => DType::F64,
        other => other,
    }
}

fn integer_value_preserved(
    value: i128,
    to: DType,
) -> std::result::Result<(), SameValueError> {
    if let Some((min, max)) = signed_bounds(to) {
        return if (min..=max).contains(&value) {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    if let Some(max) = unsigned_max(to) {
        return if value >= 0 && (value as u128) <= max {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    if to.is_float() {
        let Scalar::Float(cast) = Scalar::Int(value as i64).cast(to) else {
            unreachable!()
        };
        return if cast.is_finite() && cast as i128 == value {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    Err(SameValueError::Changed)
}

fn unsigned_value_preserved(
    value: u128,
    to: DType,
) -> std::result::Result<(), SameValueError> {
    if let Some((_, max)) = signed_bounds(to) {
        return if value <= max as u128 {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    if let Some(max) = unsigned_max(to) {
        return if value <= max {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    if to.is_float() {
        let Scalar::Float(cast) = Scalar::Uint(value as u64).cast(to) else {
            unreachable!()
        };
        return if cast.is_finite() && cast as u128 == value {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    Err(SameValueError::Changed)
}

fn real_value_preserved(
    value: f64,
    from: DType,
    to: DType,
) -> std::result::Result<(), SameValueError> {
    if !value.is_finite() {
        return if to.is_float() {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    if let Some((min, max)) = signed_bounds(to) {
        // The upper bound is exclusive because `i64::MAX as f64` rounds to
        // 2**63. Expressing the range as powers of two avoids that trap.
        let bits = to.itemsize() * 8;
        let lower = -(2.0f64).powi(bits as i32 - 1);
        let upper = (2.0f64).powi(bits as i32 - 1);
        if value < lower || value >= upper {
            return Err(SameValueError::Changed);
        }
        let cast = value as i128;
        return if (min..=max).contains(&cast) && cast as f64 == value {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    if let Some(max) = unsigned_max(to) {
        let upper = (2.0f64).powi((to.itemsize() * 8) as i32);
        if value < 0.0 || value >= upper {
            return Err(SameValueError::Changed);
        }
        let cast = value as u128;
        return if cast <= max && cast as f64 == value {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    if to.is_float() {
        let Scalar::Float(cast) = Scalar::Float(value).cast(to) else {
            unreachable!()
        };
        let Scalar::Float(round_trip) = Scalar::Float(cast).cast(from) else {
            unreachable!()
        };
        return if round_trip == value {
            Ok(())
        } else {
            Err(SameValueError::Changed)
        };
    }
    Err(SameValueError::Changed)
}

fn signed_bounds(dtype: DType) -> Option<(i128, i128)> {
    let bits = match dtype {
        DType::I8 => 8,
        DType::I16 => 16,
        DType::I32 => 32,
        DType::I64 => 64,
        _ => return None,
    };
    let upper = 1i128 << (bits - 1);
    Some((-upper, upper - 1))
}

fn unsigned_max(dtype: DType) -> Option<u128> {
    let bits = match dtype {
        DType::U8 => 8,
        DType::U16 => 16,
        DType::U32 => 32,
        DType::U64 => 64,
        _ => return None,
    };
    Some((1u128 << bits) - 1)
}

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
pub fn string_length(d: DType) -> Option<u32> {
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
    if let Some(result) = compound_can_cast(from, to, casting) {
        return result;
    }
    match casting {
        Casting::No => from == to || unsized_target,
        Casting::Equiv => from.dt == to.dt || unsized_target,
        Casting::Unsafe => true,
        Casting::Safe => cast_ok(from.dt, to.dt, false),
        Casting::SameKind => cast_ok(from.dt, to.dt, true),
    }
}

/// Descriptor-aware half of `can_cast`. Structured dtypes are not ordinary
/// VOID blobs: fields are checked by position, field counts must match even
/// for `unsafe`, and subarray shapes participate in the cast safety.
fn compound_can_cast(from: Descr, to: Descr, casting: Casting) -> Option<bool> {
    match (from.struct_def(), to.struct_def()) {
        (Some(src), Some(dst)) => {
            if src.fields.len() != dst.fields.len() {
                return Some(false);
            }
            return Some(match casting {
                Casting::No => from == to,
                Casting::Equiv => from.dt == to.dt,
                Casting::Safe | Casting::SameKind | Casting::Unsafe => src
                    .fields
                    .iter()
                    .zip(dst.fields.iter())
                    .all(|(a, b)| can_cast(a.descr, b.descr, casting)),
            });
        }
        (Some(src), None) => {
            return Some(
                casting == Casting::Unsafe
                    && src.fields.len() == 1
                    && can_cast(src.fields[0].descr, to, Casting::Unsafe),
            );
        }
        (None, Some(dst)) => {
            return Some(
                casting == Casting::Unsafe
                    && dst
                        .fields
                        .iter()
                        .all(|field| can_cast(from, field.descr, Casting::Unsafe)),
            );
        }
        (None, None) => {}
    }

    match (from.subarray_def(), to.subarray_def()) {
        (Some(src), Some(dst)) => Some(if src.shape == dst.shape {
            can_cast(src.base, dst.base, casting)
        } else {
            casting == Casting::Unsafe
        }),
        (Some(src), None) => Some(
            casting == Casting::Unsafe && can_cast(src.base, to, Casting::Unsafe),
        ),
        (None, Some(dst)) => Some(
            casting == Casting::Unsafe && can_cast(from, dst.base, Casting::Unsafe),
        ),
        (None, None) => None,
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
    fn same_value_checks_bounds_round_trip_and_complex_parts() {
        use num_complex::Complex;
        use SameValueError::*;

        assert_eq!(same_value_preserved(Scalar::Int(1), DType::I64, DType::U8), Ok(()));
        assert_eq!(same_value_preserved(Scalar::Int(-1), DType::I64, DType::U64), Err(Changed));
        assert_eq!(same_value_preserved(Scalar::Uint(u64::MAX), DType::U64, DType::F64), Err(Changed));
        assert_eq!(same_value_preserved(Scalar::Int(1 << 24), DType::I64, DType::F32), Ok(()));
        assert_eq!(same_value_preserved(Scalar::Int((1 << 24) + 1), DType::I64, DType::F32), Err(Changed));
        assert_eq!(same_value_preserved(Scalar::Float(10.0), DType::F64, DType::I8), Ok(()));
        assert_eq!(same_value_preserved(Scalar::Float(10.5), DType::F64, DType::I8), Err(Changed));
        assert_eq!(same_value_preserved(Scalar::Float(f64::NAN), DType::F64, DType::F32), Ok(()));
        assert_eq!(same_value_preserved(Scalar::Float(f64::INFINITY), DType::F64, DType::I64), Err(Changed));
        assert_eq!(same_value_preserved(Scalar::Complex(Complex::new(1.0, 0.0)), DType::C128, DType::F64), Ok(()));
        assert_eq!(same_value_preserved(Scalar::Complex(Complex::new(1.0, 2.0)), DType::C128, DType::F64), Err(Imaginary));
        assert_eq!(same_value_preserved(Scalar::Complex(Complex::new(1.0, 2.0)), DType::C128, DType::Bool), Ok(()));
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
