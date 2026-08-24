//! NumPy-compatible dtype descriptors and NEP 50 type promotion.
//!
//! Every fact encoded here (kind chars, type numbers, `str` codes, and the
//! full 14x14 promotion table) was probed directly from real numpy 2.5.2 in
//! `.venv`; see the tests at the bottom of this file for the verbatim table.
//!
//! `DType` is the *storage* type of an array element. It stays `Copy` even
//! for the compound kinds: flexible types carry their size inline, and
//! structured / subarray types carry an id into the interning registry in
//! [`crate::descr`], which hands out one id per structurally distinct
//! definition (so `==` on ids is structural equality, as numpy requires).
//!
//! Byte order is *not* part of `DType`; it lives in [`crate::descr::Descr`],
//! which is what the Python-facing `np.dtype` object is built from.

use std::fmt;

use crate::descr::registry;

/// The dtype of one array element.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    C64,
    C128,
    /// `S<n>` — n bytes of zero-padded bytes. `n == 0` is numpy's unsized `S`.
    Bytes(u32),
    /// `U<n>` — n UCS4 code points, so `4 * n` bytes.
    Str(u32),
    /// NEP 55 variable-width UTF-8 strings. The payload identifies the
    /// Python-owned StringDType parameters (`na_object` and `coerce`). The
    /// default, parameter-less descriptor always uses id 0.
    String(u32),
    /// `V<n>` — n raw bytes with no interpretation.
    Void(u32),
    /// A structured dtype; the payload indexes the struct registry.
    Struct(u32),
    /// A subarray dtype (`('f4', (2, 2))`); indexes the subarray registry.
    SubArray(u32),
    /// `M8[unit]`. The payload is a packed
    /// [`crate::datetime::DtMeta`] (base unit in the low 8 bits, multiplier
    /// above); the generic `M8` is `base == UNIT_GENERIC`.
    DateTime(u64),
    /// `m8[unit]`; see [`DType::DateTime`].
    TimeDelta(u64),
    /// `object` — arrays of it store 8-byte handles into the interning slab
    /// on the Python side (see `rnp-python/src/objects.rs`).
    Object,
}

/// Broad category of a dtype, used by the promotion lattice.
#[derive(Copy, Clone, PartialEq, Eq, Debug, PartialOrd, Ord, Hash)]
pub enum Kind {
    Bool,
    Int,
    Uint,
    Float,
    Complex,
    /// `S`
    Bytes,
    /// `U`
    Str,
    /// `T` — variable-width UTF-8 StringDType.
    String,
    /// `V`, including structured and subarray dtypes.
    Void,
    /// `M`
    DateTime,
    /// `m`
    TimeDelta,
    /// `O`
    Object,
}

/// The generic (unit-less) `M8`/`m8` storage dtype.
pub fn generic_datetime() -> DType {
    DType::DateTime(crate::datetime::DtMeta::GENERIC.pack())
}

/// The generic (unit-less) `m8` storage dtype.
pub fn generic_timedelta() -> DType {
    DType::TimeDelta(crate::datetime::DtMeta::GENERIC.pack())
}

/// The numeric dtypes, in numpy's own ordering.
pub const ALL_DTYPES: [DType; 14] = [
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

impl DType {
    /// Size of one element in bytes (numpy's `dtype.itemsize`).
    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool | DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 | DType::F16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 | DType::C64 => 8,
            DType::C128 | DType::String(_) => 16,
            DType::Bytes(n) | DType::Void(n) => n as usize,
            DType::Str(n) => 4 * n as usize,
            DType::DateTime(_) | DType::TimeDelta(_) => 8,
            DType::Struct(id) => registry::struct_def(id).itemsize,
            DType::SubArray(id) => {
                let d = registry::subarray_def(id);
                d.base.itemsize() * crate::array::shape_size(&d.shape)
            }
            DType::Object => 8,
        }
    }

    /// Alignment requirement in bytes. Complex aligns like its component.
    pub fn alignment(self) -> usize {
        match self {
            DType::C64 => 4,
            DType::C128 => 8,
            DType::Bytes(_) | DType::Void(_) => 1,
            DType::Str(_) => 4,
            DType::String(_) => 8,
            DType::Struct(id) => registry::struct_def(id).alignment,
            DType::SubArray(id) => registry::subarray_def(id).base.alignment(),
            DType::Object | DType::DateTime(_) | DType::TimeDelta(_) => 8,
            other => other.itemsize(),
        }
    }

    pub fn category(self) -> Kind {
        match self {
            DType::Bool => Kind::Bool,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => Kind::Int,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => Kind::Uint,
            DType::F16 | DType::F32 | DType::F64 => Kind::Float,
            DType::C64 | DType::C128 => Kind::Complex,
            DType::Bytes(_) => Kind::Bytes,
            DType::Str(_) => Kind::Str,
            DType::String(_) => Kind::String,
            DType::Void(_) | DType::Struct(_) | DType::SubArray(_) => Kind::Void,
            DType::DateTime(_) => Kind::DateTime,
            DType::TimeDelta(_) => Kind::TimeDelta,
            DType::Object => Kind::Object,
        }
    }

    /// numpy's `dtype.kind` character.
    pub fn kind(self) -> char {
        match self.category() {
            Kind::Bool => 'b',
            Kind::Int => 'i',
            Kind::Uint => 'u',
            Kind::Float => 'f',
            Kind::Complex => 'c',
            Kind::Bytes => 'S',
            Kind::Str => 'U',
            Kind::String => 'T',
            Kind::Void => 'V',
            Kind::DateTime => 'M',
            Kind::TimeDelta => 'm',
            Kind::Object => 'O',
        }
    }

    /// numpy's `dtype.char` (the single-character type code).
    pub fn char_code(self) -> char {
        match self {
            DType::Bool => '?',
            DType::I8 => 'b',
            DType::I16 => 'h',
            DType::I32 => 'i',
            DType::I64 => 'l',
            DType::U8 => 'B',
            DType::U16 => 'H',
            DType::U32 => 'I',
            DType::U64 => 'L',
            DType::F16 => 'e',
            DType::F32 => 'f',
            DType::F64 => 'd',
            DType::C64 => 'F',
            DType::C128 => 'D',
            DType::Bytes(_) => 'S',
            DType::Str(_) => 'U',
            DType::String(_) => 'T',
            DType::Void(_) | DType::Struct(_) | DType::SubArray(_) => 'V',
            DType::DateTime(_) => 'M',
            DType::TimeDelta(_) => 'm',
            DType::Object => 'O',
        }
    }

    /// The `&'static str` name for the numeric dtypes; `None` otherwise.
    pub fn numeric_name(self) -> Option<&'static str> {
        Some(match self {
            DType::Bool => "bool",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::F16 => "float16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::C64 => "complex64",
            DType::C128 => "complex128",
            _ => return None,
        })
    }

    /// numpy's `dtype.name`. Flexible dtypes get a bit-width suffix
    /// (`bytes40`, `str96`, `void128`), unsized ones just the stem.
    pub fn name(self) -> String {
        if let Some(n) = self.numeric_name() {
            return n.to_string();
        }
        if self == DType::Object {
            return "object".to_string();
        }
        if matches!(self, DType::String(_)) {
            return "StringDType128".to_string();
        }
        if let DType::DateTime(u) | DType::TimeDelta(u) = self {
            let stem = if matches!(self, DType::DateTime(_)) {
                "datetime64"
            } else {
                "timedelta64"
            };
            return format!("{stem}{}", crate::datetime::DtMeta::unpack(u).suffix());
        }
        let stem = match self.category() {
            Kind::Bytes => "bytes",
            Kind::Str => "str",
            _ => "void",
        };
        let bits = self.itemsize() * 8;
        if bits == 0 {
            stem.to_string()
        } else {
            format!("{}{}", stem, bits)
        }
    }

    /// numpy's `dtype.num` (the internal type number).
    pub fn num(self) -> i32 {
        match self {
            DType::Bool => 0,
            DType::I8 => 1,
            DType::U8 => 2,
            DType::I16 => 3,
            DType::U16 => 4,
            DType::I32 => 5,
            DType::U32 => 6,
            DType::I64 => 7,
            DType::U64 => 8,
            DType::F16 => 23,
            DType::F32 => 11,
            DType::F64 => 12,
            DType::C64 => 14,
            DType::C128 => 15,
            DType::Bytes(_) => 18,
            DType::Str(_) => 19,
            DType::String(_) => 2056,
            DType::Void(_) | DType::Struct(_) | DType::SubArray(_) => 20,
            DType::DateTime(_) => 21,
            DType::TimeDelta(_) => 22,
            DType::Object => 17,
        }
    }

    /// True for the 14 dtypes that have a Rust element type behind them.
    pub fn is_numeric(self) -> bool {
        self.numeric_name().is_some()
    }

    /// True when byte order is meaningful for this dtype (numpy's `|` vs
    /// `<`/`>`/`=` distinction).
    pub fn byteorder_matters(self) -> bool {
        match self {
            DType::Str(_) => true,
            DType::Bytes(_) | DType::String(_) | DType::Void(_) | DType::Struct(_) | DType::SubArray(_) => false,
            DType::Object => false,
            DType::DateTime(_) | DType::TimeDelta(_) => true,
            other => other.itemsize() > 1,
        }
    }

    /// The letter used in `dtype.str` (bool prints as `b`, not `?`).
    pub fn str_kind(self) -> char {
        if self.category() == Kind::Bool {
            'b'
        } else {
            self.kind()
        }
    }

    /// The size that appears in `dtype.str`: element count for `U`, bytes
    /// otherwise.
    pub fn str_size(self) -> usize {
        match self {
            DType::Str(n) => n as usize,
            DType::String(_) => self.itemsize(),
            other => other.itemsize(),
        }
    }

    /// The struct-module format character used by the buffer protocol.
    pub fn buffer_format(self) -> &'static str {
        match self {
            DType::Bool => "?",
            DType::I8 => "b",
            DType::I16 => "h",
            DType::I32 => "i",
            DType::I64 => "q",
            DType::U8 => "B",
            DType::U16 => "H",
            DType::U32 => "I",
            DType::U64 => "Q",
            // Python's struct module has no half; expose the raw bytes.
            DType::F16 => "e",
            DType::F32 => "f",
            DType::F64 => "d",
            DType::C64 => "Zf",
            DType::C128 => "Zd",
            _ => "B",
        }
    }

    pub fn is_bool(self) -> bool {
        self.category() == Kind::Bool
    }
    pub fn is_signed(self) -> bool {
        self.category() == Kind::Int
    }
    pub fn is_unsigned(self) -> bool {
        self.category() == Kind::Uint
    }
    pub fn is_integer(self) -> bool {
        matches!(self.category(), Kind::Int | Kind::Uint)
    }
    pub fn is_float(self) -> bool {
        self.category() == Kind::Float
    }
    pub fn is_complex(self) -> bool {
        self.category() == Kind::Complex
    }
    pub fn is_flexible(self) -> bool {
        matches!(self.category(), Kind::Bytes | Kind::Str | Kind::String | Kind::Void)
    }
    /// True for `M8`/`m8`: int64 storage plus unit metadata.
    pub fn is_datetime_like(self) -> bool {
        matches!(self, DType::DateTime(_) | DType::TimeDelta(_))
    }
    /// True only for `M8[...]`.
    pub fn is_datetime(self) -> bool {
        matches!(self, DType::DateTime(_))
    }
    /// True only for `m8[...]`.
    pub fn is_timedelta(self) -> bool {
        matches!(self, DType::TimeDelta(_))
    }
    /// `object` dtype: a descriptor with no storage behind it.
    pub fn is_object(self) -> bool {
        self == DType::Object
    }
    /// NEP 55 variable-width UTF-8 string dtype.
    pub fn is_string(self) -> bool {
        matches!(self, DType::String(_))
    }
    /// True for bool and all integer types (the "not inexact" set).
    pub fn is_exact(self) -> bool {
        matches!(self.category(), Kind::Bool | Kind::Int | Kind::Uint)
    }

    /// Size in bytes of the real component (for complex, half the itemsize).
    pub fn component_size(self) -> usize {
        match self {
            DType::C64 => 4,
            DType::C128 => 8,
            other => other.itemsize(),
        }
    }

    /// Parse a dtype spelling that carries no byte-order information beyond
    /// what [`crate::descr::Descr::parse`] would strip. Kept for the many
    /// call sites (and tests) that only care about the storage type.
    pub fn from_str(s: &str) -> Option<DType> {
        crate::descr::Descr::parse(s).map(|d| d.dt)
    }

    /// The single-character/short code names numpy accepts for the numeric
    /// dtypes, e.g. `"int64"`, `"l"`, `"i8"`, `"double"`.
    pub(crate) fn from_plain_name(s: &str) -> Option<DType> {
        Some(match s {
            "object" | "object_" | "O" | "O8" => DType::Object,
            "T" => DType::String(0),
            "bool" | "bool_" | "?" | "b1" => DType::Bool,
            "int8" | "byte" | "b" | "i1" => DType::I8,
            "int16" | "short" | "h" | "i2" => DType::I16,
            "int32" | "intc" | "i" | "i4" => DType::I32,
            "int64" | "int" | "int_" | "long" | "longlong" | "intp" | "l" | "q" | "p" | "n"
            | "i8" => DType::I64,
            "uint8" | "ubyte" | "B" | "u1" => DType::U8,
            "uint16" | "ushort" | "H" | "u2" => DType::U16,
            "uint32" | "uintc" | "I" | "u4" => DType::U32,
            "uint64" | "uint" | "ulong" | "ulonglong" | "uintp" | "L" | "Q" | "P" | "N"
            | "u8" => DType::U64,
            "float16" | "half" | "e" | "f2" => DType::F16,
            "float32" | "single" | "f" | "f4" => DType::F32,
            "float64" | "double" | "float" | "float_" | "d" | "f8" => DType::F64,
            "complex64" | "csingle" | "F" | "c8" => DType::C64,
            "complex128" | "cdouble" | "complex" | "complex_" | "D" | "c16" => DType::C128,
            _ => return None,
        })
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// NEP 50 / numpy 2.x `promote_types` for the numeric dtypes.
///
/// The rule set, derived from and verified against the full numpy table:
///  * `bool` is the identity element.
///  * int + int of the same signedness widens to the larger.
///  * signed + unsigned needs a signed type twice as wide as the unsigned
///    operand (`int8 + uint16 -> int32`); `uint64` cannot fit in any signed
///    type, so `int* + uint64 -> float64`.
///  * anything inexact promotes to a float/complex whose *component* is wide
///    enough to hold the other operand: a 1-byte integer fits in `float16`,
///    a 2-byte one in `float32`, anything wider needs `float64`. Complex-ness
///    is sticky.
pub fn promote(a: DType, b: DType) -> DType {
    use Kind::*;
    if a == b {
        return a;
    }
    // Flexible promotion is only defined for identical kinds; the widest
    // width wins. (numpy raises for e.g. int + str, which the callers turn
    // into a DTypePromotionError.)
    if a.is_flexible() || b.is_flexible() {
        return promote_flexible(a, b).unwrap_or(DType::Void(0));
    }
    if a.is_datetime_like() || b.is_datetime_like() {
        return promote_datetime(a, b).unwrap_or(DType::Void(0));
    }
    let (ka, kb) = (a.category(), b.category());
    if ka == Bool {
        return b;
    }
    if kb == Bool {
        return a;
    }

    if a.is_integer() && b.is_integer() {
        if ka == kb {
            let size = a.itemsize().max(b.itemsize());
            return int_of_size(size, ka == Int);
        }
        // Mixed signedness.
        let (signed_size, unsigned_size) = if ka == Int {
            (a.itemsize(), b.itemsize())
        } else {
            (b.itemsize(), a.itemsize())
        };
        if unsigned_size >= 8 {
            // uint64 has no signed superset.
            return DType::F64;
        }
        let need = signed_size.max(unsigned_size * 2);
        return int_of_size(need, true);
    }

    // At least one inexact operand.
    let mut component = 0usize;
    let mut complex = false;
    for d in [a, b] {
        match d.category() {
            Complex => {
                complex = true;
                component = component.max(d.component_size());
            }
            Float => component = component.max(d.itemsize()),
            // The smallest float that represents every value of the integer
            // type exactly: 1 byte -> float16, 2 bytes -> float32, else f64.
            _ => {
                component = component.max(match d.itemsize() {
                    1 => 2,
                    2 => 4,
                    _ => 8,
                })
            }
        }
    }
    match (complex, component) {
        (false, 2) => DType::F16,
        (false, 4) => DType::F32,
        (false, _) => DType::F64,
        // There is no complex32, so a half component still needs complex64.
        (true, 2) | (true, 4) => DType::C64,
        (true, _) => DType::C128,
    }
}

/// `promote_types` where at least one side is flexible. `None` means numpy
/// raises (no common type exists).
pub fn promote_flexible(a: DType, b: DType) -> Option<DType> {
    use DType::*;
    match (a, b) {
        (Bytes(n), Bytes(m)) => Some(Bytes(n.max(m))),
        (Str(n), Str(m)) => Some(Str(n.max(m))),
        (String(a), String(b)) if a == b => Some(String(a)),
        (Void(n), Void(m)) if n == m => Some(Void(n)),
        // numpy promotes bytes -> str by widening to the code-point count.
        (Bytes(n), Str(m)) | (Str(m), Bytes(n)) => Some(Str(n.max(m))),
        _ => None,
    }
}

/// `promote_types` where at least one side is datetime-like. numpy allows
/// `m8 + integer` (the integer is read in the timedelta's own unit) but no
/// other cross-kind combination, and `M8`/`m8` mixes yield `M8`.
pub fn promote_datetime(a: DType, b: DType) -> Option<DType> {
    if a.is_datetime_like() && b.is_datetime_like() {
        return crate::datetime::promote_meta(a, b).ok();
    }
    // np.promote_types('m8[s]', 'int64') is timedelta64[s]; the datetime side
    // has no such rule.
    let (dt, other) = if a.is_datetime_like() { (a, b) } else { (b, a) };
    if dt.is_timedelta() && (other.is_integer() || other.is_bool()) {
        return Some(dt);
    }
    None
}

fn int_of_size(size: usize, signed: bool) -> DType {
    match (size, signed) {
        (1, true) => DType::I8,
        (2, true) => DType::I16,
        (s, true) if s <= 4 => DType::I32,
        (_, true) => DType::I64,
        (1, false) => DType::U8,
        (2, false) => DType::U16,
        (s, false) if s <= 4 => DType::U32,
        (_, false) => DType::U64,
    }
}

/// Result dtype of `np.divide` / true division: integral inputs go to float64.
pub fn promote_for_division(a: DType, b: DType) -> DType {
    let p = promote(a, b);
    if p.is_exact() {
        DType::F64
    } else {
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The complete `numpy.promote_types` table for the numeric dtypes,
    /// generated from real numpy 2.5.2 (`.venv`) and pasted verbatim.
    const PROMOTION_TABLE: &[(DType, DType, DType)] = &include!("promotion_table.inc");

    #[test]
    fn promotion_matches_numpy_exactly() {
        assert_eq!(PROMOTION_TABLE.len(), 196);
        for &(a, b, want) in PROMOTION_TABLE {
            let got = promote(a, b);
            assert_eq!(got, want, "promote({a}, {b}) = {got}, numpy says {want}");
        }
    }

    #[test]
    fn promotion_is_commutative_and_idempotent() {
        for a in ALL_DTYPES {
            assert_eq!(promote(a, a), a);
            for b in ALL_DTYPES {
                assert_eq!(promote(a, b), promote(b, a));
            }
        }
    }

    #[test]
    fn division_promotion() {
        assert_eq!(promote_for_division(DType::I32, DType::I32), DType::F64);
        assert_eq!(promote_for_division(DType::Bool, DType::Bool), DType::F64);
        assert_eq!(promote_for_division(DType::U64, DType::U64), DType::F64);
        assert_eq!(promote_for_division(DType::F32, DType::F32), DType::F32);
        assert_eq!(promote_for_division(DType::I16, DType::F32), DType::F32);
        assert_eq!(promote_for_division(DType::C64, DType::C64), DType::C64);
    }

    #[test]
    fn dtype_metadata_matches_numpy() {
        // (name, num, kind, char, str, itemsize) probed from numpy 2.5.2.
        let expect: &[(DType, &str, i32, char, char, usize)] = &[
            (DType::Bool, "bool", 0, 'b', '?', 1),
            (DType::I8, "int8", 1, 'i', 'b', 1),
            (DType::I16, "int16", 3, 'i', 'h', 2),
            (DType::I32, "int32", 5, 'i', 'i', 4),
            (DType::I64, "int64", 7, 'i', 'l', 8),
            (DType::U8, "uint8", 2, 'u', 'B', 1),
            (DType::U16, "uint16", 4, 'u', 'H', 2),
            (DType::U32, "uint32", 6, 'u', 'I', 4),
            (DType::U64, "uint64", 8, 'u', 'L', 8),
            (DType::F16, "float16", 23, 'f', 'e', 2),
            (DType::F32, "float32", 11, 'f', 'f', 4),
            (DType::F64, "float64", 12, 'f', 'd', 8),
            (DType::C64, "complex64", 14, 'c', 'F', 8),
            (DType::C128, "complex128", 15, 'c', 'D', 16),
        ];
        for &(d, name, num, kind, ch, isz) in expect {
            assert_eq!(d.name(), name);
            assert_eq!(d.num(), num);
            assert_eq!(d.kind(), kind);
            assert_eq!(d.char_code(), ch);
            assert_eq!(d.itemsize(), isz);
        }
    }

    #[test]
    fn flexible_metadata_matches_numpy() {
        // np.dtype('S5'): itemsize 5, kind S, num 18, name bytes40
        assert_eq!(DType::Bytes(5).itemsize(), 5);
        assert_eq!(DType::Bytes(5).name(), "bytes40");
        assert_eq!(DType::Bytes(0).name(), "bytes");
        assert_eq!(DType::Bytes(5).num(), 18);
        assert_eq!(DType::Bytes(5).alignment(), 1);
        // np.dtype('U3'): itemsize 12, kind U, num 19, name str96
        assert_eq!(DType::Str(3).itemsize(), 12);
        assert_eq!(DType::Str(3).name(), "str96");
        assert_eq!(DType::Str(3).num(), 19);
        assert_eq!(DType::Str(3).alignment(), 4);
        // np.dtype('V10'): itemsize 10, kind V, num 20, name void80
        assert_eq!(DType::Void(10).itemsize(), 10);
        assert_eq!(DType::Void(10).name(), "void80");
        assert_eq!(DType::Void(10).num(), 20);
    }

    /// `DType` is copied into every loop prologue and compared on every
    /// operand, so its width is a performance fact worth pinning: 8 bytes of
    /// payload plus the discriminant.
    #[test]
    fn dtype_stays_small() {
        assert_eq!(std::mem::size_of::<DType>(), 16);
        assert!(std::mem::size_of::<crate::descr::Descr>() <= 24);
    }

    #[test]
    fn dtype_parsing() {
        assert_eq!(DType::from_str("int64"), Some(DType::I64));
        assert_eq!(DType::from_str("i8"), Some(DType::I64));
        assert_eq!(DType::from_str("<f8"), Some(DType::F64));
        assert_eq!(DType::from_str("float64"), Some(DType::F64));
        assert_eq!(DType::from_str("?"), Some(DType::Bool));
        assert_eq!(DType::from_str("|b1"), Some(DType::Bool));
        assert_eq!(DType::from_str("c16"), Some(DType::C128));
        assert_eq!(DType::from_str("uint8"), Some(DType::U8));
        assert_eq!(DType::from_str("=u2"), Some(DType::U16));
        assert_eq!(DType::from_str("e"), Some(DType::F16));
        assert_eq!(DType::from_str("nonsense"), None);
        // Round-trip every dtype through its own spellings.
        for d in ALL_DTYPES {
            assert_eq!(DType::from_str(&d.name()), Some(d));
            assert_eq!(DType::from_str(&d.char_code().to_string()), Some(d));
        }
    }
}
