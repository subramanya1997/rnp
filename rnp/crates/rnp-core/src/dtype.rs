//! NumPy-compatible dtype descriptors and NEP 50 type promotion.
//!
//! Every fact encoded here (kind chars, type numbers, `str` codes, and the
//! full 13x13 promotion table) was probed directly from real numpy 2.5.2 in
//! `.venv`; see the tests at the bottom of this file for the verbatim table.

use std::fmt;

/// The 13 base dtypes supported at M0.
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
    F32,
    F64,
    C64,
    C128,
}

/// Broad category of a dtype, used by the promotion lattice.
#[derive(Copy, Clone, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Kind {
    Bool,
    Int,
    Uint,
    Float,
    Complex,
}

pub const ALL_DTYPES: [DType; 13] = [
    DType::Bool,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
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
            DType::I16 | DType::U16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 | DType::C64 => 8,
            DType::C128 => 16,
        }
    }

    /// Alignment requirement in bytes. Complex aligns like its component.
    pub fn alignment(self) -> usize {
        match self {
            DType::C64 => 4,
            DType::C128 => 8,
            other => other.itemsize(),
        }
    }

    pub fn category(self) -> Kind {
        match self {
            DType::Bool => Kind::Bool,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => Kind::Int,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => Kind::Uint,
            DType::F32 | DType::F64 => Kind::Float,
            DType::C64 | DType::C128 => Kind::Complex,
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
            DType::F32 => 'f',
            DType::F64 => 'd',
            DType::C64 => 'F',
            DType::C128 => 'D',
        }
    }

    /// numpy's `dtype.name`.
    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::C64 => "complex64",
            DType::C128 => "complex128",
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
            DType::F32 => 11,
            DType::F64 => 12,
            DType::C64 => 14,
            DType::C128 => 15,
        }
    }

    /// numpy's `dtype.byteorder`: `|` for single-byte types, `=` otherwise.
    pub fn byteorder(self) -> char {
        if self.itemsize() == 1 {
            '|'
        } else {
            '='
        }
    }

    /// numpy's `dtype.str`, e.g. `<i8`. Single-byte types use `|`.
    pub fn str_code(self) -> String {
        let prefix = if self.itemsize() == 1 { '|' } else { '<' };
        let kind = if self.category() == Kind::Bool {
            'b'
        } else {
            self.kind()
        };
        format!("{}{}{}", prefix, kind, self.itemsize())
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
            DType::F32 => "f",
            DType::F64 => "d",
            DType::C64 => "Zf",
            DType::C128 => "Zd",
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

    /// Parse the dtype spellings numpy accepts for these 13 types:
    /// names (`"int64"`), char codes (`"l"`, `"?"`), sized codes (`"i8"`),
    /// byte-order-prefixed codes (`"<f8"`, `"=i4"`, `"|b1"`), and the
    /// platform aliases (`"int"`, `"float"`, `"double"`, ...).
    pub fn from_str(s: &str) -> Option<DType> {
        let s = s.trim();
        // Long/alias names first.
        let by_name = match s {
            "bool" | "bool_" | "?" | "b1" | "|b1" | "<b1" | ">b1" | "=b1" => Some(DType::Bool),
            "int8" | "byte" | "b" | "i1" => Some(DType::I8),
            "int16" | "short" | "h" | "i2" => Some(DType::I16),
            "int32" | "intc" | "i" | "i4" => Some(DType::I32),
            "int64" | "int" | "int_" | "long" | "longlong" | "intp" | "l" | "q" | "p" | "i8" => {
                Some(DType::I64)
            }
            "uint8" | "ubyte" | "B" | "u1" => Some(DType::U8),
            "uint16" | "ushort" | "H" | "u2" => Some(DType::U16),
            "uint32" | "uintc" | "I" | "u4" => Some(DType::U32),
            "uint64" | "uint" | "ulong" | "ulonglong" | "uintp" | "L" | "Q" | "P" | "u8" => {
                Some(DType::U64)
            }
            "float32" | "single" | "f" | "f4" => Some(DType::F32),
            "float64" | "double" | "float" | "float_" | "d" | "f8" => Some(DType::F64),
            "complex64" | "csingle" | "F" | "c8" => Some(DType::C64),
            "complex128" | "cdouble" | "complex" | "complex_" | "D" | "c16" => Some(DType::C128),
            _ => None,
        };
        if by_name.is_some() {
            return by_name;
        }
        // Byte-order prefix + code, e.g. "<f8", ">i4", "=u2", "|i1".
        let mut chars = s.chars();
        match chars.next() {
            Some('<') | Some('>') | Some('=') | Some('|') => {
                let rest: String = chars.collect();
                // Only native / byte-order-agnostic layouts are supported at M0.
                if s.starts_with('>') && rest != "b1" && rest.len() > 1 && &rest[1..] != "1" {
                    return None;
                }
                DType::from_str(&rest)
            }
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// NEP 50 / numpy 2.x `promote_types` for the 13 base dtypes.
///
/// The rule set, derived from and verified against the full numpy table:
///  * `bool` is the identity element.
///  * int + int of the same signedness widens to the larger.
///  * signed + unsigned needs a signed type twice as wide as the unsigned
///    operand (`int8 + uint16 -> int32`); `uint64` cannot fit in any signed
///    type, so `int* + uint64 -> float64`.
///  * anything inexact promotes to a float/complex whose *component* is wide
///    enough: integers of <= 2 bytes fit in `float32`, wider ones need
///    `float64`. Complex-ness is sticky.
pub fn promote(a: DType, b: DType) -> DType {
    use Kind::*;
    if a == b {
        return a;
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
            // Integers of 1-2 bytes are exactly representable in float32.
            _ => component = component.max(if d.itemsize() <= 2 { 4 } else { 8 }),
        }
    }
    match (complex, component) {
        (false, 4) => DType::F32,
        (false, _) => DType::F64,
        (true, 4) => DType::C64,
        (true, _) => DType::C128,
    }
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

    /// The complete `numpy.promote_types` table for the 13 base dtypes,
    /// generated from real numpy 2.5.2 (`.venv`) and pasted verbatim.
    const PROMOTION_TABLE: &[(DType, DType, DType)] = &include!("promotion_table.inc");

    #[test]
    fn promotion_matches_numpy_exactly() {
        assert_eq!(PROMOTION_TABLE.len(), 169);
        for &(a, b, want) in PROMOTION_TABLE {
            let got = promote(a, b);
            assert_eq!(
                got, want,
                "promote({a}, {b}) = {got}, numpy says {want}"
            );
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
        let expect: &[(DType, &str, i32, char, char, &str, usize)] = &[
            (DType::Bool, "bool", 0, 'b', '?', "|b1", 1),
            (DType::I8, "int8", 1, 'i', 'b', "|i1", 1),
            (DType::I16, "int16", 3, 'i', 'h', "<i2", 2),
            (DType::I32, "int32", 5, 'i', 'i', "<i4", 4),
            (DType::I64, "int64", 7, 'i', 'l', "<i8", 8),
            (DType::U8, "uint8", 2, 'u', 'B', "|u1", 1),
            (DType::U16, "uint16", 4, 'u', 'H', "<u2", 2),
            (DType::U32, "uint32", 6, 'u', 'I', "<u4", 4),
            (DType::U64, "uint64", 8, 'u', 'L', "<u8", 8),
            (DType::F32, "float32", 11, 'f', 'f', "<f4", 4),
            (DType::F64, "float64", 12, 'f', 'd', "<f8", 8),
            (DType::C64, "complex64", 14, 'c', 'F', "<c8", 8),
            (DType::C128, "complex128", 15, 'c', 'D', "<c16", 16),
        ];
        for &(d, name, num, kind, ch, s, isz) in expect {
            assert_eq!(d.name(), name);
            assert_eq!(d.num(), num);
            assert_eq!(d.kind(), kind);
            assert_eq!(d.char_code(), ch);
            assert_eq!(d.str_code(), s);
            assert_eq!(d.itemsize(), isz);
        }
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
        assert_eq!(DType::from_str("nonsense"), None);
        // Round-trip every dtype through its own spellings.
        for d in ALL_DTYPES {
            assert_eq!(DType::from_str(d.name()), Some(d));
            assert_eq!(DType::from_str(&d.char_code().to_string()), Some(d));
            assert_eq!(DType::from_str(&d.str_code()), Some(d));
        }
    }
}
