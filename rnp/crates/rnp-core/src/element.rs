//! Element types and the C-cast semantics numpy uses for `astype`.

use crate::dtype::DType;
use crate::fpe;
use num_complex::Complex;

pub type C32 = Complex<f32>;
pub type C64v = Complex<f64>;

/// Canonical invalid-result NaN produced by NumPy's platform loops.
#[inline]
pub fn invalid_nan() -> f64 {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        f64::from_bits(0xfff8_0000_0000_0000)
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        f64::NAN
    }
}

/// A single element value, held losslessly in the widest representation of
/// its category.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    Complex(C64v),
}

impl Scalar {
    /// The "natural" dtype of this scalar (used when inferring array dtypes
    /// from Python values).
    pub fn natural_dtype(&self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int(_) => DType::I64,
            Scalar::Uint(_) => DType::U64,
            Scalar::Float(_) => DType::F64,
            Scalar::Complex(_) => DType::C128,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match *self {
            Scalar::Bool(b) => b as u8 as f64,
            Scalar::Int(i) => i as f64,
            Scalar::Uint(u) => u as f64,
            Scalar::Float(f) => f,
            Scalar::Complex(c) => c.re,
        }
    }

    fn as_i64(&self) -> i64 {
        match *self {
            Scalar::Bool(b) => b as i64,
            Scalar::Int(i) => i,
            Scalar::Uint(u) => u as i64,
            Scalar::Float(f) => f2i64(f),
            Scalar::Complex(c) => f2i64(c.re),
        }
    }

    fn as_u64(&self) -> u64 {
        match *self {
            Scalar::Bool(b) => b as u64,
            Scalar::Int(i) => i as u64,
            Scalar::Uint(u) => u,
            Scalar::Float(f) => f2u64(f),
            Scalar::Complex(c) => f2u64(c.re),
        }
    }

    /// Truncate-to-i32-then-wrap, which is what a C cast to a narrow integer
    /// type compiles to on the platforms numpy targets. Verified against
    /// numpy: `np.float64(-1.7).astype(np.uint8) == 255`.
    fn as_narrow(&self) -> i32 {
        match *self {
            Scalar::Bool(b) => b as i32,
            Scalar::Int(i) => i as i32,
            Scalar::Uint(u) => u as i32,
            Scalar::Float(f) => f as i32,
            Scalar::Complex(c) => c.re as i32,
        }
    }

    fn as_bool(&self) -> bool {
        match *self {
            Scalar::Bool(b) => b,
            Scalar::Int(i) => i != 0,
            Scalar::Uint(u) => u != 0,
            Scalar::Float(f) => f != 0.0,
            Scalar::Complex(c) => c.re != 0.0 || c.im != 0.0,
        }
    }

    fn as_complex(&self) -> C64v {
        match *self {
            Scalar::Complex(c) => c,
            other => Complex::new(other.as_f64(), 0.0),
        }
    }

    /// Cast to `dtype` with numpy's `unsafe` (C-cast) semantics.
    pub fn cast(self, dtype: DType) -> Scalar {
        match dtype {
            DType::Bool => Scalar::Bool(self.as_bool()),
            DType::I8 => Scalar::Int(self.as_narrow() as i8 as i64),
            DType::I16 => Scalar::Int(self.as_narrow() as i16 as i64),
            DType::I32 => Scalar::Int(match self {
                Scalar::Float(f) => f as i32 as i64,
                Scalar::Complex(c) => c.re as i32 as i64,
                other => other.as_i64() as i32 as i64,
            }),
            DType::I64 => Scalar::Int(self.as_i64()),
            DType::U8 => Scalar::Uint(self.as_narrow() as u8 as u64),
            DType::U16 => Scalar::Uint(self.as_narrow() as u16 as u64),
            DType::U32 => Scalar::Uint(match self {
                Scalar::Float(f) => f2u32(f) as u64,
                Scalar::Complex(c) => f2u32(c.re) as u64,
                other => other.as_u64() as u32 as u64,
            }),
            DType::U64 => Scalar::Uint(self.as_u64()),
            DType::F16 => Scalar::Float(F16::from_f64(self.as_f64()).to_f64()),
            DType::F32 => Scalar::Float(self.as_f64() as f32 as f64),
            DType::F64 => Scalar::Float(self.as_f64()),
            DType::C64 => {
                let c = self.as_complex();
                Scalar::Complex(Complex::new(c.re as f32 as f64, c.im as f32 as f64))
            }
            DType::C128 => Scalar::Complex(self.as_complex()),
            // Flexible dtypes hold bytes, not numbers: `Scalar` cannot
            // represent them, and every path that could reach here is
            // guarded by `DType::is_numeric`.
            _ => self,
        }
    }

    /// Cast as part of a NumPy cast loop, recording its IEEE status.
    ///
    /// Plain [`Self::cast`] is also used to normalise operands before an
    /// arithmetic kernel.  Those bookkeeping conversions are not observable
    /// cast loops and must not leak flags into the operation's accumulator.
    #[inline]
    pub fn cast_with_fpe(self, dtype: DType) -> Scalar {
        self.record_cast_fpe(dtype);
        self.cast(dtype)
    }

    /// Record the IEEE condition NumPy attributes to a cast loop.
    #[inline]
    fn record_cast_fpe(self, dtype: DType) {
        if matches!(dtype.kind(), 'i' | 'u') {
            let real = match self {
                Scalar::Float(v) => Some(v),
                Scalar::Complex(v) => Some(v.re),
                _ => None,
            };
            if real.is_some_and(|v| !v.is_finite()) {
                fpe::raise(fpe::INVALID);
            }
            return;
        }

        let overflow = match self {
            Scalar::Float(v) => match dtype {
                DType::F16 => v.is_finite() && F16::from_f64(v).to_f64().is_infinite(),
                DType::F32 | DType::C64 => v.is_finite() && (v as f32).is_infinite(),
                _ => false,
            },
            Scalar::Complex(v) => {
                matches!(dtype, DType::F32 | DType::C64)
                    && ((v.re.is_finite() && (v.re as f32).is_infinite())
                        || (v.im.is_finite() && (v.im as f32).is_infinite()))
            }
            // Every core integer fits all modeled float formats except f16.
            Scalar::Int(v) => {
                dtype == DType::F16 && F16::from_f64(v as f64).to_f64().is_infinite()
            }
            Scalar::Uint(v) => {
                dtype == DType::F16 && F16::from_f64(v as f64).to_f64().is_infinite()
            }
            Scalar::Bool(_) => false,
        };
        if overflow {
            fpe::raise(fpe::OVER);
        }
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
pub(crate) fn f2i64(f: f64) -> i64 {
    f as i64
}

/// The x86-64 conversion instruction used by NumPy's manylinux loops returns
/// `INT64_MIN` for NaN and every out-of-range input. Rust's `as` conversion is
/// deliberately saturating, so reproduce the hardware/C-loop result here.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn f2i64(f: f64) -> i64 {
    const TWO63: f64 = 9_223_372_036_854_775_808.0;
    if f.is_finite() && (-TWO63..TWO63).contains(&f) {
        f as i64
    } else {
        i64::MIN
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn f2u32(f: f64) -> u32 {
    f as u32
}

/// GCC lowers a float-to-u32 cast through a signed 64-bit truncation on
/// x86-64, then keeps the low word. This makes finite negative values wrap.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn f2u32(f: f64) -> u32 {
    f2i64(f) as u32
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn f2u64(f: f64) -> u64 {
    f as u64
}

/// GCC's x86-64 unsigned conversion splits at 2**63, converts through a
/// signed lane, then restores the high bit. Match that generated loop exactly.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn f2u64(f: f64) -> u64 {
    const TWO63: f64 = 9_223_372_036_854_775_808.0;
    if f >= TWO63 {
        (f2i64(f - TWO63) as u64) ^ (1u64 << 63)
    } else {
        f2i64(f) as u64
    }
}

/// IEEE-754 binary16, stored as its raw bits.
///
/// numpy has no hardware half type either: every arithmetic operation is
/// performed in `float` and converted back (`npy_half_add` and friends in
/// `halffloat.c`), which is exactly what the `Arith`/`Cmp` impls do.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct F16(pub u16);

impl F16 {
    pub const ZERO: F16 = F16(0);

    #[inline]
    pub fn to_f32(self) -> f32 {
        f16_bits_to_f32(self.0)
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    #[inline]
    pub fn from_f32(v: f32) -> F16 {
        F16(f32_to_f16_bits(v))
    }

    /// Direct double -> half conversion. Going via `f32` would round twice
    /// and disagree with numpy's `npy_double_to_half` on the values that sit
    /// exactly on a half-precision rounding boundary.
    #[inline]
    pub fn from_f64(v: f64) -> F16 {
        F16(f64_to_f16_bits(v))
    }

    pub fn is_nan(self) -> bool {
        (self.0 & 0x7C00) == 0x7C00 && (self.0 & 0x03FF) != 0
    }
}

/// Round-to-nearest-even f32 -> binary16.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = (x >> 23) & 0xFF;
    let man = x & 0x007F_FFFF;
    if exp == 0xFF {
        // numpy's `npy_float_to_halfbits` keeps the top 10 mantissa bits, so
        // a NaN payload survives the round trip through binary16.
        return sign | 0x7C00 | ((man >> 13) as u16);
    }
    if exp == 0 {
        // f32 subnormals are far below the half subnormal range.
        return sign;
    }
    let half_exp = exp as i32 - 127 + 15;
    if half_exp >= 0x1F {
        return sign | 0x7C00;
    }
    if half_exp <= 0 {
        let shift = (14 - half_exp) as u32;
        if shift > 24 {
            return sign;
        }
        let full = man | 0x0080_0000;
        let mut out = (full >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        if (full & round_bit) != 0 && (full & (3 * round_bit - 1)) != 0 {
            out += 1;
        }
        return sign | out;
    }
    let mut bits = sign | ((half_exp as u16) << 10) | ((man >> 13) as u16);
    let round_bit = 1u32 << 12;
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
        // A carry out of the mantissa flows into the exponent, which is the
        // correct result (and reaches inf at the top).
        bits += 1;
    }
    bits
}

/// Round-to-nearest-even f64 -> binary16.
pub fn f64_to_f16_bits(value: f64) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 48) & 0x8000) as u16;
    let exp = (x >> 52) & 0x7FF;
    let man = x & 0x000F_FFFF_FFFF_FFFF;
    if exp == 0x7FF {
        // As the f32 path: `npy_double_to_halfbits` keeps the top 10 bits.
        return sign | 0x7C00 | ((man >> 42) as u16);
    }
    if exp == 0 {
        return sign;
    }
    let half_exp = exp as i64 - 1023 + 15;
    if half_exp >= 0x1F {
        return sign | 0x7C00;
    }
    if half_exp <= 0 {
        let shift = (43 - half_exp) as u32;
        if shift > 53 {
            return sign;
        }
        let full = man | 0x0010_0000_0000_0000;
        let mut out = (full >> shift) as u16;
        let round_bit = 1u64 << (shift - 1);
        if (full & round_bit) != 0 && (full & (3 * round_bit - 1)) != 0 {
            out += 1;
        }
        return sign | out;
    }
    let mut bits = sign | ((half_exp as u16) << 10) | ((man >> 42) as u16);
    let round_bit = 1u64 << 41;
    if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
        bits += 1;
    }
    bits
}

/// binary16 -> f32 (always exact).
pub fn f16_bits_to_f32(i: u16) -> f32 {
    let sign = ((i & 0x8000) as u32) << 16;
    let exp = (i & 0x7C00) as u32;
    let man = (i & 0x03FF) as u32;
    if exp == 0x7C00 {
        // inf / nan
        return f32::from_bits(sign | 0x7F80_0000 | (man << 13));
    }
    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: renormalise. numpy's `npy_halfbits_to_floatbits`
        // shifts once *before* the loop, which is what makes half bits `1`
        // come out as 2^-24 rather than 2^-25.
        let mut m = man << 1;
        let mut e = 0u32;
        while m & 0x0400 == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 - e) << 23;
        return f32::from_bits(sign | exp32 | ((m & 0x03FF) << 13));
    }
    let exp32 = ((exp >> 10) + (127 - 15)) << 23;
    f32::from_bits(sign | exp32 | (man << 13))
}

/// A Rust type that can live inside an `NdArray` buffer.
pub trait Element: Copy + 'static + std::fmt::Debug {
    const DTYPE: DType;
    fn from_scalar(s: Scalar) -> Self;
    fn to_scalar(self) -> Scalar;
}

macro_rules! impl_int_element {
    ($t:ty, $dt:expr, $variant:ident) => {
        impl Element for $t {
            const DTYPE: DType = $dt;
            fn from_scalar(s: Scalar) -> Self {
                match s.cast_with_fpe($dt) {
                    Scalar::Int(i) => i as $t,
                    Scalar::Uint(u) => u as $t,
                    other => other.as_i64() as $t,
                }
            }
            fn to_scalar(self) -> Scalar {
                Scalar::$variant(self as _)
            }
        }
    };
}

impl_int_element!(i8, DType::I8, Int);
impl_int_element!(i16, DType::I16, Int);
impl_int_element!(i32, DType::I32, Int);
impl_int_element!(i64, DType::I64, Int);
impl_int_element!(u8, DType::U8, Uint);
impl_int_element!(u16, DType::U16, Uint);
impl_int_element!(u32, DType::U32, Uint);
impl_int_element!(u64, DType::U64, Uint);

/// numpy stores `bool` as a single byte holding 0 or 1.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct NpBool(pub u8);

impl NpBool {
    pub fn get(self) -> bool {
        self.0 != 0
    }
    pub fn new(b: bool) -> Self {
        NpBool(b as u8)
    }
}

impl Element for NpBool {
    const DTYPE: DType = DType::Bool;
    fn from_scalar(s: Scalar) -> Self {
        match s.cast_with_fpe(DType::Bool) {
            Scalar::Bool(b) => NpBool::new(b),
            _ => unreachable!(),
        }
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Bool(self.get())
    }
}

impl Element for F16 {
    const DTYPE: DType = DType::F16;
    fn from_scalar(s: Scalar) -> Self {
        match s.cast_with_fpe(DType::F16) {
            Scalar::Float(f) => F16::from_f64(f),
            _ => unreachable!(),
        }
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Float(self.to_f64())
    }
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
    fn from_scalar(s: Scalar) -> Self {
        match s.cast_with_fpe(DType::F32) {
            Scalar::Float(f) => f as f32,
            _ => unreachable!(),
        }
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Float(self as f64)
    }
}

impl Element for f64 {
    const DTYPE: DType = DType::F64;
    fn from_scalar(s: Scalar) -> Self {
        match s.cast_with_fpe(DType::F64) {
            Scalar::Float(f) => f,
            _ => unreachable!(),
        }
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Float(self)
    }
}

impl Element for C32 {
    const DTYPE: DType = DType::C64;
    fn from_scalar(s: Scalar) -> Self {
        match s.cast_with_fpe(DType::C64) {
            Scalar::Complex(c) => Complex::new(c.re as f32, c.im as f32),
            _ => unreachable!(),
        }
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Complex(Complex::new(self.re as f64, self.im as f64))
    }
}

impl Element for C64v {
    const DTYPE: DType = DType::C128;
    fn from_scalar(s: Scalar) -> Self {
        match s.cast_with_fpe(DType::C128) {
            Scalar::Complex(c) => c,
            _ => unreachable!(),
        }
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Complex(self)
    }
}

/// Run `$body` with `$T` bound to the Rust type for the given runtime dtype.
///
/// This is the single dispatch point used by every typed inner loop.
#[macro_export]
macro_rules! dispatch_dtype {
    ($dtype:expr, $T:ident, $body:block) => {
        match $dtype {
            $crate::dtype::DType::Bool => {
                type $T = $crate::element::NpBool;
                $body
            }
            $crate::dtype::DType::I8 => {
                type $T = i8;
                $body
            }
            $crate::dtype::DType::I16 => {
                type $T = i16;
                $body
            }
            $crate::dtype::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::dtype::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::dtype::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::dtype::DType::U16 => {
                type $T = u16;
                $body
            }
            $crate::dtype::DType::U32 => {
                type $T = u32;
                $body
            }
            $crate::dtype::DType::U64 => {
                type $T = u64;
                $body
            }
            $crate::dtype::DType::F16 => {
                type $T = $crate::element::F16;
                $body
            }
            $crate::dtype::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::dtype::DType::F64 => {
                type $T = f64;
                $body
            }
            $crate::dtype::DType::C64 => {
                type $T = $crate::element::C32;
                $body
            }
            $crate::dtype::DType::C128 => {
                type $T = $crate::element::C64v;
                $body
            }
            // datetime64 / timedelta64 are stored as int64; the *unit* is
            // metadata, so every generic loop (copy, indexing, sort, buffer
            // access) can treat them as i64. The operations where the unit
            // matters -- casting and arithmetic -- intercept before they get
            // here, in `crate::datetime` and `crate::ops`.
            $crate::dtype::DType::DateTime(_) | $crate::dtype::DType::TimeDelta(_) => {
                type $T = i64;
                $body
            }
            // Unreachable: every caller guards on `DType::is_numeric` first,
            // because flexible dtypes have no scalar element type.
            other => panic!("dispatch_dtype: {other:?} is not a numeric dtype"),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_to_narrow_int_matches_numpy() {
        // Probed: np.array([...], f8).astype(t).tolist()
        let vals = [
            -1.7f64,
            -0.5,
            -300.0,
            3.9,
            300.0,
            1e20,
            -1e20,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            255.0,
            256.0,
        ];
        let want_i8: [i64; 12] = [-1, 0, -44, 3, 44, -1, 0, 0, -1, 0, -1, 0];
        let want_u8: [u64; 12] = [255, 0, 212, 3, 44, 255, 0, 0, 255, 0, 255, 0];
        let want_i16: [i64; 12] = [-1, 0, -300, 3, 300, -1, 0, 0, -1, 0, 255, 256];
        let want_u16: [u64; 12] = [65535, 0, 65236, 3, 300, 65535, 0, 0, 65535, 0, 255, 256];
        let want_i32: [i64; 12] = [
            -1,
            0,
            -300,
            3,
            300,
            2147483647,
            -2147483648,
            0,
            2147483647,
            -2147483648,
            255,
            256,
        ];
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        let want_u32: [u64; 12] = [0, 0, 0, 3, 300, 4294967295, 0, 0, 4294967295, 0, 255, 256];
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let want_u32: [u64; 12] = [
            4_294_967_295,
            0,
            4_294_966_996,
            3,
            300,
            0,
            0,
            0,
            0,
            0,
            255,
            256,
        ];
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        let want_i64: [i64; 12] = [
            -1,
            0,
            -300,
            3,
            300,
            i64::MAX,
            i64::MIN,
            0,
            i64::MAX,
            i64::MIN,
            255,
            256,
        ];
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let want_i64: [i64; 12] = [
            -1,
            0,
            -300,
            3,
            300,
            i64::MIN,
            i64::MIN,
            i64::MIN,
            i64::MIN,
            i64::MIN,
            255,
            256,
        ];
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        let want_u64: [u64; 12] = [0, 0, 0, 3, 300, u64::MAX, 0, 0, u64::MAX, 0, 255, 256];
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let want_u64: [u64; 12] = [
            u64::MAX,
            0,
            u64::MAX - 299,
            3,
            300,
            0,
            1u64 << 63,
            1u64 << 63,
            0,
            1u64 << 63,
            255,
            256,
        ];

        for (i, &v) in vals.iter().enumerate() {
            let s = Scalar::Float(v);
            assert_eq!(i8::from_scalar(s) as i64, want_i8[i], "i8 @{i}");
            assert_eq!(u8::from_scalar(s) as u64, want_u8[i], "u8 @{i}");
            assert_eq!(i16::from_scalar(s) as i64, want_i16[i], "i16 @{i}");
            assert_eq!(u16::from_scalar(s) as u64, want_u16[i], "u16 @{i}");
            assert_eq!(i32::from_scalar(s) as i64, want_i32[i], "i32 @{i}");
            assert_eq!(u32::from_scalar(s) as u64, want_u32[i], "u32 @{i}");
            assert_eq!(i64::from_scalar(s), want_i64[i], "i64 @{i}");
            assert_eq!(u64::from_scalar(s), want_u64[i], "u64 @{i}");
        }
    }

    #[test]
    fn int_to_int_wraps_like_c() {
        // np.array([-1,-300,300], i4).astype('uint64')
        assert_eq!(u64::from_scalar(Scalar::Int(-1)), u64::MAX);
        assert_eq!(u64::from_scalar(Scalar::Int(-300)), u64::MAX - 299);
        // np.array([-1,300,-300], i8).astype('int8')
        assert_eq!(i8::from_scalar(Scalar::Int(-1)), -1);
        assert_eq!(i8::from_scalar(Scalar::Int(300)), 44);
        assert_eq!(i8::from_scalar(Scalar::Int(-300)), -44);
    }

    #[test]
    fn to_bool_is_nonzero_test() {
        assert!(NpBool::from_scalar(Scalar::Float(f64::NAN)).get());
        assert!(!NpBool::from_scalar(Scalar::Float(0.0)).get());
        assert!(!NpBool::from_scalar(Scalar::Float(-0.0)).get());
        assert!(NpBool::from_scalar(Scalar::Complex(Complex::new(0.0, 1.0))).get());
        assert!(!NpBool::from_scalar(Scalar::Complex(Complex::new(0.0, 0.0))).get());
    }

    #[test]
    fn complex_to_real_discards_imaginary() {
        let s = Scalar::Complex(Complex::new(1.0, 2.0));
        assert_eq!(f32::from_scalar(s), 1.0);
        assert_eq!(f64::from_scalar(s), 1.0);
        assert_eq!(i64::from_scalar(s), 1);
    }

    #[test]
    fn float_downcast_overflows_to_infinity() {
        assert_eq!(f32::from_scalar(Scalar::Float(1e300)), f32::INFINITY);
    }

    #[test]
    fn invalid_nan_has_the_platform_sign() {
        assert!(invalid_nan().is_nan());
        assert_eq!(
            invalid_nan().is_sign_negative(),
            cfg!(all(target_os = "linux", target_arch = "x86_64"))
        );
    }
}

#[cfg(test)]
mod half_tests {
    use super::*;

    /// Every one of the 65536 half bit patterns must survive a round trip
    /// through `f32` and through `f64` -- numpy's `test_half_conversions`.
    #[test]
    fn every_half_round_trips_through_f32_and_f64() {
        for bits in 0u16..=0xFFFF {
            let h = F16(bits);
            let f = h.to_f32();
            if f.is_nan() {
                continue; // the NaN payload comparison is what numpy skips
            }
            assert_eq!(F16::from_f32(f).0, bits, "f32 round trip of {bits:#06x}");
            assert_eq!(F16::from_f64(f as f64).0, bits, "f64 round trip of {bits:#06x}");
        }
    }

    #[test]
    fn half_subnormals_have_the_right_magnitude() {
        // np.uint16(1).view(np.float16) == 5.960464477539063e-08 == 2**-24
        assert_eq!(F16(1).to_f64(), 2.0f64.powi(-24));
        assert_eq!(F16(0x03FF).to_f64(), 1023.0 * 2.0f64.powi(-24));
        assert_eq!(F16(0x0400).to_f64(), 2.0f64.powi(-14));
    }
}
