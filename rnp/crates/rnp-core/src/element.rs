//! Element types and the C-cast semantics numpy uses for `astype`.

use crate::dtype::DType;
use num_complex::Complex;

pub type C32 = Complex<f32>;
pub type C64v = Complex<f64>;

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

    fn as_f64(&self) -> f64 {
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
                Scalar::Float(f) => f as u32 as u64,
                Scalar::Complex(c) => c.re as u32 as u64,
                other => other.as_u64() as u32 as u64,
            }),
            DType::U64 => Scalar::Uint(self.as_u64()),
            DType::F32 => Scalar::Float(self.as_f64() as f32 as f64),
            DType::F64 => Scalar::Float(self.as_f64()),
            DType::C64 => {
                let c = self.as_complex();
                Scalar::Complex(Complex::new(c.re as f32 as f64, c.im as f32 as f64))
            }
            DType::C128 => Scalar::Complex(self.as_complex()),
        }
    }
}

fn f2i64(f: f64) -> i64 {
    f as i64
}
fn f2u64(f: f64) -> u64 {
    f as u64
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
                match s.cast($dt) {
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
        match s.cast(DType::Bool) {
            Scalar::Bool(b) => NpBool::new(b),
            _ => unreachable!(),
        }
    }
    fn to_scalar(self) -> Scalar {
        Scalar::Bool(self.get())
    }
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
    fn from_scalar(s: Scalar) -> Self {
        match s.cast(DType::F32) {
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
        match s.cast(DType::F64) {
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
        match s.cast(DType::C64) {
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
        match s.cast(DType::C128) {
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
        let want_u32: [u64; 12] = [0, 0, 0, 3, 300, 4294967295, 0, 0, 4294967295, 0, 255, 256];
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
        let want_u64: [u64; 12] = [0, 0, 0, 3, 300, u64::MAX, 0, 0, u64::MAX, 0, 255, 256];

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
}
