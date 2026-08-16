//! Unary ufuncs.
//!
//! Type resolution follows numpy's loop tables (probed from `np.<f>.types` in
//! `.venv` and reproduced in `harness/dev_check.py`): the transcendental
//! functions have `e/f/d` and `F/D` loops only, so integers are lifted to
//! `float64`; `floor`/`ceil`/`trunc`/`rint` additionally have integer loops
//! where they are the identity; `isnan`/`isinf`/`isfinite`/`signbit`/
//! `logical_not` return bool.

use crate::array::NdArray;
use crate::dtype::DType;
use crate::element::{Element, NpBool, Scalar, C32, C64v, F16};
use crate::error::{Error, Result};
use crate::fpe;
use crate::loops::{unary1, unary1_flagged};
use crate::ops::{Arith, Cmp, CplxPow, FpClass, IntOps, SignedBits};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnOp {
    Negative,
    Positive,
    Absolute,
    Fabs,
    Sign,
    Rint,
    Floor,
    Ceil,
    Trunc,
    Sqrt,
    Cbrt,
    Square,
    Reciprocal,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log2,
    Log10,
    Log1p,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Sinh,
    Cosh,
    Tanh,
    Arcsinh,
    Arccosh,
    Arctanh,
    Deg2rad,
    Rad2deg,
    Conjugate,
    Invert,
    LogicalNot,
    IsNan,
    IsInf,
    IsFinite,
    Signbit,
    Spacing,
    BitwiseCount,
    OnesLike,
}

impl UnOp {
    pub fn name(self) -> &'static str {
        use UnOp::*;
        match self {
            Negative => "negative",
            Positive => "positive",
            Absolute => "absolute",
            Fabs => "fabs",
            Sign => "sign",
            Rint => "rint",
            Floor => "floor",
            Ceil => "ceil",
            Trunc => "trunc",
            Sqrt => "sqrt",
            Cbrt => "cbrt",
            Square => "square",
            Reciprocal => "reciprocal",
            Exp => "exp",
            Exp2 => "exp2",
            Expm1 => "expm1",
            Log => "log",
            Log2 => "log2",
            Log10 => "log10",
            Log1p => "log1p",
            Sin => "sin",
            Cos => "cos",
            Tan => "tan",
            Arcsin => "arcsin",
            Arccos => "arccos",
            Arctan => "arctan",
            Sinh => "sinh",
            Cosh => "cosh",
            Tanh => "tanh",
            Arcsinh => "arcsinh",
            Arccosh => "arccosh",
            Arctanh => "arctanh",
            Deg2rad => "deg2rad",
            Rad2deg => "rad2deg",
            Conjugate => "conjugate",
            Invert => "invert",
            LogicalNot => "logical_not",
            IsNan => "isnan",
            IsInf => "isinf",
            IsFinite => "isfinite",
            Signbit => "signbit",
            Spacing => "spacing",
            BitwiseCount => "bitwise_count",
            OnesLike => "_ones_like",
        }
    }

    pub fn from_name(s: &str) -> Option<UnOp> {
        use UnOp::*;
        Some(match s {
            "negative" => Negative,
            "positive" => Positive,
            "absolute" | "abs" => Absolute,
            "fabs" => Fabs,
            "sign" => Sign,
            "rint" => Rint,
            "floor" => Floor,
            "ceil" => Ceil,
            "trunc" => Trunc,
            "sqrt" => Sqrt,
            "cbrt" => Cbrt,
            "square" => Square,
            "reciprocal" => Reciprocal,
            "exp" => Exp,
            "exp2" => Exp2,
            "expm1" => Expm1,
            "log" => Log,
            "log2" => Log2,
            "log10" => Log10,
            "log1p" => Log1p,
            "sin" => Sin,
            "cos" => Cos,
            "tan" => Tan,
            "arcsin" | "asin" => Arcsin,
            "arccos" | "acos" => Arccos,
            "arctan" | "atan" => Arctan,
            "sinh" => Sinh,
            "cosh" => Cosh,
            "tanh" => Tanh,
            "arcsinh" | "asinh" => Arcsinh,
            "arccosh" | "acosh" => Arccosh,
            "arctanh" | "atanh" => Arctanh,
            "deg2rad" | "radians" => Deg2rad,
            "rad2deg" | "degrees" => Rad2deg,
            "conjugate" | "conj" => Conjugate,
            "invert" | "bitwise_invert" | "bitwise_not" => Invert,
            "logical_not" => LogicalNot,
            "isnan" => IsNan,
            "isinf" => IsInf,
            "isfinite" => IsFinite,
            "signbit" => Signbit,
            "spacing" => Spacing,
            "bitwise_count" => BitwiseCount,
            "_ones_like" => OnesLike,
            _ => return None,
        })
    }

    /// Ops that produce a bool array whatever the input dtype.
    fn returns_bool(self) -> bool {
        matches!(
            self,
            UnOp::IsNan | UnOp::IsInf | UnOp::IsFinite | UnOp::Signbit | UnOp::LogicalNot
        )
    }

    /// Ops with an integer loop that is the identity (numpy really does have
    /// `l->l` entries for these).
    fn identity_on_ints(self) -> bool {
        matches!(
            self,
            UnOp::Floor | UnOp::Ceil | UnOp::Trunc | UnOp::Positive | UnOp::Conjugate
        )
    }

    /// Ops whose loop table covers bool, integers and floats alike.
    fn works_on_ints(self) -> bool {
        matches!(
            self,
            UnOp::Negative
                | UnOp::Positive
                | UnOp::Absolute
                | UnOp::Sign
                | UnOp::Square
                | UnOp::Reciprocal
                | UnOp::Conjugate
                | UnOp::Invert
                | UnOp::BitwiseCount
                | UnOp::Floor
                | UnOp::Ceil
                | UnOp::Trunc
                | UnOp::OnesLike
        )
    }

    /// Ops with `F->F` / `D->D` loops.
    fn works_on_complex(self) -> bool {
        !matches!(
            self,
            UnOp::Fabs
                | UnOp::Cbrt
                | UnOp::Signbit
                | UnOp::Spacing
                | UnOp::Deg2rad
                | UnOp::Rad2deg
                | UnOp::Invert
                | UnOp::BitwiseCount
                | UnOp::Floor
                | UnOp::Ceil
                | UnOp::Trunc
        )
    }

    /// `(compute dtype, output dtype)` for one input dtype.
    pub fn resolve(self, d: DType) -> Result<(DType, DType)> {
        use UnOp::*;
        if !d.is_numeric() {
            return Err(unsupported(self));
        }
        if self == Invert || self == BitwiseCount {
            if !(d.is_integer() || d.is_bool()) {
                return Err(unsupported(self));
            }
            let out = if self == BitwiseCount { DType::U8 } else { d };
            let compute = if self == BitwiseCount && d.is_bool() {
                DType::U8
            } else {
                d
            };
            return Ok((compute, out));
        }
        if self.returns_bool() {
            if self == Signbit {
                if d.is_complex() {
                    return Err(unsupported(self));
                }
                // numpy lifts integers into the smallest float loop.
                let f = if d.is_float() { d } else { crate::dtype::promote(d, DType::F16) };
                return Ok((f, DType::Bool));
            }
            return Ok((d, DType::Bool));
        }
        if d.is_complex() {
            if !self.works_on_complex() {
                return Err(unsupported(self));
            }
            // `absolute` on complex yields the real component type.
            let out = if self == Absolute {
                if d == DType::C64 {
                    DType::F32
                } else {
                    DType::F64
                }
            } else {
                d
            };
            return Ok((d, out));
        }
        if d.is_float() {
            return Ok((d, d));
        }
        // bool / integers.
        if d.is_bool() {
            // `positive` and `sign` use numpy's
            // `PyUFunc_SimpleUniformOperationTypeResolver` (via
            // `PyUFunc_SignTypeResolver` for `sign`), which resolves the
            // common dtype and then demands an exact loop: neither has a
            // `?->?` entry, so bool is refused rather than lifted to int8.
            if self == Positive || self == Sign {
                return Err(crate::error::ufunc_no_loop(self.name(), &[&d.name()]));
            }
            // `negative` goes through `PyUFunc_NegativeTypeResolver`, which
            // rejects bool *before* loop lookup with its own message -- so
            // this one is a plain `TypeError`, not a `UFuncTypeError`.
            if self == Negative {
                return Err(Error::TypeError(
                    "The numpy boolean negative, the `-` operator, is not \
                     supported, use the `~` operator or the logical_not \
                     function instead."
                        .to_string(),
                ));
            }
        }
        if self.works_on_ints() {
            let compute = match self {
                // `square`, `reciprocal` and `conjugate` have no bool loop;
                // `absolute` does.
                Square | Reciprocal | Conjugate | Absolute if d.is_bool() => {
                    if self == Absolute {
                        DType::Bool
                    } else {
                        DType::I8
                    }
                }
                _ => d,
            };
            return Ok((compute, compute));
        }

        // Everything else is float-only. numpy picks the *smallest* float
        // loop the input casts safely into, so `np.exp(np.uint8(1))` is
        // float16 and `np.exp(np.int16(1))` is float32.
        let f = crate::dtype::promote(d, DType::F16);
        Ok((f, f))
    }
}

// ---------------------------------------------------------------------------
// float32 sine and cosine
// ---------------------------------------------------------------------------
//
// numpy does not send `np.sin(float32)` to the C library. Its `f->f` loop is
// the vectorised Cody-Waite routine in
// `umath/loops_trigonometric.dispatch.cpp`, whose result differs from `sinf`
// in the last bit for ordinary arguments like 1.0. The transcription below is
// that routine, one lane at a time; every constant is the same hex float, and
// every `hn::MulAdd` is a real fused multiply-add, which is what makes it
// reproduce the vector loop exactly rather than approximately.
//
// Checked against numpy 2.5.2 over 80k arguments per function -- uniform,
// sub-radian, huge, and uniformly random bit patterns -- with zero mismatches.

/// `0x1.45f306p-1`: 2/pi, for the quadrant estimate.
const TWO_OVER_PI: f32 = f32::from_bits(0x3F22F983);
/// `0x1.8p+23`: adding and subtracting this rounds a float to an integer.
const RINT_MAGIC: f32 = f32::from_bits(0x4B400000);
/// pi/2 split into three parts, so `x - q*pi/2` keeps its digits.
const CODYW_HI: f32 = f32::from_bits(0xBFC90FD8);
const CODYW_MED: f32 = f32::from_bits(0xB4A8885A);
const CODYW_LO: f32 = f32::from_bits(0xA7C234C4);
/// Beyond these the Cody-Waite reduction cancels catastrophically and numpy
/// falls back to the C library.
const MAX_CODY_COS: f32 = 71476.0625;
const MAX_CODY_SIN: f32 = 117435.992;

/// `simd_cosine_poly_f32`: cos(r) for `r` in [-pi/4, pi/4], as a polynomial in
/// `r**2`.
#[inline]
fn cosine_poly_f32(x2: f32) -> f32 {
    let r = f32::from_bits(0x37CC730B).mul_add(x2, f32::from_bits(0xBAB6036E));
    let r = r.mul_add(x2, f32::from_bits(0x3D2AAA9E));
    let r = r.mul_add(x2, -0.5);
    r.mul_add(x2, 1.0)
}

/// `simd_sine_poly_f32`: sin(r) for `r` in [-pi/4, pi/4].
#[inline]
fn sine_poly_f32(x: f32, x2: f32) -> f32 {
    let r = f32::from_bits(0x363E9DDE).mul_add(x2, f32::from_bits(0xB95035DD));
    let r = r.mul_add(x2, f32::from_bits(0x3C0888CD));
    let r = r.mul_add(x2, f32::from_bits(0xBE2AAAAB));
    let r = r.mul_add(x2, 0.0);
    r.mul_add(x, x)
}

/// One lane of numpy's `simd_sincos_f32`.
#[inline]
fn sincos_f32(x: f32, is_cos: bool) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let max_cody = if is_cos { MAX_CODY_COS } else { MAX_CODY_SIN };
    if !(x.abs() <= max_cody) {
        // Infinities land here too, and `sinf(inf)` is the NaN (and the
        // `invalid`) numpy reports.
        return if is_cos { x.cos() } else { x.sin() };
    }
    // q = rint(x * 2/pi), the quadrant.
    let quadrant = (x * TWO_OVER_PI + RINT_MAGIC) - RINT_MAGIC;
    // Cody-Waite: x* = x - q*pi/2, in [-pi/4, pi/4].
    let r = quadrant.mul_add(CODYW_HI, x);
    let r = quadrant.mul_add(CODYW_MED, r);
    let r = quadrant.mul_add(CODYW_LO, r);
    let r2 = r * r;
    // `cos` is `sin` shifted one quadrant; then the sign follows bit 1 of q.
    let iq = quadrant as i32 + is_cos as i32;
    let v = if iq & 1 == 0 {
        sine_poly_f32(r, r2)
    } else {
        cosine_poly_f32(r2)
    };
    if iq & 2 == 2 {
        0.0 - v
    } else {
        v
    }
}

#[inline]
fn np_sin_f32(x: f32) -> f32 {
    sincos_f32(x, false)
}

#[inline]
fn np_cos_f32(x: f32) -> f32 {
    sincos_f32(x, true)
}

/// `asinh(x) = sign(x) * log(|x| + sqrt(x*x + 1))`, evaluated so that a huge
/// `|x|` does not overflow the intermediate square.
#[inline]
fn stable_asinh(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let a = x.abs();
    let r = if a > 1e150 {
        a.ln() + std::f64::consts::LN_2
    } else if a > 1.0 {
        (a + (a * a + 1.0).sqrt()).ln()
    } else {
        // log1p keeps the small-argument case accurate.
        (a + a * a / (1.0 + (1.0 + a * a).sqrt())).ln_1p()
    };
    r.copysign(x)
}

/// `acosh(x) = log(x + sqrt(x*x - 1))`, in the form that keeps the digits
/// near `x == 1` (where `x + sqrt(...)` cancels) and does not overflow for a
/// huge `x`.
#[inline]
fn stable_acosh(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x < 1.0 {
        return f64::NAN;
    }
    if x > 1e150 {
        return x.ln() + std::f64::consts::LN_2;
    }
    let t = x - 1.0;
    if t < 0.5 {
        // log1p(t + sqrt(t*(t+2))) is exact to the last few bits here.
        return (t + (t * (t + 2.0)).sqrt()).ln_1p();
    }
    (x + (x * x - 1.0).sqrt()).ln()
}

/// `atanh(x) = 0.5 * log1p(2x / (1 - x))`, which keeps its accuracy as
/// `|x| -> 1` where the naive `0.5 * log((1+x)/(1-x))` cancels.
#[inline]
fn stable_atanh(x: f64) -> f64 {
    if x.is_nan() || x == 0.0 {
        return x;
    }
    let a = x.abs();
    if a > 1.0 {
        return f64::NAN;
    }
    if a == 1.0 {
        return f64::INFINITY.copysign(x);
    }
    let r = if a < 0.5 {
        0.5 * (2.0 * a + 2.0 * a * a / (1.0 - a)).ln_1p()
    } else {
        0.5 * (2.0 * a / (1.0 - a)).ln_1p()
    };
    r.copysign(x)
}

fn unsupported(op: UnOp) -> Error {
    Error::TypeError(format!(
        "ufunc '{}' not supported for the input types, and the inputs could \
         not be safely coerced to any supported types according to the \
         casting rule ''safe''",
        op.name()
    ))
}

// ---------------------------------------------------------------------------
// Real-float transcendentals
// ---------------------------------------------------------------------------

/// The float-domain unary functions, computed in `f32`/`f64` natively and via
/// `f32` for binary16 (which is what numpy's half loops do).
pub trait FloatUn: Element + FpClass {
    fn fu(self, op: UnOp) -> Self;
    fn fu_signbit(self) -> bool;
    /// The conditions numpy's `spacing` loop reports for `spacing(self) == r`.
    ///
    /// `npy_spacing` (float and double) raises nothing for a non-finite
    /// argument -- it just returns a NaN -- and overflows only at the top of
    /// the range. `npy_half_spacing` is a separate routine that calls
    /// `npy_set_floatstatus_invalid()` explicitly for an infinite or NaN
    /// argument, so the half loop reports one more condition than the others.
    fn fu_spacing_flags(self, r: Self) -> u8;
}

macro_rules! impl_float_un {
    ($t:ty, $pi:expr, $sin:path, $cos:path) => {
        impl FloatUn for $t {
            #[inline]
            fn fu(self, op: UnOp) -> Self {
                use UnOp::*;
                let x = self;
                match op {
                    Negative => -x,
                    Positive => x,
                    Absolute | Fabs => x.abs(),
                    Sign => {
                        if x.is_nan() {
                            x
                        } else if x > 0.0 {
                            1.0
                        } else if x < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    }
                    Rint => x.round_ties_even(),
                    Floor => x.floor(),
                    Ceil => x.ceil(),
                    Trunc => x.trunc(),
                    Sqrt => x.sqrt(),
                    Cbrt => x.cbrt(),
                    Square => x * x,
                    Reciprocal => 1.0 / x,
                    Exp => x.exp(),
                    Exp2 => x.exp2(),
                    Expm1 => x.exp_m1(),
                    Log => x.ln(),
                    Log2 => x.log2(),
                    Log10 => x.log10(),
                    Log1p => x.ln_1p(),
                    Sin => $sin(x),
                    Cos => $cos(x),
                    Tan => x.tan(),
                    Arcsin => x.asin(),
                    Arccos => x.acos(),
                    Arctan => x.atan(),
                    Sinh => x.sinh(),
                    Cosh => x.cosh(),
                    Tanh => x.tanh(),
                    // The library forms overflow (`asinh`) or lose most of
                    // the mantissa (`acosh`, `atanh`) near their edges; these
                    // are numpy's own formulations.
                    Arcsinh => stable_asinh(x as f64) as Self,
                    Arccosh => stable_acosh(x as f64) as Self,
                    Arctanh => stable_atanh(x as f64) as Self,
                    Deg2rad => x * ($pi / 180.0),
                    Rad2deg => x * (180.0 / $pi),
                    Conjugate => x,
                    Spacing => crate::ops::RealFloat::r_spacing(x),
                    OnesLike => 1.0,
                    other => panic!("float has no unary loop for {}", other.name()),
                }
            }
            #[inline]
            fn fu_signbit(self) -> bool {
                self.is_sign_negative()
            }
            #[inline]
            fn fu_spacing_flags(self, r: Self) -> u8 {
                if r.is_infinite() && self.is_finite() {
                    crate::fpe::OVER
                } else {
                    0
                }
            }
        }
    };
}

impl_float_un!(f64, std::f64::consts::PI, f64::sin, f64::cos);
// numpy's float32 `sin`/`cos` are its own vectorised polynomial, not `sinf`.
impl_float_un!(f32, std::f32::consts::PI, np_sin_f32, np_cos_f32);

impl FloatUn for F16 {
    #[inline]
    fn fu(self, op: UnOp) -> Self {
        // numpy's half loops widen to float, call the float routine, and
        // round back. `spacing` and `sign` are done on the half bits so the
        // subnormal results stay exact.
        match op {
            UnOp::Spacing => crate::ops::RealFloat::r_spacing(self),
            UnOp::Negative => F16(self.0 ^ 0x8000),
            UnOp::Absolute | UnOp::Fabs => F16(self.0 & 0x7FFF),
            UnOp::Positive | UnOp::Conjugate => self,
            other => F16::from_f32(self.to_f32().fu(other)),
        }
    }
    #[inline]
    fn fu_signbit(self) -> bool {
        self.0 & 0x8000 != 0
    }
    #[inline]
    fn fu_spacing_flags(self, _r: Self) -> u8 {
        // Straight from `npy_half_spacing`: infinite or NaN argument raises
        // `invalid`, the largest finite half raises `overflow`.
        if self.0 & 0x7C00 == 0x7C00 {
            crate::fpe::INVALID
        } else if self.0 == 0x7BFF {
            crate::fpe::OVER
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Complex transcendentals
// ---------------------------------------------------------------------------

// The platform's C99 complex functions.
//
// numpy `#define`s `npy_csin` and friends straight to these: the fallbacks in
// `npy_math_complex.c.src` sit behind `#ifndef HAVE_CSIN@C@`, and the meson
// probe sets `HAVE_*` for the whole C99 complex family on every platform this
// port builds on. Calling the same routines is what makes the results match
// numpy bit for bit -- including the C99 Annex G special-value tables, which
// live in the C library rather than in numpy -- and, since the IEEE exception
// flags numpy reports come out of that same libm, it is also what makes the
// `RuntimeWarning`s match.
//
// `num_complex::Complex<T>` is a `#[repr(C)]` `{re, im}` pair, which is the
// same parameter class as C's `double _Complex` / `float _Complex` under both
// AArch64 AAPCS and the x86-64 System V ABI (a two-member homogeneous
// floating-point aggregate, passed and returned in two vector registers).
extern "C" {
    fn csqrt(z: C64v) -> C64v;
    fn cexp(z: C64v) -> C64v;
    fn clog(z: C64v) -> C64v;
    fn csin(z: C64v) -> C64v;
    fn ccos(z: C64v) -> C64v;
    fn ctan(z: C64v) -> C64v;
    fn casin(z: C64v) -> C64v;
    fn cacos(z: C64v) -> C64v;
    fn catan(z: C64v) -> C64v;
    fn csinh(z: C64v) -> C64v;
    fn ccosh(z: C64v) -> C64v;
    fn ctanh(z: C64v) -> C64v;
    fn casinh(z: C64v) -> C64v;
    fn cacosh(z: C64v) -> C64v;
    fn catanh(z: C64v) -> C64v;
    pub(crate) fn cpow(x: C64v, y: C64v) -> C64v;

    fn csqrtf(z: C32) -> C32;
    fn cexpf(z: C32) -> C32;
    fn clogf(z: C32) -> C32;
    fn csinf(z: C32) -> C32;
    fn ccosf(z: C32) -> C32;
    fn ctanf(z: C32) -> C32;
    fn casinf(z: C32) -> C32;
    fn cacosf(z: C32) -> C32;
    fn catanf(z: C32) -> C32;
    fn csinhf(z: C32) -> C32;
    fn ccoshf(z: C32) -> C32;
    fn ctanhf(z: C32) -> C32;
    fn casinhf(z: C32) -> C32;
    fn cacoshf(z: C32) -> C32;
    fn catanhf(z: C32) -> C32;
    pub(crate) fn cpowf(x: C32, y: C32) -> C32;
}

/// The complex unary functions, one arm per `nc_*` wrapper in numpy's
/// `umath/funcs.inc.src`. `C32` uses the single-precision libm entry points,
/// which is what numpy's `nc_*f` wrappers do -- not a double-precision compute
/// rounded back.
pub trait CplxUn: Element + FpClass {
    fn cu(self, op: UnOp) -> Self;
    fn c_abs(self) -> f64;
}

/// `CDOUBLE_sign`, transcribed from `umath/loops.c.src`.
#[inline]
fn c_sign_f64(z: C64v) -> C64v {
    let m = z.re.hypot(z.im);
    if m.is_nan() {
        C64v::new(f64::NAN, f64::NAN)
    } else if m.is_infinite() {
        if z.re.is_infinite() {
            if z.im.is_infinite() {
                C64v::new(f64::NAN, f64::NAN)
            } else {
                C64v::new(if z.re > 0.0 { 1.0 } else { -1.0 }, 0.0)
            }
        } else {
            C64v::new(0.0, if z.im > 0.0 { 1.0 } else { -1.0 })
        }
    } else if m == 0.0 {
        C64v::new(0.0, 0.0)
    } else {
        C64v::new(z.re / m, z.im / m)
    }
}

/// `CDOUBLE_reciprocal`, transcribed from `umath/loops.c.src`. This is *not*
/// `1/z` through the complex division loop: numpy's reciprocal keeps the sign
/// of the zero imaginary part (`1/(1+0j)` is `1-0j`).
#[inline]
fn c_recip_f64(z: C64v) -> C64v {
    if z.im.abs() <= z.re.abs() {
        let r = z.im / z.re;
        let d = z.re + z.im * r;
        C64v::new(1.0 / d, -r / d)
    } else {
        let r = z.re / z.im;
        let d = z.re * r + z.im;
        C64v::new(r / d, -1.0 / d)
    }
}

fn cplx_un_f64(z: C64v, op: UnOp) -> C64v {
    use UnOp::*;
    // SAFETY: every call below is a C99 `<complex.h>` entry point used with its
    // standard signature; see the `extern` block above for the ABI argument.
    unsafe {
        match op {
            Negative => C64v::new(-z.re, -z.im),
            Positive => z,
            Conjugate => C64v::new(z.re, -z.im),
            Absolute => C64v::new(z.re.hypot(z.im), 0.0),
            Sign => c_sign_f64(z),
            Rint => C64v::new(z.re.round_ties_even(), z.im.round_ties_even()),
            Square => C64v::new(
                z.re.mul_add(z.re, -(z.im * z.im)),
                z.re.mul_add(z.im, z.im * z.re),
            ),
            Reciprocal => c_recip_f64(z),
            Sqrt => csqrt(z),
            Exp => cexp(z),
            // `nc_exp2`: exp(z * ln 2).
            Exp2 => cexp(C64v::new(
                z.re * std::f64::consts::LN_2,
                z.im * std::f64::consts::LN_2,
            )),
            // `nc_expm1`, which is built from the *real* expm1/exp/sin/cos so
            // that a small real part keeps its digits.
            Expm1 => {
                let a = (z.im / 2.0).sin();
                C64v::new(
                    z.re.exp_m1() * z.im.cos() - 2.0 * a * a,
                    z.re.exp() * z.im.sin(),
                )
            }
            Log => clog(z),
            // `nc_log2`/`nc_log10` scale `clog` by log2(e)/log10(e). numpy
            // *multiplies*; dividing by ln 2 differs in the last bit.
            Log2 => {
                let l = clog(z);
                C64v::new(l.re * std::f64::consts::LOG2_E, l.im * std::f64::consts::LOG2_E)
            }
            Log10 => {
                let l = clog(z);
                C64v::new(
                    l.re * std::f64::consts::LOG10_E,
                    l.im * std::f64::consts::LOG10_E,
                )
            }
            // `nc_log1p`, again from the real routines.
            Log1p => C64v::new(
                (z.re + 1.0).hypot(z.im).ln(),
                z.im.atan2(z.re + 1.0),
            ),
            Sin => csin(z),
            Cos => ccos(z),
            Tan => ctan(z),
            Arcsin => casin(z),
            Arccos => cacos(z),
            Arctan => catan(z),
            Sinh => csinh(z),
            Cosh => ccosh(z),
            Tanh => ctanh(z),
            Arcsinh => casinh(z),
            Arccosh => cacosh(z),
            Arctanh => catanh(z),
            OnesLike => C64v::new(1.0, 0.0),
            other => panic!("complex has no unary loop for {}", other.name()),
        }
    }
}

/// `CFLOAT_sign`.
#[inline]
fn c_sign_f32(z: C32) -> C32 {
    let m = z.re.hypot(z.im);
    if m.is_nan() {
        C32::new(f32::NAN, f32::NAN)
    } else if m.is_infinite() {
        if z.re.is_infinite() {
            if z.im.is_infinite() {
                C32::new(f32::NAN, f32::NAN)
            } else {
                C32::new(if z.re > 0.0 { 1.0 } else { -1.0 }, 0.0)
            }
        } else {
            C32::new(0.0, if z.im > 0.0 { 1.0 } else { -1.0 })
        }
    } else if m == 0.0 {
        C32::new(0.0, 0.0)
    } else {
        C32::new(z.re / m, z.im / m)
    }
}

/// `CFLOAT_reciprocal`.
#[inline]
fn c_recip_f32(z: C32) -> C32 {
    if z.im.abs() <= z.re.abs() {
        let r = z.im / z.re;
        let d = z.re + z.im * r;
        C32::new(1.0 / d, -r / d)
    } else {
        let r = z.re / z.im;
        let d = z.re * r + z.im;
        C32::new(r / d, -1.0 / d)
    }
}

fn cplx_un_f32(z: C32, op: UnOp) -> C32 {
    use UnOp::*;
    // SAFETY: as `cplx_un_f64`.
    unsafe {
        match op {
            Negative => C32::new(-z.re, -z.im),
            Positive => z,
            Conjugate => C32::new(z.re, -z.im),
            Absolute => C32::new(z.re.hypot(z.im), 0.0),
            Sign => c_sign_f32(z),
            Rint => C32::new(z.re.round_ties_even(), z.im.round_ties_even()),
            Square => C32::new(
                z.re.mul_add(z.re, -(z.im * z.im)),
                z.re.mul_add(z.im, z.im * z.re),
            ),
            Reciprocal => c_recip_f32(z),
            Sqrt => csqrtf(z),
            Exp => cexpf(z),
            Exp2 => cexpf(C32::new(
                z.re * std::f32::consts::LN_2,
                z.im * std::f32::consts::LN_2,
            )),
            Expm1 => {
                let a = (z.im / 2.0).sin();
                C32::new(
                    z.re.exp_m1() * z.im.cos() - 2.0 * a * a,
                    z.re.exp() * z.im.sin(),
                )
            }
            Log => clogf(z),
            Log2 => {
                let l = clogf(z);
                C32::new(l.re * std::f32::consts::LOG2_E, l.im * std::f32::consts::LOG2_E)
            }
            Log10 => {
                let l = clogf(z);
                C32::new(
                    l.re * std::f32::consts::LOG10_E,
                    l.im * std::f32::consts::LOG10_E,
                )
            }
            Log1p => C32::new((z.re + 1.0).hypot(z.im).ln(), z.im.atan2(z.re + 1.0)),
            Sin => csinf(z),
            Cos => ccosf(z),
            Tan => ctanf(z),
            Arcsin => casinf(z),
            Arccos => cacosf(z),
            Arctan => catanf(z),
            Sinh => csinhf(z),
            Cosh => ccoshf(z),
            Tanh => ctanhf(z),
            Arcsinh => casinhf(z),
            Arccosh => cacoshf(z),
            Arctanh => catanhf(z),
            OnesLike => C32::new(1.0, 0.0),
            other => panic!("complex has no unary loop for {}", other.name()),
        }
    }
}

impl CplxUn for C64v {
    #[inline]
    fn cu(self, op: UnOp) -> Self {
        cplx_un_f64(self, op)
    }
    #[inline]
    fn c_abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl CplxUn for C32 {
    #[inline]
    fn cu(self, op: UnOp) -> Self {
        cplx_un_f32(self, op)
    }
    #[inline]
    fn c_abs(self) -> f64 {
        // `CFLOAT_absolute` is `npy_hypotf`: single precision throughout.
        self.re.hypot(self.im) as f64
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Element-wise unary ufunc.
pub fn unary(a: &NdArray, op: UnOp) -> Result<NdArray> {
    if a.dtype.is_flexible() || a.dtype.is_object() {
        return Err(unsupported(op));
    }
    let (compute, out_dtype) = op.resolve(a.dtype)?;
    let owned;
    let src = if a.dtype == compute {
        a
    } else {
        owned = a.astype(compute);
        &owned
    };
    let out = NdArray::empty(src.shape.clone(), out_dtype)?;
    let n = out.size();

    if op.returns_bool() {
        // SAFETY: `out` is a freshly allocated contiguous bool array of n bytes.
        let o = unsafe { out.buffer.as_mut_ptr() };
        run_bool_out(src, o, n, compute, op);
        return Ok(out);
    }
    if op == UnOp::BitwiseCount {
        // SAFETY: `out` is a freshly allocated contiguous u8 array of n bytes.
        let o = unsafe { out.buffer.as_mut_ptr() };
        crate::dispatch_dtype!(compute, T, {
            run_popcount::<T>(src, o, n);
        });
        return Ok(out);
    }
    if op == UnOp::Absolute && compute.is_complex() {
        // SAFETY: the output holds n elements of the real component type.
        match compute {
            DType::C64 => {
                let o = unsafe { out.buffer.as_mut_ptr() as *mut f32 };
                unary1::<C32, f32, _>(src, o, n, |z| z.c_abs() as f32);
            }
            _ => {
                let o = unsafe { out.buffer.as_mut_ptr() as *mut f64 };
                unary1::<C64v, f64, _>(src, o, n, |z| z.c_abs());
            }
        }
        return Ok(out);
    }

    match compute {
        DType::Bool => {
            // SAFETY: freshly allocated contiguous bool output.
            let o = unsafe { out.buffer.as_mut_ptr() as *mut NpBool };
            run_bool_un(src, o, n, op);
        }
        DType::I8 => run_int_un::<i8>(src, &out, n, op),
        DType::I16 => run_int_un::<i16>(src, &out, n, op),
        DType::I32 => run_int_un::<i32>(src, &out, n, op),
        DType::I64 => run_int_un::<i64>(src, &out, n, op),
        DType::U8 => run_int_un::<u8>(src, &out, n, op),
        DType::U16 => run_int_un::<u16>(src, &out, n, op),
        DType::U32 => run_int_un::<u32>(src, &out, n, op),
        DType::U64 => run_int_un::<u64>(src, &out, n, op),
        DType::F16 => run_float_un::<F16>(src, &out, n, op),
        DType::F32 => run_float_un::<f32>(src, &out, n, op),
        DType::F64 => run_float_un::<f64>(src, &out, n, op),
        DType::C64 => run_cplx_un::<C32>(src, &out, n, op),
        DType::C128 => run_cplx_un::<C64v>(src, &out, n, op),
        other => return Err(Error::NotImplemented(format!("unary on {other:?}"))),
    }
    Ok(out)
}

fn run_bool_out(a: &NdArray, o: *mut u8, n: usize, compute: DType, op: UnOp) {
    macro_rules! go {
        ($T:ty) => {
            match op {
                UnOp::LogicalNot => unary1::<$T, u8, _>(a, o, n, |x| !Cmp::c_truthy(x) as u8),
                UnOp::IsNan => unary1::<$T, u8, _>(a, o, n, |x| FpClass::fp_nan(x) as u8),
                UnOp::IsInf => unary1::<$T, u8, _>(a, o, n, |x| FpClass::fp_inf(x) as u8),
                UnOp::IsFinite => unary1::<$T, u8, _>(a, o, n, |x| FpClass::fp_finite(x) as u8),
                UnOp::Signbit => unary1::<$T, u8, _>(a, o, n, |x| FloatUn::fu_signbit(x) as u8),
                other => panic!("no bool-output loop for {}", other.name()),
            }
        };
    }
    macro_rules! go_int {
        ($T:ty) => {
            match op {
                UnOp::LogicalNot => unary1::<$T, u8, _>(a, o, n, |x| !Cmp::c_truthy(x) as u8),
                // Integers are never NaN or infinite.
                UnOp::IsNan | UnOp::IsInf => unary1::<$T, u8, _>(a, o, n, |_| 0u8),
                UnOp::IsFinite => unary1::<$T, u8, _>(a, o, n, |_| 1u8),
                other => panic!("no integer bool-output loop for {}", other.name()),
            }
        };
    }
    match compute {
        DType::Bool => go_int!(NpBool),
        DType::I8 => go_int!(i8),
        DType::I16 => go_int!(i16),
        DType::I32 => go_int!(i32),
        DType::I64 => go_int!(i64),
        DType::U8 => go_int!(u8),
        DType::U16 => go_int!(u16),
        DType::U32 => go_int!(u32),
        DType::U64 => go_int!(u64),
        DType::F16 => go!(F16),
        DType::F32 => go!(f32),
        DType::F64 => go!(f64),
        DType::C64 => match op {
            UnOp::LogicalNot => unary1::<C32, u8, _>(a, o, n, |x| !Cmp::c_truthy(x) as u8),
            UnOp::IsNan => unary1::<C32, u8, _>(a, o, n, |x| FpClass::fp_nan(x) as u8),
            UnOp::IsInf => unary1::<C32, u8, _>(a, o, n, |x| FpClass::fp_inf(x) as u8),
            UnOp::IsFinite => unary1::<C32, u8, _>(a, o, n, |x| FpClass::fp_finite(x) as u8),
            other => panic!("no complex bool-output loop for {}", other.name()),
        },
        _ => match op {
            UnOp::LogicalNot => unary1::<C64v, u8, _>(a, o, n, |x| !Cmp::c_truthy(x) as u8),
            UnOp::IsNan => unary1::<C64v, u8, _>(a, o, n, |x| FpClass::fp_nan(x) as u8),
            UnOp::IsInf => unary1::<C64v, u8, _>(a, o, n, |x| FpClass::fp_inf(x) as u8),
            UnOp::IsFinite => unary1::<C64v, u8, _>(a, o, n, |x| FpClass::fp_finite(x) as u8),
            other => panic!("no complex bool-output loop for {}", other.name()),
        },
    }
}

fn run_popcount<T: Element + Send + Sync>(a: &NdArray, o: *mut u8, n: usize) {
    macro_rules! go {
        ($T:ty) => {
            unary1::<$T, u8, _>(a, o, n, |x| IntOps::i_popcount(x))
        };
    }
    match T::DTYPE {
        DType::I8 => go!(i8),
        DType::I16 => go!(i16),
        DType::I32 => go!(i32),
        DType::I64 => go!(i64),
        DType::U8 => go!(u8),
        DType::U16 => go!(u16),
        DType::U32 => go!(u32),
        DType::U64 => go!(u64),
        other => panic!("bitwise_count on {other:?}"),
    }
}

fn run_bool_un(a: &NdArray, o: *mut NpBool, n: usize, op: UnOp) {
    match op {
        UnOp::Invert => unary1::<NpBool, NpBool, _>(a, o, n, |x| NpBool::new(!x.get())),
        UnOp::Positive | UnOp::Conjugate | UnOp::Rint | UnOp::Floor | UnOp::Ceil | UnOp::Trunc
        | UnOp::Absolute => unary1::<NpBool, NpBool, _>(a, o, n, |x| x),
        UnOp::OnesLike => unary1::<NpBool, NpBool, _>(a, o, n, |_| NpBool::new(true)),
        other => panic!("bool has no unary loop for {}", other.name()),
    }
}

fn run_int_un<T>(a: &NdArray, out: &NdArray, n: usize, op: UnOp)
where
    T: Element + Send + Sync + IntOps + SignedBits + Arith,
{
    // SAFETY: `out` is a freshly allocated contiguous array of n T's.
    let o = unsafe { out.buffer.as_mut_ptr() as *mut T };
    match op {
        UnOp::Negative => unary1::<T, T, _>(a, o, n, |x| x.i_negative()),
        UnOp::Absolute => unary1::<T, T, _>(a, o, n, |x| SignedBits::i_absolute(x)),
        UnOp::Sign => unary1::<T, T, _>(a, o, n, |x| x.i_sign()),
        UnOp::Square => unary1::<T, T, _>(a, o, n, |x| Arith::a_mul(x, x)),
        UnOp::Reciprocal => unary1::<T, T, _>(a, o, n, |x| x.i_reciprocal()),
        UnOp::Invert => unary1::<T, T, _>(a, o, n, |x| x.i_not()),
        UnOp::OnesLike => unary1::<T, T, _>(a, o, n, |_| T::from_scalar(Scalar::Int(1))),
        _ if op.identity_on_ints() => unary1::<T, T, _>(a, o, n, |x| x),
        other => panic!("integers have no unary loop for {}", other.name()),
    }
}

/// The float error rule for a unary result. `divide_like` names the ops where
/// numpy reports "divide by zero" rather than "overflow" for an infinite
/// result — `log*`, `reciprocal`, `arctanh` and friends, whose infinity comes
/// from a pole rather than from exceeding the exponent range.
#[inline]
fn is_pole_op(op: UnOp) -> bool {
    matches!(
        op,
        UnOp::Log
            | UnOp::Log2
            | UnOp::Log10
            | UnOp::Log1p
            | UnOp::Reciprocal
            | UnOp::Arctanh
            | UnOp::Tan
    )
}

/// Dispatch a float unary loop with the op as a *compile-time* constant.
///
/// Passing `op` into the closure as a runtime value would leave the whole
/// `FloatUn::fu` match inside the loop body, which costs ~4x on `abs_f64`
/// because nothing vectorises. One arm per op keeps each loop a single
/// inlined instruction sequence.
macro_rules! float_un_arms {
    ($op:expr, $T:ty, $a:expr, $o:expr, $n:expr, $drive:ident, [$($v:ident),* $(,)?]) => {
        match $op {
            $( UnOp::$v => $drive!(UnOp::$v), )*
            other => panic!("float has no unary loop for {}", other.name()),
        }
    };
}

fn run_float_un<T>(a: &NdArray, out: &NdArray, n: usize, op: UnOp)
where
    T: Element + Send + Sync + FloatUn,
{
    // SAFETY: `out` is a freshly allocated contiguous array of n T's.
    let o = unsafe { out.buffer.as_mut_ptr() as *mut T };

    // These can never raise, so they run through the unflagged driver.
    macro_rules! plain {
        ($c:expr) => {
            unary1::<T, T, _>(a, o, n, |x| x.fu($c))
        };
    }
    if matches!(
        op,
        UnOp::Negative
            | UnOp::Positive
            | UnOp::Absolute
            | UnOp::Fabs
            | UnOp::Conjugate
            | UnOp::Rint
            | UnOp::Floor
            | UnOp::Ceil
            | UnOp::Trunc
            | UnOp::Sign
            | UnOp::OnesLike
    ) {
        float_un_arms!(op, T, a, o, n, plain, [
            Negative, Positive, Absolute, Fabs, Conjugate, Rint, Floor, Ceil,
            Trunc, Sign, OnesLike,
        ]);
        return;
    }

    if op == UnOp::Spacing {
        unary1_flagged::<T, T, _, _, _>(
            a,
            o,
            n,
            |x| x.fu(UnOp::Spacing),
            // The half loop also reports a NaN *argument*, so the watch has to
            // look at the operand rather than only at the result.
            |x: T, r: T| !r.fp_finite() || !x.fp_finite(),
            |x, r| x.fu_spacing_flags(r),
        );
        return;
    }

    let pole = is_pole_op(op);
    macro_rules! flagged {
        ($c:expr) => {
            unary1_flagged::<T, T, _, _, _>(
                a,
                o,
                n,
                |x| x.fu($c),
                |_x: T, r: T| !r.fp_finite(),
                move |x, r| {
                    if r.fp_nan() {
                        if x.fp_nan() {
                            0
                        } else {
                            fpe::INVALID
                        }
                    } else if r.fp_inf() && x.fp_finite() {
                        if pole {
                            fpe::DIVIDE
                        } else {
                            fpe::OVER
                        }
                    } else {
                        0
                    }
                },
            )
        };
    }
    float_un_arms!(op, T, a, o, n, flagged, [
        Sqrt, Cbrt, Square, Reciprocal, Exp, Exp2, Expm1, Log, Log2, Log10,
        Log1p, Sin, Cos, Tan, Arcsin, Arccos, Arctan, Sinh, Cosh, Tanh,
        Arcsinh, Arccosh, Arctanh, Deg2rad, Rad2deg,
    ]);
}

fn run_cplx_un<T>(a: &NdArray, out: &NdArray, n: usize, op: UnOp)
where
    T: Element + Send + Sync + CplxUn + CplxPow,
{
    // SAFETY: `out` is a freshly allocated contiguous array of n T's.
    let o = unsafe { out.buffer.as_mut_ptr() as *mut T };
    // These loops call the platform's libm, so the error conditions numpy
    // reports are the ones the CPU status register picked up -- exactly what
    // numpy's own complex loops read back. See `fpe::hw_take`.
    crate::loops::unary1_hw::<T, _>(a, o, n, move |x| x.cu(op));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Scalar;

    fn f64arr(v: &[f64]) -> NdArray {
        let s: Vec<Scalar> = v.iter().map(|&x| Scalar::Float(x)).collect();
        NdArray::from_scalars(&s, DType::F64).unwrap()
    }

    fn get_f(a: &NdArray, i: usize) -> f64 {
        match a.get_flat(i) {
            Scalar::Float(f) => f,
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn resolution_matches_numpy_loop_tables() {
        // Transcendentals lift to the *smallest* float loop that fits, which
        // is what numpy's `e/f/d` table selects: uint8 -> float16, int16 ->
        // float32, int32 -> float64.
        assert_eq!(UnOp::Exp.resolve(DType::I32).unwrap(), (DType::F64, DType::F64));
        assert_eq!(UnOp::Exp.resolve(DType::U8).unwrap(), (DType::F16, DType::F16));
        assert_eq!(UnOp::Exp.resolve(DType::I16).unwrap(), (DType::F32, DType::F32));
        assert_eq!(UnOp::Exp.resolve(DType::F32).unwrap(), (DType::F32, DType::F32));
        // floor/ceil/trunc keep integers, but rint has no integer loop.
        assert_eq!(UnOp::Floor.resolve(DType::I16).unwrap(), (DType::I16, DType::I16));
        assert_eq!(UnOp::Rint.resolve(DType::I16).unwrap(), (DType::F32, DType::F32));
        // absolute on complex drops to the real component type.
        assert_eq!(UnOp::Absolute.resolve(DType::C64).unwrap(), (DType::C64, DType::F32));
        assert_eq!(
            UnOp::Absolute.resolve(DType::C128).unwrap(),
            (DType::C128, DType::F64)
        );
        // predicates return bool.
        assert_eq!(UnOp::IsNan.resolve(DType::F32).unwrap().1, DType::Bool);
        assert_eq!(UnOp::LogicalNot.resolve(DType::I8).unwrap().1, DType::Bool);
        // bitwise_count returns uint8.
        assert_eq!(UnOp::BitwiseCount.resolve(DType::I64).unwrap().1, DType::U8);
        // invert rejects floats; signbit accepts integers (numpy lifts them).
        assert!(UnOp::Invert.resolve(DType::F64).is_err());
        assert_eq!(UnOp::Signbit.resolve(DType::I32).unwrap(), (DType::F64, DType::Bool));
        assert!(UnOp::Cbrt.resolve(DType::C128).is_err());
    }

    #[test]
    fn float_specials_match_numpy() {
        let a = f64arr(&[0.0, -0.0, 1.0, -1.0, f64::INFINITY, f64::NAN]);
        let s = unary(&a, UnOp::Sqrt).unwrap();
        assert_eq!(get_f(&s, 0), 0.0);
        // np.sqrt(-0.0) is -0.0, not nan.
        assert!(get_f(&s, 1) == 0.0 && get_f(&s, 1).is_sign_negative());
        assert_eq!(get_f(&s, 2), 1.0);
        assert!(get_f(&s, 3).is_nan());
        assert!(get_f(&s, 4).is_infinite());

        let l = unary(&f64arr(&[0.0, -1.0, 1.0]), UnOp::Log).unwrap();
        assert!(get_f(&l, 0) == f64::NEG_INFINITY);
        assert!(get_f(&l, 1).is_nan());
        assert_eq!(get_f(&l, 2), 0.0);
    }

    #[test]
    fn integer_negative_wraps() {
        let a = NdArray::from_scalars(&[Scalar::Int(-128)], DType::I64)
            .unwrap()
            .astype(DType::I8);
        let n = unary(&a, UnOp::Negative).unwrap();
        assert_eq!(n.get_flat(0), Scalar::Int(-128));
    }

    fn c128(v: &[(f64, f64)]) -> NdArray {
        let s: Vec<Scalar> = v
            .iter()
            .map(|&(re, im)| Scalar::Complex(C64v::new(re, im)))
            .collect();
        NdArray::from_scalars(&s, DType::C128).unwrap()
    }

    fn get_c(a: &NdArray, i: usize) -> C64v {
        match a.get_flat(i) {
            Scalar::Complex(z) => z,
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn complex_annex_g_specials_match_numpy() {
        // Probed from numpy 2.5.2, which reaches the platform's C99 complex
        // functions for all of these; `num_complex`'s generic formulas give
        // nan+nanj for every one.
        let inf = f64::INFINITY;
        let z = c128(&[(inf, 0.0)]);
        let pi2 = std::f64::consts::FRAC_PI_2;
        assert_eq!(get_c(&unary(&z, UnOp::Arccos).unwrap(), 0), C64v::new(0.0, -inf));
        assert_eq!(get_c(&unary(&z, UnOp::Arccosh).unwrap(), 0), C64v::new(inf, 0.0));
        assert_eq!(get_c(&unary(&z, UnOp::Arcsin).unwrap(), 0), C64v::new(pi2, inf));
        assert_eq!(get_c(&unary(&z, UnOp::Arcsinh).unwrap(), 0), C64v::new(inf, 0.0));
        assert_eq!(get_c(&unary(&z, UnOp::Arctan).unwrap(), 0), C64v::new(pi2, 0.0));
        assert_eq!(get_c(&unary(&z, UnOp::Arctanh).unwrap(), 0), C64v::new(0.0, pi2));
        assert_eq!(get_c(&unary(&z, UnOp::Cosh).unwrap(), 0), C64v::new(inf, 0.0));
        assert_eq!(get_c(&unary(&z, UnOp::Sinh).unwrap(), 0), C64v::new(inf, 0.0));
        assert_eq!(get_c(&unary(&z, UnOp::Tanh).unwrap(), 0), C64v::new(1.0, 0.0));
        assert_eq!(get_c(&unary(&z, UnOp::Sign).unwrap(), 0), C64v::new(1.0, 0.0));
        // reciprocal(inf+0j) is -0j, not +0j: numpy's loop is `1/d, -r/d`.
        let r = get_c(&unary(&z, UnOp::Reciprocal).unwrap(), 0);
        assert!(r.re == 0.0 && r.im == 0.0 && r.im.is_sign_negative(), "{r:?}");
        // log10 scales clog by log10(e); dividing by ln 10 is off by an ULP.
        let t = c128(&[(f64::MIN_POSITIVE, 0.0)]);
        assert_eq!(
            get_c(&unary(&t, UnOp::Log10).unwrap(), 0).re,
            -307.6526555685888
        );
        // sin(x+0j) keeps the imaginary zero positive.
        let big = c128(&[(f64::MAX, 0.0)]);
        assert!(get_c(&unary(&big, UnOp::Sin).unwrap(), 0).im.is_sign_positive());
        assert_eq!(
            get_c(&unary(&big, UnOp::Tan).unwrap(), 0),
            C64v::new(-0.004962015874444894, 0.0)
        );
    }

    #[test]
    fn float32_sin_cos_are_numpys_own_polynomial() {
        // numpy's `f->f` loop is the Cody-Waite routine, not `sinf`: these two
        // differ from the libm answer in the last bit.
        assert_eq!(np_sin_f32(1.0).to_bits(), 0x3F576AA5);
        assert_eq!(np_cos_f32(2.0).to_bits(), 0xBED51132);
        // Outside the Cody-Waite range numpy falls back to the C library.
        assert_eq!(np_cos_f32(200000.0), 200000.0f32.cos());
        assert!(np_sin_f32(f32::NAN).is_nan());
    }

    #[test]
    fn half_spacing_matches_numpy() {
        // np.spacing(np.float16(-1.0)) is +2**-11: the half loop always steps
        // toward +inf, unlike the float and double ones.
        let a = NdArray::from_scalars(
            &[
                Scalar::Float(1.0),
                Scalar::Float(-1.0),
                Scalar::Float(-2.0),
                Scalar::Float(0.0),
            ],
            DType::F64,
        )
        .unwrap()
        .astype(DType::F16);
        let s = unary(&a, UnOp::Spacing).unwrap();
        let v: Vec<f64> = (0..4)
            .map(|i| match s.get_flat(i) {
                Scalar::Float(f) => f,
                o => panic!("{o:?}"),
            })
            .collect();
        assert_eq!(v, vec![0.0009765625, 0.00048828125, 0.0009765625, 5.960464477539063e-8]);
    }

    #[test]
    fn spacing_matches_numpy() {
        // np.spacing(1.0) == 2.220446049250313e-16
        let a = f64arr(&[1.0, -1.0]);
        let s = unary(&a, UnOp::Spacing).unwrap();
        assert_eq!(get_f(&s, 0), 2.220446049250313e-16);
        assert_eq!(get_f(&s, 1), -2.220446049250313e-16);
    }
}
