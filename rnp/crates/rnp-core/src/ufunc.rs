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
        if self.works_on_ints() {
            let compute = match self {
                // `sign`, `negative`, `square`, `reciprocal` and
                // `conjugate` have no bool loop; `absolute` does.
                Sign | Negative | Square | Reciprocal | Conjugate | Absolute
                    if d.is_bool() =>
                {
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
}

macro_rules! impl_float_un {
    ($t:ty, $pi:expr) => {
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
                    Sin => x.sin(),
                    Cos => x.cos(),
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
        }
    };
}

impl_float_un!(f64, std::f64::consts::PI);
impl_float_un!(f32, std::f32::consts::PI);

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
}

// ---------------------------------------------------------------------------
// Complex transcendentals
// ---------------------------------------------------------------------------

/// The complex unary functions. `C32` computes in double and rounds back,
/// which is what numpy's `nc_*f` wrappers do.
pub trait CplxUn: Element + FpClass {
    fn cu(self, op: UnOp) -> Self;
    fn c_abs(self) -> f64;
}

fn cplx_un_f64(z: C64v, op: UnOp) -> C64v {
    use UnOp::*;
    match op {
        Negative => -z,
        Positive | Conjugate if op == Positive => z,
        Conjugate => C64v::new(z.re, -z.im),
        Absolute => C64v::new(z.norm(), 0.0),
        Sign => {
            // numpy 2.x: sign(z) = z / |z|, with sign(0) == 0.
            if z.re == 0.0 && z.im == 0.0 {
                C64v::new(0.0, 0.0)
            } else if z.re.is_nan() || z.im.is_nan() {
                C64v::new(f64::NAN, f64::NAN)
            } else {
                let m = z.norm();
                C64v::new(z.re / m, z.im / m)
            }
        }
        Rint => C64v::new(z.re.round_ties_even(), z.im.round_ties_even()),
        Square => C64v::new(
            z.re.mul_add(z.re, -(z.im * z.im)),
            z.re.mul_add(z.im, z.im * z.re),
        ),
        Reciprocal => Arith::a_div(C64v::new(1.0, 0.0), z),
        Sqrt => z.sqrt(),
        Exp => z.exp(),
        Exp2 => (z * std::f64::consts::LN_2).exp(),
        Expm1 => z.exp() - C64v::new(1.0, 0.0),
        Log => z.ln(),
        Log2 => z.ln() / std::f64::consts::LN_2,
        Log10 => z.ln() / std::f64::consts::LN_10,
        Log1p => (z + C64v::new(1.0, 0.0)).ln(),
        Sin => z.sin(),
        Cos => z.cos(),
        Tan => z.tan(),
        Arcsin => z.asin(),
        Arccos => z.acos(),
        Arctan => z.atan(),
        Sinh => z.sinh(),
        Cosh => z.cosh(),
        Tanh => z.tanh(),
        Arcsinh => z.asinh(),
        Arccosh => z.acosh(),
        Arctanh => z.atanh(),
        OnesLike => C64v::new(1.0, 0.0),
        other => panic!("complex has no unary loop for {}", other.name()),
    }
}

impl CplxUn for C64v {
    #[inline]
    fn cu(self, op: UnOp) -> Self {
        cplx_un_f64(self, op)
    }
    #[inline]
    fn c_abs(self) -> f64 {
        self.norm()
    }
}

impl CplxUn for C32 {
    #[inline]
    fn cu(self, op: UnOp) -> Self {
        let r = cplx_un_f64(C64v::new(self.re as f64, self.im as f64), op);
        C32::new(r.re as f32, r.im as f32)
    }
    #[inline]
    fn c_abs(self) -> f64 {
        // numpy's `cabsf` works in single precision.
        (self.re as f64).hypot(self.im as f64)
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
        // numpy's spacing reports overflow (at the top of the range) and
        // nothing else.
        unary1_flagged::<T, T, _, _, _>(
            a,
            o,
            n,
            |x| x.fu(UnOp::Spacing),
            |r: T| !r.fp_finite(),
            |x, r| {
                if r.fp_inf() && x.fp_finite() {
                    fpe::OVER
                } else {
                    0
                }
            },
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
                |r: T| !r.fp_finite(),
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
    unary1::<T, T, _>(a, o, n, move |x| x.cu(op));
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

    #[test]
    fn spacing_matches_numpy() {
        // np.spacing(1.0) == 2.220446049250313e-16
        let a = f64arr(&[1.0, -1.0]);
        let s = unary(&a, UnOp::Spacing).unwrap();
        assert_eq!(get_f(&s, 0), 2.220446049250313e-16);
        assert_eq!(get_f(&s, 1), -2.220446049250313e-16);
    }
}
