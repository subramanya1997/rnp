//! Binary element-wise operations with numpy broadcasting and promotion.
//!
//! Every inner loop here is a transcription of the corresponding
//! `numpy/_core/src/umath/loops.c.src` (or `npy_math.c.src`) body, including
//! the signed-zero and NaN tie-breaks, the wrapping integer semantics, and the
//! floating-point error flags numpy would set in the CPU status register.

use crate::array::NdArray;
use crate::dtype::{promote, promote_for_division, DType};
use crate::element::{Element, NpBool, Scalar, C32, C64v, F16};
use crate::error::{Error, Result};
use crate::fpe;
use crate::iter::{broadcast_shapes, broadcast_to, offsets};
use crate::loops::{binary2, binary2_flagged};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // --- M3 ---
    Pow,
    FloatPower,
    FloorDiv,
    Mod,
    Fmod,
    Minimum,
    Maximum,
    Fmin,
    Fmax,
    Arctan2,
    Hypot,
    Copysign,
    Nextafter,
    Logaddexp,
    Logaddexp2,
    Heaviside,
    Ldexp,
    BitAnd,
    BitOr,
    BitXor,
    LShift,
    RShift,
    Gcd,
    Lcm,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
}

impl BinOp {
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    /// True for the ops whose result is a bool array regardless of input.
    pub fn is_logical(self) -> bool {
        matches!(
            self,
            BinOp::LogicalAnd | BinOp::LogicalOr | BinOp::LogicalXor
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "subtract",
            BinOp::Mul => "multiply",
            BinOp::Div => "divide",
            BinOp::Eq => "equal",
            BinOp::Ne => "not_equal",
            BinOp::Lt => "less",
            BinOp::Le => "less_equal",
            BinOp::Gt => "greater",
            BinOp::Ge => "greater_equal",
            BinOp::Pow => "power",
            BinOp::FloatPower => "float_power",
            BinOp::FloorDiv => "floor_divide",
            BinOp::Mod => "remainder",
            BinOp::Fmod => "fmod",
            BinOp::Minimum => "minimum",
            BinOp::Maximum => "maximum",
            BinOp::Fmin => "fmin",
            BinOp::Fmax => "fmax",
            BinOp::Arctan2 => "arctan2",
            BinOp::Hypot => "hypot",
            BinOp::Copysign => "copysign",
            BinOp::Nextafter => "nextafter",
            BinOp::Logaddexp => "logaddexp",
            BinOp::Logaddexp2 => "logaddexp2",
            BinOp::Heaviside => "heaviside",
            BinOp::Ldexp => "ldexp",
            BinOp::BitAnd => "bitwise_and",
            BinOp::BitOr => "bitwise_or",
            BinOp::BitXor => "bitwise_xor",
            BinOp::LShift => "left_shift",
            BinOp::RShift => "right_shift",
            BinOp::Gcd => "gcd",
            BinOp::Lcm => "lcm",
            BinOp::LogicalAnd => "logical_and",
            BinOp::LogicalOr => "logical_or",
            BinOp::LogicalXor => "logical_xor",
        }
    }

    pub fn from_name(s: &str) -> Option<BinOp> {
        Some(match s {
            "add" => BinOp::Add,
            "subtract" => BinOp::Sub,
            "multiply" => BinOp::Mul,
            "divide" | "true_divide" => BinOp::Div,
            "equal" => BinOp::Eq,
            "not_equal" => BinOp::Ne,
            "less" => BinOp::Lt,
            "less_equal" => BinOp::Le,
            "greater" => BinOp::Gt,
            "greater_equal" => BinOp::Ge,
            "power" | "pow" => BinOp::Pow,
            "float_power" => BinOp::FloatPower,
            "floor_divide" => BinOp::FloorDiv,
            "remainder" | "mod" => BinOp::Mod,
            "fmod" => BinOp::Fmod,
            "minimum" => BinOp::Minimum,
            "maximum" => BinOp::Maximum,
            "fmin" => BinOp::Fmin,
            "fmax" => BinOp::Fmax,
            "arctan2" | "atan2" => BinOp::Arctan2,
            "hypot" => BinOp::Hypot,
            "copysign" => BinOp::Copysign,
            "nextafter" => BinOp::Nextafter,
            "logaddexp" => BinOp::Logaddexp,
            "logaddexp2" => BinOp::Logaddexp2,
            "heaviside" => BinOp::Heaviside,
            "ldexp" => BinOp::Ldexp,
            "bitwise_and" => BinOp::BitAnd,
            "bitwise_or" => BinOp::BitOr,
            "bitwise_xor" => BinOp::BitXor,
            "left_shift" | "bitwise_left_shift" => BinOp::LShift,
            "right_shift" | "bitwise_right_shift" => BinOp::RShift,
            "gcd" => BinOp::Gcd,
            "lcm" => BinOp::Lcm,
            "logical_and" => BinOp::LogicalAnd,
            "logical_or" => BinOp::LogicalOr,
            "logical_xor" => BinOp::LogicalXor,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// Per-element traits
// ---------------------------------------------------------------------------

/// Arithmetic over one concrete element type.
pub trait Arith: Element {
    fn a_add(self, o: Self) -> Self;
    fn a_sub(self, o: Self) -> Self;
    fn a_mul(self, o: Self) -> Self;
    fn a_div(self, o: Self) -> Self;
}

/// Ordering/equality over one concrete element type, with numpy's semantics
/// (NaN compares false to everything; complex orders lexicographically).
pub trait Cmp: Element {
    fn c_eq(self, o: Self) -> bool;
    fn c_lt(self, o: Self) -> bool;
    fn c_le(self, o: Self) -> bool;
    /// numpy's truth test, used by the logical ufuncs.
    fn c_truthy(self) -> bool;
}

/// The tie-breaking min/max family, defined for every real and complex type.
pub trait MinMax: Element {
    /// `minimum`: NaN in *either* operand propagates.
    fn m_min(self, o: Self) -> Self;
    fn m_max(self, o: Self) -> Self;
    /// `fmin`/`fmax`: NaN in the second operand is ignored.
    fn m_fmin(self, o: Self) -> Self;
    fn m_fmax(self, o: Self) -> Self;
}

/// Classification used by the floating-point error-flag checks.
pub trait FpClass: Copy {
    fn fp_nan(self) -> bool;
    fn fp_inf(self) -> bool;
    fn fp_zero(self) -> bool;
    #[inline]
    fn fp_finite(self) -> bool {
        !self.fp_nan() && !self.fp_inf()
    }
}

/// A complex element's two components, widened to `f64`.
///
/// Only the flag rules use this; the loops themselves work in the element's own
/// precision.
pub trait CplxParts: FpClass {
    fn c_re(self) -> f64;
    fn c_im(self) -> f64;
}

/// What `v / (+0.0)` raises, for the two independent component divisions
/// numpy's `cdiv` performs when the denominator is a complex zero.
#[inline]
fn zero_div_flag(v: f64) -> u8 {
    if v == 0.0 {
        fpe::INVALID
    } else if v.is_finite() {
        fpe::DIVIDE
    } else {
        // inf/0 is an exact infinity, and nan/0 a quiet NaN: neither signals.
        0
    }
}

/// The conditions numpy reports for complex `x / y`.
///
/// `cdiv` (`npy_math_complex.c.src`) returns `(ar/|br|, ai/|bi|)` for a zero
/// denominator, i.e. two separate divisions by a positive zero -- which is why
/// `(1+0j)/0j` reports `divide by zero` (from `1/0`) *and* `invalid` (from
/// `0/0`), where a single value-level look at the `inf+nanj` result would only
/// see the latter.
#[inline]
fn cplx_div_flags<T: CplxParts>(x: T, y: T, r: T) -> u8 {
    if y.c_re() == 0.0 && y.c_im() == 0.0 {
        return zero_div_flag(x.c_re()) | zero_div_flag(x.c_im());
    }
    scale_flags(x, y, r, true)
}

/// The cheap "might this element have raised something?" test the flagged
/// loop drivers scan with. One compare, so it stays inside the vectorised
/// body. The operands are ignored, so they cost nothing.
#[inline]
fn watch_nonfinite<T: FpClass>(_x: T, _y: T, r: T) -> bool {
    !r.fp_finite()
}

/// As [`watch_nonfinite`], plus the zero results that a product or quotient
/// can only reach by underflowing.
///
/// Returned as a closure over an already-read flag rather than reading
/// `fpe::watch_underflow()` per element: that read is a relaxed atomic load,
/// which LLVM will not hoist out of a loop, and its presence in the body cost
/// `multiply` its vectorisation entirely (1.6x numpy, against `add`'s 0.7x,
/// for the same memory traffic). The state cannot change while the loop runs.
#[inline]
fn watch_scale_fn<T: FpClass>() -> impl Fn(T, T, T) -> bool + Copy + Send + Sync {
    let watching = fpe::watch_underflow();
    move |_x: T, _y: T, r: T| !r.fp_finite() || (r.fp_zero() && watching)
}

macro_rules! impl_fpclass_real {
    ($($t:ty),*) => {$(
        impl FpClass for $t {
            #[inline] fn fp_nan(self) -> bool { self.is_nan() }
            #[inline] fn fp_inf(self) -> bool { self.is_infinite() }
            #[inline] fn fp_zero(self) -> bool { self == 0.0 }
        }
    )*};
}
impl_fpclass_real!(f32, f64);

impl FpClass for F16 {
    #[inline]
    fn fp_nan(self) -> bool {
        self.to_f32().is_nan()
    }
    #[inline]
    fn fp_inf(self) -> bool {
        self.to_f32().is_infinite()
    }
    #[inline]
    fn fp_zero(self) -> bool {
        (self.0 & 0x7FFF) == 0
    }
}

macro_rules! impl_fpclass_cplx {
    ($($t:ty),*) => {$(
        impl FpClass for $t {
            #[inline] fn fp_nan(self) -> bool { self.re.is_nan() || self.im.is_nan() }
            #[inline] fn fp_inf(self) -> bool { self.re.is_infinite() || self.im.is_infinite() }
            #[inline] fn fp_zero(self) -> bool { self.re == 0.0 && self.im == 0.0 }
        }
    )*};
}
impl_fpclass_cplx!(C32, C64v);

// ---------------------------------------------------------------------------
// Integer implementations
// ---------------------------------------------------------------------------

macro_rules! impl_int_ops {
    ($($t:ty),*) => {$(
        impl Arith for $t {
            #[inline] fn a_add(self, o: Self) -> Self { self.wrapping_add(o) }
            #[inline] fn a_sub(self, o: Self) -> Self { self.wrapping_sub(o) }
            #[inline] fn a_mul(self, o: Self) -> Self { self.wrapping_mul(o) }
            // Never reached: integer `divide` promotes to float64 first.
            #[inline] fn a_div(self, o: Self) -> Self {
                if o == 0 { 0 } else { self.wrapping_div(o) }
            }
        }
        impl Cmp for $t {
            #[inline] fn c_eq(self, o: Self) -> bool { self == o }
            #[inline] fn c_lt(self, o: Self) -> bool { self < o }
            #[inline] fn c_le(self, o: Self) -> bool { self <= o }
            #[inline] fn c_truthy(self) -> bool { self != 0 }
        }
        impl MinMax for $t {
            #[inline] fn m_min(self, o: Self) -> Self { if self <= o { self } else { o } }
            #[inline] fn m_max(self, o: Self) -> Self { if self >= o { self } else { o } }
            #[inline] fn m_fmin(self, o: Self) -> Self { if self <= o { self } else { o } }
            #[inline] fn m_fmax(self, o: Self) -> Self { if self >= o { self } else { o } }
        }
    )*};
}

impl_int_ops!(i8, i16, i32, i64, u8, u16, u32, u64);

/// Split out so the `abs`/`< 0` bodies can differ between signed and
/// unsigned types while the rest of [`IntOps`] stays in one macro.
pub trait SignedBits: Copy {
    fn i_absolute(self) -> Self;
    fn i_is_negative(self) -> bool;
}

/// The integer-only ufuncs. Split from `Arith` because bool and the float
/// types have no meaningful implementation.
pub trait IntOps: Element + SignedBits {
    fn i_and(self, o: Self) -> Self;
    fn i_or(self, o: Self) -> Self;
    fn i_xor(self, o: Self) -> Self;
    fn i_shl(self, o: Self) -> Self;
    fn i_shr(self, o: Self) -> Self;
    fn i_gcd(self, o: Self) -> Self;
    fn i_lcm(self, o: Self) -> Self;
    fn i_not(self) -> Self;
    fn i_popcount(self) -> u8;
    /// `floor_divide`; sets the divide flag and returns 0 for a zero divisor.
    fn i_floordiv(self, o: Self) -> Self;
    /// Python-convention `remainder`.
    fn i_mod(self, o: Self) -> Self;
    /// C-convention `fmod`.
    fn i_fmod(self, o: Self) -> Self;
    /// Repeated-squaring integer power (the exponent is known non-negative).
    fn i_pow(self, o: Self) -> Self;
    fn i_negative(self) -> Self;
    fn i_sign(self) -> Self;
    fn i_reciprocal(self) -> Self;
}

macro_rules! impl_int_ufuncs {
    ($($t:ty, $signed:expr, $bits:expr, $u:ty);* $(;)?) => {$(
        impl IntOps for $t {
            #[inline] fn i_and(self, o: Self) -> Self { self & o }
            #[inline] fn i_or(self, o: Self) -> Self { self | o }
            #[inline] fn i_xor(self, o: Self) -> Self { self ^ o }
            // numpy's npy_lshift casts the shift count to size_t first, so a
            // negative count is huge and yields 0, exactly like an over-wide one.
            #[inline] fn i_shl(self, o: Self) -> Self {
                let n = o as i64 as u64;
                if n < $bits { self.wrapping_shl(n as u32) } else { 0 }
            }
            #[inline] #[allow(unused_comparisons)] fn i_shr(self, o: Self) -> Self {
                let n = o as i64 as u64;
                if n < $bits {
                    self.wrapping_shr(n as u32)
                } else if $signed && self < 0 {
                    !0
                } else {
                    0
                }
            }
            // numpy's `npy_gcd`/`npy_lcm` for a signed type take the magnitude
            // of both operands *in the unsigned type* -- `(utype)0 - (utype)a`,
            // so `abs(INT_MIN)` is `2**31` rather than wrapping back to
            // `INT_MIN` -- run the unsigned algorithm, and cast the result
            // back. That is what makes `lcm(INT_MAX, -1)` come out `INT_MAX`
            // and `gcd(INT_MIN, INT_MIN)` come out `INT_MIN`.
            #[inline] #[allow(unused_comparisons)] fn i_gcd(self, o: Self) -> Self {
                // `(utype)0 - (utype)v`, numpy's spelling of "magnitude,
                // without ever negating inside the signed type".
                let uabs = |v: Self| -> $u {
                    if v < 0 { (0 as $u).wrapping_sub(v as $u) } else { v as $u }
                };
                let (mut a, mut b) = (uabs(self), uabs(o));
                while a != 0 {
                    let c = a;
                    a = b % a;
                    b = c;
                }
                b as Self
            }
            #[inline] #[allow(unused_comparisons)] fn i_lcm(self, o: Self) -> Self {
                let uabs = |v: Self| -> $u {
                    if v < 0 { (0 as $u).wrapping_sub(v as $u) } else { v as $u }
                };
                let (ua, ub) = (uabs(self), uabs(o));
                let (mut a, mut b) = (ua, ub);
                while a != 0 {
                    let c = a;
                    a = b % a;
                    b = c;
                }
                if b == 0 { 0 } else { (ua / b).wrapping_mul(ub) as Self }
            }
            #[inline] fn i_not(self) -> Self { !self }
            #[inline] fn i_popcount(self) -> u8 { SignedBits::i_absolute(self).count_ones() as u8 }
            #[inline] #[allow(unused_comparisons)] fn i_floordiv(self, o: Self) -> Self {
                if o == 0 { fpe::raise(fpe::DIVIDE); return 0; }
                // numpy reports `overflow` for the one signed quotient that is
                // not representable, `INT_MIN // -1`.
                if $signed && self == Self::MIN && o == !0 {
                    fpe::raise(fpe::OVER);
                    return Self::MIN;
                }
                let d = self.wrapping_div(o);
                let r = self.wrapping_rem(o);
                if $signed && r != 0 && ((r < 0) != (o < 0)) { d.wrapping_sub(1) } else { d }
            }
            #[inline] #[allow(unused_comparisons)] fn i_mod(self, o: Self) -> Self {
                if o == 0 { fpe::raise(fpe::DIVIDE); return 0; }
                let r = self.wrapping_rem(o);
                if $signed && r != 0 && ((r < 0) != (o < 0)) { r.wrapping_add(o) } else { r }
            }
            #[inline] fn i_fmod(self, o: Self) -> Self {
                if o == 0 { fpe::raise(fpe::DIVIDE); return 0; }
                self.wrapping_rem(o)
            }
            #[inline] fn i_pow(self, o: Self) -> Self {
                let mut base = self;
                let mut exp = o;
                let mut acc: Self = 1;
                while exp > 0 {
                    if exp & 1 == 1 { acc = acc.wrapping_mul(base); }
                    exp >>= 1;
                    if exp > 0 { base = base.wrapping_mul(base); }
                }
                acc
            }
            #[inline] fn i_negative(self) -> Self { self.wrapping_neg() }
            #[inline] fn i_sign(self) -> Self {
                if self > 0 { 1 } else if $signed && SignedBits::i_is_negative(self) { !0 } else { 0 }
            }
            // Probed from numpy: `reciprocal(0)` is -1 for int8/int16 (which
            // compute in `int` and truncate), the type's maximum for
            // int32/int64, and all-ones for every unsigned type.
            #[inline] fn i_reciprocal(self) -> Self {
                if self == 0 {
                    // numpy reports both conditions here.
                    fpe::raise(fpe::DIVIDE | fpe::INVALID);
                    return if $bits <= 16 && $signed { !0 } else { Self::MAX };
                }
                (1 as Self).wrapping_div(self)
            }
        }
    )*};
}

// The signed types need `wrapping_abs`; the unsigned ones are already
// non-negative and `abs` would not compile.
macro_rules! impl_int_signed_bits {
    ($($t:ty),*) => {$(
        impl SignedBits for $t {
            #[inline] fn i_absolute(self) -> Self { self.wrapping_abs() }
            #[inline] fn i_is_negative(self) -> bool { self < 0 }
        }
    )*};
}
macro_rules! impl_int_unsigned_bits {
    ($($t:ty),*) => {$(
        impl SignedBits for $t {
            #[inline] fn i_absolute(self) -> Self { self }
            #[inline] fn i_is_negative(self) -> bool { false }
        }
    )*};
}

impl_int_signed_bits!(i8, i16, i32, i64);
impl_int_unsigned_bits!(u8, u16, u32, u64);

impl_int_ufuncs!(
    i8, true, 8, u8; i16, true, 16, u16; i32, true, 32, u32; i64, true, 64, u64;
    u8, false, 8, u8; u16, false, 16, u16; u32, false, 32, u32; u64, false, 64, u64;
);

// ---------------------------------------------------------------------------
// Float implementations
// ---------------------------------------------------------------------------

macro_rules! impl_float_ops {
    ($($t:ty),*) => {$(
        impl Arith for $t {
            #[inline] fn a_add(self, o: Self) -> Self { self + o }
            #[inline] fn a_sub(self, o: Self) -> Self { self - o }
            #[inline] fn a_mul(self, o: Self) -> Self { self * o }
            #[inline] fn a_div(self, o: Self) -> Self { self / o }
        }
        impl Cmp for $t {
            #[inline] fn c_eq(self, o: Self) -> bool { self == o }
            #[inline] fn c_lt(self, o: Self) -> bool { self < o }
            #[inline] fn c_le(self, o: Self) -> bool { self <= o }
            #[inline] fn c_truthy(self) -> bool { self != 0.0 }
        }
        // `maximum`/`minimum` propagate a NaN from either operand;
        // `fmax`/`fmin` ignore one. Probed from numpy 2.5.2, the signed-zero
        // tie is resolved by sign, not by operand order:
        // `maximum(-0., 0.) == maximum(0., -0.) == +0.`, and both minima
        // give `-0.` -- which the naive `>=`/`<=` form gets wrong one way.
        impl MinMax for $t {
            #[inline] fn m_min(self, o: Self) -> Self {
                if self.is_nan() { self }
                else if o.is_nan() { o }
                else if self < o { self }
                else if o < self { o }
                else if self.is_sign_negative() { self } else { o }
            }
            #[inline] fn m_max(self, o: Self) -> Self {
                if self.is_nan() { self }
                else if o.is_nan() { o }
                else if self > o { self }
                else if o > self { o }
                else if self.is_sign_negative() { o } else { self }
            }
            #[inline] fn m_fmin(self, o: Self) -> Self {
                if self.is_nan() { o }
                else if o.is_nan() { self }
                else if self < o { self }
                else if o < self { o }
                else if self.is_sign_negative() { self } else { o }
            }
            #[inline] fn m_fmax(self, o: Self) -> Self {
                if self.is_nan() { o }
                else if o.is_nan() { self }
                else if self > o { self }
                else if o > self { o }
                else if self.is_sign_negative() { o } else { self }
            }
        }
    )*};
}

impl_float_ops!(f32, f64);

// numpy performs every half-precision operation in `float` and converts the
// result back (`npy_half_add` etc. in `halffloat.c`).
impl Arith for F16 {
    #[inline]
    fn a_add(self, o: Self) -> Self {
        F16::from_f32(self.to_f32() + o.to_f32())
    }
    #[inline]
    fn a_sub(self, o: Self) -> Self {
        F16::from_f32(self.to_f32() - o.to_f32())
    }
    #[inline]
    fn a_mul(self, o: Self) -> Self {
        F16::from_f32(self.to_f32() * o.to_f32())
    }
    #[inline]
    fn a_div(self, o: Self) -> Self {
        F16::from_f32(self.to_f32() / o.to_f32())
    }
}

impl Cmp for F16 {
    #[inline]
    fn c_eq(self, o: Self) -> bool {
        self.to_f32() == o.to_f32()
    }
    #[inline]
    fn c_lt(self, o: Self) -> bool {
        self.to_f32() < o.to_f32()
    }
    #[inline]
    fn c_le(self, o: Self) -> bool {
        self.to_f32() <= o.to_f32()
    }
    #[inline]
    fn c_truthy(self) -> bool {
        (self.0 & 0x7FFF) != 0
    }
}

// The half loops (`loops_half.dispatch.c.src`) are plain
//   `npy_half_le(a, b) ? a : b`
// comparisons with no signed-zero fixup, unlike the float and double ones. So
// where `maximum(-0.f, 0.f)` is `+0.f`, `maximum(np.float16(-0), np.float16(0))`
// is `-0.` -- the *first* operand wins every tie. Probed against numpy 2.5.2.
impl MinMax for F16 {
    #[inline]
    fn m_min(self, o: Self) -> Self {
        if self.is_nan() || o.is_nan() {
            if self.is_nan() {
                self
            } else {
                o
            }
        } else if self.c_le(o) {
            self
        } else {
            o
        }
    }
    #[inline]
    fn m_max(self, o: Self) -> Self {
        if self.is_nan() || o.is_nan() {
            if self.is_nan() {
                self
            } else {
                o
            }
        } else if o.c_le(self) {
            self
        } else {
            o
        }
    }
    #[inline]
    fn m_fmin(self, o: Self) -> Self {
        if o.is_nan() {
            self
        } else if self.is_nan() {
            o
        } else if self.c_le(o) {
            self
        } else {
            o
        }
    }
    #[inline]
    fn m_fmax(self, o: Self) -> Self {
        if o.is_nan() {
            self
        } else if self.is_nan() {
            o
        } else if o.c_le(self) {
            self
        } else {
            o
        }
    }
}

/// The real-float-only ufuncs.
///
/// `F16` implements each by widening to `f32`, computing, and narrowing —
/// which is exactly what numpy's half loops do.
pub trait RealFloat: Element + FpClass {
    fn to_d(self) -> f64;
    fn from_d(v: f64) -> Self;
    fn r_pow(self, o: Self) -> Self;
    fn r_atan2(self, o: Self) -> Self;
    fn r_hypot(self, o: Self) -> Self;
    fn r_copysign(self, o: Self) -> Self;
    fn r_nextafter(self, o: Self) -> Self;
    fn r_spacing(self) -> Self;
    fn r_logaddexp(self, o: Self) -> Self;
    fn r_logaddexp2(self, o: Self) -> Self;
    fn r_heaviside(self, o: Self) -> Self;
    fn r_fmod(self, o: Self) -> Self;
    /// numpy's `npy_divmod`: `(floor_quotient, python_remainder)`.
    fn r_divmod(self, o: Self) -> (Self, Self);
    /// The conditions `npy_divmod` leaves behind -- shared by `floor_divide`
    /// and `divmod`, which both keep its quotient.
    ///
    /// Only two operations inside it can signal: the `a / b` the zero-divisor
    /// branch returns, and the `div = (a - fmod(a, b)) / b` of the main path.
    /// When that quotient overflows, the `div - floor(div)` that snaps it to an
    /// integer is `inf - inf`, so numpy reports `invalid` alongside the
    /// `overflow` -- probed: `np.floor_divide(1.8e308, 2.2e-308)` warns twice.
    fn r_divmod_flags(self, o: Self) -> u8;
    /// The strictly smaller set `remainder` reports.
    ///
    /// `npy_remainder` throws the quotient away, and with it the zero-divisor
    /// branch's division: probed, `np.remainder(1.0, 0.0)` raises nothing at
    /// all where `np.divmod(1.0, 0.0)` raises divide-by-zero. What is left is
    /// the overflow of the main path's quotient.
    fn r_remainder_flags(self, o: Self) -> u8;
    fn r_ldexp(self, n: i64) -> Self;
    fn r_signbit(self) -> bool;
}

/// `nextafter` over the raw bits — the standard formulation, and the one
/// numpy's `npy_nextafter` falls back to.
#[inline]
fn next_after_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return f64::from_bits(1) * y.signum();
    }
    let bits = x.to_bits();
    let next = if (y > x) == (x > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next)
}

#[inline]
fn next_after_f32(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return f32::from_bits(1) * y.signum();
    }
    let bits = x.to_bits();
    let next = if (y > x) == (x > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next)
}

/// `nextafter` for binary16, done on the half bits directly.
#[inline]
fn next_after_f16(x: F16, y: F16) -> F16 {
    let (xf, yf) = (x.to_f32(), y.to_f32());
    if xf.is_nan() || yf.is_nan() {
        return F16::from_f32(xf + yf);
    }
    if xf == yf {
        return y;
    }
    if xf == 0.0 {
        return F16(if yf > 0.0 { 1 } else { 0x8001 });
    }
    // Map to a monotone integer ordering, step, and map back.
    let b = x.0;
    let mono: i32 = if b & 0x8000 != 0 {
        -((b & 0x7FFF) as i32)
    } else {
        (b & 0x7FFF) as i32
    };
    let step = if yf > xf { 1 } else { -1 };
    let m = mono + step;
    let bits = if m < 0 {
        0x8000u16 | ((-m) as u16)
    } else {
        m as u16
    };
    F16(bits)
}

/// numpy's `npy_divmod`, transcribed from `npy_math.c.src`.
#[inline]
fn divmod_f64(a: f64, b: f64) -> (f64, f64) {
    let mut mod_ = a % b;
    if b == 0.0 {
        return (a / b, mod_);
    }
    let mut div = (a - mod_) / b;
    if mod_ != 0.0 {
        if (b < 0.0) != (mod_ < 0.0) {
            mod_ += b;
            div -= 1.0;
        }
    } else {
        mod_ = 0.0_f64.copysign(b);
    }
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 {
            f + 1.0
        } else {
            f
        }
    } else {
        0.0_f64.copysign(a / b)
    };
    (floordiv, mod_)
}

#[inline]
fn divmod_f32(a: f32, b: f32) -> (f32, f32) {
    let mut mod_ = a % b;
    if b == 0.0 {
        return (a / b, mod_);
    }
    let mut div = (a - mod_) / b;
    if mod_ != 0.0 {
        if (b < 0.0) != (mod_ < 0.0) {
            mod_ += b;
            div -= 1.0;
        }
    } else {
        mod_ = 0.0_f32.copysign(b);
    }
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 {
            f + 1.0
        } else {
            f
        }
    } else {
        0.0_f32.copysign(a / b)
    };
    (floordiv, mod_)
}

/// `logaddexp`, transcribed from numpy's `npy_logaddexp`.
#[inline]
fn logaddexp_f64(x: f64, y: f64) -> f64 {
    if x == y {
        // Includes the ±inf == ±inf cases.
        return x + std::f64::consts::LN_2;
    }
    let tmp = x - y;
    if tmp > 0.0 {
        x + (-tmp).exp().ln_1p()
    } else if tmp <= 0.0 {
        y + tmp.exp().ln_1p()
    } else {
        tmp
    }
}

#[inline]
fn logaddexp2_f64(x: f64, y: f64) -> f64 {
    if x == y {
        return x + 1.0;
    }
    let tmp = x - y;
    // numpy computes log2(1+2**-|t|) as log1p(2**-|t|) / ln 2.
    if tmp > 0.0 {
        x + (-tmp).exp2().ln_1p() / std::f64::consts::LN_2
    } else if tmp <= 0.0 {
        y + tmp.exp2().ln_1p() / std::f64::consts::LN_2
    } else {
        tmp
    }
}

impl RealFloat for f64 {
    #[inline]
    fn to_d(self) -> f64 {
        self
    }
    #[inline]
    fn from_d(v: f64) -> Self {
        v
    }
    #[inline]
    fn r_pow(self, o: Self) -> Self {
        self.powf(o)
    }
    #[inline]
    fn r_atan2(self, o: Self) -> Self {
        self.atan2(o)
    }
    #[inline]
    fn r_hypot(self, o: Self) -> Self {
        self.hypot(o)
    }
    #[inline]
    fn r_copysign(self, o: Self) -> Self {
        self.copysign(o)
    }
    #[inline]
    fn r_nextafter(self, o: Self) -> Self {
        next_after_f64(self, o)
    }
    #[inline]
    fn r_spacing(self) -> Self {
        if self.is_nan() {
            return self;
        }
        if self.is_infinite() {
            return f64::NAN;
        }
        // numpy's `spacing(-0.0)` is `+5e-324`, not `-5e-324`.
        if self == 0.0 {
            return f64::from_bits(1);
        }
        next_after_f64(self, f64::INFINITY.copysign(self)) - self
    }
    #[inline]
    fn r_logaddexp(self, o: Self) -> Self {
        logaddexp_f64(self, o)
    }
    #[inline]
    fn r_logaddexp2(self, o: Self) -> Self {
        logaddexp2_f64(self, o)
    }
    #[inline]
    fn r_heaviside(self, o: Self) -> Self {
        if self.is_nan() {
            f64::NAN
        } else if self == 0.0 {
            o
        } else if self < 0.0 {
            0.0
        } else {
            1.0
        }
    }
    #[inline]
    fn r_fmod(self, o: Self) -> Self {
        self % o
    }
    #[inline]
    fn r_divmod(self, o: Self) -> (Self, Self) {
        divmod_f64(self, o)
    }
    #[inline]
    fn r_divmod_flags(self, o: Self) -> u8 {
        if o == 0.0 {
            let q = self / o;
            return if q.is_nan() {
                if self.is_nan() { 0 } else { fpe::INVALID }
            } else if q.is_infinite() && self.is_finite() {
                fpe::DIVIDE
            } else {
                0
            };
        }
        self.r_remainder_flags(o)
    }
    #[inline]
    fn r_remainder_flags(self, o: Self) -> u8 {
        if o != 0.0 && self.is_finite() && o.is_finite() && ((self - self % o) / o).is_infinite() {
            fpe::OVER | fpe::INVALID
        } else {
            0
        }
    }
    #[inline]
    fn r_ldexp(self, n: i64) -> Self {
        scalbn_f64(self, n)
    }
    #[inline]
    fn r_signbit(self) -> bool {
        self.is_sign_negative()
    }
}

/// `x * 2**n` without ever forming an intermediate infinity, so that
/// `ldexp(0.0, 2**31 - 1)` is 0.0 rather than `0 * inf == nan`.
#[inline]
fn scalbn_f64(x: f64, n: i64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mut y = x;
    let mut n = n;
    while n > 1000 {
        y *= 2.0_f64.powi(1000);
        n -= 1000;
        if !y.is_finite() {
            return y;
        }
    }
    while n < -1000 {
        y *= 2.0_f64.powi(-1000);
        n += 1000;
        if y == 0.0 {
            return y;
        }
    }
    y * 2.0_f64.powi(n as i32)
}

impl RealFloat for f32 {
    #[inline]
    fn to_d(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_d(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn r_pow(self, o: Self) -> Self {
        self.powf(o)
    }
    #[inline]
    fn r_atan2(self, o: Self) -> Self {
        self.atan2(o)
    }
    #[inline]
    fn r_hypot(self, o: Self) -> Self {
        self.hypot(o)
    }
    #[inline]
    fn r_copysign(self, o: Self) -> Self {
        self.copysign(o)
    }
    #[inline]
    fn r_nextafter(self, o: Self) -> Self {
        next_after_f32(self, o)
    }
    #[inline]
    fn r_spacing(self) -> Self {
        if self.is_nan() {
            return self;
        }
        if self.is_infinite() {
            return f32::NAN;
        }
        if self == 0.0 {
            return f32::from_bits(1);
        }
        next_after_f32(self, f32::INFINITY.copysign(self)) - self
    }
    #[inline]
    fn r_logaddexp(self, o: Self) -> Self {
        if self == o {
            return self + std::f32::consts::LN_2;
        }
        let tmp = self - o;
        if tmp > 0.0 {
            self + (-tmp).exp().ln_1p()
        } else if tmp <= 0.0 {
            o + tmp.exp().ln_1p()
        } else {
            tmp
        }
    }
    #[inline]
    fn r_logaddexp2(self, o: Self) -> Self {
        if self == o {
            return self + 1.0;
        }
        let tmp = self - o;
        if tmp > 0.0 {
            self + (-tmp).exp2().ln_1p() / std::f32::consts::LN_2
        } else if tmp <= 0.0 {
            o + tmp.exp2().ln_1p() / std::f32::consts::LN_2
        } else {
            tmp
        }
    }
    #[inline]
    fn r_heaviside(self, o: Self) -> Self {
        if self.is_nan() {
            f32::NAN
        } else if self == 0.0 {
            o
        } else if self < 0.0 {
            0.0
        } else {
            1.0
        }
    }
    #[inline]
    fn r_fmod(self, o: Self) -> Self {
        self % o
    }
    #[inline]
    fn r_divmod(self, o: Self) -> (Self, Self) {
        divmod_f32(self, o)
    }
    #[inline]
    fn r_divmod_flags(self, o: Self) -> u8 {
        if o == 0.0 {
            let q = self / o;
            return if q.is_nan() {
                if self.is_nan() { 0 } else { fpe::INVALID }
            } else if q.is_infinite() && self.is_finite() {
                fpe::DIVIDE
            } else {
                0
            };
        }
        self.r_remainder_flags(o)
    }
    #[inline]
    fn r_remainder_flags(self, o: Self) -> u8 {
        if o != 0.0 && self.is_finite() && o.is_finite() && ((self - self % o) / o).is_infinite() {
            fpe::OVER | fpe::INVALID
        } else {
            0
        }
    }
    #[inline]
    fn r_ldexp(self, n: i64) -> Self {
        scalbn_f64(self as f64, n) as f32
    }
    #[inline]
    fn r_signbit(self) -> bool {
        self.is_sign_negative()
    }
}

impl RealFloat for F16 {
    #[inline]
    fn to_d(self) -> f64 {
        self.to_f64()
    }
    #[inline]
    fn from_d(v: f64) -> Self {
        F16::from_f64(v)
    }
    #[inline]
    fn r_pow(self, o: Self) -> Self {
        F16::from_f32(self.to_f32().powf(o.to_f32()))
    }
    #[inline]
    fn r_atan2(self, o: Self) -> Self {
        F16::from_f32(self.to_f32().atan2(o.to_f32()))
    }
    #[inline]
    fn r_hypot(self, o: Self) -> Self {
        F16::from_f32(self.to_f32().hypot(o.to_f32()))
    }
    #[inline]
    fn r_copysign(self, o: Self) -> Self {
        F16((self.0 & 0x7FFF) | (o.0 & 0x8000))
    }
    #[inline]
    fn r_nextafter(self, o: Self) -> Self {
        next_after_f16(self, o)
    }
    /// `npy_half_spacing` (`npymath/halffloat.cpp`), transcribed on the half
    /// bits. It is *not* `nextafter(x, sign(x)*inf) - x`: numpy's half spacing
    /// always steps toward `+inf`, so `spacing(-1.0)` is `+2**-11` where the
    /// float and double loops give `-2**-53`-style negatives.
    #[inline]
    fn r_spacing(self) -> Self {
        let h = self.0;
        let h_exp = h & 0x7C00;
        let h_sig = h & 0x03FF;
        if h_exp == 0x7C00 {
            // infinite or NaN
            F16::from_f32(f32::NAN)
        } else if h == 0x7BFF {
            F16(0x7C00)
        } else if h & 0x8000 != 0 && h_sig == 0 {
            // Negative powers of two: the gap below is half the gap above.
            if h_exp > 0x2C00 {
                F16(h_exp - 0x2C00)
            } else if h_exp > 0x0400 {
                F16(1 << ((h_exp >> 10) - 2))
            } else {
                F16(0x0001)
            }
        } else if h_exp > 0x2800 {
            F16(h_exp - 0x2800)
        } else if h_exp > 0x0400 {
            F16(1 << ((h_exp >> 10) - 1))
        } else {
            F16(0x0001)
        }
    }
    #[inline]
    fn r_logaddexp(self, o: Self) -> Self {
        F16::from_f32(self.to_f32().r_logaddexp(o.to_f32()))
    }
    #[inline]
    fn r_logaddexp2(self, o: Self) -> Self {
        F16::from_f32(self.to_f32().r_logaddexp2(o.to_f32()))
    }
    #[inline]
    fn r_heaviside(self, o: Self) -> Self {
        F16::from_f32(self.to_f32().r_heaviside(o.to_f32()))
    }
    #[inline]
    fn r_fmod(self, o: Self) -> Self {
        F16::from_f32(self.to_f32() % o.to_f32())
    }
    #[inline]
    fn r_divmod(self, o: Self) -> (Self, Self) {
        let (d, m) = divmod_f32(self.to_f32(), o.to_f32());
        (F16::from_f32(d), F16::from_f32(m))
    }
    /// The half loops widen to `float` and call `npy_floor_dividef`, so the
    /// intermediate quotient has ~8 orders of magnitude of headroom and never
    /// overflows for half operands. Only the narrowing of the result back to
    /// half can, which is the ordinary value-level rule.
    #[inline]
    fn r_divmod_flags(self, o: Self) -> u8 {
        float_flags(self, o, self.r_divmod(o).0, true)
    }
    /// ...and therefore `remainder` on half reports nothing at all, which is
    /// what numpy does: `np.remainder(np.float16(1), np.float16(0))` is a
    /// silent NaN.
    #[inline]
    fn r_remainder_flags(self, _o: Self) -> u8 {
        0
    }
    #[inline]
    fn r_ldexp(self, n: i64) -> Self {
        F16::from_f32(self.to_f32().r_ldexp(n))
    }
    #[inline]
    fn r_signbit(self) -> bool {
        self.0 & 0x8000 != 0
    }
}

/// Smith's algorithm, transcribed from numpy's `@TYPE@_divide` inner loop in
/// `umath/loops.c.src`.
///
/// Two things force a hand-written loop rather than `num_complex`'s `Div`:
/// that one overflows on operands like `1e-200 + 1e-200j` and returns
/// `nan + nanj` instead of `inf + nanj` when dividing by zero. The `mul_add`
/// calls reproduce the FMA contraction numpy's C loop is compiled with on
/// aarch64 -- without them the results differ from numpy by an ULP.
macro_rules! complex_div {
    ($t:ty, $f:ty) => {
        #[inline]
        fn a_div(self, o: Self) -> Self {
            let (ar, ai, br, bi) = (self.re, self.im, o.re, o.im);
            let (br_abs, bi_abs) = (br.abs(), bi.abs());
            if br_abs >= bi_abs {
                if br_abs == 0.0 && bi_abs == 0.0 {
                    // Division by zero yields a complex inf or nan; numpy
                    // divides by the *absolute* values here.
                    return <$t>::new(ar / br_abs, ai / bi_abs);
                }
                let rat = bi / br;
                let scl = (1.0 as $f) / bi.mul_add(rat, br);
                <$t>::new(ai.mul_add(rat, ar) * scl, (-ar).mul_add(rat, ai) * scl)
            } else {
                let rat = br / bi;
                let scl = (1.0 as $f) / br.mul_add(rat, bi);
                <$t>::new(ar.mul_add(rat, ai) * scl, ai.mul_add(rat, -ar) * scl)
            }
        }
    };
}

macro_rules! impl_complex_ops {
    ($($t:ty, $f:ty);*) => {$(
        impl Arith for $t {
            #[inline] fn a_add(self, o: Self) -> Self { self + o }
            #[inline] fn a_sub(self, o: Self) -> Self { self - o }
            // numpy's `@TYPE@_multiply` inner loop is
            //   out.re = a.re*b.re - a.im*b.im;
            //   out.im = a.re*b.im + a.im*b.re;
            // which clang contracts into an FMA per statement on aarch64.
            // Plain `*` on num_complex rounds twice and differs by an ULP.
            #[inline] fn a_mul(self, o: Self) -> Self {
                <$t>::new(
                    self.re.mul_add(o.re, -(self.im * o.im)),
                    self.re.mul_add(o.im, self.im * o.re),
                )
            }
            complex_div!($t, $f);
        }
        // numpy orders complex lexicographically: real part first, then imag
        // -- but the `CLT`/`CLE` macros in `umath/loops.c.src` refuse to answer
        // from the real parts alone when either *imaginary* part is a NaN, so
        // `(1+nanj) < (2+0j)` is false where the naive lexicographic form says
        // true.
        impl Cmp for $t {
            #[inline] fn c_eq(self, o: Self) -> bool { self.re == o.re && self.im == o.im }
            #[inline] fn c_lt(self, o: Self) -> bool {
                (self.re < o.re && !self.im.is_nan() && !o.im.is_nan())
                    || (self.re == o.re && self.im < o.im)
            }
            #[inline] fn c_le(self, o: Self) -> bool {
                (self.re < o.re && !self.im.is_nan() && !o.im.is_nan())
                    || (self.re == o.re && self.im <= o.im)
            }
            #[inline] fn c_truthy(self) -> bool { self.re != 0.0 || self.im != 0.0 }
        }
        // `C@TYPE@_maximum` keeps its first operand when that operand has a NaN
        // component or already compares `>=`; `C@TYPE@_fmax` instead lets the
        // *second* operand's NaN be the one that is skipped over.
        impl MinMax for $t {
            #[inline] fn m_min(self, o: Self) -> Self {
                if self.fp_nan() || self.c_le(o) { self } else { o }
            }
            #[inline] fn m_max(self, o: Self) -> Self {
                if self.fp_nan() || o.c_le(self) { self } else { o }
            }
            #[inline] fn m_fmin(self, o: Self) -> Self {
                if o.fp_nan() || self.c_le(o) { self } else { o }
            }
            #[inline] fn m_fmax(self, o: Self) -> Self {
                if o.fp_nan() || o.c_le(self) { self } else { o }
            }
        }
        // The components, so the shared complex flag rules can reach them.
        impl CplxParts for $t {
            #[inline] fn c_re(self) -> f64 { self.re as f64 }
            #[inline] fn c_im(self) -> f64 { self.im as f64 }
        }
    )*};
}

impl_complex_ops!(C32, f32; C64v, f64);

// numpy's bool `add` is logical or and `mul` is logical and; `subtract` is
// rejected before it reaches an inner loop.
impl Arith for NpBool {
    #[inline]
    fn a_add(self, o: Self) -> Self {
        NpBool::new(self.get() || o.get())
    }
    #[inline]
    fn a_sub(self, o: Self) -> Self {
        NpBool::new(self.get() ^ o.get())
    }
    #[inline]
    fn a_mul(self, o: Self) -> Self {
        NpBool::new(self.get() && o.get())
    }
    #[inline]
    fn a_div(self, o: Self) -> Self {
        NpBool::new(o.get() && self.get())
    }
}

impl Cmp for NpBool {
    #[inline]
    fn c_eq(self, o: Self) -> bool {
        self.get() == o.get()
    }
    #[inline]
    fn c_lt(self, o: Self) -> bool {
        !self.get() && o.get()
    }
    #[inline]
    fn c_le(self, o: Self) -> bool {
        !self.get() || o.get()
    }
    #[inline]
    fn c_truthy(self) -> bool {
        self.get()
    }
}

impl MinMax for NpBool {
    #[inline]
    fn m_min(self, o: Self) -> Self {
        NpBool::new(self.get() && o.get())
    }
    #[inline]
    fn m_max(self, o: Self) -> Self {
        NpBool::new(self.get() || o.get())
    }
    #[inline]
    fn m_fmin(self, o: Self) -> Self {
        self.m_min(o)
    }
    #[inline]
    fn m_fmax(self, o: Self) -> Self {
        self.m_max(o)
    }
}

// ---------------------------------------------------------------------------
// Flexible (S/U) comparisons
// ---------------------------------------------------------------------------

/// Comparisons between flexible (`S`/`U`) arrays.
///
/// Only the comparison ufuncs are defined for strings; arithmetic on them is a
/// later milestone. Both operands are compared on their *logical* value, i.e.
/// with the trailing NUL padding numpy uses stripped off.
pub fn binary_flexible(a: &NdArray, b: &NdArray, op: BinOp) -> Result<NdArray> {
    if !op.is_comparison() {
        return Err(Error::NotImplemented(format!(
            "ufunc '{}' is not supported for dtypes ({}, {})",
            op.name(),
            a.dtype(),
            b.dtype()
        )));
    }
    if a.dtype().category() != b.dtype().category() {
        return Err(Error::NotImplemented(format!(
            "comparing {} with {} is not implemented yet",
            a.dtype(), b.dtype()
        )));
    }
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let av = broadcast_to(a, &out_shape)?;
    let bv = broadcast_to(b, &out_shape)?;
    let out = NdArray::empty(out_shape, DType::Bool)?;

    let a_offs: Vec<isize> = offsets(&av.shape, &av.strides, av.byte_offset).collect();
    let b_offs: Vec<isize> = offsets(&bv.shape, &bv.strides, bv.byte_offset).collect();
    let o_offs: Vec<isize> = offsets(&out.shape, &out.strides, out.byte_offset).collect();

    for k in 0..o_offs.len() {
        let x = logical_bytes(&av, a_offs[k]);
        let y = logical_bytes(&bv, b_offs[k]);
        let ord = x.cmp(&y);
        let r = match op {
            BinOp::Eq => ord.is_eq(),
            BinOp::Ne => ord.is_ne(),
            BinOp::Lt => ord.is_lt(),
            BinOp::Le => ord.is_le(),
            BinOp::Gt => ord.is_gt(),
            _ => ord.is_ge(),
        };
        out.write_at(o_offs[k], Scalar::Bool(r));
    }
    Ok(out)
}

/// An element's code units with numpy's trailing-NUL padding removed, so
/// that `S3` `b"ab\0"` equals `S5` `b"ab\0\0\0"`. `U` elements are decoded
/// from UCS4 so that ordering compares code points, not their little-endian
/// bytes.
pub(crate) fn logical_bytes(arr: &NdArray, off: isize) -> Vec<u32> {
    let raw = arr.element_bytes_at(off);
    let raw: &[u8] = &raw;
    let mut v: Vec<u32> = match arr.dtype() {
        DType::Str(_) => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => raw.iter().map(|&b| b as u32).collect(),
    };
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

// ---------------------------------------------------------------------------
// Type resolution
// ---------------------------------------------------------------------------

/// Promote `bool` to `int8`: the ops whose loop table starts at `bb->b`.
fn no_bool(d: DType) -> DType {
    if d == DType::Bool {
        DType::I8
    } else {
        d
    }
}

/// The smallest float type that can hold `d` -- numpy picks the first
/// matching loop from its `ee->e`, `ff->f`, `dd->d` table, so
/// `np.hypot(np.uint8(1), np.uint8(1))` comes out float16.
fn to_float(d: DType) -> DType {
    match d {
        DType::F16 | DType::F32 | DType::F64 | DType::C64 | DType::C128 => d,
        _ => crate::dtype::promote(d, DType::F16),
    }
}

/// numpy's message when no loop matches the input types.
fn unsupported(op: BinOp) -> Error {
    Error::TypeError(format!(
        "ufunc '{}' not supported for the input types, and the inputs could \
         not be safely coerced to any supported types according to the \
         casting rule ''safe''",
        op.name()
    ))
}

/// The dtype the inner loop runs in, and the dtype of the result.
pub fn result_dtypes(a: DType, b: DType, op: BinOp) -> Result<(DType, DType)> {
    use BinOp::*;
    if op.is_comparison() {
        let compute = promote(a, b);
        return Ok((compute, DType::Bool));
    }
    if op.is_logical() {
        return Ok((promote(a, b), DType::Bool));
    }
    match op {
        Div => {
            let d = promote_for_division(a, b);
            Ok((d, d))
        }
        Sub if a.is_bool() && b.is_bool() => Err(Error::TypeError(
            "numpy boolean subtract, the `-` operator, is not supported, use \
             the bitwise_xor, the `^` operator, or the logical_xor function \
             instead."
                .into(),
        )),
        Add | Sub | Mul | Minimum | Maximum | Fmin | Fmax => {
            let d = promote(a, b);
            Ok((d, d))
        }
        Pow => {
            let d = no_bool(promote(a, b));
            Ok((d, d))
        }
        FloatPower => {
            let d = promote(a, b);
            let d = if d.is_complex() { DType::C128 } else { DType::F64 };
            Ok((d, d))
        }
        FloorDiv | Mod | Fmod => {
            let d = no_bool(promote(a, b));
            if d.is_complex() {
                return Err(unsupported(op));
            }
            Ok((d, d))
        }
        Arctan2 | Hypot | Copysign | Nextafter | Logaddexp | Logaddexp2 | Heaviside => {
            let d = to_float(promote(a, b));
            if d.is_complex() {
                return Err(unsupported(op));
            }
            Ok((d, d))
        }
        Ldexp => {
            // numpy's loop table is `ei->e, fi->f, di->d, gi->g` (plus the
            // `l` variants): the *first* operand picks the float loop and the
            // second must cast safely to an integer. bool counts as an
            // integer on the right (`?` casts safely to `i`) and lifts to
            // float16 on the left.
            let d = to_float(a);
            if d.is_complex() || !(b.is_integer() || b.is_bool()) {
                return Err(unsupported(op));
            }
            Ok((d, d))
        }
        BitAnd | BitOr | BitXor => {
            let d = promote(a, b);
            if !(d.is_integer() || d.is_bool()) {
                return Err(unsupported(op));
            }
            Ok((d, d))
        }
        LShift | RShift => {
            let d = no_bool(promote(a, b));
            if !d.is_integer() {
                return Err(unsupported(op));
            }
            Ok((d, d))
        }
        Gcd | Lcm => {
            // `gcd`/`lcm` use numpy's `PyUFunc_SimpleUniformOperationTypeResolver`,
            // which resolves one common dtype and then demands an *exact*
            // loop for it. There is no `??->?` entry, so bool is rejected
            // rather than lifted to int8, and the failure surfaces as
            // `UFuncTypeError` rather than the legacy `TypeError`.
            let d = promote(a, b);
            if !d.is_integer() {
                let n = d.name();
                return Err(crate::error::ufunc_no_loop(op.name(), &[&n, &n]));
            }
            Ok((d, d))
        }
        _ => unreachable!("handled above"),
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Element-wise binary op with numpy broadcasting + promotion.
pub fn binary(a: &NdArray, b: &NdArray, op: BinOp) -> Result<NdArray> {
    Ok(binary_multi(a, b, op)?.0)
}

/// As [`binary`], but `divmod` also yields the remainder.
pub fn binary_multi(a: &NdArray, b: &NdArray, op: BinOp) -> Result<(NdArray, Option<NdArray>)> {
    // Byte-swapped storage is normalised to the host order *here*, once per
    // operand, and never inside a loop: everything below sees native bytes.
    // The conversion lives in a cold, never-inlined function so that the
    // kernel-carrying `binary_multi_native` stays non-recursive and keeps
    // vectorising.
    if !a.is_native() || !b.is_native() {
        return binary_multi_swapped(a, b, op);
    }
    binary_multi_native(a, b, op)
}

#[cold]
#[inline(never)]
fn binary_multi_swapped(
    a: &NdArray,
    b: &NdArray,
    op: BinOp,
) -> Result<(NdArray, Option<NdArray>)> {
    binary_multi_native(&a.to_native(), &b.to_native(), op)
}

fn binary_multi_native(
    a: &NdArray,
    b: &NdArray,
    op: BinOp,
) -> Result<(NdArray, Option<NdArray>)> {
    // numpy has dedicated `qQ->?` / `Qq->?` comparison loops so that
    // `uint64(2**64-1) < int64(-1)` is answered exactly rather than through
    // the float64 both types would otherwise promote to.
    if op.is_comparison()
        && ((a.dtype() == DType::U64 && b.dtype().is_signed())
            || (b.dtype() == DType::U64 && a.dtype().is_signed()))
    {
        return Ok((compare_mixed_64(a, b, op)?, None));
    }
    if a.dtype().is_flexible() || b.dtype().is_flexible() {
        return Ok((binary_flexible(a, b, op)?, None));
    }
    if a.dtype().is_object() || b.dtype().is_object() {
        return Err(Error::NotImplemented(format!(
            "ufunc '{}' on object arrays is not implemented",
            op.name()
        )));
    }
    let (compute, out_dtype) = result_dtypes(a.dtype(), b.dtype(), op)?;
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;

    // Cast before broadcasting so the (cheap) cast copy stays small.
    let a_cast;
    let a_ref = if a.dtype() == compute {
        a
    } else {
        a_cast = a.astype(compute);
        &a_cast
    };
    let b_cast;
    let b_ref = if b.dtype() == compute {
        b
    } else {
        b_cast = b.astype(compute);
        &b_cast
    };

    // numpy raises rather than producing a wrong integer for a negative
    // exponent, and it checks the whole operand up front.
    if op == BinOp::Pow && compute.is_integer() && compute.is_signed() {
        if any_negative(b_ref) {
            return Err(Error::ValueError(
                "Integers to negative integer powers are not allowed.".into(),
            ));
        }
    }

    let av = broadcast_to(a_ref, &out_shape)?;
    let bv = broadcast_to(b_ref, &out_shape)?;
    let out = NdArray::empty(out_shape, out_dtype)?;

    crate::dispatch_dtype!(compute, T, {
        run_binary::<T>(&av, &bv, &out, op);
    });
    Ok((out, None))
}

/// Exact comparison between a `uint64` and a signed integer array, done in
/// 128 bits so no value is rounded on its way through a common type.
fn compare_mixed_64(a: &NdArray, b: &NdArray, op: BinOp) -> Result<NdArray> {
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let av = broadcast_to(&a.astype(a.dtype()), &out_shape)?;
    let bv = broadcast_to(&b.astype(b.dtype()), &out_shape)?;
    let out = NdArray::empty(out_shape, DType::Bool)?;
    let ao: Vec<isize> = offsets(&av.shape, &av.strides, av.byte_offset).collect();
    let bo: Vec<isize> = offsets(&bv.shape, &bv.strides, bv.byte_offset).collect();
    let oo: Vec<isize> = offsets(&out.shape, &out.strides, out.byte_offset).collect();
    for k in 0..oo.len() {
        let x = as_i128(av.read_at(ao[k]));
        let y = as_i128(bv.read_at(bo[k]));
        let r = match op {
            BinOp::Eq => x == y,
            BinOp::Ne => x != y,
            BinOp::Lt => x < y,
            BinOp::Le => x <= y,
            BinOp::Gt => x > y,
            _ => x >= y,
        };
        out.write_at(oo[k], Scalar::Bool(r));
    }
    Ok(out)
}

/// `np.divmod`: both outputs in one pass.
pub fn divmod(a: &NdArray, b: &NdArray) -> Result<(NdArray, NdArray)> {
    let (compute, _) = result_dtypes(a.dtype(), b.dtype(), BinOp::Mod)?;
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let ac = a.astype(compute);
    let bc = b.astype(compute);
    let av = broadcast_to(&ac, &out_shape)?;
    let bv = broadcast_to(&bc, &out_shape)?;
    let q = NdArray::empty(out_shape.clone(), compute)?;
    let r = NdArray::empty(out_shape, compute)?;
    let n = q.size();
    crate::dispatch_dtype!(compute, T, {
        // SAFETY: `q`/`r` are freshly allocated contiguous arrays of n T's.
        let (pq, pr) = unsafe {
            (
                q.buffer.as_mut_ptr() as *mut T,
                r.buffer.as_mut_ptr() as *mut T,
            )
        };
        divmod_loop::<T>(&av, &bv, pq, pr, n, compute);
    });
    Ok((q, r))
}

fn divmod_loop<T: Element>(
    a: &NdArray,
    b: &NdArray,
    q: *mut T,
    r: *mut T,
    n: usize,
    dt: DType,
) {
    // The two outputs share a walk, so this goes through the odometer rather
    // than the specialised drivers.
    let ia: Vec<isize> = offsets(&a.shape, &a.strides, a.byte_offset).collect();
    let ib: Vec<isize> = offsets(&b.shape, &b.strides, b.byte_offset).collect();
    for i in 0..n {
        let x = a.read_at(ia[i]);
        let y = b.read_at(ib[i]);
        let (qq, rr) = scalar_divmod(x, y, dt);
        // SAFETY: i < n and both outputs hold n elements of T.
        unsafe {
            *q.add(i) = T::from_scalar(qq);
            *r.add(i) = T::from_scalar(rr);
        }
    }
}

/// Scalar `divmod` over the `Scalar` value type, shared by the array loop and
/// the scalar-type path on the Python side.
pub fn scalar_divmod(x: Scalar, y: Scalar, dt: DType) -> (Scalar, Scalar) {
    if dt.is_integer() {
        let (xi, yi) = (as_i128(x), as_i128(y));
        if yi == 0 {
            fpe::raise(fpe::DIVIDE);
            return (Scalar::Int(0), Scalar::Int(0));
        }
        if dt.is_signed() {
            // The one signed quotient that does not fit: numpy reports
            // `overflow` and hands back the wrapped `INT_MIN`.
            let min = -(1i128 << (dt.itemsize() * 8 - 1));
            if xi == min && yi == -1 {
                fpe::raise(fpe::OVER);
                return (Scalar::Int(min as i64), Scalar::Int(0));
            }
        }
        let mut d = xi / yi;
        let mut m = xi % yi;
        if m != 0 && ((m < 0) != (yi < 0)) {
            m += yi;
            d -= 1;
        }
        return if dt.is_signed() {
            (Scalar::Int(d as i64), Scalar::Int(m as i64))
        } else {
            (Scalar::Uint(d as u64), Scalar::Uint(m as u64))
        };
    }
    // The narrower float loops compute in their own precision, and the error
    // conditions differ accordingly -- `divmod(float32_max, float32_tiny)`
    // overflows where the same values in double do not.
    let (xf, yf) = (as_f64(x), as_f64(y));
    match dt {
        DType::F32 => {
            let (a, b) = (xf as f32, yf as f32);
            let (d, m) = divmod_f32(a, b);
            fpe::raise(a.r_divmod_flags(b));
            (Scalar::Float(d as f64), Scalar::Float(m as f64))
        }
        DType::F16 => {
            let (a, b) = (F16::from_f64(xf), F16::from_f64(yf));
            let (d, m) = a.r_divmod(b);
            fpe::raise(a.r_divmod_flags(b));
            (Scalar::Float(d.to_f64()), Scalar::Float(m.to_f64()))
        }
        _ => {
            let (d, m) = divmod_f64(xf, yf);
            fpe::raise(xf.r_divmod_flags(yf));
            (Scalar::Float(d), Scalar::Float(m))
        }
    }
}

fn as_i128(s: Scalar) -> i128 {
    match s {
        Scalar::Bool(b) => b as i128,
        Scalar::Int(i) => i as i128,
        Scalar::Uint(u) => u as i128,
        Scalar::Float(f) => f as i128,
        Scalar::Complex(c) => c.re as i128,
    }
}

fn as_f64(s: Scalar) -> f64 {
    match s {
        Scalar::Bool(b) => b as u8 as f64,
        Scalar::Int(i) => i as f64,
        Scalar::Uint(u) => u as f64,
        Scalar::Float(f) => f,
        Scalar::Complex(c) => c.re,
    }
}

fn any_negative(a: &NdArray) -> bool {
    for off in offsets(&a.shape, &a.strides, a.byte_offset) {
        if let Scalar::Int(i) = a.read_at(off) {
            if i < 0 {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Per-category loop dispatch
// ---------------------------------------------------------------------------

/// Route one compute dtype's loop. The op sets that a type does not support
/// are unreachable: `result_dtypes` rejects them first.
fn run_binary<T: Element + Send + Sync>(a: &NdArray, b: &NdArray, out: &NdArray, op: BinOp)
where
    T: Arith + Cmp + MinMax,
{
    let n = out.size();
    if op.is_comparison() || op.is_logical() {
        // SAFETY: `out` is a freshly allocated contiguous bool array of n bytes.
        let o = unsafe { out.buffer.as_mut_ptr() };
        // The complex ordered comparisons are the only comparison loops that
        // report anything; see `run_cplx_cmp`.
        if T::DTYPE.is_complex() && matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
            match T::DTYPE {
                DType::C64 => run_cplx_cmp::<C32>(a, b, o, n, op),
                _ => run_cplx_cmp::<C64v>(a, b, o, n, op),
            }
            return;
        }
        match op {
            BinOp::Eq => binary2::<T, u8, _>(a, b, o, n, |x, y| x.c_eq(y) as u8),
            BinOp::Ne => binary2::<T, u8, _>(a, b, o, n, |x, y| !x.c_eq(y) as u8),
            BinOp::Lt => binary2::<T, u8, _>(a, b, o, n, |x, y| x.c_lt(y) as u8),
            BinOp::Le => binary2::<T, u8, _>(a, b, o, n, |x, y| x.c_le(y) as u8),
            BinOp::Gt => binary2::<T, u8, _>(a, b, o, n, |x, y| y.c_lt(x) as u8),
            BinOp::Ge => binary2::<T, u8, _>(a, b, o, n, |x, y| y.c_le(x) as u8),
            BinOp::LogicalAnd => {
                binary2::<T, u8, _>(a, b, o, n, |x, y| (x.c_truthy() && y.c_truthy()) as u8)
            }
            BinOp::LogicalOr => {
                binary2::<T, u8, _>(a, b, o, n, |x, y| (x.c_truthy() || y.c_truthy()) as u8)
            }
            BinOp::LogicalXor => {
                binary2::<T, u8, _>(a, b, o, n, |x, y| (x.c_truthy() != y.c_truthy()) as u8)
            }
            _ => unreachable!(),
        }
        return;
    }
    // SAFETY: `out` was freshly allocated C-contiguous with `n` elements of T.
    let o = unsafe { out.buffer.as_mut_ptr() as *mut T };
    match op {
        // The four arithmetic ops are the hot ones. Integers and bool cannot
        // raise a floating-point condition, so they keep the unflagged (and
        // therefore vectorisable) driver; the inexact types go through the
        // flagged one so `1.0/0.0` reports `divide by zero`.
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
            if T::DTYPE.is_exact() {
                match op {
                    BinOp::Add => binary2::<T, T, _>(a, b, o, n, |x, y| x.a_add(y)),
                    BinOp::Sub => binary2::<T, T, _>(a, b, o, n, |x, y| x.a_sub(y)),
                    BinOp::Mul => binary2::<T, T, _>(a, b, o, n, |x, y| x.a_mul(y)),
                    _ => binary2::<T, T, _>(a, b, o, n, |x, y| x.a_div(y)),
                }
            } else {
                run_inexact_arith::<T>(a, b, o, n, op)
            }
        }
        BinOp::Minimum => binary2::<T, T, _>(a, b, o, n, |x, y| x.m_min(y)),
        BinOp::Maximum => binary2::<T, T, _>(a, b, o, n, |x, y| x.m_max(y)),
        BinOp::Fmin => binary2::<T, T, _>(a, b, o, n, |x, y| x.m_fmin(y)),
        BinOp::Fmax => binary2::<T, T, _>(a, b, o, n, |x, y| x.m_fmax(y)),
        other => run_binary_typed::<T>(a, b, o, n, other),
    }
}

/// The complex `<`/`<=`/`>`/`>=` loops, which report `invalid` for a NaN.
///
/// numpy's `CLT`/`CLE`/`CGT`/`CGE` macros compare the real parts with C's
/// `<`/`>`, the *signalling* predicates, so a NaN real part on either side sets
/// the invalid flag; the imaginary parts are only reached (and can only signal)
/// when the real parts compare equal. `equal`/`not_equal` use `==`/`!=`, which
/// are quiet, and report nothing.
fn run_cplx_cmp<T>(a: &NdArray, b: &NdArray, o: *mut u8, n: usize, op: BinOp)
where
    T: Element + Send + Sync + Cmp + CplxParts,
{
    macro_rules! go {
        ($f:expr) => {
            binary2_flagged::<T, u8, _, _, _>(
                a,
                b,
                o,
                n,
                $f,
                |x: T, y: T, _r: u8| {
                    x.c_re().is_nan()
                        || y.c_re().is_nan()
                        || (x.c_re() == y.c_re() && (x.c_im().is_nan() || y.c_im().is_nan()))
                },
                |_, _, _| fpe::INVALID,
            )
        };
    }
    match op {
        BinOp::Lt => go!(|x: T, y: T| x.c_lt(y) as u8),
        BinOp::Le => go!(|x: T, y: T| x.c_le(y) as u8),
        BinOp::Gt => go!(|x: T, y: T| y.c_lt(x) as u8),
        _ => go!(|x: T, y: T| y.c_le(x) as u8),
    }
}

/// As [`float_flags`], plus the underflow test that only makes sense for a
/// product or quotient: a non-zero finite times a non-zero finite can only
/// reach zero by underflowing, whereas `1.0 - 1.0` legitimately is zero.
#[inline]
fn scale_flags<T: FpClass>(x: T, y: T, r: T, divide_like: bool) -> u8 {
    if r.fp_zero() && !x.fp_zero() && !y.fp_zero() && x.fp_finite() && y.fp_finite() {
        return fpe::UNDER;
    }
    float_flags(x, y, r, divide_like)
}

/// `add`/`subtract`/`multiply`/`divide` over a float or complex type, with the
/// FP error flags numpy's hardware status register would have picked up.
fn run_inexact_arith<T>(a: &NdArray, b: &NdArray, o: *mut T, n: usize, op: BinOp)
where
    T: Element + Send + Sync + Arith,
{
    macro_rules! go {
        ($T:ty, $divflags:expr) => {
            match op {
                BinOp::Add => binary2_flagged::<$T, $T, _, _, _>(
                    a, b, o as *mut $T, n,
                    |x, y| Arith::a_add(x, y),
                    watch_nonfinite,
                    |x, y, r| float_flags(x, y, r, false),
                ),
                BinOp::Sub => binary2_flagged::<$T, $T, _, _, _>(
                    a, b, o as *mut $T, n,
                    |x, y| Arith::a_sub(x, y),
                    watch_nonfinite,
                    |x, y, r| float_flags(x, y, r, false),
                ),
                BinOp::Mul => binary2_flagged::<$T, $T, _, _, _>(
                    a, b, o as *mut $T, n,
                    |x, y| Arith::a_mul(x, y),
                    watch_scale_fn(),
                    |x, y, r| scale_flags(x, y, r, false),
                ),
                _ => binary2_flagged::<$T, $T, _, _, _>(
                    a, b, o as *mut $T, n,
                    |x, y| Arith::a_div(x, y),
                    watch_scale_fn(),
                    $divflags,
                ),
            }
        };
    }
    match T::DTYPE {
        DType::F16 => go!(F16, |x, y, r| scale_flags(x, y, r, true)),
        DType::F32 => go!(f32, |x, y, r| scale_flags(x, y, r, true)),
        DType::F64 => go!(f64, |x, y, r| scale_flags(x, y, r, true)),
        DType::C64 => go!(C32, cplx_div_flags),
        DType::C128 => go!(C64v, cplx_div_flags),
        other => panic!("run_inexact_arith: {other:?}"),
    }
}

/// The remaining ops, which only exist for some categories. Dispatch happens
/// on the runtime dtype so the trait bounds stay off the generic driver.
fn run_binary_typed<T: Element + Send + Sync>(
    a: &NdArray,
    b: &NdArray,
    o: *mut T,
    n: usize,
    op: BinOp,
) {
    // The compute dtype is `T::DTYPE`; re-dispatch on it and transmute the
    // output pointer, which is sound because the two are the same type.
    match T::DTYPE {
        DType::Bool => run_bool_extra(a, b, o as *mut NpBool, n, op),
        DType::I8 => run_int_extra::<i8>(a, b, o as *mut i8, n, op),
        DType::I16 => run_int_extra::<i16>(a, b, o as *mut i16, n, op),
        DType::I32 => run_int_extra::<i32>(a, b, o as *mut i32, n, op),
        DType::I64 => run_int_extra::<i64>(a, b, o as *mut i64, n, op),
        DType::U8 => run_int_extra::<u8>(a, b, o as *mut u8, n, op),
        DType::U16 => run_int_extra::<u16>(a, b, o as *mut u16, n, op),
        DType::U32 => run_int_extra::<u32>(a, b, o as *mut u32, n, op),
        DType::U64 => run_int_extra::<u64>(a, b, o as *mut u64, n, op),
        DType::F16 => run_float_extra::<F16>(a, b, o as *mut F16, n, op),
        DType::F32 => run_float_extra::<f32>(a, b, o as *mut f32, n, op),
        DType::F64 => run_float_extra::<f64>(a, b, o as *mut f64, n, op),
        DType::C64 => run_complex_extra::<C32>(a, b, o as *mut C32, n, op),
        DType::C128 => run_complex_extra::<C64v>(a, b, o as *mut C64v, n, op),
        other => panic!("run_binary_typed: unsupported dtype {other:?}"),
    }
}

fn run_bool_extra(a: &NdArray, b: &NdArray, o: *mut NpBool, n: usize, op: BinOp) {
    match op {
        BinOp::BitAnd => binary2::<NpBool, NpBool, _>(a, b, o, n, |x, y| {
            NpBool::new(x.get() && y.get())
        }),
        BinOp::BitOr => binary2::<NpBool, NpBool, _>(a, b, o, n, |x, y| {
            NpBool::new(x.get() || y.get())
        }),
        BinOp::BitXor => binary2::<NpBool, NpBool, _>(a, b, o, n, |x, y| {
            NpBool::new(x.get() != y.get())
        }),
        other => panic!("bool has no loop for {}", other.name()),
    }
}

fn run_int_extra<T>(a: &NdArray, b: &NdArray, o: *mut T, n: usize, op: BinOp)
where
    T: Element + Send + Sync + IntOps + SignedBits,
{
    match op {
        BinOp::BitAnd => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_and(y)),
        BinOp::BitOr => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_or(y)),
        BinOp::BitXor => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_xor(y)),
        BinOp::LShift => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_shl(y)),
        BinOp::RShift => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_shr(y)),
        BinOp::Gcd => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_gcd(y)),
        BinOp::Lcm => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_lcm(y)),
        BinOp::FloorDiv => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_floordiv(y)),
        BinOp::Mod => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_mod(y)),
        BinOp::Fmod => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_fmod(y)),
        BinOp::Pow => binary2::<T, T, _>(a, b, o, n, |x, y| x.i_pow(y)),
        other => panic!("integers have no loop for {}", other.name()),
    }
}

/// The flag rule for a float binary result: NaN from non-NaN inputs is
/// `invalid`; an infinity from finite inputs is `overflow`, or `divide` when
/// the op is a division and the denominator was zero.
#[inline]
fn float_flags<T: FpClass>(x: T, y: T, r: T, divide_like: bool) -> u8 {
    if r.fp_nan() {
        return if x.fp_nan() || y.fp_nan() {
            0
        } else {
            fpe::INVALID
        };
    }
    if r.fp_inf() && x.fp_finite() && y.fp_finite() {
        return if divide_like && y.fp_zero() {
            fpe::DIVIDE
        } else {
            fpe::OVER
        };
    }
    0
}

fn run_float_extra<T>(a: &NdArray, b: &NdArray, o: *mut T, n: usize, op: BinOp)
where
    T: Element + Send + Sync + RealFloat + Arith,
{
    match op {
        BinOp::Pow | BinOp::FloatPower => binary2_flagged::<T, T, _, _, _>(
            a,
            b,
            o,
            n,
            |x, y| x.r_pow(y),
            watch_nonfinite,
            |x, y, r| {
                // numpy: 0 ** -k reports divide-by-zero, everything else that
                // overflows to infinity reports overflow.
                if r.fp_nan() && !x.fp_nan() && !y.fp_nan() {
                    fpe::INVALID
                } else if r.fp_inf() && x.fp_finite() && y.fp_finite() {
                    if x.fp_zero() {
                        fpe::DIVIDE
                    } else {
                        fpe::OVER
                    }
                } else {
                    0
                }
            },
        ),
        // Every condition `floor_divide` can report leaves the quotient
        // non-finite, so the plain watch still covers it.
        BinOp::FloorDiv => binary2_flagged::<T, T, _, _, _>(
            a,
            b,
            o,
            n,
            |x, y| x.r_divmod(y).0,
            watch_nonfinite,
            |x, y, _r| x.r_divmod_flags(y),
        ),
        // `remainder` can raise while returning an ordinary finite number (the
        // overflow belongs to the quotient it computed on the way), so the
        // watch has to look at the operands: one extra division in a loop that
        // already does an `fmod` and a division.
        BinOp::Mod => binary2_flagged::<T, T, _, _, _>(
            a,
            b,
            o,
            n,
            |x, y| x.r_divmod(y).1,
            |x: T, y: T, r: T| !r.fp_finite() || !Arith::a_div(x, y).fp_finite(),
            |x, y, _r| x.r_remainder_flags(y),
        ),
        // `fmod(inf, 2)` is a NaN that numpy does *not* flag, so the rule
        // here is "both inputs were finite" rather than "neither was a NaN".
        BinOp::Fmod => binary2_flagged::<T, T, _, _, _>(
            a,
            b,
            o,
            n,
            |x, y| x.r_fmod(y),
            watch_nonfinite,
            // numpy's fmod raises nothing at all, even for `fmod(1.0, 0.0)`.
            |_, _, _| 0,
        ),
        BinOp::Arctan2 => binary2::<T, T, _>(a, b, o, n, |x, y| x.r_atan2(y)),
        BinOp::Hypot => binary2_flagged::<T, T, _, _, _>(
            a, b, o, n,
            |x, y| x.r_hypot(y),
            watch_nonfinite,
            |x, y, r| float_flags(x, y, r, false),
        ),
        BinOp::Copysign => binary2::<T, T, _>(a, b, o, n, |x, y| x.r_copysign(y)),
        // `nextafter(MAX, inf)` overflows; the *direction* operand being
        // infinite is normal and must not suppress the flag.
        BinOp::Nextafter => binary2_flagged::<T, T, _, _, _>(
            a, b, o, n,
            |x, y| x.r_nextafter(y),
            watch_nonfinite,
            |x, _, r| {
                if r.fp_inf() && x.fp_finite() {
                    fpe::OVER
                } else {
                    0
                }
            },
        ),
        // numpy's `npy_logaddexp` starts with `x == y`, then `tmp > 0`, then
        // `tmp <= 0` -- and C's `>`/`<=` are the signalling predicates, so a
        // NaN operand (which fails all three) sets the invalid flag on the way
        // past. Probed: `np.logaddexp(np.nan, 1.0)` warns, `np.add` does not.
        BinOp::Logaddexp => binary2_flagged::<T, T, _, _, _>(
            a, b, o, n,
            |x, y| x.r_logaddexp(y),
            watch_nonfinite,
            |x, y, r| {
                if x.fp_nan() || y.fp_nan() {
                    fpe::INVALID
                } else {
                    float_flags(x, y, r, false)
                }
            },
        ),
        BinOp::Logaddexp2 => binary2_flagged::<T, T, _, _, _>(
            a, b, o, n,
            |x, y| x.r_logaddexp2(y),
            watch_nonfinite,
            |x, y, r| {
                if x.fp_nan() || y.fp_nan() {
                    fpe::INVALID
                } else {
                    float_flags(x, y, r, false)
                }
            },
        ),
        BinOp::Heaviside => binary2::<T, T, _>(a, b, o, n, |x, y| x.r_heaviside(y)),
        BinOp::Ldexp => binary2_flagged::<T, T, _, _, _>(
            a, b, o, n,
            |x, y| {
                let e = y.to_d() as i64;
                x.r_ldexp(e)
            },
            watch_nonfinite,
            |x, _, r| {
                if r.fp_inf() && x.fp_finite() {
                    fpe::OVER
                } else {
                    0
                }
            },
        ),
        other => panic!("floats have no loop for {}", other.name()),
    }
}

fn run_complex_extra<T>(a: &NdArray, b: &NdArray, o: *mut T, n: usize, op: BinOp)
where
    T: Element + Send + Sync + Arith + FpClass + CplxPow,
{
    match op {
        // `npy_cpow` reaches the platform's `cpow` for most arguments and does
        // a complex division of its own for a negative integer exponent, so
        // (as with the complex transcendentals) the conditions numpy reports
        // are whatever those actually signalled -- `tiny ** -2` underflows to
        // zero and then divides by it, which no look at the operands would
        // predict. See `fpe::hw_take`.
        BinOp::Pow | BinOp::FloatPower => {
            crate::loops::binary2_hw::<T, _>(a, b, o, n, |x, y| x.c_pow(y))
        }
        other => panic!("complex has no loop for {}", other.name()),
    }
}

/// Complex `power`, transcribed from numpy's `npy_cpow`.
pub trait CplxPow: Copy {
    fn c_pow(self, o: Self) -> Self;
}

impl CplxPow for C64v {
    #[inline]
    fn c_pow(self, o: Self) -> Self {
        cpow_f64(self, o)
    }
}

impl CplxPow for C32 {
    #[inline]
    fn c_pow(self, o: Self) -> Self {
        cpow_f32(self, o)
    }
}

/// numpy's `npy_cpow` (`npymath/npy_math_complex.c.src`): the small integer
/// exponents are unrolled/repeated-squared so that infinities come out right,
/// and everything else falls through to the platform's `cpow` -- which is what
/// numpy does too, since `HAVE_CPOW` holds everywhere it builds.
fn cpow_f64(a: C64v, b: C64v) -> C64v {
    let (ar, ai, br, bi) = (a.re, a.im, b.re, b.im);
    // z**0 is 1 for every z, 0**0 included.
    if br == 0.0 && bi == 0.0 {
        return C64v::new(1.0, 0.0);
    }
    if ar == 0.0 && ai == 0.0 {
        if br > 0.0 {
            return C64v::new(0.0, 0.0);
        }
        // numpy raises `invalid` here by evaluating `inf - inf`.
        fpe::raise(fpe::INVALID);
        return C64v::new(f64::NAN, f64::NAN);
    }
    if bi == 0.0 && br > -100.0 && br < 100.0 && br == (br as i64) as f64 {
        let n = br as i64;
        // Unrolled so that an infinite operand is not multiplied by a zero.
        if n == 1 {
            return C64v::new(ar, ai);
        }
        if n == 2 {
            return mul_c(a, a);
        }
        if n == 3 {
            return mul_c(a, mul_c(a, a));
        }
        let mut acc = C64v::new(1.0, 0.0);
        let mut p = a;
        let mut e = n.unsigned_abs();
        loop {
            if e & 1 == 1 {
                acc = mul_c(acc, p);
            }
            e >>= 1;
            if e == 0 {
                break;
            }
            p = mul_c(p, p);
        }
        return if n < 0 {
            cdiv_c(C64v::new(1.0, 0.0), acc)
        } else {
            acc
        };
    }
    // SAFETY: C99 `cpow` used with its standard signature; see the `extern`
    // block in `ufunc.rs` for the ABI argument.
    unsafe { crate::ufunc::cpow(a, b) }
}

fn cpow_f32(a: C32, b: C32) -> C32 {
    let (ar, ai, br, bi) = (a.re, a.im, b.re, b.im);
    if br == 0.0 && bi == 0.0 {
        return C32::new(1.0, 0.0);
    }
    if ar == 0.0 && ai == 0.0 {
        if br > 0.0 {
            return C32::new(0.0, 0.0);
        }
        fpe::raise(fpe::INVALID);
        return C32::new(f32::NAN, f32::NAN);
    }
    if bi == 0.0 && br > -100.0 && br < 100.0 && br == (br as i64) as f32 {
        let n = br as i64;
        if n == 1 {
            return C32::new(ar, ai);
        }
        if n == 2 {
            return mul_c32(a, a);
        }
        if n == 3 {
            return mul_c32(a, mul_c32(a, a));
        }
        let mut acc = C32::new(1.0, 0.0);
        let mut p = a;
        let mut e = n.unsigned_abs();
        loop {
            if e & 1 == 1 {
                acc = mul_c32(acc, p);
            }
            e >>= 1;
            if e == 0 {
                break;
            }
            p = mul_c32(p, p);
        }
        return if n < 0 {
            cdiv_c32(C32::new(1.0, 0.0), acc)
        } else {
            acc
        };
    }
    // SAFETY: as `cpow_f64`.
    unsafe { crate::ufunc::cpowf(a, b) }
}

#[inline]
fn mul_c(x: C64v, y: C64v) -> C64v {
    C64v::new(
        x.re.mul_add(y.re, -(x.im * y.im)),
        x.re.mul_add(y.im, x.im * y.re),
    )
}

#[inline]
fn mul_c32(x: C32, y: C32) -> C32 {
    C32::new(
        x.re.mul_add(y.re, -(x.im * y.im)),
        x.re.mul_add(y.im, x.im * y.re),
    )
}

#[inline]
fn cdiv_c(x: C64v, y: C64v) -> C64v {
    Arith::a_div(x, y)
}

#[inline]
fn cdiv_c32(x: C32, y: C32) -> C32 {
    Arith::a_div(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Scalar;

    fn arr(v: &[i64], d: DType) -> NdArray {
        let s: Vec<Scalar> = v.iter().map(|&x| Scalar::Int(x)).collect();
        NdArray::from_scalars(&s, DType::I64).unwrap().astype(d)
    }

    #[test]
    fn result_dtype_matches_numpy() {
        // np.add(int32, float32).dtype == float64 (NEP 50)
        assert_eq!(
            result_dtypes(DType::I32, DType::F32, BinOp::Add).unwrap().1,
            DType::F64
        );
        assert_eq!(
            result_dtypes(DType::I16, DType::F32, BinOp::Add).unwrap().1,
            DType::F32
        );
        // Integer true-division always lands on float64.
        assert_eq!(
            result_dtypes(DType::I64, DType::I64, BinOp::Div).unwrap().1,
            DType::F64
        );
        assert_eq!(
            result_dtypes(DType::F32, DType::F32, BinOp::Div).unwrap().1,
            DType::F32
        );
        // Comparisons always produce bool.
        assert_eq!(
            result_dtypes(DType::F64, DType::I8, BinOp::Lt).unwrap().1,
            DType::Bool
        );
        // bool - bool is a TypeError in numpy.
        assert!(result_dtypes(DType::Bool, DType::Bool, BinOp::Sub).is_err());
        assert!(result_dtypes(DType::Bool, DType::Bool, BinOp::Add).is_ok());
        // The ops whose loop table starts at int8 push bool up.
        assert_eq!(
            result_dtypes(DType::Bool, DType::Bool, BinOp::Pow).unwrap().1,
            DType::I8
        );
        assert_eq!(
            result_dtypes(DType::Bool, DType::Bool, BinOp::BitAnd).unwrap().1,
            DType::Bool
        );
        // Float-only ufuncs lift integers to float64.
        assert_eq!(
            result_dtypes(DType::I32, DType::I32, BinOp::Arctan2).unwrap().1,
            DType::F64
        );
        assert_eq!(
            result_dtypes(DType::F32, DType::F32, BinOp::Hypot).unwrap().1,
            DType::F32
        );
        // float_power always lands on double.
        assert_eq!(
            result_dtypes(DType::F32, DType::F32, BinOp::FloatPower).unwrap().1,
            DType::F64
        );
    }

    #[test]
    fn add_promotes_and_broadcasts() {
        let a = arr(&[1, 2, 3], DType::I32).reshape(&[3, 1]).unwrap();
        let b = arr(&[10, 20], DType::I64).reshape(&[1, 2]).unwrap();
        let c = binary(&a, &b, BinOp::Add).unwrap();
        assert_eq!(c.shape, vec![3, 2]);
        assert_eq!(c.dtype(), DType::I64);
        assert_eq!(
            c.to_vec(),
            [11, 21, 12, 22, 13, 23]
                .iter()
                .map(|&v| Scalar::Int(v))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn integer_division_yields_float64() {
        let a = arr(&[1, 0], DType::I64);
        let b = arr(&[0, 0], DType::I64);
        let c = binary(&a, &b, BinOp::Div).unwrap();
        assert_eq!(c.dtype(), DType::F64);
        assert_eq!(c.get_flat(0), Scalar::Float(f64::INFINITY));
        match c.get_flat(1) {
            Scalar::Float(f) => assert!(f.is_nan()),
            other => panic!("expected nan, got {other:?}"),
        }
    }

    #[test]
    fn integer_overflow_wraps_like_numpy() {
        // np.add(np.int8(127), np.int8(1)) == -128
        let a = arr(&[127], DType::I8);
        let c = binary(&a, &arr(&[1], DType::I8), BinOp::Add).unwrap();
        assert_eq!(c.dtype(), DType::I8);
        assert_eq!(c.get_flat(0), Scalar::Int(-128));
    }

    #[test]
    fn comparisons_return_bool_arrays() {
        let a = arr(&[1, 2, 3], DType::I32);
        let b = arr(&[3, 2, 1], DType::I32);
        let c = binary(&a, &b, BinOp::Lt).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(
            c.to_vec(),
            vec![Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(false)]
        );
        let e = binary(&a, &b, BinOp::Ge).unwrap();
        assert_eq!(
            e.to_vec(),
            vec![Scalar::Bool(false), Scalar::Bool(true), Scalar::Bool(true)]
        );
    }

    #[test]
    fn nan_compares_false_except_ne() {
        let a = NdArray::from_scalars(&[Scalar::Float(f64::NAN)], DType::F64).unwrap();
        for (op, want) in [
            (BinOp::Eq, false),
            (BinOp::Lt, false),
            (BinOp::Le, false),
            (BinOp::Gt, false),
            (BinOp::Ge, false),
            (BinOp::Ne, true),
        ] {
            let c = binary(&a, &a, op).unwrap();
            assert_eq!(c.get_flat(0), Scalar::Bool(want), "{:?}", op);
        }
    }

    #[test]
    fn strided_operands_take_the_slow_path_correctly() {
        let a = NdArray::arange(0.0, 10.0, 1.0, DType::I64).unwrap();
        let s = a.slice_axis(0, 0, 5, 2); // [0,2,4,6,8]
        let t = a.slice_axis(0, 1, 5, 2); // [1,3,5,7,9]
        let c = binary(&s, &t, BinOp::Add).unwrap();
        assert_eq!(
            c.to_vec(),
            [1, 5, 9, 13, 17]
                .iter()
                .map(|&v| Scalar::Int(v))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn incompatible_shapes_error() {
        let a = NdArray::zeros(vec![3], DType::F64).unwrap();
        let b = NdArray::zeros(vec![4], DType::F64).unwrap();
        let e = binary(&a, &b, BinOp::Add).unwrap_err();
        assert!(matches!(e, Error::ValueError(_)));
        assert!(e.message().contains("broadcast"));
    }

    #[test]
    fn complex_division_matches_numpy_edge_cases() {
        use num_complex::Complex;
        let mk = |v: Vec<(f64, f64)>| {
            let s: Vec<Scalar> = v
                .into_iter()
                .map(|(re, im)| Scalar::Complex(Complex::new(re, im)))
                .collect();
            NdArray::from_scalars(&s, DType::C128).unwrap()
        };
        // np.divide(np.array([3+0j, -4+0j]), np.array([0j, 1e-200+1e-200j]))
        // -> [inf+nanj, -2e+200+2e+200j]
        let a = mk(vec![(3.0, 0.0), (-4.0, 0.0)]);
        let b = mk(vec![(0.0, 0.0), (1e-200, 1e-200)]);
        let c = binary(&a, &b, BinOp::Div).unwrap();
        match c.get_flat(0) {
            Scalar::Complex(z) => {
                assert!(z.re.is_infinite() && z.re > 0.0, "{z:?}");
                assert!(z.im.is_nan(), "{z:?}");
            }
            other => panic!("{other:?}"),
        }
        match c.get_flat(1) {
            // The naive formula overflows to -inf+infj here.
            Scalar::Complex(z) => assert_eq!((z.re, z.im), (-2e200, 2e200)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn bool_add_is_logical_or() {
        let a = NdArray::from_scalars(&[Scalar::Bool(true), Scalar::Bool(false)], DType::Bool)
            .unwrap();
        let b = NdArray::from_scalars(&[Scalar::Bool(false), Scalar::Bool(false)], DType::Bool)
            .unwrap();
        let c = binary(&a, &b, BinOp::Add).unwrap();
        assert_eq!(c.dtype(), DType::Bool);
        assert_eq!(c.to_vec(), vec![Scalar::Bool(true), Scalar::Bool(false)]);
    }

    #[test]
    fn integer_floor_divide_and_remainder_follow_python() {
        // np.floor_divide([7,-7,7,-7],[3,3,-3,-3]) -> [2,-3,-3,2]
        // np.remainder  ([7,-7,7,-7],[3,3,-3,-3]) -> [1, 2,-2,-1]
        let a = arr(&[7, -7, 7, -7], DType::I64);
        let b = arr(&[3, 3, -3, -3], DType::I64);
        let q = binary(&a, &b, BinOp::FloorDiv).unwrap();
        let r = binary(&a, &b, BinOp::Mod).unwrap();
        let get = |x: &NdArray, i| match x.get_flat(i) {
            Scalar::Int(v) => v,
            o => panic!("{o:?}"),
        };
        assert_eq!(
            (0..4).map(|i| get(&q, i)).collect::<Vec<_>>(),
            vec![2, -3, -3, 2]
        );
        assert_eq!(
            (0..4).map(|i| get(&r, i)).collect::<Vec<_>>(),
            vec![1, 2, -2, -1]
        );
    }

    #[test]
    fn negative_integer_power_is_rejected() {
        let a = arr(&[2], DType::I64);
        let b = arr(&[-1], DType::I64);
        let e = binary(&a, &b, BinOp::Pow).unwrap_err();
        assert!(e.message().contains("negative integer powers"));
    }

    #[test]
    fn lcm_and_gcd_match_numpy_at_the_signed_boundary() {
        // numpy takes both magnitudes in the *unsigned* type, so
        // `lcm(INT_MAX, -1)` is `INT_MAX` (not `-INT_MAX`) and
        // `gcd(INT_MIN, INT_MIN)` stays `INT_MIN`.
        let mk = |v: i64| arr(&[v], DType::I32);
        let get = |a: NdArray| match a.get_flat(0) {
            Scalar::Int(v) => v,
            o => panic!("{o:?}"),
        };
        const MAX: i64 = i32::MAX as i64;
        const MIN: i64 = i32::MIN as i64;
        for (x, y, lcm, gcd) in [
            (MAX, -1, MAX, 1),
            (MIN, -1, MIN, 1),
            (MAX, 2, -2, 1),
            (MIN, 2, MIN, 2),
            (MIN, MIN, MIN, MIN),
            (MAX, MIN, MIN, 1),
            (0, -1, 0, 1),
            (MIN, 0, 0, MIN),
            (-6, 4, 12, 2),
            (6, -4, 12, 2),
        ] {
            assert_eq!(get(binary(&mk(x), &mk(y), BinOp::Lcm).unwrap()), lcm,
                       "lcm({x}, {y})");
            assert_eq!(get(binary(&mk(x), &mk(y), BinOp::Gcd).unwrap()), gcd,
                       "gcd({x}, {y})");
        }
    }

    #[test]
    fn half_min_max_break_signed_zero_ties_on_operand_order() {
        // The half loops have no signed-zero fixup, so unlike float32/float64
        // the *first* operand wins: `maximum(f16(-0), f16(0))` is `-0.`.
        let z = F16(0x0000);
        let nz = F16(0x8000);
        assert_eq!(MinMax::m_max(nz, z).0, 0x8000);
        assert_eq!(MinMax::m_max(z, nz).0, 0x0000);
        assert_eq!(MinMax::m_min(z, nz).0, 0x0000);
        assert_eq!(MinMax::m_min(nz, z).0, 0x8000);
        assert_eq!(MinMax::m_fmax(nz, z).0, 0x8000);
        assert_eq!(MinMax::m_fmin(z, nz).0, 0x0000);
        // The wider types keep numpy's sign-based rule.
        assert!(MinMax::m_max(-0.0f64, 0.0f64).is_sign_positive());
        assert!(MinMax::m_max(0.0f64, -0.0f64).is_sign_positive());
        assert!(MinMax::m_min(-0.0f64, 0.0f64).is_sign_negative());
        assert!(MinMax::m_min(0.0f64, -0.0f64).is_sign_negative());
    }

    #[test]
    fn divmod_and_remainder_flags_match_numpy() {
        let max = f64::MAX;
        let tiny = f64::MIN_POSITIVE;
        // b == 0: `a / b` is what signals, and only for a finite non-zero a.
        assert_eq!(RealFloat::r_divmod_flags(1.0f64, 0.0), fpe::DIVIDE);
        assert_eq!(RealFloat::r_divmod_flags(0.0f64, 0.0), fpe::INVALID);
        assert_eq!(RealFloat::r_divmod_flags(f64::INFINITY, 0.0), 0);
        assert_eq!(RealFloat::r_divmod_flags(f64::NAN, 0.0), 0);
        // The quotient overflowing reports both conditions.
        assert_eq!(
            RealFloat::r_divmod_flags(max, tiny),
            fpe::OVER | fpe::INVALID
        );
        assert_eq!(RealFloat::r_divmod_flags(max, 2.0), 0);
        // `remainder` keeps only the overflow half, and half precision keeps
        // nothing at all.
        assert_eq!(RealFloat::r_remainder_flags(1.0f64, 0.0), 0);
        assert_eq!(
            RealFloat::r_remainder_flags(max, tiny),
            fpe::OVER | fpe::INVALID
        );
        assert_eq!(
            RealFloat::r_remainder_flags(F16::from_f32(65504.0), F16::from_f32(6.103515625e-5)),
            0
        );
        // float32 overflows where the same values in double would not.
        assert_eq!(
            RealFloat::r_divmod_flags(f32::MAX, f32::MIN_POSITIVE),
            fpe::OVER | fpe::INVALID
        );
    }

    #[test]
    fn int_floor_divide_reports_the_one_overflow() {
        fpe::clear();
        assert_eq!(IntOps::i_floordiv(i32::MIN, -1), i32::MIN);
        assert_eq!(fpe::take(), fpe::OVER);
        fpe::clear();
        assert_eq!(IntOps::i_floordiv(i32::MIN, 1), i32::MIN);
        assert_eq!(fpe::take(), 0);
    }

    #[test]
    fn complex_comparisons_follow_numpys_macros() {
        let n = f64::NAN;
        // `CLT` will not answer from the real parts when either imaginary part
        // is a NaN, so this is false where plain lexicographic order says true.
        assert!(!Cmp::c_lt(C64v::new(1.0, n), C64v::new(2.0, 0.0)));
        assert!(Cmp::c_lt(C64v::new(1.0, 0.0), C64v::new(2.0, 0.0)));
        // `maximum` keeps a NaN from either side; `fmax` skips the second's.
        let m = MinMax::m_max(C64v::new(2.0, 0.0), C64v::new(1.0, n));
        assert!(m.re == 1.0 && m.im.is_nan(), "{m:?}");
        assert_eq!(
            MinMax::m_fmin(C64v::new(1.0, n), C64v::new(2.0, 0.0)),
            C64v::new(2.0, 0.0)
        );
    }

    #[test]
    fn complex_divide_by_zero_reports_both_conditions() {
        // numpy's `cdiv` divides each component separately: 1/0 is a
        // divide-by-zero and 0/0 an invalid, both from the one element.
        let one = C64v::new(1.0, 0.0);
        let zero = C64v::new(0.0, 0.0);
        assert_eq!(
            cplx_div_flags(one, zero, Arith::a_div(one, zero)),
            fpe::DIVIDE | fpe::INVALID
        );
        let onei = C64v::new(1.0, 1.0);
        assert_eq!(cplx_div_flags(onei, zero, Arith::a_div(onei, zero)), fpe::DIVIDE);
        assert_eq!(cplx_div_flags(zero, zero, Arith::a_div(zero, zero)), fpe::INVALID);
    }

    #[test]
    fn shifts_clamp_like_numpy() {
        let a = arr(&[1, -1], DType::I32);
        let b = arr(&[40, 40], DType::I32);
        let l = binary(&a, &b, BinOp::LShift).unwrap();
        let r = binary(&a, &b, BinOp::RShift).unwrap();
        assert_eq!(l.get_flat(0), Scalar::Int(0));
        assert_eq!(r.get_flat(1), Scalar::Int(-1));
    }
}

// ---------------------------------------------------------------------------
// Allocation-free scalar path
// ---------------------------------------------------------------------------
//
// `np.float64(1.5) + np.float64(2.5)` used to build two 0-d arrays, run the
// full array driver over them and allocate a third for the result: three heap
// buffers, three `Arc`s and three shape/stride vectors for one add. The
// functions below run the *same* element kernels (the `Arith`/`Cmp`/`MinMax`/
// `IntOps`/`RealFloat` trait methods) and the *same* flag attribution helpers
// over plain `Scalar` values, so nothing about what is computed changes; only
// the plumbing around it goes away. `tests/scalar_path.rs` pins that
// equivalence for every (dtype, dtype, op) triple over a value grid.

/// Scalar counterpart of [`binary`], with no allocation at all.
///
/// `None` means "this pair is not handled here" (flexible / object dtypes);
/// the caller falls back to the array path so the error messages stay numpy's.
#[allow(unused_assignments)] // `out` is always written by `dispatch_dtype!`
pub fn binary_scalar(
    xa: Scalar,
    dta: DType,
    xb: Scalar,
    dtb: DType,
    op: BinOp,
) -> Option<Result<(DType, Scalar)>> {
    // The dedicated `qQ->?` / `Qq->?` comparison loops, so that
    // `uint64(2**64-1) < int64(-1)` is answered in 128 bits rather than
    // through the float64 the two would otherwise promote to.
    if op.is_comparison()
        && ((dta == DType::U64 && dtb.is_signed()) || (dtb == DType::U64 && dta.is_signed()))
    {
        let (x, y) = (as_i128(xa), as_i128(xb));
        let r = match op {
            BinOp::Eq => x == y,
            BinOp::Ne => x != y,
            BinOp::Lt => x < y,
            BinOp::Le => x <= y,
            BinOp::Gt => x > y,
            _ => x >= y,
        };
        return Some(Ok((DType::Bool, Scalar::Bool(r))));
    }
    // Only the dtypes `dispatch_dtype!` has a Rust type for; everything else
    // (flexible, object, datetime, ...) goes back to the array path so the
    // messages it raises stay numpy's.
    let simple = |d: DType| {
        matches!(
            d,
            DType::Bool
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
                | DType::F16
                | DType::F32
                | DType::F64
                | DType::C64
                | DType::C128
        )
    };
    if !simple(dta) || !simple(dtb) {
        return None;
    }
    let (compute, out_dtype) = match result_dtypes(dta, dtb, op) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    if op == BinOp::Pow && compute.is_integer() && compute.is_signed() {
        if let Scalar::Int(i) = xb.cast(compute) {
            if i < 0 {
                return Some(Err(Error::ValueError(
                    "Integers to negative integer powers are not allowed.".into(),
                )));
            }
        }
    }
    let mut out = Scalar::Bool(false);
    crate::dispatch_dtype!(compute, T, {
        out = scalar_binary_typed::<T>(T::from_scalar(xa), T::from_scalar(xb), op);
    });
    Some(Ok((out_dtype, out)))
}

/// One compute dtype's scalar kernel — the mirror image of [`run_binary`].
fn scalar_binary_typed<T: Element>(x: T, y: T, op: BinOp) -> Scalar
where
    T: Arith + Cmp + MinMax,
{
    if op.is_comparison() || op.is_logical() {
        if T::DTYPE.is_complex() && matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) {
            // The complex ordered compares use C's signalling predicates, so a
            // NaN anywhere they look sets `invalid`; see `run_cplx_cmp`.
            let flags = |renan: bool, imnan: bool, eq: bool| {
                if renan || (eq && imnan) {
                    fpe::raise(fpe::INVALID);
                }
            };
            let r = match T::DTYPE {
                DType::C64 => {
                    let (a, b) = (C32::from_scalar(x.to_scalar()), C32::from_scalar(y.to_scalar()));
                    flags(
                        a.c_re().is_nan() || b.c_re().is_nan(),
                        a.c_im().is_nan() || b.c_im().is_nan(),
                        a.c_re() == b.c_re(),
                    );
                    cmp_pick(a, b, op)
                }
                _ => {
                    let (a, b) = (
                        C64v::from_scalar(x.to_scalar()),
                        C64v::from_scalar(y.to_scalar()),
                    );
                    flags(
                        a.c_re().is_nan() || b.c_re().is_nan(),
                        a.c_im().is_nan() || b.c_im().is_nan(),
                        a.c_re() == b.c_re(),
                    );
                    cmp_pick(a, b, op)
                }
            };
            return Scalar::Bool(r);
        }
        let r = match op {
            BinOp::Eq => x.c_eq(y),
            BinOp::Ne => !x.c_eq(y),
            BinOp::Lt => x.c_lt(y),
            BinOp::Le => x.c_le(y),
            BinOp::Gt => y.c_lt(x),
            BinOp::Ge => y.c_le(x),
            BinOp::LogicalAnd => x.c_truthy() && y.c_truthy(),
            BinOp::LogicalOr => x.c_truthy() || y.c_truthy(),
            BinOp::LogicalXor => x.c_truthy() != y.c_truthy(),
            _ => unreachable!(),
        };
        return Scalar::Bool(r);
    }
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
            if T::DTYPE.is_exact() {
                match op {
                    BinOp::Add => x.a_add(y),
                    BinOp::Sub => x.a_sub(y),
                    BinOp::Mul => x.a_mul(y),
                    _ => x.a_div(y),
                }
                .to_scalar()
            } else {
                scalar_inexact_arith::<T>(x, y, op)
            }
        }
        BinOp::Minimum => x.m_min(y).to_scalar(),
        BinOp::Maximum => x.m_max(y).to_scalar(),
        BinOp::Fmin => x.m_fmin(y).to_scalar(),
        BinOp::Fmax => x.m_fmax(y).to_scalar(),
        other => scalar_binary_extra::<T>(x, y, other),
    }
}

#[inline]
fn cmp_pick<T: Cmp>(a: T, b: T, op: BinOp) -> bool {
    match op {
        BinOp::Lt => a.c_lt(b),
        BinOp::Le => a.c_le(b),
        BinOp::Gt => b.c_lt(a),
        _ => b.c_le(a),
    }
}

/// `add`/`subtract`/`multiply`/`divide` on one inexact scalar — the same
/// kernels and the same flag attribution [`run_inexact_arith`] uses.
fn scalar_inexact_arith<T: Element + Arith>(x: T, y: T, op: BinOp) -> Scalar {
    macro_rules! go {
        ($T:ty, $divflags:expr) => {{
            let (a, b) = (
                <$T as Element>::from_scalar(x.to_scalar()),
                <$T as Element>::from_scalar(y.to_scalar()),
            );
            let (r, flags) = match op {
                BinOp::Add => {
                    let r = Arith::a_add(a, b);
                    (r, float_flags(a, b, r, false))
                }
                BinOp::Sub => {
                    let r = Arith::a_sub(a, b);
                    (r, float_flags(a, b, r, false))
                }
                BinOp::Mul => {
                    let r = Arith::a_mul(a, b);
                    (r, scale_flags(a, b, r, false))
                }
                _ => {
                    let r = Arith::a_div(a, b);
                    #[allow(clippy::redundant_closure_call)]
                    (r, ($divflags)(a, b, r))
                }
            };
            fpe::raise(flags);
            Element::to_scalar(r)
        }};
    }
    match T::DTYPE {
        DType::F16 => go!(F16, |x, y, r| scale_flags(x, y, r, true)),
        DType::F32 => go!(f32, |x, y, r| scale_flags(x, y, r, true)),
        DType::F64 => go!(f64, |x, y, r| scale_flags(x, y, r, true)),
        DType::C64 => go!(C32, cplx_div_flags),
        DType::C128 => go!(C64v, cplx_div_flags),
        other => panic!("scalar_inexact_arith: {other:?}"),
    }
}

/// The remaining ops, re-dispatched on the runtime dtype exactly as
/// [`run_binary_typed`] does.
fn scalar_binary_extra<T: Element>(x: T, y: T, op: BinOp) -> Scalar {
    macro_rules! as_t {
        ($T:ty) => {
            (
                <$T as Element>::from_scalar(x.to_scalar()),
                <$T as Element>::from_scalar(y.to_scalar()),
            )
        };
    }
    match T::DTYPE {
        DType::Bool => {
            let (a, b) = as_t!(NpBool);
            let r = match op {
                BinOp::BitAnd => a.get() && b.get(),
                BinOp::BitOr => a.get() || b.get(),
                BinOp::BitXor => a.get() != b.get(),
                other => panic!("bool has no loop for {}", other.name()),
            };
            Element::to_scalar(NpBool::new(r))
        }
        DType::I8 => scalar_int_extra::<i8>(as_t!(i8), op),
        DType::I16 => scalar_int_extra::<i16>(as_t!(i16), op),
        DType::I32 => scalar_int_extra::<i32>(as_t!(i32), op),
        DType::I64 => scalar_int_extra::<i64>(as_t!(i64), op),
        DType::U8 => scalar_int_extra::<u8>(as_t!(u8), op),
        DType::U16 => scalar_int_extra::<u16>(as_t!(u16), op),
        DType::U32 => scalar_int_extra::<u32>(as_t!(u32), op),
        DType::U64 => scalar_int_extra::<u64>(as_t!(u64), op),
        DType::F16 => scalar_float_extra::<F16>(as_t!(F16), op),
        DType::F32 => scalar_float_extra::<f32>(as_t!(f32), op),
        DType::F64 => scalar_float_extra::<f64>(as_t!(f64), op),
        DType::C64 => scalar_complex_extra::<C32>(as_t!(C32), op),
        DType::C128 => scalar_complex_extra::<C64v>(as_t!(C64v), op),
        other => panic!("scalar_binary_extra: unsupported dtype {other:?}"),
    }
}

fn scalar_int_extra<T>((x, y): (T, T), op: BinOp) -> Scalar
where
    T: Element + IntOps + SignedBits,
{
    // The integer kernels raise their own flags (`i_floordiv` reports a zero
    // divisor and `INT_MIN // -1`), exactly as they do inside the array loop.
    let r = match op {
        BinOp::BitAnd => x.i_and(y),
        BinOp::BitOr => x.i_or(y),
        BinOp::BitXor => x.i_xor(y),
        BinOp::LShift => x.i_shl(y),
        BinOp::RShift => x.i_shr(y),
        BinOp::Gcd => x.i_gcd(y),
        BinOp::Lcm => x.i_lcm(y),
        BinOp::FloorDiv => x.i_floordiv(y),
        BinOp::Mod => x.i_mod(y),
        BinOp::Fmod => x.i_fmod(y),
        BinOp::Pow => x.i_pow(y),
        other => panic!("integers have no loop for {}", other.name()),
    };
    r.to_scalar()
}

fn scalar_float_extra<T>((x, y): (T, T), op: BinOp) -> Scalar
where
    T: Element + RealFloat + Arith,
{
    let (r, flags) = match op {
        BinOp::Pow | BinOp::FloatPower => {
            let r = x.r_pow(y);
            let f = if r.fp_nan() && !x.fp_nan() && !y.fp_nan() {
                fpe::INVALID
            } else if r.fp_inf() && x.fp_finite() && y.fp_finite() {
                if x.fp_zero() {
                    fpe::DIVIDE
                } else {
                    fpe::OVER
                }
            } else {
                0
            };
            (r, f)
        }
        BinOp::FloorDiv => (x.r_divmod(y).0, x.r_divmod_flags(y)),
        BinOp::Mod => (x.r_divmod(y).1, x.r_remainder_flags(y)),
        // numpy's fmod raises nothing at all, even for `fmod(1.0, 0.0)`.
        BinOp::Fmod => (x.r_fmod(y), 0),
        BinOp::Arctan2 => (x.r_atan2(y), 0),
        BinOp::Hypot => {
            let r = x.r_hypot(y);
            let f = float_flags(x, y, r, false);
            (r, f)
        }
        BinOp::Copysign => (x.r_copysign(y), 0),
        BinOp::Nextafter => {
            let r = x.r_nextafter(y);
            let f = if r.fp_inf() && x.fp_finite() {
                fpe::OVER
            } else {
                0
            };
            (r, f)
        }
        BinOp::Logaddexp | BinOp::Logaddexp2 => {
            let r = if op == BinOp::Logaddexp {
                x.r_logaddexp(y)
            } else {
                x.r_logaddexp2(y)
            };
            let f = if x.fp_nan() || y.fp_nan() {
                fpe::INVALID
            } else {
                float_flags(x, y, r, false)
            };
            (r, f)
        }
        BinOp::Heaviside => (x.r_heaviside(y), 0),
        BinOp::Ldexp => {
            let r = x.r_ldexp(y.to_d() as i64);
            let f = if r.fp_inf() && x.fp_finite() {
                fpe::OVER
            } else {
                0
            };
            (r, f)
        }
        other => panic!("floats have no loop for {}", other.name()),
    };
    fpe::raise(flags);
    r.to_scalar()
}

fn scalar_complex_extra<T>((x, y): (T, T), op: BinOp) -> Scalar
where
    T: Element + Arith + FpClass + CplxPow,
{
    match op {
        // As the array loop: `npy_cpow` reaches the platform's `cpow`, so the
        // conditions reported are whatever the hardware actually signalled.
        BinOp::Pow | BinOp::FloatPower => {
            fpe::hw_clear();
            let r = x.c_pow(y);
            fpe::raise(fpe::hw_take());
            r.to_scalar()
        }
        other => panic!("complex has no loop for {}", other.name()),
    }
}

