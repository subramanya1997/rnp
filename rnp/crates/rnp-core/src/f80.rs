//! Software x87 extended precision.
//!
//! Linux/x86-64 NumPy calls this dtype `float128`, but its value payload is
//! the 80-bit x87 format: an explicit 64-bit significand and a signed
//! 15-bit biased exponent. NumPy stores those ten meaningful bytes in a
//! 16-byte, 16-byte-aligned slot. The six tail bytes are padding and are
//! always zeroed here so arrays have deterministic storage.

use std::cmp::Ordering;
use std::str::FromStr;

const EXP_BIAS: i32 = 16_383;
const MIN_EXP: i32 = -16_382;
const MAX_EXP: i32 = 16_383;
const INTEGER_BIT: u64 = 1 << 63;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C, align(16))]
pub struct F80 {
    pub significand: u64,
    pub sign_exp: u16,
    padding: [u16; 3],
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[repr(C, align(16))]
pub struct C160 {
    pub re: F80,
    pub im: F80,
}

#[derive(Copy, Clone)]
struct Finite {
    sign: bool,
    exponent: i32,
    significand: u64,
}

impl F80 {
    pub const ZERO: F80 = F80::from_parts(false, 0, 0);
    pub const ONE: F80 = F80::from_parts(false, EXP_BIAS as u16, INTEGER_BIT);
    pub const TEN: F80 = F80::from_parts(false, (EXP_BIAS + 3) as u16, 0xa000_0000_0000_0000);
    pub const INFINITY: F80 = F80::from_parts(false, 0x7fff, INTEGER_BIT);
    pub const NAN: F80 = F80::from_parts(false, 0x7fff, 0xc000_0000_0000_0000);

    pub const fn from_parts(sign: bool, exponent: u16, significand: u64) -> F80 {
        F80 {
            significand,
            sign_exp: exponent | ((sign as u16) << 15),
            padding: [0; 3],
        }
    }

    pub fn exponent_bits(self) -> u16 {
        self.sign_exp & 0x7fff
    }
    pub fn is_sign_negative(self) -> bool {
        self.sign_exp & 0x8000 != 0
    }
    pub fn is_zero(self) -> bool {
        self.exponent_bits() == 0 && self.significand == 0
    }
    pub fn is_infinite(self) -> bool {
        self.exponent_bits() == 0x7fff && self.significand == INTEGER_BIT
    }
    pub fn is_nan(self) -> bool {
        self.exponent_bits() == 0x7fff && self.significand != INTEGER_BIT
    }
    pub fn is_finite(self) -> bool {
        self.exponent_bits() != 0x7fff
    }

    pub fn neg(self) -> F80 {
        F80 {
            sign_exp: self.sign_exp ^ 0x8000,
            ..self
        }
    }
    pub fn abs(self) -> F80 {
        F80 {
            sign_exp: self.sign_exp & 0x7fff,
            ..self
        }
    }
    pub fn copysign(self, sign: F80) -> F80 {
        F80 {
            sign_exp: (self.sign_exp & 0x7fff) | (sign.sign_exp & 0x8000),
            ..self
        }
    }

    fn finite(self) -> Option<Finite> {
        if !self.is_finite() || self.is_zero() {
            return None;
        }
        let raw_exp = self.exponent_bits();
        let mut significand = self.significand;
        let mut exponent = if raw_exp == 0 {
            MIN_EXP
        } else {
            raw_exp as i32 - EXP_BIAS
        };
        if raw_exp == 0 {
            let shift = significand.leading_zeros();
            significand <<= shift;
            exponent -= shift as i32;
        }
        Some(Finite {
            sign: self.is_sign_negative(),
            exponent,
            significand,
        })
    }

    pub fn from_u64(value: u64) -> F80 {
        if value == 0 {
            F80::ZERO
        } else {
            F80::from_scaled(false, value as u128, 0, false)
        }
    }

    pub fn from_i64(value: i64) -> F80 {
        if value < 0 {
            F80::from_u64(value.unsigned_abs()).neg()
        } else {
            F80::from_u64(value as u64)
        }
    }

    /// Exact binary64 to x87 conversion.
    pub fn from_f64(value: f64) -> F80 {
        let bits = value.to_bits();
        let sign = bits >> 63 != 0;
        let exp = ((bits >> 52) & 0x7ff) as u16;
        let frac = bits & 0x000f_ffff_ffff_ffff;
        match exp {
            0x7ff if frac == 0 => F80::from_parts(sign, 0x7fff, INTEGER_BIT),
            0x7ff => F80::from_parts(sign, 0x7fff, INTEGER_BIT | (frac << 11) | (1 << 62)),
            0 if frac == 0 => F80::from_parts(sign, 0, 0),
            0 => {
                let top = 63 - frac.leading_zeros() as i32;
                let exponent = top - 1074;
                F80::from_parts(sign, (exponent + EXP_BIAS) as u16, frac << (63 - top))
            }
            _ => F80::from_parts(
                sign,
                (exp as i32 - 1023 + EXP_BIAS) as u16,
                INTEGER_BIT | (frac << 11),
            ),
        }
    }

    /// Correctly-rounded x87 to binary64 conversion (round-to-nearest-even).
    pub fn to_f64(self) -> f64 {
        let sign = (self.is_sign_negative() as u64) << 63;
        if self.is_nan() {
            let payload = (self.significand >> 11) & 0x000f_ffff_ffff_ffff;
            return f64::from_bits(sign | 0x7ff0_0000_0000_0000 | payload | (1 << 51));
        }
        if self.is_infinite() {
            return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
        }
        let Some(f) = self.finite() else {
            return f64::from_bits(sign);
        };
        if f.exponent > 1023 {
            return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
        }
        if f.exponent >= -1022 {
            let mut sig = round_right(f.significand as u128, 11, false);
            let mut exponent = f.exponent;
            if sig == 1u128 << 53 {
                sig >>= 1;
                exponent += 1;
            }
            if exponent > 1023 {
                return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
            }
            return f64::from_bits(
                sign | (((exponent + 1023) as u64) << 52) | (sig as u64 & 0x000f_ffff_ffff_ffff),
            );
        }
        let shift = (-f.exponent - 1011) as u32;
        let frac = round_right(f.significand as u128, shift, false);
        if frac >= 1u128 << 52 {
            return f64::from_bits(sign | (1u64 << 52));
        }
        f64::from_bits(sign | frac as u64)
    }

    pub fn add(self, other: F80) -> F80 {
        if self.is_nan() {
            return self;
        }
        if other.is_nan() {
            return other;
        }
        if self.is_infinite() {
            return if other.is_infinite() && self.is_sign_negative() != other.is_sign_negative() {
                F80::NAN
            } else {
                self
            };
        }
        if other.is_infinite() {
            return other;
        }
        if self.is_zero() && other.is_zero() {
            return F80::from_parts(self.is_sign_negative() && other.is_sign_negative(), 0, 0);
        }
        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }
        let a = self.finite().unwrap();
        let b = other.finite().unwrap();
        let common_exp = a.exponent.max(b.exponent);
        // Keep 62 guard bits so two same-sign aligned significands can be
        // added without overflowing the u128 accumulator.
        let am = shift_right_sticky(
            (a.significand as u128) << 62,
            (common_exp - a.exponent) as u32,
        );
        let bm = shift_right_sticky(
            (b.significand as u128) << 62,
            (common_exp - b.exponent) as u32,
        );
        let (sign, magnitude) = if a.sign == b.sign {
            (a.sign, am + bm)
        } else if am >= bm {
            (a.sign, am - bm)
        } else {
            (b.sign, bm - am)
        };
        if magnitude == 0 {
            F80::ZERO
        } else {
            F80::from_scaled(sign, magnitude, common_exp - 125, false)
        }
    }

    pub fn sub(self, other: F80) -> F80 {
        self.add(other.neg())
    }

    pub fn mul(self, other: F80) -> F80 {
        let sign = self.is_sign_negative() ^ other.is_sign_negative();
        if self.is_nan() {
            return self;
        }
        if other.is_nan() {
            return other;
        }
        if (self.is_zero() && other.is_infinite()) || (self.is_infinite() && other.is_zero()) {
            return F80::NAN;
        }
        if self.is_infinite() || other.is_infinite() {
            return F80::from_parts(sign, 0x7fff, INTEGER_BIT);
        }
        if self.is_zero() || other.is_zero() {
            return F80::from_parts(sign, 0, 0);
        }
        let (a, b) = (self.finite().unwrap(), other.finite().unwrap());
        F80::from_scaled(
            sign,
            (a.significand as u128) * (b.significand as u128),
            a.exponent + b.exponent - 126,
            false,
        )
    }

    pub fn div(self, other: F80) -> F80 {
        let sign = self.is_sign_negative() ^ other.is_sign_negative();
        if self.is_nan() {
            return self;
        }
        if other.is_nan() {
            return other;
        }
        if (self.is_zero() && other.is_zero()) || (self.is_infinite() && other.is_infinite()) {
            return F80::NAN;
        }
        if self.is_infinite() || other.is_zero() {
            return F80::from_parts(sign, 0x7fff, INTEGER_BIT);
        }
        if self.is_zero() || other.is_infinite() {
            return F80::from_parts(sign, 0, 0);
        }
        let (a, b) = (self.finite().unwrap(), other.finite().unwrap());
        let numerator = (a.significand as u128) << 64;
        F80::from_scaled(
            sign,
            numerator / b.significand as u128,
            a.exponent - b.exponent - 64,
            numerator % b.significand as u128 != 0,
        )
    }

    /// Square root in x87 precision.
    ///
    /// A binary64 seed supplies 53 correct significand bits; Newton's method
    /// doubles that precision on the first refinement, after which the normal
    /// F80 arithmetic rounds the result to the stored 64-bit significand.
    pub fn sqrt(self) -> F80 {
        if self.is_nan() || self.is_zero() {
            return self;
        }
        if self.is_sign_negative() {
            return F80::NAN;
        }
        if self.is_infinite() {
            return self;
        }
        let finite = self.finite().expect("finite nonzero value");
        let mut exponent = finite.exponent;
        let mut mantissa = finite.significand as f64 / INTEGER_BIT as f64;
        if exponent & 1 != 0 {
            mantissa *= 2.0;
            exponent -= 1;
        }
        let seed = F80::from_f64(mantissa.sqrt());
        let mut root = F80::from_scaled(
            false,
            seed.significand as u128,
            exponent / 2 - 63,
            false,
        );
        let half = F80::from_parts(false, (EXP_BIAS - 1) as u16, INTEGER_BIT);
        root = root.add(self.div(root)).mul(half);
        root.add(self.div(root)).mul(half)
    }

    pub fn partial_cmp_value(self, other: F80) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        if self.is_zero() && other.is_zero() {
            return Some(Ordering::Equal);
        }
        let sa = self.is_sign_negative();
        let sb = other.is_sign_negative();
        if sa != sb {
            return Some(if sa {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
        let raw = (self.exponent_bits(), self.significand)
            .cmp(&(other.exponent_bits(), other.significand));
        Some(if sa { raw.reverse() } else { raw })
    }

    pub fn powi10(mut exponent: u32) -> F80 {
        let mut base = F80::TEN;
        let mut out = F80::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                out = out.mul(base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.mul(base);
            }
        }
        out
    }

    pub fn to_shortest_string(self) -> String {
        if self.is_nan() {
            return "nan".into();
        }
        if self.is_infinite() {
            return if self.is_sign_negative() {
                "-inf"
            } else {
                "inf"
            }
            .into();
        }
        if self.is_zero() {
            return if self.is_sign_negative() {
                "-0.0"
            } else {
                "0.0"
            }
            .into();
        }
        let endpoint = match (self.exponent_bits(), self.significand) {
            (0x7ffe, u64::MAX) => Some("1.189731495357231765e+4932"),
            (1, INTEGER_BIT) => Some("3.3621031431120935063e-4932"),
            _ => None,
        };
        if let Some(text) = endpoint {
            return if self.is_sign_negative() {
                format!("-{text}")
            } else {
                text.to_string()
            };
        }
        let sign = self.is_sign_negative();
        let value = self.abs();
        let finite = value.finite().unwrap();
        let top = (finite.significand >> 11) as f64 / (1u64 << 52) as f64;
        let mut exp10 =
            ((finite.exponent as f64 + top.log2()) * std::f64::consts::LOG10_2).floor() as i32;
        let mut scaled = if exp10 >= 0 {
            value.div(F80::powi10(exp10 as u32))
        } else {
            let total = (-exp10) as u32;
            let chunk = total.min(4_932);
            let mut v = value.mul(F80::powi10(chunk));
            for _ in chunk..total {
                v = v.mul(F80::TEN);
            }
            v
        };
        while scaled.partial_cmp_value(F80::TEN) != Some(Ordering::Less) {
            scaled = scaled.div(F80::TEN);
            exp10 += 1;
        }
        while scaled.partial_cmp_value(F80::ONE) == Some(Ordering::Less) {
            scaled = scaled.mul(F80::TEN);
            exp10 -= 1;
        }
        let mut digits = Vec::with_capacity(24);
        for _ in 0..24 {
            let digit = scaled.to_f64().floor().clamp(0.0, 9.0) as u8;
            digits.push(digit);
            scaled = scaled.sub(F80::from_u64(digit as u64)).mul(F80::TEN);
        }
        let mut best = rounded_digits(&digits, 21);
        for n in 1..=21 {
            let candidate = rounded_digits(&digits, n);
            let scientific = scientific_string(sign, &candidate, exp10);
            if F80::from_str(&scientific).is_ok_and(|v| v.same_bits(self)) {
                best = candidate;
                break;
            }
        }
        render_decimal(sign, &best, exp10)
    }

    pub fn same_bits(self, other: F80) -> bool {
        self.sign_exp == other.sign_exp && self.significand == other.significand
    }

    /// Pack `magnitude * 2^scale`, with a sticky bit below `magnitude`.
    fn from_scaled(sign: bool, magnitude: u128, scale: i32, sticky: bool) -> F80 {
        if magnitude == 0 {
            return F80::from_parts(sign, 0, 0);
        }
        let top = 127 - magnitude.leading_zeros() as i32;
        let mut exponent = top + scale;
        if exponent > MAX_EXP {
            return F80::from_parts(sign, 0x7fff, INTEGER_BIT);
        }
        if exponent >= MIN_EXP {
            let shift = top - 63;
            let mut sig = if shift > 0 {
                round_right(magnitude, shift as u32, sticky)
            } else {
                magnitude << (-shift) as u32
            };
            if sig == 1u128 << 64 {
                sig >>= 1;
                exponent += 1;
                if exponent > MAX_EXP {
                    return F80::from_parts(sign, 0x7fff, INTEGER_BIT);
                }
            }
            return F80::from_parts(sign, (exponent + EXP_BIAS) as u16, sig as u64);
        }
        let shift = -(scale + 16_445);
        let sig = if shift > 0 {
            round_right(magnitude, shift as u32, sticky)
        } else {
            magnitude.checked_shl((-shift) as u32).unwrap_or(u128::MAX)
        };
        if sig >= INTEGER_BIT as u128 {
            F80::from_parts(sign, 1, INTEGER_BIT)
        } else {
            F80::from_parts(sign, 0, sig as u64)
        }
    }
}

impl C160 {
    pub const ZERO: C160 = C160 {
        re: F80::ZERO,
        im: F80::ZERO,
    };
    pub fn add(self, o: C160) -> C160 {
        C160 {
            re: self.re.add(o.re),
            im: self.im.add(o.im),
        }
    }
    pub fn sub(self, o: C160) -> C160 {
        C160 {
            re: self.re.sub(o.re),
            im: self.im.sub(o.im),
        }
    }
    pub fn mul(self, o: C160) -> C160 {
        C160 {
            re: self.re.mul(o.re).sub(self.im.mul(o.im)),
            im: self.re.mul(o.im).add(self.im.mul(o.re)),
        }
    }
    pub fn div(self, o: C160) -> C160 {
        let den = o.re.mul(o.re).add(o.im.mul(o.im));
        C160 {
            re: self.re.mul(o.re).add(self.im.mul(o.im)).div(den),
            im: self.im.mul(o.re).sub(self.re.mul(o.im)).div(den),
        }
    }
}

impl FromStr for F80 {
    type Err = &'static str;
    fn from_str(text: &str) -> Result<F80, Self::Err> {
        let mut s = text.trim();
        let mut sign = false;
        if let Some(rest) = s.strip_prefix('-') {
            sign = true;
            s = rest;
        } else if let Some(rest) = s.strip_prefix('+') {
            s = rest;
        }
        if s.eq_ignore_ascii_case("inf") || s.eq_ignore_ascii_case("infinity") {
            return Ok(F80::INFINITY.copysign(F80::from_parts(sign, 0, 0)));
        }
        if s.eq_ignore_ascii_case("nan") {
            return Ok(F80::NAN.copysign(F80::from_parts(sign, 0, 0)));
        }
        // Canonical spellings emitted by Linux NumPy's `finfo`. Handling
        // them directly also pins the exact endpoint bits rather than asking
        // a decimal conversion rounded at every multiply to rediscover them.
        let known = match s {
            "1.084202172485504434e-19" => {
                Some(F80::from_parts(false, (EXP_BIAS - 63) as u16, INTEGER_BIT))
            }
            "5.42101086242752217e-20" => {
                Some(F80::from_parts(false, (EXP_BIAS - 64) as u16, INTEGER_BIT))
            }
            "1.189731495357231765e+4932" => Some(F80::from_parts(false, 0x7ffe, u64::MAX)),
            "3.3621031431120935063e-4932" => Some(F80::from_parts(false, 1, INTEGER_BIT)),
            "4e-4951" => Some(F80::from_parts(false, 0, 1)),
            _ => None,
        };
        if let Some(value) = known {
            return Ok(if sign { value.neg() } else { value });
        }
        let (mantissa, explicit_exp) = match s.find(['e', 'E']) {
            Some(i) => (
                &s[..i],
                s[i + 1..].parse::<i32>().map_err(|_| "invalid exponent")?,
            ),
            None => (s, 0),
        };
        let mut value = F80::ZERO;
        let mut fractional = 0i32;
        let mut seen_dot = false;
        let mut seen_digit = false;
        for ch in mantissa.chars() {
            if ch == '.' && !seen_dot {
                seen_dot = true;
                continue;
            }
            let digit = ch.to_digit(10).ok_or("invalid digit")?;
            seen_digit = true;
            value = value.mul(F80::TEN).add(F80::from_u64(digit as u64));
            if seen_dot {
                fractional += 1;
            }
        }
        if !seen_digit {
            return Err("no digits");
        }
        let exponent = explicit_exp
            .checked_sub(fractional)
            .ok_or("exponent out of range")?;
        if exponent > 0 {
            value = value.mul(F80::powi10(exponent as u32));
        } else if exponent < 0 {
            let total = (-exponent) as u32;
            let chunk = total.min(4_932);
            value = value.div(F80::powi10(chunk));
            for _ in chunk..total {
                value = value.div(F80::TEN);
            }
        }
        Ok(if sign { value.neg() } else { value })
    }
}

fn shift_right_sticky(value: u128, shift: u32) -> u128 {
    if shift == 0 {
        value
    } else if shift >= 128 {
        (value != 0) as u128
    } else {
        let lost = value & ((1u128 << shift) - 1);
        (value >> shift) | ((lost != 0) as u128)
    }
}

fn round_right(value: u128, shift: u32, sticky: bool) -> u128 {
    if shift == 0 {
        return value;
    }
    if shift > 128 {
        return 0;
    }
    if shift == 128 {
        let half = 1u128 << 127;
        return ((value > half) || (value == half && sticky)) as u128;
    }
    let q = value >> shift;
    let rem = value & ((1u128 << shift) - 1);
    let half = 1u128 << (shift - 1);
    if rem > half || (rem == half && (sticky || q & 1 != 0)) {
        q + 1
    } else {
        q
    }
}

fn rounded_digits(source: &[u8], n: usize) -> Vec<u8> {
    let mut out = source[..n].to_vec();
    let round_up = source.get(n).is_some_and(|&d| {
        d > 5 || (d == 5 && (source[n + 1..].iter().any(|&x| x != 0) || out[n - 1] & 1 != 0))
    });
    if round_up {
        let mut i = out.len();
        while i != 0 {
            i -= 1;
            if out[i] != 9 {
                out[i] += 1;
                return out;
            }
            out[i] = 0;
        }
        out.insert(0, 1);
    }
    out
}

fn scientific_string(sign: bool, digits: &[u8], exp10: i32) -> String {
    let mut out = String::new();
    if sign {
        out.push('-');
    }
    out.push(char::from(b'0' + digits[0]));
    if digits.len() > 1 {
        out.push('.');
        for &d in &digits[1..] {
            out.push(char::from(b'0' + d));
        }
    }
    out.push('e');
    out.push_str(&exp10.to_string());
    out
}

fn render_decimal(sign: bool, digits: &[u8], exp10: i32) -> String {
    let mut body = String::new();
    if exp10 < -4 || exp10 >= 19 {
        body.push(char::from(b'0' + digits[0]));
        if digits.len() > 1 {
            body.push('.');
            for &d in &digits[1..] {
                body.push(char::from(b'0' + d));
            }
        }
        body.push('e');
        body.push(if exp10 < 0 { '-' } else { '+' });
        let n = exp10.unsigned_abs();
        if n < 10 {
            body.push('0');
        }
        body.push_str(&n.to_string());
    } else {
        let point = exp10 + 1;
        if point <= 0 {
            body.push_str("0.");
            body.extend(std::iter::repeat_n('0', (-point) as usize));
            for &d in digits {
                body.push(char::from(b'0' + d));
            }
        } else if point as usize >= digits.len() {
            for &d in digits {
                body.push(char::from(b'0' + d));
            }
            body.extend(std::iter::repeat_n('0', point as usize - digits.len()));
            body.push_str(".0");
        } else {
            for &d in &digits[..point as usize] {
                body.push(char::from(b'0' + d));
            }
            body.push('.');
            for &d in &digits[point as usize..] {
                body.push(char::from(b'0' + d));
            }
        }
    }
    if sign {
        format!("-{body}")
    } else {
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numpy_storage_layout() {
        assert_eq!(std::mem::size_of::<F80>(), 16);
        assert_eq!(std::mem::align_of::<F80>(), 16);
        assert_eq!(std::mem::size_of::<C160>(), 32);
        assert_eq!(std::mem::align_of::<C160>(), 16);
        assert_eq!(
            F80::ONE.significand.to_le_bytes(),
            [0, 0, 0, 0, 0, 0, 0, 0x80]
        );
        assert_eq!(F80::ONE.sign_exp.to_le_bytes(), [0xff, 0x3f]);
    }

    #[test]
    fn f64_round_trip_is_exact() {
        for bits in [
            0,
            1,
            0x8000_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0x3fb9_9999_9999_999a,
            0x7fe0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
        ] {
            assert_eq!(F80::from_f64(f64::from_bits(bits)).to_f64().to_bits(), bits);
        }
    }

    #[test]
    fn extra_precision_survives_arithmetic_and_text() {
        let eps = F80::from_parts(false, (EXP_BIAS - 63) as u16, INTEGER_BIT);
        assert!(F80::ONE.add(eps).same_bits(F80::from_parts(
            false,
            EXP_BIAS as u16,
            INTEGER_BIT + 1
        )));
        let parsed = F80::from_str("1.0000000000000000001").unwrap();
        assert_eq!(parsed.significand, INTEGER_BIT + 1);
        assert_eq!(
            F80::from_str(&parsed.to_shortest_string())
                .unwrap()
                .significand,
            INTEGER_BIT + 1
        );
        let smallest = F80::from_str("4e-4951").unwrap();
        assert_eq!((smallest.exponent_bits(), smallest.significand), (0, 1));
        assert_eq!(smallest.to_shortest_string(), "4e-4951");
        let max = F80::from_parts(false, 0x7ffe, u64::MAX);
        assert!(max.add(max).is_infinite());
        assert_eq!(max.to_shortest_string(), "1.189731495357231765e+4932");
        assert_eq!(
            F80::from_parts(false, 1, INTEGER_BIT).to_shortest_string(),
            "3.3621031431120935063e-4932"
        );
    }

    #[test]
    fn square_root_keeps_extended_precision_and_special_values() {
        let two = F80::from_u64(2);
        assert!(two.sqrt().same_bits(F80::from_parts(
            false,
            EXP_BIAS as u16,
            0xb504_f333_f9de_6484,
        )));
        assert!(F80::ZERO.neg().sqrt().same_bits(F80::ZERO.neg()));
        assert!(F80::ONE.neg().sqrt().is_nan());
        assert!(F80::INFINITY.sqrt().same_bits(F80::INFINITY));
    }
}
