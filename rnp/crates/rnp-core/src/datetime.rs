//! datetime64 / timedelta64: unit metadata, calendar conversion, ISO 8601
//! parsing and formatting, unit casting and the arithmetic kernels.
//!
//! Every rule here is a direct port of numpy 2.5.2's
//! `_core/src/multiarray/datetime.c`, `datetime_strings.c` and the DATETIME /
//! TIMEDELTA loops in `_core/src/umath/loops.c.src`. The unit enum keeps
//! numpy's own numbering, gap included, because the casting and promotion
//! rules are written as `<=` comparisons on it.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// numpy's `NPY_DATETIMEUNIT`. Note the gap at 3: business days were removed
/// but the enum value was kept so that the ordering never changed.
pub const UNIT_Y: u8 = 0;
pub const UNIT_M: u8 = 1;
pub const UNIT_W: u8 = 2;
pub const UNIT_B: u8 = 3;
pub const UNIT_D: u8 = 4;
pub const UNIT_H: u8 = 5;
pub const UNIT_MIN: u8 = 6;
pub const UNIT_S: u8 = 7;
pub const UNIT_MS: u8 = 8;
pub const UNIT_US: u8 = 9;
pub const UNIT_NS: u8 = 10;
pub const UNIT_PS: u8 = 11;
pub const UNIT_FS: u8 = 12;
pub const UNIT_AS: u8 = 13;
pub const UNIT_GENERIC: u8 = 14;

/// The not-a-time sentinel, shared by datetime64 and timedelta64.
pub const NAT: i64 = i64::MIN;

/// Unit spellings indexed by the enum value above. Index 3 is the removed
/// business-day slot and is never produced.
pub const UNIT_NAMES: [&str; 15] = [
    "Y", "M", "W", "B", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as", "generic",
];

/// The 13 real units, in coarse-to-fine order — what `gen_tables.py` walks.
pub const REAL_UNITS: [u8; 13] = [
    UNIT_Y, UNIT_M, UNIT_W, UNIT_D, UNIT_H, UNIT_MIN, UNIT_S, UNIT_MS, UNIT_US, UNIT_NS, UNIT_PS,
    UNIT_FS, UNIT_AS,
];

/// numpy's `_datetime_factors`: the multiplier from unit `i` to unit `i+1`.
const FACTORS: [u64; 15] = [
    1, // Y (not used)
    1, // M (not used)
    7, // W -> D (through the B gap)
    1, // B gap
    24, 60, 60, 1000, 1000, 1000, 1000, 1000, 1000, 1, 0,
];

/// The unit name numpy accepts in a dtype string, or `None`.
pub fn parse_unit(s: &str) -> Option<u8> {
    match s {
        "Y" => Some(UNIT_Y),
        "M" => Some(UNIT_M),
        "W" => Some(UNIT_W),
        "D" => Some(UNIT_D),
        "h" => Some(UNIT_H),
        "m" => Some(UNIT_MIN),
        "s" => Some(UNIT_S),
        "ms" => Some(UNIT_MS),
        "us" | "\u{b5}s" | "\u{3bc}s" => Some(UNIT_US),
        "ns" => Some(UNIT_NS),
        "ps" => Some(UNIT_PS),
        "fs" => Some(UNIT_FS),
        "as" => Some(UNIT_AS),
        "generic" => Some(UNIT_GENERIC),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// numpy's `PyArray_DatetimeMetaData`: a base unit plus a multiplier.
///
/// Packed into the `u64` payload of [`DType::DateTime`] / [`DType::TimeDelta`]
/// as `base | (num << 8)`; `num` is a C `int` in numpy and is validated to
/// `1..=i32::MAX` (plus the `0` numpy also accepts) at parse time.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct DtMeta {
    pub base: u8,
    pub num: u32,
}

impl DtMeta {
    pub const GENERIC: DtMeta = DtMeta {
        base: UNIT_GENERIC,
        num: 1,
    };

    pub fn new(base: u8, num: u32) -> DtMeta {
        DtMeta { base, num }
    }

    pub fn unit(base: u8) -> DtMeta {
        DtMeta { base, num: 1 }
    }

    pub fn pack(self) -> u64 {
        (self.base as u64) | ((self.num as u64) << 8)
    }

    pub fn unpack(v: u64) -> DtMeta {
        DtMeta {
            base: (v & 0xFF) as u8,
            num: (v >> 8) as u32,
        }
    }

    pub fn is_generic(self) -> bool {
        self.base == UNIT_GENERIC
    }

    /// The `[unit]` suffix numpy prints, empty for generic units.
    pub fn suffix(self) -> String {
        if self.is_generic() {
            String::new()
        } else if self.num == 1 {
            format!("[{}]", UNIT_NAMES[self.base as usize])
        } else {
            format!("[{}{}]", self.num, UNIT_NAMES[self.base as usize])
        }
    }

    /// numpy's `metastr_to_unicode(meta, 0)`, used in error messages.
    pub fn metastr(self) -> String {
        if self.is_generic() {
            "[generic]".into()
        } else {
            self.suffix()
        }
    }
}

/// The metadata of a datetime-like dtype, or `None` for anything else.
pub fn meta_of(dt: DType) -> Option<DtMeta> {
    match dt {
        DType::DateTime(v) | DType::TimeDelta(v) => Some(DtMeta::unpack(v)),
        _ => None,
    }
}

/// `dt` with its metadata replaced (keeping datetime vs timedelta).
pub fn with_meta(dt: DType, m: DtMeta) -> DType {
    match dt {
        DType::DateTime(_) => DType::DateTime(m.pack()),
        DType::TimeDelta(_) => DType::TimeDelta(m.pack()),
        other => other,
    }
}

pub fn datetime(m: DtMeta) -> DType {
    DType::DateTime(m.pack())
}

pub fn timedelta(m: DtMeta) -> DType {
    DType::TimeDelta(m.pack())
}

// ---------------------------------------------------------------------------
// Unit conversion factors
// ---------------------------------------------------------------------------

/// numpy's `get_datetime_units_factor`; `0` signals overflow.
pub fn units_factor(bigbase: u8, littlebase: u8) -> u64 {
    let mut factor: u64 = 1;
    let mut unit = bigbase;
    while unit < littlebase {
        // numpy's `npy_uint64` multiply wraps, and the top-byte test below is
        // what catches it (`m` -> `as` really does wrap before it trips).
        factor = factor.wrapping_mul(FACTORS[unit as usize]);
        // numpy detects overflow by disallowing the top 8 bits.
        if factor & 0xff00_0000_0000_0000 != 0 {
            return 0;
        }
        unit += 1;
    }
    factor
}

fn gcd_u64(mut x: u64, mut y: u64) -> u64 {
    if x > y {
        std::mem::swap(&mut x, &mut y);
    }
    while x != y && y != 0 {
        let tmp = x % y;
        x = y;
        y = tmp;
    }
    x
}

/// numpy's `get_datetime_conversion_factor`, as an exact `(num, denom)`.
pub fn conversion_factor(src: DtMeta, dst: DtMeta) -> Result<(i64, i64)> {
    if src.is_generic() {
        return Ok((1, 1));
    }
    if dst.is_generic() {
        return Err(Error::ValueError(
            "Cannot convert from specific units to generic units in NumPy \
             datetimes or timedeltas"
                .into(),
        ));
    }
    let (src_base, dst_base, swapped) = if src.base <= dst.base {
        (src.base, dst.base, false)
    } else {
        (dst.base, src.base, true)
    };
    let mut num: u64 = 1;
    let mut denom: u64 = 1;
    if src_base != dst_base {
        // Year/month conversions use the 400-year average.
        if src_base == UNIT_Y {
            if dst_base == UNIT_M {
                num *= 12;
            } else if dst_base == UNIT_W {
                num *= 97 + 400 * 365;
                denom *= 400 * 7;
            } else {
                num *= 97 + 400 * 365;
                denom *= 400;
                num *= units_factor(UNIT_D, dst_base);
            }
        } else if src_base == UNIT_M {
            if dst_base == UNIT_W {
                num *= 97 + 400 * 365;
                denom *= 400 * 12 * 7;
            } else {
                num *= 97 + 400 * 365;
                denom *= 400 * 12;
                num *= units_factor(UNIT_D, dst_base);
            }
        } else {
            num *= units_factor(src_base, dst_base);
        }
    }
    if num == 0 {
        return Err(Error::OverflowError(format!(
            "Integer overflow while computing the conversion factor between \
             NumPy datetime units {} and {}",
            UNIT_NAMES[src_base as usize], UNIT_NAMES[dst_base as usize]
        )));
    }
    if swapped {
        std::mem::swap(&mut num, &mut denom);
    }
    num *= src.num as u64;
    denom *= dst.num as u64;
    let g = gcd_u64(num, denom);
    Ok(((num / g) as i64, (denom / g) as i64))
}

/// numpy's `_datetime_scale_with_overflow_check`.
pub fn scale_with_overflow_check(dt: i64, num: i64, denom: i64, type_name: &str) -> Result<i64> {
    if dt == NAT {
        return Ok(NAT);
    }
    let pos_limit = i64::MAX / num;
    let neg_limit = (i64::MAX - denom + 1) / num;
    if dt > pos_limit || dt < -neg_limit {
        return Err(Error::OverflowError(format!(
            "Overflow when converting between {type_name} units"
        )));
    }
    Ok(if dt < 0 {
        (dt * num - (denom - 1)) / denom
    } else {
        dt * num / denom
    })
}

/// numpy's `datetime_metadata_divides`.
pub fn metadata_divides(dividend: DtMeta, divisor: DtMeta, strict_nonlinear: bool) -> bool {
    if dividend.is_generic() {
        return true;
    }
    if divisor.is_generic() {
        return false;
    }
    let mut num1 = dividend.num as u64;
    let mut num2 = divisor.num as u64;
    if dividend.base != divisor.base {
        if dividend.base == UNIT_Y {
            if divisor.base == UNIT_M {
                num1 *= 12;
            } else if strict_nonlinear {
                return false;
            } else {
                return true;
            }
        } else if divisor.base == UNIT_Y {
            if dividend.base == UNIT_M {
                num2 *= 12;
            } else if strict_nonlinear {
                return false;
            } else {
                return true;
            }
        } else if dividend.base == UNIT_M || divisor.base == UNIT_M {
            if strict_nonlinear {
                return false;
            } else {
                return true;
            }
        }
        if dividend.base > divisor.base {
            num2 *= units_factor(divisor.base, dividend.base);
            if num2 == 0 {
                return false;
            }
        } else {
            num1 *= units_factor(dividend.base, divisor.base);
            if num1 == 0 {
                return false;
            }
        }
    }
    if num1 & 0xff00_0000_0000_0000 != 0 || num2 & 0xff00_0000_0000_0000 != 0 {
        return false;
    }
    num2 != 0 && num1 % num2 == 0
}

/// numpy's `compute_datetime_metadata_greatest_common_divisor`.
pub fn gcd_meta(m1: DtMeta, m2: DtMeta, strict1: bool, strict2: bool) -> Result<DtMeta> {
    if m1.is_generic() {
        return Ok(m2);
    }
    if m2.is_generic() {
        return Ok(m1);
    }
    let mut num1 = m1.num as u64;
    let mut num2 = m2.num as u64;
    let base;
    if m1.base == m2.base {
        base = m1.base;
    } else {
        // Years and months are incompatible with everything but each other.
        if m1.base == UNIT_Y {
            if m2.base == UNIT_M {
                num1 *= 12;
            } else if strict1 {
                return Err(incompatible(m1, m2));
            }
        } else if m2.base == UNIT_Y {
            if m1.base == UNIT_M {
                num2 *= 12;
            } else if strict2 {
                return Err(incompatible(m1, m2));
            }
        } else if m1.base == UNIT_M {
            if strict1 {
                return Err(incompatible(m1, m2));
            }
        } else if m2.base == UNIT_M && strict2 {
            return Err(incompatible(m1, m2));
        }
        // Take the finer base (numpy's enum grows as units shrink).
        if m1.base > m2.base {
            base = m1.base;
            num2 *= units_factor(m2.base, m1.base);
            if num2 == 0 {
                return Err(units_overflow(m1, m2));
            }
        } else {
            base = m2.base;
            num1 *= units_factor(m1.base, m2.base);
            if num1 == 0 {
                return Err(units_overflow(m1, m2));
            }
        }
    }
    let num = gcd_u64(num1, num2);
    if num == 0 || num > i32::MAX as u64 {
        return Err(units_overflow(m1, m2));
    }
    Ok(DtMeta {
        base,
        num: num as u32,
    })
}

fn incompatible(m1: DtMeta, m2: DtMeta) -> Error {
    Error::TypeError(format!(
        "Cannot get a common metadata divisor for Numpy datetime metadata {} \
         and {} because they have incompatible nonlinear base time units.",
        m1.metastr(),
        m2.metastr()
    ))
}

fn units_overflow(m1: DtMeta, m2: DtMeta) -> Error {
    Error::OverflowError(format!(
        "Integer overflow getting a common metadata divisor for NumPy \
         datetime metadata {} and {}.",
        m1.metastr(),
        m2.metastr()
    ))
}

/// numpy's `datetime_type_promotion`: `M8` wins over `m8`, and the strictness
/// about nonlinear units follows the *timedelta*-ness of each operand.
pub fn promote_meta(a: DType, b: DType) -> Result<DType> {
    let ma = meta_of(a).expect("datetime dtype");
    let mb = meta_of(b).expect("datetime dtype");
    let a_is_td = matches!(a, DType::TimeDelta(_));
    let b_is_td = matches!(b, DType::TimeDelta(_));
    let is_dt = !a_is_td || !b_is_td;
    // numpy 2.x promotes through the new DType machinery: `PyArray_PromoteTypes`
    // first asks `datetime_common_dtype` for the common *DType class* (M8 wins
    // over m8), then casts each descr into that class before handing the pair
    // to `common_instance` (= `datetime_type_promotion`). So a mixed M8/m8 pair
    // reaches the GCD as two *datetime* descrs, and the nonlinear-unit
    // strictness that `datetime_type_promotion` derives from `type_num ==
    // NPY_TIMEDELTA` is off for *both* operands. Only an m8/m8 pair is strict.
    //   np.promote_types('M8[D]', 'm8[Y]') -> dtype('<M8[D]')
    //   np.promote_types('m8[D]', 'm8[Y]') -> TypeError
    let strict = a_is_td && b_is_td;
    let m = gcd_meta(ma, mb, strict, strict)?;
    Ok(if is_dt { datetime(m) } else { timedelta(m) })
}

// ---------------------------------------------------------------------------
// Casting rules
// ---------------------------------------------------------------------------

/// numpy's `can_cast_datetime64_units`. `casting` is the crate's enum.
pub fn can_cast_datetime_units(src: u8, dst: u8, casting: crate::casting::Casting) -> bool {
    use crate::casting::Casting::*;
    match casting {
        Unsafe => true,
        SameKind => {
            if src == UNIT_GENERIC || dst == UNIT_GENERIC {
                src == UNIT_GENERIC
            } else {
                true
            }
        }
        Safe => {
            if src == UNIT_GENERIC || dst == UNIT_GENERIC {
                src == UNIT_GENERIC
            } else {
                src <= dst
            }
        }
        No | Equiv => src == dst,
    }
}

/// numpy's `can_cast_timedelta64_units`: there is a hard barrier between the
/// nonlinear date units (`Y`, `M`) and everything finer.
pub fn can_cast_timedelta_units(src: u8, dst: u8, casting: crate::casting::Casting) -> bool {
    use crate::casting::Casting::*;
    match casting {
        Unsafe => true,
        SameKind => {
            if src == UNIT_GENERIC || dst == UNIT_GENERIC {
                src == UNIT_GENERIC
            } else {
                (src <= UNIT_M && dst <= UNIT_M) || (src > UNIT_M && dst > UNIT_M)
            }
        }
        Safe => {
            if src == UNIT_GENERIC || dst == UNIT_GENERIC {
                src == UNIT_GENERIC
            } else {
                src <= dst && ((src <= UNIT_M && dst <= UNIT_M) || (src > UNIT_M && dst > UNIT_M))
            }
        }
        No | Equiv => src == dst,
    }
}

/// numpy's `can_cast_datetime64_metadata` / `can_cast_timedelta64_metadata`.
pub fn can_cast_meta(
    src: DtMeta,
    dst: DtMeta,
    is_timedelta: bool,
    casting: crate::casting::Casting,
) -> bool {
    use crate::casting::Casting::*;
    let units = if is_timedelta {
        can_cast_timedelta_units
    } else {
        can_cast_datetime_units
    };
    match casting {
        Unsafe => true,
        SameKind => units(src.base, dst.base, SameKind),
        Safe => units(src.base, dst.base, Safe) && metadata_divides(src, dst, is_timedelta),
        No | Equiv => src.base == dst.base && src.num == dst.num,
    }
}

// ---------------------------------------------------------------------------
// Calendar
// ---------------------------------------------------------------------------

/// The broken-down datetime numpy calls `npy_datetimestruct`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct Dts {
    pub year: i64,
    pub month: i32,
    pub day: i32,
    pub hour: i32,
    pub min: i32,
    pub sec: i32,
    pub us: i32,
    pub ps: i32,
    pub as_: i32,
}

impl Dts {
    pub fn epoch() -> Dts {
        Dts {
            year: 1970,
            month: 1,
            day: 1,
            ..Default::default()
        }
    }
    pub fn nat() -> Dts {
        Dts {
            year: NAT,
            month: 1,
            day: 1,
            ..Default::default()
        }
    }
    pub fn is_nat(&self) -> bool {
        self.year == NAT
    }
}

pub const DAYS_PER_MONTH: [[i32; 12]; 2] = [
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
];

pub fn is_leapyear(year: i64) -> bool {
    (year & 3) == 0 && ((year % 100) != 0 || (year % 400) == 0)
}

/// Hinnant's `days_from_civil`, exactly as numpy's `get_datetimestruct_days`.
pub fn days_from_civil(dts: &Dts) -> i64 {
    let y = dts.year - if dts.month <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let mp = if dts.month > 2 {
        dts.month - 3
    } else {
        dts.month + 9
    } as u32;
    let doy = (153 * mp + 2) / 5 + dts.day as u32 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe as i64 - 719_468
}

fn extract_unit(d: &mut i64, unit: i64) -> i64 {
    let mut div = *d / unit;
    let mut m = *d % unit;
    if m < 0 {
        m += unit;
        div -= 1;
    }
    *d = m;
    div
}

/// numpy's `days_to_yearsdays`: turns days-since-epoch into (year, day-of-year).
fn days_to_yearsdays(days_: &mut i64) -> i64 {
    const DAYS_PER_400: i64 = 400 * 365 + 100 - 4 + 1;
    const DAYS_OFFSET: i64 = 365 * 30 + 7;
    let mut days;
    let mut year;
    if *days_ < i64::MIN + DAYS_OFFSET {
        days = *days_ + (DAYS_PER_400 - DAYS_OFFSET);
        year = 400 * extract_unit(&mut days, DAYS_PER_400);
        year -= 400;
    } else {
        days = *days_ - DAYS_OFFSET;
        year = 400 * extract_unit(&mut days, DAYS_PER_400);
    }
    if days >= 366 {
        year += 100 * ((days - 1) / (100 * 365 + 25 - 1));
        days = (days - 1) % (100 * 365 + 25 - 1);
        if days >= 365 {
            year += 4 * ((days + 1) / (4 * 365 + 1));
            days = (days + 1) % (4 * 365 + 1);
            if days >= 366 {
                year += (days - 1) / 365;
                days = (days - 1) % 365;
            }
        }
    }
    *days_ = days;
    year + 2000
}

/// numpy's `set_datetimestruct_days` (Neri-Schneider with a classical fallback).
pub fn set_days(mut days: i64, dts: &mut Dts) {
    if (-12_699_422..=1_061_042_401).contains(&days) {
        const K: u32 = 12_699_422;
        const L: u32 = 32_800;
        let n = (days as i32 as u32).wrapping_add(K);
        let n_1 = 4 * n + 3;
        let c = n_1 / 146_097;
        let n_c = n_1 % 146_097 / 4;
        let n_2 = 4 * n_c + 3;
        let p_2 = 2_939_745u64 * n_2 as u64;
        let z = (p_2 / 4_294_967_296) as u32;
        let n_y = ((p_2 % 4_294_967_296) as u32) / 2_939_745 / 4;
        let n_3 = 2141 * n_y + 197_913;
        let m = n_3 / 65_536;
        let d = n_3 % 65_536 / 2141;
        let j = u32::from(n_y >= 306);
        let y = 100 * c + z;
        dts.year = (y.wrapping_sub(L) as i32 as i64) + j as i64;
        dts.month = if j == 1 { m as i32 - 12 } else { m as i32 };
        dts.day = d as i32 + 1;
        return;
    }
    dts.year = days_to_yearsdays(&mut days);
    let lengths = &DAYS_PER_MONTH[is_leapyear(dts.year) as usize];
    for (i, &len) in lengths.iter().enumerate() {
        if days < len as i64 {
            dts.month = i as i32 + 1;
            dts.day = days as i32 + 1;
            return;
        }
        days -= len as i64;
    }
}

/// numpy's `NpyDatetime_ConvertDatetimeStructToDatetime64`.
pub fn dts_to_dt64(meta: DtMeta, dts: &Dts) -> Result<i64> {
    if dts.is_nat() {
        return Ok(NAT);
    }
    if meta.is_generic() {
        return Err(Error::ValueError(
            "Cannot create a NumPy datetime other than NaT with generic units".into(),
        ));
    }
    let mut ret: i64 = if meta.base == UNIT_Y {
        dts.year - 1970
    } else if meta.base == UNIT_M {
        12 * (dts.year - 1970) + (dts.month as i64 - 1)
    } else {
        let days = days_from_civil(dts);
        match meta.base {
            UNIT_W => {
                if days >= 0 {
                    days / 7
                } else {
                    (days - 6) / 7
                }
            }
            UNIT_D => days,
            UNIT_H => days * 24 + dts.hour as i64,
            UNIT_MIN => (days * 24 + dts.hour as i64) * 60 + dts.min as i64,
            UNIT_S => ((days * 24 + dts.hour as i64) * 60 + dts.min as i64) * 60 + dts.sec as i64,
            UNIT_MS => {
                (((days * 24 + dts.hour as i64) * 60 + dts.min as i64) * 60 + dts.sec as i64) * 1000
                    + (dts.us / 1000) as i64
            }
            UNIT_US => {
                (((days * 24 + dts.hour as i64) * 60 + dts.min as i64) * 60 + dts.sec as i64)
                    * 1_000_000
                    + dts.us as i64
            }
            UNIT_NS => {
                ((((days * 24 + dts.hour as i64) * 60 + dts.min as i64) * 60 + dts.sec as i64)
                    * 1_000_000
                    + dts.us as i64)
                    * 1000
                    + (dts.ps / 1000) as i64
            }
            UNIT_PS => {
                ((((days * 24 + dts.hour as i64) * 60 + dts.min as i64) * 60 + dts.sec as i64)
                    * 1_000_000
                    + dts.us as i64)
                    * 1_000_000
                    + dts.ps as i64
            }
            UNIT_FS => {
                (((((days * 24 + dts.hour as i64) * 60 + dts.min as i64) * 60 + dts.sec as i64)
                    * 1_000_000
                    + dts.us as i64)
                    * 1_000_000
                    + dts.ps as i64)
                    * 1000
                    + (dts.as_ / 1000) as i64
            }
            UNIT_AS => {
                (((((days * 24 + dts.hour as i64) * 60 + dts.min as i64) * 60 + dts.sec as i64)
                    * 1_000_000
                    + dts.us as i64)
                    * 1_000_000
                    + dts.ps as i64)
                    * 1_000_000
                    + dts.as_ as i64
            }
            _ => {
                return Err(Error::ValueError(
                    "NumPy datetime metadata with corrupt unit value".into(),
                ))
            }
        }
    };
    if meta.num > 1 {
        let n = meta.num as i64;
        ret = if ret >= 0 {
            ret / n
        } else {
            (ret - n + 1) / n
        };
    }
    Ok(ret)
}

/// numpy's `NpyDatetime_ConvertDatetime64ToDatetimeStruct`.
pub fn dt64_to_dts(meta: DtMeta, dt: i64) -> Result<Dts> {
    let mut out = Dts::epoch();
    if dt == NAT {
        out.year = NAT;
        return Ok(out);
    }
    if meta.is_generic() {
        return Err(Error::ValueError(
            "Cannot convert a NumPy datetime value other than NaT with generic units".into(),
        ));
    }
    let mut dt = if meta.num > 1 {
        scale_with_overflow_check(dt, meta.num as i64, 1, "datetime64")?
    } else {
        dt.wrapping_mul(meta.num as i64)
    };
    match meta.base {
        UNIT_Y => out.year = 1970 + dt,
        UNIT_M => {
            out.year = 1970 + extract_unit(&mut dt, 12);
            out.month = dt as i32 + 1;
        }
        UNIT_W => set_days(dt.wrapping_mul(7), &mut out),
        UNIT_D => set_days(dt, &mut out),
        UNIT_H => {
            let days = extract_unit(&mut dt, 24);
            set_days(days, &mut out);
            out.hour = dt as i32;
        }
        UNIT_MIN => {
            let days = extract_unit(&mut dt, 60 * 24);
            set_days(days, &mut out);
            out.hour = extract_unit(&mut dt, 60) as i32;
            out.min = dt as i32;
        }
        UNIT_S => {
            let days = extract_unit(&mut dt, 60 * 60 * 24);
            set_days(days, &mut out);
            out.hour = extract_unit(&mut dt, 60 * 60) as i32;
            out.min = extract_unit(&mut dt, 60) as i32;
            out.sec = dt as i32;
        }
        UNIT_MS => {
            let days = extract_unit(&mut dt, 1000 * 60 * 60 * 24);
            set_days(days, &mut out);
            out.hour = extract_unit(&mut dt, 1000 * 60 * 60) as i32;
            out.min = extract_unit(&mut dt, 1000 * 60) as i32;
            out.sec = extract_unit(&mut dt, 1000) as i32;
            out.us = (dt * 1000) as i32;
        }
        UNIT_US => {
            let days = extract_unit(&mut dt, 1_000_000i64 * 60 * 60 * 24);
            set_days(days, &mut out);
            out.hour = extract_unit(&mut dt, 1_000_000i64 * 60 * 60) as i32;
            out.min = extract_unit(&mut dt, 1_000_000i64 * 60) as i32;
            out.sec = extract_unit(&mut dt, 1_000_000i64) as i32;
            out.us = dt as i32;
        }
        UNIT_NS => {
            let days = extract_unit(&mut dt, 1_000_000_000i64 * 60 * 60 * 24);
            set_days(days, &mut out);
            out.hour = extract_unit(&mut dt, 1_000_000_000i64 * 60 * 60) as i32;
            out.min = extract_unit(&mut dt, 1_000_000_000i64 * 60) as i32;
            out.sec = extract_unit(&mut dt, 1_000_000_000i64) as i32;
            out.us = extract_unit(&mut dt, 1000) as i32;
            out.ps = (dt * 1000) as i32;
        }
        UNIT_PS => {
            let days = extract_unit(&mut dt, 1_000_000_000_000i64 * 60 * 60 * 24);
            set_days(days, &mut out);
            out.hour = extract_unit(&mut dt, 1_000_000_000_000i64 * 60 * 60) as i32;
            out.min = extract_unit(&mut dt, 1_000_000_000_000i64 * 60) as i32;
            out.sec = extract_unit(&mut dt, 1_000_000_000_000i64) as i32;
            out.us = extract_unit(&mut dt, 1_000_000i64) as i32;
            out.ps = dt as i32;
        }
        UNIT_FS => {
            // The whole range is only +-2.6 hours.
            out.hour = extract_unit(&mut dt, 1_000_000_000_000_000i64 * 60 * 60) as i32;
            if out.hour < 0 {
                out.year = 1969;
                out.month = 12;
                out.day = 31;
                out.hour += 24;
            }
            out.min = extract_unit(&mut dt, 1_000_000_000_000_000i64 * 60) as i32;
            out.sec = extract_unit(&mut dt, 1_000_000_000_000_000i64) as i32;
            out.us = extract_unit(&mut dt, 1_000_000_000i64) as i32;
            out.ps = extract_unit(&mut dt, 1000) as i32;
            out.as_ = (dt * 1000) as i32;
        }
        UNIT_AS => {
            // The whole range is only +-9.2 seconds.
            out.sec = extract_unit(&mut dt, 1_000_000_000_000_000_000i64) as i32;
            if out.sec < 0 {
                out.year = 1969;
                out.month = 12;
                out.day = 31;
                out.hour = 23;
                out.min = 59;
                out.sec += 60;
            }
            out.us = extract_unit(&mut dt, 1_000_000_000_000i64) as i32;
            out.ps = extract_unit(&mut dt, 1_000_000i64) as i32;
            out.as_ = dt as i32;
        }
        _ => {
            return Err(Error::RuntimeError(
                "NumPy datetime metadata is corrupted with invalid base unit".into(),
            ))
        }
    }
    Ok(out)
}

/// numpy's `cast_datetime_to_datetime`.
pub fn cast_datetime(src: DtMeta, dst: DtMeta, v: i64) -> Result<i64> {
    if src.base == dst.base && src.num == dst.num {
        return Ok(v);
    }
    if v == NAT {
        return Ok(NAT);
    }
    // numpy uses the datetimestruct round trip when a nonlinear unit is
    // involved, and the exact rational scale factor otherwise.
    if src.base == UNIT_Y || src.base == UNIT_M || dst.base == UNIT_Y || dst.base == UNIT_M {
        let dts = dt64_to_dts(src, v)?;
        return dts_to_dt64(dst, &dts);
    }
    let (num, denom) = conversion_factor(src, dst)?;
    scale_with_overflow_check(v, num, denom, "datetime64")
}

/// numpy's `cast_timedelta_to_timedelta`: always the rational scale factor,
/// so `m8[Y] -> m8[D]` uses the 400-year average.
pub fn cast_timedelta(src: DtMeta, dst: DtMeta, v: i64) -> Result<i64> {
    if (src.base == dst.base && src.num == dst.num) || v == NAT {
        return Ok(v);
    }
    let (num, denom) = conversion_factor(src, dst)?;
    scale_with_overflow_check(v, num, denom, "timedelta64")
}

/// Cast one element the way an *array* cast does, which is not quite the way
/// a scalar conversion does.
///
/// numpy's `get_nbo_cast_datetime_transfer_function` picks the
/// datetimestruct round trip only for a **datetime64** whose source or
/// destination is a nonlinear unit; a timedelta64 always goes through the
/// rational scale factor (so `m8[Y] -> m8[D]` uses the 400-year average).
/// Either way the overflow message names `datetime64`, whatever the kind --
/// probed: `np.array([-2**63+1], 'm8[as]').astype('m8[ms]')` reports
/// "Overflow when converting between datetime64 units".
pub fn cast_value_array(src: DType, dst: DType, v: i64) -> Result<i64> {
    let ms = meta_of(src).expect("datetime dtype");
    let md = meta_of(dst).expect("datetime dtype");
    let same_kind = src.is_datetime() == dst.is_datetime();
    if !same_kind {
        // numpy's datetime <-> timedelta cast keeps the integer value.
        return Ok(v);
    }
    if ms.base == md.base && ms.num == md.num {
        return Ok(v);
    }
    // numpy computes the conversion factor *before* choosing a loop, and
    // fails the whole cast when it overflows -- even for the nonlinear units
    // that would then take the calendar path. `M8[Y] -> M8[ps]` is exactly
    // that case: the Y->ps factor overflows, so numpy raises rather than
    // going through the datetimestruct.
    let (num, denom) = conversion_factor(ms, md)?;
    if v == NAT {
        return Ok(NAT);
    }
    if src.is_datetime()
        && (ms.base == UNIT_Y || ms.base == UNIT_M || md.base == UNIT_Y || md.base == UNIT_M)
    {
        let dts = dt64_to_dts(ms, v)?;
        return dts_to_dt64(md, &dts);
    }
    scale_with_overflow_check(v, num, denom, "datetime64")
}

/// Cast one element between two datetime-like dtypes (they must be the same
/// kind; datetime<->timedelta is a bit-preserving reinterpretation in numpy).
pub fn cast_value(src: DType, dst: DType, v: i64) -> Result<i64> {
    let ms = meta_of(src).expect("datetime dtype");
    let md = meta_of(dst).expect("datetime dtype");
    let same_kind = matches!(
        (src, dst),
        (DType::DateTime(_), DType::DateTime(_)) | (DType::TimeDelta(_), DType::TimeDelta(_))
    );
    if !same_kind {
        // numpy's datetime<->timedelta cast keeps the integer value.
        return Ok(v);
    }
    if matches!(src, DType::TimeDelta(_)) {
        cast_timedelta(ms, md, v)
    } else {
        cast_datetime(ms, md, v)
    }
}

// ---------------------------------------------------------------------------
// ISO 8601
// ---------------------------------------------------------------------------

/// The outcome of parsing an ISO 8601 string.
pub struct ParsedIso {
    pub dts: Dts,
    /// The finest unit the string actually carried.
    pub bestunit: u8,
    /// True for `""`/`NaT`/`today`/`now`.
    pub special: bool,
    /// True when a timezone was present, so the caller can emit numpy's
    /// `UserWarning` at the right moment.
    pub had_timezone: bool,
}

/// numpy's `add_minutes_to_datetimestruct`: carries into hours and days and
/// then fixes up a single month/year boundary (numpy does not loop either).
pub fn add_minutes(dts: &mut Dts, minutes: i32) {
    let mut m = (dts.min + minutes) as i64;
    let hcarry = extract_unit(&mut m, 60);
    dts.min = m as i32;
    let mut h = dts.hour as i64 + hcarry;
    let dcarry = extract_unit(&mut h, 24);
    dts.hour = h as i32;
    dts.day += dcarry as i32;
    if dts.day < 1 {
        dts.month -= 1;
        if dts.month < 1 {
            dts.year -= 1;
            dts.month = 12;
        }
        dts.day += DAYS_PER_MONTH[is_leapyear(dts.year) as usize][dts.month as usize - 1];
    } else if dts.day > 28 {
        let isleap = is_leapyear(dts.year) as usize;
        if dts.day > DAYS_PER_MONTH[isleap][dts.month as usize - 1] {
            dts.day -= DAYS_PER_MONTH[isleap][dts.month as usize - 1];
            dts.month += 1;
            if dts.month > 12 {
                dts.year += 1;
                dts.month = 1;
            }
        }
    }
}

/// numpy's `NpyDatetime_ParseISO8601Datetime`, without the `today`/`now`
/// clock reads (the caller supplies those, so the core stays pure).
pub fn parse_iso8601(s: &str) -> Result<ParsedIso> {
    let mut out = Dts {
        year: 0,
        month: 1,
        day: 1,
        ..Default::default()
    };
    let b = s.as_bytes();
    if b.is_empty() || (b.len() == 3 && s.eq_ignore_ascii_case("nat")) {
        return Ok(ParsedIso {
            dts: Dts::nat(),
            bestunit: UNIT_GENERIC,
            special: true,
            had_timezone: false,
        });
    }
    let mut i = 0usize;
    let n = b.len();
    while i < n && (b[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    let negative_year = i < n && b[i] == b'-';
    if i < n && (b[i] == b'-' || b[i] == b'+') {
        i += 1;
    }
    if i >= n {
        return Err(parse_error(s, i));
    }
    // YEAR
    let mut year: i64 = 0;
    let mut ndigits = 0;
    while i < n && b[i].is_ascii_digit() {
        year = year.wrapping_mul(10).wrapping_add((b[i] - b'0') as i64);
        i += 1;
        ndigits += 1;
    }
    if ndigits == 0 {
        return Err(parse_error(s, i));
    }
    out.year = if negative_year { -year } else { year };
    let year_leap = is_leapyear(out.year) as usize;

    let bestunit;
    'body: {
        if i == n {
            bestunit = UNIT_Y;
            break 'body;
        } else if b[i] == b'-' {
            i += 1;
        } else {
            return Err(parse_error(s, i));
        }
        if i == n {
            return Err(parse_error(s, i));
        }
        // MONTH
        if i + 2 <= n && b[i].is_ascii_digit() && b[i + 1].is_ascii_digit() {
            out.month = 10 * (b[i] - b'0') as i32 + (b[i + 1] - b'0') as i32;
            if out.month < 1 || out.month > 12 {
                return Err(Error::ValueError(format!(
                    "Month out of range in datetime string \"{s}\""
                )));
            }
            i += 2;
        } else {
            return Err(parse_error(s, i));
        }
        if i == n {
            bestunit = UNIT_M;
            break 'body;
        } else if b[i] == b'-' {
            i += 1;
        } else {
            return Err(parse_error(s, i));
        }
        if i == n {
            return Err(parse_error(s, i));
        }
        // DAY
        if i + 2 <= n && b[i].is_ascii_digit() && b[i + 1].is_ascii_digit() {
            out.day = 10 * (b[i] - b'0') as i32 + (b[i + 1] - b'0') as i32;
            if out.day < 1 || out.day > DAYS_PER_MONTH[year_leap][out.month as usize - 1] {
                return Err(Error::ValueError(format!(
                    "Day out of range in datetime string \"{s}\""
                )));
            }
            i += 2;
        } else {
            return Err(parse_error(s, i));
        }
        if i == n {
            bestunit = UNIT_D;
            break 'body;
        } else if b[i] != b'T' && b[i] != b' ' {
            return Err(parse_error(s, i));
        } else {
            i += 1;
        }
        // HOUR
        if i + 2 <= n && b[i].is_ascii_digit() && b[i + 1].is_ascii_digit() {
            out.hour = 10 * (b[i] - b'0') as i32 + (b[i + 1] - b'0') as i32;
            if out.hour >= 24 {
                return Err(Error::ValueError(format!(
                    "Hours out of range in datetime string \"{s}\""
                )));
            }
            i += 2;
        } else {
            return Err(parse_error(s, i));
        }
        if i < n && b[i] == b':' {
            i += 1;
        } else {
            bestunit = UNIT_H;
            break 'body;
        }
        if i == n {
            return Err(parse_error(s, i));
        }
        // MINUTE
        if i + 2 <= n && b[i].is_ascii_digit() && b[i + 1].is_ascii_digit() {
            out.min = 10 * (b[i] - b'0') as i32 + (b[i + 1] - b'0') as i32;
            if out.min >= 60 {
                return Err(Error::ValueError(format!(
                    "Minutes out of range in datetime string \"{s}\""
                )));
            }
            i += 2;
        } else {
            return Err(parse_error(s, i));
        }
        if i < n && b[i] == b':' {
            i += 1;
        } else {
            bestunit = UNIT_MIN;
            break 'body;
        }
        if i == n {
            return Err(parse_error(s, i));
        }
        // SECOND
        if i + 2 <= n && b[i].is_ascii_digit() && b[i + 1].is_ascii_digit() {
            out.sec = 10 * (b[i] - b'0') as i32 + (b[i + 1] - b'0') as i32;
            if out.sec >= 60 {
                return Err(Error::ValueError(format!(
                    "Seconds out of range in datetime string \"{s}\""
                )));
            }
            i += 2;
        } else {
            return Err(parse_error(s, i));
        }
        if i < n && b[i] == b'.' {
            i += 1;
        } else {
            bestunit = UNIT_S;
            break 'body;
        }
        // MICROSECONDS (0..6 digits)
        let mut numdigits = 0;
        for _ in 0..6 {
            out.us *= 10;
            if i < n && b[i].is_ascii_digit() {
                out.us += (b[i] - b'0') as i32;
                i += 1;
                numdigits += 1;
            }
        }
        if i == n || !b[i].is_ascii_digit() {
            bestunit = if numdigits > 3 { UNIT_US } else { UNIT_MS };
            break 'body;
        }
        // PICOSECONDS
        let mut numdigits = 0;
        for _ in 0..6 {
            out.ps *= 10;
            if i < n && b[i].is_ascii_digit() {
                out.ps += (b[i] - b'0') as i32;
                i += 1;
                numdigits += 1;
            }
        }
        if i == n || !b[i].is_ascii_digit() {
            bestunit = if numdigits > 3 { UNIT_PS } else { UNIT_NS };
            break 'body;
        }
        // ATTOSECONDS
        let mut numdigits = 0;
        for _ in 0..6 {
            out.as_ *= 10;
            if i < n && b[i].is_ascii_digit() {
                out.as_ += (b[i] - b'0') as i32;
                i += 1;
                numdigits += 1;
            }
        }
        bestunit = if numdigits > 3 { UNIT_AS } else { UNIT_FS };
    }

    let mut had_timezone = false;
    if i < n {
        had_timezone = true;
        if b[i] == b'Z' {
            if i + 1 == n {
                return Ok(ParsedIso {
                    dts: out,
                    bestunit,
                    special: false,
                    had_timezone,
                });
            }
            i += 1;
        } else if b[i] == b'-' || b[i] == b'+' {
            let neg = b[i] == b'-';
            i += 1;
            let mut oh: i32;
            let mut om = 0i32;
            if i + 2 <= n && b[i].is_ascii_digit() && b[i + 1].is_ascii_digit() {
                oh = 10 * (b[i] - b'0') as i32 + (b[i + 1] - b'0') as i32;
                i += 2;
                if oh >= 24 {
                    return Err(Error::ValueError(format!(
                        "Timezone hours offset out of range in datetime string \"{s}\""
                    )));
                }
            } else {
                return Err(parse_error(s, i));
            }
            if i < n {
                if b[i] == b':' {
                    i += 1;
                }
                if i + 2 <= n && b[i].is_ascii_digit() && b[i + 1].is_ascii_digit() {
                    om = 10 * (b[i] - b'0') as i32 + (b[i + 1] - b'0') as i32;
                    i += 2;
                    if om >= 60 {
                        return Err(Error::ValueError(format!(
                            "Timezone minutes offset out of range in datetime string \"{s}\""
                        )));
                    }
                } else {
                    return Err(parse_error(s, i));
                }
            }
            if neg {
                oh = -oh;
                om = -om;
            }
            add_minutes(&mut out, -60 * oh - om);
        }
        while i < n && (b[i] as char).is_ascii_whitespace() {
            i += 1;
        }
        if i != n {
            return Err(parse_error(s, i));
        }
    }
    let _ = start;
    Ok(ParsedIso {
        dts: out,
        bestunit,
        special: false,
        had_timezone,
    })
}

fn parse_error(s: &str, pos: usize) -> Error {
    Error::ValueError(format!(
        "Error parsing datetime string \"{s}\" at position {pos}"
    ))
}

/// numpy's `lossless_unit_from_datetimestruct`.
pub fn lossless_unit(dts: &Dts) -> u8 {
    if dts.as_ % 1000 != 0 {
        UNIT_AS
    } else if dts.as_ != 0 {
        UNIT_FS
    } else if dts.ps % 1000 != 0 {
        UNIT_PS
    } else if dts.ps != 0 {
        UNIT_NS
    } else if dts.us % 1000 != 0 {
        UNIT_US
    } else if dts.us != 0 {
        UNIT_MS
    } else if dts.sec != 0 {
        UNIT_S
    } else if dts.min != 0 {
        UNIT_MIN
    } else if dts.hour != 0 {
        UNIT_H
    } else if dts.day != 1 {
        UNIT_D
    } else if dts.month != 1 {
        UNIT_M
    } else {
        UNIT_Y
    }
}

/// numpy's `NpyDatetime_MakeISO8601Datetime`, minus the local-timezone
/// support (`local=1` with an automatic offset), which the shim never asks
/// for. `base == None` auto-detects, as numpy's `-1` does.
pub fn make_iso8601(
    dts: &Dts,
    base: Option<u8>,
    utc: bool,
    casting: crate::casting::Casting,
) -> Result<String> {
    if dts.is_nat() || base == Some(UNIT_GENERIC) {
        return Ok("NaT".into());
    }
    let mut base = match base {
        None => {
            let mut b = lossless_unit(dts);
            if b == UNIT_H {
                b = UNIT_MIN;
            } else if b < UNIT_D {
                b = UNIT_D;
            }
            b
        }
        Some(UNIT_W) => UNIT_D,
        Some(b) => b,
    };
    if base == UNIT_B {
        base = UNIT_D;
    }
    if casting != crate::casting::Casting::Unsafe {
        let unitprec = lossless_unit(dts);
        if casting != crate::casting::Casting::SameKind && unitprec > base {
            return Err(Error::TypeError(format!(
                "Cannot create a string with unit precision '{}' from the NumPy \
                 datetime, which has data at unit precision '{}', requires \
                 'unsafe' or 'same_kind' casting",
                UNIT_NAMES[base as usize], UNIT_NAMES[unitprec as usize]
            )));
        }
    }
    // numpy formats the year with `"%04" NPY_INT64_FMT`, whose field width
    // counts the sign: year -1 prints as `-001`, not `-0001`. Rust's `{:04}`
    // has exactly the same rule, so no special case is needed (nor allowed).
    let mut s = format!("{:04}", dts.year);
    if base == UNIT_Y {
        return Ok(s);
    }
    s.push_str(&format!("-{:02}", dts.month));
    if base == UNIT_M {
        return Ok(s);
    }
    s.push_str(&format!("-{:02}", dts.day));
    if base == UNIT_D {
        return Ok(s);
    }
    s.push_str(&format!("T{:02}", dts.hour));
    if base != UNIT_H {
        s.push_str(&format!(":{:02}", dts.min));
        if base != UNIT_MIN {
            s.push_str(&format!(":{:02}", dts.sec));
            if base != UNIT_S {
                s.push_str(&format!(".{:03}", dts.us / 1000));
                if base != UNIT_MS {
                    s.push_str(&format!("{:03}", dts.us % 1000));
                    if base != UNIT_US {
                        s.push_str(&format!("{:03}", dts.ps / 1000));
                        if base != UNIT_NS {
                            s.push_str(&format!("{:03}", dts.ps % 1000));
                            if base != UNIT_PS {
                                s.push_str(&format!("{:03}", dts.as_ / 1000));
                                if base != UNIT_FS {
                                    s.push_str(&format!("{:03}", dts.as_ % 1000));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if utc {
        s.push('Z');
    }
    Ok(s)
}

/// numpy's `NpyDatetime_GetDatetimeISO8601StrLen(0, base)`.
pub fn iso8601_strlen(base: u8) -> usize {
    let mut len = 21usize; // 64-bit year
    if base == UNIT_GENERIC {
        return 4;
    }
    // Fall-through accumulation, coarse to fine.
    if base >= UNIT_M {
        len += 3;
    }
    if base >= UNIT_W {
        len += 3;
    }
    if base >= UNIT_H {
        len += 3;
    }
    if base >= UNIT_MIN {
        len += 3;
    }
    if base >= UNIT_S {
        len += 3;
    }
    if base >= UNIT_MS {
        len += 4;
    }
    for b in [UNIT_US, UNIT_NS, UNIT_PS, UNIT_FS, UNIT_AS] {
        if base >= b {
            len += 3;
        }
    }
    if base >= UNIT_H {
        len += 1; // "Z"
    }
    len + 1
}

/// The plural unit names `str(np.timedelta64(...))` uses.
pub const UNIT_PLURALS: [&str; 15] = [
    "years",
    "months",
    "weeks",
    "business days",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
    "nanoseconds",
    "picoseconds",
    "femtoseconds",
    "attoseconds",
    "generic time units",
];

/// `str(np.timedelta64(v, meta))`: the value scaled by the multiplier, then
/// the plural unit name (numpy prints `np.timedelta64(2,'7s')` as
/// `'14 seconds'`).
pub fn timedelta_str(meta: DtMeta, v: i64) -> String {
    if v == NAT {
        return "NaT".into();
    }
    let scaled = v.wrapping_mul(meta.num as i64);
    format!("{} {}", scaled, UNIT_PLURALS[meta.base as usize])
}

/// The value rendered as a python-visible `str` in the dtype's own unit —
/// what `.astype('U')` and `str(scalar)` produce.
pub fn value_to_string(dt: DType, v: i64) -> Result<String> {
    let m = meta_of(dt).expect("datetime dtype");
    if matches!(dt, DType::TimeDelta(_)) {
        return Ok(timedelta_str(m, v));
    }
    if v == NAT {
        return Ok("NaT".into());
    }
    let dts = dt64_to_dts(m, v)?;
    make_iso8601(
        &dts,
        if m.is_generic() {
            Some(UNIT_GENERIC)
        } else {
            Some(m.base)
        },
        false,
        crate::casting::Casting::Unsafe,
    )
}

/// The `S<n>`/`U<n>` width numpy gives a datetime-like -> string cast.
pub fn string_cast_len(dt: DType) -> usize {
    match meta_of(dt) {
        // Probed: a timedelta string cast is always 21 wide, whatever the unit.
        Some(_) if dt.is_timedelta() => 21,
        Some(m) => iso8601_strlen(m.base),
        None => 0,
    }
}

// ---------------------------------------------------------------------------
// timedelta breakdown, for `.astype(object)`
// ---------------------------------------------------------------------------

/// numpy's `npy_timedeltastruct`: `(day, sec, us, ps, as)`.
pub type Tds = (i64, i32, i32, i32, i32);

/// numpy's `convert_timedelta_to_timedeltastruct`, in full (the sub-microsecond
/// fields included), which is what `timedelta_hash` needs.
pub fn timedelta_struct(meta: DtMeta, td: i64) -> Option<Tds> {
    if td == NAT {
        return None;
    }
    let mut td = td.checked_mul(meta.num as i64)?;
    Some(match meta.base {
        UNIT_W => (td.checked_mul(7)?, 0, 0, 0, 0),
        UNIT_D => (td, 0, 0, 0, 0),
        UNIT_H => {
            let d = extract_unit(&mut td, 24);
            (d, (td * 60 * 60) as i32, 0, 0, 0)
        }
        UNIT_MIN => {
            let d = extract_unit(&mut td, 60 * 24);
            (d, (td * 60) as i32, 0, 0, 0)
        }
        UNIT_S => {
            let d = extract_unit(&mut td, 60 * 60 * 24);
            (d, td as i32, 0, 0, 0)
        }
        UNIT_MS => {
            let d = extract_unit(&mut td, 1000 * 60 * 60 * 24);
            let s = extract_unit(&mut td, 1000);
            (d, s as i32, (td * 1000) as i32, 0, 0)
        }
        UNIT_US => {
            let d = extract_unit(&mut td, 1_000_000i64 * 60 * 60 * 24);
            let s = extract_unit(&mut td, 1_000_000i64);
            (d, s as i32, td as i32, 0, 0)
        }
        UNIT_NS => {
            let d = extract_unit(&mut td, 1_000_000_000i64 * 60 * 60 * 24);
            let s = extract_unit(&mut td, 1_000_000_000i64);
            let u = extract_unit(&mut td, 1000);
            (d, s as i32, u as i32, (td * 1000) as i32, 0)
        }
        UNIT_PS => {
            let d = extract_unit(&mut td, 1_000_000_000_000i64 * 60 * 60 * 24);
            let s = extract_unit(&mut td, 1_000_000_000_000i64);
            let u = extract_unit(&mut td, 1_000_000i64);
            (d, s as i32, u as i32, td as i32, 0)
        }
        UNIT_FS => {
            let s = extract_unit(&mut td, 1_000_000_000_000_000i64);
            let u = extract_unit(&mut td, 1_000_000_000i64);
            let p = extract_unit(&mut td, 1000);
            (0, s as i32, u as i32, p as i32, (td * 1000) as i32)
        }
        UNIT_AS => {
            let s = extract_unit(&mut td, 1_000_000_000_000_000_000i64);
            let u = extract_unit(&mut td, 1_000_000_000_000i64);
            let p = extract_unit(&mut td, 1_000_000i64);
            (0, s as i32, u as i32, p as i32, td as i32)
        }
        _ => return None,
    })
}

/// numpy's `convert_timedelta_to_timedeltastruct`: (days, secs, us) for the
/// units python's `datetime.timedelta` can represent.
pub fn timedelta_parts(meta: DtMeta, td: i64) -> Option<(i64, i32, i32)> {
    if td == NAT {
        return None;
    }
    let mut td = td.checked_mul(meta.num as i64)?;
    let (days, sec, us) = match meta.base {
        UNIT_W => (td.checked_mul(7)?, 0, 0),
        UNIT_D => (td, 0, 0),
        UNIT_H => {
            let d = extract_unit(&mut td, 24);
            (d, (td * 60 * 60) as i32, 0)
        }
        UNIT_MIN => {
            let d = extract_unit(&mut td, 60 * 24);
            let h = extract_unit(&mut td, 60);
            (d, (h * 60 * 60 + td * 60) as i32, 0)
        }
        UNIT_S => {
            let d = extract_unit(&mut td, 60 * 60 * 24);
            (d, td as i32, 0)
        }
        UNIT_MS => {
            let d = extract_unit(&mut td, 1000 * 60 * 60 * 24);
            let s = extract_unit(&mut td, 1000);
            (d, s as i32, (td * 1000) as i32)
        }
        UNIT_US => {
            let d = extract_unit(&mut td, 1_000_000i64 * 60 * 60 * 24);
            let s = extract_unit(&mut td, 1_000_000i64);
            (d, s as i32, td as i32)
        }
        _ => return None,
    };
    Some((days, sec, us))
}

// ---------------------------------------------------------------------------
// Conversion to Python's `datetime` objects
// ---------------------------------------------------------------------------

/// What numpy's `convert_datetime_to_pyobject` / `convert_timedelta_to_pyobject`
/// hand back for a single datetime-like value. The Python layer turns this into
/// the actual object; the rules for *which* object live here, next to the
/// calendar code they depend on.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PyDtObj {
    /// NaT, or any value with generic units: `None`.
    Nothing,
    /// The raw stored integer, for the units and magnitudes Python's
    /// `datetime` module cannot represent.
    Int(i64),
    /// `datetime.date(year, month, day)`.
    Date { year: i64, month: i32, day: i32 },
    /// `datetime.datetime(year, month, day, hour, min, sec, us)`.
    DateTime {
        year: i64,
        month: i32,
        day: i32,
        hour: i32,
        min: i32,
        sec: i32,
        us: i32,
    },
    /// `datetime.timedelta(days, seconds, microseconds)`.
    Delta { days: i64, secs: i32, us: i32 },
}

/// numpy's `convert_datetime_to_pyobject` (`datetime.c`).
pub fn datetime_to_pyobj(meta: DtMeta, dt: i64) -> PyDtObj {
    if dt == NAT || meta.is_generic() {
        return PyDtObj::Nothing;
    }
    // Anything finer than microseconds has no `datetime` representation.
    if meta.base > UNIT_US {
        return PyDtObj::Int(dt);
    }
    let Ok(dts) = dt64_to_dts(meta, dt) else {
        return PyDtObj::Int(dt);
    };
    // Outside `datetime`'s year range, or on a leap second, numpy gives up and
    // returns the raw integer.
    if dts.year < 1 || dts.year > 9999 || dts.sec == 60 {
        return PyDtObj::Int(dt);
    }
    if meta.base > UNIT_D {
        PyDtObj::DateTime {
            year: dts.year,
            month: dts.month,
            day: dts.day,
            hour: dts.hour,
            min: dts.min,
            sec: dts.sec,
            us: dts.us,
        }
    } else {
        PyDtObj::Date {
            year: dts.year,
            month: dts.month,
            day: dts.day,
        }
    }
}

/// numpy's `convert_timedelta_to_pyobject` (`datetime.c`).
pub fn timedelta_to_pyobj(meta: DtMeta, td: i64) -> PyDtObj {
    if td == NAT {
        return PyDtObj::Nothing;
    }
    // Sub-microsecond precision, the nonlinear units and generic units all
    // fall back to the raw integer.
    if meta.base > UNIT_US
        || meta.base == UNIT_Y
        || meta.base == UNIT_M
        || meta.is_generic()
    {
        return PyDtObj::Int(td);
    }
    let Some((days, secs, us)) = timedelta_parts(meta, td) else {
        return PyDtObj::Int(td);
    };
    // `datetime.timedelta` tops out at +-999999999 days.
    if days < -999_999_999 || days > 999_999_999 {
        return PyDtObj::Int(td);
    }
    PyDtObj::Delta { days, secs, us }
}

/// [`datetime_to_pyobj`] / [`timedelta_to_pyobj`], dispatched on the dtype.
/// `None` when `dt` is not a datetime-like dtype at all.
pub fn value_to_pyobj(dt: DType, v: i64) -> Option<PyDtObj> {
    let meta = meta_of(dt)?;
    Some(match dt {
        DType::TimeDelta(_) => timedelta_to_pyobj(meta, v),
        _ => datetime_to_pyobj(meta, v),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_years_pad_like_numpy() {
        // numpy's `%04lld` counts the sign toward the field width.
        for (year, want) in [
            (-1i64, "-001"),
            (-12, "-012"),
            (-123, "-123"),
            (-1234, "-1234"),
            (-12345, "-12345"),
            (0, "0000"),
            (1, "0001"),
            (123, "0123"),
            (12345, "12345"),
        ] {
            let dts = Dts {
                year,
                month: 1,
                day: 1,
                ..Default::default()
            };
            assert_eq!(
                make_iso8601(&dts, Some(UNIT_Y), false, crate::casting::Casting::Unsafe).unwrap(),
                want,
                "year {year}"
            );
        }
    }

    #[test]
    fn pyobj_conversion_matches_numpy() {
        let m = DtMeta::unit;
        // M8: date for D and coarser, datetime for h..us, int beyond.
        assert_eq!(
            datetime_to_pyobj(m(UNIT_Y), -1),
            PyDtObj::Date {
                year: 1969,
                month: 1,
                day: 1
            }
        );
        assert!(matches!(
            datetime_to_pyobj(m(UNIT_H), -7),
            PyDtObj::DateTime { year: 1969, .. }
        ));
        assert_eq!(datetime_to_pyobj(m(UNIT_NS), 1000), PyDtObj::Int(1000));
        // Out of `datetime`'s year range -> raw int.
        assert_eq!(
            datetime_to_pyobj(m(UNIT_Y), 2147483648),
            PyDtObj::Int(2147483648)
        );
        assert_eq!(datetime_to_pyobj(m(UNIT_D), NAT), PyDtObj::Nothing);
        // m8: Y/M and sub-us are ints, W..us are timedeltas.
        assert_eq!(timedelta_to_pyobj(m(UNIT_Y), -1), PyDtObj::Int(-1));
        assert_eq!(timedelta_to_pyobj(m(UNIT_M), 7), PyDtObj::Int(7));
        assert_eq!(
            timedelta_to_pyobj(m(UNIT_W), -1),
            PyDtObj::Delta {
                days: -7,
                secs: 0,
                us: 0
            }
        );
        // Beyond timedelta's +-999999999 day range -> raw int.
        assert_eq!(
            timedelta_to_pyobj(m(UNIT_MS), i64::MAX),
            PyDtObj::Int(i64::MAX)
        );
        assert_eq!(timedelta_to_pyobj(m(UNIT_NS), 1), PyDtObj::Int(1));
        assert_eq!(timedelta_to_pyobj(m(UNIT_S), NAT), PyDtObj::Nothing);
    }

    #[test]
    fn mixed_datetime_timedelta_promotion_is_nonstrict() {
        // np.promote_types('M8[D]', 'm8[Y]') -> M8[D], but the same pair of
        // timedeltas is a TypeError.
        let m8y = timedelta(DtMeta::unit(UNIT_Y));
        let m8d = timedelta(DtMeta::unit(UNIT_D));
        let m8m = timedelta(DtMeta::unit(UNIT_M));
        let dt_d = datetime(DtMeta::unit(UNIT_D));
        let dt_ns = datetime(DtMeta::unit(UNIT_NS));
        assert_eq!(promote_meta(dt_d, m8y).unwrap(), dt_d);
        assert_eq!(promote_meta(m8y, dt_d).unwrap(), dt_d);
        assert_eq!(promote_meta(m8m, dt_ns).unwrap(), dt_ns);
        assert!(promote_meta(m8d, m8y).is_err());
        // The overflow cases stay overflows.
        let dt_as = datetime(DtMeta::unit(UNIT_AS));
        assert!(matches!(
            promote_meta(dt_as, m8m),
            Err(Error::OverflowError(_))
        ));
    }

    #[test]
    fn civil_round_trips() {
        // Walk every day over a wide range and check both directions.
        for days in (-800_000i64..800_000).step_by(7) {
            let mut dts = Dts::epoch();
            set_days(days, &mut dts);
            assert_eq!(days_from_civil(&dts), days, "days {days} -> {dts:?}");
        }
        // Known anchors, probed from numpy.
        for (y, m, d, want) in [
            (1970, 1, 1, 0i64),
            (1969, 12, 31, -1),
            (2000, 3, 1, 11017),
            (1600, 2, 29, -135081),
            (-4713, 11, 24, -2440588),
            (9999, 12, 31, 2932896),
        ] {
            let dts = Dts {
                year: y,
                month: m,
                day: d,
                ..Default::default()
            };
            assert_eq!(days_from_civil(&dts), want, "{y}-{m}-{d}");
            let mut back = Dts::epoch();
            set_days(want, &mut back);
            assert_eq!((back.year, back.month, back.day), (y, m, d));
        }
    }

    #[test]
    fn leapyears_match_numpy() {
        for y in [1600i64, 1700, 1800, 1900, 2000, 2004, 2100, -1, -4, -400] {
            let want = (y % 4 == 0) && (y % 100 != 0 || y % 400 == 0);
            assert_eq!(is_leapyear(y), want, "year {y}");
        }
    }

    #[test]
    fn unit_conversion_matches_numpy() {
        // np.timedelta64(1, 'D').astype('m8[h]') == 24
        let d = DtMeta::unit(UNIT_D);
        let h = DtMeta::unit(UNIT_H);
        assert_eq!(cast_timedelta(d, h, 1).unwrap(), 24);
        // Fine -> coarse floors toward negative infinity.
        assert_eq!(cast_timedelta(h, d, 25).unwrap(), 1);
        assert_eq!(cast_timedelta(h, d, -1).unwrap(), -1);
        assert_eq!(cast_timedelta(h, d, -25).unwrap(), -2);
        // Years are the 400-year average for timedelta.
        let y = DtMeta::unit(UNIT_Y);
        assert_eq!(cast_timedelta(y, d, 1).unwrap(), 365);
        // Datetimes go through the calendar instead.
        assert_eq!(cast_datetime(y, d, 1).unwrap(), 365);
        assert_eq!(cast_datetime(y, d, 2).unwrap(), 730);
        assert_eq!(cast_datetime(d, y, 364).unwrap(), 0);
        assert_eq!(cast_datetime(d, y, 365).unwrap(), 1);
        assert_eq!(cast_datetime(d, y, -1).unwrap(), -1);
    }

    #[test]
    fn nat_is_preserved_by_casts() {
        let s = DtMeta::unit(UNIT_S);
        let ns = DtMeta::unit(UNIT_NS);
        assert_eq!(cast_timedelta(s, ns, NAT).unwrap(), NAT);
        assert_eq!(cast_datetime(s, ns, NAT).unwrap(), NAT);
        assert_eq!(cast_datetime(DtMeta::unit(UNIT_Y), ns, NAT).unwrap(), NAT);
    }

    /// `(timedelta, datetime)` promotion tables generated straight from real
    /// numpy 2.5.2 by `harness/gen_tables.py`.
    #[allow(clippy::type_complexity)]
    const UNIT_TABLES: (
        &[(u8, u8, Option<(u8, u32)>)],
        &[(u8, u8, Option<(u8, u32)>)],
    ) = include!("datetime_units.inc");

    #[test]
    fn unit_promotion_matches_numpy_exactly() {
        let (td, dt) = UNIT_TABLES;
        assert_eq!(td.len(), 196);
        assert_eq!(dt.len(), 196);
        for (&(a, b, want), strict) in td
            .iter()
            .map(|e| (e, true))
            .chain(dt.iter().map(|e| (e, false)))
        {
            let got = gcd_meta(DtMeta::unit(a), DtMeta::unit(b), strict, strict).ok();
            let got = got.map(|m| (m.base, m.num));
            assert_eq!(
                got,
                want,
                "gcd({}, {}) strict={strict}",
                UNIT_NAMES[a as usize],
                UNIT_NAMES[b as usize]
            );
        }
    }

    #[test]
    fn gcd_meta_matches_numpy_table() {
        // A few rows probed from np.promote_types on m8.
        let g = |a: u8, b: u8| gcd_meta(DtMeta::unit(a), DtMeta::unit(b), true, true);
        assert_eq!(g(UNIT_W, UNIT_D).unwrap(), DtMeta::unit(UNIT_D));
        assert_eq!(g(UNIT_Y, UNIT_M).unwrap(), DtMeta::unit(UNIT_M));
        assert!(g(UNIT_Y, UNIT_D).is_err());
        assert!(g(UNIT_W, UNIT_PS).is_err());
        assert_eq!(g(UNIT_MS, UNIT_AS).unwrap(), DtMeta::unit(UNIT_AS));
        // Multipliers take the GCD.
        assert_eq!(
            gcd_meta(DtMeta::new(UNIT_D, 6), DtMeta::new(UNIT_D, 4), true, true).unwrap(),
            DtMeta::new(UNIT_D, 2)
        );
    }

    #[test]
    fn iso_parse_and_format() {
        let p = parse_iso8601("2020-01-02T03:04:05.123456789").unwrap();
        assert_eq!(p.bestunit, UNIT_NS);
        assert_eq!(p.dts.us, 123456);
        assert_eq!(p.dts.ps, 789000);
        let v = dts_to_dt64(DtMeta::unit(UNIT_NS), &p.dts).unwrap();
        assert_eq!(v, 1577934245123456789);
        assert_eq!(
            value_to_string(DType::DateTime(DtMeta::unit(UNIT_NS).pack()), v).unwrap(),
            "2020-01-02T03:04:05.123456789"
        );
        assert_eq!(parse_iso8601("2020").unwrap().bestunit, UNIT_Y);
        assert_eq!(parse_iso8601("2020-01").unwrap().bestunit, UNIT_M);
        assert_eq!(parse_iso8601("2020-01-02").unwrap().bestunit, UNIT_D);
        assert!(parse_iso8601("NaT").unwrap().dts.is_nat());
        assert!(parse_iso8601("").unwrap().dts.is_nat());
    }
}
