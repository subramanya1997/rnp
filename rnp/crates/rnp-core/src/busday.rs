//! Business-day calendar operations for `datetime64[D]` values.
//!
//! This is a direct Rust transcription of NumPy 2.5.2's
//! `datetime_busday.c` and the calendar normalization in
//! `datetime_busdaycal.c`. Dates are signed counts from 1970-01-01 and
//! holidays are kept sorted, unique, and restricted to days enabled by the
//! weekmask.

use crate::datetime::{self as dtm, NAT};
use crate::error::{Error, Result};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Roll {
    Following,
    ModifiedFollowing,
    Preceding,
    ModifiedPreceding,
    Nat,
    Raise,
}

impl Roll {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "forward" | "following" => Ok(Self::Following),
            "modifiedfollowing" => Ok(Self::ModifiedFollowing),
            "backward" | "preceding" => Ok(Self::Preceding),
            "modifiedpreceding" => Ok(Self::ModifiedPreceding),
            "nat" => Ok(Self::Nat),
            "raise" => Ok(Self::Raise),
            _ => Err(Error::ValueError(format!(
                "Invalid business day roll parameter \"{value}\""
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusDayCalendar {
    weekmask: [bool; 7],
    busdays_in_weekmask: i64,
    holidays: Vec<i64>,
}

impl BusDayCalendar {
    /// Construct and normalize a calendar like `normalize_holidays_list`.
    pub fn new(weekmask: [bool; 7], mut holidays: Vec<i64>) -> Result<Self> {
        let busdays_in_weekmask = weekmask.iter().filter(|&&v| v).count() as i64;
        if busdays_in_weekmask == 0 {
            return Err(Error::ValueError(
                "the business day weekmask must have at least one valid business day".into(),
            ));
        }
        holidays.sort_unstable();
        holidays.dedup();
        holidays.retain(|&date| date != NAT && weekmask[day_of_week(date)]);
        Ok(Self {
            weekmask,
            busdays_in_weekmask,
            holidays,
        })
    }

    pub fn weekmask(&self) -> [bool; 7] {
        self.weekmask
    }

    pub fn holidays(&self) -> &[i64] {
        &self.holidays
    }

    pub fn is_business_day(&self, date: i64) -> bool {
        date != NAT
            && self.weekmask[day_of_week(date)]
            && self.holidays.binary_search(&date).is_err()
    }

    /// `apply_business_day_offset` from `datetime_busday.c`.
    pub fn offset(&self, date: i64, mut offset: i64, roll: Roll) -> Result<i64> {
        let (mut date, mut dow) = self.apply_roll(date, roll)?;
        if date == NAT {
            return Ok(NAT);
        }

        if offset > 0 {
            let mut holidays_begin = self.holidays.partition_point(|&h| h < date);
            let weeks = offset / self.busdays_in_weekmask;
            date = checked_add_days(date, weeks * 7)?;
            offset %= self.busdays_in_weekmask;

            let holidays_temp = self.holidays.partition_point(|&h| h <= date);
            offset += (holidays_temp - holidays_begin) as i64;
            holidays_begin = holidays_temp;

            while offset > 0 {
                date = checked_add_days(date, 1)?;
                dow = (dow + 1) % 7;
                if self.weekmask[dow]
                    && self.holidays[holidays_begin..]
                        .binary_search(&date)
                        .is_err()
                {
                    offset -= 1;
                }
            }
        } else if offset < 0 {
            let mut holidays_end = self.holidays.partition_point(|&h| h <= date);
            let weeks = offset / self.busdays_in_weekmask;
            date = checked_add_days(date, weeks * 7)?;
            offset %= self.busdays_in_weekmask;

            let holidays_temp = self.holidays[..holidays_end].partition_point(|&h| h < date);
            offset -= (holidays_end - holidays_temp) as i64;
            holidays_end = holidays_temp;

            while offset < 0 {
                date = checked_add_days(date, -1)?;
                dow = (dow + 6) % 7;
                if self.weekmask[dow] && self.holidays[..holidays_end].binary_search(&date).is_err()
                {
                    offset += 1;
                }
            }
        }
        Ok(date)
    }

    /// `apply_business_day_count`: count `[begin, end)`, preserving NumPy's
    /// asymmetric reversed-range boundary correction (gh-23197).
    pub fn count(&self, mut begin: i64, mut end: i64) -> Result<i64> {
        if begin == NAT || end == NAT {
            return Err(Error::ValueError(
                "Cannot compute a business day count with a NaT (not-a-time) date".into(),
            ));
        }
        if begin == end {
            return Ok(0);
        }

        let swapped = begin > end;
        if swapped {
            std::mem::swap(&mut begin, &mut end);
            begin = checked_add_days(begin, 1)?;
            end = checked_add_days(end, 1)?;
        }

        let holidays_begin = self.holidays.partition_point(|&h| h < begin);
        let holidays_end = self.holidays.partition_point(|&h| h < end);
        let mut count = -((holidays_end - holidays_begin) as i64);

        let whole_weeks = (end - begin) / 7;
        count += whole_weeks * self.busdays_in_weekmask;
        begin += whole_weeks * 7;

        let mut dow = day_of_week(begin);
        while begin < end {
            if self.weekmask[dow] {
                count += 1;
            }
            begin += 1;
            dow = (dow + 1) % 7;
        }
        Ok(if swapped { -count } else { count })
    }

    fn apply_roll(&self, mut date: i64, roll: Roll) -> Result<(i64, usize)> {
        if date == NAT {
            if roll == Roll::Raise {
                return Err(Error::ValueError("NaT input in busday_offset".into()));
            }
            return Ok((NAT, 0));
        }

        let mut dow = day_of_week(date);
        if self.is_business_day(date) {
            return Ok((date, dow));
        }
        let start_date = date;
        let start_dow = dow;
        match roll {
            Roll::Following | Roll::ModifiedFollowing => {
                loop {
                    date = checked_add_days(date, 1)?;
                    dow = (dow + 1) % 7;
                    if self.is_business_day(date) {
                        break;
                    }
                }
                if roll == Roll::ModifiedFollowing && month_number(start_date) != month_number(date)
                {
                    date = start_date;
                    dow = start_dow;
                    loop {
                        date = checked_add_days(date, -1)?;
                        dow = (dow + 6) % 7;
                        if self.is_business_day(date) {
                            break;
                        }
                    }
                }
            }
            Roll::Preceding | Roll::ModifiedPreceding => {
                loop {
                    date = checked_add_days(date, -1)?;
                    dow = (dow + 6) % 7;
                    if self.is_business_day(date) {
                        break;
                    }
                }
                if roll == Roll::ModifiedPreceding && month_number(start_date) != month_number(date)
                {
                    date = start_date;
                    dow = start_dow;
                    loop {
                        date = checked_add_days(date, 1)?;
                        dow = (dow + 1) % 7;
                        if self.is_business_day(date) {
                            break;
                        }
                    }
                }
            }
            Roll::Nat => return Ok((NAT, dow)),
            Roll::Raise => {
                return Err(Error::ValueError(
                    "Non-business day date in busday_offset".into(),
                ))
            }
        }
        Ok((date, dow))
    }
}

/// 1970-01-05 was a Monday, so 1970-01-01 (day zero) is Thursday.
fn day_of_week(date: i64) -> usize {
    (date - 4).rem_euclid(7) as usize
}

fn month_number(date: i64) -> i64 {
    let mut dts = dtm::Dts::epoch();
    dtm::set_days(date, &mut dts);
    dts.year * 12 + dts.month as i64
}

fn checked_add_days(date: i64, days: i64) -> Result<i64> {
    date.checked_add(days)
        .ok_or_else(|| Error::OverflowError("Integer overflow in business day operation".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weekdays(holidays: Vec<i64>) -> BusDayCalendar {
        BusDayCalendar::new([true, true, true, true, true, false, false], holidays).unwrap()
    }

    #[test]
    fn normalization_drops_nat_duplicates_and_weekends() {
        let cal = weekdays(vec![NAT, 3, 4, 4, 7, 358]);
        assert_eq!(cal.holidays(), &[4, 7, 358]);
    }

    #[test]
    fn offset_and_roll_match_numpy_examples() {
        let cal = weekdays(vec![]);
        assert_eq!(cal.offset(0, 0, Roll::Following).unwrap(), 0);
        assert_eq!(cal.offset(2, 0, Roll::Preceding).unwrap(), 1);
        assert_eq!(cal.offset(0, 25, Roll::Raise).unwrap(), 35);
        assert_eq!(cal.offset(35, -25, Roll::Raise).unwrap(), 0);
        assert_eq!(cal.offset(NAT, 1, Roll::Following).unwrap(), NAT);
    }

    #[test]
    fn holidays_and_reversed_counts_match_numpy_boundaries() {
        let cal = weekdays(vec![1]);
        assert_eq!(cal.offset(0, 1, Roll::Raise).unwrap(), 4);
        assert_eq!(cal.count(0, 4).unwrap(), 1);
        assert_eq!(cal.count(4, 0).unwrap(), -1);
        assert_eq!(cal.count(0, 1).unwrap(), 1);
        assert_eq!(cal.count(1, 0).unwrap(), 0);
    }
}
