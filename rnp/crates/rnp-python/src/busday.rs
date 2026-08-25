//! Python bindings for the pure-Rust business-day kernels.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use rnp_core::busday::{BusDayCalendar, Roll};

fn calendar(weekmask: Vec<bool>, holidays: Vec<i64>) -> PyResult<BusDayCalendar> {
    let weekmask: [bool; 7] = weekmask.try_into().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("A business day weekmask array must have length 7")
    })?;
    BusDayCalendar::new(weekmask, holidays).map_err(crate::err)
}

#[pyfunction]
fn _busday_normalize(weekmask: Vec<bool>, holidays: Vec<i64>) -> PyResult<Vec<i64>> {
    Ok(calendar(weekmask, holidays)?.holidays().to_vec())
}

#[pyfunction]
fn _busday_offset(
    dates: Vec<i64>,
    offsets: Vec<i64>,
    roll: &str,
    weekmask: Vec<bool>,
    holidays: Vec<i64>,
) -> PyResult<Vec<i64>> {
    if dates.len() != offsets.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "busday_offset inputs were not broadcast to the same size",
        ));
    }
    let cal = calendar(weekmask, holidays)?;
    let roll = Roll::parse(roll).map_err(crate::err)?;
    dates
        .into_iter()
        .zip(offsets)
        .map(|(date, offset)| cal.offset(date, offset, roll).map_err(crate::err))
        .collect()
}

#[pyfunction]
fn _busday_count(
    begins: Vec<i64>,
    ends: Vec<i64>,
    weekmask: Vec<bool>,
    holidays: Vec<i64>,
) -> PyResult<Vec<i64>> {
    if begins.len() != ends.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "busday_count inputs were not broadcast to the same size",
        ));
    }
    let cal = calendar(weekmask, holidays)?;
    begins
        .into_iter()
        .zip(ends)
        .map(|(begin, end)| cal.count(begin, end).map_err(crate::err))
        .collect()
}

#[pyfunction]
fn _is_busday(dates: Vec<i64>, weekmask: Vec<bool>, holidays: Vec<i64>) -> PyResult<Vec<bool>> {
    let cal = calendar(weekmask, holidays)?;
    Ok(dates
        .into_iter()
        .map(|date| cal.is_business_day(date))
        .collect())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_busday_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(_busday_offset, m)?)?;
    m.add_function(wrap_pyfunction!(_busday_count, m)?)?;
    m.add_function(wrap_pyfunction!(_is_busday, m)?)?;
    Ok(())
}
