//! Floating-point error flags (numpy's `np.errstate` / `np.seterr` model).
//!
//! numpy gets these from the CPU's FP status register. The port raises them
//! explicitly from the inner loops instead: the loops already have the operand
//! and result values in registers, and an explicit test is the only portable
//! way to attribute an integer divide-by-zero (for which no hardware flag
//! exists) to the right ufunc.
//!
//! The accumulator is a process-global atomic rather than a thread-local
//! because the elementwise loops split across rayon workers; the ufunc entry
//! points on the Python side drain it once the loop has joined.

use std::sync::atomic::{AtomicU8, Ordering};

pub const DIVIDE: u8 = 1;
pub const OVER: u8 = 2;
pub const UNDER: u8 = 4;
pub const INVALID: u8 = 8;

static FLAGS: AtomicU8 = AtomicU8::new(0);

/// Whether the underflow condition is worth detecting.
///
/// numpy's default `under` action is 'ignore', and the only way to spot an
/// underflow from the result alone is to treat *every* zero product as a
/// candidate -- which drags `multiply` down by 4x on arrays that legitimately
/// contain zeros. So the check is compiled into the loop but gated on the
/// error state: `np.seterr(under=...)`/`np.errstate(under=...)` turns it on.
static WATCH_UNDER: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_watch_underflow(on: bool) {
    WATCH_UNDER.store(on, Ordering::Relaxed);
}

#[inline]
pub fn watch_underflow() -> bool {
    WATCH_UNDER.load(Ordering::Relaxed)
}

/// Record one or more error conditions.
#[inline]
pub fn raise(f: u8) {
    if f != 0 {
        FLAGS.fetch_or(f, Ordering::Relaxed);
    }
}

/// Read and clear the accumulated flags.
pub fn take() -> u8 {
    FLAGS.swap(0, Ordering::Relaxed)
}

/// Read the flags without clearing them.
pub fn peek() -> u8 {
    FLAGS.load(Ordering::Relaxed)
}

pub fn clear() {
    FLAGS.store(0, Ordering::Relaxed);
}

/// The flag a float result implies, given whether the operation was a
/// division-like one and whether any input was already non-finite.
///
/// numpy's rules, probed op by op:
///  * a NaN produced from non-NaN inputs is `invalid`
///  * an infinity produced from finite inputs is `overflow`, except for the
///    division-like ops (`divide`, `log`, `arctanh`, ...) where a zero
///    denominator reports `divide by zero` instead
#[inline]
pub fn float_flag(result_nan: bool, result_inf: bool, inputs_finite: bool, any_nan_in: bool,
                  divide_like: bool) -> u8 {
    if result_nan && !any_nan_in {
        return INVALID;
    }
    if result_inf && inputs_finite {
        return if divide_like { DIVIDE } else { OVER };
    }
    0
}
