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

// ---------------------------------------------------------------------------
// The CPU's own IEEE exception flags
// ---------------------------------------------------------------------------
//
// Everywhere else the port derives the flags from the operand and result
// values, because that is the only portable way to attribute an *integer*
// divide-by-zero and because an explicit test keeps the hot loops free of any
// dependency on the FP environment.
//
// The complex transcendentals are the one place where that cannot work. numpy
// `#define`s `npy_csin` and friends straight to the C library's `csin` (see
// `HAVE_CSIN` in its generated `_numpyconfig.h`; the fallbacks in
// `npy_math_complex.c.src` are only compiled where the platform lacks them), so
// the conditions numpy reports for, say, `arctanh(1.8e308+0j)` are whatever
// *that* libm happened to signal internally -- there is no value-level rule to
// transcribe. This port calls the same libm, so reading the same status
// register reproduces numpy exactly, which is also how numpy's own complex
// loops do it (`npy_get_floatstatus_barrier`).

// C99 <fenv.h> exception bits. The values are the ABI's, not ours, so they are
// queried through the mask constants the platform defines.
#[cfg(target_arch = "aarch64")]
mod fe {
    pub const INEXACT: i32 = 0x10;
    pub const UNDERFLOW: i32 = 0x08;
    pub const OVERFLOW: i32 = 0x04;
    pub const DIVBYZERO: i32 = 0x02;
    pub const INVALID: i32 = 0x01;
}
#[cfg(not(target_arch = "aarch64"))]
mod fe {
    pub const INEXACT: i32 = 0x20;
    pub const UNDERFLOW: i32 = 0x10;
    pub const OVERFLOW: i32 = 0x08;
    pub const DIVBYZERO: i32 = 0x04;
    pub const INVALID: i32 = 0x01;
}

const FE_ALL: i32 =
    fe::INEXACT | fe::UNDERFLOW | fe::OVERFLOW | fe::DIVBYZERO | fe::INVALID;

extern "C" {
    fn feclearexcept(excepts: i32) -> i32;
    fn fetestexcept(excepts: i32) -> i32;
}

/// Clear the calling thread's IEEE exception flags.
///
/// Must bracket the loop together with [`hw_take`]; the register is per thread,
/// so a loop split across rayon has to clear and read inside each chunk.
#[inline]
pub fn hw_clear() {
    // SAFETY: `feclearexcept` is a pure C99 libm call with no memory effects;
    // the argument is the platform's own FE_* mask.
    unsafe {
        feclearexcept(FE_ALL);
    }
}

/// This thread's IEEE exception flags, translated to this module's bits, and
/// cleared. `inexact` is dropped: numpy never reports it.
#[inline]
pub fn hw_take() -> u8 {
    // SAFETY: as `hw_clear`.
    let raised = unsafe {
        let r = fetestexcept(FE_ALL);
        feclearexcept(FE_ALL);
        r
    };
    let mut f = 0u8;
    if raised & fe::DIVBYZERO != 0 {
        f |= DIVIDE;
    }
    if raised & fe::OVERFLOW != 0 {
        f |= OVER;
    }
    if raised & fe::INVALID != 0 {
        f |= INVALID;
    }
    // numpy's default `under` action is 'ignore' and the hardware raises it far
    // more eagerly than the value-level test elsewhere in this port does, so it
    // is only collected when the error state asks for it.
    if watch_underflow() && raised & fe::UNDERFLOW != 0 {
        f |= UNDER;
    }
    f
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
