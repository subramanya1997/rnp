//! The datetime64 / timedelta64 ufunc loops and their type resolution.
//!
//! Type resolution is a port of `PyUFunc_AdditionTypeResolver` and friends in
//! `numpy/_core/src/umath/ufunc_type_resolution.c`; the inner loops are ports
//! of the `DATETIME_*` / `TIMEDELTA_*` bodies in `loops.c.src`. numpy's loops
//! raise `OverflowError` rather than wrapping, and treat a result that lands
//! exactly on `NPY_DATETIME_NAT` as overflow too, because it would otherwise
//! be silently misread as NaT.

use crate::array::NdArray;
use crate::datetime::{self as dtm, NAT};
use crate::dtype::DType;
use crate::element::Scalar;
use crate::error::{
    ufunc_binary_resolution, ufunc_input_casting, ufunc_no_loop, Error, Result,
};
use crate::iter::{broadcast_shapes, broadcast_to, offsets};
use crate::ops::BinOp;

/// The dtype family name the `_UFuncNoLoopError` payload carries.
fn family(d: DType) -> String {
    match d {
        DType::DateTime(_) => "datetime64".into(),
        DType::TimeDelta(_) => "timedelta64".into(),
        DType::Bytes(_) => "bytes".into(),
        DType::Str(_) => "str".into(),
        DType::Void(_) | DType::Struct(_) | DType::SubArray(_) => "void".into(),
        DType::Object => "object".into(),
        other => other.name(),
    }
}

fn no_loop(op: BinOp, a: DType, b: DType) -> Error {
    let (fa, fb) = (family(a), family(b));
    ufunc_no_loop(op.name(), &[&fa, &fb])
}

fn reso_error(op: BinOp, a: &NdArray, b: &NdArray) -> Error {
    ufunc_binary_resolution(op.name(), &a.descr.str_code(), &b.descr.str_code())
}

/// numpy's default ufunc casting rule for the inputs.
const UFUNC_CASTING: crate::casting::Casting = crate::casting::Casting::SameKind;

/// Reject input `i` when it cannot reach the loop dtype the resolver picked,
/// exactly as numpy's `PyUFunc_ValidateCasting` does.
fn check_input_casting(ufunc: &str, from: DType, to: DType, i: usize) -> Result<()> {
    use crate::descr::Descr;
    if from == to || crate::casting::can_cast(Descr::native(from), Descr::native(to), UFUNC_CASTING)
    {
        return Ok(());
    }
    Err(ufunc_input_casting(
        ufunc,
        "same_kind",
        &Descr::native(from).str_code(),
        &Descr::native(to).str_code(),
        i,
        2,
    ))
}

/// What the two operands must be cast to, and what comes out.
#[derive(Copy, Clone, Debug)]
struct Loop {
    ain: DType,
    bin: DType,
    out: DType,
    kind: Kernel,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Kernel {
    /// `M + m -> M`, `m + m -> m`, `M - m -> M`, `M - M -> m`, `m - m -> m`
    AddSub,
    /// `m * q -> m` / `q * m -> m`
    MulInt,
    /// `m * d -> m` / `d * m -> m`
    MulFloat,
    /// `m / q -> m` (C truncating division; a zero divisor yields NaT)
    DivInt,
    /// `m / d -> m`
    DivFloat,
    /// `m / m -> d`
    DivTd,
    /// `m // m -> q`
    FloorDivTd,
    /// `m % m -> m`
    RemTd,
    /// comparison
    Cmp,
    /// minimum / maximum (NaT-propagating)
    MinMax,
    /// fmin / fmax (NaT-ignoring)
    FMinMax,
}

/// True when `op` is one this module owns.
pub fn handles(a: DType, b: DType) -> bool {
    a.is_datetime_like() || b.is_datetime_like()
}

/// Resolve the loop numpy would pick, or the exact error it would raise.
fn resolve(op: BinOp, a: &NdArray, b: &NdArray) -> Result<Loop> {
    use BinOp::*;
    let (da, db) = (a.dtype(), b.dtype());
    let int_like = |d: DType| d.is_integer() || d.is_bool();
    match op {
        Add | Sub => {
            let is_add = op == Add;
            if da.is_timedelta() {
                if db.is_timedelta() {
                    let p = dtm::promote_meta(da, db)?;
                    return Ok(Loop {
                        ain: p,
                        bin: p,
                        out: p,
                        kind: Kernel::AddSub,
                    });
                }
                if db.is_datetime() {
                    if !is_add {
                        return Err(reso_error(op, a, b));
                    }
                    let p = dtm::promote_meta(da, db)?;
                    let m = dtm::meta_of(p).unwrap();
                    return Ok(Loop {
                        ain: dtm::timedelta(m),
                        bin: p,
                        out: p,
                        kind: Kernel::AddSub,
                    });
                }
                if int_like(db) {
                    return Ok(Loop {
                        ain: da,
                        bin: da,
                        out: da,
                        kind: Kernel::AddSub,
                    });
                }
                return Err(reso_error(op, a, b));
            }
            if da.is_datetime() {
                if db.is_timedelta() {
                    let p = dtm::promote_meta(da, db)?;
                    let m = dtm::meta_of(p).unwrap();
                    return Ok(Loop {
                        ain: p,
                        bin: dtm::timedelta(m),
                        out: p,
                        kind: Kernel::AddSub,
                    });
                }
                if int_like(db) {
                    let m = dtm::meta_of(da).unwrap();
                    return Ok(Loop {
                        ain: da,
                        bin: dtm::timedelta(m),
                        out: da,
                        kind: Kernel::AddSub,
                    });
                }
                if db.is_datetime() && !is_add {
                    // M8[<A>] - M8[<B>] -> m8[gcd]
                    let p = dtm::promote_meta(da, db)?;
                    let m = dtm::meta_of(p).unwrap();
                    return Ok(Loop {
                        ain: p,
                        bin: p,
                        out: dtm::timedelta(m),
                        kind: Kernel::AddSub,
                    });
                }
                return Err(reso_error(op, a, b));
            }
            if int_like(da) {
                if db.is_timedelta() {
                    return Ok(Loop {
                        ain: db,
                        bin: db,
                        out: db,
                        kind: Kernel::AddSub,
                    });
                }
                if db.is_datetime() && is_add {
                    let m = dtm::meta_of(db).unwrap();
                    return Ok(Loop {
                        ain: dtm::timedelta(m),
                        bin: db,
                        out: db,
                        kind: Kernel::AddSub,
                    });
                }
            }
            Err(reso_error(op, a, b))
        }
        Mul => {
            if da.is_timedelta() {
                if int_like(db) {
                    return Ok(Loop {
                        ain: da,
                        bin: DType::I64,
                        out: da,
                        kind: Kernel::MulInt,
                    });
                }
                if db.is_float() {
                    return Ok(Loop {
                        ain: da,
                        bin: DType::F64,
                        out: da,
                        kind: Kernel::MulFloat,
                    });
                }
            } else if db.is_timedelta() {
                if int_like(da) {
                    return Ok(Loop {
                        ain: DType::I64,
                        bin: db,
                        out: db,
                        kind: Kernel::MulInt,
                    });
                }
                if da.is_float() {
                    return Ok(Loop {
                        ain: DType::F64,
                        bin: db,
                        out: db,
                        kind: Kernel::MulFloat,
                    });
                }
            }
            Err(reso_error(op, a, b))
        }
        Div | FloorDiv => {
            if da.is_timedelta() {
                if db.is_timedelta() {
                    let p = dtm::promote_meta(da, db)?;
                    return Ok(Loop {
                        ain: p,
                        bin: p,
                        out: if op == FloorDiv {
                            DType::I64
                        } else {
                            DType::F64
                        },
                        kind: if op == FloorDiv {
                            Kernel::FloorDivTd
                        } else {
                            Kernel::DivTd
                        },
                    });
                }
                // numpy's division resolver takes ISINTEGER only: bool is
                // rejected, unlike multiplication.
                if db.is_integer() {
                    return Ok(Loop {
                        ain: da,
                        bin: DType::I64,
                        out: da,
                        kind: Kernel::DivInt,
                    });
                }
                if db.is_float() {
                    return Ok(Loop {
                        ain: da,
                        bin: DType::F64,
                        out: da,
                        kind: Kernel::DivFloat,
                    });
                }
            }
            Err(reso_error(op, a, b))
        }
        Mod => {
            if da.is_timedelta() && db.is_timedelta() {
                let p = dtm::promote_meta(da, db)?;
                return Ok(Loop {
                    ain: p,
                    bin: p,
                    out: p,
                    kind: Kernel::RemTd,
                });
            }
            Err(reso_error(op, a, b))
        }
        Minimum | Maximum | Fmin | Fmax => {
            if da.is_datetime_like() && db.is_datetime_like() {
                if da.is_datetime() != db.is_datetime() {
                    return Err(reso_error(op, a, b));
                }
                let p = dtm::promote_meta(da, db)?;
                return Ok(Loop {
                    ain: p,
                    bin: p,
                    out: p,
                    kind: if matches!(op, Fmin | Fmax) {
                        Kernel::FMinMax
                    } else {
                        Kernel::MinMax
                    },
                });
            }
            if da.is_timedelta() && int_like(db) {
                return Ok(Loop {
                    ain: da,
                    bin: da,
                    out: da,
                    kind: if matches!(op, Fmin | Fmax) {
                        Kernel::FMinMax
                    } else {
                        Kernel::MinMax
                    },
                });
            }
            if db.is_timedelta() && int_like(da) {
                return Ok(Loop {
                    ain: db,
                    bin: db,
                    out: db,
                    kind: if matches!(op, Fmin | Fmax) {
                        Kernel::FMinMax
                    } else {
                        Kernel::MinMax
                    },
                });
            }
            Err(no_loop(op, da, db))
        }
        Eq | Ne | Lt | Le | Gt | Ge => {
            if da.is_datetime_like() && db.is_datetime_like() {
                if da.is_datetime() != db.is_datetime() {
                    return Err(reso_error(op, a, b));
                }
                let p = dtm::promote_meta(da, db)?;
                return Ok(Loop {
                    ain: p,
                    bin: p,
                    out: DType::Bool,
                    kind: Kernel::Cmp,
                });
            }
            // Integers compare against timedelta (they cast safely into it).
            if da.is_timedelta() && int_like(db) {
                return Ok(Loop {
                    ain: da,
                    bin: da,
                    out: DType::Bool,
                    kind: Kernel::Cmp,
                });
            }
            if db.is_timedelta() && int_like(da) {
                return Ok(Loop {
                    ain: db,
                    bin: db,
                    out: DType::Bool,
                    kind: Kernel::Cmp,
                });
            }
            Err(no_loop(op, da, db))
        }
        _ => Err(no_loop(op, da, db)),
    }
}

/// The datetime half of [`crate::ops::binary_multi`].
pub fn binary(a: &NdArray, b: &NdArray, op: BinOp) -> Result<(NdArray, Option<NdArray>)> {
    let lp = match resolve(op, a, b) {
        Ok(l) => l,
        // `==` / `!=` fall back to an all-False / all-True result when no
        // *loop* fits (numpy's ufunc returns NotImplemented and
        // `ndarray.__eq__` substitutes the constant). A metadata failure is
        // different: `m8[Y] == m8[fs]` raises the TypeError and
        // `m8[as] == m8[s]` the OverflowError, because those happen *while*
        // promoting rather than instead of it.
        Err(e) => {
            let no_loop = matches!(
                e,
                Error::UFuncBinaryResolution { .. } | Error::UFuncNoLoop { .. }
            );
            if no_loop && matches!(op, BinOp::Eq | BinOp::Ne) {
                let shape = broadcast_shapes(&a.shape, &b.shape)?;
                let out = NdArray::full(
                    shape,
                    DType::Bool,
                    Scalar::Bool(op == BinOp::Ne),
                )?;
                return Ok((out, None));
            }
            return Err(e);
        }
    };

    // The resolver only names the loop; numpy's ufunc machinery then insists
    // that every input reach that loop's dtype under the call's casting rule
    // (`same_kind` by default). `M8[D] + m8[Y]` promotes to `M8[D]`, so input 1
    // would have to become `m8[D]` -- which m8's nonlinear units forbid -- and
    // numpy reports `_UFuncInputCastingError` rather than a resolution failure.
    check_input_casting(op.name(), a.dtype(), lp.ain, 0)?;
    check_input_casting(op.name(), b.dtype(), lp.bin, 1)?;

    let av = a.try_astype(lp.ain)?;
    let bv = b.try_astype(lp.bin)?;
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let av = broadcast_to(&av, &out_shape)?;
    let bv = broadcast_to(&bv, &out_shape)?;
    let out = NdArray::empty(out_shape.clone(), lp.out)?;

    let ao: Vec<isize> = offsets(&av.shape, &av.strides, av.byte_offset).collect();
    let bo: Vec<isize> = offsets(&bv.shape, &bv.strides, bv.byte_offset).collect();
    let oo: Vec<isize> = offsets(&out.shape, &out.strides, out.byte_offset).collect();

    match lp.kind {
        Kernel::AddSub => {
            let sub = op == BinOp::Sub;
            let what = overflow_label(op, a.dtype(), b.dtype());
            for k in 0..oo.len() {
                let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
                let v = if x == NAT || y == NAT {
                    NAT
                } else {
                    let r = if sub {
                        x.checked_sub(y)
                    } else {
                        x.checked_add(y)
                    };
                    match r {
                        Some(r) if r != NAT => r,
                        _ => return Err(Error::OverflowError(what)),
                    }
                };
                out.write_at(oo[k], Scalar::Int(v));
            }
        }
        Kernel::MulInt => {
            let what = "Overflow in timedelta64 * int64 multiplication".to_string();
            for k in 0..oo.len() {
                let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
                let (td, n) = if av.dtype().is_timedelta() {
                    (x, y)
                } else {
                    (y, x)
                };
                let v = if td == NAT {
                    NAT
                } else {
                    match td.checked_mul(n) {
                        Some(r) if r != NAT => r,
                        _ => return Err(Error::OverflowError(what)),
                    }
                };
                out.write_at(oo[k], Scalar::Int(v));
            }
        }
        Kernel::MulFloat => {
            for k in 0..oo.len() {
                let (td, f) = if av.dtype().is_timedelta() {
                    (int_at(&av, ao[k]), float_at(&bv, bo[k]))
                } else {
                    (int_at(&bv, bo[k]), float_at(&av, ao[k]))
                };
                let v = if td == NAT {
                    NAT
                } else {
                    let r = td as f64 * f;
                    if r.is_finite() {
                        crate::element::f2i64(r)
                    } else {
                        NAT
                    }
                };
                out.write_at(oo[k], Scalar::Int(v));
            }
        }
        Kernel::DivInt => {
            for k in 0..oo.len() {
                let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
                let v = if x == NAT || y == 0 {
                    if y == 0 && x != NAT {
                        crate::fpe::raise(crate::fpe::DIVIDE);
                    }
                    NAT
                } else {
                    x.wrapping_div(y)
                };
                out.write_at(oo[k], Scalar::Int(v));
            }
        }
        Kernel::DivFloat => {
            for k in 0..oo.len() {
                let (x, f) = (int_at(&av, ao[k]), float_at(&bv, bo[k]));
                let v = if x == NAT {
                    NAT
                } else {
                    let r = x as f64 / f;
                    if r.is_finite() {
                        crate::element::f2i64(r)
                    } else {
                        NAT
                    }
                };
                out.write_at(oo[k], Scalar::Int(v));
            }
        }
        Kernel::DivTd => {
            for k in 0..oo.len() {
                let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
                let v = if x == NAT || y == NAT {
                    f64::NAN
                } else {
                    x as f64 / y as f64
                };
                out.write_at(oo[k], Scalar::Float(v));
            }
        }
        Kernel::FloorDivTd => {
            for k in 0..oo.len() {
                let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
                let v = if x == NAT || y == NAT {
                    crate::fpe::raise(crate::fpe::INVALID);
                    0
                } else if y == 0 {
                    crate::fpe::raise(crate::fpe::DIVIDE);
                    0
                } else {
                    let mut q = x.wrapping_div(y);
                    if ((x > 0) != (y > 0)) && q.wrapping_mul(y) != x {
                        q -= 1;
                    }
                    q
                };
                out.write_at(oo[k], Scalar::Int(v));
            }
        }
        Kernel::RemTd => {
            for k in 0..oo.len() {
                let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
                let v = if x == NAT || y == NAT {
                    NAT
                } else if y == 0 {
                    crate::fpe::raise(crate::fpe::DIVIDE);
                    NAT
                } else {
                    let rem = x.wrapping_rem(y);
                    if (x > 0) == (y > 0) || rem == 0 {
                        rem
                    } else {
                        rem + y
                    }
                };
                out.write_at(oo[k], Scalar::Int(v));
            }
        }
        Kernel::Cmp => {
            for k in 0..oo.len() {
                let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
                let nat = x == NAT || y == NAT;
                let r = match op {
                    BinOp::Ne => x != y || nat,
                    BinOp::Eq => x == y && !nat,
                    BinOp::Lt => x < y && !nat,
                    BinOp::Le => x <= y && !nat,
                    BinOp::Gt => x > y && !nat,
                    _ => x >= y && !nat,
                };
                out.write_at(oo[k], Scalar::Bool(r));
            }
        }
        Kernel::MinMax => {
            let is_max = matches!(op, BinOp::Maximum);
            for k in 0..oo.len() {
                let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
                let v = if x == NAT {
                    x
                } else if y == NAT {
                    y
                } else if is_max {
                    if x > y {
                        x
                    } else {
                        y
                    }
                } else if x < y {
                    x
                } else {
                    y
                };
                out.write_at(oo[k], Scalar::Int(v));
            }
        }
        Kernel::FMinMax => {
            let is_max = matches!(op, BinOp::Fmax);
            for k in 0..oo.len() {
                let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
                let v = if x == NAT {
                    y
                } else if y == NAT {
                    x
                } else if is_max {
                    if x >= y {
                        x
                    } else {
                        y
                    }
                } else if x <= y {
                    x
                } else {
                    y
                };
                out.write_at(oo[k], Scalar::Int(v));
            }
        }
    }
    Ok((out, None))
}

/// `np.divmod` on two timedelta64 operands: numpy's `TIMEDELTA_mm_qm_divmod`.
pub fn divmod(a: &NdArray, b: &NdArray) -> Result<(NdArray, NdArray)> {
    let lp = resolve(BinOp::Mod, a, b)?;
    check_input_casting("divmod", a.dtype(), lp.ain, 0)?;
    check_input_casting("divmod", b.dtype(), lp.bin, 1)?;
    let av = a.try_astype(lp.ain)?;
    let bv = b.try_astype(lp.bin)?;
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let av = broadcast_to(&av, &out_shape)?;
    let bv = broadcast_to(&bv, &out_shape)?;
    let q = NdArray::empty(out_shape.clone(), DType::I64)?;
    let r = NdArray::empty(out_shape, lp.out)?;
    let ao: Vec<isize> = offsets(&av.shape, &av.strides, av.byte_offset).collect();
    let bo: Vec<isize> = offsets(&bv.shape, &bv.strides, bv.byte_offset).collect();
    let qo: Vec<isize> = offsets(&q.shape, &q.strides, q.byte_offset).collect();
    for k in 0..qo.len() {
        let (x, y) = (int_at(&av, ao[k]), int_at(&bv, bo[k]));
        let (quo, rem) = if x == NAT || y == NAT {
            crate::fpe::raise(crate::fpe::INVALID);
            (0, NAT)
        } else if y == 0 {
            crate::fpe::raise(crate::fpe::DIVIDE);
            (0, NAT)
        } else {
            let quo = x.wrapping_div(y);
            let rem = x.wrapping_rem(y);
            if (x > 0) == (y > 0) || rem == 0 {
                (quo, rem)
            } else {
                (quo - 1, rem + y)
            }
        };
        q.write_at(qo[k], Scalar::Int(quo));
        r.write_at(qo[k], Scalar::Int(rem));
    }
    Ok((q, r))
}

fn overflow_label(op: BinOp, a: DType, b: DType) -> String {
    let name = |d: DType| {
        if d.is_datetime() {
            "datetime64"
        } else if d.is_timedelta() {
            "timedelta64"
        } else {
            "timedelta64"
        }
    };
    let sym = if op == BinOp::Sub {
        ("-", "subtraction")
    } else {
        ("+", "addition")
    };
    format!(
        "Overflow in {} {} {} {}",
        name(a),
        sym.0,
        name(b),
        sym.1
    )
}

#[inline]
fn int_at(a: &NdArray, off: isize) -> i64 {
    match a.read_at(off) {
        Scalar::Int(i) => i,
        Scalar::Uint(u) => u as i64,
        Scalar::Bool(b) => b as i64,
        Scalar::Float(f) => f as i64,
        Scalar::Complex(c) => c.re as i64,
    }
}

#[inline]
fn float_at(a: &NdArray, off: isize) -> f64 {
    a.read_at(off).as_f64()
}

// ---------------------------------------------------------------------------
// Unary loops
// ---------------------------------------------------------------------------

/// `np.isnat`: true exactly on the NaT sentinel.
pub fn isnat(a: &NdArray) -> Result<NdArray> {
    if !a.dtype().is_datetime_like() {
        return Err(Error::TypeError(
            "ufunc 'isnat' is only defined for np.datetime64 and np.timedelta64.".into(),
        ));
    }
    let n = a.to_native();
    let out = NdArray::empty(n.shape.clone(), DType::Bool)?;
    let oo: Vec<isize> = offsets(&out.shape, &out.strides, out.byte_offset).collect();
    for (k, off) in offsets(&n.shape, &n.strides, n.byte_offset).enumerate() {
        out.write_at(oo[k], Scalar::Bool(int_at(&n, off) == NAT));
    }
    Ok(out)
}

/// Unary ufunc loops shared by datetime64 and timedelta64.
pub fn unary(a: &NdArray, name: &str) -> Result<NdArray> {
    let dt = a.dtype();
    let out_dt = match name {
        "negative" | "positive" | "absolute" | "fabs" => {
            if !dt.is_timedelta() {
                let f = family(dt);
                return Err(ufunc_no_loop(name, &[&f]));
            }
            dt
        }
        "sign" => {
            if !dt.is_timedelta() {
                let f = family(dt);
                return Err(ufunc_no_loop(name, &[&f]));
            }
            DType::F64
        }
        "isfinite" | "isinf" | "isnan" | "isnat" => DType::Bool,
        _ => {
            let f = family(dt);
            return Err(ufunc_no_loop(name, &[&f]));
        }
    };
    let n = a.to_native();
    let out = NdArray::empty(n.shape.clone(), out_dt)?;
    let oo: Vec<isize> = offsets(&out.shape, &out.strides, out.byte_offset).collect();
    for (k, off) in offsets(&n.shape, &n.strides, n.byte_offset).enumerate() {
        let x = int_at(&n, off);
        let v = match name {
            "negative" => Scalar::Int(if x == NAT { NAT } else { x.wrapping_neg() }),
            "positive" => Scalar::Int(x),
            "absolute" | "fabs" => Scalar::Int(if x == NAT {
                NAT
            } else if x >= 0 {
                x
            } else {
                x.wrapping_neg()
            }),
            "sign" => Scalar::Float(if x == NAT {
                f64::NAN
            } else if x > 0 {
                1.0
            } else if x < 0 {
                -1.0
            } else {
                0.0
            }),
            "isfinite" => Scalar::Bool(x != NAT),
            "isinf" => Scalar::Bool(false),
            "isnan" => Scalar::Bool(x == NAT),
            _ => Scalar::Bool(x == NAT),
        };
        out.write_at(oo[k], v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datetime::{DtMeta, UNIT_D, UNIT_S};

    fn td(vals: &[i64], base: u8) -> NdArray {
        let a = NdArray::empty(vec![vals.len() as isize], dtm::timedelta(DtMeta::unit(base)))
            .unwrap();
        for (i, &v) in vals.iter().enumerate() {
            a.write_at((i * 8) as isize, Scalar::Int(v));
        }
        a
    }

    #[test]
    fn datetime_predicates_classify_only_nat_as_nan() {
        let a = td(&[0, NAT, -1], UNIT_D);
        let finite = unary(&a, "isfinite").unwrap();
        let infinite = unary(&a, "isinf").unwrap();
        let nan = unary(&a, "isnan").unwrap();
        assert_eq!(finite.get_flat(0), Scalar::Bool(true));
        assert_eq!(finite.get_flat(1), Scalar::Bool(false));
        assert_eq!(infinite.get_flat(0), Scalar::Bool(false));
        assert_eq!(infinite.get_flat(1), Scalar::Bool(false));
        assert_eq!(nan.get_flat(0), Scalar::Bool(false));
        assert_eq!(nan.get_flat(1), Scalar::Bool(true));
        assert_eq!(nan.get_flat(2), Scalar::Bool(false));
    }
    fn dt(vals: &[i64], base: u8) -> NdArray {
        let a =
            NdArray::empty(vec![vals.len() as isize], dtm::datetime(DtMeta::unit(base))).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            a.write_at((i * 8) as isize, Scalar::Int(v));
        }
        a
    }
    fn ints(a: &NdArray) -> Vec<i64> {
        (0..a.size()).map(|i| int_at(a, (i * 8) as isize)).collect()
    }

    #[test]
    fn nat_propagates_through_arithmetic() {
        let a = td(&[1, NAT, 3], UNIT_S);
        let b = td(&[1, 1, NAT], UNIT_S);
        let (r, _) = binary(&a, &b, BinOp::Add).unwrap();
        assert_eq!(ints(&r), vec![2, NAT, NAT]);
        let (r, _) = binary(&a, &b, BinOp::Sub).unwrap();
        assert_eq!(ints(&r), vec![0, NAT, NAT]);
        let (r, _) = binary(&a, &b, BinOp::Mod).unwrap();
        assert_eq!(ints(&r), vec![0, NAT, NAT]);
        // Comparisons with NaT are all False except `!=`.
        for (op, want) in [
            (BinOp::Eq, [true, false, false]),
            (BinOp::Ne, [false, true, true]),
            (BinOp::Lt, [false, false, false]),
            (BinOp::Ge, [true, false, false]),
        ] {
            let (r, _) = binary(&a, &b, op).unwrap();
            let got: Vec<bool> = (0..3)
                .map(|i| matches!(r.read_at(i as isize), Scalar::Bool(true)))
                .collect();
            assert_eq!(got, want.to_vec(), "{op:?}");
        }
    }

    #[test]
    fn datetime_minus_datetime_uses_the_promoted_unit() {
        let a = dt(&[1], UNIT_D);
        let b = dt(&[0], UNIT_S);
        let (r, _) = binary(&a, &b, BinOp::Sub).unwrap();
        assert_eq!(r.dtype(), dtm::timedelta(DtMeta::unit(UNIT_S)));
        assert_eq!(ints(&r), vec![86400]);
    }

    #[test]
    fn datetime_plus_datetime_is_a_type_error() {
        let a = dt(&[1], UNIT_D);
        assert!(binary(&a, &a, BinOp::Add).is_err());
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn float_timedelta_overflow_becomes_nat_on_manylinux() {
        let a = td(&[1i64 << 62], UNIT_S);
        let f = NdArray::from_scalars(&[Scalar::Float(2.5)], DType::F64).unwrap();
        let (r, _) = binary(&a, &f, BinOp::Mul).unwrap();
        assert_eq!(ints(&r), vec![NAT]);
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

use crate::reduce::ReduceOp;

/// The dtype a reduction produces, or the error numpy raises. `sum` on
/// datetime64 fails because there is no `MM->M` add loop; `prod` fails for
/// both because there is no datetime multiply loop at all.
fn reduce_out_dtype(arr: &NdArray, op: ReduceOp) -> Result<DType> {
    let d = arr.dtype();
    let s = arr.descr.str_code();
    match op {
        ReduceOp::ArgMin | ReduceOp::ArgMax => Ok(DType::I64),
        ReduceOp::Min | ReduceOp::Max => Ok(d),
        ReduceOp::Sum => {
            if d.is_datetime() {
                Err(ufunc_binary_resolution("add", &s, &s))
            } else {
                Ok(d)
            }
        }
        ReduceOp::Prod => Err(ufunc_binary_resolution("multiply", &s, &s)),
    }
}

/// Fold one lane of int64 datetime values.
fn fold(vals: impl Iterator<Item = i64>, op: ReduceOp, seed: Option<i64>) -> Result<(i64, i64)> {
    let mut acc: Option<i64> = seed;
    let mut best_idx: i64 = 0;
    let mut nat_idx: Option<i64> = None;
    for (i, v) in vals.enumerate() {
        let i = i as i64;
        if matches!(op, ReduceOp::ArgMin | ReduceOp::ArgMax) && v == NAT && nat_idx.is_none() {
            // numpy's datetime argmin/argmax stop at the first NaT, exactly
            // as the float loops stop at the first NaN.
            nat_idx = Some(i);
        }
        acc = Some(match acc {
            None => {
                best_idx = i;
                v
            }
            Some(a) => match op {
                ReduceOp::Sum => {
                    if a == NAT || v == NAT {
                        NAT
                    } else {
                        match a.checked_add(v) {
                            Some(r) if r != NAT => r,
                            _ => {
                                return Err(Error::OverflowError(
                                    "Overflow in timedelta64 + timedelta64 addition".into(),
                                ))
                            }
                        }
                    }
                }
                ReduceOp::Min => {
                    if a == NAT || v == NAT {
                        NAT
                    } else if v < a {
                        v
                    } else {
                        a
                    }
                }
                ReduceOp::Max => {
                    if a == NAT || v == NAT {
                        NAT
                    } else if v > a {
                        v
                    } else {
                        a
                    }
                }
                ReduceOp::ArgMin => {
                    if v < a {
                        best_idx = i;
                        v
                    } else {
                        a
                    }
                }
                ReduceOp::ArgMax => {
                    if v > a {
                        best_idx = i;
                        v
                    } else {
                        a
                    }
                }
                ReduceOp::Prod => unreachable!(),
            },
        });
    }
    if let Some(n) = nat_idx {
        best_idx = n;
    }
    Ok((acc.unwrap_or(0), best_idx))
}

/// The datetime half of [`crate::reduce::reduce_all_with`].
pub fn reduce_all(arr: &NdArray, op: ReduceOp, seed: Option<Scalar>) -> Result<Scalar> {
    let out_dt = reduce_out_dtype(arr, op)?;
    let n = arr.to_native();
    if n.size() == 0 && !matches!(op, ReduceOp::Sum) {
        return Err(Error::ValueError(format!(
            "zero-size array to reduction operation {} which has no identity",
            op.name()
        )));
    }
    let seed = seed.map(|s| match s {
        Scalar::Int(i) => i,
        other => other.as_f64() as i64,
    });
    let vals = offsets(&n.shape, &n.strides, n.byte_offset).map(|o| int_at(&n, o));
    let (v, idx) = fold(vals, op, seed)?;
    Ok(if out_dt == DType::I64 && matches!(op, ReduceOp::ArgMin | ReduceOp::ArgMax) {
        Scalar::Int(idx)
    } else {
        Scalar::Int(v)
    })
}

/// The datetime half of [`crate::reduce::reduce_axis_with`].
pub fn reduce_axis(
    arr: &NdArray,
    axis: usize,
    op: ReduceOp,
    keepdims: bool,
    seed: Option<Scalar>,
) -> Result<NdArray> {
    let out_dt = reduce_out_dtype(arr, op)?;
    let n = arr.to_native();
    if axis >= n.ndim() {
        return Err(Error::AxisError(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis,
            n.ndim()
        )));
    }
    let len = n.shape[axis].max(0) as usize;
    if len == 0 && !matches!(op, ReduceOp::Sum) {
        return Err(Error::ValueError(format!(
            "zero-size array to reduction operation {} which has no identity",
            op.name()
        )));
    }
    let step = n.strides[axis];
    let mut oshape = n.shape.clone();
    if keepdims {
        oshape[axis] = 1;
    } else {
        oshape.remove(axis);
    }
    let out = NdArray::empty(oshape, out_dt)?;
    let mut lane_shape = n.shape.clone();
    let mut lane_strides = n.strides.clone();
    lane_shape.remove(axis);
    lane_strides.remove(axis);
    let seed = seed.map(|s| match s {
        Scalar::Int(i) => i,
        other => other.as_f64() as i64,
    });
    let oo: Vec<isize> = offsets(&out.shape, &out.strides, out.byte_offset).collect();
    for (k, base) in offsets(&lane_shape, &lane_strides, n.byte_offset).enumerate() {
        let vals = (0..len as isize).map(|j| int_at(&n, base + j * step));
        let (v, idx) = fold(vals, op, seed)?;
        out.write_at(
            oo[k],
            Scalar::Int(if matches!(op, ReduceOp::ArgMin | ReduceOp::ArgMax) {
                idx
            } else {
                v
            }),
        );
    }
    Ok(out)
}
