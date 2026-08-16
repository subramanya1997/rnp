//! The Python-facing ufunc machinery: `__call__`, `.reduce`, `.accumulate`,
//! and the scalar-type bridge.
//!
//! The `ufunc` *object* lives in the shim (`rnp_numpy/_ufunc.py`) because it
//! needs numpy's exact attribute surface; everything it does numerically ends
//! up in one of the entry points here.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use rnp_core::element::Scalar;
use rnp_core::{fpe, BinOp, DType, NdArray, UnOp};

use crate::convert::{array_from_any, np_scalar, scalar_from_py, weak_dtype, weak_promote};
use crate::pyarray::{store_or_wrap, PyNdArray};
use crate::pydtype::{dtype_from_any, PyDType};

use std::sync::OnceLock;

static FPE_REPORTER: OnceLock<Py<PyAny>> = OnceLock::new();

/// Install the shim's `_errstate` reporter, which turns the engine's flags
/// into numpy's RuntimeWarnings (or FloatingPointErrors) under the current
/// `np.seterr` state.
#[pyfunction]
pub fn _register_fpe_reporter(f: Py<PyAny>) {
    let _ = FPE_REPORTER.set(f);
}

/// Drain the engine's FP flags and hand them to the shim.
pub fn report_fpe(py: Python<'_>, where_: &str) -> PyResult<()> {
    let flags = fpe::take();
    if flags == 0 {
        return Ok(());
    }
    if let Some(f) = FPE_REPORTER.get() {
        f.bind(py).call1((flags, where_))?;
    }
    Ok(())
}

/// What a ufunc name resolves to.
#[derive(Copy, Clone)]
pub enum Ufn {
    Un(UnOp),
    Bin(BinOp),
    DivMod,
    Frexp,
    Modf,
}

pub fn lookup(name: &str) -> Option<Ufn> {
    match name {
        "divmod" => Some(Ufn::DivMod),
        "frexp" => Some(Ufn::Frexp),
        "modf" => Some(Ufn::Modf),
        _ => {
            if let Some(b) = BinOp::from_name(name) {
                return Some(Ufn::Bin(b));
            }
            UnOp::from_name(name).map(Ufn::Un)
        }
    }
}

/// One coerced input: an array plus whether it was a *strong* operand
/// (an array or a numpy scalar) under NEP 50.
struct Operand {
    arr: Option<NdArray>,
    weak: Option<Scalar>,
}

fn coerce(obj: &Bound<'_, PyAny>) -> PyResult<Operand> {
    if let Ok(a) = obj.cast::<PyNdArray>() {
        return Ok(Operand {
            arr: Some(a.borrow().arr.clone()),
            weak: None,
        });
    }
    if let Some((dt, v)) = np_scalar(obj)? {
        let mut a = NdArray::zeros(vec![], dt).map_err(crate::err)?;
        a.set(&[], v).map_err(crate::err)?;
        return Ok(Operand {
            arr: Some(a),
            weak: None,
        });
    }
    // A bare Python number is weak: it takes the other operand's dtype.
    if let Some(s) = scalar_from_py(obj) {
        return Ok(Operand {
            arr: None,
            weak: Some(s),
        });
    }
    Ok(Operand {
        arr: Some(array_from_any(obj, None, false)?),
        weak: None,
    })
}

/// Resolve NEP 50 weak scalars against the strong operands and hand back
/// concrete arrays.
/// The dtype a *weak* Python scalar in slot `i` must resolve against, for the
/// ufuncs whose loop table is not uniform across the operands.
///
/// numpy resolves a weak scalar against the slot of the loop it lands in, not
/// against the promotion of every operand: `np.ldexp(np.float64(0.5), 2)`
/// picks the `di->d` loop, so the exponent becomes `int32` -- not float64,
/// which would leave no matching loop at all.
fn weak_slot_dtype(f: Ufn, i: usize) -> Option<DType> {
    match (f, i) {
        (Ufn::Bin(rnp_core::BinOp::Ldexp), 1) => Some(DType::I32),
        _ => None,
    }
}

fn resolve_operands(
    objs: &[Bound<'_, PyAny>],
    comparison: bool,
    f: Ufn,
) -> PyResult<Vec<NdArray>> {
    let ops: Vec<Operand> = objs.iter().map(coerce).collect::<PyResult<_>>()?;
    let mut strong: Option<DType> = None;
    for o in &ops {
        if let Some(a) = &o.arr {
            strong = Some(match strong {
                None => a.dtype(),
                Some(d) => rnp_core::promote(d, a.dtype()),
            });
        }
    }
    let mut out = Vec::with_capacity(ops.len());
    for (i, o) in ops.into_iter().enumerate() {
        match (o.arr, o.weak) {
            (Some(a), _) => out.push(a),
            (None, Some(s)) => {
                let slot = weak_slot_dtype(f, i);
                let d = match slot.or(strong) {
                    Some(d) => weak_dtype(d, s, comparison)?,
                    None => s.natural_dtype(),
                };
                let mut a = NdArray::zeros(vec![], d).map_err(crate::err)?;
                a.set(&[], s).map_err(crate::err)?;
                out.push(a);
            }
            _ => unreachable!(),
        }
    }
    Ok(out)
}

/// Overlay `res` onto `base` wherever `mask` is true (numpy's `where=`).
fn apply_where(res: &NdArray, base: &NdArray, mask: &NdArray) -> PyResult<NdArray> {
    let m = rnp_core::iter::broadcast_to(mask, &res.shape).map_err(crate::err)?;
    let b = rnp_core::iter::broadcast_to(base, &res.shape).map_err(crate::err)?;
    let out = NdArray::empty(res.shape.clone(), res.dtype()).map_err(crate::err)?;
    let ro: Vec<isize> =
        rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset).collect();
    let mo: Vec<isize> = rnp_core::iter::offsets(&m.shape, &m.strides, m.byte_offset).collect();
    let bo: Vec<isize> = rnp_core::iter::offsets(&b.shape, &b.strides, b.byte_offset).collect();
    let oo: Vec<isize> =
        rnp_core::iter::offsets(&out.shape, &out.strides, out.byte_offset).collect();
    for k in 0..oo.len() {
        let take = matches!(m.read_at(mo[k]), Scalar::Bool(true) | Scalar::Int(1))
            || !matches!(m.read_at(mo[k]), Scalar::Bool(false) | Scalar::Int(0));
        out.write_at(oo[k], if take { res.read_at(ro[k]) } else { b.read_at(bo[k]) });
    }
    Ok(out)
}

/// `ufunc(*args, out=, where=, dtype=, casting=)`.
#[pyfunction]
#[pyo3(signature = (name, args, out = None, where_ = None, casting = None, dtype = None))]
pub fn _ufunc_call<'py>(
    py: Python<'py>,
    name: &str,
    args: &Bound<'py, PyTuple>,
    out: Option<&Bound<'py, PyAny>>,
    where_: Option<&Bound<'py, PyAny>>,
    casting: Option<&str>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = casting;
    let f = lookup(name).ok_or_else(|| {
        PyNotImplementedErrorShim::new(format!("ufunc '{name}' is not implemented by rnp yet"))
    })?;
    let objs: Vec<Bound<'py, PyAny>> = args.iter().collect();
    let mut inputs =
        resolve_operands(&objs, matches!(f, Ufn::Bin(b) if b.is_comparison()), f)?;
    if let Some(d) = dtype {
        if !d.is_none() {
            let dt = dtype_from_any(d)?;
            inputs = inputs.iter().map(|a| a.astype(dt)).collect();
        }
    }

    let (res, res2) = match f {
        Ufn::Un(op) => {
            if inputs.len() != 1 {
                return Err(PyTypeError::new_err(format!(
                    "invalid number of arguments to ufunc '{name}'"
                )));
            }
            (rnp_core::unary(&inputs[0], op).map_err(crate::err)?, None)
        }
        Ufn::Bin(op) => {
            if inputs.len() != 2 {
                return Err(PyTypeError::new_err(format!(
                    "invalid number of arguments to ufunc '{name}'"
                )));
            }
            (
                rnp_core::binary(&inputs[0], &inputs[1], op).map_err(crate::err)?,
                None,
            )
        }
        Ufn::DivMod => {
            let (q, r) = rnp_core::divmod(&inputs[0], &inputs[1]).map_err(crate::err)?;
            (q, Some(r))
        }
        Ufn::Frexp | Ufn::Modf => {
            let (a, b) = split_pair(&inputs[0], matches!(f, Ufn::Frexp))?;
            (a, Some(b))
        }
    };

    // `where=` keeps the destination's existing values outside the mask.
    let res = match where_ {
        Some(w) if !w.is_none() && !matches!(w.extract::<bool>(), Ok(true)) => {
            let mask = array_from_any(w, Some(DType::Bool), false)?;
            let base = match out {
                Some(o) if !o.is_none() => o
                    .cast::<PyNdArray>()
                    .map_err(|_| PyTypeError::new_err("return arrays must be of ArrayType"))?
                    .borrow()
                    .arr
                    .clone(),
                _ => NdArray::zeros(res.shape.clone(), res.dtype()).map_err(crate::err)?,
            };
            apply_where(&res, &base, &mask)?
        }
        _ => res,
    };

    if let Some(second) = res2 {
        let a = store_or_wrap(py, res, None)?;
        let b = store_or_wrap(py, second, None)?;
        return Ok(PyTuple::new(py, [a, b])?.into_any());
    }
    store_or_wrap(py, res, out)
}

/// `frexp` / `modf`: the two-output float decompositions.
fn split_pair(a: &NdArray, frexp: bool) -> PyResult<(NdArray, NdArray)> {
    let dt = match a.dtype() {
        DType::F16 | DType::F32 | DType::F64 => a.dtype(),
        d if d.is_complex() => {
            return Err(PyTypeError::new_err(
                "ufunc not supported for complex inputs",
            ))
        }
        // As the other float-only ufuncs: the smallest loop that fits.
        d => rnp_core::promote(d, DType::F16),
    };
    let src = a.astype(dt);
    let first = NdArray::empty(src.shape.clone(), dt).map_err(crate::err)?;
    let second =
        NdArray::empty(src.shape.clone(), if frexp { DType::I32 } else { dt }).map_err(crate::err)?;
    let offs: Vec<isize> =
        rnp_core::iter::offsets(&src.shape, &src.strides, src.byte_offset).collect();
    let fo: Vec<isize> =
        rnp_core::iter::offsets(&first.shape, &first.strides, first.byte_offset).collect();
    let so: Vec<isize> =
        rnp_core::iter::offsets(&second.shape, &second.strides, second.byte_offset).collect();
    for k in 0..offs.len() {
        let x = match src.read_at(offs[k]) {
            Scalar::Float(f) => f,
            _ => 0.0,
        };
        if frexp {
            let (m, e) = frexp_f64(x);
            first.write_at(fo[k], Scalar::Float(m));
            second.write_at(so[k], Scalar::Int(e as i64));
        } else {
            let ip = x.trunc();
            // `modf(-0.0)` is `(-0.0, -0.0)`: the fractional part keeps the
            // sign even when it is zero.
            let fp = if x.is_infinite() {
                0.0_f64.copysign(x)
            } else {
                (x - ip).copysign(x)
            };
            first.write_at(fo[k], Scalar::Float(fp));
            second.write_at(so[k], Scalar::Float(ip));
        }
    }
    Ok((first, second))
}

fn frexp_f64(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let mut exp = ((bits >> 52) & 0x7FF) as i32;
    if exp == 0 {
        // Subnormal: scale up first.
        let (m, e) = frexp_f64(x * 2.0_f64.powi(64));
        return (m, e - 64);
    }
    exp -= 1022;
    let mantissa = f64::from_bits((bits & !(0x7FFu64 << 52)) | (1022u64 << 52));
    (mantissa, exp)
}

/// A `NotImplementedError` builder that keeps the call sites readable.
struct PyNotImplementedErrorShim;
impl PyNotImplementedErrorShim {
    fn new(msg: String) -> PyErr {
        pyo3::exceptions::PyNotImplementedError::new_err(msg)
    }
}

// ---------------------------------------------------------------------------
// reduce / accumulate
// ---------------------------------------------------------------------------

/// numpy's identity element for the ops that have one.
fn identity_of(op: BinOp, dt: DType) -> Option<Scalar> {
    Some(match op {
        BinOp::Add | BinOp::BitOr | BinOp::BitXor => {
            if dt.is_float() {
                Scalar::Float(0.0)
            } else if dt.is_bool() {
                Scalar::Bool(false)
            } else {
                Scalar::Int(0)
            }
        }
        BinOp::Mul => {
            if dt.is_float() {
                Scalar::Float(1.0)
            } else if dt.is_bool() {
                Scalar::Bool(true)
            } else {
                Scalar::Int(1)
            }
        }
        BinOp::BitAnd => {
            if dt.is_bool() {
                Scalar::Bool(true)
            } else {
                Scalar::Int(-1)
            }
        }
        BinOp::LogicalAnd => Scalar::Bool(true),
        BinOp::LogicalOr | BinOp::LogicalXor => Scalar::Bool(false),
        BinOp::Hypot => Scalar::Float(0.0),
        BinOp::Logaddexp | BinOp::Logaddexp2 => Scalar::Float(f64::NEG_INFINITY),
        BinOp::Gcd => Scalar::Int(0),
        _ => return None,
    })
}

/// Fold `op` along `axis`. The heavy-hitters (`add`, `multiply`, `minimum`,
/// `maximum`, and the logical pair) are routed to the native reductions in
/// `rnp_core::reduce`, which reproduce numpy's pairwise summation exactly;
/// everything else folds slice by slice.
fn reduce_along(a: &NdArray, axis: usize, op: BinOp, initial: Option<Scalar>) -> PyResult<NdArray> {
    let n = a.shape[axis];
    if n == 0 {
        let ident = initial.or_else(|| identity_of(op, a.dtype())).ok_or_else(|| {
            PyValueError::new_err(format!(
                "zero-size array to reduction operation {} which has no identity",
                op.name()
            ))
        })?;
        let mut shape = a.shape.clone();
        shape.remove(axis);
        let (_, out_dt) = rnp_core::ops::result_dtypes(a.dtype(), a.dtype(), op).map_err(crate::err)?;
        let mut z = NdArray::zeros(shape, out_dt).map_err(crate::err)?;
        z.fill(ident.cast(out_dt));
        return Ok(z);
    }
    let mut acc = match &initial {
        Some(s) => {
            let mut shape = a.shape.clone();
            shape.remove(axis);
            let mut z = NdArray::zeros(shape, a.dtype()).map_err(crate::err)?;
            z.fill(s.cast(a.dtype()));
            z
        }
        None => a.slice_axis(axis, 0, 1, 1).remove_axis(axis).copy(),
    };
    let start = if initial.is_some() { 0 } else { 1 };
    for i in start..n {
        let slice = a.slice_axis(axis, i, 1, 1).remove_axis(axis);
        acc = rnp_core::binary(&acc, &slice, op).map_err(crate::err)?;
    }
    Ok(acc)
}

/// The reduce ops that have a native fast path.
fn native_reduce_op(op: BinOp) -> Option<rnp_core::ReduceOp> {
    Some(match op {
        BinOp::Add => rnp_core::ReduceOp::Sum,
        BinOp::Mul => rnp_core::ReduceOp::Prod,
        BinOp::Minimum => rnp_core::ReduceOp::Min,
        BinOp::Maximum => rnp_core::ReduceOp::Max,
        _ => return None,
    })
}

#[pyfunction]
#[pyo3(signature = (name, a, axis, dtype = None, out = None, keepdims = false,
                    initial = None, where_ = None))]
#[allow(clippy::too_many_arguments)]
pub fn _ufunc_reduce<'py>(
    py: Python<'py>,
    name: &str,
    a: &Bound<'py, PyAny>,
    // Taken as a required argument (rather than `Option`) because an explicit
    // `axis=None` means "reduce every axis", which PyO3 would otherwise
    // collapse into the same `None` as "argument omitted".
    axis: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
    initial: Option<&Bound<'py, PyAny>>,
    where_: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let op = match lookup(name) {
        Some(Ufn::Bin(b)) => b,
        _ => {
            return Err(PyValueError::new_err(format!(
                "reduce only supported for binary functions ('{name}')"
            )))
        }
    };
    let mut arr = array_from_any(a, None, false)?;
    if let Some(d) = dtype {
        if !d.is_none() {
            arr = arr.astype(dtype_from_any(d)?);
        }
    }
    // `where=` replaces the masked-out elements with the identity.
    let init_scalar = match initial {
        Some(i) if !i.is_none() => scalar_from_py(i),
        _ => None,
    };
    // The mask, kept alongside the identity substitution below: numpy's
    // masked inner loop reduces each contiguous run of set mask values with a
    // fresh pairwise tree, which the engine can only reproduce if it can still
    // see where the runs are.
    let mut where_mask: Option<NdArray> = None;
    if let Some(w) = where_ {
        if !w.is_none() && !matches!(w.extract::<bool>(), Ok(true)) {
            let ident = init_scalar
                .or_else(|| identity_of(op, arr.dtype()))
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "reduction operation '{}' does not have an identity, so a \
                         where mask requires an initial value",
                        op.name()
                    ))
                })?;
            let mask = array_from_any(w, Some(DType::Bool), false)?;
            let mut base = NdArray::zeros(arr.shape.clone(), arr.dtype()).map_err(crate::err)?;
            base.fill(ident.cast(arr.dtype()));
            where_mask = Some(
                rnp_core::iter::broadcast_to(&mask, &arr.shape).map_err(crate::err)?,
            );
            arr = apply_where(&arr, &base, &mask)?;
        }
    }
    if arr.ndim() == 0 {
        return Err(PyValueError::new_err("cannot reduce on a scalar"));
    }
    // numpy accumulates `add`/`multiply` over bool and narrow integers in the
    // platform integer, exactly as `np.sum`/`np.prod` do.
    if dtype.map(|d| d.is_none()).unwrap_or(true)
        && matches!(op, BinOp::Add | BinOp::Mul)
        && arr.dtype().is_exact()
        && arr.dtype().itemsize() < 8
    {
        arr = arr.astype(if arr.dtype().is_unsigned() {
            DType::U64
        } else {
            DType::I64
        });
    }

    let init = init_scalar;

    if op.is_logical() || op.is_comparison() {
        arr = arr.astype(DType::Bool);
    }
    // The axes to reduce, innermost last.
    let axes = resolve_axes(&arr, axis)?;
    let mut removed: Vec<usize> = axes.clone();
    removed.sort_unstable();
    let native = native_reduce_op(op).filter(|_| arr.dtype().is_numeric());
    // numpy seeds the accumulator with `initial=` before the first element is
    // folded in, which for a sequential fold is *not* the same as combining it
    // with the finished result. A chain of per-axis reductions cannot express
    // that, so the engine is only told about `initial=` when the whole
    // reduction is one collapse; otherwise it stays a post-hoc fold.
    let one_collapse =
        native.is_some() && (removed.len() == 1 || removed.len() == arr.ndim());
    let ropts = rnp_core::reduce::ReduceOpts {
        mask: where_mask.as_ref(),
        seed: if one_collapse { init } else { None },
    };
    let post_init = if one_collapse { None } else { init };

    // `axis=None` is a *single* reduction over the flattened operand, not a
    // chain of per-axis ones: numpy's iterator coalesces the axes of a
    // contiguous operand into one run and runs one pairwise tree over it. A
    // chain of per-axis reductions brackets the additions completely
    // differently and diverges in the last bits.
    let mut cur = if removed.len() == arr.ndim() && native.is_some() {
        let rop = native.expect("checked above");
        let v = rnp_core::reduce::reduce_all_with(&arr, rop, ropts)
            .map_err(crate::err)?;
        let out0 = NdArray::zeros(vec![], rnp_core::reduce::reduce_dtype(rop, arr.dtype()))
            .map_err(crate::err)?;
        out0.write_at(out0.byte_offset, v);
        out0
    } else {
        let mut cur = arr;
        for (nth, &ax) in removed.iter().rev().enumerate() {
            // The mask and the seed only describe the operand as it came in,
            // so they can only steer the first collapse.
            let o = if nth == 0 {
                ropts
            } else {
                rnp_core::reduce::ReduceOpts::default()
            };
            cur = match native {
                Some(rop) if cur.dtype().is_numeric() => {
                    rnp_core::reduce::reduce_axis_with(&cur, ax, rop, false, o)
                        .map_err(crate::err)?
                }
                _ => reduce_along(&cur, ax, op, None)?,
            };
        }
        cur
    };
    // Only reached when the reduction was a chain, where the engine could not
    // seed the accumulator itself.
    if let Some(s) = post_init {
        let mut seed = NdArray::zeros(cur.shape.clone(), cur.dtype()).map_err(crate::err)?;
        seed.fill(s.cast(cur.dtype()));
        cur = rnp_core::binary(&seed, &cur, op).map_err(crate::err)?;
    }
    if keepdims {
        let mut shape = cur.shape.clone();
        for &ax in removed.iter() {
            shape.insert(ax, 1);
        }
        cur = cur.reshape(&shape).map_err(crate::err)?;
    }
    store_or_wrap(py, cur, out)
}

fn resolve_axes(arr: &NdArray, axis: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let nd = arr.ndim() as isize;
    let norm = |i: isize| -> PyResult<usize> {
        let j = if i < 0 { i + nd } else { i };
        if j < 0 || j >= nd {
            return Err(PyValueError::new_err(format!(
                "axis {i} is out of bounds for array of dimension {nd}"
            )));
        }
        Ok(j as usize)
    };
    if axis.is_none() {
        return Ok((0..arr.ndim()).collect());
    }
    if let Ok(i) = axis.extract::<isize>() {
        return Ok(vec![norm(i)?]);
    }
    let t: Vec<isize> = axis.extract()?;
    t.into_iter().map(norm).collect()
}

#[pyfunction]
#[pyo3(signature = (name, a, axis = 0, dtype = None, out = None))]
pub fn _ufunc_accumulate<'py>(
    py: Python<'py>,
    name: &str,
    a: &Bound<'py, PyAny>,
    axis: isize,
    dtype: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let op = match lookup(name) {
        Some(Ufn::Bin(b)) => b,
        _ => {
            return Err(PyValueError::new_err(format!(
                "accumulate only supported for binary functions ('{name}')"
            )))
        }
    };
    let mut arr = array_from_any(a, None, false)?;
    if let Some(d) = dtype {
        if !d.is_none() {
            arr = arr.astype(dtype_from_any(d)?);
        }
    } else if matches!(op, BinOp::Add | BinOp::Mul)
        && arr.dtype().is_exact()
        && arr.dtype().itemsize() < 8
    {
        // numpy accumulates bools and narrow ints in the platform int, as
        // `np.cumsum`/`np.cumprod` do.
        arr = arr.astype(if arr.dtype().is_unsigned() {
            DType::U64
        } else {
            DType::I64
        });
    }
    if op.is_logical() || op.is_comparison() {
        arr = arr.astype(DType::Bool);
    }
    let nd = arr.ndim() as isize;
    let ax = if axis < 0 { axis + nd } else { axis };
    if ax < 0 || ax >= nd {
        return Err(PyValueError::new_err(format!(
            "axis {axis} is out of bounds for array of dimension {nd}"
        )));
    }
    let ax = ax as usize;
    let n = arr.shape[ax];
    // Accumulate slice by slice; the result dtype is the op's own.
    let mut slices: Vec<NdArray> = Vec::with_capacity(n.max(0) as usize);
    let mut acc: Option<NdArray> = None;
    for i in 0..n {
        let s = arr.slice_axis(ax, i, 1, 1).remove_axis(ax);
        let next = match &acc {
            None => s.copy(),
            Some(prev) => rnp_core::binary(prev, &s, op).map_err(crate::err)?,
        };
        slices.push(next.clone());
        acc = Some(next);
    }
    let out_dt = slices.first().map(|s| s.dtype()).unwrap_or(arr.dtype());
    let res = NdArray::empty(arr.shape.clone(), out_dt).map_err(crate::err)?;
    for (i, s) in slices.iter().enumerate() {
        let dst = res.slice_axis(ax, i as isize, 1, 1).remove_axis(ax);
        let so: Vec<isize> = rnp_core::iter::offsets(&s.shape, &s.strides, s.byte_offset).collect();
        let dof: Vec<isize> =
            rnp_core::iter::offsets(&dst.shape, &dst.strides, dst.byte_offset).collect();
        for (k, &d) in dof.iter().enumerate() {
            dst.write_at(d, s.read_at(so[k]));
        }
    }
    store_or_wrap(py, res, out)
}

// ---------------------------------------------------------------------------
// Floating-point error state
// ---------------------------------------------------------------------------

/// Read and clear the accumulated FP error flags.
#[pyfunction]
pub fn _fpe_take() -> u8 {
    fpe::take()
}

#[pyfunction]
pub fn _fpe_clear() {
    fpe::clear();
}

/// Turn the (costly) underflow detection on or off; the shim calls this from
/// `np.seterr`.
#[pyfunction]
pub fn _watch_underflow(on: bool) {
    fpe::set_watch_underflow(on);
}

// ---------------------------------------------------------------------------
// Scalar bridge
// ---------------------------------------------------------------------------

/// Cast a Python value to `dtype` and hand back the Python-native result —
/// what a numpy scalar constructor stores.
#[pyfunction]
pub fn _scalar_cast<'py>(
    py: Python<'py>,
    dtype: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let dt = dtype_from_any(dtype)?;
    let s = match np_scalar(value)? {
        Some((_, s)) => s,
        None => match scalar_from_py(value) {
            Some(s) => s,
            None => {
                // Fall through to the array path so that strings, sequences and
                // buffer objects raise numpy's own messages.
                let a = array_from_any(value, Some(dt), false)?;
                if a.size() != 1 {
                    return Err(PyValueError::new_err(
                        "only length-1 arrays can be converted to Python scalars",
                    ));
                }
                a.get_flat(0)
            }
        },
    };
    crate::convert::scalar_to_py(py, s.cast(dt))
}

/// `scalar OP scalar` / `scalar OP python-number`, with NEP 50 weakness.
///
/// `a_weak`/`b_weak` say whether that operand was a bare Python number (and so
/// adopts the other's dtype). Returns `(dtype, value, fp_flags)`.
#[pyfunction]
#[pyo3(signature = (name, a, b))]
pub fn _scalar_binop<'py>(
    py: Python<'py>,
    name: &str,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let op = match lookup(name) {
        Some(Ufn::Bin(o)) => o,
        Some(Ufn::DivMod) => {
            let (x, y) = scalar_pair(a, b)?;
            let dt = rnp_core::promote(x.0, y.0);
            let dt = if dt == DType::Bool { DType::I8 } else { dt };
            fpe::clear();
            let (q, r) = rnp_core::ops::scalar_divmod(x.1.cast(dt), y.1.cast(dt), dt);
            let flags = fpe::take();
            return Ok(PyTuple::new(
                py,
                [
                    triple(py, dt, q, flags)?,
                    triple(py, dt, r, 0)?,
                ],
            )?
            .into_any());
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "'{name}' is not a binary ufunc"
            )))
        }
    };
    let (x, y) = scalar_pair(a, b)?;
    let mut xa = NdArray::zeros(vec![], x.0).map_err(crate::err)?;
    xa.set(&[], x.1).map_err(crate::err)?;
    let mut ya = NdArray::zeros(vec![], y.0).map_err(crate::err)?;
    ya.set(&[], y.1).map_err(crate::err)?;
    fpe::clear();
    let r = rnp_core::binary(&xa, &ya, op).map_err(crate::err)?;
    let mut flags = fpe::take();
    // numpy reports integer overflow for *scalar* operations only.
    if r.dtype().is_integer() && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Pow) {
        if int_overflowed(op, x.1.cast(r.dtype()), y.1.cast(r.dtype()), r.get_flat(0), r.dtype()) {
            flags |= fpe::OVER;
        }
    }
    triple(py, r.dtype(), r.get_flat(0), flags)
}

/// A unary ufunc on one scalar.
#[pyfunction]
pub fn _scalar_unop<'py>(
    py: Python<'py>,
    name: &str,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let op = match lookup(name) {
        Some(Ufn::Un(o)) => o,
        _ => {
            return Err(PyValueError::new_err(format!(
                "'{name}' is not a unary ufunc"
            )))
        }
    };
    let (dt, v) = one_scalar(a)?;
    let mut xa = NdArray::zeros(vec![], dt).map_err(crate::err)?;
    xa.set(&[], v).map_err(crate::err)?;
    fpe::clear();
    let r = rnp_core::unary(&xa, op).map_err(crate::err)?;
    let mut flags = fpe::take();
    if r.dtype().is_integer() && op == UnOp::Negative {
        if let (Scalar::Int(i), Scalar::Int(o)) = (v.cast(r.dtype()), r.get_flat(0)) {
            if i != 0 && i == o {
                flags |= fpe::OVER;
            }
        }
    }
    triple(py, r.dtype(), r.get_flat(0), flags)
}

/// `(dtype, value, flags)` — what the Python scalar wrapper needs.
fn triple<'py>(py: Python<'py>, dt: DType, v: Scalar, flags: u8) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyTuple::new(
        py,
        [
            PyDType::new(dt).into_pyobject(py)?.into_any(),
            crate::convert::scalar_to_py(py, v)?,
            flags.into_pyobject(py)?.into_any(),
        ],
    )?
    .into_any())
}

fn one_scalar(a: &Bound<'_, PyAny>) -> PyResult<(DType, Scalar)> {
    if let Some(t) = np_scalar(a)? {
        return Ok(t);
    }
    if let Some(s) = scalar_from_py(a) {
        return Ok((s.natural_dtype(), s));
    }
    Err(PyTypeError::new_err("expected a numpy or Python scalar"))
}

/// Classify both operands and apply NEP 50 weak promotion.
fn scalar_pair(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<((DType, Scalar), (DType, Scalar))> {
    let sa = np_scalar(a)?;
    let sb = np_scalar(b)?;
    let (va, vb) = (scalar_from_py(a), scalar_from_py(b));
    match (sa, sb) {
        (Some(x), Some(y)) => Ok((x, y)),
        (Some(x), None) => {
            let s = vb.ok_or_else(|| PyTypeError::new_err("unsupported operand"))?;
            Ok((x, (weak_promote(x.0, s), s)))
        }
        (None, Some(y)) => {
            let s = va.ok_or_else(|| PyTypeError::new_err("unsupported operand"))?;
            Ok(((weak_promote(y.0, s), s), y))
        }
        (None, None) => {
            let (s, t) = (
                va.ok_or_else(|| PyTypeError::new_err("unsupported operand"))?,
                vb.ok_or_else(|| PyTypeError::new_err("unsupported operand"))?,
            );
            Ok(((s.natural_dtype(), s), (t.natural_dtype(), t)))
        }
    }
}

/// Did an integer scalar op leave the type's range? Recomputed in 128 bits.
fn int_overflowed(op: BinOp, x: Scalar, y: Scalar, r: Scalar, dt: DType) -> bool {
    let wide = |s: Scalar| -> i128 {
        match s {
            Scalar::Int(i) => i as i128,
            Scalar::Uint(u) => u as i128,
            Scalar::Bool(b) => b as i128,
            _ => 0,
        }
    };
    let (a, b) = (wide(x), wide(y));
    let exact: i128 = match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => match a.checked_mul(b) {
            Some(v) => v,
            None => return true,
        },
        BinOp::Pow => {
            if b < 0 || b > 127 {
                return false;
            }
            let mut acc: i128 = 1;
            for _ in 0..b {
                acc = match acc.checked_mul(a) {
                    Some(v) => v,
                    None => return true,
                };
            }
            acc
        }
        _ => return false,
    };
    let got = wide(r);
    // Unsigned results wrap into the same 128-bit value only when in range.
    if dt.is_unsigned() {
        return exact < 0 || exact != got;
    }
    exact != got
}


// ---------------------------------------------------------------------------
// The allocation-free scalar fast path
// ---------------------------------------------------------------------------
//
// `np.float64(1.5) + np.float64(2.5)` is ~2.4us against numpy's 0.15us, and
// almost none of that is arithmetic: the old bridge looked the ufunc up by
// name, probed both operands with `getattr`, built two 0-d `NdArray`s, ran the
// array driver (which allocates a third), and handed back a freshly
// constructed `dtype` object inside a tuple. This path does none of that.
// `rnp_core::ops::binary_scalar` runs the same element kernels over `Scalar`
// values, operands are classified by *type pointer* against a registry the
// shim fills in at class-creation time, and the result is `(code, value,
// flags)` where `code` indexes a tuple of wrapper callables on the Python
// side -- no `dtype` object is created at all.

/// The dtypes a numpy scalar can have, in the order the shim's wrapper table
/// uses. `_scalar_dtype_names()` hands the same order to Python.
pub const SCALAR_DTYPES: [DType; 14] = [
    DType::Bool,
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
    DType::F16,
    DType::F32,
    DType::F64,
    DType::C64,
    DType::C128,
];

#[inline]
pub fn dtype_code(d: DType) -> Option<usize> {
    SCALAR_DTYPES.iter().position(|&x| x == d)
}

/// The binary ops reachable from a scalar operator method, in opcode order.
/// The shim resolves each name to its index once, when it builds the classes.
const SCALAR_BINOPS: [BinOp; 18] = [
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::FloorDiv,
    BinOp::Mod,
    BinOp::Pow,
    BinOp::BitAnd,
    BinOp::BitOr,
    BinOp::BitXor,
    BinOp::LShift,
    BinOp::RShift,
    BinOp::Lt,
    BinOp::Le,
    BinOp::Gt,
    BinOp::Ge,
    BinOp::Eq,
    BinOp::Ne,
];

const SCALAR_BINOP_NAMES: [&str; 18] = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "floor_divide",
    "remainder",
    "power",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "left_shift",
    "right_shift",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "equal",
    "not_equal",
];

/// `type(scalar) -> DType`, filled in by `_register_scalar_class`. Reading it
/// is one pointer hash, which replaces two `getattr`s plus a downcast.
static TYPE_REG: OnceLock<std::sync::RwLock<std::collections::HashMap<usize, DType>>> =
    OnceLock::new();
/// The shim's `generic` base class, used only on the miss path.
static GENERIC_CLS: OnceLock<Py<PyAny>> = OnceLock::new();

fn type_reg() -> &'static std::sync::RwLock<std::collections::HashMap<usize, DType>> {
    TYPE_REG.get_or_init(|| std::sync::RwLock::new(std::collections::HashMap::new()))
}

#[pyfunction]
pub fn _register_scalar_class(cls: &Bound<'_, PyAny>, dtype: &Bound<'_, PyAny>) -> PyResult<()> {
    let dt = dtype_from_any(dtype)?;
    let ptr = cls.as_ptr() as usize;
    type_reg().write().unwrap().insert(ptr, dt);
    Ok(())
}

#[pyfunction]
pub fn _register_generic_class(cls: Py<PyAny>) {
    let _ = GENERIC_CLS.set(cls);
}

/// The dtype names in wrapper-table order, so the shim can build the table.
#[pyfunction]
pub fn _scalar_dtype_names() -> Vec<String> {
    SCALAR_DTYPES.iter().map(|d| d.name()).collect()
}

/// The opcode for one ufunc name, or `None` when the scalar fast path has no
/// entry for it (the shim then keeps using the general bridge).
#[pyfunction]
pub fn _scalar_binop_code(name: &str) -> Option<usize> {
    SCALAR_BINOP_NAMES.iter().position(|&n| n == name)
}

/// One classified operand.
enum SOperand {
    /// A numpy scalar: strong under NEP 50, keeps its own dtype.
    Strong(DType, Scalar),
    /// A bare Python number: weak, adopts the other operand's dtype.
    Weak(Scalar),
    /// Anything else.
    Other,
}

#[inline]
fn classify(obj: &Bound<'_, PyAny>) -> PyResult<SOperand> {
    // Exact builtins first: a numpy scalar is never one of these *exactly*
    // (`float64` is a subclass of `float`), so these four pointer compares
    // cannot shadow it.
    if obj.is_exact_instance_of::<pyo3::types::PyBool>() {
        return Ok(SOperand::Weak(Scalar::Bool(obj.extract::<bool>()?)));
    }
    if obj.is_exact_instance_of::<pyo3::types::PyFloat>() {
        return Ok(SOperand::Weak(Scalar::Float(obj.extract::<f64>()?)));
    }
    if obj.is_exact_instance_of::<pyo3::types::PyInt>() {
        return Ok(match scalar_from_py(obj) {
            Some(s) => SOperand::Weak(s),
            // An int wider than 64 bits: no `Scalar` holds it, and the old
            // bridge raised `TypeError` for it. Keep that.
            None => SOperand::Other,
        });
    }
    if obj.is_exact_instance_of::<pyo3::types::PyComplex>() {
        return Ok(match scalar_from_py(obj) {
            Some(s) => SOperand::Weak(s),
            None => SOperand::Other,
        });
    }
    let dt = {
        let reg = type_reg().read().unwrap();
        reg.get(&(obj.get_type().as_ptr() as usize)).copied()
    };
    if let Some(dt) = dt {
        let v = obj.getattr(pyo3::intern!(obj.py(), "_v"))?;
        if let Some(s) = scalar_from_py(&v) {
            return Ok(SOperand::Strong(dt, s.cast(dt)));
        }
        // A flexible scalar (`str_`, `void`, ...) whose payload is not a
        // number: not for this path.
        return Ok(SOperand::Other);
    }
    // Unregistered: subclasses of our scalar types, other libraries' numbers.
    if let Some((d, s)) = np_scalar(obj)? {
        return Ok(SOperand::Strong(d, s));
    }
    Ok(match scalar_from_py(obj) {
        Some(s) => SOperand::Weak(s),
        None => SOperand::Other,
    })
}

/// The guard that used to live in `_scalars._binary`, kept verbatim so the
/// distinction it draws survives: `NotImplemented` (so Python tries the other
/// operand's reflected method, which is how `np.float64(1) < np.arange(3)`
/// reaches `ndarray.__gt__`) when either operand is an array or not a number
/// at all, and the engine's `TypeError` for a `generic`/`int` the engine has
/// no `Scalar` for (an `int` wider than 64 bits, a `str_`, ...).
///
/// Returns `Ok(None)` for the `NotImplemented` case.
fn unusable<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if a.is_instance_of::<PyNdArray>() || b.is_instance_of::<PyNdArray>() {
        return Ok(None);
    }
    let numberish = |o: &Bound<'py, PyAny>| -> bool {
        o.is_instance_of::<pyo3::types::PyInt>()
            || o.is_instance_of::<pyo3::types::PyFloat>()
            || o.is_instance_of::<pyo3::types::PyComplex>()
            || GENERIC_CLS
                .get()
                .is_some_and(|g| o.is_instance(g.bind(py)).unwrap_or(false))
    };
    if !numberish(a) || !numberish(b) {
        return Ok(None);
    }
    Err(PyTypeError::new_err("unsupported operand"))
}

/// `scalar OP scalar` with no allocation: the whole point of this module.
///
/// Returns `(dtype_code, value, fp_flags)`, or `None` when Python should see
/// `NotImplemented`.
#[pyfunction]
pub fn _scalar_binop2<'py>(
    py: Python<'py>,
    code: usize,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let op = SCALAR_BINOPS[code];
    let (oa, ob) = (classify(a)?, classify(b)?);
    let ((dta, xa), (dtb, xb)) = match (oa, ob) {
        (SOperand::Strong(d, s), SOperand::Strong(e, t)) => ((d, s), (e, t)),
        (SOperand::Strong(d, s), SOperand::Weak(t)) => ((d, s), (weak_promote(d, t), t)),
        (SOperand::Weak(s), SOperand::Strong(e, t)) => ((weak_promote(e, s), s), (e, t)),
        (SOperand::Weak(s), SOperand::Weak(t)) => ((s.natural_dtype(), s), (t.natural_dtype(), t)),
        _ => return unusable(py, a, b),
    };
    fpe::clear();
    let (out_dt, mut value) =
        match rnp_core::ops::binary_scalar(xa.cast(dta), dta, xb.cast(dtb), dtb, op) {
            Some(r) => r.map_err(crate::err)?,
            // Flexible / object / datetime operands: fall back to the general
            // array bridge so their messages stay numpy's.
            None => {
                let mut ya = NdArray::zeros(vec![], dta).map_err(crate::err)?;
                ya.set(&[], xa).map_err(crate::err)?;
                let mut yb = NdArray::zeros(vec![], dtb).map_err(crate::err)?;
                yb.set(&[], xb).map_err(crate::err)?;
                let r = rnp_core::binary(&ya, &yb, op).map_err(crate::err)?;
                (r.dtype(), r.get_flat(0))
            }
        };
    let mut flags = fpe::take();
    // numpy reports integer overflow for *scalar* operations only.
    if out_dt.is_integer() && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Pow) {
        if int_overflowed(op, xa.cast(out_dt), xb.cast(out_dt), value, out_dt) {
            flags |= fpe::OVER;
        }
    }
    if out_dt == DType::F16 {
        // The Python side stores float16 as a double; `Scalar::Float` already
        // holds the widened value, so nothing to do -- kept explicit so the
        // invariant is visible.
        value = value.cast(DType::F64);
    }
    let dc = dtype_code(out_dt).ok_or_else(|| {
        PyValueError::new_err(format!(
            "scalar '{}' produced a {} result, which has no scalar class",
            op.name(),
            out_dt.name()
        ))
    })?;
    Ok(Some(
        PyTuple::new(
            py,
            [
                dc.into_pyobject(py)?.into_any(),
                crate::convert::scalar_to_py(py, value)?,
                flags.into_pyobject(py)?.into_any(),
            ],
        )?
        .into_any(),
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_ufunc_call, m)?)?;
    m.add_function(wrap_pyfunction!(_ufunc_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(_ufunc_accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(_fpe_take, m)?)?;
    m.add_function(wrap_pyfunction!(_register_fpe_reporter, m)?)?;
    m.add_function(wrap_pyfunction!(_fpe_clear, m)?)?;
    m.add_function(wrap_pyfunction!(_watch_underflow, m)?)?;
    m.add_function(wrap_pyfunction!(_scalar_cast, m)?)?;
    m.add_function(wrap_pyfunction!(_scalar_binop, m)?)?;
    m.add_function(wrap_pyfunction!(_scalar_unop, m)?)?;
    m.add_function(wrap_pyfunction!(_scalar_binop2, m)?)?;
    m.add_function(wrap_pyfunction!(_scalar_binop_code, m)?)?;
    m.add_function(wrap_pyfunction!(_scalar_dtype_names, m)?)?;
    m.add_function(wrap_pyfunction!(_register_scalar_class, m)?)?;
    m.add_function(wrap_pyfunction!(_register_generic_class, m)?)?;
    Ok(())
}

/// Silence the unused-import warning for `PyDict` when features change.
#[allow(dead_code)]
fn _unused(_: &Bound<'_, PyDict>) {}
