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

static FPE_REPORTER: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

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
fn resolve_operands(objs: &[Bound<'_, PyAny>], comparison: bool) -> PyResult<Vec<NdArray>> {
    let ops: Vec<Operand> = objs.iter().map(coerce).collect::<PyResult<_>>()?;
    let mut strong: Option<DType> = None;
    for o in &ops {
        if let Some(a) = &o.arr {
            strong = Some(match strong {
                None => a.dtype,
                Some(d) => rnp_core::promote(d, a.dtype),
            });
        }
    }
    let mut out = Vec::with_capacity(ops.len());
    for o in ops {
        match (o.arr, o.weak) {
            (Some(a), _) => out.push(a),
            (None, Some(s)) => {
                let d = match strong {
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
    let out = NdArray::empty(res.shape.clone(), res.dtype).map_err(crate::err)?;
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
    let mut inputs = resolve_operands(&objs, matches!(f, Ufn::Bin(b) if b.is_comparison()))?;
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
                _ => NdArray::zeros(res.shape.clone(), res.dtype).map_err(crate::err)?,
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
    let dt = match a.dtype {
        DType::F16 | DType::F32 | DType::F64 => a.dtype,
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
        let ident = initial.or_else(|| identity_of(op, a.dtype)).ok_or_else(|| {
            PyValueError::new_err(format!(
                "zero-size array to reduction operation {} which has no identity",
                op.name()
            ))
        })?;
        let mut shape = a.shape.clone();
        shape.remove(axis);
        let (_, out_dt) = rnp_core::ops::result_dtypes(a.dtype, a.dtype, op).map_err(crate::err)?;
        let mut z = NdArray::zeros(shape, out_dt).map_err(crate::err)?;
        z.fill(ident.cast(out_dt));
        return Ok(z);
    }
    let mut acc = match &initial {
        Some(s) => {
            let mut shape = a.shape.clone();
            shape.remove(axis);
            let mut z = NdArray::zeros(shape, a.dtype).map_err(crate::err)?;
            z.fill(s.cast(a.dtype));
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
    if let Some(w) = where_ {
        if !w.is_none() && !matches!(w.extract::<bool>(), Ok(true)) {
            let ident = init_scalar
                .or_else(|| identity_of(op, arr.dtype))
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "reduction operation '{}' does not have an identity, so a \
                         where mask requires an initial value",
                        op.name()
                    ))
                })?;
            let mask = array_from_any(w, Some(DType::Bool), false)?;
            let mut base = NdArray::zeros(arr.shape.clone(), arr.dtype).map_err(crate::err)?;
            base.fill(ident.cast(arr.dtype));
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
        && arr.dtype.is_exact()
        && arr.dtype.itemsize() < 8
    {
        arr = arr.astype(if arr.dtype.is_unsigned() {
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
    let mut cur = arr;
    let mut removed: Vec<usize> = axes.clone();
    removed.sort_unstable();
    for &ax in removed.iter().rev() {
        cur = match native_reduce_op(op) {
            Some(rop) if cur.dtype.is_numeric() => {
                rnp_core::reduce_axis(&cur, ax, rop, false).map_err(crate::err)?
            }
            _ => reduce_along(&cur, ax, op, None)?,
        };
    }
    // `initial` seeds the whole reduction once, not once per axis, so it is
    // folded in after every axis has collapsed.
    if let Some(s) = init {
        let mut seed = NdArray::zeros(cur.shape.clone(), cur.dtype).map_err(crate::err)?;
        seed.fill(s.cast(cur.dtype));
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
        && arr.dtype.is_exact()
        && arr.dtype.itemsize() < 8
    {
        // numpy accumulates bools and narrow ints in the platform int, as
        // `np.cumsum`/`np.cumprod` do.
        arr = arr.astype(if arr.dtype.is_unsigned() {
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
    let out_dt = slices.first().map(|s| s.dtype).unwrap_or(arr.dtype);
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
    if r.dtype.is_integer() && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Pow) {
        if int_overflowed(op, x.1.cast(r.dtype), y.1.cast(r.dtype), r.get_flat(0), r.dtype) {
            flags |= fpe::OVER;
        }
    }
    triple(py, r.dtype, r.get_flat(0), flags)
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
    if r.dtype.is_integer() && op == UnOp::Negative {
        if let (Scalar::Int(i), Scalar::Int(o)) = (v.cast(r.dtype), r.get_flat(0)) {
            if i != 0 && i == o {
                flags |= fpe::OVER;
            }
        }
    }
    triple(py, r.dtype, r.get_flat(0), flags)
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
    Ok(())
}

/// Silence the unused-import warning for `PyDict` when features change.
#[allow(dead_code)]
fn _unused(_: &Bound<'_, PyDict>) {}
