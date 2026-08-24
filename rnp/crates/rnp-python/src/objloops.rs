//! Object-dtype ufunc loops.
//!
//! numpy's object loops are thin wrappers around the Python protocols: the
//! arithmetic ufuncs call `PyNumber_*`, the comparisons call
//! `PyObject_RichCompare`, and a large family of the transcendental ufuncs
//! call a *method of the same name* on each element (`np.sqrt` on object
//! calls `element.sqrt()`).  Which family a given ufunc belongs to is not
//! guessable -- it is `TD(O, f=...)` versus `TD(P, f=...)` in numpy's
//! `generate_umath.py` -- so the table below was transcribed from that file
//! and then checked element by element against real numpy 2.5.2.
//!
//! Everything here needs the GIL, so it lives on the `rnp-python` side; the
//! engine never sees a Python object.

use std::ffi::CStr;
use std::sync::OnceLock;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use rnp_core::element::Scalar;
use rnp_core::{DType, NdArray};

use crate::convert::element_to_py;
use crate::objects;
use crate::pyarray::{store_or_wrap, PyNdArray};

// ---------------------------------------------------------------------------
// The loop table
// ---------------------------------------------------------------------------

/// How a unary ufunc computes one element.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum UnObj {
    /// `PyNumber_Negative` and friends.
    Neg,
    Pos,
    Abs,
    Invert,
    /// `PyNumber_Multiply(o, o)`.
    Square,
    /// `PyNumber_TrueDivide(1, o)`.
    Reciprocal,
    /// `Py_get_one`: always the Python int `1`.
    One,
    /// `PyObject_Not`, giving a Python `bool`.
    LogicalNot,
    /// `math.floor` / `math.ceil` / `math.trunc`.
    Math(&'static str),
    /// numpy's three-way probe against the int `0`.
    Sign,
    /// `PyUFunc_O_O_method`: call the same-named method on the element.
    Method(&'static str),
}

/// How a binary ufunc computes one element.
///
/// No `PartialEq`: pyo3's `CompareOp` has none, and every use site is a
/// `matches!` pattern anyway.
#[derive(Copy, Clone, Debug)]
pub enum BinObj {
    Add,
    Sub,
    Mul,
    TrueDiv,
    FloorDiv,
    Rem,
    /// `PyNumber_Power(x, y, Py_None)`.
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    LShift,
    RShift,
    /// Python's `a and b` / `a or b`, returning one of the operands.
    LogicalAnd,
    LogicalOr,
    /// `PyObject_RichCompareBool(a, b, Py_GE)` picks `a`, else `b`.
    Max,
    /// ... `Py_LE`.
    Min,
    Gcd,
    Lcm,
    /// The six comparisons. `bool_out` is decided by the caller: numpy
    /// registers both `OO->?` (the default) and `OO->O`.
    Cmp(pyo3::pyclass::CompareOp),
    /// `PyUFunc_OO_O_method`: `a.meth(b)`.
    Method(&'static str),
}

#[derive(Copy, Clone, Debug)]
pub enum ObjOp {
    Un(UnObj),
    Bin(BinObj),
}

use pyo3::pyclass::CompareOp as Cmp;

/// The object loop for one ufunc name, or `None` when numpy registers no
/// object loop at all for it.
pub fn classify(name: &str) -> Option<ObjOp> {
    let un = |k| Some(ObjOp::Un(k));
    let bin = |k| Some(ObjOp::Bin(k));
    match name {
        // -- unary, operator-driven ---------------------------------------
        "negative" => return un(UnObj::Neg),
        "positive" => return un(UnObj::Pos),
        "absolute" | "abs" => return un(UnObj::Abs),
        "invert" | "bitwise_invert" | "bitwise_not" => return un(UnObj::Invert),
        "square" => return un(UnObj::Square),
        "reciprocal" => return un(UnObj::Reciprocal),
        "_ones_like" => return un(UnObj::One),
        "logical_not" => return un(UnObj::LogicalNot),
        "floor" => return un(UnObj::Math("floor")),
        "ceil" => return un(UnObj::Math("ceil")),
        "trunc" => return un(UnObj::Math("trunc")),
        "sign" => return un(UnObj::Sign),

        // -- unary, method-driven -----------------------------------------
        "conjugate" | "conj" => return un(UnObj::Method("conjugate")),
        "sqrt" => return un(UnObj::Method("sqrt")),
        "cbrt" => return un(UnObj::Method("cbrt")),
        "exp" => return un(UnObj::Method("exp")),
        "exp2" => return un(UnObj::Method("exp2")),
        "expm1" => return un(UnObj::Method("expm1")),
        "log" => return un(UnObj::Method("log")),
        "log2" => return un(UnObj::Method("log2")),
        "log10" => return un(UnObj::Method("log10")),
        "log1p" => return un(UnObj::Method("log1p")),
        "sin" => return un(UnObj::Method("sin")),
        "cos" => return un(UnObj::Method("cos")),
        "tan" => return un(UnObj::Method("tan")),
        "arcsin" | "asin" => return un(UnObj::Method("arcsin")),
        "arccos" | "acos" => return un(UnObj::Method("arccos")),
        "arctan" | "atan" => return un(UnObj::Method("arctan")),
        "sinh" => return un(UnObj::Method("sinh")),
        "cosh" => return un(UnObj::Method("cosh")),
        "tanh" => return un(UnObj::Method("tanh")),
        "arcsinh" | "asinh" => return un(UnObj::Method("arcsinh")),
        "arccosh" | "acosh" => return un(UnObj::Method("arccosh")),
        "arctanh" | "atanh" => return un(UnObj::Method("arctanh")),
        "rint" => return un(UnObj::Method("rint")),
        "fabs" => return un(UnObj::Method("fabs")),
        "degrees" => return un(UnObj::Method("degrees")),
        "radians" => return un(UnObj::Method("radians")),
        "deg2rad" => return un(UnObj::Method("deg2rad")),
        "rad2deg" => return un(UnObj::Method("rad2deg")),
        "bitwise_count" => return un(UnObj::Method("bit_count")),

        // -- binary, operator-driven --------------------------------------
        "add" => return bin(BinObj::Add),
        "subtract" => return bin(BinObj::Sub),
        "multiply" => return bin(BinObj::Mul),
        "divide" | "true_divide" => return bin(BinObj::TrueDiv),
        "floor_divide" => return bin(BinObj::FloorDiv),
        "remainder" | "mod" => return bin(BinObj::Rem),
        "power" | "pow" => return bin(BinObj::Pow),
        "bitwise_and" => return bin(BinObj::BitAnd),
        "bitwise_or" => return bin(BinObj::BitOr),
        "bitwise_xor" => return bin(BinObj::BitXor),
        "left_shift" | "bitwise_left_shift" => return bin(BinObj::LShift),
        "right_shift" | "bitwise_right_shift" => return bin(BinObj::RShift),
        "logical_and" => return bin(BinObj::LogicalAnd),
        "logical_or" => return bin(BinObj::LogicalOr),
        "maximum" | "fmax" => return bin(BinObj::Max),
        "minimum" | "fmin" => return bin(BinObj::Min),
        "gcd" => return bin(BinObj::Gcd),
        "lcm" => return bin(BinObj::Lcm),

        // -- binary, method-driven ----------------------------------------
        "fmod" => return bin(BinObj::Method("fmod")),
        "arctan2" | "atan2" => return bin(BinObj::Method("arctan2")),
        "hypot" => return bin(BinObj::Method("hypot")),
        "logical_xor" => return bin(BinObj::Method("logical_xor")),

        // -- comparisons ---------------------------------------------------
        "equal" => return bin(BinObj::Cmp(Cmp::Eq)),
        "not_equal" => return bin(BinObj::Cmp(Cmp::Ne)),
        "less" => return bin(BinObj::Cmp(Cmp::Lt)),
        "less_equal" => return bin(BinObj::Cmp(Cmp::Le)),
        "greater" => return bin(BinObj::Cmp(Cmp::Gt)),
        "greater_equal" => return bin(BinObj::Cmp(Cmp::Ge)),

        // Everything else -- `isnan`, `divmod`, `ldexp`, `logaddexp`, ... --
        // has no `O` loop at all.
        _ => None,
    }
}

/// The shim's `_UFuncInputCastingError` builder. The engine cannot construct
/// it itself -- the payload needs the ufunc *object* and real dtype objects --
/// so `rnp_numpy._objectops` registers a factory at import time.
static CAST_ERROR: OnceLock<Py<PyAny>> = OnceLock::new();

#[pyfunction]
pub fn _register_object_cast_error(f: Py<PyAny>) {
    let _ = CAST_ERROR.set(f);
}

/// numpy's `AxisError`, registered by the shim for the same reason: it is a
/// Python class (`numpy.exceptions.AxisError`) the engine cannot define.
static AXIS_ERROR: OnceLock<Py<PyAny>> = OnceLock::new();

#[pyfunction]
pub fn _register_axis_error(cls: Py<PyAny>) {
    let _ = AXIS_ERROR.set(cls);
}

/// `numpy.exceptions.AxisError`, falling back to `ValueError` (which it
/// subclasses) when the shim has not registered the class.
pub fn axis_error(msg: String) -> PyErr {
    Python::attach(|py| match AXIS_ERROR.get() {
        Some(c) => PyErr::from_type(
            c.bind(py).cast::<pyo3::types::PyType>().expect("AxisError is a class").clone(),
            (msg,),
        ),
        None => PyValueError::new_err(msg),
    })
}

/// numpy's `Cannot cast ufunc 'add' input 0 from dtype('O') to ...`, as the
/// `UFuncTypeError` subclass numpy raises. Falls back to a plain `TypeError`
/// with the same text when the shim is not present.
fn cast_error(py: Python<'_>, name: &str, to: DType) -> PyErr {
    let msg = format!(
        "Cannot cast ufunc '{name}' input 0 from dtype('O') to dtype('{}') \
         with casting rule 'same_kind'",
        to.name()
    );
    if let Some(f) = CAST_ERROR.get() {
        if let Ok(exc) = f.bind(py).call1((name, "same_kind", "O", to.name(), 0usize)) {
            return PyErr::from_value(exc);
        }
    }
    PyTypeError::new_err(msg)
}

/// numpy's message when the operand types reach no loop. This is the *legacy
/// type resolver's* plain `TypeError`, not a `_UFuncNoLoopError`; probed
/// against real numpy for `np.isnan(np.array([1], dtype=object))`.
pub fn no_loop_error(name: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "ufunc '{name}' not supported for the input types, and the inputs \
         could not be safely coerced to any supported types according to the \
         casting rule ''safe''"
    ))
}

/// The reduction identity numpy uses for an *empty* object reduction.
pub fn identity<'py>(py: Python<'py>, op: BinObj) -> Option<Bound<'py, PyAny>> {
    let int = |v: i64| v.into_pyobject(py).ok().map(|o| o.into_any());
    match op {
        BinObj::Add | BinObj::BitOr | BinObj::BitXor | BinObj::Gcd => int(0),
        BinObj::Mul => int(1),
        BinObj::BitAnd => int(-1),
        BinObj::LogicalAnd => Some(true.into_pyobject(py).ok()?.to_owned().into_any()),
        BinObj::LogicalOr => Some(false.into_pyobject(py).ok()?.to_owned().into_any()),
        BinObj::Method("logical_xor") => Some(false.into_pyobject(py).ok()?.to_owned().into_any()),
        BinObj::Method("hypot") => int(0),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Per-element evaluation
// ---------------------------------------------------------------------------

/// `type(obj)->tp_name`, which is what numpy's method-loop error prints. It is
/// *not* `__name__` (`numpy.float64` vs `float64`) and *not* the fully
/// qualified name (`Bare` vs `__main__.Bare`).
fn tp_name(obj: &Bound<'_, PyAny>) -> String {
    let ty = obj.get_type();
    // SAFETY: `ty` is a live, borrowed type object for as long as `ty` is in
    // scope, and `tp_name` on a valid `PyTypeObject` is a NUL-terminated
    // static or heap string owned by that type.
    unsafe {
        let p = (ty.as_ptr() as *mut pyo3::ffi::PyTypeObject).as_ref();
        match p.and_then(|t| (!t.tp_name.is_null()).then_some(t.tp_name)) {
            Some(n) => CStr::from_ptr(n).to_string_lossy().into_owned(),
            None => "<unknown>".to_string(),
        }
    }
}

/// `PyUFunc_O_O_method`: fetch a callable attribute of the same name and call
/// it with no arguments. A missing (or non-callable) attribute becomes
/// numpy's own `TypeError`, chaining whatever the lookup raised as the cause.
fn call_method0<'py>(
    x: &Bound<'py, PyAny>,
    meth: &str,
    i: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let looked_up = x.getattr(meth);
    let f = match looked_up {
        Ok(f) if f.is_callable() => f,
        other => {
            let cause = other.err();
            let err = PyTypeError::new_err(format!(
                "loop of ufunc does not support argument {i} of type {} which \
                 has no callable {meth} method",
                tp_name(x)
            ));
            if let Some(c) = cause {
                let py = x.py();
                err.set_cause(py, Some(c));
            }
            return Err(err);
        }
    };
    f.call0()
}

/// The context a loop needs beyond the elements themselves.
pub struct Ctx<'py> {
    py: Python<'py>,
    math: Bound<'py, PyAny>,
}

impl<'py> Ctx<'py> {
    pub fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Ctx {
            py,
            math: py.import("math")?.into_any(),
        })
    }
}

pub fn eval_un<'py>(
    ctx: &Ctx<'py>,
    op: UnObj,
    x: &Bound<'py, PyAny>,
    i: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    Ok(match op {
        UnObj::Neg => x.neg()?,
        UnObj::Pos => x.pos()?,
        UnObj::Abs => x.abs()?,
        UnObj::Invert => x.bitnot()?,
        UnObj::Square => x.mul(x)?,
        UnObj::Reciprocal => 1i64.into_pyobject(py)?.into_any().div(x)?,
        UnObj::One => 1i64.into_pyobject(py)?.into_any(),
        UnObj::LogicalNot => (!x.is_truthy()?).into_pyobject(py)?.to_owned().into_any(),
        UnObj::Math(f) => ctx.math.call_method1(f, (x,))?,
        UnObj::Sign => sign(ctx, x)?,
        UnObj::Method(m) => call_method0(x, m, i)?,
    })
}

/// `OBJECT_sign`: probe `< 0`, then `> 0`, then `== 0`; a value that answers
/// "no" to all three is numpy's "unorderable" case.
fn sign<'py>(ctx: &Ctx<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let zero = 0i64.into_pyobject(py)?.into_any();
    if x.lt(&zero)? {
        return Ok((-1i64).into_pyobject(py)?.into_any());
    }
    if x.gt(&zero)? {
        return Ok(1i64.into_pyobject(py)?.into_any());
    }
    if x.eq(&zero)? {
        return Ok(zero);
    }
    Err(PyTypeError::new_err("unorderable types for comparison"))
}

pub fn eval_bin<'py>(
    ctx: &Ctx<'py>,
    op: BinObj,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    Ok(match op {
        BinObj::Add => a.add(b)?,
        BinObj::Sub => a.sub(b)?,
        BinObj::Mul => a.mul(b)?,
        BinObj::TrueDiv => a.div(b)?,
        BinObj::FloorDiv => a.floor_div(b)?,
        BinObj::Rem => a.rem(b)?,
        BinObj::Pow => a.pow(b, py.None())?,
        BinObj::BitAnd => a.bitand(b)?,
        BinObj::BitOr => a.bitor(b)?,
        BinObj::BitXor => a.bitxor(b)?,
        BinObj::LShift => a.lshift(b)?,
        BinObj::RShift => a.rshift(b)?,
        BinObj::LogicalAnd => {
            if a.is_truthy()? {
                b.clone()
            } else {
                a.clone()
            }
        }
        BinObj::LogicalOr => {
            if a.is_truthy()? {
                a.clone()
            } else {
                b.clone()
            }
        }
        BinObj::Max => {
            if a.ge(b)? {
                a.clone()
            } else {
                b.clone()
            }
        }
        BinObj::Min => {
            if a.le(b)? {
                a.clone()
            } else {
                b.clone()
            }
        }
        BinObj::Gcd => gcd(ctx, a, b)?,
        BinObj::Lcm => {
            // numpy: `abs(a // gcd(a, b) * b)`.
            let g = gcd(ctx, a, b)?;
            a.floor_div(&g)?.mul(b)?.abs()?
        }
        BinObj::Cmp(c) => a.rich_compare(b, c)?,
        BinObj::Method(m) => a.call_method1(m, (b,))?,
    })
}

/// `npy_ObjectGCD`: try `math.gcd`, and on *any* failure fall back to the
/// Euclidean loop numpy keeps in `numpy._core._internal._gcd`, whose result is
/// then made positive.
fn gcd<'py>(
    ctx: &Ctx<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(g) = ctx.math.call_method1("gcd", (a, b)) {
        return Ok(g);
    }
    // `_gcd`, transcribed: the `isfinite` guard is what produces numpy's
    // "must be real number, not X" for a type with no numeric protocol.
    for v in [a, b] {
        if !ctx.math.call_method1("isfinite", (v,))?.is_truthy()? {
            return Err(PyValueError::new_err(
                "Can only find greatest common divisor of finite arguments.",
            ));
        }
    }
    let mut x = a.clone();
    let mut y = b.clone();
    while y.is_truthy()? {
        let r = x.rem(&y)?;
        x = y;
        y = r;
    }
    x.abs()
}

// ---------------------------------------------------------------------------
// Operand handling
// ---------------------------------------------------------------------------

/// True when this operand forces the object loop.
pub fn is_object_operand(obj: &Bound<'_, PyAny>) -> bool {
    match obj.cast::<PyNdArray>() {
        Ok(a) => a.borrow().arr.dtype().is_object(),
        Err(_) => false,
    }
}

/// Coerce one ufunc argument to an array the object loop can read from.
/// Non-object arrays stay in their own dtype and are converted element by
/// element, which is what numpy's `O` cast does.
fn coerce(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    if let Ok(a) = obj.cast::<PyNdArray>() {
        return Ok(a.borrow().arr.clone());
    }
    // A typed operand keeps its dtype and is converted one element at a time,
    // which is what numpy's cast-to-`O` does: `np.float64(2.5)` reaches the
    // loop as a Python `float`, not as a numpy scalar.
    match crate::convert::array_from_any(obj, None, false) {
        Ok(a) => Ok(a),
        Err(_) => objects::array_from_objects(obj),
    }
}

fn broadcast_all(inputs: &[NdArray]) -> PyResult<Vec<isize>> {
    let mut shape: Vec<isize> = Vec::new();
    for a in inputs {
        shape = rnp_core::iter::broadcast_shapes(&shape, &a.shape).map_err(crate::err)?;
    }
    Ok(shape)
}

/// A truth test over one element of a `where=` mask.
fn mask_at(m: &NdArray, off: isize) -> bool {
    !matches!(m.read_at(off), Scalar::Bool(false) | Scalar::Int(0) | Scalar::Uint(0))
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// `ufunc(*args, out=, where=)` for object operands.
pub fn call<'py>(
    py: Python<'py>,
    name: &str,
    objs: &[Bound<'py, PyAny>],
    out: Option<&Bound<'py, PyAny>>,
    where_: Option<&Bound<'py, PyAny>>,
    dtype: Option<DType>,
) -> PyResult<Bound<'py, PyAny>> {
    let op = classify(name).ok_or_else(|| no_loop_error(name))?;
    // An explicit non-object `dtype=` cannot be honoured over object inputs:
    // numpy refuses the cast rather than converting. Comparisons report it
    // differently, because their default loop is `OO->?` and the signature
    // simply does not exist.
    if let Some(dt) = dtype {
        if !dt.is_object() {
            if matches!(op, ObjOp::Bin(BinObj::Cmp(_))) {
                return Err(PyTypeError::new_err(format!(
                    "No loop matching the specified signature and casting was \
                     found for ufunc {name}"
                )));
            }
            return Err(cast_error(py, name, dt));
        }
    }
    let nin = match op {
        ObjOp::Un(_) => 1,
        ObjOp::Bin(_) => 2,
    };
    if objs.len() != nin {
        return Err(PyTypeError::new_err(format!(
            "invalid number of arguments to ufunc '{name}'"
        )));
    }
    let ctx = Ctx::new(py)?;
    let inputs: Vec<NdArray> = objs.iter().map(coerce).collect::<PyResult<_>>()?;
    let shape = broadcast_all(&inputs)?;

    // The destination array, if one was handed in: it decides the loop for
    // comparisons (numpy registers `OO->?` and `OO->O`, and picks the latter
    // when the output is an object array).
    let out_arr = match out {
        Some(o) if !o.is_none() => Some(
            o.cast::<PyNdArray>()
                .map_err(|_| PyTypeError::new_err("return arrays must be of ArrayType"))?
                .borrow()
                .arr
                .clone(),
        ),
        _ => None,
    };
    // `OO->?` is the default comparison loop; `OO->O` is selected by asking
    // for it, either through `dtype=object` or an object destination.
    let bool_out = matches!(op, ObjOp::Bin(BinObj::Cmp(_)))
        && dtype != Some(DType::Object)
        && !out_arr.as_ref().is_some_and(|a| a.dtype().is_object());
    let res_dt = if bool_out { DType::Bool } else { DType::Object };

    let mask = match where_ {
        Some(w) if !w.is_none() && !matches!(w.extract::<bool>(), Ok(true)) => Some(
            rnp_core::iter::broadcast_to(
                &crate::convert::array_from_any(w, Some(DType::Bool), false)?,
                &shape,
            )
            .map_err(crate::err)?,
        ),
        _ => None,
    };

    let res = NdArray::zeros(shape.clone(), res_dt).map_err(crate::err)?;
    // Outside the mask numpy leaves the destination alone; with no
    // destination the result is whatever `np.empty` held, which for object is
    // `None` -- and that is exactly what `zeros` gives here.
    if mask.is_some() {
        if let Some(dest) = &out_arr {
            if dest.shape == shape && dest.dtype() == res_dt {
                let d: Vec<isize> =
                    rnp_core::iter::offsets(&dest.shape, &dest.strides, dest.byte_offset).collect();
                let r: Vec<isize> =
                    rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset).collect();
                for (k, &ro) in r.iter().enumerate() {
                    res.write_at(ro, dest.read_at(d[k]));
                }
            }
        }
    }

    let bcast: Vec<NdArray> = inputs
        .iter()
        .map(|a| rnp_core::iter::broadcast_to(a, &shape).map_err(crate::err))
        .collect::<PyResult<_>>()?;
    let offs: Vec<Vec<isize>> = bcast
        .iter()
        .map(|a| rnp_core::iter::offsets(&a.shape, &a.strides, a.byte_offset).collect())
        .collect();
    let ro: Vec<isize> =
        rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset).collect();
    let mo: Option<Vec<isize>> = mask
        .as_ref()
        .map(|m| rnp_core::iter::offsets(&m.shape, &m.strides, m.byte_offset).collect());

    if let Some(dest) = &out_arr {
        if dest.shape != shape {
            let mut parts: Vec<String> = bcast.iter().map(|a| fmt_shape(&a.shape)).collect();
            parts.push(fmt_shape(&dest.shape));
            return Err(PyValueError::new_err(format!(
                "operands could not be broadcast together with shapes {} ",
                parts.join(" ")
            )));
        }
    }

    for k in 0..ro.len() {
        if let (Some(m), Some(mofs)) = (mask.as_ref(), mo.as_ref()) {
            if !mask_at(m, mofs[k]) {
                continue;
            }
        }
        let value = match op {
            ObjOp::Un(u) => {
                let x = element_to_py(py, &bcast[0], offs[0][k])?;
                eval_un(&ctx, u, &x, k)?
            }
            ObjOp::Bin(b) => {
                let x = element_to_py(py, &bcast[0], offs[0][k])?;
                let y = element_to_py(py, &bcast[1], offs[1][k])?;
                eval_bin(&ctx, b, &x, &y)?
            }
        };
        if bool_out {
            res.write_at(ro[k], Scalar::Bool(value.is_truthy()?));
        } else {
            objects::write(&res, ro[k], &value);
        }
    }
    store_or_wrap(py, res, out)
}

// ---------------------------------------------------------------------------
// reduce / accumulate
// ---------------------------------------------------------------------------

/// One sequential left fold over `items`, seeded with `seed` when given.
fn fold<'py>(
    ctx: &Ctx<'py>,
    op: BinObj,
    name: &str,
    seed: Option<Bound<'py, PyAny>>,
    items: &mut dyn Iterator<Item = PyResult<Bound<'py, PyAny>>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut acc = match seed {
        Some(s) => s,
        None => match items.next() {
            Some(v) => v?,
            None => {
                return identity(ctx.py, op).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "zero-size array to reduction operation {name} which has no identity"
                    ))
                })
            }
        },
    };
    for v in items {
        acc = eval_bin(ctx, op, &acc, &v?)?;
    }
    Ok(acc)
}

/// `ufunc.reduce` over an object array.
pub fn reduce<'py>(
    py: Python<'py>,
    name: &str,
    arr: &NdArray,
    axes: &[usize],
    keepdims: bool,
    initial: Option<&Bound<'py, PyAny>>,
    out: Option<&Bound<'py, PyAny>>,
    where_: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let op = match classify(name) {
        Some(ObjOp::Bin(b)) => b,
        Some(ObjOp::Un(_)) => {
            return Err(PyValueError::new_err(format!(
                "reduce only supported for binary functions ('{name}')"
            )))
        }
        None => return Err(no_loop_error(name)),
    };
    if matches!(op, BinObj::Cmp(_)) {
        return Err(PyTypeError::new_err(format!(
            "No loop matching the specified signature and casting was found \
             for ufunc {name}"
        )));
    }
    let seed = match initial {
        Some(i) if !i.is_none() => Some(i.clone()),
        _ => None,
    };
    // Object reductions carry no usable identity for a masked fold: numpy
    // rejects `where=` unless `initial=` was given, for *every* object loop
    // including `add` (probed against 2.5.2).
    let masked = matches!(where_, Some(w) if !w.is_none()
                          && !matches!(w.extract::<bool>(), Ok(true)));
    if masked && seed.is_none() {
        return Err(PyValueError::new_err(format!(
            "reduction operation '{name}' does not have an identity, so to use \
             a where mask one has to specify 'initial'"
        )));
    }
    let mask = if masked {
        Some(
            rnp_core::iter::broadcast_to(
                &crate::convert::array_from_any(
                    where_.expect("checked above"),
                    Some(DType::Bool),
                    false,
                )?,
                &arr.shape,
            )
            .map_err(crate::err)?,
        )
    } else {
        None
    };

    let ctx = Ctx::new(py)?;
    let mut sorted: Vec<usize> = axes.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let res = if sorted.len() == arr.ndim() {
        // A single fold over the whole operand in C order, as numpy's
        // coalesced iterator does.
        let offs: Vec<isize> =
            rnp_core::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset).collect();
        let mofs: Option<Vec<isize>> = mask
            .as_ref()
            .map(|m| rnp_core::iter::offsets(&m.shape, &m.strides, m.byte_offset).collect());
        let mut it = offs.iter().enumerate().filter_map(|(k, &o)| {
            if let (Some(m), Some(mo)) = (mask.as_ref(), mofs.as_ref()) {
                if !mask_at(m, mo[k]) {
                    return None;
                }
            }
            Some(element_to_py(py, arr, o))
        });
        let v = fold(&ctx, op, name, seed.clone(), &mut it)?;
        let z = NdArray::zeros(vec![], DType::Object).map_err(crate::err)?;
        objects::write(&z, z.byte_offset, &v);
        z
    } else {
        let mut cur = arr.clone();
        let mut cur_mask = mask.clone();
        for (nth, &ax) in sorted.iter().rev().enumerate() {
            cur = reduce_one_axis(
                &ctx,
                op,
                name,
                &cur,
                ax,
                if nth == 0 { seed.clone() } else { None },
                if nth == 0 { cur_mask.take() } else { None },
            )?;
        }
        cur
    };

    let res = if keepdims {
        let mut shape = res.shape.clone();
        for &ax in sorted.iter() {
            shape.insert(ax, 1);
        }
        res.reshape(&shape).map_err(crate::err)?
    } else {
        res
    };
    store_or_wrap(py, res, out)
}

fn reduce_one_axis<'py>(
    ctx: &Ctx<'py>,
    op: BinObj,
    name: &str,
    arr: &NdArray,
    axis: usize,
    seed: Option<Bound<'py, PyAny>>,
    mask: Option<NdArray>,
) -> PyResult<NdArray> {
    let py = ctx.py;
    let n = arr.shape[axis];
    let mut oshape = arr.shape.clone();
    oshape.remove(axis);
    let out = NdArray::zeros(oshape.clone(), DType::Object).map_err(crate::err)?;
    let oo: Vec<isize> =
        rnp_core::iter::offsets(&out.shape, &out.strides, out.byte_offset).collect();
    // Offsets of the `axis == 0` hyperplane, one per output element.
    let base = arr.slice_axis(axis, 0, 1, 1).remove_axis(axis);
    let bo: Vec<isize> =
        rnp_core::iter::offsets(&base.shape, &base.strides, base.byte_offset).collect();
    let step = arr.strides[axis];
    let mbase = mask.as_ref().map(|m| {
        let b = m.slice_axis(axis, 0, 1, 1).remove_axis(axis);
        let offs: Vec<isize> =
            rnp_core::iter::offsets(&b.shape, &b.strides, b.byte_offset).collect();
        (m.clone(), offs, m.strides[axis])
    });
    for (k, &o) in oo.iter().enumerate() {
        let start = bo[k];
        let mut it = (0..n).filter_map(|i| {
            if let Some((m, offs, mstep)) = &mbase {
                if !mask_at(m, offs[k] + i * mstep) {
                    return None;
                }
            }
            Some(element_to_py(py, arr, start + i * step))
        });
        let v = fold(ctx, op, name, seed.clone(), &mut it)?;
        objects::write(&out, o, &v);
    }
    Ok(out)
}

/// `ufunc.accumulate` over an object array.
pub fn accumulate<'py>(
    py: Python<'py>,
    name: &str,
    arr: &NdArray,
    axis: usize,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let op = match classify(name) {
        Some(ObjOp::Bin(b)) => b,
        Some(ObjOp::Un(_)) => {
            return Err(PyValueError::new_err(format!(
                "accumulate only supported for binary functions ('{name}')"
            )))
        }
        None => return Err(no_loop_error(name)),
    };
    let ctx = Ctx::new(py)?;
    let res = NdArray::zeros(arr.shape.clone(), DType::Object).map_err(crate::err)?;
    let n = arr.shape[axis];
    let base = arr.slice_axis(axis, 0, 1, 1).remove_axis(axis);
    let bo: Vec<isize> =
        rnp_core::iter::offsets(&base.shape, &base.strides, base.byte_offset).collect();
    let rbase = res.slice_axis(axis, 0, 1, 1).remove_axis(axis);
    let ro: Vec<isize> =
        rnp_core::iter::offsets(&rbase.shape, &rbase.strides, rbase.byte_offset).collect();
    let (step, rstep) = (arr.strides[axis], res.strides[axis]);
    for k in 0..bo.len() {
        let mut acc: Option<Bound<'py, PyAny>> = None;
        for i in 0..n {
            let x = element_to_py(py, arr, bo[k] + i * step)?;
            let v = match acc {
                None => x,
                Some(prev) => eval_bin(&ctx, op, &prev, &x)?,
            };
            objects::write(&res, ro[k] + i * rstep, &v);
            acc = Some(v);
        }
    }
    store_or_wrap(py, res, out)
}

/// numpy's `(2,)` / `(2, 3)` shape spelling.
fn fmt_shape(shape: &[isize]) -> String {
    if shape.len() == 1 {
        return format!("({},)", shape[0]);
    }
    format!(
        "({})",
        shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// Silences the unused-import warning when the feature set changes.
#[allow(dead_code)]
fn _unused(_: &Bound<'_, PyTuple>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_covers_the_probed_object_loops() {
        for n in [
            "add", "subtract", "multiply", "divide", "floor_divide", "remainder", "power",
            "bitwise_and", "left_shift", "equal", "less", "maximum", "gcd", "lcm", "hypot",
            "arctan2", "fmod", "logical_and", "logical_or", "logical_xor",
        ] {
            assert!(classify(n).is_some(), "{n} should have an object loop");
        }
        for n in [
            "negative", "positive", "absolute", "invert", "sign", "square", "reciprocal",
            "conjugate", "sqrt", "rint", "floor", "ceil", "trunc", "logical_not", "_ones_like",
        ] {
            assert!(classify(n).is_some(), "{n} should have an object loop");
        }
    }

    #[test]
    fn classify_rejects_the_ufuncs_numpy_gives_no_object_loop() {
        for n in [
            "isnan", "isinf", "isfinite", "signbit", "spacing", "copysign", "nextafter",
            "ldexp", "logaddexp", "logaddexp2", "heaviside", "float_power", "divmod", "frexp",
            "modf",
        ] {
            assert!(classify(n).is_none(), "{n} must have no object loop");
        }
    }

    #[test]
    fn method_loops_are_the_ones_numpy_marks_with_td_p() {
        assert_eq!(classify("sqrt").map(|o| matches!(o, ObjOp::Un(UnObj::Method("sqrt")))), Some(true));
        assert_eq!(
            classify("conj").map(|o| matches!(o, ObjOp::Un(UnObj::Method("conjugate")))),
            Some(true)
        );
        assert_eq!(
            classify("bitwise_count").map(|o| matches!(o, ObjOp::Un(UnObj::Method("bit_count")))),
            Some(true)
        );
        // ... while `negative` is the operator.
        assert!(matches!(classify("negative"), Some(ObjOp::Un(UnObj::Neg))));
    }

    #[test]
    fn no_loop_message_is_numpys() {
        let e = no_loop_error("isnan");
        Python::attach(|py| {
            assert_eq!(
                e.value(py).to_string(),
                "ufunc 'isnan' not supported for the input types, and the inputs could not be \
                 safely coerced to any supported types according to the casting rule ''safe''"
            );
        });
    }
}
