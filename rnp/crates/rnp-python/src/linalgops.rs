//! The Python bridge for the matmul family (`matmul`, `vecdot`, `matvec`,
//! `vecmat`) and the two non-ufunc shapes built on the same kernel,
//! `np.dot` and `np.inner`.
//!
//! Everything shape- and dtype-related lives in `rnp_core::matmul`; this file
//! only converts arguments, hands `out=` back the way numpy does, and
//! reproduces `np.dot`'s stricter `out=` contract (which, unlike a ufunc's,
//! refuses anything that is not already the exact dtype and C-contiguous).

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use rnp_core::matmul::{self, MatKind};
use rnp_core::{DType, NdArray};

use crate::convert::array_from_any;
use crate::pyarray::{store_or_wrap, PyNdArray};
use crate::pydtype::dtype_from_any;

/// The shape of an `out=` argument, or `None` when it was not given.
///
/// numpy lets `out` take part in resolving `matmul`'s optional core
/// dimensions, so the shape has to be known before the kernel runs.
fn out_shape(out: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<isize>>> {
    match out {
        Some(o) if !o.is_none() => {
            let cell = o
                .cast::<PyNdArray>()
                .map_err(|_| PyTypeError::new_err("return arrays must be of ArrayType"))?;
            let shape = cell.borrow().arr.shape.clone();
            Ok(Some(shape))
        }
        _ => Ok(None),
    }
}

/// `matmul` / `vecdot` / `matvec` / `vecmat`, dispatched by name.
#[pyfunction]
#[pyo3(signature = (name, a, b, out = None, dtype = None))]
pub fn _matmul<'py>(
    py: Python<'py>,
    name: &str,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let kind = MatKind::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown gufunc {:?}", name)))?;
    let aa = array_from_any(a, None, false)?;
    let bb = array_from_any(b, None, false)?;
    let dt = match dtype {
        Some(d) if !d.is_none() => Some(dtype_from_any(d)?),
        _ => None,
    };
    let os = out_shape(out)?;
    rnp_core::fpe::clear();
    let res = matmul::matmul(kind, &aa, &bb, os.as_deref(), dt).map_err(crate::err)?;
    crate::ufuncs::report_fpe(py, kind.name())?;
    store_or_wrap(py, res, out)
}

/// The dtype the loop would run in, so the shim can raise numpy's
/// `_UFuncOutputCastingError` before anything is computed.
#[pyfunction]
pub fn _matmul_dtype(name: &str, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<String> {
    let kind = MatKind::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown gufunc {:?}", name)))?;
    let aa = array_from_any(a, None, false)?;
    let bb = array_from_any(b, None, false)?;
    Ok(matmul::result_dtype(kind, aa.dtype(), bb.dtype())
        .map_err(crate::err)?
        .name())
}

/// The resolved dimensions of a call, for the shim's object-dtype path (which
/// has to run the products through Python's own `*` and `+`).
///
/// Returns `(loop_shape, out_shape, rows, inner, cols, a_rowless, b_colless)`.
#[pyfunction]
#[pyo3(signature = (name, a_shape, b_shape, out_shape = None))]
#[allow(clippy::type_complexity)]
pub fn _matmul_plan(
    name: &str,
    a_shape: Vec<isize>,
    b_shape: Vec<isize>,
    out_shape: Option<Vec<isize>>,
) -> PyResult<(Vec<isize>, Vec<isize>, isize, isize, isize, bool, bool)> {
    let kind = MatKind::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown gufunc {:?}", name)))?;
    let p = matmul::plan(kind, &a_shape, &b_shape, out_shape.as_deref()).map_err(crate::err)?;
    Ok((
        p.loop_shape,
        p.out_shape,
        p.rows,
        p.inner,
        p.cols,
        p.a_rowless,
        p.b_colless,
    ))
}

/// `np.dot`. Its `out=` is not a ufunc output: numpy demands the exact result
/// dtype, the exact shape and a C-contiguous buffer, and says so in one
/// message rather than casting.
#[pyfunction]
#[pyo3(signature = (a, b, out = None))]
pub fn _dot<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let aa = array_from_any(a, None, false)?;
    let bb = array_from_any(b, None, false)?;
    rnp_core::fpe::clear();
    let res = matmul::dot(&aa, &bb).map_err(crate::err)?;
    crate::ufuncs::report_fpe(py, "dot")?;
    finish_strict(py, res, out)
}

/// `np.inner`, contracting the last axis of both operands. numpy's `inner`
/// takes no `out=` at all, so neither does this.
#[pyfunction]
pub fn _inner<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let aa = array_from_any(a, None, false)?;
    let bb = array_from_any(b, None, false)?;
    rnp_core::fpe::clear();
    let res = matmul::inner(&aa, &bb).map_err(crate::err)?;
    crate::ufuncs::report_fpe(py, "inner")?;
    store_or_wrap(py, res, None)
}

/// Store into `dot`'s `out=`, with numpy's all-or-nothing check.
///
/// numpy distinguishes exactly two failures here: an output of the right rank
/// but the wrong extents gets "output array has wrong dimensions", and
/// everything else (wrong rank, wrong dtype, not C-contiguous, read-only)
/// gets the single "not acceptable" message.
fn finish_strict<'py>(
    py: Python<'py>,
    res: NdArray,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let dest = match out {
        Some(o) if !o.is_none() => o,
        _ => return store_or_wrap(py, res, None),
    };
    let cell = dest
        .cast::<PyNdArray>()
        .map_err(|_| PyValueError::new_err("output must be an array"))?;
    let (acceptable, same_shape) = {
        let target = &cell.borrow().arr;
        (
            target.dtype() == res.dtype()
                && target.ndim() == res.ndim()
                && target.is_c_contiguous()
                && target.flags.writeable,
            target.shape == res.shape,
        )
    };
    if !acceptable {
        return Err(PyValueError::new_err(
            "output array is not acceptable (must have the right datatype, \
             number of dimensions, and be a C-Array)",
        ));
    }
    if !same_shape {
        return Err(PyValueError::new_err("output array has wrong dimensions"));
    }
    store_or_wrap(py, res, out)
}

/// `a @ b` / `b @ a`. numpy's `ndarray.__matmul__` is `np.matmul`, including
/// its refusal of 0-d operands and its scalar (not 0-d array) return.
pub fn matmul_operator(
    py: Python<'_>,
    me: &NdArray,
    other: &Bound<'_, PyAny>,
    reflected: bool,
) -> PyResult<Py<PyAny>> {
    let rhs = match array_from_any(other, None, false) {
        Ok(a) => a,
        Err(_) => return Ok(py.NotImplemented()),
    };
    let (a, b) = if reflected { (&rhs, me) } else { (me, &rhs) };
    rnp_core::fpe::clear();
    let res = matmul::matmul(MatKind::MatMul, a, b, None, None).map_err(crate::err)?;
    crate::ufuncs::report_fpe(py, "matmul")?;
    if res.ndim() == 0 {
        // Every ufunc hands back a numpy scalar rather than a 0-d array.
        let dt = res.dtype();
        return Ok(crate::convert::npscalar_to_py(py, dt, res.get_flat(0))?.unbind());
    }
    Ok(PyNdArray::into_py_any(res, py)?.into_any())
}

/// `a @= b`. numpy computes `a @ b` and writes it back, which only works when
/// the product has `a`'s own shape.
pub fn imatmul(slf: &Bound<'_, PyNdArray>, other: &Bound<'_, PyAny>) -> PyResult<()> {
    let me = slf.borrow().arr.clone();
    if !me.flags.writeable {
        return Err(PyValueError::new_err("output array is read-only"));
    }
    let rhs = array_from_any(other, None, false)?;
    rnp_core::fpe::clear();
    let res = matmul::matmul(MatKind::MatMul, &me, &rhs, Some(&me.shape), None)
        .map_err(crate::err)?;
    crate::ufuncs::report_fpe(slf.py(), "matmul")?;
    if res.shape != me.shape {
        return Err(PyValueError::new_err(format!(
            "output array has wrong dimensions: {:?} instead of {:?}",
            res.shape, me.shape
        )));
    }
    let src: Vec<isize> =
        rnp_core::iter::offsets(&res.shape, &res.strides, res.byte_offset).collect();
    let dst: Vec<isize> =
        rnp_core::iter::offsets(&me.shape, &me.strides, me.byte_offset).collect();
    for (&s, &d) in src.iter().zip(dst.iter()) {
        me.write_at(d, res.read_at(s));
    }
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(_matmul_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(_matmul_plan, m)?)?;
    m.add_function(wrap_pyfunction!(_dot, m)?)?;
    m.add_function(wrap_pyfunction!(_inner, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_solve, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_inv, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_inv_noraise, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_det, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_slogdet, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_lstsq, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_svd, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_eig, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_eigh, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_qr_raw, m)?)?;
    m.add_function(wrap_pyfunction!(_lapack_qr_q, m)?)?;
    Ok(())
}

fn lapack_dtype(name: &str) -> PyResult<DType> {
    match name {
        "float32" => Ok(DType::F32),
        "float64" => Ok(DType::F64),
        "complex64" => Ok(DType::C64),
        "complex128" => Ok(DType::C128),
        _ => Err(PyTypeError::new_err(format!("unsupported LAPACK dtype {name:?}"))),
    }
}

#[pyfunction]
fn _lapack_solve(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    vector: bool,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let aa = array_from_any(a, None, false)?;
    let bb = array_from_any(b, None, false)?;
    let out = rnp_core::lapack::solve(&aa, &bb, vector, lapack_dtype(dtype)?)
        .map_err(crate::err)?;
    PyNdArray::into_py_any(out, py).map(|value| value.into_any())
}

#[pyfunction]
fn _lapack_inv(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let aa = array_from_any(a, None, false)?;
    let out = rnp_core::lapack::inv(&aa, lapack_dtype(dtype)?, false).map_err(crate::err)?;
    PyNdArray::into_py_any(out, py).map(|value| value.into_any())
}

#[pyfunction]
fn _lapack_inv_noraise(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let aa = array_from_any(a, None, false)?;
    let out = rnp_core::lapack::inv(&aa, lapack_dtype(dtype)?, true).map_err(crate::err)?;
    PyNdArray::into_py_any(out, py).map(|value| value.into_any())
}

#[pyfunction]
fn _lapack_det(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let aa = array_from_any(a, None, false)?;
    let out = rnp_core::lapack::det(&aa, lapack_dtype(dtype)?).map_err(crate::err)?;
    PyNdArray::into_py_any(out, py).map(|value| value.into_any())
}

#[pyfunction]
fn _lapack_slogdet(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    dtype: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let aa = array_from_any(a, None, false)?;
    let (sign, logabs) =
        rnp_core::lapack::slogdet(&aa, lapack_dtype(dtype)?).map_err(crate::err)?;
    Ok((PyNdArray::into_py_any(sign, py)?.into_any(),
        PyNdArray::into_py_any(logabs, py)?.into_any()))
}

#[pyfunction]
fn _lapack_cholesky(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    upper: bool,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let aa = array_from_any(a, None, false)?;
    let out = rnp_core::lapack::cholesky(&aa, upper, lapack_dtype(dtype)?)
        .map_err(crate::err)?;
    PyNdArray::into_py_any(out, py).map(|value| value.into_any())
}

#[pyfunction]
fn _lapack_lstsq(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    rcond: f64,
    dtype: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>, i64, Py<PyAny>)> {
    let aa = array_from_any(a, None, false)?;
    let bb = array_from_any(b, None, false)?;
    let out = rnp_core::lapack::lstsq(&aa, &bb, rcond, lapack_dtype(dtype)?)
        .map_err(crate::err)?;
    Ok((
        PyNdArray::into_py_any(out.x, py)?.into_any(),
        PyNdArray::into_py_any(out.residuals, py)?.into_any(),
        out.rank,
        PyNdArray::into_py_any(out.singular_values, py)?.into_any(),
    ))
}

#[pyfunction]
fn _lapack_svd(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    full: bool,
    vectors: bool,
    complex: bool,
) -> PyResult<(Option<Py<PyAny>>, Py<PyAny>, Option<Py<PyAny>>)> {
    let aa = array_from_any(a, None, false)?;
    let out = rnp_core::lapack::svd(&aa, full, vectors, complex).map_err(crate::err)?;
    Ok((
        out.u.map(|value| PyNdArray::into_py_any(value, py).map(|v| v.into_any())).transpose()?,
        PyNdArray::into_py_any(out.singular_values, py)?.into_any(),
        out.vh.map(|value| PyNdArray::into_py_any(value, py).map(|v| v.into_any())).transpose()?,
    ))
}

#[pyfunction]
fn _lapack_eig(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    vectors: bool,
    complex: bool,
) -> PyResult<(Py<PyAny>, Option<Py<PyAny>>)> {
    let aa = array_from_any(a, None, false)?;
    let out = rnp_core::lapack::eig(&aa, vectors, complex).map_err(crate::err)?;
    Ok((
        PyNdArray::into_py_any(out.values, py)?.into_any(),
        out.vectors.map(|value| PyNdArray::into_py_any(value, py).map(|v| v.into_any())).transpose()?,
    ))
}

#[pyfunction]
fn _lapack_eigh(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    upper: bool,
    vectors: bool,
    complex: bool,
) -> PyResult<(Py<PyAny>, Option<Py<PyAny>>)> {
    let aa = array_from_any(a, None, false)?;
    let out = rnp_core::lapack::eigh(&aa, upper, vectors, complex).map_err(crate::err)?;
    Ok((
        PyNdArray::into_py_any(out.values, py)?.into_any(),
        out.vectors.map(|value| PyNdArray::into_py_any(value, py).map(|v| v.into_any())).transpose()?,
    ))
}

#[pyfunction]
fn _lapack_qr_raw(py:Python<'_>,a:&Bound<'_,PyAny>,complex:bool)->PyResult<Py<PyAny>>{
    let aa=array_from_any(a,None,false)?;
    let out=rnp_core::lapack::qr_raw(&aa,complex).map_err(crate::err)?;
    PyNdArray::into_py_any(out,py).map(|value|value.into_any())
}

#[pyfunction]
fn _lapack_qr_q(py:Python<'_>,a:&Bound<'_,PyAny>,tau:&Bound<'_,PyAny>,complete:bool,complex:bool)->PyResult<Py<PyAny>>{
    let aa=array_from_any(a,None,false)?;let tt=array_from_any(tau,None,false)?;
    let out=rnp_core::lapack::qr_q(&aa,&tt,complete,complex).map_err(crate::err)?;
    PyNdArray::into_py_any(out,py).map(|value|value.into_any())
}
