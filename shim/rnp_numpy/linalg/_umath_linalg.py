"""Gufunc-shaped wrappers around the Rust Apple Accelerate LAPACK bridge."""

import _rnp

from .._stubs import ufunc_stub

_ilp64 = True


def _dtype(signature, *arrays):
    if signature:
        inputs = signature.split("->", 1)[0]
        if "D" in inputs:
            return "complex128"
        if "F" in inputs:
            return "complex64"
        if "f" in inputs:
            return "float32"
        return "float64"
    return "complex128" if any(getattr(a.dtype, "kind", "") == "c"
                                for a in arrays) else "float64"


def _linalg_call(call, *args):
    try:
        return call(*args)
    except ValueError as exc:
        from ._linalg import LinAlgError
        raise LinAlgError(str(exc)) from None


def _solve(a, b, *, signature=None, **kwargs):
    return _linalg_call(_rnp._lapack_solve, a, b, False,
                        _dtype(signature, a, b))


def _solve1(a, b, *, signature=None, **kwargs):
    return _linalg_call(_rnp._lapack_solve, a, b, True,
                        _dtype(signature, a, b))


def _inv(a, *, signature=None, **kwargs):
    return _linalg_call(_rnp._lapack_inv, a, _dtype(signature, a))


def _inv_noraise(a, *, signature=None, **kwargs):
    return _rnp._lapack_inv_noraise(a, _dtype(signature, a))


def _det(a, *, signature=None, **kwargs):
    out = _rnp._lapack_det(a, _dtype(signature, a))
    return out[()] if out.ndim == 0 else out


def _slogdet(a, *, signature=None, **kwargs):
    sign, logabs = _rnp._lapack_slogdet(a, _dtype(signature, a))
    if sign.ndim == 0:
        sign = sign[()]
        logabs = logabs[()]
    return sign, logabs


def _cholesky(a, upper, signature=None):
    return _linalg_call(_rnp._lapack_cholesky, a, upper,
                        _dtype(signature, a))


def _lstsq(a, b, rcond, *, signature=None, **kwargs):
    return _linalg_call(_rnp._lapack_lstsq, a, b, float(rcond),
                        _dtype(signature, a, b))


def _complex_signature(signature, a):
    return _dtype(signature, a).startswith("complex")


def _eig(a, vectors, *, signature=None, **kwargs):
    values, vecs = _linalg_call(_rnp._lapack_eig, a, vectors,
                                _complex_signature(signature, a))
    return (values, vecs) if vectors else values


def _eigh(a, upper, vectors, *, signature=None, **kwargs):
    values, vecs = _linalg_call(_rnp._lapack_eigh, a, upper, vectors,
                                _complex_signature(signature, a))
    return (values, vecs) if vectors else values


def _svd(a, full, vectors, *, signature=None, **kwargs):
    u, values, vh = _linalg_call(_rnp._lapack_svd, a, full, vectors,
                                 _complex_signature(signature, a))
    return (u, values, vh) if vectors else values


def _qr_raw(a, *, signature=None, **kwargs):
    return _linalg_call(_rnp._lapack_qr_raw, a,
                        _complex_signature(signature, a))


def _qr_q(a, tau, complete, *, signature=None, **kwargs):
    return _linalg_call(_rnp._lapack_qr_q, a, tau, complete,
                        _complex_signature(signature, a))


def _gufunc(name, signature, nin=1, nout=1, impl=None):
    f = ufunc_stub(f"numpy.linalg._umath_linalg.{name}", nin=nin, nout=nout)
    f.signature = signature
    f._impl = impl
    return f


det = _gufunc("det", "(m,m)->()", impl=_det)
slogdet = _gufunc("slogdet", "(m,m)->(),()", nout=2, impl=_slogdet)
inv = _gufunc("inv", "(m, m)->(m, m)", impl=_inv)
inv_noraise = _gufunc("inv_noraise", "(m, m)->(m, m)", impl=_inv_noraise)
cholesky_lo = _gufunc("cholesky_lo", "(m,m)->(m,m)",
                      impl=lambda a, signature=None, **kw:
                      _cholesky(a, False, signature))
cholesky_up = _gufunc("cholesky_up", "(m,m)->(m,m)",
                      impl=lambda a, signature=None, **kw:
                      _cholesky(a, True, signature))
eig = _gufunc("eig", "(m,m)->(m),(m,m)", nout=2,
              impl=lambda a, **kw: _eig(a, True, **kw))
eigvals = _gufunc("eigvals", "(m,m)->(m)",
                  impl=lambda a, **kw: _eig(a, False, **kw))
eigh_lo = _gufunc("eigh_lo", "(m,m)->(m),(m,m)", nout=2,
                  impl=lambda a, **kw: _eigh(a, False, True, **kw))
eigh_up = _gufunc("eigh_up", "(m,m)->(m),(m,m)", nout=2,
                  impl=lambda a, **kw: _eigh(a, True, True, **kw))
eigvalsh_lo = _gufunc("eigvalsh_lo", "(m,m)->(m)",
                      impl=lambda a, **kw: _eigh(a, False, False, **kw))
eigvalsh_up = _gufunc("eigvalsh_up", "(m,m)->(m)",
                      impl=lambda a, **kw: _eigh(a, True, False, **kw))
solve = _gufunc("solve", "(m,m),(m,n)->(m,n)", nin=2, impl=_solve)
solve1 = _gufunc("solve1", "(m,m),(m)->(m)", nin=2, impl=_solve1)
lstsq = _gufunc("lstsq", "(m,n),(m,k),()->(n,k),(k),(),(m)",
                nin=3, nout=4, impl=_lstsq)
qr_complete = _gufunc("qr_complete", "(m,n),(k)->(m,m)", nin=2,
                      impl=lambda a, tau, **kw: _qr_q(a, tau, True, **kw))
qr_r_raw = _gufunc("qr_r_raw", "(m,n)->(k)", impl=_qr_raw)
qr_reduced = _gufunc("qr_reduced", "(m,n),(k)->(m,k)", nin=2,
                     impl=lambda a, tau, **kw: _qr_q(a, tau, False, **kw))
svd = _gufunc("svd", "(m,n)->(p)",
              impl=lambda a, **kw: _svd(a, False, False, **kw))
svd_f = _gufunc("svd_f", "(m,n)->(m,m),(p),(n,n)", nout=3,
                impl=lambda a, **kw: _svd(a, True, True, **kw))
svd_s = _gufunc("svd_s", "(m,n)->(m,p),(p),(p,n)", nout=3,
                impl=lambda a, **kw: _svd(a, False, True, **kw))


def __getattr__(name):
    return _gufunc(name, None)
