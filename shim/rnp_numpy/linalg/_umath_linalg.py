"""Stand-in for numpy.linalg._umath_linalg (the LAPACK-backed gufunc module).

The port has no LAPACK binding, so every entry is a loud gufunc stub carrying
the real signature (upstream tests introspect `signature`/`nin`/`nout`).
"""

from .._stubs import ufunc_stub

_ilp64 = False


def _gufunc(name, signature, nin=1, nout=1):
    f = ufunc_stub(f"numpy.linalg._umath_linalg.{name}", nin=nin, nout=nout)
    f.signature = signature
    return f


det = _gufunc("det", "(m,m)->()")
slogdet = _gufunc("slogdet", "(m,m)->(),()", nout=2)
inv = _gufunc("inv", "(m, m)->(m, m)")
cholesky_lo = _gufunc("cholesky_lo", "(m,m)->(m,m)")
cholesky_up = _gufunc("cholesky_up", "(m,m)->(m,m)")
eig = _gufunc("eig", "(m,m)->(m),(m,m)", nout=2)
eigvals = _gufunc("eigvals", "(m,m)->(m)")
eigh_lo = _gufunc("eigh_lo", "(m,m)->(m),(m,m)", nout=2)
eigh_up = _gufunc("eigh_up", "(m,m)->(m),(m,m)", nout=2)
eigvalsh_lo = _gufunc("eigvalsh_lo", "(m,m)->(m)")
eigvalsh_up = _gufunc("eigvalsh_up", "(m,m)->(m)")
solve = _gufunc("solve", "(m,m),(m,n)->(m,n)", nin=2)
solve1 = _gufunc("solve1", "(m,m),(m)->(m)", nin=2)
lstsq = _gufunc("lstsq", "(m,n),(m,k),()->(n,k),(k),(),(m)", nin=3, nout=4)
qr_complete = _gufunc("qr_complete", "(m,n),(k)->(m,m)", nin=2)
qr_r_raw = _gufunc("qr_r_raw", "(m,n)->(k)")
qr_reduced = _gufunc("qr_reduced", "(m,n),(k)->(m,k)", nin=2)
svd = _gufunc("svd", "(m,n)->(p)")
svd_f = _gufunc("svd_f", "(m,n)->(m,m),(p),(n,n)", nout=3)
svd_s = _gufunc("svd_s", "(m,n)->(m,p),(p),(p,n)", nout=3)


def __getattr__(name):
    return _gufunc(name, None)
