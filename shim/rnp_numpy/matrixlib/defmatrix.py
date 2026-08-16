"""`numpy.matrixlib.defmatrix` — placeholder.

`np.matrix` is not ported.  This module exists because upstream's
`numpy/lib/_shape_base_impl.py` and `numpy/lib/_index_tricks_impl.py` import
`matrix`/`bmat` at *module* scope, so without the name every test file that
touches `numpy.lib` fails to collect and scores zero.  Constructing a matrix
raises, which is the honest answer; importing one does not.
"""
from .._stubs import inert_class, not_implemented

__all__ = ['matrix', 'bmat', 'asmatrix']

matrix = inert_class("matrix")
bmat = not_implemented("numpy.bmat")
asmatrix = not_implemented("numpy.asmatrix")
