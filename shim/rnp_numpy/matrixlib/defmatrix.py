"""Two-dimensional ``matrix`` compatibility built on the Rust ndarray."""

import ast
import inspect

import rnp_numpy as np

__all__ = ['matrix', 'bmat', 'asmatrix']


def _convert_from_string(data):
    for char in '[]':
        data = data.replace(char, '')
    rows = []
    width = None
    for row in data.split(';'):
        values = []
        for column in row.split(','):
            values.extend(ast.literal_eval(value) for value in column.split())
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError("Rows not the same size.")
        rows.append(values)
    return rows


def _as_base(value):
    if isinstance(value, matrix):
        return np.ndarray.view(value, np.ndarray)
    if isinstance(value, list):
        return [_as_base(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_as_base(item) for item in value)
    if isinstance(value, dict):
        return {key: _as_base(item) for key, item in value.items()}
    return value


def _matrix_result(value, axis=None):
    if isinstance(value, tuple):
        return tuple(_matrix_result(item, axis=axis) for item in value)
    if not isinstance(value, np.ndarray):
        return value
    if value.ndim == 0:
        return value[()]
    if value.ndim > 2:
        collapsed = tuple(size for size in value.shape if size > 1)
        if len(collapsed) > 2:
            raise ValueError("shape too large to be a matrix.")
        value = value.reshape(collapsed or (1, 1))
    if value.ndim == 1:
        value = value.reshape((-1, 1) if axis == 1 else (1, -1))
    return np._ndarray_view_base(value, matrix)


class matrix(np.ndarray):
    """A NumPy-compatible array subclass that remains two-dimensional."""

    __array_priority__ = 10.0

    def __new__(cls, data, dtype=None, copy=True):
        if isinstance(data, cls):
            if dtype is None and not copy:
                return data
            data = np.ndarray.view(data, np.ndarray)
        elif isinstance(data, str):
            data = _convert_from_string(data)

        copy_arg = None if not copy else True
        arr = np.array(data, dtype=dtype, copy=copy_arg, subok=False)
        if arr.ndim > 2:
            raise ValueError("shape too large to be a matrix.")
        if arr.ndim == 0:
            arr = arr.reshape((1, 1))
        elif arr.ndim == 1:
            arr = arr.reshape((1, arr.shape[0]))
        if copy:
            arr = arr.copy()
        return np._ndarray_view_base(arr, cls)

    def __array_finalize__(self, obj):
        self._getitem = False

    def __dir__(self):
        return [name for name in super().__dir__() if name != 'tostring']

    def __array__(self, dtype=None, copy=None):
        result = np.ndarray.view(self, np.ndarray)
        if dtype is not None and result.dtype != np.dtype(dtype):
            result = result.astype(dtype, subok=False, copy=True)
        elif copy:
            result = result.copy()
        return result

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        outputs = kwargs.pop('out', None)
        base_inputs = tuple(_as_base(item) for item in inputs)
        implementation = ufunc if method == '__call__' else getattr(ufunc, method)
        result = implementation(*base_inputs, **kwargs)
        if isinstance(result, np.ndarray) and result.ndim > 2:
            result = np._ndarray_view_base(result, matrix)
        else:
            result = _matrix_result(result)
        if outputs is None:
            return result
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        results = result if isinstance(result, tuple) else (result,)
        for output, value in zip(outputs, results):
            if output is not None:
                output[...] = value
        return outputs[0] if len(outputs) == 1 else outputs

    def __array_function__(self, func, types, args, kwargs):
        name = getattr(func, '__name__', '')
        reductions = {
            'sum', 'prod', 'mean', 'std', 'var', 'max', 'min', 'ptp',
            'any', 'all', 'argmax', 'argmin',
        }
        if name in reductions and args and args[0] is self:
            return getattr(self, name)(*args[1:], **kwargs)
        if name == 'transpose' and args and args[0] is self:
            return self.transpose(*args[1:], **kwargs)
        if name == 'reshape' and args and args[0] is self:
            return self.reshape(*args[1:], **kwargs)
        if name == 'ravel' and args and args[0] is self:
            order = kwargs.get('order', 'C')
            return self._ravel_base(order)

        implementation = getattr(func, '_implementation', func)
        result = implementation(
            *tuple(_as_base(item) for item in args),
            **{key: _as_base(value) for key, value in kwargs.items()},
        )
        return _matrix_result(result, axis=kwargs.get('axis'))

    def __getitem__(self, index):
        result = self.A[index]
        if not isinstance(result, np.ndarray):
            return result
        if result.ndim == 0:
            return result[()]
        if result.ndim == 1:
            try:
                column = len(index) > 1 and np.isscalar(index[1])
            except TypeError:
                column = False
            result = result.reshape((-1, 1) if column else (1, -1))
        return np._ndarray_view_base(result, matrix)

    def __setitem__(self, index, value):
        target = self.A[index]
        value = _as_base(value)
        if (isinstance(value, np.ndarray) and target.ndim == 1 and
                value.size == target.size):
            value = value.reshape(target.shape)
        self.A[index] = value

    def _reduce(self, name, axis=None, dtype=None, out=None, **kwargs):
        base = np.ndarray.view(self, np.ndarray)
        method = getattr(base, name)
        call = {'axis': axis, 'out': _as_base(out)}
        if dtype is not None:
            call['dtype'] = dtype
        call.update(kwargs)
        if axis is not None:
            call['keepdims'] = True
        result = method(**call)
        if out is not None:
            return out
        return _matrix_result(result, axis=axis)

    def sum(self, axis=None, dtype=None, out=None):
        return self._reduce('sum', axis, dtype, out)

    def prod(self, axis=None, dtype=None, out=None):
        return self._reduce('prod', axis, dtype, out)

    def mean(self, axis=None, dtype=None, out=None):
        return self._reduce('mean', axis, dtype, out)

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        return self._reduce('std', axis, dtype, out, ddof=ddof)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        return self._reduce('var', axis, dtype, out, ddof=ddof)

    def max(self, axis=None, out=None):
        return self._reduce('max', axis, out=out)

    def min(self, axis=None, out=None):
        return self._reduce('min', axis, out=out)

    def ptp(self, axis=None, out=None):
        return self._reduce('ptp', axis, out=out)

    def any(self, axis=None, out=None, keepdims=False):
        return self._reduce('any', axis, out=out)

    def all(self, axis=None, out=None, keepdims=False):
        return self._reduce('all', axis, out=out)

    def argmax(self, axis=None, out=None):
        return self._reduce('argmax', axis, out=out)

    def argmin(self, axis=None, out=None):
        return self._reduce('argmin', axis, out=out)

    def argsort(self, axis=-1, kind=None, order=None, *, stable=None):
        result = self.A.argsort(axis=axis, kind=kind, order=order, stable=stable)
        return _matrix_result(result)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        result = self.A.diagonal(offset=offset, axis1=axis1, axis2=axis2)
        return _matrix_result(result)

    def cumprod(self, axis=None, dtype=None, out=None):
        result = self.A.cumprod(axis=axis, dtype=dtype, out=_as_base(out))
        return out if out is not None else _matrix_result(result)

    def cumsum(self, axis=None, dtype=None, out=None):
        result = self.A.cumsum(axis=axis, dtype=dtype, out=_as_base(out))
        return out if out is not None else _matrix_result(result)

    def copy(self, order='C'):
        return matrix(np.ndarray.view(self, np.ndarray).copy(order=order), copy=False)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        if subok and not copy and np.dtype(dtype) == self.dtype:
            return self
        result = np.ndarray.view(self, np.ndarray).astype(
            dtype, order=order, casting=casting, subok=False, copy=copy)
        return matrix(result, copy=False) if subok else result

    def reshape(self, *shape, order='C'):
        base = np.ndarray.view(self, np.ndarray)
        result = base.reshape(*shape) if order == 'C' else np.reshape(base, shape, order=order)
        if result.ndim > 2:
            return np._ndarray_view_base(result, matrix)
        return _matrix_result(result)

    def _ravel_base(self, order='C'):
        base = np.ndarray.view(self, np.ndarray)
        if order == 'F':
            return base.transpose().ravel()
        if order == 'A' and base.flags.f_contiguous and not base.flags.c_contiguous:
            return base.transpose().ravel()
        return base.ravel()

    def transpose(self, *axes):
        result = np.ndarray.view(self, np.ndarray).transpose(*axes)
        return _matrix_result(result)

    def ravel(self, order='C'):
        return matrix(self._ravel_base(order), copy=False)

    def flatten(self, order='C'):
        return matrix(self.ravel(order).A.copy(), copy=False)

    def squeeze(self, axis=None):
        base = np.ndarray.view(self, np.ndarray)
        result = base.squeeze() if axis is None else base.squeeze(axis=axis)
        return matrix(result, copy=False)

    def tolist(self):
        return np.ndarray.view(self, np.ndarray).tolist()

    def byteswap(self, inplace=False):
        return _matrix_result(self.A.byteswap(inplace=inplace))

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        result = self.A.trace(
            offset=offset, axis1=axis1, axis2=axis2,
            dtype=dtype, out=_as_base(out))
        return out if out is not None else matrix(result, copy=False)

    def clip(self, min=None, max=None, out=None, **kwargs):
        result = self.A.clip(min, max, out=_as_base(out), **kwargs)
        return out if out is not None else _matrix_result(result)

    def compress(self, condition, axis=None, out=None):
        result = self.A.compress(condition, axis=axis, out=_as_base(out))
        return out if out is not None else _matrix_result(result, axis=axis)

    def repeat(self, repeats, axis=None):
        return _matrix_result(self.A.repeat(repeats, axis=axis), axis=axis)

    def swapaxes(self, axis1, axis2):
        return _matrix_result(self.A.swapaxes(axis1, axis2))

    def dot(self, other, out=None):
        result = np.dot(self.A, _as_base(other), out=_as_base(out))
        return out if out is not None else _matrix_result(result)

    def _elementwise(self, operation, *inputs):
        values = [_as_base(value) for value in inputs]
        return _matrix_result(operation(*values))

    def __add__(self, other):
        return self._elementwise(np.add, self, other)

    def __radd__(self, other):
        return self._elementwise(np.add, other, self)

    def __sub__(self, other):
        return self._elementwise(np.subtract, self, other)

    def __rsub__(self, other):
        return self._elementwise(np.subtract, other, self)

    def __truediv__(self, other):
        return self._elementwise(np.true_divide, self, other)

    def __rtruediv__(self, other):
        return self._elementwise(np.true_divide, other, self)

    def __neg__(self):
        return self._elementwise(np.negative, self)

    def __pos__(self):
        return self._elementwise(np.positive, self)

    def __invert__(self):
        return self._elementwise(np.invert, self)

    def __abs__(self):
        return self._elementwise(np.absolute, self)

    def __mul__(self, other):
        if isinstance(other, (np.ndarray, list, tuple)):
            return matrix(np.dot(self.A, _as_base(asmatrix(other))), copy=False)
        if np.isscalar(other):
            return _matrix_result(np.multiply(self.A, other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (np.ndarray, list, tuple)):
            return matrix(np.dot(_as_base(other), self.A), copy=False)
        return _matrix_result(np.multiply(other, self.A))

    def __imul__(self, other):
        self[...] = self * other
        return self

    def __matmul__(self, other):
        return matrix(np.matmul(self.A, _as_base(other)), copy=False)

    def __rmatmul__(self, other):
        return matrix(np.matmul(_as_base(other), self.A), copy=False)

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, np.integer)):
            return NotImplemented
        if self.shape[0] != self.shape[1]:
            raise np.linalg.LinAlgError("Last 2 dimensions of the array must be square")
        base = self.A
        if exponent < 0:
            base = np.linalg.inv(base)
            exponent = -exponent
        result = np.eye(self.shape[0], dtype=self.dtype)
        while exponent:
            if exponent & 1:
                result = np.dot(result, base)
            exponent >>= 1
            if exponent:
                base = np.dot(base, base)
        return matrix(result, copy=False)

    def __ipow__(self, exponent):
        self[...] = self ** exponent
        return self

    def __rpow__(self, other):
        return NotImplemented

    @property
    def A(self):
        return np.ndarray.view(self, np.ndarray)

    @property
    def real(self):
        return matrix(self.A.real, copy=False)

    @property
    def imag(self):
        return matrix(self.A.imag, copy=False)

    @property
    def A1(self):
        return self.A.ravel()

    @property
    def I(self):
        function = np.linalg.inv if self.shape[0] == self.shape[1] else np.linalg.pinv
        return matrix(function(self.A), copy=False)

    @property
    def T(self):
        return self.transpose()

    @property
    def H(self):
        return matrix(self.A.conjugate().transpose(), copy=False)

    getA = A.fget
    getA1 = A1.fget
    getI = I.fget
    getT = T.fget
    getH = H.fget

    def __repr__(self):
        body = repr(self.A)
        if body.startswith('array(') and body.endswith(')'):
            body = body[6:-1]
        body = body.replace('\n       ', '\n        ')
        return f"matrix({body})"


def asmatrix(data, dtype=None):
    return matrix(data, dtype=dtype, copy=False)


def _from_string(specification, global_dict, local_dict):
    rows = []
    for row in specification.split(';'):
        columns = []
        for name in row.replace(',', ' ').split():
            try:
                value = local_dict[name]
            except KeyError:
                try:
                    value = global_dict[name]
                except KeyError:
                    raise NameError(f"name {name!r} is not defined") from None
            columns.append(value)
        rows.append(np.concatenate([_as_base(value) for value in columns], axis=-1))
    return np.concatenate(rows, axis=0)


def bmat(obj, ldict=None, gdict=None):
    if isinstance(obj, str):
        if gdict is None:
            frame = inspect.currentframe().f_back
            global_dict = frame.f_globals
            local_dict = frame.f_locals
        else:
            global_dict = gdict
            local_dict = ldict
        return matrix(_from_string(obj, global_dict, local_dict), copy=False)

    if isinstance(obj, (tuple, list)):
        rows = []
        for row in obj:
            if isinstance(row, np.ndarray):
                values = [_as_base(item) for item in obj]
                return matrix(np.concatenate(values, axis=-1), copy=False)
            rows.append(np.concatenate([_as_base(item) for item in row], axis=-1))
        return matrix(np.concatenate(rows, axis=0), copy=False)
    if isinstance(obj, np.ndarray):
        return matrix(obj, copy=False)
    raise TypeError("bmat expects a string or array-like input")
