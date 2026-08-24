"""Scalar distribution kernels transcribed from NumPy distributions.c."""

import ctypes
import math
import struct

from ._ziggurat import (
    FE_DOUBLE,
    FE_FLOAT,
    FI_DOUBLE,
    FI_FLOAT,
    KE_DOUBLE,
    KE_FLOAT,
    KI_DOUBLE,
    KI_FLOAT,
    WE_DOUBLE,
    WE_FLOAT,
    WI_DOUBLE,
    WI_FLOAT,
    ZIGGURAT_EXP_R,
    ZIGGURAT_EXP_R_F,
    ZIGGURAT_NOR_INV_R,
    ZIGGURAT_NOR_INV_R_F,
    ZIGGURAT_NOR_R,
    ZIGGURAT_NOR_R_F,
)


def _f32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]


_libm = ctypes.CDLL(None)
for _name, _argc in (("log1pf", 1), ("logf", 1), ("expf", 1),
                     ("sqrtf", 1), ("powf", 2), ("fmaf", 3)):
    _func = getattr(_libm, _name)
    _func.argtypes = [ctypes.c_float] * _argc
    _func.restype = ctypes.c_float


def _unary_f(name, value):
    return float(getattr(_libm, name)(ctypes.c_float(value)))


def _powf(left, right):
    return float(_libm.powf(ctypes.c_float(left), ctypes.c_float(right)))


def _fmaf(left, right, addend):
    return float(_libm.fmaf(ctypes.c_float(left), ctypes.c_float(right),
                            ctypes.c_float(addend)))


_WI_FLOAT = tuple(_f32(v) for v in WI_FLOAT)
_FI_FLOAT = tuple(_f32(v) for v in FI_FLOAT)
_WE_FLOAT = tuple(_f32(v) for v in WE_FLOAT)
_FE_FLOAT = tuple(_f32(v) for v in FE_FLOAT)
_NOR_R_F = _f32(ZIGGURAT_NOR_R_F)
_NOR_INV_R_F = _f32(ZIGGURAT_NOR_INV_R_F)
_EXP_R_F = _f32(ZIGGURAT_EXP_R_F)


class DistributionKernels:
    """Exact scalar kernels over one NumPy-compatible BitGenerator."""

    def __init__(self, bit_generator):
        self.bit_generator = bit_generator

    def _next_float(self):
        return _f32((self.bit_generator.next_uint32() >> 8) * _f32(1.0 / 16777216.0))

    def standard_exponential_f(self):
        ri = self.bit_generator.next_uint32() >> 1
        idx = ri & 0xFF
        ri >>= 8
        x = _f32(ri * _WE_FLOAT[idx])
        if ri < KE_FLOAT[idx]:
            return x
        if idx == 0:
            return _f32(_EXP_R_F - _unary_f("log1pf", -self._next_float()))
        delta = _f32(_FE_FLOAT[idx - 1] - _FE_FLOAT[idx])
        if _fmaf(delta, self._next_float(), _FE_FLOAT[idx]) < _unary_f("expf", -x):
            return x
        return self.standard_exponential_f()

    def standard_normal_f(self):
        while True:
            r = self.bit_generator.next_uint32()
            idx = r & 0xFF
            sign = (r >> 8) & 1
            rabs = (r >> 9) & 0x007FFFFF
            x = _f32(rabs * _WI_FLOAT[idx])
            if sign:
                x = _f32(-x)
            if rabs < KI_FLOAT[idx]:
                return x
            if idx == 0:
                while True:
                    xx = _f32(-_NOR_INV_R_F * _unary_f("log1pf", -self._next_float()))
                    yy = _f32(-_unary_f("log1pf", -self._next_float()))
                    if _f32(yy + yy) > _f32(xx * xx):
                        tail = _f32(_NOR_R_F + xx)
                        return _f32(-tail) if ((rabs >> 8) & 1) else tail
            else:
                delta = _f32(_FI_FLOAT[idx - 1] - _FI_FLOAT[idx])
                rhs_arg = _f32(_f32(-0.5 * x) * x)
                if _fmaf(delta, self._next_float(), _FI_FLOAT[idx]) < _unary_f("expf", rhs_arg):
                    return x

    def standard_gamma_f(self, shape):
        shape = _f32(shape)
        if shape == _f32(1.0):
            return self.standard_exponential_f()
        if shape == 0.0:
            return _f32(0.0)
        if shape < 1.0:
            while True:
                u = self._next_float()
                v = self.standard_exponential_f()
                if u <= _f32(1.0 - shape):
                    x = _powf(u, _f32(1.0 / shape))
                    if x <= v:
                        return x
                else:
                    y = _f32(-_unary_f("logf", _f32(_f32(1.0 - u) / shape)))
                    x = _powf(_fmaf(shape, y, _f32(1.0 - shape)),
                              _f32(1.0 / shape))
                    if x <= _f32(v + y):
                        return x
        b = _f32(shape - _f32(1.0 / 3.0))
        c = _f32(1.0 / _unary_f("sqrtf", _f32(9.0 * b)))
        while True:
            while True:
                x = self.standard_normal_f()
                v = _fmaf(c, x, 1.0)
                if v > 0.0:
                    break
            v = _f32(_f32(v * v) * v)
            u = self._next_float()
            x2 = _f32(x * x)
            if u < _fmaf(-_f32(0.0331), _f32(x2 * x2), 1.0):
                return _f32(b * v)
            inner = _f32(_f32(1.0 - v) + _unary_f("logf", v))
            rhs = _fmaf(b, inner, _f32(_f32(0.5 * x) * x))
            if _unary_f("logf", u) < rhs:
                return _f32(b * v)

    def standard_exponential(self):
        ri = self.bit_generator.next_uint64() >> 3
        idx = ri & 0xFF
        ri >>= 8
        x = ri * WE_DOUBLE[idx]
        if ri < KE_DOUBLE[idx]:
            return x
        return self._standard_exponential_unlikely(idx, x)

    def _standard_exponential_unlikely(self, idx, x):
        if idx == 0:
            return ZIGGURAT_EXP_R - math.log1p(-self.bit_generator.next_double())
        if (math.fma(FE_DOUBLE[idx - 1] - FE_DOUBLE[idx],
                self.bit_generator.next_double(), FE_DOUBLE[idx])
                < math.exp(-x)):
            return x
        return self.standard_exponential()

    def standard_normal(self):
        while True:
            r = self.bit_generator.next_uint64()
            idx = r & 0xFF
            r >>= 8
            sign = r & 1
            rabs = (r >> 1) & 0x000FFFFFFFFFFFFF
            x = rabs * WI_DOUBLE[idx]
            if sign:
                x = -x
            if rabs < KI_DOUBLE[idx]:
                return x
            if idx == 0:
                while True:
                    xx = -ZIGGURAT_NOR_INV_R * math.log1p(
                        -self.bit_generator.next_double()
                    )
                    yy = -math.log1p(-self.bit_generator.next_double())
                    if yy + yy > xx * xx:
                        return (-(ZIGGURAT_NOR_R + xx)
                                if ((rabs >> 8) & 1)
                                else ZIGGURAT_NOR_R + xx)
            elif (math.fma(FI_DOUBLE[idx - 1] - FI_DOUBLE[idx],
                  self.bit_generator.next_double(), FI_DOUBLE[idx])
                  < math.exp(-0.5 * x * x)):
                return x

    def standard_gamma(self, shape):
        if shape == 1.0:
            return self.standard_exponential()
        if shape == 0.0:
            return 0.0
        if shape < 1.0:
            while True:
                u = self.bit_generator.next_double()
                v = self.standard_exponential()
                if u <= 1.0 - shape:
                    x = math.pow(u, 1.0 / shape)
                    if x <= v:
                        return x
                else:
                    y = -math.log((1.0 - u) / shape)
                    x = math.pow(math.fma(shape, y, 1.0 - shape), 1.0 / shape)
                    if x <= v + y:
                        return x
        b = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * b)
        while True:
            while True:
                x = self.standard_normal()
                v = math.fma(c, x, 1.0)
                if v > 0.0:
                    break
            v = v * v * v
            u = self.bit_generator.next_double()
            x2 = x * x
            if u < math.fma(-0.0331, x2 * x2, 1.0):
                return b * v
            rhs = math.fma(b, 1.0 - v + math.log(v), 0.5 * x2)
            if math.log(u) < rhs:
                return b * v

    def beta(self, a, b):
        # NumPy uses Johnk's algorithm for a,b <= 1 and gamma otherwise.
        if a <= 1.0 and b <= 1.0:
            if a < 3e-103 and b < 3e-103:
                return float((a + b) * self.bit_generator.next_double() < a)
            while True:
                u = self.bit_generator.next_double()
                v = self.bit_generator.next_double()
                x = math.pow(u, 1.0 / a)
                y = math.pow(v, 1.0 / b)
                xpy = x + y
                if xpy <= 1.0 and u + v > 0.0:
                    if x > 0.0 and y > 0.0:
                        return x / xpy
                    log_x = math.log(u) / a
                    log_y = math.log(v) / b
                    delta = log_x - log_y
                    if delta > 0.0:
                        return math.exp(-math.log1p(math.exp(-delta)))
                    return math.exp(delta - math.log1p(math.exp(delta)))
        ga = self.standard_gamma(a)
        gb = self.standard_gamma(b)
        return ga / (ga + gb)

    def chisquare(self, df):
        return 2.0 * self.standard_gamma(df / 2.0)

    def f(self, dfnum, dfden):
        return (self.chisquare(dfnum) * dfden) / (self.chisquare(dfden) * dfnum)

    def standard_cauchy(self):
        return self.standard_normal() / self.standard_normal()

    def pareto(self, a):
        return math.expm1(self.standard_exponential() / a)

    def weibull(self, a):
        if a == 0.0:
            return 0.0
        return math.pow(self.standard_exponential(), 1.0 / a)

    def power(self, a):
        return math.pow(-math.expm1(-self.standard_exponential()), 1.0 / a)

    def laplace(self, loc, scale):
        u = self.bit_generator.next_double()
        if u >= 0.5:
            return math.fma(-scale, math.log(2.0 - u - u), loc)
        if u > 0.0:
            return math.fma(scale, math.log(u + u), loc)
        return self.laplace(loc, scale)

    def gumbel(self, loc, scale):
        u = 1.0 - self.bit_generator.next_double()
        if u < 1.0:
            return math.fma(-scale, math.log(-math.log(u)), loc)
        return self.gumbel(loc, scale)

    def logistic(self, loc, scale):
        u = self.bit_generator.next_double()
        if u > 0.0:
            return math.fma(scale, math.log(u / (1.0 - u)), loc)
        return self.logistic(loc, scale)

    def lognormal(self, mean, sigma):
        return math.exp(math.fma(sigma, self.standard_normal(), mean))

    def rayleigh(self, scale):
        return scale * math.sqrt(2.0 * self.standard_exponential())

    def standard_t(self, df):
        num = self.standard_normal()
        denom = self.standard_gamma(df / 2.0)
        return math.sqrt(df / 2.0) * num / math.sqrt(denom)

    def triangular(self, left, mode, right):
        base = right - left
        leftbase = mode - left
        ratio = leftbase / base
        leftprod = leftbase * base
        rightprod = (right - mode) * base
        u = self.bit_generator.next_double()
        if u <= ratio:
            return left + math.sqrt(u * leftprod)
        return right - math.sqrt((1.0 - u) * rightprod)

    def geometric(self, p):
        if p >= 1.0 / 3.0:
            x = 1
            total = product = p
            q = 1.0 - p
            u = self.bit_generator.next_double()
            while u > total:
                product *= q
                total += product
                x += 1
            return x
        z = math.ceil(-self.standard_exponential() / math.log1p(-p))
        return min(z, (1 << 63) - 1)

    def logseries(self, p):
        r = math.log1p(-p)
        while True:
            v = self.bit_generator.next_double()
            if v >= p:
                return 1
            u = self.bit_generator.next_double()
            q = -math.expm1(r * u)
            if v <= q * q:
                result = math.floor(1.0 + math.log(v) / math.log(q))
                if result < 1 or v == 0.0:
                    continue
                return result
            if v >= q:
                return 1
            return 2

    def zipf(self, a):
        if a >= 1025.0:
            return 1
        am1 = a - 1.0
        b = math.pow(2.0, am1)
        umin = math.pow(float((1 << 63) - 1), -am1)
        while True:
            u01 = self.bit_generator.next_double()
            u = math.fma(u01, umin, 1.0 - u01)
            v = self.bit_generator.next_double()
            x = math.floor(math.pow(u, -1.0 / am1))
            if x > (1 << 63) - 1 or x < 1.0:
                continue
            t = math.pow(1.0 + 1.0 / x, am1)
            if v * x * (t - 1.0) / (b - 1.0) <= t / b:
                return int(x)

    @staticmethod
    def _loggam(x):
        coeffs = (
            8.333333333333333e-02, -2.777777777777778e-03,
            7.936507936507937e-04, -5.952380952380952e-04,
            8.417508417508418e-04, -1.917526917526918e-03,
            6.410256410256410e-03, -2.955065359477124e-02,
            1.796443723688307e-01, -1.39243221690590e00,
        )
        if x == 1.0 or x == 2.0:
            return 0.0
        n = int(7.0 - x) if x < 7.0 else 0
        x0 = x + n
        x2 = (1.0 / x0) * (1.0 / x0)
        gl0 = coeffs[9]
        for k in range(8, -1, -1):
            gl0 = math.fma(gl0, x2, coeffs[k])
        gl = gl0 / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * math.log(x0) - x0
        if x < 7.0:
            for _ in range(1, n + 1):
                gl -= math.log(x0 - 1.0)
                x0 -= 1.0
        return gl

    def poisson(self, lam):
        if lam == 0.0:
            return 0
        if lam < 10.0:
            enlam = math.exp(-lam)
            x = 0
            product = 1.0
            while True:
                product *= self.bit_generator.next_double()
                if product > enlam:
                    x += 1
                else:
                    return x
        slam = math.sqrt(lam)
        loglam = math.log(lam)
        b = math.fma(2.53, slam, 0.931)
        a = math.fma(0.02483, b, -0.059)
        invalpha = 1.1239 + 1.1328 / (b - 3.4)
        vr = 0.9277 - 3.6224 / (b - 2.0)
        while True:
            u = self.bit_generator.next_double() - 0.5
            v = self.bit_generator.next_double()
            us = 0.5 - abs(u)
            k = math.floor(math.fma(2.0 * a / us + b, u, lam + 0.43))
            if us >= 0.07 and v <= vr:
                return k
            if k < 0 or (us < 0.013 and v > us):
                continue
            lhs = math.log(v) + math.log(invalpha) - math.log(a / (us * us) + b)
            rhs = -lam + k * loglam - self._loggam(k + 1.0)
            if lhs <= rhs:
                return k

    def _binomial_inversion(self, n, p):
        q = 1.0 - p
        qn = math.exp(n * math.log1p(-p))
        np_ = n * p
        bound = min(n, int(np_ + 10.0 * math.sqrt(np_ * q + 1.0)))
        x = 0
        px = qn
        u = self.bit_generator.next_double()
        while u > px:
            x += 1
            if x > bound:
                x = 0
                px = qn
                u = self.bit_generator.next_double()
            else:
                u -= px
                px = ((n - x + 1) * p * px) / (x * q)
        return x

    def _binomial_btpe(self, n, p):
        r = min(p, 1.0 - p)
        q = 1.0 - r
        fm = math.fma(float(n), r, r)
        m = math.floor(fm)
        p1 = math.floor(math.fma(2.195, math.sqrt(n * r * q), -4.6 * q)) + 0.5
        xm = m + 0.5
        xl = xm - p1
        xr = xm + p1
        c = 0.134 + 20.5 / (15.3 + m)
        aa = (fm - xl) / (fm - xl * r)
        laml = aa * (1.0 + aa / 2.0)
        aa = (xr - fm) / (xr * q)
        lamr = aa * (1.0 + aa / 2.0)
        p2 = p1 * (1.0 + 2.0 * c)
        p3 = p2 + c / laml
        p4 = p3 + c / lamr
        nrq = n * r * q
        while True:
            u = self.bit_generator.next_double() * p4
            v = self.bit_generator.next_double()
            if u <= p1:
                y = math.floor(xm - p1 * v + u)
            elif u <= p2:
                x = xl + (u - p1) / c
                v = v * c + 1.0 - abs(m - x + 0.5) / p1
                if v > 1.0:
                    continue
                y = math.floor(x)
                if not self._binomial_btpe_accept(n, r, q, m, xm, y, v, nrq):
                    continue
            elif u <= p3:
                if v == 0.0:
                    continue
                y = math.floor(xl + math.log(v) / laml)
                if y < 0:
                    continue
                v = v * (u - p2) * laml
                if not self._binomial_btpe_accept(n, r, q, m, xm, y, v, nrq):
                    continue
            else:
                if v == 0.0:
                    continue
                y = math.floor(xr - math.log(v) / lamr)
                if y > n:
                    continue
                v = v * (u - p3) * lamr
                if not self._binomial_btpe_accept(n, r, q, m, xm, y, v, nrq):
                    continue
            return n - y if p > 0.5 else y

    @staticmethod
    def _binomial_btpe_accept(n, r, q, m, xm, y, v, nrq):
        k = abs(y - m)
        if not (k > 20 and k < nrq / 2.0 - 1.0):
            s = r / q
            aa = s * (n + 1)
            f = 1.0
            if m < y:
                for i in range(m + 1, y + 1):
                    f *= aa / i - s
            elif m > y:
                for i in range(y + 1, m + 1):
                    f /= aa / i - s
            return v <= f
        rho = (k / nrq) * ((k * (k / 3.0 + 0.625) + 1.0 / 6.0) / nrq + 0.5)
        t = -k * k / (2.0 * nrq)
        logv = math.log(v)
        if logv < t - rho:
            return True
        if logv > t + rho:
            return False
        x1 = y + 1.0
        f1 = m + 1.0
        z = n + 1.0 - m
        w = n - y + 1.0

        def correction(value):
            value2 = value * value
            return (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / value2)
                    / value2) / value2) / value2) / value / 166320.0

        threshold = (xm * math.log(f1 / x1)
                     + (n - m + 0.5) * math.log(z / w)
                     + (y - m) * math.log(w * r / (x1 * q))
                     + correction(f1) + correction(z)
                     - correction(x1) - correction(w))
        return logv <= threshold

    def binomial(self, n, p):
        if n == 0 or p == 0.0:
            return 0
        if p <= 0.5:
            if p * n <= 30.0:
                return self._binomial_inversion(n, p)
            return self._binomial_btpe(n, p)
        q = 1.0 - p
        if q * n <= 30.0:
            return n - self._binomial_inversion(n, q)
        return self._binomial_btpe(n, p)

    def negative_binomial(self, n, p):
        y = self.standard_gamma(n) * ((1.0 - p) / p)
        return self.poisson(y)

    def noncentral_chisquare(self, df, nonc):
        if math.isnan(nonc):
            return math.nan
        if nonc == 0.0:
            return self.chisquare(df)
        if df > 1.0:
            chi2 = self.chisquare(df - 1.0)
            n = self.standard_normal() + math.sqrt(nonc)
            return math.fma(n, n, chi2)
        count = self.poisson(nonc / 2.0)
        return self.chisquare(df + 2.0 * count)

    def noncentral_f(self, dfnum, dfden, nonc):
        t = self.noncentral_chisquare(dfnum, nonc) * dfden
        return t / (self.chisquare(dfden) * dfnum)

    def wald(self, mean, scale):
        y = self.standard_normal()
        y = mean * y * y
        d = 1.0 + math.sqrt(1.0 + 4.0 * scale / y)
        x = mean * (1.0 - 2.0 / d)
        u = self.bit_generator.next_double()
        if u <= mean / (mean + x):
            return x
        return mean * mean / x

    def vonmises(self, mu, kappa):
        if math.isnan(kappa):
            return math.nan
        if kappa < 1e-8:
            return math.pi * (2.0 * self.bit_generator.next_double() - 1.0)
        if kappa < 1e-5:
            s = 1.0 / kappa + kappa
        elif kappa <= 1e6:
            r = 1.0 + math.sqrt(math.fma(4.0 * kappa, kappa, 1.0))
            rho = (r - math.sqrt(2.0 * r)) / (2.0 * kappa)
            s = math.fma(rho, rho, 1.0) / (2.0 * rho)
        else:
            result = math.fma(math.sqrt(1.0 / kappa), self.standard_normal(), mu)
            if result < -math.pi:
                result += 2.0 * math.pi
            if result > math.pi:
                result -= 2.0 * math.pi
            return result
        while True:
            u = self.bit_generator.next_double()
            z = math.cos(math.pi * u)
            w = math.fma(s, z, 1.0) / (s + z)
            y = kappa * (s - w)
            v = self.bit_generator.next_double()
            if math.fma(y, 2.0 - y, -v) >= 0.0 or math.log(y / v) + 1.0 - y >= 0.0:
                break
        u = self.bit_generator.next_double()
        result = math.acos(w)
        if u < 0.5:
            result = -result
        result += mu
        negative = result < 0.0
        mod = math.fmod(abs(result) + math.pi, 2.0 * math.pi) - math.pi
        return -mod if negative else mod
