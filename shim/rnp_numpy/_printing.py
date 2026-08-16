"""numpy's Dragon4 float formatters and the print-options state.

`format_float_positional` / `format_float_scientific` are numpy's public
handles on the Dragon4 printer.  Everything encoded here was probed against
numpy 2.5.2:

  * ``precision`` counts digits *after the decimal point* -- of the whole
    number in positional/fractional mode, of the mantissa in scientific mode
    -- except in ``fractional=False`` positional mode where it counts
    *significant* digits;
  * ``unique=True`` prints the shortest digit string that round-trips through
    the value's own float type.  ``precision`` can only *shorten* that string
    and ``min_digits`` can only *lengthen* it; whenever either bound bites,
    the digits come from the value's exact decimal expansion instead (which
    is why ``fsci64(-2.0**-97, unique=True, precision=15)`` keeps the unique
    ``...095`` but ``unique=False, precision=15`` gives the exact ``...094``);
  * the C implementation writes into a 16384-byte buffer and raises
    ``RuntimeError('Float formatting result too large')`` when the result --
    padding included -- would not fit (gh-28068).

Every float is a dyadic rational, so its decimal expansion is finite and
`decimal.Decimal` reproduces it exactly; the "exact" digits are obtained by
rounding that expansion half-to-even at the requested place, which is what
Dragon4's cutoff mode does.
"""

import math as _math
import struct as _struct
from decimal import Decimal as _Decimal
from decimal import ROUND_HALF_EVEN as _HALF_EVEN
from decimal import localcontext as _localcontext

__all__ = ["format_float_positional", "format_float_scientific"]

#: numpy's dragon4 scratch buffer, NUL included.
_BUFFER_SIZE = 16384

#: `struct` codes for the float types whose round-trip width differs.
_PACK = {"float16": "e", "float32": "f", "float64": "d"}


# ---------------------------------------------------------------------------
# print options
#
# The options themselves live in `rnp_numpy._core.printoptions.format_options`
# (numpy's own `ContextVar`); `rnp_numpy._core.arrayprint` owns the public
# `set_printoptions` / `get_printoptions` / `printoptions` API over it.  Only
# the `legacy` level is needed here, and by `_scalars`.
# ---------------------------------------------------------------------------


def _legacy_level():
    """The active legacy mode as a comparable integer (`sys.maxsize` = off)."""
    from ._core.printoptions import format_options
    return format_options.get()["legacy"]


# ---------------------------------------------------------------------------
# digit generation
# ---------------------------------------------------------------------------


def _float_and_name(x):
    """`(python float, float-type name)` for anything these formatters take."""
    d = getattr(x, "dtype", None)
    name = getattr(d, "name", None)
    if name not in _PACK:
        # longdouble/clongdouble and plain Python floats all round-trip
        # through a double in this port.
        name = "float64"
    return float(x), name


def _unique_digits(v, name):
    """Shortest round-tripping decimal for ``|v|``, as `(digits, exp10)`.

    ``v`` is ``d1.d2d3... * 10**exp10``.  The round-trip test uses the
    *narrow* type's bytes, so `np.float32(0.1)` yields the short ``"1"``
    digits rather than the float64 spelling of the same bits.

    Simply rounding ``v`` to ``p`` significant digits is not enough: a float
    that sits exactly on a power of two has an asymmetric rounding interval,
    so the nearest ``p``-digit decimal can fail to round-trip while its
    neighbour succeeds (``-2.0**-97`` is ``-6.310887241768095e-30``, not the
    correctly-rounded ``...094``).  Both neighbours are therefore tried.
    """
    code = _PACK[name]
    target = _struct.pack(code, v)
    exact = _Decimal(v)
    for p in range(1, 18):
        with _localcontext() as ctx:
            ctx.prec = p
            base = +exact
        with _localcontext() as ctx:
            ctx.prec = 1000000  # the exact expansion runs to ~750 digits
            step = _Decimal(1).scaleb(base.adjusted() - p + 1)
            best = None
            for cand in (base, base - step, base + step):
                try:
                    if _struct.pack(code, float(cand)) != target:
                        continue
                except (OverflowError, ValueError):
                    continue
                if best is None or abs(cand - exact) < abs(best - exact):
                    best = cand
        if best is not None:
            digits = "".join(str(d) for d in best.as_tuple()[1])
            return (digits.lstrip("0") or "0"), best.adjusted()
    raise AssertionError("no round-tripping decimal")  # pragma: no cover


def _exhaust_low(value):
    """Decimal place of ``value``'s last non-zero digit.

    Dragon4's exact-digit loop stops as soon as the remainder is zero, so this
    is where digit *generation* ends; anything below it is padding that only
    ``trim='k'`` puts back.
    """
    _, digits, exp = _Decimal(value).as_tuple()
    n = len(digits)
    while n > 1 and digits[n - 1] == 0:
        n -= 1
    return exp + (len(digits) - n)


def _round_at(value, low):
    """Round |value| half-to-even at decimal place ``10**low``.

    Returns ``(digits, low)`` with ``|value| ~= int(digits) * 10**low``.
    """
    exact = _Decimal(value).copy_abs()
    with _localcontext() as ctx:
        ctx.prec = 1000000
        q = exact.quantize(_Decimal(1).scaleb(low), rounding=_HALF_EVEN)
    digits = "".join(str(d) for d in q.as_tuple()[1]).lstrip("0") or "0"
    if q > exact:
        # Rounding *up* carries through the trailing nines, and dragon4 drops
        # them instead of writing zeros: float16 0.1 (0.0999755859375) cut at
        # four decimals is the one digit "1", not "1000".
        stripped = digits.rstrip("0")
        if stripped:
            return stripped, low + (len(digits) - len(stripped))
    return digits, low


# ---------------------------------------------------------------------------
# the formatters
# ---------------------------------------------------------------------------


def _too_large():
    raise RuntimeError("Float formatting result too large")


def _check_size(left, right):
    if left + 1 + right + 1 > _BUFFER_SIZE:
        _too_large()


def _validate(trim, **nonneg):
    """numpy's argument checks, in numpy's order and with numpy's messages."""
    for name, value in nonneg.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} must be >= 0")
    precision, min_digits = nonneg.get("precision"), nonneg.get("min_digits")
    # `None` is spelled `-1` inside numpy, so a zero bound never triggers this.
    if (precision is not None and min_digits is not None
            and min_digits > 0 and precision > 0 and min_digits > precision):
        raise ValueError("min_digits must be less than or equal to precision")
    if trim not in ("k", ".", "0", "-"):
        raise TypeError("if supplied, trim must be 'k', '.', '0' or '-' "
                        "found `%100s`" % (trim,))


def _apply_trim(intstr, fracstr, trim, scientific=False):
    """numpy's four trailing-digit policies.

    `trim='-'` drops the decimal point along with the zeros -- but only in
    positional notation.  The scientific formatter never removes a point it
    has already written, so ``min_digits=3`` on 1.0 gives ``1.e+00``.
    """
    if trim == "k":
        return intstr, fracstr, True
    had_fraction = bool(fracstr)
    stripped = fracstr.rstrip("0")
    if stripped:
        return intstr, stripped, True
    if trim == ".":
        return intstr, "", True
    if trim == "0":
        return intstr, "0", True
    return intstr, "", scientific and had_fraction  # trim == '-'


def _pad(left, right, has_point, pad_left, pad_right):
    out = left + ("." if has_point else "")
    if pad_right is not None and pad_right >= len(right):
        # a point trimmed away still costs its column once right-padding is on
        if not has_point:
            out += " "
        out += right + " " * (pad_right - len(right))
    else:
        out += right
    if pad_left is not None and pad_left > len(left):
        out = " " * (pad_left - len(left)) + out
    return out


def _resolve_low(exp10, udigits, unique, precision, min_digits, place_of):
    """Pick the decimal place of the last printed digit.

    `place_of(n)` maps a digit count to the corresponding place exponent.
    """
    u_low = exp10 - (len(udigits) - 1)
    if not unique:
        if precision is None:
            raise TypeError("in non-unique mode `precision` must be supplied")
        return place_of(precision)
    low = u_low
    # `min_digits` deepens the cutoff, `precision` then caps it -- in that
    # order, so `precision=0, min_digits=3` stops the generator at the units
    # place and pads the rest with zeros rather than printing exact digits.
    if min_digits is not None:
        low = min(low, place_of(min_digits))
    if precision is not None:
        low = max(low, place_of(precision))
    return low


def _zero_pad(fracstr, nwhole, trim, unique, precision, min_digits,
              total_length):
    """numpy pads to `min_digits` (unique) / `precision` (exact) under trim 'k'.

    The digit generator stops as soon as the value is exhausted or the cutoff
    is reached; the trailing zeros that make ``1.5`` print as ``1.5000`` are
    added afterwards, and only when no trimming is asked for.
    """
    if trim != "k":
        return fracstr
    add = min_digits if unique else precision
    if add is None:
        return fracstr
    want = add - nwhole if total_length else add
    if want > len(fracstr):
        fracstr += "0" * (want - len(fracstr))
    return fracstr


def _nonfinite(v, sign):
    """`inf`/`nan` ignore every padding and trimming option."""
    if v != v:
        return "nan"
    if v > 0:
        return "+inf" if sign else "inf"
    return "-inf"


def format_float_positional(x, precision=None, unique=True, fractional=True,
                            trim='k', sign=False, pad_left=None,
                            pad_right=None, min_digits=None):
    """Format a floating-point scalar in positional notation."""
    v, name = _float_and_name(x)
    if not fractional and precision == 0:
        raise ValueError("precision must be greater than 0 if fractional=False")
    _validate(trim, precision=precision, pad_left=pad_left,
              pad_right=pad_right, min_digits=min_digits)
    if not unique and precision is None:
        raise TypeError("in non-unique mode `precision` must be supplied")
    if v != v or v in (float("inf"), float("-inf")):
        return _nonfinite(v, sign)

    neg = _is_neg(v)
    udigits, uexp = _unique_digits(abs(v), name)
    # The cutoff is measured against the *exact* value's leading digit, which
    # is not always the shortest form's: 9.9999999999999995e-08 is uniquely
    # `1e-07`, yet `precision=16` must still print its 17 exact digits.
    exp10 = _Decimal(abs(v)).adjusted()

    if fractional:
        def place_of(n):
            return -n
    else:
        def place_of(n, _e=exp10):
            return _e - n + 1

    low = _resolve_low(uexp, udigits, unique, precision, min_digits, place_of)

    signstr = "-" if neg else ("+" if sign else "")
    # The size check runs on the *requested* geometry, before any digits are
    # produced -- numpy sizes its buffer the same way, so `pad_left=2**31-1`
    # raises rather than trying to build the string.
    _check_size(max(pad_left or 0, len(signstr) + max(1, exp10 + 1)),
                max(pad_right or 0, max(0, -low)))

    if v == 0:
        # dragon4 emits a single '0' for zero whatever the cutoffs say; any
        # further zeros are padding.
        digits, low = "0", 0
    elif unique and low == uexp - (len(udigits) - 1):
        digits, low = udigits, uexp - (len(udigits) - 1)
    else:
        ndig = exp10 - low + 1  # significant digits asked for
        if not unique:
            low = max(low, _exhaust_low(v))
        digits, low = _round_at(v, low)
        if not fractional and digits != "0":
            new_exp = low + len(digits) - 1
            if new_exp > exp10:  # a carry (9.99 -> 10.0) moved the lead digit
                digits, low = _round_at(v, new_exp - ndig + 1)

    intstr, fracstr = _split(digits, low)
    fracstr = _zero_pad(fracstr, len(intstr), trim, unique, precision,
                        min_digits, not fractional)
    _check_size(max(pad_left or 0, len(signstr) + len(intstr)),
                max(pad_right or 0, len(fracstr)))
    intstr, fracstr, has_point = _apply_trim(intstr, fracstr, trim)
    return _pad(signstr + intstr, fracstr, has_point, pad_left, pad_right)


def format_float_scientific(x, precision=None, unique=True, trim='k',
                            sign=False, pad_left=None, exp_digits=None,
                            min_digits=None):
    """Format a floating-point scalar in scientific notation."""
    v, name = _float_and_name(x)
    _validate(trim, precision=precision, pad_left=pad_left,
              exp_digits=exp_digits, min_digits=min_digits)
    if not unique and precision is None:
        raise TypeError("in non-unique mode `precision` must be supplied")
    if v != v or v in (float("inf"), float("-inf")):
        return _nonfinite(v, sign)

    neg = _is_neg(v)
    udigits, uexp = _unique_digits(abs(v), name)
    u_low = uexp - (len(udigits) - 1)
    exp10 = _Decimal(abs(v)).adjusted()

    def place_of(n, _e=exp10):
        return _e - n

    low = _resolve_low(uexp, udigits, unique, precision, min_digits, place_of)
    ndig = exp10 - low + 1  # mantissa digits asked for
    signstr = "-" if neg else ("+" if sign else "")
    _check_size(max(pad_left or 0, len(signstr) + 1),
                max(0, ndig - 1) + 2 + max(2 if exp_digits is None
                                           else exp_digits, 4))
    if v == 0:
        digits, out_exp = "0", 0
    elif unique and low == u_low:
        digits = udigits
        out_exp = uexp
    else:
        if not unique:
            low = max(low, _exhaust_low(v))
        digits, low = _round_at(v, low)
        if digits == "0":
            digits = "0" * max(1, ndig if unique else 1)
            out_exp = 0
        else:
            out_exp = low + len(digits) - 1
            if out_exp > exp10:
                # a carry (9.99 -> 10.0) moved the leading digit one place up
                digits, low = _round_at(v, out_exp - ndig + 1)
                out_exp = low + len(digits) - 1

    intstr, fracstr = digits[:1], digits[1:]
    fracstr = _zero_pad(fracstr, 1, trim, unique, precision, min_digits, False)
    expstr = "e%s%0*d" % ("+" if out_exp >= 0 else "-",
                          2 if exp_digits is None else exp_digits,
                          abs(out_exp))
    intstr, fracstr, has_point = _apply_trim(intstr, fracstr, trim,
                                             scientific=True)
    return _pad(signstr + intstr, fracstr + expstr, has_point, pad_left, None)


def _is_neg(v):
    return _math.copysign(1.0, v) < 0


def _split(digits, low):
    """`(integer part, fractional part)` of ``int(digits) * 10**low``."""
    n = len(digits)
    if low >= 0:
        intstr = digits + "0" * low
        return (intstr.lstrip("0") or "0"), ""
    if n + low > 0:
        return digits[:n + low].lstrip("0") or "0", digits[n + low:]
    return "0", "0" * (-(n + low)) + digits
