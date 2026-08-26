//! One-dimensional pocketfft kernels and ndarray axis orchestration.
//!
//! This is a safe Rust transcription of pocketfft's complex FFTPACK and
//! Bluestein plans. Expression grouping, factor order, twiddle construction,
//! buffer swaps, and final scaling follow `pocketfft_hdronly.h`.

use num_complex::Complex;

use crate::{C160, DType, Error, F80, NdArray, Result, Scalar};

type C64 = Complex<f64>;

fn f80_twiddle(phase: usize, n: usize, forward: bool) -> C160 {
    let angle = F80::PI
        .add(F80::PI)
        .mul(F80::from_u64((phase % n) as u64))
        .div(F80::from_u64(n as u64));
    let (sin, cos) = angle.sin_cos_0_2pi();
    C160 {
        re: cos,
        im: if forward { sin.neg() } else { sin },
    }
}

#[derive(Clone, Copy)]
struct F80Accum {
    high: F80,
    low: F80,
}

impl F80Accum {
    fn zero() -> Self {
        Self { high: F80::ZERO, low: F80::ZERO }
    }

    fn add(&mut self, value: F80) {
        let sum = self.high.add(value);
        let virtual_value = sum.sub(self.high);
        let error = self
            .high
            .sub(sum.sub(virtual_value))
            .add(value.sub(virtual_value));
        let tail = self.low.add(error);
        let high = sum.add(tail);
        self.low = tail.sub(high.sub(sum));
        self.high = high;
    }

    fn add_product(&mut self, left: F80, right: F80, negative: bool) {
        let product = left.mul(right);
        let splitter = F80::from_u64((1u64 << 32) + 1);
        let split_left = splitter.mul(left);
        let left_high = split_left.sub(split_left.sub(left));
        let left_low = left.sub(left_high);
        let split_right = splitter.mul(right);
        let right_high = split_right.sub(split_right.sub(right));
        let right_low = right.sub(right_high);
        let error = left_high
            .mul(right_high)
            .sub(product)
            .add(left_high.mul(right_low))
            .add(left_low.mul(right_high))
            .add(left_low.mul(right_low));
        if negative {
            self.add(product.neg());
            self.add(error.neg());
        } else {
            self.add(product);
            self.add(error);
        }
    }

    fn value(self) -> F80 {
        self.high.add(self.low)
    }
}

fn f80_twiddles(n: usize, forward: bool) -> Vec<C160> {
    let mut twiddles = vec![C160::ZERO; n];
    twiddles[0] = C160 { re: F80::ONE, im: F80::ZERO };
    for phase in 1..=n / 2 {
        let value = f80_twiddle(phase, n, forward);
        twiddles[phase] = value;
        if phase != n - phase {
            twiddles[n - phase] = C160 { re: value.re, im: value.im.neg() };
        }
    }
    twiddles
}

fn c2c_f80_unscaled(input: &[C160], forward: bool) -> Vec<C160> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }
    let twiddles = f80_twiddles(n, forward);
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut re = F80Accum::zero();
        let mut im = F80Accum::zero();
        for (j, value) in input.iter().enumerate() {
            let phase = ((k as u128 * j as u128) % n as u128) as usize;
            let twiddle = twiddles[phase];
            re.add_product(value.re, twiddle.re, false);
            re.add_product(value.im, twiddle.im, true);
            im.add_product(value.re, twiddle.im, false);
            im.add_product(value.im, twiddle.re, false);
        }
        out.push(C160 { re: re.value(), im: im.value() });
    }
    out
}

/// Extended-precision complex transform.
///
/// This path deliberately favors correctness over speed.  The software F80
/// backend only exists on Linux/x86-64, and a compensated DFT keeps every
/// operation, including twiddle construction and normalization, in that
/// format instead of silently passing through binary64.
fn c2c_f80(input: &[C160], forward: bool, scale: F80) -> Vec<C160> {
    if forward {
        return c2c_f80_unscaled(input, true)
            .into_iter()
            .map(|value| C160 {
                re: value.re.mul(scale),
                im: value.im.mul(scale),
            })
            .collect();
    }

    let n = input.len();
    let inverse_n = F80::ONE.div(F80::from_u64(n as u64));
    let mut estimate: Vec<C160> = c2c_f80_unscaled(input, false)
        .into_iter()
        .map(|value| C160 {
            re: value.re.mul(inverse_n),
            im: value.im.mul(inverse_n),
        })
        .collect();
    // The rounded F80 twiddle matrix is not perfectly unitary.  One residual
    // correction makes this software inverse solve the matrix actually used
    // by the forward transform, rather than assuming its conjugate is exact.
    let projected = c2c_f80_unscaled(&estimate, true);
    let residual: Vec<C160> = input
        .iter()
        .zip(projected)
        .map(|(&wanted, actual)| wanted.sub(actual))
        .collect();
    let correction = c2c_f80_unscaled(&residual, false);
    for (value, correction) in estimate.iter_mut().zip(correction) {
        value.re = value.re.add(correction.re.mul(inverse_n));
        value.im = value.im.add(correction.im.mul(inverse_n));
    }
    let factor = if scale.same_bits(inverse_n) {
        F80::ONE
    } else {
        scale.mul(F80::from_u64(n as u64))
    };
    estimate
        .into_iter()
        .map(|value| C160 {
            re: value.re.mul(factor),
            im: value.im.mul(factor),
        })
        .collect()
}

fn scale_f80(scale: f64, n: usize) -> F80 {
    if scale == 1.0 {
        F80::ONE
    } else if scale == 1.0 / n as f64 {
        F80::ONE.div(F80::from_u64(n as u64))
    } else {
        F80::ONE.div(F80::from_u64(n as u64).sqrt())
    }
}

fn r2c_f80(input: &[F80], scale: F80) -> Vec<C160> {
    let n = input.len();
    let twiddles = f80_twiddles(n, true);
    let mut out = Vec::with_capacity(n / 2 + 1);
    for k in 0..=n / 2 {
        let mut re = F80Accum::zero();
        let mut im = F80Accum::zero();
        for (j, &value) in input.iter().enumerate() {
            let phase = ((k as u128 * j as u128) % n as u128) as usize;
            re.add_product(value, twiddles[phase].re, false);
            im.add_product(value, twiddles[phase].im, false);
        }
        out.push(C160 {
            re: re.value().mul(scale),
            im: im.value().mul(scale),
        });
    }
    out
}

fn hermitian_f80(input: &[C160], n: usize) -> Vec<C160> {
    let mut spectrum = vec![C160::ZERO; n];
    let take = input.len().min(n / 2 + 1);
    if take != 0 {
        spectrum[0].re = input[0].re;
    }
    for k in 1..take {
        spectrum[k] = input[k];
        if k == n - k {
            spectrum[k].im = F80::ZERO;
        } else {
            spectrum[n - k] = C160 {
                re: input[k].re,
                im: input[k].im.neg(),
            };
        }
    }
    spectrum
}

fn c2r_f80(input: &[C160], n: usize, scale: F80) -> Vec<F80> {
    let spectrum = hermitian_f80(input, n);
    let mut result: Vec<F80> = c2c_f80(&spectrum, false, scale)
        .into_iter()
        .map(|value| value.re)
        .collect();
    for _ in 0..2 {
        let projected = r2c_f80(&result, F80::ONE);
        let residual: Vec<C160> = input
            .iter()
            .take(projected.len())
            .zip(projected)
            .enumerate()
            .map(|(index, (&wanted, actual))| C160 {
                re: wanted.re.sub(actual.re),
                im: if index == 0 || (n & 1 == 0 && index == n / 2) {
                    F80::ZERO
                } else {
                    wanted.im.sub(actual.im)
                },
            })
            .collect();
        let correction_spectrum = hermitian_f80(&residual, n);
        let correction = c2c_f80(&correction_spectrum, false, scale);
        for (value, correction) in result.iter_mut().zip(correction) {
            *value = value.add(correction.re);
        }
    }
    result
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct C {
    r: f64,
    i: f64,
}

impl C {
    #[inline]
    const fn new(r: f64, i: f64) -> Self {
        Self { r, i }
    }
    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.r + other.r, self.i + other.i)
    }
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self::new(self.r - other.r, self.i - other.i)
    }
    #[inline]
    fn mul(self, value: f64) -> Self {
        Self::new(self.r * value, self.i * value)
    }
}

#[inline]
fn pm(c: C, d: C) -> (C, C) {
    (c.add(d), c.sub(d))
}

#[inline]
fn pm_inplace(a: &mut C, b: &mut C) {
    let t = *a;
    *a = a.add(*b);
    *b = t.sub(*b);
}

#[inline]
fn rot_x90(a: &mut C, fwd: bool) {
    let tmp = if fwd { -a.r } else { a.r };
    a.r = if fwd { a.i } else { -a.i };
    a.i = tmp;
}

#[inline]
fn special_mul(v1: C, v2: C, fwd: bool) -> C {
    if fwd {
        C::new(
            v1.r.mul_add(v2.r, v1.i * v2.i),
            v1.i.mul_add(v2.r, -(v1.r * v2.i)),
        )
    } else {
        C::new(
            v1.r.mul_add(v2.r, -(v1.i * v2.i)),
            v1.r.mul_add(v2.i, v1.i * v2.r),
        )
    }
}

struct Twiddles {
    n: usize,
    mask: usize,
    shift: usize,
    v1: Vec<C>,
    v2: Vec<C>,
}

impl Twiddles {
    fn calc(mut x: usize, n: usize, ang: f64) -> C {
        x <<= 3;
        if x < 4 * n {
            if x < 2 * n {
                if x < n {
                    let (s, c) = (x as f64 * ang).sin_cos();
                    C::new(c, s)
                } else {
                    let (s, c) = ((2 * n - x) as f64 * ang).sin_cos();
                    C::new(s, c)
                }
            } else {
                x -= 2 * n;
                if x < n {
                    let (s, c) = (x as f64 * ang).sin_cos();
                    C::new(-s, c)
                } else {
                    let (s, c) = ((2 * n - x) as f64 * ang).sin_cos();
                    C::new(-c, s)
                }
            }
        } else {
            x = 8 * n - x;
            if x < 2 * n {
                if x < n {
                    let (s, c) = (x as f64 * ang).sin_cos();
                    C::new(c, -s)
                } else {
                    let (s, c) = ((2 * n - x) as f64 * ang).sin_cos();
                    C::new(s, -c)
                }
            } else {
                x -= 2 * n;
                if x < n {
                    let (s, c) = (x as f64 * ang).sin_cos();
                    C::new(-s, -c)
                } else {
                    let (s, c) = ((2 * n - x) as f64 * ang).sin_cos();
                    C::new(-c, -s)
                }
            }
        }
    }

    fn new(n: usize) -> Self {
        let ang = 0.25 * 3.141592653589793238462643383279502884197_f64 / n as f64;
        let nval = (n + 2) / 2;
        let mut shift = 1usize;
        while (1usize << shift) * (1usize << shift) < nval {
            shift += 1;
        }
        let mask = (1usize << shift) - 1;
        let mut v1 = vec![C::default(); mask + 1];
        v1[0] = C::new(1.0, 0.0);
        for (i, value) in v1.iter_mut().enumerate().skip(1) {
            *value = Self::calc(i, n, ang);
        }
        let mut v2 = vec![C::default(); (nval + mask) / (mask + 1)];
        v2[0] = C::new(1.0, 0.0);
        for (i, value) in v2.iter_mut().enumerate().skip(1) {
            *value = Self::calc(i * (mask + 1), n, ang);
        }
        Self {
            n,
            mask,
            shift,
            v1,
            v2,
        }
    }

    #[inline]
    fn get(&self, mut idx: usize) -> C {
        if 2 * idx <= self.n {
            let x1 = self.v1[idx & self.mask];
            let x2 = self.v2[idx >> self.shift];
            C::new(
                x1.r.mul_add(x2.r, -(x1.i * x2.i)),
                x1.r.mul_add(x2.i, x1.i * x2.r),
            )
        } else {
            idx = self.n - idx;
            let x1 = self.v1[idx & self.mask];
            let x2 = self.v2[idx >> self.shift];
            C::new(
                x1.r.mul_add(x2.r, -(x1.i * x2.i)),
                -x1.r.mul_add(x2.i, x1.i * x2.r),
            )
        }
    }
}

struct Factor {
    fct: usize,
    tw: Vec<C>,
    tws: Vec<C>,
}

#[inline]
fn ch_idx(ido: usize, l1: usize, a: usize, b: usize, c: usize) -> usize {
    a + ido * (b + l1 * c)
}

#[inline]
fn cc_idx(ido: usize, radix: usize, a: usize, b: usize, c: usize) -> usize {
    a + ido * (b + radix * c)
}

#[inline]
fn wa_idx(ido: usize, x: usize, i: usize) -> usize {
    i - 1 + x * (ido - 1)
}

fn pass2(ido: usize, l1: usize, cc: &[C], ch: &mut [C], wa: &[C], fwd: bool) {
    for k in 0..l1 {
        for i in 0..ido {
            let c0 = cc[cc_idx(ido, 2, i, 0, k)];
            let c1 = cc[cc_idx(ido, 2, i, 1, k)];
            ch[ch_idx(ido, l1, i, k, 0)] = c0.add(c1);
            let value = c0.sub(c1);
            ch[ch_idx(ido, l1, i, k, 1)] = if i == 0 {
                value
            } else {
                special_mul(value, wa[wa_idx(ido, 0, i)], fwd)
            };
        }
    }
}

fn pass3(ido: usize, l1: usize, cc: &[C], ch: &mut [C], wa: &[C], fwd: bool) {
    let tw1r = -0.5;
    let tw1i = (if fwd { -1.0 } else { 1.0 }) * 0.8660254037844386467637231707529362_f64;
    for k in 0..l1 {
        for i in 0..ido {
            let t0 = cc[cc_idx(ido, 3, i, 0, k)];
            let (t1, t2) = pm(cc[cc_idx(ido, 3, i, 1, k)], cc[cc_idx(ido, 3, i, 2, k)]);
            ch[ch_idx(ido, l1, i, k, 0)] = t0.add(t1);
            let ca = t0.add(t1.mul(tw1r));
            let cb = C::new(-t2.i * tw1i, t2.r * tw1i);
            let (a, b) = pm(ca, cb);
            ch[ch_idx(ido, l1, i, k, 1)] = if i == 0 {
                a
            } else {
                special_mul(a, wa[wa_idx(ido, 0, i)], fwd)
            };
            ch[ch_idx(ido, l1, i, k, 2)] = if i == 0 {
                b
            } else {
                special_mul(b, wa[wa_idx(ido, 1, i)], fwd)
            };
        }
    }
}

fn pass4(ido: usize, l1: usize, cc: &[C], ch: &mut [C], wa: &[C], fwd: bool) {
    for k in 0..l1 {
        for i in 0..ido {
            let (t2, t1) = pm(cc[cc_idx(ido, 4, i, 0, k)], cc[cc_idx(ido, 4, i, 2, k)]);
            let (t3, mut t4) = pm(cc[cc_idx(ido, 4, i, 1, k)], cc[cc_idx(ido, 4, i, 3, k)]);
            rot_x90(&mut t4, fwd);
            let values = [t2.add(t3), t1.add(t4), t2.sub(t3), t1.sub(t4)];
            for (u, value) in values.into_iter().enumerate() {
                ch[ch_idx(ido, l1, i, k, u)] = if i == 0 || u == 0 {
                    value
                } else {
                    special_mul(value, wa[wa_idx(ido, u - 1, i)], fwd)
                };
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pair5(t0: C, t1: C, t2: C, t3: C, t4: C, ar: f64, br: f64, ai: f64, bi: f64) -> (C, C) {
    let ca = C::new(
        br.mul_add(t2.r, ar.mul_add(t1.r, t0.r)),
        br.mul_add(t2.i, ar.mul_add(t1.i, t0.i)),
    );
    let cb = C::new(-ai.mul_add(t4.i, bi * t3.i), ai.mul_add(t4.r, bi * t3.r));
    pm(ca, cb)
}

fn pass5(ido: usize, l1: usize, cc: &[C], ch: &mut [C], wa: &[C], fwd: bool) {
    let tw1r = 0.3090169943749474241022934171828191_f64;
    let tw1i = (if fwd { -1.0 } else { 1.0 }) * 0.9510565162951535721164393333793821_f64;
    let tw2r = -0.8090169943749474241022934171828191_f64;
    let tw2i = (if fwd { -1.0 } else { 1.0 }) * 0.5877852522924731291687059546390728_f64;
    for k in 0..l1 {
        for i in 0..ido {
            let t0 = cc[cc_idx(ido, 5, i, 0, k)];
            let (t1, t4) = pm(cc[cc_idx(ido, 5, i, 1, k)], cc[cc_idx(ido, 5, i, 4, k)]);
            let (t2, t3) = pm(cc[cc_idx(ido, 5, i, 2, k)], cc[cc_idx(ido, 5, i, 3, k)]);
            ch[ch_idx(ido, l1, i, k, 0)] = C::new(t0.r + t1.r + t2.r, t0.i + t1.i + t2.i);
            let (a1, a4) = pair5(t0, t1, t2, t3, t4, tw1r, tw2r, tw1i, tw2i);
            let (a2, a3) = pair5(t0, t1, t2, t3, t4, tw2r, tw1r, tw2i, -tw1i);
            for (u, value) in [(1, a1), (4, a4), (2, a2), (3, a3)] {
                ch[ch_idx(ido, l1, i, k, u)] = if i == 0 {
                    value
                } else {
                    special_mul(value, wa[wa_idx(ido, u - 1, i)], fwd)
                };
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pair7(
    t1: C,
    t2: C,
    t3: C,
    t4: C,
    t5: C,
    t6: C,
    t7: C,
    x1: f64,
    x2: f64,
    x3: f64,
    y1: f64,
    y2: f64,
    y3: f64,
) -> (C, C) {
    let ca = C::new(
        x3.mul_add(t4.r, x2.mul_add(t3.r, x1.mul_add(t2.r, t1.r))),
        x3.mul_add(t4.i, x2.mul_add(t3.i, x1.mul_add(t2.i, t1.i))),
    );
    let cb = C::new(
        -y3.mul_add(t5.i, y1.mul_add(t7.i, y2 * t6.i)),
        y3.mul_add(t5.r, y1.mul_add(t7.r, y2 * t6.r)),
    );
    pm(ca, cb)
}

fn pass7(ido: usize, l1: usize, cc: &[C], ch: &mut [C], wa: &[C], fwd: bool) {
    let s = if fwd { -1.0 } else { 1.0 };
    let tw1r = 0.6234898018587335305250048840042398_f64;
    let tw1i = s * 0.7818314824680298087084445266740578_f64;
    let tw2r = -0.2225209339563144042889025644967948_f64;
    let tw2i = s * 0.9749279121818236070181316829939312_f64;
    let tw3r = -0.9009688679024191262361023195074451_f64;
    let tw3i = s * 0.433883739117558120475768332848359_f64;
    for k in 0..l1 {
        for i in 0..ido {
            let t1 = cc[cc_idx(ido, 7, i, 0, k)];
            let (t2, t7) = pm(cc[cc_idx(ido, 7, i, 1, k)], cc[cc_idx(ido, 7, i, 6, k)]);
            let (t3, t6) = pm(cc[cc_idx(ido, 7, i, 2, k)], cc[cc_idx(ido, 7, i, 5, k)]);
            let (t4, t5) = pm(cc[cc_idx(ido, 7, i, 3, k)], cc[cc_idx(ido, 7, i, 4, k)]);
            ch[ch_idx(ido, l1, i, k, 0)] =
                C::new(t1.r + t2.r + t3.r + t4.r, t1.i + t2.i + t3.i + t4.i);
            let pairs = [
                (
                    1,
                    6,
                    pair7(
                        t1, t2, t3, t4, t5, t6, t7, tw1r, tw2r, tw3r, tw1i, tw2i, tw3i,
                    ),
                ),
                (
                    2,
                    5,
                    pair7(
                        t1, t2, t3, t4, t5, t6, t7, tw2r, tw3r, tw1r, tw2i, -tw3i, -tw1i,
                    ),
                ),
                (
                    3,
                    4,
                    pair7(
                        t1, t2, t3, t4, t5, t6, t7, tw3r, tw1r, tw2r, tw3i, -tw1i, tw2i,
                    ),
                ),
            ];
            for (u1, u2, (a, b)) in pairs {
                ch[ch_idx(ido, l1, i, k, u1)] = if i == 0 {
                    a
                } else {
                    special_mul(a, wa[wa_idx(ido, u1 - 1, i)], fwd)
                };
                ch[ch_idx(ido, l1, i, k, u2)] = if i == 0 {
                    b
                } else {
                    special_mul(b, wa[wa_idx(ido, u2 - 1, i)], fwd)
                };
            }
        }
    }
}

#[inline]
fn rot_x45(a: &mut C, fwd: bool) {
    const HSQT2: f64 = 0.707106781186547524400844362104849_f64;
    if fwd {
        let tmp = a.r;
        a.r = HSQT2 * (a.r + a.i);
        a.i = HSQT2 * (a.i - tmp);
    } else {
        let tmp = a.r;
        a.r = HSQT2 * (a.r - a.i);
        a.i = HSQT2 * (a.i + tmp);
    }
}

#[inline]
fn rot_x135(a: &mut C, fwd: bool) {
    const HSQT2: f64 = 0.707106781186547524400844362104849_f64;
    if fwd {
        let tmp = a.r;
        a.r = HSQT2 * (a.i - a.r);
        a.i = HSQT2 * (-tmp - a.i);
    } else {
        let tmp = a.r;
        a.r = HSQT2 * (-a.r - a.i);
        a.i = HSQT2 * (tmp - a.i);
    }
}

fn pass8(ido: usize, l1: usize, cc: &[C], ch: &mut [C], wa: &[C], fwd: bool) {
    for k in 0..l1 {
        for i in 0..ido {
            let (mut a1, mut a5) = pm(cc[cc_idx(ido, 8, i, 1, k)], cc[cc_idx(ido, 8, i, 5, k)]);
            let (mut a3, mut a7) = pm(cc[cc_idx(ido, 8, i, 3, k)], cc[cc_idx(ido, 8, i, 7, k)]);
            if i == 0 {
                pm_inplace(&mut a1, &mut a3);
                rot_x90(&mut a3, fwd);
                rot_x90(&mut a7, fwd);
            } else {
                rot_x90(&mut a7, fwd);
                pm_inplace(&mut a1, &mut a3);
                rot_x90(&mut a3, fwd);
            }
            pm_inplace(&mut a5, &mut a7);
            rot_x45(&mut a5, fwd);
            rot_x135(&mut a7, fwd);
            let (mut a0, mut a4) = pm(cc[cc_idx(ido, 8, i, 0, k)], cc[cc_idx(ido, 8, i, 4, k)]);
            let (mut a2, mut a6) = pm(cc[cc_idx(ido, 8, i, 2, k)], cc[cc_idx(ido, 8, i, 6, k)]);
            let values = if i == 0 {
                let (v0, v4) = pm(a0.add(a2), a1);
                let (v2, v6) = pm(a0.sub(a2), a3);
                rot_x90(&mut a6, fwd);
                let (v1, v5) = pm(a4.add(a6), a5);
                let (v3, v7) = pm(a4.sub(a6), a7);
                [v0, v1, v2, v3, v4, v5, v6, v7]
            } else {
                pm_inplace(&mut a0, &mut a2);
                rot_x90(&mut a6, fwd);
                pm_inplace(&mut a4, &mut a6);
                [
                    a0.add(a1),
                    a4.add(a5),
                    a2.add(a3),
                    a6.add(a7),
                    a0.sub(a1),
                    a4.sub(a5),
                    a2.sub(a3),
                    a6.sub(a7),
                ]
            };
            for (u, value) in values.into_iter().enumerate() {
                ch[ch_idx(ido, l1, i, k, u)] = if i == 0 || u == 0 {
                    value
                } else {
                    special_mul(value, wa[wa_idx(ido, u - 1, i)], fwd)
                };
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pair11(t: &[C; 11], x: [f64; 5], y: [f64; 5]) -> (C, C) {
    // These are overloaded complex operations in the source; contraction
    // does not cross the inlined operator boundaries.
    let ca = t[0]
        .add(t[1].mul(x[0]))
        .add(t[2].mul(x[1]))
        .add(t[3].mul(x[2]))
        .add(t[4].mul(x[3]))
        .add(t[5].mul(x[4]));
    let cb = C::new(
        -y[4].mul_add(
            t[6].i,
            y[3].mul_add(
                t[7].i,
                y[2].mul_add(t[8].i, y[0].mul_add(t[10].i, y[1] * t[9].i)),
            ),
        ),
        y[4].mul_add(
            t[6].r,
            y[3].mul_add(
                t[7].r,
                y[2].mul_add(t[8].r, y[0].mul_add(t[10].r, y[1] * t[9].r)),
            ),
        ),
    );
    pm(ca, cb)
}

fn pass11(ido: usize, l1: usize, cc: &[C], ch: &mut [C], wa: &[C], fwd: bool) {
    let s = if fwd { -1.0 } else { 1.0 };
    let xr = [
        0.8412535328311811688618116489193677_f64,
        0.4154150130018864255292741492296232_f64,
        -0.1423148382732851404437926686163697_f64,
        -0.6548607339452850640569250724662936_f64,
        -0.9594929736144973898903680570663277_f64,
    ];
    let yi = [
        s * 0.5406408174555975821076359543186917_f64,
        s * 0.9096319953545183714117153830790285_f64,
        s * 0.9898214418809327323760920377767188_f64,
        s * 0.7557495743542582837740358439723444_f64,
        s * 0.2817325568414296977114179153466169_f64,
    ];
    let xs = [
        [0, 1, 2, 3, 4],
        [1, 3, 4, 2, 0],
        [2, 4, 1, 0, 3],
        [3, 2, 0, 4, 1],
        [4, 0, 3, 1, 2],
    ];
    let ys: [[isize; 5]; 5] = [
        [1, 2, 3, 4, 5],
        [2, 4, -5, -3, -1],
        [3, -5, -2, 1, 4],
        [4, -3, 1, 5, -2],
        [5, -1, 4, -2, 3],
    ];
    for k in 0..l1 {
        for i in 0..ido {
            let mut t = [C::default(); 11];
            t[0] = cc[cc_idx(ido, 11, i, 0, k)];
            for j in 1..=5 {
                (t[j], t[11 - j]) = pm(
                    cc[cc_idx(ido, 11, i, j, k)],
                    cc[cc_idx(ido, 11, i, 11 - j, k)],
                );
            }
            ch[ch_idx(ido, l1, i, k, 0)] = C::new(
                t[0].r + t[1].r + t[2].r + t[3].r + t[4].r + t[5].r,
                t[0].i + t[1].i + t[2].i + t[3].i + t[4].i + t[5].i,
            );
            for row in 0..5 {
                let x = xs[row].map(|j| xr[j]);
                let y = ys[row].map(|j| {
                    if j < 0 {
                        -yi[(-j - 1) as usize]
                    } else {
                        yi[(j - 1) as usize]
                    }
                });
                let (a, b) = pair11(&t, x, y);
                let u1 = row + 1;
                let u2 = 10 - row;
                ch[ch_idx(ido, l1, i, k, u1)] = if i == 0 {
                    a
                } else {
                    special_mul(a, wa[wa_idx(ido, u1 - 1, i)], fwd)
                };
                ch[ch_idx(ido, l1, i, k, u2)] = if i == 0 {
                    b
                } else {
                    special_mul(b, wa[wa_idx(ido, u2 - 1, i)], fwd)
                };
            }
        }
    }
}

fn passg(
    ido: usize,
    ip: usize,
    l1: usize,
    cc: &mut [C],
    ch: &mut [C],
    wa: &[C],
    csarr: &[C],
    fwd: bool,
) {
    let ipph = (ip + 1) / 2;
    let idl1 = ido * l1;
    let mut wal = vec![C::default(); ip];
    wal[0] = C::new(1.0, 0.0);
    for i in 1..ip {
        wal[i] = C::new(csarr[i].r, if fwd { -csarr[i].i } else { csarr[i].i });
    }

    for k in 0..l1 {
        for i in 0..ido {
            ch[ch_idx(ido, l1, i, k, 0)] = cc[cc_idx(ido, ip, i, 0, k)];
        }
    }
    for j in 1..ipph {
        let jc = ip - j;
        for k in 0..l1 {
            for i in 0..ido {
                let (a, b) = pm(cc[cc_idx(ido, ip, i, j, k)], cc[cc_idx(ido, ip, i, jc, k)]);
                ch[ch_idx(ido, l1, i, k, j)] = a;
                ch[ch_idx(ido, l1, i, k, jc)] = b;
            }
        }
    }
    for k in 0..l1 {
        for i in 0..ido {
            let mut tmp = ch[ch_idx(ido, l1, i, k, 0)];
            for j in 1..ipph {
                tmp = tmp.add(ch[ch_idx(ido, l1, i, k, j)]);
            }
            cc[ch_idx(ido, l1, i, k, 0)] = tmp;
        }
    }
    for l in 1..ipph {
        let lc = ip - l;
        for ik in 0..idl1 {
            let h0 = ch[ik];
            let h1 = ch[ik + idl1];
            let h2 = ch[ik + 2 * idl1];
            cc[ik + idl1 * l].r = wal[2 * l].r.mul_add(h2.r, wal[l].r.mul_add(h1.r, h0.r));
            cc[ik + idl1 * l].i = wal[2 * l].r.mul_add(h2.i, wal[l].r.mul_add(h1.i, h0.i));
            cc[ik + idl1 * lc].r = (-wal[l].i).mul_add(
                ch[ik + idl1 * (ip - 1)].i,
                -(wal[2 * l].i * ch[ik + idl1 * (ip - 2)].i),
            );
            cc[ik + idl1 * lc].i = wal[l].i.mul_add(
                ch[ik + idl1 * (ip - 1)].r,
                wal[2 * l].i * ch[ik + idl1 * (ip - 2)].r,
            );
        }
        let mut iwal = 2 * l;
        let mut j = 3usize;
        let mut jc = ip - 3;
        while j < ipph - 1 {
            iwal += l;
            if iwal > ip {
                iwal -= ip;
            }
            let xwal = wal[iwal];
            iwal += l;
            if iwal > ip {
                iwal -= ip;
            }
            let xwal2 = wal[iwal];
            for ik in 0..idl1 {
                cc[ik + idl1 * l].r += xwal
                    .r
                    .mul_add(ch[ik + idl1 * j].r, xwal2.r * ch[ik + idl1 * (j + 1)].r);
                cc[ik + idl1 * l].i += xwal
                    .r
                    .mul_add(ch[ik + idl1 * j].i, xwal2.r * ch[ik + idl1 * (j + 1)].i);
                cc[ik + idl1 * lc].r -= xwal
                    .i
                    .mul_add(ch[ik + idl1 * jc].i, xwal2.i * ch[ik + idl1 * (jc - 1)].i);
                cc[ik + idl1 * lc].i += xwal
                    .i
                    .mul_add(ch[ik + idl1 * jc].r, xwal2.i * ch[ik + idl1 * (jc - 1)].r);
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            iwal += l;
            if iwal > ip {
                iwal -= ip;
            }
            let xwal = wal[iwal];
            for ik in 0..idl1 {
                cc[ik + idl1 * l].r = xwal.r.mul_add(ch[ik + idl1 * j].r, cc[ik + idl1 * l].r);
                cc[ik + idl1 * l].i = xwal.r.mul_add(ch[ik + idl1 * j].i, cc[ik + idl1 * l].i);
                cc[ik + idl1 * lc].r =
                    (-xwal.i).mul_add(ch[ik + idl1 * jc].i, cc[ik + idl1 * lc].r);
                cc[ik + idl1 * lc].i = xwal.i.mul_add(ch[ik + idl1 * jc].r, cc[ik + idl1 * lc].i);
            }
            j += 1;
            jc -= 1;
        }
    }
    if ido == 1 {
        for j in 1..ipph {
            let jc = ip - j;
            for ik in 0..idl1 {
                let (a, b) = pm(cc[ik + idl1 * j], cc[ik + idl1 * jc]);
                cc[ik + idl1 * j] = a;
                cc[ik + idl1 * jc] = b;
            }
        }
    } else {
        for j in 1..ipph {
            let jc = ip - j;
            for k in 0..l1 {
                let (a, b) = pm(cc[ch_idx(ido, l1, 0, k, j)], cc[ch_idx(ido, l1, 0, k, jc)]);
                cc[ch_idx(ido, l1, 0, k, j)] = a;
                cc[ch_idx(ido, l1, 0, k, jc)] = b;
                for i in 1..ido {
                    let (x1, x2) = pm(cc[ch_idx(ido, l1, i, k, j)], cc[ch_idx(ido, l1, i, k, jc)]);
                    cc[ch_idx(ido, l1, i, k, j)] =
                        special_mul(x1, wa[(j - 1) * (ido - 1) + i - 1], fwd);
                    cc[ch_idx(ido, l1, i, k, jc)] =
                        special_mul(x2, wa[(jc - 1) * (ido - 1) + i - 1], fwd);
                }
            }
        }
    }
}

struct CfftPlan {
    length: usize,
    factors: Vec<Factor>,
}

impl CfftPlan {
    fn new(length: usize) -> Self {
        assert!(length != 0, "zero-length FFT requested");
        if length == 1 {
            return Self {
                length,
                factors: Vec::new(),
            };
        }
        let mut raw = Vec::new();
        let mut len = length;
        while len & 7 == 0 {
            raw.push(8);
            len >>= 3;
        }
        while len & 3 == 0 {
            raw.push(4);
            len >>= 2;
        }
        if len & 1 == 0 {
            len >>= 1;
            raw.push(2);
            let last = raw.len() - 1;
            raw.swap(0, last);
        }
        let mut divisor = 3usize;
        while divisor * divisor <= len {
            while len % divisor == 0 {
                raw.push(divisor);
                len /= divisor;
            }
            divisor += 2;
        }
        if len > 1 {
            raw.push(len);
        }

        let twiddle = Twiddles::new(length);
        let mut l1 = 1usize;
        let mut factors = Vec::with_capacity(raw.len());
        for ip in raw {
            let ido = length / (l1 * ip);
            let mut tw = Vec::with_capacity((ip - 1) * (ido.saturating_sub(1)));
            for j in 1..ip {
                for i in 1..ido {
                    tw.push(twiddle.get(j * l1 * i));
                }
            }
            let mut tws = Vec::new();
            if ip > 11 {
                for j in 0..ip {
                    tws.push(twiddle.get(j * l1 * ido));
                }
            }
            factors.push(Factor { fct: ip, tw, tws });
            l1 *= ip;
        }
        Self { length, factors }
    }

    fn exec(&self, c: &mut [C], fct: f64, fwd: bool) {
        if self.length == 1 {
            c[0] = c[0].mul(fct);
            return;
        }
        let mut ch = vec![C::default(); self.length];
        let mut primary = true;
        let mut l1 = 1usize;
        for factor in &self.factors {
            let ip = factor.fct;
            let ido = self.length / (ip * l1);
            if ip > 11 {
                if primary {
                    passg(ido, ip, l1, c, &mut ch, &factor.tw, &factor.tws, fwd);
                } else {
                    passg(ido, ip, l1, &mut ch, c, &factor.tw, &factor.tws, fwd);
                }
            } else {
                let run = |src: &[C], dst: &mut [C]| match ip {
                    2 => pass2(ido, l1, src, dst, &factor.tw, fwd),
                    3 => pass3(ido, l1, src, dst, &factor.tw, fwd),
                    4 => pass4(ido, l1, src, dst, &factor.tw, fwd),
                    5 => pass5(ido, l1, src, dst, &factor.tw, fwd),
                    7 => pass7(ido, l1, src, dst, &factor.tw, fwd),
                    8 => pass8(ido, l1, src, dst, &factor.tw, fwd),
                    11 => pass11(ido, l1, src, dst, &factor.tw, fwd),
                    _ => unreachable!(),
                };
                if primary {
                    run(c, &mut ch);
                } else {
                    run(&ch, c);
                }
                primary = !primary;
            }
            l1 *= ip;
        }
        if !primary {
            if fct != 1.0 {
                for i in 0..self.length {
                    c[i] = ch[i].mul(fct);
                }
            } else {
                c.copy_from_slice(&ch);
            }
        } else if fct != 1.0 {
            for value in c {
                *value = value.mul(fct);
            }
        }
    }
}

fn largest_prime_factor(mut n: usize) -> usize {
    let mut res = 1;
    while n & 1 == 0 {
        res = 2;
        n >>= 1;
    }
    let mut x = 3;
    while x * x <= n {
        while n % x == 0 {
            res = x;
            n /= x;
        }
        x += 2;
    }
    if n > 1 {
        res = n;
    }
    res
}

fn cost_guess(mut n: usize) -> f64 {
    let ni = n;
    let mut result = 0.0;
    while n & 1 == 0 {
        result += 2.0;
        n >>= 1;
    }
    let mut x = 3usize;
    while x * x <= n {
        while n % x == 0 {
            result += if x <= 5 { x as f64 } else { 1.1 * x as f64 };
            n /= x;
        }
        x += 2;
    }
    if n > 1 {
        result += if n <= 5 { n as f64 } else { 1.1 * n as f64 };
    }
    result * ni as f64
}

fn good_size_cmplx(n: usize) -> usize {
    if n <= 12 {
        return n;
    }
    let mut bestfac = 2 * n;
    let mut f11 = 1usize;
    while f11 < bestfac {
        let mut f117 = f11;
        while f117 < bestfac {
            let mut f1175 = f117;
            while f1175 < bestfac {
                let mut x = f1175;
                while x < n {
                    x *= 2;
                }
                loop {
                    if x < n {
                        x *= 3;
                    } else if x > n {
                        if x < bestfac {
                            bestfac = x;
                        }
                        if x & 1 != 0 {
                            break;
                        }
                        x >>= 1;
                    } else {
                        return n;
                    }
                }
                f1175 *= 5;
            }
            f117 *= 7;
        }
        f11 *= 11;
    }
    bestfac
}

struct FftBlue {
    n: usize,
    n2: usize,
    plan: CfftPlan,
    bk: Vec<C>,
    bkf: Vec<C>,
}

impl FftBlue {
    fn new(n: usize) -> Self {
        let n2 = good_size_cmplx(2 * n - 1);
        let plan = CfftPlan::new(n2);
        let tmp = Twiddles::new(2 * n);
        let mut bk = vec![C::default(); n];
        bk[0] = C::new(1.0, 0.0);
        let mut coeff = 0usize;
        for m in 1..n {
            coeff += 2 * m - 1;
            if coeff >= 2 * n {
                coeff -= 2 * n;
            }
            bk[m] = tmp.get(coeff);
        }
        let mut tbkf = vec![C::default(); n2];
        let xn2 = 1.0 / n2 as f64;
        tbkf[0] = bk[0].mul(xn2);
        for m in 1..n {
            tbkf[m] = bk[m].mul(xn2);
            tbkf[n2 - m] = tbkf[m];
        }
        for value in &mut tbkf[n..=n2 - n] {
            *value = C::new(0.0, 0.0);
        }
        plan.exec(&mut tbkf, 1.0, true);
        let bkf = tbkf[..n2 / 2 + 1].to_vec();
        Self {
            n,
            n2,
            plan,
            bk,
            bkf,
        }
    }

    fn exec(&self, c: &mut [C], fct: f64, fwd: bool) {
        let mut akf = vec![C::default(); self.n2];
        for m in 0..self.n {
            akf[m] = special_mul(c[m], self.bk[m], fwd);
        }
        let zero = akf[0].mul(0.0);
        for value in &mut akf[self.n..] {
            *value = zero;
        }
        self.plan.exec(&mut akf, 1.0, true);
        akf[0] = special_mul(akf[0], self.bkf[0], !fwd);
        for m in 1..(self.n2 + 1) / 2 {
            akf[m] = special_mul(akf[m], self.bkf[m], !fwd);
            akf[self.n2 - m] = special_mul(akf[self.n2 - m], self.bkf[m], !fwd);
        }
        if self.n2 & 1 == 0 {
            akf[self.n2 / 2] = special_mul(akf[self.n2 / 2], self.bkf[self.n2 / 2], !fwd);
        }
        self.plan.exec(&mut akf, 1.0, false);
        for m in 0..self.n {
            c[m] = special_mul(akf[m], self.bk[m], fwd).mul(fct);
        }
    }
}

enum PocketPlan {
    Pack(CfftPlan),
    Blue(Box<FftBlue>),
}

impl PocketPlan {
    fn new(length: usize) -> Self {
        let tmp = if length < 50 {
            0
        } else {
            largest_prime_factor(length)
        };
        if tmp * tmp <= length {
            return Self::Pack(CfftPlan::new(length));
        }
        let comp1 = cost_guess(length);
        let comp2 = 2.0 * cost_guess(good_size_cmplx(2 * length - 1)) * 1.5;
        if comp2 < comp1 {
            Self::Blue(Box::new(FftBlue::new(length)))
        } else {
            Self::Pack(CfftPlan::new(length))
        }
    }

    fn exec(&self, c: &mut [C], fct: f64, fwd: bool) {
        match self {
            Self::Pack(plan) => plan.exec(c, fct, fwd),
            Self::Blue(plan) => plan.exec(c, fct, fwd),
        }
    }
}

/// Complex-to-complex transform of one contiguous logical vector.
pub fn c2c(input: &[C64], forward: bool, scale: f64) -> Vec<C64> {
    let mut data: Vec<C> = input.iter().map(|z| C::new(z.re, z.im)).collect();
    PocketPlan::new(data.len()).exec(&mut data, scale, forward);
    data.into_iter().map(|z| C64::new(z.r, z.i)).collect()
}

#[inline]
fn mulpm(c: f64, d: f64, e: f64, f: f64) -> (f64, f64) {
    (c.mul_add(e, d * f), c.mul_add(f, -(d * e)))
}

fn radf2(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    let ci = |a, b, c| a + ido * (b + l1 * c);
    let hi = |a, b, c| a + ido * (b + 2 * c);
    for k in 0..l1 {
        let (a, b) = (
            cc[ci(0, k, 0)] + cc[ci(0, k, 1)],
            cc[ci(0, k, 0)] - cc[ci(0, k, 1)],
        );
        ch[hi(0, 0, k)] = a;
        ch[hi(ido - 1, 1, k)] = b;
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            ch[hi(0, 1, k)] = -cc[ci(ido - 1, k, 1)];
            ch[hi(ido - 1, 0, k)] = cc[ci(ido - 1, k, 0)];
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let (tr2, ti2) = mulpm(wa[i - 2], wa[i - 1], cc[ci(i - 1, k, 1)], cc[ci(i, k, 1)]);
            let (a, b) = (cc[ci(i - 1, k, 0)] + tr2, cc[ci(i - 1, k, 0)] - tr2);
            ch[hi(i - 1, 0, k)] = a;
            ch[hi(ic - 1, 1, k)] = b;
            let (a, b) = (ti2 + cc[ci(i, k, 0)], ti2 - cc[ci(i, k, 0)]);
            ch[hi(i, 0, k)] = a;
            ch[hi(ic, 1, k)] = b;
        }
    }
}

#[inline]
fn rearrange(rx: &mut f64, ix: &mut f64, ry: &mut f64, iy: &mut f64) {
    let t1 = *rx + *ry;
    let t2 = *ry - *rx;
    let t3 = *ix + *iy;
    let t4 = *ix - *iy;
    *rx = t1;
    *ix = t3;
    *ry = t4;
    *iy = t2;
}

fn radf3(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    const TAUR: f64 = -0.5;
    const TAUI: f64 = 0.8660254037844386467637231707529362_f64;
    let ci = |a, b, c| a + ido * (b + l1 * c);
    let hi = |a, b, c| a + ido * (b + 3 * c);
    for k in 0..l1 {
        let cr2 = cc[ci(0, k, 1)] + cc[ci(0, k, 2)];
        ch[hi(0, 0, k)] = cc[ci(0, k, 0)] + cr2;
        ch[hi(0, 2, k)] = TAUI * (cc[ci(0, k, 2)] - cc[ci(0, k, 1)]);
        ch[hi(ido - 1, 1, k)] = TAUR.mul_add(cr2, cc[ci(0, k, 0)]);
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let (mut dr2, mut di2) =
                mulpm(wa[i - 2], wa[i - 1], cc[ci(i - 1, k, 1)], cc[ci(i, k, 1)]);
            let o = ido - 1;
            let (mut dr3, mut di3) = mulpm(
                wa[o + i - 2],
                wa[o + i - 1],
                cc[ci(i - 1, k, 2)],
                cc[ci(i, k, 2)],
            );
            rearrange(&mut dr2, &mut di2, &mut dr3, &mut di3);
            ch[hi(i - 1, 0, k)] = cc[ci(i - 1, k, 0)] + dr2;
            ch[hi(i, 0, k)] = cc[ci(i, k, 0)] + di2;
            let tr2 = TAUR.mul_add(dr2, cc[ci(i - 1, k, 0)]);
            let ti2 = TAUR.mul_add(di2, cc[ci(i, k, 0)]);
            let tr3 = TAUI * dr3;
            let ti3 = TAUI * di3;
            ch[hi(i - 1, 2, k)] = tr2 + tr3;
            ch[hi(ic - 1, 1, k)] = tr2 - tr3;
            ch[hi(i, 2, k)] = ti3 + ti2;
            ch[hi(ic, 1, k)] = ti3 - ti2;
        }
    }
}

fn radf4(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    const H: f64 = 0.707106781186547524400844362104849_f64;
    let ci = |a, b, c| a + ido * (b + l1 * c);
    let hi = |a, b, c| a + ido * (b + 4 * c);
    for k in 0..l1 {
        let tr1 = cc[ci(0, k, 3)] + cc[ci(0, k, 1)];
        ch[hi(0, 2, k)] = cc[ci(0, k, 3)] - cc[ci(0, k, 1)];
        let tr2 = cc[ci(0, k, 0)] + cc[ci(0, k, 2)];
        ch[hi(ido - 1, 1, k)] = cc[ci(0, k, 0)] - cc[ci(0, k, 2)];
        ch[hi(0, 0, k)] = tr2 + tr1;
        ch[hi(ido - 1, 3, k)] = tr2 - tr1;
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            let ti1 = -H * (cc[ci(ido - 1, k, 1)] + cc[ci(ido - 1, k, 3)]);
            let tr1 = H * (cc[ci(ido - 1, k, 1)] - cc[ci(ido - 1, k, 3)]);
            ch[hi(ido - 1, 0, k)] = cc[ci(ido - 1, k, 0)] + tr1;
            ch[hi(ido - 1, 2, k)] = cc[ci(ido - 1, k, 0)] - tr1;
            ch[hi(0, 3, k)] = ti1 + cc[ci(ido - 1, k, 2)];
            ch[hi(0, 1, k)] = ti1 - cc[ci(ido - 1, k, 2)];
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let o = ido - 1;
            let (cr2, ci2) = mulpm(wa[i - 2], wa[i - 1], cc[ci(i - 1, k, 1)], cc[ci(i, k, 1)]);
            let (cr3, ci3) = mulpm(
                wa[o + i - 2],
                wa[o + i - 1],
                cc[ci(i - 1, k, 2)],
                cc[ci(i, k, 2)],
            );
            let (cr4, ci4) = mulpm(
                wa[2 * o + i - 2],
                wa[2 * o + i - 1],
                cc[ci(i - 1, k, 3)],
                cc[ci(i, k, 3)],
            );
            let tr1 = cr4 + cr2;
            let tr4 = cr4 - cr2;
            let ti1 = ci2 + ci4;
            let ti4 = ci2 - ci4;
            let tr2 = cc[ci(i - 1, k, 0)] + cr3;
            let tr3 = cc[ci(i - 1, k, 0)] - cr3;
            let ti2 = cc[ci(i, k, 0)] + ci3;
            let ti3 = cc[ci(i, k, 0)] - ci3;
            ch[hi(i - 1, 0, k)] = tr2 + tr1;
            ch[hi(ic - 1, 3, k)] = tr2 - tr1;
            ch[hi(i, 0, k)] = ti1 + ti2;
            ch[hi(ic, 3, k)] = ti1 - ti2;
            ch[hi(i - 1, 2, k)] = tr3 + ti4;
            ch[hi(ic - 1, 1, k)] = tr3 - ti4;
            ch[hi(i, 2, k)] = tr4 + ti3;
            ch[hi(ic, 1, k)] = tr4 - ti3;
        }
    }
}

fn radf5(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    const R1: f64 = 0.3090169943749474241022934171828191_f64;
    const I1: f64 = 0.9510565162951535721164393333793821_f64;
    const R2: f64 = -0.8090169943749474241022934171828191_f64;
    const I2: f64 = 0.5877852522924731291687059546390728_f64;
    let ci = |a, b, c| a + ido * (b + l1 * c);
    let hi = |a, b, c| a + ido * (b + 5 * c);
    for k in 0..l1 {
        let cr2 = cc[ci(0, k, 4)] + cc[ci(0, k, 1)];
        let ci5 = cc[ci(0, k, 4)] - cc[ci(0, k, 1)];
        let cr3 = cc[ci(0, k, 3)] + cc[ci(0, k, 2)];
        let ci4 = cc[ci(0, k, 3)] - cc[ci(0, k, 2)];
        ch[hi(0, 0, k)] = cc[ci(0, k, 0)] + cr2 + cr3;
        ch[hi(ido - 1, 1, k)] = R2.mul_add(cr3, R1.mul_add(cr2, cc[ci(0, k, 0)]));
        ch[hi(0, 2, k)] = I1.mul_add(ci5, I2 * ci4);
        ch[hi(ido - 1, 3, k)] = R1.mul_add(cr3, R2.mul_add(cr2, cc[ci(0, k, 0)]));
        ch[hi(0, 4, k)] = I2.mul_add(ci5, -(I1 * ci4));
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let o = ido - 1;
            let (mut dr2, mut di2) =
                mulpm(wa[i - 2], wa[i - 1], cc[ci(i - 1, k, 1)], cc[ci(i, k, 1)]);
            let (mut dr3, mut di3) = mulpm(
                wa[o + i - 2],
                wa[o + i - 1],
                cc[ci(i - 1, k, 2)],
                cc[ci(i, k, 2)],
            );
            let (mut dr4, mut di4) = mulpm(
                wa[2 * o + i - 2],
                wa[2 * o + i - 1],
                cc[ci(i - 1, k, 3)],
                cc[ci(i, k, 3)],
            );
            let (mut dr5, mut di5) = mulpm(
                wa[3 * o + i - 2],
                wa[3 * o + i - 1],
                cc[ci(i - 1, k, 4)],
                cc[ci(i, k, 4)],
            );
            rearrange(&mut dr2, &mut di2, &mut dr5, &mut di5);
            rearrange(&mut dr3, &mut di3, &mut dr4, &mut di4);
            ch[hi(i - 1, 0, k)] = cc[ci(i - 1, k, 0)] + dr2 + dr3;
            ch[hi(i, 0, k)] = cc[ci(i, k, 0)] + di2 + di3;
            let tr2 = R2.mul_add(dr3, R1.mul_add(dr2, cc[ci(i - 1, k, 0)]));
            let ti2 = R2.mul_add(di3, R1.mul_add(di2, cc[ci(i, k, 0)]));
            let tr3 = R1.mul_add(dr3, R2.mul_add(dr2, cc[ci(i - 1, k, 0)]));
            let ti3 = R1.mul_add(di3, R2.mul_add(di2, cc[ci(i, k, 0)]));
            let tr5 = I1.mul_add(dr5, I2 * dr4);
            let ti5 = I1.mul_add(di5, I2 * di4);
            let tr4 = I2.mul_add(dr5, -I1 * dr4);
            let ti4 = I2.mul_add(di5, -I1 * di4);
            ch[hi(i - 1, 2, k)] = tr2 + tr5;
            ch[hi(ic - 1, 1, k)] = tr2 - tr5;
            ch[hi(i, 2, k)] = ti5 + ti2;
            ch[hi(ic, 1, k)] = ti5 - ti2;
            ch[hi(i - 1, 4, k)] = tr3 + tr4;
            ch[hi(ic - 1, 3, k)] = tr3 - tr4;
            ch[hi(i, 4, k)] = ti4 + ti3;
            ch[hi(ic, 3, k)] = ti4 - ti3;
        }
    }
}

fn radb2(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    let ci = |a, b, c| a + ido * (b + 2 * c);
    let hi = |a, b, c| a + ido * (b + l1 * c);
    for k in 0..l1 {
        ch[hi(0, k, 0)] = cc[ci(0, 0, k)] + cc[ci(ido - 1, 1, k)];
        ch[hi(0, k, 1)] = cc[ci(0, 0, k)] - cc[ci(ido - 1, 1, k)];
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            ch[hi(ido - 1, k, 0)] = 2.0 * cc[ci(ido - 1, 0, k)];
            ch[hi(ido - 1, k, 1)] = -2.0 * cc[ci(0, 1, k)];
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            ch[hi(i - 1, k, 0)] = cc[ci(i - 1, 0, k)] + cc[ci(ic - 1, 1, k)];
            let tr2 = cc[ci(i - 1, 0, k)] - cc[ci(ic - 1, 1, k)];
            let ti2 = cc[ci(i, 0, k)] + cc[ci(ic, 1, k)];
            ch[hi(i, k, 0)] = cc[ci(i, 0, k)] - cc[ci(ic, 1, k)];
            let (a, b) = mulpm(wa[i - 2], wa[i - 1], ti2, tr2);
            ch[hi(i, k, 1)] = a;
            ch[hi(i - 1, k, 1)] = b;
        }
    }
}

fn radb3(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    const R: f64 = -0.5;
    const I: f64 = 0.8660254037844386467637231707529362_f64;
    let ci = |a, b, c| a + ido * (b + 3 * c);
    let hi = |a, b, c| a + ido * (b + l1 * c);
    for k in 0..l1 {
        let tr2 = 2.0 * cc[ci(ido - 1, 1, k)];
        let cr2 = R.mul_add(tr2, cc[ci(0, 0, k)]);
        ch[hi(0, k, 0)] = cc[ci(0, 0, k)] + tr2;
        let ci3 = 2.0 * I * cc[ci(0, 2, k)];
        ch[hi(0, k, 2)] = cr2 + ci3;
        ch[hi(0, k, 1)] = cr2 - ci3;
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let tr2 = cc[ci(i - 1, 2, k)] + cc[ci(ic - 1, 1, k)];
            let ti2 = cc[ci(i, 2, k)] - cc[ci(ic, 1, k)];
            let cr2 = R.mul_add(tr2, cc[ci(i - 1, 0, k)]);
            let ci2 = R.mul_add(ti2, cc[ci(i, 0, k)]);
            ch[hi(i - 1, k, 0)] = cc[ci(i - 1, 0, k)] + tr2;
            ch[hi(i, k, 0)] = cc[ci(i, 0, k)] + ti2;
            let cr3 = I * (cc[ci(i - 1, 2, k)] - cc[ci(ic - 1, 1, k)]);
            let ci3 = I * (cc[ci(i, 2, k)] + cc[ci(ic, 1, k)]);
            let dr3 = cr2 + ci3;
            let dr2 = cr2 - ci3;
            let di2 = ci2 + cr3;
            let di3 = ci2 - cr3;
            let (a, b) = mulpm(wa[i - 2], wa[i - 1], di2, dr2);
            ch[hi(i, k, 1)] = a;
            ch[hi(i - 1, k, 1)] = b;
            let o = ido - 1;
            let (a, b) = mulpm(wa[o + i - 2], wa[o + i - 1], di3, dr3);
            ch[hi(i, k, 2)] = a;
            ch[hi(i - 1, k, 2)] = b;
        }
    }
}

fn radb4(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    const S: f64 = 1.414213562373095048801688724209698_f64;
    let ci = |a, b, c| a + ido * (b + 4 * c);
    let hi = |a, b, c| a + ido * (b + l1 * c);
    for k in 0..l1 {
        let tr2 = cc[ci(0, 0, k)] + cc[ci(ido - 1, 3, k)];
        let tr1 = cc[ci(0, 0, k)] - cc[ci(ido - 1, 3, k)];
        let tr3 = 2.0 * cc[ci(ido - 1, 1, k)];
        let tr4 = 2.0 * cc[ci(0, 2, k)];
        ch[hi(0, k, 0)] = tr2 + tr3;
        ch[hi(0, k, 2)] = tr2 - tr3;
        ch[hi(0, k, 3)] = tr1 + tr4;
        ch[hi(0, k, 1)] = tr1 - tr4;
    }
    if ido & 1 == 0 {
        for k in 0..l1 {
            let ti1 = cc[ci(0, 3, k)] + cc[ci(0, 1, k)];
            let ti2 = cc[ci(0, 3, k)] - cc[ci(0, 1, k)];
            let tr2 = cc[ci(ido - 1, 0, k)] + cc[ci(ido - 1, 2, k)];
            let tr1 = cc[ci(ido - 1, 0, k)] - cc[ci(ido - 1, 2, k)];
            ch[hi(ido - 1, k, 0)] = tr2 + tr2;
            ch[hi(ido - 1, k, 1)] = S * (tr1 - ti1);
            ch[hi(ido - 1, k, 2)] = ti2 + ti2;
            ch[hi(ido - 1, k, 3)] = -S * (tr1 + ti1);
        }
    }
    if ido <= 2 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let tr2 = cc[ci(i - 1, 0, k)] + cc[ci(ic - 1, 3, k)];
            let tr1 = cc[ci(i - 1, 0, k)] - cc[ci(ic - 1, 3, k)];
            let ti1 = cc[ci(i, 0, k)] + cc[ci(ic, 3, k)];
            let ti2 = cc[ci(i, 0, k)] - cc[ci(ic, 3, k)];
            let tr4 = cc[ci(i, 2, k)] + cc[ci(ic, 1, k)];
            let ti3 = cc[ci(i, 2, k)] - cc[ci(ic, 1, k)];
            let tr3 = cc[ci(i - 1, 2, k)] + cc[ci(ic - 1, 1, k)];
            let ti4 = cc[ci(i - 1, 2, k)] - cc[ci(ic - 1, 1, k)];
            ch[hi(i - 1, k, 0)] = tr2 + tr3;
            let cr3 = tr2 - tr3;
            ch[hi(i, k, 0)] = ti2 + ti3;
            let ci3 = ti2 - ti3;
            let cr4 = tr1 + tr4;
            let cr2 = tr1 - tr4;
            let ci2 = ti1 + ti4;
            let ci4 = ti1 - ti4;
            let o = ido - 1;
            for (q, (ciq, crq)) in [(0, (ci2, cr2)), (1, (ci3, cr3)), (2, (ci4, cr4))] {
                let (a, b) = mulpm(wa[q * o + i - 2], wa[q * o + i - 1], ciq, crq);
                ch[hi(i, k, q + 1)] = a;
                ch[hi(i - 1, k, q + 1)] = b;
            }
        }
    }
}

fn radb5(ido: usize, l1: usize, cc: &[f64], ch: &mut [f64], wa: &[f64]) {
    const R1: f64 = 0.3090169943749474241022934171828191_f64;
    const I1: f64 = 0.9510565162951535721164393333793821_f64;
    const R2: f64 = -0.8090169943749474241022934171828191_f64;
    const I2: f64 = 0.5877852522924731291687059546390728_f64;
    let ci = |a, b, c| a + ido * (b + 5 * c);
    let hi = |a, b, c| a + ido * (b + l1 * c);
    for k in 0..l1 {
        let ti5 = 2.0 * cc[ci(0, 2, k)];
        let ti4 = 2.0 * cc[ci(0, 4, k)];
        let tr2 = 2.0 * cc[ci(ido - 1, 1, k)];
        let tr3 = 2.0 * cc[ci(ido - 1, 3, k)];
        ch[hi(0, k, 0)] = cc[ci(0, 0, k)] + tr2 + tr3;
        let cr2 = R2.mul_add(tr3, R1.mul_add(tr2, cc[ci(0, 0, k)]));
        let cr3 = R1.mul_add(tr3, R2.mul_add(tr2, cc[ci(0, 0, k)]));
        let (ci5, ci4) = mulpm(ti5, ti4, I1, I2);
        ch[hi(0, k, 4)] = cr2 + ci5;
        ch[hi(0, k, 1)] = cr2 - ci5;
        ch[hi(0, k, 3)] = cr3 + ci4;
        ch[hi(0, k, 2)] = cr3 - ci4;
    }
    if ido == 1 {
        return;
    }
    for k in 0..l1 {
        for i in (2..ido).step_by(2) {
            let ic = ido - i;
            let tr2 = cc[ci(i - 1, 2, k)] + cc[ci(ic - 1, 1, k)];
            let tr5 = cc[ci(i - 1, 2, k)] - cc[ci(ic - 1, 1, k)];
            let ti5 = cc[ci(i, 2, k)] + cc[ci(ic, 1, k)];
            let ti2 = cc[ci(i, 2, k)] - cc[ci(ic, 1, k)];
            let tr3 = cc[ci(i - 1, 4, k)] + cc[ci(ic - 1, 3, k)];
            let tr4 = cc[ci(i - 1, 4, k)] - cc[ci(ic - 1, 3, k)];
            let ti4 = cc[ci(i, 4, k)] + cc[ci(ic, 3, k)];
            let ti3 = cc[ci(i, 4, k)] - cc[ci(ic, 3, k)];
            ch[hi(i - 1, k, 0)] = cc[ci(i - 1, 0, k)] + tr2 + tr3;
            ch[hi(i, k, 0)] = cc[ci(i, 0, k)] + ti2 + ti3;
            let cr2 = R2.mul_add(tr3, R1.mul_add(tr2, cc[ci(i - 1, 0, k)]));
            let ci2 = R2.mul_add(ti3, R1.mul_add(ti2, cc[ci(i, 0, k)]));
            let cr3 = R1.mul_add(tr3, R2.mul_add(tr2, cc[ci(i - 1, 0, k)]));
            let ci3 = R1.mul_add(ti3, R2.mul_add(ti2, cc[ci(i, 0, k)]));
            let (cr5, cr4) = mulpm(tr5, tr4, I1, I2);
            let (ci5, ci4) = mulpm(ti5, ti4, I1, I2);
            let dr4 = cr3 + ci4;
            let dr3 = cr3 - ci4;
            let di3 = ci3 + cr4;
            let di4 = ci3 - cr4;
            let dr5 = cr2 + ci5;
            let dr2 = cr2 - ci5;
            let di2 = ci2 + cr5;
            let di5 = ci2 - cr5;
            let o = ido - 1;
            for (q, (di, dr)) in [
                (0, (di2, dr2)),
                (1, (di3, dr3)),
                (2, (di4, dr4)),
                (3, (di5, dr5)),
            ] {
                let (a, b) = mulpm(wa[q * o + i - 2], wa[q * o + i - 1], di, dr);
                ch[hi(i, k, q + 1)] = a;
                ch[hi(i - 1, k, q + 1)] = b;
            }
        }
    }
}

fn radfg(ido: usize, ip: usize, l1: usize, cc: &mut [f64], ch: &mut [f64], wa: &[f64], cs: &[f64]) {
    let ipph = (ip + 1) / 2;
    let idl1 = ido * l1;
    let c1 = |a, b, c| a + ido * (b + l1 * c);
    let cpack = |a, b, c| a + ido * (b + ip * c);
    if ido > 1 {
        for j in 1..ipph {
            let jc = ip - j;
            let is = (j - 1) * (ido - 1);
            let is2 = (jc - 1) * (ido - 1);
            for k in 0..l1 {
                let mut q = is;
                let mut q2 = is2;
                for i in (1..=ido - 2).step_by(2) {
                    let t1 = cc[c1(i, k, j)];
                    let t2 = cc[c1(i + 1, k, j)];
                    let t3 = cc[c1(i, k, jc)];
                    let t4 = cc[c1(i + 1, k, jc)];
                    let x1 = wa[q].mul_add(t1, wa[q + 1] * t2);
                    let x2 = wa[q].mul_add(t2, -wa[q + 1] * t1);
                    let x3 = wa[q2].mul_add(t3, wa[q2 + 1] * t4);
                    let x4 = wa[q2].mul_add(t4, -wa[q2 + 1] * t3);
                    cc[c1(i, k, j)] = x3 + x1;
                    cc[c1(i + 1, k, jc)] = x3 - x1;
                    cc[c1(i + 1, k, j)] = x2 + x4;
                    cc[c1(i, k, jc)] = x2 - x4;
                    q += 2;
                    q2 += 2;
                }
            }
        }
    }
    for j in 1..ipph {
        let jc = ip - j;
        for k in 0..l1 {
            let a = c1(0, k, jc);
            let b = c1(0, k, j);
            let t = cc[a];
            cc[a] -= cc[b];
            cc[b] += t;
        }
    }
    for l in 1..ipph {
        let lc = ip - l;
        for ik in 0..idl1 {
            ch[ik + idl1 * l] =
                cs[4 * l].mul_add(cc[ik + 2 * idl1], cs[2 * l].mul_add(cc[ik + idl1], cc[ik]));
            ch[ik + idl1 * lc] = cs[2 * l + 1].mul_add(
                cc[ik + idl1 * (ip - 1)],
                cs[4 * l + 1] * cc[ik + idl1 * (ip - 2)],
            );
        }
        let mut iang = 2 * l;
        let mut j = 3usize;
        let mut jc = ip - 3;
        while j < ipph - 3 {
            let mut ar = [0.; 4];
            let mut ai = [0.; 4];
            for q in 0..4 {
                iang += l;
                if iang >= ip {
                    iang -= ip;
                }
                ar[q] = cs[2 * iang];
                ai[q] = cs[2 * iang + 1];
            }
            for ik in 0..idl1 {
                let rhs = ar[3].mul_add(
                    cc[ik + idl1 * (j + 3)],
                    ar[2].mul_add(
                        cc[ik + idl1 * (j + 2)],
                        ar[0].mul_add(cc[ik + idl1 * j], ar[1] * cc[ik + idl1 * (j + 1)]),
                    ),
                );
                ch[ik + idl1 * l] += rhs;
                let rhs = ai[3].mul_add(
                    cc[ik + idl1 * (jc - 3)],
                    ai[2].mul_add(
                        cc[ik + idl1 * (jc - 2)],
                        ai[0].mul_add(cc[ik + idl1 * jc], ai[1] * cc[ik + idl1 * (jc - 1)]),
                    ),
                );
                ch[ik + idl1 * lc] += rhs;
            }
            j += 4;
            jc -= 4;
        }
        while j < ipph - 1 {
            iang += l;
            if iang >= ip {
                iang -= ip;
            }
            let ar1 = cs[2 * iang];
            let ai1 = cs[2 * iang + 1];
            iang += l;
            if iang >= ip {
                iang -= ip;
            }
            let ar2 = cs[2 * iang];
            let ai2 = cs[2 * iang + 1];
            for ik in 0..idl1 {
                ch[ik + idl1 * l] += ar1.mul_add(cc[ik + idl1 * j], ar2 * cc[ik + idl1 * (j + 1)]);
                ch[ik + idl1 * lc] +=
                    ai1.mul_add(cc[ik + idl1 * jc], ai2 * cc[ik + idl1 * (jc - 1)]);
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            iang += l;
            if iang >= ip {
                iang -= ip;
            }
            let ar = cs[2 * iang];
            let ai = cs[2 * iang + 1];
            for ik in 0..idl1 {
                ch[ik + idl1 * l] = ar.mul_add(cc[ik + idl1 * j], ch[ik + idl1 * l]);
                ch[ik + idl1 * lc] = ai.mul_add(cc[ik + idl1 * jc], ch[ik + idl1 * lc]);
            }
            j += 1;
            jc -= 1;
        }
    }
    for ik in 0..idl1 {
        ch[ik] = cc[ik];
    }
    for j in 1..ipph {
        for ik in 0..idl1 {
            ch[ik] += cc[ik + idl1 * j];
        }
    }
    for k in 0..l1 {
        for i in 0..ido {
            cc[cpack(i, 0, k)] = ch[c1(i, k, 0)];
        }
    }
    for j in 1..ipph {
        let jc = ip - j;
        let j2 = 2 * j - 1;
        for k in 0..l1 {
            cc[cpack(ido - 1, j2, k)] = ch[c1(0, k, j)];
            cc[cpack(0, j2 + 1, k)] = ch[c1(0, k, jc)];
        }
    }
    if ido == 1 {
        return;
    }
    for j in 1..ipph {
        let jc = ip - j;
        let j2 = 2 * j - 1;
        for k in 0..l1 {
            let mut i = 1usize;
            let mut ic = ido - i - 2;
            while i <= ido - 2 {
                cc[cpack(i, j2 + 1, k)] = ch[c1(i, k, j)] + ch[c1(i, k, jc)];
                cc[cpack(ic, j2, k)] = ch[c1(i, k, j)] - ch[c1(i, k, jc)];
                cc[cpack(i + 1, j2 + 1, k)] = ch[c1(i + 1, k, j)] + ch[c1(i + 1, k, jc)];
                cc[cpack(ic + 1, j2, k)] = ch[c1(i + 1, k, jc)] - ch[c1(i + 1, k, j)];
                i += 2;
                if ic >= 2 {
                    ic -= 2;
                }
            }
        }
    }
}

fn radbg(ido: usize, ip: usize, l1: usize, cc: &mut [f64], ch: &mut [f64], wa: &[f64], cs: &[f64]) {
    let ipph = (ip + 1) / 2;
    let idl1 = ido * l1;
    let cp = |a, b, c| a + ido * (b + ip * c);
    let c1 = |a, b, c| a + ido * (b + l1 * c);
    for k in 0..l1 {
        for i in 0..ido {
            ch[c1(i, k, 0)] = cc[cp(i, 0, k)];
        }
    }
    for j in 1..ipph {
        let jc = ip - j;
        let j2 = 2 * j - 1;
        for k in 0..l1 {
            ch[c1(0, k, j)] = 2.0 * cc[cp(ido - 1, j2, k)];
            ch[c1(0, k, jc)] = 2.0 * cc[cp(0, j2 + 1, k)];
        }
    }
    if ido != 1 {
        for j in 1..ipph {
            let jc = ip - j;
            let j2 = 2 * j - 1;
            for k in 0..l1 {
                let mut i = 1usize;
                let mut ic = ido - i - 2;
                while i <= ido - 2 {
                    ch[c1(i, k, j)] = cc[cp(i, j2 + 1, k)] + cc[cp(ic, j2, k)];
                    ch[c1(i, k, jc)] = cc[cp(i, j2 + 1, k)] - cc[cp(ic, j2, k)];
                    ch[c1(i + 1, k, j)] = cc[cp(i + 1, j2 + 1, k)] - cc[cp(ic + 1, j2, k)];
                    ch[c1(i + 1, k, jc)] = cc[cp(i + 1, j2 + 1, k)] + cc[cp(ic + 1, j2, k)];
                    i += 2;
                    if ic >= 2 {
                        ic -= 2;
                    }
                }
            }
        }
    }
    for l in 1..ipph {
        let lc = ip - l;
        for ik in 0..idl1 {
            cc[ik + idl1 * l] =
                cs[4 * l].mul_add(ch[ik + 2 * idl1], cs[2 * l].mul_add(ch[ik + idl1], ch[ik]));
            cc[ik + idl1 * lc] = cs[2 * l + 1].mul_add(
                ch[ik + idl1 * (ip - 1)],
                cs[4 * l + 1] * ch[ik + idl1 * (ip - 2)],
            );
        }
        let mut iang = 2 * l;
        let mut j = 3usize;
        let mut jc = ip - 3;
        while j < ipph - 3 {
            let mut ar = [0.; 4];
            let mut ai = [0.; 4];
            for q in 0..4 {
                iang += l;
                if iang > ip {
                    iang -= ip;
                }
                ar[q] = cs[2 * iang];
                ai[q] = cs[2 * iang + 1];
            }
            for ik in 0..idl1 {
                cc[ik + idl1 * l] += ar[3].mul_add(
                    ch[ik + idl1 * (j + 3)],
                    ar[2].mul_add(
                        ch[ik + idl1 * (j + 2)],
                        ar[0].mul_add(ch[ik + idl1 * j], ar[1] * ch[ik + idl1 * (j + 1)]),
                    ),
                );
                cc[ik + idl1 * lc] += ai[3].mul_add(
                    ch[ik + idl1 * (jc - 3)],
                    ai[2].mul_add(
                        ch[ik + idl1 * (jc - 2)],
                        ai[0].mul_add(ch[ik + idl1 * jc], ai[1] * ch[ik + idl1 * (jc - 1)]),
                    ),
                );
            }
            j += 4;
            jc -= 4;
        }
        while j < ipph - 1 {
            iang += l;
            if iang > ip {
                iang -= ip;
            }
            let ar1 = cs[2 * iang];
            let ai1 = cs[2 * iang + 1];
            iang += l;
            if iang > ip {
                iang -= ip;
            }
            let ar2 = cs[2 * iang];
            let ai2 = cs[2 * iang + 1];
            for ik in 0..idl1 {
                cc[ik + idl1 * l] += ar1.mul_add(ch[ik + idl1 * j], ar2 * ch[ik + idl1 * (j + 1)]);
                cc[ik + idl1 * lc] +=
                    ai1.mul_add(ch[ik + idl1 * jc], ai2 * ch[ik + idl1 * (jc - 1)]);
            }
            j += 2;
            jc -= 2;
        }
        while j < ipph {
            iang += l;
            if iang > ip {
                iang -= ip;
            }
            let ar = cs[2 * iang];
            let ai = cs[2 * iang + 1];
            for ik in 0..idl1 {
                cc[ik + idl1 * l] = ar.mul_add(ch[ik + idl1 * j], cc[ik + idl1 * l]);
                cc[ik + idl1 * lc] = ai.mul_add(ch[ik + idl1 * jc], cc[ik + idl1 * lc]);
            }
            j += 1;
            jc -= 1;
        }
    }
    for j in 1..ipph {
        for ik in 0..idl1 {
            ch[ik] += ch[ik + idl1 * j];
        }
    }
    for j in 1..ipph {
        let jc = ip - j;
        for k in 0..l1 {
            ch[c1(0, k, jc)] = cc[c1(0, k, j)] + cc[c1(0, k, jc)];
            ch[c1(0, k, j)] = cc[c1(0, k, j)] - cc[c1(0, k, jc)];
        }
    }
    if ido == 1 {
        return;
    }
    for j in 1..ipph {
        let jc = ip - j;
        for k in 0..l1 {
            for i in (1..=ido - 2).step_by(2) {
                ch[c1(i, k, j)] = cc[c1(i, k, j)] - cc[c1(i + 1, k, jc)];
                ch[c1(i, k, jc)] = cc[c1(i, k, j)] + cc[c1(i + 1, k, jc)];
                ch[c1(i + 1, k, j)] = cc[c1(i + 1, k, j)] + cc[c1(i, k, jc)];
                ch[c1(i + 1, k, jc)] = cc[c1(i + 1, k, j)] - cc[c1(i, k, jc)];
            }
        }
    }
    for j in 1..ip {
        let is = (j - 1) * (ido - 1);
        for k in 0..l1 {
            let mut q = is;
            for i in (1..=ido - 2).step_by(2) {
                let t1 = ch[c1(i, k, j)];
                let t2 = ch[c1(i + 1, k, j)];
                ch[c1(i, k, j)] = wa[q].mul_add(t1, -wa[q + 1] * t2);
                ch[c1(i + 1, k, j)] = wa[q].mul_add(t2, wa[q + 1] * t1);
                q += 2;
            }
        }
    }
}

struct RealFactor {
    fct: usize,
    tw: Vec<f64>,
    tws: Vec<f64>,
}
struct RealPack {
    length: usize,
    factors: Vec<RealFactor>,
}

impl RealPack {
    fn new(length: usize) -> Self {
        if length == 1 {
            return Self {
                length,
                factors: Vec::new(),
            };
        }
        let mut raw = Vec::new();
        let mut len = length;
        while len % 4 == 0 {
            raw.push(4);
            len >>= 2;
        }
        if len % 2 == 0 {
            len >>= 1;
            raw.push(2);
            let q = raw.len() - 1;
            raw.swap(0, q);
        }
        let mut d = 3;
        while d * d <= len {
            while len % d == 0 {
                raw.push(d);
                len /= d;
            }
            d += 2;
        }
        if len > 1 {
            raw.push(len);
        }
        let roots = Twiddles::new(length);
        let nf = raw.len();
        let mut l1 = 1;
        let mut factors = Vec::with_capacity(nf);
        for (k, ip) in raw.into_iter().enumerate() {
            let ido = length / (l1 * ip);
            let mut tw = vec![0.; (ip - 1) * (ido.saturating_sub(1))];
            if k < nf - 1 {
                for j in 1..ip {
                    for i in 1..=(ido - 1) / 2 {
                        let z = roots.get(j * l1 * i);
                        let q = (j - 1) * (ido - 1) + 2 * i - 2;
                        tw[q] = z.r;
                        tw[q + 1] = z.i;
                    }
                }
            }
            let mut tws = Vec::new();
            if ip > 5 {
                tws = vec![0.; 2 * ip];
                tws[0] = 1.;
                for i in (2..=2 * ip - 2).step_by(2) {
                    let ic = 2 * ip - i;
                    let z = roots.get((i / 2) * (length / ip));
                    tws[i] = z.r;
                    tws[i + 1] = z.i;
                    tws[ic] = z.r;
                    tws[ic + 1] = -z.i;
                }
            }
            factors.push(RealFactor { fct: ip, tw, tws });
            l1 *= ip;
        }
        Self { length, factors }
    }
    fn exec(&self, c: &mut [f64], fct: f64, forward: bool) {
        if self.length == 1 {
            c[0] *= fct;
            return;
        }
        let mut ch = vec![0.; self.length];
        let mut primary = true;
        if forward {
            let mut l1 = self.length;
            for factor in self.factors.iter().rev() {
                let ip = factor.fct;
                let ido = self.length / l1;
                l1 /= ip;
                if ip > 5 {
                    if primary {
                        radfg(ido, ip, l1, c, &mut ch, &factor.tw, &factor.tws)
                    } else {
                        radfg(ido, ip, l1, &mut ch, c, &factor.tw, &factor.tws)
                    }
                } else {
                    let run = |src: &[f64], dst: &mut [f64]| match ip {
                        2 => radf2(ido, l1, src, dst, &factor.tw),
                        3 => radf3(ido, l1, src, dst, &factor.tw),
                        4 => radf4(ido, l1, src, dst, &factor.tw),
                        5 => radf5(ido, l1, src, dst, &factor.tw),
                        _ => unreachable!(),
                    };
                    if primary {
                        run(c, &mut ch)
                    } else {
                        run(&ch, c)
                    }
                    primary = !primary;
                }
            }
        } else {
            let mut l1 = 1;
            for factor in &self.factors {
                let ip = factor.fct;
                let ido = self.length / (ip * l1);
                if ip > 5 {
                    if primary {
                        radbg(ido, ip, l1, c, &mut ch, &factor.tw, &factor.tws)
                    } else {
                        radbg(ido, ip, l1, &mut ch, c, &factor.tw, &factor.tws)
                    }
                } else {
                    let run = |src: &[f64], dst: &mut [f64]| match ip {
                        2 => radb2(ido, l1, src, dst, &factor.tw),
                        3 => radb3(ido, l1, src, dst, &factor.tw),
                        4 => radb4(ido, l1, src, dst, &factor.tw),
                        5 => radb5(ido, l1, src, dst, &factor.tw),
                        _ => unreachable!(),
                    };
                    if primary {
                        run(c, &mut ch)
                    } else {
                        run(&ch, c)
                    }
                }
                primary = !primary;
                l1 *= ip;
            }
        }
        if !primary {
            if fct != 1. {
                for i in 0..self.length {
                    c[i] = fct * ch[i];
                }
            } else {
                c.copy_from_slice(&ch);
            }
        } else if fct != 1. {
            for x in c {
                *x *= fct;
            }
        }
    }
}

impl FftBlue {
    fn exec_real(&self, c: &mut [f64], fct: f64, forward: bool) {
        let mut tmp = vec![C::default(); self.n];
        if forward {
            for m in 0..self.n {
                tmp[m] = C::new(c[m], 0.0 * c[0]);
            }
            self.exec(&mut tmp, fct, true);
            c[0] = tmp[0].r;
            for p in 1..self.n {
                let m = (p + 1) / 2;
                c[p] = if p & 1 == 1 { tmp[m].r } else { tmp[m].i };
            }
        } else {
            tmp[0] = C::new(c[0], 0.0 * c[0]);
            for p in 1..self.n {
                let m = (p + 1) / 2;
                if p & 1 == 1 {
                    tmp[m].r = c[p];
                } else {
                    tmp[m].i = c[p];
                }
            }
            if self.n & 1 == 0 {
                tmp[self.n / 2].i = 0.0 * c[0];
            }
            for m in 1..(self.n + 1) / 2 {
                tmp[self.n - m] = C::new(tmp[m].r, -tmp[m].i);
            }
            self.exec(&mut tmp, fct, false);
            for m in 0..self.n {
                c[m] = tmp[m].r;
            }
        }
    }
}

enum RealPlan {
    Pack(RealPack),
    Blue(Box<FftBlue>),
}
impl RealPlan {
    fn new(n: usize) -> Self {
        let tmp = if n < 50 { 0 } else { largest_prime_factor(n) };
        if tmp * tmp <= n {
            return Self::Pack(RealPack::new(n));
        }
        let c1 = 0.5 * cost_guess(n);
        let c2 = 2.0 * cost_guess(good_size_cmplx(2 * n - 1)) * 1.5;
        if c2 < c1 {
            Self::Blue(Box::new(FftBlue::new(n)))
        } else {
            Self::Pack(RealPack::new(n))
        }
    }
    fn exec(&self, c: &mut [f64], fct: f64, forward: bool) {
        match self {
            Self::Pack(p) => p.exec(c, fct, forward),
            Self::Blue(p) => p.exec_real(c, fct, forward),
        }
    }
}

pub fn r2c(input: &[f64], scale: f64) -> Vec<C64> {
    let n = input.len();
    let mut data = input.to_vec();
    RealPlan::new(n).exec(&mut data, scale, true);
    let mut out = vec![C64::new(0., 0.); n / 2 + 1];
    out[0] = C64::new(data[0], 0.);
    for (k, z) in out.iter_mut().enumerate().skip(1) {
        z.re = data[2 * k - 1];
        if 2 * k < n {
            z.im = data[2 * k];
        }
    }
    out
}
pub fn c2r(input: &[C64], n: usize, scale: f64) -> Vec<f64> {
    let mut data = vec![0.; n];
    if let Some(z) = input.first() {
        data[0] = z.re;
    }
    for (k, z) in input.iter().enumerate().skip(1).take(n / 2) {
        data[2 * k - 1] = z.re;
        if 2 * k < n {
            data[2 * k] = z.im;
        }
    }
    if n & 1 == 0 && n / 2 < input.len() {
        data[n - 1] = input[n / 2].re;
    }
    RealPlan::new(n).exec(&mut data, scale, false);
    data
}

fn decode_batch(mut flat: usize, shape: &[isize], axis: usize, index: &mut [isize]) {
    for ax in (0..shape.len()).rev() {
        if ax == axis {
            index[ax] = 0;
            continue;
        }
        let dim = shape[ax] as usize;
        index[ax] = (flat % dim) as isize;
        flat /= dim;
    }
}

fn scale_f32(scale: f64, n: usize) -> f32 {
    if scale == 1.0 {
        1.0
    } else if scale == 1.0 / n as f64 {
        1.0f32 / n as f32
    } else {
        1.0f32 / (n as f32).sqrt()
    }
}

/// Apply a complex transform independently along one ndarray axis.
pub fn c2c_axis(
    input: &NdArray,
    n: usize,
    axis: usize,
    forward: bool,
    scale: f64,
    out_dtype: DType,
) -> Result<NdArray> {
    if n == 0 {
        return Err(Error::ValueError(
            "Invalid number of FFT data points (0) specified.".into(),
        ));
    }
    if axis >= input.ndim() {
        return Err(Error::IndexError("tuple index out of range".into()));
    }
    let mut shape = input.shape.clone();
    shape[axis] = n as isize;
    let output = NdArray::zeros(shape.clone(), out_dtype)?;
    let batches = shape
        .iter()
        .enumerate()
        .filter(|(ax, _)| *ax != axis)
        .map(|(_, &d)| d as usize)
        .product::<usize>();
    let take = n.min(input.shape[axis] as usize);
    let scaled_single = out_dtype == DType::C64 && scale != 1.0;
    let mut src_index = vec![0isize; input.ndim()];
    let mut dst_index = vec![0isize; input.ndim()];
    for batch in 0..batches {
        decode_batch(batch, &shape, axis, &mut src_index);
        dst_index.copy_from_slice(&src_index);
        if out_dtype == DType::C160 {
            let mut line = vec![C160::ZERO; n];
            for (j, value) in line.iter_mut().take(take).enumerate() {
                src_index[axis] = j as isize;
                *value = match input.read_at(input.byte_index(&src_index)) {
                    Scalar::Complex160(value) => value,
                    Scalar::Float80(value) => C160 { re: value, im: F80::ZERO },
                    Scalar::Complex(value) => C160 {
                        re: F80::from_f64(value.re),
                        im: F80::from_f64(value.im),
                    },
                    other => C160 { re: F80::from_f64(other.as_f64()), im: F80::ZERO },
                };
            }
            for (j, value) in c2c_f80(&line, forward, scale_f80(scale, n))
                .into_iter()
                .enumerate()
            {
                dst_index[axis] = j as isize;
                output.write_at(output.byte_index(&dst_index), Scalar::Complex160(value));
            }
            continue;
        }
        if scaled_single {
            let mut line = vec![Complex::<f32>::new(0.0, 0.0); n];
            for (j, value) in line.iter_mut().take(take).enumerate() {
                src_index[axis] = j as isize;
                *value = match input.read_at(input.byte_index(&src_index)) {
                    Scalar::Complex(value) => {
                        Complex::new(value.re as f32, value.im as f32)
                    }
                    other => Complex::new(other.as_f64() as f32, 0.0),
                };
            }
            for (j, value) in crate::fft_single::c2c(
                &line,
                forward,
                scale_f32(scale, n),
            )
            .into_iter()
            .enumerate()
            {
                dst_index[axis] = j as isize;
                output.write_at(
                    output.byte_index(&dst_index),
                    Scalar::Complex(C64::new(value.re as f64, value.im as f64)),
                );
            }
            continue;
        }
        let mut line = vec![C64::new(0.0, 0.0); n];
        for (j, value) in line.iter_mut().take(take).enumerate() {
            src_index[axis] = j as isize;
            let scalar = input.read_at(input.byte_index(&src_index));
            *value = match scalar {
                Scalar::Complex(value) => value,
                other => C64::new(other.as_f64(), 0.0),
            };
        }
        let transformed = c2c(&line, forward, scale);
        for (j, value) in transformed.into_iter().enumerate() {
            dst_index[axis] = j as isize;
            output.write_at(output.byte_index(&dst_index), Scalar::Complex(value));
        }
    }
    Ok(output)
}

pub fn r2c_axis(
    input: &NdArray,
    n: usize,
    axis: usize,
    scale: f64,
    out_dtype: DType,
) -> Result<NdArray> {
    let mut shape = input.shape.clone();
    shape[axis] = (n / 2 + 1) as isize;
    let output = NdArray::zeros(shape.clone(), out_dtype)?;
    let batches = shape
        .iter()
        .enumerate()
        .filter(|(a, _)| *a != axis)
        .map(|(_, d)| *d as usize)
        .product();
    let take = n.min(input.shape[axis] as usize);
    let scaled_single = out_dtype == DType::C64 && scale != 1.0;
    let mut si = vec![0isize; input.ndim()];
    let mut di = si.clone();
    for batch in 0..batches {
        decode_batch(batch, &shape, axis, &mut si);
        di.copy_from_slice(&si);
        if out_dtype == DType::C160 {
            let mut line = vec![F80::ZERO; n];
            for j in 0..take {
                si[axis] = j as isize;
                line[j] = match input.read_at(input.byte_index(&si)) {
                    Scalar::Float80(value) => value,
                    other => F80::from_f64(other.as_f64()),
                };
            }
            for (j, value) in r2c_f80(&line, scale_f80(scale, n)).into_iter().enumerate() {
                di[axis] = j as isize;
                output.write_at(output.byte_index(&di), Scalar::Complex160(value));
            }
            continue;
        }
        if scaled_single {
            let mut line = vec![0.0f32; n];
            for j in 0..take {
                si[axis] = j as isize;
                line[j] = input.read_at(input.byte_index(&si)).as_f64() as f32;
            }
            for (j, z) in crate::fft_single::r2c(&line, scale_f32(scale, n))
                .into_iter()
                .enumerate()
            {
                di[axis] = j as isize;
                output.write_at(
                    output.byte_index(&di),
                    Scalar::Complex(C64::new(z.re as f64, z.im as f64)),
                );
            }
            continue;
        }
        let mut line = vec![0.; n];
        for j in 0..take {
            si[axis] = j as isize;
            line[j] = input.read_at(input.byte_index(&si)).as_f64();
        }
        for (j, z) in r2c(&line, scale).into_iter().enumerate() {
            di[axis] = j as isize;
            output.write_at(output.byte_index(&di), Scalar::Complex(z));
        }
    }
    Ok(output)
}

pub fn c2r_axis(
    input: &NdArray,
    n: usize,
    axis: usize,
    scale: f64,
    out_dtype: DType,
) -> Result<NdArray> {
    let mut shape = input.shape.clone();
    shape[axis] = n as isize;
    let output = NdArray::zeros(shape.clone(), out_dtype)?;
    let batches = shape
        .iter()
        .enumerate()
        .filter(|(a, _)| *a != axis)
        .map(|(_, d)| *d as usize)
        .product();
    let take = (n / 2 + 1).min(input.shape[axis] as usize);
    let scaled_single = out_dtype == DType::F32 && scale != 1.0;
    let mut si = vec![0isize; input.ndim()];
    let mut di = si.clone();
    for batch in 0..batches {
        decode_batch(batch, &shape, axis, &mut si);
        di.copy_from_slice(&si);
        if out_dtype == DType::F80 {
            let mut line = vec![C160::ZERO; take];
            for (j, value) in line.iter_mut().enumerate() {
                si[axis] = j as isize;
                *value = match input.read_at(input.byte_index(&si)) {
                    Scalar::Complex160(value) => value,
                    Scalar::Float80(value) => C160 { re: value, im: F80::ZERO },
                    Scalar::Complex(value) => C160 {
                        re: F80::from_f64(value.re),
                        im: F80::from_f64(value.im),
                    },
                    other => C160 { re: F80::from_f64(other.as_f64()), im: F80::ZERO },
                };
            }
            for (j, value) in c2r_f80(&line, n, scale_f80(scale, n)).into_iter().enumerate() {
                di[axis] = j as isize;
                output.write_at(output.byte_index(&di), Scalar::Float80(value));
            }
            continue;
        }
        if scaled_single {
            let mut line = vec![Complex::<f32>::new(0.0, 0.0); take];
            for (j, value) in line.iter_mut().enumerate() {
                si[axis] = j as isize;
                *value = match input.read_at(input.byte_index(&si)) {
                    Scalar::Complex(value) => {
                        Complex::new(value.re as f32, value.im as f32)
                    }
                    other => Complex::new(other.as_f64() as f32, 0.0),
                };
            }
            for (j, value) in crate::fft_single::c2r(
                &line,
                n,
                scale_f32(scale, n),
            )
            .into_iter()
            .enumerate()
            {
                di[axis] = j as isize;
                output.write_at(
                    output.byte_index(&di),
                    Scalar::Float(value as f64),
                );
            }
            continue;
        }
        let mut line = vec![C64::new(0., 0.); take];
        for (j, z) in line.iter_mut().enumerate() {
            si[axis] = j as isize;
            *z = match input.read_at(input.byte_index(&si)) {
                Scalar::Complex(v) => v,
                v => C64::new(v.as_f64(), 0.),
            };
        }
        for (j, x) in c2r(&line, n, scale).into_iter().enumerate() {
            di[axis] = j as isize;
            output.write_at(output.byte_index(&di), Scalar::Float(x));
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_forward_and_inverse() {
        let input = [1.0, 2.0, 3.0, 4.0].map(|x| C64::new(x, 0.0));
        let got = c2c(&input, true, 1.0);
        assert_eq!(
            got,
            vec![
                C64::new(10.0, 0.0),
                C64::new(-2.0, 2.0),
                C64::new(-2.0, 0.0),
                C64::new(-2.0, -2.0)
            ]
        );
        let back = c2c(&got, false, 0.25);
        for (a, b) in back.iter().zip(input) {
            assert!((a - b).norm() < 1e-14);
        }
    }

    #[test]
    fn real_forward_and_inverse() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let spectrum = r2c(&input, 1.0);
        let back = c2r(&spectrum, input.len(), 1.0 / input.len() as f64);
        assert_eq!(back, input);
    }

    #[test]
    fn f80_round_trip_meets_numpy_longdouble_tolerance() {
        let tolerance = F80::EPSILON.mul(F80::from_u64(5));
        let reversed_tolerance = F80::EPSILON.mul(F80::from_u64(6));
        for n in 1..32 {
            let input: Vec<C160> = (0..n)
                .map(|i| C160 {
                    re: F80::from_f64(((i * 17 + 3) % 29) as f64 / 29.0),
                    im: F80::from_f64(((i * 11 + 5) % 31) as f64 / 31.0),
                })
                .collect();
            let transformed = c2c_f80(&input, true, F80::ONE);
            let back = c2c_f80(
                &transformed,
                false,
                F80::ONE.div(F80::from_u64(n as u64)),
            );
            for (actual, expected) in back.iter().zip(&input) {
                let re_error = actual.re.sub(expected.re).abs();
                let im_error = actual.im.sub(expected.im).abs();
                assert!(re_error.partial_cmp_value(tolerance)
                    != Some(std::cmp::Ordering::Greater), "n={n}, re error={} eps",
                    re_error.div(F80::EPSILON).to_f64());
                assert!(im_error.partial_cmp_value(tolerance)
                    != Some(std::cmp::Ordering::Greater), "n={n}, im error={} eps",
                    im_error.div(F80::EPSILON).to_f64());
            }

            let real_input: Vec<F80> = input.iter().map(|value| value.re).collect();
            let spectrum = r2c_f80(&real_input, F80::ONE);
            let real_back = c2r_f80(
                &spectrum,
                n,
                F80::ONE.div(F80::from_u64(n as u64)),
            );
            for (actual, expected) in real_back.iter().zip(&real_input) {
                assert!(actual.sub(*expected).abs().partial_cmp_value(tolerance)
                    != Some(std::cmp::Ordering::Greater), "real n={n}");
            }

            let take = n / 2 + 1;
            let arbitrary_spectrum = input[..take].to_vec();
            let mut expected_spectrum = arbitrary_spectrum.clone();
            expected_spectrum[0].im = F80::ZERO;
            if n & 1 == 0 {
                expected_spectrum[n / 2].im = F80::ZERO;
            }
            let reconstructed = c2r_f80(
                &arbitrary_spectrum,
                n,
                F80::ONE.div(F80::from_u64(n as u64)),
            );
            let spectrum_back = r2c_f80(&reconstructed, F80::ONE);
            for (actual, expected) in spectrum_back.iter().zip(&expected_spectrum) {
                let re_error = actual.re.sub(expected.re).abs();
                let im_error = actual.im.sub(expected.im).abs();
                assert!(re_error.partial_cmp_value(reversed_tolerance)
                    != Some(std::cmp::Ordering::Greater), "reversed real n={n}, error={} eps",
                    re_error.div(F80::EPSILON).to_f64());
                assert!(im_error.partial_cmp_value(reversed_tolerance)
                    != Some(std::cmp::Ordering::Greater), "reversed imag n={n}, error={} eps",
                    im_error.div(F80::EPSILON).to_f64());
            }
        }
    }
}
