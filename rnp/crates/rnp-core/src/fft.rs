//! One-dimensional pocketfft kernels and ndarray axis orchestration.
//!
//! This is a safe Rust transcription of pocketfft's complex FFTPACK and
//! Bluestein plans. Expression grouping, factor order, twiddle construction,
//! buffer swaps, and final scaling follow `pocketfft_hdronly.h`.

use num_complex::Complex;

use crate::{DType, Error, NdArray, Result, Scalar};

type C64 = Complex<f64>;

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
                    C::new((x as f64 * ang).cos(), (x as f64 * ang).sin())
                } else {
                    C::new(
                        ((2 * n - x) as f64 * ang).sin(),
                        ((2 * n - x) as f64 * ang).cos(),
                    )
                }
            } else {
                x -= 2 * n;
                if x < n {
                    C::new(-(x as f64 * ang).sin(), (x as f64 * ang).cos())
                } else {
                    C::new(
                        -((2 * n - x) as f64 * ang).cos(),
                        ((2 * n - x) as f64 * ang).sin(),
                    )
                }
            }
        } else {
            x = 8 * n - x;
            if x < 2 * n {
                if x < n {
                    C::new((x as f64 * ang).cos(), -(x as f64 * ang).sin())
                } else {
                    C::new(
                        ((2 * n - x) as f64 * ang).sin(),
                        -((2 * n - x) as f64 * ang).cos(),
                    )
                }
            } else {
                x -= 2 * n;
                if x < n {
                    C::new(-(x as f64 * ang).sin(), -(x as f64 * ang).cos())
                } else {
                    C::new(
                        -((2 * n - x) as f64 * ang).cos(),
                        -((2 * n - x) as f64 * ang).sin(),
                    )
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
    let mut src_index = vec![0isize; input.ndim()];
    let mut dst_index = vec![0isize; input.ndim()];
    for batch in 0..batches {
        decode_batch(batch, &shape, axis, &mut src_index);
        dst_index.copy_from_slice(&src_index);
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
}
