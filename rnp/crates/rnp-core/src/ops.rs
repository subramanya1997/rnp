//! Binary element-wise operations with numpy broadcasting and promotion.

use crate::array::NdArray;
use crate::dtype::{promote, promote_for_division, DType};
use crate::element::{Element, NpBool, C32, C64v, F16};
use crate::error::{Error, Result};
use crate::iter::{broadcast_shapes, broadcast_to, offsets};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinOp {
    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "subtract",
            BinOp::Mul => "multiply",
            BinOp::Div => "divide",
            BinOp::Eq => "equal",
            BinOp::Ne => "not_equal",
            BinOp::Lt => "less",
            BinOp::Le => "less_equal",
            BinOp::Gt => "greater",
            BinOp::Ge => "greater_equal",
        }
    }
}

/// Arithmetic over one concrete element type.
pub trait Arith: Element {
    fn a_add(self, o: Self) -> Self;
    fn a_sub(self, o: Self) -> Self;
    fn a_mul(self, o: Self) -> Self;
    fn a_div(self, o: Self) -> Self;
}

/// Ordering/equality over one concrete element type, with numpy's semantics
/// (NaN compares false to everything; complex orders lexicographically).
pub trait Cmp: Element {
    fn c_eq(self, o: Self) -> bool;
    fn c_lt(self, o: Self) -> bool;
    fn c_le(self, o: Self) -> bool;
}

macro_rules! impl_int_ops {
    ($($t:ty),*) => {$(
        impl Arith for $t {
            #[inline] fn a_add(self, o: Self) -> Self { self.wrapping_add(o) }
            #[inline] fn a_sub(self, o: Self) -> Self { self.wrapping_sub(o) }
            #[inline] fn a_mul(self, o: Self) -> Self { self.wrapping_mul(o) }
            // Never reached: integer `divide` promotes to float64 first.
            #[inline] fn a_div(self, o: Self) -> Self {
                if o == 0 { 0 } else { self.wrapping_div(o) }
            }
        }
        impl Cmp for $t {
            #[inline] fn c_eq(self, o: Self) -> bool { self == o }
            #[inline] fn c_lt(self, o: Self) -> bool { self < o }
            #[inline] fn c_le(self, o: Self) -> bool { self <= o }
        }
    )*};
}

impl_int_ops!(i8, i16, i32, i64, u8, u16, u32, u64);

macro_rules! impl_float_ops {
    ($($t:ty),*) => {$(
        impl Arith for $t {
            #[inline] fn a_add(self, o: Self) -> Self { self + o }
            #[inline] fn a_sub(self, o: Self) -> Self { self - o }
            #[inline] fn a_mul(self, o: Self) -> Self { self * o }
            #[inline] fn a_div(self, o: Self) -> Self { self / o }
        }
        impl Cmp for $t {
            #[inline] fn c_eq(self, o: Self) -> bool { self == o }
            #[inline] fn c_lt(self, o: Self) -> bool { self < o }
            #[inline] fn c_le(self, o: Self) -> bool { self <= o }
        }
    )*};
}

impl_float_ops!(f32, f64);

// numpy performs every half-precision operation in `float` and converts the
// result back (`npy_half_add` etc. in `halffloat.c`).
impl Arith for F16 {
    #[inline]
    fn a_add(self, o: Self) -> Self {
        F16::from_f32(self.to_f32() + o.to_f32())
    }
    #[inline]
    fn a_sub(self, o: Self) -> Self {
        F16::from_f32(self.to_f32() - o.to_f32())
    }
    #[inline]
    fn a_mul(self, o: Self) -> Self {
        F16::from_f32(self.to_f32() * o.to_f32())
    }
    #[inline]
    fn a_div(self, o: Self) -> Self {
        F16::from_f32(self.to_f32() / o.to_f32())
    }
}

impl Cmp for F16 {
    #[inline]
    fn c_eq(self, o: Self) -> bool {
        self.to_f32() == o.to_f32()
    }
    #[inline]
    fn c_lt(self, o: Self) -> bool {
        self.to_f32() < o.to_f32()
    }
    #[inline]
    fn c_le(self, o: Self) -> bool {
        self.to_f32() <= o.to_f32()
    }
}

/// Smith's algorithm, transcribed from numpy's `@TYPE@_divide` inner loop in
/// `umath/loops.c.src`.
///
/// Two things force a hand-written loop rather than `num_complex`'s `Div`:
/// that one overflows on operands like `1e-200 + 1e-200j` and returns
/// `nan + nanj` instead of `inf + nanj` when dividing by zero. The `mul_add`
/// calls reproduce the FMA contraction numpy's C loop is compiled with on
/// aarch64 -- without them the results differ from numpy by an ULP.
macro_rules! complex_div {
    ($t:ty, $f:ty) => {
        #[inline]
        fn a_div(self, o: Self) -> Self {
            let (ar, ai, br, bi) = (self.re, self.im, o.re, o.im);
            let (br_abs, bi_abs) = (br.abs(), bi.abs());
            if br_abs >= bi_abs {
                if br_abs == 0.0 && bi_abs == 0.0 {
                    // Division by zero yields a complex inf or nan; numpy
                    // divides by the *absolute* values here.
                    return <$t>::new(ar / br_abs, ai / bi_abs);
                }
                let rat = bi / br;
                let scl = (1.0 as $f) / bi.mul_add(rat, br);
                <$t>::new(ai.mul_add(rat, ar) * scl, (-ar).mul_add(rat, ai) * scl)
            } else {
                let rat = br / bi;
                let scl = (1.0 as $f) / br.mul_add(rat, bi);
                <$t>::new(ar.mul_add(rat, ai) * scl, ai.mul_add(rat, -ar) * scl)
            }
        }
    };
}

macro_rules! impl_complex_ops {
    ($($t:ty, $f:ty);*) => {$(
        impl Arith for $t {
            #[inline] fn a_add(self, o: Self) -> Self { self + o }
            #[inline] fn a_sub(self, o: Self) -> Self { self - o }
            // numpy's `@TYPE@_multiply` inner loop is
            //   out.re = a.re*b.re - a.im*b.im;
            //   out.im = a.re*b.im + a.im*b.re;
            // which clang contracts into an FMA per statement on aarch64.
            // Plain `*` on num_complex rounds twice and differs by an ULP.
            #[inline] fn a_mul(self, o: Self) -> Self {
                <$t>::new(
                    self.re.mul_add(o.re, -(self.im * o.im)),
                    self.re.mul_add(o.im, self.im * o.re),
                )
            }
            complex_div!($t, $f);
        }
        // numpy orders complex lexicographically: real part first, then imag.
        impl Cmp for $t {
            #[inline] fn c_eq(self, o: Self) -> bool { self.re == o.re && self.im == o.im }
            #[inline] fn c_lt(self, o: Self) -> bool {
                self.re < o.re || (self.re == o.re && self.im < o.im)
            }
            #[inline] fn c_le(self, o: Self) -> bool {
                self.re < o.re || (self.re == o.re && self.im <= o.im)
            }
        }
    )*};
}

impl_complex_ops!(C32, f32; C64v, f64);

// numpy's bool `add` is logical or and `mul` is logical and; `subtract` is
// rejected before it reaches an inner loop.
impl Arith for NpBool {
    #[inline]
    fn a_add(self, o: Self) -> Self {
        NpBool::new(self.get() || o.get())
    }
    #[inline]
    fn a_sub(self, o: Self) -> Self {
        NpBool::new(self.get() ^ o.get())
    }
    #[inline]
    fn a_mul(self, o: Self) -> Self {
        NpBool::new(self.get() && o.get())
    }
    #[inline]
    fn a_div(self, o: Self) -> Self {
        NpBool::new(o.get() && self.get())
    }
}

impl Cmp for NpBool {
    #[inline]
    fn c_eq(self, o: Self) -> bool {
        self.get() == o.get()
    }
    #[inline]
    fn c_lt(self, o: Self) -> bool {
        !self.get() && o.get()
    }
    #[inline]
    fn c_le(self, o: Self) -> bool {
        !self.get() || o.get()
    }
}

/// Comparisons between flexible (`S`/`U`) arrays.
///
/// Only the comparison ufuncs are defined for strings at M1; arithmetic on
/// them is a later milestone. Both operands are promoted to the wider width
/// of the same kind and compared on their *logical* value, i.e. with the
/// trailing NUL padding numpy uses stripped off.
pub fn binary_flexible(a: &NdArray, b: &NdArray, op: BinOp) -> Result<NdArray> {
    if !op.is_comparison() {
        return Err(Error::NotImplemented(format!(
            "ufunc '{}' is not supported for dtypes ({}, {})",
            op.name(),
            a.dtype,
            b.dtype
        )));
    }
    let same_kind = a.dtype.category() == b.dtype.category();
    if !same_kind || !matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
    {
        return Err(Error::NotImplemented(format!(
            "comparing {} with {} is not implemented yet",
            a.dtype, b.dtype
        )));
    }
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let av = broadcast_to(a, &out_shape)?;
    let bv = broadcast_to(b, &out_shape)?;
    let out = NdArray::empty(out_shape, DType::Bool)?;

    let a_offs: Vec<isize> = offsets(&av.shape, &av.strides, av.byte_offset).collect();
    let b_offs: Vec<isize> = offsets(&bv.shape, &bv.strides, bv.byte_offset).collect();
    let o_offs: Vec<isize> = offsets(&out.shape, &out.strides, out.byte_offset).collect();

    for k in 0..o_offs.len() {
        let x = logical_bytes(&av, a_offs[k]);
        let y = logical_bytes(&bv, b_offs[k]);
        let ord = x.cmp(&y);
        let r = match op {
            BinOp::Eq => ord.is_eq(),
            BinOp::Ne => ord.is_ne(),
            BinOp::Lt => ord.is_lt(),
            BinOp::Le => ord.is_le(),
            BinOp::Gt => ord.is_gt(),
            _ => ord.is_ge(),
        };
        out.write_at(o_offs[k], crate::element::Scalar::Bool(r));
    }
    Ok(out)
}

/// An element's code units with numpy's trailing-NUL padding removed, so
/// that `S3` `b"ab\0"` equals `S5` `b"ab\0\0\0"`. `U` elements are decoded
/// from UCS4 so that ordering compares code points, not their little-endian
/// bytes.
fn logical_bytes(arr: &NdArray, off: isize) -> Vec<u32> {
    let raw = arr.raw_bytes_at(off);
    let mut v: Vec<u32> = match arr.dtype {
        DType::Str(_) => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => raw.iter().map(|&b| b as u32).collect(),
    };
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

/// The dtype the inner loop runs in, and the dtype of the result.
pub fn result_dtypes(a: DType, b: DType, op: BinOp) -> Result<(DType, DType)> {
    if op.is_comparison() {
        let compute = promote(a, b);
        return Ok((compute, DType::Bool));
    }
    if op == BinOp::Div {
        let d = promote_for_division(a, b);
        return Ok((d, d));
    }
    if op == BinOp::Sub && a.is_bool() && b.is_bool() {
        return Err(Error::TypeError(
            "numpy boolean subtract, the `-` operator, is not supported, use \
             the bitwise_xor, the `^` operator, or the logical_xor function \
             instead."
                .into(),
        ));
    }
    let d = promote(a, b);
    Ok((d, d))
}

/// Element-wise binary op with numpy broadcasting + promotion.
pub fn binary(a: &NdArray, b: &NdArray, op: BinOp) -> Result<NdArray> {
    if a.dtype.is_flexible() || b.dtype.is_flexible() {
        return binary_flexible(a, b, op);
    }
    let (compute, out_dtype) = result_dtypes(a.dtype, b.dtype, op)?;
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;

    // Cast before broadcasting so the (cheap) cast copy stays small.
    let a_cast;
    let a_ref = if a.dtype == compute {
        a
    } else {
        a_cast = a.astype(compute);
        &a_cast
    };
    let b_cast;
    let b_ref = if b.dtype == compute {
        b
    } else {
        b_cast = b.astype(compute);
        &b_cast
    };

    let av = broadcast_to(a_ref, &out_shape)?;
    let bv = broadcast_to(b_ref, &out_shape)?;
    let out = NdArray::empty(out_shape, out_dtype)?;

    crate::dispatch_dtype!(compute, T, {
        if op.is_comparison() {
            run_cmp::<T>(&av, &bv, &out, op);
        } else {
            run_arith::<T>(&av, &bv, &out, op);
        }
    });
    Ok(out)
}

/// True when the array's elements sit at `byte_offset + i * itemsize` in
/// order, i.e. a flat typed slice can be formed.
#[inline]
fn flat_ptr<T>(a: &NdArray) -> Option<*const T> {
    if !a.flags.c_contiguous {
        return None;
    }
    // SAFETY: `byte_offset` is always a multiple of the itemsize and the
    // allocation is 64-byte aligned, so the result is well aligned for T; the
    // array's `size()` elements from here are inside the allocation.
    unsafe { Some(a.buffer.as_ptr().offset(a.byte_offset) as *const T) }
}

/// The inner loop, generic over the element type *and* the operation.
///
/// `F` is a distinct zero-sized type per call site, so the operation inlines
/// into the loop body and LLVM can vectorise the contiguous path. Passing a
/// `fn(T, T) -> T` pointer instead costs roughly 4x on f64 adds.
#[inline]
fn elementwise<T: Element, U: Copy, F: Fn(T, T) -> U>(
    a: &NdArray,
    b: &NdArray,
    o: *mut U,
    n: usize,
    f: F,
) {
    if let (Some(pa), Some(pb)) = (flat_ptr::<T>(a), flat_ptr::<T>(b)) {
        // Fast path: both operands are contiguous and already the out shape.
        for i in 0..n {
            // SAFETY: i < n and all three buffers hold at least n elements.
            unsafe { *o.add(i) = f(*pa.add(i), *pb.add(i)) }
        }
        return;
    }
    if a.ndim() <= 1 && b.ndim() <= 1 {
        // 1-D strided operands (`x[::2] + y[::2]`): step two pointers instead
        // of running the odometer for every element.
        let (sa, sb) = (
            a.strides.first().copied().unwrap_or(0),
            b.strides.first().copied().unwrap_or(0),
        );
        let (base_a, base_b) = (a.byte_offset, b.byte_offset);
        for i in 0..n {
            // SAFETY: base + i*stride stays within a's (and b's) elements for
            // i < n, since both were broadcast to the output shape.
            unsafe {
                let x = std::ptr::read_unaligned(
                    a.buffer.as_ptr().offset(base_a + i as isize * sa) as *const T,
                );
                let y = std::ptr::read_unaligned(
                    b.buffer.as_ptr().offset(base_b + i as isize * sb) as *const T,
                );
                *o.add(i) = f(x, y);
            }
        }
        return;
    }

    let ia = offsets(&a.shape, &a.strides, a.byte_offset);
    let ib = offsets(&b.shape, &b.strides, b.byte_offset);
    for (i, (oa, ob)) in ia.zip(ib).enumerate() {
        // SAFETY: offsets come from in-bounds strided iteration over a and b,
        // and i < n because the iterators yield exactly n elements each.
        unsafe {
            let x = std::ptr::read_unaligned(a.buffer.as_ptr().offset(oa) as *const T);
            let y = std::ptr::read_unaligned(b.buffer.as_ptr().offset(ob) as *const T);
            *o.add(i) = f(x, y);
        }
    }
}

fn run_arith<T: Arith>(a: &NdArray, b: &NdArray, out: &NdArray, op: BinOp) {
    let n = out.size();
    // SAFETY: `out` was freshly allocated C-contiguous with `n` elements of T.
    let o = unsafe { out.buffer.as_mut_ptr() as *mut T };
    match op {
        BinOp::Add => elementwise::<T, T, _>(a, b, o, n, |x, y| x.a_add(y)),
        BinOp::Sub => elementwise::<T, T, _>(a, b, o, n, |x, y| x.a_sub(y)),
        BinOp::Mul => elementwise::<T, T, _>(a, b, o, n, |x, y| x.a_mul(y)),
        BinOp::Div => elementwise::<T, T, _>(a, b, o, n, |x, y| x.a_div(y)),
        _ => unreachable!("comparison routed to run_cmp"),
    }
}

fn run_cmp<T: Cmp>(a: &NdArray, b: &NdArray, out: &NdArray, op: BinOp) {
    let n = out.size();
    // SAFETY: `out` is a freshly allocated contiguous bool array of n bytes.
    let o = unsafe { out.buffer.as_mut_ptr() };
    match op {
        BinOp::Eq => elementwise::<T, u8, _>(a, b, o, n, |x, y| x.c_eq(y) as u8),
        BinOp::Ne => elementwise::<T, u8, _>(a, b, o, n, |x, y| !x.c_eq(y) as u8),
        BinOp::Lt => elementwise::<T, u8, _>(a, b, o, n, |x, y| x.c_lt(y) as u8),
        BinOp::Le => elementwise::<T, u8, _>(a, b, o, n, |x, y| x.c_le(y) as u8),
        BinOp::Gt => elementwise::<T, u8, _>(a, b, o, n, |x, y| y.c_lt(x) as u8),
        BinOp::Ge => elementwise::<T, u8, _>(a, b, o, n, |x, y| y.c_le(x) as u8),
        _ => unreachable!("arithmetic routed to run_arith"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Scalar;

    fn arr(v: &[i64], d: DType) -> NdArray {
        let s: Vec<Scalar> = v.iter().map(|&x| Scalar::Int(x)).collect();
        NdArray::from_scalars(&s, DType::I64).unwrap().astype(d)
    }

    #[test]
    fn result_dtype_matches_numpy() {
        // np.add(int32, float32).dtype == float64 (NEP 50)
        assert_eq!(
            result_dtypes(DType::I32, DType::F32, BinOp::Add).unwrap().1,
            DType::F64
        );
        assert_eq!(
            result_dtypes(DType::I16, DType::F32, BinOp::Add).unwrap().1,
            DType::F32
        );
        // Integer true-division always lands on float64.
        assert_eq!(
            result_dtypes(DType::I64, DType::I64, BinOp::Div).unwrap().1,
            DType::F64
        );
        assert_eq!(
            result_dtypes(DType::F32, DType::F32, BinOp::Div).unwrap().1,
            DType::F32
        );
        // Comparisons always produce bool.
        assert_eq!(
            result_dtypes(DType::F64, DType::I8, BinOp::Lt).unwrap().1,
            DType::Bool
        );
        // bool - bool is a TypeError in numpy.
        assert!(result_dtypes(DType::Bool, DType::Bool, BinOp::Sub).is_err());
        assert!(result_dtypes(DType::Bool, DType::Bool, BinOp::Add).is_ok());
    }

    #[test]
    fn add_promotes_and_broadcasts() {
        let a = arr(&[1, 2, 3], DType::I32).reshape(&[3, 1]).unwrap();
        let b = arr(&[10, 20], DType::I64).reshape(&[1, 2]).unwrap();
        let c = binary(&a, &b, BinOp::Add).unwrap();
        assert_eq!(c.shape, vec![3, 2]);
        assert_eq!(c.dtype, DType::I64);
        assert_eq!(
            c.to_vec(),
            [11, 21, 12, 22, 13, 23]
                .iter()
                .map(|&v| Scalar::Int(v))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn integer_division_yields_float64() {
        let a = arr(&[1, 0], DType::I64);
        let b = arr(&[0, 0], DType::I64);
        let c = binary(&a, &b, BinOp::Div).unwrap();
        assert_eq!(c.dtype, DType::F64);
        assert_eq!(c.get_flat(0), Scalar::Float(f64::INFINITY));
        match c.get_flat(1) {
            Scalar::Float(f) => assert!(f.is_nan()),
            other => panic!("expected nan, got {other:?}"),
        }
    }

    #[test]
    fn integer_overflow_wraps_like_numpy() {
        // np.add(np.int8(127), np.int8(1)) == -128
        let a = arr(&[127], DType::I8);
        let c = binary(&a, &arr(&[1], DType::I8), BinOp::Add).unwrap();
        assert_eq!(c.dtype, DType::I8);
        assert_eq!(c.get_flat(0), Scalar::Int(-128));
    }

    #[test]
    fn comparisons_return_bool_arrays() {
        let a = arr(&[1, 2, 3], DType::I32);
        let b = arr(&[3, 2, 1], DType::I32);
        let c = binary(&a, &b, BinOp::Lt).unwrap();
        assert_eq!(c.dtype, DType::Bool);
        assert_eq!(
            c.to_vec(),
            vec![Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(false)]
        );
        let e = binary(&a, &b, BinOp::Ge).unwrap();
        assert_eq!(
            e.to_vec(),
            vec![Scalar::Bool(false), Scalar::Bool(true), Scalar::Bool(true)]
        );
    }

    #[test]
    fn nan_compares_false_except_ne() {
        let a = NdArray::from_scalars(&[Scalar::Float(f64::NAN)], DType::F64).unwrap();
        for (op, want) in [
            (BinOp::Eq, false),
            (BinOp::Lt, false),
            (BinOp::Le, false),
            (BinOp::Gt, false),
            (BinOp::Ge, false),
            (BinOp::Ne, true),
        ] {
            let c = binary(&a, &a, op).unwrap();
            assert_eq!(c.get_flat(0), Scalar::Bool(want), "{:?}", op);
        }
    }

    #[test]
    fn strided_operands_take_the_slow_path_correctly() {
        let a = NdArray::arange(0.0, 10.0, 1.0, DType::I64).unwrap();
        let s = a.slice_axis(0, 0, 5, 2); // [0,2,4,6,8]
        let t = a.slice_axis(0, 1, 5, 2); // [1,3,5,7,9]
        let c = binary(&s, &t, BinOp::Add).unwrap();
        assert_eq!(
            c.to_vec(),
            [1, 5, 9, 13, 17]
                .iter()
                .map(|&v| Scalar::Int(v))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn incompatible_shapes_error() {
        let a = NdArray::zeros(vec![3], DType::F64).unwrap();
        let b = NdArray::zeros(vec![4], DType::F64).unwrap();
        let e = binary(&a, &b, BinOp::Add).unwrap_err();
        assert!(matches!(e, Error::ValueError(_)));
        assert!(e.message().contains("broadcast"));
    }

    #[test]
    fn complex_division_matches_numpy_edge_cases() {
        use num_complex::Complex;
        let mk = |v: Vec<(f64, f64)>| {
            let s: Vec<Scalar> = v
                .into_iter()
                .map(|(re, im)| Scalar::Complex(Complex::new(re, im)))
                .collect();
            NdArray::from_scalars(&s, DType::C128).unwrap()
        };
        // np.divide(np.array([3+0j, -4+0j]), np.array([0j, 1e-200+1e-200j]))
        // -> [inf+nanj, -2e+200+2e+200j]
        let a = mk(vec![(3.0, 0.0), (-4.0, 0.0)]);
        let b = mk(vec![(0.0, 0.0), (1e-200, 1e-200)]);
        let c = binary(&a, &b, BinOp::Div).unwrap();
        match c.get_flat(0) {
            Scalar::Complex(z) => {
                assert!(z.re.is_infinite() && z.re > 0.0, "{z:?}");
                assert!(z.im.is_nan(), "{z:?}");
            }
            other => panic!("{other:?}"),
        }
        match c.get_flat(1) {
            // The naive formula overflows to -inf+infj here.
            Scalar::Complex(z) => assert_eq!((z.re, z.im), (-2e200, 2e200)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn bool_add_is_logical_or() {
        let a = NdArray::from_scalars(&[Scalar::Bool(true), Scalar::Bool(false)], DType::Bool)
            .unwrap();
        let b = NdArray::from_scalars(&[Scalar::Bool(false), Scalar::Bool(false)], DType::Bool)
            .unwrap();
        let c = binary(&a, &b, BinOp::Add).unwrap();
        assert_eq!(c.dtype, DType::Bool);
        assert_eq!(
            c.to_vec(),
            vec![Scalar::Bool(true), Scalar::Bool(false)]
        );
    }
}
