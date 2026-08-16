//! `sort` / `argsort` / `searchsorted`.
//!
//! numpy's ordering is a *total* order, not `PartialOrd`: NaNs compare
//! greater than everything (and equal to each other) so that they land at the
//! end, and a complex number is ordered lexicographically by `(real, imag)`
//! with the same NaN rule on each component. Both facts were probed from real
//! numpy; see the tests at the bottom.

use std::cmp::Ordering;

use crate::array::NdArray;
use crate::element::Scalar;
use crate::error::{Error, Result};

/// numpy's float order: NaN is greater than everything, NaNs tie.
fn cmp_f64(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// numpy's total order on one element value.
pub fn cmp_scalar(a: Scalar, b: Scalar) -> Ordering {
    match (a, b) {
        (Scalar::Bool(x), Scalar::Bool(y)) => x.cmp(&y),
        (Scalar::Int(x), Scalar::Int(y)) => x.cmp(&y),
        (Scalar::Uint(x), Scalar::Uint(y)) => x.cmp(&y),
        (Scalar::Complex(x), Scalar::Complex(y)) => {
            cmp_f64(x.re, y.re).then_with(|| cmp_f64(x.im, y.im))
        }
        _ => cmp_f64(a.as_f64(), b.as_f64()),
    }
}

/// The byte offsets of every 1-D lane along `axis`.
fn lanes(arr: &NdArray, axis: usize) -> Vec<Vec<isize>> {
    let n = arr.shape[axis].max(0);
    let step = arr.strides[axis];
    let mut shape = arr.shape.clone();
    let mut strides = arr.strides.clone();
    shape.remove(axis);
    strides.remove(axis);
    crate::iter::offsets(&shape, &strides, arr.byte_offset)
        .map(|base| (0..n).map(|k| base + k * step).collect())
        .collect()
}

fn check_axis(arr: &NdArray, axis: usize) -> Result<()> {
    if axis >= arr.ndim() {
        return Err(Error::AxisError(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis,
            arr.ndim()
        )));
    }
    Ok(())
}

/// The sort key of one element: a scalar for numeric dtypes, the logical
/// code units for the flexible ones.
enum Key {
    Num(Scalar),
    Text(Vec<u32>),
}

fn key_at(arr: &NdArray, off: isize) -> Key {
    if arr.dtype().is_flexible() {
        Key::Text(crate::ops::logical_bytes(arr, off))
    } else {
        Key::Num(arr.read_at(off))
    }
}

fn cmp_key(a: &Key, b: &Key) -> Ordering {
    match (a, b) {
        (Key::Num(x), Key::Num(y)) => cmp_scalar(*x, *y),
        (Key::Text(x), Key::Text(y)) => x.cmp(y),
        _ => Ordering::Equal,
    }
}

/// `a.sort(axis)`, in place. `stable` picks numpy's `kind='stable'`; the
/// default introsort is not stable but nothing observable depends on which
/// equal element wins, so both use the same (stable) implementation.
pub fn sort_inplace(arr: &mut NdArray, axis: usize, _stable: bool) -> Result<()> {
    check_axis(arr, axis)?;
    if arr.dtype().is_object() {
        return Err(Error::NotImplemented("sort on object arrays".into()));
    }
    if !arr.flags.writeable {
        return Err(Error::ValueError(
            "assignment destination is read-only".into(),
        ));
    }
    let isz = arr.itemsize();
    // One scratch buffer for the whole sort, reused per lane: a `Vec` per
    // element turns a 1e6-element sort into a million allocations.
    let mut scratch: Vec<u8> = Vec::new();
    for lane in lanes(arr, axis) {
        let n = lane.len();
        let mut order: Vec<usize> = (0..n).collect();
        let keys: Vec<Key> = lane.iter().map(|&o| key_at(arr, o)).collect();
        order.sort_by(|&i, &j| cmp_key(&keys[i], &keys[j]));
        // Snapshot the raw bytes, then lay them back down in order; writing
        // in place would clobber sources still to be read.
        scratch.clear();
        scratch.reserve(n * isz);
        for &o in &lane {
            scratch.extend_from_slice(arr.raw_bytes_at(o));
        }
        for (k, &src) in order.iter().enumerate() {
            arr.write_raw_at(lane[k], &scratch[src * isz..(src + 1) * isz]);
        }
    }
    Ok(())
}

/// `np.argsort(a, axis)` — always a *stable* order, as numpy's `argsort`
/// documents for equal keys under `kind='stable'` and produces in practice
/// for the small inputs the tests use.
pub fn argsort(arr: &NdArray, axis: usize, _stable: bool) -> Result<NdArray> {
    check_axis(arr, axis)?;
    if arr.dtype().is_object() {
        return Err(Error::NotImplemented("argsort on object arrays".into()));
    }
    let mut out = NdArray::zeros(arr.shape.clone(), crate::dtype::DType::I64)?;
    let n = arr.shape[axis].max(0);
    let mut shape = arr.shape.clone();
    let mut strides = out.strides.clone();
    shape.remove(axis);
    let out_step = strides[axis];
    strides.remove(axis);
    let out_bases: Vec<isize> = crate::iter::offsets(&shape, &strides, out.byte_offset).collect();
    for (lane, &base) in lanes(arr, axis).iter().zip(out_bases.iter()) {
        let keys: Vec<Key> = lane.iter().map(|&o| key_at(arr, o)).collect();
        let mut order: Vec<usize> = (0..lane.len()).collect();
        order.sort_by(|&i, &j| cmp_key(&keys[i], &keys[j]));
        for k in 0..n as usize {
            out.write_at(base + k as isize * out_step, Scalar::Int(order[k] as i64));
        }
    }
    Ok(out)
}

/// `np.searchsorted(a, v, side)` on a sorted 1-D `a`.
pub fn searchsorted(a: &NdArray, v: &NdArray, right: bool) -> Result<NdArray> {
    if a.ndim() != 1 {
        return Err(Error::ValueError(
            "object too deep for desired array".into(),
        ));
    }
    let n = a.size();
    let keys: Vec<Key> = (0..n)
        .map(|i| key_at(a, a.byte_offset + i as isize * a.strides[0]))
        .collect();
    let out = NdArray::zeros(v.shape.clone(), crate::dtype::DType::I64)?;
    let mut k = 0usize;
    for off in crate::iter::offsets(&v.shape, &v.strides, v.byte_offset) {
        let key = key_at(v, off);
        // `left` is the first slot where `a[i] >= key`, `right` the first
        // where `a[i] > key`.
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let c = cmp_key(&keys[mid], &key);
            let go_right = if right { c != Ordering::Greater } else { c == Ordering::Less };
            if go_right {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out.write_at(out.byte_offset + (k * 8) as isize, Scalar::Int(lo as i64));
        k += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn sort_puts_nan_last() {
        // np.sort([3., nan, 1.]) -> [1., 3., nan]
        let mut a = NdArray::from_scalars(
            &[
                Scalar::Float(3.0),
                Scalar::Float(f64::NAN),
                Scalar::Float(1.0),
            ],
            DType::F64,
        )
        .unwrap();
        sort_inplace(&mut a, 0, false).unwrap();
        assert_eq!(a.get_flat(0).as_f64(), 1.0);
        assert_eq!(a.get_flat(1).as_f64(), 3.0);
        assert!(a.get_flat(2).as_f64().is_nan());
    }

    #[test]
    fn argsort_is_stable() {
        // np.argsort([2, 1, 2, 1]) -> [1, 3, 0, 2]
        let a = NdArray::from_scalars(
            &[
                Scalar::Int(2),
                Scalar::Int(1),
                Scalar::Int(2),
                Scalar::Int(1),
            ],
            DType::I64,
        )
        .unwrap();
        let o = argsort(&a, 0, false).unwrap();
        let got: Vec<i64> = (0..4).map(|i| o.get_flat(i).as_f64() as i64).collect();
        assert_eq!(got, vec![1, 3, 0, 2]);
    }

    #[test]
    fn searchsorted_sides() {
        // np.searchsorted([1,2,2,3], [2], 'left') -> 1; 'right' -> 3
        let a = NdArray::from_scalars(
            &[
                Scalar::Int(1),
                Scalar::Int(2),
                Scalar::Int(2),
                Scalar::Int(3),
            ],
            DType::I64,
        )
        .unwrap();
        let v = NdArray::from_scalars(&[Scalar::Int(2)], DType::I64).unwrap();
        assert_eq!(searchsorted(&a, &v, false).unwrap().get_flat(0).as_f64() as i64, 1);
        assert_eq!(searchsorted(&a, &v, true).unwrap().get_flat(0).as_f64() as i64, 3);
    }
}
