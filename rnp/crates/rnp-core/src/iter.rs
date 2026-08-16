//! Strided iteration and numpy broadcasting.

use crate::array::NdArray;
use crate::error::{Error, Result};

/// Iterator over the byte offsets of every element, in C order.
pub struct OffsetIter<'a> {
    shape: &'a [isize],
    strides: &'a [isize],
    counter: Vec<isize>,
    offset: isize,
    remaining: usize,
}

pub fn offsets<'a>(shape: &'a [isize], strides: &'a [isize], base: isize) -> OffsetIter<'a> {
    let remaining = shape.iter().map(|&d| d.max(0) as usize).product();
    OffsetIter {
        shape,
        strides,
        counter: vec![0; shape.len()],
        offset: base,
        remaining,
    }
}

impl<'a> Iterator for OffsetIter<'a> {
    type Item = isize;

    #[inline]
    fn next(&mut self) -> Option<isize> {
        if self.remaining == 0 {
            return None;
        }
        let out = self.offset;
        self.remaining -= 1;
        // Odometer increment over the trailing axes (C order).
        for ax in (0..self.shape.len()).rev() {
            self.counter[ax] += 1;
            self.offset += self.strides[ax];
            if self.counter[ax] < self.shape[ax] {
                break;
            }
            self.offset -= self.strides[ax] * self.counter[ax];
            self.counter[ax] = 0;
        }
        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> ExactSizeIterator for OffsetIter<'a> {}

/// numpy's broadcasting rule: right-align the shapes; each axis pair must be
/// equal or contain a 1.
pub fn broadcast_shapes(a: &[isize], b: &[isize]) -> Result<Vec<isize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0isize; n];
    for i in 0..n {
        let da = if i < n - a.len() { 1 } else { a[i - (n - a.len())] };
        let db = if i < n - b.len() { 1 } else { b[i - (n - b.len())] };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(Error::ValueError(format!(
                "operands could not be broadcast together with shapes {} {} ",
                fmt_shape(a),
                fmt_shape(b)
            )));
        };
    }
    Ok(out)
}

fn fmt_shape(s: &[isize]) -> String {
    let inner = s
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    if s.len() == 1 {
        format!("({},)", inner)
    } else {
        format!("({})", inner)
    }
}

/// A zero-copy view of `arr` stretched to `shape` (broadcast axes get stride 0).
pub fn broadcast_to(arr: &NdArray, shape: &[isize]) -> Result<NdArray> {
    if arr.shape == shape {
        return Ok(arr.clone());
    }
    if shape.len() < arr.ndim() {
        return Err(Error::ValueError(format!(
            "could not broadcast {} into shape {}",
            fmt_shape(&arr.shape),
            fmt_shape(shape)
        )));
    }
    let pad = shape.len() - arr.ndim();
    let mut strides = vec![0isize; shape.len()];
    for i in 0..arr.ndim() {
        let d = arr.shape[i];
        if d == shape[i + pad] {
            strides[i + pad] = arr.strides[i];
        } else if d == 1 {
            strides[i + pad] = 0;
        } else {
            return Err(Error::ValueError(format!(
                "could not broadcast {} into shape {}",
                fmt_shape(&arr.shape),
                fmt_shape(shape)
            )));
        }
    }
    let mut out = arr.clone();
    out.shape = shape.to_vec();
    out.strides = strides;
    out.flags.owndata = false;
    out.flags.writeable = false;
    out.update_flags();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn broadcast_shape_rules_match_numpy() {
        // Verified against np.broadcast_shapes.
        let cases: &[(&[isize], &[isize], &[isize])] = &[
            (&[3], &[3], &[3]),
            (&[3, 1], &[1, 4], &[3, 4]),
            (&[5, 4], &[4], &[5, 4]),
            (&[15, 3, 5], &[15, 1, 5], &[15, 3, 5]),
            (&[15, 3, 5], &[3, 5], &[15, 3, 5]),
            (&[15, 3, 5], &[3, 1], &[15, 3, 5]),
            (&[8, 1, 6, 1], &[7, 1, 5], &[8, 7, 6, 5]),
            (&[], &[2, 3], &[2, 3]),
            (&[0], &[1], &[0]),
        ];
        for &(a, b, want) in cases {
            assert_eq!(broadcast_shapes(a, b).unwrap(), want.to_vec(), "{a:?} {b:?}");
            assert_eq!(broadcast_shapes(b, a).unwrap(), want.to_vec());
        }
        // Incompatible, per numpy.
        assert!(broadcast_shapes(&[3], &[4]).is_err());
        assert!(broadcast_shapes(&[2, 3], &[2, 4]).is_err());
    }

    #[test]
    fn broadcast_to_uses_zero_strides() {
        let a = NdArray::arange(0.0, 3.0, 1.0, DType::I64).unwrap();
        let b = broadcast_to(&a, &[2, 3]).unwrap();
        assert_eq!(b.shape, vec![2, 3]);
        assert_eq!(b.strides, vec![0, 8]);
        assert!(!b.flags.writeable);
        let vals = b.to_vec();
        assert_eq!(vals.len(), 6);
        assert_eq!(vals[0], vals[3]);
        assert_eq!(vals[2], vals[5]);
    }

    #[test]
    fn offset_iteration_order_is_c_order() {
        let a = NdArray::arange(0.0, 6.0, 1.0, DType::I32)
            .unwrap()
            .reshape(&[2, 3])
            .unwrap();
        let t = a.transpose();
        let got: Vec<_> = t.to_vec();
        // a = [[0,1,2],[3,4,5]]; a.T ravels to 0,3,1,4,2,5
        let want: Vec<_> = [0, 3, 1, 4, 2, 5]
            .iter()
            .map(|&v| crate::element::Scalar::Int(v))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn empty_arrays_iterate_zero_times() {
        let a = NdArray::zeros(vec![0, 3], DType::F64).unwrap();
        assert_eq!(offsets(&a.shape, &a.strides, 0).count(), 0);
    }
}
