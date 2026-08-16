//! The `NdArray` header: a dtype + shape + byte-strides view onto a shared
//! `Buffer`, exactly mirroring numpy's memory model.

use std::sync::Arc;

use crate::buffer::Buffer;
use crate::dtype::DType;
use crate::element::{Element, Scalar};
use crate::error::{Error, Result};

/// numpy's array flags (the subset modelled at M0).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Flags {
    pub c_contiguous: bool,
    pub f_contiguous: bool,
    pub writeable: bool,
    pub owndata: bool,
    pub aligned: bool,
}

impl Default for Flags {
    fn default() -> Self {
        Flags {
            c_contiguous: true,
            f_contiguous: true,
            writeable: true,
            owndata: true,
            aligned: true,
        }
    }
}

#[derive(Clone)]
pub struct NdArray {
    pub buffer: Arc<Buffer>,
    /// Offset of element (0, 0, ...) from the start of the buffer, in bytes.
    pub byte_offset: isize,
    pub shape: Vec<isize>,
    /// Strides in BYTES, as in numpy.
    pub strides: Vec<isize>,
    pub dtype: DType,
    pub flags: Flags,
}

/// C-order strides (in bytes) for `shape`.
pub fn c_strides(shape: &[isize], itemsize: usize) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc = itemsize as isize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i].max(0);
    }
    strides
}

/// Fortran-order strides (in bytes) for `shape`.
pub fn f_strides(shape: &[isize], itemsize: usize) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc = itemsize as isize;
    for i in 0..shape.len() {
        strides[i] = acc;
        acc *= shape[i].max(0);
    }
    strides
}

pub fn shape_size(shape: &[isize]) -> usize {
    shape.iter().map(|&d| d.max(0) as usize).product()
}

/// Contiguity test using numpy's rules: dimensions of length <= 1 impose no
/// stride constraint, and a zero-sized array is both C- and F-contiguous.
pub fn is_c_contiguous(shape: &[isize], strides: &[isize], itemsize: usize) -> bool {
    if shape_size(shape) == 0 {
        return true;
    }
    let mut expected = itemsize as isize;
    for i in (0..shape.len()).rev() {
        if shape[i] != 1 {
            if strides[i] != expected {
                return false;
            }
            expected *= shape[i];
        }
    }
    true
}

pub fn is_f_contiguous(shape: &[isize], strides: &[isize], itemsize: usize) -> bool {
    if shape_size(shape) == 0 {
        return true;
    }
    let mut expected = itemsize as isize;
    for i in 0..shape.len() {
        if shape[i] != 1 {
            if strides[i] != expected {
                return false;
            }
            expected *= shape[i];
        }
    }
    true
}

impl NdArray {
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        shape_size(&self.shape)
    }

    pub fn itemsize(&self) -> usize {
        self.dtype.itemsize()
    }

    pub fn nbytes(&self) -> usize {
        self.size() * self.itemsize()
    }

    pub fn is_c_contiguous(&self) -> bool {
        self.flags.c_contiguous
    }

    pub fn is_f_contiguous(&self) -> bool {
        self.flags.f_contiguous
    }

    /// Recompute the contiguity flags from shape/strides.
    pub fn update_flags(&mut self) {
        let isz = self.itemsize();
        self.flags.c_contiguous = is_c_contiguous(&self.shape, &self.strides, isz);
        self.flags.f_contiguous = is_f_contiguous(&self.shape, &self.strides, isz);
        self.flags.aligned =
            (self.buffer.as_ptr() as usize).wrapping_add(self.byte_offset as usize)
                % self.dtype.alignment()
                == 0;
    }

    fn new_uninit(shape: Vec<isize>, dtype: DType) -> Result<NdArray> {
        for &d in &shape {
            if d < 0 {
                return Err(Error::ValueError(
                    "negative dimensions are not allowed".into(),
                ));
            }
        }
        let n = shape_size(&shape)
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ValueError("array is too big".into()))?;
        let strides = c_strides(&shape, dtype.itemsize());
        let mut a = NdArray {
            buffer: Arc::new(Buffer::uninitialized(n)),
            byte_offset: 0,
            shape,
            strides,
            dtype,
            flags: Flags::default(),
        };
        a.update_flags();
        Ok(a)
    }

    /// `np.empty` — uninitialised contents.
    pub fn empty(shape: Vec<isize>, dtype: DType) -> Result<NdArray> {
        NdArray::new_uninit(shape, dtype)
    }

    /// `np.zeros`. Every supported dtype has an all-zero-bytes zero value.
    pub fn zeros(shape: Vec<isize>, dtype: DType) -> Result<NdArray> {
        for &d in &shape {
            if d < 0 {
                return Err(Error::ValueError(
                    "negative dimensions are not allowed".into(),
                ));
            }
        }
        let n = shape_size(&shape)
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ValueError("array is too big".into()))?;
        let strides = c_strides(&shape, dtype.itemsize());
        let mut a = NdArray {
            buffer: Arc::new(Buffer::zeroed(n)),
            byte_offset: 0,
            shape,
            strides,
            dtype,
            flags: Flags::default(),
        };
        a.update_flags();
        Ok(a)
    }

    /// `np.full` — every element set to `value` (cast to `dtype`).
    pub fn full(shape: Vec<isize>, dtype: DType, value: Scalar) -> Result<NdArray> {
        let mut a = NdArray::new_uninit(shape, dtype)?;
        a.fill(value);
        Ok(a)
    }

    pub fn ones(shape: Vec<isize>, dtype: DType) -> Result<NdArray> {
        NdArray::full(shape, dtype, Scalar::Int(1))
    }

    /// Wrap raw bytes (copied) as a C-contiguous array.
    pub fn from_bytes(bytes: &[u8], shape: Vec<isize>, dtype: DType) -> Result<NdArray> {
        let need = shape_size(&shape) * dtype.itemsize();
        if bytes.len() < need {
            return Err(Error::ValueError(format!(
                "buffer is smaller than requested size ({} < {})",
                bytes.len(),
                need
            )));
        }
        let strides = c_strides(&shape, dtype.itemsize());
        let mut a = NdArray {
            buffer: Arc::new(Buffer::from_bytes(&bytes[..need])),
            byte_offset: 0,
            shape,
            strides,
            dtype,
            flags: Flags::default(),
        };
        a.update_flags();
        Ok(a)
    }

    /// Build a 1-D array from scalars, casting each to `dtype`.
    pub fn from_scalars(values: &[Scalar], dtype: DType) -> Result<NdArray> {
        let mut a = NdArray::new_uninit(vec![values.len() as isize], dtype)?;
        for (i, &v) in values.iter().enumerate() {
            a.set_flat(i, v);
        }
        Ok(a)
    }

    /// `np.arange(start, stop, step, dtype)`.
    pub fn arange(start: f64, stop: f64, step: f64, dtype: DType) -> Result<NdArray> {
        if step == 0.0 {
            return Err(Error::ValueError("arange: step cannot be zero".into()));
        }
        let n = ((stop - start) / step).ceil();
        let n = if n.is_finite() && n > 0.0 { n as isize } else { 0 };
        let mut a = NdArray::new_uninit(vec![n], dtype)?;
        if dtype.is_exact() {
            let (s0, st) = (start as i64, step as i64);
            for i in 0..n {
                a.set_flat(i as usize, Scalar::Int(s0 + st * i as i64));
            }
            return Ok(a);
        }
        // numpy's `@TYPE@_fill` writes the first two elements, derives
        // `delta = buf[1] - buf[0]` *in the array's own precision*, and then
        // fills `buf[i] = start + i * delta`. Two subtleties, both verified
        // against real numpy: rounding makes `delta` differ from `step` in the
        // last bit, and the C expression is compiled with FMA contraction, so
        // `start + i * delta` is a *single* rounding. Using `+`/`*` separately
        // diverges from numpy by an ULP on roughly a quarter of inputs.
        if dtype == DType::F32 {
            let s0 = start as f32;
            let s1 = s0 + step as f32;
            let delta = s1 - s0;
            for i in 0..n {
                let v = match i {
                    0 => s0,
                    1 => s1,
                    _ => (i as f32).mul_add(delta, s0),
                };
                a.set_flat(i as usize, Scalar::Float(v as f64));
            }
        } else {
            let s1 = start + step;
            let delta = s1 - start;
            for i in 0..n {
                let v = match i {
                    0 => start,
                    1 => s1,
                    _ => (i as f64).mul_add(delta, start),
                };
                a.set_flat(i as usize, Scalar::Float(v));
            }
        }
        Ok(a)
    }

    /// Byte address of the element at `index` (assumed in bounds).
    #[inline]
    pub fn byte_index(&self, index: &[isize]) -> isize {
        let mut off = self.byte_offset;
        for (i, &ix) in index.iter().enumerate() {
            off += ix * self.strides[i];
        }
        off
    }

    #[inline]
    fn check_index(&self, index: &[isize]) -> Result<()> {
        if index.len() != self.ndim() {
            return Err(Error::IndexError(format!(
                "index has {} dimensions, array has {}",
                index.len(),
                self.ndim()
            )));
        }
        for (i, &ix) in index.iter().enumerate() {
            if ix < 0 || ix >= self.shape[i] {
                return Err(Error::IndexError(format!(
                    "index {} is out of bounds for axis {} with size {}",
                    ix, i, self.shape[i]
                )));
            }
        }
        Ok(())
    }

    /// Read the element at a multi-index.
    pub fn get(&self, index: &[isize]) -> Result<Scalar> {
        self.check_index(index)?;
        Ok(self.read_at(self.byte_index(index)))
    }

    /// Write a (cast) scalar at a multi-index.
    pub fn set(&mut self, index: &[isize], value: Scalar) -> Result<()> {
        if !self.flags.writeable {
            return Err(Error::ValueError(
                "assignment destination is read-only".into(),
            ));
        }
        self.check_index(index)?;
        let off = self.byte_index(index);
        self.write_at(off, value);
        Ok(())
    }

    /// Read the element at a byte offset from the buffer start.
    #[inline]
    pub fn read_at(&self, byte_off: isize) -> Scalar {
        // SAFETY: `byte_off` is produced by `byte_index` from an in-bounds
        // index (or by an iterator over in-bounds indices), so the itemsize
        // bytes at that offset lie inside the allocation.
        unsafe {
            let p = self.buffer.as_ptr().offset(byte_off);
            crate::dispatch_dtype!(self.dtype, T, {
                (std::ptr::read_unaligned(p as *const T)).to_scalar()
            })
        }
    }

    /// Write a scalar (cast to the array dtype) at a byte offset.
    #[inline]
    pub fn write_at(&self, byte_off: isize, value: Scalar) {
        // SAFETY: as `read_at`; the caller guarantees the offset is in range,
        // and the array is known writeable.
        unsafe {
            let p = self.buffer.as_mut_ptr().offset(byte_off);
            crate::dispatch_dtype!(self.dtype, T, {
                std::ptr::write_unaligned(p as *mut T, T::from_scalar(value))
            })
        }
    }

    /// Read the `i`-th element in C order (logical flat index).
    pub fn get_flat(&self, i: usize) -> Scalar {
        self.read_at(self.flat_byte_offset(i))
    }

    /// Write the `i`-th element in C order.
    pub fn set_flat(&mut self, i: usize, v: Scalar) {
        let off = self.flat_byte_offset(i);
        self.write_at(off, v);
    }

    #[inline]
    fn flat_byte_offset(&self, mut i: usize) -> isize {
        if self.flags.c_contiguous {
            return self.byte_offset + (i * self.itemsize()) as isize;
        }
        let mut off = self.byte_offset;
        for ax in (0..self.ndim()).rev() {
            let d = self.shape[ax].max(1) as usize;
            off += (i % d) as isize * self.strides[ax];
            i /= d;
        }
        off
    }

    /// Set every element to `value`.
    pub fn fill(&mut self, value: Scalar) {
        let n = self.size();
        if self.flags.c_contiguous {
            let isz = self.itemsize() as isize;
            let base = self.byte_offset;
            for i in 0..n as isize {
                self.write_at(base + i * isz, value);
            }
        } else {
            for off in crate::iter::offsets(&self.shape, &self.strides, self.byte_offset) {
                self.write_at(off, value);
            }
        }
    }

    /// A C-contiguous copy (`np.copy`).
    pub fn copy(&self) -> NdArray {
        let mut out = NdArray::new_uninit(self.shape.clone(), self.dtype).expect("copy alloc");
        if self.flags.c_contiguous {
            let n = self.nbytes();
            // SAFETY: both regions hold `n` valid bytes and are distinct
            // allocations (`out` was freshly allocated).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buffer.as_ptr().offset(self.byte_offset),
                    out.buffer.as_mut_ptr(),
                    n,
                );
            }
        } else {
            let isz = self.itemsize();
            let mut k = 0isize;
            for off in crate::iter::offsets(&self.shape, &self.strides, self.byte_offset) {
                // SAFETY: `off` is an in-bounds element offset in self, and
                // `k * isz` is in bounds in the freshly allocated output.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.buffer.as_ptr().offset(off),
                        out.buffer.as_mut_ptr().offset(k * isz as isize),
                        isz,
                    );
                }
                k += 1;
            }
        }
        out.update_flags();
        out
    }

    /// `astype` with numpy's `unsafe` casting rules. Always returns a fresh
    /// C-contiguous array (numpy's `copy=True` default).
    pub fn astype(&self, dtype: DType) -> NdArray {
        if dtype == self.dtype {
            return self.copy();
        }
        let mut out = NdArray::new_uninit(self.shape.clone(), dtype).expect("astype alloc");
        let n = self.size();
        if self.flags.c_contiguous {
            // Typed inner loop: both sides are flat, so the per-element cast
            // inlines instead of routing through the `Scalar` enum.
            crate::dispatch_dtype!(self.dtype, S, {
                crate::dispatch_dtype!(dtype, D, {
                    // SAFETY: `self` is contiguous with n elements of S at
                    // `byte_offset`, and `out` was freshly allocated with n
                    // elements of D.
                    unsafe {
                        let src = self.buffer.as_ptr().offset(self.byte_offset) as *const S;
                        let dst = out.buffer.as_mut_ptr() as *mut D;
                        for i in 0..n {
                            *dst.add(i) = D::from_scalar((*src.add(i)).to_scalar());
                        }
                    }
                })
            });
            out.update_flags();
            return out;
        }
        let dst_isz = dtype.itemsize() as isize;
        let mut k = 0isize;
        for off in crate::iter::offsets(&self.shape, &self.strides, self.byte_offset) {
            let v = self.read_at(off);
            out.write_at(k * dst_isz, v);
            k += 1;
        }
        out.update_flags();
        out
    }

    /// A transposed view (reversed axes).
    pub fn transpose(&self) -> NdArray {
        let mut out = self.clone();
        out.shape.reverse();
        out.strides.reverse();
        out.flags.owndata = false;
        out.update_flags();
        out
    }

    /// Permute axes into `axes` order (a view).
    pub fn permute(&self, axes: &[usize]) -> Result<NdArray> {
        if axes.len() != self.ndim() {
            return Err(Error::ValueError("axes don't match array".into()));
        }
        let mut seen = vec![false; self.ndim()];
        let mut out = self.clone();
        for (i, &ax) in axes.iter().enumerate() {
            if ax >= self.ndim() || seen[ax] {
                return Err(Error::ValueError("repeated or invalid axis".into()));
            }
            seen[ax] = true;
            out.shape[i] = self.shape[ax];
            out.strides[i] = self.strides[ax];
        }
        out.flags.owndata = false;
        out.update_flags();
        Ok(out)
    }

    /// Resolve `-1` and validate a reshape target against `self.size()`.
    pub fn resolve_shape(&self, spec: &[isize]) -> Result<Vec<isize>> {
        let size = self.size();
        let mut unknown = None;
        let mut known: usize = 1;
        for (i, &d) in spec.iter().enumerate() {
            if d == -1 {
                if unknown.is_some() {
                    return Err(Error::ValueError(
                        "can only specify one unknown dimension".into(),
                    ));
                }
                unknown = Some(i);
            } else if d < 0 {
                return Err(Error::ValueError("negative dimensions not allowed".into()));
            } else {
                known = known.saturating_mul(d as usize);
            }
        }
        let mut out: Vec<isize> = spec.to_vec();
        if let Some(i) = unknown {
            if known == 0 || size % known != 0 {
                return Err(Error::ValueError(format!(
                    "cannot reshape array of size {} into shape {:?}",
                    size, spec
                )));
            }
            out[i] = (size / known) as isize;
        } else if known != size {
            return Err(Error::ValueError(format!(
                "cannot reshape array of size {} into shape {:?}",
                size, spec
            )));
        }
        Ok(out)
    }

    /// Reshape. Returns a view when the array is C-contiguous, else a copy.
    pub fn reshape(&self, spec: &[isize]) -> Result<NdArray> {
        let shape = self.resolve_shape(spec)?;
        if self.flags.c_contiguous {
            let mut out = self.clone();
            out.strides = c_strides(&shape, self.itemsize());
            out.shape = shape;
            out.flags.owndata = false;
            out.update_flags();
            Ok(out)
        } else {
            let mut out = self.copy();
            out.strides = c_strides(&shape, self.itemsize());
            out.shape = shape;
            out.update_flags();
            Ok(out)
        }
    }

    /// A view with an axis of length 1 sliced away, or with a sub-range taken.
    /// `start` is the index of the first element along `axis`, `len` the count,
    /// `step` the stride multiplier.
    pub fn slice_axis(&self, axis: usize, start: isize, len: isize, step: isize) -> NdArray {
        let mut out = self.clone();
        out.byte_offset += start * self.strides[axis];
        out.shape[axis] = len.max(0);
        out.strides[axis] = self.strides[axis] * step;
        out.flags.owndata = false;
        out.update_flags();
        out
    }

    /// Drop `axis` (which must have been reduced to a single index already).
    pub fn remove_axis(&self, axis: usize) -> NdArray {
        let mut out = self.clone();
        out.shape.remove(axis);
        out.strides.remove(axis);
        out.flags.owndata = false;
        out.update_flags();
        out
    }

    /// Insert a length-1 axis at `axis` (numpy's `newaxis`).
    pub fn insert_axis(&self, axis: usize) -> NdArray {
        let mut out = self.clone();
        out.shape.insert(axis, 1);
        // numpy gives a newaxis a stride of 0 (`a[None].strides == (0, ...)`).
        out.strides.insert(axis, 0);
        out.flags.owndata = false;
        out.update_flags();
        out
    }

    /// Every element in C order, as scalars.
    pub fn to_vec(&self) -> Vec<Scalar> {
        crate::iter::offsets(&self.shape, &self.strides, self.byte_offset)
            .map(|off| self.read_at(off))
            .collect()
    }

    /// The raw bytes of the element at `byte_off`.
    ///
    /// This is how flexible dtypes (`S`/`U`/`V` and structured records) are
    /// read and written: they have no `Scalar` representation.
    pub fn raw_bytes_at(&self, byte_off: isize) -> &[u8] {
        // SAFETY: `byte_off` addresses one in-bounds element, so `itemsize`
        // bytes from there lie inside the allocation, and the borrow is tied
        // to `&self` which keeps the buffer alive.
        unsafe {
            std::slice::from_raw_parts(self.buffer.as_ptr().offset(byte_off), self.itemsize())
        }
    }

    /// Write raw element bytes, zero-padding or truncating to `itemsize`.
    pub fn write_raw_at(&self, byte_off: isize, src: &[u8]) {
        let n = self.itemsize();
        let k = src.len().min(n);
        // SAFETY: as `raw_bytes_at`; the destination holds `itemsize` bytes.
        unsafe {
            let p = self.buffer.as_mut_ptr().offset(byte_off);
            std::ptr::copy_nonoverlapping(src.as_ptr(), p, k);
            std::ptr::write_bytes(p.add(k), 0, n - k);
        }
    }

    /// Typed contiguous slice, if the array is C-contiguous and matches `T`.
    pub fn as_slice<T: Element>(&self) -> Option<&[T]> {
        if self.dtype != T::DTYPE || !self.flags.c_contiguous {
            return None;
        }
        // SAFETY: the array is C-contiguous with `size()` elements of exactly
        // `T`'s dtype starting at `byte_offset`, all inside the allocation.
        unsafe {
            Some(std::slice::from_raw_parts(
                self.buffer.as_ptr().offset(self.byte_offset) as *const T,
                self.size(),
            ))
        }
    }
}

impl std::fmt::Debug for NdArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NdArray")
            .field("dtype", &self.dtype.name())
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_match_numpy() {
        // np.zeros((2,3,4), np.int32).strides == (48, 16, 4)
        assert_eq!(c_strides(&[2, 3, 4], 4), vec![48, 16, 4]);
        // np.zeros((2,3,4), np.float64, order='F').strides == (8, 16, 48)
        assert_eq!(f_strides(&[2, 3, 4], 8), vec![8, 16, 48]);
        assert_eq!(c_strides(&[], 8), Vec::<isize>::new());
    }

    #[test]
    fn zeros_and_flags() {
        let a = NdArray::zeros(vec![2, 3], DType::F64).unwrap();
        assert_eq!(a.size(), 6);
        assert_eq!(a.nbytes(), 48);
        assert!(a.flags.c_contiguous);
        assert!(!a.flags.f_contiguous); // 2x3 is not both
        assert!(a.flags.owndata);
        for i in 0..6 {
            assert_eq!(a.get_flat(i), Scalar::Float(0.0));
        }
    }

    #[test]
    fn one_d_is_both_contiguous() {
        let a = NdArray::zeros(vec![5], DType::I32).unwrap();
        assert!(a.flags.c_contiguous && a.flags.f_contiguous);
        let s = NdArray::zeros(vec![], DType::I32).unwrap();
        assert!(s.flags.c_contiguous && s.flags.f_contiguous);
        assert_eq!(s.size(), 1);
    }

    #[test]
    fn arange_int_and_float() {
        let a = NdArray::arange(0.0, 5.0, 1.0, DType::I64).unwrap();
        assert_eq!(a.shape, vec![5]);
        assert_eq!(a.to_vec(), (0..5).map(Scalar::Int).collect::<Vec<_>>());
        let b = NdArray::arange(0.0, 3.0, 0.5, DType::F64).unwrap();
        assert_eq!(b.shape, vec![6]);
        assert_eq!(b.get_flat(3), Scalar::Float(1.5));
        // Empty when the range is degenerate (matches numpy).
        assert_eq!(NdArray::arange(5.0, 0.0, 1.0, DType::I64).unwrap().size(), 0);
    }

    #[test]
    fn arange_float_matches_numpy_bit_for_bit() {
        // Probed: np.arange(-2.5365855523037606, 14.370417634771437,
        //                   0.8450327557930141)[5] on numpy 2.5.2.
        let a = NdArray::arange(
            -2.5365855523037606,
            14.370417634771437,
            0.8450327557930141,
            DType::F64,
        )
        .unwrap();
        assert_eq!(a.size(), 21);
        match a.get_flat(5) {
            Scalar::Float(f) => assert_eq!(f.to_bits(), 1.6885782266613103f64.to_bits()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn transpose_is_a_view_with_reversed_strides() {
        let a = NdArray::zeros(vec![2, 3], DType::I32).unwrap();
        let t = a.transpose();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.strides, vec![4, 12]);
        assert!(t.flags.f_contiguous);
        assert!(!t.flags.c_contiguous);
        assert!(Arc::ptr_eq(&a.buffer, &t.buffer));
    }

    #[test]
    fn reshape_views_share_the_buffer() {
        let a = NdArray::arange(0.0, 6.0, 1.0, DType::I32).unwrap();
        let r = a.reshape(&[2, 3]).unwrap();
        assert_eq!(r.shape, vec![2, 3]);
        assert!(Arc::ptr_eq(&a.buffer, &r.buffer));
        assert_eq!(r.get(&[1, 2]).unwrap(), Scalar::Int(5));
        let r2 = a.reshape(&[-1, 2]).unwrap();
        assert_eq!(r2.shape, vec![3, 2]);
        assert!(a.reshape(&[4, 2]).is_err());
    }

    #[test]
    fn slicing_produces_correct_views() {
        let a = NdArray::arange(0.0, 10.0, 1.0, DType::I64).unwrap();
        let s = a.slice_axis(0, 1, 4, 2); // a[1:9:2]
        assert_eq!(s.shape, vec![4]);
        assert_eq!(s.strides, vec![16]);
        assert!(!s.flags.c_contiguous);
        assert_eq!(
            s.to_vec(),
            vec![
                Scalar::Int(1),
                Scalar::Int(3),
                Scalar::Int(5),
                Scalar::Int(7)
            ]
        );
        assert_eq!(s.copy().to_vec(), s.to_vec());
    }

    #[test]
    fn astype_round_trips_for_all_pairs() {
        use crate::dtype::ALL_DTYPES;
        // Values chosen to be exactly representable in every dtype.
        let src = NdArray::from_scalars(
            &[Scalar::Int(0), Scalar::Int(1), Scalar::Int(3)],
            DType::I64,
        )
        .unwrap();
        for a in ALL_DTYPES {
            for b in ALL_DTYPES {
                if a == DType::Bool || b == DType::Bool {
                    continue;
                }
                let x = src.astype(a);
                let y = x.astype(b);
                let z = y.astype(a);
                assert_eq!(x.to_vec(), z.to_vec(), "round trip {a} -> {b} -> {a}");
                assert_eq!(y.dtype, b);
                assert_eq!(y.shape, vec![3]);
                assert!(y.flags.c_contiguous);
            }
        }
    }

    #[test]
    fn astype_of_a_strided_view_reads_the_right_elements() {
        let a = NdArray::arange(0.0, 10.0, 1.0, DType::I64).unwrap();
        let s = a.slice_axis(0, 0, 5, 2);
        let f = s.astype(DType::F32);
        assert_eq!(
            f.to_vec(),
            vec![0.0, 2.0, 4.0, 6.0, 8.0]
                .into_iter()
                .map(Scalar::Float)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fill_and_full() {
        let a = NdArray::full(vec![2, 2], DType::F32, Scalar::Float(2.5)).unwrap();
        assert!(a.to_vec().iter().all(|&v| v == Scalar::Float(2.5)));
        let o = NdArray::ones(vec![3], DType::U8).unwrap();
        assert_eq!(o.to_vec(), vec![Scalar::Uint(1); 3]);
    }

    #[test]
    fn out_of_bounds_index_errors() {
        let a = NdArray::zeros(vec![2, 3], DType::I32).unwrap();
        assert!(a.get(&[2, 0]).is_err());
        assert!(a.get(&[0]).is_err());
        assert!(a.get(&[1, 2]).is_ok());
    }
}
