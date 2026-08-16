//! The `NdArray` header: a dtype + shape + byte-strides view onto a shared
//! `Buffer`, exactly mirroring numpy's memory model.

use std::sync::Arc;

use crate::buffer::Buffer;
use crate::descr::{ByteOrder, Descr};
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
    /// The full descriptor — storage type *plus* byte order and C-type alias.
    ///
    /// The header carries a `Descr` rather than a bare [`DType`] for two
    /// reasons numpy's own `PyArrayObject` carries `PyArray_Descr*`: an array
    /// of `'>i4'` differs from one of `'<i4'` only in the descriptor, and
    /// `np.array(np.longlong(2)).dtype.type is np.longlong` only holds if the
    /// array remembers which C-type spelling it was built from.
    pub descr: Descr,
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

/// The element count, or `None` when the product overflows.
///
/// The unchecked [`shape_size`] is what every in-bounds walk uses; this one
/// guards *allocation*, where `np.zeros([975] * 7)` would otherwise wrap to a
/// small number, hand a bogus length to the allocator and abort the process.
pub fn checked_shape_size(shape: &[isize]) -> Option<usize> {
    let mut n: usize = 1;
    for &d in shape {
        n = n.checked_mul(d.max(0) as usize)?;
    }
    Some(n)
}

/// numpy's message when a requested array cannot be represented.
fn too_big() -> Error {
    Error::ValueError("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.".into())
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

/// Every dtype the engine can allocate.
///
/// `object` arrays store an 8-byte *handle* into the interning slab that
/// lives on the Python side (`rnp-python/src/objects.rs`); slot 0 is `None`,
/// which is why a zeroed object array reads back as `None` just like numpy's.
/// The largest allocation the port will attempt, mirroring numpy's own
/// refusal to try (it raises rather than letting the allocator fail): half of
/// the 64-bit address space is far beyond any real machine.
const MAX_ALLOC: usize = 1usize << 47;

pub fn check_storable(_dtype: DType) -> Result<()> {
    Ok(())
}

/// How one element of `dt` decomposes for byte-swapping: `(unit, count)`
/// means `count` consecutive runs of `unit` bytes, each reversed.
///
/// `None` means the type has no byte order at all (`S`, `V`, `O`, 1-byte
/// types). Structured and subarray dtypes are handled recursively by
/// [`swap_element`], not here, because their members can disagree.
pub fn swap_layout(dt: DType) -> Option<(usize, usize)> {
    match dt {
        DType::Bool | DType::I8 | DType::U8 => None,
        DType::I16 | DType::U16 | DType::F16 => Some((2, 1)),
        DType::I32 | DType::U32 | DType::F32 => Some((4, 1)),
        DType::I64 | DType::U64 | DType::F64 => Some((8, 1)),
        DType::DateTime(_) | DType::TimeDelta(_) => Some((8, 1)),
        // A complex is two reals side by side; the *halves* swap, the pair
        // order does not. This is what numpy's `@TYPE@_copyswapn` does.
        DType::C64 => Some((4, 2)),
        DType::C128 => Some((8, 2)),
        // `U<n>` is n UCS4 code points, each swapped independently.
        DType::Str(n) => {
            if n == 0 {
                None
            } else {
                Some((4, n as usize))
            }
        }
        DType::Bytes(_) | DType::Void(_) | DType::Object => None,
        DType::Struct(_) | DType::SubArray(_) => None,
    }
}

/// Byte-swap one element of `descr` in place at `p`.
///
/// # Safety
/// `p` must point at `descr.itemsize()` writable bytes.
pub unsafe fn swap_element(p: *mut u8, descr: Descr) {
    match descr.dt {
        DType::Struct(id) => {
            let def = crate::descr::registry::struct_def(id);
            for f in &def.fields {
                // SAFETY: field offsets are inside the record by construction.
                unsafe { swap_element(p.add(f.offset), f.descr) };
            }
        }
        DType::SubArray(id) => {
            let def = crate::descr::registry::subarray_def(id);
            let n = shape_size(&def.shape);
            let isz = def.base.itemsize();
            for i in 0..n {
                // SAFETY: the subarray holds `n` base elements contiguously.
                unsafe { swap_element(p.add(i * isz), def.base) };
            }
        }
        _ => {
            if let Some((unit, count)) = swap_layout(descr.dt) {
                for i in 0..count {
                    // SAFETY: `unit * count <= itemsize` for every layout above.
                    unsafe {
                        let q = p.add(i * unit);
                        for k in 0..unit / 2 {
                            std::ptr::swap(q.add(k), q.add(unit - 1 - k));
                        }
                    }
                }
            }
        }
    }
}

impl NdArray {
    /// The storage dtype. Byte order and C-type alias live in [`Self::descr`].
    #[inline]
    pub fn dtype(&self) -> DType {
        self.descr.dt
    }

    /// True when the elements are stored in the host's byte order, which is
    /// the only case the compute loops ever see: every entry point calls
    /// [`Self::to_native`] first, and that is a no-op here.
    #[inline(always)]
    pub fn is_native(&self) -> bool {
        // One byte compare for every non-compound dtype. The compound case
        // walks the registry, so it is kept out of line: this predicate sits
        // at the top of every hot entry point and must not grow one.
        match self.descr.dt {
            DType::Struct(_) | DType::SubArray(_) => self.compound_is_native(),
            _ => self.descr.bo != ByteOrder::Big,
        }
    }

    #[cold]
    #[inline(never)]
    fn compound_is_native(&self) -> bool {
        self.descr.isnative()
    }

    /// Reverse the bytes of every element in place, leaving the descriptor
    /// alone. This is `ndarray.byteswap(inplace=True)`.
    pub fn byteswap_inplace(&mut self) {
        if swap_layout(self.descr.dt).is_none()
            && !matches!(self.descr.dt, DType::Struct(_) | DType::SubArray(_))
        {
            return;
        }
        let descr = self.descr;
        for off in crate::iter::offsets(&self.shape, &self.strides, self.byte_offset) {
            // SAFETY: `off` addresses one in-bounds element of `itemsize`
            // bytes inside the allocation.
            unsafe { swap_element(self.buffer.as_mut_ptr().offset(off), descr) };
        }
    }

    /// The same values in the host's byte order.
    ///
    /// Returns `self` untouched when the array is already native — which is
    /// the *only* thing this costs on the native path: one predictable
    /// branch, taken once per operand, outside every loop.
    pub fn to_native(&self) -> std::borrow::Cow<'_, NdArray> {
        if self.is_native() {
            return std::borrow::Cow::Borrowed(self);
        }
        let mut out = self.copy();
        out.byteswap_inplace();
        out.descr = self.descr.newbyteorder(Some('=')).unwrap_or(self.descr);
        std::borrow::Cow::Owned(out)
    }

    /// `self` with its storage in the same byte order as `other`, for the raw
    /// element copies that flexible dtypes use (a `'>U3'` destination must
    /// receive big-endian code points).
    pub fn in_order_of(&self, other: &NdArray) -> std::borrow::Cow<'_, NdArray> {
        if self.is_native() == other.is_native() {
            return std::borrow::Cow::Borrowed(self);
        }
        let mut o = self.copy();
        o.byteswap_inplace();
        o.descr = self.descr.newbyteorder(Some('S')).unwrap_or(self.descr);
        std::borrow::Cow::Owned(o)
    }

    /// A copy of `self` whose bytes are in `descr`'s byte order (the storage
    /// dtype is unchanged). Used when writing into a non-native destination.
    pub fn with_byteorder(&self, descr: Descr) -> NdArray {
        debug_assert_eq!(descr.dt, self.descr.dt);
        // A deep copy first: `byteswap_inplace` writes through the shared
        // `Arc<Buffer>`, so it must never run on a header-only clone.
        let mut out = self.copy();
        if out.is_native() != descr.isnative() {
            out.byteswap_inplace();
        }
        out.descr = descr;
        out
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        shape_size(&self.shape)
    }

    pub fn itemsize(&self) -> usize {
        self.dtype().itemsize()
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
                % self.dtype().alignment()
                == 0;
    }

    fn new_uninit(shape: Vec<isize>, dtype: DType) -> Result<NdArray> {
        NdArray::new_uninit_descr(shape, Descr::native(dtype))
    }

    fn new_uninit_descr(shape: Vec<isize>, descr: Descr) -> Result<NdArray> {
        let dtype = descr.dt;
        check_storable(dtype)?;
        for &d in &shape {
            if d < 0 {
                return Err(Error::ValueError(
                    "negative dimensions are not allowed".into(),
                ));
            }
        }
        let n = checked_shape_size(&shape)
            .and_then(|n| n.checked_mul(dtype.itemsize()))
            .filter(|&n| n <= MAX_ALLOC)
            .ok_or_else(too_big)?;
        let strides = c_strides(&shape, dtype.itemsize());
        let mut a = NdArray {
            buffer: Arc::new(Buffer::uninitialized(n)),
            byte_offset: 0,
            shape,
            strides,
            descr,
            flags: Flags::default(),
        };
        a.update_flags();
        Ok(a)
    }

    /// `np.empty` — uninitialised contents.
    pub fn empty(shape: Vec<isize>, dtype: DType) -> Result<NdArray> {
        NdArray::new_uninit(shape, dtype)
    }

    pub fn empty_descr(shape: Vec<isize>, descr: Descr) -> Result<NdArray> {
        NdArray::new_uninit_descr(shape, descr)
    }

    /// `np.zeros`. Every supported dtype has an all-zero-bytes zero value
    /// *in either byte order*, so this needs no swap pass.
    pub fn zeros(shape: Vec<isize>, dtype: DType) -> Result<NdArray> {
        NdArray::zeros_descr(shape, Descr::native(dtype))
    }

    pub fn zeros_descr(shape: Vec<isize>, descr: Descr) -> Result<NdArray> {
        let dtype = descr.dt;
        check_storable(dtype)?;
        for &d in &shape {
            if d < 0 {
                return Err(Error::ValueError(
                    "negative dimensions are not allowed".into(),
                ));
            }
        }
        let n = checked_shape_size(&shape)
            .and_then(|n| n.checked_mul(dtype.itemsize()))
            .filter(|&n| n <= MAX_ALLOC)
            .ok_or_else(too_big)?;
        let strides = c_strides(&shape, dtype.itemsize());
        let mut a = NdArray {
            buffer: Arc::new(Buffer::zeroed(n)),
            byte_offset: 0,
            shape,
            strides,
            descr,
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

    pub fn full_descr(shape: Vec<isize>, descr: Descr, value: Scalar) -> Result<NdArray> {
        let mut a = NdArray::new_uninit_descr(shape, descr)?;
        a.fill(value);
        Ok(a)
    }

    pub fn ones(shape: Vec<isize>, dtype: DType) -> Result<NdArray> {
        NdArray::full(shape, dtype, Scalar::Int(1))
    }

    pub fn ones_descr(shape: Vec<isize>, descr: Descr) -> Result<NdArray> {
        NdArray::full_descr(shape, descr, Scalar::Int(1))
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
            descr: Descr::native(dtype),
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
        if self.dtype().is_flexible() || self.dtype().is_object() {
            // Flexible elements have no scalar value; expose the leading
            // bytes so that generic code (copies, comparisons) still works.
            let raw = self.raw_bytes_at(byte_off);
            let mut buf = [0u8; 8];
            let k = raw.len().min(8);
            buf[..k].copy_from_slice(&raw[..k]);
            return Scalar::Uint(u64::from_le_bytes(buf));
        }
        if !self.is_native() {
            return self.read_at_swapped(byte_off);
        }
        // SAFETY: `byte_off` is produced by `byte_index` from an in-bounds
        // index (or by an iterator over in-bounds indices), so the itemsize
        // bytes at that offset lie inside the allocation.
        unsafe {
            let p = self.buffer.as_ptr().offset(byte_off);
            crate::dispatch_dtype!(self.dtype(), T, {
                (std::ptr::read_unaligned(p as *const T)).to_scalar()
            })
        }
    }

    /// The byte-swapped-storage half of [`Self::read_at`], kept out of line so
    /// the native path stays a single predictable branch.
    #[cold]
    fn read_at_swapped(&self, byte_off: isize) -> Scalar {
        let isz = self.itemsize();
        debug_assert!(isz <= 16);
        let mut buf = [0u8; 16];
        // SAFETY: `byte_off` addresses `isz <= 16` in-bounds bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.as_ptr().offset(byte_off),
                buf.as_mut_ptr(),
                isz,
            );
            swap_element(buf.as_mut_ptr(), self.descr);
            crate::dispatch_dtype!(self.dtype(), T, {
                (std::ptr::read_unaligned(buf.as_ptr() as *const T)).to_scalar()
            })
        }
    }

    /// Write a scalar (cast to the array dtype) at a byte offset.
    #[inline]
    pub fn write_at(&self, byte_off: isize, value: Scalar) {
        if self.dtype().is_flexible() || self.dtype().is_object() {
            // As `read_at`: store the scalar's little-endian bytes, padded or
            // truncated to the element size. Nothing interprets them, so no
            // wrong *numeric* answer can escape.
            let bits: u64 = match value {
                Scalar::Bool(b) => b as u64,
                Scalar::Int(i) => i as u64,
                Scalar::Uint(u) => u,
                Scalar::Float(f) => f.to_bits(),
                Scalar::Complex(c) => c.re.to_bits(),
            };
            self.write_raw_at(byte_off, &bits.to_le_bytes());
            return;
        }
        if !self.is_native() {
            return self.write_at_swapped(byte_off, value);
        }
        // SAFETY: as `read_at`; the caller guarantees the offset is in range,
        // and the array is known writeable.
        unsafe {
            let p = self.buffer.as_mut_ptr().offset(byte_off);
            crate::dispatch_dtype!(self.dtype(), T, {
                std::ptr::write_unaligned(p as *mut T, T::from_scalar(value))
            })
        }
    }

    /// The byte-swapped-storage half of [`Self::write_at`].
    #[cold]
    fn write_at_swapped(&self, byte_off: isize, value: Scalar) {
        let isz = self.itemsize();
        debug_assert!(isz <= 16);
        let mut buf = [0u8; 16];
        // SAFETY: the scratch buffer holds `isz <= 16` bytes; the destination
        // holds `isz` in-bounds bytes.
        unsafe {
            crate::dispatch_dtype!(self.dtype(), T, {
                std::ptr::write_unaligned(buf.as_mut_ptr() as *mut T, T::from_scalar(value))
            });
            swap_element(buf.as_mut_ptr(), self.descr);
            std::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                self.buffer.as_mut_ptr().offset(byte_off),
                isz,
            );
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
        let mut out = NdArray::new_uninit_descr(self.shape.clone(), self.descr).expect("copy alloc");
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
        if !self.is_native() {
            return self.astype_swapped(dtype);
        }
        self.astype_native(dtype)
    }

    /// numpy's `astype` always lands in the native order unless a
    /// byte-swapped *descriptor* was asked for (see [`Self::astype_descr`]).
    ///
    /// Kept out of line so that [`Self::astype_native`] — which carries the
    /// vectorised cast kernel — is neither recursive nor inlined into a cold
    /// path.
    #[cold]
    #[inline(never)]
    fn astype_swapped(&self, dtype: DType) -> NdArray {
        let n = self.to_native();
        if dtype == n.dtype() {
            n.into_owned()
        } else {
            n.astype_native(dtype)
        }
    }

    fn astype_native(&self, dtype: DType) -> NdArray {
        if dtype == self.dtype() {
            return self.copy();
        }
        let mut out = NdArray::new_uninit(self.shape.clone(), dtype).expect("astype alloc");
        let n = self.size();
        if self.flags.c_contiguous {
            // Typed inner loop: both sides are flat, so the per-element cast
            // inlines instead of routing through the `Scalar` enum.
            crate::dispatch_dtype!(self.dtype(), S, {
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

    /// `astype` to a full descriptor: cast the values through the native
    /// path, then store them in the target's byte order.
    pub fn astype_descr(&self, d: Descr) -> NdArray {
        let src = self.to_native();
        let mut out = src.astype(d.dt);
        out.descr = Descr::native(d.dt);
        out.into_descr(d)
    }

    /// Re-label a uniquely-owned array with `descr`, byte-swapping the
    /// storage when the new descriptor is not native. The storage dtype must
    /// not change.
    pub fn into_descr(mut self, descr: Descr) -> NdArray {
        debug_assert_eq!(descr.dt, self.descr.dt);
        if self.is_native() == descr.isnative() {
            self.descr = descr;
            return self;
        }
        // `byteswap_inplace` writes through the buffer, so a shared one has
        // to be copied first.
        if Arc::strong_count(&self.buffer) != 1 {
            self = self.copy();
        }
        self.byteswap_inplace();
        self.descr = descr;
        self
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

    /// The element bytes at `byte_off`, in the *host's* byte order.
    ///
    /// `raw_bytes_at` hands back storage bytes, which for a `'>U3'` array are
    /// not the ones any interpreter wants; this is the accessor for code that
    /// reads a flexible element's *value*.
    pub fn element_bytes_at(&self, byte_off: isize) -> std::borrow::Cow<'_, [u8]> {
        let raw = self.raw_bytes_at(byte_off);
        if self.is_native() {
            return std::borrow::Cow::Borrowed(raw);
        }
        let mut v = raw.to_vec();
        // SAFETY: `v` holds exactly one element's worth of bytes.
        unsafe { swap_element(v.as_mut_ptr(), self.descr) };
        std::borrow::Cow::Owned(v)
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
        if self.dtype() != T::DTYPE || !self.flags.c_contiguous || !self.is_native() {
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
            .field("dtype", &self.dtype().name())
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
                assert_eq!(y.dtype(), b);
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
