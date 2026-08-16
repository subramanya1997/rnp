//! numpy's indexing model: basic indexing (views) and advanced indexing
//! (integer-array and boolean-mask gathers/scatters).
//!
//! The layout of the result of a mixed basic/advanced index is transcribed
//! from `numpy/_core/src/multiarray/mapping.c`:
//!
//! * when any advanced index is present, plain integers are converted to 0-d
//!   integer index arrays (so they take part in broadcasting and in the
//!   "consecutive" test),
//! * a boolean array of `k` dimensions is replaced by the `k` integer arrays
//!   `nonzero()` returns, consuming `k` axes,
//! * a 0-d boolean contributes a length-1 (True) or length-0 (False) index
//!   array and consumes no axis,
//! * the broadcast shape of the index arrays is spliced into the subspace at
//!   the position of the first advanced index when the advanced indices are
//!   *consecutive* (nothing but other advanced indices between them — a slice
//!   or a newaxis in between counts as a separator), and is otherwise placed
//!   at the front.

use crate::array::{shape_size, NdArray};
use crate::dtype::DType;
use crate::element::{Element, Scalar};
use crate::error::{Error, Result};

/// The unresolved components of a Python `slice`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SliceSpec {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: Option<isize>,
}

/// One element of an index expression, after the Python layer has classified
/// it (index arrays are already materialised as `NdArray`s).
#[derive(Clone)]
pub enum IndexItem {
    Int(isize),
    Slice(SliceSpec),
    NewAxis,
    Ellipsis,
    /// An integer-kind index array (any width; values are read as i64).
    IntArray(NdArray),
    /// A boolean mask with `ndim >= 1`.
    BoolArray(NdArray),
    /// A 0-d boolean (`a[True]`, `a[np.array(False)]`).
    ZeroDBool(bool),
}

impl IndexItem {
    fn consumed_axes(&self) -> usize {
        match self {
            IndexItem::Int(_) | IndexItem::Slice(_) | IndexItem::IntArray(_) => 1,
            IndexItem::BoolArray(b) => b.ndim(),
            IndexItem::NewAxis | IndexItem::Ellipsis | IndexItem::ZeroDBool(_) => 0,
        }
    }

    fn is_advanced(&self) -> bool {
        matches!(
            self,
            IndexItem::IntArray(_) | IndexItem::BoolArray(_) | IndexItem::ZeroDBool(_)
        )
    }
}

/// Python's `slice.indices(n)`.
pub fn resolve_slice(spec: SliceSpec, n: isize) -> Result<(isize, isize, isize)> {
    let step = spec.step.unwrap_or(1);
    if step == 0 {
        return Err(Error::ValueError("slice step cannot be zero".into()));
    }
    let (lower, upper) = if step < 0 { (-1, n - 1) } else { (0, n) };
    let clamp = |v: isize| -> isize {
        let v = if v < 0 { v + n } else { v };
        v.max(lower).min(upper)
    };
    let start = match spec.start {
        Some(s) => clamp(s),
        None => {
            if step < 0 {
                n - 1
            } else {
                0
            }
        }
    };
    let stop = match spec.stop {
        Some(s) => clamp(s),
        None => {
            if step < 0 {
                -1
            } else {
                n
            }
        }
    };
    let len = if step > 0 {
        if stop > start {
            (stop - start - 1) / step + 1
        } else {
            0
        }
    } else if stop < start {
        (stop - start + 1) / step + 1
    } else {
        0
    };
    Ok((start, len, step))
}

/// The result of applying an index expression.
pub enum Indexed {
    /// Basic indexing: a view onto the same buffer.
    View { arr: NdArray, scalarize: bool },
    /// Advanced indexing: a gather/scatter plan.
    Fancy(FancyPlan),
}

/// Everything needed to gather from (or scatter into) the source array.
pub struct FancyPlan {
    /// Shape of the indexing *result*.
    pub shape: Vec<isize>,
    /// True when the result is 0-d and no slice/ellipsis/newaxis appeared.
    pub scalarize: bool,
    /// Byte offsets contributed by the broadcast index arrays, in C order
    /// over `b_shape`.
    outer: Vec<isize>,
    /// Shape of the subspace (dims from slices and newaxis), source strides.
    sub_shape: Vec<isize>,
    sub_strides: Vec<isize>,
    sub_base: isize,
    /// Where `b_shape` is spliced into `sub_shape`.
    insert: usize,
}

impl FancyPlan {
    /// Number of elements in the result.
    pub fn len(&self) -> usize {
        shape_size(&self.shape)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Call `f(k, src_byte_offset)` for every result element, in C order.
    ///
    /// The common case (one index array, no subspace) degenerates to a walk
    /// over `outer`, so no second offset buffer is ever built.
    #[inline]
    pub fn for_each<F: FnMut(usize, isize)>(&self, mut f: F) {
        let p: usize = shape_size(&self.sub_shape[..self.insert]);
        let q: usize = shape_size(&self.sub_shape[self.insert..]);
        if p == 1 && q == 1 {
            let base = self.sub_base;
            for (k, &o) in self.outer.iter().enumerate() {
                f(k, o + base);
            }
            return;
        }
        let sub: Vec<isize> =
            crate::iter::offsets(&self.sub_shape, &self.sub_strides, self.sub_base).collect();
        debug_assert_eq!(sub.len(), p * q);
        let mut k = 0usize;
        for pi in 0..p {
            for &o in &self.outer {
                for qi in 0..q {
                    f(k, o + sub[pi * q + qi]);
                    k += 1;
                }
            }
        }
    }

    /// Source byte offsets for every element of the result, in C order.
    pub fn src_offsets(&self) -> Vec<isize> {
        let mut out = vec![0isize; self.len()];
        self.for_each(|k, o| out[k] = o);
        out
    }
}

/// The number of dimensions the index consumes, for error reporting.
fn count_consumed(items: &[IndexItem]) -> usize {
    items.iter().map(|i| i.consumed_axes()).sum()
}

/// Expand a single ellipsis (or append implicit trailing full slices).
fn expand(arr: &NdArray, items: &[IndexItem]) -> Result<Vec<IndexItem>> {
    let mut n_ellipsis = 0usize;
    for it in items {
        if matches!(it, IndexItem::Ellipsis) {
            n_ellipsis += 1;
        }
    }
    if n_ellipsis > 1 {
        return Err(Error::IndexError(
            "an index can only have a single ellipsis ('...')".into(),
        ));
    }
    let consumed = count_consumed(items);
    if consumed > arr.ndim() {
        return Err(Error::IndexError(format!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            arr.ndim(),
            consumed
        )));
    }
    let fill = arr.ndim() - consumed;
    let mut out: Vec<IndexItem> = Vec::with_capacity(items.len() + fill);
    for it in items {
        if matches!(it, IndexItem::Ellipsis) {
            for _ in 0..fill {
                out.push(IndexItem::Slice(SliceSpec::default()));
            }
        } else {
            out.push(it.clone());
        }
    }
    if n_ellipsis == 0 {
        for _ in 0..fill {
            out.push(IndexItem::Slice(SliceSpec::default()));
        }
    }
    Ok(out)
}

/// `np.nonzero`: one intp array per dimension, C-order over the true entries.
pub fn nonzero(arr: &NdArray) -> Vec<NdArray> {
    // Fast path: a contiguous 1-D boolean mask, which is what every
    // `a[mask]` goes through.
    if arr.ndim() == 1 && arr.dtype == DType::Bool && arr.flags.c_contiguous {
        let n = arr.size();
        // SAFETY: `n` contiguous bytes of bool at `byte_offset`.
        let bytes = unsafe {
            std::slice::from_raw_parts(arr.buffer.as_ptr().offset(arr.byte_offset), n)
        };
        let count = bytes.iter().filter(|&&b| b != 0).count();
        let out = NdArray::empty(vec![count as isize], DType::I64).expect("nonzero alloc");
        // SAFETY: `out` was allocated with `count` i64 elements.
        unsafe {
            let p = out.buffer.as_mut_ptr() as *mut i64;
            let mut k = 0usize;
            for (i, &b) in bytes.iter().enumerate() {
                if b != 0 {
                    *p.add(k) = i as i64;
                    k += 1;
                }
            }
        }
        return vec![out];
    }
    let nd = arr.ndim().max(1);
    let mut cols: Vec<Vec<i64>> = vec![Vec::new(); nd];
    if arr.ndim() == 0 {
        if truthy(arr.get_flat(0)) {
            cols[0].push(0);
        }
    } else {
        let mut counter = vec![0isize; arr.ndim()];
        for off in crate::iter::offsets(&arr.shape, &arr.strides, arr.byte_offset) {
            if truthy(arr.read_at(off)) {
                for (ax, c) in counter.iter().enumerate() {
                    cols[ax].push(*c as i64);
                }
            }
            // odometer
            for ax in (0..arr.ndim()).rev() {
                counter[ax] += 1;
                if counter[ax] < arr.shape[ax] {
                    break;
                }
                counter[ax] = 0;
            }
        }
    }
    cols.into_iter()
        .map(|c| {
            let vals: Vec<Scalar> = c.into_iter().map(Scalar::Int).collect();
            NdArray::from_scalars(&vals, DType::I64).expect("nonzero alloc")
        })
        .collect()
}

fn truthy(s: Scalar) -> bool {
    match s {
        Scalar::Bool(b) => b,
        Scalar::Int(i) => i != 0,
        Scalar::Uint(u) => u != 0,
        Scalar::Float(f) => f != 0.0,
        Scalar::Complex(c) => c.re != 0.0 || c.im != 0.0,
    }
}

#[inline]
fn scalar_to_i64(s: Scalar) -> i64 {
    match s {
        Scalar::Int(i) => i,
        Scalar::Uint(u) => u as i64,
        Scalar::Bool(b) => b as i64,
        Scalar::Float(f) => f as i64,
        Scalar::Complex(c) => c.re as i64,
    }
}

/// Resolve one (possibly negative) index against an axis of length `n`.
#[inline]
fn resolve_one(raw: i64, n: i64, axis: usize) -> Result<isize> {
    let v = if raw < 0 { raw + n } else { raw };
    if v < 0 || v >= n {
        return Err(Error::IndexError(format!(
            "index {} is out of bounds for axis {} with size {}",
            raw, axis, n
        )));
    }
    Ok(v as isize)
}

/// Add `index * stride` into `outer` straight from the index array, without
/// materialising the widened index values first.
fn fold_index_array(
    outer: &mut [isize],
    a: &NdArray,
    stride: isize,
    len: isize,
    axis: usize,
) -> Result<()> {
    let n = len as i64;
    if a.flags.c_contiguous && !a.dtype.is_flexible() {
        let mut err: Option<Error> = None;
        crate::dispatch_dtype!(a.dtype, T, {
            // SAFETY: `a` is contiguous with `outer.len()` elements of `T`.
            unsafe {
                let p = a.buffer.as_ptr().offset(a.byte_offset) as *const T;
                for (i, slot) in outer.iter_mut().enumerate() {
                    let raw = scalar_to_i64(std::ptr::read_unaligned(p.add(i)).to_scalar());
                    match resolve_one(raw, n, axis) {
                        Ok(v) => *slot += v * stride,
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
            }
        });
        return match err {
            Some(e) => Err(e),
            None => Ok(()),
        };
    }
    for (slot, o) in outer
        .iter_mut()
        .zip(crate::iter::offsets(&a.shape, &a.strides, a.byte_offset))
    {
        *slot += resolve_one(scalar_to_i64(a.read_at(o)), n, axis)? * stride;
    }
    Ok(())
}

/// Read an index array's values as `i64`.
///
/// The contiguous case gets a typed loop: index arrays are usually the same
/// length as the result, so this is on the hot path of every fancy index.
fn index_values(a: &NdArray) -> Vec<i64> {
    let n = a.size();
    let mut out = Vec::with_capacity(n);
    if a.flags.c_contiguous && !a.dtype.is_flexible() {
        crate::dispatch_dtype!(a.dtype, T, {
            // SAFETY: `a` is contiguous with `n` elements of `T` at
            // `byte_offset`, all inside the allocation.
            unsafe {
                let p = a.buffer.as_ptr().offset(a.byte_offset) as *const T;
                for i in 0..n {
                    out.push(scalar_to_i64(std::ptr::read_unaligned(p.add(i)).to_scalar()));
                }
            }
        });
        return out;
    }
    for o in crate::iter::offsets(&a.shape, &a.strides, a.byte_offset) {
        out.push(scalar_to_i64(a.read_at(o)));
    }
    out
}

/// Apply an index expression to `arr`.
pub fn index(arr: &NdArray, items: &[IndexItem]) -> Result<Indexed> {
    let plain = !items.iter().any(|i| i.is_advanced());
    let has_nonint = items
        .iter()
        .any(|i| matches!(i, IndexItem::Slice(_) | IndexItem::Ellipsis | IndexItem::NewAxis));
    let expanded = expand(arr, items)?;

    if plain {
        let mut cur = arr.clone();
        let mut ax = 0usize;
        for it in &expanded {
            match it {
                IndexItem::NewAxis => {
                    cur = cur.insert_axis(ax);
                    ax += 1;
                }
                IndexItem::Int(i) => {
                    let n = cur.shape[ax];
                    let j = if *i < 0 { i + n } else { *i };
                    if j < 0 || j >= n {
                        return Err(Error::IndexError(format!(
                            "index {} is out of bounds for axis {} with size {}",
                            i, ax, n
                        )));
                    }
                    cur = cur.slice_axis(ax, j, 1, 1).remove_axis(ax);
                }
                IndexItem::Slice(s) => {
                    let (start, len, step) = resolve_slice(*s, cur.shape[ax])?;
                    cur = cur.slice_axis(ax, start, len, step);
                    ax += 1;
                }
                IndexItem::Ellipsis => unreachable!("expanded"),
                _ => unreachable!("not advanced"),
            }
        }
        let scalarize = cur.ndim() == 0 && !has_nonint;
        return Ok(Indexed::View {
            arr: cur,
            scalarize,
        });
    }

    // ---- advanced indexing -------------------------------------------
    // (array, source stride, axis length, axis number) for each index array;
    // a 0-d boolean has stride 0 and no bounds check (len < 0).
    struct Fancy {
        /// The index array itself, when there is one. Keeping it (instead of
        /// eagerly widening every entry to `i64`) lets the common
        /// no-broadcast case fold the index array straight into `outer`,
        /// which is one fewer full pass over the indices.
        src: Option<NdArray>,
        vals: Vec<i64>,
        shape: Vec<isize>,
        stride: isize,
        len: isize,
        axis: usize,
    }
    let mut fancies: Vec<Fancy> = Vec::new();
    let mut sub_shape: Vec<isize> = Vec::new();
    let mut sub_strides: Vec<isize> = Vec::new();
    let mut sub_base = arr.byte_offset;
    // Position (in `sub_shape`) where the broadcast dims would be spliced.
    let mut insert: Option<usize> = None;
    // Consecutiveness: track whether a separator appeared after the first
    // advanced index but before the last.
    let mut seen_fancy = false;
    let mut separator_after_fancy = false;
    let mut consecutive = true;

    let mut ax = 0usize;
    for it in &expanded {
        match it {
            IndexItem::NewAxis => {
                if seen_fancy {
                    separator_after_fancy = true;
                }
                sub_shape.push(1);
                sub_strides.push(0);
            }
            IndexItem::Slice(s) => {
                if seen_fancy {
                    separator_after_fancy = true;
                }
                let (start, len, step) = resolve_slice(*s, arr.shape[ax])?;
                sub_base += start * arr.strides[ax];
                sub_shape.push(len);
                sub_strides.push(arr.strides[ax] * step);
                ax += 1;
            }
            IndexItem::Int(i) => {
                // Converted to a 0-d index array (numpy's `prepare_index`).
                if separator_after_fancy {
                    consecutive = false;
                }
                if insert.is_none() {
                    insert = Some(sub_shape.len());
                }
                seen_fancy = true;
                fancies.push(Fancy {
                    src: None,
                    vals: vec![*i as i64],
                    shape: vec![],
                    stride: arr.strides[ax],
                    len: arr.shape[ax],
                    axis: ax,
                });
                ax += 1;
            }
            IndexItem::IntArray(a) => {
                if separator_after_fancy {
                    consecutive = false;
                }
                if insert.is_none() {
                    insert = Some(sub_shape.len());
                }
                seen_fancy = true;
                fancies.push(Fancy {
                    src: Some(a.clone()),
                    vals: Vec::new(),
                    shape: a.shape.clone(),
                    stride: arr.strides[ax],
                    len: arr.shape[ax],
                    axis: ax,
                });
                ax += 1;
            }
            IndexItem::BoolArray(b) => {
                if separator_after_fancy {
                    consecutive = false;
                }
                if insert.is_none() {
                    insert = Some(sub_shape.len());
                }
                seen_fancy = true;
                for j in 0..b.ndim() {
                    if b.shape[j] != arr.shape[ax + j] {
                        return Err(Error::IndexError(format!(
                            "boolean index did not match indexed array along axis {}; \
                             size of axis is {} but size of corresponding boolean axis is {}",
                            ax + j,
                            arr.shape[ax + j],
                            b.shape[j]
                        )));
                    }
                }
                for (j, col) in nonzero(b).into_iter().enumerate() {
                    let n = col.size() as isize;
                    let col = col;
                    fancies.push(Fancy {
                        src: Some(col),
                        vals: Vec::new(),
                        shape: vec![n],
                        stride: arr.strides[ax + j],
                        len: arr.shape[ax + j],
                        axis: ax + j,
                    });
                }
                ax += b.ndim();
            }
            IndexItem::ZeroDBool(v) => {
                if separator_after_fancy {
                    consecutive = false;
                }
                if insert.is_none() {
                    insert = Some(sub_shape.len());
                }
                seen_fancy = true;
                let n = if *v { 1 } else { 0 };
                fancies.push(Fancy {
                    src: None,
                    vals: vec![0; n],
                    shape: vec![n as isize],
                    stride: 0,
                    len: -1,
                    axis: usize::MAX,
                });
            }
            IndexItem::Ellipsis => unreachable!("expanded"),
        }
    }

    // Broadcast every index array together.
    let mut b_shape: Vec<isize> = Vec::new();
    for f in &fancies {
        b_shape = crate::iter::broadcast_shapes(&b_shape, &f.shape).map_err(|_| {
            let shapes = fancies
                .iter()
                .map(|g| fmt_shape(&g.shape))
                .collect::<Vec<_>>()
                .join(" ");
            Error::IndexError(format!(
                "shape mismatch: indexing arrays could not be broadcast together \
                 with shapes {} ",
                shapes
            ))
        })?;
    }
    let bsize = shape_size(&b_shape);

    // Accumulate the byte offset contributed by every index array.
    let mut outer = vec![0isize; bsize];
    for f in &fancies {
        // Fast path: this index array already has the broadcast shape, so
        // its values map one-to-one onto `outer`.
        if f.shape == b_shape {
            if f.len < 0 {
                continue;
            }
            match &f.src {
                Some(a) => fold_index_array(&mut outer, a, f.stride, f.len, f.axis)?,
                None => {
                    let n = f.len as i64;
                    for (slot, &raw) in outer.iter_mut().zip(f.vals.iter()) {
                        *slot += resolve_one(raw, n, f.axis)? * f.stride;
                    }
                }
            }
            continue;
        }
        // Broadcast f's values to `b_shape` by index arithmetic.
        let owned_vals;
        let vals: &[i64] = match &f.src {
            Some(a) => {
                owned_vals = index_values(a);
                &owned_vals
            }
            None => &f.vals,
        };
        let pad = b_shape.len() - f.shape.len();
        let mut fstr = vec![0isize; b_shape.len()];
        let mut acc = 1isize;
        for i in (0..f.shape.len()).rev() {
            fstr[i + pad] = if f.shape[i] == 1 { 0 } else { acc };
            acc *= f.shape[i];
        }
        let mut counter = vec![0isize; b_shape.len()];
        let mut src = 0isize;
        for slot in outer.iter_mut() {
            let raw = vals[src as usize];
            if f.len >= 0 {
                let v = if raw < 0 { raw + f.len as i64 } else { raw };
                if v < 0 || v >= f.len as i64 {
                    return Err(Error::IndexError(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        raw, f.axis, f.len
                    )));
                }
                *slot += v as isize * f.stride;
            }
            for k in (0..b_shape.len()).rev() {
                counter[k] += 1;
                src += fstr[k];
                if counter[k] < b_shape[k] {
                    break;
                }
                src -= fstr[k] * counter[k];
                counter[k] = 0;
            }
        }
    }

    let insert = if consecutive { insert.unwrap_or(0) } else { 0 };
    let mut shape: Vec<isize> = Vec::with_capacity(sub_shape.len() + b_shape.len());
    shape.extend_from_slice(&sub_shape[..insert]);
    shape.extend_from_slice(&b_shape);
    shape.extend_from_slice(&sub_shape[insert..]);
    let scalarize = shape.is_empty() && !has_nonint;

    Ok(Indexed::Fancy(FancyPlan {
        shape,
        scalarize,
        outer,
        sub_shape,
        sub_strides,
        sub_base,
        insert,
    }))
}

fn fmt_shape(s: &[isize]) -> String {
    if s.len() == 1 {
        format!("({},)", s[0])
    } else {
        format!(
            "({})",
            s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")
        )
    }
}

/// Gather the elements selected by `plan` into a fresh C-contiguous array.
pub fn gather(arr: &NdArray, plan: &FancyPlan) -> Result<NdArray> {
    let out = NdArray::empty(plan.shape.clone(), arr.dtype)?;
    let isz = arr.itemsize();
    if arr.dtype.is_flexible() || arr.dtype.is_object() {
        plan.for_each(|k, o| out.write_raw_at((k * isz) as isize, arr.raw_bytes_at(o)));
        return Ok(out);
    }
    crate::dispatch_dtype!(arr.dtype, T, {
        let src = arr.buffer.as_ptr();
        // SAFETY: `out` was freshly allocated with `plan.len()` elements of T.
        let dst = unsafe { out.buffer.as_mut_ptr() } as *mut T;
        plan.for_each(|k, o| {
            // SAFETY: every offset the plan yields addresses one in-bounds
            // element of `arr`, and `out` holds `plan.len()` elements of `T`.
            unsafe { *dst.add(k) = std::ptr::read_unaligned(src.offset(o) as *const T) }
        });
    });
    Ok(out)
}

/// Scatter `src` (already broadcast to `plan.shape`) into `arr`.
pub fn scatter(arr: &NdArray, plan: &FancyPlan, src: &NdArray) -> Result<()> {
    if arr.dtype.is_flexible() || arr.dtype.is_object() {
        let src_offs: Vec<isize> =
            crate::iter::offsets(&src.shape, &src.strides, src.byte_offset).collect();
        plan.for_each(|k, d| arr.write_raw_at(d, src.raw_bytes_at(src_offs[k])));
        return Ok(());
    }
    // A broadcast scalar source (the `a[idx] = 1.0` case) needs no source
    // walk at all; otherwise collect the source offsets once.
    if src.size() == 1 || src.strides.iter().all(|&s| s == 0) {
        let v = src.read_at(src.byte_offset);
        crate::dispatch_dtype!(arr.dtype, T, {
            // SAFETY: the plan only yields in-bounds element offsets of `arr`.
            let base = unsafe { arr.buffer.as_mut_ptr() };
            let val = T::from_scalar(v);
            // SAFETY: the plan yields in-bounds element offsets of `arr`.
            plan.for_each(|_, d| unsafe {
                std::ptr::write_unaligned(base.offset(d) as *mut T, val)
            });
        });
        return Ok(());
    }
    let src_offs: Vec<isize> =
        crate::iter::offsets(&src.shape, &src.strides, src.byte_offset).collect();
    crate::dispatch_dtype!(arr.dtype, T, {
        // SAFETY: the plan only yields in-bounds element offsets of `arr`.
        let dbase = unsafe { arr.buffer.as_mut_ptr() };
        let sbase = src.buffer.as_ptr();
        // SAFETY: both offset sets are in-bounds element offsets, and `src`
        // has already been broadcast to the plan's shape.
        plan.for_each(|k, d| unsafe {
            let v = std::ptr::read_unaligned(sbase.offset(src_offs[k]) as *const T);
            std::ptr::write_unaligned(dbase.offset(d) as *mut T, v);
        });
    });
    Ok(())
}

/// `np.take` with numpy's three bounds modes.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TakeMode {
    Raise,
    Wrap,
    Clip,
}

impl TakeMode {
    pub fn from_str(s: &str) -> Option<TakeMode> {
        match s {
            "raise" => Some(TakeMode::Raise),
            "wrap" => Some(TakeMode::Wrap),
            "clip" => Some(TakeMode::Clip),
            _ => None,
        }
    }

    /// Fold `i` into `[0, n)`. `None` means "raise".
    pub fn apply(self, i: i64, n: i64) -> Option<i64> {
        match self {
            TakeMode::Raise => {
                let v = if i < 0 { i + n } else { i };
                if v < 0 || v >= n {
                    None
                } else {
                    Some(v)
                }
            }
            TakeMode::Wrap => {
                if n == 0 {
                    return None;
                }
                Some(i.rem_euclid(n))
            }
            TakeMode::Clip => {
                if n == 0 {
                    return None;
                }
                Some(i.clamp(0, n - 1))
            }
        }
    }
}

/// `a.take(indices, axis=axis, mode=mode)`.
pub fn take(arr: &NdArray, indices: &NdArray, axis: Option<usize>, mode: TakeMode) -> Result<NdArray> {
    let idx = index_values(indices);
    let (src, ax) = match axis {
        None => {
            let flat = if arr.flags.c_contiguous {
                let n = arr.size() as isize;
                arr.reshape(&[n])?
            } else {
                let n = arr.size() as isize;
                arr.copy().reshape(&[n])?
            };
            (flat, 0usize)
        }
        Some(a) => (arr.clone(), a),
    };
    let n = src.shape[ax];
    let mut resolved = Vec::with_capacity(idx.len());
    for &i in &idx {
        match mode.apply(i, n as i64) {
            Some(v) => resolved.push(v as isize),
            None => {
                return Err(Error::IndexError(format!(
                    "index {} is out of bounds for axis {} with size {}",
                    i, ax, n
                )))
            }
        }
    }
    // Result shape: src.shape with axis `ax` replaced by indices.shape.
    let mut shape: Vec<isize> = Vec::new();
    shape.extend_from_slice(&src.shape[..ax]);
    shape.extend_from_slice(&indices.shape);
    shape.extend_from_slice(&src.shape[ax + 1..]);
    let out = NdArray::empty(shape, src.dtype)?;

    let outer: usize = shape_size(&src.shape[..ax]);
    let inner: usize = shape_size(&src.shape[ax + 1..]);
    // Offsets of the (outer, inner) grid in the source.
    let mut pre_shape: Vec<isize> = src.shape[..ax].to_vec();
    let pre_strides: Vec<isize> = src.strides[..ax].to_vec();
    let post_shape: Vec<isize> = src.shape[ax + 1..].to_vec();
    let post_strides: Vec<isize> = src.strides[ax + 1..].to_vec();
    if pre_shape.is_empty() {
        pre_shape = vec![];
    }
    let pre: Vec<isize> = crate::iter::offsets(&pre_shape, &pre_strides, src.byte_offset).collect();
    let post: Vec<isize> = crate::iter::offsets(&post_shape, &post_strides, 0).collect();
    debug_assert_eq!(pre.len(), outer);
    debug_assert_eq!(post.len(), inner);
    let stride = src.strides[ax];
    let isz = src.itemsize();
    // When the trailing axes are contiguous each (outer, index) pair copies
    // one run of `inner` elements, which is a single memcpy.
    let post_contig = post.len() == inner
        && post.windows(2).all(|w| w[1] - w[0] == isz as isize);
    if post_contig && !src.dtype.is_flexible() && !src.dtype.is_object() {
        let run = inner * isz;
        // SAFETY: `pre`/`resolved` address in-bounds runs of `src`, and `out`
        // was allocated with exactly `pre.len() * resolved.len() * inner`
        // elements.
        unsafe {
            let sp = src.buffer.as_ptr();
            let dp = out.buffer.as_mut_ptr();
            let mut k = 0usize;
            for &p in &pre {
                for &r in &resolved {
                    std::ptr::copy_nonoverlapping(
                        sp.offset(p + r * stride),
                        dp.add(k * run),
                        run,
                    );
                    k += 1;
                }
            }
        }
        return Ok(out);
    }
    let mut k = 0isize;
    for &p in &pre {
        for &r in &resolved {
            for &q in &post {
                let s = p + r * stride + q;
                if src.dtype.is_flexible() || src.dtype.is_object() {
                    out.write_raw_at(k * isz as isize, src.raw_bytes_at(s));
                } else {
                    out.write_at(k * isz as isize, src.read_at(s));
                }
                k += 1;
            }
        }
    }
    Ok(out)
}

/// `np.put(a, indices, values, mode)` — flat, in place.
pub fn put(arr: &NdArray, indices: &[i64], values: &NdArray, mode: TakeMode) -> Result<()> {
    let n = arr.size() as i64;
    if values.size() == 0 {
        return Ok(());
    }
    let vals: Vec<isize> =
        crate::iter::offsets(&values.shape, &values.strides, values.byte_offset).collect();
    for (k, &i) in indices.iter().enumerate() {
        let v = mode.apply(i, n).ok_or_else(|| {
            Error::IndexError(format!(
                "index {} is out of bounds for axis 0 with size {}",
                i, n
            ))
        })?;
        let dst = flat_offset(arr, v as usize);
        let s = vals[k % vals.len()];
        if arr.dtype.is_flexible() {
            arr.write_raw_at(dst, values.raw_bytes_at(s));
        } else {
            arr.write_at(dst, values.read_at(s));
        }
    }
    Ok(())
}

/// Byte offset of the `i`-th element in C order (works for any strides).
pub fn flat_offset(arr: &NdArray, mut i: usize) -> isize {
    if arr.flags.c_contiguous {
        return arr.byte_offset + (i * arr.itemsize()) as isize;
    }
    let mut off = arr.byte_offset;
    for ax in (0..arr.ndim()).rev() {
        let d = arr.shape[ax].max(1) as usize;
        off += (i % d) as isize * arr.strides[ax];
        i /= d;
    }
    off
}

/// `np.putmask(a, mask, values)`.
pub fn putmask(arr: &NdArray, mask: &NdArray, values: &NdArray) -> Result<()> {
    if mask.size() != arr.size() {
        return Err(Error::ValueError(format!(
            "putmask: mask and data must be the same size, got {} and {}",
            mask.size(),
            arr.size()
        )));
    }
    if values.size() == 0 {
        return Ok(());
    }
    let mvals: Vec<isize> =
        crate::iter::offsets(&mask.shape, &mask.strides, mask.byte_offset).collect();
    let vals: Vec<isize> =
        crate::iter::offsets(&values.shape, &values.strides, values.byte_offset).collect();
    for i in 0..arr.size() {
        if !truthy(mask.read_at(mvals[i])) {
            continue;
        }
        let dst = flat_offset(arr, i);
        let s = vals[i % vals.len()];
        if arr.dtype.is_flexible() {
            arr.write_raw_at(dst, values.raw_bytes_at(s));
        } else {
            arr.write_at(dst, values.read_at(s));
        }
    }
    Ok(())
}

/// `np.compress(condition, a, axis)`.
pub fn compress(arr: &NdArray, cond: &NdArray, axis: Option<usize>) -> Result<NdArray> {
    let flat: Vec<isize> =
        crate::iter::offsets(&cond.shape, &cond.strides, cond.byte_offset).collect();
    let keep: Vec<i64> = flat
        .iter()
        .enumerate()
        .filter(|(_, &o)| truthy(cond.read_at(o)))
        .map(|(i, _)| i as i64)
        .collect();
    let n = match axis {
        None => arr.size(),
        Some(a) => arr.shape[a].max(0) as usize,
    };
    if flat.len() > n {
        return Err(Error::ValueError(format!(
            "condition contains entries that are out of bounds \
             (condition has size {} but the array axis has size {})",
            flat.len(),
            n
        )));
    }
    let idx = NdArray::from_scalars(
        &keep.iter().map(|&v| Scalar::Int(v)).collect::<Vec<_>>(),
        DType::I64,
    )?;
    take(arr, &idx, axis, TakeMode::Raise)
}

/// `np.choose(a, choices, mode)`.
pub fn choose(sel: &NdArray, choices: &[NdArray], mode: TakeMode) -> Result<NdArray> {
    if choices.is_empty() {
        return Err(Error::ValueError("choose: needs at least one array".into()));
    }
    let mut shape = sel.shape.clone();
    for c in choices {
        shape = crate::iter::broadcast_shapes(&shape, &c.shape)?;
    }
    let mut dt = choices[0].dtype;
    for c in &choices[1..] {
        dt = crate::dtype::promote(dt, c.dtype);
    }
    let bsel = crate::iter::broadcast_to(sel, &shape)?;
    let bch: Vec<NdArray> = choices
        .iter()
        .map(|c| crate::iter::broadcast_to(c, &shape))
        .collect::<Result<_>>()?;
    let out = NdArray::empty(shape.clone(), dt)?;
    let n = choices.len() as i64;
    let sel_offs: Vec<isize> =
        crate::iter::offsets(&bsel.shape, &bsel.strides, bsel.byte_offset).collect();
    let ch_offs: Vec<Vec<isize>> = bch
        .iter()
        .map(|c| crate::iter::offsets(&c.shape, &c.strides, c.byte_offset).collect())
        .collect();
    let isz = out.itemsize() as isize;
    for i in 0..sel_offs.len() {
        let raw = match bsel.read_at(sel_offs[i]) {
            Scalar::Int(v) => v,
            Scalar::Uint(v) => v as i64,
            Scalar::Bool(b) => b as i64,
            Scalar::Float(f) => f as i64,
            Scalar::Complex(c) => c.re as i64,
        };
        let k = mode.apply(raw, n).ok_or_else(|| {
            Error::ValueError("invalid entry in choice array".to_string())
        })? as usize;
        out.write_at(i as isize * isz, bch[k].read_at(ch_offs[k][i]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a234() -> NdArray {
        NdArray::arange(0.0, 24.0, 1.0, DType::I64)
            .unwrap()
            .reshape(&[2, 3, 4])
            .unwrap()
    }

    fn ints(v: &[i64]) -> NdArray {
        NdArray::from_scalars(&v.iter().map(|&x| Scalar::Int(x)).collect::<Vec<_>>(), DType::I64)
            .unwrap()
    }

    fn shape_of(arr: &NdArray, items: &[IndexItem]) -> Vec<isize> {
        match index(arr, items).unwrap() {
            Indexed::View { arr, .. } => arr.shape,
            Indexed::Fancy(p) => p.shape,
        }
    }

    #[test]
    fn slice_resolution_matches_python() {
        assert_eq!(
            resolve_slice(SliceSpec { start: None, stop: None, step: Some(-1) }, 5).unwrap(),
            (4, 5, -1)
        );
        assert_eq!(
            resolve_slice(SliceSpec { start: Some(1), stop: Some(9), step: Some(2) }, 10).unwrap(),
            (1, 4, 2)
        );
        assert_eq!(
            resolve_slice(SliceSpec { start: Some(-2), stop: None, step: None }, 5).unwrap(),
            (3, 2, 1)
        );
        assert!(resolve_slice(SliceSpec { start: None, stop: None, step: Some(0) }, 5).is_err());
    }

    #[test]
    fn basic_indexing_shapes() {
        let a = a234();
        assert_eq!(shape_of(&a, &[IndexItem::Int(0)]), vec![3, 4]);
        assert_eq!(shape_of(&a, &[IndexItem::NewAxis]), vec![1, 2, 3, 4]);
        assert_eq!(
            shape_of(&a, &[IndexItem::Ellipsis, IndexItem::Int(0)]),
            vec![2, 3]
        );
        assert_eq!(
            shape_of(
                &a,
                &[IndexItem::Slice(SliceSpec { start: None, stop: None, step: Some(-1) })]
            ),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn advanced_layout_matches_numpy() {
        // Probed against numpy 2.5.2 (see PLAN.md M2 notes).
        let a = NdArray::arange(0.0, (7 * 3 * 4 * 5) as f64, 1.0, DType::I64)
            .unwrap()
            .reshape(&[7, 3, 4, 5])
            .unwrap();
        let full = IndexItem::Slice(SliceSpec::default());
        // a[:, [0,1], None, [0,1]] -> (2, 7, 1, 5): newaxis separates.
        assert_eq!(
            shape_of(
                &a,
                &[
                    full.clone(),
                    IndexItem::IntArray(ints(&[0, 1])),
                    IndexItem::NewAxis,
                    IndexItem::IntArray(ints(&[0, 1]))
                ]
            ),
            vec![2, 7, 1, 5]
        );
        // a[:, [0,1], [0,1], None] -> (7, 2, 1, 5): adjacent.
        assert_eq!(
            shape_of(
                &a,
                &[
                    full.clone(),
                    IndexItem::IntArray(ints(&[0, 1])),
                    IndexItem::IntArray(ints(&[0, 1])),
                    IndexItem::NewAxis
                ]
            ),
            vec![7, 2, 1, 5]
        );
        // a[:, [0,1], :, [0,1]] -> (2, 7, 4)
        assert_eq!(
            shape_of(
                &a,
                &[
                    full.clone(),
                    IndexItem::IntArray(ints(&[0, 1])),
                    full.clone(),
                    IndexItem::IntArray(ints(&[0, 1]))
                ]
            ),
            vec![2, 7, 4]
        );
        // a[0, :, [0,1]] on (2,3,4) -> (2, 3): the int becomes a 0-d index.
        let b = a234();
        assert_eq!(
            shape_of(
                &b,
                &[IndexItem::Int(0), full.clone(), IndexItem::IntArray(ints(&[0, 1]))]
            ),
            vec![2, 3]
        );
        // a[True] -> (1, 2, 3, 4)
        assert_eq!(
            shape_of(&b, &[IndexItem::ZeroDBool(true)]),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            shape_of(&b, &[IndexItem::ZeroDBool(false)]),
            vec![0, 2, 3, 4]
        );
    }

    #[test]
    fn gather_values_are_right() {
        let a = a234();
        let full = IndexItem::Slice(SliceSpec::default());
        let plan = match index(&a, &[IndexItem::IntArray(ints(&[1, 0])), full]).unwrap() {
            Indexed::Fancy(p) => p,
            _ => panic!("expected fancy"),
        };
        let out = gather(&a, &plan).unwrap();
        assert_eq!(out.shape, vec![2, 3, 4]);
        assert_eq!(out.get_flat(0), Scalar::Int(12));
        assert_eq!(out.get_flat(12), Scalar::Int(0));
    }

    #[test]
    fn out_of_bounds_reports_the_original_value() {
        let a = a234();
        let e = match index(&a, &[IndexItem::IntArray(ints(&[5]))]) {
            Err(e) => e,
            Ok(_) => panic!("expected an IndexError"),
        };
        match e {
            Error::IndexError(m) => {
                assert_eq!(m, "index 5 is out of bounds for axis 0 with size 2")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn boolean_mask_selects_in_c_order() {
        let a = a234();
        let mut mask = NdArray::zeros(vec![2, 3], DType::Bool).unwrap();
        mask.set(&[0, 0], Scalar::Bool(true)).unwrap();
        mask.set(&[1, 2], Scalar::Bool(true)).unwrap();
        let plan = match index(&a, &[IndexItem::BoolArray(mask)]).unwrap() {
            Indexed::Fancy(p) => p,
            _ => panic!(),
        };
        assert_eq!(plan.shape, vec![2, 4]);
        let out = gather(&a, &plan).unwrap();
        assert_eq!(out.get_flat(0), Scalar::Int(0));
        assert_eq!(out.get_flat(4), Scalar::Int(20));
    }

    #[test]
    fn take_modes() {
        let a = NdArray::arange(0.0, 5.0, 1.0, DType::I64).unwrap();
        let got = take(&a, &ints(&[-1, 7]), None, TakeMode::Wrap).unwrap();
        assert_eq!(got.to_vec(), vec![Scalar::Int(4), Scalar::Int(2)]);
        let got = take(&a, &ints(&[-9, 7]), None, TakeMode::Clip).unwrap();
        assert_eq!(got.to_vec(), vec![Scalar::Int(0), Scalar::Int(4)]);
        assert!(take(&a, &ints(&[7]), None, TakeMode::Raise).is_err());
    }
}
