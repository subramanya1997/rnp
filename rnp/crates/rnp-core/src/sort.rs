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
/// code units for the flexible ones, and -- for a *structured* dtype -- the
/// keys of its fields in declaration order.
enum Key {
    Num(Scalar),
    Text(Vec<u32>),
    /// datetime64 / timedelta64: NaT sorts last, like NaN.
    Time(i64),
    /// Raw bytes compared with `memcmp`, which is what numpy's
    /// `STRING_compare` does for a subarray or opaque-void field.
    Raw(Vec<u8>),
    /// A structured element: its fields' keys, compared in order.
    Fields(Vec<Key>),
}

/// How to read one leaf of a (possibly nested) structured element.
enum Leaf {
    /// Read a scalar through this view's descriptor (which carries the
    /// field's byte order, so a `'>i4'` field still compares numerically).
    Num(NdArray),
    /// datetime64 / timedelta64 leaf.
    Time(NdArray),
    /// An `'S'`/`'V'`/subarray leaf: `memcmp` over the field's whole width.
    /// The view is relabelled `'V<width>'` so that `raw_bytes_at` hands back
    /// exactly those bytes.
    Raw(NdArray),
    /// A `'U'` leaf: UCS-4 code units, not bytes -- numpy's
    /// `UNICODE_compare` orders by code point, which a little-endian
    /// `memcmp` would get wrong.
    Text(NdArray),
}

/// The recipe for one array's sort keys, built once per sort.
///
/// For everything but a structured dtype this is a single leaf covering the
/// whole element; for a structured dtype it is the depth-first list of leaves
/// with their byte offsets, which is exactly the order numpy's `VOID_compare`
/// walks (field by field, recursing into nested structs) -- see
/// `VOID_compare` in `upstream/numpy/_core/src/multiarray/arraytypes.c.src`.
struct KeyPlan {
    leaves: Vec<(isize, Leaf)>,
    /// True when the plan is a single whole-element leaf, so `key_at` can
    /// skip building a `Fields` vector.
    plain: bool,
}

/// The leaf for a non-structured descriptor, as seen through `arr`.
fn leaf_of(arr: &NdArray, descr: crate::descr::Descr) -> Leaf {
    use crate::dtype::DType;
    // A subarray field, an opaque void and a bytes field are all raw memcmp:
    // numpy routes each of them to `STRING_compare`.
    match descr.dt {
        DType::SubArray(_) | DType::Void(_) | DType::Bytes(_) => {
            let raw = crate::descr::Descr::new(
                DType::Void(descr.itemsize() as u32),
                crate::descr::ByteOrder::NotApplicable,
            );
            Leaf::Raw(arr.field_view(raw, 0))
        }
        DType::Str(_) => Leaf::Text(arr.field_view(descr, 0)),
        d if d.is_datetime_like() => Leaf::Time(arr.field_view(descr, 0)),
        _ => Leaf::Num(arr.field_view(descr, 0)),
    }
}

/// Append the leaves of `descr` (a field's descriptor) at `base`.
fn push_leaves(
    arr: &NdArray,
    descr: crate::descr::Descr,
    base: usize,
    out: &mut Vec<(isize, Leaf)>,
) {
    match descr.struct_def() {
        Some(def) => {
            for f in def.fields.iter() {
                push_leaves(arr, f.descr, base + f.offset, out);
            }
        }
        None => out.push((base as isize, leaf_of(arr, descr))),
    }
}

fn key_plan(arr: &NdArray) -> KeyPlan {
    if arr.descr.is_struct() {
        let mut leaves = Vec::new();
        push_leaves(arr, arr.descr, 0, &mut leaves);
        return KeyPlan { leaves, plain: false };
    }
    KeyPlan {
        leaves: vec![(0, leaf_of(arr, arr.descr))],
        plain: true,
    }
}

fn leaf_key(leaf: &Leaf, off: isize) -> Key {
    match leaf {
        Leaf::Num(v) => Key::Num(v.read_at(off)),
        Leaf::Time(v) => Key::Time(match v.read_at(off) {
            Scalar::Int(i) => i,
            s => s.as_f64() as i64,
        }),
        Leaf::Raw(v) => Key::Raw(v.raw_bytes_at(off).to_vec()),
        Leaf::Text(v) => Key::Text(crate::ops::logical_bytes(v, off)),
    }
}

fn key_at(plan: &KeyPlan, off: isize) -> Key {
    if plan.plain {
        let (delta, leaf) = &plan.leaves[0];
        return leaf_key(leaf, off + delta);
    }
    Key::Fields(
        plan.leaves
            .iter()
            .map(|(delta, leaf)| leaf_key(leaf, off + delta))
            .collect(),
    )
}

/// numpy's datetime order: NaT is greater than everything and ties with
/// itself, exactly as NaN does for floats (probed: `np.sort` puts NaT last).
fn cmp_time(a: i64, b: i64) -> Ordering {
    match (a == crate::datetime::NAT, b == crate::datetime::NAT) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.cmp(&b),
    }
}

fn cmp_key(a: &Key, b: &Key) -> Ordering {
    match (a, b) {
        (Key::Num(x), Key::Num(y)) => cmp_scalar(*x, *y),
        (Key::Time(x), Key::Time(y)) => cmp_time(*x, *y),
        (Key::Text(x), Key::Text(y)) => x.cmp(y),
        (Key::Raw(x), Key::Raw(y)) => x.cmp(y),
        // A structured element compares on its first field, then on the
        // second when those tie, and so on -- numpy's `VOID_compare`.
        (Key::Fields(x), Key::Fields(y)) => {
            for (p, q) in x.iter().zip(y.iter()) {
                let c = cmp_key(p, q);
                if c != Ordering::Equal {
                    return c;
                }
            }
            Ordering::Equal
        }
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
    let plan = key_plan(arr);
    // One scratch buffer for the whole sort, reused per lane: a `Vec` per
    // element turns a 1e6-element sort into a million allocations.
    let mut scratch: Vec<u8> = Vec::new();
    for lane in lanes(arr, axis) {
        let n = lane.len();
        let mut order: Vec<usize> = (0..n).collect();
        let keys: Vec<Key> = lane.iter().map(|&o| key_at(&plan, o)).collect();
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
    let plan = key_plan(arr);
    for (lane, &base) in lanes(arr, axis).iter().zip(out_bases.iter()) {
        let keys: Vec<Key> = lane.iter().map(|&o| key_at(&plan, o)).collect();
        let mut order: Vec<usize> = (0..lane.len()).collect();
        order.sort_by(|&i, &j| cmp_key(&keys[i], &keys[j]));
        for k in 0..n as usize {
            out.write_at(base + k as isize * out_step, Scalar::Int(order[k] as i64));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// introselect — `partition` / `argpartition`
// ---------------------------------------------------------------------------

/// Below this length quickselect stops paying for itself and a straight sort
/// of the window is both faster and simpler.
const SELECT_INSERTION_CUTOFF: usize = 24;

/// Three-way (Dutch-flag) partition of `order[lo..hi]` around the element
/// currently at `pivot_pos`. Returns `(lt, gt)` with `[lo, lt)` strictly less
/// than the pivot, `[lt, gt)` equal to it, and `[gt, hi)` strictly greater.
///
/// Three-way matters here: numpy's `partition` is routinely called on arrays
/// with heavy duplication, where a two-way split degrades to quadratic.
fn partition3(order: &mut [usize], keys: &[Key], lo: usize, hi: usize, pivot_pos: usize) -> (usize, usize) {
    order.swap(lo, pivot_pos);
    // The pivot is identified by its *key index*, which is stable under the
    // swaps below, unlike its position.
    let pivot = order[lo];
    let (mut lt, mut i, mut gt) = (lo, lo + 1, hi);
    while i < gt {
        match cmp_key(&keys[order[i]], &keys[pivot]) {
            Ordering::Less => {
                order.swap(lt, i);
                lt += 1;
                i += 1;
            }
            Ordering::Greater => {
                gt -= 1;
                order.swap(i, gt);
            }
            Ordering::Equal => i += 1,
        }
    }
    (lt, gt)
}

fn median_of_3(order: &[usize], keys: &[Key], lo: usize, hi: usize) -> usize {
    let mid = lo + (hi - lo) / 2;
    let last = hi - 1;
    let (a, b, c) = (&keys[order[lo]], &keys[order[mid]], &keys[order[last]]);
    if cmp_key(a, b) == Ordering::Less {
        if cmp_key(b, c) == Ordering::Less {
            mid
        } else if cmp_key(a, c) == Ordering::Less {
            last
        } else {
            lo
        }
    } else if cmp_key(a, c) == Ordering::Less {
        lo
    } else if cmp_key(b, c) == Ordering::Less {
        last
    } else {
        mid
    }
}

/// Blum-Floyd-Pratt-Rivest-Tarjan median of medians: the guaranteed-linear
/// pivot introselect falls back on when the cheap median-of-3 has produced
/// too many lopsided splits. Leaves `order[lo..hi]` permuted and returns the
/// position of a pivot with a constant-fraction rank guarantee.
fn median_of_medians(order: &mut [usize], keys: &[Key], lo: usize, hi: usize) -> usize {
    let n = hi - lo;
    let ngroups = n.div_ceil(5);
    for g in 0..ngroups {
        let s = lo + g * 5;
        let e = (s + 5).min(hi);
        order[s..e].sort_by(|&i, &j| cmp_key(&keys[i], &keys[j]));
        // Collect each group's median into `order[lo..lo + ngroups]`. Later
        // groups start at `lo + 5g >= lo + g`, so this only ever overwrites
        // slots whose median has already been harvested.
        let med = s + (e - s) / 2;
        order.swap(lo + g, med);
    }
    let mid = lo + ngroups / 2;
    select_nth(order, keys, lo, lo + ngroups, mid);
    mid
}

/// Introselect: place the element of rank `k` (relative to the whole slice)
/// at `order[k]`, with everything in `[lo, k)` `<=` it and everything in
/// `(k, hi)` `>=` it. Quickselect with a median-of-3 pivot, falling back to
/// median-of-medians once the recursion depth exceeds `2*log2(n)` — which is
/// exactly what numpy's `introselect` does.
fn select_nth(order: &mut [usize], keys: &[Key], mut lo: usize, mut hi: usize, k: usize) {
    debug_assert!(k >= lo && k < hi);
    let mut budget = 2 * (usize::BITS - (hi - lo).max(1).leading_zeros()) as usize;
    loop {
        if hi <= lo + 1 {
            return;
        }
        if hi - lo <= SELECT_INSERTION_CUTOFF {
            order[lo..hi].sort_by(|&i, &j| cmp_key(&keys[i], &keys[j]));
            return;
        }
        let pivot_pos = if budget == 0 {
            median_of_medians(order, keys, lo, hi)
        } else {
            budget -= 1;
            median_of_3(order, keys, lo, hi)
        };
        let (lt, gt) = partition3(order, keys, lo, hi, pivot_pos);
        if k < lt {
            hi = lt;
        } else if k < gt {
            // `k` landed in the run of pivot-equal elements: done.
            return;
        } else {
            lo = gt;
        }
    }
}

/// Run introselect for every requested rank on one lane's index permutation.
///
/// The ranks are handled in increasing order, each one selected only within
/// the suffix left of the previous one: after rank `k` is placed, everything
/// at `[.., k]` is `<=` everything after it, so the next rank is still the
/// correct *global* rank when sought in `[k+1, n)`. That is how numpy
/// exploits the already-partitioned regions between multiple `kth`.
fn select_all(order: &mut [usize], keys: &[Key], kths: &[usize]) {
    let n = order.len();
    let mut lo = 0usize;
    for &k in kths {
        if k >= n {
            continue;
        }
        select_nth(order, keys, lo, n, k);
        lo = k + 1;
    }
}

/// The `kth` list, sorted and deduplicated — the form `select_all` wants.
fn normalize_kths(kth: &[usize]) -> Vec<usize> {
    let mut ks = kth.to_vec();
    ks.sort_unstable();
    ks.dedup();
    ks
}

/// `a.partition(kth, axis)`, in place.
pub fn partition_inplace(arr: &mut NdArray, kth: &[usize], axis: usize) -> Result<()> {
    check_axis(arr, axis)?;
    if arr.dtype().is_object() {
        return Err(Error::NotImplemented("partition on object arrays".into()));
    }
    if !arr.flags.writeable {
        return Err(Error::ValueError(
            "assignment destination is read-only".into(),
        ));
    }
    let ks = normalize_kths(kth);
    let isz = arr.itemsize();
    let plan = key_plan(arr);
    let mut scratch: Vec<u8> = Vec::new();
    for lane in lanes(arr, axis) {
        let n = lane.len();
        if n <= 1 || ks.is_empty() {
            continue;
        }
        let keys: Vec<Key> = lane.iter().map(|&o| key_at(&plan, o)).collect();
        let mut order: Vec<usize> = (0..n).collect();
        select_all(&mut order, &keys, &ks);
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

/// `np.argpartition(a, kth, axis)` — the index permutation that
/// `partition` would apply.
pub fn argpartition(arr: &NdArray, kth: &[usize], axis: usize) -> Result<NdArray> {
    check_axis(arr, axis)?;
    if arr.dtype().is_object() {
        return Err(Error::NotImplemented("argpartition on object arrays".into()));
    }
    let ks = normalize_kths(kth);
    let mut out = NdArray::zeros(arr.shape.clone(), crate::dtype::DType::I64)?;
    let n = arr.shape[axis].max(0);
    let mut shape = arr.shape.clone();
    let mut strides = out.strides.clone();
    shape.remove(axis);
    let out_step = strides[axis];
    strides.remove(axis);
    let out_bases: Vec<isize> = crate::iter::offsets(&shape, &strides, out.byte_offset).collect();
    let plan = key_plan(arr);
    for (lane, &base) in lanes(arr, axis).iter().zip(out_bases.iter()) {
        let keys: Vec<Key> = lane.iter().map(|&o| key_at(&plan, o)).collect();
        let mut order: Vec<usize> = (0..lane.len()).collect();
        select_all(&mut order, &keys, &ks);
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
    let (aplan, vplan) = (key_plan(a), key_plan(v));
    let keys: Vec<Key> = (0..n)
        .map(|i| key_at(&aplan, a.byte_offset + i as isize * a.strides[0]))
        .collect();
    let out = NdArray::zeros(v.shape.clone(), crate::dtype::DType::I64)?;
    let mut k = 0usize;
    for off in crate::iter::offsets(&v.shape, &v.strides, v.byte_offset) {
        let key = key_at(&vplan, off);
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

    /// A tiny xorshift so the randomised checks below are deterministic and
    /// dependency-free.
    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn i64_array(v: &[i64]) -> NdArray {
        let s: Vec<Scalar> = v.iter().map(|&x| Scalar::Int(x)).collect();
        NdArray::from_scalars(&s, DType::I64).unwrap()
    }

    fn f64_array(v: &[f64]) -> NdArray {
        let s: Vec<Scalar> = v.iter().map(|&x| Scalar::Float(x)).collect();
        NdArray::from_scalars(&s, DType::F64).unwrap()
    }

    fn to_i64(a: &NdArray) -> Vec<i64> {
        (0..a.size()).map(|i| a.get_flat(i).as_f64() as i64).collect()
    }

    /// The only postcondition `partition` promises: `out[k]` is what a sorted
    /// array would hold at `k`, everything before is `<=` and everything
    /// after is `>=`. Order *within* each side is unspecified.
    fn assert_partitioned(orig: &[i64], out: &[i64], kths: &[usize]) {
        let mut sorted = orig.to_vec();
        sorted.sort_unstable();
        let mut check = out.to_vec();
        check.sort_unstable();
        assert_eq!(check, sorted, "partition must be a permutation of the input");
        for &k in kths {
            assert_eq!(out[k], sorted[k], "rank {k} misplaced in {out:?}");
            assert!(out[..k].iter().all(|&x| x <= out[k]), "left of {k} in {out:?}");
            assert!(out[k + 1..].iter().all(|&x| x >= out[k]), "right of {k} in {out:?}");
        }
    }

    #[test]
    fn partition_invariant_random() {
        let mut state = 0x2545F491_4F6CDD1Du64;
        for n in [2usize, 3, 5, 17, 24, 25, 64, 257, 1000] {
            for trial in 0..12 {
                // Trial 0 uses a tiny value range on purpose: heavy
                // duplication is the case a two-way split gets wrong.
                let modulus = if trial % 3 == 0 { 3 } else { 1000 };
                let vals: Vec<i64> =
                    (0..n).map(|_| (rng(&mut state) % modulus) as i64).collect();
                for &k in &[0usize, n / 2, n - 1] {
                    let mut a = i64_array(&vals);
                    partition_inplace(&mut a, &[k], 0).unwrap();
                    assert_partitioned(&vals, &to_i64(&a), &[k]);
                }
            }
        }
    }

    #[test]
    fn partition_multiple_kth() {
        let mut state = 0x9E3779B9_7F4A7C15u64;
        let n = 300usize;
        let vals: Vec<i64> = (0..n).map(|_| (rng(&mut state) % 500) as i64).collect();
        let kths = [0usize, 1, 7, 150, 151, 298, 299];
        let mut a = i64_array(&vals);
        partition_inplace(&mut a, &kths, 0).unwrap();
        assert_partitioned(&vals, &to_i64(&a), &kths);

        // Unsorted and duplicated `kth` must behave identically.
        let mut b = i64_array(&vals);
        partition_inplace(&mut b, &[299, 7, 150, 7, 0, 151, 1, 298], 0).unwrap();
        assert_partitioned(&vals, &to_i64(&b), &kths);
    }

    #[test]
    fn argpartition_invariant() {
        let mut state = 0xDEADBEEF_CAFEF00Du64;
        let n = 200usize;
        let vals: Vec<i64> = (0..n).map(|_| (rng(&mut state) % 50) as i64).collect();
        let a = i64_array(&vals);
        let kths = [0usize, 99, 199];
        let idx = argpartition(&a, &kths, 0).unwrap();
        let perm = to_i64(&idx);
        let mut seen = perm.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..n as i64).collect::<Vec<_>>(), "not a permutation");
        let out: Vec<i64> = perm.iter().map(|&i| vals[i as usize]).collect();
        assert_partitioned(&vals, &out, &kths);
    }

    #[test]
    fn partition_nan_goes_last() {
        // NaN is greater than everything under numpy's total order, so
        // selecting the last rank must surface a NaN.
        let vals = [3.0, f64::NAN, 1.0, f64::NAN, 2.0];
        let mut a = f64_array(&vals);
        partition_inplace(&mut a, &[2], 0).unwrap();
        let got: Vec<f64> = (0..5).map(|i| a.get_flat(i).as_f64()).collect();
        assert_eq!(&got[..3], &[1.0, 2.0, 3.0]);
        assert!(got[3].is_nan() && got[4].is_nan());

        let mut b = f64_array(&vals);
        partition_inplace(&mut b, &[4], 0).unwrap();
        assert!(b.get_flat(4).as_f64().is_nan());
    }

    #[test]
    fn partition_dtypes_and_degenerate() {
        // Empty and single-element lanes are no-ops, not errors.
        for n in [0usize, 1] {
            let vals: Vec<i64> = (0..n as i64).collect();
            let mut a = i64_array(&vals);
            partition_inplace(&mut a, &[0], 0).unwrap();
            assert_eq!(to_i64(&a), vals);
        }
        // Unsigned, bool and complex all go through the same comparator.
        let mut u = NdArray::from_scalars(
            &[Scalar::Uint(9), Scalar::Uint(2), Scalar::Uint(5)],
            DType::U64,
        )
        .unwrap();
        partition_inplace(&mut u, &[1], 0).unwrap();
        assert_eq!(u.get_flat(1).as_f64(), 5.0);

        let mut b = NdArray::from_scalars(
            &[Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)],
            DType::Bool,
        )
        .unwrap();
        partition_inplace(&mut b, &[0], 0).unwrap();
        assert_eq!(b.get_flat(0).as_f64(), 0.0);
    }

    #[test]
    fn partition_matches_sort_at_every_rank() {
        // Selecting every rank in turn must reproduce a full sort.
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let n = 120usize;
        let vals: Vec<i64> = (0..n).map(|_| (rng(&mut state) % 40) as i64).collect();
        let all: Vec<usize> = (0..n).collect();
        let mut a = i64_array(&vals);
        partition_inplace(&mut a, &all, 0).unwrap();
        let mut sorted = vals.clone();
        sorted.sort_unstable();
        assert_eq!(to_i64(&a), sorted);
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
