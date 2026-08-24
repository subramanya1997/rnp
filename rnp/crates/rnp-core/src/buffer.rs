//! Reference-counted, aligned byte buffer backing every array.
//!
//! Views share a `Arc<Buffer>`; mutation happens through raw pointers because
//! several `NdArray` headers may alias the same bytes (exactly like numpy).
//! Rust's aliasing rules are therefore upheld by convention (the `WRITEABLE`
//! flag) rather than by the borrow checker, so every access goes through the
//! raw-pointer API below.

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;

/// Allocation alignment. 64 bytes keeps every element type naturally aligned
/// and leaves room for future SIMD inner loops.
pub const ALLOC_ALIGN: usize = 64;

/// Keep-alive token for memory this process did not allocate.
///
/// `rnp-core` has no idea what a Python object is, so a `Buffer` that wraps
/// foreign memory carries an opaque owner instead. Dropping the owner is what
/// releases the exporter's claim (for the PyO3 binding that is a `Py<PyAny>`
/// being decref'd, plus whatever `Py_buffer` bookkeeping goes with it).
///
/// Implementors must guarantee that the address range handed to
/// [`Buffer::from_foreign`] stays valid, and stays at that address, for as
/// long as the owner value is alive.
pub trait ForeignOwner: Send + Sync {}

enum Backing {
    /// Allocated by us with this layout; freed on drop.
    Owned(Layout),
    /// Someone else's memory; `_owner` keeps it alive and drops last.
    Foreign(Box<dyn ForeignOwner>),
}

pub struct Buffer {
    ptr: NonNull<u8>,
    len: usize,
    backing: Backing,
}

// SAFETY: `Buffer` owns a unique heap allocation and exposes no interior
// references. Cross-thread sharing is sound; unsynchronised *mutation* is the
// caller's responsibility, as it is in numpy itself.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn layout_for(len: usize) -> Layout {
        // A zero-sized allocation is not permitted, so round up to the align.
        let size = len.max(ALLOC_ALIGN);
        Layout::from_size_align(size, ALLOC_ALIGN).expect("invalid buffer layout")
    }

    /// Allocate `len` zeroed bytes.
    pub fn zeroed(len: usize) -> Buffer {
        let layout = Self::layout_for(len);
        // SAFETY: `layout` has non-zero size (>= ALLOC_ALIGN) and valid align.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Buffer { ptr, len, backing: Backing::Owned(layout) }
    }

    /// Allocate `len` uninitialised bytes. Callers must write every byte that
    /// will later be read.
    pub fn uninitialized(len: usize) -> Buffer {
        let layout = Self::layout_for(len);
        // SAFETY: `layout` has non-zero size (>= ALLOC_ALIGN) and valid align.
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Buffer { ptr, len, backing: Backing::Owned(layout) }
    }

    /// Allocate and copy `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Buffer {
        let buf = Buffer::uninitialized(bytes.len());
        // SAFETY: `buf` was just allocated with at least `bytes.len()` bytes,
        // and the two regions cannot overlap (the allocation is fresh).
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.ptr.as_ptr(), bytes.len());
        }
        buf
    }

    /// Wrap `len` bytes starting at `ptr` that some other runtime owns.
    ///
    /// # Safety
    /// The caller must guarantee that, for as long as `owner` is alive:
    /// * `ptr` is non-null and `[ptr, ptr+len)` is a single valid, readable
    ///   allocation, and
    /// * the allocation is never moved, freed, or shrunk.
    ///
    /// Dropping the returned `Buffer` drops `owner` and nothing else — the
    /// memory is emphatically *not* freed here.
    pub unsafe fn from_foreign(
        ptr: *mut u8,
        len: usize,
        owner: Box<dyn ForeignOwner>,
    ) -> Buffer {
        let ptr = NonNull::new(ptr).expect("foreign buffer pointer must be non-null");
        Buffer { ptr, len, backing: Backing::Foreign(owner) }
    }

    /// True when the bytes belong to a foreign exporter (numpy's `OWNDATA`
    /// is the negation of this for a base array).
    pub fn is_foreign(&self) -> bool {
        matches!(self.backing, Backing::Foreign(_))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Raw mutable pointer to the allocation.
    ///
    /// # Safety
    /// The caller must ensure no conflicting reads/writes happen concurrently
    /// and that writes stay within `len()` bytes.
    pub unsafe fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` initialised-or-not bytes owned by
        // self; `u8` has no invalid bit patterns.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        match &self.backing {
            // SAFETY: `ptr` came from `alloc`/`alloc_zeroed` with exactly this
            // layout, and this is the sole owner (an `Arc<Buffer>` reaching
            // refcount zero), so nothing can observe the freed memory.
            Backing::Owned(layout) => unsafe { dealloc(self.ptr.as_ptr(), *layout) },
            // The exporter owns the bytes; dropping `self.backing` drops the
            // keep-alive token, which is the only release we are allowed to do.
            Backing::Foreign(_) => {}
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer").field("len", &self.len).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_is_zero_and_aligned() {
        let b = Buffer::zeroed(100);
        assert_eq!(b.len(), 100);
        assert!(b.as_slice().iter().all(|&x| x == 0));
        assert_eq!(b.as_ptr() as usize % ALLOC_ALIGN, 0);
    }

    #[test]
    fn empty_allocation_is_valid() {
        let b = Buffer::zeroed(0);
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
    }

    #[test]
    fn from_bytes_round_trips() {
        let b = Buffer::from_bytes(&[1, 2, 3, 4]);
        assert_eq!(b.as_slice(), &[1, 2, 3, 4]);
    }

    // ---- foreign (adopted) buffers -------------------------------------

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Stands in for the Python object that guarantees adopted memory: it owns
    /// the allocation and records that it was released exactly once.
    struct TestOwner {
        data: Vec<u8>,
        released: Arc<AtomicUsize>,
    }

    impl ForeignOwner for TestOwner {}

    impl Drop for TestOwner {
        fn drop(&mut self) {
            self.released.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn adopt(released: &Arc<AtomicUsize>, bytes: &[u8]) -> Buffer {
        let mut owner = Box::new(TestOwner {
            data: bytes.to_vec(),
            released: Arc::clone(released),
        });
        let ptr = owner.data.as_mut_ptr();
        let len = owner.data.len();
        // SAFETY: `owner` owns the `Vec` the pointer came from and is moved
        // into the `Buffer`, so the allocation outlives every use of `ptr`;
        // a boxed `Vec`'s heap buffer does not move when the `Box` moves.
        unsafe { Buffer::from_foreign(ptr, len, owner) }
    }

    #[test]
    fn foreign_buffer_reads_the_owners_bytes_and_never_frees_them() {
        let released = Arc::new(AtomicUsize::new(0));
        let buf = adopt(&released, &[9, 8, 7, 6]);
        assert!(buf.is_foreign());
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.as_slice(), &[9, 8, 7, 6]);
        assert_eq!(released.load(Ordering::SeqCst), 0);
        drop(buf);
        // Dropping the buffer released the owner exactly once — and the owner,
        // not the buffer, is what freed the bytes.
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn every_view_keeps_the_owner_alive() {
        let released = Arc::new(AtomicUsize::new(0));
        let shared = Arc::new(adopt(&released, &[1, 2, 3, 4, 5, 6, 7, 8]));
        // Two `NdArray` headers sharing the allocation, as views do.
        let view_a = Arc::clone(&shared);
        let view_b = Arc::clone(&shared);
        drop(shared);
        assert_eq!(released.load(Ordering::SeqCst), 0);
        drop(view_a);
        assert_eq!(released.load(Ordering::SeqCst), 0, "one view still lives");
        assert_eq!(view_b.as_slice()[7], 8, "bytes are still readable");
        drop(view_b);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn writes_through_a_foreign_buffer_reach_the_owner() {
        let released = Arc::new(AtomicUsize::new(0));
        let buf = adopt(&released, &[0; 4]);
        // SAFETY: sole owner, write is within `len()`.
        unsafe { *buf.as_mut_ptr().add(2) = 0x5a };
        assert_eq!(buf.as_slice(), &[0, 0, 0x5a, 0]);
    }

    #[test]
    fn a_zero_length_foreign_buffer_is_legal() {
        let released = Arc::new(AtomicUsize::new(0));
        let buf = adopt(&released, &[]);
        assert!(buf.is_empty());
        assert_eq!(buf.as_slice(), &[] as &[u8]);
        drop(buf);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn owned_buffers_are_not_foreign() {
        assert!(!Buffer::zeroed(8).is_foreign());
        assert!(!Buffer::uninitialized(8).is_foreign());
        assert!(!Buffer::from_bytes(&[1]).is_foreign());
    }
}
