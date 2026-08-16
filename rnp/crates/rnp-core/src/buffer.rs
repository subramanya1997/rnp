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

pub struct Buffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
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
        Buffer { ptr, len, layout }
    }

    /// Allocate `len` uninitialised bytes. Callers must write every byte that
    /// will later be read.
    pub fn uninitialized(len: usize) -> Buffer {
        let layout = Self::layout_for(len);
        // SAFETY: `layout` has non-zero size (>= ALLOC_ALIGN) and valid align.
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Buffer { ptr, len, layout }
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
        // SAFETY: `ptr` came from `alloc`/`alloc_zeroed` with exactly `layout`.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
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
}
