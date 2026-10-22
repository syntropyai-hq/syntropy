// SPDX-License-Identifier: MIT OR Apache-2.0

//! The one place `unsafe` is allowed in this crate.

use std::alloc::{GlobalAlloc, Layout, System};

pub struct CountingAlloc;

// SAFETY: every method delegates to `System`.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}
