// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hot-path guards (RFC-0060 §3.5): instrumentation that turns ADR-0012 §6's invariants into
//! assertions. A steady-state decode step must make no heap allocation, and [`CountingAlloc`]
//! counts the allocations each thread makes so a test can assert that. It also tracks each
//! thread's live and peak heap bytes ([`peak_bytes_this_thread`]), so a parser test can bound
//! its memory against its input (RFC-0060's bound for parsers on hostile input).
//!
//! A test binary installs the allocator for itself:
//!
//! ```
//! use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};
//!
//! #[global_allocator]
//! static ALLOCATOR: CountingAlloc = CountingAlloc;
//!
//! fn main() {
//!     let before = alloc_count_this_thread();
//!     let boxed = Box::new(7_u64);
//!     assert_eq!(alloc_count_this_thread() - before, 1);
//!     drop(boxed);
//! }
//! ```
//!
//! Without that line in the binary, nothing is counted and [`alloc_count_this_thread`] stays at
//! zero. The lock and driver-call counters and `assert_steady_state` of RFC-0060 §3.5 are planned
//! (QUAL-E3-T11).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// Counts heap allocations on the current thread, then hands every call to the system allocator.
///
/// `alloc`, `alloc_zeroed` and `realloc` each count once, because each can take memory from the
/// heap; `dealloc` does not count. Every call also adjusts the thread's live bytes by the size
/// it takes or gives back. The count is per thread, so a test reads only what its own
/// thread allocated, whatever other threads do.
pub struct CountingAlloc;

thread_local! {
    /// This thread's allocation count. The `const` initialiser and a type with no destructor keep
    /// the access from allocating or registering a destructor, so counting inside the allocator
    /// cannot call back into it.
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

thread_local! {
    /// The bytes this thread has allocated and not yet freed, and the most that has been at
    /// once since the last reset. Memory freed on another thread than the one that allocated it
    /// is subtracted there, so a thread's figure can go below zero.
    static LIVE_BYTES: Cell<i64> = const { Cell::new(0) };
    static PEAK_BYTES: Cell<i64> = const { Cell::new(0) };
}

/// The number of heap allocations the current thread has made since it started, while
/// [`CountingAlloc`] was the binary's global allocator.
pub fn alloc_count_this_thread() -> u64 {
    ALLOCATIONS.try_with(Cell::get).unwrap_or(0)
}

/// The bytes the current thread has allocated and not freed, while [`CountingAlloc`] was the
/// binary's global allocator.
pub fn live_bytes_this_thread() -> i64 {
    LIVE_BYTES.try_with(Cell::get).unwrap_or(0)
}

/// Starts a new peak window on the current thread: the peak becomes the bytes live now.
pub fn reset_peak_bytes_this_thread() {
    let live = live_bytes_this_thread();
    let _ = PEAK_BYTES.try_with(|p| p.set(live));
}

/// The most bytes the current thread has had live at once since
/// [`reset_peak_bytes_this_thread`]. A test measures a window's peak heap use as this minus the
/// live bytes at the reset.
pub fn peak_bytes_this_thread() -> i64 {
    PEAK_BYTES.try_with(Cell::get).unwrap_or(0)
}

/// Adds `delta` to this thread's live bytes and raises the peak to match. Like [`count`], it
/// never allocates and never panics.
fn track(delta: i64) {
    let _ = LIVE_BYTES.try_with(|live| {
        let now = live.get().wrapping_add(delta);
        live.set(now);
        let _ = PEAK_BYTES.try_with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

/// A size as a signed byte count. Allocation sizes never exceed `isize::MAX`.
fn signed(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// Adds one to this thread's count. It never allocates and never panics: `try_with` fails quietly
/// once the thread's locals are gone, and the add wraps.
fn count() {
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get().wrapping_add(1)));
}

// SAFETY: every method passes its arguments unchanged to `System`, which implements
// `GlobalAlloc`'s contract, and returns what `System` returns. The only other work is `count()`,
// which neither allocates, unwinds, nor touches the memory involved, so the contract the caller
// relies on is `System`'s.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        track(signed(layout.size()));
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract for `layout`, which is the
        // contract `System.alloc` needs.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        track(signed(layout.size()));
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        track(signed(new_size) - signed(layout.size()));
        // SAFETY: `ptr` came from this allocator, so from `System`, with `layout`; the caller
        // upholds the rest of `GlobalAlloc::realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        track(-signed(layout.size()));
        // SAFETY: `ptr` came from this allocator, so from `System`, with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}
