// SPDX-License-Identifier: MIT OR Apache-2.0

//! The atomics, cell and `Arc` the ring uses: the standard library's, or loom's under
//! `--cfg loom`, so the same code runs under loom's model checker.

#[cfg(loom)]
pub(crate) use loom::cell::UnsafeCell;
#[cfg(loom)]
pub(crate) use loom::sync::Arc;
#[cfg(loom)]
pub(crate) use loom::sync::atomic::{
    AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering, fence,
};

/// Threads for the unit tests: loom's under `--cfg loom`, so the model checker sees them.
#[cfg(all(test, loom))]
pub(crate) use loom::thread;
#[cfg(all(test, not(loom)))]
pub(crate) use std::thread;

#[cfg(not(loom))]
pub(crate) use std::sync::Arc;
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{
    AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering, fence,
};

/// `std::cell::UnsafeCell` with loom's `with` and `with_mut` API, so the ring's code is the same in
/// both builds.
#[cfg(not(loom))]
#[derive(Debug)]
pub(crate) struct UnsafeCell<T>(std::cell::UnsafeCell<T>);

#[cfg(not(loom))]
impl<T> UnsafeCell<T> {
    pub(crate) fn new(value: T) -> UnsafeCell<T> {
        UnsafeCell(std::cell::UnsafeCell::new(value))
    }

    /// Calls `f` with a pointer for reading the value.
    pub(crate) fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
        f(self.0.get())
    }

    /// Calls `f` with a pointer for writing the value.
    pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }
}
