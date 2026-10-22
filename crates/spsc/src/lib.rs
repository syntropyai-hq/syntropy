// SPDX-License-Identifier: MIT OR Apache-2.0

//! The lock-free single-producer, single-consumer ring of RFC-0001 §8.3, which `io`, `net` and
//! `ipc` share.
//!
//! Layer 1 in RFC-0001 §4.1. `io`, `net` and `ipc` may depend on this crate inside layer 1, and it
//! depends on no workspace crate and has no tokio, so `exec`'s hot threads can use it too
//! (RFC-0001 §4.2 rules 2 and 3, notes of 2026-10-22).
//!
//! ```
//! let (mut tx, mut rx) = syntropy_spsc::ring::<u32>(4).unwrap();
//! tx.push(7).unwrap();
//! assert_eq!(rx.pop(), Ok(7));
//! assert_eq!(rx.pop(), Err(syntropy_spsc::PopError::Empty));
//! drop(tx);
//! assert_eq!(rx.pop(), Err(syntropy_spsc::PopError::Closed));
//! ```
//!
//! **The ring** (ADR-0012 item 4). Its capacity is a power of two, and every slot is allocated
//! when the ring is made, at `Warming`, so pushing and popping never allocate. The producer's and
//! the consumer's indices sit on cache lines of their own, and each end keeps a private copy of
//! the other's index, which it reloads only when the ring looks full or empty. A full ring
//! refuses the push and hands the value back. The caller decides whether to hold the producing
//! step or to drop the value and count it (RFC-0001 §8.4).
//!
//! **Waiting** (ADR-0012 item 5). A hot consumer calls [`Consumer::wait`]. It spins for a bounded
//! budget, [`SPIN_DEFAULT`] (50 µs) or [`SPIN_ON_BATTERY`] (10 µs), and then parks its thread with
//! `std::thread::park` until the producer pushes or closes. A consumer on an async runtime takes
//! a ring made with [`ring_with_wake`], and registers its [`WakeHandle`] with the runtime: an
//! eventfd on Linux and Android, a kqueue user event on Apple platforms, an event handle on
//! Windows. Either way, a push wakes the consumer only while it is waiting, so a busy ring makes
//! no system call.
//!
//! **Closing.** Dropping either end closes the ring. The producer's pushes then fail with
//! [`PushError::Closed`], and the consumer pops what is left, then gets [`PopError::Closed`].
//!
//! `unsafe` is allowed in this crate for the ring's slots and the wake handle's OS calls (RFC-0001
//! §4.2 rule 8). Every `unsafe` block and impl carries a `// SAFETY:` comment that states its
//! invariant (ADR-0001 item 4).
//!
//! Built with `RUSTFLAGS="--cfg loom"`, the ring uses loom's atomics and cells, so its tests run
//! under loom's model checker. The waiting and wake-handle code is left out of that build.

#![deny(clippy::undocumented_unsafe_blocks)]
#![deny(unsafe_op_in_unsafe_fn)]

mod pad;
mod ring;
mod sync;
#[cfg(not(loom))]
mod wait;
#[cfg(all(
    not(loom),
    any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        windows
    )
))]
mod wake;

pub use ring::{CapacityError, Consumer, PopError, Producer, PushError, ring};
#[cfg(not(loom))]
pub use wait::{Arm, SPIN_DEFAULT, SPIN_ON_BATTERY};
#[cfg(all(
    not(loom),
    any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        windows
    )
))]
pub use wake::{WakeHandle, ring_with_wake};
