// SPDX-License-Identifier: MIT OR Apache-2.0

//! Waiting for a ring to have a value (ADR-0012 item 5): a hot consumer spins, then parks its
//! thread; a consumer on an async runtime arms the ring and waits on its wake handle.
//!
//! **Never a lost wake-up.** A waiting consumer sets the ring's `waiting` flag, issues a sequentially
//! consistent fence, and checks the ring again. A producer, after its push or close, issues the same
//! fence and reads the flag. Of two such fences, one comes first in their single total order. So
//! either the consumer's second check sees the push, or the producer sees the flag and wakes the
//! consumer. The producer clears the flag when it wakes the consumer, so a busy ring, whose
//! consumer is not waiting, makes no system call.

use std::sync::OnceLock;
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use crate::ring::Consumer;
use crate::sync::{Ordering, fence};

/// How long a hot consumer spins before it parks, by default: 50 µs (ADR-0012 item 5).
pub const SPIN_DEFAULT: Duration = Duration::from_micros(50);

/// How long a hot consumer spins before it parks on battery power: 10 µs (ADR-0012 item 5).
pub const SPIN_ON_BATTERY: Duration = Duration::from_micros(10);

/// Spins between two reads of the clock while spinning, so the clock costs little.
const SPINS_PER_CLOCK_READ: u32 = 64;

/// Who a producer wakes when the consumer is waiting.
#[derive(Debug, Default)]
pub(crate) struct Waker {
    /// The thread that waits in [`Consumer::wait`], once it has.
    thread: OnceLock<Thread>,
    /// The handle an async consumer waits on, for a ring made with
    /// [`ring_with_wake`](crate::ring_with_wake).
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        windows
    ))]
    pub(crate) handle: Option<crate::wake::WakeHandle>,
}

impl Waker {
    /// A waker that also signals `handle`.
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        windows
    ))]
    pub(crate) fn with_handle(handle: crate::wake::WakeHandle) -> Waker {
        Waker {
            thread: OnceLock::new(),
            handle: Some(handle),
        }
    }

    /// Wakes the consumer: unparks its thread, if one has waited, and signals the wake handle, if
    /// the ring has one.
    pub(crate) fn notify(&self) {
        if let Some(thread) = self.thread.get() {
            thread.unpark();
        }
        #[cfg(any(
            target_os = "linux",
            target_os = "android",
            target_vendor = "apple",
            windows
        ))]
        if let Some(handle) = &self.handle {
            // Signalling fails only on an OS error the producer cannot act on. The consumer's
            // next arm still checks the ring itself, so no value is lost, only this wake-up.
            let _ = handle.notify();
        }
    }
}

/// What [`Consumer::arm`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// A value is ready to pop, or the producer is gone. The consumer is not waiting.
    Ready,
    /// The ring is empty and the consumer is waiting: the producer's next push or its close
    /// wakes it.
    Waiting,
}

impl<T> Consumer<T> {
    /// Waits until a value is ready to pop or the producer is gone: spins for `spin`, then parks
    /// this thread until the producer wakes it (ADR-0012 item 5). Returns at once if a value is
    /// ready. Allocates nothing once this thread has waited before.
    ///
    /// `std::thread::park` puts the thread to sleep: on a futex on Linux and Android, on
    /// `WaitOnAddress` on Windows, and on a dispatch semaphore on Apple platforms.
    ///
    /// # Panics
    ///
    /// If this consumer has already waited on another thread. The producer unparks that thread, so
    /// waiting here could sleep forever.
    pub fn wait(&mut self, spin: Duration) {
        if self.is_ready() {
            return;
        }
        let start = Instant::now();
        let mut spins: u32 = 0;
        loop {
            std::hint::spin_loop();
            if self.is_ready() {
                return;
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(SPINS_PER_CLOCK_READ) && start.elapsed() >= spin {
                break;
            }
        }
        let me = thread::current();
        let bound = self.shared.waker.thread.get_or_init(|| me.clone());
        assert_eq!(
            bound.id(),
            me.id(),
            "a ring's consumer waits on the thread that first waited on it"
        );
        while self.arm() == Arm::Waiting {
            thread::park();
            // `park` may return without an unpark. Ready or not, the consumer stops waiting;
            // `arm` starts the wait again if the ring is still empty.
            self.shared.flags.waiting.store(false, Ordering::Relaxed);
        }
    }

    /// [`Consumer::wait`], then [`Consumer::pop`]: the next value, or `None` once the producer is
    /// gone and nothing is left.
    pub fn pop_wait(&mut self, spin: Duration) -> Option<T> {
        loop {
            match self.pop() {
                Ok(value) => return Some(value),
                Err(crate::PopError::Closed) => return None,
                Err(crate::PopError::Empty) => self.wait(spin),
            }
        }
    }

    /// Starts a wait for a consumer that sleeps elsewhere, such as on its ring's
    /// [`WakeHandle`](crate::WakeHandle) in an async runtime. Returns [`Arm::Ready`] if a value is
    /// ready or the producer is gone, and [`Arm::Waiting`] if the producer's next push or close
    /// will wake the consumer.
    ///
    /// The loop for such a consumer: wait until the handle is signalled, `reset` it, pop until the
    /// ring is empty, and `arm` again, waiting only on [`Arm::Waiting`]. Resetting before popping
    /// matters: a signal that arrives during the pops then stays set for the next wait.
    pub fn arm(&mut self) -> Arm {
        self.shared.flags.waiting.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        if self.is_ready() {
            self.shared.flags.waiting.store(false, Ordering::Relaxed);
            Arm::Ready
        } else {
            Arm::Waiting
        }
    }

    /// Whether a value is ready to pop or the producer is gone.
    fn is_ready(&mut self) -> bool {
        self.has_value() || self.shared.flags.closed.load(Ordering::Acquire)
    }
}
