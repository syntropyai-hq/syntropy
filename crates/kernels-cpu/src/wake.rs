// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU provider's device-triggered wake (RFC-0040 §4.2, `arm_wake`'s CPU row; RFC-0012
//! §3.3).
//!
//! A thread that waits inside a step may park only on a wake armed for its next doorbell. On
//! the CPU, "the device" is a stream runner, and its wake is the writer's: after the release
//! store that raises the doorbell, the runner wakes the waiter if the waiter has announced that
//! it is parking. A [`CpuWaker`] is that announcement and that wake:
//!
//! - the waiter calls [`CpuWaker::arm`], checks its doorbell once more, and if the doorbell is
//!   still down calls [`CpuWaker::park`];
//! - the runner, at the stream's `arm_wake` item, calls [`CpuWaker::wake`].
//!
//! **Never a lost wake-up.** `arm` sets the waker's `waiting` word, then issues a sequentially
//! consistent fence, before the waiter's second look at the doorbell. `wake` issues the same
//! fence after the doorbell's store, then reads the word. Of the two fences one comes first, so
//! either the waiter sees the doorbell or the runner sees the word and unparks the waiter. The
//! runner clears the word as it wakes, so it makes no wake-up call for a waiter that is not
//! parking.
//!
//! The park is `std::thread::park_timeout`: a futex on Linux and Android, `WaitOnAddress` on
//! Windows, and a dispatch semaphore on Apple platforms. A waiter learns from
//! [`CpuWaker::park`] whether the wake or its timeout ended the park, which is how a test shows
//! that no step waited for a timer.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering, fence};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// The wake a parked waiter is waiting for. One waiter thread per waker: the first to arm it.
#[derive(Debug, Default)]
pub struct CpuWaker {
    /// 1 while the waiter is parking or parked; cleared by the wake.
    waiting: AtomicU32,
    /// The waiter's thread, bound by its first `arm`.
    thread: OnceLock<Thread>,
}

/// How a [`CpuWaker::park`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parked {
    /// The runner's wake ended it.
    Woken,
    /// The timeout ended it with no wake.
    TimedOut,
}

impl CpuWaker {
    /// A waker no thread waits on yet.
    #[must_use]
    pub fn new() -> CpuWaker {
        CpuWaker::default()
    }

    /// Announces that the calling thread is about to park. The caller then checks its doorbell
    /// once more, and parks only if it is still down.
    ///
    /// # Panics
    ///
    /// If another thread armed this waker before. The wake unparks that thread, so parking
    /// here could sleep through it.
    pub fn arm(&self) {
        let me = thread::current();
        let bound = self.thread.get_or_init(|| me.clone());
        assert_eq!(
            bound.id(),
            me.id(),
            "a CpuWaker wakes the thread that first armed it"
        );
        self.waiting.store(1, Ordering::SeqCst);
        fence(Ordering::SeqCst);
    }

    /// Parks until the wake, or until `timeout` passes with none. `park_timeout` may return
    /// early with no unpark, so this parks again until one or the other happens.
    pub fn park(&self, timeout: Duration) -> Parked {
        // The real clock, on purpose: the timeout bounds how long the thread sleeps, never what
        // the step computes or in what order, as with the ring's spin budget in `spsc`.
        let deadline = Instant::now() + timeout;
        while self.waiting.load(Ordering::Acquire) == 1 {
            let now = Instant::now();
            if now >= deadline {
                return if self.waiting.swap(0, Ordering::AcqRel) == 1 {
                    Parked::TimedOut
                } else {
                    Parked::Woken
                };
            }
            thread::park_timeout(deadline - now);
        }
        Parked::Woken
    }

    /// Ends the announcement, whether or not the thread parked.
    pub fn disarm(&self) {
        self.waiting.store(0, Ordering::Release);
    }

    /// The writer's wake, after its release store: unparks the waiter if it announced that it
    /// is parking. Returns whether it did. Allocation-free and lock-free.
    pub fn wake(&self) -> bool {
        fence(Ordering::SeqCst);
        if self.waiting.load(Ordering::Relaxed) == 1 && self.waiting.swap(0, Ordering::AcqRel) == 1
        {
            if let Some(thread) = self.thread.get() {
                thread.unpark();
            }
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use super::*;

    #[test]
    fn a_parked_waiter_is_woken_and_says_so() {
        let waker = Arc::new(CpuWaker::new());
        let doorbell = Arc::new(AtomicBool::new(false));
        let (w, d) = (Arc::clone(&waker), Arc::clone(&doorbell));
        let waiter = thread::spawn(move || {
            w.arm();
            let outcome = if d.load(Ordering::Acquire) {
                Parked::Woken
            } else {
                w.park(Duration::from_secs(10))
            };
            w.disarm();
            outcome
        });
        thread::sleep(Duration::from_millis(20));
        doorbell.store(true, Ordering::Release);
        let _ = waker.wake();
        assert_eq!(waiter.join().unwrap(), Parked::Woken);
    }

    #[test]
    fn no_wake_ends_in_a_timeout_and_an_unarmed_waker_makes_no_call() {
        let waker = CpuWaker::new();
        assert!(!waker.wake(), "nobody is parking");
        waker.arm();
        assert_eq!(waker.park(Duration::from_millis(5)), Parked::TimedOut);
        waker.disarm();
        assert!(!waker.wake());
    }

    #[test]
    fn a_waker_wakes_one_thread() {
        let waker = Arc::new(CpuWaker::new());
        waker.arm();
        waker.disarm();
        let w = Arc::clone(&waker);
        let panic = thread::spawn(move || w.arm()).join().unwrap_err();
        let message = panic.downcast_ref::<String>().unwrap();
        assert!(message.contains("first armed it"), "{message}");
    }
}
