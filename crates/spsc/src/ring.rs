// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ring: a fixed array of slots and two free-running indices (RFC-0001 §8.3).
//!
//! The producer owns `tail`, the next slot it writes, and the consumer owns `head`, the next slot
//! it reads. Both count up forever and wrap at `usize::MAX`, and a slot's position is the index
//! masked by `capacity − 1`. The ring holds `tail − head` values, so it is full at `capacity` and
//! empty at 0. Each end publishes its index with a release store, and reads the other's with an
//! acquire load. So a slot's write happens before the consumer reads it, and its read happens
//! before the producer writes it again.

use std::fmt;
use std::mem::MaybeUninit;

use crate::pad::CachePadded;
use crate::sync::{Arc, AtomicBool, AtomicUsize, Ordering, UnsafeCell, fence};
#[cfg(not(loom))]
pub(crate) use crate::wait::Waker;

/// Under loom there is no thread to park and no wake handle. The waker counts its wake-ups
/// instead, so a model can check that a consumer that armed and saw [`Arm::Waiting`] was woken.
///
/// [`Arm::Waiting`]: crate::Arm::Waiting
#[cfg(loom)]
#[derive(Debug, Default)]
pub(crate) struct Waker {
    notified: AtomicUsize,
}

#[cfg(loom)]
impl Waker {
    fn notify(&self) {
        self.notified.fetch_add(1, Ordering::Relaxed);
    }

    /// How many times the producer has woken the consumer.
    #[cfg(test)]
    pub(crate) fn notified(&self) -> usize {
        self.notified.load(Ordering::Relaxed)
    }
}

/// Why [`ring`] refused a capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityError {
    /// The capacity asked for.
    pub capacity: usize,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a ring's capacity is a power of two from 1 to {}, not {}",
            MAX_CAPACITY, self.capacity
        )
    }
}

impl std::error::Error for CapacityError {}

/// The largest capacity: the largest power of two an index difference can hold without wrapping
/// past it.
const MAX_CAPACITY: usize = 1 << (usize::BITS - 2);

/// Why a push failed. Either way the value comes back to the caller.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PushError<T> {
    /// Every slot holds a value the consumer has not read yet.
    Full(T),
    /// The consumer is gone, so nothing will read the value.
    Closed(T),
}

impl<T> PushError<T> {
    /// The value that was not pushed.
    pub fn into_inner(self) -> T {
        match self {
            PushError::Full(value) | PushError::Closed(value) => value,
        }
    }
}

impl<T> fmt::Debug for PushError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PushError::Full(_) => f.write_str("Full(..)"),
            PushError::Closed(_) => f.write_str("Closed(..)"),
        }
    }
}

impl<T> fmt::Display for PushError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PushError::Full(_) => f.write_str("the ring is full"),
            PushError::Closed(_) => f.write_str("the ring's consumer is gone"),
        }
    }
}

impl<T> std::error::Error for PushError<T> {}

/// Why a pop returned no value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PopError {
    /// The ring is empty, and the producer may push more.
    Empty,
    /// The ring is empty, and the producer is gone, so it never will be anything else.
    Closed,
}

impl fmt::Display for PopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PopError::Empty => f.write_str("the ring is empty"),
            PopError::Closed => f.write_str("the ring is empty and its producer is gone"),
        }
    }
}

impl std::error::Error for PopError {}

/// The flags both ends write, rarely: together on a cache line of their own.
#[derive(Debug)]
pub(crate) struct Flags {
    /// Set when either end is dropped.
    pub(crate) closed: AtomicBool,
    /// Set by a consumer about to wait, and cleared by the producer that wakes it.
    pub(crate) waiting: AtomicBool,
}

pub(crate) struct Shared<T> {
    /// The consumer's index: the next slot to read.
    head: CachePadded<AtomicUsize>,
    /// The producer's index: the next slot to write.
    tail: CachePadded<AtomicUsize>,
    pub(crate) flags: CachePadded<Flags>,
    /// `capacity − 1`.
    mask: usize,
    /// The slots from `head` up to `tail` hold values; the others are uninitialised.
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Who to wake when the consumer waits: its thread, and the wake handle if the ring has one.
    pub(crate) waker: Waker,
}

// SAFETY: the producer and the consumer each hold an `Arc<Shared<T>>`, possibly on two threads.
// A slot's value moves from the producer's thread to the consumer's, which needs `T: Send`. Each
// slot is written only by the producer, while it lies outside `head..tail`, and read only by the
// consumer, while it lies inside, and the release and acquire pair on the index that moves it
// across orders the two. No `&T` is ever shared between the threads, so `T: Sync` is not needed.
unsafe impl<T: Send> Send for Shared<T> {}
// SAFETY: as for `Send`: shared access goes through the atomics and through slots that, by the
// index protocol above, only one end touches at a time.
unsafe impl<T: Send> Sync for Shared<T> {}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        // Both ends are gone, and `Arc`'s last drop orders their final index stores before this.
        let tail = self.tail.load(Ordering::Acquire);
        let mut at = self.head.load(Ordering::Acquire);
        while at != tail {
            let slot = &self.slots[at & self.mask];
            // SAFETY: slots in `head..tail` hold values the consumer never read, and with both
            // ends gone nothing else can touch them, so each is dropped exactly once.
            slot.with_mut(|value| unsafe { (*value).assume_init_drop() });
            at = at.wrapping_add(1);
        }
    }
}

impl<T> Shared<T> {
    /// Wakes the consumer if it is waiting, after the push or close that made it ready. The
    /// fence pairs with the one in the consumer's wait: either the consumer sees the new index or
    /// flag, or this sees `waiting` set, so a wake-up is never lost.
    pub(crate) fn wake_consumer(&self) {
        fence(Ordering::SeqCst);
        if self.flags.waiting.load(Ordering::Relaxed)
            && self.flags.waiting.swap(false, Ordering::AcqRel)
        {
            self.waker.notify();
        }
    }
}

/// The writing end of a ring. It is `Send`, so it can move to the thread that produces, and it
/// is not `Clone`: there is one producer.
pub struct Producer<T> {
    shared: Arc<Shared<T>>,
    /// This end's copy of `tail`.
    tail: usize,
    /// The consumer's `head` as last read; the ring is no fuller than this says.
    head: usize,
}

/// The reading end of a ring. It is `Send`, so it can move to the thread that consumes, and it is
/// not `Clone`: there is one consumer.
pub struct Consumer<T> {
    pub(crate) shared: Arc<Shared<T>>,
    /// This end's copy of `head`.
    head: usize,
    /// The producer's `tail` as last read; the ring holds at least `tail − head` values.
    tail: usize,
}

/// A ring of `capacity` slots, all allocated now. `capacity` must be a power of two.
///
/// # Errors
///
/// [`CapacityError`] when `capacity` is 0, not a power of two, or above 2^62 on a 64-bit target.
pub fn ring<T>(capacity: usize) -> Result<(Producer<T>, Consumer<T>), CapacityError> {
    with_waker(capacity, Waker::default())
}

/// [`ring`], with the given [`Waker`].
pub(crate) fn with_waker<T>(
    capacity: usize,
    waker: Waker,
) -> Result<(Producer<T>, Consumer<T>), CapacityError> {
    starting_at(capacity, waker, 0)
}

/// A ring whose indices both start at `start`, so a test can run them across `usize::MAX`.
pub(crate) fn starting_at<T>(
    capacity: usize,
    waker: Waker,
    start: usize,
) -> Result<(Producer<T>, Consumer<T>), CapacityError> {
    if !capacity.is_power_of_two() || capacity > MAX_CAPACITY {
        return Err(CapacityError { capacity });
    }
    let slots = (0..capacity)
        .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
        .collect();
    let shared = Arc::new(Shared {
        head: CachePadded(AtomicUsize::new(start)),
        tail: CachePadded(AtomicUsize::new(start)),
        flags: CachePadded(Flags {
            closed: AtomicBool::new(false),
            waiting: AtomicBool::new(false),
        }),
        mask: capacity - 1,
        slots,
        waker,
    });
    Ok((
        Producer {
            shared: shared.clone(),
            tail: start,
            head: start,
        },
        Consumer {
            shared,
            head: start,
            tail: start,
        },
    ))
}

impl<T> Producer<T> {
    /// Pushes `value`, and wakes the consumer if it is waiting. Never allocates or blocks.
    ///
    /// # Errors
    ///
    /// [`PushError::Full`] when every slot holds an unread value, and [`PushError::Closed`] when
    /// the consumer is gone. Both hand `value` back.
    pub fn push(&mut self, value: T) -> Result<(), PushError<T>> {
        let shared = &*self.shared;
        if shared.flags.closed.load(Ordering::Acquire) {
            return Err(PushError::Closed(value));
        }
        if self.tail.wrapping_sub(self.head) > shared.mask {
            self.head = shared.head.load(Ordering::Acquire);
            if self.tail.wrapping_sub(self.head) > shared.mask {
                return Err(PushError::Full(value));
            }
        }
        let slot = &shared.slots[self.tail & shared.mask];
        // SAFETY: the slot at `tail` is outside `head..tail`: the acquire load of `head` above, or
        // an earlier one, saw the consumer's read of it finish, so no one else touches it. Writing
        // a `MaybeUninit` drops nothing, and the slot held no live value.
        slot.with_mut(|cell| unsafe { cell.write(MaybeUninit::new(value)) });
        self.tail = self.tail.wrapping_add(1);
        shared.tail.store(self.tail, Ordering::Release);
        shared.wake_consumer();
        Ok(())
    }

    /// The number of slots.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.shared.mask + 1
    }

    /// Whether the consumer is gone.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.shared.flags.closed.load(Ordering::Acquire)
    }
}

impl<T> Drop for Producer<T> {
    fn drop(&mut self) {
        // The close is ordered after the last push twice over: it is a release store, and every
        // push ends with `wake_consumer`'s sequentially consistent fence, which alone makes any
        // later store, this one included, a release for a consumer that reads it. So a relaxed
        // close would still be correct today, and no test can tell the two apart. The release
        // keeps the close correct if the push path ever loses its fence
        // (`a_value_pushed_before_the_close_is_never_lost`).
        self.shared.flags.closed.store(true, Ordering::Release);
        self.shared.wake_consumer();
    }
}

impl<T> Consumer<T> {
    /// Pops the oldest value. Never allocates or blocks.
    ///
    /// # Errors
    ///
    /// [`PopError::Empty`] when the ring is empty, and [`PopError::Closed`] when it is empty and
    /// the producer is gone. Values pushed before the producer went are still popped first.
    pub fn pop(&mut self) -> Result<T, PopError> {
        if !self.has_value() {
            return Err(if self.is_closed_and_empty() {
                PopError::Closed
            } else {
                PopError::Empty
            });
        }
        let shared = &*self.shared;
        let slot = &shared.slots[self.head & shared.mask];
        // SAFETY: the slot at `head` is inside `head..tail`: the acquire load of `tail` in
        // `has_value` saw the producer's write of it, and the producer will not write it again
        // until the release store of `head` below. Reading it once moves the value out, and the
        // slot counts as uninitialised from then on.
        let value = slot.with(|cell| unsafe { (*cell).assume_init_read() });
        self.head = self.head.wrapping_add(1);
        shared.head.store(self.head, Ordering::Release);
        Ok(value)
    }

    /// Whether a value is ready to pop, reloading the producer's index if this end's copy says
    /// the ring is empty.
    pub(crate) fn has_value(&mut self) -> bool {
        if self.head == self.tail {
            self.tail = self.shared.tail.load(Ordering::Acquire);
        }
        self.head != self.tail
    }

    /// Whether the producer is gone and nothing is left to pop. The producer's last push happens
    /// before its close, so seeing the close and then reloading `tail` finds that push.
    pub(crate) fn is_closed_and_empty(&mut self) -> bool {
        self.shared.flags.closed.load(Ordering::Acquire) && !self.has_value()
    }

    /// The number of values ready to pop: at least this many, since the producer may push more.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shared
            .tail
            .load(Ordering::Acquire)
            .wrapping_sub(self.head)
    }

    /// Whether no value is ready to pop.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of slots.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.shared.mask + 1
    }
}

impl<T> Drop for Consumer<T> {
    fn drop(&mut self) {
        self.shared.flags.closed.store(true, Ordering::Release);
    }
}

impl<T> fmt::Debug for Producer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Producer")
            .field("capacity", &self.capacity())
            .field("closed", &self.is_closed())
            .finish()
    }
}

impl<T> fmt::Debug for Consumer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Consumer")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests;
