// SPDX-License-Identifier: MIT OR Apache-2.0

//! A CPU device's streams: in-order work queues (RFC-0040 §7.3, §4.1).
//!
//! A CPU stream is a preallocated single-producer, single-consumer ring of work items
//! (`syntropy-spsc`). The device's provider calls push onto it: launches, copies, event records
//! and waits, doorbell signals and waits, and wakes. A [`StreamRunner`] executes the items in
//! order, on whichever thread holds it:
//!
//! - a CPU pool thread (PLC-E7-T07);
//! - the simulator, through `ComputePool::run_inline` (RFC-0060 §2.3);
//! - `synchronize`, which drains the stream on the calling thread.
//!
//! [`StreamRunner::run`] never spins or parks. It returns at the first item that has to wait
//! (a doorbell not yet raised, or an event not yet reached), and the caller decides how to wait.
//! [`StreamRunner::drain`] is that decision for a thread inside a step: an acquire spin with
//! `pause` back-off, which parks only on an armed device-triggered wake (RFC-0040 §4.2, the CPU
//! row).
//!
//! Pushing and running allocate nothing and take no lock: the ring is preallocated, and an item
//! holds only plain values, a kernel's entry point, and `Arc` handles that a push clones.

use std::sync::Arc;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::time::Duration;

use syntropy_kernels::{BufferSlice, Error, HostFlag, OpArgs, WaitCmp};
use syntropy_spsc::{Consumer, PopError};

use crate::dispatch::KernelFn;
use crate::doorbell::Doorbells;
use crate::wake::{CpuWaker, Parked};

/// The items one stream's ring holds: 4096 \[target\], a chosen budget. A decode step's op list
/// is tens to a few hundred items, two descriptor parities at most are queued at once, and a
/// full ring is refused at push, never grown.
pub const STREAM_CAPACITY: usize = 4096;

/// How many back-off rounds [`StreamRunner::drain`] spins before it considers parking:
/// 1 + 2 + … + 64 `pause` hints, then 64 per round. 64 rounds is about 4,000 hints, a few
/// microseconds \[estimate: tens of cycles per hint at 3–4 GHz].
const SPIN_ROUNDS: u32 = 64;

/// The most `pause` hints in one back-off round.
const MAX_PAUSES: u32 = 64;

/// How long one park in [`StreamRunner::drain`] lasts at most before the runner looks again:
/// 100 ms \[target\]. A wake that never comes then costs a periodic look, not a hang, and RFC-0012
/// §13's stall watchdog still reports the stalled step.
const PARK_LIMIT: Duration = Duration::from_millis(100);

/// An event: how many records were issued, and how far the device has got.
#[derive(Debug, Default)]
pub(crate) struct EventState {
    /// Records pushed so far. Only the device's own thread pushes, but `poll` reads it.
    issued: AtomicU64,
    /// The latest record the runner reached.
    done: AtomicU64,
}

impl EventState {
    /// A new record: its generation.
    fn issue(&self) -> u64 {
        self.issued.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// The latest record pushed, which a wait pushed now waits for.
    fn issued(&self) -> u64 {
        self.issued.load(Ordering::Relaxed)
    }

    /// Whether every record pushed so far has been reached: `poll`.
    pub(crate) fn complete(&self) -> bool {
        self.done.load(Ordering::Acquire) >= self.issued()
    }

    fn reached(&self, generation: u64) -> bool {
        self.done.load(Ordering::Acquire) >= generation
    }
}

/// The device's health, shared by its runners: 0, or the code of the sticky error.
#[derive(Debug, Default)]
pub(crate) struct Health {
    sticky: AtomicU16,
}

impl Health {
    pub(crate) fn sticky(&self) -> Option<u16> {
        match self.sticky.load(Ordering::Acquire) {
            0 => None,
            code => Some(code),
        }
    }

    fn stick(&self, code: u16) {
        // The first sticky error is the one the device reports.
        let _ = self
            .sticky
            .compare_exchange(0, code.max(1), Ordering::AcqRel, Ordering::Acquire);
    }
}

/// One unit of a stream's work.
#[derive(Clone, Debug)]
pub(crate) enum Item {
    /// Runs a kernel on the runner's thread.
    Launch(KernelFn),
    /// Copies `len` bytes between two host addresses, resolved and bounds-checked at push.
    Copy { src: u64, dst: u64, len: usize },
    /// A copy named by buffer handles, as a captured op list keeps it. It is resolved to a
    /// [`Item::Copy`] each time it is pushed, so a buffer freed since the capture fails the
    /// replay instead of being written. It never reaches a ring.
    CopyRef { src: BufferSlice, dst: BufferSlice },
    /// Marks the event reached at `generation`. A captured record gets its generation at
    /// each replay.
    Record {
        event: Arc<EventState>,
        generation: u64,
    },
    /// Waits for the event to reach `generation`: the latest record pushed before the wait.
    WaitEvent {
        event: Arc<EventState>,
        generation: u64,
    },
    /// Raises a doorbell: a release store of `value`.
    Signal { slot: u32, value: u32 },
    /// Waits for a doorbell, then writes `reset`: RFC-0040 §4.2's consumer reset.
    WaitHost {
        slot: u32,
        value: u32,
        cmp: WaitCmp,
        reset: Option<u32>,
    },
    /// Wakes a parked waiter, after the stores before it.
    Wake(Arc<CpuWaker>),
}

impl Item {
    /// The item to push now. A record takes the event's next generation, and a wait the latest
    /// one, so a replayed op list records and waits afresh each time.
    pub(crate) fn resolved(&self) -> Item {
        match self {
            Item::Record { event, .. } => Item::Record {
                generation: event.issue(),
                event: Arc::clone(event),
            },
            Item::WaitEvent { event, .. } => Item::WaitEvent {
                generation: event.issued(),
                event: Arc::clone(event),
            },
            other => other.clone(),
        }
    }
}

/// What a stream's runner and its device share.
#[derive(Debug, Default)]
pub(crate) struct StreamShared {
    /// Items the runner has finished.
    pub(crate) completed: AtomicU64,
    pub(crate) health: Arc<Health>,
}

/// What stopped [`StreamRunner::run`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Run {
    /// Every item pushed so far has run.
    Idle,
    /// The next item waits for a doorbell that is not raised yet.
    WaitingForFlag(HostFlag),
    /// The next item waits for an event not reached yet.
    WaitingForEvent,
}

/// Executes one stream's items, in order, on the thread that holds it. `Send`, so a CPU pool
/// thread can take it from the device (`CpuProvider::take_runner`).
#[derive(Debug)]
pub struct StreamRunner {
    rx: Consumer<Item>,
    /// An item that had to wait, popped but not run.
    pending: Option<Item>,
    doorbells: Doorbells,
    shared: Arc<StreamShared>,
}

impl StreamRunner {
    pub(crate) fn new(rx: Consumer<Item>, doorbells: Doorbells, shared: Arc<StreamShared>) -> Self {
        StreamRunner {
            rx,
            pending: None,
            doorbells,
            shared,
        }
    }

    /// Runs items in order until the stream is empty or the next item has to wait. It never
    /// spins and never parks: a waiting item is kept and tried first next time.
    ///
    /// # Errors
    ///
    /// A kernel's error, which makes the device's health sticky, or the sticky error of an
    /// earlier run.
    pub fn run(&mut self) -> Result<Run, Error> {
        if let Some(code) = self.shared.health.sticky() {
            return Err(sticky(code));
        }
        loop {
            let item = match self.pending.take() {
                Some(item) => item,
                None => match self.rx.pop() {
                    Ok(item) => item,
                    Err(PopError::Empty | PopError::Closed) => return Ok(Run::Idle),
                },
            };
            match self.execute(&item)? {
                Run::Idle => {
                    self.shared.completed.fetch_add(1, Ordering::Release);
                }
                waiting => {
                    self.pending = Some(item);
                    return Ok(waiting);
                }
            }
        }
    }

    /// Runs the stream until it is empty, waiting the way a thread inside a step may: an
    /// acquire spin with `pause` back-off. With `waker`, a long wait parks on it, armed for the
    /// doorbell's producer to wake; without one, it never parks (RFC-0040 §4.2, the CPU row).
    ///
    /// Returns how many parks the wake ended and how many ended at their time limit.
    ///
    /// # Errors
    ///
    /// As [`StreamRunner::run`].
    pub fn drain(&mut self, waker: Option<&CpuWaker>) -> Result<Parks, Error> {
        let mut parks = Parks::default();
        let mut pauses = 1;
        let mut rounds = 0;
        loop {
            if self.run()? == Run::Idle {
                return Ok(parks);
            }
            match waker {
                Some(waker) if rounds >= SPIN_ROUNDS => {
                    waker.arm();
                    // A look after arming, so a doorbell raised before the arm is not slept
                    // through.
                    if self.run()? == Run::Idle {
                        waker.disarm();
                        return Ok(parks);
                    }
                    match waker.park(PARK_LIMIT) {
                        Parked::Woken => parks.woken += 1,
                        Parked::TimedOut => parks.timed_out += 1,
                    }
                    waker.disarm();
                    rounds = 0;
                    pauses = 1;
                }
                _ => {
                    for _ in 0..pauses {
                        std::hint::spin_loop();
                    }
                    pauses = (pauses * 2).min(MAX_PAUSES);
                    rounds += 1;
                }
            }
        }
    }

    fn execute(&self, item: &Item) -> Result<Run, Error> {
        match item {
            Item::Launch(kernel) => {
                if let Err(error) = kernel(&OpArgs::none()) {
                    self.shared.health.stick(error.code.0);
                    return Err(error);
                }
            }
            Item::Copy { src, dst, len } => {
                // SAFETY: the provider resolved both addresses at push from buffers of this
                // device that cover `len` bytes from them, and the tier manager frees a buffer
                // only between steps, once the streams using it have drained (RFC-0001 §9.1).
                // `copy` allows the two ranges to overlap.
                unsafe {
                    std::ptr::copy(*src as *const u8, *dst as *mut u8, *len);
                }
            }
            Item::CopyRef { .. } => {
                return Err(crate::device::internal(
                    "an unresolved copy reached a stream's ring".to_owned(),
                ));
            }
            Item::Record { event, generation } => {
                // `fetch_max`, so an event recorded on two streams never moves back.
                event.done.fetch_max(*generation, Ordering::AcqRel);
            }
            Item::WaitEvent { event, generation } => {
                if !event.reached(*generation) {
                    return Ok(Run::WaitingForEvent);
                }
            }
            Item::Signal { slot, value } => {
                self.doorbells.at(*slot).store(*value, Ordering::Release);
            }
            Item::WaitHost {
                slot,
                value,
                cmp,
                reset,
            } => {
                let flag = self.doorbells.at(*slot);
                let seen = flag.load(Ordering::Acquire);
                let passed = match cmp {
                    WaitCmp::Eq => seen == *value,
                    WaitCmp::Geq => seen >= *value,
                };
                if !passed {
                    return Ok(Run::WaitingForFlag(self.doorbells.flag_at(*slot)));
                }
                if let Some(reset) = reset {
                    flag.store(*reset, Ordering::Release);
                }
            }
            Item::Wake(waker) => {
                let _ = waker.wake();
            }
        }
        Ok(Run::Idle)
    }
}

/// How the parks in a [`StreamRunner::drain`] ended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Parks {
    /// Parks the producer's wake ended.
    pub woken: u32,
    /// Parks that reached their time limit with no wake.
    pub timed_out: u32,
}

/// The error a device with a sticky error returns.
pub(crate) fn sticky(code: u16) -> Error {
    Error {
        class: syntropy_core::ErrorClass::Device,
        code: syntropy_kernels::ErrorCode(code),
        message: "the CPU device has a sticky error from an earlier kernel".to_owned(),
    }
}
