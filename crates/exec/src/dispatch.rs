// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pollable dispatcher (RFC-0012 §3.1; RFC-0060 §2.3, amendment A3; ADR-0600 decision 3;
//! PLC-E7-T17).
//!
//! A [`Dispatcher`] runs one stage's step loop one pass at a time. [`Dispatcher::poll_once`]
//! ingests what the inbound rings hold, advances the step in flight, launches at most one new
//! step, and returns a [`PollOutcome`] that says what can happen next. It never spins and never
//! parks: the caller decides how to wait.
//!
//! - **In production**, the dispatcher thread loops over it ([`Dispatcher::run`]). It polls again
//!   at once after `Progress`, and spins with `pause` back-off while a step is in flight. It parks
//!   only between steps, on its inbound rings, or inside a step on an armed device-triggered wake
//!   (RFC-0012 §3.3).
//! - **In the simulator**, the world calls it when a device's virtual timeline says the
//!   dispatcher has work, so one OS thread runs every node (ADR-0600 decision 3).
//!
//! The dispatcher is generic over its two halves, so `exec` stays below `sched` (RFC-0001 §4.1):
//!
//! - [`StepScheduler`] is the scheduler side, which `sched` implements. It covers inbound frames,
//!   step composition, sampling at the tail, outbound frames, and when the stage next wants a
//!   poll with nothing in flight.
//! - [`StepDevice`] is the device side. It launches a step's program and services its doorbells
//!   (RFC-0012 §3.2), handing their CPU work to the compute pool ([`crate::pool`]).
//!
//! The real step loop over them, with step descriptors, doorbell flags and KV appends, is
//! PLC-E7-T03's.
//!
//! **Time** comes from the dispatcher's [`TickSource`], a generic parameter (RFC-0060 §2.1 rule
//! T1). That is the hardware counter in production and the simulator's virtual time under test.
//! Every call of `poll_once` is timed on it, and a call longer than its budget, [`POLL_BUDGET`]
//! by default, counts as an overrun in [`PollStats`]. That is the watchdog that shows a pass
//! neither spins nor parks.

use std::time::Duration;

use syntropy_core::{MonoTime, Nanos, TickSource};
use syntropy_kernels::{Error, HostWaker};
use syntropy_spsc::Arm;

pub use run::{LoopConfig, NoDeviceWake, WakeWaiter};

mod run;

/// How long one call of [`Dispatcher::poll_once`] may take before its watchdog counts an
/// overrun: 50 µs \[target\].
///
/// The figure is the spin budget between steps, `sched.spin_us` (ADR-0012 item 5). A pass that
/// holds the thread longer than the loop would spin before parking is waiting by another name,
/// and how to wait is the caller's choice, not the pass's.
pub const POLL_BUDGET: Duration = Duration::from_micros(50);

/// What one pass of [`Dispatcher::poll_once`] found (RFC-0012 §3.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PollOutcome {
    /// The pass did work: poll again at once.
    Progress,
    /// A step is in flight and nothing moved it this pass.
    Waiting {
        /// When the step's next doorbell is predicted, from the plan's segment costs. For a step
        /// held by a full outbound ring, the time of the pass.
        expect: MonoTime,
        /// How a parked waiter can be woken for that doorbell.
        wake: WakeSource,
    },
    /// Nothing is in flight. Only ring events and timers can create work.
    Idle {
        /// When the stage next wants a poll even if no frame arrives: the predicted next frame
        /// less `sched.prewake_us`, an admission deadline, or a timer (RFC-0012 §3.3).
        next_wake: Option<MonoTime>,
    },
    /// A sticky device error or a compute-pool failure. The runner reports it and exits with its
    /// class (RFC-0012 §13). Every later pass returns it again.
    Fault(Error),
}

/// How a thread parked inside a step can be woken for the step's next doorbell (RFC-0012 §3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WakeSource {
    /// No device-triggered wake is placed for this doorbell, so a waiter spins. That holds when
    /// the backend has no `arm_wake`, or the predicted gap is under
    /// `exec.inflight_park_threshold`.
    Spin,
    /// The device signals this waker after the doorbell write: the step's program carries the
    /// provider's `arm_wake` node for it (RFC-0040 §4.2).
    Device(HostWaker),
}

/// Where a step in flight stands once its fired doorbells are serviced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepProgress {
    /// The step's outputs are ready: its `O[p]` flag is set (RFC-0012 §3.2).
    Done,
    /// The step's next doorbell has not fired.
    Pending {
        /// When that doorbell is predicted.
        expect: MonoTime,
        /// How a parked waiter can be woken for it.
        wake: WakeSource,
    },
}

/// What [`StepDevice::service`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Serviced {
    /// Doorbells that had fired and were serviced in this call, each reset and its host work
    /// done.
    pub fired: u32,
    /// Where the step stands now.
    pub progress: StepProgress,
}

/// The device side of a stage: its step program on the device, and the doorbells the program
/// raises (RFC-0012 §3.2).
///
/// Every method returns without waiting. Neither spins on a flag nor synchronises the device
/// (RFC-0001 invariant 9).
pub trait StepDevice<Step> {
    /// Starts `step`'s program: writes its descriptor and submits the captured graph (RFC-0012
    /// §3.1 step 3).
    ///
    /// # Errors
    ///
    /// A device error, which faults the dispatcher.
    fn launch(&mut self, step: &mut Step, now: MonoTime) -> Result<(), Error>;

    /// Services the doorbells of `step` that have fired, in layer order. For each, it resets the
    /// flag and does the doorbell's host work, such as `assign`, a copy batch, or a work item
    /// for the CPU compute pool. It stops at the first doorbell that has not fired, because a
    /// later one cannot fire before it (RFC-0012 §3.1 step 2).
    ///
    /// # Errors
    ///
    /// A sticky device error, or a failure the compute pool reported. Either faults the
    /// dispatcher.
    fn service(&mut self, step: &mut Step, now: MonoTime) -> Result<Serviced, Error>;
}

/// The scheduler side of a stage, which `sched` implements (RFC-0012 §1.1). It owns the inbound
/// and outbound rings, the sequences, and step composition at the head and sampling at the
/// tail.
///
/// Every method returns without waiting.
pub trait StepScheduler {
    /// One step: its rows, positions and buffers, as the scheduler composed it.
    type Step;

    /// Drains the control, sequence and frame rings and ingests what they hold (RFC-0012 §3.1
    /// step 1). Returns whether anything arrived.
    fn ingest(&mut self, now: MonoTime) -> bool;

    /// The next step to launch, if the stage has one. The dispatcher asks only when no step is
    /// in flight (RFC-0012 §3.1 step 3).
    fn next_step(&mut self, now: MonoTime) -> Option<Self::Step>;

    /// The step's outputs are ready. At the tail, this samples its tokens (RFC-0012 §8, §9).
    fn finish(&mut self, step: &mut Self::Step, now: MonoTime);

    /// Queues the step's outbound frames. Returns `false` when the outbound ring is full: the
    /// step stays in flight and its credit stall accrues (RFC-0001 §8.4 rule 1).
    fn try_send(&mut self, step: &mut Self::Step, now: MonoTime) -> bool;

    /// The step has left the stage. Its record goes to telemetry.
    fn sent(&mut self, step: Self::Step, now: MonoTime);

    /// When the stage next wants a poll with nothing in flight, if it does (RFC-0012 §3.3).
    fn next_wake(&self, now: MonoTime) -> Option<MonoTime>;

    /// Starts a park on every inbound ring for the calling thread
    /// (`syntropy_spsc::Consumer::arm_thread`). Returns [`Arm::Waiting`] only if every ring is
    /// empty and armed, so the next push or close on any of them unparks the thread.
    fn arm_inbound(&mut self) -> Arm;

    /// Ends the park that [`StepScheduler::arm_inbound`] started.
    fn disarm_inbound(&mut self);

    /// Whether the stage is done: every inbound ring has closed and nothing is left to run. The
    /// production loop then returns.
    fn closed(&self) -> bool;
}

/// How the watchdog saw the passes so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PollStats {
    /// Calls of [`Dispatcher::poll_once`].
    pub polls: u64,
    /// Calls longer than the budget.
    pub overruns: u64,
    /// The longest call, on the dispatcher's [`TickSource`].
    pub longest: Nanos,
}

/// Where a step in flight is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Launched; its doorbells are serviced until it is done.
    Serving,
    /// Done and finished; its frames wait for room on the outbound ring.
    Sending,
}

struct InFlight<Step> {
    step: Step,
    phase: Phase,
    /// The step's next doorbell, as its last service found it.
    next: Option<(MonoTime, WakeSource)>,
}

/// One stage's pollable step loop (RFC-0012 §3.1). See the [module docs](self).
pub struct Dispatcher<S: StepScheduler, D, T> {
    scheduler: S,
    device: D,
    ticks: T,
    in_flight: Option<InFlight<S::Step>>,
    fault: Option<Error>,
    budget: Nanos,
    stats: PollStats,
}

impl<S, D, T> Dispatcher<S, D, T>
where
    S: StepScheduler,
    D: StepDevice<S::Step>,
    T: TickSource,
{
    /// A dispatcher over `scheduler` and `device`, timed on `ticks`, with nothing in flight and
    /// the [`POLL_BUDGET`] watchdog.
    pub fn new(scheduler: S, device: D, ticks: T) -> Dispatcher<S, D, T> {
        Dispatcher {
            scheduler,
            device,
            ticks,
            in_flight: None,
            fault: None,
            budget: duration_nanos(POLL_BUDGET),
            stats: PollStats::default(),
        }
    }

    /// The same dispatcher with the watchdog's budget set to `budget`.
    #[must_use]
    pub fn with_budget(mut self, budget: Duration) -> Dispatcher<S, D, T> {
        self.budget = duration_nanos(budget);
        self
    }

    /// One non-blocking pass (RFC-0012 §3.1):
    ///
    /// 1. ingest the inbound rings;
    /// 2. service the step in flight's fired doorbells, and once it is done, finish it and send
    ///    its frames;
    /// 3. with nothing in flight, launch the scheduler's next step.
    ///
    /// It returns `Progress` if any of that did work. Otherwise it returns `Waiting` while a step
    /// is in flight, and `Idle` only when none is. It never spins and never parks. Its time on the
    /// dispatcher's [`TickSource`] goes to the watchdog ([`Dispatcher::stats`]).
    pub fn poll_once(&mut self, now: MonoTime) -> PollOutcome {
        let started = self.ticks.ticks();
        let outcome = self.pass(now);
        let took = self
            .ticks
            .ticks_to_nanos(self.ticks.ticks().wrapping_sub(started));
        self.stats.polls += 1;
        if took > self.budget {
            self.stats.overruns += 1;
        }
        self.stats.longest = self.stats.longest.max(took);
        outcome
    }

    fn pass(&mut self, now: MonoTime) -> PollOutcome {
        if let Some(fault) = &self.fault {
            return PollOutcome::Fault(fault.clone());
        }
        // 1. Control and inbound, never blocking.
        let mut progress = self.scheduler.ingest(now);

        // 2. Advance the step in flight.
        let mut left = false;
        if let Some(flight) = &mut self.in_flight {
            if flight.phase == Phase::Serving {
                match self.device.service(&mut flight.step, now) {
                    Ok(serviced) => {
                        progress |= serviced.fired > 0;
                        match serviced.progress {
                            StepProgress::Done => {
                                self.scheduler.finish(&mut flight.step, now);
                                flight.phase = Phase::Sending;
                                flight.next = None;
                                progress = true;
                            }
                            StepProgress::Pending { expect, wake } => {
                                flight.next = Some((expect, wake));
                            }
                        }
                    }
                    Err(error) => return self.faulted(error),
                }
            }
            if flight.phase == Phase::Sending && self.scheduler.try_send(&mut flight.step, now) {
                left = true;
            }
        }
        if left && let Some(flight) = self.in_flight.take() {
            self.scheduler.sent(flight.step, now);
            progress = true;
        }

        // 3. Launch the next step if the stage is free. In latency mode that means nothing in
        //    flight (RFC-0012 §3.1 rules).
        if self.in_flight.is_none()
            && let Some(mut step) = self.scheduler.next_step(now)
        {
            if let Err(error) = self.device.launch(&mut step, now) {
                return self.faulted(error);
            }
            self.in_flight = Some(InFlight {
                step,
                phase: Phase::Serving,
                next: None,
            });
            progress = true;
        }

        if progress {
            return PollOutcome::Progress;
        }
        match &self.in_flight {
            // Never `Idle` while a step is in flight. A step held by a full outbound ring, or not
            // yet serviced, is retried at once by a spinning waiter.
            Some(flight) => {
                let (expect, wake) = flight.next.unwrap_or((now, WakeSource::Spin));
                PollOutcome::Waiting { expect, wake }
            }
            None => PollOutcome::Idle {
                next_wake: self.scheduler.next_wake(now),
            },
        }
    }

    /// Keeps `error`, so every later pass returns it too.
    fn faulted(&mut self, error: Error) -> PollOutcome {
        self.fault = Some(error.clone());
        PollOutcome::Fault(error)
    }

    /// Whether a step is in flight.
    #[must_use]
    pub fn in_flight(&self) -> bool {
        self.in_flight.is_some()
    }

    /// The watchdog's view of the passes so far.
    #[must_use]
    pub fn stats(&self) -> PollStats {
        self.stats
    }

    /// The watchdog's budget.
    #[must_use]
    pub fn budget(&self) -> Nanos {
        self.budget
    }

    /// The scheduler side.
    pub fn scheduler(&self) -> &S {
        &self.scheduler
    }

    /// The scheduler side, for a simulator that feeds it between passes.
    pub fn scheduler_mut(&mut self) -> &mut S {
        &mut self.scheduler
    }

    /// The device side.
    pub fn device(&self) -> &D {
        &self.device
    }

    /// The device side, for a simulator that moves its timeline between passes.
    pub fn device_mut(&mut self) -> &mut D {
        &mut self.device
    }

    /// The dispatcher's time source.
    pub fn ticks(&self) -> &T {
        &self.ticks
    }
}

/// `duration` in nanoseconds, saturating at `u64::MAX`.
fn duration_nanos(duration: Duration) -> Nanos {
    Nanos(u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests;
