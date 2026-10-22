// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's world and event loop (RFC-0060 §8.2, §8.6; ADR-0600 decisions 5–6).
//!
//! A [`SimWorld`] runs one simulated timeline on one thread. It owns:
//! - the [`SimClock`], which only its loop advances;
//! - a [`SimEntropy`] built from the seed, the run's only randomness;
//! - the event queue;
//! - the [`Model`]: the state the events act on;
//! - the [`Checker`]s;
//! - the [`TraceHasher`].
//!
//! **The loop.** [`SimWorld::run_until`] pops the earliest event and advances the clock to its
//! instant. It hands the event to the model, which schedules what it emits through a
//! [`Schedule`]. It records the event in the trace, then runs every checker's safety check
//! against the new state. Liveness checks run at milestones the scenario sets. A failed check
//! stops the run with a [`Violation`] carrying the seed, the event index and the trace hash of
//! the events up to it, so the run can be replayed to that point.
//!
//! **Order.** Events are ordered by instant, then by a tie-break drawn from the seeded entropy
//! when the event is scheduled, then by the order of scheduling. Different seeds explore
//! different interleavings of simultaneous events, which is where most distributed races live,
//! and one seed always replays one interleaving (ADR-0600 decision 5).
//!
//! The world is generic over its model. The cluster the later simulator tasks build is one
//! model: its nodes, the fabric (QUAL-E8-T02) and the scenario's events (§8.2's `SimEvent`).

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::sync::Arc;

use syntropy_core::{Blake3, Clock, Entropy, MonoTime, Nanos};

use super::trace::{Hex, TraceEvent, TraceHasher};
use super::{SimClock, SimClockHandle, SimEntropy};

/// What a simulation runs: the state its events act on.
pub trait Model {
    /// One event. The trace hash records it ([`TraceEvent`]).
    type Event: TraceEvent;

    /// Handles `event` at `now`, scheduling what it emits through `schedule`.
    fn handle(
        &mut self,
        now: MonoTime,
        event: Self::Event,
        schedule: &mut Schedule<'_, Self::Event>,
    );
}

/// A property of a model, checked as the run goes (RFC-0060 §8.6).
///
/// A safety property ([`after_event`](Self::after_event)) must hold after every event, and a
/// liveness property ([`at_milestone`](Self::at_milestone)) by each milestone. A check returns
/// `Err` with what it found broken, and the run stops there.
pub trait Checker<M> {
    /// The checker's name, as a violation reports it.
    fn name(&self) -> &str;

    /// Checks the model after an event.
    fn after_event(&mut self, model: &M, now: MonoTime) -> Result<(), String> {
        let _ = (model, now);
        Ok(())
    }

    /// Checks the model at the milestone `label`, once every event up to its instant has run.
    fn at_milestone(&mut self, model: &M, now: MonoTime, label: &str) -> Result<(), String> {
        let _ = (model, now, label);
        Ok(())
    }
}

/// What a model's code is given instead of the host's clock and randomness (RFC-0060 §2.1 rule
/// T1): a read-only view of the world's clock and the world's entropy.
#[derive(Clone, Debug)]
pub struct Env {
    pub clock: SimClockHandle,
    pub entropy: Arc<SimEntropy>,
}

/// One queued event and its place in the order.
struct Scheduled<E> {
    at: MonoTime,
    tiebreak: u64,
    seq: u64,
    event: E,
}

impl<E> Scheduled<E> {
    fn key(&self) -> (MonoTime, u64, u64) {
        (self.at, self.tiebreak, self.seq)
    }
}

impl<E> PartialEq for Scheduled<E> {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl<E> Eq for Scheduled<E> {}

impl<E> PartialOrd for Scheduled<E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E> Ord for Scheduled<E> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key().cmp(&other.key())
    }
}

/// The event queue: earliest first, then by tie-break, then by scheduling order.
struct Queue<E> {
    heap: BinaryHeap<Reverse<Scheduled<E>>>,
    /// The scheduling order of the next event; it never repeats, so the order is total.
    next_seq: u64,
}

impl<E> Queue<E> {
    fn push(&mut self, at: MonoTime, event: E, entropy: &SimEntropy) {
        let scheduled = Scheduled {
            at,
            tiebreak: entropy.u64(),
            seq: self.next_seq,
            event,
        };
        self.next_seq += 1;
        self.heap.push(Reverse(scheduled));
    }

    fn next_at(&self) -> Option<MonoTime> {
        self.heap.peek().map(|Reverse(scheduled)| scheduled.at)
    }
}

/// How a handler schedules the events it emits.
pub struct Schedule<'w, E> {
    now: MonoTime,
    queue: &'w mut Queue<E>,
    entropy: &'w SimEntropy,
}

impl<E> Schedule<'_, E> {
    /// The instant of the event being handled.
    pub fn now(&self) -> MonoTime {
        self.now
    }

    /// Schedules `event` at `at`. An event at the current instant runs after the one being
    /// handled, in an order the seed decides among the others at that instant.
    ///
    /// # Panics
    ///
    /// If `at` is earlier than now: a handler that schedules into the past has a bug.
    pub fn at(&mut self, at: MonoTime, event: E) {
        assert!(
            at >= self.now,
            "an event scheduled in the past: {at:?} at {:?}",
            self.now
        );
        self.queue.push(at, event, self.entropy);
    }

    /// Schedules `event` `delay` after now.
    pub fn after(&mut self, delay: Nanos, event: E) {
        let at = self
            .now
            .checked_add(delay)
            .unwrap_or_else(|| panic!("virtual time overflows: {:?} + {delay:?}", self.now));
        self.at(at, event);
    }

    /// The run's entropy, for a handler that draws at random.
    pub fn entropy(&self) -> &SimEntropy {
        self.entropy
    }
}

/// Where a check failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Found {
    /// A safety check, after the event at this index (from 0) in processing order.
    AfterEvent { index: u64 },
    /// A liveness check, at this milestone.
    AtMilestone { label: String },
}

/// A failed check, with what a replay needs to reach it (RFC-0060 §8.6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub checker: String,
    pub message: String,
    pub seed: u64,
    pub found: Found,
    /// The virtual instant of the check.
    pub at: MonoTime,
    /// Events processed when the check failed. For a safety check the last of them is the event
    /// it failed after.
    pub events: u64,
    /// The trace hash of those events: a replay with the same seed reaches the same value at the
    /// same point.
    pub trace_prefix: Blake3,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "checker `{}` failed ", self.checker)?;
        match &self.found {
            Found::AfterEvent { index } => write!(f, "after event {index}")?,
            Found::AtMilestone { label } => write!(f, "at milestone `{label}`")?,
        }
        write!(
            f,
            " (t = {} ns, seed {}, {} events, trace prefix {}): {}",
            self.at.0,
            self.seed,
            self.events,
            Hex(&self.trace_prefix),
            self.message
        )
    }
}

/// How [`SimWorld::run_until`] ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    /// Every event and milestone up to `at` ran, and the clock is at `at`.
    Reached { at: MonoTime, events: u64 },
    /// A check failed, and the world stopped there. Later runs return the same violation.
    Violated(Violation),
}

/// One simulated timeline (RFC-0060 §8.2).
pub struct SimWorld<M: Model> {
    seed: u64,
    clock: SimClock,
    entropy: Arc<SimEntropy>,
    queue: Queue<M::Event>,
    /// Milestones by instant, then by the order they were set.
    milestones: BTreeMap<(MonoTime, u64), String>,
    next_milestone: u64,
    model: M,
    checkers: Vec<Box<dyn Checker<M>>>,
    trace: TraceHasher,
    stopped: Option<Violation>,
}

impl<M: Model> SimWorld<M> {
    /// A world for `seed`, at virtual time 0, with the model `build` makes from the world's
    /// clock and entropy. Nothing is scheduled yet.
    pub fn new(seed: u64, build: impl FnOnce(Env) -> M) -> SimWorld<M> {
        let clock = SimClock::new();
        let entropy = Arc::new(SimEntropy::new(seed));
        let model = build(Env {
            clock: clock.handle(),
            entropy: Arc::clone(&entropy),
        });
        SimWorld {
            seed,
            clock,
            entropy,
            queue: Queue {
                heap: BinaryHeap::new(),
                next_seq: 0,
            },
            milestones: BTreeMap::new(),
            next_milestone: 0,
            model,
            checkers: Vec::new(),
            trace: TraceHasher::new(),
            stopped: None,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The current virtual instant.
    pub fn now(&self) -> MonoTime {
        self.clock.now()
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    /// The model, for a scenario to set up before the run. An event is the way to change it
    /// during a run.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Events processed so far.
    pub fn events(&self) -> u64 {
        self.trace.events()
    }

    /// Events scheduled and not yet processed.
    pub fn pending(&self) -> usize {
        self.queue.heap.len()
    }

    /// The trace hash of the events processed so far.
    pub fn trace_hash(&self) -> Blake3 {
        self.trace.digest()
    }

    /// Schedules `event` at `at`, from outside the run: a scenario's initial events.
    ///
    /// # Panics
    ///
    /// If `at` is earlier than now.
    pub fn schedule(&mut self, at: MonoTime, event: M::Event) {
        assert!(
            at >= self.now(),
            "an event scheduled in the past: {at:?} at {:?}",
            self.now()
        );
        self.queue.push(at, event, &self.entropy);
    }

    /// Sets a milestone at `at`: once every event up to `at` has run, every checker's liveness
    /// check runs with `label`. A scenario sets one at its end.
    pub fn milestone(&mut self, at: MonoTime, label: impl Into<String>) {
        self.milestones
            .insert((at, self.next_milestone), label.into());
        self.next_milestone += 1;
    }

    pub fn add_checker(&mut self, checker: impl Checker<M> + 'static) {
        self.checkers.push(Box::new(checker));
    }

    /// Runs every event and milestone up to `until`, then leaves the clock at `until`. Stops
    /// early, for good, on a failed check.
    ///
    /// # Panics
    ///
    /// If `until` is earlier than now.
    pub fn run_until(&mut self, until: MonoTime) -> RunOutcome {
        if let Some(violation) = &self.stopped {
            return RunOutcome::Violated(violation.clone());
        }
        assert!(
            until >= self.now(),
            "run_until cannot go back: {until:?} at {:?}",
            self.now()
        );
        loop {
            let next_event = self.queue.next_at().filter(|at| *at <= until);
            let next_milestone = self
                .milestones
                .keys()
                .next()
                .map(|(at, _)| *at)
                .filter(|at| *at <= until);
            // A milestone runs once every event at or before its instant has.
            let milestone_first = match (next_milestone, next_event) {
                (Some(milestone), Some(event)) => milestone < event,
                (Some(_), None) => true,
                (None, _) => false,
            };
            if milestone_first {
                let ((at, _), label) = self.milestones.pop_first().expect("a milestone was peeked");
                self.clock.advance_to(at);
                if let Some(violation) = self.check_milestone(at, &label) {
                    return self.stop(violation);
                }
            } else if next_event.is_some() {
                let Reverse(scheduled) = self.queue.heap.pop().expect("an event was peeked");
                if let Some(violation) = self.process(scheduled) {
                    return self.stop(violation);
                }
            } else {
                break;
            }
        }
        self.clock.advance_to(until);
        RunOutcome::Reached {
            at: until,
            events: self.events(),
        }
    }

    /// Processes one event: the clock, the trace, the model, then the safety checks.
    fn process(&mut self, scheduled: Scheduled<M::Event>) -> Option<Violation> {
        let Scheduled { at, event, .. } = scheduled;
        self.clock.advance_to(at);
        let index = self.trace.events();
        self.trace.record(at, &event);
        let mut schedule = Schedule {
            now: at,
            queue: &mut self.queue,
            entropy: &self.entropy,
        };
        self.model.handle(at, event, &mut schedule);
        for checker in &mut self.checkers {
            if let Err(message) = checker.after_event(&self.model, at) {
                return Some(Violation {
                    checker: checker.name().to_owned(),
                    message,
                    seed: self.seed,
                    found: Found::AfterEvent { index },
                    at,
                    events: index + 1,
                    trace_prefix: self.trace.digest(),
                });
            }
        }
        None
    }

    fn check_milestone(&mut self, at: MonoTime, label: &str) -> Option<Violation> {
        for checker in &mut self.checkers {
            if let Err(message) = checker.at_milestone(&self.model, at, label) {
                return Some(Violation {
                    checker: checker.name().to_owned(),
                    message,
                    seed: self.seed,
                    found: Found::AtMilestone {
                        label: label.to_owned(),
                    },
                    at,
                    events: self.trace.events(),
                    trace_prefix: self.trace.digest(),
                });
            }
        }
        None
    }

    fn stop(&mut self, violation: Violation) -> RunOutcome {
        self.stopped = Some(violation.clone());
        RunOutcome::Violated(violation)
    }
}

impl<M: Model> fmt::Debug for SimWorld<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimWorld")
            .field("seed", &self.seed)
            .field("now", &self.now())
            .field("events", &self.events())
            .field("pending", &self.pending())
            .field("checkers", &self.checkers.len())
            .field("stopped", &self.stopped.is_some())
            .finish_non_exhaustive()
    }
}
