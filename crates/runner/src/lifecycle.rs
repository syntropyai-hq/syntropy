// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner lifecycle machine of RFC-0001 §3.3 (DIST-E1-T05).
//!
//! [`Lifecycle`] is a pure state machine. Its caller, the runner's control loop, turns what it
//! sees into [`Event`]s: a verified shard, a committed pack, a `Drain` from the daemon, a fault.
//! [`Lifecycle::apply`] either moves to a new [`RunnerState`] and returns the one
//! [`RunnerMsg::StateChanged`] the caller sends, or updates the state in place and returns
//! nothing, or refuses the event and leaves the machine as it was.
//!
//! - **Which edges exist** is [`RunnerStateKind::can_become`], in `core`. This machine never takes
//!   an edge that table lacks, and DIST-E1-T06's supervisor checks the reports it receives against
//!   the same table.
//! - **One message per transition.** A transition is a change of state kind, or the
//!   `Ready → Ready` edge that adopts a plan delta under a new epoch. Each one returns exactly
//!   one `StateChanged`, which carries the new state and the epoch. Progress inside a state (bytes
//!   fetched, a warm step started, the live sequence count) changes the state's fields and sends
//!   nothing: byte-level progress travels as RFC-0050 §4.3's `LoadProgress`, never as a transition.
//! - **No clock.** The machine reads no time (RFC-0060 §2.1, rule T1). A drain's deadline is the
//!   caller's to watch; it reports [`Event::DeadlineReached`] when the deadline passes.
//! - **The restart budget is the supervisor's** (RFC-0001 §3.4, DIST-E1-T06). The machine counts
//!   restarts and reports the count in `Failed`, and takes `Failed → Stopped` only when the
//!   supervisor gives up ([`Event::GiveUp`]).

use std::fmt;

use syntropy_core::{
    Bytes, Deadline, Degradation, DrainReason, ErrorClass, PlanEpoch, RunnerState, RunnerStateKind,
    StopReason, WarmStep,
};
use syntropy_ipc::cmd::RunnerMsg;

/// What the runner's control loop saw. Each event belongs to one state or a set of them; in any
/// other state the machine refuses it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// `Fetching`: `have` bytes of the shard verified so far. In place.
    FetchProgress {
        /// Bytes verified, at most the shard's size.
        have: Bytes,
    },
    /// `Fetching → Packing`: every chunk of the shard is present and verified. `pack` is the
    /// size of the pack to build or validate.
    ShardVerified {
        /// The pack's bytes.
        pack: Bytes,
    },
    /// `Packing`: `done` bytes packed so far. In place.
    PackProgress {
        /// Bytes packed, at most the pack's size.
        done: Bytes,
    },
    /// `Packing → Loading`: the sidecar is committed, or a valid pack was found. `resident` is
    /// the bytes `Loading` makes resident.
    PackCommitted {
        /// The bytes to make resident.
        resident: Bytes,
    },
    /// `Loading`: `done` bytes resident so far. In place.
    LoadProgress {
        /// Bytes resident, at most the total.
        done: Bytes,
    },
    /// `Loading → Warming`: the home tiers are resident and the ring channels open.
    Resident,
    /// `Warming`: a later warm step started. Steps go in [`WarmStep`]'s order and may be
    /// skipped (M0 skips `Canary`, RFC-0060 §5.8), never repeated or taken back. In place.
    WarmStep(WarmStep),
    /// `Warming → Ready`: the warm checks and the instance canary passed.
    WarmPassed,
    /// `Ready` or `Draining`: the sequences live now. In place, except that in `Draining` a
    /// count of zero ends the drain: `Draining → Stopped`.
    Sequences {
        /// Live sequences.
        live: u32,
    },
    /// `Ready`: how the runner is degraded now (RFC-0033 §7.1). In place.
    Degraded(Option<Degradation>),
    /// `Ready → Ready`: `ApplyPlanDelta` adopted in place, under a later epoch.
    PlanDelta {
        /// The delta's epoch, later than the machine's.
        epoch: PlanEpoch,
    },
    /// `Ready → Draining`: the daemon's `Drain`, or its `Shutdown` as a drain for
    /// [`DrainReason::Shutdown`]. A drain always passes through `Draining`, even with no live
    /// sequence: the caller then reports [`Event::Sequences`] with zero.
    Drain {
        /// Why.
        reason: DrainReason,
        /// How long the drain may take.
        deadline: Deadline,
    },
    /// `Draining → Stopped`: the drain's deadline passed with sequences still live.
    DeadlineReached,
    /// Any working state `→ Failed`: the runner failed with an error of this class.
    Fault {
        /// What kind of failure.
        class: ErrorClass,
    },
    /// `Failed → Fetching`: the supervisor restarted the runner, in a new process.
    Restart,
    /// `Failed → Stopped`: the supervisor gave up.
    GiveUp(GiveUp),
}

impl Event {
    /// The event's name, without its fields, as a refusal reports it.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Event::FetchProgress { .. } => "FetchProgress",
            Event::ShardVerified { .. } => "ShardVerified",
            Event::PackProgress { .. } => "PackProgress",
            Event::PackCommitted { .. } => "PackCommitted",
            Event::LoadProgress { .. } => "LoadProgress",
            Event::Resident => "Resident",
            Event::WarmStep(_) => "WarmStep",
            Event::WarmPassed => "WarmPassed",
            Event::Sequences { .. } => "Sequences",
            Event::Degraded(_) => "Degraded",
            Event::PlanDelta { .. } => "PlanDelta",
            Event::Drain { .. } => "Drain",
            Event::DeadlineReached => "DeadlineReached",
            Event::Fault { .. } => "Fault",
            Event::Restart => "Restart",
            Event::GiveUp(_) => "GiveUp",
        }
    }
}

/// Why the supervisor gave up on a failed runner (RFC-0001 §3.3, §18.34).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GiveUp {
    /// The restart budget is spent: `Stopped { Drained(RestartBudgetSpent) }`.
    BudgetSpent,
    /// The error is one no restart can fix: `Stopped { Fatal(class) }`, with the last
    /// failure's class.
    Fatal,
}

/// An event the machine refused. The machine is unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The event does not apply in this state.
    WrongState {
        /// The state the machine is in.
        state: RunnerStateKind,
        /// The event's name.
        event: &'static str,
    },
    /// Progress past the total: `have > need`, or `done > total`.
    PastTotal {
        /// The progress reported.
        progress: Bytes,
        /// The total it is progress towards.
        total: Bytes,
    },
    /// A warm step at or before the one under way.
    WarmStepBack {
        /// The step under way.
        current: WarmStep,
        /// The step reported.
        reported: WarmStep,
    },
    /// A plan delta whose epoch is not later than the machine's.
    StaleEpoch {
        /// The machine's epoch.
        current: PlanEpoch,
        /// The delta's.
        reported: PlanEpoch,
    },
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::WrongState { state, event } => {
                write!(f, "{event} does not apply in {state:?}")
            }
            Refused::PastTotal { progress, total } => {
                write!(f, "progress {} is past the total {}", progress.0, total.0)
            }
            Refused::WarmStepBack { current, reported } => {
                write!(f, "warm step {reported:?} is not after {current:?}")
            }
            Refused::StaleEpoch { current, reported } => {
                write!(f, "plan epoch {reported:?} is not after {current:?}")
            }
        }
    }
}

impl std::error::Error for Refused {}

/// The lifecycle of one runner: its state, the plan epoch it serves, and its restart count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lifecycle {
    state: RunnerState,
    epoch: PlanEpoch,
    /// The shard's bytes, which `Fetching` re-enters with after a restart.
    shard: Bytes,
    restarts: u32,
}

impl Lifecycle {
    /// A runner that received `Init` and `AssignPlan` for `epoch`, and starts fetching a shard of
    /// `shard` bytes: the `[*] → Fetching` edge. Returns the machine and the `StateChanged` that
    /// reports the edge.
    #[must_use]
    pub fn start(epoch: PlanEpoch, shard: Bytes) -> (Lifecycle, RunnerMsg) {
        let machine = Lifecycle {
            state: RunnerState::Fetching {
                have: Bytes(0),
                need: shard,
            },
            epoch,
            shard,
            restarts: 0,
        };
        let msg = machine.state_changed();
        (machine, msg)
    }

    /// The state now.
    #[must_use]
    pub const fn state(&self) -> &RunnerState {
        &self.state
    }

    /// The plan epoch the runner serves.
    #[must_use]
    pub const fn epoch(&self) -> PlanEpoch {
        self.epoch
    }

    /// Restarts so far.
    #[must_use]
    pub const fn restarts(&self) -> u32 {
        self.restarts
    }

    /// Applies `event`. Returns the `StateChanged` to send when the event is a transition, and
    /// `None` when it updates the state in place. A refused event changes nothing.
    ///
    /// # Errors
    ///
    /// [`Refused`] when the event does not apply in this state, or its value is out of order.
    pub fn apply(&mut self, event: Event) -> Result<Option<RunnerMsg>, Refused> {
        let from = self.state.kind();
        let refuse = |event: &Event| Refused::WrongState {
            state: from,
            event: event.name(),
        };
        let next = match (&mut self.state, event) {
            (RunnerState::Fetching { have, need }, Event::FetchProgress { have: now }) => {
                in_place(have, now, *need)?;
                None
            }
            (RunnerState::Fetching { .. }, Event::ShardVerified { pack }) => {
                Some(RunnerState::Packing {
                    done: Bytes(0),
                    total: pack,
                })
            }
            (RunnerState::Packing { done, total }, Event::PackProgress { done: now }) => {
                in_place(done, now, *total)?;
                None
            }
            (RunnerState::Packing { .. }, Event::PackCommitted { resident }) => {
                Some(RunnerState::Loading {
                    done: Bytes(0),
                    total: resident,
                })
            }
            (RunnerState::Loading { done, total }, Event::LoadProgress { done: now }) => {
                in_place(done, now, *total)?;
                None
            }
            (RunnerState::Loading { .. }, Event::Resident) => Some(RunnerState::Warming {
                step: WarmStep::Prepare,
            }),
            (RunnerState::Warming { step }, Event::WarmStep(reported)) => {
                if reported <= *step {
                    return Err(Refused::WarmStepBack {
                        current: *step,
                        reported,
                    });
                }
                *step = reported;
                None
            }
            (RunnerState::Warming { .. }, Event::WarmPassed) => Some(RunnerState::Ready {
                live_sequences: 0,
                degraded: None,
            }),
            (RunnerState::Ready { live_sequences, .. }, Event::Sequences { live }) => {
                *live_sequences = live;
                None
            }
            (RunnerState::Ready { degraded, .. }, Event::Degraded(now)) => {
                *degraded = now;
                None
            }
            (RunnerState::Ready { .. }, Event::PlanDelta { epoch }) => {
                if epoch <= self.epoch {
                    return Err(Refused::StaleEpoch {
                        current: self.epoch,
                        reported: epoch,
                    });
                }
                self.epoch = epoch;
                return Ok(Some(self.state_changed()));
            }
            (RunnerState::Ready { live_sequences, .. }, Event::Drain { reason, deadline }) => {
                Some(RunnerState::Draining {
                    reason,
                    deadline,
                    live_sequences: *live_sequences,
                })
            }
            (RunnerState::Draining { reason, .. }, Event::Sequences { live: 0 }) => {
                Some(RunnerState::Stopped {
                    reason: StopReason::Drained(*reason),
                })
            }
            (RunnerState::Draining { live_sequences, .. }, Event::Sequences { live }) => {
                *live_sequences = live;
                None
            }
            (RunnerState::Draining { reason, .. }, Event::DeadlineReached) => {
                Some(RunnerState::Stopped {
                    reason: StopReason::Drained(*reason),
                })
            }
            (
                RunnerState::Fetching { .. }
                | RunnerState::Packing { .. }
                | RunnerState::Loading { .. }
                | RunnerState::Warming { .. }
                | RunnerState::Ready { .. }
                | RunnerState::Draining { .. },
                Event::Fault { class },
            ) => Some(RunnerState::Failed {
                class,
                during: from,
                restarts: self.restarts,
            }),
            (RunnerState::Failed { .. }, Event::Restart) => {
                self.restarts = self.restarts.saturating_add(1);
                Some(RunnerState::Fetching {
                    have: Bytes(0),
                    need: self.shard,
                })
            }
            (RunnerState::Failed { class, .. }, Event::GiveUp(why)) => Some(RunnerState::Stopped {
                reason: match why {
                    GiveUp::BudgetSpent => StopReason::Drained(DrainReason::RestartBudgetSpent),
                    GiveUp::Fatal => StopReason::Fatal(*class),
                },
            }),
            (_, event) => return Err(refuse(&event)),
        };
        Ok(next.map(|next| {
            debug_assert!(
                from.can_become(next.kind()),
                "{from:?} -> {:?}",
                next.kind()
            );
            self.state = next;
            self.state_changed()
        }))
    }

    /// The message that reports the state now.
    fn state_changed(&self) -> RunnerMsg {
        RunnerMsg::StateChanged {
            state: self.state.clone(),
            epoch: self.epoch,
        }
    }
}

/// Moves a progress counter forward in place, refusing a value past `total`.
fn in_place(counter: &mut Bytes, now: Bytes, total: Bytes) -> Result<(), Refused> {
    if now > total {
        return Err(Refused::PastTotal {
            progress: now,
            total,
        });
    }
    *counter = now;
    Ok(())
}

#[cfg(test)]
mod tests;
