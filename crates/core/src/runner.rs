// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner lifecycle's plain data that the daemon↔runner command channel carries (RFC-0001
//! §3.5, §5.11): [`Deadline`] and [`DrainReason`] (DIST-E1-T15).
//!
//! `ipc` holds the message enums and may name only layer-0 types, so the types the messages carry
//! live here (RFC-0001 §3.5, note of 2026-10-23; §18.28). DIST-E1-T05 adds `RunnerState` and its
//! other field types beside them.
//!
//! Over the command channel these are postcard-encoded, where an enum's variant is its index, so
//! a variant is only ever appended, and each change bumps `ipc_protocol` (RFC-0001 §5, note of
//! 2026-10-22).

use serde::{Deserialize, Serialize};

use crate::clock::MonoTime;
use crate::plan::PlanEpoch;
use crate::units::Nanos;

/// How long a drain or a shutdown may take, counted from when its message is received (RFC-0001
/// §3.5, §18.28). It holds a duration, never an instant: an instant read on one machine's clock
/// means nothing on another's, and RFC-0011 §13's `RouteRequest` carries `Option<Deadline>`
/// between nodes. `RunnerState::Draining` carries one too. Owner: RFC-0001.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Deadline {
    /// The time allowed, from receipt.
    pub within: Nanos,
}

impl Deadline {
    /// A deadline `within` from receipt.
    #[must_use]
    pub const fn within(within: Nanos) -> Deadline {
        Deadline { within }
    }

    /// When it expires on the receiver's clock, given when the receiver got the message. It
    /// saturates at the clock's end.
    #[must_use]
    pub fn expires_at(self, received: MonoTime) -> MonoTime {
        received.saturating_add(self.within)
    }
}

/// Why a runner drains (RFC-0001 §5.11). A re-plan's successor epoch travels only in
/// [`DrainReason::Replan`]: the `Drain` message has no field of its own for it (§18.28). Owner:
/// RFC-0001.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrainReason {
    /// A new plan replaces the runner's: `successor` is its epoch.
    Replan {
        /// The epoch that replaces the draining one.
        successor: PlanEpoch,
    },
    /// The instance is unloaded.
    Unload,
    /// The node shuts down.
    Shutdown,
    /// The node leaves the cluster.
    NodeLeaving,
    /// The runner spent its restart budget (RFC-0001 §3.4; RFC-0033 §8.2).
    RestartBudgetSpent,
}
