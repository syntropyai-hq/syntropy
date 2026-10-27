// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner lifecycle's plain data that the daemon↔runner command channel carries (RFC-0001
//! §3.5, §5.11): [`Deadline`] and [`DrainReason`] (DIST-E1-T15); [`RunnerState`] and the types it
//! holds, [`RunnerStateKind`], [`WarmStep`], [`StopReason`], and [`Degradation`] with
//! [`DegradeFlags`] and [`DegradeDetail`] (RFC-0033 §7.1; DIST-E1-T05).
//!
//! `ipc` holds the message enums and may name only layer-0 types, so the types the messages carry
//! live here (RFC-0001 §3.5, note of 2026-10-23; §18.28).
//!
//! [`RunnerStateKind::can_become`] is RFC-0001 §3.3's transition table as one pure lookup, so the
//! runner's lifecycle machine (`syntropy-runner`) and the daemon's supervisor (DIST-E1-T06), which
//! sit side by side in layer 5 and so cannot share code above `core`, read the same edges.
//!
//! Over the command channel these are postcard-encoded, where an enum's variant is its index, so
//! a variant is only ever appended, and each change bumps `ipc_protocol` (RFC-0001 §5, note of
//! 2026-10-22).

use serde::{Deserialize, Serialize};

use bitflags::bitflags;
use smallvec::SmallVec;

use crate::clock::MonoTime;
use crate::error::ErrorClass;
use crate::ids::{DeviceId, LinkId, StageIx};
use crate::plan::PlanEpoch;
use crate::resource::{TierKind, TransportKind};
use crate::units::{Bytes, Nanos};

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

/// A runner's lifecycle state with its progress (RFC-0001 §3.3, §5.11). `StateChanged` carries it
/// on every transition. Owner: RFC-0001 (states, transitions), RFC-0033 (restart policy).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunnerState {
    /// Fetching the shard from the store: `have` of `need` bytes verified.
    Fetching {
        /// Bytes verified so far.
        have: Bytes,
        /// The shard's bytes.
        need: Bytes,
    },
    /// Building or validating the per-device pack.
    Packing {
        /// Bytes packed so far.
        done: Bytes,
        /// The pack's bytes.
        total: Bytes,
    },
    /// Making the home tiers resident and opening the ring channels.
    Loading {
        /// Bytes resident so far.
        done: Bytes,
        /// The bytes to make resident.
        total: Bytes,
    },
    /// Preparing ops, capturing graphs, filling caches, probing links and running the canary.
    Warming {
        /// The step under way.
        step: WarmStep,
    },
    /// Serving sequences.
    Ready {
        /// Sequences live now.
        live_sequences: u32,
        /// How the runner is degraded, if it is (RFC-0033 §7.1).
        degraded: Option<Degradation>,
    },
    /// Admitting nothing new; finishing or migrating live sequences.
    Draining {
        /// Why.
        reason: DrainReason,
        /// How long the drain may take, from the `Drain` message (§18.28: `deadline_ms` before).
        deadline: Deadline,
        /// Sequences still live.
        live_sequences: u32,
    },
    /// The process exited or was killed; the supervisor restarts it or gives up.
    Failed {
        /// What kind of failure.
        class: ErrorClass,
        /// The state it failed in.
        during: RunnerStateKind,
        /// Restarts so far in the budget's window.
        restarts: u32,
    },
    /// Terminal.
    Stopped {
        /// Why.
        reason: StopReason,
    },
}

impl RunnerState {
    /// The state without its fields.
    #[must_use]
    pub const fn kind(&self) -> RunnerStateKind {
        match self {
            RunnerState::Fetching { .. } => RunnerStateKind::Fetching,
            RunnerState::Packing { .. } => RunnerStateKind::Packing,
            RunnerState::Loading { .. } => RunnerStateKind::Loading,
            RunnerState::Warming { .. } => RunnerStateKind::Warming,
            RunnerState::Ready { .. } => RunnerStateKind::Ready,
            RunnerState::Draining { .. } => RunnerStateKind::Draining,
            RunnerState::Failed { .. } => RunnerStateKind::Failed,
            RunnerState::Stopped { .. } => RunnerStateKind::Stopped,
        }
    }
}

/// A [`RunnerState`] without its fields, as the event log records it (RFC-0001 §5.11). Moved here
/// from `cluster`'s log payload, which re-exports it; the variants keep their order, so the log's
/// encoding is unchanged. Owner: RFC-0001.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RunnerStateKind {
    /// Fetching shards.
    Fetching,
    /// Packing shards into packs.
    Packing,
    /// Loading packs onto devices.
    Loading,
    /// Warming: prepare, capture, cache fill, link probe, canary.
    Warming,
    /// Serving.
    Ready,
    /// Finishing its live sequences before it stops.
    Draining,
    /// Failed.
    Failed,
    /// Stopped.
    Stopped,
}

impl RunnerStateKind {
    /// Every kind, in order.
    pub const ALL: [RunnerStateKind; 8] = [
        RunnerStateKind::Fetching,
        RunnerStateKind::Packing,
        RunnerStateKind::Loading,
        RunnerStateKind::Warming,
        RunnerStateKind::Ready,
        RunnerStateKind::Draining,
        RunnerStateKind::Failed,
        RunnerStateKind::Stopped,
    ];

    /// The kind as the shared-memory header's `RunnerState` word carries it (RFC-0001 §3.5): its
    /// position in [`RunnerStateKind::ALL`], from 0.
    #[must_use]
    pub const fn discriminant(self) -> u32 {
        self as u32
    }

    /// The kind a header word names, if it names one.
    #[must_use]
    pub const fn from_discriminant(word: u32) -> Option<RunnerStateKind> {
        if (word as usize) < RunnerStateKind::ALL.len() {
            Some(RunnerStateKind::ALL[word as usize])
        } else {
            None
        }
    }

    /// Whether RFC-0001 §3.3 has an edge from `self` to `next`: the forward path `Fetching →
    /// Packing → Loading → Warming → Ready → Draining → Stopped`, `Ready → Ready` for
    /// `ApplyPlanDelta`, every working state to `Failed`, and `Failed` to `Fetching` (the
    /// supervisor's restart, in a new process) or to `Stopped`. `Stopped` has no way out.
    ///
    /// `Ready → Ready` is an edge whatever the epochs. §3.3 takes it only for a plan delta under a
    /// later epoch, which is the caller's to check: `StateChanged` carries the epoch.
    #[must_use]
    pub const fn can_become(self, next: RunnerStateKind) -> bool {
        use RunnerStateKind::{
            Draining, Failed, Fetching, Loading, Packing, Ready, Stopped, Warming,
        };
        matches!(
            (self, next),
            (Fetching, Packing)
                | (Packing, Loading)
                | (Loading, Warming)
                | (Warming, Ready)
                | (Ready, Ready)
                | (Ready, Draining)
                | (Draining, Stopped)
                | (
                    Fetching | Packing | Loading | Warming | Ready | Draining,
                    Failed
                )
                | (Failed, Fetching)
                | (Failed, Stopped)
        )
    }
}

/// The steps of `Warming` (RFC-0001 §3.3, §5.11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WarmStep {
    /// Preparing ops and the final selection pass.
    Prepare,
    /// Capturing step graphs per verify-window size.
    Capture,
    /// Filling the expert cache from the persisted frequency profile.
    CacheFill,
    /// Measuring α_eff through the real GPU→wire→GPU path.
    LinkProbe,
    /// The instance canary (from M1; reported `not-run` in M0, RFC-0060 §5.8).
    Canary,
}

/// Why a runner stopped (RFC-0001 §5.11, §18.28): after a drain, or from `Failed`. From `Failed`,
/// a spent restart budget is `Drained(DrainReason::RestartBudgetSpent)` and a fatal error
/// `Fatal(class)`, so the supervisor and the event log name the same reason (§18.34). It is the
/// runner's reason, not the API's completion reason. Owner: RFC-0001.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StopReason {
    /// After a drain, for this reason; or the restart budget was spent.
    Drained(DrainReason),
    /// An error of this class that no restart can fix.
    Fatal(ErrorClass),
}

/// How a `Ready` runner is degraded: a set, since it may be degraded in several ways at once
/// (RFC-0033 §7.1). Every flag is visible in the dashboard and in response metadata (RFC-0001
/// §10.2). None changes output (D3). Owner: RFC-0033.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Degradation {
    /// Which ways.
    pub flags: DegradeFlags,
    /// The detail of those that have one.
    pub detail: SmallVec<[DegradeDetail; 2]>,
}

bitflags! {
    /// The ways a runner may be degraded (RFC-0033 §7.1), with RFC-0033's bit positions.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct DegradeFlags: u32 {
        /// Switching neighbours in place (RFC-0032 §4.4).
        const RELINKING = 1 << 0;
        /// Replaying KV for some sequences (RFC-0033 §4.2).
        const RECOVERING = 1 << 1;
        /// Hosting a lost stage's layers in slower tiers (RFC-0033 §7.2).
        const ABSORBED_STAGE = 1 << 2;
        /// An edge runs on a lower-ranked transport (RFC-0031 §7.4).
        const TRANSPORT_DEMOTED = 1 << 3;
        /// An op fell back to the ggml provider (RFC-0040).
        const PROVIDER_FALLBACK = 1 << 4;
        /// Speculative decoding is off.
        const SPECULATION_OFF = 1 << 5;
        /// Graph capture is unavailable; the executor replays launches.
        const CAPTURE_OFF = 1 << 6;
        /// The expert cache or KV hot window shrank under pressure (RFC-0020).
        const CACHE_REDUCED = 1 << 7;
        /// Some history was dropped under pressure (RFC-0033 §4.1.2).
        const HISTORY_PARTIAL = 1 << 8;
        /// History is disabled for this instance (RFC-0033 §4.1.2).
        const HISTORY_OFF = 1 << 9;
    }
}

/// The detail of one degradation (RFC-0033 §7.1). `Fallback { op, provider }` names a
/// `kernels` type, so it arrives with DIST-E14-T13, appended after these: postcard numbers
/// variants by their order. Owner: RFC-0033.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradeDetail {
    /// Stages absorbed into this runner, and the tiers their layers live in.
    Absorbed {
        /// The absorbed stages.
        stages: SmallVec<[StageIx; 2]>,
        /// The tiers that hold their layers.
        tiers: SmallVec<[TierKind; 2]>,
    },
    /// An edge moved to a lower-ranked transport.
    Demoted {
        /// The edge.
        edge: LinkId,
        /// The transport it ran on.
        from: TransportKind,
        /// The transport it runs on now.
        to: TransportKind,
    },
    /// A device's cache gave back bytes under pressure.
    Reduced {
        /// The device.
        device: DeviceId,
        /// The bytes given back.
        bytes: Bytes,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC-0001 §3.3's edges, and none other.
    #[test]
    fn the_transition_table_is_section_3_3s() {
        use RunnerStateKind::*;
        let edges = [
            (Fetching, Packing),
            (Packing, Loading),
            (Loading, Warming),
            (Warming, Ready),
            (Ready, Ready),
            (Ready, Draining),
            (Draining, Stopped),
            (Fetching, Failed),
            (Packing, Failed),
            (Loading, Failed),
            (Warming, Failed),
            (Ready, Failed),
            (Draining, Failed),
            (Failed, Fetching),
            (Failed, Stopped),
        ];
        for from in RunnerStateKind::ALL {
            for to in RunnerStateKind::ALL {
                assert_eq!(
                    from.can_become(to),
                    edges.contains(&(from, to)),
                    "{from:?} → {to:?}"
                );
            }
        }
    }

    #[test]
    fn header_words_name_kinds_in_order() {
        for (at, kind) in RunnerStateKind::ALL.into_iter().enumerate() {
            assert_eq!(kind.discriminant(), at as u32);
            assert_eq!(RunnerStateKind::from_discriminant(at as u32), Some(kind));
        }
        assert_eq!(RunnerStateKind::from_discriminant(8), None);
    }

    #[test]
    fn a_state_names_its_kind() {
        let state = RunnerState::Failed {
            class: ErrorClass::Device,
            during: RunnerStateKind::Loading,
            restarts: 2,
        };
        assert_eq!(state.kind(), RunnerStateKind::Failed);
        assert_eq!(
            RunnerState::Warming {
                step: WarmStep::Canary
            }
            .kind(),
            RunnerStateKind::Warming
        );
    }
}
