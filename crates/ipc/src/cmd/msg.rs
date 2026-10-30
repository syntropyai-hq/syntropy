// SPDX-License-Identifier: MIT OR Apache-2.0

//! The command channel's messages (RFC-0001 §3.5): the ones whose field types exist at M0a.
//! DIST-E1-T03 built the first set; DIST-E1-T15 appended `Drain`, `Shutdown`, `MemoryPressure`
//! and `SequenceEnded`, whose field types it put in `core` (protocol 1.1). COMP-E10-T11
//! appended `FetchShard` and `PackCommitted`, with `ShardSpec` and `PackKey` in `core`
//! (protocol 5.0). DIST-E1-T17 appended `ProbeResult`, with `CapabilityProfile` in `core`
//! (protocol 6.0).
//!
//! The tasks that own the other messages' field types add those messages (RFC-0001 §3.5, note of
//! 2026-10-23). Postcard writes a variant as its index and a struct's fields in order, with no
//! names, so an added variant or field changes the encoding: each such change bumps
//! [`PROTOCOL`](super::PROTOCOL). New variants go at the end.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::ledger::LedgerGrant;
use syntropy_core::resource::CapabilityProfile;
use syntropy_core::runner::{Deadline, DrainReason};
use syntropy_core::sequence::SequenceOutcome;
use syntropy_core::{
    Blake3, Bytes, CancelReason, FetchPriority, InstanceId, NodeId, PackKey, PlacementPlan,
    PlanDelta, PlanEpoch, PressureLevel, RunnerState, SequenceId, ShardSpec, StageIx, TierId,
};

use crate::shm::{IpcProtocol, ShmLayout};

/// A message's number on the side that sent it. Each side numbers its messages from 1, and a
/// reply names the message it answers (RFC-0001 §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RequestId(pub u64);

/// One frame's contents: a message, its number, and the number of the message it answers.
///
/// The envelope's layout is frozen, as `Hello`'s is (see [`RunnerMsg::Hello`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Envelope<M> {
    /// The sender's number for this message.
    pub id: RequestId,
    /// The other side's number for the message this one answers, if it answers one.
    pub reply_to: Option<RequestId>,
    /// The message.
    pub msg: M,
}

/// Daemon to runner (RFC-0001 §3.5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DaemonMsg {
    /// The first message after the runner's `Hello`: it fixes the runner's budgets and identity.
    /// `config` (PROD-E1-T07) and `grant` (DIST-E7-T04) join it with their tasks.
    Init(Init),
    /// The plan and the stages of it this runner serves. It starts `Fetching`, or is checked
    /// against the current shard to choose between `ApplyPlanDelta` and drain-and-replace.
    AssignPlan {
        /// The committed plan, shared with the daemon's other users of it.
        #[serde(with = "shared")]
        plan: Arc<PlacementPlan>,
        /// This runner's stages, as indices into `plan.stages`.
        stages: SmallVec<[StageIx; 2]>,
    },
    /// The store's progress during `Fetching`.
    ShardProgress {
        /// The bytes present and verified.
        have: Bytes,
        /// The bytes the shard needs.
        need: Bytes,
        /// Whether every chunk is present and verified.
        complete: bool,
    },
    /// Budget, prefetch or speculative changes that leave the shard unchanged.
    ApplyPlanDelta {
        /// The epoch of the plan the delta produces.
        epoch: PlanEpoch,
        /// The changes.
        delta: PlanDelta,
    },
    /// A local cancel. Remote cancels arrive on the data plane.
    CancelSequence {
        /// The sequence.
        seq: SequenceId,
        /// Why ingress cancels it (DIST-E1-T14's `CancelReason`).
        reason: CancelReason,
    },
    /// Close every channel to a revoked node within one heartbeat period (RFC-0034 §6.1).
    PeerRevoked {
        /// The revoked node.
        node: NodeId,
    },
    /// An IPC round-trip check. The runner answers with `Pong`.
    Ping,
    /// Stop admitting sequences; finish or migrate the ones in flight within `deadline`. A
    /// re-plan's successor epoch travels in `reason`, as `DrainReason::Replan` (RFC-0001 §18.28).
    Drain {
        /// Why the runner drains.
        reason: DrainReason,
        /// How long the drain may take, from this message's receipt.
        deadline: Deadline,
    },
    /// Drain, then exit, within `deadline`.
    Shutdown {
        /// How long the drain and exit may take, from this message's receipt.
        deadline: Deadline,
    },
    /// The OS or the user needs memory back (RFC-0001 §10.3).
    MemoryPressure {
        /// How hard the pressure is.
        level: PressureLevel,
        /// The bytes to give back.
        reclaim: Bytes,
        /// The tier the pressure is on: `None` is host RAM, `Some` one GPU's budget drop (a DXGI
        /// or NVML budget, RFC-0010).
        tier: Option<TierId>,
    },
}

/// Runner to daemon (RFC-0001 §3.5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RunnerMsg {
    /// The handshake, sent first.
    ///
    /// `Hello` is variant 0, and neither its fields nor the [`Envelope`] around it change without
    /// a new `ipc_protocol` major version. So a daemon of any build reads any runner's `Hello`,
    /// and refuses a mismatch by its build ID or protocol rather than by a decode error.
    Hello(Hello),
    /// A lifecycle transition (RFC-0001 §3.3). The daemon forwards it as
    /// `ClusterCommand::ReportRunner`. Since protocol 2.0 it carries the whole [`RunnerState`];
    /// the shared-memory header keeps the kind's `u32` ([`RunnerStateKind::discriminant`]).
    ///
    /// [`RunnerStateKind::discriminant`]: syntropy_core::RunnerStateKind::discriminant
    StateChanged {
        /// The new state.
        state: RunnerState,
        /// The epoch of the plan the runner serves.
        epoch: PlanEpoch,
    },
    /// The answer to `Ping`.
    Pong,
    /// A sequence finished, failed or migrated.
    SequenceEnded {
        /// The sequence.
        seq: SequenceId,
        /// How it ended.
        outcome: SequenceOutcome,
    },
    /// Ask the store for a shard (RFC-0041 §7). The daemon hands it to the store's fetch planner
    /// (COMP-E12-T04), which answers with `ShardProgress`. Added with protocol 5.0
    /// (COMP-E10-T11).
    FetchShard {
        /// The shard: the node's groups and the chunks covering them.
        spec: ShardSpec,
        /// How urgently to fetch it, ahead of the node's other fetches.
        priority: FetchPriority,
    },
    /// A pack is committed (RFC-0041 §5.2's commit protocol); the store records it for garbage
    /// collection and for serving its extents to peers. Added with protocol 5.0 (COMP-E10-T11).
    PackCommitted {
        /// The pack.
        key: PackKey,
        /// BLAKE3 of the sidecar file's bytes.
        sidecar: Blake3,
    },
    /// A probe's measurements (RFC-0001 §3.5). A probe-mode process sends it once, after
    /// `Hello`, and exits (DIST-E1-T17). The probe leaves `profile.node` and `profile.version`
    /// for the daemon to stamp: only the daemon knows its node and its profile's version. Added
    /// with protocol 6.0.
    ProbeResult {
        /// What the probe measured.
        profile: CapabilityProfile,
    },
}

/// The runner's handshake (RFC-0001 §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    /// The runner's process ID.
    pub pid: u32,
    /// The runner's build ID: the 32-byte BLAKE3 digest of its `BuildId` and features, which
    /// `cli` computes (`syntropy_cli::build_id::id`, RFC-0062 §8.1). The shared-memory header
    /// carries the same bytes.
    pub build_id: [u8; 32],
    /// The protocol the runner speaks.
    pub ipc_protocol: IpcProtocol,
    /// The digest of the runner's log-callsite, metric and span-name tables (RFC-0053 §4.1).
    pub obs_tables: Blake3,
}

/// The daemon's first message (RFC-0001 §3.5), without `config` and `grant` until their tasks
/// add them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Init {
    /// The daemon's build ID. A runner of another build refuses to pair.
    pub build_id: [u8; 32],
    /// The model instance the runner serves.
    pub instance: InstanceId,
    /// The bytes per tier the daemon grants the runner (RFC-0020 §3).
    pub ledger: LedgerGrant,
    /// The shared-memory region's rings.
    pub shm: ShmLayout,
}

/// Serde for an `Arc<T>` as its `T`, without serde's `rc` feature.
mod shared {
    use std::sync::Arc;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<T: Serialize, S: Serializer>(
        value: &Arc<T>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        T::serialize(value, serializer)
    }

    pub(super) fn deserialize<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<T>, D::Error> {
        T::deserialize(deserializer).map(Arc::new)
    }
}
