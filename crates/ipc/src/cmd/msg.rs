// SPDX-License-Identifier: MIT OR Apache-2.0

//! The command channel's messages (RFC-0001 §3.5): the ones whose field types exist at M0a.
//!
//! The tasks that own the other messages' field types add those messages (RFC-0001 §3.5, note of
//! 2026-10-23). Postcard writes a variant as its index and a struct's fields in order, with no
//! names, so an added variant or field changes the encoding: each such change bumps
//! [`PROTOCOL`](super::PROTOCOL). New variants go at the end.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::ledger::LedgerGrant;
use syntropy_core::{
    Blake3, Bytes, CancelReason, InstanceId, NodeId, PlacementPlan, PlanDelta, PlanEpoch,
    SequenceId, StageIx,
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
    /// A lifecycle transition. The daemon forwards it as `ClusterCommand::ReportRunner`.
    StateChanged {
        /// The new state, as its `RunnerState` discriminant: the type is DIST-E1-T05's, and
        /// the shared-memory header carries the same `u32`.
        state: u32,
        /// The epoch of the plan the runner serves.
        epoch: PlanEpoch,
    },
    /// The answer to `Ping`.
    Pong,
}

/// The runner's handshake (RFC-0001 §3.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    /// The runner's process ID.
    pub pid: u32,
    /// The runner's build ID. `BuildId` is QUAL-E16-T03's; the channel carries its 32-byte
    /// encoding, as the shared-memory header does.
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
