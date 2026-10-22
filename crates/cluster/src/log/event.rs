// SPDX-License-Identifier: MIT OR Apache-2.0

//! The base events of the cluster event log (RFC-0001 §5.10, RFC-0032 §6.5).

use serde::{Deserialize, Serialize};
use syntropy_core::SessionId;
use syntropy_core::resource::GraphDelta;
use syntropy_core::{GraphVersion, InstanceId, InstancePolicy, ModelId, NodeId, PlanEpoch};

use super::payload::{
    ChunkSetDigest, ConfigPatch, ManifestDigest, NodeInfo, PlanBody, PlanSummary, RunnerStateKind,
    UnroutableReason, WorkloadReport,
};

/// A fact in the event log, appended only by the coordinator (RFC-0001 §5.10). Every node folds
/// it into its [`ClusterState`](super::ClusterState) with [`apply`](super::apply).
///
/// These are the base variants RFC-0001 §5.10 lists, but `ChunkIdsCommitted`: its evidence is
/// a signed record (DIST-E9-T16), and COMP-E12-T06 defines it. The rest of RFC-0032 §6.5's
/// catalogue joins through the owner hooks (DIST-E9-T15, DIST-E9-T03).
///
/// The CBOR form is serde's default for an enum: a one-entry map from the variant's name to its
/// fields. A decoder ignores fields it does not know (ADR-0305 item 3), so an event a later
/// minor extends still decodes, and the chain is computed over its bytes as received.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClusterEvent {
    /// A node joined; `info` carries its description.
    NodeAdmitted {
        /// The node.
        info: NodeInfo,
    },
    /// Gossip suspects the node. Informational only.
    NodeSuspected {
        /// The node.
        node: NodeId,
    },
    /// The node's failure is confirmed; its instances are re-placed.
    NodeFailed {
        /// The node.
        node: NodeId,
    },
    /// The node left the cluster.
    NodeLeft {
        /// The node.
        node: NodeId,
    },
    /// The node went to sleep: no suspicion, no failure.
    NodeSleeping {
        /// The node.
        node: NodeId,
        /// Its incarnation when it went to sleep.
        incarnation: u64,
    },
    /// The node is restarting: suspicion and recovery are held until `back_by_ms`.
    NodeRestarting {
        /// The node.
        node: NodeId,
        /// Coordinator wall-clock time by which it is expected back.
        back_by_ms: u64,
    },
    /// The node's admission was revoked.
    NodeRevoked {
        /// The node.
        node: NodeId,
    },
    /// The node's mutable description changed.
    NodeInfoChanged {
        /// The node.
        node: NodeId,
        /// Its new description.
        info: NodeInfo,
    },
    /// A coordinator was elected for `term`.
    CoordinatorElected {
        /// The coordinator.
        node: NodeId,
        /// Its lease term.
        term: u64,
        /// Coordinator wall-clock time at which the lease ends unless renewed.
        lease_until_ms: u64,
    },
    /// The resource graph changed.
    GraphUpdated {
        /// The graph's new version.
        version: GraphVersion,
        /// What changed.
        delta: GraphDelta,
    },
    /// A model was registered.
    ModelRegistered {
        /// The model.
        model: ModelId,
        /// Its manifest's digest; the body travels by digest.
        manifest: ManifestDigest,
    },
    /// An instance was requested; its reaction runs the solver.
    InstanceRequested {
        /// The new instance.
        instance: InstanceId,
        /// The model it serves.
        model: ModelId,
        /// Its policy, boxed because it is far larger than any other event's fields; the box
        /// does not change the encoding.
        policy: Box<InstancePolicy>,
    },
    /// A plan was committed for an instance.
    PlanCommitted {
        /// The plan, inline or by digest.
        plan: PlanBody,
        /// Its summary, for logs and the dashboard.
        summary: PlanSummary,
    },
    /// A runner's state changed.
    RunnerStateChanged {
        /// The runner's instance.
        instance: InstanceId,
        /// Its node.
        node: NodeId,
        /// The plan epoch it runs.
        epoch: PlanEpoch,
        /// Its new state.
        state: RunnerStateKind,
    },
    /// Every runner of the plan is ready and the canary passed.
    InstanceRoutable {
        /// The instance.
        instance: InstanceId,
        /// The routable plan's epoch.
        epoch: PlanEpoch,
        /// The plan's published KV reach, which ingress checks before routing.
        context_length: u32,
    },
    /// The instance stopped being routable.
    InstanceUnroutable {
        /// The instance.
        instance: InstanceId,
        /// Why.
        reason: UnroutableReason,
    },
    /// A retiring plan's runners have all stopped.
    PlanRetired {
        /// The instance.
        instance: InstanceId,
        /// The retired plan's epoch.
        epoch: PlanEpoch,
    },
    /// The instance is gone.
    InstanceRemoved {
        /// The instance.
        instance: InstanceId,
    },
    /// An instance's latest workload report.
    WorkloadUpdated {
        /// The model.
        model: ModelId,
        /// The instance.
        instance: InstanceId,
        /// The report, which replaces the instance's previous one.
        report: WorkloadReport,
    },
    /// A node announced the chunks it holds for a model.
    ChunksAnnounced {
        /// The node.
        node: NodeId,
        /// The model.
        model: ModelId,
        /// The digest of its chunk set.
        have: ChunkSetDigest,
    },
    /// A persisted session was pinned to an instance.
    SessionPinned {
        /// The session.
        session: SessionId,
        /// The instance it is pinned to.
        instance: InstanceId,
        /// The nodes holding its KV.
        holders: Vec<NodeId>,
    },
    /// A persisted session was dropped.
    SessionDropped {
        /// The session.
        session: SessionId,
    },
    /// The cluster-scoped configuration changed.
    ClusterConfigChanged {
        /// The change.
        patch: ConfigPatch,
    },
}
