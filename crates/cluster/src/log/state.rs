// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ClusterState`, the fold of the log, and `apply` (RFC-0032 §6.2, §6.3, §6.5).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use syntropy_core::SessionId;
use syntropy_core::policy::{ConfigValue, KeyPath};
use syntropy_core::{Blake3, GraphVersion, InstanceId, InstancePolicy, ModelId, NodeId, PlanEpoch};
use syntropy_net::ctl::envelope::{self, EncodeError};

use super::entry::{LogEntry, genesis};
use super::event::ClusterEvent;
use super::payload::{
    ChunkSetDigest, ConfigPatch, ManifestDigest, NodeInfo, PlanBody, PlanSummary, RunnerStateKind,
    WorkloadReport,
};

/// The whole cluster state: the fold of the log (RFC-0032 §6.2). Plain data, `Clone`, and
/// serialised deterministically by the envelope's CBOR, so [`state_hash`] is the same for the
/// same log on every platform.
///
/// This is the shell DIST-E9-T01 builds: the fields the base events fold into. Later tasks add
/// the rest of RFC-0032 §6.2's fields, each with a serde default so a state they extend still
/// decodes:
/// - the authority-derived fields and their signed records: `revoked`, `api_keys`,
///   `revoked_api_keys`, `rotations`, `keyholders`, `invites` and `pairing` (DIST-E9-T16, with
///   DIST-E9-T03's events);
/// - `dedup`, the command windows (DIST-E9-T02);
/// - `orphans`, the runners reported after a heal (RFC-0032 §7.5);
/// - the members' certificates, approvals, roles and flap history (RFC-0030, DIST-E9-T03), and
///   the cluster feature level's recomputation (DIST-E9-T04).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterState {
    /// The cluster key's public half, which RFC-0034 §4 calls the `ClusterId`. DIST-E2-T06's
    /// `ClusterId` replaces the bare bytes (DIST-E9-T16).
    pub cluster: [u8; 32],
    /// (index, term) of the last applied entry; (0, 0) before the first.
    pub applied: (u64, u64),
    /// The chain value of the last applied entry, or the genesis value before the first.
    pub chain: Blake3,
    /// The cluster's feature level: the minimum over members (RFC-0032 §6.4). DIST-E9-T04
    /// recomputes it in `apply`; until then it stays at its initial 0.
    pub feature_level: u32,
    /// The current coordinator.
    pub coordinator: Option<CoordinatorRecord>,
    /// The highest term in any applied entry or election.
    pub max_term: u64,
    /// The members, by node.
    pub members: BTreeMap<NodeId, MemberRecord>,
    /// The resource graph's state (RFC-0010).
    pub graph: GraphState,
    /// The registered models.
    pub models: BTreeMap<ModelId, ModelRecord>,
    /// The instances.
    pub instances: BTreeMap<InstanceId, InstanceRecord>,
    /// The persisted sessions (RFC-0022). Live sequences are never logged.
    pub sessions: BTreeMap<SessionId, SessionRecord>,
    /// The chunk set each node announced for each model.
    pub chunks: BTreeMap<(NodeId, ModelId), ChunkSetDigest>,
    /// The cluster-scoped configuration keys (RFC-0001 §11.1).
    pub config: ClusterConfig,
    /// The coordinator's allocation counters.
    pub alloc: AllocState,
}

/// The elected coordinator (RFC-0032 §6.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinatorRecord {
    /// The coordinator.
    pub node: NodeId,
    /// Its term.
    pub term: u64,
    /// The index of the `CoordinatorElected` entry.
    pub elected_at_index: u64,
}

/// One member (RFC-0032 §6.2, RFC-0030). DIST-E9-T03 and DIST-E9-T16 add its roles, admission
/// generation, certificate and approval, flap history and feature level.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberRecord {
    /// The member's description.
    pub info: NodeInfo,
    /// Its status.
    pub status: MemberStatus,
    /// Its incarnation, as its last `NodeSleeping` recorded it.
    pub incarnation: u64,
}

/// A member's status, as the base events set it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberStatus {
    /// Admitted and not otherwise reported.
    Alive,
    /// Gossip suspects it. Informational only.
    Suspected,
    /// Its failure is confirmed.
    Failed,
    /// It left.
    Left,
    /// It is asleep.
    Sleeping,
    /// It is restarting, and is expected back by this coordinator wall-clock time.
    Restarting {
        /// When it is expected back.
        back_by_ms: u64,
    },
    /// Its admission was revoked.
    Revoked,
}

/// The resource graph's state in the log. PLC-E1-T11 assembles the graph itself from the
/// `GraphUpdated` deltas; the shell records the version.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphState {
    /// The graph's version (RFC-0010 §1.4).
    pub version: GraphVersion,
}

/// One registered model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelRecord {
    /// Its manifest's digest.
    pub manifest: ManifestDigest,
    /// The latest workload report of each of its instances.
    pub workload: BTreeMap<InstanceId, WorkloadReport>,
}

/// One instance (RFC-0032 §6.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstanceRecord {
    /// The model it serves.
    pub model: ModelId,
    /// Its policy.
    pub policy: InstancePolicy,
    /// Its current plan.
    pub current: Option<CommittedPlan>,
    /// Plans draining under make-before-break or recovery, oldest first.
    pub retiring: Vec<CommittedPlan>,
    /// Each runner's state, by node and plan epoch.
    pub runners: BTreeMap<(NodeId, PlanEpoch), RunnerStateKind>,
    /// The routable plan and its published context length.
    pub routable: Option<Routable>,
    /// The index of the `InstanceRequested` entry.
    pub requested_at: u64,
}

/// A committed plan, as `PlanCommitted` carried it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommittedPlan {
    /// The plan, inline or by digest.
    pub body: PlanBody,
    /// Its summary.
    pub summary: PlanSummary,
}

impl CommittedPlan {
    /// The plan's epoch.
    pub fn epoch(&self) -> PlanEpoch {
        self.summary.epoch
    }
}

/// An instance's routable plan (RFC-0032 §6.5, `InstanceRoutable`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Routable {
    /// The routable plan's epoch.
    pub epoch: PlanEpoch,
    /// The plan's published KV reach, which ingress checks before routing.
    pub context_length: u32,
}

/// One persisted session (RFC-0022).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionRecord {
    /// The instance it is pinned to.
    pub instance: InstanceId,
    /// The nodes holding its KV.
    pub holders: Vec<NodeId>,
}

/// The cluster-scoped configuration keys, each with the entry that set it (RFC-0054 §2.3: a
/// proposer names the (index, term) of the value it last saw).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// The keys set, by path.
    pub keys: BTreeMap<KeyPath, ClusterValue>,
}

/// One cluster-scoped key's value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterValue {
    /// The value.
    pub value: ConfigValue,
    /// (index, term) of the `ClusterConfigChanged` entry that set it.
    pub set_at: (u64, u64),
}

/// The coordinator's allocation counters (RFC-0032 §6.2). `CoordinatorElected` resets the
/// per-term ones. DIST-E9-T02's `decide` allocates from them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocState {
    /// The term the counters belong to.
    pub term: u64,
    /// The next plan sequence number in `term`: one past the highest committed in it.
    pub next_plan_seq: u64,
    /// One past the highest instance ID requested in any term. Instance IDs are never reused,
    /// so an election does not reset it.
    pub next_instance: u64,
}

impl ClusterState {
    /// The state before the first entry of the cluster whose key's public half is `cluster`.
    pub fn new(cluster: [u8; 32]) -> Self {
        ClusterState {
            cluster,
            applied: (0, 0),
            chain: genesis(&cluster),
            feature_level: 0,
            coordinator: None,
            max_term: 0,
            members: BTreeMap::new(),
            graph: GraphState::default(),
            models: BTreeMap::new(),
            instances: BTreeMap::new(),
            sessions: BTreeMap::new(),
            chunks: BTreeMap::new(),
            config: ClusterConfig::default(),
            alloc: AllocState::default(),
        }
    }
}

/// The state's hash: BLAKE3 of its deterministic CBOR. Two folds of the same log give the same
/// hash on every platform.
pub fn state_hash(state: &ClusterState) -> Result<Blake3, EncodeError> {
    let bytes = envelope::to_cbor(state)?;
    Ok(Blake3(*blake3::hash(&bytes).as_bytes()))
}

/// Folds one entry into `state` (RFC-0032 §6.3, with §6.5's effects for the base events).
///
/// Pure and total: it reads only `state` and the entry, never the clock or `at_ms`, and it
/// never panics. Every node runs `validate()` (DIST-E9-T12) first, so an entry that reaches
/// `apply` meets its event's preconditions; an event whose subject is absent anyway changes
/// nothing but `applied`, `chain` and `max_term`.
pub fn apply(state: &mut ClusterState, entry: &LogEntry) {
    state.applied = (entry.index, entry.term);
    state.chain = entry.chain();
    state.max_term = state.max_term.max(entry.term);
    match &entry.event {
        ClusterEvent::NodeAdmitted { info } => {
            state
                .members
                .entry(info.id)
                .and_modify(|member| {
                    member.info = info.clone();
                    member.status = MemberStatus::Alive;
                })
                .or_insert_with(|| MemberRecord {
                    info: info.clone(),
                    status: MemberStatus::Alive,
                    incarnation: 0,
                });
        }
        ClusterEvent::NodeSuspected { node } => {
            if let Some(member) = state.members.get_mut(node)
                && member.status == MemberStatus::Alive
            {
                member.status = MemberStatus::Suspected;
            }
        }
        ClusterEvent::NodeFailed { node } => set_status(state, node, MemberStatus::Failed),
        ClusterEvent::NodeLeft { node } => set_status(state, node, MemberStatus::Left),
        ClusterEvent::NodeSleeping { node, incarnation } => {
            if let Some(member) = state.members.get_mut(node) {
                member.status = MemberStatus::Sleeping;
                member.incarnation = *incarnation;
            }
        }
        ClusterEvent::NodeRestarting { node, back_by_ms } => set_status(
            state,
            node,
            MemberStatus::Restarting {
                back_by_ms: *back_by_ms,
            },
        ),
        ClusterEvent::NodeRevoked { node } => set_status(state, node, MemberStatus::Revoked),
        ClusterEvent::NodeInfoChanged { node, info } => {
            if let Some(member) = state.members.get_mut(node) {
                member.info = info.clone();
            }
        }
        ClusterEvent::CoordinatorElected { node, term, .. } => {
            state.coordinator = Some(CoordinatorRecord {
                node: *node,
                term: *term,
                elected_at_index: entry.index,
            });
            state.max_term = state.max_term.max(*term);
            state.alloc = AllocState {
                term: *term,
                next_plan_seq: 0,
                next_instance: state.alloc.next_instance,
            };
        }
        ClusterEvent::GraphUpdated { version, .. } => state.graph.version = *version,
        ClusterEvent::ModelRegistered { model, manifest } => {
            state
                .models
                .entry(*model)
                .and_modify(|record| record.manifest = *manifest)
                .or_insert_with(|| ModelRecord {
                    manifest: *manifest,
                    workload: BTreeMap::new(),
                });
        }
        ClusterEvent::InstanceRequested {
            instance,
            model,
            policy,
        } => {
            state.alloc.next_instance = state.alloc.next_instance.max(instance.0.saturating_add(1));
            state
                .instances
                .entry(*instance)
                .or_insert_with(|| InstanceRecord {
                    model: *model,
                    policy: (**policy).clone(),
                    current: None,
                    retiring: Vec::new(),
                    runners: BTreeMap::new(),
                    routable: None,
                    requested_at: entry.index,
                });
        }
        ClusterEvent::PlanCommitted { plan, summary } => {
            let epoch = summary.epoch;
            if epoch.term == state.alloc.term {
                state.alloc.next_plan_seq =
                    state.alloc.next_plan_seq.max(epoch.seq.saturating_add(1));
            }
            if let Some(record) = state.instances.get_mut(&summary.instance) {
                let committed = CommittedPlan {
                    body: plan.clone(),
                    summary: summary.clone(),
                };
                // The previous plan drains if it ever served, and is dropped if it never did.
                if let Some(previous) = record.current.replace(committed)
                    && record.routable.map(|r| r.epoch) == Some(previous.epoch())
                {
                    record.retiring.push(previous);
                }
            }
        }
        ClusterEvent::RunnerStateChanged {
            instance,
            node,
            epoch,
            state: runner,
        } => {
            if let Some(record) = state.instances.get_mut(instance) {
                record.runners.insert((*node, *epoch), *runner);
            }
        }
        ClusterEvent::InstanceRoutable {
            instance,
            epoch,
            context_length,
        } => {
            if let Some(record) = state.instances.get_mut(instance) {
                record.routable = Some(Routable {
                    epoch: *epoch,
                    context_length: *context_length,
                });
            }
        }
        ClusterEvent::InstanceUnroutable { instance, .. } => {
            if let Some(record) = state.instances.get_mut(instance) {
                record.routable = None;
            }
        }
        ClusterEvent::PlanRetired { instance, epoch } => {
            if let Some(record) = state.instances.get_mut(instance) {
                record.retiring.retain(|plan| plan.epoch() != *epoch);
                record.runners.retain(|(_, runs), _| runs != epoch);
            }
        }
        ClusterEvent::InstanceRemoved { instance } => {
            state.instances.remove(instance);
        }
        ClusterEvent::WorkloadUpdated {
            model,
            instance,
            report,
        } => {
            if let Some(record) = state.models.get_mut(model) {
                record.workload.insert(*instance, report.clone());
            }
        }
        ClusterEvent::ChunksAnnounced { node, model, have } => {
            state.chunks.insert((*node, *model), have.clone());
        }
        ClusterEvent::SessionPinned {
            session,
            instance,
            holders,
        } => {
            state.sessions.insert(
                *session,
                SessionRecord {
                    instance: *instance,
                    holders: holders.clone(),
                },
            );
        }
        ClusterEvent::SessionDropped { session } => {
            state.sessions.remove(session);
        }
        ClusterEvent::ClusterConfigChanged { patch } => {
            apply_patch(&mut state.config, patch, (entry.index, entry.term));
        }
    }
}

fn set_status(state: &mut ClusterState, node: &NodeId, status: MemberStatus) {
    if let Some(member) = state.members.get_mut(node) {
        member.status = status;
    }
}

/// Sets the patch's keys, then removes its unset keys.
fn apply_patch(config: &mut ClusterConfig, patch: &ConfigPatch, at: (u64, u64)) {
    for (key, value) in &patch.set {
        config.keys.insert(
            key.clone(),
            ClusterValue {
                value: value.clone(),
                set_at: at,
            },
        );
    }
    for key in &patch.unset {
        config.keys.remove(key);
    }
}
