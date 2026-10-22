// SPDX-License-Identifier: MIT OR Apache-2.0

//! proptest strategies for the base events. Every ID comes from a small pool, so a random
//! sequence keeps referring to the same nodes, instances, models and sessions, and reaches the
//! interesting folds: a plan replacing a routable one, a runner of a retired plan, a key set
//! then unset.

use proptest::prelude::*;
use syntropy_cluster::log::payload::{
    ConfigPatch, ManifestDigest, NodeInfo, Opaque, PlanBody, PlanSummary, RunnerStateKind,
    StageSummary, WorkloadReport,
};
use syntropy_cluster::log::{ClusterEvent, CommandId};
use syntropy_core::plan::cost::{BatchingLimits, PredictedCost};
use syntropy_core::policy::{ConfigValue, KeyPath};
use syntropy_core::resource::GraphDelta;
use syntropy_core::{
    Blake3, Bytes, ExactnessPolicy, GraphVersion, InstanceId, InstancePolicy, LayerIx, ModelId,
    NodeId, Objective, PlacementPlan, PlanDigest, PlanEpoch, SessionId,
};

pub fn node() -> impl Strategy<Value = NodeId> {
    (0u8..4).prop_map(|n| NodeId([n; 32]))
}

pub fn instance() -> impl Strategy<Value = InstanceId> {
    (0u64..4).prop_map(InstanceId)
}

pub fn model() -> impl Strategy<Value = ModelId> {
    (0u8..3).prop_map(|m| ModelId(Blake3([m; 32])))
}

pub fn session() -> impl Strategy<Value = SessionId> {
    (0u8..3).prop_map(|s| SessionId([s; 16]))
}

pub fn epoch() -> impl Strategy<Value = PlanEpoch> {
    (0u64..3, 0u64..4).prop_map(|(term, seq)| PlanEpoch { term, seq })
}

pub fn command_id() -> impl Strategy<Value = CommandId> {
    (node(), 0u64..1_000).prop_map(|(origin, counter)| CommandId { origin, counter })
}

/// A CBOR value of the kind an owner's type would encode.
pub fn opaque() -> impl Strategy<Value = Opaque> {
    let leaf = prop_oneof![
        Just(ciborium::Value::Null),
        any::<i64>().prop_map(|n| ciborium::Value::Integer(n.into())),
        "[a-z]{0,8}".prop_map(ciborium::Value::Text),
        any::<bool>().prop_map(ciborium::Value::Bool),
    ];
    leaf.prop_recursive(2, 8, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..3).prop_map(ciborium::Value::Array),
            prop::collection::vec(("[a-z]{1,6}", inner), 0..3).prop_map(|entries| {
                ciborium::Value::Map(
                    entries
                        .into_iter()
                        .map(|(k, v)| (ciborium::Value::Text(k), v))
                        .collect(),
                )
            }),
        ]
    })
    .prop_map(Opaque)
}

fn workload() -> impl Strategy<Value = WorkloadReport> {
    (
        opaque(),
        prop::collection::vec(opaque(), 0..2),
        opaque(),
        opaque(),
    )
        .prop_map(
            |(experts, acceptance, context, concurrency)| WorkloadReport {
                experts,
                acceptance,
                context,
                concurrency,
            },
        )
}

fn config_value() -> impl Strategy<Value = ConfigValue> {
    prop_oneof![
        any::<bool>().prop_map(ConfigValue::Bool),
        any::<i64>().prop_map(ConfigValue::Integer),
        prop::sample::select(vec![0.0, -1.5, 0.25, 1e9]).prop_map(ConfigValue::Float),
        "[a-z]{0,8}".prop_map(ConfigValue::String),
    ]
}

fn key() -> impl Strategy<Value = KeyPath> {
    prop::sample::select(vec![
        "allow_approximate",
        "log.ack_timeout",
        "network_scope",
    ])
    .prop_map(KeyPath::from)
}

fn patch() -> impl Strategy<Value = ConfigPatch> {
    (
        prop::collection::vec((key(), config_value()), 0..3),
        prop::collection::vec(key(), 0..2),
        opaque(),
    )
        .prop_map(|(set, unset, origin)| ConfigPatch { set, unset, origin })
}

/// The smallest plan the core type allows, for the inline form of `PlanCommitted`.
pub fn plan(epoch: PlanEpoch, instance: InstanceId, model: ModelId) -> PlacementPlan {
    PlacementPlan {
        epoch,
        instance,
        model,
        graph: GraphVersion(1),
        policy: Blake3([7; 32]),
        objective: Objective::default(),
        exactness: ExactnessPolicy::default(),
        stages: Vec::new(),
        tensors: Vec::new(),
        expert_cache: Vec::new(),
        kv: Vec::new(),
        prefetch: Vec::new(),
        scratch: Vec::new(),
        speculative: None,
        batching: BatchingLimits::default(),
        predicted: PredictedCost::default(),
        assumptions: Vec::new(),
    }
}

fn plan_committed() -> impl Strategy<Value = ClusterEvent> {
    (
        instance(),
        epoch(),
        model(),
        any::<bool>(),
        prop::collection::vec((node(), 0u16..8, 0u16..8), 0..3),
    )
        .prop_map(|(instance, epoch, model, inline, stages)| {
            let plan = if inline {
                PlanBody::Inline(Box::new(plan(epoch, instance, model)))
            } else {
                PlanBody::ByDigest {
                    digest: PlanDigest(Blake3([epoch.seq as u8; 32])),
                    len: Bytes(70_000),
                }
            };
            let stages = stages
                .into_iter()
                .map(|(node, a, b)| StageSummary {
                    node,
                    layers: LayerIx(a.min(b))..LayerIx(a.max(b)),
                })
                .collect();
            ClusterEvent::PlanCommitted {
                plan,
                summary: PlanSummary {
                    instance,
                    epoch,
                    stages,
                },
            }
        })
}

fn runner_state() -> impl Strategy<Value = RunnerStateKind> {
    prop::sample::select(vec![
        RunnerStateKind::Fetching,
        RunnerStateKind::Packing,
        RunnerStateKind::Loading,
        RunnerStateKind::Warming,
        RunnerStateKind::Ready,
        RunnerStateKind::Draining,
        RunnerStateKind::Failed,
        RunnerStateKind::Stopped,
    ])
}

fn delta() -> impl Strategy<Value = GraphDelta> {
    (
        prop::collection::vec(node(), 0..2),
        prop::collection::vec(node(), 0..2),
    )
        .prop_map(|(nodes_added, nodes_removed)| GraphDelta {
            nodes_added,
            nodes_removed,
            ..GraphDelta::default()
        })
}

/// Any base event, every variant reachable.
pub fn event() -> impl Strategy<Value = ClusterEvent> {
    prop_oneof![
        node().prop_map(|id| ClusterEvent::NodeAdmitted {
            info: NodeInfo { id }
        }),
        node().prop_map(|node| ClusterEvent::NodeSuspected { node }),
        node().prop_map(|node| ClusterEvent::NodeFailed { node }),
        node().prop_map(|node| ClusterEvent::NodeLeft { node }),
        (node(), 0u64..5)
            .prop_map(|(node, incarnation)| ClusterEvent::NodeSleeping { node, incarnation }),
        (node(), any::<u64>())
            .prop_map(|(node, back_by_ms)| ClusterEvent::NodeRestarting { node, back_by_ms }),
        node().prop_map(|node| ClusterEvent::NodeRevoked { node }),
        (node(), node()).prop_map(|(node, id)| ClusterEvent::NodeInfoChanged {
            node,
            info: NodeInfo { id }
        }),
        (node(), 0u64..4, any::<u64>()).prop_map(|(node, term, lease_until_ms)| {
            ClusterEvent::CoordinatorElected {
                node,
                term,
                lease_until_ms,
            }
        }),
        (any::<u64>(), delta()).prop_map(|(v, delta)| ClusterEvent::GraphUpdated {
            version: GraphVersion(v),
            delta
        }),
        (model(), any::<[u8; 32]>()).prop_map(|(model, d)| ClusterEvent::ModelRegistered {
            model,
            manifest: ManifestDigest(Blake3(d)),
        }),
        (instance(), model(), any::<bool>()).prop_map(|(instance, model, preload)| {
            ClusterEvent::InstanceRequested {
                instance,
                model,
                policy: Box::new(InstancePolicy {
                    preload,
                    ..InstancePolicy::default()
                }),
            }
        }),
        plan_committed(),
        (instance(), node(), epoch(), runner_state()).prop_map(|(instance, node, epoch, state)| {
            ClusterEvent::RunnerStateChanged {
                instance,
                node,
                epoch,
                state,
            }
        }),
        (instance(), epoch(), any::<u32>()).prop_map(|(instance, epoch, context_length)| {
            ClusterEvent::InstanceRoutable {
                instance,
                epoch,
                context_length,
            }
        }),
        (instance(), opaque())
            .prop_map(|(instance, reason)| ClusterEvent::InstanceUnroutable { instance, reason }),
        (instance(), epoch())
            .prop_map(|(instance, epoch)| ClusterEvent::PlanRetired { instance, epoch }),
        instance().prop_map(|instance| ClusterEvent::InstanceRemoved { instance }),
        (model(), instance(), workload()).prop_map(|(model, instance, report)| {
            ClusterEvent::WorkloadUpdated {
                model,
                instance,
                report,
            }
        }),
        (node(), model(), opaque()).prop_map(|(node, model, have)| ClusterEvent::ChunksAnnounced {
            node,
            model,
            have
        }),
        (session(), instance(), prop::collection::vec(node(), 0..3)).prop_map(
            |(session, instance, holders)| ClusterEvent::SessionPinned {
                session,
                instance,
                holders,
            }
        ),
        session().prop_map(|session| ClusterEvent::SessionDropped { session }),
        patch().prop_map(|patch| ClusterEvent::ClusterConfigChanged { patch }),
    ]
}

/// One entry's inputs: how far the term moves, the cause, and the event.
pub fn step() -> impl Strategy<Value = (u64, Option<CommandId>, ClusterEvent)> {
    (0u64..2, prop::option::of(command_id()), event())
}
