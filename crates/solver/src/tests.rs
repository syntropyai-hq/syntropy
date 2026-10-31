// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of RFC-0011 §2's solver types and entry points (PLC-E4-T01): each entry point returns
//! its typed "not built yet" error, and every type with serde round-trips through JSON.

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use smallvec::SmallVec;
use syntropy_core::plan::{PlanEpoch, StageCost, StageTerm};
use syntropy_core::policy::{ExactnessPolicy, InstancePolicy, Objective};
use syntropy_core::replan::ReplanReason;
use syntropy_core::resource::ResourceGraph;
use syntropy_core::sequence::Priority;
use syntropy_core::{
    Blake3, Bytes, DType, GraphVersion, InstanceId, LayerIx, MeasureSource, Measured, ModelId,
    Nanos, NodeId, SessionId, StageIx, TierId,
};
use syntropy_manifest::census::{
    ClassTotals, ModelManifest, SourceFormat, TensorCensus, TensorClass, TensorGroup,
};
use syntropy_manifest::hparams::HyperParams;
use syntropy_manifest::kv::{KvFormula, LayerKv};

use super::*;

fn graph() -> ResourceGraph {
    ResourceGraph {
        version: GraphVersion(1),
        nodes: vec![],
        devices: vec![],
        tiers: vec![],
        links: vec![],
        profiles: Default::default(),
        reservations: vec![],
    }
}

fn manifest() -> ModelManifest {
    let groups = vec![TensorGroup {
        id: syntropy_core::TensorGroupId(0),
        class: TensorClass::DenseEveryToken,
        layer: None,
        tensors: vec![],
        bytes: Bytes(128),
        read_per_token: Bytes(128),
        shared_with: None,
    }];
    ModelManifest {
        id: ModelId(Blake3([3; 32])),
        family: "llama".into(),
        format: SourceFormat::Gguf { version: 3 },
        hparams: HyperParams::new(1, 8, 16),
        census: TensorCensus {
            tensors: vec![],
            totals: ClassTotals::of(&groups),
            groups,
        },
        kv: KvFormula::new(
            vec![LayerKv::PerToken {
                elems_per_token: 16,
                window: None,
                latent: false,
            }],
            DType::F16,
        ),
        tokenizer: None,
        ignored: vec![],
    }
}

fn plan() -> PlacementPlan {
    PlacementPlan {
        epoch: PlanEpoch { term: 1, seq: 1 },
        instance: InstanceId(1),
        model: ModelId(Blake3([3; 32])),
        graph: GraphVersion(1),
        policy: Blake3([2; 32]),
        objective: Objective::Latency,
        exactness: ExactnessPolicy::Exact,
        stages: vec![],
        tensors: vec![],
        expert_cache: vec![],
        kv: vec![],
        prefetch: vec![],
        scratch: vec![],
        speculative: None,
        batching: BatchingLimits::default(),
        predicted: PredictedCost::default(),
        assumptions: vec![],
        ext: PlacementPlanExt::default(),
    }
}

/// Runs `check` on a solve input over the fixtures above.
fn with_input(check: impl FnOnce(&SolveInput<'_>)) {
    let (graph, manifest, policy) = (graph(), manifest(), InstancePolicy::default());
    let workload = WorkloadStats::default();
    let input = SolveInput {
        graph: &graph,
        manifest: &manifest,
        policy: &policy,
        current: None,
        workload: &workload,
        reason: SolveReason::Load,
        limits: SolveLimits::default(),
    };
    check(&input);
}

#[test]
fn solve_is_not_built_yet() {
    with_input(|input| {
        assert_eq!(
            solve(input),
            Err(SolveError::NotBuilt {
                entry: Entry::Solve,
                task: "PLC-E4-T02"
            })
        );
    });
}

#[test]
fn solve_delta_is_not_built_yet() {
    with_input(|input| {
        assert_eq!(
            solve_delta(input, &plan()),
            Err(SolveError::NotBuilt {
                entry: Entry::SolveDelta,
                task: "PLC-E6-T03"
            })
        );
    });
}

#[test]
fn route_is_not_built_yet() {
    let pins = [(SessionId([7; 16]), InstanceId(1))];
    let view = RoutingView {
        sessions: &pins,
        ..RoutingView::default()
    };
    assert_eq!(
        route(&view, &RouteRequest::default()),
        Err(SolveError::NotBuilt {
            entry: Entry::Route,
            task: "PLC-E6-T19"
        })
    );
}

#[test]
fn evaluate_is_not_built_yet() {
    with_input(|input| {
        assert_eq!(
            evaluate(&plan(), input),
            Err(SolveError::NotBuilt {
                entry: Entry::Evaluate,
                task: "PLC-E4-T02"
            })
        );
    });
}

#[test]
fn not_built_names_the_function_and_its_task() {
    let error = SolveError::NotBuilt {
        entry: Entry::SolveDelta,
        task: "PLC-E6-T03",
    };
    assert_eq!(
        error.to_string(),
        "the solver's `solve_delta` is not built yet; PLC-E6-T03 builds it"
    );
}

/// RFC-0011 §16's defaults; the work limits are unbounded until spike S1 sets them.
#[test]
fn limits_default_to_rfc_0011_s16() {
    let limits = SolveLimits::default();
    assert_eq!(limits.max_nodes, 8);
    assert_eq!(limits.safety_time_limit, Nanos(2_000_000_000));
    assert_eq!(limits.mip_rel_gap, 0.01);
    assert_eq!(limits.mip_node_limit, u32::MAX);
    assert_eq!(limits.simplex_iteration_limit, u64::MAX);
}

/// Every field of the limits and the workload statistics has a serde default.
#[test]
fn absent_fields_decode_to_their_defaults() {
    let limits: SolveLimits = serde_json::from_str(r#"{"max_nodes":4}"#).unwrap();
    assert_eq!(
        limits,
        SolveLimits {
            max_nodes: 4,
            ..SolveLimits::default()
        }
    );
    let stats: WorkloadStats = serde_json::from_str("{}").unwrap();
    assert_eq!(stats, WorkloadStats::default());
}

fn round_trip<T>(value: &T)
where
    T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_string(value).unwrap();
    let back: T = serde_json::from_str(&json).unwrap();
    assert_eq!(&back, value, "{json}");
}

fn source() -> impl Strategy<Value = MeasureSource> {
    prop_oneof![
        Just(MeasureSource::Startup),
        Just(MeasureSource::Online),
        Just(MeasureSource::Prior),
    ]
}

fn tier() -> impl Strategy<Value = TierId> {
    (any::<[u8; 32]>(), any::<u16>()).prop_map(|(node, ordinal)| TierId {
        node: NodeId(node),
        ordinal,
    })
}

prop_compose! {
    fn hit_curve()(
        layer in any::<u16>(),
        pool in any::<u64>(),
        points in prop::collection::vec((any::<u64>(), 0f32..=1.0), 0..12),
        source in source(),
    ) -> HitCurve {
        HitCurve {
            layer: LayerIx(layer),
            pool: Bytes(pool),
            points: points.into_iter().map(|(c, h)| (Bytes(c), h)).collect(),
            source,
        }
    }
}

prop_compose! {
    fn union_curve()(
        layer in any::<u16>(),
        experts in any::<u16>(),
        top_k in any::<u16>(),
        window in prop::collection::vec((any::<u16>(), 0f32..=512.0), 0..10),
        batch in prop::collection::vec((any::<u16>(), 0f32..=512.0), 0..10),
        source in source(),
    ) -> UnionCurve {
        UnionCurve {
            layer: LayerIx(layer),
            experts,
            top_k,
            window: window.into_iter().collect(),
            batch: batch.into_iter().collect(),
            source,
        }
    }
}

prop_compose! {
    fn measured()(p50 in 0f32..=1.0, p99 in 0f32..=1.0, samples in any::<u32>(),
                   at_unix_ms in any::<u64>(), source in source()) -> Measured<f32> {
        Measured { p50, p99, samples, at_unix_ms, source }
    }
}

prop_compose! {
    fn workload()(
        hit_curves in prop::collection::vec(hit_curve(), 0..4),
        union in prop::collection::vec(union_curve(), 0..4),
        tier_hit in prop::collection::vec((tier(), any::<u16>(), measured()), 0..4),
    ) -> WorkloadStats {
        WorkloadStats {
            hit_curves,
            union,
            tier_hit: tier_hit.into_iter().map(|(t, l, m)| (t, LayerIx(l), m)).collect(),
        }
    }
}

prop_compose! {
    fn limits()(
        max_nodes in any::<u16>(),
        mip_node_limit in any::<u32>(),
        simplex_iteration_limit in any::<u64>(),
        safety in any::<u64>(),
        mip_rel_gap in 0f64..=1.0,
    ) -> SolveLimits {
        SolveLimits {
            max_nodes,
            mip_node_limit,
            simplex_iteration_limit,
            safety_time_limit: Nanos(safety),
            mip_rel_gap,
        }
    }
}

fn reason() -> impl Strategy<Value = SolveReason> {
    prop_oneof![
        Just(SolveReason::Load),
        Just(SolveReason::Delta),
        Just(SolveReason::Replan(ReplanReason::User)),
        any::<[u8; 32]>()
            .prop_map(|node| SolveReason::Replan(ReplanReason::NodeJoined { node: NodeId(node) })),
    ]
}

prop_compose! {
    fn stage_cost()(
        stage in any::<u16>(),
        terms in prop::collection::vec((0u8..18, any::<u64>()), 0..18),
        overlapped in prop::collection::vec((0u8..18, any::<u64>()), 0..4),
        cpu_layer_max in any::<u64>(),
        prefill_per_chunk in any::<u64>(),
    ) -> StageCost {
        let term = |(code, nanos): (u8, u64)| (StageTerm::from_code(code).unwrap(), Nanos(nanos));
        StageCost {
            stage: StageIx(stage),
            terms: terms.into_iter().map(term).collect(),
            overlapped: overlapped.into_iter().map(term).collect(),
            cpu_layer_max: Nanos(cpu_layer_max),
            bytes: SmallVec::new(),
            resident: SmallVec::new(),
            prefill_per_chunk: Nanos(prefill_per_chunk),
        }
    }
}

prop_compose! {
    fn outcome()(
        with_plan in any::<bool>(),
        keep_current in any::<bool>(),
        term in any::<u64>(),
        stages in prop::collection::vec(stage_cost(), 0..3),
        infeasible in any::<bool>(),
        wall in any::<u64>(),
    ) -> SolveOutcome {
        let explanation = PlanExplanation {
            epoch: PlanEpoch { term, seq: 1 },
            ..PlanExplanation::default()
        };
        let report = InfeasibilityReport {
            degraded_option: Some(DegradedReason::DenseOnNvme),
            ..InfeasibilityReport::default()
        };
        let predicted = PredictedCost { stages, ..PredictedCost::default() };
        SolveOutcome {
            plan: with_plan.then(|| PlacementPlan { predicted, ..plan() }),
            keep_current,
            explanation,
            infeasible: infeasible.then_some(report),
            stats: SolveStats { wall: Nanos(wall), ..SolveStats::default() },
        }
    }
}

proptest! {
    #[test]
    fn hit_curves_round_trip(curve in hit_curve()) {
        round_trip(&curve);
    }

    #[test]
    fn union_curves_round_trip(curve in union_curve()) {
        round_trip(&curve);
    }

    #[test]
    fn workload_stats_round_trip(stats in workload()) {
        round_trip(&stats);
    }

    #[test]
    fn limits_round_trip(limits in limits()) {
        round_trip(&limits);
    }

    #[test]
    fn reasons_round_trip(reason in reason()) {
        round_trip(&reason);
    }

    #[test]
    fn outcomes_round_trip(outcome in outcome()) {
        round_trip(&outcome);
    }

    #[test]
    fn route_requests_round_trip(
        prompt_tokens in any::<u32>(),
        session in proptest::option::of(any::<[u8; 16]>()),
        greedy in any::<bool>(),
    ) {
        round_trip(&RouteRequest {
            prompt_tokens,
            session: session.map(SessionId),
            greedy,
            priority: Priority::Background,
            ..RouteRequest::default()
        });
    }

    #[test]
    fn route_decisions_round_trip(
        instance in any::<u64>(),
        seq in any::<u64>(),
        prefill_chunk in any::<u32>(),
        initial_window in any::<u16>(),
        ttft in any::<u64>(),
    ) {
        round_trip(&RouteDecision {
            instance: InstanceId(instance),
            epoch: PlanEpoch { term: 1, seq },
            prefill_chunk,
            initial_window,
            predicted_ttft: Nanos(ttft),
        });
    }

    #[test]
    fn evaluations_round_trip(t_tok in any::<u64>(), stages in prop::collection::vec(stage_cost(), 0..3)) {
        let mut evaluation = Evaluation::default();
        evaluation.predicted.t_tok = Nanos(t_tok);
        evaluation.predicted.stages = stages;
        round_trip(&evaluation);
    }
}
