// SPDX-License-Identifier: MIT OR Apache-2.0

//! Message fixtures: a plan with every field filled and a delta with every kind of change, so a
//! round trip passes every part of their encodings. Core's own tests cover the plan types'
//! values in depth (PLC-E4-T17); these check that postcard carries them.

use smallvec::smallvec;
use syntropy_core::ledger::LedgerGrant;
use syntropy_core::plan::*;
use syntropy_core::resource::{TierKind, TransportKind};
use syntropy_core::{
    ApproxKnobs, ApproxParams, Blake3, Bytes, DeviceId, Endpoint as LinkEnd, ExactnessPolicy,
    GraphVersion, InstanceId, LayerIx, LinkId, LinkKind, MeasureSource, ModelId, Nanos, NetMedium,
    NodeId, Objective, PlacementPlan, PlanDelta, PlanEpoch, QuantScheme, RdmaKind, StageIx,
    TensorGroupId, TierId,
};

use super::super::Init;
use crate::shm::{RingRole, RingSpec, ShmLayout};

pub(super) fn node(n: u8) -> NodeId {
    NodeId([n; 32])
}

fn device(n: u8, ordinal: u16) -> DeviceId {
    DeviceId {
        node: node(n),
        ordinal,
    }
}

fn tier(n: u8, ordinal: u16) -> TierId {
    TierId {
        node: node(n),
        ordinal,
    }
}

fn link(a: u8, b: u8) -> LinkId {
    LinkId {
        src: LinkEnd::Device(device(a, 0)),
        dst: LinkEnd::Tier(tier(b, 1)),
        kind: LinkKind::Network(NetMedium::Rdma(RdmaKind::RoceV2)),
        path: 1,
    }
}

fn hop(a: u8, b: u8) -> HopPlan {
    HopPlan {
        link: link(a, b),
        transport: TransportKind::Rdma(RdmaKind::Thunderbolt5),
        wire: WireDType::Q8B32,
        predicted: Nanos(41_000),
        prefill_transport: Some((link(b, a), TransportKind::Quic)),
        relay: Some(RelayVia {
            node: node(9),
            first: link(a, 9),
            second: link(9, b),
            transport: TransportKind::QuicRelay,
        }),
        predicted_p99: Nanos(97_000),
    }
}

fn split() -> TpSplit {
    TpSplit {
        attn: smallvec![16, 16],
        ffn: smallvec![4096, 4096, 2048],
        experts: Some(smallvec![128, 128]),
    }
}

fn expert_cache() -> ExpertCacheBudget {
    ExpertCacheBudget {
        device: device(1, 0),
        tier: tier(1, 0),
        bytes: Bytes(12 << 30),
        per_layer: Some(vec![Bytes(200 << 20), Bytes(180 << 20)]),
        policy: CachePolicy::DecayedLfu {
            decay: 0.95,
            adapt_every_steps: 64,
            hysteresis: 0.1,
            max_replace: 8,
            guard: true,
        },
        miss_split: MissSplit {
            copied_share: 0.25,
            mode: MissSplitMode::Fitted,
        },
    }
}

fn kv(stage: u16) -> KvBudget {
    KvBudget {
        stage: StageIx(stage),
        tier: tier(1, 0),
        bytes: Bytes(4 << 30),
        dtype: KvDType::K8V4 {
            block: 32,
            rotate: true,
        },
        spill: Some(tier(1, 1)),
        spill_compute: SpillCompute::CpuSplit { max_rows: 512 },
    }
}

fn prefetch() -> PrefetchWindow {
    PrefetchWindow {
        from: tier(1, 2),
        to: tier(1, 1),
        max_inflight: Bytes(256 << 20),
        lookahead_layers: 2,
    }
}

fn speculative() -> SpeculativeConfig {
    SpeculativeConfig {
        source: DraftSource::DraftModel {
            model: ModelId(Blake3([7; 32])),
            device: device(1, 0),
        },
        max_window: 6,
        tree: Some(TreeShape {
            parents: vec![0, 0, 1, 1, 2],
        }),
        acceptance: Acceptance::Exact,
    }
}

fn batching() -> BatchingLimits {
    BatchingLimits {
        max_sequences: 64,
        prefill_chunk: 2048,
        micro_batch: 4,
        prefill_ladder: smallvec![256, 512, 1024, 2048, 4096],
    }
}

fn predicted() -> PredictedCost {
    PredictedCost {
        t_tok: Nanos(48_000_000),
        t_pass: Nanos(52_000_000),
        tokens_per_pass: 1.75,
        ttft: smallvec![(512, Nanos(400_000_000)), (4096, Nanos(2_900_000_000))],
        ttft_ref: (1024, Nanos(800_000_000)),
        ttft_cap: Some(Nanos(10_000_000_000)),
        jitter: Some(Jitter {
            stall_rate: 0.002,
            t_tok_p99: Nanos(90_000_000),
        }),
        context_length: 131_072,
        pass_by_window: smallvec![(1, Nanos(48_000_000)), (4, Nanos(60_000_000))],
        throughput: Some(83.5),
        stages: vec![StageCost {
            stage: StageIx(1),
            terms: smallvec![
                (StageTerm::DenseVram, Nanos(20_000_000)),
                (StageTerm::HopSend, Nanos(1_000_000)),
            ],
            overlapped: smallvec![(StageTerm::ExpertHitCpu, Nanos(3_000_000))],
            cpu_layer_max: Nanos(400_000),
            bytes: smallvec![(tier(1, 1), Bytes(1 << 30))],
            resident: smallvec![ResidentBytes {
                tier: tier(1, 1),
                before: Bytes(1 << 30),
                after: Bytes(3 << 30),
                by_purpose: smallvec![(MemoryPurpose::DenseWeights, Bytes(2 << 30))],
            }],
            prefill_per_chunk: Nanos(90_000_000),
        }],
    }
}

/// A two-stage plan with every field filled, and each enum at a variant that carries data.
pub(super) fn plan() -> PlacementPlan {
    PlacementPlan {
        epoch: PlanEpoch { term: 7, seq: 2 },
        instance: InstanceId(41),
        model: ModelId(Blake3([3; 32])),
        graph: GraphVersion(19),
        policy: Blake3([4; 32]),
        objective: Objective::Throughput {
            per_sequence_floor_tps: 12.5,
            target_concurrency: Some(16),
            ttft_ceiling: Some(Nanos(3_000_000_000)),
        },
        exactness: ExactnessPolicy::Approximate {
            knobs: ApproxKnobs::from_bits_truncate(0b11),
            params: ApproxParams::default(),
        },
        stages: vec![
            StageAssignment {
                stage: StageIx(0),
                node: node(1),
                layers: LayerIx(0)..LayerIx(30),
                role: StageRole::from_bits_truncate(0b01),
                mode: ParallelismMode::CrossNodeTp {
                    members: smallvec![
                        StageMember {
                            node: node(1),
                            devices: smallvec![device(1, 0), device(1, 1)],
                            link: link(1, 2),
                        },
                        StageMember {
                            node: node(2),
                            devices: smallvec![device(2, 0)],
                            link: link(2, 1),
                        },
                    ],
                    split: split(),
                    collective: CollectiveAlgo::Ring,
                },
                next_hop: hop(1, 3),
                predicted_step: Nanos(20_000_000),
            },
            StageAssignment {
                stage: StageIx(1),
                node: node(3),
                layers: LayerIx(30)..LayerIx(61),
                role: StageRole::from_bits_truncate(0b10),
                mode: ParallelismMode::CrossNodeEp {
                    members: smallvec![StageMember {
                        node: node(3),
                        devices: smallvec![device(3, 0)],
                        link: link(3, 1),
                    }],
                    owners: ExpertOwnerMap {
                        layers: vec![(LayerIx(31), vec![0, 1, 0, 1]), (LayerIx(32), vec![1])],
                    },
                },
                next_hop: hop(3, 1),
                predicted_step: Nanos(22_000_000),
            },
        ],
        tensors: vec![
            TierAssignment {
                group: TensorGroupId(0),
                owner: node(1),
                home: tier(1, 0),
                compute: ComputeSite {
                    decode: device(1, 0),
                    batched: None,
                },
                replicas: smallvec![],
            },
            TierAssignment {
                group: TensorGroupId(1),
                owner: node(3),
                home: tier(3, 2),
                compute: ComputeSite {
                    decode: device(3, 1),
                    batched: Some(StreamSite {
                        device: device(3, 0),
                        via: link(3, 3),
                        min_tokens: 96,
                    }),
                },
                replicas: smallvec![node(1), node(2), node(4)],
            },
        ],
        expert_cache: vec![expert_cache()],
        kv: vec![kv(0), kv(1)],
        prefetch: vec![prefetch()],
        scratch: vec![(device(1, 0), Bytes(1 << 30))],
        speculative: Some(speculative()),
        batching: batching(),
        predicted: predicted(),
        assumptions: vec![
            Assumption {
                input: AssumedInput::QuantBw {
                    device: device(1, 0),
                    tier: tier(1, 1),
                    scheme: QuantScheme::Mxfp4 { block: 32 },
                },
                source: MeasureSource::Prior,
                sensitivity: Nanos(1_500_000),
            },
            Assumption {
                input: AssumedInput::LinkP99 { link: link(1, 3) },
                source: MeasureSource::Online,
                sensitivity: Nanos(250_000),
            },
        ],
        ext: PlacementPlanExt {
            transition: TransitionStrategy::MakeBeforeBreak,
            explanation: Some(Blake3([9; 32])),
            solve: SolveStats {
                path: SolvePath::MilpIncumbent,
                gap: 0.004,
                mip_nodes: 812,
                simplex_iterations: 40_000,
                wall: Nanos(1_200_000_000),
            },
            degraded: Some(DegradedReason::MultiStageOnNode),
            move_bytes: Bytes(5 << 30),
            certification: PlanCertification {
                status: CertStatus::Derived,
                runtime: CertifiedRuntime {
                    prefix_reuse: true,
                    kv_spill_tiers: smallvec![TierKind::HostPinned],
                    max_concurrency: 8,
                    max_window: 4,
                },
                relies_on: vec![CertRef {
                    kind: CertKind::Family,
                    id: Blake3([4; 32]),
                }],
            },
            rdma_qps: smallvec![(NodeId([1; 32]), 2)],
            per_site_layout: vec![TensorGroupId(3)],
            wan_rate_class: smallvec![(NodeId([2; 32]), 6)],
        },
    }
}

/// The plan with every optional field it can leave unset left unset, which is where a field
/// skipped on the way out would show.
pub(super) fn sparse_plan() -> PlacementPlan {
    let mut plan = plan();
    plan.objective = Objective::Throughput {
        per_sequence_floor_tps: 1.0,
        target_concurrency: None,
        ttft_ceiling: None,
    };
    plan.exactness = ExactnessPolicy::Approximate {
        knobs: ApproxKnobs::from_bits_truncate(0b100),
        params: ApproxParams::default(),
    };
    for stage in &mut plan.stages {
        stage.next_hop.prefill_transport = None;
        stage.next_hop.relay = None;
        if let ParallelismMode::CrossNodeTp { split, .. } = &mut stage.mode {
            split.experts = None;
        }
    }
    for tensor in &mut plan.tensors {
        tensor.compute.batched = None;
    }
    for budget in &mut plan.expert_cache {
        budget.per_layer = None;
    }
    for kv in &mut plan.kv {
        kv.spill = None;
    }
    plan.speculative = None;
    plan.predicted.ttft_cap = None;
    plan.predicted.jitter = None;
    plan.predicted.throughput = None;
    plan
}

/// A delta with one change of every kind.
pub(super) fn delta() -> PlanDelta {
    PlanDelta {
        from: PlanEpoch { term: 7, seq: 2 },
        to: PlanEpoch { term: 7, seq: 3 },
        changes: vec![
            DeltaItem::ExpertCache {
                device: device(1, 0),
                budget: expert_cache(),
            },
            DeltaItem::Kv {
                stage: StageIx(1),
                budgets: smallvec![kv(1)],
            },
            DeltaItem::Prefetch { window: prefetch() },
            DeltaItem::Speculative {
                config: Some(speculative()),
            },
            DeltaItem::Speculative { config: None },
            DeltaItem::Batching { limits: batching() },
            DeltaItem::Predicted { cost: predicted() },
            DeltaItem::Relink {
                stage: StageIx(0),
                prev: node(3),
                next: node(3),
                hop: hop(1, 3),
            },
            DeltaItem::ExtendShard {
                stages: smallvec![StageIx(1), StageIx(2), StageIx(3)],
            },
        ],
    }
}

/// An `Init` with a three-tier grant and the five ring roles.
pub(super) fn init() -> Init {
    let ring = |role, slots, slot_bytes| RingSpec {
        role,
        slots,
        slot_bytes,
    };
    Init {
        build_id: [0x5a; 32],
        instance: InstanceId(41),
        ledger: LedgerGrant::new([
            (tier(1, 0), Bytes(20 << 30)),
            (tier(1, 1), Bytes(96 << 30)),
            (tier(1, 2), Bytes(1 << 40)),
        ])
        .expect("distinct tiers"),
        shm: ShmLayout {
            rings: vec![
                ring(RingRole::Telemetry, 256, 256),
                ring(RingRole::Step { dispatcher: 1 }, 256, 64),
                ring(RingRole::LocalTokens, 1024, 64),
                ring(RingRole::LocalSequences, 1024, 64),
                ring(RingRole::Log, 256, 256),
            ],
        },
    }
}
