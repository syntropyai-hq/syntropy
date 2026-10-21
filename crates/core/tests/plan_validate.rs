// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PlacementPlan::validate` against RFC-0060 §3.2's P-solver-1 invariants (PLC-E4-T17).
//!
//! One valid two-stage plan passes; each test then breaks one invariant and checks the
//! violation `validate` reports.

use smallvec::smallvec;
use syntropy_core::plan::*;
use syntropy_core::resource::TransportKind;
use syntropy_core::*;

// ---- Fixture ----------------------------------------------------------------------------
//
// Two nodes, A and B, one GPU each. A runs stage 0 (layers 0..2, HEAD), B stage 1 (layers
// 2..4, TAIL). The embedding (group 0, RowLookup) and the LM head (group 3, Head) are tied:
// one tensor record, two groups, one copy on each node.
//
// Usage per tier with the fixture's numbers:
//   A VRAM: embedding 10 + layer group 20 + KV 30 + scratch 5 = 65 of 100
//   A RAM:  expert cache 40                                  = 40 of 1000
//   B VRAM: layer group 20 + head 10 + KV 30 + scratch 5     = 65 of 100

const A: NodeId = NodeId([0xA; 32]);
const B: NodeId = NodeId([0xB; 32]);

fn tier(node: NodeId, ordinal: u16) -> TierId {
    TierId { node, ordinal }
}

fn gpu(node: NodeId) -> DeviceId {
    DeviceId { node, ordinal: 0 }
}

fn link(from: NodeId, to: NodeId) -> LinkId {
    LinkId {
        src: Endpoint::Device(gpu(from)),
        dst: Endpoint::Device(gpu(to)),
        kind: LinkKind::Network(NetMedium::Ethernet),
        path: 0,
    }
}

fn stage(ix: u16, node: NodeId, layers: std::ops::Range<u16>, role: StageRole) -> StageAssignment {
    StageAssignment {
        stage: StageIx(ix),
        node,
        layers: LayerIx(layers.start)..LayerIx(layers.end),
        role,
        mode: ParallelismMode::Local,
        next_hop: HopPlan {
            link: link(node, if node == A { B } else { A }),
            transport: TransportKind::Quic,
            wire: WireDType::F32,
            predicted: Nanos(1_000),
        },
        predicted_step: Nanos(10_000),
    }
}

fn assign(group: u32, node: NodeId, ordinal: u16) -> TierAssignment {
    TierAssignment {
        group: TensorGroupId(group),
        owner: node,
        home: tier(node, ordinal),
        compute: ComputeSite {
            decode: gpu(node),
            batched: None,
        },
        replicas: smallvec![],
    }
}

fn kv(ix: u16, node: NodeId, bytes: u64) -> KvBudget {
    KvBudget {
        stage: StageIx(ix),
        tier: tier(node, 0),
        bytes: Bytes(bytes),
        dtype: KvDType::Native,
        spill: Some(tier(node, 1)),
        spill_compute: SpillCompute::GpuStream,
    }
}

fn valid_plan() -> PlacementPlan {
    PlacementPlan {
        epoch: PlanEpoch { term: 1, seq: 1 },
        instance: InstanceId(1),
        model: ModelId(Blake3([1; 32])),
        graph: GraphVersion(1),
        policy: Blake3([2; 32]),
        objective: Objective::Latency,
        exactness: ExactnessPolicy::Exact,
        stages: vec![
            stage(0, A, 0..2, StageRole::HEAD),
            stage(1, B, 2..4, StageRole::TAIL),
        ],
        tensors: vec![
            assign(0, A, 0),
            assign(1, A, 0),
            assign(2, B, 0),
            assign(3, B, 0),
        ],
        expert_cache: vec![ExpertCacheBudget {
            device: gpu(A),
            tier: tier(A, 1),
            bytes: Bytes(40),
            per_layer: None,
            policy: CachePolicy::Lru,
            miss_split: MissSplit {
                copied_share: 0.0,
                mode: MissSplitMode::Fitted,
            },
        }],
        kv: vec![kv(0, A, 30), kv(1, B, 30)],
        prefetch: vec![],
        scratch: vec![(gpu(A), Bytes(5)), (gpu(B), Bytes(5))],
        speculative: None,
        batching: BatchingLimits::default(),
        predicted: PredictedCost::default(),
        assumptions: vec![],
    }
}

fn group(id: u32, role: GroupRole, bytes: u64, shared_with: Option<u32>) -> GroupFacts {
    GroupFacts {
        id: TensorGroupId(id),
        role,
        bytes: Bytes(bytes),
        shared_with: shared_with.map(TensorGroupId),
    }
}

struct Bounds {
    groups: Vec<GroupFacts>,
    usable: Vec<(TierId, Bytes)>,
    scratch_tiers: Vec<(DeviceId, TierId)>,
}

impl Bounds {
    fn new() -> Self {
        Bounds {
            groups: vec![
                group(0, GroupRole::RowLookup, 10, Some(3)),
                group(1, GroupRole::Other, 20, None),
                group(2, GroupRole::Other, 20, None),
                group(3, GroupRole::Head, 10, Some(0)),
            ],
            usable: vec![
                (tier(A, 0), Bytes(100)),
                (tier(A, 1), Bytes(1000)),
                (tier(B, 0), Bytes(100)),
                (tier(B, 1), Bytes(1000)),
            ],
            scratch_tiers: vec![(gpu(A), tier(A, 0)), (gpu(B), tier(B, 0))],
        }
    }

    fn get(&self) -> PlanBounds<'_> {
        PlanBounds {
            n_layers: 4,
            groups: &self.groups,
            usable: &self.usable,
            scratch_tiers: &self.scratch_tiers,
        }
    }

    fn set_usable(&mut self, t: TierId, bytes: u64) {
        self.usable.iter_mut().find(|(id, _)| *id == t).unwrap().1 = Bytes(bytes);
    }
}

fn check(plan: &PlacementPlan) -> Result<(), PlanViolation> {
    plan.validate(&Bounds::new().get())
}

// ---- The valid plan ---------------------------------------------------------------------

#[test]
fn valid_plan_passes() {
    assert_eq!(check(&valid_plan()), Ok(()));
}

#[test]
fn valid_plan_passes_at_exact_capacity() {
    let mut bounds = Bounds::new();
    bounds.set_usable(tier(A, 0), 65);
    bounds.set_usable(tier(A, 1), 40);
    bounds.set_usable(tier(B, 0), 65);
    assert_eq!(valid_plan().validate(&bounds.get()), Ok(()));
}

#[test]
fn one_stage_plan_with_both_roles_passes() {
    let mut plan = valid_plan();
    plan.stages = vec![stage(0, A, 0..4, StageRole::HEAD | StageRole::TAIL)];
    plan.tensors = vec![
        assign(0, A, 0),
        assign(1, A, 0),
        assign(2, A, 0),
        assign(3, A, 0),
    ];
    plan.kv = vec![kv(0, A, 30)];
    plan.scratch = vec![(gpu(A), Bytes(5))];
    assert_eq!(check(&plan), Ok(()));
}

// ---- Invariant 1: one TierAssignment per group ------------------------------------------

#[test]
fn rejects_a_group_with_no_assignment() {
    let mut plan = valid_plan();
    plan.tensors.remove(2);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::GroupUnassigned {
            group: TensorGroupId(2)
        })
    );
}

#[test]
fn rejects_a_group_assigned_twice() {
    let mut plan = valid_plan();
    plan.tensors.push(assign(1, A, 1));
    assert_eq!(
        check(&plan),
        Err(PlanViolation::GroupAssignedTwice {
            group: TensorGroupId(1)
        })
    );
}

#[test]
fn rejects_an_assignment_for_a_group_not_in_the_census() {
    let mut plan = valid_plan();
    plan.tensors.push(assign(9, A, 0));
    assert_eq!(
        check(&plan),
        Err(PlanViolation::UnknownGroup {
            group: TensorGroupId(9)
        })
    );
}

#[test]
fn rejects_a_home_tier_off_the_owner_node() {
    let mut plan = valid_plan();
    plan.tensors[1].home = tier(B, 0);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::HomeOffOwner {
            group: TensorGroupId(1),
            owner: A,
            home: tier(B, 0),
        })
    );
}

// ---- Invariant 2: layer ranges cover [0, n_layers) exactly once, in ring order -----------

#[test]
fn rejects_a_plan_with_no_stages() {
    let mut plan = valid_plan();
    plan.stages.clear();
    assert_eq!(check(&plan), Err(PlanViolation::NoStages));
}

#[test]
fn rejects_a_gap_between_stages() {
    let mut plan = valid_plan();
    plan.stages[1].layers = LayerIx(3)..LayerIx(4);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::LayerGap {
            stage: StageIx(1),
            expected: LayerIx(2),
            found: LayerIx(3),
        })
    );
}

#[test]
fn rejects_overlapping_stages() {
    let mut plan = valid_plan();
    plan.stages[1].layers = LayerIx(1)..LayerIx(4);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::LayerGap {
            stage: StageIx(1),
            expected: LayerIx(2),
            found: LayerIx(1),
        })
    );
}

#[test]
fn rejects_a_first_stage_not_starting_at_layer_zero() {
    let mut plan = valid_plan();
    plan.stages[0].layers = LayerIx(1)..LayerIx(2);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::LayerGap {
            stage: StageIx(0),
            expected: LayerIx(0),
            found: LayerIx(1),
        })
    );
}

#[test]
fn rejects_layers_left_uncovered_at_the_end() {
    let mut plan = valid_plan();
    plan.stages[1].layers = LayerIx(2)..LayerIx(3);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::LayersEnd {
            expected: LayerIx(4),
            found: LayerIx(3),
        })
    );
}

#[test]
fn rejects_layers_past_the_end() {
    let mut plan = valid_plan();
    plan.stages[1].layers = LayerIx(2)..LayerIx(5);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::LayersEnd {
            expected: LayerIx(4),
            found: LayerIx(5),
        })
    );
}

#[test]
fn rejects_an_empty_stage() {
    let mut plan = valid_plan();
    plan.stages[0].layers = LayerIx(0)..LayerIx(0);
    plan.stages[1].layers = LayerIx(0)..LayerIx(4);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::EmptyStage { stage: StageIx(0) })
    );
}

#[test]
fn rejects_stages_out_of_ring_order() {
    let mut plan = valid_plan();
    plan.stages.swap(0, 1);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::StageOutOfOrder {
            position: 0,
            stage: StageIx(1),
        })
    );
}

#[test]
fn rejects_a_head_role_off_the_first_stage() {
    let mut plan = valid_plan();
    plan.stages[1].role = StageRole::HEAD | StageRole::TAIL;
    assert_eq!(
        check(&plan),
        Err(PlanViolation::RoleMismatch {
            stage: StageIx(1),
            role: StageRole::HEAD | StageRole::TAIL,
        })
    );
}

#[test]
fn rejects_a_tail_without_the_tail_role() {
    let mut plan = valid_plan();
    plan.stages[1].role = StageRole::DRAFT;
    assert_eq!(
        check(&plan),
        Err(PlanViolation::RoleMismatch {
            stage: StageIx(1),
            role: StageRole::DRAFT,
        })
    );
}

// ---- Invariant 3: RowLookup on the head, Head on the tail --------------------------------

#[test]
fn rejects_a_row_lookup_group_off_the_head() {
    let mut plan = valid_plan();
    plan.tensors[0] = assign(0, B, 0);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::LookupOffHead {
            group: TensorGroupId(0),
            owner: B,
            head: A,
        })
    );
}

#[test]
fn rejects_a_head_group_off_the_tail() {
    let mut plan = valid_plan();
    plan.tensors[3] = assign(3, A, 0);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::HeadOffTail {
            group: TensorGroupId(3),
            owner: A,
            tail: B,
        })
    );
}

// ---- Invariant 4: per-node per-tier usage within usable ----------------------------------

#[test]
fn rejects_weights_over_usable() {
    let mut bounds = Bounds::new();
    bounds.groups[1].bytes = Bytes(56); // A VRAM: 10 + 56 + 30 + 5 = 101
    assert_eq!(
        valid_plan().validate(&bounds.get()),
        Err(PlanViolation::OverCapacity {
            tier: tier(A, 0),
            used: Bytes(101),
            usable: Bytes(100),
        })
    );
}

#[test]
fn counts_kv() {
    let mut plan = valid_plan();
    plan.kv[1].bytes = Bytes(66); // B VRAM: 20 + 10 + 66 + 5 = 101
    assert_eq!(
        check(&plan),
        Err(PlanViolation::OverCapacity {
            tier: tier(B, 0),
            used: Bytes(101),
            usable: Bytes(100),
        })
    );
}

#[test]
fn counts_scratch() {
    let mut plan = valid_plan();
    plan.scratch[0].1 = Bytes(41); // A VRAM: 10 + 20 + 30 + 41 = 101
    assert_eq!(
        check(&plan),
        Err(PlanViolation::OverCapacity {
            tier: tier(A, 0),
            used: Bytes(101),
            usable: Bytes(100),
        })
    );
}

#[test]
fn counts_the_expert_cache() {
    let mut bounds = Bounds::new();
    bounds.set_usable(tier(A, 1), 39);
    assert_eq!(
        valid_plan().validate(&bounds.get()),
        Err(PlanViolation::OverCapacity {
            tier: tier(A, 1),
            used: Bytes(40),
            usable: Bytes(39),
        })
    );
}

#[test]
fn counts_the_duplicated_embedding_and_head_on_each_node() {
    // The tied head is one tensor record, but B holds its own copy: 65 bytes, of which the
    // head's 10 are the duplicate. Without it B would need only 55.
    let mut bounds = Bounds::new();
    bounds.set_usable(tier(B, 0), 64);
    assert_eq!(
        valid_plan().validate(&bounds.get()),
        Err(PlanViolation::OverCapacity {
            tier: tier(B, 0),
            used: Bytes(65),
            usable: Bytes(64),
        })
    );
}

#[test]
fn counts_tied_groups_once_on_one_tier() {
    // One stage on A: both tied groups on A VRAM hold one copy, 10 + 20 + 20 + 30 + 5 = 85.
    let mut plan = valid_plan();
    plan.stages = vec![stage(0, A, 0..4, StageRole::HEAD | StageRole::TAIL)];
    plan.tensors = vec![
        assign(0, A, 0),
        assign(1, A, 0),
        assign(2, A, 0),
        assign(3, A, 0),
    ];
    plan.kv = vec![kv(0, A, 30)];
    plan.scratch = vec![(gpu(A), Bytes(5))];
    let mut bounds = Bounds::new();
    bounds.set_usable(tier(A, 0), 85);
    assert_eq!(plan.validate(&bounds.get()), Ok(()));
    bounds.set_usable(tier(A, 0), 84);
    assert_eq!(
        plan.validate(&bounds.get()),
        Err(PlanViolation::OverCapacity {
            tier: tier(A, 0),
            used: Bytes(85),
            usable: Bytes(84),
        })
    );
}

#[test]
fn rejects_a_tier_with_no_budget() {
    let mut plan = valid_plan();
    plan.kv[0].tier = tier(A, 7);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::UnknownTier { tier: tier(A, 7) })
    );
}

#[test]
fn rejects_scratch_on_a_device_with_no_scratch_tier() {
    let mut plan = valid_plan();
    let stray = DeviceId {
        node: A,
        ordinal: 3,
    };
    plan.scratch.push((stray, Bytes(1)));
    assert_eq!(
        check(&plan),
        Err(PlanViolation::UnknownDevice { device: stray })
    );
}

#[test]
fn usage_saturates_instead_of_wrapping() {
    let mut plan = valid_plan();
    plan.kv[0].bytes = Bytes(u64::MAX);
    assert_eq!(
        check(&plan),
        Err(PlanViolation::OverCapacity {
            tier: tier(A, 0),
            used: Bytes(u64::MAX),
            usable: Bytes(100),
        })
    );
}

#[test]
fn violations_display_a_reason() {
    let v = PlanViolation::OverCapacity {
        tier: tier(A, 0),
        used: Bytes(101),
        usable: Bytes(100),
    };
    assert_eq!(
        v.to_string(),
        "tier 0 holds 101 bytes, above its usable 100"
    );
}
