// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PlacementPlan::validate` against RFC-0060 §3.2's P-solver-1 invariants (PLC-E4-T17).
//!
//! One valid two-stage plan passes; each test then breaks one invariant and checks the
//! violation `validate` reports.

use smallvec::smallvec;
use syntropy_core::plan::*;
use syntropy_core::resource::{HostPoolInfo, TierKind, TransportKind};
use syntropy_core::*;

// ---- Fixture ----------------------------------------------------------------------------
//
// Two nodes, A and B, one GPU each. A runs stage 0 (layers 0..2, HEAD), B stage 1 (layers
// 2..4, TAIL). The embedding (group 0, RowLookup) and the LM head (group 3, Head) are tied:
// one tensor record, two groups, one copy on each node.
//
// Tiers per node: 0 is VRAM, 1 is HostPinned and 2 is HostPageable. The two host tiers form
// one NUMA domain's pool, with a pin limit of 1000 and a joint usable of 1000.
//
// Usage per tier with the fixture's numbers:
//   A VRAM:       embedding 10 + layer group 20 + KV 30 + scratch 5 = 65 of 100
//   A HostPinned: expert cache 40                                  = 40 of 1000
//   B VRAM:       layer group 20 + head 10 + KV 30 + scratch 5     = 65 of 100

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

fn pool(node: NodeId, usable: u64, pin_limit: u64) -> HostPoolInfo {
    HostPoolInfo {
        numa: 0,
        tiers: [tier(node, 1), tier(node, 2)],
        usable: Bytes(usable),
        pin_limit: Bytes(pin_limit),
    }
}

struct Bounds {
    n_layers: u16,
    groups: Vec<GroupFacts>,
    usable: Vec<(TierId, TierKind, Bytes)>,
    scratch_tiers: Vec<(DeviceId, TierId)>,
    host_pools: Vec<HostPoolInfo>,
}

impl Bounds {
    fn new() -> Self {
        Bounds {
            n_layers: 4,
            groups: vec![
                group(0, GroupRole::RowLookup, 10, None),
                group(1, GroupRole::Other, 20, None),
                group(2, GroupRole::Other, 20, None),
                group(3, GroupRole::Head, 10, Some(0)),
            ],
            usable: vec![
                (tier(A, 0), TierKind::Vram, Bytes(100)),
                (tier(A, 1), TierKind::HostPinned, Bytes(1000)),
                (tier(A, 2), TierKind::HostPageable, Bytes(1000)),
                (tier(B, 0), TierKind::Vram, Bytes(100)),
                (tier(B, 1), TierKind::HostPinned, Bytes(1000)),
                (tier(B, 2), TierKind::HostPageable, Bytes(1000)),
            ],
            scratch_tiers: vec![(gpu(A), tier(A, 0)), (gpu(B), tier(B, 0))],
            host_pools: vec![pool(A, 1000, 1000), pool(B, 1000, 1000)],
        }
    }

    fn get(&self) -> PlanBounds<'_> {
        PlanBounds {
            n_layers: self.n_layers,
            groups: &self.groups,
            usable: &self.usable,
            scratch_tiers: &self.scratch_tiers,
            host_pools: &self.host_pools,
        }
    }

    fn set_usable(&mut self, t: TierId, bytes: u64) {
        self.usable
            .iter_mut()
            .find(|(id, _, _)| *id == t)
            .unwrap()
            .2 = Bytes(bytes);
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

#[test]
fn rejects_a_tail_role_on_an_earlier_stage() {
    let mut plan = valid_plan();
    plan.stages[0].role = StageRole::HEAD | StageRole::TAIL;
    assert_eq!(
        check(&plan),
        Err(PlanViolation::RoleMismatch {
            stage: StageIx(0),
            role: StageRole::HEAD | StageRole::TAIL,
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
    // RFC-0011 C9's `E_dup · e_p`: the tail is not the head, so B holds its own copy of the
    // tied rows. B uses 65 bytes, of which the head's 10 are the duplicate.
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
    // RFC-0011 C6: a one-stage plan counts the shared rows once.
    // A VRAM holds 10 + 20 + 20 + 30 + 5 = 85.
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

// ---- Tied rows across tiers, MTP lookups, host pools and replicas ------------------------

fn one_stage_plan() -> PlacementPlan {
    let mut plan = valid_plan();
    plan.stages = vec![stage(0, A, 0..4, StageRole::HEAD | StageRole::TAIL)];
    plan.tensors = vec![
        assign(0, A, 1),
        assign(1, A, 0),
        assign(2, A, 0),
        assign(3, A, 0),
    ];
    plan.kv = vec![kv(0, A, 30)];
    plan.scratch = vec![(gpu(A), Bytes(5))];
    plan
}

#[test]
fn counts_tied_rows_once_across_tiers_of_a_one_stage_plan() {
    // RFC-0011 C6: "a one-stage plan counts the shared rows once", whichever tiers the two
    // groups sit on. The embedding is in A's HostPinned, the tied head in A's VRAM; VRAM
    // holds only 20 + 20 + 30 + 5 = 75.
    let mut bounds = Bounds::new();
    bounds.set_usable(tier(A, 0), 75);
    assert_eq!(one_stage_plan().validate(&bounds.get()), Ok(()));
    bounds.set_usable(tier(A, 0), 74);
    assert_eq!(
        one_stage_plan().validate(&bounds.get()),
        Err(PlanViolation::OverCapacity {
            tier: tier(A, 0),
            used: Bytes(75),
            usable: Bytes(74),
        })
    );
}

/// The fixture with an MTP module as layer 4 on the tail. Group 4 is the module's own
/// `RowLookup` group over the embedding record, homed in B's HostPageable.
fn mtp_plan_and_bounds() -> (PlacementPlan, Bounds) {
    let mut plan = valid_plan();
    plan.stages[1].layers = LayerIx(2)..LayerIx(5);
    plan.tensors.push(assign(4, B, 2));
    let mut bounds = Bounds::new();
    bounds.n_layers = 5;
    bounds
        .groups
        .push(group(4, GroupRole::TailLookup, 10, Some(0)));
    (plan, bounds)
}

#[test]
fn accepts_an_mtp_lookup_on_the_tail() {
    let (plan, bounds) = mtp_plan_and_bounds();
    assert_eq!(plan.validate(&bounds.get()), Ok(()));
}

#[test]
fn rejects_an_mtp_lookup_off_the_tail() {
    let (mut plan, bounds) = mtp_plan_and_bounds();
    plan.tensors[4] = assign(4, A, 2);
    assert_eq!(
        plan.validate(&bounds.get()),
        Err(PlanViolation::LookupOffTail {
            group: TensorGroupId(4),
            owner: A,
            tail: B,
        })
    );
}

#[test]
fn charges_an_mtp_lookup_on_a_tail_that_is_not_the_head() {
    // RFC-0011 C9's `E_dup · e_p`: B's HostPageable holds the module's 10 bytes of rows.
    let (plan, mut bounds) = mtp_plan_and_bounds();
    bounds.set_usable(tier(B, 2), 9);
    assert_eq!(
        plan.validate(&bounds.get()),
        Err(PlanViolation::OverCapacity {
            tier: tier(B, 2),
            used: Bytes(10),
            usable: Bytes(9),
        })
    );
}

#[test]
fn counts_an_mtp_lookup_once_in_a_one_stage_plan() {
    // RFC-0011 C6: one stage, so the module's rows are the prologue's and add nothing.
    let mut plan = one_stage_plan();
    plan.stages[0].layers = LayerIx(0)..LayerIx(5);
    plan.tensors.push(assign(4, A, 2));
    let mut bounds = Bounds::new();
    bounds.n_layers = 5;
    bounds
        .groups
        .push(group(4, GroupRole::TailLookup, 10, Some(0)));
    bounds.set_usable(tier(A, 2), 0);
    assert_eq!(plan.validate(&bounds.get()), Ok(()));
}

#[test]
fn rejects_a_host_pool_counted_twice() {
    // 900 bytes HostPinned + 900 bytes HostPageable: each fits its own tier's usable of 1000,
    // but the domain's pool holds only 1000 (RFC-0011 C9's joint host rows).
    let mut plan = valid_plan();
    plan.expert_cache[0].bytes = Bytes(900);
    plan.kv.push(KvBudget {
        tier: tier(A, 2),
        bytes: Bytes(900),
        ..kv(0, A, 0)
    });
    assert_eq!(
        check(&plan),
        Err(PlanViolation::HostPoolOverCapacity {
            tiers: [tier(A, 1), tier(A, 2)],
            used: Bytes(1800),
            usable: Bytes(1000),
        })
    );
}

/// The fixture with 300 bytes in A's HostPinned and 700 in A's HostPageable. The two differ,
/// so a pool sum that counted one tier twice would give 600 or 1400, not 1000.
fn asymmetric_host_plan() -> PlacementPlan {
    let mut plan = valid_plan();
    plan.expert_cache[0].bytes = Bytes(300);
    plan.kv.push(KvBudget {
        tier: tier(A, 2),
        bytes: Bytes(700),
        ..kv(0, A, 0)
    });
    plan
}

#[test]
fn accepts_a_host_pool_at_exact_capacity() {
    let mut bounds = Bounds::new();
    bounds.host_pools[0] = pool(A, 1000, 1000);
    assert_eq!(asymmetric_host_plan().validate(&bounds.get()), Ok(()));
}

#[test]
fn rejects_a_host_pool_one_byte_over_with_unequal_tiers() {
    let mut bounds = Bounds::new();
    bounds.host_pools[0] = pool(A, 999, 1000);
    assert_eq!(
        asymmetric_host_plan().validate(&bounds.get()),
        Err(PlanViolation::HostPoolOverCapacity {
            tiers: [tier(A, 1), tier(A, 2)],
            used: Bytes(1000),
            usable: Bytes(999),
        })
    );
}

#[test]
fn accepts_pinned_bytes_at_the_pin_limit() {
    let mut bounds = Bounds::new();
    bounds.host_pools[0] = pool(A, 1000, 40);
    assert_eq!(valid_plan().validate(&bounds.get()), Ok(()));
}

#[test]
fn rejects_pinned_bytes_one_byte_over_the_pin_limit() {
    let mut bounds = Bounds::new();
    bounds.host_pools[0] = pool(A, 1000, 39);
    assert_eq!(
        valid_plan().validate(&bounds.get()),
        Err(PlanViolation::HostPinOverLimit {
            tier: tier(A, 1),
            used: Bytes(40),
            pin_limit: Bytes(39),
        })
    );
}

#[test]
fn rejects_replicas_it_cannot_charge() {
    let mut plan = valid_plan();
    plan.tensors[2].replicas = smallvec![A];
    assert_eq!(
        check(&plan),
        Err(PlanViolation::ReplicasUnchecked {
            group: TensorGroupId(2)
        })
    );
}

#[test]
fn rejects_a_host_tier_with_no_pool() {
    // A's HostPinned holds the expert cache, but no pool record names A's domain, so the
    // joint rows could not be checked.
    let mut bounds = Bounds::new();
    bounds.host_pools.remove(0);
    assert_eq!(
        valid_plan().validate(&bounds.get()),
        Err(PlanViolation::HostTierWithoutPool { tier: tier(A, 1) })
    );
}

#[test]
fn charges_an_untied_head_in_a_one_stage_plan() {
    // An untied LM head has its own tensor record, so it is charged even when the tail is
    // the head. A VRAM holds 20 + 20 + 10 + 30 + 5 = 85.
    let mut bounds = Bounds::new();
    bounds.groups[3].shared_with = None;
    bounds.set_usable(tier(A, 0), 85);
    assert_eq!(one_stage_plan().validate(&bounds.get()), Ok(()));
    bounds.set_usable(tier(A, 0), 84);
    assert_eq!(
        one_stage_plan().validate(&bounds.get()),
        Err(PlanViolation::OverCapacity {
            tier: tier(A, 0),
            used: Bytes(85),
            usable: Bytes(84),
        })
    );
}

#[test]
fn rejects_shared_with_naming_a_group_not_in_the_census() {
    let mut bounds = Bounds::new();
    bounds.groups[3].shared_with = Some(TensorGroupId(99));
    assert_eq!(
        valid_plan().validate(&bounds.get()),
        Err(PlanViolation::SharedWithUnknown {
            group: TensorGroupId(3),
            twin: TensorGroupId(99),
        })
    );
}

#[test]
fn rejects_a_head_sharing_rows_with_a_non_lookup_group() {
    let mut bounds = Bounds::new();
    bounds.groups[3].shared_with = Some(TensorGroupId(1));
    assert_eq!(
        valid_plan().validate(&bounds.get()),
        Err(PlanViolation::SharedWithNotLookup {
            group: TensorGroupId(3),
            twin: TensorGroupId(1),
        })
    );
}

#[test]
fn rejects_an_mtp_lookup_sharing_rows_with_the_head() {
    let (plan, mut bounds) = mtp_plan_and_bounds();
    bounds.groups[4].shared_with = Some(TensorGroupId(3));
    assert_eq!(
        plan.validate(&bounds.get()),
        Err(PlanViolation::SharedWithNotLookup {
            group: TensorGroupId(4),
            twin: TensorGroupId(3),
        })
    );
}
