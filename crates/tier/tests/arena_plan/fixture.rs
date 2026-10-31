// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builders for the plans, censuses and profiles the `plan_arenas` tests use.
//!
//! One node layout serves every test. Devices: 0 is a GPU, 1 the CPU of NUMA domain 0. Tiers:
//! 0 is the GPU's `Vram`, 1 `HostPinned` and 2 `HostPageable` of NUMA domain 0, 3 an `Nvme`
//! volume. A test that needs a second node names [`OTHER`]'s tiers, which the profile does not
//! list.

use std::ops::Range;

use smallvec::smallvec;
use syntropy_core::plan::{
    BatchingLimits, CachePolicy, ComputeSite, ExpertCacheBudget, HopPlan, KvBudget, KvDType,
    MissSplit, MissSplitMode, ParallelismMode, PlanEpoch, PredictedCost, SpillCompute,
    StageAssignment, StageRole, StreamSite, TierAssignment, WireDType,
};
use syntropy_core::policy::{ExactnessPolicy, Objective};
use syntropy_core::resource::{
    CapabilityProfile, CapabilityProfileExt, NodeInventory, TierInfo, TierInfoExt, TierKind,
    TransportKind,
};
use syntropy_core::{
    Blake3, Bytes, DeviceId, Endpoint, ExpertIx, GraphVersion, InstanceId, LayerIx, LinkId,
    LinkKind, ModelId, Nanos, NetMedium, NodeId, PlacementPlan, StageIx, TensorGroupId, TierId,
};
use syntropy_manifest::census::{ClassTotals, TensorCensus, TensorClass, TensorGroup};

/// The runner's node.
pub const NODE: NodeId = NodeId([0x11; 32]);
/// Another node of the cluster.
pub const OTHER: NodeId = NodeId([0x22; 32]);

pub const VRAM: u16 = 0;
pub const PINNED: u16 = 1;
pub const PAGEABLE: u16 = 2;
pub const NVME: u16 = 3;

pub const GB: u64 = 1_000_000_000;
pub const MB: u64 = 1_000_000;
pub const MIB: u64 = 1 << 20;

pub fn tier(ordinal: u16) -> TierId {
    TierId {
        node: NODE,
        ordinal,
    }
}

pub fn other_tier(ordinal: u16) -> TierId {
    TierId {
        node: OTHER,
        ordinal,
    }
}

pub fn gpu() -> DeviceId {
    DeviceId {
        node: NODE,
        ordinal: 0,
    }
}

pub fn cpu() -> DeviceId {
    DeviceId {
        node: NODE,
        ordinal: 1,
    }
}

fn tier_info(ordinal: u16, kind: TierKind, affinity: &[DeviceId], numa: Option<u16>) -> TierInfo {
    TierInfo {
        id: tier(ordinal),
        kind,
        capacity: Bytes(1 << 40),
        usable: Bytes(1 << 40),
        affinity: affinity.iter().copied().collect(),
        pin_limit: None,
        volume: None,
        ext: TierInfoExt {
            numa,
            ..TierInfoExt::default()
        },
    }
}

/// The runner node's profile: the four tiers in its inventory.
pub fn profile() -> CapabilityProfile {
    let tiers = vec![
        tier_info(VRAM, TierKind::Vram, &[gpu()], None),
        tier_info(PINNED, TierKind::HostPinned, &[cpu(), gpu()], Some(0)),
        tier_info(PAGEABLE, TierKind::HostPageable, &[cpu()], Some(0)),
        tier_info(NVME, TierKind::Nvme, &[], None),
    ];
    CapabilityProfile {
        node: NODE,
        version: 1,
        devices: Vec::new(),
        storage: Vec::new(),
        links: Vec::new(),
        ext: CapabilityProfileExt {
            inventory: Some(NodeInventory {
                devices: Vec::new(),
                tiers,
                links: Vec::new(),
                host_pools: Vec::new(),
            }),
            ..CapabilityProfileExt::default()
        },
    }
}

/// A census of `groups`, each given its index as its ID.
pub fn census(groups: Vec<(TensorClass, Option<u16>, u64)>) -> TensorCensus {
    let groups: Vec<TensorGroup> = groups
        .into_iter()
        .enumerate()
        .map(|(i, (class, layer, bytes))| TensorGroup {
            id: TensorGroupId(u32::try_from(i).unwrap()),
            class,
            layer: layer.map(LayerIx),
            tensors: Vec::new(),
            bytes: Bytes(bytes),
            read_per_token: Bytes(bytes),
            shared_with: None,
        })
        .collect();
    TensorCensus {
        tensors: Vec::new(),
        totals: ClassTotals::of(&groups),
        groups,
    }
}

pub fn expert(layer: u16, id: u16) -> TensorClass {
    TensorClass::Expert {
        layer: LayerIx(layer),
        id: ExpertIx(id),
    }
}

pub fn stage(ix: u16, node: NodeId, layers: Range<u16>, role: StageRole) -> StageAssignment {
    StageAssignment {
        stage: StageIx(ix),
        node,
        layers: LayerIx(layers.start)..LayerIx(layers.end),
        role,
        mode: ParallelismMode::Local,
        next_hop: HopPlan {
            link: LinkId {
                src: Endpoint::Device(gpu()),
                dst: Endpoint::Device(gpu()),
                kind: LinkKind::Network(NetMedium::Ethernet),
                path: 0,
            },
            transport: TransportKind::Quic,
            wire: WireDType::F32,
            predicted: Nanos(1_000),
            prefill_transport: None,
            relay: None,
            predicted_p99: Nanos(1_000),
        },
        predicted_step: Nanos(10_000),
    }
}

/// Group `group` homed on `home`, computed by `decode`, streamed to the GPU when `stream`.
pub fn assign(group: u32, home: TierId, decode: DeviceId, stream: bool) -> TierAssignment {
    TierAssignment {
        group: TensorGroupId(group),
        owner: home.node,
        home,
        compute: ComputeSite {
            decode,
            batched: stream.then(|| StreamSite {
                device: gpu(),
                via: LinkId {
                    src: Endpoint::Tier(home),
                    dst: Endpoint::Device(gpu()),
                    kind: LinkKind::Pcie,
                    path: 0,
                },
                min_tokens: 12,
            }),
        },
        replicas: smallvec![],
    }
}

pub fn cache(device: DeviceId, tier: TierId, bytes: u64) -> ExpertCacheBudget {
    ExpertCacheBudget {
        device,
        tier,
        bytes: Bytes(bytes),
        per_layer: None,
        policy: CachePolicy::Lru,
        miss_split: MissSplit {
            copied_share: 0.26,
            mode: MissSplitMode::Fixed,
        },
    }
}

pub fn kv(stage: u16, tier: TierId, bytes: u64, spill: Option<TierId>) -> KvBudget {
    KvBudget {
        stage: StageIx(stage),
        tier,
        bytes: Bytes(bytes),
        dtype: KvDType::Native,
        spill,
        spill_compute: SpillCompute::GpuStream,
    }
}

/// A plan with these stages and nothing else.
pub fn plan(stages: Vec<StageAssignment>) -> PlacementPlan {
    PlacementPlan {
        epoch: PlanEpoch { term: 1, seq: 1 },
        instance: InstanceId(1),
        model: ModelId(Blake3([1; 32])),
        graph: GraphVersion(1),
        policy: Blake3([2; 32]),
        objective: Objective::default(),
        exactness: ExactnessPolicy::default(),
        stages,
        tensors: Vec::new(),
        expert_cache: Vec::new(),
        kv: Vec::new(),
        prefetch: Vec::new(),
        scratch: Vec::new(),
        speculative: None,
        batching: BatchingLimits::default(),
        predicted: PredictedCost::default(),
        assumptions: Vec::new(),
        ext: Default::default(),
    }
}
