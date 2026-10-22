// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stages of the ring (RFC-0001 §5.4): each stage's node, layer range, role, parallelism
//! mode and outbound hop, with RFC-0032 §5.1's split types.

use std::ops::Range;

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ids::{DeviceId, LayerIx, LinkId, NodeId, StageIx};
use crate::resource::TransportKind;
use crate::units::Nanos;

/// One stage of the ring (RFC-0001 §5.4). Owner: RFC-0011.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageAssignment {
    /// The stage's position in the ring; equal to its index in `PlacementPlan::stages`.
    pub stage: StageIx,
    /// The node that owns the stage's ring endpoints. In the cross-node modes this is the
    /// primary member of the fast island.
    pub node: NodeId,
    /// The stage's decoder layers: contiguous and half-open.
    pub layers: Range<LayerIx>,
    /// Head, tail and drafting roles.
    pub role: StageRole,
    /// How the stage's layers are split across devices and nodes.
    pub mode: ParallelismMode,
    /// Outbound hop to `stages[(stage + 1) % len]`.
    pub next_hop: HopPlan,
    /// The evaluator's predicted step time for the stage.
    pub predicted_step: Nanos,
}

bitflags! {
    /// The roles a stage plays in the ring (RFC-0001 §5.4). Owner: RFC-0011.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct StageRole: u8 {
        /// Holds the embeddings and receives sequences from ingress: `stages[0]`.
        const HEAD  = 1 << 0;
        /// Holds the LM head and the sampler: the last stage.
        const TAIL  = 1 << 1;
        /// Runs the drafter.
        const DRAFT = 1 << 2;
    }
}

/// How a stage's layers are split (RFC-0001 §5.4). Owner: RFC-0032 (cross-node modes),
/// RFC-0012 (`Local` and `IntraNodeTp` execution).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelismMode {
    /// No intra-layer split. Each layer runs on the compute sites its `TierAssignment`s name,
    /// in layer order; inside a MoE layer, cache hits run on the GPU while misses run in place
    /// on the CPU. The default.
    Local,
    /// Tensor parallel across GPUs of this node over NVLink or PCIe P2P.
    IntraNodeTp {
        /// The participating devices, in member order.
        devices: SmallVec<[DeviceId; 8]>,
        /// How each layer's work is divided.
        split: TpSplit,
    },
    /// Tensor parallel across a fast island of nodes (ADR-0006).
    CrossNodeTp {
        /// The island's members; `members[0]` is the primary.
        members: SmallVec<[StageMember; 4]>,
        /// How each layer's work is divided.
        split: TpSplit,
        /// The all-reduce algorithm.
        collective: CollectiveAlgo,
    },
    /// Expert parallel across a fast island: attention on the primary, each expert on its
    /// owner. Same gate as `CrossNodeTp`.
    CrossNodeEp {
        /// The island's members; `members[0]` is the primary.
        members: SmallVec<[StageMember; 4]>,
        /// Which member owns each expert.
        owners: ExpertOwnerMap,
    },
}

/// One member of a fast island (RFC-0001 §5.4). Owner: RFC-0032.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageMember {
    /// The member node.
    pub node: NodeId,
    /// Its participating devices.
    pub devices: SmallVec<[DeviceId; 4]>,
    /// The link from the primary to this member.
    pub link: LinkId,
}

/// How a layer's work is divided among members, in member order (RFC-0032 §5.1).
///
/// A split may be uneven, weighted by each member's `W_eff`, at the legal granularity the family
/// declares: whole KV-head groups for attention, a tile multiple of the quant block for FFN
/// columns (ADR-0006 item 2; RFC-0043). Checking a split against a family's granularities is
/// DIST-E13-T01's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpSplit {
    /// KV-head groups per member (GQA), or query heads per member (MLA).
    pub attn: SmallVec<[u16; 8]>,
    /// FFN columns per member, in units of the family's tile (RFC-0043).
    pub ffn: SmallVec<[u32; 8]>,
    /// Tensor parallel inside experts: columns per member per expert.
    pub experts: Option<SmallVec<[u32; 8]>>,
}

/// The all-reduce algorithm of a tensor-parallel island (RFC-0032 §5.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollectiveAlgo {
    /// One-shot all-reduce.
    OneShot,
    /// Ring all-reduce.
    Ring,
}

/// Expert owner per (MoE layer, expert) for `CrossNodeEp`; values index `members`
/// (RFC-0032 §5.1).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertOwnerMap {
    /// Per MoE layer, the owning member of each expert, by expert index.
    pub layers: Vec<(LayerIx, Vec<u8>)>,
}

/// A stage's outbound hop (RFC-0001 §5.4). Owner: RFC-0032, which adds its §3.3 fields with
/// serde defaults.
///
/// A self-describing format such as JSON decodes a hop written without the §3.3 fields: both
/// options come back `None` and `predicted_p99` comes back 0. The IPC channel's encoding,
/// `postcard`, is positional (RFC-0001 §3.5), so there the added fields are an `ipc_protocol`
/// change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HopPlan {
    /// The link the hop uses.
    pub link: LinkId,
    /// The transport on that link.
    pub transport: TransportKind,
    /// Activation dtype on the wire. Exact mode: `F32` by default; `BF16` only where the
    /// conformance manifest holds a `WireCert` for the family and phase (ADR-0019 decision 5).
    pub wire: WireDType,
    /// α + S/β at batch 1.
    pub predicted: Nanos,
    /// The (link, transport) of the prefill lane (RFC-0032 §3.2), when its cost at S_pf beats
    /// the decode transport by more than 2× (RFC-0031 §3). Decode never moves to it. `None`:
    /// the prefill lane uses the decode lane's transport.
    #[serde(default)]
    pub prefill_transport: Option<(LinkId, TransportKind)>,
    /// Token-return hops only: forward through an intermediate node's runner (RFC-0032 §3.4).
    #[serde(default)]
    pub relay: Option<RelayVia>,
    /// Predicted p99 of α + S/β at batch 1; the drift detector compares the measured p99 with
    /// it. 0, the serde default, means the plan carries no tail prediction for the hop: a plan
    /// written before this field existed has none, and inventing one from `predicted` would
    /// claim a tail the solver never costed.
    #[serde(default)]
    pub predicted_p99: Nanos,
}

/// The intermediate node a relayed token-return hop forwards through (RFC-0032 §3.3, §3.4).
/// Owner: RFC-0032.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RelayVia {
    /// The node whose runner copies each frame across.
    pub node: NodeId,
    /// The link from the sending stage to `node`.
    pub first: LinkId,
    /// The link from `node` to the receiving stage.
    pub second: LinkId,
    /// The transport on both links.
    pub transport: TransportKind,
}

/// Activation dtype on the wire (RFC-0031 §6.4, which gives each its frame code). Owner:
/// RFC-0031; RFC-0042 owns the numerics of the compressed codes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WireDType {
    /// IEEE binary32. Exact. The serde default.
    #[default]
    F32,
    /// bfloat16. Exact only where the reference holds BF16 at that boundary.
    BF16,
    /// IEEE binary16. Needs `WIRE_COMPRESS`.
    F16,
    /// Per 32 elements: an F16 scale, then 32 × i8. Needs `WIRE_COMPRESS`.
    Q8B32,
    /// Per 32 elements: an F16 scale, then 32 × E4M3. Needs `WIRE_COMPRESS`.
    F8E4M3B32,
}
