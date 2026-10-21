// SPDX-License-Identifier: MIT OR Apache-2.0

//! Arena layout: purposes, pin classes, registrations, slot geometry and the per-runner
//! [`ArenaPlan`] (RFC-0020 §2, §3). Construction per tier kind, segment mapping and the slot
//! allocators come in later tasks (RFC-0020 §4, §8; TIER-E1-T07).
//!
//! This is one of the two modules where RFC-0020 §1 allows `unsafe`. It holds none yet, so it
//! keeps the crate-level `deny(unsafe_code)`; the task that adds the first `unsafe` block lifts
//! it here with a scoped `allow`.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::{Bytes, DeviceId, ExtentRef, LayerIx, TensorGroupId, TierId};

/// Why an arena exists. There is one budget per (tier, purpose), so one purpose cannot starve
/// another (RFC-0020 §2, RFC-0001 §9.1).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Purpose {
    /// Weights of the groups homed on the tier, one extent per group.
    Weights,
    /// The adaptive cache of MoE experts (RFC-0023).
    ExpertCache,
    /// Copy, prefetch and demand staging slots for experts; fixed per epoch (RFC-0023 §6).
    ExpertStaging,
    /// KV pages (RFC-0022).
    Kv,
    /// Step scratch; bump-allocated and frozen at `Warming`.
    Scratch,
    /// Captured graphs, owned by the provider inside the reservation.
    Graphs,
    /// The I/O engine's staging ring (RFC-0021).
    IoStaging,
    /// Network buffer pools (RFC-0031).
    NetPool,
    /// Cached embedding rows of `RowLookup` groups homed on `Nvme` (RFC-0023).
    RowCache,
    /// The activation-history store (RFC-0033 §4.1).
    History,
}

/// How a host arena is held in memory (RFC-0020 §2, "Pin classes").
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PinClass {
    /// Locked and registered with every provider that DMAs from it. `HostPinned` tier.
    DmaPinned,
    /// Locked against paging but not DMA-registered; read by the CPU only. `HostPageable` tier.
    Locked,
    /// Anonymous, never file-backed, not locked; read by the CPU only. `HostPageable` tier.
    Pageable,
}

bitflags! {
    /// What an arena must be registered with before `Loading` ends (RFC-0020 §3).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct Registrations: u8 {
        /// `KernelProvider::register_host` on every GPU in [`ArenaSpec::dma_devices`].
        const GPU_DMA    = 1 << 0;
        /// Also mapped into the GPU address space (doorbells, partial outputs).
        const GPU_MAPPED = 1 << 1;
        /// io_uring or IoRing registered buffers, at most 1 GiB per buffer (ADR-0013).
        const IO_FIXED   = 1 << 2;
        /// `Transport::register` (RFC-0031).
        const RDMA_MR    = 1 << 3;
        /// `MTLResidencySet` or `useHeap` on Apple.
        const RESIDENCY  = 1 << 4;
    }
}

/// How an arena is divided into slots (RFC-0020 §3).
///
/// `Classes` keeps up to 64 classes inline, as RFC-0020 §3 specifies: one class per MoE layer
/// fits without a heap allocation, and the value is built once per arena, not per step.
#[non_exhaustive]
#[expect(
    clippy::large_enum_variant,
    reason = "RFC-0020 §3 keeps 64 slot classes inline; one value per arena"
)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotGeometry {
    /// One slot per tensor group, laid out in plan order, each start aligned to `align`.
    Extents {
        /// Alignment of each slot's start.
        align: Bytes,
    },
    /// Fixed-size slots per class; class i holds `count` slots of `size` bytes. `ExpertCache`
    /// uses one class per MoE layer (RFC-0023).
    Classes(SmallVec<[SlotClass; 64]>),
    /// Fixed-size pages (KV, RFC-0022).
    Pages {
        /// Bytes per page.
        page: Bytes,
        /// Number of pages.
        count: u32,
    },
    /// A ring of equal slots (staging, network pools).
    Ring {
        /// Bytes per slot.
        slot: Bytes,
        /// Number of slots.
        count: u32,
    },
    /// Bump allocation, frozen at `Warming` (scratch).
    Bump,
}

/// One slot class of [`SlotGeometry::Classes`] (RFC-0020 §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotClass {
    /// Bytes per slot.
    pub size: Bytes,
    /// Number of slots.
    pub count: u32,
    /// The MoE layer the class serves, for `ExpertCache` classes. Defaults to `None`.
    #[serde(default)]
    pub layer: Option<LayerIx>,
}

/// One arena: a contiguous reservation inside one tier for one purpose (RFC-0020 §2, §3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArenaSpec {
    /// The tier the arena lives in.
    pub tier: TierId,
    /// Why the arena exists.
    pub purpose: Purpose,
    /// Initial size.
    pub bytes: Bytes,
    /// Reservation size; greater than `bytes` only for elastic arenas (RFC-0020 §8.3).
    pub max_bytes: Bytes,
    /// How the arena is divided into slots.
    pub geometry: SlotGeometry,
    /// How the arena is held in memory.
    pub pin: PinClass,
    /// The NUMA node the pages are bound to before first touch. Defaults to `None` (unbound).
    #[serde(default)]
    pub numa: Option<u16>,
    /// What the arena is registered with before `Loading` ends. Defaults to none.
    #[serde(default)]
    pub registrations: Registrations,
    /// The GPUs that DMA from the arena. Defaults to none.
    #[serde(default)]
    pub dma_devices: SmallVec<[DeviceId; 4]>,
    /// Order in which this arena is pinned (RFC-0020 §6); lower first.
    pub pin_priority: u16,
    /// Present for elastic arenas, built from segments that can be mapped and unmapped
    /// (RFC-0020 §8.3). Defaults to `None` (fixed).
    #[serde(default)]
    pub elastic: Option<SegmentSpec>,
}

/// Segment layout of an elastic arena (RFC-0020 §3, §8.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentSpec {
    /// Bytes per segment.
    pub segment: Bytes,
    /// Segments the arena never shrinks below.
    pub min_segments: u32,
}

/// The tier manager's whole layout for one runner, derived from the plan before any byte is
/// allocated (RFC-0020 §3). `TierManager::plan_arenas` (TIER-E1-T02) builds it as a pure
/// function of (plan, stages, census, profile, config).
///
/// Every field defaults (container `#[serde(default)]`), so a record that predates a field
/// still decodes, in map form and, for trailing fields, in sequence form.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ArenaPlan {
    /// Every arena of the runner.
    pub arenas: Vec<ArenaSpec>,
    /// The extent of every group homed on `Nvme`, which has no arena (RFC-0020 §2). The
    /// pack sidecar gives them (RFC-0041); `ExtentRef` is `syntropy-core`'s (RFC-0020 §3).
    pub nvme_extents: Vec<(TensorGroupId, ExtentRef)>,
    /// One budget line per tier, for the ledger check and SC-11.
    pub per_tier: Vec<(TierId, TierBudgetLine)>,
}

/// One tier's budget, split by purpose (RFC-0020 §3).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TierBudgetLine {
    /// Bytes reserved on the tier.
    pub reserved: Bytes,
    /// Reserved bytes per purpose.
    pub by_purpose: SmallVec<[(Purpose, Bytes); 9]>,
    /// What the tier keeps free.
    pub headroom: Bytes,
}

/// Names one arena: the shard (the owning device's ordinal) and the arena's index in it
/// (RFC-0020 §3, supporting types).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ArenaRef {
    /// The shard, which is the owning device's ordinal.
    pub shard: u16,
    /// The arena's index within the shard.
    pub arena: u8,
}
