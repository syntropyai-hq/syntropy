// SPDX-License-Identifier: MIT OR Apache-2.0

//! `plan_arenas`: the runner's [`ArenaPlan`] as a pure function of (plan, stages, census,
//! profile, config) (RFC-0020 §2, §3; TIER-E1-T02).
//!
//! Nothing here allocates, reads the clock or iterates a hash map. Every map is a `BTreeMap`,
//! and the arenas come out sorted by (tier, purpose), so two calls on equal inputs return equal
//! plans whose serialised forms are byte-equal.
//!
//! # What the function builds
//!
//! The runner's node is `caps.node`; tier kinds come from `caps.ext.inventory`. A group is the
//! runner's when its home tier is on the runner's node and its layer is in one of `stages`'
//! layer ranges. A group without a layer belongs to the head stage, or to the tail stage when
//! it is a `Head` group.
//!
//! | Purpose | Arenas | Size | Pin class |
//! |---|---|---|---|
//! | `Weights` | one per home tier other than `Nvme` | Σ over the runner's groups homed there of the group's bytes rounded up to [`TierConfig::extent_align`] | by tier kind, below |
//! | `ExpertCache` | one per `ExpertCacheBudget` on the node | `ExpertCacheBudget::bytes`, one slot class per MoE layer | by tier kind; elastic |
//! | `ExpertStaging` | one per `Vram` tier with an expert cache; one per `HostPinned` tier that stages for those GPUs | [`TierConfig::expert_staging`] per GPU, [`TierConfig::expert_staging_host`] per GPU | by tier kind |
//! | `Kv` | one per tier holding the runner's stages' `KvBudget`s | Σ `KvBudget::bytes` | by tier kind |
//! | `Scratch` | one per tier holding a device's `plan.scratch` | Σ of the devices' scratch | by tier kind |
//! | `IoStaging` | one per `HostPinned` tier that stages for a GPU holding an arena | `io.stage_slots × io.stage_slot` per GPU | `DmaPinned` |
//!
//! Pin classes follow the tier kind: `HostPinned` arenas are `DmaPinned`; `HostPageable`
//! arenas are `Locked`, attempted, falling back to `Pageable` at run time (RFC-0020 §4.2);
//! `Vram` and `Unified` arenas are `Locked`, never `DmaPinned`, so `DmaPinned` always means a
//! registered host range (RFC-0020 §3, note of 2026-10-31).
//!
//! A `HostPinned` arena that a GPU copies from carries `GPU_DMA` with that GPU in
//! `dma_devices`. The GPUs that copy from a tier's `Weights` are the `StreamSite` devices of
//! its groups and, for expert groups, the devices of the node's `Vram` expert caches
//! (admissions and copied misses, RFC-0023 §3). `IO_FIXED` marks the arenas RFC-0020 §4.2
//! lists: `IoStaging`, the host `ExpertCache` and `ExpertStaging` over an `Nvme` home, and host
//! KV pages.
//!
//! # Plan bytes per tier
//!
//! A tier's *plan bytes* are the bytes the plan puts on it for this runner: the bytes of each
//! group homed there, each rounded up to [`TierConfig::extent_align`], plus the `bytes` of every
//! `ExpertCacheBudget`, `KvBudget` and scratch entry on the tier, plus the staging sizes from
//! [`TierConfig`]. For every tier, the sum of its arenas' `max_bytes` equals its plan bytes,
//! and its [`TierBudgetLine::reserved`] is that sum. `Nvme` has no arena and no budget line.
//!
//! # Left for later tasks
//!
//! - **The multi-site layout check** (RFC-0020 §7.3) and its `AcceptedLayouts` argument:
//!   TIER-E1-T12 (RFC-0020 §3, note of 2026-10-31).
//! - **`nvme_extents`** stays empty. Its extents come from the pack sidecar (RFC-0041,
//!   COMP-E11-T06), which is not an input here; the census holds only source extents, which
//!   are not the pack's. Nothing is homed on `Nvme` in M0.
//! - **`Graphs`**: sized from the provider's capture (COMP-E4-T05).
//! - **`NetPool`**: RFC-0031's pool sizing (TIER-E2-T21).
//! - **`RowCache`**: cached rows of `RowLookup` groups homed on `Nvme` (RFC-0020 §2). No task
//!   in the plan builds this arena yet; issue #2050 tracks it.
//! - **`History`**: `recovery.history_ram` (TIER-E2-T22).
//! - **The VRAM headroom's display rule** and the `tier.*` keys: [`TierConfig`] holds plain
//!   values until TIER-E1-T09 registers the keys.
//! - **Provider allocation granularity**: segment sizes are rounded to
//!   [`TierConfig::extent_align`] here; the device build rounds them up to the provider's
//!   granularity (RFC-0020 §8.2; TIER-E1-T07).
//! - **KV page geometry**: the `Kv` arena is cut in [`TierConfig::kv_page`] units; RFC-0022's
//!   per-class pools are carved from it by TIER-E6-T11.
//! - **Pin priority 2's order** inside a `Weights` arena (experts outside the initial VRAM
//!   fill first, by profile frequency) needs the expert profile (TIER-E8-T07).

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;

use smallvec::SmallVec;
use syntropy_core::plan::StageRole;
use syntropy_core::resource::{CapabilityProfile, TierInfo, TierKind};
use syntropy_core::{Bytes, DeviceId, LayerIx, PlacementPlan, StageIx, TensorGroupId, TierId};
use syntropy_manifest::census::{TensorCensus, TensorClass};

use super::{
    ArenaPlan, ArenaSpec, PinClass, Purpose, Registrations, SegmentSpec, SlotClass, SlotGeometry,
    TierBudgetLine,
};

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// Pin priorities (RFC-0020 §6, "Order"). Lower is pinned first. Arenas that are not pinned
/// (`Vram`, `Unified`) carry [`PRIO_FIRST`].
const PRIO_FIRST: u16 = 0;
/// `ExpertCache` and `ExpertStaging` slots in `HostPinned`.
const PRIO_EXPERT_SLOTS: u16 = 1;
/// `DmaPinned` weight ranges.
const PRIO_DMA_WEIGHTS: u16 = 2;
/// `Kv` pages in host memory.
const PRIO_KV: u16 = 3;
/// `Locked` ranges of `HostPageable` arenas.
const PRIO_LOCKED: u16 = 4;

/// The values `plan_arenas` reads that the plan does not carry (RFC-0020 §14; RFC-0021 §14).
///
/// A plain struct until TIER-E1-T09 registers the `tier.*` keys in RFC-0054's schema. Every
/// byte size is a \[target\], the RFCs' default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierConfig {
    /// Alignment of each `Weights` extent and each expert slot: 4 KiB, the direct-I/O
    /// alignment of pack extents (RFC-0021 §6, RFC-0023 §4).
    pub extent_align: Bytes,
    /// `resource.vram_headroom`: VRAM left for other programs (RFC-0020 §8.1, §14). The default
    /// is 256 MiB, the value for a GPU that drives no display; the caller passes 512 MiB for a
    /// display GPU.
    pub vram_headroom: Bytes,
    /// `tier.vram_segment`: the smallest elastic segment of a device `ExpertCache` (RFC-0020
    /// §8.2). Default 256 MiB.
    pub vram_segment: Bytes,
    /// `tier.pin_step`: the largest pinning step, and the segment size of a host `ExpertCache`
    /// before rounding down to whole slots (RFC-0020 §6.1). Default 1 GiB.
    pub pin_step: Bytes,
    /// The `Kv` arena's page unit (RFC-0020 §3, `SlotGeometry::Pages`). Default 2 MiB.
    pub kv_page: Bytes,
    /// `io.stage_slot`: one slot of the load staging ring (RFC-0021 §8, §14). Default 64 MiB.
    pub io_stage_slot: Bytes,
    /// `io.stage_slots`: slots of the load staging ring per GPU; 0 means no `IoStaging` arena
    /// (RFC-0021 §8, §14). Default 4.
    pub io_stage_slots: u32,
    /// `ExpertStaging` bytes in VRAM per GPU with a `Vram` expert cache; 0 means absent.
    /// RFC-0023 §4 sizes it; until then the caller passes it. Default 0.
    pub expert_staging: Bytes,
    /// `ExpertStaging` bytes in `HostPinned` per GPU with a `Vram` expert cache, used when the
    /// runner has experts homed on `Nvme` or a `HostPageable` expert cache (RFC-0023 §3); 0
    /// means absent. Default 0.
    pub expert_staging_host: Bytes,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            extent_align: Bytes(4 * KIB),
            vram_headroom: Bytes(256 * MIB),
            vram_segment: Bytes(256 * MIB),
            pin_step: Bytes(GIB),
            kv_page: Bytes(2 * MIB),
            io_stage_slot: Bytes(64 * MIB),
            io_stage_slots: 4,
            expert_staging: Bytes(0),
            expert_staging_host: Bytes(0),
        }
    }
}

/// Why `plan_arenas` refused its inputs.
///
/// Every variant is an inconsistency between the plan, the census, the profile and the
/// config: a solver or runner bug, class `Internal`. RFC-0020 §15 publishes no code for it,
/// so it is a typed error, not a `syntropy_core::Error` (RFC-0020 §3, note of 2026-10-31).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArenaPlanError {
    /// The profile carries no node inventory, so no tier has a kind.
    NoInventory,
    /// The plan names a tier of this node that the inventory does not list.
    UnknownTier(TierId),
    /// The plan names a group the census does not hold at its index.
    UnknownGroup(TensorGroupId),
    /// A runner stage is not in the plan, or not at its index.
    UnknownStage(StageIx),
    /// A budget sits on a tier kind its purpose cannot use (RFC-0020 §2's table).
    PurposeTier {
        /// The purpose.
        purpose: Purpose,
        /// The tier.
        tier: TierId,
        /// The tier's kind.
        kind: TierKind,
    },
    /// A group the plan streams to a GPU is homed on `HostPageable`, which is never a DMA
    /// source (RFC-0020 §2, RFC-0001 §9.4).
    DmaFromPageable(TensorGroupId),
    /// An expert cache's `per_layer` list does not have one entry per MoE layer of the runner,
    /// or sums to more than the cache's bytes.
    PerLayer {
        /// The cache's tier.
        tier: TierId,
        /// MoE layers of the runner.
        layers: usize,
        /// Entries in `per_layer`.
        entries: usize,
    },
    /// An expert cache has bytes but the runner has no expert group with bytes to size its
    /// slots.
    NoExperts(TierId),
    /// A device has scratch but no tier to hold it.
    NoScratchTier(DeviceId),
    /// A GPU needs host staging but the node has no `HostPinned` tier.
    NoHostPinned(DeviceId),
    /// A size or slot count does not fit its type.
    Overflow,
}

impl fmt::Display for ArenaPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoInventory => write!(f, "the capability profile carries no node inventory"),
            Self::UnknownTier(t) => write!(f, "tier {} is not in the inventory", t.ordinal),
            Self::UnknownGroup(g) => write!(f, "group {} is not in the census", g.0),
            Self::UnknownStage(s) => write!(f, "stage {} is not in the plan", s.0),
            Self::PurposeTier {
                purpose,
                tier,
                kind,
            } => write!(
                f,
                "a {purpose:?} budget sits on tier {}, a {kind:?} tier",
                tier.ordinal
            ),
            Self::DmaFromPageable(g) => {
                write!(f, "group {} streams to a GPU from a HostPageable home", g.0)
            }
            Self::PerLayer {
                tier,
                layers,
                entries,
            } => write!(
                f,
                "the expert cache on tier {} has {entries} per-layer budgets for {layers} MoE \
                 layers, or they exceed its bytes",
                tier.ordinal
            ),
            Self::NoExperts(t) => write!(
                f,
                "the expert cache on tier {} has bytes but the runner has no expert group",
                t.ordinal
            ),
            Self::NoScratchTier(d) => write!(f, "device {} has no tier for scratch", d.ordinal),
            Self::NoHostPinned(d) => {
                write!(
                    f,
                    "GPU {} needs host staging but there is no HostPinned tier",
                    d.ordinal
                )
            }
            Self::Overflow => write!(f, "an arena size or slot count overflows"),
        }
    }
}

impl std::error::Error for ArenaPlanError {}

/// The runner's arena layout (RFC-0020 §3). See the module docs for the rules.
///
/// # Errors
///
/// [`ArenaPlanError`] when the inputs disagree; see each variant.
pub fn plan_arenas(
    plan: &PlacementPlan,
    stages: &[StageIx],
    census: &TensorCensus,
    caps: &CapabilityProfile,
    cfg: &TierConfig,
) -> Result<ArenaPlan, ArenaPlanError> {
    let ctx = Ctx::new(plan, stages, caps)?;
    let mut b = Builder::default();

    // Expert caches on the node, and the GPUs whose VRAM caches copy from host memory.
    let mut caches = Vec::new();
    for c in &plan.expert_cache {
        if c.tier.node != ctx.node {
            continue;
        }
        let kind = ctx.kind(c.tier)?;
        if kind == TierKind::Nvme {
            return Err(ArenaPlanError::PurposeTier {
                purpose: Purpose::ExpertCache,
                tier: c.tier,
                kind,
            });
        }
        caches.push((c, kind));
    }
    let copy_gpus: BTreeSet<DeviceId> = caches
        .iter()
        .filter(|(_, k)| *k == TierKind::Vram)
        .map(|(c, _)| c.device)
        .collect();

    // Weights, and the slot size of each MoE layer.
    let mut weights: BTreeMap<TierId, (Bytes, BTreeSet<DeviceId>)> = BTreeMap::new();
    let mut slot_of_layer: BTreeMap<LayerIx, Bytes> = BTreeMap::new();
    let mut nvme_experts = false;
    for t in &plan.tensors {
        if t.home.node != ctx.node {
            continue;
        }
        let g = census
            .groups
            .get(t.group.0 as usize)
            .filter(|g| g.id == t.group)
            .ok_or(ArenaPlanError::UnknownGroup(t.group))?;
        if !ctx.owns(g.layer, &g.class) {
            continue;
        }
        let kind = ctx.kind(t.home)?;
        let size = align_up(g.bytes, cfg.extent_align)?;
        let is_expert = matches!(g.class, TensorClass::Expert { .. });
        if let (true, Some(l)) = (is_expert, g.layer) {
            let s = slot_of_layer.entry(l).or_default();
            *s = (*s).max(size);
        }
        if kind == TierKind::Nvme {
            nvme_experts |= is_expert;
            continue;
        }
        let streamed = t.compute.batched.map(|s| s.device);
        if kind == TierKind::HostPageable && streamed.is_some() {
            return Err(ArenaPlanError::DmaFromPageable(t.group));
        }
        let (bytes, dma) = weights.entry(t.home).or_default();
        *bytes = add(*bytes, size)?;
        if kind == TierKind::HostPinned {
            dma.extend(streamed);
            if is_expert {
                dma.extend(copy_gpus.iter().copied());
            }
        }
    }
    for (tier, (bytes, dma)) in weights {
        let kind = ctx.kind(tier)?;
        let prio = match kind {
            TierKind::HostPinned => PRIO_DMA_WEIGHTS,
            TierKind::HostPageable => PRIO_LOCKED,
            _ => PRIO_FIRST,
        };
        let regs = gpu_dma(kind, &dma);
        b.push(ctx.spec(
            tier,
            Purpose::Weights,
            bytes,
            SlotGeometry::Extents {
                align: cfg.extent_align,
            },
            regs,
            dma,
            prio,
            None,
        )?);
    }

    // Expert caches: one slot class per MoE layer.
    let layers: Vec<(LayerIx, Bytes)> = slot_of_layer.into_iter().collect();
    let largest = layers.iter().map(|(_, s)| *s).max().unwrap_or_default();
    let pageable_cache = caches.iter().any(|(_, k)| *k == TierKind::HostPageable);
    for (c, kind) in &caches {
        if c.bytes.0 == 0 {
            continue;
        }
        if largest.0 == 0 {
            return Err(ArenaPlanError::NoExperts(c.tier));
        }
        let counts = slot_counts(c.tier, c.bytes, c.per_layer.as_deref(), &layers)?;
        let classes = layers
            .iter()
            .zip(counts)
            .map(|(&(layer, size), count)| SlotClass {
                size,
                count,
                layer: Some(layer),
            })
            .collect();
        let host = matches!(kind, TierKind::HostPinned | TierKind::HostPageable);
        let segment = if host {
            // Host segments are `tier.pin_step` rounded down to whole slots (RFC-0020 §6.1).
            Bytes((cfg.pin_step.0 / largest.0).max(1) * largest.0)
        } else {
            let want = Bytes(cfg.vram_segment.0.max(largest.0.saturating_mul(8)));
            align_up(want, cfg.extent_align)?
        };
        // `min_ec` is the larger of one segment and 5% of the cache (RFC-0020 §8.1).
        let min_segments = c.bytes.0.div_ceil(20).div_ceil(segment.0).max(1);
        let elastic = SegmentSpec {
            segment,
            min_segments: u32::try_from(min_segments).map_err(|_| ArenaPlanError::Overflow)?,
        };
        let (regs, dma, prio) = match kind {
            TierKind::HostPinned => {
                let mut regs = gpu_dma(*kind, &copy_gpus);
                if nvme_experts {
                    regs |= Registrations::IO_FIXED;
                }
                (regs, copy_gpus.clone(), PRIO_EXPERT_SLOTS)
            }
            TierKind::HostPageable => (Registrations::empty(), BTreeSet::new(), PRIO_LOCKED),
            _ => (Registrations::empty(), BTreeSet::new(), PRIO_FIRST),
        };
        b.push(ctx.spec(
            c.tier,
            Purpose::ExpertCache,
            c.bytes,
            SlotGeometry::Classes(classes),
            regs,
            dma,
            prio,
            Some(elastic),
        )?);
    }

    // Expert staging: in VRAM beside each VRAM cache, and in pinned host memory when experts
    // reach VRAM from NVMe or from the pageable cache (RFC-0023 §3).
    if !copy_gpus.is_empty() && largest.0 > 0 {
        let one_class = |bytes: Bytes| -> Result<SlotGeometry, ArenaPlanError> {
            let count = u32::try_from(bytes.0 / largest.0).map_err(|_| ArenaPlanError::Overflow)?;
            Ok(SlotGeometry::Classes(SmallVec::from_elem(
                SlotClass {
                    size: largest,
                    count,
                    layer: None,
                },
                1,
            )))
        };
        if cfg.expert_staging.0 > 0 {
            let vram: BTreeSet<TierId> = caches
                .iter()
                .filter(|(_, k)| *k == TierKind::Vram)
                .map(|(c, _)| c.tier)
                .collect();
            for tier in vram {
                b.push(ctx.spec(
                    tier,
                    Purpose::ExpertStaging,
                    cfg.expert_staging,
                    one_class(cfg.expert_staging)?,
                    Registrations::empty(),
                    BTreeSet::new(),
                    PRIO_FIRST,
                    None,
                )?);
            }
        }
        if cfg.expert_staging_host.0 > 0 && (nvme_experts || pageable_cache) {
            for (tier, gpus) in ctx.host_pinned_for(&copy_gpus)? {
                let bytes = mul(cfg.expert_staging_host, gpus.len() as u64)?;
                let mut regs = Registrations::GPU_DMA;
                if nvme_experts {
                    regs |= Registrations::IO_FIXED;
                }
                b.push(ctx.spec(
                    tier,
                    Purpose::ExpertStaging,
                    bytes,
                    one_class(bytes)?,
                    regs,
                    gpus,
                    PRIO_EXPERT_SLOTS,
                    None,
                )?);
            }
        }
    }

    // KV: one arena per tier, summing the runner's stages' budgets there.
    let mut kv: BTreeMap<TierId, Bytes> = BTreeMap::new();
    let mut kv_gpus: BTreeMap<StageIx, BTreeSet<DeviceId>> = BTreeMap::new();
    for k in &plan.kv {
        if k.tier.node != ctx.node || !stages.contains(&k.stage) || k.bytes.0 == 0 {
            continue;
        }
        let kind = ctx.kind(k.tier)?;
        if !matches!(
            kind,
            TierKind::Vram | TierKind::Unified | TierKind::HostPinned
        ) {
            return Err(ArenaPlanError::PurposeTier {
                purpose: Purpose::Kv,
                tier: k.tier,
                kind,
            });
        }
        let sum = kv.entry(k.tier).or_default();
        *sum = add(*sum, k.bytes)?;
        if kind == TierKind::Vram {
            kv_gpus
                .entry(k.stage)
                .or_default()
                .extend(ctx.tier(k.tier)?.affinity.iter().copied());
        }
    }
    for (tier, bytes) in kv {
        let kind = ctx.kind(tier)?;
        let (regs, dma, prio) = if kind == TierKind::HostPinned {
            // Host KV pages are spill and reserve pages: copied to and from the GPUs that hold
            // the same stages' KV in VRAM, and read and written by the I/O engine.
            let dma: BTreeSet<DeviceId> = plan
                .kv
                .iter()
                .filter(|k| k.tier == tier && stages.contains(&k.stage))
                .filter_map(|k| kv_gpus.get(&k.stage))
                .flatten()
                .copied()
                .collect();
            (gpu_dma(kind, &dma) | Registrations::IO_FIXED, dma, PRIO_KV)
        } else {
            (Registrations::empty(), BTreeSet::new(), PRIO_FIRST)
        };
        let page = cfg.kv_page.0.max(1);
        let count = u32::try_from(bytes.0 / page).map_err(|_| ArenaPlanError::Overflow)?;
        b.push(ctx.spec(
            tier,
            Purpose::Kv,
            bytes,
            SlotGeometry::Pages {
                page: Bytes(page),
                count,
            },
            regs,
            dma,
            prio,
            None,
        )?);
    }

    // Scratch: on the device's own memory, or on its NUMA domain's pinned tier for a CPU.
    let mut scratch: BTreeMap<TierId, Bytes> = BTreeMap::new();
    for &(device, bytes) in &plan.scratch {
        if device.node != ctx.node || bytes.0 == 0 {
            continue;
        }
        let tier = ctx
            .device_tier(device)
            .ok_or(ArenaPlanError::NoScratchTier(device))?;
        let sum = scratch.entry(tier).or_default();
        *sum = add(*sum, bytes)?;
    }
    for (tier, bytes) in scratch {
        b.push(ctx.spec(
            tier,
            Purpose::Scratch,
            bytes,
            SlotGeometry::Bump,
            Registrations::empty(),
            BTreeSet::new(),
            PRIO_FIRST,
            None,
        )?);
    }

    // Load staging: a ring per GPU whose VRAM holds an arena of this runner (RFC-0021 §8).
    if cfg.io_stage_slots > 0 && cfg.io_stage_slot.0 > 0 {
        let used: BTreeSet<TierId> = b.arenas.iter().map(|a| a.tier).collect();
        let gpus: BTreeSet<DeviceId> = ctx
            .tiers
            .values()
            .filter(|t| t.kind == TierKind::Vram && used.contains(&t.id))
            .flat_map(|t| t.affinity.iter().copied())
            .collect();
        for (tier, gpus) in ctx.host_pinned_for(&gpus)? {
            let slots = u64::from(cfg.io_stage_slots) * gpus.len() as u64;
            let count = u32::try_from(slots).map_err(|_| ArenaPlanError::Overflow)?;
            b.push(ctx.spec(
                tier,
                Purpose::IoStaging,
                mul(cfg.io_stage_slot, slots)?,
                SlotGeometry::Ring {
                    slot: cfg.io_stage_slot,
                    count,
                },
                Registrations::GPU_DMA | Registrations::IO_FIXED,
                gpus,
                PRIO_FIRST,
                None,
            )?);
        }
    }

    b.finish(&ctx, cfg)
}

/// The inputs every step reads.
struct Ctx<'a> {
    node: syntropy_core::NodeId,
    tiers: BTreeMap<TierId, &'a TierInfo>,
    layers: Vec<Range<LayerIx>>,
    head: bool,
    tail: bool,
}

impl<'a> Ctx<'a> {
    fn new(
        plan: &PlacementPlan,
        stages: &[StageIx],
        caps: &'a CapabilityProfile,
    ) -> Result<Self, ArenaPlanError> {
        let inv = caps
            .ext
            .inventory
            .as_ref()
            .ok_or(ArenaPlanError::NoInventory)?;
        let tiers = inv
            .tiers
            .iter()
            .filter(|t| t.id.node == caps.node)
            .map(|t| (t.id, t))
            .collect();
        let mut layers = Vec::new();
        let (mut head, mut tail) = (false, false);
        for &s in stages {
            let st = plan
                .stages
                .get(usize::from(s.0))
                .filter(|st| st.stage == s)
                .ok_or(ArenaPlanError::UnknownStage(s))?;
            layers.push(st.layers.clone());
            head |= st.role.contains(StageRole::HEAD);
            tail |= st.role.contains(StageRole::TAIL);
        }
        Ok(Self {
            node: caps.node,
            tiers,
            layers,
            head,
            tail,
        })
    }

    fn tier(&self, id: TierId) -> Result<&'a TierInfo, ArenaPlanError> {
        self.tiers
            .get(&id)
            .copied()
            .ok_or(ArenaPlanError::UnknownTier(id))
    }

    fn kind(&self, id: TierId) -> Result<TierKind, ArenaPlanError> {
        self.tier(id).map(|t| t.kind)
    }

    /// Whether a group with this layer and class belongs to the runner's stages.
    fn owns(&self, layer: Option<LayerIx>, class: &TensorClass) -> bool {
        match layer {
            Some(l) => self.layers.iter().any(|r| r.contains(&l)),
            None if matches!(class, TensorClass::Head) => self.tail,
            None => self.head,
        }
    }

    /// The tier a device's scratch lives on: its `Vram` or `Unified` tier, else the
    /// `HostPinned` tier of its NUMA domain.
    fn device_tier(&self, device: DeviceId) -> Option<TierId> {
        let on = |kind: TierKind| {
            self.tiers
                .values()
                .find(|t| t.kind == kind && t.affinity.contains(&device))
                .map(|t| t.id)
        };
        on(TierKind::Vram)
            .or_else(|| on(TierKind::Unified))
            .or_else(|| on(TierKind::HostPinned))
    }

    /// Groups GPUs by the `HostPinned` tier that stages for them: the one whose affinity holds
    /// the GPU, else the node's first `HostPinned` tier.
    fn host_pinned_for(
        &self,
        gpus: &BTreeSet<DeviceId>,
    ) -> Result<BTreeMap<TierId, BTreeSet<DeviceId>>, ArenaPlanError> {
        let pinned: Vec<&TierInfo> = self
            .tiers
            .values()
            .filter(|t| t.kind == TierKind::HostPinned)
            .copied()
            .collect();
        let mut out: BTreeMap<TierId, BTreeSet<DeviceId>> = BTreeMap::new();
        for &gpu in gpus {
            let tier = pinned
                .iter()
                .find(|t| t.affinity.contains(&gpu))
                .or_else(|| pinned.first())
                .ok_or(ArenaPlanError::NoHostPinned(gpu))?;
            out.entry(tier.id).or_default().insert(gpu);
        }
        Ok(out)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per ArenaSpec field the caller decides"
    )]
    fn spec(
        &self,
        tier: TierId,
        purpose: Purpose,
        bytes: Bytes,
        geometry: SlotGeometry,
        registrations: Registrations,
        dma: BTreeSet<DeviceId>,
        pin_priority: u16,
        elastic: Option<SegmentSpec>,
    ) -> Result<ArenaSpec, ArenaPlanError> {
        let info = self.tier(tier)?;
        let host = matches!(info.kind, TierKind::HostPinned | TierKind::HostPageable);
        Ok(ArenaSpec {
            tier,
            purpose,
            bytes,
            max_bytes: bytes,
            geometry,
            pin: pin_class(info.kind),
            numa: if host { info.ext.numa } else { None },
            registrations,
            dma_devices: dma.into_iter().collect(),
            pin_priority,
            elastic,
        })
    }
}

/// The arenas so far.
#[derive(Default)]
struct Builder {
    arenas: Vec<ArenaSpec>,
}

impl Builder {
    fn push(&mut self, a: ArenaSpec) {
        self.arenas.push(a);
    }

    /// Sorts the arenas by (tier, purpose) and writes one budget line per tier.
    fn finish(mut self, ctx: &Ctx<'_>, cfg: &TierConfig) -> Result<ArenaPlan, ArenaPlanError> {
        // Stable, so arenas of one (tier, purpose) keep their build order.
        self.arenas.sort_by_key(|a| (a.tier, a.purpose));
        let mut lines: BTreeMap<TierId, BTreeMap<Purpose, Bytes>> = BTreeMap::new();
        for a in &self.arenas {
            let b = lines
                .entry(a.tier)
                .or_default()
                .entry(a.purpose)
                .or_default();
            *b = add(*b, a.max_bytes)?;
        }
        let mut per_tier = Vec::with_capacity(lines.len());
        for (tier, by) in lines {
            let mut reserved = Bytes(0);
            for b in by.values() {
                reserved = add(reserved, *b)?;
            }
            let headroom = if ctx.kind(tier)? == TierKind::Vram {
                cfg.vram_headroom
            } else {
                Bytes(0)
            };
            per_tier.push((
                tier,
                TierBudgetLine {
                    reserved,
                    by_purpose: by.into_iter().collect(),
                    headroom,
                },
            ));
        }
        Ok(ArenaPlan {
            arenas: self.arenas,
            nvme_extents: Vec::new(),
            per_tier,
        })
    }
}

/// The pin class of an arena on a tier of `kind` (module docs).
fn pin_class(kind: TierKind) -> PinClass {
    match kind {
        TierKind::HostPinned => PinClass::DmaPinned,
        _ => PinClass::Locked,
    }
}

/// `GPU_DMA` for a `HostPinned` arena that some GPU copies from.
fn gpu_dma(kind: TierKind, dma: &BTreeSet<DeviceId>) -> Registrations {
    if kind == TierKind::HostPinned && !dma.is_empty() {
        Registrations::GPU_DMA
    } else {
        Registrations::empty()
    }
}

/// Slots per MoE layer for a cache of `bytes`. With `per_layer`, layer i gets
/// `per_layer[i] / s_i` slots. Without it, every layer gets the same count, as RFC-0023 §4's
/// split does with no profile, and the bytes left over give one more slot to each layer in
/// layer order while they last.
fn slot_counts(
    tier: TierId,
    bytes: Bytes,
    per_layer: Option<&[Bytes]>,
    layers: &[(LayerIx, Bytes)],
) -> Result<Vec<u32>, ArenaPlanError> {
    let count = |n: u64| u32::try_from(n).map_err(|_| ArenaPlanError::Overflow);
    match per_layer {
        Some(per) => {
            let total = per.iter().try_fold(Bytes(0), |s, b| add(s, *b))?;
            if per.len() != layers.len() || total > bytes {
                return Err(ArenaPlanError::PerLayer {
                    tier,
                    layers: layers.len(),
                    entries: per.len(),
                });
            }
            per.iter()
                .zip(layers)
                .map(|(b, (_, s))| count(b.0.checked_div(s.0).unwrap_or(0)))
                .collect()
        }
        None => {
            let row = layers.iter().try_fold(Bytes(0), |s, (_, b)| add(s, *b))?;
            let Some(each) = bytes.0.checked_div(row.0) else {
                return Err(ArenaPlanError::NoExperts(tier));
            };
            let mut left = bytes.0 - each * row.0;
            layers
                .iter()
                .map(|(_, s)| {
                    let extra = u64::from(left >= s.0);
                    left -= extra * s.0;
                    count(each + extra)
                })
                .collect()
        }
    }
}

fn add(a: Bytes, b: Bytes) -> Result<Bytes, ArenaPlanError> {
    a.0.checked_add(b.0)
        .map(Bytes)
        .ok_or(ArenaPlanError::Overflow)
}

fn mul(a: Bytes, n: u64) -> Result<Bytes, ArenaPlanError> {
    a.0.checked_mul(n)
        .map(Bytes)
        .ok_or(ArenaPlanError::Overflow)
}

/// `b` rounded up to a multiple of `align`; an `align` of 0 or 1 leaves it.
fn align_up(b: Bytes, align: Bytes) -> Result<Bytes, ArenaPlanError> {
    if align.0 <= 1 {
        return Ok(b);
    }
    b.0.div_ceil(align.0)
        .checked_mul(align.0)
        .map(Bytes)
        .ok_or(ArenaPlanError::Overflow)
}

#[cfg(test)]
mod tests;
