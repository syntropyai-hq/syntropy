// SPDX-License-Identifier: MIT OR Apache-2.0

//! Building an [`ArenaPlan`]'s arenas at `Loading` (RFC-0020 §4, §8.1; TIER-E1-T07).
//!
//! - **`Vram`** ([`build_vram`]): one fixed slab per arena through
//!   [`KernelProvider::alloc`], the fallback row of RFC-0020 §4.1. Address reservation and
//!   mapped segments (`reserve_only`, `map_segment`) are TIER-E3-T02's, so an elastic
//!   `ExpertCache` is a fixed slab here too and a shrink needs a runner reload (`TIER_VMM_UNSUPPORTED`
//!   in RFC-0020 §15).
//! - **The §4.4 order**: the context reserve is measured first ([`open_measured`]), then the
//!   arenas are allocated `Scratch`, `Graphs`, `Weights`, `Kv`, `ExpertStaging`, any other
//!   purpose in plan order, and `ExpertCache` last ([`vram_rank`]).
//! - **The §8.1 check** ([`fit_vram`]) runs before the first allocation, so a tier that is
//!   short allocates nothing.
//! - **`HostPageable`** ([`build_host_pageable`]): anonymous memory from [`OsMemory::reserve`],
//!   committed to the arena's initial size (RFC-0020 §4.2). Locking is the pinning loop's
//!   (RFC-0020 §6).
//!
//! The builder is the only caller of `alloc` for model memory (RFC-0020 §4: "providers never
//! allocate model memory on their own initiative"). Nothing here holds `unsafe`: the provider
//! and the OS adapter own every raw address.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use syntropy_core::error::codes;
use syntropy_core::{Bytes, DeviceId, Error, ErrorClass, RetryHint, TierId};
use syntropy_kernels::{
    AllocRequest, DeviceBuffer, DeviceCtx, KernelProvider, MemKind, OpenOptions,
};

use super::{ArenaPlan, ArenaSpec, Purpose, SlotGeometry};
use crate::os::{HostRange, HugePages, OsMemError, OsMemory};

/// The alignment of a `Vram` slab whose geometry names none. Providers round it up to their own
/// minimum (RFC-0040 §5.1).
pub const SLAB_ALIGN: Bytes = Bytes(256);

/// The context reserve of one device (RFC-0020 §8.1; RFC-0001 §9.3 item 1): device free memory
/// before and after the context is created and the provider initialised. RFC-0020 §8.1 makes
/// this measured value the one `Loading` uses; the probe's value is only a prior.
///
/// On CUDA the driver reads free memory only inside a context, so `free_before` is read in a
/// transient primary context that `enumerate` retains and releases. The difference is then what
/// opening the device adds above that base context: streams, events and provider state. The
/// §8.1 check stays exact, because `free_before − bytes()` is the free memory the arenas are
/// allocated from either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextReserve {
    /// Device free memory before `open` (`free_0` in RFC-0020 §8.1).
    pub free_before: Bytes,
    /// Device free memory after `open`.
    pub free_after: Bytes,
}

impl ContextReserve {
    /// The reserve, `free_before − free_after`, or 0 when another program freed memory
    /// in between.
    #[must_use]
    pub fn bytes(self) -> Bytes {
        Bytes(self.free_before.0.saturating_sub(self.free_after.0))
    }
}

/// The free memory `p` reports now for the device at provider-local index `ordinal`.
#[expect(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct with its fields inline; a build fails once, in Loading"
)]
fn free_now(p: &dyn KernelProvider, ordinal: u16) -> Result<Bytes, Error> {
    let devices = p.enumerate()?;
    devices
        .get(usize::from(ordinal))
        .map(|d| d.memory.free)
        .ok_or_else(|| {
            Error::new(
                ErrorClass::Unsupported,
                codes::kernels::OP_UNSUPPORTED,
                format!("provider {} has no device {ordinal}", p.id().0),
            )
        })
}

/// Opens `dev` and measures its context reserve (RFC-0020 §4.4 step 1, §8.1).
///
/// Free memory is read from `enumerate` before `open`. After it, it is read from
/// `memory_info`, or from `enumerate` again where the provider does not implement
/// `memory_info`.
///
/// # Errors
///
/// The provider's error from `enumerate`, `open` or `memory_info`, or `Unsupported` when the
/// provider has no device at `dev.ordinal`.
#[expect(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct with its fields inline; a build fails once, in Loading"
)]
pub fn open_measured(
    p: &dyn KernelProvider,
    dev: DeviceId,
    opts: &OpenOptions,
) -> Result<(DeviceCtx, ContextReserve), Error> {
    let free_before = free_now(p, dev.ordinal)?;
    let ctx = p.open(dev, opts)?;
    let free_after = match p.memory_info(&ctx) {
        Ok(info) => info.free,
        Err(e) if e.class == ErrorClass::Unsupported => free_now(p, dev.ordinal)?,
        Err(e) => return Err(e),
    };
    Ok((
        ctx,
        ContextReserve {
            free_before,
            free_after,
        },
    ))
}

/// The position of `purpose` in RFC-0020 §4.4's `Vram` order: `Scratch`, `Graphs`, `Weights`,
/// `Kv`, `ExpertStaging`, then every other purpose, and the elastic `ExpertCache` last, so a
/// shortfall shrinks it first (§8.2).
#[must_use]
pub fn vram_rank(purpose: Purpose) -> u8 {
    match purpose {
        Purpose::Scratch => 0,
        Purpose::Graphs => 1,
        Purpose::Weights => 2,
        Purpose::Kv => 3,
        Purpose::ExpertStaging => 4,
        Purpose::ExpertCache => 6,
        _ => 5,
    }
}

/// The figures of RFC-0020 §8.1's check for one `Vram` tier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VramNeeds {
    /// The context reserve (`ctx`).
    pub context: Bytes,
    /// Every arena on the tier but `ExpertCache` (`scratch + graphs + dense_hot + kv_vram +
    /// staging`).
    pub fixed: Bytes,
    /// The planned expert cache (`need_ec`).
    pub expert: Bytes,
    /// The least expert cache the tier may map (`min_ec`).
    pub min_expert: Bytes,
    /// The expert cache's segment; a reduced cache is a whole number of them.
    pub segment: Bytes,
    /// What the tier keeps free (`headroom`).
    pub headroom: Bytes,
}

impl VramNeeds {
    /// `need_fix`: the context reserve plus every fixed arena.
    #[must_use]
    pub fn need_fix(&self) -> Bytes {
        Bytes(self.context.0.saturating_add(self.fixed.0))
    }
}

/// The outcome of RFC-0020 §8.1's check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VramFit {
    /// Everything fits as planned.
    All,
    /// The fixed arenas fit, and only `expert` bytes of the expert cache, a whole number of
    /// segments at least `min_ec`: a `Capacity` drift, and the runner still reaches `Ready`.
    Reduced {
        /// The expert-cache bytes that fit.
        expert: Bytes,
    },
    /// Even the fixed arenas and `min_ec` do not fit: `Loading` fails with `TIER_VRAM_SHORT`.
    Short,
}

/// RFC-0020 §8.1's check of `n` against `free_before` (`free_0`). Sums saturate, so a figure
/// near `u64::MAX` reads as not fitting rather than wrapping.
#[must_use]
pub fn fit_vram(n: &VramNeeds, free_before: Bytes) -> VramFit {
    let base = n.need_fix().0.saturating_add(n.headroom.0);
    if base.saturating_add(n.expert.0) <= free_before.0 {
        return VramFit::All;
    }
    if base.saturating_add(n.min_expert.0) > free_before.0 {
        return VramFit::Short;
    }
    let room = free_before.0 - base;
    let segment = n.segment.0.max(1);
    let expert = (room / segment * segment)
        .max(n.min_expert.0)
        .min(n.expert.0);
    VramFit::Reduced {
        expert: Bytes(expert),
    }
}

/// `min_ec` of one expert-cache arena (RFC-0020 §8.1): `min_segments × segment` from its
/// elastic spec. A spec with no minimum gets the default, the larger of one segment and 5% of
/// the planned cache rounded up to whole segments. A fixed arena cannot shrink, so its minimum
/// is its size.
fn min_expert(a: &ArenaSpec) -> (Bytes, Bytes) {
    match a.elastic {
        Some(e) if e.segment.0 > 0 => {
            let seg = e.segment.0;
            let min = if e.min_segments > 0 {
                u64::from(e.min_segments).saturating_mul(seg)
            } else {
                let five = a.bytes.0.div_ceil(20).div_ceil(seg).saturating_mul(seg);
                five.max(seg)
            };
            (Bytes(min.min(a.bytes.0)), Bytes(seg))
        }
        _ => (a.bytes, a.bytes.max(Bytes(1))),
    }
}

/// The figures of [`fit_vram`] for the arenas of `plan` on `tier`.
#[must_use]
pub fn vram_needs(plan: &ArenaPlan, tier: TierId, context: Bytes) -> VramNeeds {
    let mut n = VramNeeds {
        context,
        headroom: plan
            .per_tier
            .iter()
            .find(|(t, _)| *t == tier)
            .map_or(Bytes(0), |(_, line)| line.headroom),
        ..VramNeeds::default()
    };
    for a in plan.arenas.iter().filter(|a| a.tier == tier) {
        if a.purpose == Purpose::ExpertCache {
            let (min, seg) = min_expert(a);
            n.expert.0 = n.expert.0.saturating_add(a.bytes.0);
            n.min_expert.0 = n.min_expert.0.saturating_add(min.0);
            n.segment = if n.segment.0 == 0 {
                seg
            } else {
                n.segment.min(seg)
            };
        } else {
            n.fixed.0 = n.fixed.0.saturating_add(a.bytes.0);
        }
    }
    n
}

/// The alignment of `a`'s slab: an `Extents` geometry's own, else [`SLAB_ALIGN`].
fn slab_align(a: &ArenaSpec) -> Bytes {
    match &a.geometry {
        SlotGeometry::Extents { align } if align.0 > 0 => *align,
        _ => SLAB_ALIGN,
    }
}

/// One `Vram` arena as built: a fixed slab from [`KernelProvider::alloc`].
#[derive(Debug, PartialEq, Eq)]
pub struct VramArena {
    /// The arena's index in [`ArenaPlan::arenas`].
    pub index: usize,
    /// Why the arena exists.
    pub purpose: Purpose,
    /// The planned size, [`ArenaSpec::bytes`].
    pub planned: Bytes,
    /// The slab. Its `bytes` is `planned` except for an `ExpertCache` reduced by §8.1.
    pub buffer: DeviceBuffer,
}

/// What building one `Vram` tier measured and decided (RFC-0020 §8.1, §13). The residency
/// digest (TIER-E1-T09) carries it to the daemon; until that type exists, this is where the
/// measured context reserve is recorded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VramReport {
    /// The tier.
    pub tier: TierId,
    /// The measured context reserve.
    pub context_reserve: ContextReserve,
    /// The §8.1 figures, with the measured reserve in `context`.
    pub needs: VramNeeds,
    /// The §8.1 outcome.
    pub fit: VramFit,
    /// Expert-cache bytes allocated: the planned cache, or less after a §8.1 reduction.
    pub expert_mapped: Bytes,
    /// Whether the arenas sit on mapped segments. Always false here: the fixed-slab path of
    /// RFC-0020 §4.1 (TIER-E3-T02 adds segments).
    pub segments: bool,
}

impl VramReport {
    /// The `TIER_VRAM_EC_REDUCED` drift (`Drift`, RFC-0020 §15) when §8.1 mapped the expert
    /// cache below plan, with `vram_usable` (the context reserve plus every byte allocated) in
    /// its context; `None` otherwise.
    #[must_use]
    pub fn drift(&self) -> Option<Error> {
        if !matches!(self.fit, VramFit::Reduced { .. }) {
            return None;
        }
        let expert = self.expert_mapped;
        let usable = self.needs.need_fix().0.saturating_add(expert.0);
        Some(
            Error::new(
                ErrorClass::Drift,
                codes::tier::VRAM_EC_REDUCED,
                format!(
                    "the expert cache on tier {} is mapped at {} of {} planned bytes",
                    self.tier.ordinal, expert.0, self.needs.expert.0
                ),
            )
            .with_context("tier", self.tier.ordinal.to_string())
            .with_context("vram_usable", usable.to_string())
            .with_context("expert_planned", self.needs.expert.0.to_string())
            .with_context("expert_mapped", expert.0.to_string()),
        )
    }
}

/// The `Vram` arenas of one device, allocated. The tier manager owns the slabs: [`release`]
/// returns them through `free`, which needs the provider and the context, so dropping this
/// value without `release` leaks them until the context is destroyed.
///
/// [`release`]: VramArenas::release
#[derive(Debug)]
#[must_use = "the slabs are returned only by `release`"]
pub struct VramArenas {
    /// The arenas in allocation order (RFC-0020 §4.4).
    pub arenas: Vec<VramArena>,
    /// What the build measured and decided.
    pub report: VramReport,
}

impl VramArenas {
    /// The slab of the arena at `index` in [`ArenaPlan::arenas`].
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&DeviceBuffer> {
        self.arenas
            .iter()
            .find(|a| a.index == index)
            .map(|a| &a.buffer)
    }

    /// Frees every slab, in the reverse of allocation order.
    pub fn release(self, p: &dyn KernelProvider, ctx: &DeviceCtx) {
        free_reverse(p, ctx, self.arenas);
    }
}

fn free_reverse(p: &dyn KernelProvider, ctx: &DeviceCtx, arenas: Vec<VramArena>) {
    for a in arenas.into_iter().rev() {
        p.free(ctx, a.buffer);
    }
}

fn vram_short(tier: TierId, n: &VramNeeds, free_before: Bytes) -> Error {
    Error::new(
        ErrorClass::Capacity,
        codes::tier::VRAM_SHORT,
        format!(
            "tier {} needs {} fixed bytes, {} of expert cache at least and {} of headroom, but {} are free",
            tier.ordinal,
            n.need_fix().0,
            n.min_expert.0,
            n.headroom.0,
            free_before.0
        ),
    )
    .with_retry(RetryHint::OnNewPlan)
    .with_context("tier", tier.ordinal.to_string())
    .with_context("free", free_before.0.to_string())
    .with_context("need_fix", n.need_fix().0.to_string())
    .with_context("min_ec", n.min_expert.0.to_string())
    .with_context("headroom", n.headroom.0.to_string())
}

/// Splits `total` expert-cache bytes over the expert-cache arenas of `order`, trimming from the
/// last arena backwards, each to no less than its own minimum and by whole segments.
fn expert_sizes(plan: &ArenaPlan, order: &[usize], total: Bytes) -> Vec<(usize, Bytes)> {
    let ec: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&i| plan.arenas[i].purpose == Purpose::ExpertCache)
        .collect();
    let planned: u64 = ec.iter().map(|&i| plan.arenas[i].bytes.0).sum();
    let mut cut = planned.saturating_sub(total.0);
    let mut out: Vec<(usize, Bytes)> = ec.iter().map(|&i| (i, plan.arenas[i].bytes)).collect();
    for (i, size) in out.iter_mut().rev() {
        if cut == 0 {
            break;
        }
        let (min, seg) = min_expert(&plan.arenas[*i]);
        let spare = size.0 - min.0;
        let take = cut
            .min(spare)
            .div_ceil(seg.0)
            .saturating_mul(seg.0)
            .min(spare);
        size.0 -= take;
        cut = cut.saturating_sub(take);
    }
    out
}

/// Builds every arena of `plan` on the `Vram` tier `tier` of the device behind `ctx`
/// (RFC-0020 §4.1, §4.4 step 2, §8.1).
///
/// The §8.1 check runs first, with `reserve` measured by [`open_measured`]. Each arena is then
/// one `MemKind::Device` slab, in [`vram_rank`] order. A zero-byte arena gets no slab. If an
/// allocation fails, every slab already allocated is freed, in reverse, before the error is
/// returned.
///
/// # Errors
///
/// `TIER_VRAM_SHORT` (`Capacity`) when the check fails, with nothing allocated; or the
/// provider's `alloc` error, with the arena's purpose and size in its context.
#[expect(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct with its fields inline; a build fails once, in Loading"
)]
pub fn build_vram(
    p: &dyn KernelProvider,
    ctx: &DeviceCtx,
    plan: &ArenaPlan,
    tier: TierId,
    reserve: ContextReserve,
) -> Result<VramArenas, Error> {
    let needs = vram_needs(plan, tier, reserve.bytes());
    let fit = fit_vram(&needs, reserve.free_before);
    let expert_total = match fit {
        VramFit::All => needs.expert,
        VramFit::Reduced { expert } => expert,
        VramFit::Short => return Err(vram_short(tier, &needs, reserve.free_before)),
    };
    let mut order: Vec<usize> = (0..plan.arenas.len())
        .filter(|&i| plan.arenas[i].tier == tier)
        .collect();
    // A stable sort keeps plan order within a rank.
    order.sort_by_key(|&i| vram_rank(plan.arenas[i].purpose));
    let expert = expert_sizes(plan, &order, expert_total);
    let expert_mapped = Bytes(expert.iter().map(|(_, b)| b.0).sum());

    let mut built: Vec<VramArena> = Vec::with_capacity(order.len());
    for &i in &order {
        let a = &plan.arenas[i];
        let bytes = expert
            .iter()
            .find(|(j, _)| *j == i)
            .map_or(a.bytes, |(_, b)| *b);
        if bytes.0 == 0 {
            continue;
        }
        let req = AllocRequest {
            bytes,
            kind: MemKind::Device,
            align: slab_align(a),
            reserve_only: false,
        };
        match p.alloc(ctx, req) {
            Ok(buffer) => built.push(VramArena {
                index: i,
                purpose: a.purpose,
                planned: a.bytes,
                buffer,
            }),
            Err(e) => {
                free_reverse(p, ctx, built);
                return Err(e
                    .with_context("arena", i.to_string())
                    .with_context("purpose", format!("{:?}", a.purpose))
                    .with_context("bytes", bytes.0.to_string()));
            }
        }
    }
    Ok(VramArenas {
        arenas: built,
        report: VramReport {
            tier,
            context_reserve: reserve,
            needs,
            fit,
            expert_mapped,
            segments: false,
        },
    })
}

/// One `HostPageable` arena as built: an anonymous range, committed to its initial size.
#[derive(Debug)]
pub struct HostArena {
    /// The arena's index in [`ArenaPlan::arenas`].
    pub index: usize,
    /// Why the arena exists.
    pub purpose: Purpose,
    /// The range: [`ArenaSpec::max_bytes`] reserved.
    pub range: HostRange,
    /// The bytes committed from its start, [`ArenaSpec::bytes`].
    pub committed: Bytes,
    /// The arena asked for a NUMA node and the bind failed twice, so the range is unbound
    /// (`TIER_NUMA_BIND`, RFC-0020 §15; the profile marks the domain unbound).
    pub unbound: bool,
}

/// The `HostPageable` arenas of one tier, reserved and committed. [`release`] returns them to
/// the OS adapter that reserved them.
///
/// [`release`]: HostArenas::release
#[derive(Debug)]
#[must_use = "the ranges are returned only by `release`"]
pub struct HostArenas {
    /// The arenas in plan order.
    pub arenas: Vec<HostArena>,
}

impl HostArenas {
    /// Returns every range to `os`, in the reverse of reservation order.
    pub fn release(self, os: &dyn OsMemory) {
        release_reverse(os, self.arenas);
    }
}

fn release_reverse(os: &dyn OsMemory, arenas: Vec<HostArena>) {
    for a in arenas.into_iter().rev() {
        os.release(a.range);
    }
}

/// Builds every arena of `plan` on the `HostPageable` tier `tier` (RFC-0020 §4.2, §4.4 step 3):
/// [`OsMemory::reserve`] of `max_bytes`, bound to the arena's NUMA node, then
/// [`OsMemory::commit`] of the first `bytes`. A zero-byte arena reserves nothing. If a call
/// fails, every range already reserved is released, in reverse, before the error is returned.
///
/// The ranges stay pageable; the pinning loop locks a `Locked` arena step by step (RFC-0020
/// §6). Anonymous memory on Linux and Windows comes from the adapters in [`crate::os`]; the
/// macOS adapter is TIER-E2-T04's.
///
/// # Errors
///
/// The adapter's [`OsMemError`]: `TIER_HOST_RESERVE` when the OS refuses the reservation or the
/// commit.
pub fn build_host_pageable(
    os: &dyn OsMemory,
    plan: &ArenaPlan,
    tier: TierId,
    huge: HugePages,
) -> Result<HostArenas, OsMemError> {
    let mut built: Vec<HostArena> = Vec::new();
    for (i, a) in plan.arenas.iter().enumerate() {
        if a.tier != tier || a.max_bytes.max(a.bytes).0 == 0 {
            continue;
        }
        let range = match os.reserve(a.max_bytes.max(a.bytes), a.numa, huge) {
            Ok(r) => r,
            Err(e) => {
                release_reverse(os, built);
                return Err(e);
            }
        };
        if let Err(e) = os.commit(&range, 0..a.bytes.0) {
            os.release(range);
            release_reverse(os, built);
            return Err(e);
        }
        built.push(HostArena {
            index: i,
            purpose: a.purpose,
            unbound: a.numa.is_some() && range.numa.is_none(),
            range,
            committed: a.bytes,
        });
    }
    Ok(HostArenas { arenas: built })
}

#[cfg(test)]
mod tests;
