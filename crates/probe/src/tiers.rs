// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host tiers of a node, one `HostPinned` and one `HostPageable` per NUMA domain, and the
//! host pool record that links each domain's pair (RFC-0010 §2.2; RFC-0020 §2, §5).
//!
//! [`host_tiers`] is a pure step over what E1 found ([`NumaHostMem`]) and the node's final
//! ordinals ([`Ordinals`], from `ordinal::assign` over E1's candidates plus the devices E2–E5
//! found). Every OS module emits the same inputs, so one step serves Linux, WSL2, Windows and
//! macOS; the rule set is chosen by [`HostOs`].
//!
//! Per domain n:
//!
//! - **Affinity.** Both tiers' `affinity` is the domain's CPU device plus every GPU whose PCI
//!   NUMA node is n. A GPU whose NUMA node is unknown, or names a domain with no host memory,
//!   attaches to domain 0 (to the lowest domain when there is no domain 0). Devices are listed
//!   in ordinal order.
//! - **Budget.** RFC-0020 §5's pure rule, `budget::host_budget`, gives the host-wide
//!   `host_usable` and `pin_limit`; `budget::numa_available` and `budget::host_pool` give the
//!   domain's `host_usable_n = min(numa_available_n + own_n, host_usable)` and `pin_limit_n =
//!   min(pin_limit, host_usable_n)`. `HostPinned_n.usable = pin_limit_n` and
//!   `HostPageable_n.usable = host_usable_n` (RFC-0020 §5, "What the solver must enforce").
//! - **Records.** Each tier's `capacity` is the domain's own RAM ([`NumaHostMem::node_total`]),
//!   its `ext.host` the domain's [`HostMemInfo`] (with the domain's `numa_free`), and its
//!   `ext.budget_source` [`BudgetSource::HostRules`]. One [`HostPoolInfo`] per domain links
//!   `[HostPinned_n, HostPageable_n]`.
//!
//! On macOS E1 gives the one domain no `HostPinned` slot (ADR-0014 capacity rules; RFC-0020
//! §5), so the step emits one `HostPageable` tier and no pool record: a pool links two tiers,
//! and the `HostPageable` tier's own `usable` already bounds what it holds.

use std::collections::BTreeMap;
use std::fmt;

use smallvec::SmallVec;
use syntropy_core::budget::{self, HostBudgetInputs, HostBudgetPolicy, HostOs};
use syntropy_core::ids::{DeviceId, TierId};
use syntropy_core::resource::ordinal::{Ordinals, TierSlot};
use syntropy_core::resource::{
    BudgetSource, DeviceKind, HostMemInfo, HostPoolInfo, TierInfo, TierInfoExt, TierKind,
};
use syntropy_core::units::Bytes;

use crate::host::NumaHostMem;

/// What [`host_tiers`] reads.
#[derive(Clone, Copy, Debug)]
pub struct HostTierInputs<'a> {
    /// Which of RFC-0020 §5's rule sets applies. WSL2 is [`HostOs::Linux`] with
    /// [`HostMemInfo::wsl2`] set, as E1 records it.
    pub os: HostOs,
    /// E1's per-domain memory inputs, one per NUMA domain with host memory.
    pub host_mem: &'a [NumaHostMem],
    /// The node's ordinals, which give the host tiers' IDs and the devices' NUMA nodes.
    pub ordinals: &'a Ordinals,
    /// `ResourcePolicy`'s budget fields, resolved.
    pub policy: &'a HostBudgetPolicy,
    /// `own_n`: what this node's runners already hold in each domain's host RAM, by domain;
    /// a domain not listed holds nothing. The host-wide `own` is their sum. Empty at first
    /// probe.
    pub own: &'a [(u16, Bytes)],
}

/// The host tiers and pool records of a node.
#[derive(Clone, Debug, PartialEq)]
pub struct HostTiers {
    /// The `HostPinned` and `HostPageable` tiers, in ordinal order.
    pub tiers: Vec<TierInfo>,
    /// One record per domain with both tiers, in domain order (`NodeInventory::host_pools`).
    pub pools: Vec<HostPoolInfo>,
}

/// Why [`host_tiers`] could not build a domain's tiers. Each is an inconsistency between the
/// ordinals and E1's records, or a reading the rule cannot use; the step never guesses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostTierError {
    /// The ordinals have a host tier for a domain E1 reported no memory for.
    NoMemory {
        /// The domain.
        numa: u16,
    },
    /// The ordinals have a `HostPageable` tier but no `HostPinned` tier for a domain where the
    /// rule set has one, or the other way round.
    PinnedMismatch {
        /// The domain.
        numa: u16,
    },
    /// A host with more than one domain reported no per-domain memory for this one, so its
    /// share of the host budget is unknown (`budget::host_pool` would give it the whole host's).
    NoNodeMemory {
        /// The domain.
        numa: u16,
    },
    /// The domain's reclaimable-memory sum overflows: a corrupt reading.
    NodeMemoryOverflow {
        /// The domain.
        numa: u16,
    },
}

impl fmt::Display for HostTierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMemory { numa } => {
                write!(f, "NUMA domain {numa} has host tiers but no memory record")
            }
            Self::PinnedMismatch { numa } => write!(
                f,
                "NUMA domain {numa}'s HostPinned tier does not match the host's rule set"
            ),
            Self::NoNodeMemory { numa } => write!(
                f,
                "NUMA domain {numa} reports no per-domain memory on a multi-domain host"
            ),
            Self::NodeMemoryOverflow { numa } => {
                write!(f, "NUMA domain {numa}'s memory figures overflow")
            }
        }
    }
}

impl std::error::Error for HostTierError {}

/// Whether the rule set has a `HostPinned` tier: not on Apple silicon or Tier 3 (RFC-0020 §5).
fn has_pinned(os: HostOs) -> bool {
    matches!(os, HostOs::Linux | HostOs::Windows)
}

/// Builds the host tiers and pool records of a node (RFC-0010 §2.2).
pub fn host_tiers(inputs: &HostTierInputs<'_>) -> Result<HostTiers, HostTierError> {
    let ords = inputs.ordinals;
    // Each domain's [HostPinned, HostPageable] tier IDs, in domain order.
    let mut slots: BTreeMap<u16, (Option<TierId>, Option<TierId>)> = BTreeMap::new();
    for (id, slot) in &ords.tiers {
        match slot {
            TierSlot::HostPinned { numa } => slots.entry(*numa).or_default().0 = Some(*id),
            TierSlot::HostPageable { numa } => slots.entry(*numa).or_default().1 = Some(*id),
            _ => {}
        }
    }

    let domains: Vec<u16> = slots.keys().copied().collect();
    let home = |numa: Option<u16>| -> Option<u16> {
        match numa {
            Some(n) if domains.contains(&n) => Some(n),
            // `domains` is sorted, so this is domain 0 when the host has one.
            _ => domains.first().copied(),
        }
    };
    let affinity = |numa: u16| -> SmallVec<[DeviceId; 4]> {
        ords.devices
            .iter()
            .filter(|(_, d)| match d.kind {
                DeviceKind::Cpu => d.numa == Some(numa),
                DeviceKind::DiscreteGpu | DeviceKind::IntegratedGpu => home(d.numa) == Some(numa),
                _ => false,
            })
            .map(|(id, _)| *id)
            .collect()
    };

    let own_n = |numa: u16| -> Bytes {
        inputs
            .own
            .iter()
            .filter(|(n, _)| *n == numa)
            .fold(Bytes(0), |s, (_, b)| Bytes(s.0.saturating_add(b.0)))
    };
    let own_total = inputs
        .own
        .iter()
        .fold(Bytes(0), |s, (_, b)| Bytes(s.0.saturating_add(b.0)));

    let mut tiers = Vec::new();
    let mut pools = Vec::new();
    for (&numa, &(pinned, pageable)) in &slots {
        let mem = inputs
            .host_mem
            .iter()
            .find(|m| m.numa == numa)
            .ok_or(HostTierError::NoMemory { numa })?;
        let Some(pageable) = pageable else {
            return Err(HostTierError::PinnedMismatch { numa });
        };
        if pinned.is_some() != has_pinned(inputs.os) {
            return Err(HostTierError::PinnedMismatch { numa });
        }
        let host = HostBudgetInputs {
            os: inputs.os,
            mem: &mem.mem,
            own: own_total,
            own_locked: Bytes(0),
            working_set: None,
            user_locked_vm: Bytes(0),
        };
        let whole = budget::host_budget(&host, inputs.policy);
        let available = match mem.node_mem {
            Some(node) => Some(
                budget::numa_available(&node, inputs.policy.numa_reserve())
                    .ok_or(HostTierError::NodeMemoryOverflow { numa })?,
            ),
            None if slots.len() > 1 => return Err(HostTierError::NoNodeMemory { numa }),
            None => None,
        };
        let affinity = affinity(numa);
        let tier = |id: TierId, kind: TierKind, usable: Bytes, pin_limit: Option<Bytes>| TierInfo {
            id,
            kind,
            capacity: mem.node_total,
            usable,
            affinity: affinity.clone(),
            pin_limit,
            volume: None,
            ext: TierInfoExt {
                numa: Some(numa),
                budget_source: BudgetSource::HostRules,
                host: Some(HostMemInfo::clone(&mem.mem)),
                context_reserve: None,
            },
        };
        match pinned {
            Some(pinned) => {
                let pool =
                    budget::host_pool(numa, [pinned, pageable], &whole, available, own_n(numa));
                tiers.push(tier(
                    pinned,
                    TierKind::HostPinned,
                    pool.pin_limit,
                    Some(pool.pin_limit),
                ));
                tiers.push(tier(pageable, TierKind::HostPageable, pool.usable, None));
                pools.push(pool);
            }
            None => {
                // No `HostPinned` tier, so no pool: the same per-domain cap, on one tier.
                let pool =
                    budget::host_pool(numa, [pageable, pageable], &whole, available, own_n(numa));
                tiers.push(tier(pageable, TierKind::HostPageable, pool.usable, None));
            }
        }
    }
    tiers.sort_by_key(|t| t.id.ordinal);
    Ok(HostTiers { tiers, pools })
}
