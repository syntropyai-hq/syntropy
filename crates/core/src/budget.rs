// SPDX-License-Identifier: MIT OR Apache-2.0

//! RFC-0020 §5's host budget rule: `host_usable`, `pin_limit` and `io_fixed` per OS, the
//! per-NUMA pool record the solver's joint rows read, and the per-user `io_fixed` ledger.
//!
//! Every function is pure. The OS facts arrive as a [`HostMemInfo`] that RFC-0010's probe or the
//! tier manager's snapshot filled, so the same rule runs at probe time (RFC-0010 §2.2) and
//! before every pinning step (RFC-0020 §6). The joint rows themselves
//! (`HostPinned_n ≤ pin_limit_n`, `HostPinned_n + HostPageable_n ≤ host_usable_n`) are checked
//! by [`PlacementPlan::validate`](crate::PlacementPlan::validate) over the [`HostPoolInfo`]
//! records [`host_pool`] builds (ADR-0200 item 3).
//!
//! Arithmetic saturates: a term that would go below zero is zero, and a sum that would pass
//! `u64::MAX` is `u64::MAX`.

use crate::ids::{InstanceId, TierId};
use crate::resource::{HostMemInfo, HostPoolInfo};
use crate::units::Bytes;

const GIB: u64 = 1 << 30;

/// The floor of RFC-0020 §5's default headroom: 4 GiB \[target\].
pub const DEFAULT_HEADROOM_MIN: Bytes = Bytes(4 * GIB);

/// The share of RAM in RFC-0020 §5's default headroom: 0.10 \[target\].
pub const DEFAULT_HEADROOM_FRACTION: f32 = 0.10;

/// RFC-0020 §5's default `numa_reserve`: 1 GiB \[target\].
pub const DEFAULT_NUMA_RESERVE: Bytes = Bytes(GIB);

/// Which of RFC-0020 §5's rules applies.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostOs {
    /// Linux on bare metal or in a container; the cgroup limit, v1 or v2, arrives in
    /// [`HostMemInfo::cgroup_limit`]. Inside WSL2 as well: [`HostMemInfo::wsl2`] is the one
    /// record of that, and selects WSL2's `max_pinned_fraction` default, the Linux rule
    /// otherwise applying to the VM's `MemTotal`.
    Linux,
    /// Windows. [`HostMemInfo::cgroup_limit`] carries the job object's limit − usage.
    Windows,
    /// macOS on Apple silicon: one `Unified` tier, no `HostPinned` tier.
    MacOs,
    /// Android and iOS (Tier 3).
    Mobile,
}

impl HostOs {
    /// RFC-0020 §5's per-OS `max_pinned_fraction` default: 0.85 on Linux, 0.75 on WSL2 (Linux
    /// with `wsl2` set), 0.50 on Windows \[target\]. `None` on macOS, where it does not apply
    /// unless the user sets it, and on Tier 3, where `pin_limit` is 0.
    pub fn default_max_pinned_fraction(self, wsl2: bool) -> Option<f32> {
        match self {
            HostOs::Linux if wsl2 => Some(0.75),
            HostOs::Linux => Some(0.85),
            HostOs::Windows => Some(0.50),
            HostOs::MacOs | HostOs::Mobile => None,
        }
    }
}

/// The policy inputs of the rule: `ResourcePolicy`'s fields that RFC-0020 §5 reads (RFC-0054
/// §4.3), with the config layer's `Reserve` already resolved against RAM.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HostBudgetPolicy {
    /// `resource.ram_headroom` resolved to bytes. `None` takes RFC-0020 §5's default,
    /// [`default_headroom`].
    pub ram_headroom: Option<Bytes>,
    /// `ResourcePolicy::max_pinned_fraction`. `None` takes
    /// [`HostOs::default_max_pinned_fraction`]. Read to a resolution of one part per million;
    /// values outside [0, 1] are clamped and NaN counts as 0.
    pub max_pinned_fraction: Option<f32>,
    /// `numa_reserve`, subtracted once from each node's estimate. `None` takes
    /// [`DEFAULT_NUMA_RESERVE`].
    pub numa_reserve: Option<Bytes>,
}

impl HostBudgetPolicy {
    /// The headroom on a host with `total` bytes of RAM.
    pub fn headroom(&self, total: Bytes) -> Bytes {
        self.ram_headroom.unwrap_or_else(|| default_headroom(total))
    }

    /// The `numa_reserve` in effect.
    pub fn numa_reserve(&self) -> Bytes {
        self.numa_reserve.unwrap_or(DEFAULT_NUMA_RESERVE)
    }

    fn max_pinned_fraction(&self, os: HostOs, wsl2: bool) -> Option<f32> {
        self.max_pinned_fraction
            .or_else(|| os.default_max_pinned_fraction(wsl2))
    }
}

/// What the rule reads besides the policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostBudgetInputs<'a> {
    /// Which rule applies. WSL2 is not an `os` of its own: it is [`HostOs::Linux`] with
    /// [`HostMemInfo::wsl2`] set.
    pub os: HostOs,
    /// The OS facts.
    pub mem: &'a HostMemInfo,
    /// `own`: the bytes this node's runners already hold in host RAM, so a re-plan of a running
    /// instance does not count its own memory against itself (RFC-0020 §5). 0 at first probe.
    pub own: Bytes,
    /// Linux: the bytes this node's runners already hold locked that
    /// [`LockAccounting::vmlck_before`](crate::resource::LockAccounting::vmlck_before) counts.
    /// They are added back to `lockable`, the `own` convention, so a re-evaluation before a
    /// pinning step does not count a running plan's locked memory against itself. The add-back
    /// is clamped to `VmLck`, so `lockable` never exceeds `RLIMIT_MEMLOCK`. The caller passes a
    /// live `VmLck` together with its live `own_locked`, never a cached probe-time `VmLck`. 0 at
    /// first probe. Unused elsewhere.
    pub own_locked: Bytes,
    /// macOS: `recommendedMaxWorkingSetSize`, read in the probe runner (RFC-0010 §2.2). `None`
    /// leaves the term out, and the two RAM terms still hold the reserve (RFC-0020 §5, note of
    /// 2026-10-22).
    pub working_set: Option<Bytes>,
    /// Linux: the user's `locked_vm`, less what this node's runners hold under the
    /// [`IoFixedLedger`] (the `own` convention). Unused elsewhere.
    pub user_locked_vm: Bytes,
}

/// The rule's result for one host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostBudget {
    /// The headroom in effect.
    pub headroom: Bytes,
    /// `host_usable`: `HostPageable.usable`, shared with `HostPinned`. On macOS, the `Unified`
    /// tier's `usable`.
    pub host_usable: Bytes,
    /// `pin_limit`: `HostPinned.pin_limit`, at most `host_usable`. 0 where there is no
    /// `HostPinned` tier (macOS, Tier 3).
    pub pin_limit: Bytes,
    /// `io_fixed`: what io_uring fixed-buffer registration may take for the whole user. `None`
    /// is unlimited (`CAP_IPC_LOCK`, or no `RLIMIT_MEMLOCK`).
    pub io_fixed: Option<Bytes>,
}

/// RFC-0020 §5's default headroom: `max(4 GiB, 0.10 · RAM)`.
pub fn default_headroom(total: Bytes) -> Bytes {
    DEFAULT_HEADROOM_MIN.max(fraction_of(total, DEFAULT_HEADROOM_FRACTION))
}

/// `host_usable` for this host.
///
/// - Linux, WSL2: `min(MemAvailable + own − headroom, cgroup headroom + own)`.
/// - Windows: `min(ullAvailPhys + own − headroom, job-object headroom + own)`.
/// - macOS: `min(recommendedMaxWorkingSetSize, hw.memsize − headroom, available + own −
///   headroom)`, with no fixed percentage (ADR-0007); also `≤ max_pinned_fraction · RAM` only
///   when the user set the fraction.
/// - Tier 3: `available − headroom`.
pub fn host_usable(inputs: &HostBudgetInputs<'_>, policy: &HostBudgetPolicy) -> Bytes {
    let mem = inputs.mem;
    let headroom = policy.headroom(mem.total);
    let avail = sub(add(mem.available, inputs.own), headroom);
    match inputs.os {
        HostOs::Linux | HostOs::Windows => match mem.cgroup_limit {
            Some(limit) => avail.min(add(limit, inputs.own)),
            None => avail,
        },
        HostOs::MacOs => {
            let mut usable = avail.min(sub(mem.total, headroom));
            if let Some(ws) = inputs.working_set {
                usable = usable.min(ws);
            }
            if let Some(f) = policy.max_pinned_fraction {
                usable = usable.min(fraction_of(mem.total, f));
            }
            usable
        }
        HostOs::Mobile => sub(mem.available, headroom),
    }
}

/// `pin_limit` for this host.
///
/// - Linux, WSL2: `min(host_usable, max_pinned_fraction · MemTotal, lockable)`, where
///   `lockable` is unlimited with `CAP_IPC_LOCK` or no `RLIMIT_MEMLOCK`, or when the probe
///   showed provider registration is not charged to it, and `RLIMIT_MEMLOCK − VmLck +
///   min(own_locked, VmLck)` otherwise. An untested charge (`None`) counts as charged.
/// - Windows: `min(host_usable, max_pinned_fraction · ullTotalPhys, lockable_measured)`. Until
///   H5 measures `lockable_measured`, the static estimate is
///   [`HostMemInfo::windows_lockable`], the working-set ceiling obtainable without privilege
///   (RFC-0010 §3.2). With neither, `pin_limit` is 0: the rule never assumes a lock budget it
///   was not given.
/// - macOS, Tier 3: 0.
pub fn pin_limit(inputs: &HostBudgetInputs<'_>, policy: &HostBudgetPolicy) -> Bytes {
    let mem = inputs.mem;
    let lock = mem.lock.as_ref();
    let lockable = match inputs.os {
        HostOs::Linux => {
            let uncharged = lock.and_then(|l| l.registration_charged) == Some(false);
            match mem.memlock_limit {
                Some(limit) if !uncharged => {
                    let vmlck = lock.map(|l| l.vmlck_before).unwrap_or_default();
                    // Never add back more than VmLck counts, so lockable ≤ RLIMIT_MEMLOCK.
                    Some(sub(add(limit, inputs.own_locked.min(vmlck)), vmlck))
                }
                _ => None,
            }
        }
        HostOs::Windows => Some(
            lock.and_then(|l| l.lockable_measured)
                .or(mem.windows_lockable)
                .unwrap_or(Bytes(0)),
        ),
        HostOs::MacOs | HostOs::Mobile => return Bytes(0),
    };
    let mut limit = host_usable(inputs, policy);
    if let Some(f) = policy.max_pinned_fraction(inputs.os, mem.wsl2) {
        limit = limit.min(fraction_of(mem.total, f));
    }
    if let Some(l) = lockable {
        limit = limit.min(l);
    }
    limit
}

/// `io_fixed`: `RLIMIT_MEMLOCK − locked_vm` of the user on Linux and WSL2, unlimited (`None`)
/// with `CAP_IPC_LOCK` or no limit.
///
/// RFC-0020 §5 defines `io_fixed` for Linux only. Elsewhere this returns `Some(0)`: no
/// fixed-buffer budget is granted, and RFC-0021 uses unregistered reads, which loses
/// efficiency, not correctness (RFC-0020 §5, note of 2026-10-22).
pub fn io_fixed(inputs: &HostBudgetInputs<'_>) -> Option<Bytes> {
    match inputs.os {
        HostOs::Linux => inputs
            .mem
            .memlock_limit
            .map(|limit| sub(limit, inputs.user_locked_vm)),
        HostOs::Windows | HostOs::MacOs | HostOs::Mobile => Some(Bytes(0)),
    }
}

/// The whole rule for one host.
pub fn host_budget(inputs: &HostBudgetInputs<'_>, policy: &HostBudgetPolicy) -> HostBudget {
    HostBudget {
        headroom: policy.headroom(inputs.mem.total),
        host_usable: host_usable(inputs, policy),
        pin_limit: pin_limit(inputs, policy),
        io_fixed: io_fixed(inputs),
    }
}

/// One NUMA node's reclaimable memory, from `/sys/devices/system/node/node<n>/meminfo`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeMem {
    /// `MemFree`.
    pub mem_free: Bytes,
    /// `Active(file)`.
    pub active_file: Bytes,
    /// `Inactive(file)`.
    pub inactive_file: Bytes,
    /// `SReclaimable`.
    pub sreclaimable: Bytes,
}

/// RFC-0020 §5's per-node available estimate: `MemFree + Active(file) + Inactive(file) +
/// SReclaimable − numa_reserve`, floored at zero, with the reserve subtracted here once. `None`
/// when the sum overflows, so a corrupt reading never yields a plausible estimate.
pub fn numa_available(node: &NodeMem, numa_reserve: Bytes) -> Option<Bytes> {
    let sum = node
        .mem_free
        .0
        .checked_add(node.active_file.0)?
        .checked_add(node.inactive_file.0)?
        .checked_add(node.sreclaimable.0)?;
    Some(sub(Bytes(sum), numa_reserve))
}

/// The pool record of one NUMA domain (RFC-0010 §2.2), which the joint rows read:
/// `host_usable_n = min(numa_available_n + own_n, host_usable)` and `pin_limit_n =
/// min(pin_limit, host_usable_n)`.
///
/// `numa_available` is [`numa_available`]'s estimate for the domain. `None`, for a host that
/// reports no per-node memory, gives the domain the whole host's budget, which is right only
/// for a host with one domain. `own_n` is what this node's runners hold in the domain.
pub fn host_pool(
    numa: u16,
    tiers: [TierId; 2],
    budget: &HostBudget,
    numa_available: Option<Bytes>,
    own_n: Bytes,
) -> HostPoolInfo {
    let usable = match numa_available {
        Some(avail) => add(avail, own_n).min(budget.host_usable),
        None => budget.host_usable,
    };
    HostPoolInfo {
        numa,
        tiers,
        usable,
        pin_limit: budget.pin_limit.min(usable),
    }
}

/// The daemon's division of the user's `io_fixed` among the node's runners (RFC-0020 §5).
///
/// io_uring charges registered buffers to the user's `locked_vm`, not the process's, so the
/// runners share one budget. The sum of the grants never exceeds the limit.
///
/// The limit is fixed at [`IoFixedLedger::new`]. To re-evaluate `io_fixed`, build a new ledger
/// over the new limit and replay the runners' current grants into it with
/// [`IoFixedLedger::grant`]; dropping the old ledger without replaying forgets bytes the runners
/// still hold registered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoFixedLedger {
    limit: Option<Bytes>,
    grants: Vec<(InstanceId, Bytes)>,
}

impl IoFixedLedger {
    /// A ledger over [`io_fixed`]'s result. `None` is unlimited.
    pub fn new(limit: Option<Bytes>) -> Self {
        Self {
            limit,
            grants: Vec::new(),
        }
    }

    /// Sets `runner`'s grant to at most `want`, within what the other runners leave, and
    /// returns it. A grant replaces the runner's previous one.
    pub fn grant(&mut self, runner: InstanceId, want: Bytes) -> Bytes {
        let others = self
            .grants
            .iter()
            .filter(|(r, _)| *r != runner)
            .fold(Bytes(0), |sum, &(_, b)| add(sum, b));
        let granted = match self.limit {
            Some(limit) => want.min(sub(limit, others)),
            None => want,
        };
        match self.grants.iter_mut().find(|(r, _)| *r == runner) {
            Some(entry) => entry.1 = granted,
            None => self.grants.push((runner, granted)),
        }
        granted
    }

    /// Returns `runner`'s grant to the pool.
    pub fn release(&mut self, runner: InstanceId) {
        self.grants.retain(|(r, _)| *r != runner);
    }

    /// `runner`'s grant; 0 when it has none.
    pub fn granted(&self, runner: InstanceId) -> Bytes {
        self.grants
            .iter()
            .find(|(r, _)| *r == runner)
            .map_or(Bytes(0), |&(_, b)| b)
    }

    /// The sum of every grant.
    pub fn total_granted(&self) -> Bytes {
        self.grants
            .iter()
            .fold(Bytes(0), |sum, &(_, b)| add(sum, b))
    }

    /// The limit the ledger divides. `None` is unlimited.
    pub fn limit(&self) -> Option<Bytes> {
        self.limit
    }
}

/// `f · total`, with `f` clamped to [0, 1] and read to one part per million so that a decimal
/// fraction such as 0.85 gives the exact decimal product. NaN counts as 0.
fn fraction_of(total: Bytes, f: f32) -> Bytes {
    const PPM: u128 = 1_000_000;
    // NaN.clamp is NaN, and a NaN cast to an integer is 0.
    let ppm = (f64::from(f.clamp(0.0, 1.0)) * PPM as f64).round() as u128;
    Bytes((u128::from(total.0) * ppm / PPM) as u64)
}

fn add(a: Bytes, b: Bytes) -> Bytes {
    Bytes(a.0.saturating_add(b.0))
}

fn sub(a: Bytes, b: Bytes) -> Bytes {
    Bytes(a.0.saturating_sub(b.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_of_is_decimal_exact() {
        assert_eq!(fraction_of(Bytes(64 * GIB), 0.85), Bytes(58_411_555_225));
        assert_eq!(fraction_of(Bytes(100), 0.10), Bytes(10));
        assert_eq!(fraction_of(Bytes(100), 1.5), Bytes(100));
        assert_eq!(fraction_of(Bytes(100), -1.0), Bytes(0));
        assert_eq!(fraction_of(Bytes(100), f32::NAN), Bytes(0));
        assert_eq!(fraction_of(Bytes(u64::MAX), 1.0), Bytes(u64::MAX));
    }

    #[test]
    fn default_headroom_is_the_larger_term() {
        assert_eq!(default_headroom(Bytes(16 * GIB)), Bytes(4 * GIB));
        assert_eq!(default_headroom(Bytes(128 * GIB)), Bytes(13_743_895_347));
    }
}
