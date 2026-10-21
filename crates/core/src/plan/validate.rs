// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`PlacementPlan::validate`]: RFC-0060 §3.2's P-solver-1 invariants.
//!
//! A plan is valid when:
//! 1. the census has exactly one [`TierAssignment`] per tensor group, its home tier is on its
//!    owner (RFC-0001 §5.4), and it lists no replicas (see [`PlanViolation::ReplicasUnchecked`]);
//! 2. the stages are in ring order and their layer ranges cover `[0, n_layers)` exactly
//!    once, with `HEAD` on the first stage and `TAIL` on the last;
//! 3. the prologue's `RowLookup` groups are owned by the head's node, and `Head` groups and
//!    MTP modules' own `RowLookup` groups by the tail's node (RFC-0011 C6);
//! 4. every tier's usage is at most its `usable`, and every NUMA domain's host RAM passes
//!    RFC-0011 C9's joint host rows. Usage counts the groups homed on the tier, KV, scratch
//!    and expert-cache budgets, and the shared rows a tail holds (RFC-0011 C6, C9).
//!
//! The census lives in `syntropy-manifest`, above `core`, so the caller passes the facts the
//! check needs as a [`PlanBounds`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::ids::{DeviceId, LayerIx, NodeId, StageIx, TensorGroupId, TierId};
use crate::resource::{HostPoolInfo, TierKind};
use crate::units::Bytes;

use super::stage::StageRole;
use super::{PlacementPlan, TierAssignment};

/// What [`PlacementPlan::validate`] needs to know about one census group (RFC-0001 §5.3
/// `TensorGroup`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupFacts {
    /// The group.
    pub id: TensorGroupId,
    /// Which stage must hold it, if any.
    pub role: GroupRole,
    /// Its stored size.
    pub bytes: Bytes,
    /// `TensorGroup::shared_with` (RFC-0001 §5.3, RFC-0041 §3.1): the prologue's `RowLookup`
    /// group whose tensor record this group reads, for a tied LM head's `Head` group and for
    /// an MTP module's own `RowLookup` group (`TailLookup`).
    ///
    /// `validate` checks it: it must name a census group
    /// ([`PlanViolation::SharedWithUnknown`]), and on a `Head` or `TailLookup` group that
    /// group must be a prologue `RowLookup` ([`PlanViolation::SharedWithNotLookup`]).
    pub shared_with: Option<TensorGroupId>,
}

/// Which stage must hold a group (RFC-0011 C6), derived from its `TensorClass` (RFC-0001
/// §5.3).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupRole {
    /// A `TensorClass::RowLookup` group of the prologue (token, n-gram and per-layer
    /// embedding tables): owned by the head stage's node.
    RowLookup,
    /// A `TensorClass::RowLookup` group of an MTP module, that is, of a layer with
    /// `LayerSpec::mtp`. It is linked by `shared_with` to the prologue's group and is placed
    /// on the tail with the MTP layers (RFC-0011 C6, RFC-0041 §3.1, RFC-0043 §2.2 item 3).
    TailLookup,
    /// `TensorClass::Head`: owned by the tail stage's node.
    Head,
    /// Any other class.
    Other,
}

/// What a plan is checked against: the census and the budgets of the tiers it may use.
///
/// Every budget is C9's right-hand side (RFC-0011 §6): already net of the fixed reserves
/// (`reserve_i`) and of the bytes other instances hold on the tier.
#[derive(Clone, Copy, Debug)]
pub struct PlanBounds<'a> {
    /// The number of layers the stages must cover: decoder layers plus MTP modules, whose
    /// `LayerIx` values follow the last decoder layer (RFC-0001 §5.3).
    pub n_layers: u16,
    /// Every census group, in census order.
    pub groups: &'a [GroupFacts],
    /// Every tier the plan may use: its kind (`TierInfo::kind`) and its `TierInfo::usable`.
    pub usable: &'a [(TierId, TierKind, Bytes)],
    /// The tier that holds each device's scratch: its memory tier (`Vram` or `Unified`) for a
    /// GPU, its NUMA domain's host tier for a CPU.
    pub scratch_tiers: &'a [(DeviceId, TierId)],
    /// One record per NUMA domain (RFC-0010 §2.2), for C9's joint host rows. It must list
    /// every domain whose `HostPinned` or `HostPageable` tier the plan uses: a host tier the
    /// plan charges that no pool names fails with [`PlanViolation::HostTierWithoutPool`].
    pub host_pools: &'a [HostPoolInfo],
}

/// The first P-solver-1 invariant a plan breaks (RFC-0060 §3.2).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanViolation {
    /// A census group has no `TierAssignment`.
    GroupUnassigned {
        /// The group.
        group: TensorGroupId,
    },
    /// A census group has more than one `TierAssignment`.
    GroupAssignedTwice {
        /// The group.
        group: TensorGroupId,
    },
    /// A `TierAssignment` names a group the census does not have.
    UnknownGroup {
        /// The group.
        group: TensorGroupId,
    },
    /// A group's home tier is not on its owner node.
    HomeOffOwner {
        /// The group.
        group: TensorGroupId,
        /// Its owner.
        owner: NodeId,
        /// Its home tier.
        home: TierId,
    },
    /// A `TierAssignment` lists replicas, which `validate` cannot charge yet, so it fails
    /// closed. RFC-0011 §11.2 charges a replica's bytes "to that stage's tier in C9", but the
    /// plan records only the replica's `NodeId`, not the tier holding it. Lifting this is
    /// tracked in issue #1515.
    ReplicasUnchecked {
        /// The group.
        group: TensorGroupId,
    },
    /// A group's `shared_with` names a group the census does not have.
    SharedWithUnknown {
        /// The group.
        group: TensorGroupId,
        /// The group it names.
        twin: TensorGroupId,
    },
    /// A `Head` or `TailLookup` group's `shared_with` names a group that is not a prologue
    /// `RowLookup` group.
    SharedWithNotLookup {
        /// The group.
        group: TensorGroupId,
        /// The group it names.
        twin: TensorGroupId,
    },
    /// The plan has no stages.
    NoStages,
    /// A stage's index differs from its position in the ring.
    StageOutOfOrder {
        /// Its position in `PlacementPlan::stages`.
        position: usize,
        /// The index it carries.
        stage: StageIx,
    },
    /// A stage's range does not start where the previous one ended: a gap or an overlap.
    LayerGap {
        /// The stage.
        stage: StageIx,
        /// Where its range should start.
        expected: LayerIx,
        /// Where it starts.
        found: LayerIx,
    },
    /// A stage's layer range is empty or reversed.
    EmptyStage {
        /// The stage.
        stage: StageIx,
    },
    /// The last stage's range does not end at `n_layers`.
    LayersEnd {
        /// `n_layers`.
        expected: LayerIx,
        /// Where the last range ends.
        found: LayerIx,
    },
    /// `HEAD` is not exactly on the first stage, or `TAIL` not exactly on the last.
    RoleMismatch {
        /// The stage.
        stage: StageIx,
        /// Its roles.
        role: StageRole,
    },
    /// A prologue `RowLookup` group is not owned by the head's node.
    LookupOffHead {
        /// The group.
        group: TensorGroupId,
        /// Its owner.
        owner: NodeId,
        /// The head's node.
        head: NodeId,
    },
    /// An MTP module's `RowLookup` group is not owned by the tail's node.
    LookupOffTail {
        /// The group.
        group: TensorGroupId,
        /// Its owner.
        owner: NodeId,
        /// The tail's node.
        tail: NodeId,
    },
    /// A `Head` group is not owned by the tail's node.
    HeadOffTail {
        /// The group.
        group: TensorGroupId,
        /// Its owner.
        owner: NodeId,
        /// The tail's node.
        tail: NodeId,
    },
    /// The plan uses a tier with no entry in `PlanBounds::usable`.
    UnknownTier {
        /// The tier.
        tier: TierId,
    },
    /// The plan gives scratch to a device with no entry in `PlanBounds::scratch_tiers`.
    UnknownDevice {
        /// The device.
        device: DeviceId,
    },
    /// A tier holds more than its `usable` bytes.
    OverCapacity {
        /// The tier.
        tier: TierId,
        /// What the plan puts there.
        used: Bytes,
        /// What it may hold.
        usable: Bytes,
    },
    /// The plan charges a `HostPinned` or `HostPageable` tier that no `PlanBounds::host_pools`
    /// record names, so C9's joint host rows cannot be checked for it.
    HostTierWithoutPool {
        /// The tier.
        tier: TierId,
    },
    /// A NUMA domain's `HostPinned` tier holds more than the domain's `pin_limit` (RFC-0011
    /// C9's first joint host row).
    HostPinOverLimit {
        /// The `HostPinned` tier.
        tier: TierId,
        /// What the plan puts there.
        used: Bytes,
        /// The domain's `pin_limit`.
        pin_limit: Bytes,
    },
    /// A NUMA domain's `HostPinned` and `HostPageable` tiers together hold more than the
    /// domain's `host_usable` (RFC-0011 C9's second joint host row).
    HostPoolOverCapacity {
        /// `[HostPinned, HostPageable]` of the domain.
        tiers: [TierId; 2],
        /// What the plan puts in both.
        used: Bytes,
        /// The domain's `host_usable`.
        usable: Bytes,
    },
}

impl fmt::Display for PlanViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GroupUnassigned { group } => {
                write!(f, "tensor group {} has no tier assignment", group.0)
            }
            Self::GroupAssignedTwice { group } => {
                write!(
                    f,
                    "tensor group {} has more than one tier assignment",
                    group.0
                )
            }
            Self::UnknownGroup { group } => {
                write!(
                    f,
                    "tier assignment names tensor group {}, not in the census",
                    group.0
                )
            }
            Self::HomeOffOwner { group, home, .. } => write!(
                f,
                "tensor group {} is homed on tier {} of another node than its owner",
                group.0, home.ordinal
            ),
            Self::ReplicasUnchecked { group } => write!(
                f,
                "tensor group {} lists replicas, whose tiers the plan does not record",
                group.0
            ),
            Self::SharedWithUnknown { group, twin } => write!(
                f,
                "tensor group {} shares rows with group {}, not in the census",
                group.0, twin.0
            ),
            Self::SharedWithNotLookup { group, twin } => write!(
                f,
                "tensor group {} shares rows with group {}, which is not a prologue row lookup",
                group.0, twin.0
            ),
            Self::NoStages => f.write_str("the plan has no stages"),
            Self::StageOutOfOrder { position, stage } => write!(
                f,
                "stage at ring position {position} carries index {}",
                stage.0
            ),
            Self::LayerGap {
                stage,
                expected,
                found,
            } => write!(
                f,
                "stage {} starts at layer {}, expected {}",
                stage.0, found.0, expected.0
            ),
            Self::EmptyStage { stage } => write!(f, "stage {} has an empty layer range", stage.0),
            Self::LayersEnd { expected, found } => write!(
                f,
                "the last stage ends at layer {}, expected {}",
                found.0, expected.0
            ),
            Self::RoleMismatch { stage, role } => write!(
                f,
                "stage {} has roles {role:?}; HEAD belongs on the first stage only and TAIL on \
                 the last only",
                stage.0
            ),
            Self::LookupOffHead { group, .. } => write!(
                f,
                "row-lookup group {} is not owned by the head's node",
                group.0
            ),
            Self::LookupOffTail { group, .. } => write!(
                f,
                "MTP row-lookup group {} is not owned by the tail's node",
                group.0
            ),
            Self::HeadOffTail { group, .. } => {
                write!(f, "head group {} is not owned by the tail's node", group.0)
            }
            Self::UnknownTier { tier } => {
                write!(
                    f,
                    "the plan uses tier {} with no usable budget",
                    tier.ordinal
                )
            }
            Self::UnknownDevice { device } => write!(
                f,
                "the plan gives scratch to device {} with no scratch tier",
                device.ordinal
            ),
            Self::OverCapacity { tier, used, usable } => write!(
                f,
                "tier {} holds {} bytes, above its usable {}",
                tier.ordinal, used.0, usable.0
            ),
            Self::HostTierWithoutPool { tier } => write!(
                f,
                "host tier {} is used but no host pool record names it",
                tier.ordinal
            ),
            Self::HostPinOverLimit {
                tier,
                used,
                pin_limit,
            } => write!(
                f,
                "pinned tier {} holds {} bytes, above its domain's pin limit {}",
                tier.ordinal, used.0, pin_limit.0
            ),
            Self::HostPoolOverCapacity {
                tiers,
                used,
                usable,
            } => write!(
                f,
                "host tiers {} and {} hold {} bytes together, above their domain's usable {}",
                tiers[0].ordinal, tiers[1].ordinal, used.0, usable.0
            ),
        }
    }
}

impl std::error::Error for PlanViolation {}

impl PlacementPlan {
    /// Checks RFC-0060 §3.2's P-solver-1 invariants against the census and tier budgets in
    /// `bounds`, and returns the first one the plan breaks. Checks run in the module's order;
    /// within one check, in census, ring or tier order, so the result is deterministic.
    ///
    /// A plan whose `TierAssignment`s list replicas fails with
    /// [`PlanViolation::ReplicasUnchecked`] until the plan records where replicas sit
    /// (RFC-0011 §11.2; M2).
    pub fn validate(&self, bounds: &PlanBounds<'_>) -> Result<(), PlanViolation> {
        let assigned = self.check_assignments(bounds)?;
        self.check_stages(bounds.n_layers)?;
        self.check_head_and_tail(bounds, &assigned)?;
        self.check_usage(bounds, &assigned)
    }

    /// Invariant 1: one `TierAssignment` per census group, homed on its owner, no replicas;
    /// and every `shared_with` names a census group, a prologue `RowLookup` for tail-side
    /// groups.
    fn check_assignments(
        &self,
        bounds: &PlanBounds<'_>,
    ) -> Result<BTreeMap<TensorGroupId, &TierAssignment>, PlanViolation> {
        let census: BTreeSet<TensorGroupId> = bounds.groups.iter().map(|g| g.id).collect();
        let mut assigned = BTreeMap::new();
        for t in &self.tensors {
            if !census.contains(&t.group) {
                return Err(PlanViolation::UnknownGroup { group: t.group });
            }
            if assigned.insert(t.group, t).is_some() {
                return Err(PlanViolation::GroupAssignedTwice { group: t.group });
            }
            if t.home.node != t.owner {
                return Err(PlanViolation::HomeOffOwner {
                    group: t.group,
                    owner: t.owner,
                    home: t.home,
                });
            }
            if !t.replicas.is_empty() {
                return Err(PlanViolation::ReplicasUnchecked { group: t.group });
            }
        }
        if let Some(g) = bounds.groups.iter().find(|g| !assigned.contains_key(&g.id)) {
            return Err(PlanViolation::GroupUnassigned { group: g.id });
        }
        let roles: BTreeMap<TensorGroupId, GroupRole> =
            bounds.groups.iter().map(|g| (g.id, g.role)).collect();
        for g in bounds.groups {
            let Some(twin) = g.shared_with else { continue };
            let Some(&twin_role) = roles.get(&twin) else {
                return Err(PlanViolation::SharedWithUnknown { group: g.id, twin });
            };
            let tail_side = matches!(g.role, GroupRole::Head | GroupRole::TailLookup);
            if tail_side && twin_role != GroupRole::RowLookup {
                return Err(PlanViolation::SharedWithNotLookup { group: g.id, twin });
            }
        }
        Ok(assigned)
    }

    /// Invariant 2: ring order, exact layer cover, head and tail roles.
    fn check_stages(&self, n_layers: u16) -> Result<(), PlanViolation> {
        let last = match self.stages.len() {
            0 => return Err(PlanViolation::NoStages),
            n => n - 1,
        };
        let mut next = LayerIx(0);
        for (position, s) in self.stages.iter().enumerate() {
            if usize::from(s.stage.0) != position {
                return Err(PlanViolation::StageOutOfOrder {
                    position,
                    stage: s.stage,
                });
            }
            if s.layers.start != next {
                return Err(PlanViolation::LayerGap {
                    stage: s.stage,
                    expected: next,
                    found: s.layers.start,
                });
            }
            if s.layers.end <= s.layers.start {
                return Err(PlanViolation::EmptyStage { stage: s.stage });
            }
            next = s.layers.end;
            let head_ok = s.role.contains(StageRole::HEAD) == (position == 0);
            let tail_ok = s.role.contains(StageRole::TAIL) == (position == last);
            if !(head_ok && tail_ok) {
                return Err(PlanViolation::RoleMismatch {
                    stage: s.stage,
                    role: s.role,
                });
            }
        }
        if next != LayerIx(n_layers) {
            return Err(PlanViolation::LayersEnd {
                expected: LayerIx(n_layers),
                found: next,
            });
        }
        Ok(())
    }

    /// Invariant 3: prologue lookups on the head's node; heads and MTP lookups on the tail's
    /// (RFC-0011 C6).
    fn check_head_and_tail(
        &self,
        bounds: &PlanBounds<'_>,
        assigned: &BTreeMap<TensorGroupId, &TierAssignment>,
    ) -> Result<(), PlanViolation> {
        // check_stages has run, so there is at least one stage.
        let (Some(head), Some(tail)) = (self.stages.first(), self.stages.last()) else {
            return Err(PlanViolation::NoStages);
        };
        for g in bounds.groups {
            let owner = assigned[&g.id].owner;
            match g.role {
                GroupRole::RowLookup if owner != head.node => {
                    return Err(PlanViolation::LookupOffHead {
                        group: g.id,
                        owner,
                        head: head.node,
                    });
                }
                GroupRole::TailLookup if owner != tail.node => {
                    return Err(PlanViolation::LookupOffTail {
                        group: g.id,
                        owner,
                        tail: tail.node,
                    });
                }
                GroupRole::Head if owner != tail.node => {
                    return Err(PlanViolation::HeadOffTail {
                        group: g.id,
                        owner,
                        tail: tail.node,
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Invariant 4: every tier's usage at most its `usable`, and C9's joint host rows.
    ///
    /// Shared rows follow RFC-0011 C6 and C9. A tail-side group that reads the prologue's
    /// embedding record (a tied LM head's `Head` group, an MTP module's `RowLookup` group:
    /// `shared_with` is set and the role is `Head` or `TailLookup`) is C9's duplicated-row
    /// term `E_dup · e_p`. `e_p` is 1 only when the tail is not the head, so "a one-stage
    /// plan counts the shared rows once" (C6): there the group adds nothing, whichever tier it
    /// is homed on. In a plan with two or more stages it is charged in full.
    ///
    /// `validate` charges duplicated rows to the tier the plan records as the group's home.
    /// C9 charges `E_dup` to the tail's `HostPageable` and a head group's `Hd` at its own tier,
    /// and RFC-0011 does not yet say which applies to a tied head, so this can be stricter
    /// than C9. The open question is tracked in issue #1515. `e_p` is taken from the stage
    /// count, not the placement node, so a `MultiStageOnNode` plan (PLC-E4-T15) would be
    /// charged the rows twice where C9 charges them once, which is conservative.
    fn check_usage(
        &self,
        bounds: &PlanBounds<'_>,
        assigned: &BTreeMap<TensorGroupId, &TierAssignment>,
    ) -> Result<(), PlanViolation> {
        let mut used: BTreeMap<TierId, Bytes> = BTreeMap::new();
        let mut charge = |tier: TierId, bytes: Bytes| {
            let total = used.entry(tier).or_default();
            *total = Bytes(total.0.saturating_add(bytes.0));
        };

        let tail_is_head = self.stages.len() == 1;
        for g in bounds.groups {
            let duplicated_row = g.shared_with.is_some()
                && matches!(g.role, GroupRole::Head | GroupRole::TailLookup);
            if duplicated_row && tail_is_head {
                continue;
            }
            charge(assigned[&g.id].home, g.bytes);
        }
        for kv in &self.kv {
            charge(kv.tier, kv.bytes);
        }
        for cache in &self.expert_cache {
            charge(cache.tier, cache.bytes);
        }
        for &(device, bytes) in &self.scratch {
            let tier = bounds
                .scratch_tiers
                .iter()
                .find(|(d, _)| *d == device)
                .map(|&(_, t)| t)
                .ok_or(PlanViolation::UnknownDevice { device })?;
            charge(tier, bytes);
        }

        for (&tier, &used) in &used {
            let (kind, usable) = bounds
                .usable
                .iter()
                .find(|(t, _, _)| *t == tier)
                .map(|&(_, k, b)| (k, b))
                .ok_or(PlanViolation::UnknownTier { tier })?;
            if used > usable {
                return Err(PlanViolation::OverCapacity { tier, used, usable });
            }
            let host = matches!(kind, TierKind::HostPinned | TierKind::HostPageable);
            if host && !bounds.host_pools.iter().any(|p| p.tiers.contains(&tier)) {
                return Err(PlanViolation::HostTierWithoutPool { tier });
            }
        }

        // C9's joint host rows, per NUMA domain (RFC-0001 §5.2, RFC-0020 §5).
        let on = |tier: TierId| used.get(&tier).copied().unwrap_or_default();
        for pool in bounds.host_pools {
            let [pinned, pageable] = pool.tiers;
            if on(pinned) > pool.pin_limit {
                return Err(PlanViolation::HostPinOverLimit {
                    tier: pinned,
                    used: on(pinned),
                    pin_limit: pool.pin_limit,
                });
            }
            let both = Bytes(on(pinned).0.saturating_add(on(pageable).0));
            if both > pool.usable {
                return Err(PlanViolation::HostPoolOverCapacity {
                    tiers: pool.tiers,
                    used: both,
                    usable: pool.usable,
                });
            }
        }
        Ok(())
    }
}
