// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`PlacementPlan::validate`]: RFC-0060 §3.2's P-solver-1 invariants.
//!
//! A plan is valid when:
//! 1. the census has exactly one [`TierAssignment`] per tensor group, and its home tier is
//!    on its owner (RFC-0001 §5.4);
//! 2. the stages are in ring order and their layer ranges cover `[0, n_layers)` exactly
//!    once, with `HEAD` on the first stage and `TAIL` on the last;
//! 3. every `RowLookup` group is owned by the head's node and every `Head` group by the
//!    tail's node;
//! 4. on every node, every tier's usage is at most its `usable`, counting the groups homed
//!    there, KV, scratch and expert-cache budgets, and both copies of a tied embedding and
//!    head that sit on different tiers.
//!
//! The census lives in `syntropy-manifest`, above `core`, so the caller passes the facts the
//! check needs as a [`PlanBounds`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::ids::{DeviceId, LayerIx, NodeId, StageIx, TensorGroupId, TierId};
use crate::units::Bytes;

use super::stage::StageRole;
use super::{PlacementPlan, TierAssignment};

/// What [`PlacementPlan::validate`] needs to know about one census group (RFC-0001 §5.3
/// `TensorGroup`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupFacts {
    /// The group.
    pub id: TensorGroupId,
    /// Whether its placement is pinned to the head or the tail.
    pub role: GroupRole,
    /// Its stored size.
    pub bytes: Bytes,
    /// The group that shares its tensor record: a tied embedding's `RowLookup` group and the
    /// `Head` group (RFC-0001 §5.3 `TensorGroup::shared_with`).
    pub shared_with: Option<TensorGroupId>,
}

/// The part of a group's `TensorClass` that P-solver-1 constrains (RFC-0001 §5.3).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroupRole {
    /// `TensorClass::RowLookup`: must be owned by the head stage's node.
    RowLookup,
    /// `TensorClass::Head`: must be owned by the tail stage's node.
    Head,
    /// Any other class.
    Other,
}

/// What a plan is checked against: the census and the tiers' budgets.
#[derive(Clone, Copy, Debug)]
pub struct PlanBounds<'a> {
    /// The number of layers the stages must cover: decoder layers plus MTP modules, whose
    /// `LayerIx` values follow the last decoder layer (RFC-0001 §5.3).
    pub n_layers: u16,
    /// Every census group, in census order.
    pub groups: &'a [GroupFacts],
    /// `TierInfo::usable` of every tier the plan may use.
    pub usable: &'a [(TierId, Bytes)],
    /// The tier that holds each device's scratch: its memory tier (`Vram` or `Unified`) for a
    /// GPU, its NUMA domain's host tier for a CPU.
    pub scratch_tiers: &'a [(DeviceId, TierId)],
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
    /// A `RowLookup` group is not owned by the head's node.
    LookupOffHead {
        /// The group.
        group: TensorGroupId,
        /// Its owner.
        owner: NodeId,
        /// The head's node.
        head: NodeId,
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
        }
    }
}

impl std::error::Error for PlanViolation {}

impl PlacementPlan {
    /// Checks RFC-0060 §3.2's P-solver-1 invariants against the census and tier budgets in
    /// `bounds`, and returns the first one the plan breaks. Checks run in the module's order;
    /// within one check, in census, ring or tier order, so the result is deterministic.
    pub fn validate(&self, bounds: &PlanBounds<'_>) -> Result<(), PlanViolation> {
        let assigned = self.check_assignments(bounds)?;
        self.check_stages(bounds.n_layers)?;
        self.check_head_and_tail(bounds, &assigned)?;
        self.check_usage(bounds, &assigned)
    }

    /// Invariant 1: one `TierAssignment` per census group, homed on its owner.
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
        }
        if let Some(g) = bounds.groups.iter().find(|g| !assigned.contains_key(&g.id)) {
            return Err(PlanViolation::GroupUnassigned { group: g.id });
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

    /// Invariant 3: `RowLookup` groups on the head's node, `Head` groups on the tail's.
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

    /// Invariant 4: every tier's usage at most its `usable`.
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

        for g in bounds.groups {
            let home = assigned[&g.id].home;
            // A tied embedding and head share one tensor record. Two copies are resident only
            // when the two groups sit on different tiers; on one tier the record is held once,
            // charged to the lower group ID.
            let held_by_twin = g.shared_with.is_some_and(|twin| {
                twin < g.id && assigned.get(&twin).is_some_and(|t| t.home == home)
            });
            if !held_by_twin {
                charge(home, g.bytes);
            }
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

        for (tier, used) in used {
            let usable = bounds
                .usable
                .iter()
                .find(|(t, _)| *t == tier)
                .map(|&(_, b)| b)
                .ok_or(PlanViolation::UnknownTier { tier })?;
            if used > usable {
                return Err(PlanViolation::OverCapacity { tier, used, usable });
            }
        }
        Ok(())
    }
}
