// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`InstancePolicy`], the instance-scoped bundle `EnsureInstance` carries, and its parts
//! (RFC-0001 §11.3, RFC-0054 §4.5, §6.2, §6.3). Owner: RFC-0054 (type and construction),
//! RFC-0011 (how the solver uses each field).
//!
//! Pins and tensor overrides are data here, as the policy tables write them. PROD-E1-T09
//! compiles them against a model's census into solver constraints.

use std::collections::BTreeMap;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use super::resource::ResourceOverlay;
use super::value::{ConfigValue, DeviceSelector, KeyPath, LayerRange, NodeSelector, TierSelector};
use super::{ExactnessPolicy, Fidelity, Objective};
use crate::canon;
use crate::ids::{Blake3, ExpertIx};
use crate::plan::{SpeculativeConfig, StageRole};
use crate::units::Nanos;

/// Carried by `EnsureInstance`, which is idempotent by `(ModelId, digest)`
/// (RFC-0001 §11.3, RFC-0054 §4.5).
///
/// It holds only what the policy tables and an explicit `EnsureInstance` set, never a
/// node-computed default, so every ingress node builds the same policy for a model and the
/// same [`canonical_bytes`](InstancePolicy::canonical_bytes), and so the same digest.
/// `default_fidelity` is amendment A6's, and `context` and the fields after it are RFC-0054
/// §4.5's additions. Every field but `objective` and `exactness` decodes to its default when
/// absent.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstancePolicy {
    /// What the solver and scheduler optimise.
    pub objective: Objective,
    /// The knobs this instance allows.
    pub exactness: ExactnessPolicy,
    /// What a request that states no fidelity gets. `Exact` unless set; `Approximate`
    /// requires an `Approximate` `exactness` (RFC-0054 §5.4, §7).
    #[serde(default)]
    pub default_fidelity: Fidelity,
    /// The instance's narrowing of each stage node's `ResourcePolicy` (amendment A9).
    #[serde(default)]
    pub resources: ResourceOverlay,
    /// Operator overrides the solver treats as constraints (RFC-0054 §6.2).
    #[serde(default)]
    pub pins: Vec<PlacementPin>,
    /// Regexes on tensor names, compiled to `Tier` and `Compute` pins (RFC-0054 §6.3).
    #[serde(default)]
    pub tensor_overrides: Vec<TensorOverride>,
    /// Speculative decoding.
    #[serde(default)]
    pub speculative: SpeculativeMode,
    /// Instance-scope keys that domain RFCs declare (for example RFC-0011
    /// `solver.context_target`, RFC-0032 `ring.max_inflight_steps`, RFC-0033
    /// `recovery.history`), by key path. Covered by the digest.
    #[serde(default)]
    pub domain: BTreeMap<KeyPath, ConfigValue>,
    /// Context length and sequence count.
    #[serde(default)]
    pub context: ContextPolicy,
    /// The nodes this instance may use.
    #[serde(default)]
    pub nodes: NodeFilter,
    /// How long an idle instance stays loaded.
    #[serde(default)]
    pub keep_alive: KeepAlive,
    /// Load at daemon start; never evicted for another model.
    #[serde(default)]
    pub preload: bool,
    /// Deterministic batching (ADR-0019 decision 8; M3).
    #[serde(default)]
    pub deterministic: bool,
    /// Whether dense weights may have their home on `Nvme` (ADR-0008).
    #[serde(default)]
    pub dense_on_nvme: DenseNvmePolicy,
}

impl InstancePolicy {
    /// The canonical serialisation ([`canon`]): the bytes the digest is computed over. Two
    /// nodes with the same cluster config build the same policy for automatic placement, and so
    /// the same bytes (RFC-0054 §4.5).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        // Nothing in an `InstancePolicy` serialises a 128-bit integer or reports a custom
        // error, the only failures of the canonical encoding.
        canon::to_bytes(self).expect("an InstancePolicy always has a canonical encoding")
    }

    /// The policy's digest: `hash` over [`canonical_bytes`](InstancePolicy::canonical_bytes).
    /// `EnsureInstance` is idempotent by `(ModelId, digest)`. RFC-0054 §4.5 makes the digest
    /// BLAKE3; `core` links no hasher, so the crate that links BLAKE3 passes it in
    /// (RFC-0054 §4.5, note of 2026-10-22).
    pub fn digest_with(&self, hash: impl FnOnce(&[u8]) -> Blake3) -> Blake3 {
        hash(&self.canonical_bytes())
    }
}

/// Speculative decoding for an instance (RFC-0001 §11.3; RFC-0054 §5.4 `speculative.mode`).
/// Owner: RFC-0012.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeculativeMode {
    /// The solver decides. The default.
    #[default]
    Auto,
    /// No speculative decoding.
    Off,
    /// This drafter, whether or not the solver would choose it.
    Force(SpeculativeConfig),
}

/// Context length and sequence count (RFC-0054 §4.5, §5.4). `None` lets the solver choose.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ContextPolicy {
    /// `context_length`; `None` is `auto`: what the committed plan can hold for
    /// `max_sequences`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    /// `max_sequences`: how many sequences the instance holds KV for; `None` is `auto`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_sequences: Option<u16>,
}

/// The nodes an instance may use: those matching an `include` selector and no `exclude`
/// selector (RFC-0054 §5.4 `nodes.include`, `nodes.exclude`; §6.4). Default: every node.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeFilter {
    /// Selectors a node must match one of.
    pub include: Vec<NodeSelector>,
    /// Selectors a node must match none of.
    pub exclude: Vec<NodeSelector>,
}

impl Default for NodeFilter {
    fn default() -> NodeFilter {
        NodeFilter {
            include: vec![NodeSelector::Any],
            exclude: Vec::new(),
        }
    }
}

/// How long an idle instance stays loaded (RFC-0054 §5.4 `keep_alive`).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum KeepAlive {
    /// `max(1 h, 60 × t_reload)` \[target\], with `t_reload` measured per model. The default.
    #[default]
    Auto,
    /// This long.
    For(Nanos),
    /// Never unloaded for idleness.
    Never,
}

/// Whether dense weights may have their home on `Nvme` (RFC-0054 §5.4 `dense_on_nvme`;
/// ADR-0008).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum DenseNvmePolicy {
    /// Never.
    Never,
    /// Only when no placement without it exists. The default.
    #[default]
    IfOnlyOption,
    /// Whenever the solver finds it better.
    Allow,
}

/// A placement constraint or preference (RFC-0054 §6.2). Pins never post-edit a solved plan
/// (ADR-0008 decision 8). Owner: RFC-0054 (syntax and meaning), RFC-0011 (encoding as MILP
/// constraints).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacementPin {
    /// What the pin constrains.
    pub kind: PinKind,
    /// Whether the solver must satisfy it.
    #[serde(default)]
    pub strength: PinStrength,
}

/// Whether the solver must satisfy a pin (RFC-0054 §6.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum PinStrength {
    /// A constraint. The default.
    #[default]
    Require,
    /// A preference the solver pays `weight` to break; `config`'s default weight is 1.0 \[target\],
    /// one TPOT-millisecond.
    Prefer {
        /// The penalty, in TPOT-milliseconds.
        weight: f32,
    },
}

/// What a pin constrains (RFC-0054 §6.2). Layer ranges are held unresolved, as written.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinKind {
    /// Layers in `layers` belong to stages on a node matching `node`.
    Layers {
        /// The layers.
        layers: LayerRange,
        /// The node.
        node: NodeSelector,
    },
    /// The selected groups have their home on this tier, optionally on a specific node or
    /// tier.
    Tier {
        /// The groups.
        groups: GroupSelector,
        /// The home tier.
        tier: TierSelector,
    },
    /// The selected groups execute on this device at decode (`ComputeSite::decode`).
    Compute {
        /// The groups.
        groups: GroupSelector,
        /// The decode device.
        device: DeviceSelector,
    },
    /// The HEAD, TAIL or DRAFT role runs on a matching node.
    Role {
        /// The role.
        role: StageRole,
        /// The node.
        node: NodeSelector,
    },
    /// No stage, home tier or compute site on matching nodes or devices.
    Exclude {
        /// What is excluded.
        target: ExcludeTarget,
    },
}

/// Tensor groups selected by class, and optionally by layer and expert range (RFC-0054
/// §6.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupSelector {
    /// The groups' class.
    pub class: ClassSel,
    /// Only groups in these layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<LayerRange>,
    /// Only these experts, for expert groups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experts: Option<Range<ExpertIx>>,
}

/// A group class a pin selects (RFC-0054 §6.2 `groups`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ClassSel {
    /// `"dense"`.
    Dense,
    /// `"expert"`.
    Expert,
    /// `"lookup"`.
    Lookup,
    /// `"head"`.
    Head,
    /// `"router"`.
    Router,
    /// `"attention"`.
    Attention,
    /// `"ffn"`.
    Ffn,
    /// `"all"`.
    All,
}

/// What an `Exclude` pin keeps placement off (RFC-0054 §6.2).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExcludeTarget {
    /// Matching nodes: no stage, home tier or compute site on them.
    Node(NodeSelector),
    /// Matching devices: no home tier or compute site on them.
    Device(DeviceSelector),
}

/// A regular expression over source-format tensor names, with a home tier and an optional
/// decode device (RFC-0054 §6.3). PROD-E1-T09 compiles it to `Tier` and `Compute` pins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorOverride {
    /// The `match` pattern's source text, matched against whole names. `config` validates and
    /// compiles it.
    pub pattern: String,
    /// Only tensors in these layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<LayerRange>,
    /// Only these experts of a fused expert tensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experts: Option<Range<ExpertIx>>,
    /// The selected groups' home tier.
    pub tier: TierSelector,
    /// The selected groups' decode device, if set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute: Option<DeviceSelector>,
}
