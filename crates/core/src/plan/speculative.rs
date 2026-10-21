// SPDX-License-Identifier: MIT OR Apache-2.0

//! Speculative decoding in a plan (RFC-0001 §5.4). Owner: RFC-0012.

use serde::{Deserialize, Serialize};

use crate::ids::{DeviceId, ModelId, TensorGroupId};

/// The plan's speculative decoding setup (RFC-0001 §5.4). Owner: RFC-0012.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// The drafter.
    pub source: DraftSource,
    /// γ_max. A verify window holds γ+1 positions, so graphs, KV provisional slots and the
    /// `Verify` bucket are sized for `T_max = max_window + 1`, derived wherever needed and
    /// never stored in a second field. Deltas may lower it, never raise it.
    pub max_window: u16,
    /// The draft tree; `None` for a chain.
    pub tree: Option<TreeShape>,
    /// How drafted tokens are accepted.
    pub acceptance: Acceptance,
}

/// Where draft tokens come from (RFC-0001 §5.4; RFC-0012 §9.1 says where each runs).
/// Owner: RFC-0012.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DraftSource {
    /// The model's native MTP modules, on the tail.
    Mtp {
        /// How many MTP heads are chained.
        heads: u8,
    },
    /// A draft head supplied with the model, on the tail. v1 produces none.
    DraftHead {
        /// The census group holding the head.
        group: TensorGroupId,
    },
    /// A separate draft model, on the head's fastest device.
    DraftModel {
        /// The draft model.
        model: ModelId,
        /// The device it runs on.
        device: DeviceId,
    },
}

/// The shape of a draft tree (RFC-0001 §5.4 names it; no RFC defines its fields yet). The
/// minimal form follows the draft tree of RFC-0031 §6.5's `Token` frame: one parent index
/// per draft node. Owner: RFC-0012.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeShape {
    /// Per draft node, in node order, the index of its parent node.
    pub parents: Vec<u16>,
}

/// How drafted tokens are accepted (RFC-0001 §5.4; ADR-0017). Owner: RFC-0012.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Acceptance {
    /// A drafted token survives only if it equals what the target would have produced at that
    /// position under the sequence's sampler. The default, and the only exact choice.
    #[default]
    Exact,
}
