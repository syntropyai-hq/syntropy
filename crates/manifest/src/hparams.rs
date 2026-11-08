// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HyperParams`, the hyperparameters a family module reads from a checkpoint's metadata
//! (RFC-0001 §5.3; RFC-0043 §5.2).
//!
//! RFC-0043 §5.2 fixes the shared fields every family fills: `n_layers`, `hidden`, `vocab`,
//! `residual_streams` and per layer a `LayerSpec`. `LayerSpec`, with its `MixerKind` and
//! `FfnKind`, carries the block parameters of RFC-0043 §3 and comes with the blocks
//! (COMP-E15-T02); until then a family keeps its per-layer values in [`HyperParams::extra`].
//! `ModelId` never hashes this type (RFC-0041 §1), so it can grow between releases.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The [`HyperParams::extra`] key a family stores its routed-expert count under, which the
/// classification engine reads to bound expert indices and check stacked expert tensors
/// (RFC-0043 §4.1).
pub const N_EXPERTS: &str = "n_experts";

/// The [`HyperParams::extra`] key a family stores its routed experts per token under, which the
/// census reads for an expert group's `read_per_token` prior (RFC-0041 §3.1).
pub const TOP_K: &str = "top_k";

/// One family-specific hyperparameter value.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HParamValue {
    /// A count or size.
    U64(u64),
    /// A scale, eps or base.
    F64(f64),
    /// A flag.
    Bool(bool),
    /// A name, such as a RoPE scaling type.
    Str(String),
    /// A per-layer list, such as window sizes.
    U64s(Vec<u64>),
}

/// A model's hyperparameters (RFC-0001 §5.3). Owner: RFC-0043.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HyperParams {
    /// Decoder layers, MTP modules not included.
    pub n_layers: u16,
    /// The hidden size.
    pub hidden: u32,
    /// The vocabulary size.
    pub vocab: u32,
    /// Residual streams; above 1 for hyper-connection models.
    pub residual_streams: u8,
    /// Family-specific values, by name.
    pub extra: BTreeMap<String, HParamValue>,
}

impl HyperParams {
    /// The shared fields, one residual stream and no extra values.
    #[must_use]
    pub fn new(n_layers: u16, hidden: u32, vocab: u32) -> Self {
        Self {
            n_layers,
            hidden,
            vocab,
            residual_streams: 1,
            extra: BTreeMap::new(),
        }
    }

    /// Sets the extra value `key`.
    #[must_use]
    pub fn with(mut self, key: &str, value: HParamValue) -> Self {
        self.extra.insert(key.to_owned(), value);
        self
    }

    /// The count named `key`: `n_layers`, `hidden`, `vocab`, `residual_streams`, or an extra
    /// [`HParamValue::U64`]. `None` for any other name or type.
    #[must_use]
    pub fn count(&self, key: &str) -> Option<u64> {
        match key {
            "n_layers" => Some(self.n_layers.into()),
            "hidden" => Some(self.hidden.into()),
            "vocab" => Some(self.vocab.into()),
            "residual_streams" => Some(self.residual_streams.into()),
            _ => match self.extra.get(key) {
                Some(HParamValue::U64(n)) => Some(*n),
                _ => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_read_shared_fields_and_extra_values() {
        let hp = HyperParams::new(2, 64, 256)
            .with(N_EXPERTS, HParamValue::U64(8))
            .with("rope_theta", HParamValue::F64(1e4));
        assert_eq!(hp.count("n_layers"), Some(2));
        assert_eq!(hp.count("hidden"), Some(64));
        assert_eq!(hp.count("vocab"), Some(256));
        assert_eq!(hp.count("residual_streams"), Some(1));
        assert_eq!(hp.count(N_EXPERTS), Some(8));
        assert_eq!(hp.count("rope_theta"), None, "not a count");
        assert_eq!(hp.count("missing"), None);
    }
}
