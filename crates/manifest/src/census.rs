// SPDX-License-Identifier: MIT OR Apache-2.0

//! The census's access-pattern classes (RFC-0001 §5.3 `TensorClass`; RFC-0041 §3).
//!
//! Here so far: [`TensorClass`], which a family's classification rules produce (RFC-0043 §4),
//! and [`TowerIx`]. The records, groups and totals of `TensorCensus` come with the census
//! builder (COMP-E10-T07).

use syntropy_core::{ExpertIx, LayerIx};

/// A multimodal encoder tower's index within its model (RFC-0043 §2.4). Reserved: no v1 family
/// places a tower.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TowerIx(pub u8);

/// A tensor's access pattern, which decides its home tier, caching and prefetch policy
/// (RFC-0001 §5.3). Owner: RFC-0041.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorClass {
    /// Read in full every token: attention and mixer weights, norms, dense FFNs, shared
    /// experts, hyper-connection weights.
    DenseEveryToken,
    /// One routed expert, read only when the router selects it.
    Expert {
        /// Its layer.
        layer: LayerIx,
        /// Its index within the layer.
        id: ExpertIx,
    },
    /// Tables addressed by row from token IDs: token embeddings, n-gram and per-layer
    /// embedding tables.
    RowLookup,
    /// The LM head and MTP output heads; held by the tail stage.
    Head,
    /// A MoE router.
    Router {
        /// Its layer.
        layer: LayerIx,
    },
    /// Reserved (RFC-0043 §2.4, §4.3): a multimodal encoder tower or projector. No v1 family
    /// produces it.
    Encoder {
        /// The tower.
        tower: TowerIx,
    },
}
