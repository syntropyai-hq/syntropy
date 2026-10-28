// SPDX-License-Identifier: MIT OR Apache-2.0

//! `KvFormula`: the KV and recurrent state a model keeps per layer, and the bytes it stores and
//! reads (RFC-0001 §5.3; RFC-0022 §2, §5.4). Owner: RFC-0022.
//!
//! A family builds one from its hyperparameters (RFC-0043): [`LayerKv::gqa`] and
//! [`LayerKv::mla`] give the per-token entry of the two common attention kinds. Two functions turn
//! the formula into bytes, and every party sizes KV through them, so there is one formula:
//! - [`KvFormula::stored`]: what a sequence of `tokens` tokens keeps on a range of layers, which
//!   KV budgets and the solver's store term use;
//! - [`KvFormula::read_per_step`]: what one decode step at `context` tokens reads on that range,
//!   the `B_vram + B_host` of RFC-0022 §5.4's read term.
//!
//! Both take the plan's [`KvDType`] (RFC-0022 §9). `Native` stores each layer in the reference's
//! KV dtype ([`KvFormula::dtype`]). `Q8` and `K8V4` store per-token, sparse and compressed entries
//! below it. Recurrent state (`Fixed`) is never quantized (RFC-0022 §9), and an auxiliary cache
//! keeps its own dtype ([`AuxKv::dtype`]).
//!
//! [`dtype`] holds RFC-0042 §4.2's rule that fills [`KvFormula::dtype`] per family and source,
//! and the instance's `kv_dtype` override. [`KvFormula::kv_settings`] gives the formula's part of
//! RFC-0022 §7.1's `KvSettings`, which prefix keys hash.

use std::ops::Range;

use serde::{Deserialize, Serialize};
use syntropy_core::plan::KvDType;
use syntropy_core::policy::{ApproxKnobs, ApproxParams};
use syntropy_core::{Bytes, DType, KvSettings, LayerIx, RopeSettings};

/// KV and recurrent-state size per layer (RFC-0001 §5.3, RFC-0022 §2). Owner: RFC-0022.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvFormula {
    /// One entry per layer, indexed by [`LayerIx`].
    pub layers: Vec<LayerKv>,
    /// Per-token side caches, such as a sparse indexer's keys.
    #[serde(default)]
    pub aux: Vec<AuxKv>,
    /// The reference's KV dtype per layer (exact mode).
    #[serde(default)]
    pub dtype: KvDTypeSpec,
    /// A bump invalidates prefixes and saved sessions.
    #[serde(default)]
    pub layout_version: u16,
}

/// One layer's KV or state (RFC-0001 §5.3, RFC-0022 §2). "Elements" are in the layer's KV dtype.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKv {
    /// `2 · n_kv_heads · d_head` elements per token (MHA, GQA), or `d_c + d_rope` (MLA).
    /// `window` caps stored tokens for sliding-window layers. Cost model §3.6.
    PerToken {
        /// Elements per token.
        elems_per_token: u32,
        /// The most recent tokens kept, for a sliding-window layer.
        window: Option<u32>,
        /// Whether the entry is one latent shared by K and V (MLA) rather than a K/V pair.
        /// `K8V4` sizes a latent as `Q8` (OQ-IO-30). An addition with a serde default, as
        /// `Compressed` is (RFC-0022 §2, note of 2026-10-28; RFC-0001 §18.46). `Sparse` and
        /// `Compressed` carry the same flag.
        #[serde(default)]
        latent: bool,
    },
    /// Fixed state per sequence: linear attention and state-space layers.
    Fixed {
        /// Elements per sequence.
        elems_per_seq: u64,
    },
    /// Full per-token cache stored, at most `selected` positions read per step.
    Sparse {
        /// Elements per token.
        elems_per_token: u32,
        /// Positions read per step.
        selected: u32,
        /// Whether the entry is one latent (a DSA-class indexer over MLA) rather than a K/V pair
        /// (block-sparse and QSA-class layers, stored as GQA). Set by the family; as for
        /// `PerToken`.
        #[serde(default)]
        latent: bool,
    },
    /// One compressed entry per `tokens_per_entry` tokens plus a short raw `window`; all
    /// entries, or `selected` of them, read per step (DeepSeek V4-class; M3).
    Compressed {
        /// Elements per entry, compressed or raw.
        elems_per_entry: u32,
        /// Tokens pooled into one compressed entry, at least 1.
        tokens_per_entry: u16,
        /// The most recent tokens kept raw.
        window: Option<u32>,
        /// Compressed entries read per step; `None` reads them all.
        selected: Option<u32>,
        /// Whether each entry, compressed or raw, is one latent rather than a K/V pair. Set by
        /// the family; as for `PerToken`.
        #[serde(default)]
        latent: bool,
    },
}

impl LayerKv {
    /// An MHA or GQA layer: K and V, `2 · n_kv_heads · d_head` elements per token, every token
    /// kept unless `window` caps them.
    #[must_use]
    pub const fn gqa(n_kv_heads: u32, d_head: u32, window: Option<u32>) -> LayerKv {
        LayerKv::PerToken {
            elems_per_token: 2 * n_kv_heads * d_head,
            window,
            latent: false,
        }
    }

    /// An MLA layer: the latent `d_c` and the decoupled RoPE key `d_rope`, one of each per token,
    /// shared by every head.
    #[must_use]
    pub const fn mla(d_c: u32, d_rope: u32) -> LayerKv {
        LayerKv::PerToken {
            elems_per_token: d_c + d_rope,
            window: None,
            latent: true,
        }
    }
}

/// A per-token side cache of one layer (RFC-0022 §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuxKv {
    /// The layer it belongs to.
    pub layer: LayerIx,
    /// What it is.
    pub kind: AuxKind,
    /// Elements per token.
    pub elems_per_token: u32,
    /// Its dtype, which the KV dtype knob does not change.
    pub dtype: DType,
}

/// What an [`AuxKv`] holds (RFC-0022 §2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuxKind {
    /// A sparse indexer's per-token keys, read over the whole context every step.
    SparseIndexer,
}

/// The reference's KV dtype per layer (RFC-0022 §2): one dtype, with per-layer exceptions.
/// RFC-0042 §4.2's rule fills it per family and source ([`KvDTypeRule::reference`]), and an
/// instance's `kv_dtype` override changes it ([`KvDTypeRule::resolve`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvDTypeSpec {
    /// The dtype of every layer not in `layers`.
    pub default: DType,
    /// Layers whose dtype differs from `default`.
    #[serde(default)]
    pub layers: Vec<(LayerIx, DType)>,
}

impl Default for KvDTypeSpec {
    /// BF16 everywhere: the serde default for a record without the field.
    fn default() -> Self {
        Self {
            default: DType::BF16,
            layers: Vec::new(),
        }
    }
}

impl KvDTypeSpec {
    /// One dtype for every layer.
    #[must_use]
    pub const fn uniform(dtype: DType) -> Self {
        Self {
            default: dtype,
            layers: Vec::new(),
        }
    }

    /// Layer `layer`'s dtype.
    #[must_use]
    pub fn of(&self, layer: LayerIx) -> DType {
        self.layers
            .iter()
            .find(|(l, _)| *l == layer)
            .map_or(self.default, |(_, dtype)| *dtype)
    }
}

/// A formula that does not describe a model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvFormulaError {
    /// A `Compressed` layer pools 0 tokens per entry. A layer with only a raw window is
    /// `PerToken` with a window (OQ-MF-16 (a)).
    NoTokensPerEntry {
        /// The layer.
        layer: LayerIx,
    },
    /// An auxiliary cache or a dtype exception names a layer the formula does not have.
    NoSuchLayer {
        /// The layer named.
        layer: LayerIx,
    },
}

impl std::fmt::Display for KvFormulaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoTokensPerEntry { layer } => {
                write!(
                    f,
                    "layer {}: a compressed layer pools 0 tokens per entry",
                    layer.0
                )
            }
            Self::NoSuchLayer { layer } => write!(f, "layer {} is not in the formula", layer.0),
        }
    }
}

impl std::error::Error for KvFormulaError {}

impl KvFormula {
    /// RFC-0022 §7.1's `KvSettings` for an instance of this model: the formula gives the
    /// dtype per layer (in normal form), the layout version, and each layer's window and
    /// compression ratio; the caller gives `P`, the stored `KvDType` of each page class, the
    /// resolved RoPE configurations and the instance's knobs and their parameters, which the
    /// formula does not hold. A sequence's own knobs join them through `KvSettings::with_sequence`.
    /// [`KvSettings::hash_with`] then gives `kv_settings_hash`.
    #[must_use]
    pub fn kv_settings(
        &self,
        page_tokens: u32,
        stored: Vec<KvDType>,
        rope: Vec<RopeSettings>,
        knobs: ApproxKnobs,
        params: ApproxParams,
    ) -> KvSettings {
        let dtype = self.dtype.normalized();
        let (windows, compression) = self
            .layers
            .iter()
            .map(|layer| match *layer {
                LayerKv::PerToken { window, .. } => (window, None),
                LayerKv::Compressed {
                    window,
                    tokens_per_entry,
                    ..
                } => (window, Some(tokens_per_entry)),
                LayerKv::Fixed { .. } | LayerKv::Sparse { .. } => (None, None),
            })
            .unzip();
        KvSettings {
            page_tokens,
            stored,
            dtype: dtype.default,
            dtype_layers: dtype.layers,
            layout_version: self.layout_version,
            rope,
            windows,
            compression,
            knobs,
            params,
        }
    }

    /// A formula with these layers, no auxiliary cache, `dtype` everywhere and layout version 0.
    #[must_use]
    pub fn new(layers: Vec<LayerKv>, dtype: DType) -> Self {
        Self {
            layers,
            aux: Vec::new(),
            dtype: KvDTypeSpec::uniform(dtype),
            layout_version: 0,
        }
    }

    /// Checks that the formula describes a model.
    ///
    /// # Errors
    ///
    /// [`KvFormulaError`] for a compressed layer with no tokens per entry, or a layer reference
    /// past the last layer.
    pub fn check(&self) -> Result<(), KvFormulaError> {
        for (i, layer) in self.layers.iter().enumerate() {
            if let LayerKv::Compressed {
                tokens_per_entry: 0,
                ..
            } = layer
            {
                return Err(KvFormulaError::NoTokensPerEntry {
                    layer: LayerIx(i as u16),
                });
            }
        }
        let named = self
            .aux
            .iter()
            .map(|aux| aux.layer)
            .chain(self.dtype.layers.iter().map(|(layer, _)| *layer));
        for layer in named {
            if usize::from(layer.0) >= self.layers.len() {
                return Err(KvFormulaError::NoSuchLayer { layer });
            }
        }
        Ok(())
    }

    /// The bytes one sequence of `tokens` tokens keeps on `layers`, stored as `dtype`. A layer
    /// past the last counts nothing. A `Fixed` layer counts its state once, whatever `tokens`
    /// is, because a sequence holds it from the moment it opens.
    #[must_use]
    pub fn stored(&self, layers: Range<LayerIx>, tokens: u64, dtype: KvDType) -> Bytes {
        let layers = self.indices(&layers);
        let main = self.sum(layers.clone(), dtype, |kv| match kv {
            LayerKv::PerToken {
                elems_per_token,
                window,
                latent,
            } => Count::Entries(capped(tokens, window), elems_per_token, !latent),
            LayerKv::Fixed { elems_per_seq } => Count::State(elems_per_seq),
            LayerKv::Sparse {
                elems_per_token,
                latent,
                ..
            } => Count::Entries(tokens, elems_per_token, !latent),
            LayerKv::Compressed {
                elems_per_entry,
                tokens_per_entry,
                window,
                latent,
                ..
            } => Count::Entries(
                compressed(tokens, tokens_per_entry) + capped(tokens, window),
                elems_per_entry,
                !latent,
            ),
        });
        saturate(main + self.aux_bytes(layers, tokens))
    }

    /// The bytes one decode step at `context` tokens reads on `layers`, stored as `dtype`:
    /// RFC-0022 §5.4's `B_vram + B_host`, before the hot window splits them.
    ///
    /// - `PerToken`: every kept token, at most `window`.
    /// - `Fixed`: the state, once.
    /// - `Sparse`: at most `selected` positions of the main cache, plus the indexer's keys for
    ///   the whole context, which the layer's [`AuxKv`] carries.
    /// - `Compressed`: every formed entry, or at most `selected` of them, plus the raw window.
    #[must_use]
    pub fn read_per_step(&self, layers: Range<LayerIx>, context: u64, dtype: KvDType) -> Bytes {
        let layers = self.indices(&layers);
        let main = self.sum(layers.clone(), dtype, |kv| match kv {
            LayerKv::PerToken {
                elems_per_token,
                window,
                latent,
            } => Count::Entries(capped(context, window), elems_per_token, !latent),
            LayerKv::Fixed { elems_per_seq } => Count::State(elems_per_seq),
            LayerKv::Sparse {
                elems_per_token,
                selected,
                latent,
            } => Count::Entries(context.min(u64::from(selected)), elems_per_token, !latent),
            LayerKv::Compressed {
                elems_per_entry,
                tokens_per_entry,
                window,
                selected,
                latent,
            } => {
                let formed = compressed(context, tokens_per_entry);
                let read = selected.map_or(formed, |s| formed.min(u64::from(s)));
                Count::Entries(read + capped(context, window), elems_per_entry, !latent)
            }
        });
        saturate(main + self.aux_bytes(layers, context))
    }

    /// `layers` as indices into [`KvFormula::layers`], clamped to its length.
    fn indices(&self, layers: &Range<LayerIx>) -> Range<usize> {
        let end = usize::from(layers.end.0).min(self.layers.len());
        usize::from(layers.start.0).min(end)..end
    }

    /// The sum over `layers` of each layer's count, in its storage.
    fn sum(&self, layers: Range<usize>, dtype: KvDType, count: impl Fn(LayerKv) -> Count) -> u128 {
        let mut total: u128 = 0;
        for i in layers {
            let layer = LayerIx(i as u16);
            let native = self.dtype.of(layer);
            total += match count(self.layers[i]) {
                Count::Entries(n, elems, pair) => {
                    u128::from(n) * entry_bytes(elems, pair, native, dtype)
                }
                Count::State(elems) => bits_to_bytes(u128::from(elems) * u128::from(native.bits())),
            };
        }
        total
    }

    /// The auxiliary caches on these layers for `tokens` tokens, each in its own dtype.
    fn aux_bytes(&self, layers: Range<usize>, tokens: u64) -> u128 {
        self.aux
            .iter()
            .filter(|aux| layers.contains(&usize::from(aux.layer.0)))
            .map(|aux| {
                u128::from(tokens)
                    * bits_to_bytes(u128::from(aux.elems_per_token) * u128::from(aux.dtype.bits()))
            })
            .sum()
    }
}

/// What one layer counts: `n` entries of `elems` elements, which are K/V pairs when the flag is
/// set, or a state of `elems` elements.
enum Count {
    Entries(u64, u32, bool),
    State(u64),
}

/// `tokens`, at most `window`.
fn capped(tokens: u64, window: Option<u32>) -> u64 {
    window.map_or(tokens, |w| tokens.min(u64::from(w)))
}

/// The compressed entries formed after `tokens` tokens: entry j forms when token
/// `r·(j + 1) − 1` is processed (OQ-MF-12), so `⌊tokens / r⌋`. `r = 0` forms none.
fn compressed(tokens: u64, tokens_per_entry: u16) -> u64 {
    tokens.checked_div(u64::from(tokens_per_entry)).unwrap_or(0)
}

/// The bytes of one entry of `elems` elements whose reference dtype is `native`, stored as
/// `dtype` (RFC-0022 §9).
/// - `Native`: `elems` elements of `native`.
/// - `Q8 { block }`: one byte per element and an F16 scale per `block` elements.
/// - `K8V4 { block }`: for a K/V pair (`pair`), the first half (K) as `Q8` and the second half
///   (V) at four bits per element with an F16 scale per `block` elements. A latent (a layer's
///   `latent` flag) is sized as `Q8`. RFC-0022 §9 defines `K8V4` by K and V planes only, and a
///   latent is not a K/V pair. `Q8` is the conservative upper bound, and the default until
///   OQ-IO-30 is settled.
///
/// A `block` of 0 is taken as 1.
fn entry_bytes(elems: u32, pair: bool, native: DType, dtype: KvDType) -> u128 {
    let elems = u128::from(elems);
    let scales = |n: u128, block: u16| n.div_ceil(u128::from(block.max(1))) * 2;
    match dtype {
        KvDType::Q8 { block } => elems + scales(elems, block),
        KvDType::K8V4 { block, .. } if !pair => elems + scales(elems, block),
        KvDType::K8V4 { block, .. } => {
            let k = elems.div_ceil(2);
            let v = elems - k;
            k + scales(k, block) + v.div_ceil(2) + scales(v, block)
        }
        // `Native`. `KvDType` is `core`'s and non-exhaustive, so this arm also catches a storage
        // form added later. That form is then sized as native, an over-estimate, until it gains
        // its own rule above, and the assertion makes tests fail loudly meanwhile.
        _ => {
            debug_assert!(
                matches!(dtype, KvDType::Native),
                "KvDType {dtype:?} has no sizing rule"
            );
            bits_to_bytes(elems * u128::from(native.bits()))
        }
    }
}

fn bits_to_bytes(bits: u128) -> u128 {
    bits.div_ceil(8)
}

fn saturate(bytes: u128) -> Bytes {
    Bytes(u64::try_from(bytes).unwrap_or(u64::MAX))
}

pub mod dtype;

pub use dtype::{KvDTypeError, KvDTypeOverride, KvDTypeRule, KvDTypeSection, KvSource};

#[cfg(test)]
mod prefix_tests;
#[cfg(test)]
mod tests;
