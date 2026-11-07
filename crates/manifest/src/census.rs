// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tensor census and the model manifest (RFC-0001 §5.3; RFC-0041 §3, §3.1).
//!
//! - [`TensorClass`] is what a family's classification rules produce (RFC-0043 §4).
//! - [`TensorRecord`] is one tensor, or one routed expert's slice of a stacked tensor, with its
//!   extent in a source file. [`TensorGroup`] is the solver's placement unit, built by RFC-0041
//!   §3.1's rules. [`ClassTotals`] sums the groups by class.
//! - [`ModelManifest`] is one model variant: its `ModelId`, family, hyperparameters, census, KV
//!   formula, tokenizer and the tensors it ignores. It is stored as postcard compressed with zstd
//!   ([`ModelManifest::encode`], [`ModelManifest::decode`]; RFC-0041 §3.2).
//!
//! The builder is `syntropy_graph::census::build`: it resolves the family, which lives in
//! `graph`, a layer above this crate (RFC-0041 §3, note of 2026-10-31). RFC-0001 §5.3's
//! `name`, `source`, `speculative`, `siblings` and `reference` join [`ModelManifest`] with the
//! tasks that fill them.

use std::fmt;

use serde::{Deserialize, Serialize};
use syntropy_core::{Blake3, Bytes, ExpertIx, LayerIx, ModelId, QuantScheme, TensorGroupId};

use crate::hparams::HyperParams;
use crate::kv::KvFormula;
use crate::slice::SourceExtent;
use crate::tokenizer::TokenizerSpec;

/// A multimodal encoder tower's index within its model (RFC-0043 §2.4). Reserved: no v1 family
/// places a tower.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TowerIx(pub u8);

/// A tensor's access pattern, which decides its home tier, caching and prefetch policy
/// (RFC-0001 §5.3). Owner: RFC-0041.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// A class without its layer and expert: the key [`ClassTotals`] sums by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ClassKind {
    /// [`TensorClass::DenseEveryToken`].
    DenseEveryToken,
    /// [`TensorClass::Expert`].
    Expert,
    /// [`TensorClass::RowLookup`].
    RowLookup,
    /// [`TensorClass::Head`].
    Head,
    /// [`TensorClass::Router`].
    Router,
    /// [`TensorClass::Encoder`].
    Encoder,
}

impl TensorClass {
    /// The class without its layer and expert.
    #[must_use]
    pub fn kind(&self) -> ClassKind {
        match self {
            Self::DenseEveryToken => ClassKind::DenseEveryToken,
            Self::Expert { .. } => ClassKind::Expert,
            Self::RowLookup => ClassKind::RowLookup,
            Self::Head => ClassKind::Head,
            Self::Router { .. } => ClassKind::Router,
            Self::Encoder { .. } => ClassKind::Encoder,
        }
    }
}

/// Index into [`TensorCensus::tensors`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TensorId(pub u32);

/// One tensor, or one routed expert's slice of a stacked tensor (RFC-0001 §5.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRecord {
    /// Its index in [`TensorCensus::tensors`].
    pub id: TensorId,
    /// The name as the source spells it; a slice is `<tensor>[e]` (RFC-0041 §2.1).
    pub name: String,
    /// Its class.
    pub class: TensorClass,
    /// Its layer, if it has one.
    pub layer: Option<LayerIx>,
    /// Its shape in the source's order: ggml's `ne`, innermost first, for GGUF; outermost
    /// first for safetensors.
    pub shape: Vec<u64>,
    /// How its elements are stored.
    pub quant: QuantScheme,
    /// Its bytes in the source: `source.len`.
    pub bytes: Bytes,
    /// Where its bytes are.
    pub source: SourceExtent,
    /// Its scale and zero-point extents (RFC-0041 §2.2); empty for the M0 schemes, whose scales
    /// are inline (GGUF) or absent (BF16).
    pub aux: Vec<SourceExtent>,
    /// The Merkle root over its chunk IDs, once its chunks are committed (RFC-0041 §1); `None`
    /// from the census, which reads headers only.
    pub hash: Option<Blake3>,
}

/// The solver's placement unit (RFC-0001 §5.3; RFC-0041 §3.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorGroup {
    /// Its index in [`TensorCensus::groups`].
    pub id: TensorGroupId,
    /// Its class.
    pub class: TensorClass,
    /// Its layer, if it has one.
    pub layer: Option<LayerIx>,
    /// Its tensors.
    pub tensors: Vec<TensorId>,
    /// The bytes of its tensors and their auxiliary extents.
    pub bytes: Bytes,
    /// Bytes read per token at batch 1 (RFC-0041 §3.1): all of them for dense, head and router
    /// groups; one row for a lookup table; `bytes × top_k / n_experts` for an expert.
    pub read_per_token: Bytes,
    /// The group that holds the same tensor for the other end of the model: a tied embedding's
    /// `RowLookup` and `Head` groups name each other (RFC-0041 §3.1).
    pub shared_with: Option<TensorGroupId>,
}

/// Bytes stored and read per token for one class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassTotal {
    /// Bytes of the class's groups.
    pub bytes: Bytes,
    /// Their bytes read per token at batch 1.
    pub read_per_token: Bytes,
}

/// [`ClassTotal`] per class, summed over groups, so a tied embedding counts once as
/// `RowLookup` and once as `Head`, as the two nodes that hold it do.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassTotals {
    /// One entry per class that has a group, in [`ClassKind`] order.
    pub by_class: Vec<(ClassKind, ClassTotal)>,
}

impl ClassTotals {
    /// The total of `kind`, zero if it has no group.
    #[must_use]
    pub fn get(&self, kind: ClassKind) -> ClassTotal {
        self.by_class
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, t)| *t)
            .unwrap_or_default()
    }

    /// The totals of `groups`.
    #[must_use]
    pub fn of(groups: &[TensorGroup]) -> Self {
        let mut by_class: Vec<(ClassKind, ClassTotal)> = Vec::new();
        for group in groups {
            let kind = group.class.kind();
            let at = match by_class.iter().position(|(k, _)| *k == kind) {
                Some(at) => at,
                None => {
                    by_class.push((kind, ClassTotal::default()));
                    by_class.len() - 1
                }
            };
            let total = &mut by_class[at].1;
            total.bytes.0 += group.bytes.0;
            total.read_per_token.0 += group.read_per_token.0;
        }
        by_class.sort_by_key(|(k, _)| *k);
        Self { by_class }
    }
}

/// Every tensor, classified by access pattern (RFC-0001 §5.3).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorCensus {
    /// The records, in source order, a stacked tensor's slices in expert order.
    pub tensors: Vec<TensorRecord>,
    /// The groups.
    pub groups: Vec<TensorGroup>,
    /// The groups' bytes, per class.
    pub totals: ClassTotals,
    /// The token embedding: the prologue's one `RowLookup` record that maps token IDs to rows
    /// (RFC-0043 §5.2). An n-gram or per-layer table is never it. `None` only in a census built
    /// by hand without one, from which [`ModelManifest::vocab_limit`] refuses to derive a
    /// limit.
    pub token_embedding: Option<TensorId>,
    /// The LM head record, or `None` when the head is tied to the embedding: there is no head
    /// record, and the embedding serves as the head (RFC-0041 §3.1).
    pub lm_head: Option<TensorId>,
}

impl TensorCensus {
    /// The bytes of every record, each counted once: what RFC-0060 §7.2 check 1 compares with
    /// the source's tensor bytes, less the ignored tensors.
    #[must_use]
    pub fn record_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(|t| t.bytes.0 + t.aux.iter().map(|a| a.len).sum::<u64>())
            .sum()
    }
}

/// The container the weights come from (RFC-0001 §5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceFormat {
    /// GGUF, with its version.
    Gguf {
        /// The GGUF version.
        version: u32,
    },
    /// safetensors.
    Safetensors,
}

/// One model variant, as the census describes it (RFC-0001 §5.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Its `ModelId` (RFC-0041 §1).
    pub id: ModelId,
    /// Its family's frozen tag (RFC-0043 §5.1).
    pub family: String,
    /// The container.
    pub format: SourceFormat,
    /// Its hyperparameters.
    pub hparams: HyperParams,
    /// Its tensors and groups.
    pub census: TensorCensus,
    /// KV and state per layer.
    pub kv: KvFormula,
    /// Its tokenizer, when the census was given tokenizer files: a safetensors checkpoint's side
    /// files. A GGUF source's comes with the conversion from its metadata (RFC-0043 §8.3).
    pub tokenizer: Option<TokenizerSpec>,
    /// The tensors the family classifies `Ignored`, with the reason (RFC-0043 §4.3). Never
    /// placed or fetched.
    pub ignored: Vec<(String, String)>,
}

/// The format tag that starts a stored manifest, so a later layout is told apart. `SYNMAN02`
/// added [`TensorCensus::token_embedding`] and [`TensorCensus::lm_head`] (COMP-E10-T16).
const MANIFEST_TAG: &[u8; 8] = b"SYNMAN02";

/// The zstd level a manifest is stored at: 3, zstd's default \[target\].
const ZSTD_LEVEL: i32 = 3;

/// The largest manifest decoded, 256 MiB, ten times the largest RFC-0041 §3.2 sizes
/// \[target\], so a corrupt or hostile blob cannot make the decoder allocate without bound.
pub const MAX_MANIFEST_BYTES: usize = 256 << 20;

/// Why a stored manifest cannot be read back: `HeaderInvalid` (RFC-0041 §10 code 6), class
/// `Integrity`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestCodecError {
    /// What is wrong.
    pub reason: String,
}

impl ManifestCodecError {
    /// RFC-0041 §10's code, `HeaderInvalid`.
    #[must_use]
    pub fn code(&self) -> u16 {
        6
    }
}

impl fmt::Display for ManifestCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not a stored model manifest: {}", self.reason)
    }
}

impl std::error::Error for ManifestCodecError {}

/// Why a manifest gives no `vocab_limit`: `HeaderInvalid` (RFC-0041 §10 code 6), class
/// `Integrity`. The limit guards device tables, so a census that cannot name its token
/// embedding or LM head fails closed rather than guess.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabLimitError {
    /// What is wrong.
    pub reason: String,
}

impl VocabLimitError {
    /// RFC-0041 §10's code, `HeaderInvalid`.
    #[must_use]
    pub fn code(&self) -> u16 {
        6
    }

    /// The error class.
    #[must_use]
    pub fn class(&self) -> syntropy_core::ErrorClass {
        syntropy_core::ErrorClass::Integrity
    }
}

impl fmt::Display for VocabLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no vocab_limit: {}", self.reason)
    }
}

impl std::error::Error for VocabLimitError {}

impl ModelManifest {
    /// `vocab_limit = min(token-embedding rows, LM-head rows, hparams.vocab)`, the bound every
    /// token ID a request carries must stay below (RFC-0050 §2.3, A9). The sampler's count
    /// tables have `hparams.vocab` entries (RFC-0012 §8.1). A table's rows are its outermost
    /// dimension: the first of a safetensors shape, the last of GGUF's `ne`. A tied model has
    /// no head record, and its head's rows are the embedding's.
    ///
    /// # Errors
    ///
    /// [`VocabLimitError`] when the census marks no token embedding, or a marked record is
    /// missing, of the wrong class, or not a two-dimensional table.
    pub fn vocab_limit(&self) -> Result<u32, VocabLimitError> {
        let fail = |reason: String| Err(VocabLimitError { reason });
        let Some(embedding) = self.census.token_embedding else {
            return fail("the census marks no token embedding".into());
        };
        let embed_rows = self.table_rows(embedding, ClassKind::RowLookup, "token embedding")?;
        let head_rows = match self.census.lm_head {
            Some(head) => self.table_rows(head, ClassKind::Head, "LM head")?,
            None => embed_rows,
        };
        let limit = embed_rows.min(head_rows).min(u64::from(self.hparams.vocab));
        if limit == 0 {
            return fail("the vocabulary is empty".into());
        }
        // At most `hparams.vocab`, a `u32`.
        Ok(u32::try_from(limit).unwrap_or(self.hparams.vocab))
    }

    /// The rows of record `id`, which must be a two-dimensional table of class `kind`.
    fn table_rows(
        &self,
        id: TensorId,
        kind: ClassKind,
        what: &str,
    ) -> Result<u64, VocabLimitError> {
        let fail = |reason: String| VocabLimitError { reason };
        let record = self
            .census
            .tensors
            .get(id.0 as usize)
            .filter(|r| r.id == id)
            .ok_or_else(|| {
                fail(format!(
                    "the {what} names record {}, which does not exist",
                    id.0
                ))
            })?;
        if record.class.kind() != kind {
            return Err(fail(format!(
                "the {what} `{}` is of class {:?}, not {kind:?}",
                record.name,
                record.class.kind()
            )));
        }
        let [a, b] = record.shape[..] else {
            return Err(fail(format!(
                "the {what} `{}` has shape {:?}, not two dimensions",
                record.name, record.shape
            )));
        };
        Ok(match self.format {
            SourceFormat::Gguf { .. } => b,
            SourceFormat::Safetensors => a,
        })
    }

    /// The stored form: an 8-byte format tag, then the manifest as postcard, compressed with
    /// zstd (RFC-0041 §3.2).
    ///
    /// # Errors
    ///
    /// [`ManifestCodecError`] if the encoders fail, which a manifest built by the census does
    /// not make them do.
    pub fn encode(&self) -> Result<Vec<u8>, ManifestCodecError> {
        let plain = postcard::to_stdvec(self).map_err(|error| ManifestCodecError {
            reason: format!("postcard: {error}"),
        })?;
        let packed =
            zstd::bulk::compress(&plain, ZSTD_LEVEL).map_err(|error| ManifestCodecError {
                reason: format!("zstd: {error}"),
            })?;
        let mut out = Vec::with_capacity(MANIFEST_TAG.len() + packed.len());
        out.extend_from_slice(MANIFEST_TAG);
        out.extend_from_slice(&packed);
        Ok(out)
    }

    /// Reads [`encode`](Self::encode)'s form back. The tokenizer's ID is computed again from
    /// what it covers.
    ///
    /// # Errors
    ///
    /// [`ManifestCodecError`] for a missing tag, a stream that does not decompress within
    /// [`MAX_MANIFEST_BYTES`], or bytes that are not a manifest.
    pub fn decode(bytes: &[u8]) -> Result<Self, ManifestCodecError> {
        let packed =
            bytes
                .strip_prefix(MANIFEST_TAG.as_slice())
                .ok_or_else(|| ManifestCodecError {
                    reason: "it does not start with the manifest tag".into(),
                })?;
        // A streaming read with a cap, so a decode allocates for what the stream holds, never
        // the cap up front.
        let mut plain = Vec::new();
        let decoder =
            zstd::stream::read::Decoder::new(packed).map_err(|error| ManifestCodecError {
                reason: format!("zstd: {error}"),
            })?;
        let limit = u64::try_from(MAX_MANIFEST_BYTES).unwrap_or(u64::MAX) + 1;
        std::io::Read::read_to_end(&mut std::io::Read::take(decoder, limit), &mut plain).map_err(
            |error| ManifestCodecError {
                reason: format!("zstd: {error}"),
            },
        )?;
        if plain.len() > MAX_MANIFEST_BYTES {
            return Err(ManifestCodecError {
                reason: format!("it decompresses past {MAX_MANIFEST_BYTES} bytes"),
            });
        }
        postcard::from_bytes(&plain).map_err(|error| ManifestCodecError {
            reason: format!("postcard: {error}"),
        })
    }
}

#[cfg(test)]
mod tests;
