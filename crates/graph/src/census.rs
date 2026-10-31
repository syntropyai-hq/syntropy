// SPDX-License-Identifier: MIT OR Apache-2.0

//! The census builder (RFC-0041 §3, §3.1; COMP-E10-T07): a checkpoint's headers in, its
//! [`ModelManifest`] out.
//!
//! RFC-0041 §3 names it `manifest::census::build`, but it resolves the family through
//! [`families()`], and `graph` is a layer above `manifest` (RFC-0001 §4.1). So the types live in
//! `syntropy_manifest::census` and the builder here (RFC-0041 §3, note of 2026-10-31). It reads
//! headers only, never weight bytes. The steps, as §3 numbers them:
//!
//! 1. **Raw tensors** from the GGUF header or the safetensors headers: name, shape in the
//!    source's order, quantization scheme and extent.
//! 2. **The family**, failing closed. No family accepts the architecture names: `UnknownFamily`
//!    (1). The names resolve to more than one family, such as `architectures[0]` to one and
//!    `model_type` to another: `AmbiguousFamily`, also code 1 (ADR-0003 decision 6).
//! 3. **Hyperparameters** through the family (`HyperParamInvalid`, `Graph` code 3).
//! 4. **Classification** of every tensor. Any tensor no rule classifies, or whose shape fails its
//!    rule, refuses the load as `UnknownTensor` (3), naming every such tensor. `Ignored` tensors
//!    go to `ModelManifest::ignored` with their reason.
//! 5. **Slicing**: a stacked expert tensor becomes one record per expert (RFC-0041 §2.1).
//! 6. Auxiliary extents. The M0 schemes have none (GGUF scales are inline, BF16 has none), so
//!    every record's `aux` is empty; quantised safetensors schemes bind theirs with their task.
//! 7. **Groups** by §3.1's rules, below, with `bytes` and `read_per_token`.
//! 8. **KV formula and tokenizer** through the family.
//! 10. **`ModelId`** and the totals. Steps 9 (speculative support) and 11 (the summary), and the
//!     siblings, come with their tasks.
//!
//! **Groups** (RFC-0041 §3.1):
//! - one per (layer, block) for the dense tensors of a decoder or MTP layer, so a norm joins
//!   the block the family's rule names for it;
//! - in a MoE block, the router and the norms form the router group, and any other dense tensor
//!   (a shared expert) a dense group of its own;
//! - one per (layer, expert), holding every tensor of that expert;
//! - one per lookup table;
//! - one head group per layer that has a head, the epilogue's dense tensors (the final norm)
//!   joining the last one. With no head tensor the embedding is tied: the head group holds the
//!   embedding's record too, and the two groups name each other in `shared_with`.
//!
//! **`read_per_token`** at batch 1: all bytes for dense, head and router groups; one row,
//! `bytes / rows` with `rows` the table's outermost dimension, for a lookup table; and
//! `bytes × top_k / n_experts` for an expert group (the `top_k` and `n_experts` extra
//! hyperparameters), until RFC-0023's statistics replace the prior.
//!
//! **Checks before the manifest is returned** (`HeaderInvalid`, 6): every decoder layer has a
//! classified tensor, the model has a token embedding, every expert of a MoE layer has every
//! tensor its siblings have, and the records' bytes plus the ignored tensors' bytes equal the
//! source's tensor bytes (RFC-0060 §7.2 check 1).

use std::collections::BTreeMap;
use std::fmt;

use syntropy_core::{Bytes, DType, ErrorClass, ExpertIx, LayerIx, QuantScheme, TensorGroupId};
use syntropy_manifest::census::{
    ClassTotals, ModelManifest, SourceFormat, TensorCensus, TensorClass, TensorGroup, TensorId,
    TensorRecord,
};
use syntropy_manifest::gguf::GgufHeader;
use syntropy_manifest::hparams::{HyperParams, N_EXPERTS, TOP_K};
use syntropy_manifest::identity::{IdentityError, SourceFile, model_id};
use syntropy_manifest::kv::KvSource;
use syntropy_manifest::safetensors::SafetensorsHeader;
use syntropy_manifest::slice::{SliceError, SourceExtent, gguf_experts, safetensors_experts};
use syntropy_manifest::tokenizer::{TokenizerSources, TokenizerSpecError};

use crate::classify::{BlockSel, Classification};
use crate::families::{Family, FamilyError, MetaView, Registry, families};
use crate::ir::{Block, TensorRole};

/// Where the weights' headers come from.
#[derive(Clone, Copy, Debug)]
pub enum Weights<'a> {
    /// A GGUF file, or one file of a split set; its tensors carry their file index.
    Gguf(&'a GgufHeader),
    /// A safetensors checkpoint's headers, one per shard, each shard's file index its position.
    Safetensors(&'a [SafetensorsHeader]),
}

/// What the census reads: headers and metadata, never weight bytes.
#[derive(Clone, Copy, Debug)]
pub struct Sources<'a> {
    /// The metadata the family is resolved by and reads hyperparameters from.
    pub meta: MetaView<'a>,
    /// The tensors.
    pub weights: Weights<'a>,
    /// What RFC-0042 §4.2's KV dtype rule needs to know of the source.
    pub kv_source: KvSource,
    /// The tokenizer side files, for a safetensors checkpoint.
    pub tokenizer: Option<TokenizerSources<'a>>,
    /// Every source file the census reads, for `ModelId` (RFC-0041 §1).
    pub files: &'a [SourceFile],
}

/// Why the census refuses a model. Codes are `(Domain::Formats, n)` from RFC-0041 §10, except
/// `HyperParamInvalid`, which is `(Domain::Graph, 3)` (RFC-0043 §10).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CensusError {
    /// `UnknownFamily` (1): no family accepts the architecture names.
    UnknownFamily {
        /// The names the metadata gives.
        names: Vec<String>,
        /// The tags this build registers.
        supported: Vec<&'static str>,
    },
    /// `UnknownFamily` (1): the names resolve to more than one family, so which one the
    /// checkpoint is cannot be told.
    AmbiguousFamily {
        /// The names the metadata gives.
        names: Vec<String>,
        /// The families they resolve to.
        families: Vec<&'static str>,
    },
    /// `HyperParamInvalid`, `(Graph, 3)`.
    HyperParams(FamilyError),
    /// `UnknownTensor` (3): tensors the family cannot classify, a shape that fails its rule
    /// among them.
    UnknownTensor {
        /// Every such tensor, in source order.
        names: Vec<String>,
    },
    /// `UnknownQuant` (4): a tensor whose stored type has no scheme here.
    UnknownQuant {
        /// The tensor.
        name: String,
        /// Its type as the header names it.
        dtype: String,
    },
    /// `HeaderInvalid` (6): a stacked expert tensor that does not slice.
    Slice(SliceError),
    /// `HeaderInvalid` (6): the tensors do not form the model the hyperparameters describe.
    Incomplete {
        /// What is missing.
        reason: String,
    },
    /// The tokenizer side files cannot be read (class `Integrity`; no code yet).
    Tokenizer(TokenizerSpecError),
    /// `ModelId` cannot be formed: no weights file, or no family tag.
    Identity(IdentityError),
}

impl CensusError {
    /// RFC-0041 §10's code, or RFC-0043 §10's for `HyperParams`; `None` where none is assigned.
    #[must_use]
    pub fn code(&self) -> Option<u16> {
        match self {
            Self::UnknownFamily { .. } | Self::AmbiguousFamily { .. } => Some(1),
            Self::HyperParams(e) => Some(e.code()),
            Self::UnknownTensor { .. } => Some(3),
            Self::UnknownQuant { .. } => Some(4),
            Self::Slice(_) | Self::Incomplete { .. } => Some(6),
            Self::Tokenizer(_) | Self::Identity(_) => None,
        }
    }

    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::UnknownFamily { .. }
            | Self::AmbiguousFamily { .. }
            | Self::HyperParams(_)
            | Self::UnknownTensor { .. }
            | Self::UnknownQuant { .. } => ErrorClass::Unsupported,
            Self::Slice(_) | Self::Incomplete { .. } | Self::Tokenizer(_) => ErrorClass::Integrity,
            Self::Identity(_) => ErrorClass::InvalidRequest,
        }
    }
}

impl fmt::Display for CensusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFamily { names, supported } => write!(
                f,
                "UnknownFamily (Formats 1): no family accepts {names:?}; this build supports {supported:?}"
            ),
            Self::AmbiguousFamily { names, families } => write!(
                f,
                "UnknownFamily (Formats 1): {names:?} name more than one family, {families:?}"
            ),
            Self::HyperParams(e) => write!(f, "{e}"),
            Self::UnknownTensor { names } => write!(
                f,
                "UnknownTensor (Formats 3): the family cannot classify {names:?}"
            ),
            Self::UnknownQuant { name, dtype } => write!(
                f,
                "UnknownQuant (Formats 4): `{name}` is stored as {dtype}, which has no scheme"
            ),
            Self::Slice(e) => write!(f, "HeaderInvalid (Formats 6): {e}"),
            Self::Incomplete { reason } => write!(f, "HeaderInvalid (Formats 6): {reason}"),
            Self::Tokenizer(e) => write!(f, "the tokenizer cannot be read: {e}"),
            Self::Identity(e) => write!(f, "no model ID: {e}"),
        }
    }
}

impl std::error::Error for CensusError {}

/// Builds the manifest of `sources` with this build's families ([`families()`]).
///
/// # Errors
///
/// [`CensusError`]: the family, a tensor or the headers refuse the model (module docs).
pub fn build(sources: &Sources<'_>) -> Result<ModelManifest, CensusError> {
    build_in(families(), sources)
}

/// [`build`] with the families of `registry`.
///
/// # Errors
///
/// As [`build`].
pub fn build_in(registry: &Registry, sources: &Sources<'_>) -> Result<ModelManifest, CensusError> {
    let raw = raw_tensors(sources.weights)?;
    let family = resolve(registry, &sources.meta)?;
    let hp = family
        .parse_hparams(&sources.meta)
        .map_err(CensusError::HyperParams)?;

    // Step 4: classify every tensor, failing closed on any the family cannot.
    let mut classified = Vec::new();
    let mut ignored = Vec::new();
    let mut ignored_bytes = 0u64;
    let mut unknown = Vec::new();
    for tensor in &raw {
        match family.classify(&tensor.name, &tensor.shape, &hp) {
            Classification::Classified {
                class,
                layer,
                role,
                stacked_experts,
                block,
            } => classified.push((tensor, class, layer, role, stacked_experts, block)),
            Classification::Ignored { reason } => {
                ignored.push((tensor.name.clone(), reason.to_owned()));
                ignored_bytes += tensor.extent.len;
            }
            // `Classification` is non-exhaustive: an outcome added later refuses the tensor
            // until the census knows it.
            _ => unknown.push(tensor.name.clone()),
        }
    }
    if !unknown.is_empty() {
        return Err(CensusError::UnknownTensor { names: unknown });
    }

    // Step 5: records, slicing stacked expert tensors.
    let mut records: Vec<TensorRecord> = Vec::new();
    let mut placed: Vec<Placed> = Vec::new();
    for (tensor, class, layer, role, stacked, block) in classified {
        let mut push = |name: String, class: TensorClass, shape: Vec<u64>, extent: SourceExtent| {
            let id = TensorId(u32::try_from(records.len()).unwrap_or(u32::MAX));
            placed.push(Placed {
                id,
                class,
                layer,
                role,
                block,
                rows: outermost(&shape, sources.weights),
            });
            records.push(TensorRecord {
                id,
                name,
                class,
                layer,
                shape,
                quant: tensor.quant,
                bytes: Bytes(extent.len),
                source: extent,
                aux: Vec::new(),
                hash: None,
            });
        };
        match (stacked, class) {
            (Some(n), TensorClass::Expert { layer: l, .. }) => {
                for slice in tensor.slice(sources.weights, n)? {
                    let class = TensorClass::Expert {
                        layer: l,
                        id: slice.expert,
                    };
                    push(slice.name, class, slice.shape, slice.extent);
                }
            }
            _ => push(
                tensor.name.clone(),
                class,
                tensor.shape.clone(),
                tensor.extent,
            ),
        }
    }

    check_complete(&hp, &placed)?;
    let groups = group(&hp, &records, &placed)?;

    // RFC-0060 §7.2 check 1: every tensor byte is in a record or an ignored tensor, once.
    let source_bytes: u64 = raw.iter().map(|t| t.extent.len).sum();
    let census = TensorCensus {
        totals: ClassTotals::of(&groups),
        tensors: records,
        groups,
    };
    if census.record_bytes() + ignored_bytes != source_bytes {
        return Err(CensusError::Incomplete {
            reason: format!(
                "the census holds {} bytes and ignores {ignored_bytes}, and the source's tensors hold {source_bytes}",
                census.record_bytes()
            ),
        });
    }

    let kv = family.kv_formula(&hp, sources.kv_source);
    let tokenizer = sources
        .tokenizer
        .map(|files| family.tokenizer_spec(files))
        .transpose()
        .map_err(CensusError::Tokenizer)?;
    let id = model_id(family.name(), sources.files).map_err(CensusError::Identity)?;
    Ok(ModelManifest {
        id,
        family: family.name().to_owned(),
        format: match sources.weights {
            Weights::Gguf(h) => SourceFormat::Gguf { version: h.version },
            Weights::Safetensors(_) => SourceFormat::Safetensors,
        },
        hparams: hp,
        census,
        kv,
        tokenizer,
        ignored,
    })
}

/// Resolves the family, failing closed: none, or more than one, refuses the model.
fn resolve(registry: &Registry, meta: &MetaView<'_>) -> Result<&'static dyn Family, CensusError> {
    let names = meta.architectures().map_err(CensusError::HyperParams)?;
    let mut found: Vec<&'static dyn Family> = Vec::new();
    for name in &names {
        for entry in registry.entries() {
            let family = entry.family;
            if family.accepts().contains(name) && !found.iter().any(|f| f.name() == family.name()) {
                found.push(family);
            }
        }
    }
    let owned = || names.iter().map(|&n| n.to_owned()).collect();
    match found.as_slice() {
        [] => Err(CensusError::UnknownFamily {
            names: owned(),
            supported: registry.tags(),
        }),
        [one] => Ok(*one),
        many => Err(CensusError::AmbiguousFamily {
            names: owned(),
            families: many.iter().map(|f| f.name()).collect(),
        }),
    }
}

/// One tensor as a header gives it.
struct Raw {
    name: String,
    shape: Vec<u64>,
    quant: QuantScheme,
    extent: SourceExtent,
    /// Where to find it again for slicing: the shard and the tensor's index in it.
    at: (usize, usize),
}

impl Raw {
    fn slice(
        &self,
        weights: Weights<'_>,
        n: u16,
    ) -> Result<Vec<syntropy_manifest::slice::ExpertSlice>, CensusError> {
        match weights {
            Weights::Gguf(h) => {
                let tensor = h
                    .tensor(self.at.1)
                    .expect("the tensor was read from this header");
                gguf_experts(&tensor, n)
            }
            Weights::Safetensors(shards) => {
                let tensor = &shards[self.at.0].tensors[self.at.1];
                safetensors_experts(tensor, self.extent.file, n)
            }
        }
        .map_err(CensusError::Slice)
    }
}

fn raw_tensors(weights: Weights<'_>) -> Result<Vec<Raw>, CensusError> {
    match weights {
        Weights::Gguf(h) => Ok(h
            .tensors()
            .enumerate()
            .map(|(i, t)| Raw {
                name: t.name.to_owned(),
                shape: t.ne().to_vec(),
                quant: t.scheme(),
                extent: SourceExtent {
                    file: t.file,
                    offset: t.start,
                    len: t.bytes,
                },
                at: (0, i),
            })
            .collect()),
        Weights::Safetensors(shards) => {
            let mut out = Vec::new();
            for (s, shard) in shards.iter().enumerate() {
                let file = u16::try_from(s).map_err(|_| CensusError::Incomplete {
                    reason: format!("{} shards are more than a file index holds", shards.len()),
                })?;
                for (i, t) in shard.tensors.iter().enumerate() {
                    // Unquantised types only: an FP8, FP6 or FP4 tensor needs its
                    // `quantization_config` scheme and scales, which come with their task.
                    let quant = match t.dtype.dtype() {
                        Some(d)
                            if matches!(
                                d,
                                DType::F32
                                    | DType::F16
                                    | DType::BF16
                                    | DType::I8
                                    | DType::U8
                                    | DType::I32
                            ) =>
                        {
                            QuantScheme::Plain(d)
                        }
                        _ => {
                            return Err(CensusError::UnknownQuant {
                                name: t.name.clone(),
                                dtype: t.dtype.name().to_owned(),
                            });
                        }
                    };
                    out.push(Raw {
                        name: t.name.clone(),
                        shape: t.shape.clone(),
                        quant,
                        extent: SourceExtent {
                            file,
                            offset: t.offset,
                            len: t.len,
                        },
                        at: (s, i),
                    });
                }
            }
            Ok(out)
        }
    }
}

/// A table's rows: its outermost dimension, last in GGUF's order and first in safetensors'.
fn outermost(shape: &[u64], weights: Weights<'_>) -> u64 {
    match weights {
        Weights::Gguf(_) => shape.last().copied(),
        Weights::Safetensors(_) => shape.first().copied(),
    }
    .unwrap_or(1)
    .max(1)
}

/// A record with what grouping needs.
#[derive(Clone, Copy)]
struct Placed {
    id: TensorId,
    class: TensorClass,
    layer: Option<LayerIx>,
    role: TensorRole,
    block: BlockSel,
    rows: u64,
}

/// What forms a group, in the order groups are listed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    Lookup(u32),
    Dense(u16, u8, u8),
    Router(u16),
    Expert(u16, u16),
    Head(u16),
}

/// RFC-0041 §3.1's groups (module docs).
fn group(
    hp: &HyperParams,
    records: &[TensorRecord],
    placed: &[Placed],
) -> Result<Vec<TensorGroup>, CensusError> {
    let mut members: BTreeMap<Key, (TensorClass, Option<LayerIx>, Vec<TensorId>)> = BTreeMap::new();
    let mut epilogue_dense = Vec::new();
    let mut lookup_rows: BTreeMap<u32, u64> = BTreeMap::new();
    let mut embedding: Option<(u32, TensorId)> = None;
    for p in placed {
        let layer = p.layer.map_or(u16::MAX, |l| l.0);
        let key = match (p.class, p.block) {
            (TensorClass::RowLookup, _) => {
                let n = p.id.0;
                lookup_rows.insert(n, p.rows);
                if p.role == TensorRole::Embed && embedding.is_none() {
                    embedding = Some((n, p.id));
                }
                Key::Lookup(n)
            }
            (TensorClass::Expert { layer: l, id }, _) => Key::Expert(l.0, id.0),
            (TensorClass::Router { layer: l }, _) => Key::Router(l.0),
            (TensorClass::Head, _) => Key::Head(layer),
            (TensorClass::DenseEveryToken, BlockSel::Epilogue) => {
                epilogue_dense.push(p.id);
                continue;
            }
            (TensorClass::DenseEveryToken, BlockSel::Layer(Block::Moe))
                if p.role == TensorRole::Norm =>
            {
                Key::Router(layer)
            }
            (TensorClass::DenseEveryToken, block) => {
                let (b, sub) = block_code(block);
                Key::Dense(layer, b, sub)
            }
            (class, _) => {
                return Err(CensusError::Incomplete {
                    reason: format!(
                        "`{}` is of a class the census does not place: {class:?}",
                        records[p.id.0 as usize].name
                    ),
                });
            }
        };
        let class = match key {
            Key::Router(l) => TensorClass::Router { layer: LayerIx(l) },
            _ => p.class,
        };
        members
            .entry(key)
            .or_insert_with(|| (class, p.layer, Vec::new()))
            .2
            .push(p.id);
    }

    // The epilogue's dense tensors join the last head group; with no head, the embedding is tied
    // and a head group over it is made.
    let tied = !members.keys().any(|k| matches!(k, Key::Head(_)));
    if tied {
        let Some((_, embed)) = embedding else {
            return Err(CensusError::Incomplete {
                reason: "the model has neither an LM head nor a token embedding to tie one to"
                    .into(),
            });
        };
        let mut tensors = epilogue_dense.clone();
        tensors.push(embed);
        members.insert(Key::Head(u16::MAX), (TensorClass::Head, None, tensors));
    } else if let Some((_, (_, _, tensors))) = members
        .iter_mut()
        .rev()
        .find(|(k, _)| matches!(k, Key::Head(_)))
    {
        tensors.splice(0..0, epilogue_dense.iter().copied());
    }

    let n_experts = hp.count(N_EXPERTS).unwrap_or(0);
    let top_k = hp.count(TOP_K).unwrap_or(0);
    // The expert prior needs both counts: a MoE model without them would read as reading no
    // expert bytes at all, so it is refused rather than placed on that.
    if members.keys().any(|k| matches!(k, Key::Expert(..)))
        && (n_experts == 0 || top_k == 0 || top_k > n_experts)
    {
        return Err(CensusError::Incomplete {
            reason: format!(
                "routed experts need `{N_EXPERTS}` and `{TOP_K}` with 0 < top_k <= n_experts; the family gives {n_experts} and {top_k}"
            ),
        });
    }
    let bytes_of = |ids: &[TensorId]| -> u64 {
        ids.iter()
            .map(|id| {
                let r = &records[id.0 as usize];
                r.bytes.0 + r.aux.iter().map(|a| a.len).sum::<u64>()
            })
            .sum()
    };
    let mut groups = Vec::with_capacity(members.len());
    let mut lookup_group = None;
    let mut head_group = None;
    for (key, (class, layer, tensors)) in members {
        let id = TensorGroupId(u32::try_from(groups.len()).unwrap_or(u32::MAX));
        let bytes = bytes_of(&tensors);
        let read = match key {
            Key::Lookup(n) => bytes / lookup_rows.get(&n).copied().unwrap_or(1),
            Key::Expert(..) => {
                u64::try_from(u128::from(bytes) * u128::from(top_k) / u128::from(n_experts))
                    .unwrap_or(u64::MAX)
            }
            _ => bytes,
        };
        if matches!(key, Key::Lookup(n) if embedding.is_some_and(|(e, _)| e == n)) {
            lookup_group = Some(id);
        }
        if key == Key::Head(u16::MAX) && tied {
            head_group = Some(id);
        }
        groups.push(TensorGroup {
            id,
            class,
            layer,
            tensors,
            bytes: Bytes(bytes),
            read_per_token: Bytes(read),
            shared_with: None,
        });
    }
    if let (Some(lookup), Some(head)) = (lookup_group, head_group) {
        groups[lookup.0 as usize].shared_with = Some(head);
        groups[head.0 as usize].shared_with = Some(lookup);
    }
    Ok(groups)
}

/// A dense group's block, as a sortable code: the block's position, and the MTP module.
fn block_code(block: BlockSel) -> (u8, u8) {
    match block {
        BlockSel::Prologue => (0, 0),
        BlockSel::Layer(b) => (1, b as u8),
        BlockSel::Mtp(n) => (2, n),
        BlockSel::Epilogue => (3, 0),
        BlockSel::Encoder => (4, 0),
    }
}

/// The checks of the module docs that need only the classified tensors.
fn check_complete(hp: &HyperParams, placed: &[Placed]) -> Result<(), CensusError> {
    let incomplete = |reason: String| Err(CensusError::Incomplete { reason });
    for layer in 0..hp.n_layers {
        if !placed.iter().any(|p| p.layer == Some(LayerIx(layer))) {
            return incomplete(format!("decoder layer {layer} has no tensor"));
        }
    }
    if !placed
        .iter()
        .any(|p| p.class == TensorClass::RowLookup && p.role == TensorRole::Embed)
    {
        return incomplete("the model has no token embedding".into());
    }
    // Every expert of a layer has the same roles, and the layer every expert up to `n_experts`.
    let mut experts: BTreeMap<u16, BTreeMap<ExpertIx, Vec<TensorRole>>> = BTreeMap::new();
    for p in placed {
        if let TensorClass::Expert { layer, id } = p.class {
            experts
                .entry(layer.0)
                .or_default()
                .entry(id)
                .or_default()
                .push(p.role);
        }
    }
    let n_experts = hp.count(N_EXPERTS).unwrap_or(0);
    for (layer, by_expert) in &experts {
        if by_expert.len() as u64 != n_experts {
            return incomplete(format!(
                "layer {layer} has {} routed experts, and the hyperparameters name {n_experts}",
                by_expert.len()
            ));
        }
        let sorted = |roles: &[TensorRole]| {
            let mut roles: Vec<String> = roles.iter().map(|r| format!("{r:?}")).collect();
            roles.sort_unstable();
            roles
        };
        let mut all = by_expert.iter().map(|(e, roles)| (*e, sorted(roles)));
        let Some((_, first)) = all.next() else {
            continue;
        };
        if let Some((e, _)) = all.find(|(_, roles)| *roles != first) {
            return incomplete(format!(
                "in layer {layer}, expert {} does not have the tensors expert 0 has",
                e.0
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
