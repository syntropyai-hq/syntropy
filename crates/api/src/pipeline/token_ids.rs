// SPDX-License-Identifier: MIT OR Apache-2.0

//! The token-ID range check at ingress (RFC-0050 §2.3; PROD-E2-T20).
//!
//! Token IDs reach unsafe device code: the sampler indexes per-sequence count tables by token ID
//! (RFC-0012 §8.1), and `RowGather` indexes embedding rows (RFC-0040 §3.1). So every token ID a
//! request carries is checked here, in one function, before routing: each must be below the
//! model's `vocab_limit`, the smallest of its embedding rows, LM-head rows and sampler table size
//! ([`vocab_limit`]). The first ID that is not fails the request with
//! [`TokenIdOutOfRange`], which names the field, the position and the limit. Nothing is clamped
//! or dropped.
//!
//! [`check_token_ids`] is the validator, one call per field. [`check_request`] runs it over the
//! fields a [`GenerateRequest`] carries at M0, the native `tokens` and `logit_bias`, and over the
//! rendered prompt; the pipeline calls it after stage 5. The other fields in [`TokenField`] come
//! with the endpoints that parse them, and each of those tasks calls [`check_token_ids`] for its
//! field.
//!
//! [`manifest_vocab_limit`] derives the limit from a [`ModelManifest`] (RFC-0050 A9: from
//! `HyperParams` and the census, with no new manifest field), and [`ModelInfo::from_manifest`]
//! puts it in the [`ModelInfo`] stage 3 returns.

use std::fmt;

use syntropy_core::ErrorClass;
use syntropy_manifest::census::{ModelManifest, SourceFormat, TensorClass, TensorRecord};
use syntropy_manifest::hparams::HyperParams;

use super::stages::ModelInfo;

use super::types::{GenerateRequest, Input};

/// A request field that carries token IDs (RFC-0050 §2.3).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenField {
    /// The native API's prompt `tokens` (RFC-0050 §13.1).
    Tokens,
    /// The parity endpoint's `teacher.tokens` (RFC-0050 §13.6).
    TeacherTokens,
    /// The keys of `logit_bias`.
    LogitBias,
    /// Stop token IDs.
    StopTokenIds,
    /// A token-array `prompt` on legacy `/v1/completions` (RFC-0050 §3.5).
    CompletionsPrompt,
    /// A token-array `input` on embeddings (RFC-0050 §14).
    EmbeddingsInput,
    /// The token set of a compiled grammar (RFC-0050 §7).
    GrammarTokens,
    /// The prompt as stage 5 rendered and tokenized it. An ID out of range here means the
    /// tokenizer and the manifest disagree, which census should have refused (RFC-0043 §5.2).
    RenderedPrompt,
}

impl TokenField {
    /// The field as an error names it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            TokenField::Tokens => "tokens",
            TokenField::TeacherTokens => "teacher.tokens",
            TokenField::LogitBias => "logit_bias",
            TokenField::StopTokenIds => "stop_token_ids",
            TokenField::CompletionsPrompt => "prompt",
            TokenField::EmbeddingsInput => "input",
            TokenField::GrammarTokens => "constraint",
            TokenField::RenderedPrompt => "rendered prompt",
        }
    }
}

impl fmt::Display for TokenField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A token ID at or above the model's `vocab_limit`: 400 `token_id_out_of_range` (RFC-0050
/// §12.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenIdOutOfRange {
    /// The field that holds it.
    pub field: TokenField,
    /// Its position in the field, counted from 0: the first offending one. For `logit_bias`, the
    /// entry's position in the order the client listed them.
    pub position: u64,
    /// The ID.
    pub id: u32,
    /// The model's `vocab_limit`.
    pub vocab_limit: u32,
}

impl TokenIdOutOfRange {
    /// The error's code in RFC-0050 §12's registry.
    pub const CODE: &'static str = "token_id_out_of_range";
    /// The code's number in that registry.
    pub const NUMBER: u16 = 33;

    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        ErrorClass::InvalidRequest
    }
}

impl fmt::Display for TokenIdOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}[{}] is token ID {}, not below the model's vocab_limit of {}",
            self.field, self.position, self.id, self.vocab_limit
        )
    }
}

impl std::error::Error for TokenIdOutOfRange {}

/// `vocab_limit = min(embedding rows, LM-head rows, sampler table size)` (RFC-0050 §2.3).
#[must_use]
pub fn vocab_limit(embedding_rows: u32, lm_head_rows: u32, sampler_table: u32) -> u32 {
    embedding_rows.min(lm_head_rows).min(sampler_table)
}

/// Why a manifest gives no `vocab_limit`. Census refuses a model without a token embedding
/// (`HeaderInvalid`), so a manifest from the census builder never fails here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VocabLimitError {
    /// What is missing or malformed.
    pub reason: String,
}

impl VocabLimitError {
    /// Its class (RFC-0001 §10): the manifest is not what the census builds.
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        ErrorClass::Integrity
    }
}

impl fmt::Display for VocabLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no vocab_limit: {}", self.reason)
    }
}

impl std::error::Error for VocabLimitError {}

/// A model's `vocab_limit`, from its manifest (RFC-0050 §2.3, A9). See [`vocab_limit_of`].
///
/// # Errors
///
/// [`VocabLimitError`] as [`vocab_limit_of`].
pub fn manifest_vocab_limit(manifest: &ModelManifest) -> Result<u32, VocabLimitError> {
    vocab_limit_of(manifest.format, &manifest.hparams, &manifest.census.tensors)
}

/// `vocab_limit = min(embedding rows, LM-head rows, sampler table size)` from the census records
/// and the hyperparameters (RFC-0050 A9):
///
/// - **Embedding rows**: the rows of every [`TensorClass::RowLookup`] record, the smallest. The
///   class is the tables addressed by row from token IDs (the token embedding, and n-gram and
///   per-layer embedding tables), so an ID must be in range for each; a larger extra table
///   changes nothing, and a smaller one lowers the limit rather than let an ID past it.
/// - **LM-head rows**: the rows of every [`TensorClass::Head`] record (the LM head, and MTP
///   output heads), the smallest. With none the embedding is tied: the census puts the
///   embedding's record in the head group instead, and the head has the embedding's rows.
/// - **Sampler table size**: `hparams.vocab`. The sampler's per-sequence count tables have one
///   entry per vocabulary token (RFC-0012 §8.1), and the manifest records no other size for them.
///
/// A record's rows are its vocabulary dimension: the outermost in a safetensors shape
/// (`[vocab, hidden]`), the innermost-last in GGUF's `ne` order (`[hidden, vocab]`). A
/// one-dimensional record, such as a head bias, has its length as rows.
///
/// # Errors
///
/// [`VocabLimitError`] when no record is a `RowLookup` table, or a lookup or head record has an
/// empty shape.
pub fn vocab_limit_of(
    format: SourceFormat,
    hparams: &HyperParams,
    tensors: &[TensorRecord],
) -> Result<u32, VocabLimitError> {
    let rows = |r: &TensorRecord| -> Result<u64, VocabLimitError> {
        let row = match format {
            SourceFormat::Safetensors => r.shape.first(),
            SourceFormat::Gguf { .. } => r.shape.last(),
        };
        row.copied().ok_or_else(|| VocabLimitError {
            reason: format!("`{}` has an empty shape", r.name),
        })
    };
    let smallest = |class: fn(&TensorClass) -> bool| -> Result<Option<u64>, VocabLimitError> {
        tensors
            .iter()
            .filter(|r| class(&r.class))
            .try_fold(None, |min: Option<u64>, r| {
                let n = rows(r)?;
                Ok(Some(min.map_or(n, |m| m.min(n))))
            })
    };
    let embedding =
        smallest(|c| matches!(c, TensorClass::RowLookup))?.ok_or_else(|| VocabLimitError {
            reason: "the census has no RowLookup table, so no token embedding".to_owned(),
        })?;
    let head = smallest(|c| matches!(c, TensorClass::Head))?.unwrap_or(embedding);
    let limit = embedding.min(head).min(u64::from(hparams.vocab));
    Ok(u32::try_from(limit).expect("at most hparams.vocab, a u32"))
}

impl ModelInfo {
    /// The [`ModelInfo`] of a model whose manifest is `manifest`: its name, its context length,
    /// and its [`manifest_vocab_limit`].
    ///
    /// # Errors
    ///
    /// [`VocabLimitError`] when the manifest gives no `vocab_limit`.
    pub fn from_manifest(
        name: impl Into<String>,
        context_length: u32,
        manifest: &ModelManifest,
    ) -> Result<ModelInfo, VocabLimitError> {
        Ok(ModelInfo {
            name: name.into(),
            context_length,
            vocab_limit: manifest_vocab_limit(manifest)?,
        })
    }
}

/// Checks that every ID of `field` is below `vocab_limit`, in one pass.
///
/// # Errors
///
/// [`TokenIdOutOfRange`] for the first ID that is not.
pub fn check_token_ids(
    field: TokenField,
    ids: impl IntoIterator<Item = u32>,
    vocab_limit: u32,
) -> Result<(), TokenIdOutOfRange> {
    match ids
        .into_iter()
        .enumerate()
        .find(|&(_, id)| id >= vocab_limit)
    {
        None => Ok(()),
        Some((position, id)) => Err(TokenIdOutOfRange {
            field,
            position: position as u64,
            id,
            vocab_limit,
        }),
    }
}

/// Checks the token IDs of `request` and its rendered `prompt` against `vocab_limit`: the native
/// `tokens`, then the `logit_bias` keys, then the rendered prompt.
///
/// # Errors
///
/// [`TokenIdOutOfRange`] for the first offending ID, in that order of fields.
pub fn check_request(
    request: &GenerateRequest,
    prompt: &[u32],
    vocab_limit: u32,
) -> Result<(), TokenIdOutOfRange> {
    if let Input::Tokens(tokens) = &request.input {
        check_token_ids(TokenField::Tokens, tokens.iter().copied(), vocab_limit)?;
    }
    check_token_ids(
        TokenField::LogitBias,
        request.sampling.logit_bias.iter().map(|&(id, _)| id),
        vocab_limit,
    )?;
    check_token_ids(
        TokenField::RenderedPrompt,
        prompt.iter().copied(),
        vocab_limit,
    )
}
