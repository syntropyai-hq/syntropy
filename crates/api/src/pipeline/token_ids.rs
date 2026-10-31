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

use std::fmt;

use syntropy_core::ErrorClass;

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
