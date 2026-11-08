// SPDX-License-Identifier: MIT OR Apache-2.0

//! Span-aware encoding: template spans with special-token matching on, content spans with it off
//! (RFC-0043 §8.7, RFC-0050 §6.4, ADR-0431 decision 7).
//!
//! Rendering a chat template yields a segment map: [`Span`]s marked [`SpanKind::Template`] (text
//! the template wrote) or [`SpanKind::Content`] (a value from the request). [`SpanEncoder::encode`]
//! turns the map into token IDs in one call:
//!
//! 1. The tokenizer's added tokens (its control tokens, special or not) are matched only inside
//!    template spans, the way the engine matches them in a whole string: leftmost-longest,
//!    with `lstrip` and `rstrip` taking whitespace from the same template span only. A match
//!    never reaches into content.
//! 2. The text between two matches, which can run across template and content spans, is encoded
//!    by a copy of the tokenizer whose added tokens are never matched, so content becomes the
//!    ordinary tokens of its text, byte for byte. Nothing is stripped or rewritten.
//!
//! When no added token's text lies in content, whole or in part, the IDs are exactly the ones the
//! engine gives the concatenated text, because the engine also splits at added tokens first and
//! encodes the pieces between them alone. When content holds one, or content and the template
//! spell one together across a span boundary, it stays text.
//!
//! [`SpecialTokens::Raw`] is the native API's `raw_special_tokens` opt-in (RFC-0050 §13.1): every
//! span is matched as template text, which gives the engine's own encoding of the concatenation.
//!
//! An added token is a control token here whether or not the tokenizer marks it `special`:
//! Qwen3's `<think>` and `<tool_call>` are added tokens with `special: false`, and the engine's own
//! "encode special tokens as text" switch would still match them. The literal copy marks every
//! added token special before it turns that switch on.
//!
//! The encoder fails closed. It refuses a tokenizer with an added token it cannot match the way
//! the engine does (`normalized: true`, `single_word: true`, empty text), and it refuses an
//! encoding in which ordinary text produced an added token's ID, which happens only when the
//! model's own vocabulary spells a control token.

use std::collections::BTreeMap;
use std::fmt;

use syntropy_core::ErrorClass;
use tokenizers::Tokenizer;

/// Who wrote a span's text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpanKind {
    /// The chat template: role headers, turn boundaries, tool-call scaffolding.
    Template,
    /// A value from the request: a message of any role, a tool result, a tool description, a
    /// document.
    Content,
}

/// One piece of a rendered prompt.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    /// Who wrote it.
    pub kind: SpanKind,
    /// Its text.
    pub text: String,
}

impl Span {
    /// Text the template wrote.
    pub fn template(text: impl Into<String>) -> Span {
        Span {
            kind: SpanKind::Template,
            text: text.into(),
        }
    }

    /// Text from the request.
    pub fn content(text: impl Into<String>) -> Span {
        Span {
            kind: SpanKind::Content,
            text: text.into(),
        }
    }
}

/// Whether control-token text in content spans becomes control tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SpecialTokens {
    /// Content is tokenized literally (the default everywhere).
    #[default]
    Literal,
    /// Content is matched like template text: the native API's `raw_special_tokens: true`.
    Raw,
}

/// Why a tokenizer or an encoding was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpanError {
    /// The tokenizer has an added token whose matching options the encoder does not reproduce.
    UnsupportedAddedToken {
        /// The token's ID.
        id: u32,
        /// Its text.
        content: String,
        /// The option: `normalized`, `single_word` or `empty`.
        option: &'static str,
    },
    /// Ordinary text encoded to an added token's ID: the model's vocabulary spells a control
    /// token, so a literal encoding cannot keep it out.
    ControlFromText {
        /// The ID.
        id: u32,
        /// The text that produced it.
        text: String,
    },
    /// The tokenizer engine failed.
    Engine(String),
}

impl SpanError {
    /// The error class.
    pub fn class(&self) -> ErrorClass {
        match self {
            SpanError::UnsupportedAddedToken { .. } | SpanError::ControlFromText { .. } => {
                ErrorClass::Unsupported
            }
            SpanError::Engine(_) => ErrorClass::Internal,
        }
    }
}

impl fmt::Display for SpanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpanError::UnsupportedAddedToken {
                id,
                content,
                option,
            } => write!(
                f,
                "added token {id} ({content:?}) uses `{option}`, which span-aware encoding does \
                 not support"
            ),
            SpanError::ControlFromText { id, text } => write!(
                f,
                "ordinary text {text:?} encoded to control token {id}: the model's vocabulary \
                 spells a control token"
            ),
            SpanError::Engine(e) => write!(f, "tokenizer engine: {e}"),
        }
    }
}

impl std::error::Error for SpanError {}

/// One added token, as the matcher needs it.
#[derive(Clone, Debug)]
struct Added {
    id: u32,
    content: String,
    lstrip: bool,
    rstrip: bool,
}

/// A tokenizer that encodes segment maps (RFC-0043 §8.7). Build one per model and share it.
#[derive(Clone)]
pub struct SpanEncoder {
    tokenizer: Tokenizer,
    /// The same tokenizer with every added token marked special and special-token matching off.
    literal: Tokenizer,
    /// Added tokens by first byte, longest first within each byte.
    by_first_byte: BTreeMap<u8, Vec<Added>>,
    /// Every added token's ID, sorted.
    control_ids: Vec<u32>,
}

impl fmt::Debug for SpanEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpanEncoder")
            .field("control_ids", &self.control_ids)
            .finish_non_exhaustive()
    }
}

impl SpanEncoder {
    /// Builds the encoder for a loaded tokenizer.
    ///
    /// # Errors
    ///
    /// [`SpanError::UnsupportedAddedToken`] for an added token that is `normalized`,
    /// `single_word` or empty; [`SpanError::Engine`] when the tokenizer does not serialize and
    /// load again.
    pub fn new(tokenizer: Tokenizer) -> Result<SpanEncoder, SpanError> {
        let mut added: Vec<Added> = Vec::new();
        for (id, token) in tokenizer.get_added_tokens_decoder() {
            let option = if token.content.is_empty() {
                Some("empty")
            } else if token.normalized {
                Some("normalized")
            } else if token.single_word {
                Some("single_word")
            } else {
                None
            };
            if let Some(option) = option {
                return Err(SpanError::UnsupportedAddedToken {
                    id,
                    content: token.content,
                    option,
                });
            }
            added.push(Added {
                id,
                content: token.content,
                lstrip: token.lstrip,
                rstrip: token.rstrip,
            });
        }
        let mut control_ids: Vec<u32> = added.iter().map(|a| a.id).collect();
        control_ids.sort_unstable();
        control_ids.dedup();

        let mut by_first_byte: BTreeMap<u8, Vec<Added>> = BTreeMap::new();
        for a in added {
            by_first_byte
                .entry(a.content.as_bytes()[0])
                .or_default()
                .push(a);
        }
        for list in by_first_byte.values_mut() {
            // Longest first; equal lengths cannot both match at one position.
            list.sort_by_key(|a| std::cmp::Reverse(a.content.len()));
        }

        let literal = literal_copy(&tokenizer)?;
        Ok(SpanEncoder {
            tokenizer,
            literal,
            by_first_byte,
            control_ids,
        })
    }

    /// The tokenizer it was built from.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Every added token's ID, sorted: the IDs no content span may produce.
    pub fn control_ids(&self) -> &[u32] {
        &self.control_ids
    }

    /// Whether `id` is an added token's.
    pub fn is_control(&self, id: u32) -> bool {
        self.control_ids.binary_search(&id).is_ok()
    }

    /// The text of every added token, in ID order.
    pub fn control_texts(&self) -> Vec<(u32, &str)> {
        let mut out: Vec<(u32, &str)> = self
            .by_first_byte
            .values()
            .flatten()
            .map(|a| (a.id, a.content.as_str()))
            .collect();
        out.sort_unstable();
        out
    }

    /// Encodes a segment map. No BOS or EOS is added: the template writes them, or the caller
    /// adds them from `TokenizerSpec::add_bos` and `add_eos`.
    ///
    /// # Errors
    ///
    /// [`SpanError::ControlFromText`] when ordinary text produced a control token's ID;
    /// [`SpanError::Engine`] when the engine fails.
    pub fn encode(&self, spans: &[Span], mode: SpecialTokens) -> Result<Vec<u32>, SpanError> {
        let mut text = String::with_capacity(spans.iter().map(|s| s.text.len()).sum());
        // Byte ranges of the text in which added tokens are matched.
        let mut regions: Vec<(usize, usize)> = Vec::new();
        for span in spans {
            let start = text.len();
            text.push_str(&span.text);
            let matched = span.kind == SpanKind::Template || mode == SpecialTokens::Raw;
            if matched && start < text.len() {
                match regions.last_mut() {
                    Some(last) if last.1 == start => last.1 = text.len(),
                    _ => regions.push((start, text.len())),
                }
            }
        }

        let mut ids = Vec::new();
        let mut done = 0;
        for &(region_start, region_end) in &regions {
            let mut at = region_start;
            while at < region_end {
                let Some(found) = self.match_at(&text, at, region_end) else {
                    at += text[at..].chars().next().map_or(1, char::len_utf8);
                    continue;
                };
                let mut start = at;
                if found.lstrip {
                    let floor = done.max(region_start);
                    start = floor + text[floor..at].trim_end().len();
                }
                let mut stop = at + found.content.len();
                if found.rstrip {
                    stop = region_end - text[stop..region_end].trim_start().len();
                }
                self.encode_literal(&text[done..start], &mut ids)?;
                ids.push(found.id);
                done = stop;
                at = stop;
            }
        }
        self.encode_literal(&text[done..], &mut ids)?;
        Ok(ids)
    }

    /// The longest added token at `at` that ends by `end`.
    fn match_at(&self, text: &str, at: usize, end: usize) -> Option<&Added> {
        let rest = &text.as_bytes()[at..end];
        self.by_first_byte
            .get(rest.first()?)?
            .iter()
            .find(|a| rest.starts_with(a.content.as_bytes()))
    }

    fn encode_literal(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), SpanError> {
        if text.is_empty() {
            return Ok(());
        }
        let encoding = self
            .literal
            .encode_fast(text, false)
            .map_err(|e| SpanError::Engine(e.to_string()))?;
        if let Some(&id) = encoding.get_ids().iter().find(|&&id| self.is_control(id)) {
            return Err(SpanError::ControlFromText {
                id,
                text: text.to_owned(),
            });
        }
        ids.extend_from_slice(encoding.get_ids());
        Ok(())
    }
}

/// The tokenizer with every added token marked special and special-token matching off, so no
/// added token is matched in the text it encodes. IDs are unchanged: only the flags move.
fn literal_copy(tokenizer: &Tokenizer) -> Result<Tokenizer, SpanError> {
    let engine = |e: &dyn fmt::Display| SpanError::Engine(e.to_string());
    let json = tokenizer.to_string(false).map_err(|e| engine(&e))?;
    let mut value: serde_json::Value = serde_json::from_str(&json).map_err(|e| engine(&e))?;
    if let Some(tokens) = value
        .get_mut("added_tokens")
        .and_then(serde_json::Value::as_array_mut)
    {
        for token in tokens {
            if let Some(token) = token.as_object_mut() {
                token.insert("special".into(), serde_json::Value::Bool(true));
            }
        }
    }
    let bytes = serde_json::to_vec(&value).map_err(|e| engine(&e))?;
    let mut literal = Tokenizer::from_bytes(bytes).map_err(|e| engine(&e))?;
    literal.set_encode_special_tokens(true);
    Ok(literal)
}
