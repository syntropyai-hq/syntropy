// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF tokenizers converted to the engine's model at census time (RFC-0043 §8.3; ADR-0431
//! decisions 2 and 3; COMP-E17-T03).
//!
//! A GGUF file carries its tokenizer as `tokenizer.ggml.*` metadata. [`from_gguf`] turns that
//! metadata into a serialized `tokenizer.json`, the one form `syntropy-text` loads for every
//! model (RFC-0043 §8.2), and wraps it in a [`TokenizerSpec`]. The conversion targets the
//! source repository's own `tokenizer.json`: encoding the same text through both gives the
//! same token IDs (RFC-0043 §8.6).
//!
//! **`gpt2`, byte-level BPE.** The vocabulary is `tokenizer.ggml.tokens` in ID order and the
//! merges are `tokenizer.ggml.merges` in file order, each `"left right"`. The pre-tokenizer is
//! a regex `Split` (behaviour `Isolated`) followed by a byte-level step without its own regex,
//! and the decoder is byte-level. The regex, the normalizer and the BPE `ignore_merges` flag
//! come from `tokenizer.ggml.pre` through [`PRE_TOKENIZERS`]. A name outside the table, or no
//! name, is `TokenizerUnsupported` (`(Graph, 6)`): a guessed regex silently changes tokens.
//!
//! **`llama`, SentencePiece-style BPE.** The vocabulary is every token, the merges are derived
//! from `tokenizer.ggml.scores` (below), byte fallback is on (`<0xNN>` tokens), and runs of
//! unknown pieces fuse into one unknown token. `tokenizer.ggml.add_space_prefix` (default
//! `true`) prepends the whitespace marker `▁`, and every space becomes `▁`. Both steps are a
//! normalizer, as the source repositories' `tokenizer.json` files of this kind have them, so
//! the marker is also prepended after each added token. The decoder maps `▁` back to a space,
//! joins byte-fallback tokens into UTF-8, and strips the one leading space the prefix added.
//! A file that asks for normalization this conversion does not model, `remove_extra_whitespaces`
//! set or a `precompiled_charsmap`, is `TokenizerUnsupported`.
//!
//! **Merges in score order.** For each token `t`, in order of score, highest first, and of ID
//! where scores tie, every split `t = l + r` with `l` and `r` both tokens is a merge; the splits
//! of one token are ordered by the IDs of `l`, then `r`. This is the standard
//! SentencePiece-to-BPE conversion: BPE then applies the merge whose result scores highest.
//!
//! **Token types** (`tokenizer.ggml.token_type`, required): normal (1) and byte (6) tokens are
//! the model's vocabulary; unknown (2) and control (3) tokens become special added tokens and
//! user-defined (4) tokens added tokens that are not special; unused (5) tokens get no entry, as
//! their IDs have no text. For `llama` every token but the unused ones is also in the model's
//! vocabulary, as SentencePiece models keep them. Every added token is matched on its raw text,
//! with no stripping. A GGUF file records only these types, so a token the source marked as not
//! special but the file exports as control (the exporter does this for `<|…|>` spellings) is
//! special here.
//!
//! **Specials** come from their keys: BOS, EOS, unknown, padding, and the end-of-turn set, which
//! is EOS with `eot_token_id` and `eom_token_id`. FIM prefix, middle and suffix come from
//! `fim_pre_token_id`, `fim_mid_token_id` and `fim_suf_token_id` (or the older `prefix_token_id`,
//! `middle_token_id` and `suffix_token_id`), all three or none; with none set, from the added
//! tokens spelled as in [`FIM_SPELLINGS`], as for `tokenizer.json` sources. `add_bos` and
//! `add_eos` come from `add_bos_token` and `add_eos_token`. Without the key, `add_eos` is
//! `false` and `add_bos` is `true` for `llama` and the pre-tokenizer's default for `gpt2`
//! ([`PreTokenizer::add_bos`]). A post-processor adds BOS and EOS as the flags say. The chat
//! template is `tokenizer.chat_template`, used as shipped.
//!
//! A file without `tokenizer.ggml.model` has no tokenizer, and [`from_gguf`] returns `None`. A
//! model name other than `gpt2` and `llama` is `TokenizerUnsupported`: `bert` and `t5` are
//! COMP-E17-T08's (M1a). Metadata of the wrong type, arrays of different lengths, a token
//! that is not UTF-8 or is listed twice, a merge or special that names no token, and an
//! unknown token type are [`TokenizeError::Invalid`], class `Integrity`, with no code yet, as for
//! `tokenizer.json` sources.

use std::collections::HashMap;
use std::fmt;

use serde_json::{Map, Value, json};
use syntropy_core::ErrorClass;
use syntropy_manifest::gguf::{MetaError, MetaView, ValueRef};
use syntropy_manifest::tokenizer::{FIM_SPELLINGS, SpecialTokens, TemplateSource, TokenizerSpec};

#[cfg(test)]
pub(crate) mod tests;

/// The tokenizer model's name.
pub const MODEL: &str = "tokenizer.ggml.model";
/// The pre-tokenizer's name, for `gpt2`.
pub const PRE: &str = "tokenizer.ggml.pre";
/// Every token's text, in ID order.
pub const TOKENS: &str = "tokenizer.ggml.tokens";
/// Every token's type, in ID order.
pub const TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
/// Every token's score, in ID order, for `llama`.
pub const SCORES: &str = "tokenizer.ggml.scores";
/// The BPE merges, highest priority first, for `gpt2`.
pub const MERGES: &str = "tokenizer.ggml.merges";
/// Whether `llama` prepends the whitespace marker.
pub const ADD_SPACE_PREFIX: &str = "tokenizer.ggml.add_space_prefix";
/// SentencePiece's whitespace folding, which this conversion does not model.
pub const REMOVE_EXTRA_WHITESPACES: &str = "tokenizer.ggml.remove_extra_whitespaces";
/// SentencePiece's character normalization map, which this conversion does not model.
pub const PRECOMPILED_CHARSMAP: &str = "tokenizer.ggml.precompiled_charsmap";
/// Whether a prompt starts with BOS.
pub const ADD_BOS: &str = "tokenizer.ggml.add_bos_token";
/// Whether a prompt ends with EOS.
pub const ADD_EOS: &str = "tokenizer.ggml.add_eos_token";
/// The shipped chat template.
pub const CHAT_TEMPLATE: &str = "tokenizer.chat_template";

/// The special-token ID keys, by role.
const BOS_ID: &str = "tokenizer.ggml.bos_token_id";
const EOS_ID: &str = "tokenizer.ggml.eos_token_id";
const UNK_ID: &str = "tokenizer.ggml.unknown_token_id";
const PAD_ID: &str = "tokenizer.ggml.padding_token_id";
const EOT_ID: &str = "tokenizer.ggml.eot_token_id";
const EOM_ID: &str = "tokenizer.ggml.eom_token_id";
/// The FIM prefix, middle and suffix keys, current spelling first.
const FIM_KEYS: [[&str; 3]; 2] = [
    [
        "tokenizer.ggml.fim_pre_token_id",
        "tokenizer.ggml.fim_mid_token_id",
        "tokenizer.ggml.fim_suf_token_id",
    ],
    [
        "tokenizer.ggml.prefix_token_id",
        "tokenizer.ggml.middle_token_id",
        "tokenizer.ggml.suffix_token_id",
    ],
];

/// SentencePiece's whitespace marker.
const SPACE_MARKER: &str = "\u{2581}";

/// A `tokenizer.ggml.token_type` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_i64(v: i64) -> Option<Self> {
        Some(match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => return None,
        })
    }

    /// Whether the token is an added token, and if so whether it is special.
    fn added(self) -> Option<bool> {
        match self {
            Self::Unknown | Self::Control => Some(true),
            Self::UserDefined => Some(false),
            Self::Normal | Self::Unused | Self::Byte => None,
        }
    }
}

/// What a `gpt2` pre-tokenizer name stands for: the settings of the source repositories'
/// `tokenizer.json` files that GGUF files with this name are exported from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreTokenizer {
    /// The `tokenizer.ggml.pre` value.
    pub name: &'static str,
    /// The split regex, as the source's `Split` pre-tokenizer spells it.
    pub regex: &'static str,
    /// Whether the source normalizes text to NFC first.
    pub nfc: bool,
    /// BPE's `ignore_merges`: a word that is itself a token is taken whole.
    pub ignore_merges: bool,
    /// `add_bos` when the file has no `add_bos_token` key.
    pub add_bos: bool,
}

/// The known `gpt2` pre-tokenizer names (RFC-0043 §8.3). Keeping it current with the ggml pin
/// is a pin-bump checklist item (RFC-0043 open question 5; COMP-E17-T06). A name missing here
/// is refused, never guessed.
pub const PRE_TOKENIZERS: [PreTokenizer; 3] = [
    // The original byte-level BPE split.
    PreTokenizer {
        name: "gpt-2",
        regex: r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        nfc: false,
        ignore_merges: false,
        add_bos: false,
    },
    // Llama 3 and its successors, Llama-3.2-1B among them (the `llama` family's M0 checkpoint).
    PreTokenizer {
        name: "llama-bpe",
        regex: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        nfc: false,
        ignore_merges: true,
        add_bos: true,
    },
    // Qwen2 and its successors, the Qwen3 M0 checkpoints among them: single digits, and NFC.
    PreTokenizer {
        name: "qwen2",
        regex: r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        nfc: true,
        ignore_merges: false,
        add_bos: false,
    },
];

/// The entry of [`PRE_TOKENIZERS`] named `name`.
#[must_use]
pub fn pre_tokenizer(name: &str) -> Option<&'static PreTokenizer> {
    PRE_TOKENIZERS.iter().find(|p| p.name == name)
}

/// Why a GGUF tokenizer cannot be converted.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenizeError {
    /// `TokenizerUnsupported`, `(Graph, 6)`, class `Unsupported`: a tokenizer model,
    /// pre-tokenizer name or setting this build does not convert.
    Unsupported {
        /// The key.
        key: &'static str,
        /// Its value, `None` when the key is missing.
        value: Option<String>,
    },
    /// The metadata does not describe a tokenizer (class `Integrity`; no code yet).
    Invalid {
        /// The key.
        key: String,
        /// What is wrong.
        reason: String,
    },
}

impl TokenizeError {
    /// RFC-0043 §10's `Graph` code, where one is assigned.
    #[must_use]
    pub fn code(&self) -> Option<u16> {
        match self {
            Self::Unsupported { .. } => Some(6),
            Self::Invalid { .. } => None,
        }
    }

    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Unsupported { .. } => ErrorClass::Unsupported,
            Self::Invalid { .. } => ErrorClass::Integrity,
        }
    }

    fn invalid(key: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Invalid {
            key: key.into(),
            reason: reason.into(),
        }
    }

    fn unsupported(key: &'static str, value: Option<&str>) -> Self {
        Self::Unsupported {
            key,
            value: value.map(str::to_owned),
        }
    }
}

impl From<MetaError> for TokenizeError {
    fn from(e: MetaError) -> Self {
        Self::invalid(e.key, format!("holds {}, expected {}", e.found, e.expected))
    }
}

impl fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported {
                key,
                value: Some(v),
            } => write!(
                f,
                "TokenizerUnsupported (Graph 6): `{key}` is {v:?}, which this build does not convert"
            ),
            Self::Unsupported { key, value: None } => write!(
                f,
                "TokenizerUnsupported (Graph 6): `{key}` is missing, so the tokenizer cannot be converted"
            ),
            Self::Invalid { key, reason } => write!(f, "GGUF tokenizer `{key}`: {reason}"),
        }
    }
}

impl std::error::Error for TokenizeError {}

/// The tokenizer kinds this build converts.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// `gpt2`, with its pre-tokenizer.
    ByteLevel(&'static PreTokenizer),
    /// `llama`, with `add_space_prefix`.
    SentencePiece { space_prefix: bool },
}

/// The tokenizer of a GGUF file's metadata, converted (module documentation). `None` when the
/// file has no `tokenizer.ggml.model`.
///
/// # Errors
///
/// [`TokenizeError::Unsupported`] for a model, pre-tokenizer name or setting this build does
/// not convert, and [`TokenizeError::Invalid`] for metadata that does not describe a tokenizer.
pub fn from_gguf(meta: &MetaView) -> Result<Option<TokenizerSpec>, TokenizeError> {
    let Some(model) = meta.str(MODEL)? else {
        return Ok(None);
    };
    let kind = match model {
        "gpt2" => {
            let pre = meta.str(PRE)?;
            let table = pre
                .and_then(pre_tokenizer)
                .ok_or_else(|| TokenizeError::unsupported(PRE, pre))?;
            Kind::ByteLevel(table)
        }
        "llama" => {
            // SentencePiece-style files name no pre-tokenizer, or `default`.
            match meta.str(PRE)? {
                None | Some("default") => {}
                Some(other) => return Err(TokenizeError::unsupported(PRE, Some(other))),
            }
            if meta.bool(REMOVE_EXTRA_WHITESPACES)? == Some(true) {
                return Err(TokenizeError::unsupported(
                    REMOVE_EXTRA_WHITESPACES,
                    Some("true"),
                ));
            }
            if meta.contains(PRECOMPILED_CHARSMAP) {
                return Err(TokenizeError::unsupported(
                    PRECOMPILED_CHARSMAP,
                    Some("present"),
                ));
            }
            Kind::SentencePiece {
                space_prefix: meta.bool(ADD_SPACE_PREFIX)?.unwrap_or(true),
            }
        }
        other => return Err(TokenizeError::unsupported(MODEL, Some(other))),
    };

    let vocab = Vocab::read(meta)?;
    let specials = specials(meta, &vocab)?;
    let add_bos = meta.bool(ADD_BOS)?.unwrap_or(match kind {
        Kind::ByteLevel(pre) => pre.add_bos,
        Kind::SentencePiece { .. } => true,
    });
    let add_eos = meta.bool(ADD_EOS)?.unwrap_or(false);
    for (flag, key, id) in [
        (add_bos, BOS_ID, specials.bos),
        (add_eos, EOS_ID, specials.eos),
    ] {
        if flag && id.is_none() {
            return Err(TokenizeError::invalid(
                key,
                "is missing, but the file asks for the token to be added",
            ));
        }
    }

    let merges = match kind {
        Kind::ByteLevel(_) => vocab.listed_merges(meta)?,
        Kind::SentencePiece { .. } => vocab.score_merges(meta)?,
    };
    let model_json = serialize(&vocab, &merges, kind, &specials, add_bos, add_eos);
    let chat_template = meta.str(CHAT_TEMPLATE)?.map(str::to_owned);
    let template_source = if chat_template.is_some() {
        TemplateSource::Shipped
    } else {
        TemplateSource::None
    };
    Ok(Some(TokenizerSpec::new(
        model_json,
        specials,
        add_bos,
        add_eos,
        chat_template,
        template_source,
    )))
}

/// The tokens and their types, validated.
struct Vocab<'a> {
    tokens: Vec<&'a str>,
    types: Vec<TokenType>,
    /// The ID of every token the model's vocabulary holds.
    ids: HashMap<&'a str, u32>,
}

impl<'a> Vocab<'a> {
    fn read(meta: &'a MetaView) -> Result<Self, TokenizeError> {
        // A SentencePiece vocabulary keeps its added tokens in the model too.
        let sentencepiece = meta.str(MODEL)? == Some("llama");
        let array = meta
            .array(TOKENS)?
            .ok_or_else(|| TokenizeError::invalid(TOKENS, "is missing"))?;
        let strings = array
            .strings()
            .ok_or_else(|| TokenizeError::invalid(TOKENS, "is not an array of strings"))?;
        let tokens = strings
            .enumerate()
            .map(|(i, s)| {
                std::str::from_utf8(s)
                    .map_err(|_| TokenizeError::invalid(TOKENS, format!("token {i} is not UTF-8")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if tokens.is_empty() {
            return Err(TokenizeError::invalid(TOKENS, "is empty"));
        }
        if u32::try_from(tokens.len()).is_err() {
            return Err(TokenizeError::invalid(TOKENS, "has more than 2³² tokens"));
        }

        let array = meta.array(TOKEN_TYPE)?.ok_or_else(|| {
            TokenizeError::invalid(
                TOKEN_TYPE,
                "is missing, so added and special tokens are unknown",
            )
        })?;
        let types = array
            .iter()
            .enumerate()
            .map(|(i, v)| {
                integer(v).and_then(TokenType::from_i64).ok_or_else(|| {
                    TokenizeError::invalid(TOKEN_TYPE, format!("token {i} has type {v:?}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if types.len() != tokens.len() {
            return Err(TokenizeError::invalid(
                TOKEN_TYPE,
                format!("has {} entries for {} tokens", types.len(), tokens.len()),
            ));
        }

        let mut ids = HashMap::with_capacity(tokens.len());
        let mut added = HashMap::new();
        for (id, (&text, &ty)) in (0u32..).zip(tokens.iter().zip(&types)) {
            if ty == TokenType::Unused {
                continue;
            }
            let in_model = sentencepiece || ty.added().is_none();
            let seen = if in_model {
                ids.insert(text, id)
            } else {
                added.insert(text, id)
            };
            if let Some(first) = seen {
                return Err(TokenizeError::invalid(
                    TOKENS,
                    format!("tokens {first} and {id} are both {text:?}"),
                ));
            }
        }
        Ok(Self { tokens, types, ids })
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    /// The ID of an added token spelled `text`.
    fn added_id(&self, text: &str) -> Option<u32> {
        (0u32..)
            .zip(self.tokens.iter().zip(&self.types))
            .find(|(_, (t, ty))| **t == text && ty.added().is_some())
            .map(|(id, _)| id)
    }

    /// `tokenizer.ggml.merges`, each checked against the vocabulary.
    fn listed_merges(&self, meta: &'a MetaView) -> Result<Vec<(&'a str, &'a str)>, TokenizeError> {
        let array = meta
            .array(MERGES)?
            .ok_or_else(|| TokenizeError::invalid(MERGES, "is missing"))?;
        let strings = array
            .strings()
            .ok_or_else(|| TokenizeError::invalid(MERGES, "is not an array of strings"))?;
        strings
            .enumerate()
            .map(|(i, s)| {
                let bad = |why: &str| TokenizeError::invalid(MERGES, format!("merge {i} {why}"));
                let s = std::str::from_utf8(s).map_err(|_| bad("is not UTF-8"))?;
                let (l, r) = s
                    .split_once(' ')
                    .ok_or_else(|| bad("is not two tokens separated by a space"))?;
                if r.contains(' ') {
                    return Err(bad("has more than two parts"));
                }
                let joined = format!("{l}{r}");
                for part in [l, r, joined.as_str()] {
                    if !self.ids.contains_key(part) {
                        return Err(bad(&format!("names {part:?}, which is not a token")));
                    }
                }
                Ok((l, r))
            })
            .collect()
    }

    /// The merges derived from `tokenizer.ggml.scores` (module documentation).
    fn score_merges(&self, meta: &'a MetaView) -> Result<Vec<(&'a str, &'a str)>, TokenizeError> {
        let array = meta
            .array(SCORES)?
            .ok_or_else(|| TokenizeError::invalid(SCORES, "is missing"))?;
        let scores = array
            .iter()
            .enumerate()
            .map(|(i, v)| match v {
                ValueRef::F32(s) if s.is_finite() => Ok(s),
                _ => Err(TokenizeError::invalid(
                    SCORES,
                    format!("token {i} has score {v:?}, expected a finite f32"),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if scores.len() != self.len() {
            return Err(TokenizeError::invalid(
                SCORES,
                format!("has {} entries for {} tokens", scores.len(), self.len()),
            ));
        }

        let mut order: Vec<u32> = self.ids.values().copied().collect();
        order.sort_unstable_by(|&a, &b| {
            scores[a as usize]
                .total_cmp(&scores[b as usize])
                .reverse()
                .then(a.cmp(&b))
        });
        let mut merges = Vec::new();
        for id in order {
            let text = self.tokens[id as usize];
            let mut local: Vec<(u32, u32, &str, &str)> = text
                .char_indices()
                .skip(1)
                .filter_map(|(at, _)| {
                    let (l, r) = text.split_at(at);
                    Some((*self.ids.get(l)?, *self.ids.get(r)?, l, r))
                })
                .collect();
            local.sort_unstable_by_key(|&(l, r, _, _)| (l, r));
            merges.extend(local.into_iter().map(|(_, _, l, r)| (l, r)));
        }
        Ok(merges)
    }
}

/// A metadata integer of any width.
fn integer(v: ValueRef<'_>) -> Option<i64> {
    match v {
        ValueRef::U8(x) => Some(x.into()),
        ValueRef::I8(x) => Some(x.into()),
        ValueRef::U16(x) => Some(x.into()),
        ValueRef::I16(x) => Some(x.into()),
        ValueRef::U32(x) => Some(x.into()),
        ValueRef::I32(x) => Some(x.into()),
        ValueRef::U64(x) => i64::try_from(x).ok(),
        ValueRef::I64(x) => Some(x),
        _ => None,
    }
}

/// The special-token IDs (module documentation), each naming a token that has text.
fn specials(meta: &MetaView, vocab: &Vocab<'_>) -> Result<SpecialTokens, TokenizeError> {
    let id = |key: &str| -> Result<Option<u32>, TokenizeError> {
        let Some(id) = meta.u32(key)? else {
            return Ok(None);
        };
        match vocab.types.get(id as usize) {
            None => Err(TokenizeError::invalid(
                key,
                format!("is {id}, past the last of {} tokens", vocab.len()),
            )),
            Some(TokenType::Unused) => Err(TokenizeError::invalid(
                key,
                format!("is {id}, an unused token"),
            )),
            Some(_) => Ok(Some(id)),
        }
    };

    let mut fim = None;
    for keys in FIM_KEYS {
        let [p, m, s] = keys.map(id);
        match (p?, m?, s?) {
            (None, None, None) => {}
            (Some(p), Some(m), Some(s)) => {
                fim = Some((p, m, s));
                break;
            }
            _ => {
                return Err(TokenizeError::invalid(
                    keys[0],
                    "FIM prefix, middle and suffix are not all set",
                ));
            }
        }
    }
    let fim = fim.or_else(|| {
        FIM_SPELLINGS.iter().find_map(|(p, m, s)| {
            Some((vocab.added_id(p)?, vocab.added_id(m)?, vocab.added_id(s)?))
        })
    });

    let eos = id(EOS_ID)?;
    let end_of_turn = [eos, id(EOT_ID)?, id(EOM_ID)?]
        .into_iter()
        .flatten()
        .collect();
    Ok(SpecialTokens {
        bos: id(BOS_ID)?,
        eos,
        end_of_turn,
        pad: id(PAD_ID)?,
        unk: id(UNK_ID)?,
        fim_prefix: fim.map(|f| f.0),
        fim_middle: fim.map(|f| f.1),
        fim_suffix: fim.map(|f| f.2),
    })
}

/// The `tokenizer.json` bytes (module documentation).
fn serialize(
    vocab: &Vocab<'_>,
    merges: &[(&str, &str)],
    kind: Kind,
    specials: &SpecialTokens,
    add_bos: bool,
    add_eos: bool,
) -> Vec<u8> {
    let added_tokens: Vec<Value> = (0u32..)
        .zip(vocab.tokens.iter().zip(&vocab.types))
        .filter_map(|(id, (text, ty))| {
            let special = ty.added()?;
            Some(json!({
                "id": id,
                "content": text,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": special,
            }))
        })
        .collect();

    // The model's vocabulary in ID order; `ids` holds exactly its tokens.
    let mut model_vocab = Map::new();
    for (id, text) in (0u32..).zip(&vocab.tokens) {
        if vocab.ids.get(text) == Some(&id) {
            model_vocab.insert((*text).to_owned(), id.into());
        }
    }
    let merges: Vec<Value> = merges.iter().map(|(l, r)| json!([l, r])).collect();
    let unk = specials
        .unk
        .filter(|&id| vocab.ids.get(vocab.tokens[id as usize]) == Some(&id))
        .map(|id| vocab.tokens[id as usize]);

    let template = template_processing(vocab, specials, add_bos, add_eos);
    let byte_level = json!({
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": false,
    });
    let (normalizer, pre_tokenizer, post_processor, decoder, model) = match kind {
        Kind::ByteLevel(pre) => (
            if pre.nfc {
                json!({ "type": "NFC" })
            } else {
                Value::Null
            },
            json!({
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": { "Regex": pre.regex },
                        "behavior": "Isolated",
                        "invert": false,
                    },
                    byte_level.clone(),
                ],
            }),
            match template {
                Some(t) => json!({ "type": "Sequence", "processors": [byte_level.clone(), t] }),
                None => byte_level.clone(),
            },
            byte_level.clone(),
            json!({
                "type": "BPE",
                "dropout": null,
                "unk_token": unk,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": pre.ignore_merges,
                "vocab": model_vocab,
                "merges": merges,
            }),
        ),
        Kind::SentencePiece { space_prefix } => {
            let mut normalizers = Vec::new();
            if space_prefix {
                normalizers.push(json!({ "type": "Prepend", "prepend": SPACE_MARKER }));
            }
            normalizers.push(json!({
                "type": "Replace",
                "pattern": { "String": " " },
                "content": SPACE_MARKER,
            }));
            let mut decoders = vec![
                json!({
                    "type": "Replace",
                    "pattern": { "String": SPACE_MARKER },
                    "content": " ",
                }),
                json!({ "type": "ByteFallback" }),
                json!({ "type": "Fuse" }),
            ];
            if space_prefix {
                decoders.push(json!({ "type": "Strip", "content": " ", "start": 1, "stop": 0 }));
            }
            (
                json!({ "type": "Sequence", "normalizers": normalizers }),
                Value::Null,
                template.unwrap_or(Value::Null),
                json!({ "type": "Sequence", "decoders": decoders }),
                json!({
                    "type": "BPE",
                    "dropout": null,
                    "unk_token": unk,
                    "continuing_subword_prefix": null,
                    "end_of_word_suffix": null,
                    "fuse_unk": true,
                    "byte_fallback": true,
                    "ignore_merges": false,
                    "vocab": model_vocab,
                    "merges": merges,
                }),
            )
        }
    };

    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": post_processor,
        "decoder": decoder,
        "model": model,
    });
    serde_json::to_vec(&tokenizer).expect("a JSON value always serializes")
}

/// The post-processor that adds BOS and EOS as the flags say; `None` when neither is added.
fn template_processing(
    vocab: &Vocab<'_>,
    specials: &SpecialTokens,
    add_bos: bool,
    add_eos: bool,
) -> Option<Value> {
    let bos = specials.bos.filter(|_| add_bos);
    let eos = specials.eos.filter(|_| add_eos);
    if bos.is_none() && eos.is_none() {
        return None;
    }
    let text = |id: u32| vocab.tokens[id as usize];
    let special =
        |id: u32, type_id: u32| json!({ "SpecialToken": { "id": text(id), "type_id": type_id } });
    let sequence =
        |name: &str, type_id: u32| json!({ "Sequence": { "id": name, "type_id": type_id } });
    let around = |name: &str, type_id: u32| {
        bos.map(|id| special(id, type_id))
            .into_iter()
            .chain([sequence(name, type_id)])
            .chain(eos.map(|id| special(id, type_id)))
    };
    let single: Vec<Value> = around("A", 0).collect();
    let pair: Vec<Value> = around("A", 0).chain(around("B", 1)).collect();
    let mut special_tokens = Map::new();
    for id in bos.into_iter().chain(eos) {
        special_tokens.insert(
            text(id).to_owned(),
            json!({ "id": text(id), "ids": [id], "tokens": [text(id)] }),
        );
    }
    Some(json!({
        "type": "TemplateProcessing",
        "single": single,
        "pair": pair,
        "special_tokens": special_tokens,
    }))
}
