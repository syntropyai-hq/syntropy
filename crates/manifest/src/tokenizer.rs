// SPDX-License-Identifier: MIT OR Apache-2.0

//! `TokenizerSpec` for `tokenizer.json` sources, and `TokenizerId` (RFC-0043 §8.1; RFC-0041 §1;
//! ADR-0411 decision 5; COMP-E17-T01).
//!
//! [`TokenizerSpec::from_sources`] builds the spec of a safetensors checkpoint from its
//! tokenizer side files ([`TokenizerSources`]):
//!
//! - **`model`** is the shipped `tokenizer.json`, byte for byte, so request-time code loads it
//!   with no conversion (RFC-0043 §8.2).
//! - **Special tokens** ([`SpecialTokens`]). BOS, EOS, pad and unknown come from
//!   `tokenizer_config.json` (`bos_token`, `eos_token`, `pad_token`, `unk_token`), each a
//!   string or an added-token object, resolved to an ID through `tokenizer.json`'s
//!   `added_tokens` and then its vocabulary. The end-of-turn set is EOS together with every ID
//!   in `generation_config.json`'s `eos_token_id` (an integer or a list). The FIM prefix,
//!   middle and suffix tokens are the added tokens spelled as in [`FIM_SPELLINGS`], whether or
//!   not `tokenizer.json` marks them special: some checkpoints (the Qwen3-MoE repositories
//!   among them) ship their FIM tokens as added but not special. No token is looked up by a
//!   family or model name.
//! - **`add_bos` / `add_eos`** come from `tokenizer_config.json`'s `add_bos_token` and
//!   `add_eos_token` where it sets them. Otherwise they are read from `tokenizer.json`'s
//!   post-processor: a template-processing post-processor whose single-sequence template
//!   starts (ends) with a special token adds BOS (EOS).
//! - **The template.** The shipped chat template, verbatim: a separate `chat_template.jinja`
//!   when the checkpoint ships one, else `tokenizer_config.json`'s `chat_template` (a string,
//!   or the entry named `default` of a list of named templates).
//! - **[`TokenizerId`]** is BLAKE3 over `model`, the specials, `add_bos`, `add_eos` and the
//!   template in use ([`TokenizerSpec::template_in_use`]), so it changes when any of them
//!   does (RFC-0043 §8.5). Nothing it covers depends on the order something was listed in:
//!   the specials are single IDs and one sorted, deduplicated set (`end_of_turn`); `model` is
//!   the shipped file's bytes, hashed as they are, the same bytes `FileDigest` commits to, so
//!   its added tokens and maps are covered through the file, never re-encoded from a parse.
//!
//! The fields are private and read through accessors, so the ID can never fall out of step
//! with what it covers: the one way to change the template in use,
//! [`TokenizerSpec::with_template_source`], recomputes it.
//!
//! **Sessions.** [`TokenizerSpec::id`] is the value RFC-0022's saved sessions record in their
//! metadata. It is not part of a session's key: a session resumed under a different
//! `TokenizerId` warns and proceeds (RFC-0043 open question 6, answered; OQ-MF-14). Nothing
//! here keys on it either. `ModelId` does not include it (ADR-0411 decision 5).

use std::fmt;

use serde::{Deserialize, Serialize};
use syntropy_core::{Blake3, ErrorClass};

use crate::safetensors::json::Json;

#[cfg(test)]
mod tests;

/// The tokenizer side files' names (RFC-0041 §2.2).
pub const TOKENIZER_JSON: &str = "tokenizer.json";
/// See [`TOKENIZER_JSON`].
pub const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
/// See [`TOKENIZER_JSON`].
pub const GENERATION_CONFIG_JSON: &str = "generation_config.json";
/// A chat template shipped as its own file, which takes precedence over
/// `tokenizer_config.json`'s `chat_template`.
pub const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";

/// The domain tag that starts the `TokenizerId` preimage.
const ID_TAG: &[u8] = b"syntropy/tokenizer-id/v1";

/// The token spellings read as FIM prefix, middle and suffix. Matched against every added
/// token, marked special or not; a set is taken only when all three of its spellings are
/// added tokens.
pub const FIM_SPELLINGS: [(&str, &str, &str); 3] = [
    ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"),
    ("<fim_prefix>", "<fim_middle>", "<fim_suffix>"),
    ("<PRE>", "<MID>", "<SUF>"),
];

/// Content identity of a tokenizer and the chat template in use (RFC-0041 §1; ADR-0411
/// decision 5). Recorded in session metadata, never a session key (OQ-MF-14).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenizerId(pub Blake3);

impl fmt::Display for TokenizerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// The tokenizer's special token IDs (RFC-0043 §8.1).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Beginning of sequence.
    pub bos: Option<u32>,
    /// End of sequence.
    pub eos: Option<u32>,
    /// Every end-of-turn token, sorted, each once: [`TokenizerSpec::new`] sorts and dedupes
    /// it, so the set, not the order it was listed in, is what `TokenizerId` covers. Holds
    /// `eos` when the spec was read from its sources and EOS is set.
    pub end_of_turn: Vec<u32>,
    /// Padding.
    pub pad: Option<u32>,
    /// Unknown token.
    pub unk: Option<u32>,
    /// FIM prefix.
    pub fim_prefix: Option<u32>,
    /// FIM middle.
    pub fim_middle: Option<u32>,
    /// FIM suffix.
    pub fim_suffix: Option<u32>,
}

/// Where the template in use comes from (RFC-0043 §8.5).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemplateSource {
    /// The template the checkpoint ships, [`TokenizerSpec::chat_template`].
    Shipped,
    /// The family's default template, used only when the checkpoint ships none
    /// (COMP-E17-T05 applies the precedence).
    FamilyDefault(String),
    /// The user's `models.<name>.chat_template` override (RFC-0043 §10).
    UserOverride(String),
    /// No template: none shipped and none supplied.
    None,
}

/// The tokenizer side files of one checkpoint, as bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenizerSources<'a> {
    /// `tokenizer.json`. Required.
    pub tokenizer_json: &'a [u8],
    /// `tokenizer_config.json`, when shipped.
    pub tokenizer_config: Option<&'a [u8]>,
    /// `generation_config.json`, when shipped.
    pub generation_config: Option<&'a [u8]>,
    /// `chat_template.jinja`, when shipped.
    pub chat_template_file: Option<&'a [u8]>,
}

/// A side file that cannot be read as a tokenizer source. Class `Integrity`; RFC-0043 §10 and
/// RFC-0041 §10 name no code for it yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerSpecError {
    /// The file.
    pub file: &'static str,
    /// What is wrong.
    pub reason: String,
}

impl TokenizerSpecError {
    fn new(file: &'static str, reason: impl Into<String>) -> Self {
        Self {
            file,
            reason: reason.into(),
        }
    }

    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        ErrorClass::Integrity
    }
}

impl fmt::Display for TokenizerSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.file, self.reason)
    }
}

impl std::error::Error for TokenizerSpecError {}

/// A model's tokenizer data (RFC-0043 §8.1). It serializes as its parts, without `id`, and
/// deserializes through [`TokenizerSpec::new`], which computes `id` again, so a stored spec can
/// never carry an ID that does not match what it covers (the manifest stores it, RFC-0041 §3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "TokenizerSpecParts", from = "TokenizerSpecParts")]
pub struct TokenizerSpec {
    model: Vec<u8>,
    specials: SpecialTokens,
    add_bos: bool,
    add_eos: bool,
    chat_template: Option<String>,
    template_source: TemplateSource,
    id: TokenizerId,
}

/// [`TokenizerSpec`]'s stored form: its parts, without the ID.
#[derive(Clone, Serialize, Deserialize)]
struct TokenizerSpecParts {
    model: Vec<u8>,
    specials: SpecialTokens,
    add_bos: bool,
    add_eos: bool,
    chat_template: Option<String>,
    template_source: TemplateSource,
}

impl From<TokenizerSpec> for TokenizerSpecParts {
    fn from(spec: TokenizerSpec) -> Self {
        Self {
            model: spec.model,
            specials: spec.specials,
            add_bos: spec.add_bos,
            add_eos: spec.add_eos,
            chat_template: spec.chat_template,
            template_source: spec.template_source,
        }
    }
}

impl From<TokenizerSpecParts> for TokenizerSpec {
    fn from(parts: TokenizerSpecParts) -> Self {
        Self::new(
            parts.model,
            parts.specials,
            parts.add_bos,
            parts.add_eos,
            parts.chat_template,
            parts.template_source,
        )
    }
}

impl TokenizerSpec {
    /// A spec from its parts, with the ID computed. The template in use is the one
    /// `template_source` names. The end-of-turn list is sorted and deduplicated first, so the
    /// same set listed in any order gives the same spec and the same ID.
    #[must_use]
    pub fn new(
        model: Vec<u8>,
        mut specials: SpecialTokens,
        add_bos: bool,
        add_eos: bool,
        chat_template: Option<String>,
        template_source: TemplateSource,
    ) -> Self {
        specials.end_of_turn.sort_unstable();
        specials.end_of_turn.dedup();
        let mut spec = Self {
            model,
            specials,
            add_bos,
            add_eos,
            chat_template,
            template_source,
            id: TokenizerId(Blake3::default()),
        };
        spec.id = spec.compute_id();
        spec
    }

    /// The spec of a `tokenizer.json` source (module documentation). The template in use is
    /// the shipped one, or none.
    ///
    /// # Errors
    ///
    /// [`TokenizerSpecError`] when a file is not valid JSON, has a field of the wrong kind, or
    /// names a special token the tokenizer does not hold.
    pub fn from_sources(src: TokenizerSources<'_>) -> Result<Self, TokenizerSpecError> {
        let tok = object(TOKENIZER_JSON, src.tokenizer_json)?;
        let cfg = src
            .tokenizer_config
            .map(|b| object(TOKENIZER_CONFIG_JSON, b))
            .transpose()?
            .unwrap_or_default();
        let gen_cfg = src
            .generation_config
            .map(|b| object(GENERATION_CONFIG_JSON, b))
            .transpose()?
            .unwrap_or_default();
        let vocab = Vocab::new(&tok)?;

        let named = |key: &str| -> Result<Option<u32>, TokenizerSpecError> {
            match token_content(get(&cfg, key), key)? {
                Some(content) => vocab.id(&content).map(Some).ok_or_else(|| {
                    TokenizerSpecError::new(
                        TOKENIZER_CONFIG_JSON,
                        format!("{key} {content:?} is not a token of {TOKENIZER_JSON}"),
                    )
                }),
                None => Ok(None),
            }
        };
        let bos = named("bos_token")?;
        let eos = named("eos_token")?;
        let pad = named("pad_token")?;
        let unk = named("unk_token")?;

        let mut end_of_turn: Vec<u32> = eos.into_iter().collect();
        match get(&gen_cfg, "eos_token_id") {
            None | Some(Json::Null) => {}
            Some(Json::Uint(n)) => end_of_turn.push(id_u32(*n)?),
            Some(Json::Array(items)) => {
                for item in items {
                    match item {
                        Json::Uint(n) => end_of_turn.push(id_u32(*n)?),
                        other => {
                            return Err(TokenizerSpecError::new(
                                GENERATION_CONFIG_JSON,
                                format!("eos_token_id holds {}, not a token ID", other.kind()),
                            ));
                        }
                    }
                }
            }
            Some(other) => {
                return Err(TokenizerSpecError::new(
                    GENERATION_CONFIG_JSON,
                    format!("eos_token_id is {}", other.kind()),
                ));
            }
        }
        end_of_turn.sort_unstable();
        end_of_turn.dedup();
        if let Some(bad) = end_of_turn.iter().find(|id| !vocab.holds(**id)) {
            return Err(TokenizerSpecError::new(
                GENERATION_CONFIG_JSON,
                format!("eos_token_id {bad} is not a token of {TOKENIZER_JSON}"),
            ));
        }

        let (fim_prefix, fim_middle, fim_suffix) = FIM_SPELLINGS
            .iter()
            .find_map(|(p, m, s)| {
                Some((vocab.added_id(p)?, vocab.added_id(m)?, vocab.added_id(s)?))
            })
            .map_or((None, None, None), |(p, m, s)| (Some(p), Some(m), Some(s)));

        let (pp_bos, pp_eos) = post_processor_adds(get(&tok, "post_processor"));
        let add_bos = flag(&cfg, "add_bos_token")?.unwrap_or(pp_bos);
        let add_eos = flag(&cfg, "add_eos_token")?.unwrap_or(pp_eos);

        let chat_template = match src.chat_template_file {
            Some(bytes) => Some(String::from_utf8(bytes.to_vec()).map_err(|_| {
                TokenizerSpecError::new(CHAT_TEMPLATE_FILE, "the template is not UTF-8")
            })?),
            None => config_template(get(&cfg, "chat_template"))?,
        };
        let template_source = if chat_template.is_some() {
            TemplateSource::Shipped
        } else {
            TemplateSource::None
        };
        Ok(Self::new(
            src.tokenizer_json.to_vec(),
            SpecialTokens {
                bos,
                eos,
                end_of_turn,
                pad,
                unk,
                fim_prefix,
                fim_middle,
                fim_suffix,
            },
            add_bos,
            add_eos,
            chat_template,
            template_source,
        ))
    }

    /// The serialized tokenizer model (`tokenizer.json` form).
    #[must_use]
    pub fn model(&self) -> &[u8] {
        &self.model
    }

    /// The special token IDs.
    #[must_use]
    pub fn specials(&self) -> &SpecialTokens {
        &self.specials
    }

    /// Whether encoding adds BOS.
    #[must_use]
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Whether encoding adds EOS.
    #[must_use]
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    /// The shipped chat template, verbatim.
    #[must_use]
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Where the template in use comes from.
    #[must_use]
    pub fn template_source(&self) -> &TemplateSource {
        &self.template_source
    }

    /// The template in use: the shipped one, the family default or the user's override, as
    /// [`template_source`](Self::template_source) says.
    #[must_use]
    pub fn template_in_use(&self) -> Option<&str> {
        match &self.template_source {
            TemplateSource::Shipped => self.chat_template.as_deref(),
            TemplateSource::FamilyDefault(t) | TemplateSource::UserOverride(t) => Some(t),
            TemplateSource::None => None,
        }
    }

    /// The `TokenizerId`: what RFC-0022's saved sessions record in their metadata, without
    /// keying on it (a resume under another ID warns and proceeds; OQ-MF-14).
    #[must_use]
    pub fn id(&self) -> TokenizerId {
        self.id
    }

    /// The spec with another template in use, and its ID recomputed. The shipped template is
    /// kept as it is.
    #[must_use]
    pub fn with_template_source(mut self, source: TemplateSource) -> Self {
        self.template_source = source;
        self.id = self.compute_id();
        self
    }

    /// BLAKE3 over a length-prefixed encoding of everything the ID covers.
    fn compute_id(&self) -> TokenizerId {
        let mut h = blake3::Hasher::new();
        let bytes = |h: &mut blake3::Hasher, b: &[u8]| {
            h.update(&(b.len() as u64).to_le_bytes());
            h.update(b);
        };
        let opt = |h: &mut blake3::Hasher, v: Option<u32>| match v {
            Some(id) => {
                h.update(&[1]);
                h.update(&id.to_le_bytes());
            }
            None => {
                h.update(&[0]);
            }
        };
        bytes(&mut h, ID_TAG);
        bytes(&mut h, &self.model);
        let s = &self.specials;
        for v in [
            s.bos,
            s.eos,
            s.pad,
            s.unk,
            s.fim_prefix,
            s.fim_middle,
            s.fim_suffix,
        ] {
            opt(&mut h, v);
        }
        h.update(&(s.end_of_turn.len() as u64).to_le_bytes());
        for id in &s.end_of_turn {
            h.update(&id.to_le_bytes());
        }
        h.update(&[u8::from(self.add_bos), u8::from(self.add_eos)]);
        match self.template_in_use() {
            Some(t) => {
                h.update(&[1]);
                bytes(&mut h, t.as_bytes());
            }
            None => {
                h.update(&[0]);
            }
        }
        TokenizerId(Blake3(*h.finalize().as_bytes()))
    }
}

// ---- Reading the side files ------------------------------------------------------------------

type Object = Vec<(String, Json)>;

fn object(file: &'static str, bytes: &[u8]) -> Result<Object, TokenizerSpecError> {
    match Json::parse(bytes) {
        Ok(Json::Object(o)) => Ok(o),
        Ok(other) => Err(TokenizerSpecError::new(
            file,
            format!("the file is {}, not a JSON object", other.kind()),
        )),
        Err(e) => Err(TokenizerSpecError::new(
            file,
            format!("not valid JSON: {e}"),
        )),
    }
}

fn get<'a>(o: &'a Object, key: &str) -> Option<&'a Json> {
    o.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn id_u32(n: u64) -> Result<u32, TokenizerSpecError> {
    u32::try_from(n).map_err(|_| {
        TokenizerSpecError::new(GENERATION_CONFIG_JSON, format!("token ID {n} passes u32"))
    })
}

/// A special token field of `tokenizer_config.json`: a string, an added-token object with a
/// `content` string, or null.
fn token_content(v: Option<&Json>, key: &str) -> Result<Option<String>, TokenizerSpecError> {
    match v {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Str(s)) => Ok(Some(s.clone())),
        Some(Json::Object(o)) => match get(o, "content") {
            Some(Json::Str(s)) => Ok(Some(s.clone())),
            _ => Err(TokenizerSpecError::new(
                TOKENIZER_CONFIG_JSON,
                format!("{key} is an object without a content string"),
            )),
        },
        Some(other) => Err(TokenizerSpecError::new(
            TOKENIZER_CONFIG_JSON,
            format!("{key} is {}", other.kind()),
        )),
    }
}

fn flag(cfg: &Object, key: &str) -> Result<Option<bool>, TokenizerSpecError> {
    match get(cfg, key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Bool(b)) => Ok(Some(*b)),
        Some(other) => Err(TokenizerSpecError::new(
            TOKENIZER_CONFIG_JSON,
            format!("{key} is {}, not a boolean", other.kind()),
        )),
    }
}

/// `chat_template`: a string, or a list of `{name, template}` from which `default` is taken.
fn config_template(v: Option<&Json>) -> Result<Option<String>, TokenizerSpecError> {
    let bad = |what: String| TokenizerSpecError::new(TOKENIZER_CONFIG_JSON, what);
    match v {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Str(s)) => Ok(Some(s.clone())),
        Some(Json::Array(items)) => {
            for item in items {
                let Json::Object(o) = item else {
                    return Err(bad(format!("a chat_template entry is {}", item.kind())));
                };
                match (get(o, "name"), get(o, "template")) {
                    (Some(Json::Str(n)), Some(Json::Str(t))) => {
                        if n == "default" {
                            return Ok(Some(t.clone()));
                        }
                    }
                    _ => return Err(bad("a chat_template entry lacks name or template".into())),
                }
            }
            Ok(None)
        }
        Some(other) => Err(bad(format!("chat_template is {}", other.kind()))),
    }
}

/// Whether `tokenizer.json`'s post-processor adds a special token before (BOS) and after
/// (EOS) a single sequence. Only a `TemplateProcessing` post-processor, alone or inside a
/// `Sequence`, adds tokens this way.
fn post_processor_adds(pp: Option<&Json>) -> (bool, bool) {
    let Some(Json::Object(o)) = pp else {
        return (false, false);
    };
    match get(o, "type") {
        Some(Json::Str(t)) if t == "TemplateProcessing" => {
            let Some(Json::Array(single)) = get(o, "single") else {
                return (false, false);
            };
            let is_special = |item: Option<&Json>| matches!(item, Some(Json::Object(e)) if get(e, "SpecialToken").is_some());
            (is_special(single.first()), is_special(single.last()))
        }
        Some(Json::Str(t)) if t == "Sequence" => match get(o, "processors") {
            Some(Json::Array(ps)) => ps.iter().fold((false, false), |(b, e), p| {
                let (pb, pe) = post_processor_adds(Some(p));
                (b || pb, e || pe)
            }),
            _ => (false, false),
        },
        _ => (false, false),
    }
}

/// The token IDs `tokenizer.json` defines: its added tokens and its model's vocabulary.
struct Vocab<'a> {
    /// (content, id).
    added: Vec<(&'a str, u32)>,
    /// The model's vocabulary, when it is a JSON object of string to ID (BPE, WordPiece).
    vocab: Option<&'a Object>,
    /// A Unigram vocabulary: a list of `[piece, score]`, the ID being the position.
    unigram: Option<&'a [Json]>,
}

impl<'a> Vocab<'a> {
    fn new(tok: &'a Object) -> Result<Self, TokenizerSpecError> {
        let bad = |what: String| TokenizerSpecError::new(TOKENIZER_JSON, what);
        let mut added = Vec::new();
        match get(tok, "added_tokens") {
            None | Some(Json::Null) => {}
            Some(Json::Array(items)) => {
                for item in items {
                    let Json::Object(o) = item else {
                        return Err(bad(format!("an added token is {}", item.kind())));
                    };
                    let (Some(Json::Uint(id)), Some(Json::Str(content))) =
                        (get(o, "id"), get(o, "content"))
                    else {
                        return Err(bad("an added token lacks id or content".into()));
                    };
                    let id = u32::try_from(*id)
                        .map_err(|_| bad(format!("added token id {id} passes u32")))?;
                    added.push((content.as_str(), id));
                }
            }
            Some(other) => return Err(bad(format!("added_tokens is {}", other.kind()))),
        }
        let (mut vocab, mut unigram) = (None, None);
        if let Some(Json::Object(model)) = get(tok, "model") {
            match get(model, "vocab") {
                Some(Json::Object(v)) => vocab = Some(v),
                Some(Json::Array(v)) => unigram = Some(v.as_slice()),
                _ => {}
            }
        }
        Ok(Self {
            added,
            vocab,
            unigram,
        })
    }

    /// The ID of `content`: an added token first, then the vocabulary.
    fn id(&self, content: &str) -> Option<u32> {
        if let Some(id) = self.added_id(content) {
            return Some(id);
        }
        if let Some(v) = self.vocab {
            return match get(v, content) {
                Some(Json::Uint(id)) => u32::try_from(*id).ok(),
                _ => None,
            };
        }
        let pos = self.unigram?.iter().position(|e| match e {
            Json::Array(pair) => matches!(pair.first(), Some(Json::Str(p)) if p == content),
            _ => false,
        })?;
        u32::try_from(pos).ok()
    }

    /// The ID of the added token `content`, marked special or not.
    fn added_id(&self, content: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|(c, _)| *c == content)
            .map(|(_, id)| *id)
    }

    /// Whether `id` is a token: an added token's ID, or inside the vocabulary.
    fn holds(&self, id: u32) -> bool {
        if self.added.iter().any(|(_, a)| *a == id) {
            return true;
        }
        if let Some(v) = self.vocab {
            return v
                .iter()
                .any(|(_, j)| matches!(j, Json::Uint(n) if *n == u64::from(id)));
        }
        self.unigram.is_some_and(|u| (id as usize) < u.len())
    }
}
