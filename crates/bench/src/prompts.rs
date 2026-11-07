// SPDX-License-Identifier: MIT OR Apache-2.0

//! `bench-prompts-v1`: the benchmark prompt set and exact-length prompt construction
//! (RFC-0061 §3.1; QUAL-E12-T05).
//!
//! The set's items are drawn from the parity corpus sources (`testdata/corpus/SOURCES.toml`) by
//! `bench/prompts/select.py`, which writes the manifest `bench/prompts/bench-prompts-v1.toml`:
//! each item's category (`chat`, `code`, `long`, `agent`), source, locator, licence and the
//! SHA-256 of its text. The text itself is not in git; an [`ItemStore`] supplies it and
//! [`DirStore`] refuses text whose hash differs. [`PromptSet::check`] checks every item's
//! licence record against its source.
//!
//! [`Builder`] builds token sequences for one tokenizer:
//!
//! - [`Builder::prompt`]: exactly `n` tokens of a category. Each item of the category, in
//!   manifest order and followed by the set's separator, is tokenized on its own, and the token
//!   streams are concatenated, cycling through the items again when they run out; the prompt is
//!   the first `n` tokens, so it ends on a token boundary. The same tokenizer and key give the
//!   same tokens, and a shorter prompt is a prefix of a longer one.
//! - [`Builder::nonce`]: the `Cold` scenarios' per-repetition nonce, `nonce_tokens` tokens of the
//!   `chat` stream at an offset set by the repetition (RFC-0061 §3.1).
//! - [`Builder::agent_turns`]: an `agent` item replayed turn by turn, where turn `k + 1`'s prompt
//!   is turn `k`'s with the next turn's tokens appended: the shared-prefix pattern of vision S4,
//!   which the `Warm` scenarios run.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// A prompt category (RFC-0061 §3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    /// Conversations.
    Chat,
    /// Source files.
    Code,
    /// Long documents.
    Long,
    /// Multi-turn conversations replayed with a shared prefix.
    Agent,
}

impl Category {
    /// Every category.
    pub const ALL: [Category; 4] = [
        Category::Chat,
        Category::Code,
        Category::Long,
        Category::Agent,
    ];

    /// The corpus category whose sources serve it (RFC-0060 §5.2): `agent` items are
    /// conversations.
    pub fn corpus_category(self) -> &'static str {
        match self {
            Category::Chat | Category::Agent => "chat",
            Category::Code => "code",
            Category::Long => "long",
        }
    }
}

/// One item of the set.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Item {
    /// Its stable key.
    pub id: String,
    /// Its category.
    pub category: Category,
    /// The `SOURCES.toml` source it is drawn from.
    pub source: String,
    /// Its licence, an SPDX expression: the source's.
    pub licence: String,
    /// Where in the source it is: the file, its git blob, and a dialog ID for conversations.
    pub locator: BTreeMap<String, String>,
    /// The SHA-256 of its text, lowercase hex.
    pub sha256: String,
    /// Its text's length in bytes.
    pub bytes: usize,
    /// For `agent` items: each turn's length in bytes, in order; they add up to `bytes`.
    #[serde(default)]
    pub turn_bytes: Option<Vec<usize>>,
}

/// The prompt set manifest.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptSet {
    /// The manifest's schema version: 1.
    pub schema: u32,
    /// The set's name and version, `bench-prompts-v1`.
    pub set: String,
    /// The sources file the items cite, relative to the repository root.
    pub sources: String,
    /// What follows each item when a category's items are joined.
    pub separator: String,
    /// The `Cold` nonce's length in tokens.
    pub nonce_tokens: usize,
    /// The items, in their order within each category.
    #[serde(default, rename = "item")]
    pub items: Vec<Item>,
}

/// Why a prompt cannot be built, or a manifest is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PromptError {
    /// The manifest or the sources file does not parse.
    Parse(String),
    /// The manifest breaks a rule [`PromptSet::check`] enforces.
    Invalid(Vec<String>),
    /// An item's text is missing or differs from the manifest.
    Item {
        /// The item.
        id: String,
        /// What is wrong.
        reason: String,
    },
    /// A category has no item.
    Empty(Category),
    /// The tokenizer failed, or gave no token for an item.
    Tokenizer(String),
}

impl fmt::Display for PromptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PromptError::Parse(e) => write!(f, "cannot read the prompt set: {e}"),
            PromptError::Invalid(problems) => {
                write!(f, "the prompt set is invalid: {}", problems.join("; "))
            }
            PromptError::Item { id, reason } => write!(f, "item {id}: {reason}"),
            PromptError::Empty(c) => write!(f, "the prompt set has no {c:?} item"),
            PromptError::Tokenizer(e) => write!(f, "tokenizer: {e}"),
        }
    }
}

impl std::error::Error for PromptError {}

/// What [`PromptSet::check`] reads from `SOURCES.toml`.
#[derive(Clone, Debug, Deserialize)]
pub struct Sources {
    policy: Policy,
    #[serde(rename = "source")]
    sources: Vec<Source>,
}

#[derive(Clone, Debug, Deserialize)]
struct Policy {
    allowed: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct Source {
    id: String,
    categories: BTreeMap<String, String>,
    licence: String,
    #[serde(default)]
    attribution_file: Option<String>,
}

impl Sources {
    /// Parses `SOURCES.toml`.
    ///
    /// # Errors
    ///
    /// [`PromptError::Parse`] when it does not parse.
    pub fn parse(text: &str) -> Result<Sources, PromptError> {
        toml::from_str(text).map_err(|e| PromptError::Parse(format!("SOURCES.toml: {e}")))
    }
}

impl PromptSet {
    /// Parses a manifest.
    ///
    /// # Errors
    ///
    /// [`PromptError::Parse`] when it does not parse.
    pub fn parse(text: &str) -> Result<PromptSet, PromptError> {
        toml::from_str(text).map_err(|e| PromptError::Parse(e.to_string()))
    }

    /// The items of `category`, in manifest order.
    pub fn items(&self, category: Category) -> impl Iterator<Item = &Item> {
        self.items.iter().filter(move |i| i.category == category)
    }

    /// Checks the manifest: schema 1; unique item IDs; each item's source exists and serves its
    /// category; its licence is the source's, and every licence the expression names is on the
    /// corpus policy's allow list; a CC-BY source has its attribution file under `root`; the
    /// hash is 64 lowercase hex digits; an `agent` item has turn lengths that add up to its
    /// length, and no other item has them.
    ///
    /// # Errors
    ///
    /// [`PromptError::Invalid`] with every problem found.
    pub fn check(&self, sources: &Sources, root: &Path) -> Result<(), PromptError> {
        let mut problems = Vec::new();
        if self.schema != 1 {
            problems.push(format!("schema {} is not 1", self.schema));
        }
        if self.nonce_tokens == 0 {
            problems.push("nonce_tokens is 0".to_owned());
        }
        let mut seen = std::collections::BTreeSet::new();
        for item in &self.items {
            let id = &item.id;
            if !seen.insert(id.as_str()) {
                problems.push(format!("{id}: duplicate ID"));
            }
            let Some(source) = sources.sources.iter().find(|s| s.id == item.source) else {
                problems.push(format!(
                    "{id}: source {} is not in SOURCES.toml",
                    item.source
                ));
                continue;
            };
            if !source
                .categories
                .contains_key(item.category.corpus_category())
            {
                problems.push(format!(
                    "{id}: source {} does not serve `{}`",
                    source.id,
                    item.category.corpus_category()
                ));
            }
            if item.licence != source.licence {
                problems.push(format!(
                    "{id}: licence {} is not the source's, {}",
                    item.licence, source.licence
                ));
            }
            for term in licence_terms(&item.licence) {
                if !sources.policy.allowed.iter().any(|a| a == term) {
                    problems.push(format!("{id}: licence {term} is not allowed"));
                }
            }
            if item.licence.contains("CC-BY") {
                match &source.attribution_file {
                    Some(f) if root.join(f).is_file() => {}
                    Some(f) => problems.push(format!("{id}: attribution file {f} is missing")),
                    None => problems.push(format!("{id}: a CC-BY source has no attribution file")),
                }
            }
            if item.sha256.len() != 64
                || !item
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                problems.push(format!("{id}: sha256 is not 64 lowercase hex digits"));
            }
            match (&item.turn_bytes, item.category) {
                (Some(turns), Category::Agent) => {
                    if turns.is_empty() || turns.iter().sum::<usize>() != item.bytes {
                        problems.push(format!("{id}: turn_bytes do not add up to bytes"));
                    }
                }
                (None, Category::Agent) => problems.push(format!("{id}: no turn_bytes")),
                (Some(_), _) => problems.push(format!("{id}: turn_bytes on a non-agent item")),
                (None, _) => {}
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(PromptError::Invalid(problems))
        }
    }
}

/// The licence identifiers an SPDX expression of `OR` and `AND` names.
fn licence_terms(expr: &str) -> impl Iterator<Item = &str> {
    expr.split(|c: char| c.is_whitespace() || c == '(' || c == ')')
        .filter(|t| !t.is_empty() && *t != "OR" && *t != "AND")
}

/// Supplies an item's text.
pub trait ItemStore {
    /// The text of `item`, checked against its hash and length.
    ///
    /// # Errors
    ///
    /// [`PromptError::Item`] when it is missing or differs.
    fn text(&self, item: &Item) -> Result<String, PromptError>;
}

/// Checks `text` against `item`'s hash and length.
///
/// # Errors
///
/// [`PromptError::Item`] when they differ.
pub fn verify(item: &Item, text: &str) -> Result<(), PromptError> {
    let fail = |reason: String| {
        Err(PromptError::Item {
            id: item.id.clone(),
            reason,
        })
    };
    if text.len() != item.bytes {
        return fail(format!(
            "{} bytes, the manifest says {}",
            text.len(),
            item.bytes
        ));
    }
    let digest = Sha256::digest(text.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    if hex != item.sha256 {
        return fail(format!("SHA-256 {hex}, the manifest says {}", item.sha256));
    }
    Ok(())
}

/// Item text in a directory, one file per item named by its SHA-256: what `select.py --out`
/// writes.
#[derive(Clone, Debug)]
pub struct DirStore(pub PathBuf);

impl ItemStore for DirStore {
    fn text(&self, item: &Item) -> Result<String, PromptError> {
        let path = self.0.join(&item.sha256);
        let bytes = std::fs::read(&path).map_err(|e| PromptError::Item {
            id: item.id.clone(),
            reason: format!("{}: {e}", path.display()),
        })?;
        let text = String::from_utf8(bytes).map_err(|_| PromptError::Item {
            id: item.id.clone(),
            reason: "not UTF-8".to_owned(),
        })?;
        verify(item, &text)?;
        Ok(text)
    }
}

/// A tokenizer, as prompt construction uses it: text to IDs, with no special tokens added.
pub trait Tokenize {
    /// The IDs of `text`.
    ///
    /// # Errors
    ///
    /// The engine's message.
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
}

impl Tokenize for syntropy_text::tokenizer::Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        // The engine's method (this trait's `encode` would shadow its `encode`).
        self.encode_fast(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }
}

/// Builds prompts for one tokenizer (module docs).
pub struct Builder<'a> {
    set: &'a PromptSet,
    store: &'a dyn ItemStore,
    tokenizer: &'a dyn Tokenize,
}

impl fmt::Debug for Builder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Builder")
            .field("set", &self.set.set)
            .finish_non_exhaustive()
    }
}

impl<'a> Builder<'a> {
    /// A builder over `set`, reading item text from `store`.
    pub fn new(
        set: &'a PromptSet,
        store: &'a dyn ItemStore,
        tokenizer: &'a dyn Tokenize,
    ) -> Builder<'a> {
        Builder {
            set,
            store,
            tokenizer,
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, PromptError> {
        self.tokenizer.encode(text).map_err(PromptError::Tokenizer)
    }

    /// Exactly `n` tokens of `category` (module docs).
    ///
    /// # Errors
    ///
    /// [`PromptError::Empty`] for a category with no item; [`PromptError::Item`] for missing or
    /// changed text; [`PromptError::Tokenizer`] when the engine fails or every item encodes to
    /// no token.
    pub fn prompt(&self, category: Category, n: usize) -> Result<Vec<u32>, PromptError> {
        let items: Vec<&Item> = self.set.items(category).collect();
        if items.is_empty() {
            return Err(PromptError::Empty(category));
        }
        let mut out = Vec::with_capacity(n);
        let mut streams: Vec<Option<Vec<u32>>> = vec![None; items.len()];
        let mut i = 0;
        while out.len() < n {
            let at = i % items.len();
            if streams[at].is_none() {
                let text = self.store.text(items[at])?;
                streams[at] = Some(self.encode(&(text + &self.set.separator))?);
            }
            let stream = streams[at].as_deref().unwrap_or_default();
            out.extend_from_slice(&stream[..stream.len().min(n - out.len())]);
            i += 1;
            if i % items.len() == 0 && out.is_empty() {
                return Err(PromptError::Tokenizer(format!(
                    "every {category:?} item encodes to no token"
                )));
            }
        }
        Ok(out)
    }

    /// The nonce for repetition `rep`: `nonce_tokens` tokens of the `chat` stream, starting at
    /// `rep · nonce_tokens` (module docs).
    ///
    /// # Errors
    ///
    /// As [`prompt`](Self::prompt).
    pub fn nonce(&self, rep: u32) -> Result<Vec<u32>, PromptError> {
        let k = self.set.nonce_tokens;
        let start = rep as usize * k;
        let mut stream = self.prompt(Category::Chat, start + k)?;
        Ok(stream.split_off(start))
    }

    /// The `agent` item `id` replayed turn by turn: element `k` is the prompt of turn `k`, the
    /// tokens of turns `0..=k`, each turn tokenized on its own (module docs).
    ///
    /// # Errors
    ///
    /// [`PromptError::Item`] for an unknown or malformed item or missing text;
    /// [`PromptError::Tokenizer`] when the engine fails.
    pub fn agent_turns(&self, id: &str) -> Result<Vec<Vec<u32>>, PromptError> {
        let fail = |reason: &str| PromptError::Item {
            id: id.to_owned(),
            reason: reason.to_owned(),
        };
        let item = self
            .set
            .items(Category::Agent)
            .find(|i| i.id == id)
            .ok_or_else(|| fail("no such agent item"))?;
        let turns = item
            .turn_bytes
            .as_ref()
            .ok_or_else(|| fail("no turn_bytes"))?;
        let text = self.store.text(item)?;
        let mut at = 0;
        let mut prompt = Vec::new();
        let mut out = Vec::with_capacity(turns.len());
        for &len in turns {
            let turn = text
                .get(at..at + len)
                .ok_or_else(|| fail("a turn does not end on a character boundary"))?;
            at += len;
            prompt.extend(self.encode(turn)?);
            out.push(prompt.clone());
        }
        Ok(out)
    }
}
