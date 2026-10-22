// SPDX-License-Identifier: MIT OR Apache-2.0

//! The files of a safetensors checkpoint as source files (RFC-0041 §1, §2.2 "Side files").
//!
//! Each file the census reads is a source file with its own `FileDigest` and a role.
//! [`classify`] gives a file name its role; [`SourceRole::in_model_id`] says whether
//! RFC-0041 §1's `ModelId` formula hashes it. The tokenizer files are source files but stay
//! out of `ModelId` (they make `TokenizerId`). Reading and hashing the side files belong to
//! the census and the store (COMP-E10-T06, COMP-E10-T07); this module only names them.

use super::index::INDEX_FILE;

/// What a source file is to the census.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceRole {
    /// A safetensors file or shard.
    Weights,
    /// `model.safetensors.index.json`.
    WeightIndex,
    /// `config.json`: the architecture, the hyperparameters and `quantization_config`.
    Config,
    /// `generation_config.json`.
    GenerationConfig,
    /// `tokenizer.json`.
    Tokenizer,
    /// `tokenizer_config.json`, which holds the chat template.
    TokenizerConfig,
    /// `special_tokens_map.json`.
    SpecialTokensMap,
}

impl SourceRole {
    /// Whether RFC-0041 §1's `ModelId` hashes a file of this role.
    #[must_use]
    pub fn in_model_id(self) -> bool {
        matches!(
            self,
            SourceRole::Weights
                | SourceRole::WeightIndex
                | SourceRole::Config
                | SourceRole::GenerationConfig
        )
    }
}

/// The side files RFC-0041 §2.2 names, with their roles.
pub const SIDE_FILES: [(&str, SourceRole); 5] = [
    ("config.json", SourceRole::Config),
    ("generation_config.json", SourceRole::GenerationConfig),
    ("tokenizer.json", SourceRole::Tokenizer),
    ("tokenizer_config.json", SourceRole::TokenizerConfig),
    ("special_tokens_map.json", SourceRole::SpecialTokensMap),
];

/// The role of the file named `name` (a file name, not a path), or `None` when the reader
/// does not read it.
#[must_use]
pub fn classify(name: &str) -> Option<SourceRole> {
    if name == INDEX_FILE {
        return Some(SourceRole::WeightIndex);
    }
    if let Some((_, role)) = SIDE_FILES.iter().find(|(n, _)| *n == name) {
        return Some(*role);
    }
    (name.len() > ".safetensors".len() && name.ends_with(".safetensors"))
        .then_some(SourceRole::Weights)
}

/// One source file of a checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceFile {
    /// The file name.
    pub name: String,
    /// Its role.
    pub role: SourceRole,
    /// Its length in bytes.
    pub len: u64,
}

/// The source files among `files` (name, length), sorted by role then name; the others are
/// left out.
#[must_use]
pub fn source_files<'a>(files: impl IntoIterator<Item = (&'a str, u64)>) -> Vec<SourceFile> {
    let mut out: Vec<SourceFile> = files
        .into_iter()
        .filter_map(|(name, len)| {
            classify(name).map(|role| SourceFile {
                name: name.to_owned(),
                role,
                len,
            })
        })
        .collect();
    out.sort_by(|a, b| (a.role, &a.name).cmp(&(b.role, &b.name)));
    out
}
