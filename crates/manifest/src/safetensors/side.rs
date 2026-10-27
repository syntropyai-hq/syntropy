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
    /// Every role, each at its [`position`](Self::position); a build-time check below holds
    /// the two together. A new role takes the next position and is added here.
    pub const ALL: [SourceRole; 7] = [
        SourceRole::Weights,
        SourceRole::WeightIndex,
        SourceRole::Config,
        SourceRole::GenerationConfig,
        SourceRole::Tokenizer,
        SourceRole::TokenizerConfig,
        SourceRole::SpecialTokensMap,
    ];

    /// The role's stable name, as records such as the store's local index write it. The match
    /// names every role, so a role added later fails to compile here until it has a name, and
    /// a name is never changed or reused.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            SourceRole::Weights => "weights",
            SourceRole::WeightIndex => "weight_index",
            SourceRole::Config => "config",
            SourceRole::GenerationConfig => "generation_config",
            SourceRole::Tokenizer => "tokenizer",
            SourceRole::TokenizerConfig => "tokenizer_config",
            SourceRole::SpecialTokensMap => "special_tokens_map",
        }
    }

    /// The role named `name`, the inverse of [`name`](Self::name). It does not go through
    /// [`ALL`](Self::ALL), so a role left out of `ALL` still reads back.
    #[must_use]
    pub fn from_name(name: &str) -> Option<SourceRole> {
        Some(match name {
            "weights" => SourceRole::Weights,
            "weight_index" => SourceRole::WeightIndex,
            "config" => SourceRole::Config,
            "generation_config" => SourceRole::GenerationConfig,
            "tokenizer" => SourceRole::Tokenizer,
            "tokenizer_config" => SourceRole::TokenizerConfig,
            "special_tokens_map" => SourceRole::SpecialTokensMap,
            _ => return None,
        })
    }

    /// Where the role sits in [`ALL`](Self::ALL). The match names every role, so a new role
    /// fails to compile here until it has a position.
    #[must_use]
    pub const fn position(self) -> usize {
        match self {
            SourceRole::Weights => 0,
            SourceRole::WeightIndex => 1,
            SourceRole::Config => 2,
            SourceRole::GenerationConfig => 3,
            SourceRole::Tokenizer => 4,
            SourceRole::TokenizerConfig => 5,
            SourceRole::SpecialTokensMap => 6,
        }
    }

    /// Whether RFC-0041 §1's `ModelId` hashes a file of this role. It is defined by
    /// [`role_code`](crate::identity::role_code), so the two cannot disagree.
    #[must_use]
    pub fn in_model_id(self) -> bool {
        crate::identity::role_code(self).is_some()
    }
}

// The build fails unless `ALL` lists each role at its position, so `ALL` cannot skip, repeat or
// reorder a role. A new role must first get a position (`position`'s match is exhaustive); one
// given the next position and left out of `ALL` is not caught here, which only `ALL` growing
// with it fixes. `from_name` does not read `ALL`, so such a role still reads back from a record.
const _: () = {
    let mut i = 0;
    while i < SourceRole::ALL.len() {
        assert!(
            SourceRole::ALL[i].position() == i,
            "SourceRole::ALL must list each role at its position"
        );
        i += 1;
    }
};

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
