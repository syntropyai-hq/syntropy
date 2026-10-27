// SPDX-License-Identifier: MIT OR Apache-2.0

//! `FileDigest` and `ModelId` (RFC-0041 §1; ADR-0411 decisions 1 and 2; COMP-E10-T06).
//!
//! - [`FileDigest`] is the SHA-256 of a whole source file. For a Hub file it is the object ID the
//!   revision's file listing publishes, read with [`FileDigest::from_hex`] before any download.
//!   For a local file, or a file the listing gives no digest, [`FileDigest::of_source`] hashes
//!   every byte through a [`RangeSource`]. The two agree for the same file, so the same model
//!   gets one ID from either.
//! - [`model_id`] is the v1 formula. Its inputs are the family's frozen tag (RFC-0043 §5.1) and,
//!   for each source file the census reads in a role the formula covers, the role, the digest
//!   and the length. Nothing Syntropy derives is an input: not `HyperParams`, `LayerSpec`,
//!   `QuantScheme` or any encoding of them. So a release that adds a field to those types, or
//!   normalises them differently, leaves the ID of the same files unchanged. File names are no
//!   input either, so renaming a local file keeps its ID.
//!
//! **The bytes hashed.** RFC-0041 §1 gives the formula's parts, and this is how they are written:
//!
//! ```text
//! BLAKE3(
//!     "syntropy/model-id/v1"                       -- the 20 ASCII bytes, no terminator
//!   ‖ u64 LE byte length of the tag ‖ the tag      -- UTF-8, as FROZEN_TAGS lists it
//!   ‖ for each file, sorted by (role code, digest):
//!         role code (1 byte) ‖ digest (32 bytes) ‖ u64 LE length
//! )
//! ```
//!
//! The role codes are fixed: 1 weights (a GGUF file or split, or a safetensors file or shard),
//! 2 the weight index JSON, 3 `config.json`, 4 `generation_config.json`, and 5, reserved, the
//! quantization config file no reader names yet. A file in another role, such as a tokenizer
//! file, is read by the census but is not hashed (it makes `TokenizerId`). The tag is
//! length-prefixed and each file record is 41 bytes, so no two inputs write the same bytes.

use std::fmt;

use sha2::{Digest as _, Sha256};
use syntropy_core::{Blake3, ModelId};

use crate::range::{RangeError, RangeSource};
use crate::safetensors::side::SourceRole;

/// The domain tag of the v1 `ModelId`.
pub const MODEL_ID_DOMAIN: &[u8] = b"syntropy/model-id/v1";

/// How many bytes [`FileDigest::of_source`] reads at a time.
const READ_SIZE: usize = 1 << 20;

/// The SHA-256 of a whole source file (RFC-0041 §1).
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct FileDigest(pub [u8; 32]);

impl FileDigest {
    /// The SHA-256 of `bytes`.
    #[must_use]
    pub fn of_bytes(bytes: &[u8]) -> FileDigest {
        FileDigest(Sha256::digest(bytes).into())
    }

    /// The SHA-256 of every byte of `source`, read in order, a megabyte at a time.
    ///
    /// # Errors
    ///
    /// The source's error for a read that fails.
    pub fn of_source<S: RangeSource>(source: &S) -> Result<FileDigest, RangeError> {
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; READ_SIZE];
        let len = source.len();
        let mut at = 0u64;
        while at < len {
            let n = usize::try_from((len - at).min(READ_SIZE as u64)).expect("at most READ_SIZE");
            source.read_at(at, &mut buf[..n])?;
            hasher.update(&buf[..n]);
            at += n as u64;
        }
        Ok(FileDigest(hasher.finalize().into()))
    }

    /// The digest the Hub publishes as a file's object ID: 64 hex digits, either case.
    #[must_use]
    pub fn from_hex(hex: &str) -> Option<FileDigest> {
        if hex.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok()?;
        }
        Some(FileDigest(out))
    }

    /// The digest as 64 lower-case hex digits.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Display for FileDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for FileDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FileDigest({})", self.to_hex())
    }
}

/// One source file as [`model_id`] sees it: its role, its digest and its length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceFile {
    /// What the file is to the census. A GGUF file or split is [`SourceRole::Weights`].
    pub role: SourceRole,
    /// Its SHA-256.
    pub digest: FileDigest,
    /// Its length in bytes.
    pub len: u64,
}

/// The role code [`model_id`] writes for `role`, or `None` for a role it does not hash.
#[must_use]
pub fn role_code(role: SourceRole) -> Option<u8> {
    match role {
        SourceRole::Weights => Some(1),
        SourceRole::WeightIndex => Some(2),
        SourceRole::Config => Some(3),
        SourceRole::GenerationConfig => Some(4),
        // 5 is the quantization config file's, which no reader names yet.
        _ => None,
    }
}

/// Why a `ModelId` cannot be computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdentityError {
    /// The family tag is empty.
    NoFamilyTag,
    /// No source file holds weights.
    NoWeights,
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFamilyTag => f.write_str("a model ID needs its family's tag"),
            Self::NoWeights => f.write_str("a model ID needs at least one weights file"),
        }
    }
}

impl std::error::Error for IdentityError {}

/// The v1 `ModelId` of a model of the family tagged `family_tag` whose census reads `files`
/// (RFC-0041 §1). The files may come in any order; those in a role the formula does not hash,
/// such as the tokenizer's, are left out.
///
/// # Errors
///
/// [`IdentityError`] for an empty tag, or no weights file.
pub fn model_id(family_tag: &str, files: &[SourceFile]) -> Result<ModelId, IdentityError> {
    if family_tag.is_empty() {
        return Err(IdentityError::NoFamilyTag);
    }
    let mut records: Vec<(u8, FileDigest, u64)> = files
        .iter()
        .filter_map(|f| role_code(f.role).map(|code| (code, f.digest, f.len)))
        .collect();
    if !records.iter().any(|(code, _, _)| *code == 1) {
        return Err(IdentityError::NoWeights);
    }
    records.sort_unstable_by_key(|r| (r.0, r.1));
    let mut hasher = blake3::Hasher::new();
    hasher.update(MODEL_ID_DOMAIN);
    hasher.update(&(family_tag.len() as u64).to_le_bytes());
    hasher.update(family_tag.as_bytes());
    for (code, digest, len) in records {
        hasher.update(&[code]);
        hasher.update(&digest.0);
        hasher.update(&len.to_le_bytes());
    }
    Ok(ModelId(Blake3(*hasher.finalize().as_bytes())))
}

#[cfg(test)]
mod tests;
