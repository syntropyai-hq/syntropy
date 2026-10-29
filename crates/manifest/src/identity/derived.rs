// SPDX-License-Identifier: MIT OR Apache-2.0

//! The identities derived from chunk IDs and the plan: `TensorRecord::hash`, `ShardSpec::digest`
//! and `ExtentKey` (RFC-0041 §1; ADR-0411 decision 3; COMP-E10-T11).
//!
//! Each is BLAKE3 over a domain tag and fixed-width fields, every integer little-endian. Any
//! change to the bytes below is a new identity version, with a new tag, not an edit of v1.
//!
//! **`TensorRecord::hash`** ([`tensor_hash`]) is a Merkle root over the tensor's `ChunkId`s, in
//! file order, so a tensor's identity follows from the chunk IDs alone and never needs its bytes
//! read again (ADR-0411 decision 3). The tree is a binary Merkle tree in the shape RFC 6962 §2.1
//! gives, with domain-separated leaves and nodes, under the tag `T = "syntropy/tensor-hash/v1"`
//! (23 ASCII bytes):
//!
//! ```text
//! leaf(id)        = BLAKE3(T ‖ 0x00 ‖ id)                -- id: the 32-byte ChunkId
//! node(l, r)      = BLAKE3(T ‖ 0x01 ‖ l ‖ r)
//! root([])        = BLAKE3(T ‖ 0x02)                     -- a zero-length tensor covers no chunk
//! root([d])       = leaf(d)
//! root(d[0..n])   = node(root(d[0..k]), root(d[k..n]))   -- n > 1, k the largest power of 2 below n
//! ```
//!
//! The split at the largest power of two gives every leaf count one tree, with no duplicated
//! node, so an odd count needs no padding and two different ID lists never give one root by
//! repeating a last element. The 0x00 and 0x01 prefixes keep a leaf from passing as a node.
//! [`ChunkTable::tensor_hash`] finds a tensor's chunks in a table: every tensor start and
//! auxiliary-extent start is a chunk boundary (RFC-0041 §6.1), so the chunks of an extent are
//! exactly those that start inside it. The hash is `None` until every one is committed
//! (RFC-0041 §6.3). `TensorRecord` itself comes with the census builder (COMP-E10-T07); its
//! `hash` field holds this value.
//!
//! **`ShardSpec::digest`** ([`shard_spec`], [`shard_digest`]) is `core`'s
//! [`ShardSpec::digest_preimage`] under BLAKE3: `"syntropy/shard/v1"`, the `ModelId`, the count
//! of distinct groups and each group ascending.
//!
//! **`ExtentKey`** ([`ExtentKey::of`]) names one stored copy of one tensor group in one layout,
//! so a pack that needs a group already stored in that layout references it instead of building
//! it again (RFC-0041 §5.2, §5.3):
//!
//! ```text
//! BLAKE3(
//!     "syntropy/extent/v1"                  -- the 18 ASCII bytes
//!   ‖ u64 LE count of tensor hashes
//!   ‖ each TensorRecord::hash, in the group's order   -- 32 bytes each
//!   ‖ u16 LE layout ID                      -- the extent's own layout (RFC-0040 §5.4)
//!   ‖ u16 LE layout-table version           -- PackKey::layout (RFC-0041 §1)
//! )
//! ```
//!
//! The order of the tensor hashes is the order of the components in the extent, so it is kept
//! as given: the same tensors laid out in another order are other bytes.

use std::fmt;
use std::ops::Range;

use syntropy_core::{Blake3, ChunkId, LayoutVersion, ModelId, NodeId, ShardSpec, TensorGroupId};

use crate::chunk::ChunkTable;

/// The domain tag of the v1 `TensorRecord::hash` tree.
pub const TENSOR_HASH_DOMAIN: &[u8] = b"syntropy/tensor-hash/v1";

/// The domain tag of the v1 `ExtentKey`.
pub const EXTENT_KEY_DOMAIN: &[u8] = b"syntropy/extent/v1";

const LEAF: u8 = 0x00;
const NODE: u8 = 0x01;
const EMPTY: u8 = 0x02;

/// BLAKE3 of `bytes`, as `core`'s digest type: the hash `core`'s `*_with` functions take.
#[must_use]
pub fn blake3_digest(bytes: &[u8]) -> Blake3 {
    Blake3(*blake3::hash(bytes).as_bytes())
}

/// The Merkle root over a tensor's chunk IDs, in file order (the module docs' tree): the value of
/// `TensorRecord::hash` once every chunk is committed.
#[must_use]
pub fn tensor_hash(ids: &[ChunkId]) -> Blake3 {
    match ids {
        [] => tagged(&[&[EMPTY]]),
        [id] => tagged(&[&[LEAF], &id.0.0]),
        _ => {
            // The largest power of two strictly below `len`.
            let k = 1 << (usize::BITS - 1 - (ids.len() - 1).leading_zeros());
            let left = tensor_hash(&ids[..k]);
            let right = tensor_hash(&ids[k..]);
            tagged(&[&[NODE], &left.0, &right.0])
        }
    }
}

/// BLAKE3 over the tensor-hash tag and `parts`.
fn tagged(parts: &[&[u8]]) -> Blake3 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(TENSOR_HASH_DOMAIN);
    for part in parts {
        hasher.update(part);
    }
    Blake3(*hasher.finalize().as_bytes())
}

/// Why a table cannot give a tensor's hash: the extents passed do not fit its chunks, so they
/// come from another file or another header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorHashError {
    /// A non-empty extent does not start at a chunk boundary.
    NotAChunkStart {
        /// The extent's start.
        offset: u64,
    },
    /// An extent ends past the end of the file.
    PastEnd {
        /// The extent's end.
        end: u64,
        /// The file's length.
        file_len: u64,
    },
}

impl TensorHashError {
    /// The `Formats` code RFC-0041 §10 gives it: 6 (`HeaderInvalid`), as for `ChunkError`. The
    /// tensor's extents and the table's boundaries both come from the header, so a mismatch
    /// means one of them is not this file's.
    #[must_use]
    pub fn code(&self) -> u16 {
        6
    }
}

impl fmt::Display for TensorHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAChunkStart { offset } => {
                write!(
                    f,
                    "a tensor extent starts at {offset}, which is no chunk's start"
                )
            }
            Self::PastEnd { end, file_len } => {
                write!(
                    f,
                    "a tensor extent ends at {end}, past the file's end at {file_len}"
                )
            }
        }
    }
}

impl std::error::Error for TensorHashError {}

impl ChunkTable {
    /// The `TensorRecord::hash` of a tensor whose bytes lie in `extents` of this file, in order:
    /// its source extent, then each auxiliary extent (`TensorRecord::source` and `aux`).
    /// `Ok(None)` while any chunk covering them is uncommitted (RFC-0041 §6.3). An empty extent
    /// covers no chunk.
    ///
    /// # Errors
    ///
    /// [`TensorHashError`] when a non-empty extent does not start at a chunk boundary, or ends
    /// past the file's end.
    pub fn tensor_hash(&self, extents: &[Range<u64>]) -> Result<Option<Blake3>, TensorHashError> {
        let mut ids = Vec::new();
        let mut complete = true;
        for extent in extents {
            if extent.end > self.file_len {
                return Err(TensorHashError::PastEnd {
                    end: extent.end,
                    file_len: self.file_len,
                });
            }
            if extent.is_empty() {
                continue;
            }
            let first = self
                .records
                .binary_search_by_key(&extent.start, |record| record.chunk.offset)
                .map_err(|_| TensorHashError::NotAChunkStart {
                    offset: extent.start,
                })?;
            for record in self.records[first..]
                .iter()
                .take_while(|record| record.chunk.offset < extent.end)
            {
                match record.committed {
                    Some((id, _)) => ids.push(ChunkId(id)),
                    None => complete = false,
                }
            }
        }
        Ok(complete.then(|| tensor_hash(&ids)))
    }
}

/// The shard of `groups` of `model` on `node`, covered by `chunks`, with its BLAKE3 digest
/// (`core`'s [`ShardSpec::new_with`] under BLAKE3).
#[must_use]
pub fn shard_spec(
    model: ModelId,
    node: NodeId,
    groups: Vec<TensorGroupId>,
    chunks: Vec<ChunkId>,
) -> ShardSpec {
    ShardSpec::new_with(model, node, groups, chunks, blake3_digest)
}

/// `ShardSpec::digest` of `groups`, in any order and with repeats, of `model`.
#[must_use]
pub fn shard_digest(model: ModelId, groups: &[TensorGroupId]) -> Blake3 {
    blake3_digest(&ShardSpec::digest_preimage(model, groups))
}

/// The identity of one stored copy of a tensor group in one layout (RFC-0041 §1, §5.2). Extents
/// with equal keys hold equal bytes, so a pack reuses one rather than building it again.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct ExtentKey(pub Blake3);

impl ExtentKey {
    /// The key of a group whose tensors have `tensor_hashes`, in the extent's component order,
    /// stored in layout `layout` under layout-table version `version` (the module docs' bytes).
    #[must_use]
    pub fn of(
        tensor_hashes: &[Blake3],
        layout: LayoutVersion,
        version: LayoutVersion,
    ) -> ExtentKey {
        let mut hasher = blake3::Hasher::new();
        hasher.update(EXTENT_KEY_DOMAIN);
        hasher.update(&(tensor_hashes.len() as u64).to_le_bytes());
        for hash in tensor_hashes {
            hasher.update(&hash.0);
        }
        hasher.update(&layout.0.to_le_bytes());
        hasher.update(&version.0.to_le_bytes());
        ExtentKey(Blake3(*hasher.finalize().as_bytes()))
    }
}

#[cfg(test)]
mod tests;
