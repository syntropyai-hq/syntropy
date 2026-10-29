// SPDX-License-Identifier: MIT OR Apache-2.0

//! A node's shard and the key of its pack (RFC-0001 §5.6; RFC-0041 §1; COMP-E10-T11).
//!
//! [`ShardSpec`] and [`PackKey`] travel on the command channel, in `FetchShard` and
//! `PackCommitted` (RFC-0001 §3.5), so they are plain data here rather than in `manifest`, which
//! re-exports them (RFC-0001 §18.28). [`LayoutVersion`] moved here from `kernels` with them;
//! `kernels` re-exports it, so `LayoutVersion::SOURCE` still resolves there.
//!
//! **`ShardSpec::digest`.** RFC-0041 §1 makes it BLAKE3 over `model` and the sorted group list.
//! `core` links no hasher, so the crate that links BLAKE3 passes it in, as for
//! `InstancePolicy::digest_with` (RFC-0054 §4.5, note of 2026-10-22).
//! `syntropy_manifest::identity::shard_spec` is that caller. The bytes hashed are:
//!
//! ```text
//! "syntropy/shard/v1"                       -- the 17 ASCII bytes, no terminator
//! ‖ ModelId                                 -- 32 bytes
//! ‖ u64 LE count of distinct groups
//! ‖ each TensorGroupId as u32 LE, ascending, each once
//! ```
//!
//! The groups are sorted and deduplicated before they are hashed, so the digest is a function of
//! the model and the set of groups alone. The node, the chunk list and the plan epoch are not
//! inputs. So an epoch change that leaves a node's groups unchanged keeps the digest, and with it
//! the [`PackKey`], and the pack is reused rather than rebuilt (RFC-0001 §5.6).

use serde::{Deserialize, Serialize};

use crate::ids::{Blake3, ChunkId, ModelId, NodeId, TensorGroupId};
use crate::resource::DeviceClass;

/// The domain tag that starts the [`ShardSpec::digest`] preimage.
pub const SHARD_DIGEST_DOMAIN: &[u8] = b"syntropy/shard/v1";

/// A weight byte layout's number (RFC-0001 §5.6, §5.7; RFC-0040 §3.1, §5.4; RFC-0041 §1).
///
/// It has two uses, both from RFC-0001:
/// - in `OpKey::layout` and a pack extent's `layout`, the layout an op reads:
///   [`LayoutVersion::SOURCE`] for the source format's own layout, or a provider's repack ID
///   (RFC-0040 §5.4);
/// - in [`PackKey::layout`], the release's layout-table version, one number bumped whenever any
///   provider's repack output changes (RFC-0041 §1).
///
/// Every repack is lossless, and a provider that offers a weight-bound op for a scheme also
/// offers a `SOURCE` variant of it, so `SOURCE` is always a valid choice (RFC-0040 §5.4). Owner:
/// RFC-0040.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct LayoutVersion(pub u16);

impl LayoutVersion {
    /// The source format's layout, unchanged (RFC-0040 §5.4, ID 0).
    pub const SOURCE: Self = Self(0);
}

/// The tensor byte ranges one node must hold locally to run its stages (RFC-0001 §5.6). Owner:
/// RFC-0041.
///
/// Build it with [`ShardSpec::new_with`], which sorts and deduplicates `groups` and computes
/// `digest`. A value decoded from the command channel is taken as sent; the store that acts on
/// a `FetchShard` can check it with [`ShardSpec::digest_is_consistent`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardSpec {
    /// The model.
    pub model: ModelId,
    /// The node that holds the shard.
    pub node: NodeId,
    /// The tensor groups the node's stages read, ascending, each once.
    pub groups: Vec<TensorGroupId>,
    /// The BLAKE3-addressed chunks covering the groups.
    pub chunks: Vec<ChunkId>,
    /// BLAKE3 over `model` and the sorted group list (the module docs' bytes).
    pub digest: Blake3,
}

impl ShardSpec {
    /// The shard of `groups` of `model` on `node`, covered by `chunks`. `groups` may be in any
    /// order and repeat; the spec holds them ascending, each once. `hash` is BLAKE3, which `core`
    /// does not link.
    pub fn new_with(
        model: ModelId,
        node: NodeId,
        mut groups: Vec<TensorGroupId>,
        chunks: Vec<ChunkId>,
        hash: impl FnOnce(&[u8]) -> Blake3,
    ) -> ShardSpec {
        groups.sort_unstable();
        groups.dedup();
        let digest = hash(&digest_preimage(model, &groups));
        ShardSpec {
            model,
            node,
            groups,
            chunks,
            digest,
        }
    }

    /// The bytes [`ShardSpec::digest`] hashes for `model` and `groups`, which may be in any order
    /// and repeat (the module docs' layout).
    #[must_use]
    pub fn digest_preimage(model: ModelId, groups: &[TensorGroupId]) -> Vec<u8> {
        let mut sorted = groups.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        digest_preimage(model, &sorted)
    }

    /// Whether `digest` is `hash` over `model` and `groups`, and `groups` is ascending with no
    /// repeat, as [`ShardSpec::new_with`] builds it.
    pub fn digest_is_consistent(&self, hash: impl FnOnce(&[u8]) -> Blake3) -> bool {
        self.groups.windows(2).all(|pair| pair[0] < pair[1])
            && hash(&digest_preimage(self.model, &self.groups)) == self.digest
    }
}

/// The preimage of `groups`, already ascending and each once.
fn digest_preimage(model: ModelId, groups: &[TensorGroupId]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(SHARD_DIGEST_DOMAIN.len() + 32 + 8 + 4 * groups.len());
    bytes.extend_from_slice(SHARD_DIGEST_DOMAIN);
    bytes.extend_from_slice(&model.0.0);
    bytes.extend_from_slice(&(groups.len() as u64).to_le_bytes());
    for group in groups {
        bytes.extend_from_slice(&group.0.to_le_bytes());
    }
    bytes
}

/// Identifies a pack in a node's local cache (RFC-0001 §5.6; RFC-0041 §1). Owner: RFC-0041.
///
/// It holds the shard digest, not the plan epoch, so an epoch change that leaves a node's shard
/// unchanged reuses the pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PackKey {
    /// The model.
    pub model: ModelId,
    /// The device class the pack's layouts are for.
    pub class: DeviceClass,
    /// The shard's [`ShardSpec::digest`].
    pub shard: Blake3,
    /// The release's layout-table version (RFC-0041 §1). Each extent carries its own layout ID
    /// (RFC-0041 §5.2), so one pack may hold several layouts.
    pub layout: LayoutVersion,
}

impl PackKey {
    /// The key of `shard`'s pack for devices of `class`, under layout-table version `layout`.
    #[must_use]
    pub fn for_shard(shard: &ShardSpec, class: DeviceClass, layout: LayoutVersion) -> PackKey {
        PackKey {
            model: shard.model,
            class,
            shard: shard.digest,
            layout,
        }
    }
}

#[cfg(test)]
mod tests;
