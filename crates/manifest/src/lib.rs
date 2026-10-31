// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF and safetensors readers, the tensor census, the model manifest, the KV formula and the pack
//! sidecar format.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0041. It holds the `RangeSource` every reader
//! parses through, with the local-file and recorded-fixture sources and the bounded,
//! coalescing header-read routine ([`range`], RFC-0041 §2), the GGUF and safetensors readers,
//! `config.json` read as metadata ([`config`]), `TokenizerSpec`, and the first census and
//! hyperparameter types ([`census`], [`hparams`]) that the family modules in `graph` produce
//! (RFC-0043 §4, §5.2); expert slicing ([`slice`](mod@slice), RFC-0041 §2.1, §2.2), which turns a
//! stacked expert tensor into one record per expert with its own extent; a model's identity:
//! each source file's `FileDigest` and the v1 `ModelId` ([`identity`], RFC-0041 §1); and the KV
//! formula ([`kv`], RFC-0022 §2), whose `stored` and `read_per_step` size KV for every party;
//! and chunk boundaries with the chunk-table format ([`chunk`], RFC-0041 §6.1, §6.2); and the
//! identities derived from chunk IDs and the plan: `TensorRecord::hash`, `ShardSpec::digest` and
//! `ExtentKey` ([`identity`], RFC-0041 §1). It re-exports `core`'s `ShardSpec`, `PackKey`,
//! `ChunkId` and `LayoutVersion`, which RFC-0041 owns (RFC-0001 §18.28). The census's records,
//! groups and totals and the `ModelManifest`, with its stored form, are in [`census`]; their
//! builder is `syntropy_graph::census`, a layer up, because it resolves the family (RFC-0041 §3).
//! The pack sidecar comes with a later task.

#![forbid(unsafe_code)]

pub mod census;
pub mod chunk;
pub mod config;
pub mod gguf;
pub mod hparams;
pub mod identity;
pub mod kv;
pub mod range;
pub mod safetensors;
pub mod slice;
pub mod tokenizer;

pub use identity::{
    ExtentKey, FileDigest, IdentityError, SourceFile, TensorHashError, model_id, shard_digest,
    shard_spec, tensor_hash,
};
pub use range::{FileSource, HeaderReader, RangeError, RangeSource};
// `core` holds these as plain data because the command channel carries them (RFC-0001 §18.28);
// RFC-0041 owns them, and `manifest` re-exports them.
pub use syntropy_core::{ChunkId, LayoutVersion, PackKey, ShardSpec};
