// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF and safetensors readers, the tensor census, the model manifest, the KV formula and the pack
//! sidecar format.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0041. It holds the `RangeSource` every reader
//! parses through, with the local-file and recorded-fixture sources and the bounded,
//! coalescing header-read routine ([`range`], RFC-0041 §2), the GGUF and safetensors readers,
//! `config.json` read as metadata ([`config`]), `TokenizerSpec`, and the first census and
//! hyperparameter types ([`census`], [`hparams`]) that the family modules in `graph` produce
//! (RFC-0043 §4, §5.2), and expert slicing ([`slice`](mod@slice), RFC-0041 §2.1, §2.2), which turns a
//! stacked expert tensor into one record per expert with its own extent, and the KV formula
//! ([`kv`], RFC-0022 §2), whose `stored` and `read_per_step` size KV for every party. The census
//! builder, the manifest and the pack sidecar come with later tasks.

#![forbid(unsafe_code)]

pub mod census;
pub mod config;
pub mod gguf;
pub mod hparams;
pub mod kv;
pub mod range;
pub mod safetensors;
pub mod slice;
pub mod tokenizer;

pub use range::{FileSource, HeaderReader, RangeError, RangeSource};
