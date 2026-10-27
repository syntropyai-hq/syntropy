// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF and safetensors readers, the tensor census, the model manifest, the KV formula and the pack
//! sidecar format.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0041. It holds the `RangeSource` every reader
//! parses through, with the local-file and recorded-fixture sources and the bounded,
//! coalescing header-read routine ([`range`], RFC-0041 §2), the GGUF and safetensors readers,
//! `config.json` read as metadata ([`config`]), `TokenizerSpec`, and the first census and
//! hyperparameter types ([`census`], [`hparams`]) that the family modules in `graph` produce
//! (RFC-0043 §4, §5.2). The census builder, the manifest, the KV formula and the pack sidecar
//! come with later tasks.

#![forbid(unsafe_code)]

pub mod census;
pub mod config;
pub mod gguf;
pub mod hparams;
pub mod range;
pub mod safetensors;
pub mod tokenizer;

pub use range::{FileSource, HeaderReader, RangeError, RangeSource};
