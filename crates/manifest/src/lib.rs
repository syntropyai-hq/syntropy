// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF and safetensors readers, the tensor census, the model manifest, the KV formula and the pack
//! sidecar format.
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0041. It holds the `RangeSource` every reader
//! parses through, with the local-file and recorded-fixture sources and the bounded,
//! coalescing header-read routine ([`range`], RFC-0041 §2). The readers, the census, the
//! manifest, the KV formula and the pack sidecar come with later tasks.

#![forbid(unsafe_code)]

pub mod range;
pub mod safetensors;
pub mod tokenizer;

pub use range::{FileSource, HeaderReader, RangeError, RangeSource};
