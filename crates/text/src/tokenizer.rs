// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tokenizer engine (RFC-0043 §8.2, ADR-0431 decision 1).
//!
//! Tokenization runs on the HF `tokenizers` crate, pinned exactly in the workspace manifest and
//! built with its pure-Rust regex backend (`fancy-regex`), never the Oniguruma C library, so it
//! cross-compiles to every ADR-0020 target. Every model reaches it in one form: the serialized
//! `tokenizer.json` that `TokenizerSpec::model` holds, shipped by a safetensors source or converted
//! from GGUF metadata at census time (RFC-0043 §8.3). Encoding, detokenization and the segment map
//! of RFC-0050 build on [`load`]; they are planned.

pub use tokenizers::Tokenizer;

/// Loads the engine from a serialized `tokenizer.json`, as `TokenizerSpec::model` holds it.
///
/// # Errors
///
/// The engine's own error when the bytes are not a tokenizer it can read.
pub fn load(model: &[u8]) -> tokenizers::Result<Tokenizer> {
    Tokenizer::from_bytes(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word-level model behind a regex pre-tokenizer, so loading and encoding run the engine's
    /// regex backend.
    const MODEL: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": { "Regex": "\\s+" },
            "behavior": "Removed",
            "invert": false
        },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": { "[UNK]": 0, "hello": 1, "world": 2 },
            "unk_token": "[UNK]"
        }
    }"#;

    #[test]
    fn loads_and_encodes_through_the_regex_pre_tokenizer() {
        let tokenizer = load(MODEL.as_bytes()).expect("the model loads");
        let encoding = tokenizer
            .encode("hello  world\tother", false)
            .expect("the text encodes");
        assert_eq!(encoding.get_ids(), [1, 2, 0]);
    }

    #[test]
    fn refuses_bytes_that_are_not_a_tokenizer() {
        assert!(load(b"{\"model\": 1}").is_err());
    }
}
