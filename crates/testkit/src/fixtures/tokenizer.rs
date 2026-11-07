// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Nano` byte-level tokenizer (RFC-0060 §12.1): one token per byte, then the specials.
//!
//! Token `b` for `b < 256` is byte `b`, spelled as byte-level BPE spells bytes: printable bytes
//! as themselves and the rest shifted to code points from U+0100 up. There are no merges, so
//! every text tokenizes to its bytes. The same vocabulary is written as `tokenizer.json` (with
//! `tokenizer_config.json`) for a safetensors checkpoint and as `tokenizer.ggml.*` metadata for
//! a GGUF file.

use std::fmt::Write;

/// The special tokens, in ID order from 256.
pub const SPECIALS: [&str; 3] = ["<s>", "</s>", "<pad>"];
/// BOS.
pub const BOS: u32 = 256;
/// EOS.
pub const EOS: u32 = 257;
/// Padding.
pub const PAD: u32 = 258;
/// The vocabulary size: 256 bytes and the specials.
pub const VOCAB: u32 = 256 + SPECIALS.len() as u32;

/// Every token's spelling, in ID order.
pub fn tokens() -> Vec<String> {
    let printable = |b: u32| matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
    let mut shifted = 0;
    let mut out: Vec<String> = (0..256u32)
        .map(|b| {
            let c = if printable(b) {
                b
            } else {
                shifted += 1;
                255 + shifted
            };
            char::from_u32(c).expect("below U+0200").to_string()
        })
        .collect();
    out.extend(SPECIALS.iter().map(|s| (*s).to_owned()));
    out
}

/// `s` as a JSON string.
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).expect("a String"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `tokenizer.json`: a byte-level BPE model with no merges, the specials as added tokens.
pub fn tokenizer_json() -> String {
    let tokens = tokens();
    let mut added = String::new();
    for (i, s) in SPECIALS.iter().enumerate() {
        if i > 0 {
            added.push_str(", ");
        }
        write!(
            added,
            r#"{{"id": {}, "content": {}, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}}"#,
            256 + i,
            json_str(s)
        )
        .expect("a String");
    }
    let mut vocab = String::new();
    for (id, t) in tokens.iter().enumerate() {
        if id > 0 {
            vocab.push_str(", ");
        }
        write!(vocab, "{}: {id}", json_str(t)).expect("a String");
    }
    format!(
        concat!(
            "{{\"version\": \"1.0\", \"truncation\": null, \"padding\": null, ",
            "\"added_tokens\": [{added}], \"normalizer\": null, ",
            "\"pre_tokenizer\": {{\"type\": \"ByteLevel\", \"add_prefix_space\": false, ",
            "\"trim_offsets\": true, \"use_regex\": false}}, \"post_processor\": null, ",
            "\"decoder\": {{\"type\": \"ByteLevel\", \"add_prefix_space\": false, ",
            "\"trim_offsets\": true, \"use_regex\": false}}, ",
            "\"model\": {{\"type\": \"BPE\", \"dropout\": null, \"unk_token\": null, ",
            "\"continuing_subword_prefix\": null, \"end_of_word_suffix\": null, ",
            "\"fuse_unk\": false, \"byte_fallback\": false, \"ignore_merges\": false, ",
            "\"vocab\": {{{vocab}}}, \"merges\": []}}}}\n"
        ),
        added = added,
        vocab = vocab
    )
}

/// `tokenizer_config.json`: the specials by spelling, BOS added, EOS not.
pub fn tokenizer_config_json() -> String {
    format!(
        "{{\"bos_token\": {}, \"eos_token\": {}, \"pad_token\": {}, \"add_bos_token\": true, \"add_eos_token\": false}}\n",
        json_str(SPECIALS[0]),
        json_str(SPECIALS[1]),
        json_str(SPECIALS[2])
    )
}
