// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental detokenization on the synthetic tokenizers (PROD-E2-T03; RFC-0050 Testing,
//! "Detokenization is lossless").
//!
//! The tokenizers are the two byte-level files laid out like the M0 and Llama 3 tokenizers
//! (`tests/data/tokenizer/*`), and a SentencePiece-style one with byte fallback built here: a
//! `▁` normalizer, `<0xNN>` byte tokens, and the decoder sequence the source repositories ship for
//! that kind (replace `▁`, byte fallback, fuse, strip one leading space). The run on the real
//! Qwen3 file is `tests/detok_real.rs`.

use std::sync::LazyLock;

use proptest::prelude::*;
use serde_json::json;

use super::*;
use crate::tokenizer;

const QWEN3_SHAPED: &str = include_str!("../../tests/data/tokenizer/qwen3-shaped/tokenizer.json");
const LLAMA3_SHAPED: &str = include_str!("../../tests/data/tokenizer/llama3-shaped/tokenizer.json");

/// The pieces of the byte-fallback tokenizer after its three specials and 256 byte tokens.
const SPM_PIECES: [&str; 18] = [
    "▁", "h", "e", "l", "o", "w", "r", "d", "é", "▁h", "el", "lo", "▁w", "or", "ld", "▁hel",
    "▁wor", "▁é",
];
const SPM_MERGES: [&str; 9] = [
    "▁ h", "e l", "l o", "▁ w", "o r", "l d", "▁h el", "▁w or", "▁ é",
];

/// A SentencePiece-style BPE tokenizer with byte fallback, as `tokenizer.json`.
fn spm_json() -> String {
    let mut vocab = serde_json::Map::new();
    for (i, t) in ["<unk>", "<s>", "</s>"].iter().enumerate() {
        vocab.insert((*t).to_owned(), json!(i));
    }
    for b in 0..=255u32 {
        vocab.insert(format!("<0x{b:02X}>"), json!(3 + b));
    }
    for (i, p) in SPM_PIECES.iter().enumerate() {
        vocab.insert((*p).to_owned(), json!(259 + i));
    }
    let special = |id: u32, content: &str| {
        json!({ "id": id, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true })
    };
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [special(0, "<unk>"), special(1, "<s>"), special(2, "</s>")],
        "normalizer": { "type": "Sequence", "normalizers": [
            { "type": "Prepend", "prepend": "▁" },
            { "type": "Replace", "pattern": { "String": " " }, "content": "▁" }
        ]},
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": { "type": "Sequence", "decoders": [
            { "type": "Replace", "pattern": { "String": "▁" }, "content": " " },
            { "type": "ByteFallback" },
            { "type": "Fuse" },
            { "type": "Strip", "content": " ", "start": 1, "stop": 0 }
        ]},
        "model": {
            "type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true,
            "ignore_merges": false, "vocab": vocab, "merges": SPM_MERGES
        }
    })
    .to_string()
}

/// The three tokenizers, loaded once.
static TOKENIZERS: LazyLock<Vec<(&'static str, Tokenizer)>> = LazyLock::new(|| {
    vec![
        (
            "qwen3-shaped",
            tokenizer::load(QWEN3_SHAPED.as_bytes()).unwrap(),
        ),
        (
            "llama3-shaped",
            tokenizer::load(LLAMA3_SHAPED.as_bytes()).unwrap(),
        ),
        (
            "sentencepiece-byte-fallback",
            tokenizer::load(spm_json().as_bytes()).unwrap(),
        ),
    ]
});

fn tokenizers() -> &'static [(&'static str, Tokenizer)] {
    &TOKENIZERS
}

fn spm() -> &'static Tokenizer {
    &TOKENIZERS[2].1
}

fn id(t: &Tokenizer, token: &str) -> u32 {
    t.token_to_id(token)
        .unwrap_or_else(|| panic!("{token:?} is not in the vocabulary"))
}

/// Pushes `chunks` in turn; returns each push's delta, then the flush.
fn deltas(t: &Tokenizer, chunks: &[&[u32]], skip_special: bool) -> Vec<String> {
    let mut d = Detokenizer::new(skip_special);
    let mut out: Vec<String> = chunks.iter().map(|c| d.push(t, c).unwrap()).collect();
    out.push(d.finish(t).unwrap());
    out
}

#[test]
fn a_character_split_across_byte_tokens_is_released_whole() {
    let t = spm();
    // "€" is E2 82 AC, three byte-fallback tokens.
    let euro = [id(t, "<0xE2>"), id(t, "<0x82>"), id(t, "<0xAC>")];
    let hel = id(t, "▁hel");
    let lo = id(t, "lo");
    let got = deltas(
        t,
        &[&[hel], &[lo], &[euro[0]], &[euro[1]], &[euro[2]], &[hel]],
        true,
    );
    // The run of byte tokens is held until a token of another spelling ends it.
    assert_eq!(got, ["hel", "lo", "", "", "", "€ hel", ""]);
    assert_eq!(
        t.decode(&[hel, lo, euro[0], euro[1], euro[2], hel], true)
            .unwrap(),
        "hello€ hel"
    );
}

#[test]
fn a_byte_token_that_invalidates_its_run_never_follows_released_text() {
    let t = spm();
    let bytes: Vec<u32> = ["<0xE2>", "<0x82>", "<0xAC>", "<0x80>"]
        .iter()
        .map(|b| id(t, b))
        .collect();
    // E2 82 AC 80 is not valid UTF-8, so the decoder writes U+FFFD for each of the four bytes,
    // "€" included. Had "€" been released, the stream could not match the decode.
    let full = t.decode(&bytes, true).unwrap();
    assert_eq!(full, "\u{FFFD}".repeat(4));
    let chunks: Vec<&[u32]> = bytes.iter().map(std::slice::from_ref).collect();
    let got = deltas(t, &chunks, true);
    assert_eq!(got, ["", "", "", "", full.as_str()]);
    // A skipped special token does not end the run: the decoder never sees it.
    let s = id(t, "<s>");
    let got = deltas(t, &[&bytes[..3], &[s], &bytes[3..]], true);
    assert_eq!(got.concat(), full);
}

#[test]
fn the_first_tokens_leading_space_is_stripped_once() {
    let t = spm();
    let (hel, wor) = (id(t, "▁hel"), id(t, "▁wor"));
    assert_eq!(
        deltas(t, &[&[hel], &[wor], &[hel]], true),
        ["hel", " wor", " hel", ""]
    );
}

#[test]
fn incomplete_bytes_at_the_end_are_flushed_by_finish() {
    let t = spm();
    let (hel, e2) = (id(t, "▁hel"), id(t, "<0xE2>"));
    let got = deltas(t, &[&[hel, e2]], true);
    assert_eq!(got, ["", "hel\u{FFFD}"]);
    assert_eq!(got.concat(), t.decode(&[hel, e2], true).unwrap());
}

#[test]
fn byte_level_characters_split_across_tokens_are_held() {
    // The byte-level alphabet spells bytes E2 82 AC as "â", "Ĥ", "¬".
    for (name, t) in tokenizers().iter().take(2) {
        let parts: Vec<u32> = ["â", "Ĥ", "¬"].iter().map(|s| id(t, s)).collect();
        let got = deltas(t, &[&parts[..1], &parts[1..2], &parts[2..]], true);
        assert_eq!(got, ["", "", "€", ""], "{name}");
    }
}

#[test]
fn the_window_stays_short_on_clean_text() {
    let t = spm();
    let ids = [id(t, "▁hel"), id(t, "lo"), id(t, "▁wor"), id(t, "ld")];
    let mut d = Detokenizer::new(true);
    for i in 0..400 {
        d.push(t, &[ids[i % ids.len()]]).unwrap();
        assert!(
            d.tokens.len() - d.prefix <= 2,
            "window {}",
            d.tokens.len() - d.prefix
        );
        assert!(d.pending().is_empty());
    }
}

#[test]
fn specials_follow_skip_special() {
    let t = spm();
    let (s, hel) = (id(t, "<s>"), id(t, "▁hel"));
    assert_eq!(deltas(t, &[&[s], &[hel]], true).concat(), "hel");
    assert_eq!(
        deltas(t, &[&[s], &[hel]], false).concat(),
        t.decode(&[s, hel], false).unwrap()
    );
}

/// Random IDs over the whole vocabulary, specials and byte tokens included, and random cuts.
fn case() -> impl Strategy<Value = (usize, Vec<u32>, Vec<usize>, bool)> {
    (0..3usize, any::<bool>()).prop_flat_map(|(which, skip)| {
        let n = u32::try_from(tokenizers()[which].1.get_vocab_size(true)).unwrap();
        (
            Just(which),
            prop::collection::vec(0..n, 0..48),
            prop::collection::vec(0..48usize, 0..16),
            Just(skip),
        )
    })
}

/// Byte tokens only, so characters split across tokens are common.
fn byte_case() -> impl Strategy<Value = (Vec<u32>, Vec<usize>)> {
    (
        prop::collection::vec(prop_oneof![3..259u32, 259..277u32, 0..3u32], 0..48),
        prop::collection::vec(0..48usize, 0..16),
    )
}

fn cut<'a>(ids: &'a [u32], cuts: &[usize]) -> Vec<&'a [u32]> {
    let mut at: Vec<usize> = cuts.iter().map(|c| c % (ids.len() + 1)).collect();
    at.extend([0, ids.len()]);
    at.sort_unstable();
    at.dedup();
    at.windows(2).map(|w| &ids[w[0]..w[1]]).collect()
}

fn check(t: &Tokenizer, ids: &[u32], cuts: &[usize], skip: bool) -> Result<(), TestCaseError> {
    let full = t.decode(ids, skip).unwrap();
    let got = deltas(t, &cut(ids, cuts), skip);
    prop_assert_eq!(got.concat(), full.clone());
    if !full.contains(REPLACEMENT) {
        prop_assert!(got.iter().all(|d| !d.contains(REPLACEMENT)), "{got:?}");
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1024, ..ProptestConfig::default() })]

    /// RFC-0050 Testing: the concatenated deltas equal the full decode, and no delta contains
    /// U+FFFD unless the full decode does, on each tokenizer and for any split into pushes.
    #[test]
    fn deltas_concatenate_to_the_full_decode((which, ids, cuts, skip) in case()) {
        let (_, t) = &tokenizers()[which];
        check(t, &ids, &cuts, skip)?;
    }

    /// The same on byte-fallback tokens, which split characters across tokens.
    #[test]
    fn byte_fallback_deltas_concatenate_to_the_full_decode((ids, cuts) in byte_case(), skip in any::<bool>()) {
        check(spm(), &ids, &cuts, skip)?;
    }
}
