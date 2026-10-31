// SPDX-License-Identifier: MIT OR Apache-2.0

//! A real GGUF file's tokenizer, converted, against its source repository's `tokenizer.json`
//! (COMP-E17-T03; RFC-0043 §8.3, §8.6).
//!
//! [`a_real_gguf_tokenizer_matches_its_source`] is `#[ignore]`d: it needs the two files, which
//! no CI job fetches yet. Run it with
//!
//! ```text
//! SYNTROPY_TOKENIZE_GGUF=<model.gguf> SYNTROPY_TOKENIZE_SOURCE=<tokenizer.json> \
//!     cargo test -p syntropy-graph --test tokenize_real -- --include-ignored --nocapture
//! ```
//!
//! `SYNTROPY_TOKENIZE_EXTRA=<file>` adds the paragraphs of a UTF-8 text file, split at blank
//! lines, to the corpus.
//!
//! It encodes the project corpus (`tests/data/tokenize/corpus.json`) through both, with and
//! without the post-processor's specials, and requires the same token IDs. It requires
//! byte-level round trips: decoding gives the text back, or the source's own normalized form of
//! it where the source normalizes (NFC). With special-token matching off, as for content spans
//! (RFC-0043 §8.7), it requires the same IDs except on items that hold the text of a token the
//! two mark differently, which a GGUF file cannot record (`syntropy_graph::tokenize`).

use std::collections::BTreeMap;
use std::fs::File;

use serde_json::Value;
use syntropy_graph::tokenize::from_gguf;
use syntropy_manifest::gguf::{GgufHeader, Limits};
use syntropy_manifest::range::FileSource;
use tokenizers::Tokenizer;

const GGUF_VAR: &str = "SYNTROPY_TOKENIZE_GGUF";
const SOURCE_VAR: &str = "SYNTROPY_TOKENIZE_SOURCE";
const EXTRA_VAR: &str = "SYNTROPY_TOKENIZE_EXTRA";

fn corpus() -> Vec<String> {
    let doc: Value =
        serde_json::from_str(include_str!("data/tokenize/corpus.json")).expect("the corpus");
    doc["items"]
        .as_array()
        .expect("items")
        .iter()
        .map(|v| v.as_str().expect("a string").to_owned())
        .collect()
}

fn ids(t: &Tokenizer, text: &str, add_special: bool) -> Vec<u32> {
    t.encode(text, add_special)
        .unwrap_or_else(|e| panic!("{text:?} encodes: {e}"))
        .get_ids()
        .to_vec()
}

/// Every added token's text and whether it is special.
fn added(model: &[u8]) -> BTreeMap<String, bool> {
    let v: Value = serde_json::from_slice(model).expect("a tokenizer.json");
    v["added_tokens"]
        .as_array()
        .expect("added_tokens")
        .iter()
        .map(|a| {
            (
                a["content"].as_str().expect("content").to_owned(),
                a["special"].as_bool().expect("special"),
            )
        })
        .collect()
}

#[test]
#[ignore = "needs a GGUF file and its source tokenizer.json (module docs)"]
fn a_real_gguf_tokenizer_matches_its_source() {
    let gguf = std::env::var(GGUF_VAR).unwrap_or_else(|_| panic!("set {GGUF_VAR}"));
    let source_path = std::env::var(SOURCE_VAR).unwrap_or_else(|_| panic!("set {SOURCE_VAR}"));
    let file = FileSource::from_file(File::open(&gguf).expect("the GGUF file opens"))
        .expect("the GGUF file's length");
    let header = GgufHeader::read(&file, &Limits::default()).expect("the GGUF header reads");
    let spec = from_gguf(&header.metadata)
        .expect("the tokenizer converts")
        .expect("the file has a tokenizer");
    let source_bytes = std::fs::read(&source_path).expect("the source tokenizer.json reads");

    let mut converted = Tokenizer::from_bytes(spec.model()).expect("the conversion loads");
    let mut source = Tokenizer::from_bytes(&source_bytes).expect("the source loads");
    let mut corpus = corpus();
    if let Ok(extra) = std::env::var(EXTRA_VAR) {
        let text = std::fs::read_to_string(&extra).expect("the extra corpus reads as UTF-8");
        corpus.extend(
            text.split("\n\n")
                .filter(|p| !p.is_empty())
                .map(str::to_owned),
        );
    }

    let mut compared = 0;
    for text in &corpus {
        for add_special in [false, true] {
            assert_eq!(
                ids(&converted, text, add_special),
                ids(&source, text, add_special),
                "{text:?}, add_special {add_special}"
            );
            compared += 1;
        }
        let encoded = ids(&converted, text, false);
        let back = converted.decode(&encoded, false).expect("decodes");
        let source_back = source.decode(&encoded, false).expect("decodes");
        assert_eq!(back, source_back, "{text:?}");
        let normalized = source
            .get_normalizer()
            .map(|n| {
                let mut s = tokenizers::NormalizedString::from(text.as_str());
                tokenizers::Normalizer::normalize(n, &mut s).expect("normalizes");
                s.get().to_owned()
            })
            .unwrap_or_else(|| text.clone());
        assert_eq!(back, normalized, "{text:?} round-trips");
    }

    // Special-token matching off: tokens whose special flag differs are the only difference.
    let (ours, theirs) = (added(spec.model()), added(&source_bytes));
    let differs: Vec<&String> = ours
        .iter()
        .filter(|(t, s)| theirs.get(*t).is_some_and(|x| x != *s))
        .map(|(t, _)| t)
        .collect();
    converted.set_encode_special_tokens(true);
    source.set_encode_special_tokens(true);
    let (mut same, mut skipped) = (0, 0);
    for text in &corpus {
        if differs.iter().any(|t| text.contains(t.as_str())) {
            skipped += 1;
            continue;
        }
        assert_eq!(
            ids(&converted, text, false),
            ids(&source, text, false),
            "{text:?}, special matching off"
        );
        same += 1;
    }

    let s = spec.specials();
    println!(
        "{} items, {compared} encodings equal, round trips equal; special matching off: {same} equal, {skipped} skipped",
        corpus.len()
    );
    println!("added tokens whose special flag differs from the source: {differs:?}");
    println!(
        "bos {:?} eos {:?} end_of_turn {:?} pad {:?} unk {:?} fim {:?}/{:?}/{:?} add_bos {} add_eos {}",
        s.bos,
        s.eos,
        s.end_of_turn,
        s.pad,
        s.unk,
        s.fim_prefix,
        s.fim_middle,
        s.fim_suffix,
        spec.add_bos(),
        spec.add_eos()
    );
    println!("TokenizerId {}", spec.id());
}
