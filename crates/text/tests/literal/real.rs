// SPDX-License-Identifier: MIT OR Apache-2.0

//! The literal-tokenization checks on the M0 tokenizer files themselves (COMP-E17-T13).
//!
//! Both M0 families ship Qwen3's tokenizer, byte-identical in the two repositories
//! (`crates/manifest/tests/data/tokenizer/real-m0.json`), so one file covers them. The test reads
//! it from the fixture cache `cargo xtask fixtures fetch` fills (`SYNTROPY_FIXTURE_CACHE`, objects
//! named by BLAKE3) and hashes the bytes against `testdata/fixtures.toml` before use. It is
//! `#[ignore]`d because no CI job fetches fixtures yet. The Llama 3.2 tokenizer is gated; its leg
//! runs with QUAL-E7-T15's import.

use std::path::Path;

use syntropy_text::tokenizer::injection::{self, ContentPosition};
use syntropy_text::tokenizer::{self, Span, SpanEncoder, SpecialTokens};

use crate::support::{FAMILIES, concat};

const CACHE_VAR: &str = "SYNTROPY_FIXTURE_CACHE";
/// The M0 tokenizers, by their `testdata/fixtures.toml` names.
const M0_TOKENIZERS: [&str; 2] = ["qwen3-0.6b/tokenizer.json", "qwen3-30b-a3b/tokenizer.json"];

/// The BLAKE3 `testdata/fixtures.toml` records for `name`.
fn manifest_blake3(name: &str) -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fixtures.toml");
    let text = std::fs::read_to_string(&manifest).expect("testdata/fixtures.toml reads");
    let wanted = format!("name    = \"{name}\"");
    let mut lines = text.lines().skip_while(|l| l.trim() != wanted);
    assert!(
        lines.next().is_some(),
        "{name} is not in the fixture manifest"
    );
    let line = lines
        .take_while(|l| !l.starts_with("[[fixture]]"))
        .find(|l| l.starts_with("blake3"))
        .unwrap_or_else(|| panic!("{name} has no blake3"));
    let hash = line.split('"').nth(1).expect("a quoted hash");
    assert_eq!(hash.len(), 64, "{name} is recorded but not imported");
    hash.to_owned()
}

fn load(name: &str) -> SpanEncoder {
    let cache = std::env::var(CACHE_VAR).unwrap_or_else(|_| {
        panic!("set {CACHE_VAR} to the cache `cargo xtask fixtures fetch` filled")
    });
    let hash = manifest_blake3(name);
    let bytes = std::fs::read(Path::new(&cache).join(&hash))
        .unwrap_or_else(|e| panic!("{name}: {e}; run `cargo xtask fixtures fetch --name {name}`"));
    assert_eq!(
        blake3::hash(&bytes).to_hex().as_str(),
        hash,
        "{name}: bytes differ"
    );
    SpanEncoder::new(tokenizer::load(&bytes).expect("the tokenizer loads")).expect("supported")
}

#[test]
#[ignore = "needs the fixture cache; no CI job fetches fixtures yet"]
fn m0_tokenizers_tokenize_content_literally() {
    let family = &FAMILIES[0];
    for name in M0_TOKENIZERS {
        let encoder = load(name);
        let tokenizer = encoder.tokenizer();
        // Every added token, special or not, is a control token here.
        assert_eq!(
            encoder.control_ids().len(),
            tokenizer.get_added_tokens_decoder().len()
        );
        for marker in [
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<|im_start|>",
        ] {
            let id = tokenizer.token_to_id(marker).expect("an added token");
            assert!(encoder.is_control(id), "{name}: {marker}");
        }
        let needles = injection::needles(&encoder, family.markers, family.role_headers);
        for case in injection::cases(&needles) {
            let spans = crate::support::render(family.layout, case.position, &case.content);
            let ids = encoder
                .encode(&spans, SpecialTokens::Literal)
                .expect("encodes");
            let frame = encoder
                .encode(
                    &crate::support::template_only(&spans),
                    SpecialTokens::Literal,
                )
                .expect("encodes");
            let controls = |ids: &[u32]| -> Vec<u32> {
                ids.iter()
                    .copied()
                    .filter(|&id| encoder.is_control(id))
                    .collect()
            };
            assert_eq!(controls(&ids), controls(&frame), "{name}: {case:?}");
            let alone = encoder
                .encode(
                    &[Span::content(case.content.as_str())],
                    SpecialTokens::Literal,
                )
                .expect("encodes");
            assert!(controls(&alone).is_empty(), "{name}: {case:?}");
            assert_eq!(
                tokenizer.decode(&alone, false).expect("decodes"),
                case.content
            );
        }
        // With no control text in content, the IDs are the engine's on the whole prompt.
        for position in ContentPosition::ALL {
            let spans = crate::support::render(family.layout, position, "Hello, wörld! 1234\n\tok");
            let ids = encoder
                .encode(&spans, SpecialTokens::Literal)
                .expect("encodes");
            let engine = tokenizer.encode(concat(&spans), false).expect("encodes");
            assert_eq!(ids, engine.get_ids(), "{name}: {position:?}");
        }
    }
}
