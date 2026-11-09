// SPDX-License-Identifier: MIT OR Apache-2.0

//! The M0 families' shipped chat template against the reference engine's renderings
//! (PROD-E3-T10).
//!
//! Both M0 checkpoints (Qwen3-0.6B for `llama`, Qwen3-30B-A3B for `qwen_moe`) ship the same
//! `tokenizer_config.json`, byte for byte (`crates/manifest/tests/data/tokenizer/real-m0.json`).
//! The test reads it from the fixture cache `cargo xtask fixtures fetch` fills
//! (`SYNTROPY_FIXTURE_CACHE`, objects named by BLAKE3), checks the bytes against
//! `testdata/fixtures.toml`, builds its `TokenizerSpec`, and renders `conversations.json` with
//! the template in use. Each rendering must equal `expected-real-m0.json`, which `record.py
//! --real` recorded. The injection check runs on the real template with the real tokenizer.
//! It is `#[ignore]`d because no CI job fetches fixtures yet.

use std::path::Path;

use syntropy_manifest::tokenizer::{TokenizerSources, TokenizerSpec};
use syntropy_text::template::{ChatTemplate, Limits};
use syntropy_text::tokenizer::{self, SpanEncoder, injection};

use crate::{DATE, check_golden, check_injection, json};

const CACHE_VAR: &str = "SYNTROPY_FIXTURE_CACHE";
const CHECKPOINTS: [[&str; 3]; 2] = [
    [
        "qwen3-0.6b/tokenizer.json",
        "qwen3-0.6b/tokenizer_config.json",
        "qwen3-0.6b/generation_config.json",
    ],
    [
        "qwen3-30b-a3b/tokenizer.json",
        "qwen3-30b-a3b/tokenizer_config.json",
        "qwen3-30b-a3b/generation_config.json",
    ],
];

/// The bytes of fixture `name`, checked against the manifest's BLAKE3.
fn fixture(name: &str) -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/fixtures.toml");
    let text = std::fs::read_to_string(&manifest).expect("fixtures.toml");
    let block = text
        .split(&format!("name    = \"{name}\""))
        .nth(1)
        .unwrap_or_else(|| panic!("{name} is not in the manifest"));
    let hash = block
        .split("blake3  = \"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("a blake3");
    let cache = std::env::var(CACHE_VAR).unwrap_or_else(|_| {
        panic!("set {CACHE_VAR} to the cache `cargo xtask fixtures fetch` fills")
    });
    let bytes = std::fs::read(Path::new(&cache).join(hash))
        .unwrap_or_else(|e| panic!("{name}: {e}; run `cargo xtask fixtures fetch --name {name}`"));
    assert_eq!(blake3::hash(&bytes).to_hex().as_str(), hash, "{name}");
    bytes
}

#[test]
#[ignore = "needs the fixture cache; no CI job fetches fixtures yet"]
fn m0_templates_match_the_reference_engine() {
    let expected = json("template/expected-real-m0.json");
    let recorded = &expected["templates"][0];
    for [tok, config, generation] in CHECKPOINTS {
        let (tok, config, generation) = (fixture(tok), fixture(config), fixture(generation));
        let spec = TokenizerSpec::from_sources(TokenizerSources {
            tokenizer_json: &tok,
            tokenizer_config: Some(&config),
            generation_config: Some(&generation),
            chat_template_file: None,
        })
        .expect("spec");
        let source = spec.template_in_use().expect("a template");
        let template = ChatTemplate::new(source, Limits::default()).expect("loads");
        assert!(check_golden(&template, recorded, DATE, true) >= 17);

        let encoder = SpanEncoder::new(tokenizer::load(&tok).expect("loads")).expect("supported");
        let needles = injection::needles(
            &encoder,
            &["<tool_call>", "</tool_call>", "<think>", "</think>"],
            &[
                "<|im_start|>system\n",
                "<|im_start|>user\n",
                "<|im_start|>assistant\n",
            ],
        );
        for case in injection::cases(&needles) {
            check_injection(&template, &encoder, &case.content, "hello");
            check_injection(&template, &encoder, "Be brief.", &case.content);
        }
    }
}
