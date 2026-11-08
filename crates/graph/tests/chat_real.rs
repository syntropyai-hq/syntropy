// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ChatFormat` on the M0 families' recorded repository files (COMP-E17-T04; RFC-0043 §8.4).
//!
//! [`m0_chat_formats_resolve_from_the_fixture_store`] reads each M0 checkpoint's tokenizer side
//! files from the fixture cache that `cargo xtask fixtures fetch` fills (`<cache>/<blake3>`,
//! the BLAKE3 taken from `testdata/fixtures.toml`), builds its `TokenizerSpec`, and resolves the
//! family's `ChatFormat`. The spec's `TokenizerId` must equal the one COMP-E17-T14 pinned in
//! `crates/manifest/tests/data/tokenizer/real-m0.json`; it covers the tokenizer's bytes, its
//! specials and the template, so a different file cannot pass. The expected markers were read
//! from the files by hand, apart from this crate: the IDs of `tokenizer.json`'s added tokens
//! `<tool_call>`, `</tool_call>`, `<think>` and `</think>`, and the template's tool-call and
//! reasoning syntax. It is `#[ignore]`d because it needs the cache: no CI job fetches fixtures
//! yet. Llama-3.2-1B's files are gated; its run waits for QUAL-E7-T15's import.

use std::path::{Path, PathBuf};

use syntropy_graph::chat::{self, ChatFormat, ChatRule, Marker, ReasoningSyntax, ToolCallFormat};
use syntropy_manifest::tokenizer::{TokenizerSources, TokenizerSpec};

const CACHE_VAR: &str = "SYNTROPY_FIXTURE_CACHE";

struct Checkpoint {
    hub: &'static str,
    rule: &'static ChatRule,
    /// The fixture names of `tokenizer.json`, `tokenizer_config.json` and
    /// `generation_config.json`.
    files: [&'static str; 3],
    /// `real-m0.json`'s `tokenizer_id` for it.
    tokenizer_id: &'static str,
}

const QWEN3_ID: &str = "8ce2893d3699c49201eaa85d07f1fc25c12a6b3ec0fb7dad2d90b6d76610a4ed";

const CHECKPOINTS: [Checkpoint; 2] = [
    Checkpoint {
        hub: "Qwen/Qwen3-0.6B",
        rule: &chat::LLAMA,
        files: [
            "qwen3-0.6b/tokenizer.json",
            "qwen3-0.6b/tokenizer_config.json",
            "qwen3-0.6b/generation_config.json",
        ],
        tokenizer_id: QWEN3_ID,
    },
    Checkpoint {
        hub: "Qwen/Qwen3-30B-A3B",
        rule: &chat::QWEN_MOE,
        files: [
            "qwen3-30b-a3b/tokenizer.json",
            "qwen3-30b-a3b/tokenizer_config.json",
            "qwen3-30b-a3b/generation_config.json",
        ],
        tokenizer_id: QWEN3_ID,
    },
];

/// Both Qwen3 repositories ship byte-identical tokenizer files, so one expectation serves both:
/// `<tool_call>` 151657, `</tool_call>` 151658, `<think>` 151667 and `</think>` 151668 are added
/// tokens (not marked special); the generation prompt writes an empty reasoning section only
/// when thinking is off, so output does not start open; the end-of-turn set {151643, 151645}
/// without EOS 151645 leaves 151643.
fn qwen3_expected() -> ChatFormat {
    ChatFormat {
        tool_calls: ToolCallFormat::TaggedJson,
        tool_markers: [Marker::Special(151_657), Marker::Special(151_658)]
            .into_iter()
            .collect(),
        reasoning: Some(ReasoningSyntax::Markers {
            open: Marker::Special(151_667),
            close: Marker::Special(151_668),
            starts_open: false,
        }),
        stop: [151_643].into_iter().collect(),
    }
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The BLAKE3 `testdata/fixtures.toml` records for the fixture `name`.
fn blake3_of(manifest: &str, name: &str) -> String {
    let want = format!("\"{name}\"");
    let mut lines = manifest.lines();
    while let Some(line) = lines.next() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == "name" && value.trim() == want {
            for line in lines.by_ref() {
                if let Some((k, v)) = line.split_once('=')
                    && k.trim() == "blake3"
                {
                    return v.trim().trim_matches('"').to_owned();
                }
            }
        }
    }
    panic!("{name} is not in testdata/fixtures.toml")
}

/// The real-file run. Needs `SYNTROPY_FIXTURE_CACHE`.
#[test]
#[ignore = "needs the fixture cache (SYNTROPY_FIXTURE_CACHE)"]
fn m0_chat_formats_resolve_from_the_fixture_store() {
    let cache = std::env::var_os(CACHE_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {CACHE_VAR} to the fixture cache"));
    let manifest = std::fs::read_to_string(repo().join("testdata/fixtures.toml")).unwrap();
    for cp in &CHECKPOINTS {
        let [tok, cfg, generation] = cp.files.map(|name| {
            let path = cache.join(blake3_of(&manifest, name));
            std::fs::read(&path).unwrap_or_else(|e| panic!("{name}: {}: {e}", path.display()))
        });
        let spec = TokenizerSpec::from_sources(TokenizerSources {
            tokenizer_json: &tok,
            tokenizer_config: Some(&cfg),
            generation_config: Some(&generation),
            chat_template_file: None,
        })
        .unwrap_or_else(|e| panic!("{}: {e}", cp.hub));
        assert_eq!(
            spec.id().to_string(),
            cp.tokenizer_id,
            "{}: not the recorded files",
            cp.hub
        );
        let format = chat::resolve(cp.rule, &spec).unwrap_or_else(|e| panic!("{}: {e}", cp.hub));
        assert_eq!(format, qwen3_expected(), "{} ({})", cp.hub, cp.rule.family);
        eprintln!("{} ({}): ok, {format:?}", cp.hub, cp.rule.family);
    }
}

/// The pinned IDs are COMP-E17-T14's, and the files are in the manifest. Runs everywhere.
#[test]
fn the_checkpoints_match_the_recorded_expectations() {
    let real_m0 =
        std::fs::read_to_string(repo().join("crates/manifest/tests/data/tokenizer/real-m0.json"))
            .unwrap();
    let manifest = std::fs::read_to_string(repo().join("testdata/fixtures.toml")).unwrap();
    for cp in &CHECKPOINTS {
        assert!(
            real_m0.contains(&format!("\"hub\": \"{}\"", cp.hub)),
            "{}",
            cp.hub
        );
        assert!(
            real_m0.contains(&format!("\"tokenizer_id\": \"{}\"", cp.tokenizer_id)),
            "{}",
            cp.hub
        );
        for name in cp.files {
            assert_eq!(blake3_of(&manifest, name).len(), 64, "{name}");
        }
    }
}
