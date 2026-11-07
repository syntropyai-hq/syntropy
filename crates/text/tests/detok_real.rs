// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental detokenization on the M0 tokenizer file itself (PROD-E2-T03).
//!
//! Both M0 families ship Qwen3's tokenizer, byte-identical in the two repositories, so one file
//! covers them. The test reads it from the fixture cache `cargo xtask fixtures fetch` fills
//! (`SYNTROPY_FIXTURE_CACHE`, objects named by BLAKE3) and checks the bytes against
//! `testdata/fixtures.toml` before use. It is `#[ignore]`d because no CI job fetches fixtures yet.
//! The Llama 3.2 tokenizer is gated; its leg runs with QUAL-E7-T15's import.

use std::path::Path;

use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};
use syntropy_text::detok::Detokenizer;
use syntropy_text::tokenizer::{self, Tokenizer};

const CACHE_VAR: &str = "SYNTROPY_FIXTURE_CACHE";
const NAME: &str = "qwen3-0.6b/tokenizer.json";

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

fn load() -> Tokenizer {
    let cache = std::env::var(CACHE_VAR).unwrap_or_else(|_| {
        panic!("set {CACHE_VAR} to the cache `cargo xtask fixtures fetch` filled")
    });
    let hash = manifest_blake3(NAME);
    let bytes = std::fs::read(Path::new(&cache).join(&hash))
        .unwrap_or_else(|e| panic!("{NAME}: {e}; run `cargo xtask fixtures fetch --name {NAME}`"));
    assert_eq!(
        blake3::hash(&bytes).to_hex().as_str(),
        hash,
        "{NAME}: bytes differ"
    );
    tokenizer::load(&bytes).expect("the tokenizer loads")
}

/// The deltas of `ids` pushed in pieces cut at `cuts`, then the flush.
fn deltas(t: &Tokenizer, ids: &[u32], cuts: &[usize], skip: bool) -> Vec<String> {
    let mut at: Vec<usize> = cuts.iter().map(|c| c % (ids.len() + 1)).collect();
    at.extend([0, ids.len()]);
    at.sort_unstable();
    at.dedup();
    let mut d = Detokenizer::new(skip);
    let mut out: Vec<String> = at
        .windows(2)
        .map(|w| d.push(t, &ids[w[0]..w[1]]).unwrap())
        .collect();
    out.push(d.finish(t).unwrap());
    out
}

fn check(t: &Tokenizer, ids: &[u32], cuts: &[usize], skip: bool) -> Result<(), TestCaseError> {
    let full = t.decode(ids, skip).unwrap();
    let got = deltas(t, ids, cuts, skip);
    prop_assert_eq!(got.concat(), full.clone());
    if !full.contains('\u{FFFD}') {
        prop_assert!(got.iter().all(|d| !d.contains('\u{FFFD}')), "{:?}", got);
    }
    Ok(())
}

#[test]
#[ignore = "needs the fixture cache; no CI job fetches fixtures yet"]
fn deltas_concatenate_to_the_full_decode_on_the_m0_tokenizer() {
    let t = load();
    let n = u32::try_from(t.get_vocab_size(true)).unwrap();

    // Text through the tokenizer, pushed one token at a time: scripts whose characters span
    // several byte-level tokens, emoji sequences and combining marks.
    let texts = [
        "Hello, world! The answer is 42.",
        "日本語のテキストと中文文本，还有한국어 텍스트.",
        "Emoji: 👩‍👩‍👧‍👦 🇦🇺 🧑🏽‍💻 and ✨ sparkles.",
        "Combining: e\u{301} a\u{308} n\u{303}; RTL: שלום עולם, مرحبا بالعالم.",
        "<think>\nreasoning\n</think>\n\nThe answer.<|im_end|>",
    ];
    let mut encoded = 0usize;
    for text in texts {
        let ids = t.encode(text, false).unwrap().get_ids().to_vec();
        let every: Vec<usize> = (0..ids.len()).collect();
        for skip in [false, true] {
            check(&t, &ids, &every, skip).unwrap();
        }
        encoded += ids.len();
    }

    // Random IDs over the whole vocabulary, added tokens included, and random cuts.
    let cases = 4096;
    let mut runner = TestRunner::new(Config {
        cases,
        ..Config::default()
    });
    let strategy = (
        prop::collection::vec(0..n, 0..64),
        prop::collection::vec(0..64usize, 0..16),
        any::<bool>(),
    );
    runner
        .run(&strategy, |(ids, cuts, skip)| check(&t, &ids, &cuts, skip))
        .unwrap();
    println!(
        "{} encoded tokens over {} texts; {cases} random cases over {n} IDs",
        encoded,
        texts.len()
    );
}
