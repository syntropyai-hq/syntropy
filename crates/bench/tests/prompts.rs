// SPDX-License-Identifier: MIT OR Apache-2.0

//! QUAL-E12-T05: `bench-prompts-v1` and exact-length prompt construction (RFC-0061 §3.1).
//!
//! - The checked-in manifest passes [`PromptSet::check`] against `testdata/corpus/SOURCES.toml`:
//!   every item has a licence record that is its source's and on the corpus allow list, and
//!   CC-BY items have their attribution files.
//! - On synthetic items and the Qwen3-shaped tokenizer of COMP-E17-T13's tests: every prompt is
//!   exactly the requested length, the same tokenizer and key give the same tokens across
//!   builders, a shorter prompt is a prefix of a longer one, items cycle when they run out, the
//!   nonce differs by repetition, and `agent` turns extend each other (vision S4).
//! - The store refuses changed or missing text, and `check` refuses each broken rule.
//! - `real_items_build_exact_prompts` runs on the real items (`select.py --out`) and the M0
//!   tokenizer from the fixture cache, `#[ignore]`d because no CI job holds them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use syntropy_bench::prompts::{
    Builder, Category, DirStore, Item, ItemStore, PromptError, PromptSet, Sources, Tokenize, verify,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &str) -> String {
    std::fs::read_to_string(root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn sources() -> Sources {
    Sources::parse(&read("testdata/corpus/SOURCES.toml")).expect("SOURCES.toml parses")
}

fn shipped() -> PromptSet {
    PromptSet::parse(&read("bench/prompts/bench-prompts-v1.toml")).expect("the manifest parses")
}

#[test]
fn the_shipped_manifest_has_a_licence_record_for_every_item() {
    let set = shipped();
    assert_eq!(set.set, "bench-prompts-v1");
    set.check(&sources(), &root()).expect("valid");
    for c in [Category::Chat, Category::Code, Category::Agent] {
        assert!(set.items(c).count() >= 8, "{c:?}");
    }
    // Ten code sources, one file each.
    let code: std::collections::BTreeSet<&str> = set
        .items(Category::Code)
        .map(|i| i.source.as_str())
        .collect();
    assert_eq!(code.len(), 10);
}

// ---- Synthetic items -------------------------------------------------------------------------

fn sha(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// An in-memory store.
struct Mem(BTreeMap<String, String>);

impl ItemStore for Mem {
    fn text(&self, item: &Item) -> Result<String, PromptError> {
        let text = self.0.get(&item.sha256).ok_or_else(|| PromptError::Item {
            id: item.id.clone(),
            reason: "missing".into(),
        })?;
        verify(item, text)?;
        Ok(text.clone())
    }
}

fn item(id: &str, category: Category, text: &str, turns: Option<Vec<usize>>) -> Item {
    let (source, licence) = match category {
        Category::Code => ("code-go", "BSD-3-Clause"),
        _ => ("taskmaster-1", "CC-BY-4.0"),
    };
    Item {
        id: id.into(),
        category,
        source: source.into(),
        licence: licence.into(),
        locator: BTreeMap::new(),
        sha256: sha(text),
        bytes: text.len(),
        turn_bytes: turns,
    }
}

const TURNS: [&str; 4] = [
    "User: book a table for two",
    "\n\nAssistant: Which restaurant?",
    "\n\nUser: The one on Main Street, at 7.",
    "\n\nAssistant: Done. Anything else?",
];

fn synthetic() -> (PromptSet, Mem) {
    let texts = [
        (
            "chat-a",
            Category::Chat,
            "User: hello there\n\nAssistant: Hi! How can I help?",
        ),
        (
            "chat-b",
            Category::Chat,
            "User: what time is it in Tokyo?\n\nAssistant: About 9pm.",
        ),
        (
            "code-a",
            Category::Code,
            "fn main() {\n    println!(\"hello\");\n}\n",
        ),
    ];
    let mut items: Vec<Item> = texts
        .iter()
        .map(|(id, c, t)| item(id, *c, t, None))
        .collect();
    let agent: String = TURNS.concat();
    items.push(item(
        "agent-a",
        Category::Agent,
        &agent,
        Some(TURNS.iter().map(|t| t.len()).collect()),
    ));
    let mut store: BTreeMap<String, String> = texts
        .iter()
        .map(|(_, _, t)| (sha(t), (*t).to_owned()))
        .collect();
    store.insert(sha(&agent), agent);
    let set = PromptSet {
        schema: 1,
        set: "bench-prompts-test".into(),
        sources: "testdata/corpus/SOURCES.toml".into(),
        separator: "\n\n".into(),
        nonce_tokens: 8,
        items,
    };
    (set, Mem(store))
}

fn tokenizer() -> syntropy_text::tokenizer::Tokenizer {
    let json = read("crates/text/tests/data/tokenizer/qwen3-shaped/tokenizer.json");
    syntropy_text::tokenizer::load(json.as_bytes()).expect("the fixture loads")
}

#[test]
fn the_synthetic_set_is_valid() {
    let (set, _) = synthetic();
    set.check(&sources(), &root()).expect("valid");
}

#[test]
fn prompts_have_exactly_the_requested_length_and_are_deterministic() {
    let (set, store) = synthetic();
    let tok = tokenizer();
    let a = Builder::new(&set, &store, &tok);
    // A second builder over a second tokenizer instance: nothing carries over.
    let tok2 = tokenizer();
    let b = Builder::new(&set, &store, &tok2);
    for c in [Category::Chat, Category::Code, Category::Agent] {
        let long = a.prompt(c, 4_096).expect("builds");
        assert_eq!(long.len(), 4_096, "{c:?}");
        for n in [1, 7, 128, 2_048, 4_096] {
            let p = a.prompt(c, n).expect("builds");
            assert_eq!(p.len(), n, "{c:?} {n}");
            assert_eq!(p, b.prompt(c, n).expect("builds"), "{c:?} {n}");
            assert_eq!(p[..], long[..n], "{c:?} {n}: a prefix of the longer prompt");
        }
        assert!(a.prompt(c, 0).expect("builds").is_empty());
    }
}

#[test]
fn items_are_joined_with_the_separator_and_cycle() {
    let (set, store) = synthetic();
    let tok = tokenizer();
    let b = Builder::new(&set, &store, &tok);
    let enc = |t: &str| Tokenize::encode(&tok, t).expect("encodes");
    let a = enc("User: hello there\n\nAssistant: Hi! How can I help?\n\n");
    let bb = enc("User: what time is it in Tokyo?\n\nAssistant: About 9pm.\n\n");
    let one_cycle: Vec<u32> = a.iter().chain(&bb).copied().collect();
    let p = b
        .prompt(Category::Chat, one_cycle.len() * 2 + 3)
        .expect("builds");
    assert_eq!(p[..one_cycle.len()], one_cycle[..]);
    assert_eq!(p[one_cycle.len()..2 * one_cycle.len()], one_cycle[..]);
    assert_eq!(p[2 * one_cycle.len()..], a[..3]);
}

#[test]
fn the_nonce_is_eight_chat_tokens_that_change_with_the_repetition() {
    let (set, store) = synthetic();
    let tok = tokenizer();
    let b = Builder::new(&set, &store, &tok);
    let stream = b.prompt(Category::Chat, 8 * 16).expect("builds");
    let mut seen = std::collections::BTreeSet::new();
    for rep in 0..16u32 {
        let n = b.nonce(rep).expect("builds");
        assert_eq!(n.len(), 8);
        assert_eq!(n[..], stream[rep as usize * 8..rep as usize * 8 + 8]);
        assert_eq!(n, b.nonce(rep).expect("builds"));
        seen.insert(n);
    }
    assert!(seen.len() > 8, "nonces repeat too often: {}", seen.len());
}

#[test]
fn agent_turns_extend_each_other() {
    let (set, store) = synthetic();
    let tok = tokenizer();
    let b = Builder::new(&set, &store, &tok);
    let turns = b.agent_turns("agent-a").expect("builds");
    assert_eq!(turns.len(), TURNS.len());
    for k in 1..turns.len() {
        assert!(turns[k].len() > turns[k - 1].len());
        assert_eq!(turns[k][..turns[k - 1].len()], turns[k - 1][..], "turn {k}");
        let delta = Tokenize::encode(&tok, TURNS[k]).expect("encodes");
        assert_eq!(turns[k][turns[k - 1].len()..], delta[..], "turn {k}");
    }
    assert!(matches!(
        b.agent_turns("chat-a"),
        Err(PromptError::Item { .. })
    ));
}

#[test]
fn changed_or_missing_text_is_refused() {
    let (set, mut store) = synthetic();
    let tok = tokenizer();
    let chat_a = set.items[0].clone();
    store.0.insert(
        chat_a.sha256.clone(),
        "User: hellO there\n\nAssistant: Hi! How can I help?".into(),
    );
    let b = Builder::new(&set, &store, &tok);
    let e = b.prompt(Category::Chat, 4).unwrap_err();
    assert!(e.to_string().contains("SHA-256"), "{e}");
    store.0.remove(&chat_a.sha256);
    let b = Builder::new(&set, &store, &tok);
    assert!(matches!(
        b.prompt(Category::Chat, 4),
        Err(PromptError::Item { .. })
    ));
    // A category with no item.
    assert_eq!(
        Builder::new(&set, &store, &tok).prompt(Category::Long, 4),
        Err(PromptError::Empty(Category::Long))
    );
    // A directory store checks the same way.
    let dir = std::env::temp_dir().join(format!("syntropy-bench-prompts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(&chat_a.sha256), "tampered").unwrap();
    let e = DirStore(dir.clone()).text(&chat_a).unwrap_err();
    assert!(e.to_string().contains("bytes"), "{e}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn check_refuses_each_broken_rule() {
    let (base, _) = synthetic();
    type Edit = fn(&mut PromptSet);
    let cases: [(&str, Edit, &str); 8] = [
        (
            "an unknown source",
            |s| s.items[0].source = "nowhere".into(),
            "not in SOURCES.toml",
        ),
        (
            "a licence not the source's",
            |s| s.items[0].licence = "MIT".into(),
            "not the source's",
        ),
        (
            "a source of another category",
            |s| {
                s.items[2].source = "taskmaster-1".into();
                s.items[2].licence = "CC-BY-4.0".into();
            },
            "does not serve `code`",
        ),
        (
            "a duplicate ID",
            |s| s.items[1].id = "chat-a".into(),
            "duplicate",
        ),
        (
            "a bad hash",
            |s| s.items[0].sha256 = "XYZ".into(),
            "64 lowercase hex",
        ),
        (
            "turns that do not add up",
            |s| s.items[3].turn_bytes = Some(vec![1, 2]),
            "do not add up",
        ),
        (
            "turns on a chat item",
            |s| s.items[0].turn_bytes = Some(vec![1]),
            "non-agent",
        ),
        ("schema 2", |s| s.schema = 2, "schema 2"),
    ];
    for (what, edit, says) in cases {
        let mut s = base.clone();
        edit(&mut s);
        let e = s.check(&sources(), &root()).unwrap_err();
        assert!(e.to_string().contains(says), "{what}: {e}");
    }
    // A licence the corpus policy does not allow, even where the source has it.
    let text = r#"
        [policy]
        allowed = ["MIT"]
        [[source]]
        id = "taskmaster-1"
        categories = { chat = "primary" }
        licence = "CC-BY-4.0"
    "#;
    let e = base
        .check(&Sources::parse(text).unwrap(), &root())
        .unwrap_err();
    assert!(e.to_string().contains("CC-BY-4.0 is not allowed"), "{e}");
    assert!(e.to_string().contains("no attribution file"), "{e}");
}

// ---- Real items ------------------------------------------------------------------------------

/// The fixture-cache BLAKE3 of the M0 tokenizer (`testdata/fixtures.toml`, qwen3-0.6b).
const QWEN3_TOKENIZER: &str = "b0cb923fc505fdf0a53f0287654fa26577d3f333d4134350da0a97664b228739";

#[test]
#[ignore = "needs the item text (`select.py --out`) and the fixture cache"]
fn real_items_build_exact_prompts() {
    let items = std::env::var("SYNTROPY_BENCH_PROMPTS").expect("set SYNTROPY_BENCH_PROMPTS");
    let cache = std::env::var("SYNTROPY_FIXTURE_CACHE").expect("set SYNTROPY_FIXTURE_CACHE");
    let bytes = std::fs::read(Path::new(&cache).join(QWEN3_TOKENIZER)).expect("tokenizer");
    assert_eq!(blake3::hash(&bytes).to_hex().as_str(), QWEN3_TOKENIZER);
    let tok = syntropy_text::tokenizer::load(&bytes).expect("loads");
    let set = shipped();
    let store = DirStore(PathBuf::from(items));
    for item in &set.items {
        store.text(item).unwrap_or_else(|e| panic!("{e}"));
    }
    let b = Builder::new(&set, &store, &tok);
    // RFC-0061 §3.1's standard grid prompt lengths.
    for c in [Category::Chat, Category::Code, Category::Agent] {
        let longest = b.prompt(c, 32_768).expect("builds");
        for n in [128, 2_048, 8_192, 32_768] {
            let p = b.prompt(c, n).expect("builds");
            assert_eq!(p.len(), n);
            assert_eq!(p[..], longest[..n]);
        }
    }
    for item in set.items(Category::Agent) {
        let turns = b.agent_turns(&item.id).expect("builds");
        assert!(turns.len() >= 8, "{}", item.id);
        for k in 1..turns.len() {
            assert_eq!(turns[k][..turns[k - 1].len()], turns[k - 1][..]);
        }
    }
    assert_eq!(b.nonce(3).expect("builds").len(), 8);
}
