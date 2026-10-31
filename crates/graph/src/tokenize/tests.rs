// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF tokenizer conversion on synthetic metadata (COMP-E17-T03).
//!
//! The conformance tests build a source `tokenizer.json` shaped as the source repositories ship
//! them, export it to GGUF metadata as the exporter does (tokens in ID order, token types from
//! the added tokens, merges as `"left right"`), convert the metadata back, and encode the
//! project corpus (`tests/data/tokenize/corpus.json`) through both with the engine: the token
//! IDs must be equal. Other tests check token IDs worked out by hand, so they do not lean on
//! the export. The run on a real GGUF file is `tests/tokenize_real.rs`.

use proptest::prelude::*;
use serde_json::{Value, json};
use syntropy_core::ErrorClass;
use syntropy_manifest::gguf::{GgufHeader, Limits};
use syntropy_manifest::range::{RangeError, RangeSource};
use tokenizers::Tokenizer;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::tokenizer::{NormalizedString, Normalizer};

use super::*;

// ---- GGUF metadata -----------------------------------------------------------------------------

/// A metadata value to write.
#[derive(Clone, Debug)]
pub(crate) enum Kv {
    Str(String),
    U32(u32),
    I32(i32),
    Bool(bool),
    Strs(Vec<String>),
    I32s(Vec<i32>),
    F32s(Vec<f32>),
    /// A string array holding raw bytes, which need not be UTF-8.
    RawStrs(Vec<Vec<u8>>),
}

fn put_str(out: &mut Vec<u8>, s: &[u8]) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s);
}

impl Kv {
    /// The value's type ID and bytes, as GGUF v3 lays them out.
    fn encode(&self, out: &mut Vec<u8>) {
        let array = |out: &mut Vec<u8>, elem: u32, n: usize| {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&elem.to_le_bytes());
            out.extend_from_slice(&(n as u64).to_le_bytes());
        };
        match self {
            Self::Str(s) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                put_str(out, s.as_bytes());
            }
            Self::U32(v) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Self::I32(v) => {
                out.extend_from_slice(&5u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Self::Bool(v) => {
                out.extend_from_slice(&7u32.to_le_bytes());
                out.push(u8::from(*v));
            }
            Self::Strs(v) => {
                array(out, 8, v.len());
                for s in v {
                    put_str(out, s.as_bytes());
                }
            }
            Self::RawStrs(v) => {
                array(out, 8, v.len());
                for s in v {
                    put_str(out, s);
                }
            }
            Self::I32s(v) => {
                array(out, 5, v.len());
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            Self::F32s(v) => {
                array(out, 6, v.len());
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
}

/// The key-value pairs encoded, for a caller that writes its own GGUF file.
pub(crate) fn encode_kvs(kvs: &[(String, Kv)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (k, v) in kvs {
        put_str(&mut out, k.as_bytes());
        v.encode(&mut out);
    }
    out
}

struct Mem(Vec<u8>);

impl RangeSource for Mem {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        let start = usize::try_from(offset).unwrap();
        buf.copy_from_slice(&self.0[start..start + buf.len()]);
        Ok(())
    }
}

/// A GGUF file with no tensors and these key-value pairs, read back.
fn header(kvs: &[(String, Kv)]) -> GgufHeader {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    out.extend_from_slice(&encode_kvs(kvs));
    GgufHeader::read(Mem(out), &Limits::default()).expect("a valid synthetic GGUF")
}

fn convert(kvs: &[(String, Kv)]) -> Result<Option<TokenizerSpec>, TokenizeError> {
    from_gguf(&header(kvs).metadata)
}

fn spec(kvs: &[(String, Kv)]) -> TokenizerSpec {
    convert(kvs)
        .expect("the tokenizer converts")
        .expect("the file has a tokenizer")
}

fn engine(spec: &TokenizerSpec) -> Tokenizer {
    Tokenizer::from_bytes(spec.model()).expect("the engine loads the converted model")
}

/// Replaces or adds `key`.
fn set(kvs: &mut Vec<(String, Kv)>, key: &str, v: Kv) {
    kvs.retain(|(k, _)| k != key);
    kvs.push((key.to_owned(), v));
}

fn unset(kvs: &mut Vec<(String, Kv)>, key: &str) {
    kvs.retain(|(k, _)| k != key);
}

// ---- Source tokenizers and the export ----------------------------------------------------------

/// The byte-level alphabet: each byte's printable stand-in, as byte-level BPE vocabularies
/// spell them.
fn byte_char(b: u8) -> char {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    if printable(b) {
        return char::from(b);
    }
    let shift = (0..b).filter(|&x| !printable(x)).count();
    char::from_u32(256 + u32::try_from(shift).unwrap()).unwrap()
}

fn byte_level(s: &str) -> String {
    s.bytes().map(byte_char).collect()
}

/// An added token of a source `tokenizer.json`.
fn added(id: u32, content: &str, special: bool) -> Value {
    json!({
        "id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
        "normalized": false, "special": special,
    })
}

/// A byte-level BPE source in the layout of the M0 repositories' `tokenizer.json`: the byte
/// alphabet, merges of common pieces, a whole word reachable only with `ignore_merges`, and
/// special and non-special added tokens.
fn byte_level_source(pre: &PreTokenizer, add_bos: bool) -> Value {
    let mut vocab: Vec<String> = (0..=255u8).map(|b| byte_char(b).to_string()).collect();
    let pairs = [
        (" ", "t"),
        ("h", "e"),
        (" t", "he"),
        ("i", "n"),
        (" ", "a"),
        ("e", "r"),
        ("o", "n"),
        (" ", "s"),
        ("r", "e"),
        ("a", "n"),
        (" a", "n"),
        (" an", "d"),
        ("1", "2"),
        ("12", "3"),
        ("4", "5"),
        (" ", " "),
        ("  ", "  "),
        ("\n", "\n"),
        ("'", "s"),
        ("'", "ll"),
        ("l", "l"),
        ("é", ""),
        ("i", "s"),
        (" ", "is"),
        ("o", "u"),
        ("o", "r"),
        (" w", "or"),
        (" ", "w"),
        ("今", ""),
        ("天", ""),
        ("今", "天"),
    ];
    let mut merges = Vec::new();
    for (l, r) in pairs {
        if r.is_empty() {
            // A multi-byte character merged from its bytes.
            let bytes = byte_level(l);
            let mut chars: Vec<String> = bytes.chars().map(String::from).collect();
            while chars.len() > 1 {
                let joined = format!("{}{}", chars[0], chars[1]);
                merges.push(json!([chars[0], chars[1]]));
                if !vocab.contains(&joined) {
                    vocab.push(joined.clone());
                }
                chars.splice(0..2, [joined]);
            }
            continue;
        }
        let (l, r) = (byte_level(l), byte_level(r));
        let joined = format!("{l}{r}");
        merges.push(json!([l, r]));
        if !vocab.contains(&joined) {
            vocab.push(joined);
        }
    }
    // A word no merge reaches: taken whole only under `ignore_merges`.
    vocab.push(byte_level(" quick"));
    vocab.push(byte_level("quick"));

    let n = u32::try_from(vocab.len()).unwrap();
    let specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];
    let plain = [
        "<think>",
        "</think>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ];
    let mut added_tokens = Vec::new();
    for (i, t) in specials.iter().enumerate() {
        added_tokens.push(added(n + i as u32, t, true));
    }
    for (i, t) in plain.iter().enumerate() {
        added_tokens.push(added(n + 3 + i as u32, t, false));
    }
    let model_vocab: serde_json::Map<String, Value> = (0u32..)
        .zip(&vocab)
        .map(|(id, t)| (t.clone(), id.into()))
        .collect();

    // GPT-2's own file runs the split inside the byte-level step; the others split first.
    let pre_tokenizer = if pre.name == "gpt-2" {
        json!({ "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true })
    } else {
        json!({ "type": "Sequence", "pretokenizers": [
            { "type": "Split", "pattern": { "Regex": pre.regex }, "behavior": "Isolated", "invert": false },
            { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false },
        ]})
    };
    let byte_level_step = json!({ "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true });
    let post_processor = if add_bos {
        json!({ "type": "Sequence", "processors": [byte_level_step, {
            "type": "TemplateProcessing",
            "single": [{ "SpecialToken": { "id": "<|endoftext|>", "type_id": 0 } }, { "Sequence": { "id": "A", "type_id": 0 } }],
            "pair": [{ "SpecialToken": { "id": "<|endoftext|>", "type_id": 0 } }, { "Sequence": { "id": "A", "type_id": 0 } },
                     { "SpecialToken": { "id": "<|endoftext|>", "type_id": 1 } }, { "Sequence": { "id": "B", "type_id": 1 } }],
            "special_tokens": { "<|endoftext|>": { "id": "<|endoftext|>", "ids": [n], "tokens": ["<|endoftext|>"] } },
        }]})
    } else {
        byte_level_step
    };
    json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added_tokens,
        "normalizer": if pre.nfc { json!({ "type": "NFC" }) } else { Value::Null },
        "pre_tokenizer": pre_tokenizer,
        "post_processor": post_processor,
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
            "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false,
            "ignore_merges": pre.ignore_merges, "vocab": model_vocab, "merges": merges,
        },
    })
}

/// Whether the exporter writes an added token as control: special, or spelled `<|…|>`.
fn exported_as_control(content: &str, special: bool) -> bool {
    special || (content.starts_with("<|") && content.ends_with("|>"))
}

/// A byte-level source's GGUF metadata, as the exporter writes it. BOS is the first added
/// token, `<|endoftext|>`, as in the M0 files; `add_bos_token` is left out, so the
/// pre-tokenizer's default applies.
fn export_byte_level(source: &Value, pre: &str) -> Vec<(String, Kv)> {
    let vocab = source["model"]["vocab"].as_object().unwrap();
    let added = source["added_tokens"].as_array().unwrap();
    let n = vocab.len() + added.len();
    let mut tokens = vec![String::new(); n];
    let mut types = vec![1i32; n];
    for (t, id) in vocab {
        tokens[usize::try_from(id.as_u64().unwrap()).unwrap()].clone_from(t);
    }
    for a in added {
        let id = usize::try_from(a["id"].as_u64().unwrap()).unwrap();
        let content = a["content"].as_str().unwrap();
        tokens[id] = content.to_owned();
        types[id] = if exported_as_control(content, a["special"].as_bool().unwrap()) {
            3
        } else {
            4
        };
    }
    let merges = source["model"]["merges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| format!("{} {}", m[0].as_str().unwrap(), m[1].as_str().unwrap()))
        .collect();
    vec![
        (MODEL.into(), Kv::Str("gpt2".into())),
        (PRE.into(), Kv::Str(pre.into())),
        (TOKENS.into(), Kv::Strs(tokens)),
        (TOKEN_TYPE.into(), Kv::I32s(types)),
        (MERGES.into(), Kv::Strs(merges)),
        (BOS_ID.into(), Kv::U32(u32::try_from(vocab.len()).unwrap())),
    ]
}

/// The project corpus (`tests/data/tokenize/corpus.json`).
pub(crate) fn corpus() -> Vec<String> {
    let doc: Value =
        serde_json::from_str(include_str!("../../tests/data/tokenize/corpus.json")).unwrap();
    doc["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

fn ids(t: &Tokenizer, text: &str, add_special: bool) -> Vec<u32> {
    t.encode(text, add_special)
        .unwrap_or_else(|e| panic!("{text:?} encodes: {e}"))
        .get_ids()
        .to_vec()
}

fn nfc(s: &str) -> String {
    let mut n = NormalizedString::from(s);
    NFC.normalize(&mut n).unwrap();
    n.get().to_owned()
}

/// Encodes every corpus item through both tokenizers, with and without the post-processor's
/// specials, and checks the IDs agree; returns how many items were compared.
fn assert_same_tokens(source: &Tokenizer, converted: &Tokenizer, corpus: &[String]) -> usize {
    for text in corpus {
        for add_special in [false, true] {
            assert_eq!(
                ids(converted, text, add_special),
                ids(source, text, add_special),
                "{text:?}, add_special {add_special}"
            );
        }
    }
    corpus.len()
}

// ---- Conformance: byte-level BPE ---------------------------------------------------------------

#[test]
fn every_pre_tokenizer_converts_token_for_token_on_the_corpus() {
    let corpus = corpus();
    for pre in &PRE_TOKENIZERS {
        let src = byte_level_source(pre, pre.add_bos);
        let source = Tokenizer::from_bytes(serde_json::to_vec(&src).unwrap()).unwrap();
        let spec = spec(&export_byte_level(&src, pre.name));
        let converted = engine(&spec);
        assert_eq!(
            assert_same_tokens(&source, &converted, &corpus),
            corpus.len(),
            "{}",
            pre.name
        );
        assert_eq!(spec.add_bos(), pre.add_bos, "{}", pre.name);
    }
}

#[test]
fn byte_level_bpe_round_trips_every_corpus_string() {
    for pre in &PRE_TOKENIZERS {
        let src = byte_level_source(pre, false);
        let converted = engine(&spec(&export_byte_level(&src, pre.name)));
        for text in corpus() {
            let back = converted
                .decode(&ids(&converted, &text, false), false)
                .unwrap();
            // A pre-tokenizer that normalizes to NFC returns the NFC form, as the source does.
            let expected = if pre.nfc { nfc(&text) } else { text.clone() };
            assert_eq!(back, expected, "{}: {text:?}", pre.name);
        }
    }
}

#[test]
fn the_conversion_keeps_the_split_rule_of_each_name() {
    // Hand-worked: GPT-2's split keeps a run of digits whole, llama-bpe cuts it into threes,
    // qwen2 into single digits. "12345" has merges 1+2, 12+3 and 4+5.
    let digit = |pre: &str, text: &str| -> Vec<String> {
        let table = pre_tokenizer(pre).unwrap();
        let src = byte_level_source(table, false);
        let t = engine(&spec(&export_byte_level(&src, pre)));
        t.encode(text, false).unwrap().get_tokens().to_vec()
    };
    assert_eq!(digit("gpt-2", "12345"), ["123", "45"]);
    assert_eq!(digit("llama-bpe", "12345"), ["123", "45"]);
    assert_eq!(digit("qwen2", "12345"), ["1", "2", "3", "4", "5"]);
    assert_eq!(digit("gpt-2", "1234567"), ["123", "45", "6", "7"]);
    assert_eq!(digit("llama-bpe", "1234567"), ["123", "45", "6", "7"]);
}

#[test]
fn ignore_merges_follows_the_name() {
    // " quick" is a token no merge reaches: taken whole only where the name sets ignore_merges.
    for (pre, whole) in [("gpt-2", false), ("llama-bpe", true), ("qwen2", false)] {
        let src = byte_level_source(pre_tokenizer(pre).unwrap(), false);
        let t = engine(&spec(&export_byte_level(&src, pre)));
        let tokens = t.encode("the quick", false).unwrap().get_tokens().to_vec();
        assert_eq!(
            tokens.contains(&"Ġquick".to_owned()),
            whole,
            "{pre}: {tokens:?}"
        );
    }
}

#[test]
fn nfc_follows_the_name() {
    // "e" + U+0301 is "é" after NFC, which has merges of its own bytes.
    for (pre, composed) in [("gpt-2", false), ("llama-bpe", false), ("qwen2", true)] {
        let src = byte_level_source(pre_tokenizer(pre).unwrap(), false);
        let t = engine(&spec(&export_byte_level(&src, pre)));
        let tokens = t.encode("e\u{301}", false).unwrap().get_tokens().to_vec();
        assert_eq!(tokens == [byte_level("é")], composed, "{pre}: {tokens:?}");
    }
}

#[test]
fn added_tokens_follow_the_token_types() {
    let src = byte_level_source(pre_tokenizer("qwen2").unwrap(), false);
    let kvs = export_byte_level(&src, "qwen2");
    let spec = spec(&kvs);
    let model: Value = serde_json::from_slice(spec.model()).unwrap();
    let added: Vec<(String, bool)> = model["added_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            (
                a["content"].as_str().unwrap().to_owned(),
                a["special"].as_bool().unwrap(),
            )
        })
        .collect();
    let expect: Vec<(String, bool)> = [
        ("<|endoftext|>", true),
        ("<|im_start|>", true),
        ("<|im_end|>", true),
        ("<think>", false),
        ("</think>", false),
        // Not special in the source, but the exporter writes `<|…|>` spellings as control.
        ("<|fim_prefix|>", true),
        ("<|fim_middle|>", true),
        ("<|fim_suffix|>", true),
    ]
    .into_iter()
    .map(|(t, s)| (t.to_owned(), s))
    .collect();
    assert_eq!(added, expect);
    // Added tokens are matched on their text, and are not in the BPE vocabulary.
    let t = engine(&spec);
    let vocab_len = model["model"]["vocab"].as_object().unwrap().len() as u32;
    assert_eq!(
        ids(&t, "<|im_start|>x<think>", false),
        [vocab_len + 1, u32::from(b'x'), vocab_len + 3]
    );
    assert!(model["model"]["vocab"].get("<|im_start|>").is_none());
}

#[test]
fn unused_tokens_have_no_entry_and_keep_every_other_id() {
    let src = byte_level_source(pre_tokenizer("qwen2").unwrap(), false);
    let mut kvs = export_byte_level(&src, "qwen2");
    let (Kv::Strs(mut tokens), Kv::I32s(mut types)) = (kvs[2].1.clone(), kvs[3].1.clone()) else {
        unreachable!()
    };
    let n = tokens.len() as u32;
    for i in 0..3 {
        tokens.push(format!("[PAD{}]", n + i));
        types.push(5);
    }
    // An unused token may repeat another token's text.
    tokens.push("<think>".to_owned());
    types.push(5);
    set(&mut kvs, TOKENS, Kv::Strs(tokens));
    set(&mut kvs, TOKEN_TYPE, Kv::I32s(types));
    let t = engine(&spec(&kvs));
    assert_eq!(t.token_to_id(&format!("[PAD{n}]")), None);
    assert_eq!(t.token_to_id("<think>"), Some(n - 5));
    assert_eq!(t.id_to_token(n + 1), None);
}

// ---- Conformance: SentencePiece-style BPE ------------------------------------------------------

/// A SentencePiece-style vocabulary: (text, score, type). IDs are positions. Scores make
/// `bc` outrank `ab` though `ab` has the lower ID, so score order and ID order disagree.
fn spm_vocab(bytes: bool) -> Vec<(String, f32, i32)> {
    let mut v: Vec<(String, f32, i32)> = vec![
        ("<unk>".into(), 0.0, 2),
        ("<s>".into(), 0.0, 3),
        ("</s>".into(), 0.0, 3),
    ];
    if bytes {
        v.extend((0..=255u8).map(|b| (format!("<0x{b:02X}>"), 0.0, 6)));
    }
    for (t, s) in [
        ("▁", -1.0),
        ("a", -2.0),
        ("b", -3.0),
        ("c", -4.0),
        ("ab", -6.0),
        ("bc", -5.0),
        ("▁a", -7.0),
        ("abc", -8.0),
        ("▁▁", -9.0),
        ("x", -10.0),
        ("y", -11.0),
        ("z", -12.0),
        ("xy", -13.0),
        ("yz", -13.0),
        ("e", -14.0),
        ("h", -15.0),
        ("l", -16.0),
        ("o", -17.0),
        ("he", -18.0),
        ("ll", -19.0),
        ("hell", -20.0),
        ("hello", -21.0),
        ("▁hello", -22.0),
        ("é", -23.0),
    ] {
        v.push((t.into(), s, 1));
    }
    v.push(("[INST]".into(), 0.0, 4));
    v
}

fn spm_kvs(vocab: &[(String, f32, i32)]) -> Vec<(String, Kv)> {
    vec![
        (MODEL.into(), Kv::Str("llama".into())),
        (
            TOKENS.into(),
            Kv::Strs(vocab.iter().map(|v| v.0.clone()).collect()),
        ),
        (SCORES.into(), Kv::F32s(vocab.iter().map(|v| v.1).collect())),
        (
            TOKEN_TYPE.into(),
            Kv::I32s(vocab.iter().map(|v| v.2).collect()),
        ),
        (UNK_ID.into(), Kv::U32(0)),
        (BOS_ID.into(), Kv::U32(1)),
        (EOS_ID.into(), Kv::U32(2)),
    ]
}

fn spm_tokens(t: &Tokenizer, text: &str) -> Vec<String> {
    t.encode(text, false).unwrap().get_tokens().to_vec()
}

#[test]
fn sentencepiece_merges_apply_in_score_order() {
    // Hand-worked on "▁abc": ▁+a scores -7, a+b -6, b+c -5. b+c goes first, then ▁+a; a+bc
    // would be "abc" (-8), but "a" is already merged into "▁a", and "▁abc" is no token.
    let t = engine(&spec(&spm_kvs(&spm_vocab(true))));
    assert_eq!(spm_tokens(&t, "abc"), ["▁a", "bc"]);
    assert_eq!(spm_tokens(&t, "hello"), ["▁hello"]);
    assert_eq!(spm_tokens(&t, "a hello"), ["▁a", "▁hello"]);
    // "▁xyz": xy and yz tie at -13; the lower ID, xy, goes first.
    assert_eq!(spm_tokens(&t, "xyz"), ["▁", "xy", "z"]);
}

#[test]
fn equal_scores_merge_in_id_order() {
    let mut vocab = spm_vocab(true);
    let xy = vocab.iter().position(|v| v.0 == "xy").unwrap();
    let yz = vocab.iter().position(|v| v.0 == "yz").unwrap();
    vocab.swap(xy, yz);
    let t = engine(&spec(&spm_kvs(&vocab)));
    assert_eq!(spm_tokens(&t, "xyz"), ["▁", "x", "yz"]);
}

#[test]
fn sentencepiece_falls_back_to_bytes_and_fuses_unknowns() {
    let t = engine(&spec(&spm_kvs(&spm_vocab(true))));
    // "ü" is no token: its UTF-8 bytes C3 BC.
    assert_eq!(spm_tokens(&t, "aü"), ["▁a", "<0xC3>", "<0xBC>"]);
    assert_eq!(t.decode(&ids(&t, "aü é", false), false).unwrap(), "aü é");
    // Without byte tokens, a run of unknown characters is one unknown token.
    let t = engine(&spec(&spm_kvs(&spm_vocab(false))));
    assert_eq!(spm_tokens(&t, "aüü"), ["▁a", "<unk>"]);
}

#[test]
fn the_space_prefix_follows_its_key() {
    let mut kvs = spm_kvs(&spm_vocab(true));
    let t = engine(&spec(&kvs));
    assert_eq!(spm_tokens(&t, "abc  a"), ["▁a", "bc", "▁", "▁a"]);
    assert_eq!(
        t.decode(&ids(&t, "abc  a", false), false).unwrap(),
        "abc  a"
    );
    set(&mut kvs, ADD_SPACE_PREFIX, Kv::Bool(false));
    let t = engine(&spec(&kvs));
    assert_eq!(spm_tokens(&t, "abc a"), ["abc", "▁a"]);
    assert_eq!(t.decode(&ids(&t, "abc a", false), false).unwrap(), "abc a");
}

#[test]
fn sentencepiece_adds_bos_by_default() {
    let mut kvs = spm_kvs(&spm_vocab(true));
    let spec_default = spec(&kvs);
    assert!(spec_default.add_bos());
    assert!(!spec_default.add_eos());
    assert_eq!(ids(&engine(&spec_default), "a", true), [1, 265]);
    set(&mut kvs, ADD_BOS, Kv::Bool(false));
    set(&mut kvs, ADD_EOS, Kv::Bool(true));
    let s = spec(&kvs);
    assert_eq!(ids(&engine(&s), "a", true), [265, 2]);
    assert_eq!(ids(&engine(&s), "a", false), [265]);
}

/// The SentencePiece-style vocabulary as a source repository's `tokenizer.json` holds it: the
/// legacy layout, with merges written out in score order by hand.
fn spm_source() -> Value {
    let vocab = spm_vocab(true);
    let model_vocab: serde_json::Map<String, Value> = (0u32..)
        .zip(&vocab)
        .map(|(id, v)| (v.0.clone(), id.into()))
        .collect();
    let merges = [
        ["b", "c"],
        ["a", "b"],
        ["▁", "a"],
        ["a", "bc"],
        ["ab", "c"],
        ["▁", "▁"],
        ["x", "y"],
        ["y", "z"],
        ["h", "e"],
        ["l", "l"],
        ["he", "ll"],
        ["hell", "o"],
        ["▁", "hello"],
    ];
    json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [added(0, "<unk>", true), added(1, "<s>", true), added(2, "</s>", true),
                         added(u32::try_from(vocab.len() - 1).unwrap(), "[INST]", false)],
        "normalizer": { "type": "Sequence", "normalizers": [
            { "type": "Prepend", "prepend": "▁" },
            { "type": "Replace", "pattern": { "String": " " }, "content": "▁" },
        ]},
        "pre_tokenizer": null,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{ "SpecialToken": { "id": "<s>", "type_id": 0 } }, { "Sequence": { "id": "A", "type_id": 0 } }],
            "pair": [{ "SpecialToken": { "id": "<s>", "type_id": 0 } }, { "Sequence": { "id": "A", "type_id": 0 } },
                     { "SpecialToken": { "id": "<s>", "type_id": 1 } }, { "Sequence": { "id": "B", "type_id": 1 } }],
            "special_tokens": { "<s>": { "id": "<s>", "ids": [1], "tokens": ["<s>"] } },
        },
        "decoder": { "type": "Sequence", "decoders": [
            { "type": "Replace", "pattern": { "String": "▁" }, "content": " " },
            { "type": "ByteFallback" }, { "type": "Fuse" },
            { "type": "Strip", "content": " ", "start": 1, "stop": 0 },
        ]},
        "model": {
            "type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
            "vocab": model_vocab, "merges": merges,
        },
    })
}

#[test]
fn sentencepiece_converts_token_for_token_on_the_corpus() {
    let source = Tokenizer::from_bytes(serde_json::to_vec(&spm_source()).unwrap()).unwrap();
    let converted = engine(&spec(&spm_kvs(&spm_vocab(true))));
    let mut corpus = corpus();
    corpus.extend(
        [
            "abc",
            "a bc abc",
            "hello hello",
            "xyz zyx",
            "  ab  ",
            "<s>abc</s>",
            "[INST] hello [INST]",
            "é ü",
        ]
        .map(String::from),
    );
    assert_same_tokens(&source, &converted, &corpus);
    for text in &corpus {
        let ids = ids(&converted, text, false);
        assert_eq!(
            converted.decode(&ids, false).unwrap(),
            source.decode(&ids, false).unwrap(),
            "{text:?}"
        );
    }
}

#[test]
fn the_derived_merges_are_listed_in_score_order() {
    let spec = spec(&spm_kvs(&spm_vocab(true)));
    let model: Value = serde_json::from_slice(spec.model()).unwrap();
    let merges: Vec<String> = model["model"]["merges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| format!("{} {}", m[0].as_str().unwrap(), m[1].as_str().unwrap()))
        .collect();
    assert_eq!(
        merges,
        [
            "b c",
            "a b",
            "▁ a",
            "a bc",
            "ab c",
            "▁ ▁",
            "x y",
            "y z",
            "h e",
            "l l",
            "he ll",
            "hell o",
            "▁ hello"
        ]
    );
}

// ---- Specials, flags and template --------------------------------------------------------------

fn qwen_like() -> Vec<(String, Kv)> {
    let src = byte_level_source(pre_tokenizer("qwen2").unwrap(), false);
    export_byte_level(&src, "qwen2")
}

/// The ID of the `i`th added token of [`byte_level_source`].
fn added_id(i: u32) -> u32 {
    let src = byte_level_source(pre_tokenizer("qwen2").unwrap(), false);
    u32::try_from(src["model"]["vocab"].as_object().unwrap().len()).unwrap() + i
}

#[test]
fn specials_come_from_their_keys() {
    let mut kvs = qwen_like();
    let (eot, im_end, im_start) = (added_id(0), added_id(2), added_id(1));
    set(&mut kvs, BOS_ID, Kv::U32(eot));
    set(&mut kvs, EOS_ID, Kv::U32(im_end));
    set(&mut kvs, PAD_ID, Kv::U32(eot));
    set(&mut kvs, UNK_ID, Kv::U32(5));
    set(&mut kvs, EOT_ID, Kv::U32(im_start));
    set(&mut kvs, EOM_ID, Kv::U32(added_id(3)));
    set(&mut kvs, CHAT_TEMPLATE, Kv::Str("{{ messages }}".into()));
    let s = spec(&kvs);
    assert_eq!(
        *s.specials(),
        SpecialTokens {
            bos: Some(eot),
            eos: Some(im_end),
            end_of_turn: vec![im_start, im_end, added_id(3)],
            pad: Some(eot),
            unk: Some(5),
            // By spelling: the file names no FIM keys.
            fim_prefix: Some(added_id(5)),
            fim_middle: Some(added_id(6)),
            fim_suffix: Some(added_id(7)),
        }
    );
    assert!(!s.add_bos() && !s.add_eos());
    assert_eq!(s.chat_template(), Some("{{ messages }}"));
    assert_eq!(*s.template_source(), TemplateSource::Shipped);
}

#[test]
fn fim_keys_win_over_spellings_in_either_form() {
    for keys in FIM_KEYS {
        let mut kvs = qwen_like();
        for (k, id) in keys.iter().zip([10, 11, 12]) {
            set(&mut kvs, k, Kv::U32(id));
        }
        let s = spec(&kvs);
        let sp = s.specials();
        assert_eq!(
            (sp.fim_prefix, sp.fim_middle, sp.fim_suffix),
            (Some(10), Some(11), Some(12))
        );
    }
}

#[test]
fn no_fim_tokens_leave_fim_unset() {
    let mut vocab = spm_vocab(true);
    vocab.push(("<|fim_prefix|>".into(), 0.0, 3));
    vocab.push(("<|fim_middle|>".into(), 0.0, 3));
    // A normal token spelled like the suffix is not an added token.
    vocab.push(("<|fim_suffix|>".into(), 0.0, 1));
    let s = spec(&spm_kvs(&vocab));
    assert_eq!(s.specials().fim_prefix, None);
    assert_eq!(s.specials().fim_suffix, None);
}

#[test]
fn add_bos_without_a_key_takes_the_pre_tokenizers_default() {
    for pre in &PRE_TOKENIZERS {
        let mut kvs = export_byte_level(&byte_level_source(pre, false), pre.name);
        let s = spec(&kvs);
        assert_eq!(s.add_bos(), pre.add_bos, "{}", pre.name);
        let first = ids(&engine(&s), "the", true)[0];
        assert_eq!(first == added_id(0), pre.add_bos, "{}", pre.name);
        set(&mut kvs, ADD_BOS, Kv::Bool(!pre.add_bos));
        assert_eq!(spec(&kvs).add_bos(), !pre.add_bos, "{}", pre.name);
    }
}

#[test]
fn the_conversion_is_deterministic_and_the_id_covers_the_template() {
    let mut kvs = qwen_like();
    let a = spec(&kvs);
    assert_eq!(a, spec(&kvs));
    assert_eq!(*a.template_source(), TemplateSource::None);
    set(&mut kvs, CHAT_TEMPLATE, Kv::Str("t".into()));
    let b = spec(&kvs);
    assert_eq!(a.model(), b.model());
    assert_ne!(a.id(), b.id());
}

#[test]
fn a_file_without_a_tokenizer_model_has_no_tokenizer() {
    let mut kvs = qwen_like();
    unset(&mut kvs, MODEL);
    assert_eq!(convert(&kvs), Ok(None));
    assert_eq!(convert(&[]), Ok(None));
}

// ---- Refusals ----------------------------------------------------------------------------------

fn unsupported(kvs: &[(String, Kv)]) -> (&'static str, Option<String>) {
    match convert(kvs) {
        Err(e @ TokenizeError::Unsupported { .. }) => {
            assert_eq!(e.code(), Some(6));
            assert_eq!(e.class(), ErrorClass::Unsupported);
            assert!(e.to_string().starts_with("TokenizerUnsupported (Graph 6)"));
            let TokenizeError::Unsupported { key, value } = e else {
                unreachable!()
            };
            (key, value)
        }
        other => panic!("expected TokenizerUnsupported, got {other:?}"),
    }
}

#[test]
fn an_unknown_pre_tokenizer_name_is_refused() {
    let mut kvs = qwen_like();
    for name in ["qwen35", "", "QWEN2", "default", "llama3"] {
        set(&mut kvs, PRE, Kv::Str(name.into()));
        assert_eq!(unsupported(&kvs), (PRE, Some(name.to_owned())));
    }
    unset(&mut kvs, PRE);
    assert_eq!(unsupported(&kvs), (PRE, None));
}

#[test]
fn an_unknown_tokenizer_model_is_refused() {
    let mut kvs = qwen_like();
    for name in ["bert", "t5", "rwkv", "none", "GPT2"] {
        set(&mut kvs, MODEL, Kv::Str(name.into()));
        assert_eq!(unsupported(&kvs), (MODEL, Some(name.to_owned())));
    }
}

#[test]
fn unmodelled_sentencepiece_settings_are_refused() {
    let base = spm_kvs(&spm_vocab(true));
    let mut kvs = base.clone();
    set(&mut kvs, PRE, Kv::Str("qwen2".into()));
    assert_eq!(unsupported(&kvs).0, PRE);
    let mut kvs = base.clone();
    set(&mut kvs, PRE, Kv::Str("default".into()));
    assert!(convert(&kvs).is_ok());
    let mut kvs = base.clone();
    set(&mut kvs, REMOVE_EXTRA_WHITESPACES, Kv::Bool(true));
    assert_eq!(unsupported(&kvs).0, REMOVE_EXTRA_WHITESPACES);
    set(&mut kvs, REMOVE_EXTRA_WHITESPACES, Kv::Bool(false));
    assert!(convert(&kvs).is_ok());
    let mut kvs = base;
    set(&mut kvs, PRECOMPILED_CHARSMAP, Kv::I32s(vec![]));
    assert_eq!(unsupported(&kvs).0, PRECOMPILED_CHARSMAP);
}

fn invalid(kvs: &[(String, Kv)]) -> (String, String) {
    match convert(kvs) {
        Err(e @ TokenizeError::Invalid { .. }) => {
            assert_eq!(e.code(), None);
            assert_eq!(e.class(), ErrorClass::Integrity);
            let TokenizeError::Invalid { key, reason } = e else {
                unreachable!()
            };
            (key, reason)
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

#[test]
fn malformed_vocabularies_are_invalid() {
    let base = qwen_like();
    let (Kv::Strs(tokens), Kv::I32s(types)) = (base[2].1.clone(), base[3].1.clone()) else {
        unreachable!()
    };
    let case = |key: &str, v: Option<Kv>| {
        let mut kvs = base.clone();
        match v {
            Some(v) => set(&mut kvs, key, v),
            None => unset(&mut kvs, key),
        }
        invalid(&kvs)
    };
    assert_eq!(case(TOKENS, None).0, TOKENS);
    assert_eq!(case(TOKENS, Some(Kv::Strs(vec![]))).0, TOKENS);
    assert_eq!(case(TOKENS, Some(Kv::I32s(vec![1]))).0, TOKENS);
    assert_eq!(case(TOKEN_TYPE, None).0, TOKEN_TYPE);
    assert_eq!(
        case(TOKEN_TYPE, Some(Kv::I32s(types[1..].to_vec()))).0,
        TOKEN_TYPE
    );
    for bad in [0, 7, -1] {
        let mut t = types.clone();
        t[3] = bad;
        let (key, reason) = case(TOKEN_TYPE, Some(Kv::I32s(t)));
        assert_eq!(key, TOKEN_TYPE);
        assert!(reason.contains("token 3"), "{reason}");
    }
    let mut raw: Vec<Vec<u8>> = tokens.iter().map(|t| t.clone().into_bytes()).collect();
    raw[7] = vec![0xFF, 0xFE];
    assert!(case(TOKENS, Some(Kv::RawStrs(raw))).1.contains("token 7"));
    let mut dup = tokens.clone();
    dup[9] = dup[8].clone();
    assert!(
        case(TOKENS, Some(Kv::Strs(dup)))
            .1
            .contains("tokens 8 and 9")
    );
    assert_eq!(case(MODEL, Some(Kv::U32(1))).0, MODEL);
}

#[test]
fn malformed_merges_are_invalid() {
    let case = |merges: Option<Vec<&str>>| {
        let mut kvs = qwen_like();
        match merges {
            Some(m) => set(
                &mut kvs,
                MERGES,
                Kv::Strs(m.into_iter().map(String::from).collect()),
            ),
            None => unset(&mut kvs, MERGES),
        }
        invalid(&kvs)
    };
    assert_eq!(case(None).0, MERGES);
    assert!(case(Some(vec!["h e", "he"])).1.contains("merge 1"));
    assert!(case(Some(vec!["h e r"])).1.contains("more than two"));
    assert!(case(Some(vec!["h \u{1}"])).1.contains("not a token"));
    // Both parts are tokens, but their join is not.
    assert!(case(Some(vec!["q z"])).1.contains("\"qz\""));
}

#[test]
fn malformed_scores_are_invalid() {
    let vocab = spm_vocab(true);
    let case = |scores: Option<Kv>| {
        let mut kvs = spm_kvs(&vocab);
        match scores {
            Some(s) => set(&mut kvs, SCORES, s),
            None => unset(&mut kvs, SCORES),
        }
        invalid(&kvs)
    };
    let scores: Vec<f32> = vocab.iter().map(|v| v.1).collect();
    assert_eq!(case(None).0, SCORES);
    assert!(
        case(Some(Kv::F32s(scores[1..].to_vec())))
            .1
            .contains("entries")
    );
    let mut nan = scores.clone();
    nan[4] = f32::NAN;
    assert!(case(Some(Kv::F32s(nan))).1.contains("token 4"));
    let mut inf = scores;
    inf[5] = f32::INFINITY;
    assert!(case(Some(Kv::F32s(inf))).1.contains("token 5"));
    assert!(
        case(Some(Kv::I32s(vec![0; vocab.len()])))
            .1
            .contains("token 0")
    );
}

#[test]
fn malformed_specials_are_invalid() {
    let base = qwen_like();
    let Kv::Strs(tokens) = &base[2].1 else {
        unreachable!()
    };
    let n = u32::try_from(tokens.len()).unwrap();
    let case = |pairs: &[(&str, Kv)]| {
        let mut kvs = base.clone();
        for (k, v) in pairs {
            set(&mut kvs, k, v.clone());
        }
        invalid(&kvs)
    };
    assert!(case(&[(EOS_ID, Kv::U32(n))]).1.contains("past the last"));
    assert_eq!(case(&[(EOS_ID, Kv::I32(-1))]).0, EOS_ID);
    assert_eq!(case(&[(BOS_ID, Kv::Str("1".into()))]).0, BOS_ID);
    // Two of the three FIM keys.
    let (key, _) = case(&[(FIM_KEYS[0][0], Kv::U32(1)), (FIM_KEYS[0][2], Kv::U32(2))]);
    assert_eq!(key, FIM_KEYS[0][0]);
    // BOS asked for, none named.
    let mut kvs = base.clone();
    unset(&mut kvs, BOS_ID);
    set(&mut kvs, ADD_BOS, Kv::Bool(true));
    assert_eq!(invalid(&kvs).0, BOS_ID);
    assert_eq!(case(&[(ADD_EOS, Kv::Bool(true))]).0, EOS_ID);
    assert_eq!(case(&[(ADD_BOS, Kv::U32(1))]).0, ADD_BOS);

    // A special naming an unused token.
    let mut kvs = base.clone();
    let (Kv::Strs(mut t), Kv::I32s(mut ty)) = (base[2].1.clone(), base[3].1.clone()) else {
        unreachable!()
    };
    t.push("[PAD]".into());
    ty.push(5);
    set(&mut kvs, TOKENS, Kv::Strs(t));
    set(&mut kvs, TOKEN_TYPE, Kv::I32s(ty));
    set(&mut kvs, PAD_ID, Kv::U32(n));
    let (key, reason) = invalid(&kvs);
    assert_eq!(key, PAD_ID);
    assert!(reason.contains("unused"), "{reason}");
}

// ---- Properties --------------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Any text encodes to the same IDs through the source and the conversion, and the
    /// byte-level conversion decodes back to the text (NFC-normalized for `qwen2`).
    #[test]
    fn any_text_matches_the_source_and_round_trips(
        text in "(\\PC|[ \t\r\n]|<\\|im_end\\|>|<think>| |'s|[0-9]{1,6}){0,24}",
        which in 0usize..PRE_TOKENIZERS.len(),
    ) {
        let pre = &PRE_TOKENIZERS[which];
        let src = byte_level_source(pre, pre.add_bos);
        let source = Tokenizer::from_bytes(serde_json::to_vec(&src).unwrap()).unwrap();
        let converted = engine(&spec(&export_byte_level(&src, pre.name)));
        for add_special in [false, true] {
            prop_assert_eq!(ids(&converted, &text, add_special), ids(&source, &text, add_special));
        }
        let back = converted.decode(&ids(&converted, &text, false), false).unwrap();
        prop_assert_eq!(back, if pre.nfc { nfc(&text) } else { text.clone() });
    }

    /// The SentencePiece-style conversion matches its legacy source on text over its alphabet.
    #[test]
    fn any_text_matches_the_sentencepiece_source(text in "[abcxyzhelo é<>s/ ]{0,24}") {
        let source = Tokenizer::from_bytes(serde_json::to_vec(&spm_source()).unwrap()).unwrap();
        let converted = engine(&spec(&spm_kvs(&spm_vocab(true))));
        for add_special in [false, true] {
            prop_assert_eq!(ids(&converted, &text, add_special), ids(&source, &text, add_special));
        }
    }
}
