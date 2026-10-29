// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of `TokenizerSpec` and `TokenizerId` (COMP-E17-T01; RFC-0043 §8.1, §8.5).
//!
//! The fixtures under `tests/data/tokenizer/` are synthetic: tiny vocabularies and original
//! templates, laid out as the M0 families' repositories lay out their tokenizer files
//! (`llama-like` and `sentencepiece-like` for `llama`, `qwen-moe-like` for `qwen_moe`). None
//! comes from a real checkpoint.

use std::collections::BTreeSet;

use proptest::prelude::*;

use super::*;

macro_rules! fixture {
    ($dir:literal, $file:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/tokenizer/",
            $dir,
            "/",
            $file
        ))
    };
}

fn llama() -> TokenizerSources<'static> {
    TokenizerSources {
        tokenizer_json: fixture!("llama-like", "tokenizer.json"),
        tokenizer_config: Some(fixture!("llama-like", "tokenizer_config.json")),
        generation_config: Some(fixture!("llama-like", "generation_config.json")),
        chat_template_file: None,
    }
}

fn qwen() -> TokenizerSources<'static> {
    TokenizerSources {
        tokenizer_json: fixture!("qwen-moe-like", "tokenizer.json"),
        tokenizer_config: Some(fixture!("qwen-moe-like", "tokenizer_config.json")),
        generation_config: Some(fixture!("qwen-moe-like", "generation_config.json")),
        chat_template_file: None,
    }
}

fn sentencepiece() -> TokenizerSources<'static> {
    TokenizerSources {
        tokenizer_json: fixture!("sentencepiece-like", "tokenizer.json"),
        tokenizer_config: Some(fixture!("sentencepiece-like", "tokenizer_config.json")),
        generation_config: None,
        chat_template_file: Some(fixture!("sentencepiece-like", "chat_template.jinja")),
    }
}

#[test]
fn llama_like_spec_extracts() {
    let s = TokenizerSpec::from_sources(llama()).unwrap();
    assert_eq!(s.model(), llama().tokenizer_json);
    assert_eq!(
        s.specials(),
        &SpecialTokens {
            bos: Some(9),
            eos: Some(16),
            end_of_turn: vec![10, 15, 16],
            pad: None,
            unk: None,
            fim_prefix: None,
            fim_middle: None,
            fim_suffix: None,
        }
    );
    // No add_bos_token key: the post-processor's template adds BOS.
    assert!(s.add_bos());
    assert!(!s.add_eos());
    let t = s.chat_template().unwrap();
    assert!(t.starts_with("{{ bos_token }}") && t.contains("<|eot|>"));
    assert_eq!(s.template_source(), &TemplateSource::Shipped);
    assert_eq!(s.template_in_use(), s.chat_template());
}

#[test]
fn qwen_moe_like_spec_extracts() {
    let s = TokenizerSpec::from_sources(qwen()).unwrap();
    assert_eq!(
        s.specials(),
        &SpecialTokens {
            bos: None,
            eos: Some(11),
            end_of_turn: vec![9, 11],
            pad: Some(9),
            unk: None,
            fim_prefix: Some(12),
            fim_middle: Some(13),
            fim_suffix: Some(14),
        }
    );
    assert!(!s.add_bos());
    assert!(!s.add_eos());
    assert!(s.chat_template().unwrap().contains("<|im_start|>"));
}

#[test]
fn sentencepiece_like_spec_extracts_with_the_template_file() {
    let src = sentencepiece();
    let s = TokenizerSpec::from_sources(src).unwrap();
    let sp = s.specials();
    assert_eq!(
        (sp.bos, sp.eos, sp.unk, sp.pad, sp.end_of_turn.as_slice()),
        (Some(1), Some(2), Some(0), None, &[2][..])
    );
    assert!(s.add_bos());
    assert!(!s.add_eos());
    // The separate template file wins over tokenizer_config.json's, verbatim.
    assert_eq!(
        s.chat_template().unwrap().as_bytes(),
        src.chat_template_file.unwrap()
    );
}

#[test]
fn the_template_list_form_takes_default() {
    let cfg = br#"{"chat_template": [{"name": "tool_use", "template": "T"}, {"name": "default", "template": "D"}]}"#;
    let s = TokenizerSpec::from_sources(TokenizerSources {
        tokenizer_config: Some(cfg),
        ..qwen()
    })
    .unwrap();
    assert_eq!(s.chat_template(), Some("D"));
    let none = TokenizerSpec::from_sources(TokenizerSources {
        tokenizer_config: None,
        ..qwen()
    })
    .unwrap();
    assert_eq!(none.chat_template(), None);
    assert_eq!(none.template_source(), &TemplateSource::None);
    assert_eq!(none.template_in_use(), None);
}

#[test]
fn tokenizer_id_changes_when_the_template_in_use_changes() {
    let shipped = TokenizerSpec::from_sources(llama()).unwrap();
    let user = shipped
        .clone()
        .with_template_source(TemplateSource::UserOverride("{{ messages }}".into()));
    let family = shipped
        .clone()
        .with_template_source(TemplateSource::FamilyDefault("{{ messages }} ".into()));
    let none = shipped.clone().with_template_source(TemplateSource::None);
    let ids: BTreeSet<TokenizerId> = [&shipped, &user, &family, &none]
        .iter()
        .map(|s| s.id())
        .collect();
    assert_eq!(ids.len(), 4);
    // The shipped template is kept; only the template in use moved.
    assert_eq!(user.chat_template(), shipped.chat_template());
    assert_eq!(user.template_in_use(), Some("{{ messages }}"));
    // Going back restores the ID: it is a function of what it covers.
    assert_eq!(
        user.with_template_source(TemplateSource::Shipped).id(),
        shipped.id()
    );
    // An override equal to the shipped text gives the same ID: the source tag is not hashed,
    // the text in use is.
    let same = shipped
        .clone()
        .with_template_source(TemplateSource::UserOverride(
            shipped.chat_template().unwrap().to_owned(),
        ));
    assert_eq!(same.id(), shipped.id());
    // A one-byte change of the shipped template changes it.
    let mut cfg = String::from_utf8(llama().tokenizer_config.unwrap().to_vec()).unwrap();
    cfg = cfg.replacen("<|eot|>{% endfor %}", "<|eot|> {% endfor %}", 1);
    let edited = TokenizerSpec::from_sources(TokenizerSources {
        tokenizer_config: Some(cfg.as_bytes()),
        ..llama()
    })
    .unwrap();
    assert_ne!(edited.id(), shipped.id());
}

/// An edit of a spec's parts: model, specials, `add_bos`, `add_eos`.
type Edit = dyn Fn(&mut Vec<u8>, &mut SpecialTokens, &mut bool, &mut bool);

#[test]
fn tokenizer_id_covers_model_specials_and_flags() {
    let base = TokenizerSpec::from_sources(qwen()).unwrap();
    let rebuild = |f: &Edit| {
        let (mut m, mut s, mut b, mut e) = (
            base.model().to_vec(),
            base.specials().clone(),
            base.add_bos(),
            base.add_eos(),
        );
        f(&mut m, &mut s, &mut b, &mut e);
        TokenizerSpec::new(
            m,
            s,
            b,
            e,
            base.chat_template().map(str::to_owned),
            TemplateSource::Shipped,
        )
        .id()
    };
    assert_eq!(rebuild(&|_, _, _, _| {}), base.id());
    let changed = [
        rebuild(&|m, _, _, _| m.push(b'\n')),
        rebuild(&|_, s, _, _| s.eos = Some(9)),
        rebuild(&|_, s, _, _| s.end_of_turn.push(12)),
        rebuild(&|_, s, _, _| s.fim_prefix = None),
        rebuild(&|_, _, b, _| *b = true),
        rebuild(&|_, _, _, e| *e = true),
    ];
    let mut seen: BTreeSet<TokenizerId> = changed.iter().copied().collect();
    seen.insert(base.id());
    assert_eq!(seen.len(), changed.len() + 1);
}

/// The value RFC-0022 records in session metadata: a stable, serializable 32-byte digest,
/// printed as 64 hex digits, and the same for the same inputs on every call.
#[test]
fn tokenizer_id_is_exposed_for_session_metadata() {
    let a = TokenizerSpec::from_sources(llama()).unwrap().id();
    let b = TokenizerSpec::from_sources(llama()).unwrap().id();
    assert_eq!(a, b);
    let json = serde_json::to_string(&a).unwrap();
    assert_eq!(serde_json::from_str::<TokenizerId>(&json).unwrap(), a);
    assert_eq!(a.to_string().len(), 64);
    assert_ne!(a, TokenizerSpec::from_sources(qwen()).unwrap().id());
}

#[test]
fn malformed_or_inconsistent_sources_are_integrity_errors() {
    let tok = llama().tokenizer_json;
    let cases: [(TokenizerSources<'_>, &str); 7] = [
        (
            TokenizerSources {
                tokenizer_json: b"[]",
                ..Default::default()
            },
            TOKENIZER_JSON,
        ),
        (
            TokenizerSources {
                tokenizer_json: b"{\"added_tokens\": 3}",
                ..Default::default()
            },
            TOKENIZER_JSON,
        ),
        (
            TokenizerSources {
                tokenizer_json: tok,
                tokenizer_config: Some(br#"{"bos_token": "<nope>"}"#),
                ..Default::default()
            },
            TOKENIZER_CONFIG_JSON,
        ),
        (
            TokenizerSources {
                tokenizer_json: tok,
                tokenizer_config: Some(br#"{"add_bos_token": "yes"}"#),
                ..Default::default()
            },
            TOKENIZER_CONFIG_JSON,
        ),
        (
            TokenizerSources {
                tokenizer_json: tok,
                tokenizer_config: Some(br#"{"chat_template": 4}"#),
                ..Default::default()
            },
            TOKENIZER_CONFIG_JSON,
        ),
        (
            TokenizerSources {
                tokenizer_json: tok,
                generation_config: Some(br#"{"eos_token_id": [99999]}"#),
                ..Default::default()
            },
            GENERATION_CONFIG_JSON,
        ),
        (
            TokenizerSources {
                tokenizer_json: tok,
                chat_template_file: Some(b"\xff"),
                ..Default::default()
            },
            CHAT_TEMPLATE_FILE,
        ),
    ];
    for (src, file) in cases {
        let e = TokenizerSpec::from_sources(src).unwrap_err();
        assert_eq!((e.file, e.class()), (file, ErrorClass::Integrity), "{e}");
    }
}

#[test]
fn fim_needs_all_three_tokens_marked_special_or_not() {
    // Only a FIM prefix token: no FIM set.
    let only_prefix = br#"{"added_tokens": [{"id": 0, "content": "<|fim_prefix|>", "special": true}], "model": {"vocab": {}}}"#;
    let s = TokenizerSpec::from_sources(TokenizerSources {
        tokenizer_json: only_prefix,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(s.specials().fim_prefix, None);
    // All three, none marked special: the set is found.
    let not_special = br#"{"added_tokens": [{"id": 0, "content": "<PRE>"}, {"id": 1, "content": "<MID>", "special": false}, {"id": 2, "content": "<SUF>"}]}"#;
    let s = TokenizerSpec::from_sources(TokenizerSources {
        tokenizer_json: not_special,
        ..Default::default()
    })
    .unwrap();
    let sp = s.specials();
    assert_eq!(
        (sp.fim_prefix, sp.fim_middle, sp.fim_suffix),
        (Some(0), Some(1), Some(2))
    );
}

/// Regression (review of #1739): the Qwen3-MoE repositories ship their FIM tokens as added
/// tokens with `"special": false`. The fixture mirrors that, and the FIM IDs are still found.
#[test]
fn qwen_fim_tokens_not_marked_special_are_found() {
    let tok = String::from_utf8(qwen().tokenizer_json.to_vec()).unwrap();
    for name in ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"] {
        let at = tok.find(&format!("\"content\": \"{name}\"")).unwrap();
        let entry = &tok[at..at + tok[at..].find('}').unwrap()];
        assert!(entry.contains("\"special\": false"), "{entry}");
    }
    let sp = TokenizerSpec::from_sources(qwen())
        .unwrap()
        .specials()
        .clone();
    assert_eq!(
        (sp.fim_prefix, sp.fim_middle, sp.fim_suffix),
        (Some(12), Some(13), Some(14))
    );
}

/// Review of #1739: the end-of-turn tokens are a set. The same IDs in any order, or with a
/// repeat, give the same spec and the same `TokenizerId`.
#[test]
fn end_of_turn_order_and_repeats_do_not_change_the_id() {
    let base = TokenizerSpec::from_sources(llama()).unwrap();
    let with = |eot: Vec<u32>| {
        let mut sp = base.specials().clone();
        sp.end_of_turn = eot;
        TokenizerSpec::new(
            base.model().to_vec(),
            sp,
            base.add_bos(),
            base.add_eos(),
            base.chat_template().map(str::to_owned),
            TemplateSource::Shipped,
        )
    };
    let a = with(vec![10, 15, 16]);
    let b = with(vec![16, 10, 15]);
    let c = with(vec![15, 16, 10, 16, 10]);
    assert_eq!(a.id(), base.id());
    assert_eq!(b, a);
    assert_eq!(c, a);
    assert_eq!(b.specials().end_of_turn, [10, 15, 16]);
    assert_ne!(with(vec![10, 15]).id(), a.id());
    // The same set listed differently in generation_config.json gives the same ID.
    let gen_cfg = br#"{"eos_token_id": [16, 15, 10, 15]}"#;
    let d = TokenizerSpec::from_sources(TokenizerSources {
        generation_config: Some(gen_cfg),
        ..llama()
    })
    .unwrap();
    assert_eq!(d.id(), base.id());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Any permutation of the end-of-turn set, with repeats, gives the same `TokenizerId`.
    #[test]
    fn end_of_turn_is_hashed_as_a_set(
        ids in prop::collection::vec(0u32..1000, 0..8),
        extra in prop::collection::vec(0usize..8, 0..4),
        seed in any::<u64>(),
    ) {
        let base = TokenizerSpec::from_sources(qwen()).unwrap();
        let spec = |eot: Vec<u32>| {
            let mut sp = base.specials().clone();
            sp.end_of_turn = eot;
            TokenizerSpec::new(base.model().to_vec(), sp, false, false, None, TemplateSource::None)
        };
        let mut shuffled = ids.clone();
        for i in extra {
            if let Some(&v) = ids.get(i) {
                shuffled.push(v);
            }
        }
        // A deterministic shuffle from the seed.
        let mut x = seed | 1;
        for i in (1..shuffled.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            shuffled.swap(i, (x % (i as u64 + 1)) as usize);
        }
        prop_assert_eq!(spec(ids).id(), spec(shuffled).id());
    }

    /// Arbitrary side-file text never panics; a rejection is an `Integrity` error.
    #[test]
    fn arbitrary_sources_never_panic(
        tok in "\\{[\"a-z0-9_:,\\[\\]{} <|>-]{0,80}\\}",
        cfg in proptest::option::of("\\{[\"a-z0-9_:,\\[\\]{} <|>-]{0,80}\\}"),
        gen_cfg in proptest::option::of("\\{[\"a-z0-9_:,\\[\\]{} ]{0,40}\\}"),
    ) {
        let src = TokenizerSources {
            tokenizer_json: tok.as_bytes(),
            tokenizer_config: cfg.as_deref().map(str::as_bytes),
            generation_config: gen_cfg.as_deref().map(str::as_bytes),
            chat_template_file: None,
        };
        if let Err(e) = TokenizerSpec::from_sources(src) {
            prop_assert_eq!(e.class(), ErrorClass::Integrity);
        }
    }

    /// Any two different templates in use give different IDs.
    #[test]
    fn different_templates_give_different_ids(a in ".{0,40}", b in ".{0,40}") {
        prop_assume!(a != b);
        let base = TokenizerSpec::from_sources(qwen()).unwrap();
        let x = base.clone().with_template_source(TemplateSource::UserOverride(a));
        let y = base.with_template_source(TemplateSource::UserOverride(b));
        prop_assert_ne!(x.id(), y.id());
    }
}

/// `TokenizerId`'s bytes, pinned (COMP-E10-T11): a golden computed apart from this crate, by a
/// pure-Python BLAKE3 that matches the published BLAKE3 test vectors, over `compute_id`'s
/// encoding: the length-prefixed tag and model, the seven optional specials, the end-of-turn
/// set, the two flags and the template in use.
#[test]
fn tokenizer_id_golden() {
    let specials = SpecialTokens {
        bos: Some(1),
        eos: Some(2),
        end_of_turn: vec![7, 2, 7],
        ..SpecialTokens::default()
    };
    let spec = TokenizerSpec::new(
        br#"{"model":{"type":"BPE"}}"#.to_vec(),
        specials,
        true,
        false,
        Some("{{ messages }}".to_owned()),
        TemplateSource::Shipped,
    );
    assert_eq!(
        spec.id().to_string(),
        "217943fe1c49c20a7f7ba46bddb9cc5ee87cdd542975625930a92ac8d0330bd2"
    );
}
