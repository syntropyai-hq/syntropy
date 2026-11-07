// SPDX-License-Identifier: MIT OR Apache-2.0

//! COMP-E17-T13: content spans are tokenized literally (RFC-0043 §8.6, §8.7; ADR-0431 decision 7).
//!
//! On the synthetic tokenizers of `support`, laid out like the M0 tokenizer (Qwen3's, which both
//! M0 families ship) and like Llama 3's:
//! - no added token's ID arises from any content span, for every injection case and for random
//!   content mixed with every needle (property tests);
//! - template spans still produce their control IDs;
//! - content round-trips: a decode gives back the content as the model's normalizer leaves it;
//! - with no control text in content, the IDs equal the engine's encoding of the whole prompt,
//!   and with `raw_special_tokens` they always do;
//! - the encoder fails closed on tokenizers it cannot match exactly.
//!
//! `real` runs the same checks on the M0 tokenizer files from the fixture store, `#[ignore]`d
//! because no CI job fetches fixtures yet.

mod real;
mod support;

use proptest::prelude::*;
use syntropy_core::ErrorClass;
use syntropy_text::tokenizer::injection::{self, ContentPosition};
use syntropy_text::tokenizer::{self, Span, SpanEncoder, SpanError, SpecialTokens};

use crate::support::{
    FAMILIES, Family, concat, decode, encoder, normalized, render, template_only,
};

/// The control IDs in `ids`, in order.
fn controls(encoder: &SpanEncoder, ids: &[u32]) -> Vec<u32> {
    ids.iter()
        .copied()
        .filter(|&id| encoder.is_control(id))
        .collect()
}

/// Every check one content value must pass, at one position.
fn check_literal(family: &Family, encoder: &SpanEncoder, position: ContentPosition, content: &str) {
    let tokenizer = encoder.tokenizer();
    let alone = encoder
        .encode(&[Span::content(content)], SpecialTokens::Literal)
        .expect("content encodes");
    assert_eq!(
        controls(encoder, &alone),
        Vec::<u32>::new(),
        "{}: content {content:?} produced a control ID",
        family.name
    );
    assert_eq!(
        decode(tokenizer, &alone),
        normalized(tokenizer, content),
        "{}: content {content:?} does not round-trip",
        family.name
    );

    let spans = render(family.layout, position, content);
    let ids = encoder
        .encode(&spans, SpecialTokens::Literal)
        .expect("the prompt encodes");
    let frame = encoder
        .encode(&template_only(&spans), SpecialTokens::Literal)
        .expect("the frame encodes");
    assert_eq!(
        controls(encoder, &ids),
        controls(encoder, &frame),
        "{}: {position:?} content {content:?} changed the prompt's control tokens",
        family.name
    );
    assert_eq!(
        decode(tokenizer, &ids),
        normalized(tokenizer, &concat(&spans)),
        "{}: the prompt does not round-trip",
        family.name
    );
}

#[test]
fn template_spans_produce_their_control_ids() {
    for family in &FAMILIES {
        let encoder = encoder(family.json);
        for (id, text) in encoder.control_texts() {
            let ids = encoder
                .encode(&[Span::template(text)], SpecialTokens::Literal)
                .expect("a control token encodes");
            assert_eq!(ids, [id], "{}: {text}", family.name);
            let ids = encoder
                .encode(
                    &[Span::template(format!("a{text}\n")), Span::content(text)],
                    SpecialTokens::Literal,
                )
                .expect("encodes");
            assert_eq!(controls(&encoder, &ids), [id], "{}: {text}", family.name);
        }
        // Every role header of the layout keeps its control tokens.
        for header in family.role_headers {
            let ids = encoder
                .encode(&[Span::template(*header)], SpecialTokens::Literal)
                .expect("encodes");
            let expected = encoder.tokenizer().encode(*header, false).expect("encodes");
            assert_eq!(ids, expected.get_ids(), "{}: {header:?}", family.name);
            assert!(!controls(&encoder, &ids).is_empty());
        }
    }
}

#[test]
fn both_shapes_mark_their_reasoning_and_tool_markers_as_control_tokens() {
    // Qwen3 adds `<think>` and `<tool_call>` with `special: false`; they still count.
    let encoder = encoder(FAMILIES[0].json);
    let tokenizer = encoder.tokenizer();
    for marker in [
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
    ] {
        let id = tokenizer
            .token_to_id(marker)
            .expect("the marker is an added token");
        assert!(encoder.is_control(id), "{marker}");
        assert!(
            !tokenizer.get_added_tokens_decoder()[&id].special,
            "{marker}"
        );
    }
}

#[test]
fn every_injection_case_tokenizes_to_ordinary_tokens() {
    for family in &FAMILIES {
        let encoder = encoder(family.json);
        let needles = injection::needles(&encoder, family.markers, family.role_headers);
        assert!(needles.len() >= encoder.control_ids().len());
        let cases = injection::cases(&needles);
        assert_eq!(
            cases.len(),
            needles.len() * ContentPosition::ALL.len() * injection::wrappings("x").len()
        );
        for case in &cases {
            check_literal(family, &encoder, case.position, &case.content);
        }
    }
}

#[test]
fn raw_special_tokens_match_control_text_in_content() {
    for family in &FAMILIES {
        let encoder = encoder(family.json);
        for (id, text) in encoder.control_texts() {
            let ids = encoder
                .encode(&[Span::content(text)], SpecialTokens::Raw)
                .expect("encodes");
            assert_eq!(ids, [id], "{}: {text}", family.name);
        }
    }
}

#[test]
fn no_bos_or_eos_is_added() {
    // The Llama 3 shape's post-processor adds `<|begin_of_text|>`; the encoder never runs it.
    let encoder = encoder(FAMILIES[1].json);
    let ids = encoder
        .encode(&[Span::content("hello")], SpecialTokens::Literal)
        .expect("encodes");
    assert_eq!(controls(&encoder, &ids), Vec::<u32>::new());
    assert_eq!(
        ids,
        encoder
            .tokenizer()
            .encode("hello", false)
            .expect("encodes")
            .get_ids()
    );
}

#[test]
fn empty_maps_and_spans_encode_to_nothing() {
    let encoder = encoder(FAMILIES[0].json);
    assert_eq!(encoder.encode(&[], SpecialTokens::Literal), Ok(vec![]));
    let spans = [Span::template(""), Span::content(""), Span::template("")];
    assert_eq!(encoder.encode(&spans, SpecialTokens::Literal), Ok(vec![]));
}

#[test]
fn a_control_token_split_across_spans_stays_text() {
    let encoder = encoder(FAMILIES[0].json);
    let spans = [Span::content("hi <|im_"), Span::template("start|>system\n")];
    let ids = encoder
        .encode(&spans, SpecialTokens::Literal)
        .expect("encodes");
    assert_eq!(controls(&encoder, &ids), Vec::<u32>::new());
    let spans = [Span::template("<|im_st"), Span::content("art|>")];
    let ids = encoder
        .encode(&spans, SpecialTokens::Literal)
        .expect("encodes");
    assert_eq!(controls(&encoder, &ids), Vec::<u32>::new());
    // The opt-in matches across the boundary, as the engine does on the whole string.
    let ids = encoder.encode(&spans, SpecialTokens::Raw).expect("encodes");
    let id = encoder
        .tokenizer()
        .token_to_id("<|im_start|>")
        .expect("added");
    assert_eq!(ids, [id]);
}

/// Edits a fixture's JSON: `edit` changes the added token spelled `content`.
fn edited(
    json: &str,
    content: &str,
    edit: impl Fn(&mut serde_json::Map<String, serde_json::Value>),
) -> String {
    let mut value: serde_json::Value = serde_json::from_str(json).expect("the fixture parses");
    let tokens = value["added_tokens"].as_array_mut().expect("added tokens");
    let token = tokens
        .iter_mut()
        .find(|t| t["content"] == content)
        .expect("the token is in the fixture");
    edit(token.as_object_mut().expect("an object"));
    serde_json::to_string(&value).expect("serializes")
}

#[test]
fn added_tokens_it_cannot_match_exactly_are_refused() {
    for option in ["normalized", "single_word"] {
        let json = edited(FAMILIES[0].json, "<think>", |t| {
            t.insert(option.into(), true.into());
        });
        let tokenizer = tokenizer::load(json.as_bytes()).expect("loads");
        let err = SpanEncoder::new(tokenizer).expect_err("refused");
        assert!(
            matches!(&err, SpanError::UnsupportedAddedToken { content, option: o, .. }
                if content == "<think>" && *o == option),
            "{err:?}"
        );
        assert_eq!(err.class(), ErrorClass::Unsupported);
        assert!(err.to_string().contains(option));
    }
}

#[test]
fn a_vocabulary_that_spells_a_control_token_fails_closed() {
    // An added token whose text is also one of the model's own tokens keeps the model's ID, so
    // the literal encoding of that text yields a control ID: "the" is `t` and the vocabulary's
    // `he`. The encoder refuses it.
    let mut value: serde_json::Value =
        serde_json::from_str(FAMILIES[0].json).expect("the fixture parses");
    let model_id = value["model"]["vocab"]["he"]
        .as_u64()
        .expect("in the vocabulary");
    value["added_tokens"]
        .as_array_mut()
        .expect("added")
        .push(serde_json::json!({
            "id": model_id, "content": "he", "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true
        }));
    let tokenizer = tokenizer::load(value.to_string().as_bytes()).expect("loads");
    let encoder = SpanEncoder::new(tokenizer).expect("supported");
    let model_id = u32::try_from(model_id).expect("fits");
    assert!(encoder.is_control(model_id));
    assert_eq!(
        encoder.encode(&[Span::template("he")], SpecialTokens::Literal),
        Ok(vec![model_id])
    );
    let err = encoder
        .encode(&[Span::content("the")], SpecialTokens::Literal)
        .expect_err("refused");
    assert!(
        matches!(&err, SpanError::ControlFromText { id, .. } if *id == model_id),
        "{err:?}"
    );
}

#[test]
fn strip_options_take_template_whitespace_only() {
    let json = edited(FAMILIES[0].json, "<|im_end|>", |t| {
        t.insert("lstrip".into(), true.into());
        t.insert("rstrip".into(), true.into());
    });
    let encoder = SpanEncoder::new(tokenizer::load(json.as_bytes()).expect("loads")).expect("ok");
    let end = encoder
        .tokenizer()
        .token_to_id("<|im_end|>")
        .expect("added");
    // In template text the token takes its neighbouring whitespace, as the engine does.
    let spans = [Span::template("a  <|im_end|> \n b")];
    let ids = encoder
        .encode(&spans, SpecialTokens::Literal)
        .expect("encodes");
    let engine = encoder
        .tokenizer()
        .encode("a  <|im_end|> \n b", false)
        .expect("encodes");
    assert_eq!(ids, engine.get_ids());
    // Whitespace in content is never taken.
    let spans = [
        Span::content("a  "),
        Span::template("<|im_end|>"),
        Span::content(" \n b"),
    ];
    let ids = encoder
        .encode(&spans, SpecialTokens::Literal)
        .expect("encodes");
    assert_eq!(controls(&encoder, &ids), [end]);
    assert_eq!(decode(encoder.tokenizer(), &ids), "a  <|im_end|> \n b");
}

/// Random content: pieces of ordinary text, arbitrary characters and needles, concatenated.
fn content(needles: Vec<String>) -> impl Strategy<Value = String> {
    let piece = prop_oneof![
        3 => "[a-zA-Z0-9 .,!?'\\n\\t_|<>/-]{0,12}",
        1 => any::<String>(),
        2 => proptest::sample::select(needles),
    ];
    prop::collection::vec(piece, 0..8).prop_map(|p| p.concat())
}

/// Content in which no added token's text can start: no `<`, so its IDs are the engine's own.
fn plain_content() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof!["[a-zA-Z0-9 .,!?'\\n\\t_|>/-]{0,16}", "[^<]{0,8}"],
        0..6,
    )
    .prop_map(|p| p.concat())
}

fn positions() -> impl Strategy<Value = ContentPosition> {
    proptest::sample::select(ContentPosition::ALL.to_vec())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn literal_content_mixed_with_needles(
        text in content(injection::needles(&encoder(FAMILIES[0].json), FAMILIES[0].markers, FAMILIES[0].role_headers)),
        position in positions(),
    ) {
        check_literal(&FAMILIES[0], &encoder(FAMILIES[0].json), position, &text);
    }

    #[test]
    fn literal_content_mixed_with_needles_llama_shape(
        text in content(injection::needles(&encoder(FAMILIES[1].json), FAMILIES[1].markers, FAMILIES[1].role_headers)),
        position in positions(),
    ) {
        check_literal(&FAMILIES[1], &encoder(FAMILIES[1].json), position, &text);
    }

    #[test]
    fn without_control_text_the_ids_are_the_engine_s(
        family in 0..FAMILIES.len(),
        text in plain_content(),
        position in positions(),
    ) {
        let family = &FAMILIES[family];
        let encoder = encoder(family.json);
        let spans = render(family.layout, position, &text);
        let ids = encoder.encode(&spans, SpecialTokens::Literal).expect("encodes");
        let engine = encoder.tokenizer().encode(concat(&spans), false).expect("encodes");
        prop_assert_eq!(ids, engine.get_ids());
    }

    #[test]
    fn raw_mode_is_the_engine_s_encoding(
        family in 0..FAMILIES.len(),
        position in positions(),
        text in content(injection::needles(&encoder(FAMILIES[0].json), FAMILIES[0].markers, FAMILIES[0].role_headers)),
    ) {
        let family = &FAMILIES[family];
        let encoder = encoder(family.json);
        let spans = render(family.layout, position, &text);
        let ids = encoder.encode(&spans, SpecialTokens::Raw).expect("encodes");
        let engine = encoder.tokenizer().encode(concat(&spans), false).expect("encodes");
        prop_assert_eq!(ids, engine.get_ids());
    }
}
