// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E3-T10: chat-template rendering into template and content spans (RFC-0050 §6.1, §6.4;
//! ADR-0431).
//!
//! - **Goldens.** `tests/data/template/expected.json` holds the reference engine's renderings
//!   of `conversations.json` through two templates written for these tests, laid out like the
//!   M0 families' ChatML template and like Llama 3's. `record.py` recorded them. Each rendering
//!   must match byte for byte, and a conversation the reference refuses with `raise_exception`
//!   must be refused with the same message. `real` runs the M0 families' shipped template
//!   against `expected-real-m0.json`, `#[ignore]`d because it needs the fixture cache.
//! - **Spans.** Every message's text is a content span. The marking survives the template's
//!   string handling, so the reasoning a template splits out of an assistant turn stays
//!   content.
//! - **Injection.** Every control string of the family (each added token's text, the
//!   `ChatFormat` tool-call and reasoning markers, the role headers), injected into system and
//!   user content, yields no added-token ID inside a content span, decodes back to the content,
//!   and leaves the template's own control tokens exactly as they were (property test, on the
//!   synthetic Qwen3-shaped tokenizer of COMP-E17-T13).
//! - **Hostile templates.** Endless loops stop at the fuel limit, unbounded recursion at the
//!   recursion limit, a huge output at the size limit, and forged content markers are refused,
//!   all without a panic. A property test renders random templates built from Jinja fragments
//!   and checks only that each returns, with a result or an error.

mod real;

use std::path::Path;

use proptest::prelude::*;
use serde_json::Value as Json;
use syntropy_core::ErrorClass;
use syntropy_text::template::{
    ChatTemplate, Date, Limits, Message, RenderRequest, Role, TemplateError,
};
use syntropy_text::tokenizer::injection;
use syntropy_text::tokenizer::{self, Span, SpanEncoder, SpanKind, SpecialTokens};

fn data(path: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(path);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

pub(crate) fn json(path: &str) -> Json {
    serde_json::from_str(&data(path)).expect("valid JSON")
}

/// A conversation of `conversations.json` as messages.
pub(crate) fn messages(conv: &Json) -> Vec<Message> {
    conv["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .map(|m| Message {
            role: match m["role"].as_str().expect("role") {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                r => panic!("role {r}"),
            },
            content: m["content"].as_str().expect("content").to_owned(),
            reasoning: m["reasoning_content"].as_str().map(str::to_owned),
        })
        .collect()
}

pub(crate) fn joined(spans: &[Span]) -> String {
    spans.iter().map(|s| s.text.as_str()).collect()
}

/// Renders every case of one recorded template and compares it with the recording.
/// With `slices`, every content span must also be a slice of a message: true for templates
/// that only cut content, false for one that transforms it (`upper`, `tojson`).
pub(crate) fn check_golden(
    template: &ChatTemplate,
    recorded: &Json,
    date: Date,
    slices: bool,
) -> usize {
    let convs = json("template/conversations.json");
    let convs = convs["conversations"].as_array().expect("conversations");
    let kwargs = &recorded["kwargs"];
    let cases = recorded["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), convs.len());
    for (conv, case) in convs.iter().zip(cases) {
        let id = conv["id"].as_str().expect("id");
        assert_eq!(case["id"].as_str(), Some(id));
        let msgs = messages(conv);
        let req = RenderRequest {
            messages: &msgs,
            add_generation_prompt: conv["add_generation_prompt"].as_bool().expect("bool"),
            enable_thinking: conv["enable_thinking"].as_bool(),
            bos_token: kwargs["bos_token"].as_str(),
            eos_token: kwargs["eos_token"].as_str(),
            date: Some(date),
        };
        match (
            template.render(&req),
            case.get("rendered"),
            case.get("raised"),
        ) {
            (Ok(spans), Some(want), None) => {
                assert_eq!(joined(&spans), want.as_str().expect("string"), "{id}");
                if slices {
                    check_content_spans(&spans, &msgs, id);
                }
            }
            (Err(TemplateError::Raised(msg)), None, Some(want)) => {
                assert_eq!(Some(msg.as_str()), want.as_str(), "{id}");
            }
            (got, _, _) => panic!("{id}: {got:?}, recorded {case}"),
        }
    }
    cases.len()
}

/// Every content span is text from the request: a slice of a message or its reasoning.
fn check_content_spans(spans: &[Span], msgs: &[Message], id: &str) {
    for s in spans.iter().filter(|s| s.kind == SpanKind::Content) {
        assert!(
            msgs.iter().any(|m| m.content.contains(&s.text)
                || m.reasoning.as_deref().is_some_and(|r| r.contains(&s.text))),
            "{id}: content span {:?} is not request text",
            s.text
        );
    }
}

pub(crate) const DATE: Date = Date {
    year: 2026,
    month: 11,
    day: 7,
};

fn synthetic(name: &str) -> (ChatTemplate, Json) {
    let expected = json("template/expected.json");
    assert_eq!(expected["date"].as_str(), Some("2026-11-07"));
    let recorded = expected["templates"]
        .as_array()
        .expect("templates")
        .iter()
        .find(|t| t["name"].as_str() == Some(name))
        .expect("recorded")
        .clone();
    let source = data(&format!("template/{name}.jinja"));
    (
        ChatTemplate::new(&source, Limits::default()).expect("loads"),
        recorded,
    )
}

#[test]
fn renderings_match_the_reference_engine_byte_for_byte() {
    for name in ["qwen3-shaped", "llama3-shaped"] {
        let (template, recorded) = synthetic(name);
        let n = check_golden(&template, &recorded, DATE, name == "qwen3-shaped");
        assert!(n >= 17, "{name}: {n} cases");
    }
}

#[test]
fn message_text_is_content_and_template_text_is_not() {
    let (template, _) = synthetic("qwen3-shaped");
    let msgs = [
        Message::new(Role::System, "SYS"),
        Message::new(Role::User, "USR"),
        Message::new(Role::Assistant, "<think>\nRSN\n</think>\n\nANS"),
    ];
    let spans = template
        .render(&RenderRequest {
            add_generation_prompt: false,
            ..RenderRequest::new(&msgs)
        })
        .expect("renders");
    let content: Vec<&str> = spans
        .iter()
        .filter(|s| s.kind == SpanKind::Content)
        .map(|s| s.text.as_str())
        .collect();
    // The reasoning the template split out of the assistant turn stays content; the `<think>`
    // tags the template wrote around it are template text.
    assert_eq!(content, ["SYS", "USR", "RSN", "ANS"]);
    let template_text: String = spans
        .iter()
        .filter(|s| s.kind == SpanKind::Template)
        .map(|s| s.text.as_str())
        .collect();
    assert!(template_text.contains("<think>\n"));
    assert!(!template_text.contains("RSN"));
}

#[test]
fn the_json_of_request_values_is_content() {
    let (template, _) = synthetic("llama3-shaped");
    let msgs = [
        Message::new(Role::User, "a"),
        Message::new(Role::Assistant, "b"),
        Message::new(Role::User, "<|eot_id|>"),
    ];
    let spans = template
        .render(&RenderRequest {
            bos_token: Some("<|begin_of_text|>"),
            date: Some(DATE),
            ..RenderRequest::new(&msgs)
        })
        .expect("renders");
    let last = spans.last().expect("spans");
    assert_eq!(last.kind, SpanKind::Template);
    let json_span = &spans[spans.len() - 2];
    assert_eq!(json_span.kind, SpanKind::Content);
    assert_eq!(json_span.text, r#"{"last": "<|EOT_ID|>", "n": 3}"#);
}

#[test]
fn a_template_that_needs_a_date_fails_without_one() {
    let (template, _) = synthetic("llama3-shaped");
    let msgs = [Message::new(Role::User, "a")];
    let e = template.render(&RenderRequest::new(&msgs)).unwrap_err();
    assert!(matches!(e, TemplateError::Render(_)), "{e:?}");
    assert_eq!(e.class(), ErrorClass::Unsupported);
}

#[test]
fn limits_leave_the_m0_template_headroom() {
    // The goldens' longest conversation renders with a hundredth of the default fuel.
    let (_, recorded) = synthetic("qwen3-shaped");
    let tight = Limits {
        fuel: Limits::default().fuel / 100,
        fuel_per_message: 0,
        ..Limits::default()
    };
    let template = ChatTemplate::new(&data("template/qwen3-shaped.jinja"), tight).expect("loads");
    check_golden(&template, &recorded, DATE, true);
}

// --- Injection ---------------------------------------------------------------------------

const MARKERS: [&str; 4] = ["<tool_call>", "</tool_call>", "<think>", "</think>"];
const ROLE_HEADERS: [&str; 4] = [
    "<|im_start|>system\n",
    "<|im_start|>user\n",
    "<|im_start|>assistant\n",
    "<|im_end|>\n<|im_start|>system\n",
];

pub(crate) fn qwen_encoder() -> SpanEncoder {
    let json = data("tokenizer/qwen3-shaped/tokenizer.json");
    SpanEncoder::new(tokenizer::load(json.as_bytes()).expect("loads")).expect("supported")
}

fn controls(encoder: &SpanEncoder, ids: &[u32]) -> Vec<u32> {
    ids.iter()
        .copied()
        .filter(|&id| encoder.is_control(id))
        .collect()
}

/// What the tokenizer's normalizer makes of `text`, which is what a decode gives back.
fn normalized(encoder: &SpanEncoder, text: &str) -> String {
    let mut n = tokenizers::NormalizedString::from(text);
    if let Some(normalizer) = encoder.tokenizer().get_normalizer() {
        tokenizers::Normalizer::normalize(normalizer, &mut n).expect("normalizes");
    }
    n.get().to_owned()
}

/// Renders `system` and `user` content through `template` and checks the injection property.
pub(crate) fn check_injection(
    template: &ChatTemplate,
    encoder: &SpanEncoder,
    system: &str,
    user: &str,
) {
    let msgs = [
        Message::new(Role::System, system),
        Message::new(Role::User, user),
    ];
    let spans = template
        .render(&RenderRequest::new(&msgs))
        .expect("renders");
    let ids = encoder
        .encode(&spans, SpecialTokens::Literal)
        .expect("encodes");

    // The template's own control tokens, exactly: the frame with every content value emptied.
    let frame: Vec<Span> = spans
        .iter()
        .map(|s| match s.kind {
            SpanKind::Template => s.clone(),
            SpanKind::Content => Span::content(""),
        })
        .collect();
    let frame_ids = encoder
        .encode(&frame, SpecialTokens::Literal)
        .expect("encodes");
    let frame_controls = controls(encoder, &frame_ids);
    assert_eq!(
        controls(encoder, &ids),
        frame_controls,
        "{system:?} {user:?}"
    );
    // The family's special tokens still come out of the template spans.
    let im_start = encoder
        .tokenizer()
        .token_to_id("<|im_start|>")
        .expect("im_start");
    let im_end = encoder
        .tokenizer()
        .token_to_id("<|im_end|>")
        .expect("im_end");
    assert_eq!(frame_controls.iter().filter(|&&i| i == im_start).count(), 3);
    assert_eq!(frame_controls.iter().filter(|&&i| i == im_end).count(), 2);

    // Each content span alone: no control ID, and it decodes back to its text.
    let mut seen = Vec::new();
    for s in spans.iter().filter(|s| s.kind == SpanKind::Content) {
        let ids = encoder
            .encode(std::slice::from_ref(s), SpecialTokens::Literal)
            .expect("encodes");
        assert!(controls(encoder, &ids).is_empty(), "{:?}", s.text);
        let decoded = encoder.tokenizer().decode(&ids, false).expect("decodes");
        assert_eq!(decoded, normalized(encoder, &s.text));
        seen.push(s.text.clone());
    }
    // And the content spans are the two values, whole.
    let want: Vec<String> = [system, user]
        .into_iter()
        .filter(|v| !v.is_empty())
        .map(str::to_owned)
        .collect();
    assert_eq!(seen, want);
}

#[test]
fn every_control_string_in_system_and_user_content_stays_literal() {
    let (template, _) = synthetic("qwen3-shaped");
    let encoder = qwen_encoder();
    let needles = injection::needles(&encoder, &MARKERS, &ROLE_HEADERS);
    assert!(needles.len() > 10);
    for case in injection::cases(&needles) {
        match case.position {
            injection::ContentPosition::System => {
                check_injection(&template, &encoder, &case.content, "hello");
            }
            injection::ContentPosition::User => {
                check_injection(&template, &encoder, "Be brief.", &case.content);
            }
            // No tools or documents reach the M0 template yet (PROD-E3-T01).
            injection::ContentPosition::ToolResult | injection::ContentPosition::Document => {}
        }
    }
}

fn content_strategy(needles: Vec<String>) -> impl Strategy<Value = String> {
    let piece = prop_oneof![
        3 => proptest::sample::select(needles),
        2 => "[a-zA-Z0-9 .,!?\n]{0,12}",
        1 => "\\PC{0,6}",
    ];
    proptest::collection::vec(piece, 0..8).prop_map(|v| v.concat())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// Fuzz: random content mixed with every control string, in system and user content.
    #[test]
    fn fuzzed_injection_stays_literal(
        system in content_strategy(injection::needles(&qwen_encoder(), &MARKERS, &ROLE_HEADERS)),
        user in content_strategy(injection::needles(&qwen_encoder(), &MARKERS, &ROLE_HEADERS)),
    ) {
        let (template, _) = synthetic("qwen3-shaped");
        // The NFC normalizer may merge a combining mark with the template text next to it;
        // the property is about control tokens, so the content is kept NFC-stable.
        let encoder = qwen_encoder();
        prop_assume!(normalized(&encoder, &system) == system);
        prop_assume!(normalized(&encoder, &user) == user);
        check_injection(&template, &encoder, &system, &user);
    }
}

// --- Hostile templates -------------------------------------------------------------------

fn small() -> Limits {
    Limits {
        fuel: 50_000,
        fuel_per_message: 0,
        recursion: 50,
        max_bytes: 1 << 20,
    }
}

fn render_with(source: &str, limits: Limits) -> Result<Vec<Span>, TemplateError> {
    let msgs = [Message::new(Role::User, "hi")];
    ChatTemplate::new(source, limits)?.render(&RenderRequest::new(&msgs))
}

#[test]
fn an_endless_loop_stops_at_the_fuel_limit() {
    for source in [
        "{% for i in range(100000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}",
        "{% for m in messages %}{% for i in range(100000) %}{% for j in range(100000) %}{{ i }}{% endfor %}{% endfor %}{% endfor %}",
        "{% set ns = namespace(n=0) %}{% for i in range(100000) %}{% for j in range(100000) %}\
         {% set ns.n = ns.n + 1 %}{% endfor %}{% endfor %}",
    ] {
        let e = render_with(source, small()).unwrap_err();
        assert_eq!(e, TemplateError::OutOfFuel, "{source}");
        assert_eq!(e.class(), ErrorClass::Capacity);
    }
}

#[test]
fn unbounded_recursion_stops_at_the_recursion_limit() {
    let e = render_with(
        "{% for x in [1] recursive %}{{ loop([1]) }}{% endfor %}",
        small(),
    )
    .unwrap_err();
    assert_eq!(e, TemplateError::Recursion);
    // Deep nesting written out: refused when parsed or when run, never a panic.
    let deep = format!(
        "{}x{}",
        "{% if true %}".repeat(400),
        "{% endif %}".repeat(400)
    );
    let e = render_with(&deep, small()).unwrap_err();
    assert!(
        matches!(e, TemplateError::Recursion | TemplateError::Syntax(_)),
        "{e:?}"
    );
}

#[test]
fn a_huge_output_is_refused() {
    let e = render_with("{{ 'x' * 2000000 }}", small()).unwrap_err();
    assert!(matches!(e, TemplateError::TooLarge { .. }), "{e:?}");
}

#[test]
fn forged_markers_and_missing_features_are_refused() {
    for (source, want) in [
        ("{{ '\u{FDD1}' }}", "markup"),
        ("{{ '\u{FDD0}x' }}", "markup"),
        ("{{ messages[0].content[1:] }}", "markup"),
        ("{% macro m() %}x{% endmacro %}{{ m() }}", "syntax"),
        ("{% include 'x' %}", "syntax-or-render"),
        ("{% extends 'x' %}", "syntax-or-render"),
        ("{{ messages[0].content.format() }}", "render"),
        ("{{ 'a' | nosuchfilter }}", "syntax-or-render"),
        ("{{ raise_exception('no') }}", "raised"),
        ("{% if %}", "syntax"),
    ] {
        let e = render_with(source, small()).unwrap_err();
        let ok = match want {
            "markup" => matches!(e, TemplateError::Markup(_)),
            "syntax" => matches!(e, TemplateError::Syntax(_)),
            "render" => matches!(e, TemplateError::Render(_)),
            "syntax-or-render" => {
                matches!(e, TemplateError::Syntax(_) | TemplateError::Render(_))
            }
            "raised" => e == TemplateError::Raised("no".to_owned()),
            _ => false,
        };
        assert!(ok, "{source}: {e:?}");
    }
}

const FRAGMENTS: [&str; 24] = [
    "{% for i in range(1000) %}",
    "{% endfor %}",
    "{% for m in messages %}",
    "{% if messages %}",
    "{% else %}",
    "{% endif %}",
    "{{ messages[0].content }}",
    "{{ messages[0].content.split('h') }}",
    "{{ messages[0].content.strip() }}",
    "{{ messages[0].content | tojson }}",
    "{{ messages[0].content | trim | upper }}",
    "{{ loop.index }}",
    "{% set ns = namespace(x='') %}",
    "{% set ns.x = ns.x ~ ns.x ~ 'ab' %}",
    "{{ ns.x }}",
    "{{ 'x' * 1000 }}",
    "{{ raise_exception('r') }}",
    "{{ strftime_now('%Y') }}",
    "{{ undefined_thing.attr }}",
    "{{ 1 / 0 }}",
    "{{ [1, 2][5] }}",
    "{{ '\u{FDD1}' }}",
    "{% for x in [1] recursive %}{{ loop([1]) }}{% endfor %}",
    "text<|im_start|>",
];

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// Random templates from Jinja fragments, balanced or not: each render returns a result
    /// or an error, never a panic, within the limits.
    #[test]
    fn random_hostile_templates_never_panic(
        picks in proptest::collection::vec(0usize..FRAGMENTS.len(), 0..24),
    ) {
        let source: String = picks.iter().map(|&i| FRAGMENTS[i]).collect();
        let result = render_with(&source, small());
        if let Err(e) = &result {
            prop_assert_ne!(e.class(), ErrorClass::Internal, "{}: {}", source, e);
        }
        if let Ok(spans) = &result {
            prop_assert!(joined(spans).len() <= small().max_bytes);
        }
    }
}
