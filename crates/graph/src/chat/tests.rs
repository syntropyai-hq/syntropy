// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ChatFormat` on synthetic tokenizers laid out like the M0 repositories' files. The real
//! files are `tests/chat_real.rs`'s.

use smallvec::smallvec;
use syntropy_manifest::tokenizer::{SpecialTokens, TemplateSource, TokenizerSpec};

use super::*;

/// A `tokenizer.json` holding only `added`: (content, id, special).
fn model(added: &[(&str, u32, bool)]) -> Vec<u8> {
    let items: Vec<String> = added
        .iter()
        .map(|(c, id, special)| {
            format!(r#"{{"id": {id}, "content": "{c}", "special": {special}}}"#)
        })
        .collect();
    format!(
        r#"{{"added_tokens": [{}], "model": {{"type": "BPE", "vocab": {{}}}}}}"#,
        items.join(", ")
    )
    .into_bytes()
}

/// A spec with `added`, EOS `eos`, end-of-turn `eot` and the shipped `template`.
fn spec(
    added: &[(&str, u32, bool)],
    eos: u32,
    eot: &[u32],
    template: Option<&str>,
) -> TokenizerSpec {
    TokenizerSpec::new(
        model(added),
        SpecialTokens {
            eos: Some(eos),
            end_of_turn: eot.to_vec(),
            ..SpecialTokens::default()
        },
        false,
        false,
        template.map(str::to_owned),
        if template.is_some() {
            TemplateSource::Shipped
        } else {
            TemplateSource::None
        },
    )
}

/// The added tokens of the Qwen3 repositories that matter here, with their shipped IDs and
/// flags: the turn tokens special, the tool-call and reasoning tags not.
const QWEN_ADDED: &[(&str, u32, bool)] = &[
    ("<|endoftext|>", 151_643, true),
    ("<|im_start|>", 151_644, true),
    ("<|im_end|>", 151_645, true),
    ("<tool_call>", 151_657, false),
    ("</tool_call>", 151_658, false),
    ("<tool_response>", 151_665, false),
    ("</tool_response>", 151_666, false),
    ("<think>", 151_667, false),
    ("</think>", 151_668, false),
];

/// A template with the structure of the Qwen3 one: tools listed in the system turn with an
/// example call, reasoning split off on its close tag, calls written as tagged JSON, tool
/// results wrapped in tags, and a generation prompt that writes an empty reasoning section
/// when thinking is off.
const QWEN_TEMPLATE: &str = r#"{%- if tools %}{{- '<|im_start|>system\n# Tools\n<tools>' }}{%- for tool in tools %}{{- tool | tojson }}{%- endfor %}{{- "</tools>\nReturn each call within <tool_call></tool_call> tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}{%- endif %}
{%- for message in messages %}
{%- if message.role == "assistant" %}
{%- if '</think>' in content %}{%- set content = content.split('</think>')[-1] %}{%- endif %}
{{- '<|im_start|>assistant\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- for tool_call in message.tool_calls %}{{- '<tool_call>\n{"name": "' }}{{- tool_call.name }}{{- '", "arguments": ' }}{{- tool_call.arguments | tojson }}{{- '}\n</tool_call>' }}{%- endfor %}
{{- '<|im_end|>\n' }}
{%- elif message.role == "tool" %}{{- '<|im_start|>user\n<tool_response>\n' + content + '\n</tool_response><|im_end|>\n' }}
{%- else %}{{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}
{%- endif %}"#;

fn qwen() -> TokenizerSpec {
    spec(
        QWEN_ADDED,
        151_645,
        &[151_643, 151_645],
        Some(QWEN_TEMPLATE),
    )
}

fn markers(open: Marker, close: Marker, starts_open: bool) -> Option<ReasoningSyntax> {
    Some(ReasoningSyntax::Markers {
        open,
        close,
        starts_open,
    })
}

#[test]
fn qwen_shaped_files_give_tagged_json_and_reasoning_on_token_ids() {
    for rule in [&LLAMA, &QWEN_MOE] {
        let f = resolve(rule, &qwen()).unwrap();
        assert_eq!(f.tool_calls, ToolCallFormat::TaggedJson, "{}", rule.family);
        assert_eq!(
            f.tool_markers.as_slice(),
            [Marker::Special(151_657), Marker::Special(151_658)]
        );
        assert_eq!(
            f.reasoning,
            markers(Marker::Special(151_667), Marker::Special(151_668), false)
        );
        assert_eq!(f.stop.as_slice(), [151_643]);
    }
}

/// Every marker and stop token follows the tokenizer and template: rename the tags, renumber
/// every token and change the end-of-turn set, and the format follows.
#[test]
fn renamed_and_renumbered_specials_are_followed() {
    let renamed = QWEN_TEMPLATE
        .replace("tool_call>", "fn>")
        .replace("think>", "reason>")
        .replace("<|im_end|>", "<|eot|>");
    let added = [
        ("<|eot|>", 7, true),
        ("<|end|>", 8, true),
        ("<|stop|>", 9, true),
        ("<|im_start|>", 10, true),
        ("<fn>", 11, true),
        ("</fn>", 12, true),
        ("<reason>", 13, false),
        ("</reason>", 14, false),
    ];
    let f = resolve(&QWEN_MOE, &spec(&added, 7, &[9, 7, 8], Some(&renamed))).unwrap();
    assert_eq!(f.tool_calls, ToolCallFormat::TaggedJson);
    assert_eq!(
        f.tool_markers.as_slice(),
        [Marker::Special(11), Marker::Special(12)]
    );
    assert_eq!(
        f.reasoning,
        markers(Marker::Special(13), Marker::Special(14), false)
    );
    assert_eq!(f.stop.as_slice(), [8, 9]);
}

/// A marker the tokenizer does not hold as an added token is text.
#[test]
fn markers_outside_the_added_tokens_are_text() {
    let added: Vec<_> = QWEN_ADDED
        .iter()
        .copied()
        .filter(|(c, _, _)| !c.contains("think"))
        .collect();
    let f = resolve(
        &LLAMA,
        &spec(&added, 151_645, &[151_645], Some(QWEN_TEMPLATE)),
    )
    .unwrap();
    assert_eq!(f.tool_markers[0], Marker::Special(151_657));
    assert_eq!(
        f.reasoning,
        markers(
            Marker::Text("<think>".into()),
            Marker::Text("</think>".into()),
            false
        )
    );
    assert!(f.stop.is_empty());
}

/// A generation prompt that opens the section unconditionally starts output inside it, and
/// the open tag is found there even when nothing splits on the close tag.
#[test]
fn a_generation_prompt_that_opens_the_section_starts_open() {
    let t = "{%- for m in messages %}{{ m.content }}{%- endfor %}\
             {%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}\
             {#- the close tag appears in prose: </think> #}";
    let f = resolve(&QWEN_MOE, &spec(QWEN_ADDED, 151_645, &[], Some(t))).unwrap();
    assert_eq!(
        f.reasoning,
        markers(Marker::Special(151_667), Marker::Special(151_668), true)
    );
    assert_eq!(f.tool_calls, ToolCallFormat::None);
}

/// The generation-prompt block ends at its own `endif`, past nested blocks: a tag written after
/// it does not count.
#[test]
fn the_generation_prompt_block_ends_at_its_own_endif() {
    let t = "{%- if add_generation_prompt %}{%- if x %}a{%- endif %}b{%- endif %}<think></think>";
    assert_eq!(generation_prompt(t), Some("{%- if x %}a{%- endif %}b"));
    let f = resolve(&LLAMA, &spec(QWEN_ADDED, 151_645, &[], Some(t))).unwrap();
    assert_eq!(f.reasoning, None);
}

/// A Llama-3-style template writes calls as bare JSON with no marker: no pattern, no reasoning.
#[test]
fn a_template_without_markers_declares_no_pattern() {
    let t = "{%- for m in messages %}{{ '<|start_header_id|>' + m.role + '<|end_header_id|>\\n\\n' }}\
             {%- for c in m.tool_calls %}{{- '{\"name\": \"' + c.name + '\", \"parameters\": ' }}\
             {{- c.arguments | tojson }}{{- '}' }}{%- endfor %}{{ '<|eot_id|>' }}{%- endfor %}\
             {%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}";
    let added = [
        ("<|begin_of_text|>", 128_000, true),
        ("<|end_of_text|>", 128_001, true),
        ("<|start_header_id|>", 128_006, true),
        ("<|end_header_id|>", 128_007, true),
        ("<|eom_id|>", 128_008, true),
        ("<|eot_id|>", 128_009, true),
    ];
    let f = resolve(
        &LLAMA,
        &spec(&added, 128_009, &[128_001, 128_008, 128_009], Some(t)),
    )
    .unwrap();
    assert_eq!(f.tool_calls, ToolCallFormat::None);
    assert!(f.tool_markers.is_empty());
    assert_eq!(f.reasoning, None);
    assert_eq!(f.stop.as_slice(), [128_001, 128_008]);
}

/// A Mistral-class template: a control token, then `[` and the calls. The tools list, which
/// another control token opens with `[` but no `{"name"` object, is not taken.
#[test]
fn a_leading_marker_before_a_list_of_calls_is_a_bare_json_list() {
    let t = "{%- if tools %}{{- '[AVAILABLE_TOOLS] [' }}{%- for t in tools %}{{- '{\"type\": \"function\"}' }}{%- endfor %}{{- '][/AVAILABLE_TOOLS]' }}{%- endif %}\
             {%- for m in messages %}{%- if m.tool_calls %}{{- '[TOOL_CALLS] [' }}\
             {%- for c in m.tool_calls %}{{- '{\"name\": \"' + c.name + '\", \"arguments\": ' + c.arguments|tojson + '}' }}{%- endfor %}\
             {{- ']</s>' }}{%- endif %}{%- endfor %}";
    let added = [
        ("</s>", 2, true),
        ("[AVAILABLE_TOOLS]", 6, true),
        ("[/AVAILABLE_TOOLS]", 7, true),
        ("[TOOL_CALLS]", 9, true),
    ];
    let s = spec(&added, 2, &[2], Some(t));
    let f = resolve(&LLAMA, &s).unwrap();
    assert_eq!(f.tool_calls, ToolCallFormat::BareJsonList);
    assert_eq!(f.tool_markers.as_slice(), [Marker::Special(9)]);
    assert_eq!(
        resolve(&QWEN_MOE, &s),
        Err(ChatFormatError::Undeclared {
            family: "qwen_moe",
            what: "the BareJsonList tool-call pattern".into()
        })
    );
}

#[test]
fn no_template_in_use_declares_nothing_but_the_stop_tokens() {
    let f = resolve(
        &QWEN_MOE,
        &spec(QWEN_ADDED, 151_645, &[151_643, 151_645], None),
    )
    .unwrap();
    assert_eq!(
        f,
        ChatFormat {
            tool_calls: ToolCallFormat::None,
            tool_markers: SmallVec::new(),
            reasoning: None,
            stop: smallvec![151_643],
        }
    );
}

/// The template in use is read: a user override replaces the shipped one.
#[test]
fn the_template_in_use_is_read() {
    let s = qwen().with_template_source(TemplateSource::UserOverride("{{ messages }}".into()));
    let f = resolve(&QWEN_MOE, &s).unwrap();
    assert_eq!(f.tool_calls, ToolCallFormat::None);
    assert_eq!(f.reasoning, None);
}

#[test]
fn an_unclosed_tool_call_tag_is_refused() {
    let t = "{{- '<call>\\n{\"name\": \"' + c.name + '\"}' }}";
    assert_eq!(
        resolve(&LLAMA, &spec(QWEN_ADDED, 151_645, &[], Some(t))),
        Err(ChatFormatError::Unclosed {
            open: "<call>".into()
        })
    );
}

#[test]
fn two_tool_call_tags_are_ambiguous() {
    let t = "<a>{\"name\": 1}</a><b>\\n {\\\"name\\\": 2}</b>";
    let e = resolve(&LLAMA, &spec(QWEN_ADDED, 151_645, &[], Some(t))).unwrap_err();
    assert_eq!(
        e,
        ChatFormatError::Ambiguous {
            role: "tool-call",
            found: ["<a>".into(), "<b>".into()]
        }
    );
    assert_eq!(e.class(), ErrorClass::Unsupported);
}

#[test]
fn two_reasoning_tags_are_ambiguous() {
    let t = "{{ c.split('</think>') }}{{ c.split(\"</reason>\") }}";
    assert!(matches!(
        resolve(&LLAMA, &spec(QWEN_ADDED, 151_645, &[], Some(t))),
        Err(ChatFormatError::Ambiguous {
            role: "reasoning",
            ..
        })
    ));
}

/// A family that declares no reasoning refuses a template that has it.
#[test]
fn undeclared_reasoning_is_refused() {
    let rule = ChatRule {
        family: "test",
        tool_calls: &[ToolCallFormat::TaggedJson],
        reasoning: false,
    };
    assert_eq!(
        resolve(&rule, &qwen()),
        Err(ChatFormatError::Undeclared {
            family: "test",
            what: "reasoning markers".into()
        })
    );
}

/// The tool-call tag is never taken as the reasoning tag, even when content is split on it.
#[test]
fn the_tool_call_tag_is_not_reasoning() {
    let t = "{{- '<tool_call>\\n{\"name\": \"' }}{{ '</tool_call>' }}{{ c.split('</tool_call>') }}";
    let f = resolve(&LLAMA, &spec(QWEN_ADDED, 151_645, &[], Some(t))).unwrap();
    assert_eq!(f.tool_calls, ToolCallFormat::TaggedJson);
    assert_eq!(f.reasoning, None);
}

#[test]
fn an_unreadable_model_is_an_integrity_error() {
    let s = TokenizerSpec::new(
        b"[]".to_vec(),
        SpecialTokens::default(),
        false,
        false,
        None,
        TemplateSource::None,
    );
    let e = resolve(&LLAMA, &s).unwrap_err();
    assert!(matches!(e, ChatFormatError::Tokenizer(_)), "{e}");
    assert_eq!(e.class(), ErrorClass::Integrity);
}

#[test]
fn tags_are_read_whole() {
    assert_eq!(tag_at("<think>x", 0), Some("<think>"));
    assert_eq!(tag_at("</think>", 0), None);
    assert_eq!(tag_at("<>", 0), None);
    assert_eq!(tag_at("< a>", 0), None);
    assert_eq!(tag_at("<a", 0), None);
    assert_eq!(tag_at("<｜Assistant｜>", 0), Some("<｜Assistant｜>"));
    assert_eq!(tag_at(&format!("<{}>", "a".repeat(80)), 0), None);
}

/// Multi-byte text near a list marker's window edge does not split a character.
#[test]
fn the_list_window_respects_character_boundaries() {
    let t = format!("[M] [{}{{\"name\": 1}}", "é".repeat(LIST_WINDOW));
    let added = [("[M]", 3, true)];
    let f = resolve(&LLAMA, &spec(&added, 1, &[], Some(&t))).unwrap();
    assert_eq!(f.tool_calls, ToolCallFormat::None);
}
