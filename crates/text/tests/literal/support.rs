// SPDX-License-Identifier: MIT OR Apache-2.0

//! The synthetic tokenizers and the two template layouts the tests render.

use syntropy_text::tokenizer::injection::ContentPosition;
use syntropy_text::tokenizer::{self, Span, SpanEncoder, Tokenizer};
use tokenizers::{NormalizedString, Normalizer};

/// A template layout: how the M0 tokenizer families' chat templates frame each message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// `<|im_start|>role\n … <|im_end|>\n` (the Qwen3 tokenizer, both M0 families).
    ImStart,
    /// `<|start_header_id|>role<|end_header_id|>\n\n … <|eot_id|>` (Llama 3).
    Header,
}

/// One synthetic tokenizer laid out like a real one.
pub struct Family {
    pub name: &'static str,
    pub json: &'static str,
    pub layout: Layout,
    /// The tool-call and reasoning marker texts a `ChatFormat` resolves for the layout, for the
    /// injection needles: added tokens (`Marker::Special`) or ordinary text (`Marker::Text`).
    pub markers: &'static [&'static str],
    /// The role headers the layout writes, for the injection needles.
    pub role_headers: &'static [&'static str],
}

/// The synthetic tokenizers: the real files' pre-tokenizer, normalizer, decoder and added-token
/// flags, with a small byte-level vocabulary (`tests/data/tokenizer/*`).
pub const FAMILIES: [Family; 2] = [
    Family {
        name: "qwen3-shaped",
        json: include_str!("../data/tokenizer/qwen3-shaped/tokenizer.json"),
        layout: Layout::ImStart,
        // Tagged JSON and reasoning markers, all added tokens with `special: false`.
        markers: &["<tool_call>", "</tool_call>", "<think>", "</think>"],
        role_headers: &[
            "<|im_start|>system\n",
            "<|im_start|>user\n",
            "<|im_start|>assistant\n",
            "<|im_end|>\n<|im_start|>system\n",
        ],
    },
    Family {
        name: "llama3-shaped",
        json: include_str!("../data/tokenizer/llama3-shaped/tokenizer.json"),
        layout: Layout::Header,
        // The bare-list marker is an added token; the tagged and reasoning tags are ordinary
        // text in this vocabulary, as `Marker::Text`.
        markers: &[
            "<|python_tag|>",
            "<tool_call>",
            "</tool_call>",
            "<think>",
            "</think>",
        ],
        role_headers: &[
            "<|start_header_id|>system<|end_header_id|>\n\n",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>ipython<|end_header_id|>\n\n",
            "<|eot_id|><|start_header_id|>system<|end_header_id|>\n\n",
        ],
    },
];

pub fn encoder(json: &str) -> SpanEncoder {
    let tokenizer = tokenizer::load(json.as_bytes()).expect("the fixture loads");
    SpanEncoder::new(tokenizer).expect("the fixture is supported")
}

/// Renders a conversation with `content` at `position`, between ordinary turns, as a template
/// would: template spans for the framing, content spans for the values.
pub fn render(layout: Layout, position: ContentPosition, content: &str) -> Vec<Span> {
    let mut spans = Vec::new();
    let turn = |spans: &mut Vec<Span>, role: &str, open: &str, value: &str, close: &str| {
        match layout {
            Layout::ImStart => spans.push(Span::template(format!("<|im_start|>{role}\n{open}"))),
            Layout::Header => spans.push(Span::template(format!(
                "<|start_header_id|>{role}<|end_header_id|>\n\n{open}"
            ))),
        }
        spans.push(Span::content(value));
        match layout {
            Layout::ImStart => spans.push(Span::template(format!("{close}<|im_end|>\n"))),
            Layout::Header => spans.push(Span::template(format!("{close}<|eot_id|>"))),
        }
    };
    if layout == Layout::Header {
        spans.push(Span::template("<|begin_of_text|>"));
    }
    let tool_role = match layout {
        Layout::ImStart => "user",
        Layout::Header => "ipython",
    };
    let system = if position == ContentPosition::System {
        content
    } else {
        "Be brief."
    };
    turn(&mut spans, "system", "", system, "");
    match position {
        ContentPosition::System => turn(&mut spans, "user", "", "hello", ""),
        ContentPosition::User => turn(&mut spans, "user", "", content, ""),
        ContentPosition::ToolResult => {
            turn(&mut spans, "user", "", "call the tool", "");
            turn(
                &mut spans,
                tool_role,
                "<tool_response>\n",
                content,
                "\n</tool_response>",
            );
        }
        ContentPosition::Document => turn(
            &mut spans,
            "user",
            "Document:\n",
            content,
            "\n\nSummarise it.",
        ),
    }
    match layout {
        Layout::ImStart => spans.push(Span::template("<|im_start|>assistant\n")),
        Layout::Header => spans.push(Span::template(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )),
    }
    spans
}

/// The spans with every content value emptied: the template's own control tokens, in order.
pub fn template_only(spans: &[Span]) -> Vec<Span> {
    spans
        .iter()
        .map(|s| match s.kind {
            syntropy_text::tokenizer::SpanKind::Template => s.clone(),
            syntropy_text::tokenizer::SpanKind::Content => Span::content(""),
        })
        .collect()
}

pub fn concat(spans: &[Span]) -> String {
    spans.iter().map(|s| s.text.as_str()).collect()
}

/// What the tokenizer's normalizer makes of `text`: what a decode gives back.
pub fn normalized(tokenizer: &Tokenizer, text: &str) -> String {
    let mut n = NormalizedString::from(text);
    if let Some(normalizer) = tokenizer.get_normalizer() {
        normalizer.normalize(&mut n).expect("the normalizer runs");
    }
    n.get().to_owned()
}

pub fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> String {
    tokenizer.decode(ids, false).expect("the IDs decode")
}
