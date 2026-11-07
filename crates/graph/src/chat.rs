// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ChatFormat`: each family's tool-call pattern, reasoning syntax and extra stop tokens, with
//! every marker resolved against the tokenizer (RFC-0043 §8.4, §8.7; ADR-0431 decisions 6
//! and 7; COMP-E17-T04).
//!
//! A family declares which of RFC-0050 §6.2's patterns its checkpoints may use
//! ([`ChatRule`]). [`resolve`] reads the markers of one checkpoint from its [`TokenizerSpec`]:
//! the template in use and the tokenizer's added tokens. No marker string is written down per
//! family or per model size; every one is read from the template's structure.
//!
//! - **Tagged JSON.** The tool-call open tag is a tag `<X>` the template writes directly before
//!   a JSON object whose first key is `"name"` (`<X>\n{"name": …`), and the close tag is
//!   `</X>`, which the template must also write. Between the tag and the object only
//!   whitespace and the escapes `\n`, `\r` and `\t` may stand; the quotes may be escaped
//!   (`\"name\"`), as they are inside a template string literal.
//! - **Bare JSON list.** Taken only when no tagged pattern is found: an added token the
//!   template writes directly before `[`, followed within [`LIST_WINDOW`] bytes, and before any
//!   other added token, by an object whose first key is `"name"`. So a tools list that another
//!   control token opens with `[` is not taken for the calls.
//! - **No pattern.** A template that declares neither gives [`ToolCallFormat::None`], and
//!   RFC-0050 falls back to grammar-constrained JSON. So does a spec with no template in use.
//! - **Reasoning.** The close marker is a tag `</X>` the template splits content on
//!   (`.split('</X>')`), or the close of a tag `<X>` the generation-prompt block writes when
//!   the template also writes `</X>`. `starts_open` is set when the generation-prompt block (the
//!   body of `{% if add_generation_prompt %}`) writes the open marker without the close one, so
//!   the model's output starts inside the section.
//! - **Resolution.** A marker is [`Marker::Special`] with its ID when its text is one of the
//!   tokenizer's added tokens, whether or not `tokenizer.json` marks it `special`, and
//!   [`Marker::Text`] otherwise. An added token always encodes to its own ID, so the model
//!   emits it as that ID, and RFC-0050's parsers match it on the ID (RFC-0043 §8.7 step 5).
//! - **Stop tokens.** `stop` is the tokenizer's end-of-turn set without EOS.
//!
//! [`resolve`] fails closed: two different tool-call tags or reasoning tags, a tool-call open
//! tag with no close, or a pattern the family does not declare is a [`ChatFormatError`].
//! Registration with `syntropy-text`'s parsers is COMP-E17-T12's.

use std::fmt;

use smallvec::SmallVec;
use syntropy_core::ErrorClass;
use syntropy_manifest::tokenizer::{AddedToken, TokenizerSpec, TokenizerSpecError};

#[cfg(test)]
mod tests;

/// How far past a bare-list marker's `[` the template must name a call (`{"name"`), in bytes.
pub const LIST_WINDOW: usize = 256;

/// One of RFC-0050 §6.2's tool-call patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolCallFormat {
    /// `<tool_call>{"name": …, "arguments": …}</tool_call>`.
    TaggedJson,
    /// A begin marker, one `name` + separator + JSON arguments entry per call, an end marker.
    SpecialTokenSections,
    /// `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>` and its variants.
    TaggedKeyValue,
    /// A message header naming a recipient and a content type, then JSON.
    ChannelAddressed,
    /// A leading marker token, then `[{"name": …, "arguments": …}]`.
    BareJsonList,
    /// The template declares no syntax; RFC-0050 constrains a JSON call instead.
    None,
}

/// A marker, parsed on its token ID when the tokenizer defines it as one token and on its text
/// otherwise (RFC-0043 §8.7).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Marker {
    /// An added token's ID.
    Special(u32),
    /// Text that is ordinary in this model's vocabulary.
    Text(String),
}

/// How a model marks its reasoning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReasoningSyntax {
    /// Open and close markers around the section.
    Markers {
        /// The open marker.
        open: Marker,
        /// The close marker.
        close: Marker,
        /// Output starts inside the section: the generation prompt writes `open`.
        starts_open: bool,
    },
    /// A named channel.
    Channel {
        /// The channel's name.
        name: String,
    },
}

/// A checkpoint's chat format (RFC-0043 §8.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatFormat {
    /// The tool-call pattern.
    pub tool_calls: ToolCallFormat,
    /// The pattern's markers: open and close for [`ToolCallFormat::TaggedJson`], the leading
    /// marker for [`ToolCallFormat::BareJsonList`], none for [`ToolCallFormat::None`].
    pub tool_markers: SmallVec<[Marker; 4]>,
    /// The reasoning syntax, when the template declares one.
    pub reasoning: Option<ReasoningSyntax>,
    /// End-of-turn tokens beyond EOS, sorted.
    pub stop: SmallVec<[u32; 4]>,
}

/// What a family declares: the patterns its checkpoints may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatRule {
    /// The family's tag.
    pub family: &'static str,
    /// The tool-call patterns it accepts, besides [`ToolCallFormat::None`].
    pub tool_calls: &'static [ToolCallFormat],
    /// Whether it accepts reasoning markers.
    pub reasoning: bool,
}

/// `llama`: tagged JSON, or a bare JSON list for Mistral-class templates; reasoning markers
/// where the template declares them.
pub const LLAMA: ChatRule = ChatRule {
    family: "llama",
    tool_calls: &[ToolCallFormat::TaggedJson, ToolCallFormat::BareJsonList],
    reasoning: true,
};

/// `qwen_moe`: tagged JSON, with reasoning markers.
pub const QWEN_MOE: ChatRule = ChatRule {
    family: "qwen_moe",
    tool_calls: &[ToolCallFormat::TaggedJson],
    reasoning: true,
};

/// A chat format that cannot be resolved.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatFormatError {
    /// The tokenizer's added tokens cannot be read.
    Tokenizer(TokenizerSpecError),
    /// The template writes two different markers for one role.
    Ambiguous {
        /// `"tool-call"` or `"reasoning"`.
        role: &'static str,
        /// The two marker texts.
        found: [String; 2],
    },
    /// The template writes a tool-call open tag but never its close tag.
    Unclosed {
        /// The open tag.
        open: String,
    },
    /// The template uses a pattern the family does not declare.
    Undeclared {
        /// The family's tag.
        family: &'static str,
        /// What the template uses.
        what: String,
    },
}

impl ChatFormatError {
    /// Its class (RFC-0001 §10): `Integrity` for an unreadable tokenizer, `Unsupported`
    /// otherwise. RFC-0043 §10 names no code for it yet.
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Tokenizer(e) => e.class(),
            _ => ErrorClass::Unsupported,
        }
    }
}

impl fmt::Display for ChatFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokenizer(e) => write!(f, "{e}"),
            Self::Ambiguous { role, found } => write!(
                f,
                "the template writes two {role} markers, {:?} and {:?}",
                found[0], found[1]
            ),
            Self::Unclosed { open } => {
                write!(
                    f,
                    "the template writes the tool-call tag {open:?} but never closes it"
                )
            }
            Self::Undeclared { family, what } => {
                write!(f, "family {family} does not declare {what}")
            }
        }
    }
}

impl std::error::Error for ChatFormatError {}

impl From<TokenizerSpecError> for ChatFormatError {
    fn from(e: TokenizerSpecError) -> Self {
        Self::Tokenizer(e)
    }
}

/// The chat format of the checkpoint `spec` describes, under `rule` (module documentation).
///
/// # Errors
///
/// [`ChatFormatError`] when the added tokens cannot be read, the template is ambiguous or
/// leaves a tool-call tag unclosed, or it uses a pattern `rule` does not declare.
pub fn resolve(rule: &ChatRule, spec: &TokenizerSpec) -> Result<ChatFormat, ChatFormatError> {
    let added = spec.added_tokens()?;
    let marker = |text: &str| resolve_marker(&added, text);
    let s = spec.specials();
    let stop = s
        .end_of_turn
        .iter()
        .copied()
        .filter(|id| Some(*id) != s.eos)
        .collect();
    let Some(template) = spec.template_in_use() else {
        return Ok(ChatFormat {
            tool_calls: ToolCallFormat::None,
            tool_markers: SmallVec::new(),
            reasoning: None,
            stop,
        });
    };

    let tool_tag = tagged_json_tag(template)?;
    let (tool_calls, tool_markers) = match &tool_tag {
        Some(tag) => {
            let close = close_of(tag);
            if !template.contains(&close) {
                return Err(ChatFormatError::Unclosed { open: tag.clone() });
            }
            (
                ToolCallFormat::TaggedJson,
                SmallVec::from_iter([marker(tag), marker(&close)]),
            )
        }
        None => match bare_list_marker(template, &added)? {
            Some(m) => (
                ToolCallFormat::BareJsonList,
                SmallVec::from_iter([marker(&m)]),
            ),
            None => (ToolCallFormat::None, SmallVec::new()),
        },
    };
    if tool_calls != ToolCallFormat::None && !rule.tool_calls.contains(&tool_calls) {
        return Err(ChatFormatError::Undeclared {
            family: rule.family,
            what: format!("the {tool_calls:?} tool-call pattern"),
        });
    }

    let reasoning = match reasoning_tag(template, tool_tag.as_deref())? {
        None => None,
        Some(_) if !rule.reasoning => {
            return Err(ChatFormatError::Undeclared {
                family: rule.family,
                what: "reasoning markers".into(),
            });
        }
        Some(open) => {
            let close = close_of(&open);
            let starts_open = generation_prompt(template)
                .is_some_and(|block| block.contains(&open) && !block.contains(&close));
            Some(ReasoningSyntax::Markers {
                open: marker(&open),
                close: marker(&close),
                starts_open,
            })
        }
    };

    Ok(ChatFormat {
        tool_calls,
        tool_markers,
        reasoning,
        stop,
    })
}

/// `text` as a marker: its added token's ID, or the text itself.
fn resolve_marker(added: &[AddedToken], text: &str) -> Marker {
    added
        .iter()
        .find(|t| t.content == text)
        .map_or_else(|| Marker::Text(text.to_owned()), |t| Marker::Special(t.id))
}

/// `</X>` for the tag `<X>`.
fn close_of(open: &str) -> String {
    format!("</{}", &open[1..])
}

/// The tag `<X>` (not a close tag) that starts at byte `at`, if one does.
fn tag_at(s: &str, at: usize) -> Option<&str> {
    let rest = s.get(at..)?;
    if !rest.starts_with('<') || rest[1..].starts_with('/') {
        return None;
    }
    let end = rest[1..].find(|c: char| {
        c == '>' || c == '<' || c == '"' || c == '\'' || c == '\\' || c.is_whitespace()
    })? + 1;
    (end > 1 && rest[end..].starts_with('>') && end <= 64).then(|| &rest[..=end])
}

/// `s` from byte `at` on, past whitespace and the escapes `\n`, `\r` and `\t`.
fn skip_blank(s: &str, mut at: usize) -> usize {
    let b = s.as_bytes();
    loop {
        match b.get(at) {
            Some(c) if c.is_ascii_whitespace() => at += 1,
            Some(b'\\') if matches!(b.get(at + 1), Some(b'n' | b'r' | b't')) => at += 2,
            _ => return at,
        }
    }
}

/// Whether a JSON object whose first key is `"name"` starts at byte `at`, allowing blanks and
/// escaped quotes.
fn names_a_call(s: &str, at: usize) -> bool {
    let Some(rest) = s.get(at..) else {
        return false;
    };
    let Some(rest) = rest.strip_prefix('{') else {
        return false;
    };
    let after = skip_blank(rest, 0);
    let rest = &rest[after..];
    rest.starts_with("\"name\"") || rest.starts_with("\\\"name\\\"")
}

/// Records `found` as the one marker of `role`, or fails on a second, different one.
fn one(slot: &mut Option<String>, found: &str, role: &'static str) -> Result<(), ChatFormatError> {
    match slot {
        Some(have) if have != found => Err(ChatFormatError::Ambiguous {
            role,
            found: [have.clone(), found.to_owned()],
        }),
        Some(_) => Ok(()),
        None => {
            *slot = Some(found.to_owned());
            Ok(())
        }
    }
}

/// The tag the template writes before a `{"name"` object, if any.
fn tagged_json_tag(template: &str) -> Result<Option<String>, ChatFormatError> {
    let mut tag = None;
    for (at, _) in template.match_indices('<') {
        if let Some(t) = tag_at(template, at)
            && names_a_call(template, skip_blank(template, at + t.len()))
        {
            one(&mut tag, t, "tool-call")?;
        }
    }
    Ok(tag)
}

/// The added token the template writes before `[` and a `{"name"` object, with no other added
/// token between, if any.
fn bare_list_marker(
    template: &str,
    added: &[AddedToken],
) -> Result<Option<String>, ChatFormatError> {
    let mut marker = None;
    for t in added.iter().filter(|t| !t.content.is_empty()) {
        for (at, _) in template.match_indices(t.content.as_str()) {
            let open = skip_blank(template, at + t.content.len());
            if !template[open..].starts_with('[') {
                continue;
            }
            let mut end = template.len().min(open + 1 + LIST_WINDOW);
            while !template.is_char_boundary(end) {
                end -= 1;
            }
            let mut window = &template[open + 1..end];
            if let Some(next) = added
                .iter()
                .filter(|o| !o.content.is_empty())
                .filter_map(|o| window.find(o.content.as_str()))
                .min()
            {
                window = &window[..next];
            }
            let names = window
                .match_indices('{')
                .any(|(i, _)| names_a_call(window, i));
            if names {
                one(&mut marker, &t.content, "tool-call")?;
            }
        }
    }
    Ok(marker)
}

/// The reasoning open tag `<X>`: from `.split('</X>')`, or a tag the generation-prompt block
/// writes whose close the template also writes. The tool-call tag is never taken.
fn reasoning_tag(template: &str, tool: Option<&str>) -> Result<Option<String>, ChatFormatError> {
    let mut open = None;
    for (at, _) in template.match_indices(".split(") {
        let q = skip_blank(template, at + ".split(".len());
        let Some(quote @ ('\'' | '"')) = template[q..].chars().next() else {
            continue;
        };
        let body = &template[q + 1..];
        let Some(end) = body.find(quote) else {
            continue;
        };
        if let Some(name) = body[..end].strip_prefix("</") {
            let tag = format!("<{name}");
            if tag_at(&tag, 0) == Some(tag.as_str()) && Some(tag.as_str()) != tool {
                one(&mut open, &tag, "reasoning")?;
            }
        }
    }
    if let Some(block) = generation_prompt(template) {
        for (at, _) in block.match_indices('<') {
            if let Some(t) = tag_at(block, at)
                && Some(t) != tool
                && template.contains(&close_of(t))
            {
                one(&mut open, t, "reasoning")?;
            }
        }
    }
    Ok(open)
}

/// The body of the template's `{% if add_generation_prompt %}` block, nested blocks included.
fn generation_prompt(template: &str) -> Option<&str> {
    let mut tags = template.match_indices("{%").filter_map(|(at, _)| {
        let end = template[at..].find("%}")? + at;
        let body = template[at + 2..end].trim_matches(|c: char| c == '-' || c == '+');
        Some((at, end + 2, body.trim()))
    });
    let start = tags
        .find(|(_, _, body)| {
            body.strip_prefix("if add_generation_prompt")
                .is_some_and(|rest| rest.is_empty() || rest.starts_with(char::is_whitespace))
        })?
        .1;
    let mut depth = 1usize;
    for (at, _, body) in tags {
        if body.starts_with("if ") {
            depth += 1;
        } else if body == "endif" {
            depth -= 1;
            if depth == 0 {
                return Some(&template[start..at]);
            }
        }
    }
    None
}
