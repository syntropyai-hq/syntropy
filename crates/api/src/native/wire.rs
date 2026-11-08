// SPDX-License-Identifier: MIT OR Apache-2.0

//! The native generate endpoint's request and response types, as JSON carries them (RFC-0050
//! §13.1, §5.1; ADR-0503 decision 2).
//!
//! These are the one source of truth for the endpoint's JSON: the server parses [`GenerateBody`]
//! and writes [`EventLine`] and [`GenerateResponse`], the CLI's client writes and reads the same
//! types (RFC-0051 §1), and [`crate::openapi`] generates the OpenAPI 3.1 document from their
//! JSON Schemas. A field added here appears in all three at once.
//!
//! Every request type refuses unknown fields, so a misspelt control fails with 400
//! `invalid_request` instead of being ignored. The response types are read with unknown fields
//! ignored, so an older client still reads a newer daemon's events (ADR-0503 decision 4).

use std::collections::BTreeSet;
use std::fmt;

use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::routes::NativeError;

/// The body of `POST /v1/syntropy/generate` (RFC-0050 §13.1).
///
/// Exactly one of `messages`, `prompt` and `tokens` is the input.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GenerateBody {
    /// The model reference (RFC-0050 §4.1).
    pub model: String,
    /// A conversation, rendered with the model's chat template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,
    /// A raw prompt, tokenized without a template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Token IDs, used as the prompt as they are. Each must be below the model's `vocab_limit`
    /// (RFC-0050 §2.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u32>>,
    /// Tool definitions (RFC-0050 §6). Only an empty list is accepted until the tool-calling
    /// tasks (PROD-E3-T02, PROD-E3-T06) build the template and parser paths.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    /// Which tool the model may call: `auto` or `none` without tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// An output constraint (RFC-0050 §7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint: Option<Constraint>,
    /// Reasoning controls (RFC-0050 §8.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// Sampling parameters, stop conditions and the token limit.
    #[serde(default, skip_serializing_if = "Sampling::is_empty")]
    pub sampling: Sampling,
    /// Independent choices; 1 when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>,
    /// `exact` or `approximate`; the instance's default when absent, which is `exact` (D3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fidelity: Option<FidelityName>,
    /// The approximate knobs the request accepts, by name (RFC-0054 §4.2), such as
    /// `topk_reduce`; only with `fidelity: "approximate"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approximate: Option<Vec<String>>,
    /// The native session the request continues, `ses_` and 26 base32 characters (RFC-0050
    /// §9.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
    /// Overrides `kv.autosave` for the session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persist: Option<bool>,
    /// The scheduling priority; `normal` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<PriorityName>,
    /// Asks for deterministic generation, which needs a deterministic instance (ADR-0019
    /// decision 8).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deterministic: Option<bool>,
    /// `auto`, or `off` to switch speculative decoding off. A request cannot force it on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative: Option<Speculative>,
    /// Records every step span (RFC-0053 §3.3); the response then carries `trace_id`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub trace: bool,
    /// Whether the response streams; `true` when absent. A stream is Server-Sent Events, or
    /// NDJSON with `?format=ndjson`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Batches text deltas for this many milliseconds, 0 to 200; 0, the default, sends each
    /// delta as it is released (RFC-0050 §5.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = 200))]
    pub stream_coalesce_ms: Option<u16>,
    /// Includes the token IDs that released each piece of text.
    #[serde(default, skip_serializing_if = "is_false")]
    pub return_tokens: bool,
    /// Tokenizes control-token text in message content as control tokens instead of literally
    /// (RFC-0050 §6.4). `false` by default.
    #[serde(default, skip_serializing_if = "is_false")]
    pub raw_special_tokens: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// One message of a native conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Message {
    /// Who wrote it.
    pub role: Role,
    /// The text, or a list of content blocks.
    pub content: Content,
    /// An assistant turn's reasoning, sent back for the template to keep or drop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// The tools an assistant turn called.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<MessageToolCall>,
    /// The call a `tool` message answers; required on `tool` messages only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Whether the tool failed; `tool` messages only.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
    /// Marks the prompt up to the end of this message as a prefix to share and keep cached
    /// (RFC-0050 §9.1, §9.5), as a block's `cache_control` does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// A message's role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The system prompt.
    System,
    /// The user.
    User,
    /// The model.
    Assistant,
    /// A tool's result.
    Tool,
}

/// A message's content: a string, or content blocks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Content {
    /// Plain text.
    Text(String),
    /// Content blocks, in order.
    Blocks(Vec<ContentBlock>),
}

/// One content block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContentBlock {
    /// Text, tokenized literally unless `raw_special_tokens` (RFC-0050 §6.4).
    Text {
        /// The text.
        text: String,
        /// Marks the prompt up to the end of this block as a prefix to share and keep cached
        /// (RFC-0050 §9.1, §9.5).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

/// A cache breakpoint, as Anthropic content blocks carry it (RFC-0050 §9.1, §9.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CacheControl {
    /// The only kind, `ephemeral`.
    #[serde(rename = "type")]
    pub kind: CacheKind,
    /// How long the prefix stays resident: `5m`, the default, or `1h`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CacheTtl>,
}

/// The kind of a cache breakpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CacheKind {
    /// Kept for the breakpoint's TTL.
    Ephemeral,
}

/// A cache breakpoint's TTL.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum CacheTtl {
    /// Five minutes.
    #[default]
    #[serde(rename = "5m")]
    FiveMinutes,
    /// One hour.
    #[serde(rename = "1h")]
    OneHour,
}

/// A tool call an assistant turn made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MessageToolCall {
    /// The call's ID.
    pub id: String,
    /// The tool's name.
    pub name: String,
    /// The arguments, as JSON text.
    pub arguments: String,
}

/// `tool_choice`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides.
    Auto,
    /// No tool calls.
    None,
    /// A tool call is required.
    Required,
}

/// An output constraint (RFC-0050 §7.1): exactly one of its kinds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Constraint {
    /// Any JSON object.
    #[serde(default, skip_serializing_if = "is_false")]
    pub json_object: bool,
    /// A JSON Schema the output matches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
    /// A regular expression the output matches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    /// A Lark-style grammar the output matches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    /// The output is one of these strings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choice: Option<Vec<String>>,
    /// With `json_schema`: an unsupported keyword fails the request instead of being ignored.
    #[serde(default, skip_serializing_if = "is_false")]
    pub strict: bool,
}

/// Reasoning controls (RFC-0050 §8.3): `effort`, `budget_tokens`, or `enabled: false`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Reasoning {
    /// How much to reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    /// Reasoning tokens allowed before the end-of-reasoning marker is forced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// `false` switches reasoning off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// A reasoning effort.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    /// Least.
    Minimal,
    /// Low.
    Low,
    /// Medium.
    Medium,
    /// High.
    High,
}

/// Sampling parameters, stop conditions and the token limit (RFC-0050 §13.1). Absent fields
/// take the model's defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    /// Temperature, at least 0; 0 is greedy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus mass, in (0, 1].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k; 0 is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Min-p, in [0, 1]; 0 is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    /// Presence penalty, over emitted tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty, over emitted tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Repetition penalty, over the prompt and the emitted tokens; 1 is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// A bias added to each listed token's logit, keyed by token ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<LogitBias>,
    /// The sampler seed; drawn and recorded when absent (RFC-0012 §8.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Stop strings, matched on the output text; a match is not part of the output.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Stop token IDs, beside the model's end-of-sequence and end-of-turn tokens. Each must be
    /// below the model's `vocab_limit` (RFC-0050 §2.3).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_token_ids: Vec<u32>,
    /// Tokens to emit at most; up to the model's context when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Top-k alternatives per position, at most 20.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u8>,
}

impl Sampling {
    fn is_empty(&self) -> bool {
        *self == Sampling::default()
    }
}

/// `logit_bias`: token IDs, written as decimal object keys, each with a bias. The order the
/// client wrote them in is kept, so an error names the entry's position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogitBias(pub Vec<(u32, f32)>);

impl Serialize for LogitBias {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (id, bias) in &self.0 {
            map.serialize_entry(&id.to_string(), bias)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for LogitBias {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Entries;
        impl<'de> Visitor<'de> for Entries {
            type Value = LogitBias;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object from token IDs to biases")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<LogitBias, A::Error> {
                let mut entries: Vec<(u32, f32)> = Vec::new();
                // The IDs seen so far, so a duplicate costs O(log n) to find: a body of
                // `MAX_BODY` bytes holds millions of entries.
                let mut seen = BTreeSet::new();
                while let Some((key, bias)) = map.next_entry::<String, f32>()? {
                    let id = parse_token_key(&key).ok_or_else(|| {
                        serde::de::Error::custom(format_args!(
                            "logit_bias key {key:?} is not a token ID: a decimal number from 0 \
                             to 4294967295"
                        ))
                    })?;
                    if !seen.insert(id) {
                        return Err(serde::de::Error::custom(format_args!(
                            "logit_bias names token ID {id} twice"
                        )));
                    }
                    entries.push((id, bias));
                }
                Ok(LogitBias(entries))
            }
        }
        deserializer.deserialize_map(Entries)
    }
}

/// A decimal token ID with no sign, no leading zero and no other characters.
fn parse_token_key(key: &str) -> Option<u32> {
    let canonical =
        key == "0" || (!key.starts_with('0') && key.bytes().all(|b| b.is_ascii_digit()));
    if canonical && !key.is_empty() {
        key.parse().ok()
    } else {
        None
    }
}

impl JsonSchema for LogitBias {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "LogitBias".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "description": "A bias added to each listed token's logit. The keys are token IDs \
                            in decimal; each must be below the model's `vocab_limit`.",
            "type": "object",
            "propertyNames": { "pattern": "^(0|[1-9][0-9]{0,9})$" },
            "additionalProperties": { "type": "number" }
        })
    }
}

/// `fidelity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FidelityName {
    /// Output within ADR-0019's tolerances.
    Exact,
    /// Approximate knobs allowed, where the instance and the cluster allow them.
    Approximate,
}

/// `priority`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PriorityName {
    /// The one sequence whose step goes first.
    Interactive,
    /// The default.
    Normal,
    /// Runs when the others leave room.
    Background,
}

/// `speculative`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Speculative {
    /// The instance's plan decides.
    Auto,
    /// No speculative decoding for this request.
    Off,
}

/// One event of a native generation stream (RFC-0050 §5.1): an SSE event's `data`, whose SSE
/// `id` and `event` lines repeat its `id` and `event`, or one NDJSON line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EventLine {
    /// `<sequence>:<n>`, with `n` counting this sequence's events from 0, so a client can
    /// re-attach after the last one it saw (RFC-0050 §5.5).
    pub id: String,
    /// The event.
    #[serde(flatten)]
    pub event: Event,
}

/// A native stream's events (RFC-0050 §5.1), named by `event`.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// The sequence is open on an instance.
    Started {
        /// The sequence, as the native API writes its ID.
        sequence: String,
        /// The instance serving it.
        instance: u64,
        /// The plan epoch it was routed under.
        plan_epoch: Epoch,
        /// Tokens in the rendered prompt.
        prompt_tokens: u32,
        /// Prompt tokens served from a cache.
        cached_tokens: u32,
    },
    /// Output text.
    Text {
        /// The text.
        text: String,
        /// With `return_tokens`: the token IDs that released this text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens: Option<Vec<u32>>,
    },
    /// Reasoning text (RFC-0050 §8).
    Reasoning {
        /// The text.
        text: String,
    },
    /// A tool call begins.
    ToolCall {
        /// The call's index within the choice.
        index: u16,
        /// The call's ID.
        call_id: String,
        /// The tool's name.
        name: String,
    },
    /// More of a tool call's arguments.
    ToolCallDelta {
        /// The call's index.
        index: u16,
        /// The next JSON text of its arguments.
        arguments: String,
    },
    /// The choice finished.
    Done {
        /// Why.
        finish_reason: FinishReason,
        /// The stop string that matched, with `stop_sequence`.
        stop_sequence: Option<String>,
        /// Token counts.
        usage: Usage,
        /// Times.
        timings: Timings,
    },
    /// The request failed after its stream began (RFC-0050 §12.2).
    Error {
        /// The error.
        error: NativeError,
    },
}

impl Event {
    /// The event's name: its SSE `event` line and its `event` field.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Event::Started { .. } => "started",
            Event::Text { .. } => "text",
            Event::Reasoning { .. } => "reasoning",
            Event::ToolCall { .. } => "tool_call",
            Event::ToolCallDelta { .. } => "tool_call_delta",
            Event::Done { .. } => "done",
            Event::Error { .. } => "error",
        }
    }
}

/// A plan epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Epoch {
    /// The committing coordinator's lease term.
    pub term: u64,
    /// The plan's sequence number within the term.
    pub seq: u64,
}

/// Why a choice finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model ended its turn, or emitted a stop token ID.
    Stop,
    /// `max_tokens` was reached.
    Length,
    /// A stop string matched.
    StopSequence,
    /// The model called tools.
    ToolCalls,
    /// The sequence was cancelled.
    Cancelled,
}

/// Token counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Usage {
    /// Tokens in the rendered prompt.
    pub prompt_tokens: u32,
    /// Tokens emitted and counted; tokens after a matched stop string are not counted.
    pub completion_tokens: u32,
    /// Prompt tokens served from a cache.
    pub cached_tokens: u32,
}

/// Times, in nanoseconds from the request's arrival at ingress.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Timings {
    /// To the first token, if one arrived.
    pub ttft_ns: Option<u64>,
    /// To the finish.
    pub total_ns: u64,
}

/// The response to a request with `stream: false` (RFC-0050 §13.1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GenerateResponse {
    /// The sequence, as the native API writes its ID.
    pub sequence: String,
    /// The instance that served it.
    pub instance: u64,
    /// The plan epoch it was routed under.
    pub plan_epoch: Epoch,
    /// The choices, by index.
    pub choices: Vec<Choice>,
    /// Token counts over every choice.
    pub usage: Usage,
    /// Times.
    pub timings: Timings,
    /// The trace's ID, 32 lower-case hex digits, when the request asked for `trace`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
}

/// One choice of a [`GenerateResponse`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Choice {
    /// Its index.
    pub index: u8,
    /// The output text.
    pub text: String,
    /// The reasoning text, when the model reasoned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// The tools the model called.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ResponseToolCall>,
    /// With `return_tokens`: every token ID the choice emitted and counted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<u32>>,
    /// Why it finished.
    pub finish_reason: FinishReason,
    /// The stop string that matched, with `stop_sequence`.
    pub stop_sequence: Option<String>,
}

/// A tool call in a [`Choice`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ResponseToolCall {
    /// The call's ID.
    pub call_id: String,
    /// The tool's name.
    pub name: String,
    /// The arguments, as JSON text.
    pub arguments: String,
}

/// The JSON Schema of `ErrorClass` (RFC-0001 §5.12), as the native error carries it: one of the
/// class names.
pub(crate) fn error_class_schema(_: &mut SchemaGenerator) -> Schema {
    json_schema!({
        "description": "What kind of failure it is (RFC-0001 §5.12).",
        "type": "string",
        "enum": ERROR_CLASSES
    })
}

/// Every `ErrorClass` name, in its declaration order.
pub(crate) const ERROR_CLASSES: [&str; 13] = [
    "InvalidRequest",
    "Unsupported",
    "Capacity",
    "Transient",
    "Link",
    "Device",
    "Integrity",
    "Correctness",
    "Drift",
    "Timeout",
    "Auth",
    "Cancelled",
    "Internal",
];

#[cfg(test)]
mod tests {
    use syntropy_core::ErrorClass;

    use super::*;

    #[test]
    fn the_error_class_schema_names_every_class() {
        let all = [
            ErrorClass::InvalidRequest,
            ErrorClass::Unsupported,
            ErrorClass::Capacity,
            ErrorClass::Transient,
            ErrorClass::Link,
            ErrorClass::Device,
            ErrorClass::Integrity,
            ErrorClass::Correctness,
            ErrorClass::Drift,
            ErrorClass::Timeout,
            ErrorClass::Auth,
            ErrorClass::Cancelled,
            ErrorClass::Internal,
        ];
        let names: Vec<Value> = all
            .iter()
            .map(|c| serde_json::to_value(c).unwrap())
            .collect();
        assert_eq!(names, ERROR_CLASSES.map(Value::from));
    }

    #[test]
    fn logit_bias_keeps_order_and_refuses_bad_keys() {
        let bias: LogitBias =
            serde_json::from_str(r#"{"9": 1.5, "0": -2, "4294967295": 0}"#).unwrap();
        assert_eq!(bias.0, [(9, 1.5), (0, -2.0), (u32::MAX, 0.0)]);
        assert_eq!(
            serde_json::to_string(&bias).unwrap(),
            r#"{"9":1.5,"0":-2.0,"4294967295":0.0}"#
        );
        for bad in [
            r#"{"-1": 1}"#,
            r#"{"01": 1}"#,
            r#"{"4294967296": 1}"#,
            r#"{"x": 1}"#,
            r#"{"": 1}"#,
            r#"{"+1": 1}"#,
            r#"{"1": 1, "1": 2}"#,
        ] {
            assert!(serde_json::from_str::<LogitBias>(bad).is_err(), "{bad}");
        }
    }

    /// A map with an entry for every ID below 2^20 reads in one pass: with a linear duplicate
    /// search it would take about 5 × 10^11 comparisons, and the duplicate at its end is still
    /// found.
    #[test]
    fn a_large_logit_bias_reads_in_near_linear_time() {
        let n: u32 = 1 << 20;
        let mut text = String::from("{");
        for id in 0..n {
            let _ = std::fmt::Write::write_fmt(&mut text, format_args!("\"{id}\":1,"));
        }
        let mut duplicated = text.clone();
        text.pop();
        text.push('}');
        duplicated.push_str("\"0\":2}");
        let bias: LogitBias = serde_json::from_str(&text).unwrap();
        assert_eq!(bias.0.len(), n as usize);
        let error = serde_json::from_str::<LogitBias>(&duplicated).unwrap_err();
        assert!(error.to_string().contains("token ID 0 twice"), "{error}");
    }

    #[test]
    fn events_carry_their_name_beside_their_id() {
        let line = EventLine {
            id: "s:3".to_owned(),
            event: Event::Text {
                text: "hi".to_owned(),
                tokens: None,
            },
        };
        let value = serde_json::to_value(&line).unwrap();
        assert_eq!(
            value,
            serde_json::json!({ "id": "s:3", "event": "text", "text": "hi" })
        );
        assert_eq!(serde_json::from_value::<EventLine>(value).unwrap(), line);
        assert_eq!(line.event.name(), "text");
    }
}
