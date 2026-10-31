// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shape-neutral request and event stream (RFC-0050 §2.1).
//!
//! Each API shape's front end parses its body into one [`GenerateRequest`], and its encoder turns
//! the one stream of [`GenEvent`]s back into its wire format. These are the M0 fields. The fields
//! RFC-0050 §2.1 lists that need types other tasks build come with those tasks: `tools`,
//! `tool_choice` and `parallel_tool_calls` (§6, PROD-E3), `constraint` (§7, PROD-E3-T05),
//! `reasoning` (§8), `overrides` (RFC-0054 §7, PROD-E1-T10), `ContentPart::Media` (PROD-E4-T17),
//! `TextDelta::logprobs` (PROD-E2-T06), the `fidelity` reports on `Started` and `Finished`
//! (PROD-E4-T03), `Loading` (PROD-E2-T09) and `Recovered` (PROD-E2-T18).

use std::fmt;
use std::net::SocketAddr;

use syntropy_core::{InstanceId, Nanos, PlanEpoch, SequenceId};

use super::token_ids::TokenIdOutOfRange;

/// The API shape a request arrived in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ApiShape {
    /// `POST /v1/chat/completions`.
    OpenAiChat,
    /// `POST /v1/responses`.
    OpenAiResponses,
    /// `POST /v1/completions`.
    OpenAiCompletions,
    /// `POST /v1/messages`.
    AnthropicMessages,
    /// The native API (RFC-0050 §13).
    Native,
    /// The local-runner compatibility shape (ADR-0502).
    CompatLocal,
}

/// A model reference as the client wrote it. RFC-0050 §4.1's grammar and its resolution are
/// PROD-E2-T08's; the pipeline passes it to [`Stages::resolve_model`](super::Stages::resolve_model).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModelRef(pub String);

/// The shape-neutral request (RFC-0050 §2.1). Owner: RFC-0050.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerateRequest {
    /// The shape it arrived in.
    pub shape: ApiShape,
    /// The model it names.
    pub model: ModelRef,
    /// The prompt.
    pub input: Input,
    /// Sampling parameters, stop strings and the token limit.
    pub sampling: SamplingRequest,
    /// Independent choices. M0 serves one; `n > 1` forks from the shared prefix (PROD-E2-T06).
    pub n: u8,
    /// Top-k alternatives per position, at most 20. The tail returns them from PROD-E2-T06.
    pub logprobs: Option<u8>,
    /// Whether and how the response streams.
    pub stream: StreamMode,
    /// Time from the request's arrival to its deadline. A duration, not an instant, so the
    /// pipeline reads time only through its [`Clock`](syntropy_core::Clock) (RFC-0060 §2.1 rule
    /// T1); it becomes `SequenceLimits::deadline`.
    pub deadline: Option<Nanos>,
    /// Native only: content may become control tokens (RFC-0050 §6.4). Default false.
    pub raw_special_tokens: bool,
    /// Who sent it.
    pub client: ClientMeta,
    /// Whether the request asked for a full trace: `syntropy.trace = true` in the body or
    /// `Syntropy-Trace: 1` (RFC-0053 §3.2, §3.3). Every step of its sequence is then traced.
    pub trace: bool,
}

/// The prompt (RFC-0050 §2.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Input {
    /// A conversation.
    Messages {
        /// The system prompt, outside the message list as some shapes carry it.
        system: Vec<ContentPart>,
        /// The messages, in order.
        messages: Vec<ChatMessage>,
    },
    /// Legacy completions and fill-in-the-middle.
    Prompt {
        /// The text before the gap.
        prompt: String,
        /// The text after it.
        suffix: Option<String>,
    },
    /// Token IDs, native only.
    Tokens(Vec<u32>),
}

/// One message of a conversation (RFC-0050 §2.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatMessage {
    /// A system message.
    System(Vec<ContentPart>),
    /// A user message.
    User(Vec<ContentPart>),
    /// An assistant turn, with its reasoning and tool calls.
    Assistant {
        /// What it said.
        content: Vec<ContentPart>,
        /// Its reasoning, when the client sends it back.
        reasoning: Option<String>,
        /// The tools it called.
        tool_calls: Vec<ToolCall>,
    },
    /// A tool's result.
    ToolResult {
        /// The call it answers.
        call_id: String,
        /// The result.
        content: Vec<ContentPart>,
        /// Whether the tool failed.
        is_error: bool,
    },
}

/// One part of a message's content. `Media` is PROD-E4-T17's (RFC-0050 §3.6).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContentPart {
    /// Text, tokenized literally (RFC-0050 §6.4).
    Text(String),
}

/// A tool call an assistant turn made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    /// The call's ID.
    pub id: String,
    /// The tool's name.
    pub name: String,
    /// Its arguments, as JSON text.
    pub arguments: String,
}

/// The sampling fields of a request (RFC-0050 §2.1). `None` takes the model's default, or
/// RFC-0054 §5's fallback.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingRequest {
    /// Temperature; 0 is greedy.
    pub temperature: Option<f32>,
    /// Nucleus sampling's mass.
    pub top_p: Option<f32>,
    /// Top-k; 0 is off.
    pub top_k: Option<u32>,
    /// Min-p; 0 is off.
    pub min_p: Option<f32>,
    /// Presence penalty, over emitted tokens.
    pub presence_penalty: Option<f32>,
    /// Frequency penalty, over emitted tokens.
    pub frequency_penalty: Option<f32>,
    /// Repetition penalty, over the prompt and the emitted tokens.
    pub repetition_penalty: Option<f32>,
    /// Logit bias per token ID.
    pub logit_bias: Vec<(u32, f32)>,
    /// The client's seed. Without one ingress draws 64 bits and records them (RFC-0012 §8.2).
    pub seed: Option<u64>,
    /// Stop strings, matched at ingress on detokenized text (RFC-0050 §5.2).
    pub stop: Vec<String>,
    /// Tokens to emit at most; `None` allows up to the model's context.
    pub max_tokens: Option<u32>,
}

/// Whether the response streams (RFC-0050 §2.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StreamMode {
    /// One buffered response.
    #[default]
    Off,
    /// Server-sent events.
    Sse {
        /// Whether a final usage chunk is sent (OpenAI's `stream_options.include_usage`).
        include_usage: bool,
    },
}

/// Who sent a request (RFC-0050 §2.1). Authentication (§11) fills it in before the pipeline runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClientMeta {
    /// The API key's ID, when one was presented.
    pub key_id: Option<String>,
    /// The principal, hashed, whose prefix-cache pages the request may share (RFC-0050 §9.5);
    /// `None` is the local principal.
    pub principal: Option<[u8; 16]>,
    /// The peer's address, for a TCP client.
    pub peer: Option<SocketAddr>,
    /// The `User-Agent` header.
    pub user_agent: Option<String>,
    /// The W3C `traceparent` header, as received. Ingress continues its trace and returns a
    /// `traceparent` only when the request carried one (RFC-0053 §3.2).
    pub traceparent: Option<String>,
}

/// The shape-neutral event stream (RFC-0050 §2.1). Encoders consume it.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum GenEvent {
    /// The sequence is open on an instance.
    Started {
        /// The sequence.
        seq: SequenceId,
        /// The instance serving it.
        instance: InstanceId,
        /// The plan epoch it was routed under.
        epoch: PlanEpoch,
        /// Tokens in the rendered prompt.
        prompt_tokens: u32,
        /// Prompt tokens served from a cache. 0 until the head reports its prefix hit.
        cached_tokens: u32,
    },
    /// Reasoning text (RFC-0050 §8).
    ReasoningDelta {
        /// The choice.
        choice: u8,
        /// The text.
        text: String,
    },
    /// Output text.
    TextDelta {
        /// The choice.
        choice: u8,
        /// The text.
        text: String,
    },
    /// A tool call begins (RFC-0050 §6).
    ToolCallStart {
        /// The choice.
        choice: u8,
        /// The call's index within the choice.
        index: u16,
        /// The call's ID.
        id: String,
        /// The tool's name.
        name: String,
    },
    /// More of a tool call's arguments.
    ToolCallArgsDelta {
        /// The choice.
        choice: u8,
        /// The call's index.
        index: u16,
        /// The next JSON text.
        json: String,
    },
    /// A tool call ends.
    ToolCallEnd {
        /// The choice.
        choice: u8,
        /// The call's index.
        index: u16,
    },
    /// A choice finished.
    Finished {
        /// The choice.
        choice: u8,
        /// Why.
        reason: StopReason,
        /// Token counts.
        usage: Usage,
        /// Time to the first token and in all.
        timings: Timings,
    },
    /// The request failed.
    Failed {
        /// Why.
        error: ApiError,
    },
}

/// Why a choice finished (RFC-0050 §2.1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StopReason {
    /// The model ended its turn: its end-of-sequence or end-of-turn token, or a stop token ID of
    /// the request.
    EndOfTurn,
    /// `max_tokens` was reached.
    Length,
    /// A stop string matched; it is not part of the output.
    StopSequence(String),
    /// The model called tools.
    ToolCalls,
    /// The sequence was cancelled.
    Cancelled,
}

/// Token counts for a finished choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Usage {
    /// Tokens in the rendered prompt.
    pub prompt_tokens: u32,
    /// Tokens emitted and counted. Tokens after a matched stop string are not counted
    /// (RFC-0050 §5.2).
    pub completion_tokens: u32,
    /// Prompt tokens served from a cache.
    pub cached_tokens: u32,
}

/// Times for a finished choice, read from the pipeline's `Clock`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Timings {
    /// From the start of the pipeline to the first token, if one arrived.
    pub ttft: Option<Nanos>,
    /// From the start of the pipeline to the finish.
    pub total: Nanos,
}

/// Why a request failed. PROD-E2-T05 maps each to an HTTP status and each shape's error body,
/// with the `Api` error-code registry (RFC-0050 §12).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ApiError {
    /// The request is malformed or names something invalid.
    InvalidRequest(String),
    /// The request asks for something this build does not serve yet.
    Unsupported(String),
    /// No model matches the reference.
    ModelNotFound(String),
    /// The prompt plus `max_tokens` exceeds the model's context (RFC-0050 §2.2 step 5).
    ContextLengthExceeded {
        /// Prompt tokens plus `max_tokens`.
        need: u64,
        /// The model's context length.
        limit: u32,
    },
    /// A token ID at or above the model's `vocab_limit` (RFC-0050 §2.3): 400
    /// `token_id_out_of_range`, naming the field, the first offending position and the limit.
    TokenIdOutOfRange(TokenIdOutOfRange),
    /// No capacity now: admission refused, or the runner's control ring is full. Retryable.
    Capacity(String),
    /// The runner's rings are gone or broke their protocol.
    RunnerUnavailable(String),
    /// The node has minted every `SequenceId` of this boot (2^32 of them); only a restart, with a
    /// new boot counter, mints more.
    SequenceIdsExhausted,
    /// A sequence-scoped error at a stage (RFC-0001 §10.4): the tail finished the sequence with
    /// this `ErrorCode`, in its numeric form.
    Sequence {
        /// The error's domain.
        domain: u16,
        /// The code within the domain.
        code: u16,
    },
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApiError::InvalidRequest(why) => write!(f, "invalid request: {why}"),
            ApiError::Unsupported(what) => write!(f, "not supported: {what}"),
            ApiError::ModelNotFound(name) => write!(f, "no model matches {name:?}"),
            ApiError::ContextLengthExceeded { need, limit } => write!(
                f,
                "the prompt and max_tokens need {need} tokens, over the model's context of {limit}"
            ),
            ApiError::TokenIdOutOfRange(e) => write!(f, "token ID out of range: {e}"),
            ApiError::Capacity(why) => write!(f, "no capacity: {why}"),
            ApiError::RunnerUnavailable(why) => write!(f, "the runner is unavailable: {why}"),
            ApiError::SequenceIdsExhausted => {
                f.write_str("this node has used every sequence ID of this boot")
            }
            ApiError::Sequence { domain, code } => {
                write!(f, "the sequence failed with error {domain}.{code}")
            }
        }
    }
}

impl std::error::Error for ApiError {}

impl From<TokenIdOutOfRange> for ApiError {
    fn from(e: TokenIdOutOfRange) -> ApiError {
        ApiError::TokenIdOutOfRange(e)
    }
}
