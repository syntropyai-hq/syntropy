// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native generate: `POST /v1/syntropy/generate` (RFC-0050 §13.1, §5.1; PROD-E4-T02, which
//! extends PROD-E4-T13's M0 endpoint).
//!
//! - **Types.** The request and response are [`wire`]'s types, which the CLI's client uses too
//!   and from which [`crate::openapi`] generates the OpenAPI document (ADR-0503 decision 2).
//! - **Request.** [`wire::GenerateBody`]: `messages`, `prompt` or `tokens`, the sampling
//!   fields, stop strings and stop token IDs, reasoning, fidelity and knobs, priority,
//!   determinism, speculation, trace, `return_tokens`, `raw_special_tokens` and `cache_control`
//!   on messages. A misspelt or unknown field is 400 `invalid_request`. A body over
//!   [`MAX_BODY`] is 413 `request_too_large`.
//! - **Not served yet.** Each of these is validated, then refused with 400
//!   `unsupported_feature` naming the task that serves it, so a client never gets a response
//!   that silently ignored what it asked for: tool definitions and `tool_choice: required`
//!   (PROD-E3-T02, PROD-E3-T06), `constraint` (PROD-E3-T05), `logprobs` (PROD-E2-T06), and
//!   `session` and `persist` (PROD-E4-T19). `n > 1` is refused by the pipeline until
//!   PROD-E2-T06. `deterministic: true` and `speculative: "off"` are refused by
//!   [`Stages::resolve_policy`](crate::pipeline::Stages::resolve_policy)'s default, on an
//!   instance that cannot give them.
//! - **Range check.** Every ID in `tokens`, `sampling.logit_bias` and `sampling.stop_token_ids`
//!   is checked against the model's `vocab_limit` by [`Ingress::start`], with PROD-E2-T20's
//!   validator, before routing and before any `SequenceOpen`: 400 `token_id_out_of_range`
//!   naming the field, position and limit.
//! - **Responses.** A stream (the default) is Server-Sent Events, `text/event-stream`, or NDJSON
//!   with `?format=ndjson`. Each event is one [`wire::EventLine`], with `id: "<sequence>:<n>"`
//!   counting the sequence's events from 0; SSE repeats it as the event's `id` line, so a client
//!   can re-attach with `Last-Event-ID` (§5.5, M3). SSE writes `: keepalive` after
//!   [`SSE_KEEPALIVE`] without an event. `stream_coalesce_ms` batches text deltas. With
//!   `stream: false` the response is one [`wire::GenerateResponse`]. A request that carried a
//!   `traceparent` gets one back (RFC-0053 §3.2).
//!
//! **Driving.** The HTTP handler never touches the rings. It hands the request to a
//! [`GenerateQueue`], and [`drive`], on the thread that owns the runner's [`Ingress`], starts it
//! and polls every live [`Generation`](crate::pipeline::Generation) into its response. A client
//! that leaves sets the generation's [`ClientWatch`], so the sequence is cancelled and released
//! (RFC-0050 §5.3).
//!
//! A sequence ID is written `<ingress short ID, 16 hex digits>-<boot>-<n>` here. RFC-0050 does
//! not fix its text form; re-attaching (§5.5, M3) will use this one unless that task changes it.

pub mod wire;

use std::convert::Infallible;
use std::fmt::Write as _;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::task::{Context, Poll};
use std::time::Duration;

use http_body_util::{BodyExt as _, Limited};
use hyper::body::{Body as HttpBody, Bytes, Frame, Incoming, SizeHint};
use hyper::header::{self, HeaderValue};
use hyper::{Request, Response, StatusCode};
use syntropy_config::RequestOverrides;
use syntropy_core::{ApproxKnobs, ErrorClass, Fidelity, Nanos, Priority, SequenceId, SessionId};
use tokio::sync::{mpsc as async_mpsc, oneshot};
use tokio::time::Sleep;

use crate::pipeline::{
    ApiError, ApiShape, ChatMessage, ClientGone, ClientMeta, ClientWatch, ContentPart, EventSink,
    GenEvent, GenerateRequest, Ingress, Input, ModelRef, Progress, ReasoningEffort,
    ReasoningRequest, SamplingRequest, StopReason, StreamMode, ToolCall,
};
use crate::routes::{ApiState, Body, NativeError, Reply, error, json};
use crate::server::Via;
use wire::{
    CacheTtl, Choice, Content, ContentBlock, Effort, Epoch, Event, EventLine, FidelityName,
    FinishReason, GenerateBody, GenerateResponse, Message, PriorityName, Reasoning,
    ResponseToolCall, Role, Speculative, Timings, ToolChoice, Usage,
};

/// The path.
pub const PATH: &str = "/v1/syntropy/generate";

/// The largest request body read: `api.max_body`'s default (RFC-0054 §5.2), until that key
/// exists in the API's configuration.
pub const MAX_BODY: usize = 32 << 20;

/// The time without an event after which an SSE stream writes `: keepalive`: `api.sse_keepalive`'s
/// default of 15 s (RFC-0050 §5.1), until that key exists in the API's configuration.
pub const SSE_KEEPALIVE: Duration = Duration::from_secs(15);

/// The largest `stream_coalesce_ms` (RFC-0050 §5.3).
pub const MAX_COALESCE_MS: u16 = 200;

/// The most logprob alternatives a request may ask for (RFC-0050 §2.1).
const MAX_LOGPROBS: u8 = 20;

/// A request, accepted and validated, for the thread that drives the ingress.
#[derive(Debug)]
pub struct Start {
    /// The request.
    pub request: GenerateRequest,
    /// Whether `text` events carry their tokens.
    pub return_tokens: bool,
    /// Set when the client goes away.
    pub watch: ClientWatch,
    /// Where the generation's events go.
    pub events: EventSender,
    /// The outcome of stages 3 to 9: [`Opened`] once `SequenceOpen` is sent.
    pub reply: oneshot::Sender<Result<Opened, ApiError>>,
}

/// What the driving thread tells the handler once a request's `SequenceOpen` is sent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Opened {
    /// The request's trace ID (RFC-0053 §3.2).
    pub trace_id: [u8; 16],
    /// The `traceparent` the response returns, when the request carried one.
    pub traceparent: Option<String>,
}

/// The HTTP side's handle on the thread that drives the ingress: requests go in, in order.
#[derive(Clone, Debug)]
pub struct GenerateQueue(mpsc::Sender<Start>);

/// The driving thread's side of a [`GenerateQueue`].
#[derive(Debug)]
pub struct StartReceiver(mpsc::Receiver<Start>);

/// A queue and its receiver, for [`drive`].
pub fn generate_queue() -> (GenerateQueue, StartReceiver) {
    let (tx, rx) = mpsc::channel();
    (GenerateQueue(tx), StartReceiver(rx))
}

impl GenerateQueue {
    /// Hands `start` to the driving thread.
    ///
    /// # Errors
    ///
    /// [`ApiError::RunnerUnavailable`] when the driving thread has stopped.
    pub fn submit(&self, start: Start) -> Result<(), ApiError> {
        self.0
            .send(start)
            .map_err(|_| ApiError::RunnerUnavailable("the ingress driver has stopped".to_owned()))
    }
}

/// A generation's events, from the driving thread to the response. The response drops its end
/// when the client goes, and a send then fails as [`ClientGone`].
///
/// The channel is unbounded: one generation sends at most `max_tokens` tokens' events, and
/// `max_tokens` is bounded by the model's context (RFC-0050 §2.2 step 5).
#[derive(Clone, Debug)]
pub struct EventSender(async_mpsc::UnboundedSender<GenEvent>);

impl EventSink for EventSender {
    fn emit(&mut self, event: GenEvent) -> Result<(), ClientGone> {
        self.0.send(event).map_err(|_| ClientGone)
    }
}

/// How long [`drive`] waits for a new request while no generation is live. Requests wake it at
/// once; the wait only bounds how late it sees `stop`.
const IDLE_WAIT: Duration = Duration::from_millis(50);

/// Drives `ingress` until `stop` is set or every [`GenerateQueue`] is dropped: starts each
/// request that arrives on `starts`, and polls every live generation and the ingress's waiting
/// controls. While a generation is live it polls without sleeping, yielding between rounds, so
/// a token reaches its response as soon as it reaches `local_tokens`; with none live it blocks
/// on `starts`.
pub fn drive(ingress: &Ingress<'_>, starts: &StartReceiver, stop: &AtomicBool) {
    let mut live = Vec::new();
    while !stop.load(Ordering::Acquire) {
        let next = if live.is_empty() && ingress.waiting_controls() == 0 {
            match starts.0.recv_timeout(IDLE_WAIT) {
                Ok(start) => Some(start),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            starts.0.try_recv().ok()
        };
        if let Some(start) = next {
            let Start {
                request,
                return_tokens,
                watch,
                events,
                reply,
            } = start;
            match ingress.start(&request, watch) {
                Ok(mut generation) => {
                    generation.return_tokens(return_tokens);
                    let opened = Opened {
                        trace_id: generation.trace().trace_id,
                        traceparent: generation.response_traceparent(),
                    };
                    if reply.send(Ok(opened)).is_ok() {
                        live.push((generation, events));
                    }
                    // A handler that went away before the reply drops the generation, which
                    // cancels and releases its sequence.
                }
                Err(error) => {
                    let _ = reply.send(Err(error));
                }
            }
        }
        live.retain_mut(|(generation, events)| {
            matches!(generation.poll(events), Progress::Pending)
        });
        let _ = ingress.flush();
        if !live.is_empty() {
            std::thread::yield_now();
        }
    }
    // Generations still live are cancelled and released as they drop; send what fits now. The
    // owner flushes again while `waiting_controls` is not 0.
    drop(live);
    let _ = ingress.flush();
}

/// How a response is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Server-Sent Events, the default for a stream.
    Sse,
    /// One JSON object per line, with `?format=ndjson`.
    Ndjson,
    /// One [`GenerateResponse`], with `stream: false`.
    Json,
}

/// A body, parsed and validated.
#[derive(Debug)]
struct Parsed {
    request: GenerateRequest,
    return_tokens: bool,
    mode: Mode,
    /// `stream_coalesce_ms`, when not 0.
    coalesce: Option<Duration>,
}

/// `POST /v1/syntropy/generate`.
pub(crate) fn generate(state: &ApiState, request: Request<Incoming>) -> Reply<'_> {
    Box::pin(async move {
        let ndjson = request
            .uri()
            .query()
            .is_some_and(|q| q.split('&').any(|pair| pair == "format=ndjson"));
        let client = client_meta(&request);
        let bytes = match Limited::new(request.into_body(), MAX_BODY).collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(e) if e.is::<http_body_util::LengthLimitError>() => {
                return error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    ErrorClass::InvalidRequest,
                    "request_too_large",
                    &format!("the body is larger than {MAX_BODY} bytes"),
                );
            }
            Err(e) => {
                return failure(&ApiError::InvalidRequest(format!("reading the body: {e}")));
            }
        };
        let parsed = match parse(&bytes, client, ndjson) {
            Ok(parsed) => parsed,
            Err(e) => return failure(&e),
        };
        let Some(queue) = state.node.generate_queue() else {
            return failure(&ApiError::RunnerUnavailable(
                "no runner serves generation on this node".to_owned(),
            ));
        };
        let (events, received) = async_mpsc::unbounded_channel();
        let (reply, replied) = oneshot::channel();
        let watch = ClientWatch::default();
        let traced = parsed.request.trace;
        let start = Start {
            request: parsed.request,
            return_tokens: parsed.return_tokens,
            watch: watch.clone(),
            events: EventSender(events),
            reply,
        };
        // From here, a handler dropped before it answers means the client went away.
        let gone = GoneOnDrop(Some(watch.clone()));
        if let Err(e) = queue.submit(start) {
            return failure(&e);
        }
        let opened = match replied.await {
            Ok(Ok(opened)) => opened,
            Ok(Err(e)) => return failure(&e),
            Err(_) => {
                return failure(&ApiError::RunnerUnavailable(
                    "the ingress driver stopped".to_owned(),
                ));
            }
        };
        let mut response = match parsed.mode {
            Mode::Json => {
                let trace_id = traced.then(|| hex(&opened.trace_id));
                collect(received, parsed.return_tokens, trace_id).await
            }
            Mode::Sse | Mode::Ndjson => {
                let body = StreamBody::new(
                    received,
                    watch,
                    parsed.mode,
                    parsed.return_tokens,
                    parsed.coalesce,
                );
                let mut response = Response::new(Body::new(body));
                let headers = response.headers_mut();
                let content_type = if parsed.mode == Mode::Sse {
                    "text/event-stream"
                } else {
                    "application/x-ndjson"
                };
                headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
                headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
                headers.insert(
                    header::HeaderName::from_static("x-accel-buffering"),
                    HeaderValue::from_static("no"),
                );
                response
            }
        };
        // The stream's body watches the client from here; a buffered response has ended.
        gone.disarm();
        if let Some(value) = opened
            .traceparent
            .and_then(|t| HeaderValue::from_str(&t).ok())
        {
            response
                .headers_mut()
                .insert(header::HeaderName::from_static("traceparent"), value);
        }
        response
    })
}

/// Sets a [`ClientWatch`] when dropped, unless disarmed: a handler future that is dropped before
/// it answers has lost its client.
struct GoneOnDrop(Option<ClientWatch>);

impl GoneOnDrop {
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for GoneOnDrop {
    fn drop(&mut self) {
        if let Some(watch) = self.0.take() {
            watch.client_gone();
        }
    }
}

/// The client's metadata the pipeline keeps.
fn client_meta(request: &Request<Incoming>) -> ClientMeta {
    let header = |name: header::HeaderName| {
        request
            .headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    ClientMeta {
        peer: match request.extensions().get::<Via>() {
            Some(Via::Tcp { remote, .. }) => Some(*remote),
            _ => None,
        },
        user_agent: header(header::USER_AGENT),
        traceparent: header(header::HeaderName::from_static("traceparent")),
        ..ClientMeta::default()
    }
}

/// Parses and validates a body; `ndjson` is whether the query asked for `format=ndjson`.
fn parse(bytes: &[u8], client: ClientMeta, ndjson: bool) -> Result<Parsed, ApiError> {
    let body: GenerateBody = serde_json::from_slice(bytes)
        .map_err(|e| ApiError::InvalidRequest(format!("the body: {e}")))?;
    validate(body, client, ndjson)
}

fn invalid(why: impl Into<String>) -> ApiError {
    ApiError::InvalidRequest(why.into())
}

fn unsupported(why: impl Into<String>) -> ApiError {
    ApiError::Unsupported(why.into())
}

/// Checks a parsed body and turns it into the pipeline's request.
fn validate(body: GenerateBody, client: ClientMeta, ndjson: bool) -> Result<Parsed, ApiError> {
    let input = input(body.messages, body.prompt, body.tokens)?;

    if !body.tools.is_empty() {
        return Err(unsupported(
            "tool definitions on native generate: the tool-calling tasks (PROD-E3-T02, \
             PROD-E3-T06) serve them",
        ));
    }
    if body.tool_choice == Some(ToolChoice::Required) {
        return Err(invalid("`tool_choice: \"required\"` needs `tools`"));
    }
    if let Some(constraint) = &body.constraint {
        check_constraint(constraint)?;
        return Err(unsupported(
            "`constraint`: output constraints need the masks at the tail (PROD-E3-T05)",
        ));
    }
    let reasoning = reasoning(body.reasoning)?;
    let sampling = sampling(body.sampling)?;
    let logprobs = sampling.1;

    let mut overrides = RequestOverrides {
        fidelity: body.fidelity.map(|f| match f {
            FidelityName::Exact => Fidelity::Exact,
            FidelityName::Approximate => Fidelity::Approximate,
        }),
        knobs: body.approximate.as_deref().map(knobs).transpose()?,
        priority: body.priority.map(|p| match p {
            PriorityName::Interactive => Priority::Interactive,
            PriorityName::Normal => Priority::Normal,
            PriorityName::Background => Priority::Background,
        }),
        speculative_off: body.speculative == Some(Speculative::Off),
        deterministic: body.deterministic,
        trace: body.trace,
        ..RequestOverrides::default()
    };
    if let Some(session) = &body.session {
        let _: SessionId = session.parse().map_err(|e| {
            invalid(format!(
                "`session` is not a session ID (`ses_` and 26 base32 characters): {e}"
            ))
        })?;
        return Err(unsupported(
            "`session`: binding a request to a native session is PROD-E4-T19's",
        ));
    }
    if body.persist.is_some() {
        return Err(invalid("`persist` applies to a `session`"));
    }
    overrides.session = None;

    let mode = match (body.stream, ndjson) {
        (Some(false), true) => {
            return Err(invalid(
                "`?format=ndjson` is a stream format, and the body asks for `stream: false`",
            ));
        }
        (Some(false), false) => Mode::Json,
        (_, true) => Mode::Ndjson,
        (_, false) => Mode::Sse,
    };
    let coalesce = match body.stream_coalesce_ms {
        Some(ms) if ms > MAX_COALESCE_MS => {
            return Err(invalid(format!(
                "`stream_coalesce_ms` is {ms}, above {MAX_COALESCE_MS}"
            )));
        }
        Some(ms) if ms > 0 => Some(Duration::from_millis(u64::from(ms))),
        _ => None,
    };
    let request = GenerateRequest {
        shape: ApiShape::Native,
        model: ModelRef(body.model),
        input,
        reasoning,
        sampling: sampling.0,
        n: body.n.unwrap_or(1),
        logprobs,
        stream: match mode {
            Mode::Sse => StreamMode::Sse {
                include_usage: true,
            },
            Mode::Ndjson => StreamMode::Ndjson,
            Mode::Json => StreamMode::Off,
        },
        deadline: None,
        raw_special_tokens: body.raw_special_tokens,
        overrides,
        client,
        trace: body.trace,
    };
    Ok(Parsed {
        request,
        return_tokens: body.return_tokens,
        mode,
        coalesce,
    })
}

/// The input: exactly one of `messages`, `prompt` and `tokens`.
fn input(
    messages: Option<Vec<Message>>,
    prompt: Option<String>,
    tokens: Option<Vec<u32>>,
) -> Result<Input, ApiError> {
    match (messages, prompt, tokens) {
        (Some(messages), None, None) => {
            if messages.is_empty() {
                return Err(invalid("`messages` must hold at least one message"));
            }
            let messages = messages
                .into_iter()
                .enumerate()
                .map(|(i, m)| message(i, m))
                .collect::<Result<_, _>>()?;
            Ok(Input::Messages {
                system: Vec::new(),
                messages,
            })
        }
        (None, Some(prompt), None) => {
            if prompt.is_empty() {
                return Err(invalid("`prompt` must not be empty"));
            }
            Ok(Input::Prompt {
                prompt,
                suffix: None,
            })
        }
        (None, None, Some(tokens)) => {
            if tokens.is_empty() {
                return Err(invalid("`tokens` must hold at least one token"));
            }
            Ok(Input::Tokens(tokens))
        }
        (None, None, None) => Err(invalid(
            "the body needs one of `messages`, `prompt` and `tokens`",
        )),
        _ => Err(invalid(
            "the body takes only one of `messages`, `prompt` and `tokens`",
        )),
    }
}

/// How long a breakpoint asks its prefix to stay resident (RFC-0050 §9.1).
fn ttl(ttl: Option<CacheTtl>) -> Nanos {
    let minutes = match ttl.unwrap_or_default() {
        CacheTtl::FiveMinutes => 5,
        CacheTtl::OneHour => 60,
    };
    Nanos(minutes * 60 * 1_000_000_000)
}

/// Message `i`, as the pipeline holds it.
fn message(i: usize, m: Message) -> Result<ChatMessage, ApiError> {
    let mut content = Vec::new();
    match m.content {
        Content::Text(text) => content.push(ContentPart::Text(text)),
        Content::Blocks(blocks) => {
            for block in blocks {
                let ContentBlock::Text {
                    text,
                    cache_control,
                } = block;
                content.push(ContentPart::Text(text));
                if let Some(cc) = cache_control {
                    content.push(ContentPart::CacheBreakpoint { ttl: ttl(cc.ttl) });
                }
            }
        }
    }
    if let Some(cc) = m.cache_control {
        content.push(ContentPart::CacheBreakpoint { ttl: ttl(cc.ttl) });
    }
    let only =
        |what: &str, role: &str| invalid(format!("messages[{i}]: `{what}` is for {role} messages"));
    if m.role != Role::Assistant {
        if m.reasoning.is_some() {
            return Err(only("reasoning", "assistant"));
        }
        if !m.tool_calls.is_empty() {
            return Err(only("tool_calls", "assistant"));
        }
    }
    if m.role != Role::Tool {
        if m.tool_call_id.is_some() {
            return Err(only("tool_call_id", "tool"));
        }
        if m.is_error {
            return Err(only("is_error", "tool"));
        }
    }
    Ok(match m.role {
        Role::System => ChatMessage::System(content),
        Role::User => ChatMessage::User(content),
        Role::Assistant => ChatMessage::Assistant {
            content,
            reasoning: m.reasoning,
            tool_calls: m
                .tool_calls
                .into_iter()
                .map(|c| ToolCall {
                    id: c.id,
                    name: c.name,
                    arguments: c.arguments,
                })
                .collect(),
        },
        Role::Tool => ChatMessage::ToolResult {
            call_id: m.tool_call_id.ok_or_else(|| {
                invalid(format!(
                    "messages[{i}]: a tool message needs `tool_call_id`"
                ))
            })?,
            content,
            is_error: m.is_error,
        },
    })
}

/// A constraint names exactly one kind, and `strict` only with `json_schema`.
fn check_constraint(c: &wire::Constraint) -> Result<(), ApiError> {
    let kinds = [
        c.json_object,
        c.json_schema.is_some(),
        c.regex.is_some(),
        c.grammar.is_some(),
        c.choice.is_some(),
    ];
    match kinds.iter().filter(|&&k| k).count() {
        1 => {}
        0 => {
            return Err(invalid(
                "`constraint` names one of `json_object`, `json_schema`, `regex`, `grammar` \
                 and `choice`",
            ));
        }
        _ => {
            return Err(invalid(
                "`constraint` names only one of `json_object`, `json_schema`, `regex`, \
                 `grammar` and `choice`",
            ));
        }
    }
    if c.strict && c.json_schema.is_none() {
        return Err(invalid("`constraint.strict` applies to `json_schema`"));
    }
    if c.choice.as_ref().is_some_and(Vec::is_empty) {
        return Err(invalid("`constraint.choice` must hold at least one string"));
    }
    Ok(())
}

/// `reasoning`: `effort`, `budget_tokens`, or `enabled: false`.
fn reasoning(reasoning: Option<Reasoning>) -> Result<ReasoningRequest, ApiError> {
    let Some(r) = reasoning else {
        return Ok(ReasoningRequest::Default);
    };
    if r.enabled == Some(false) {
        if r.effort.is_some() || r.budget_tokens.is_some() {
            return Err(invalid(
                "`reasoning.enabled: false` takes no `effort` or `budget_tokens`",
            ));
        }
        return Ok(ReasoningRequest::Off);
    }
    match (r.effort, r.budget_tokens) {
        (Some(_), Some(_)) => Err(invalid(
            "`reasoning` takes `effort` or `budget_tokens`, not both",
        )),
        (Some(effort), None) => Ok(ReasoningRequest::Effort(match effort {
            Effort::Minimal => ReasoningEffort::Minimal,
            Effort::Low => ReasoningEffort::Low,
            Effort::Medium => ReasoningEffort::Medium,
            Effort::High => ReasoningEffort::High,
        })),
        (None, Some(budget)) => Ok(ReasoningRequest::Budget(budget)),
        (None, None) => Ok(ReasoningRequest::Default),
    }
}

/// The approximate knobs named in `approximate`, by their RFC-0054 §4.2 names.
fn knobs(names: &[String]) -> Result<ApproxKnobs, ApiError> {
    let mut knobs = ApproxKnobs::empty();
    for name in names {
        let knob = (name.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'))
            .then(|| ApproxKnobs::from_name(&name.to_ascii_uppercase()))
            .flatten()
            .ok_or_else(|| {
                let known: Vec<String> = ApproxKnobs::all()
                    .iter_names()
                    .map(|(n, _)| n.to_ascii_lowercase())
                    .collect();
                invalid(format!(
                    "`approximate` names {name:?}, which is not a knob; the knobs are {}",
                    known.join(", ")
                ))
            })?;
        knobs |= knob;
    }
    Ok(knobs)
}

/// `sampling`, checked: the pipeline's fields, and `logprobs`.
fn sampling(s: wire::Sampling) -> Result<(SamplingRequest, Option<u8>), ApiError> {
    let finite = |name: &str, v: Option<f32>| match v {
        Some(v) if !v.is_finite() => Err(invalid(format!(
            "`sampling.{name}` must be a finite number, not {v}"
        ))),
        _ => Ok(v),
    };
    let temperature = finite("temperature", s.temperature)?;
    if let Some(t) = temperature
        && t < 0.0
    {
        return Err(invalid(format!(
            "`sampling.temperature` must be a finite number at least 0, not {t}"
        )));
    }
    let top_p = finite("top_p", s.top_p)?;
    if let Some(p) = top_p
        && !(p > 0.0 && p <= 1.0)
    {
        return Err(invalid(format!(
            "`sampling.top_p` must be above 0 and at most 1, not {p}"
        )));
    }
    let min_p = finite("min_p", s.min_p)?;
    if let Some(p) = min_p
        && !(0.0..=1.0).contains(&p)
    {
        return Err(invalid(format!(
            "`sampling.min_p` must be from 0 to 1, not {p}"
        )));
    }
    let repetition_penalty = finite("repetition_penalty", s.repetition_penalty)?;
    if let Some(r) = repetition_penalty
        && r <= 0.0
    {
        return Err(invalid(format!(
            "`sampling.repetition_penalty` must be above 0, not {r}"
        )));
    }
    let logit_bias = s.logit_bias.map(|b| b.0).unwrap_or_default();
    if let Some((id, bias)) = logit_bias.iter().find(|(_, b)| !b.is_finite()) {
        return Err(invalid(format!(
            "`sampling.logit_bias` for token ID {id} must be a finite number, not {bias}"
        )));
    }
    if s.stop.iter().any(String::is_empty) {
        return Err(invalid("`sampling.stop` holds an empty string"));
    }
    match s.logprobs {
        Some(k) if k > MAX_LOGPROBS => {
            return Err(invalid(format!(
                "`sampling.logprobs` is {k}, above {MAX_LOGPROBS}"
            )));
        }
        Some(_) => {
            return Err(unsupported(
                "`sampling.logprobs` on native generate: the logprob events are PROD-E2-T06's",
            ));
        }
        None => {}
    }
    let request = SamplingRequest {
        temperature,
        top_p,
        top_k: s.top_k,
        min_p,
        presence_penalty: finite("presence_penalty", s.presence_penalty)?,
        frequency_penalty: finite("frequency_penalty", s.frequency_penalty)?,
        repetition_penalty,
        logit_bias,
        seed: s.seed,
        stop: s.stop,
        stop_token_ids: s.stop_token_ids,
        max_tokens: s.max_tokens,
    };
    Ok((request, s.logprobs))
}

/// The status, class and code of a request that fails before its stream starts (RFC-0050
/// §12.1), for the errors the pipeline returns. PROD-E2-T05 owns the full mapping.
fn status(error: &ApiError) -> (StatusCode, ErrorClass, &'static str) {
    match error {
        ApiError::InvalidRequest(_) => (
            StatusCode::BAD_REQUEST,
            ErrorClass::InvalidRequest,
            "invalid_request",
        ),
        ApiError::Unsupported(_) => (
            StatusCode::BAD_REQUEST,
            ErrorClass::Unsupported,
            "unsupported_feature",
        ),
        ApiError::ModelNotFound(_) => (
            StatusCode::NOT_FOUND,
            ErrorClass::InvalidRequest,
            "model_not_found",
        ),
        ApiError::ContextLengthExceeded { .. } => (
            StatusCode::BAD_REQUEST,
            ErrorClass::InvalidRequest,
            "context_length_exceeded",
        ),
        ApiError::TokenIdOutOfRange(e) => (
            StatusCode::BAD_REQUEST,
            e.class(),
            crate::pipeline::TokenIdOutOfRange::CODE,
        ),
        ApiError::Request(e) => (
            StatusCode::from_u16(e.native_status()).unwrap_or(StatusCode::BAD_REQUEST),
            e.class(),
            e.name(),
        ),
        ApiError::Capacity(_) => (
            StatusCode::TOO_MANY_REQUESTS,
            ErrorClass::Capacity,
            "admission_queue_full",
        ),
        ApiError::RunnerUnavailable(_) | ApiError::SequenceIdsExhausted => (
            StatusCode::SERVICE_UNAVAILABLE,
            ErrorClass::Capacity,
            "instance_unavailable",
        ),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            ErrorClass::Internal,
            "internal_error",
        ),
    }
}

/// The native error for `e`.
fn native_error(e: &ApiError) -> NativeError {
    let (_, class, code) = status(e);
    NativeError {
        class,
        code: code.to_owned(),
        message: e.to_string(),
    }
}

/// The error response for a request that failed before its stream.
fn failure(e: &ApiError) -> Response<Body> {
    let (status, class, code) = status(e);
    let mut response = error(status, class, code, &e.to_string());
    if matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE
    ) {
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    }
    response
}

/// `bytes` as lower-case hex.
fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// Reads a generation's events into one [`GenerateResponse`] (`stream: false`).
async fn collect(
    mut events: async_mpsc::UnboundedReceiver<GenEvent>,
    return_tokens: bool,
    trace_id: Option<String>,
) -> Response<Body> {
    let mut started = None;
    let mut choice = Choice {
        index: 0,
        text: String::new(),
        reasoning: None,
        tool_calls: Vec::new(),
        tokens: return_tokens.then(Vec::new),
        finish_reason: FinishReason::Stop,
        stop_sequence: None,
    };
    while let Some(event) = events.recv().await {
        match event {
            GenEvent::Started {
                seq,
                instance,
                epoch,
                ..
            } => {
                started = Some((
                    sequence_text(seq),
                    instance.0,
                    Epoch {
                        term: epoch.term,
                        seq: epoch.seq,
                    },
                ));
            }
            GenEvent::TextDelta { text, .. } => choice.text.push_str(&text),
            GenEvent::Tokens { tokens, .. } => {
                if let Some(all) = &mut choice.tokens {
                    all.extend(tokens);
                }
            }
            GenEvent::ReasoningDelta { text, .. } => {
                choice.reasoning.get_or_insert_default().push_str(&text);
            }
            GenEvent::ToolCallStart { id, name, .. } => {
                choice.tool_calls.push(ResponseToolCall {
                    call_id: id,
                    name,
                    arguments: String::new(),
                });
            }
            GenEvent::ToolCallArgsDelta { index, json, .. } => {
                if let Some(call) = choice.tool_calls.get_mut(usize::from(index)) {
                    call.arguments.push_str(&json);
                }
            }
            GenEvent::ToolCallEnd { .. } => {}
            GenEvent::Finished {
                reason,
                usage,
                timings,
                ..
            } => {
                let Some((sequence, instance, plan_epoch)) = started else {
                    break;
                };
                (choice.finish_reason, choice.stop_sequence) = finish(reason);
                let response = GenerateResponse {
                    sequence,
                    instance,
                    plan_epoch,
                    choices: vec![choice],
                    usage: usage_body(usage),
                    timings: timings_body(timings),
                    trace_id,
                };
                return json(StatusCode::OK, &response);
            }
            GenEvent::Failed { error } => return failure(&error),
        }
    }
    failure(&ApiError::RunnerUnavailable(
        "the generation ended without finishing".to_owned(),
    ))
}

fn finish(reason: StopReason) -> (FinishReason, Option<String>) {
    match reason {
        StopReason::EndOfTurn => (FinishReason::Stop, None),
        StopReason::Length => (FinishReason::Length, None),
        StopReason::StopSequence(s) => (FinishReason::StopSequence, Some(s)),
        StopReason::ToolCalls => (FinishReason::ToolCalls, None),
        StopReason::Cancelled => (FinishReason::Cancelled, None),
    }
}

fn usage_body(usage: crate::pipeline::Usage) -> Usage {
    Usage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        cached_tokens: usage.cached_tokens,
    }
}

fn timings_body(timings: crate::pipeline::Timings) -> Timings {
    Timings {
        ttft_ns: timings.ttft.map(|t| t.0),
        total_ns: timings.total.0,
    }
}

/// A streamed response's body: the generation's events, as SSE or NDJSON.
struct StreamBody {
    events: async_mpsc::UnboundedReceiver<GenEvent>,
    watch: ClientWatch,
    encoder: Encoder,
    framer: Framer,
    coalescer: Coalescer,
    /// When the held delta is written, with coalescing.
    flush_at: Option<Pin<Box<Sleep>>>,
    /// SSE only: when `: keepalive` is written if nothing else is.
    keepalive: Option<Pin<Box<Sleep>>>,
    out: String,
    done: bool,
}

impl StreamBody {
    fn new(
        events: async_mpsc::UnboundedReceiver<GenEvent>,
        watch: ClientWatch,
        mode: Mode,
        return_tokens: bool,
        coalesce: Option<Duration>,
    ) -> StreamBody {
        StreamBody {
            events,
            watch,
            encoder: Encoder::new(return_tokens),
            framer: Framer::new(mode),
            coalescer: Coalescer {
                window: coalesce,
                held: None,
            },
            flush_at: None,
            keepalive: (mode == Mode::Sse).then(|| Box::pin(tokio::time::sleep(SSE_KEEPALIVE))),
            out: String::new(),
            done: false,
        }
    }

    /// Writes `event`, or holds it to coalesce.
    fn push(&mut self, event: Event) {
        let (ready, started_holding) = self.coalescer.push(event);
        for event in ready {
            self.framer.frame(event, &mut self.out);
        }
        if started_holding && let Some(window) = self.coalescer.window {
            self.flush_at = Some(Box::pin(tokio::time::sleep(window)));
        }
    }

    /// Writes the held delta, if any.
    fn release(&mut self) {
        self.flush_at = None;
        if let Some(event) = self.coalescer.held.take() {
            self.framer.frame(event, &mut self.out);
        }
    }

    /// The bytes written so far, as one frame; the keep-alive restarts.
    fn emit(&mut self) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        if let Some(keepalive) = &mut self.keepalive {
            *keepalive = Box::pin(tokio::time::sleep(SSE_KEEPALIVE));
        }
        let out = std::mem::take(&mut self.out);
        Poll::Ready(Some(Ok(Frame::data(Bytes::from(out)))))
    }
}

impl HttpBody for StreamBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }
        loop {
            match this.events.poll_recv(cx) {
                Poll::Ready(Some(event)) => {
                    for event in this.encoder.encode(event) {
                        this.push(event);
                    }
                    if !this.out.is_empty() {
                        return this.emit();
                    }
                }
                Poll::Ready(None) => {
                    this.release();
                    this.done = true;
                    if this.out.is_empty() {
                        return Poll::Ready(None);
                    }
                    return this.emit();
                }
                Poll::Pending => break,
            }
        }
        if let Some(flush_at) = &mut this.flush_at
            && flush_at.as_mut().poll(cx).is_ready()
        {
            this.release();
            if !this.out.is_empty() {
                return this.emit();
            }
        }
        if let Some(keepalive) = &mut this.keepalive
            && keepalive.as_mut().poll(cx).is_ready()
        {
            this.out.push_str(": keepalive\n\n");
            return this.emit();
        }
        Poll::Pending
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

impl Drop for StreamBody {
    /// The connection is done with the body: if the generation is still live, the client is gone.
    fn drop(&mut self) {
        self.watch.client_gone();
    }
}

/// Holds text and reasoning deltas for `stream_coalesce_ms` and merges those of one kind
/// (RFC-0050 §5.3). Without a window it holds nothing.
struct Coalescer {
    window: Option<Duration>,
    held: Option<Event>,
}

impl Coalescer {
    /// The events ready to write, and whether a new delta began to be held.
    fn push(&mut self, event: Event) -> (Vec<Event>, bool) {
        if self.window.is_none() {
            return (vec![event], false);
        }
        let mergeable = matches!(event, Event::Text { .. } | Event::Reasoning { .. });
        match (&mut self.held, event) {
            (
                Some(Event::Text { text, tokens }),
                Event::Text {
                    text: more,
                    tokens: more_tokens,
                },
            ) => {
                text.push_str(&more);
                if let (Some(tokens), Some(more_tokens)) = (tokens, more_tokens) {
                    tokens.extend(more_tokens);
                }
                (Vec::new(), false)
            }
            (Some(Event::Reasoning { text }), Event::Reasoning { text: more }) => {
                text.push_str(&more);
                (Vec::new(), false)
            }
            (held, event) => {
                let mut ready: Vec<Event> = held.take().into_iter().collect();
                if mergeable {
                    *held = Some(event);
                    (ready, true)
                } else {
                    ready.push(event);
                    (ready, false)
                }
            }
        }
    }
}

/// Numbers events and writes them as SSE events or NDJSON lines.
struct Framer {
    mode: Mode,
    sequence: String,
    next_id: u64,
}

impl Framer {
    fn new(mode: Mode) -> Framer {
        Framer {
            mode,
            sequence: String::new(),
            next_id: 0,
        }
    }

    fn frame(&mut self, event: Event, out: &mut String) {
        if let Event::Started { sequence, .. } = &event {
            self.sequence.clone_from(sequence);
        }
        let line = EventLine {
            id: format!("{}:{}", self.sequence, self.next_id),
            event,
        };
        self.next_id += 1;
        // The wire types always serialise.
        let data = serde_json::to_string(&line).unwrap_or_default();
        match self.mode {
            Mode::Sse => {
                let _ = write!(
                    out,
                    "event: {}\nid: {}\ndata: {data}\n\n",
                    line.event.name(),
                    line.id
                );
            }
            Mode::Ndjson | Mode::Json => {
                out.push_str(&data);
                out.push('\n');
            }
        }
    }
}

/// Turns the pipeline's events into native events (RFC-0050 §5.1).
struct Encoder {
    return_tokens: bool,
    /// With `return_tokens`, the text released since the last `Tokens` event.
    held_text: String,
}

impl Encoder {
    fn new(return_tokens: bool) -> Encoder {
        Encoder {
            return_tokens,
            held_text: String::new(),
        }
    }

    /// The native events for one pipeline event; none when it is held.
    fn encode(&mut self, event: GenEvent) -> Vec<Event> {
        let mut out = Vec::new();
        match event {
            GenEvent::Started {
                seq,
                instance,
                epoch,
                prompt_tokens,
                cached_tokens,
            } => out.push(Event::Started {
                sequence: sequence_text(seq),
                instance: instance.0,
                plan_epoch: Epoch {
                    term: epoch.term,
                    seq: epoch.seq,
                },
                prompt_tokens,
                cached_tokens,
            }),
            GenEvent::TextDelta { text, .. } if self.return_tokens => {
                self.held_text.push_str(&text);
            }
            GenEvent::TextDelta { text, .. } => out.push(Event::Text { text, tokens: None }),
            GenEvent::Tokens { tokens, .. } => out.push(Event::Text {
                text: std::mem::take(&mut self.held_text),
                tokens: Some(tokens),
            }),
            GenEvent::ReasoningDelta { text, .. } => out.push(Event::Reasoning { text }),
            GenEvent::ToolCallStart {
                index, id, name, ..
            } => out.push(Event::ToolCall {
                index,
                call_id: id,
                name,
            }),
            GenEvent::ToolCallArgsDelta { index, json, .. } => out.push(Event::ToolCallDelta {
                index,
                arguments: json,
            }),
            GenEvent::ToolCallEnd { .. } => {}
            GenEvent::Finished {
                reason,
                usage,
                timings,
                ..
            } => {
                self.release_held(&mut out);
                let (finish_reason, stop_sequence) = finish(reason);
                out.push(Event::Done {
                    finish_reason,
                    stop_sequence,
                    usage: usage_body(usage),
                    timings: timings_body(timings),
                });
            }
            GenEvent::Failed { error } => {
                self.release_held(&mut out);
                out.push(Event::Error {
                    error: native_error(&error),
                });
            }
        }
        out
    }

    /// Text held for a `Tokens` event that will not come.
    fn release_held(&mut self, out: &mut Vec<Event>) {
        if !self.held_text.is_empty() {
            out.push(Event::Text {
                text: std::mem::take(&mut self.held_text),
                tokens: Some(Vec::new()),
            });
        }
    }
}

/// A sequence ID as the native API writes it.
fn sequence_text(seq: SequenceId) -> String {
    format!("{:016x}-{}-{}", seq.ingress, seq.boot, seq.n)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use syntropy_core::{InstanceId, PlanEpoch};

    use super::*;
    use crate::pipeline::{Timings, Usage};

    fn started() -> GenEvent {
        GenEvent::Started {
            seq: SequenceId {
                ingress: 0xab,
                boot: 2,
                n: 7,
            },
            instance: InstanceId(4),
            epoch: PlanEpoch { term: 1, seq: 3 },
            prompt_tokens: 5,
            cached_tokens: 0,
        }
    }

    fn finished(reason: StopReason) -> GenEvent {
        GenEvent::Finished {
            choice: 0,
            reason,
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 2,
                cached_tokens: 0,
            },
            timings: Timings {
                ttft: Some(Nanos(10)),
                total: Nanos(30),
            },
        }
    }

    /// Encodes `events` with `mode` and no coalescing.
    fn run(mode: Mode, return_tokens: bool, events: Vec<GenEvent>) -> String {
        let mut encoder = Encoder::new(return_tokens);
        let mut framer = Framer::new(mode);
        let mut out = String::new();
        for event in events {
            for event in encoder.encode(event) {
                framer.frame(event, &mut out);
            }
        }
        out
    }

    fn lines(out: &str) -> Vec<Value> {
        out.lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn events_become_numbered_lines() {
        let out = run(
            Mode::Ndjson,
            false,
            vec![
                started(),
                GenEvent::TextDelta {
                    choice: 0,
                    text: "hi\n".to_owned(),
                },
                finished(StopReason::StopSequence("\n\n".to_owned())),
            ],
        );
        let lines = lines(&out);
        let seq = "00000000000000ab-2-7";
        assert_eq!(lines[0]["id"], format!("{seq}:0"));
        assert_eq!(lines[0]["event"], "started");
        assert_eq!(lines[0]["sequence"], seq);
        assert_eq!(lines[1]["event"], "text");
        assert_eq!(lines[1]["text"], "hi\n");
        assert_eq!(lines[2]["id"], format!("{seq}:2"));
        assert_eq!(lines[2]["finish_reason"], "stop_sequence");
        assert_eq!(lines[2]["stop_sequence"], "\n\n");
        assert_eq!(lines[2]["usage"]["completion_tokens"], 2);
        assert_eq!(lines[2]["timings"]["ttft_ns"], 10);
    }

    #[test]
    fn sse_events_carry_the_id_and_name_twice() {
        let out = run(
            Mode::Sse,
            false,
            vec![
                started(),
                GenEvent::TextDelta {
                    choice: 0,
                    text: "a".to_owned(),
                },
            ],
        );
        let seq = "00000000000000ab-2-7";
        let events: Vec<&str> = out.split("\n\n").filter(|e| !e.is_empty()).collect();
        assert_eq!(events.len(), 2, "{out:?}");
        let text = events[1];
        assert!(
            text.starts_with(&format!("event: text\nid: {seq}:1\ndata: ")),
            "{text}"
        );
        let data: Value = serde_json::from_str(text.split("data: ").nth(1).unwrap()).unwrap();
        assert_eq!(data["id"], format!("{seq}:1"));
        assert_eq!(data["event"], "text");
    }

    #[test]
    fn with_return_tokens_text_travels_with_its_tokens() {
        let mut encoder = Encoder::new(true);
        let _ = encoder.encode(started());
        assert!(
            encoder
                .encode(GenEvent::TextDelta {
                    choice: 0,
                    text: "a".to_owned(),
                })
                .is_empty()
        );
        let out = encoder.encode(GenEvent::Tokens {
            choice: 0,
            tokens: vec![3, 4],
        });
        assert_eq!(
            out,
            [Event::Text {
                text: "a".to_owned(),
                tokens: Some(vec![3, 4])
            }]
        );
        // Text held when the stream ends is released before `done`.
        let _ = encoder.encode(GenEvent::TextDelta {
            choice: 0,
            text: "b".to_owned(),
        });
        let out = encoder.encode(finished(StopReason::Length));
        assert_eq!(
            out[0],
            Event::Text {
                text: "b".to_owned(),
                tokens: Some(Vec::new())
            }
        );
        assert!(matches!(
            out[1],
            Event::Done {
                finish_reason: FinishReason::Length,
                ..
            }
        ));
    }

    #[test]
    fn the_coalescer_merges_deltas_of_one_kind() {
        let text = |t: &str, tokens: &[u32]| Event::Text {
            text: t.to_owned(),
            tokens: Some(tokens.to_vec()),
        };
        let mut c = Coalescer {
            window: Some(Duration::from_millis(5)),
            held: None,
        };
        assert_eq!(c.push(text("a", &[1])), (vec![], true));
        assert_eq!(c.push(text("b", &[2, 3])), (vec![], false));
        let reasoning = Event::Reasoning {
            text: "r".to_owned(),
        };
        // Another kind releases the merged text and is held itself.
        assert_eq!(
            c.push(reasoning.clone()),
            (vec![text("ab", &[1, 2, 3])], true)
        );
        let done = Event::Error {
            error: native_error(&ApiError::Capacity("x".to_owned())),
        };
        assert_eq!(c.push(done.clone()), (vec![reasoning, done], false));
        // Without a window nothing is held.
        let mut off = Coalescer {
            window: None,
            held: None,
        };
        assert_eq!(off.push(text("a", &[])), (vec![text("a", &[])], false));
    }

    fn ok(body: &str) -> Result<Parsed, ApiError> {
        parse(body.as_bytes(), ClientMeta::default(), false)
    }

    fn code(body: &str) -> &'static str {
        status(&ok(body).unwrap_err()).2
    }

    #[test]
    fn every_control_reaches_the_request() {
        let parsed = ok(r#"{
            "model": "m",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]},
                {"role": "user", "content": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"role": "assistant", "content": "", "reasoning": "r", "tool_calls": [{"id": "c1", "name": "f", "arguments": "{}"}]},
                {"role": "tool", "content": "out", "tool_call_id": "c1", "is_error": true}
            ],
            "tool_choice": "auto",
            "reasoning": {"budget_tokens": 77},
            "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.0, "seed": 7,
                         "presence_penalty": 0.5, "frequency_penalty": 0.25, "repetition_penalty": 1.1,
                         "logit_bias": {"5": -1.0, "2": 3.0}, "stop": ["\n\n"], "stop_token_ids": [9, 8],
                         "max_tokens": 64},
            "n": 1,
            "fidelity": "approximate", "approximate": ["topk_reduce", "expert_skip"],
            "priority": "interactive",
            "deterministic": false, "speculative": "auto",
            "trace": true,
            "stream": false, "stream_coalesce_ms": 0,
            "return_tokens": true,
            "raw_special_tokens": true
        }"#)
        .unwrap();
        let r = &parsed.request;
        assert_eq!(parsed.mode, Mode::Json);
        assert!(parsed.return_tokens);
        assert_eq!(parsed.coalesce, None);
        assert!(r.raw_special_tokens);
        assert!(r.trace && r.overrides.trace);
        assert_eq!(r.reasoning, ReasoningRequest::Budget(77));
        assert_eq!(r.sampling.logit_bias, [(5, -1.0), (2, 3.0)]);
        assert_eq!(r.sampling.stop_token_ids, [9, 8]);
        assert_eq!(r.sampling.stop, ["\n\n"]);
        assert_eq!(r.sampling.top_k, Some(40));
        assert_eq!(r.sampling.repetition_penalty, Some(1.1));
        assert_eq!(r.overrides.fidelity, Some(Fidelity::Approximate));
        assert_eq!(
            r.overrides.knobs,
            Some(ApproxKnobs::TOPK_REDUCE | ApproxKnobs::EXPERT_SKIP)
        );
        assert_eq!(r.overrides.priority, Some(Priority::Interactive));
        assert_eq!(r.overrides.deterministic, Some(false));
        assert!(!r.overrides.speculative_off);
        let Input::Messages { messages, .. } = &r.input else {
            panic!("{:?}", r.input)
        };
        let five = Nanos(300_000_000_000);
        let hour = Nanos(3_600_000_000_000);
        assert_eq!(
            messages[0],
            ChatMessage::System(vec![
                ContentPart::Text("sys".to_owned()),
                ContentPart::CacheBreakpoint { ttl: five }
            ])
        );
        assert_eq!(
            messages[1],
            ChatMessage::User(vec![
                ContentPart::Text("hi".to_owned()),
                ContentPart::CacheBreakpoint { ttl: hour }
            ])
        );
        assert!(
            matches!(&messages[2], ChatMessage::Assistant { reasoning: Some(r), tool_calls, .. }
            if r == "r" && tool_calls[0].id == "c1")
        );
        assert!(
            matches!(&messages[3], ChatMessage::ToolResult { call_id, is_error: true, .. }
            if call_id == "c1")
        );

        let stream = |body: &str, ndjson| parse(body.as_bytes(), ClientMeta::default(), ndjson);
        let tokens = r#"{"model":"m","tokens":[1],"stream_coalesce_ms":200}"#;
        let sse = stream(tokens, false).unwrap();
        assert_eq!(sse.mode, Mode::Sse);
        assert_eq!(sse.coalesce, Some(Duration::from_millis(200)));
        assert_eq!(stream(tokens, true).unwrap().mode, Mode::Ndjson);
        for (body, want) in [
            (
                r#"{"reasoning":{"effort":"low"}}"#,
                ReasoningRequest::Effort(ReasoningEffort::Low),
            ),
            (r#"{"reasoning":{"enabled":false}}"#, ReasoningRequest::Off),
            (
                r#"{"reasoning":{"enabled":true}}"#,
                ReasoningRequest::Default,
            ),
            (r#"{}"#, ReasoningRequest::Default),
        ] {
            let mut value: Value = serde_json::from_str(body).unwrap();
            value["model"] = json!("m");
            value["prompt"] = json!("p");
            let parsed = ok(&value.to_string()).unwrap();
            assert_eq!(parsed.request.reasoning, want, "{body}");
        }
        let off = ok(r#"{"model":"m","prompt":"p","speculative":"off"}"#).unwrap();
        assert!(off.request.overrides.speculative_off);
    }

    #[test]
    fn bad_bodies_are_refused() {
        for (body, want) in [
            (r#"{"model":"m"}"#, "invalid_request"),
            (r#"{"model":"m","tokens":[]}"#, "invalid_request"),
            (r#"{"model":"m","prompt":""}"#, "invalid_request"),
            (r#"{"model":"m","messages":[]}"#, "invalid_request"),
            (
                r#"{"model":"m","tokens":[1],"prompt":"p"}"#,
                "invalid_request",
            ),
            (r#"{"model":"m","tokens":[-1]}"#, "invalid_request"),
            (r#"{"model":"m","tokens":[4294967296]}"#, "invalid_request"),
            (r#"{"model":"m","tokens":[1],"bogus":1}"#, "invalid_request"),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"top_q":1}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"temperature":-1}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"top_p":0}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"min_p":2}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"repetition_penalty":0}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"stop":[""]}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"logprobs":21}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"logit_bias":{"x":1}}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"stop_token_ids":[4294967296]}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"stream_coalesce_ms":201}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"approximate":["warp"]}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"approximate":["TOPK_REDUCE"]}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"fidelity":"fuzzy"}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"priority":"urgent"}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"speculative":"on"}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"session":"nope"}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"persist":true}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"tool_choice":"required"}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"reasoning":{"effort":"low","budget_tokens":3}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"reasoning":{"enabled":false,"effort":"low"}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"constraint":{}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"constraint":{"regex":"a","choice":["a"]}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"constraint":{"regex":"a","strict":true}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","tokens":[1],"constraint":{"choice":[]}}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"x","tool_call_id":"c"}]}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"x","reasoning":"r"}]}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","messages":[{"role":"tool","content":"x"}]}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":[{"type":"image","url":"u"}]}]}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"x","cache_control":{"type":"persistent"}}]}"#,
                "invalid_request",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"x","cache_control":{"type":"ephemeral","ttl":"2h"}}]}"#,
                "invalid_request",
            ),
            ("not json", "invalid_request"),
            // Validated, then refused until their tasks serve them.
            (
                r#"{"model":"m","tokens":[1],"tools":[{"name":"f"}]}"#,
                "unsupported_feature",
            ),
            (
                r#"{"model":"m","tokens":[1],"constraint":{"json_schema":{"type":"object"},"strict":true}}"#,
                "unsupported_feature",
            ),
            (
                r#"{"model":"m","tokens":[1],"constraint":{"choice":["a","b"]}}"#,
                "unsupported_feature",
            ),
            (
                r#"{"model":"m","tokens":[1],"sampling":{"logprobs":2}}"#,
                "unsupported_feature",
            ),
            (
                r#"{"model":"m","tokens":[1],"session":"ses_00000000000000000000000000"}"#,
                "unsupported_feature",
            ),
        ] {
            assert_eq!(code(body), want, "{body}");
        }
        let both = parse(
            br#"{"model":"m","tokens":[1],"stream":false}"#,
            ClientMeta::default(),
            true,
        )
        .unwrap_err();
        assert_eq!(status(&both).2, "invalid_request");
    }

    #[test]
    fn policy_errors_keep_their_codes_and_native_status() {
        use syntropy_config::RequestError;
        let conflict = ApiError::Request(RequestError::FidelityConflict {
            knobs: ApproxKnobs::KV_LOSSY,
        });
        assert_eq!(
            status(&conflict),
            (
                StatusCode::CONFLICT,
                ErrorClass::InvalidRequest,
                "fidelity_conflict"
            )
        );
        let knobs = ApiError::Request(RequestError::KnobsNeedApproximate);
        assert_eq!(status(&knobs).0, StatusCode::BAD_REQUEST);
        assert_eq!(status(&knobs).2, "knobs_need_approximate");
    }
}
