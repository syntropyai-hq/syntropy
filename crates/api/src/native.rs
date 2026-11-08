// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native generate for M0 benchmarking (PROD-E4-T13): `POST /v1/syntropy/generate` with token
//! input, streamed as NDJSON (RFC-0050 §13.1, §13.6, §5.1).
//!
//! The M0 benchmark driver (RFC-0061 §3.1) sends token IDs and reads one JSON object per event.
//! This is the part of PROD-E4-T02's endpoint it needs:
//!
//! - **Gate.** The route answers only while `debug.bench_api` is on, and only on a loopback
//!   listener or the local socket, by [`DebugSection::allows`]. Otherwise it answers exactly as a
//!   path with no route does: 404 `unsupported_endpoint`. PROD-E4-T02 opens the endpoint in M1.
//! - **Request.** `{"model", "tokens": [u32], "sampling": {"temperature", "seed", "max_tokens"},
//!   "stream": true, "return_tokens": bool}` with `?format=ndjson`. Any other field is refused:
//!   `messages` and `prompt` with 400 `unsupported_feature`, unknown fields with 400
//!   `invalid_request`. SSE (no `format=ndjson`) and `stream: false` are PROD-E4-T02's and get
//!   400 `unsupported_feature`. A body over [`MAX_BODY`] gets 413 `request_too_large`.
//! - **Range check.** Every ID in `tokens` is checked against the model's `vocab_limit` by
//!   [`Ingress::start`], with PROD-E2-T20's validator, before routing and before any
//!   `SequenceOpen`: 400 `token_id_out_of_range` naming the field, position and limit.
//! - **Stream.** `200` with `Content-Type: application/x-ndjson`, one object per line, each with
//!   `id: "<sequence>:<n>"` counting the events from 0 and `event` naming it: `started`, `text`,
//!   `done` or `error` (RFC-0050 §5.1's native events). With `return_tokens`, each `text` event
//!   carries the `tokens` that released its text, so the IDs across the stream are exactly the
//!   ones `done.usage.completion_tokens` counts.
//!
//! **Driving.** The HTTP handler never touches the rings. It hands the request to a
//! [`GenerateQueue`], and [`drive`], on the thread that owns the runner's [`Ingress`], starts it
//! and polls every live [`Generation`](crate::pipeline::Generation) into its stream. A stream the
//! client drops sets the generation's [`ClientWatch`], so the sequence is cancelled and released
//! (RFC-0050 §5.3).
//!
//! A sequence ID is written `<ingress short ID, 16 hex digits>-<boot>-<n>` here. RFC-0050 does
//! not fix its text form; re-attaching (§5.5, M3) will use this one unless that task changes it.

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
use serde::Deserialize;
use serde_json::{Value, json};
use syntropy_config::{DebugApi, DebugSection, Peer};
use syntropy_core::{ErrorClass, SequenceId};
use tokio::sync::{mpsc as async_mpsc, oneshot};

use crate::pipeline::{
    ApiError, ApiShape, ClientGone, ClientMeta, ClientWatch, EventSink, GenEvent, GenerateRequest,
    Ingress, Input, ModelRef, Progress, SamplingRequest, StopReason, StreamMode,
};
use crate::routes::{ApiState, Body, Reply, error, unsupported_endpoint};
use crate::server::Via;

/// The path.
pub const PATH: &str = "/v1/syntropy/generate";

/// The largest request body read: `api.max_body`'s default (RFC-0054 §5.2), until that key
/// exists in the API's configuration.
pub const MAX_BODY: usize = 32 << 20;

/// The request body, as the M0 endpoint takes it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeRequest {
    model: String,
    tokens: Option<Vec<u32>>,
    /// Refused: text input is PROD-E4-T02's.
    messages: Option<Value>,
    /// Refused: text input is PROD-E4-T02's.
    prompt: Option<Value>,
    #[serde(default)]
    sampling: NativeSampling,
    stream: Option<bool>,
    #[serde(default)]
    return_tokens: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeSampling {
    temperature: Option<f32>,
    seed: Option<u64>,
    max_tokens: Option<u32>,
}

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
    /// The outcome of stages 3 to 9: `Ok` once `SequenceOpen` is sent.
    pub reply: oneshot::Sender<Result<(), ApiError>>,
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

/// A generation's events, from the driving thread to the response body. The body drops its end
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
/// a token reaches its stream as soon as it reaches `local_tokens`; with none live it blocks on
/// `starts`.
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
                    if reply.send(Ok(())).is_ok() {
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

/// `POST /v1/syntropy/generate`.
pub(crate) fn generate(state: &ApiState, request: Request<Incoming>) -> Reply<'_> {
    Box::pin(async move {
        if !gate_open(state.node.debug_gates(), request.extensions().get::<Via>()) {
            return unsupported_endpoint(request.method(), PATH);
        }
        let ndjson = request
            .uri()
            .query()
            .is_some_and(|q| q.split('&').any(|pair| pair == "format=ndjson"));
        if !ndjson {
            return failure(&ApiError::Unsupported(
                "native generate streams NDJSON only until PROD-E4-T02; add `?format=ndjson`"
                    .to_owned(),
            ));
        }
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
        let (request, return_tokens) = match parse(&bytes, client) {
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
        let start = Start {
            request,
            return_tokens,
            watch: watch.clone(),
            events: EventSender(events),
            reply,
        };
        if let Err(e) = queue.submit(start) {
            return failure(&e);
        }
        match replied.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return failure(&e),
            Err(_) => {
                return failure(&ApiError::RunnerUnavailable(
                    "the ingress driver stopped".to_owned(),
                ));
            }
        }
        let body = NdjsonBody {
            events: received,
            watch,
            encoder: Encoder::new(return_tokens),
        };
        let mut response = Response::new(Body::new(body));
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/x-ndjson"),
        );
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        response
    })
}

/// Whether the route answers a request that came `via`: `debug.bench_api` is on, and the request
/// reached a loopback listener from a loopback peer, or the local socket. The listener's address
/// is the bound one, never the accepted stream's, so a wildcard bind is not loopback. A request
/// without its `Via` is refused.
fn gate_open(gates: DebugSection, via: Option<&Via>) -> bool {
    let peer = match via {
        Some(Via::Tcp { local, remote, .. }) => Peer::Tcp {
            listener: local.ip(),
            remote: remote.ip(),
        },
        Some(Via::LocalSocket(_)) => Peer::LocalSocket,
        None => return false,
    };
    gates.allows(DebugApi::Bench, &peer)
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

/// Parses and validates a body into the pipeline's request and `return_tokens`.
fn parse(bytes: &[u8], client: ClientMeta) -> Result<(GenerateRequest, bool), ApiError> {
    let body: NativeRequest = serde_json::from_slice(bytes)
        .map_err(|e| ApiError::InvalidRequest(format!("the body: {e}")))?;
    if body.messages.is_some() || body.prompt.is_some() {
        return Err(ApiError::Unsupported(
            "native generate takes `tokens` input only until PROD-E4-T02".to_owned(),
        ));
    }
    if body.stream == Some(false) {
        return Err(ApiError::Unsupported(
            "native generate streams only until PROD-E4-T02".to_owned(),
        ));
    }
    let tokens = body
        .tokens
        .ok_or_else(|| ApiError::InvalidRequest("`tokens` is required".to_owned()))?;
    if tokens.is_empty() {
        return Err(ApiError::InvalidRequest(
            "`tokens` must hold at least one token".to_owned(),
        ));
    }
    if let Some(t) = body.sampling.temperature
        && !(t.is_finite() && t >= 0.0)
    {
        return Err(ApiError::InvalidRequest(format!(
            "`sampling.temperature` must be a finite number at least 0, not {t}"
        )));
    }
    let request = GenerateRequest {
        shape: ApiShape::Native,
        model: ModelRef(body.model),
        input: Input::Tokens(tokens),
        sampling: SamplingRequest {
            temperature: body.sampling.temperature,
            seed: body.sampling.seed,
            max_tokens: body.sampling.max_tokens,
            ..SamplingRequest::default()
        },
        n: 1,
        logprobs: None,
        stream: StreamMode::Ndjson,
        deadline: None,
        raw_special_tokens: false,
        client,
        trace: false,
    };
    Ok((request, body.return_tokens))
}

/// The status and code a request that fails before its stream starts gets (RFC-0050 §12.1),
/// for the errors the pipeline returns. PROD-E2-T05 owns the full mapping.
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

/// The response body: the generation's events, one JSON line each.
struct NdjsonBody {
    events: async_mpsc::UnboundedReceiver<GenEvent>,
    watch: ClientWatch,
    encoder: Encoder,
}

impl HttpBody for NdjsonBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        loop {
            match self.events.poll_recv(cx) {
                Poll::Ready(Some(event)) => {
                    let lines = self.encoder.encode(event);
                    if !lines.is_empty() {
                        return Poll::Ready(Some(Ok(Frame::data(Bytes::from(lines)))));
                    }
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

impl Drop for NdjsonBody {
    /// The connection is done with the body: if the generation is still live, the client is gone.
    fn drop(&mut self) {
        self.watch.client_gone();
    }
}

/// Turns events into NDJSON lines (RFC-0050 §5.1, native shape with `format=ndjson`).
struct Encoder {
    sequence: String,
    next_id: u64,
    return_tokens: bool,
    /// With `return_tokens`, the text released since the last `Tokens` event.
    held_text: String,
}

impl Encoder {
    fn new(return_tokens: bool) -> Encoder {
        Encoder {
            sequence: String::new(),
            next_id: 0,
            return_tokens,
            held_text: String::new(),
        }
    }

    /// The lines for one event, each ending in `\n`; empty when the event is held.
    fn encode(&mut self, event: GenEvent) -> String {
        let mut out = String::new();
        match event {
            GenEvent::Started {
                seq,
                instance,
                epoch,
                prompt_tokens,
                cached_tokens,
            } => {
                self.sequence = sequence_text(seq);
                let sequence = self.sequence.clone();
                self.line(
                    &mut out,
                    "started",
                    json!({
                        "sequence": sequence,
                        "instance": instance.0,
                        "plan_epoch": { "term": epoch.term, "seq": epoch.seq },
                        "prompt_tokens": prompt_tokens,
                        "cached_tokens": cached_tokens,
                    }),
                );
            }
            GenEvent::TextDelta { text, .. } if self.return_tokens => {
                self.held_text.push_str(&text)
            }
            GenEvent::TextDelta { text, .. } => {
                self.line(&mut out, "text", json!({ "text": text }))
            }
            GenEvent::Tokens { tokens, .. } => {
                let text = std::mem::take(&mut self.held_text);
                self.line(&mut out, "text", json!({ "text": text, "tokens": tokens }));
            }
            GenEvent::ReasoningDelta { text, .. } => {
                self.line(&mut out, "reasoning", json!({ "text": text }));
            }
            GenEvent::ToolCallStart {
                index, id, name, ..
            } => {
                self.line(
                    &mut out,
                    "tool_call",
                    json!({ "index": index, "call_id": id, "name": name }),
                );
            }
            GenEvent::ToolCallArgsDelta { index, json, .. } => {
                self.line(
                    &mut out,
                    "tool_call_delta",
                    json!({ "index": index, "arguments": json }),
                );
            }
            GenEvent::ToolCallEnd { .. } => {}
            GenEvent::Finished {
                reason,
                usage,
                timings,
                ..
            } => {
                self.release_held(&mut out);
                let (finish, stop) = match reason {
                    StopReason::EndOfTurn => ("stop", None),
                    StopReason::Length => ("length", None),
                    StopReason::StopSequence(s) => ("stop_sequence", Some(s)),
                    StopReason::ToolCalls => ("tool_calls", None),
                    StopReason::Cancelled => ("cancelled", None),
                };
                self.line(
                    &mut out,
                    "done",
                    json!({
                        "finish_reason": finish,
                        "stop_sequence": stop,
                        "usage": {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "cached_tokens": usage.cached_tokens,
                        },
                        "timings": {
                            "ttft_ns": timings.ttft.map(|t| t.0),
                            "total_ns": timings.total.0,
                        },
                    }),
                );
            }
            GenEvent::Failed { error } => {
                self.release_held(&mut out);
                let (_, class, code) = status(&error);
                self.line(
                    &mut out,
                    "error",
                    json!({ "error": { "class": class, "code": code, "message": error.to_string() } }),
                );
            }
        }
        out
    }

    /// Text held for a `Tokens` event that will not come.
    fn release_held(&mut self, out: &mut String) {
        if !self.held_text.is_empty() {
            let text = std::mem::take(&mut self.held_text);
            self.line(out, "text", json!({ "text": text, "tokens": [] }));
        }
    }

    fn line(&mut self, out: &mut String, event: &str, mut fields: Value) {
        let id = format!("{}:{}", self.sequence, self.next_id);
        self.next_id += 1;
        if let Value::Object(map) = &mut fields {
            map.insert("id".to_owned(), Value::String(id));
            map.insert("event".to_owned(), Value::String(event.to_owned()));
        }
        // A `Value` always serialises.
        let _ = write!(out, "{fields}");
        out.push('\n');
    }
}

/// A sequence ID as the native API writes it.
fn sequence_text(seq: SequenceId) -> String {
    format!("{:016x}-{}-{}", seq.ingress, seq.boot, seq.n)
}

#[cfg(test)]
mod tests {
    use syntropy_core::{InstanceId, Nanos, PlanEpoch};

    use super::*;
    use crate::pipeline::{Timings, Usage};

    fn lines(out: &str) -> Vec<Value> {
        out.lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

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

    #[test]
    fn events_become_numbered_lines() {
        let mut encoder = Encoder::new(false);
        let mut out = encoder.encode(started());
        out += &encoder.encode(GenEvent::TextDelta {
            choice: 0,
            text: "hi\n".to_owned(),
        });
        out += &encoder.encode(finished(StopReason::StopSequence("\n\n".to_owned())));
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
    fn with_return_tokens_text_travels_with_its_tokens() {
        let mut encoder = Encoder::new(true);
        let _ = encoder.encode(started());
        assert_eq!(
            encoder.encode(GenEvent::TextDelta {
                choice: 0,
                text: "a".to_owned(),
            }),
            ""
        );
        let out = encoder.encode(GenEvent::Tokens {
            choice: 0,
            tokens: vec![3, 4],
        });
        let line = &lines(&out)[0];
        assert_eq!(line["event"], "text");
        assert_eq!(line["text"], "a");
        assert_eq!(line["tokens"], json!([3, 4]));
        // Text held when the stream ends is released before `done`.
        let _ = encoder.encode(GenEvent::TextDelta {
            choice: 0,
            text: "b".to_owned(),
        });
        let out = lines(&encoder.encode(finished(StopReason::Length)));
        assert_eq!(out[0]["text"], "b");
        assert_eq!(out[0]["tokens"], json!([]));
        assert_eq!(out[1]["event"], "done");
        assert_eq!(out[1]["finish_reason"], "length");
    }

    #[test]
    fn the_gate_opens_on_loopback_with_bench_api_only() {
        use crate::bind::BindClass;
        let via = |local: &str, remote: &str, class| Via::Tcp {
            local: local.parse().unwrap(),
            remote: remote.parse().unwrap(),
            class,
        };
        let on = DebugSection {
            bench_api: true,
            ..DebugSection::OFF
        };
        let others = DebugSection {
            parity_api: true,
            bench_api: false,
            telemetry_api: true,
        };
        let v4 = via("127.0.0.1:7433", "127.0.0.1:50000", BindClass::Loopback);
        let v6 = via("[::1]:7433", "[::1]:50000", BindClass::Loopback);
        assert!(gate_open(on, Some(&v4)));
        assert!(gate_open(on, Some(&v6)));
        // Key off, or only the other keys on: closed.
        assert!(!gate_open(DebugSection::OFF, Some(&v4)));
        assert!(!gate_open(others, Some(&v4)));
        // A wildcard or LAN listener, even to a local client; a remote peer on loopback.
        assert!(!gate_open(
            on,
            Some(&via("0.0.0.0:7433", "127.0.0.1:1", BindClass::Lan))
        ));
        assert!(!gate_open(
            on,
            Some(&via("[::]:7433", "[::1]:1", BindClass::Lan))
        ));
        assert!(!gate_open(
            on,
            Some(&via("192.168.1.2:7433", "192.168.1.3:1", BindClass::Lan))
        ));
        assert!(!gate_open(
            on,
            Some(&via("127.0.0.1:7433", "192.168.1.3:1", BindClass::Loopback))
        ));
        // No `Via` at all.
        assert!(!gate_open(on, None));
    }

    #[test]
    fn bodies_are_validated() {
        let ok = |body: &str| parse(body.as_bytes(), ClientMeta::default());
        let (request, return_tokens) = ok(
            r#"{"model":"m","tokens":[1,2],"sampling":{"temperature":0,"seed":7,"max_tokens":3},"stream":true,"return_tokens":true}"#,
        )
        .unwrap();
        assert!(return_tokens);
        assert_eq!(request.input, Input::Tokens(vec![1, 2]));
        assert_eq!(request.sampling.seed, Some(7));
        assert_eq!(request.sampling.max_tokens, Some(3));
        assert_eq!(request.sampling.temperature, Some(0.0));
        assert_eq!(request.shape, ApiShape::Native);
        assert!(!request.raw_special_tokens);

        let code = |body: &str| status(&ok(body).unwrap_err()).2;
        assert_eq!(code(r#"{"model":"m"}"#), "invalid_request");
        assert_eq!(code(r#"{"model":"m","tokens":[]}"#), "invalid_request");
        assert_eq!(code(r#"{"model":"m","tokens":[-1]}"#), "invalid_request");
        assert_eq!(
            code(r#"{"model":"m","tokens":[4294967296]}"#),
            "invalid_request"
        );
        assert_eq!(
            code(r#"{"model":"m","tokens":[1],"bogus":1}"#),
            "invalid_request"
        );
        assert_eq!(
            code(r#"{"model":"m","tokens":[1],"sampling":{"top_p":1}}"#),
            "invalid_request"
        );
        assert_eq!(
            code(r#"{"model":"m","tokens":[1],"sampling":{"temperature":-1}}"#),
            "invalid_request"
        );
        assert_eq!(
            code(r#"{"model":"m","prompt":"hi"}"#),
            "unsupported_feature"
        );
        assert_eq!(
            code(r#"{"model":"m","messages":[]}"#),
            "unsupported_feature"
        );
        assert_eq!(
            code(r#"{"model":"m","tokens":[1],"stream":false}"#),
            "unsupported_feature"
        );
        assert_eq!(code("not json"), "invalid_request");
    }
}
