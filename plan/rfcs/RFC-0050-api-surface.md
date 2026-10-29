# RFC-0050: API surface
- Status: Draft
- Area: Product surface and operations
- Depends on: RFC-0001 (§2.3, §3.5, §5.8, §5.9, §6, §8.4, §10.4), RFC-0054, RFC-0011 (routing, preview, explanation), RFC-0012 (sampler, speculative windows), RFC-0022 (prefix cache, sessions), RFC-0034 (threat model), RFC-0042 (approximate knobs), RFC-0043 (tokenizers, templates, family parsers), ADR-0009, ADR-0010, ADR-0015, ADR-0019, ADR-0502, ADR-0503, ADR-0504, ADR-0505
- Date: 2026-10-19

## Summary

Every node daemon serves the same HTTP API, and whichever node receives a request is its ingress (RFC-0001 §6.1). This RFC defines that API:

- **Compatibility surfaces** (ADR-0502): OpenAI Chat Completions, Responses, Completions (legacy, for fill-in-the-middle clients), Embeddings and Models; Anthropic Messages and token counting. Existing clients and SDKs work unchanged against `http://127.0.0.1:7433`. Chat Completions, Messages, stateless Responses and fill-in-the-middle completions are in place by M1, embeddings in M2, and stored Responses in M3 (§1.2).
- **A native API** under `/v1/syntropy/` (ADR-0503): generation with full control (fidelity, sessions, priority, tracing), models and pulls, fit estimates, placement previews and plan explanations, instances, sessions, cluster membership and invites, configuration, live events and metrics. It is HTTP/JSON with Server-Sent Events. There is no gRPC in v1.
- **One shape-neutral pipeline** behind every surface: parse → authenticate → resolve the model → apply the chat template and tokenize → resolve policy (RFC-0054 §7) → route or place → `SequenceOpen` → stream `TokenFrame`s → detokenize, stop strings, reasoning and tool-call parsing → encode in the request's API shape.
- **Automatic placement on first request.** A request for a model that is in the cluster's stores but not loaded triggers `EnsureInstance` and waits for `InstanceRoutable`, with progress on streams. It never fails merely because nobody loaded the model first.
- **Tool calling, structured output and grammar-constrained decoding.** JSON Schema, regular expressions, grammars and forced tool calls compile at ingress to a token-level automaton that the tail's sampler applies as a mask (ADR-0504). Constraints are part of the requested sampler, so they are compatible with exact mode.
- **Reasoning content** is parsed per model family and returned in each API's own field: `reasoning_content`, Responses reasoning items, or Anthropic `thinking` blocks.
- **Fidelity labels** on every response (D3, ADR-0019): `exact` or `approximate`, with the knobs enabled and the knobs that actually changed the output, as extension fields and headers.
- **Errors** map `ErrorClass` (RFC-0001 §5.12) to each shape's status codes and error bodies, before and after the first streamed token.
- **Content is tokenized literally.** A control-token string inside user, tool or document content never becomes a control token, and tool-call and reasoning markers that are special tokens are recognised by token ID. Every token ID a request supplies is range-checked at ingress (§2.3, §6.4).
- **Authentication** (ADR-0505, meeting RFC-0034 §11's requirements): loopback is open for inference and read-only routes, with Host, Origin and CSRF checks against DNS rebinding and cross-site requests. Administrative, session and keyholder routes need the local socket or a key, even on loopback. Every non-loopback bind requires an API key and TLS. TLS comes from a per-node local CA that `syntropy trust` installs on clients. Overlay-only binds may omit TLS, and plain HTTP on a LAN needs an explicit, permanently flagged opt-in.
- **A debug surface for the quality gates**: debug state, teacher-forced parity taps, a per-step telemetry drain and benchmark generation controls. The last three are off by default in every build and loopback-only (§13.6).

## Motivation and goals / non-goals

The research found API breadth to be table stakes: chat clients, coding agents and IDE plugins speak the OpenAI and Anthropic shapes, and a server that lacks one of them simply is not used by those tools (`multi-device-inference.md` §12.2, §15.11). The same survey found two UX failures this API must avoid:
- placement that has to be triggered by hand before any request succeeds, so first requests return "model not found" (§12.3);
- tokens routed through a central process, which collapsed under load (§2.3).

The second is ruled out by RFC-0001 invariants 1–2. This RFC makes sure the API layer does not reintroduce either.

### Goals

- G1. Unmodified OpenAI and Anthropic SDKs and common clients work for chat, streaming, tool calls, structured output, reasoning and embeddings. The supported field set is explicit, and every unsupported field is either ignored with a documented reason or rejected (§3.6).
- G2. Any node can be the ingress. The API never contacts the coordinator on the request path.
- G3. A request for a model that exists in the cluster succeeds after placement, however long that takes, within a stated wait budget.
- G4. Approximate output is never served to a request that did not opt in. A per-model `default_fidelity = approximate` (RFC-0054 amendment A6; ADR-0019 decision 7 as revised) counts as the operator's opt-in for requests that state no fidelity. Every response says which fidelity it got (D3, SC-29).
- G5. The API layer adds a bounded, measured cost per request and per token (Performance section), and a slow client never slows the ring (RFC-0001 §8.4 rule 3).
- G6. Everything the dashboard and CLI show comes from the native API, so scripts can do whatever the UI does.

### Non-goals

- Image, audio, video, file upload, fine-tuning, batch, assistant, vector-store and moderation endpoints. v1 is text-only. A model its publisher ships as multimodal is served for text and labelled "text only in Syntropy" (§3.4). The content-part and modality fields a later family needs are reserved now (§2.1, §3.6), and the first multimodal family RFC-0043 §2.4 plans is image input for `qwen_hybrid`, after 1.0.
- Content filtering or moderation of outputs.
- A full chat client (vision §3.3). From M2 the dashboard has one minimal chat view over the native API (RFC-0051 §5.12), with no history sync or attachments.
- Multi-tenant accounts, billing or quotas beyond per-key rate limits.
- The common local-runner REST shape (`/api/chat`, `/api/generate`, `/api/tags`) as a default surface. It is an opt-in M3 compatibility module (ADR-0502).

## Design

### 1. Listeners, protocols and route table

#### 1.1 Listeners

- **HTTP listener(s)** at `api.bind` (default `127.0.0.1:7433` and `[::1]:7433`), served by `hyper` on the daemon's `ctl` runtime (RFC-0001 §8.1). HTTP/1.1 always; HTTP/2 over TLS when TLS is on, and cleartext HTTP/2 (prior knowledge) on loopback. `TCP_NODELAY` is set on every accepted socket so a streamed delta leaves the host when it is written.
- **Local socket** (`api.local_socket`, default on): a Unix-domain socket at `<runtime dir>/api.sock` (mode 0600, in a 0700 directory) or a named pipe `\\.\pipe\syntropy-<sid>-api` with a same-SID ACL on Windows. The CLI prefers it. The peer's identity is checked with the OS peer-credential call, so the local socket needs no API key even when the HTTP listener does.
- The same routes are served on both. The dashboard's static assets are served at `/` on the HTTP listener (RFC-0051 §5.1).

*Note 2026-10-27 (PROD-E2-T01, as built):* how the listeners start.
- **A missing address family.** A loopback listener whose address family the host lacks is skipped with a warning when another `api.bind` listener binds. Its bind fails with "address not available" or "address family not supported": `[::1]` in a container without IPv6, or on a kernel booted with `ipv6.disable=1`. So the default `["127.0.0.1:7433", "[::1]:7433"]` still starts on `127.0.0.1` there. Any other bind error, and a start where no listener binds, fails the start.
- **A stale local socket.** A socket file left by a daemon that was killed without its cleanup is replaced at start. It is replaced only when it refuses connections and is a socket owned by the daemon's user, read without following a link. A socket that accepts a connection belongs to a running daemon, and the start fails.
- **Header timeout.** Each HTTP/1 connection must send a request's headers within 30 s [target], a chosen budget.
- **Not served yet.** Listeners beyond loopback, which need TLS and keys (§11.1), are refused at start until PROD-E5-T06 and PROD-E5-T01 serve them.

#### 1.2 Route table

| Surface | Method and path | Purpose | Milestone |
|---|---|---|---|
| OpenAI | `POST /v1/chat/completions` | chat, streaming, tools, structured output, reasoning, logprobs | M0 (minimal), M1 (full) |
| OpenAI | `POST /v1/completions` | raw prompt and fill-in-the-middle (`suffix`) | M1 |
| OpenAI | `POST /v1/responses` | Responses API, stateless: no record kept (§3.2) | M1 |
| OpenAI | `POST /v1/responses` with `store`, `previous_response_id` or `background`; `GET`, `DELETE /v1/responses/{id}`; `POST /v1/responses/{id}/cancel`; `GET /v1/responses/{id}/input_items` | stored Responses, on the ingress node only (§9.2); retrieval and deletion need the local socket or a key on loopback (§11.1) | M3 |
| OpenAI | `POST /v1/embeddings` | embeddings (§14) | M2 |
| OpenAI and Anthropic | `GET /v1/models`, `GET /v1/models/{id}` | model list; shape chosen by headers (§3.4) | M0 |
| Anthropic | `POST /v1/messages` | Messages, streaming, tools, thinking, structured output | M1 |
| Anthropic | `POST /v1/messages/count_tokens` | token count after the chat template | M1 |
| Native | `POST /v1/syntropy/generate` | generation with every Syntropy control (§13) | M1 |
| Native | `/v1/syntropy/models…` | list, pull, delete, show, estimate (§13.2) | M1 |
| Native | `/v1/syntropy/placement/preview`, `/v1/syntropy/instances…` | previews, instances, plans, explanations (§13.3) | M1 (one node), M2 |
| Native | `/v1/syntropy/sessions…` | sessions (§9.3): create and bind in M2, for the dashboard's chat view; list, show, persist, fork and delete in M3 | M2, M3 |
| Native | `/v1/syntropy/cluster…`, `/v1/syntropy/nodes…` | membership, invites, approvals, per-node views (§13.4) | M2 (invites M3) |
| Native | `/v1/syntropy/config…` | node and cluster configuration (RFC-0054 §10.2) | M1 |
| Native | `GET /v1/syntropy/events`, `GET /v1/syntropy/metrics/live` | SSE streams for the dashboard (§13.5) | M1 |
| Native | `GET /v1/syntropy/doctor`, `POST /v1/syntropy/debug/dump` | diagnostics (RFC-0053 §5, §6) | M1 |
| Native | `GET /v1/syntropy/debug/state` | cluster state, residency summaries, plans with explanations and predicted costs, resolved-config hash (RFC-0001 §13; §13.6) | M0 |
| Debug | `POST /v1/syntropy/parity` | teacher-forced parity taps; only with `debug.parity_api`, loopback only (§13.6) | M0 |
| Debug | `GET /v1/syntropy/debug/telemetry` | per-step record drain; only with `debug.telemetry_api`, loopback only (§13.6) | M0 |
| Native | `GET /v1/syntropy/version` | build, protocol versions, support tier | M0 |
| Ops | `GET /healthz`, `GET /readyz` | liveness, readiness (no auth on loopback; `dashboard` scope otherwise) | M0 |
| Ops | `GET /metrics` | Prometheus exposition when `obs.prometheus = true` (RFC-0053 §7.1) | M1 |
| Ops | `GET /v1/syntropy/tls/ca` | the node's local CA certificate for `syntropy trust` (§11.1.1); no key | M1 |
| Ops | `GET /v1/syntropy/tls/cluster-cas` | every member's CA certificate with its node-key binding, for `syntropy trust --cluster` (§11.1.1); no key | M2 |
| Native | `POST /v1/syntropy/auth/login-token` (local socket only); `POST /v1/syntropy/auth/exchange` | the dashboard's one-time login URL on loopback (§11.4) | M1 |
| Compat | `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps`, `/api/version` | local-runner shape, behind `api.compat.local_runner` | M3 |

`/readyz` returns 200 when the daemon has a cluster state replica (or is solo) and its API is serving; it does not wait for any model. A load balancer in front of several nodes should use it, not model availability.

### 2. The ingress pipeline

Each API shape has a **front end** that parses its body into one shape-neutral request, and an **encoder** that turns one shape-neutral event stream back into its wire format. Everything between them is shared. This lives in `syntropy-api` (pipeline, front ends, encoders) and `syntropy-text` (templates, tokenizers, detokenizer, stop strings, reasoning and tool-call parsers, constraint compiler), per RFC-0001 §4.1.

#### 2.1 Types

```rust
/// Shape-neutral request. Owner: RFC-0050.
pub struct GenerateRequest {
    pub shape: ApiShape,                 // OpenAiChat | OpenAiResponses | OpenAiCompletions
                                         // | AnthropicMessages | Native | CompatLocal
    pub model: ModelRef,                 // §4.1
    pub input: Input,
    pub tools: Vec<ToolSpec>,            // §6
    pub tool_choice: ToolChoice,         // Auto | None | Required | Named(String)
    pub parallel_tool_calls: bool,
    pub constraint: Option<OutputConstraint>, // §7
    pub reasoning: ReasoningRequest,     // §8
    pub sampling: SamplingRequest,       // temperature, top_p, top_k, min_p, penalties,
                                         // logit_bias, seed, stop strings, max_tokens
    pub n: u8,                           // independent choices; ≤ api.max_n
    pub logprobs: Option<u8>,            // top-k alternatives per position; ≤ 20
    pub stream: StreamMode,              // Off | Sse { include_usage: bool }
    pub overrides: RequestOverrides,     // RFC-0054 §7: fidelity, knobs, priority, session…
    pub deadline: Option<Instant>,
    pub raw_special_tokens: bool,        // native only; default false: content is tokenized literally (§6.4)
    pub client: ClientMeta,              // key id, principal (§9.5), peer address, user agent, traceparent
}

pub enum Input {
    Messages { system: Vec<ContentPart>, messages: Vec<ChatMessage> },
    Prompt { prompt: String, suffix: Option<String> },   // legacy completions, FIM
    Tokens(Vec<u32>),                                    // native only
}
pub enum ChatMessage {
    System(Vec<ContentPart>),
    User(Vec<ContentPart>),
    Assistant { content: Vec<ContentPart>, reasoning: Option<String>, tool_calls: Vec<ToolCall> },
    ToolResult { call_id: String, content: Vec<ContentPart>, is_error: bool },
}
#[non_exhaustive]
pub enum ContentPart {
    Text(String),
    /// Reserved for the first multimodal family (RFC-0043). Parsed so the shape can be checked,
    /// rejected with `unsupported_content_type` in v1 (§3.6). In the token stream it becomes the
    /// placeholder IDs plus content hash that RFC-0022 §7.1 keys prefixes by.
    Media { kind: MediaKind /* Image | Audio | Video | File */, source: MediaSource, hash: Blake3 },
}

/// Shape-neutral event stream produced by the pipeline. Encoders consume it.
pub enum GenEvent {
    Loading { phase: LoadPhase, fraction: f32, eta: Option<Nanos> }, // automatic placement (§4.3)
    Started { seq: SequenceId, instance: InstanceId, epoch: PlanEpoch,
              prompt_tokens: u32, cached_tokens: u32, fidelity: FidelityReport },
    ReasoningDelta { choice: u8, text: String },
    TextDelta { choice: u8, text: String, logprobs: Option<Vec<TokenLogprob>> },
    ToolCallStart { choice: u8, index: u16, id: String, name: String },
    ToolCallArgsDelta { choice: u8, index: u16, json: String },
    ToolCallEnd { choice: u8, index: u16 },
    Recovered { kind: RecoveryKind, gap: Nanos },        // migration, replay or re-prefill (RFC-0001 §7.5)
    Finished { choice: u8, reason: StopReason, usage: Usage, fidelity: FidelityReport, timings: Timings },
    Failed { error: ApiError },
}
pub enum StopReason { EndOfTurn, Length, StopSequence(String), ToolCalls, Cancelled }
```

#### 2.2 Stages

```text
handle(http_request):
  1  auth        = authenticate(http_request)                         // §11; before reading the body; fixes the principal (§9.5)
  2  req         = front_end[shape].parse(body)                       // InvalidRequest on bad input
  3  model       = resolve_model_ref(req.model)                       // §4.1
  4  overrides   = parse_request_overrides(req)                       // RFC-0054 §7; validated, not yet resolved
  5  spans       = text::render(model.template, req.input, req.tools, req.reasoning)  // chat template; template-literal
                                                                      // and content spans kept apart (§6.4)
     tokens      = text::tokenize_spans(model.tokenizer, spans, req.raw_special_tokens) // content literally;
                                                                      // blocking pool if > 16 KiB
     check_token_ids(tokens, req, model)                              // §2.3; 400 token_id_out_of_range
     need        = tokens.len() + max_tokens
     check need ≤ model's maximum context                             // else context_length_exceeded at once
  6  constraint  = text::compile(req.constraint, req.tools, req.tool_choice, model.tokenizer) // §7, cached
  7  instance    = solver::route(model, tokens, session) over the candidates that pass §4.2's filters:
                   resolve_request(overrides, candidate.policy, cluster) succeeds (RFC-0054 §7)
                   and need ≤ the candidate's planned context_length
     if none passes:
       some routable instance failed only on context → fail context_length_exceeded      // §4.2; no re-plan
       resolve_request(overrides, table-built policy, cluster) fails → fail with its error // fidelity_conflict,
                                                                                         // at once, no wait
       otherwise → ensure_instance_and_wait(model, table-built policy, need)             // §4.3
  8  admit       = kv_admission(instance, tokens, max_tokens)          // RFC-0001 §8.4 rule 5; queue or reject
  9  seqs        = for i in 0..req.n: SequenceOpen{ prompt, prefix, sampling, constraint, limits,
                                                    exactness: policy.sequence_knobs, priority,
                                                    prefix_scope: scope(auth, req) }   // §9.5
                   n > 1 forks from the shared prefix (KvStore::fork), so the prompt is prefilled once
 10  stream: for each TokenFrame entry from the tail (local ring or sequence channel):
         text  = detok.push(tokens)             // UTF-8-safe incremental detokenizer
         text  = stops.push(text)               // holds back a possible stop-string prefix
         evts  = parsers.push(text)             // reasoning and tool-call parsers (§6, §8)
         encoder.emit(evts)                     // shape-specific SSE or buffered JSON
         if a stop string matched: send SequenceControl::Cancel; discard later tokens
 11  finish: usage, fidelity report, timings; Release { persist } (RFC-0001 §6.2)
```

Steps 1–9 run once per request on the `ctl` runtime. Step 10 runs per `TokenFrame`, on the task that owns the client connection. Client disconnect at any point sends `Cancel` around the ring (RFC-0001 §6.2), and the tokens already in flight are discarded.

**Transcript.** Ingress keeps the prompt tokens and emitted tokens of every live sequence (RFC-0001 §6.1 step 5). They are what makes re-prefill recovery possible, and they are also the record behind stored Responses (§9.2).

**Tokenization happens once, at ingress.** Stop strings, reasoning markers and tool-call markers are detected on detokenized text at ingress. Token-level stop sets (EOS, the template's end-of-turn tokens) and grammar masks are enforced by the tail's sampler (RFC-0001 §3.2), which is where the next token is chosen.

**Penalty counting basis** (defined by RFC-0012 §8.1 and §8.6, stated here because the API fields map onto it):
- `presence_penalty` and `frequency_penalty` count **emitted tokens only**, reasoning tokens included, as the OpenAI fields specify.
- `repetition_penalty` (§3.4) counts **prompt plus emitted tokens**. The prompt is the full rendered prompt, including earlier turns restored from a session or prefix cache.
- The basis is the same for a fresh sequence, for each choice of `n > 1` (each choice keeps its own emitted counts over the shared prompt), for a resumed session, and for a sequence that recovery re-prefills or migrates. Ingress's transcript is what the tail rebuilds the counts from, so a recovered sequence draws from the same distribution it would have drawn from without the failure.

*Note 2026-10-27 (PROD-E2-T02, as built).* `syntropy_api::pipeline` holds the pipeline as a library layer below the HTTP server. A route runs stages 1 and 2, hands the `GenerateRequest` to `Ingress::start` (stages 3 to 9), and polls the `Generation` it returns with the shape's encoder as its `EventSink` (stages 10 and 11). The build fixes what §2.1 and §2.2 leave open:
- **Seams.** Stages 3, 5, 7 and 8, the detokenizer and the stop matcher sit behind the `Stages`, `Detokenize` and `StopMatch` traits, which their own tasks implement (PROD-E2-T08, PROD-E3-T10, PLC-E6-T19, PROD-E2-T03). Stage 4's overrides (PROD-E1-T10), stage 6's constraint (PROD-E3-T05) and §2.3's range check (PROD-E2-T20) come with their tasks.
- **M0 fields.** `GenerateRequest` and `GenEvent` hold the fields whose types exist at M0a. `tools`, `tool_choice`, `parallel_tool_calls`, `constraint`, `reasoning`, `overrides`, `ContentPart::Media`, `TextDelta::logprobs`, the `fidelity` reports, `Loading` and `Recovered` arrive with the tasks that build their types. `n > 1` fails with `Unsupported` until PROD-E2-T06.
- **`deadline`** is a duration from the request's arrival, not an `Instant`. The pipeline reads time only through its `Clock` (RFC-0060 §2.1 rule T1), and the value becomes `SequenceLimits::deadline` unchanged.
- **Ending.** Each sequence ends with one `Release`, and ingress drops its transcript then. A matched stop string sends `Cancel` first, and so does a client gone. The pipeline checks the client before every write, so the `Cancel` goes before the next write would have, and a failed write counts as a client gone. A dropped `Generation` that has not ended does the same. A tail-reported error (`FinishReason::Error`) becomes `GenEvent::Failed` with `ApiError::Sequence` and a `Release`, with no `Cancel`.
- **Rings.** A `Cancel` or `Release` that finds `local_sequences` full waits in order and is sent first on the next ring call; an `Open` that finds it full fails with `Capacity`. Nothing waits for room, so the owner of the ingress calls `flush` until no control waits. Entries for a sequence that is not live are dropped and counted.
- **Sequence IDs.** One `SequenceIdMinter` per node per boot mints every `SequenceId` (RFC-0001 §5.9), and every ingress on the node, one per runner, shares it. After `n = u32::MAX` it refuses with `SequenceIdsExhausted` and never wraps.

#### 2.3 Token IDs are range-checked at ingress

Token IDs reach unsafe device code: the sampler indexes per-sequence count tables in device memory by token ID (RFC-0012 §8.1), and `RowGather` indexes embedding rows (RFC-0040 §3.1). An ID at or above a table's size would be an out-of-bounds read or, through the penalty tables, an out-of-bounds write. So ingress validates every token ID a request can carry, in one place, before routing:
- prompt `tokens` on the native API, teacher tokens on the parity endpoint (§13.6), `logit_bias` keys, stop token IDs, and the token set of a compiled grammar;
- each ID must be below `vocab_limit = min(embedding rows, LM-head rows, sampler table size)`, which the model's manifest carries (RFC-0043 §5.2 checks at census that the tokenizer's largest ID and every special-token ID are below the vocabulary).

*Note (2026-10-27):* the list also covers a token-array `prompt` on legacy `/v1/completions` (§3.5) and token-array `input` on embeddings (§14), which take token IDs from the client the same way. The validator is one function; PROD-E2-T20 builds it, and each endpoint task tests it through its own route.

A request with an ID outside the range fails with 400 `token_id_out_of_range`, naming the field, the first offending position and `vocab_limit`. Nothing is clamped or dropped silently. The check costs one pass over the IDs and sits inside the pre-`SequenceOpen` budget (Performance). Debug kernels keep bounds assertions as a second line (RFC-0040); they are not the defence.

### 3. API shapes

Each subsection lists what the front end reads and what the encoder writes. A field not listed is handled by §3.6. Every ignored field is named in a `Syntropy-Warning` response header (`Syntropy-Warning: ignored=user,metadata`), so clients can discover gaps without the request failing.

#### 3.1 OpenAI Chat Completions (`POST /v1/chat/completions`)

| Request field | Handling |
|---|---|
| `model` | §4.1 |
| `messages` | roles `system`, `developer` (treated as `system`), `user`, `assistant` (with `tool_calls`, and `reasoning_content` accepted on input), `tool` (`tool_call_id`). Text content only |
| `tools`, `tool_choice`, `parallel_tool_calls` | §6; function tools only |
| `response_format` | `text`; `json_object`; `json_schema { name, schema, strict }` (§7) |
| `temperature`, `top_p`, `seed`, `stop` (string or up to 16 strings), `presence_penalty`, `frequency_penalty`, `logit_bias` | sampler (RFC-0012) |
| `max_completion_tokens`, `max_tokens` | output limit, reasoning tokens included; default: the rest of the context window |
| `n` | ≤ `api.max_n`; choices fork from one prefill |
| `logprobs`, `top_logprobs` (≤ 20) | from the model's distribution before temperature, top-p and other sampling transforms, documented as such |
| `stream`, `stream_options.include_usage` | §5 |
| `reasoning_effort` | §8 |
| `service_tier` | `flex` → `Background` priority; others → `Normal` (RFC-0054 §7) |
| `prompt_cache_key` | routing affinity hint for the prefix cache (§9.4) |
| extensions: `top_k`, `min_p`, `repetition_penalty`, `syntropy { fidelity, approximate, session, priority, deterministic, trace }` | §3.4 |

The response is the standard `chat.completion` object. Fields with Syntropy-specific values:
- `system_fingerprint` is `fp_` plus the first 10 hex digits of BLAKE3(build ID, plan epoch, model ID). It changes exactly when the serving configuration does, which is what the field is for.
- `usage.prompt_tokens_details.cached_tokens` counts prefix-cache or session hits; `usage.completion_tokens_details.reasoning_tokens` counts reasoning tokens.
- `message.reasoning_content` carries reasoning (§8).
- The `syntropy` extension object is described in §10.
- `finish_reason`: `stop` for end of turn and stop strings, `length`, `tool_calls`.

*Note 2026-10-23 (OQ-NET-35): `top_logprobs = 0`.* `logprobs: true` without `top_logprobs` asks for the chosen tokens' logprobs alone: `SamplingParams::logprobs = Some(0)` (RFC-0001 §5.8). So does every surface that maps to `Some(0)`, §3.5's legacy `logprobs: 0` included, and the rule here applies to all of them. An instance whose tail runs in the ingress's process carries it as is. RFC-0031 §6.5's `Token` entry has no form for it yet, so on an instance whose tail is on another node, ingress asks the tail for `Some(1)` and drops the alternative before encoding the response. The chosen token's logprob is the same value either way, so the response stays exact. Once DIST-E8-T18 gives the entry its form behind capability bit 16 (RFC-0031 §10), ingress asks for `Some(0)` only when every peer the entry crosses advertises the bit: the tail's channels to the ingress and, for the ring-return copy, which carries the logprobs too, to the head. Otherwise it keeps the rewrite.

#### 3.2 OpenAI Responses (`POST /v1/responses`)

The front end ships in two halves. **Stateless Responses in M1**: every request is served from its own `input`, and nothing is kept after the stream ends. This is what coding-agent clients that speak only Responses need to connect. **Stored Responses in M3**: `store`, `previous_response_id`, `background` and the retrieval routes, with records on the ingress node only (§9.2).

| Request field | Handling |
|---|---|
| `input` | a string, or items: `message` (roles `user`, `assistant`, `system`, `developer`; `input_text` and `output_text` parts), `function_call`, `function_call_output`, `reasoning` |
| `instructions` | prepended as a system message; not inherited through `previous_response_id`, matching the published semantics |
| `previous_response_id` | §9.2: continues a stored response, reusing its session KV (M3). Before M3 it fails with `unsupported_feature` |
| `store` | from M3, default `api.responses.store` (true): a record on the ingress node (§9.2); `false` keeps nothing after the stream ends. Before M3 every response is stateless, and `store: true` (or an absent `store`) is served without a record and named in `Syntropy-Warning` |
| `tools`, `tool_choice`, `parallel_tool_calls` | function tools only; built-in tool types are rejected with `unsupported_tool` |
| `text.format` | `text`, `json_object`, `json_schema` (§7) |
| `reasoning.effort` | §8; `reasoning.summary` is ignored (Syntropy returns the model's own reasoning text, not a summary) |
| `max_output_tokens`, `temperature`, `top_p`, `metadata`, `service_tier`, `prompt_cache_key` | as for chat; `metadata` is stored with the response |
| `background` | runs detached; the client polls `GET /v1/responses/{id}` or cancels with `POST …/cancel` (M3; before M3, `unsupported_feature`) |
| `include` | ignored, named in `Syntropy-Warning`; reasoning items the client sends back in `input` are rendered through the template like any other |
| `truncation` | `disabled` only; `auto` is accepted, treated as `disabled`, and named in `Syntropy-Warning` |
| `stream` | §5 |

Output items, in order: a `reasoning` item whose `content` holds `reasoning_text` parts (and an empty `summary`), then `message` items with `output_text` parts, and `function_call` items (`call_id`, `name`, `arguments`). `status` is `completed`, `incomplete` (with `incomplete_details.reason = "max_output_tokens"`), `failed`, `cancelled`, `queued` or `in_progress`. `usage` reports `input_tokens`, `input_tokens_details.cached_tokens`, `output_tokens`, `output_tokens_details.reasoning_tokens` and `total_tokens`.

#### 3.3 Anthropic Messages (`POST /v1/messages`, `POST /v1/messages/count_tokens`)

| Request field | Handling |
|---|---|
| headers `anthropic-version`, `anthropic-beta` | accepted; `anthropic-version` selects Anthropic shapes on shared paths (§3.4); unknown betas are ignored and named in `Syntropy-Warning` |
| `model`, `system` (string or text blocks), `messages` | content blocks `text`, `tool_use`, `tool_result` (with `is_error`), `thinking` (signature accepted, see §8); `redacted_thinking` is dropped; `image` and `document` are rejected |
| `max_tokens` | required, as in the published API; thinking tokens count against it |
| `stop_sequences`, `temperature`, `top_p`, `top_k` | sampler |
| `tools` (`name`, `description`, `input_schema`, `strict`), `tool_choice` (`auto`, `any`, `tool`, `none`, `disable_parallel_tool_use`) | §6 |
| `thinking` (`{type: "enabled", budget_tokens}` or `{type: "disabled"}`) | §8 |
| `output_config.format` (`json_schema`); the older `output_format` is accepted as an alias | §7 |
| `cache_control` on blocks | prefix-pin hint (§9.4) |
| `metadata.user_id`, `service_tier` | ignored |

Responses are `message` objects with `content` blocks `thinking` (`thinking`, `signature`), `text` and `tool_use` (`id` = `toolu_` + 24 base62 characters, `name`, `input`). `stop_reason` is `end_turn`, `max_tokens`, `stop_sequence` (with `stop_sequence` set) or `tool_use`. `usage` carries `input_tokens`, `output_tokens`, `cache_read_input_tokens` (prefix-cache hits) and `cache_creation_input_tokens` (tokens newly written to a pinned prefix, §9.4). `count_tokens` renders the template with the given tools and returns `{ "input_tokens": n }`.

#### 3.4 Shared paths, extension fields and headers

- **`GET /v1/models` serves two shapes.** A request carrying `anthropic-version` gets the Anthropic list shape (`data[]` of `{type: "model", id, display_name, created_at}` with `has_more`, `first_id`, `last_id`); any other request gets the OpenAI shape (`object: "list"`, `data[]` of `{id, object: "model", created, owned_by: "syntropy"}`). Both add a `syntropy` object per model: `state` (`ready`, `loading`, `available`, `remote-only`), `context_length`, `fidelity`, `instances`, and `modalities`: `{ "input": ["text"], "source": [...] }`, where `source` lists what the publisher's model accepts. When `source` holds more than text, `text_only: true` is set and the native and dashboard model lists show "text only in Syntropy".
- **Extension object.** Requests may carry a top-level `syntropy` object (SDKs pass it through `extra_body`). `approximate` names the knobs the request is willing to have applied. The effective set is its intersection with the instance policy's knobs, and their parameters come from the policy (RFC-0054 §7):

```json
"syntropy": {
  "fidelity": "approximate",
  "approximate": ["topk_reduce", "expert_skip"],
  "session": "ses_6f2k…",
  "priority": "background",
  "deterministic": false,
  "speculative": "off",
  "trace": true
}
```

- **Header equivalents** for clients that cannot add body fields: `Syntropy-Fidelity`, `Syntropy-Approximate` (comma-separated knob names; parameters always come from the instance policy), `Syntropy-Session`, `Syntropy-Priority`, `Syntropy-Trace`. The W3C `traceparent` header is honoured (RFC-0053 §3.2). Where a body field and a header disagree, the request fails with `InvalidRequest` rather than picking one.
- **Unknown fields** in the `syntropy` object are errors (they are ours, so a typo should fail). Unknown fields elsewhere follow §3.6.

#### 3.5 Legacy completions (`POST /v1/completions`)

`prompt` (string, array of strings, token array), `suffix` (fill-in-the-middle, rendered with the FIM tokens declared in the tokenizer metadata; a model without them fails with `unsupported_feature`), `max_tokens` (default 16, as published), `echo`, `logprobs` (integer, ≤ 5 as published, ≤ 20 accepted), `stop`, `temperature`, `top_p`, `n`, `seed`, `stream`. `best_of > 1` is rejected. No chat template is applied. This surface exists for code-completion clients, which send raw FIM prompts.

#### 3.6 What is not supported

| Item | Behaviour |
|---|---|
| Endpoints for audio, images, files, uploads, batches, fine-tuning, assistants, threads, vector stores, moderations, realtime | 404 in the request's shape, code `unsupported_endpoint` |
| Image, audio, video, file or document content parts | 400 `unsupported_content_type`. They parse into the reserved `ContentPart::Media` (§2.1), so the error names the part and the model's `modalities`, and a later multimodal family turns them on without a new request shape |
| Control-token strings inside content (`<\|im_start\|>`, a tool-call begin marker, a channel header) | tokenized as the literal text they are (§6.4); the native API's `raw_special_tokens` is the only opt-in |
| `modalities` other than `["text"]`, `audio` output | 400 `unsupported_feature` |
| Built-in tools (`web_search`, `file_search`, `code_interpreter`, `computer_use`, MCP tool types) | 400 `unsupported_tool` |
| `user`, `metadata` (chat), `store` (chat), `safety_identifier`, `include`, `reasoning.summary` | ignored, named in `Syntropy-Warning` |
| `prediction` (predicted outputs) | ignored in v1, named in `Syntropy-Warning`. It is a natural draft source for exact speculative decoding and is listed for RFC-0012 after v1 |
| `truncation: "auto"` | treated as `disabled`, named in `Syntropy-Warning` |
| `n > 1` with tools or constraints | supported; every choice gets its own constraint state |
| An unknown **value** of a known field, such as a `thinking.type` other than `enabled` or `disabled`, or a `tool_choice` or tool `type` not listed above | 400 `unsupported_feature`, naming the field and the value, unless a mapping is listed here. Listed mapping: an Anthropic adaptive thinking mode → the family's default reasoning budget. The SDK conformance suite includes such cases |

### 4. Model references, routing by name and automatic placement

#### 4.1 Model references

```text
model-ref   := alias
             | "id:" hex{8,64}                       // ModelId prefix
             | [ "hf:" ] owner "/" repo [ ":" variant ] [ "@" revision ]
             | repo [ ":" variant ]                  // short form, if unique in the cluster
variant     := a GGUF quantization tag (Q4_K_M, IQ3_XXS, …) | a file name | "native"
revision    := a Hub branch, tag or commit
```

Resolution, first match wins:

1. Node `aliases`, then `cluster_aliases` (RFC-0054 §5.2, §5.3).
2. `id:` prefix against registered `ModelId`s. An ambiguous prefix fails with the candidates listed.
3. Exact canonical name (`owner/repo:variant@revision`) among registered models (`ModelRegistered` in cluster state).
4. Short form, if exactly one registered model matches.
5. A Hub reference that is not registered: with `api.auto_pull = false` (default), fail `model_not_found` with the hint `syntropy pull <ref>`. With `auto_pull = true`, register and fetch it, subject to the licence rule of ADR-0018 decision 7 and RFC-0051 §3.2. It fails with `licence_acceptance_required`, naming the step, when the model is gated and this node's Hub token has not accepted its terms on the Hub, or when its licence is custom (use conditions beyond an Apache-2.0/MIT-class licence) and nobody has accepted it on this node with `syntropy pull --accept-licence` or the dashboard's "I accept". Syntropy never accepts terms on the user's behalf, and the API never automates a Hub acceptance.

**Default variant.** When a reference names a multi-variant repository without a variant, the choice is deterministic: a variant that already has a routable instance; else one already in some node's store; else the variant the model picker recommends for this cluster (RFC-0051 §3.3). The response's `model` field echoes the string the client sent, because clients compare it, and `syntropy.model` carries the canonical name and `ModelId`.

Automatic downloads are off by default because a mistyped model name must not start a 400 GB download from an API call.

#### 4.2 Routing

Routing is `solver::route` from RFC-0001 §6.1 step 2, run locally on the ingress node's replica of cluster state. The API layer adds four filters before it scores:
- **fidelity**: each candidate instance is checked with `resolve_request` against **its own** policy. An exact request may use an instance whose plan has no per-instance knob, even if the policy allows per-sequence knobs, which are simply not applied. It never uses an instance whose plan carries a per-instance knob. The policy built from the `policy.*` tables is used only for `EnsureInstance`, when no candidate qualifies. The request is resolved against that policy first, and a resolution error (`fidelity_conflict`) is returned at once. It never waits for a load that could not serve it;
- **context**: the candidate's planned `context_length` (published with the committed plan, `PredictedCost::context_length`) covers `prompt_tokens + max_tokens`. The solver already plans the largest context the memory allows for `max_sequences`, so a per-request re-plan cannot help. If routable instances pass every other filter but none covers the request, it fails at once with 400 `context_length_exceeded`. The error names the largest planned limit, the model's maximum, and the keys that raise the limit at the next plan (`context_length`, a lower `max_sequences`). This is the one outcome, shared with RFC-0054 §4.5;
- **determinism**: `deterministic = true` only reaches instances whose policy has it;
- **session affinity**: a pinned session's holders win ties, and a `prompt_cache_key` or `Syntropy-Session` hashes to a preferred instance among equals (§9.4);
- **state**: only `InstanceRoutable` instances, and never an instance whose ring includes a node currently `Suspected`, unless it is the only one. A node that announced a planned restart (`Restarting`, RFC-0030 §5.9) has runners in orphan mode that admit nothing (RFC-0052 §6.5). So instances on it are skipped while another instance qualifies. Otherwise the request waits for that node's `NodeAlive`, bounded by its `back_within_ms` and by `api.load_wait`, then routes or fails with retryable `instance_unavailable`.

If an instance-scoped error arrives before the first token, ingress re-routes to another routable instance once (RFC-0001 §10.4). After the first token, recovery belongs to RFC-0001 §7.5, and the client sees a `Recovered` event (§5.4) rather than an error.

#### 4.3 Automatic placement on first request

```text
ensure_instance_and_wait(model, policy, need, deadline):
    key = (model.id, policy.digest())
    if a wait for key already exists on this node: join it                   // coalesce
    evictable = idle instances (no live sequences, preload = false), least recently used first
    preview = solver::preview(PreviewInput { census: Full(manifest), policy, evictable,
                                             candidates: 1, effort: Full, .. })   // RFC-0051 §3.1
    match preview.verdict:
        DoesNotFit(report) => fail InvalidRequest code=model_does_not_fit with report   // 400; not retryable
        _ => {}
    if preview.best.predicted.context_length < need:                          // the largest context this
        fail InvalidRequest code=context_length_exceeded                      // placement can hold
    if preview.best.evicts is not empty: submit UnloadInstance for each       // §Proposed amendments A2
    eta = preview.best.time_to_ready                                         // fetch + pack + load + warm
    submit EnsureInstance { model, policy }                                  // idempotent
    if eta > api.load_wait: fail Capacity code=model_loading, Retry-After=eta   // load continues
    loop until InstanceRoutable{instance with this policy digest} or deadline or api.load_wait:
        poll LoadProgress over the node-query channel from each stage node, 1 Hz
        emit Loading { phase, fraction, eta } where the response may still carry progress (below)
    on timeout: fail Capacity code=model_loading, Retry-After = remaining eta
```

- **Coalescing.** Concurrent requests for one (model, policy) share one wait and one `EnsureInstance`, which is idempotent anyway.
- **Progress and when headers are committed**. An HTTP response's status is fixed by its first byte. So:
  - **OpenAI and Anthropic shapes**, by default, hold the response, writing no bytes, until the first token or an error, for at most `api.load_wait`. Every placement failure therefore still gets its §12.1 status and `Retry-After`.
  - **A client that opts in** with `Syntropy-Progress: 1` (or `syntropy.progress = true`) receives `Loading` as SSE comments (`: loading 43% packing on pc`), which spec-compliant parsers ignore, plus a comment every `api.sse_keepalive`. From the first comment on, errors take the §12.2 in-stream form, which carries `code` and `retry_after` in its body.
  - **The native API** always streams `event: loading`, and its in-stream error carries the full `Error` with its retry hint.
  - **Non-streaming requests** are held silently.
  - Clients with short read timeouts should opt into progress, stream through the native API, or preload the model (`preload = true`, or `syntropy run`).
- **Where progress comes from.** `RunnerStateChanged` in the event log carries only state. Byte-level progress (fetch bytes and sources, pack and load fractions) is served by each stage node as `LoadProgress { instance, node, phase, done, total, sources }` over the node-query channel (amendment A4), polled by the waiting ingress at 1 Hz while a load is in flight. The event log never carries per-chunk progress (ADR-0009).
- **Does not fit** means the preview found no feasible plan even after evicting every idle instance. The error body carries the fit report: per node and tier, bytes needed against bytes usable, and the binding constraint, plus smaller sibling variants that would fit (`ModelManifest::siblings`), each with its predicted tokens/s. This is a 400, not a 503, because retrying cannot help until the hardware or the request changes.
- **Busy, not unfit.** If the model would fit but live sequences hold the capacity, ingress queues the request (bounded by `api.queue.max_wait`) and then fails with `Capacity` and `Retry-After`.
- **Keep-alive.** An instance with no sequences for its `keep_alive` is unloaded with `UnloadInstance`, by the coordinator, so one decision is made per cluster.
- **Model list.** `GET /v1/models` lists every registered model that some node holds in its store (`available`), every routable one (`ready`), and every one being placed (`loading`), so a client's model picker shows what a request can reach.

### 5. Streaming

#### 5.1 Wire formats

All streams are Server-Sent Events (`Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no` so reverse proxies do not buffer).

| Shape | Events |
|---|---|
| OpenAI chat | `data: {chat.completion.chunk}` per delta. The first chunk has `delta.role`; each choice's last chunk has `finish_reason`; with `include_usage`, a final chunk with empty `choices` and `usage`; then `data: [DONE]` |
| OpenAI legacy completions | `data: {text_completion}` chunks, then `data: [DONE]` |
| OpenAI Responses | named events, each with a `sequence_number`: `response.created`, `response.in_progress`, `response.output_item.added`, `response.reasoning_text.delta` / `.done`, `response.content_part.added`, `response.output_text.delta` / `.done`, `response.content_part.done`, `response.function_call_arguments.delta` / `.done`, `response.output_item.done`, then exactly one of `response.completed`, `response.incomplete`, `response.failed` |
| Anthropic | `message_start`, then per block `content_block_start` → `content_block_delta` (`thinking_delta`, `signature_delta`, `text_delta`, `input_json_delta`) → `content_block_stop`, then `message_delta` (`stop_reason`, `stop_sequence`, `usage.output_tokens`), `message_stop` |
| Native | `event: started`, `loading`, `reasoning`, `text`, `tool_call`, `tool_call_delta`, `recovered`, `done`, `error`, each with an SSE `id: <sequence>:<n>` counting events, so a client can re-attach with `Last-Event-ID` (§5.5); `?format=ndjson` gives one JSON object per line instead, with the same `n` in an `id` field |

**Keep-alive.** Every `api.sse_keepalive` (15 s) without an event, the encoder writes a comment line (`: keepalive`) in the OpenAI and native shapes and a `ping` event in the Anthropic shape, which is what Anthropic clients expect.

**Ordering for `n > 1`.** Choices interleave in arrival order; every chunk names its `index`. Each choice's `finish_reason` arrives on its own last chunk. The `[DONE]` sentinel follows the last choice.

#### 5.2 From tokens to text

- **Incremental detokenization.** Ingress keeps two offsets per choice into the emitted token list. Each new batch is decoded over the window `tokens[prefix..]`, the previously released text for that window is subtracted, and the remainder is released only if it does not end in an incomplete UTF-8 sequence. Byte-fallback tokenizers that split one character across tokens therefore never emit a replacement character mid-stream. The window start advances on each clean boundary, so the cost per token stays constant.
- **Stop strings** can span tokens. The stop matcher holds back the longest suffix of released text that is a proper prefix of some stop string (at most `max_stop_len − 1` bytes). On a match it releases the text before the stop string, finishes the choice with `StopSequence`, and sends `Cancel`. Tokens the tail produced after the matched one, at most one ring traversal's worth (RFC-0001 §6.2), are discarded and not counted in usage.
- **Parsers** for reasoning (§8) and tool calls (§6) run after the stop matcher, on released text, with their own hold-back for marker prefixes.

#### 5.3 Back-pressure and disconnects

- Each choice buffers at most `api.max_buffered_tokens` (256) undelivered tokens. Above that, ingress sends `Pause` to the tail and `Resume` when the buffer drains below half (RFC-0001 §8.4 rule 3). The ring never waits for a client.
- A closed client connection is detected on the next write or by the socket's error event, and ingress sends `Cancel`. On HTTP/2, a `RST_STREAM` cancels the same way. The exception is a stream a client can re-attach to (§5.5), which is paused for the detach grace first.
- **Coalescing.** The native API accepts `stream_coalesce_ms` (0–200, default 0), which batches deltas for clients on slow links. The OpenAI and Anthropic shapes never coalesce: one event per detokenized delta.

#### 5.4 Recovery and errors mid-stream

- When a stage is lost and the sequence resumes through migration, replay or re-prefill (RFC-0001 §7.5), the stream continues. Native clients get `event: recovered` with the kind and the gap; the other shapes get nothing in-band, and the final `syntropy` object lists the recovery (§10). Emitted tokens are never retracted or repeated.
- An unrecoverable error after the first token ends the stream with the shape's in-stream error (§12.2).

#### 5.5 Detach grace and re-attach (M3)

The ingress is almost always the client's own machine (loopback is the default bind, §11.1), so when the ingress dies the client usually dies with it, and a replicated ingress transcript would rescue a case that rarely has a surviving client. v1 has no resumable-stream API across ingress loss. The common failure is different: a dropped client link, such as a laptop's Wi-Fi blip, with both ends alive. For that, ingress holds the sequence instead of cancelling it.
- **Which streams.** Native streams, and stored Responses streams (`store: true`). Chat Completions, Messages and stateless Responses have no way to re-attach, so a disconnect there cancels at once, as in §5.3.
- **Holding.** On disconnect ingress sends `Pause` to the tail (the same control as back-pressure, RFC-0001 §8.4 rule 3) and keeps the sequence, its KV and the undelivered events for `api.detach_grace` (60 s [target]). An ingress-only node whose channel to the tail stays down past `link.failover_timeout` gets the same window at the tail (RFC-0033 §9.2, DIST).
- **Re-attaching.** A native client reconnects to `GET /v1/syntropy/generate/{sequence}/events` with `Last-Event-ID`. A Responses client calls `GET /v1/responses/{id}?stream=true&starting_after=<sequence_number>`. Ingress replays the events after that point from its buffer, sends `Resume`, and the stream continues with no repeated or missing text. Only the principal that opened the sequence may re-attach (§9.5).
- **Expiry.** When the window ends without a re-attach, ingress sends `Cancel`. A session-bound sequence keeps its KV under `kv.autosave` (RFC-0022), so a later request on the session prefills only the new suffix.
- **Cost.** A paused sequence holds its KV pages and admission slot for at most the window; RFC-0001 §8.4's KV admission counts it as live.

### 6. Tool calling

#### 6.1 Into the prompt

```rust
pub struct ToolSpec {
    pub name: String,                    // [a-zA-Z0-9_-]{1,64}
    pub description: Option<String>,
    pub parameters: serde_json::Value,   // JSON Schema (OpenAI `parameters`, Anthropic `input_schema`)
    pub strict: bool,                    // arguments must validate: enforced by a constraint (§7)
}
pub struct ToolCall { pub id: String, pub name: String, pub arguments: String /* JSON text */ }
```

Tools reach the model through its chat template, which receives `tools` in the conventional template variable shape (a list of `{type: "function", function: {name, description, parameters}}`). Templates are Jinja, rendered with `minijinja` plus the helpers model templates use (`tojson`, `raise_exception`, `strftime_now`). A per-model `chat_template` override exists (RFC-0054 §5.3) for models whose shipped template is broken. Tool results from either shape normalise to `ChatMessage::ToolResult` and render through the same template. Rendering returns spans, not one string: text the template itself writes is a template span, and every value that came from the request (message text, tool results, tool descriptions and schemas, documents) is a content span (§6.4).

#### 6.2 Out of the text

Each architecture family module registers a tool-call format with `syntropy-text` (RFC-0043 owns the per-family choice; this RFC owns the parser framework). The registration also says, per marker, whether it is a **special token** in that family's tokenizer or **ordinary text**. The formats needed for the v1 families are all variants of these patterns:

| Pattern | Shape in the text | Used by (v1 families, ADR-0003) |
|---|---|---|
| Tagged JSON | `<tool_call>{"name": …, "arguments": …}</tool_call>` | Qwen dense and MoE, several Llama/Mistral fine-tunes |
| Special-token sections | a begin marker, then one entry per call, each `name` + separator + JSON arguments, then an end marker | DeepSeek-V3-class, Kimi-class |
| Tagged key/value arguments | `<tool_call>name` then `<arg_key>k</arg_key><arg_value>v</arg_value>` pairs, or `<function=name><parameter=k>v</parameter></function>` | GLM-class; Qwen coder variants |
| Channel-addressed messages | a message header naming a recipient (`to=functions.name`) and a content type, followed by JSON | gpt-oss-class |
| Bare JSON list | a leading marker token, then `[{"name": …, "arguments": …}]` | Mistral-class |

**Markers are matched where the distinction lives.** Each family's `ChatFormat` resolves every marker to `Marker::Special(id)` or `Marker::Text` (RFC-0043 §8.4, §8.7). Where a format's markers are special tokens (the special-token sections, the channel headers, the leading marker token, and tagged formats whose tags the tokenizer defines as special), the parser detects start and end markers by **token ID** on the `TokenFrame` entries, never by matching text. A model that quotes a document containing `<tool_call>{…}` as text therefore produces text, not a call, because content was tokenized literally (§6.4). Only formats whose markers are ordinary text in the tokenizer fall back to text matching. A response parsed that way carries `tool_markers_text` in `syntropy.warnings`, and the model's native detail (§13.2) reports `tool_markers: "text"`, so agent clients can tell.

The parser is an incremental state machine per choice:

```text
state Text:        release text until a possible start marker; hold back a marker prefix
                   on a full start marker → InCall (emit ToolCallStart once the name is known)
state InCall:      route bytes to the format's argument reader;
                   JSON-argument formats stream the raw JSON fragments as ToolCallArgsDelta;
                   key/value formats convert each completed pair to JSON and stream that
                   on the end marker → validate arguments as JSON → ToolCallEnd → Text (parallel calls)
end of turn:       finish with StopReason::ToolCalls if any call completed
```

- **IDs.** `call_` + 24 base62 characters in OpenAI shapes, and `toolu_` + 24 base62 characters in the Anthropic shape. When a model emits its own call IDs (some special-token formats do), the API ID is a reversible encoding of the model's ID: the prefix plus base62 of the model ID's bytes, padded to the shape's usual length. Tool results that come back with the API ID therefore decode to the model's ID on any node, after any restart, with no stored mapping.
- **Malformed calls.** Without a constraint, a model can emit a call whose arguments are not valid JSON. The parser then returns the raw text as ordinary content, sets `finish_reason` as if no call was made, and adds `tool_call_unparsed` to `syntropy.warnings`. It does not invent a call. With `strict: true`, or `tool_choice` set to `required`, `any` or a named tool, the constraint of §7 makes a malformed call impossible.
- **`parallel_tool_calls = false`** (`disable_parallel_tool_use = true`) ends generation after the first completed call. When a constraint is active, the grammar admits one call only.

#### 6.3 Forced and strict tool calls

`tool_choice = required`, a named tool, `any` (Anthropic) or any tool with `strict: true` compile to a constraint (§7): the grammar for "the family's tool-call syntax, with arguments matching that tool's schema". With `auto` plus strict tools, the grammar is "free text, or a tool call whose arguments match the schema of the named tool". The start marker is what switches the automaton into the schema-constrained branch, so free text stays unconstrained.

#### 6.4 Control tokens in content

Templates render roles, turn boundaries and tool-call scaffolding with special tokens. If the rendered prompt were tokenized as one string, any user message, tool result, retrieved web page or file containing a literal control string (`<|im_start|>system`, a tool-call begin marker, a channel header) would become a real control token. That forges roles, system turns and tool results, and agent tool results are attacker-controlled content by design. So:
- **Spans.** `text::render` returns the prompt as a sequence of spans, each marked `Template` (written by the template) or `Content` (a value from the request). The template engine marks every interpolated request value as content; the family's helpers cannot unmark it.
- **Tokenizing.** Template spans are tokenized with special-token matching on. Content spans are tokenized with special-token matching **off**, so a control string in content becomes the ordinary tokens of its text, byte for byte. Nothing is stripped or rejected: the model sees exactly what the user sent. The tokenizer's span-aware encoder provides both modes in one call (RFC-0043 §8.7).
- **The one opt-in.** The native API's `raw_special_tokens: true` (§13.1) tokenizes content spans with special-token matching on, for clients that build prompts with control tokens on purpose. No vendor shape has it, and native `tokens` input is already raw token IDs, subject only to the range check (§2.3).
- **Output.** Tool-call and reasoning markers that are special tokens are parsed on token IDs (§6.2, §8.1), so text in the model's output cannot open a call either.
- **Fuzzing.** The parser fuzz targets (Testing) inject every v1 family's control strings into user, system, tool-result and tool-description content, and check that the token stream holds no special-token ID inside any content span and that no injected marker yields a parsed call or a role change.

### 7. Structured output and grammar-constrained decoding

#### 7.1 Constraint kinds

```rust
pub enum OutputConstraint {
    JsonObject,                                   // any JSON object
    JsonSchema { schema: serde_json::Value, strict: bool },
    Regex(String),                                // native only
    Grammar { syntax: GrammarSyntax, text: String }, // native only; Lark-style EBNF
    Choice(Vec<String>),                          // native only; one of these strings
    ToolCall { tools: Vec<ToolSpec>, mode: ToolConstraintMode }, // §6.3; built internally
}
```

#### 7.2 Engine

The compiler (ADR-0504) turns a constraint into a token-level automaton:

1. **Lowering.** JSON Schema becomes a context-free grammar over bytes. Regex and `Choice` become lexer rules. A tool-call constraint becomes the family's tool-call syntax with each tool's argument schema spliced in. Reasoning models get the reasoning section left free, with the constraint applied after the end-of-reasoning marker (§8), so a structured answer does not suppress the model's reasoning.
2. **Automaton.** An Earley-style parser over a lexer built from regular-expression derivatives handles any context-free grammar, including recursive schemas.
3. **Token masks.** At each step the allowed set of next tokens is computed by walking a trie of the vocabulary's byte strings against the parser state. Lexer states whose answer does not depend on parser context are cached per (grammar, tokenizer).

The engine is the `llguidance` library (MIT, Rust), behind a `ConstraintEngine` trait in `syntropy-text` so that it can be replaced (ADR-0504).

**JSON Schema coverage.** Supported: `type` (including unions), `properties`, `required`, `additionalProperties` (false or a schema), `items`, `prefixItems`, `minItems`/`maxItems`, `enum`, `const`, `anyOf`, `oneOf` (treated as `anyOf`), `allOf` (merged where possible), `$ref` and `$defs` (recursion allowed), `pattern`, `format` (`date-time`, `date`, `time`, `uuid`, `email`, `ipv4`, `ipv6`), `minLength`/`maxLength`, `minimum`/`maximum` for integers. Unsupported keywords under `strict: true` fail the request with `constraint_unsupported`, naming the keyword. Without `strict`, they are ignored and named in `Syntropy-Warning`. Property order follows the schema's `properties` order, which is what model training data most often shows.

*Note 2026-10-22 (PROD-E3-T04):* how the coverage above is applied.
- **`$ref`.** It is supported as a document-local reference, `#` or `#/...`. The engine is built without its remote-reference support (`llguidance`'s `referencing` feature pulls `borrow-or-share`, whose MIT-0 licence is outside ADR-0018's allow list), so any other `$ref` counts as an unsupported keyword. Under `strict` it fails with `constraint_unsupported`, naming `$ref`; otherwise it is ignored and named in `Syntropy-Warning`.
- **`definitions`.** The draft-07 name for `$defs` counts as `$defs`.
- **Annotations.** Annotations (`title`, `description`, `default`, `examples`, `$comment`, `$schema`, `$id`, `readOnly`, `writeOnly`, `deprecated`) constrain nothing and are accepted. Vendor extensions (`x-...`) are ignored without a warning, except the engine's own option key, `x-guidance`, which is unsupported, so a request cannot change how its schema compiles.
- **Format names.** A `format` outside the list above is named in the error or warning as `format: <name>`.
- **Separators.** Constrained JSON uses `", "` between items and `": "` after keys, with no other whitespace between tokens, so a model cannot loop on whitespace inside a structure.
- **`oneOf`.** "Treated as `anyOf`" holds only where nothing is lost: when the engine can prove the options exclusive, as when they differ in type or constant, no output matches two of them. Where it cannot, an output may match two options, which `oneOf` forbids. Under `strict` such a `oneOf` fails with `constraint_unsupported`, naming `oneOf` and its location. Without `strict` it compiles as `anyOf` and is named in `Syntropy-Warning`, so the approximation is always labelled (D3).
- **Tokens that start with byte `0xFF`.** No UTF-8 text holds that byte, so no constrained output contains such a text token, and the engine never allows one. The rest of the vocabulary stays constrained.
- **Escapes.** Constrained JSON strings use no `\uXXXX` escapes. Each character is written as itself or with its short escape (`\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`), so each string has one spelling. Because of that, an additional property's name, once decoded, can never equal a declared name and leave that property's value unchecked. The cost is that a constrained string cannot hold any other control character.

#### 7.3 Where masks run

- **Compile at ingress**, cached by BLAKE3(constraint, tokenizer). The compiled grammar travels to the tail by reference: `SamplingParams` carries a `GrammarRef` (the hash), and the first `SequenceOpen` for a new hash carries the serialised grammar inline. The tail caches it by hash (proposed amendment A3). Ingress never ships the same grammar twice to one tail runner.
- **Apply at the tail.** The sampler holds the automaton state per sequence. After sampling token t, it advances the state and immediately computes the mask for position t+1, on a CPU core of the tail, while the ring runs the next step's forward pass. The mask is ready before the logits for t+1 arrive unless mask computation takes longer than a whole ring traversal. The mask is a bitset over the vocabulary, applied to the logits before every sampling transform.
- **Speculative windows.** With drafting at the tail (MTP or draft heads, RFC-0001 §6.4), the drafter samples under the same automaton, so drafts are always grammatical. Verification computes masks along the drafted path, position by position, and acceptance uses the masked target distribution. This keeps acceptance exact with respect to the constrained sampler.
- **Forced completion.** When the automaton has exactly one legal continuation for several tokens (for example the rest of a fixed key name), the tail can emit those tokens without sampling. They are still computed through the model in the next window, so KV stays correct. This is an optimisation the scheduler may use (RFC-0012); it does not change output.

#### 7.4 Exactness

A constraint is part of the sampler the request asked for: the reference forward pass applies the same mask to the same logits. So constrained decoding is compatible with exact mode and carries no approximate label. The same holds for logit bias, penalties and reasoning budgets (§8.3).

### 8. Reasoning content

#### 8.1 Parsing

Reasoning models write their reasoning in-band. Each family module declares its reasoning syntax (RFC-0043): a pair of markers (`<think>` … `</think>` and its variants), or a channel name in channel-addressed formats (the `analysis` channel in gpt-oss-class models). Some templates open the reasoning section themselves, so the model's output starts inside it; the family declares that too, and the parser starts in the reasoning state.

The reasoning parser runs before the tool-call parser. Text inside the section becomes `ReasoningDelta`; text after it becomes `TextDelta` and feeds the tool-call parser. Markers are never released as text. Where a family's reasoning markers are special tokens, the parser detects them by token ID, as §6.2 does for tool calls, so literal marker text in content or output never opens or closes a reasoning section.

#### 8.2 Output per shape

| Shape | Where reasoning goes |
|---|---|
| OpenAI chat | `message.reasoning_content` / `delta.reasoning_content`, a widely used extension field. `api.openai.reasoning_field` can rename it to `reasoning`, put it inline in `content` between the model's own markers (`inline`), or drop it (`omit`) for clients that choke on unknown fields |
| OpenAI Responses | a `reasoning` output item with `reasoning_text` content parts, streamed with `response.reasoning_text.delta` |
| Anthropic | `thinking` content blocks. Each block ends with a `signature`: base64 of a BLAKE3 keyed hash of the thinking text under a node-local key, so the field is present for clients that require it. Signatures are not checked on input, because thinking returned by a client is only rendered into the prompt by the template |
| Native | `event: reasoning`; `reasoning` field in the final object |

`usage` counts reasoning tokens in the output totals and reports them separately (`reasoning_tokens`; for the Anthropic shape they are part of `output_tokens`, as published).

#### 8.3 Controls

| Input | Mapping |
|---|---|
| OpenAI `reasoning_effort` / Responses `reasoning.effort` (`minimal`, `low`, `medium`, `high`) | the family's own control where it has one (a system-prompt level line, or a template flag); otherwise a reasoning budget of 0, 1024, 4096 or unlimited tokens respectively |
| Anthropic `thinking: {type: "enabled", budget_tokens}` | the family's "thinking on" template flag plus a budget of `budget_tokens` |
| Anthropic `thinking: {type: "disabled"}`, or effort `minimal` on a family with an off switch | the family's "thinking off" flag; families with no off switch get budget 0 |
| none | `reasoning.default` for the model (RFC-0054 §5.3) |

**Budget enforcement.** When a sequence's reasoning tokens reach its budget, the tail's sampler forces the end-of-reasoning marker through the constraint mechanism of §7 (a mask that allows only the marker's tokens), and generation continues with the answer. The client asked for the budget, so this is part of the requested sampling, not an approximate knob.

**Multi-turn.** Reasoning carried in earlier assistant turns is passed to the template, which decides whether to keep it (most templates for these families drop it from history). Syntropy does not strip or keep it on its own.

### 9. Sessions, prefix reuse and stored responses

RFC-0022 owns KV, the prefix cache and session persistence. This section defines how the API exposes them.

#### 9.1 Stateless shapes get prefix reuse for free

OpenAI Chat Completions and Anthropic Messages resend the whole conversation each turn. The head's prefix-cache lookup (RFC-0001 §6.2) finds the longest cached prefix in the caller's scope (§9.5), so only the new suffix is prefilled (SC-20). No client change is needed. Usage reports the reused tokens as `cached_tokens` or `cache_read_input_tokens`.

Two optional extensions make reuse durable and routable:
- `Syntropy-Session: <id>` (or `syntropy.session`) binds the request to a session. The session's KV can then be persisted when the sequence ends (`kv.autosave`, RFC-0022), survive restarts, and attract routing to its holders (`PinSession`, RFC-0001 §5.10). A request for an unknown session ID creates it. Whether its KV persists on release follows `kv.autosave` (RFC-0022), and the native API's `persist` field overrides it per session.
- `prompt_cache_key` (OpenAI) and Anthropic `cache_control` breakpoints are hints. `prompt_cache_key` is hashed into the routing tie-break, so requests sharing a key land on the instance that holds their prefix. A `cache_control` breakpoint asks the head to keep the prefix up to that block resident for its TTL (`ephemeral` 5 minutes, or the TTL given), and the tokens written that way are reported as `cache_creation_input_tokens`. A breakpoint also marks that prefix as shareable with other principals (§9.5), which is how a shared system prompt keeps caching across clients.

#### 9.2 Responses: `previous_response_id`

A stored response records `{ id, session: SessionId, model: ModelId, policy digest, transcript tokens up to and including this response, output items, metadata, created_at }` in the ingress node's state directory, under `api.responses.retention` (7 days). `previous_response_id` then works like this:

```text
continue(prev_id, new_input):
    r = responses.get(prev_id)                        // 404 response_not_found if missing or expired
    require r.model == resolved model and the session binding matches   // else session_mismatch (§12.1)
    prompt = r.transcript ++ render(new_input)        // instructions are not inherited
    SequenceOpen { session: r.session, prefix: PrefixRef::Session(r.session, r.transcript.len()) }
```

The session KV reaches exactly the stored transcript, so the head prefills only the new input, whether the KV is still resident, parked on NVMe or held by another node (RFC-0022).

**Stored responses live on the ingress node only** (M3; OQ-PRD-10, answered). There is no copy beside the session KV on the session's holders. A copy there would put a plaintext transcript on every holder's disk for the retention period, including other people's invited laptops, to rescue the rare case in which the ingress dies but its client survives. Session blobs on holders are ciphertext under a per-session key the ingress holds (RFC-0022), so a holder could not read such a copy after the ingress was lost anyway. Stored responses are listed in RFC-0034 §13's inventory of data at rest.
- A request carrying a `previous_response_id` that another node stored is forwarded by ID prefix: response IDs embed the storing node's short ID (`resp_<node short id, 6 base62><20 base62>`), and ingress fetches the record over the control connection's node-query channel (proposed amendment A4).
- If that node is gone, the response is gone too. The API says so with `response_not_found`, and the client resends its full input, which the prefix cache still shortens where it can. The detach grace (§5.5) covers the common failure, a dropped link.
- Response IDs are bearer capabilities: never logged, never on the event stream, compared in constant time. Retrieving or deleting a stored response over loopback needs the local socket or a key (§11.1).

`store: false` keeps nothing: no record and no persisted KV beyond the normal prefix cache.

#### 9.3 Native sessions

```text
POST   /v1/syntropy/sessions                    { model, persist?: bool, label? }   → { id, … }
GET    /v1/syntropy/sessions                    ?model=…                                               → list with sizes, tiers, holders
GET    /v1/syntropy/sessions/{id}                                                                        → detail
POST   /v1/syntropy/sessions/{id}/persist       park now: KV to NVMe on its holders
POST   /v1/syntropy/sessions/{id}/fork          { at_token? } → new session sharing pages copy-on-write
DELETE /v1/syntropy/sessions/{id}               DropSession; frees KV everywhere
```

Session IDs are `ses_` plus the base32 of the 16-byte `SessionId` (RFC-0001 §5.9). A session is bound to its model's identity hash and every setting that changes KV contents (RFC-0022). Using it with another model or KV dtype fails with `session_mismatch` rather than producing garbage.

#### 9.4 What the client sees in usage

| Field | Meaning |
|---|---|
| `cached_tokens` / `cache_read_input_tokens` | prompt tokens not prefilled because of a prefix-cache or session hit; only hits in the caller's own scope, plus hits inside a shared prefix the request itself marks with `cache_control` (§9.5), so the figure tells a client nothing about another client's prompts |
| `cache_creation_input_tokens` | prompt tokens prefilled and kept under a `cache_control` breakpoint or a persisting session |
| `syntropy.timings.restore_ms` | time spent restoring parked KV, separate from prefill (SC-19) |

#### 9.5 Prefix-cache scope

A prefix cache shared by every client of an instance, together with `cached_tokens`, is an exact oracle: one client can confirm a guessed prefix of another client's conversation, page by page, by reading the count. So the cache is scoped per principal by default:
- **The principal** is fixed by authentication (§11): the key ID of the API key presented, or `local` for loopback requests without a key and for local-socket peers, which are always the daemon's own OS user. From M2 a cluster-wide key's `key_id` is its principal on every member (RFC-0034 §8.4).
- **Per-principal pages.** Ingress passes the principal to the head in `SequenceOpen::prefix_scope` (amendment A7). The head derives the prefix key's `scope_salt` from it (RFC-0022 §7.1), so two principals never share pages and never learn anything from each other's hits.
- **Shared prefixes.** The prefix up to a client's last `cache_control` breakpoint (Anthropic shape, and the same field on native messages) is written under the instance-wide scope (`shared_until`), and any principal whose request marks the same prefix can hit it. Sharing rounds down to whole 64-token pages: a page that straddles the breakpoint stays private. A client marks only what it means to share, such as a system prompt, and `cached_tokens` counts a shared hit only when the request itself marked that prefix.
- **Single-user setups lose nothing.** A machine whose only clients are local is one principal, `local`, so every local client shares the cache as before.
- **Sessions** are bound to the principal that created them. Another principal naming the session ID gets `session_not_found`, never the session.
- RFC-0022 §7.1 owns the scope mechanism: `kv.prefix_scope = principal` is the default, and `instance` is an opt-in to one shared cache for setups that want it. RFC-0034 owns the threat model. This section fixes how the API derives the principal and what the client sees.

### 10. Fidelity and approximate-mode labels

ADR-0019 decision 7 requires every response to carry its fidelity and the active knobs. RFC-0001 §10.4 requires Syntropy metadata on every response. Which fidelity a request gets is decided by RFC-0054 §7, following RFC-0001 §11.3: approximate only when the request, the instance policy and the cluster all opt in, with the knob sets intersected. A request that asked for approximate but resolved to exact is served exactly, and `approximate_not_available` appears in `syntropy.warnings`.

```rust
/// Owner: RFC-0050.
pub struct FidelityReport {
    pub fidelity: Fidelity,                  // Exact | Approximate
    pub enabled: ApproxFlags,                // knobs in effect for this request (RFC-0054 §4.2)
    pub applied: ApproxFlags,                // union of TokenEntry::approx over the sequence
    pub contract: ContractRef,               // tolerance table version (RFC-0042), e.g. "parity-v3"
    pub certification: Certification,        // which manifest vouches for the serving plan (RFC-0060 §6.4)
    pub refused: Vec<CertRefusal>,           // run-time features withheld as uncertified (RFC-0060 §6.3)
    /// First position past the family's certified context (`FamilyCert`, ADR-0019 decision 7;
    /// RFC-0042 §5.7), when this sequence reaches it; None while every position is within it.
    pub beyond_certified_from: Option<u32>,
}
/// Read from the committed plan, which carries its certification status and the run-time values
/// certified for its plan-time path (RFC-0011; RFC-0060 §6.3, §6.4). Owner: RFC-0050.
pub enum Certification {
    Certified,                               // every op, path and wire of the plan holds a full certificate
    Derived,                                 // at least one op runs under a derived certificate whose canary passed
    Uncertified,                             // no valid manifest for this build profile, or `--uncertified`
    CiTest,                                  // the manifest is signed by the CI test key (RFC-0054 §7.1)
}
pub enum CertRefusal {
    PrefixReuse,                             // prefix-cache hits disabled on this instance
    SessionResume,                           // the session re-prefills from its transcript instead of resuming KV
    Speculation,                             // speculative decoding disabled
    Concurrency { certified: u16 },          // batching capped at the certified sequence count
    KvSpill { stage: StageIx, tier: TierKind, outcome: SpillCert },   // KV_SPILL_UNCERTIFIED (RFC-0022 §5.3)
}
pub enum SpillCert {
    Substituted { used: SpillCompute },      // the plan's spill path is uncertified; the certified other path runs (slower)
    NotSpilled,                              // neither path is certified: the tier is left out of KvHeadroom
    UncertifiedUsed { used: SpillCompute },  // approximate sequences only: an uncertified path runs, and is labelled
}
pub struct SyntropyMeta {
    pub model: String, pub model_id: ModelId,
    pub instance: InstanceId, pub plan_epoch: PlanEpoch,
    pub fidelity: FidelityReport,
    pub degraded: Vec<Degradation>,          // RFC-0001 §10.2 declared paths in effect
    pub recovery: Vec<RecoveryEvent>,        // migrations, replays, re-prefills during this request
    pub warnings: Vec<String>,               // ignored fields, unparsed tool calls …
    pub timings: Timings,                    // queue, load, ttft, decode tok/s, restore
    pub repro: ReproFingerprint,             // M4; below
}
/// Tells a user or a test harness when a changed output is expected. Owner: RFC-0050.
pub struct ReproFingerprint {
    pub build_id: BuildId,
    pub plan_epoch: PlanEpoch,
    pub device_set: Blake3,                  // digest of the plan's (node, device, provider) assignment
    pub deterministic: bool,                 // the instance's same-device deterministic option (ADR-0019 decision 8)
    pub seed: u64,                           // the seed the sampler used, client-supplied or generated
}
```

**Where it appears.**

| Shape | Non-streaming | Streaming |
|---|---|---|
| OpenAI chat and completions | top-level `syntropy` object | `syntropy` with `fidelity` and `enabled` on the first chunk; the full object on the final chunk (the usage chunk when `include_usage`, else the last choice's finishing chunk) |
| OpenAI Responses | `syntropy` on the response object | on `response.created` (fidelity, enabled) and `response.completed` / `incomplete` / `failed` (full) |
| Anthropic | `syntropy` on the message | on `message_start.message` (fidelity, enabled) and on `message_delta` (full) |
| Native | first-class fields | `started` and `done` events |

**Headers.** Every response, streaming or not, carries `Syntropy-Fidelity: exact` or `approximate` and, when approximate, `Syntropy-Approximate: <enabled knob names>`. Headers go out before the first token, so they can report only enabled knobs; the body reports what was applied.

**Certification and refusals**.
- **Two different things, two words.** `certification` says which manifest vouches for the serving plan, and `Uncertified` there means no valid manifest. `beyond_certified_from` says where the sequence passed the longest context the family is certified for. That is the context label ADR-0019 decision 7 requires, and it is called "beyond certified context", never "uncertified".
- **Not a fidelity change.** `certification`, `refused` and `beyond_certified_from` never change `fidelity`. An exact response stays exact; these fields say what vouches for that, and what was withheld to keep it so.
- **Where they appear.** `certification` and `refused` are known at routing time, so they appear wherever `fidelity` does, first chunks included. `beyond_certified_from` appears on the first chunk when the prompt already passes the certified context, and otherwise in the final object once generation passes it.
- **The context label.** A response with `beyond_certified_from` set is labelled, wherever a person reads it, "exact up to token N; beyond the certified context after it", and carries `Syntropy-Certified-Context: <N>`.
- **The `Syntropy-Certification` header** is sent only when the value is not `certified`: `derived`, `uncertified` or `ci-test`.
- **Wording.** Wherever a person reads it, the label is "exact (derived certification)", "exact (uncertified)" or "exact (CI test manifest)". The same goes for approximate responses.
- **Which refusals are listed.** `refused` lists the refusals from the instance's committed plan that bear on this request. `KvSpill` comes from each stage's tier manager instead: it decides the outcome per tier when the plan is applied and reports it under `KV_SPILL_UNCERTIFIED` in the stage's `ResidencySummary` (RFC-0020, RFC-0022 §5.3), which ingress reads.
  - `PrefixReuse`, `Concurrency`, and `KvSpill` with `Substituted` or `NotSpilled`, are always listed when present;
  - `KvSpill` with `UncertifiedUsed` only on approximate requests, because exact sequences never use an uncertified path;
  - `SessionResume` only when the request names a session;
  - `Speculation` only when the instance policy enables speculation.
- **Never silent.** A refusal is never a silent fallback. The scheduler and tier manager enforce it (RFC-0012, RFC-0022), and the response says it happened.

**Byte-identical output across devices is not promised** (ADR-0019 decision 8). Two requests with equal `repro` fingerprints, `deterministic` set and the same input reproduce on the same build, plan epoch and devices (ADR-0019 decision 8; the `deterministic` option, M3). A changed fingerprint says the output may change. Nothing more is claimed.

**Provenance record** (opt-in, M4). With `api.provenance = true`, ingress writes one local record per completed response to `<state>/provenance/`: `ModelId`, `TokenizerId`, the `ReproFingerprint`, the sampler settings, and BLAKE3 hashes of the input token IDs and of the output token IDs. Records hold hashes, never content, so they fall under the same privacy rules as metrics. They never leave the machine, they are kept until `syntropy provenance prune` or `api.provenance.retention` removes them, and doctor STO-001 counts their size. A user who must show which model, build and settings produced a given output keeps the output and checks its hash against the record.

**Strict clients.** Official SDKs accept unknown response fields. For any client that rejects them, `api.extensions = "headers"` drops the body object and keeps the headers; `off` drops both (the dashboard and CLI still show fidelity).

**Wording.** Wherever a person reads it (CLI banner, dashboard badge, docs), an approximate response is labelled "approximate" followed by the knob names, never a softer word. The CLI prints it before the first token of `syntropy run` (RFC-0051 §2.4).

### 11. Authentication and bind policy

This RFC owns client-facing API authentication: binds, TLS sources, key format, scopes, headers, rate limits and the dashboard login (ADR-0505). RFC-0034 §11 delegates it here and lists the threat-model requirements it must meet, R1–R5. RFC-0001 §12 zone Z3 fixes that a non-loopback bind needs API keys and TLS.

#### 11.1 Binds

RFC-0001 §12 zone Z3, as updated: any non-loopback bind requires an API key **and** TLS, with two exceptions for the TLS part only.

| Bind | API key | Transport |
|---|---|---|
| loopback (`127.0.0.1`, `::1`) and the local socket | on loopback, none by default for `inference` routes and the read-only routes of the `dashboard` scope; `admin` and `sessions` routes, the session binding headers, stored-response retrieval and deletion, and every action that needs a keyholder signature require a key on loopback HTTP (401 `local_auth_required`, whose hint names the local socket, a key or `syntropy dashboard`). `api.auth.mode = "always"` requires keys for every route on loopback, which `doctor` recommends on hosts with several interactive users (RFC-0034 R4). The local socket checks OS peer credentials instead and carries every scope | plain HTTP on loopback; the socket is local |
| addresses on interfaces RFC-0030 §7.2 classifies as `Overlay` (WireGuard-based, encrypted end to end, such as a tailnet) | required | TLS may be omitted: the overlay already encrypts the path |
| any other address (LAN, wildcard, public) | required | TLS required. Plain HTTP is possible only on LAN addresses, with the explicit opt-in `api.allow_plaintext_lan = true`, which the dashboard and `doctor` flag permanently as insecure (SEC-005) |

- **Why loopback is split.** Any local process can reach `127.0.0.1`: other OS users, a co-located service with an SSRF bug, a container sharing the host's network namespace, an agent's HTTP tool. The Host and Origin checks of §11.2 stop browsers only, because a non-browser client writes its own `Host` and sends no `Origin`. Chat clients and SDKs on loopback keep working with no key; what a stray local process can no longer do is mint invites, approve joins, create keys or change configuration on a keyholder. The CLI is unaffected, because it uses the local socket.
- **Remote clients, in the order the docs recommend** (confirmed): an overlay bind first (key required, no TLS, §11.1 table); then `syntropy trust` with its environment-variable output; from M2, a desktop that runs Syntropy joins as a **client-only** member at LAN pairing (ingress-only, RFC-0030 §6), calls its own loopback and needs neither a key nor a CA, because its requests ride the cluster's mutually authenticated planes. Phones and machines without Syntropy use the overlay or a user-supplied certificate.
- **Refusal to start.** The daemon refuses to start a non-loopback bind until at least one API key exists. The error names the command: `syntropy api-key create --name … --scope inference`. TLS never blocks start, because the local CA (below) supplies a certificate without configuration.
- **Wildcards.** `0.0.0.0` and `::` count as LAN binds. They are served with TLS. An overlay interface reached through a wildcard bind still gets TLS, because the exception applies only to binds restricted to overlay interfaces.
- **TLS sources** (`api.tls`), TLS 1.3 through `rustls`:
  - `"auto"` (default): the local CA for LAN binds; no TLS for binds restricted to overlay interfaces.
  - `"local_ca"`: always the local CA, overlay binds included.
  - `{ cert, key }`: the user's files, for example from their own CA or a certificate issued for a tailnet name. Reloaded on change.
  - `"none"`: valid only for overlay-restricted binds, or LAN binds with `api.allow_plaintext_lan = true`. Configuration validation rejects it anywhere else.

*Note 2026-10-22 (PROD-E5-T02, as built; open, for the maintainers): which addresses are LAN addresses.* The table above allows plain HTTP, with the opt-in, on LAN addresses only, and sets them apart from public ones. `syntropy_api::bind::classify` reads them as:
- IPv4 `10/8`, `172.16/12`, `192.168/16` and link-local `169.254/16`;
- IPv6 unique local `fc00::/7` and link-local `fe80::/10`;
- the wildcards `0.0.0.0` and `::`, as above. A wildcard counts as a LAN bind for the API key, but it listens on every interface, a public one included, so it is always served with TLS: `api.tls = "none"` on a wildcard is rejected even with the opt-in, and plain HTTP needs a specific LAN address.

An IPv4-mapped IPv6 address takes its IPv4 address's class. Every other non-loopback address is public: TLS always, with no opt-in. That includes `100.64/10`, the shared range tailnets use, until PROD-E5-T09 classifies binds by RFC-0030's interface classes; until then a tailnet bind needs TLS. Loopback listeners serve plain HTTP whatever `api.tls` says, `"local_ca"` and `{ cert, key }` included.

#### 11.1.1 The local CA

TLS on a LAN is practical only if clients can trust the certificate without a public CA. Each node therefore runs a small certificate authority of its own.

- **The CA.**
  - Generated at the first non-loopback bind: an ECDSA P-256 key. ECDSA rather than Ed25519, because not every client TLS stack accepts Ed25519.
  - Stored in the state directory (mode 0600), or in the OS keystore when RFC-0034's `security.keystore` provides one.
  - Bound to the node identity by a statement signed with the node key (RFC-0034 R5).
  - Its fingerprint is shown as six words (RFC-0034 §6.5) by `syntropy identity show`, `doctor` and the dashboard.
- **One CA per node, never a shared CA key**. No node holds another node's CA key, and issuance never goes through the coordinator, so one stolen disk yields one node's CA, not a CA every client trusts for every node.
- **Name constraints** (narrowed under). The CA certificate carries critical X.509 name constraints. It may issue only for:
  - the node's DNS names: `node.name`, the OS host name, `<host>.local`, and `api.server_names`;
  - the subnets the node serves when the CA is generated: each bound interface's /24 (IPv4) or /64 (IPv6), and the node's own overlay address as a /32 or /128.
  A leaked CA key therefore cannot impersonate public sites, nor a router, NAS or other private host outside those subnets, to the machines that trust it. When the node's served subnets change (it moves to another network), it generates a new CA, `doctor` SEC-006 reports that clients must re-trust, and the dashboard says so.
- **Leaf certificates.** The daemon issues a leaf for the node's current names and LAN addresses, valid for 30 days. It re-issues on address change or 7 days before expiry, and serves the new leaf without dropping connections.
- **`syntropy trust <node>`.** Run on a client machine, this installs a node's CA there.
  1. It fetches the CA certificate from `GET /v1/syntropy/tls/ca`. This is the only route served without a key on non-loopback binds, and it returns public material only.
  2. It shows the six-word fingerprint and asks the user to compare it with the one the node displays (`syntropy identity show` or the dashboard).
  3. On confirmation, it installs the CA into the **user's** trust store by default:
     - the login keychain on macOS;
     - the current-user Root store on Windows;
     - on Linux, the user's NSS database and the PEM file below. The system store, through `update-ca-certificates` or `update-ca-trust`, only with `--system` (needs sudo, asked first), because there every user of the machine trusts it.
  4. Every change is recorded in the client's host-change ledger (RFC-0052 §11), so `syntropy trust --remove <node>` and `syntropy uninstall` revert it.
- **`syntropy trust --cluster <node>`** (M2). It fetches every member's CA certificate from `GET /v1/syntropy/tls/cluster-cas` on that node. Each certificate is bound to its node key by the node-key statement above, and each node key is bound to the cluster key by the node's admission certificate (RFC-0034 §5). The command shows the cluster ID's six-word fingerprint once, checks every binding, installs all the CAs in one step under the same store rules, and picks up new members when run again.
- **Friction, stated plainly.** Trusting the OS store is not always enough, and the docs say so:
  - **Client runtimes with their own CA bundles.** Python's common HTTP stacks use a bundled certificate set, and Node.js ignores the OS store unless told otherwise. So `syntropy trust` also writes a PEM file and prints the environment variables those clients read: `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`.
  - **Phones.** iOS needs a configuration profile installed by hand, and then full trust enabled in Settings. Android apps ignore user-installed CAs unless they opt in. For phones the practical routes are an overlay network (no TLS needed, key still required) or a certificate from the user's own CA via `{ cert, key }`.
  - **Every client machine** needs this step once per node it talks to, or once per cluster with `trust --cluster` (open question 8, answered).
- **Plain HTTP on the LAN.** `api.allow_plaintext_lan = true` exists for users who accept the risk, for example on an isolated lab network. Keys are still required. The dashboard shows a permanent "API traffic on this LAN is unencrypted" banner, `doctor` reports SEC-005 as a warning on every run, and the startup log says so on every start.

*Note 2026-10-22 (PROD-E5-T03): how the CA is built.*
- **Storage.** The CA lives in one file, `tls/ca.json`, in the state directory. It holds the key (PKCS#8), the certificate (DER), the scope and the node-key binding (RFC-0034 §3's `syntropy-tls-ca-v1`, note of the same date), which covers the certificate's exact bytes.
  - The file is mode 0600, in a 0700 directory. A file or directory whose mode was widened is narrowed again when the CA is loaded.
  - The `tls` directory and `ca.json` must be a directory and a regular file themselves. A link is refused, not followed, so a link planted in the state directory cannot get another directory narrowed or written into.
  - It is written to a new file, flushed, renamed over the old one, and then the directory is flushed, so a replacement is one atomic step.
  - A file the daemon cannot read, or whose parts disagree, is refused and left as it is. No key is overwritten without a reason the user can see.
  - Agreeing means the binding verifies over the stored certificate, and the certificate is the key's own, self-signed, with exactly the constraints the node mints for the stored scope.
  - `GET /v1/syntropy/tls/ca` serves one PEM block encoded from those bytes, so it carries nothing the binding does not cover.
- **Randomness.** The CA and leaf keys come from ring's `SystemRandom`, the OS CSPRNG of RFC-0034 §3. ring's random source is sealed, so it cannot draw from `core`'s `Entropy`.
- **The CA certificate.** It is valid for 10 years [target, a chosen budget], since it is replaced when its scope changes, not on a schedule.
  - Its path length is 0. That is advisory: a client that takes the CA as a trust anchor may ignore the anchor's basic constraints, as rustls does, and then accepts a sub-CA it signed. The name constraints are the guarantee every client enforces, and they still apply below such a sub-CA.
  - Its key usages are certificate and CRL signing.
  - A certificate's validity starts 1 hour before it is issued [target, a chosen budget], for clients whose clocks run behind.
- **Names.** A dNSName constraint also permits every name below it (RFC 5280 §4.2.1.10).
  - So the scope leaves out single-label names. A node named `box` would otherwise get a CA for the whole `.box` top-level domain.
  - The node is still reachable as `<host>.local` and by its other names of two or more labels.
  - A leaf carries the scope's names exactly; a name below one of them is refused.
  - A limit: a name of two labels can still cover a public zone. A node name or `api.server_names` entry that is a public suffix, such as `co.uk`, or a bare registrable domain, such as `example.com`, gives the CA that whole zone. `api.server_names` should hold the node's own host names only.
- **Addresses.** The scope leaves out public and loopback addresses: a local CA covers only private, shared (RFC 6598), link-local and unique-local ones.
  - An address family with no served subnet is excluded whole, so the CA can issue for no address of that family.
  - A node that serves only public addresses gets a CA for its names alone.
  - The CA covers the whole served /24 or /64, so a leaked key can impersonate any host inside those subnets, a neighbour on the same LAN included, though none outside them.
- **Replacement.** A new CA replaces the old one when the node's names or served subnets change, and also when the node key changes (`identity reset`), since the old binding names another identity. The replacement is recorded first in `tls/ca-replaced.json`: when, why, and the old and new fingerprints, `BLAKE3("syntropy-fp-v1" ‖ SPKI DER)` (RFC-0034 §6.5). SEC-006 reads that record. A CA that has been replaced issues no more leaves: before each leaf it checks that `tls/ca.json` still holds its certificate.
- **Rotation.** The listener serves the current leaf through a resolver that swaps leaves. A connection keeps the leaf it was handshaken with, and the next full handshake gets the new one. A client that resumes an earlier TLS session sees no certificate until its session ends.

#### 11.2 Requests from browsers

A loopback listener without authentication is reachable by any web page the user visits, through DNS rebinding or cross-site requests (RFC-0034 R1). So on every listener:
- **Host check.** On loopback listeners, whatever `api.auth.mode` says, the `Host` header must be a loopback name or address, or an entry in `api.server_names`. This strict list exists because DNS rebinding is the threat there. On non-loopback listeners, the allowed set is the local CA leaf certificate's name set, recomputed whenever addresses change: `node.name`, the OS host name, `<host>.local`, every current interface address served by the bind (all of them for a wildcard bind), and `api.server_names`. Any other `Host` gets 421 (Misdirected Request).
- **Origin check.** A state-changing request carrying an `Origin` header other than the dashboard's own origin is rejected. CORS is off by default; `api.cors_origins` is an explicit allowlist, and it applies to `inference` routes only: it never opens an `admin` or `sessions` route to another origin. Preflight `OPTIONS` requests are answered only for listed origins and `inference` routes.
- **CSRF.** The dashboard sends a per-session CSRF token on every state-changing request.

#### 11.3 API keys

This RFC fixes the format, storage and scopes (ADR-0505 items 3–4):
- **Format.** `syk_` followed by 43 base64url characters (32 random bytes), shown once at creation. The **key ID** is the first 8 characters after `syk_`. It is stored beside the hash, shown by `api-key list`, used as `key_id` for cluster-wide keys (RFC-0034 §8.4), and used to key the failure back-off below. It is a lookup handle, not a secret: 8 characters reveal 48 of the 256 random bits.
- **Storage.** BLAKE3 hashes in `<state>/api-keys.toml` (mode 0600), with name, scopes, creation and last-use times; compared in constant time; hot-reloadable.
- **Presentation.** `Authorization: Bearer <key>` or `x-api-key: <key>`, so OpenAI and Anthropic SDKs work as configured.
- **Scopes, and which routes each opens:**

| Scope | Routes |
|---|---|
| `inference` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses*`, `/v1/messages*`, `/v1/embeddings`, `GET /v1/models`, `POST /v1/syntropy/generate` |
| `sessions` | `/v1/syntropy/sessions*`, and the session binding headers on inference routes |
| `admin` | pulls, instances, placement previews, configuration, cluster operations, invites, admissions, revocation, API keys, debug dumps |
| `dashboard` | the dashboard, and the read-only native routes it uses: `GET` on `/v1/syntropy/{cluster,nodes,instances,models,doctor}`, the `events` and `metrics/live` streams, `/metrics` |

- **Management.** `syntropy api-key create --name phone --scope inference [--model <name>]… [--instance <id>]… [--expires 90d] [--cluster]`, `list`, `revoke`, `rotate <id>` (RFC-0051 §2). Over the API, the same operations need `admin`. Optional model and instance lists narrow any key: an `inference` key scoped to one model gets 403 `insufficient_scope` on every other.
- **Cluster-wide keys** (RFC-0034 §8.4; M2). RFC-0001 lets any node be the ingress, so one key reaches the cluster through any member.
  - `api-key create --cluster` generates the secret and shows it once, on the node where it runs.
  - The node submits `CreateApiKey { key_id, hash, scopes, instances, models, expires_ms }`, which a keyholder signs. A non-keyholder node forwards it to a keyholder for an operator's approval. The secret never enters the log.
  - Every daemon checks a presented key against its node-local keys and the log's `api_keys` set, by hash and in constant time.
  - `rotate` creates the new key, then revokes the old one; the two overlap so clients rotate without downtime.
  - Revocation is keyholder-signed and monotonic, and a revoked key ID is never re-admitted.
  - Cluster-wide keys are events, not configuration (RFC-0054 §2.3 step 7).
- **Listeners that do not require keys ignore presented credentials**. A loopback listener with `api.auth.mode = "auto"` never inspects `Authorization` or `x-api-key`, and never counts them as failures. SDKs that refuse to start without a key can therefore use any dummy value.
- **Failures.** A failed attempt returns 401. Back-off is keyed by (peer address, presented key ID, §11.3 "Format"): more than 10 failures per minute for one pair return 429 for 5 minutes, for that pair only. The local socket and loopback addresses are never locked out address-wide, so one misconfigured local client cannot lock out the CLI, the dashboard or other local clients. `api.rate_limit.per_key` adds an optional per-key request limit.
- **Approximate mode is not a scope.** Whether a request may run approximate is decided by the instance policy and the cluster key `allow_approximate` (RFC-0054 §7).

#### 11.4 Dashboard sessions

On loopback the dashboard opens read-only, with the `dashboard` scope's routes. Views that change state get an `admin` session from a **one-time login URL**: `syntropy dashboard`, which proves the OS user over the local socket, asks `POST /v1/syntropy/auth/login-token` (local socket only) for a single-use token and opens `http://127.0.0.1:7433/#login=<token>`. The token sits in the URL fragment, so it never reaches a server log or a `Referer`. The page exchanges it once at `POST /v1/syntropy/auth/exchange` for a session cookie carrying every scope the local socket has, and the token expires unused after `api.dashboard_login_ttl` (2 min [target]). The Windows Start-menu entry "Syntropy dashboard" runs `syntropy dashboard`, so a user who never opens a terminal still lands signed in. On a non-loopback bind the dashboard shows a login page that takes a key with the `dashboard` or `admin` scope. It sets an `HttpOnly`, `Secure`, `SameSite=Strict` session cookie, valid for 12 hours and bound to the key ID. Revoking the key ends its sessions. Views that change state (load, config, invites) require `admin`; with a `dashboard`-scoped key they are read-only.

#### 11.5 Debug endpoints

The endpoints of §13.6 are outside the scope system. They answer only on loopback listeners and the local socket, and only while their `debug.*` key is set (RFC-0054 §7.1). From any other peer they return 404, whatever key the caller holds.

### 12. Error mapping

#### 12.1 Before the first token

Errors become the request shape's own error body:
- OpenAI: `{"error": {"message", "type", "param", "code"}}`;
- Anthropic: `{"type": "error", "error": {"type", "message"}}`;
- native: `{"error": {"class", "code", "domain", "message", "retry", "scope", "context"}}`, which is the full `Error` of RFC-0001 §5.12 with the message for people and the code for programs.

| Condition | `ErrorClass` | HTTP | OpenAI `type` / `code` | Anthropic `type` |
|---|---|---|---|---|
| malformed body, bad field, unknown `syntropy` field | `InvalidRequest` | 400 | `invalid_request_error` / `invalid_request` | `invalid_request_error` |
| prompt plus `max_tokens` over the context window | `InvalidRequest` | 400 | `invalid_request_error` / `context_length_exceeded` | `invalid_request_error` |
| a token ID at or above the model's `vocab_limit` in `tokens`, teacher tokens, `logit_bias` or stop IDs (§2.3) | `InvalidRequest` | 400 | `invalid_request_error` / `token_id_out_of_range` | `invalid_request_error` |
| gated model without Hub acceptance, or custom licence not accepted on this node (§4.1) | `InvalidRequest` | 400 | `invalid_request_error` / `licence_acceptance_required` | `invalid_request_error` |
| feature, content type, tool or endpoint not supported | `Unsupported` | 400 (404 for endpoints) | `invalid_request_error` / `unsupported_feature`, `unsupported_content_type`, `unsupported_tool`, `unsupported_endpoint` | `invalid_request_error` (`not_found_error` for endpoints) |
| exact request whose only instances carry instance-wide approximate knobs; approximate request that leaves out a plan's knob (RFC-0054 §7, option (c)) or whose knobs hold a bit this version cannot name (revised 2026-10-23, note below) | `InvalidRequest` | 400 in vendor shapes (409 in the native API) | `invalid_request_error` / `fidelity_conflict`, naming the knobs | `invalid_request_error` |
| knobs named without `fidelity: "approximate"`; `BenchControls` without `debug.bench_api` | `InvalidRequest` | 400 | `invalid_request_error` / `knobs_need_approximate`, `debug_api_disabled` | `invalid_request_error` |
| model unknown | `InvalidRequest` | 404 | `invalid_request_error` / `model_not_found` | `not_found_error` |
| model does not fit anywhere, even after evicting idle instances | `Capacity` | 400 | `invalid_request_error` / `model_does_not_fit` | `invalid_request_error` |
| pins or overrides infeasible | `Capacity` | 400 | `invalid_request_error` / `placement_infeasible` | `invalid_request_error` |
| model loading beyond `api.load_wait` | `Capacity` | 503 + `Retry-After` | `server_error` / `model_loading` | `overloaded_error` (529) |
| admission queue full or `queue.max_wait` passed | `Capacity` | 429 + `Retry-After` | `rate_limit_error` / `admission_queue_full` | `overloaded_error` (529) |
| per-key rate limit, auth back-off | `Capacity` | 429 + `Retry-After` | `rate_limit_error` / `rate_limited` | `rate_limit_error` |
| missing or invalid key | `Auth` | 401 | `authentication_error` / `invalid_api_key` | `authentication_error` |
| an `admin`, `sessions` or keyholder route over loopback HTTP without a key (§11.1) | `Auth` | 401 | `authentication_error` / `local_auth_required` | `authentication_error` |
| key lacks scope | `Auth` | 403 | `permission_error` / `insufficient_scope` | `permission_error` |
| session or stored response mismatch | `InvalidRequest` | 400 in vendor shapes (409 in the native API) | `invalid_request_error` / `session_mismatch` | `invalid_request_error` |
| stored response or session not found | `InvalidRequest` | 404 | `invalid_request_error` / `response_not_found`, `session_not_found` | `not_found_error` |
| body larger than `api.max_body` | `InvalidRequest` | 413 | `invalid_request_error` / `request_too_large` | `request_too_large` |
| request deadline before the first token | `Timeout` | 504 | `server_error` / `timeout` | `timeout_error` |
| stage, link or device failure not recovered before the first token, after one re-route (§4.2) | `Link`, `Device` | 503 + `Retry-After: 1` | `server_error` / `instance_unavailable` | `overloaded_error` (529) |
| no instance passes its canary | `Correctness` | 503 | `server_error` / `no_conforming_instance` | `api_error` (500) |
| constraint compile failure | `InvalidRequest` or `Unsupported` | 400 | `invalid_request_error` / `constraint_invalid`, `constraint_unsupported` | `invalid_request_error` |
| the only instances of the model are unroutable, or recovery found no feasible repair (`InstanceNotRoutable`, RFC-0032 §10; `NoRepairCapacity`, RFC-0033 §13) | `Capacity` | 503 + `Retry-After` | `server_error` / `instance_unavailable` | `overloaded_error` (529) |
| `POST /v1/syntropy/cluster/join` with a malformed invite token (`TokenMalformed`, RFC-0030 §10) | `InvalidRequest` | 400 | — (native route only) | — (native route only) |
| a join the keyholder or the joining node refuses: invite token expired, used up or revoked, pairing denied, clocks more than 10 min apart, or a node key made on other hardware (`TokenExpired`, `TokenExhausted`, `TokenRevoked`, `PairDenied`, `ClockSkew`, `IdentityHostMismatch`; RFC-0030 §6.7, §10) | `Auth` | 403 | — (native route only) | — (native route only) |
| a join refused because the two nodes' protocol majors differ (`ProtoMismatch`, RFC-0030 §10) | `Unsupported` | 400 | — (native route only) | — (native route only) |
| an operator action that needs a keyholder signature with no keyholder reachable (`KeyholderRequired`, RFC-0034 §8.3, §15) | `Auth` | 403 | — (native route only) | — (native route only) |
| a cluster configuration proposal rejected: the key changed since the patch's base, or it sits above the cluster's feature level (`ConfigConflict`, `FeatureLevelTooLow`; RFC-0032 §10, RFC-0054 §2.3), reported as `cluster_proposal_rejected` | `InvalidRequest`, `Unsupported` | 409 | — (native route only) | — (native route only) |
| `Internal` | `Internal` | 500 | `server_error` / `internal_error` | `api_error` |

`Integrity`, `Drift` and `Transient` never reach clients directly: the store, the coordinator or the retrying layer handles them (RFC-0001 §10.1). If one escapes, it maps to 503 `instance_unavailable`.

**Distributed error codes** (note 2026-10-22, issue #1668). The `Cluster`, `Net`, `Dist`, `Recovery` and `Security` domains (RFC-0030 §10, RFC-0031 §12, RFC-0032 §10, RFC-0033 §13, RFC-0034 §15) follow two rules:
- **A code that can reach an API client has a row above.** The cluster rows are native routes only (`/v1/syntropy/cluster…`, `/v1/syntropy/nodes…`, `/v1/syntropy/config…`; §1.2), which the OpenAI and Anthropic shapes never serve. Their body is the native error, which names the domain code.
- **A code that never leaves the cluster is exempt.** These are raised between nodes during peer-to-peer protocol, handshakes and admission, or reported as node diagnostics through `syntropy doctor` and the dashboard, never as the error of a request:
  - `Cluster`: 0x0102 `NodeConflict`, 0x010A `LocalNetworkDenied`, 0x010B `NetworkProfilePublic`;
  - `Net`: 0x0201 `ProtoMajorMismatch`, 0x0202 `CredentialInvalid`, 0x0203 `NotANeighbour`, 0x0204 `UnknownFrameType`, 0x0205 `FrameTooLarge`, 0x0206 `WireDTypeMismatch`, 0x020D `RdmaNotEnabled`, 0x020E `RdmaOsBuildMismatch`, 0x020F `ShmUserMismatch`, 0x0212 `ProtoRangeDisjoint`;
  - `Dist`: 0x030E `EntryRejected`;
  - `Recovery`: 0x0407 `HistoryDropped`;
  - `Security`: 0x0501 `CertInvalidSignature`, 0x0502 `CertExpired`, 0x0503 `CertRevoked`, 0x0504 `CertWrongCluster`, 0x0505 `RoleInsufficient`, 0x0506 `CredentialExpired`, 0x0508 `RotationWindowClosed`, 0x0510 `UntrustedFabric`, 0x0511 `AuthorityInvalid`, 0x0512 `RotationChainInvalid`, 0x0513 `NotCoordinatorEligible`.

  If an exempt code ever reaches an API response, it renders as the server-side shape, 503 `instance_unavailable`, whatever its class, because from the client's view the cluster could not serve the request. It never takes a client-side shape (400, 401 or 403) and never a new ad-hoc shape.
- *Note 2026-10-22 (#1669):* `WireDTypeMismatch` (0x0206) joins the `Net` exempt list, beside `UnknownFrameType` (0x0204).
  - RFC-0031 §12 classes both `Unsupported` (its revision notes of 2026-10-22), and no row above maps that class for them.
  - Both are a stage refusing a peer stage's frame: one in another dtype than the channel's `ChannelSpec::wire` (RFC-0031 §6.4), the other of a type the negotiated minor lacks. The refusal travels in the channel's `Close` and never becomes a request's error. If it ever reached an API response, it would render as 503 `instance_unavailable`, by the rule above.
- *Note 2026-10-23 (DIST-E7-T02):* `FrameTooLarge` (0x0205) joins the `Net` exempt list, beside 0x0204 and 0x0206. RFC-0031 §12 now classes it `InvalidRequest` (its revision note of this date), the class of a 400 row above. It is a stage refusing a peer stage's frame longer than the channel's slot size, and the refusal travels in the channel's `Close`. If it ever reached an API response, it would render as 503 `instance_unavailable` by the rule above, never as 400.
- *Note 2026-10-22 (#1698):* a join with a revoked invite token, `JoinReject::Revoked`, had neither a code nor a row. RFC-0030 §10 now gives it `TokenRevoked` (0x010F, `Auth`), and the join-refusal row above maps it with `TokenExpired` and `TokenExhausted`, to 403 on the native route. `KeyholderRequired`'s row rests on RFC-0034 §8.3's note of this date, which says when it is raised.
- **`DuplicateCommand` (0x0305) never appears in `Error::code`.** The coordinator answers a command whose `CommandId` it has seen with the original result (RFC-0032 §6.3), so the code names the dedup, not an error a caller receives.
- Every other code in these domains maps through a whole-class row above (`Link`, `Device`, `Timeout`, `Correctness`, `Internal`), or through the rule just above for `Integrity`, `Drift` and `Transient`. The one `Cancelled` code, `IngressLost` (0x0404), has no response to carry it, because the client's connection is already gone (RFC-0033 §13).
- *Note 2026-10-28 (COMP-E13-T15):* the `Kernels` domain (RFC-0040 §13) and the `Exec` domain (RFC-0012 §13) follow the same rules.
  - Their codes are raised inside a node, and the node handles them there. Selection, the module loader and the tier manager answer the `Unsupported` and `Capacity` ones by choosing another candidate or domain, or by re-planning. None of these is ever the error of a request, so they are exempt:
  - `Kernels`: 0x0001 `OpUnsupported`, 0x0002 `ModuleMissing`, 0x0004 `ModuleAbi`, 0x0005 `DriverTooOld`, 0x0007 `CaptureFailed`, 0x0009 `AllocFailed`, 0x000A `RegisterFailed`, 0x000C `GgmlImportFailed`, 0x000D `AmxDenied`, 0x000F `NoCertifiedCandidate`;
  - The other codes map through a whole-class row: `ParityFailed` (`Correctness`), `DeviceSticky` (`Device`), `DoorbellTimeout` (`Timeout`), and `WorkspaceExceeded` and `ProviderPanicked` (`Internal`). `ModuleIntegrity` and `ManifestInvalid` follow the rule for `Integrity`. Every `Exec` code maps the same way: `RingClosed`, `StepStalled` and `PoolBarrierMissed` (`Internal`).

The Anthropic shape uses 529 for overload, as the published API does, because Anthropic SDKs retry it. The OpenAI shape uses 429 and 503, which OpenAI SDKs retry.

**Retry signalling**. The official OpenAI and Anthropic SDKs retry 408, 409, 429 and every status ≥ 500 by default, unless the response carries `x-should-retry`. So:
- **No 409 in vendor shapes.** A client error that a retry cannot fix uses a 4xx status other than 408, 409 or 429 there. `fidelity_conflict` and `session_mismatch` are 400, and `model_does_not_fit` and `placement_infeasible` are already 400. Server-side errors whose retry hint is `Never`, such as `internal_error` (500) and `no_conforming_instance` (503), keep their status, and `x-should-retry: false` (next item) stops the SDK retries. The native API keeps 409 for conflicts, because its clients follow the `retry` hint in the body.
- **The header on every error, in every shape.** `x-should-retry: false` when `Error::retry` is `Never`. `x-should-retry: true` on `model_loading`, `instance_unavailable`, `admission_queue_full` and `rate_limited`, together with `Retry-After`.
- **A conformance case** counts the SDK's attempts per error code: one attempt for every non-retryable code, and the SDK's configured maximum for retryable ones.

**`fidelity_conflict`** (note 2026-10-23, from the review of #1771). `resolve_request` (RFC-0054 §7) returns it in three cases, and §12.1's row now names all three: an exact request whose only instances carry instance-wide approximate knobs; an approximate request that leaves out a knob the instance's plan applies to every sequence (RFC-0054 §7's option (c)); and a request on a plan that holds a knob bit this version cannot name, which it treats as instance-wide and names in hex. Each case is `InvalidRequest`, code 7, 400 in vendor shapes and 409 in the native API, and routing tries another candidate on it.

#### 12.2 After the first token

| Shape | In-stream error |
|---|---|
| OpenAI chat and completions | `data: {"error": {…}}`, then the stream closes without `[DONE]` |
| OpenAI Responses | `event: error`, then `response.failed` with `error` set |
| Anthropic | `event: error` with `{"type": "error", "error": {…}}` |
| Native | `event: error` with the full `Error` |

In every shape the in-stream error body carries the error `code` and, for retryable errors, `retry_after` in seconds, because the HTTP status and headers were committed with the first byte (§4.3). A sequence whose tail reports `finish = Error(code)` (RFC-0001 §10.4) ends this way. A cancelled sequence ends with the shape's normal finish and `syntropy.warnings` naming the cause, unless the client itself disconnected.

#### 12.3 Error codes (domain `Api`, never reused)

| Code | Name | Code | Name |
|---|---|---|---|
| 1 | `invalid_request` | 14 | `admission_queue_full` |
| 2 | `context_length_exceeded` | 15 | `rate_limited` |
| 3 | `unsupported_feature` | 16 | `invalid_api_key` |
| 4 | `unsupported_content_type` | 17 | `insufficient_scope` |
| 5 | `unsupported_tool` | 18 | reserved |
| 6 | `unsupported_endpoint` | 19 | `session_mismatch` |
| 7 | `fidelity_conflict` | 20 | `response_not_found` |
| 8 | `knobs_need_approximate` | 21 | `session_not_found` |
| 9 | reserved | 22 | `request_too_large` |
| 10 | `model_not_found` | 23 | `timeout` |
| 11 | `model_does_not_fit` | 24 | `instance_unavailable` |
| 12 | `placement_infeasible` | 25 | `no_conforming_instance` |
| 13 | `model_loading` | 26 | `constraint_invalid` |
| 27 | `constraint_unsupported` | 28 | `licence_acceptance_required` |
| 29 | reserved | 30 | `internal_error` |
| 31 | `debug_api_disabled` | 32 | `api_tls_or_keys_missing` (startup) |
| 33 | `token_id_out_of_range` | 34 | `local_auth_required` |
| 35 | `cluster_proposal_rejected` (the native error for `ConfigConflict` and `FeatureLevelTooLow`; RFC-0054's `Config` code 13 names the same condition) | | |

### 13. Native API

The native API is HTTP/JSON plus SSE under `/v1/syntropy/` (ADR-0503). Its request and response types are the Rust types of `syntropy-api`, and an OpenAPI 3.1 document generated from them is served at `/v1/syntropy/openapi.json` and published with each release. The CLI (RFC-0051) and dashboard use nothing else. Within `/v1/syntropy/`, changes are additive. Breaking changes happen only in a product major release after 1.0, under RFC-0062 §10.1. They move to `/v2/syntropy/`, and both versions are served for at least one minor release.

#### 13.1 Generation

`POST /v1/syntropy/generate`:

```json
{
  "model": "qwen3-235b-a22b:Q4_K_M",
  "messages": [{"role": "user", "content": "…"}],     // or "prompt": "…", or "tokens": [1, 2, 3]
  "tools": [], "tool_choice": "auto",
  "constraint": {"json_schema": {…}, "strict": true},  // or regex, grammar, choice
  "reasoning": {"effort": "medium"},                    // or {"budget_tokens": 2048} or {"enabled": false}
  "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "min_p": 0.0, "seed": 7,
               "stop": ["\n\n"], "max_tokens": 1024, "logprobs": 5},
  "n": 1,
  "fidelity": "exact",                                  // or "approximate", optionally with "approximate": ["topk_reduce"]
  "session": "ses_…", "persist": true,
  "priority": "interactive",
  "deterministic": false, "speculative": "auto",
  "trace": false,
  "stream": true, "stream_coalesce_ms": 0,
  "return_tokens": false,                               // include token IDs in deltas
  "raw_special_tokens": false                           // true: control-token strings in content become
                                                        // control tokens (§6.4); default literal
}
```

Native messages may carry `cache_control` like Anthropic content blocks, marking the prefix up to them as shareable (§9.5). Every token ID in `tokens` and `sampling.logit_bias`, and every stop token ID, is range-checked (§2.3).

The non-streaming response contains `choices[]` (text, reasoning, tool calls, token IDs when asked, logprobs, stop reason), `usage`, and every `SyntropyMeta` field (§10) as first-class fields, plus `trace_id` when the request was traced. A streaming response can be re-attached within the detach grace at `GET /v1/syntropy/generate/{sequence}/events` with `Last-Event-ID` (§5.5, M3).

#### 13.2 Models

| Endpoint | Behaviour |
|---|---|
| `GET /v1/syntropy/models` | registered models with state, variants, which nodes hold how many bytes, licence identifier, licence class (`permissive` or `custom`), gated flag, this node's acceptance state, `modalities` and the "text only in Syntropy" flag (§3.4); `?catalogue=true` adds catalogue entries not yet pulled (RFC-0051 §3.3) |
| `GET /v1/syntropy/models/{ref}` | manifest summary: family, hyperparameters, census class totals, KV bytes per token, speculative support, siblings, `vocab_limit` (§2.3), and `tool_markers` and `reasoning_markers` (`special_tokens` or `text`, §6.2) |
| `POST /v1/syntropy/models/{ref}/licence/accept` | records, on this node only, that its operator accepted a custom-licence model's terms after seeing the summary and link (`admin` scope; RFC-0051 §3.2). It never touches the Hub, and gated models still need the user's own Hub acceptance |
| `GET /v1/syntropy/models/{ref}/tensors` | the census: name, class, group, layer, quant scheme, bytes |
| `POST /v1/syntropy/models/pull` | `{ref, variant?, nodes?: "planned" \| "all" \| [selectors]}`: register, then fetch shards. `planned` (default) fetches only what a preview plan assigns to each node (ADR-0016). Returns a pull ID; progress on `GET …/pulls/{id}` and on the events stream |
| `DELETE /v1/syntropy/models/{ref}` | unload instances, drop the model's chunks and packs from every store (`?nodes=` to limit) |
| `POST /v1/syntropy/models/{ref}/estimate` | the fit estimate: a preview (§13.3) without registering anything. Works from Hub headers (range reads) or a catalogue census summary, so it runs before any download |

Licence handling (ADR-0018 decision 7): pulls and estimates return the model card's licence identifier and class. A pull fails with `licence_acceptance_required` until the terms are accepted: for a gated model, by the user on the Hub with the node's own token; for a custom-licence model, by an explicit local acceptance recorded per model and node. Acceptance is never automated.

#### 13.3 Placement previews, instances and plan explanations

| Endpoint | Behaviour |
|---|---|
| `POST /v1/syntropy/placement/preview` | `{model, policy overrides?, candidates?: 1..3}` → up to 3 candidate plans from `solver::preview`, each with predicted decode tok/s, TTFT at 1K/4K/32K prompt, per-node and per-tier memory deltas, download bytes per node with an ETA, the strategy (make-before-break or break-before-make), the assumptions (`Prior` inputs) and a fit verdict (`fits`, `degraded`, `does_not_fit` with the binding constraint). Nothing is committed. Rate-limited to one concurrent preview per client, since a MILP solve can take up to the 2 s time limit (SC-31) |
| `GET /v1/syntropy/instances` | instances with model, policy, plan epoch, ring (stages, nodes, layer ranges, modes), runner states per node, routable flag, live sequences, degradations |
| `POST /v1/syntropy/instances` | `EnsureInstance` with a policy; returns at once with the instance ID; progress on the events stream |
| `DELETE /v1/syntropy/instances/{id}` | `UnloadInstance` (drains first) |
| `GET /v1/syntropy/instances/{id}/plan` | the committed `PlacementPlan` |
| `GET /v1/syntropy/instances/{id}/plan/explain` | the plan explanation from RFC-0011: per stage, why its layer range and node; per tensor group, home tier, compute site and the reason (binding constraint, cost term and its predicted delta if moved, or a pin) |
| `POST /v1/syntropy/instances/{id}/replan` | `RequestReplan { reason: User }` |

Shape of one memory delta, which the dashboard draws (RFC-0051 §5.4). The values are illustrative, not measurements:

```json
{ "node": "pc", "tier": "Vram", "device": "cuda:0",
  "before": {"used": 3221225472, "usable": 23622320128},
  "after":  {"used": 22548578304, "usable": 23622320128},
  "by_purpose": {"context_reserve": 805306368, "scratch": 1073741824, "dense_hot": 10415243264,
                 "kv": 4294967296, "expert_cache": 5153960755, "headroom_kept": 1288490188} }
```

#### 13.4 Cluster and nodes

| Endpoint | Behaviour |
|---|---|
| `GET /v1/syntropy/cluster` | cluster name, members with state (alive, suspected, failed, left), coordinator and lease term, event-log index, protocol versions, pending admissions |
| `GET /v1/syntropy/cluster/graph` | the `ResourceGraph`: nodes, devices, tiers, links with measured α/β/p99 and chosen transport |
| `GET /v1/syntropy/nodes/{id}` | one node: hardware, support tier, backends, capability profile, ledger (who holds what), store usage, runner list |
| `GET /v1/syntropy/nodes/{id}/telemetry` | live tier usage, residency digests, I/O rate, NVMe temperature, memory-pressure level; fetched from that node over the control connection's node-query channel when `id` is not this node (proposed amendment A4) |
| `POST /v1/syntropy/cluster/invites` | `{ttl, uses, scope, own_device: bool}` → invite token and a join command (format: RFC-0030, RFC-0034). `own_device` answers "your own device, or someone else's?"; it defaults to false, which makes the invite ingress-only (`MEMBER \| INGRESS`) unless `scope` says otherwise, and the admission notice shows which was chosen |
| `GET`, `DELETE /v1/syntropy/cluster/invites[/{id}]` | list and revoke |
| `POST /v1/syntropy/cluster/join` | `{token}` on the joining node |
| `GET /v1/syntropy/cluster/pending`; `POST …/pending/{id}/approve`, `…/deny` | pending LAN joins under `cluster.lan_admission = confirm` (ADR-0302): each shows the joining node's name, hardware summary, power class (mains or battery) and the 6-digit short authentication string both devices display (RFC-0030 §6). `approve` takes `{ sas_choice, role: "stage_host" \| "client_only", coordinator: bool, keyholder: bool, off_lan_reach: bool }`; the role is chosen in the keyholder's approval, never by the joiner, and `client_only` admits `MEMBER \| INGRESS`. The pending record says which of these the node is eligible for and the default of each (RFC-0051 §2.6): `coordinator` and `keyholder` are offered only for a wired, mains-powered desktop and default to false; `off_lan_reach` (recorded as `NodeInfo::off_lan_reach`) is asked only for a battery-powered node and defaults to true; `client_only` is the M2 default for a device the approver marks as not their own |
| `GET /v1/syntropy/cluster/keyholders` | keyholder count, which members hold the role, whether a recovery code exists, and the status line the dashboard and `cluster status` show (RFC-0051 §2.6) |
| `POST /v1/syntropy/cluster/recover`; `POST /v1/syntropy/cluster/refound` | keyholder recovery with the offline recovery code, which always rotates the cluster key, and the re-found flow for survivors of a lost keyholder (protocol: RFC-0034, DIST; `admin` scope, local socket or key) |
| `POST /v1/syntropy/cluster/pairing-window` | `{for, joins}`: opens a pairing window, as `syntropy cluster open` does (ADR-0302 decision 2) |
| `POST /v1/syntropy/cluster/leave`; `POST /v1/syntropy/nodes/{id}/revoke` | `Leave`, `Revoke` |

#### 13.5 Live streams for the dashboard

- `GET /v1/syntropy/events`: SSE of cluster events as they apply on this node (membership, plans committed, runner state changes, instance routable or unroutable, pulls, config changes, degradations, transport demotions), plus local notices (reload reports, update status). `Last-Event-ID` resumes from a log index within the retained tail.
- `GET /v1/syntropy/metrics/live`: SSE at 1 Hz with per-instance and per-node aggregates (decode tok/s, TTFT and TPOT percentiles, queue depth, expert-cache hit rate by layer, KV usage by tier, link α, predicted-versus-measured per cost term). The set and names are RFC-0053 §2's. Remote nodes' figures come from their telemetry digests over the node-query channel, fetched only while a client is subscribed.

#### 13.6 Debug surface

These endpoints exist so the quality RFCs can test and benchmark the signed release artefacts themselves rather than special builds (RFC-0060 §1.3, RFC-0061 §10.2). The three gated ones answer only on loopback listeners and the local socket, and only while their key is set (RFC-0054 §7.1). Otherwise they return 404.

| Endpoint | Gate | Request | Response |
|---|---|---|---|
| `GET /v1/syntropy/debug/state` | `dashboard` scope (RFC-0001 §13 names it); no `debug.*` key | `?instance=` optional | JSON: `ClusterState` rebuilt from the event log; each runner's `ResidencySummary`; each instance's current `PlacementPlan` with RFC-0011's explanation and `PredictedCost` per `CostTerm` (RFC-0053 §2.2); per stage, the plan's per-operation provider selection, as `{ op_class, provider, cert: CertLevel }` entries (`Certified` or `Derived`, RFC-0040 §6.2; RFC-0060 §6.4), from which RFC-0060's driver derives `ExecPath`; the resolved-config hash, BLAKE3 over the canonical `ResolvedConfig` (RFC-0054 §2.2), which RFC-0061 records as `BenchResult::config_hash` |
| `POST /v1/syntropy/parity` | `debug.parity_api` | NDJSON body, streamed. Line 1: `{ instance? \| model, policy?, teacher: { tokens } \| { corpus_ref }, parity: ParityTapSpec }`, with `ParityTapSpec`, `ForceMode` and `TapMode` as in RFC-0060 §5.5. With `TapMode::Gather`, each later line is `{ pos, top: [u32; K], tail: [u32; M] }`: the reference top-K and tail-sample ids for an upcoming position (RFC-0060 §5.3), which ingress forwards to the tail over the `Tap` channel | NDJSON stream: one `TapRow` (below) per scored position, then a summary with instance, plan epoch and fidelity. With `TapMode::FullLocal`, the first record instead names a shared-memory region of full F32 logits and gives its size; the driver must run on the tail's host |
| `GET /v1/syntropy/debug/telemetry` | `debug.telemetry_api` | `?instance=…&since=<unix ms>&until=<unix ms>&follow=bool&format=ndjson\|binary` | the per-step records of every stage of the instance for the window (RFC-0053 §2.5), gathered from other nodes over the node-query channel. NDJSON by default; `binary` streams the records' postcard encoding with a header naming the schema version |
| `POST /v1/syntropy/generate` with `bench: { ignore_eos: true, gen_tokens: N }` | `debug.bench_api` | as §13.1 | as §13.1. Exactly `N` tokens are generated: EOS, stop sets and stop strings are ignored. The response carries `bench` in `syntropy.warnings` |

*Note 2026-10-23 (PLC-E7-T01):* `/debug/state` also lists, per runner, each hot thread's `HotThreadReport` (`syntropy_exec::thread`, RFC-0012 §1.1 note): `kind` (`device` or `cpu`), `device`, `core`, `numa_node`, `numa_local`, `priority` (`nice`, `level`, `user_interactive` or `unchanged`), `unpinned` (why, when the thread has no core) and `spin_budget_ns`. The endpoint's task, PROD-E4-T01, serves it.

- **`TapRow`** mirrors one position of RFC-0031's tap block (RFC-0060 §5.4–§5.5):
  ```rust
  pub struct TapRow {
      pub pos: u32,
      pub argmax: u32,            // ties to the lower id
      pub margin: f32,            // top-1 minus top-2 logit
      pub lse: f64,               // logsumexp over the full vocabulary, computed in F64
      pub lse_complement: f64,    // logsumexp over V∖T, where T is the top-K ids sent, computed in F64
      pub values: TapValues,      // Gather { top, tail }: the candidate's F32 logits at the K top-K
                                  // ids and the M tail-sample ids, in the order sent;
                                  // FullLocal { shm_offset }: the row's offset in the mapped region
  }
  ```
  *Revised 2026-10-22 (QUAL-E5-T19), a plan correction for the maintainers:* this record was named `LogitTap`, one per position. `LogitTap` is now RFC-0060 §5.5's per-entry block, `{ k, tail, rows: Vec<TapRow> }`, which RFC-0001 §5.8's `TokenEntry::tap` and RFC-0031 §6.5 already describe, so the endpoint streams one `TapRow` per scored position. Under `FullLocal`, a row's `shm_offset` is the row offset RFC-0012 §8.5 mentions.
  F64 values are written with round-trip precision. A non-finite value is written as the string `"nan"`, `"inf"` or `"-inf"` and is never clamped, so the harness can treat it as an error (RFC-0060 §5.4).
- **The parity endpoint drives a real instance.** It opens a sequence whose `SequenceOpen` carries the `ParityTapSpec` (RFC-0060 amendment A4). The tail replaces sampling with the teacher token, so every stage sees service-shaped traffic. Ingress applies no template, tokenizer or detokenizer: the teacher stream is token IDs.
- **Cost of the drain.** The drain runs on the daemon's `ctl` runtime and reads records the telemetry ring already carries. RFC-0061's A/A rule requires that decode rate with the drain active stays within 2% of the rate without it.
- **Nothing is added to the shipped surface.** The gated endpoints are compiled into every build, because the release artefacts are what the quality gates must test. They stay unreachable unless an operator on the machine sets a key.

### 14. Embeddings

`POST /v1/embeddings` (M2) takes `model`, `input` (a string, an array of strings, a token array or an array of token arrays), `encoding_format` (`float` or `base64`) and `dimensions`. Token arrays are range-checked (§2.3).

- **Which policy.** Models whose manifest declares an embedding output: a pooling mode (`last_token`, `mean` or `cls`) and an optional normalisation, read from the model's metadata by the family module (RFC-0043). Decoder-only embedding models built on v1 families (Qwen-class embedding models, for example) are covered. One encoder-only embedding family, `bert`, also enters v1 at M2 with this endpoint (OQ-MF-15, answered): single node only, with no pipeline and no recovery paths, on the CPU and CUDA (RFC-0043). A request for any other encoder-only family fails with `unsupported_feature`. A generative model without a declared pooling mode is rejected rather than given an improvised one.
- **Execution.** Each input is a prefill-only sequence: `SequenceOpen` with `limits.max_tokens = 0` and an embedding flag. The tail applies pooling (and normalisation) instead of the LM head and returns the vector to ingress. RFC-0001 has no frame for that, so this RFC proposes an `EmbeddingResult { seq, dims, vector: PayloadRef }` message on the tail→ingress channel (amendment A5). Vectors are F32, at most a few tens of KiB.
- **`dimensions`** truncates and renormalises only for models whose manifest declares nested (Matryoshka-style) dimensions; otherwise a value below the native size fails with `unsupported_feature`.
- **Batching.** Inputs of one request run as concurrent sequences on one instance, under the instance's batching limits. An embedding model in latency mode still serves many inputs per request, because prefill-only sequences batch freely.
- **Fidelity.** Embeddings carry the same `syntropy.fidelity` object; there are no embedding-specific approximate knobs in v1.
- `usage` reports `prompt_tokens` and `total_tokens`.

## Performance model and budgets

The API sits beside the critical path, not on it. The tail sends each token to the head and to ingress at the same time (RFC-0001 §6.2), so per-token work at ingress delays only delivery to the client, never the next decode step. Ingress work before `SequenceOpen` adds directly to TTFT.

| Item | Budget [target] | Derivation |
|---|---|---|
| Ingress work before `SequenceOpen`: auth, parse, model resolution, policy, routing, admission. Excludes tokenization, template rendering and first-time constraint compiles | ≤ 1 ms p99 | ≤ 10% of the shortest TTFT on Tier 1 hardware. Estimate for that TTFT: a 100-token prompt on a 7B Q4 model on an RTX 4090 is ≈ 100 × 2 × 7·10⁹ FLOP / 165 TFLOP/s ≈ 8.5 ms of prefill, plus one ≈ 4.7 ms decode step (3.79 GB / 800 GB/s), ≈ 13 ms [estimate, cost model §3.5 and SC-2 inputs] |
| Tokenization and template rendering | ≤ 5% of the plan's predicted prefill time for the same prompt | keeps them invisible next to prefill (SC-18's 1.15 factor has to cover all overheads) |
| Per-token ingress work: detokenize, stop matching, parsers, encoding, socket write | ≤ 20 µs p99 per token | off the critical path, so the budget is set by capacity: at 20 µs one `ctl` core sustains 50,000 tokens/s, which is ≈ 280 streams at the ≈ 175 tok/s a 7B Q4 model can reach on an RTX 4090 (1 / (4.7 ms + 0.95 ms SC-2 `t_fixed`)) [estimate] |
| Delivery delay added by ingress: `TokenFrame` arrival on `local_tokens` or the sequence channel to bytes written to the socket | ≤ 50 µs p99 | matches ADR-0015's revisit trigger for the runner → API path |
| Grammar mask for the next position, at the tail | ≤ min(0.5 × predicted ring step, 1 ms) per token | the mask is computed while the ring runs the next step (§7.3), so it stays hidden when it takes under half the step. The library's authors report ≈ 50 µs of CPU time per token at a 128K vocabulary [reported by library docs]; RFC-0061 measures it on the v1 tokenizers |
| First compile of a JSON Schema constraint | ≤ 50 ms p99 for schemas up to 50 properties; cached afterwards | chosen budget: invisible next to an agent turn's TTFT |
| Ingress memory per live sequence | 4 bytes per transcript token plus the detokenizer and parser state | a 128K-token transcript is 512 KiB [estimate] |
| Concurrent streams per daemon | ≥ 1,000 with the default `ctl` threads | chosen budget: far above household use (vision §3.3), so the API is never the limit |

None of these are measurements. RFC-0061 adds a benchmark per row, run against a fake runner from `syntropy-testkit` so that only API cost is measured.

## Alternatives considered

| Alternative | Why not |
|---|---|
| gRPC for the native API | browsers cannot call it without a proxy, so the dashboard would need a second API; it needs a protobuf toolchain in every client language; HTTP/2 trailers fail through some proxies. The token rates involved (hundreds per second) do not need a binary encoding (ADR-0503) |
| WebSockets for streaming | every target client already speaks SSE; SSE works through HTTP proxies and is one-way, which is all streaming needs |
| Tokenize at the head stage | RFC-0001 tokenizes at ingress; it also lets ingress check context length and compile constraints before routing, and keeps the tokenizer out of the runner |
| Apply grammar masks at ingress | ingress does not choose tokens; a mask round trip per token would add a hop to every step |
| Write our own constraint engine | months of work to reach what an existing MIT-licensed Rust library already does at tens of microseconds per token. Kept behind a trait so we can replace it (ADR-0504) |
| Download unknown models automatically on request | a typo could start a 400 GB download; off by default, opt-in through `api.auto_pull` |
| 503 for a model that does not fit | clients and SDKs retry 503, which cannot help; 400 with a fit report tells the user what to change |
| Separate ports or path prefixes for OpenAI and Anthropic | both SDKs default to `/v1/…`; a header check on the one shared path (`/v1/models`) is enough |
| Node-local API keys only | remote clients would need a key per node, which conflicts with "any node can be the ingress"; cluster-wide keys exist as keyholder-signed events (RFC-0034 §8.4, §11.3) |
| Store Responses state in the event log | request content must never enter the control plane (ADR-0009); stored per ingress node instead |
| Mean-pool any generative model for embeddings | produces vectors the model was not trained to make; only models that declare pooling are served |
| Ship the common local-runner REST shape in v1 | widens the compatibility surface to test before the core shapes are solid; an opt-in M4 module (ADR-0502) |

## Testing and validation

| Claim | Test |
|---|---|
| Unmodified SDKs work (G1) | conformance suite run with the official OpenAI and Anthropic SDKs (Python and TypeScript, pinned versions updated by Renovate): chat, streaming, tools (single, parallel, forced, strict), structured output, reasoning, logprobs, embeddings, model listing, error handling and SDK retries. The daemon runs against a `syntropy-testkit` fake runner that emits scripted token streams, so the suite needs no GPU and runs on every PR touching `api` or `text` |
| Wire formats are exact | byte-exact golden SSE transcripts per shape and event type, reviewed when they change |
| Detokenization is lossless | property test: for random token sequences over each v1 tokenizer, including byte-fallback tokens, the concatenation of released deltas equals the full decode, and no delta contains U+FFFD unless the full decode does |
| Stop strings never leak | property test against a naive search over the full text: same match position; held-back bytes never exceed `max_stop_len − 1`; nothing after a match is released |
| Tool-call parsing is chunking-independent | for each format of §6.2, random chunkings of recorded model outputs give the same calls as one-shot parsing; malformed calls come back as text with the warning |
| Constrained output always validates | for schemas from a random generator plus a corpus of real tool schemas, masked random sampling (fake logits) produces JSON that a JSON Schema validator accepts, every time; with speculation on, temperature-0 outputs equal non-speculative outputs, and at temperature > 0 a statistical test finds no distribution difference |
| Exact stays exact | property test over random (cluster config, instance policy, request) triples: no response carries a knob in `enabled` or `applied` unless all three opted in; the enabled set is always the intersection; an exact request whose only instances carry instance-wide knobs always gets `fidelity_conflict` (400 in vendor shapes, 409 native); an approximate request that resolves to exact always carries `approximate_not_available` |
| Routing fails fast and never loops | fake-runner cluster: an exact request whose only instance and whose table-built policy both carry an instance-wide knob gets `fidelity_conflict` with no `EnsureInstance` submitted and no wait; a request longer than every routable instance's planned context gets `context_length_exceeded` naming the largest planned limit, and automatic placement refuses one longer than the preview's planned context before loading anything; with one instance's node `Restarting`, requests go to another instance, or wait and then route once `NodeAlive` arrives |
| The debug surface is closed by default | in every build profile, each gated endpoint returns 404 until its `debug.*` key is set, and still returns 404 to non-loopback peers with any key; `bench` controls without `debug.bench_api` fail with `debug_api_disabled`; parity taps through the API match the testkit reference driver on fixture models (RFC-0060), including `lse_complement` and the tail-sample logits, and a planted NaN arrives as `"nan"`; the drain's A/A cost stays under 2% (RFC-0061) |
| Positions past the certified context are labelled | fake-runner fixture with a `FamilyCert` certified context of C: a request whose prompt plus output stays within C carries no `beyond_certified_from`; one whose prompt is longer than C carries `beyond_certified_from = C` from the first chunk; one that crosses C while generating carries it in the final object; `fidelity` stays `exact` in all three, and the CLI and dashboard render "beyond the certified context" and never "uncertified" for it |
| Certification is labelled | fault-injected plans on the fake runner: a derived-certificate op yields `certification: derived` and the `Syntropy-Certification` header; a plan whose certified run-time values exclude prefix reuse or a session resume lists the matching `refused` entries and serves without them; a stage whose tier manager reports `KV_SPILL_UNCERTIFIED` for `Nvme` yields `KvSpill` with `Substituted` or `NotSpilled` on every request, and `UncertifiedUsed` only on approximate ones; a CI-test-signed manifest is refused unless `debug.parity_api` or `debug.bench_api` is set and every API listener is loopback, and is labelled `ci_test` when accepted (RFC-0054 §7.1) |
| Browser-borne attacks fail | integration tests: DNS-rebinding `Host`, cross-origin `POST` with and without preflight, CSRF on dashboard mutations, public bind without TLS refused, auth back-off engages, key comparison is constant-time (statistical timing test) |
| LAN TLS works and stays constrained | integration tests: a fresh node bound to a LAN address serves TLS from its local CA with no configuration; `syntropy trust` installs and removes the CA on Linux, macOS and Windows test hosts (ledger round trip); a leaf for a public DNS name or public IP cannot be issued, because the CA's name constraints reject it; overlay-only binds serve plain HTTP and still demand a key; `api.allow_plaintext_lan` without the opt-in fails validation, and with it the dashboard banner and doctor SEC-005 appear |
| Automatic placement behaves | multi-node simulation: 50 concurrent first requests for one model produce one `EnsureInstance`; LRU eviction of idle instances; a model that cannot fit returns 400 with a fit report and siblings; for a request without `Syntropy-Progress`, `api.load_wait` expiry returns 503 (OpenAI) or 529 (Anthropic) with `Retry-After` and the load continues; with progress opted in, the same expiry arrives as an in-stream error carrying `model_loading` and `retry_after`; a per-sequence-knob instance serves an exact request without a new `EnsureInstance` |
| Clients behave as vendors' SDKs expect | SDK attempt counting per error code; an SDK with a dummy key against default loopback succeeds; a wildcard LAN bind is reachable by `.local` name and by IP over TLS; multi-turn tool use with model-native call IDs across two ingress nodes and a restart renders the right IDs |
| Recovery is invisible to clients | chaos test (RFC-0060): kill a mid-ring stage during a stream; the client's stream continues with no duplicated or missing text, and the final metadata lists the recovery |
| Budgets hold | benchmarks in RFC-0061: pre-`SequenceOpen` latency, per-token ingress cost, delivery delay, 1,000 concurrent streams, mask time per token on each v1 tokenizer |
| Parsers are robust | `cargo-fuzz` on every front end's body parser, the SSE encoder, every tool-call and reasoning parser, and template rendering with hostile templates (`minijinja` fuel and recursion limits) |
| Content cannot forge control tokens | fuzz and conformance cases per v1 family: every control string of the family's template and tool format, injected into user, system, tool-result, tool-description and document content in every shape, leaves no special-token ID inside a content span of the token stream, opens no role, and yields no parsed tool call or reasoning section; a model output that quotes a tool-call marker as text yields text; with native `raw_special_tokens: true` the same strings do become control tokens (§6.4) |
| Out-of-range token IDs never reach a device | API fuzz cases with IDs at `vocab_limit`, above it and at `u32::MAX` in native `tokens`, parity teacher tokens, `logit_bias` keys and stop IDs: every one gets 400 `token_id_out_of_range` before `SequenceOpen` (§2.3) |
| Loopback is split by scope | a non-browser client on loopback with a forged `Host` and no `Origin` cannot create an invite, approve a join, create a key, change configuration, use a session route or read a stored response, and gets 401 `local_auth_required`; the same calls succeed over the local socket; chat and model-list calls without a key still succeed; a one-time login token works once, expires after `api.dashboard_login_ttl`, and never appears in a log; `api.cors_origins` never admits an `admin` route |
| The prefix cache leaks nothing across principals | two API keys on one instance: the second key's identical prompt reports `cached_tokens = 0` and prefills in full, while a prompt that marks the same system prefix with `cache_control` reports exactly that prefix, rounded down to whole 64-token pages; two loopback clients without keys still share hits; a session created under one key is `session_not_found` under another (§9.5) |
| A dropped link does not lose a generation | fake-runner test: a native stream and a stored Responses stream are disconnected mid-generation and re-attached within `api.detach_grace` by `Last-Event-ID` and `starting_after`, with no repeated or missing text; a re-attach after the window gets 404 and the sequence was cancelled; a Chat Completions disconnect cancels at once; another principal cannot re-attach (§5.5) |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | `POST /v1/chat/completions` (messages, `stream`, `max_tokens`, `temperature`, `top_p`, `stop`, `seed`), `GET /v1/models`, `/healthz`, `/readyz`, `/v1/syntropy/version`, the local socket; template rendering into template and content spans with content tokenized literally (§6.4); the token-ID range check (§2.3); incremental detokenizer and stop matching; error mapping for the classes M0 can produce |
| M1 | full Chat Completions (tools, `tool_choice`, reasoning, logprobs, `n`, `response_format` including `json_schema`); Anthropic Messages and `count_tokens`; stateless Responses; legacy completions with FIM; the constraint engine; reasoning and tool-call parsers for the M1 families, on token IDs where the markers are special tokens; native `generate` with `raw_special_tokens`, models (with licence classes, local acceptance and the text-only label), pulls, estimates, one-node previews, instances, plan explanations, config, events and live metrics; automatic placement on first request; bind policy with the local CA (narrow name constraints, user store), overlay rule and the plain-HTTP-on-LAN opt-in, Host and Origin checks, the loopback scope split and the dashboard's one-time login URL, node API keys, `syntropy trust`; per-principal prefix-cache scope for node-local keys; fidelity labels for exact mode |
| M2 | routing across nodes; cluster, node and graph endpoints, with the pairing approval's role and grant choices, keyholder status, recovery and re-found routes; node-query telemetry for remote nodes; cluster-wide API keys (RFC-0034 §8.4) and per-principal prefix scope for them; `trust --cluster`; embeddings, with the M2 embedding family; native session create and the `Syntropy-Session` binding; multi-node previews |
| M3 | stored Responses with `previous_response_id`, `background` and the retrieval routes, on the ingress node only; the detach grace and re-attach (§5.5); the rest of the native sessions API; `cache_control` and `prompt_cache_key` handling; invites (with the own-or-someone-else's question) and admissions endpoints; recovery events; full approximate-mode labelling with every RFC-0042 knob; `deterministic`; the opt-in local-runner compatibility module |
| M4 | the published, frozen OpenAPI document for `/v1/syntropy`; SDK conformance as a release gate on Tier 1 platforms; the reproducibility fingerprint in `SyntropyMeta` and the opt-in provenance record (§10) |

Tool calling, Anthropic Messages, structured output, stateless Responses and FIM land in M1, ahead of the vision's M3 checklist, because single-node agent use (scenario S4) is an M1 workload, a Responses-only coding agent cannot connect without them, and they are cheap once the pipeline exists. The vision's M3 checkbox is met early, not changed.

## Open questions

1. **Client tolerance of extension fields.** Official SDKs accept unknown fields, but some third-party clients validate strictly. Spike, run twice (PROD-E2-T19, then PROD-E2-T12): once against the M0 chat endpoint for the base shape, then against the full M1 surface. Test by category (chat front ends, coding agents including a Responses-only agent CLI, two FIM autocomplete plugins, IDE plugins, typed SDKs) with `api.extensions` at its default, and record which ones break. The fallbacks (`headers`, `reasoning_field = "inline"` or `"omit"`) exist, but the default should suit the majority: keep the body object only if no top-ten client breaks.
2. **Thinking signatures.** Merged into question 1's run (OQ-PRD-09): the same client set, with one extra assertion on whether any Anthropic-shape client checks `signature` beyond its presence.
3. **Where stored Responses live (answered, OQ-PRD-10).** On the ingress node only, with no copy beside the session KV on the holders, and listed in RFC-0034 §13's inventory (§9.2). After ingress loss a continuation gets `response_not_found` and the client resends its input. Revisit a copy only for holders run by the same user, and only if the session key can be unwrapped by that user's other keyholders.
4. **Mask cost at large vocabularies.** Several v1 families have vocabularies near or above 200K tokens. Benchmark the constraint engine on each v1 tokenizer with deep schemas (RFC-0061). If the per-token mask exceeds half a ring step on fast single-GPU plans, give the tail sampler two cores for masks.
5. **Predicted outputs as drafts.** Moved to the post-1.0 backlog (OQ-PRD-12): few clients send `prediction`, and using it needs a scheduler change (RFC-0012).
6. **Encoder-only embedding families (answered, OQ-MF-15).** One, `bert`, enters v1 at M2 with the embeddings endpoint, single node only, on the CPU and CUDA (§14; RFC-0043).
7. **Default port (answered).** See RFC-0054 open question 2.
8. **One CA per node or per cluster (answered).** One CA per node, never a shared CA key. `syntropy trust --cluster <node>` installs every member's CA in one step (§11.1.1).

## Proposed amendments to RFC-0001

- **A1. §6.1 step 3.** Before holding a request for `InstanceRoutable`, ingress runs a local preview. If no feasible plan exists even after evicting idle instances, it fails at once with `model_does_not_fit` (HTTP 400 and a fit report) rather than waiting for the deadline. Holding a request for a model that can never fit only delays the answer.
- **A2. §5.10 `EnsureInstance`.** Add `evictable: SmallVec<[InstanceId; 4]>`: idle instances the requester's preview showed must be unloaded to make room. The coordinator unloads them and commits the new plan in one decision. Without it, two ingress nodes evicting for different models can race and unload more than needed.
- **A3. §5.8 `SamplingParams`.** Carry constraints by reference: `constraint: Option<GrammarRef>` (BLAKE3 of the compiled grammar) plus `grammar_inline: Option<Bytes>` on the first `SequenceOpen` that uses a given hash, so tails cache compiled grammars; and `reasoning_budget: Option<u32>` with the family's end-of-reasoning token sequence, so the tail can force the marker (§8.3).
- **A4. §2.2 planes.** Add a request/response *node-query channel* on the control connection, separate from the event log, for low-rate reads that one daemon makes of another: telemetry snapshots for the dashboard, stored-response lookups (§9.2), cluster-wide doctor runs and debug-dump collection (RFC-0053). It carries no per-token traffic, and nothing on it is logged as an event. RFC-0030 or RFC-0031 would own the message set.
- **A5. §5.8.** Add an `EmbeddingResult { seq, dims, vector: PayloadRef }` message on the tail → ingress channel and an `embed: Option<PoolingMode>` field in `SequenceLimits`, for prefill-only embedding sequences (§14).
- **A6. §10.4.** Name the extension object `syntropy` in every shape and the headers `Syntropy-Fidelity`, `Syntropy-Approximate` and `Syntropy-Warning`, as fixed in §10.
- **A7. §5.8 `SequenceOpen`.** Add `prefix_scope: PrefixScope { principal: Option<[u8; 16]>, shared_until: Option<u32> }`: the principal's salt for the prefix key (none for `local`), and the token index up to which a `cache_control`-marked prefix is also written under the instance-wide scope (§9.5; RFC-0022 §7.1 derives `scope_salt` from it). Without it the head cannot scope prefixes per principal.
- **A8. §12, zone Z3.** Loopback is trusted for `inference` and read-only routes only. `admin`, `sessions`, stored-response retrieval and keyholder actions need the peer-credential-checked local socket or a key on loopback too (§11.1), because Z3 otherwise trusts every local process, not only the OS user Z1 trusts.
- **A9. §6.1 step 1.** The ingress applies the chat template with template and content spans kept apart, tokenizes content spans with special-token matching off (§6.4), and range-checks every token ID the request supplies against the model's `vocab_limit = min(embedding rows, LM-head rows, sampler table size)` before routing (§2.3). `vocab_limit` is derived from `HyperParams` and the census, so `ModelManifest` needs no new field.
