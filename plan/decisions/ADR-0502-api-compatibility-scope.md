# ADR-0502: API compatibility scope for v1
- Status: Accepted (2026-10-20); revised 2026-10-20 (see Revisions)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the product surface and operations area)
- Related: ADR-0000 (D3, D5), ADR-0019; RFC-0050, RFC-0051, RFC-0060, RFC-0062

## Context
Syntropy serves no full chat client of its own (vision §3.3); from M2 the dashboard has one minimal chat view over the native API (RFC-0051 §5.12). Its value reaches people through clients they already use: chat front ends, coding agents, IDE plugins, scripts written against vendor SDKs. The research is blunt on this: API breadth is table stakes, and the surveyed cluster runtime that offered OpenAI Chat Completions, Responses, Anthropic Messages and the common local-runner REST shape let third-party tools work against a cluster unchanged (`multi-device-inference.md` §12.2, §15.11).

D5 allows naming API shapes such as the OpenAI and Anthropic APIs.

Every surface carries a cost. Each one needs:
- a front end and an encoder;
- an error mapping;
- streaming event formats;
- tool-call and reasoning conventions;
- conformance tests against that vendor's SDKs, kept current as the vendors evolve.

The vendor APIs also have large areas that make no sense for local inference: files, fine-tuning, batches, hosted tools, moderation and realtime audio.

## Options considered
| Option | For | Against |
|---|---|---|
| A. OpenAI Chat Completions only | the most widely supported shape; smallest test matrix | Anthropic-shape agents and Responses-only clients cannot connect; reasoning and tool conventions are weakest in this shape |
| B. OpenAI Chat Completions, Responses, Completions (legacy), Embeddings, Models; Anthropic Messages and `count_tokens`; a native API | covers chat front ends, coding agents (which increasingly use Responses or Messages), FIM code completion and embeddings | four generation front ends to keep conformant |
| C. B plus the common local-runner REST shape (`/api/chat`, `/api/generate`, `/api/tags` …) in v1 | some desktop clients speak only that shape | a fifth shape with its own streaming format (NDJSON) and model-management semantics that conflict with automatic placement; diverts M1–M3 effort |
| D. Everything in the vendor APIs, with stubs for what does not apply | maximal "compatibility" on paper | stubs that pretend to work are worse than clear 404s; untestable |

## Decision
**Option B for v1, with the local-runner shape as an opt-in M3 module (`api.compat.local_runner`).**

**When each shape lands** (revised 2026-10-20; see Revisions). Chat Completions from M0 and Messages from M1, as before. Stateless Responses (`store: false`, no stored records) and fill-in-the-middle legacy completions in M1, because they are front ends and encoders over the existing pipeline and a Responses-only coding agent cannot connect without them. Embeddings in M2. Stored Responses (`store: true`, `previous_response_id`, `background`) in M3, held on the ingress node only. The local-runner shape in M3.

1. **Supported surfaces.**
   - OpenAI: `/v1/chat/completions`, `/v1/responses` (with stored state and `previous_response_id`), `/v1/completions` (raw prompts and fill-in-the-middle), `/v1/embeddings`, `/v1/models`.
   - Anthropic: `/v1/messages` and `/v1/messages/count_tokens`.
   - Native: `/v1/syntropy/…`.
2. **Supported features across those surfaces.**
   - streaming;
   - function tools, including forced and strict calls;
   - structured output (`json_object`, JSON Schema);
   - reasoning content in each shape's own field;
   - logprobs where the shape has them;
   - `n > 1`, seeds and stop sequences.
3. **Explicitly unsupported**, each with a clear error in the request's own shape:
   - image, audio and file inputs: v1 is text-only. A model whose publisher ships it as multimodal is served for text and labelled "text only in Syntropy" in the picker, the model list and `GET /v1/models`. The content-part and modality fields that a later multimodal family needs are reserved in the API types now, and rejected with `unsupported_content_type` until a family supports them;
   - hosted tools (web search, file search, code interpreter, computer use, MCP tool types);
   - files, batches, fine-tuning, assistants, vector stores, moderation and realtime endpoints.
   Fields that do not apply locally (`user`, `metadata` on chat, `reasoning.summary`) are ignored and named in a `Syntropy-Warning` header.
4. **Extensions** use one top-level `syntropy` object and `Syntropy-*` headers, never fields that impersonate vendor fields. The exception is `reasoning_content` in Chat Completions: it is a widely used convention, and it can be renamed or omitted (`api.openai.reasoning_field`).
5. **Conformance is tested against the official SDKs** on every PR that touches the API, and it gates releases from M4.
6. **The local-runner shape** is implemented as a thin front end over the same pipeline in M3. Its model-management verbs map onto pulls and instances. It is off by default, and it never creates behaviour the native API lacks.

## Consequences
- **Positive.**
  - Chat front ends, coding agents, IDE completion plugins and RAG pipelines can all point at a Syntropy node.
  - Errors for unsupported features are explicit, so users know at once what does not work.
- **Negative.**
  - Four generation front ends and their SDK conformance suites to maintain, tracking two vendors' API changes.
  - Clients that speak only the local-runner shape wait until M3.
  - The Responses front end ships in two halves (stateless in M1, stored in M3), so its conformance suite grows in two steps.
- **Work created.** RFC-0050 (front ends, encoders, error maps), RFC-0060 (SDK conformance suite on the fake runner), RFC-0062 (SDK version pins under Renovate; release gate), RFC-0051 (docs listing supported fields per shape).

## What would make us revisit
- A vendor deprecates a shape (for example Chat Completions in favour of Responses) and major clients follow. Shift the conformance priority and keep the old shape until clients move.
- Image input for `qwen_hybrid` checkpoints with a vision tower, the first multimodal family RFC-0043 §2.4 plans after 1.0, is added. The reserved content parts then become in scope for that family.
- User reports show a large class of clients that speak only the local-runner shape. Pull the compatibility module forward from M3.
- Hosted-tool equivalents become a product goal (for example local web search). That would be a new ADR, because it makes Syntropy an agent runtime rather than an inference server.

## Revisions
- Revised 2026-10-20: stateless Responses and FIM completions move to M1, embeddings to M2, stored Responses stay in M3 on the ingress node only, and the local-runner module moves from M4 to M3. Coding agents that speak only Responses and IDE autocomplete plugins that speak only FIM could not connect until M3, and the stateless shapes are front ends and encoders over the existing pipeline.
- Revised 2026-10-20: decision 3: v1 is text-only. Multimodal models are labelled "text only in Syntropy", the API's content-part and modality hooks are reserved so a later family needs no new request shape, and the revisit trigger points at the first multimodal family RFC-0043 plans.
- Revised 2026-10-20: Context: Syntropy serves no full chat client, because the dashboard gains one minimal chat view in M2 over the native API. It adds no shape to this ADR's list.
