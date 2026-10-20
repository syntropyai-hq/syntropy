# ADR-0503: The native API is HTTP/JSON with Server-Sent Events; no gRPC in v1
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the product surface and operations area)
- Related: ADR-0009, ADR-0010, ADR-0015, ADR-0502; RFC-0050, RFC-0051, RFC-0053

## Context
Syntropy needs an API of its own beyond the vendor shapes. It must expose:
- placement previews and plan explanations;
- instances, sessions and cluster membership;
- invites, configuration and diagnostics;
- live events and metrics;
- generation with every Syntropy control (fidelity, sessions, priority, tracing, token IDs).

Three consumers depend on it:
- **the dashboard**, which is a web page served by the daemon;
- **the CLI**, which talks to the local daemon (RFC-0001 §3.1);
- **user scripts and integrations.**

Token rates per stream are at most a few hundred per second, and the control-plane objects are small JSON documents. Data-plane traffic between stages never uses this API (ADR-0010).

## Options considered
| Option | For | Against |
|---|---|---|
| A. HTTP/JSON + SSE, with an OpenAPI 3.1 document generated from the Rust types | browsers, curl and every language can call it; the same transport as the vendor shapes; SSE passes through ordinary HTTP proxies; one server stack | JSON is larger and slower to parse than protobuf (irrelevant at these rates); no built-in bidirectional streaming |
| B. gRPC (HTTP/2 + protobuf) | typed clients in many languages; bidirectional streams | browsers need gRPC-Web and a translating proxy, so the dashboard would need a second API; protobuf toolchain in every client; HTTP/2 trailers break through some proxies and on some platforms |
| C. A + gRPC side by side | best of both for power users | two APIs to version, document and test; every endpoint built twice |
| D. WebSocket JSON-RPC | one connection for requests and events | weaker tooling, no caching or standard semantics per method; harder to put behind proxies with auth |

## Decision
1. **Option A.** The native API is HTTP/1.1 and HTTP/2, JSON bodies, and Server-Sent Events for streams. It lives under `/v1/syntropy/`. NDJSON is available on generation streams with `?format=ndjson` for clients that prefer it.
2. **One source of truth for types.** Request and response types are Rust structs in `syntropy-api`. An OpenAPI 3.1 document is generated from them, served at `/v1/syntropy/openapi.json` and published per release. TypeScript types for the dashboard come from the same structs.
3. **The CLI and dashboard use only this API.** There are no private endpoints, so anything they do can be scripted.
4. **Versioning.** Changes within `/v1/syntropy/` are additive. Breaking changes happen only in a product major release after 1.0 (RFC-0062 §10.1). They move to `/v2/syntropy/`, and both versions are served for at least one minor release.
5. **Local transport.** The CLI reaches the daemon through a Unix-domain socket or named pipe with peer-credential checks, falling back to loopback HTTP.

## Consequences
- **Positive.**
  - One server stack, one auth path and one set of security checks for vendor shapes, native calls and the dashboard.
  - Users can explore the API with curl.
  - Generated docs and types never drift from the code.
- **Negative.**
  - No first-class typed clients in other languages. Users generate them from the OpenAPI document.
  - JSON parsing costs some CPU on telemetry streams. At 1 Hz for the dashboard this is negligible.
- **Work created.** RFC-0050 (endpoints), RFC-0051 (CLI and dashboard as API clients), RFC-0062 (OpenAPI publication and compatibility checks in CI).

## What would make us revisit
- An integration needs a high-rate bidirectional stream: for example interactive token-by-token steering at more than 1,000 messages per second per client, where SSE plus POST adds measurable latency. Add a WebSocket or gRPC endpoint for that use only.
- A significant ecosystem of third-party tools asks for gRPC clients. Generate a gRPC facade from the same types, as a separate module.
