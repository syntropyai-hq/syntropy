# ADR-0432: Proposed: the Hub's HTTPS client and the crate that links it

- Status: Proposed (2026-10-29), for the maintainers; answers OQ-MF-18
- Date: 2026-10-29
- Deciders: maintainers (drafted by the compute and models area)
- Related: RFC-0041 §2 (`RangeSource`, Hub sources), §7.2 (the origin client), §12 (tests); RFC-0001 §4.1 (crate table), §4.2 rules 3 and 6; ADR-0018 §3 (dependency policy); COMP-E10-T14, COMP-E12-T02

## Context

- RFC-0041 §2 reads Hub sources through `RangeSource` with HTTP range requests against commit-pinned resolve URLs. RFC-0041 §7.2 fixes the client's behaviour:
  - redirects to the storage CDN, with the `Range` header kept;
  - HTTP/2, with at most `hub.max_streams` requests in flight (default 16), adapted down on 429 and 503;
  - `Retry-After` honoured;
  - jittered retries up to `hub.max_retries` (default 8);
  - 416 or a length mismatch reported as `Integrity`.
- COMP-E10-T14 builds the Hub range source, and COMP-E12-T02's Hub client builds on it. Both wait on this choice (`plan/open-questions.md`, OQ-MF-18).
- `RangeSource::read_at` is **blocking** (`crates/manifest/src/range.rs`). `manifest`'s `FileSource` and its recorded-fixture `Replay` implement it.
- RFC-0001 §4.1's crate table gives `store` the Hub range reads. `store` is layer 2 and depends on `manifest`, `io` and `net`.
- RFC-0001 §4.2 rule 3 bars `tokio` from `io`. Rule 6 lets exactly one crate reach each external system listed in `xtask/src/check_deps/policy.rs`'s `SYSTEMS`, following edges through external packages. An entry marked `direct_only` counts a member's direct edges only; today only the OS random source's entry (`getrandom`) uses it. No HTTP client is listed.
- A new direct dependency needs a maintainer's approval (AGENTS.md); `cargo xtask ci dep-diff` lists them.
- What the lockfile holds on `main` at `3e65e2c5`:
  - `reqwest` 0.13.5, `hyper` 1.12.0, `hyper-util` 0.1.21, `hyper-rustls` 0.27.10, `h2` 0.4.20, `rustls` 0.23.46 and `rustls-platform-verifier` are in `Cargo.lock`, through `iroh` under `syntropy-net`.
  - The packages that depend on `reqwest` are `iroh`, `iroh-metrics`, `iroh-relay`, `n0-dns-resolver` and `tokio-rustls-acme`.
  - The resolved `reqwest` features are `json`, `stream` and `rustls-no-provider`.
  - `syntropy-api` and `syntropy-cli` use `hyper` directly. `net` builds its own TLS configs on `ring`'s provider, for its raw-public-key cluster transport (`crates/net/src/transport/tcp/tls.rs`); that setup is not a template for web TLS.

## Options considered

The linking crate:

| Option | For | Against |
|---|---|---|
| **(a) `store`** | RFC-0001 §4.1 already assigns it the Hub range reads; the Hub source implements `manifest`'s `RangeSource` beside `FileSource` and `Replay` | `store` gains an HTTP stack |
| (b) `net` | Already reaches these crates through `iroh` | `net` is the cluster transport, not the model store; RFC-0001 §4.1 would change |
| (c) `io` | Beside the OS I/O back ends | Rule 3 bars `tokio` from `io`, and every candidate but `ureq` brings it |

The client, and how it meets the blocking `read_at`:

| Option | For | Against |
|---|---|---|
| **`reqwest` with its `blocking` feature** | Matches the blocking trait with no bridge of ours: the blocking client runs its own runtime on a background thread. One shared client, called from several threads, multiplexes their requests over HTTP/2, so §7.2's in-flight limit holds per connection | The `blocking` feature may add packages; `dep-diff` shows them |
| `reqwest` async, behind a bridge in `store` | One async client for the Hub source and COMP-E12-T02's concurrent chunk reads | A bridge of our own: a runtime thread and channels, since `block_on` panics inside a runtime. Or change `read_at` to async, an RFC-0041 §2 edit that every source and reader follows |
| `ureq` with `rustls` | Blocking, so it matches the trait; small; reaches nothing else, so rule 6 can check it transitively | HTTP/1.1 only, against §7.2's HTTP/2: one connection per in-flight read; a new crate in the lockfile |
| `hyper` with `hyper-rustls` | Already in the lockfile and used by `api` and `cli` | Redirects, timeouts, pooling and the blocking bridge are ours to write |

## Decision (proposed)

**(a) `store`, with `reqwest`'s blocking client and `rustls`.** COMP-E12-T02 may move to the async client and a bridge if its concurrency needs it; that is its decision.

1. **Direct dependencies `store` gains**, all already in the lockfile: `reqwest` (features `blocking`, `http2`, `rustls-no-provider`, default features off, so no OpenSSL and no system TLS library is linked, ADR-0018 §3), `rustls` (for the client config on `ring`'s provider, passed to the client explicitly) and `rustls-platform-verifier` (so a Hub read trusts what the OS trusts). `dep-diff` reports these edges, and any package `blocking` adds, for the maintainers' approval.
2. **Rule 6.** `reqwest` joins `SYSTEMS` with `only_in: "store"` and `direct_only: true`. It must be `direct_only`, because `net` reaches `reqwest` transitively through `iroh`, which a transitive entry would refuse. COMP-E10-T14's criterion "`check-deps` shows the client linked by exactly one crate" then holds for direct dependencies, and should say "directly".
3. **What the source must do beyond the client**, from RFC-0041 §7.2. COMP-E10-T14 tests these against the local range-serving server RFC-0041 §12 names:
   - keep `Range` across the redirect to the CDN;
   - never send a Hub credential to another host on a redirect;
   - bound the redirect count;
   - re-resolve the commit-pinned URL when a signed CDN URL expires mid-download;
   - limit in-flight requests to `hub.max_streams` and adapt it down on 429 and 503, which `reqwest` does not do on its own;
   - follow §7.2's retry and `Integrity` rules.
4. **Proxies.** With default features off, `reqwest` ignores `HTTPS_PROXY`. Whether the Hub source honours the system proxy, through `reqwest`'s `system-proxy` feature or a `hub.proxy` key, is a question for the maintainers. This ADR does not settle it.

## Consequences

- Positive: no new crate for the client itself; the direct edges in decision 1 go to the maintainers through `dep-diff`.
- Positive: the blocking client fits `RangeSource` as it is, so no trait or RFC-0041 §2 change is needed.
- Negative: the blocking client keeps a runtime thread per client. One shared client per process keeps that to one.
- Negative: rule 6 can check only direct use of `reqwest`, since `iroh` uses it too, so an indirect use through another wrapper would not be caught.
- Negative: on Android, `rustls-platform-verifier` needs its Java-side component (`rustls-platform-verifier-android`, in the lockfile), which the Tier 3 app packaging must carry.
- Note: `traceability.md` shows this ADR as not yet cited by a task. That clears when a task row implements it after acceptance.

## If accepted, the follow-up edits

- RFC-0001 §4.2 rule 6 and §18: record the `reqwest` entry and why it is `direct_only`.
- `xtask/src/check_deps/policy.rs`: the `SYSTEMS` entry, and the doc comment that names only the OS random source as `direct_only`.
- COMP-E10-T14's criterion: "linked directly by exactly one crate".
- `plan/open-questions.md`: move OQ-MF-18 to Settled.

## What would make us revisit

- `iroh` drops `reqwest`, so keeping it adds the crate to the lockfile. Then `ureq` or `hyper` may cost less.
- COMP-E12-T02 needs async reads for its concurrency, and the bridge costs more than an async `RangeSource`.
- A Tier 3 target cannot build `rustls`'s crypto provider or the platform verifier.
