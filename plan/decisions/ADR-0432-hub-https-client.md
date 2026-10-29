# ADR-0432: Hub range reads use `reqwest` with `rustls`, linked by `store`

- Status: Proposed (2026-10-29), for the maintainers; answers OQ-MF-18
- Date: 2026-10-29
- Deciders: maintainers (drafted by the compute and models area)
- Related: RFC-0041 §2 (`RangeSource`, Hub sources), §7.2; RFC-0001 §4.1 (crate table), §4.2 rule 6; ADR-0018 §3 (dependency policy); COMP-E10-T14, COMP-E12-T02

## Context

- RFC-0041 §2 reads Hub sources through `RangeSource` with HTTP range requests against commit-pinned resolve URLs, following redirects. COMP-E10-T14 builds that source, and COMP-E12-T02's Hub client builds on it. Both wait on this choice (`plan/open-questions.md`, OQ-MF-18).
- RFC-0001 §4.1's crate table already gives `store` the Hub range reads. `store` is layer 2 and depends on `manifest`, `io` and `net`.
- RFC-0001 §4.2 rule 6 lets exactly one crate reach each external system's bindings (`xtask/src/check_deps/policy.rs`, `SYSTEMS`). It follows edges through external packages, unless a system is marked `direct_only`, which counts a member's direct edges only, as the `getrandom` entry does. No HTTP client is listed there today.
- A new direct dependency needs a maintainer's approval (AGENTS.md). `cargo xtask ci dep-diff` lists what a change adds.
- What the lockfile already holds, checked on `main` at `3e65e2c5`:
  - `reqwest` 0.13.5, `hyper` 1.12.0, `hyper-util` 0.1.21, `hyper-rustls` 0.27.10, `h2` 0.4.20 and `rustls` 0.23.46 are in `Cargo.lock`. They come through `iroh` (`iroh-relay`, `iroh-dns`, `n0-dns-resolver`) under `syntropy-net`.
  - The resolved `reqwest` features are `json`, `stream` and `rustls-no-provider`.
  - `syntropy-api` and `syntropy-cli` already use `hyper` directly, for the local API and its client.

## Options considered

The linking crate:

| Option | For | Against |
|---|---|---|
| **(a) `store`** | RFC-0001 §4.1 already assigns it the Hub range reads; the Hub source would implement `manifest`'s `RangeSource`, as `manifest`'s local-file and recorded-fixture sources do | `store` gains an async HTTP stack |
| (b) `net` | Beside the cluster transport, which already links `iroh` and through it these crates | `net` is the cluster's transport, not the model store's; RFC-0001 §4.1 would need revising, and `store` would reach Hub reads through `net` |
| (c) `io` | Beside the OS I/O back ends | `io` is a compute-path crate under rule 3 (no `tokio`), and an async HTTP client brings `tokio` |

The client:

| Option | For | Against |
|---|---|---|
| **`reqwest` with `rustls`** | HTTP/2 multiplexing for COMP-E12-T02's concurrent chunk reads; redirects, range headers and timeouts built in; already in the lockfile, so no new crate | Higher-level, with features to keep pinned |
| `ureq` with `rustls` | Blocking and small | HTTP/1.1 only, so one connection per concurrent read; a new crate in the lockfile |
| `hyper` with `hyper-rustls` | Already in the lockfile and used by `api` and `cli` | Redirects, timeouts and pooling written by hand, which is more code to review |

## Decision (proposed)

**(a) with `reqwest` and `rustls`.**

1. `store` is the one crate with a direct dependency on `reqwest`. `reqwest` joins rule 6's `SYSTEMS` table with `only_in: "store"` and `direct_only: true`. It must be `direct_only`: `net` already reaches `reqwest` transitively through `iroh` (`iroh-relay`, `iroh-dns`, `n0-dns-resolver`), which a transitive entry would refuse.
2. Features: `http2`, `rustls-no-provider` and `stream`, default features off, so no OpenSSL and no system TLS library is linked (ADR-0018 §3). The client gets a `rustls` config built on `ring`'s provider, passed explicitly, as `net` builds its own TLS configs (`crates/net/src/transport/tcp/tls.rs`).
3. Certificate verification uses `rustls-platform-verifier`, already in the lockfile, so a Hub read trusts what the OS trusts.
4. COMP-E10-T14's criterion "`check-deps` shows the client linked by exactly one crate" holds for direct dependencies, through the `direct_only` entry. Transitively, `net` reaches the same crates through `iroh`, so the criterion should say "directly".

## Consequences

- Positive: no new crate in the lockfile. `dep-diff` reports a new direct edge, `store` to `reqwest`, for the maintainers' approval, and no new package.
- Positive: HTTP/2 lets COMP-E12-T02 keep many range reads on one connection per host.
- Negative: `store` becomes an async crate for its Hub path.
- Negative: rule 6 can check only direct use of `reqwest`, since `iroh` uses it too. An indirect use through another HTTP wrapper would not be caught.

## What would make us revisit

- `iroh` drops `reqwest`, so keeping it means adding the crate to the lockfile. Then `hyper` with `hyper-rustls` may cost less.
- A Tier 3 target cannot build `rustls`'s crypto provider.
