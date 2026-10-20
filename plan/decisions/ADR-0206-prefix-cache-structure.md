# ADR-0206: Prefix cache as a page-hash radix tree with token-granular tails, authoritative at the head

- Status: Accepted (2026-10-20; revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0022))
- Related: ADR-0010, ADR-0019, ADR-0205; RFC-0001 §5.8, §5.9, §6.2; RFC-0022 §7; RFC-0031; RFC-0032; RFC-0034

## Context

- Agent clients re-send long shared prefixes (system prompt, tool schemas, files), and full prompt re-reads are the largest open complaint among the surveyed engine's users (`tiered-memory-inference.md` §12.1). SC-20 requires that the tokens prefilled equal the tokens after the longest cached-prefix match, exactly.
- The surveyed datacenter stacks address prefixes with a radix tree over tokens or with chained prefix hashes over blocks (§12.2).
- Syntropy's twist is the pipeline: each stage holds KV only for its own layers, and **only the head (and stages holding row-lookup tables) knows the token IDs**. Downstream stages see activations. A prefix hit is usable only if every stage still holds its pages for that prefix.
- Stages evict under their own memory pressure. If each stage evicts independently, the head cannot know what the whole ring holds without asking every stage on every request, which adds a ring round trip to TTFT.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Token-level radix tree at every stage | Token-exact; standard | Downstream stages do not have the tokens |
| B. Flat hash of block-chained keys, page-granular, independent per stage | Simple; content-addressed | Not token-exact (SC-20); no structure for leaf eviction; stages diverge |
| C. Prefix cache only at the head, recomputing downstream KV | No consistency problem | Downstream stages would recompute their layers for the prefix: no saving where it matters |
| D. Query every stage before each prefill | Always consistent | One extra ring traversal per request on the TTFT path |
| **E. Chained BLAKE3 page keys computed at the head and stamped into frames; a radix tree of whole pages at every stage, plus up to 4 partial tails per node for token granularity; the head decides lookups and normal eviction; stages may move pages down tiers freely and drop only as a last resort, reporting the drop; a missing key at attach triggers a restart from the shortest held prefix** | Token-exact; no extra round trip in the common case; downstream stages need no tokens | A rare race costs one wasted chunk traversal; side-channel messages to encode and relay |

## Decision

**Option E**, as specified in RFC-0022 §7:

1. `k_b = BLAKE3(k_(b−1) ‖ tokens of page b)`, rooted in the model content hash and the KV-settings hash. *Revised 2026-10-20 (below):* the scope is per principal by default. Pages inside a client's `cache_control`-marked span are keyed without a principal and shared; every later page mixes in the principal's scope salt. `kv.prefix_scope = instance` is the explicit opt-in to one shared cache.
2. The head computes each page's key when it seals and stamps it into the side channel of the frame carrying the page's last position; every stage indexes its own copy of the page under that key.
3. Every stage keeps a radix tree of whole pages. Partial pages left at release become tails, at most 4 per node, matched by token comparison. Hybrid models match only up to nodes with recurrent-state checkpoints, which are taken at fixed positions every stage can predict.
4. The head is authoritative. It runs lookups and normal eviction (lowest `(hits + 1) / age` leaves first) and broadcasts `PrefixEvict`. Stages move index pages down tiers on their own, drop only at the lowest tier or at Critical pressure, and report `PrefixDropped`. `attach` failures report `PrefixMiss`, and the head restarts prefill from the shortest held prefix.

## Consequences

Positive:
- SC-20 holds at token granularity.
- TTFT pays no consistency round trip in the common case.
- The same keys address saved sessions' restored pages, so a resumed session is immediately a prefix hit for its next turn.

Negative:
- A race between a downstream drop and a lookup costs one wasted prefill traversal of a chunk.
- Prefix pages occupy KV pools; the head's eviction is the only global bound, so a stage much shorter on memory than the others relies on `PrefixPressure` messages.
- Side-channel encoding (RFC-0031) and relaying (RFC-0032) work.

## What would make us revisit

- Measured `PrefixMiss` restarts exceed 1% of prefix hits on the agent benchmark: make downstream drops ask the head first, or give stages proportional index budgets set by the solver.
- Multi-tenant use grows beyond the household case: make `api_key` scope the default (RFC-0034). *Acted on 2026-10-20: per-principal scope is the default (below).*
- A cross-instance prefix store becomes desirable (RFC-0022 Open question 5).

## Revisions

- **Revised 2026-10-20.** Decision 1: the default prefix-cache scope is per principal (an API key, or the local principal for open loopback), and only prefixes a client marks with `cache_control` are shared across principals (RFC-0022 §7.1). Reason: `cached_tokens` makes a shared cache an exact oracle for other clients' prompts, page by page; a single-user setup has one principal and loses nothing. Node-local keys get it in M1, cluster-wide keys in M2. `cached_tokens` counts only hits in the caller's own scope or in a span the caller marked.
