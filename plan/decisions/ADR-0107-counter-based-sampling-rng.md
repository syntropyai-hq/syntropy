# ADR-0107: Counter-based sampling RNG keyed by (request seed, sequence), indexed by a per-sequence draw counter

- Status: Accepted (revised 2026-10-19)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the placement and scheduling area)
- Related: ADR-0017 item 4, ADR-0019 item 8; RFC-0001 §5.9 (`SequenceId`), §7.5; RFC-0012 §8.2, §9.5; RFC-0032 §3.7; RFC-0033 §4.2, §4.5; RFC-0040 §7.1; RFC-0042 §4; RFC-0050 §2 (`n`)

## Context

Sampling happens at the tail stage. Three features depend on how random numbers are produced:
- **Recovery.** A sequence can move to a new tail after a re-plan or a stage loss, by KV migration, activation-history replay or re-prefill (RFC-0001 §7.5). A sequential RNG would have to ship its state with the sequence, and replay would have to re-draw the right number of times.
- **Speculative verification.** Rejection sampling draws an acceptance uniform per drafted position and a residual sample on rejection (ADR-0017 item 4). The number of draws per position depends on acceptance.
- **Deterministic mode.** ADR-0019 item 8 promises identical tokens for a fixed seed on the same build, plan epoch and devices.
- **Several choices per request.** An API request with `n > 1` (RFC-0050 §2) carries one seed and forks its choices from a shared prefix, so a key built only from the seed and the position draws identical numbers for every choice, and every choice comes out identical.
- **One definition across areas.** The first version of this ADR keyed draws by (seed, position, draw index), RFC-0032/0033 by (seed, `SequenceId`, `rng_counter`), and RFC-0040/0042 by (seed, sequence, position). The kernels and the Rust reference must implement exactly one.

## Options considered

| Option | For | Against |
|---|---|---|
| A. A sequential PRNG per sequence (state advanced per draw) | conventional | state must migrate; the stream position depends on how many draws speculation consumed, so speculative and plain runs diverge in their random streams |
| B. A fresh random seed per step | no state | not reproducible; deterministic mode impossible |
| C1. A counter-based generator keyed by (seed, position, draw index) | no state besides the seed | the n choices of one request are identical; tree and rejection draws per position are unbounded, so a fixed index per role cannot cover them |
| **C2. A counter-based generator (Philox4x32-10 class) keyed by (request seed, `SequenceId`), indexed by `rng_counter`, a per-sequence draw counter that increases by one per draw in a fixed order, reported on every `TokenEntry`** | each choice of an `n > 1` request is its own `SequenceId`, so choices differ; recovery resumes from the reported counter; the draw count may depend on acceptance; draws are generated in-kernel in the captured graph | the counter is state that ingress must keep and pass on recovery (RFC-0033 §4.5); the draw order within a step is a versioned contract |

## Decision

**Option C2**, binding across areas.
- **Key.** `(generated_seed, SequenceId)` when ingress generates the seed, and `(client_seed, choice_index)` when the request supplies one. Ingress always sets `SamplingParams::seed`: the client's seed, or 64 random bits it generates and records in the transcript. Each choice of an `n > 1` request is its own `SequenceId` (RFC-0001 §5.9), so the choices draw different numbers.
- **Counter.** `rng_counter: u64` per sequence starts at 0 and increases by exactly one per draw. Draw `c` of a sequence is `philox(key, c)`. The counter, not the token position, indexes draws, because speculative verification can consume several draws per position.
- **Draw order within a step**, per sequence: plain sampled decode takes one draw for the token. A chain verify window takes, for drafted positions i = 1, 2, … in order, one acceptance uniform per draft tried; then one residual draw at the first rejection, or one bonus draw if every draft was accepted. A tree verify window takes, at each accepted node in path order, one uniform per child tried in try order, then one residual draw if no child is accepted. Greedy sequences take no draws. Parity force modes (RFC-0012 §8.5) take none from this stream.
- **Where the state lives.** The step descriptor carries each sequence's key and its `rng_counter` at step start; `SampleSeeded` and `VerifyAccept` compute their draws in place and write the counter after the step into the result block. The tail reports it in every `TokenEntry` (RFC-0032 §3.7), ingress keeps the latest value with the transcript, and a recovered sequence resumes from it (RFC-0033 §4.5). The tail keeps no other RNG state.
- **Client-supplied seeds.** ADR-0019 item 8 asks for identical tokens for a fixed seed on the same build, epoch and devices, across requests, and a fresh `SequenceId` per request would break that. So when the request supplies a seed, the key's sequence component is the choice index (0 … n−1): choices stay distinct and a repeated request reproduces its tokens. When ingress generates the seed, the key is `(generated_seed, SequenceId)`. The counter is always `rng_counter`. RFC-0001 records it.
- **Versioning.** The generator, the key encoding and the draw order are versioned with the sampler; a change is breaking for deterministic mode and is recorded in the build.

## Consequences

- Positive: the n choices of a request differ; recovery needs one counter per sequence, which ingress already keeps; draw counts may depend on acceptance without breaking resumption; deterministic mode stays cheap.
- Negative: speculative and plain decoding no longer share draws at a position, so their sampled outputs agree in distribution, not token for token (ADR-0019 already promises only distributional agreement); ingress must persist `rng_counter` with the transcript; users who rely on unseeded randomness see seeds in the native API's metadata (they are not secret).
- Work created: RFC-0040's `SampleSeeded` and `VerifyAccept` implement exactly this key, counter and draw order on every provider, tested bit for bit against a Rust reference (RFC-0042 §4 aligns its table); RFC-0032/0033 keep `rng_counter` in `TokenEntry`, `ResumePoint` and `Reconcile`; tests for `n > 1` distinctness and for cross-request determinism of client-seeded requests. The choice-index key applies whenever the client supplies a seed, with or without a `deterministic` flag, and `rng_counter` is u64 in every type and on the wire (`TokenEntry`, `ResumePoint`, `Reconcile`, RFC-0031's field; the compute and distributed areas align to it).

## What would make us revisit

- A backend cannot generate the chosen generator's output bit-identically to the reference at acceptable cost. Then pick a simpler counter-based generator for all backends.
- Users need per-request statistical independence guarantees beyond what a 64-bit seed space provides.
