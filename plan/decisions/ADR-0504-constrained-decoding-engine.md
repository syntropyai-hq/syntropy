# ADR-0504: Constrained decoding: compile at ingress, mask at the tail, with an existing grammar engine behind a trait
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the product surface and operations area)
- Related: ADR-0001, ADR-0010, ADR-0017, ADR-0018, ADR-0019; RFC-0001 (§3.2, §5.8, §6.4), RFC-0050 (§6, §7, §8), RFC-0012, RFC-0043, RFC-0061

## Context
Agents need output they can parse:
- JSON matching a schema;
- tool calls whose arguments validate;
- sometimes a regex or a fixed set of choices.

Without constraints, small and heavily quantized models break JSON. The research notes that 2-bit quants "break JSON and tool calls" (ADR-0019 context). Constrained decoding makes such output impossible: at each step, every token that cannot continue a valid output is masked out.

Constraints must also fit Syntropy's architecture:
- **Where the token is chosen.** Tokens are chosen at the tail stage, which samples (RFC-0001 §6.2), and the tail may be on a different node from ingress.
- **Speculative decoding.** Drafts are produced at the tail (MTP and draft heads) or at the head (draft models), and verified in windows (RFC-0001 §6.4).
- **Exactness.** A constraint must not turn an exact request into an approximate one.
- **Cost.** Mask computation must not add to the per-token critical path.

Building a fast, general grammar engine is substantial work. It needs:
- an Earley or similar parser for arbitrary context-free grammars;
- a lexer;
- JSON Schema lowering with recursion;
- a vocabulary trie walk with caching of context-independent tokens.

An existing Rust library, `llguidance` (MIT), implements this. Its authors report about 50 µs of CPU time per token at a 128K vocabulary, with negligible start-up cost [reported by library docs]. Its licence is on ADR-0018's allow list.

## Options considered
| Option | For | Against |
|---|---|---|
| A. Mask at ingress, round trip per token | ingress already has the tokenizer and the request | adds a network hop to every decode step when ingress is not the tail; breaks the ring's one-traversal-per-token property |
| B. Validate after generation and retry | trivial | wastes whole generations; never guarantees success; useless for streaming |
| C. Compile at ingress, apply masks at the tail, overlapped with the forward pass; engine is an existing library behind a trait | no extra hop; mask cost hidden behind the ring; months of engine work avoided; replaceable | a dependency in the sampling path; compiled grammars must reach the tail |
| D. As C, with a grammar engine written in-house | full control | duplicated effort; slower to reach the same coverage and speed |
| E. JSON mode only (no schemas, no grammars) | simple | strict tool calls and schema outputs, which agents need, are impossible |

## Decision
**Option C.**

1. **Compile at ingress.**
   - `syntropy-text` lowers JSON Schema, regexes, choices, Lark-style grammars and tool-call constraints into the engine's grammar.
   - Compiled artefacts are cached by BLAKE3(constraint, tokenizer).
   - Unsupported schema keywords fail strict requests and are ignored with a warning otherwise.
2. **Ship by reference.**
   - `SamplingParams` carries the grammar's hash.
   - The first `SequenceOpen` that uses a given hash carries the compiled grammar inline.
   - Tail runners cache grammars by hash (RFC-0050 amendment A3).
3. **Mask at the tail, ahead of time.**
   - After sampling token t, the tail advances the automaton and computes the mask for t+1 while the ring runs the next step.
   - The mask is applied to the logits before every sampling transform.
4. **Speculation stays exact.**
   - Tail-side drafters sample under the same automaton, so drafts are always grammatical.
   - Verification computes masks along the drafted path and accepts against the masked target distribution.
   - A draft model at the head receives the grammar too, so its drafts are grammatical. Head drafts are only proposals, and verification at the tail decides.
5. **Exactness.**
   - A constraint is part of the requested sampler. The reference applies the same mask, so constrained output is exact-mode output.
   - Reasoning budgets are enforced by forcing the end-of-reasoning marker through the same mechanism.
6. **Engine.**
   - `llguidance` is used behind a `ConstraintEngine` trait (`compile`, `new_state`, `advance`, `mask_into`, `is_accepting`).
   - It is pinned by Renovate's major-version hold, and its licence is checked by `cargo-deny`.
   - A conformance suite of schemas and grammars runs against whatever implements the trait.
7. **Reasoning sections stay free.** For reasoning models, the constraint applies after the end-of-reasoning marker.

## Consequences
- **Positive.**
  - Strict tool calls and schema outputs work on every model and quantization, with no extra hop.
  - Constraints compose with speculative decoding without losing exactness.
- **Negative.**
  - The tail needs CPU time for masks: one core per instance's tail in the worst case.
  - A third-party library sits on the sampling path. A bug there yields invalid masks, so the conformance suite and fuzzing are mandatory.
  - Compiled grammars add a small payload to the first `SequenceOpen` per schema.
- **Work created.**
  - RFC-0050: lowering, caching and errors.
  - RFC-0012: sampler integration, verify under masks and forced-token optimisation.
  - RFC-0043: per-family tool-call syntax as grammar fragments.
  - RFC-0061: a mask-cost benchmark per v1 tokenizer.
  - RFC-0060: property tests that sampled outputs always validate.

## What would make us revisit
- Per-token mask time on a v1 tokenizer exceeds half a ring step on fast single-GPU plans after giving the sampler two cores. Then add GPU-side mask application or token-class precomputation.
- The library's licence or maintenance changes. The trait allows replacing it with another engine or an in-house one.
- A model family's tool-call syntax cannot be expressed as a context-free grammar over its tokens. That case would need a dedicated state machine.
