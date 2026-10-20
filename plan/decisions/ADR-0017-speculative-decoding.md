# ADR-0017: Speculative decoding as a first-class amortiser for slow tiers and network hops
- Status: Accepted (revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D2, D3), ADR-0006, ADR-0008, ADR-0010, ADR-0019; RFC-0011, RFC-0012, RFC-0022, RFC-0023, RFC-0032, RFC-0042, RFC-0043

## Context
Syntropy's single-stream decode pays two costs that speculation amortises.
- **Slow tiers.** These are bytes read from pinned RAM, NVMe or a CPU execute-in-place pool.
- **Network hops.** Each one is α + S/β per stage boundary, per token (ADR-0010).

If the target model checks a window of drafted tokens in one pass, it pays both costs once per window instead of once per token.

The arithmetic, from `plan/research/hardware-and-performance-model.md` §3.10:
- **Expected tokens per target pass.** With per-token acceptance `a` and draft length γ, it is `(1 − a^{γ+1}) / (1 − a)`. For a = 0.8 that is 3.36 at γ = 4 and 4.33 at γ = 8 (estimate).
- **Dense target, memory-bound.** `t_target(γ+1) ≈ t_target(1)` while γ+1 stays under the ridge batch of every tier the target touches. For a CPU-hosted tier at Q4 that ridge is ≈3–9 tokens, so γ ≈ 4–6 is the useful range on hybrid desktops. The draft must run in fast memory so that `t_draft ≪ t_target`.
- **MoE target.** Verifying γ+1 tokens reads the union of their experts. For DeepSeek-V3, 5 tokens touch ≈4.7× the routed bytes of one. Native MTP heads are the cheap case: a 2-token union is ≈1.97× routed bytes, with 85–90% second-token acceptance and 1.8× tokens/s [verified].

Field results [reported by surveyed systems]:
- **Tiered single-GPU MoE with an MTP draft of up to 3 tokens** (`tiered-memory-inference.md` §13):
  - 2.4–3.2 tokens per pass and 1.6–1.8× decode, with output identical to greedy decoding.
  - That is lower than the 2.5–3× typical with all weights in VRAM, because each extra token routes to its own CPU-resident experts.
  - Drafting continues only while the MTP head is at least 50% confident.
  - When streaming on Macs, 3 drafts hide the per-layer handshake.
- **Draft on the head device, verify through the pipeline** (`multi-device-inference.md` §11): 25–45% lower time per token for 14–70B models. Llama 3.3-70B went from 803 to 442 ms/token with a 3B draft.
- **Asynchronous pipelined speculation:** up to 2.15×.
- **Trees for sparse edge pipelines:** 1.28–1.79×.
- **Large draft trees for I/O-bound targets:** up to 20 tokens per target iteration.
- **Prompt-lookup drafting** (copying from the context): 6–11% faster code edits.

The research recommends shipping draft-on-head first, then evaluating asynchronous speculation and wider trees where the target is I/O-bound (§15.9).

Constraints:
- D3 requires exact output by default.
- Hybrid linear-attention layers (Gated-DeltaNet or Mamba-class, ADR-0003) carry recurrent state that a rejected branch must not corrupt.

## Options considered
1. **No speculative decoding.** It leaves 1.6–1.8× on tiered MoE and 25–45% on pipelines unused. Rejected under D7.
2. **A separate draft model only.** It works for any family that has a vocabulary-compatible small model. It costs fast memory and a second model's load, and acceptance varies by pairing. Kept as the third choice.
3. **Native MTP heads only.** Cheapest, with high acceptance, but only some families ship them (DeepSeek-V3/V4-class and others). Kept as the first choice.
4. **Trained draft heads** (trained single-layer draft heads that autoregress on the target's hidden state). Cheap at runtime with high acceptance, but someone must train and publish one per family. Kept as the second choice.
5. **Lenient or typical acceptance.** Accepting near-matches raises the acceptance rate but changes output. Allowed only as an opt-in approximate knob (ADR-0019).
6. **Prompt-lookup drafting.** It needs no weights, and its benefit is narrow (repetitive replies, code edits). Used as a supplementary drafter merged into the tree, never alone.

## Decision
1. **Drafter preference per model.**
   1. Native MTP heads when the manifest declares them.
   2. A trained draft head for the family when one is published for that exact target revision. *Revised 2026-10-20: the project produces no draft heads in v1; this choice applies only to a head supplied with the model and passing the conformance gate (below).*
   3. A draft model sharing the target's tokenizer.

   Prompt-lookup candidates are merged into the tree for every drafter at no weight cost. The manifest (RFC-0043) records which drafters exist and their tensor census, so the solver places them like any other tensor group.
2. **Placement.** Drafters run where their inputs are.
   - MTP heads and draft heads consume the target's final hidden state, so they run on the tail stage, next to the LM head and sampler. The embedding rows they need are a `RowLookup` group the solver may duplicate there (ADR-0008 already costs duplicated embedding bytes).
   - A separate draft model runs on the head node, on its fastest device.
   - Either way the drafter's bytes go in the fastest tier available on that node. A drafter that would land on NVMe is disabled.
3. **One ring traversal per window.** The tail verifies, samples, drafts the next window and sends accepted tokens plus the draft tree to the head (ADR-0010). Each stage processes the whole tree in one pass.
   - Frames carry T positions. For example, T = 8 at hidden size 7168 in F32 is 224 KiB per hop (estimate).
   - Only one window is in flight per sequence until asynchronous speculation is enabled (item 7).
4. **Exact acceptance in exact mode.**
   - Greedy requests accept the longest tree path that matches the target's argmax at each position.
   - Sampled requests use speculative rejection sampling generalised to trees, which preserves the target distribution.
   - The sampler state and RNG stay on the tail stage.
   - Exactness is still defined by ADR-0019: speculative and non-speculative decoding must agree within the published tolerance, because a verify batch of T rounds differently from a batch of 1. The parity harness tests both.
5. **Tree shape and window size come from the solver.**
   - Per plan, RFC-0011 picks γ and tree width to minimise the expected time per token, `(t_draft + t_verify(T) + Σ_hops (α + S_T/β)) / E[tokens per pass]`.
   - `t_verify(T)` includes the expert union U(T) per tier for MoE and the ridge batch of every tier the target touches.
   - The acceptance rate `a` is learned online per (drafter, target, greedy or sampled), seeded from 0.7. *Revised 2026-10-19:* v1 has no task-class dimension, because no task classifier is defined; the per-sequence posterior adapts to the task within a few windows (RFC-0012 §11).
   - At run time the scheduler (RFC-0012) prunes branches whose drafter confidence is below 0.5, and lowers γ when measured acceptance falls. It never exceeds the plan's T budget, because CUDA graphs and KV slots are sized for it.
6. **State rollback.**
   - Attention KV for rejected tree nodes is masked during verify and freed afterwards.
   - For families with recurrent or linear-attention layers, verification uses a chain (no branching) of at most 4 tokens, with a state snapshot per position. Trees are enabled for those families only once RFC-0012 shows a tree-capable snapshot within the KV budget.
7. **Staging** (revised 2026-10-19).
   - M1, on one node: model-native MTP heads and independent draft models. M1's exit needs SC-21 on scenario (a), a dense model without MTP heads, so the draft model cannot wait.
   - M3 (with the rest of cross-stage speculation, per RFC-0001 §16): MTP and draft models across pipeline stages, the draft-head loader (no project-trained heads; revised 2026-10-20), wide trees for I/O-bound targets, and asynchronous pipelined speculation behind a flag, each enabled by default only after RFC-0061 shows a gain on Tier 1 hardware.
8. **Off when it does not pay.** In the throughput objective (D2's second mode), speculation is off unless the solver predicts a gain at the current batch. It is also off when measured acceptance stays below the break-even point for the plan for 64 consecutive windows.

## Consequences
- **Positive.**
  - Slow-tier reads and network round trips are paid once per window, which is the regime where tiered and multi-node decode is weakest.
  - Exact mode keeps speculation.
  - The solver owns the trade-off, so MoE models on slow tiers automatically get short windows.
- **Negative.**
  - The tail stage carries drafter bytes and, for MTP and draft heads, duplicated embedding rows.
  - Graph capture multiplies by the number of window sizes.
  - Trained draft heads need an offline training and publishing pipeline per family. That is project work outside the runtime, and it has licence obligations per model (ADR-0018). *Since 2026-10-20 v1 has no such pipeline and publishes no heads (below).*
- **Work created.**
  - RFC-0012: the verify loop, tree attention masks, rollback and snapshots.
  - RFC-0011: the window and tree cost terms.
  - RFC-0023: expert-union prefetch across a tree.
  - RFC-0022: KV slot reservation for trees.
  - RFC-0032: multi-position frames.
  - RFC-0043: drafter declarations in the manifest.
  - RFC-0042 and RFC-0060: spec-versus-non-spec parity.
  - RFC-0061: per-family speed gates.

## What would make us revisit
- On Tier 1 hardware, measured speedup from MTP or draft heads on tiered MoE is below 1.2× across the conformance families. Then make speculation per-family opt-in rather than default.
- Speculative and non-speculative decoding cannot be kept within the ADR-0019 tolerance for some dtype or backend. Disable speculation in exact mode for that pair.
- Model families stop shipping MTP heads, and trained draft heads cannot be sourced under compatible licences. Promote prompt lookup and draft models.
- Asynchronous speculation shows a gain at least 1.5× larger than synchronous trees on two-plus-node pipelines. Make it the M3 default.

## Revisions

- **Revised 2026-10-19.**
  - Item 7: single-node draft models move from M3 to M1, alongside MTP heads (RFC-0012 amendment C3). Trained draft heads, trees for I/O-bound targets, asynchronous speculation and cross-stage speculation stay in M3.
  - Item 5: acceptance is learned per (drafter, target, greedy or sampled) rather than per task class.
  - Options item 4: the published method's name is replaced by a description of the technique (D5).
- **Revised 2026-10-20.**
  - Items 1 and 7: draft heads are dropped from v1. Nothing in the plan trains, evaluates, licenses or publishes them, so the project produces none and publishes none. The loader stays (PLC-E9-T23, M3, tested with a project-synthetic head), so a head that ships with a model's own files can run once its family passes the conformance gate, and community-trained heads can follow later under the conformance and benchmark gate.
  - Speculation in v1 therefore uses, in order: the model's MTP heads where the family has them, a draft model sharing the target's tokenizer, and prompt-lookup candidates merged into either (RFC-0012 §9.1). Scenario (a)'s SC-21 already rests on a draft model, so no milestone exit depends on draft heads.
  - If draft heads ever return as project artefacts, they are published only for bases under permissive licences (Apache-2.0 or MIT) at first, because derivatives of other families carry naming, attribution and flow-down duties.
