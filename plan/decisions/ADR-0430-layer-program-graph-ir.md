# ADR-0430: The graph IR is a static, layer-structured program whose nodes read census groups

- Status: Accepted (2026-10-20); revised 2026-10-20 (decisions 7 and 8)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0002, ADR-0003, ADR-0006, ADR-0017, ADR-0019; RFC-0001 §5.3, §6.3; RFC-0043 §2, §6; RFC-0011, RFC-0012, RFC-0040 §4.4, RFC-0060 §4.1

## Context

- ADR-0003: each family module provides the census, the KV formula, legal stage boundaries, a segment builder for any legal layer range per shape class and verify-window size, and optional tensor-parallel sharding.
- The solver assigns tensor groups, never single tensors (RFC-0001 §5.3), and cuts plans only at layer boundaries where the residual stream (plus declared state) is all that crosses (ADR-0006, ADR-0003).
- The executor replays one captured graph per (shape class, verify-window size), with every per-step value read from a step descriptor (RFC-0040 §3.5, §4.4). Data-dependent control flow cannot be captured.
- The reference forward pass interprets the same graph the product executes (RFC-0060 §4.1; RFC-0042 §7).
- Models in scope mix attention kinds per layer, recurrent scans, routed experts, lookups by token id and hyper-connections (`backends-and-portability.md` §4.1).

## Options considered

| Option | For | Against |
|---|---|---|
| A. A general tensor dataflow graph, with layers and groups inferred | Familiar; flexible | Boundary legality, census classes and placement units must be inferred, and inference fails open |
| B. Per-family imperative code with no IR (the family calls kernels directly) | Least machinery | The solver, census, segment builder, reference and conformance tests would each need their own description of the model |
| **C. A layer program: prologue, layers, epilogue, MTP modules; each a small SSA program of registered `OpKind`s over typed values with symbolic `Rows`, `Seqs` and `Ctx`; nodes name (tensor group, role) and carry a node class; no control flow; routing inside ops; trees as rows** | One structure every consumer reads; boundaries and placement units are explicit; capture-friendly; mechanically validated | Expressiveness limited to what fits the layer shape; anything else needs a new op kind |

## Decision

**Option C.**

1. `ModelGraph { prologue, layers, epilogue, mtp, residual, boundaries }`; each `Program` is SSA over `OpKind` nodes with typed values.
2. Nodes reference weights as `WeightRef { group, role }`; MoE nodes use `GroupSel::Experts { layer }` so expert groups are chosen per step, not enumerated.
3. Each node has a `NodeClass` (Dense, Expert, Router, Head, Lookup, Elementwise, State) from which the step-program lowering places it without architecture knowledge.
4. A boundary is legal when every value live across it is a residual stream or declared side data (token IDs, per-layer inputs); `ir::validate` checks this, SSA, types, group coverage and op coverage after every build.
5. No control-flow nodes; no dynamic shapes beyond `Rows`, `Seqs`, `Ctx`; KV and recurrent state live in the KV store, not in the IR.
6. `graph::reference` interprets the IR with `kernels::reference` op semantics; `graph` therefore depends on `kernels` (RFC-0001 §4.1 amendment).
7. *Revised 2026-10-20:* **Multimodal hooks are reserved, v1 is text-only.** `ModelGraph` carries `encoders: Vec<EncoderProgram>`, `NodeClass` has `Encoder`, `OpKind::EmbedInject` is registered with reference semantics only, and the census reserves `TensorClass::Encoder { tower }`. Every v1 graph leaves them unused, and `ir::validate` enforces that. The first multimodal family is image input for `qwen_hybrid`, after 1.0 (RFC-0043 §2.4).
8. *Revised 2026-10-20:* **Bidirectional attention and pooling fit the layer program.** An encoder-only embedding model is a prologue, layers and an epilogue with one residual stream; `AttnKind::Bidirectional` (no causal mask, the whole input in one prefill step) and `OpKind::Pool` carry it, with no structural change to the IR (RFC-0043 §5.4).

## Consequences

- Positive: census, solver boundaries, segment building, TP splits, the reference and conformance all derive from one validated structure.
- Positive: captured step graphs follow directly from the IR's static shape.
- Negative: an architecture that does not fit the layer-program shape (for example, encoder-decoder models, or cross-layer KV sharing spread across many layers) needs IR extensions; cross-layer sharing is representable only as illegal boundaries. Encoder-only models do fit (decision 8), and so does an encoder feeding a decoder through `EmbedInject` (decision 7).
- Negative: new attention or scan variants need new op kinds and kernels before a family can use them (ADR-0002's coverage cost, unchanged).

## What would make us revisit

- A Tier 1 family appears whose structure cannot be expressed as a layer program without making most boundaries illegal.
- The capture requirement relaxes (a backend gains capturable conditional nodes that Syntropy adopts), so limited control flow in the IR would pay.

## Revisions

- Revised 2026-10-20: decision 7 reserves the multimodal hooks now, because adding an encoder segment and placeholder injection after the IR and census types freeze would break them; decision 8 records that the IR already expresses an encoder-only embedding family with two registered additions.
