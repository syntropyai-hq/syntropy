# ADR-0003: Model graphs are defined in Syntropy

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0000 (D3, D7, D8), ADR-0002, ADR-0005, ADR-0006, ADR-0008, ADR-0017, ADR-0019, ADR-0021, RFC-0011, RFC-0032, RFC-0041, RFC-0042, RFC-0043, RFC-0060

## Context

ADR-0002 makes Syntropy own its runtime, so it also owns the model graphs that the runtime executes. This ADR fixes how those graphs are written.

The 2026 architecture landscape (`plan/research/backends-and-portability.md` §4.1, §4.2) is varied, but it is built from a small set of recurring blocks:

- **Attention:** GQA (Llama 3.x, Mistral Medium 3.5, Qwen dense), sliding-window/global interleaves (Gemma 4 at 5:1 with a 1024 window; gpt-oss alternating 128/full with sinks), MLA (DeepSeek-V3, Kimi K2.5, Mistral Large 3, GLM-5), compressed attention with an indexer (DeepSeek V4: per-layer compress ratios of 4 or 128), and sparse or indexed attention (DSA, Qwen Sparse Attention, MiniMax block-sparse).
- **Linear attention and SSM layers:** Gated DeltaNet (Qwen3.5/3.6/3.8), KDA (Kimi K3), Mamba-2 (Nemotron hybrids). They have a fixed per-sequence state instead of a growing KV cache.
- **FFN:** dense, routed+shared MoE (top-4 to top-16 of 32 to 896 experts), latent MoE (Kimi K3 at dim 3584).
- **Extras:** MTP heads (DeepSeek-V3; a 4B MTP module in Qwen3.8-Flash-Next), n-gram and per-layer embedding tables (a 51B n-gram table in Qwen3.8-Flash-Next; Gemma E-models), gated hyper-connections (`tiered-memory-inference.md` §3.1).

Forces on how graphs are defined:

- **Placement needs per-tensor facts.** The solver must know each tensor's access pattern (dense every token, routed expert, row lookup) and the KV or state bytes per token for each layer type (`tiered-memory-inference.md` §3.1, §20 item 1). KV per token ranges from 20 KiB (Qwen3.6-35B-A3B, hybrid) to 368 KiB (GLM-4.6, GQA) at BF16 (`backends-and-portability.md` §4.2).
- **Stage boundaries must respect layer type.** Linear and full-attention layers differ widely in cost and state, so equal layer counts are the wrong split (§4.2). Recurrent state must stay with one sticky owner (`multi-device-inference.md` §3.4).
- **Tensor parallelism needs an explicit sharding strategy per architecture.** Surveyed runtimes run unknown architectures pipeline-only (`multi-device-inference.md` §4.4).
- **The single-model trap.** The fastest surveyed tiering engine hard-codes one geometry (`H = 2560`, a 1,382,400-byte expert blob) and refuses every other shape (`tiered-memory-inference.md` §3.2, §18, §20 item 14).
- **Split mode should fail closed** for untested architectures. One surveyed runtime enforces a certified roster tied to its engine version (`multi-device-inference.md` §3.4).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Reuse graph builders from ggml's model library | About 160 families on day one | Rejected by ADR-0002: its scheduler and buffer model come with the builders |
| B. Trace or export graphs from the publisher's framework code (Python tracing, ONNX export) | No per-family code | Needs Python or a 2 GB-protobuf workaround (`backends-and-portability.md` §3.1); traced graphs carry no census, stage-boundary or sharding semantics; dynamic routing traces badly |
| C. A declarative graph description format, interpreted at load | New families as data | The format grows into a language; routing, indexers and recurrent scans need escape hatches anyway; harder to type-check and debug |
| **D. One Rust module per family, built on shared blocks; per-model differences from manifest hyperparameters** | Typed, testable, shares kernels and blocks; census and sharding live next to the graph | Syntropy writes code for every family, so each new family is a set of tasks and a gated change (RFC-0043 §7) |
| E. Hand-specialised engines per model | Fastest per model | The single-model trap above |

## Decision

**Option D.**

1. **Each architecture family is a Rust module** in the graph crate (RFC-0043), built on shared blocks:
   - attention: GQA/MHA, MLA, sliding-window (including sinks), sparse or indexed, compressed with indexer;
   - linear attention: Gated-DeltaNet and Mamba-class scans;
   - FFN: dense, routed+shared MoE, latent MoE;
   - MTP heads.
2. **Per-model differences come only from manifest hyperparameters** (ADR-0005): layer count, layer-type pattern, head counts and dims, expert counts and top-k, window sizes, compress ratios, norm and RoPE variants. A new size or fine-tune of a supported family needs no code.
3. **Every family module must provide:**
   - the **tensor census**: each tensor classified as `DenseEveryToken`, `Expert{layer, id}`, `RowLookup`, `Head` or `Router`;
   - the **KV bytes-per-token formula** per layer type, and the fixed state size for recurrent layers;
   - **legal stage boundaries**: the layer indices where only the residual stream (plus any declared per-sequence state that moves with it) crosses;
   - a **graph segment builder** for any legal layer range `[a, b)`, for the decode and prefill shape classes and each verify-window size used by speculative decoding (ADR-0017);
   - an optional **tensor-parallel sharding strategy**. A family without one is never placed with tensor parallelism (ADR-0006).
4. **v1 families:** Llama/Mistral/Qwen dense; Gemma-class sliding/global; DeepSeek-V3/V4-class MLA and compressed attention with MoE; Kimi-class; Qwen-MoE and Qwen hybrid (Gated DeltaNet plus gated or sparse attention); gpt-oss-class; GLM-class.
5. **A conformance suite gates each family** (RFC-0060). A family ships only when it passes all of:
   - teacher-forced top-1 agreement and KL divergence against an independent reference forward pass at the same quantization, within the ADR-0019 tolerances;
   - the same checks with the model split into 2 and 3 stages on the simulated fabric, at every legal-boundary pattern the test selects;
   - the same checks with weights placed in VRAM, in pinned RAM with execute-in-place, and behind the NVMe tier;
   - every kernel provider that claims the family's ops.
6. **Fail closed.** An unknown architecture string or a family that has not passed conformance at this Syntropy version is refused with an error that names it. A developer flag can load an uncertified family on one device only, labelled experimental everywhere; it is never placed across nodes.
7. **Family bring-up and freshness** (added 2026-10-20).
   - Agents write each family module as tasks, through the gates, the parity and conformance harness, including the model-definition oracle (ADR-0019 decision 1a), and the review agent. A maintainer approves each new family (ADR-0021 decision 5), because a new family brings a new graph that only its conformance run has checked. A community-contributed family module takes the same path.
   - **Targets** [estimate, from RFC-0043 §7's steps; replaced by the times measured for the M1–M3 families]: conformance within 4 weeks of a Tier 1 family's public release when existing blocks suffice, and within 8 weeks when it needs a new block and Tier 1 kernels.
   - A new release in a family Syntropy already supports reaches the catalogue within 7 days [target].
   - Time to conformance is measured per family from M1. Each family's task set records the rig hours it needed on each Tier 1 backend, the gate run time and the conformance jobs' rig queue wait.
   - When rig time runs short, native kernels fall back to the ggml provider in the scope-cut order (RFC-0001 §16): HIP and Vulkan first.

## Consequences

Positive:
- The census, KV formula, stage boundaries and sharding strategy are written once per family, next to the graph, and the solver consumes them directly.
- About eight family modules cover the v1 roster because blocks are shared, instead of one file per model.
- Conformance gives a clear definition of "supported", which the product can show per model.

Negative:
- A new architecture runs only after Syntropy implements and certifies it. This is a coverage lag against runtimes that reuse a large builder library.
- Coverage is a capacity question measured in tasks (ADR-0021 decision 6), not people. Agents write each family as tasks: the module, its fixtures, any new block and its kernels. The conformance gates and the publisher-reference check certify it, and a maintainer approves the merge as a gated change class (RFC-0043 §7; ADR-0021 decision 5). The lag is therefore bounded by agent throughput, lab rig time for new kernels and maintainer approval. Decision 7 sets the target times from a family's public release to conformance.
- New attention types need both a block and kernels (ADR-0002), so one model release can mean work in two crates.
- The conformance suite needs reference outputs per family and quantization, which costs CI storage and compute (RFC-0062).

Work created: RFC-0043 (IR, block library, family trait, boundary rules), RFC-0041 (manifest fields each family reads), RFC-0060 (conformance suite), RFC-0042 (tolerances per dtype), RFC-0011 (consuming legal boundaries and census), RFC-0032 (tensor-parallel sharding strategies).

## What would make us revisit

- The time from a Tier 1 family's public release to passing conformance misses its decision 7 target (4 weeks, or 8 with a new block) twice in a row. *Revised 2026-10-20:* the clock runs from the family's public release, the day its publisher releases the weights, until its `FamilyCert` passes the gates, not from the first commit of the family's task, so a family started late still trips it. The trigger's earlier unit, one release cycle, applies again once a release cadence is defined. Each miss records which constraint held it up: task throughput, a gate failure, lab rig time or maintainer approval. That separates a capacity shortfall, which is fixed by relieving the constraint, from a flaw in owning the graphs, which reopens this ADR.
- Unsupported-architecture refusals become the top category in issue reports for two consecutive releases.
- A declarative format proves it can express at least 90% of the v1 roster with no Rust code and no measurable decode loss. That would justify moving the long tail to data while keeping Rust modules for the hot families.
- Linked to ADR-0002's note on model coverage: if the research's coverage concern proves right, these triggers fire first.

## Revisions

- **Revised 2026-10-20:** decision 7 (new) sets how families are brought up: agents write them through the gates and the review agent, a maintainer approves each new family, and conformance targets of 4 weeks (existing blocks) and 8 weeks (a new block) apply, with 7 days to the catalogue for a release in a supported family. The first revisit trigger now runs from the family's public release, not from the first commit of its task, because a trigger measured from the first commit never fires for a family that is started late; it uses the 4- and 8-week targets until a release cadence is defined.
