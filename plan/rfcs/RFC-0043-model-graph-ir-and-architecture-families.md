# RFC-0043: Model graph IR and architecture families

- Status: Draft
- Area: compute and models
- Depends on: RFC-0001 (§4.1, §5.3, §5.4, §5.8, §6.3, §6.4, §7.1), RFC-0011, RFC-0012, RFC-0022, RFC-0023, RFC-0032, RFC-0040, RFC-0041, RFC-0042, RFC-0050, RFC-0060; ADR-0002, ADR-0003, ADR-0006, ADR-0017, ADR-0019, ADR-0021, ADR-0602; ADR-0430, ADR-0431
- Date: 2026-10-19
- Revised: 2026-10-20: the model-definition oracle in the gate (§9.2); literal content tokenization (§8.7); reserved multimodal hooks (§2.4); the encoder-only embedding family (§5.4); how families are added and when the coverage trigger fires (§7); `FamilyCert::golden` and `max_ctx` (§9.4); open questions 1, 3, 4 and 6 answered and 2 partly answered

## Summary

ADR-0003 makes each architecture family a Rust module built from shared blocks, with per-model differences coming only from hyperparameters. This RFC defines what those modules produce and consume:

- **The graph IR**: a layer-structured, shape-polymorphic program over `OpKind`s (RFC-0040 §3.1), in which each node names the tensor group it reads. It is not a general dataflow graph. Layer boundaries are the only cut points, and every node carries the census class of its weights, so the solver, executor and census all read the same structure.
- **The block library**: attention (GQA/MHA, sliding with sinks, MLA, sparse and indexed, compressed), linear-attention and state-space scans (Gated DeltaNet, KDA, Mamba-2), dense FFN, routed-plus-shared MoE with latent variants and router exposure, embeddings including n-gram and per-layer tables, hyper-connections, LM head and MTP modules. For each block the RFC gives the ops it lowers to, the census classes of its weights, its KV or state formula, its tensor-parallel split and what it tells the expert cache about routing.
- **The `Family` trait** and the v1 family modules: Llama-class dense, Gemma-class, gpt-oss-class, DeepSeek-V3-class (also Kimi K2-class and Mistral Large 3), DeepSeek-V4-class, Kimi-K3-class, Qwen MoE and Qwen hybrid, GLM-class, and from M2 one encoder-only embedding family (`bert`).
- **Stage cut points, segment building, and tensor-parallel sharding**, which the solver (RFC-0011), the scheduler (RFC-0012) and distributed execution (RFC-0032) consume.
- **Tokenizers and chat templates**: the HF `tokenizers` crate as the engine, with GGUF tokenizer metadata converted into its models, and `minijinja` for templates (ADR-0431). Content from users, tools and documents is tokenized literally, so it can never become a control token (§8.7).
- **The conformance gate inputs** each family must supply, including golden data from the publisher's own reference code for the model-definition oracle (§9.2; RFC-0042 §7.1), and how a new family is added.
- **Reserved hooks for multimodal input**: v1 is text-only, but the IR, census and node classes reserve an encoder segment and an `EmbedInject` node now, so a later release adds image input without an IR break (§2.4).

## Motivation and goals / non-goals

Placement, caching and execution all depend on facts that only the architecture knows: which tensors are read every token, which are experts, which are lookups; how many KV bytes each layer type stores; where a model may be cut; how a layer splits across tensor-parallel devices; when an MoE layer's routing is known. ADR-0003 puts these facts next to the graph so they are written once. This RFC makes that concrete enough to implement, and makes sure every consumer reads the same structure.

The 2026 landscape (`backends-and-portability.md` §4.1, §4.2) is varied: GQA, sliding/global interleaves with sinks, MLA, compressed attention with indexers, block-sparse and token-sparse attention, Gated DeltaNet, KDA, Mamba-2, routed MoE from 32 to 896 experts with top-4 to top-16, latent MoE, MTP heads, n-gram and per-layer embedding tables, hyper-connections. It is still a small set of blocks. The single-model trap, where a surveyed engine hard-codes one geometry (`tiered-memory-inference.md` §3.2), is what the block library prevents.

### Goals

1. One IR that the census, the solver's boundary rules, the step-program lowering (RFC-0012) and the reference forward pass (RFC-0042 §7) all derive from.
2. Every v1 family from ADR-0003 expressed as hyperparameters over shared blocks, with nothing model-specific outside its family module.
3. Fail closed: an unknown architecture, an unclassifiable tensor or an uncertified family is refused with a precise error.
4. A new family needs one module, one fixture spec and a conformance run, not changes across the codebase.

### Non-goals

- Training, autograd or graph optimisation passes beyond fusion patterns the providers declare.
- A declarative format for families (ADR-0003 option C).
- Multimodal input in v1. v1 is text-only: vision, audio and other tower tensors are classified `Ignored` and never loaded (§4.3). The IR, the census and the node classes reserve the hooks now (§2.4), and the first multimodal family is named there, so the work after 1.0 adds code without changing these types.
- Encoder-only models in v1, except one encoder-only embedding family: `bert`, single node only, from M2 (§5.4). Decoder-only embedding models on v1 families are supported too (§5.4).

## Design

### 1. The `graph` crate

| Module | Contents |
|---|---|
| `graph::ir` | the IR types (§2) and their validator |
| `graph::blocks` | the block library (§3): each block builds IR nodes from its hyperparameters |
| `graph::families` | one module per family (§5) and the static registry `graph::families()` |
| `graph::classify` | name-pattern matching for the census (§4) |
| `graph::segment` | stage cut points and the segment builder (§6) |
| `graph::shard` | tensor-parallel sharding rules (§6.3) |
| `graph::tokenize` | `TokenizerSpec` extraction and GGUF tokenizer conversion (§8) |
| `graph::reference` | the scalar interpreter of the IR used by the reference forward pass (RFC-0042 §7, RFC-0060 §4.1) |

`graph` depends on `manifest` and, for the op vocabulary (`OpKind`, `QuantScheme`-aware op keys) and the reference op semantics, on `kernels`. Both are lower layers, so the dependency respects RFC-0001 §4.2 rule 2; the crate table in RFC-0001 §4.1 lists only `manifest` and is amended below. `graph` contains no `unsafe` code and no backend code.

### 2. The graph IR

ADR-0430 records the design. The IR is a **layer program**: a model is a prologue, a list of layers, an epilogue and optional MTP modules. Each part is a small SSA program of `OpKind` nodes over typed values.

#### 2.1 Types

```rust
pub struct ModelGraph {
    pub family: &'static str,
    pub prologue: Program,                  // embeddings, n-gram and per-layer tables, input scaling
    pub layers: Vec<LayerProgram>,          // index = LayerIx
    pub epilogue: Program,                  // final norm, LM head, logit soft-capping
    pub mtp: Vec<LayerProgram>,             // MTP modules, LayerIx after the last decoder layer
    pub encoders: Vec<EncoderProgram>,      // reserved (§2.4): empty for every v1 family
    pub residual: ResidualSpec,             // number of streams, dtype F32 (RFC-0042 §4.1)
    pub boundaries: BoundarySet,            // §6.1
}

/// Reserved for multimodal input after 1.0 (§2.4). An encoder turns one non-text input into
/// rows of the text model's hidden size, which the prologue's `EmbedInject` node places.
pub struct EncoderProgram {
    pub modality: Modality,                 // Image | Audio | Video
    pub body: Program,                      // tower and projector; every weight node has NodeClass::Encoder
    pub output: ValueType,                  // [Rows, hidden] in F32
}

pub struct LayerProgram {
    pub layer: LayerIx,
    pub kind: LayerKind,                    // { mixer: MixerKind, ffn: FfnKind } (RFC-0001 HyperParams::layers)
    pub blocks: SmallVec<[Block; 3]>,       // typically [mixer, ffn]; hyper-connection mixing adds one
    pub body: Program,
}

pub struct Program {
    pub inputs: SmallVec<[ValueId; 4]>,     // residual streams in; token IDs where the program needs them
    pub nodes: Vec<Node>,                   // topologically ordered
    pub outputs: SmallVec<[ValueId; 4]>,
}

pub struct Node {
    pub op: OpKind,                         // RFC-0040 §3.1
    pub inputs: SmallVec<[ValueId; 4]>,
    pub outputs: SmallVec<[ValueId; 2]>,
    pub weights: SmallVec<[WeightRef; 4]>,  // which tensor group, which role inside it
    pub attrs: OpAttrs,                     // static parameters: eps, RoPE config, window, top-k, scales
    pub class: NodeClass,                   // Dense | Expert | Router | Head | Lookup | Elementwise | State | Encoder (reserved, §2.4)
    pub fuse: FuseHints,                    // which neighbours the lowering may fuse (RFC-0040 OpKey::variant)
}

pub struct WeightRef { pub group: GroupSel, pub role: TensorRole }
pub enum GroupSel {
    Fixed(TensorGroupId),                   // dense groups, heads, routers, tables
    Experts { layer: LayerIx },             // MoeGrouped: the expert groups of this layer, chosen per step
}
pub enum TensorRole { Q, K, V, Qkv, O, QA, QB, KvA, KvB, Gate, Up, GateUp, Down, Router, RouterBias,
                      Norm, Embed, Head, Indexer, ScanIn, ScanGate, ScanOut, Conv, Proj, Other(u8) }

pub struct ValueType { pub dtype: DType, pub dims: SmallVec<[Dim; 4]> }
pub enum Dim { Const(u32), Rows, Ctx, Seqs }   // Rows = token rows of the step; Ctx = context length
```

**Shape polymorphism.** Values are typed with symbolic `Rows`, `Ctx` and `Seqs`. Lowering a segment to a bucket (RFC-0040 §3.2) substitutes `Rows` and `Seqs` with the bucket size. `Ctx` stays dynamic: kernels read it from the step descriptor.

**Weights by group, not by tensor.** Nodes name a `TensorGroupId` and a role. The census groups (RFC-0041 §3.1) are exactly the units the IR reads, so the solver's `TierAssignment` per group maps straight onto nodes, and the executor binds each node to the compute site of its group (RFC-0001 §6.3).

**Node classes.** `NodeClass` mirrors `TensorClass` for weight-reading nodes and adds `Elementwise` (norms, residual adds, activations: run on the device of the neighbouring weight node) and `State` (KV writes, scans: run where the layer's KV lives, RFC-0022). The step-program lowering uses it to place ops without knowing the architecture.

#### 2.2 Validation

`ir::validate` runs after every family build and in tests:

1. SSA: every value defined once, before use; programs acyclic.
2. Types: every node's input and output types match its `OpKind`'s signature in the registry (RFC-0040 §3.1).
3. Weights: every `WeightRef` names a group of the matching class, every census group is read by at least one node, and no group is read by nodes in two different layers. The one exception is an MTP module that shares the LM head: it reads the epilogue's `Head` group, which is always on the same stage because §6.1 forbids separating MTP modules from the epilogue. Tied embeddings and the MTP module's embedding read are not exceptions: the census gives each reader its own group, linked by `TensorGroup::shared_with` (RFC-0041 §3.1), so the MTP module's `RowGather` binds to a group the tail holds.
4. Boundaries: every boundary in `BoundarySet` satisfies §6.1.
5. Coverage: every `OpKind` used is in the family's declared `required_ops` (RFC-0040 §11.1).
6. Reserved hooks: in v1, `encoders` is empty and no node is an `EmbedInject` or of class `Encoder` (§2.4).
7. Attention direction: `AttnPrefill(Bidirectional)` appears only in an encoder-only graph (§5.4), and an encoder-only graph has no `AttnDecode`, no sampler ops and no KV kept across steps.

#### 2.3 What the IR is not

There are no control-flow nodes. Data-dependent routing is expressed inside ops (`MoeRoute` → `MoeGrouped` with `GroupSel::Experts`), and speculative trees are rows with parent indices in the step descriptor (RFC-0040 §3.5). There are no dynamic shapes beyond `Rows`, `Seqs` and `Ctx`, and no tensors that live across steps except KV and recurrent state, which the KV store owns (RFC-0022). These restrictions are what let a captured step graph serve every step (RFC-0040 §4.4).

#### 2.4 Reserved multimodal hooks

v1 serves text only. Adding an encoder stage, image-token prefill and per-image caching after the IR and census types freeze would break them, so these are reserved now. Each one exists as a type; no v1 family produces it, and `ir::validate` rejects a v1 graph that uses it.

| Hook | What it is | v1 state |
|---|---|---|
| `ModelGraph::encoders` and `EncoderProgram` (§2.1) | one program per input modality: the tower (for images, a vision encoder) and its projector, producing rows of the text model's hidden size | always empty |
| `NodeClass::Encoder` | weight-reading nodes of an encoder program. They run once per encoded input, not once per token, so the step-program lowering and the solver can place them apart from the decoder's per-token work | never produced |
| `OpKind::EmbedInject` (RFC-0040 §3.1) | a prologue node that writes an encoder's output rows over the embedding rows at the positions of the input's placeholder tokens. RFC-0022 §7.1 already keys non-text inputs in the prefix cache by placeholder IDs followed by the content hash | registered with its `kernels::reference` semantics; no provider kernel |
| `TensorClass::Encoder { tower }` (RFC-0001 §5.3, proposed) and `ClassSel::Encoder` | the census class for tower and projector tensors: dense, read once per encoded input | family classification rules already recognise tower tensors by name and tag them with their tower (§4.3), but classify them `Ignored` in v1 |
| API hooks | image content parts and a per-model input-modality list (RFC-0050) | text-only; image parts are rejected as today |

*Note 2026-10-29 (as built, part of COMP-E15-T11):*
- **Rule 6.** `ir::validate` refuses a graph with an encoder program, an `EmbedInject` node or an `Encoder` node: `Rule::ReservedHooks`, with the violations `EncoderProgram`, `EmbedInject` and `EncoderNode`. Rules 1 to 5 still check encoder programs like any other program.
- **`EmbedInject`'s semantics.** `kernels::reference::embed_inject` writes encoder row i over the embedding row at the i-th placeholder position, and every other row keeps its bits. The positions must be in range and distinct. They may come in any order, since a family may encode its inputs in another order than they appear. The function is generic over the element type and only moves values, so it gives the same result in both reference modes, F32 and F64. The positions are an argument; how the request supplies them is the multimodal family's to define.
- **The census class.** `TensorClass::Encoder { tower }` is already in `manifest`'s census, and `ClassSel::Encoder` in `graph`'s classification rules.
- **Still to come.** The task's last criterion has not landed: each v1 family's classification rules tagging tower tensors `Ignored`, checked on recorded headers of its multimodal checkpoints. It waits on the families' own tasks and the fixture store.

**The first multimodal family is `qwen_hybrid`, with image input**, in a release after 1.0. Its Qwen3.5-class checkpoints ship a vision tower, it is an M1 family with small checkpoints, and its Apache-2.0 licence lets the fixtures and the oracle's golden data be public (RFC-0060 §12.3). It is the target that ADR-0502's revisit trigger ("multimodal model families enter the v1 list") points at. Choosing a different family later is an ADR change, not a type change.

### 3. The block library

A block is a Rust function that appends nodes to a layer's program from a parameter struct filled by the family's hyperparameter parser. Blocks are the only place an attention, scan or FFN variant is described. Families compose them.

#### 3.1 Mixers (attention and scans)

| Block | Parameters | Lowers to (`OpKind`) | Weights (class) | KV or state (`LayerKv`, layout) | TP split (§6.3) |
|---|---|---|---|---|---|
| `Gqa` (covers MHA and MQA) | heads, KV heads, head dim, Q/K norm, bias, partial rotary, RoPE kind and scaling, logit soft-cap, attention scale, no-RoPE layers | `RmsNorm` → `Gemv/Gemm` (QKV, often fused) → `Rope` folded into `KvWrite(Gqa)` → `AttnDecode/AttnPrefill(Gqa)` → `Gemv/Gemm` (O) → `Add` | QKV, O, norms: `DenseEveryToken` | `PerToken { 2·n_kv·d_head }`, `gqa.hpd` | by KV-head groups: whole groups of `n_q/n_kv` query heads with their KV head |
| `SlidingGqa` | as `Gqa` plus window, sinks (learned per-head logits), separate RoPE base for local layers | as `Gqa` with `SlidingGqa { sinks }` attention | as `Gqa`, plus a sink vector | `PerToken { window: Some(w) }`, `gqa.hpd` | as `Gqa` |
| `Mla` | heads, `q_lora_rank` (optional), `kv_lora_rank` d_c, RoPE head dim d_rope, nope and value head dims | `RmsNorm` → Q down/up projections → KV down projection → norm → `KvWrite(Mla)` of [latent, RoPE key] → absorbed `AttnDecode(Mla)` (prefill may use the unabsorbed form) → O | Q_A, Q_B, KV_A, KV_B, O, norms: `DenseEveryToken` | `PerToken { d_c + d_rope }`, `mla.pt` | by heads; the latent cache and KV down projection are replicated per member |
| `SparseIndexed` (token top-k, DSA-class) | as `Mla` or `Gqa`, plus indexer heads, indexer dim, top-k | adds `AttnIndex(TokenTopK)` before attention; attention reads only selected positions (`TopKSparse`) | indexer weights: `DenseEveryToken` | main cache as its base block; `Sparse { selected: k }`; plus `AuxKv::SparseIndexer`, `indexer.pt` | by heads; the indexer is replicated |
| `BlockSparse` (QSA- and block-sparse classes) | block size, block budget or top-k blocks | `AttnIndex(BlockTopK)` → `AttnDecode(BlockSparse)` | as `Gqa` plus block-scoring weights if any | `Sparse { selected }`; aux block summaries where the family keeps them | as `Gqa` |
| `Compressed` (DeepSeek-V4-class) | per-layer compression ratio r: 0, 4 or 128 in the published configurations; raw window of the last 128 tokens; indexer top-k, on r = 4 layers only; a learned per-head sink; single KV head of dim 512; RoPE by ratio: `rope_theta` without scaling on r = 0 layers, `compress_rope_theta` with YaRN scaling on r > 0 layers (`model.py` lines 475–481) | every token: `KvWrite(Compressed)` of its raw-window entry. r > 0: the compressor projection → `KvWrite(Compressed)` of entry j once its last token is processed. r = 4 only: `AttnIndex(CompressedTopK)` over the formed entries. Then `AttnDecode/AttnPrefill(Compressed { sinks: true })`: one softmax, with the sink, over the raw window plus the entries the layer reads, the indexer's top-k on r = 4 layers and every formed entry on r = 128 layers. An r = 0 layer is the same block with no compressor and no indexer: its attention reads the raw window alone | compressor (r > 0), indexer (r = 4), attention weights, sink: `DenseEveryToken` | r = 4: `Compressed { elems_per_entry: 512, tokens_per_entry: 4, window: Some(128), selected: Some(index_topk) }`, plus the indexer's keys (OQ-MF-16 (d)); r = 128: `Compressed { elems_per_entry: 512, tokens_per_entry: 128, window: Some(128), selected: None }`; r = 0: `PerToken { elems_per_token: 512, window: Some(128) }`, the raw window alone. Layout `compressed.pe`; the raw window as a `Windowed` class (RFC-0022 §3) | by heads where there are several; replicated with one KV head |
| `GatedDeltaNet` | heads (K and V), d_k, d_v, conv kernel width, gating form | `CausalConv1d` → projections → `ScanDecode/ScanPrefill(GatedDeltaNet)` → gated output norm → O | in/gate/out projections, conv weights: `DenseEveryToken` | `Fixed { heads·d_k·d_v + conv state }`, `state.f32` | by heads |
| `Kda` (Kimi-K3-class linear attention) | as `GatedDeltaNet` with per-channel gating | as above with `Kda` scans | as above | `Fixed`, `state.f32` | by heads |
| `Mamba2` | heads, d_state, conv width, chunk size, groups | `CausalConv1d` → `ScanDecode/ScanPrefill(Mamba2)` → gated norm → out projection | in/out projections, A, D, dt bias: `DenseEveryToken` | `Fixed { heads·d_head·d_state + conv state }`, `state.f32` | by heads (and groups) |
| `EncoderAttn` (encoder-only embedding models, §5.4) | heads, head dim, QKV bias, absolute or rotary positions, pre- or post-norm | `Gemm` (QKV) → optional `Rope` → `KvWrite(Gqa)` for every row of the input → `AttnPrefill(Bidirectional)`: every query attends to every key of its own sequence, with no causal mask → `Gemm` (O) → `Add` → `LayerNorm` where post-norm | QKV, O, norms: `DenseEveryToken` | `PerToken { 2·n_heads·d_head }`, `gqa.hpd`; pages released when the prefill-only sequence ends | none: encoder families run on one node and never with TP (§5.4) |

Rules common to mixers:

- **Gated attention** (an output gate on the attention result, as in Qwen hybrid and Kimi-K3-class gated MLA) is a flag on `Gqa`/`Mla`: one extra projection and an elementwise gate, no new op kind.
- **K = V sharing** (Gemma-4-class global layers) is a flag on `Gqa`: one projection serves both, and the cache stores one plane. It uses the `gqa.hpd` layout with the V plane omitted, which halves `elems_per_token` in `KvFormula`.
- **RoPE variants** (plain, linear scaling, YaRN, Llama-3-style frequency scaling, partial rotary, per-layer base for local and global layers) are `OpAttrs` of the `Rope` op, computed once into tables at `prepare` (RFC-0042 §4.2 fixes their precision).

*Revised 2026-10-23 (COMP-E16-T18; OQ-MF-16 (a)–(c)): the `Compressed` row.* The row follows DeepSeek's reference code at `deepseek-ai/DeepSeek-V4-Pro` revision `b5968e9190ef611bbf34a7229255be88a0e937c1` (open question 2), which runs all three layer kinds through one `Attention` class (`model.py` lines 453–482):
- **Three ratios.** r = 0 layers keep the raw window only: the MTP layer, and V4-Flash's layers 0 and 1. They lower to the same block, so the raw window's normalised 512-channel vectors and their FP8 rounding go through one write path and one kernel.
- **One indexer kind.** Only r = 4 layers have an indexer, and it selects compressed entries, not tokens (`AttnIndex(CompressedTopK)`). r = 128 layers read every formed entry (`selected: None`).
- **The sink** sits inside the one softmax over the window and the entries (RFC-0040 §3.1's `Compressed { sinks }`).

#### 3.2 Feed-forward

| Block | Parameters | Lowers to | Weights (class) | Notes |
|---|---|---|---|---|
| `DenseFfn` | inter dim, activation (SiLU-gated, GELU-gated, clamped SiLU-gated for gpt-oss-class) | `RmsNorm` → `GateUpAct` → `Gemv/Gemm` (down) → `Add` | gate/up/down: `DenseEveryToken` | TP by inter columns in tiles of the quant block (§6.3) |
| `Moe` | experts E, top-k, shared experts, routing (softmax or sigmoid; optional per-expert bias; optional group-limited top-k with `n_group`, `topk_group`; routed scaling factor; renormalisation), expert inter dim, latent dim (latent MoE) | `RmsNorm` → `MoeRoute` (router logits, selection, weights, written to the doorbell) → shared expert as `DenseFfn` nodes → `MoeGrouped` over `GroupSel::Experts` → `MoeCombine` → `Add` | router (+bias): `Router`; each expert: `Expert { layer, id }`; shared: `DenseEveryToken`; latent down/up projections: `DenseEveryToken` | the router node declares `RoutingKnown` (§3.4); EP and TP rules in §6.3 |

**Latent MoE** (Kimi-K3-class, expert dimension 3584): a dense down-projection into the latent space before routing to experts and an up-projection after the combine. Both are `DenseEveryToken` nodes around the MoE block, and the experts work in the latent dimension, which sets their byte size (`backends-and-portability.md` §4.3).

**Dense-first layers** (DeepSeek-V3-class: the first few layers are dense) are simply layers whose `FfnKind` is `Dense`; nothing special.

#### 3.3 Embeddings, head, MTP and residual structure

| Block | Lowers to | Weights (class) | Notes |
|---|---|---|---|
| `Embed` | `RowGather` → optional scale (√hidden in Gemma-class) | token table: `RowLookup` | prologue; head stage |
| `NgramEmbed` (Qwen3.8-Flash-Next-class) | hash of the last n tokens → `RowGather` into the n-gram table → add to the residual at the layer the family names | table: `RowLookup`, typically `Nvme`-homed (`backends-and-portability.md` §4.3) | the n-gram ids depend only on token ids, so lookups issue as soon as a token is sampled (RFC-0001 §6.3 step 5) |
| `PerLayerEmbed` (Gemma E-class) | `RowGather` of the token's row in the per-layer table, split into one input per layer, each projected and gated inside its layer | table: `RowLookup`; per-layer projections: `DenseEveryToken` | the table is read in the prologue only; per-layer inputs travel as side data to stages that hold later layers (§6.1) |
| `HyperConn` (hyper-connection models) | `HyperMix` before and after each block, over `residual_streams` streams | mixing weights: `DenseEveryToken` | every stream is a residual value; frames carry all of them (RFC-0001 §5.8) |
| `LmHead` | final norm → `Logits` → optional logit soft-cap | head: `Head` (tied: §4.2) | epilogue; tail stage |
| `Pool` (embedding models) | optional final norm → `Pool(kind)` (mean over the sequence's rows, the first row, or the last row) → optional L2 normalisation, as `EmbeddingSpec` declares | none | epilogue in place of `LmHead` for embedding models, decoder-only or encoder-only (§5.4) |
| `TokenTypeEmbed` (BERT-class encoders) | `RowGather` of the segment-type row (type 0 for single-segment input) → `Add`; and, for absolute positions, `RowGather` of the position row → `Add` | tables: `RowLookup` | prologue |
| `Mtp` | the family's MTP module: projection of [final hidden, embedding of the next token] → one decoder layer built from the same blocks → shared or own head | own layer weights by class; shares embedding and head where the checkpoint does | tail stage (RFC-0001 §6.4); `SpeculativeSupport::Mtp { heads }` |

#### 3.4 Router exposure

Each `Moe` block declares, per layer, when its routing is known, which RFC-0023 §9 uses for prefetch (`RoutingKnown`, RFC-0023 §2):

| Value | When the family sets it |
|---|---|
| `AtRouter` | default: the router reads the post-attention residual of its own layer |
| `PreAttention` | the router reads the layer's input residual, so routing is known before attention runs |
| `PreGate { lookahead }` | the model's own gate at layer l selects the experts of layer `l + lookahead` |
| `TokenHash` | the layer routes by a hash of the token id (DeepSeek-V4-class `num_hash_layers`), so routing is known when the token is sampled |

The router node writes its logits and selection to the doorbell (RFC-0023 §2), and for prediction the family exposes the next layers' router weights to the predictor fused into the router op (RFC-0023 §9).

### 4. Census classification

#### 4.1 Rules as data inside the family module

Each family carries a table of name patterns, one per source convention (GGUF names and the original safetensors names), compiled once into a matcher:

```rust
pub struct ClassRule {
    pub pattern: &'static str,              // e.g. "blk.{L}.attn_q.weight", "model.layers.{L}.self_attn.q_proj.weight"
    pub block: BlockSel,                    // which block of layer L, or Prologue / Epilogue / Mtp(n)
    pub role: TensorRole,
    pub class: ClassSel,                    // DenseEveryToken | Expert | ExpertsStacked | RowLookup | Head | Router
    pub aux_of: Option<&'static str>,       // scale and zero-point tensors bind to this weight (RFC-0041 §2.2)
}
pub enum Classification {
    Classified { class: TensorClass, layer: Option<LayerIx>, role: TensorRole, stacked_experts: Option<u16> },
    Ignored { reason: &'static str },
    Unknown,
}
```

`classify` matches the name, then checks the shape against the hyperparameters: a `Q` projection must be `[n_heads·d_head, hidden]` in source order, and a stacked expert tensor must have `n_experts` in its outermost dimension. A name that matches with the wrong shape is `Unknown`, not a guess. RFC-0041 §3 turns the result into records and groups.

*Note 2026-10-23 (COMP-E16-T01, as built):* `syntropy_graph::classify` holds the engine. The choices this section leaves open:

- `ClassRule` has a `shape` field, one `ShapeDim` per dimension (`Const`, `Hp(name)`, `Mul(a, b)`, `Any`), written in the source's order: ggml's `ne`, innermost first, for GGUF, and outermost first for safetensors. Each `RuleTable` states its order.
- Patterns use `{L}` (a decoder layer, or for an MTP rule the module's layer `n_layers + n`), `{E}` (a routed expert, bounded by the `n_experts` hyperparameter) and `{N}` (any other index, such as a tower layer, matched but not read). Each matches a decimal number with no leading zero. `Matcher::new` refuses a pattern that could be read two ways, and placeholders that do not fit the rule's block and class.
- The first rule whose pattern matches decides. A failed shape, layer or expert check gives `Unknown`, and no later rule is tried.
- `Classification::Classified` also carries the rule's `BlockSel`, because the census groups norms with their block (RFC-0041 §3.1). For a stacked expert tensor, `class` is the first slice's class, `Expert { layer, id: 0 }`, and `stacked_experts` gives the count.
- `ClassSel` has one more variant, `Ignored { reason }`, for training-only tensors.
- `classify_tensors` returns a `TensorPlan` built from names and shapes alone. It lists the tensors to fetch, the ignored tensors with their reasons, and the unknown ones. A fetch reads only `TensorPlan::fetch`.

#### 4.2 Special cases

- **Tied embeddings**: no head tensor present, or the family's flag says tied. The census emits one record and two groups (`RowLookup` at the head, `Head` at the tail) linked by `shared_with` (RFC-0041 §3.1).
- **Fused tensors** (`attn_qkv`, `ffn_gate_up`, `ffn_gate_up_exps`): one record with role `Qkv` or `GateUp`; the block lowers to the fused op variant.
- **Experts stored per expert** (safetensors `…experts.E.gate_proj.weight`) and **stacked** (GGUF `…_exps`, gpt-oss-class `…experts.gate_up_proj_blocks`) end in the same per-expert records.
- **Router bias** (DeepSeek-V3-class `e_score_correction_bias`) belongs to the `Router` group.

#### 4.3 Ignored tensors

Multimodal towers and projectors (vision encoders, audio encoders, `mm_projector`-style adapters) and training-only tensors are `Ignored` with a reason, listed in `ModelManifest::ignored`, and never fetched. A model whose text path needs one of them, for example an image placeholder token that the template cannot avoid, is still served text-only; the API returns `Unsupported` for multimodal inputs (RFC-0050).

A family's classification rules match tower and projector tensors by name with `ClassSel::Encoder { tower }` and shape-check them like any other rule, then classify them `Ignored` with the reason "multimodal input arrives after 1.0" (§2.4). A later release changes that one outcome to `Classified { class: Encoder { tower }, … }`; the rules themselves, and the census for text-only use, do not change. A tower tensor that matches no rule is `Unknown` and refuses the load, as any unknown tensor does.

### 5. Family modules

#### 5.1 The `Family` trait

```rust
pub trait Family: Send + Sync {
    fn name(&self) -> &'static str;                                   // "deepseek3", "gemma", …; frozen tag (below)
    /// GGUF general.architecture values and HF config architectures/model_type values it accepts.
    fn accepts(&self) -> &'static [&'static str];
    /// Informational, shown in logs and the dashboard. Certification keys on the module's
    /// code hash (§9.4), never on this number.
    fn version(&self) -> u32;

    fn parse_hparams(&self, meta: &MetaView) -> Result<HyperParams, Error>;
    fn classify(&self, name: &str, shape: &[u64], hp: &HyperParams) -> Classification;
    fn build(&self, hp: &HyperParams, census: &TensorCensus) -> Result<ModelGraph, Error>;
    fn kv_formula(&self, hp: &HyperParams, src: SourceFormat) -> KvFormula;      // RFC-0022 §2 owns the type
    fn routing_known(&self, hp: &HyperParams) -> Vec<(LayerIx, RoutingKnown)>;  // RFC-0023 §2
    fn speculative_support(&self, hp: &HyperParams, census: &TensorCensus) -> SpeculativeSupport;
    fn tp_sharding(&self) -> Option<&'static dyn TpSharding> { None }           // §6.3; None: never TP
    fn tokenizer_spec(&self, meta: &MetaView) -> Result<TokenizerSpec, Error>;  // §8
    fn chat_format(&self, meta: &MetaView) -> ChatFormat;                        // §8.4
    fn embedding(&self, meta: &MetaView) -> Option<EmbeddingSpec> { None }      // §5.4
    fn required_ops(&self, hp: &HyperParams) -> OpSet;                           // RFC-0040 §11.1
    fn fixture_spec(&self) -> FamilySpec;                                        // RFC-0060 §12.1
}
```

`MetaView` is a typed view over GGUF key-values or `config.json` plus `generation_config.json`, with the same accessors for both, so `parse_hparams` is written once per family. Families are compiled in and registered statically. There are no plugins.

**Family tags are frozen.** `name()` is hashed into `ModelId` (RFC-0041 §1). A tag is never renamed and never reused for a different module; a module that changes what it computes for the same files keeps its tag only if the change is a bug fix certified by the harness, and otherwise registers a new tag. The registry test fails if a tag in the checked-in tag list disappears or is reassigned.

*Note 2026-10-23 (COMP-E16-T01, as built):* `syntropy_graph::families` holds the trait, the registry and the list.

- **The frozen list.** `FROZEN_TAGS` lists §5.5's ten module names in table order. Each tag is bound to the module path it registers from (`syntropy_graph::families::<tag>`). The tags are fixed before COMP-E10-T06 hashes any of them into `ModelId`. A listed tag may have no registered module yet.
- **The checks.** `Registry::check` fails in five cases:
  - a registered tag is unlisted;
  - a registered tag is bound to another module;
  - two registrations share a tag;
  - two families accept one architecture name;
  - the list repeats a tag or a module.

  `crates/graph/tests/family_tags.rs` pins the list, so the test fails if an entry is removed, renamed, reordered or rebound.
- **The trait.** The trait has the methods whose types exist: `name`, `accepts`, `version`, `parse_hparams`, `classify` and `required_ops`. The others are added with the tasks that add their types.
- **`MetaView`.** `MetaView` is an enum over the GGUF reader's metadata and `syntropy_manifest::config::HfConfig`. A family names each value once as a `Key`, with its GGUF key (`{arch}` reads as `general.architecture`) and its `config.json` or `generation_config.json` path.
- **Errors.** `UnknownArchitecture` (1) and `HyperParamInvalid` (3) are `FamilyError` variants, and `code()` gives the number. A wrong type, a missing required key and a value outside the family's range are all code 3. `core` has no `Domain::Graph` yet.
- **`HyperParams`.** `HyperParams` lives in `syntropy_manifest::hparams`, with the shared fields and `extra`. `LayerSpec` is added with the blocks.

*Note 2026-10-31 (COMP-E10-T07): `kv_formula` and `tokenizer_spec` join the trait.* The census needs both (RFC-0041 §3 step 8), and their types now exist.
- **`kv_formula(&self, hp, source: KvSource) -> KvFormula`.** The source parameter is `syntropy_manifest::kv::KvSource`, the type RFC-0042 §4.2's rule (`KvDTypeRule`) already takes, in place of the `SourceFormat` named above. It carries what the rule reads: GGUF, or safetensors with its compute dtype and whether the weights are quantised.
- **`tokenizer_spec(&self, sources: TokenizerSources) -> Result<TokenizerSpec, TokenizerSpecError>`.**
  - It takes the tokenizer side files, not a `MetaView`, because the spec is built from `tokenizer.json` and its companions (§8.1's note of 2026-10-22), which `MetaView` does not hold.
  - It has a default, `TokenizerSpec::from_sources`. A family overrides it only to supply what its checkpoints leave out.
  - GGUF sources, converted from metadata (§8.3), come with that conversion's task.

#### 5.2 Hyperparameters

`HyperParams` (RFC-0001 §5.3) keeps per-family values in `extra`. This RFC fixes the shared fields every family fills: `n_layers`, `hidden`, `vocab`, `residual_streams`, and per layer a `LayerSpec { mixer: MixerKind, ffn: FfnKind, mtp: bool }`, where `MixerKind` and `FfnKind` carry the block parameters of §3. `parse_hparams` normalises names and defaults, so the same model from GGUF and from safetensors yields equal `HyperParams`. `ModelId` does not hash `HyperParams` (RFC-0041 §1), so these types can grow between releases without changing any ID.

#### 5.3 KV formula and dtype

`kv_formula` returns one `LayerKv` per layer from the block table of §3.1, the `AuxKv` entries for indexers, and `KvFormula::dtype`: the reference's KV dtype per layer as RFC-0042 §4.2 defines it. A family may force BF16 where F16 would overflow, which RFC-0042 open question 4 anticipates. The conformance gate checks the formula against what the KV store actually allocates and reads (RFC-0060 §7.2 check 2).

#### 5.4 Embedding models

`embedding` returns `EmbeddingSpec { pooling: LastToken | Mean | Cls, normalize: bool, dims: u32 }` when the checkpoint declares an embedding output (RFC-0050 §14). The epilogue is then the `Pool` block (§3.3) in place of `LmHead`. Both kinds of embedding model ship when the embeddings API does, at M2 (RFC-0050):

- **Decoder-only embedding models on v1 families** (Qwen-class embedding checkpoints, for example): the family's own module with the `Pool` epilogue.
- **One encoder-only embedding family, `bert`.** Users ask for embeddings from small encoder models, which are cheap to serve on one node. It accepts BERT- and XLM-RoBERTa-class encoders, and encoder checkpoints with rotary positions and gated FFNs where their configuration fits the same blocks: `TokenTypeEmbed` and `Embed` in the prologue, `EncoderAttn` with `DenseFfn` (GELU or gated) per layer, `Pool` in the epilogue. Its tokenizers are WordPiece and Unigram, which §8.3 already converts.
  - **Single node only.** A `bert` model runs as one stage on one device: no pipeline split (its `BoundarySet` has no interior legal boundary), no tensor parallelism and no recovery path beyond restarting the request (RFC-0033 never applies to it). Its `FamilyCert` has `splits` and `tp` false.
  - **Two backends.** It is certified on the CPU provider and on CUDA, the first GPU provider complete for the M1 roster. Other backends reach it through ggml where ggml maps its ops, at the coverage level CI measures (RFC-0040 §11.2); none is promised.
  - **Execution.** An embedding request is a prefill-only sequence (RFC-0050 §14). Bidirectional attention needs every key before any query's output, so the whole input runs in one prefill step: the family's prefill buckets go up to its maximum input length, and a longer input is rejected rather than chunked.

**The IR covers bidirectional attention and pooling** (confirmed 2026-10-20; OQ-MF-15). No structural change is needed: an encoder-only model is a prologue, a list of layers and an epilogue like any other `ModelGraph`, with one residual stream. Two registered additions carry it: `AttnKind::Bidirectional`, attention without the causal mask over the step's own rows, and `OpKind::Pool(PoolKind)` with `PoolKind { Mean, Cls, LastToken }` (RFC-0040 §3.1). Both have `kernels::reference` semantics; ggml maps the first (`flash_attn_ext` without a mask), and the second is native on CPU and CUDA. `LayerNorm`, GELU and `RowGather` already exist.

#### 5.5 The v1 families

| Module | Accepts (examples) | Blocks | Family specifics |
|---|---|---|---|
| `llama` | Llama 3.x, Mistral dense (Medium 3.5), Qwen2/Qwen3 dense | `Gqa`, `DenseFfn`, `Embed`, `LmHead` | QK-norm (Qwen3), QKV bias (Qwen2), Llama-3 and YaRN RoPE scaling, tied embeddings on small sizes |
| `gemma` | Gemma 3, Gemma 4 dense and MoE, Gemma E-class | `Gqa`, `SlidingGqa`, `DenseFfn`, `Moe`, `Embed`, `PerLayerEmbed`, `LmHead` | sliding/global interleave (5:1 with window 1024 in Gemma 4), separate local and global RoPE bases, K = V sharing and 512-wide heads on global layers (31B), sandwich norms, embedding scaling, logit soft-cap where configured, GELU-gated FFN |
| `gptoss` | gpt-oss-20b/120b | `SlidingGqa` (sinks) alternating with `Gqa`, `Moe`, `LmHead` | 128-token window on alternate layers, attention sinks, MXFP4 experts, clamped SiLU-gated experts, softmax top-4 routing |
| `deepseek3` | DeepSeek-V3/R1/V3.x, Kimi K2/K2.5, Mistral Large 3, GLM-5-class where the configuration is DeepSeek-architecture | `Mla`, `SparseIndexed` (DSA-class variants), `DenseFfn` (first layers), `Moe`, `Mtp`, `LmHead` | sigmoid routing with per-expert bias and group-limited top-k, routed scaling factor, shared expert, FP8 block-scaled originals with declared dynamic activations (RFC-0042 §4.3 rule a), MTP layers |
| `deepseek4` | DeepSeek-V4-Flash/Pro | `Compressed` on every attention layer (r = 0, 4 or 128; the raw window is part of the block), with `AttnIndex(CompressedTopK)` as its indexer on r = 4 layers; `Moe` with `TokenHash` routing on the first `num_hash_layers` layers, `Mtp`, `HyperConn` where the configuration declares it | per-layer compression ratios (a `compress_ratios` list, revised 2026-10-23 from the reference at revision `b5968e9190ef611bbf34a7229255be88a0e937c1`, open question 2), single 512-dim KV head, FP4 experts with FP8 elsewhere, 1M context. DeepSeek-V4-Pro's configuration declares hyper-connections with `hc_mult = 4`, so four residual streams cross every stage boundary and its hop bytes and activation history are 4× a single-stream model's (OQ-MF-12; `config.json` read 2026-10-20) |
| `kimi3` | Kimi K3-class | `Kda`, `Mla` with output gate, latent `Moe` (896 + 2 shared, top-16), `LmHead` | MXFP4 weights in groups of 32, 69 KDA + 24 gated MLA layers. The checkpoint declares no activation quantisation (`input_activations: null`), so its tensor-core FP4 path falls under RFC-0042 §4.3 rule (c) and needs W4A8 certification per family; until then it widens with F32 activations (OQ-MF-13; `config.json` read 2026-10-20) |
| `qwen_moe` | Qwen3-MoE (GQA MoE) | `Gqa`, `Moe` | softmax top-8 routing, no shared expert |
| `qwen_hybrid` | Qwen3.5, Qwen3.6, Qwen3.8 hybrid MoE | `GatedDeltaNet`, `Gqa` with output gate, `BlockSparse` (QSA-class), `Moe` with shared expert, `NgramEmbed`, `Mtp` | 3:1 linear-to-attention patterns, 2048-token sparse budget, an n-gram table at the layer the configuration names |
| `glm` | GLM-4.5/4.6 (GQA MoE), GLM-5.x (MLA with indexed sparse attention), GLM-5.3-Flash-class (linear plus sparse) | `Gqa`, `Mla`, `SparseIndexed`, `GatedDeltaNet` or `Kda` as configured, `Moe`, `Mtp` | the module reads which attention form each layer uses from the configuration; where a GLM checkpoint is exactly DeepSeek-architecture, `deepseek3` accepts it instead |
| `bert` (M2) | BERT- and XLM-RoBERTa-class encoder-only embedding models | `TokenTypeEmbed`, `Embed`, `EncoderAttn`, `DenseFfn`, `Pool` | encoder-only, single node only, CPU and CUDA; absolute or rotary positions and pre- or post-norm read from the configuration (§5.4) |

Facts about each model come from its configuration (`backends-and-portability.md` §4.1). The table names what the module must handle; the conformance gate (§9) decides when each family ships on each backend (RFC-0040 §11.5).

*Note 2026-10-31 (COMP-E16-T03, as built): the `llama` module.* `syntropy_graph::families::llama` is registered in `graph::families()`.
- **Accepts.** GGUF `llama`, `qwen2` and `qwen3`, and `config.json` `LlamaForCausalLM`, `MistralForCausalLM`, `Qwen2ForCausalLM` and `Qwen3ForCausalLM`, with their `model_type` values. The name picks the variant: Qwen3 has QK-norm, Qwen2 has a bias on Q, K and V, and Llama and Mistral have neither. The variant is stored as the `qk_norm` and `qkv_bias` flags in `HyperParams::extra`. The norm and bias rules apply only where the flag is set, so a tensor the variant does not have is `Unknown`.
- **Equal across sources.** GGUF and `config.json` give equal `HyperParams`, with floats kept at F32 precision (§5.2). The one exception is tied embeddings: GGUF has no key for them, so a GGUF file reads as untied, and a GGUF file with no `output.weight` is tied by §4.2's first rule.
- **Tied with a stored head.** When `tie_word_embeddings` is set and the checkpoint also stores `lm_head.weight`, as Qwen3-0.6B does, the head reads the token table, as the reference does. The stored tensor is shape-checked and classified `Ignored`, so it is never fetched.
- **RoPE parameters.** §3.1 names the Llama-3-style and YaRN rules but lists no parameters. The `Gqa` block's `RopeScaling` now carries them, read from `config.json`'s `rope_scaling`:
  - Llama-3-style: `factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings`.
  - YaRN: `factor` and `original_max_position_embeddings` (default `max_position_embeddings`); `beta_fast` (default 32) and `beta_slow` (default 1); and `attention_factor`, carried only when stated.

  Each parameter becomes a `rope_*` attribute of `KvWrite` and of the attention op, with `rope_kind` codes 2 and 3.
- **Refused as `HyperParamInvalid` (3), never ignored:**
  - a RoPE rule other than plain, linear, Llama-3-style or YaRN;
  - YaRN with `mscale`, `mscale_all_dim` or `truncate: false`;
  - YaRN from GGUF, whose parameter keys this module does not map;
  - the newer `rope_parameters` layout;
  - a sliding window in effect;
  - a bias on O or in the FFN;
  - partial rotary;
  - an activation other than SiLU.
- **Llama-3-style scaling in GGUF.** GGUF stores it as a `rope_freqs.weight` table, which no rule classifies. So such a file is refused as `Unknown` until the `Rope` op can take a frequency table.
- **KV.** `kv_formula` gives `PerToken { 2 · n_kv_heads · d_head }` on every layer, with RFC-0042 §4.2's rule (`KvDTypeRule::reference`): F16 for GGUF and the compute dtype for a BF16 or F16 safetensors checkpoint.
- **`FamilySpec`.** `syntropy_graph::spec::FamilySpec` lists variants, because QK-norm, QKV bias and tied embeddings are fixed for a whole checkpoint. For `llama` there are four, which between them cover every rule and every RoPE kind. `llama::fixture_spec()` returns it. The trait method joins `Family` with the generator (QUAL-E7-T01).
- **Chat format.** The chat format is COMP-E17-T04's, which adds `ChatFormat`.
- **The golden census.** In CI it runs on a header derived from Qwen3-0.6B's published configuration: names and shapes only, with no file of the checkpoint in the repository. An opt-in test runs the same census on the fixture store's recorded Qwen3-0.6B files (QUAL-E7-T06), found through `SYNTROPY_QWEN3_0_6B_DIR`, against the same golden file. CI does not run that test.
- **Conformance checks.** RFC-0060 §7.2 checks 1–3 for `llama` are QUAL-E6-T06's and QUAL-E6-T22's: on the `nano` and `micro` fixtures, on Qwen3-0.6B's recorded headers (checks 1–2) and on its weights (check 3). See `dependency-fixes.md`, 2026-10-31.

### 6. Stage cut points, segments and tensor-parallel sharding

#### 6.1 Legal boundaries

A boundary `b` sits between layer `b − 1` and layer `b`; boundaries 0 (prologue only before it) and `n_layers` (epilogue only after it) always exist. `BoundarySet` is a bitset over `0..=n_layers` plus, per boundary, the side data that must cross it:

```rust
pub struct BoundarySet { pub legal: BitVec, pub side: Vec<SideData> }
pub struct SideData { pub token_ids: bool, pub layer_inputs: u32 /* elements per token */, pub bytes_per_token: u32 }
```

The rule, checked mechanically on the IR by `ir::validate`: **boundary b is legal when every value live across it is a residual stream or declared side data.** Declared side data is:

- **token IDs**, needed by later layers that route by token hash (`TokenHash`) or add n-gram rows (`NgramEmbed`);
- **per-layer inputs** computed in the prologue for later layers (`PerLayerEmbed`), `layer_inputs` elements per token per remaining layer.

Everything else makes the boundary illegal: a cross-layer KV-sharing group (a later layer reading an earlier layer's KV), an MTP module separated from the epilogue, or a block pair the family marks atomic. Recurrent state never crosses a boundary, because it belongs to its layer's stage (RFC-0022). `bytes_per_token` goes into the solver's hop size `S_hop` (cost model §3.7) next to the residual streams, and RFC-0031 carries it in `ActivationFrame::side` (RFC-0001 §5.8).

The solver's "blocks" are the runs of layers between consecutive legal boundaries (RFC-0011 §3). Linear and full-attention layers differ widely in cost and state, so the solver weighs each layer by its own cost rather than cutting by layer count (`backends-and-portability.md` §4.2).

#### 6.2 Segment building

```rust
pub fn segment(g: &ModelGraph, layers: Range<LayerIx>, role: StageRole, shape: ShapeClass) -> Result<SegmentIr, Error>;
```

A `SegmentIr` is the stage's part of the IR for one shape class:

| Role | Inputs | Contents | Outputs |
|---|---|---|---|
| head (`HEAD`) | token IDs from the step descriptor | prologue, layers `[0, a)` | residual streams and side data to the outbound frame |
| middle | the inbound frame's residual streams and side data (`StepDesc::in_payload`, RFC-0040 §3.5) | layers `[a, b)` | the outbound frame |
| tail (`TAIL`) | the inbound frame | layers `[b, n)`, epilogue, sampler ops (`LogitsProcess`, `TopKTopP`, `SampleSeeded`, `VerifyAccept`), MTP modules when `DRAFT` | accepted tokens, drafts |
| head and tail (one-stage plan) | token IDs | everything | tokens |

The segment names groups, not devices. RFC-0012 lowers it to a step program by binding each node to its group's compute site from the plan's `TierAssignment`s and inserting device hand-offs, doorbells and the MoE protocol (RFC-0040 §3.4). The segment is built for each shape class the plan uses (decode, verify window sizes, batch buckets, prefill chunk), as ADR-0003 decision 3 requires.

#### 6.3 Tensor-parallel sharding

A family that implements `TpSharding` may be placed with `IntraNodeTp`, `CrossNodeTp` or `CrossNodeEp` (RFC-0001 §5.4). A family without it never is (ADR-0003 decision 3).

```rust
pub trait TpSharding: Sync {
    /// Split one layer's blocks across `weights.len()` members, in proportion to `weights` (measured W_eff),
    /// at the block's legal granularity. Returns per-member slices of each group and the collective points.
    fn split(&self, layer: &LayerProgram, weights: &[f32]) -> Result<LayerSplit, Error>;
}
```

| Block | Split | Granularity | Collectives per layer |
|---|---|---|---|
| `Gqa`, `SlidingGqa` | QKV column-parallel, O row-parallel | whole KV-head groups (a KV head with its query heads) | all-reduce after O |
| `Mla` | per head: Q projections and `KV_B` column-parallel, O row-parallel; `KV_A`, its norm and the latent cache replicated | whole heads | all-reduce after O |
| `SparseIndexed`, `Compressed` | as the base attention; indexers and compressors replicated | whole heads | all-reduce after O |
| `GatedDeltaNet`, `Kda`, `Mamba2` | per head; convolution per channel group | whole heads | all-reduce after the out projection |
| `DenseFfn`, shared experts | gate/up column-parallel, down row-parallel | inter columns in multiples of the quant block and the kernel tile (the larger of the two) | all-reduce after down |
| `Moe`, tensor-parallel | every expert split like `DenseFfn`; router replicated | as `DenseFfn` | all-reduce after the combine |
| `Moe`, expert-parallel | whole experts owned per member (`ExpertOwnerMap`); router replicated; dispatch and combine | whole experts | dispatch before and combine after the experts (RFC-0032) |
| `Embed`, `LmHead` | replicated on the primary member; the LM head may be vocab-parallel only inside a node | — | an all-gather of logits when vocab-parallel |

Uneven splits weighted by measured `W_eff` are allowed at these granularities (RFC-0032 §5). Partial sums cross the collective in F32 in exact mode, and members reduce in member order, so every member holds the same bits (RFC-0040 §7.6; RFC-0042 §4.1). A BF16 collective needs the same certification as a BF16 wire (RFC-0042 §8.2).

v1 provides `TpSharding` for `llama`, `qwen_moe`, `deepseek3` and `gptoss` at M3, the families where cross-node TP and EP have the clearest cases in the cost model (§3.8, §4b1). Others gain it when their sharding passes conformance.

### 7. Adding a family

1. **Module.** Write `graph::families::<name>`: `accepts`, `parse_hparams`, classification rules, `build` from existing blocks, `kv_formula`, `routing_known`, `speculative_support`, `tokenizer_spec`, `chat_format`, `required_ops`, `fixture_spec`, and `TpSharding` if wanted.
2. **New block, only if needed.** A new attention, scan or FFN variant needs a block here, any new `OpKind`s with a `kernels::reference` implementation, a ggml mapping or an explicit "native only" mark, and kernels on the Tier 1 backends (RFC-0040 §3.1, §11). Until those kernels exist, the family can be certified only on the backends that have them.
3. **Fixtures.** Generate `nano` and `micro` fixtures from the `FamilySpec` (RFC-0060 §12.1) and choose one real checkpoint (§9.3).
4. **Golden data.** Run the publisher's reference code in F32 on the dequantised weights of the chosen checkpoint, and on the `micro` fixture where no checkpoint fits one host, to produce the model-definition oracle's golden logits and norms and the family's `τ_def` (RFC-0042 §7.1). Store them in the fixture store under its licence rule.
5. **Conformance.** Run the gate (§9), the oracle included. On pass, CI writes the family's `FamilyCert` into the release's `ConformanceManifest` (§9.4).
6. **Coverage.** CI computes the family's level per backend (RFC-0040 §11.2), and the release notes publish it.
7. **Approval.** Adding a family, or changing what a certified family computes, is a gated change class. A maintainer approves it explicitly (**Maintainer action**), after every gate above has passed.

**Who does it** (ADR-0021). Agents write family modules, their fixtures and any new block as tasks from `plan/implementation/`, through the gates: the parity and conformance harness of §9, the model-definition oracle against the publisher's reference code (RFC-0042 §7.1), and an independent review agent that checks the change against this section and the cited RFCs and is never the agent that wrote the module. Community family modules go through the same gates and review. No family is assigned to a named person; a maintainer approves each new family (step 7). Whatever merges, a family without a matching `FamilyCert` stays refused at load (§9.4).

**Rig hours per family.** Certifying a family, not writing it, is the scarce input: conformance, the oracle and η targets run on serial lab rigs (ADR-0021 decision 6). So the task that adds a family states, before work starts, its rig hours per Tier 1 backend and the untimed GPU time it needs (device class and hours) as `[estimate]` requirements, and records the measured values when the `FamilyCert` is issued. When rig time runs short, native kernels drop to ggml level G in the order of RFC-0001 §16's scope-cut list, Tier 2 backends first.

**Targets and the revisit trigger.**
- **Targets** [estimate, from §7's steps; replaced by the times measured for the M1–M3 families]: a Tier 1 family certified within 4 weeks of its public release when existing blocks suffice, and within 8 weeks when it needs a new block and Tier 1 kernels. RFC-0051's catalogue adds a new release within an already-certified family within 7 days [target].
- **The signal.** For every family, CI records the days from the family's public release to its first `FamilyCert`, and the rig-queue wait of its conformance jobs.
- **The trigger.** ADR-0003's revisit trigger, a Tier 1 family's time from public release to conformance exceeding one release cycle twice running, is measured from the family's **public release date**, not from the module's first commit (corrected 2026-10-20: measuring from the first commit meant a family started late could never trip it). One release cycle is the project's minor-release cadence, which open question OQ-PRC-19 settles.

### 8. Tokenizers and chat templates

The division of labour: this RFC owns the tokenizer **data** (what a model's tokenizer is, extracted at census time) and the library choice (ADR-0431). RFC-0050's `syntropy-text` crate runs tokenization, detokenization, template rendering and output parsing at request time.

#### 8.1 `TokenizerSpec`

```rust
pub struct TokenizerSpec {
    /// The tokenizer as a serialized `tokenizers` model (tokenizer.json form), so ingress
    /// loads it without re-converting. For safetensors sources it is the shipped
    /// tokenizer.json; for GGUF sources it is converted from metadata (§8.3).
    pub model: Vec<u8>,
    pub specials: SpecialTokens,            // bos, eos, every end-of-turn id, pad, unk, FIM prefix/middle/suffix
    pub add_bos: bool, pub add_eos: bool,
    pub chat_template: Option<String>,      // shipped template, verbatim
    pub template_source: TemplateSource,    // Shipped | FamilyDefault | UserOverride
    pub id: TokenizerId,                    // RFC-0041 §1: over `model`, specials and the template in use
}
```

*Note 2026-10-22 (COMP-E17-T01, `TokenizerSpec` for `tokenizer.json` sources as built):* `syntropy_manifest::tokenizer::TokenizerSpec::from_sources` builds the spec from a checkpoint's `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` and `chat_template.jinja`. The parts the struct above leaves open:
- **Fields are private, with accessors** of the same names, so `id` cannot fall out of step with what it covers. `with_template_source` is the one way to change the template in use, and it recomputes `id`. `TemplateSource` carries the template text for `FamilyDefault` and `UserOverride`, and has a fourth variant, `None`, for a checkpoint that ships no template when none is supplied. `chat_template` stays the shipped text either way.
- **Specials.** BOS, EOS, pad and unknown are `tokenizer_config.json`'s `bos_token`, `eos_token`, `pad_token` and `unk_token` (a string or an added-token object), resolved through `tokenizer.json`'s added tokens, then its vocabulary. The end-of-turn set is EOS together with `generation_config.json`'s `eos_token_id` (an integer or a list), sorted. FIM prefix, middle and suffix are added tokens spelled as in `FIM_SPELLINGS`, all three or none, whether or not `tokenizer.json` marks them special: the Qwen3-MoE repositories ship theirs as not special. No token is looked up by a family or model name.
- **`add_bos` and `add_eos`** come from `add_bos_token` and `add_eos_token` where the config sets them. Otherwise they are read from the post-processor: a `TemplateProcessing` single-sequence template that starts (ends) with a special token adds BOS (EOS).
- **The shipped template** is `chat_template.jinja` when present, else the config's `chat_template`: a string, or the entry named `default` of a list.
- **`TokenizerId`** is BLAKE3 over `"syntropy/tokenizer-id/v1"`, `model`, the specials, `add_bos`, `add_eos` and the template in use, with every field length-prefixed. No input depends on listing order. The specials are single IDs plus the end-of-turn set, which `TokenizerSpec::new` sorts and deduplicates. `model` is hashed as the shipped bytes, so its added tokens and maps are covered through the file and never re-encoded. The two flags are included because they change the tokens of a prompt. The source tag is not hashed, so an override equal to the shipped text keeps the ID. `TokenizerSpec::id` is the value RFC-0022's saved sessions record in their metadata; nothing keys on it (Open question 6, OQ-MF-14).
- **Errors.** A malformed or inconsistent side file is `TokenizerSpecError`, class `Integrity`, naming the file. No code is assigned yet; the codes are COMP-E10-T09's.

*Note 2026-10-31 (COMP-E17-T14, the real M0 repositories):* the specs of the M0 checkpoints' recorded files (RFC-0060 §12.2) are checked against IDs read from the files by hand, in `crates/manifest/tests/data/tokenizer/real-m0.json`. The test reads the files from the fixture cache, by their `testdata/fixtures.toml` names, and refuses a file pinned to another revision, bytes whose BLAKE3 differs, and a gated or non-OSI file without the lab credential. Qwen3-0.6B (`llama`) and Qwen3-30B-A3B (`qwen_moe`) ship byte-identical tokenizer files: BOS none (`bos_token` is null), EOS 151645, end-of-turn {151643, 151645}, pad 151643, FIM 151659–151661, neither flag set. Llama-3.2-1B's files wait for QUAL-E7-T15's import. A `tokenizer.json` whose `model.type` is outside `MODEL_TYPES` (BPE, Unigram, WordLevel, WordPiece) is now a `TokenizerSpecError`, because §8.2's engine could not load it; a file without a `type` is read as before.

#### 8.2 The engine

Tokenization uses the HF `tokenizers` crate, version-pinned and built without its C regex dependency (its pure-Rust regex feature), so it cross-compiles to every target in ADR-0020, including iOS and Android. Its licence (Apache-2.0) is compatible with ADR-0018. For safetensors sources it reads the shipped `tokenizer.json` directly, which gives the publisher's own behaviour.

#### 8.3 GGUF tokenizers

GGUF files carry the tokenizer as metadata. The census converts it into the same engine's model:

| `tokenizer.ggml.model` | Converted to | From |
|---|---|---|
| `gpt2` (byte-level BPE) | BPE with byte-level pre-tokenizer and decoder | `tokens`, `merges`; the pre-tokenizer regex comes from `tokenizer.ggml.pre` through a table of known pre-tokenizer names. An unknown name is `Unsupported (Graph, 6)`: guessing a regex silently changes tokens |
| `llama` (SentencePiece-style BPE) | BPE with byte fallback (`<0xNN>` tokens) and a whitespace-marker pre-tokenizer | `tokens`, `scores` (merges derived in score order, the standard SentencePiece-to-BPE conversion), `add_space_prefix` |
| `bert` | WordPiece | `tokens` |
| `t5` | Unigram | `tokens`, `scores` |
| others | — | `Unsupported` in v1 |

Token types (`tokenizer.ggml.token_type`: normal, unknown, control, user-defined, unused, byte) become added and special tokens. Special IDs, `add_bos` and `add_eos` come from their keys, and the template from `tokenizer.chat_template`.

*Note 2026-10-31 (COMP-E17-T03, the `gpt2` and `llama` conversions as built):* `syntropy_graph::tokenize::from_gguf` converts the metadata, and the census calls it for a GGUF source given no tokenizer side files. A file without `tokenizer.ggml.model` has no tokenizer. The parts the table above leaves open:
- **Pre-tokenizer names.** `PRE_TOKENIZERS` holds `gpt-2`, `llama-bpe` (Llama 3 and later) and `qwen2` (Qwen2 and later, the Qwen3 M0 checkpoints among them). Each entry gives the split regex, whether the source normalizes to NFC (`qwen2` does), BPE's `ignore_merges` (`llama-bpe` sets it) and `add_bos` when the file has no `add_bos_token` key. The model is a regex `Split` (`Isolated`), then a byte-level step, as the source repositories' files have it. A missing name is refused like an unknown one.
- **`llama`.** The whitespace marker is a normalizer, `Prepend("▁")` (when `add_space_prefix`, default true) then `Replace(" ", "▁")`, as in the source repositories' `tokenizer.json` files of this kind. So the marker is also prepended after each added token. Merges are ordered by the merged token's score, highest first, then by its ID, and within one token by the IDs of the left and right parts. A file with `remove_extra_whitespaces` set or a `precompiled_charsmap` is refused with code 6: neither is modelled. `bert` and `t5` are refused until COMP-E17-T08 (M1a).
- **Token types.** Normal and byte tokens are the BPE vocabulary. Unknown and control tokens become special added tokens, and user-defined tokens added tokens that are not special. Unused tokens get no entry, and every other ID is kept. A GGUF file records only the type. The exporter writes `<|…|>` spellings as control even where the source marks them not special, as for Qwen3's six FIM and repository tokens. So with special-token matching off (§8.7) the two paths differ on text that holds those tokens' spellings.
- **Specials.** The end-of-turn set is EOS with `eot_token_id` and `eom_token_id`. FIM IDs come from `fim_pre`/`fim_mid`/`fim_suf_token_id` or their older names, all three or none; otherwise from `FIM_SPELLINGS`, as in §8.1. A GGUF file does not carry `generation_config.json`'s EOS list. So the Qwen3 files give end-of-turn {151645}, where the `tokenizer.json` path gives {151643, 151645}, and the two paths' `TokenizerId`s differ.
- **Errors.** An unknown model or pre-tokenizer name, or an unmodelled setting, is `TokenizerUnsupported (Graph, 6)`. Metadata that does not describe a tokenizer is `TokenizeError::Invalid`, class `Integrity`, with no code yet, as for `TokenizerSpecError`.
- **Conformance.** The project corpus is `crates/graph/tests/data/tokenize/corpus.json` until RFC-0060's corpus is published (QUAL-E5-T22). The publisher's GGUF files of Qwen3-0.6B and Qwen3-30B-A3B tokenize it as their `tokenizer.json` does, token for token, and round-trip. The test is `crates/graph/tests/tokenize_real.rs`, and the measured run is in the task's change. The `llama-bpe` entry is written from the Llama 3 `tokenizer.json` layout. Its run on Llama-3.2-1B needs the gated files (QUAL-E7-T15).

#### 8.4 Chat format declared per family

```rust
pub struct ChatFormat {
    pub tool_calls: ToolCallFormat,         // one of RFC-0050 §6.2's patterns, with this family's markers
    pub tool_markers: SmallVec<[Marker; 4]>,// the pattern's begin/end markers, resolved against the tokenizer (§8.7)
    pub reasoning: Option<ReasoningSyntax>, // Markers { open: Marker, close: Marker, starts_open } | Channel { name }
    pub stop: SmallVec<[u32; 4]>,           // end-of-turn tokens beyond EOS, from the tokenizer's specials
}
pub enum ToolCallFormat { TaggedJson, SpecialTokenSections, TaggedKeyValue, ChannelAddressed, BareJsonList, None }
/// A marker is parsed on its token ID whenever the tokenizer defines it as a special token,
/// and on text only when it is ordinary text in this model's vocabulary (§8.7).
pub enum Marker { Special(u32), Text(&'static str) }
```

| Module | Tool calls (RFC-0050 §6.2) | Reasoning |
|---|---|---|
| `llama` | `TaggedJson`, or `BareJsonList` for Mistral-class checkpoints (chosen from the template) | markers where the template declares them |
| `gemma` | read from the shipped template at M1; `None` if it declares no syntax, in which case RFC-0050 falls back to grammar-constrained JSON | none |
| `gptoss` | `ChannelAddressed` | `Channel { name: "analysis" }` |
| `deepseek3`, `deepseek4`, `kimi3` | `SpecialTokenSections` | markers from the tokenizer's special tokens; `starts_open` where the template opens the section |
| `qwen_moe`, `qwen_hybrid` | `TaggedJson` | markers (`<think>` … `</think>` and variants) |
| `glm` | `TaggedKeyValue` | markers from the tokenizer's special tokens |

Marker strings are read from the tokenizer's special tokens and the template, never hard-coded per model size. Each marker is resolved at census time: `Marker::Special(id)` when the tokenizer lists its string as a special token, `Marker::Text` otherwise.

#### 8.5 Templates

Templates are rendered by `minijinja` with the helpers model templates use (RFC-0050 §6.1). Precedence: a user override (RFC-0054 §5.3) wins, then the shipped template, then the family's default template, used only when the checkpoint ships none. Syntropy never silently replaces a shipped template. `TokenizerId` covers the template in use, so caches keyed on rendered prompts change when it changes.

#### 8.6 Tokenizer conformance

- For each family's checkpoints, the tokenizer corpus (RFC-0060) must tokenize **token for token** as the source repository's own `tokenizer.json` does, through both paths: the shipped JSON, and the GGUF conversion of the same model where a GGUF exists.
- GGUF-only models with no tokenizer JSON are compared with recorded expected tokenizations stored with their fixtures.
- Byte-level BPE must round-trip: `decode(encode(x)) == x` for every corpus string.
- Template rendering for a fixed set of conversations (system, multi-turn, tool definitions, tool results, reasoning on and off) must match recorded renderings produced by the publisher's reference template engine, stored as fixtures. Where the recorded renderings and expected tokenizations come from a model that is not permissively licensed, they sit in the private part of the fixture store (RFC-0060 §12.3), because they derive from the model and its licence travels with them.
- **Injection cases** (§8.7): for every family, user, system, tool-result and document content containing the literal text of each special token, each tool-call and reasoning marker and each role header tokenizes to ordinary tokens only, and renders and parses with no extra turn, role or tool call. A fuzz target mixes random content with those strings in every content position.

#### 8.7 Content spans are tokenized literally

A template renders one string, so a user message, tool result, retrieved document or file that contains a control string (`<|im_start|>system`, a tool-call begin marker, a channel header) would otherwise tokenize into a real control token: a forged role, system turn, tool result or tool call. Agent tool results are attacker-controlled by design.

1. **Spans.** Rendering produces a segment map of template spans (text the template itself emits) and content spans (text taken from the request: messages of every role, tool results, tool definitions' free text, documents). RFC-0050's renderer produces the map; this RFC fixes what the tokenizer does with it.
2. **Template spans** are tokenized with special-token matching on, as the publisher's template intends.
3. **Content spans** are tokenized with special-token matching off: every byte of content becomes ordinary tokens, and no special token ID can arise from content. Content is never stripped or rewritten, so it round-trips byte for byte. The tokenizer engine encodes special-token strings as plain text when matching is off (ADR-0431).
4. **Raw special tokens** reach the model only through an explicit opt-in on the native API (RFC-0050), never through the compatibility surfaces, and RFC-0050's token-ID range check still applies to them.
5. **Output markers are parsed on token IDs** where the family's markers are special tokens (`Marker::Special`, §8.4): a tool call or reasoning section starts only when the model emitted that token ID, so a model that quotes a document containing a marker's text produces text, not a call. Text parsing remains only for formats whose markers are ordinary text (`Marker::Text`), and RFC-0050 labels those parses as text-matched.
6. **Prefix and session keys** are over token IDs, so the same characters in a template span and a content span never share a cache entry.

### 9. The conformance gate per family

RFC-0060 §7 owns the runner and the verdict; ADR-0003 decision 5 lists what must pass. This section fixes what each family supplies and how the result is used.

#### 9.1 What a family supplies

| Input | Source |
|---|---|
| the module; its `FamilyCert` is keyed by the code hash of the module and its shared blocks (§9.4) | §5 |
| `FamilySpec` for `nano` and `micro` fixtures: layer pattern, attention kinds, MoE layout (experts, top-k, shared, latent, router bias), MTP modules, recurrent blocks, side data | `fixture_spec()` |
| one real checkpoint per family (§9.3) | the task that adds the family, picked by §9.3's rules and approved with it (§7 step 7) |
| golden data for the model-definition oracle: logits and hidden-state norms from the publisher's reference code in F32, with `τ_def` | §7 step 4; RFC-0042 §7.1 |
| `KvFormula`, `BoundarySet`, `TpSharding` (optional), `RoutingKnown`, `ChatFormat`, tokenizer fixtures | §3–§8 |

#### 9.2 What must pass

RFC-0060 §7.2's checks (census, KV formula, single-device, split, tier and provider parity), plus:

- **the model-definition oracle** (RFC-0042 §7.1): `RefExecutor` against the family's golden data from the publisher's reference code, within `τ_def`. It gates every `FamilyCert` from M0, starting with `llama` and `qwen_moe`, and reruns on every change that alters the family's code hash (§9.4), which covers every change to the module or to a shared block it uses. The parity checks above cannot replace it: they compare two executions of the same module;
- the tokenizer and template conformance of §8.6, including its injection cases;
- for families with `TpSharding`, parity with `IntraNodeTp` on the lab's dual-GPU host and `CrossNodeTp` on the simulated fabric, at two and three members with uneven weights;
- for families with `TokenHash`, `NgramEmbed` or `PerLayerEmbed`, split parity at a boundary that forces side data across the hop.

#### 9.3 Real checkpoints

This RFC confirms RFC-0060 §12.2's candidate roster and makes the two picks it leaves open:

- **Qwen hybrid:** Qwen3.6-35B-A3B (30 Gated DeltaNet and 10 attention layers, `backends-and-portability.md` §4.1), the smallest hybrid in the research tables.
- **GLM-class:** GLM-4.7-Flash, once QUAL's desk check (QUAL-E7-T06) confirms its MIT licence on the model card (OQ-MF-04). Until the family's conformance work starts at M3, the `micro` fixture covers it nightly.
- **Licence rule for the roster** (OQ-MF-04): nightly fixtures use ungated, OSI-licensed checkpoints, quantised from the original weights, so DeepSeek-V2-Lite gives way to Moonlight-16B-A3B and Gemma 3 1B to a Gemma 4 E-class model; Llama-licensed checkpoints are used in the lab only. RFC-0060 §12.2 holds the roster.
- **`bert`:** an encoder-only embedding checkpoint picked by the same desk check before its M2 conformance work.

DeepSeek-V4-class and Kimi-K3-class have no small checkpoints. They are covered by `micro` fixtures nightly, and by the full checkpoints under RFC-0060's sampled-layer rules on the lab's large-memory Macs per release candidate. Their oracle golden data comes from the publisher's code run on the `micro` fixture and from single-layer outputs on the full checkpoint (RFC-0042 §7.1).

#### 9.4 Certification and fail-closed loading

A family is certified by a `FamilyCert` in the release's signed `ConformanceManifest` (ADR-0602; RFC-0060 §6.4):

```rust
pub struct FamilyCert { pub family: ArchFamily, pub code: Blake3, pub quants: SmallVec<[QuantClass; 8]>,
                        pub splits: bool, pub tp: bool, pub tiers: TierShapes,
                        pub golden: Blake3,     // the oracle golden data the certificate passed against (RFC-0042 §7.1)
                        pub max_ctx: u32 }      // longest context at which parity passed (RFC-0042 §5.7)
```

`golden` names the golden data set by its hash, so a change of checkpoint revision or of the publisher's reference commit is visible in the manifest. `max_ctx` bounds the exact label: positions past it are served labelled beyond the certified context, with `beyond_certified_from` set (RFC-0042 §5.7).

`code` is the BLAKE3 of the family module's source files and of every shared block it uses. `graph`'s build script computes it and embeds it as `graph::families::<f>::CODE_HASH`. The certificate therefore stays valid for every build in which those bytes are unchanged, and a change to one shared block invalidates exactly the families that use it.

**The registry is gated on it.** `graph::families()` returns a family only when the manifest holds a `FamilyCert` for it whose `code` equals the compiled-in hash. At load:

- an architecture no module accepts is refused (`(Graph, 1)`);
- a family with no matching `FamilyCert` is refused (`(Graph, 2)`), naming it;
- a quantization class outside `quants` is refused for that family, naming the class;
- the solver uses `splits` and `tp` to decide whether it may cut the family across stages or place it with tensor parallelism, and `tiers` for the residency shapes (RFC-0011; RFC-0060 §6.4). A certified family on a backend or placement its certificates do not cover is filtered, not refused (RFC-0040 §11.3);
- `--experimental-family` loads a family without a certificate on one device only, labelled experimental in the API, CLI and dashboard, and never placed across nodes (ADR-0003 decision 6). A build with no valid manifest behaves as ADR-0602 decision 6 states.

`Family::version` is informational: it appears in logs and the dashboard, and certification never keys on it.

### 10. Configuration and error codes

| Key | Default | Meaning |
|---|---|---|
| `models.experimental_family` | `false` | allow one family without a `FamilyCert` on one device (§9.4) |
| `models.<name>.chat_template` | unset | template override (RFC-0054 §5.3) |
| `models.<name>.kv_dtype` | family default | force BF16 KV where F16 overflows (RFC-0042 §4.2); changes `KvFormula::dtype` and therefore prefix and session keys |

*Note 2026-10-28 (COMP-E13-T06):* RFC-0054 §5.1 keeps per-model settings out of `models.*`, so `models.<name>.kv_dtype` is the instance key `kv_dtype` of a `[policy."<selector>"]` table (RFC-0054 §5.4), as `models.<name>.chat_template` is `chat_template` there. It takes `f16`, `bf16` or `f32`; unset, the family's reference dtype stands. It replaces only the source's default dtype in `KvFormula::dtype` (`KvDTypeRule::resolve`). It never replaces a family's requirement, for every layer or per layer, or a declared FP8 cache. It never narrows the default, where F16 over a BF16 default counts as narrowing; BF16 over F16, this key's sanctioned use, does not (RFC-0042 §4.2, note of 2026-10-28). Each of those is refused with an error that names `approximate.kv_lossy` where it applies. The row above keeps its old name for the record.

Codes are `(Domain::Graph, n)`:

| Code | Name | Class | Raised when |
|---|---|---|---|
| 1 | `UnknownArchitecture` | `Unsupported` | no module accepts the architecture name |
| 2 | `FamilyNotCertified` | `Unsupported` | the manifest holds no `FamilyCert` for the family whose `code` matches this build's module, or the quantization class is not in `quants` |
| 3 | `HyperParamInvalid` | `Unsupported` | a required key is missing or out of range |
| 4 | `GraphInvalid` | `Internal` | `ir::validate` fails on a built graph (a module bug) |
| 5 | `BoundaryIllegal` | `Internal` | a plan cuts at an illegal boundary (a solver bug; the plan is rejected) |
| 6 | `TokenizerUnsupported` | `Unsupported` | an unknown GGUF tokenizer model or pre-tokenizer name |
| 7 | `TemplateError` | `InvalidRequest` | template rendering fails for a request (raised by `syntropy-text`, code reserved here) |

## Performance model and budgets

Nothing in this RFC runs per token. Its performance effects are indirect, and all figures below are **targets** or **estimates**:

- **Graph build and validation** run once per (family, hyperparameters) at census time and again per segment and shape class at `Warming`. Target: ≤ 100 ms for building and validating the largest v1 graph (Kimi-K3-class, 93 layers) on one core [target, set so it never shows up in time to first token]. `GroupSel::Experts` keeps expert groups out of the node list, so graph size grows with layers, not with experts.
- **Hop size.** Side data adds to `S_hop` (cost model §3.7). Token IDs cost 4 bytes per row. Per-layer inputs are larger: at 256 elements per layer per token in F32, a boundary early in a 30-layer Gemma-E-class model carries ≈ 30 KiB per token next to the residual stream [estimate: 30 × 256 × 4 B]. The solver sees `bytes_per_token` per boundary and prices it (RFC-0011).
- **TP granularity.** Splits at whole KV-head groups and quant-block tiles limit how evenly heterogeneous members can be balanced. For a GQA layer with 8 KV heads over two members weighted 0.6:0.4, the closest split is 5:3 heads (0.625:0.375), a 4 % imbalance on attention [estimate], which RFC-0011's cost model includes.
- **Tokenizer conversion** of a 256 K-entry GGUF vocabulary happens once per `TokenizerId` at census time and is cached. Request-time tokenization belongs to RFC-0050.

## Alternatives considered

| Alternative | Why not |
|---|---|
| A general tensor dataflow graph (ONNX-like), with layers recovered by analysis | Loses the group, layer and boundary structure every consumer needs; boundary legality and census classes would have to be inferred, and inference fails open (ADR-0430) |
| Control-flow nodes for routing and speculation | Data-dependent control flow breaks whole-step capture; routing inside ops and trees as rows keep the graph static (RFC-0040 §4.4) |
| Per-model engines | The single-model trap (`tiered-memory-inference.md` §3.2) |
| A declarative family format | ADR-0003 option C |
| A Syntropy-owned tokenizer implementation | Re-implements BPE, SentencePiece-style BPE, WordPiece and Unigram plus normalisers, with token-exact parity against publishers' tokenizers as an open-ended task. The HF engine is the publishers' own behaviour for `tokenizer.json` models (ADR-0431) |
| A Syntropy-owned Jinja subset for templates | Model templates use a wide slice of Jinja; `minijinja` with Python-compatibility helpers already covers it (RFC-0050) |
| Slicing per-layer embedding tables per stage | Their rows interleave layers, so per-layer slices are strided and chunk-hostile; the head computes per-layer inputs and sends them as side data (§3.3, §6.1); revisited in Open questions |

## Testing and validation

| Claim | Verification |
|---|---|
| Graphs are well formed | `ir::validate` on every family at every fixture size and on recorded real-model headers; property tests generating random hyperparameters within each family's ranges |
| Families build what they claim | snapshot tests per family: per-layer node lists, op sets and weight roles, reviewed on change |
| Census rules are complete and fail closed | recorded headers of real checkpoints classify with no `Unknown`; a corrupted name or shape becomes `Unknown` and the load is refused (RFC-0041 §10) |
| Boundaries are right | for every legal boundary, the union of the two segments equals the full graph and the only crossing values are residual streams and declared side data; split parity at side-data boundaries (§9.2) |
| TP splits are exact | per block, recombining member partial outputs in member order reproduces the single-device result within op-level tolerance; uneven weights included |
| `KvFormula` is right | RFC-0060 §7.2 check 2, at contexts {1, 17, 4096} and every window edge |
| Tokenizers are token-exact | §8.6 on the tokenizer corpus for every family, both paths |
| Templates render correctly | §8.6 recorded renderings |
| Hyperparameter parsing is robust | fuzzing `parse_hparams` with hostile metadata values (RFC-0060) |
| Conformance | §9 per family per release |
| The module computes the published model | the model-definition oracle per family against the publisher's golden data within `τ_def`; each module-level planted defect (RFC-0042 §7.1) fails it while passing every parity check |
| Content cannot forge control tokens | §8.6's injection cases and fuzz target on every family: no special token ID from a content span, no forged turn, role or tool call; markers that are special tokens parse only on their IDs |
| Reserved hooks stay reserved | `ir::validate` rejects a v1 graph with an `EncoderProgram`, an `EmbedInject` node or an `Encoder` node; every v1 family's tower tensors classify `Ignored` with the post-1.0 reason |
| Bidirectional attention and pooling | `AttnPrefill(Bidirectional)` and `Pool` against `kernels::reference`; a `bert` checkpoint's embeddings match the publisher's reference code within `τ_def`; an input longer than the family's largest prefill bucket is rejected, never chunked |

## Rollout / milestones

| Milestone | Delivered |
|---|---|
| **M0** | IR types and validator, with the reserved multimodal hooks (§2.4); blocks `Gqa`, `DenseFfn`, `Embed`, `LmHead`, and `Moe` with softmax top-k routing and no shared expert; the `llama` family and the `qwen_moe` family (the MoE family RFC-0001 §16's M0 parity run needs, with every expert resident in VRAM, RFC-0023 M0), each certified through the model-definition oracle; `graph::reference`; single-stage segments; tokenizer engine with `tokenizer.json` and GGUF `gpt2`/`llama` conversions; literal tokenization of content spans with its injection cases (§8.7); chat formats for `llama` and `qwen_moe`, markers resolved to token IDs |
| **M1** | blocks `SlidingGqa`, `Mla`, `Moe` extensions (shared experts, sigmoid and group-limited routing, latent), `GatedDeltaNet`, `BlockSparse`, `Mtp`, `NgramEmbed`; families `gemma`, `gptoss`, `deepseek3`, `qwen_hybrid`; `RoutingKnown` for all MoE families; single-device and tier conformance for the M1 roster |
| **M2** | multi-stage segments; `BoundarySet` with side data; split conformance on the simulated fabric for the M1 roster; with the embeddings API, decoder-only embedding models and the encoder-only `bert` family on CPU and CUDA, single node only (`AttnKind::Bidirectional`, `Pool`, `EncoderAttn`, `TokenTypeEmbed`) |
| **M3** | blocks `Kda`, `Compressed`, `SparseIndexed`, `HyperConn`, `PerLayerEmbed`; families `kimi3`, `deepseek4`, `glm`; `TpSharding` for `llama`, `qwen_moe`, `deepseek3`, `gptoss`; every v1 family certified (ADR-0003 decision 4) |
| **M4** | `TpSharding` for the remaining families as they pass; new families on the release cadence |

## Open questions

1. ~~Per-layer inputs as side data~~ **Answered (2026-10-20; OQ-MF-11):** keep forwarding per-layer inputs as side data (≈ 30 KiB per token at an early boundary for Gemma-E-class models [estimate]); the M3 comparison spike against duplicating the table is dropped.
2. ~~DeepSeek-V4-class details~~ **Answered (2026-10-21; OQ-MF-12, COMP-E16-T15):** read from DeepSeek's reference code, `inference/model.py` in the Hugging Face repository `deepseek-ai/DeepSeek-V4-Pro` at revision `b5968e9190ef611bbf34a7229255be88a0e937c1` ([file](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py)). The file is unchanged since commit `a1fd202`, 2026-04-27, and line numbers below are at that revision. `deepseek-ai/DeepSeek-V4-Flash` at revision `60d8d70770c6776ff598c94bb586a859a38244f1` carries a byte-identical `model.py` (same SHA-256). This is a reading of the code, not a run of it.
   - **Configuration.** `config.json` at the same revision, unchanged since commit `4504094` (2026-04-27): `hc_mult = 4`, `hc_sinkhorn_iters = 20`, `hc_eps = 1e-6`; `index_n_heads = 64`, `index_head_dim = 128`, `index_topk = 1024`; and `compress_ratios` with 62 entries, one for each of the 61 layers and a last one for the MTP layer: 128 for layers 0 and 1, then 4 on even and 128 on odd layers from 2 to 60, and 0 for the MTP layer (30 layers at 4, 31 at 128). Beside them are `sliding_window = 128`, `head_dim = 512`, `num_key_value_heads = 1`, `qk_rope_head_dim = 64`, `compress_rope_theta = 160000` and `num_hash_layers = 3`. The reference reads `inference/config.json`, which has the same values for the fields it carries, some under other names (`window_size`, `n_hash_layers`, `rope_head_dim`). It has no `hc_eps` or `num_key_value_heads`: `hc_eps = 1e-6` is the code's default (line 80), and the single KV head is fixed in the code (`wkv` projects to one `head_dim`, line 460). V4-Flash's `config.json` also has `hc_mult = 4`, has `index_topk = 512`, and its 44-entry `compress_ratios` starts with two 0 entries.
   - **Three layer kinds**, by the layer's ratio r (`Attention.__init__`, lines 453–482). r = 0: no compression. r = 4: overlapping entries, read through the indexer. r = 128: entries without overlap, all of them read, no indexer. Every layer also keeps a raw window of the last 128 tokens, the current one included, each stored as one RMS-normalised 512-channel vector with RoPE on 64 channels and FP8 rounding on the rest (lines 502–507, 519–523, 530; `get_window_topk_idxs`, lines 254–265). Layers with r > 0 apply RoPE with `compress_rope_theta` and YaRN scaling; r = 0 layers use `rope_theta` without it (lines 475–481).
   - **Which tokens form entry j** (`Compressor.forward`, lines 316–377). Each token t gets a value `wkv(x_t)` and a per-channel score `wgate(x_t) + ape[t mod r]`, computed in F32: 512 channels, or two sets of 512 when r = 4 (`overlap`, line 290). For r = 128, entry j pools block j, tokens 128j to 128j + 127: per channel, a softmax over the block's scores weights the sum of its values. For r = 4, entry j pools 8 tokens, the first channel set of block j − 1 and the second of block j, under one softmax per channel (`overlap_transform`, lines 307–314; the decode path, lines 346–354). Entry 0 has no block −1, and that half gets score −∞.
   - **After pooling**, the entry is cast to BF16 and RMS-normalised, its last 64 channels get RoPE at position rj (the first token of its block), and its other 448 channels are rounded through FP8 E4M3 with one power-of-two scale per 64 channels and kept in BF16 (lines 362–372). It is stored at entry slot j (lines 373–376). One 512-element vector per entry is both key and value: the attention kernel reads the same row for both (`sparse_attn_kernel`, `inference/kernel.py` lines 325–343).
   - **When it forms, and remainders.** Entry j forms in the step that processes token r(j + 1) − 1 (line 344). A prompt of n tokens forms ⌊n / r⌋ entries at once (lines 326–328, 374). Tokens of an incomplete block wait in the compressor's buffers `kv_state` and `score_state` (lines 303–304), not in the cache; for r = 4 these buffers also keep block j − 1's first channel set for the next entry. Query position s reads entry j only when r(j + 1) ≤ s + 1 (`get_compress_topk_idxs`, lines 268–276; the indexer's mask, lines 424–430). The raw window is at least r long, so every token is in the window, in a formed entry, or in both. Nothing is de-duplicated between the two.
   - **The indexer** (r = 4 layers only; `Indexer`, lines 380–433) keeps its own compressed keys, 128 channels per entry, formed by the same rule with its own weights and followed by a Hadamard rotation and FP4 rounding (lines 398–399, 368–370). Its query is its own `wq_b` applied to the attention's low-rank query `qr`: 64 heads of 128 channels, with RoPE, the same rotation and FP4 rounding (lines 411–416). Entry j's score is the sum over heads of `relu(q_h · k_j)`, each head weighted by a per-token projection of the layer input (lines 418–421). The layer takes the top `min(index_topk, entries formed)` (line 427). Attention then runs one softmax, with a learned per-head sink, over the raw window and the selected entries (lines 456, 507–514, 528, 533).
   - **Hyper-connections.** The embedding is copied into `hc_mult` streams of 7,168 channels (line 805). Every block mixes them before and after attention and the FFN (`Block.hc_pre` and `hc_post`, lines 673–700), and the head reduces them to one (lines 728–735). Four streams cross every stage boundary, as §5.5 says.
   - **Where this plan differs from the code.** Tracked as OQ-MF-16, which COMP-E16-T18 settles for (a)–(c) and TIER-E6-T16 for (d) and (e). *Settled 2026-10-23 for (a)–(c) (COMP-E16-T18):* RFC-0040 §3.1 (`Compressed { sinks }` and the indexer rule) and §3.1 and §5.5 here hold the answer. (d) and (e) stay open for TIER-E6-T16. (a) §3.1 and RFC-0022 §2 give r as 4 or 128. Both models also use r = 0, a layer with the raw window only: the MTP layer, and layers 0 and 1 of V4-Flash. (b) §3.1's `Compressed` row gives every compressed layer an indexer top-k, and its `AttnDecode(Compressed)` has no sink. Only r = 4 layers have an indexer, r = 128 layers read all their entries (`selected: None`), and every layer's learned sink sits inside the one softmax over the window and the entries (`inference/kernel.py` line 346). (c) §5.5 lists `SparseIndexed` for the indexer, but V4's indexer picks compressed entries, not tokens, which is §3.1's `AttnIndex(CompressedTopK)`. (d) The indexer's keys are per entry, 128 elements per 4 tokens, while RFC-0022's `AuxKv` counts `elems_per_token` and indexes the `Aux` class by token page. (e) The compressor's buffers are per-sequence state outside the KV pages. RFC-0022 §6 rolls back formed entries only, and nothing yet carries the buffers through rollback, prefix reuse or saved sessions (RFC-0022 open question 7).
   - **Golden data.** At the start of a sequence the reference writes block j − 1 into the r = 4 buffers only when the prompt holds a whole block (lines 330–332), and nothing clears them between sequences. A sequence that starts with fewer than 4 tokens, on a model object that has already run another sequence, would mix the earlier sequence's last block into its entry 0. Golden data from the publisher's code (COMP-E16-T14) should use a fresh model object per sequence or prompts of at least 4 tokens. The reference also runs one prefill from position 0 and then one token per step (`generate.py` lines 50–51; `model.py` lines 347, 356 and 530 take a single token), so its golden data never forms an entry inside a multi-token step that starts later, such as a prefill chunk or a verify window, and cannot check gap (e) there. This is unconfirmed static analysis.
3. ~~GLM-class real checkpoint~~ **Answered (2026-10-20; OQ-MF-04):** GLM-4.7-Flash, subject to QUAL's licence desk check (§9.3).
4. ~~Kimi-K3-class declared activations~~ **Answered (2026-10-20; OQ-MF-13):** the checkpoint declares none (`input_activations: null`; MXFP4 weights in groups of 32), so its tensor-core FP4 path falls under RFC-0042 §4.3 rule (c) and needs W4A8 certification (§5.5).
5. **GGUF pre-tokenizer names.** The table of known names must follow ggml pin bumps (RFC-0040 §8.4); unknown names are refused until added. Settled as a pin-bump checklist item.
6. ~~Should `TokenizerId` key persisted sessions?~~ **Answered (2026-10-20; OQ-MF-14):** no. A saved session stores the `TokenizerId` in its metadata, and a resume under a different one warns but proceeds (RFC-0022 §8).

## Proposed amendments to RFC-0001

1. **§4.1, `graph` row.** "Depends on (workspace): `manifest`" should read "`manifest`, `kernels` (the op vocabulary and the reference op semantics)". The IR's nodes are `OpKind`s, and `graph::reference` interprets them with `kernels::reference` (RFC-0042 §7, RFC-0060 §4.1). Both crates are in lower layers, so rule 2 of §4.2 holds.
2. **§5.3, `HyperParams::layers`.** `LayerSpec` carries `MixerKind` and `FfnKind` with the block parameters of §3, and `mtp: bool` (also RFC-0041). These are additions, listed for readers.
3. **§5.8, `ActivationFrame::side`.** Besides token IDs, side data includes per-layer inputs for families with per-layer embeddings (§6.1). RFC-0031 owns the layout; the boundary's `bytes_per_token` tells it the size.
4. **§5.3, `ModelManifest`.** Add `ignored: Vec<(String, &'static str)>` for tensors classified `Ignored` (§4.3), and state that `TokenizerSpec` carries the serialized tokenizer model (§8.1).
5. **§5.3, `TensorClass`** (added 2026-10-20). Reserve `Encoder { tower: TowerIx }` for multimodal tower and projector tensors: dense, read once per encoded input. No v1 family produces it; v1 classifies those tensors `Ignored` (§2.4, §4.3).
6. **§5.7, `KernelClass` and `OpKey`** (added 2026-10-20). RFC-0040 §3.1 adds `AttnKind::Bidirectional`, `OpKind::Pool(PoolKind)` and the reserved `OpKind::EmbedInject`; `KernelClass::Attention(AttnKind)` therefore gains the bidirectional kind (§5.4, §2.4).
