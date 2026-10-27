# RFC-0022: KV cache, prefix cache and sessions

- Status: Draft
- Area: tiers (KV store)
- Depends on: RFC-0001 (§5.3 `KvFormula`, §5.4 `KvBudget`, §5.8, §5.9, §6.2–§6.4, §7.5, §8.4, §9.3, §10.3, §17 item 1), RFC-0011, RFC-0012, RFC-0020, RFC-0021, RFC-0023, RFC-0031, RFC-0032, RFC-0033, RFC-0040, RFC-0041, RFC-0042, RFC-0043, RFC-0050; ADR-0010, ADR-0017, ADR-0019, ADR-0205, ADR-0206, ADR-0207, ADR-0208
- Date: 2026-10-19
- Revised: 2026-10-20: prefix-cache scope per principal with client-marked shared spans (§7.1); incremental re-saves, an endurance budget (§8.1, §8.6) and session blobs sealed under per-blob content keys wrapped to the session's public key (§8.2); open questions 5 and 6 moved to the post-1.0 backlog

## Summary

Each stage's runner holds the KV cache and recurrent state of every live sequence for the layers it executes, in a paged store that spans `Vram`, `HostPinned` and `Nvme`. This RFC specifies:

- the per-layer KV layout for every attention type in the v1 families: GQA/MHA, MLA, sliding-window, sparse or indexed, compressed, and the fixed-size state of linear-attention and state-space layers;
- pages of 64 tokens (ADR-0205), grouped into page classes with one shared block table per class, laid out layer-first on the GPU and page-first on the host and NVMe;
- the hot window: which pages stay in VRAM, how the rest spill, and where attention over spilled pages runs: on the CPU where the pages sit with an exact merge, or on the GPU after streaming them, chosen per attention kind from a cost that counts the query heads sharing each KV element (ADR-0207);
- KV for speculative verify windows and trees, and recurrent-state snapshots;
- a prefix cache keyed by a chained BLAKE3 page hash, organised as a radix tree with token-granular partial tails, in which the head stage decides matches and evictions for the whole ring (ADR-0206);
- saved sessions as per-layer, 4 KiB-aligned blobs bound to the model content hash and a KV-settings hash, restorable under a different stage split (ADR-0208);
- opt-in lossy KV quantization as an approximate-mode knob;
- KV export and import for migration, re-placement and session transfer;
- KV budgets, the headroom that drives admission control, and the KV rungs of the memory-pressure ladder.

It also answers RFC-0001 §17 item 1: `KvHandle` covers recurrent state. State lives in the same store as a page class of its own, with the same handle, residency, export and persistence paths.

## Motivation and goals / non-goals

KV is the fourth access pattern after dense weights, experts and row lookups (`tiered-memory-inference.md` §3.1). On a home cluster it matters for two reasons (§12.2): agents re-send long shared prefixes, so prefix reuse decides TTFT; and KV competes with the expert cache for VRAM, so long contexts silently cost expert capacity (≈95 → ≈56 tok/s at 262K in the surveyed engine until KV was moved out of VRAM, §5.2, §12.1). The cost model's numbers: Llama-3.3-70B stores 320 KiB per token, DeepSeek-V3's MLA 68.6 KiB, and hybrid linear-attention models 20–30 KiB; every decode step reads the whole cache (`hardware-and-performance-model.md` §3.6, `backends-and-portability.md` §4.2). A surveyed engine restored 63K tokens of state (1.2 GB) in under a second against ≈30 s to re-prefill (§12.1).

**Goals**

1. **Exact by default.** In exact mode, KV is stored and moved bit-for-bit in the model's KV dtype across every tier, migration and session restore (ADR-0019 item 5). Attention over spilled pages passes the parity harness.
2. **Prefix contract.** Tokens prefilled = tokens after the longest cached-prefix match, exactly, at token granularity (SC-20).
3. **Fast resume.** A resumed session's TTFT ≤ 1.25 × `KV_bytes / W_read(tier)` + prefill of the new tokens (SC-19).
4. **One mechanism for every attention type**, including recurrent state, so pipelines, migration and sessions need no per-family code beyond the layout declaration.
5. **Bounded VRAM.** The hot window never exceeds `KvBudget`; long contexts spill rather than take expert capacity, as the plan decided.
6. **No allocation and no lock on the token path**: pages come from preallocated pools; page tables are preallocated per sequence slot.

**Non-goals**

- Choosing the KV budget per tier or the trade against the expert cache (RFC-0011). This RFC provides the cost terms the solver uses.
- The attention kernels themselves (RFC-0040), beyond the partial-attention-with-log-sum-exp contract they must meet (§5.3).
- Wire byte layouts (RFC-0031). This RFC defines the logical chunk format (§10).
- Approximate attention that changes which positions are attended (`approx_attention`, ADR-0019 §6). Only sparsity defined by the model itself is used in exact mode.
- KV on another node's RAM as a tier. It is designed for (§5.5) but not built before 1.0.

## Design

### 1. Ownership and structure

- The `kv` module of `syntropy-tier` holds one `KvStore` per stage hosted by the runner. The stage's **primary dispatcher** (RFC-0012) is the single writer of its KV pages and page tables (RFC-0001 invariant 4). CPU threads that compute partial attention over host pages read them; they never write.
- Pages live in arenas the tier manager builds: the `Kv` arena in `Vram` (or `Unified`) and KV spill pages in `HostPinned`, both sized from `KvBudget` (RFC-0020 §2). `Nvme` pages live in a preallocated spill file per runner per volume (RFC-0021 §13).
- Moving a page between tiers is a copy on the dispatcher's copy stream (VRAM ↔ host) or an I/O request (host ↔ NVMe). Completions return to the dispatcher, which flips the page's state.
- The housekeeping thread decides policy (what to spill, what to admit back, prefix-cache tiering) and posts commands to the dispatcher over an SPSC ring, as for the tier manager (RFC-0020 §1).
- The **head stage** additionally owns the prefix index's authority (§7) and the session index for sessions it has saved (§8).

### 2. KV layout per attention type

`KvFormula` (RFC-0001 §5.3) is owned here. Its `LayerKv` enum gains one variant and the formula gains an auxiliary list, both with serde defaults:

```rust
pub enum LayerKv {
    PerToken { elems_per_token: u32, window: Option<u32> },   // RFC-0001
    Fixed { elems_per_seq: u64 },                              // RFC-0001
    Sparse { elems_per_token: u32, selected: u32 },            // RFC-0001
    /// Added: one stored entry per `tokens_per_entry` tokens (compressed attention), with an
    /// optional window of raw entries and an optional selected count per step.
    Compressed { elems_per_entry: u32, tokens_per_entry: u16, window: Option<u32>, selected: Option<u32> },
}
pub struct KvFormula {
    pub layers: Vec<LayerKv>,                                   // RFC-0001
    #[serde(default)] pub aux: Vec<AuxKv>,                      // added: indexer caches and similar
    #[serde(default)] pub dtype: KvDTypeSpec,                   // added: the reference's KV dtype per layer
    #[serde(default)] pub layout_version: u16,                  // added: bumps invalidate prefixes and sessions
}
pub struct AuxKv { pub layer: LayerIx, pub kind: AuxKind, pub elems_per_token: u32, pub dtype: DType }
pub enum AuxKind { SparseIndexer }
```

Per-layer contents. "Elements" are in the layer's KV dtype; RFC-0043 supplies the numbers per family from the hyperparameters, and RFC-0040 owns the in-page element order (`KvLayoutId`) each attention kernel reads.

| Attention type | `LayerKv` | Per-token (or per-sequence) contents | Read per decode step | Example (cost model §3.6, `backends-and-portability.md` §4.2) |
|---|---|---|---|---|
| MHA / GQA | `PerToken { window: None }` | K and V: `2 · n_kv_heads · d_head` | the whole context | Llama-3.3-70B: 320 KiB/token over 80 layers, BF16 |
| Sliding-window | `PerToken { window: Some(w) }` | as GQA, at most `w` tokens kept | the last `w` tokens | gpt-oss-120b: 18 layers capped at 128 tokens |
| MLA | `PerToken` | latent `d_c` plus decoupled RoPE key `d_rope` (one per token, shared by heads) | the whole context | DeepSeek-V3: 61 × (512 + 64) elements = 68.6 KiB/token |
| Sparse / indexed (DSA-class, QSA-class, block-sparse) | `Sparse { selected }` plus an `AuxKv::SparseIndexer` | full per-token cache, plus the indexer's small per-token keys | the indexer's keys for the whole context, plus `selected` positions of the main cache | QSA-class: at most ≈2,048 positions; DSA-class: top-2048 |
| Compressed (per-layer compression ratio) | `Compressed { tokens_per_entry: r, window, selected }` | one compressed entry per r tokens, plus a short raw window | the compressed entries (or the selected ones) plus the raw window | DeepSeek V4-class: r = 4 or 128 per layer, raw window 128 |
| Linear attention, state space (Gated DeltaNet-, Mamba-class) | `Fixed { elems_per_seq }` | the recurrent state (and the short-convolution state where the family has one) | the state, once | Qwen3.5-class: 45 of 60 layers |

Rules:

- **KV dtype in exact mode** is the reference's KV dtype for the family (BF16 or F16, or FP8 where the model's own configuration specifies an FP8 cache), recorded in `KvFormula::dtype` by RFC-0043 and checked by the parity harness. Lowering it is the `kv_lossy` knob (§9).
- **Hybrid models** mix classes per layer; a stage's store holds whichever classes its layers need. Stage cut points therefore move state and KV of different classes together (§10).
- **Hyper-connection models** do not change KV; their extra residual streams travel in activation frames (RFC-0001 §5.8).

*Note 2026-10-28 (as built, TIER-E6-T01).* `KvFormula` is `syntropy_manifest::kv`, beside `HyperParams`. `stored` and `read_per_step` are the one implementation of these sizes, and `tier` and `solver` both depend on `manifest` (RFC-0001 §4.1). Rules the table above leaves open:
- **`KvDTypeSpec`** is one dtype for every layer plus per-layer exceptions. Its serde default is BF16 everywhere, and RFC-0042 §4.2's rule fills it per family and source (COMP-E13-T06).
- **Counting.**
  - A `Fixed` layer counts its state once per sequence, whatever the token count, 0 included.
  - A `Compressed` layer stores ⌊tokens / r⌋ entries, since an entry forms when its last token is processed (OQ-MF-12). It also stores a raw window of at most `window` tokens, each `elems_per_entry` elements wide.
  - A step reads every formed entry, or at most `selected`, plus the window.
  - An `AuxKv` stores and reads one entry per token over the whole context.
- **Lossy dtypes (§9)** apply to `PerToken`, `Sparse` and `Compressed` entries:
  - `Q8 { block }` is one byte per element plus an F16 scale per `block` elements.
  - `K8V4 { block }` stores the entry's first half as `Q8` and its second half at 4 bits, each with F16 scales.
  - Recurrent state keeps the reference dtype, and an auxiliary cache keeps its own dtype.
  - §9 defines `K8V4` by K and V planes only. An entry that is not a K/V pair, such as an MLA latent, is sized as `Q8`, the conservative upper bound, until OQ-IO-30 is settled. `LayerKv::PerToken` gains `latent: bool` with a serde default of `false` (an owner addition, like `Compressed`), and `LayerKv::mla` sets it. A `Sparse` or `Compressed` entry counts as a latent, since the known families store one there.

### 3. Pages, page classes and pools

**Page size.** A page holds `P = 64` consecutive token positions (ADR-0205), or, for `Compressed` layers, 64 consecutive entries. `kv.page_tokens` may be set to 16, 32, 64, 128 or 256 per instance; it is part of the KV-settings hash (§7.1), so prefixes and sessions never mix page sizes.

**Page classes.** Layers whose pages advance in lockstep share one block table. A stage has up to five classes:

| Class | Layers | Block table indexed by | Notes |
|---|---|---|---|
| `Full` | `PerToken { window: None }` and `Sparse` | token page `⌊pos / P⌋` | the common case |
| `Windowed(w)` | `PerToken { window: Some(w) }`, one class per distinct `w` | token page, modulo `⌈w / P⌉ + 1` | a ring: the oldest page is reused once it leaves the window |
| `Compressed(r)` | `Compressed { tokens_per_entry: r }`, one class per distinct r | entry page `⌊pos / (r · P)⌋` | raw-window tokens live in a `Windowed` class of their own |
| `State` | `Fixed` | one slot per sequence (plus snapshots, §6) | a "page" is one layer's whole state |
| `Aux` | `AuxKv` caches | token page | small; always in the fastest tier |

**One page index for all layers of a class.** Within a class, page index `p` addresses the same token range in every layer of the stage. The GPU pool of a class is one array per layer, each with the same capacity, so a sequence's block table (page indexes in token order) serves every layer of the class. This is the **layer-first** layout: attention for layer l reads only layer l's array.

**Host and NVMe pages are page-first.** In `HostPinned` and on `Nvme`, page `p` of a class is one contiguous record holding that token range for every layer of the class, each layer's slice padded to 4 KiB. Moving a page between tiers is one large transfer instead of one per layer, which is what made the surveyed layer-first/page-first split up to 2× faster (`tiered-memory-inference.md` §12.2). The conversion is a gather on the way down and a scatter on the way up, done with the provider's batched copies (`cudaMemcpyBatchAsync` on CUDA 13) or a `kv_pack`/`kv_unpack` op (RFC-0040), on the copy stream.

**Pools.** For each class and each tier in `KvBudget` (RFC-0001 §5.4) there is a pool of `capacity` pages: `Vram` (the hot window), `HostPinned` (spill) and `Nvme` (spill file). The host spill tier is always `HostPinned`, never `HostPageable`: `GpuStream` and prefill streaming copy spill pages to the GPU, and every DMA source must be `DmaPinned` (RFC-0020 §7.5). RFC-0011 should restrict KV host placements to `HostPinned` (follow-up to PLC). Capacities are fixed per epoch from the budgets. Free pages are kept on a preallocated stack per pool. A page has one of these states:

```text
Free ──alloc──▶ Writable (the sequence's tail page; positions being appended)
Writable ──page full──▶ Sealed (immutable; may be shared, spilled, saved)
Sealed ──spill issued──▶ Moving{to, ticket} ──completion──▶ Sealed (in the new tier; old copy freed after its fence)
Sealed/Writable ──last reference dropped──▶ Free (after the step fence, RFC-0020 §7.4)
```

Sealed pages are immutable, so they can be shared by sequences (prefixes, forks) through a reference count, and moved without coordinating with readers beyond the step fence.

### 4. Types and the `KvStore` interface

Extends RFC-0001 §5.9. `KvHandle`'s fields are unchanged; `KvResidency`, `PageTableId` and the supporting types are defined here.

```rust
pub struct PageId { pub tier: u8, pub index: u32 }             // 8 bytes; tier is the node-local ordinal

pub enum PageClass { Full, Windowed { window: u32 }, Compressed { ratio: u16 }, State, Aux }

/// Per sequence and stage: one block table per class, preallocated for the sequence slot's
/// maximum context (`BatchingLimits` × context limit), so appends never allocate.
pub struct PageTable {
    pub classes: SmallVec<[ClassTable; 5]>,
}
pub struct ClassTable {
    pub class: PageClass,
    pub pages: Vec<PageRef>,       // capacity fixed at sequence-slot creation
    pub provisional: u16,          // positions appended but not committed (speculative, §6)
}
pub struct PageRef { pub page: PageId, pub shared: bool }      // shared: copy-on-write

/// RFC-0001 KvHandle::pages: index of the sequence slot's PageTable in the stage's store.
pub struct PageTableId(pub u32);
pub struct TailId(pub u32);                                     // index into the prefix index's tail pool
/// A run of a sequence's pages in one class, in token order.
pub struct PageSpan { pub class: PageClass, pub first_page: u32, pub pages: u32 }
pub struct MoveTicket(pub u64);                                 // completion arrives on the dispatcher's ring
/// Where restore reads a session blob from.
pub enum SessionSource { Local { manifest: PathBuf }, Remote { holders: SmallVec<[NodeId; 4]>, manifest: Blake3 } }
/// Leased sealed pages ready to stream as KvChunks (§10); dropping it releases the leases.
pub struct KvExport { pub seq: SequenceId, pub chunks: Vec<KvChunkDesc>, pub cut_at: u32, pub leases: Vec<CountedLease> }  // RFC-0020 §7.4 counted leases
pub struct KvChunkDesc { pub layer: LayerIx, pub class: PageClass, pub first_token: u32, pub n_tokens: u32, pub pages: PageSpan }

/// RFC-0001 KvHandle::residency.
pub struct KvResidency { pub vram: u32, pub host: u32, pub nvme: u32, pub moving: u32 }   // pages per tier

/// What the head found in its prefix index (§7) or session index (§8).
pub enum PrefixRef {
    Cached { last_full: PrefixKey, tokens: u32, tail: Option<TailId>, epoch: PlanEpoch },
    Session { session: SessionId, tokens: u32, blob: Blake3 },
}

pub enum KvDType {                                              // RFC-0001 names it; defined here
    Native,                                                     // the reference's KV dtype (exact)
    Q8 { block: u16 },                                          // approximate: int8, one F16 scale per block
    K8V4 { block: u16, rotate: bool },                          // approximate: int8 K, 4-bit V after a Hadamard rotation
}

impl KvStore {
    // RFC-0001 §5.9: open, append, truncate, fork, persist, release. Added:

    /// Head only: longest cached prefix of `tokens`, at token granularity (§7.3).
    pub fn lookup_prefix(&self, tokens: &[u32]) -> Option<PrefixRef>;
    /// Every stage: attach the head's PrefixRef to a new handle; fails with KV_PREFIX_MISS if
    /// this stage no longer holds it (§7.5).
    pub fn attach(&mut self, h: &mut KvHandle, prefix: &PrefixRef) -> Result<u32, Error>;
    /// Restore a saved session's pages for this stage's layers (§8.4).
    pub fn restore(&mut self, seq: SequenceId, session: SessionId, source: SessionSource) -> Result<KvHandle, Error>;
    /// Move sealed pages between tiers (housekeeping-driven, §5).
    pub fn spill(&mut self, h: &KvHandle, pages: PageSpan, to: TierId) -> Result<MoveTicket, Error>;
    pub fn fetch(&mut self, h: &KvHandle, pages: PageSpan, to: TierId) -> Result<MoveTicket, Error>;
    /// Migration and transfer (§10).
    pub fn export(&self, h: &KvHandle, layers: Range<LayerIx>, tokens: Range<u32>) -> Result<KvExport, Error>;
    pub fn import(&mut self, seq: SequenceId, layers: Range<LayerIx>, chunks: &mut dyn Iterator<Item = KvChunk>) -> Result<KvHandle, Error>;
    /// Admission control input (§11).
    pub fn headroom(&self) -> KvHeadroom;
    /// Apply a KvBudget change from a PlanDelta (§11).
    pub fn apply_budget(&mut self, b: &KvBudget) -> Result<(), Error>;
}
```

### 5. Tiering: hot window, spill and attention over spilled pages

#### 5.1 Which pages are hot

The `Vram` pool of each class is the **hot window**; its size is fixed by `KvBudget` for the epoch. The policy differs by what a decode step reads:

| Class | Read per step | Hot set | Spill victim |
|---|---|---|---|
| `Full` (dense attention) | every page | each live sequence's newest pages, plus its first page (attention sinks); idle sequences' pages go first | the oldest sealed pages of the sequence holding the most VRAM pages above its share |
| `Full` with `Sparse` layers | the indexer's selected pages | pages by decayed selection frequency (decay 0.8 per step, the same machinery as the expert cache's decayed LFU, RFC-0023 §5), plus the newest page | the lowest decayed count |
| `Windowed` | the window | always all of it (the ring is small) | never spilled |
| `Compressed` | entries or selected entries | as `Full` or sparse `Full`, by whether the layer selects | as above |
| `State` | the state | always hot | never spilled while the sequence is live; spilled whole when it is idle |
| `Aux` | the indexer keys | always hot | never spilled while live |

A sequence's **share** of the hot window in latency mode is everything not needed by others: the interactive sequence first, then others in priority order. In throughput mode each live sequence gets `budget / live` pages, and a sequence above its share spills first.

**Idle sequences** (paused by back-pressure, RFC-0001 §8.4 rule 3, or sessions between turns whose sequence slot is held) spill whole to `HostPinned` after `kv.idle_spill_after` (default 2 s idle), and to `Nvme` under pressure (§12).

#### 5.2 Moving pages

- **Down (spill).** The dispatcher gathers the page's per-layer slices into a `HostPinned` page (page-first) on the copy stream after the step fence of its last write; the VRAM page is freed once the copy completes. Host → NVMe is a write of the page record to the spill file at the page's slot (RFC-0021 §13), `Background` priority.
- **Up (fetch).** NVMe → host by a read request (`PrefetchNear` priority when a resume or prefix hit needs it now, `Demand` if a step is waiting); host → VRAM by a scatter copy. A page is visible to attention only after its copy's event has fired.
- **Bit-for-bit.** Pages move in their stored dtype with no conversion, so exact mode is preserved by construction (ADR-0019 item 5); the parity harness still covers it (§Testing).

#### 5.3 Attention over spilled pages: two compute sites, chosen by cost

A dense-attention step reads every page. When some of a sequence's pages are in `HostPinned`, there are two ways to attend over them (ADR-0207):

- **`GpuStream`**: copy the host pages to VRAM stream buffers on the copy stream and attend over everything on the GPU;
- **`CpuSplit`**: the CPU pool of the pages' NUMA domain attends over the host pages where they sit, the GPU attends over the VRAM pages, and the two partial results are merged exactly.

Which one wins depends on the attention kind, not only on DRAM against PCIe bandwidth. Each KV element is read by every query head that shares it, so the CPU does `2 · g_eff` FLOP per KV byte-pair per query row, where `g_eff` is the number of query heads served per stored element: `n_q_heads / n_kv_heads` for MHA and GQA, and for absorbed MLA about `n_heads · (2·d_c + d_rope) / (d_c + d_rope)`, ≈ 1.9 · n_heads. Per spilled byte and a pass of T query rows (decode T = 1; a verify window or batch, T > 1):

```text
t_split(T)  = B_host / W_cpu,attn(kind, T)                                         CPU side; runs concurrently with the GPU's VRAM part
W_cpu,attn(kind, T) = min( W_cpu ,  b_kv · F_cpu,attn / (2 · g_eff · T) )         KV bytes per second the CPU's AttnPartial consumes at T rows
t_stream    = B_host / W_pcie                                                     copy; overlapped with the GPU's VRAM part
T*_kv       = b_kv · F_cpu,attn / (2 · g_eff · W_pcie)                             CpuSplit wins only for T < T*_kv (and W_cpu > W_pcie)
```

Worked values [estimate], with the cost model's F_cpu = 1 TFLOPS, `b_kv` = 2 (BF16) and `W_pcie` = 25 GB/s:
- Llama-3.3-70B, GQA with g = 8: `T*_kv` ≈ 2·10¹² / (2 · 8 · 25·10⁹) = 5. Decode (T = 1) is memory-bound on the CPU, so `CpuSplit` wins; a verify window of 5 or more rows should stream.
- DeepSeek-V3, absorbed MLA with 128 heads: `g_eff` ≈ 243, so ≈ 242 FLOP per stored byte and `T*_kv` ≈ 0.16. `GpuStream` wins even at T = 1, because the CPU would consume KV at only 4–8 GB/s against 24 GB/s over PCIe.

**The compute site is a plan decision.** `KvBudget` gains `#[serde(default)] spill_compute: SpillCompute` per (stage, class), chosen by the solver from these terms (RFC-0011). **What is measured**: RFC-0010 measures `W_cpu,attn(kind, T)`, in KV bytes per second, at T = 1 and at the plan's verify-window size; `F_cpu,attn` is derived from the two points (at the larger T, `F_cpu,attn = 2 · g_eff · T · W_cpu,attn(kind, T) / b_kv` once that point is compute-bound), and `T*_kv` is the T at which `W_cpu,attn(kind, T)` falls to `W_pcie`. Every RFC uses these names and units:

```rust
pub enum SpillCompute {
    /// Host pages are attended by the CPU for passes of fewer than `max_rows` query rows,
    /// and streamed for larger passes. `max_rows` starts at the plan's T*_kv and is refined
    /// online from live timings, exactly as RFC-0020 §10 refines n*.
    CpuSplit { max_rows: u16 },
    /// Host pages are always streamed to the GPU.
    GpuStream,
}
```

Defaults when unmeasured: `GpuStream` for MLA and for any kind whose `g_eff ≥ 32`; `CpuSplit { max_rows: T*_kv }` for GQA and MHA. Sparse and indexed kinds use `GpuStream` until their `AttnPartial` kernels exist (M3, RFC-0040). The runtime executes what the plan says and reports drift if the measured crossover moves by more than 20%.

**Only certified spill paths in exact mode** (RFC-0060 §6.3–§6.4). Whether a page spills, and by which path, is a run-time dimension the solver cannot check, so the tier manager enforces it. Before a sequence's KV spills to a tier, the KV store checks the plan's certified run-time values and the `ConformanceManifest` for the current provider and device class, the attention kind, the spill path (`CpuSplit`, which needs `OpCert`s for `AttnPartial(kind)` and `AttnMerge`, or `GpuStream`) and the target tier (`HostPinned`, `Nvme`). For a sequence whose effective policy is `Exact`:
1. the plan's `spill_compute` path, if certified;
2. otherwise the other path, if it is certified (a certified substitute: a performance degradation, reported);
3. otherwise no spill to that tier: pages stay in the higher tier, the tier is left out of the stage's `KvHeadroom` (§11), and a sequence that cannot grow is paused or fails with `Capacity` under §11's rule.

An uncertified spill path is never a silent fallback. Only a sequence whose effective policy is already `Approximate` (ADR-0019 §7, RFC-0001 §11.3) may use one, and its responses then carry the degradation. Every refusal or substitution is reported as a degradation in the response's fidelity metadata (RFC-0050) and counted in telemetry (`KV_SPILL_UNCERTIFIED`, §14). The current outcome per (stage, class, tier) is published as `KvSpillOutcome` in the residency digest (RFC-0020 §13), so the dashboard and `syntropy doctor` can show which spill path each tier uses.

**`GpuStream` mechanics.** Per layer, the host pages of the pass are copied into a VRAM stream buffer in `Scratch` (double-buffered across layers, sized for the plan's largest spilled span per layer) on the copy stream, and the attention kernel reads VRAM pages and the stream buffer through one page list. Sources are KV spill pages, which are `DmaPinned` (RFC-0020 §2). Prefill chunks always stream.

**`CpuSplit` mechanics.**
- *Ops* (RFC-0040 adds them to the closed op list): `AttnPartial(AttnKind)` returns, per query row and head, the unnormalised output `o` (F32), the running maximum `m` and the softmax denominator `l`; `AttnMerge` combines two partials and an optional per-head sink logit `s` for families with learned attention sinks:

```text
m = max(m_1, m_2, s)                                   (s = −∞ when the family has no sinks)
o = (e^(m_1 − m) · o_1 + e^(m_2 − m) · o_2) / (e^(m_1 − m) · l_1 + e^(m_2 − m) · l_2 + e^(s − m))
```

  The sink enters the denominator exactly once and contributes no value. This is the rescaling a blocked attention kernel performs between blocks, so the result equals single-device attention up to rounding; the parity harness decides that it passes (ADR-0019 item 4). For MLA (if a plan ever selects `CpuSplit` for it) the CPU works in the latent space and the GPU applies the value up-projection after the merge.
- *Empty side.* A side with no pages in the pass returns `o = 0`, `m = −∞`, `l = 0`, and always raises its flag, so the GPU's wait never depends on page counts.
- *Hand-off buffers.* A host `Scratch` sub-arena, `AttnHandoff`, registered `GPU_MAPPED`, double-buffered across layers. Per layer it holds the query, `rows × n_q_heads × d_q × 4` bytes (F32), and the partials, `rows × n_q_heads × (d_v + 2) × 4` bytes per NUMA domain [estimate for Llama-3.3-70B at T = 1: query 64 × 128 × 4 = 32 KiB]. Its size is part of `plan.scratch`.
- *Doorbell.* A per-layer `AttnDoorbell { step: u32, rows: u16, host_pages: u32, selected: [[u32; selected]; rows] }` block (the `selected` array only for sparse kinds: the selected host **positions** per row, bounded by the layer's `selected` count, because token-level selection can touch up to `selected` distinct pages) next to, not inside, the MoE `DoorbellLayer` (RFC-0023 §2). The GPU writes the query and rings it after projection and RoPE; the CPU pool computes, writes `(o, m, l)` and its flag; `AttnMerge` waits on the flags device-side. The flags follow RFC-0040 §4.2's constant-valued consumer-reset handshake, as the MoE doorbells do (RFC-0023 §6): the GPU sets `Ra(l, d) = 1` for each NUMA domain's pool, domain d writes `Qa(l, d) = 1` every step, even when its side is empty, the consumer of each flag resets it to 0 after its `EQ 1` wait, and `step` is a payload check, not a flag. No captured value depends on the step, and no host synchronisation is added (RFC-0001 invariant 9).

**Sparse layers.** The indexer runs on the GPU over the `Aux` cache, which is always hot. Selected pages resident in VRAM are attended by the GPU; selected host pages are handled by the class's `spill_compute`. The selection is computed on the GPU inside the step, so under `CpuSplit` (M3) the GPU also writes the selected host positions per row into the attention doorbell, at most the layer's `selected` count per row; in M1 sparse kinds stream. Selections recur from step to step, so the frequency-ordered hot set of §5.1 keeps most of them in VRAM, and admission copies of newly hot pages are limited per step like expert admissions (RFC-0021 §8, `kv.admit_pcie_share` = 0.10). This is the case where spilling is cheap: only the few selected pages that miss VRAM move or are computed, whatever the context length (the surveyed engine's sparse-attention model, `tiered-memory-inference.md` §12.1).

**Unified-memory devices** have neither path: the KV pages are in the `Unified` tier the GPU reads, and the next tier down is `Nvme` (§5.2).

#### 5.4 What the solver sees

Per stage and class, for a pass of T query rows (cost model §3.6, generalised; `B_vram` and `B_host` from `KvFormula::read_per_step` split by the hot window):

```text
CpuSplit:   t_kv = max( B_vram / W_gpu ,  B_host / W_cpu,attn(kind, T) ) + t_switch
GpuStream:  t_kv = max( B_vram / W_gpu ,  B_host / W_pcie ) + B_host / W_gpu
```

These are the canonical forms. `GpuStream` overlaps the copy of host pages with the GPU's attention over VRAM pages and then attends over the streamed pages, so it is a maximum plus a term, not a sum; RFC-0011 C7 should model each form with an epigraph variable (`τ_kv ≥` each term), as C8 does for routed time, rather than the additive form, which overstates `GpuStream` by up to 2× when the two terms are equal. RFC-0011 chooses `spill_compute` per (stage, class) by these terms and uses the same terms in its objective, so the plan predicts the path the runtime executes. It trades the hot window against the expert cache with them: shrinking the hot window by ΔB adds the class's spill cost for ΔB and adds `ΔB / s_expert` experts to the VRAM cache. For dense MLA this trade rarely pays (§Performance); for sparse and hybrid models it often does.

#### 5.5 KV on another node's RAM

The design allows a `KvBudget::spill` tier on another node (pooling spare DRAM across the cluster, `tiered-memory-inference.md` §12.2): pages would move over the bulk plane and be fetched back before use. It is not built before 1.0: a 2.5 GbE link (≈0.3 GB/s) is slower than a local NVMe spill at several GB/s, so it pays only on RDMA-class links, which M3 introduces. Open questions.

### 6. Speculative windows, trees and recurrent-state snapshots

ADR-0017 makes speculative decoding first-class; RFC-0012 runs it. The KV store provides the slots and the rollback.

**Reservation.** Each sequence slot reserves room for `T_max` provisional positions in every token-indexed class, where `T_max` is the plan's verify window `γ_max + 1` or its tree size (RFC-0001 §5.4 `SpeculativeConfig`). `T_max` is the epoch's upper bound: a `PlanDelta` may lower the window but never raise it past the slots reserved at `Warming`, because nothing is allocated after `Warming` (RFC-0001 invariant 9); raising it needs a new epoch. The reservation is part of the sequence's footprint for admission (§11). `Windowed` rings keep `w + T_max` positions so a rejected tail can be dropped without losing positions still inside the window.

**Chain windows.** A verify frame for positions `p..p+γ` appends them provisionally (`ClassTable::provisional`). When the tail accepts `j` drafts, positions up to `p+j` commit; the next frame starts at `p+j+1`, and its arrival truncates everything from there (implicit truncation, RFC-0001 §6.4). Truncation only moves the commit mark and returns whole pages emptied by it.

**Trees.** Tree nodes take provisional slots in node order, and the attention kernel applies the tree mask (RFC-0040). After verification, the accepted path's KV is compacted into consecutive positions `p+1..p+j` with an in-device copy of `j` positions per layer, then the rest is truncated. Forking for tree branches across pages uses `KvStore::fork`, which shares sealed pages and copies only the writable tail page.

**Recurrent state.** A `State` layer updates its whole state in place at every position, so it can never be truncated; it is rolled back by **restoring** an earlier copy and recomputing forward (RFC-0033 §4.2.1). The `State` class keeps, per live sequence:
- **Device slots: `max(2, 1 + T_chain)`.** Plain decode ping-pongs between two of them: step k reads slot A and writes slot B, step k+1 reads B and writes A, so the slot not written last holds the state at the start of the last step (the **rollback slot**), with no copy. A verify chain uses `1 + T_chain` of them: the committed state and one snapshot after each provisional position, written by the scan kernel as it advances; accepting `j` makes snapshot `j` the committed state by exchanging slot indexes, with no copy. ADR-0017 limits chains to 4 tokens for these layers, and trees are enabled for hybrid families only after RFC-0012 shows a tree-capable snapshot scheme within the KV budget (ADR-0017 item 6).
- **Host checkpoints: the two newest.** Every 1,024 committed positions (`kv.state_checkpoint_pages × P`, which must equal RFC-0033's `recovery.state_checkpoint_tokens`; configuration validation rejects a mismatch), the dispatcher copies the committed state into a `HostPinned` checkpoint slot on the copy stream and keeps the two newest per live sequence. Position 0, the zero state, is an implicit checkpoint. These are the same positions as the prefix cache's state checkpoints (§7.2), so one copy serves both: a prefix-index node that references a checkpoint keeps it alive beyond the two-newest rule, charged to the prefix cache's share of the pool.
- **`state_pos`.** Every state slot and checkpoint records the position it reflects (the number of positions absorbed). Before a stage applies a row at position p to a `State` layer, its dispatcher checks `state_pos(seq) = p`; a mismatch refuses the row with `StatePositionMismatch` (RFC-0033 §4.2.1), so no recurrent update is ever applied twice.

```rust
impl KvStore {
    /// Positions the sequence's State layers on this stage have absorbed.
    pub fn state_pos(&self, h: &KvHandle) -> u32;
    /// The highest position ≤ `limit` the State layers can be restored to: `state_pos` itself,
    /// the rollback slot's position, the newest checkpoint ≤ limit, or 0. Reported in REPORT.
    pub fn state_restore_point(&self, h: &KvHandle, limit: u32) -> u32;
    /// Restore, returning the new `state_pos`. Attention KV is untouched (RFC-0040 `kv_write_from`).
    pub fn restore_state(&mut self, h: &mut KvHandle, to: StateRestore) -> Result<u32, Error>;
}
pub enum StateRestore { RollbackSlot, Checkpoint { pos: u32 }, Zero }
```

**Budget** (counted in §11's State-class sizing): `max(2, 1 + T_chain) · S_state` on the device and `2 · S_state` in `HostPinned` per live sequence, where `S_state` is the sum over the stage's `State` layers of `elems_per_seq × bytes`; checkpoint copies add `S_state / 1,024` bytes of device-to-host traffic per committed token, amortised [derivation from the rule above].

**Compressed layers.** The family module (RFC-0043) defines when a compressed entry forms. An entry formed inside a verify window goes into a provisional entry slot and is rolled back like a position, so speculative and plain decoding see the same entries.

**Exactness.** Rollback never rewrites committed KV, so speculative and non-speculative decoding differ only by the rounding of a batch of T against a batch of 1, which the parity harness covers (ADR-0017 item 4).

### 7. Prefix cache

#### 7.1 Keys

The prefix key of page `b` of a token sequence is a hash chain (ADR-0206):

```text
k_root = BLAKE3("syntropy-kv-prefix-v1" ‖ ModelId ‖ kv_settings_hash ‖ shared_salt)
k_b    = BLAKE3(k_(b−1) ‖ tokens[b·P .. (b+1)·P] as u32 little-endian)              while page b lies wholly
                                                                                    inside the shared span
k_b    = BLAKE3(k_(b−1) ‖ scope_salt ‖ tokens[b·P .. (b+1)·P] as u32 little-endian) for every later page
         (k_(−1) = k_root)
```

`kv_settings_hash` is BLAKE3 over every setting that changes KV contents or layout: the stored `KvDType` per class, `KvFormula::dtype` and `layout_version`, `P`, the resolved RoPE configuration (base, scaling type and factors, original context length), sliding-window sizes, compression ratios, and any approximate knob that touches KV. It is the same at every stage of an instance. The key does not include the stage's layer range: every stage stores its own layers' pages under the same key. **Scope (since 2026-10-20).** The API reports how many prompt tokens a prefix hit saved (`cached_tokens`, RFC-0050 §9.4), so a cache shared between clients is an exact oracle: one client could confirm, page by page, a guessed prefix of another client's conversation. The default scope is therefore **per principal**: `kv.prefix_scope = principal`. A principal is what ingress authenticated the request as, an API key's ID, or for an open loopback request the local OS user; ingress derives it (RFC-0050, RFC-0034) and passes `SequenceOpen::prefix_scope: PrefixScope { principal: Option<[u8; 16]>, shared_until: Option<u32> }` to the head. `principal` is an opaque 16-byte value, so no stage learns who the principal is, and `None` means the local principal (loopback without a key, and local-socket peers of the daemon's own user); `scope_salt` is those 16 bytes, or a fixed local value for `None`. `shared_until` is the token position of the client's last `cache_control` breakpoint. Pages are shared across principals only inside a **shared span**: the first `shared_until` tokens, rounded down to whole pages, so a page that straddles the breakpoint is private (RFC-0050 maps the API's `cache_control` field). Inside it the key chain carries no principal; from the first page after it, every key mixes in `scope_salt`, so continuations stay private even when they follow a shared system prompt. A request with no `cache_control` mark has an empty shared span, and all its pages are keyed in its own scope. `shared_salt` is empty unless `kv.prefix_scope = instance`, the explicit opt-in that restores one cache for every client of the instance. A single-user setup has one principal and loses nothing. `cached_tokens` counts only hits in the caller's own scope or in a shared span the caller itself marked. RFC-0034 §2 and §13 record the threat; node-local API keys get per-principal scope in M1 and cluster-wide keys in M2, when a principal can reach the instance through another node's ingress. Non-text inputs enter the token stream as placeholder IDs followed by their content hash, which RFC-0050 defines, so a different image never matches.

`PrefixKey` (RFC-0001 §5.9) is `k_b` for the last full page a sequence shares.

#### 7.2 Index

Every stage keeps a `PrefixIndex`: a hash map from page key to node, which forms a radix tree whose edges are whole pages.

```rust
pub struct PrefixNode {
    pub key: PrefixKey,
    pub parent: Option<PrefixKey>,
    pub pages: SmallVec<[PageId; 5]>,          // one per class for this token page (State: see below)
    pub children: u16,
    pub tails: SmallVec<[TailId; 2]>,          // partial pages hanging off this node, ≤ kv.max_tails (4)
    pub hits: u32, pub last_use_ms: u64,
}
pub struct Tail { pub parent: PrefixKey, pub tokens: SmallVec<[u32; 64]>, pub pages: SmallVec<[PageId; 5]> }
```

Index pages are ordinary sealed pages with a reference held by the index, so a page shared by a live sequence and the index is stored once. `State` layers have no per-page state, so a node can also carry a **state checkpoint**: the recurrent state after the node's last token. Every stage with `State` layers takes one when a sequence seals page `b` with `(b + 1)` a multiple of `kv.state_checkpoint_pages` (default 16 pages, 1,024 tokens), and one for every tail. Because the positions are fixed, the head knows where checkpoints exist without asking. A prefix match on a hybrid model stops at the deepest node with a checkpoint, because recurrent state cannot be rebuilt from attention pages. A checkpoint costs `elems_per_seq` per `State` layer, the same as one live sequence's state.

#### 7.3 Lookup (head only)

```text
lookup_prefix(tokens):
    k := k_root; best := none
    for b in 0 .. ⌊len(tokens) / P⌋:
        k := BLAKE3(k ‖ tokens[b·P ..][..P])
        if k ∉ index: break
        if the family has State layers and (b + 1) mod kv.state_checkpoint_pages ≠ 0: continue   // walk on; best only moves at checkpoints
        best := (k, (b+1)·P)
    tail_match := longest common prefix between tokens[best.tokens ..] and each tail of best (token comparison)
                  (for families with State layers, a tail counts only if all of it matches, since its checkpoint is at its end)
    return Cached { last_full: best.k, tokens: best.tokens + tail_match, tail, epoch }
```

Hashing one 64-token page is 256 bytes of BLAKE3, so a 128K-token prompt costs 2,048 hashes, well under a millisecond. The matched tail is copied into a fresh writable page for the new sequence (copy-on-write), so the match is token-exact and the new sequence prefills exactly the tokens after it (SC-20).

#### 7.4 Who knows the keys

Only the head (and stages holding `RowLookup` groups) sees token IDs. So the head computes each page's key when the page seals and stamps it into the side channel of the frame that carries the page's last position: `PageKeys { seq, first_page, keys }`, 32 bytes per 64 tokens. Every stage inserts its own copy of the sealed page under that key. On `Release`, each stage converts the sequence's last partial page into a tail if it holds at least `kv.min_tail_tokens` (8) tokens.

#### 7.5 Consistency around the ring

The head is authoritative for which entries exist (ADR-0206). The rule that keeps stages consistent without a round of messages per request:

1. **Normal eviction is the head's.** When the head needs room, or a downstream stage reports `PrefixPressure { stage, pages }`, the head picks leaves by lowest `(hits + 1) / age`, removes them from its index and sends `PrefixEvict { keys }` on the next forward frame. Every stage drops those pages when it sees the message.
2. **Stages may move index pages down without asking.** A stage that needs VRAM or host pages for a live sequence moves index-only pages to the next tier on its own (§7.6). Data is not lost, so consistency holds.
3. **Stages may drop only as a last resort**: when the lowest tier is full or at Critical memory pressure. They then report `PrefixDropped { keys }`, which the tail relays to the head on the ring return.
4. **Races are caught at attach.** A stage whose `attach` finds the referenced key missing reports `PrefixMiss { seq, stage, have_tokens }` on the sequence's first frame. The tail returns `Restart { seq, from: min(have_tokens) }` to the head with the ring-return token frame, and the head re-issues prefill from that position. This costs one wasted chunk traversal and only happens when rule 3 races a lookup.

Side-channel items are carried in `ActivationFrame::side` and `TokenFrame` (RFC-0001 §5.8); RFC-0031 encodes them and RFC-0032 relays them.

#### 7.6 Prefix pages across tiers

Index-only pages follow the tier ladder by recency: `Vram` → `HostPinned` → `Nvme`. Writing to NVMe is **write-through-selective** (`tiered-memory-inference.md` §12.2): an index-only page leaving `HostPinned` is written to the spill file only if its node has `hits ≥ kv.nvme_min_hits` (2); otherwise the head is asked to evict it (rule 1) or, at the lowest tier, it is dropped (rule 3). A prefix hit on pages in `HostPinned` is used where they are, through split attention (§5.3), or fetched to VRAM when the plan's hot window has room; a hit on `Nvme` pages is fetched at `PrefetchNear` priority while earlier chunks prefill.

#### 7.7 Lifetime across plan epochs

The index lives with the runner. A budget-only `PlanDelta` keeps it. A drain-and-replace epoch keeps a stage's entries only if that stage's layer range, page size and KV settings are unchanged; otherwise the new runner starts with an empty index for that stage and the head learns of the gap through `PrefixMiss` (rule 4). Saved sessions (§8) are the durable form; the prefix cache is a cache.

### 8. Saved sessions

A saved session is the durable form of a session's KV and recurrent state (glossary: saved session). It lets an agent's next turn, or a conversation reopened next week, prefill only its new tokens.

#### 8.1 When a session is saved

- **On release with `persist`.** Ingress sets `SequenceControl::Release { persist: Some(session) }` (RFC-0001 §5.8) when the request asked for it, or by the default policy `kv.autosave = true` for sessions whose committed context is at least `kv.autosave_min_tokens` (1,024). Every stage writes its blob; the head writes the token list. Autosave stays on by default, with the endurance budget of §8.6.
- **Every re-save is incremental (by M1).** When a session already has a manifest at a stage, a later save writes only the pages sealed since the previous save, as one new segment, plus the new manifest; it never rewrites earlier segments. A Llama-70B-class agent session at 20K context costs ≈ 6.25 GiB per full save, against ≈ 0.6 GiB for a turn of ≈ 2K new tokens [estimate: 320 KiB per token]. Segments are compacted into one only when the session is restored, and only within the endurance budget (§8.6).
- **Checkpoints of long live sessions.** For a sequence with a session above `kv.checkpoint_tokens` (32K), the pages sealed since the last checkpoint are written as a new **segment** every `kv.checkpoint_tokens` tokens (`Background` I/O), so a runner crash loses at most the tokens since the last segment, and the final save writes only the last segment. A segment is never appended to after it is renamed into place.
- **Under memory pressure** (rung 1 of RFC-0020 §11): an idle session's pages are saved, then freed.

**Durability and the log entry.** A stage's saved session is a set of segment files plus a **manifest** that lists every segment with its token range and BLAKE3. Segments and then the manifest are each written to a temp name, data-synced and renamed (RFC-0021 §13); the session is durable at a stage when its manifest is. Each stage then sends `SessionSaved { session, stage, manifest: Blake3 }` toward the head on the side channel; downstream stages' messages ride forward frames to the tail, which relays them on the ring return, as for `PrefixDropped` (§7.5). When the head has one `SessionSaved` per stage, its runner asks its daemon to submit `ClusterCommand::PinSession { session, instance, holders }` (RFC-0001 §5.10). If they do not all arrive within `kv.save_timeout` (30 s), no `PinSession` is submitted and the partial save is left to retention. Only saved sessions enter the event log; live sequences never do.

#### 8.2 Blob format

Per (session, stage), the store holds segment files `sessions/<session-id>/<stage-digest>/<n>.kvb` in the layout below, and a manifest `…/manifest` (magic `SYNKVSM1`, version, session ID, stage layer range, `kv_settings_hash`, then one entry per segment: index, first token, token count, length, BLAKE3), all written through RFC-0021 §13. Little-endian; every section starts on a 4 KiB boundary so it can be read with direct I/O (ADR-0208).

```text
Header (offset 0, 256 bytes)
  0x000  magic            [u8; 8]   "SYNKVSB1"
  0x008  version          u16       1
  0x00A  flags            u16       bit 0: lossy KV dtype (approximate); bit 1: segment of a checkpointed session (its token range is in the manifest)
  0x00C  header_len       u32       256
  0x010  session_id       [u8; 16]
  0x020  model_id         [u8; 32]  ModelId (model content hash)
  0x040  kv_settings_hash [u8; 32]  §7.1
  0x060  token_hash       [u8; 32]  BLAKE3 over the committed token IDs (u32 LE)
  0x080  layer_first      u16       first layer in this blob
  0x082  layer_end        u16       one past the last layer
  0x084  page_tokens      u16       P
  0x086  n_classes        u8
  0x087  reserved         u8
  0x088  tokens           u32       committed positions covered
  0x08C  n_sections       u32
  0x090  created_unix_ms  u64
  0x098  kv_format        u32       Syntropy KV format version; a reader rejects versions it does not know
  0x09C  reserved         [u8; 100] zero
  0x100  section table    n_sections × 64 bytes:
           layer u16, class u8, dtype u8, layout u16, reserved u16,
           first_token u32, n_tokens u32, offset u64, length u64,
           blake3 [u8; 32]
  ...    zero padding to the next multiple of 4096
Sections
  One section per (layer, class, contiguous token range), page-first within the layer:
  pages in token order, each page's slice padded to 4 KiB. State layers: one section holding
  the state at `tokens`. Aux caches: one section per layer.
Token list (head blob only)
  One extra section with class = 0xFF: the committed token IDs, u32 LE.
```

Per-layer sections are what let a session resume under a different stage split (§8.4). Every section carries its own BLAKE3, checked on restore before any page is used.

**Encryption (from M2).** From M2 a session's blobs can land on nodes other than the session's ingress, including laptops run by other people, so every blob a holder writes is ciphertext. RFC-0034 §13 owns the key scheme and the ciphers; this RFC owns where its pieces sit in the blob:
- **Keys.** The session's ingress generates an X25519 key pair per session and keeps the private half (RFC-0034 §4, §13). Every stage receives the public half in `SequenceOpen::session_pk` (RFC-0032 §3.6). No holder ever receives a key that can open anything it wrote.
- **Saving.** Each segment file is one blob. Before writing it, the holder draws a fresh content key, seals the segment's sections and, in the head's blob, the token list with it (an AEAD, with a nonce derived from the section index, which is unique because the key is used for one segment only), wraps the content key to `session_pk` with an ephemeral X25519 exchange, and drops the content key. Incremental re-saves (§8.1) therefore add one new wrapped key per new segment and never re-encrypt earlier segments.
- **Format.** Header flag bit 2 marks an encrypted blob. A key section (class `0xFE`, listed in the section table, not itself encrypted) holds the ephemeral public key, the wrapped content key with its tag, and the fingerprint of `session_pk`; `SessionSaved` names the blob by that fingerprint only. The section table's BLAKE3 values cover the ciphertext, so the store verifies, moves and garbage-collects blobs without any key. The manifest stays plaintext: it lists segment indexes, token counts, lengths and hashes, and carries no token IDs and no KV bytes.
- **Restoring.** The stage that restores, whether it holds the blob or fetched its sections from a holder under a changed split (§8.4), sends the blob's key section to the ingress over the authenticated control connection; the ingress unwraps it and returns that blob's content key only. The stage decrypts into its runner's memory, runs §8.3's identity checks on the plaintext, and drops the key when the restore ends. With the ingress gone, nothing can be unwrapped, and the resume re-prefills from the transcript, which also lives on the ingress.
- **Revocation and purge.** A revoked member keeps only ciphertext it cannot read. `syntropy sessions purge` (RFC-0051) deletes the session's private key at the ingress and asks every holder to delete its segments and manifest; a copy a holder fails to delete stays unreadable.
- **M1.** Blobs never leave the ingress node in M1, so M1 writes plaintext blobs in this format with bit 2 clear. Stored Responses records stay on the ingress only and are listed in RFC-0034 §13's inventory (RFC-0050, OQ-PRD-10).

#### 8.3 Identity binding

A blob may be restored only if all of these match the restoring instance: `model_id`, `kv_settings_hash`, `page_tokens`, `kv_format`, and the session's `token_hash` against the tokens ingress holds for the session's transcript. Any mismatch makes the session unusable for that instance: ingress falls back to a full prefill from the transcript, and the stale blob is left to retention. This is the binding the surveyed engine uses (model files plus every setting that changes meaning, `tiered-memory-inference.md` §12.1), with the token hash added so a session can never resume against a different conversation.

#### 8.4 Restore

```text
SequenceOpen { session: Some(s), prefix: Some(Session { session: s, tokens, blob }) }   (ingress found it in its replica of cluster state)
head: forwards the PrefixRef on the first frame, as for a prefix hit
each stage:
    if a local blob covers exactly its layer range and passes §8.3:
        read its sections with direct I/O into pages (HostPinned or Vram per the hot-window policy), PrefetchNear priority
    else (the split changed, or the blob is on another node):
        fetch the sections for its layers from the holders over the bulk plane (RFC-0041 store; chunks verified by BLAKE3)
        and import them (§10)
    if neither works: report PrefixMiss { have_tokens: 0 }; the head re-prefills from the transcript (§7.5 rule 4)
prefill continues from `tokens` with the new turn's tokens
```

Restored pages are also inserted into the prefix index under the keys the head recomputes from the token list, so the next turn hits the prefix cache without reading the blob again. [Target] SC-19: TTFT ≤ 1.25 × `KV_bytes / W_read(tier)` + prefill of the new tokens.

#### 8.5 Retention

Blobs are store-owned files (RFC-0041 GC). Defaults: `kv.session_retention` = 7 days since last use; `kv.session_disk_max` = the smaller of 10% of the store's disk budget and 50 GiB per node; least recently used blobs go first. Deleting a blob submits `DropSession` when it was the last holder. A user can list and delete sessions through the API and CLI (RFC-0050, RFC-0051), and `syntropy sessions purge` deletes everywhere (§8.2).

#### 8.6 Endurance budget (by M1)

Autosave, KV spill, packs and compaction together can approach a consumer drive's rated endurance within a year under an agent workload: a 20K-context agent session saved in full on every turn writes ≈ 6.25 GiB a turn [estimate: 320 KiB per token]. Each node therefore keeps a write budget per volume, read from SMART through RFC-0021 §13: `bytes_per_point = data units written ÷ percentage used` gives the drive's bytes per point of rated endurance, and the budget is `kv.endurance_budget` (default 10 points of rated endurance per year [target]) for everything Syntropy writes there. When Syntropy's trailing 30-day write rate exceeds the budget's rate, session saves back off first (`kv.autosave_min_tokens` doubles at each step, then only explicit `persist` saves), then KV spill to `Nvme` is limited to sequences that cannot otherwise be admitted; pack builds are never throttled. While SMART is unreadable, or percentage used is still 0, no budget can be computed: nothing backs off, and `doctor` shows Syntropy's write volume and says why.

### 9. KV quantization (approximate mode)

Storing KV below the model's KV dtype is the `kv_lossy` knob (ADR-0019 §6, RFC-0001 §11.3 `KV_LOSSY`). It is opt-in per instance, never a fallback, and labelled on every response it touches.

| `KvDType` | Storage | Memory against BF16 | Evidence |
|---|---|---|---|
| `Native` | the reference's KV dtype | 1× | exact |
| `Q8 { block: 32 }` | int8 values, one F16 scale per 32 elements | ≈0.53× | the surveyed engine defaults to 8-bit above 4–8K context (`tiered-memory-inference.md` §12.1) |
| `K8V4 { block: 32, rotate: true }` | int8 K; V rotated by a Hadamard transform, then 4-bit with F16 scales | ≈0.40× | the surveyed engine's 8-bit K with rotated 4-bit V: 23% less memory than 8-bit |

Rules:

- **Instance-wide.** The dtype is a plan field (`KvBudget::dtype`, RFC-0001 §5.4), because every stage must agree and prefix keys depend on it. Ingress never routes an exact request to an instance whose plan is approximate (RFC-0001 §5.8).
- **Quantize on write, dequantize in the attention kernel.** Pages never change dtype after they are written, so spill and migration stay bit-for-bit within the chosen dtype.
- **Keys and blobs record it.** `kv_settings_hash` includes it, and saved blobs set the lossy flag, so a lossy session never resumes on an exact instance.
- **Labels.** Every `TokenEntry` from such an instance carries `ApproxFlags::KV_LOSSY` (RFC-0001 §5.8), and the harness publishes the knob's measured KL and top-1 cost before it is offered (ADR-0019 §6).
- **Recurrent state is never quantized** by this knob.

### 10. KV migration and export

Migration moves a sequence's KV for some layers from one runner to another: when a re-placement moves a layer range while sequences are live (RFC-0001 §7.5 "KV migration"), when a session's blob sections are fetched from another node (§8.4), and when RFC-0033 recovers a stage. The unit is a **`KvChunk`**: one layer, one class, one contiguous token range, page-first.

```rust
pub struct KvChunk {
    pub instance: InstanceId,
    pub epoch: PlanEpoch,                  // epoch of the exporter
    pub seq: SequenceId,
    pub layer: LayerIx,
    pub class: PageClass,
    pub dtype: KvDType,
    pub layout: KvLayoutId,                // RFC-0040 in-page element order
    pub page_tokens: u16,
    pub first_token: u32,
    pub n_tokens: u32,
    pub kv_settings_hash: Blake3,
    pub payload: PayloadRef,               // page-first bytes, each page slice padded to 4 KiB
    pub hash: Blake3,                      // over the payload
}
```

RFC-0031 owns its byte layout on the wire and RFC-0032 its transport (bulk plane, lower priority than the data plane). The rules:

- **Export** takes counted leases on the sealed pages (RFC-0020 §7.4), gathers them page-first into `NetPool` buffers and streams chunks. Writable tail pages are exported after the sequence is paused at a step boundary, so the export is a consistent cut at `tokens`.
- **Import** checks `kv_settings_hash`, `page_tokens` and the hash, allocates pages, scatters into the layer-first layout and builds the block tables. A mismatch rejects the import, and the caller falls back to replay or re-prefill (RFC-0001 §7.5).
- **Bit-for-bit.** Chunks move in the stored dtype. In exact mode, migration and session transfer never convert.
- **Cost model term** for the solver and RFC-0033's choice of recovery path: `t_migrate = KV_bytes(layers, tokens) / β + α`, against replay (`history bytes / β + prefill of the lost layers`) and re-prefill. The cost model's example: 1.3 GB for a 4K-token Llama-70B prompt, ≈4.5 s at 2.5 GbE (§5.4).

### 11. Budgets, headroom and admission

**Budgets.** `KvBudget { stage, tier, bytes, dtype, spill }` (RFC-0001 §5.4) becomes pool capacities: for each class, `pages = bytes · share(class) / page_bytes(class)`, where `share` is the class's fraction of the stage's per-token KV bytes (`KvFormula::stored`). `Windowed`, `State` and `Aux` classes are sized first at their worst case for the plan's maximum live sequences, because they never spill while live; for `State` that worst case includes the rollback slots and the two host checkpoints per sequence (§6).

**Per-sequence footprint** used for admission, at the stage:

```text
footprint(seq) = KvFormula::stored(layers, min(prompt + max_tokens, prompt + kv.reserve_gen), dtype)
               + T_max positions (§6) + State and Aux classes for one sequence
```

`kv.reserve_gen` (default 2,048 tokens) avoids reserving a full `max_tokens` that most sequences never use. A sequence that outgrows its reservation takes pages from the spill tiers.

**Headroom**. One type, in tokens, per tier:

```rust
/// Free capacity of a stage's KV store in tokens, per tier: for each tier, the minimum over the
/// token-indexed classes (Full, Compressed scaled by its ratio) of free pages × P. Windowed, State
/// and Aux are sized for the plan's maximum live sequences and never limit admission.
pub struct KvHeadroom { pub tokens: [u32; 3] }                 // [Vram, HostPinned, Nvme]
```

The **admission channel** is the data plane: every stage writes its `KvHeadroom` (three u32) into the side channel of each frame it sends, as RFC-0012 §10.1 specifies, and the head's admission reads the latest values from the ring. The copy in the residency digest is never the binding decision: ingress pre-checks against the instance's gossiped digest (RFC-0001 §8.4 rule 5), and the head decides from the fresh per-frame values. Ingress admits a sequence only when every stage's headroom covers its footprint across the hot window and spill tiers together (RFC-0001 §8.4 rule 5); otherwise it queues (bounded) or rejects with `Capacity`.

**When a live sequence cannot grow** because every tier is full: the stage first evicts index-only prefix pages (§7.5), then spills idle sequences to `Nvme`. If that is not enough, the sequence that cannot grow is **paused** (RFC-0012 §10.3): it is left out of steps, its pages stay allocated, and it resumes when pages free up. It fails with `Capacity` and `RetryHint::Backoff` only when the request's policy forbids pausing or its deadline passes. A running sequence's own pages are never dropped to make room for another.

**Plan deltas.** `apply_budget` grows or shrinks pool capacities within the arena of RFC-0020 §8.6: shrinking the `Vram` hot window spills the excess first, then releases pages.

### 12. Memory pressure

The KV store owns rung 1 of RFC-0020 §11.4 and contributes to rung 3:

1. **Rung 1: idle sessions.** Sessions with no live sequence, and sequences paused longer than `kv.idle_spill_after`, are saved (§8.1) and their pages freed, oldest first. Predicted cost: none for live sequences; the next resume of each pays `KV_bytes / W_nvme`.
2. **Rung 3 share.** Index-only prefix pages in `HostPinned` move to `Nvme` (if they qualify, §7.6) or are evicted by the head.

The `yield_menu` entries (RFC-0020 §11.3) for these rungs list the bytes and that predicted cost.

### 13. Configuration

Schema in RFC-0054. `instance` scope unless marked.

| Key | Default | Reload | Meaning |
|---|---|---|---|
| `kv.page_tokens` | 64 | restart | P (ADR-0205); 16–256, power of two |
| `kv.dtype` | `native` | restart | `native` / `q8` / `k8v4`; anything but `native` needs approximate mode (§9) |
| `kv.reserve_gen` | 2,048 | hot | generation tokens reserved at admission (§11) |
| `kv.idle_spill_after` | 2 s | hot | idle sequences spill to `HostPinned` (§5.1) |
| `kv.admit_pcie_share` | 0.10 | hot | PCIe share per step for sparse-attention page admission (§5.3) |
| `kv.prefix_cache` | `true` | restart | enable the prefix cache |
| `kv.prefix_scope` | `principal` (was `instance` until 2026-10-20) | restart | `principal` / `instance` (§7.1); `api_key` is an alias of `principal` |
| `kv.max_tails` | 4 | restart | partial tails per node |
| `kv.min_tail_tokens` | 8 | hot | smallest tail kept on release |
| `kv.state_checkpoint_pages` | 16 | restart | recurrent-state checkpoint spacing (§6, §7.2); `× P` must equal RFC-0033's `recovery.state_checkpoint_tokens` |
| `kv.nvme_min_hits` | 2 | hot | write-through-selective threshold (§7.6) |
| `kv.autosave` | `true` | hot | save sessions on release by default (§8.1) |
| `kv.autosave_min_tokens` | 1,024 | hot | smallest session saved by default |
| `kv.checkpoint_tokens` | 32,768 | hot | segment saves of long **live** sessions (every re-save on release is incremental regardless, §8.1) |
| `kv.endurance_budget` (node) | 10 points of rated endurance per year [target] | hot | §8.6 write budget per volume |
| `kv.save_timeout` | 30 s | hot | wait for every stage's `SessionSaved` before `PinSession` (§8.1) |
| `kv.session_retention` (node) | 7 days | hot | §8.5 |
| `kv.session_disk_max` (node) | min(10 % of the store budget, 50 GiB) | hot | §8.5 |

### 14. Failure handling and error codes

Domain `Kv`.

| Code | Class | Raised when | Action |
|---|---|---|---|
| `KV_POOL_FULL` (1) | `Capacity` | a live sequence cannot get a page in any tier (§11) | the sequence is paused (RFC-0012 §10.3); it fails, retryable, only if its policy forbids pausing or its deadline passes |
| `KV_ADMISSION` (2) | `Capacity` | ingress finds too little headroom | queue or 429/503 with `Retry-After` (RFC-0001 §10.1) |
| `KV_PREFIX_MISS` (3) | `Transient` | `attach` finds a key missing (§7.5 rule 4) | `Restart` from the shortest held prefix |
| `KV_SESSION_MISMATCH` (4) | `InvalidRequest` | identity binding fails (§8.3) | re-prefill from the transcript; the response notes that the session could not be resumed |
| `KV_SESSION_CORRUPT` (5) | `Integrity` | a section's BLAKE3 fails | try another holder; otherwise re-prefill; the blob is deleted |
| `KV_IMPORT_REJECTED` (6) | `Integrity` | chunk settings or hash mismatch (§10) | the caller chooses replay or re-prefill |
| `KV_SPILL_IO` (7) | `Transient` → `Capacity` | a spill write fails (`IO_NO_SPACE` or repeated `IO_SHORT_READ`) | spill to that volume disabled for the epoch; the pages stay where they were; the headroom shrinks |
| `KV_LOSSY_ON_EXACT` (8) | `InvalidRequest` | an exact request reaches a lossy instance or blob | rejected with the active knobs named (ADR-0019 §7) |
| `KV_SPILL_UNCERTIFIED` (9) | `Unsupported` | a spill path or tier is not certified for the provider, attention kind and tier (§5.3) | exact sequences use a certified substitute or do not spill; reported as a degradation, never served silently |

KV pages are only ever read in state `Sealed` or `Writable` and never while `Moving`, so none of these failures can return wrong tokens.

## Performance model and budgets

All figures are **[estimate]** with derivation or **[target]**.

**KV read per step** (cost model §3.6): `B_kv = context · KV_per_token`. At 32K context, Llama-70B reads 10 GiB per step and DeepSeek-V3 2.1 GiB; hybrid linear-attention models 20–30 KiB per token, 2.5–3.75 GiB at 128K (`backends-and-portability.md` §4.2).

**Spilled dense KV, by attention kind** [estimates; F_cpu = 1–2 TFLOPS, DDR5 at 70 GB/s, pinned PCIe at 24 GB/s, VRAM at 800 GB/s, as in the cost model]:
- *Llama-3.3-70B (GQA, g = 8), 32K context, 2 GiB in VRAM and 8 GiB spilled, decode.* `CpuSplit`: 8 GiB / 70 GB/s ≈ 123 ms, memory-bound (the CPU needs only 8 FLOP per byte), overlapped with the GPU's ≈2.7 ms. `GpuStream`: 8 GiB / 24 GB/s ≈ 358 ms. All in VRAM: ≈13 ms. `CpuSplit` is ≈3× better than streaming at T = 1; from about 5 query rows per pass, streaming wins (§5.3).
- *DeepSeek-V3 (absorbed MLA), 32K context, all 2.1 GiB (≈2.25 GB) spilled, decode.* `CpuSplit` is compute-bound at ≈242 FLOP per byte: ≈0.28–0.56 s. `GpuStream`: ≈94 ms. All in VRAM: ≈3 ms. Streaming is 3–6× better than the CPU, and keeping the KV hot is ≈30× better than either.

So spilling dense attention is expensive for every kind on a desktop, and the solver should keep it hot unless the VRAM it frees buys more than its cost. For dense MLA it almost never does: the ≈91 ms of streaming against VRAM is close to half of scenario b2's ≈217 ms token. Spilling pays where only a small part of the cache is read per step: sparse or indexed attention with a VRAM hot set, and linear-attention hybrids whose per-token KV is small. At the surveyed engine's 262K context with sparse attention, moving KV to RAM let VRAM hold 3,872 experts instead of 1,589 and took decode from 50.9 to 62.6 tok/s (`tiered-memory-inference.md` §12.1). That is the trade RFC-0011 makes with §5.4's terms.

**Page-size overheads (P = 64).** Internal fragmentation is at most 63 tokens per class per sequence: ≈19.7 MiB for Llama-70B at 320 KiB per token, ≈4.2 MiB for DeepSeek-V3. Block tables cost 8 bytes per page per class: a 128K-token sequence is 2,048 pages, 16 KiB. One page of DeepSeek-V3 MLA KV is 64 × 1,152 B = 72 KiB per layer, ≈4.5 MB across 61 layers in the page-first host layout, which is a direct-I/O-friendly transfer (ADR-0205).

**Prefix lookup.** 2,048 BLAKE3 calls on 256-byte inputs for a 128K prompt: well under 1 ms on one core. [Target] SC-20: tokens prefilled equal tokens after the longest cached-prefix match, exactly.

**Resume.** [Target] SC-19: `TTFT ≤ 1.25 · KV_bytes / W_read + t_prefill(new)`. The surveyed 63K-token session (1.2 GB) from NVMe at 6 GB/s: ≈0.2 s, so ≤ 0.25 s plus the new turn's prefill, against ≈30 s to re-prefill (`tiered-memory-inference.md` §12.1).

**Migration.** `t_migrate = KV_bytes / β + α`: 1.3 GB for a 4K-token Llama-70B prompt is ≈4.5 s on 2.5 GbE (cost model §5.4); on RDMA over Thunderbolt 5 at 2.5–4 GB/s, ≈0.3–0.5 s.

**Speculative overhead.** Tree compaction copies `j` positions per layer per window: for DeepSeek-V3 at j = 3, 3 × 1,152 B × 61 layers ≈ 211 KB of in-device copies per window, microseconds of VRAM bandwidth. Recurrent snapshots cost `T_chain` extra states per live sequence, at most 4.

**Hot path.** [Target] zero allocations and zero locks per step in the KV store (ADR-0012 item 6); page-state changes are dispatcher-local. A running sequence's block-table changes go through the table-delta ring of RFC-0023 §5.4; the whole block table of a sequence that has not yet run (new, prefix-attached or restored) is written by a direct copy before its first step, so a 128K-token attach (≈2,048 pages per class) never floods the ring.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Page size 16 (common GPU paged-attention block) | 4× more block-table entries and 4× smaller tier transfers; finer prefix sharing is recovered by token-granular tails (ADR-0205). |
| Page size 256 | Up to 255 tokens of fragmentation per class per sequence, and coarse sealing delays tiering and prefix insertion. |
| One page index per layer | Every layer would need its own block table and kernel argument; layers of a class advance in lockstep, so one table suffices. |
| Same layout on GPU and host | Layer-first on the host makes every spill one small transfer per layer; page-first on the GPU makes attention kernels stride across layers (ADR-0205). |
| Always compute spilled KV on the CPU (the earlier draft) | Wrong for MLA, where the CPU is compute-bound at ≈242 FLOP per stored byte and streaming is 3–6× faster; right only for small `g_eff` at small T. Replaced by the per-kind, per-pass choice (§5.3; ADR-0207). |
| Always stream spilled KV over PCIe | ≈3× slower than the CPU for GQA at decode (§Performance). Kept as `GpuStream`: the default for MLA, for passes of at least `T*_kv` rows, and for prefill. |
| Prefix hash table without a tree (flat, page-granular) | Loses token granularity (SC-20) and gives no structure for leaf eviction. The page-hash radix tree has both (ADR-0206). |
| Token-granular radix tree keyed by token IDs | Exact, but each stage would need the token IDs, which downstream stages do not have; content keys can be computed once at the head and stamped. |
| Independent prefix eviction per stage | Stages would diverge and every prefix use would need a round of checks; the head-authoritative rule needs none in the common case. |
| Sessions as one opaque blob per stage | Cannot resume under a different stage split; per-layer sections can (ADR-0208). |
| Quantized KV by default | Changes output (perplexity +8–12% at 4-bit in the surveyed engine, `tiered-memory-inference.md` §12.1); D3 forbids it outside approximate mode. |
| A separate type for recurrent state | Duplicates handles, residency, migration and persistence; a `State` page class reuses all of them (RFC-0001 §17 item 1). |

## Testing and validation

| Claim | Test | Level |
|---|---|---|
| Page tables and pools | random sequences of open, append, seal, fork, truncate, spill, fetch and release against a reference model; no page is leaked, double-freed, read while `Moving`, or written after sealing | property, model-based |
| Bit-for-bit tier moves | random pages spilled to `HostPinned` and `Nvme` and fetched back compare equal byte for byte; page-first gather and layer-first scatter are inverses | property, integration |
| Both spill paths are exact within tolerance | parity harness with the hot window forced to every split from 0% to 100% of the context, under `CpuSplit` and `GpuStream`, for GQA, MLA, sliding-window, sparse and compressed layers, and for a family with attention sinks; the empty-side rule exercised (all pages on one side); must pass the ADR-0019 tolerances for the dtype | parity (RFC-0060) |
| The chosen path is the cheaper one | sweep T on the CUDA rig for GQA and MLA with KV spilled; the measured crossover is within 20% of the plan's `T*_kv` (else drift is reported) | benchmark |
| Speculative rollback | chain and tree verify against plain decoding with forced acceptance patterns; committed KV is identical; recurrent snapshots select the right state | property, parity |
| SC-20 prefix contract | an agent benchmark replaying multi-turn traces with shared prefixes: prefilled tokens equal tokens after the longest match, exactly, including token-granular tails and hybrid-model checkpoints | integration, benchmark |
| Prefix consistency across stages | `testkit` multi-node simulation with forced downstream drops and evictions racing lookups: every race ends in `PrefixMiss` and a correct restart, never in a stage using KV it does not have | simulation |
| SC-19 resume | save and resume sessions of 4K–128K tokens on every Tier 1 rig, same split and changed split; TTFT within the target; output matches an uninterrupted run within tolerance | benchmark, parity |
| Identity binding | resume after changing the model file, KV dtype, RoPE scaling, page size or transcript: always refused, always falls back to re-prefill | integration |
| Blob crash safety | kill the runner mid-save and mid-checkpoint; no torn blob is accepted (section hashes, atomic rename) | chaos |
| Prefix scope per principal (§7.1) | two principals send the same 4K prefix: the second gets `cached_tokens = 0` and prefills in full; with the prefix marked by `cache_control` in both requests, the second hits the shared span and nothing after it; a principal's continuation after a shared span never hits another principal's pages; with `kv.prefix_scope = instance` both hit; one principal on loopback sees no change from the old default | integration, property |
| Incremental re-saves (§8.1) | a session saved after turns of 2K new tokens writes one new segment per turn whose size matches the new pages, never rewriting earlier segments; restore from many segments equals restore from one | integration |
| Endurance budget (§8.6) | a fake SMART source and a scripted write load: below the budget nothing backs off; above it, the save threshold doubles, then only explicit `persist` saves run; packs are never throttled; with SMART unreadable nothing backs off and `doctor` says why | unit, simulation |
| Encrypted blobs (§8.2, M2) | a session saved on a remote holder: every segment on its disk carries flag bit 2 and a key section, holds no plaintext canary, and opens only with a content key the ingress unwrapped; a test-build memory scan finds no content key on the holder after the save or after the restore; two incremental saves produce two key sections and leave the first segment's bytes unchanged; a stage restoring under a changed split gets keys only for the blobs it presents; with the ingress gone the resume re-prefills; after purge, a holder copy that was not deleted cannot be read | integration, security |
| Migration | export from one runner and import into another over the simulated fabric, then continue decoding; output matches within tolerance | simulation, parity |
| Lossy KV labelling | every response from a `q8` or `k8v4` instance carries `KV_LOSSY`; exact requests are refused there | integration |
| No allocation or lock per step | the `testkit` counting allocator and lock instrumentation during decode with spills in flight | unit |

## Rollout / milestones

RFC-0001 §16 lists KV paging, the prefix cache and saved sessions in M1, with SC-19 and SC-20 as M1 exit criteria. RFC-0001 §16 governs (the vision's checklist now matches): single-node behaviour lands in M1, cross-node parts in M2 and M3:

| Milestone | Lands |
|---|---|
| M0 | `KvStore` with `Full`, `Windowed` and `State` classes in `Vram` only; P = 64; block tables; no tiering, no speculative slots |
| M1 | chain verify windows and recurrent snapshots for MTP; `HostPinned` and `Nvme` spill with page-first host layout; both spill paths and the `spill_compute` plan field (`CpuSplit` with the exact merge on the CUDA and CPU providers, `GpuStream` on CUDA; Apple has no spill path, §5.3); the sparse class with its indexer cache for the M1 families' sparse and indexed layers (the M1 MoE families are DeepSeek-V3-class MLA, Qwen-MoE/hybrid and gpt-oss-class, so M1 KV layouts are MLA, GQA, sliding-window, sparse and linear-attention state); prefix cache on a single-stage instance with per-principal scope for node-local keys and `cache_control` shared spans; saved sessions on one node (blob format v1, autosave, incremental re-saves, the endurance budget, retention); SC-19 and SC-20 on single-device rigs |
| M2 | prefix cache across stages (page-key stamping, head-authoritative eviction, `PrefixMiss` restarts) and per-principal scope for cluster-wide keys; encrypted session blobs (per-blob content keys wrapped to `session_pk`, unwrapped only by the ingress), and purge across holders; KV export and import for drain-and-replace re-placement |
| M3 | session restore under a changed split and from remote holders; KV migration for fault recovery (with RFC-0033); tree verify; lossy KV dtypes as approximate knobs; the `Compressed` class with the DeepSeek-V4-class family (compressed attention arrives in M3) |
| Post-1.0 | KV on another node's RAM (§5.5) |

## Open questions

1. **Page size.** 64 is chosen from transfer-size and fragmentation arithmetic, not measurement. An M1 benchmark of decode attention throughput at P = 16, 32, 64 and 128 on the CUDA and Metal providers, plus spill bandwidth at each, confirms or changes ADR-0205.
2. **CPU attention throughput per kind.** `T*_kv` depends on `F_cpu,attn` per attention kind, which no source measures. RFC-0010 probes it in M1 at T = 1 and at the plan's verify window for GQA and MLA, and the M1 benchmark checks the crossover. If GQA decode on the CPU falls short of DRAM bandwidth, `CpuSplit` loses its T = 1 advantage and `GpuStream` becomes the default everywhere.
3. **State checkpoint spacing.** 16 pages trades checkpoint memory against prefix granularity for hybrid models. Agent traces in M1 decide.
4. **Session autosave default.** On by default above 1,024 tokens is chosen for the agent scenario (vision S4), and stays on (2026-10-20), with incremental re-saves and the endurance budget (§8.6). After M2's disk data, revisit turning `kv.autosave` off on nodes that are not the session's ingress.
5. **Prefix sharing across instances of the same model.** Two replicas with identical stage splits could share saved prefix pages through the store. Not designed. **Moved to the post-1.0 backlog on 2026-10-20 (OQ-IO-25).**
6. **KV on another node's RAM** (§5.5): worth building only if RDMA-class links are common among users. **Moved to the post-1.0 backlog on 2026-10-20 (OQ-IO-10)**, re-scoped: the evidence comes from shared benchmark reports and issues, not from telemetry, which ADR-0510 decision 1 rules out.
7. ~~Compressed-attention families~~ **Answered (2026-10-21; OQ-MF-12, COMP-E16-T15):** the entry-formation rule, read from DeepSeek's reference `inference/model.py` in the Hugging Face repository `deepseek-ai/DeepSeek-V4-Pro` at revision `b5968e9190ef611bbf34a7229255be88a0e937c1`. RFC-0043 open question 2 has the configuration fields, the indexer and the line references.
   - A layer's ratio r, from `compress_ratios`, is 0, 4 or 128. r = 0 means the layer forms no entries. Every layer keeps a raw window of the last 128 tokens, the current one included.
   - Entry j pools block j, tokens rj to rj + r − 1, with a per-channel softmax over learned scores. It is then RMS-normalised, gets RoPE at position rj on 64 channels, and has its other 448 channels rounded through FP8. When r = 4, each token projects two channel sets: entry j pools block j through the second set and block j − 1 through the first, under the same per-channel softmax, so it spans 8 tokens. Entry 0 has no block −1.
   - Entry j forms in the step that processes token r(j + 1) − 1 and is stored at entry slot j, which is §3's entry page `⌊pos / (r · P)⌋`. One 512-element vector per entry is both key and value, so `elems_per_entry` is 512. The raw window likewise stores one 512-element vector per token.
   - Query position s reads entry j only when r(j + 1) ≤ s + 1. r = 128 layers read every formed entry (`selected: None`). r = 4 layers read the indexer's top `index_topk` entries: 1,024 on V4-Pro, 512 on V4-Flash. Both kinds also read the raw window, and the window and the entries can cover the same tokens.
   - Tokens of an incomplete block are in no entry until the block completes. The window, 128 tokens, is at least r long, so they are always inside it.
   - **Gaps, tracked as OQ-MF-16 for TIER-E6-T16.** The reference keeps the projected values and scores of the incomplete block, and for r = 4 those of block j − 1, in per-sequence buffers outside the KV pages. Rolling back a verify window, reusing a cached prefix and restoring a saved session all need that state, and §6 rolls back only formed entries. The indexer keeps 128 elements per entry, not per token, which `AuxKv::elems_per_token` and the token-paged `Aux` class do not express. This answer leaves §2, §3 and §6 unchanged.

## Proposed amendments to RFC-0001

1. **§5.3 `KvFormula`.** Add the `LayerKv::Compressed` variant and the `aux`, `dtype` and `layout_version` fields with serde defaults (§2). RFC-0022 owns the type, so these are owner additions; listed here because RFC-0043 and RFC-0011 read them.
2. **§17 item 1 resolved.** `KvHandle` covers recurrent state: `State` is a page class in the same store, with the same residency, export and persistence paths. No sibling type is needed.
3. **§5.8 side channel.** `ActivationFrame::side` and `TokenFrame` carry the prefix-cache items `PageKeys`, `PrefixEvict`, `PrefixDropped`, `PrefixPressure`, `PrefixMiss` and `Restart` (§7.5), `SessionSaved` (§8.1) and each stage's `KvHeadroom` (§11). RFC-0031 encodes them; RFC-0032 relays them.
5. **§5.4 `KvBudget`.** Add `spill_compute: SpillCompute` with a serde default (§5.3). RFC-0022 owns the type; listed because RFC-0011 must choose it.
4. **§16 milestones.** Record which KV items are M1 (single node) and which are M2/M3 (cross-node), as in Rollout, so RFC-0001 §16 and the vision agree.
