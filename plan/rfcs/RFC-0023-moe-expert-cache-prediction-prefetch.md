# RFC-0023: MoE expert cache, prediction and prefetch

- Status: Draft
- Area: tiers (expert cache)
- Depends on: RFC-0001 (§5.3, §5.4 `ExpertCacheBudget`/`PrefetchWindow`, §5.5, §6.3, §6.4, §8.2–§8.4, §9.3), RFC-0010, RFC-0011, RFC-0012, RFC-0020, RFC-0021, RFC-0022, RFC-0032, RFC-0040, RFC-0041, RFC-0043; ADR-0006, ADR-0007, ADR-0008, ADR-0012, ADR-0017, ADR-0019, ADR-0209, ADR-0210, ADR-0211
- Date: 2026-10-19
- Revised: 2026-10-20: who may see learned profiles, and corpus-derived curves (§13); the M0 routing-trace spike decides M1's NVMe expert scope (Rollout); open question 7 and the sibling-profile seed (§5.6) answered

## Summary

On a machine whose MoE model is larger than VRAM, each routed expert lives in RAM or on NVMe, and a token reads only the few experts its router picks. This RFC makes those reads cheap. It specifies:

- a per-device **expert cache** in the fastest tier: per-layer byte-sized slots, decayed-LFU admission with hysteresis, replacements that never stall a step, a shadow-LRU guard that keeps the hit rate at or above LRU's, and a learned frequency profile persisted across restarts (ADR-0209);
- a **device-resident slot table** updated through a host-mapped delta ring applied by the first node of each captured step, so captured graphs stay valid while the cache changes;
- the **MoE step**: hits on the GPU, copied misses over PCIe by the copy engine, in-place misses on the CPU in the NUMA-local pinned arena, all concurrent, with no host synchronisation;
- the **online three-site balance**: a per-layer recursive-least-squares fit of CPU, PCIe and GPU costs, including the slowdown DMA causes the CPU, that chooses the copied share each step (ADR-0211);
- **next-layer prediction** fused into the router kernel on the device that runs the router, plus exact routing knowledge where a family has it (pre-attention routers, pre-gates, token-hash routing) (ADR-0210);
- **prefetch** at every slow boundary: NVMe → RAM, RAM → VRAM, and, from M3, another node's RAM → local;
- expert unions for verify windows, streaming of the routed pool for prefill chunks above n*;
- **single ownership** of every expert in the cluster and inside a node, with hot replication only as an explicit plan entry;
- **hit-rate telemetry** that gives the solver per-layer `h(c)` curves and popularity buckets.

## Motivation and goals / non-goals

MoE models make RAM and NVMe usable tiers because a token reads its active parameters, not the total: DeepSeek-V3 touches 5.4% of its parameters per token, a hot set of ≈9.7 GB at Q4_K_M plus ≈12.3 GB of routed experts (`hardware-and-performance-model.md` §3.9). How fast it runs then depends on where the routed bytes come from (§3.3, §4 b2):

- every byte from dual-channel DDR5 at ≈70 GB/s caps decode at ≈5.3 tok/s, and from NVMe at 6 GB/s far lower, so the hit rate of the RAM tier and the CPU's DRAM bandwidth decide scenario S1;
- the surveyed single-GPU engine's decayed LFU with hysteresis lifted its VRAM hit rate from 0.50 (static profile) to 0.72; per-layer admission instead of one shared counter took an offline hit rate from 2.97% to 70.4%; per-layer byte sizing fitted 3,556 experts instead of 2,673; replacements off the critical path took decode from 91.7 to 94.4 tok/s; duplicate residency across GPUs wasted ≈2,000 slots (`tiered-memory-inference.md` §5.2, §5.3, §16);
- sending a measured share of misses over PCIe while the CPU computes the rest took i-quant decode from 70 to 78 tok/s, while for a RAM-bound kernel the right share was 0.2 (§5.4);
- next-layer router prediction issued about half of the SSD reads early, for +14% (§6).

**Goals**

1. **Hit rate at least as good as the baselines**: on every recorded routing trace, ≥ max(cache fraction, LRU at equal capacity) (SC-9).
2. **MoE over RAM and NVMe within 10% of the overlap model**: `t_tok ≤ 1.10 · t_overlap(h)` (SC-5), with this RFC's VRAM-cache term added.
3. **≥ 50% of NVMe expert misses issued before that layer's router runs** (SC-8).
4. **Expert reads from NVMe at ≥ 0.85 × the probe's sustained rate** (SC-7, through RFC-0021).
5. **Cache changes never block a step** (RFC-0001 §8.4 rule 7), and captured graphs are never re-captured because the cache changed.
6. **Exact by default.** Every selected expert is computed exactly once, at full precision, on whichever site holds it (ADR-0006 item 8). Skipping, deferring or substituting experts are approximate knobs only.
7. **No geometry hard-coded.** Expert sizes, counts, top-k and shared experts come from the manifest (vision §9 principle 4).

**Non-goals**

- Choosing per-device cache budgets, home tiers or ownership. The solver does (RFC-0011); this RFC supplies the hit-rate curves and buckets it uses (§13).
- The grouped MoE kernel itself (RFC-0040). This RFC fixes its inputs: the slot table, the copy plan and the doorbell layout.
- Neuron-level sparsity, which needs activation-sparse models (`tiered-memory-inference.md` §7).
- Learned or forecast-driven replacement policies. They are the upgrade path behind `CachePolicy` (§5.6), not v1.

## Design

### 1. Ownership and structure

The `experts` module of `syntropy-tier`:

| Component | Runs on | Writes |
|---|---|---|
| `ExpertCache` per (device, cache tier) | the device's dispatcher | its slots' residency (RFC-0020 §7), its slot table, its delta ring |
| `Stats` | the housekeeping thread | decayed counts, shadow LRU, prediction accuracy, `h(c)` estimates |
| `Policy` | the housekeeping thread | replacement decisions, posted to the dispatcher as `CacheCommand`s |
| `Balancer` per MoE layer | the housekeeping thread fits; the dispatcher reads the current split once per step | the per-step split parameters |
| `Predictor` | fused into the router op on the router's device (§9) | doorbell prediction fields |
| `MissPlanner` | the dispatcher and the CPU pool's lead thread, from the doorbell | copy plans, in-place work lists |
| `Prefetcher` | the dispatcher | read requests to the I/O engine, admission copies |

Everything that changes a slot goes through the dispatcher of the device that owns the slot (RFC-0001 invariant 4). The housekeeping thread decides and posts; it never touches a slot. Statistics flow from hot threads to housekeeping over lossy SPSC rings (RFC-0001 §8.3).

### 2. Types

RFC-0001 §5.4 defines `ExpertCacheBudget { device, tier, bytes, per_layer, policy, miss_split }` and `PrefetchWindow { from, to, max_inflight, lookahead_layers }`. This RFC defines the types they name:

```rust
/// ExpertCacheBudget::policy. Parameters with defaults from §15.
pub enum CachePolicy {
    DecayedLfu { decay: f32, adapt_every_steps: u16, hysteresis: f32, max_replace: u16, guard: bool },
    Lru,                                   // baseline and guard fallback
    Static,                            // fill from the profile once; never replace (deterministic runs)
}

/// ExpertCacheBudget::miss_split: the starting point of the balancer (§8).
pub struct MissSplit {
    pub copied_share: f32,                 // fraction of missed bytes copied over PCIe at batch 1
    pub mode: MissSplitMode,               // Fitted (default) | Fixed (deterministic runs, tests)
}

/// From the manifest (RFC-0043): what is known about routing, and when.
pub enum RoutingKnown {
    AtRouter,                              // default: known when the layer's router runs
    PreAttention,                          // the router runs before the layer's attention
    PreGate { lookahead: u8 },             // layer l's pre-gate selects layer l+lookahead's experts
    TokenHash,                             // expert = f(token id); known once the token is sampled
}

/// One entry of the per-layer doorbell the router writes (RFC-0001 §6.3 step 3.1).
#[repr(C)]
pub struct RouteSlot {
    pub expert: u16,
    pub tokens: u16,                       // tokens in the pass that selected it (≥ 1)
    pub cache_slot: u32,                   // slot index, or MISS (u32::MAX)
}
#[repr(C)]
pub struct DoorbellLayer {
    pub seq: u32,                          // step number: a payload check, not a flag (flags: §6 handshake)
    pub n_route: u16,                      // distinct experts selected in this pass
    pub n_pred: u16,
    pub route: [RouteSlot; MAX_UNION],     // MAX_UNION sized from the plan's largest pass (§11)
    pub pred: [PredSlot; MAX_PRED],        // predicted (layer, expert, score) for lookahead layers (§9)
}
#[repr(C)]
pub struct PredSlot {
    pub ahead: u8,                         // j: predicts layer l + j, 1 ≤ j ≤ d
    pub _pad: u8,
    pub expert: u16,
    pub score: f32,                        // router probability mass, summed over the pass's tokens
}
// MAX_UNION = min(E, P_max · k); MAX_PRED = d_max · min(E, P_max · (k + predict_extra)),
// both from the plan's largest pass and lookahead (§9, §11), fixed per epoch. P_max is the
// most tokens in one pass of the plan, the largest `rows` of its buckets (RFC-0012 §2), prefill
// buckets included; it is not T_max, which RFC-0012 uses for the verify window. predict_extra is
// its value at the epoch's start: a hot change below it only lowers how many of the written
// predictions the prefetcher uses, and a raise requests a new epoch (§15). (Revised 2026-10-21.)
pub enum GuardMode { Lfu, Lru, Freeze }   // §5.3

pub enum CacheCommand {
    Replace { layer: LayerIx, admit: ExpertIx, victim_slot: u32 },
    Promote { layer: LayerIx, expert: ExpertIx, staging_slot: u32, victim_slot: u32 },
    Prefetch { layer: LayerIx, expert: ExpertIx, to: TierId, useful_until: StepLayer },
    SetPolicy { layer: LayerIx, mode: GuardMode },
}

/// Learned popularity, persisted as the pack's `XPROFILE` side record (RFC-0041 owns the file layout).
pub struct ExpertProfile {
    pub model: ModelId,
    pub census: Blake3,                    // census digest; a mismatch discards the profile
    pub tokens_observed: u64,
    pub counts: Vec<Vec<f32>>,             // [moe layer][expert], decayed counts normalised per layer to sum 1
    pub resident_hint: Vec<(LayerIx, ExpertIx)>,   // what was resident at the last save, in eviction order
    #[serde(default)] pub hit_curves: Vec<HitCurve>,     // §13, so a restart starts from learned curves
    #[serde(default)] pub union: Vec<UnionCurve>,
    pub saved_unix_ms: u64,
}

/// The balancer's fitted coefficients for one layer (§8), reported for RFC-0010's profile.
pub struct BalancerFit { pub a: f32, pub c: f32, pub d: f32, pub g: f32, pub p: f32, pub k: f32, pub samples: u32, pub residual: f32 }

/// RFC-0020 §13 `ResidencySummary::expert`.
pub struct ExpertCacheDigest {
    /// Measured hit fraction of routed decode reads per cache tier over the last
    /// `experts.curve_window` accesses, directly comparable with the hit curves evaluated at
    /// that tier's current per-layer budgets. RFC-0010 uses it for tier-shortfall drift.
    pub tier_hit: SmallVec<[(TierId, f32); 3]>,
    pub copied_share: f32,
    pub nvme_early: f32,                                // SC-8
    pub predict_recall: SmallVec<[(u8, f32); 2]>,       // per lookahead depth
    /// §13: over all MoE layers, the largest gap between a published hull and its shadow value
    /// at the layer's current budget (the grid always contains it). Above 0.03 marks a tier
    /// shortfall as possible curve over-prediction rather than workload change (RFC-0010 §7).
    pub hull_gap: f32,
    pub replacements_per_s: f32,
}
```

**Where the doorbell types live** (revised 2026-10-21). `RouteSlot`, `PredSlot` and `DoorbellLayer` are defined in `syntropy-kernels`, not in `tier`'s `experts` module. Every provider's router op writes the doorbell and `kernels::moe::assign` reads it (RFC-0040 §3.4), and a provider crate depends only on `kernels` (RFC-0001 §4.1, §4.2 rule 2). This section still specifies their layout. `DoorbellLayer` is a fixed header followed by the `route` and `pred` arrays, whose lengths `MAX_UNION` and `MAX_PRED` the `experts` module computes from the plan and which stay fixed for the epoch.

### 3. Cache levels

| Cache tier | Over home | Exists on | Misses are served by |
|---|---|---|---|
| `Vram` | `HostPinned` or `HostPageable` (RAM home) | discrete GPUs | copied misses (GPU) and in-place misses (CPU) |
| `Vram` | `Nvme` | discrete GPUs, when the plan homes the routed pool on NVMe | the `HostPinned` cache below, else an NVMe demand read |
| RAM cache: a pinned part (`HostPinned`) and a pageable part (`HostPageable`) | `Nvme` | any node whose routed pool exceeds its RAM budget | NVMe demand reads into a RAM slot; the CPU computes in place |
| `Unified` | `Nvme` | Apple, Strix Halo, GB10-class | NVMe demand reads straight into the `Unified` slot (RFC-0021 §8) |

**Exclusive over an `Nvme` home.** For `Nvme`-homed experts, the VRAM cache and the RAM cache hold disjoint sets, so the experts resident in fast memory number `y + u` bytes: `y` in VRAM and `u` in RAM. This is the "union" RFC-0011's hit curves are evaluated on (§13).
- **Up.** VRAM admission of an `Nvme`-homed expert copies it from its RAM-cache slot when it is there (from the pinned part directly, or through a pinned staging slot when it sits in the pageable part, below), and frees that slot once the copy lands. Otherwise the expert lands in a host staging slot first and goes straight up.
- **Down.** A VRAM victim whose home is `Nvme` is demoted: a D2H copy into a RAM-cache slot if the RAM policy would admit it against the RAM tier's own victim (same hysteresis), counted against the per-step admission budget (RFC-0021 §8); otherwise it is dropped and read again from NVMe if it returns.
- **Why.** Keeping the VRAM set inside RAM as well (inclusive) would spend up to the VRAM cache's size of RAM on duplicates [estimate: ≈10 GB on scenario b2, ≈2.5% of the routed pool, worth ≈51 ms per token under the uniform prior by RFC-0020's rung-2 arithmetic].

**The RAM cache has two parts**. RFC-0020 §2 and §5 split the RAM cache over an `Nvme` home so that only bytes a device copies count against the DMA budget:
- **Pinned part** (`HostPinned`, `DmaPinned`, `u^P` bytes from the plan, within `pin_limit`): the only RAM source of any host-to-device copy, that is, copied misses and VRAM admissions, and the only RAM destination of demotions. It is registered as io_uring fixed buffers where `io_fixed` allows.
- **Pageable part** (`HostPageable`, locked when allowed, `u − u^P` bytes): in-place misses only. Nothing ever DMAs from it (RFC-0020 §7.5 invariant 5).
- **Which experts sit where.** RAM residents are ordered by decayed count. The pinned part holds the hottest `u^P` bytes of them, because the experts that miss VRAM most often are the ones copied misses and VRAM admissions want. At each adaptation (§5.2), housekeeping compares the hottest pageable resident with the coldest pinned one, and exchanges them when the first beats the second by the hysteresis factor. An exchange is three host copies through a pinned staging slot, issued after both slots' fences, with residency flipped by the owner dispatcher when they complete. It is charged to the same per-epoch byte budget as admissions. Until the flip both experts are served from their old slots.
- **The host-to-device rule.** Every H2D source must be `DmaPinned` and `Resident`. `copy_eligible` (§5.4) marks only RAM-homed `DmaPinned` experts and residents of the pinned part. A VRAM admission of an expert in the pageable part is first copied by the CPU into a pinned staging slot of `ExpertStaging`, then copied from there to VRAM.
- **Shrink.** Rung 2 of RFC-0020 §11.4 releases the pageable part first, then the pinned part.
- On unified-memory machines and CPU-only nodes there is one part, since nothing is DMA'd.

**Layouts.** Every move between tiers is a raw byte copy, so a copy keeps its source's layout. An expert computed by both the GPU (hits, copied misses) and the CPU (in-place misses) either has one shared layout both kernels accept (the default) or is a `per_site_layout` group with a GPU-layout host copy beside its CPU-layout home; then VRAM admissions and copied misses copy from the GPU-layout host copy, and in-place misses read the home (RFC-0020 §7.3). RFC-0011 chooses which, per group.

For RAM-homed experts there is nothing to keep exclusive: the whole pool is resident in RAM as its home, and a VRAM-cache copy is a second copy by design.

On unified-memory machines there is no RAM-homed level: every expert in the `Unified` arena is already where the GPU computes it (RFC-0020 §9).

### 4. Slot geometry and per-layer budgets

**Slot size per layer.** All experts of one MoE layer have the same shapes, and normally the same quant scheme, so a layer has one slot size: `s_l = max over experts of the layer of the expert's pack extent length` (4 KiB-aligned, RFC-0021 §6). Mixed-precision checkpoints that quantize experts of one layer differently get one slot class per distinct size within the layer. Sizing slots per layer rather than uniformly is what fitted 3,556 experts instead of 2,673 in the surveyed engine (`tiered-memory-inference.md` §5.2).

**Per-layer budgets.** `ExpertCacheBudget::per_layer` comes from the solver when present. Otherwise the cache splits `bytes` across layers by marginal value, using the profile:

```text
for each layer l: v_l(n) = h_l(n) · (1/W_miss,l − 1/W_hit,l) · B_expert,l      // expected per-token time saved by n slots
greedy: repeatedly give one slot to the layer with the largest v_l(n_l + 1) − v_l(n_l) per byte, until bytes are used
floor: first give every layer f = min(k_l, ⌊bytes / Σ_l s_l⌋) slots; if bytes cannot give every layer k_l slots,
       f < k_l (possibly 0) and the remaining bytes go by marginal value alone, so the layers with the
       lowest marginal value go without beyond f (in b2, 58 × 8 × 26.5 MB ≈ 12.3 GB exceeds
       the ≈10 GB VRAM cache, so f = 6 slots per layer there) [estimate]
```

`h_l(n)` is the layer's popularity curve (§13). With no profile, `h_l(n) = n / E_l` (uniform), which gives every layer the same number of slots per byte.

**Physical layout.** Slots of all layers are striped across the elastic segments of the `ExpertCache` arena (RFC-0020 §8.3), so a shrink removes slots from every layer in proportion. `ExpertStaging` holds two kinds of slots: **copy slots** for copied misses (enough for the largest per-layer copied share of one pass, double-buffered across adjacent layers) and **prefetch slots** for RAM → VRAM prefetch (`PrefetchWindow::max_inflight` worth).

### 5. Policy, admission and the slot table

#### 5.1 Statistics

After every pass, the dispatcher copies each MoE layer's `RouteSlot` list (expert, tokens) from the doorbell into a lossy statistics ring to the housekeeping thread: a few hundred bytes per layer per step. It is RFC-0001 §8.3's hot-thread ring of statistics deltas, one per dispatcher, and the step loop's other statistics deltas travel in the same ring as other record kinds (revised 2026-10-21). Housekeeping keeps, per (layer, expert), a decayed count `c[l][e]: f32`, incremented by the number of tokens that selected the expert, and per layer the hit, copied-miss and in-place-miss counts. Prefill passes are counted with weight `experts.prefill_weight` (0.1), so one long prompt does not overwrite decode-time popularity.

#### 5.2 Decayed LFU with hysteresis (default policy)

Every `adapt_every_steps` decode steps (default 4), per cache and per layer:

```text
for each e: c[l][e] ← c[l][e] · decay                                 // decay default 0.7 per adaptation
R := resident experts of layer l, ascending by c                        // victims first
C := non-resident experts of layer l, descending by c                   // candidates first
budget := min(max_replace, bytes_allowed_this_epoch / s_l)              // §5.5 bounds the bytes
i := 0
while i < budget and C[i] and R[i] and c[l][C[i]] ≥ hysteresis · max(c[l][R[i]], ε):    // hysteresis default 1.5
    post Replace { layer: l, admit: C[i], victim_slot: slot(R[i]) }   // or Promote if C[i] sits in a prefetch slot
    i += 1
```

These are the surveyed engine's values (`tiered-memory-inference.md` §5.3): adaptation every 4 rounds, decay 0.7, factor 1.5, up to 96 replacements. They are starting defaults, to be tuned against recorded traces in M1 (Open questions); the guard in §5.3 bounds the damage if they are wrong.

#### 5.3 The guard

SC-9 requires a hit rate of at least max(cache fraction, LRU at equal capacity) on every recorded trace. Decayed LFU usually beats LRU, but routing that shifts quickly can favour recency, and scan-like traces can push both below the cache fraction. So, per layer, housekeeping runs three metadata-only **shadows** fed the same access stream, each holding `n_l` entries:
- decayed LFU with hysteresis (the default policy);
- LRU;
- `Freeze`: the resident set as it was at the start of the window, never changed. On traces where nothing beats the uniform prior it holds roughly the cache fraction without spending admissions.

Every `experts.guard_window` (1,024) decode accesses of the layer, the layer adopts the mode of the shadow with the highest hit rate over the last two windows, with no margin. A switch needs that shadow to lead in two consecutive windows (dwell), which is the hysteresis against noise. In `Lru` mode victims are chosen least-recently-used first and admission compares recency order; in `Freeze` mode no replacement is made. Switches are posted as `SetPolicy`. The three shadows cost O(k) per token per layer on the housekeeping thread.

What this guarantees, and what SC-9 should say: after the dwell, a layer runs the policy that led over the previous two windows, so on a trace whose behaviour holds for longer than two windows its hit rate tracks the best of the three shadows. A trace that changes faster than that can still lose up to one window's difference. RFC-0023 therefore asks for SC-9 to be measured on the **aggregate** hit rate over every recorded trace, after a warm-up of 4,096 accesses per layer, with a tolerance of 0.5 points [target] (ARCH follow-up), and the trace simulator gates on that aggregate, not per layer.

#### 5.4 The slot table and the table-delta ring

The MoE kernel finds experts through two device-resident arrays per cache: `slot_of[layer][expert] → u32` (slot index or `MISS`) and `slot_addr[slot] → u64` (fixed for the epoch, RFC-0020 §8). Updating them must not require re-capturing graphs or a host synchronisation (RFC-0001 §5.5). Mechanism, also used by RFC-0022 for KV block tables:

```text
host-mapped TableDeltaRing (one per device, single producer = the dispatcher):
    entries:  { table: u16, index: u32, value: u32 }      capacity: see below
    head:     u64                                          written by the dispatcher with a release store
step parameter block (host-mapped, written by the dispatcher before it launches step s):
    delta_head: u64                                        the ring position step s must apply up to
device:
    applied:  u64 in device memory, echoed to a host-mapped word

step program, first node (captured once):  apply_deltas
    h := param.delta_head                                  // not a volatile read of head: the version is fixed at launch
    for i in applied .. h: table[entry.table][entry.index] ← entry.value
    applied ← h; echo applied to the host-mapped word; record step number s with it
```

Every step applies exactly the deltas posted before its launch, in order, before any kernel of that step reads a table; deltas posted later wait for the next step. The dispatcher learns from the echo at which step `s` each delta took effect.

**One version per step for every site**. The tables the deltas maintain include RFC-0040's `copy_eligible[layer]` bitmap, which marks experts whose current source is `DmaPinned` and `Resident` and so may be copied (§6). The host sites of the MoE step (the dispatcher and the CPU pools) keep a mirror of every table and evaluate step s against the mirror **at `param.delta_head`**, never at a later version. A copy source (a RAM-home range, a RAM-cache slot) is never unpinned, evicted or overwritten before the last step that saw it eligible has completed: its retirement waits for that step's fence (RFC-0020 §7.4).

**Capacity and the KV block tables**. Block tables of a sequence that has not yet run (a new sequence, an attached prefix, a restored session) are written with a direct copy on the copy stream before the sequence's first step, ordered by that step's fence, not through the ring. The ring carries only incremental changes of running sequences and expert-cache replacements, so its capacity is fixed per epoch at the next power of two ≥ `2 · (max_live_sequences · page_classes · (⌈T_max / P⌉ + 1) + max_replace · caches)`, minimum 4,096. With that sizing an overflow means a bug, for KV and experts alike (`EXPERT_DELTA_RING_FULL`, class `Internal`). Providers without graph capture launch `apply_deltas` as the step's first op, with the same semantics. RFC-0040 provides `apply_deltas` for every native provider (it reads `delta_head` from the parameter block at run time, so it is replay-safe); for the ggml provider it is a small host-side copy the dispatcher issues before the step.

#### 5.5 Executing a replacement

```text
Replace { layer l, admit e, victim slot v }       (on the dispatcher, between steps)
  1. post delta slot_of[l][victim(v)] ← MISS; victim copy → Evicting               (RFC-0020 §7.2)
  2. when the echo shows the delta applied at step s: v.free_after ← s
  3. issue the admission copy into v on the copy stream, waiting device-side for step s's completion event:
       RAM-homed e:                 H2D from e's DmaPinned home
       Nvme-homed e in RAM cache:   pinned part: H2D from its slot; pageable part: CPU copy into a pinned
                                    ExpertStaging slot, then H2D from that slot (never DMA from pageable memory)
       Nvme-homed e not in RAM:     read request (PrefetchNear) into a host staging slot, then H2D
     copy → Inflight
     if the victim is Nvme-homed and the RAM policy admits it: D2H demotion copy into a DmaPinned
       RAM-cache slot first, on the same copy stream, before the admission overwrites v (§3)
  4. on the copy's completion (event polled by the dispatcher, never waited on):
       post delta slot_of[l][e] ← v; copy → Resident
       if e is Nvme-homed and came from a RAM-cache slot: free that slot (the sets stay exclusive, §3)
Promote { l, e, staging slot p, victim slot v }    (e already sits, Resident, in prefetch slot p)
  1. post delta slot_of[l][victim(v)] ← MISS; wait for its fence (as Replace steps 1–2)
  2. in-device copy p → v on the copy stream (≈26.5 MB at ≈800 GB/s is ≈33 µs [estimate])
  3. on completion: post delta slot_of[l][e] ← v; p returns to the prefetch pool
  Slot roles never change, so per-layer quotas, striping, slot sizing and shrink victims stay correct
 . Prefetch slots are sized to the largest expert of the stage, because they hold any layer.
```

Between steps 1 and 4 the expert is simply a miss and is served by the miss path, so a replacement never delays a step (RFC-0001 §8.4 rule 7).

**Demand fills**. The RAM expert cache over an `Nvme` home is owned by the dispatcher of the GPU (or `Unified` device) whose misses it serves (RFC-0020 §7.1), so promotion, demotion and demand fills are all writes by one thread. A demand read for an expert missing from both caches lands in:
1. the next slot of the layer's **victim queue** in the RAM cache's pageable part (an NVMe direct read needs no registration; the expert moves to the pinned part later only through the exchange of §3), which housekeeping keeps filled with the slots the policy would evict next and the count threshold a newcomer must beat (popping it on the token path executes a decision already made, not a new one), when the expert's count beats that threshold; otherwise
2. a **demand staging slot** in the host `ExpertStaging` pool, sized `2 · MAX_UNION_l · s_l` for the largest layer (double-buffered across adjacent layers) [estimate for DeepSeek-V3 at T = 1: 2 × 8 × 26.5 MB ≈ 424 MB], freed when its layer's step completes. Admission copies are limited to `io.admit_pcie_share` of the PCIe link per step, after the step's copied misses (RFC-0021 §8); housekeeping converts that into `bytes_allowed_this_epoch`. For a `HostPinned` cache over `Nvme`, steps 3 and 4 are a single read into the slot.

#### 5.6 Initial fill and the profile

During `Warming` (RFC-0001 §3.3 `CacheFill`), each cache is filled in this order until its per-layer budgets are used:

1. the profile's `resident_hint`, newest first: what was resident when the profile was saved;
2. then by profile count, highest first;
3. without its own profile, from a sibling pack's profile of the same `ModelId` on this node (for example after the node gained a device class and built a new pack); the new pack's own counts take over after warm-up, at `experts.curve_window` decode accesses per layer (OQ-MOE-14, answered 2026-10-20; RFC-0041 open question 3);
4. without any profile, by the family's routing bias where the model has one (a per-expert bias term in the router, as in DeepSeek-V3-class models), else round-robin by expert index, which is the uniform prior.

Fill reads run at `Load` priority. The runner may reach `Ready` before the fill finishes; unfilled slots are misses until their copies land.

#### 5.7 Upgrade path

`CachePolicy` is an enum on purpose. A learned recency-frequency policy or a forecast-driven Belady policy (`tiered-memory-inference.md` §5.5) can be added as a variant, provided it beats `DecayedLfu` on the recorded traces and keeps the guard.

### 6. The MoE step: hits, copied misses, in-place misses

This expands RFC-0001 §6.3 step 3 for one MoE layer `l` on a node with a discrete GPU. The captured step graph contains every GPU node below; the only host actions are the dispatcher's copy issue and the CPU pool's work, both driven by host-mapped flags (RFC-0001 §8.2).

```mermaid
sequenceDiagram
  participant G as GPU compute stream (captured graph)
  participant D as Dispatcher thread
  participant C as GPU copy stream
  participant P as CPU pool (NUMA-local)
  G->>G: router(l) + fused predictor(l+1..l+d)
  G-->>D: doorbell(l): route[] with cache_slot or MISS, pred[]; Rd(l) = 1
  G-->>P: same doorbell; Rc(l, d) = 1 per NUMA domain d (each consumer resets its own flag)
  G->>G: grouped expert kernel over hits (VRAM slots), shared expert
  D->>D: split misses with the step's balance parameters (§8): copied set, in-place set
  D->>C: H2D copies of the copied set into copy slots; then F(l) = 1 (written directly if there are no copies)
  P->>P: same split, computes the in-place set from the pinned arena (§7)
  P-->>G: partial outputs + Q(l, d) = 1, every step, even with no in-place experts
  G->>G: wait F(l) EQ 1, write 0; grouped expert kernel over copy slots
  G->>G: wait Q(l, d) EQ 1 for every d, write 0; combine: hits + copied + CPU partials (router weights)
```

**The split is deterministic.** The dispatcher and the CPU pool each compute the split of the misses from the same inputs: the doorbell's miss list and the step's balance parameters (§8), which the dispatcher writes into a per-step parameter block before launching the step. The rule: order misses by `tokens` descending, then expert index; walk in that order and assign a miss to the copied set if it is marked in `copy_eligible[l]` at the step's table version (`param.delta_head`, §5.4), the copied bytes stay at or below the step's `B_copy(l)`, and layer l's half of the copy slots is not yet full; everything else is in-place. A miss whose source is not `DmaPinned` and `Resident` is never marked eligible (RFC-0020 §6), and `ExpertStaging` is fixed per epoch, so the copied set can never exceed the mapped copy slots. The i-th expert of the copied set goes to copy slot i of layer l's half of the double-buffered copy slots. The GPU's second expert kernel applies the same rule to the doorbell to find each copied expert's slot, so no copy plan travels from host to device. The dispatcher, the CPU pool and the GPU need no messages beyond the doorbell and the two flags, and they agree by construction.

**Misses on `Nvme`.** A miss whose home is `Nvme` and that is not in the RAM cache is a demand read (RFC-0021 priority `Demand`), unless a prefetch for it is already in flight, in which case it is promoted to `Demand`. It always goes to the in-place set: the CPU pool waits on its completion flag, which the dispatcher sets when the I/O completion arrives, and then computes it from the RAM slot. In exact mode the step waits for it (glossary: demand read).

**Stream-ordered waits.** Doorbell flags use RFC-0040 §4.2's constant-valued consumer-reset handshake. Each consumer has its own flag: `Rd(l)` for the dispatcher, `Rc(l, d)` for each NUMA domain's pool. The copy stream writes `F(l) = 1` after the copies (the dispatcher writes it directly when there are none), and the GPU resets it after its wait. Domain d's pool writes `Q(l, d) = 1` every step, even with no in-place experts, and the GPU resets it. Captured waits are `EQ 1` followed by a write of 0, so no captured value depends on the step. `seq`/`step` stay as payload checks, not flags. Metal events keep per-step values. On CUDA and HIP the captured waits and writes are native stream wait-value and write-value operations, which occupy no SM; Metal and Vulkan use their event and timeline paths (RFC-0012). Spin and store kernels running the same constant-valued handshake (store 1; wait for 1, then write 0; never a comparison against a step number) remain only as the fallback where native wait-value operations are absent (RFC-0040 §4.2). If a provider has no device-side wait at all, the dispatcher splits the step at MoE layers and launches the remainder after the flags, which costs launches but no correctness (ADR-0014 rule 2).

**Combine.** The combine kernel sums the routed experts' outputs weighted by their router weights in a fixed order (by expert index), regardless of which site computed them, so the result does not depend on the split beyond rounding. CPU partial outputs are written in F32 (ADR-0019 accumulation rules, RFC-0042).

**Several GPUs in one node.** Under `ParallelismMode::Local` with layers split across GPUs, each layer's experts belong to the cache of the GPU running that layer. Under `IntraNodeTp`, experts are sharded with the layer and every member caches the same experts' shards: the primary's policy decides, and members apply identical slot-table deltas at the same step (§12).

### 7. CPU execute-in-place with NUMA locality

- **Where experts live.** On a multi-socket or multi-NUMA host, the solver homes each layer's RAM-resident experts across the NUMA domains' host tiers (RFC-0020 §2), interleaving them by popularity so each domain carries a similar share of expected miss traffic. Each domain's CPU device computes only the experts homed in its own domain, so every in-place byte is read from local memory (the surveyed NUMA-aware design reported up to 63% more decode on dual-socket machines, `tiered-memory-inference.md` §2.3).
- **Who computes.** Each NUMA domain's CPU compute pool (RFC-0001 §8.2) spins on the layer's doorbell, takes the in-place misses homed in its domain, and splits them across its cores by a fixed per-core assignment: whole experts per core when there are at least as many experts as cores, otherwise each expert's rows across cores with a barrier (ADR-0012 item 2). There is no work stealing.
- **Verify windows and batches.** An in-place expert selected by several tokens is computed as one small GEMM over those tokens, so its weights are read once per pass (§11).
- **Kernels.** RFC-0040's CPU provider supplies the quantized expert GEMV/GEMM per ISA (AVX-512/VNNI, AMX, AVX2, NEON, SME2; no SVE2 kernels in v1, ADR-0402). Whether a kernel is RAM-bound or compute-bound on this CPU decides the right copied share (`tiered-memory-inference.md` §14), which is why the balancer fits it per layer instead of assuming.
- **Outputs.** Each domain writes F32 partial outputs into its slice of a host-mapped buffer and sets its own flag word; the combine waits for all of them (`Q(l)` is one word per domain).

### 8. The online three-site balance

The three sites of a MoE layer are the GPU over VRAM hits, the PCIe copy engine feeding the GPU, and the CPU pools over RAM. The balancer chooses, per layer and per step, how many missed bytes to copy, `B_copy(l)`, to minimise the layer's time (ADR-0211).

**Cost model per layer** (fitted; bytes are missed-expert bytes in this pass):

```text
T_cpu(l)  = a_l + c_l · B_inplace + d_l · B_copy        d_l: DMA reads slowing the CPU pool's DRAM reads
T_gpu(l)  = g_l + p_l · B_copy + k_l · B_hit             p_l ≈ 1/W_pcie; copied experts run after their copies land
B_inplace = B_miss − B_copy
layer time ≈ max(T_cpu(l), T_gpu(l))
```

Setting `T_cpu = T_gpu` gives the continuous optimum, clamped to the bytes eligible for copying:

```text
B*_copy(l) = clamp( (a_l + c_l·B_miss − g_l − k_l·B_hit) / (c_l − d_l + p_l),  0,  B_copyable )
```

`B_copy(l)` for the step is `B*_copy(l)` evaluated with the step's expected `B_miss` and `B_hit` (the last 8 steps' averages for that layer), and the deterministic walk of §6 turns it into whole experts. It is written into the step's parameter block before launch, so the decision costs nothing on the token path.

**Fit.** Each step yields one sample per layer: `T_cpu` from the pool's timestamps (domain start to flag), `T_gpu` from device timestamps written by the kernels into host-mapped telemetry (the global timer on CUDA, `MTLCounterSampleBuffer` on Metal), and the byte counts. Housekeeping fits `(a, c, d)` and `(g, p, k)` per layer by recursive least squares with forgetting factor `experts.balancer_forget` (0.98 per sample). Until a layer has 32 samples, `B_copy` comes from the plan's `MissSplit::copied_share`. The surveyed engine's closed-form share `L / (L + 1.85 R)`, clamped to 0.10–0.60 (§5.4 of the research note), is the prior the solver uses when RFC-0010 has no fit yet; the constant was fitted to three machines, so it is only a prior.

**Guards.**
- The fit is rejected, and the prior kept, if any coefficient is negative or the residual exceeds 30% of the mean sample; the layer is flagged in telemetry.
- `B_copy` changes by at most 25% of `B_miss` from one step to the next, so a noisy sample cannot swing the split.
- A deterministic run (`MissSplitMode::Fixed`, ADR-0019 item 8) freezes the split for its duration.

**Unified-memory devices** have no PCIe term. Where RFC-0010 shows that the CPU and the integrated GPU together read more bandwidth than the GPU alone (expected on Strix Halo, whose CPU side is CCD-link-limited, `hardware-and-performance-model.md` §2.1), the same fit runs with the two compute sites and `p_l` replaced by the contention coefficient; otherwise the GPU computes every expert.

**Helper GPUs.** A node whose plan gives a layer's experts disjoint caches on several GPUs (ADR-0006 item 5) adds one site per GPU to the same fit. That is an M3 extension; v1 balances one GPU and the CPU pools.

### 9. Prediction

Prefetch needs to know a layer's experts before its router runs. Two sources, by family capability (`RoutingKnown`, from the manifest, RFC-0043):

**Exact knowledge.**

| `RoutingKnown` | Known when | Used for |
|---|---|---|
| `TokenHash` (hash-routed layers, such as DeepSeek V4-class `num_hash_layers`) | the token is sampled | every hash-routed layer's experts are requested one step ahead, like `RowLookup` rows (RFC-0001 §6.3 step 5), with `PrefetchNear` priority |
| `PreGate { lookahead }` | layer `l`'s pre-gate runs | layer `l + lookahead`'s experts, exactly |
| `PreAttention` | the start of layer `l` | layer `l`'s experts, one attention block ahead |
| `AtRouter` | layer `l`'s router runs | needs prediction (below) |

**Next-layer prediction (default).** The residual stream changes little between adjacent layers, so layer `l+j`'s router applied to layer `l`'s MoE input predicts layer `l+j`'s experts well (`tiered-memory-inference.md` §6). Syntropy computes this on the device that runs layer `l`'s router, fused into the router op as extra rows of the same GEMV, and writes the top `k + experts.predict_extra` predicted experts per lookahead layer, with their scores, into the doorbell's `pred` field (ADR-0210). The predictor needs the `Router` groups of layers `l+1..l+d`, which are already on that device as part of the hot set. Only when a family's router runs on the CPU (CPU-only nodes) does a host copy of the routers feed the predictor.

This differs from the surveyed design, which applied a host copy of the next router on a CPU thread (§6). On a discrete-GPU desktop the CPU's DRAM bandwidth is the scarce resource that computes in-place misses (scenario b2), while the router weights are already in VRAM, so the GPU computes the prediction for a few microseconds per layer (§Performance) instead of taking DRAM bandwidth from the misses.

**Across a stage boundary.** For the first `d` MoE layers of the next stage, the stage needs their routers. The plan places replicas of those small `Router` groups on the previous stage (`TierAssignment::replicas`, RFC-0001 §5.4), and the predictions travel as `RouterHints { layer, experts }` in the `ActivationFrame` side channel (RFC-0001 §5.8), so the next stage's prefetches start before its first frame arrives (M2).

**Depth.** `PrefetchWindow::lookahead_layers` from the plan: 1 by default for RAM-homed experts, 2 for `Nvme`-homed ones, where a read takes longer than one layer's compute. If a layer's measured recall at depth `j` (the fraction of selected experts that were predicted) stays below 0.3 for 1,024 steps, prefetch at that depth is turned off for that layer, and back on when a probe window of 64 steps measures recall above 0.4.

**Verify windows and batches.** The predictor runs for every token in the pass, and predicted experts are merged across tokens with summed scores.

### 10. Prefetch

Prefetch moves predicted or known experts one tier closer to their compute site before they are needed, within the plan's `PrefetchWindow { from, to, max_inflight, lookahead_layers }` entries. When a window is full, new predictions are dropped, not queued (RFC-0001 §8.4 rule 4).

| Boundary | Issued when | Lands in | Used as | Priority (RFC-0021) |
|---|---|---|---|---|
| `Nvme` → `HostPinned` (or `Unified`) | a predicted or known expert is not resident in RAM (or `Unified`) | a RAM-cache slot if the RAM policy admits it (prediction score counts as a provisional access), otherwise a host staging slot | an in-place miss that needs no demand read; or the source of a VRAM copy | `PrefetchNear` for exact knowledge and depth 1; `PrefetchFar` beyond |
| `HostPinned` → `Vram` | a predicted expert is not in the VRAM cache, its score is above `experts.vram_prefetch_min_score`, and the balancer's model says the PCIe link has idle time this step | a prefetch slot in `ExpertStaging` | a hit; promoted into the cache if the policy then admits it (§5.5 `Promote`) | copy stream, after copied misses and within `io.admit_pcie_share` |
| another node's RAM → local (M3) | the plan homes an expert group on another node's `HostPinned` tier, which pays only on RDMA-class links | a local RAM or VRAM slot | as above | bulk-plane read, below the data plane (RFC-0031) |

Rules:

- **Useful-until.** Each prefetch carries `useful_until = (step, layer)` of its target use, so the I/O thread drops it if the consumer has passed it before submission, and cancels it in flight where the engine can (RFC-0021 §3).
- **Demand wins.** A demand read for an expert already being prefetched is merged with the prefetch and promoted to `Demand`, never issued twice.
- **NVMe prefetch is gated on expected value**. When the volume is the bottleneck (`VolumeIoStats.busy` ≥ 0.8 of the window, RFC-0021 §11), a wasted read costs as much drive time as a useful one saves. A candidate with measured precision p for its (depth, score bucket) saves `p · t_stall` of stall and wastes `(1 − p) · s_e / W_nvme` of drive time, so it is read only if `p · t_stall > (1 − p) · s_e / W_nvme`, with `t_stall` the layer's measured demand-read stall (prior `s_e / W_nvme`, which gives p > 0.5). Precision starts at a prior of 0.5 and is measured once 1,024 predictions per (depth, bucket) have been scored. On a volume that is not busy, every candidate in the window may be read. Prefetch bytes per token are also capped at `(1 + experts.prefetch_slack) · (1 − h_ram) · (1 − h_v) · B_r` from the plan's prediction (slack 0.25), so a run of bad predictions cannot crowd out demand reads.
- **What "issued" means for SC-8.** A read counts as issued early when it has been submitted to the device (not merely posted to the I/O ring) before its layer's router runs. ARCH follow-up: state this in SC-8.
- **RAM → VRAM prefetch is conditional.** It helps only while the CPU pool is the bottleneck of the layer and the link is idle, which the balancer's fitted model predicts per layer: prefetching an expert saves its in-place cost `c_l · s_e` and spends `p_l · s_e` of link time. The prefetcher issues it only when `T_cpu(l) > T_gpu(l)` in the step's prediction and the copy fits in the link's predicted idle time. The surveyed engine did not use router lookahead for this boundary (`tiered-memory-inference.md` §6); Syntropy enables it by default under this condition and measures it (Open questions).
- **Accounting.** A prefetched expert that is used counts as a hit for the cache that holds it and as an early issue for SC-8; one that is not used counts against the layer's precision.

### 11. Verify windows and prefill

**Verify windows.** A pass of T tokens (a verify window or a small batch) touches the union of their experts, `U(T) ≤ min(E, T·k)` per layer, uniformly `E·(1 − (1 − k/E)^T)` (cost model §3.9). The doorbell lists each distinct expert once with its token count, so:

- a hit is read once from VRAM for all its tokens;
- the deterministic split (§6) puts the experts with the most tokens first into the copied set, because the GPU amortises a copy over more tokens than the CPU amortises its read;
- an in-place expert is one small GEMM over its tokens (§7);
- `MAX_UNION` in the doorbell is sized for the plan's largest pass: `min(E, P_max · k)`, with `P_max` that pass's tokens (§2).

This is the MoE limit of speculation (RFC-0001 §6.4): the solver sizes γ knowing that a window of T tokens costs `U(T)` expert reads per layer, and on tiered MoE plans often chooses MTP with γ = 1 (ADR-0017).

**Prefill chunks.** A chunk of tens of tokens touches nearly every expert of a layer (cost model §3.5, §3.9). At or above the layer's `n*` (RFC-0020 §10, applied to expert kernels), the layer's misses are streamed to the GPU in expert order and computed there:

```text
for each MoE layer l of the chunk:
    group the chunk's tokens by selected expert
    hits: computed from the VRAM cache
    misses, in expert order: H2D from the RAM home (or NVMe → RAM staging → H2D) into a ring of copy slots,
        computed per expert as the copies land; the copy of expert i+1 overlaps the compute of expert i
```

The ring uses the copy slots plus, for the duration of the chunk, prefetch slots lent from `ExpertStaging`. Below `n*`, misses are split as in decode.

**What bounds prefill**. A chunk of ≥ 256 tokens touches essentially every expert of a layer: the chance that a given expert is untouched is `(1 − 8/256)^256` ≈ 3·10⁻⁴ for DeepSeek-V3 [estimate]. So every chunk reads the whole routed pool once, and the part of the pool not resident in RAM or VRAM comes from NVMe. Per chunk and stage:

```text
t_pool(chunk) = max( B_pool,RAM / W_pcie ,  B_pool,NVMe / W_nvme ) + overlap losses
```

In b2 the NVMe-resident part is ≈394 − 115 − 10 ≈ 269 GB, ≈45 s per chunk at 6 GB/s, so a 32K prompt in 8,192-token chunks would read it four times, ≈3 minutes [estimate]. The streaming term is NVMe-bound, not PCIe-bound, and the remedy is to read the pool once per prompt, not once per chunk:
- **Layer-major prefill for `Nvme`-homed pools.** A stage whose routed pool is not wholly in RAM runs the prompt's chunks layer by layer: every chunk passes layer l (its experts streamed once, in expert order, serving all chunks' tokens grouped by expert) before any chunk enters layer l+1. The activations of all chunks are held in the stage's scratch, `prompt × hidden × streams × 4` bytes [estimate for DeepSeek-V3: 32K × 7,168 × 4 ≈ 0.94 GB], so the group of chunks run layer-major is bounded by the scratch budget, and a prompt longer than that is processed in as few groups as fit. The pool is read once per group.
- **The cost rule.** RFC-0012's chunk and group sizing and SC-18's streamed term use `t_pool` with `W_nvme` for the NVMe-resident bytes (follow-ups to RFC-0012, RFC-0011 and ARCH for SC-18). RFC-0011's fixed 8,192-token chunk cap does not apply to stages running layer-major.
- **Pipelines.** Layer-major prefill delays the next stage until the group finishes, so stage overlap is lost for that prompt; the NVMe read dominates such stages anyway. Stages whose pool is RAM-resident keep chunked, pipelined prefill.

### 12. Single ownership and hot replication

- **Across the cluster.** Every expert group has one owner node (`TierAssignment::owner`, RFC-0001 invariant 11). Under pipeline parallelism each stage owns its layers' experts, so ownership follows from the layer ranges. Under `CrossNodeEp`, `ExpertOwnerMap` names the owner of each expert, and a hot expert may have explicit replicas in `TierAssignment::replicas`, chosen by the solver from this RFC's popularity buckets (§13). Routing a token to a replicated expert is a cluster-consistent decision inside the collective (RFC-0032): every member picks the same replica by a fixed rule (the replica on the token's own node if there is one, else the lowest node ID among holders), so no message is needed to agree.
- **Inside a node.** An expert is cached by at most one device's cache at a time. With layers split across GPUs this is automatic. With disjoint helper-GPU caches (M3), a node-level owner word per (layer, expert) is set by an atomic compare-exchange from `NONE` to the admitting device before its admission copy, and cleared after eviction; a device never admits an expert another device owns. That removes the duplicate residency that cost ≈2,000 slots in the surveyed engine (`tiered-memory-inference.md` §16). Under the exclusive rule of §3, an `Nvme`-homed expert sits in the RAM cache or the VRAM cache, never both, except while a promotion or demotion copy is in flight.
- **Under `IntraNodeTp`.** Each member caches its shard of the same experts. The primary's policy decides; replacements are broadcast as slot-table deltas tagged with the step at which they take effect, and every member applies them at that step (RFC-0032 owns the collective; this RFC owns the rule that all members agree).

### 13. Learned profile and telemetry for the solver

**The profile.** `ExpertProfile` (§2) is written as the pack's `XPROFILE` side record, `side/<pack-key>.xprofile`, with its own schema version and not into the sidecar, at `Draining`, every `experts.profile_write_every` (10 min) while `Ready`, and at shutdown, by atomic rename (RFC-0041 owns the file). It records decayed counts normalised per layer, the resident set at save time in eviction order, and the census digest. A profile whose census digest does not match is ignored. The next start fills the cache from it (§5.6), so it begins where the last run ended.

**What the solver gets.** RFC-0011 plans expert placement as byte budgets over learned curves (ADR-0102) and reads them from `WorkloadStats::hit_curves` and `WorkloadStats::union` (RFC-0011 §2). This RFC owns their contents and produces them per MoE layer. RFC-0011 uses hit-curve segments in place of ADR-0008's popularity buckets, so no buckets are published.

```rust
/// Defined in RFC-0011 §2; produced here. One per MoE layer.
/// H_l(c): the hit fraction of the layer's routed reads that this RFC's cache policy achieves
/// with c bytes of the layer's experts resident in fast memory (VRAM ∪ RAM cache, exclusive
/// sets, §3). Concave, piecewise linear, non-decreasing; H(0) = 0, H(pool) = 1.
pub struct HitCurve { pub layer: LayerIx, pub pool: Bytes, pub points: SmallVec<[(Bytes, f32); 9]>, pub source: MeasureSource }

/// Named in RFC-0011 §2; defined here. One per MoE layer.
/// U_l(T): expected distinct experts of the layer touched by one pass of T tokens.
pub struct UnionCurve {
    pub layer: LayerIx,
    pub experts: u16,                                   // E_l
    pub top_k: u16,                                     // k_l
    /// T consecutive tokens of one sequence: verify windows, drafts (§11).
    pub window: SmallVec<[(u16, f32); 8]>,              // T ∈ {1, 2, 4, 8, 16, 32, 64, 128}
    /// T tokens from different sequences: batches in throughput mode.
    pub batch: SmallVec<[(u16, f32); 8]>,
    pub source: MeasureSource,
}

/// What a runner reports for its layers (Proposed amendments, item 5).
pub struct ExpertWorkloadReport {
    pub model: ModelId,
    pub census: Blake3,
    pub hit_curves: Vec<HitCurve>,
    pub union: Vec<UnionCurve>,
    pub tokens_observed: u64,                           // decode tokens behind the curves
    pub fit: Vec<(LayerIx, BalancerFit)>,               // §8 coefficients, for RFC-0010's profile
    /// Measured hit rate per cache tier and layer, in RFC-0011's `WorkloadStats::tier_hit` shape
    ///. VRAM: hits / routed reads, compared with H_l(y_l). RAM cache over an Nvme
    /// home: RAM hits / VRAM misses, compared with (H_l(u_l + y_l) − H_l(y_l)) / (1 − H_l(y_l)).
    pub tier_hit: Vec<(TierId, LayerIx, Measured<f32>)>,
}
```

**Hit curves: how they are measured.**

- **Grid.** Per layer, slot counts `n ∈ {0} ∪ {k·2^i < E} ∪ {E} ∪ {current n_l}`, at most 9, dropping the `k·2^i` point closest to `n_l` if there are more; bytes are `n · s_l`. The grid depends only on the layer's geometry and current budget, so runners of the same model produce comparable points.
- **Shadow caches.** Housekeeping runs, per layer and grid point, a metadata-only shadow of the configured policy (decayed LFU with hysteresis and the guard, §5.2–§5.3) at that capacity, fed the same access stream. The decayed counts are shared across shadows; each shadow keeps only its resident set. A shadow's hit fraction over the last `experts.curve_window` (16,384) decode accesses of the layer is the point's value. This measures what the policy *achieves* at each capacity, not the oracle share of the top `n` experts, which would overstate it. Cost: with DeepSeek-V3's 58 MoE layers and 9 points, ≈4.2K bitset lookups per token and ≈130K operations per adaptation [estimate], a small fraction of the housekeeping thread.
- **Shape.** RFC-0011 needs a concave curve. The published points are the upper concave hull of the shadow values together with (0, 0) and (pool, 1), made non-decreasing. Where the hull exceeds a shadow value by more than 0.03, the gap is reported in `ExpertCacheDigest::hull_gap` so drift detection can tell over-prediction from a real workload change.
- **Source.** `Prior`, with `H_l(c) = c / pool_l` (the uniform prior RFC-0011 names), until the layer has 4,096 decode accesses; `Online` after. Prefill accesses are excluded, because the solver's routed term is a decode term.
- **What it does not include.** The shadows do not model admission bandwidth. A tier whose admissions are throttled (RFC-0021 §8) can fall short of the curve; its measured hit rate is reported separately (`ExpertCacheDigest::tier_hit`) and a persistent shortfall is drift.

**Union curves: how they are measured.**

- **Directly** from every pass's doorbell: `(T, n_route)` per layer for the window and batch shapes that actually run (decode T = 1, verify windows T = γ + 1, throughput batches).
- **By resampling** for the other grid values: housekeeping keeps a reservoir of the last 4,096 decode tokens' selections per layer (≈3.8 MB for DeepSeek-V3 [estimate]). `window(T)` is the mean distinct count over sliding runs of T consecutive tokens of one sequence; `batch(T)` is the mean over T tokens drawn from different sequences, or, in single-user use, from points at least 256 tokens apart, which approximates independent sequences.
- **Shape.** Non-decreasing and concave in T, with `k ≤ U(T) ≤ min(E, k·T)`; direct measurements win at their own T.
- **Source.** `Prior` (the uniform formula `E·(1 − (1 − k/E)^T)` for both shapes) until 1,024 decode tokens are in the reservoir; `Online` after.

**Reporting.** The runner sends `ExpertWorkloadReport` through its daemon to the coordinator (Proposed amendments, item 5) when, for any layer, a hit curve moves by more than 0.02 at the plan's current per-layer budget or a union value moves by more than 5% at the plan's window or batch size, at most once per 60 s per instance, and at `Draining`. The coordinator merges reports from replicas of the same model by token-weighted averaging at identical grid points, and RFC-0011 decides whether a delta solve follows (RFC-0011 §12.3). The curves are also persisted with the profile, so a restart or a new instance of the same model starts from them rather than from the prior.

**Corpus-derived curves.** Before a user's own machine has run a model, the only defensible decode rate for a tiered MoE plan is the uniform prior (≈ 0.7 tok/s for DeepSeek-V3 Q4 on a 128 GB PC, RFC-0011 §15.2), while the plan's headline case needs a learned h ≈ 0.9. The project therefore publishes, per MoE family, hit and union curves measured by replaying its own traces of the public parity corpus (S-01, TIER-E8-T09) through this policy. They carry no user data. The preview shows them as the upper end of a labelled range ("measured on the project's corpus, not on your use"), the uniform prior as the lower end (RFC-0011 §14.1). The solver still plans from the uniform prior until the node's own curves exist, because no seed profile ships in v1 (OQ-MOE-03).

**Who may see a profile (OQ-MOE-13, answered 2026-10-20).** A learned profile, its curves and its `ExpertWorkloadReport` reflect what users asked, and can be personal data. Rules:
- **Among the nodes one operator runs** in a cluster, profiles and curves are shared: persisted per node, reported to the coordinator, merged across replicas (above). Those machines already see the prompts.
- **With a node run by another operator** (an invite-joined member, RFC-0030, RFC-0034), nothing is exchanged unless both operators opt in (`experts.share_profiles`, node scope, set on both sides). Without both opt-ins, the coordinator sends that node only the budgets in its plan; the node keeps its own counts in memory for its cache policy, persists no `XPROFILE` derived from the other operator's traffic, and sends no `ExpertWorkloadReport`, so the solver plans that node's layers from the uniform prior.
- **Never published.** No profile or curve derived from users' prompts leaves the cluster: not with packs pushed to a hub, not over P2P to other clusters, not in benchmark reports.
- **Corpus-derived seed profiles.** The project may publish profiles derived from its public corpus as fill seeds for new installs, only if S-01 shows that the cold start costs a material share of the first session's hit rate; that reverses OQ-MOE-03's "no seed in v1" and is decided in TIER-E10-T13.

**Measured rates** for telemetry and the dashboard: hit rate per cache tier, copied and in-place shares, prediction precision and recall per depth, the share of NVMe misses issued early (SC-8), admission bytes and churn. The balancer's fitted coefficients go in the report's `fit`, which answers RFC-0011's question on whether DMA interference `d` is linear in the copy rate: the fit's residuals are reported against copy rate. Per-step counters go to the telemetry ring, and the metrics named in §15 go to RFC-0053.

### 14. Approximate knobs

None of these is ever a fallback (RFC-0001 §10.2). Each is opt-in, carried in `ExactnessPolicy::Approximate { knobs }`, published with its measured KL and top-1 cost before it is offered, and reported in every `TokenEntry::approx` it touches (ADR-0019 §6, §7).

| Knob | Scope | Behaviour |
|---|---|---|
| `expert_skip` | sequence | a miss that is not among the token's `experts.skip_keep` highest-weighted selected experts, and which is not resident in RAM, is not computed; the remaining weights are renormalised to sum to the original total |
| `expert_defer` | sequence | in-place misses of layer l may complete during layer l+1's attention, and their contribution is added at layer l+1's combine; the model then sees a delayed residual update |
| `miss_low_precision` | instance (needs plan bytes) | misses are computed from a lower-precision copy of the expert held in VRAM; the plan budgets the copies |
| `topk_reduce` | sequence | the router keeps `k' < k` experts per token and renormalises |

Exact mode never skips: a missed expert on `Nvme` is a demand read and the step waits for it.

### 15. Configuration

Schema in RFC-0054. `instance` scope; most are reloadable because they only tune runtime behaviour.

| Key | Default | Reload | Meaning |
|---|---|---|---|
| `experts.policy` | `decayed_lfu` | restart | `decayed_lfu` / `lru` / `static` |
| `experts.adapt_every` | 4 decode steps | hot | adaptation period (§5.2) |
| `experts.decay` | 0.7 | hot | decay per adaptation |
| `experts.hysteresis` | 1.5 | hot | replacement factor |
| `experts.max_replace` | 96 per cache per adaptation | hot | upper bound on replacements |
| `experts.guard` | `true` | hot | the three-shadow guard (§5.3) |
| `experts.share_profiles` (node) | `false` | hot | exchange learned profiles and curves with nodes run by another operator; takes effect only when both nodes set it (§13). Nodes one operator runs share regardless |
| `experts.guard_window` | 1,024 decode accesses per layer | hot | guard evaluation window; a switch needs two consecutive leading windows |
| `experts.prefetch_slack` | 0.25 | hot | prefetch bytes per token above the predicted miss bytes (§10) |
| `experts.prefill_weight` | 0.1 | hot | weight of prefill accesses in the counts |
| `experts.predict_extra` | ⌈k/2⌉ | hot | extra predicted experts per lookahead layer. The router writes the epoch's value, and `MAX_PRED` is sized from it (§2). A hot change below it lowers how many of those predictions the prefetcher uses. A raise above it requests a new epoch with `ReplanReason::Config { keys }`, as RFC-0001 §7.2 does for a `hot` key the solver reads; that epoch re-sizes `MAX_PRED` and re-captures, so the raise applies once the instance adopts the new plan (revised 2026-10-21) |
| `experts.vram_prefetch` | `auto` | hot | `auto` / `off`: RAM → VRAM prefetch (§10) |
| `experts.vram_prefetch_min_score` | 0.5 (normalised router probability mass) | hot | prediction score needed for RAM → VRAM prefetch |
| `experts.balancer` | `fitted` | hot | `fitted` / `fixed` (§8) |
| `experts.balancer_forget` | 0.98 | hot | RLS forgetting factor |
| `experts.profile_write_every` | 10 min | hot | §13 |
| `experts.curve_window` | 16,384 decode accesses per layer | hot | shadow hit-curve window (§13) |
| `experts.skip_keep` | k − 2 | hot | `expert_skip` only |

Metrics (RFC-0001 §13): `syntropy_tier_expert_hit_ratio{device,tier,layer_group}`, `syntropy_tier_expert_miss_total{kind=copied|inplace|demand}`, `syntropy_tier_expert_copied_share{layer_group}`, `syntropy_tier_expert_predict_recall{depth}`, `syntropy_tier_expert_nvme_early_ratio`, `syntropy_tier_expert_replacements_total`, `syntropy_tier_expert_guard_mode{layer_group}`. Layer groups are buckets of 8 layers, keeping label sets bounded.

### 16. Failure handling and error codes

Domain `Expert`.

| Code | Class | Raised when | Action |
|---|---|---|---|
| `EXPERT_DEMAND_FAILED` (1) | `Integrity` | a demand read failed after retries (RFC-0021 `IO_EXTENT_BAD`) | another resident copy is used if one exists; otherwise the step's sequences fail with `Integrity` and the extent is rebuilt |
| `EXPERT_COPY_FAILED` (2) | `Device` | an admission or copied-miss copy reports a device error | sticky errors fail the runner (RFC-0001 §10.1); a copied miss's expert is recomputed in place for that step if the device is still healthy |
| `EXPERT_DELTA_RING_FULL` (3) | `Internal` | more than 4,096 table deltas pending | the dispatcher stops posting replacements until applied; counted; indicates a policy bug |
| `EXPERT_PROFILE_MISMATCH` (4) | `Drift` | the `XPROFILE` record's census digest or schema version differs | profile ignored; uniform prior; logged |
| `EXPERT_FIT_REJECTED` (5) | `Drift` | balancer fit fails its guards (§8) | prior split kept for the layer; flagged in telemetry |
| `EXPERT_OWNER_CONFLICT` (6) | `Internal` | two devices try to own one expert in a node | the second admission is abandoned; counted |

Every failure leaves the expert served by a slower correct path or stops the sequence; none can produce output from the wrong expert, because the slot table points only at `Resident` copies (RFC-0020 §7.5 invariant 4).

## Performance model and budgets

All figures are **[estimate]** with derivation or **[target]**.

**The MoE term with a VRAM cache.** SC-5's overlap model (vision §7.2, cost model §3.3) leaves the VRAM-cache term to this RFC. Per token, summing over MoE layers, with `B_r` routed bytes, `h_v` the VRAM-cache hit rate (by bytes), `s` the copied share of VRAM misses, and `h_ram` the RAM hit rate for `Nvme`-homed experts:

```text
t_overlap = B_hot / W_gpu
          + max( h_v · B_r / W_gpu  +  s · (1 − h_v) · B_r / W_pcie ,       GPU side: hits, then copied misses
                 (1 − s) · (1 − h_v) · B_r / W_ram_exec ,                   CPU side: in-place misses
                 ((1 − h_ram) · (1 − h_v) · B_r + B_waste) / W_nvme )       NVMe side: misses not in RAM, plus mispredicted prefetch bytes
```

`B_waste` is the bytes per token prefetched from NVMe and never used, measured; the value gate of §10 keeps it small on a busy drive. Without it, one wrong candidate per layer that misses RAM would add 58 × 26.5 MB ≈ 1.5 GB per token in b2, ≈256 ms at 6 GB/s, more than the whole NVMe term [estimate]. With `h_v = 0`, `s = 0` and `B_waste = 0` it reduces to SC-5's formula. [Target] SC-5: `t_tok ≤ 1.10 · t_overlap` with `h_v`, `s`, `h_ram` and `B_waste` measured; ARCH follow-up: SC-5's formula should carry the `B_waste` term.

**Scenario b2** (RTX 4090, 128 GB DDR5, Gen4 NVMe; DeepSeek-V3 Q4_K_M): `B_r` = 12.3 GB, `W_ram_exec` = 70 GB/s, `W_nvme` = 6 GB/s. At `h_ram` = 0.9 and `h_v` = 0 the overlap model gives ≈217 ms, ≈4.6 tok/s (vision SC-5 example). The VRAM cache left after the hot set (9.7 GB), 32K of KV (2.1 GiB) and buffers is ≈10 GB, 2.5% of the 394 GB routed pool. Under the uniform prior that is `h_v` ≈ 0.025, worth ≈5 ms per token. Its real value depends on routing skew: the surveyed 512-expert, top-10 model reached 0.72 with ≈18% of its expert bytes cached (`tiered-memory-inference.md` §5.2, §5.3), while DeepSeek-class load-balanced routing is expected to be flatter (cost model §6 item 4). M1 measures it; nothing in this design depends on the skew being high.

**Prediction cost.** A DeepSeek-V3 router is 256 × 7,168 BF16 weights ≈ 3.7 MB. Predicting one layer ahead for 58 MoE layers reads ≈215 MB from VRAM per token, ≈0.27 ms at 800 GB/s; depth 2 is ≈0.54 ms, 0.1–0.25% of b2's ≈217 ms token. The same reads from DDR5 on the CPU would take ≈3–6 ms of the bandwidth that computes in-place misses (ADR-0210). Prediction is off for a stage whose experts all sit in its fastest tier.

**Early issue (SC-8).** A next-layer predictor issued about half of the SSD misses early in the surveyed engine (`tiered-memory-inference.md` §6). [Target] SC-8: ≥ 50% of NVMe expert misses issued before their layer's router runs. Exact routing knowledge (`TokenHash`, `PreGate`, `PreAttention`) raises the share for families that have it.

**Balancer.** A fixed share is wrong in both directions: 0.2 suits a RAM-bound kernel, 0.55 an arithmetic-bound one, and the latter took decode from 70 to 78 tok/s (`tiered-memory-inference.md` §5.4). [Target] the fitted split's layer time within 5% of the best fixed split found by a sweep on the same machine, pack and trace.

**Hit rate.** [Target] SC-9 as restated in §5.3 (ARCH follow-up): the aggregate hit rate over every recorded trace, after a warm-up of 4,096 accesses per layer, is at least max(cache fraction, LRU at equal capacity) − 0.5 points. The guard tracks the best of three shadows; a trace whose routing changes faster than two guard windows can still lose up to one window's difference, which is why the target is an aggregate with a tolerance.

**Overheads.** Slot table: `n_moe · E · 4` bytes, ≈59 KB for DeepSeek-V3; slot addresses 8 bytes per slot. Statistics: ≈`n_moe · U(T) · 8` bytes per pass, ≈3.7 KB per token for DeepSeek-V3 at T = 1. Delta application: at most `max_replace` entries per adaptation, microseconds per step. [Target] zero allocations and zero locks per step in this module (ADR-0012 item 6).

**Admission bandwidth.** With `io.admit_pcie_share` = 0.25 at ≈24 GB/s pinned H2D, admissions may use ≈6 GB/s of link time, ≈1.2 GB per 200 ms step in b2 and ≈180 MB per 30 ms step on a mostly-hot model [estimate]. This bounds how fast the cache can follow a shift in routing.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Plain LRU | The generic baseline; tens of points of hit rate below decayed LFU and learned policies in the surveyed work (`tiered-memory-inference.md` §5.5). Kept as the guard's fallback (§5.3). |
| Whole-layer placement for MoE | Per-expert caching beat whole-layer pinning in the same VRAM by 1.6–2.2× (`tiered-memory-inference.md` §5.5). |
| One shared admission counter, uniform slot sizes | 2.97% against 70.4% hit rate; 2,673 against 3,556 experts in the same bytes (§5.2). |
| Replacing on the critical path | ≈3% decode lost in the surveyed engine (§5.3); RFC-0001 §8.4 rule 7 forbids it. |
| Inclusive RAM and VRAM caches | Wastes RAM equal to the VRAM cache over an `Nvme` home (§3). |
| Prediction on a CPU thread from host router copies | Spends the CPU's DRAM bandwidth, the scarce resource on discrete-GPU desktops, on work the GPU does in microseconds (ADR-0210). Kept for CPU-only nodes. |
| A fixed closed-form miss split | Its constant was fitted to three machines; kernels flip between RAM-bound and compute-bound by format (§8; ADR-0211). Kept as the prior. |
| Host-written copy plans or per-step re-capture for dynamic routing | A plan message adds a hop and a fence per layer; re-capture adds host work per token. The deterministic split rule needs neither (§6). |
| Copied misses read by the expert kernel over PCIe from mapped host memory | Avoids staging but uses SMs to drive PCIe reads and depends on access patterns per kernel. Allowed as a provider choice where it measures faster (RFC-0040); the copy engine is the default. |
| Skipping missed experts by default | Changes output (`tiered-memory-inference.md` §17); D3 makes it an approximate knob. |
| Learned or forecast-driven replacement in v1 | Promising in the surveyed papers, but needs traces Syntropy does not have yet. The `CachePolicy` enum is the extension point (§5.7). |
| Expert parallelism over Ethernet for spare capacity | Two messages per MoE layer per token; viable only on links that pass the ADR-0006 gate. |

## Testing and validation

| Claim | Test | Level |
|---|---|---|
| SC-9 hit rate | a trace-driven cache simulator in `testkit` replays recorded routing traces (from parity-corpus runs of every v1 MoE family) against every policy at a grid of capacities; decayed LFU with the guard must be ≥ max(cache fraction, LRU) on every trace and capacity; results are regression-gated | simulation, CI |
| Replacement mechanics never expose a non-resident slot | model-based test of `Replace` and `Promote` with randomised copy latencies, step timings and evictions, against a fake provider in `testkit` that checks every slot read against the residency model | property |
| Split agreement | for random doorbells, parameter blocks and eligibility tables, the dispatcher's, the CPU pool's and the GPU kernel's (reference implementation) splits and copy-slot assignments are identical; with eligibility deltas posted concurrently with launches (version skew injected), every site still evaluates at `param.delta_head` and no copy source is retired before the last step that saw it eligible completes | property |
| Exactness across sites | parity harness with the split forced to 0%, 50% and 100% copied, cache sizes from zero to the whole pool, prefetch on and off, guard modes; all within the ADR-0019 tolerance | parity (RFC-0060) |
| Balancer fit | synthetic cost generators with known coefficients and noise: the fit converges within 64 samples; bad data triggers the guard; on the CUDA rig with Q4_K and i-quant packs the fitted split is within 5% of the best fixed split from a sweep | unit, benchmark |
| Prediction | recall and precision per family and depth on recorded traces; SC-8 on the b2 rig | simulation, benchmark |
| Curves the solver reads | on recorded traces, each shadow hit-curve point is within 0.03 of a real cache run at that capacity; union predictions are within 5% of the `n_route` measured at the same T; published hit curves are concave, non-decreasing, and pass through (0, 0) and (pool, 1) | simulation, property |
| SC-5, SC-7 | decode on the b2-class rig (RTX 4090 or 5090, 128 GB DDR5, Gen4 and Gen5 NVMe) at measured `h`: within 1.10 of the overlap model | benchmark (RFC-0061) |
| Profile persistence | restart mid-workload: the hit rate in the first 1,000 steps after restart is within 5 points of the last 1,000 before it | integration |
| Single ownership | a two-GPU helper-cache configuration (M3): no expert is ever resident in both GPUs' caches | property, integration |
| Approximate knobs | each knob's KL and top-1 cost is measured and published before it can be enabled; responses carry its label | parity, integration |
| Profile sharing rules (§13) | a simulated cluster with an owner node and an invite-joined node of another operator hosting a MoE stage: with either side's `experts.share_profiles` false, the guest node writes no `XPROFILE`, sends no `ExpertWorkloadReport`, and receives no merged curves, and the solver's inputs for its layers are the uniform prior; with both true, reports flow both ways; no code path uploads a user-derived profile outside the cluster | simulation, integration |
| No allocation or lock per step | the `testkit` counting allocator and lock instrumentation on a MoE decode with replacements and prefetches in flight | unit |
| Failure paths | inject copy errors, demand-read failures and a full delta ring; the step completes on a slower correct path or fails cleanly | chaos (RFC-0060) |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | the MoE family used for parity runs with every expert resident in VRAM; doorbell layout and routing statistics; the trace recorder that feeds the simulator; S-01 (TIER-E8-T09, M0 week 1): teacher-forced routing traces of every v1 MoE family on the public parity corpus, replayed offline through a standalone simulator; per-family h(c) and union curves, policy defaults, and **the M1 NVMe expert scope decision** below |
| M1 | everything on one node: `Vram` cache over a RAM home, `HostPinned` cache over `Nvme`, `Unified` cache over `Nvme` (Apple); slot table and delta ring on CUDA and Metal; decayed LFU with the guard; profile persistence; the three-site step with copy-engine copied misses; the balancer; fused next-layer prediction (every M1 family is `RoutingKnown::AtRouter`); NVMe → RAM and RAM → VRAM prefetch; verify-window unions for MTP; prefill streaming of the routed pool; hit curves and union curves (§13) for the single-node solver; SC-5, SC-7, SC-8 and SC-9 on the single-device rigs |
| M2 | per-stage ownership across a pipeline; router replicas and `RouterHints` across stage boundaries; `ExpertWorkloadReport` to the coordinator and merging across replicas |
| M3 | the profile-sharing gate for nodes run by other operators (§13); `CrossNodeEp` ownership maps and hot replication; helper-GPU disjoint caches with the owner word; prefetch from another node's RAM on RDMA-class links; approximate knobs with published costs; `apply_deltas` and device-side waits on the HIP and Vulkan providers; exact routing knowledge (`TokenHash`, `PreGate`, `PreAttention`) with the M3 families that declare it |
| M4 | evaluation of a learned replacement policy against decayed LFU on the accumulated traces; corpus-derived seed profiles, if S-01 showed the cold start matters, and opt-in profile exchange between operators (§13, TIER-E10-T13); Tier 2/3 platforms |

**M1's NVMe expert scope depends on S-01.** At the uniform prior the S1 rig runs DeepSeek-V3 at 0.6–0.7 tok/s, and the headline rate needs a hit rate that load-balanced routers may never give, so the traces come before the path is built. The NVMe expert path (the `HostPinned` and `Unified` caches over `Nvme`, NVMe demand reads and NVMe → RAM prefetch) stays in M1 only if S-01's curves make ≥ 2 tok/s plausible on rig (b2) under RFC-0011 §15.2's model; otherwise it moves to M2, and M1 leads scenario S1 with the MoE class that fits in RAM (Qwen3-235B-class, 4.3 tok/s at its uniform prior of h = 0.84 [estimate: the cost model at its uniform prior]). Until the curves exist, S1 is presented with the RAM-fitting class, and DeepSeek-V3 at Q4 on 128 GB is a capacity demonstration, not a speed claim.

## Open questions

1. **Policy defaults.** Decay 0.7 per 4 steps, hysteresis 1.5 and 96 replacements come from one engine running a 512-expert, top-10 model. DeepSeek-class routers are trained for load balance and may show less skew (cost model §6 item 4). S-01 sweeps the three parameters on recorded traces of every v1 MoE family in M0 week 1 (moved from M1) and sets per-family defaults if they differ; the product cache re-measures in M1.
2. **What the VRAM cache is worth on load-balanced routers.** If `h_v` stays near the uniform prior for DeepSeek-class models, the solver should give that VRAM to the KV hot window or a higher-precision hot set instead. S-01's traces answer it in M0, and RFC-0011's trade uses the measured curve either way.
3. **RAM → VRAM prefetch.** The surveyed engine did not do it; Syntropy enables it under the balancer's condition (§10). An M1 A/B on the b2 rig decides whether `auto` stays the default.
4. **Device-side waits on Vulkan.** A Vulkan submission cannot wait on a host-set value in the middle of a command buffer; the fallback splits the step at MoE layers. The cost of that split on the M3 Vulkan provider is unmeasured.
5. **Small experts.** 2026 models with 1–3 MiB experts (`backends-and-portability.md` §4.3) make per-expert overheads (doorbell entries, copies, I/O requests) relatively larger. M1 measures whether experts of one layer should be grouped into multi-expert slots for those models.
6. **Routing bias as a cold-start prior.** Whether a family's router bias predicts popularity better than uniform is untested; M1 compares initial hit rates with both.
7. **Shared profiles. Answered 2026-10-20 (OQ-MOE-13):** share among the nodes one operator runs; never publish profiles derived from users' prompts; exchange with nodes run by other operators only when both opt in; the project may publish corpus-derived profiles if S-01 shows the cold start matters (§13).

## Proposed amendments to RFC-0001

1. **§5.3 `TensorClass::Router` comment and the glossary's "Next-layer expert prediction".** Replace "a host copy feeds next-layer expert prediction" with "the copy on the device that runs the router feeds next-layer prediction; a host copy is used only where the router runs on the CPU" (ADR-0210).
2. **§6.3 step 3.** Record that the dispatcher and the CPU pool both read the layer's doorbell, that the dispatcher issues copied-miss copies on the copy stream followed by a stream write-value, that the miss split is a deterministic function of the doorbell and the step's parameter block, and that the GPU derives copy-slot assignments from the same rule (§6).
3. **§5.5 closing paragraph.** "Updating the cache means writing that table" should name the mechanism: a host-mapped table-delta ring applied by the first node of each captured step (§5.4), which RFC-0022 also uses for KV block tables and RFC-0040 implements as `apply_deltas`.
4. **§5.4.** `CachePolicy` and `MissSplit` are defined in §2 of this RFC; listed here because RFC-0011 emits them.
5. **§5.10 workload reports.** RFC-0011 reads `WorkloadStats` from the coordinator's `ClusterState`, but no command carries it there, and `ReportProfile` is for per-node hardware profiles. Add `ClusterCommand::ReportWorkload { instance, node, report: ExpertWorkloadReport }` and `ClusterEvent::WorkloadUpdated { model, report }`, rate-limited as in §13 (at most once per 60 s per instance, only on a material change), so the log stays free of per-request traffic (RFC-0001 §5.10's closing rule). RFC-0012 would add its acceptance and context statistics to the same command.

**Requested of RFC-0011** (not RFC-0001; listed so the two RFCs can be reconciled):

- **`tier_hit`** reaches the coordinator in `ExpertWorkloadReport::tier_hit`, per (tier, layer) as RFC-0011 declares it, with the comparison points defined in §13. RFC-0011's C8 already uses the exclusive union `z^R ≤ a·x + s·(u + Σ_d y)`; the earlier request to adopt it is withdrawn.
- **The RAM expert cache over an `Nvme` home is split** into a `DmaPinned` copyable part and a `HostPageable` in-place part (RFC-0020 §2, §5). C9 should charge the copyable part to `HostPinned` under `pin_limit` and the rest to `HostPageable`, under the joint per-NUMA row, and bound `m^P` by the reads that the copyable part and the VRAM cache serve.
- **Prefill over `Nvme`-homed pools** costs the NVMe-resident pool bytes at `W_nvme` once per layer-major group, and the 8,192-token chunk cap does not apply to such stages (§11).
- **`UnionCurve` has two shapes**: `window` for verify windows and `batch` for `b_μ` (unchanged).
- **Hit curves exclude prefill** and model the policy without admission-bandwidth limits (§13); a tier's measured shortfall against the curve is workload drift (unchanged).
