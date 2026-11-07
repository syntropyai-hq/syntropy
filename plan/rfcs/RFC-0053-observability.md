# RFC-0053: Observability
- Status: Draft
- Area: Product surface and operations
- Depends on: RFC-0001 (§2.2, §3.5, §8.2, §8.3, §8.4, §10, §13, §14), RFC-0010 (profiles, drift), RFC-0011 (`PredictedCost`), RFC-0012 (step loop), RFC-0031 (frame layout), RFC-0050, RFC-0051, RFC-0052, RFC-0054, ADR-0009, ADR-0012, ADR-0015, ADR-0510
- Date: 2026-10-19

## Summary

Syntropy's central claim is a number: tokens/s within a few percent of what the measured hardware allows (SC-3, SC-4). Observability exists to check that claim on every machine, continuously, and to explain the gap when there is one. This RFC defines:

- **Metrics recorded on hot threads without atomics contention, locks or allocation.** They flow runner → daemon over the lossy telemetry ring (RFC-0001 §3.5) and are aggregated by the daemon. Every stage step is broken into the same **cost terms** the solver predicts (`CostTerm`, §2.2), so predicted and measured time are compared term by term. That comparison is what drives drift detection (RFC-0010) and SC-30.
- **One trace per request spanning the ring.** The root span is at ingress, with spans for routing, queueing, placement, prefill chunks, decode steps on every stage, and hops. Step spans are sampled coherently across stages: the head decides per step, and the decision rides in `ActivationFrame::trace`. A request can ask to be traced in full.
- **Structured logs.** Hot threads write binary records that the daemon formats, never strings. Sinks: stderr, files, journald, macOS unified logging and the Windows event log. Prompts and outputs are never logged unless the operator explicitly enables it.
- **`syntropy doctor`.** Stable, documented checks across install, drivers, hardware sanity (probe versus spec), OS limits, storage, network, RDMA, cluster, security and configuration. Each check gives evidence and an exact fix.
- **A debug state dump**: a local, redacted bundle of everything needed to diagnose a problem, with an itemised manifest. It is never uploaded.
- **Optional export.** A Prometheus endpoint and an OTLP exporter (metrics, traces, logs), both off by default and both run by the daemon, never by runners.
- **Zero telemetry by default** (ADR-0510). No analytics, usage pings or crash uploads. The only traffic a node originates beyond its cluster is what the user asked for (model downloads), update metadata (switchable off), and relays when invites are in use.

## Motivation and goals / non-goals

The research found three gaps that matter here.
- **Unexplained per-token overhead.** One fitted engine spends ≈5 ms per token outside its byte reads, unexplained by anything it reports (vision §1.4).
- **Silent regressions.** Decode regressions of ≈16–20% shipped between nearby releases of one engine (SC-32).
- **Failures without diagnostics.** Clusters failed in ways that only showed up as broken pipes and re-elections (`multi-device-inference.md` §10.5).

A system that places bytes by a cost model must measure the same terms the model predicts, or it cannot tell a bad placement from a bad kernel.

### Goals

- G1. Every stage step reports its time split into the solver's cost terms, so predicted versus measured is available per term, per stage, live (SC-3, SC-30).
- G2. A request can be followed across every node it touched, with consistent timing.
- G3. Observability costs at most 0.5% of step time at default settings (Performance section).
- G4. Every common misconfiguration the research found (single-channel RAM, x4 PCIe links, io_uring blocked by seccomp, WSL2 pinning caps, unenabled TB5 RDMA, NVMe throttling, chipset-attached drives) is diagnosed by name, with the fix.
- G5. Nothing about a user's prompts, outputs or usage leaves their machines unless they configure an exporter or share a file themselves.

### Non-goals

- A hosted metrics service, crash reporting service or usage analytics, including opt-in ones (ADR-0510).
- Per-expert or per-request time series in Prometheus. Those belong in traces and the flight recorder, not in labels.
- Profiling kernels. Vendor profilers do that; Syntropy exposes the hooks they need, such as NVTX-style ranges on CUDA where the provider supports them (RFC-0040).

## Design

### 1. Data flow

```text
runner process                                       daemon process
┌──────────────────────────────────────────┐        ┌──────────────────────────────────────────┐
│ hot threads (dispatchers, CPU pool, I/O) │        │ telemetry intake                         │
│   per-thread metric cells (single writer)│        │   ├─ metric store (aggregates, 1 s)      │
│   per-thread span ring, log ring         │        │   ├─ drift input → ReportProfile (0010)  │
│              │ (SPSC, lossy)             │ shm    │   ├─ flight recorder (bounded)           │
│              ▼                           │ rings  │   ├─ /v1/syntropy/metrics/live (SSE)     │
│ drain task (net runtime): every 250 ms    ├──────▶│   ├─ /metrics (Prometheus, opt-in)       │
│   encodes TelemetryRecord (postcard)     │        │   └─ OTLP exporter (opt-in)              │
└──────────────────────────────────────────┘        │ log formatter → sinks                    │
                                                    │ node-query: serve digests to other nodes │
                                                    └──────────────────────────────────────────┘
```

- **Runners never export anything.** They write into shared-memory rings (RFC-0001 §3.5: `telemetry`, `log`). Only the daemon talks to exporters, files or other nodes. This keeps async runtimes and network code off runner hot threads (ADR-0012), and it means a runner crash loses at most one drain period (250 ms) of metric deltas.
- **One producer per ring.** The `telemetry` ring's only producer is the drain task. Each dispatcher writes its `StepRecord`s into its **own** SPSC step ring in the shared-memory region (one per dispatcher, declared in `ShmLayout`), which the daemon reads directly. The proposed RFC-0001 §3.5 amendment A3 lists both rings.
- **Rings are lossy** (RFC-0001 §8.4 rule 8). A full ring drops the oldest records and increments a drop counter, which is itself exported. Losing telemetry never slows a step. *Revised 2026-10-27 (RFC-0001 §18.43):* a full ring drops the new record, not the oldest, since an SPSC producer cannot take back a record its consumer has not read; the drain sends the dropped deltas again (§2.1's note).
- **Nothing per token enters the event log** (ADR-0009). Telemetry reaches other nodes only through the node-query channel (RFC-0050 amendment A4), and only while something asks: a dashboard view, `doctor --cluster`, `debug dump --cluster`, or a trace lookup.

```rust
/// Runner → daemon telemetry record (postcard-encoded in the `telemetry` ring). Owner: RFC-0053.
pub enum TelemetryRecord {
    Counters { at: Nanos, thread: ThreadKind, deltas: SmallVec<[(MetricId, u64); 16]> },
    Histograms { at: Nanos, thread: ThreadKind, deltas: SmallVec<[(MetricId, HistDelta); 4]> },
    Gauges { at: Nanos, values: SmallVec<[(MetricId, f64); 16]> },
    Spans { records: SmallVec<[SpanRecord; 8]> },
    Notice { at: Nanos, kind: NoticeKind },   // degradations, demotions, pressure steps, cache events
}
```

*Revised 2026-10-27 (PROD-E8-T01; RFC-0001 §18.43):* the deltas name a `SeriesId`, one metric with its label values bound at registration (§2.1), not a `MetricId`, which cannot tell two label sets apart. A new last variant, `Series { entries }`, tells the daemon each ID's metric and label values before any record uses it. As built, `at` is a `MonoTime` from `TickSource::now()`, and `Spans` and `Notice` come with the tasks that define their payloads, after `Series`. §2.1's note gives the rest.

### 2. Metrics

#### 2.1 Recording on hot threads

```rust
/// Declared statically; registered at Warming. Owner: RFC-0053.
pub struct MetricDesc {
    pub id: MetricId, pub name: &'static str, pub kind: MetricKind,   // Counter | Gauge | Histogram
    pub unit: Unit, pub labels: &'static [LabelKey], pub help: &'static str, pub owner: &'static str,
}

/// One per hot thread, cache-line aligned. Single writer: the owning thread. Readers (the
/// drain task) see torn-free values because each cell is an AtomicU64 written with
/// Relaxed load + store (no read-modify-write, so no lock prefix or exclusive monitor loop).
pub struct LocalMetrics { cells: Box<[CachePadded<AtomicU64>]>, hists: Box<[LocalHistogram]> }
impl LocalMetrics {
    #[inline] pub fn add(&self, c: CounterIx, v: u64);   // load, add, store
    #[inline] pub fn record(&self, h: HistIx, v: u64);   // bucket index by leading zeros, then add
}
```

- **Histograms** are log-linear: power-of-two ranges with 8 linear sub-buckets each, giving ≤ 12.5% relative bucket width. Time histograms cover 64 ns to ≈275 s (2⁶ to 2³⁸ ns); byte histograms cover 1 KiB to 4 TiB (2¹⁰ to 2⁴² B), with smaller values in the first bucket. Either way that is 32 ranges × 8 = 256 fixed buckets, 2 KiB per histogram per thread. Percentiles are computed at read time from bucket counts.
- **Time source**. All hot-thread timing in `obs` goes through RFC-0060 §2.1's `TickSource`, which RFC-0001 §5.14 declares. It is a generic parameter, never `dyn`. In production it is monomorphised to the invariant TSC on x86-64, `CNTVCT_EL0` on arm64, or the OS monotonic counter, at zero cost. In the simulator it is the node's device timeline.
  - Durations are `ticks()` deltas converted with `ticks_to_nanos`.
  - Record timestamps (`StepRecord.at`, `SpanRecord.start`) come from `now()`, so they share one origin with protocol events.
  - GPU-side event timestamps are mapped with `device_to_ticks` through the provider's latest `DeviceClockSync`.
  - The tick-to-`Clock` offset is re-sampled after every resume from suspend.
  - `obs` reads no clock directly.
- **Labels are resolved at registration**, not on each record. A hot thread holds `CounterIx` handles already bound to their label set: (stage, device, layer…). Recording never hashes a label.
- **The drain.** A task on the runner's `net` runtime snapshots all cells every 250 ms (RFC-0001 §13), computes deltas, and writes `TelemetryRecord`s into the `telemetry` ring. Hot threads never take part. The daemon folds deltas into its store. For the series the dashboard draws (≤ 2,000), it keeps 1-second resolution for 15 minutes and 1-minute resolution for 24 hours in memory. Every other series keeps only its current value. Longer retention is the exporters' job.

*Revised 2026-10-27 (PROD-E8-T01, as built; RFC-0001 §18.43).* `syntropy-obs` holds the registry, the cells, the histograms, the `TelemetryRecord` codec and the drain. The build revises §1's record listing and lossy-ring rule, and settles how this section is built:
- **Series, not metrics, in records** (revises §1). A record names a `SeriesId`: one metric with its label values bound at registration. §1's listing names a `MetricId`, which cannot tell two label sets apart. A new last variant, `TelemetryRecord::Series`, tells the daemon each ID's metric and label values. The first drain sends every one before any record uses it. `Spans` and `Notice` come with the tasks that define their payloads (PROD-E8-T07 for spans), appended after `Series`.
- **Timestamps** are `MonoTime`, from `TickSource::now()`. `Gauges` carries no thread, as listed, so a gauge series has one writing thread: the drain sends the latest value it reads from each thread's cells.
- **Histograms** take `Nanos` and `Bytes` units only, as this section gives ranges for those two. Each also keeps the sum of recorded values, so an exporter can report one.
- **Cells.** A thread's cells are packed into 128-byte lines of their own, rather than one padded line per cell, so no two threads share a line. The single writer is asserted on every write in debug builds.
- **The drain** is a value the runner drives: its `net` runtime calls `Drain::poll` from a timer, since `obs`, at layer 0, cannot depend on `net` or `tokio`. `poll` drains once per 250 ms of the `TickSource`'s time, however often it is called.
- **A full ring** (revises §1). An SPSC producer cannot take back a record the consumer has not read, so a full ring drops the new record, not the oldest, and counts it in `syntropy_obs_ring_dropped_records_total{ring}`. The drain keeps its last read for the series a dropped record carried, so the next drain sends those deltas again. A crash still loses at most one period, as long as the ring has room.
- **One writer per ring.** `obs::ring::RingWriter` claims its ring (`telemetry`, or one step ring per dispatcher) in the region's `RingClaims`. A second claim of the same ring fails a debug assertion.
- **No `tracing` on hot threads.** The `hot-threads` step of `cargo xtask ci lint` reads the Rust tokens of the modules `xtask/src/hot_threads.rs` lists, and rejects `tracing` there.

#### 2.2 Cost terms: predicted against measured

The solver predicts token time as a sum of terms (cost model §3.2–§3.10; `PlacementPlan::predicted`, RFC-0001 §5.4). Runners measure the same terms with the same keys:

```rust
/// One key space for prediction (RFC-0011 PredictedCost) and measurement (this RFC). RFC-0011's
/// StageCost.terms uses exactly this enum (RFC-0011). Owner: RFC-0053 (keys),
/// RFC-0011 (how each is predicted), RFC-0012 (where each is measured).
#[repr(u8)]
pub enum CostTerm {
    DenseVram, DenseUnified, DenseHostCpu,  // dense groups by where they execute (GPU on Vram, iGPU on Unified, CPU on RAM)
    ExpertHitGpu,                           // expert-cache hits computed on a GPU (Vram or Unified slot)
    ExpertHitCpu,                           // experts homed in RAM, computed in place by the CPU
    ExpertMissCopied,                       // missed experts copied over PCIe (MissSplit share)
    NvmeDemandWait,                         // time a step waits on demand reads
    KvRead,                                 // all tiers; per-tier bytes are in StepRecord.bytes
    RowLookup,
    Switch,                                 // device hand-offs inside a stage (t_switch × count)
    Collective,                             // cross-node or intra-node TP/EP collectives (RFC-0032)
    HopSend,                                // serialisation and send of the outbound frame
    HopWait,                                // waiting for the inbound frame beyond the previous stage's compute
    CreditStall,                            // RFC-0001 §8.4 rule 1
    Fixed,                                  // t_fixed: launch, synchronisation, framework overhead
    Sampler, ConstraintMask, Draft,         // tail and head extras
}   // 18 variants; tiers appear only where a term really varies by executing site
// Mapping from RFC-0011 §2's current names: ExpertGpu → ExpertHitGpu; ExpertCpu → ExpertHitCpu (+ the
// in-place share of misses); ExpertNvme → NvmeDemandWait; Collective → Collective;
// Hop → HopSend + HopWait; Sample → Sampler + ConstraintMask + Draft.
pub struct TermVector([Nanos; 18]);   // dense, in enum order; zero where not applicable
```

**How terms are measured.**
- **CPU-side terms** (CPU-pool expert work, I/O waits, hop send and wait, sampler, masks, stalls) are timed every step through `TickSource::ticks`, converted with `ticks_to_nanos` (§2.1).
- **GPU-side terms** (dense reads, cache hits, KV reads) need device timestamps. These are recorded only on sampled steps (`obs.trace.step_sample`, 1 in 64 by default) with provider timing events placed at term boundaries. RFC-0012 decides whether that means a timed variant of each captured step graph or event nodes that stay in the graph. Unsampled steps report the step total, so `Fixed` there is "total minus measured terms" and is reported only on sampled steps.
- **Term attribution on unified memory**, where CPU and GPU share bandwidth, follows the solver's split: a term is whatever the plan predicted it as.
- **Parallel sites use RFC-0011 §2's attribution**. RFC-0011 defines `StageCost.terms` as critical-path attributions that sum to the step time. Where sites run in parallel (a MoE block's GPU, CPU and NVMe work, C8; KV's GPU and CPU-split parts, C7), the block's time goes to the binding site's term, and the other sites' times go to `overlapped`. Measurement does exactly the same:
  - **Per block.** The step program marks each parallel block from the committed plan. Each site in it is timed inside that block, CPU sites through `TickSource`, and GPU sites with the sampled-step events mapped by `device_to_ticks`.
  - **Attribution.** The block's time, the maximum over its sites, goes to the binding site's term. Every other site's time goes to that term's entry in `overlapped`. Summed over a step, the critical-path terms add up to `total`.
  - **Comparison.** Predicted `terms` are compared with measured `terms`, and predicted `overlapped` with measured `overlapped`, never across the two sets.
  - **A different binding site is real drift.** If the plan predicted the GPU binding and the CPU binds in practice, the plan is wrong about that block, and the ratio says so.
  - **Unsampled steps.** A block with a GPU site can be attributed only on sampled steps. Ratios for the terms of such blocks therefore use sampled steps only, and terms of CPU-only blocks use every step.

The daemon computes per (instance, stage, term) the measured mean over the last 60 s, the predicted value, and their ratio. It publishes them as metrics (§2.3), on the live stream (RFC-0050 §13.5), and to RFC-0010's drift detector. A ratio beyond `replan.drift_threshold` for `replan.drift_window` is drift (RFC-0001 §7.4).

#### 2.3 Catalogue

Names follow RFC-0001 §13: `syntropy_<crate>_<thing>_<unit>`, snake case, base units (seconds, bytes), `_total` for counters. The crate is the one that records the metric. Owners may add metrics; renaming one is a breaking change under RFC-0062's compatibility policy.

| Area | Metric | Type | Labels |
|---|---|---|---|
| stage | `syntropy_exec_step_seconds` | histogram | instance, stage, kind (`decode`, `prefill`, `verify`, `replay`) |
| stage | `syntropy_exec_term_seconds` | histogram | instance, stage, term |
| stage | `syntropy_exec_term_predicted_seconds` | gauge | instance, stage, term |
| stage | `syntropy_exec_term_ratio` | gauge (measured ÷ predicted, 60 s) | instance, stage, term |
| stage | `syntropy_exec_tokens_total` | counter | instance, stage, kind |
| stage | `syntropy_exec_t_fixed_seconds` | gauge | node, device |
| tiers | `syntropy_tier_read_bytes_total` | counter | node, tier, device, class (`dense`, `expert`, `kv`, `lookup`) |
| tiers | `syntropy_tier_bytes` | gauge | node, tier, purpose (RFC-0001 §9.1, §9.3 purposes), state (`used`, `budget`, `usable`) |
| tiers | `syntropy_tier_pinned_bytes` | gauge | node, state (`locked`, `target`) |
| I/O | `syntropy_io_read_bytes_total` | counter | node, tier, priority (`demand`, `prefetch`, `load`) |
| I/O | `syntropy_io_read_seconds` | histogram | node, tier, priority |
| I/O | `syntropy_io_inflight` | gauge | node, tier |
| I/O | `syntropy_io_nvme_temperature_celsius`, `syntropy_io_nvme_throttle_events_total` | gauge, counter | node, tier |
| expert cache | `syntropy_tier_expert_lookups_total` | counter | node, device, layer, result (`hit`, `miss_in_place`, `miss_copied`, `miss_nvme`) |
| expert cache | `syntropy_tier_expert_hit_ratio` | gauge (decayed, RFC-0023 window) | node, device, layer |
| expert cache | `syntropy_tier_expert_admissions_total`, `syntropy_tier_expert_evictions_total` | counter | node, device |
| prefetch | `syntropy_tier_prefetch_total` | counter | node, path (`nvme_ram`, `ram_vram`, `peer`), outcome (`useful`, `late`, `wasted`, `dropped`) |
| prefetch | `syntropy_tier_router_prediction_recall` | gauge | node, layer |
| KV | `syntropy_tier_kv_bytes` | gauge | instance, stage, tier, state (`used`, `budget`) |
| KV | `syntropy_tier_kv_pages_spilled_total` | counter | instance, stage, to_tier |
| KV | `syntropy_tier_prefix_cache_tokens_total` | counter | instance, result (`hit`, `miss`) |
| KV | `syntropy_tier_sessions` / `syntropy_tier_session_restore_seconds` | gauge / histogram | node, state (`live`, `parked`) |
| links | `syntropy_net_link_alpha_seconds` | gauge | link, transport, quantile (`0.5`, `0.99`) |
| links | `syntropy_net_link_goodput_bytes_per_second` | gauge | link, transport |
| links | `syntropy_net_link_bytes_total` | counter | link, plane (`data`, `bulk`, `control`) |
| links | `syntropy_net_link_retransmits_total`, `syntropy_net_link_credit_stall_seconds_total`, `syntropy_net_transport_demotions_total` | counter | link, transport |
| requests | `syntropy_api_request_ttft_seconds`, `syntropy_api_request_tpot_seconds`, `syntropy_api_request_queue_seconds` | histogram | model, shape, fidelity |
| requests | `syntropy_api_requests_total` | counter | model, shape, fidelity, outcome (`ok` or an error class) |
| requests | `syntropy_api_tokens_generated_total` | counter | model, fidelity |
| requests | `syntropy_api_sequences_active` | gauge | instance, priority |
| requests | `syntropy_api_approx_applied_total` | counter | model, knob |
| requests | `syntropy_api_admission_rejected_total` | counter | instance, reason |
| speculation | `syntropy_sched_spec_windows_total`, `syntropy_sched_spec_accepted_tokens_total` | counter | instance, source |
| speculation | `syntropy_sched_spec_window_tokens` | histogram | instance |
| runners | `syntropy_daemon_runner_state` | gauge (1 for the current state) | instance, node, state |
| runners | `syntropy_daemon_runner_restarts_total` | counter | instance, node, class |
| runners | `syntropy_runner_phase_seconds` | histogram | phase (`fetching` … `warming`) |
| runners | `syntropy_runner_spin_seconds_total` | counter | instance, thread_kind |
| memory | `syntropy_daemon_memory_pressure_level` | gauge | node |
| memory | `syntropy_tier_memory_reclaimed_bytes_total` | counter | node, step (ladder step 1–4) |
| memory | `syntropy_tier_memory_yield_cost_seconds` | gauge (predicted token-time increase) | node, instance |
| cluster | `syntropy_cluster_log_index`, `syntropy_cluster_members` | gauge | state |
| cluster | `syntropy_cluster_events_total`, `syntropy_cluster_elections_total`, `syntropy_cluster_gossip_suspicions_total` | counter | type |
| store | `syntropy_store_bytes` | gauge | node, kind (`chunks`, `packs`, `sessions`) |
| store | `syntropy_store_fetch_bytes_total` | counter | node, source (`local`, `cluster`, `origin`) |
| store | `syntropy_store_chunk_integrity_failures_total` | counter | node, source |
| process | `syntropy_obs_process_resident_bytes`, `syntropy_obs_process_cpu_seconds_total` | gauge, counter | process (`daemon`, `runner`) |
| obs | `syntropy_obs_telemetry_dropped_total`, `syntropy_obs_log_dropped_total` | counter | ring |
| drift | `syntropy_probe_drift_reports_total` | counter | term |

`syntropy_cluster_events_total` per type is how ADR-0009's revisit trigger ("more than a few hundred events/s in a steady cluster") is detected.

#### 2.4 Labels and cardinality

- **Identity labels** are RFC-0001 §13's bounded set: `node`, `instance`, `stage`, `device`, `tier` (an NVMe volume is its `Nvme` tier), `link`, `backend`, `op_class`.
- **Enumerated labels.** The catalogue also uses labels whose values are fixed at compile time: `kind`, `term`, `class`, `purpose`, `state`, `priority`, `result`, `outcome`, `path`, `plane`, `transport`, `quantile`, `phase`, `step`, `reason`, `source`, `knob`, `fidelity`, `shape`, `thread_kind`, `ring`, `to_tier`.
- **Bounded data labels.** Two labels take data values:
  - `layer`, on expert-cache metrics only, bounded by the model's layer count;
  - `model`, on request metrics only, bounded by the registered models.
- **The enumerated and data labels need RFC-0001 §13's list extended** (amendment A5). Until it is, they are what this RFC proposes, and no other label is allowed.
- **Forbidden labels**: request, sequence, session, trace or expert IDs; tensor names; prompt-derived values; user agents; API key IDs.
- **Budget**: at most 20,000 series per node for the largest v1 model on the largest supported node [target]. Per-layer expert metrics dominate. A 61-MoE-layer model on a node with one GPU and two NUMA CPU devices gives 61 × 3 × 4 lookup results ≈ 730 lookup series [estimate], well inside the budget. A CI test computes the series count for every v1 family's largest model and fails above the budget.

#### 2.5 Per-step records and the telemetry drain

RFC-0001 §13 requires one fixed-size record per stage step, always on. This RFC owns its encoding. RFC-0061 needs it per term to measure the RFC-0001 §14 decomposition instead of inferring it (RFC-0061 amendment B1).

```rust
/// One per stage per step, written by the stage's primary dispatcher into its own SPSC step ring.
/// Fields are ordered by alignment, so `#[repr(C)]` has no internal padding: exactly 256 bytes with
/// RFC-0001's sizes (InstanceId 16 B, PlanEpoch u64, StageIx u16), checked by a compile-time
/// assert on `size_of::<StepRecord> == 256`. Owner: RFC-0053.
#[repr(C)]
pub struct StepRecord {
    pub instance: InstanceId,         //   0
    pub at: Nanos,                    //  16  step start, monotonic
    pub epoch: PlanEpoch,             //  24
    pub step: u64,                    //  32
    pub total: Nanos,                 //  40  step wall time
    pub kernel: Nanos,                //  48  GPU or CPU kernel time (RFC-0001 §13)
    pub inbound_wait: Nanos,          //  56  waiting for the inbound frame (HopWait)
    pub hop_send: Nanos,              //  64  outbound send to completion ticket (HopSend)
    pub credit_stall: Nanos,          //  72  outbound credit stall on this stage's hop
    pub miss_in_place: Nanos,         //  80  t_miss, CPU execute-in-place share
    pub miss_nvme_wait: Nanos,        //  88  t_miss, NVMe demand wait share
    pub switch_total: Nanos,          //  96  Σ t_switch inside the step
    pub sample: Nanos,                // 104  tail only: sampler, masks and drafting
    pub bytes: [u64; 5],              // 112  bytes read per TierKind (Vram, Unified, HostPinned, HostPageable, Nvme)
    pub tokens: u32,                  // 152  positions processed (prefill chunk size, 1, or the window)
    pub expert_hits: u32,             // 156  summed over the stage's MoE layers
    pub expert_misses: u32,           // 160
    pub term_us: [u32; 16],           // 164  µs per slot; u32 µs saturates at ≈71 min
    pub stage: StageIx,               // 228
    pub accepted: u16,                // 230  tokens accepted in a verify window; 0 otherwise
    pub kind: StepKind,               // 232  repr(u8): PrefillChunk | Decode | Verify | Replay | Canary
    pub sampled: bool,                // 233  GPU-side terms measured on this step (§2.2)
    pub n_terms: u8,                  // 234  slots [0, n_terms): critical-path terms, summing to `total`
    pub n_overlapped: u8,             // 235  next n_overlapped slots: non-binding parallel sites (§2.2)
    pub term_ids: [CostTerm; 16],     // 236  repr(u8); n_terms + n_overlapped ≤ 16
    pub _reserved: [u8; 4],           // 252  zero; brings the record to 256
}
```

- **Where records go.** The daemon reads each dispatcher's step ring and keeps records in a **step recorder** separate from the flight recorder, bounded by `obs.step_recorder` (default 32 MiB per node). At 256 B per record that holds 131,072 records: about 7 minutes at 300 steps/s, or 12 minutes at 175 steps/s, for one local stage [estimate]. It feeds RFC-0010's online refinement and drift detection and computes the §2.2 ratios. The drain serves whatever window the recorder still holds.
- **Size and rate.** 256 bytes per stage step. At 300 steps/s that is 300 × 256 B ≈ 75 KiB/s per stage [estimate], inside the ring budget (Performance section).
- **The drain.** `GET /v1/syntropy/debug/telemetry?instance=…&since=…&until=…&follow=…` (RFC-0050 §13.6) streams these records for a time window from every stage of the instance. Remote stages' records are fetched over the node-query channel. It is enabled only by `debug.telemetry_api`, loopback only (RFC-0054 §7.1). It runs on the daemon's `ctl` runtime and reads what the ring already carries, so it adds no work on hot threads. RFC-0061's A/A rule bounds its cost: decode rate with the drain active stays within 2% of the rate with it off, checked at M0 and after every change to this record.
- **Schema versioning.** The record carries no version field. The drain's response header names the schema version, and a change to `StepRecord` bumps it under RFC-0062 §10.


### 3. Tracing

#### 3.1 Span model

One trace per request. Its trace ID is minted at ingress, or taken from an incoming `traceparent`. Span names follow RFC-0001 §13: `syntropy.<crate>.<operation>`.

```text
syntropy.api.request             ingress     model, shape, fidelity, prompt/output tokens, finish
├─ syntropy.api.route            ingress     candidates, chosen instance, score terms
├─ syntropy.api.place            ingress     automatic placement wait (RFC-0050 §4.3), when it happened
├─ syntropy.api.queue            ingress     admission wait
├─ syntropy.sched.prefill        head        prompt tokens, cached tokens
│  └─ syntropy.exec.step  ×(chunks × stages)      stage, kind=prefill, chunk, term durations as attributes
│     └─ syntropy.net.hop                    from, to, transport, bytes
├─ syntropy.sched.decode         head        tokens, windows, acceptance
│  └─ syntropy.exec.step  ×(sampled steps × stages) stage, kind=decode|verify, window, term durations
│     └─ syntropy.net.hop
└─ syntropy.cluster.recovery     coordinator/head  kind (migration, replay, re-prefill), gap
```

- **Attributes on the root span.** The OpenTelemetry GenAI semantic conventions where they fit: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`. Syntropy attributes go under `syntropy.*` (`syntropy.fidelity`, `syntropy.instance`, `syntropy.plan_epoch`, `syntropy.approx_applied`).
- **Term durations are attributes, not child spans.** A step carries up to 16 terms. As spans they would multiply volume by 16 for no extra insight.
- **No content.** Spans never carry prompt or output text or token IDs.

#### 3.2 Context propagation

- **HTTP.** An incoming W3C `traceparent` (and `tracestate`) is continued. Its sampled flag forces full tracing only if `obs.trace.honor_remote_sampled` is true (default false), so an untrusted client cannot force a node into full tracing. Responses return `traceparent` for the request's root span.
- **Ingress → head.** `SequenceOpen` carries the trace context of `syntropy.sched.prefill` and `syntropy.sched.decode` (RFC-0001 §13).
- **Between stages.** `ActivationFrame::trace: Option<TraceContext>` (RFC-0001 §5.8) is set only on sampled steps:

```rust
/// 25 bytes on the wire when present; RFC-0031 owns the layout. Owner: RFC-0053.
pub struct TraceContext {
    pub trace_id: [u8; 16],
    pub parent_span: [u8; 8],   // the previous stage's step span: every stage's step span is a sibling under
                                // the sequence span, linked to its predecessor, so the ring reads as a chain
    pub flags: u8,              // bit 0: sampled; bit 1: full trace (every step)
}
```

`TokenFrame` carries the same context on sampled steps (RFC-0001 §13), so the return path from the tail to the head and to ingress appears as a `syntropy.net.hop` span.

*Note 2026-10-22 (DIST-E1-T14): one `TraceContext`, 26 bytes.* The block above has 25 bytes because it leaves out W3C traceparent's version byte. RFC-0031 §6.3's `TRACE` item, which owns the layout, carries all 26: version, trace-id (16 bytes), parent span id (8) and flags. The one `TraceContext` is defined in `core` with that shape (RFC-0001 §5.8, note of the same date), and `net` encodes it. Its fields are `version`, `trace_id`, `span_id` and `flags`: `span_id` is the parent span above, the sender's span. Its `flags` keep this section's bits: bit 0 sampled, as W3C's trace-flags, and bit 1 full trace.

*Note 2026-10-22 (#1766): an incoming flags byte keeps only bit 0.* W3C Trace Context Level 2 gives trace-flags bit 1 to `random-trace-id`, which clients following W3C Level 2 may set, while `TraceContext::flags` uses bit 1 for full trace. A client's flags byte copied as it is would switch on full tracing, which costs every stage work and records more per-request detail than the client asked for.
- **In.** Ingress keeps only bit 0 (`sampled`) of an incoming `traceparent`'s flags. The HTTP rule above then decides whether that bit forces full tracing.
- **Where it stays.** The kept bit 0 stays at ingress, with the incoming header in the request's `ClientMeta` (RFC-0050 §2.1), as the HTTP rule's input. The `TraceContext` ingress puts in `SequenceOpen` takes its flags from Syntropy's policy alone, so a client's bit 0 never makes the head sample every step. The head still decides per step (§3.3).
- **Syntropy's own bits**, bit 1 among them, are set only by Syntropy's policy: `syntropy.trace = true`, `Syntropy-Trace: 1`, or a sampled flag honoured under `obs.trace.honor_remote_sampled`.
- **Out.** Ingress sets a returned `traceparent`'s flags itself and never copies `TraceContext::flags`:
  - bit 0 is 1, Syntropy's recording decision, since the root spans are always recorded (§3.3);
  - bit 1 (`random-trace-id`) equals the incoming `traceparent`'s bit 1, since the trace-id was continued, as W3C §3.2.2.5.2 requires and §3.4's mutation list leaves unchanged;
  - Syntropy's full-trace bit never leaves the node.
- **Which responses.** W3C §6.3 says to return `traceparent` only to a caller that took part in the trace. So a response carries one only when its request carried a `traceparent`. A trace ingress minted is found by response ID instead (§3.6).
- **Every W3C trace context the node emits** follows the same rule, exported OTLP spans (§7.2) included. OTLP's `Span.flags` holds the W3C trace-flags in bits 0–7, so an exported span carries bit 1 as its trace's `random-trace-id`, the incoming bit for a continued trace-id and 1 for one ingress minted from random bytes, and never the full-trace bit.
- **Not chosen:** moving full trace to another bit. W3C reserves the higher bits for later levels, so any of them could collide again.

#### 3.3 Sampling

| Spans | Recorded | Kept in |
|---|---|---|
| `api.request`, `api.route`, `api.place`, `api.queue`, `sched.prefill`, `sched.decode`, `cluster.recovery` | always (a handful per request) | flight recorder; exported when OTLP is on |
| `exec.step`, `net.hop` | when the head samples the step: per sequence, every step for the first 4 decode steps (RFC-0001 §13), then each step with probability `obs.trace.step_sample` (1/64), using a hash of (trace ID, step) so the decision is deterministic and reproducible | as above |
| every `exec.step` and `net.hop` | when the request asked for a full trace (`syntropy.trace = true`, `Syntropy-Trace: 1`, or an honoured remote sampled flag) | as above |

**The head decides once per step, and every stage obeys the frame's flag**, so a sampled step has spans on every stage. Statistically sampled steps from independent per-stage decisions would almost never line up across a 3-stage ring. The first decode steps are always sampled, because they include cold-cache effects that users ask about.

#### 3.4 Recording on hot threads

Hot threads write fixed 64-byte `SpanRecord { trace_id, span_id, parent_id, name: SpanNameId, start: Nanos (monotonic), dur: u32 ns, attrs: [u32; 6] }` into a per-thread SPSC ring. Recording allocates nothing and touches no strings. Housekeeping batches the records into `TelemetryRecord::Spans`. The daemon turns them into full spans: it resolves name IDs, converts monotonic time to wall time with the node's offset, and attaches the per-term durations from the matching `StepRecord`.

*Note 2026-10-28 (PROD-E8-T07, as built): `syntropy_obs::trace`.* Each point below is **for the maintainers to confirm**.
- **The record is 64 bytes with four attributes.** The fields listed above sum to 70 bytes, not 64. `SpanRecord` is `repr(C)`, held to 64 bytes by a build-time assertion, and encoded little-endian at the same offsets:
  - `trace_id` 0–15, `span_id` 16–23, `parent_id` 24–31;
  - `start` (monotonic nanoseconds) 32–39, `dur` (u32 ns) 40–43;
  - `name` (`SpanNameId`, u16) 44–45, `attr_count` (u16, at most 4) 46–47;
  - four u32 attributes at 48–63.

  The daemon takes a step's per-term durations from the matching `StepRecord`, so a span's own attributes need at most four. `SpanName::attrs` names them per span.
- **The span ring is in-process.** Each hot thread owns its ring, through a `SpanWriter` that takes no lock and drops and counts when the ring is full. Housekeeping batches the records into a new last variant, `TelemetryRecord::Spans`, each batch as many records as fit a `telemetry` slot. So no shared-memory ring is added. A 3.x daemon cannot decode the new variant, so `ipc_protocol` goes to 4.0 (RFC-0001 §18.49).
- **The sampling hash (§3.3)** is FNV-1a-64 as the FNV specification gives it (offset basis `0xcbf29ce484222325`, prime `0x100000001b3`, checked against its published vectors) over the 16 trace-ID bytes, then the step index as a u64 little-endian. A step is sampled when:
  - the hash is below ⌊`obs.trace.step_sample` · 2⁶⁴⌋, which is 2⁵⁸ at the default 1/64;
  - or it is one of the first four decode steps;
  - or the request is a full trace.

  The step index is the sequence's step counter, counting prefill chunks and decode steps alike. A prefill chunk is sampled by the hash or by a full trace only. The arithmetic is integer only, so the decision is the same on every host and in every replay.
- **The keys**, all node scope:
  - `obs.trace.step_sample`, a fraction, default 1/64, hot reload;
  - `obs.trace.honor_remote_sampled`, a bool, default false, hot reload;
  - `obs.flight_recorder`, a byte size, default 64 MiB, node reload.

  `obs` declares them, and the binary registers them.
- **The flight recorder** counts each entry at its record's size (64 bytes for a span, 48 for a log record) and evicts the oldest first. It looks traces up by trace ID. So `obs.flight_recorder` bounds the records' bytes, not memory. Each entry is a 72-byte value in memory, 1.125× a span's count and 1.5× a log record's, so the Performance model's 64 MiB row is about 72–96 MiB of heap by the mix of spans and logs [estimate, from the type sizes], before the deque's spare capacity.
- **Span durations** are u32 nanoseconds, at most about 4.29 s; `SpanRecord::dur_ns` saturates a longer span there.
- **The API side is wired.** `api`'s ingress runs `IngressTrace` on `ClientMeta::traceparent`, puts its context in `SequenceOpen::trace`, and returns `Generation::response_traceparent`. The head's sampler call, the stages' span records and the daemon's flight recorder are PROD-E8-T14's.
- **Ingress** follows §3.2. An unparseable `traceparent` counts as none: a new trace is started, and none is returned. New IDs are never all zero.

#### 3.5 Clocks and hop timing

- **Each node's timestamps** come from `TickSource::now()` on its monotonic `Clock` timeline (§2.1). The daemon maintains its offset to wall time.
- **Cross-node offsets** are estimated per link from the link probes' round trips (RFC-0010): offset = (t₁ + t₄ − t₂ − t₃) / 2, the standard four-timestamp method, refreshed by the periodic online re-probes. The error is bounded by half the path asymmetry, so it is accurate to roughly α/2 on symmetric LAN links and much less accurate on asymmetric Wi-Fi.
- **Hop spans are therefore measured on one clock where possible.** The sender records send start to send-completion ticket. The receiver records its wait for the frame as `HopWait`. The ring as a whole, from the head's send to the head's receipt of the token, is measured on the head's clock alone. That gives a consistency check that needs no clock alignment: Σ stage steps + Σ hops ≈ ring time. Gaps in it are reported as `unattributed` on the decode span.

#### 3.6 Assembly and viewing

- **Each node keeps its own spans** in its flight recorder, and exports them itself when OTLP is on. A collector joins spans by trace ID.
- **Without a collector**, the dashboard's trace view asks the ingress node. Ingress fetches the spans for that trace ID from every node in the request's ring over the node-query channel and renders a waterfall: ring steps as rows, terms as coloured segments, predicted terms as outlines.
- **Lookup by request ID.** Traces of the last `obs.flight_recorder` worth of requests can be found by response ID (`chatcmpl-…`, `msg_…`, `resp_…`), which ingress maps to trace IDs.

### 4. Logs

#### 4.1 Producing records

- **Daemon and runner non-hot code** (tokio tasks, the runner's `net` domain) use the `tracing` crate's events with structured fields.
- **Hot threads** never format. They push `LogRecord { callsite: CallsiteId, level, at: Nanos, args: [u64; 4] }` into the runner's `log` ring (RFC-0001 §3.5). The callsite table (format string, field names, file and line) is static per build. The daemon and runner may be different builds after an update re-adopts runners (RFC-0052 §6.5). So each runner sends a digest of its callsite, metric and span-name tables in `Hello`, and on a mismatch sends the tables themselves; the daemon decodes that runner's records with them. The ring record schemas (`TelemetryRecord`, `StepRecord`, `LogRecord`, `SpanRecord`) are part of `ipc_protocol`, so a schema change disables re-adoption for that update. The daemon formats on receipt.
- **Every record carries**: timestamp (UTC, microseconds), level, target (crate and module), node, and where applicable instance, stage, sequence (as an opaque short ID), trace ID, error code, plus event fields.
- **Repeat suppression.** Identical (callsite, level, key fields) records within 10 s collapse into one with `repeated=N`, so a flapping link cannot flood the log. *Note 2026-10-27 (PROD-E8-T04):* the first record of a run is written at once, and the repeats in its window become one summary record with `repeated = N`, the number it stands for, when the window closes. 1,000 identical warnings in 10 s therefore write the first and one record with `repeated = 999`. Holding the first back for 10 s would delay every log line.

*Note (2026-10-27, open for the maintainers):* the M0 shared-memory layout has one `log` ring per runner, and an SPSC ring has exactly one writer. Several hot threads can't share that writer without a lock, which ADR-0012 forbids on hot threads. So today only one hot thread per runner writes `LogRecord`s. The recommended default is one `log` ring per hot thread, as amendment A3 gives each dispatcher its own step ring, with the daemon draining them all. That changes `ShmLayout` and so `ipc_protocol`. Until it is decided, other hot threads report through metrics, which are per-thread already.

#### 4.2 Sinks

| Situation | Sinks |
|---|---|
| foreground (`syntropy serve` in a terminal) | stderr, human format with colour (`logging.format = auto`) |
| systemd service (detected through `JOURNAL_STREAM`) | journald with structured fields; no file unless `logging.file` names one |
| launchd agent or daemon | unified logging (subsystem: the app bundle identifier, RFC-0052 §9; one category per crate) plus the rotating file |
| Windows service or logon task | rotating file; warnings and errors also to the Windows event log (source `Syntropy`) |
| any, with `logging.format = "json"` | JSON Lines to the chosen sink, one object per record |

The rotating file is `<state>/logs/daemon.log`, 16 MiB × 5 files. Runner logs go to the same sinks, tagged with instance and pid; there are no separate runner files.

#### 4.3 Levels and content

- `info` (default) records lifecycle, plans, membership, degradations, reload reports, pulls and errors. It records nothing per token or per request apart from errors.
- `debug` adds per-request summaries: model, shape, token counts, timings, fidelity, instance. It never includes content.
- `trace` adds per-step detail for development.
- **Content.** Prompts, outputs and token IDs are logged only with `logging.include_content = true`, only at `debug`, and the daemon then prints a warning at start and on every reload. Secrets are typed `Secret<T>` (RFC-0054 §10.1) and print as `[redacted]` at every level.

`syntropy logs` reads through the API: the daemon serves the recent tail from its flight recorder and follows live records.

*Note 2026-10-27 (PROD-E8-T04, as built).* The build settles what §4.1–§4.3 leave open:
- **Where it lives.** `LogRecord`, `Callsite` and the `log` ring's `LogWriter` are in `syntropy_obs::log`, a hot-thread module under the `hot-threads` lint. The record is a fixed 48-byte little-endian encoding: callsite `u32`, level `u8`, three zero bytes, `at: MonoTime`, then the four `u64` arguments. The daemon's pipeline, formats and sinks are in `syntropy_daemon::logging`. The `[logging]` section is `syntropy_config::logging`, beside the `debug.*` keys, with RFC-0054 §5.2's five keys.
- **Content** travels in fields marked as content: a `tracing` field named `content.<name>`, or a callsite argument whose bit is set in `Callsite::content`. Messages never carry content, so a runner callsite's content argument shows as `[content]` in its message. The pipeline drops content fields unless `logging.include_content` is on and the record is at `debug` or `trace`. While it is on, the logger warns at start and on every reload, and so does the resolved configuration.
- **Repeat suppression** (§4.1). The first record of a run is written at once and opens a 10 s window, timed on the record's own `MonoTime` and the logger's injected `Clock`. Records in the window that share the callsite, the level and the key fields are counted, not written. When the window closes, one record stands for them, carrying `repeated = N`, the number it replaces. So 1,000 identical warnings in 10 s are written as the first one and one record with `repeated = 999`. At most 4,096 runs are tracked; past that, the oldest run's summary is written early.
- **Timestamps** are UTC with microseconds. They come from one sample of the `Clock`'s wall time and monotonic time, then monotonic differences, so every record's time comes from the injected `Clock`.
- **Sinks** follow §4.2's table: `JOURNAL_STREAM` means systemd, a `XPC_SERVICE_NAME` other than `0` means launchd, and a Windows service is told by its own start path.
  - journald gets its native protocol over `/run/systemd/journal/socket`, with every field as `SYNTROPY_<NAME>`.
  - Unified logging gets one `os_log_t` per crate, written as `%{public}s` through the call the `os_log` macros make.
  - The Windows event log gets warnings and errors from source `Syntropy`, without a message file, which is packaging's.
  - Both OS facilities are linked by `syntropy-io` alone (`io::os::log_sink`), from `libSystem` and `advapi32.dll`, which rule 7 already allows.
  - A user-named `logging.file` is written in every situation; `auto` writes `<state>/logs/daemon.log` only where the table keeps a file.

### 5. `syntropy doctor`

#### 5.1 Checks and findings

```rust
pub struct DoctorCheck {
    pub id: &'static str,                 // stable, e.g. "OS-003"; documented, never reused
    pub area: Area,
    pub needs: Needs,                     // Offline | Daemon | Probe | Cluster
    pub run: fn(&DoctorCtx) -> Finding,
}
pub struct Finding {
    pub status: Status,                   // Pass | Info | Warn | Fail | Skip { why }
    pub summary: String,
    pub evidence: Vec<(String, Figure<String>)>,   // measured values, versions, limits
    pub fix: Option<Fix>,
}
pub struct Fix {
    pub description: String,
    pub command: Option<String>,          // exact command to run, for this OS
    pub doc: Option<String>,              // docs page for the check ID
    pub automatic: Option<AutoFix>,       // only for safe, reversible, host-ledgered changes
}
```

- **Modes.** `doctor --offline` runs the checks that need no daemon (install, drivers, OS limits, storage, config). With a daemon it also runs the checks that read the probe, runners and cluster state. `--cluster` runs on every member over the node-query channel and groups findings by node.
- **Exit code.** Non-zero when any check fails.
- **`--fix`** applies only fixes marked `automatic`, asks for each one on a terminal (`--yes` otherwise), and records every host change in the RFC-0052 §11 ledger, so `uninstall` reverts it. It never changes firmware or BIOS settings, sysctls the user did not name, or anything on another node. It never changes the Windows GPU watchdog (`TdrDelay`) or hardware-accelerated GPU scheduling: both are machine-wide, need a reboot and affect every application on the desktop, so DRV-006 and DRV-007 only report them, and kernels are designed to fit under the default watchdogs instead (RFC-0040). The only OS limits Syntropy changes are the macOS GPU wired limit and, from M3, the Linux TTM `pages_limit`, both through `syntropy tune` with consent and a ledger entry (RFC-0051 §2.1).

#### 5.2 Check list (v1)

| ID | Check | Typical evidence and fix |
|---|---|---|
| INS-001 | core binary and every backend module come from one release, and their signatures verify | module and core build IDs; reinstall the module |
| INS-002 | service installed, enabled and pointing at the current version | unit or agent path; `syntropy service install` |
| INS-003 | host-change ledger matches the host: no missing or orphaned entries | ledger entries; `syntropy doctor --fix` re-applies or forgets |
| INS-004 | the API port could not be bound | the bind error; on Windows, whether the port sits in an excluded range that Hyper-V or WSL reserved (`netsh int ipv4 show excludedportrange protocol=tcp`), and the `api.bind` setting that moves it (OQ-PRD-13) |
| DRV-001 | NVIDIA driver ≥ 580 for the CUDA 13 module; GPU sm_75 or newer, else the legacy module (Tier 3) | driver version, compute capability |
| DRV-002 | AMD GPU on the ROCm list with the kernel driver present; otherwise Vulkan is in use. The check reads the running daemon's live groups (`/proc/<pid>/status`) and compares them with the user's configured groups. A `render` or `video` membership that is configured but not live means a re-login or reboot is pending, and the finding says so | gfx target, runtime state, live and configured groups |
| DRV-003 | Vulkan 1.3+ driver; cooperative-matrix support; an AMD or Intel GPU with no Vulkan loader or driver | driver and API versions, extension list; the distribution's loader and driver package names |
| DRV-004 | macOS version for Metal 4 paths (26+) and for RDMA over TB5 (26.2+) | OS build |
| DRV-005 | WSL2 paravirtual GPU: pinned allocations capped near 1 GiB (RFC-0001 §9.2) | detected; explains reduced pinning |
| DRV-006 | Windows GPU watchdog: the effective `TdrDelay` and `TdrLevel` for each display GPU. Report only, never changed | registry values or their defaults; whether a planned long prefill was split to fit (RFC-0040) |
| DRV-007 | Windows hardware-accelerated GPU scheduling on or off. Report only, never changed | the setting as the probe read it; the measured `t_fixed` both ways where the lab has measured it |
| DRV-008 | Linux: a `libcuda.so` on the default library search path that is not the file `libcuda.so.1` resolves to, such as a native-Linux driver copied into a WSL2 system; and on WSL2, more than one `libcuda.so.1` in the loader's cache. Report only. The CUDA provider opens only `libcuda.so.1` (RFC-0040 §7.1, note 2026-10-31), so with a duplicate it depends on the cache order, and other CUDA programs may load the copy and find no GPU | both paths after resolving links, whether they are the same file, and each `libcuda.so.1` the cache lists, in order |
| HW-001 | CPU DRAM bandwidth well below the DIMM configuration's peak (single channel, or a single-CCD limit) | probe W_eff against peak; the check names which |
| HW-002 | GPU PCIe link below its maximum generation or width | current and maximum link from NVML or sysfs |
| HW-003 | any device's probed W_eff below 0.5 × published peak (vision principle 2) | figures; links to HW-001 or HW-002 when they explain it |
| HW-004 | NVMe sustained rate far below burst (thermal throttling) or drive on chipset lanes | sustained and burst rates, temperature trace |
| HW-005 | GPU clocks or power state throttled during the probe | clocks, power state |
| OS-001 | Linux pinning ceiling (`RLIMIT_MEMLOCK`, `CAP_IPC_LOCK`, cgroup) below the planned pinned budget. The check reads the running daemon's live limit, not the configured one, and says when they differ because the user manager predates the change (re-login or reboot needed) | live and configured limits; the exact `limits.d` or unit drop-in |
| OS-002 | container memory limit (cgroup `memory.max`) below what plans need | limit |
| OS-003 | io_uring unavailable (`io_uring_disabled` sysctl, seccomp, Android, ChromeOS); the `pread` pool is active | cause; for Docker, the seccomp profile line |
| OS-004 | Windows IoRing availability and lockable working set | build number, limits |
| OS-005 | macOS GPU wired limit lower than the plan would like | current limit; the opt-in setting and its trade-off (never applied silently, ADR-0014) |
| OS-006 | heavy paging to disk or memory compression since the last load | paging and compressor activity; suggests a larger `resource.ram_headroom` or `resource.memory_yield = "aggressive"` |
| OS-007 | power: on battery, low-power mode, or a power-saving CPU governor | state |
| OS-008 | Linux unified-memory iGPU (Strix Halo class): the TTM `pages_limit` caps the GPU's share of RAM below what the plan wants | current limit and RAM; the plan that fits now and after tuning; the `syntropy tune` step (M3), never applied silently |
| STO-001 | free space against `store.min_free_bytes` and pending pulls; the size of the opt-in provenance records (RFC-0050 §10) | per volume |
| STO-002 | filesystem refuses `O_DIRECT` or reports no direct-I/O alignment; buffered mode active | volume, filesystem |
| STO-003 | store on a network filesystem or rotational disk | volume type |
| STO-004 | pack spot check: sample extents re-hashed against the sidecar | mismatches |
| STO-005 | a writable store root in chunk-file mode: its volume has no sparse files (exFAT, FAT32, HFS+), so each chunk is its own file and reads that span chunks are slower (RFC-0041 §4.2) | the root, and what the start-up probe found; move `store.dirs` to a volume with sparse files |
| NET-001 | LAN discovery blocked: multicast filtered, firewall profile, macOS Local Network permission denied | per interface; the exact firewall command or settings path |
| NET-002 | node on Wi-Fi while wired is available; links with p99 above 4 × p50 | link figures |
| NET-003 | MTU mismatch or fragmentation on data-plane links | path MTU |
| NET-004 | UDP blocked, so TCP fallback is in use | — |
| NET-005 | relay in use where a direct path should work (hole punching failed) | path type |
| NET-006 | metered connection in use | interface |
| RDMA-001 | RDMA over TB5 hardware present but not enabled | the Recovery-mode step (ADR-0011) |
| RDMA-002 | macOS build mismatch between RDMA peers | both builds |
| RDMA-003 | RoCE-capable NIC whose interface is not in `security.trusted_fabric` | interface |
| RDMA-004 | RDMA NIC present but `libibverbs` missing | package name per distro |
| CLU-001 | members, versions and protocol compatibility | per node |
| CLU-002 | clock skew between nodes beyond 50 ms (traces and lease timing degrade) | offsets |
| CLU-003 | suspicions, flaps and elections over the last hour | counts |
| CLU-004 | recent split and heal; re-proposed config | log indices |
| CLU-005 | the cluster has one keyholder and no recovery code | keyholder list; the promotion and recovery-code commands (RFC-0051 §2.6) |
| SEC-001 | each API listener's category, auth and TLS (RFC-0050 §11.1) | listeners |
| SEC-002 | state directory and identity key permissions | modes |
| SEC-003 | config files holding secrets readable by other users | paths |
| SEC-004 | admin-scope keys used from non-loopback addresses; expired or unused keys | key IDs, never values |
| SEC-005 | `api.allow_plaintext_lan = true`: API traffic on the LAN is unencrypted. Reported as a warning on every run while it is set | the binds affected; the command to turn it off |
| SEC-006 | local CA certificate near expiry or its leaf missing current names; clients that fail TLS against it, counted from handshake errors | expiry, names, failure counts; `syntropy trust` hint |
| CFG-001 | config warnings, unknown `SYNTROPY_*` variables, pending restarts, proposal conflicts | from RFC-0054 |
| CFG-002 | approximate mode in effect on any model | models and knobs, so it is never invisible |
| CFG-003 | a `debug.*` key is on (RFC-0054 §7.1), or the loaded conformance manifest is signed by the CI test key | which keys; the manifest's signer; the command to turn them off |
| RUN-001 | runner restarts in the last 24 h, by class | counts, last error |
| RUN-002 | measured ÷ predicted outside ±15% on any term (SC-30) | terms and ratios |
| RUN-003 | degraded modes in effect | RFC-0001 §10.2 paths |
| RUN-004 | telemetry or log drops | counters |
| RUN-005 | KV admission rejections or queue timeouts | counts |
| RUN-006 | daemon heartbeat stalls that put runners into orphan mode without a restart (RFC-0052 §6.5) | count and longest stall in the last 24 h |
| PRV-001 | what leaves this node: exporters, update checks, relays | destinations (§8); with `doctor --network`, every outbound destination contacted in the last 24 hours with byte counts (RFC-0034 §13) |

**Revision note, 2026-10-29 (COMP-E11-T02): STO-005.** RFC-0041 §4.2 says chunk-file mode "is flagged in `syntropy doctor`", and no check in this list did so. STO-005 is that check. It reads `v1/storage-mode.json`, which the store writes after each writable start's sparse probe, with `syntropy_store::read_storage_mode`. The record names the mode and the reason the volume has no sparse files, and reading it needs no writer lock, so the check runs under `--offline` while a daemon holds the root. PROD-E9-T02 builds it with the other STO checks.

### 6. Debug state dump

`syntropy doctor --bundle [--cluster] [--out FILE]`, the name RFC-0034 §13 uses, writes one `tar.zst` file locally. `syntropy debug dump` is an alias, and the API form is `POST /v1/syntropy/debug/dump` with `admin` scope. The bundle is never uploaded.

| Included | Source |
|---|---|
| `MANIFEST.txt`: every file in the bundle, its size and SHA-256, and the redactions applied | generated |
| build info: version, build ID, protocol versions, support tier, modules and their signatures | daemon |
| OS facts: kernel or OS build, driver and runtime versions, CPU, memory, devices, interfaces (names and address families, not addresses unless `--with-addresses`) | probe |
| resolved configuration with origins; secrets replaced by `[redacted]` | RFC-0054 |
| cluster state snapshot (the whole-state view of ADR-0009) and the last 10,000 log entries, with interface addresses, endpoint addresses and relay URLs replaced by stable per-bundle placeholders unless `--with-addresses` is given; the replaced fields are listed in `MANIFEST.txt` | cluster |
| resource graph and capability profiles | RFC-0010 |
| committed plans, their explanations and `PredictedCost` | RFC-0011 |
| runner states, restart history, residency digests, cache statistics | runners |
| metrics: current values and the in-memory 24 h store | §2 |
| flight recorder: spans and log records (no content) | §3, §4 |
| doctor report | §5 |
| crash records: panic messages, backtraces, last log lines, sticky device errors | state directory |

**Address redaction is applied to every record that carries addresses**, not only to OS facts. The "Nothing leaks" canary test includes an address canary.

**Never included**, whatever the flags: prompts, outputs, token IDs, KV pages, weights, API keys, Hub tokens, the node's secret key, invite tokens, TLS private keys, or stored Responses. The bundle builder works from an allow-list of record types, not a deny-list, so a new record type is excluded until someone adds it deliberately.

- **Size.** Capped at 200 MiB by default (`--max-size`). The flight recorder and metrics history are truncated first.
- **`--cluster`** collects each member's bundle over the node-query channel, with large parts over the bulk plane, and nests them by node name.
- **On finish** the command prints the path, size and SHA-256, plus a reminder to read `MANIFEST.txt` before sharing.

**Crash records.** A panic in the daemon or a runner, and a runner's exit by signal or with a sticky device error (RFC-0001 §3.4), write a record to `<state>/crashes/`: build ID, panic message and backtrace, the last 200 log lines, and the device error report if any. The 20 most recent are kept. They are shown by `doctor` (RUN-001) and included in dumps, and nothing sends them anywhere.

**Revision note, 2026-10-29 (PROD-E9-T04): the bundle and crash records, as built.** `syntropy_daemon::bundle` builds the bundle and `syntropy_daemon::crash` keeps the records.
- **Files.** `MANIFEST.txt`, `build.json`, `config.txt` (`config show`'s text, secrets redacted), `cluster-state.json`, `cluster-log.jsonl`, `flight-recorder.jsonl`, `doctor.json` and `crashes/<name>.json`, in a POSIX ustar archive compressed with zstd level 3. `MANIFEST.txt` lists each other file as `<SHA-256>  <bytes>  <path>`, then where addresses were replaced, where secrets were replaced by `[redacted]`, what was left out, what was truncated, and what was not collected. `bundle::verify` checks a bundle against it.
- **The allow-list is of record types and of fields.** Each record is serialised, then projected through a declared shape: at every level the shape names field by field, a field or enum variant it does not name is left out, and only its name is listed in `MANIFEST.txt`. So a field a record type gains later, such as RFC-0032 §6.2's `api_keys` or `invites` in `ClusterState`, is excluded until someone adds it deliberately. Below a level the shape keeps whole, new fields are kept too, and every string in them passes the address redactor; such subtrees are plain data (identifiers, counts, enums). A payload the log keeps as `Opaque` (a CBOR value no RFC has fixed yet) is left out wherever it occurs. Configuration values are replaced by `[redacted]` wherever the state and the log hold them: cluster keys, a config patch, and an instance policy's instance-scope keys (`domain`), since such a key may be secret-typed; `config.txt` shows them with `config show`'s redaction. Hot-thread log records keep only their non-content arguments, through the build's callsite table; a record whose callsite is unknown keeps no arguments. Records are serialised human-readable, so an address type is its text (`203.0.113.7`, `[2001:db8::5]:9000`), never an integer array no text scan would find; a map key that is not a string is written as text.
- **Addresses.** Every string kept, and every map key, is scanned. URLs (`scheme://…`) become `<url-N>`. IP addresses become `<addr-N>` wherever they occur: alone, with a port, bracketed, with a zone, inside a longer token such as a dotted key (`p2p.peers.2001:db8::5.x`) or after a `word:` prefix, and IPv4 with zero-padded octets. A candidate with no digit counts only when no letter or digit touches it, so `Vec::new` and `daemon::logging` stay. A host name outside a URL is kept, with or without a port (`relay.example.net:3478`); the maintainers decide whether such endpoints are replaced too. With `--with-addresses` addresses are kept, but a URL's userinfo (`user:password@`) is replaced by `[redacted]@`, since it is a credential, and `MANIFEST.txt` lists where.
- **Size.** The cap is checked against the archive before compression, less 256 KiB kept for `MANIFEST.txt`, and against the compressed file. The flight recorder is truncated first, then the cluster log, oldest entries first; `TooLarge` if it still does not fit. The metrics history is not collected yet, so it is not truncated.
- **Crash records** are one owner-only JSON file each, `<unix ms, 20 digits>-<pid>-<n>.json`. Their log lines come from the logger's tail of the last 200 records, formatted as text with every content field removed whatever `logging.include_content` says. A panic is recorded by a hook that then runs the hook it replaced. A runner's exit is recorded from its exit status: a signal on Unix, and on Windows a system `NTSTATUS` error code (severity bits set, customer and reserved bits clear), so `exit(-1)` is not an exception. A runner's exit with a sticky device error, and the device error report, have no record field yet and no owning task; the task that wires device errors to the supervisor should add them.
- **Wiring.** The command and `POST /v1/syntropy/debug/dump` are PROD-E6-T08's, and RUN-001's listing is PROD-E9-T03's. A runner's exit is recorded by the process host (`ProcessRunner::record_crashes`) when it reaps a runner it was given a `CrashSink` for; a runner killed with `SIGKILL` through that host leaves a record that the next bundle holds. `--cluster` waits for the node-query channel.

**Revision note, 2026-10-30 (PROD-E9-T12): the process host's own kills.** A kill the host makes is recorded from the host's reason, on every OS, not from how the exit status reads: on Windows `TerminateProcess` leaves exit code 1, which no status rule can tell from a program's own exit. RFC-0001 §3.4 gives the host two kills, and both are abnormal: the supervisor's kill of a hung or stalled runner, and the kill of a freshly spawned runner whose `Hello` is refused. `ProcessRunner::kill` takes the host's own reason type, `HostKill`: `Supervisor(KillReason)` (`hung` or `stalled`) or `Refused`. The reap writes `CrashCause::HostKilled { reason, signal, exit_code }`, with the signal on Unix and the exit code where the status has one (on Windows, the code `TerminateProcess` left), in place of a plain `Signal` record. The supervisor's kills reach `ProcessRunner::kill` through the supervisor shell's `Effects::kill`, which DIST-E1-T16 wires. A shutdown drains runners and never kills them, and dropping a runner's host closes its channel without a kill, so neither leaves a record; nor does a clean exit. An exit the host did not cause is recorded from its status as before: a signal on Unix, a system `NTSTATUS` error on Windows. A kill counts as the host's only if `Child::kill` succeeded on a runner that had not already exited, and only if the status is what the kill leaves: on Unix the `SIGKILL` the host sent, on Windows exit code 1, the code `Child::kill` passes to `TerminateProcess`. A runner that exits on its own between the host's check and its kill, with any other status, is recorded from that status. Revised 2026-10-30 after hosted Windows CI caught a runner's own exit code 101 recorded as the host's kill: the residual window is now only a runner that itself exits with code 1 in it, which cannot be told from the kill, and is accepted.

### 7. Optional export: Prometheus and OpenTelemetry

Both are off by default, run only in the daemon, and export only this node's data. A cluster is monitored by pointing the monitoring system at every node.

#### 7.1 Prometheus

With `obs.prometheus = true`, `GET /metrics` on the API listener serves the §2.3 catalogue in the Prometheus text format, or OpenMetrics when the scraper asks for it. Histograms are exported as native bucketed histograms, with buckets merged to powers of two to keep the series count bounded. On a non-loopback listener the endpoint needs a `dashboard`-scope key (RFC-0050 §11.3). A scrape is served from the daemon's aggregate store and never touches runners.

#### 7.2 OTLP

With `obs.otlp.endpoint` set, the daemon runs the OpenTelemetry SDK exporters (`opentelemetry`, `opentelemetry-otlp`) over gRPC or HTTP/protobuf (`obs.otlp.protocol`), with `obs.otlp.headers` for authentication. It exports:
- **metrics**: the §2.3 catalogue every 10 s, cumulative temporality;
- **traces**: every span in §3.1 that was recorded, through a batch processor with a bounded queue;
- **logs**: records at `info` and above, or at the level of `logging.level` if lower.

Resource attributes: `service.name = "syntropy"`, `service.version`, `service.instance.id` (node ID, short form), `host.name` (`node.name`), `syntropy.cluster` (`cluster.name`), `syntropy.support_tier`. Queues are bounded: on overflow the exporter drops and counts, and it never applies back-pressure to telemetry intake. The exporter's own failures are logged with repeat suppression, and doctor PRV-001 shows the endpoint.

Configuring an exporter to a host other than this node is the one way to make Syntropy send observability data off the machine, and it is the user's explicit act. The daemon logs the destination at start.

### 8. Privacy: zero telemetry by default

ADR-0510 decides that Syntropy has no telemetry pipeline of any kind: no usage statistics, crash uploads, install counts, model-choice reporting or opt-in variants of these. This is everything a node originates, by default:

| Destination | What | When | How to turn it off |
|---|---|---|---|
| cluster members | control, data and bulk planes; node-query responses | while a member | `syntropy cluster leave` |
| model origins (the Hub or another URL) | range and chunk requests for models, with the node's own Hub token | only for models the user pulls (or `api.auto_pull`) | do not pull; `--offline` (RFC-0041 §7.5) |
| the Hub's metadata endpoints | header range reads for estimates, model cards and licence identifiers, gated-access status for this node's token | only when a user asks for an estimate, opens a model page or starts a pull | `--offline` |
| the update repository (RFC-0052 §7) | TUF metadata (`timestamp.json`, then snapshot and targets) and, when updating, artefacts: plain HTTPS GETs whose paths name the platform and channel. The installed version is not sent; an artefact download reveals the version being fetched. No cookies or identifiers, as RFC-0034 §13 also states | every `update.check_interval` (24 h) and on `update check` | `update.mode = "off"` |
| NVIDIA's redistributable server | HTTPS GETs of the CUDA archives the `cuda13` module's manifest pins (RFC-0052 §4.4) | only when `setup` installs or updates `cuda13` | `--modules none`; the installed-toolkit fallback |
| relay servers | the relay protocol's connection, which shows the relay operator which node IDs communicate | only while the cluster has invite-joined members or outstanding invites, or a member that answered yes to "Reach this cluster away from home?" at pairing, recorded as `NodeInfo::off_lan_reach` (RFC-0034 §13; RFC-0051 §2.6). The built-in relays are a hosted relay service listed in the signed `relays.json` target (RFC-0052 §7.2); free public relays are used only for internal and alpha testing | `net.relay_mode = "custom"` for the user's own relays, or `"disabled"` (RFC-0030) |
| anything else | nothing | — | — |

Two possible destinations are deliberately absent. Model-signature checks (RFC-0034's `models.signatures`) use a signature trust root embedded in each release, not fetched. Configuration files reference their JSON Schema by a local `file://` path (RFC-0054 §1), so editors fetch nothing. Doctor PRV-001 lists exactly this table as it applies to the node, with the current settings. `syntropy doctor --network` adds what was actually contacted in the last 24 hours, with byte counts (RFC-0034 §13). That includes the user-triggered paths: pulls, Hub metadata reads and update checks run by hand. Anything that would add a row needs an ADR that supersedes ADR-0510.

**Demand signals without telemetry.** The project learns about platforms and failures from what users choose to share: `syntropy bench --report` writes a commit-tied benchmark file (RFC-0061), `doctor --bundle` writes a diagnostic bundle, and both are attached to issues by hand.

**Tester reports**. The alpha testers run the single-device product from the M1 release candidate, and send reports by hand (RFC-0061 §6.5). A report must hold what their purpose needs and nothing more, so `syntropy bench --report` writes one `tar.zst` file with:
- the benchmark results with their key, where the rig fingerprint is computed from this machine's probe instead of a lab inventory record;
- the capability profile and probe export (RFC-0010), including NVMe sustained and burst rates and the thermal trace HW-004 reads (OQ-HW-03);
- predicted against measured per cost term for each run (§2.2), so prediction error can be collected per host class (SC-30);
- `doctor --json` output;
- a `MANIFEST.txt` like the bundle's (§6), listing every file and every redaction.
It is built from the same allow-list as `doctor --bundle`, with addresses replaced unless `--with-addresses` is given, and it never holds prompts, outputs, token IDs, keys or invite text; the canary test covers it. Nothing sends it anywhere.

#### 8.1 Privacy notice for the project's own surfaces

The software sends nothing to the project (ADR-0510), but the surfaces the project runs still handle personal data: the release host and CDN log IP addresses, and issues, shared bundles, tester reports and the relays involve people. With the first public release the project publishes a one-page privacy notice that states:
- what each surface logs: the release host and CDN, the website, issue intake, shared benchmark reports and bundles, tester reports;
- CDN and release-host log retention, at most 30 days [target];
- that the website uses no analytics and no cookies, so it needs no cookie banner;
- how shared dumps and reports are handled and deleted, including deletion on request;
- that AI agents process issues, shared files and tester reports, and which processors run that tooling;
- who operates the relays and what a relay sees (the node-ID pairs above).
The notice is written so that it meets GDPR and UK GDPR transparency requirements for EU and UK users of the release host. Agents draft it and keep it in step with the destination table above (PROD-E9-T10); publishing it is a **Maintainer action**.

## Performance model and budgets

Observability shares hot threads with the token path, so its cost counts against SC-3: measured token time may exceed the cost model by at most 5%. Observability may use at most a tenth of that.

| Item | Budget [target] | Basis |
|---|---|---|
| Total observability overhead on step time at default settings | ≤ 0.5% | one tenth of SC-3's 5% unexplained share; measured as an A/B against a build with recording compiled out, on the reference rigs |
| Counter increment on a hot thread | ≤ 5 ns | a relaxed load and store on an owned cache line, with no read-modify-write instruction |
| Histogram record | ≤ 20 ns | bucket index from leading zeros, plus one counter add |
| Span record on a sampled step | ≤ 100 ns per span | one 64-byte write into an SPSC ring |
| CPU-side term timing every step | ≤ 16 cycle-counter reads per stage step, ≈ 0.4 µs [estimate at ≈25 ns per read] | ≤ 0.01% of a 5 ms step; the budget is set for the fastest plausible steps (≈ 5.7 ms on 7B Q4 on an RTX 4090, RFC-0050 Performance) |
| GPU-side term timing | only on sampled steps (1 in 64) | keeps device timing events off most steps; if a timed graph variant costs X% of a step, the amortised cost is X/64% |
| Telemetry ring traffic per runner | ≤ 1 MiB/s at default sampling | chosen so the ring and the daemon's intake stay negligible |
| Daemon memory for observability | ≤ 64 MiB flight recorder (spans, logs) + ≤ 32 MiB step recorder (`obs.step_recorder`) + ≤ 40 MiB metric history | the step recorder is budgeted separately, because 15 minutes of step records would exceed the flight recorder: 175 × 900 × 256 B ≈ 40 MB per stage at 175 steps/s [estimate]. Metric history is kept only for the series the dashboard draws (≤ 2,000): 1 s points for 15 min ≈ 14 MB plus 1 min points for 24 h ≈ 23 MB [estimate]. All bounded so the daemon stays within `daemon.max_rss` |
| `/metrics` scrape | ≤ 50 ms for 20,000 series | served from aggregates; no runner involvement |
| `doctor --offline` | ≤ 10 s | cached probe results are reused; no microbenchmark is re-run unless `--rerun-probe` |

The flight recorder size bounds how far back traces and logs can be inspected without an exporter. At default sampling a decode-heavy instance produces, per sequence, about 8 + (tokens − 8)/64 sampled steps × stages × 2 spans [estimate], so 64 MiB of 64-byte records (≈1 M spans) covers thousands of sequences on a 3-stage ring.

## Alternatives considered

| Alternative | Why not |
|---|---|
| The `tracing` crate's spans and events on hot threads | subscriber dispatch, possible allocation and lock-protected registries on the per-token path (invariant 9); kept for non-hot code only |
| A Prometheus client library's atomics on hot threads | shared atomic counters bounce cache lines between dispatcher and CPU-pool threads; per-thread cells with a snapshot thread cost nothing in contention |
| The OpenTelemetry SDK inside runners | puts an async exporter and its runtime in the runner; ADR-0012 forbids that on hot paths, and the daemon can export everything anyway |
| Always tracing every step | ≈ stages × 2 spans per token, which buries exporters and the flight recorder; coherent sampling plus on-demand full traces gives the same insight |
| Per-stage independent sampling | sampled steps would rarely line up across stages, so no complete ring picture would exist |
| Per-term child spans | up to 16× span volume for data that fits in attributes |
| Opt-in telemetry with anonymous IDs | ADR-0510: any upload path is a liability for a local-first system, and opt-in data skews toward enthusiasts anyway; shared bench reports and dumps serve the same purpose |
| Logging prompts at debug level by default | debug logs get attached to issues; content must take an explicit switch |
| Exporting per-expert metrics | hundreds of thousands of series for frontier MoE models; per-layer aggregates plus traces are enough |
| Kernel-level tracing (eBPF) for I/O and scheduling | Linux-only; useful for development, documented as a technique, not a product feature |

## Testing and validation

| Claim | Test |
|---|---|
| Overhead ≤ 0.5% | benchmark in RFC-0061: decode tok/s with observability compiled out against default settings, on every Tier 1 reference rig; a regression gate at the budget |
| Hot-path recording is allocation-free and lock-free | a test allocator that panics on allocation from hot threads after `Warming`; a lint forbidding `tracing` macros in hot-thread modules |
| Measured terms add up | on the fake-device testkit with injected per-term delays, `StepRecord` reproduces each injected term within one histogram bucket, and Σ critical-path terms equals the step total within 1%; with injected overlapping GPU and CPU sites in one MoE block, the binding site's term gets the block time, the other site appears in `overlapped`, and a plan predicting the same binding site shows ratios within the testkit's noise; `size_of::<StepRecord> == 256` |
| Cross-node traces are coherent | multi-node simulation with a skewed clock per node: every sampled step has a span on every stage with correct parent links; estimated offsets are within the simulated asymmetry bound; Σ stage + Σ hop ≈ ring time |
| Sampling is deterministic | the same (trace ID, step) gives the same decision on every node and in replay |
| Nothing leaks | canary test: prompts, outputs, API keys, Hub tokens and invite tokens carrying unique canary strings run through the full suite; the canaries must not appear in any log sink, metric, span, debug dump or crash record; an address canary (interface address and relay URL) never appears in a bundle made without `--with-addresses`; `doctor --network` tests cover the user-triggered Hub metadata path |
| Cross-build decoding works | a daemon of build N decodes logs, metrics and spans from a re-adopted runner of build N−1 through the `Hello` tables, checked against golden records |
| No unexpected traffic | network-namespace test (Linux) and firewall-log test (macOS, Windows): a daemon idle for 24 h with default settings contacts only the update repository and nothing else; with `update.mode = "off"` it contacts nothing. A scripted session then exercises the user-triggered paths (an estimate, a model page, a pull, `update check`): every destination matches a row of §8, update requests carry no installed version, and `doctor --network` lists each one |
| Cardinality stays bounded | the series count for each v1 family's largest model on the largest reference node stays under 20,000 |
| Prometheus and OTLP output are valid | `promtool check metrics` on scrape output; an OTLP test collector receives metrics, traces and logs with the stated resource attributes |
| Doctor checks are right | unit tests with fake OS facts for each check (memlock limits, seccomp-blocked io_uring, WSL2, macOS wired limit, single-channel RAM, x4 links, unenabled TB5 RDMA, throttled NVMe), plus golden output |
| Dumps are complete and safe | a dump from the testkit cluster contains every file listed in §6, its manifest hashes verify, and the canary check passes |
| Rings drop rather than block | stress test fills the telemetry and log rings; step times are unaffected and drop counters match |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | `obs` crate: metric descriptors, per-thread cells, histograms, telemetry and log rings; daemon log formatting and sinks; `StepRecord` (§2.5) and the telemetry drain behind `debug.telemetry_api`, which RFC-0061's M0 harness needs; `syntropy_exec_step_seconds`, `t_fixed`, request TTFT and TPOT; the `CostTerm` keys shared with RFC-0011 |
| M1 | the tier, I/O, expert-cache, KV, runner and memory-pressure metrics (the vision's M1 metrics item); per-term predicted against measured with drift input to RFC-0010; flight recorder; local tracing; `doctor` (local checks, including DRV-006 and DRV-007 report-only and INS-004); `doctor --bundle` (local); `bench --report` for testers; crash records; Prometheus endpoint; the privacy notice with the first public release (§8.1) |
| M2 | link, cluster and store metrics; trace propagation through `ActivationFrame::trace`; clock-offset estimation; `doctor --cluster`; `doctor --bundle --cluster`; trace assembly over the node-query channel |
| M3 | OTLP export of metrics, traces and logs; on-demand full traces; recovery spans; speculation metrics |
| M4 | catalogue frozen under the compatibility policy; overhead and leak tests as release gates (the vision's M4 "complete observability" item) |

## Open questions

1. **GPU per-term timing.** A timed variant of every captured step graph doubles graph memory; event nodes left in every graph cost a little on every step. Which is cheaper per backend? Spike with RFC-0012 and RFC-0040 on CUDA graphs and Metal indirect command buffers in M1. The 0.5% budget decides.
2. **Unified-memory attribution.** On `Unified` tiers, CPU and GPU terms compete for one memory bus, so measured terms may not add up the way the solver's split assumes. Is a combined `UnifiedRead` term more honest? Decide from M1 Apple and Strix Halo measurements.
3. **GenAI semantic conventions.** The OpenTelemetry GenAI conventions are still evolving. Pin to a version, and map through a single module so that renames are one change.
4. **Cluster-wide Prometheus (dropped until users ask, OQ-PRD-23).** An aggregate `/metrics?scope=cluster` built from node-query digests would be convenient for one scraper, at the cost of coupling and staleness. It is post-1.0 and comes back only on user demand.
5. **Windows event log volume.** Is warnings-and-above the right threshold, or does it flood the log on flapping Wi-Fi links despite repeat suppression? Measure in M1 soak tests.

## Proposed amendments to RFC-0001

- **A1. §13 per-step telemetry ring.** Name the record `StepRecord` and give its encoding to this RFC (§2.5). Add the fields RFC-0061 amendment B1 asks for: `t_miss` split into CPU in place and NVMe demand, `t_switch` total, `t_sample`, `credit_stall` per outbound hop, accepted tokens per verify window, and the step's token count. Without them, §14's decomposition is inferred rather than measured.
- **A2. §14.** Key `PredictedCost` by `CostTerm` (§2.2), so each runner compares like with like, and SC-3 and SC-30 can be checked per term. RFC-0011 owns how each term is predicted.
- **A3. §3.5 IPC.** The `telemetry` ring carries `TelemetryRecord` (§1), with the drain task as its only producer. Add one SPSC **step ring** per dispatcher to `ShmLayout`, carrying `StepRecord` (§2.5). The `log` ring carries `LogRecord` with static callsite IDs. `Hello` carries a digest of the runner's callsite, metric and span-name tables, and the tables themselves on mismatch, so a daemon can decode a re-adopted runner of the previous build (§4.1). The ring schemas are part of `ipc_protocol` (RFC-0052 amendment A1).
- **A4. §2.2 planes.** The node-query channel proposed in RFC-0050 amendment A4 is also how telemetry digests, trace assembly, the telemetry drain for remote stages, `doctor --cluster` and `doctor --bundle --cluster` reach other nodes. None of it enters the event log.
- **A5. §13 labels.** Extend the label list. Keep the identity labels as written, and add:
  - enumerated labels whose values are fixed at compile time (`kind`, `term`, `class`, `purpose`, `state`, `priority`, `result`, `outcome`, `path`, `plane`, `transport`, `quantile`, `phase`, `step`, `reason`, `source`, `knob`, `fidelity`, `shape`, `thread_kind`, `ring`, `to_tier`);
  - `layer`, on expert-cache metrics only;
  - `model`, on request metrics only.
  Per-term step times, hit-rate-by-layer and per-shape request metrics cannot be expressed with the current eight labels.
- **A6. §13 debug surface.** List the three gated endpoints beside `/debug/state`: `/v1/syntropy/parity` (RFC-0060), `/v1/syntropy/debug/telemetry` (RFC-0061) and bench controls on generation. All three are loopback-only and off unless their `debug.*` key is set (RFC-0050 §13.6, RFC-0054 §7.1).
