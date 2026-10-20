# RFC-0012: Execution scheduler and speculative decoding

- Status: Draft
- Area: placement and scheduling
- Depends on: RFC-0001, RFC-0011, RFC-0022, RFC-0023, RFC-0040, RFC-0031, RFC-0032, RFC-0060 (§2.3, §5.5), ADR-0600, ADR-0104, ADR-0105, ADR-0106, ADR-0107, ADR-0002, ADR-0006, ADR-0010, ADR-0012, ADR-0017, ADR-0019
- Date: 2026-10-19
- Revised: 2026-10-20: up to 4 concurrent decode rows in latency mode (§4.6); v1 produces no draft heads (§9.1); open question 6 and the grammar replay on resume (§8.6) answered

## Summary

This RFC specifies how a runner executes its stages: the `exec` crate (step programs lowered from the graph, per-device dispatchers, the CPU compute pool, graph capture) and the `sched` crate (the step scheduler, chunked prefill, batching, the sampler, speculative drafting and verification, admission). It covers:
- the **step loop** each runner's dispatchers run, with no async runtime, no allocation, no lock and no synchronising driver call per step (RFC-0001 invariant 9);
- **chunked pipelined prefill** around the ring, and the switch to PCIe-streamed weights above n*;
- **single-stream decode** around the ring, where the tail returns each token to the head and to ingress at once;
- **throughput mode**: continuous batching with D micro-batch groups circulating so every stage is busy, KV-based admission and per-request priorities;
- **whole-step graph capture** (CUDA graphs, Metal indirect command buffers, prebuilt Vulkan command buffers) in which MoE routing, KV paging, tree masks and sampling read device-resident tables, so captured graphs serve every step;
- **sampling at the tail** with a counter-based RNG, grammar masks and token-level stops;
- **concurrent decode rows in latency mode**: up to 4 sequences decode together on dense and RAM-resident MoE instances when their KV fits, while NVMe-tiered MoE instances serve one sequence at a time (§4.6);
- **speculative decoding**: MTP heads on the tail, a draft model on the head, prompt-lookup candidates merged in, and draft heads on the tail only where one is supplied with the model (v1 produces none, §9.1), chain and tree verify with exact acceptance, a window size bounded by the plan and adapted per sequence from measured acceptance;
- **admission control** at the head, fed by per-request routing at ingress (RFC-0011 §13).

The head stage's scheduler decides what every step of an instance contains. Other stages execute frames in step order. The tail samples, accepts and drafts.

## Motivation and goals / non-goals

Decode time is spent in kernels, DMA, I/O and the data plane; everything between them is overhead on a 10–300 ms token (ADR-0012). Evidence the design responds to:
- A two-point fit gives t_fixed ≈ 5 ms per token for one existing GPU engine (cost model §2.2); SC-2 targets ≤ 0.2 · B_ref / W_probe, ≈ 0.95 ms on an RTX 4090.
- A driver sleep/wake round trip costs ≈ 0.3 ms on Windows, so a surveyed engine spins on a host doorbell the GPU writes and captures the whole pass as one CUDA graph per verify-window size (`tiered-memory-inference.md` §5.1).
- A MoE layer hands off GPU → CPU → GPU each token when misses execute in place; at 60 layers and 20 µs per wait that is 2.4 ms per token if waits are slow (ADR-0012).
- Pipelined chunked prefill is the main place a pipeline speeds up a single request; the bubble fraction is (D−1)/(m+D−1) for m chunks (cost model §3.7).
- Speculative windows amortise slow tiers and hops: 1.6–1.8× on a tiered single-GPU MoE engine with an MTP draft, and 25–45% lower time per token for 14–70B models with a draft on the head of a pipeline (ADR-0017).
- MoE throughput does not scale with batch depth on slow tiers, because 8 tokens touch 7.2× the routed bytes of one (cost model §3.9).

**Goals**
1. Meet the scheduler's rows of the RFC-0001 §14 budget: `t_fixed` per SC-2, no host synchronisation inside a step, ≤ 5% unexplained time per token (SC-3, SC-13).
2. Exact output by default (D3): every speculative, batching and capture path stays inside ADR-0019's tolerance; nothing in this RFC turns an approximate knob on.
3. Single-stream latency first (D2): the active interactive sequence's step waits at most for one running, capped prefill chunk of another sequence per stage (§4.3).
4. Throughput mode reaches ≥ 0.85 × `min(M, D) / (Σ_s t_s + D · t_hop)` (SC-15).
5. Realised speculative speedup ≥ 0.85 × the predicted value, and speculation switches itself off when the prediction is below 1 (SC-21).
6. TTFT ≤ 1.15 × the cost-model bound plus queueing (SC-18).

**Non-goals**
- Kernel implementations and the provider contract (RFC-0040). This RFC uses `OpKind`, `StepDesc`, doorbells and the capture rules RFC-0040 defines.
- Expert-cache policy, prediction and prefetch (RFC-0023). This RFC calls them at fixed points in the step.
- KV page management (RFC-0022). This RFC calls `KvStore` at fixed points.
- Wire formats (RFC-0031) and cross-node TP/EP collectives and recovery (RFC-0032, RFC-0033).
- Detokenisation, stop strings, tool and reasoning parsing (RFC-0050, at ingress).
- Training, evaluating or publishing draft heads. v1 produces none; the loader stays so heads supplied with a model, or accepted later under the conformance and benchmark gate, can run (§9.1, ADR-0017).

## Design

### 1. Structure and roles

#### 1.1 Crates and threads

| Component | Crate | Thread (RFC-0001 §8.2) | Role |
|---|---|---|---|
| Step program builder | `exec` | runner main thread during `Warming` | lowers the stage's layer range to a step program per bucket and captures it (§2, §7) |
| Device dispatcher | `exec` | one per GPU/NPU device and one per CPU device | owns `DeviceCtx`, streams, graphs, the device's residency entries and expert cache; runs the step loop (§3) |
| Primary dispatcher | `exec` | the dispatcher of the device that holds the stage's first layer | also hosts `sched`, owns the stage's KV store (single writer, RFC-0022 §1) and the inbound and outbound channels |
| Scheduler | `sched` | called by the primary dispatcher once per step; not a thread | composes steps at the head (§4–§6, §10); at every stage, orders frames, commits KV, tracks sequences |
| Sampler | `sched` | tail's primary dispatcher | logits processing, sampling, acceptance, drafting (§8, §9) |
| CPU compute pool | `exec` | per NUMA domain, sized by the plan | in-place kernels: dense RAM layers, expert misses, CPU attention over spilled KV, grammar masks when configured |
| Housekeeping | `exec` | one, low priority | expert-cache admission (RFC-0023), KV spill scheduling (RFC-0022), statistics |

`sched` depends on `exec`, `tier`, `graph` and `text` only (RFC-0001 §4.1): no network crate, no tokio. Frames arrive and leave through SPSC rings to the runner's `net` domain (RFC-0001 §8.3).

#### 1.2 Roles by stage position

| Position | Receives | Decides | Sends |
|---|---|---|---|
| Head | `SequenceOpen`, `SequenceControl` from ingress; `TokenFrame` (ring return) from the tail | step composition: which sequences, which positions, prefill chunking, verify windows when drafting on the head; admission; prefix-cache match (RFC-0022 §7.3) | `ActivationFrame` to stage 1 (or runs the whole ring when it is the only stage) |
| Middle | `ActivationFrame` from its predecessor | nothing about composition: it executes frames in `step` order | `ActivationFrame` to its successor |
| Tail | `ActivationFrame` from its predecessor | sampling, acceptance, drafting for MTP and draft heads, token-level stops | `TokenFrame` to the head (with the next draft) and to ingress, at once |

A one-stage plan is head and tail at once; the ring return is an in-process hand-off with no frame.

The head's authority over composition follows from two RFC-0001 rules: every stage processes each sequence's steps in order and serves its decode lane before replay and prefill (RFC-0001 §5.8 as amended by C7; RFC-0032 §3.2), and the next step of a sequence cannot begin before its token returns (§6.2). One writer of composition per instance means no stage ever waits for a negotiation.


### 2. Step programs

A **step program** is everything one stage does for one step of one bucket, lowered once in `Warming` from the graph IR (RFC-0043), the plan's `TierAssignment`s and the providers' prepared ops.

```rust
pub struct StepProgram {
    pub stage: StageIx,
    pub bucket: Bucket,
    pub segments: Vec<Segment>,                  // in execution order; segments on different devices may overlap
    pub doorbells: DoorbellMap,                  // flags per RFC-0040 §4.2: per MoE layer Rd[l], Rc[l][d], F[l], Q[l][d];
                                                 // per CPU attention site Ra[l][d], Qa[l][d]; per hand-off H[i];
                                                 // per descriptor parity DC[p], O[p], M[p], tap_flag[p]
    pub input: InputSpec,                        // frame payload → first segment; or embeddings at the head
    pub output: OutputSpec,                      // registered send buffer, or logits/sample block at the tail
    pub kv_layers: Range<LayerIx>,
    pub parallel_blocks: Vec<ParallelBlock>,     // §11: spans whose sites run concurrently
}
/// A span whose sites run concurrently: a MoE layer's GPU, CPU and NVMe work, or a split-attention
/// layer's GPU and CPU parts. Built at lowering from the committed plan's `StageCost` (RFC-0011 §2).
pub struct ParallelBlock {
    pub segments: Range<u16>,                    // indices into `segments`
    /// Per site: its term and its start and end timing points (a timestamp slot on device sites,
    /// RFC-0040 §3.7; a `TickSource` read on CPU sites).
    pub sites: SmallVec<[(CostTerm, TimingPoint, TimingPoint); 3]>,
    pub predicted_binding: CostTerm,             // the plan's binding site for this block
}
/// Buckets follow RFC-0040 §3.2: a prepared op handles any rows_active ≤ rows.
pub enum Bucket {
    Decode,                                      // rows 1
    Verify { rows: u16 },                        // T_max = γ_max + 1, or the tree size budget
    Batch { rows: u16 },                         // powers of two up to micro_batch × T (throughput mode)
    Prefill { chunk: u32 },                      // execute-in-place sites
    PrefillStream { chunk: u32 },                // HostPinned groups streamed over PCIe (chunk ≥ n*)
}
pub enum Segment {
    Device { device: DeviceId, graph: Option<CapturedGraph>, eager: Vec<PreparedLaunch> },
    CpuPool { domain: u16, ops: Vec<CpuOp> },    // dense RAM-homed groups, CPU attention partials
    MoeLayer { layer: LayerIx, gpu: DeviceId, cpu: u16 },   // RFC-0040 §3.4 rows 1–7, all three sites
    HandOff { from: DeviceId, to: DeviceId, flag: DoorbellId, bytes: Bytes },
}
```

**Lowering rules.**
1. Walk the stage's layers in order. Consecutive groups with the same `ComputeSite::decode` device form one `Device` or `CpuPool` segment; a change of device inserts a `HandOff` (one activation through host-mapped memory and a doorbell, `t_switch`). The solver's rounding already made each device's layers consecutive where it could (RFC-0011 §11.1), so most stages have one or two hand-offs.
2. Each MoE layer becomes a `MoeLayer` segment that encodes RFC-0040 §3.4's seven rows: route on the GPU, route doorbell, hits on the GPU, copied misses through staging, in-place CPU misses, and an F32 rank-order combine.
3. Attention over KV pages that the plan spills to `HostPinned` uses RFC-0022 §5.3's split attention: the GPU computes over hot pages and the CPU pool over spilled ones, merged exactly.
4. `PrefillStream` replaces each group in the tier manager's current `stream_set(stage, n)` (RFC-0020 §10) with a double-buffered stream. The set, not the plan's static `min_tokens`, decides: it excludes groups whose home is still `Underpinned` and follows the live n*. The program for the stream set the tier manager reports at `Warming` is captured then, with the other buckets (§7.1). If the set changes while serving, the stage runs its `PrefillStream` steps eagerly for the new set: prefill is not launch-bound (ADR-0105), and nothing is captured or allocated on the token path. Where the provider has per-group predicated copy nodes (a device-side skip read from the descriptor), one captured program serves every subset of the captured set instead. Groups outside the set run in place in the same program. In the stream: the copy of group i+1 into VRAM staging overlaps the GPU's compute of group i (SC-10). MoE layers in this bucket route all chunk rows first, then stream the union of selected experts from RAM in expert order and run a grouped GEMM on the GPU; experts missing from RAM are demand-read through the I/O engine ahead of their turn.
5. The head's program starts with `RowGather` for token embeddings and other `RowLookup` tables. The tail's program ends with `Logits`, `LogitsProcess`, sampling and, for verify buckets, `VerifyAccept` (§8, §9).
6. Draft programs are separate step programs on the drafter's device: MTP or draft heads at the tail (buckets for 1 and for the tree width), a draft model at the head (its own Decode and Prefill buckets).

The bucket set per stage is: `Decode`; `Verify { T_max }` when speculation is on, with `T_max = SpeculativeConfig::max_window + 1` (RFC-0001 §5.4: `max_window` is γ_max); `Batch` buckets in throughput mode; a **prefill ladder** of `Prefill` buckets at {64, 256, 1024, 4096} tokens that are below the plan's `prefill_chunk`, plus `prefill_chunk` itself; and `PrefillStream { prefill_chunk }` when any group streams at that chunk. The router picks a prompt's chunk from the instance's captured ladder (RFC-0011 §13; RFC-0032 §3.8 references this set). A verify window shorter than `T_max` runs in the `Verify { T_max }` bucket with fewer active rows; memory-bound GPU ops read weights once whatever the row count, and CPU kernels compute only `rows_active` rows (RFC-0040 §3.2).

This RFC owns the graph count; RFC-0040 and ADR-0105 refer to this table. With a plan chunk of 4096 and defaults:

| Program | Buckets | Parities | Instances |
|---|---|---|---|
| Stage step, latency mode | `Decode`, `Verify`, prefill ladder (64, 256, 1024, 4096), `PrefillStream` | 2 | (1 + 1 + 4 + 1) × 2 = 14 |
| Stage step, latency mode with concurrent rows (§4.6; dense and RAM-resident MoE stages only) | `Batch` at 2 and 4 rows | 2 | 4 |
| Stage step, throughput mode extra | `Batch` at 2, 4, 8, 16, 32, 64 rows (up to `micro_batch × T`) | 2 | up to 12 |
| Profiled variants (RFC-0010 §6.1) | `Decode`, `Verify` and each `Batch` bucket | 1 | 2 in latency mode, up to 8 in throughput mode |
| Tail drafter (MTP or draft head) | one step, and one per tree width when trees are on | 1 | 1–2 |
| Head draft model | `Decode` and its own prefill ladder | 1 | up to 5 |

A latency-mode stage therefore holds 16 instances (18 with an MTP tail drafter), plus 4 with concurrent rows and their 2 profiled variants; a throughput-mode stage up to 34.

### 3. The step loop on each runner

#### 3.1 The primary dispatcher

The dispatcher is a pollable state machine (RFC-0060 §2.3, proposed amendment A3), so the deterministic simulator can step it and production can wrap it in the spin-then-park loop of ADR-0012:

```rust
// syntropy-exec, public surface
impl Dispatcher {
    /// One non-blocking pass: polls inbound rings, advances the step in flight, launches at most
    /// one new step program, queues outbound frames and reaps completions. Never spins and never
    /// parks; the caller decides how to wait.
    pub fn poll_once(&mut self, now: MonoTime) -> PollOutcome;
}
pub enum PollOutcome {
    Progress,
    /// A step is in flight and its next doorbell has not fired. `expect` is the plan's predicted
    /// time of that doorbell (from `StageCost` segment terms); `wake` says how this backend can
    /// wake a parked waiter for it (§3.3).
    Waiting { expect: MonoTime, wake: WakeSource },
    /// Nothing in flight: only ring events and timers can create work.
    Idle { next_wake: Option<MonoTime> },
    Fault(Error),
}

impl ComputePool {
    /// Executes one doorbell's CPU work item on the calling thread. Production pool threads run
    /// the same items as they spin (§3.2); the simulator calls this when its virtual timeline
    /// reaches the item, so one OS thread can run every node. Work and outputs are identical.
    pub fn run_inline(&mut self, work: &PoolWork) -> Result<(), Error>;
}
```

```text
fn poll_once(now) -> PollOutcome:
    progress = false

    // 1. Control and inbound, never blocking.
    drain control ring          // ApplyPlanDelta, Drain, Cancel, MemoryPressure relayed from the net domain
    drain sequence ring         // head: SequenceOpen, SequenceControl (RFC-0001 §5.8)
    drain frame rings           // InFrame (activations) or, at the head, TokenFrame (ring return)
    sched.ingest(...)           // commits or truncates KV for returned tokens; updates sequence state

    // 2. Advance the step in flight (a small state machine: Launched → Serving → Done → Sending).
    if let Some(st) = in_flight:
        for each pending doorbell of st, in layer order (§3.2):
            if its flag reads 1: reset it to 0 (§3.2), then do its host work (assign, copy_batch,
                enqueue PoolWork); progress = true
            else: break                             // later doorbells cannot fire before earlier ones
        if st.state == Serving and O[st.parity] reads 1:   // step done; consumed when its parity is reused
            if tail: sched.on_logits_ready(st) → TokenFrames to head and ingress (§8, §9)
            st.state = Sending
        if st.state == Sending:
            match channel.try_send(OutFrame { payload: send_buffer[st.parity] }):
                Ok  => telemetry.push(StepRecord); housekeeping.push(stats); in_flight = None; progress = true
                WouldBlock => credit_stall accrues; the step stays in flight (RFC-0001 §8.4 rule 1)

    // 3. Start the next step if the stage is free (latency mode: no step in flight;
    //    throughput mode: the other descriptor parity is free). Parity p is free when DC[p] and
    //    O[p] read 1: its previous step is done and its descriptor was copied (both start at 1).
    if stage_free():
        if let Some(step) = sched.next_step(now):  // head: compose (§4–§6, §9, §10); others: next InFrame
            p = parity(step); reset DC[p] and O[p] to 0   // consumed before p is resubmitted (RFC-0040 §4.2)
            kv.append(step.rows)                    // provisional positions for verify rows
            write StepDesc[p]                       // rows, positions, tokens, tree parents, page tables,
                                                    // sampler states, moe_epoch, frame-in address, tap rows
            if head: issue RowLookup reads not yet resident   // demand reads first (RFC-0021)
            replay this device's graph: provider.replay(graph[step.bucket][p]); eager launches
                for capture breaks
            post Launch { step, p } to the ring of every other device dispatcher of the stage;
                each replays its own graph on its next poll (ADR-0012: one submitter per device)
            in_flight = step (state Serving); progress = true

    if a sticky device error or a pool failure was observed: return Fault(error)    // §13
    if progress: return Progress
    if in_flight is Some(st):                       // never Idle while a step is in flight
        return Waiting { expect: st.next_doorbell_predicted, wake: st.next_doorbell_wake }
    return Idle { next_wake: min(predicted next frame − sched.prewake_us,             // §3.3
                                 earliest admission-queue deadline, earliest controller timer) }

production thread:
    loop {
        match dispatcher.poll_once(now()) {
            Progress                 => continue,
            Waiting { expect, wake } => if expect − now() > exec.inflight_park_threshold
                                            and wake is a device-triggered source:
                                            arm wake; park until it fires or expect − margin    // §3.3
                                        else: spin with pause back-off (no parking)
            Idle { next_wake }       => spin up to sched.spin_us, then park until next_wake or a ring event,
            Fault(e)                 => report e over IPC and exit with its class,
        }
    }
```

Rules:
- **One step in flight per stage in latency mode.** A stage has at most one step executing; the head may prepare the next descriptor (other parity) while one runs, which matters only in throughput mode, where consecutive steps belong to different groups.
- **Order per sequence, precedence per lane.** A non-head stage executes each sequence's frames in `step` order, which its lane's FIFO channel already delivers. Frames of different sequences may run out of global step order, because the stage takes a ready decode-lane frame before a replay frame, and a replay frame before a prefill-lane frame (RFC-0032 §3.2, ADR-0106). A frame for an epoch the runner does not serve is dropped (RFC-0001 invariant 7).
- **The activation input.** The receive buffer is registered pinned memory (RFC-0001 §9.4). Its device-visible address goes into the descriptor, and the captured graph's first node copies it into the stage's fixed input buffer (a 16–224 KiB device read at PCIe rate for decode and verify frames). Prefill frames (MiB-scale) are copied by a `CopyIn` submission the dispatcher issues before the replay, with a device-side event wait. On `Unified` devices the receive buffer is imported and read in place.
- **The activation output.** The graph's last node writes the hidden state into a registered send buffer (host-mapped) and raises `O[p]`. RDMA transports send from that buffer without a copy; IP transports hand it to the net domain's socket path (RFC-0031).
- **Plan deltas** are applied between steps: the dispatcher finishes the current step, replaces budgets, slot tables and limits (RFC-0020 §12), and only then composes the next step.

#### 3.2 Doorbells inside a step

Doorbell flags use RFC-0040 §4.2's constant-valued, consumer-reset handshake, which RFC-0040 owns (ADR-0404 decision 6). Each flag has one producer and one consumer. The producer only ever writes 1; the consumer waits for 1, writes 0, and only then replies or produces anything that can lead to the flag's next set. Every flag is raised exactly once per step, even when its work is empty, so no captured wait depends on the step number: captured waits are `EQ 1` against a constant. Step numbers travel only as payload checks that the host asserts (`DoorbellLayer::seq`, the attention doorbell's `step`). Spin and store kernels that poll a flag are only the fallback where a device has no native wait-value operations, and they run the same handshake.

| Flag (producer → consumer) | Who waits | What the consumer does after its reset | Mechanism |
|---|---|---|---|
| `Rd[l]` (GPU → dispatcher): route written, MoE layer l | the dispatcher, on each `poll_once` | runs `assign` (RFC-0040 §3.4) on the route block; issues `copy_batch` for `COPY` experts on `CopyIn` followed by `signal_host(F[l])`, or stores `F[l] = 1` itself when there are none; pushes routing statistics and the layer l+1 prediction to RFC-0023 | host-mapped flags; acquire loads |
| `Rc[l][d]` (GPU → domain d's CPU pool): route written | the lead thread of domain d's pool, spinning with `pause` back-off | runs the same `assign`, enqueues NVMe demand reads for its `CPU` experts not in RAM (RFC-0021), computes them in place, writes its outputs and stores `Q[l][d] = 1`, also when it has no `CPU` experts | same; in the simulator the pool's work runs through `ComputePool::run_inline` |
| `F[l]`, `Q[l][d]` (host → GPU) | the GPU, device-side | `wait_host` for 1 and the reset to 0, before `MoeGrouped` over copied experts (`F[l]`) and before `MoeCombine` (every `Q[l][d]`) | captured wait-value nodes (RFC-0040 §4.2) |
| `Ra[l][d]` (GPU → domain d's pool), `Qa[l][d]` (pool → GPU): CPU attention over host KV pages | the pool's lead thread; the GPU | the pool computes its `AttnPartial` over host pages and stores `Qa[l][d] = 1`; `AttnMerge` waits on every `Qa[l][d]` (RFC-0040 §3.6, RFC-0022 §5.3) | same |
| `H[i]` (GPU → CPU pool), `H[i+1]` (CPU pool → GPU): hand-off around a RAM-homed dense segment | the pool's lead thread; the GPU | the pool runs the segment, writes the activation to host-mapped memory and stores `H[i+1] = 1`, which the GPU waits on and resets | same |
| `DC[p]`, `O[p]` (GPU → dispatcher), one pair per descriptor parity p | the dispatcher | `DC[p]`: the graph's first node has copied `StepDesc[p]`; `O[p]`: the step is done (§3.1 step 2). Both start at 1, and the dispatcher consumes both from p's previous use before it resubmits p (§3.1 step 3) | same |

Every flag is re-armed after each capture and after any abandoned step: all set to 0, with `DC[p]` and `O[p]` at 1 (RFC-0040 §4.2). A producer that skips a set leaves its flag at 0, so the step stalls and the stall watchdog reports it (§13); a stale value never satisfies a wait. Each flag sits on its own 64-byte line, and a host store to a flag the device waits on is a release store with nothing else on that line.

The dispatcher issues at most one asynchronous submission per MoE layer per step (the copy batch), and only when the layer has copied misses. None of these calls waits for the device. The host never calls a stream or device synchronise inside a step (RFC-0001 invariant 9).

A stage whose layers span several GPUs of one node in `Local` mode launches every device's graph at step start; each later device's first node waits device-side on the hand-off flag the earlier one writes, so the host is not on the hand-off path.

#### 3.3 Waiting inside a step, idling between steps

A hot thread is in one of two waiting situations, and they need different mechanisms.

**Inside a step** (`Waiting`). The events that move a step forward are doorbell flags the device writes into host-mapped memory (§3.2). A store from the device wakes no parked thread, so a dispatcher or CPU-pool thread that parked on a ring event would sleep through it. The rule is therefore: **while a step is in flight, neither the dispatcher nor the pool's lead thread parks on anything except a device-triggered wake source armed for the next doorbell.** They spin with `pause` back-off by default. Spinning between doorbells is long in some plans: on scenario (b2) the routed work of 126–205 ms over 58 MoE layers puts 2.2–3.5 ms between route doorbells, and on scenario (a) the GPU segment runs about 24 ms before the CPU pool's hand-off flag fires. When the predicted gap exceeds `exec.inflight_park_threshold` (default 2 ms on mains power, 0.5 ms on battery [target]), the thread arms the backend's device-triggered wake (the provider's `arm_wake`, RFC-0040 §4.2), parks, and resumes spinning `exec.inflight_wake_margin` (default 200 µs) before the predicted doorbell time if the wake has not fired:

| Backend | Device-triggered wake for a doorbell | Where it is placed |
|---|---|---|
| CUDA, HIP | a host function (`cuLaunchHostFunc`/`hipLaunchHostFunc`, a host node in the captured graph) whose callback signals the waiter's eventfd, futex or Windows event | on a forked side branch after the doorbell write, so the compute stream does not wait for the callback |
| Metal | `MTLSharedEvent` signalled after the doorbell's encoder; an `MTLSharedEventListener` `notifyListener:atValue:block:` block signals the waiter | in the step's command buffer |
| Vulkan | a timeline-semaphore signal after the doorbell write in the same submission; the parked thread waits with `vkWaitSemaphores` on that value, with a timeout at the predicted doorbell time | in the step's submission |
| CPU (pool → dispatcher, or dispatcher → pool) | the writer does its release store, then a futex or event wake if the waiter announced it is parking | — |

A device-triggered wake adds the driver's callback latency plus an OS wake, which is why spinning stays the default below the threshold. That latency is not measured yet; RFC-0010's H2 step gains a wake variant that times each path per device (`DeviceCapsExt::wake`), and the threshold is set from it [the defaults above are placeholders until measured]. The plan only arms wakes for doorbells whose predicted gap exceeds the threshold, so on fast stages there are no host nodes on the token path at all. The CPU pool's worker threads wait on the lead thread's barrier with a futex, which a CPU store can wake. A hardware CI test runs a stage with a 10 ms CPU segment and checks that no step waits for a timer or the stall watchdog.

**Between steps** (`Idle`). A dispatcher spins for `sched.spin_us` (default 50 µs, 10 µs on battery; ADR-0012 item 5), then parks on the inbound ring's wait primitive. Waking a parked thread is on the token's critical path at every stage. In latency mode a stage's next frame arrives about one `t_tok` after its previous one, so the dispatcher sets a timed wake-up at `predicted arrival − sched.prewake_us` (default 300 µs) and spins from there, using the plan's per-stage prediction refined by the last 16 arrivals. A frame that arrives while parked still wakes it through the ring's event. All times come from the dispatcher's `TickSource` (RFC-0060 §2.1): a statically dispatched generic parameter, monomorphised to the TSC or OS monotonic clock in production and to the simulator's device timelines under test; `poll_once`'s `now` is read from it. Spin time is reported, so the power cost is visible (ADR-0012).

### 4. Chunked pipelined prefill

#### 4.1 Chunking at the head

On `SequenceOpen`, the head looks up the longest cached prefix (prefix cache or a restored session, RFC-0022 §7.3, §8.4) and splits the uncached suffix of `u` tokens into chunks of `c` tokens, with c from `RouteDecision::prefill_chunk` (RFC-0011 §13): the plan's chunk, or for short prompts the smallest captured ladder size ≥ ⌈u / D⌉, so every stage works on the prompt at once. The last chunk may be shorter and uses the smallest bucket that holds it. Each chunk is one step with `FrameKind::Prefill { chunk: k, last }`; the first carries the `SequenceOpen` metadata on its side channel (RFC-0001 §5.8).

#### 4.2 Pipelining

The head issues chunk k+1 as soon as chunk k's frame is queued, subject to channel credit. Every stage starts chunk k+1 as soon as it has sent chunk k, so stage s runs chunk k while stage s+1 runs chunk k−1, and with m chunks over D stages the bubble fraction is (D−1)/(m+D−1) (cost model §3.7):

```text
TTFT(u) ≈ Σ_s t_s(c) + (m − 1) · max_s t_s(c) + Σ_hops (α + c · S_tok / β) + t_logits + t_sample
```

which is the expression the router uses (RFC-0011 §13) and SC-18 bounds. Each stage appends KV for the chunk's positions as it runs it. On the last chunk the tail computes logits for the final position only, samples the first token and sends it to ingress and to the head.

#### 4.3 Interleaving with decode in latency mode

A chunk step occupies a stage for `t_s(c)`, so a decode step that arrives behind it waits up to one chunk per stage. The head therefore:
1. always schedules a ready decode or verify step of an active interactive sequence before any prefill chunk;
2. issues prefill chunks for other sequences only while no such step is ready, and caps their chunk size at the largest captured c with `max_s t_s(c) ≤ sched.decode_stall_budget` (default 0.5 × the active sequence's predicted `t_tok`, floor 64 tokens) whenever another sequence is decoding;
3. sends prefill chunks on the **prefill lane** and decode and verify steps on the **decode lane**: separate channels with separate credits, and every stage takes decode-lane frames before prefill-lane frames at each step boundary (RFC-0032 §3.2). Outstanding-chunk policy per lane: on the decode lane, one step in flight per sequence; on the prefill lane, up to `sched.prefill_inflight` chunks per instance (default D, enough to keep every stage busy);
4. uses the full plan chunk, pipelined as in §4.2, when nothing else is decoding.

Without the lanes, the cap would not bound the stall: the head pumps chunks as fast as credit allows, and in one FIFO channel they would queue ahead of the decode frame at the slowest stage, which never reorders. With RFC-0031's 3–4 activation credits that is up to 4 chunks there, or 2 · `t_tok` at the 0.5 · `t_tok` cap. With the prefill lane, a decode frame waits at most for the **one running chunk** at each stage and never behind queued ones, so the slowdown is at most one running, capped chunk per stage per token (goal 3). Prefill keeps its pipelining, because queued chunks no longer block decode. The stall budget is a chosen default for RFC-0061 to tune.

#### 4.4 Execute in place versus streamed weights

A stage runs the `PrefillStream` program when the chunk reaches a `HostPinned` group's `StreamSite::min_tokens` (n* from the probed curves, RFC-0011 §11.2), and the `Prefill` program otherwise. For dense groups the comparison uses the chunk size c. For expert groups it uses the tokens each expert sees, `c · k / E` per expert on average (`c / 32` on DeepSeek-V3): a streamed expert's copy hides only when that count reaches n*, so tiered MoE prefill is copy-bound per chunk at any practical chunk size. On such stages a chunk of ≥ 256 tokens touches essentially every expert (cost model §3.5, §3.9), so each chunk streams the RAM-resident pool through PCIe and re-reads the NVMe-resident share at `W_nvme`; the solver therefore picks the largest chunk activations allow for NVMe-homed pools (RFC-0011 §11.2). The CPU in-place path is used for short prompts where `c < n*`.

**Layer-major prefill groups for NVMe-homed pools**. Re-reading the NVMe-resident pool once per chunk is the dominant prefill cost on scenario (b2): ≈ 281 GB per chunk at ≈ 6 GB/s is ≈ 47 s [estimate: 394 GB of routed experts less RFC-0011 §15.2's ≈ 104 GB RAM and ≈ 9.5 GB VRAM caches]. A stage whose routed pool is `Nvme`-homed therefore runs prefill in **layer-major groups**. It accumulates up to G consecutive chunks of a prompt, with G the largest number whose activations fit the stage's scratch (`G · c · h_model · streams · 4 B` within `plan.scratch`; a whole 32 K prompt at width 7168 is ≈ 0.94 GB [estimate]). It then runs the group layer by layer: for each MoE layer it routes every row of the group, reads the layer's needed experts once (prefetching the next layer's from NVMe while computing), and computes all the group's rows before moving on. The pool is read once per group instead of once per chunk. The cost is latency to the next stage: the group's first chunk leaves the stage only when the whole group is done, so cross-stage pipelining runs at group granularity. RFC-0011 §11.2 costs this per group with `t_pool`. Spike PLC-E8-T15 measures it on the (b2) rig and sets the default for when to use groups (by prompt length) and the group-size cap.

#### 4.5 Prefill hops

A prefill frame carries `c · h_model · streams · 4 B` in exact mode: 64 MiB for a 2048-token chunk at width 8192. The plan's chunk rule makes each hop's transfer shorter than the next stage's compute for that chunk (RFC-0011 §11.2 rule b), so hops hide behind compute except for the first chunk's. RDMA receives land in the stage's registered pool without a copy; IP transports pay one host copy (RFC-0001 §9.4).

#### 4.6 Concurrent decode rows in latency mode

Agents fan out: parallel sub-agents, tool calls, and the title or suggestion request many chat front-ends send after each message. With one decoding sequence at a time they would run serially until throughput mode at M3. From M1, latency mode therefore lets up to `sched.latency_decode_rows` (default 4) sequences decode in one step:

- **Where.** On instances whose stages are all dense, or whose MoE routed pools are homed in VRAM, RAM or `Unified` memory (no stage with an `Nvme`-homed pool), and only up to the plan's certified `max_concurrency` (§10.5).
- **When KV fits.** The plan still reserves KV for `max_sequences` (RFC-0054; latency default 2) at its `context_length`. A further sequence is admitted as a concurrent decoder only if its footprint fits every stage's current headroom (§10.1); it is never given reserved KV, and it is the first paused if the active sequences need pages (§10.3).
- **How.** The head composes one step from every concurrent sequence's decode row, in the `Batch { rows }` bucket at 2 or 4 rows. Memory-bound ops read the weights once for all rows, so the main stream pays little: RFC-0040 §9 row 2 targets `time(rows) ≤ 1.10 × time(rows = 1)` for 2–8 rows on dense stages. With speculation, each sequence keeps its own window and the step's rows are the sum of the windows, bounded by the largest captured bucket; the controller shrinks windows to fit, as throughput-mode groups do (§6.1). Prefill of a new sequence still interleaves under §4.3's cap. Each sequence keeps its own RNG counter, count tables and grammar state, so its output does not depend on which others share its steps (exact mode, ADR-0019).
- **NVMe-tiered MoE stays serial.** On an instance with an `Nvme`-homed pool, extra rows multiply the experts touched (cost model §3.9), and a second prompt's prefill would re-read the pool on the drive the active decode's demand misses need. A second request there waits in the head's admission queue with its position reported (`Admission::Queued { position }`, RFC-0011 §13), and neither prefills nor decodes until the active sequence ends or pauses. The API's response metadata says why (RFC-0050).

### 5. Single-stream decode around the ring

One decode step per token, or per verify window (§9). The critical path of one token is:

```text
t_tok = t_head_in (embedding rows)
      + Σ_s t_s(1)                     each stage's step: dense, KV, MoE three-site work, hand-offs, t_fixed
      + Σ_hops (α + S/β)               activation frames, and the tail → head ring return
      + t_logits + t_sample (tail)
```

and it is what the plan predicts per term (RFC-0011 `StageCost`). Mechanics at each position:

- **Tail.** After the sampler writes the token to its host-mapped block and signals O, the dispatcher sends one `TokenFrame` to the head (ring return) and one to ingress, from the same buffer. Delivery to ingress is therefore off the next token's critical path (RFC-0001 §6.2).
- **Head.** On `TokenFrame` arrival the head commits the sequence's KV, writes the token into the descriptor and starts the step. `RowLookup` rows addressed by the new token (token embeddings, n-gram and per-layer embedding tables) are read at once: resident tables are a gather inside the graph; NVMe-homed tables need a demand read of a few KiB, which costs one NVMe QD1 latency (≈45 µs at 4 KiB on a Gen4 drive, cost model §2.4) and is on the critical path. The head issues those reads before writing the descriptor, and a verify window batches the reads for all its positions.
- **Middle stages** wake on frame arrival (§3.3), run the step, send.
- **Expert prefetch** for the next token starts during the current one: next-layer prediction (RFC-0023) issues reads into the plan's prefetch windows inside the step, so they overlap the rest of the ring traversal.

With one sequence, each stage is busy `t_s / t_tok` of the time; the pipeline adds capacity and bandwidth, not single-stream parallelism (cost model §3.7). Speculative windows (§9) are how a single stream uses the idle time.

#### 5.1 Replay steps

Recovery after a stage loss rebuilds KV by replaying activations (RFC-0033 §4.2). Replay steps differ from every other step in four ways, and the step loop handles them explicitly:

- **They can start mid-ring.** The replay source may be the head, recomputing from the token transcript, or a stage u > 0 sending rows from its own outbound history. A stage that is the source composes Replay steps for the affected sequences itself, from its history, in prefill-ladder-sized frames; this is the one case where a stage other than the head composes steps (ADR-0106 is amended to say so).
- **They use their own channel and step class.** Replay frames arrive on `Replay`-purpose channels (RFC-0031) with `FrameKind::Replay`. A stage keeps the steps of each lane in order and serves lanes by precedence **decode lane > replay > prefill lane** (RFC-0033 §4.2), so replay never reorders live steps and never delays a waiting decode step. A sequence that has finished replaying resumes decode on the Activations channel while others still replay (RFC-0033 §4.2).
- **They write KV selectively.** Each replay row carries `kv_write_from = h_k`, the stage's committed KV position for that sequence from the handshake's REPORT. Attention reads existing KV for earlier positions, and KV is written only for positions `p ≥ h_k`. Rows below `drop_below_k` are not computed at all. `kv_write_from` is a per-row field of `StepDesc` (RFC-0040 owns the layout).
- **They end at the tail without sampling.** The tail computes its layers for the replayed positions, discards the logits (no sampling, no `rng_counter` change), and on the last chunk records completion for the handshake's DONE. Every stage that computes replay rows appends its outbound activations to its own history, refilling it.

The handshake messages themselves (QUERY, PLAN, DONE `Recover` entries) are not steps: the dispatcher hands them, unchanged, to the recovery core (`HandshakeCore`, RFC-0033) through its control ring, and acts on the PLAN entries it returns. Replay steps use the prefill buckets, so they need no extra graphs.


### 6. Batched and throughput mode

Throughput mode is the plan's `Objective::Throughput` (D2's second mode). The plan fixes `micro_batch` (b_μ) and `max_sequences` (RFC-0011 §7.2); this section fills the pipeline.

#### 6.1 Groups circulating the ring

The head keeps `G = D` **groups** (D = number of stages), each holding up to b_μ decoding sequences. A group's step is issued when every sequence in it has its token back from the tail; the next group's step follows as soon as the head has sent the previous one. In steady state the D groups occupy the D stages, each stage runs one group per step, and the bubble disappears (cost model §3.7):

```text
throughput ≈ b_μ / θ,   θ = max(max_s t_s(b_μ), max_hops t_hop(b_μ))
per-sequence rate ≈ 1 / (D · θ)          ≥ per_sequence_floor_tps by the plan's C12
```

`sched.next_step()` at the head, in throughput mode:

```text
ready = groups whose sequences have all returned (or whose late members are paused, §10.3)
if a prefill chunk is pending and the prefill share allows (one prefill step per G decode steps
   while groups are waiting; unlimited when no group is ready):
    return the next chunk step (§4)
if ready is non-empty:
    g = the ready group with the oldest return time (round-robin under equal priority)
    return Step { bucket: Batch { rows: Σ_{seq ∈ g} window(seq) }, rows: g's sequences }
return None
```

- **Joining.** A new sequence prefills in chunk steps of its own, then joins the group with the fewest members at its next return. Groups never exceed b_μ sequences; when every group is full, admission queues (§10).
- **Leaving.** Finished, cancelled or paused sequences leave their group at the step boundary. When two groups fall below b_μ/2 each, the head merges them at their next common return, so the step count per token stays near optimal.
- **MoE union.** A batched step routes all its rows at once, so each MoE layer serves the union of their experts (RFC-0040's `assign` over all rows). The solver already priced `U(b_μ)` (RFC-0011 §7.2); the scheduler's job is only not to exceed b_μ.
- **Speculation** is off unless the plan enabled it for this b_μ (ADR-0017 item 8). When on, a group's rows are the sum of its sequences' windows, bounded by the largest `Batch` bucket.

#### 6.2 Several pipelines

Replicas of a model are separate instances with separate plans (RFC-0001 §3.2). Ingress spreads requests across them by routing score (RFC-0011 §13); there is no cross-instance scheduling inside runners.

#### 6.3 Priorities

`SequenceOpen::priority` has three classes: `Interactive`, `Normal`, `Background`. In latency mode, at most one `Interactive` sequence is the "active" stream whose steps go first (§4.3); others queue behind it in arrival order. In throughput mode, groups are ordered by the highest priority they contain, and an `Interactive` arrival that cannot be admitted for lack of KV may pause the newest `Background` sequence (§10.3).

### 7. Graph capture with dynamic routing

RFC-0040 §4.4 defines the capture mechanisms and rules per backend; this section is the policy.

#### 7.1 What is captured

- **Whole steps.** For each (step program, bucket, descriptor parity) the executor captures the full device side of the step: the descriptor copy, every layer's ops, doorbell signals and device-side waits, and the write of the output into the send buffer. One replay per device per step is the target; capture breaks (an op with `capture_safe = false`) split a step into segments with eager launches between them.
- **When.** All buckets in `Warming` (`WarmStep::Capture`), before the canary, so the canary exercises the captured graphs. Capture uses CUDA's thread-local capture mode (HIP's equivalent), so another thread's host registration, which RFC-0020 §6's progressive pinning keeps doing on the housekeeping thread, does not invalidate a capture in progress. RFC-0020 §6 also finishes registration before `Warming` captures and takes its capture lock for any later registration, so the two rules are belt and braces: either alone keeps captures valid. Capture and `cuGraphInstantiate` run on a low-priority helper thread with the device's context current; neither submits work to the device, and the instantiated executable is handed to the device's dispatcher, which swaps it in between steps (ADR-0012's one-submitter rule holds).
- **Pre-sizing.** Everything a delta may change is sized in `Warming` for its upper bound in the epoch: provisional KV slots and the verify bucket for `T_max` = `SpeculativeConfig::max_window + 1`, the prefill ladder up to the plan's chunk, and `micro_batch` buckets up to `BatchingLimits::micro_batch`. A `PlanDelta` may lower these, never raise them; a raise is a new epoch (RFC-0001 §7.3). So nothing is allocated after `Warming` (invariant 9), and a delta never needs a capture on the token path.
- **Memory.** Graph memory is measured at `Warming` and must fit `plan.scratch`; a shortfall drops the `PrefillStream` bucket first (prefill falls back to in-place execution, slower and still exact), then the 64-token prefill bucket.

#### 7.2 What changes between replays without re-capture

Everything per-step lives in device-resident tables that the graph reads through the step descriptor (RFC-0040 §3.5):

| Dynamic thing | Where it lives | Writer |
|---|---|---|
| active rows, positions, token IDs, sequence slots | `StepDesc` arrays | head scheduler (rows), every stage (its copy) |
| MoE expert residency and miss split | per-layer slot tables, changed only by table deltas that the graph's first node (`ApplyTableDeltas`) applies up to `StepDesc.delta_head`, a ring position the dispatcher writes into the step parameter block when it writes the descriptor, so every site runs step s at one table version (RFC-0040 §3.4, RFC-0023 §5.4) | the device's dispatcher, which posts deltas to the table-delta ring and sets `delta_head` |
| KV pages per sequence | page tables in device memory | the stage's primary dispatcher via `KvStore` (RFC-0022) |
| tree structure of a verify window | `tree_parent[]` in the descriptor; the attention mask derives from it | head (draft model) or tail (MTP), copied into the descriptor |
| sampler parameters and RNG counters | `seq_sampler[]` in the descriptor | tail |
| grammar masks | a pinned mask buffer per sequence slot, referenced from the descriptor | tail CPU (§8.3) |
| frame input address | descriptor field | dispatcher |

A captured graph stays valid for the plan epoch (RFC-0001 §5.5): expert-cache slots and KV page pools have fixed addresses; contents change.

#### 7.3 Fallbacks

| Situation | Behaviour |
|---|---|
| Capture fails for a bucket | that bucket runs eagerly: the same prepared ops launched in order on the same streams, with the same doorbells. Output is identical; `t_fixed` grows by the launch cost, which RFC-0010 measured (`LaunchCaps::eager`), and the plan's prediction is corrected by a delta. Reported as `Ready { degraded: CaptureOff }` (RFC-0001 §10.2 item 4) |
| Provider without capture (CPU) | the dispatcher replays its prebuilt op list; that is the CPU's normal path |
| Metal and Vulkan stages with host-interacting MoE layers | §7.5: whole-step capture of host waits is not available; the step is pre-built and submitted once, and each host wait costs a measured `t_wait` |
| `config features.disable = ["graph_capture"]` | all buckets eager (diagnostic kill switch, ADR-0014 rule 4) |

#### 7.5 Backends without whole-step capture of host waits

CUDA and HIP capture device-side waits on host values inside one graph (RFC-0040 §4.4). Metal and Vulkan cannot, so on those backends "one replay per device per step" holds only for stages whose layers never wait on the host. A layer waits on the host when its MoE misses run on the CPU or come from NVMe, or when a dense segment hands off to the CPU pool. For the other stages:

| Backend | What a step is | What it costs |
|---|---|---|
| Metal | one command buffer per step: each segment is an `executeCommandsInBuffer` of a pre-encoded indirect command buffer, with `encodeWaitForEvent`/`encodeSignalEvent` on shared events between segments at each host wait (RFC-0040 §4.2). On the Metal 4 path the step's command buffers are instead committed together up front with queue-level waits between them; M5 measures both paths and the cheaper per MoE layer is used, per OS version (RFC-0040 open question 1). The structure is fixed for the epoch and per-step values live in the descriptor, so the dispatcher encodes the next step's command buffer while the current step runs or while it idles, keeping encoding off the critical path | per step: one commit, plus `t_wait,Metal` per host wait for the GPU-side event wait and resume |
| Vulkan | one `vkQueueSubmit2` per step carrying every segment's batch, chained by timeline-semaphore waits on values the host signals with `vkSignalSemaphore`; Vulkan permits waits submitted before their signals | per step: one submission, plus per host wait one `vkSignalSemaphore` from the host and `t_wait,Vk` on the device (RFC-0040's `n_submissions`) |

- **Unified stages with no CPU or NVMe miss site** drop the per-layer route doorbell from the program altogether: hits run in place, routing statistics are read from the route block after the step, and RFC-0023 updates the slot table between steps. An Apple stage whose experts all sit in `Unified` memory therefore has no host waits.
- **The solver sees the cost.** RFC-0010 adds probe step M5, a k-layer stack with one host wait per layer per backend, which yields `t_wait` per device and, on Metal, per wait path. The solver charges it per host-interacting MoE layer in place of `t_switch` on these backends (RFC-0011 §5), and may then prefer placements without CPU misses on them.
- **Scale [estimate].** A 58-layer MoE stage on Metal with NVMe or CPU misses in every layer has 58 host waits per step. At an assumed `t_wait` of 10–50 µs (unmeasured; M5 replaces it) that is 0.6–2.9 ms per step, above SC-2's ≈1.2 ms on M3 Ultra. SC-2 is therefore **provisional** for Metal and Vulkan stages with host-interacting MoE layers: its value for them is set from the M1 measurement of `t_wait` (spike PLC-E7-T20). Until then such plans predict and display their `t_fixed`, and the host-work target below is per backend.

#### 7.4 No synchronising host calls on the token path

Per step and device, the host's interactions with the device are: one replay submission per segment; for each MoE layer with copied misses, one asynchronous `copy_batch` submission and one `signal_host` enqueue; host-side stores to doorbell flags; reads of host-mapped result blocks after a flag fires. None of these blocks on device progress. `Event::wait` from the host, stream or device synchronise, and blocking memory copies are forbidden after `Warming` and caught in CI by an instrumented provider shim in `testkit` (RFC-0060).


#### 7.6 GPU segment budget (OS watchdogs)

Windows resets the display driver when a display GPU does not yield within `TdrDelay` (2 s by default), and macOS terminates a process whose command buffer runs past its GPU watchdog (about 5 s). A 4096-token prefill chunk of a 70B dense stage takes about 22 s [estimate, RFC-0040 §4.5], and Syntropy never changes those machine-wide settings (ADR-0014 rule 6). The step-program builder therefore keeps every unit of GPU work the OS times under the device's segment budget, `exec.max_gpu_segment_ms`:
- **Planning.** At `Warming`, for each bucket, the builder sums each op's `OpCaps::est` at the bucket shape (RFC-0040 §3) and splits any unit whose estimate exceeds the budget.
- **CUDA and HIP on a device that reports a run-time limit** (`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT` = 1, recorded by the probe, RFC-0010): prefill `Gemm`, `GateUpAct` and `AttnPrefill` run as row-range launches, so one prefill chunk becomes several launches inside the same captured segment, each estimated within the budget. The bucket set and the graph count (§2) do not change.
- **Metal:** a prefill step whose estimate exceeds the budget is encoded as several command buffers, one per layer group, each within the budget and committed in order with the step's event values. Decode and verify steps stay one command buffer per segment (§7.5).
- **Vulkan:** the same rule per submission (M3), since the Windows watchdog applies whatever the API.
- **No budget:** compute-only GPUs (no display attached, run-time limit 0) and the CPU.
- **In-step host waits** (spin kernels, device-side waits on host flags) are bounded by the same budget: past it the step fails with `DoorbellTimeout` and the runner abandons it before the OS acts (RFC-0040 §4.5 item 2).

If live timings show a split unit running past the budget, the runner reports it as drift, and the next `Warming` re-splits that unit using the measured time in place of `OpCaps::est` (RFC-0010 §6).

### 8. Sampling at the tail

#### 8.1 Pipeline

At the tail, after the last layer, the step program runs on the device that holds the `Head` group (RFC-0011 rounding keeps it on the tail's GPU run):

| # | Op (RFC-0040 §3.1) | Rows | Notes |
|---|---|---|---|
| 1 | `Logits` | rows flagged in the descriptor: one per decoding sequence, every position of a verify window, the final position of a last prefill chunk | final norm + LM head; F32 logits |
| 2 | `LogitsProcess` | same | temperature; penalties from two per-sequence count tables in device memory: `emitted` (presence and frequency penalties) and `context` (repetition penalty, prompt plus emitted tokens, initialised from the prompt at `SequenceOpen`); the counting basis is §8.6's; logit bias; grammar mask (§8.3) |
| 3a | `Argmax` | greedy sequences | — |
| 3b | `TopKTopP`, `SampleSeeded` | sampled sequences, plain decode | counter-based RNG (§8.2) |
| 3c | `VerifyAccept` | verify windows | exact acceptance (§9.5) |
| 4 | log-softmax and top-n | sequences that asked for logprobs | — |
| 5 | write the result block, `signal_host(O[p])` | — | accepted tokens, count, first position, logprobs, finish flags |

Both count tables are updated by the same op that commits accepted tokens, so penalties at the next step see them without a host round trip. A fresh sequence and a resumed one (§8.6) build them from the same tokens, so their penalties agree. The whole pipeline is inside the captured graph.

#### 8.2 Random numbers

Sampling uses a counter-based generator (Philox4x32-10 class) keyed by `(generated_seed, SequenceId)`, or by `(client_seed, choice_index)` when the request supplies a seed, and indexed by `rng_counter`, a per-sequence draw counter (ADR-0107):
- **Key.** Ingress always sets `SamplingParams::seed` (the client's, or 64 random bits it records in the transcript). Each choice of an `n > 1` request differs in its key's second component (its own `SequenceId`, or its choice index), so choices draw different numbers. With a client-supplied seed the key uses the choice index, so a repeated request reproduces its tokens on the same build, epoch and devices (ADR-0019 item 8; ADR-0107).
- **Counter.** `rng_counter: u64` (everywhere it is carried) starts at 0 and increases by one per draw. Draw `c` is `philox(key, c)`. The counter, not the position, indexes draws, because a verify window can consume several draws per position.
- **Draw order in a step**, per sequence: plain sampled decode takes one draw. A verify window takes, in the order §9.5's rules try candidates, one uniform per candidate tried, then one draw for the residual or bonus token. Greedy sequences take none.
- **State.** The descriptor carries each sequence's key and its counter at step start; `SampleSeeded` and `VerifyAccept` advance it in place and write the final value to the result block. The tail reports it in every `TokenEntry` (RFC-0032 §3.7); ingress keeps the latest value with the transcript; a recovered or migrated sequence resumes from it (RFC-0033 §4.5, §8.6 here).

Speculative and plain decoding therefore consume different numbers of draws and agree in distribution, not token for token, for sampled requests; greedy requests take no draws and agree exactly on the reference executor.

#### 8.3 Grammar masks

Grammar-constrained sequences carry a compiled constraint (RFC-0050 §7). The tail owns the automaton state. After each step the sampler's helper core advances it with the accepted tokens and computes the next position's allowed-token bitset into the sequence's pinned mask buffer, and marks it ready. Each sampling tail step has one mask flag `M[p]` for its descriptor parity p, under RFC-0040 §4.2's handshake, with the helper core as its one producer: the helper writes 1 once every constrained row of the step has its mask ready, and at once when the step has none, so it is raised exactly once per step. The step number travels only as a payload check in the mask buffer. That work runs while the ring executes the next step, so it is off the critical path whenever it takes less than one ring traversal minus the tail's own step; RFC-0050 budgets ≤ min(0.5 × ring step, 1 ms). `LogitsProcess` waits device-side for `M[p] = 1` and resets it before applying the mask, so a slow mask delays only the step that carries its sequence, and never produces an unmasked sample.

For verify windows, masks are needed for every drafted position. MTP and draft heads draft at the tail, which owns the automaton, so the tail computes masks along the draft as it builds it, and drafts that violate the grammar are not proposed. A draft model drafts at the head, which has no grammar state; its drafts may be rejected by the masked target, which costs acceptance but never exactness.

#### 8.4 Stops and limits

The tail checks token-level stop sets (EOS, the template's end-of-turn tokens; RFC-0050) and `max_tokens` on every accepted token and sets `TokenEntry::finish`. Tokens after a stop in the same verify window are discarded at the tail, never emitted. Stop strings, which can span tokens, are matched at ingress, which then sends `Cancel` (RFC-0001 §6.2).

#### 8.5 Parity taps

The parity harness measures a real instance by teacher forcing (RFC-0060 §5.5). It opens a sequence with `SequenceOpen::parity: Option<ParityTapSpec>` and reads results from `TokenEntry::tap: Option<LogitTap>` (RFC-0060 proposed amendment A4). RFC-0060 defines the types; this section defines what the scheduler and the tail do with them.

- **Entry.** Only through `POST /v1/syntropy/parity`, which answers when the node key `debug.parity_api` is true and only on loopback binds (RFC-0060 §5.5). The head admits a parity sequence like any other `Normal` sequence; it may share steps and groups with service traffic, whose rows it never changes.
- **Teacher tokens.** The `TeacherStreamId` names corpus tokens delivered to the tail (and to the head when a draft model drafts there). In every `ForceMode`, the token the tail sends around the ring is the teacher token, so each stage sees exactly the prefill, decode or verify traffic it would see in service:
  - `Prefill`: positions `[0, n)` as chunked prefill at the plan's chunk size (§4); every position of every chunk is a tap row.
  - `Decode { prefill_len }`: prefill `[0, p)`, then one decode step per position, with the teacher token replacing sampling.
  - `Verify { prefill_len, gamma, accept, seed }`: windows of γ+1 positions. A synthetic drafter at the drafting site proposes, at each position, the teacher token with probability `accept` and a different token otherwise; the Bernoulli draws come from a separate counter-based stream keyed by the tap's `seed` and the position, so the sampling stream and its `rng_counter` are untouched. Acceptance is forced: a draft is accepted exactly when it equals the teacher token, and the bonus is the teacher token at the first non-accepted position. Windows therefore exercise full acceptance, partial acceptance and KV truncation (RFC-0001 §6.4) with a fixed pattern.
- **What is tapped.** The model's own distribution: log-softmax of the `Logits` output before `LogitsProcess`, so temperature, penalties, bias and grammar masks never enter a parity measurement. Every row of a verify window is tapped.
- **`TapMode::FullLocal`.** F32 logits of each tapped row go into a shared-memory region the driver maps; this works only when the driver runs on the tail's host. `LogitTap` carries the row offsets.
- **`TapMode::Gather { k }`.** The driver streams the reference top-K ids for upcoming positions to the tail over a data-plane channel of purpose `Tap` (RFC-0031 owns the layout). The tail places them in a pinned id buffer that the descriptor references; a gather op after `Logits` writes, per tapped position, the candidate's log-probabilities at those ids, its logsumexp, its argmax and its top-2 margin into the result block. If a position's ids have not arrived, that step waits device-side on a tap flag, the same mechanism as grammar masks (§8.3); parity runs are not latency-sensitive.
- **Logit taps need no extra graphs.** Logit-tap work is controlled by descriptor fields (`rows_tap`, the id buffer address), so the tail's existing captured programs serve both service and parity steps; with no tapped rows the tap ops have zero active rows.
- **Layer taps use a tap variant, built on request**. Hidden-state taps at stage boundaries and at sampled layers, which RFC-0060 §5.6's layer-local parity needs for models too large for a full reference, are available in release builds behind the same `debug.parity_api` gate. When a parity sequence asks for layer taps, each affected stage builds a tap-enabled variant of its step programs (the same ops plus copies of the tapped hidden states into a host-mapped tap buffer) and captures it in an idle window; tapped steps replay the variant and every other step the normal program. Its cost is measured and reported with the parity result, and it is never built unless asked for. The tap scratch (logits for a whole prefill chunk at vocabulary width) is reserved at `Warming` only when `debug.parity_api` is set, plus a small fixed region for the warm-up canary, which uses the same mechanism on a short fixed sequence (RFC-0060 A4).


#### 8.6 Tail state on resume

The tail holds four pieces of per-sequence state that a replacement or migrated tail must rebuild before the sequence's first resumed step (RFC-0033 §4.3, §4.5). Ingress holds the transcript (prompt tokens, emitted tokens) and the last `rng_counter`, and passes them with the resume point:

| State | Rebuilt from | Rule |
|---|---|---|
| Count tables for penalties (§8.1) | the transcript | presence and frequency penalties count **emitted tokens only** (`emitted`); the repetition penalty counts **prompt plus emitted tokens** (`context`). This RFC defines that basis, and the fresh path in §8.1 uses the same one. Both tables are rebuilt on the host and copied to the sequence's device slot before the first step |
| Grammar automaton | the transcript | advance the compiled constraint over the emitted tokens, then compute the mask for position `need` (§8.3). `advance` is deterministic and needs no mask, so the replay is exact on a replacement tail (OQ-NET-27, answered 2026-10-20; ADR-0504 decisions 3 and 6; PLC-E9-T18) |
| RNG | the key, `(generated_seed, SequenceId)` or, when the client supplied the seed, `(client_seed, choice_index)`, both kept in the transcript (RFC-0032 §8.2), and the reported `rng_counter` (u64) | resume drawing at that counter (§8.2) |
| Window controller | the plan's acceptance prior | re-seeded from the prior (§9.3); the old posterior is not carried |

With these rebuilt, a resumed sampled sequence draws from the same distribution it would have drawn from without the failure, so RFC-0033 §4.3's exact-continuation claim holds for sequences with penalties and grammars. A parity test with penalties and a grammar kills the tail mid-sequence and checks the resumed distribution (Testing).

### 9. Speculative decoding

#### 9.1 Drafters and where they run

| `DraftSource` (RFC-0001 §5.4) | Runs on | Input | Draft step |
|---|---|---|---|
| `Mtp { heads }` | tail, on the device with the LM head | the target's final hidden state at the last accepted position and the embedding row of the accepted token (the tail holds a duplicate of the embedding rows, RFC-0011 C6) | one MTP module pass per drafted token, chained |
| `DraftHead { group }` | tail | same | one draft-head pass per drafted token. **v1 produces no draft heads**: nothing in the plan trains, evaluates, licenses or publishes them. The loader stays, so a head supplied with a model's own files runs here once its family passes the conformance gate, and community heads can follow later under the conformance and benchmark gate |
| `DraftModel { model, device }` | head, on its fastest device | accepted tokens | one draft-model decode step per drafted token, with its own KV |
| prompt lookup (supplementary) | wherever drafting runs | the sequence's tokens | match the last `n` tokens (default 3) against the context; propose the continuation |

Preference follows ADR-0017 item 1 and the plan's `SpeculativeConfig::source`. Without a draft head, speculation uses the model's MTP heads where the family has them, then a draft model on the head, with prompt lookup merged into either; the solver's outer loop (RFC-0011 §7.3) scans only the drafters that exist. RFC-0054's `speculative.mode = off` on a request disables drafting for that sequence only.

#### 9.2 Window budget

The plan fixes `SpeculativeConfig::max_window` = γ_max (RFC-0001 §5.4), chosen by the solver's outer loop (RFC-0011 §7.3); a verify window holds at most `T_max = max_window + 1` positions. Captured verify graphs and KV provisional slots are sized for it (RFC-0022 §6), so the runtime never exceeds it. Within the budget the scheduler picks each sequence's window per step.

#### 9.3 Per-sequence window controller

```text
state per sequence: â (acceptance estimate), â_depth[i] for trees, wins/losses counters
prior: Beta(α0, β0) with mean = the plan's learned acceptance for (drafter, greedy or sampled), weight 16 windows (§11)
first window: γ = SequenceOpen::initial_window (RFC-0011 §13's RouteDecision), capped at γ_max
after each window with j accepted of γ drafted: update the Beta posterior with j successes and
    (j < γ ? 1 : 0) failures (geometric-acceptance model)

choose γ for the next window:
    γ* = argmax over γ in 0 ..= γ_max − (tree reserve) of
            E(â, γ) / (t_draft(γ) + t_pass(γ + 1))
    E(a, γ) = (1 − a^(γ+1)) / (1 − a)
    t_pass(T): the plan's PredictedCost::pass_by_window[T] (RFC-0011), scaled by this instance's
               measured ratio of actual to predicted step time over the last 64 windows
    t_draft(γ): measured per-step draft time × γ
while drafting position i:
    stop early if the drafter's probability for its proposal < speculative.min_confidence (0.5,
    ADR-0017 item 5), or the grammar rejects it (§8.3)
```

- **Off rule.** γ = 0 is in the search, so the controller drafts only when drafting is predicted to beat plain decoding. When it has chosen γ* = 0 for 64 consecutive steps, the sequence stops drafting altogether (saving the drafter's cost) and retries one window of γ = 1 every 256 tokens to notice a change (ADR-0017 item 8).
- **Instance check (SC-21).** The scheduler tracks realised tokens per second against the controller's prediction over 1,000 windows; below 0.85 × prediction it sends `WorkloadDrift { term: SpeculativeAcceptance }` (RFC-0010 §7.1) so the solver can re-size `T_max` by a delta.
- **Sampled requests** start from the sampled-acceptance prior, which is lower than greedy for the same drafter.

#### 9.4 Chain verify around the ring

```text
tail, window w at positions p..p+γ just verified:
    VerifyAccept → accepted drafts d1..dj, bonus token b at p+j+1
    TokenFrame to ingress: d1..dj, b                         (emitted tokens are final)
    draft next window: MTP/draft head from the hidden state at the accepted position → e1..eγ'
    TokenFrame to head: d1..dj, b + draft e1..eγ'            (RFC-0001 §6.4)
head:
    (draft model) runs γ' draft steps from b, truncating its own KV to the accepted length first
    builds Verify { window: γ'+1 } with rows [b, e1, …, eγ'] at positions p+j+1 .. p+j+1+γ'
every stage, on the frame:
    the frame's first position p+j+1 truncates KV from that position (implicit, RFC-0001 §6.4)
    appends γ'+1 provisional positions, runs the Verify bucket
tail: logits at the γ'+1 rows → VerifyAccept → next window
```

One window costs one ring traversal, and while every tier the stage touches stays memory-bound its stage time stays close to a single token's (cost model §3.10). The scheduler keeps exactly one window in flight per sequence unless asynchronous speculation is on (§9.8).

#### 9.5 Exact acceptance

Acceptance compares drafts with the target's processed distributions at each verify row: after temperature, penalties (with token counts that include the earlier drafted positions on the same path), logit bias and the grammar mask. For a draft position i with target distribution `p_i` and drafter distribution `q_i`:

| Case | Rule |
|---|---|
| Greedy, chain | accept while `d_i = argmax p_i`; the bonus is `argmax` of the first non-accepted row |
| Sampled, chain, point-mass drafts (the default for every drafter) | accept `d_i` if `u < p_i(d_i)`, with `u` the next draw; on the first rejection, sample the residual `p_i` with `d_i` removed and renormalised, and stop; if all are accepted, sample the bonus from `p_{γ+1}` |
| Sampled, chain, stochastic drafts (tail-side MTP or draft heads only, `speculative.stochastic_draft = true`) | accept `d_i` if `u < min(1, p_i(d_i) / q_i(d_i))`; on the first rejection, sample `normalise(max(0, p_i − q_i))` and stop; if all are accepted, sample the bonus from `p_{γ+1}`. `q_i` is available because the drafter runs on the same device as the sampler |
| Greedy, tree | accept the longest root-to-leaf path whose tokens equal the target's argmax at each node |
| Sampled, tree | children are distinct point masses: the drafter's top-k distinct tokens at that node, plus any prompt-lookup branch, deduplicated. At each accepted node, `p'` starts as the node's target distribution; for each child c in try order (drafter score, lookup branch last), accept c if `u < p'(c)`; otherwise set `p'` to `p'` with c removed and renormalised, and try the next child. If no child is accepted, sample from the final `p'` and stop. This is the sequential-test form of sampling from p, so the output distribution is exactly p for any set of distinct candidates. Trees with children sampled from q are not used in v1 |

Every rule consumes draws from the sequence's counter in exactly the order written (§8.2). A head-side draft model always proposes its argmax, a point mass, so the tail never needs the drafter's distribution and no frame carries it. These rules preserve the target's output distribution exactly with respect to the verify pass's logits (ADR-0017 item 4). Logits from a T-row pass can round differently from a 1-row pass; ADR-0019 bounds that, and the parity harness runs speculative against plain decoding for every (dtype class, backend) in exact mode. A pair that fails has speculation disabled in exact mode (ADR-0017's revisit rule). Lenient acceptance exists only behind the `SPEC_LENIENT` approximate knob.

#### 9.6 Trees

Tree windows are an M3 feature (ADR-0017 item 7; RFC-0022 §6 for KV). Shapes come from a small library of static templates per (T_max, width), each a parent array of at most T_max nodes. The controller picks the template that maximises expected accepted length under the sequence's per-depth acceptance `â_depth`, using the standard expectation for independent per-node acceptance. Prompt-lookup continuations are inserted as an extra branch when they do not duplicate an existing path. The head (draft model) or the tail (MTP, draft heads) writes the parent array into the descriptor; attention applies the tree mask from it (RFC-0040), and after acceptance the tail's result tells each stage which path to compact (RFC-0022 §6). Families with recurrent or linear-attention layers verify chains of at most 4 tokens with per-position state snapshots until a tree-capable snapshot scheme passes (ADR-0017 item 6).

#### 9.7 MoE windows

A verify window of T rows touches the union `U(T)` of its experts per MoE layer (cost model §3.9). Each stage's MoE layer routes all T rows in one `MoeRoute`, runs `assign` over them, and prefetches the predicted union for the next layer (RFC-0023 owns expert-union prefetch across a tree). The solver priced `U(T)` when it chose `T_max`, which is why tiered MoE plans often get γ = 1 with MTP (RFC-0001 §6.4).

#### 9.8 Asynchronous pipelined speculation (M3, flag)

With `speculative.async = true` and a draft model on the head, the head keeps up to `speculative.async_depth` (default 2) windows in flight per sequence: while window w traverses the ring, it drafts w+1 on the assumption that w is fully accepted, and issues it immediately behind w. When w's result returns, w+1 is valid if every draft of w was accepted and w's bonus token equals w+1's first token; otherwise the head issues a corrected window at the right position, whose first position truncates every stage's KV, and w+1's results are discarded at the tail. Wasted work is the cost; exactness is unaffected, since only verified tokens are emitted. Enabled by default only after RFC-0061 shows a gain on Tier 1 rigs (ADR-0017 item 7).


### 10. Admission control and request handling

#### 10.1 Two gates

1. **Ingress** routes the request to an instance and checks the instance's advertised KV headroom (RFC-0011 §13; RFC-0001 §8.4 rule 5). This gate works on a digest that can be a few hundred milliseconds old.
2. **The head** makes the binding decision when `SequenceOpen` arrives, against current numbers:

```text
footprint(seq, stage s) = KV pages for min(prompt + max_tokens, context_target(seq)) positions
                        + T_max provisional positions (speculation) + state slots (RFC-0022 §11)
admit if for every stage s: footprint(seq, s) ≤ headroom(s)
         and live sequences < BatchingLimits::max_sequences, or, in latency mode on an instance
             §4.6 allows, live sequences < max(max_sequences, sched.latency_decode_rows)
             with no reserved KV for the extra sequence
         and (throughput mode) a group has room, or one will after pending prefills
headroom(s) is the stage's `KvHeadroom { tokens: [Vram, HostPinned, Nvme] }` (RFC-0022 §11),
    which every stage writes into the side channel of each frame it sends and the tail into each
    TokenFrame (RFC-0031 encodes it); the footprint is checked across the hot window
    and spill tiers together
```

`context_target(seq)` is the request's own limit if it has one, else the plan's published `context_length` (RFC-0011 §11.2). A sequence that later grows past its admitted footprint asks the KV store for pages as it goes; if a stage runs out, the sequence is paused (§10.3), never failed mid-stream for KV unless the policy forbids pausing.

A sequence that does not fit waits in the head's admission queue, bounded by `sched.max_queued` (default 64) and ordered by priority, then arrival. A queued request whose deadline passes, or that arrives to a full queue, ends with `Capacity` and a `Retry-After` hint computed from the predicted drain time of the live sequences (RFC-0001 §10.1). The head publishes queue length and recent prefill rate in its telemetry digest for routing.

#### 10.2 Sequence control

| Message (RFC-0001 §5.8) | At the head | At other stages |
|---|---|---|
| `Open` | prefix match, admission, prefill chunking | KV allocation on the first frame (metadata rides the side channel) |
| `Cancel` | drop pending chunks and windows; forward on the next frame's side channel | release KV at the step boundary; the tail emits `finish = Cancelled` if anything was in flight |
| `Pause` (slow client), sent by ingress **to the head** (amendment C6) | leave the sequence out of steps; KV stays allocated and counted | nothing |
| `Resume` | eligible again from the next step | nothing |
| `Release { persist }` | forward; hand persisted KV to the store (RFC-0022 §8) | same |

#### 10.3 Preemption and pausing

In latency mode the active interactive sequence is never paused by the scheduler. A concurrent decoder admitted beyond `max_sequences` (§4.6) holds no reserved KV and is paused first when the others need pages. A new `Interactive` request that cannot be admitted may pause the newest `Background` sequence, then the newest `Normal` one, until it fits; paused sequences' KV can spill to RAM or NVMe under RFC-0022's rules and they resume when headroom returns. In throughput mode the same order applies across groups. No sequence is paused for longer than `sched.max_pause` (default 30 s) without its client being told through the API's usual keep-alive; after that it ends with `Capacity`.

#### 10.5 Runtime path gates

Some dimensions of an execution path are known only per request: whether a prefix-cache or session hit is reused, whether KV spills to NVMe, the concurrency, speculative windows. The solver gates plan-time dimensions through the conformance manifest (RFC-0011 §4 F7; RFC-0060 §6), and the plan records the run-time values it is certified for in `PlacementPlanExt::certification` (`prefix_reuse`, the KV spill tiers, the maximum concurrency and window; RFC-0011 §2). At admission the head applies those gates: for an instance whose path lacks cover it disables prefix reuse (the sequence prefills in full) or speculation (γ = 0) for that dimension, and the tier manager disables NVMe KV spill (RFC-0020, RFC-0022). Each such restriction is reported in the response's fidelity metadata (RFC-0050). Pair certification itself, including how a failing path invalidates its pairs, is RFC-0060's.

#### 10.4 Drain, plan deltas and epochs

On `Drain`, the head stops admitting, finishes or migrates live sequences (RFC-0001 §7.5) and reports `SequenceEnded` for each. On `ApplyPlanDelta`, budgets and limits change between steps; a lower `max_sequences` takes effect by not admitting, never by evicting; a lower `SpeculativeConfig::max_window` takes effect at each sequence's next window; deltas never raise it (§7.1). Frames from an epoch the runner does not serve are dropped (invariant 7).

### 11. Telemetry from the step loop

RFC-0053 owns `StepRecord` (§2.5 there): one fixed-size record per stage per step, written by the stage's primary dispatcher into the `telemetry` ring, keyed by RFC-0053's `CostTerm` enum, which RFC-0011's `StageCost` also uses, so predictions and measurements share one key space. This RFC fills it: `kind` (including `Replay` and `Canary`), `tokens`, `accepted`, `total` (launch to `O`), `inbound_wait`, `hop_send`, `credit_stall`, `miss_in_place`, `miss_nvme_wait`, `switch_total` (including `t_wait` on backends of §7.5), `sample`, `bytes` per tier, expert hits and misses, and the per-term durations on profiled steps (RFC-0010 §6.1). All times come from the dispatcher's `TickSource`, or from device timestamps mapped through it, so the simulator controls them (RFC-0060 §2.1, §2.3). Hot-thread counters and histograms are keyed by a per-(runner, role) context object passed to the dispatcher, not by thread-local storage, so when the simulator runs every node on one OS thread their counters stay separate.

**Per-block attribution**. Measured parallel work is attributed the way RFC-0011 §2 predicts it. For each `ParallelBlock`, the site that ends last gets the block's time under its term in `StepRecord`, and every other site's time goes to `overlapped` under its own term, summed over the step's blocks. CPU-side sites are timed every step; device-side sites only on profiled steps (RFC-0010 §6.1), where the program's timing points mark the block boundaries. A block whose binding site differs from `predicted_binding` is real drift, and RFC-0053 §2.2 counts it as such; otherwise predicted and measured values are compared within `terms` and within `overlapped`, so overlapping sites never raise spurious drift.

**Acceptance and context statistics.** The tail aggregates, per (drafter, model), windows tried, drafts proposed, drafts accepted and draft time, and the head aggregates the context-length distribution of live and recent sequences:

```rust
/// Carried to the coordinator in the same `ClusterCommand::ReportWorkload` as RFC-0023's
/// ExpertWorkloadReport (proposed as an RFC-0001 amendment), under the same rate limit
/// (at most once per 60 s per instance, and at Draining).
pub struct AcceptanceReport {
    pub drafter: DraftSource,
    pub greedy: AcceptanceStats,                 // windows, proposed, accepted, per-depth accepted counts
    pub sampled: AcceptanceStats,
    pub draft_time: Measured<Nanos>,             // per draft step
}
pub struct ContextStats { pub p50: u32, pub p90: u32, pub max: u32, pub sequences: u32,   // tokens
                         pub uncached_prompt_p90: u32 }   // RFC-0011 §7.1's L_ttft; prior 4096
pub struct ConcurrencyStats { pub mean_live: f32, pub p90_live: u32, pub mean_queued: f32 }
```

Acceptance is learned per (drafter, model) and per request kind (greedy or sampled). v1 has no task-class dimension: a classifier for task classes is not defined anywhere, and the per-sequence posterior (§9.3) adapts to the task within a few windows. ADR-0017 item 5's "(drafter, target, task class)" is therefore narrowed to (drafter, target, greedy or sampled) until a classifier exists.

### 12. Configuration knobs

`speculative.mode`, `speculative.source`, `speculative.draft_model` and `speculative.max_window` are RFC-0054's keys with this RFC as owner; the `sched.*` keys below are instance scope unless marked.

| Key | Default | Meaning |
|---|---|---|
| `sched.spin_us` | 50 µs (10 µs on battery) | spin before parking (ADR-0012) |
| `sched.prewake_us` | 300 µs | timed wake-up ahead of a predicted frame (§3.3) |
| `sched.decode_stall_budget` | 0.5 × predicted t_tok, ≥ 64 tokens of chunk | cap on prefill chunks while an interactive sequence decodes (§4.3) |
| `sched.prefill_inflight` | D (the stage count) | prefill-lane chunks in flight per instance (§4.3) |
| `sched.prefill_share` | 1 prefill step per G decode steps | throughput-mode interleave (§6.1) |
| `sched.max_queued` | 64 | head admission queue length |
| `sched.latency_decode_rows` | 4 [target] | latency mode: most sequences decoding in one step on dense and RAM-resident MoE instances (§4.6); 1 forces serial decoding. NVMe-tiered MoE instances always use 1 |
| `sched.max_pause` | 30 s | longest scheduler-initiated pause |
| `speculative.min_confidence` | 0.5 | drafter confidence to keep drafting (ADR-0017) |
| `speculative.lookup_ngram` | 3 | prompt-lookup match length |
| `speculative.async` | false | asynchronous pipelined speculation (§9.8) |
| `speculative.async_depth` | 2 | windows in flight per sequence when async |
| `exec.profile_every` | 64 | profiled-step period (shared with RFC-0010 `probe.online.profile_every`) |
| `exec.max_gpu_segment_ms` (node, per device) | 500 ms on GPUs that report a run-time limit (WDDM display GPUs); 1,000 ms on Metal [target, RFC-0040 §4.5]; none on compute-only GPUs and the CPU | the longest unit of GPU work the OS times (§7.6); lower values split more and cost a few launch or commit overheads per prefill chunk |

### 13. Failure handling

| Failure | Detection | Action |
|---|---|---|
| Kernel fault or sticky device error during a step | provider `health`, or the O doorbell never fires within `runner.stall_factor` × predicted step | the runner reports `Device` and exits; the supervisor restarts it; the stage's sequences recover as a stage loss (RFC-0001 §7.5, RFC-0033) |
| A CPU pool thread panics | the pool's barrier misses its deadline | `Internal`; the runner exits (a partial expert output would be silently wrong, so no in-process retry) |
| Demand NVMe read fails | I/O completion error | retried once on the same engine, then on the `pread` fallback engine (RFC-0021); if both fail, `Integrity` or `Device`: the sequence ends with an error, the pack extent is re-verified |
| Outbound credit exhausted for longer than `link.failover_timeout` | `credit_stall` accumulates | `Link`: transport failover (RFC-0031), then re-plan if the edge is down (RFC-0001 §10.1) |
| Capture fails at `Warming` | provider error | eager fallback for that bucket (§7.3), reported as degraded |
| Canary mismatch | `Warming` | `Correctness`: the runner does not reach `Ready`; op selection moves to the next provider that passes parity (RFC-0040 §6) |
| Grammar mask not ready | `M[p]` late | the sequence's step waits device-side; if the mask engine errors, the sequence ends with `InvalidRequest` (an uncompilable or stuck grammar), never with an unmasked sample |
| Acceptance anomalies (accepted count > drafted, NaN logits) | sampler checks | `Internal` for the sequence, which ends with an error; logged with the step record for reproduction |
| A frame from an unknown sequence or a stale epoch | sequence table lookup | dropped and counted; never an error to the client |
| A step stalls because a waiter parked on a source that cannot wake it | the stall watchdog; the CI test with a 10 ms CPU segment (§3.3) | a bug: `Internal`; the rule of §3.3 makes it impossible by construction, and the test guards it |
| KV exhausted mid-sequence | `KvStore::append` returns `Capacity` | pause the sequence (§10.3); if pausing is not allowed by policy, end it with `Capacity` |


## Performance model and budgets

The scheduler owns the `t_fixed`, `n_switch · t_switch` and `t_sample` rows of RFC-0001 §14 and the realised fraction of the speculative and throughput predictions. All values are **targets** or **estimates**, as labelled.

| Budget | Value | Derivation |
|---|---|---|
| `t_fixed` per stage per step | ≤ 0.2 · B_ref / W_probe [target, SC-2]: ≈ 0.95 ms on an RTX 4090, ≈ 1.2 ms on M3 Ultra | one replay per device, one descriptor copy, doorbell polls; graph launch overhead is the measured `LaunchCaps::captured_node` × nodes (RFC-0010) |
| Host work before replay | CUDA and HIP: ≤ 20 µs per decode step at the head, ≤ 10 µs elsewhere [target]. Metal and Vulkan: the same, plus one commit or submission per step, with command encoding moved off the critical path (§7.5) | descriptor rows, KV append, page-table updates: a few cache lines per sequence |
| Host waits on Metal and Vulkan | `t_fixed` = commit or submit + n_wait × `t_wait`, with `t_wait` measured by RFC-0010 M5 [estimate 10–50 µs, unmeasured]; SC-2 is provisional for those stages until the M1 spike sets it (§7.5) | §7.5 |
| Hand-off per CPU↔GPU crossing | ≤ t_switch(probe) + 2 µs [target] | doorbell spin, no driver call; t_switch measured by RFC-0010 H2 (5–20 µs estimate, cost model §3.2) |
| MoE layer host path | ≤ 5 µs from `Rd[l]` to copy submission [target] | `assign` over ≤ T·k selections plus one `copy_batch` call |
| Wake-up on the critical path | ≤ 5 µs per stage with pre-wake [target] | spinning at predicted arrival; a cold futex wake costs tens of µs [estimate], which is why §3.3 pre-wakes |
| Sampler (`Logits` to O) | inside the tail's `t_fixed` plus the LM head's bytes / W_eff | LM head read is a byte term the solver already counts |
| Grammar mask | ≤ min(0.5 × ring step, 1 ms) per token [target, RFC-0050] | computed during the next ring traversal |
| Pipelined prefill | TTFT ≤ 1.15 × the cost-model bound + queueing [target, SC-18] | §4.2 formula |
| Interactive stall from other prefills | ≤ one capped chunk per stage per token [target] | §4.3 |
| Throughput mode | ≥ 0.85 × `min(M, D) / (Σ_s t_s + D · t_hop)` [target, SC-15] | §6.1 groups keep D stages busy |
| Speculation | realised ≥ 0.85 × predicted `E[tok] · t_target(1) / (γ·t_draft + t_target(γ+1))` [target, SC-21] | §9.3 controller uses the same expression with measured inputs |

**Worked example (scenario (a) with a draft model, estimate).** RFC-0011 §15.1 predicts γ* = 3 on a 1 TFLOPS CPU with a single-node draft model (M1, amendment C3): t_pass(4) ≈ 381 ms including three draft steps, 2.53 tokens per window at a = 0.7, so ≈ 151 ms per token (6.6 tok/s). The scheduler's own share of that window is one `t_fixed` (≈ 1 ms target), two CPU↔GPU crossings (the RAM-homed layers run first and the GPU run ends with the LM head, RFC-0011 §11.1) at t_switch each, and the sampler; the other ≈ 99% is byte time the solver predicted. SC-21 then requires ≥ 0.85 × (367 / 151) ≈ 0.85 × 2.43 ≈ 2.07× realised speedup over plain decoding (2.73 tok/s) on that rig, i.e. ≥ 5.6 tok/s.

## Alternatives considered

| Alternative | Why not |
|---|---|
| **Distributed scheduling** (each stage composes its own batches) | stages would need to agree on step contents per step, a round of messages on the token path; the head already sees every sequence's state because tokens return to it |
| **Scheduler on the tail** | the tail has the sampled tokens first, but sequences enter at the head and prefill chunks start there; the head also holds the prefix index (RFC-0022 §7.3) |
| **Mixed steps** (decode rows and prefill rows in one step under a token budget) | needs a mixed bucket per combination of decode rows and chunk size, multiplying captured graphs, and mixes `AttnDecode` with `AttnPrefill` in one program. Homogeneous steps with a capped chunk (§4.3) bound interactive stalls with 5 buckets |
| **Per-op eager launches** | t_fixed grows with op count (hundreds of launches per step on a 60-layer MoE stage); kept only as the fallback (§7.3) |
| **Re-capturing graphs per window size** (one graph per γ) | one verify bucket at T_max serves every γ ≤ γ_max at no cost on memory-bound ops, and the CPU computes only active rows; separate graphs per γ multiply memory for no gain |
| **Host-driven MoE dispatch** (synchronise, read routes, launch) | a synchronising driver call per MoE layer; ≈ 0.3 ms per round trip on Windows (`tiered-memory-inference.md` §5.1) times 60 layers |
| **Stateful RNG per sequence** | migration and replay would need to ship RNG state; counter-based draws make the state `(seed, position)` (ADR-0107) |
| **Fixed γ from the plan** | acceptance varies by prompt and task; the controller's measured â adapts within a window or two while the plan bounds memory |
| **Lenient acceptance by default** | changes output (D3); available only as `SPEC_LENIENT` |
| **Asynchronous speculation by default** | its reported gain (up to 2.15×) comes from a different runtime; enabled by default only after RFC-0061 measures it on Syntropy (ADR-0017 item 7) |

## Testing and validation

| Claim | Test | Kind |
|---|---|---|
| Steady-state steps do not allocate, lock or synchronise | counting allocator, lock instrumentation and an instrumented provider shim that fails on any host-side wait after `Warming` (ADR-0012 item 6; RFC-0060) | unit, CI on every Tier 1 backend |
| Doorbells follow the constant-valued handshake | RFC-0040's 1,000-step replay with random 0–10 ms CPU delays per MoE layer, driven through the step loop: no wait is satisfied by a stale flag; deleting any consumer reset stalls the test at the watchdog; after an injected abandoned step and after a re-capture every flag is re-armed (0, with `DC[p]` and `O[p]` at 1) and the next step completes | integration, CI on every Tier 1 backend |
| Dispatchers are deterministic under simulation | the `testkit` simulator drives every dispatcher through `poll_once` and the CPU pool through `run_inline` on one OS thread: two runs with the same seed produce identical step records, frames and tokens; `poll_once` never blocks (a watchdog fails the test on any call longer than its budget) | simulation (RFC-0060 §2.3, ADR-0600) |
| Parity taps do not perturb service traffic | a parity sequence in a step with service sequences: the service rows' outputs are bit-identical with and without it; `Verify` force mode produces the configured acceptance pattern for a fixed seed | integration, parity |
| Concurrent decode rows (§4.6) | on a dense fixture and a RAM-resident MoE fixture, 4 sequences decoding together produce, for each sequence, the same tokens (greedy) or the same draws (seeded) as that sequence decoding alone, within ADR-0019's tolerance for the batched bucket; the first sequence's step time with 4 rows is ≤ 1.10 × its time alone on the SC-4 rigs (RFC-0040 §9 row 2); a fifth sequence queues; on an `Nvme`-homed MoE fixture a second request is `Queued` with its position and issues no pool read until the first ends | parity; benchmark; simulation |
| Steps stay within the OS GPU watchdog (§7.6) | on the Windows rig's display GPU, a 4096-token prefill chunk of a 70B-class stage runs as row-range launches with every launch's measured time ≤ `exec.max_gpu_segment_ms`, and no display-driver reset occurs over 1,000 chunks; on the Mac rig, prefill command buffers each finish within 1 s and decode steps stay one command buffer per segment; on a compute-only GPU nothing is split; a host wait held past the budget ends the step with `DoorbellTimeout` | integration (hardware) |
| Captured and eager execution are identical | every bucket run captured and eager on the same inputs: bit-identical outputs on the same device | parity, per provider |
| Speculative decoding preserves the target distribution | greedy: speculative output equals plain greedy output token for token on the reference executor (exact arithmetic); sampled: chi-square test of token frequencies, speculative vs plain, over 10⁶ draws on a toy model with known distributions, for chain rules with point-mass and stochastic drafters, and for trees with deduplicated top-k children mixed with a prompt-lookup branch | property, statistical |
| Speculative vs plain within ADR-0019 tolerance on real devices | parity harness: teacher-forced KL and top-1 agreement, speculative vs plain, per (dtype class, backend) | parity (RFC-0060) |
| Mid-step waits never stall | hardware CI on CUDA, Metal and Vulkan rigs: a stage with a 10 ms CPU segment and MoE layers 2–4 ms apart runs 10,000 steps with no step waiting on a timer or the stall watchdog, with the device-triggered wake both on and off | integration (hardware) |
| Resumed tails continue exactly | parity test with frequency, presence and repetition penalties and a JSON grammar: kill the tail mid-sequence, resume on another node, and check that the resumed token distribution matches an uninterrupted run (chi-square over many seeds) and that grammar masks stay valid | parity, chaos |
| RNG streams are distinct and resumable | an `n = 4` request yields four different completions for one seed; a client-seeded request repeated on the same build, epoch and devices reproduces its tokens, whether or not it also asks for `deterministic`; a migrated sequence resumed from its `rng_counter` continues the same stream | unit, integration |
| Rollback is correct | forced acceptance patterns (accept 0, all, random j) for chain, tree and recurrent-state families: committed KV and state equal those of plain decoding | property (with RFC-0022) |
| Counter-based RNG reproduces | the same seed on a migrated sequence gives the same draws per position; deterministic mode reproduces tokens on the same build, epoch and devices | unit, integration |
| Pipelined prefill bubble | simulated stages with configured t_s and hops in `testkit`: measured TTFT within 2% of the §4.2 formula for m = 1…64 chunks and D = 1…4 | simulation |
| Interactive stall bound | a decoding interactive sequence with concurrent long prefills: per-token delay never exceeds one capped chunk per stage | simulation, then benchmark |
| Throughput groups fill the pipeline | D = 2…4 simulated stages, M = D · b_μ sequences: aggregate within 5% of `b_μ / θ`; real rigs: SC-15 | simulation, benchmark |
| Admission never over-commits KV | random arrivals, prompt lengths and `max_tokens` against per-stage page pools: no `Capacity` mid-sequence except by design (pause), no deadlock in the admission queue | property, model-based |
| Window controller converges | synthetic drafters with fixed acceptance a ∈ {0.3, 0.5, 0.7, 0.9}: γ settles within 16 windows to the optimum of the §9.3 expression; drafting turns off when it cannot pay | simulation |
| Speculative speedup | SC-21 on the reference rigs: MTP on scenario (b2), draft model on scenario (a), across stages on scenario (c) | benchmark (RFC-0061) |
| TTFT | SC-18 on every reference rig | benchmark |
| Fault behaviour | kill a CPU pool thread, inject a sticky device error mid-step, drop a demand read: the runner fails cleanly with the stated class and the sequence recovers per RFC-0033 | chaos (RFC-0060) |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | step programs for dense families on one device (Decode and Prefill buckets), the dispatcher loop, CUDA graph capture and eager fallback, CPU provider op lists, doorbells, `Dispatcher::poll_once` and `ComputePool::run_inline` (the deterministic simulator needs them from the start, RFC-0060 §2.3), single-sequence decode, greedy and seeded sampling with the counter-based RNG, token-level stops, parity taps in `Prefill` and `Decode` force modes with `FullLocal` and `Gather` (the parity harness is an M0 deliverable, ADR-0019 item 9), the canary on the tap mechanism |
| M1 | the `Verify` parity force mode with its synthetic drafter; MoE layers with the three-site split (hits, copied misses, CPU misses) and NVMe demand reads; `PrefillStream` and n*; Metal capture; chain verify with MTP heads and draft models on one node, the window controller, recurrent-state snapshots for chains; grammar masks; admission and the latency-mode policy (§4.3); the GPU segment budget on CUDA display GPUs and Metal (§7.6); concurrent decode rows in latency mode, and the serial queue on NVMe-tiered MoE (§4.6); telemetry records |
| M2 | multi-stage rings: frames between runners, chunked pipelined prefill across stages, ring return of tokens, pre-wake; intra-node multi-GPU stages in `Local` mode |
| M3 | speculative windows across stages (RFC-0001 §16 M3) with draft-on-head and MTP; the draft-head loader, tested with a project-synthetic head (no project-trained heads, §9.1); tree verify (RFC-0022 M3); throughput mode with groups, priorities and preemption; asynchronous pipelined speculation behind its flag; Vulkan and HIP step programs |
| M4 | per-backend tuning of spin and pre-wake budgets; SC-15, SC-18 and SC-21 as regression gates |

ADR-0017 item 7 stages draft-on-head across pipeline stages at M2; RFC-0001 §16 places "speculative decoding across stages" at M3. This RFC follows RFC-0001 and asks for the ADR to be aligned (amendment C3).

## Open questions

1. **Pre-wake accuracy on Wi-Fi rings.** Arrival-time prediction is good on wired links; with a 250 ms p99 hop the pre-wake either spins long or misses. Measure the power and latency trade-off on the scenario (c) rig in M2.
2. **Frame-in by kernel read versus copy submission.** Reading the receive buffer through host-mapped memory from the graph's first node saves a submission but runs at PCIe read rate from a GPU kernel. For 224 KiB verify frames it should be microseconds; measure on CUDA and HIP, and decide the threshold at which a `CopyIn` submission takes over.
3. **Prefill share in throughput mode.** One prefill step per G decode steps is a guess. A queueing simulation with recorded arrival traces (RFC-0061) should set it, possibly as an adaptive controller on TTFT versus per-sequence rate.
4. **Tree templates.** The template library and its selection rule need data: per-depth acceptance curves for MTP, draft heads and draft models on the conformance families, collected in M2 dogfooding.
5. **Metal and Vulkan host-wait cost.** `t_wait` on both Metal paths and on Vulkan timeline waits is unmeasured. RFC-0010's M5 probe measures it per device, and the solver charges it per host-interacting layer (RFC-0011 §3.3). Spike PLC-E7-T20 decides whether Metal MoE stages default to whole-layer placement (no CPU misses) when `t_wait` is tens of microseconds.
6. **Penalties inside verify windows. Answered 2026-10-20 (OQ-PL-20):** apply penalties as a sparse adjustment per row before the softmax, over the tokens that row's counts (and the drafted tokens earlier on its path) touch, never as a dense pass over the vocabulary. The `LogitsProcess` microbenchmark at vocabularies near 200K confirms the cost (PLC-E9-T14).
7. **Device-triggered wake latency.** The host-function, shared-event-listener and timeline-wait paths of §3.3 add a driver callback and an OS wake; their latency per backend is unmeasured. RFC-0010's H2 wake variant measures it, and `exec.inflight_park_threshold` is set from it (PLC-E7-T18).

## Proposed amendments to RFC-0001

- **C1. Step descriptor input address.** §3.1 here puts the receive buffer's device-visible address in the step descriptor so the captured graph's first node can read the frame. RFC-0040 owns `StepDesc`; RFC-0001 §6.3 item 6 could mention that inputs arrive through the descriptor as well as outputs through a registered buffer.
- **C2. Headroom on the side channel.** Head admission (§10.1) needs every stage's current KV headroom. Proposed: `ActivationFrame::side` and `TokenFrame` carry a per-stage KV headroom word (RFC-0031 owns the layout), and RFC-0001 §8.4 rule 5 should say the head makes the binding admission decision with these values while ingress uses the digest.
- **C3. Single-node draft models at M1 (accepted).** Single-node draft-model speculation is at M1, as this RFC plans, because M1's SC-21 exit is on a dense model without MTP heads; trees, asynchronous and cross-stage speculation stay at M3. Trained draft heads are dropped from v1 (2026-10-20): only the loader remains, at M3. ADR-0017 item 7, RFC-0001 §16 and the vision record the same staging.
- **C6. `Pause` and `Resume` go to the head.** RFC-0001 §5.8 routes `Pause` from ingress to the tail, but only the head composes steps (ADR-0106). Proposed: `Pause { seq } // ingress → head` and the same for `Resume`, with §8.4 rule 3 updated.
- **C4. Priority classes.** `SequenceOpen::priority` is typed `Priority` without values. Proposed: `Interactive | Normal | Background` (§6.3), with RFC-0050 mapping API fields onto them.
- **C5. Pollable dispatchers and parity taps.** This RFC adopts RFC-0060's proposed amendments A3 (`Dispatcher::poll_once`, `ComputePool::run_inline`; RFC-0001 §8.2's threads become loops over them) and A4 (`SequenceOpen::parity`, `TokenEntry::tap`, a `Tap` channel purpose). No further change is needed beyond those.
- **C7. Step order is per sequence.** §5.8's `ActivationFrame::step` comment and §6.2's "every stage processes steps in order" predate the two prefill and decode lanes (RFC-0032 §3.2). Proposed: steps are strictly increasing per instance as identifiers, each stage processes a sequence's steps in order, and frames of different sequences may be processed out of global step order under lane precedence (decode > replay > prefill).
