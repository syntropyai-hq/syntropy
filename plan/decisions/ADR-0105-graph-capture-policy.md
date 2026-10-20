# ADR-0105: Capture whole steps per bucket; route dynamically through device tables

- Status: Accepted (revised 2026-10-19 and 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the placement and scheduling area)
- Related: ADR-0012, ADR-0014, ADR-0002; RFC-0001 invariant 9; RFC-0012 §2, §7; RFC-0040 §3.4, §3.5, §4.4; SC-2

## Context

`t_fixed` is a first-order term: ≈5 ms per token for one existing GPU engine against SC-2's ≈1 ms target on an RTX 4090 (cost model §2.2). A 60-layer MoE stage issues hundreds of kernels per step. A surveyed engine captures the whole pass as one CUDA graph per verify-window size and spins on host doorbells instead of driver waits (`tiered-memory-inference.md` §5.1). Graph capture fixes addresses and shapes at capture time, while Syntropy's steps change every time: which experts are cached, which KV pages a sequence uses, how many rows are active, the tree shape of a window, sampler seeds.

Backends differ: CUDA and HIP capture streams with memory-op nodes; Metal replays indirect command buffers; Vulkan has no portable in-command-buffer wait on a host value, so each host wait splits a step into submissions (RFC-0040 §4.2).

## Options considered

| Option | For | Against |
|---|---|---|
| A. Eager launches only | simplest; nothing to invalidate | launch cost per op on every step; the main contributor to t_fixed |
| B. Capture per layer or per op group | smaller graphs; easier invalidation | still one submission per layer; hand-offs between graphs need host involvement |
| C. Capture per exact step shape (rows, γ, tree) | no padding | combinatorial graph count and memory; re-capture on the token path |
| **D. Capture whole steps per (step program, bucket, descriptor parity); every per-step value read from device-resident tables via a step descriptor copied by the graph's first node; MoE routing, KV paging, tree masks and sampler state as tables; host work limited to doorbell spins and asynchronous copy submissions; eager fallback per bucket and per op** | one replay per device per step; graphs valid for the whole plan epoch; bounded graph count (RFC-0012 §2's table: 16 instances per stage in latency mode, up to 34 in throughput mode) | kernels must read dynamic state from tables (the provider contract in RFC-0040); padding rows inside buckets; Vulkan pays one submission per host wait |

## Decision

**Option D.**
- Buckets: `Decode`, `Verify { T_max }`, `Batch { 2^k }` up to micro_batch × T in throughput mode, `Batch { 2 }` and `Batch { 4 }` in latency mode on instances that allow concurrent decode rows (dense or RAM-resident MoE; added 2026-10-20), a prefill ladder ({64, 256, 1024, 4096} below the plan chunk, plus the chunk), and `PrefillStream { chunk }` when streaming applies. Two descriptor parities each; profiled and drafter programs one each (RFC-0012 §2).
- All buckets are captured in `Warming`, before the canary, sized for the epoch's maxima of `T_max`, prefill chunk and micro-batch; deltas may only lower them, and a raise is a new epoch. Capture runs in thread-local mode on a helper thread, and the device's own dispatcher swaps executables in between steps.
- Expert-cache slots, KV page pools, scratch, staging rings and doorbell pages have fixed addresses for the epoch; their contents change without invalidating graphs.
- A bucket that fails to capture runs eagerly with identical output and is reported as degraded; `features.disable = ["graph_capture"]` forces eager for diagnosis.

## Consequences

- Positive: t_fixed approaches one submission per device per step; the token path has no synchronising driver call; dynamic routing costs nothing at the host.
- Negative: graph memory counts against `plan.scratch` and is measured in `Warming`; every op must be written against the step descriptor. Whole-step capture of host waits exists only on CUDA and HIP: on Metal a step is one pre-encoded command buffer with a shared-event wait per host-interacting layer, and on Vulkan one submission with timeline waits (RFC-0012 §7.5), so those stages pay `t_wait` per host wait, measured by RFC-0010's M5 probe and charged by the solver. SC-2 is provisional for Metal and Vulkan stages whose MoE layers wait on the host, its value to be set from the M1 `t_wait` measurement. Capture uses thread-local mode on a helper thread, and buckets are pre-sized for the epoch's maxima, so deltas never capture or allocate on the token path.
- Work created: RFC-0040 owns the descriptor and capture rules; RFC-0060 tests captured-versus-eager equality per provider.

## What would make us revisit

- Graph memory on 16 GB GPUs exceeds 3% of VRAM with the default buckets. Then drop `PrefillStream` capture and run it eagerly (prefill is not launch-bound).
- A backend gains a portable device-side wait on host memory (Vulkan) or cheaper waits (Metal measurements in M1). Then remove the submission splits.
- Measured t_fixed with capture is within 10% of eager on some backend. Then default that backend to eager and save the memory.

## Revisions

- **Revised 2026-10-20.** Latency mode lets up to 4 sequences decode in one step on dense and RAM-resident MoE instances when KV fits (RFC-0012 §4.6), so those stages capture `Batch { 2 }` and `Batch { 4 }` as well: 4 more graph instances plus 2 profiled variants per stage (RFC-0012 §2). Instances with an `Nvme`-homed pool stay serial and capture no extra buckets.
