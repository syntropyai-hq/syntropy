# ADR-0404: Whole-step capture with a device-resident step descriptor and stream-ordered doorbells

- Status: Accepted (2026-10-20); revised 2026-10-20 (Context; decision 8)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0002, ADR-0012, ADR-0014, ADR-0019; RFC-0001 §2.3 (invariant 9), §5.5, §6.3, §8.2; RFC-0040 §3.4, §3.5, §4; RFC-0012, RFC-0023

## Context

- RFC-0001 invariant 9: after `Warming`, hot threads make no synchronising driver call per token. §8.2: the GPU writes routed expert IDs to a host-mapped flag; the CPU pool spins on it; the GPU waits on the CPU's flag device-side. §5.5: expert-cache slots keep fixed addresses for the plan epoch, and the MoE kernel reads a device-resident slot table, so cache changes need no re-capture.
- A surveyed single-GPU MoE engine captures its whole pass as one CUDA graph per verify-window size and spins on a pinned doorbell, because a driver sleep/wake round trip costs about 0.3 ms on Windows (`tiered-memory-inference.md` §5.1). SC-2 caps `t_fixed` near 1 ms on Tier 1 GPUs.
- A captured graph bakes its kernel arguments. Anything that changes per step (positions, KV lengths, row counts, expert selections) must be read from memory at a fixed address.
- Stream memory operations (wait on and write a 32-bit value) exist on CUDA and HIP, but must be probed. *Revised 2026-10-20:* current CUDA has no attribute for the 32-bit operations; the deprecated `CAN_USE_STREAM_MEM_OPS_V1` reads 0 on WDDM devices where they work, so the probe is functional (RFC-0040 §4.2). Their value is a host immediate: once captured, it is fixed in the graph and can only be changed by patching the node from the host (`cuGraphExecBatchMemOpNodeSetParams`). Pre-recorded Vulkan fills are fixed the same way. Metal has CPU-signalable shared events whose values are set per step. Vulkan has no portable way for a command buffer to wait mid-buffer on a host-written value.
- The three MoE sites (GPU hits, GPU-computed copied misses, CPU in-place misses) must agree on which expert runs where, or work is duplicated or lost.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Per-op launches, host-driven | Simple; dynamic arguments | Hundreds of launches per MoE token; above the SC-2 budget |
| B. Capture, then patch node arguments each step | Keeps arguments in launch parameters | One host call per changed node per step; no equivalent on Metal ICBs or Vulkan |
| C. Capture; host decides the MoE split and sends a dispatch list the GPU waits for | Single decision point | A host round trip before GPU hit work can start |
| **D. Capture; all per-step values in a device-resident step descriptor copied by the graph's first node; table changes applied by a delta-ring node at the start of each step; MoE split by one pure function (RFC-0023's rule) every site evaluates from the same doorbell and byte budget; doorbells by stream memory operations, spin kernels where those are missing, and split submissions on Vulkan** | One submission per step on CUDA, HIP and Metal; GPU hit work starts right after routing; sites agree without messages | `assign` must be bit-identical in five kernel languages; the Vulkan path pays one host signal per wait |
| D-i. As D, doorbells carrying monotonic step numbers, captured nodes patched per step | Stale flags excluded by value | ≈ 180 node patches per step on a 61-layer MoE stage [estimate], against a ≤ 20 µs host-work target |
| D-ii. As D, constant values alternating by mirror parity, never reset | No host patching | A missed or abandoned step lets a stale value pass the same parity's wait two steps later, silently |
| **D-iii. As D, constant values with a consumer reset (producer writes 1; consumer waits for 1, then writes 0)** | No host patching; a protocol fault stalls instead of passing stale data | Every flag must be raised every step; multi-consumer doorbells become one flag per consumer |

## Decision

**Option D.**

1. Every captured step graph begins by copying the step descriptor from a pinned mirror; mirrors come in pairs with one graph instance per parity, and a "descriptor consumed" doorbell guards reuse.
2. Graphs reference only fixed-address buffers; dynamic values come from the descriptor and tables it points to.
3. The MoE layer protocol is RFC-0040 §3.4 (policy and tables in RFC-0023): route, signal; hits; copy wait then copied misses; CPU wait then combine. Outputs are combined per row in ascending expert index, in F32.
4. The split function (`kernels::moe::assign`) is specified once and implemented identically in Rust, CUDA, HIP, MSL and Slang, with a cross-implementation property test.
5. Slot-table, copy-eligibility and KV block-table changes go through a host-mapped delta ring that the step graph's `ApplyTableDeltas` node applies, up to the ring position the descriptor's `delta_head` names, before any kernel of the step reads a table (RFC-0023 §5.4). Every site evaluates a step at that one table version.
6. Doorbell mechanisms: CUDA/HIP stream memory operations, falling back, only where those are absent, to a one-thread store kernel and a one-warp spin kernel that run the same constant-valued handshake (never step-number comparisons); Metal shared events; Vulkan segments submitted together in one `vkQueueSubmit2`, each waiting on a timeline-semaphore value the host signals; CPU atomics. **Captured and pre-recorded doorbells use constant values with a consumer reset (D-iii; RFC-0040 §4.2).** Each flag has one producer and one consumer; the producer writes 1; the consumer waits for `EQ 1`, then writes 0 before producing anything that leads to the flag's next set; every flag is raised every step, even when its work is empty; step-level flags (descriptor consumed, step done, host-set step flags) exist per mirror parity. The handshake governs host-mapped flags (CUDA, HIP, Vulkan device-to-host fills, CPU); Metal events and Vulkan timeline waits carry per-step values set at encode or submit time. A flag is raised once per pass of the program part that owns it (once per drafted token for a chained MTP module). Correctness: a set at step s+1 is causally after the consumer's reset at step s, because the GPU passes a layer at step s+1 only after that layer's step-s waits, which the host satisfies only after resetting, and the host sets a device-waited flag for step s+1 only after seeing that step's route flag. Metal event values and Vulkan timeline wait values are supplied per step, so they carry step numbers. Flags sit on separate 64-byte lines. Stages whose MoE layers cannot miss are captured without per-layer doorbells.
7. CUDA capture uses thread-local mode, so host registration on another thread does not invalidate a capture in progress.
8. *Revised 2026-10-20 (ADR-0014 rule 6):* **Every in-step wait ends before the OS GPU watchdog.** A spin kernel or device-side wait on a host flag is bounded by the device's GPU segment budget; past it the step fails with `DoorbellTimeout` and is abandoned before the OS resets the GPU or kills the process. On CUDA, whether native memory operations are used is decided by a functional probe at `open` (capture a wait and a reset, replay with a host write), never by a device attribute.

## Consequences

- Positive: the host submits once per step on CUDA, HIP and Metal, which is what makes the SC-2 budget reachable.
- Positive: expert-cache adaptation, KV growth and batching changes never invalidate a graph.
- Positive: the MoE result is independent of the split decision, except for each site's own rounding, which ADR-0019 accepts and the harness measures.
- Negative: kernels must handle `rows_active` below their bucket and read dimensions indirectly, which costs some generality in kernel code.
- Negative: on Vulkan, each MoE layer with a CPU site adds a host semaphore signal to the step; the solver sees this and may prefer HIP on AMD.
- Negative: every doorbell flag is raised and reset every step even when its layer has nothing for that consumer, which costs one round trip per flag; stages that cannot miss drop the flags entirely.
- Negative: a spin kernel occupies one warp while the CPU computes misses.

## What would make us revisit

- Measured doorbell round trips on a Tier 1 backend exceed the cost model's `t_switch` range (5–20 µs) by more than 2×, so execute-in-place loses to streaming at batch 1 where the model says it should win.
- Vulkan gains a portable device-side wait on host memory. Drop the split.
- Spin kernels measurably slow hit kernels (RFC-0040 open question 3). Move to copy-engine-only signalling or larger wait granularity.
- The 1,000-step replay test with random CPU delays (RFC-0040 Testing) finds a stale-flag or lost-wake case on any backend. Then that backend's doorbells move to the constant-valued spin and store kernels of RFC-0040 §4.2's fallback.

## Log

- Revised 2026-10-19: decision 6 changed from monotonic step-number flags to the constant-valued consumer-reset handshake, and decision 7 added, because captured memory-operation values are fixed at capture.
- Revised 2026-10-19: decision 6 scopes the handshake to host-mapped flags and counts raises per pass; the revisit bullet names the constant-valued fallback.
- Revised 2026-10-20: the CUDA doorbell probe becomes functional, because no device attribute reports the 32-bit operations, and decision 8 bounds every in-step wait below the OS GPU watchdog (ADR-0014 rule 6).
