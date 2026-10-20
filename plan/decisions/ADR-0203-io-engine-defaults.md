# ADR-0203: I/O engine defaults: io_uring with plain direct reads, fixed buffers only for hot slots, SQPOLL off, byte-capped prefetch

- Status: Accepted (2026-10-20; revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0021))
- Related: ADR-0012, ADR-0013, ADR-0200; RFC-0001 §8.2–§8.4; RFC-0010; RFC-0020 §5; RFC-0021 §3, §4; RFC-0052; RFC-0061

## Context

ADR-0013 fixes the engines per OS and the per-volume selection by measurement. Within the Linux default it leaves several choices open, and each one has a cost:

- **Registered (fixed) buffers** save the per-I/O page-pinning and address translation, but are charged to `RLIMIT_MEMLOCK` and limited to 1 GiB each (ADR-0013). The default memlock limit on most current Linux systems is a few MiB unless a service unit raises it. Model weights are tens to hundreds of GB.
- **`SQPOLL`** removes submission syscalls by running a kernel thread that spins on the submission queue. It costs a core, and the CPU compute pool needs every core it can get for in-place experts (ADR-0012; CPU-hosted experts reach only 0.24–0.35 of DRAM peak end to end, `hardware-and-performance-model.md` §0).
- **`IOPOLL`** needs NVMe polled queues, which consumer installs do not configure.
- **Request rates are low.** Scenario b2 at h = 0.9 reads ≈5.5 GB/s of experts, ≈210 expert reads or ≈1.5K 4 MiB chunks per second; rows add ≈1.6K reads/s at 100 tok/s. One blocking thread manages ≈80K reads/s (`tiered-memory-inference.md` §10.3). Depth on large reads matters; submission cost does not.
- **Demand against prefetch.** In-flight reads cannot be pre-empted. Without a bound, a demand read for a missed expert can wait behind tens of MB of speculative reads, while the step stalls.
- No published io_uring numbers exist for this workload (ADR-0013, Context).

## Options considered

| Option | For | Against |
|---|---|---|
| A. io_uring with every destination registered | Lowest per-I/O CPU | Fails under default memlock limits; ties registration of tens of GB to `RLIMIT_MEMLOCK`; gains little at 4 MiB+ requests |
| B. `PreadPool` everywhere | Works in every sandbox | Depth costs one blocked thread per read; more CPU per GB |
| C. io_uring with `SQPOLL` on | No submission syscalls | Spends a core that in-place experts need, for rates far below where submission cost matters |
| **D. io_uring (`SINGLE_ISSUER`, `DEFER_TASKRUN` when available) with registered files; fixed buffers only for hot destinations; plain `O_DIRECT` reads elsewhere; `SQPOLL` and `IOPOLL` off unless measured; strict priority classes with prefetch bytes in flight capped while demand is pending** | Reaches depth from one thread; works under default limits; keeps cores for compute; bounds demand latency | Two read paths (fixed and plain) to test; the byte cap can leave some prefetch bandwidth unused when demand is frequent |

## Decision

**Option D.**

1. Linux default `UringDirect`: one ring per I/O thread; `IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN`, falling back to `COOP_TASKRUN`, then none; registered files.
2. Fixed buffers only for `IoStaging`, RAM expert-cache and staging slots over an `Nvme` home, row caches and KV spill pages, within the memlock budget computed by RFC-0020 §5. Everything else, including bulk loading, uses `IORING_OP_READ` with `O_DIRECT`. Registration failure falls back to plain reads.
3. `SQPOLL` only when the probe measures ≥ 15% more throughput at the real request mix and a core is spare after the compute reservation; `IOPOLL` only with polled queues present and a measured gain.
4. Five priority classes: `Demand` > `PrefetchNear` > `Load` > `PrefetchFar` > `Background`. While any demand read is queued or in flight, prefetch in flight is capped at `io.prefetch_inflight_max` (32 MiB); otherwise prefetch may fill the probe's `qd_target`.
5. `PreadPool` (16 threads, `O_DIRECT`) wherever io_uring is unavailable; the same priority rules run on the I/O thread in front of it, and its effective queue depth is its thread count.
6. **Relation to ADR-0013.** ADR-0013 item 2 registers "the tier manager's pinned anonymous arena" as fixed buffers. This ADR narrows that to the hot destinations of item 2 above, because registration is charged to the user's `locked_vm` against `RLIMIT_MEMLOCK`. It amends ADR-0013 item 2's scope, and ADR-0013 records it.
7. **No direct-to-GPU storage path in v1** (added 2026-10-20). Every NVMe read lands in host memory the tier manager owns, through the engines above, and a device copy or in-place compute follows. GPUDirect Storage (cuFile) and DirectStorage are not built; `syntropy doctor` and the probe keep a read-only capability report per (volume, GPU): whether cuFile opens and whether it would get a peer-to-peer path or compatibility mode, and whether DirectStorage is present (RFC-0021 §9; ADR-0013 items 4 and 6 as revised).

## Consequences

Positive:
- Runs at full depth without raised limits; packaging raising `LimitMEMLOCK` (RFC-0052) only makes hot-slot reads slightly cheaper.
- No core is lost to submission polling by default.
- A demand read's wait behind prefetch is bounded: ≈5.6 ms at 32 MiB and 6 GB/s in the worst, strictly in-order case [estimate].

Negative:
- Two read paths on Linux.
- The 32 MiB cap is a chosen number, not a measurement (RFC-0021 Open question 2).
- If the RFC-0061 benchmark shows io_uring no better than the pool, a whole engine is maintained for little gain.

## What would make us revisit

- The RFC-0061 benchmark shows `UringDirect` within 5% of `PreadPool` in throughput and within 20% in CPU time per GB at expert sizes (ADR-0013's revisit condition): keep io_uring only for rows and KV.
- Small-expert models (1–3 MiB experts) or row-heavy models push request rates high enough that the probe enables `SQPOLL` on most machines: make it the default for those models.
- Measured p99 demand wait under saturating prefetch exceeds one expert's read time by more than 2× at the default cap: lower the cap or add in-flight cancellation of far prefetch.
- NVIDIA's documented GDS support list grows to GeForce cards, as the capability report would show: re-add a cuFile path behind a measurement gate (ADR-0013).

## Revisions

- **Revised 2026-10-20.** Item 7 added: GPUDirect Storage and DirectStorage are dropped from v1, with a read-only capability report kept. GDS direct mode is documented for Tesla and Quadro only, and DirectStorage's gain is GPU decompression, which uncompressed quantised weights do not use. TIER-E5-T20 and TIER-E5-T21 leave the build.
