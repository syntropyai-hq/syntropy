# ADR-0401: Per-op provider selection: one memory domain per device, then availability, parity and a 3 % benchmark margin

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0002, ADR-0014, ADR-0019, ADR-0602; RFC-0001 §5.7; RFC-0040 §2.3, §6; RFC-0010, RFC-0011

## Context

- ADR-0002 item 5: choose each op per device by availability, then by microbenchmark at the model's real shapes, separately for decode and prefill shapes; cache per hardware fingerprint, driver version and Syntropy version; ties to native; a native kernel must pass parity first.
- Several providers can drive one GPU (native CUDA, ggml CUDA, native Vulkan, ggml Vulkan). Buffers, streams and events are shareable only within one API. Mixing APIs on a device means two allocators the tier manager must reconcile and no common graph capture.
- On AMD, published results show ggml's Vulkan backend faster than its HIP backend for decode on some cards, and HIP faster for prefill on others (`backends-and-portability.md` §1.4). Neither API wins by rule.
- Benchmarks are noisy. If the fastest candidate is always picked, choices flip between restarts and plans become hard to compare.
- The pack layout must be fixed before packing, but the final choice is made in `Warming` (RFC-0001 §3.3, §5.7).
- A GEMV timed while its weights sit in L2 overstates W_eff by the cache-to-DRAM ratio, and the solver would then plan with wrong numbers.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Fixed provider per device (native where present, ggml otherwise) | Simple; no benchmark time | Misses ops where ggml is faster; no recovery when a native kernel fails parity on new hardware |
| B. Fastest candidate per op, any API | Maximum speed on paper | Two APIs on one device: duplicate allocators, cross-API copies, no whole-step capture |
| C. Provider-reported cost estimates only | No benchmark time | Estimates are exactly what measurement is meant to replace (cost model §5.2) |
| **D. One memory domain per device; per op inside it, filter by availability and parity, then benchmark at plan shapes on out-of-cache synthetic weights; a challenger needs a 3 % margin; ties to native; layout pass in `Fetching`, final pass in `Warming`; results cached** | Keeps one allocator and one queue per device; measured; stable across restarts | Benchmark time on first start; a domain choice per device that can be wrong for some ops |

## Decision

**Option D.**

1. **Domain first.** NVIDIA uses `Cuda` whenever the CUDA provider opens the device. Apple uses `Metal`. CPUs use `Host`. AMD and Intel GPUs benchmark `Hip`/`Vulkan` (or `Vulkan`/`Ggml(Sycl)`) and take the domain with the lower predicted step time, by a 5 % margin, else HIP (or Vulkan). ggml-only backends use their ggml domain.
2. **Candidates** must answer `query`, satisfy the RFC-0042 accumulation contract, match the packed layout, be deterministic when asked, hold a certificate in the release's `ConformanceManifest` (an `OpCert` matching the kernel's content hash, ADR-0602; or a derived certificate for a target no lab rig can execute, under RFC-0040 §6.2's three conditions: the same `source_key`, a binary compatible under CUDA's same-major rule or JIT-compiled from identical PTX, and a first-load canary that matches the reference digest; such results are labelled `exact (derived certification)`, and a failed canary falls back to a certified provider or marks the path approximate), and pass a runtime spot check against `kernels::reference`. An op with no certificate is never selected.
3. **Benchmark protocol:** synthetic weights of the exact scheme and shape, rotated so the working set is at least 2× the last-level cache; device-event timing over captured loops of at least 2 ms; median of 5 after 3 warm-ups; a default budget of 3 s per runner, spent in decreasing order of predicted step-time share.
4. **Rule:** cost = measured time + forced segment breaks × `t_submit` + forced fences × `t_fence`. The incumbent (native if present, else the cached choice) is replaced only by a challenger at least 3 % cheaper.
5. **Two passes:** a layout pass in `Fetching` on synthetic weights (so packing knows the layout), and a final pass in `Warming` among layout-compatible candidates. The layout is chosen per tensor group across every reader that may compute on it (all MoE sites, and the fallback device of level-F ops): by default one layout every reader accepts (`SOURCE` always qualifies), weighted by each reader's expected share of the group's bytes; per-site copies only where the plan marks the group, with the extra bytes charged by the solver (RFC-0040 §5.4).
6. **Cache** keyed by (device fingerprint, device class, domain provider builds, driver, `OpKey`, `ShapeSig`), stored per node and in the pack's selection side record (RFC-0041 §5.2). A hit skips the benchmark but never the runtime parity spot check.
7. **The probe measures W_eff and F_eff with the kernel this rule selects**, so the capability profile describes the kernels that will run.

## Consequences

- Positive: plans use measured numbers from the real kernels; selection is reproducible across restarts; a broken kernel on unseen hardware is caught before it serves.
- Positive: on AMD and Intel the API choice is made by measurement, not by assumption.
- Negative: the first start on a new device spends up to the budget on benchmarks (later starts hit the cache).
- Negative: a device whose best ops are split across two APIs gets only one of them. Accepted for one allocator and whole-step capture.

## What would make us revisit

- Selection flips more than once per ten restarts on an unchanged machine in RFC-0061's lab. Raise the margin or add repeated measurement.
- On some device class, the best cross-API mix beats the best single domain by more than 10 % on the step program. That would justify cross-domain interop for that class.
- First-start benchmark time is the largest item in time-to-first-token for new users (RFC-0061). Ship per-device-class selection seeds in the release.

## Log

- Revised 2026-10-19: decision 5 makes the layout choice joint per tensor group, because per-device choices broke the three-site MoE split; decision 6's storage follows RFC-0041's side records, which keep these types out of `manifest`'s layer.
