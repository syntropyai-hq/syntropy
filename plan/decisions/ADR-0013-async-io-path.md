# ADR-0013: Async I/O path: direct I/O into owned pinned memory, one engine per OS, measured fallbacks
- Status: Accepted (revised 2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0005, ADR-0007, ADR-0012, ADR-0014, ADR-0020; RFC-0010, RFC-0020, RFC-0021, RFC-0022, RFC-0023, RFC-0041, RFC-0062

## Context
ADR-0007 makes NVMe a capacity tier read with direct I/O, into memory the tier manager owns. This ADR picks the mechanism on each OS. The evidence:

- **Request shape.** Expert reads are large and contiguous: one DeepSeek-V3-class expert at Q4_K_M is ≈26.5 MB (estimate), and 2026 MoE experts are 1–3 MiB (`plan/research/hardware-and-performance-model.md` §2.4, `backends-and-portability.md` §7.2).
  - At QD1 with 4 KiB reads, a 990 PRO delivers 90 MB/s against 7450 MB/s sequential [verified].
  - Throughput comes from size and queue depth (QD 2–8 for expert-sized reads) and from keeping each expert contiguous on disk. Packs guarantee that by aligning every expert to 4 KiB (ADR-0005).
- **Effective rates in the field.** Expert-sized random reads through the page cache ran at ≈1.5–2.6 GB/s on a Gen4 drive on Windows, and at ≈2.0 GB/s against a 17.5 GB/s Apple SSD (`tiered-memory-inference.md` §11, inferred). The engine has to close that gap.
- **mmap and the page cache lose on hot data.**
  - A mapped expert pool read at ≈19 GB/s against 42.8 GB/s from an anonymous arena on Windows.
  - Row lookups ran at 2.1K vs 14.4K rows/s against direct I/O on SATA (`tiered-memory-inference.md` §10.1, §10.4, reported).
  - The exception is macOS: on a 17.5 GB/s Apple SSD, custom caches and prefetch lost to the OS page cache (§8, reported). There, clean file pages are dropped under pressure rather than compressed. An unwired anonymous arena caused compressor panics in one engine.
- **Submission cost.** One thread doing blocking `pread` peaks at about 80K reads/s, ≈12 µs of kernel time per read. A 16-thread `O_DIRECT` pool on Linux and a 4-thread IOCP pool on Windows were enough in one engine (§10.3, verified code). We found no published io_uring numbers for this workload (§10.4).
- **Thermal throttling.** Gen5 drives without heatsinks throttle within seconds under load, so the solver must cost NVMe at the throttled steady-state rate (`hardware-and-performance-model.md` §2.4).
- **Platform constraints.**
  - io_uring is blocked by Docker's default seccomp profile, which returns EPERM. It is unavailable to Android apps and disabled on ChromeOS, and the `io_uring_disabled` sysctl (Linux 6.6+) can turn it off (`backends-and-portability.md` §6.1, §5.4; Docker seccomp docs).
  - io_uring registered buffers must be anonymous memory, at most 1 GiB each, and are charged to `RLIMIT_MEMLOCK` (io_uring_register(2)).
  - IoRing needs Windows build 22000+ and exposes registered buffers and handles (Microsoft Learn, `ioringapi.h`).
  - `F_NOCACHE` does not purge pages already cached, so it must be set before the first read (`hardware-and-performance-model.md` §2.4).
  - GPUDirect Storage runs on consumer GeForce cards only in compatibility mode, a CPU bounce buffer, unless `gdscheck` says otherwise. NVIDIA's ≥2× claim applies to PCIe-switch topologies.
  - DirectStorage is built around D3D12 resources. It measured ≈3 GB/s uncompressed on Gen3 and helps mainly with GDeflate-compressed assets. Quantised weights compress poorly (estimate).

## Options considered
1. **mmap and the page cache everywhere.** Simple and zero-copy, but residency is left to kernel LRU, against ADR-0007. Measured 2.2× slower on Windows for a hot pool. Kept only for cold, read-once data and where measurement on macOS says it wins.
2. **A portable blocking-`pread` thread pool everywhere.** Works in every sandbox. The thread count equals queue depth, and each request costs ≈12 µs of a core. It is the mandatory fallback, but not the first choice where the OS offers a ring.
3. **Linux AIO (libaio).** O_DIRECT only, superseded by io_uring, with no registered buffers. Rejected.
4. **GPUDirect Storage as the primary NVIDIA path.** Linux-only and effectively datacenter-only. Rejected as a dependency; used opportunistically.
5. **DirectStorage as the primary Windows path.** D3D12-centred and unproven for uncompressed quantised weights. Kept as an optional GPU-upload path behind a measurement gate.
6. **GPU-initiated storage access.** It needs custom NVMe drivers, so users cannot install it. Rejected.
7. **Chosen: one `IoEngine` interface with a native engine per OS, the portable pool as universal fallback, and per-volume selection by measurement.**

## Decision
1. **Interface.** `syntropy-io` exposes `IoEngine`: batched submission of `ReadReq { file, offset, len, dst }`, where `dst` is a slice of a tier-manager-registered buffer, plus a completion poll. It runs on the I/O threads defined in ADR-0012. Writes (pack building, KV session persistence in RFC-0022) use the same interface.
2. **Linux: `UringDirect`.**
   - Files are opened `O_DIRECT` and registered.
   - The tier manager's pinned anonymous arena is registered as fixed buffers in chunks of 1 GiB or less, within the memlock budget, and reads use `IORING_OP_READ_FIXED`. *Narrowed 2026-10-19 by ADR-0203 item 6:* only the hot read destinations are registered as fixed buffers, because registration is charged to the user's `locked_vm` against `RLIMIT_MEMLOCK`; other destinations use ordinary reads.
   - Alignment comes from `statx(STATX_DIOALIGN)`, falling back to the logical block size.
   - `SQPOLL` and `IOPOLL` are off by default. Each is enabled per volume only if the probe shows submission-bound throughput, because SQPOLL costs a core (ADR-0012).
3. **Linux fallback: `PreadPool`.**
   - Used when `io_uring_setup` returns ENOSYS or EPERM, or the sysctl disables it. That covers Docker default seccomp, Android and ChromeOS.
   - Blocking `pread` with `O_DIRECT`, 16 threads by default, affined to the I/O cores.
   - On filesystems that refuse `O_DIRECT`, buffered `pread` followed by `POSIX_FADV_DONTNEED`. The volume is flagged "buffered" in the resource graph.
4. **Windows.**
   - `IoRing` on build 22000+, with registered buffers and handles and files opened with `FILE_FLAG_NO_BUFFERING`. Each operation is checked with `IsIoRingOpSupported`.
   - Fallback: an IOCP engine with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and 4 submitting threads. This also covers Windows 10.
   - DirectStorage is not built in v1 (revised 2026-10-20, below). The probe reports whether it is available, read-only.
5. **macOS.**
   - `MacNoCache`: `F_NOCACHE` set at open, before any read, page-aligned destinations and a `preadv` pool (8 threads by default), with requests of at least 32 KiB.
   - `MacMapped`: `mmap` with `madvise`, used for a volume when the probe measures it faster at the model's real request size.
   - On unified memory the destination buffer is GPU-visible as allocated, so there is no second copy (ADR-0014).
6. **GPUDirect Storage: not in v1** (revised 2026-10-20, below; the earlier text made it opportunistic). There is no cuFile read path. A read-only capability report records, per (volume, GPU) pair, whether cuFile could open and whether it would get a peer-to-peer path or compatibility mode, so the revisit trigger can be checked on real machines.
7. **Selection.**
   - RFC-0010's probe runs every available engine against the model file itself, at the census's real request sizes (expert size, row-bundle size) and QD 1, 4 and 16.
   - It then runs a background 60–120 s sustained read during load and stores the throttled rate.
   - The fastest engine per (node, volume) and its sustained rate go into the capability profile.
   - It re-probes when throughput drifts more than 20% for longer than a window, or when the drive's SMART temperature crosses its warning threshold.
8. **Shaping.** One aligned request per expert or row bundle. Large reads are split at the probe's best chunk size, 4 MiB by default, and are always issued at depth. The engine never reads through the page cache, except in `MacMapped` and the flagged buffered mode.

## Consequences
- **Positive.**
  - The hot path never depends on page-cache behaviour.
  - Every sandbox still works, more slowly, through the pool.
  - Per-volume measurement replaces per-OS folklore, and the solver gets honest, throttled NVMe rates.
- **Negative.**
  - There are six engines to maintain (`UringDirect`, `PreadPool`, `IoRing`, IOCP, `MacNoCache`, `MacMapped`). The two optional paths (GDS and DirectStorage) were dropped from v1 on 2026-10-20 and survive only as a read-only capability report.
  - Registered buffers count against memlock. The tier manager must budget this, and packaging (RFC-0052) must raise `RLIMIT_MEMLOCK` through the service unit where allowed.
- **Work created.**
  - RFC-0021: the engines, chunking and completion plumbing.
  - RFC-0020: arena registration and the memlock budget.
  - RFC-0010: the I/O probe and drift detection.
  - RFC-0041: pack alignment guarantees.
  - RFC-0022: session file writes.
  - RFC-0062: CI runs the Linux suite twice, once with io_uring denied by seccomp, and runs Windows on IoRing and IOCP.
  - RFC-0061: a benchmark of `UringDirect` against `PreadPool` at expert sizes, since no published number exists.

## What would make us revisit
- RFC-0061 shows `UringDirect` within 5% of `PreadPool` in throughput and within 20% in CPU time per GB at expert sizes on Tier 1 Linux hardware. Then keep io_uring only for row lookups and KV, where request counts are high.
- `MacMapped` wins on every Apple SSD measured. Make it the macOS default and drop `MacNoCache` to a fallback.
- GDS gains true peer-to-peer support on GeForce through upstream P2PDMA, NVIDIA's documented GDS support list grows beyond Tesla and Quadro cards (cuFile was open-sourced in August 2026), or DirectStorage gains a supported CUDA or Vulkan destination. Then re-add the path behind a measurement gate.
- Resolved (2026-10-19): DirectStorage stays behind a measurement gate and outside M1. The research gives no evidence that DirectStorage helps uncompressed quantised weights, and its D3D12 resource model needs interop to reach CUDA or Vulkan buffers. This ADR keeps it, as decided, but behind a measurement gate and outside M1.

## Revisions

- **Revised 2026-10-19.** ADR-0203 item 6 narrows item 2's fixed-buffer scope to the hot destinations; recorded here so the two ADRs agree.
- **Revised 2026-10-20.** Items 4 and 6: GPUDirect Storage (cuFile) and DirectStorage are dropped from v1, and a read-only capability report stays. NVIDIA's GDS troubleshooting guide (v1.19.1, checked 2026-10-20) documents direct mode for Tesla and Quadro GPUs only, so GeForce users would get the CPU bounce path Syntropy already builds (OQ-IO-02, answered). DirectStorage's gain is GPU decompression, and uncompressed quantised weights compress poorly (OQ-IO-07, dropped). D7 is still met: io_uring and IoRing read into registered memory, and unified memory gives Apple, Strix Halo and GB10 a direct NVMe path. TIER-E5-T20 and TIER-E5-T21 are removed from the build (RFC-0021 §9). Revisit if NVIDIA's GDS support list grows.
