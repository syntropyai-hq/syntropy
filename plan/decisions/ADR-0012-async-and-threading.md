# ADR-0012: Async and threading: tokio off the token path, pinned OS threads on it
- Status: Accepted (revised 2026-10-22)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0001, ADR-0006, ADR-0007, ADR-0011, ADR-0013, ADR-0015; RFC-0012, RFC-0020, RFC-0021, RFC-0023, RFC-0031, RFC-0040, RFC-0053

## Context
Decode time is spent in kernels, DMA, I/O submission and the data plane (ADR-0001). The host code between those must add as little as possible. Evidence:

- **Fixed per-token overhead.** A two-point fit on one GPU engine gives t_fixed ≈ 5 ms per token (`plan/research/hardware-and-performance-model.md` §3.4, estimate). Each extra device or backend on the token path adds its own launch and sync cost. At 30–90 tok/s, a 33–11 ms budget, host overhead is a first-order term.
- **CPU↔GPU hand-offs inside a layer.** A 16 KiB activation is ≈1 µs of PCIe wire time plus ≈5–20 µs of synchronisation (§3.2, estimate). With execute-in-place (ADR-0006), a MoE layer can hand off twice per token: GPU attention, then CPU experts for misses, then GPU again. 60 layers × 2 hand-offs × 20 µs is 2.4 ms per token if waits are slow.
- **Driver sleep/wake.** A driver sleep/wake round trip costs about 0.3 ms on Windows. A surveyed engine therefore spins a CPU thread on a pinned "doorbell" word that the GPU writes, and captures the whole pass as one CUDA graph per verify-window size, so the host makes no synchronising driver call per token (`plan/research/tiered-memory-inference.md` §5.1, verified code).
- **I/O submission cost.** One thread issuing blocking reads peaks at about 80K reads/s, ≈12 µs of kernel time per read, while an NVMe drive serves several times that at depth (`tiered-memory-inference.md` §10.3, verified code). Expert reads are 1–26 MiB (`hardware-and-performance-model.md` §2.4), so the drive needs queue depth 2–32, not thousands of submitting threads.
- **Execute-in-place uses the CPU's cores.** CPU-hosted experts reach only 0.24–0.35 of DRAM peak end to end (`hardware-and-performance-model.md` §0). Every core taken by bookkeeping is taken from the bandwidth-bound expert kernels.
- **The control side** (API, cluster gossip, iroh, downloads) is I/O-heavy and latency-tolerant. It wants async ergonomics and the mature Rust ecosystem, which is built on tokio.

## Options considered
1. **tokio everywhere, hot path included.**
   - Work stealing moves tasks between cores, which loses L2 and NUMA locality for the dispatcher and CPU kernels.
   - A task wake costs an OS park/unpark when the runtime is idle.
   - JSON parsing or TLS handshakes can share a thread with a token step.
   - CUDA and HIP contexts are thread-current, which clashes with tasks migrating between threads.
   - Rejected for hot paths.
2. **A thread-per-core async runtime built on io_uring for the I/O engine.**
   - Linux-only, and it ties registered-buffer and `O_DIRECT` alignment control to the runtime's buffer model.
   - Windows and macOS need different engines anyway (ADR-0013).
   - Rejected. The engine drives the rings directly in a plain loop.
3. **A work-stealing data-parallel pool for CPU kernels.** Good for prefill-sized GEMM, but it adds scheduling jitter to the decode path's small, latency-critical ops. Allowed off the token path only: pack building (ADR-0005), import and checksum.
4. **Separate OS processes per role** (I/O process, compute process). This puts IPC on every token. Rejected. ADR-0015 places every hot thread inside the runner.
5. **Chosen: tokio for control, network and API; dedicated pinned OS threads with no async runtime for compute dispatch, CPU kernels, the I/O engine and polled data-plane backends; lock-free SPSC rings between them.**

## Decision
1. **Async domain.** One multi-threaded tokio runtime per process runs the control plane, API, iroh control and bulk connections, store and downloads, and the daemon↔runner channel.
   - In a runner its thread count defaults to 2.
   - In the daemon it defaults to min(4, logical cores / 4).
   - No token-path code awaits.
2. **Hot domain: dedicated OS threads**, each created at runner start, pinned to a core on the NUMA node local to the device it serves, and run at the highest priority the OS grants without elevated privileges. The thread types:

| Thread | Count | Owns |
|---|---|---|
| Device dispatcher | 1 per GPU/NPU device | that device's context, streams or queues, captured graphs and the doorbell polling for that device. No other thread submits to it |
| CPU compute pool | 1 per CPU device (NUMA domain) | a static set of cores. Ops are split by a fixed per-core assignment and a barrier per op, with no work stealing. The ggml provider's CPU thread pool is given the same core mask |
| I/O engine | 1 thread per I/O core | one submission ring per thread (io_uring, IoRing or a portable equivalent, ADR-0013), shared-nothing. Default: one I/O core per NVMe volume that holds a tier, at most 2 per NUMA domain. Raised only when the probe shows submission-bound throughput |
| Data-plane poller | 1 per RDMA or shm backend in use | busy-polls completion queues and rings. QUIC data-plane edges run on a separate single-threaded tokio runtime pinned to its own core, so control traffic never shares it |

3. **"Thread-per-core" means shared-nothing threads that each own one core and one ring.** It does not mean one thread on every core. The cores left over after reservation belong to the CPU compute pool, and CPU capability profiles (RFC-0010) are measured with exactly that reservation in place.
4. **Hand-off.**
   - Every cross-thread queue on the token path is a bounded, preallocated, cache-line-padded single-producer single-consumer ring.
   - Fan-in is a set of SPSC rings polled by the one consumer. There are no MPMC queues and no mutexes on the token path.
   - Rings into the async domain signal through one eventfd or event handle per ring, so tokio is woken without polling.
5. **Waiting.**
   - Hot consumers spin for a bounded budget, then park on the OS address-wait primitive. The default budget is 50 µs, and 10 µs on battery. *Revised 2026-10-22 (below): on Apple platforms the park is a kernel wait on a dispatch semaphore, not an address wait.*
   - GPU→CPU completion inside a step uses a pinned host doorbell word written by the device, not a driver event wait.
6. **Invariants, enforced in CI.**
   - `syntropy-exec`, the engine module of `syntropy-io` and every `syntropy-kernels-*` crate have no dependency on tokio. A `cargo tree` check gates merges.
   - A steady-state decode step makes no heap allocation and takes no lock. This is checked by a counting allocator and lock instrumentation in `syntropy-testkit`.

## Consequences
- **Positive.**
  - Token-path latency is bounded by spin budgets and rings, not by a scheduler's choices.
  - Device contexts never migrate between threads.
  - I/O submission scales by adding rings, not threads.
  - Control code keeps the ecosystem's async libraries.
- **Negative.**
  - Reserved cores are not available to CPU kernels. On an 8-core laptop, one dispatcher, one I/O core and two tokio threads leave 4–5 cores for compute.
  - Spinning costs power. That is why the battery budget is lower and why the dashboard reports spin time.
  - There are two concurrency models to learn, and debugging crosses the ring boundary. Every ring carries a trace ID.
- **Work created.**
  - RFC-0012: the dispatcher loop, graph capture per window size and doorbells.
  - RFC-0021: the I/O engine threads and ring sizing.
  - RFC-0020 and RFC-0023: tier-manager and expert-cache actions as ring messages.
  - RFC-0031: the data-plane pollers and the pinned QUIC runtime.
  - RFC-0040: how a `KernelProvider` receives its thread and core assignment.
  - RFC-0053: per-thread lock-free trace buffers, drained by the async domain.

## What would make us revisit
- A same-hardware A/B on Tier 1 CI shows a tokio-hosted dispatcher within 2% of the dedicated-thread design on t_fixed and p99 step time. The extra model would then not be paying for itself.
- Core reservation costs more than 10% of measured CPU W_eff on 8-core hosts. Merge the I/O and data-plane threads on small hosts.
- A device runtime gains a host-callback mechanism with lower measured latency than doorbell polling.
- Resolved (2026-10-19): "one io_uring per core" means shared-nothing I/O threads, each owning its core and ring, sized per NVMe volume, rather than a ring on every core. A ring on every core would take cores from execute-in-place experts with no throughput gain at 1–26 MiB request sizes.

## Revisions

- **Revised 2026-10-22 (DIST-E1-T01).** Item 5 on Apple platforms. The `spsc` ring's hot consumer parks with `std::thread::park`. On Linux and Android std parks on a futex, and on Windows on `WaitOnAddress`: both address waits. On Apple platforms std parks on a dispatch semaphore, because the address-wait calls there (`__ulock_wait`, `__ulock_wake`) are private API. That is std's source at 1.99.1, `library/std/src/sys/sync/thread_parking/`. So on macOS the park is a kernel wait, not an address wait. It keeps the item's aim: a bounded spin, then a sleep in the kernel until the producer wakes the thread. `os_sync_wait_on_address`, public from macOS 14.4, is a later option for a true address wait there, with std's parker as the fallback on older releases. Maintainer decision, 2026-10-22: accept std's parker. RFC-0001 §18.18 records it.
- **Revised 2026-10-23 (PLC-E7-T01).** Item 2 on Apple platforms. macOS and iOS give a thread no way to be bound to a core: Apple silicon ignores the `THREAD_AFFINITY_POLICY` tags that exist, and there is no other affinity call. There, a hot thread runs unpinned, and the OS places it. It asks for QoS class `USER_INTERACTIVE`, the highest scheduling class a thread can take without elevated privileges, and its `HotThreadReport` gives no core, NUMA node 0 and the reason. Apple silicon has one memory node, so no thread is ever off its device's node. Elsewhere the item stands: on Linux and Android, `sched_setaffinity` pins, and the priority is the lowest nice value `RLIMIT_NICE` allows, usually the starting 0. On Windows, `SetThreadAffinityMask` pins at `THREAD_PRIORITY_TIME_CRITICAL`, within the process's primary group, the processor group of up to 64 logical processors that its threads share by default. Each core's NUMA node is looked up in that group with `GetNumaProcessorNodeEx`. On a host with more than 64 logical processors, hot threads use only that group's cores. Before Windows 11, a process with threads in several groups gets no affinity mask, so the topology call fails with that reason rather than reporting no cores. The calls are `syntropy_io::os`'s (RFC-0001 §4.2 rule 8, note of 2026-10-22).
