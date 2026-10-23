# RFC-0021: Weight I/O path

- Status: Draft
- Area: tiers (I/O engine)
- Depends on: RFC-0001 (§2.3, §3.3, §8.2–§8.4, §9.1, §9.4), RFC-0010, RFC-0020, RFC-0022, RFC-0023, RFC-0040, RFC-0041, RFC-0052; ADR-0005, ADR-0007, ADR-0012, ADR-0013, ADR-0014, ADR-0020, ADR-0203, ADR-0204
- Date: 2026-10-19
- Revised: 2026-10-20: GPUDirect Storage and DirectStorage dropped from v1 (§9); reads never reach punched source ranges (§6); per-volume write accounting from SMART (§13); open questions 3 (FileVault part), 6 and the `RowLookup` question answered. 2026-10-21: destinations and buffer registration go through `unsafe` (§2); §2's interface details as `syntropy-io` defines them

## Summary

The I/O engine moves bytes between `Nvme` and memory the tier manager owns: pack extents into weight arenas at load, missed and predicted experts into cache slots, row-lookup rows a step ahead, and KV pages and session blobs out and back. It never reads through the OS page cache for anything the plan places in a fast tier (ADR-0007), and it costs NVMe at the sustained, throttled rate it actually measures (ADR-0013).

This RFC specifies the `IoEngine` interface and its six engines (`UringDirect`, `PreadPool`, `IoRingEngine`, `IocpEngine`, `MacNoCache`, `MacMapped`), a read-only capability report for GPUDirect Storage and DirectStorage (neither is built in v1, §9), the I/O threads and their priority scheduling, the alignment contract with the pack format, queue-depth control, the staging pipelines that carry bytes from NVMe through pinned memory into VRAM, thermal-throttling detection, writes, integrity checks and the per-OS matrix.

The main decisions: io_uring with `O_DIRECT` is the Linux default and the blocking `pread` pool is the fallback everywhere a ring is unavailable; `SQPOLL` and `IOPOLL` stay off unless measured; fixed (registered) buffers are used only for the hot destinations, because they are charged to `RLIMIT_MEMLOCK` (ADR-0203). Demand reads go ahead of every prefetch, and prefetch is bounded in bytes in flight so a demand read never waits behind more than a few milliseconds of prefetch. Reads on the token path are not re-hashed; integrity is checked when a pack is committed, after an unclean shutdown, and by an idle-time scrubber (ADR-0204).

## Motivation and goals / non-goals

NVMe is a capacity tier, not a bandwidth tier: every GB of dense overflow costs ≈170 ms per token at 6 GB/s, and MoE over NVMe works only with a high RAM hit rate and predicted misses read in whole-expert units (`hardware-and-performance-model.md` §0, §3.3). The surveyed field shows how much is lost between the drive's rating and what an engine gets: expert-sized random reads through the page cache delivered ≈1.5–2.6 GB/s from a Gen4 drive rated 7.45 GB/s, and ≈2 GB/s from a 17.5 GB/s Apple SSD (`tiered-memory-inference.md` §11). One thread issuing blocking reads tops out at ≈80K reads/s (§10.3). Gen5 drives without heatsinks throttle within seconds (`hardware-and-performance-model.md` §2.4).

**Goals**

1. **Expert reads at ≥ 0.85 × the probe's sustained, throttled rate** at the same request size and queue depth (SC-7).
2. **Demand first.** A demand read is submitted ahead of every queued prefetch and waits behind at most `io.prefetch_inflight_max` bytes of prefetch already in flight.
3. **Run everywhere.** Every sandbox that blocks io_uring (Docker's default seccomp profile, Android, ChromeOS, the `io_uring_disabled` sysctl) still works through the pool; Windows 10 works through IOCP; filesystems that refuse direct I/O work in a flagged buffered mode.
4. **No page cache for hot data,** and no CPU copy between the drive and its destination, except where a measured A/B chose `MacMapped` or a source extent is read around because it is not yet packed.
5. **Honest rates for the solver.** Each volume's engine, queue depth, chunk size and sustained rate come from measurement and are refreshed when they drift.
6. **No allocation and no lock on the I/O thread's hot loop** after `Warming`; submissions and completions travel on SPSC rings (ADR-0012).

**Non-goals**

- The pack's byte layout. RFC-0041 owns it; this RFC states the alignment contract it must meet (§6).
- Choosing what to prefetch. RFC-0023 (experts) and RFC-0022 (KV) decide; this RFC executes reads within their windows.
- Network transfer of weights or KV (RFC-0041, RFC-0031), though the staging pools are shared.
- GPU-initiated storage access, which needs custom NVMe drivers (ADR-0013 option 6).

## Design

### 1. Overview and ownership

`syntropy-io` (layer 1) contains the engines and the I/O thread loop. It depends only on `core`; io_uring, IoRing and IOCP bindings live only here (RFC-0001 §4.2 rule 6). There are no GDS bindings in v1 (§9); the capability report's cuFile and DirectStorage queries are loaded with `dlopen`/`LoadLibraryExW` by the probe and nothing reads through them.

```text
dispatchers (one per device) ──ReadRequest──▶ ┐
housekeeping (KV spill, sessions, scrub) ─────▶ ├─ per-producer SPSC rings ─▶ I/O thread (one per volume) ─▶ IoEngine ─▶ NVMe
loader (Loading only) ────────────────────────▶ ┘                                    │
dispatchers ◀── ReadCompletion (per-producer SPSC rings) ◀────────────────────────────┘
```

- **Who owns what.** The I/O thread owns its engine, its submission ring, registered files and registered buffer indexes. The tier manager owns every destination buffer; the I/O engine writes bytes into it but never changes residency. A completion goes back to the producer's dispatcher, which, as single writer, flips residency (RFC-0001 §8.3; RFC-0020 §7.2).
- **One I/O thread per NVMe volume that holds a tier,** at most two per NUMA domain, pinned to a core on the NUMA node of the volume's controller (`/sys/class/nvme/<ctrl>/device/numa_node` on Linux) (ADR-0012). A volume with both packs and KV spill files has one thread serving both.
- **Pack files** are opened once at `Loading`, by the I/O thread, with direct I/O, and registered with the engine where it supports registered files.

### 2. Interface

ADR-0013 fixes the shape: batched submission of reads into tier-manager-registered buffers, plus a completion poll. The full types:

```rust
// ---- producer-facing (RFC-0001 §8.3 ReadRequest / ReadCompletion) -------------------

pub struct ReadRequest {
    pub ticket: IoTicket,                  // producer-unique; echoed in the completion
    pub extent: ExtentRef,                 // (file index, offset, len), aligned per §6
    pub dest: IoDest,
    pub priority: IoPriority,
    /// Prefetch only: drop the request if it has not been submitted when the consumer
    /// reaches this point (RFC-0023 sets it to the target layer of the target step).
    pub useful_until: Option<StepLayer>,
}
pub enum IoDest {
    /// A slot in an IO_FIXED arena (RFC-0020 §2): engine buffer index plus offset.
    Fixed { buf: BufIndex, offset: u64 },
    /// Any tier-manager memory, aligned per §6; used for bulk Loading reads.
    Raw { addr: u64 },
}
pub enum IoPriority {
    Demand,        // the current step waits for it (glossary: demand read)
    PrefetchNear,  // predicted for the next layer or the next step
    Load,          // Loading and pack building
    PrefetchFar,   // predicted further ahead; first to be dropped
    Background,    // KV spill, session save, scrub
}
pub struct ReadCompletion { pub ticket: IoTicket, pub result: Result<Bytes, IoError>, pub latency: Nanos }

pub struct WriteRequest { pub ticket: IoTicket, pub extent: ExtentRef, pub src: IoDest, pub priority: IoPriority, pub sync: SyncMode }
pub enum SyncMode { None, DataSync, FullSync }   // fdatasync / fsync / FlushFileBuffers

// ---- engine-facing ------------------------------------------------------------------

/// One instance per I/O thread. Not Sync: it is driven only by its thread.
pub trait IoEngine: Send {
    fn kind(&self) -> EngineKind;   // UringDirect | PreadPool | IoRing | Iocp | MacNoCache | MacMapped
    fn caps(&self) -> EngineCaps;   // fixed buffers, fixed files, writes, cancel, max depth, buffered?
    fn open(&mut self, path: &Path, mode: OpenMode) -> Result<FileIx, IoError>;
    fn register_buffers(&mut self, regions: &[HostRegion]) -> Result<SmallVec<[BufIndex; 16]>, IoError>;
    /// Never blocks. Returns how many were accepted; the rest stay with the caller.
    fn submit(&mut self, batch: &mut SubmitBatch) -> Result<u32, IoError>;
    /// Non-blocking unless `wait` allows it; appends completions.
    fn reap(&mut self, out: &mut CompletionBuf, wait: Wait) -> u32;
    fn cancel(&mut self, ticket: IoTicket) -> CancelOutcome;   // NotFound | Cancelled | TooLate
    fn in_flight(&self) -> (u32, Bytes);
}
pub struct EngineCaps {
    pub fixed_buffers: bool, pub fixed_files: bool, pub writes: bool, pub cancel: bool,
    pub max_depth: u32, pub direct: bool,          // false = flagged buffered mode (§4.2)
    pub dio: DioAlign,                             // §6
}
pub struct DioAlign { pub mem: u32, pub offset: u32 }
```

`SubmitBatch` and `CompletionBuf` are preallocated per thread at `Warming`, so the loop never allocates.

**Destinations are named through `unsafe`** (revised 2026-10-21). An engine reads and writes whatever memory a request names. If safe code could choose that memory, or how much of it a transfer covers, a crate under `forbid(unsafe_code)` could direct a write into memory it does not own, and every engine's `// SAFETY:` comment would rest on code that cannot hold `unsafe` (RFC-0001 §4.2 rule 8; ADR-0001 item 4). So the obligation sits with the tier manager, which owns every destination (§1) and may use `unsafe`:
- `IoDest` is a struct with private fields, built only by `unsafe fn IoDest::fixed(buf, offset, len)` and `unsafe fn IoDest::raw(addr, len)`. The length is part of the destination, so the caller vouches for a range it chose. Engines read it through `IoDest::kind()`, which returns the plain `IoDestKind { Fixed { buf, offset }, Raw { addr } }`, and `IoDest::len()`.
- Each constructor's `# Safety` contract runs from construction, not submission: the range is the caller's, valid, and used by nothing else until the value is dropped unsubmitted or, once submitted, until the request's completion is reaped. The caller can't see when a forwarding crate submits, so a slot whose destination never comes back stays reserved; a leak is safe, an early reuse is not. For `fixed` the range must hold on every engine that has a buffer at index `buf`, because safe code picks the engine a request reaches. An engine never touches memory for a `buf` it has not registered, or past that buffer's end; it fails such an operation instead.
- `SubmitBatch::push`, the only way to an engine, refuses an operation whose extent is longer than its destination and returns it (`PushError::ExceedsDest`). That is a producer bug, of class `Internal` like `IO_MISALIGNED`. A batch is full only for `PushError::Full`.
- `IoDest` is neither `Copy` nor `Clone`, so `ReadRequest`, `WriteRequest` and `IoOp` are not either. A request moves into the batch, and safe code cannot resubmit a stale copy after its slot is reused. The rings still move requests without allocating.
- `register_buffers` is an `unsafe fn`: each region stays the caller's, and valid, until the engine is dropped. `HostRegion` stays plain data, defined in `syntropy-core` because `kernels` registers host regions too (RFC-0040 §2.1); `core` forbids `unsafe`, which suits a type with no obligations of its own.
- `EngineKind`, `DioAlign`, `FileIx`, `HostRegion`, `ExtentRef`, `StepLayer`, `Bytes` and `Nanos` are `syntropy-core` types, which `syntropy-io` re-exports. `core`'s `StorageCaps` and `tier`'s `ArenaPlan` serialise them, and `core` can depend on no other workspace crate (RFC-0001 §4.2 rule 1).
- `ReadCompletion` and the other types stay `Copy`.

**Interface details `syntropy-io` settles** (revised 2026-10-21). Where the block above is silent or differs:
1. `register_buffers` returns `Vec<BufIndex>`, not `SmallVec<[BufIndex; 16]>`. Registration runs at `Loading`, outside the loop.
2. A write completes as a `ReadCompletion`; `result` is the bytes written. From an engine, a count below the extent's length is a short transfer, which the I/O thread resubmits (§16).
3. `SubmitBatch` holds `IoOp { Read(ReadRequest), Write(WriteRequest) }`, one per chunk (§3.3). `submit` accepts a prefix and removes it from the batch, so what remains was not submitted.
4. `Wait::Until(Nanos)` is a relative timeout: block until a completion arrives or that much time passes.
5. `BufIndex` is 16 bits, the narrower of the native indexes (io_uring's `buf_index`; I/O rings take 32), so every value is valid on every engine.
6. `CancelOutcome::Cancelled` still produces a completion, carrying `IO_CANCELLED`. `TooLate` completes normally and the I/O thread discards it as stale.
7. *(2026-10-22, TIER-E5-T02.)* Two safe methods let the I/O thread chunk and coalesce without new `unsafe`. `IoDest::split_at(at)` gives two disjoint halves of the range. `IoDest::join(next)` joins two `Fixed` ranges of one buffer when `next` starts where the first ends. Each result names only bytes its inputs already vouched for. `Raw` ranges never join, because two adjacent ones may lie in different allocations.

### 3. Threads, rings, priorities and queue depth

#### 3.1 The I/O thread loop

```text
loop:
    drain producer rings round-robin into five priority queues (fixed-capacity, preallocated)
    drop PrefetchFar/PrefetchNear entries whose useful_until has passed (read from the consumer's progress word)
    split each new request into chunks of ≤ io.chunk (default 4 MiB; probe-tuned, §10)
    admit chunks to the engine under the depth rules of §3.2, highest priority first
    engine.submit(batch)
    engine.reap(completions, Wait::None)
    for each completion: when every chunk of a request is done, post ReadCompletion to the producer's ring
    if nothing is in flight and every ring is empty:
        spin up to io.spin (50 µs; 10 µs on battery), then park on the rings' doorbell (eventfd / event handle / futex)
    elif nothing can be submitted:
        engine.reap(completions, Wait::Until(next ring check, ≤ 100 µs))     // blocking inside the kernel is fine on this thread
```

Producers signal a parked I/O thread through one doorbell word per ring (ADR-0012 item 5). A dispatcher never waits for the I/O thread: it posts a request and carries on, and finds the completion when it next polls its ring.

*Note 2026-10-22 (TIER-E5-T02, as built):* `syntropy_io::IoLoop` runs this loop, and each producer holds an `IoClient`. The details below are choices this section leaves open:
- **Doorbells.** Each request ring has a doorbell word of its own, on its own cache line. To park, the loop sets every word, issues a sequentially consistent fence and drains the rings once more. A producer issues the same fence after its push and unparks the loop if its word is set. `syntropy-spsc`'s own wait parks a consumer on one ring, so the many-ring park lives in `io`.
- **Draining.** Each visit drains one producer's ring in full, starting at a different producer each pass. That keeps the rows a token needs, posted together, adjacent in their queue, so they can coalesce. A producer's requests in the loop, plus its completions it hasn't taken yet, never exceed its completion ring's capacity, so a completion always has a slot. The producer takes completions only through `IoClient`, which counts each one and rings the doorbell, so a loop parked on a full completion ring wakes when the producer frees a slot. A producer that doesn't take its completions is not drained.
- **Coalescing.** A read joins the read at the back of its queue when the file bytes and the `Fixed` destination bytes both continue that read, and the destination of the read already queued is exactly as long as its extent. A coalesced read holds at most the chunk size and 16 requests. A short transfer credits the requests in extent order.
- **Writes** are submitted whole, never chunked or coalesced, because their `SyncMode` covers the whole extent.
- **An extent longer than its destination** completes at intake with `IO_MISALIGNED` and never reaches the engine. §16 gives that producer bug no code of its own (open, for the maintainers).
- Not in this task: §3.2's depth rules beyond "highest class first", and the `useful_until` drops, are TIER-E5-T07. The one retry of a short transfer is TIER-E5-T15, so a short transfer completes with `IO_SHORT_READ` for now.

#### 3.2 Depth and priority rules

Per volume, the probe provides `qd_target`: the smallest queue depth that reaches ≥ 95% of the best throughput at the model's real request size (§10). Then:

| Class | May use | Notes |
|---|---|---|
| `Demand` | any free depth up to `io.qd_max` (64); on `PreadPool` and `MacNoCache` the effective depth is capped by the pool's thread count (16 and 8 by default) | always submitted first; if depth is full, it is the next submission when any chunk completes |
| `PrefetchNear` | up to `qd_target`, and at most `io.prefetch_inflight_max` (32 MiB) in flight while any `Demand` is queued or in flight | with no demand pending, prefetch may fill `qd_target`; what is worth reading is decided before submission by RFC-0023 §10's value gate and per-token byte cap, using this engine's `VolumeIoStats` |
| `Load` | up to `qd_target` | only during `Loading` and pack building, when no demand traffic exists; yields to `Demand` like prefetch otherwise |
| `PrefetchFar` | what `PrefetchNear` leaves, within `qd_target` | dropped first when the window is full (RFC-0001 §8.4 rule 4) |
| `Background` | 2 in flight, and only while the volume's demand-plus-prefetch traffic is below `1 − io.write_max_share` (0.25) of its sustained rate | KV spill, session saves, scrub |

In-flight I/O cannot be pre-empted, so the cap on prefetch bytes in flight is what bounds a demand read's wait. A demand read submitted behind 32 MiB of prefetch waits at most about `32 MiB / W_nvme`, ≈5.6 ms at 6 GB/s [estimate], and in practice less because the drive serves commands concurrently.

**Online depth adjustment.** If a volume's measured throughput stays below 0.85 × its sustained rate (SC-7) for 10 s while requests are queued, `qd_target` rises by 2, up to `io.qd_max`, and falls back by 2 if throughput does not improve within the next 10 s. Changes are logged with their measured effect.

#### 3.3 Chunking and coalescing

- **Experts and streamable units** are one contiguous, aligned extent each (RFC-0041). An extent larger than the chunk size is split into chunks submitted together, so one expert is read at depth: a 26.5 MB DeepSeek-V3 Q4_K_M expert is 7 chunks of ≤ 4 MiB [estimate].
- **Rows** (`RowLookup`) are read as the aligned block range covering each row (§6). The rows one token needs are submitted as one batch, typically 16 for the surveyed n-gram table (`tiered-memory-inference.md` §9). Adjacent blocks in the same batch are coalesced into one read. A token's row batch read one step ahead completes in less than a token time at QD 1 and row-bundle sizes (OQ-IO-09, answered 2026-10-20 from the cost model §2.4), so `RowLookup` tables above `solver.lookup_ram_max` live on NVMe and need no RAM cache; TIER-E5-T26 keeps the p99 as a regression row.
- **KV pages** are written and read in page runs (RFC-0022), one request per run.

### 4. Linux engines

#### 4.1 `UringDirect` (default)

**Availability.** `io_uring_setup` succeeds and `/proc/sys/kernel/io_uring_disabled` permits this process. `ENOSYS` or `EPERM` from setup (Docker's default seccomp profile, Android, ChromeOS, or the sysctl) selects `PreadPool` with no error to the user beyond a `doctor` note (ADR-0013 item 3).

**Ring setup.** One ring per I/O thread, 256 SQ entries, CQ size 512 (`IORING_SETUP_CQSIZE`). Flags are tried in order, dropping the newest on `EINVAL`: `IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN`, then `IORING_SETUP_COOP_TASKRUN`, then none. With `DEFER_TASKRUN`, completions are processed only when the I/O thread enters the kernel to reap, which suits a thread that owns its ring and reaps in its own loop.

**Files.** Pack files, the KV spill file and session blobs are opened `O_RDONLY|O_DIRECT|O_CLOEXEC` (writes: `O_RDWR|O_DIRECT`) and registered with `io_uring_register_files`; SQEs use `IOSQE_FIXED_FILE`.

**Buffers.** RFC-0020 marks arenas `IO_FIXED`: `IoStaging`, `ExpertCache` and `ExpertStaging` over an `Nvme` home, `RowCache`, and KV spill pages. They are registered with `io_uring_register_buffers` in buffers of at most 1 GiB each, within `io_fixed` from RFC-0020 §5, since registered buffers are charged to `RLIMIT_MEMLOCK` (ADR-0013). Reads into them use `IORING_OP_READ_FIXED`. Everything else, including bulk `Loading` reads into `Weights` arenas, uses `IORING_OP_READ` with `O_DIRECT` into the raw address; the kernel pins the pages for the duration of each I/O. If registration fails (`ENOMEM`, `EPERM`), the engine continues with `IORING_OP_READ` everywhere and reports the reduced efficiency in its caps.

**Writes and sync.** `IORING_OP_WRITE_FIXED` or `IORING_OP_WRITE`, then `IORING_OP_FSYNC` with `IORING_FSYNC_DATASYNC` for `SyncMode::DataSync`.

**Cancellation.** `IORING_OP_ASYNC_CANCEL` by user data for prefetches whose `useful_until` passes while in flight. A cancel that loses the race completes normally and the completion is discarded as stale (RFC-0020 §7.3).

**`SQPOLL` and `IOPOLL`** (ADR-0203). Both are off by default:
- `SQPOLL` costs a core spinning in the kernel. It is enabled for a volume only when the probe measures ≥ 15% more throughput with it at the real request mix *and* the core budget has a spare core after the compute pool's reservation (ADR-0012). Row-lookup-heavy models are the likely case. The SQPOLL thread is pinned with `IORING_SETUP_SQ_AFF` next to the I/O core.
- `IOPOLL` needs polled queues configured in the NVMe driver (`/sys/block/<dev>/queue/io_poll` = 1), which consumer installs do not have. It is used only when that is true and the probe shows a gain.

#### 4.2 `PreadPool` (universal fallback)

- **Shape.** The I/O thread keeps the priority queues and depth rules of §3 and hands chunks to `io.pool_threads` (16 by default) blocking threads over one SPSC ring each, round-robin; each pool thread runs `pread`/`pwrite` and posts its completion on its own SPSC ring back. No multi-producer queue is used. The pool threads are affined to the I/O core set; they spend almost all their time blocked in the kernel. Thread count equals maximum depth (ADR-0013).
- **Direct I/O.** Files are opened `O_DIRECT`. If `open` or the first aligned read returns `EINVAL` (tmpfs, some FUSE filesystems, WSL2's `/mnt/c` drvfs mounts), the file is opened buffered, every read is followed by `posix_fadvise(POSIX_FADV_DONTNEED)` on its range, and the volume is flagged `buffered` in the resource graph so the solver and `doctor` see it. On WSL2, `doctor` tells the user to keep the model store on the Linux filesystem.
- **Cancellation** removes queued chunks; a chunk already in a `pread` completes and is discarded.

*Note 2026-10-22 (TIER-E5-T03, as built):* `syntropy_io::pool::PreadPool` (Linux and Windows) and `MacNoCache` (§5) share one pool. Choices this section leaves open:
- **Depth.** The engine sends an operation only to a worker with nothing outstanding, so `max_depth` is the thread count and each worker's ring holds at most one operation, plus file handles sent at `open`.
- **tmpfs.** Since Linux 6.6 tmpfs accepts `O_DIRECT`, but its pages are the page cache. A file on tmpfs or ramfs (`fstatfs`) is therefore opened buffered and flagged on every kernel, before the `EINVAL` checks on `open` and on the first aligned read (one 4 KiB block at offset 0, read at `open`).
- **Windows.** Synchronous I/O on one handle is serialised, so each worker opens the file itself with `FILE_FLAG_NO_BUFFERING` and issues positional `ReadFile`/`WriteFile`. The buffered fallback there is flagged only; Windows has no per-range cache drop.
- **The flag.** A buffered file sets `EngineCaps::direct = false` for the engine, which the probe copies into `StorageCapsExt::buffered`; `buffered_reason()` returns the `IO_DIRECT_REFUSED` error with its OS error.
- **Errors.** A transfer's `EINVAL` (`ERROR_INVALID_PARAMETER`) is `IO_MISALIGNED`, `ENOSPC`/`EDQUOT` (disk-full on Windows) `IO_NO_SPACE`, anything else `IO_SHORT_READ`. A file that cannot be opened has no code in §16; it is `IO_EXTENT_BAD` with the OS error (open, for the maintainers). A `Fixed` destination outside the registered buffers, or an unknown file index, completes with `IO_MISALIGNED` and is never touched.
- **Cancellation** marks an operation no worker has started, which then completes with `IO_CANCELLED` and no system call; one a worker has started is `TooLate`.
- **Alignment.** Each backend calls TIER-E5-T05's `align::query` when it opens a file, and `IoEngine::file_dio` returns that file's alignment. `EngineCaps::dio` stays zero (the 4096-byte floor), since the alignment is per file (§6).
- Not in this task: pinning workers to the I/O cores is the runner's.

### 5. macOS engines

ADR-0013 defines two engines and chooses between them per volume by measurement (§10).

**`MacNoCache`.** Files are opened read-only and `fcntl(F_NOCACHE, 1)` is set before the first read, because `F_NOCACHE` does not purge pages already cached; `fcntl(F_RDAHEAD, 0)` turns off read-ahead. A pool of `io.pool_threads_macos` (8 by default) threads issues `preadv` with 16 KiB-aligned destinations in the `Unified` arena, in requests of at least 32 KiB; smaller rows are coalesced up to that size. On Apple silicon the destination is GPU-visible as allocated, so a read is the only copy (RFC-0020 §9).

*Note 2026-10-22 (TIER-E5-T03, as built):* `MacNoCache` sets `F_NOCACHE` and then `F_RDAHEAD` in `open`, before any worker holds the file. "Requests of at least 32 KiB" is met by gathering: reads in one submission batch that continue one another in the file are issued as one `preadv` of up to 16 destinations, until it holds at least 32 KiB. A read that nothing continues is issued at its own size. A cancel of a gathered read is `TooLate`. A refused `F_NOCACHE` leaves the file buffered and flags the volume, as §4.2 does for `O_DIRECT`; a refused `F_RDAHEAD` is ignored.

**`MacMapped`.** The pack file is mapped read-only (`mmap(PROT_READ, MAP_SHARED)`) with `madvise(MADV_RANDOM)`. A read is `madvise(MADV_WILLNEED)` on the extent when it is requested, followed by a copy from the mapping into the destination slot by a pool thread once the pages are resident (`mincore` polled at 50 µs). After the copy, the mapping's range is released with `madvise(MADV_DONTNEED)`; the clean file pages that remain belong to the OS's cache, are reclaimable without compression, and are never counted as residency. This keeps residency owned by the tier manager (ADR-0007 item 6): `MacMapped` is a read mechanism, not a cache. Its extra cost is one memory copy per byte read, which the A/B includes.

*Note 2026-10-23 (TIER-E5-T09, as built):* `syntropy_io::pool::MacMapped` runs on the pool of §4.2 with `io.pool_threads_macos` workers. Choices this section leaves open:
- **The mapping.** `open` for reading maps the whole file at its length then, and marks it `MADV_RANDOM`. A read covers the pages of its extent: `MADV_WILLNEED`, `mincore` polled every 50 µs, the copy, then `MADV_DONTNEED` on the same pages, which may include a page a concurrent read shares at an edge. That read faults the page in again; its bytes stay right.
- **The wait is bounded** at 20 ms [target] (`RESIDENT_WAIT_MAX`, a chosen budget). Past it, or if `mincore` fails, the read is a `pread` of the file, which returns an I/O error as §16 expects. A copy that faults a page in and meets an I/O error raises `SIGBUS` instead.
- **Ends of file.** A read from or past the mapped length returns 0 bytes; one that crosses it returns the bytes before it, as `pread` does. Committed packs do not change (RFC-0041), so the length holds. A file truncated while mapped raises `SIGBUS` on the pages it lost.
- **Writes** go through `F_NOCACHE` as in `MacNoCache` (§13). A file opened for writing is not mapped, since it grows, and is read with `pread` on that handle. An empty file is not mapped either, and reads of it return 0 bytes.
- **A refused `mmap`** fails `open` with `IO_ENGINE_UNAVAILABLE` and the OS error, so `MacNoCache` reads that file. A mapped file is never the flagged buffered mode: `EngineCaps::direct` changes only for a file opened for writing whose `F_NOCACHE` is refused.
- **Residency.** A completion reports only the bytes copied into the slot, and the engine has no other report. RFC-0020 §3 keys a home's `Loading` state to the read's `IoTicket`, so the residency map (TIER-E1-T04) learns of a read only from its completion and never records a mapped page. The unit check is the engine's half: every range advised in is released once its read completes.
- Contiguous reads are not gathered, since the copy costs the same per byte at any size.

### 6. Alignment rules and the pack contract

**Volume alignment.** Per open file, the engine determines `DioAlign { mem, offset }`:

| OS | Source | Fallback |
|---|---|---|
| Linux | `statx(STATX_DIOALIGN)`: `stx_dio_mem_align`, `stx_dio_offset_align` (Linux 6.1+) | the block device's `queue/logical_block_size` for both; filesystems that return 0 (direct I/O unsupported) go to buffered mode (§4.2) |
| Windows | `GetFileInformationByHandleEx(FileStorageInfo)`: `PhysicalBytesPerSectorForPerformance` for offsets and lengths; buffer alignment to the same value | `IOCTL_STORAGE_QUERY_PROPERTY` (`StorageAccessAlignmentProperty`) |
| macOS | `F_NOCACHE` has no alignment rule; Syntropy uses the 16 KiB VM page | — |

`A = max(4096, offset)` and `M = max(4096, mem, OS page size)`.

**What the pack guarantees** (RFC-0041 implements; ADR-0005 item 3):

1. Every `Expert` group, every streamable unit (a group with a `StreamSite` or a `Vram` home), every KV-page-sized unit and every row bundle starts at a multiple of `A_pack`, which is 4 KiB, or the volume's `A` when that is larger at pack-build time.
2. Every such unit's length is padded to a multiple of `A_pack`, so one read of `len` bytes from `offset` is legal with direct I/O.
3. An expert's gate, up and down matrices are contiguous, so one extent is one expert.

**What the tier manager guarantees** (RFC-0020): every `IoDest` is aligned to `M`, and every slot length is a multiple of `A_pack`.

**Rows.** Row tables are not padded per row, since padding a table of small rows to 4 KiB each would multiply its size. A row is read as the aligned block range `[⌊off/A⌋·A, ⌈(off+len)/A⌉·A)` into a `RowCache` slot, and the consumer reads the row at its offset inside the slot. Read amplification is at most `(len + 2A − 2)/len`; rows are IOPS-bound, not bandwidth-bound, so this costs little (§Performance).

**Read-around for unpacked extents.** When a unit is not yet packed, or the pack exceeds its disk budget (ADR-0005 item 4), the engine reads the aligned window covering the source extent into an `IoStaging` slot and the I/O thread copies the payload into the destination. The read efficiency `len / window_len` is reported per volume to the capability profile, so the solver sees the cost.

**Punched source ranges.** Once a SOURCE-layout pack extent is committed, the store may punch the matching source ranges when free space would otherwise fall below its threshold (RFC-0041 §4.5 owns the rule, the range map and `store.min_free_bytes`, with `store.punch_covered_sources = auto | never`). The I/O path keeps two rules so a punched range is never read: (1) read-around and peer-serving reads of a source range resolve through the store's range map first, and a range that a committed extent holds is read from the extent; (2) the store asks the volume's I/O thread to punch, and the thread issues the punch as a `Background` operation only after the range map has switched and every in-flight read of that range has completed (`fallocate(FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE)` on Linux, `FSCTL_SET_ZERO_DATA` on a sparse file on Windows, `F_PUNCHHOLE` on APFS). A read that would still reach a punched range is a bug, rejected before submission with an `Internal` error, like a misaligned request.

**Checks.** Every request is checked against `A` and `M` before submission. A misaligned request is a bug (`IO_MISALIGNED`, `Internal`) and is rejected before it reaches the kernel, never silently bounced.

*Note 2026-10-22 (TIER-E5-T05, as built):* `syntropy_io::align` holds this section. The details below are choices it leaves open:
- **Query.** `align::query(file, path)` returns `Aligned { dio, source }`, `Unsupported` (`statx` reports zero for both values, so the file goes to buffered mode) or `Unknown` (nothing answered; the 4 KiB floor applies). On Linux `statx` is called through `syscall`, and the fallback reads `/sys/dev/block/<major>:<minor>/queue/logical_block_size`, or the parent device's for a partition. On Windows the fallback opens the file's volume with no access rights for `StorageAccessAlignmentProperty` and uses its physical sector size. Engines record `query(..).dio()` per file at `open` and return it from `IoEngine::file_dio`; the pool engines of TIER-E5-T03 report nothing per file until they call it, so the floor applies to them.
- **`A` and `M`.** `RequestAlign::from_dio` takes the least common multiple where the table takes the maximum. The two agree for powers of two, which is every value an OS reports; for any other value the multiple keeps a request aligned to every part.
- **The check** runs in `IoLoop` at intake and again as each chunk joins the batch, so a failing request never reaches the engine; `IoLoop::misaligned()` counts the rejections. It uses the request's own file's alignment, read at each request: `IoEngine::file_dio(file)`, which an engine records at `open`, or else the engine's current `EngineCaps::dio`. A file opened after the loop was built is therefore checked against its volume. A `Fixed` destination is checked by its offset inside the registered buffer; the buffer's start is the tier manager's guarantee. Each request's chunks are `io.chunk` rounded down to a multiple of its file's `A` and `M`, so chunks of an aligned request stay aligned.
- **The pack rule** is `RequestAlign::pack_extent`, which RFC-0041's pack builder is to call. Rows and read-around use `RequestAlign::window`. `ReadAroundStats` counts each volume's payload and window bytes and writes their ratio into `StorageCapsExt::read_around_efficiency`. Copying the payload out of the `IoStaging` slot, and calling `report` when the profile is built, belong to the read-around consumer (COMP-E11-T08) and the probe (PLC-E2-T08, TIER-E5-T10): the loop holds no pointer to a `Fixed` buffer.

### 7. Windows engines

**`IoRingEngine`** (Windows 11, build 22000 and later):

- `QueryIoRingCapabilities` gives the highest supported version; `CreateIoRing` is called with that version, 256 submission entries and 512 completion entries. `IsIoRingOpSupported` is checked for every operation used. Writes and flushes are used only where the ring reports them supported; otherwise writes go to the IOCP engine on the same thread (one engine for reads, one for writes).
- Files are opened with `CreateFileW(GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING)` and registered with `BuildIoRingRegisterFileHandles`; `IO_FIXED` arenas are registered with `BuildIoRingRegisterBuffers` (each `cudaHostRegister`ed lock range of RFC-0020 §4.2 is one registered buffer).
- Reads are `BuildIoRingReadFile` with registered handle and buffer references; `SubmitIoRing` submits without waiting; `PopIoRingCompletion` reaps. A completion event set with `SetIoRingCompletionEvent` is the I/O thread's park handle.
- Cancellation: `BuildIoRingCancelRequest`.

**`IocpEngine`** (Windows 10, and wherever IoRing is unavailable or lacks an operation):

- Files are opened with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and associated with one completion port per I/O thread.
- Reads are overlapped `ReadFile` calls, which return immediately, so one thread reaches depth; completions are reaped with `GetQueuedCompletionStatusEx`. If the probe finds one thread submission-bound, up to `io.iocp_threads − 1` helper threads (default total 4, ADR-0013) take submission batches over SPSC rings.
- Cancellation: `CancelIoEx` on the request's `OVERLAPPED`.

Unbuffered I/O on Windows requires sector-aligned offsets, lengths and buffers; §6 provides them.

*Note 2026-10-23 (TIER-E5-T08, as built):* `syntropy_io::windows` holds both engines. Choices this section leaves open:
- **Loading.** The IoRing functions are resolved at run time from `api-ms-win-core-ioring-l1-1-0.dll`, and `BuildIoRingWriteFile` and `BuildIoRingFlushFile` from `kernel32.dll`. A binary that links the crate therefore starts on a Windows without them, where `IoRingEngine::new` returns `IO_ENGINE_UNAVAILABLE` and `IocpEngine` is used.
- **Files.** Both engines open files with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and `FILE_SHARE_READ | FILE_SHARE_WRITE`, so the IOCP engine that takes a ring's writes can open the same file. A file system that refuses unbuffered opens (`ERROR_INVALID_PARAMETER`) gets a buffered handle, and `EngineCaps::direct` is false.
- **Registration.** Files and `IO_FIXED` regions are registered while nothing is in flight, the whole table each time; one asked for while operations are in flight is made by the first `reap` that finds the ring idle. `IORING_BUFFER_INFO`'s length is 32 bits, so a larger region is registered in 2 GiB pieces, and an operation that crosses a piece's end uses its raw address. A registration that fails leaves raw handles and addresses (`IO_REGISTER_FAILED` continues unregistered).
- **Writes.** The ring writes when `IsIoRingOpSupported` reports both write and flush. A write with a sync mode is followed by `BuildIoRingFlushFile` (`FILE_FLUSH_DATA` for `DataSync`) and completes when the flush does. Otherwise writes go to an `IocpEngine` the ring engine owns, which flushes with `FlushFileBuffers` on the I/O thread when the write's completion is reaped. That flush blocks the I/O thread, so every completion behind it, demand reads included, waits for it; only `Background` pack and session writes ask for a sync (§13).
- **Depth.** The ring holds half its completion entries in flight, 256, since each operation may also have a cancellation's completion queued. One call moves at most 1 GiB; a longer operation completes short (§16).
- **Waiting.** `reap` blocks on the ring's completion event or in `GetQueuedCompletionStatusEx` only for whole milliseconds. A shorter wait polls, because a Windows timer wait rounds up to the timer tick.
- **Cancellation.** Both engines ask the OS (`BuildIoRingCancelRequest`, `CancelIoEx`) and return `CancelOutcome::TooLate`: the operation is in the kernel, and completes normally or with `IO_CANCELLED`.
- **Helper submitters.** `IocpOptions::helpers` threads issue the calls, taking slots over SPSC rings in turn; a call that fails at once is posted to the port with `PostQueuedCompletionStatus`. The engine takes the count from its caller; deciding that one thread is submission-bound is §10's selection.

### 8. Pinned staging to GPU: the DMA pipelines

Discrete GPUs cannot read `Nvme` directly in the default path, so bytes travel NVMe → pinned host memory → VRAM. The two halves are owned by different threads: the I/O thread fills host memory, and the GPU's dispatcher issues the copy on its copy stream (`KernelProvider::copy`), because only that thread may touch the device (RFC-0001 §8.2).

**Pipelines:**

| Flow | Path | Staging | Visibility |
|---|---|---|---|
| `Vram`-homed group at `Loading` | NVMe → `IoStaging` slot → H2D copy → `Weights` arena | `io.stage_slots` (4) slots of `io.stage_slot` (64 MiB) per GPU, a ring: slot k is refilled by the I/O thread once its copy's event has fired | the group is `Resident` when its last chunk's copy event fires |
| Expert admission into VRAM, RAM-homed (copied miss or cache fill, RFC-0023) | RAM home (`DmaPinned`) → H2D copy → `ExpertCache` or `ExpertStaging` slot | none | RFC-0023 publishes the slot-table entry after the copy event |
| Expert admission into VRAM, `Nvme`-homed, resident in the RAM cache | RAM cache slot → H2D copy → VRAM slot | none | as above |
| Expert prefetch or admission into VRAM, `Nvme`-homed, not in RAM | NVMe → `ExpertStaging` host slot → H2D copy → VRAM slot (the VRAM and RAM caches hold disjoint sets, RFC-0023 §3) | a host staging slot | as above |
| Demotion of an `Nvme`-homed VRAM victim (RFC-0023 §3) | D2H copy → RAM-cache slot, before the slot is overwritten | none | the RAM copy is `Resident` when the D2H event fires |
| Expert into RAM only (CPU computes it in place) | NVMe → RAM cache slot (`IO_FIXED`) | none | the dispatcher flips the copy to `Resident` on completion; the CPU pool sees it through the step's miss list |
| `Unified` devices | NVMe → `Unified` slot directly | none | on completion |
| Prefill streaming of `HostPinned` groups (RFC-0020 §10) | RAM home → H2D into a VRAM stream buffer | stream buffers in `Scratch` | device-side event wait before compute |

**Copy issue rules** (with RFC-0023):

- On CUDA 13 a step's queued admission copies are issued together with `cudaMemcpyBatchAsync`; elsewhere as a loop of async copies on the copy stream (ADR-0014).
- Copied misses on the critical path are issued before admission copies in the same step. Admission copies are limited per step to `ρ · W_pcie · t_step` bytes (ρ = `io.admit_pcie_share`, default 0.25), so cache filling never delays a copied miss by more than that share of the link.
- A copy into a slot waits device-side for that slot's `free_after` fence (RFC-0020 §7.4).

### 9. GPUDirect Storage and DirectStorage: not in v1

**Revised 2026-10-20 (ADR-0013 items 4 and 6, ADR-0203 item 7).** Neither path is built in v1. NVIDIA's GDS troubleshooting guide (v1.19.1, checked 2026-10-20) documents GDS direct mode for Tesla and Quadro GPUs only, so GeForce users would get compatibility mode, a CPU bounce through host memory, which is the path §8 already builds (open question 6, answered). DirectStorage's gain is GPU decompression, and uncompressed quantised weights compress poorly (OQ-IO-07, dropped). D7 is still met: io_uring and IoRing read into registered memory, and unified memory gives Apple, Strix Halo and GB10 a direct NVMe path (§12).

**Capability report (read-only).** The probe records per (volume, GPU), in `StorageCapsExt::direct_paths`, without reading model data through either API: on Linux with NVIDIA, whether `libcufile` loads, whether `cuFileDriverOpen` succeeds, and whether `cuFileDriverGetProperties` reports a peer-to-peer path or compatibility mode for the volume's filesystem; on Windows, whether the DirectStorage runtime is present. `syntropy doctor` shows it. It exists so the revisit condition (NVIDIA's documented GDS support list growing to GeForce cards) can be checked on real machines, and a gated cuFile path can be re-added then with a ≥ 10% A/B requirement.

The design below is kept for that revisit and is **not built in v1**.

**GPUDirect Storage** (Linux, NVIDIA; M3):

- Enabled for a (volume, GPU) pair only if `cuFileDriverOpen` succeeds, `cuFileDriverGetProperties` reports the kernel module loaded and a peer-to-peer path for the volume's filesystem rather than compatibility mode, and the GPU's BAR1 aperture (`nvmlDeviceGetBAR1MemoryInfo`) can map the expert arena. On consumer GeForce cards the docs give no assurance of a direct path, so the expected outcome there is "off" unless the probe proves otherwise (`hardware-and-performance-model.md` §2.4).
- `cuFileHandleRegister` on the `O_DIRECT` descriptor and `cuFileBufRegister` on the VRAM `ExpertCache` arena's segments. Reads use `GdsPort::read_async`, implemented by `syntropy-kernels-cuda`, which enqueues `cuFileReadAsync` on the copy-in stream; never the blocking `cuFileRead` on the dispatcher thread (RFC-0040 §7.7; RFC-0001 invariant 9). The I/O thread schedules them with the same priorities as any other read and posts the enqueue request to the dispatcher, whose copy-in stream carries it; completion is the stream's event, polled like any copy.
- It must beat `UringDirect` → pinned → copy by ≥ 10% at the real request size in the probe's A/B; otherwise it stays off.

**DirectStorage** (Windows; not before M3, never default): D3D12-centred, measured ≈3 GB/s uncompressed on Gen3, and helpful mainly for GDeflate-compressed assets (ADR-0013). Using it for CUDA or Vulkan destinations needs D3D12 shared heaps imported into the other API. It is built only if an M3 spike shows a ≥ 10% gain over `IoRingEngine` → pinned → copy on a Tier 1 Windows rig; until then the probe does not consider it.

### 10. Engine selection by measurement

RFC-0010 runs the I/O probe; this RFC defines what it measures and what it records. Per (node, volume holding a pack):

1. **Candidates**: every engine available on the OS (§4, §5, §7). GDS and DirectStorage are reported, never candidates (§9).
2. **Request mix**: the census's real sizes for the model being loaded: one expert extent, one streamable unit, one row bundle, one KV page run. Without a model (first probe), the defaults are 4 KiB, 1 MiB, 8 MiB and 32 MiB.
3. **Quick probe** (≈5 s within SC-22's 15 s budget): reads of the pack file itself (or the source file before packing), at QD 1, 4 and 16, for each candidate; no writes to the user's SSD.
4. **Choice**: the highest throughput at the dominant request size; within 5%, the lower CPU time per GB (`getrusage` or `GetThreadTimes` on the I/O thread).
5. **Tuning**: `qd_target` (the knee, §3.2), `chunk` (the best of 1, 2, 4 and 8 MiB at `qd_target`), and whether `SQPOLL` pays (§4.1).
6. **Sustained run**: 60–120 s of reads at `qd_target` in the background after the first `Loading` completes, or in an idle window, never while `Packing` or `Loading` use the volume (RFC-0010 S2). It stores the throttled rate as `StorageCaps::sustained_read` (RFC-0001 §5.2). The first token never waits for it (SC-22).

**Outputs to RFC-0010.** Engine selection produces the `StorageCaps` fields below, once per probe. During operation, every I/O thread publishes one `VolumeIoStats` record per volume per second and a `ThrottleEvent` on every state change; both are defined in §11 and are what RFC-0010's aggregator uses for the sustained rate and `ThrottleInfo` (RFC-0010 §6.3). The I/O engine never sends `ReportProfile` itself; RFC-0010 decides that.

`StorageCaps` gains these fields (serde defaults):

```rust
pub struct StorageCaps {
    // RFC-0001 fields: tier, sustained_read, burst_read, extent, queue_depth, small_read, temperature_c
    #[serde(default)] pub engine: EngineKind,
    #[serde(default)] pub chunk: Bytes,
    #[serde(default)] pub dio: DioAlign,
    #[serde(default)] pub buffered: bool,          // §4.2 flagged mode
    #[serde(default)] pub sqpoll: bool,
    #[serde(default)] pub read_around_efficiency: Option<f32>,   // §6
    #[serde(default)] pub shared_lanes: bool,      // chipset-attached (cost model §2.4); from RFC-0010 topology
}
```

### 11. Thermal throttling and drift

MoE expert streaming is a sustained read load for hours, and Gen5 drives without heatsinks throttle within about a dozen seconds (`hardware-and-performance-model.md` §2.4). Two detectors run on the I/O thread's statistics, published to the housekeeping thread every second:

- **Temperature.** Linux: the NVMe hwmon device (`/sys/class/nvme/<ctrl>/hwmon*/temp1_input`, with `temp1_max` and `temp1_crit` where exposed). Windows: `IOCTL_STORAGE_QUERY_PROPERTY` with `StorageDeviceTemperatureProperty`, on a handle opened with no access rights. macOS: no public interface; throughput only.
- **Throughput.** Busy-time throughput: bytes read divided by the time with at least one read in flight, in 1 s windows. The detector compares the 10 s moving average with `sustained_read` only while the request mix is unchanged: the same dominant size bucket and a mean queue depth within ±25% of `qd_target`. Windows with a different mix are published but not used for state changes.

**State machine** (per volume):

```text
Normal ──10 s average < 0.80 × sustained_read for 20 consecutive comparable windows──▶ Throttled
Throttled ──10 s average > 0.90 × sustained_read for 60 consecutive comparable windows──▶ Normal
any ──temperature ≥ warning − io.thermal_margin──▶ also flagged near_thermal_limit (cleared 3 °C below)
```

**Output.** Published by the I/O thread to the housekeeping thread, which forwards it to RFC-0010's aggregator and the telemetry ring:

```rust
/// One per volume per 1 s window.
pub struct VolumeIoStats {
    pub tier: TierId,                         // the Nvme tier on this volume
    pub window_end_ms: u64,
    pub busy: Nanos,                          // time in the window with ≥ 1 read in flight
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub busy_rate: Option<BytesPerSec>,       // bytes_read / busy; None when busy = 0
    pub comparable: bool,                     // request mix matched (dominant size bucket, qd within ±25%)
    pub size_bucket: u8,                      // log2 of the dominant request size
    pub qd_mean: f32,
    pub reads_by_class: [u32; 5],             // Demand, PrefetchNear, Load, PrefetchFar, Background
    pub demand_wait_p99: Option<Nanos>,       // §3.2 bound, measured
    pub temperature_c: Option<f32>,
    pub temp_warning_c: Option<f32>,          // temp1_max or the drive's warning threshold, if exposed
    pub state: ThrottleState,
}
pub enum ThrottleState { Normal, Throttled { since_ms: u64 } }
/// On every transition. RFC-0010 fills ThrottleInfo from the Normal → Throttled event.
pub struct ThrottleEvent {
    pub tier: TierId,
    pub at_ms: u64,
    pub to: ThrottleState,
    pub onset_after: Nanos,                   // continuous comparable load before the drop
    pub ratio: f32,                           // 10 s average / sustained_read at the transition
    pub temperature_c: Option<f32>,
    pub near_thermal_limit: bool,
}
```

On `Throttled`, or while `near_thermal_limit` is set:

1. `Background` traffic pauses (KV spill and session writes wait, unless memory pressure needs them, RFC-0020 §11).
2. RFC-0010's aggregator, which receives these records, updates `sustained_read` with an `Online` value and `ThrottleInfo`, and sends `ReportProfile` if the state outlasts `replan.drift_window` (RFC-0010 §6.3, §7). RFC-0001 §7.2 decides whether to re-plan (ADR-0013 item 7). Throttling is class `Drift`, never a failure.

The engine does not duty-cycle reads to cool the drive: in exact mode a demand read must be served, and it is the solver's job to move bytes off a slow volume.

### 12. Zero-copy paths and the per-OS matrix

The paths of RFC-0001 §9.4, with the engine that serves each:

| OS / hardware | Default engine | Fallback | NVMe → host | Host → VRAM | Copies per byte (NVMe → compute) |
|---|---|---|---|---|---|
| Linux, NVIDIA or AMD discrete | `UringDirect` | `PreadPool` | `O_DIRECT` into pinned memory (`READ_FIXED` for hot slots) | provider copy on the copy stream | 1 DMA to RAM, 1 over PCIe; none for CPU-computed bytes |
| Linux in Docker (default seccomp), Android, ChromeOS | `PreadPool` | buffered pool | `pread` with `O_DIRECT` | as above | as above |
| WSL2 | `UringDirect` on the Linux filesystem | `PreadPool`; buffered on `/mnt/*` | as Linux | `cudaHostRegister`ed memory, copy | as above |
| Windows 11 | `IoRingEngine` | `IocpEngine` | unbuffered read into registered, `cudaHostRegister`ed memory | provider copy | as Linux |
| Windows 10 | `IocpEngine` | — | unbuffered overlapped read | provider copy | as Linux |
| macOS (Apple silicon) | `MacNoCache` or `MacMapped`, by measurement | the other one | read into the `Unified` slot | none | 1 (`MacMapped`: plus one memory copy) |
| Strix Halo, GB10-class | `UringDirect` (Linux) / `IoRingEngine` (Windows) | as their OS | read into the imported `Unified` slot | none | 1 |
| CPU-only nodes | as their OS | as their OS | read into the host arena | — | 1 |
| iOS (Tier 3) | `MacNoCache` | — | read into the arena | — | 1 |

Forbidden on every path (RFC-0001 §9.4): copies from pageable memory to a device, and mmap of a hot tier, except `MacMapped`'s measured, copy-out use.

### 13. Writes: packs, KV spill and sessions

Writes use the same engines and the `Background` or `Load` class (ADR-0013 item 1).

| Writer | File | Discipline |
|---|---|---|
| Pack builder (`Packing`, RFC-0041) | `<pack>.tmp` in the pack directory | preallocate (`fallocate` on Linux, `F_PREALLOCATE` on macOS, `SetFileInformationByHandle(FileAllocationInfo)` on Windows); aligned direct writes of whole units; `DataSync`; write the sidecar; rename both into place atomically; `fsync` the directory |
| KV spill (RFC-0022) | one preallocated spill file per runner per volume, `kv-spill-<instance>-<epoch>.bin` | page runs written in place at page-aligned offsets; no sync on the hot path, because spilled KV is a cache of state the runner still owns and is discarded on restart; deleted at `Stopped` |
| Saved sessions (RFC-0022) | session blobs in the store | write to a temp name, `DataSync`, atomic rename, then `PinSession` is submitted (RFC-0001 §5.10) |

Writes never use the page cache for these files (`O_DIRECT`, unbuffered handles, `F_NOCACHE`), so saving a 1.2 GB session does not evict anything else the OS caches. Write bandwidth is limited to `io.write_max_share` of the volume's sustained rate while reads are active (§3.2).

**Endurance budget (by M1).** Every write Syntropy causes is counted per volume: pack builds, KV spill, session saves and compaction. The I/O thread reads the drive's SMART endurance fields at start and once an hour (NVMe `Percentage Used` and `Data Units Written`; the Windows storage reliability counters; on macOS, IOKit's NVMe SMART data where readable) and publishes `VolumeIoStats::endurance { percent_used, written_by_syntropy, budget_left }`. The budget and the back-off it drives belong to RFC-0022 §8.6: session saves and KV spill back off first; pack builds are never throttled by it, because a plan cannot run without its pack. Where SMART is unreadable, no budget is computed; `doctor` shows the bytes Syntropy has written and says why.

### 14. Integrity

Decision (ADR-0204): **reads on the token path are not hashed.** Hashing a 26.5 MB expert with BLAKE3 costs milliseconds of a core per miss on the same CPU that computes misses in place, which is the bottleneck in the cost model's scenario b2. Instead:

1. **At pack commit** (RFC-0041), every extent's BLAKE3 hash is computed from the bytes written and checked against the source tensor hash before the sidecar is renamed into place.
2. **After an unclean shutdown**, the store marks packs that were open for writing as unverified; the next `Loading` verifies their extents before use.
3. **At load**, a sample is verified in full on the compute pool's cores before `Warming`: every extent of the dense hot set, which every token reads, plus a random 1% of the remaining extents. [Target] the check costs ≤ 2% of load time; it is skipped for packs verified at commit in this boot.
4. **An idle-time scrubber** verifies every pack extent at `io.scrub_rate` (64 MB/s), only on mains power, only while the volume's read traffic is below 10% of its sustained rate, and pausing at once when demand traffic appears.
5. **On mismatch**: `Integrity` (RFC-0001 §10.1). The extent is marked bad in the sidecar; the runner reports it; the store rebuilds the extent from verified source chunks (refetching them if needed); the group is not served from the bad extent again.

Short reads, `EIO` and `ENODATA` are retried once at the same offset. A second failure is `Integrity` for that extent, handled as in item 5.

### 15. Configuration

`node` scope unless marked; schema in RFC-0054.

| Key | Default | Reload | Meaning |
|---|---|---|---|
| `io.engine` | `auto` | restart | `auto` / `uring` / `pool` / `ioring` / `iocp` / `nocache` / `mapped`: override the measured choice |
| `io.pool_threads` | 16 (Linux, Windows pool), 8 (`io.pool_threads_macos`) | restart | `PreadPool` and `MacNoCache` depth |
| `io.iocp_threads` | 4 | restart | IOCP submitting threads, upper bound |
| `io.chunk` | probe; 4 MiB before probing | restart | split size for large reads |
| `io.qd_max` | 64 | hot | depth ceiling per volume |
| `io.prefetch_inflight_max` | 32 MiB | hot | prefetch bytes in flight while a demand read is pending |
| `io.write_max_share` | 0.25 | hot | write share of sustained bandwidth while reads are active |
| `io.sqpoll` | `auto` (off unless the probe shows ≥ 15%) | restart | `SQPOLL` policy (ADR-0203) |
| `io.iopoll` | `off` | restart | `IOPOLL` policy |
| `io.stage_slot`, `io.stage_slots` | 64 MiB, 4 per GPU | restart | load staging ring |
| `io.admit_pcie_share` | 0.25 | hot | PCIe share for cache admission copies per step (§8) |
| `io.spin` | 50 µs, 10 µs on battery | hot | I/O thread spin before parking |
| `io.thermal_margin` | 3 °C | hot | §11 |
| `io.scrub_rate` | 64 MB/s | hot | §14; 0 disables |

### 16. Failure handling and error codes

Domain `Io`.

| Code | Class | Raised when | Action |
|---|---|---|---|
| `IO_ENGINE_UNAVAILABLE` (1) | `Unsupported` | ring setup fails with `ENOSYS`/`EPERM`, IoRing missing | next engine in the fallback chain; recorded in the profile |
| `IO_DIRECT_REFUSED` (2) | `Unsupported` | `O_DIRECT` or unbuffered open or read refused | buffered mode, volume flagged (§4.2) |
| `IO_REGISTER_FAILED` (3) | `Capacity` | fixed-buffer or file registration fails | unregistered operations; caps updated |
| `IO_SHORT_READ` (4) | `Transient` | short read, `EIO`, `ENODATA` | one retry; then `IO_EXTENT_BAD` |
| `IO_EXTENT_BAD` (5) | `Integrity` | repeated read failure or hash mismatch (§14) | extent marked bad; rebuild; never served |
| `IO_NO_SPACE` (6) | `Capacity` | `ENOSPC` on pack, spill or session writes | pack: fall back to read-around from source (ADR-0005 item 4); spill: KV spill disabled for the volume and RFC-0022 notified; session: save fails visibly |
| `IO_MISALIGNED` (7) | `Internal` | a request violates §6 | rejected before submission; counted; a bug |
| `IO_THROTTLED` (8) | `Drift` | §11 | report; re-plan rules |
| `IO_GDS_FALLBACK` (9) | — | retired 2026-10-20 with the GDS path (§9); the code stays reserved | — |
| `IO_CANCELLED` (10) | `Cancelled` | a prefetch cancelled or dropped as no longer useful | completion discarded |

A demand read that fails after retries fails the step's sequence set with `Integrity` only if the group has no other readable copy; if a RAM or VRAM copy exists (a cached copy of an `Nvme`-homed expert), the step uses it.

## Performance model and budgets

All figures are **[estimate]** with derivation or **[target]**; none is a Syntropy measurement.

**The NVMe term.** From cost model §3.3: `t_nvme = (1 − h_ram) · B_routed / W_nvme(sustained)` per token for MoE; `B_overflow / W_nvme` for dense overflow, ≈170 ms per GB at 6 GB/s. This RFC's job is to make `W_nvme` in that formula the probe's sustained rate rather than the 1.5–2.6 GB/s the surveyed page-cache paths delivered on Gen4. [Target] SC-7: expert reads at ≥ 0.85 × the probe's sustained, throttled rate at the same size and depth: a 6 GB/s probe means ≥ 5.1 GB/s of expert bytes.

**Request rates are low; depth matters, not submission cost.** Scenario b2 at h = 0.9: 1.2 GB of NVMe reads per token (cost model §4 b2) at ≈4.6 tok/s is ≈5.5 GB/s, or ≈210 expert reads per second at 26.5 MB, ≈1.5K chunks per second at 4 MiB. The surveyed 125B model's SSD mode read ≈0.33 GB per token at ≈8 tok/s with 1.35–2.6 MiB experts: under 2K reads per second. Row lookups: 16 rows per token at 100 tok/s is 1.6K reads per second. One thread's blocking ceiling is ≈80K reads/s (`tiered-memory-inference.md` §10.3). So submission cost is not the limit at these rates; reaching depth 4–16 on large reads is, and io_uring does that from one thread while the pool needs one thread per in-flight read (ADR-0203).

**Demand latency bound.** A demand read of size S waits behind at most `io.prefetch_inflight_max` of prefetch: `t_demand ≤ (S + 32 MiB) / W_nvme`, ≈10 ms for a 26.5 MB expert at 6 GB/s, against ≈4.4 ms alone, in the worst case where the drive serves commands strictly in order. Drives serve queued commands concurrently, so the expected figure is lower; the M1 benchmark measures it.

**Rows.** A 1.4 KB row read as one or two aligned 4 KiB blocks reads ≤ 8 KiB: 16 rows per token is ≤ 128 KiB per token, issued one step ahead (`tiered-memory-inference.md` §9). At QD1 latency of ≈45 µs per 4 KiB read on a 990 PRO (cost model §2.4), 16 parallel reads complete in well under a 10 ms decode step.

**Staging.** H2D from pinned memory at ≈24 GB/s on PCIe 4.0 x16 (cost model §2.3); a 26.5 MB admission copy takes ≈1.1 ms. With `io.admit_pcie_share` = 0.25 and a 200 ms step (b2), admissions may use ≈1.2 GB of link per step [estimate]; at a 30 ms step on an all-VRAM-hot model, ≈180 MB.

**Load.** [Target] SC-23: `t_pack + t_load ≤ 1.25 · (2·S / W_read + S / W_write)` for a shard of S bytes: 42.5 GB on a Gen4 drive (7.45 GB/s read, 6 GB/s write assumed) ≈ 18.5 s → ≤ 23 s. The load path reads each pack extent once with direct I/O at `qd_target`, so it is bounded by `S / W_read` plus the integrity sample (≤ 2%, §14).

**CPU cost.** [Target] the I/O thread uses < 25% of its core at the b2 read rate, and the I/O path adds zero memory copies on the default paths (§12). [Target] ≤ 1 I/O core per NVMe volume holding a tier (ADR-0012).

## Alternatives considered

| Alternative | Why not |
|---|---|
| mmap and the page cache everywhere | Residency follows kernel LRU (ADR-0007); measured 2.2× slower for a hot pool on Windows and 7× slower for rows on SATA (`tiered-memory-inference.md` §10.4). Kept as `MacMapped`, copy-out only, where the probe chooses it. |
| `PreadPool` everywhere | Works in every sandbox, but depth costs one blocked thread per in-flight read and ≈12 µs of kernel time per read. Kept as the fallback. |
| Linux AIO (libaio) | `O_DIRECT` only, superseded by io_uring, no registered buffers (ADR-0013 option 3). |
| Fixed buffers for every destination, including `Weights` | Fixed buffers are charged to `RLIMIT_MEMLOCK` and capped at 1 GiB each; registering tens of GB fails under default limits and gains little for multi-MiB reads (ADR-0203). |
| `SQPOLL` on by default | Costs a core that the CPU compute pool needs for in-place experts (ADR-0012); request rates here are low (§Performance). |
| Strict FIFO with no byte cap on prefetch | A demand read could wait behind tens of MB of speculative reads. |
| Pre-empting prefetch by cancelling in-flight reads on every demand | Cancels waste the bandwidth already spent and race with completions; the byte cap bounds the wait more cheaply. |
| Hashing every read | Milliseconds of CPU per expert miss on the bottleneck core (ADR-0204). |
| GDS or DirectStorage as primary paths | Linux datacenter-only and D3D12-centred respectively (ADR-0013 options 4 and 5). |
| GDS and DirectStorage as opportunistic, gated paths (the plan until 2026-10-20) | GDS direct mode is documented for Tesla and Quadro only, so on GeForce the cuFile path would be code with no consumer benefit; DirectStorage adds nothing for uncompressed quantised weights. Dropped from v1, with a read-only capability report kept |
| Duty-cycling reads when the drive is hot | Exact mode must serve demand reads; slowing them silently hides a placement problem the solver should fix. |

## Testing and validation

| Claim | Test | Level |
|---|---|---|
| Every engine returns the right bytes | a conformance suite run against every engine available on the CI host: random aligned extents of a fixture file read into fixed and raw destinations and compared with `pread` through the page cache; writes read back | unit, per engine |
| Priority and depth rules (§3.2) | a fake engine with scripted latencies: demand reads are always submitted first; prefetch in flight never exceeds the byte cap while demand is pending; `PrefetchFar` drops first; `useful_until` drops stale prefetches | unit, property |
| No allocation, no lock in the loop | the `testkit` counting allocator and lock instrumentation around a steady-state read workload (ADR-0012 item 6) | unit |
| Fallback chain | the Linux suite runs twice in CI, once with io_uring denied by a seccomp profile (ADR-0013 work item for RFC-0062); Windows runs IoRing and IOCP; a tmpfs-backed run exercises buffered mode | integration |
| Alignment (§6) | property test: any extent produced by the pack builder with any `DioAlign` the OS reports passes the request check; misaligned requests are rejected before submission | property |
| SC-7 | expert-sized reads at `qd_target` on every Tier 1 rig's volume, ≥ 0.85 × the probe's sustained rate | benchmark (RFC-0061) |
| `UringDirect` against `PreadPool` | throughput and CPU time per GB at expert, row and KV-page sizes on Tier 1 Linux hardware; feeds ADR-0013's revisit condition | benchmark |
| `MacNoCache` against `MacMapped` | the same A/B on every Apple rig; the result per machine is recorded | benchmark |
| Demand latency bound | demand reads injected under saturating prefetch; p99 wait ≤ the §Performance bound | benchmark |
| Thermal detection | a Gen5 drive without a heatsink in the lab, sustained reads for 30 minutes; `Throttled` within 20 s of the rate drop; `ReportProfile` follows | lab benchmark; fitting the drive is a **Maintainer action** (ADR-0021) |
| Integrity | flip bytes in a pack extent on disk; the scrubber or the load sample finds it; the extent is rebuilt and never served | integration, chaos |
| Crash safety of writes | kill the runner mid-pack-build and mid-session-save; no torn pack or blob is ever accepted (atomic rename) | chaos (RFC-0060) |
| Direct-path capability report (§9) | on a CUDA Linux CI host without `libcufile`, a fixture with a fake `libcufile` reporting compatibility mode, and a Windows host with and without the DirectStorage runtime: `direct_paths` matches each case, no read goes through either API, and `doctor` prints the result | integration |

*Note 2026-10-23 (TIER-E5-T33, as built):* the first row's suite is in two parts. `syntropy_io::conformance` is the harness: `check` takes a constructor for any `IoEngine` and a file, builds and drives the engine on one thread of its own (a ring set up with `IORING_SETUP_SINGLE_ISSUER` takes submissions only from its creator, §4.1), reads random aligned extents into fixed and raw destinations, and compares each with a buffered `std::fs` read of the same extent. That read, a seek and a read, goes through the page cache as `pread` does, and works on every OS, Windows included, which has no `pread`. When the engine writes, `check` also writes random aligned extents to a scratch file and reads them back, through `std::fs` and through the engine. It also checks that each operation completes once, with its own ticket and its whole length (one round gives every operation a different length, so swapped tickets show); that it touches no memory outside its destination; that no write changes the file past its extent; and that no engine call outlasts the run's timeout. A run is fixed by a seed, which it prints, and `SYNTROPY_IO_CONFORMANCE_SEED` repeats it. Each engine task runs the harness over a file that `write_test_file` generates (TIER-E5-T33), and TIER-E5-T04 runs it over the nano fixture files against every engine available on the CI host.

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | `IoEngine` interface; `PreadPool` with `O_DIRECT` on Linux and Windows, `MacNoCache` on macOS; the conformance suite; reads for `Loading` only |
| M1 | `UringDirect` with registered files and fixed buffers for hot slots; `IoRingEngine` and `IocpEngine`; `MacMapped` and the per-volume A/B; priority classes, byte-capped prefetch, `useful_until`; staging pipelines into VRAM; thermal detection with `VolumeIoStats` and `ThrottleEvent`; pack, spill and session-blob writes (with RFC-0022, whose saved sessions are M1); integrity checks and the scrubber; SC-7 on the single-device rigs |
| M2 | engine selection results shared in the resource graph for remote-node planning; no new engines |
| M3 | no GDS or DirectStorage path (dropped from v1, §9); the read-only capability report lands with the M1 probe |
| M4 | Tier 2/3 platforms: Android pool, iOS `MacNoCache`, ChromeOS pool; I/O regression gates in RFC-0061 |

## Open questions

1. **io_uring against the pool at expert sizes.** No published numbers exist for this workload (ADR-0013). The M1 benchmark decides whether ADR-0013's revisit condition (within 5% throughput and 20% CPU) holds, in which case io_uring is kept only for rows and KV.
2. **Demand wait under saturating prefetch.** The 32 MiB byte cap is a chosen default. The M1 benchmark measures p99 demand wait at 8, 16, 32 and 64 MiB on Gen4 and Gen5 drives and sets the default.
3. **macOS sustained reads and FileVault overhead.** No public data on Apple SSD throttling or APFS cost for large sequential reads (`hardware-and-performance-model.md` §6 item 5). Measured in M1 on the Apple rigs. **FileVault part answered 2026-10-20 (OQ-IO-01):** FileVault cannot change `W_nvme`, because the internal volume is always encrypted and the AES engine sits in the DMA path; the FileVault A/B is dropped and the throttle soak stays.
4. **Provider registration and io_uring fixed buffers on the same pages.** Registering the same pinned range with both CUDA and io_uring is expected to work, since both pin the pages; it is checked on every CI driver in M1.
5. **NVMe temperature without privileges on Windows.** `StorageDeviceTemperatureProperty` on a zero-access handle is expected to work for a standard user; to be confirmed on the Windows rigs in M1. Throughput detection works regardless.
6. **GDS on GeForce. Answered 2026-10-20 (OQ-IO-02):** no. NVIDIA documents GDS direct mode for Tesla and Quadro GPUs only, so GeForce gets compatibility mode; the cuFile path is dropped from v1 (§9), and the capability report watches for a change.

## Proposed amendments to RFC-0001

1. **§5.2 `StorageCaps`.** Add `engine`, `chunk`, `dio`, `buffered`, `sqpoll`, `read_around_efficiency` and `shared_lanes` with serde defaults (§10). These are additions the owner of `CapabilityProfile` (RFC-0010) may make; listed so RFC-0010 adopts them.
2. **§8.2 I/O engine row.** "Thread per I/O core, one submission ring each" should say "one I/O thread per NVMe volume that holds a tier, at most two per NUMA domain", matching ADR-0012.
3. **§9.4 forbidden paths.** Add the `MacMapped` exception explicitly: a measured mmap read mechanism that copies into owned slots and never counts the page cache as residency.
