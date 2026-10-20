# ADR-0410: Packs are sidecars over a shared, content-addressed extent store, holding only what needs a pack

- Status: Accepted (2026-10-20); revised 2026-10-20 (decision 3)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0005, ADR-0007, ADR-0013, ADR-0015, ADR-0016; RFC-0001 §5.6; RFC-0041 §5; RFC-0021, RFC-0023, RFC-0040 §5.4

## Context

- ADR-0005: the pack is the per-device execution layout, in a local cache; every expert and streamable unit starts on a 4 KiB (or the direct-I/O) boundary; optional backend layouts; a sidecar with offsets, sizes, hashes and the expert-frequency profile; units are content-addressed so a new epoch reuses unchanged units; rebuildable and safe to delete. Laying the file out in the unit that is read gave +17 % decode on one measured machine (`tiered-memory-inference.md` §8).
- RFC-0001 §5.6 keys a pack by (`ModelId`, `DeviceClass`, shard digest, `LayoutVersion`), so a changed shard gives a new key even when most groups are unchanged.
- A surveyed engine's repack needed about 40 GB of extra disk, and a multi-GB repack at setup is a listed anti-pattern (`tiered-memory-inference.md` §19, §20 item 13).
- SC-23 budgets pack plus load at `1.25 · (2·S/W_read + S/W_write)`.
- Groups homed in memory are read once per load, sequentially; groups homed on NVMe are read at runtime, one request per group.

## Options considered

| Option | For | Against |
|---|---|---|
| A. One monolithic file per `PackKey` holding the whole shard | Simple to delete and copy | A re-plan rewrites everything; packs of different shards cannot share bytes; doubles disk for memory-homed groups |
| B. A file per extent | Trivial dedupe | Hundreds of thousands of files for a large MoE shard; open and metadata costs on the runtime read path |
| **C. Append-only segment files per device class holding aligned, content-addressed extents (keyed by source tensor hashes plus layout); one sidecar per `PackKey` referencing extents; extents only for NVMe-homed groups and non-source layouts** | Incremental re-plans; sharing across shards; few files; disk use limited to what the runtime needs | An extent index and garbage collection to maintain; compaction of dead segment space |

## Decision

**Option C.**

1. Extents live in `packs/<device-class>/seg/<n>.seg`, preallocated in 4 GiB steps, written only at the tail with direct I/O, never rewritten in place; compaction moves live extents out of unleased segments that are more than half dead, updating only the index. One pack container version covers segment headers, index records and sidecars.
2. Extent alignment `A = max(4096, the volume's direct-I/O offset alignment, the Windows sector size, the macOS page size)`; components inside an extent start on 64-byte boundaries; stored length is padded to `A`.
3. `ExtentKey = BLAKE3("syntropy/extent/v1" ‖ tensor hashes ‖ layout ID ‖ layout version)`; an append-only index maps it to (segment, offset, length, BLAKE3 of stored bytes). There are no persisted reference counts: extents are collected by mark and sweep from the committed sidecars plus in-progress build-intent files. *Revised 2026-10-20:* the mark set also includes every extent that the store's range map names for a source chunk still referenced by a shard, because under disk pressure a `SOURCE`-layout extent may be the node's only copy of those source bytes (RFC-0041 §4.5). The range map keys on `ExtentKey`, so compaction never rewrites it.
4. A pack is a postcard + zstd sidecar per `PackKey`, listing its extents by `ExtentKey` (with a location hint used only to rebuild a lost index) and the groups it loads from source blobs. It holds only `manifest` and `core` types. It is committed by write–fsync–rename after its extents and their index records are durable, and never changes afterwards. A group may have one extent per stored layout when the plan keeps per-site copies (RFC-0040 §5.4); `PackKey::layout` is the release's layout-table version. Selection entries and the expert profile are separate `SideRecord` files beside the sidecar, encoded by their owners with their own schema versions; the canary digest is a side record per (model, quantization) beside the manifest.
4a. A runner holds a lease on its pack (an advisory lock released by the OS when the process exits) from `Loading` until it exits. Leases and compactions are ordered by an index generation that each compaction raises on commit. A lease pins the generation current when it was granted, and a compaction deletes its old segment only once no lease holding an older generation names an extent in it. Leased sidecars are not evicted and leased segments are not compacted.
5. A group gets an extent only if its home tier is `Nvme` or its selected layout is not `SOURCE` (or `pack.copy_source_layout` is set). Memory-homed `SOURCE` groups load straight from the source blob with aligned read-around.
6. The expert profile side record is rewritten periodically and at drain; the sidecar and extents never change after they are written.

## Consequences

- Positive: a re-plan writes only new extents; a pack can be deleted at any time (its extents become garbage); SC-23 holds by construction.
- Positive: on plans that fit in memory with source layouts, the pack is just a sidecar, so no disk is spent duplicating weights.
- Negative: memory-homed groups are re-read from source blobs at every load. They stay in their source layout, so no repack cost is paid again, but a slow source (NAS) needs `pack.copy_source_layout`.
- Negative: the extent index and compaction are new code with crash-consistency requirements (RFC-0041 testing).

## What would make us revisit

- ADR-0005's own trigger: packed reads beat aligned read-around from the source by less than 5 % decode on Tier 1 machines. Then packs hold only repacked layouts.
- Dead-segment compaction costs more than 5 % of a drive's write endurance budget per month on a typical re-planning workload.
- Pack overhead exceeds 30 % of shard size on typical Tier 1 devices (ADR-0005's trigger).

## Log

- Revised 2026-10-19: decisions 1, 3, 4 and 6 revised and 4a added, to allow several layouts per group, to make GC, compaction and eviction crash-safe against running instances, to keep sidecar types inside `manifest`'s layer, and to give the pack one container version with a separately versioned expert profile.
- Revised 2026-10-19: decision 4a orders leases against an in-progress compaction by index generation.
- Revised 2026-10-20: decision 3's mark set gains extents named by the range map. When free space would fall below `store.min_free_bytes`, the store punches source chunks that a committed `SOURCE`-layout extent holds unchanged, after syncing a map entry from the chunk to the extent, so an NVMe-tiered MoE plan no longer keeps its experts twice on a full disk.
