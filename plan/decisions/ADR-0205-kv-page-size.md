# ADR-0205: KV pages of 64 tokens, layer-first on the GPU and page-first on host and NVMe

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the tiers area (RFC-0022))
- Related: ADR-0013, ADR-0017, ADR-0019; RFC-0001 §5.9; RFC-0021 §6; RFC-0022 §3; RFC-0040

## Context

The KV store pages every sequence's cache so that pages can be shared (prefixes, forks), spilled across tiers, saved and migrated. The page size trades:

- **Fragmentation.** A sequence wastes up to P − 1 positions per page class. At 320 KiB per token (Llama-3.3-70B, `hardware-and-performance-model.md` §3.6), P = 64 wastes at most ≈19.7 MiB per sequence; P = 256 wastes ≈80 MiB.
- **Transfer size.** Spill, fetch, save and migration move pages. Bigger pages mean fewer, larger transfers. Direct I/O wants 4 KiB-multiple, contiguous runs (RFC-0021 §6).
- **Prefix granularity.** Prefix keys are per page, so larger pages share less. RFC-0022 recovers token granularity with partial tails, so page size no longer limits the SC-20 contract.
- **Kernel addressing.** GPU paged-attention kernels typically use block sizes of 16–64 tokens; each block-table entry is an indirection per block per step.
- **Layout.** A GPU attention kernel for layer l wants layer l's pages contiguous (layer-first). A tier transfer wants all layers of a token range in one record (page-first). The surveyed datacenter stacks used layer-first on the GPU and page-first on the host and got up to 2× from larger transfers (`tiered-memory-inference.md` §12.2).

Worked sizes for one page at P = 64 [estimate]: GQA with 8 KV heads × 128 × BF16 is 4 KiB per token per layer, so 256 KiB per layer; DeepSeek-V3 MLA is 1,152 B per token per layer, so 72 KiB per layer and ≈4.5 MB across 61 layers; gpt-oss-120b's 8 × 64 GQA is 128 KiB per layer. All are 4 KiB multiples.

## Options considered

| Option | For | Against |
|---|---|---|
| A. P = 16, one layout everywhere | Fine sharing; common kernel block size | 4× the block-table entries of 64; small tier transfers |
| B. P = 256 | Few, large transfers | Up to 255 tokens wasted per class per sequence; pages seal late, delaying spill and prefix insertion |
| **C. P = 64; layer-first pools on the GPU with one page index shared by all layers of a class; page-first records on host and NVMe, each layer slice padded to 4 KiB; gather and scatter on the copy stream** | Small fragmentation; MB-scale tier transfers; one block table per class; direct-I/O-aligned | A gather/scatter op per tier move; page size is a setting that must be part of every KV identity |

## Decision

**Option C.**

1. `P = 64` positions per page (entries for compressed layers). `kv.page_tokens` may be 16–256, a power of two, per instance, and is part of the KV-settings hash, so prefixes and sessions never mix sizes.
2. Layers that advance in lockstep form a page class with one block table: `Full`, `Windowed(w)`, `Compressed(r)`, `State`, `Aux`.
3. GPU pools are layer-first; host and NVMe pages are page-first, one record per page index holding every layer of the class, each layer's slice 4 KiB-padded.
4. Tier moves gather and scatter with the provider's batched copies or `kv_pack`/`kv_unpack` (RFC-0040).

## Consequences

Positive:
- Spill and save of DeepSeek-V3 KV move ≈4.5 MB records, not 61 separate 72 KiB copies.
- Block tables stay small (16 KiB for a 128K-token sequence).
- Every page record is directly readable with direct I/O.

Negative:
- Kernels must accept 64-token blocks; a provider whose best attention kernel wants another block size pays an indirection or a copy.
- A gather/scatter kernel per backend.
- Changing P invalidates all prefixes and sessions of that instance.

## What would make us revisit

- The M1 benchmark of decode attention throughput at P = 16, 32, 64 and 128 on the CUDA and Metal providers shows another size more than 5% faster with no loss in spill bandwidth.
- Workloads where many concurrent sequences hold short contexts make fragmentation significant: drop to 32 for throughput mode.
- A future attention kernel family needs a per-layer block size that differs within a class.
