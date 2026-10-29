# RFC-0041: Model formats, packs and weight distribution

- Status: Draft
- Area: compute and models
- Depends on: RFC-0001 (§3.3, §5.3, §5.6, §7.1, §9), RFC-0021, RFC-0023, RFC-0030, RFC-0031, RFC-0034, RFC-0040, RFC-0042, RFC-0043, RFC-0052, RFC-0054; ADR-0005, ADR-0007, ADR-0009, ADR-0013, ADR-0015, ADR-0016, ADR-0019; ADR-0410 to ADR-0413
- Date: 2026-10-19
- Revised: 2026-10-20: source ranges punched under disk pressure when a pack extent holds them (§4.5); licence acceptance on receiving nodes (§7.5); no project-published packs or draft heads, and fixtures under the fixture store's licence rule (Non-goals, Testing); same-class pack transfer off by default (§8.5); header parsing in a child process (§2); open questions 2, 3, 4 and 6 answered; 2026-10-22: the text form of the `<device-class>` directory name (§4.1; ADR-0405)

## Summary

This RFC turns a user's model reference ("this Hub repo at this revision", "this GGUF on disk") into the bytes a runner executes. It defines:

- the identities: `FileDigest` (SHA-256 of a source file, equal to the Hub's object ID), `ChunkId` (BLAKE3 of a chunk), `ModelId`, `ShardSpec::digest`, `PackKey`, and how each is computed without reading weight bytes the node does not need;
- readers for GGUF (files never modified) and safetensors (BF16, FP16, FP8, MXFP4, NVFP4 and compressed-tensors), both producing one `ModelManifest`;
- the census builder that classifies every tensor and slices stacked expert tensors into per-expert records;
- the on-disk model store: sparse source blobs keyed by `FileDigest`, chunk tables, manifests, and a per-device-class pack store;
- the pack format: content-addressed, aligned extents in append-only segment files, plus a sidecar per `PackKey`, rebuilt incrementally when a plan changes;
- Hub integration through HTTP range reads (and the Hub's chunked storage where offered), and the peer-to-peer chunk protocol that fetches each origin byte once per cluster.

Key decisions: `ModelId` commits to source content through file digests, not per-tensor hashes, so it is known from headers alone (ADR-0411). Chunk hashes are learned from the first fetch and committed through the coordinator, which makes cluster nodes verifiable sources (ADR-0412). Packs store extents only for groups that need them (NVMe-homed groups and repacked layouts) in a shared extent store, so a re-plan writes only new extents (ADR-0410). Safetensors quantization schemes map onto native `QuantScheme`s with no conversion (ADR-0413).

## Motivation and goals / non-goals

ADR-0005 fixes the formats (GGUF unchanged, safetensors natively, a per-device pack, no new interchange format). ADR-0016 fixes distribution (shard-only range fetch, cluster first, BLAKE3 chunks, resume). Neither says how identities are computed before any weight bytes move, how a pack is laid out byte by byte, or how a chunk's hash becomes known when no node ever holds a whole file. Those gaps decide whether "fetch only your shard" and "verify every byte" can both hold. A 2026 frontier model is 0.6–1.5 TB (`backends-and-portability.md` §0 item 6), so any step that reads or copies the whole model on one node is a defect.

### Goals

1. Census, `ModelId` and placement from headers only: no weight bytes before a plan exists (RFC-0001 §7.1 step 2).
2. Every byte a runner executes is verified against a hash the cluster agreed on, or against the origin's own content hash.
3. Each origin byte crosses the WAN at most once per cluster when the origin publishes chunk metadata, and at most twice without it, for the second witness's read (§6.3, RFC-0034 §10.1).
4. One aligned read per expert or streamable unit (ADR-0005); pack build and load within SC-23 (`t_pack + t_load ≤ 1.25 · (2·S/W_read + S/W_write)`).
5. A re-plan writes only extents that did not exist before; a pack can be deleted at any time and rebuilt.
6. Users' files stay byte-identical; local models are indexed in place.

### Non-goals

- A Syntropy model format for exchange (ADR-0005). Packs are local and never sent between nodes, except as a same-class shortcut that stays off by default (§8.5; ADR-0016 decision 7).
- Artefacts derived from third-party models published by the project. The project publishes no packs and no trained draft heads; draft heads are dropped from v1, and the loader for draft heads a publisher ships is kept. A model's licence carries duties to anything derived from it, so the project publishes none. Fixtures, recorded headers and golden data derived from a model live in the fixture store, publicly readable only when the model is permissively licensed or the fixture is project-synthetic (RFC-0060 §12.3).
- Requantization during import. Optional requantization tools are RFC-0042 §9 and produce a new model identity.
- The byte layout of bulk-plane messages and the iroh blob plane itself: RFC-0031 owns them. This RFC defines the messages' meaning, sources, verification and resume.
- Tokenizer and chat-template semantics (RFC-0043 §8); this RFC only extracts their data.

## Design

### 1. Identities

| Identity | Definition | Known when | Used for |
|---|---|---|---|
| `FileDigest` | SHA-256 of a whole source file | Hub: from the revision's file metadata (the object ID of LFS- or Xet-stored files), before any download. Local: computed once by the store, cached by (device, inode, size, mtime) | anchors `ModelId`; whole-file check whenever a node holds a complete file (ADR-0016 decision 5) |
| `ChunkId` | BLAKE3 of one chunk's bytes | after the first fetch of the chunk (§6.3), or at registration for local files | content addressing, P2P verification, dedupe |
| `ModelId` (RFC-0001 §5.3) | defined below | after headers are read | instances, plans, packs, sessions, prefix caches |
| `TensorRecord::hash` | `Option<Blake3>`: BLAKE3 over the tensor's ordered `ChunkId`s (a Merkle root, not a hash of raw bytes). It is unset (`None`) until every chunk covering the tensor is committed (§6.3) | once the tensor's chunks are committed | pack derivation keys, integrity checks |
| `ShardSpec::digest` | as RFC-0001 §5.6: BLAKE3 over `model` and the sorted group list | at plan commit | `PackKey` |
| `PackKey` | as RFC-0001 §5.6: (`ModelId`, `DeviceClass`, shard digest, `LayoutVersion`). The `layout` field is the release's **layout-table version**, one number bumped whenever any provider's repack output changes; each extent carries its own layout ID and version (§5.2), so one pack may hold several layouts | at plan commit | sidecar lookup |
| `ExtentKey` | BLAKE3 over (`"syntropy/extent/v1"`, the group's tensor hashes, the layout ID and its version) | once the group's chunks are known | extent dedupe across packs (§5.2) |
| `TokenizerId` | BLAKE3 over the canonical tokenizer data and chat template (RFC-0043 §8) | after headers | API-side caches; not part of `ModelId` |

**`ModelId`:**

```text
ModelId = BLAKE3(
    "syntropy/model-id/v1"
  ‖ family tag                          -- RFC-0043 §5.1's frozen tag table: a tag is never renamed or reused
  ‖ for each source file the census reads, sorted by (role, FileDigest):
        role ‖ FileDigest ‖ length
)
role ∈ { weights (GGUF or safetensors shard), weight index JSON, config.json, generation_config.json,
         quantization config file }
```

The file digests commit to every byte the census reads, including every tensor byte and every hyperparameter, so any change to the model's files changes `ModelId`. Nothing Syntropy derives (canonical `HyperParams`, `LayerSpec`, `QuantScheme` encodings) is hashed, so a release that adds a field to those types or changes how it normalises them leaves the ID of the same files unchanged. File names are not hashed either, so renaming a local file keeps its ID. The one release-to-release change that does alter an ID is a change of which family accepts a checkpoint, because the family tag is hashed; that is a deliberate semantic change and is listed in the release notes as such (ADR-0622). Golden IDs for the recorded fixtures are checked in, so any other change fails CI. Tokenizer files stay out of the ID (`TokenizerId`). It can be computed from headers and file metadata alone, which is what RFC-0001 §7.1 needs: `RegisterModel` is sent before any weight is fetched. The cost is that the same tensor bytes in a GGUF and in a safetensors file get different `ModelId`s. Byte-level dedupe across containers still happens at the chunk level, because chunk boundaries are tensor-relative (§6.1) and chunks are content-addressed. ADR-0411 records the trade-off; the RFC-0001 sentence it changes is in the amendments.

**Revision note, 2026-10-28 (COMP-E10-T06): the bytes the v1 formula hashes.** The formula above names its parts; `syntropy_manifest::identity` writes them as follows, and the golden IDs fix them. The domain string is its 20 ASCII bytes with no terminator. The family tag is a u64 little-endian byte length, then its UTF-8 bytes. Each file is one 41-byte record: a role code, the 32-byte SHA-256 and a u64 little-endian length. The role codes are 1 weights, 2 the weight index JSON, 3 `config.json`, 4 `generation_config.json` and 5 the quantization config file (reserved; no reader names that file yet). Files are sorted by (role code, `FileDigest`), so the order the census finds them in does not matter. A file in another role, such as a tokenizer file, is left out. An ID needs a non-empty tag and at least one weights file. The length prefix and the fixed record size mean no two inputs write the same bytes. Any change to this encoding is a new formula version, not an edit of v1.

**Revision note, 2026-10-28 (COMP-E10-T11): the bytes of the other identities.** `syntropy_manifest::identity` writes them as follows, every integer little-endian. Golden values, computed by an implementation separate from the crate's, fix them. Any change is a new version with a new tag. Two choices below are for the maintainers to confirm, as RFC-0001 §18.50 records: the tree's shape, and the reading of `ExtentKey`'s version as the layout-table version.
- **`TensorRecord::hash`** is a binary Merkle tree over the tensor's `ChunkId`s in file order, with tag `T = "syntropy/tensor-hash/v1"`. A leaf is `BLAKE3(T ‖ 0x00 ‖ id)` and a node is `BLAKE3(T ‖ 0x01 ‖ left ‖ right)`. The root of one ID is its leaf. The root of n > 1 IDs is the node over the roots of the first k and the remaining n − k IDs, where k is the largest power of two below n, the shape of RFC 6962 §2.1. An odd count is never padded, so no last element is duplicated, and the 0x00 and 0x01 prefixes keep a leaf from passing as a node. A zero-length tensor covers no chunk; its root is `BLAKE3(T ‖ 0x02)`.
  - A tensor's IDs are those of the chunks that start inside its source extent, then inside each auxiliary extent, in that order. Every such start is a chunk boundary (§6.1), so these chunks cover the extents exactly. `ChunkTable::tensor_hash` collects them. It gives `None` while any of them is uncommitted (§6.3), and refuses an extent that starts at no chunk boundary or ends past the file.
  - `TensorRecord` comes with the census builder (COMP-E10-T07), and its `hash` holds this value.
- **`ShardSpec::digest`** is `BLAKE3("syntropy/shard/v1" ‖ ModelId ‖ u64 count ‖ each TensorGroupId as u32, ascending, each once)`. The node, the chunk list and the plan epoch are not inputs, so an unchanged group set keeps the digest and the `PackKey` across epochs (RFC-0001 §5.6).
- **`ExtentKey`** is `BLAKE3("syntropy/extent/v1" ‖ u64 count ‖ each TensorRecord::hash, in the extent's component order ‖ u16 layout ID ‖ u16 layout version)`.
  - The layout ID is the extent's own layout (`SidecarExtent::layout`, RFC-0040 §5.4). The version is the layout-table version that `PackKey::layout` carries, the one layout version the plan defines. A per-layout version would let a provider update rebuild fewer extents; it needs a new key version.
  - The tensor hashes keep their order, because the same tensors laid out in another order are other bytes.
- **`TokenizerId`** keeps the encoding COMP-E17-T01 built (RFC-0043 §8.1). COMP-E10-T11 adds a golden value for it.

### 2. Source readers

Both readers are safe Rust under `#![forbid(unsafe_code)]`, run on untrusted bytes (ADR-0001 decision 4), are fuzzed (RFC-0060), and read through a `RangeSource` trait so the same code parses a local file, a sparse blob or a Hub URL. From M1, the headers of a newly added model are parsed in a short-lived child process with resource limits and no network, which returns the validated manifest to the store; OS sandbox profiles for it stay at M4 (RFC-0034 §10.4; OQ-SEC-02).

```rust
pub trait RangeSource {
    fn len(&self) -> u64;
    /// Reads exactly `buf.len()` bytes at `offset`. Hub sources issue an HTTP range request;
    /// requests are coalesced and bounded by `store.header_max_bytes`.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), Error>;
}
```

#### 2.1 GGUF

| Item | Rule |
|---|---|
| Versions | GGUF v2 and v3, little-endian. v1 and big-endian files are rejected (`(Formats, 2)`) |
| Limits | tensor count ≤ 2²⁰; KV pairs ≤ 2¹⁶; one string ≤ 64 MiB; one array ≤ 2²⁴ elements; whole header ≤ `store.header_max_bytes` (default 1 GiB, enough for large tokenizer vocabularies). Any excess is `Integrity` |
| Reading | the reader starts with the first 4 MiB and extends by range reads until the tensor-info table ends; it never reads the data section |
| Alignment | `general.alignment` (any power of two, default 32); the data section starts at the aligned end of the tensor infos |
| Split files | names `-NNNNN-of-MMMMM.gguf` plus `split.no`, `split.count`, `split.tensors.count`; every split's header is read; tensor names must be unique across splits; each record carries its file index |
| Types | each ggml type ID maps to `QuantScheme::Ggml(GgmlType)`, with block size and block bytes from RFC-0042 §2.1's table for the pinned ggml version. An unknown type ID is `Unsupported` naming the type, never guessed |
| Metadata | the family module (RFC-0043 §5) maps `{arch}.*` keys to `HyperParams`; `tokenizer.ggml.*` and `tokenizer.chat_template` go to `TokenizerSpec` |
| Expert tensors | a tensor that the family's classifier marks as a stacked expert tensor (for example `blk.L.ffn_{gate,up,down,gate_up}_exps.weight`, expert index outermost, `ne[2] = n_expert`) becomes `n_expert` slice records named `…[e]`, each with extent (`offset + e · nb[2]`, `nb[2]`) |

The file is never written. Local GGUFs are indexed in place (§4.3).

#### 2.2 safetensors

| Item | Rule |
|---|---|
| Container | 8-byte little-endian header length (≤ 100 MiB), then a JSON header; `__metadata__` is a string map; each tensor has `dtype`, `shape` and `data_offsets` relative to the data start. The reader checks offsets lie within the file, do not overlap, and match `numel × element size` (sub-byte types round up per row as the dtype defines) |
| Sharding | `model.safetensors.index.json` maps tensor names to files; each shard's header is read |
| Side files | `config.json` (architecture, hyperparameters, `quantization_config`), `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` (chat template), `special_tokens_map.json`; each is a source file with its own `FileDigest` |
| dtypes | `F32`, `F16`, `BF16`, `F8_E4M3`, `F8_E5M2`, `F8_E8M0`, `F4` (E2M1, two per byte), `F6_E2M3`, `F6_E3M2`, `I8`, `U8`, `I16`, `I32`, `I64`, `BOOL`. F6 tensors are readable but have no kernels; they get an exact widened copy in the pack (RFC-0042 §3) |

*Note 2026-10-22 (COMP-E10-T10, as built):* `syntropy_manifest::safetensors::SafetensorsDtype` reads every dtype above, and nothing else (`F64` is `UnsupportedContainer`). Each variant is named as `syntropy_core::DType` names its format (`F4` is `F4E2M1`).

*Revision 2026-10-22 (COMP-E10-T10; RFC-0001 §18.19) of the Container row.* The old wording was "match `numel × element size` (sub-byte types round up per row as the dtype defines)". The new rule is whole-tensor bit packing, as the format's reference reader does it. A tensor's elements are packed in order across the whole tensor, not row by row, so it holds `numel × bits` bits. A tensor whose bit count is not a whole number of bytes is `HeaderInvalid`, for example an `F4 [3]` or an `F6` scalar. Any other tensor takes `numel × bits / 8` bytes. The reason is exactness: the per-row rule refuses valid files, such as an `F6 [4, 2]` of 6 bytes, and accepts invalid ones, such as an `F4 [3]`. The row above keeps its original text. A tensor with a zero dimension takes no bytes, but an overflow in its other dimensions is still `HeaderInvalid`. A `data_offsets` length other than the computed one is `HeaderInvalid`. `BOOL` is one byte per element and has no `DType`, because RFC-0042 §1.1 lists no boolean element type, so `SafetensorsDtype::dtype` returns `None` for it.

`quantization_config` maps to `QuantScheme` with no conversion (ADR-0413):

| `quant_method` / `format` | Tensors involved | `QuantScheme` |
|---|---|---|
| none | the weight | `Plain(dtype)` |
| `fp8`, `weight_block_size: [128, 128]` | `W` (`F8_E4M3`) and `W_scale_inv` (`F32`, one per 128×128 block) | `Fp8 { format: F8E4M3, scale: Block { rows: 128, cols: 128 } }`; dequantised value = element × block scale (RFC-0042 §2.3) |
| `fp8`, per-tensor or per-channel | `W` and `W_scale` | `Fp8 { scale: Tensor \| Channel }` |
| `mxfp4` | `W_blocks` (`U8`, 16 bytes per 32 elements) and `W_scales` (`U8` holding E8M0) | `Mxfp4 { block: 32 }` |
| `compressed-tensors`, `pack-quantized` | `weight_packed` (`I32` holding 8 × INT4 or 4 × INT8), `weight_scale`, optional `weight_zero_point`, `weight_shape` | `CompressedTensors(CtScheme::PackedInt { bits, group, symmetric, zero_point })` |
| `compressed-tensors`, `float-quantized` | `weight` (`F8_E4M3`) and `weight_scale` | `CompressedTensors(CtScheme::Fp8 { strategy })` |
| `compressed-tensors`, `nvfp4-pack-quantized` | `weight_packed` (`U8`, E2M1 pairs), `weight_scale` (`F8_E4M3` per 16 elements), `weight_global_scale` (`F32`) | `Nvfp4 { block: 16, scale: F8E4M3 }` with the global scale as an auxiliary scalar |
| `compressed-tensors`, `mxfp4-pack-quantized` | `weight_packed`, `weight_scale` (E8M0 per 32) | `Mxfp4 { block: 32 }` |
| anything else | — | `Unsupported` naming the method and format |

Modules on the config's `ignore` list stay in their stored dtype. Scale, zero-point and global-scale tensors are bound to their weight as **auxiliary extents** of one logical `TensorRecord` (field `aux`, added with a serde default), so a group always carries everything its kernel needs and a shard never holds a weight without its scales. Stacked expert tensors (expert index in dimension 0) are sliced as in §2.1; per-expert tensors (`…experts.E.gate_proj.weight`) are taken as they are.

*Note 2026-10-22 (COMP-E10-T04, the container reader as built):* `syntropy_manifest::safetensors` reads the container, the sharded index and the side-file roles. It supports `F32`, `F16` and `BF16`; any other dtype is `UnsupportedContainer` (2) naming it until COMP-E10-T10. The choices this section leaves open:
- **Reads.** The first 8 bytes, then the header, through the shared header-read routine: two requests, no byte of the data section. A declared length above 100 MiB, or past `store.header_max_bytes`, is `HeaderLimit` (5); a header past the file's end is `HeaderInvalid` (6).
- **Strict JSON.** The header must start with `{`, may be padded with trailing whitespace, and must not repeat a key at any level, so a tensor named twice or given two `data_offsets` is `HeaderInvalid`. A tensor entry has exactly `dtype`, `shape` and `data_offsets`; another field is `HeaderInvalid`. `__metadata__` must map strings to strings. Nesting is bounded by the JSON parser's recursion limit (128).
- **Offsets.** Each tensor's `[begin, end]` must satisfy `begin ≤ end ≤` the data section's length, and `end − begin = numel × element size`, with overflow checked. Tensors with bytes must not overlap; an empty tensor holds no byte, so it overlaps nothing. Gaps between tensors are allowed.
- **Index.** `model.safetensors.index.json` is read whole, bounded like a header. It may hold only `metadata` (whose `total_size` is recorded, not checked) and `weight_map`. Every shard name must be a plain file name: no path separator, `:`, NUL, `.` or `..`. The index and the shards must agree exactly: every indexed tensor in its shard, every shard tensor indexed to that shard, no tensor in two shards.
- **Side files.** `side::classify` gives each file name its role; `SourceRole::in_model_id` is true for weights, the index, `config.json` and `generation_config.json`, as §1's formula lists. Reading and hashing them are COMP-E10-T06 and COMP-E10-T07.

### 3. The census builder

`manifest::census::build(sources) -> Result<ModelManifest, Error>` runs on the requesting daemon (RFC-0001 §7.1) and reads headers only.

```text
1. read every source header (§2); collect raw tensors (name, scheme, shape, extents) and metadata
2. resolve the family: GGUF general.architecture, or config.json architectures[0] / model_type
      → graph::families().resolve(..); an unknown name is Unsupported (ADR-0003 decision 6: fail closed)
3. hparams = family.parse_hparams(metadata)                         -- RFC-0043 §5
4. for each raw tensor: c = family.classify(name, shape, &hparams)  -- RFC-0043 §4
      Classified { class, layer, role }     → keep
      Ignored { reason }                    → record in manifest.ignored (e.g. a vision tower in v1)
      Unknown                               → Unsupported naming the tensor (fail closed)
5. slice stacked expert tensors into per-expert records (§2.1)
6. bind auxiliary extents (scales, zero points) to their weights (§2.2)
7. form groups (rules below); compute bytes and read_per_token per group
8. kv = family.kv_formula(&hparams); tokenizer = family.tokenizer_spec(metadata)
9. speculative = family.speculative_support(&hparams, tensors)      -- MTP layers, known draft heads
10. ModelId (§1); totals; siblings (lazily, §3.2)
11. summary = census::summarize(&manifest, &boundaries)            -- §3.3; boundaries from the family's graph
```

#### 3.1 Grouping rules

RFC-0001 §5.3 fixes the grouping: one group per (layer, mixer), per (layer, dense FFN), per (layer, expert), per lookup table, one for the head, one per router. This RFC adds the details:

- **Norms and small vectors** join the group of the block they feed: the pre-attention norm joins the mixer group, the pre-FFN norm joins the FFN group (or, in a MoE layer, the router group), so every group is executable on its own.
- **Shared experts** form a `DenseEveryToken` group per (layer, shared); they are never `Expert`.
- **An expert group** holds the expert's gate, up and down matrices (or fused gate-up and down) with their auxiliary extents.
- **MTP modules** get `LayerIx` values after the last decoder layer, flagged in `LayerSpec` (`mtp: true`), and are grouped like any layer: their experts are `Expert` groups, their projection and output head are `Head`. The solver keeps them on the tail (RFC-0001 §6.4). An MTP module that embeds the next token reads its own `RowLookup` group over the embedding's `TensorRecord`, linked by `shared_with` to the prologue's group exactly as tied embeddings are below, so the tail holds the rows RFC-0011 already charges for tail-side drafters. An MTP module that shares the LM head reads the epilogue's `Head` group, which is on the same stage.
- **Tied embeddings.** When the LM head shares the embedding tensor, the census emits one `TensorRecord` and two groups: a `RowLookup` group for the head stage and a `Head` group for the tail stage, linked by a new `TensorGroup::shared_with: Option<TensorGroupId>`. The solver counts the bytes on both nodes (RFC-0011's duplicated embedding/head term); the store fetches the chunks once per node.
- **`read_per_token`** at batch 1: all bytes for `DenseEveryToken`, `Head` and `Router`; `rows_per_token × row_bytes` for `RowLookup`, where the family supplies `rows_per_token` (1 for token embeddings, the n-gram fan-out for n-gram tables); `bytes × top_k / n_experts` for an expert group until RFC-0023's routing statistics replace the prior.

#### 3.2 Siblings and size

`siblings` lists the other quantizations in the same Hub repo or local directory, with byte totals per class. They come from header range reads of the sibling files, fetched lazily the first time the solver or the model picker asks (RFC-0051), and cached by `FileDigest`.

A manifest for a Kimi-K3-class model holds about 93 × 896 × 3 ≈ 250 K expert tensor records [estimate from `backends-and-portability.md` §4.1]. At about 90 bytes each that is ≈ 22 MB before compression [estimate]. Manifests are stored and sent zstd-compressed, by digest over the bulk plane (RFC-0001 §5.10, `RegisterModel { manifest: ManifestDigest }`).

#### 3.3 The census summary

Previews run before a model is registered, and often for models no node has read headers for (RFC-0011 §14.1, RFC-0051 §3.1). They need a summary that is small enough to ship in the catalogue and complete enough for the solver's DP. It is not detailed enough for per-group assignment, so a preview built from it carries `precision: Summary`.

```rust
/// A deterministic projection of a ModelManifest. Owner: RFC-0041; consumers RFC-0011 §14.1, RFC-0051 §3.
pub struct CensusSummary {
    pub version: u16,                       // schema version; consumers accept current and previous
    pub model: Option<ModelId>,             // None for a catalogue variant that is not yet registered
    pub family: FamilyTag,                  // RFC-0043 §5.1's frozen tag
    pub dtype_class: DTypeClass,            // the model's class (RFC-0042 §1.2)
    pub n_layers: u32, pub hidden: u32, pub vocab: u32, pub residual_streams: u8, pub max_context: u32,
    pub layers: Vec<LayerSummary>,          // decoder layers in order, then MTP layers
    pub lookups: Vec<LookupSummary>,        // one per RowLookup table
    pub head_bytes: u64,
    pub kv: KvFormula,                      // RFC-0022 §2, per layer
    pub boundaries: Vec<(u32, u32)>,        // legal boundary index, bytes per token crossing it (RFC-0043 §6.1)
    pub read_per_token: u64,                // batch 1, with the top_k / n_experts prior for experts (§3.1)
    pub total_bytes: u64,
}
pub struct LayerSummary {
    pub spec: LayerSpec,                    // mixer, FFN, mtp (RFC-0043 §5.2)
    pub dense_bytes: u64,                   // DenseEveryToken plus Router bytes of the layer
    pub n_experts: u16, pub top_k: u16, pub n_shared: u16,
    pub expert_bytes: u64,                  // bytes per routed expert, uniform within a layer
    pub expert_scheme: Option<QuantScheme>,
    pub dense_scheme: QuantScheme,          // the scheme carrying most of dense_bytes
}
pub struct LookupSummary { pub bytes: u64, pub row_bytes: u32, pub rows_per_token: u32 }
```

- **One function builds it.** `census::summarize` is a pure function of the manifest and the family's `BoundarySet`. The census builder calls it at step 11, and the catalogue build (RFC-0051 §3.2) calls the same function on the same headers. A catalogue summary therefore equals the one a node would compute from those files, and a test checks that for every catalogue variant.
- **What is exact and what is a prior.** Byte totals, per-expert sizes, top-k, lookup and head bytes, the KV formula and the boundaries are exact. Expert popularity is not in the summary, so `read_per_token` uses the uniform prior until RFC-0023's statistics exist.
- **Size.** About 100 bytes per layer plus the KV formula, a few KB for a 61-layer model [estimate], against ≈ 22 MB for the full manifest of the largest models. It is postcard-encoded and travels by value.

### 4. The model store

The store is a daemon component (`syntropy-store`); runners write packs through its API (RFC-0001 §3.2).

#### 4.1 Roots and layout

`store.dirs` lists writable roots (the first with enough free space receives new data) and `store.readonly_dirs` lists roots that are only read, such as a shared NAS mount (ADR-0016 decision 7). Each root has this layout:

```text
<root>/v1/
  blobs/<sha256>.blob                 source file bytes at their original offsets; sparse while partial
  blobs/<sha256>.inv                  inventory: chunk-table digest + bitmap of verified chunks (fsync'd)
  rangemap/<sha256>.map               chunks punched from the blob and held in a SOURCE-layout pack extent (§4.5)
  tables/<sha256>.chunks              chunk table of that file (§6.2)
  manifests/<model-id>.manifest       ModelManifest, postcard + zstd
  manifests/<model-id>.<quant>.refdigest   warm-up canary digest record (§5.3)
  refs/hub/<owner>/<repo>/<revision>  → resolved commit and ModelId
  refs/local/<alias>                  → ModelId
  local/<model-id>.sources            in-place local files: path, FileDigest, (device, inode, size, mtime)
  packs/<device-class>/seg/<n>.seg    pack extent segments (§5.2)
  packs/<device-class>/extents.idx    ExtentKey → (segment, offset, length, BLAKE3); no reference counts
  packs/<device-class>/side/<pack-key>.side      the sidecar, immutable once committed
  packs/<device-class>/side/<pack-key>.<kind>    side records: selection entries, expert profile (§5.2)
  packs/<device-class>/build/<pack-key>.intent   build intent, present only while a build is in progress
  staging/<sha256>/<chunk>.part       partial downloads, promoted only after verification
  sessions/                           persisted session KV (RFC-0022)
  locks/                              advisory locks; one writer process per root; runner pack leases
```

**The `<device-class>` directory name** (ADR-0405 decision 6; added 2026-10-22). It is one text form of the `DeviceClass` value (RFC-0040 §2.2). The form is a pure, one-to-one function of the value, and it never changes between releases for an unchanged value. It uses lowercase ASCII letters, digits and `-` only, so it is a valid directory name on every supported file system:

| Variant | Form | Example |
|---|---|---|
| `Cuda { sm }` | `cuda-sm<sm in decimal>` | `cuda-sm89` |
| `Hip { gfx }` | `hip-gfx<gfx in lowercase hex>` | `hip-gfx1100`, `hip-gfx942` |
| `Metal { family, metal4 }` | `metal-<family in lowercase>`, then `-metal4` when `metal4` is set | `metal-apple9-metal4` |
| `Vulkan { vendor, coopmat, subgroup }` | `vulkan-<vendor in lowercase hex, at least 4 digits>-sg<subgroup in decimal>`, then `-coopmat` when `coopmat` is set | `vulkan-1002-sg32-coopmat` |
| `Cpu { isa }` | `cpu-<isa in lowercase>` | `cpu-x86v4amx`, `cpu-arm92sme2` |
| `Npu { vendor, model }` | `npu-<vendor in lowercase hex>-<model in lowercase hex>` | never written in v1: no provider produces it, so no pack is keyed by it |

A field added to a variant later is written only when its value differs from its default, so existing directories keep their names. A new variant gets a new prefix. When a node's class changes, for example after an OS update that enables Metal 4, a granted AMX permission or a `features.disable` change, its new packs go in a new directory. The old directory is never renamed. The store reclaims it under the pack budget like any unused pack (§4.4).

Backend modules are not kept in the store. They live in the running version's `modules/` directory (RFC-0052 §4.3, RFC-0054 §5.1), and `kernels.module_dir = <store>/modules` is only a developer override (RFC-0040 §12).

#### 4.2 Sparse source blobs

A source file that a node holds only in part is a **sparse blob**: a file of the source's full length, created sparse (`ftruncate` on Linux and macOS; `FSCTL_SET_SPARSE` on Windows), into which verified chunks are written at their original offsets through the direct-I/O write path (ADR-0013 decision 1). Holes cost no disk. The inventory bitmap, not the file, says which ranges are valid. Readers (Loading, read-around, P2P serving) consult the bitmap and never read a hole.

When every chunk of a blob is present, the store computes the whole-file SHA-256 and compares it with the `FileDigest` (ADR-0016 decision 5). A match marks the blob complete, and the blob is then byte-identical to the origin's file. A mismatch revokes the file's chunk table (§6.3) and refetches.

Volumes without sparse-file support (exFAT, FAT32; detected at store start by allocating a 1 GiB sparse probe file) switch that root to **chunk-file mode**: one file per chunk under `chunks/<id[0..2]>/<id>`. Reads that span chunks issue one request per chunk. This mode is slower and is flagged in `syntropy doctor`.

Disk usage is accounted in allocated blocks, not file lengths.

**Revision note, 2026-10-29 (COMP-E11-T02): sparse blobs, the inventory and chunk-file mode, as built.** `syntropy_store::blob` holds them, and the OS calls are `syntropy_io::os::{set_sparse, punch_hole, allocated_bytes}`, since `store` has no `unsafe`.
- **The probe.** `Blobs::open` runs at a writable root's start. It creates `staging/sparse-probe`, marks it sparse, extends it to 1 GiB, writes 1 MiB of incompressible bytes in the middle, punches them out, and removes the file. The volume has sparse files only when the extension allocated less than half its length and the punch freed at least half of the 1 MiB. Otherwise the root is in chunk-file mode, and `Blobs::mode` gives the reason: the sparse flag refused (Windows on FAT32 and exFAT), the extension allocated (exFAT and FAT32 on Linux and macOS, HFS+), the punch refused, or the punch freed nothing. `doctor` reports it as STO-005 (RFC-0053 §5.2).
- **The inventory**, `blobs/<sha256>.inv`, is a fixed-size file updated in place: a 64-byte header (magic `SYNCINV1`; version u32 1; a mode byte, 0 sparse blob and 1 chunk files; a complete byte; two reserved bytes; the chunk table's shape digest; the chunk count as a u64; eight reserved bytes), then a bitmap of ⌈count / 8⌉ bytes, bit *i* mod 8 of byte ⌊*i* / 8⌋ for chunk *i*, then in chunk-file mode each chunk's 32-byte `ChunkId`. The "chunk-table digest" of §4.1 is `ChunkTable::shape_digest`: a BLAKE3 over the `FileDigest`, the file length and each record's offset, length and header flag, but not the IDs or their evidence, so the bitmap stays valid while IDs are committed (§6.3). An inventory is read only against a table of the same shape.
- **Order of writes.** A chunk's bytes are written and synced before its bit is set and synced; in chunk-file mode the chunk file is written under a temporary name, synced and renamed, its directory synced, and the ID written and synced first. An eviction clears and syncs the bit before it punches the range or deletes the chunk file, so a crash never leaves a bit for bytes that are gone. `Blob::reclaim` frees the blocks a crash between the two steps leaves. A write whose bytes fail a committed `ChunkId` is `ChunkMismatch` (7) and writes nothing.
- **Chunk files are shared by ID.** Two chunks with the same bytes, in one file or two, use one chunk file. An eviction deletes it only when no inventory on the root still holds that ID, and a malformed inventory on the root stops every such delete.
- **Completion.** With every chunk held, `Blob::complete` hashes the file; a match sets the complete byte, and a mismatch is `FileDigestMismatch` (8) and evicts every chunk, so nothing of the file is served until the caller revokes its table and refetches.
- **Not yet.** Writes are buffered positional writes, each synced, until `syntropy-io` offers the direct-I/O write path of ADR-0013 decision 1. Blobs on read-only roots are not opened yet.

#### 4.3 Local models, indexed in place

A local path (`syntropy run ./model.gguf`, or files under a configured models directory) is indexed, never copied or modified:

1. Read the headers (§2) and build the census.
2. In one multi-threaded pass over each file, compute its SHA-256 (`FileDigest`) and the BLAKE3 of each chunk (§6.1). The chunk table is complete from the start.
3. Record (path, `FileDigest`, device, inode, size, mtime) in `local/<model-id>.sources`.

A change to size, mtime or inode invalidates the record; the next use re-indexes and yields a new `ModelId`. Bytes read from a local source are copied into the tier manager's memory or into pack extents at `Loading`/`Packing`, so a later edit of the user's file cannot change a running instance. The one exception is read-around of `SOURCE`-layout groups when the pack budget is full (§5.1), which re-reads the user's file at runtime. Such reads are checked against the chunk table when `store.verify_local_reads` is on (default off: the file is the user's own and was hashed at indexing).

Indexing cost is one sequential read of the files the node holds. SHA-256 is a sequential chain within a file, so one file hashes at one core's rate, roughly 1.5–2.5 GB/s with SHA extensions [estimate]; parallelism comes only from hashing split files side by side (BLAKE3 chunk hashes are parallel within a file). The bound is `t_index ≥ max(Σ bytes / W_read, max_f bytes_f / R_sha, Σ bytes / (min(cores, files) · R_sha))`. A 404 GB GGUF in nine ≈ 45 GB split files is drive-bound at 6 GB/s, ≈ 67 s [estimate]; a single unsplit 120 GB file takes ≥ 48 s on its own whatever the drive [estimate]. The census is available as soon as the headers are read, so the model picker can show fit estimates at once. `ModelId` needs the SHA-256 that this pass produces, so the pass gates `RegisterModel` for local models. On first use the CLI shows its progress (RFC-0051), and the result is cached, so later starts skip it.

**Revision note, 2026-10-28 (COMP-E11-T03): the local record and what it trusts.** `syntropy_store::local` writes `local/<model-id>.sources` as one JSON object: `version` 1, the `ModelId` in hex, the family tag, and per file its canonical absolute path, role name, SHA-256 in hex, size, device, inode and mtime (seconds and nanoseconds). It is written under a temporary name, synced, renamed into place, and the directory is synced. A record whose name, fields or recomputed `ModelId` disagree is ignored, and the model is hashed again. A path that is not valid Unicode is not recorded, so such a model is hashed at every start. The stamp is read through the open handle: (device, inode) is `st_dev` and `st_ino` on Unix, and the volume serial number and 64-bit file index on Windows (on ReFS that index may not be unique, so size and mtime carry more of the check there). An mtime has the file system's granularity (a coarse-clock tick on Linux, 1 s on HFS+, 2 s on FAT32), so a write in the same tick as the file's previous write leaves the stamp unchanged; one that lands while the file is hashed would pair the stamp with a digest of bytes the file no longer holds. So the racily-clean rule applies: a file whose mtime is less than 2 s before the pass's start, by the wall time of the clock the pass is given, is recorded as racy, and a racy stamp is never reused. The next pass hashes the file again, and by then a later write has moved its mtime. A settled stamp that matches is trusted without reading the file. So a file rewritten in place at the same size, with its mtime set back, keeps its old digest; this section already accepts that for the user's own files. A file whose stamp changes while it is hashed is `LocalSourceChanged`; a same-tick change is caught by the next pass instead, through the racy rule. `check` compares stamps only and does not see a same-tick change. The pass stamps every file and closes it, then hashes up to `min(cores, files)` files side by side, each whole on one thread that opens it and checks its stamp first, so at most one file per thread is open. Chunk hashes in the same pass are COMP-E11-T14.

**Revision note, 2026-10-28 (COMP-E11-T14): the chunk hashes in the indexing pass, as built.**
- **One read, two hashes.** `syntropy_manifest::chunk::hash_chunks` reads each file once, front to back, in 1 MiB reads. The reading thread feeds SHA-256, and a second thread feeds the same buffers to one BLAKE3 hasher per chunk, cutting at the boundaries of §6.1. So the two hashes overlap rather than add up.
- **The chunks come from the census.** It computes them from each file's header with the slices of its stacked tensors (`gguf_chunks`, `safetensors_chunks`). A side file with no tensors, such as `config.json`, is all header region.
- **The table** is written to `tables/<sha256>.chunks` with every chunk's ID, so a local source's table is complete when the model registers. The IDs carry evidence form 0 (provisional): they are this node's own, and in a cluster they count as a witness only as §6.3 and COMP-E11-T17 say.
- **What a later start needs.** It skips a file only when its stamp is settled and unchanged and its table is on disk, matches the file's chunks and holds every ID. A racy stamp re-reads the file and rewrites its table.

#### 4.4 Garbage collection and budgets

- **Chunks** are referenced by the shards of plans this node participates in, and by being the only cluster copy of a chunk another node needs.
- **Second copies for repair** (RFC-0033 §9.3) are a garbage-collection class of their own. A second-copy chunk is held under a standby reference for the committed plan whose repair solve predicted this node as the absorber, never under a shard reference. It is the first thing reclaimed (below), before any other chunk, and a standby reference never counts as "the only cluster copy" above, so that protection does not keep it. It is dropped as soon as a newer plan no longer predicts this node for that stage. A copy is admitted only if, after writing it, the target volume keeps at least `store.min_free_bytes` plus a margin of one more `store.min_free_bytes` free (2 × `store.min_free_bytes`, 40 GiB at the defaults), and only while the node's second copies stay within `store.standby_max_bytes`. Second copies are evicted first anyway, so the margin exists to stop churn, where a copy is written, evicted by the next fetch's admission and written again. Unreferenced chunks are evicted LRU when space is needed, punching holes in sparse blobs (`fallocate(FALLOC_FL_PUNCH_HOLE)` on Linux, `F_PUNCHHOLE` on macOS, `FSCTL_SET_ZERO_DATA` on Windows) and updating the bitmap.
- **Pack extents** are collected by mark and sweep, never by persisted reference counts. The live set is every extent named by a committed sidecar on the root, plus the extents listed in any build-intent file, plus every extent the range map (§4.5) names for a chunk that a shard still references. The store's writer process computes it at start and before each sweep, under the root's writer lock. An extent outside the live set is dead; a segment with no live extent is deleted. A crash anywhere in the commit protocol (§5.2) leaves at worst unnamed extents, which the next sweep reclaims.
- **Sidecars** are evicted LRU by last use, per model, when `store.pack_max_bytes` is exceeded (RFC-0054), except sidecars under a **lease**. A runner takes a lease on its `PackKey` from the store before `Loading` and holds it until it exits. The lease is an advisory lock file in `locks/`, held open by the runner process, so it is released by the OS when the runner dies. A leased pack's sidecar is never evicted, and no segment holding one of its extents is compacted.
- **Free-space admission**: a fetch or pack build is admitted only if the target volume has room for it plus `store.min_free_bytes`. Otherwise the store reclaims in this order, cheapest loss first, and fails with `Capacity` naming the volume only when all three fall short: (1) second copies held for repair (above; they are evicted before any other chunk); (2) source ranges that a committed `SOURCE`-layout extent already holds, punched under §4.5, which loses no bytes from the node; (3) LRU eviction of unreferenced chunks and sidecars.
- Re-placement frees a node's old chunks only after the new plan is `Ready` (ADR-0016 decision 2).

#### 4.5 Source ranges held only in a pack extent

§5.1 packs every `Nvme`-homed group while its source blob stays on disk, so an NVMe-tiered MoE plan needs about 1.7–2× the model's size: ≈ 404 GB of source plus ≈ 279 GB of pack, ≈ 683 GB and up to ≈ 800 GB, for DeepSeek-V3 Q4 on S1's machine [estimate, from vision §1.3 and §5.1]. When disk is short, the store removes that duplication.

- **What qualifies.** A source chunk qualifies when (a) its `ChunkId` is known and was verified on this node, and in a cluster is committed with evidence (§6.3), so a refetch can be checked, (b) a committed pack extent in `SOURCE` layout holds its bytes unchanged, contiguously, and (c) that extent's hash verified when it was written or last loaded. Repacked layouts never qualify: their bytes are not the source's. Chunk boundaries fall on tensor and expert starts (§6.1), so a `SOURCE`-layout extent of one group covers whole chunks.
- **When it runs.** `store.punch_covered_sources = auto` (the default) punches qualifying chunks only when an admission (§4.4) would otherwise leave the volume below `store.min_free_bytes`, and only as many as that admission needs. Volumes with room keep both copies, which keeps re-plans and peer serving on the plain path. `never` turns it off.
- **The range map.** Before punching, the store records `(FileDigest, chunk index) → (ExtentKey, offset)` in `rangemap/<sha256>.map` beside the blob's inventory, and syncs it. Only then does it punch the chunk's range (or delete its chunk file in chunk-file mode, §4.2). The chunk's inventory bit stays set: the node still holds the chunk, so inventories, `AnnounceChunks` and origin assignment do not change. A crash between the two steps leaves a map entry for bytes still in the blob, which is harmless.
- **Reads go through the map.** `Loading`, read-around, a pack build for another device class, and peer serving (§8.4) read a mapped chunk from its extent, at the offset the map gives, resolved through `extents.idx`, so compaction never rewrites the map. Holes are still never read.
- **Losing the extent.** An extent named by the map stays live while a shard references any of its chunks (§4.4). If its only sidecar is evicted, the store first writes the chunks back into the blob when the volume has room; otherwise it clears their inventory bits, and the fetch planner refetches them from cluster holders or the origin when they are next needed. An extent that fails its hash (`ExtentMismatch`, 9) clears the bits of every chunk it held.
- **What users see.** The plan reports the true footprint both ways: `shard source bytes + S_pack` with both copies kept, and `shard source bytes + S_pack − S_pack,SOURCE` with every qualifying chunk punched, where `S_pack,SOURCE` is the pack's `SOURCE`-layout part. RFC-0051's picker shows the true `disk_needed` from these.

### 5. Packs

ADR-0410 records the format decisions.

#### 5.1 What goes into a pack

A pack is built per `PackKey` for the groups of one node's shard on one device class. A group gets a pack extent when at least one of these holds:

1. **Its home tier on this node is `Nvme`.** The runtime reads it on demand and by prefetch, so it needs one aligned request per group (ADR-0005).
2. **Its selected layout is not `SOURCE`** (RFC-0040 §5.4, §6.1). Storing the repacked bytes means a restart does not repeat the repack.
3. **`pack.copy_source_layout = true`**, set for sources on slow or remote storage such as a read-only NAS root.

Every other group (homed in `Vram`, `Unified`, `HostPinned` or `HostPageable`, in `SOURCE` layout) is **loaded straight from its source blob** at `Loading`, with an aligned read-around (§5.5). It is read once per load, sequentially, so alignment waste does not matter and a pack copy would only double disk use. The sidecar lists these groups as `source_groups`.

On a typical MoE desktop plan (example (b2) in the cost model: hot set in VRAM, the hottest experts in RAM, the rest on NVMe) the pack therefore holds the NVMe-homed experts plus any CPU-repacked experts. On a plan where everything fits in memory and no repack is chosen, the pack is only a sidecar.

#### 5.2 Segments, extents and the sidecar

**Segments.** Extents live in append-only segment files per device class, `packs/<device-class>/seg/<n>.seg`, shared by every pack of that class on the node. A segment is preallocated in steps of `pack.segment_bytes` (default 4 GiB) and written only at its tail, with direct I/O. Its first alignment unit is a header: magic `SYNSEG01`, the **pack container version**, device class, segment number, alignment, creation time. Segments are never rewritten in place. A background compaction copies the live extents of an unleased segment that is more than half dead into a new segment, `fsync`s it, appends the new locations to the index and `fsync`s that, and only then deletes the old segment. Sidecars name extents by `ExtentKey`, so compaction never rewrites a sidecar. A crash during compaction leaves both copies; the index's later record wins and the sweep removes the old segment.

**Leases and compaction, ordered by generation.** The extent index carries a generation number `g`, written in its records and raised by one each time a compaction commits its new locations. Leases are granted, and generations committed, under the root's writer lock, so a lease never sees a half-committed generation. A lease records the generation current when it is granted, and its resolved location table is that generation's. A compaction runs in three steps:
1. At start, under generation g, it picks a segment with no extent of a current lease, then copies that segment's live extents into a new segment and syncs it.
2. It appends their new locations as generation g + 1, syncs the index, and commits.
3. It deletes the old segment only if no lease still holding a generation ≤ g names an extent in it. Otherwise the deletion waits until those leases are released, and the segment, by then dead, goes in the next sweep.

A lease granted while a compaction is between steps 1 and 2 therefore pins the pre-compaction generation, and its runner keeps reading the old segment until it releases. A lease granted after step 2 reads the new locations.

**One container version.** The pack container version covers the segment header, the index record format and the sidecar format together. It is written in all three, and the reader accepts only the current version; a container change makes existing packs misses, which are rebuilt (packs are caches). The side records below are versioned separately, because they outlive layout changes (RFC-0062 §10.1).

**Extent alignment.** `A = max(4096, the volume's direct-I/O offset alignment (STATX_DIOALIGN on Linux), the sector size on Windows, the OS page size on macOS)`. The macOS page size (16 KiB on Apple silicon) lets a `MacMapped` volume map an extent directly. Every extent starts on a multiple of `A` and its stored length is padded to a multiple of `A`.

**Extent layout.** Components (weight, block scales, zero points, global scale) follow one another in the order the layout defines, each starting on a 64-byte boundary:

```text
offset k·A:  [component 0][pad to 64 B][component 1][pad to 64 B] … [pad to A]
```

Overhead is at most `A − 1` bytes per extent plus 63 bytes per component: for a 1.69 MiB Qwen3.6-class expert at Q4_K with A = 4 KiB, under 0.3 % [estimate from `backends-and-portability.md` §4.3].

**Extent index.** `packs/<device-class>/extents.idx` maps `ExtentKey` → (segment, offset, stored length, BLAKE3 of the stored bytes). It is an append-only log of records, in which a later record for a key replaces an earlier one, with a periodic compacted snapshot. Locations are resolved through it at `Loading`: the store hands a leased runner its pack's resolved location table. If the index is lost, it is rebuilt from the location hints in the sidecars, each checked against the extent's hash; a hint made stale by compaction fails the check, and that extent is rebuilt.

**Sidecar.** One file per `PackKey`, postcard-encoded and zstd-compressed:

```rust
pub struct PackSidecar {
    pub magic: [u8; 8],                         // b"SYNPACK1"
    pub version: u16,
    pub key: PackKey,                           // RFC-0001 §5.6
    pub manifest: ManifestDigest,
    pub build: BuildId,                         // the binary that wrote it
    pub created_by: PlanEpoch,                  // RFC-0001 §5.6: informational, not part of the key
    pub alignment: u32,
    pub extents: Vec<SidecarExtent>,            // sorted by (TensorGroupId, layout)
    pub source_groups: Vec<TensorGroupId>,      // loaded from source blobs (§5.1)
}
pub struct SidecarExtent {
    pub group: TensorGroupId,
    pub key: ExtentKey,                                // the extent's identity; locations come from the index
    pub layout: LayoutVersion,
    pub readers: SmallVec<[ReaderId; 4]>,              // who reads this copy (RFC-0040 §5.4)
    pub len: u64,                                      // stored (padded) length
    pub hash: Blake3,                                  // over the stored bytes
    pub hint: (u32, u64),                              // (segment, offset) at commit; used only to rebuild a lost index
    pub components: SmallVec<[ExtentComponent; 6]>,    // (TensorId, part, offset in extent, length)
}
/// Who reads a stored copy (RFC-0040 §5.4). Defined in `core`; `ProviderName` is the owned text
/// of the provider's ID, so the sidecar postcard-encodes without `kernels` types.
pub struct ReaderId { pub device: DeviceId, pub provider: ProviderName }
/// Envelope for data the store keeps for other crates and never interprets.
pub struct SideRecord {
    pub kind: [u8; 8],          // b"SELECT01", b"XPROFILE", b"REFDIGST"
    pub schema: u16,            // the owner's schema version for `bytes`
    pub bytes: Vec<u8>,         // encoded and decoded only by the owner
}
```

The sidecar holds only types `manifest` owns or `core` defines, so it compiles where RFC-0001 §4.1 places the format.

**Which pack holds a multi-reader extent.** An extent lives in the pack of the device class whose tier holds that copy's home. Host and `Nvme` homes go in the CPU class's pack, even when a GPU reads the copy through admissions or copied misses. A GPU class's pack holds only copies homed in that GPU's VRAM. `readers` names every device that reads the copy, whichever pack holds it. Three kinds of data that other crates own are kept as `SideRecord` files beside it, each written by the same temporary-file-and-rename step:

| Record | File | Owner (encodes and decodes `bytes`) | Written |
|---|---|---|---|
| selection entries | `side/<pack-key>.select` | `syntropy-kernels` (RFC-0040 §6.5) | at the end of `Warming` |
| expert profile | `side/<pack-key>.xprofile` | `syntropy-tier` (RFC-0023 §2, §13) | periodically while `Ready`, at `Draining` and at shutdown |
| canary digest | `manifests/<model-id>.<quant>.refdigest` | `syntropy-graph` (RFC-0060 §5.8's `ReferenceDigest`) | once per (model, quantization), §5.3 |

Each owner keeps decoders for its current and previous `schema`, so a record migrates across one change (RFC-0062 §10.1). A record whose `schema` is unknown is ignored and rewritten, never an error.

**Commit protocol.** Write the build-intent file listing the `ExtentKey`s the build will name → `fsync` → write extents → `fsync` segments → append index records → `fsync` → write the sidecar to a temporary name → `fsync` → atomic rename → `fsync` the directory → delete the intent file. Extents the build reuses are protected from the start by the intent file. A crash before the rename leaves extents named by no sidecar, which the next sweep reclaims once the dead intent file (one whose builder holds no lock) is discarded. A crash after the rename leaves a complete pack, because every extent it names was indexed and synced first. The runner reports `PackCommitted { key, sidecar }` (RFC-0001 §3.5).

**Side records are updated in place; the sidecar and extents never change.**

#### 5.3 Building a pack

The build request is a `PackBuild { key, groups: Vec<(TensorGroupId, SmallVec<[LayoutChoice; 2]>)> }`, with `LayoutChoice { provider, layout, readers: SmallVec<[ReaderId; 4]> }` as picked by RFC-0040's layout pass. A group has one entry under the default shared policy, whose readers are all the group's readers, and one entry per stored layout when the plan marks it `per_site_layout` (RFC-0040 §5.4). The runner executes it in `Packing`, using the store's API and its own providers' `repack`.

```text
for (g, choice) in shard groups needing an extent (§5.1) × their LayoutChoices, in Loading order:
    k = ExtentKey(g's tensor hashes, choice's layout ID, layout version)
    if extents.idx has k: reference it                     -- incremental: no work
    else if pack.fetch_from_peers and a same-class cluster node lists k
            and its transfer is predicted faster than building: fetch it (§8.5)
    else: queue g for building

pipeline over a bounded pinned window (pack.build_window, default 1 GiB):
    read:   the I/O engine reads g's source extents (aligned read-around, §5.5) into the window
    repack: pack-builder threads (default: cores − reserved − 2) run provider.repack, or a copy
            for SOURCE layout; BLAKE3 over the result
    write:  the I/O engine appends at the segment tail with direct I/O, in batches of ≥ 8 MiB
commit (§5.2)
```

- **Incremental by construction.** A re-plan that changes a node's shard produces a new `PackKey`, but every group whose tensors and layout are unchanged has the same `ExtentKey` and costs nothing. Only new groups, or groups whose layout version changed after a provider update, are built.
- **Write once, verify cheaply.** The extent hash is computed over the buffer that is written. `Loading` verifies the extents of groups it makes resident in memory, as it reads them. Runtime reads of `Nvme`-homed extents are not re-hashed; `syntropy doctor --verify-packs` checks everything, and any mismatch is class `Integrity`: drop the extent and rebuild it.
- **No whole-model step.** A pack is built per shard, so a node never processes groups it does not run.
- **Canary digest.** The warm-up canary (RFC-0001 §3.3) needs a `ReferenceDigest` (RFC-0060 §5.8) on the node that runs the tail stage. It is a side record per (model, quantization), `manifests/<model-id>.<quant>.refdigest`, not part of any sidecar. The source, in order: a digest from the project's digest index for this `ModelId` and pack quantization, fetched with the manifest; otherwise the store's cached record; otherwise one computed during `Loading`. The digest depends only on the model and its quantization, not on placement or layout, because repacks are lossless and the reference dequantises exactly (RFC-0042 §7), so it is computed once per cluster and cached.
  - **Computing it.** `graph::reference` in F32Fixed semantics runs layer-major over the 32 canary tokens during `Loading`. For groups that `Loading` makes resident it reads the bytes as they are loaded. `Nvme`-homed groups are not read at `Loading` (§5.1), so for them the pass makes one extra sequential read of their extents or source ranges, budgeted below. On a multi-node plan each node computes its own layer range and sends the 32-token boundary hidden states (`32 × hidden × streams × 4 B`, about 0.9 MB at hidden 7168) to the next stage over the bulk plane; the tail finishes the head and the digest and publishes it to the store. No node needs weights outside its shard.
  - **Cost.** The pass is serial across stages: stage k starts its layers only when stage k−1's boundary states arrive. Its wall time is `t_canary = Σ_k F_k / F_ref,k + max_k (B_nvme,k / W_read,k)`, where `F_k` is stage k's share of the reference FLOPs, `F_ref,k` the reference executor's measured CPU rate on that node, and `B_nvme,k` the extra bytes read for `Nvme`-homed groups. RFC-0060 §5.8 estimates ≈ 2.4 TFLOP for a 37 B-active model, 12–24 s of CPU in total [estimate]; splitting it over stages does not shorten it. The target is that `t_canary` ends before the tail's `Warming` would otherwise end [target]; when it does not, `Warming` waits for it, and the wait is reported as first-start cost. It runs once per (model, quantization) per cluster, so only a first start pays it.
- **Fused pack-and-load (allowed optimisation).** When a newly built extent belongs to a memory-homed group, `Loading` may take it from the build window instead of reading it back, which removes one read from SC-23's budget. It is an optimisation, not a requirement.

#### 5.4 Pack budget

`store.pack_max_bytes` (RFC-0054) caps packs per root. When a build would exceed it after LRU eviction of other models' sidecars:

1. groups whose only reason for an extent is a non-`SOURCE` layout fall back to `SOURCE` layout, and selection picks among `SOURCE`-layout kernels (RFC-0040 §5.4);
2. `Nvme`-homed `SOURCE`-layout groups are read around from the source blob at runtime (§5.5), and the I/O engine reports the lower efficiency to the profile (ADR-0005 decision 4);
3. the plan records both as `assumptions`, so the dashboard shows the cost.

#### 5.5 Aligned read-around

To read a source extent `[o, o + n)` that is not aligned, the I/O engine reads `[⌊o/A⌋·A, ⌈(o + n)/A⌉·A)` into a staging slot and uses the inner range. The waste is below `2A` per read. For a sparse blob, both ends of the widened range must be in verified chunks; chunk boundaries fall on tensor starts and on 4 MiB steps (§6.1), so the widened range can reach into a neighbouring tensor's chunk. When that chunk is absent, the read is split at the chunk boundary and the edge is read with a buffered read of at most one alignment unit. RFC-0021 owns the mechanics; this RFC owns the rule that holes are never read.

### 6. Chunk tables

#### 6.1 Chunk boundaries

For each source file, boundaries are a pure function of its header, so every node computes the same chunks without talking to anyone:

1. a boundary at offset 0 and at the start of the data section;
2. a boundary at every tensor start, every auxiliary-extent start and every expert-slice start (§2.1);
3. inside a run between two such boundaries, a boundary every `chunk_bytes` = 4 MiB (ADR-0016 decision 1);
4. padding between tensors belongs to the preceding chunk, so chunks tile the file exactly.

A shard therefore maps to whole chunks with no over-fetch, and identical tensor bytes in two files (a sibling quant sharing tensors, a re-upload) give identical chunks and identical `ChunkId`s.

#### 6.2 Table format

`tables/<sha256>.chunks` is a 64-byte header (magic `SYNCHNK1`, version, `FileDigest`, file length, entry count) followed by fixed 48-byte records:

| Bytes | Field |
|---|---|
| 0–7 | offset (u64 LE) |
| 8–11 | length (u32 LE) |
| 12–15 | flags (u32): bit 0 = ID committed, bit 1 = header region, bits 2–3 = evidence form (0 provisional, 1 `OriginHashes`, 2 `Witnesses`, 3 `KeyholderVerified`) |
| 16–47 | `ChunkId` (BLAKE3), zero until committed |

Record count is `Σ over boundary runs of ⌈run / 4 MiB⌉`, because rule 2 forces a boundary at every expert slice. For a Kimi-K3-class model, ≈ 1.4 TB over ≈ 250 K expert records (§3.2) is ≈ 5.6 MB per record, two chunks each, so ≈ 500 K records and ≈ 24 MB of table per model [estimate]. ADR-0016's ≈ 335 K assumed no forced boundaries.

**Revision note, 2026-10-28 (COMP-E12-T01): the header's widths, as `syntropy_manifest::chunk` writes them.** The text names the header's fields and their order. Their widths and positions, every integer little-endian, are:
- bytes 0–7: the magic `SYNCHNK1`;
- 8–11: the version, a u32, 1;
- 12–15: reserved and zero, so the digest starts on a 16-byte boundary;
- 16–47: the `FileDigest`;
- 48–55: the file length, a u64;
- 56–63: the entry count, a u64.

The records are as the table above gives them. A reader refuses:
- an undefined flag bit, or a record of length 0;
- an uncommitted record that carries an evidence form or a non-zero ID;
- a length that disagrees with the entry count;
- records that do not tile the file from offset 0.

On boundaries (§6.1), rule 2's auxiliary-extent starts are of two kinds. An unstacked weight's scale, zero-point or global-scale tensor is a tensor of the file, so its start is a tensor start. A stacked expert tensor's auxiliary tensor is stacked too, MXFP4's `W_blocks` and `W_scales` for example, and expert *e*'s auxiliary extent is that tensor's *e*-th slice (§2.1, §2.2), which starts inside it. So the caller passes the slices of every stacked tensor, weights and auxiliary tensors alike, and each expert's scale slice starts a chunk of its own. Each 4 MiB step of rule 3 counts from its run's start. A reader of a stored or received table also refuses a record longer than 4 MiB or one that runs past the file, and uses the table only if it matches the chunks it computes from the header (`ChunkTable::matches`).

#### 6.3 How chunk IDs become known (ADR-0412)

No node needs to hold a whole file, so a chunk's BLAKE3 can only be computed by a node that holds the chunk. **The first fetcher is never the sole authority.** A batch of chunk IDs binds the cluster only with evidence, in one of RFC-0034 §10.1's three forms:

1. **`OriginHashes`.** The origin publishes content-addressed chunk metadata for the file (the Hub's chunked-storage path, §7.4). The batch body records, for each `ChunkId`, the origin hashes covering its byte range. Every node that receives such a chunk, from the origin or from a peer, checks it against origin metadata it fetched itself over HTTPS, as well as against the `ChunkId`. The proposer is trusted for nothing.
2. **`Witnesses`.** Two `STAGE_HOST` members with distinct node keys each computed the entries from bytes they read themselves, and each signs (file SHA-256, batch digest). A witness reads either from the origin or from a local file whose `FileDigest` matched the origin's published file digest. Where there is no origin metadata, the second node that needs a single-witness chunk reads it from the origin and co-signs. It does not copy it from the first.
3. **`KeyholderVerified`.** A keyholder read the whole file, matched its SHA-256 against the origin's published digest, computed the table, and signed (file SHA-256, batch digest).

```text
local source:  the indexing node (§4.3) computes every ChunkId. Its table is one witness only if its
               FileDigest matches the origin's published digest; it is committed with a second witness
               or a keyholder signature. With no origin, RFC-0034 §10.1 item 4 applies: a keyholder
               import, or an ApproveImport request in which the keyholder streams and checks the file
origin fetch:  the fetching node verifies the bytes against the origin (TLS; plus the origin's chunk
               hashes on the chunked-storage path, §7.4), computes ChunkId, stores the chunk, and adds
               (index, ChunkId, covering origin hashes if any) to a pending batch
batch flush:   every 250 ms or 4096 entries, the node publishes the batch body on the bulk plane and sends
               ClusterCommand::ProposeChunkIds { file, range, body: Blake3, evidence: ChunkEvidence }
coordinator:   fetches the body by digest, checks every entry against the committed table:
                 empty → accept; equal → accept (idempotent); different → reject the batch
               checks the evidence (RFC-0034 §8.3); without valid evidence the batch is provisional
               appends ClusterEvent::ChunkIdsCommitted { file, range, body, evidence: Option<ChunkEvidence> }
               for an accepted batch (None: provisional); a later proposal of the same body with evidence
               upgrades it
every node:    fetches committed bodies by digest from any holder and fills its table copy
```

**Provisional chunks.** A node may use a provisional chunk it read from the origin itself. No node serves a provisional chunk to another node for packing (§8.2). Every node verifies every chunk it receives against the committed table before writing it. A one-node cluster needs no evidence, because only that node consumes its chunks.

**Files with no origin.** A local file that no origin publishes is covered by RFC-0034 §10.1 item 4, which uses the keyholder form with the operator standing in for the origin's digest. On a keyholder node, the import itself is the evidence. On any other node, the importer sends an `ApproveImport` request; once the operator approves, the keyholder streams the file once over the LAN, checks its SHA-256, recomputes the table and signs it. A matching digest alone is not evidence here. Until the approval, the batches stay provisional: the importer uses them, and no peer packs from them.

The log carries digests, about 122 events of ≈ 100 bytes for a 1.4 TB model [estimate: 500 K / 4096 batches], so the table stays out of the log's steady-state size (ADR-0016's revisit trigger). Bodies are served like manifests.

**Conflicts.** A rejected batch means a node computed a different hash for a committed chunk. The proposer discards its copy and refetches from the origin. If a fresh origin fetch still disagrees, the file is quarantined (`(Formats, 12)`): one node with origin access downloads the whole file and checks its SHA-256 against `FileDigest`. The correct side's table stands, and the other side's chunks are deleted. The coordinator records the event for the dashboard, and RFC-0034 decides whether a node that keeps disagreeing becomes suspect.

**Waiting for commitment.** A node that needs a chunk whose ID is not committed waits until the ID is committed with evidence and some holder lists the chunk (§8.1), then fetches and verifies. If the batch stays provisional because it has only one witness, the node reads the chunk from the origin itself and co-signs as the second witness (§8.3). Commitment adds at most one batch interval plus one coordinator round trip, small against the transfer itself.

### 7. Hub integration and fetch planning

#### 7.1 Resolving a reference

- Hub references are `owner/repo[@revision]` plus an optional file pattern or quant selector (the CLI's sugar is RFC-0051's). The store resolves the revision to a **commit hash** through the Hub API, and from then on uses only the commit hash. A branch moving later never changes a registered model.
- The file listing for that commit gives each file's path, size and content digest: the SHA-256 object ID for files stored through the Hub's large-file storage, which is the `FileDigest`. A file listed without a digest (small Git-stored files such as `config.json`) is fetched whole, which is cheap, and hashed locally.
- The base URL is configurable (`hub.endpoint`) for mirrors. API responses are recorded as fixtures in tests, so changes to the Hub API show up as test failures, not field failures.
- **Generic HTTPS origins** (ADR-0016 decision 3) must support `Range` and must supply a digest (`url#sha256=<hex>`, or a sibling `.sha256` file). Without one, the requesting node downloads the whole file once, hashes it, and registers it as a local source.

#### 7.2 Header reads

Headers are read with HTTP range requests against the commit-pinned resolve URL for each file (following redirects to the storage CDN with the `Range` header preserved): the first 4 MiB of a GGUF, extended as needed; the first 8 bytes of a safetensors file, then its header. Headers are cached in the store by `FileDigest`. Nothing past the header is read before a plan exists.

#### 7.3 Ranged chunk fetch

| Item | Rule |
|---|---|
| Request | `Range: bytes=o-(o+len−1)` for one chunk; adjacent chunks needed by the same node are coalesced up to 64 MiB per request |
| Concurrency | HTTP/2; at most `hub.max_streams` (default 16) requests in flight per node, adapted down on 429/503 |
| Errors | 429/503 honour `Retry-After`; other 5xx and resets retry with jittered exponential back-off up to `hub.max_retries` (default 8); 416 or a length mismatch is `Integrity` (the file changed, or the table is wrong) |
| Resume | a partial chunk keeps its received prefix in `staging/<sha256>/<index>.part`; the retry asks for the remaining range |
| Writes | received bytes go to the staging slot; after verification the chunk is written into its sparse blob at its original offset with direct I/O (§4.2) and the inventory bit is set and synced |

#### 7.4 The Hub's chunked storage path

Where the Hub serves a file through its content-defined chunked storage (`backends-and-portability.md` §3.3), the fetcher asks for the reconstruction information covering only the byte ranges it needs, downloads just those storage chunks, verifies them against the hashes that protocol provides, and assembles the requested range. This gives origin-side integrity on partial reads, which TLS alone does not. The reconstruction client lives behind the same `Origin` trait as plain range reads, so the fetch planner does not care which path served a chunk. It is an M2 deliverable. Until then, plain range reads serve every file, and the protocol's details are verified against the Hub's documentation when that client is written (Open questions).

#### 7.5 Credentials and offline mode

- Tokens for gated repos come from the node's own configuration or OS keychain (RFC-0034) and never leave the node (ADR-0016 decision 8). A node advertises only whether it has origin access for a repo, in its inventory announcement (§8.1). Origin assignment (§8.3) considers only nodes with access.
- `--offline` makes no origin calls. A fetch that needs the origin fails with the list of missing chunks and the files they belong to.
- **Licence acceptance on the receiving node** (from M3, before invites ship). When a gated or custom-licence model's chunks flow to another node, the receiving node's operator becomes a recipient of the licensed files. So for those models a node receives no chunk, from the origin or from a peer, until its operator has accepted the model's licence on that node: one prompt, recorded locally as (licence identifier, SHA-256 of the licence text, time, local user) under the store's state. A model is *gated* when its Hub repository requires accepting terms, and *custom-licence* when its licence is not on ADR-0018's allow list, as the catalogue entry or the model card's licence field states it (RFC-0051). Permissively licensed models need nothing.
  - The node announces acceptance per model beside `origin_access` in its inventory (§8.1). Holders serve such chunks only to nodes that announce it (§8.4), the fetch planner on an unaccepted node requests none, and origin assignment (§8.3) counts only accepted nodes as candidates.
  - A plan that places a stage of such a model on an unaccepted node is not refused. The plan preview lists the nodes that still need acceptance, and their fetches wait with `LicenceNotAccepted` (15) until the prompt is answered on each (RFC-0051 shows it).

#### 7.6 Fetch planning

On `FetchShard { spec, priority }` (RFC-0001 §3.5) the store:

1. expands `spec.groups` into tensor extents, then into (file, chunk index) pairs, using the manifest and the chunk tables;
2. subtracts chunks already verified locally (any root, including read-only roots);
3. orders the rest by group in the runner's `Loading` order, so `Loading` can begin before the last chunk arrives when the runner overlaps them;
4. picks a source per chunk (§8.2) and keeps at most `p2p.window` chunks in flight per source;
5. reports `ShardProgress` to the runner at least once per second.

`priority` raises a fetch above other fetches on the node: `Urgent` for recovery after a node loss (RFC-0033), `Normal` for loads, `Background` for pre-staging a sibling or a replica.

### 8. Peer-to-peer chunk distribution

The bulk plane is iroh/QUIC (ADR-0011, ADR-0016 decision 4) on ALPN `syntropy/blob/<major>`. RFC-0031 owns its byte layout, framing and pacing. This section defines the messages' meaning, how sources are chosen, and how transfers are verified and resumed.

#### 8.1 Inventories

Each node keeps, per source file it holds any part of, a roaring bitmap of verified chunk indices with a version number.

- **Coarse, in the event log:** `AnnounceChunks { node, model, have: ChunkSetDigest }` (RFC-0001 §5.10), sent when a fetch completes, when GC drops chunks, and at most once per `p2p.announce_min_s` (default 10 s) per model. `ChunkSetDigest` carries the BLAKE3 of the bitmaps plus a count, so nodes know whose inventory changed.
- **Fine, on the bulk plane:** `InventoryGet { file }` → `Inventory { file, version, bitmap, origin_access: bool, licence_accepted: bool }` (`licence_accepted` is true for models that need no acceptance, §7.5). A node that is fetching a file subscribes to the holders it uses and receives `InventoryDelta { file, from_version, added, removed }`, coalesced every 250 ms.

#### 8.2 Source selection per chunk

```text
sources(c) =
  1. local roots (writable and read-only)                                → done
  2. cluster holders of c (from inventories), excluding sources marked suspect for this model, and
     only if c's batch is committed with evidence (§6.3; provisional chunks are never served for packing),
     ranked by the measured bulk-plane β of the link to them (CapabilityProfile LinkCaps), then by load
  3. the origin, if this node is c's origin assignee or second witness (§8.3), or if c has waited p2p.origin_wait_s
     (default 30 s) with no holder and no progress from the assignee
```

A node stripes different chunks across up to `p2p.max_sources` (default 4) holders at once, with at most `p2p.window` (default 8) chunks in flight per source. A single chunk always comes from one source, which keeps verification and resume simple. When a holder is slower than its rank predicted for `p2p.reassess_s` (default 5 s), its queued chunks move to the next source.

#### 8.3 Origin assignment

ADR-0016 has the coordinator assign each chunk that no cluster node holds to exactly one fetcher. Doing that with one log event per chunk would put hundreds of thousands of events in the log, so the assignment is a **pure function of replicated state** that every node, including the coordinator, evaluates identically:

```text
candidates(c) = nodes whose shard in the newest committed plan needs c, that have origin access to its repo,
                and that have accepted its licence where the model needs acceptance (§7.5)
assignee(c)   = argmax over n in candidates(c) of   w(n) / −ln(u(c, n))
    u(c, n) = (BLAKE3(ChunkKey(c) ‖ n.node_id) as a 64-bit integer + 1) / (2^64 + 1)   -- uniform in (0, 1)
    w(n)    = 2^k(n), where k(n) is n's WAN rate class recorded in the newest committed plan: the
              log2 of its measured WAN download rate in MB/s, rounded down, as the solver saw it
              (RFC-0011). A node with no measured rate gets the median class of the measured candidates,
              or class 0 when none is measured
```

`w` is read from the committed plan, never from a live profile, so every node evaluates it from the same replicated value even while profiles are refined online (RFC-0001 §5.1). The power-of-two classes keep it stable across small measurement changes; a node moves class only when its rate changes by about 2×, and the change takes effect at the next plan epoch.

**Second witness.** For a file whose origin publishes no chunk metadata, the candidate ranked second by the same formula also reads c from the origin, if its shard needs c, and co-signs the batch (§6.3). A chunk only one node needs stays single-witness and provisional, which costs nothing, because no other node would be served it.

This is weighted rendezvous hashing: each chunk goes to exactly one candidate, the share of chunks a node gets is proportional to `w(n)`, and adding or removing a node moves only that node's share. `ChunkKey(c)` is (`FileDigest`, chunk index), since the `ChunkId` is not known yet. A coordinator change does not alter assignments at all. When the assignee leaves the plan or the cluster, the next candidate takes over at the next evaluation, and the `p2p.origin_wait_s` fallback (§8.2) covers an assignee that is alive but stuck.

#### 8.4 Transfer, verification and resume

| Message (bulk plane) | Meaning |
|---|---|
| `GetChunk { file: FileDigest, index: u32, id: ChunkId, offset: u32 }` | request chunk `index` from byte `offset` (0 for a new transfer) |
| `ChunkData { file, index, offset, bytes }` (streamed) | the bytes, from the holder's verified blob range, read with direct I/O into a registered send buffer |
| `ChunkRefused { file, index, reason }` | not held (stale inventory), over capacity, shutting down, or the requester has not accepted the model's licence (§7.5) |

- The receiver writes into a staging slot, computes BLAKE3 as bytes arrive, and on completion compares it with the committed `ChunkId`. Equal: write into the blob and set the inventory bit. Different: discard, mark the source **suspect** for this model for `p2p.suspect_ttl_s` (default 3600 s), refetch from another source, and report it to the coordinator (`ReportSuspectSource`, RFC-0034 owns the consequences).
- Chunks are at most 4 MiB, so a broken peer transfer restarts the chunk from the staged offset with the same or another source. A daemon restart resumes from the inventory bitmaps and the staging directory; staged prefixes older than `p2p.staging_ttl_s` are dropped.
- Serving runs on the daemon's `bulk` runtime (RFC-0001 §8.1), at most `p2p.serve_streams` (default 8) concurrent chunk streams per peer, and is paced so that ring hops on the same link keep their p99 (RFC-0001 §8.4 rule 6; RFC-0031).

#### 8.5 Pack extents between nodes of the same device class

ADR-0016 decision 7 lets a node take a pack from a cluster node with the same device class instead of rebuilding it. Because extents are content-addressed by `ExtentKey`, this is done per extent, not per pack: `GetExtent { class, key }` → `ExtentData { class, key, hash, bytes }`. The receiver checks the BLAKE3 against the declared hash, which guards against corruption in transit, not against a dishonest sender: rebuilding is the only way to check a repack independently. Cluster members are trusted with the weights they handle (ADR-0000 D4), but a member's repack is not something the receiver can check, so **`pack.fetch_from_peers` stays `false` at every milestone, M3 and later included** (OQ-SEC-06; this also settles the disagreement between this section and RFC-0034 §2's A5). The mechanism ships at M3 for users who opt in. When it is on: the setting is labelled "trusts members" in the configuration and the dashboard; the receiver rebuilds at least 1% of the received extents, chosen at random, and compares their hashes with what it received; and every model it serves from received extents runs the warm-up canary (RFC-0042 §7). A mismatch drops every extent from that sender, marks the sender suspect (§8.4) and rebuilds locally. §5.3 uses the path only when the predicted transfer time is lower than the predicted build time.

### 9. Configuration knobs

| Key | Default | Meaning |
|---|---|---|
| `store.dirs` / `store.readonly_dirs` | `[<data dir>/models]` / `[]` | store roots (§4.1) |
| `store.max_bytes` / `store.pack_max_bytes` / `store.min_free_bytes` | 90 % of the volume / 25 % of `store.max_bytes` / 20 GiB | budgets (§4.4, §5.4); RFC-0054 owns the defaults' final values |
| `store.header_max_bytes` | 1 GiB | parser limit on a GGUF header (§2.1) |
| `store.verify_local_reads` | `false` | hash-check runtime read-around from local user files (§4.3) |
| `pack.segment_bytes` | 4 GiB | segment preallocation step (§5.2) |
| `pack.build_window` | 1 GiB | pinned memory for the pack pipeline (§5.3) |
| `pack.copy_source_layout` | `false` | also copy memory-homed `SOURCE` groups into the pack (§5.1) |
| `pack.fetch_from_peers` | `false` | same-class extent transfer, opt-in, labelled "trusts members", with rebuild-and-compare sampling (§8.5) |
| `store.standby_max_bytes` | 10 % of the node's writable volumes, by total size [target] | per-node cap on the allocated bytes of second-copy chunks for repair, summed over the writable roots; 0 turns second copies off (§4.4, RFC-0033 §9.3); RFC-0054 owns the final value |
| `store.punch_covered_sources` | `auto` | `auto` punches source chunks held in a `SOURCE`-layout extent only when an admission would leave less than `store.min_free_bytes`; `never` keeps both copies (§4.5) |
| `hub.endpoint`, `hub.max_streams`, `hub.max_retries` | Hugging Face, 16, 8 | origin client (§7) |
| `p2p.window`, `p2p.max_sources`, `p2p.serve_streams` | 8, 4, 8 | transfer concurrency (§8) |
| `p2p.origin_wait_s`, `p2p.reassess_s`, `p2p.suspect_ttl_s`, `p2p.staging_ttl_s`, `p2p.announce_min_s` | 30, 5, 3600, 86400, 10 | timers (§8) |
| `offline` | `false` | no origin calls (§7.5) |

### 10. Failure handling and error codes

Codes are `(Domain::Formats, n)`.

| Code | Name | Class | Raised when | Action |
|---|---|---|---|---|
| 1 | `UnknownFamily` | `Unsupported` | the architecture string matches no registered family | refuse; the message names the string and the supported families |
| 2 | `UnsupportedContainer` | `Unsupported` | GGUF v1, big-endian GGUF, an unknown safetensors dtype | refuse with the reason |
| 3 | `UnknownTensor` | `Unsupported` | a tensor the family cannot classify | refuse naming the tensor (fail closed) |
| 4 | `UnknownQuant` | `Unsupported` | an unknown ggml type ID or `quantization_config` method | refuse naming it |
| 5 | `HeaderLimit` | `Integrity` | a parser limit is exceeded | refuse; the file is treated as malformed |
| 6 | `HeaderInvalid` | `Integrity` | overlapping or out-of-range offsets, a length mismatch | refuse |
| 7 | `ChunkMismatch` | `Integrity` | a received chunk fails its committed `ChunkId` | discard; mark the source suspect; refetch elsewhere |
| 8 | `FileDigestMismatch` | `Integrity` | a complete blob fails its `FileDigest` | revoke the file's table; quarantine (§6.3) |
| 9 | `ExtentMismatch` | `Integrity` | a pack extent fails its hash at `Loading` or `doctor` | drop and rebuild the extent |
| 10 | `NoSpace` | `Capacity` | a volume cannot admit a fetch or build after eviction | fail the fetch naming the volume; RFC-0011 may re-plan the node with a smaller shard |
| 11 | `OriginUnavailable` | `Link` | origin errors beyond retries, or `--offline` with missing chunks | the fetch fails with the missing chunk list; cluster sources are still tried |
| 12 | `TableConflict` | `Integrity` | conflicting `ChunkId` proposals for one chunk | quarantine and whole-file check (§6.3) |
| 13 | `GatedNoAccess` | `Auth` | no node with origin access needs or holds a chunk | the request fails naming the repo; the user must add a token on some node |
| 14 | `LocalSourceChanged` | `Integrity` | a local file's (inode, size, mtime) changed since indexing | re-index; a new `ModelId` results |
| 15 | `LicenceNotAccepted` | `Auth` | a node must receive chunks of a gated or custom-licence model whose licence its operator has not accepted (§7.5) | the fetch waits; the CLI and dashboard show the acceptance prompt on that node |

## Performance model and budgets

Every number here is a **target** or an **estimate** with its derivation; none is a measurement.

**Before placement: headers only.** A Hub census costs one listing call plus, per file, one to three range requests of at most a few MiB (§7.2). Target: census plus `ModelId` within `2 s + header bytes / WAN rate` for a Hub model [target]. A local model adds one sequential read of its files for `FileDigest` and chunk IDs, bounded below by §4.3's `t_index` (per-file SHA-256 is sequential), ≈ 67 s for a 404 GB model in nine split files at 6 GB/s [estimate], once per file version.

**Pack and load (SC-23).** SC-23 requires `t_pack + t_load ≤ 1.25 · (2·S/W_read + S/W_write)` for a shard of S bytes, with download counted separately. With §5.1's rule, the bytes split into memory-homed `SOURCE` groups `S_src` (read once, at `Loading`) and packed groups `S_pack` (read from the source, written, then read once more at `Loading`, or not re-read with the fused path):

```text
t_pack + t_load ≈ S_pack/W_read + S_pack/W_write + (S_pack + S_src)/W_read     (no fusion)
               ≤ 2·S/W_read + S/W_write                                         since S_pack + S_src = S
```

The pipeline overlaps read, repack and write, so repack CPU time is hidden while `repack throughput ≥ min(W_read, W_write)`. The CPU repacks are byte shuffles; they must run at ≥ 2 GB/s per core [target] so that 4 cores keep up with a Gen4 drive. The vision document's example: 42.5 GB on a Gen4 drive (7.45 GB/s read; 6 GB/s write assumed) is ≈ 18.5 s, so the gate is ≤ 23 s.

**Re-plan cost.** A re-placement writes only new extents and fetches only missing chunks. The solver's move cost (RFC-0011, RFC-0001 §7.2) is `Σ_n (missing chunk bytes_n / β_fetch,n + new extent bytes_n / W_write,n)`, with `β_fetch` the measured bulk rate from the chosen sources. The cost model's example, 377 GB at 3.5 GB/s, is ≈ 108 s (cost model §5.3).

**WAN once per cluster.** With weighted rendezvous assignment, origin bytes split across assignees in proportion to measured WAN rate, so a cluster's first download of a model takes about `U / Σ_assignees w(n)`, where U is the union of all shards, against `U / w(one node)` if one node fetched everything [estimate]. Chunks that several nodes need cross the WAN once and the LAN `k − 1` more times when the origin publishes chunk metadata. Without it, they cross the WAN twice (the second witness reads from the origin) and the LAN `k − 2` more times, so the first download of such a file costs up to `2U` of origin bytes [the bound, not a measurement].

**Disk.** Sparse blobs cost the shard's bytes. Packs add `S_pack` plus padding below 0.3 % for MB-scale experts (§5.2). Under disk pressure, §4.5 removes the duplicated `SOURCE`-layout part, so the footprint falls towards the shard's bytes plus the repacked part of the pack. Chunk tables cost ≈ 24 MB per 1.4 TB model (§6.2), and the extent index ≈ 80 bytes per extent [estimate].

**Hot path.** None of this RFC's code runs per token. The only runtime reads it shapes are `Nvme`-homed extents (one aligned request each, ADR-0005) and read-around under a full pack budget (§5.4), whose efficiency loss is measured and given to the solver.

## Alternatives considered

| Alternative | Why not |
|---|---|
| `ModelId` over per-tensor hashes of raw bytes (RFC-0001's current wording) | Needs every byte of the model on some node before placement, which contradicts header-only census (RFC-0001 §7.1) and shard-only fetch (ADR-0016). Container independence is kept where it matters, at the chunk level (ADR-0411) |
| Verify chunks only against the Hub's whole-file SHA-256 | Impossible for a byte range unless someone holds the whole file (ADR-0016 context) |
| Content-defined chunking (rolling hash) | Boundaries cannot be known from headers, so a shard cannot be mapped to chunks before download; tensor starts are better boundaries for this data |
| One monolithic pack file per `PackKey` | A re-plan rewrites the whole shard; no sharing between packs of different shards on one device class. The extent store makes rebuilds incremental (ADR-0410) |
| Pack every group, including memory-homed `SOURCE` groups | Doubles disk use for no runtime benefit; kept as `pack.copy_source_layout` for slow sources |
| Per-chunk origin assignments in the event log | Hundreds of thousands of log entries per model; rendezvous hashing gives the same one-fetcher property with no messages (ADR-0412) |
| Convert safetensors originals to GGUF at import | FP8 has no GGUF type, so conversion is lossy or widening; MXFP4/NVFP4 scale layouts must map exactly; rejected by ADR-0005 |
| Chunk-per-file storage everywhere | ≈ 500 K files for a large model, with open and metadata costs; kept only for volumes without sparse files |
| A public DHT or swarm protocol | Out of scope for v1 (ADR-0000 D4, ADR-0016) |

## Testing and validation

| Claim | Verification |
|---|---|
| Parsers are safe on hostile input | `cargo-fuzz` targets for the GGUF header, safetensors header, index JSON, `quantization_config`, sidecar, chunk table and inventory decoders, run in ClusterFuzzLite from M1; seed corpora come only from project-synthetic fixtures, never from headers cut out of third-party checkpoints (OQ-PRC-18; RFC-0060) |
| Census is right per family | golden tests on recorded headers of real Hub models (headers only): group counts, class totals, `read_per_token`, KV formula, `ModelId`. Headers carry the tokenizer and chat template, so recorded headers of a model that is not permissively licensed live in the private part of the fixture store, and only permissively licensed models' headers are public (RFC-0060 §12.3) |
| `ModelId` is stable | the same file through a local path, a Hub fixture and a sparse blob gives the same `ModelId` on Linux, macOS and Windows; golden IDs for the recorded fixtures are checked in and must not change across releases unless the release notes list a family-acceptance change |
| Chunk boundaries are deterministic | property test: boundaries depend only on the header; identical tensors in two synthetic files give identical chunks |
| Holes are never read | property test with random partial inventories and random read-around requests against sparse blobs; a canary byte pattern in holes must never appear in output |
| Packs are incremental | integration test: build for shard S, re-plan to S′, count bytes written; must equal the new groups' extents only |
| Crash consistency | fault injection at every `fsync`, rename and delete in the commit protocol, in compaction and in eviction, with a runner holding a lease throughout, and with a lease granted at each step of a compaction; after restart the store never serves a sidecar that references a missing or partial extent, a leased pack's extents stay readable at their resolved locations, and a sweep after recovery reclaims every unnamed extent |
| Quant schemes map without loss | per scheme, dequantise synthetic tensors through the native kernels and through `kernels::reference` (RFC-0042 §2) and compare bit for bit |
| Each origin byte crosses the WAN at most once with origin metadata, at most twice without | the same fabric with and without origin chunk metadata; a provisional chunk is never served to a peer for packing; a single-witness batch becomes evidenced after the second needing node's origin read | `syntropy-testkit` simulated fabric: 2–8 nodes, a rate-limited origin, overlapping shards; count origin bytes per chunk; kill the assignee mid-transfer and check the fallback |
| Corrupt sources are caught | a fault-injecting node serves flipped bytes; receivers reject them, mark the source suspect and finish from others |
| Chunk-ID commitment resolves conflicts | two simulated nodes propose different IDs for one chunk; the quarantine path runs and the correct ID stands |
| Hub client behaviour | recorded API fixtures plus a local range-serving HTTP server injecting 429, 503, resets and short reads |
| SC-23 | RFC-0061 benchmark per Tier 1 platform |
| Punched source ranges stay readable | with `store.punch_covered_sources = auto` and a volume driven below `store.min_free_bytes`, qualifying chunks are punched only after their range-map entry is synced; reads, peer serving and a pack build for another device class return the chunk's verified bytes from the extent; evicting the extent's sidecar writes the chunks back or clears their inventory bits; crash injection between the map write and the punch never loses a chunk; repacked extents are never used |
| Licences are accepted before chunks flow | a gated or custom-licence model is never served to, fetched by, or assigned as origin work to a node without a recorded acceptance; a permissively licensed model needs none; the acceptance record holds the licence text's hash |
| Received pack extents are checked | with `pack.fetch_from_peers = true`, at least 1% of received extents are rebuilt and compared, the canary runs, and a planted bad extent drops every extent from its sender |

## Rollout / milestones

| Milestone | Delivered |
|---|---|
| **M0** | GGUF reader; safetensors reader for `F32`/`F16`/`BF16`; census and `ModelId` for `llama` and `qwen_moe`; store layout with local indexing; whole-shard origin fetch on one node (Hub range reads, resume); no packs (`SOURCE` layouts loaded from blobs) |
| **M1** | packs (extent store, sidecars, commit protocol, incremental builds, expert profile slot); safetensors FP8, MXFP4, NVFP4 and compressed-tensors; census for the M1 roster; sparse blobs; chunk tables for local and origin sources on one node; punching source ranges held in `SOURCE`-layout extents under disk pressure, with the range map (§4.5); header parsing in a child process; SC-23 gate |
| **M2** | chunk-ID commitment through the coordinator; inventories; P2P transfer with verification and suspect handling; origin assignment by rendezvous hashing; the Hub's chunked-storage path; shard deltas on re-placement |
| **M3** | same-class pack extent transfer, off by default (§8.5); licence acceptance on receiving nodes before invites ship (§7.5); generic HTTPS origins; census for every v1 family; read-only shared roots |
| **M4** | `doctor --verify-packs`; segment compaction tuning; chunk-file mode hardening; fuzz corpus at release scale |

## Open questions

1. **The Hub's chunked-storage protocol details** (reconstruction endpoint, authentication, hash scheme) were not verified for this RFC. Default: plain range reads until the M2 client is written against the Hub's documentation; the `Origin` trait isolates the difference. Settled by an M2 spike against the published specification, which also adds a CI canary that fails when the protocol changes (COMP-E12-T08).
2. ~~Whole-file checks on shard-only clusters~~ **Answered (2026-10-20; OQ-MF-08):** accept TLS trust for origin bytes when no node holds a complete file, as ADR-0016's consequences state. No whole-file assembly job is built; the `FileDigest` is still checked whenever a node does hold a complete file (§4.2).
3. ~~Expert profile across device classes~~ **Answered (2026-10-20; OQ-MOE-14):** yes. A node gaining a device class seeds the new pack's expert profile from the sibling sidecar of the same `ModelId`, and the new pack's own counts take over after warm-up (RFC-0023 §5.6).
4. ~~Pack extent transfer trust~~ **Answered (2026-10-20; OQ-SEC-06):** `pack.fetch_from_peers` stays `false` after M3; if a user enables it, the receiver rebuilds and compares at least 1% of extents and runs every canary, and the setting is labelled "trusts members" (§8.5).
5. **Sparse-file behaviour on Windows and APFS at scale** (hole punching cost, fragmentation over months of GC). Settled by an M1 soak test on Tier 1 platforms.
6. ~~Manifest size for the largest models~~ **Answered (2026-10-20; OQ-MF-10):** keep explicit per-expert records with zstd and bulk-plane transfer (≈ 22 MB raw for Kimi-K3-class [estimate]); no implicit expert-slice encoding. A load-time regression row watches manifest size and decode time instead (COMP-E10-T08).
7. ~~Chunk-table evidence for files with no origin~~ **Resolved:** RFC-0034 §10.1 item 4. The evidence is a keyholder import, or an `ApproveImport` request in which the keyholder streams the file and checks it (§6.3).

## Proposed amendments to RFC-0001

1. **§5.3, `ModelManifest::id`.** Replace "BLAKE3 over the ordered (tensor name, quant scheme, shape, content hash) list and the canonical hyperparameters. Identical tensors give an identical ModelId whatever the container format." with "BLAKE3 over the family tag and, per source file the census reads, (role, `FileDigest`, length). Computable from headers and file metadata alone, and stable across releases for the same files (RFC-0041 §1)." Reason: per-tensor content hashes need every byte before placement, which contradicts §7.1 step 2 and ADR-0016. Byte-level dedupe across containers happens at the chunk level (ADR-0411).
2. **§5.3, `TensorRecord::hash`.** Replace "over the tensor's bytes; chunk IDs derive from it" with "Merkle root over the tensor's `ChunkId`s, filled in once its chunks are committed; chunk IDs are BLAKE3 of chunk bytes (RFC-0041 §6)". The dependency runs from chunks to tensors, not the reverse.
3. **§5.3, additions (listed for readers):** `TensorRecord::aux` (scale and zero-point extents), `TensorGroup::shared_with` (tied embeddings), `LayerSpec::mtp`.
4. **§5.10, additions:** `ClusterCommand::ProposeChunkIds { file, range, body, evidence: ChunkEvidence }`, `ClusterCommand::ReportSuspectSource { node, model, chunk }` and `ClusterEvent::ChunkIdsCommitted { file, range, body, evidence: Option<ChunkEvidence> }`, where `None` marks a provisional batch (RFC-0041 §6.3, §8.4; RFC-0032 §6.5; RFC-0034 §10.1).
5. **§3.3 versus ADR-0015.** RFC-0001's lifecycle table has the runner build packs through the store API; ADR-0015's table says the daemon's store does it. This RFC follows RFC-0001: the runner has the providers needed for `repack`, and the daemon never loads kernel providers (RFC-0001 invariant 3). ADR-0015's table should be corrected.
6. **Glossary, "Pack", and ADR-0005 decision 3.** Both say packs are keyed by (model content hash, device class, plan epoch). RFC-0001 §5.6 keys them by shard digest, not epoch, so that an epoch change with an unchanged shard reuses the pack. The glossary should follow RFC-0001.
