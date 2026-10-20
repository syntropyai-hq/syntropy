# ADR-0411: `ModelId` commits to source content through file digests, so it is known from headers alone

- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0005, ADR-0016, ADR-0019; RFC-0001 §5.3, §7.1; RFC-0041 §1; RFC-0022, RFC-0034

## Context

- RFC-0001 §5.3 defines `ModelId` as BLAKE3 over the ordered (tensor name, quant scheme, shape, content hash) list and the hyperparameters, so identical tensors give one `ModelId` whatever the container.
- RFC-0001 §7.1 registers the model (`RegisterModel { model: ModelId }`) after reading headers only, before any placement, and ADR-0016 has each node fetch only its shard. A per-tensor hash of raw bytes cannot be computed under those rules without reading every byte of the model, which can be over 1 TB.
- The Hub publishes a SHA-256 per large file (its object ID) in revision metadata, before download. Local files can be hashed by the node that holds them.
- `ModelId` keys plans, packs, persisted sessions and prefix caches. It must change when any weight byte or hyperparameter changes.
- Chunks are content-addressed by BLAKE3, and chunk boundaries fall at tensor starts (ADR-0016), so identical tensor bytes in two containers already give identical chunks.

## Options considered

| Option | For | Against |
|---|---|---|
| A. RFC-0001's wording: hash of each tensor's raw bytes | Container-independent | Needs all bytes before placement; contradicts header-only census and shard-only fetch |
| B. Hash of headers and file names only | Instant | Does not commit to weight content; a re-uploaded file with changed weights and the same header would keep its identity |
| **C. Per tensor: (name, scheme, shape, SHA-256 of its file, offset, length), plus canonical hyperparameters** | Commits to every byte through the file digest; computable from Hub metadata or one local hashing pass; same value from any reader path | Same bytes in two different containers get different `ModelId`s |
| D. Two identities: a provisional one for placement, a content one later | Container independence eventually | Two keys for every cache and plan; re-keying sessions and packs when the content one arrives |

## Decision

**Option C.**

1. `FileDigest` is the SHA-256 of a whole source file: the Hub's object ID where published, otherwise computed by the store (local files, Git-stored small files, generic HTTPS origins that supply no digest are downloaded whole first).
2. `ModelId = BLAKE3("syntropy/model-id/v1" ‖ family tag ‖ per source file the census reads, sorted by (role, FileDigest): role ‖ FileDigest ‖ length)`. Roles are weights, weight index, `config.json`, `generation_config.json` and quantization config. Nothing Syntropy derives is hashed, so the same files keep their ID across releases; family tags are frozen (RFC-0043 §5.1). Golden IDs for recorded fixtures are checked in CI.
3. `TensorRecord::hash` becomes the Merkle root over the tensor's committed `ChunkId`s, filled in when known. It feeds pack `ExtentKey`s and integrity checks, not `ModelId`.
4. Byte-level dedupe across containers is left to chunk content addressing.
5. The tokenizer and chat template get a separate `TokenizerId`; they do not change KV contents for a given token sequence, so they are not part of `ModelId`.

## Consequences

- Positive: `RegisterModel` and placement happen right after header reads for Hub models; a local model needs one hashing pass, cached thereafter.
- Positive: any weight or hyperparameter change gives a new `ModelId`, which invalidates packs, sessions and prefix caches as it should.
- Negative: a GGUF and a safetensors file holding byte-identical tensors are two models. They still share chunks on disk and over the network.
- Negative: generic HTTPS origins without a published digest cost a full download before registration.
- Requires amending RFC-0001 §5.3 (RFC-0041's amendments 1 and 2).

## What would make us revisit

- Users routinely run the same weights from two containers and the duplicate packs or sessions cost them real disk or time.
- The Hub stops publishing per-file SHA-256 digests in revision metadata, or publishes only its chunked-storage hashes. Then `FileDigest` becomes that hash, under a new identity version string.

## Log

- Revised 2026-10-19: decision 2 changed from hashing canonical hyperparameters and per-tensor records to hashing source-file digests and a frozen family tag, because any release adding a field to `HyperParams` would silently have changed the ID of the same files.
