# ADR-0623: Content-addressed kernel object cache with a trust split, and reproducible core builds
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted in the quality and release engineering area, with the compute area of RFC-0040)
- Related: ADR-0001, ADR-0002, ADR-0014, ADR-0018; RFC-0040; RFC-0060 §6.4; RFC-0062 §4–§6

## Context
Kernels are written in CUDA C++ with tensor-core templates, HIP, Metal Shading Language and Slang (ADR-0001). Each is compiled for many targets:
- about eight CUDA architectures plus architecture-specific variants;
- every ROCm family on AMD's list;
- two Metal library variants;
- SPIR-V.

ADR-0001 records that kernel builds dominate CI cost. Most PRs do not touch kernels, but naive CI would recompile them anyway, or rely on a general compiler cache whose device-side handling the project does not control. A shared cache that untrusted PRs could write to would also be a supply-chain hole, since a poisoned object would flow into release builds.

Separately, provenance and user trust benefit from reproducible builds. RFC-0060's certificates key on kernel object hashes, so the object bytes must be stable and well defined.

## Options considered
1. **No kernel cache; compile every time.** Simple, and it cannot be poisoned, but it pays the most expensive step on every build.
2. **A general-purpose compiler cache for everything**, device code included. One tool, but its device-compiler support covers only part of the pipeline (reports say the host side of `nvcc` compilations is what gets cached), and its trust model does not distinguish forks from `main`.
3. **Commit compiled kernels to the repository.** Fast, but binary blobs in git, opaque review, and a supply-chain risk of their own.
4. **Chosen:**
   - a project-owned, content-addressed kernel object cache: one object per (translation unit, target);
   - keys over source, every included file (from the compiler's dependency output), the toolchain banner, normalised flags and the generated instantiation lists;
   - writes only from trusted contexts, reads by everyone;
   - a general compiler cache for host code only;
   - reproducible core builds verified nightly.

## Decision
1. **Keys.**
   - `SourceKey = BLAKE3(version tag ‖ toolchain_id ‖ target ‖ normalised flags ‖ source)`.
   - A per-`SourceKey` manifest of include hashes.
   - `ObjectKey = BLAKE3(SourceKey ‖ include hashes ‖ instantiation list)`.

   `OpCert::kernel` is RFC-0040 §10.2's `kernel_hash`: for compiled kernels, the object's BLAKE3; for the Rust CPU provider, its source-plus-toolchain-plus-ISA hash (RFC-0060 §6.4). `OpCert::source_key` is the translation unit's `SourceKey`.
2. **Structure** (a requirement on RFC-0040).
   - Many small translation units, one per op family and quant-scheme group.
   - Explicit, generated instantiation lists.
   - Per-architecture objects linked into fat binaries at the end.
   - Builds call the cache through `cargo xtask kcache`.
3. **Trust.**
   - Objects live in the project bucket under `kcache/`.
   - Only `main` merge-group builds, nightly and release builds write there. Same-repository PR builds write to an isolated `kcache-pr/<branch>/` that no other build reads.
   - Everyone reads `kcache/`, and fork PRs compile what they change.
   - Nightly rebuilds a random 5% of objects from scratch and compares bytes, purging and failing on a mismatch.
4. **Architecture sets.** PR builds compile only the lab rigs' architectures. Nightly and release builds compile the full lists (RFC-0062 §4.3).
5. **Host code** uses `sccache` against the same bucket with the same trust split.
6. **Reproducibility.**
   - The Linux core binary is bit-reproducible from M2: pinned image by digest, pinned toolchain, `--locked`, vendored sources verified, path remapping, `SOURCE_DATE_EPOCH` from the commit, no build timestamps or hostnames.
   - It is verified by a nightly double build on separate runners, with a structured binary diff on mismatch.
   - Unsigned macOS and Windows payloads are reproducible from M3.
   - Backend modules are reproducible from M4, or listed in `ci/repro-exceptions.toml` with the reason.
7. **Provenance.** Build-provenance attestations and an SBOM for every nightly and release artefact (RFC-0062 §6.3).

## Consequences
- **Positive.**
  - Kernel compilation is paid once per kernel change per architecture. PRs that do not touch kernels compile only Rust.
  - A fork PR cannot place an object in any other build's path.
  - Certificates and builds share one definition of "this kernel".
  - Users and auditors can rebuild the core binary and get the same bytes.
- **Negative.**
  - The project runs and pays for a bucket, and maintains the cache tooling.
  - Kernel crates must follow the translation-unit and instantiation rules, which constrains how RFC-0040's authors organise code.
  - Device compilers that are not deterministic make some modules unreproducible until a workaround exists. They are listed, not hidden.
  - The 5% verification costs some compile time every night.

## What would make us revisit
- Kernel-cache hit rates on builds whose diffs touch no kernel source stay below 95% for a month. Then the key is too sensitive (flag or path noise) or the translation-unit structure too coarse.
- A device compiler gains a reliable built-in cache with a trust model that fits. Then replace the custom layer for that backend.
- The bucket grows beyond what its lifecycle rules keep bounded. Then prune objects unused for 30 rather than 90 days, and keep only the full architecture lists of the last two releases.
