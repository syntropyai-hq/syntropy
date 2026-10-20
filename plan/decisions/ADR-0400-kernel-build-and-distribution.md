# ADR-0400: Kernels are compiled ahead of time through a content-addressed cache; GPU providers ship as C-ABI provider libraries in backend modules

- Status: Accepted (2026-10-20); revised 2026-10-20 (decision 3)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the compute and models area)
- Related: ADR-0001, ADR-0002, ADR-0014, ADR-0015, ADR-0020, ADR-0507, ADR-0602, ADR-0623; RFC-0040 §1, §10; RFC-0052 §4, RFC-0060 §6.4, RFC-0062 §4–§5

## Context

- ADR-0001 fixes the kernel languages: CUDA C++ with CUTLASS/CuTe-class templates, HIP, MSL, Slang → SPIR-V, and Rust `std::arch` for CPUs. It wants a single core binary, with GPU vendor runtimes and large kernel payloads loaded at runtime.
- ADR-0507 (product area): the core binary contains no vendor GPU runtime, only the CPU providers, plus the Metal providers on macOS and every provider statically on iOS and Android. Linux and Windows get backend modules (`cuda13`, `cuda12-legacy`, `rocm`, `vulkan`, `sycl`), each holding a provider library, unmodified vendor redistributables and a manifest. Module version and build ID equal the core's (lockstep). Only runner and probe processes load modules, by absolute path. RFC-0052 leaves the in-process ABI to RFC-0040.
- ADR-0623 (quality area): kernels are split into many small translation units with generated instantiation lists, compiled per target through a content-addressed object cache that only trusted builds write. The object's BLAKE3 is the `OpCert::kernel` hash that op selection checks (ADR-0602).
- CUDA 13 targets sm_75 and newer. Block-scaled FP4 MMA needs the arch- or family-specific target (`sm_120a`/`sm_121a` or `sm_120f`), because `ptxas` rejects those instructions for plain `sm_120`. That PTX is not forward compatible.
- Rust has no stable ABI, so a dynamically loaded Rust provider needs a C boundary. Panics and allocator ownership must not cross it.
- macOS needs every nested binary notarised; iOS and Android forbid loading downloaded code.

## Options considered

| Option | For | Against |
|---|---|---|
| A. Everything in one binary | One file | Ships every vendor's payload to every machine; contradicts ADR-0507 |
| B. Runtime compilation (NVRTC, runtime MSL or Slang) | Small download | Compilers on user machines; slow first start; notarisation and iOS problems; numerics depend on the user's compiler |
| C. Rust provider code in the core; only kernel objects in modules | No ABI for provider logic | Puts CUDA, HIP and Vulkan host code in a core that ADR-0507 keeps free of them; splits one provider across two artefacts |
| D. Rust provider libraries passing Rust trait objects across `cdylib` | Little glue | Relies on unstable layout and on matching allocators and panic runtimes |
| **E. AOT kernels through ADR-0623's cache; CPU and Metal providers (and all providers on mobile) static in the core or app; CUDA, HIP, Vulkan and SYCL providers, each with its ggml backend, as provider libraries in their modules, behind a generated C ABI with one versioned entry symbol, a lockstep build-ID check and handle-only crossings** | Matches ADR-0507 and ADR-0623; the core stays small and vendor-free; one provider is one artefact; failures at the boundary are contained | A generated FFI layer to maintain; each module carries its own static ggml copy |

## Decision

**Option E.**

1. **Compilation.** Many small translation units per (op family, quant-scheme group); explicit generated instantiation lists; per-target objects compiled through `cargo xtask kcache` with ADR-0623's `SourceKey`/`ObjectKey`; per-architecture objects linked into fatbins, code-object bundles, metallibs or SPIR-V sets at the end. No fast-math. The only runtime compilation is the CUDA driver JIT-compiling the shipped `compute_120` PTX on GPUs newer than the build.
2. **Kernel hashes.** GPU ops report the BLAKE3 of the object that contains them; CPU ops hash their source plus toolchain and ISA level; ggml ops hash the ggml commit, patch set, toolchain, backend and flags. These are the `OpCert` keys.
3. **Where providers live.** The core holds the CPU provider and the ggml CPU provider; the macOS core also holds the Metal and ggml Metal providers with both metallibs embedded. `libsyntropy_provider_cuda` (modules `cuda13` and `cuda12-legacy`), `libsyntropy_provider_rocm`, `libsyntropy_provider_vulkan` and `libsyntropy_provider_sycl` each hold the native provider (where one exists), its ggml provider with a private static ggml copy, and its device code. iOS and Android apps link their providers statically.
   - *Revised 2026-10-20:* the CUDA modules carry Syntropy code only, so the release redistributes none of NVIDIA's files. NVIDIA's libraries, the CUDA runtime included, are not linked statically or shipped: the signed `module.toml` pins each NVIDIA redistributable archive's SHA-256, `setup` fetches the archives from NVIDIA after showing NVIDIA's licence terms once, checks them and extracts the files unmodified beside the provider library, which finds them through its own run path. The installed toolkit is the last fallback (ADR-0507 rule 7). Channels that would have to carry the libraries themselves are not built until NVIDIA's terms are confirmed to permit it per channel (RFC-0040 §10.3).
   - *Revised 2026-10-20:* at M2, `libsyntropy_provider_vulkan` ships with `ggml-vulkan` only, as a Tier 2 preview; the native Vulkan provider joins it at M3 (RFC-0040 §7.5), so AMD, Intel and integrated GPUs have a GPU path before the native provider lands.
4. **ABI.** A provider library exports exactly one symbol, `syntropy_provider_entry_v<N>`, under ELF version node `SYNTROPY_PROVIDER_<N>` on Linux, with every other symbol hidden and `-Bsymbolic` linking. It returns a `SynModule` table (magic, ABI major and minor, build ID, struct sizes, one vtable per provider). The core wraps each vtable into a `KernelProvider`. Only `#[repr(C)]` structs, pointer–length pairs and `u64` handles cross; every thunk catches panics; each side frees its own allocations; minor versions only append.
5. **Load check.** Path from `module.toml`; entry symbol for the core's major version; magic and struct sizes; minor version; build ID equal to the core's (relaxed only by `--dev-modules`, which marks results uncertified); provider IDs listed in `module.toml`. Any failure refuses the module with a message naming the failed check.
6. **Census and loader crate.** The core side of the ABI (module loading, the load check and the `ModuleProvider` proxy) and the census live in `syntropy-kernels-loader`, a layer-2 crate beside the provider crates that RFC-0001 §4.2 rules 6 and 8 must name (RFC-0040 amendment 6). The census identifies GPUs and drivers in a probe-mode process without loading modules and without creating a GPU context, loading vendor driver libraries at runtime for read-only queries and reading amdgpu/KFD sysfs on Linux instead of initialising the ROCm runtime. `syntropy setup` resolves modules from it.
7. **Contraction.** Bit-exact-class translation units are compiled with floating-point contraction off; the flag is part of their `ObjectKey` (RFC-0040 §10.2 rule 7).

## Consequences

- Positive: the core stays small and vendor-free; GPU users download only their vendor's provider library and libraries.
- Positive: a provider's host code and device code version together, and a mismatched module can never load in a release.
- Positive: certificates follow kernel bytes, so a build that leaves kernels unchanged keeps its certificates.
- Negative: a generated C ABI over the whole `KernelProvider` trait to maintain, and test for panics and ownership.
- Negative: two to three static ggml copies per machine (core plus each installed module).
- Negative: the main channels depend on NVIDIA's redistribution server staying reachable and ungated; offline installs need the toolkit fallback until a bundling channel is confirmed permitted (decision 3, revised).

## What would make us revisit

- Rust gains a stable ABI suitable for trait objects across dynamic libraries. The generated C layer could shrink.
- The FFI layer shows up in profiles above 1 % of `t_fixed` on any Tier 1 backend.
- A permissively licensed ahead-of-time kernel DSL reaches hand-written throughput on two Tier 1 backends (ADR-0001's revisit clause).
- NVIDIA gates its redistributable archives behind a login or changes their terms: `cuda13` then loads the user's installed toolkit libraries (ADR-0507 rule 7).
- NVIDIA's terms are confirmed to permit bundling inside the lockstep module: the module may then carry the libraries, which also serves offline installs.

## Log

- Revised 2026-10-19: decision 6 moves the loader and census out of `syntropy-kernels` into `syntropy-kernels-loader`, because inside `syntropy-kernels` they broke RFC-0001 §4.2 rules 6 and 8; decision 7 added so that bit-exact-class units are always compiled with contraction off.
- Revised 2026-10-20: decision 3. The CUDA modules carry no NVIDIA code; NVIDIA's archives are fetched from NVIDIA at `setup` against hashes pinned in the signed manifest, and the CUDA runtime is linked dynamically. The `vulkan` module ships at M2 as a ggml-only Tier 2 preview.
