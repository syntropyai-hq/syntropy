# ADR-0507: GPU backends ship as signed, lockstep modules downloaded after hardware detection
- Status: Accepted (2026-10-20); revised 2026-10-20 (see Revisions)
- Date: 2026-10-19
- Deciders: maintainers
- Related: ADR-0001, ADR-0002, ADR-0014, ADR-0015, ADR-0018, ADR-0020; RFC-0052 §1, §3, §4; RFC-0040; RFC-0062

## Context
Syntropy supports CUDA 13 (primary), CUDA 12 legacy, ROCm, Vulkan, SYCL and Metal (ADR-0014, ADR-0020). Several facts constrain how those backends reach users.
- **Vendor runtimes are huge.** Surveyed desktop runners that bundle every one ship 1.0–1.5 GiB archives (`backends-and-portability.md` §5.1). The research's risk list names binary size and redistribution as a risk, and recommends downloading backend modules after detection, signed (§7.2 risk 3).
- **Licences differ by vendor** (ADR-0018):
  - CUDA's runtime may be redistributed only as Attachment A components, unmodified, inside an application;
  - ROCm is mostly permissive, with some pieces under custom terms or binary-only;
  - Vulkan, Metal and DirectX are system components.
- **Platform rules.**
  - macOS needs every nested binary signed and notarised, and Metal is a system framework anyway.
  - iOS forbids downloaded code.
  - Google Play forbids fetching native code from outside Play.
  - Distribution packages must not fetch executable code into system directories.
- **One binary** (ADR-0015). The daemon never links a GPU runtime (RFC-0001 §4.2 rule 4), and vendor runtimes are loaded dynamically (rule 7).
- **Driver detection does not need a GPU SDK.** NVIDIA's driver library and NVML, AMD's KFD topology in sysfs, DXGI adapters and Vulkan instances all report enough to choose (`backends-and-portability.md` §5.2).

## Options considered
| Option | For | Against |
|---|---|---|
| A. Everything in one binary per platform | one download; works offline | 1–2.5 GiB for every user; vendor licence terms attach to the core; slow updates |
| B. One binary per backend (`syntropy-cuda`, `syntropy-rocm`, …) | simple to build | users must pick correctly before installing; mixed-GPU hosts need two; violates one binary |
| C. Core binary plus backend modules chosen by a driver census and downloaded by `setup`, signed, lockstep-versioned | each machine downloads what it can use; vendor terms confined to modules; one signed core | a module ABI, a resolution step and a second artefact class to sign, host and test |
| D. Users install vendor SDKs; Syntropy links what it finds | smallest artefacts | breaks zero-config; version skew between user SDKs and our kernels |

## Decision
**Option C**, with these rules:

1. **The core binary contains no vendor GPU runtime.** It contains:
   - the CPU providers;
   - on macOS, the Metal providers;
   - on iOS and Android, every provider statically.
   macOS, iOS and Android therefore have no downloadable modules.
2. **Modules on Linux and Windows.**
   - `cuda13` (primary);
   - `cuda12-legacy` (Tier 3);
   - `rocm`;
   - `vulkan`;
   - `sycl` (Tier 3).
   Each holds a provider library (RFC-0040), the vendor redistributables it needs, unmodified (including rocBLAS and hipBLAS in `rocm`, which the ggml HIP provider links), a manifest with file hashes, requirements and a composite licence, and the licence texts.
   - **`cuda13` ships only Syntropy code** (revised 2026-10-20; see Revisions). Its manifest pins, by SHA-256, every NVIDIA redistributable archive the provider needs (the CUDA runtime, NVRTC, nvJitLink, cuBLAS and cuBLASLt, which the ggml CUDA provider links), as NVIDIA's own redistributable manifest lists them. `setup` shows the CUDA EULA once, fetches those archives from NVIDIA's redistributable server, checks each against the pinned hash, and extracts the files unmodified beside the provider library. Syntropy redistributes no NVIDIA file in the script, MSI and app channels.
   - **Channels that would bundle** NVIDIA files, because they install under a package manager, in a container or with no network: the deb and rpm module packages (M2), the `cuda13` OCI image (M3) and the offline bundles (M4). Each ships NVIDIA files only where NVIDIA's redistribution terms allow it for that channel. Otherwise it uses the fetch path: the deb and rpm packages through NVIDIA's own repository packages, the image at first start into its state volume, and an offline bundle once the host can reach NVIDIA, with rule 7's fallback until then.
   - **`vulkan` arrives in two steps**. In M2 it ships ggml's Vulkan provider only, labelled a Tier 2 preview, with ggml-owned allocations and no elastic arenas. RFC-0040's native Vulkan provider joins it in M3.
3. **Lockstep.** A module's version and build ID equal the core's. Modules install inside the core's version directory and move with it on update and rollback. A mismatched module is refused at load.
4. **Selection.** `syntropy setup` runs a driver census in a probe-mode process (no GPU context in the daemon) and resolves modules by vendor, architecture, driver version and OS. It shows the result with sizes and support tiers before downloading. The daemon never downloads modules on its own. It reports a GPU without a matching module and names the command.
5. **Verification.** Modules are TUF targets, signed by RFC-0062's pipeline with RFC-0034's keys (ADR-0508). Their hashes are verified at download and at update. The loader re-checks the hash and the build ID immediately before `dlopen` (RFC-0034 §12.2). Only runner and probe processes load them, by absolute path, with search paths that cannot be hijacked.
6. **Distribution packages** ship modules as separate packages. `setup` prints the package command instead of downloading.
7. **Fallback.** Where the fetch route is unavailable (NVIDIA's server unreachable, or an offline host whose channel may not bundle NVIDIA files), `cuda13` loads the user's installed CUDA toolkit. That is the last fallback, and `setup` names the step. NVIDIA files are never modified, re-signed or patched: Linux files stay byte-identical, the `$ORIGIN` RPATH sits on Syntropy's provider library only, Authenticode signs only Syntropy-built files, and the driver library (`libcuda.so`, `nvcuda.dll`) always comes from the installed driver.

## Consequences
- **Positive.**
  - A CPU-only laptop downloads only the core (≤ 40 MiB target).
  - An NVIDIA desktop downloads the core plus one module.
  - Vendor terms never touch the core.
  - Mixed-GPU hosts work.
- **Negative.**
  - A provider ABI and its versioning (RFC-0040).
  - The CUDA path depends on NVIDIA's redistributable server staying reachable and ungated at install and update time.
  - A module class in the release pipeline, the TUF targets and the install matrix.
  - Air-gapped installs need offline bundles.
  - A GPU added later needs `syntropy setup` again, which the daemon prompts for.
- **Work created.**
  - RFC-0052: format, resolution, install and load rules.
  - RFC-0040: build and in-process ABI.
  - RFC-0062: module builds, signing and size gates.
  - RFC-0053: doctor INS-001 and DRV checks.

## What would make us revisit
- NVIDIA's redistribution terms allow module-only bundling in every channel. Then bundle everywhere, which also works offline.
- NVIDIA's redistributable server changes its terms or gates the archives behind a login. Then fall back to the user's toolkit, and bundle only in channels the terms allow.
- Module sizes after M1 stay within a few tens of MiB, because the providers need no large vendor libraries. Then fold the common modules into the core and drop the resolution step for them.
- The provider ABI churns so much that lockstep is not enough to keep modules loadable. Then move module loading to an out-of-process provider host.

## Revisions
- Revised 2026-10-20: rule 2: `cuda13` ships only Syntropy code, and `setup` fetches NVIDIA's redistributable archives from NVIDIA, checked against hashes pinned in the signed module manifest, after showing the CUDA EULA once. Syntropy then redistributes no NVIDIA file in the script, MSI and app channels, and SC-24's one-command install holds. Channels that would bundle NVIDIA files do so only where the redistribution terms allow, and otherwise use the fetch path. Rule 7 becomes the fallback (the user's installed toolkit) and the no-modification rules for NVIDIA files.
- Revised 2026-10-20: rule 2: the `vulkan` module ships in M2 with ggml's Vulkan provider only, as a Tier 2 preview with ggml-owned allocations, so AMD and Intel GPUs have a GPU path before RFC-0040's native Vulkan provider lands in M3.
