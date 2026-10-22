# RFC-0052: Packaging, install and updates
- Status: Draft
- Area: Product surface and operations
- Depends on: RFC-0001 (§3, §4.2, §9.2), RFC-0040 (module build and in-process ABI), RFC-0051, RFC-0053, RFC-0054, RFC-0034 (signing trust), RFC-0062 (release pipeline), ADR-0001, ADR-0014, ADR-0015, ADR-0018, ADR-0020, ADR-0507, ADR-0508, ADR-0509, ADR-0510
- Date: 2026-10-19

## Summary

Syntropy installs with one command per OS and runs its first model with no configuration (SC-24). This RFC defines what ships, how it gets onto a machine, how it stays current, and how it leaves without a trace.

- **One core binary per platform**, containing the daemon, runner, CLI, the CPU kernels for every ISA level, the ggml CPU provider, the solver, the transport stack, the dashboard and the model catalogue. On macOS the core also holds the Metal providers, so nothing else is downloaded there.
- **Backend modules downloaded after detection** on Linux and Windows (ADR-0507): `cuda13` (primary), `cuda12-legacy` (Tier 3), `rocm`, `vulkan` and `sycl` (Tier 3). Modules are signed, versioned in lockstep with the core, and selected by `syntropy setup` from a driver census that needs no GPU SDK. A machine downloads only what its GPUs can use. `cuda13` carries only Syntropy code: `setup` fetches NVIDIA's own redistributable archives from NVIDIA, checked against hashes the signed module manifest pins (§4.4). `vulkan` ships first, in M2, as a ggml-only Tier 2 preview, so AMD and Intel GPUs have a GPU path before M3.
- **Installers**:
  - Linux: `curl … | sh`, plus `.deb` and `.rpm` repositories;
  - macOS: a notarised app bundle carrying the CLI (direct download, Homebrew cask, or `.pkg` for administrators);
  - Windows: winget, an MSI and a PowerShell script, all Authenticode-signed;
  - OCI images;
  - Tier 3 mobile libraries.
- **Services by default** (ADR-0509). A user-level service on personal installs: systemd user unit, LaunchAgent, logon task. A system service for distribution packages and per-machine installs. Each drains on stop.
- **Updates** (ADR-0508). The update client is a TUF client over metadata that RFC-0062's pipeline signs with the keys RFC-0034 defines. First-run checks use detached release-key signatures. Channels are stable, beta and nightly. Downloads are staged into side-by-side version directories and checked by a self-test. The `current` pointer is switched atomically, and a health window triggers automatic rollback. Runners are re-adopted across the restart, so loaded models stay loaded. Package-manager installs leave updates to the package manager.
- **Uninstall reverts every host change.** Each change the installer, `setup` or `doctor --fix` makes is written ahead to a host-change ledger: units, PATH edits, limits files, firewall rules, linger, tasks. `syntropy uninstall` replays the ledger in reverse. A generated shell or PowerShell script can do the same if the binary is already gone.

## Motivation and goals / non-goals

The research's packaging findings set the problem (`backends-and-portability.md` §5, `multi-device-inference.md` §12.1):
- **Size.** Surveyed desktop runners ship 1.0–1.5 GiB archives because they bundle every GPU runtime for every user.
- **Fragility.** Source builds with long prerequisite lists break on toolchain changes.
- **Installer side effects.** A menu-bar installer changed host network settings, and the survey could not confirm that its uninstall reverts them (§15.12).
- **Platform rules** that constrain the design:
  - macOS requires notarisation of every nested binary, and since macOS 15 unnotarised software cannot be opened by Control-click;
  - iOS forbids downloaded code;
  - Google Play forbids apps downloading native code from outside Play;
- **The recommendation.** One small binary per OS, backend modules downloaded after detection, signed, and updated separately from the app with a signed manifest (§5.2, §5.4, §7.2 risk 3).

### Goals

- G1. One command or installer per OS, then one question and one model choice, then tokens (SC-24). No compiler, interpreter, solver or benchmark tool to install.
- G2. Download only what this machine can use: the core, plus the modules its GPUs need.
- G3. Every artefact is signed. The updater trusts nothing on the network beyond the signed metadata, including mirrors.
- G4. Updates never leave a node without a working version, and roll back on their own when the new version is unhealthy. In the default `idle` window no request is in flight, so none is failed. An update applied outside that window gives local-ingress requests `update.local_drain` to finish and ends the rest with a retryable error, because their connections belong to the daemon process. Remote-ingress sequences keep generating where re-adoption is available (§6.5, §7.3).
- G5. Uninstall leaves the host as it was, except for what the user asks to keep (models, configuration, identity).
- G6. The privacy posture of ADR-0510 holds: update checks carry no identifiers.

### Non-goals

- Building from source as an install path for users (it remains a developer path, RFC-0062).
- Kubernetes operators or Helm charts (RFC-0001 §1.2).
- Shipping GPU drivers. Drivers come from the OS or the vendor. Syntropy ships only redistributable user-space runtimes inside modules (ADR-0018).
- An app-store desktop distribution (Mac App Store, Microsoft Store). Their sandboxes forbid memory locking and long-running background services.

## Design

### 1. Artefacts

| Artefact | Targets | Contents | Support tier |
|---|---|---|---|
| core binary `syntropy` | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` (glibc ≥ 2.35, ADR-0020), `x86_64-pc-windows-msvc`, `aarch64-apple-darwin` | daemon, runner and CLI modes; native CPU provider with per-ISA variants chosen at runtime (x86-64-v2 to AVX-512/AMX; NEON to SME2); ggml CPU provider (static); HiGHS (static); iroh; dashboard bundle; catalogue; on macOS also the native Metal and ggml Metal providers ; on Windows also `syntropyw.exe`, a GUI-subsystem launcher of a few hundred KiB that starts the daemon without a console and acts as the update guard (§6.3) | per platform, ADR-0020 |
| module `cuda13` | Linux x86_64, Linux aarch64 (GB10-class), Windows x86_64 | native CUDA provider (cubins for sm_75 and newer, PTX for the newest), ggml CUDA provider, and a manifest pinning the NVIDIA redistributable archives they need, which `setup` fetches from NVIDIA (§4.4); no NVIDIA file is in the archive | 1 (2 on aarch64) |
| module `cuda12-legacy` | Linux and Windows x86_64 | ggml CUDA provider and a reduced native set for sm_60/61/70, CUDA 12 redistributables | 3 |
| module `rocm` | Linux x86_64 | native HIP provider, ggml HIP provider, ROCm user-space libraries that may be redistributed (all-architecture variant; binary-only profiler pieces excluded; hipCUB and RCCL after licence review, ADR-0018) | 2 |
| module `vulkan` | Linux x86_64 and aarch64, Windows x86_64 | M2: the ggml Vulkan provider only, labelled a Tier 2 preview, with ggml-owned allocations and no elastic arenas; M3: adds the native Vulkan provider (SPIR-V from Slang). No vendor libraries (the loader is a system component) | 2 (preview in M2) |
| module `sycl` | Linux and Windows x86_64 | ggml SYCL provider | 3 |
| `install.sh`, `install.ps1` | Linux and macOS; Windows | bootstrap scripts (§2) | — |
| `Syntropy.app` (zip, and inside `.pkg`) | macOS arm64 | the core binary as the app executable, plus Info.plist for Local Network privacy and Bonjour (§5.2) | 1 |
| `.deb`, `.rpm` and repositories | Linux x86_64 and aarch64 | `syntropy` plus one package per module (`syntropy-module-cuda13` …) | per platform |
| MSI and winget manifest | Windows x86_64 | per-user and per-machine scopes (§2.3) | 1 |
| Homebrew cask | macOS | wraps `Syntropy.app` and links the CLI | 1 |
| OCI images | `linux/amd64`, `linux/arm64` | `cpu`, `cuda13`, `rocm`, `vulkan` flavours (§8) | per flavour |
| offline bundles | every desktop target | core plus every module for that target in one archive, for air-gapped installs. NVIDIA files are included only where NVIDIA's redistribution terms allow it for offline bundles (§4.4); otherwise the bundle's `cuda13` fetches them once the host can reach NVIDIA, or uses the installed CUDA toolkit | — |
| Android AAR and reference APK; iOS XCFramework and reference app | Tier 3 | §10 | 3 |
| per release: SBOM (SPDX), provenance attestations, `THIRD-PARTY-NOTICES`, checksums | all | ADR-0018 decision 6; RFC-0062 | — |

**Versioning.** The core and its modules share one version and one build ID (lockstep). A module from another build is refused at load (§4.3). SemVer and release automation belong to RFC-0062.

### 2. Install flows

Every flow ends the same way: the binary is on the PATH, `syntropy setup` has run (§3), a service is running, and the last line printed is the dashboard URL and `syntropy run`.

#### 2.1 Linux and macOS: `curl -fsSL https://<release-host>/install.sh | sh`

```text
install.sh [--system] [--no-service] [--channel stable|beta|nightly] [--version V] [--offline-bundle FILE] [--yes]
  1. detect OS and architecture; on Linux check glibc ≥ 2.35 and say so plainly if not
  2. download over HTTPS from the release host (or read the offline bundle):
       Linux: syntropy-<ver>-<target>.tar.zst      macOS: Syntropy-<ver>.app.zip
     and check its SHA-256 against the checksum file served beside it
  3. install side by side:
       Linux user:   ~/.local/share/syntropy/versions/<ver>/syntropy, `current` → <ver>, ~/.local/bin/syntropy → current
       Linux system: /opt/syntropy/versions/<ver>/…, /usr/local/bin/syntropy → current   (sudo, asked first)
       macOS:        ~/Applications/Syntropy.app (no admin), ~/.local/bin/syntropy → the app's executable
  4. if ~/.local/bin is not on PATH: show the line to add, offer to add it to the shell's rc file
     (a ledger entry, §11), never edit silently
  5. exec `syntropy setup` (§3) with the flags passed through
```

The script is short, readable and POSIX `sh`. It trusts TLS and the checksum file. `syntropy setup` then verifies its own artefact's detached release-key signature against the release keys embedded in the binary (§7.2, RFC-0034 §12.2). That catches corrupted downloads and a mirror serving wrong bytes. It **cannot** catch a compromised release host, because a binary that checks itself against keys it carries proves nothing about who built it. Later updates go through the TUF client.

**Trust root per install path**, stated plainly:

| Path | What authenticates the first download |
|---|---|
| `curl … \| sh`, `irm … \| iex` | TLS to the release host only. Users who want more pass `--verify-key <fingerprint>` (or `-VerifyKey`), and the script then checks the artefact's detached signature against that fingerprint. The argument must be the full 256-bit fingerprint in hex. The six-word form carries 66 bits, enough for a node identity compared once, but not for a long-lived key that signs every install, so the script rejects it here and shows it only for reading aloud. The fingerprint is published separately from the release host, in both forms (RFC-0034 §6.5): in the documentation, the project's source repository, every release's notes, the package-manager channels, and an entry in the public transparency log (RFC-0034 §12.2 rule 2) |
| macOS app zip from the script | TLS, then `setup` runs `codesign --verify --deep --strict` and `spctl --assess` on the bundle. `curl` downloads carry no quarantine attribute, so Gatekeeper would not have assessed it otherwise |
| Homebrew cask, `.pkg`, winget, MSI | the platform's signature checks (Developer ID and notarisation, Authenticode) plus the package manager's own trust |
| `.deb`, `.rpm` | the distribution repository's signing key, installed once by the user |
| offline bundle | the bundle's TUF metadata and detached signatures, checked against keys embedded in the binary: integrity only, unless `--verify-key` is used |

On macOS the script installs the app bundle, not a bare binary. A launchd agent that reaches the local network triggers the Local Network privacy prompt, and the prompt needs an app bundle with a usage description to be attributed and answered properly (Apple TN3179; §5.2).

#### 2.2 macOS: Homebrew and `.pkg`

- `brew install --cask syntropy` (in a project tap until accepted upstream) installs `Syntropy.app` and links `syntropy` into Homebrew's `bin`. The cask's `postflight` runs `syntropy setup --package=brew`. Its `uninstall` stanza runs `syntropy uninstall --package-hook`, and its `zap` stanza lists the state and store directories. Updates go through `brew upgrade` (`update.mode = "package-manager"`).
- `Syntropy-<ver>.pkg` (Developer ID Installer, notarised, stapled) is for administrators and MDM. It installs `/Applications/Syntropy.app` and `/usr/local/bin/syntropy`, and optionally the system service (§6).

#### 2.3 Windows: winget, MSI, PowerShell

- `winget install Syntropy.Syntropy` installs the MSI in per-user scope by default (`MSIINSTALLPERUSER=1`, no admin): `%LOCALAPPDATA%\Programs\Syntropy\versions\<ver>\syntropy.exe` with a `current` junction, the user's PATH entry, a Start-menu entry "Syntropy dashboard", then `syntropy setup --package=msi`. `--scope machine` installs under `%ProgramFiles%\Syntropy` with the Windows service (§6).
- `irm https://<release-host>/install.ps1 | iex` does the same as the per-user MSI from a signed zip, for machines without winget. The script itself is Authenticode-signed, so execution policies that require signed scripts accept it when saved and run.
- Updates: winget-installed copies report `update.mode = "package-manager"` and point to `winget upgrade`; the script install self-updates (§7).

#### 2.4 Linux packages

- apt and dnf repositories, signed with the project's repository key (`signed-by` keyring for apt, `gpgkey` for dnf). Packages: `syntropy` (core, system service, `syntropy` system user in groups `video` and `render`) and one package per module.
- `postinst` runs `syntropy setup --package=deb|rpm` non-interactively (§3). Setup does not download modules into system directories. It records the needed module packages as a pending item. Distribution packages must not fetch executable code at runtime, and the package manager stays the owner of `/usr`.
  - **Installing the module package** (open question 7, answered: OQ-PKG-11). At the first interactive `syntropy setup` or `syntropy run`, never in `postinst`, Syntropy asks once for consent and installs the module package through PackageKit when it is present. Without PackageKit it prints the exact command (`sudo apt install syntropy-module-cuda13`). A deb or rpm install on a GPU host is then one command plus one consent, and SC-24's one-command count holds where PackageKit exists; SC-24 records the exception for hosts without it.
  - **The CUDA module package** carries NVIDIA files only where NVIDIA's redistribution terms allow it for distribution packages (§4.4). Otherwise it takes them from NVIDIA through NVIDIA's own repository packages, as package dependencies, so the package manager still owns every file under `/usr`.
- `prerm` runs `syntropy uninstall --package-hook`, which stops the service and reverts ledger entries the package manager does not own. `purge` removes `/var/lib/syntropy`.

#### 2.5 Offline bundles

`install.sh --offline-bundle syntropy-<ver>-<target>-full.tar.zst` (or the `.zip` with `install.ps1 -OfflineBundle`) installs the core and all modules for the target with no network. The fetch route of §4.4 cannot run offline, so a bundle carries NVIDIA files only where NVIDIA's redistribution terms allow it for offline bundles. Otherwise its `cuda13` module fetches them the first time the host can reach NVIDIA, or uses a CUDA toolkit already installed on the host, and `setup` says which. The bundle carries a copy of the signed TUF metadata for that release and the detached signatures, so `setup` verifies it the same way, offline. `offline = true` (RFC-0041) and `update.mode = "off"` are proposed for the node's config, and the user confirms them.

### 3. `syntropy setup`: detection and first run

```text
syntropy setup [--package=…] [--system] [--no-service] [--modules auto|none|LIST] [--yes]
  1. verify self: the running artefact's detached release-key signature (§7.2), or the signed
     TUF metadata copy inside an offline bundle; failure → delete the version directory, explain, exit 8
  2. driver census, in a probe-mode child process (no GPU context in the daemon, RFC-0001 §2.3):
       NVIDIA  dlopen the driver library and NVML: devices, compute capability, driver version
               (no CUDA context is created; driver queries only)
       AMD     Linux: KFD topology in sysfs (gfx target per node), /dev/kfd access; Windows: DXGI adapters
       Intel   sysfs or DXGI adapters
       Vulkan  create an instance, enumerate physical devices: API version, extensions, vendor
       Apple   Metal device family (macOS: nothing to download)
       WSL2    the paravirtual GPU device and the WSL driver library path
  3. resolve modules (§4.2); show the list with sizes and support tiers, and say plainly when a GPU is
     served only by the `vulkan` preview (M2) or is not served yet; confirm on a TTY
  4. fetch, verify and install modules (§4.3); for `cuda13`, show the CUDA EULA once, then fetch NVIDIA's
     redistributable archives from NVIDIA and check each against the hash pinned in module.toml (§4.4);
     package installs record a pending item instead (§2.4)
  5. host preparation, each step asked and recorded in the ledger (§11):
       Linux: memlock limit for the user (limits.d file or user-manager drop-in) if below the planned pin
              budget. It takes effect only for user managers started afterwards, so setup says a re-login
              or reboot is needed, and until then the daemon runs within the old limit
       Linux, AMD or Intel GPU: offer `usermod -aG render,video <user>` when the user is not a member;
              device access is judged by group membership as the boot-time service will see it, not by
              the interactive session's seat ACLs. Like memlock, a new group reaches a user unit only
              through a user manager started afterwards: setup says a re-login or reboot is needed, and
              until then the daemon runs on Vulkan or CPU and says why. System units get
              their groups from `SupplementaryGroups=` at each start and need no re-login
       Linux, AMD or Intel GPU without a Vulkan loader or driver: print the distribution's package
              names (doctor DRV-003 repeats them); CPU-only operation until installed
       Linux user unit: lingering, by running `loginctl enable-linger` unprivileged first and asking for
              sudo only if it fails with "Access denied" (open question 5, answered: OQ-PKG-09)
       Windows per-machine: firewall rule for bin\syntropy.exe on Private networks (stable path, §6.3)
       service install per §6 unless --no-service
  6. start the daemon (through the service), wait for /readyz, run the startup probe (≤ 15 s, SC-22)
  7. print: support tier and backends, listeners, the dashboard URL, any doctor warnings, and
           "next: syntropy run" — which opens the model picker (RFC-0051 §3.3)
```

**Non-interactive mode** (`--package=deb|rpm|msi|brew`, or `--yes`).
- Package hooks have no terminal. `setup --package=…` therefore asks nothing. It verifies, installs and starts the service, and runs the probe.
- Every decision that needs consent is recorded as a **pending item** in the state directory: modules to download or install, memlock, lingering, group membership, firewall. The first interactive `syntropy run`, `syntropy setup`, or a dashboard banner presents them.
- `--yes` accepts the recommended answers and is never used by package hooks, so no package maintainer script (`postinst` and its equivalents) silently downloads modules or edits limits.

Setup is idempotent. Running it again recomputes modules (after a GPU or driver change), repairs missing ledger items, and changes nothing that is already right. The daemon also re-runs the census at start. When it finds a GPU without a matching module, it raises a dashboard notice and doctor finding INS-001 with the exact command, but it never downloads modules on its own: new executable code arrives only through `setup` or an update the user's update mode allows.

### 4. Backend modules

RFC-0040 owns how a module is built and the in-process ABI between the runner and a provider library. This section owns how modules are packaged, chosen, fetched, verified, installed and removed.

#### 4.1 Format

`syntropy-module-<name>-<version>-<target>.tar.zst`:

```text
module.toml               manifest (below)
lib/                      the provider library (libsyntropy_provider_<name>.so | .dll) and vendor libraries
LICENSES/                 Syntropy licence texts plus each vendor component's licence or EULA
NOTICE                    composite licence statement (ADR-0018 decision 5)
```

```toml
name = "cuda13"
version = "1.4.0"
build_id = "b3:9c1f…"              # must equal the core's build ID
target = "x86_64-unknown-linux-gnu"
provider_abi = 3                   # RFC-0040's ABI version
backend = "cuda"
providers = ["cuda", "ggml-cuda"]
tier = 1
licence = "MIT OR Apache-2.0 for Syntropy code; NVIDIA components under the CUDA EULA"

[requires]
driver_min = "580.0"
gpu_archs = ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120", "sm_121"]
os_min = "linux-5.15"

[[files]]
path = "lib/libsyntropy_provider_cuda.so"
size = 0                           # filled by the release pipeline
blake3 = "…"

[[vendor_fetch]]                   # cuda13: NVIDIA archives setup fetches from NVIDIA (§4.4)
component = "libcublas"            # as named in NVIDIA's redistributable manifest
version = "…"
url = "https://…/redist/libcublas/linux-x86_64/…"   # NVIDIA's redistributable server
sha256 = "…"                       # pinned by the release pipeline from NVIDIA's manifest
files = ["lib/libcublas.so.13", "lib/libcublasLt.so.13"]   # extracted unmodified beside the provider
eula = "LICENSES/CUDA-EULA.txt"
```

The manifest is part of the module's TUF target, so the pinned hashes are covered by the release signatures. A fetched archive whose SHA-256 differs is deleted and reported as `ModuleSignatureInvalid`, as for a module file.

#### 4.2 Resolution

```text
resolve(census) -> Vec<Module>:
    for gpu in census.gpus:
        match gpu.vendor:
          NVIDIA:
            if gpu.cc ≥ 7.5 and driver ≥ 580:  want cuda13
            elif gpu.cc in {6.0, 6.1, 7.0}:    want cuda12-legacy (Tier 3); say so
            else:                              want cuda12-legacy if driver supports it, and print
                                               "driver ≥ 580 enables the supported CUDA 13 module"
            if no CUDA module is possible and Vulkan 1.3 is present: want vulkan
          AMD:
            if Linux and gpu.gfx in rocm.gpu_archs and /dev/kfd exists and the service user is in `render`
               (or will be, after the step 5 offer): want rocm
            else if Vulkan 1.3 is present:                                  want vulkan
          Intel:   want vulkan if Vulkan 1.3 is present (sycl only when asked: --modules sycl)
          Apple:   nothing (built in)
          other:   want vulkan if Vulkan 1.3 is present
    WSL2 with an NVIDIA paravirtual GPU: as NVIDIA, with the WSL2 pinning note (RFC-0001 §9.2); setup and
      doctor report WSL2 as Tier 3 (best effort), pending a WSL2 row in ADR-0020
    return the union, deduplicated; CPU-only machines get no modules
```

Several modules can be installed at once, for example `cuda13` and `vulkan` on a host with an NVIDIA and an Intel GPU. Per-op provider choice among them is RFC-0040's job.

#### 4.3 Fetch, verify, install, load

- **Fetch.** Modules are TUF targets (§7.2). The fetch resumes with HTTP `Range`, runs on the daemon's `bulk` runtime and yields to the data plane (RFC-0001 §8.4 rule 6).
- **Verify.** Length and hashes against the TUF target, then every file against `module.toml`, then `build_id` against the core. Any mismatch deletes the download and reports `ModuleSignatureInvalid` (RFC-0034 §12.2).
- **Install.** Into `<install root>/versions/<ver>/modules/<name>/`, so a module always lives beside the core of the same version and moves with it on update and rollback. Package installs use `/usr/lib/syntropy/modules/<name>/`.
- **Load.** Only runners and probe-mode processes load modules; the daemon never does (RFC-0001 §4.2 rule 4). The runner loads the provider library by absolute path:
  - Linux: vendor libraries resolve through an `$ORIGIN`-relative RPATH baked into the provider library, never through `LD_LIBRARY_PATH`;
  - Windows: `LoadLibraryExW` with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, so DLL search cannot be hijacked from the working directory.
  Immediately before `dlopen`, the loader re-checks the module's hash against its TUF target and its `build_id` and `provider_abi` against the core (RFC-0034 §12.2 rule 1). The modules directory is user-owned and loaded by absolute path only.
- **Remove.** `syntropy setup --modules auto` removes modules no device needs any more, after confirmation. Old versions' modules go with their version directory (§7.3).

#### 4.4 Vendor components

- **CUDA: fetched from NVIDIA, not redistributed** (ADR-0507 rule 2; open question 2, answered).
  - **`cuda13` ships only Syntropy code.** Its `module.toml` pins, per target, the NVIDIA redistributable archives the providers need: the CUDA runtime, NVRTC and nvJitLink, plus cuBLAS and cuBLASLt, which the ggml CUDA provider links. NVIDIA publishes these per component with SHA-256 digests in a JSON manifest (`redistrib_13.4.3.json`, released 2026-10-13, covers linux-x86_64, linux-sbsa and windows-x86_64); the release pipeline copies the hashes from it.
  - **What `setup` does.** It shows the CUDA EULA once and records that it was shown. It then downloads each archive from NVIDIA's redistributable server on the bulk runtime, checks its SHA-256 against the pinned value, and extracts the listed files unmodified beside the provider library. Syntropy itself redistributes no NVIDIA file in the script, MSI and app channels.
  - **NVIDIA files are never changed.** Linux files stay byte-identical (EULA §2.3): the `$ORIGIN` RPATH sits on Syntropy's provider library, and NVIDIA's libraries are never patched. Authenticode signs only Syntropy-built files; NVIDIA's DLLs keep NVIDIA's signatures and are pinned by hash (§9). The driver library (`libcuda.so`, `nvcuda.dll`) is never shipped or fetched: it comes from the installed driver. The module's composite licence statement keeps NVIDIA files outside `MIT OR Apache-2.0` (ADR-0018 decision 5).
  - **Channels that would bundle.** The deb and rpm module packages (M2), the `cuda13` OCI image (M3) and the offline bundles (M4) install under a package manager, in a container or with no network, so they would carry NVIDIA files. Each ships NVIDIA files only where NVIDIA's redistribution terms allow it for that channel. Otherwise it uses the fetch path: the deb and rpm packages through NVIDIA's own repository packages (§2.4), the image at first start into its state volume, checked against the pinned hashes (§8), and an offline bundle once the host can reach NVIDIA (§2.5).
  - **Last fallback: the user's installed toolkit.** When the fetch route is unavailable (NVIDIA's server unreachable or gated, or an offline host whose bundle may not carry NVIDIA files), `cuda13` dlopens the CUDA toolkit the user installed. SC-24 then needs that extra step, which `setup` names, and records it as an exception.
  - **Size.** The archive no longer carries cuBLAS and cuBLASLt, but the first install downloads them, so §12's `cuda13` budget is read as the module plus its fetched archives. Open question 3 stays answered: the libraries are needed, and their size comes from RFC-0040 §10.6's per-component report.
- **ROCm.** The `rocm` module ships the all-architecture user-space libraries, because AMD recommends that variant for redistribution (`backends-and-portability.md` §5.3). The kernel driver comes from the OS. It also carries rocBLAS and hipBLAS, which the ggml HIP provider links. There is no licence obstacle to shipping ROCm's user space: AMD lists rocBLAS, hipBLAS, hipBLASLt, HIP and CLR as MIT and ROCr as NCSA. The module leaves out RCCL (unused), ROCgdb and the binary-only pieces, and ships hipCUB's notice if HIP kernels compile its headers; hipCUB's custom licence text is checked at the pinned tag (open question 6's licence part, answered).
- **Vulkan in M2.** The M2 `vulkan` module holds only ggml's Vulkan provider (RFC-0040, COMP), as a Tier 2 preview: allocations are ggml's own, through its buffer type, and the tier manager has no elastic arenas on it (ADR-0403 decision 2's fallback). The dashboard, `setup` and the model list label it "Vulkan (preview)". M3 adds the native Vulkan provider to the same module.
- **Vulkan, Metal and DirectX** are system components and are never shipped.

### 5. Platform integration

#### 5.1 Install layout

| Install kind | Binary | Versions and modules | State and store | Config |
|---|---|---|---|---|
| Linux, `install.sh` | `~/.local/bin/syntropy` → `current` | `~/.local/share/syntropy/versions/<ver>/` | `$XDG_STATE_HOME/syntropy`, `$XDG_DATA_HOME/syntropy/store` | `$XDG_CONFIG_HOME/syntropy/` |
| Linux, `install.sh --system` | `/usr/local/bin/syntropy` → `current` | `/opt/syntropy/versions/<ver>/` | `/var/lib/syntropy/{state,store}` | `/etc/syntropy/` |
| Linux, `.deb`/`.rpm` | `/usr/bin/syntropy` | `/usr/lib/syntropy/` (one version, owned by the package manager) | `/var/lib/syntropy/{state,store}` | `/etc/syntropy/` |
| macOS, app | `~/.local/bin/syntropy` or Homebrew `bin` → app executable | `Syntropy.app` replaced as a whole; the previous app kept as `Syntropy-<ver>.app` in the versions directory `~/Library/Application Support/Syntropy/versions/` for rollback | `~/Library/Application Support/Syntropy/{state,store}` | same directory |
| Windows, per-user | `%LOCALAPPDATA%\Programs\Syntropy\bin\syntropy.exe` (stable path, §6.3) on the user PATH | previous versions in `…\Programs\Syntropy\versions\<ver>\`; modules in `bin\modules\` | `%LOCALAPPDATA%\Syntropy\{state,store}` | `%LOCALAPPDATA%\Syntropy\` |
| Windows, per-machine | `%ProgramFiles%\Syntropy\bin\syntropy.exe` (stable path) | previous versions in `%ProgramFiles%\Syntropy\versions\<ver>\` | `%ProgramData%\Syntropy\{state,store}` | `%ProgramData%\Syntropy\` |

These match RFC-0054 §1. On Linux and the script installs, `current` is a symlink switched atomically on update. On Windows the binaries keep a stable path for firewall rules, and an update swaps files by rename (§6.3, §7.3). On macOS the app bundle is swapped atomically (§7.3).

#### 5.2 macOS app bundle and the tray helpers

- `Syntropy.app/Contents/MacOS/syntropy` is the one binary. `Info.plist` sets:
  - `LSUIElement = true`: no Dock icon;
  - `NSLocalNetworkUsageDescription`: one sentence saying the app finds and talks to the user's other Syntropy devices;
  - `NSBonjourServices`: the service types RFC-0030 defines;
  - the bundle identifier (§9).
- **Signing.** Developer ID, hardened runtime and notarisation (§9). No exceptions to library validation are requested, because nothing is loaded from outside the bundle on macOS. The app is not sandboxed: the App Sandbox does not allow the memory wiring, raw device access and background service this product needs, which is also why there is no Mac App Store build.
- **The LaunchAgent** (§6) runs the app's executable, so the Local Network prompt and the firewall prompt show the app's name and usage text.
- **`syntropy tray`** (M4) is a mode of the same binary, started by a second LaunchAgent on macOS and a logon task on Windows. It shows a status icon: idle, serving, degraded or approximate in use. Its menu opens the dashboard, pauses participation (sets this node to `ingress-only` until resumed), shows update status, and quits the tray only. It uses the cross-platform `tray-icon` crate (MIT or Apache-2.0) and talks to the daemon through the local socket.

#### 5.3 Privileges and limits

| Platform | What Syntropy needs | How it gets it |
|---|---|---|
| Linux system service | lock large amounts of memory; GPU device access | unit: `LimitMEMLOCK=infinity`, `AmbientCapabilities=CAP_IPC_LOCK`, `SupplementaryGroups=video render` (§6.1) |
| Linux user install | a memlock limit large enough for the pinned budget (RFC-0001 §9.2) | `setup` offers, with sudo and consent, either `/etc/security/limits.d/90-syntropy.conf` (`<user> - memlock unlimited`) or a drop-in for the user manager (`/etc/systemd/system/user@.service.d/90-syntropy.conf` with `LimitMEMLOCK=infinity`); declined → RFC-0020 works within the existing limit and doctor OS-001 explains the cost |
| Linux, Intel AMX | permission for AMX tile state | requested at runtime with `arch_prctl`; no install change |
| macOS | wired memory within the GPU wired limit | nothing at install. Raising `iogpu.wired_limit_mb` is opt-in (`resource.macos_raise_wired_limit`): it needs admin and installs a LaunchDaemon that applies the value at boot, recorded in the ledger (ADR-0014: never silently) |
| Windows | locking pages for GPU access | `SetProcessWorkingSetSizeEx` then `VirtualLock`, which standard users may do within their quota; no privilege is granted at install |

#### 5.4 Firewalls and local-network permission

- **Linux.** No firewall changes by default. Doctor NET-001 detects blocked multicast or UDP and prints the exact `firewall-cmd` or `ufw` command; `doctor --fix` applies it with consent and records it.
- **Windows.** Per-machine installs add an inbound rule for `syntropy.exe` on the Private profile only. Per-user installs cannot add rules without admin, so the first listening socket raises the Windows Defender Firewall prompt, and `setup` tells the user beforehand to allow Private networks only.
- **macOS.** The application firewall asks on first listen for a signed app. Local Network privacy asks on first LAN access, using the bundle's usage description. `setup` triggers both while the user is watching, by starting discovery, so the prompts do not appear later on an unattended machine. Both grants must also survive updates. The bundle keeps its identifier, its Developer ID Team ID and its path (`renamex_np` swap, §7.3 step 7), and the install matrix checks the result on every macOS image: after an unattended update and after a rollback, discovery and a LAN listen raise no Local Network or application-firewall prompt. If a macOS release breaks this, the update notice asks the user to re-approve while watching, rather than leaving an unattended node silently cut off from the LAN.

### 6. Service management

ADR-0509 sets the defaults.

| Install kind | Service | Runs as | Starts | Stops |
|---|---|---|---|---|
| Linux `install.sh` | systemd user unit `~/.config/systemd/user/syntropy.service`; `setup` asks to enable lingering (`loginctl enable-linger`) so it runs without a login session, recording whether lingering was already on | the user | at boot with lingering, else at login | drain (RFC-0001 §3.4) |
| Linux `--system`, `.deb`, `.rpm` | systemd system unit `syntropy.service` | `syntropy` system user | at boot | drain |
| macOS app, cask, script | LaunchAgent `~/Library/LaunchAgents/<bundle id>.daemon.plist` | the user | at login | drain |
| macOS server option: `syntropy service install --daemon` (M1), `.pkg` (M2) | LaunchDaemon with `UserName` (§6.2) | a dedicated `_syntropy` account, or one the administrator names; root only as the fallback | at boot, once FileVault is unlocked | drain |
| Windows per-user (MSI, script) | Task Scheduler task "Syntropy" at logon, starting the GUI-subsystem launcher `syntropyw.exe` (§6.3) | the user | at logon | drain; at logoff, cancel and persist within the OS time limit (§6.3) |
| Windows per-machine | Windows service `Syntropy`, virtual account `NT SERVICE\Syntropy` | service account | at boot | drain |
| containers | none: `syntropy serve` is PID 1 | container user | container start | `SIGTERM` → drain |

#### 6.1 systemd units

```ini
[Unit]
Description=Syntropy node daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
NotifyAccess=main
# per install kind: /usr/bin/syntropy (deb, rpm) or /usr/local/bin/syntropy (install.sh --system),
# which resolves through `current`
ExecStartPre=+-/opt/syntropy/guard/syntropy __update-guard
ExecStart=/usr/bin/syntropy serve
User=syntropy
Group=syntropy
SupplementaryGroups=video render
AmbientCapabilities=CAP_IPC_LOCK
LimitMEMLOCK=infinity
LimitNOFILE=1048576
StateDirectory=syntropy
RuntimeDirectory=syntropy
RuntimeDirectoryPreserve=restart
ConfigurationDirectory=syntropy
FileDescriptorStoreMax=16
WatchdogSec=30
Restart=on-failure
RestartSec=2
KillMode=process
TimeoutStopSec=150
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectControlGroups=yes

[Install]
WantedBy=multi-user.target
```

- **`Type=notify`.** The daemon sends `READY=1` once its API listens and its cluster state is loaded, and `WATCHDOG=1` from the `ctl` runtime, so systemd restarts a wedged control plane.
- **`KillMode=process`**. On stop or restart, systemd signals only the daemon, never the runners in the unit's cgroup.
  - On a normal stop, the daemon drains its runners itself and exits.
  - On an update restart, it exits without draining, and the runners survive under §6.5's orphan rule. systemd logs them as left-over processes of the unit, which is expected.
  - If the daemon hangs past `TimeoutStopSec`, systemd kills only the daemon. The runners then expire under §6.5, so nothing outlives the unit for long.
- **`FileDescriptorStoreMax`** lets the daemon hand its listening sockets to systemd before an update restart and receive them back, so connections queue in the kernel instead of being refused (§7.3).
- **`RuntimeDirectoryPreserve=restart`** keeps `/run/syntropy`, where runners' adoption sockets live, across a restart.
- **`ExecStartPre=+-…/guard/syntropy __update-guard`** runs the update guard (§7.4), a copy of the last committed build that does not depend on the new build being able to start. The `-` keeps a missing guard from blocking start.
  - The `+` runs the guard with full privileges, outside `User=` and the sandbox, so that on `install.sh --system` it can switch `/opt/syntropy/current` back.
  - Its attack surface is small. The guard binary and every version directory are root-owned, so the daemon cannot change them. Its only input from the daemon's side is `update-pending.toml`. It only renames `current` among existing version directories and restores backups into the state directory, which the daemon owns anyway.
- **`ExecStart`** depends on install kind: `/usr/bin/syntropy` for deb and rpm; `/usr/local/bin/syntropy` → `/opt/syntropy/current` for `install.sh --system`. Package installs ship no guard, because they never self-update; the `-` makes its absence harmless.
- **Hardening.** The directives used are the ones that do not interfere with GPU device nodes, io_uring or memory locking. `PrivateDevices` and `MemoryDenyWriteExecute` are deliberately absent: the first hides GPU devices, and the second breaks GPU drivers that generate code at runtime. `ProtectSystem=strict` makes the install root read-only to the daemon. That is why system installs update through a privileged helper or by an administrator (§7.1).
- **The user unit** is the same, minus `User`, `Group`, capabilities and the `Protect*` directives that need a system manager. Its memlock limit comes from the user manager (§5.3), and its guard lives in `~/.local/share/syntropy/guard/`.

#### 6.2 launchd agent

`ProgramArguments` runs the guard, which execs the app executable: `[<guard>/syntropy, __update-guard, --then-exec, <app>/Contents/MacOS/syntropy, serve]`. The guard is a copy of the last committed build (§7.4). The plist keys:
- `RunAtLoad = true`;
- `KeepAlive = { SuccessfulExit = false }`, so it restarts on crashes but not after `syntropy stop`;
- `AbandonProcessGroup = true`, so launchd does not kill the job's process group, runners included, when the daemon exits;
- `ProcessType = Standard`;
- `ExitTimeOut = 150`.
Runners also call `setsid()` at start, so they sit in their own session and process group whatever the job settings. Logs go to unified logging (RFC-0053 §4.2), so the plist sets no standard-output paths.

**Headless Macs** (open question 4, answered). Automatic login is withdrawn as the documented answer: Apple makes it unavailable while FileVault is on, so recommending it told server users to turn off disk encryption. The server option is a LaunchDaemon in `/Library/LaunchDaemons/` with the same keys as the agent plus `UserName` (and `GroupName`), set to a dedicated `_syntropy` account that `service install --daemon` creates, or to an account the administrator names. Apple's TN3179 (revised 2026-10-06) lists any daemon started by launchd as allowed local-network access, separately from programs running as root, so a non-root daemon should not prompt. Developer forum threads report bugs here, so it ships only after the spike PROD-E10-T11 (pulled into M1) passes **with FileVault on**, on macOS 15 and 26:
- LAN discovery and listening from the `UserName` daemon, with no prompt;
- Metal device creation and wired memory with no GUI session;
- `syntropy tune` and `iogpu.wired_limit_mb` from that account;
- application-firewall behaviour, state-directory ownership, and grants surviving an unattended update;
- the retry behaviour after a first denial, in the same session as DIST's Local Network attribution spike (DIST-E3-T09).
If the non-root daemon gets a prompt or no Metal device, the fallback is a root LaunchDaemon that drops to `_syntropy` after it binds its sockets and opens its devices, tested in the same spike. The docs state the FileVault consequence: after a reboot nothing starts until the disk is unlocked, which macOS 26 on Apple silicon can do over SSH before boot (a third-party report the spike checks). The `.pkg` carries the daemon from M2 (PROD-E10-T04).

#### 6.3 Windows

- **Service** (per-machine).
  - Built with the `windows-service` crate.
  - Accepts stop and shutdown, and reports `STOP_PENDING` with wait hints while it drains.
  - Recovery actions: restart after 2 s and 10 s, then run the update guard (`<root>\guard\syntropy.exe __update-guard --request rollback`) on the third failure (§7.4).
  - That command runs as `NT SERVICE\Syntropy`, which cannot write `%ProgramFiles%`. So the guard only writes a rollback request and starts the privileged helper service `SyntropyUpdate` (§7.1), which switches back and restarts `Syntropy`.
  - The SCM does not put a service in a job object, so runners are not tied to the service process. The daemon puts its runners in its own job object, created **without** `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. It uses that job to account for runners and to terminate them on an explicit shutdown, and its own exit leaves them running.
- **Per-user** (MSI per-user, `install.ps1`).
  - The logon task starts `syntropyw.exe`, a small GUI-subsystem launcher shipped in the same signed archive. A console-subsystem `syntropy.exe` started by Task Scheduler would open a visible console, and closing it would kill the daemon.
  - The launcher starts `syntropy.exe serve` with `CREATE_NO_WINDOW`, restarts it on failure, and acts as the update guard (§7.4). It is replaced only when an update commits, so during an update's health window it is the previous build's launcher.
  - It owns a hidden top-level window. On `WM_QUERYENDSESSION` it calls `ShutdownBlockReasonCreate` and asks the daemon, through the local socket, to stop admitting work, cancel live sequences with a retryable error and persist sessions within the time Windows allows, typically seconds. A logoff or shutdown therefore ends sequences and does not drain them for 120 s, and the docs say so.
  - Runners are created with `CREATE_BREAKAWAY_FROM_JOB` when the task's job allows breakaway, and placed in the daemon's own job without kill-on-close, as for the service. If the task's job forbids breakaway, the runners die with the task's process tree. In that case §6.5 reports re-adoption as unavailable, and updates on that install use drain and restart.
- **Firewall rules and stable paths.**
  - Windows Defender Firewall program rules are keyed by executable path. So on Windows the binaries run from a stable path, `…\Syntropy\bin\syntropy.exe` and `bin\syntropyw.exe`, rather than through a versioned directory.
  - An update renames the running binary into `versions\<old>\` (Windows permits renaming a running image) and moves the new one into `bin\`. The path that firewall rules and the first-run prompt approved stays the same.
  - The install matrix checks that no firewall prompt appears after an unattended update.

#### 6.4 Commands

`syntropy service install|uninstall|start|stop|restart|status [--user|--system]` manages whichever kind applies, and records install and uninstall in the ledger. `status` combines the service manager's state with the daemon's `/readyz` and the runners' states. `syntropy stop` (RFC-0051 §2.1) stops the daemon through its local socket, so a service manager with restart-on-failure does not restart it: the daemon exits with success after a requested stop.

#### 6.5 Runner survival across a daemon restart

RFC-0001 §3.4 lets a restarted daemon re-adopt its runners, and §7.3 relies on that. ADR-0015 decision 6 forbids it as written: runners are killed by `PR_SET_PDEATHSIG`, a `NOTE_EXIT` watch, or a kill-on-close job when the daemon exits. This RFC replaces that mechanism with a bounded orphan rule, which supersedes ADR-0015 decision 6 (amendment A1).

**The orphan rule.**
1. **Detecting daemon loss.** Runners use neither parent-death signals nor exit watches. A runner detects daemon loss by end-of-file on its IPC command socket, or by the daemon's heartbeat counter in the shared-memory header (RFC-0001 §3.5) not advancing for `runner.daemon_silence` (2 s, RFC-0033 §8.2). The daemon bumps that counter from a dedicated thread that never blocks on the `ctl` runtime, so a busy control plane does not look dead.
2. **Orphan mode.** The runner keeps serving sequences whose ingress is another node. It stops admitting new sequences. It listens on its adoption socket (`<runtime dir>/adopt/<instance>.sock`, mode 0600, peer credentials checked), and keeps writing `local_tokens` and telemetry into its rings, dropping the oldest telemetry when they fill.
   - **Leaving orphan mode without adoption.** If the mode was entered on a stalled heartbeat while the command socket is still connected, and the heartbeat resumes, the runner returns to normal operation and resumes admitting, with no adoption. It reports the stall's length as a `RunnerStateChanged` note, so a daemon that keeps stalling shows up in the dashboard and in doctor RUN-006.
3. **Adoption** (in RFC-0001 §3.5's direction). A daemon that connects to the adoption socket within `runner.orphan_grace` (RFC-0033 §8.2, default 30 s) receives the runner's `Hello { pid, build_id, ipc_protocol, obs_tables }`.
   - The daemon checks `ipc_protocol`. On an observability-table digest mismatch it asks for the tables (RFC-0053 §4.1).
   - If the protocol matches and the tables are equal or translatable, it accepts, and the runner resumes normal operation.
   - Otherwise the daemon refuses the adoption, and the runner drains under step 4. A runner never has to judge whether the daemon can translate.
4. **Expiry.** If no daemon adopts the runner within `runner.orphan_grace`, it drains for at most `recovery.drain_deadline` and exits. So a runner never outlives its daemon by more than `runner.orphan_grace + recovery.drain_deadline`. That bound replaces "runners never outlive the daemon".
5. **A normal stop is unchanged.** `syntropy stop`, a service stop and an OS shutdown drain runners before the daemon exits. Only the update path exits the daemon without draining.

**Per-OS settings this needs, and where re-adoption is unavailable:**

| Platform and install | Settings | Re-adoption |
|---|---|---|
| Linux, systemd system or user unit | `KillMode=process`, `RuntimeDirectoryPreserve=restart`, `FileDescriptorStoreMax` (§6.1); no `PR_SET_PDEATHSIG` on runners | available |
| Linux, container (PID 1) | none; container restarts end every process | unavailable: images update by tag, so drain and restart |
| macOS, LaunchAgent or LaunchDaemon | `AbandonProcessGroup = true`; runners call `setsid()` (§6.2) | available |
| Windows service (per-machine) | the daemon's own job without kill-on-close (§6.3) | available |
| Windows logon task (per-user) | breakaway from the task's job, then the daemon's own job (§6.3) | available when the task's job allows breakaway (verified by PROD-E12 spike); otherwise drain and restart |
| Android, iOS (embedded runner) | runner is in-process | unavailable: app updates through the store |
| any install where `ipc_protocol` changed between builds | — | unavailable for that update: drain and restart (§7.3) |

When re-adoption is unavailable, the update path drains every runner, restarts the daemon, and reloads the models. The update then costs SC-23's load time instead of a restart, and `syntropy update status` says which path was taken.

**Verification.** A soak test per OS restarts the daemon through its real service manager (`systemctl --user restart`, `systemctl restart`, `launchctl kickstart -k`, the SCM, and Task Scheduler) while a sequence streams to a remote ingress. It checks that the sequence completes with no lost or duplicated tokens, and that each runner is adopted or expires within its bound (PROD-E12).

### 7. Updates

#### 7.1 Modes and channels

| Install kind | Default `update.mode` | Meaning |
|---|---|---|
| `install.sh`, `install.ps1`, app zip, MSI per-user (non-winget) | `auto` | check every `update.check_interval`; download in the background; apply in the update window (below) |
| `install.sh --system`, MSI per-machine | `notify` | the daemon runs unprivileged and cannot write the install root (`ProtectSystem=strict`, `%ProgramFiles%`) or restart its own system service. An administrator runs `sudo syntropy update apply` (or from an elevated prompt). An administrator may set `update.mode = "auto"` in the root-owned system file; the privileged helper then applies on the daemon's request |
| `.deb`, `.rpm`, Homebrew cask, winget, macOS `.pkg` | `package-manager` | check and notify only, naming the package command (`apt upgrade`, `brew upgrade --cask syntropy`, `winget upgrade Syntropy.Syntropy`); the self-updater never writes |
| containers, offline bundles | `off` | images are upgraded by tag; offline installs by a new bundle |
| any, by choice | `notify` | check and notify; download and apply on `syntropy update apply` |

**Who acts on each install kind**. Commit and rollback happen minutes after `update apply` returns, so system installs need a privileged actor that outlives the administrator's command.

| Install kind | Apply (§7.3 steps 1–8) | Commit (step 9) | Rollback (§7.4) |
|---|---|---|---|
| user-level: script, app zip, per-user MSI | the daemon | the daemon | the guard, as the same user |
| `install.sh --system` | `sudo syntropy update apply`; in `auto` mode, the helper on the daemon's request | the helper, on the daemon's `commit` request | the guard, run privileged by `ExecStartPre=+` (§6.1) |
| MSI per-machine | `syntropy update apply` from an elevated prompt; in `auto` mode, the helper | the helper service | the guard writes a request; the helper service switches back (§6.3) |
| `.deb`, `.rpm`, Homebrew cask, winget, macOS `.pkg` | the package manager or MDM | — | the package manager: install the older package |

- **The privileged helper.** It is installed with every system install and is no longer opt-in.
  - **Linux:** a root oneshot `syntropy-update.service` with `ReadWritePaths=/opt/syntropy /var/lib/syntropy`, started by `syntropy-update.path` when a request appears in `/var/lib/syntropy/state/update-request/`, the only directory the daemon writes for it.
  - **Windows:** `SyntropyUpdate`, a demand-start service running as LocalSystem, whose DACL grants `NT SERVICE\Syntropy` start rights only. Its requests go in `%ProgramData%\Syntropy\state\update-request\`.
  - **Requests** are `apply` (honoured only in `auto` mode), `commit` and `rollback`.
  - **Trust.** The helper trusts no file the daemon staged. It re-verifies every target it installs or promotes against its own TUF client. A commit only promotes the version named in `update-pending.toml` and refreshes the guard from it after a hash check. So a compromised daemon can at most commit early. It cannot install anything unsigned.
- **Channels.** `stable` (default), `beta`, `nightly`. Switching to a slower channel never downgrades. The node stays on its version until that channel catches up, unless the user runs `update rollback` or installs a version explicitly.
- **The update window.** `update.window = "idle"` (default) applies an update when no sequence has been live for 5 minutes and no request arrived in that time. Alternatively, a daily local time range. `update.require_mains = true` (default) defers on battery.
- **Notices** appear in the dashboard, in `syntropy update status`, and as a one-line note on the next interactive CLI command.

#### 7.2 Verifying what the updater fetches

Three RFCs share the update trust chain (ADR-0508):

| Concern | Owner |
|---|---|
| which keys exist, their custody, thresholds, rotation and compromise response: the TUF roles (root, targets, snapshot, timestamp, and the delegated `nightly` targets role), the Ed25519 release and nightly keys, the repository and platform signing identities | RFC-0034 §12.1, §12.3 |
| producing and signing artefacts, TUF metadata and detached signatures in protected release environments; channel promotion; keeping previous releases published | RFC-0062 §7, §9 |
| verifying all of it on the user's machine: `install.sh` and `install.ps1`, `setup`'s first self-check, the updater, module installation and module loading | this RFC |

**The update client is a TUF client** (RFC-0034 §12.2 rule 1). It uses the `tough` crate (MIT OR Apache-2.0), with the root metadata embedded in the binary as the trust anchor:
- **Refresh order.** timestamp → snapshot → targets, with each role's expiry and version monotonicity enforced. That rejects rollback, freeze and mix-and-match attacks, and lets any mirror or CDN serve the files untrusted.
- **Targets.** Per platform: the core artefact and every backend module, plus `catalogue.json`, the configuration JSON Schema, `signing-keys.json` (the current and next release and nightly public keys, RFC-0034 §12.1) and, from M1, `relays.json`: the built-in relay list that `net.relay_mode = auto` and `default` use (RFC-0030 §6.4). The relay list is therefore signed by the targets role like any other target, and a mirror cannot redirect relay traffic. Each release also embeds the list current at build time, which the client uses until it has fetched a newer one.
- **Custom fields on each target**: `version`, `channel`, `build_id`, `min_from_version` (the oldest version that may update straight to this one), `protocols` (control, data and blob plane majors, ADR-0011), `provider_abi`, `tier`.
- **Channels.** `stable` and `beta` are paths under the offline top-level targets role. Promotion from `beta` to `stable` (RFC-0062 §9.2) re-signs targets metadata with the offline targets key (**Maintainer action**) without rebuilding anything.
- **`nightly` is a delegated targets role**, scoped by path to `nightly/*`. Its key is held online in the KMS by the `nightly-signing` environment, so nightly builds publish without a hardware-token ceremony. The client follows the delegation from the top-level targets metadata, and a nightly key can never sign a stable or beta path. If the client library's delegation support proves insufficient (spike PROD-E12-T11, at the end of M0), the fallback is a separate nightly TUF repository whose root is embedded only in nightly builds. The client pins `tough` ≥ 0.20.0, which fixes CVE-2025-2886, a flaw in delegation search that accepted a lower-priority delegation it should have ignored, which is exactly what a path-scoped `nightly` role relies on. The hostile-mirror suite includes a terminating-delegation case (OQ-PKG-13).
- **Expiry of the offline targets role.** Targets metadata expires after 90 days (RFC-0034 §12.1). A re-signing ceremony (**Maintainer action**) is scheduled before expiry whether or not a release ships, and the release pipeline alerts at 30 days remaining (RFC-0062). An expired role makes clients refuse updates, which is safe but must not happen in practice.
- **Server-side rollback** (RFC-0062 §9.2 step 7) is expressed as a `withdrawn` list of versions in the targets metadata. The client never installs an older version on its own unless the installed version is on that signed list. A node on a withdrawn version rolls back to its previous installed version if it still has it (§7.4), or else installs the channel's current target as a signed, explicit downgrade. A user can always downgrade with `syntropy update install <version>`, which asks for confirmation.

**Before any TUF metadata is reachable**, `install.sh`, `install.ps1` and `setup`'s first self-check verify the downloaded artefact's detached Ed25519 signature. The key must be a release key embedded in the running binary, or the nightly key on `nightly` (RFC-0034 §12.2 rule 2). After that first run, the TUF client takes over.

**Modules** are TUF targets (ADR-0507). Before the runner `dlopen`s a module, the module loader re-checks the module's hash against its TUF target and its build ID against the core (RFC-0034 §12.2 rule 1). The check uses multithreaded BLAKE3 over the module's files; the cost is in the Performance section.

**Requests.** Plain `GET`s of static paths that name the platform and channel. No cookies and no identifiers (RFC-0034 §13, ADR-0510).

**Mirrors.** `update.mirror` points the client at any mirror. Because TUF metadata authenticates every file, a mirror needs no trust.

**Who signs the metadata.** RFC-0062 §7.1 produces and signs all of it: targets in the offline release ceremony and on a calendar before expiry, snapshot on each publish, the timestamp daily, and the delegated `nightly` role online, as RFC-0034 §12.1 defines (open question 8, answered). The offline signing is a **Maintainer action**, done by the maintainers who hold the offline keys. Everything around it is agent work under ADR-0021: the TUF client, the installers, the online signing jobs, the expiry monitor and the install matrix. No agent holds an offline key.

#### 7.3 Applying an update

```text
apply(target):
  1  refresh TUF metadata (timestamp → snapshot → targets, §7.2); pick the newest target on the
     channel with min_from_version ≤ installed version
  2  download core and every installed module for the target into <root>/staging/<ver>/,
     resumable; verify length and hashes against the TUF targets
  3  extract to <root>/versions/<ver>/
  4  <root>/versions/<ver>/syntropy __selftest --root <root>     (no network, no GPU context in-process):
       - parse the live configuration with the new schema
       - copy the state directory to a scratch location and run the state migrations on the copy
       - in a probe-mode child: load every module's provider library and enumerate devices
       - run the CPU parity smoke test on the tiny built-in fixture model (RFC-0060)
     failure → mark <ver> bad locally, keep the current version, report; stop
  5  back up the state files a migration touches, and every config file, into <state>/.backup/<old ver>/;
     write <state>/update-pending.toml { from, to, started_at, attempts: 0, healthy: false }
  6  before restarting:
       - stop admitting new local-ingress requests; wait up to update.local_drain for live
         local-ingress sequences to finish, then end the rest with a retryable error (their client
         connections belong to this process and cannot survive it)
       - hand listening sockets to the service manager where it can hold them (systemd's fd store);
         elsewhere, new connections are refused for the restart's duration
       - announce a planned restart to the cluster (RFC-0030 §5.9, from M3 with re-adoption):
         Leave { node, reason: Restart { back_within_ms: update.restart_budget } }, so peers suspend
         suspicion and no recovery starts. Before M3, the daemon shuts down gracefully
         with `Goodbye` beacons, and peers recover its stages at once, as for any graceful stop
  7  switch atomically:
       Linux, script installs: write the new symlink beside `current` and rename(2) it over `current`
       macOS: renamex_np(new.app, Syntropy.app, RENAME_SWAP) on APFS, then move the old bundle into
              the versions directory
       Windows: rename the running bin\ files into versions\<old>\, move the new ones into bin\
  8  restart the daemon through its service manager:
       re-adoption available (§6.5) and ipc_protocol unchanged → exit without draining; runners serving
         remote-ingress sequences continue under the orphan rule; the new daemon re-adopts them
       otherwise → drain runners first (sequences finish or migrate, RFC-0001 §7.5), then restart
  9 health window, which stays armed until all of the following hold:
       - API ready; cluster rejoined; the planned-restart notice cleared
       - for every instance, at least one runner refreshed onto the new build (drained and restarted)
         and its canary passed; remaining old-build runners refresh at the next idle window
       - at least 5 minutes elapsed
     → commit, by the actor §7.1 names for the install kind: mark update-pending healthy, refresh the
       guard copy to the new build, delete staging, keep the two previous versions, let pack GC run
```

**What survives an update, stated exactly.**
- **Remote-ingress sequences** keep generating on re-adopted runners. Their tokens go straight to the other node, and `local_tokens` is not involved.
- **Local-ingress sequences** finish before the restart (step 6) or end with a retryable error. Their HTTP connections die with the daemon process.
- **Loaded models** stay loaded when re-adoption is available (§6.5). Otherwise every runner drains and reloads.
- **In the default `idle` window** no sequence is live at the time, so step 6 waits for nothing.
- **The restart's budget** (`update.restart_budget`, default 10 s [target]) is sent as the notice's `back_within_ms`. RFC-0030 caps that at `membership.restart_grace` (30 s), and validation rejects a larger budget. If the daemon is not back within it, peers treat the node as failed and start normal recovery.

#### 7.4 Rollback

- **Who acts.** The update guard: a copy of the last committed build at `<root>/guard/`, which commit (step 9) refreshes. It runs before the daemon on every start:
  - systemd: `ExecStartPre`, with the `+` prefix on system installs so it can rewrite the root-owned install (§6.1);
  - launchd: the plist runs the guard, which execs the daemon (§6.2);
  - Windows service: the SCM recovery action on the third failure, which hands the switch to the privileged helper service (§6.3, §7.1);
  - Windows logon task: the launcher, which is still the previous build until commit (§6.3).
  The guard never depends on the new build being able to start.
- **Automatic triggers** while `update-pending.toml` is not marked healthy:
  - the guard counts starts. A fourth start within the window, meaning three crashes, switches back;
  - `/readyz` not ready within 60 s;
  - a canary failure on an instance that passed before the update;
  - the cluster refusing the node at admission for a protocol mismatch the update introduced.
- **The action.**
  - Switch back with step 7's method.
  - Restore the state and config backups.
  - Restart, and mark the version bad (it is skipped until a newer one exists).
  - Raise a notice with the cause and a `doctor --bundle` hint.
  Because commit waits for refreshed runners' canaries, a kernel or provider regression in the new build is still inside the window when it shows.
- **Manual**: `syntropy update rollback` switches to the previous installed version by the same steps.
- **Data written by the newer version.**
  - Any config file the older binary rejects (a newer `config_version`, or a key it does not know, RFC-0054 §8) is restored from the step 5 copy. The rejected file is kept beside it as `config.toml.rejected`, with a notice.
  - Packs built under a new layout version have new `PackKey`s, and the old packs are kept until commit, so a rollback does not rebuild packs.
- **Clusters.** A rollback can leave this node on an older protocol major than its cluster. It is then refused at admission, which doctor CLU-001 and the dashboard report. Coordinated updates (§7.5) exist to make that rare.

#### 7.5 Cluster-coordinated updates

With `update.cluster_coordinated = true` (cluster scope, default on):
- **Minor and patch updates** that keep every protocol major are rolling: each node applies in its own window. Mixed minor versions interoperate under RFC-0062's compatibility policy.
- **An update that changes a protocol major** is staged everywhere first.
  1. Each node downloads and self-tests, then reports `staged_build` (proposed amendment A3).
  2. When every member has staged, the coordinator picks an idle window and sets `update.switch_at` in cluster config.
  3. All nodes drain and switch together.
  A node that fails its self-test holds the switch, and the dashboard names it. A node that is offline at the switch rejoins once it has updated.

### 8. Containers

- **Images** `<registry>/syntropy:<version>-<flavour>`, with flavours `cpu` (amd64, arm64), `cuda13` (amd64, arm64), `rocm` (amd64) and `vulkan` (amd64). Each has a glibc base with the core binary and that flavour's module preinstalled. Unless NVIDIA's redistribution terms allow images to carry NVIDIA files, the `cuda13` image carries only Syntropy code and fetches the pinned NVIDIA archives at first start into the state volume, checked against the module manifest's hashes (§4.4). GPU drivers come from the host: the NVIDIA container toolkit for `cuda13`; device passthrough of `/dev/kfd` and `/dev/dri` for `rocm` and `vulkan`.
- **Entrypoint**: `syntropy serve --bind 0.0.0.0:7433`, with `/var/lib/syntropy` as a volume for state and store, `update.mode = "off"`. TLS comes from the node's local CA (RFC-0050 §11.1.1); the CA persists in the state volume, so clients trust it once.
- **First-start API key.** A non-loopback bind needs at least one API key and TLS, or the daemon refuses to start (RFC-0050 §11.1). So the image's entrypoint does three things on first start, when no key exists:
  - creates one `admin` key;
  - writes it to `/var/lib/syntropy/state/initial-admin-key` (mode 0600);
  - logs the path, never the key.
  Users read it with `docker exec … cat`, or mount their own `api-keys.toml`. The local CA's six-word fingerprint (RFC-0034 §6.5) is logged, for `syntropy trust` on clients. Users with a real certificate mount it and set `api.tls` to the files.
- **Recommended run:**

```sh
docker run -d --name syntropy --gpus all --network host \
  --ulimit memlock=-1 --cap-add IPC_LOCK \
  -v syntropy:/var/lib/syntropy <registry>/syntropy:<version>-cuda13
```

- **Host networking** is needed for LAN discovery (multicast). Under bridge networking, discovery cannot work, so the node runs with `discovery.mechanisms = []` and joins through invites or `cluster.static_peers` (RFC-0030). `setup`-style host preparation does not run in containers; the documented flags replace it.
- **io_uring** is blocked by the default seccomp profile, so the runner uses the `pread` pool (ADR-0013) and doctor OS-003 says so. The image documentation ships a seccomp profile that adds the io_uring syscalls, for users who want the faster path.
- **Not provided**: Kubernetes operators or Helm charts (RFC-0001 §1.2). A Compose file example is.

### 9. Signatures, hosting and identities

RFC-0062 §7 signs everything in its protected release environment, and RFC-0034 §12 owns the keys. This RFC states what the installers and the update client verify, and where:

| Artefact | Signature (produced by RFC-0062) | Verified, on the user's machine, by |
|---|---|---|
| macOS app bundle, `.pkg` | Developer ID Application and Installer certificates, hardened runtime, notarisation with stapled tickets | Gatekeeper at install and launch; `setup` additionally verifies the Syntropy signatures below |
| Windows `.exe`, Syntropy-built module `.dll`s, MSI, `install.ps1` | Authenticode with RFC 3161 timestamps. NVIDIA's DLLs are never re-signed: re-signing changes their bytes; they keep NVIDIA's signature and are pinned by hash (§4.4) | Windows and SmartScreen; `setup` additionally verifies the Syntropy signatures below |
| Linux packages and repository metadata | the repository keys of RFC-0034 §12.1 | apt and dnf |
| every artefact: tarballs, zips, app archives, modules, `SHA256SUMS`/`BLAKE3SUMS` | detached Ed25519 signatures with the release key, or the nightly key on `nightly` | `install.sh`, `install.ps1` and `setup`'s first self-check (§7.2) |
| TUF metadata: root, targets, snapshot, timestamp | the TUF role keys of RFC-0034 §12.1 | the update client, module installation and the module loader (§4.3, §7.2) |
| `signing-keys.json` (current and next release and nightly keys, and revocations) | a TUF target | the update client, so updated clients learn of key rotation and revocation (RFC-0034 §12.3) |
| provenance attestations and keyless signatures | RFC-0062 §6.3, RFC-0034 §12.1 | third parties and auditors; the client does not depend on them |
| OCI images | the same attestations on image digests | users' own admission policies |

- **No key material on the client.** The client holds only the embedded TUF root metadata and the embedded release and nightly public keys (RFC-0034 §12.1).
- **Signing identities** the release needs (open question 1, settled):
  - an Apple Developer ID Application and Installer certificate, with notarisation, for the macOS app and `.pkg`;
  - an Authenticode signing identity with RFC 3161 timestamps for the Windows binaries, MSI and `install.ps1`;
  - the Linux repository keys, and the TUF, release and nightly keys of RFC-0034 §12.1, held in the release KMS.
  - **The Developer ID Team ID must stay fixed.** Local Network and application-firewall grants on macOS attach to the app's code requirement, which includes the Team ID (§5.4). A change of team prompts every Mac user again, and an unattended node is cut off from the LAN until someone approves, so the macOS app is signed by one team from its first public release.
  - Holding the Developer ID and notarisation credentials and setting up each identity are **Maintainer actions**. Agents build and test the signing, notarisation and verification steps against those identities, and never create or hold the credentials.
- **Hosting.** Static HTTPS object storage behind a CDN serves manifests and artefacts at `<release-host>`. GitHub Releases is a mirror, the OCI registry holds images, and the apt and dnf repositories sit on the same static host. Because the client verifies everything against the embedded roots, any mirror can serve them. The release host and CDN log requesting IP addresses; what they log and for how long is stated in the project's privacy notice (RFC-0053 §8.1).
- **Names.** The bundle identifier (`<reverse-DNS>.syntropy`), the unified-logging subsystem (RFC-0053 §4.2), the Windows service and task names, the winget identifier and the Homebrew tap all follow from the release domain (PROD-E10-T02). They are generated from that one setting in the packaging templates rather than written out, so a change of domain before the first public release renames them consistently. After it they are fixed, because a changed bundle identifier or service name breaks installed nodes' grants and updates.

### 10. Mobile packaging (Tier 3)

Mobile nodes are best effort (ADR-0020). They run the embedded runner (ADR-0015 decision 7), and store rules shape everything:

| | Android | iOS |
|---|---|---|
| Library | AAR built with `uniffi` Kotlin bindings; `arm64-v8a` only | XCFramework with `uniffi` Swift bindings; A14 and newer |
| Backends | statically included in the app: CPU native, ggml Vulkan, OpenCL (Adreno), Hexagon. Google Play's Device and Network Abuse policy forbids fetching executable code from outside Play, so there are no downloadable modules | Metal and CPU, statically linked. App Store rules forbid downloaded code and spawning processes |
| I/O | `pread` pool (io_uring is blocked for apps) | `F_NOCACHE` reads |
| Running in the background | a foreground service with a persistent notification while participating; default policy: participate only on a charger and unmetered Wi-Fi (`resource.power`, `network.metered`) | effectively foreground-only. Background execution limits suspend the app, so the node participates while the app is open or during a background processing task on a charger. The usable memory cap comes from `os_proc_available_memory()`, and the app requests the increased-memory-limit entitlement |
| Reference app | join by scanning an invite QR code; show status, battery policy and what the phone holds | the same |
| Distribution | APK on the release host; Play later | TestFlight; App Store later |
| Updates | store-managed (or APK reinstall); never updated by the self-updater | store-managed |

The mobile apps share the cluster protocol and are subject to the same admission and protocol-version rules as desktop nodes. A phone that falls behind a protocol major is refused with a message, like any other node.

### 11. Uninstall and the host-change ledger

#### 11.1 The ledger

Every change Syntropy makes to the host outside its own directories is written to `<state>/host-changes.json` **before** it is made, marked `pending`, and marked `applied` afterwards. The writers are the installers, `setup`, `service`, `doctor --fix` and opt-in features such as the macOS wired-limit LaunchDaemon.

```rust
pub struct HostChange {
    pub id: u32,
    pub kind: HostChangeKind,
    pub state: ChangeState,              // Pending | Applied | Reverted | Abandoned
    pub at: u64, pub by_version: String,
    pub fingerprint: Option<Blake3>,     // hash of what was written, to detect later user edits
    pub prior: Option<PriorState>,       // e.g. linger was already enabled; previous file content saved at a path
}
pub enum HostChangeKind {
    FileWritten { path: PathBuf },               // units, plists, limits.d, drop-ins, completions
    Symlink { path: PathBuf, target: PathBuf },
    PathEntry { shell_rc: Option<PathBuf>, line: String } | WindowsUserPath { entry: String },
    ServiceRegistered { manager: ServiceManager, name: String },   // systemd, launchd, SCM, Task Scheduler
    LingerEnabled { user: String },
    FirewallRule { platform: FirewallKind, name: String },
    RegistryKey { hive: String, key: String },   // uninstall entry, event-log source
    SystemUser { name: String },                 // created by --system installs
    BootTimeSetting { name: String },            // macOS wired-limit LaunchDaemon
}
```

**Revert rules.**
- Revert only what still matches the recorded fingerprint. A file the user has since edited is left in place, and its path is reported.
- Restore `prior` state where one was recorded: lingering stays on if it was already on.
- Never touch anything that is not in the ledger.

After every change, `setup` regenerates `<state>/revert-host-changes.sh` (Unix) or `.ps1` (Windows) from the ledger. That script can undo everything even if the binary is already deleted.

*Revised 2026-10-22 (PROD-E12-T01):* the ledger is `syntropy_daemon::host_changes`, in the lowest crate that every Rust writer reaches (`cli` depends on `daemon`).
- **Copies.** For each `FileWritten` the ledger keeps the bytes it wrote, `<state>/host-changes/<id>.written`, and the content it replaced, `<id>.prior`, which is what `PriorState::SavedAt` points to. `fingerprint` is the BLAKE3 of the bytes written, and the binary checks against it. The revert scripts can't compute BLAKE3 with standard tools, so they compare the file with the `.written` copy instead: `cmp` on Unix, and `Get-FileHash` on both files on Windows.
- **Copies are private and short-lived.** A written file can be a config or credential file. So `<state>/host-changes/` is owner-only (0700 on Unix), and each copy is created 0600, or with the original's mode where that is stricter. The ledger itself is 0600. Every file the ledger makes is a new file, owner-only before its first byte; whatever was at its path is removed first, never opened. On Windows the copies inherit the state directory's owner-only ACL. Every written file is copied, whatever its size; RFC-0052 has no size limit for the fingerprint-only path. The files Syntropy writes outside its directories are units, plists, drop-ins and completions, a few kilobytes each, so the copies cost about twice those bytes while their changes stay open [estimate]. A copy is deleted when its change is reverted or abandoned, by the binary or by a script, and when closed changes are pruned from the ledger.
- **Markers.** A script can't edit the JSON ledger, so after it undoes a change it writes `<state>/host-changes/<id>.reverted`, and it checks for that first. That makes a second run skip what the first undid. The binary writes the same marker when it closes a change, and it treats an existing marker as reverted.
- **The ledger is trusted only when it is private.** Uninstall can run with more privilege than whoever could edit the state directory, so a revert refuses unless three things are owned by the running user, are not links, and can't be written by a group or other users: the state directory, the copies directory and the ledger. This is the binary's check, and the sh script's first step; the sh script makes the copies directory only after the state directory passes, and only if nothing is in its place. The PowerShell script refuses a state or copies directory that is a junction or symbolic link, but it doesn't check the directory's ACL. A prior copy is restored only from the ledger's own `<id>.prior`; any other recorded path is refused.
- **No link is followed.** The ledger refuses to write a file through a link in its place, and refuses a link in place of the copies directory. A link where Syntropy wrote a file, or a start-up file that is a link, counts as changed, so it is left in place and reported. A file is restored, or a start-up line removed, by writing a new file next to it, which never opens an existing path, and renaming it over the old one. The new file is owner-only until it has the old file's owner (on Unix) and mode, and only then gets its content. The scripts make that file with `mktemp`.
- **One writer at a time: assumed, not enforced (open).** The ledger is read whole when it is opened and rewritten whole on every save. Its writers are separate processes: the installers, `setup`, `service`, `doctor --fix`, opt-in features, and `trust` on a client (ADR-0505 item 6). Nothing serialises them yet. With two open at once, the later save drops the entries the earlier one added, which leaves a change on the host that uninstall does not know to revert. Or one writer removes the other's temporary file, and that save fails. An advisory lock on the state directory, held from opening the ledger to its last save, would close this. It is not part of PROD-E12-T01 as first built, and the review decides whether it is added.
- **Pending changes.** A pending change may or may not have been made. A revert undoes it only if the host shows it as recorded; otherwise it is `Abandoned`.
- **The binary's file-system host** reverts written files, symbolic links and shell start-up lines. The other kinds are reverted through the OS's tools, which the generated script calls.

#### 11.2 `syntropy uninstall`

```text
uninstall [--keep-models] [--keep-config] [--keep-identity] [--force] [--yes] [--package-hook]
  1  list what will be removed and what kept, with sizes (models can be hundreds of GB); confirm
  2  if a cluster member: if this node is the last keyholder of a cluster with other members, refuse:
       name what stops working and offer to promote another eligible member first (RFC-0051 §2.6);
       --force proceeds after the warning. Then Leave (stages are re-placed elsewhere;
       RFC-0001 §5.10), unless --force; Leave works locally with no reachable coordinator
  3  stop and unregister services and tasks (ledger entries)
  4  revert the ledger in reverse order, applying the revert rules
  5  remove versions, modules, symlinks and the binary;
     remove state (identity key, ledgers, API keys) unless --keep-identity;
     configuration unless --keep-config; the store unless --keep-models
  6  print what was removed, what was kept and where, and anything left because the user had edited it
```

- **Package hooks.** `--package-hook` (deb/rpm `prerm`, the MSI's custom action, the cask's `uninstall` stanza) does steps 2–4 only, because the package manager removes its own files. The deb/rpm `purge` step and the cask's `zap` stanza remove data directories.
- **Gone binary.** If the binary was deleted by hand, the generated revert script does steps 3–4.

The claim "uninstall reverts every host change" is tested, not asserted. The install matrix in Testing compares a host snapshot taken before install with one taken after uninstall.

### 12. Size budgets

All figures are **[target]** compressed download sizes, enforced by a release gate (RFC-0062) that fails when an artefact exceeds its budget by more than 10% without an ADR. Each module's budget includes the vendor libraries it carries. RFC-0040 §10.6's size report breaks every module down by component: Syntropy kernels, ggml, the vendor runtime, the vendor BLAS libraries and other redistributables. The gate shows that breakdown, so the BLAS share of each budget is visible on every release candidate.

| Artefact | Budget | Basis |
|---|---|---|
| core, Linux and Windows | ≤ 40 MiB | chosen: an order of magnitude below the surveyed desktop runners' 1.0–1.5 GiB archives, which bundle every GPU runtime (`backends-and-portability.md` §5.1); the core carries CPU kernels for several ISA levels, ggml CPU, HiGHS, iroh and the dashboard, which a surveyed single-binary QUIC runtime at ≈18 MB suggests is the right order (`multi-device-inference.md` §12.1) |
| core, macOS (with Metal providers) | ≤ 50 MiB | as above plus Metal libraries |
| `vulkan` module | ≤ 60 MiB | SPIR-V kernels only; no vendor libraries |
| `cuda13` module | ≤ 600 MiB | includes the CUDA runtime plus cuBLAS and cuBLASLt, which ggml's CUDA provider links and which native providers never call (RFC-0040 §7.4). No figure is claimed for their share until RFC-0040 §10.6's per-component report gives the first M1 measurement (open question 3). If the module exceeds its budget, it splits by GPU architecture family so each machine fetches one family's cubins |
| `rocm` module | ≤ 900 MiB | all-architecture ROCm user-space libraries, including hipBLAS and rocBLAS for ggml's HIP provider; measured in M3 by the same per-component report |
| OCI `cpu` image | ≤ 80 MiB | base plus core |

A machine's download is the core plus its modules: a CPU-only laptop fetches ≤ 40 MiB, and a desktop with one NVIDIA GPU fetches ≤ 640 MiB.

## Performance model and budgets

| Item | Budget [target] | Basis |
|---|---|---|
| `setup` time excluding downloads | ≤ 30 s | driver census ≤ 2 s, module verification ≈ module bytes ÷ hashing rate (BLAKE3 is multi-threaded; a 600 MiB module is well under the budget on any Tier 1 CPU [estimate]), service install ≤ 2 s, startup probe ≤ 15 s (SC-22), margin |
| install-to-first-token, excluding model download, pack and load | `setup` + one confirmation | SC-24 counts steps, not seconds; model time is SC-23's |
| download time | (core + modules) ÷ the user's bandwidth | the size budgets of §12 bound it |
| API unavailability during an update with re-adoption available | ≤ `update.restart_budget` = 10 s [target] | daemon restart plus runner re-adoption (§6.5, §7.3). Remote-ingress sequences keep generating; local-ingress sequences were drained first (step 6). The budget is the planned-restart notice's `back_within_ms`, at most `membership.restart_grace` (RFC-0030 §5.9), so peers start no recovery |
| update downtime, IPC protocol changed | drain plus reload: SC-23's `t_pack + t_load` (packs are usually reused) | runners must restart |
| module re-check before `dlopen` (hash and build ID, RFC-0034 §12.2) | ≤ 0.5 s per runner start for a 600 MiB module [target] | multithreaded BLAKE3 runs at several GB/s on Tier 1 CPUs [estimate], so 600 MiB takes a fraction of a second; RFC-0061 measures it |
| `__selftest` | ≤ 20 s | config parse, a state-migration dry run on a copy, module load and device enumeration, the CPU parity smoke test on a fixture model |
| update download impact on serving | the bulk plane's pacing: the data plane's p99 stays within 1.2 × its idle p99 (ADR-0011 decision 8; RFC-0001 §8.4 rule 6) | background download at low priority |
| `uninstall` excluding store deletion | ≤ 30 s | leave the cluster, stop services, revert the ledger |

Re-adoption is what makes `auto` updates acceptable on a personal machine. Without it, every update would cost a full model reload, which for a 400 GB MoE model is minutes even from a warm pack (SC-23 gives ≈ 23 s for a 42.5 GB shard on a Gen4 drive, scaling with bytes [estimate]).

## Alternatives considered

| Alternative | Why not |
|---|---|
| One fat binary per platform with every GPU runtime | 1–2.5 GiB downloads for everyone, most of it unusable on any given machine (`backends-and-portability.md` §7.2 risk 3); ADR-0507 |
| Separate binaries per backend (`syntropy-cuda`, `syntropy-rocm`) | users must know their GPU stack before installing; mixed-GPU hosts need two binaries; breaks "one binary" (ADR-0015) |
| Modules downloaded automatically by the daemon whenever a GPU appears | new executable code arriving without a user action or an update policy; `setup` or an allowed update must be the only paths |
| Compile kernels on the user's machine (needs a toolchain) | breaks "no prerequisites"; slow; incompatible with notarisation on macOS (`backends-and-portability.md` §5.4) |
| A single Ed25519-signed release manifest, without TUF roles | simpler, but it cannot prevent freeze attacks (a mirror replays an old, validly signed manifest) and has no standard rotation story; RFC-0034 §12 chose TUF roles plus detached release-key signatures for the first-run check, and this RFC implements that (ADR-0508) |
| An in-place binary replacement without version directories | no instant rollback; Windows cannot replace a running executable |
| System service by default everywhere | needs root or admin for every install, runs outside the user's GPU session on macOS, and puts models under a service account the user cannot easily inspect; ADR-0509 |
| Mac App Store and Microsoft Store builds | sandboxing forbids memory wiring, device access and background services |
| A container as the primary Linux install path | GPU passthrough setup, io_uring blocked by default seccomp, discovery needs host networking; kept as an option, not the default |
| Telemetry-driven staged rollouts | needs the telemetry ADR-0510 rules out; channels plus automatic local rollback cover the same risk |

## Testing and validation

| Claim | Test |
|---|---|
| Install, update, rollback and uninstall work, and uninstall reverts every host change | an install matrix on clean VMs or hosted runners: Ubuntu 22.04 and 24.04, Debian 12 and 13, Fedora (current), Arch (script only), macOS 15 and 26, Windows 11. Per target and per install kind: snapshot the host (files under known roots, services and tasks, PATH and shell rc files, firewall rules, `launchctl list`, Task Scheduler, uninstall registry keys, limits files, linger state), install → setup (CPU-only) → `run` a fixture model → update to a test version → force a rollback → update again → uninstall → snapshot again. The snapshots must match, apart from kept directories. Runs on release candidates, and on PRs that touch `packaging/` (path-filtered, RFC-0062); on macOS 15 and 26, an unattended update and a rollback each followed by discovery and a LAN listen raise no Local Network or application-firewall prompt, and on Windows 11 no Defender Firewall prompt |
| Module resolution picks the right modules | table-driven tests over recorded driver censuses: NVIDIA sm_61, sm_75, sm_89 and sm_120 with old and new drivers; AMD on and off the ROCm list; Intel Arc; WSL2; GB10; mixed NVIDIA and Intel; CPU-only |
| The updater resists hostile mirrors | `tough` client tests plus our own against a fake mirror: expired timestamp (freeze), rolled-back snapshot or targets, mix-and-match targets, altered artefact bytes, root rotation across chained root versions, a target older than the installed version without a signed `withdrawn` entry, a module whose hash or build ID differs at `dlopen`, a first-run artefact whose detached signature uses an unknown or revoked key |
| Updates fail safe | fault injection: kill -9 at each step of §7.3 and power-loss simulation (VM reset) between steps 5 and 8; the node must come up on exactly one complete version with consistent state. Injected self-test failure, crash loop and canary failure each trigger rollback |
| Re-adoption keeps models loaded | per-OS soak test through the real service manager (§6.5): a sequence streaming to a remote ingress continues across an update with an unchanged IPC protocol, with no lost or duplicated tokens; runners not adopted exit within `runner.orphan_grace + recovery.drain_deadline`; on installs where re-adoption is unavailable the update drains and reloads instead, and says so |
| Signatures are valid | CI verification of every artefact as listed in §9; Gatekeeper assessment on a clean macOS VM; SmartScreen-relevant signature checks on Windows; no NVIDIA file in any artefact carries a Syntropy signature |
| The CUDA fetch route redistributes nothing | the `cuda13` archive and the script, MSI and app artefacts contain no NVIDIA file (an inventory check against NVIDIA's manifest); on the install matrix's CUDA hosts `setup` shows the EULA once, fetches each pinned archive from NVIDIA, and refuses a tampered archive; with NVIDIA's server unreachable it falls back to the installed toolkit and says so; fetched files are byte-identical to NVIDIA's |
| Headless Macs serve with FileVault on | on R-MACF or R-MBP with FileVault on, macOS 15 and 26: the `UserName` LaunchDaemon starts after unlock, discovers and listens on the LAN with no prompt, creates a Metal device and wires memory with no GUI session, and keeps its grants across an unattended update; the fallback root daemon drops privileges after binding (PROD-E10-T11) |
| The last keyholder cannot leave silently | `uninstall` on the only keyholder of a two-node simulated cluster refuses without `--force`, and proceeds after a promotion |
| Containers behave | the `cpu` image under default Docker seccomp falls back to the `pread` pool and doctor OS-003 reports it; with the documented profile it uses io_uring; a missing memlock ulimit is reported by OS-001 |
| Budgets hold | artefact size gate per release; `setup` and `__selftest` timings recorded in CI |
| Mobile builds stay alive | nightly Android and iOS builds; an emulator or simulator smoke test joining a testkit cluster as a CPU node |

## Rollout / milestones

| Milestone | Lands |
|---|---|
| M0 | developer tarballs per Tier 1 target from CI; `cargo install` for contributors; no installers |
| M1 | `install.sh`, `install.ps1`, the macOS app (zip and Homebrew cask), the per-user MSI with winget; `setup` with the driver census and module resolution; the `cuda13` module with the fetch route (§4.4); user-level services on all three OSes, and the macOS non-root LaunchDaemon (`service install --daemon`) once the headless-Mac spike passes; the TUF client and first-run signature checks against RFC-0034's embedded keys, with the `stable`, `beta` and `nightly` channels that RFC-0062 publishes from M1; self-update with self-test, atomic switch and automatic rollback; the host-change ledger and `uninstall`; installers that verify notarisation and Authenticode signatures. This is the vision's M1 item: one binary with installers for the Tier 1 operating systems. Signed installers, modules downloaded after detection and signed updates land here, ahead of the vision's M4 item and RFC-0001 §16's M4 list. Gatekeeper makes unsigned macOS distribution impractical, and the update client has to exist before the first public release |
| M2 | `.deb`/`.rpm` and repositories (the CUDA module package bundles NVIDIA files only where the redistribution terms allow, and otherwise depends on NVIDIA's repository packages); PackageKit installs of module packages; system services; per-machine MSI; the `.pkg` with the macOS server option; the ggml-only `vulkan` module as a Tier 2 preview; rolling updates across a cluster; the last-keyholder guard in `uninstall` |
| M3 | `rocm`, and the native provider in `vulkan`, with RFC-0040's M3 providers; OCI images (the `cuda13` image bundles NVIDIA files only where the redistribution terms allow, and otherwise fetches them at first start); runner re-adoption across daemon updates (RFC-0001 §3.4, M3); coordinated major-protocol updates |
| M4 | Tier 2 and Tier 3 packaging: GB10 `cuda13` aarch64, `cuda12-legacy`, `sycl`, Android AAR and APK, iOS XCFramework; the tray helper; offline bundles (with NVIDIA files only where the redistribution terms allow); size budgets as release gates; the full install matrix as a release gate (the vision's M4 packaging items) |

## Open questions

1. **Release domain and signing identities (settled).** The derived names follow from the release domain and are generated from one setting; the release needs a Developer ID with a fixed Team ID, an Authenticode identity, and the repository and TUF keys (§9). Setting up each identity is a **Maintainer action**.
2. **CUDA redistribution (answered).** `cuda13` ships only Syntropy code and `setup` fetches NVIDIA's archives from NVIDIA against pinned hashes, showing the EULA once. Channels that would bundle NVIDIA files (deb/rpm, OCI at M3, offline bundles at M4) do so only where NVIDIA's redistribution terms allow it, and otherwise use the fetch path; the user's installed toolkit is the last fallback (§4.4).
3. **cuBLAS in the CUDA module (answered).** The ggml CUDA provider links cuBLAS and cuBLASLt, so the `cuda13` module needs them (now fetched with it, §4.4), and `rocm` carries hipBLAS and rocBLAS. Both budgets in §12 include them. The size of their share comes from the first M1 run of RFC-0040 §10.6's per-component report; no figure is claimed before it. If `cuda13` is then over budget, it splits by architecture family.
4. **Headless Macs (answered).** A non-root LaunchDaemon with `UserName`, shipped once the M1 spike passes with FileVault on; a root LaunchDaemon that drops privileges is the fallback; automatic login is withdrawn as the documented answer (§6.2).
5. **Lingering without root (answered: OQ-PKG-09).** No per-distribution table. `setup` runs `loginctl enable-linger` unprivileged and asks for sudo only on "Access denied": upstream systemd's polkit action `org.freedesktop.login1.set-self-linger` defaults to allowed, so most distributions permit it (§3 step 5). The install matrix records which ones overrode it.
6. **ROCm libraries.** Shipping the all-architecture ROCm user space in the module versus requiring a system ROCm install. The licence part is answered: no obstacle, with RCCL, ROCgdb and the binary-only pieces left out and hipCUB's notice shipped if its headers are compiled in (§4.4). What remains is size and support burden against zero-config, measured in M3. The product default to build to meanwhile: require a system ROCm install, and make `vulkan` the zero-configuration AMD path.
7. **Package installs and modules (answered: OQ-PKG-11).** Yes, with consent, at the first interactive `setup` or `run`, never in `postinst`: through PackageKit when present, otherwise by printing the command (§2.4).
8. **TUF metadata in the pipeline (answered).** RFC-0062 §7.1 now produces and signs the TUF metadata: `snapshot` on each publish, `timestamp` re-signed daily by the scheduled `tuf-timestamp` job, the delegated `nightly` role, and a calendar-scheduled targets re-signing ceremony.
9. **Module directory default (settled: OQ-PKG-12).** The default is the running version's `modules/` directory, with `<store>/modules` an override for development, as RFC-0040's key table and RFC-0054 §5.1 state and PROD-E11-T07 implements.

## Proposed amendments to RFC-0001

- **A1. §3.4 and §3.5, re-adoption across builds.** The `Hello` check currently kills a runner whose build ID differs from the daemon's. For in-place upgrades (§7.3), re-adoption must accept a runner from the previous build when the **IPC protocol version** matches. `Hello` should carry `ipc_protocol` and the observability table digest (RFC-0053 §4.1) beside `build_id`. A fresh spawn still requires an equal build ID. Adoption requires an equal IPC protocol, plus tables that are equal or translatable. This also needs **ADR-0015 decision 6 superseded** by §6.5's bounded orphan rule:
  - no `PR_SET_PDEATHSIG`, `NOTE_EXIT` kill or kill-on-close job for runners;
  - daemon loss detected by IPC end-of-file or the heartbeat;
  - runners live at most `runner.orphan_grace + recovery.drain_deadline` without a daemon;
  - the per-OS service-manager settings of §6.1–§6.3;
  - drain and restart wherever re-adoption is unavailable.
  RFC-0062 §10.1's IPC row changes to match (same build for spawning, same IPC protocol for adoption).
- **A2. §3.1.** Add the hidden `__selftest` mode (§7.3) beside `__runner`.
- **A3. §5.2 `NodeInfo`.** Add `staged_build: Option<BuildId>` for coordinated updates (§7.5), and `install: InstallKind` so the dashboard and doctor can say how a node is updated. The cluster-scope key `update.switch_at` that drives the switch is defined in RFC-0054 §5.2.
- **A4. §4.2 rule 7.** State that GPU backends ship as modules: directories beside the core binary of the same version, located and loaded only by runner and probe processes (§4.3). The core binary therefore carries no vendor runtime on Linux and Windows.
- **A5. ADR-0015 decision 1 ("one binary").** On Windows, the product ships `syntropy.exe` plus the launcher `syntropyw.exe` (§6.3). A console-subsystem binary cannot run as a windowless per-user service, and a GUI-subsystem binary cannot serve as a CLI. The launcher contains no inference or control-plane code.
- **A6. §4.2 rule 7 (with A4).** The `cuda13` module carries no NVIDIA file. `setup` fetches NVIDIA's redistributable archives from NVIDIA, checked against hashes the signed module manifest pins (§4.4), so the rule that the core carries no vendor runtime holds for the module's archive too.
