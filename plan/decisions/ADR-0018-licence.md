# ADR-0018: Licence: MIT OR Apache-2.0, with an enforced dependency policy
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the architecture area)
- Related: ADR-0001, ADR-0002, ADR-0008, ADR-0011, ADR-0014, ADR-0016, ADR-0017, ADR-0021, ADR-0431, ADR-0507; RFC-0041, RFC-0051, RFC-0052, RFC-0062

## Context
Syntropy ships one system binary, kernel code in several languages and backend modules downloaded per platform (ADR-0001, ADR-0014). It must be embeddable: as mobile nodes through uniffi bindings, in distro packages, and in other people's products. That reach is part of "runs on any supported device" (D7).

The licence has to be compatible with what Syntropy links and ships.

| Component | Licence | How Syntropy uses it |
|---|---|---|
| ggml (tensor library) | MIT | linked; the `KernelProvider` for broad coverage (ADR-0002) |
| HiGHS | MIT | statically linked solver (ADR-0008) |
| iroh | MIT OR Apache-2.0 | transport (ADR-0011) |
| tokio | MIT | async runtime (ADR-0012) |
| BLAKE3 (`blake3` crate) | CC0-1.0 OR Apache-2.0 (and Apache-2.0 WITH LLVM-exception) | chunk hashing (ADR-0016) |
| CUTLASS / CuTe | BSD-3-Clause | header templates compiled into CUDA kernels |
| Slang | Apache-2.0 WITH LLVM-exception | build-time shader compiler |
| uniffi | MPL-2.0 | mobile bindings generator and runtime |
| `webpki-roots` (1.0.10, through iroh's relay client) | CDLA-Permissive-2.0 (the root-certificate data) | TLS trust roots |
| `tokenizers`, `minijinja` | Apache-2.0 | tokenizer and chat-template engine (ADR-0431) |
| rdma-core (libibverbs) | GPL-2.0 OR BSD-2-Clause | loaded at runtime, not bundled (ADR-0011) |
| CUDA runtime and libraries | NVIDIA CUDA EULA. Attachment A lists redistributables (`cudart`, `cublas`, `cublasLt`, `nvrtc`, `nvJitLink`, …) under conditions: object code only, inside an application with "material additional functionality", Linux files unmodified (`plan/research/backends-and-portability.md` §5.3) | shipped only inside the CUDA backend module |
| ROCm | mostly MIT, Apache, BSD or NCSA; hipCUB and RCCL "custom"; some profiler pieces binary-only (§5.3) | shipped only inside the ROCm backend module |
| Metal, Vulkan loader, DirectX | system components (§5.3) | not redistributed |

Model weights carry their own licences: community licences, gated terms and acceptable-use policies. Syntropy moves weights only between nodes in one owner's cluster (ADR-0016, D4). v1 produces no draft heads (ADR-0017). Any the project publishes later would be derived from base models.

Rust convention is dual MIT OR Apache-2.0. MIT is GPLv2-compatible. Apache-2.0 adds an explicit patent grant with a termination clause, which matters for a project that implements published numerical and scheduling techniques.
- *Revised 2026-10-20:* the MIT option keeps Syntropy's own **source** GPLv2-compatible. The shipped **binaries** are not, because they contain Apache-2.0-only code (`tokenizers`, `minijinja`), and Apache-2.0 cannot be combined with GPLv2-only code. That costs Syntropy's users nothing, but no document, notice or release text may claim the binaries are GPLv2-compatible.

The licences in the table above were read on crates.io on 2026-10-20.

## Options considered
1. **MIT only.** Maximally permissive and short, but it has no patent grant. Contributors' patents remain a latent risk for users and embedders. Rejected.
2. **Apache-2.0 only.** It has the patent grant, but it is incompatible with GPLv2-only projects, and it departs from the Rust ecosystem's dual-licence norm that downstream crates and distros expect. Rejected.
3. **MIT OR Apache-2.0.** Each recipient picks one: MIT for GPLv2 compatibility of the source, Apache-2.0 for the patent grant. Matches iroh and most of the Rust ecosystem. Chosen.
4. **GPL-3.0 or AGPL-3.0.** Copyleft prevents closed forks. It blocks embedding in proprietary apps, creates App Store friction for iOS nodes, and deters the vendor and distro packaging that "installs anywhere" needs. Rejected.
5. **Source-available licences** (BUSL, FAIR-style, non-commercial clauses). Not open source, unpackageable by most distros, and contributors avoid them. Rejected.
6. **MPL-2.0.** File-level copyleft, uncommon for Rust system projects, and it adds compliance work for embedders with no benefit to Syntropy. Rejected for Syntropy's own code. It is acceptable for unmodified dependencies.

## Decision
1. **All Syntropy-authored code is licensed `MIT OR Apache-2.0`.**
   - This covers Rust, CUDA C++, HIP, Metal Shading Language, Slang, build scripts and test kits.
   - The repository root has `LICENSE-MIT` and `LICENSE-APACHE`.
   - Every crate sets `license = "MIT OR Apache-2.0"`.
   - Every source file starts with `SPDX-License-Identifier: MIT OR Apache-2.0`.
2. **Contributions are inbound = outbound,** under the same dual licence, certified by a Developer Certificate of Origin `Signed-off-by` line. There is no CLA.
   - MIT and Apache-2.0 both allow sublicensing, so inbound = outbound with a DCO keeps relicensing or other terms open later without a CLA.
   - *Revised 2026-10-20:* every commit, agent-authored ones included, carries a DCO sign-off and passes the same `dco` check (RFC-0062 §12.3). The sign-off is made by the person who submits the change, as on any commit; for agent work that is the maintainer who directs it (ADR-0021 decision 7).
   - The project writes no AI attribution into commits (ADR-0021 §1).
3. **Dependency policy for anything linked into `syntropy` or into a backend module, enforced by `cargo-deny` in CI** (RFC-0062):
   - Allowed: MIT, Apache-2.0 (including WITH LLVM-exception), BSD-2-Clause, BSD-3-Clause, ISC, Zlib, CC0-1.0, BSL-1.0, Unicode-3.0, and CDLA-Permissive-2.0 (added 2026-10-20: `webpki-roots` carries it for its root-certificate data, and iroh's relay client uses that crate).
   - Allowed for unmodified dependencies only: MPL-2.0.
   - Denied: GPL-*, LGPL-*, AGPL-*, SSPL, BUSL, and any licence with field-of-use or non-commercial restrictions.
   - Non-Rust code (vendored C/C++, kernel templates) is listed in a manifest and checked by the same CI job.
   - *Revised 2026-10-20:* the manifest lists vendored third-party files individually, each with its own licence. A vendored tree such as ggml's or HiGHS's gets one entry per third-party file it carries, never a single entry for the project ("ggml: MIT").
4. **System libraries loaded at runtime** (GPU drivers, libibverbs, the Vulkan loader) are dlopened and never bundled. Their licences do not bind Syntropy's code.
5. **Vendor runtimes live only in backend modules,** never in the core binary.
   - *Revised 2026-10-20:* the CUDA module ships only Syntropy code. `setup` fetches NVIDIA's redistributable archives from NVIDIA, checks each against the hash pinned in the signed module manifest, and shows the EULA once, so no channel redistributes NVIDIA files unless it bundles them. A channel that must bundle them (deb/rpm, OCI images, offline bundles) carries only Attachment A redistributables, unmodified, with the EULA notice, and only once the EULA's terms are confirmed for that channel (ADR-0507 rules 2 and 7).
   - The ROCm module ships only components whose licences allow redistribution. hipCUB and RCCL are reviewed before inclusion, and binary-only profiler components are excluded.
   - Each module's manifest states its composite licence: "MIT OR Apache-2.0 for Syntropy code; vendor components under their own terms".
6. **Notices.** A build step generates `THIRD-PARTY-NOTICES` for every artefact from the resolved dependency graph and the non-Rust manifest. A release fails if any component is unlisted.
   - *Revised 2026-10-20:* for each MPL-2.0 component, such as `uniffi`, the notices print where its source code is available, as MPL-2.0 §3.2 requires of binary distribution.
   - Every `NOTICE` file shipped by an Apache-2.0 dependency is carried into the artefact's notices, as Apache-2.0 §4(d) requires.
7. **Model weights.**
   - Syntropy distributes no weights.
   - Before the first fetch of a model, the CLI and dashboard show the licence identifier from the model card, and gated models require the user's own Hub acceptance (RFC-0041, RFC-0051).
   - v1 produces no draft heads (ADR-0017). If the project publishes draft heads later, they go out under their base model's licence terms, not the code licence.

## Consequences
- **Positive.**
  - Syntropy can be embedded in proprietary and open products alike, packaged by distros, and shipped to app stores.
  - Contributors and users get Apache-2.0's patent grant.
  - It is compatible with every core dependency (ggml, HiGHS, iroh, tokio).
- **Negative.**
  - Anyone may build a closed fork, and nothing compels them to contribute back.
  - CUDA redistribution adds per-release EULA compliance on every channel that bundles NVIDIA files. The fetch route in decision 5 avoids redistribution on the default channels, and the user's installed toolkit is the last fallback (ADR-0507 rule 7).
- **Work created.**
  - RFC-0062: `cargo-deny` configuration (with CDLA-Permissive-2.0 allowed), the per-file non-Rust licence manifest check, and DCO enforcement on every commit.
  - RFC-0052: notice generation, including MPL source pointers and propagated Apache `NOTICE` files, and per-module licence manifests.
  - RFC-0041 and RFC-0051: licence display and gated-model acceptance.

## What would make us revisit
- A dependency with a denied licence offers a decisive, measured performance gain with no permissive equivalent. It would then need to be isolated in an optional, separately distributed module, or rejected.
- ggml, HiGHS or iroh relicenses to terms incompatible with this policy. Pin to the last compatible release and plan a replacement.
- NVIDIA's redistributable server becomes unavailable or gated, or the EULA's terms rule out a bundling channel. Then that channel requires the user's installed CUDA toolkit (ADR-0507 rule 7).
- A patent assertion against a technique Syntropy implements shows the dual-licence patent grant is insufficient.
- A dual-licensed edition is planned. Then a CLA or copyright assignment is needed before contributions to it are accepted (decision 2).

## Revisions
- **Revised 2026-10-20:**
  - Decision 3 allows CDLA-Permissive-2.0, which a TLS root-certificate crate on the transport path carries, and lists vendored third-party files individually, so the notices can name every file's licence.
  - Decision 6 prints MPL source pointers and propagates Apache `NOTICE` files, which MPL-2.0 §3.2 and Apache-2.0 §4(d) require of binary distribution.
  - The Context corrects the claim that the binaries are GPLv2-compatible: they contain Apache-2.0-only code, so only the source is.
  - Decision 2: one `dco` check applies to every commit, agent-authored ones included.
  - Decision 5: the CUDA module ships only Syntropy code and fetches NVIDIA's redistributables at setup, matching ADR-0507.
  - Decision 7 and the Context: v1 produces no draft heads (ADR-0017), so the draft-head licence rule applies only if heads are published later.
