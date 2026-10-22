# Syntropy design

This directory is Syntropy's complete design and implementation plan. The Cargo workspace and a stub for every crate exist, but the crates are still empty. Everything here is a decision, a specification, or a task waiting to be built. The [repository README](../README.md) is the front page and explains what Syntropy is. This page is the index to the design.

In one paragraph: Syntropy runs one large language model across every machine you own, and across every memory tier inside each one. A single placement decides which layers run on which machine, which weights stay in VRAM, which sit in RAM and are computed there by the CPU, and which are read from NVMe as needed. The goal is the largest model, at the highest tokens/s, that your hardware can physically deliver, with output that matches a reference within published tolerances. It should install with one command, find the other machines on your network by itself, and add a laptop that's away from home with an invite token.

## What's here

| Path | Contents |
|---|---|
| [00-vision-and-scope.md](00-vision-and-scope.md) | The problem, what Syntropy is, users and scenarios, the platform matrix, target models, success criteria SC-1 to SC-32, and what "feature complete" means per milestone |
| [decisions/](decisions/) | Architecture decision records. ADR-0000 holds the product directives D1–D8, ADR-0001 to ADR-0022 the foundation, and domain ADRs are numbered by area |
| [rfcs/](rfcs/) | RFC-0001, the system architecture, and 24 domain RFCs, each specifying one part of the system |
| [implementation/](implementation/README.md) | The tasks, in six epic files, with the roadmap, dependency report, traceability matrix, risk register and the scripts that check them |
| [open-questions.md](open-questions.md) | What is still undecided, each item with the task, spike or measurement that settles it, and the questions already settled |
| [glossary.md](glossary.md) | One term per concept, and the synonyms the plan doesn't use |
| [research/](research/) | The evidence the design rests on |

## At a glance

| | |
|---|---|
| Design | 25 RFCs (RFC-0001 architecture plus 24 domain RFCs) and 96 ADRs |
| Work | 1,382 tasks in six areas, each with testable acceptance criteria and the ADR and RFC sections it implements. By milestone: M0a 163, M0b 125, M1a 463, M1b 89, M2 283, M3 208, M4 51 |
| Order | a dependency graph of 3,399 edges with no cycles; no task depends on a later milestone, and M0a and M0 are each self-contained. The critical path is 24 tasks inside M0a and 39 tasks end to end, counted in tasks because the plan sets no dates |
| Still open | the open questions in [open-questions.md](open-questions.md). Every one names the task, spike or decision that resolves it |

## Read in this order

1. **[00-vision-and-scope.md](00-vision-and-scope.md).** The problem, the scenarios, the platform matrix, target models, the success criteria (all targets derived from the cost model) and the milestone checklists.
2. **[decisions/ADR-0000](decisions/ADR-0000-product-directives.md).** The binding product directives:
   - D1: the tiering model inside a machine.
   - D2: single-user latency first.
   - D3: exact output by default.
   - D4: LAN plus invite links.
   - D5: an original design; plan documents name no third-party inference systems.
   - D6: languages chosen for speed.
   - D7: current hardware features, always with a fallback.
   - D8: open source, built by AI agents, with no team ownership.
3. **[rfcs/RFC-0001](rfcs/RFC-0001-system-architecture.md).** The architecture:
   - the system overview, planes and architectural invariants (§2);
   - the process model and crate layout (§3, §4);
   - the core types every other RFC builds on (§5);
   - the request and load lifecycles (§6, §7);
   - the per-token performance budget (§14);
   - the milestones (§16);
   - the amendment log, where every change to the architecture is recorded (§18).
4. **[glossary.md](glossary.md).** Read it before the domain RFCs: several everyday words have one precise meaning here.
5. **The domain RFCs** for the area you are working in (table below), with the ADRs they cite.
6. **[implementation/](implementation/README.md).** How a task is picked up and what "done" means, then the epics, roadmap, dependencies, traceability and risks.
7. **[open-questions.md](open-questions.md).** What is still open, by area.

## Foundational decisions

| ADR | Decision |
|---|---|
| [0001](decisions/ADR-0001-implementation-languages.md) | Rust for the system layer. Kernels in CUDA C++, HIP, Metal Shading Language and Slang→SPIR-V, plus Rust `std::arch` (with `asm!` for AMX and SME2) on the CPU |
| [0002](decisions/ADR-0002-own-the-runtime-with-kernel-providers.md) | Syntropy owns graph, executor, scheduler, residency and I/O. Kernels come through providers: native ones for hot ops, and ggml for breadth |
| [0003](decisions/ADR-0003-model-graphs-defined-in-syntropy.md) | Model graphs are Rust modules built from shared blocks |
| [0005](decisions/ADR-0005-model-formats.md) | GGUF and safetensors read unchanged, plus a per-device pack with 4 KiB-aligned experts |
| [0006](decisions/ADR-0006-parallelism.md) | Pipeline by layer range across nodes. Tensor and expert parallelism only on links that pass a measured latency gate. Execute in place inside a device |
| [0007](decisions/ADR-0007-syntropy-owns-residency.md) | An explicit tier manager owns every byte; the OS page cache never decides residency |
| [0008](decisions/ADR-0008-placement-solver.md) | One joint placement over devices, tiers, links and experts. From M1 it is solved by a DP over contiguous layer ranges with a greedy in-node cost; the full MILP on HiGHS specifies every term and is built only if it beats the DP by more than 5% in simulation |
| [0009](decisions/ADR-0009-control-plane.md) | An event-sourced control plane with one coordinator and no majority quorum |
| [0010](decisions/ADR-0010-data-plane-topology.md) | Stages connect directly in a ring. Tokens never cross the control plane |
| [0011](decisions/ADR-0011-transport-stack.md) | iroh/QUIC for control and bulk. RDMA, QUIC, shared memory and TCP for the data plane, chosen per link by measurement |
| [0014](decisions/ADR-0014-hardware-feature-policy.md) | Use the current capability (FP4/FP8 tensor cores, io_uring, RDMA over Thunderbolt 5, unified memory), always with a fallback |
| [0017](decisions/ADR-0017-speculative-decoding.md) | Speculative decoding, with exact acceptance, as the way to amortise slow tiers and network hops |
| [0018](decisions/ADR-0018-licence.md) | MIT OR Apache-2.0, contributions under a DCO sign-off, and an enforced dependency licence policy |
| [0019](decisions/ADR-0019-exactness-contract.md) | Exactness means teacher-forced agreement with a reference at the same quantization, within published tolerances |
| [0020](decisions/ADR-0020-platform-support-tiers.md) | Platform tiers. Tier 1: Linux and Windows with NVIDIA, macOS on Apple silicon, and CPU-only on all three |
| [0021](decisions/ADR-0021-development-model-ai-agents-open-source.md) | Open source from the first commit, built by AI agents executing the plan's tasks. Areas are domains, not teams. Gates and an independent review agent review every change, and maintainers approve the sensitive change classes |
| [0022](decisions/ADR-0022-production-clock-and-entropy-outside-core.md) | `core` keeps the clock and entropy traits as plain types. The production sources, `SystemClock` over the suspend-aware OS clock and `OsEntropy` over `getrandom`, live in `io`, where OS calls and `unsafe` are already allowed |

The other foundational ADRs cover MLX (0004), async and threading (0012), the I/O path (0013), the process model (0015) and weight distribution (0016). Domain ADRs are numbered by area: 01xx placement, 02xx tiers, 03xx distributed, 04xx compute, 05xx product and 06xx quality.

## Domain RFCs

| Area | RFC | Title |
|---|---|---|
| Placement | [RFC-0010](rfcs/RFC-0010-resource-graph-and-capability-probing.md) | Resource graph and capability probing |
| | [RFC-0011](rfcs/RFC-0011-joint-placement-solver.md) | Joint placement solver |
| | [RFC-0012](rfcs/RFC-0012-execution-scheduler.md) | Execution scheduler and speculative decoding |
| Tiers | [RFC-0020](rfcs/RFC-0020-tier-manager-and-residency.md) | Tier manager and residency |
| | [RFC-0021](rfcs/RFC-0021-weight-io-path.md) | Weight I/O path |
| | [RFC-0022](rfcs/RFC-0022-kv-cache-prefix-cache-and-sessions.md) | KV cache, prefix cache and sessions |
| | [RFC-0023](rfcs/RFC-0023-moe-expert-cache-prediction-prefetch.md) | MoE expert cache, prediction and prefetch |
| Distributed | [RFC-0030](rfcs/RFC-0030-discovery-membership-invites.md) | Discovery, membership and invite links |
| | [RFC-0031](rfcs/RFC-0031-transport-and-wire-protocol.md) | Transport and wire protocol |
| | [RFC-0032](rfcs/RFC-0032-distributed-execution.md) | Distributed execution |
| | [RFC-0033](rfcs/RFC-0033-fault-tolerance-and-elasticity.md) | Fault tolerance and elasticity |
| | [RFC-0034](rfcs/RFC-0034-security-and-trust.md) | Security and trust |
| Compute | [RFC-0040](rfcs/RFC-0040-kernel-providers-and-backends.md) | Kernel providers and backend integration |
| | [RFC-0041](rfcs/RFC-0041-model-formats-packs-weight-distribution.md) | Model formats, packs and weight distribution |
| | [RFC-0042](rfcs/RFC-0042-quantization-and-numerical-fidelity.md) | Quantization and numerical fidelity |
| | [RFC-0043](rfcs/RFC-0043-model-graph-ir-and-architecture-families.md) | Model graph IR and architecture families |
| Product | [RFC-0050](rfcs/RFC-0050-api-surface.md) | API surface |
| | [RFC-0051](rfcs/RFC-0051-cli-and-dashboard.md) | CLI and dashboard |
| | [RFC-0052](rfcs/RFC-0052-packaging-install-updates.md) | Packaging, install and updates |
| | [RFC-0053](rfcs/RFC-0053-observability.md) | Observability |
| | [RFC-0054](rfcs/RFC-0054-configuration-and-policy.md) | Configuration and policy |
| Quality | [RFC-0060](rfcs/RFC-0060-testing-and-correctness.md) | Testing and correctness |
| | [RFC-0061](rfcs/RFC-0061-benchmarking-and-performance-regression.md) | Benchmarking and performance regression |
| | [RFC-0062](rfcs/RFC-0062-ci-cd-and-release-engineering.md) | CI/CD and release engineering |

## How the plan holds together

- **Evidence.** [research/](research/) is organised by technique: multi-device inference, tiered-memory inference, backends and portability, and a first-principles hardware and performance model with cited bandwidth tables and worked examples. Every number in it is tagged as verified with a source, or as an estimate with its derivation.
- **Coverage.** Every ADR is cited by at least one domain RFC and implemented by at least one task. [implementation/traceability.md](implementation/traceability.md) maps each ADR, success criterion and RFC design section to the tasks that implement, deliver and verify it, and flags every gap.
- **Order.** [implementation/tools/depcheck.py](implementation/tools/README.md) checks the task graph: no cycles, no task depending on a later milestone, and an M0 that needs nothing outside itself. The roadmap, dependency report and traceability matrix are regenerated from the epics, so they can't drift from them.

## How Syntropy is built

Syntropy is open source from its first commit, and this plan is public with it. AI agents build Syntropy by executing the tasks in [implementation/](implementation/README.md): an agent takes one task, implements it on its own branch, and produces one change that meets the task's acceptance criteria and cites the RFC sections and ADRs it implements.

- **Areas are domains, not teams.** Placement, tiers, distributed, compute, product, quality, and ARCH for RFC-0001 divide the code and the plan. They route tasks and review; nobody owns one.
- **Gates review every change.** A change merges when CI's build, lint and tests pass, along with the parity and conformance gates, the performance gates where it touches a measured path, the dependency and licence policy, the D5 naming check and the task-graph check, and when an independent review agent, never the one that wrote the change, has checked it against the task and the cited RFC sections.
- **Maintainers** approve the sensitive classes of change (security-critical paths, CI and release configuration, agent instruction files, new model families, new direct dependencies, and any change a gate cannot check) and carry out the tasks marked **Maintainer action**, which need a person.

[ADR-0021](decisions/ADR-0021-development-model-ai-agents-open-source.md) is the full statement. [AGENTS.md](../AGENTS.md) is the working guide for agents, and [CONTRIBUTING.md](../CONTRIBUTING.md) covers outside contributions.

## Conventions

- **Numbers.** Every figure is a `[target]`, an `[estimate]` with its derivation, or a cited measurement. Nothing here has been measured on Syntropy, because Syntropy doesn't exist yet.
- **Names.** Per D5, plan documents name technologies Syntropy builds on (CUDA, Metal, Vulkan, ggml, io_uring, QUIC, HiGHS and so on), hardware and API shapes, but not the other inference systems studied during research.
- **Support tiers and memory tiers.** "Tier 1/2/3" with a number always means a platform support level. A memory tier is written as a tier kind (`Vram`, `HostPinned`, …) or as a phrase such as "the RAM tier".
- **IDs.** Tasks are `AREA-En-Tnn`, success criteria `SC-n`, open questions `OQ-…`. IDs are stable: a removed item leaves a gap and is never reused.

## How the design changes

- **A decision changes** through a new ADR that supersedes the old one, or a dated revision note inside the existing ADR. A new ADR starts as `Proposed`, binds no task until the maintainers accept it, and then reads `Accepted` with its date.
- **Every change** to a decision or to the architecture is recorded in [RFC-0001 §18](rfcs/RFC-0001-system-architecture.md#18-amendment-log). A domain RFC may add to RFC-0001's types and interfaces, but may not rename, remove or change the meaning of anything RFC-0001 defines without amending it first.
- **A change to the tasks** goes into the epic files, followed by a run of the plan tools so the generated files stay current ([implementation/README.md](implementation/README.md#keeping-traceability-current)).
