# Implementation roadmap: M0a to M4

This file turns the six epic files into a milestone plan. Milestone goals and exit criteria are quoted from RFC-0001 §16, and SC IDs refer to `00-vision-and-scope.md` §7. Everything between the generated markers comes from `tools/roadmap.py`, which reads the task graph through `tools/depcheck.py` and the open questions from `plan/open-questions.md`. Re-run it after any change to an epic file, RFC-0001 §16 or the open-questions register:

```sh
python3 plan/implementation/tools/roadmap.py           # rewrite the generated sections
python3 plan/implementation/tools/roadmap.py --check   # exit 1 if this file is stale
```

**No durations.** The epics give no task sizes and no estimates. Every path length in this file is a count of tasks, and no date in it is a commitment or an estimate. A critical path of 20 tasks says that 20 tasks must finish one after another; how long that takes depends on task sizes nobody has estimated yet.

How to read the figures:
- **Critical path inside a milestone** is the longest chain of tasks in that milestone, each depending on the one before, assuming every earlier milestone is complete.
- **Longest chain from project start** is the longest dependency chain, across all milestones, that ends at a task in the milestone.
- **Zero-slack tasks** lie on some chain of the critical length. Delaying any of them delays the milestone.
- **Step** is a task's earliest position inside its milestone: 1 + the length of its longest chain of prerequisites in the same milestone. Tasks at the same step with no dependency between them can run in parallel.

## Start here

Begin with the workspace and CI skeleton, the merge gates and the core types every area compiles against. The list is generated: it is one valid order of M0, M0a first, that keeps unblocking the largest amount of later work. With several agents working in parallel, each takes the next task from the top whose dependencies are done, and the zero-slack tasks of M0a go first. Tasks marked **Maintainer action** are queued for the maintainers (ADR-0021).

<!-- BEGIN GENERATED: roadmap-start -->
309 M0 tasks in a dependency order that unblocks the most first. M0a's tasks come before M0b's. Each task comes after everything it depends on; among the tasks that are ready, the one with the most transitive dependents across M0a–M4 goes first, then the one with the most direct dependents. "Step" is the earliest position a task can take in its milestone (1 + its longest chain of prerequisites in the same milestone). 8 M0 tasks have no dependencies: DIST-E17-T13, COMP-E13-T09, COMP-E16-T15, PROD-E1-T06, PROD-E10-T02, QUAL-E1-T01, QUAL-E1-T30, QUAL-E5-T01. The first 30 follow.

| # | Task | Milestone | What | Depends on | Step | Direct dependents | Unblocks (transitive, M0a–M4) |
|---|---|---|---|---|---|---|---|
| 1 | QUAL-E1-T01 | M0a | Workspace skeleton: one Cargo workspace with every crate of RFC-0001 §4.1 as a stub (`lib.rs` with… | none | 1 | 28 | 1411 |
| 2 | QUAL-E1-T02 | M0a | `xtask` framework: `cargo xtask ci <job>`, `cargo xtask test <suite>` and a registry of job and… | QUAL-E1-T01 | 2 | 10 | 1324 |
| 3 | QUAL-E1-T03 | M0a | `cargo xtask check-deps`: RFC-0001 §4.2 rules 1–9 and the release-feature rule over… | QUAL-E1-T02 | 3 | 10 | 1258 |
| 4 | QUAL-E1-T05 | M0a | Contribution checks: `dco` (every commit signed off, matching author), `pr-title` (conventional… | QUAL-E1-T02 | 3 | 4 | 1158 |
| 5 | QUAL-E1-T06 | M0a | `pr.yml` skeleton: `changes`, `lint`, `build-<os>` (the `pr` profile, plus a Linux `test-release`… | QUAL-E1-T02, QUAL-E1-T05 | 4 | 23 | 1157 |
| 6 | COMP-E1-T01 | M0a | `syntropy-kernels` crate: the `KernelProvider` trait with RFC-0001 §5.7's methods and RFC-0040… | QUAL-E1-T01, QUAL-E1-T03 | 4 | 15 | 1106 |
| 7 | COMP-E13-T01 | M0a | Types in `core`: the extended `DType`, `QuantScheme`, `ScaleGranularity`, `CtScheme`, `ActQuant`… | QUAL-E1-T01 | 2 | 6 | 1082 |
| 8 | COMP-E1-T02 | M0a | Op registry: the closed `OpKind` list of §3.1 (including `AttnPartial`, `AttnMerge`… | COMP-E1-T01, COMP-E13-T01 | 5 | 16 | 1072 |
| 9 | QUAL-E3-T25 | M0a | `CountingAlloc` and `alloc_count_this_thread` in `testkit::guards` (split from QUAL-E3-T11, so M0a… | QUAL-E1-T01 | 2 | 10 | 991 |
| 10 | QUAL-E12-T01 | M0a | `syntropy-bench` schemas: `BenchKey` (including `expert_cache: CacheStart`), `RigFingerprint`… | QUAL-E1-T01 | 2 | 4 | 989 |
| 11 | QUAL-E15-T01 | M0a | Inventory schema and fingerprints: the `bench/lab/rigs.toml` record (CPU and microcode, DIMMs, GPU… | QUAL-E12-T01 | 3 | 6 | 987 |
| 12 | QUAL-E15-T02 | M0a | R-ADA online (ADR-0610): an Ada-class 24 GB GPU (`sm_89`), an AVX2-only desktop CPU, 2 × 64 GB… | QUAL-E15-T01 | 4 | 22 | 986 |
| 13 | DIST-E1-T12 | M0a | The `spsc` crate (OQ-IO-29): the layer-1 crate that holds the lock-free SPSC ring `io`, `net` and… | QUAL-E1-T03 | 4 | 1 | 973 |
| 14 | DIST-E1-T01 | M0a | Lock-free SPSC ring library in `spsc` (OQ-IO-29): power-of-two capacity, cache-line-padded indices… | DIST-E1-T12, QUAL-E1-T01, QUAL-E3-T25 | 5 | 6 | 972 |
| 15 | QUAL-E3-T01 | M0a | `syntropy-core` determinism types: `MonoTime`, `Clock`, `Entropy`, `ProtocolCore`, `Outbox`… | QUAL-E1-T01, QUAL-E1-T03 | 4 | 12 | 949 |
| 16 | QUAL-E15-T17 | M0a | The rest of the M0 rigs (ADR-0610 decision 1): R-BW (RTX 5090, AVX-512 dual-CCD CPU, 128 GB, Gen5… | QUAL-E15-T01–T02 | 5 | 18 | 937 |
| 17 | PLC-E1-T01 | M0a | Core graph types: `DeviceId`/`TierId`/`LinkId` ordinal rules, `Measured<T>` with… | QUAL-E1-T01 | 2 | 10 | 912 |
| 18 | QUAL-E3-T02 | M0a | `testkit` deterministic clock and entropy: `SimClock` (advanced only by its owner), `SimEntropy`… | QUAL-E3-T01 | 5 | 8 | 898 |
| 19 | COMP-E9-T04 | M0a | Provider-library ABI v1: `kernels::abi::export!` thunks (expanded in the module's `kernels-*`… | COMP-E1-T01, QUAL-E1-T06 | 5 | 6 | 893 |
| 20 | QUAL-E16-T24 | M0a | Build images and toolchain pins (split from QUAL-E16-T01 so kernel build scripts depend on the… | QUAL-E1-T06 | 5 | 6 | 883 |
| 21 | TIER-E1-T13 | M0a | `LedgerGrant` in `core` (OQ-IO-27): the per-tier granted bytes that RFC-0020 §3 and §12 check an… | QUAL-E1-T01 | 2 | 2 | 882 |
| 22 | COMP-E4-T01 | M0a | Binding and enumeration: dynamic loading of the driver through `cudarc`, driver ≥ 580 check… | COMP-E1-T01, COMP-E9-T04, QUAL-E15-T02, QUAL-E15-T17 | 6 | 7 | 875 |
| 23 | COMP-E10-T01 | M0a | `RangeSource` trait with local-file and HTTP range implementations: coalescing, the header-read cap… | QUAL-E1-T01 | 2 | 3 | 870 |
| 24 | QUAL-E1-T04 | M0a | Licence policy: `deny.toml` (allowed licences including `CDLA-Permissive-2.0`, per-crate MPL-2.0… | QUAL-E1-T02 | 3 | 8 | 869 |
| 25 | COMP-E15-T01 | M0a | IR types (`ModelGraph`, `LayerProgram`, `Program`, `Node`, `WeightRef`, `GroupSel`, `TensorRole`… | COMP-E1-T02, QUAL-E1-T03 | 6 | 5 | 864 |
| 26 | COMP-E10-T02 | M0a | GGUF reader: v2 and v3, little-endian only; limits on tensor count, KV pairs, string, array and… | COMP-E10-T01 | 3 | 7 | 856 |
| 27 | COMP-E16-T01 | M0a | `Family` trait and static registry; `MetaView` over GGUF metadata and… | COMP-E10-T02, COMP-E15-T01 | 7 | 11 | 849 |
| 28 | COMP-E7-T01 | M0a | Vendoring and the core's static build: ggml's tensor library at the `third_party/ggml.lock` commit… | QUAL-E1-T01, QUAL-E16-T24 | 6 | 8 | 848 |
| 29 | DIST-E1-T02 | M0a | Shared-memory region: header (magic, build ID, heartbeat, step counter, `RunnerState` discriminant)… | DIST-E1-T01 | 6 | 4 | 845 |
| 30 | DIST-E1-T03 | M0a | IPC command channel: length-prefixed postcard frames over a UDS or named pipe, request IDs, `Hello`… | TIER-E1-T13, DIST-E1-T02 | 7 | 19 | 842 |
<!-- END GENERATED -->

## Overview

### Tasks per milestone and area

<!-- BEGIN GENERATED: roadmap-counts -->
Generated by `tools/roadmap.py` from 1428 tasks and 3811 dependency edges. Every task sits in exactly one milestone.

| Milestone | PLC | TIER | DIST | COMP | PROD | QUAL | Total |
|---|---|---|---|---|---|---|---|
| M0a | 22 | 8 | 6 | 60 | 17 | 66 | 179 |
| M0b | 16 | 12 | 2 | 33 | 17 | 50 | 130 |
| M1a | 75 | 100 | 11 | 102 | 96 | 94 | 478 |
| M1b | 11 | 19 | 2 | 23 | 16 | 21 | 92 |
| M2 | 36 | 12 | 123 | 27 | 34 | 55 | 287 |
| M3 | 24 | 23 | 64 | 46 | 27 | 27 | 211 |
| M4 | 3 | 4 | 13 | 10 | 11 | 10 | 51 |
| Total | 187 | 178 | 221 | 301 | 218 | 323 | 1428 |
<!-- END GENERATED -->

### Critical paths

<!-- BEGIN GENERATED: roadmap-summary -->
| Milestone | Tasks | Critical path inside (tasks) | Longest chain from project start (tasks) | Zero-slack tasks | Ready at step 1 |
|---|---|---|---|---|---|
| M0a | 179 | 26 | 26 | 27 | 3 |
| M0b | 130 | 7 | 28 | 7 | 67 |
| M1a | 478 | 19 | 42 | 21 | 108 |
| M1b | 92 | 7 | 39 | 7 | 24 |
| M2 | 287 | 16 | 45 | 23 | 48 |
| M3 | 211 | 9 | 47 | 18 | 64 |
| M4 | 51 | 6 | 49 | 6 | 36 |

Path lengths count tasks. The epics give no durations, so none of these figures is a time estimate.
<!-- END GENERATED -->

## Rules for every milestone

How milestones exit, how spikes gate the work they inform, and where performance targets live.

<!-- BEGIN GENERATED: roadmap-rules -->
Quoted from RFC-0001 §16, "Rules for every milestone":

> - **Re-baseline at each exit.** At every exit the task sizes and the agent-model schedule (`implementation/tools/sizing.py`, `schedsim.py`) are re-run against measured burn-up, and the scope-cut order below is re-checked.
> - **Functional close.** A task closes on its functional criteria: tests, conformance and parity. An efficiency, tok/s or SC target sits in a separate performance task that no functional task depends on and that feeds the milestone exit. No row carries a criterion that applies from a later milestone; that criterion is its own task in the later milestone (`depcheck.py` reports `AC-LATER`).
> - **Spikes gate what they inform.** Every spike is timeboxed at 5–10 days of elapsed time [target], names the default that applies if the timebox runs out, and has a dependency edge from every task its answer can change (`SPIKE-NOBOX`, `SPIKE-LEAF`). Spikes that can reverse a design run at the end of M0b or the start of M1a, and a re-plan of the tasks they touch follows them.
> - **Rigs are dependencies.** A task whose criteria need a lab rig depends on the task that brings that rig online, in the same milestone or an earlier one (`RIG-PATH`, `RIG-LATER`).
> - **The M0 spike sprint.** Spikes S-01 to S-08, and S-09's solver-scale run, are throwaway standalone harnesses, and the product tasks re-measure what they find. Each needs only its rigs and a few foundation tasks (the corpus sources, the crate types it measures), so the sprint starts as soon as those are in place and runs in parallel with M0a. The harnesses sit in M0b because the skeleton does not depend on them. The exception is S-08, the CPU kernel-language comparison, which is in M0a because M0a's CPU GEMV kernels follow its answer. M1a's NVMe expert tier waits for S-01's routing-trace hit curves.
> - **The thesis checkpoint.** The single-device baseline comparison on R-ADA under ADR-0612 gets an early read at the M0b exit (S-10) and gates the M1a exit; a failure re-plans M2 before M2 starts. From the first M1a release candidate it is a release gate: a release candidate ships only when the comparison has run on it under ADR-0612's protocol and passed, or a maintainer has waived it in writing with the measurement attached (RFC-0061 §7.7).
> - **Provisional MoE ceilings.** Until the fixed-order F32 calibration reference on a GPU lands, every exit that reports MoE tolerance ceilings (M0b, M1a, M1b) marks them `provisional`.
> - **Quality gates are never cut.** No milestone exits by removing or relaxing a gate. M4's written waiver is the only exception, and it waives an SC target, not a gate.
<!-- END GENERATED -->

## Scope-cut order

If measured burn-up shows the plan cannot reach 1.0 as scoped, scope is cut in this order and no other. The order is re-checked at every milestone exit.

<!-- BEGIN GENERATED: roadmap-cuts -->
Quoted from RFC-0001 §16, "Scope-cut order":

> If measured burn-up at a milestone exit shows that 1.0 cannot be reached as scoped, scope is cut in this order, one step at a time:
> 1. Tier 3 mobile builds (ADR-0020).
> 2. The native HIP and Vulkan providers. The ggml provider covers those backends instead, starting from M2's ggml `vulkan` module (RFC-0040).
> 3. Cross-node TP and EP (ADR-0006).
> 4. Throughput mode (RFC-0011, RFC-0012).
> 5. Asynchronous speculation (ADR-0017).
> 6. RoCE (RFC-0031).
>
> The order is re-checked at every exit against measured burn-up. A cut moves its work after 1.0 and is recorded in §18. Quality gates are never on the list: parity, conformance, the security checks and the review agent stay whatever else is cut.
<!-- END GENERATED -->

## Recurring operations

Standing work that no milestone closes. It is tracked apart from milestone burn-up, so milestone progress is not overstated.

<!-- BEGIN GENERATED: roadmap-recurring -->
Quoted from RFC-0001 §16, "Recurring operations":

> Five streams of standing work recur for as long as Syntropy ships, and no milestone closes them:
> - ggml pin bumps: rebase, regenerated bindings and full op conformance (ADR-0403);
> - lab operations: OS images and driver bumps, availability and continuity (RFC-0061);
> - releases: release candidates, soaks and promotion between channels (RFC-0062);
> - dependency and image bumps: the weekly dependency-update batches and the build images (RFC-0062);
> - new model families: implementation and certification of each (ADR-0003).
>
> Each occurrence is a recurring agent task that a scheduled job opens from the stream's template in `governance/recurring.toml`, and agents execute it through the normal gates and review (RFC-0062 §14). Physical lab upkeep and release approvals are maintainer actions. Each stream has a reserve of rig hours in the rigs' nightly and weekly windows, set at each milestone exit from the previous milestone's measured rig use. The work is tracked apart from milestone burn-up, so that milestone progress is not overstated.
<!-- END GENERATED -->

## M0a: walking skeleton

### Goal and exit criteria

<!-- BEGIN GENERATED: roadmap-m0a-goal -->
Quoted from RFC-0001 §16, "M0a: walking skeleton":

> One dense family end to end on one Linux machine, CPU and CUDA, behind every merge gate. It is the first integration point, and nothing in it is built for scale.
> - The workspace builds on every Tier 1 OS, and CI enforces the §4.2 dependency rules (RFC-0062).
> - The merge gates of ADR-0021 decision 5 run from the first commit: build, lint and tests; the DCO, licence and dependency checks; the workflow security lint; the independent review-agent job; and the sandbox in which agent runs read untrusted content. The repository's security baseline, the private vulnerability-reporting channel and the agent trust tiers are in place from the first commit too (RFC-0062 §12.5, §12.6, §13; ADR-0021; ADR-0624).
> - GGUF reader, census and manifest for `llama` at Q4_K and Q8_0. Such a model runs end to end through the OpenAI Chat Completions API with streaming, through the daemon and a real runner, on the CPU provider and on CUDA (RFC-0041, RFC-0043, RFC-0050, RFC-0051).
> - Execution is eager; this exit needs no captured graphs (RFC-0012).
> - Op selection takes native CPU and CUDA ops from a static allow-list for the `llama` op set and falls back to the ggml provider. Certificate-gated selection arrives in M0b (RFC-0040).
> - Conformance is a gate. Every op `llama` uses passes on the CPU provider in hosted CI and on CUDA on R-ADA, against the ggml provider and the F64 reference, as a required check on every change that touches a provider (RFC-0060 §4).
> - The parity harness is a gate. Teacher-forced parity of `llama` against the `testkit` reference executor is a required check on every change to a kernel, a provider, the graph or a family module, on the CPU in hosted CI and on R-ADA. Until M0b's calibration it uses the tolerance table's provisional ceilings (RFC-0060 §5, RFC-0042).
> - The model-definition oracle gates `FamilyCert`. Golden logits and hidden-state norms for `llama` come from the publisher's reference implementation, run in F32 on the same dequantised weights, and are stored offline. `llama`'s `FamilyCert` is issued only when they pass, and they re-run on every change to the family module or a shared block (RFC-0042 §7, RFC-0060 §7, ADR-0019).
> - Hardware jobs on R-ADA run for changes on this repository's branches. Fork pull requests get the hosted checks only, until M1b (RFC-0062 §3).
> - Exit: `llama` at Q4_K and Q8_0 runs end to end through the OpenAI streaming endpoint on the CPU provider and on R-ADA, eagerly, with parity `smoke` passing on both, every conformance and parity gate required and green at the exit commit, and `llama`'s `FamilyCert` issued against the publisher's reference. No SC is required at this exit.

Success criteria first required at this exit: none.
<!-- END GENERATED -->

### Plan

<!-- BEGIN GENERATED: roadmap-m0a -->
**Tasks:** 179 (PLC 22, TIER 8, DIST 6, COMP 60, PROD 17, QUAL 66).

**Critical path inside M0a:** 26 tasks, assuming every earlier milestone is done. 27 M0a tasks lie on a chain of that length (no slack); the chain below is one of them.

| Step | Task | What |
|---|---|---|
| 1 | QUAL-E1-T01 | Workspace skeleton: one Cargo workspace with every crate of RFC-0001 §4.1 as a stub (`lib.rs` with the SPDX header and… |
| 2 | QUAL-E12-T01 | `syntropy-bench` schemas: `BenchKey` (including `expert_cache: CacheStart`), `RigFingerprint`, `BenchResult`… |
| 3 | QUAL-E15-T01 | Inventory schema and fingerprints: the `bench/lab/rigs.toml` record (CPU and microcode, DIMMs, GPU, VBIOS and driver… |
| 4 | QUAL-E15-T02 | R-ADA online (ADR-0610): an Ada-class 24 GB GPU (`sm_89`), an AVX2-only desktop CPU, 2 × 64 GB DDR5-6000 in 2 channels… |
| 5 | QUAL-E12-T04 | Lab agent: §3.5 cleanliness checks (power source, governor or power plan, foreign processes, paged-out memory… |
| 6 | QUAL-E2-T01 | Lab dispatcher, core, built and tested against GitHub's current rules: `lab.yml` on `main` plus the organisation-owned… |
| 7 | QUAL-E2-T04 | `lab-trusted` environment: the gated-model Hub credential (read-only) and the results-store write credential… |
| 8 | QUAL-E2-T11 | Agent sandbox for untrusted content: agent runs that process fork PRs, issues or contributed files use a sandbox… |
| 9 | QUAL-E15-T16 | Untimed agent GPU pool, first devices (ADR-0610 decision 4): an NVIDIA GeForce host that boots either Linux or Windows… |
| 10 | PLC-E1-T04 | GPU inventory from management APIs (E2) for NVIDIA and Apple: `dlopen` of NVML (UUID, PCI v3, memory v2, BAR1, link… |
| 11 | PLC-E1-T07 | Probe-runner enumeration (E5) and merge: spawn `__runner --probe`, `KernelProvider::enumerate` and `open` per provider… |
| 12 | PLC-E1-T09 | Single-node `ResourceGraph` built by the daemon from E1–E5 output, with `GraphVersion` and `assumptions` for unmeasured… |
| 13 | PLC-E4-T02 | M0 placement rule for one device: all dense groups in the fastest tier that fits after KV, scratch and reserves, else… |
| 14 | PLC-E7-T02 | Step-program lowering for dense families: segments by compute site, `HandOff` insertion, `Decode` and `Prefill`… |
| 15 | PLC-E7-T03 | Primary dispatcher step loop: control, sequence and frame ingest; `sched.next_step()`; `KvStore.append`; double-parity… |
| 16 | PLC-E7-T04 | Graph capture policy: capture every bucket (decode, verify, the prefill ladder, `PrefillStream`) in `Warming` before… |
| 17 | PLC-E9-T02 | Tail sampling pipeline: `Logits` on flagged rows, `LogitsProcess` (temperature, logit bias; penalties from two… |
| 18 | PLC-E9-T16 | Parity taps, prefill and decode: `SequenceOpen::parity` with `ForceMode::Prefill` and `Decode` (teacher tokens replace… |
| 19 | PROD-E4-T01 | Debug surface: `GET /v1/syntropy/debug/state` (cluster state, residency summaries, plans with explanations and… |
| 20 | QUAL-E12-T02 | Benchmark driver: starts or attaches to a `release`-profile daemon; applies scenario configuration through the native… |
| 21 | QUAL-E12-T03 | Block runner and statistics: repeated ABBA designs (PR† ABBA × 3 with k = 2; release and baseline ABBA × 4 with k = 3… |
| 22 | QUAL-E2-T02 | Hardware job template and `hw-cuda-ada` dispatched through `lab.yml`: downloads the run's artefacts (no compilation on… |
| 23 | QUAL-E4-T06 | CPU and CUDA conformance in CI: CPU provider cases in `hosted-test-*` on all three OSes; CUDA cases for `sm_89` in… |
| 24 | QUAL-E6-T04 | `ConformanceManifest`, minimal at M0: types in `core` (`FamilyCert` and `OpCert` at M0, with `OpCert::rig` recording… |
| 25 | COMP-E16-T02 | `FamilyCert` gating: `graph`'s build script computes each family's code hash over its module and shared blocks… |
| 26 | COMP-E16-T14 | The oracle as a gate (RFC-0043 §9.2): `RefExecutor` in `F32Fixed`, with declared activation quantisation off, against… |

Zero-slack tasks by area: PLC 9, COMP 2, PROD 1, QUAL 15.

**Longest chain from project start to an M0a task:** 26 tasks: QUAL-E1-T01 (M0a) → QUAL-E12-T01 → QUAL-E15-T01 → QUAL-E15-T02 → QUAL-E12-T04 → QUAL-E2-T01 → QUAL-E2-T04 → QUAL-E2-T11 → QUAL-E15-T16 → PLC-E1-T04 → PLC-E1-T07 → PLC-E1-T09 → PLC-E4-T02 → PLC-E7-T02 → PLC-E7-T03 → PLC-E7-T04 → PLC-E9-T02 → PLC-E9-T16 → PROD-E4-T01 → QUAL-E12-T02 → QUAL-E12-T03 → QUAL-E2-T02 → QUAL-E4-T06 → QUAL-E6-T04 → COMP-E16-T02 → COMP-E16-T14.

**Parallel work.** Tasks per step, where a task's step is 1 + its longest chain of prerequisites inside the milestone: 1: 3, 2: 10, 3: 11, 4: 11, 5: 17, 6: 15, 7: 14, 8: 15, 9: 16, 10: 14, 11: 7, 12: 9, 13: 8, 14: 5, 15: 2, 16: 4, 17: 3, 18: 2, 19: 1, 20: 3, 21: 2, 22: 2, 23: 2, 24: 1, 25: 1, 26: 1. 3 tasks can start as soon as M0a opens.

Work streams are the epics with tasks in this milestone. "First step" and "last step" are the steps of the stream's earliest and latest tasks. Streams whose step ranges overlap, and that do not wait on each other, can run in parallel.

| Stream | Epic | Tasks | First step | Last step | Waits on (M0a streams) |
|---|---|---|---|---|---|
| PLC-E1 | Resource graph types and hardware enumeration | 7 | 2 | 12 | DIST-E1, COMP-E1, COMP-E3, COMP-E4, QUAL-E1, QUAL-E15 |
| PLC-E2 | Startup microbenchmark suite | 1 | 12 | 12 | PLC-E1, QUAL-E3 |
| PLC-E4 | Solver foundations: inputs, preprocessing, filter, evaluator and DP | 3 | 3 | 13 | PLC-E1, COMP-E10, PROD-E8 |
| PLC-E7 | Executor: step programs, dispatchers and graph capture | 5 | 6 | 16 | PLC-E4, TIER-E6, DIST-E1, COMP-E1, COMP-E4, COMP-E15, QUAL-E3, QUAL-E8 |
| PLC-E8 | Step scheduling: prefill, decode, throughput mode and admission | 1 | 16 | 16 | PLC-E7, PROD-E2 |
| PLC-E9 | Sampling and speculative decoding | 5 | 2 | 18 | PLC-E7, COMP-E1, COMP-E3, COMP-E4, COMP-E17, PROD-E2, QUAL-E1, QUAL-E3, QUAL-E5 |
| TIER-E1 | Tier manager core: arenas, handles and residency | 5 | 2 | 12 | PLC-E4, DIST-E1, COMP-E1, COMP-E3, COMP-E4, COMP-E10, QUAL-E1 |
| TIER-E6 | KV store, paging and tiering | 2 | 8 | 12 | TIER-E1, COMP-E16, QUAL-E3 |
| TIER-E8 | Expert cache: slots, policy, slot table and profile | 1 | 8 | 8 | PLC-E4, COMP-E1, QUAL-E1 |
| DIST-E1 | Process model: rings, IPC, runner spawn, lifecycle and supervision | 6 | 4 | 8 | TIER-E1, QUAL-E1, QUAL-E3 |
| COMP-E1 | Kernel provider core | 5 | 4 | 8 | COMP-E13, QUAL-E1 |
| COMP-E2 | Op selection, certification, coverage and kernel targets | 1 | 7 | 7 | COMP-E1, COMP-E13 |
| COMP-E3 | CPU provider | 9 | 6 | 11 | PLC-E1, PLC-E7, PLC-E9, COMP-E1, COMP-E7, COMP-E13, QUAL-E3, QUAL-E15 |
| COMP-E4 | CUDA provider: runtime, memory, doorbells and capture | 5 | 6 | 9 | COMP-E1, COMP-E9, QUAL-E15 |
| COMP-E5 | CUDA provider: kernels | 6 | 9 | 14 | PLC-E9, TIER-E8, COMP-E1, COMP-E4, COMP-E13, QUAL-E7, QUAL-E15 |
| COMP-E7 | ggml provider | 4 | 6 | 12 | TIER-E1, COMP-E1, QUAL-E1, QUAL-E3, QUAL-E15, QUAL-E16 |
| COMP-E9 | Kernel builds, provider libraries and the module ABI | 1 | 5 | 5 | COMP-E1, QUAL-E1 |
| COMP-E10 | Model readers, identities and the census | 6 | 2 | 9 | COMP-E16, QUAL-E1 |
| COMP-E11 | Model store and packs | 1 | 9 | 9 | COMP-E10 |
| COMP-E13 | Numerics: types, dequantisation, widening and accumulation rules | 4 | 2 | 8 | COMP-E1, COMP-E2, COMP-E7, QUAL-E1 |
| COMP-E14 | The exactness contract: tolerances, wire, canary, knobs and requantization | 4 | 3 | 14 | COMP-E4, COMP-E13, COMP-E15, QUAL-E4 |
| COMP-E15 | Graph IR, blocks, segments and tensor-parallel sharding | 3 | 6 | 9 | COMP-E1, QUAL-E1, QUAL-E3, QUAL-E4 |
| COMP-E16 | Family modules, classification and certification | 6 | 7 | 26 | COMP-E10, COMP-E13, COMP-E15, QUAL-E4, QUAL-E5, QUAL-E6, QUAL-E7 |
| COMP-E17 | Tokenizers and chat formats | 5 | 4 | 10 | COMP-E10, COMP-E16, QUAL-E1 |
| PROD-E1 | Configuration and policy | 3 | 2 | 4 | QUAL-E1 |
| PROD-E2 | API core, vendor shapes, routing and streaming | 4 | 6 | 12 | DIST-E1, COMP-E17, PROD-E1, PROD-E3, QUAL-E3 |
| PROD-E3 | Tool calling, structured output and reasoning | 1 | 11 | 11 | COMP-E17 |
| PROD-E4 | Native API, fidelity labels, debug surface, sessions and embeddings | 2 | 10 | 19 | PLC-E9, PROD-E1, PROD-E2, PROD-E8, QUAL-E5 |
| PROD-E6 | CLI and the model picker | 4 | 9 | 13 | PLC-E2, DIST-E1, PROD-E1, PROD-E2 |
| PROD-E8 | Observability core: metrics, step records, tracing and logs | 3 | 2 | 17 | PLC-E7, COMP-E3, COMP-E4, QUAL-E1 |
| QUAL-E1 | Workspace, CI skeleton and policy checks | 18 | 1 | 23 | QUAL-E2, QUAL-E14 |
| QUAL-E2 | Runner fleet and hardware-job security | 5 | 6 | 22 | QUAL-E1, QUAL-E4, QUAL-E5, QUAL-E12, QUAL-E15 |
| QUAL-E3 | Determinism foundations, testkit core and the test pyramid's base | 6 | 2 | 11 | DIST-E1, COMP-E1, COMP-E3, QUAL-E1 |
| QUAL-E4 | Reference executor and kernel conformance | 7 | 7 | 23 | COMP-E1, COMP-E3, COMP-E4, COMP-E5, COMP-E7, COMP-E13, COMP-E14, COMP-E15, QUAL-E2, QUAL-E3, QUAL-E7, QUAL-E15 |
| QUAL-E5 | Parity harness | 8 | 1 | 20 | PLC-E9, COMP-E3, COMP-E5, COMP-E14, COMP-E16, COMP-E17, PROD-E1, PROD-E3, PROD-E4, QUAL-E3, QUAL-E4, QUAL-E7, QUAL-E12 |
| QUAL-E6 | Execution-path coverage, family conformance and the `ConformanceManifest` | 3 | 20 | 24 | PLC-E1, TIER-E6, COMP-E5, COMP-E7, COMP-E10, COMP-E16, PROD-E1, PROD-E4, PROD-E6, QUAL-E4, QUAL-E5, QUAL-E7 |
| QUAL-E7 | Fixtures and test data | 3 | 3 | 10 | COMP-E10, COMP-E16, QUAL-E1, QUAL-E3 |
| QUAL-E8 | Deterministic multi-node simulator and emulated fabric | 1 | 6 | 6 | QUAL-E3 |
| QUAL-E12 | Benchmark harness, measurement protocol and results store | 6 | 2 | 21 | COMP-E17, PROD-E1, PROD-E4, QUAL-E1, QUAL-E2, QUAL-E5, QUAL-E7, QUAL-E15 |
| QUAL-E14 | Regression gates, baseline comparisons and publishing | 2 | 9 | 22 | QUAL-E12 |
| QUAL-E15 | Reference hardware lab | 6 | 3 | 17 | QUAL-E2, QUAL-E5, QUAL-E12 |
| QUAL-E16 | Build matrix, kernel cache, reproducibility, signing and provenance | 1 | 5 | 5 | QUAL-E1 |

**Spike and decision tasks in M0a:** 7.

| Task | Kind | What | Answers (OQ) | Direct dependents |
|---|---|---|---|---|
| COMP-E3-T03 | spike | Decode GEMV for Q4_K, Q8_0, Q4_0, BF16 and F16 at `X86V3`, `X86V4`, `Arm82Dot` and `Arm86I8mm`, with 8-bit activation blocks for… | – | 13 |
| COMP-E3-T18 | spike | Spike S-08, CPU kernel language and CPU/GPU expert parity, as a throwaway standalone harness in M0 week 3, pulled from M1 so it runs before… | OQ-KB-05 | 2 |
| COMP-E7-T03 | spike | Buffer interop on CPU, on the contract the S-04 spike settled: host-pointer wrapping (the CUDA half, patch P1 and the buffer-type fallback… | OQ-KB-11 | 7 |
| COMP-E7-T16 | spike | Spike S-04, the ggml buffer contract, as a throwaway standalone harness in M0 weeks 1–2, first in the COMP stream: read ggml's buffer-type… | – | 1 |
| QUAL-E5-T01 | spike | Spike: corpus sourcing, in M0's first week, because S-01 and S-03 use the same corpus | – | 6 |
| QUAL-E5-T08 | spike | Spike: reference throughput F_ref, and the size of R-LABCPU, on a candidate host before R-LABCPU is sized, because the calibration… | OQ-FID-14 | 2 |
| QUAL-E7-T06 | spike | Spike and import: real checkpoints for the M0 families, as a desk check (OQ-MF-04) | – | 12 |

**Open questions needed by M0a:** 3 (decision 3). Answered by no task: OQ-PRC-21, OQ-PRC-22, OQ-PRC-23.
<!-- END GENERATED -->

## M0b: parity, certification and the lab

### Goal and exit criteria

<!-- BEGIN GENERATED: roadmap-m0b-goal -->
Quoted from RFC-0001 §16, "M0b: parity, certification and the lab":

> The rest of M0, on the same single device: the MoE family, certification, probes, the lab and the benchmark harness.
> - `qwen_moe` (GQA with softmax top-k MoE, every expert resident in VRAM), its `FamilyCert` gated on golden logits from the publisher's F32 reference as in M0a (RFC-0043, RFC-0042 §7).
> - The other M0 schemes (Q4_0, BF16, F16) and captured execution on CUDA (RFC-0040, RFC-0012).
> - Calibration on the M0 families replaces the provisional parity ceilings. The MoE ceilings stay `provisional` until the fixed-order F32 calibration reference on a GPU lands, cross-checked against the CPU `F32Fixed` reference on the nano and micro fixtures (RFC-0042 §6).
> - A minimal signed `ConformanceManifest` (op and family certificates) gates op selection in place of M0a's allow-list; the CUDA provider loads as a provider library through `kernels-loader` (RFC-0060 §6.4, RFC-0040).
> - The `Warming` canary step is skipped and reported `not-run` (no reference digests or packs yet).
> - Probe suite v1 measures `W_eff`, pinned PCIe, NVMe sustained read and `t_fixed` within SC-22's budget (RFC-0010).
> - The lab dispatcher with just-in-time runners, the benchmark harness tied to a commit, ABBA block statistics, noise bands and the PR† regression gate in report-only mode (RFC-0061, RFC-0062).
> - The deterministic simulator core, the fuzz targets for the M0 parsers, and release automation (RFC-0060, RFC-0062).
> - The release-signing hierarchy: offline root and targets keys on hardware tokens, online keys in an HSM-backed key service, and a dry-run signing ceremony before the first public release candidate (RFC-0062 §7, RFC-0034).
> - Exit: parity passes for `llama` and `qwen_moe` on the CPU and on R-ADA (SC-29 holds from here on, every release), and single-tier decode meets SC-4 for the all-`Vram` case on the CUDA rig. The MoE ceilings are reported `provisional`. The single-device baseline comparison under ADR-0612 gets its early read on R-ADA (spike S-10), recorded with the exit.

Success criteria first required at this exit: SC-4, SC-29.
<!-- END GENERATED -->

### Plan

<!-- BEGIN GENERATED: roadmap-m0b -->
**Tasks:** 130 (PLC 16, TIER 12, DIST 2, COMP 33, PROD 17, QUAL 50).

**Critical path inside M0b:** 7 tasks, assuming every earlier milestone is done. 7 M0b tasks lie on a chain of that length (no slack); the chain below is one of them.

| Step | Task | What |
|---|---|---|
| 1 | COMP-E9-T01 | Translation-unit structure and instantiation lists for `kernels-cuda`: one unit per (op family, quant-scheme group)… |
| 2 | QUAL-E16-T04 | Kernel object cache v1 for CUDA: `SourceKey`, include `Manifest` and `ObjectKey`; `cargo xtask kcache` called from… |
| 3 | COMP-E9-T02 | `build.rs` through `cargo xtask kcache` for `kernels-cuda`: `SYNTROPY_KERNELS` set to `prebuilt`, `build` or `none`… |
| 4 | QUAL-E16-T01 | M0 build jobs and images: a digest-pinned Linux build image on glibc 2.35 with the CUDA 13 toolkit, CMake and a C++… |
| 5 | PROD-E10-T01 | Artefact layout and lockstep versioning: one core binary per Tier 1 target (glibc ≥ 2.35 baseline on Linux), version… |
| 6 | PROD-E11-T11 | `module.toml` schema (RFC-0052 §4.1): name, version, build ID, target, provider ABI, backend, providers, tier… |
| 7 | COMP-E9-T05 | Load check in `kernels-loader`: the seven ordered checks (path from `module.toml`, entry symbol for the core's major… |

Zero-slack tasks by area: COMP 3, PROD 2, QUAL 2.

**Longest chain from project start to an M0b task:** 28 tasks: QUAL-E1-T01 (M0a) → QUAL-E12-T01 → QUAL-E15-T01 → QUAL-E15-T02 → QUAL-E12-T04 → QUAL-E2-T01 → QUAL-E2-T04 → QUAL-E2-T11 → QUAL-E15-T16 → PLC-E1-T04 → PLC-E1-T07 → PLC-E1-T09 → PLC-E4-T02 → PLC-E7-T02 → PLC-E7-T03 → PLC-E7-T04 → PLC-E9-T02 → PLC-E9-T16 → PROD-E4-T01 → QUAL-E12-T02 → QUAL-E12-T03 → QUAL-E2-T02 → QUAL-E4-T06 → QUAL-E6-T04 → COMP-E16-T02 → COMP-E16-T14 → COMP-E5-T24 (M0b) → QUAL-E13-T02.

**Parallel work.** Tasks per step, where a task's step is 1 + its longest chain of prerequisites inside the milestone: 1: 67, 2: 29, 3: 12, 4: 9, 5: 9, 6: 3, 7: 1. 67 tasks can start as soon as M0b opens.

Work streams are the epics with tasks in this milestone. "First step" and "last step" are the steps of the stream's earliest and latest tasks. Streams whose step ranges overlap, and that do not wait on each other, can run in parallel.

| Stream | Epic | Tasks | First step | Last step | Waits on (M0b streams) |
|---|---|---|---|---|---|
| PLC-E1 | Resource graph types and hardware enumeration | 3 | 1 | 2 | – |
| PLC-E2 | Startup microbenchmark suite | 7 | 1 | 5 | PLC-E1, TIER-E5 |
| PLC-E3 | Profile cache, online refinement and drift detection | 3 | 1 | 2 | – |
| PLC-E5 | Joint MILP on HiGHS | 1 | 1 | 1 | – |
| PLC-E7 | Executor: step programs, dispatchers and graph capture | 2 | 1 | 2 | COMP-E6 |
| TIER-E1 | Tier manager core: arenas, handles and residency | 4 | 3 | 5 | QUAL-E3 |
| TIER-E5 | Weight I/O engine | 6 | 1 | 3 | – |
| TIER-E8 | Expert cache: slots, policy, slot table and profile | 2 | 1 | 3 | PLC-E7, QUAL-E5 |
| DIST-E2 | Identity, keys and signed structures | 1 | 1 | 1 | – |
| DIST-E17 | Security hardening: encryption policy, integrity, signing hierarchy and privacy | 1 | 1 | 1 | – |
| COMP-E1 | Kernel provider core | 2 | 1 | 1 | – |
| COMP-E2 | Op selection, certification, coverage and kernel targets | 6 | 1 | 5 | PLC-E2, COMP-E9 |
| COMP-E4 | CUDA provider: runtime, memory, doorbells and capture | 2 | 1 | 1 | – |
| COMP-E5 | CUDA provider: kernels | 1 | 1 | 1 | – |
| COMP-E6 | Metal provider | 1 | 1 | 1 | – |
| COMP-E7 | ggml provider | 3 | 1 | 2 | QUAL-E4 |
| COMP-E9 | Kernel builds, provider libraries and the module ABI | 4 | 1 | 7 | PROD-E11, QUAL-E16 |
| COMP-E11 | Model store and packs | 1 | 1 | 1 | – |
| COMP-E12 | Weight distribution: Hub, chunk tables and peer-to-peer transfer | 1 | 2 | 2 | COMP-E11 |
| COMP-E13 | Numerics: types, dequantisation, widening and accumulation rules | 4 | 1 | 4 | COMP-E7, COMP-E9 |
| COMP-E14 | The exactness contract: tolerances, wire, canary, knobs and requantization | 5 | 1 | 5 | QUAL-E4, QUAL-E5, QUAL-E7, QUAL-E16 |
| COMP-E15 | Graph IR, blocks, segments and tensor-parallel sharding | 1 | 1 | 1 | – |
| COMP-E16 | Family modules, classification and certification | 1 | 1 | 1 | – |
| COMP-E17 | Tokenizers and chat formats | 1 | 1 | 1 | – |
| PROD-E1 | Configuration and policy | 4 | 1 | 2 | – |
| PROD-E2 | API core, vendor shapes, routing and streaming | 3 | 1 | 1 | – |
| PROD-E6 | CLI and the model picker | 1 | 2 | 2 | PROD-E1 |
| PROD-E8 | Observability core: metrics, step records, tracing and logs | 2 | 1 | 2 | QUAL-E3 |
| PROD-E10 | Packaging, installers and first run | 2 | 1 | 5 | QUAL-E16, QUAL-E17 |
| PROD-E11 | Backend modules | 1 | 6 | 6 | PROD-E10 |
| PROD-E12 | Services, updates, rollback and uninstall | 1 | 2 | 2 | DIST-E17 |
| PROD-E13 | User documentation and community intake | 3 | 1 | 2 | – |
| QUAL-E1 | Workspace, CI skeleton and policy checks | 4 | 1 | 1 | – |
| QUAL-E2 | Runner fleet and hardware-job security | 1 | 2 | 2 | QUAL-E1 |
| QUAL-E3 | Determinism foundations, testkit core and the test pyramid's base | 7 | 1 | 6 | TIER-E1, TIER-E5, COMP-E1 |
| QUAL-E4 | Reference executor and kernel conformance | 3 | 1 | 5 | COMP-E9, COMP-E13, QUAL-E7, QUAL-E16 |
| QUAL-E5 | Parity harness | 5 | 1 | 4 | COMP-E17, QUAL-E4, QUAL-E7 |
| QUAL-E6 | Execution-path coverage, family conformance and the `ConformanceManifest` | 1 | 4 | 4 | QUAL-E7 |
| QUAL-E7 | Fixtures and test data | 3 | 2 | 3 | COMP-E13, QUAL-E5 |
| QUAL-E8 | Deterministic multi-node simulator and emulated fabric | 2 | 1 | 1 | – |
| QUAL-E10 | Fuzzing | 2 | 1 | 2 | PROD-E1 |
| QUAL-E12 | Benchmark harness, measurement protocol and results store | 5 | 1 | 2 | – |
| QUAL-E13 | Scenarios, success-criteria mapping and microbenchmarks | 4 | 1 | 3 | COMP-E5, QUAL-E12 |
| QUAL-E14 | Regression gates, baseline comparisons and publishing | 1 | 2 | 2 | QUAL-E12 |
| QUAL-E15 | Reference hardware lab | 2 | 1 | 2 | PLC-E2, QUAL-E12 |
| QUAL-E16 | Build matrix, kernel cache, reproducibility, signing and provenance | 7 | 1 | 6 | COMP-E9 |
| QUAL-E17 | Versioning, release pipeline, compatibility and dependencies | 3 | 2 | 3 | QUAL-E1, QUAL-E16 |

**Spike and decision tasks in M0b:** 19.

| Task | Kind | What | Answers (OQ) | Direct dependents |
|---|---|---|---|---|
| PLC-E2-T13 | benchmark | Measure the cost of deferred S1 (RFC-0010 open question 2): how often first plans use `Derived` storage values because the store was empty… | OQ-HW-09 | 0 |
| PLC-E2-T22 | spike | Spike S-05, host-path kit, as a standalone throwaway harness (M0 week 3), on R-ADA and on R-BW with its CPU pool pinned to one CCD, Linux… | – | 5 |
| PLC-E5-T02 | spike | Spike S-09 (spike S1), a standalone throwaway harness in M0 | OQ-PL-01 | 2 |
| PLC-E7-T20 | spike | Spike S-07 part 1, Apple sync kit (placement half), as a standalone throwaway harness (M0 week 1), on the agent pool's M4- or M5-class Mac… | OQ-PL-19 | 3 |
| TIER-E5-T16 | spike | Spike S-06 part 1, NVMe kit, as a standalone throwaway harness (M0 week 3): `fio` with io_uring against a blocking pread pool at expert… | OQ-IO-03 | 3 |
| TIER-E5-T17 | spike | Spike S-06 part 2, as a standalone throwaway harness (M0 week 3): p99 demand wait under saturating prefetch at in-flight caps of 8, 16, 32… | OQ-IO-19 | 1 |
| TIER-E8-T09 | spike | Spike S-01, routing traces, as a standalone throwaway harness (M0 week 1): teacher-forced capture of every v1 MoE family's router choices… | OQ-MOE-01, OQ-MOE-06, OQ-MOE-09, OQ-MOE-12 | 7 |
| COMP-E2-T16 | spike | Spike S-02, the CUDA host-overhead kit, as a throwaway standalone harness in M0 weeks 2–3, before the executor exists: one captured graph… | – | 2 |
| COMP-E6-T12 | spike | Spike S-07 part 2, Apple sync kit (provider half; part 1 is PLC-E7-T20), as a throwaway standalone Metal harness in M0 week 1: 61 host… | – | 2 |
| COMP-E7-T18 | spike | Buffer interop on CUDA (split from COMP-E7-T03, because ggml's CUDA backend is COMP-E7-T05, M0b): patch P1 (non-owning device buffer) on… | – | 2 |
| COMP-E14-T12 | spike | Spikes for RFC-0042 open questions 1 and 2: whether 5- and 6-bit types need their own class; canary sensitivity | OQ-FID-09, OQ-FID-10 | 0 |
| COMP-E14-T22 | spike | Spike S-03, parity pre-calibration, as a throwaway harness in M0 week 2: a plain F32 forward pass against ggml's CPU backend on a… | – | 1 |
| PROD-E2-T19 | spike | Spike, first pass: client tolerance of extension fields (open question 1) against the M0 chat endpoint, for the base shape: the most-used… | – | 1 |
| PROD-E12-T11 | spike | Spike, moved to the end of M0 and widened: signing-toolchain interop | OQ-PKG-13 | 2 |
| QUAL-E5-T07 | spike | Spike: `KL_K` tightness and the tail check | OQ-FID-15 | 0 |
| QUAL-E14-T16 | spike | Spike S-10: the baseline early read at the M0b exit (ADR-0612 decision 8) | – | 1 |
| QUAL-E16-T02 | spike | Spike: build times | OQ-PRC-15 | 3 |
| QUAL-E16-T28 | spike | Spike: hosted GPU runner driver check (ADR-0610 decision 8) | – | 1 |
| QUAL-E17-T22 | spike | Spike: release-please on `release/vX.Y` with frozen crate versions and draft releases, in a scratch repository (OQ-PRC-16) | – | 1 |

**Open questions needed by M0b:** 12 (benchmark 8, spike 4). Each one names a task that records the answer.
<!-- END GENERATED -->

## M1a: Linux preview (CUDA and CPU tiering)

### Goal and exit criteria

<!-- BEGIN GENERATED: roadmap-m1a-goal -->
Quoted from RFC-0001 §16, "M1a: Linux preview (CUDA and CPU tiering)":

> Single-device tiering with CUDA and the CPU on Linux, which is the story of the cost model's scenarios (a) and (b2). M1a ships publicly, labelled a preview; the launch waits for M2.
> - Tier manager with DMA-registered `HostPinned` and locked `HostPageable` arenas per NUMA domain, the Linux budgets, and progressive locking and registration; packs with 4 KiB alignment; the io_uring and thread-pool I/O engines (RFC-0020, RFC-0021, RFC-0041).
> - Expert cache with decayed LFU, CPU execute-in-place for misses, next-layer prediction fused into the router op, and prefetch. How much of the NVMe expert tier M1a takes on follows the hit curves from S-01's routing traces. Until those exist, the model class that fits in RAM leads, and DeepSeek-V3 at Q4 is a capacity demonstration (RFC-0023).
> - Solver v1 on one node (devices × tiers): the DP with greedy in-node placement, and a placement preview in the CLI. The MILP is built only if S-09's simulation study shows it beating the DP by more than 5% on a reference input (RFC-0011, ADR-0008).
> - Single-node KV: paging across `Vram` → host RAM → `Nvme` with the `spill_compute` choice (CPU split attention on the CUDA and CPU providers, streaming to the GPU), the prefix cache on a single-stage instance, saved sessions on one node with incremental saves, and recurrent state in the same store (RFC-0022). Cross-node KV follows in M2 (prefix cache across stages, KV export and import for re-placement) and M3 (KV migration for recovery, session restore under a changed split or from remote holders).
> - The CPU provider's AVX-512 and AMX paths, and CUDA conformance on R-BW (RFC-0040).
> - Single-node speculative decoding with model-native MTP heads and with independent draft models (ADR-0017 item 7). Latency mode runs up to 4 concurrent decode rows for dense models, and for RAM-resident MoE when KV fits; NVMe-tiered MoE stays serial (RFC-0012).
> - The `Warming` canary gate is enforced from M1a (in M0 it is reported `not-run`).
> - The preview's API: OpenAI Chat Completions with tools, reasoning and structured output; stateless Responses and fill-in-the-middle; Anthropic Messages; the native `generate` API; one-node placement previews and automatic placement on first request. Positions past the longest certified context are labelled beyond the certified context (`FidelityReport::beyond_certified_from`, RFC-0050 §10). `admin`, `sessions` and keyholder routes answer only over the local socket or with a key, and every non-loopback bind needs a key and TLS (RFC-0050, RFC-0051, RFC-0054).
> - Hooks reserved for multimodal input: an `EmbedInject` node, an encoder node class and census classes. v1 is text-only (RFC-0043).
> - One-command install on Linux, with the `cuda13` module fetching NVIDIA's redistributable libraries at setup; a first-run picker that asks one use question; the model catalogue (RFC-0052, RFC-0051).
> - What a public build needs: imported `cargo vet` audits for dependency updates, the expedited security-release path, signed Linux artefacts and the update channels, and header parsing of newly downloaded models in a resource-limited child process (RFC-0062, RFC-0034, RFC-0041).
> - A standalone network probe for alpha testers' homes, shipped mid-M1a, whose traces set M2's Wi-Fi priors, wireless timeouts and relay sizing (RFC-0030, RFC-0031).
> - Exit: SC-1 to SC-11 and SC-18 to SC-21 on the Linux single-device rigs, including scenario (a) and (b2) of the cost model. The single-device baseline comparison on R-ADA under ADR-0612 passes; it is a release gate from the first M1a release candidate. The MoE ceilings stay `provisional` until the GPU calibration reference lands.

Success criteria first required at this exit: SC-1, SC-2, SC-3, SC-5, SC-6, SC-7, SC-8, SC-9, SC-10, SC-11, SC-18, SC-19, SC-20, SC-21.
<!-- END GENERATED -->

### Plan

<!-- BEGIN GENERATED: roadmap-m1a -->
**Tasks:** 478 (PLC 75, TIER 100, DIST 11, COMP 102, PROD 96, QUAL 94).

**Critical path inside M1a:** 19 tasks, assuming every earlier milestone is done. 21 M1a tasks lie on a chain of that length (no slack); the chain below is one of them.

| Step | Task | What |
|---|---|---|
| 1 | COMP-E13-T04 | Exact widening: E2M1, E4M3, E5M2 and E8M0 tables; the MX scale on F32 partial sums; NVFP4's folded E4M3 scale and the… |
| 2 | COMP-E10-T05 | `quantization_config` mapping per RFC-0041 §2.2's table (FP8 block, tensor and channel; MXFP4; compressed-tensors… |
| 3 | COMP-E5-T03 | Low-precision paths: FP8 and FP4 decode by exact widening (MX scale on F32 partial sums, NVFP4 scale folded); prefill… |
| 4 | COMP-E16-T06 | `deepseek3` family (also Kimi K2-class and Mistral Large 3): MLA, DSA-class variants, sigmoid routing with bias and… |
| 5 | COMP-E16-T07 | `qwen_hybrid` family: Gated DeltaNet with gated attention, QSA-class block-sparse attention, shared experts, the n-gram… |
| 6 | QUAL-E7-T09 | Routing traces: record each pass's routing per MoE layer (its `RouteSlot` entries and token count, RFC-0060 §12.4) from… |
| 7 | TIER-E8-T02 | Trace-driven cache simulator in `testkit`: replays recorded traces against any `CachePolicy` at a grid of capacities… |
| 8 | TIER-E10-T07 | Solver curves (§13): shadow caches per layer at up to 9 grid capacities, concave hull with `hull_gap`; union curves… |
| 9 | PLC-E4-T03 | Preprocessing of the census: blocks between family-declared legal boundaries, coarsening to `solver.max_blocks`… |
| 10 | PLC-E4-T04 | Placement nodes `Local(n)` and execution pairs with rates at pass size T from `QuantBw` curves: `(GPU, Vram)`… |
| 11 | PLC-E4-T06 | Evaluator: exact `PredictedCost` from an integral plan, per-node per-tier peak residency, and the repair loop (demote… |
| 12 | PLC-E5-T06 | Speculative outer loop on the production method (the DP, or the MILP where §9.5's gate built it): solve at T = 1, scan… |
| 13 | PLC-E6-T02 | Plan assembly: `StageAssignment` (roles, mode, `HopPlan` with best measured transport and wire dtype F32 or a… |
| 14 | PLC-E6-T13 | `spill_compute` choice per (stage, KV class) (TIER-E6-T15): evaluate RFC-0022 §5.4's two canonical forms (`CpuSplit`… |
| 15 | TIER-E6-T15 | `spill_compute` adoption: read `KvBudget::spill_compute` per (stage, class), choose `CpuSplit` for passes below… |
| 16 | TIER-E6-T17 | Certification gate for KV spill in exact mode: before spilling, check the plan's certified run-time values and the… |
| 17 | PROD-E4-T03 | Fidelity labels: `FidelityReport` and `SyntropyMeta` in every shape at the points §10 lists; `Syntropy-Fidelity` and… |
| 18 | PLC-E8-T16 | Runtime path gates: at admission the head reads the plan's certified run-time values… |
| 19 | PLC-E8-T20 | Concurrent decode rows in latency mode (RFC-0012 §4.6): on instances whose stages are dense or whose routed pools are… |

Zero-slack tasks by area: PLC 10, TIER 4, COMP 5, PROD 1, QUAL 1.

**Longest chain from project start to an M1a task:** 42 tasks: QUAL-E1-T01 (M0a) → QUAL-E12-T01 → QUAL-E15-T01 → QUAL-E15-T02 → QUAL-E12-T04 → QUAL-E2-T01 → QUAL-E2-T04 → QUAL-E2-T11 → QUAL-E15-T16 → PLC-E1-T04 → PLC-E1-T07 → PLC-E1-T09 → PLC-E4-T02 → PLC-E7-T02 → PLC-E7-T03 → PLC-E7-T04 → PLC-E9-T02 → PLC-E9-T16 → PROD-E4-T01 → QUAL-E12-T02 → QUAL-E12-T03 → QUAL-E2-T02 → QUAL-E4-T06 → QUAL-E6-T04 → COMP-E16-T02 → COMP-E16-T14 → COMP-E16-T06 (M1a) → COMP-E16-T07 → QUAL-E7-T09 → TIER-E8-T02 → TIER-E10-T07 → PLC-E4-T03 → PLC-E4-T04 → PLC-E4-T06 → PLC-E5-T06 → PLC-E6-T02 → PLC-E6-T13 → TIER-E6-T15 → TIER-E6-T17 → PROD-E4-T03 → PLC-E8-T16 → PLC-E8-T20.

**Parallel work.** Tasks per step, where a task's step is 1 + its longest chain of prerequisites inside the milestone: 1: 108, 2: 64, 3: 50, 4: 40, 5: 21, 6: 31, 7: 37, 8: 20, 9: 17, 10: 10, 11: 9, 12: 18, 13: 6, 14: 17, 15: 13, 16: 11, 17: 2, 18: 3, 19: 1. 108 tasks can start as soon as M1a opens.

Work streams are the epics with tasks in this milestone. "First step" and "last step" are the steps of the stream's earliest and latest tasks. Streams whose step ranges overlap, and that do not wait on each other, can run in parallel.

| Stream | Epic | Tasks | First step | Last step | Waits on (M1a streams) |
|---|---|---|---|---|---|
| PLC-E1 | Resource graph types and hardware enumeration | 2 | 1 | 2 | TIER-E2 |
| PLC-E2 | Startup microbenchmark suite | 9 | 1 | 6 | PLC-E1, TIER-E2, COMP-E3, COMP-E4, COMP-E5 |
| PLC-E3 | Profile cache, online refinement and drift detection | 7 | 1 | 12 | PLC-E7, TIER-E3, TIER-E4, TIER-E5, TIER-E10, COMP-E4 |
| PLC-E4 | Solver foundations: inputs, preprocessing, filter, evaluator and DP | 8 | 9 | 13 | PLC-E2, TIER-E10, COMP-E2, COMP-E10, COMP-E15 |
| PLC-E5 | Joint MILP on HiGHS | 8 | 1 | 13 | PLC-E4, TIER-E2, TIER-E8, TIER-E9 |
| PLC-E6 | Plan assembly, re-solve, routing and explanations | 10 | 12 | 15 | PLC-E2, PLC-E3, PLC-E4, PLC-E5, QUAL-E6, QUAL-E15 |
| PLC-E7 | Executor: step programs, dispatchers and graph capture | 10 | 1 | 14 | PLC-E2, PLC-E3, PLC-E9, TIER-E3, TIER-E5, TIER-E6, TIER-E8, TIER-E9, TIER-E10, DIST-E1, COMP-E3, COMP-E4, COMP-E5 |
| PLC-E8 | Step scheduling: prefill, decode, throughput mode and admission | 11 | 1 | 19 | PLC-E6, PLC-E7, TIER-E6, TIER-E7, TIER-E9, TIER-E10, PROD-E4 |
| PLC-E9 | Sampling and speculative decoding | 10 | 1 | 14 | PLC-E4, PLC-E5, PLC-E7, TIER-E6, TIER-E9, TIER-E10, COMP-E5, COMP-E15, PROD-E3, QUAL-E5 |
| TIER-E1 | Tier manager core: arenas, handles and residency | 4 | 1 | 16 | PLC-E6, TIER-E9, COMP-E2, COMP-E11 |
| TIER-E2 | Host memory: budgets, pin classes, progressive pinning and unified memory | 11 | 1 | 8 | TIER-E1, TIER-E8, COMP-E4 |
| TIER-E3 | VRAM split, elastic expert segments and the runtime n* rule | 8 | 1 | 11 | PLC-E7, TIER-E1, TIER-E2, TIER-E8, COMP-E4 |
| TIER-E4 | Memory pressure and priced yield | 9 | 2 | 12 | TIER-E1, TIER-E2, TIER-E3, TIER-E7, TIER-E8, TIER-E10, DIST-E1, PROD-E1, QUAL-E11, QUAL-E15 |
| TIER-E5 | Weight I/O engine | 16 | 1 | 12 | PLC-E3, PLC-E4, TIER-E2, COMP-E11, QUAL-E15 |
| TIER-E6 | KV store, paging and tiering | 14 | 1 | 16 | PLC-E2, PLC-E6, PLC-E7, PLC-E8, TIER-E1, TIER-E3, TIER-E5, TIER-E8, COMP-E1, COMP-E3, COMP-E5, COMP-E16, QUAL-E4 |
| TIER-E7 | Prefix cache, saved sessions, lossy KV and migration | 11 | 1 | 7 | TIER-E5, TIER-E6, COMP-E11, PROD-E4, PROD-E5 |
| TIER-E8 | Expert cache: slots, policy, slot table and profile | 8 | 1 | 8 | TIER-E3, TIER-E5, COMP-E4, COMP-E11, QUAL-E7 |
| TIER-E9 | The MoE step: three sites, CPU execute-in-place and the balancer | 9 | 6 | 14 | PLC-E6, PLC-E7, PLC-E8, PLC-E9, TIER-E3, TIER-E5, TIER-E8, TIER-E10, COMP-E3, COMP-E4, COMP-E5, QUAL-E15 |
| TIER-E10 | Prediction, prefetch, solver curves and approximate knobs | 10 | 6 | 9 | PLC-E3, TIER-E1, TIER-E5, TIER-E8, TIER-E9, COMP-E3, COMP-E5, QUAL-E7 |
| DIST-E1 | Process model: rings, IPC, runner spawn, lifecycle and supervision | 4 | 1 | 2 | – |
| DIST-E4 | Membership and failure detection | 2 | 2 | 3 | DIST-E7 |
| DIST-E5 | Admission, invites, revocation and key rotation | 1 | 1 | 1 | – |
| DIST-E7 | Transports, channels and link management | 1 | 1 | 1 | – |
| DIST-E17 | Security hardening: encryption policy, integrity, signing hierarchy and privacy | 3 | 1 | 7 | COMP-E11, PROD-E12, QUAL-E16 |
| COMP-E1 | Kernel provider core | 4 | 1 | 2 | – |
| COMP-E2 | Op selection, certification, coverage and kernel targets | 9 | 1 | 15 | PLC-E6, COMP-E3, COMP-E4, COMP-E5, COMP-E11, QUAL-E4, QUAL-E6, QUAL-E12, QUAL-E17 |
| COMP-E3 | CPU provider | 11 | 2 | 8 | PLC-E7, COMP-E1, COMP-E2, COMP-E5, COMP-E11, COMP-E13, COMP-E15 |
| COMP-E4 | CUDA provider: runtime, memory, doorbells and capture | 7 | 1 | 2 | PLC-E7, TIER-E8 |
| COMP-E5 | CUDA provider: kernels | 15 | 1 | 4 | TIER-E6, COMP-E1, COMP-E4, COMP-E10, COMP-E13, COMP-E14, QUAL-E5 |
| COMP-E7 | ggml provider | 1 | 2 | 2 | QUAL-E4 |
| COMP-E9 | Kernel builds, provider libraries and the module ABI | 6 | 1 | 10 | PLC-E1, PROD-E11, QUAL-E16 |
| COMP-E10 | Model readers, identities and the census | 6 | 1 | 4 | COMP-E12, COMP-E13, COMP-E15 |
| COMP-E11 | Model store and packs | 13 | 2 | 7 | TIER-E5, COMP-E12, QUAL-E15 |
| COMP-E12 | Weight distribution: Hub, chunk tables and peer-to-peer transfer | 4 | 1 | 3 | COMP-E11, PROD-E1 |
| COMP-E13 | Numerics: types, dequantisation, widening and accumulation rules | 6 | 1 | 4 | COMP-E2, COMP-E5, COMP-E10 |
| COMP-E14 | The exactness contract: tolerances, wire, canary, knobs and requantization | 6 | 1 | 14 | TIER-E9, COMP-E3, COMP-E5, COMP-E7, COMP-E13, QUAL-E4, QUAL-E5, QUAL-E6 |
| COMP-E15 | Graph IR, blocks, segments and tensor-parallel sharding | 4 | 1 | 3 | COMP-E1 |
| COMP-E16 | Family modules, classification and certification | 5 | 1 | 5 | COMP-E3, COMP-E5, COMP-E10, COMP-E15 |
| COMP-E17 | Tokenizers and chat formats | 5 | 1 | 8 | COMP-E7, COMP-E16, PROD-E3 |
| PROD-E1 | Configuration and policy | 9 | 1 | 12 | PLC-E4, PLC-E5, TIER-E4, QUAL-E17 |
| PROD-E2 | API core, vendor shapes, routing and streaming | 9 | 1 | 16 | PLC-E4, PLC-E6, PLC-E8, PROD-E1, PROD-E3 |
| PROD-E3 | Tool calling, structured output and reasoning | 9 | 1 | 10 | PLC-E9, COMP-E17, PROD-E2, QUAL-E13 |
| PROD-E4 | Native API, fidelity labels, debug surface, sessions and embeddings | 6 | 1 | 17 | PLC-E4, PLC-E6, TIER-E6, COMP-E11, COMP-E12, PROD-E1, PROD-E2, PROD-E8 |
| PROD-E5 | API authentication, bind policy and LAN TLS | 7 | 1 | 4 | TIER-E7, PROD-E1, PROD-E9, PROD-E12 |
| PROD-E6 | CLI and the model picker | 10 | 1 | 18 | PLC-E4, PLC-E5, COMP-E10, PROD-E1, PROD-E4, PROD-E5, PROD-E9, PROD-E12 |
| PROD-E7 | Dashboard | 8 | 7 | 16 | PLC-E4, PLC-E6, TIER-E4, PROD-E1, PROD-E4, PROD-E8 |
| PROD-E8 | Observability core: metrics, step records, tracing and logs | 5 | 1 | 15 | PLC-E3, PLC-E7, TIER-E1, TIER-E10, COMP-E11, QUAL-E14 |
| PROD-E9 | Doctor, diagnostic bundles, exporters and privacy | 7 | 2 | 6 | PROD-E5, PROD-E8, PROD-E12, QUAL-E12 |
| PROD-E10 | Packaging, installers and first run | 6 | 2 | 17 | PROD-E6, PROD-E11, PROD-E12, QUAL-E13, QUAL-E16 |
| PROD-E11 | Backend modules | 7 | 1 | 10 | COMP-E9, PROD-E12 |
| PROD-E12 | Services, updates, rollback and uninstall | 6 | 1 | 5 | DIST-E1 |
| PROD-E13 | User documentation and community intake | 7 | 1 | 16 | PROD-E1, PROD-E2, PROD-E4, PROD-E5, PROD-E6, PROD-E8, PROD-E9, PROD-E10, PROD-E12 |
| QUAL-E1 | Workspace, CI skeleton and policy checks | 6 | 1 | 3 | QUAL-E2 |
| QUAL-E2 | Runner fleet and hardware-job security | 3 | 1 | 2 | QUAL-E1, QUAL-E15 |
| QUAL-E3 | Determinism foundations, testkit core and the test pyramid's base | 9 | 1 | 14 | PLC-E5, PLC-E6, PLC-E7, PLC-E9, TIER-E2, TIER-E7, TIER-E8, DIST-E1, COMP-E1, COMP-E4 |
| QUAL-E4 | Reference executor and kernel conformance | 5 | 1 | 4 | COMP-E5, COMP-E9, QUAL-E2, QUAL-E16 |
| QUAL-E5 | Parity harness | 6 | 1 | 7 | PLC-E9, COMP-E11, COMP-E13, QUAL-E4 |
| QUAL-E6 | Execution-path coverage, family conformance and the `ConformanceManifest` | 8 | 1 | 18 | PLC-E4, PLC-E5, PLC-E6, TIER-E5, TIER-E8, COMP-E2, COMP-E13, COMP-E14, PROD-E4, PROD-E7, QUAL-E4, QUAL-E5 |
| QUAL-E7 | Fixtures and test data | 4 | 1 | 6 | COMP-E5, COMP-E13, COMP-E16 |
| QUAL-E8 | Deterministic multi-node simulator and emulated fabric | 2 | 4 | 5 | TIER-E4, DIST-E1 |
| QUAL-E9 | Fault injection and chaos | 3 | 1 | 7 | TIER-E2, COMP-E11, COMP-E12 |
| QUAL-E10 | Fuzzing | 3 | 1 | 8 | COMP-E10, COMP-E11, PROD-E3, PROD-E12 |
| QUAL-E11 | Platform and feature-fallback suites | 5 | 1 | 7 | TIER-E2, COMP-E4, COMP-E5, QUAL-E2, QUAL-E15 |
| QUAL-E12 | Benchmark harness, measurement protocol and results store | 5 | 1 | 12 | PLC-E4, PLC-E7, QUAL-E15 |
| QUAL-E13 | Scenarios, success-criteria mapping and microbenchmarks | 5 | 8 | 16 | PLC-E5, PLC-E6, PLC-E7, PLC-E8, PLC-E9, TIER-E2, TIER-E3, TIER-E5, TIER-E6, TIER-E7, TIER-E8, TIER-E9, TIER-E10, COMP-E3, COMP-E4, COMP-E5, COMP-E11, PROD-E6, PROD-E10, QUAL-E7, QUAL-E11, QUAL-E12, QUAL-E15 |
| QUAL-E14 | Regression gates, baseline comparisons and publishing | 7 | 1 | 16 | QUAL-E1, QUAL-E12, QUAL-E13, QUAL-E17 |
| QUAL-E15 | Reference hardware lab | 3 | 1 | 3 | QUAL-E2 |
| QUAL-E16 | Build matrix, kernel cache, reproducibility, signing and provenance | 10 | 1 | 7 | PROD-E12, QUAL-E4, QUAL-E14, QUAL-E17 |
| QUAL-E17 | Versioning, release pipeline, compatibility and dependencies | 10 | 1 | 8 | DIST-E17, COMP-E11, PROD-E12, QUAL-E1, QUAL-E7, QUAL-E14, QUAL-E16 |

**Spike and decision tasks in M1a:** 26.

| Task | Kind | What | Answers (OQ) | Direct dependents |
|---|---|---|---|---|
| PLC-E2-T14 | spike | Spike: default probe set coverage (open question 1). Count `Derived` values in first plans across the conformance models | OQ-HW-08 | 0 |
| PLC-E5-T12 | spike | Spike S-14's solver half (mid M1): serial versus overlapped NVMe misses on the (b2) rig (open question 2, OQ-PL-09, extended to the… | OQ-PL-09, OQ-PL-10 | 0 |
| PLC-E5-T13 | spike | Spike: budget-level expert placement against per-expert assignment (ADR-0102's revisit trigger) | OQ-PL-04 | 0 |
| PLC-E8-T15 | spike | Spike: tune layer-major prefill groups (PLC-E8-T17) on the (b2) rig: measure TTFT for 4K and 32K prompts with chunked versus layer-major… | OQ-PL-22 | 0 |
| TIER-E1-T11 | spike | Spike S-15, small-VRAM overheads (M1): on the lab's Turing-class card and an 8 GB-class card, the VRAM each runner process costs (context… | – | 0 |
| TIER-E2-T08 | spike | Spike: is provider host registration charged to `RLIMIT_MEMLOCK`? `VmLck`/`VmPin` probe under a low limit on every CI driver version | OQ-IO-11 | 0 |
| TIER-E2-T09 | spike | Spike: re-measure S-05's first read (PLC-E2-T22) of commit-plus-register throughput at 64 MiB, 256 MiB and 1 GiB steps on the Linux… | OQ-IO-14 | 0 |
| TIER-E2-T20 | spike | Spike: `cuMemHostRegister`/`hipHostRegister` of 64 MiB–1 GiB from a second thread during captured-graph decode, step-time p99 against the… | OQ-IO-18 | 0 |
| TIER-E5-T18 | benchmark | Benchmark SC-7: expert-sized reads at `qd_target` on every Tier 1 rig's model volume | OQ-HW-03 | 0 |
| TIER-E5-T19 | spike | Platform facts spike: Apple SSD sustained reads and throttling (the FileVault A/B is dropped: FileVault cannot change `W_nvme`, OQ-IO-01)… | OQ-HW-10, OQ-IO-01, OQ-IO-20 | 1 |
| TIER-E6-T13 | benchmark | Benchmark: decode attention throughput and spill bandwidth at P = 16, 32, 64 and 128 on the CUDA and Metal providers | OQ-IO-21 | 0 |
| TIER-E6-T14 | spike | Spike S-11 (early M1; timebox 1 week [estimate]) | OQ-IO-22 | 0 |
| TIER-E7-T16 | spike | Spike on recorded agent traces: state checkpoint spacing and autosave threshold and retention defaults | OQ-IO-23, OQ-IO-24 | 0 |
| TIER-E9-T12 | spike | Spike S-14's tier half (mid M1, with PLC-E5-T12; timebox: 1.5 weeks for S-14 as a whole [estimate]) | OQ-MOE-04 | 0 |
| DIST-E5-T14 | spike | Spike, late in M1, before `PairRequest` freezes: the commit-then-reveal pairing exchange as a pure protocol model with a grinding-relay test | – | 1 |
| COMP-E4-T03 | spike | `host_registrar`: a `Send` registrar that makes the primary context current on the calling thread, so progressive pinning registers ranges… | – | 2 |
| COMP-E4-T18 | spike | Spike S-15, capture half: device memory per captured graph instance, one runner per model against a shared runner on 8 GB cards, and… | – | 0 |
| COMP-E11-T19 | benchmark | Benchmark: pack overhead against shard size per backend layout (the ADR-0005 and ADR-0410 trigger) | OQ-MF-02 | 0 |
| COMP-E11-T20 | spike | Spike: disk footprint of an NVMe-tiered MoE: measure source plus pack bytes for the b2 plan, and prototype a `SOURCE`-layout pack extent… | – | 1 |
| COMP-E14-T16 | spike | Spike for RFC-0042 open question 4: F16 KV overflow by family. Exit: families flagged to force BF16 KV | OQ-FID-11 | 0 |
| COMP-E14-T21 | benchmark | Benchmark: missed experts computed on the CPU against the GPU path for every quant type of the M1 roster, since the two sites round… | OQ-FID-03 | 0 |
| PROD-E2-T12 | spike | Spike, second pass: client tolerance of extension fields (open question 1) against the full M1 surface, by category: chat front ends… | OQ-PRD-08 | 2 |
| PROD-E3-T08 | spike | Spike: mask cost at large vocabularies (open question 4), on every v1 tokenizer with deep schemas | OQ-PRD-11 | 0 |
| PROD-E6-T22 | spike | Spike: disk footprint of an NVMe-tiered MoE plan: measure source plus pack bytes for the b2 plan and prototype a SOURCE-layout pack extent… | – | 0 |
| PROD-E8-T09 | spike | Spikes: GPU per-term timing method per backend (timed graph variant or resident event nodes), with PLC and COMP (open question 1)… | OQ-PRD-21, OQ-PRD-22, OQ-PRD-24 | 2 |
| QUAL-E17-T06 | spike | Patch releases: release-please on `release/vX.Y`; a `backport:vX.Y` label that opens the cherry-pick PR; patch RCs run on their tag with… | OQ-PRC-16 | 0 |

**Open questions needed by M1a:** 54 (benchmark 23, check 1, decision 4, spike 26). Answered by no task: OQ-PRC-25.
<!-- END GENERATED -->

## M1b: Windows and Apple silicon

### Goal and exit criteria

<!-- BEGIN GENERATED: roadmap-m1b-goal -->
Quoted from RFC-0001 §16, "M1b: Windows and Apple silicon":

> M1a's single-device tiering on Windows and on macOS with Apple silicon. Its scope is reviewed at M1a's exit against M1a's measurements: a feature whose M1a measurement misses its SC is re-planned before it is ported.
> - The Metal native provider and the Metal probes; `Unified` arenas, the wired limit and the macOS I/O engines (`F_NOCACHE` with `preadv`, and mapped reads); the NEON and SME2 CPU paths (RFC-0040, RFC-0020, RFC-0021).
> - Windows: working-set locking, the IoRing and IOCP engines, and the WDDM and WSL2 specifics (RFC-0020, RFC-0021, RFC-0040).
> - One-command install on Windows (MSI and winget) and on macOS (app bundle and Homebrew cask), and a non-root LaunchDaemon for headless Macs, tested with FileVault on. Signing and notarisation follow RFC-0062 §7 (RFC-0052).
> - Hardware runs for fork pull requests start, each run approved by a maintainer (RFC-0062 §3).
> - The Thunderbolt 5 RDMA raw-path probe on a pair of Macs, which decides whether Mac-pair TP and SC-17 stay in M3 (RFC-0031, RFC-0032).
> - The pairing protocol's commit-then-reveal exchange, modelled and tested against a grinding relay before `PairRequest` freezes (RFC-0030 §6).
> - Exit: SC-1 to SC-11 and SC-18 to SC-21 on the Windows and Apple single-device rigs, wherever each applies. SC-2 on Metal and Vulkan MoE stages is provisional until the M0b spike PLC-E7-T20 sets it (`00-vision-and-scope.md` SC-2). If the Ultra-class Mac in the test matrix has 256 GB, scenario (b1)'s single-node case is measured on a model that fits it (ADR-0610).

Success criteria first required at this exit: none.
<!-- END GENERATED -->

### Plan

<!-- BEGIN GENERATED: roadmap-m1b -->
**Tasks:** 92 (PLC 11, TIER 19, DIST 2, COMP 23, PROD 16, QUAL 21).

**Critical path inside M1b:** 7 tasks, assuming every earlier milestone is done. 7 M1b tasks lie on a chain of that length (no slack); the chain below is one of them.

| Step | Task | What |
|---|---|---|
| 1 | COMP-E6-T01 | Binding and libraries: `objc2-metal`, `DeviceClass::Metal { family, metal4 }` from family checks, feature bits… |
| 2 | COMP-E6-T02 | Memory: `import` with no-copy buffers over 16 KiB-aligned tier-manager memory, private heaps for scratch, residency-set… |
| 3 | COMP-E6-T03 | Doorbells and capture: shared-event signal and wait with per-step values (encoder-level on the Metal 3 path… |
| 4 | COMP-E6-T05 | Kernels I: GEMV (Q4_K, Q8_0, Q4_0, BF16, F16, then the remaining GGUF types), FP8 and FP4 by exact widening, `MoeRoute`… |
| 5 | COMP-E6-T06 | Kernels II: prefill GEMM through `matmul2d` (Metal 4) and `simdgroup_matrix` (Metal 3.2), each with and without… |
| 6 | PLC-E2-T21 | Metal probes: D1 (device open and working-set facts), M1 and M3 on the Apple GPU over `Unified`, M4, the `Unified`… |
| 7 | QUAL-E13-T20 | Windows and macOS part of QUAL-E13-T03: the M1b single-device scenarios on R-WIN, R-MACF, R-MBP and R-ULTRA-A, each M1b… |

Zero-slack tasks by area: PLC 1, COMP 5, QUAL 1.

**Longest chain from project start to an M1b task:** 39 tasks: QUAL-E1-T01 (M0a) → QUAL-E12-T01 → QUAL-E15-T01 → QUAL-E15-T02 → QUAL-E12-T04 → QUAL-E2-T01 → QUAL-E2-T04 → QUAL-E2-T11 → QUAL-E15-T16 → PLC-E1-T04 → PLC-E1-T07 → PLC-E1-T09 → PLC-E4-T02 → PLC-E7-T02 → PLC-E7-T03 → PLC-E7-T04 → PLC-E9-T02 → PLC-E9-T16 → PROD-E4-T01 → QUAL-E12-T02 → QUAL-E12-T03 → QUAL-E2-T02 → QUAL-E4-T06 → QUAL-E6-T04 → COMP-E16-T02 → COMP-E16-T14 → COMP-E16-T06 (M1a) → COMP-E16-T07 → QUAL-E7-T09 → TIER-E8-T02 → TIER-E10-T07 → PLC-E4-T03 → PLC-E4-T04 → PLC-E4-T06 → PLC-E5-T06 → PLC-E6-T02 → PLC-E6-T20 → QUAL-E13-T03 → QUAL-E13-T13 (M1b).

**Parallel work.** Tasks per step, where a task's step is 1 + its longest chain of prerequisites inside the milestone: 1: 24, 2: 20, 3: 10, 4: 19, 5: 12, 6: 6, 7: 1. 24 tasks can start as soon as M1b opens.

Work streams are the epics with tasks in this milestone. "First step" and "last step" are the steps of the stream's earliest and latest tasks. Streams whose step ranges overlap, and that do not wait on each other, can run in parallel.

| Stream | Epic | Tasks | First step | Last step | Waits on (M1b streams) |
|---|---|---|---|---|---|
| PLC-E2 | Startup microbenchmark suite | 6 | 1 | 6 | COMP-E6 |
| PLC-E3 | Profile cache, online refinement and drift detection | 1 | 4 | 4 | COMP-E6 |
| PLC-E7 | Executor: step programs, dispatchers and graph capture | 4 | 4 | 5 | COMP-E6 |
| TIER-E2 | Host memory: budgets, pin classes, progressive pinning and unified memory | 8 | 1 | 4 | COMP-E6, PROD-E6 |
| TIER-E5 | Weight I/O engine | 7 | 1 | 2 | – |
| TIER-E8 | Expert cache: slots, policy, slot table and profile | 2 | 4 | 4 | TIER-E2, COMP-E6 |
| TIER-E9 | The MoE step: three sites, CPU execute-in-place and the balancer | 1 | 4 | 4 | COMP-E6 |
| TIER-E10 | Prediction, prefetch, solver curves and approximate knobs | 1 | 5 | 5 | COMP-E6 |
| DIST-E3 | LAN discovery | 1 | 4 | 4 | PROD-E10 |
| DIST-E7 | Transports, channels and link management | 1 | 1 | 1 | – |
| COMP-E1 | Kernel provider core | 1 | 1 | 1 | – |
| COMP-E2 | Op selection, certification, coverage and kernel targets | 4 | 4 | 5 | COMP-E6 |
| COMP-E3 | CPU provider | 1 | 1 | 1 | – |
| COMP-E4 | CUDA provider: runtime, memory, doorbells and capture | 2 | 1 | 2 | – |
| COMP-E6 | Metal provider | 8 | 1 | 6 | PLC-E7 |
| COMP-E7 | ggml provider | 4 | 1 | 3 | COMP-E6 |
| COMP-E9 | Kernel builds, provider libraries and the module ABI | 2 | 3 | 3 | QUAL-E16 |
| COMP-E11 | Model store and packs | 1 | 1 | 1 | – |
| PROD-E5 | API authentication, bind policy and LAN TLS | 1 | 1 | 1 | – |
| PROD-E6 | CLI and the model picker | 1 | 1 | 1 | – |
| PROD-E8 | Observability core: metrics, step records, tracing and logs | 1 | 4 | 4 | COMP-E6 |
| PROD-E9 | Doctor, diagnostic bundles, exporters and privacy | 1 | 2 | 2 | TIER-E2 |
| PROD-E10 | Packaging, installers and first run | 6 | 3 | 6 | PROD-E12, QUAL-E16 |
| PROD-E11 | Backend modules | 1 | 2 | 2 | QUAL-E16 |
| PROD-E12 | Services, updates, rollback and uninstall | 3 | 5 | 5 | PROD-E10 |
| PROD-E13 | User documentation and community intake | 2 | 3 | 6 | PROD-E9, PROD-E10 |
| QUAL-E2 | Runner fleet and hardware-job security | 3 | 1 | 2 | QUAL-E15 |
| QUAL-E4 | Reference executor and kernel conformance | 1 | 6 | 6 | COMP-E6 |
| QUAL-E6 | Execution-path coverage, family conformance and the `ConformanceManifest` | 1 | 1 | 1 | – |
| QUAL-E9 | Fault injection and chaos | 1 | 3 | 3 | QUAL-E11 |
| QUAL-E11 | Platform and feature-fallback suites | 5 | 2 | 4 | TIER-E2, TIER-E5, COMP-E6 |
| QUAL-E12 | Benchmark harness, measurement protocol and results store | 1 | 1 | 1 | – |
| QUAL-E13 | Scenarios, success-criteria mapping and microbenchmarks | 2 | 1 | 7 | PLC-E2, PLC-E7, TIER-E2, TIER-E5, COMP-E2, COMP-E4, COMP-E6, QUAL-E15 |
| QUAL-E15 | Reference hardware lab | 4 | 1 | 2 | QUAL-E12 |
| QUAL-E16 | Build matrix, kernel cache, reproducibility, signing and provenance | 3 | 1 | 2 | COMP-E6 |

**Spike and decision tasks in M1b:** 11.

| Task | Kind | What | Answers (OQ) | Direct dependents |
|---|---|---|---|---|
| PLC-E2-T15 | spike | Spikes on the lab machines (open questions 3, 4, 5, 7): NVMe temperature source coverage without privilege; Apple internal SSD sustained… | OQ-HW-10, OQ-HW-12, OQ-IO-01 | 0 |
| TIER-E2-T10 | spike | Spike: does Windows's ≈50% DMA lock ceiling still bind in scenario b2 after ADR-0200? Exit: build an opt-in `MEM_LARGE_PAGES` path in M4… | OQ-IO-12 | 0 |
| TIER-E2-T14 | spike | Spike S-13 (it slips with R-ULTRA-A's availability; timebox: 2 days of setup, then unattended 8-hour runs [estimate]) | OQ-HW-07 | 0 |
| TIER-E5-T31 | spike | Windows and macOS part of TIER-E5-T19: Platform facts spike | – | 0 |
| DIST-E3-T09 | spike | Spike at the start of M1, once a Developer-ID-signed build exists: macOS Local Network privacy for a per-user launch agent on macOS 15 and… | OQ-NET-15 | 0 |
| DIST-E7-T20 | spike | Spike, late in M1: the raw-path RDMA probe on a TB5 Mac pair, R-MBP plus one Mac mini or two minis, on macOS 26.2 or later | – | 2 |
| COMP-E11-T12 | spike | Spike for RFC-0041 open question 5: sparse files on Windows and APFS over a month-long simulated GC workload | OQ-MF-09 | 0 |
| PROD-E8-T13 | spike | Windows and macOS part of PROD-E8-T09: Spikes: GPU per-term timing method per backend | – | 0 |
| PROD-E10-T11 | spike | Spike, pulled from M2 into M1: headless Macs (RFC-0052 open question 4, answered), run with FileVault on on macOS 15 and 26 | – | 1 |
| PROD-E12-T12 | spike | Spike: Windows logon-task breakaway | OQ-PKG-14 | 3 |
| QUAL-E13-T13 | spike | Spike: laptop noise | OQ-PRC-12 | 0 |

**Open questions needed by M1b:** 15 (benchmark 3, decision 3, spike 9). Answered by no task: OQ-PRC-24.
<!-- END GENERATED -->

## M2: multiple devices on a LAN

### Goal and exit criteria

<!-- BEGIN GENERATED: roadmap-m2-goal -->
Quoted from RFC-0001 §16, "M2: multiple devices on a LAN":

> Several machines on one LAN, and the public launch: the M2 release carries the announcement and the public benchmark page.
> - Discovery with blinded beacons sent only on networks where a member has been seen, SWIM membership, coordinator lease and event log, with the election and split-heal design model-checked before its code; admission with commit-then-reveal and number matching, and encryption (RFC-0034 core). Coordination is keyholders-only by default, offered with a second-keyholder grant at pairing, with an optional offline recovery code at bootstrap, a guard against removing the last keyholder, and re-founding (RFC-0030, RFC-0032, RFC-0034).
> - QUIC, TCP and shared-memory transports; RDMA over Thunderbolt 5 on macOS (RFC-0031).
> - Ring pipeline across nodes with chunked pipelined prefill; joint solver over devices × tiers × links. Latency mode minimises decode time subject to a TTFT cap of 2× the best candidate's TTFT (RFC-0032, RFC-0011).
> - Shard-only fetch and P2P chunk distribution (RFC-0041).
> - Session data held on other nodes is stored as ciphertext under a per-session key that the session's own node holds (RFC-0022, RFC-0034).
> - The simulated fabric for multi-node tests (RFC-0060).
> - Dashboard v1: topology, tiers, plan explanation, live rates, multi-node previews, and a minimal chat view with one conversation over native sessions; routing across nodes and cluster-wide API keys (RFC-0051, RFC-0050, RFC-0034).
> - The embeddings API with one embedding family, on a single node (RFC-0050, RFC-0043).
> - A ggml-only `vulkan` module for AMD and Intel GPUs, labelled a Tier 2 preview (RFC-0040, ADR-0020).
> - Syntropy-only nightly benchmark results are public from M2 (RFC-0061).
> - Exit: SC-12 to SC-14, SC-22 to SC-25 and SC-27 on the cost model's scenario (c) rig and, if M1b's raw-path probe passed, a two-Mac Thunderbolt rig. SC-15 (throughput mode) is an M3 exit, because throughput mode lands in M3.

Success criteria first required at this exit: SC-12, SC-13, SC-14, SC-22, SC-23, SC-24, SC-25, SC-27.
<!-- END GENERATED -->

### Plan

<!-- BEGIN GENERATED: roadmap-m2 -->
**Tasks:** 287 (PLC 36, TIER 12, DIST 123, COMP 27, PROD 34, QUAL 55).

**Critical path inside M2:** 16 tasks, assuming every earlier milestone is done. 23 M2 tasks lie on a chain of that length (no slack); the chain below is one of them.

| Step | Task | What |
|---|---|---|
| 1 | DIST-E2-T04 | Crypto helpers: strict Ed25519 verification (canonical encodings, no small-order keys), the domain-separation context… |
| 2 | DIST-E3-T02 | Open-form beacon codec (192 B, `SYNB`, Announce/Query/Reply/Goodbye) with Ed25519 signing and verification, flags bit1… |
| 3 | DIST-E3-T03 | IPv6 link-local multicast on `ff12::5359:4e54`: per-interface sockets, hop limit 1, interface-change tracking (netlink… |
| 4 | DIST-E3-T07 | Candidate pipeline: `Candidate` (a verified beacon or mDNS result, RFC-0030 §4.6) and `NearbyUpdate`; the nearby list… |
| 5 | DIST-E4-T01 | `MembershipCore` as a `ProtocolCore`: `MemberView`, member states, incarnations, and the public `MembershipInput` and… |
| 6 | DIST-E10-T01 | `ElectionCore`: a lease deadline from beats of the elected node only, on the suspend-aware clock, `COORDINATOR`-role… |
| 7 | DIST-E8-T17 | `CtlMsg` (split from DIST-E8-T12): the control-stream messages of RFC-0031 §9.1 (`Hello`, `HelloAck`, `Command`… |
| 8 | DIST-E5-T05 | Certificate issuance and verification on every control connection: `RecordCert`, `CertIssued` carrying the certificate… |
| 9 | DIST-E5-T07 | Leave and revoke: `Leave{reason}`, with `leave` completing locally when no coordinator is reachable, keyholder-signed… |
| 10 | DIST-E5-T15 | Losing the only keyholder: the optional recovery code offered once (a 128-bit seed shown as 12 words with a checksum… |
| 11 | DIST-E9-T16 | Signed records in `ClusterState` (split from DIST-E9-T01): every authority-derived field keeps the signed record that… |
| 12 | DIST-E9-T07 | Snapshots: deterministic `zstd(cbor(ClusterState))` named by BLAKE3, every 4,096 entries or 15 min, two retained… |
| 13 | QUAL-E7-T10 | Wire vectors and log snapshots: golden byte vectors for every data-plane and control-plane message type and protocol… |
| 14 | QUAL-E17-T07 | Compatibility check in RFC-0031's terms: `ci/protocol-versions.toml` with `proto_major`, `[min_supported, current]` for… |
| 15 | QUAL-E17-T08 | Rolling upgrade and log replay: snapshots and tails from each release in the window replayed by the current `apply` on… |
| 16 | DIST-E8-T14 | Compatibility window: N−1 and N+1 minors interoperate; each release declares `[min_supported, current]` for… |

Zero-slack tasks by area: DIST 17, PROD 1, QUAL 5.

**Longest chain from project start to an M2 task:** 45 tasks: QUAL-E1-T01 (M0a) → QUAL-E12-T01 → QUAL-E15-T01 → QUAL-E15-T02 → QUAL-E12-T04 → QUAL-E2-T01 → QUAL-E2-T04 → QUAL-E2-T11 → QUAL-E15-T16 → PLC-E1-T04 → PLC-E1-T07 → PLC-E1-T09 → PLC-E4-T02 → PLC-E7-T02 → PLC-E7-T03 → PLC-E7-T04 → PLC-E9-T02 → PLC-E9-T16 → PROD-E4-T01 → QUAL-E12-T02 → QUAL-E12-T03 → QUAL-E2-T02 → QUAL-E4-T06 → QUAL-E6-T04 → COMP-E16-T02 → COMP-E16-T14 → COMP-E16-T06 (M1a) → COMP-E16-T07 → QUAL-E7-T09 → TIER-E8-T02 → TIER-E10-T07 → PLC-E4-T03 → PLC-E4-T04 → PLC-E4-T06 → PLC-E5-T06 → PLC-E6-T02 → PLC-E6-T19 → PROD-E2-T08 → PROD-E2-T13 (M2) → QUAL-E8-T08 → QUAL-E8-T09 → QUAL-E6-T08 → QUAL-E6-T11 → COMP-E14-T05 → COMP-E14-T17.

**Parallel work.** Tasks per step, where a task's step is 1 + its longest chain of prerequisites inside the milestone: 1: 48, 2: 39, 3: 20, 4: 24, 5: 18, 6: 22, 7: 21, 8: 16, 9: 11, 10: 20, 11: 13, 12: 8, 13: 14, 14: 6, 15: 3, 16: 4. 48 tasks can start as soon as M2 opens.

Work streams are the epics with tasks in this milestone. "First step" and "last step" are the steps of the stream's earliest and latest tasks. Streams whose step ranges overlap, and that do not wait on each other, can run in parallel.

| Stream | Epic | Tasks | First step | Last step | Waits on (M2 streams) |
|---|---|---|---|---|---|
| PLC-E1 | Resource graph types and hardware enumeration | 2 | 3 | 5 | DIST-E6, DIST-E7, DIST-E9 |
| PLC-E2 | Startup microbenchmark suite | 1 | 7 | 7 | PLC-E1, DIST-E7 |
| PLC-E3 | Profile cache, online refinement and drift detection | 1 | 7 | 7 | DIST-E4, DIST-E7 |
| PLC-E4 | Solver foundations: inputs, preprocessing, filter, evaluator and DP | 5 | 1 | 7 | PLC-E1, TIER-E10, DIST-E9, DIST-E13, COMP-E4, COMP-E12 |
| PLC-E5 | Joint MILP on HiGHS | 2 | 3 | 4 | PLC-E4, DIST-E7 |
| PLC-E6 | Plan assembly, re-solve, routing and explanations | 13 | 1 | 13 | PLC-E1, PLC-E3, PLC-E4, PLC-E9, TIER-E7, DIST-E7, DIST-E8, DIST-E9, DIST-E11 |
| PLC-E7 | Executor: step programs, dispatchers and graph capture | 4 | 6 | 8 | DIST-E7, COMP-E4 |
| PLC-E8 | Step scheduling: prefill, decode, throughput mode and admission | 4 | 9 | 11 | PLC-E7, DIST-E8, DIST-E11 |
| PLC-E9 | Sampling and speculative decoding | 4 | 1 | 6 | TIER-E10, DIST-E8, DIST-E9, COMP-E5 |
| TIER-E2 | Host memory: budgets, pin classes, progressive pinning and unified memory | 2 | 2 | 4 | DIST-E7, COMP-E4 |
| TIER-E3 | VRAM split, elastic expert segments and the runtime n* rule | 2 | 1 | 1 | – |
| TIER-E4 | Memory pressure and priced yield | 1 | 8 | 8 | DIST-E9 |
| TIER-E7 | Prefix cache, saved sessions, lossy KV and migration | 5 | 11 | 13 | DIST-E8, DIST-E9, DIST-E11, DIST-E17 |
| TIER-E10 | Prediction, prefetch, solver curves and approximate knobs | 2 | 5 | 11 | PLC-E6, DIST-E8, DIST-E9, DIST-E11 |
| DIST-E2 | Identity, keys and signed structures | 6 | 1 | 6 | PLC-E1 |
| DIST-E3 | LAN discovery | 9 | 1 | 10 | DIST-E2, DIST-E5, DIST-E6, DIST-E7 |
| DIST-E4 | Membership and failure detection | 9 | 5 | 10 | DIST-E3, DIST-E7, DIST-E8, DIST-E9, QUAL-E8, QUAL-E15 |
| DIST-E5 | Admission, invites, revocation and key rotation | 11 | 4 | 10 | DIST-E2, DIST-E4, DIST-E7, DIST-E8, DIST-E9 |
| DIST-E6 | Interfaces, edges and Thunderbolt topology | 6 | 1 | 9 | DIST-E3, DIST-E7, DIST-E9 |
| DIST-E7 | Transports, channels and link management | 16 | 1 | 7 | PLC-E1, PLC-E4, PLC-E7, TIER-E2, DIST-E2, DIST-E6, DIST-E8, QUAL-E15 |
| DIST-E8 | Wire formats and codecs | 13 | 1 | 16 | DIST-E2, DIST-E9, DIST-E10, QUAL-E17 |
| DIST-E9 | Cluster event log | 16 | 2 | 13 | PLC-E6, TIER-E10, DIST-E2, DIST-E5, DIST-E8, DIST-E17, COMP-E12 |
| DIST-E10 | Coordinator election and split heal | 7 | 1 | 13 | DIST-E4, DIST-E5, DIST-E9, DIST-E11, QUAL-E8, QUAL-E9 |
| DIST-E11 | Ring pipeline execution | 13 | 1 | 13 | PLC-E6, PLC-E8, DIST-E7, DIST-E8, DIST-E9, QUAL-E5, QUAL-E15 |
| DIST-E12 | Two-phase scheduling integration | 3 | 7 | 15 | PLC-E6, DIST-E4, DIST-E9 |
| DIST-E13 | Cross-node TP and EP, and intra-node TP | 1 | 1 | 1 | – |
| DIST-E14 | Fault tolerance: detection, history, replay, migration and repair | 3 | 7 | 10 | DIST-E4, DIST-E7, DIST-E9, DIST-E11, DIST-E12 |
| DIST-E15 | Elasticity: stragglers, joins and coordinator co-failure | 1 | 8 | 8 | DIST-E4 |
| DIST-E17 | Security hardening: encryption policy, integrity, signing hierarchy and privacy | 6 | 1 | 10 | DIST-E5, DIST-E7, DIST-E8, DIST-E9, DIST-E11, COMP-E12 |
| DIST-E18 | Configuration keys, error codes, metrics and diagnostics | 3 | 1 | 1 | – |
| COMP-E1 | Kernel provider core | 1 | 1 | 1 | – |
| COMP-E3 | CPU provider | 1 | 2 | 2 | COMP-E1 |
| COMP-E4 | CUDA provider: runtime, memory, doorbells and capture | 3 | 1 | 1 | – |
| COMP-E5 | CUDA provider: kernels | 2 | 1 | 2 | COMP-E1 |
| COMP-E7 | ggml provider | 1 | 1 | 1 | – |
| COMP-E8 | HIP and Vulkan providers | 1 | 2 | 2 | TIER-E3, QUAL-E6, QUAL-E15 |
| COMP-E10 | Model readers, identities and the census | 1 | 2 | 2 | QUAL-E7 |
| COMP-E11 | Model store and packs | 2 | 4 | 7 | DIST-E5, DIST-E8, COMP-E12 |
| COMP-E12 | Weight distribution: Hub, chunk tables and peer-to-peer transfer | 8 | 1 | 6 | PLC-E6, DIST-E2, DIST-E7, DIST-E8, DIST-E9 |
| COMP-E14 | The exactness contract: tolerances, wire, canary, knobs and requantization | 2 | 12 | 13 | QUAL-E6 |
| COMP-E15 | Graph IR, blocks, segments and tensor-parallel sharding | 2 | 2 | 7 | PLC-E7, COMP-E1 |
| COMP-E16 | Family modules, classification and certification | 3 | 1 | 4 | COMP-E3, COMP-E5, COMP-E15 |
| PROD-E1 | Configuration and policy | 4 | 2 | 10 | DIST-E5, DIST-E9, DIST-E10, DIST-E18, QUAL-E8 |
| PROD-E2 | API core, vendor shapes, routing and streaming | 2 | 1 | 6 | DIST-E9 |
| PROD-E4 | Native API, fidelity labels, debug surface, sessions and embeddings | 6 | 1 | 12 | PLC-E4, PLC-E6, DIST-E4, DIST-E5, DIST-E8, DIST-E9, COMP-E16 |
| PROD-E5 | API authentication, bind policy and LAN TLS | 4 | 2 | 14 | TIER-E7, DIST-E5, DIST-E6, DIST-E9, DIST-E17 |
| PROD-E6 | CLI and the model picker | 3 | 10 | 14 | DIST-E5, PROD-E4 |
| PROD-E7 | Dashboard | 4 | 2 | 13 | DIST-E5, PROD-E1, PROD-E4 |
| PROD-E8 | Observability core: metrics, step records, tracing and logs | 1 | 8 | 8 | PLC-E2, DIST-E8, DIST-E9 |
| PROD-E9 | Doctor, diagnostic bundles, exporters and privacy | 1 | 6 | 6 | DIST-E6, DIST-E7, DIST-E9 |
| PROD-E10 | Packaging, installers and first run | 2 | 1 | 2 | QUAL-E16 |
| PROD-E11 | Backend modules | 1 | 3 | 3 | COMP-E8 |
| PROD-E12 | Services, updates, rollback and uninstall | 5 | 1 | 16 | DIST-E5, PROD-E1, QUAL-E17 |
| PROD-E13 | User documentation and community intake | 1 | 15 | 15 | PROD-E6, PROD-E7 |
| QUAL-E1 | Workspace, CI skeleton and policy checks | 2 | 1 | 1 | – |
| QUAL-E3 | Determinism foundations, testkit core and the test pyramid's base | 3 | 1 | 13 | PLC-E4, PLC-E5, DIST-E8, DIST-E9, DIST-E10, DIST-E11 |
| QUAL-E5 | Parity harness | 3 | 2 | 5 | PLC-E9, DIST-E8, COMP-E11 |
| QUAL-E6 | Execution-path coverage, family conformance and the `ConformanceManifest` | 4 | 1 | 11 | DIST-E8, DIST-E11, QUAL-E5, QUAL-E8 |
| QUAL-E7 | Fixtures and test data | 4 | 1 | 13 | DIST-E8, DIST-E9, QUAL-E8 |
| QUAL-E8 | Deterministic multi-node simulator and emulated fabric | 9 | 2 | 11 | DIST-E3, DIST-E4, DIST-E7, DIST-E9, DIST-E10, DIST-E14, PROD-E2, QUAL-E3, QUAL-E7, QUAL-E15 |
| QUAL-E9 | Fault injection and chaos | 7 | 2 | 16 | DIST-E7, DIST-E9, DIST-E14, QUAL-E8, QUAL-E15, QUAL-E17 |
| QUAL-E10 | Fuzzing | 1 | 14 | 14 | DIST-E4, DIST-E5, DIST-E8, DIST-E9, DIST-E10, DIST-E14, QUAL-E7, QUAL-E8 |
| QUAL-E11 | Platform and feature-fallback suites | 2 | 1 | 4 | DIST-E7, QUAL-E15 |
| QUAL-E12 | Benchmark harness, measurement protocol and results store | 2 | 1 | 1 | – |
| QUAL-E13 | Scenarios, success-criteria mapping and microbenchmarks | 5 | 4 | 14 | PLC-E5, DIST-E4, DIST-E5, DIST-E7, DIST-E8, DIST-E9, DIST-E10, DIST-E11, COMP-E8, COMP-E12, QUAL-E7, QUAL-E15 |
| QUAL-E14 | Regression gates, baseline comparisons and publishing | 5 | 1 | 13 | QUAL-E12, QUAL-E13 |
| QUAL-E15 | Reference hardware lab | 1 | 1 | 1 | – |
| QUAL-E16 | Build matrix, kernel cache, reproducibility, signing and provenance | 2 | 1 | 1 | – |
| QUAL-E17 | Versioning, release pipeline, compatibility and dependencies | 5 | 2 | 16 | DIST-E8, DIST-E9, QUAL-E1, QUAL-E7, QUAL-E8 |

**Spike and decision tasks in M2:** 16.

| Task | Kind | What | Answers (OQ) | Direct dependents |
|---|---|---|---|---|
| PLC-E7-T15 | spike | Spikes (RFC-0012 open questions 1, 2): pre-wake on Wi-Fi rings, kernel-read versus `CopyIn` threshold for frame input | OQ-PL-15, OQ-PL-16 | 0 |
| PLC-E9-T14 | spike | Spikes (RFC-0012 open questions 4 and 6): per-depth acceptance curves for tree templates from M2 dogfooding; the cost of per-row penalties… | OQ-PL-18 | 0 |
| TIER-E3-T08 | spike | Spike: how often RFC-0011's delta solves want to grow the KV hot window at the expert cache's expense | OQ-IO-15 | 0 |
| DIST-E5-T17 | spike | Spike: pairing usability with number matching, with 5 alpha testers over 2 days (with PROD) | – | 0 |
| DIST-E7-T15 | spike | Spike: α_eff through the frame path on TB5 RDMA (GPU buffer → data frame at the collective partial size → GPU buffer, two Macs, doorbell… | OQ-NET-01 | 0 |
| DIST-E7-T16 | spike | Spike: QUIC versus TCP α on LAN, per-frame credit cost on Wi-Fi, and what the iroh transport knobs that DIST-E7-T21 found reachable gain… | OQ-NET-05, OQ-NET-17, OQ-NET-19 | 0 |
| DIST-E7-T19 | benchmark | Benchmark: Thunderbolt and USB4 IP networking on Windows 11, for which no data exists | OQ-NET-02 | 0 |
| DIST-E11-T19 | spike | Spike, mid M2: pipeline decomposition on the S2 cluster over 2.5 GbE and Wi-Fi, giving SC-12's `Z`, SC-13's hop shares, the tail → head… | – | 0 |
| COMP-E7-T08 | spike | Spike for RFC-0040 open question 5: upstream acceptance of P1–P4 over the first two pin bumps after M1 | OQ-KB-03 | 0 |
| COMP-E12-T08 | spike | Spike for RFC-0041 open question 1: the Hub's chunked-storage protocol (reconstruction information, authentication, hash scheme) against… | OQ-MF-05 | 1 |
| COMP-E14-T17 | spike | Spike for RFC-0042 open question 3: which families pass BF16 on the wire, from QUAL-E6-T11's runs. Exit: the M2 `WireCert` list | OQ-FID-01 | 0 |
| PROD-E1-T17 | spike | Spike: conflicting proposals (open question 5), from M2 user testing | OQ-PRD-15 | 0 |
| PROD-E4-T10 | spike | Embeddings for the v1 encoder-only family, `bert` (RFC-0050 open question 6, answered): the endpoint serves it on a single node only, with… | OQ-PRD-12 | 0 |
| PROD-E10-T21 | spike | Spike: OpenPGP repository signing from a KMS key, before the deb and rpm repositories ship | – | 1 |
| QUAL-E8-T14 | spike | Spike: simulator fidelity | OQ-PRC-10 | 0 |
| QUAL-E12-T14 | spike | Energy per token: NVML power, readable RAPL counters, laptop battery discharge rate; J per generated token, or "not available"; plus a… | OQ-PRC-14 | 0 |

**Open questions needed by M2:** 26 (benchmark 8, decision 5, spike 13). Answered by no task: OQ-PRC-19.
<!-- END GENERATED -->

## M3: feature complete

### Goal and exit criteria

<!-- BEGIN GENERATED: roadmap-m3-goal -->
Quoted from RFC-0001 §16, "M3: feature complete":

> - Invite links with relays and hole punching, the built-in relay set read from a signed `relays.json`; Tailscale paths (RFC-0030, RFC-0034).
> - Fault tolerance: activation-history replay from every sending stage, request migration, in-place relink and shard extension, KV migration on re-placement and recovery, an evictable second copy of each stage's source chunks on the node a repair would absorb it into, a 60 s detach grace that lets a client re-attach to a paused stream, degraded modes, automatic re-placement (RFC-0033, RFC-0022, RFC-0050).
> - Cross-node TP/EP on links that pass the α gate; RoCEv2 behind `trusted_fabric` (RFC-0032, RFC-0031). If M1b's probe measures a raw-path `α_eff` above 65 µs between the Macs, SC-17 is report-only at this exit and Mac-pair TP moves to M4 or after 1.0.
> - Speculative decoding across stages; trained draft heads loaded where a model ships them, though v1 produces none; wide trees for I/O-bound targets; asynchronous speculation behind a flag (RFC-0012, ADR-0017).
> - Runner re-adoption across daemon updates (§3.4).
> - The `Compressed` KV class with the DeepSeek-V4-class family (RFC-0022, RFC-0043).
> - HIP and Vulkan native providers; ROCm and Vulkan modules (RFC-0040).
> - Throughput mode with continuous batching (RFC-0011, RFC-0012).
> - Stored Responses, kept on the ingress only, and the local-runner API shape (RFC-0050).
> - Dashboard and API surfaces for invites, admissions and recovery events, and the opt-in approximate settings, each labelled (RFC-0051, RFC-0050, RFC-0054).
> - Every v1 family from ADR-0003, each through the conformance suite (RFC-0043, RFC-0060).
> - Exit: SC-15, SC-16, SC-17, SC-26 and SC-28, with SC-17 report-only under the Mac-pair rule above; the chaos suite of RFC-0060 passes (node kill mid-generation completes the request).

Success criteria first required at this exit: SC-15, SC-16, SC-17, SC-26, SC-28.
<!-- END GENERATED -->

### Plan

<!-- BEGIN GENERATED: roadmap-m3 -->
**Tasks:** 211 (PLC 24, TIER 23, DIST 64, COMP 46, PROD 27, QUAL 27).

**Critical path inside M3:** 9 tasks, assuming every earlier milestone is done. 18 M3 tasks lie on a chain of that length (no slack); the chain below is one of them.

| Step | Task | What |
|---|---|---|
| 1 | DIST-E13-T13 | `TpSplit`, `CollectiveAlgo` and `ExpertOwnerMap` type definitions (RFC-0032 §5) with serde, split out of DIST-E13-T01… |
| 2 | COMP-E15-T08 | `TpSharding` and splits for `llama`, `qwen_moe`, `deepseek3` and `gptoss`: per-block rules and granularities (KV-head… |
| 3 | DIST-E13-T01 | `TpSplit`, `CollectiveAlgo` and `ExpertOwnerMap` types, with validation against the family's legal granularities (whole… |
| 4 | DIST-E13-T07 | Intra-node TP through provider `CollectiveOps`, with a verified P2P test copy in each direction at `Warming` and a… |
| 5 | TIER-E8-T10 | Single ownership inside a node for helper-GPU caches (owner word set by atomic compare-exchange) and identical… |
| 6 | TIER-E8-T11 | Cluster ownership under `CrossNodeEp`: `ExpertOwnerMap`, explicit hot replicas from the plan, and the fixed… |
| 7 | TIER-E9-T11 | Owner-side expert step under `CrossNodeEp`: rows arriving in EP dispatch frames for experts this node owns run through… |
| 8 | DIST-E13-T06 | Cross-node EP: `EpDispatch` to owners with selected experts, owner compute through its expert cache, `EpCombine` summed… |
| 9 | TIER-E8-T15 | Spike: when replicating a hot expert on a second node beats a single owner |

Zero-slack tasks by area: TIER 5, DIST 12, COMP 1.

**Longest chain from project start to an M3 task:** 47 tasks: QUAL-E1-T01 (M0a) → QUAL-E12-T01 → QUAL-E15-T01 → QUAL-E15-T02 → QUAL-E12-T04 → QUAL-E2-T01 → QUAL-E2-T04 → QUAL-E2-T11 → QUAL-E15-T16 → PLC-E1-T04 → PLC-E1-T07 → PLC-E1-T09 → PLC-E4-T02 → PLC-E7-T02 → PLC-E7-T03 → PLC-E7-T04 → PLC-E9-T02 → PLC-E9-T16 → PROD-E4-T01 → QUAL-E12-T02 → QUAL-E12-T03 → QUAL-E2-T02 → QUAL-E4-T06 → QUAL-E6-T04 → COMP-E16-T02 → COMP-E16-T14 → COMP-E16-T06 (M1a) → COMP-E16-T07 → QUAL-E7-T09 → TIER-E8-T02 → TIER-E10-T07 → PLC-E4-T03 → PLC-E4-T04 → PLC-E4-T06 → PLC-E5-T06 → PLC-E6-T02 → PLC-E6-T19 → PROD-E2-T08 → PROD-E2-T13 (M2) → QUAL-E8-T08 → QUAL-E8-T09 → QUAL-E6-T08 → QUAL-E6-T11 → COMP-E14-T05 → COMP-E14-T19 (M3) → PROD-E1-T18 → PROD-E1-T20.

**Parallel work.** Tasks per step, where a task's step is 1 + its longest chain of prerequisites inside the milestone: 1: 64, 2: 36, 3: 35, 4: 22, 5: 12, 6: 18, 7: 17, 8: 5, 9: 2. 64 tasks can start as soon as M3 opens.

Work streams are the epics with tasks in this milestone. "First step" and "last step" are the steps of the stream's earliest and latest tasks. Streams whose step ranges overlap, and that do not wait on each other, can run in parallel.

| Stream | Epic | Tasks | First step | Last step | Waits on (M3 streams) |
|---|---|---|---|---|---|
| PLC-E1 | Resource graph types and hardware enumeration | 1 | 2 | 2 | COMP-E8 |
| PLC-E2 | Startup microbenchmark suite | 2 | 3 | 6 | DIST-E7, DIST-E13, COMP-E8 |
| PLC-E3 | Profile cache, online refinement and drift detection | 1 | 1 | 1 | – |
| PLC-E4 | Solver foundations: inputs, preprocessing, filter, evaluator and DP | 2 | 2 | 7 | PLC-E2, DIST-E13, DIST-E14 |
| PLC-E5 | Joint MILP on HiGHS | 3 | 1 | 8 | PLC-E4 |
| PLC-E6 | Plan assembly, re-solve, routing and explanations | 3 | 1 | 8 | PLC-E4, COMP-E14, QUAL-E6 |
| PLC-E7 | Executor: step programs, dispatchers and graph capture | 2 | 3 | 4 | DIST-E11, DIST-E14, COMP-E8 |
| PLC-E8 | Step scheduling: prefill, decode, throughput mode and admission | 3 | 1 | 2 | – |
| PLC-E9 | Sampling and speculative decoding | 7 | 1 | 3 | TIER-E6, TIER-E8, COMP-E5 |
| TIER-E2 | Host memory: budgets, pin classes, progressive pinning and unified memory | 2 | 1 | 3 | COMP-E8 |
| TIER-E3 | VRAM split, elastic expert segments and the runtime n* rule | 1 | 3 | 3 | COMP-E8 |
| TIER-E4 | Memory pressure and priced yield | 1 | 1 | 1 | – |
| TIER-E6 | KV store, paging and tiering | 4 | 1 | 6 | COMP-E3, COMP-E5, COMP-E16 |
| TIER-E7 | Prefix cache, saved sessions, lossy KV and migration | 4 | 3 | 7 | DIST-E8, DIST-E11, DIST-E14, COMP-E5, QUAL-E6 |
| TIER-E8 | Expert cache: slots, policy, slot table and profile | 4 | 1 | 9 | DIST-E13 |
| TIER-E9 | The MoE step: three sites, CPU execute-in-place and the balancer | 3 | 3 | 7 | TIER-E2, TIER-E8, DIST-E8, COMP-E8 |
| TIER-E10 | Prediction, prefetch, solver curves and approximate knobs | 4 | 3 | 4 | DIST-E5, DIST-E7, DIST-E8, COMP-E15, QUAL-E6 |
| DIST-E1 | Process model: rings, IPC, runner spawn, lifecycle and supervision | 1 | 1 | 1 | – |
| DIST-E2 | Identity, keys and signed structures | 1 | 1 | 1 | – |
| DIST-E4 | Membership and failure detection | 4 | 1 | 3 | DIST-E1, DIST-E5 |
| DIST-E5 | Admission, invites, revocation and key rotation | 5 | 1 | 2 | DIST-E2, DIST-E7 |
| DIST-E7 | Transports, channels and link management | 2 | 1 | 2 | DIST-E17 |
| DIST-E8 | Wire formats and codecs | 4 | 1 | 4 | COMP-E14 |
| DIST-E10 | Coordinator election and split heal | 1 | 1 | 1 | – |
| DIST-E11 | Ring pipeline execution | 5 | 1 | 4 | PLC-E8, DIST-E8, DIST-E14 |
| DIST-E13 | Cross-node TP and EP, and intra-node TP | 12 | 1 | 8 | TIER-E9, DIST-E8, DIST-E14, COMP-E3, COMP-E5, COMP-E6, COMP-E8, COMP-E15 |
| DIST-E14 | Fault tolerance: detection, history, replay, migration and repair | 17 | 1 | 7 | PLC-E4, PLC-E7, PLC-E9, TIER-E2, TIER-E6, TIER-E7, DIST-E8, DIST-E11, COMP-E11 |
| DIST-E15 | Elasticity: stragglers, joins and coordinator co-failure | 4 | 5 | 7 | DIST-E11, DIST-E14 |
| DIST-E16 | Distributed chaos suite | 6 | 2 | 9 | DIST-E10, DIST-E13, DIST-E14 |
| DIST-E17 | Security hardening: encryption policy, integrity, signing hierarchy and privacy | 1 | 1 | 1 | – |
| DIST-E18 | Configuration keys, error codes, metrics and diagnostics | 1 | 1 | 1 | – |
| COMP-E1 | Kernel provider core | 2 | 1 | 2 | COMP-E16 |
| COMP-E2 | Op selection, certification, coverage and kernel targets | 2 | 1 | 4 | COMP-E5 |
| COMP-E3 | CPU provider | 3 | 1 | 5 | COMP-E1, COMP-E5 |
| COMP-E5 | CUDA provider: kernels | 7 | 1 | 4 | COMP-E1, COMP-E15, COMP-E16, QUAL-E6 |
| COMP-E6 | Metal provider | 4 | 1 | 3 | COMP-E1, COMP-E15, COMP-E16 |
| COMP-E7 | ggml provider | 2 | 2 | 3 | COMP-E8, COMP-E9 |
| COMP-E8 | HIP and Vulkan providers | 8 | 1 | 4 | – |
| COMP-E9 | Kernel builds, provider libraries and the module ABI | 2 | 1 | 5 | QUAL-E16 |
| COMP-E11 | Model store and packs | 3 | 1 | 2 | – |
| COMP-E12 | Weight distribution: Hub, chunk tables and peer-to-peer transfer | 2 | 1 | 1 | – |
| COMP-E14 | The exactness contract: tolerances, wire, canary, knobs and requantization | 5 | 1 | 3 | PLC-E9, COMP-E5, QUAL-E6 |
| COMP-E15 | Graph IR, blocks, segments and tensor-parallel sharding | 2 | 2 | 2 | DIST-E13, COMP-E16 |
| COMP-E16 | Family modules, classification and certification | 3 | 1 | 4 | COMP-E3, COMP-E5, COMP-E6, COMP-E15 |
| COMP-E17 | Tokenizers and chat formats | 1 | 5 | 5 | COMP-E16 |
| PROD-E1 | Configuration and policy | 2 | 6 | 7 | TIER-E7, COMP-E14 |
| PROD-E2 | API core, vendor shapes, routing and streaming | 4 | 2 | 7 | DIST-E14, PROD-E4 |
| PROD-E3 | Tool calling, structured output and reasoning | 1 | 2 | 2 | PLC-E9 |
| PROD-E4 | Native API, fidelity labels, debug surface, sessions and embeddings | 3 | 1 | 2 | DIST-E5 |
| PROD-E5 | API authentication, bind policy and LAN TLS | 1 | 2 | 2 | DIST-E2 |
| PROD-E6 | CLI and the model picker | 5 | 1 | 3 | DIST-E5, COMP-E14, PROD-E11 |
| PROD-E7 | Dashboard | 2 | 2 | 3 | DIST-E5, PROD-E4 |
| PROD-E8 | Observability core: metrics, step records, tracing and logs | 1 | 2 | 2 | DIST-E1 |
| PROD-E9 | Doctor, diagnostic bundles, exporters and privacy | 1 | 1 | 1 | – |
| PROD-E10 | Packaging, installers and first run | 2 | 3 | 4 | PROD-E11, PROD-E12 |
| PROD-E11 | Backend modules | 2 | 2 | 3 | COMP-E9 |
| PROD-E12 | Services, updates, rollback and uninstall | 2 | 1 | 3 | DIST-E1, DIST-E4, PROD-E8, QUAL-E17 |
| PROD-E13 | User documentation and community intake | 1 | 4 | 4 | PROD-E2, PROD-E6 |
| QUAL-E2 | Runner fleet and hardware-job security | 1 | 2 | 2 | QUAL-E15 |
| QUAL-E4 | Reference executor and kernel conformance | 1 | 4 | 4 | COMP-E8, QUAL-E2 |
| QUAL-E6 | Execution-path coverage, family conformance and the `ConformanceManifest` | 6 | 1 | 7 | DIST-E13, COMP-E14, PROD-E6, QUAL-E2 |
| QUAL-E9 | Fault injection and chaos | 2 | 6 | 7 | DIST-E14, QUAL-E17 |
| QUAL-E10 | Fuzzing | 1 | 2 | 2 | DIST-E5 |
| QUAL-E11 | Platform and feature-fallback suites | 2 | 4 | 5 | TIER-E2, QUAL-E2, QUAL-E4 |
| QUAL-E13 | Scenarios, success-criteria mapping and microbenchmarks | 5 | 3 | 7 | PLC-E8, PLC-E9, DIST-E5, DIST-E7, DIST-E11, DIST-E13, DIST-E14, QUAL-E9, QUAL-E15 |
| QUAL-E14 | Regression gates, baseline comparisons and publishing | 1 | 1 | 1 | – |
| QUAL-E15 | Reference hardware lab | 3 | 1 | 3 | DIST-E5 |
| QUAL-E16 | Build matrix, kernel cache, reproducibility, signing and provenance | 4 | 1 | 4 | COMP-E8 |
| QUAL-E17 | Versioning, release pipeline, compatibility and dependencies | 1 | 2 | 2 | DIST-E1 |

**Spike and decision tasks in M3:** 13.

| Task | Kind | What | Answers (OQ) | Direct dependents |
|---|---|---|---|---|
| PLC-E3-T11 | spike | Spikes: Metal timestamp perturbation (open question 8) with COMP, and the wireless tail prior and C14's 300 ms and 0.01 defaults fitted to… | OQ-NET-03, OQ-KB-19 | 0 |
| PLC-E6-T12 | spike | Spikes and follow-ups from RFC-0011's open questions 4–7: latency-mode KV reserve sizing with TIER; the recurrent-state move cost inside… | OQ-PL-11, OQ-PL-13 | 0 |
| PLC-E8-T12 | spike | Spike: prefill share in throughput mode (RFC-0012 open question 3), by queueing simulation on recorded arrival traces | OQ-PL-17 | 0 |
| TIER-E8-T15 | spike | Spike: when replicating a hot expert on a second node beats a single owner | OQ-MOE-05 | 0 |
| DIST-E13-T10 | spike | Spike: concurrency of 3 parallel TB5 RDMA sends per node in a 4-Mac full mesh | OQ-NET-21 | 0 |
| DIST-E14-T16 | spike | Spikes: history footprint in throughput mode (S5), the rate of false recovery starts on flapping links, fed with DIST-E4-T16's field traces… | OQ-NET-23, OQ-NET-24, OQ-NET-26 | 0 |
| DIST-E16-T06 | benchmark | Benchmark: ADR-0312's `t_resume` predictions against measured recoveries, over the simulator scenarios and the hardware subset | OQ-NET-28 | 0 |
| COMP-E8-T07 | spike | Spike for RFC-0040 open question 8: HIP against Vulkan for decode and prefill on RDNA3 and RDNA4 once both have native kernels | OQ-KB-17 | 0 |
| COMP-E14-T18 | spike | Spike for RFC-0042 open question 6: reference cost on 1T-class models under sampled layers | – | 0 |
| COMP-E16-T18 | decision | Decision on DeepSeek-V4's compressed attention (OQ-MF-16 (a)–(c)), before any task builds the `Compressed` block or op: revise RFC-0040… | OQ-MF-16 | 5 |
| PROD-E6-T13 | spike | Spike: hit-rate ranges for NVMe-dependent MoE (open question 3) | OQ-PRD-18 | 0 |
| PROD-E11-T08 | decision | Decision on ROCm user-space libraries (RFC-0052 open question 6): the licence part is answered (no obstacle; RCCL, ROCgdb and binary-only… | OQ-PKG-10 | 0 |
| QUAL-E6-T15 | spike | Spike: strength-2 inference | OQ-PRC-11 | 0 |

**Open questions needed by M3:** 23 (benchmark 13, check 1, decision 3, spike 6). Each one names a task that records the answer.
<!-- END GENERATED -->

## M4: hardening and 1.0

### Goal and exit criteria

<!-- BEGIN GENERATED: roadmap-m4-goal -->
Quoted from RFC-0001 §16, "M4: hardening and 1.0":

> - Tier 2 and Tier 3 platforms per ADR-0020; signed installers and update channels for every Tier 1 OS (RFC-0052, RFC-0062).
> - Performance regression gates on every Tier 1 rig; published benchmarks tied to commits, comparisons with other engines included (RFC-0061).
> - Wire-protocol and pack-format compatibility policy in force (RFC-0062).
> - Complete observability, and an external security audit of admission, transport credentials, the update client, the local API CA and API authentication, required before the 1.0 release candidate (RFC-0053, RFC-0034, RFC-0062 §13).
> - Exit: every SC target met on its rig, or waived in writing by a maintainer with the measurement attached; zero open P0 defects.

Success criteria first required at this exit: every SC not already required earlier, which is SC-30, SC-31, SC-32; and the other 29 remain required.
<!-- END GENERATED -->

### Plan

<!-- BEGIN GENERATED: roadmap-m4 -->
**Tasks:** 51 (PLC 3, TIER 4, DIST 13, COMP 10, PROD 11, QUAL 10).

**Critical path inside M4:** 6 tasks, assuming every earlier milestone is done. 6 M4 tasks lie on a chain of that length (no slack); the chain below is one of them.

| Step | Task | What |
|---|---|---|
| 1 | COMP-E9-T14 | Artefact assembly for M4: the `sycl` provider library and the static iOS and Android builds |
| 2 | COMP-E7-T15 | ggml OpenCL and Hexagon backends in the Android app's static build, with the CPU and Vulkan backends RFC-0040 §8.1… |
| 3 | PROD-E10-T13 | Android Tier 3: AAR with uniffi Kotlin bindings and the embedded runner; CPU, Vulkan, OpenCL and Hexagon backends… |
| 4 | PROD-E10-T16 | iOS Tier 3: XCFramework with uniffi Swift bindings, static Metal and CPU providers, embedded runner, memory cap from… |
| 5 | PLC-E1-T13 | Tier 2/3 enumeration: Android, iOS in-process runner, Linux arm64 boards; `HostPageable`-only tiers on mobile |
| 6 | TIER-E2-T17 | Android and iOS `HostPageable`-only adapters with `os_proc_available_memory` and the in-process runner host |

Zero-slack tasks by area: PLC 1, TIER 1, COMP 2, PROD 2.

**Longest chain from project start to an M4 task:** 49 tasks: QUAL-E1-T01 (M0a) → QUAL-E12-T01 → QUAL-E15-T01 → QUAL-E15-T02 → QUAL-E12-T04 → QUAL-E2-T01 → QUAL-E2-T04 → QUAL-E2-T11 → QUAL-E15-T16 → PLC-E1-T04 → PLC-E1-T07 → PLC-E1-T09 → PLC-E4-T02 → PLC-E7-T02 → PLC-E7-T03 → PLC-E7-T04 → PLC-E9-T02 → PLC-E9-T16 → PROD-E4-T01 → QUAL-E12-T02 → QUAL-E12-T03 → QUAL-E2-T02 → QUAL-E4-T06 → QUAL-E6-T04 → COMP-E16-T02 → COMP-E16-T14 → COMP-E16-T06 (M1a) → COMP-E16-T07 → QUAL-E7-T09 → TIER-E8-T02 → TIER-E10-T07 → PLC-E4-T03 → PLC-E4-T04 → PLC-E4-T06 → PLC-E5-T06 → PLC-E6-T02 → PLC-E6-T13 → TIER-E6-T15 → TIER-E6-T17 → PROD-E4-T03 → QUAL-E6-T16 → QUAL-E6-T23 (M2) → COMP-E8-T10 → QUAL-E13-T06 → QUAL-E13-T16 (M3) → QUAL-E13-T07 → QUAL-E14-T06 (M4) → QUAL-E17-T19 → QUAL-E17-T23.

**Parallel work.** Tasks per step, where a task's step is 1 + its longest chain of prerequisites inside the milestone: 1: 36, 2: 9, 3: 3, 4: 1, 5: 1, 6: 1. 36 tasks can start as soon as M4 opens.

Work streams are the epics with tasks in this milestone. "First step" and "last step" are the steps of the stream's earliest and latest tasks. Streams whose step ranges overlap, and that do not wait on each other, can run in parallel.

| Stream | Epic | Tasks | First step | Last step | Waits on (M4 streams) |
|---|---|---|---|---|---|
| PLC-E1 | Resource graph types and hardware enumeration | 1 | 5 | 5 | PROD-E10 |
| PLC-E6 | Plan assembly, re-solve, routing and explanations | 1 | 2 | 2 | QUAL-E14 |
| PLC-E8 | Step scheduling: prefill, decode, throughput mode and admission | 1 | 1 | 1 | – |
| TIER-E2 | Host memory: budgets, pin classes, progressive pinning and unified memory | 1 | 6 | 6 | PLC-E1, PROD-E10 |
| TIER-E5 | Weight I/O engine | 1 | 1 | 1 | – |
| TIER-E8 | Expert cache: slots, policy, slot table and profile | 1 | 1 | 1 | – |
| TIER-E10 | Prediction, prefetch, solver curves and approximate knobs | 1 | 2 | 2 | DIST-E17 |
| DIST-E1 | Process model: rings, IPC, runner spawn, lifecycle and supervision | 1 | 1 | 1 | – |
| DIST-E3 | LAN discovery | 1 | 1 | 1 | – |
| DIST-E4 | Membership and failure detection | 1 | 1 | 1 | – |
| DIST-E7 | Transports, channels and link management | 3 | 1 | 1 | – |
| DIST-E11 | Ring pipeline execution | 1 | 1 | 1 | – |
| DIST-E16 | Distributed chaos suite | 1 | 1 | 1 | – |
| DIST-E17 | Security hardening: encryption policy, integrity, signing hierarchy and privacy | 5 | 1 | 1 | – |
| COMP-E7 | ggml provider | 3 | 1 | 2 | COMP-E9 |
| COMP-E8 | HIP and Vulkan providers | 1 | 1 | 1 | – |
| COMP-E9 | Kernel builds, provider libraries and the module ABI | 2 | 1 | 2 | QUAL-E16 |
| COMP-E11 | Model store and packs | 1 | 1 | 1 | – |
| COMP-E14 | The exactness contract: tolerances, wire, canary, knobs and requantization | 2 | 1 | 1 | – |
| COMP-E17 | Tokenizers and chat formats | 1 | 1 | 1 | – |
| PROD-E1 | Configuration and policy | 1 | 1 | 1 | – |
| PROD-E4 | Native API, fidelity labels, debug surface, sessions and embeddings | 2 | 1 | 1 | – |
| PROD-E7 | Dashboard | 1 | 1 | 1 | – |
| PROD-E10 | Packaging, installers and first run | 4 | 1 | 4 | DIST-E1, COMP-E7, COMP-E9 |
| PROD-E11 | Backend modules | 1 | 3 | 3 | COMP-E7, COMP-E9 |
| PROD-E12 | Services, updates, rollback and uninstall | 1 | 2 | 2 | QUAL-E9 |
| PROD-E13 | User documentation and community intake | 1 | 1 | 1 | – |
| QUAL-E9 | Fault injection and chaos | 1 | 1 | 1 | – |
| QUAL-E10 | Fuzzing | 1 | 1 | 1 | – |
| QUAL-E11 | Platform and feature-fallback suites | 1 | 1 | 1 | – |
| QUAL-E14 | Regression gates, baseline comparisons and publishing | 1 | 1 | 1 | – |
| QUAL-E15 | Reference hardware lab | 1 | 1 | 1 | – |
| QUAL-E16 | Build matrix, kernel cache, reproducibility, signing and provenance | 1 | 1 | 1 | – |
| QUAL-E17 | Versioning, release pipeline, compatibility and dependencies | 4 | 2 | 3 | QUAL-E10, QUAL-E14 |

**Spike and decision tasks in M4:** 3.

| Task | Kind | What | Answers (OQ) | Direct dependents |
|---|---|---|---|---|
| DIST-E7-T17 | benchmark | Benchmark: CRC32C and AEAD throughput on Raspberry Pi 5-class CPUs, with the hardware CRC32C instructions and 3-way interleaving already… | – | 0 |
| DIST-E11-T16 | spike | Spike: prefill/decode disaggregation on rig (c) (a RAM copy of the remote layers on the GPU node, plus KV shipping) | OQ-PL-13 | 0 |
| DIST-E17-T07 | spike | Spike, then implementation: OpenSSF Model Signing verification (`models.signatures = off \| verify_if_present \| require`… | OQ-SEC-03 | 0 |

**Open questions needed by M4:** 5 (benchmark 1, decision 1, spike 3). Each one names a task that records the answer.
<!-- END GENERATED -->
