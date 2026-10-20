# ADR-0602: Execution-path coverage by covering arrays, certified in a signed conformance manifest
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0002, ADR-0003, ADR-0014, ADR-0019, ADR-0020; RFC-0001 §5.4, §5.7; RFC-0011, RFC-0040, RFC-0043, RFC-0060 §6, RFC-0062

## Context
ADR-0019 §4 says every exact-mode execution path the solver or scheduler can choose must pass parity, and that classification is by measurement. The path space is large:
- provider per op class;
- placement shape (single device, execute-in-place splits, tiered experts, 2- and 3-stage rings, intra- and cross-node TP/EP, degraded multi-stage);
- wire dtype;
- speculative mode and window;
- concurrency, prefix and session reuse, and KV tier;
- prefill chunk relative to n\*;
- graph capture and the deterministic option;
- force mode (prefill, decode, verify).

That is thousands of combinations per family and quantization. Two further questions follow. The runtime needs to know which paths passed. And certificates must apply to the bits that ship, which a rebuild after testing would not be.

## Options considered
1. **Exhaustive testing of the product space.** Infeasible.
2. **Random sampling of paths per run.** No pairwise guarantee, results differ run to run, and an RC is not reproducible.
3. **Test only the paths the solver currently chooses on the lab rigs.** Cheap, but blind to paths the solver will pick on users' hardware.
4. **Chosen: seeded covering arrays (strength 2 nightly, strength 3 for Tier 1 RCs), plus the always-tested shipping set (the plans the solver actually chooses on each rig).** Results go into a manifest the runtime reads.

On delivery of certificates:
- **(a) Compile them into the binary.** They would then describe a build other than the one tested.
- **(b) A signed sidecar keyed by content hashes of the code that determines numerics.** Chosen.

## Decision
1. **The path type.** `ExecPath` (RFC-0060 §6.1) is the unit of certification. Transport is excluded because the wire codec is bit-exact. Hardware feature versus fallback is handled per feature on hardware that has it (RFC-0060 §11.3).
2. **Coverage.**
   - Arrays come from a constrained greedy in-parameter-order generator, seeded by (release line, family), so an RC always tests the same paths.
   - Every PR† runs the shipping set, nightly runs strength 2, and RCs run strength 3 on Tier 1.
   - A path the solver chooses on any lab rig joins the shipping set from then on.
3. **Classification.**
   - Fail → no certificate → never chosen in exact mode.
   - Untested → **uncertified**, and allowed only if every factor pair it contains is certified (the strength-2 coverage assumption, stated as a known risk). A pair is certified only if it occurs in a passing path and in no failing path, and a failure makes the failing path's pairs and triples suspect until bisection localises it.
   - Run-time dimensions (prefix reuse, KV tier, concurrency, verify windows) are enforced per instance by the scheduler and the tier manager, which refuse values not certified for the plan's path and report the degradation. Plan-time dimensions are enforced by the solver.
   - A certificate is restored only by fixing the path. Labelling it approximate requires an ADR-0019 knob.
4. **`ConformanceManifest`** (RFC-0060 §6.4; RFC-0001 amendment A2) holds `FamilyCert`, `OpCert`, `WireCert`, `PathCert` and `KnobCost` entries.
   - Certificates are **content-keyed**: a family by the BLAKE3 of its module and shared-block sources, an op by the BLAKE3 of its compiled kernel object.
   - A certificate therefore survives builds that leave those bytes unchanged, and a change invalidates exactly what it touches.
5. **Consumers:**
   - the family registry (fail closed, ADR-0003 §6);
   - op selection (only certified ops per device class);
   - the solver (`WireCert` for BF16 hops; `FamilyCert` for splits and TP; `PathCert` for shapes);
   - the API and dashboard (certification shown per model and backend);
   - approximate-knob offers (`KnobCost`, required by ADR-0019 §6).
6. **Production and signing.**
   - An RC's manifest is generated from that RC's own runs on its exact signed artefacts (RFC-0060 §1.3), plus carried-over entries recorded with the build they were measured on.
   - **Three signers**:
     - the release key for `beta` and `stable` manifests;
     - the nightly key for `nightly` manifests;
     - a CI test key, accepted by any build only while a `debug.parity_api` or `debug.bench_api` gate is set and the API listeners are loopback-only (cluster listeners unaffected).
   - **Pipeline order** for PR† jobs and nightly: L4 certificates, then an interim manifest signed with the CI test key, then L5 parity, then the final manifest, then benchmarks. Benchmarks therefore measure native kernels before any release manifest exists.
   - **M0 scope.** A minimal manifest (`OpCert`, `FamilyCert`) lands in M0, so `OpCert`-gated selection and family refusal work from the first milestone.
   - A release build with no valid manifest serves exact mode only on ggml ops, plus the CPU provider's native ops for the ops ggml cannot run (RFC-0040 §6.2), on single-device shapes and the F32 wire, labelled "uncertified".
   - Kernels using 8-bit activations on `PackedInt` tensors that RFC-0042 §3.5 cannot transcribe exactly need a `PathCert` from M1 (RFC-0042 §4.3 rule c).
7. **Derived certificates** (added 2026-10-19). For Tier 1 NVIDIA targets no lab rig can execute (`sm_75`, `sm_90`, `sm_100` today), an op with no `OpCert` for that class is selectable in exact mode when three conditions hold:
   - an `OpCert` with the same provider, `OpKey`, shape class and `source_key` exists on another class;
   - the binary is compatible under CUDA's same-major rule, or is JIT-compiled from identical PTX;
   - the warm-up canary on first load matches the reference digest.

   Output is labelled `exact (derived certification)`. A failed canary drops the op to a fully certified fallback, or, if none exists, makes the path approximate. *Revised 2026-10-20.* ADR-0610 adds a Turing rig (R-TURING), so `sm_75` gets full certificates, and an on-demand H100- and B200-class canary on every RC from the first M1 RC (ADR-0610 decision 8); `sm_90` and `sm_100` stay derived between RCs (RFC-0060 §6.4).

8. **`FamilyCert` needs the publisher's reference** (added 2026-10-20). Every other check compares candidates with Syntropy's own family graph, so a family module that computes the wrong model (a RoPE scaling, a norm placement, a routing rule) would pass them all. A `FamilyCert` is therefore issued only when `graph::reference`, driven by `RefExecutor(F32Fixed)` with declared activation quantisation off on the family's conformance checkpoint, falls within `τ_def` of golden logits and residual-stream norms that the publisher's own reference implementation produced offline in F32 on the same dequantised weights (RFC-0042 §7.1; RFC-0043 §9.2; RFC-0060 §7.2 check 9). The certificate records the pass (`golden`) and the longest certified context (`max_ctx`). The check runs from M0a, when the parity harness becomes a gate, and re-runs on every change to the family module or a shared block it uses. Planted module-level defects prove that it catches what the other checks cannot (RFC-0060 "Testing and validation" item 1).

## Consequences
- **Positive.**
  - ADR-0019's "classification by measurement" becomes an interface the solver obeys, not a policy that relies on memory.
  - Coverage cost grows with the number of dimension values, roughly quadratically for strength 2, not with their product.
  - Small changes keep most certificates, so nightly hardware time goes to what changed.
  - Users can see what is certified for their model and backend.
- **Negative.**
  - The strength-2 assumption can miss a three-factor interaction on a path that no array covered. Strength 3 on RCs and the shipping-set rule reduce this, but do not remove it.
  - A manifest adds a release artefact that must be signed, shipped and checked at start, and its absence degrades exact mode.
  - Content hashing of family sources ties certificates to source text, so a whitespace-only edit invalidates them. A formatter-normalised hash would avoid this and is left to implementation.

## What would make us revisit
- A derived certificate's canary fails on a user's device, or a derived path later fails parity when a rig for that architecture is added. Then require a rig, or a hosted runner of that class, before certifying that architecture.
- A strength-3 array on one Tier 1 rig, run nightly for two RCs, finds a failing path that strength 2 classed as uncertified-but-allowed. Then raise nightly strength or forbid uncertified paths in exact mode.
- Manifest delivery causes start-up failures in the field (missing or corrupt sidecar) more than once per release. Then embed a fallback copy in the binary as well.
- The solver's need to evaluate pair coverage at plan time measurably slows solves past SC-31's 2 s budget.
- A family's publisher publishes no runnable reference implementation. Then that family certifies only against an independent implementation that maintainers approve, and its certificate says so.

## Revisions
- Revised 2026-10-20: decision 8 added. Every other check compares candidates with Syntropy's own family graph, so a family module that computes the wrong model would pass them all and ship labelled exact. `FamilyCert` is now gated on golden logits and residual-stream norms from the publisher's F32 reference implementation, within `τ_def`, from M0a, re-run on module and shared-block changes, with planted module-level defects. Decision 7 records the Turing rig and the per-RC canary for architectures without a timed rig.
