# ADR-0603: Test fixtures: generated synthetic models plus a content-addressed fixture store
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0003, ADR-0005, ADR-0016, ADR-0018, ADR-0019; RFC-0041, RFC-0042, RFC-0043, RFC-0060 §12, RFC-0062

## Context
Conformance and parity need model fixtures for every v1 family (ADR-0003), in every quantization class each family ships.
- Some must be small enough for every PR on a hosted CPU.
- Some must be realistic enough to exercise real kernel tiling and vocabulary sizes.
- Some must be the real checkpoints users run.

Reference outputs (RFC-0060 §5.3), cached F64 op outputs, routing traces and fuzz corpora add gigabytes more.

Three problems follow:
- Real checkpoints are large, and some are gated behind licence acceptance.
- Random-weight models produce near-uniform logits, on which top-1 agreement means nothing.
- Fork PRs must not reach credentials for gated models.

## Options considered
1. **Real checkpoints only.** Realistic, but too large for hosted PR runs. Families with no small public checkpoint cannot be tested per PR at all.
2. **Commit synthetic fixtures and references to git, through LFS.** Simple to fetch, but large binary history and LFS bandwidth quotas would bind CI. Every clone also inherits gigabytes of reach.
3. **Generate everything on every run.** No storage, but reference computation and checkpoint quantization would repeat per run.
4. **Chosen: synthetic fixtures generated deterministically from seeds** (`Nano`, `Micro`, `FullShape`), with calibrated logit entropy. Real checkpoints, reference blocks and other large data live in a **content-addressed object store**, with a manifest in git.

## Decision
1. **Synthetic fixtures** from `syntropy-testkit::fixtures::generate(spec, size, seed, quants)`:
   - **`Nano`:** ≤ 8 layers, byte-level tokenizer, CPU milliseconds per position; every PR.
   - **`Micro`:** the family's real head dims, expert width and tokenizer, with fewer layers and experts, 0.3–2 GB at Q4; PR on GPU rigs.
   - **`FullShape`:** the real census with no weights, for `Digest`-mode simulation.

   Every fixture contains every layer kind and feature its family has.
2. **Entropy calibration.** Norm gains, routers and the LM head scale are tuned so mean next-token entropy on the `smoke` corpus is 1.5–3 nats, with at most 20% near-tie positions. The measured values are recorded per fixture.
3. **Deterministic quantization** of an F32 original into every shipped quant class with RFC-0042's quantizers. A seed always gives byte-identical files.
4. **Real checkpoints:** at least one small checkpoint per family for nightly runs, plus the cost-model scenario models for RCs and benchmarks (RFC-0060 §12.2).
   - **Licence rule** (revised 2026-10-20). Nightly fixtures are ungated checkpoints under an OSI-approved licence. Llama-family checkpoints are lab-only. Fixtures are quantised from the original weights with RFC-0042's quantizers, never taken from third-party quantisations. Picking the checkpoint per family is a desk check in QUAL-E7-T06.
   - **Publisher-reference goldens** (added 2026-10-20). Each family's conformance checkpoint also stores golden logits and residual-stream norms at a few layer boundaries, produced offline in F32 by the publisher's reference implementation, pinned by commit, on the same dequantised weights (RFC-0042 §7.1). They are content-addressed objects like any reference, and `FamilyCert` is gated on them (RFC-0060 §7.2 check 9).
5. **The fixture store.**
   - A project-owned S3-compatible bucket, addressed by BLAKE3, with `testdata/fixtures.toml` in git recording name, hash, size, source (Hub repo, revision, file), licence, gated flag and users.
   - Fetches are verified by hash, and a mismatch is fatal.
   - Runners keep a size-capped LRU cache.
6. **Gated models.** The Hub credential exists only on lab runners and is never exposed to hosted runners or fork PRs. Fork PRs test on synthetic fixtures and ungated checkpoints.
7. **Retention.** Anything cited by a release's `fixtures.toml` is kept indefinitely; other objects for 180 days after last reference.
8. **Licences and access** (revised 2026-10-20).
   - Every corpus item and fixture records its licence.
   - **The bucket is private.** Public read is granted, object by object, only to project-synthetic fixtures and to fixtures whose licence is permissive, each published with its licence file beside it. Every other object, gated ones included, stays private to the lab and is never redistributed (ADR-0018 §7). The access rule applies from M0.
   - **Corpus licences.** Allowed: public domain, CC0, CC-BY (with its attribution file kept), MIT, Apache-2.0 and BSD. Excluded: CC-BY-SA, CC-BY-NC, and outputs of hosted commercial model services. The `tool` category is generated from templates licensed CC0 or MIT.
   - **Fuzz seeds.** Seed corpora come only from project-synthetic fixtures and recorded requests, never from headers cut out of third-party checkpoints, so the seeds can be public.

## Consequences
- **Positive.**
  - Every PR runs real conformance and parity logic in seconds, on fixtures that cover every family feature.
  - Git stays small, and every byte a test uses is pinned by hash.
  - References are computed once and reused, with an audit trail of what each was computed from.
  - Families without a small public checkpoint are still tested per PR.
- **Negative.**
  - Synthetic weights do not reproduce real activation statistics, especially outlier channels. So nightly real-checkpoint runs remain necessary, and op conformance adds real-tensor slices nightly (RFC-0060 §4.2).
  - The bucket is project infrastructure to operate, with lifecycle rules for every prefix.
  - Per-object public access means each public object's licence file has to be checked when it is published.
  - Entropy calibration is extra generator code whose own correctness has to be checked, by recording the measured entropy per fixture.

## What would make us revisit
- Synthetic `Micro` fixtures pass a change that a real checkpoint then fails, for the same op path, more than twice per release cycle. Then run a real-checkpoint smoke on PR† jobs, at the hardware-time cost.
- Storage use grows to five times RFC-0062's estimate. Then revisit with measured sizes, and prune `calibration` references for superseded corpus versions first.
- A model family appears whose licence forbids even private test use of its weights. Then test that family on synthetic fixtures only, and say so in its certification.

## Revisions
- Revised 2026-10-20: decision 4 gains the nightly-fixture licence rule (OQ-MF-04), so nightly runs never depend on gated or restrictively licensed weights, and the publisher-reference goldens that gate `FamilyCert`. Decision 8 makes the bucket private, with public read only for project-synthetic and permissively licensed fixtures, so nothing is redistributed whose licence does not allow it, and fixes the corpus licence list (OQ-FID-17) and the fuzz-seed rule (OQ-PRC-18), so seed corpora can be public.
