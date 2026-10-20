# ADR-0611: Performance regression gates: paired ABBA runs with noise-derived bands
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0020; RFC-0001 §14, §16; RFC-0061 §3, §8; RFC-0062

## Context
SC-32 requires that no Tier 1 reference benchmark moves outside a noise band between releases. The survey found decode regressions of about 16–20% shipped between nearby releases of one engine, so a gate is needed. A gate that fires on noise ends up routinely overridden, and one too loose to fire lets regressions accumulate one small step at a time. Consumer rigs differ widely in noise:
- desktop GPU decode at batch 1 is typically steady;
- laptops under thermal management, NVMe-bound tiered MoE, and anything over Wi-Fi are not.

Lab time is limited (RFC-0061: ≤ 10 min of timed blocks per rig per PR†), and RCs must stay within days.

## Options considered
1. **Fixed thresholds** ("fail above 5%"). Trivial, but wrong on both ends: too loose for quiet rigs, flaky for noisy ones.
2. **Compare each run with a stored historical baseline.** Cheap, but it confounds code changes with rig drift between the baseline's date and today, and it breaks on every driver or BIOS change.
3. **Unpaired A/B with t-tests.** It assumes normality and independence that thermal drift violates, and it needs more repetitions for the same power.
4. **Chosen: repeated ABBA blocks with block pairs as units, an exact sign-flip test, and a gate band derived from A/A runs of the same design, for decisions (PR† and release); level-shift detection with cumulative bisection for nightly trends.**

## Decision
Revised 2026-10-19.

1. **Design.**
   - Every gating comparison runs repeated ABBA blocks on the same rig, with instance reloads between blocks. The block pair is the unit of analysis:
     - PR†: ABBA × 3 (6 pairs, k = 2);
     - release and baseline comparisons: ABBA × 4 (8 pairs, k = 3);
     - bisection steps: ABBA × 3 (6 pairs, k = 2), the PR† design; `Sustained` series take one preload per step and k = 1.
   - The effect estimate r̂ is the mean of the pairs' log ratios of block medians.
   - Significance comes from an exact one-sided sign-flip permutation test over the pairs. No bootstrap is used at these sample sizes.
2. **Bands.**
   - **Gates** use `b_gate = max(b_min, 3 · s_AA)`, with `s_AA` the standard deviation of r̂ over the last 8 weekly A/A runs that use the gate's own design. The gate is report-only until 4 A/A runs exist in the epoch.
   - **Nightly trend detection** uses `b_trend = max(b_min, 3 · 1.4826 · MAD)` over residuals of the last 20 nightly medians around a rolling 5-run median, or 5% until 20 nightlies exist.
   - `b_min`: 2% for decode rate and TTFT; 3% for aggregate rate, per-term rows and hot-path microbenchmarks; 1% for peak memory; 5% for lifecycle times.
   - Driver image bumps are frozen during RC windows.
3. **Decisions.**
   - **PR† fails** when r̂ > `b_gate` in the bad direction and the sign-flip p < 0.05.
   - **The release is blocked** under the same rule on any Tier 1 rig. Tier 2 regressions are tracked and documented, not blocking (ADR-0020).
   - **Overrides** need a maintainer (ADR-0021 §5): on PRs, a label from a maintainer with a written reason; on releases, a maintainer's written waiver with the measurement attached. No agent can apply an override. Both are copied into the release notes.
4. **Nightly trend and bisection.**
   - The median of the last 3 nightlies is compared against the 10 before. A shift beyond `b_trend` triggers bisection over kept per-commit artefacts: from M1 by a post-merge build, from M2 by the merge queue.
   - **Like with like.** Per-commit artefacts are `pr`-profile builds. Bisection compares each candidate with the `pr` artefact of the last good nightly's commit, never with the `release`-profile nightly artefact. Step 0 confirms the shift between the two endpoints in that profile. If a shift flagged on the `release`-profile nightlies does not reproduce between `pr` builds, the bisection always rebuilds the endpoints and candidates in the `release` profile on hosted runners, so profile-specific regressions such as LTO effects are still bisected. If it still does not reproduce, the bisection names no commit.
   - Each bisection step runs the flagged scenario's own protocol as chained jobs of at most 60 minutes, and compares cumulatively against that good endpoint. A candidate is bad when r̂ exceeds `b_step = max(b_min, 3 · s_AA)` from A/A runs in the bisection design (falling back to `b_trend` until 4 exist) and the sign-flip p < 0.05.
   - A confirmation run of the named commit against its parent follows. When it, or no single commit's own step, exceeds the band, the result is a "distributed regression" with its commit range.
   - The issue it files names the commit or range and the per-term row that moved.
5. **Suspicious improvements.** A step better than max(3b, 15%) does not move the baseline until a maintainer acknowledges it. The review agent first checks that parity passes and that the bytes per tier and the generated-token counts are intact, and attaches that check to the request.
6. **Hygiene.**
   - `dirty` results (environment checks failed), `invalid: fidelity` results and results across baseline epochs never gate.
   - A series with CV above 3% for three nightlies is report-only until fixed.
7. **Calibration.**
   - Weekly A/A runs keep the PR† false-positive rate at ≤ 1% per rig.
   - Weekly planted regressions (graph capture disabled, a memory-bandwidth co-tenant, a CPU co-tenant) must be detected with ≥ 95% power at magnitudes ≥ 2b.

## Consequences
- **Positive.**
  - Thresholds track each rig's real noise.
  - Paired designs need fewer repetitions for the same sensitivity.
  - Drift and driver changes cannot masquerade as code regressions.
  - Nightly bisection points at a commit and a cost-model term, not just "something got slower".
- **Negative.**
  - ABBA doubles the runs per comparison against stored baselines, and needs both builds' artefacts on the rig.
  - The sign-flip test, the A/A-derived band and the MAD machinery must themselves be tested (RFC-0061 "Testing and validation").
  - Repeated ABBA costs more rig time per comparison than a single ABBA: about 9 `pr-quick` scenarios fit in 10 minutes instead of 15.
  - New rigs and new epochs gate loosely (5%) until 20 nightlies accumulate.
  - A slow regression smaller than b per step can still accumulate across releases. The release gate's comparison against the previous release, not the previous nightly, bounds that to b per release.

## What would make us revisit
- The A/A false-positive rate exceeds 1% on a rig after its band is widened to 5%. Then the rig, not the statistic, is the problem (ADR-0610).
- Planted regressions at 2b are missed more than 5% of the time. Then raise k, or lengthen blocks for that rig.
- Regressions below b per release accumulate to more than 2b over three releases on any reference scenario. Then add a multi-release cumulative check to the release gate.
