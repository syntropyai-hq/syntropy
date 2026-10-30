# ADR-0620: CI runner strategy: hosted builds, self-hosted lab rigs for hardware jobs only, triggered by path rules and labels
- Status: Accepted (2026-10-20)
- Date: 2026-10-19
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0001, ADR-0019, ADR-0020, ADR-0021, ADR-0610, ADR-0624; RFC-0060, RFC-0061, RFC-0062 §2–§3

## Context
ADR-0020 promises real-hardware parity and performance jobs for Tier 1 on every PR that touches `kernels-*`, `exec`, `tier`, `io` or `net`, path-filtered and concurrency-cancelled. It also promises nightly and per-RC jobs, and weekly jobs for Tier 2. ADR-0019 §9 adds the sampler and the wire to the parity trigger. Kernel builds dominate CI cost (ADR-0001).

Hardware runners are owned lab machines (ADR-0610). Those machines are bare metal with real GPUs, and they run timed benchmarks that must not share the machine with anything else. The repository is public, so PRs from forks carry untrusted code.

## Options considered
1. **Everything on hosted runners**, with paid GPU runners for hardware. GPU classes do not match ADR-0020's consumer parts, hosted macOS is reported to expose no Metal device, and every minute is billed.
2. **Everything on self-hosted lab machines.** Fast builds, but rig time is contended, toolchains have to live on rigs, and builds co-run with timed benchmarks.
3. **Hardware jobs on every PR.** Simple, but it spends rig hours on changes that cannot affect hardware results.
4. **Chosen:**
   - hosted runners for builds and every hardware-independent suite;
   - lab rigs, one ephemeral runner each, for hardware jobs only, consuming artefacts built by hosted jobs;
   - hardware jobs triggered by direct path rules taken from ADR-0019 and ADR-0020, by labels from a maintainer, by schedules and by RCs;
   - fork PRs on rigs only after a label from a maintainer, which is bound to the reviewed commit.

## Decision
1. **Hosted runners** (Linux x64 and arm64, macOS arm64, Windows x64) run `lint`, builds, unit, property and concurrency tests, simulation, the emulated fabric, fuzz smoke and cross-builds. Larger billed runners are used only with a measured reason recorded in `ci/cost.toml`.
2. **Lab rigs, reached only through a trusted dispatcher** (revised 2026-10-19).
   - A dispatcher (`lab.yml` on `main`, plus a GitHub App in a hosted `lab-dispatch` environment) is the only path to a rig. It reserves the rigs a job needs, mints a single-use just-in-time runner configuration per rig off-rig, with a random per-run label, and dispatches `lab.yml` to that label.
   - No registration credential lives on a rig. No workflow a PR can edit can name a label any rig answers to. The runner group is restricted to this repository, and to `lab.yml` on `main` where the plan supports workflow restrictions.
   - Rigs never compile. They are wiped and inventory-checked between jobs.
   - Nightly and weekly work is a chain of jobs of at most 60 minutes, with one queued chunk per rig, and PR† jobs take at most 20.
3. **Triggers.**
   - `cargo xtask ci changes` computes the affected-job set. It uses the crate-graph closure for hosted jobs and direct path rules for hardware jobs, output as `hw_required`. The rules cover `kernels` and every `kernels-*` crate, `exec`, `sched`, `tier`, `io`, `net`, `graph`, the census and KV modules, the reference and parity modules of `testkit`, the vendored ggml and `rust-toolchain.toml` (revised 2026-10-22), and the tolerance tables under `numerics/` (revised 2026-10-23), as RFC-0062 §2.2 lists them.
   - A single required check, `ci-gate`, aggregates the conditional jobs. Skipped hosted jobs pass; failed or cancelled jobs fail. Its hardware clause depends on the event (revised 2026-10-19; RFC-0062 §2.5):
     - on `pull_request`, a job in `hw_required` without a passing `hw/<job>` status on the PR head fails the gate, including for fork PRs. Until a job's rig runs it, a same-repository PR is not held for that job; a fork PR still is (revised 2026-10-23);
     - on `merge_group`, the gate evaluates hosted jobs only, plus the statuses of any member a maintainer labelled `hw:queue` (decision 5), so an unlabelled batch never waits for a rig;
     - after merge, the dispatcher runs each `main` commit's `hw_required` jobs on rigs. Those statuses gate release promotion, not merging.
   - Every PR workflow uses per-PR concurrency with cancel-in-progress.
4. **Untrusted code.**
   - Workflow runs from all outside collaborators need approval from a maintainer. The repository setting is approval for all outside contributors (`all_external_contributors`), not only first-time ones, because with first-time-only approval a contributor's second PR runs unreviewed.
   - **Fork hardware runs start after M1a.** Until then, a fork PR whose `hw_required` set is non-empty waits; it cannot merge on hosted jobs alone. From M1b, each fork run is approved by a maintainer, one `/hw run` per reviewed SHA.
   - Agents that read, build or review untrusted contributed content run without secrets, signing keys or access to the dispatcher (ADR-0021 §7). No agent can issue `/hw run` or apply `hw:queue`.
   - A fork reaches a rig only after a `/hw run <sha>` command from a maintainer (**Maintainer action**), handled by an `issue_comment` workflow on `main` that never checks out PR code and that binds the request to the reviewed head SHA.
   - The dispatcher builds that SHA with `main`'s build definition, `permissions: {}`, no secrets and read-only caches. The rig runs only those artefacts.
   - A rig that ran fork-derived artefacts is tainted, and is re-imaged before any `lab-trusted` job.
   - Fork results are tagged `untrusted`: never gating `main`, never a baseline, never published.
   - Trusted jobs get gated-model and results-store credentials through the `lab-trusted` environment, on untainted rigs only.
   - **Environments are bound to `main` and `release/*`** (revised 2026-10-20). Since 8 December 2025 GitHub evaluates environment branch rules against `GITHUB_REF`, which is `refs/pull/<n>/merge` for `pull_request` events, so a rule naming PR branches matches nothing for PR runs and lets a push or dispatch run on that branch claim the credentials. `lab-dispatch` and `lab-trusted` are restricted to `main`, plus `release/*` for release jobs. Every lab job, a same-repository PR's included, runs as `lab.yml` dispatched on `main`; the hand-off from `pr.yml` is untrusted data, accepted only as a job name from an allow-list and a SHA checked against the PR's current head.
   - **The dispatcher App** is owned by the GitHub organisation and managed by two maintainers, so the lab's only entry point does not hang on one account. It is installed on the Syntropy repository only, with least privilege: Actions write (to dispatch `lab.yml`), Commit statuses write (`hw/<job>`), Contents read, Metadata read, and the organisation's Self-hosted runners write, so JIT configurations are minted at organisation level into a runner group limited to this repository and the App needs no repository Administration write. Its private key lives only in `lab-dispatch`, mints installation tokens that last one hour, and is rotated every 90 days and whenever a delegate changes.
   - **Repository settings:** the Actions policy that requires full-length SHA pins and blocks listed actions, secret scanning with push protection, and private vulnerability reporting. The organisation's plan must offer extra organisation runner groups.
   - **Agent runs** follow ADR-0624's trust tiers: a run that reads untrusted content has no secret, no write token and no path to the dispatcher or the agent GPU pool.
   - Rig egress is allow-listed.
5. **Merge queue.** Batches of up to 5 rerun hosted checks on the merged result. The queue's required checks are `ci-gate`, `dco` and `pr-title`, all hosted. Besides them, every PR needs the independent review agent's approval. A maintainer's approval is also needed for the list kept in RFC-0062 §12.3 and `CODEOWNERS`: security-critical paths (keys and signing, admission, the update client, the hardware dispatcher), CI and release configuration, agent instruction and configuration files, new model families, new direct dependencies, and any change where a gate is waived or cannot check (flagged `gate:uncovered` by the review agent). Kernels, the solver and the rest of `net` and `cluster` merge on the gates and the review agent (ADR-0021 §5, RFC-0062 §2.5). Hardware jobs run on a merge group only when a maintainer labels a member PR `hw:queue`; `ci-gate` then also waits for those statuses, and a batch whose statuses do not arrive within the queue timeout is ejected and can be re-queued. The queue's builds, in the `pr` profile, are the per-commit artefacts kept for bisection.

## Consequences
- **Positive.**
  - The lab spends its hours on changes that can move hardware results, plus scheduled full coverage.
  - Rigs stay clean for timed runs, and the bits tested on rigs are the bits built in CI.
  - Untrusted code cannot reach a rig, a credential or a shared cache without a maintainer approving it first.
  - One required check keeps branch protection stable as jobs change.
- **Negative.**
  - Changes outside the direct paths, for example in `core`, get hardware coverage only nightly. A hardware-visible regression from such a change is found up to a day later, and bisection then attributes it.
  - Fork contributors wait for a maintainer before hardware feedback, and fork PRs that need hardware cannot merge before M1b.
  - The dispatcher is a single point of failure for lab work. When it is down, no hardware job runs and `ci-gate` holds hardware-requiring PRs.
  - Re-imaging a rig after fork-derived runs costs rig time, so fork runs are batched into one daily window.
  - Path rules can rot as files move, so a weekly self-check PR verifies them (RFC-0062 "Testing and validation").

## What would make us revisit
- Nightly bisections attribute more than one hardware regression per month to commits that the direct rules did not cover. Then widen the rules, for example to the closure for `core` changes as well.
- PR† jobs exceed 50% of a Tier 1 rig's daytime hours for two consecutive months. Then narrow the rules or add a rig.
- Hosted runners gain Metal devices or consumer NVIDIA GPUs at acceptable cost. Then move correctness-only hardware jobs there, and keep the lab for timed runs. GitHub's hosted GPU runner is checked in M0 as a backup for `sm_75` floor conformance (ADR-0610 decision 8).
- GitHub offers per-workflow runner-group restrictions on the organisation's plan. Then add them, and keep the random per-run label anyway.

## Revisions
- Revised 2026-10-30: ADR-0625 (Proposed) adds a self-hosted Linux build host for decision 1's Linux jobs of same-repository PRs, behind a repository variable and a gate the workflow lint checks. Fork PRs, merge groups and every other run keep decision 1's hosted runners, and decision 2's lab rigs are unchanged: the build host is not a rig and never runs a lab job.
- Revised 2026-10-23: decision 3's direct hardware paths gain `numerics/`. `kernels` embeds the tolerance tables there, and the op conformance a hardware job runs checks against them, so a change to a ceiling alone needs the rig as a kernel change does (ADR-0019).
- Revised 2026-10-23: decision 3's `pull_request` clause gains an interim for same-repository PRs.
  - **The rule.** While a hardware job's rig and its `lab.yml` job do not exist, `ci/jobs.toml` marks the job `planned` with the task that brings it. `ci-gate` then does not hold a same-repository PR for it, and the summary names the task.
  - **Fork PRs are unchanged.** They wait for every job in `hw_required`, as decision 4 requires.
  - **Why.** Without the interim, no PR touching `kernels`, `exec`, `sched`, `tier`, `io`, `net` or `graph` could merge until R-ADA is online and QUAL-E2-T02 has landed. Most of the engine and networking work would stop.
  - **When it ends.** It ends job by job: QUAL-E2-T02 removes the mark from `hw-cuda-ada`, and QUAL-E2-T08 adds the other Tier 1 rigs' jobs.
  - **Approval.** Taken under the maintainer's standing instruction (QUAL-E1-T07), for the maintainers to confirm. RFC-0062 §2.5 (note of 2026-10-23).
- Revised 2026-10-22: decision 3's list of direct hardware paths gains `kernels`, `exec`, `sched`, `tier`, `io`, `net` and `rust-toolchain.toml`. The context above already takes the rules from ADR-0019 §9 and ADR-0020, which name them, and RFC-0062 §2.2's `ci/jobs.toml` lists them; the decision's list had left them out.
- Revised 2026-10-20: decision 4 gains approval for all outside contributors; fork hardware runs from M1b only, each approved by a maintainer, so no fork-derived code reaches a rig before the first public preview has shipped; environments restricted to `main` and `release/*`, because GitHub now evaluates environment branch rules against the PR merge ref, with every lab job dispatched on `main` and PR outputs handled as data; the organisation-owned, least-privilege dispatcher App; the repository settings; and ADR-0624's trust tiers for agent runs. Decision 5 states the list of changes that need a maintainer's approval.
