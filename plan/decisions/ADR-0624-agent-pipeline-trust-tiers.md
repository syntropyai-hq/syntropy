# ADR-0624: Agent runs get trust by what they read: tiers, no secrets, pinned tooling and planted injections
- Status: Accepted (2026-10-20)
- Date: 2026-10-20
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0021 (decisions 1, 5, 7), ADR-0610, ADR-0620, ADR-0621; RFC-0034 §2; RFC-0062 §3.3, §12.3, §12.5, §13

## Context
AI agents write and review every change to Syntropy (ADR-0021). RFC-0034's adversaries cover the running product, not the pipeline that builds it. That pipeline has three exposures of its own:
- **Prompt injection through contributed content.** Agents read text that outsiders control: fork PRs, PR descriptions and comments, issues, commit messages, crash bundles, bench and tester reports, security reports, dependency documentation, and model files, whose GGUF metadata and chat templates are text. Any of it can carry instructions aimed at an agent: weaken a gate, plant a backdoor, leak a credential, or tell the review agent a change is approved.
- **Secrets in reach.** The KMS signing roles, the lab App keys, the results-store and gated-model credentials, cloud accounts and the agent tooling's own credentials.
- **The agent tooling's supply chain.** The agent runtime, its plug-ins and tool servers, and the instruction files in the repository shape every change. Agents can also propose dependencies that are typosquatted or do not exist.

A hijacked or runaway run is bounded by what it can reach and how long it can run. Agent-written code follows the same contribution rules as any other: the same `dco` check as every other commit.

## Options considered
1. **One trust level for every agent run**, with secrets scoped by environment only. Simple, but one injected run that reads an issue holds the same reach as one implementing a reviewed task.
2. **Maintainer review of every agent run's output.** Safe, but the review queue becomes the critical path for about 1,200 tasks, which ADR-0021 rejects.
3. **Chosen: trust follows input.** A run's reach is set by the least trusted content it has read, and the gates, not the agents, decide what merges.

## Decision
1. **Trust tiers.**
   - **Untrusted (U).** Any run that reads content outsiders control: a fork PR's diff, PR text or comments from outside, issues, crash bundles, bench and field-test reports, security reports, dependency documentation or source fetched from a registry, and model files. It runs in the agent sandbox (QUAL-E2-T11) defined on `main`: a read-only token, no secrets, no OIDC token, no signing access, no path to the hardware dispatcher or to the agent GPU pool (ADR-0610 decision 10), and allow-listed egress. Its output is a proposal, gated like a fork PR.
   - **Project (P).** A run that reads only what is already on `main` (its task, the RFC sections and ADRs the task cites, the code) and its own branch. It may push to its own branch and open a PR, holds only its own short-lived tooling credentials, and may lease the agent GPU pool. It has no web-fetch tool; fetching outside content would make it tier U.
   - **Privileged environments** (`release-signing`, `nightly-signing`, `tuf-timestamp`, `release-please`, `lab-dispatch`, `lab-trusted`). No agent run assumes them. They run deterministic workflows defined on `main`, approved where RFC-0062 §7.2 requires.
   - A run that reads tier-U content drops to tier U for the rest of that run. Tiers never rise within a run.
2. **The review agent reads PR text as data.** It judges the diff against the task's acceptance criteria and the cited RFC sections, read from `main`. Instructions in PR text, comments, commit messages or code comments are never followed. It reviews fork PRs, and any PR carrying outside content, as a tier-U run.
3. **Gates decide.** Gates are deterministic checks that an agent cannot argue past. Changes to gates, CI configuration and agent instruction and configuration files need a maintainer's approval (RFC-0062 §12.3).
4. **Run time limits.** Every agent run has a wall-clock limit set in its run definition on `main`. A run that reaches it is stopped and its output discarded. With the sandbox's reach limits, that bounds a hijacked or runaway run. Resource use is reported per line in `ci/cost.toml`, which stops nothing.
5. **Agent tooling is supply chain.**
   - The agent runtime, plug-ins and tool servers are pinned by version and digest in `ci/agent-tooling.lock`, the way Actions are pinned by SHA. A tooling bump is a reviewed PR like a dependency update, and the lock file is an agent configuration file under RFC-0062 §12.3.
   - Each CI run records the tooling versions in its provenance, not in commit messages (ADR-0021 decision 1).
   - A new direct dependency must pass `cargo-deny` and a registry check that reports its first-publish date and publisher and fails on a name that does not resolve. A maintainer approves it (RFC-0062 §12.3).
6. **Planted prompt injections.** Like RFC-0060's planted defects, a weekly job seeds known injection strings into fixture PRs, issues, tester reports and model-file metadata, and runs the review agent and the tier-U agents on them. A run that acts on one fails the job: approving a fixture that fails its criteria, emitting the sandbox canary, changing a gate label, calling the dispatcher, or writing outside its branch. A failure is filed as an escaped defect against the gate (RFC-0062 "Testing and validation" item 11).
7. **One contribution rule.** Agent-authored commits pass the same `dco` check as every other commit, with no separate gate.
8. **Security response.** Agents that reproduce a vulnerability report or draft its fix run as tier U inside the temporary private fork of the advisory: no secrets, no signing access and no way to publish. A security fix merges only with a maintainer's explicit approval (RFC-0062 §13).

## Consequences
- **Positive.**
  - An injected instruction can at most shape a proposal that the same gates and review then judge.
  - Credentials sit only in environments that agents never assume, so no agent run can sign, publish or reach a rig.
  - The tooling that writes the code is versioned and reviewed like the code.
  - The planted-injection job measures the review agent's resistance instead of assuming it.
- **Negative.**
  - Tier-U runs are slower to turn into merges: their output waits for gates and, where RFC-0062 §12.3 requires it, a maintainer.
  - Wall-clock limits can cut off a legitimate long run, which is then re-run with a larger limit set on `main`.
  - A burst of runaway runs still consumes resources until their time limits stop them; the monthly ledger shows it after the fact.

## What would make us revisit
- A planted injection succeeds twice in one quarter. Then require a maintainer's approval for every tier-U proposal until the review agent's instructions are fixed.
- A hijacked run consumes material resources before its time limit stops it. Then add a per-run resource limit beside the time limit.
- Agent tooling gains a verifiable isolation boundary that holds secrets safely. Then reconsider letting tier-P runs hold narrowly scoped credentials.
