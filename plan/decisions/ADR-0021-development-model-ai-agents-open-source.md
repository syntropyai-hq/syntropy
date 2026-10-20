# ADR-0021: Development model: open source, built by AI agents, no team ownership

- Status: Accepted (2026-10-20)
- Date: 2026-10-20
- Deciders: maintainers (directive D8)
- Related: ADR-0000 D8, ADR-0003, ADR-0018, ADR-0019, ADR-0316, ADR-0610, ADR-0620, ADR-0621, ADR-0624; RFC-0001 §15–§16, RFC-0034 §12, RFC-0060, RFC-0061, RFC-0062

## Context

Syntropy is open source from its first commit, and AI agents build it by executing the plan's tasks. The plan is written as small tasks with testable acceptance criteria and explicit dependencies, which is the form an agent can finish and an independent checker can verify.

A conventional engineering organisation puts people in each role: reviewers, release managers, a lab operator, key custodians. Without people in those roles, review, capacity and risk work differently, and the plan needs rules for each:
- **Review.** Something other than the author has to check every change, at the rate agents produce them.
- **Capacity.** Agents run in parallel cheaply, so the limits move to hardware access and to how fast changes can be verified.
- **Human-only steps.** Some actions need a person: holding offline signing keys, physical work on lab rigs, answering security reports.
- **Trust.** Agents read content that anyone can write, such as issues and contributed pull requests.

## Options considered

1. **A conventional team.** Rejected by directive D8.
2. **AI agents, with a person reviewing every change.** Simple accountability. At about 1,400 tasks, human review becomes the critical path, and its quality falls with volume.
3. **AI agents, reviewed by gates and an independent review agent, with maintainer approval for a fixed list of change classes.** It scales with agent concurrency and keeps people on the changes where judgement or a human-held key is needed. It depends completely on the gates being strong, and on them existing before the code they guard. **Chosen.**

## Decision

1. **Open source from the first commit.**
   - The repository is public, under MIT OR Apache-2.0 (ADR-0018). The plan documents are public.
   - Directive D5 still applies.
   - Commit messages carry no AI attribution lines.
   - Outside pull requests are welcome from the first commit, under the DCO (decision 7).
   - Lab hardware runs for fork pull requests (QUAL-E2-T03, QUAL-E2-T05) start in M1b, and a maintainer approves each run with `/hw run`. Until M1b a fork pull request gets the hosted gates and the review agent only, and one that needs a hardware gate waits rather than merging on hosted jobs (ADR-0620). That keeps fork hardware runs and untrusted result ingestion off the critical path of M0 and M1a.
   - The security contact and private reporting (decision 10) exist before the first commit is published.
2. **The unit of work is one task.** An agent takes a task from `plan/implementation/epics/`, implements it on its own branch, and produces one change set. The change set meets the task's acceptance criteria and cites the RFC sections and ADRs it implements. The task graph (`tools/depcheck.py`) decides what can start.
3. **Areas are domains, not teams.** PLC, TIER, DIST, COMP, PROD, QUAL and ARCH organise the code, the plan and the review routing.
   - An "Owner" column in a review table means the area whose documents must change.
   - RFC headers say "Area".
   - ADR "Deciders" lines name the maintainers.
   - Nobody owns an area.
4. **Maintainers** are the only human role. They:
   - approve the change classes in decision 5;
   - accept ADRs (decision 9);
   - do the steps that need a person, marked **Maintainer action** in the epics: holding and using offline signing keys, physical lab work (cabling, rig set-up, cable-pull and power-cut fault injection, re-imaging), the accounts the release and signing services run under, the security contact (decision 10), and community moderation.

   Each maintainer, and the keys, environments and human-only steps they hold, is listed in the **maintainer registry**, a file in the repository that QUAL-E1-T17 creates and whose path it sets. A maintainer acts only within what the registry lists, and changing it is a change in decision 5's first class.

   Keys that need several people are held by different maintainers, so that no one person can act alone: the TUF root's three tokens (any 2 of 3 sign), the TUF targets role's two tokens, and the `release-signing` environment's two approvers (ADR-0316). At least two maintainers can administer the GitHub organisation and the cloud key service's emergency access, so losing one person never locks the project out.
5. **Review is done by gates and agents.** A change merges only when all of these pass:
   - build, lint and format, as CI runs them;
   - unit, property and simulator tests;
   - the parity harness and the conformance gates (ADR-0019, RFC-0060), including the check against each publisher's reference implementation (ADR-0019 decision 1a);
   - the performance regression gates (RFC-0061) where the change touches a measured path;
   - the dependency and licence policy (ADR-0018);
   - the D5 naming check;
   - the task-graph check;
   - an **independent review agent**, which checks the change against its task's acceptance criteria and the cited RFC sections. It is never the agent that wrote the change. It applies the adversarial review pattern this plan was built with.

   These change classes also need a maintainer's explicit approval:
   - security-critical paths: keys, signing, admission, the update client and the hardware dispatcher;
   - CI and release configuration;
   - agent instruction and configuration files;
   - a new model family (ADR-0003);
   - a new direct dependency;
   - any change where a gate is waived, cannot check the change, or fails.

   The classes are the ones where a wrong change is hard to detect afterwards or hands an attacker a key, a runner or the supply chain. No change merges on its author agent's say-so alone, and agent identities are never code owners, so an agent can never be the approval a protected path needs.
6. **Capacity is measured in tasks, agent runs and rig time.**
   - Agent concurrency is cheap. The binding constraints are:
     - rig time, since each lab rig is a serial resource;
     - gate throughput, meaning how fast changes pass the gates and the review agent;
     - maintainer review of the classes in decision 5, and the Maintainer actions a task waits on.
   - Plans and roadmaps schedule against those constraints.
   - A runaway or hijacked agent run is bounded by the sandbox (QUAL-E2-T11) and by the wall-clock limit each agent run carries in its run definition on `main` (ADR-0624 decision 4). The sandbox limits what a run can reach, and the time limit limits how long it can act.
7. **External contributions** are welcome as pull requests under the Developer Certificate of Origin. They pass the same gates and the same review agent. An agent never runs untrusted contributed content with access to secrets or to lab hardware (RFC-0062 §2).
   - Every commit, whoever or whatever wrote it, carries a DCO `Signed-off-by` line and passes the one `dco` check (RFC-0062 §12.3, QUAL-E1-T05). The sign-off is made by the person who submits the change, as on any commit; for agent work that is the maintainer who directs it.
8. **Role names in the plan.** Wherever the plan says "release owners", "lab owner" or "custodian", it means a maintainer listed for that role in the registry. Wherever it says "team" or "engineers" about who builds Syntropy, it means the agents working under this ADR.
9. **Agents draft ADRs, and maintainers accept them.**
   - An agent writes a new ADR, or a dated revision note in an existing one. A new ADR starts as `Proposed`, and its status becomes `Accepted` when a maintainer accepts it.
   - A `Proposed` ADR binds no task until it is accepted, so a task never builds on a decision that may still change.
   - Every ADR in the plan was `Accepted` on 2026-10-20. Later changes go through dated revision notes or a superseding ADR.
10. **The security contact and disclosure.**
    - The security contact is the maintainers, reached through the repository host's private vulnerability reporting, as `SECURITY.md` describes. A `security@` address is added before the first release.
    - Maintainers assess every report, and at least two maintainers can respond, so a report never waits on one person.
    - Agents help reproduce and fix a report only in an isolated environment with no secrets, no signing access and no way to publish. A report's content is untrusted input (decision 7).
    - Targets: acknowledge within 3 business days, assess and agree a severity within 10 business days, fix or mitigate a critical within 30 days, and disclose within 90 days.
    - Advisories are published as GitHub security advisories.
    - Private reporting exists before the first commit is published, because nodes listen on LANs, accept invites and update themselves from the first release.

## Consequences

**Positive**
- Work can run in parallel up to the hardware limits.
- Every change is checked the same way, against written acceptance criteria.
- The plan's own form, small tasks with testable criteria and explicit dependencies, is what this model needs.

**Negative, and the work it creates**
- **The gates are the only safety net.** They must exist before the code they guard: the parity harness, the conformance suite and the publisher-reference check in M0. A gap in a gate is a gap in review, so the gates themselves need adversarial testing (planted defects, RFC-0060).
- **Kernels and platform code written by agents** must be proven on real hardware, not just in simulation. Access to the lab rigs becomes the bottleneck.
- **Prompt injection.** Agents read repository content, issues and contributed pull requests, any of which can carry instructions aimed at them. Agents working on untrusted input never see secrets, signing keys or the hardware dispatcher. Agent tooling is part of the supply chain (RFC-0062, RFC-0034, ADR-0624).
- **Human-only steps depend on few people.** The key thresholds in decision 4 and the two security responders in decision 10 keep any one person from being a single point of failure for signing or disclosure. Other Maintainer actions, such as physical lab work, are written as checklists in the repository so that any listed maintainer can carry them out.
- **Users and contributors judge an AI-built open-source project on its evidence.** Results are therefore published only with commit-tied reproduction (RFC-0061).

## What would make us revisit

- Defects that escape the gates and are found later, at a rate showing review by agents and gates isn't enough. Respond by strengthening the gate concerned, or by adding maintainer approval for that class of change.
- Maintainer approvals become the critical path. Narrow the classes in decision 5, or list more maintainers for them.
- A hijacked or runaway agent run does material harm before its sandbox or time limit stops it. Tighten the limits in ADR-0624.
- Human contributors who want standing roles beyond those in decision 4. Then write a governance document that adds roles explicitly, by superseding this ADR.

## Revisions

- **Revised 2026-10-20:**
  - Decision 1: outside pull requests from the first commit; fork hardware runs only from M1b (ADR-0620, RFC-0062), which keeps them off the critical path of M0 and M1a.
  - Decision 4: maintainers are the human role, listed in a maintainer registry, and multi-person keys are held by different maintainers so that no one person can sign alone.
  - Decision 5: the approval classes now include agent instruction and configuration files, new model families and new direct dependencies, which are the changes that can redirect agents or widen the supply chain.
  - Decision 6: the capacity limits are rig time, gate throughput and maintainer review, and the sandbox and per-run wall-clock limits bound runaway runs.
  - Decision 7: one `dco` check applies to every commit, agent-authored ones included.
  - Decision 9 (new): agents draft ADRs and maintainers accept them; a `Proposed` ADR binds no task.
  - Decision 10 (new): the maintainers are the security contact through private vulnerability reporting, with response targets matching `SECURITY.md`.
