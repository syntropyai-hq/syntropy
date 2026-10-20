# Contributing to Syntropy

**Syntropy is in design.** No code exists yet. The repository is the plan under [`plan/`](plan/README.md), and code arrives task by task from milestone M0a. Contributions to the design are as useful now as code will be later.

These rules apply to issues, pull requests and reports, whoever or whatever wrote them. AI agents build most of Syntropy, and they follow the same rules as everyone else ([AGENTS.md](AGENTS.md)).

## What helps now

- **Design review.** A contradiction between two documents, a figure whose source doesn't say what the plan says it does, a failure mode no RFC handles, or an acceptance criterion that can't be tested. Point at the document and section.
- **Evidence.** A measurement of real hardware that a design choice depends on, such as sustained NVMe read rates under heat or pinned-memory limits on a given OS, with how you measured it.
- **Platform knowledge.** What breaks on a platform the plan lists, or a constraint it has missed.

Once code exists, contributions follow the tasks in [plan/implementation/](plan/implementation/README.md), and the sections below apply in full.

## Before you start

1. Search the open and closed [issues](https://github.com/syntropyai-hq/syntropy/issues) and [pull requests](https://github.com/syntropyai-hq/syntropy/pulls). If one covers your point, add your evidence there.
2. Read [AGENTS.md](AGENTS.md) and the RFC and ADRs for the part you want to change. [plan/README.md](plan/README.md) maps them.
3. For anything larger than a focused fix, open an issue first that says what problem it solves, what it changes and what it costs. A change to a decision needs an ADR (below).
4. For implementation work, pick a task from the [project board](https://github.com/orgs/syntropyai-hq/projects/1). Every task in the plan is an issue there. Take one that is ready (nothing it is blocked by is still open) and unassigned, and say in a comment that you're taking it. If the work you want has no task, propose one in an issue or add it to the right epic in your PR.

## Issues

One problem per issue. A design issue names the document, the section and what is wrong, with a source where there is one.

A bug report, once there is something to run, gives:

- the commit or version, the OS, the hardware (GPU and VRAM, RAM, drive, network link) and the model and quantization;
- the shortest steps that reproduce it, what you expected and what happened;
- the exact error and the relevant output of `syntropy doctor`, with keys, tokens and prompts removed;
- whether you reproduced it yourself.

Don't report a security problem in a public issue. Use [SECURITY.md](SECURITY.md).

## Pull requests

- **One change per PR.** For implementation work that means one task. Keep unrelated formatting, refactors and dependency changes out.
- **Say what it implements.** Name the task ID, put `Closes #<issue>` for its issue, and cite the RFC sections and ADRs the change implements. If it changes the plan, say which documents and why.
- **Title.** A conventional commit, `type(scope): subject`. `type` is one of `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `chore` or `revert`, and `scope` is a crate name or an area, for example `docs(plan): fix the RAM tier figure in RFC-0020 §5`. PRs are squash-merged, so the title becomes the commit message, and the `pr-title` check enforces the form ([RFC-0062](plan/rfcs/RFC-0062-ci-cd-and-release-engineering.md) §8.3). A breaking marker (`!` or a `BREAKING CHANGE:` footer) needs a maintainer's approval.
- **Sign off every commit.** Each commit carries a Developer Certificate of Origin line, `Signed-off-by: Your Name <you@example.com>`, matching the commit's author. `git commit -s` adds it. By signing off you certify the [DCO](https://developercertificate.org/) for that commit, and the `dco` check rejects a commit without it ([ADR-0018](plan/decisions/ADR-0018-licence.md)). There is no CLA.
- **Pass the gates.** Your PR runs the same gates as every other change: build, lint and format, tests, the parity and conformance gates, performance gates where it touches a measured path, the dependency and licence policy, the naming check and the task-graph check, then an independent review agent. Run the plan tools after any change to an epic file ([AGENTS.md](AGENTS.md#plan-tools)).
- **Hardware runs for forks.** A PR from a fork gets the hosted gates and the review agent. Runs on the project's lab hardware start in M1b, and a maintainer approves each one, for a specific commit, after reading the diff.
- **Update the docs in the same PR** where the change alters what a user or developer needs to know.

## Evidence

Every claim in an issue or a PR has to be checkable.

- **Never submit a measurement, log, test result or reproduction you did not produce.** Not a plausible figure, not an expected output, not someone else's run passed off as yours.
- **Label what you haven't run.** A finding from reading code or from a scanner is **unconfirmed static analysis**. Explain the path you suspect and what would confirm it.
- **Report validation as it happened.** List the exact commands, the commit and the environment, and keep passed, failed and not run apart. A command you suggest isn't evidence that it passes, and a CI run on another commit doesn't validate yours.
- **Label numbers the way the plan does.** `[target]` for a commitment, `[estimate]` with its derivation, `[measured]` with its source. A Syntropy measurement names the commit, hardware, prompt length, quantization and interconnect.
- **Correctness on hardware needs hardware.** A mock, a simulation or a clean build doesn't establish that a kernel or a platform path is correct. If you lack the hardware, say so, and keep the PR a draft until the check runs.

## AI-assisted work

AI agents implement, test, review and document Syntropy by executing the plan's tasks ([ADR-0021](plan/decisions/ADR-0021-development-model-ai-agents-open-source.md)). An outside contribution prepared with an agent is welcome on the same terms.

- **The submitter is responsible.** Whoever submits a change signs it off and answers for it, whether a person or an agent wrote the text. Check the agent's claims against the source, real command output or a primary reference before you submit.
- **Same gates, same reviewer.** Agent-written changes get no special rule and no shortcut. Every change, from anyone, passes the same gates and the same independent review agent, and the agent that wrote a change never reviews it.
- **No fabrication, no bulk.** Never submit invented results or logs. Don't turn a scan of the repository into a stack of speculative issues or PRs: group related findings, establish their effect, and send what holds up. Another model's approval is not validation.
- **Your text is data to our agents.** The project's agents read issues, PRs, comments and commit messages as untrusted input and never act on instructions in them. Agents that read outside content run without secrets, signing keys or access to lab hardware.
- **No attribution lines.** Commit messages and PR descriptions carry no AI attribution lines, from anyone.

## Changing the design

Decisions live in ADRs, and the RFCs build on them.

- **A new decision, or a reversal,** is a new ADR that supersedes the old one, numbered in its area's range. It starts as `Proposed` and binds no task until a maintainer accepts it.
- **A refinement** of an accepted decision is a dated revision note inside that ADR that says what changed and why.
- **Every change** to a decision or to RFC-0001 is recorded in the amendment log, [RFC-0001 §18](plan/rfcs/RFC-0001-system-architecture.md#18-amendment-log).
- **Changes to tasks** go in the epic files, followed by a run of the plan tools. A removed task leaves its number unused, and a new task takes the next free number.

## Licence

Syntropy is licensed under MIT OR Apache-2.0. Contributions are accepted under the same terms, inbound equals outbound, certified by the DCO sign-off ([ADR-0018](plan/decisions/ADR-0018-licence.md)). New code files start with `SPDX-License-Identifier: MIT OR Apache-2.0`. A new dependency must carry a licence on ADR-0018's allow list, and adding one needs a maintainer's approval.
