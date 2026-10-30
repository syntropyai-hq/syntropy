# ADR-0625: A self-hosted Linux build host for same-repository PRs, behind a gate, with fork PRs left on hosted runners
- Status: Proposed
- Date: 2026-10-30
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0620 (decisions 1, 2, 4), ADR-0610, ADR-0624; RFC-0062 §3.1, §3.2, §12.4

## Context
ADR-0620 decision 1 runs every build and hardware-independent suite on hosted runners. Hosted minutes on this public repository carry no charge, but the organisation can run only a limited number of hosted jobs at once, and agents open many PRs in parallel. Over the last 8 `pr.yml` runs before this ADR, the Linux jobs waited 6 to 10 minutes on average for a runner before they started [measured: GitHub Actions API job timestamps, `created_at` to `started_at`, runs of 2026-10-30]:

| Job | Average wait | Average run |
|---|---|---|
| `lint` | 8.4 min | 3.0 min |
| `build-linux` | 8.3 min | 7.0 min |
| `hosted-test-linux` | 9.7 min | 10.7 min |
| `hosted-l3` | 6.8 min | 0.9 min |

A maintainer has a spare x86-64 Linux machine (20 hardware threads, 45 GB RAM, NVMe). It is not a lab rig: ADR-0610's rigs are other machines, and this one runs no timed benchmark and no hardware job.

RFC-0062 §3.2 explains why self-hosted runners on a public repository are dangerous: a workflow a fork's PR edits can run code on the machine and persist there. ADR-0620 kept every self-hosted runner behind the lab dispatcher for that reason. This ADR adds a narrower path for builds only.

## Options considered
1. **Keep hosted runners only.** No new risk, but the wait stays.
2. **Larger hosted runners.** They are billed per minute, and ADR-0620 decision 1 needs a measured reason in `ci/cost.toml`. They do not raise the concurrency limit, so they don't shorten the wait.
3. **A persistent self-hosted runner for every PR.** Fast, but a fork's code would run on the machine with the runner's credentials, and could persist and tamper with later jobs.
4. **Chosen: an ephemeral build host for same-repository PRs only.** Fork PRs, merge groups and every other event stay on hosted runners.

## Decision
1. **What runs there.** `lint`, `build-linux`, `hosted-test-linux` and `hosted-l3` of a PR whose head branch is in this repository. `changes` and `ci-gate`, which judge the run from the base branch, stay hosted, and so do `toolchains-linux` (it needs `docker buildx`), the macOS and Windows jobs, merge groups and `pr-meta.yml`. The host never runs a lab job.
2. **The gate.** A job reaches the build host only through the form `${{ (<gate>) && <build-host label> || <hosted label> }}`, where the gate is `vars.SYNTROPY_BUILD_HOST == 'on' && github.event.pull_request.head.repo.full_name == github.repository`. A fork's PR, a merge group and every other event get the hosted label, and so does every run while the repository variable `SYNTROPY_BUILD_HOST` is not `on`. A maintainer turns the variable off when the host is down, and jobs return to hosted runners on the next run. The workflow lint (RFC-0062 §12.4) refuses the build-host label `syntropy-build` anywhere outside that form, including as the fallback.
3. **One job per container, single-use registration.** A systemd unit per job slot mints a just-in-time runner configuration with the label `syntropy-build` and no default labels, starts a fresh container of the build-host image with it, and lets the container's runner take one job. The container is removed when the job ends, and the next slot run starts from the image. The token that mints configurations stays on the host, in the unit's credential store, and never enters a container. It is a fine-grained token limited to this repository with Administration write, the permission GitHub requires for repository runners. A maintainer creates it and rotates it every 90 days (**Maintainer action**).
4. **No reach into the LAN.** Job containers run unprivileged on their own bridge network. Firewall rules drop their traffic to private, link-local and tailnet addresses and to the host itself, and they use public DNS resolvers. A job reaches the internet, GitHub included, and nothing on the maintainer's network.
5. **Caches.** Jobs keep `setup-rust`'s GitHub cache, with the same read-write rule as on hosted runners. Nothing is shared between containers on the host.
6. **The host enforces the gate itself.** A job-started hook in the image runs before a job's first step and fails the job unless it is this repository's `pr.yml` on a `pull_request` whose head and base are both this repository. A fork's PR that edits `pr.yml` to name `syntropy-build`, approved by mistake under ADR-0620 decision 4, therefore runs no step on the host. The workflow lint fails such a PR too, so approvers see it.
7. **Least privilege on the host.** Job containers drop every capability and get back only what sudo, `useradd` and the tests need (`CHOWN`, `DAC_OVERRIDE`, `FOWNER`, `SETUID`, `SETGID`, `KILL`, `AUDIT_WRITE`, `SYS_NICE`), with no `NET_RAW`, Docker's default AppArmor profile but no seccomp filter (revised 2026-10-31: `syntropy-io`'s io_uring seccomp test needs an unfiltered process, as on a hosted runner), and memory and process limits. The slot unit runs with only `CAP_NET_ADMIN` and `CAP_NET_RAW` for its firewall rules, a read-only file system and a system-call filter. The token and the JIT configuration never appear in a process's arguments.

## Consequences
- **Positive.**
  - Same-repository PRs, which are most of the traffic, no longer wait for hosted Linux capacity, and they free that capacity for fork PRs and the macOS and Windows jobs.
  - The host's 20 threads speed up the build and test steps themselves.
  - Turning the host off is one variable change, with no workflow edit.
- **Negative.**
  - The maintainer's machine must stay awake and online. While it is down and the variable is still `on`, same-repository PRs' Linux jobs queue until it returns or the variable is turned off.
  - A token with Administration write on the repository lives on a machine outside GitHub. It is limited to this repository and never enters a job.
  - Build-host jobs run on Ubuntu 24.04 in a container, not on the hosted image. A tool the hosted image has and the build-host image lacks fails there first, and the image must be updated.
  - A slot's container counts against the host's 45 GB of RAM; three slots at 14 GB each is the default.

## What would make us revisit
- A fork's run reaches the build host, approved by mistake or not. Then add a runner-group workflow restriction or move the host behind the lab dispatcher.
- The organisation's plan gains organisation runner groups restricted to selected workflows. Then mint configurations into such a group.
- Hosted capacity stops being the bottleneck, for example after a plan change. Then retire the host and delete the variable.
