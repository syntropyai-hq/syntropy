# ADR-0626: Linux alone gates merges during M0a; the macOS and Windows hosted jobs run as advisory, revisited at M0b's start
- Status: Proposed
- Date: 2026-10-31
- Deciders: maintainers (drafted by the quality and release engineering area)
- Related: ADR-0020, ADR-0021 (decision 5), ADR-0610, ADR-0620 (decisions 1, 3), ADR-0625; RFC-0001 §16, §18.56; RFC-0062 §2.3, §2.4, §2.5 (note 2026-10-31); vision §5; QUAL-E1-T37

## Context
Vision §5 puts three operating systems in Tier 1: Linux x86_64, macOS arm64 and Windows x86_64, each with the CPU-only row. RFC-0062 §2.4 turns that into PR jobs on all three: `build-<os>` and `hosted-test-<os>` for Linux, macOS and Windows, plus `toolchains-macos` and `toolchains-windows`. Today `build` and `hosted-test` are matrix jobs in `pr.yml`, and `ci-gate` waits for every leg. A failed macOS or Windows leg fails the job, and so fails `ci-gate`.

RFC-0001 §16's first M0a deliverable is "The workspace builds on every Tier 1 OS". Every other M0a deliverable, and the M0a and M0b exits, are on Linux: the CPU provider in hosted CI and CUDA on R-ADA. M1 is split so that a public Linux build comes first (M1a) and the Windows and Apple ports follow (M1b). ADR-0610 first needs R-WIN in M1.

The non-Linux legs are the slowest jobs `ci-gate` waits for. Over the 60 most recent `pull_request` runs of `pr.yml` up to 2026-10-31 06:48 UTC, the median run time of each job that succeeded or failed was [measured: GitHub Actions API, `GET /actions/runs/{id}/jobs`, `started_at` to `completed_at`]:

| Job | Jobs counted | Median run |
|---|---|---|
| `hosted-test-windows` | 32 | 22.6 min |
| `hosted-test-macos` | 38 | 13.1 min |
| `hosted-test-linux` | 39 | 7.8 min |
| `build-windows` | 42 | 7.1 min |
| `build-linux` | 41 | 5.2 min |
| `build-macos` | 43 | 4.7 min |

Those runs mix hosted Linux runners and ADR-0625's build host, so the Linux figures are not one runner's. Of the 56 runs whose `ci-gate` finished, 2 had a failed macOS or Windows job. In 1 of them no Linux job failed, so the macOS or Windows leg alone failed the gate.

On 2026-10-31 the maintainers decided that M0a supports Linux only, with x86_64 the primary platform. This ADR records how CI carries that out.

## Options considered
1. **Keep all three OSes gating.** The plan as written. Every PR waits for the slowest Windows leg, and a failure seen only on macOS or Windows holds Linux work that M0a's exit does not need.
2. **Stop running the macOS and Windows jobs during M0a.** It saves the minutes, but portability breakage then goes unseen until they run again, and nobody can tell how much has built up.
3. **Chosen: the jobs keep running and reporting, but `ci-gate` does not wait for them or count them.** Each posts its own check on the PR, so a failure is visible. It neither fails nor holds the required check.

The maintainers' decision covers M0a, so this ADR ends at M0b's start and the maintainers decide then. The options for that review:
- **Chosen default: gate all three OSes again from M0b's start.** Portability debt is bounded by M0a's 200 tasks [estimate: the count `roadmap.md` gave on 2026-10-31 before this ADR], and M0b's work is checked on every Tier 1 OS as RFC-0062 §2.4 planned.
- **Extend through M0b, to M1a's start.** The roadmap gives a concrete reason: M0b adds no macOS or Windows deliverable, its exit is on Linux and R-ADA, and ADR-0610 first needs R-WIN in M1. The cost is a further 134 tasks [estimate: as above] of untested code on two OSes, which M1a or M1b then repairs. This needs a dated revision of this ADR, which only a maintainer makes.
- **Extend further, to M1b.** It matches the first macOS and Windows deliverables, but M1b would inherit M0 and M1a's untested code. This draft does not recommend it.

## Decision
1. **Scope.** From this ADR's acceptance until M0b starts, Linux x86_64 is the only platform that gates a merge. M0b starts at the commit that records M0a's exit. At that point the maintainers review this ADR: by default it ends (decision 5), or a dated revision extends it.
2. **What stays required.** Nothing changes for these, which still gate as they do today:
   - in `ci-gate`: `lint`, `build-linux`, `hosted-test-linux`, `hosted-l3`, `toolchains-linux`, `hosted-parity-nano` once it has a suite, and the hardware clause (`hw/<job>` statuses for `hw_required`);
   - the other required checks: `dco`, `pr-title`, `dep-diff` and `properties-removal`;
   - the independent review agent's approval, and a maintainer's for the classes in RFC-0062 §12.3.
3. **What becomes advisory.** `build-macos`, `build-windows`, `hosted-test-macos`, `hosted-test-windows`, `toolchains-macos` and `toolchains-windows`.
   - They still run on the same runners, and `changes` lists them on the same rules as their Linux counterparts. The macOS and Windows legs of `build` and `hosted-test` become jobs of their own for this, with the same check names, because `ci-gate` sees a matrix job's legs only as one result.
   - `ci-gate` does not wait for them, and their failure, cancellation or pending state does not fail it.
   - The list lives in one place, `ci/jobs.toml`, where each advisory job names this ADR. The base branch's `ci-gate` reads it (RFC-0062 §2.2 note of 2026-10-22), so a PR cannot widen the list for itself. A change to the list needs a maintainer, as any change to `ci/` does (RFC-0062 §12.3).
4. **Acceptance criteria with a macOS or Windows leg.** In 17 of the 200 M0a tasks before this ADR, an acceptance criterion names macOS, Windows or every OS [estimate: a search of the M0a rows' acceptance cells for those words, 2026-10-31].
   - A task closes on its Linux legs. The PR reports each macOS or Windows leg as **NOT RUN (ADR-0626)**, or with its advisory result, passed or failed, and the run link. A failed advisory leg is never reported as passed.
   - A task whose subject is only macOS or Windows, such as QUAL-E16-T30's toolchain pins, may still be done in M0a. Its advisory jobs are its evidence, and a maintainer decides whether it closes.
   - When a leg is real work in its own right, rather than a re-run, an agent splits it into a new task in M0b, or in M1b when it is part of a port, as COMP-E4-T21 took the R-ADA legs. The split follows the plan rules, and the maintainers confirm it.
   - Agents don't chase failures that occur only on macOS or Windows during M0a. They report each in the PR, and a failure on `main` gets an issue.
5. **Ending it.** At M0b's start, unless a revision extends this ADR, a PR removes the advisory marks in `ci/jobs.toml` and adds the jobs that hold the six checks back to `ci-gate`'s `needs`. Any advisory failure on `main` is fixed in that PR or before it.

## Consequences
- **Positive.**
  - `ci-gate` reports once the Linux jobs finish, instead of after the Windows tests. The median `hosted-test-windows` run is 14.8 min longer than `hosted-test-linux`'s [measured: the table above], and runner queueing comes on top of that.
  - A failure seen only on macOS or Windows no longer holds Linux work during M0a.
  - The macOS and Windows signal stays visible on every PR, so the debt can be counted at any time.
- **Negative.**
  - A change that breaks the build or the tests on macOS or Windows can merge. Breakage can pile up until M0b, and whoever restores gating pays for it.
  - `main` can show red macOS or Windows checks, and a contributor may read that as a broken tree.
  - RFC-0001 §16's first M0a deliverable, "The workspace builds on every Tier 1 OS", is then checked on every OS but enforced on Linux only, so M0a's exit does not need the macOS and Windows jobs green. On acceptance, §16 notes that, and so does vision §8, which `roadmap.py` derives from §16.
  - The runner minutes are spent as before.

## What would make us revisit
- Advisory failures on `main` stay unfixed for more than a few weeks, or a fix needs a design change that will cost more later. Then gate the OS concerned before M0b.
- A macOS or Windows deliverable moves into M0a, for example a Metal measurement on R-MACF that needs a host build first. Then gate that OS's jobs from then on.
- Hosted macOS or Windows capacity gets faster or slower enough to change the wait.
- M0b starts. The maintainers then end this ADR by decision 5, or extend it with a dated revision.
