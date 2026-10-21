# Plan tools

Scripts that read the six epic files in `../epics/`, check the task graph and write the generated sections of the integration files in `../`. Python 3, standard library only. Run them from any directory; they find the plan relative to their own location.

| Script | Reads | Writes |
|---|---|---|
| `epics.py` | `../epics/*.md` | nothing; parses task rows for the other scripts |
| `depcheck.py` | the epics | nothing in `plan/`; the summary, the problems, and optionally a JSON graph outside `plan/` |
| `deps_report.py` | `depcheck.py`'s result, `plan/open-questions.md` | the generated sections of `../dependencies.md` |
| `trace.py` | the epics, `plan/decisions/`, `plan/rfcs/`, `plan/00-vision-and-scope.md` §7 | the generated sections of `../traceability.md` |
| `roadmap.py` | `depcheck.py`'s result, RFC-0001 §16, `plan/open-questions.md`, vision §7 | the generated sections of `../roadmap.md`, and the milestone checklist in vision §8 |
| `milestone_split.py` | the epics, through `depcheck.py` | the Milestone cells (and any cut Depends cells) of the epic files; nothing with `--dry-run` |
| `gendoc.py` | – | nothing; helpers the three generators share (markers, tables, ID ranges, the open-questions and SC parsers) |
| `sizing.py` | the epics, or an `epics.py --json` dump | nothing; prints `[estimate]` size classes, effort units per milestone and work pool, and agent runs |
| `schedsim.py` | `sizing.py`, and the graph from `depcheck.py` or a `depcheck.py --graph` file | nothing; prints `[estimate]` elapsed time to each milestone exit under agent concurrency, serial lab rigs, maintainer latency and gate throughput, and the unconstrained floor |
| `gh_sync.py` | the epics, `depcheck.py`'s edges, `sizing.py`, the D5 banned-terms list (`--banned`, kept outside the repository); needs the `gh` CLI logged in with the `repo`, `project` and `read:org` scopes (`read:org` so that members' issues read as members) | GitHub, not `plan/`: one issue per epic and task on the project board, sub-issues, labels, milestones and "blocked by" links; re-runs update only what changed |
| `naming_check.py` | the files staged for commit or changed against a base ref, a commit message, standard input or named files; the D5 banned-terms list (`--banned`, `SYNTROPY_BANNED_TERMS` or `git config syntropy.bannedTerms`), read through `gh_sync.load_banned` | nothing; prints `path:line` for each matching line, never the term |
| `test_naming_check.py` | – | nothing; the unit tests for `naming_check.py`, each with its own throwaway term list, and tests that run the hooks: commit-msg on messages saved with CRLF endings, and both hooks for a repository without the hooks or tools (skipped without a POSIX `sh`) |

## `epics.py`

Parses every task row (`| ID | Task | Milestone | Depends on | Acceptance criteria | Implements |`) into a record: ID, area, epic, milestone as written, the milestones it names, the dependency cell, acceptance criteria, implements, file and line. Columns are matched by header name. Escaped pipes (`\|`) inside a cell are kept.

```sh
python3 epics.py          # tasks per file and per first milestone; malformed rows and duplicate IDs
python3 epics.py --json   # every record as JSON
```

It reports a row whose cell count differs from its table header (usually an unescaped `|`), an ID cell that is not a bare task ID, a task filed in another area's file, a milestone cell that names no milestone, and a duplicate ID. Exit status 1 if any are found.

A Milestone cell names one of the milestones RFC-0001 §16 defines, in this delivery order: `M0a` < `M0b` < `M1a` < `M1b` < `M2` < `M3` < `M4`. M0a is the walking skeleton and M0b the rest of M0; M1a is the first public preview (CUDA and CPU tiering on Linux) and M1b adds Windows and Apple silicon. A plain `M0` or `M1` is still valid and reads as `M0b` or `M1a`, the phase `milestone_split.py` gives such a task by default. Each record carries `milestones`, `first` and `last` as positions in that order (`epics.PHASES`), and `phase`, the name of the first.

A milestone cell may name several milestones, as in `M0 (Linux), M1 (macOS, Windows)` or `M2–M3`. The record keeps the first and the last; a range covers the milestones between its ends.

## `depcheck.py`

Builds the dependency graph from the "Depends on" cells and checks it. A cell is split on `;` outside parentheses and backticks. Each segment must be one of:

| Form | Meaning |
|---|---|
| `none` | no dependency |
| `PLC-E4-T17`, `PLC-E4-T17, QUAL-E1-T02` | task IDs; each may carry one trailing note, e.g. `QUAL-E1-T02 (workspace)` |
| `PROD-E7-T01–T10` | a range inside one epic (en dash or hyphen); every task between the ends |
| `TBD: <what is needed>` | a dependency no task delivers yet; listed, not an error |
| `gate: OQ-PKG-02 (note)` | an external gate: an open question (`OQ-…`) in `plan/open-questions.md`; no edge. A settled question no longer holds its task, and `deps_report.py` flags a gate whose ID is missing from the register |

A note that contains `gate: <OQ-ID>`, as in `DIST-E5-T08 (gate: OQ-NET-11)`, marks that edge as an accepted exception to the milestone checks. Task IDs written inside a note are not edges; the checker lists them as `NOTE-ID` so a reader can tell.

Problems, each making the exit status 1:

| Kind | Meaning |
|---|---|
| `WORDFORM` | a segment or item that is not one of the forms above, such as `COMP: census types` or a bare epic `QUAL-E1` |
| `DANGLING` | an ID, or an ID inside a range, that names no task |
| `CYCLE` | a strongly connected component of two or more tasks, or a task that depends on itself |
| `LATER` | a task depends on one whose first milestone is later than its own first milestone, in the order M0a < M0b < M1a < M1b < M2 < M3 < M4 |
| `M0CLOSURE` | an M0 task (M0a or M0b) reaches a non-M0 task through its dependencies, or an M0a task reaches a task outside M0a. M0 and M0a are each self-contained |
| `MULTI` | a Milestone cell that is not exactly one of `M0a`, `M0b`, `M1a`, `M1b` or `M0`–`M4`; a task that spans two milestones is split into a core task and an extension task |
| `MALFORMED`, `DUPLICATE` | rows `epics.py` could not read |

Every task sits in exactly one milestone, and `MULTI` enforces it. `epics.py` still reads a cell that names several milestones, so that the checker can report it: in such a cell the first milestone counts for the `LATER` check, and an edge to a task that starts by the cell's last milestone is listed as `SPLIT` rather than `LATER`. Gated edges are skipped by the milestone checks.

Plan-rule notes report rows that break a milestone rule of RFC-0001 §16. None of them changes the exit status; `deps_report.py` lists them for the areas to fix:

| Note | Meaning |
|---|---|
| `AC-LATER` | an acceptance criterion that applies "from" a later milestone than the task's own; that criterion belongs in its own task in that milestone |
| `SPIKE-LEAF` | a spike task (its Task cell opens with "Spike") that no task depends on, so its answer gates nothing |
| `SPIKE-NOBOX` | a spike task whose row states no timebox |
| `RIG-PATH` | a task whose acceptance criteria name a lab rig (`R-…`) but which has no dependency path to the task that brings that rig online |
| `RIG-LATER` | a task whose acceptance criteria name a rig that comes online in a later milestone |

A rig's online task is the first QUAL-E15 task, in milestone order, whose Task cell names the rig, so the map follows QUAL-E15 as it changes. A short name such as `R-ULTRA` stands for the first rig named `R-ULTRA-…`.

```sh
python3 depcheck.py                        # every problem, then the summary
python3 depcheck.py --quiet                # summary only
python3 depcheck.py --only LATER,CYCLE     # list some kinds; also SPLIT, NOTE-ID, M0-TBD, TBD, GATE,
                                           # AC-LATER, SPIKE-LEAF, SPIKE-NOBOX, RIG-PATH, RIG-LATER
python3 depcheck.py --task PLC-E4-T17      # one task's edges in both directions
python3 depcheck.py --graph ../../../target/plan-graph.json   # repository root, outside plan/
```

`--graph` writes the graph as JSON: `nodes` (one per task, with milestones, phase and file position), `edges` (`from`, `to`, `status` of `ok`, `split`, `gated` or `later`), `tbd`, `gates`, `cycles`, `problems` and `notes`. Read an edge as "`from` depends on `to`". The script refuses to write the file anywhere under `plan/`, because a generated graph in the plan would go stale with the next edit.

## The generators: `deps_report.py`, `trace.py`, `roadmap.py`

Each one rewrites only the text between its own markers in one file:

```
<!-- BEGIN GENERATED: name -->
...
<!-- END GENERATED -->
```

Text outside the markers is written by hand and kept. Each script takes the same flags:

```sh
python3 trace.py            # rewrite the generated sections
python3 trace.py --check    # exit 1 if the file is out of date; writes nothing
python3 trace.py --init     # rewrite the whole file from the script's skeleton, discarding hand edits
```

A run that changes nothing prints `unchanged`. The output carries no timestamps, so `--check` is stable and can run in CI. A missing marker stops the script with an error that names it; restore the marker or use `--init`.

`deps_report.py` writes `dependencies.md`: the summary of every `depcheck.py` check, edge counts per area pair and per milestone pair (over the seven milestones), the external gates with the state of each gate in `plan/open-questions.md` (a gate whose ID is settled is flagged), any `TBD:` or split edges, the plan-rule notes, and the tasks with the most dependents. When `depcheck.py` finds problems, the summary gives the count and lists every cycle, LATER and M0CLOSURE problem. `roadmap.py` refuses to run on a graph with problems, and `trace.py` on epics that `epics.py` cannot read.

`trace.py` writes `traceability.md`, and `python3 trace.py --gaps` prints the gaps without writing:
- **ADRs**: every file in `plan/decisions/`, its title (line 1) and status, the RFC sections whose text names it, and the tasks whose Implements cell names it (ranges such as `ADR-0100–ADR-0107` expanded). GAP: a non-superseded ADR that no domain RFC cites, or that no task implements.
- **Success criteria**: every SC row of vision §7. Non-QUAL tasks that name it in Task or Implements deliver it; QUAL tasks that name it anywhere, and tasks whose acceptance criteria name it, verify it. GAP: no deliverer or no verifier. It also flags an SC whose first deliverer or verifier sits after the first RFC-0001 §16 exit that names it.
- **Design subsections**: every `### n.` under `## Design` in RFC-0010 to RFC-0062. A task implements it when its Implements cell cites `RFC-NNNN §n` or `§n.m`. A § reference belongs to the RFC or ADR named most recently before it in the cell; one after an ADR is skipped; a cell with neither falls back to the epic's first RFC. GAP: a subsection no task cites.
- References in any cell to an ADR, SC or Design subsection that does not exist are listed with the gaps.

`roadmap.py` writes `roadmap.md`: for each of M0a, M0b, M1a, M1b, M2, M3 and M4, the goal and exit criteria quoted from RFC-0001 §16, the SCs first required at each exit, task counts per area, the critical path inside each milestone and the longest chain from project start (counted in tasks; the epics carry no durations), the tasks with zero slack, tasks per step and the epic work streams, the spike and decision tasks with the open questions they answer, and the open questions needed by the milestone. It also quotes RFC-0001 §16's "Rules for every milestone", "Scope-cut order" and "Recurring operations" subsections. The "Start here" list orders M0 by dependencies, M0a's tasks first, taking first the ready task with the most transitive dependents; `--start N` sets its length (30 by default).

`roadmap.py` also writes the `vision-milestones` section of `plan/00-vision-and-scope.md` §8. RFC-0001 §16 and the epics are authoritative for milestone scope, and vision §8 is derived from them: one checklist per milestone holding §16's deliverable bullets, its exit criteria and the milestone's task count. Change a milestone in RFC-0001 §16, never in the vision, and re-run `roadmap.py`; `--check` fails while vision §8 is stale. `--init` rewrites `roadmap.md` from its skeleton but only fills the vision's markers.

## The estimators: `sizing.py`, `schedsim.py`

These scripts turn the task graph into effort, agent-run and schedule ranges for planning. AI agents build Syntropy (ADR-0021), so the scripts count effort units and agent runs, and schedule against the constraints that bind an agent-built project: agent concurrency, serial lab rigs, the maintainers' decision and approval latency, and gate throughput. Nothing in them counts people. The epics carry no durations, so every figure the scripts print is an `[estimate]` from the assumptions below, never a measurement or a commitment. Neither script writes a file. Measured effort from the first M0 tasks supersedes them.

### `sizing.py`

Gives each task a size class from a keyword heuristic over its Task and Acceptance criteria cells, and each class a range of effort units (EU). It sums the effort per milestone and per work pool, and counts the agent runs it implies. The heuristic was checked by hand against a stratified sample of 60 tasks.

An effort unit is a relative size: how much implementation, testing and evidence a task needs compared with the others. It is not an hour or a day of anyone's time. The class ranges are relative to each other.

| Class | Rule | Effort units (low–high) |
|---|---|---|
| XL | a kernel suite in COMP-E3, -E5, -E6 or -E8 (the task names kernels, GEMV, GEMM, attention, scans, Slang or the MoE ops; binding, enumeration and spike rows excluded) | 15–35 |
| L | the acceptance criteria hold an efficiency, tok/s or peak target or a rig-measured SC; or the task text exceeds 70 words; or it is a Tier 2/3 port or a coverage-level task | 8–15 |
| M | task text over 35 words | 4–7 |
| S | everything else | 2–4 |
| D | decision rows and other maintainer-only rows: effort only, lead time not counted | 1–2 |

The D rule matches a Task cell that opens with "Maintainer decision", or that records a review only a maintainer can make. It does not match a plain "Decision" row or a spike that feeds a decision, which agents carry out.

Assumptions:
- **Work pools** group tasks by the kind of work, for review routing and the effort split: `kern` is COMP-E3 to COMP-E9, `ml` the rest of COMP plus QUAL-E4 to QUAL-E7, `infra` the rest of QUAL, `sys` PLC and TIER, `dist` DIST and `prod` PROD. Pools are not teams; any agent takes any task.
- **Agent runs.** Each task is one change set: an author run produces it and an independent review run, never the author, checks it (ADR-0021 decision 5), so a task costs at least two runs. `--units-per-run U` gives a task of E units ⌈E ÷ U⌉ author runs, each with its review run. `--retry R` adds R × the author runs, with their reviews, for failed gates and rejected reviews. The defaults, one author run per task and no retries, give the floor, not a forecast. Class D rows are maintainer work and cost no agent runs.
- **Maintainer tasks** are rows marked **Maintainer action**, or whose Task cell opens with "Maintainer decision". A row can be a maintainer task without being class D, when agents do the rest of it. `--json` flags them, along with the tasks that touch an approval epic and the lab rigs each task names.
- **Milestones.** Effort, runs and maintainer tasks are reported for each of the seven milestones; a plain M0 or M1 cell counts as M0b or M1a.
- **Approval epics** are the ones whose changes touch the security-critical paths that need a maintainer's approval (ADR-0021 decision 5): DIST-E2, DIST-E5, DIST-E17, PROD-E12, QUAL-E1, QUAL-E2, QUAL-E16 and QUAL-E17.
- **Not counted:** CI minutes and rented GPU time; recurring operations (ggml pin bumps, lab operations, releases, new model families); documentation, support and rework after spikes.

```sh
python3 sizing.py                                           # effort and runs per milestone and pool
python3 sizing.py --units-per-run U --retry R
python3 sizing.py --list XL                                 # the tasks in one class
python3 sizing.py --json                                    # class, pool, effort, maintainer and approval flags, rigs
python3 sizing.py --tasks tasks.json                        # read a saved epics.py --json dump
```

### `schedsim.py`

Runs a discrete-event schedule of the dependency graph and prints the elapsed time at which each milestone's last task finishes, cumulatively, for each agent-concurrency setting. A task starts once everything it depends on is done, then passes through these stages, skipping any of zero length:

| Stage | Resource it holds | Length |
|---|---|---|
| maintainer | a maintainer slot, after `--maintainer-latency`; only for maintainer tasks | the D class's effort; a class D task is only this stage |
| agent | an agent slot (`--agents`) | the task's effort units |
| rig | every lab rig the task names, all at once; the agent is not held while the lab job runs | `--rig-share` × the task's effort |
| gate | a gate slot (`--gate-slots`) | `--gate-units` |
| review | an agent slot: the independent review run | `--review-share` × the task's effort |
| approval | a maintainer slot, after `--approval-latency`; only for changes in an approval epic | `--approval-units` |

- Each queue takes the ready task from the earliest milestone first (`--ungated` drops this rule), then the one with the longest remaining effort path. A rig queue starts any waiting job whose rigs are all free.
- Time is counted in effort units: one EU of elapsed time is how long one agent takes for one effort unit of work. Nothing has measured that yet, so the output stays in EU unless `--unit-days` converts it to calendar days. Latencies and gate times are given in EU too.
- Without `--agents`, it runs the presets 8, 16, 32, 64 and unlimited (`0`). They are scenario inputs, not recommendations.
- Each row also reports the busiest rig with its busy time, and the maintainers' busy time, so the binding constraint shows.
- With unlimited agents and `--rig-share 0`, each milestone's exit equals the longest effort-weighted dependency chain from project start, which checks the simulator against the graph.

`--floor` prints each milestone's effort-weighted critical path with unlimited agents and no rig, maintainer or gate constraint: the shortest the milestone can be. `--cut FROM:TO` removes one edge, so a proposed re-sequencing can be priced before the epics change.

The defaults are floors or neutral choices: no maintainer, approval or gate time, no review time, one maintainer slot, unlimited gate slots, and a rig session as long as the task's own effort. Set the latencies and `--maintainer-slots` from the observed queue before reading the result as a schedule. The figures are optimistic, and the script prints no date on its own:
- every agent run succeeds first time (`sizing.py --retry` prices retries in runs, not time);
- rigs exist and decisions are taken when first needed;
- no spike reverses a default;
- no time goes to recurring operations, documentation, support or stabilisation.

```sh
python3 schedsim.py                                   # agent presets, low and high cases
python3 schedsim.py --agents 16 --maintainer-latency L --approval-latency A --approval-units U
python3 schedsim.py --agents 0 --rig-share 0          # no rig contention; equals the longest chains
python3 schedsim.py --case high --ungated
python3 schedsim.py --floor                           # unconstrained floor per milestone
python3 schedsim.py --floor --cut PLC-E9-T01:PROD-E2-T04
python3 schedsim.py --tasks tasks.json --graph ../../../target/plan-graph.json
```

By default both scripts read the epics in-process, through the functions behind `epics.py --json` and `depcheck.py --graph`, and stop if either reports problems. `--tasks` and `--graph` read saved outputs of those two commands instead. As with `depcheck.py --graph`, keep saved graphs outside `plan/`.

## `milestone_split.py`

Assigns every plain M0 and M1 task to RFC-0001 §16's split milestones by rewriting its Milestone cell, and can cut false dependency edges in the same pass:

| Milestone | Which tasks |
|---|---|
| M0a | the walking-skeleton seeds (`SEEDS` in the script, any `--seed`, and every task whose cell already says M0a) with their full dependency closure, computed from the graph after the cuts |
| M1b | plain M1 tasks that the Windows, macOS, Metal and Apple rule matches (the COMP-E6 Metal epic, or a Task cell that names one of those platforms and not Linux, CUDA or every OS; `KEEP_M1A` exempts cross-platform rows), and tasks whose cell already says M1b or that `M1B_LIST` names, with the M1 tasks that depend on those |
| M0b, M1a | every other plain M0 or M1 task, including tasks added later |

An M1a task may not depend on an M1b task. `--mode` decides what happens to an M1 task that depends on a rule-matched task. `cut` (the default) cuts the edge, keeps the dependent in M1a and moves the platform task to M1b. It then adds one M1b extension task per such dependent, in the dependent's epic with the epic's next free number. The new row's Task cell reads "Windows and macOS part of <ID>: <short title>". It depends on <ID>, on the platform tasks whose edges were cut, and on the task that brings each Windows and macOS rig online (`M1B_RIGS`: R-WIN, R-MACF, R-MBP, R-ULTRA-A) where no path reaches it already. Its acceptance criterion is that <ID>'s criteria hold on those rigs, and its Implements cell is <ID>'s. `pull` keeps the platform task in M1a and lists it, with its M1a dependents, as a task to split by hand; `push` moves the dependent to M1b too. A cell that already names a split milestone is kept, and the script stops if the split would contradict one.

Cuts are `FROM:TO` (FROM no longer depends on TO) or `FROM:TO=NEW` (it depends on NEW instead). `CUTS` holds false edges: edges whose dependent stays testable without the prerequisite; `--cut` adds more and `--no-cuts` drops `CUTS`. A range in a Depends cell that covers TO is written out without it.

```sh
python3 milestone_split.py --dry-run                  # mapping, counts, and a depcheck of the result
python3 milestone_split.py --dry-run --list           # also every task in M0a and M1b, and the extension rows
python3 milestone_split.py --dry-run --mode pull      # keep platform tasks that M1a work needs in M1a
python3 milestone_split.py --dry-run --cut QUAL-E2-T02:QUAL-E12-T03
python3 milestone_split.py                            # rewrite the epic files
```

`--dry-run` writes nothing in `plan/`: it rewrites a copy of the epics, extension rows included, in a temporary directory and runs `depcheck.py` on it. Without it the epic files are rewritten in place and checked. Record the cuts in `../dependency-fixes.md`, then run the checker and the generators.

## `naming_check.py`

The D5 naming check (AGENTS.md, "Naming (D5)") as a script. It matches each line of the text against the banned-terms list, one case-insensitive regular expression per non-blank line, read by the same `load_banned` that `gh_sync.py --banned` uses. A finding names a location only, `path:line` or `commit message:line`, so the output never shows a term or the list. Paths are checked too. A path that matches is never printed: it is reported by its position, such as `staged path #3`, which stands in for it in that file's findings.

```sh
python3 naming_check.py --staged              # files staged for commit, as the index holds them
python3 naming_check.py --base origin/main    # files this branch changed, as committed at HEAD
python3 naming_check.py --message FILE        # a commit message file; comments and the -v diff are skipped
python3 naming_check.py --stdin --label NAME  # text on standard input
python3 naming_check.py PATH...               # files on disk
python3 test_naming_check.py                  # its unit tests
```

The list is the first that is set of `--banned FILE`, `SYNTROPY_BANNED_TERMS` and `git config syntropy.bannedTerms`. With none set, the check prints one notice line and exits 0; `--require` makes that an error. Exit status is 0 when nothing matches, 1 on a match and 2 on an error. Whole files are checked, and a file with a NUL byte is skipped as binary.

The opt-in git hooks in `.githooks/` run it. `git config core.hooksPath .githooks` turns them on:
- `pre-commit` runs `cargo fmt --all --check` when Rust files are staged, then `epics.py`, `depcheck.py` and the three generators with `--check` on a copy of the staged `plan/` when files under `plan/` are staged (deletions included), then `naming_check.py --staged`.
- `commit-msg` checks for a `Signed-off-by` line that matches the author and a conventional-commit subject, then runs `naming_check.py --message`. Those two checks are a local subset of `cargo xtask ci dco` and `cargo xtask ci pr-title`, which CI runs over a PR's data and which stay authoritative.

The hooks are POSIX `sh` and build nothing. They read their shared `common.sh` from their own directory, so `core.hooksPath` can point at a `.githooks` anywhere, and they take the plan tools and the naming check from the repository being committed to. A check whose input is missing (nothing of its kind staged, no `cargo`, no Python 3, no list, no tools in the repository) prints one notice line, and the other checks still run.

## After editing an epic file

Run the checker, then the generators:

```sh
python3 depcheck.py && python3 deps_report.py && python3 trace.py && python3 roadmap.py
```

A rewritten or new dependency must be in one of the forms above; record why an edge or milestone changed in `../dependency-fixes.md`, under the rule it follows. Read the diffs of the generated sections: a new GAP, problem or flagged gate is a finding for the area concerned. Re-run `trace.py` after changing an ADR, an RFC's Design headings or vision §7, and `roadmap.py` after changing RFC-0001 §16 or `plan/open-questions.md`; `roadmap.py` also rewrites vision §8.
