#!/usr/bin/env python3
"""Estimate task sizes, task effort and agent runs from the epic files.

Every number this script prints is an [estimate]. AI agents build Syntropy
by carrying out the tasks (ADR-0021), so effort is counted in task-effort
units and agent runs, never in people. The epics carry no
durations, so each task is put in a size class by a keyword heuristic over
its Task and Acceptance criteria cells, and each class is given a range of
effort units.

An effort unit (EU) is a relative size. A class's range says how much
implementation, testing and evidence the task needs compared with the
others; it is not an hour or a day of anyone's time. The ranges are
relative to each other, and the first measured tasks replace them.

Size classes (checked by hand against a stratified sample of 60 tasks):

    XL  a kernel suite in COMP-E3, -E5, -E6 or -E8: the task names kernels,
        GEMV, GEMM, attention, scans, Slang or the MoE ops; binding,
        enumeration and spike rows are excluded              15-35 EU
    L   the acceptance criteria hold an efficiency, tok/s or peak target or
        a rig-measured SC; or the task text exceeds 70 words; or it is a
        Tier 2/3 port or a coverage-level task                8-15 EU
    M   task text over 35 words                                4-7 EU
    S   everything else                                        2-4 EU
    D   decision rows and other maintainer-only rows: effort only,
        lead time not counted                                   1-2 EU

Work pools group tasks by the kind of work, for review routing and for the
effort split: kern (COMP-E3 to COMP-E9), ml (the rest of COMP, plus
QUAL-E4 to QUAL-E7), infra (the rest of QUAL), sys (PLC, TIER),
dist (DIST), prod (PROD). Pools are not teams; any agent takes any task.

Agent runs. Each task is one change set. An author run produces it and an
independent review run, never the author, checks it (ADR-0021 decision 5),
so every task costs at least two runs. With --units-per-run U, a task of E
units takes ceil(E / U) author runs, each with its own review run. --retry R
adds R x the author runs for failed gates and rejected reviews, each with
its review run. The defaults (one author run per task, no retries) give the
floor, not a forecast. D rows are counted as maintainer work, not agent
runs.

Maintainer tasks are the rows marked **Maintainer action**, or whose Task
cell opens with "Maintainer decision": work only a maintainer can do.
schedsim.py queues them for the maintainers. A row can be a maintainer task
without being class D, when agents do the rest of it.

Not included: CI minutes and rented GPU time, recurring operations (ggml pin
bumps, lab operations, releases, new model families), documentation,
support and rework after spikes.

Inputs: by default the script reads the epics in-process, through the same
functions behind `epics.py --json`. `--tasks FILE` reads a saved
`epics.py --json` dump instead.

Usage:

    python3 sizing.py                          # effort and runs per milestone and pool
    python3 sizing.py --units-per-run U --retry R
    python3 sizing.py --list XL                # the tasks in one class
    python3 sizing.py --list D                 # the decision and maintainer-only rows
    python3 sizing.py --json                   # class, pool, effort and flags of every task
    python3 sizing.py --tasks tasks.json       # read an epics.py --json dump
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import epics  # noqa: E402

# Milestones in delivery order; a plain M0 or M1 cell counts as M0b or M1a.
MILESTONES = epics.PHASES
CLASSES = ("S", "M", "L", "XL", "D")
POOLS = ("sys", "kern", "ml", "dist", "prod", "infra")

# Effort units per class: (low case, high case). Relative sizes, not time.
EFFORT = {"S": (2, 4), "M": (4, 7), "L": (8, 15), "XL": (15, 35),
          "D": (1, 2)}
CASES = {"low": 0, "high": 1}

KERNEL_SUITE_EPICS = {"COMP-E3", "COMP-E5", "COMP-E6", "COMP-E8"}
KERNEL_EPICS = {"COMP-E3", "COMP-E4", "COMP-E5", "COMP-E6", "COMP-E7",
                "COMP-E8", "COMP-E9"}
ML_QUAL_EPICS = {"QUAL-E4", "QUAL-E5", "QUAL-E6", "QUAL-E7"}

# Epics whose changes touch the security-critical paths that ADR-0021
# decision 5 reserves for a maintainer's approval: keys and signing,
# admission, the update client, CI and release configuration, and the
# hardware dispatcher.
APPROVAL_EPICS = {"DIST-E2", "DIST-E5", "DIST-E17", "PROD-E12", "QUAL-E1",
                  "QUAL-E2", "QUAL-E16", "QUAL-E17"}

PERF_RE = re.compile(
    r"η|tok/s|≥ ?0\.\d+ ?×|peak|SC-(1|2|4|5|7|12|13|15|17|21)\b.*rig|on R-[A-Z]",
    re.I)
KERNEL_RE = re.compile(
    r"kernel|GEMV|GEMM|attention|scan|Slang|Moe(Grouped|Route)|matmul", re.I)
DECISION_RE = re.compile(
    r"^\**Maintainer decisions?\b|maintainer decision recorded", re.I)
PORT_RE = re.compile(r"iOS|Android|Tier 2/3|every op|coverage level")
NOT_SUITE_RE = re.compile(r"binding|enumeration|Spike")
# Lab rigs named in a row (ADR-0610 names, plus pair and fabric rigs).
RIG_RE = re.compile(r"\bR-[A-Z][A-Z0-9]*(?:-[AB])?\b")
MAINTAINER_RE = re.compile(r"\*\*Maintainer action\*\*"
                           r"|^\**Maintainer decisions?\b")


def size_class(rec):
    """Return the size class of one task record."""
    text, ac = rec["task"], rec["acceptance"]
    words = len(text.split())
    if DECISION_RE.search(text) and len(text) < 900:
        return "D"
    if (rec["epic"] in KERNEL_SUITE_EPICS and KERNEL_RE.search(text)
            and not NOT_SUITE_RE.search(text)):
        return "XL"
    if PERF_RE.search(ac) or words > 70 or PORT_RE.search(text):
        return "L"
    if words > 35:
        return "M"
    return "S"


def skill_pool(rec):
    """Return the work pool of one task (a kind of work, not a team)."""
    area, epic = rec["area"], rec["epic"]
    if epic in KERNEL_EPICS:
        return "kern"
    if area == "COMP" or epic in ML_QUAL_EPICS:
        return "ml"
    if area == "QUAL":
        return "infra"
    return {"PLC": "sys", "TIER": "sys", "DIST": "dist", "PROD": "prod"}[area]


def effort(rec, case):
    """Effort units for one task in the 'low' or 'high' case."""
    return EFFORT[size_class(rec)][CASES[case]]


def is_maintainer_task(rec):
    """True when the row is marked Maintainer action or is a decision."""
    return bool(MAINTAINER_RE.search(rec["task"]))


def agent_work(rec):
    """True when agents do some of the task (every class but D)."""
    return size_class(rec) != "D"


def needs_approval(rec):
    """True when the task's change needs a maintainer's approval."""
    return rec["epic"] in APPROVAL_EPICS and agent_work(rec)


def rigs(rec):
    """The lab rigs a task's text or acceptance criteria name, sorted."""
    return sorted(set(RIG_RE.findall(rec["task"] + " " + rec["acceptance"])))


def agent_runs(rec, case, units_per_run=None, retry=0.0):
    """Agent runs for one task: author runs plus one review run each."""
    if not agent_work(rec):
        return 0
    author = math.ceil(effort(rec, case) / units_per_run) if units_per_run else 1
    return 2 * author * (1 + retry)


def phase(rec):
    """The milestone a task's Milestone cell puts it in, as a PHASES name."""
    return epics.PHASES[epics.milestones_of(rec["milestone"])[0]]


def load_tasks(path=None):
    """Task records from an `epics.py --json` dump, or read in-process."""
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    records, problems = epics.load()
    if problems:
        sys.exit(f"epics.py reports {len(problems)} problems; fix them first")
    return records


def fmt(x):
    return f"{x:,.0f}"


def report(records, units_per_run, retry):
    classes = Counter(size_class(r) for r in records)
    print(f"{len(records)} tasks. Every figure below is an [estimate].")
    print()
    print("| Class | Tasks | Effort units per task (low-high) |")
    print("|---|---|---|")
    for c in ("XL", "L", "M", "S", "D"):
        lo, hi = EFFORT[c]
        print(f"| {c} | {classes[c]} | {lo}-{hi} |")
    print()
    basis = (f"{units_per_run} effort units per author run"
             if units_per_run else "one author run per task (the floor)")
    print(f"Agent runs at {basis}, retry factor {retry}, and one "
          "independent review run per author run:")
    print()
    head = ("| Milestone | Tasks | Maintainer tasks | Effort units | "
            "Agent runs |")
    print(head)
    print("|---" * (head.count("|") - 1) + "|")
    total_eu = [0, 0]
    for m in MILESTONES + ("Total",):
        rs = (records if m == "Total"
              else [r for r in records if phase(r) == m])
        eu = [sum(effort(r, c) for r in rs) for c in CASES]
        runs = [sum(agent_runs(r, c, units_per_run, retry) for r in rs)
                for c in CASES]
        own = sum(is_maintainer_task(r) for r in rs)
        row = (f"| {m} | {len(rs)} | {own} | {fmt(eu[0])}-{fmt(eu[1])} | "
               f"{fmt(runs[0])}-{fmt(runs[1])} |")
        print(row)
        if m == "Total":
            total_eu = eu
    print()
    print("| Pool | Tasks | Effort units | Share of effort |")
    print("|---|---|---|---|")
    for p in POOLS:
        rs = [r for r in records if skill_pool(r) == p]
        eu = [sum(effort(r, c) for r in rs) for c in CASES]
        share = [100 * eu[i] / total_eu[i] for i in range(2)]
        print(f"| {p} | {len(rs)} | {fmt(eu[0])}-{fmt(eu[1])} | "
              f"{share[0]:.0f}-{share[1]:.0f}% |")
    print()
    rig_tasks = [r for r in records if rigs(r) and agent_work(r)]
    appr = [r for r in records if needs_approval(r)]
    print(f"{len(rig_tasks)} tasks name a lab rig, a serial resource; "
          f"{len(appr)} touch a path that needs a maintainer's approval "
          "(ADR-0021 decision 5).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", metavar="FILE",
                    help="read an epics.py --json dump instead of the epics")
    ap.add_argument("--units-per-run", type=float, metavar="U",
                    help="effort units one author run completes (default: "
                         "one author run per task, the floor)")
    ap.add_argument("--retry", type=float, default=0.0, metavar="R",
                    help="extra author runs per author run, for failed gates "
                         "and rejected reviews (0, the floor)")
    ap.add_argument("--list", metavar="CLASS", choices=CLASSES,
                    help="list the tasks in one size class")
    ap.add_argument("--json", action="store_true",
                    help="print class, pool, effort and flags of every task")
    args = ap.parse_args()
    if args.units_per_run is not None and args.units_per_run <= 0:
        ap.error("--units-per-run must be positive")
    if args.retry < 0:
        ap.error("--retry must not be negative")
    records = load_tasks(args.tasks)
    if args.json:
        json.dump([{"id": r["id"], "milestone": r["milestone"],
                    "phase": phase(r),
                    "class": size_class(r), "pool": skill_pool(r),
                    "units_low": effort(r, "low"),
                    "units_high": effort(r, "high"),
                    "maintainer": is_maintainer_task(r),
                    "approval": needs_approval(r),
                    "rigs": rigs(r)} for r in records],
                  sys.stdout, indent=1, ensure_ascii=False)
        print()
        return 0
    if args.list:
        for r in records:
            if size_class(r) == args.list:
                print(f"{r['id']} ({r['milestone']}, {skill_pool(r)}): "
                      f"{r['task'][:100]}")
        return 0
    report(records, args.units_per_run, args.retry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
