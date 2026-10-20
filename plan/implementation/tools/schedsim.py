#!/usr/bin/env python3
"""Simulate the plan's schedule under the agent model's constraints.

Every number this script prints is an [estimate]. AI agents build Syntropy
(ADR-0021), and agent concurrency is cheap, so the schedule is bounded by
four constraints instead of people:

  1. agent concurrency: how many agent runs work at once (--agents);
  2. the lab rigs: each rig is a serial resource, one job at a time;
  3. the maintainers: maintainer tasks and approvals queue for them,
     --maintainer-slots at a time, each after a decision or approval
     latency;
  4. gate throughput: each change set holds a gate slot while the gates
     run (--gate-slots, --gate-units).

It takes each task's size class and effort units from sizing.py and the
dependency edges from depcheck.py, then runs a discrete-event schedule.
A task starts once everything it depends on is done, and passes through
these stages in order, skipping any of zero length:

    maintainer  a maintainer task (marked **Maintainer action**, or a
              maintainer decision) waits --maintainer-latency, then holds a
              maintainer slot for the D class's effort; a class D task is
              only this stage
    agent     an agent slot for the task's effort units
    rig       every rig the task names, all at once, for --rig-share x its
              effort; the agent is not held while the lab job runs
    gate      a gate slot for --gate-units
    review    an agent slot for --review-share x its effort: the
              independent review run, never the author (ADR-0021 decision 5)
    approval  for a change in an approval epic (sizing.APPROVAL_EPICS, the
              security-critical paths of ADR-0021 decision 5),
              --approval-latency, then a maintainer slot for
              --approval-units

Each queue takes the ready task from the earliest milestone first (unless
--ungated), then the one with the longest remaining effort path. A rig
queue starts any waiting job whose rigs are all free. The result is the
elapsed time at which each milestone's last task finishes, cumulatively.

Time is counted in effort units (EU): one EU of elapsed time is how long
one agent takes for one effort unit of task work. Nothing has measured
that yet, so the output stays in EU unless --unit-days converts it.
Latencies and gate times are given in EU too.

Defaults are floors or neutral choices, not forecasts: no maintainer,
approval or gate time, no review time, one maintainer slot, unlimited gate slots, and a
rig session as long as the task's own effort (--rig-share 1). The agent
presets are scenario inputs, not recommendations. The figures are
optimistic: every agent run succeeds first time (use sizing.py --retry for
runs, not time), rigs and decisions exist when first needed, no spike
reverses a default, and no time goes to recurring operations or
stabilisation.

--floor prints the effort-weighted critical path of each milestone with
unlimited agents and no rig, maintainer or gate constraint: the shortest a
milestone can be. --cut FROM:TO removes one edge, to test what a
re-sequencing saves.


Inputs: by default the script reads the epics and builds the graph
in-process, through the same functions behind `epics.py --json` and
`depcheck.py --graph`. `--tasks FILE` and `--graph FILE` read saved outputs
of those two commands instead.

Usage:

    python3 schedsim.py                        # agent presets, both cases
    python3 schedsim.py --agents 16 --maintainer-latency L --approval-units U
    python3 schedsim.py --agents 0 --rig-share 0   # unlimited agents, no rigs
    python3 schedsim.py --case high --ungated
    python3 schedsim.py --floor                # critical path per milestone
    python3 schedsim.py --floor --cut PLC-E9-T01:PROD-E2-T04
    python3 schedsim.py --graph ../../../target/plan-graph.json
"""

import argparse
import heapq
import itertools
import json
import math
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depcheck  # noqa: E402
import sizing  # noqa: E402

RANK = {m: i for i, m in enumerate(sizing.MILESTONES)}
INF = math.inf

# Agent concurrency presets (0 = unlimited). Scenario inputs only.
PRESETS = (8, 16, 32, 64, 0)

Stage = namedtuple("Stage", "kind dur latency rigs")


def load_edges(path=None):
    """(from, to) pairs from a `depcheck.py --graph` file, or in-process."""
    if path:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return [(e["from"], e["to"]) for e in doc["edges"]]
    result = depcheck.analyse()
    if result["problems"]:
        sys.exit(f"depcheck.py reports {len(result['problems'])} problems; "
                 f"fix them first")
    return [(e["from"], e["to"]) for e in result["edges"]]


def build(records, edges, cuts=()):
    tasks = {r["id"]: r for r in records}
    deps, rdeps = defaultdict(set), defaultdict(set)
    for a, b in edges:
        if (a, b) in cuts or a not in tasks or b not in tasks:
            continue
        deps[a].add(b)
        rdeps[b].add(a)
    return tasks, deps, rdeps


def topo(tasks, rdeps):
    """Tasks in dependency order (prerequisites first)."""
    indeg = defaultdict(int)
    for x in tasks:
        for y in rdeps[x]:
            indeg[y] += 1
    ready = sorted(x for x in tasks if indeg[x] == 0)
    out = []
    while ready:
        x = ready.pop()
        out.append(x)
        for y in rdeps[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                ready.append(y)
    return out


def longest_down(tasks, rdeps, dur):
    """Longest weighted path from each task to the end of the graph."""
    memo = {}
    for x in reversed(topo(tasks, rdeps)):
        memo[x] = dur[x] + max((memo[y] for y in rdeps[x]), default=0)
    return memo


def stages_for(rec, case, p):
    """The stages one task passes through, in order."""
    e = sizing.effort(rec, case)
    out = []
    if sizing.is_maintainer_task(rec) or not sizing.agent_work(rec):
        own = e if not sizing.agent_work(rec) else sizing.EFFORT["D"][
            sizing.CASES[case]]
        out.append(Stage("maintainer", own, p.maintainer_latency, None))
    if not sizing.agent_work(rec):
        return out
    out.append(Stage("agent", e, 0, None))
    rigs = frozenset(sizing.rigs(rec))
    if rigs and p.rig_share > 0:
        out.append(Stage("rig", e * p.rig_share, 0, rigs))
    if p.gate_units > 0:
        out.append(Stage("gate", p.gate_units, 0, None))
    if p.review_share > 0:
        out.append(Stage("agent", e * p.review_share, 0, None))
    if sizing.needs_approval(rec) and (p.approval_units > 0
                                       or p.approval_latency > 0):
        out.append(Stage("maintainer", p.approval_units, p.approval_latency, None))
    return out


def simulate(tasks, deps, rdeps, case, p, agents, gated=True):
    """Elapsed EU to each cumulative milestone exit, plus resource use."""
    dur = {x: sizing.effort(tasks[x], case) for x in tasks}
    down = longest_down(tasks, rdeps, dur)
    plan = {x: stages_for(tasks[x], case, p) for x in tasks}

    def key(x):
        rank = RANK[sizing.phase(tasks[x])] if gated else 0
        return (rank, -down[x], x)

    free = {"agent": agents or INF, "gate": p.gate_slots or INF,
            "maintainer": p.maintainer_slots}
    queue = {"agent": [], "gate": [], "maintainer": [], "rig": []}
    rig_busy, rig_time = set(), defaultdict(float)
    maintainer_time = 0.0
    events, tick = [], itertools.count()
    ix = {x: 0 for x in tasks}
    remaining = {x: len(deps[x]) for x in tasks}
    done = {}

    def enqueue(x, now):
        st = plan[x][ix[x]]
        if st.latency > 0:
            heapq.heappush(events, (now + st.latency, next(tick), "avail", x))
        else:
            heapq.heappush(queue[st.kind], (key(x), x))

    def advance(x, now):
        while ix[x] < len(plan[x]) and plan[x][ix[x]].dur <= 0 \
                and plan[x][ix[x]].latency <= 0:
            ix[x] += 1
        if ix[x] < len(plan[x]):
            enqueue(x, now)
            return
        done[x] = now
        for y in rdeps[x]:
            remaining[y] -= 1
            if remaining[y] == 0:
                advance(y, now)

    def start(x, now):
        st = plan[x][ix[x]]
        heapq.heappush(events, (now + st.dur, next(tick), "end", x))

    def dispatch(now):
        nonlocal maintainer_time
        for kind in ("maintainer", "agent", "gate"):
            q = queue[kind]
            while free[kind] > 0 and q:
                x = heapq.heappop(q)[1]
                free[kind] -= 1
                if kind == "maintainer":
                    maintainer_time += plan[x][ix[x]].dur
                start(x, now)
        waiting, queue["rig"] = sorted(queue["rig"]), []
        for k, x in waiting:
            need = plan[x][ix[x]].rigs
            if need & rig_busy:
                queue["rig"].append((k, x))
                continue
            rig_busy.update(need)
            for r in need:
                rig_time[r] += plan[x][ix[x]].dur
            start(x, now)
        heapq.heapify(queue["rig"])

    for x in sorted(tasks):
        if remaining[x] == 0:
            advance(x, 0.0)
    now = 0.0
    while len(done) < len(tasks):
        dispatch(now)
        if not events:
            sys.exit("no task can start: the graph has a cycle")
        now = events[0][0]
        while events and events[0][0] == now:
            _, _, what, x = heapq.heappop(events)
            st = plan[x][ix[x]]
            if what == "avail":
                if st.dur > 0:
                    heapq.heappush(queue[st.kind], (key(x), x))
                else:
                    ix[x] += 1
                    advance(x, now)
                continue
            if st.kind == "rig":
                rig_busy.difference_update(st.rigs)
            else:
                free[st.kind] += 1
            ix[x] += 1
            advance(x, now)

    finish = defaultdict(float)
    for x, t in done.items():
        m = sizing.phase(tasks[x])
        finish[m] = max(finish[m], t)
    out, run = {}, 0.0
    for m in sizing.MILESTONES:
        run = max(run, finish[m])
        out[m] = run
    busiest = max(rig_time.items(), key=lambda kv: kv[1], default=("-", 0.0))
    return out, busiest, maintainer_time


def floor(tasks, deps, case):
    """Effort-weighted critical path inside each milestone (EU)."""
    out = {}
    for m in sizing.MILESTONES:
        ids = [x for x in tasks if sizing.phase(tasks[x]) == m]
        inside = set(ids)
        up = {}

        def f(x):
            if x not in up:
                up[x] = sizing.effort(tasks[x], case) + max(
                    (f(y) for y in deps[x] if y in inside), default=0)
            return up[x]

        if not ids:
            continue
        end = max(ids, key=f)
        chain, x = [end], end
        while True:
            prev = [y for y in deps[x] if y in inside]
            if not prev:
                break
            x = max(prev, key=f)
            chain.append(x)
        out[m] = (f(end), list(reversed(chain)))
    return out


def parse_agents(text):
    try:
        vals = tuple(int(v) for v in text.split(","))
    except ValueError:
        sys.exit(f"bad --agents '{text}'; give counts like 16 or 8,32 "
                 f"(0 = unlimited)")
    if any(v < 0 for v in vals):
        sys.exit("--agents counts must not be negative")
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", metavar="FILE",
                    help="read an epics.py --json dump instead of the epics")
    ap.add_argument("--graph", metavar="FILE",
                    help="read a depcheck.py --graph file for the edges")
    ap.add_argument("--agents", metavar="N[,N...]",
                    help="agent concurrency, one value or a list; 0 means "
                         "unlimited (default: the presets "
                         + ",".join(map(str, PRESETS)) + ")")
    ap.add_argument("--case", choices=("low", "high", "both"), default="both")
    ap.add_argument("--rig-share", type=float, default=1.0, metavar="F",
                    help="rig session length as a share of the task's effort "
                         "(1.0; 0 removes rig contention)")
    ap.add_argument("--maintainer-slots", type=int,
                    default=1, metavar="N",
                    help="maintainer tasks and approvals handled at once (1)")
    ap.add_argument("--maintainer-latency", type=float,
                    default=0.0, metavar="EU",
                    help="wait before a maintainer takes up a maintainer task "
                         "(0)")
    ap.add_argument("--approval-latency", type=float, default=0.0,
                    metavar="EU",
                    help="wait before a maintainer takes up an approval (0)")
    ap.add_argument("--approval-units", type=float, default=0.0,
                    metavar="EU", help="maintainer time per approval (0)")
    ap.add_argument("--gate-slots", type=int, default=0, metavar="N",
                    help="change sets in the gates at once (0 = unlimited)")
    ap.add_argument("--gate-units", type=float, default=0.0, metavar="EU",
                    help="time a change set holds a gate slot (0)")
    ap.add_argument("--review-share", type=float, default=0.0, metavar="F",
                    help="review run length as a share of the task's effort "
                         "(0)")
    ap.add_argument("--unit-days", type=float, metavar="D",
                    help="calendar days per EU of elapsed time; without it "
                         "the output stays in EU")
    ap.add_argument("--ungated", action="store_true",
                    help="do not prefer tasks from the earliest milestone")
    ap.add_argument("--floor", action="store_true",
                    help="print the unconstrained critical path per milestone")
    ap.add_argument("--cut", metavar="FROM:TO", action="append", default=[],
                    help="remove the edge FROM depends on TO (repeatable)")
    p = ap.parse_args()
    for name in ("rig_share", "maintainer_latency", "approval_latency",
                 "approval_units", "gate_units", "review_share"):
        if getattr(p, name) < 0:
            ap.error(f"--{name.replace('_', '-')} must not be negative")
    if p.maintainer_slots < 1 or p.gate_slots < 0:
        ap.error("--maintainer-slots must be at least 1, --gate-slots at "
                 "least 0")
    if p.unit_days is not None and p.unit_days <= 0:
        ap.error("--unit-days must be positive")

    records = sizing.load_tasks(p.tasks)
    cuts = set()
    for c in p.cut:
        a, _, b = c.partition(":")
        cuts.add((a.strip(), b.strip()))
    tasks, deps, rdeps = build(records, load_edges(p.graph), cuts)
    cases = ("low", "high") if p.case == "both" else (p.case,)
    unit = "days" if p.unit_days else "EU"
    scale = p.unit_days or 1.0
    print("Every figure below is an [estimate]; see the docstring for the "
          "assumptions.")
    print()

    if p.floor:
        for case in cases:
            print(f"Unconstrained floor, {case} case (elapsed {unit}; "
                  "unlimited agents, no rig, maintainer or gate constraint):")
            for m, (eu, chain) in floor(tasks, deps, case).items():
                print(f"  {m}: {eu * scale:,.1f} {unit}, {len(chain)} tasks: "
                      f"{' → '.join(chain)}")
            print()
        return 0

    agent_sets = parse_agents(p.agents) if p.agents else PRESETS
    print(f"Elapsed {unit} from start to each cumulative milestone exit "
          f"({'ungated' if p.ungated else 'earliest milestone first'}; "
          f"rig share {p.rig_share}, maintainer slots {p.maintainer_slots}, "
          f"maintainer latency {p.maintainer_latency}, approval {p.approval_latency} + "
          f"{p.approval_units}, gates {p.gate_slots or 'unlimited'} x "
          f"{p.gate_units}, review share {p.review_share}):")
    print()
    print("| Agents | Case | " + " | ".join(sizing.MILESTONES)
          + f" | Busiest rig ({unit} busy) | Maintainers busy ({unit}) |")
    print("|---|---|" + "---|" * (len(sizing.MILESTONES) + 2))
    for n in agent_sets:
        for case in cases:
            res, (rig, rig_eu), own = simulate(
                tasks, deps, rdeps, case, p, n, gated=not p.ungated)
            print(f"| {n or 'unlimited'} | {case} | "
                  + " | ".join(f"{res[m] * scale:,.0f}"
                               for m in sizing.MILESTONES)
                  + f" | {rig} ({rig_eu * scale:,.0f}) | {own * scale:,.0f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
