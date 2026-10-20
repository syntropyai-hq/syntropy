#!/usr/bin/env python3
"""Write the generated sections of plan/implementation/roadmap.md, and
the milestone checklist in plan/00-vision-and-scope.md §8.

For each milestone (M0a, M0b, M1a, M1b, M2, M3, M4): the goal and exit
criteria quoted from RFC-0001 §16, task counts per area, the critical path
inside the milestone and the longest chain from project start (both counted
in tasks), the work streams and how they overlap, its spike and decision
tasks, and the open questions it needs (plan/open-questions.md). A "Start here" list orders the M0 tasks, M0a first, so that
the ones unblocking the most come first. RFC-0001 §16's milestone rules,
scope-cut order and recurring-operations subsections are quoted too.

Vision §8 is derived: its checklist is RFC-0001 §16's deliverables and exit
criteria per milestone, with the task count from the epics. RFC-0001 §16 and
the epics are authoritative for milestone scope.

The epics give no task sizes, so every length here is a count of tasks.

Usage:

    python3 roadmap.py            # rewrite the generated sections
    python3 roadmap.py --check    # exit 1 if the file is out of date
    python3 roadmap.py --init     # rewrite the whole file from the skeleton
    python3 roadmap.py --start 40 # length of the "Start here" list
"""

import argparse
import heapq
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import depcheck  # noqa: E402
import epics  # noqa: E402
import gendoc as g  # noqa: E402

OUT = g.IMPL_DIR / "roadmap.md"
VISION = g.PLAN_DIR / "00-vision-and-scope.md"
# RFC-0001 §16 subsections quoted in roadmap.md, by marker name.
QUOTED = {"roadmap-rules": "Rules for every milestone",
          "roadmap-cuts": "Scope-cut order",
          "roadmap-recurring": "Recurring operations"}
KIND_RE = re.compile(
    r"^(?:\*\*)?\s*(?P<kind>Spikes?|(?:Maintainer )?decisions?|"
    r"Benchmark(?=:| SC-)|Measure)\b", re.I)


# ----------------------------------------------------------------- graph

class Graph:
    def __init__(self, res):
        self.tasks = res["tasks"]
        self.deps = defaultdict(list)
        self.dependents = defaultdict(list)
        for e in res["edges"]:
            self.deps[e["from"]].append(e["to"])
            self.dependents[e["to"]].append(e["from"])
        for k in list(self.deps):
            self.deps[k] = g.sort_ids(self.deps[k])
        for k in list(self.dependents):
            self.dependents[k] = g.sort_ids(self.dependents[k])
        self.order = self._topo()
        self.depth, self.pred = self._longest(self.order, lambda t, p: True)
        self.desc = self._descendants()

    def _topo(self):
        indeg = {t: len(self.deps[t]) for t in self.tasks}
        ready = [(epics.id_key(t), t) for t, d in indeg.items() if d == 0]
        heapq.heapify(ready)
        out = []
        while ready:
            _, t = heapq.heappop(ready)
            out.append(t)
            for d in self.dependents[t]:
                indeg[d] -= 1
                if indeg[d] == 0:
                    heapq.heappush(ready, (epics.id_key(d), d))
        if len(out) != len(self.tasks):
            sys.exit("the dependency graph has a cycle; run depcheck.py")
        return out

    def _longest(self, order, keep):
        """Longest chain ending at each task, over edges keep(t, p)."""
        depth, pred = {}, {}
        for t in order:
            best, arg = 0, None
            for p in self.deps[t]:
                if p in depth and keep(t, p) and depth[p] > best:
                    best, arg = depth[p], p
            depth[t], pred[t] = best + 1, arg
        return depth, pred

    def _descendants(self):
        bit = {t: 1 << i for i, t in enumerate(self.order)}
        acc = {}
        for t in reversed(self.order):
            v = 0
            for d in self.dependents[t]:
                v |= acc[d] | bit[d]
            acc[t] = v
        return {t: bin(v).count("1") for t, v in acc.items()}

    def chain(self, end, pred):
        out = []
        while end is not None:
            out.append(end)
            end = pred[end]
        return list(reversed(out))

    def milestone(self, n):
        ids = [t for t in self.order if self.tasks[t]["first"] == n]
        inside = set(ids)
        depth, pred = self._longest(ids, lambda t, p: p in inside)
        height = {}
        for t in reversed(ids):
            height[t] = 1 + max((height[d] for d in self.dependents[t]
                                 if d in inside), default=0)
        length = max(depth.values(), default=0)
        crit = [t for t in ids if depth[t] + height[t] - 1 == length]
        end = min((t for t in ids if depth[t] == length),
                  key=epics.id_key, default=None)
        return {"ids": ids, "inside": inside, "depth": depth,
                "length": length, "critical": crit,
                "chain": self.chain(end, pred) if end else []}


# ---------------------------------------------------------------- inputs

def task_kind(rec):
    m = KIND_RE.match(rec["task"])
    if m:
        k = m["kind"].lower()
        if k.startswith("decision"):
            return "decision"
        if k.startswith("spike"):
            return "spike"
        if k in ("measure", "benchmark"):
            return "benchmark"
        return "decision"
    if re.search(r"\bspikes?\b", rec["task"], re.I):
        return "spike"
    return None


def expand_task_cell(text, tasks):
    """Task IDs named in a cell, with ranges like PLC-E2-T13–T15."""
    out = []
    for m in re.finditer(r"([A-Z]+-E\d+)-T(\d+)(?:\s*[–-]\s*T(\d+))?",
                         text or ""):
        lo = int(m[2])
        hi = int(m[3]) if m[3] else lo
        for n in range(lo, hi + 1):
            tid = f"{m[1]}-T{n:0{len(m[2])}d}"
            if tid in tasks:
                out.append(tid)
    return out


def resolution_kind(text):
    m = re.match(r"\s*(?:maintainer )?(spike|benchmark|decision|check|read)",
                 text or "", re.I)
    if not m:
        return "other"
    k = m[1].lower()
    return {"read": "check"}.get(k, k)


def needed_by(text):
    m = epics.MILESTONE_RE.search(text or "")
    return epics.phase_index(m[1]) if m else None


# -------------------------------------------------------------- sections

def counts_table(tasks):
    rows, tot = [], defaultdict(int)
    for n in g.MILESTONES:
        per = defaultdict(int)
        for r in tasks.values():
            if r["first"] == n:
                per[r["area"]] += 1
                tot[r["area"]] += 1
        rows.append([g.label(n)] + [per[a] for a in g.AREAS]
                    + [sum(per.values())])
    rows.append(["Total"] + [tot[a] for a in g.AREAS]
                + [sum(tot.values())])
    return g.table(["Milestone", *g.AREAS, "Total"], rows)


def arrow(ids, tasks=None):
    if tasks is None:
        return " → ".join(ids)
    out, prev = [], None
    for t in ids:
        m = tasks[t]["first"]
        out.append(f"{t} ({g.label(m)})" if m != prev else t)
        prev = m
    return " → ".join(out)


def what(rec, limit=120):
    return g.cell(g.abridge(re.sub(r"\*\*", "", rec["task"]), limit))


def start_here(G, n_start):
    tasks = G.tasks
    m0 = {t for t in tasks if epics.major(tasks[t]["first"]) == 0}
    indeg = {t: len(G.deps[t]) for t in m0}
    step = {}
    for n in g.MILESTONES:
        if epics.major(n) == 0:
            step.update(G.milestone(n)["depth"])

    def key(t):
        # M0a before M0b: M0a never depends on M0b, so this stays a valid
        # dependency order.
        return (tasks[t]["first"], -G.desc[t], -len(G.dependents[t]),
                epics.id_key(t))

    ready = [(key(t), t) for t in m0 if indeg[t] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, t = heapq.heappop(ready)
        order.append(t)
        for d in G.dependents[t]:
            if d in indeg:
                indeg[d] -= 1
                if indeg[d] == 0:
                    heapq.heappush(ready, (key(d), d))
    rows = []
    for i, t in enumerate(order[:n_start], 1):
        rec = tasks[t]
        rows.append([i, t, rec["phase"], what(rec, 100),
                     g.compress(G.deps[t]) or "none",
                     step[t], len(G.dependents[t]), G.desc[t]])
    roots = [t for t in m0 if not G.deps[t]]
    head = (f"{len(order)} M0 tasks in a dependency order that unblocks the "
            f"most first. M0a's tasks come before M0b's. Each task comes "
            f"after everything it depends on; among the tasks that are "
            f"ready, the one with the most transitive dependents across "
            f"M0a–M4 goes first, then the one with the most direct "
            f"dependents. \"Step\" is the earliest position a task can take "
            f"in its milestone (1 + its longest chain of prerequisites in the "
            f"same milestone). {len(roots)} M0 tasks have no dependencies: "
            f"{g.compress(roots)}. The first {min(n_start, len(order))} "
            f"follow.")
    return head + "\n\n" + g.table(
        ["#", "Task", "Milestone", "What", "Depends on", "Step",
         "Direct dependents", "Unblocks (transitive, M0a–M4)"], rows)


def goal_section(rfc_text, n, scs, gates):
    title, body = g.milestone_section(rfc_text, n)
    if title is None:
        return f"RFC-0001 §16 has no {g.label(n)} subsection."
    first = sorted(sc for sc, m in gates.items() if m == n)
    lines = [f"Quoted from RFC-0001 §16, \"{g.label(n)}: {title}\":", "",
             quote(body), ""]
    if g.label(n) == "M4":
        lines.append("Success criteria first required at this exit: every SC "
                     "not already required earlier, which is "
                     + (", ".join(f"SC-{s}" for s in first) or "none")
                     + f"; and the other {len(scs) - len(first)} remain "
                     "required.")
    else:
        lines.append("Success criteria first required at this exit: "
                     + (", ".join(f"SC-{s}" for s in first) or "none") + ".")
    return "\n".join(lines)


def quote(body):
    return "\n".join(("> " + line) if line.strip() else ">"
                     for line in body.splitlines())


def milestone_section(G, n, oqs, task_oqs, info):
    tasks = G.tasks
    M = G.milestone(n)
    ids = M["ids"]
    name = g.label(n)
    if not ids:
        plain = {"M0a": "M0", "M1b": "M1"}.get(name)
        return (f"No tasks in {name}." + (
            f" A Milestone cell that says plain {plain} counts as "
            f"{epics.PLAIN[plain]} until `tools/milestone_split.py` assigns "
            "the split." if plain else ""))
    per = defaultdict(int)
    for t in ids:
        per[tasks[t]["area"]] += 1
    out = [f"**Tasks:** {len(ids)} ("
           + ", ".join(f"{a} {per[a]}" for a in g.AREAS if per[a]) + ")."]

    # Critical path inside the milestone.
    out.append("")
    out.append(f"**Critical path inside {name}:** {M['length']} tasks, "
               "assuming every earlier milestone is done. "
               f"{len(M['critical'])} {name} tasks lie on a chain of that "
               "length (no slack); the chain below is one of them.")
    out.append("")
    out.append(g.table(["Step", "Task", "What"],
                       [[i, t, what(tasks[t])]
                        for i, t in enumerate(M["chain"], 1)]))
    crit_areas = defaultdict(int)
    for t in M["critical"]:
        crit_areas[tasks[t]["area"]] += 1
    out.append("")
    out.append("Zero-slack tasks by area: "
               + ", ".join(f"{a} {crit_areas[a]}" for a in g.AREAS
                           if crit_areas[a]) + ".")

    # Longest chain from project start.
    best = max(ids, key=lambda t: (G.depth[t], [-x for x in
                                                epics.id_key(t)]))
    best = min((t for t in ids if G.depth[t] == G.depth[best]),
               key=epics.id_key)
    chain = G.chain(best, G.pred)
    out.append("")
    out.append(f"**Longest chain from project start to an {name} task:** "
               f"{len(chain)} tasks: {arrow(chain, tasks)}.")

    # Parallel work.
    steps = defaultdict(int)
    for t in ids:
        steps[M["depth"][t]] += 1
    out.append("")
    out.append("**Parallel work.** Tasks per step, where a task's step is "
               "1 + its longest chain of prerequisites inside the "
               "milestone: "
               + ", ".join(f"{s}: {steps[s]}" for s in sorted(steps))
               + f". {steps[1]} tasks can start as soon as {name} opens.")
    streams = defaultdict(list)
    for t in ids:
        streams[tasks[t]["epic"]].append(t)
    rows = []
    for ep in sorted(streams, key=lambda e: epics.id_key(e + "-T0")):
        ts = streams[ep]
        d = [M["depth"][t] for t in ts]
        waits = set()
        for t in ts:
            for p in G.deps[t]:
                if p in M["inside"] and tasks[p]["epic"] != ep:
                    waits.add(tasks[p]["epic"])
        rows.append([ep, g.cell(info.get(ep, {}).get("title", "")), len(ts),
                     min(d), max(d),
                     ", ".join(sorted(waits, key=lambda e: epics.id_key(
                         e + "-T0"))) or "–"])
    out.append("")
    out.append("Work streams are the epics with tasks in this milestone. "
               "\"First step\" and \"last step\" are the steps of the "
               "stream's earliest and latest tasks. Streams whose step "
               "ranges overlap, and that do not wait on each other, can run "
               "in parallel.")
    out.append("")
    out.append(g.table(["Stream", "Epic", "Tasks", "First step", "Last step",
                        f"Waits on ({name} streams)"], rows))

    # Spike and decision tasks.
    spikes = g.sort_ids(t for t in ids if task_kind(tasks[t]))
    out.append("")
    if spikes:
        out.append(f"**Spike and decision tasks in {name}:** {len(spikes)}.")
        out.append("")
        out.append(g.table(
            ["Task", "Kind", "What", "Answers (OQ)",
             "Direct dependents"],
            [[t, task_kind(tasks[t]), what(tasks[t], 140),
              ", ".join(task_oqs.get(t, [])) or "–",
              len(G.dependents[t])] for t in spikes]))
    else:
        out.append(f"**Spike and decision tasks in {name}:** none.")

    # Open questions needed by this milestone.
    need = [q for q in oqs if needed_by(q.get("needed by")) == n]
    if need:
        kinds = defaultdict(int)
        for q in need:
            kinds[resolution_kind(q.get("resolution"))] += 1
        linked = [f"{q['id']} ({', '.join(q['linked'])})" for q in need
                  if q.get("linked")]
        no_task = [q["id"] for q in need
                   if not expand_task_cell(q.get("task"), tasks)
                   and not q.get("linked")]
        out.append("")
        out.append(f"**Open questions needed by {name}:** {len(need)} ("
                   + ", ".join(f"{k} {kinds[k]}" for k in sorted(kinds))
                   + "). "
                   + ("Answered by no task: "
                      + ", ".join(no_task) + "." if no_task else
                      "Each one names a task that records the answer.")
                   + (" The register's Task column is empty for "
                      + ", ".join(linked) + ", but the task in brackets "
                      "cites that RFC open question; the register row "
                      "should name it." if linked else ""))
    return "\n".join(out)


def summary_table(G):
    rows = []
    for n in g.MILESTONES:
        M = G.milestone(n)
        if not M["ids"]:
            rows.append([g.label(n), 0, "–", "–", "–", "–"])
            continue
        longest = max(G.depth[t] for t in M["ids"])
        steps = defaultdict(int)
        for t in M["ids"]:
            steps[M["depth"][t]] += 1
        rows.append([g.label(n), len(M["ids"]), M["length"], longest,
                     len(M["critical"]), steps[1]])
    return g.table(["Milestone", "Tasks", "Critical path inside (tasks)",
                    "Longest chain from project start (tasks)",
                    "Zero-slack tasks", "Ready at step 1"], rows)


def build(n_start):
    res = depcheck.analyse()
    fatal = [p for p in res["problems"] if p["kind"] in depcheck.FATAL]
    if fatal:
        sys.exit(f"depcheck.py reports {len(fatal)} problems; fix them "
                 "first")
    G = Graph(res)
    tasks = G.tasks
    info = g.epic_info()
    oqs, _ = g.open_questions()
    task_oqs = defaultdict(list)
    for q in oqs:
        for t in expand_task_cell(q.get("task"), tasks):
            task_oqs[t].append(q["id"])
    records = list(tasks.values())
    for q in oqs:
        q["linked"] = ([] if expand_task_cell(q.get("task"), tasks)
                       else g.oq_task_links(q, records, info))
        for t in q["linked"]:
            if q["id"] not in task_oqs[t]:
                task_oqs[t].append(q["id"])
    rfc_text = g.rfc_path("RFC-0001").read_text(encoding="utf-8")
    scs = g.load_scs()
    gates = g.sc_gates(rfc_text, set(scs))
    sections = {
        "roadmap-start": start_here(G, n_start),
        "roadmap-counts": (
            f"Generated by `tools/roadmap.py` from {len(tasks)} tasks and "
            f"{len(res['edges'])} dependency edges. Every task sits in "
            "exactly one milestone.\n\n" + counts_table(tasks)),
        "roadmap-summary": (
            summary_table(G) + "\n\nPath lengths count tasks. The epics "
            "give no durations, so none of these figures is a time "
            "estimate."),
    }
    for n in g.MILESTONES:
        key = marker_key(n)
        sections[f"{key}-goal"] = goal_section(rfc_text, n, scs, gates)
        sections[key] = milestone_section(G, n, oqs, task_oqs, info)
    for key, title in QUOTED.items():
        body = g.rfc0001_subsection(rfc_text, title)
        sections[key] = (f"Quoted from RFC-0001 §16, \"{title}\":\n\n"
                         + quote(body) if body else
                         f"RFC-0001 §16 has no \"{title}\" subsection.")
    vision = {"vision-milestones": vision_section(rfc_text, tasks)}
    return sections, vision, rfc_text


def marker_key(n):
    return f"roadmap-{g.label(n).lower()}"


def vision_section(rfc_text, tasks):
    """Vision §8's checklist: RFC-0001 §16's deliverables and exit per
    milestone, with the number of tasks the epics put in it."""
    per, plain = defaultdict(int), defaultdict(int)
    for r in tasks.values():
        per[r["first"]] += 1
        if r["milestone"].strip() in ("M0", "M1"):
            plain[r["first"]] += 1
    out = []
    for n in g.MILESTONES:
        title, body = g.milestone_section(rfc_text, n)
        if title is None:
            continue
        name = g.label(n)
        out.append(f"### {name}: {title}")
        out.append("")
        if not per[n]:
            out.append(f"No task in the epics is marked {name} yet.")
        else:
            note = (f", {plain[n]} of them still marked plain {name[:2]}"
                    if plain[n] else "")
            out.append(f"{per[n]} tasks in the epics{note} "
                       f"(`implementation/roadmap.md`, {name}).")
        out.append("")
        out.extend(f"- [ ] {item}" for item in g.milestone_bullets(body))
        m = re.search(r"^- Exit:\s*(.*)$", body, re.M)
        if m:
            out.append("")
            out.append(f"**Exit:** {m[1].strip()}")
        out.append("")
    return "\n".join(out).rstrip()


def skeleton(rfc_text):
    parts = ["""# Implementation roadmap: M0a to M4

This file turns the six epic files into a milestone plan. Milestone goals and exit criteria are quoted from RFC-0001 §16, and SC IDs refer to `00-vision-and-scope.md` §7. Everything between the generated markers comes from `tools/roadmap.py`, which reads the task graph through `tools/depcheck.py` and the open questions from `plan/open-questions.md`. Re-run it after any change to an epic file, RFC-0001 §16 or the open-questions register:

```sh
python3 plan/implementation/tools/roadmap.py           # rewrite the generated sections
python3 plan/implementation/tools/roadmap.py --check   # exit 1 if this file is stale
```

**No durations.** The epics give no task sizes and no estimates. Every path length in this file is a count of tasks, and no date in it is a commitment or an estimate. A critical path of 20 tasks says that 20 tasks must finish one after another; how long that takes depends on task sizes nobody has estimated yet.

How to read the figures:
- **Critical path inside a milestone** is the longest chain of tasks in that milestone, each depending on the one before, assuming every earlier milestone is complete.
- **Longest chain from project start** is the longest dependency chain, across all milestones, that ends at a task in the milestone.
- **Zero-slack tasks** lie on some chain of the critical length. Delaying any of them delays the milestone.
- **Step** is a task's earliest position inside its milestone: 1 + the length of its longest chain of prerequisites in the same milestone. Tasks at the same step with no dependency between them can run in parallel.

## Start here

Begin with the workspace and CI skeleton, the merge gates and the core types every area compiles against. The list is generated: it is one valid order of M0, M0a first, that keeps unblocking the largest amount of later work. With several agents working in parallel, each takes the next task from the top whose dependencies are done, and the zero-slack tasks of M0a go first. Tasks marked **Maintainer action** are queued for the maintainers (ADR-0021).

""", g.marker("roadmap-start"), """

## Overview

### Tasks per milestone and area

""", g.marker("roadmap-counts"), """

### Critical paths

""", g.marker("roadmap-summary"), """

## Rules for every milestone

How milestones exit, how spikes gate the work they inform, and where performance targets live.

""", g.marker("roadmap-rules"), """

## Scope-cut order

If measured burn-up shows the plan cannot reach 1.0 as scoped, scope is cut in this order and no other. The order is re-checked at every milestone exit.

""", g.marker("roadmap-cuts"), """

## Recurring operations

Standing work that no milestone closes. It is tracked apart from milestone burn-up, so milestone progress is not overstated.

""", g.marker("roadmap-recurring"), "\n"]
    for n in g.MILESTONES:
        title, _ = g.milestone_section(rfc_text, n)
        key = marker_key(n)
        parts.append(f"\n## {g.label(n)}: {title or ''}\n\n### Goal and "
                     f"exit criteria\n\n{g.marker(f'{key}-goal')}\n\n"
                     f"### Plan\n\n{g.marker(key)}\n")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if roadmap.md or vision §8 is out of date")
    ap.add_argument("--init", action="store_true",
                    help="rewrite the whole of roadmap.md from the skeleton "
                         "(vision §8 is still filled in place)")
    ap.add_argument("--start", type=int, default=30,
                    help="length of the Start here list (default 30)")
    args = ap.parse_args()
    sections, vision, rfc_text = build(args.start)
    rc = g.write_sections(OUT, skeleton(rfc_text), sections,
                          init=args.init, check=args.check)
    # Vision §8 is filled in place; --init never rewrites the vision.
    rc |= g.write_sections(VISION, "", vision, check=args.check)
    return rc


if __name__ == "__main__":
    sys.exit(main())
