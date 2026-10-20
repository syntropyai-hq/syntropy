#!/usr/bin/env python3
"""Check the task dependency graph in plan/implementation/epics/*.md.

A "Depends on" cell is split on each ';' outside parentheses and backticks.
Each segment must be one of:

    none                          no dependency
    TBD: <what is needed>         a dependency no task delivers yet
    gate: <OQ-ID> <text>          an external gate (an open question in
                                  plan/open-questions.md), no task edge
    ID[, ID ...]                  task IDs or ID ranges, each optionally
                                  followed by one parenthetical note

An ID is AREA-En-Tnn. A range is AREA-En-Tnn–Tmm (en dash or hyphen, or the
full second ID) and expands to every task between the two ends. A note that
contains `gate: <OQ-ID>` marks that edge as an accepted exception to the
milestone rules. IDs inside notes are not edges; they are reported as notes.

Problems (any of them makes the exit status 1):

    WORDFORM   a segment or item that is not in the grammar above
    DANGLING   a task ID that names no task
    CYCLE      a strongly connected component of two or more tasks, or a
               task that depends on itself
    LATER      a task depends on a task whose first milestone is later than
               its own first milestone, in the order M0a < M0b < M1a < M1b <
               M2 < M3 < M4 (plain M0 reads as M0b and plain M1 as M1a); a
               task whose milestone cell names several milestones may depend
               on one that starts by its last milestone (reported as SPLIT,
               not a problem)
    M0CLOSURE  an M0 task reaches a non-M0 task, or an M0a task reaches a
               task outside M0a, through edges that are neither SPLIT nor
               gated
    MULTI      a Milestone cell that is not exactly one milestone (`M0a`,
               `M0b`, `M1a`, `M1b`, or `M0`–`M4`); every task sits in one
               milestone, so a task that spans two is split into a core task
               and an extension task
    MALFORMED, DUPLICATE   rows epics.py could not read

Notes (listed with --only, never a problem):

    AC-LATER   an acceptance criterion that applies "from" a later milestone
               than the task's own; that criterion belongs in its own task
    SPIKE-LEAF a spike task with no dependents, so its answer gates nothing
    SPIKE-NOBOX a spike task whose row states no timebox
    RIG-PATH   a task whose acceptance criteria name a lab rig but which has
               no dependency path to the task that brings the rig online
    RIG-LATER  a task that names a rig brought online in a later milestone

A rig's online task is the first QUAL-E15 task, in milestone order, whose
Task cell names it.

Usage:

    python3 depcheck.py                       # summary and every problem
    python3 depcheck.py --graph OUT.json      # also write the graph
    python3 depcheck.py --only LATER,CYCLE    # list only these kinds
    python3 depcheck.py --task PLC-E4-T17     # one task's edges both ways

The graph file must be written outside plan/.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # keep __pycache__ out of plan/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import epics  # noqa: E402

PLAN_DIR = Path(__file__).resolve().parent.parent.parent
TASK = r"[A-Z]+-E\d+-T\d+"
ID_ONLY = re.compile(rf"^({TASK})$")
RANGE = re.compile(rf"^(?P<a>[A-Z]+-E\d+-T(?P<lo>\d+))\s*[–-]\s*"
                   rf"(?:(?P<b>[A-Z]+-E\d+)-)?T(?P<hi>\d+)$")
GATE = re.compile(r"\bgate:\s*(OQ-[A-Z]+-\d+|OD-\d+)")
SINGLE = re.compile(rf"^{epics.MILESTONE}$")
FROM_LATER = re.compile(rf"\b[Ff]rom ({epics.MILESTONE})\b")
SPIKE = re.compile(r"^(?:\*\*)?\s*Spikes?\b")
TIMEBOX = re.compile(r"\btime-?box", re.I)
RIG = re.compile(r"\bR-[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?\b")
RIG_EPIC = "QUAL-E15"
FATAL = ("MALFORMED", "DUPLICATE", "MULTI", "WORDFORM", "DANGLING", "CYCLE",
         "LATER", "M0CLOSURE")


def split_outside(text, sep):
    """Split on `sep` where it is outside parentheses and backticks."""
    parts, cur, depth, tick = [], [], 0, False
    for ch in text:
        if ch == "`":
            tick = not tick
        elif not tick and ch == "(":
            depth += 1
        elif not tick and ch == ")" and depth:
            depth -= 1
        if ch == sep and depth == 0 and not tick:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def strip_note(item):
    """Split 'X (note)' into ('X', 'note'); the note must close the item."""
    item = item.strip()
    if not item.endswith(")"):
        return item, None
    depth = 0
    for i in range(len(item) - 1, -1, -1):
        if item[i] == ")":
            depth += 1
        elif item[i] == "(":
            depth -= 1
            if depth == 0:
                head = item[:i].strip()
                return head, item[i + 1:-1].strip()
    return item, None


def expand(head):
    """Task IDs named by an item head, or None if it is not an ID or range."""
    m = ID_ONLY.match(head)
    if m:
        return [head]
    m = RANGE.match(head)
    if m:
        prefix = m["a"].rsplit("-T", 1)[0]
        if m["b"] and m["b"] != prefix:
            return None
        lo, hi = int(m["lo"]), int(m["hi"])
        if hi < lo:
            return None
        width = len(m["lo"])
        return [f"{prefix}-T{n:0{width}d}" for n in range(lo, hi + 1)]
    return None


def parse_cell(cell):
    """Return (edges, tbds, gates, problems, notes) for one Depends cell.

    edges: list of (target_id, gate_or_None, item_text)
    problems: list of (kind, text)
    """
    edges, tbds, gates, problems, notes = [], [], [], [], []
    text = cell.strip()
    if not text:
        problems.append(("WORDFORM", "(empty cell)"))
        return edges, tbds, gates, problems, notes
    for seg in split_outside(text, ";"):
        head, note = strip_note(seg)
        if re.fullmatch(r"none", head, re.I):
            continue
        if seg.startswith("TBD:"):
            tbds.append(seg[4:].strip())
            continue
        g = re.match(r"^gate:\s*(OQ-[A-Z]+-\d+|OD-\d+)\b", seg)
        if g:
            gates.append((g[1], seg))
            continue
        for item in split_outside(seg, ","):
            head, note = strip_note(item)
            ids = expand(head)
            if ids is None:
                problems.append(("WORDFORM", item))
                continue
            gate = None
            if note:
                gm = GATE.search(note)
                gate = gm[1] if gm else None
                for ref in re.findall(TASK, note):
                    notes.append(("NOTE-ID", f"{ref} inside note of '{item}'"))
            for t in ids:
                edges.append((t, gate, item))
    return edges, tbds, gates, problems, notes


def tarjan(nodes, succ):
    """Strongly connected components, iteratively."""
    index, low, on, stack, out, counter = {}, {}, set(), [], [], [0]
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(succ.get(root, ())))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            pushed = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(succ.get(w, ()))))
                    pushed = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if pushed:
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
    return out


def analyse(epics_dir=epics.EPICS_DIR):
    records, load_problems = epics.load(epics_dir)
    tasks = {r["id"]: r for r in records}
    problems = []  # dicts: kind, task, file, line, detail
    notes = []
    edges = []     # dicts: from, to, gate, item, kind (ok/split/gated)
    tbds, gates = [], []

    def add(kind, rec, detail):
        problems.append({"kind": kind, "task": rec["id"] if rec else "",
                         "file": rec["file"] if rec else "",
                         "line": rec["line"] if rec else 0,
                         "detail": detail})

    for p in load_problems:
        problems.append({"kind": p["kind"], "task": p["id"],
                         "file": p["file"], "line": p["line"],
                         "detail": p["detail"]})

    for rec in records:
        if not SINGLE.match(rec["milestone"].strip()):
            add("MULTI", rec, f"milestone cell '{rec['milestone']}'")
        es, tb, gs, probs, nts = parse_cell(rec["depends"])
        for kind, text in probs:
            add(kind, rec, text)
        for kind, text in nts:
            notes.append({"kind": kind, "task": rec["id"],
                          "file": rec["file"], "line": rec["line"],
                          "detail": text})
        for text in tb:
            tbds.append({"task": rec["id"], "milestone": rec["milestone"],
                         "file": rec["file"], "line": rec["line"],
                         "text": text})
        for oq, text in gs:
            gates.append({"task": rec["id"], "gate": oq, "text": text,
                          "file": rec["file"], "line": rec["line"]})
        seen = set()
        for target, gate, item in es:
            if target not in tasks:
                add("DANGLING", rec, f"{target} (in '{item}')")
                continue
            if target in seen:
                continue
            seen.add(target)
            edges.append({"from": rec["id"], "to": target, "gate": gate,
                          "item": item})

    succ = defaultdict(list)
    for e in edges:
        succ[e["from"]].append(e["to"])

    cycles = []
    for comp in tarjan(sorted(tasks, key=epics.id_key), succ):
        if len(comp) > 1 or comp[0] in succ.get(comp[0], ()):
            comp = sorted(comp, key=epics.id_key)
            cycles.append(comp)
            first = tasks[comp[0]]
            add("CYCLE", first, f"{len(comp)} tasks: " + ", ".join(comp))

    # Milestone checks.
    for e in edges:
        a, b = tasks[e["from"]], tasks[e["to"]]
        e["status"] = "ok"
        if b["first"] <= a["first"]:
            continue
        if e["gate"]:
            e["status"] = "gated"
            continue
        if b["first"] <= a["last"]:
            e["status"] = "split"
            notes.append({"kind": "SPLIT", "task": a["id"], "file": a["file"],
                          "line": a["line"],
                          "detail": f"{a['id']} ({a['milestone']}) → "
                                    f"{b['id']} ({b['milestone']})"})
            continue
        e["status"] = "later"
        add("LATER", a, f"{a['id']} ({a['milestone']}) → {b['id']} "
                        f"({b['milestone']})")

    strict = defaultdict(list)
    for e in edges:
        if e["status"] in ("ok", "later"):
            strict[e["from"]].append(e["to"])
    m0a = epics.phase_index("M0a")
    for tid in sorted(tasks, key=epics.id_key):
        rec = tasks[tid]
        if epics.major(rec["first"]) != 0:
            continue
        # An M0a task may reach only M0a tasks; any other M0 task only M0.
        if rec["first"] == m0a:
            def inside(r):
                return r["first"] == m0a
        else:
            def inside(r):
                return epics.major(r["first"]) == 0
        parent, todo, bad = {tid: None}, [tid], []
        while todo:
            v = todo.pop()
            for w in strict.get(v, ()):
                if w in parent:
                    continue
                parent[w] = v
                if not inside(tasks[w]):
                    bad.append(w)
                    continue
                todo.append(w)
        for w in sorted(bad, key=epics.id_key):
            path, x = [], w
            while x is not None:
                path.append(x)
                x = parent[x]
            add("M0CLOSURE", rec,
                " → ".join(reversed(path)) + f" ({tasks[w]['milestone']})")

    plan_notes(tasks, edges, notes)

    for t in tbds:
        if epics.major(tasks[t["task"]]["first"]) == 0:
            notes.append({"kind": "M0-TBD", "task": t["task"],
                          "file": t["file"], "line": t["line"],
                          "detail": t["text"]})

    return {"tasks": tasks, "edges": edges, "problems": problems,
            "notes": notes, "tbds": tbds, "gates": gates, "cycles": cycles}


def rig_online(tasks):
    """Each rig named in a QUAL-E15 Task cell, mapped to the first such task
    in milestone order: the task that brings the rig online."""
    out = {}
    lab = sorted((r for r in tasks.values() if r["epic"] == RIG_EPIC),
                 key=lambda r: (r["first"], epics.id_key(r["id"])))
    for r in lab:
        for rig in RIG.findall(r["task"]):
            out.setdefault(rig, r["id"])
    return out


def plan_notes(tasks, edges, notes):
    """The AC-LATER, SPIKE-* and RIG-* notes."""
    def note(kind, rec, detail):
        notes.append({"kind": kind, "task": rec["id"], "file": rec["file"],
                      "line": rec["line"], "detail": detail})

    succ, has_dependent = defaultdict(list), set()
    for e in edges:
        succ[e["from"]].append(e["to"])
        has_dependent.add(e["to"])
    online = rig_online(tasks)
    for tid in sorted(tasks, key=epics.id_key):
        rec = tasks[tid]
        for m in FROM_LATER.findall(rec["acceptance"]):
            if epics.phase_index(m) > rec["last"]:
                note("AC-LATER", rec, f"'from {m}' in a {rec['milestone']} "
                                      "row")
        if SPIKE.match(rec["task"]):
            if tid not in has_dependent:
                note("SPIKE-LEAF", rec, "no task depends on this spike")
            if not TIMEBOX.search(rec["task"] + " " + rec["acceptance"]):
                note("SPIKE-NOBOX", rec, "no timebox stated")
        if rec["epic"] == RIG_EPIC:
            continue
        rigs = []
        for rig in RIG.findall(rec["acceptance"]):
            if rig not in online:
                # "R-ULTRA" stands for the first rig named R-ULTRA-….
                rig = next((k for k in online if k.startswith(rig + "-")),
                           rig)
            if rig in online and rig not in rigs:
                rigs.append(rig)
        if not rigs:
            continue
        reach, todo = set(), list(succ.get(tid, ()))
        while todo:
            v = todo.pop()
            if v not in reach:
                reach.add(v)
                todo.extend(succ.get(v, ()))
        for rig in rigs:
            src = tasks[online[rig]]
            if src["first"] > rec["last"]:
                note("RIG-LATER", rec, f"{rig} comes online in "
                                       f"{online[rig]} ({src['milestone']})")
            elif online[rig] not in reach:
                note("RIG-PATH", rec, f"{rig}: no path to {online[rig]}")


def write_graph(result, out):
    out = Path(out).resolve()
    try:
        out.relative_to(PLAN_DIR.resolve())
        sys.exit(f"refusing to write {out}: the graph must live outside "
                 f"{PLAN_DIR}")
    except ValueError:
        pass
    tasks = result["tasks"]
    doc = {
        "nodes": [{k: r[k] for k in ("id", "area", "epic", "milestone",
                                     "milestones", "first", "last", "phase",
                                     "task", "file", "line")}
                  for r in sorted(tasks.values(),
                                  key=lambda r: epics.id_key(r["id"]))],
        "edges": [{k: e[k] for k in ("from", "to", "status", "gate")}
                  for e in result["edges"]],
        "tbd": result["tbds"],
        "gates": result["gates"],
        "cycles": result["cycles"],
        "problems": result["problems"],
        "notes": result["notes"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Check the epic task dependency graph.")
    ap.add_argument("--graph", metavar="PATH",
                    help="write the graph as JSON here (outside plan/)")
    ap.add_argument("--only", metavar="KINDS",
                    help="comma-separated kinds to list, e.g. LATER,CYCLE; "
                         "also SPLIT, NOTE-ID, M0-TBD, TBD, GATE, AC-LATER, "
                         "SPIKE-LEAF, SPIKE-NOBOX, RIG-PATH, RIG-LATER")
    ap.add_argument("--task", metavar="ID",
                    help="show one task's edges in both directions")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the summary")
    ap.add_argument("--epics", default=str(epics.EPICS_DIR),
                    help="directory holding the epic files")
    args = ap.parse_args()
    res = analyse(args.epics)
    tasks = res["tasks"]

    if args.task:
        t = tasks.get(args.task)
        if not t:
            sys.exit(f"no task {args.task}")
        print(f"{t['id']} ({t['milestone']}) {t['file']}:{t['line']}")
        print(f"  depends: {t['depends']}")
        for e in res["edges"]:
            if e["from"] == t["id"]:
                d = tasks[e["to"]]
                print(f"  → {d['id']} ({d['milestone']}) [{e['status']}]")
        for e in res["edges"]:
            if e["to"] == t["id"]:
                d = tasks[e["from"]]
                print(f"  ← {d['id']} ({d['milestone']}) [{e['status']}]")
        return 0

    only = set(args.only.split(",")) if args.only else None
    if not args.quiet:
        for p in sorted(res["problems"],
                        key=lambda p: (FATAL.index(p["kind"]),
                                       epics.id_key(p["task"]))):
            if only is None or p["kind"] in only:
                print(f"{p['kind']}: {p['file']}:{p['line']} {p['task']}: "
                      f"{p['detail']}")
        if only:
            for n in res["notes"]:
                if n["kind"] in only:
                    print(f"{n['kind']}: {n['file']}:{n['line']} "
                          f"{n['task']}: {n['detail']}")
            if "TBD" in only:
                for t in res["tbds"]:
                    print(f"TBD: {t['file']}:{t['line']} {t['task']} "
                          f"({t['milestone']}): {t['text']}")
            if "GATE" in only:
                for g in res["gates"]:
                    print(f"GATE: {g['file']}:{g['line']} {g['task']}: "
                          f"{g['text']}")
                for e in res["edges"]:
                    if e["gate"]:
                        print(f"GATE: {e['from']} → {e['to']} ({e['gate']})")

    count = defaultdict(int)
    for p in res["problems"]:
        count[p["kind"]] += 1
    ncount = defaultdict(int)
    for n in res["notes"]:
        ncount[n["kind"]] += 1
    m0_tasks = len({p["task"] for p in res["problems"]
                    if p["kind"] == "M0CLOSURE"})
    gated = sum(1 for e in res["edges"] if e["gate"])
    print(f"tasks: {len(tasks)}  edges: {len(res['edges'])}")
    print(f"WORDFORM: {count['WORDFORM']}  DANGLING: {count['DANGLING']}  "
          f"CYCLE: {count['CYCLE']} "
          f"({sum(len(c) for c in res['cycles'])} tasks)  "
          f"LATER: {count['LATER']}  "
          f"M0CLOSURE: {count['M0CLOSURE']} ({m0_tasks} M0 tasks)")
    print(f"MALFORMED: {count['MALFORMED']}  DUPLICATE: {count['DUPLICATE']}  "
          f"MULTI: {count['MULTI']}")
    print(f"not problems: TBD: {len(res['tbds'])} "
          f"(M0: {ncount['M0-TBD']})  SPLIT: {ncount['SPLIT']}  "
          f"gated edges: {gated}  gate segments: {len(res['gates'])}  "
          f"NOTE-ID: {ncount['NOTE-ID']}")
    print(f"plan notes: AC-LATER: {ncount['AC-LATER']}  "
          f"SPIKE-LEAF: {ncount['SPIKE-LEAF']}  "
          f"SPIKE-NOBOX: {ncount['SPIKE-NOBOX']}  "
          f"RIG-PATH: {ncount['RIG-PATH']}  RIG-LATER: {ncount['RIG-LATER']}")
    if args.graph:
        print(f"graph: {write_graph(res, args.graph)}")
    return 1 if any(count[k] for k in FATAL) else 0


if __name__ == "__main__":
    sys.exit(main())
