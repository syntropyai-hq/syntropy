#!/usr/bin/env python3
"""Split M0 into M0a and M0b, and M1 into M1a and M1b, in the epic files.

RFC-0001 §16 splits two milestones. M0a is the walking skeleton and M0b the
rest of M0. M1a is the first public preview (CUDA and CPU tiering on Linux)
and M1b adds Windows and Apple silicon. This script rewrites the Milestone
cell of every plain M0 and M1 task, and can cut dependency edges in the
same pass:

    M0a   the walking-skeleton seeds (SEEDS, any --seed, and every task
          whose cell already says M0a) plus their full dependency closure,
          computed from the graph after the cuts
    M1b   plain M1 tasks that match the Windows and Apple rule (M1B_EPICS,
          or M1B_RULE without KEEP_RULE, minus KEEP_M1A) and that no M1a
          task depends on; plus every task whose cell already says M1b or
          that M1B_LIST names, with the M1 tasks that depend on those
    M0b   every other plain M0 task, including tasks added later
    M1a   every other plain M1 task, including tasks added later

An M1a task may not depend on an M1b task, and many cross-platform M1
tasks depend on Windows, macOS or Metal tasks. --mode picks how the split
resolves such an edge:

    cut   (default) the edge is cut, so the dependent stays in M1a and the
          platform task moves to M1b. The script adds one M1b extension task
          per such dependent, in the dependent's epic with the epic's next
          free number: "Windows and macOS part of <ID>", depending on <ID>,
          the platform tasks whose edges were cut and the online tasks of
          the Windows and macOS rigs, with <ID>'s acceptance criteria to
          hold on those rigs and <ID>'s Implements cell
    pull  the rule-matched task stays in M1a; the script lists it with its
          M1a dependents, each a cross-platform task to split by hand
    push  the dependent moves to M1b with it A cell that already says M0a,
M0b, M1a or M1b is kept. The script stops if the split would contradict
one: an M0b task inside the M0a closure, or an M1a task that depends on an
M1b task.

Cuts are false edges (CUTS, plus any --cut; --no-cuts drops CUTS). A cut
`FROM:TO` removes TO from FROM's Depends cell; `FROM:TO=NEW` puts NEW in its
place. A cut whose edge no longer exists is reported and skipped.

Usage:

    python3 milestone_split.py --dry-run          # mapping, counts, and a
                                                  # depcheck of the result
    python3 milestone_split.py --dry-run --list   # also every task per group
    python3 milestone_split.py --dry-run --cut QUAL-E2-T02:QUAL-E12-T03
    python3 milestone_split.py --dry-run --mode pull
    python3 milestone_split.py                    # rewrite the epic files

--dry-run writes nothing in plan/: it checks a rewritten copy of the epics
in a temporary directory. Without it the epic files are rewritten in place
and depcheck runs on them. Record the cuts in ../dependency-fixes.md.
"""

import argparse
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # keep __pycache__ out of plan/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import depcheck  # noqa: E402
import epics  # noqa: E402
import gendoc  # noqa: E402

# The walking skeleton (RFC-0001 §16 M0a): one dense family end to end
# through the OpenAI streaming endpoint and the real runner, on CPU and CUDA,
# behind the merge gates, conformance and the parity harness.
SEEDS = (
    # End to end: endpoint, CLI, runner lifecycle, load and local models.
    "PROD-E2-T04", "PROD-E6-T02", "PROD-E6-T03", "PLC-E8-T01",
    "PLC-E9-T03", "TIER-E1-T08", "COMP-E11-T03",
    # The dense family and its certificate gate.
    "COMP-E16-T03", "COMP-E16-T02",
    # CPU and CUDA kernels for the dense family.
    "COMP-E3-T03", "COMP-E3-T08", "COMP-E5-T01", "COMP-E5-T05",
    "COMP-E5-T09", "COMP-E5-T12",
    # Gates: conformance, the parity harness, family conformance, the
    # model-definition oracle that gates FamilyCert, the hardware job on
    # R-ADA, and the M0 rigs.
    "QUAL-E4-T06", "QUAL-E5-T04", "QUAL-E5-T06", "QUAL-E6-T06",
    "COMP-E16-T14", "QUAL-E2-T02", "QUAL-E15-T17",
    # ADR-0021's merge gates, which run from the first commit, and the
    # sandbox for agent runs.
    "QUAL-E1-T03", "QUAL-E1-T04", "QUAL-E1-T05", "QUAL-E1-T06",
    "QUAL-E1-T09", "QUAL-E1-T16", "QUAL-E1-T18", "QUAL-E1-T19",
    "QUAL-E2-T11",
)

# False edges: the dependent's acceptance criteria stay testable without
# the prerequisite.
CUTS = (
    # The ingress pipeline's tests use only the fake runner (QUAL-E3-T18),
    # not FakeKernelProvider and the CPU kernels behind it.
    ("PROD-E2-T02", "QUAL-E3-T12", None),
)

# Windows and Apple work (RFC-0001 §16 M1b).
M1B_EPICS = {"COMP-E6"}  # the Metal provider
M1B_RULE = re.compile(
    r"\b(Windows|macOS|Metal|MSL|Apple|Macs?|MacBook|WDDM|IoRing|IOCP|MSI|"
    r"winget|notari[sz]\w*|LaunchDaemon|launchd|Homebrew|Authenticode|"
    r"Developer ID|WSL2?|DXGI|F_NOCACHE|MacNoCache|MacMapped|APFS|FileVault|"
    r"SME2|R-WIN|R-MACF|R-MBP|R-ULTRA(?:-[AB])?)\b")
# A task that also names Linux, CUDA or every OS is cross-platform: it stays
# in M1a unless it depends on M1b work.
KEEP_RULE = re.compile(
    r"\b(Linux|io_uring|cgroups?|MemAvailable|PSI|sysfs|hwmon|CUDA|NVML|"
    r"Tier 1 OS(?:es)?|every OS|each OS|per OS|all three|"
    r"(?:every|each|per) backend)\b")
# Fork-PR hardware runs start after M1a (RFC-0001 §16 M1b); applied only
# if the task's cell says M1.
M1B_LIST = ("QUAL-E2-T03", "QUAL-E2-T05")
# The Windows and macOS Tier 1 rigs an M1b extension task is checked on
# (ADR-0610).
M1B_RIGS = ("R-WIN", "R-MACF", "R-MBP", "R-ULTRA-A")
# Cross-platform tasks whose text names one OS in passing.
KEEP_M1A = (
    "TIER-E4-T02",   # pressure levels for every OS
    "PROD-E10-T08",  # install matrix across the Tier 1 OSes
    "PROD-E12-T06",  # update rollback on every OS
    "PROD-E12-T07",  # uninstall and package hooks on every OS
    "QUAL-E2-T07",   # lab reservation service
    "QUAL-E12-T18",  # spec records for the M1 rigs, R-BW among them
    "QUAL-E13-T03",  # M1 scenarios; (a) and (b2) are M1a's exit
    "QUAL-E13-T04",  # SC-24 install scenario per Tier 1 OS
    "QUAL-E15-T05",  # the M1 rigs, R-BW among them
    "COMP-E9-T08",   # M1 artefacts, the Linux core among them
    "QUAL-E11-T05",  # cpu-isa suite, x86 levels among them
    "TIER-E8-T08",   # cache levels end to end; Vram over RAM is M1a's
)


def parse_cut(text):
    m = re.fullmatch(rf"({depcheck.TASK}):({depcheck.TASK})(?:=({depcheck.TASK}))?",
                     text.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"not FROM:TO[=NEW]: {text}")
    return m[1], m[2], m[3]


def cut_cell(cell, target, new):
    """The Depends cell without the item naming `target`, with `new` in its
    place when given; None if no item names `target`. A range that covers
    `target` is written out without it."""
    segs, hit = [], False
    for seg in depcheck.split_outside(cell, ";"):
        head, _ = depcheck.strip_note(seg)
        if seg.startswith("TBD:") or seg.startswith("gate:") or \
                re.fullmatch(r"none", head, re.I):
            segs.append(seg)
            continue
        items = []
        for item in depcheck.split_outside(seg, ","):
            ihead, note = depcheck.strip_note(item)
            ids = depcheck.expand(ihead) or []
            if target in ids:
                hit = True
                rest = [i for i in ids if i != target]
                if rest:
                    items.append(", ".join(rest)
                                 + (f" ({note})" if note else ""))
                if new and new not in cell:
                    items.append(new)
                continue
            items.append(item)
        if items:
            segs.append(", ".join(items))
    if not hit:
        return None
    return "; ".join(segs) or "none"


def short_title(task, limit=80):
    """The head of a Task cell: the text before its first colon or
    semicolon outside backticks when that is short, else an abridgement."""
    text = re.sub(r"\*\*", "", task).strip()
    tick = False
    for i, ch in enumerate(text):
        if ch == "`":
            tick = not tick
        elif ch in ":;" and not tick:
            head = text[:i].strip()
            if 8 <= len(head) <= limit:
                return head
            break
    return gendoc.abridge(text, limit)


def extensions(records, deps, owed):
    """One M1b extension row per M1a task that lost an edge to M1b work.

    Returns a list of dicts with id, epic, base (the M1a task), task,
    milestone, depends, acceptance and implements."""
    tasks = {r["id"]: r for r in records}
    online = depcheck.rig_online(tasks)
    rig_tasks = []
    for rig in M1B_RIGS:
        t = online.get(rig)
        if t and t not in rig_tasks:
            rig_tasks.append(t)
    next_no = defaultdict(int)
    for r in records:
        next_no[r["epic"]] = max(next_no[r["epic"]], r["task_no"])
    rows = []
    for base in sorted(owed, key=epics.id_key):
        rec = tasks[base]
        need = [base] + sorted(owed[base], key=epics.id_key)
        reach, todo = set(), list(need)
        while todo:
            v = todo.pop()
            if v not in reach:
                reach.add(v)
                todo.extend(deps.get(v, ()))
        need += [t for t in rig_tasks if t not in reach]
        next_no[rec["epic"]] += 1
        tid = f"{rec['epic']}-T{next_no[rec['epic']]:02d}"
        rows.append({
            "id": tid, "epic": rec["epic"], "base": base,
            "task": f"Windows and macOS part of {base}: "
                    f"{short_title(rec['task'])}",
            "milestone": "M1b",
            "depends": ", ".join(need),
            "acceptance": f"{base}'s acceptance criteria hold on the Windows "
                          f"and macOS Tier 1 rigs ({M1B_RIGS[0]}; "
                          f"{', '.join(M1B_RIGS[1:-1])} and {M1B_RIGS[-1]})",
            "implements": rec["implements"],
        })
    return rows


def split(records, edges, seeds, cuts, mode="cut"):
    """Return (phase by task ID, report dict) for the plain M0 and M1 tasks."""
    tasks = {r["id"]: r for r in records}
    written = {t: r["milestone"].strip() for t, r in tasks.items()}
    report = {"cuts": [], "skipped": [], "errors": [], "notes": []}
    cut_pairs = set()
    for a, b, new in cuts:
        if (a, b) in edges:
            cut_pairs.add((a, b))
            report["cuts"].append((a, b, new))
        else:
            report["skipped"].append((a, b, new))
    deps = defaultdict(set)
    for a, b in edges:
        if (a, b) not in cut_pairs:
            deps[a].add(b)
    for a, b, new in report["cuts"]:
        if new:
            deps[a].add(new)
    rdeps = defaultdict(set)
    for a, bs in deps.items():
        for b in bs:
            rdeps[b].add(a)

    out = {}
    # M0a: the closure of the seeds.
    start = [s for s in seeds if s in tasks]
    report["notes"] += [f"seed {s} names no task" for s in seeds
                        if s not in tasks]
    start += [t for t, m in written.items() if m == "M0a"]
    m0a, todo = set(), list(start)
    while todo:
        t = todo.pop()
        if t in m0a:
            continue
        m0a.add(t)
        todo.extend(deps.get(t, ()))
    for t in sorted(m0a, key=epics.id_key):
        if written[t] not in ("M0", "M0a"):
            report["errors"].append(
                f"{t} ({written[t]}) is in the M0a closure")
        out[t] = "M0a"

    # M1b, fixed part: cells that say M1b and the list, with every M1 task
    # that depends on them.
    m1 = {t for t, m in written.items() if m in ("M1", "M1a", "M1b")}
    listed = {t for t, m in written.items() if m == "M1b"}
    for t in M1B_LIST:
        if t in tasks and written[t] in ("M1", "M1b"):
            listed.add(t)
        elif t in tasks:
            report["notes"].append(f"{t} is listed for M1b but its cell says "
                                   f"{written[t]}; left as it is")
    fixed, todo = set(), sorted(listed)
    while todo:
        t = todo.pop()
        if t in fixed:
            continue
        fixed.add(t)
        todo.extend(d for d in rdeps.get(t, ()) if d in m1)
    for t in sorted(fixed, key=epics.id_key):
        if written[t] == "M1a":
            report["errors"].append(
                f"{t} (M1a) depends on M1b work: "
                + ", ".join(sorted(deps[t] & fixed, key=epics.id_key)))
    # M1b, rule part: drop every candidate that M1a work depends on, until
    # none is left that does.
    rule = set()
    for t, r in tasks.items():
        if written[t] != "M1" or t in KEEP_M1A or t in fixed:
            continue
        if r["epic"] in M1B_EPICS or \
                (M1B_RULE.search(r["task"]) and not KEEP_RULE.search(r["task"])):
            rule.add(t)
    cand, kept, owed = set(rule), {}, defaultdict(set)
    if mode == "push":
        todo = sorted(cand)
        while todo:
            t = todo.pop()
            for d in rdeps.get(t, ()):
                if d in m1 and d not in cand and d not in fixed:
                    if written[d] == "M1a":
                        report["errors"].append(f"{d} (M1a) depends on M1b "
                                                f"work: {t}")
                        continue
                    cand.add(d)
                    todo.append(d)
    elif mode == "cut":
        for t in sorted(cand, key=epics.id_key):
            for d in rdeps.get(t, ()):
                if d in m1 and d not in cand and d not in fixed:
                    owed[d].add(t)
                    report["cuts"].append((d, t, None))
    else:
        changed = True
        while changed:
            changed = False
            for t in sorted(cand, key=epics.id_key):
                users = {d for d in rdeps.get(t, ())
                         if d in m1 and d not in cand and d not in fixed}
                if users:
                    cand.discard(t)
                    kept[t] = users
                    changed = True
    m1b = fixed | cand
    for t in m1b:
        out[t] = "M1b"
    report["m1b_rule"], report["m1b_list"] = cand, listed
    report["m1b_pulled"] = fixed - listed
    report["m1b_kept"] = kept
    report["owed"] = owed
    report["deps"] = deps

    for t in tasks:
        if t in out:
            continue
        out[t] = {"M0": "M0b", "M1": "M1a"}.get(written[t], written[t])
    report["changed"] = {t: p for t, p in out.items() if p != written[t]}
    return out, report


def rewrite(epics_dir, records, changed, cuts, added=()):
    """Rewrite Milestone and Depends cells in the epic files under
    `epics_dir`, and insert the `added` rows after the last row of their
    epic. Returns the number of rows rewritten or added."""
    by_file = defaultdict(dict)
    for r in records:
        by_file[r["file"]][r["line"]] = r
    last = {}
    for r in records:
        if r["epic"] not in last or r["line"] > last[r["epic"]]["line"]:
            last[r["epic"]] = r
    anchor = defaultdict(list)   # (file, line) -> rows to insert after it
    for row in added:
        a = last[row["epic"]]
        anchor[(a["file"], a["line"])].append(row)
    cut_by = defaultdict(list)
    for a, b, new in cuts:
        cut_by[a].append((b, new))
    rows = 0
    for rel, recs in by_file.items():
        path = Path(epics_dir) / Path(rel).name
        lines = path.read_text(encoding="utf-8").split("\n")
        header = None
        inserts = []                 # (index, rendered rows)
        for i, line in enumerate(lines):
            if not line.lstrip().startswith("|"):
                continue
            cells = epics.split_row(line)
            if cells and cells[0].lower() == "id":
                header = [epics.COLUMNS.get(c.lower()) for c in cells]
                continue
            rec = recs.get(i + 1)
            if rec is None or header is None:
                continue
            new = list(cells)
            tid = rec["id"]
            if tid in changed:
                new[header.index("milestone")] = changed[tid]
            for target, repl in cut_by.get(tid, ()):
                col = header.index("depends")
                cell = cut_cell(new[col], target, repl)
                if cell is not None:
                    new[col] = cell
            if new != cells:
                lines[i] = render(new)
                rows += 1
            extra = anchor.get((rel, i + 1))
            if extra:
                inserts.append((i + 1, [render([row.get(h, "") if h else ""
                                                for h in header])
                                        for row in extra]))
                rows += len(extra)
        for at, rendered in reversed(inserts):
            lines[at:at] = rendered
        path.write_text("\n".join(lines), encoding="utf-8")
    return rows


def render(cells):
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def counts(phases):
    per = defaultdict(int)
    for p in phases.values():
        per[p] += 1
    return ", ".join(f"{p} {per[p]}" for p in epics.PHASES if per[p])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the mapping and check a rewritten copy; "
                         "write nothing in plan/")
    ap.add_argument("--list", action="store_true",
                    help="list every task in M0a and M1b")
    ap.add_argument("--seed", action="append", default=[], metavar="ID",
                    help="another M0a seed (repeatable)")
    ap.add_argument("--cut", action="append", default=[], type=parse_cut,
                    metavar="FROM:TO[=NEW]", help="another edge to cut")
    ap.add_argument("--no-cuts", action="store_true",
                    help="apply only the --cut edges, not CUTS")
    ap.add_argument("--mode", choices=("cut", "pull", "push"),
                    default="cut",
                    help="how an M1a dependent of Windows or Apple work is "
                         "resolved (default cut)")
    ap.add_argument("--epics", default=str(epics.EPICS_DIR),
                    help="directory holding the epic files")
    args = ap.parse_args()

    before = depcheck.analyse(args.epics)
    fatal = [p for p in before["problems"] if p["kind"] in depcheck.FATAL]
    if fatal:
        sys.exit(f"depcheck.py reports {len(fatal)} problems in the epics "
                 "as they stand; fix them first")
    records = list(before["tasks"].values())
    edges = {(e["from"], e["to"]) for e in before["edges"]}
    cuts = ([] if args.no_cuts else list(CUTS)) + args.cut
    phases, rep = split(records, edges, SEEDS + tuple(args.seed), cuts,
                        args.mode)

    tasks = before["tasks"]
    print(f"{len(records)} tasks. Before: "
          + counts({t: epics.phase_name(r["first"]) for t, r in tasks.items()})
          + " (plain M0 counted as M0b, plain M1 as M1a).")
    print(f"After: {counts(phases)}.")
    print(f"Milestone cells rewritten: {len(rep['changed'])}.")
    m0a = sorted((t for t, p in phases.items() if p == "M0a"),
                 key=epics.id_key)
    per_area = defaultdict(int)
    for t in m0a:
        per_area[tasks[t]["area"]] += 1
    print(f"M0a: {len(m0a)} tasks ("
          + ", ".join(f"{a} {per_area[a]}" for a in epics.AREAS
                      if per_area[a]) + ").")
    print(f"M1b ({args.mode} mode): {len(rep['m1b_rule'])} by the rule"
          + (" and the tasks that depend on it" if args.mode == "push" else "")
          + f", {len(rep['m1b_list'])} by cell or list, "
          f"{len(rep['m1b_pulled'])} because they depend on those.")
    if args.mode == "pull":
        print(f"{len(rep['m1b_kept'])} rule-matched tasks stay in M1a because "
              "M1a work depends on them (split candidates; --list).")
    added = (extensions(records, rep["deps"], rep["owed"])
             if args.mode == "cut" else [])
    if args.mode == "cut":
        print(f"{len(rep['owed'])} M1a tasks lose an edge to M1b work; "
              f"{len(added)} M1b extension tasks are added (--list).")
    for a, b, new in rep["cuts"]:
        if a in rep["owed"]:
            continue
        print(f"cut: {a} no longer depends on {b}"
              + (f"; depends on {new} instead" if new else ""))
    for a, b, new in rep["skipped"]:
        print(f"cut skipped: {a} → {b} is not an edge")
    for n in rep["notes"]:
        print(f"note: {n}")
    if args.list:
        print("\nM0a:")
        for t in m0a:
            print(f"  {t}: {tasks[t]['task'][:100]}")
        for name, group in (("M1b by the rule", rep["m1b_rule"]),
                            ("M1b by cell or list", rep["m1b_list"]),
                            ("M1b because they depend on those",
                             rep["m1b_pulled"])):
            print(f"\n{name}:")
            for t in sorted(group, key=epics.id_key):
                print(f"  {t}: {tasks[t]['task'][:100]}")
        if args.mode == "cut":
            print("\nM1b extension tasks added:")
            for row in added:
                print(f"  {row['id']}: {row['task'][:110]}")
                print(f"      depends on {row['depends']}")
        print("\nRule-matched but kept in M1a, with the M1a tasks that "
              "depend on them:")
        for t in sorted(rep["m1b_kept"], key=epics.id_key):
            users = ", ".join(sorted(rep["m1b_kept"][t], key=epics.id_key))
            print(f"  {t} ← {users}: {tasks[t]['task'][:80]}")
    for e in rep["errors"]:
        print(f"ERROR: {e}")
    if rep["errors"]:
        return 1

    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            for p in Path(args.epics).glob("*.md"):
                shutil.copy(p, tmp)
            rewrite(tmp, records, rep["changed"], rep["cuts"], added)
            after = depcheck.analyse(tmp)
    else:
        rows = rewrite(args.epics, records, rep["changed"], rep["cuts"],
                       added)
        print(f"rewrote {rows} rows in {args.epics}")
        after = depcheck.analyse(args.epics)
    bad = [p for p in after["problems"] if p["kind"] in depcheck.FATAL]
    kinds = defaultdict(int)
    for p in bad:
        kinds[p["kind"]] += 1
    got = counts({t: r["phase"] for t, r in after["tasks"].items()})
    print(f"depcheck on the result: {len(after['tasks'])} tasks ({got}), "
          f"{len(after['edges'])} edges, "
          + (", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
             or "no problems") + ".")
    for p in bad[:40]:
        print(f"  {p['kind']}: {p['task']}: {p['detail']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
