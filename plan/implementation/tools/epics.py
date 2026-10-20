#!/usr/bin/env python3
"""Parse the task tables in plan/implementation/epics/*.md into records.

Each task row is a Markdown table row whose first cell is a task ID such as
PLC-E4-T17. The table header names the columns; the parser maps cells by
header name, so a table may reorder its columns.

Run directly to print a per-file summary and every malformed row or
duplicate ID. Exit status is 1 when any are found.

    python3 epics.py            # summary and problems
    python3 epics.py --json     # every record as JSON on stdout

Other scripts import `load()`.
"""

import argparse
import json
import re
import sys
from pathlib import Path

AREAS = ("PLC", "TIER", "DIST", "COMP", "PROD", "QUAL")
EPICS_DIR = Path(__file__).resolve().parent.parent / "epics"

ID_RE = re.compile(r"^(?P<area>[A-Z]+)-E(?P<epic>\d+)-T(?P<task>\d+)$")
# Milestones in delivery order (RFC-0001 §16). M0 is split into the walking
# skeleton M0a and the rest of M0, M0b; M1 into M1a, the Linux CUDA and CPU
# preview, and M1b, which adds Windows and Apple silicon.
PHASES = ("M0a", "M0b", "M1a", "M1b", "M2", "M3", "M4")
# A cell that says plain M0 or M1 is read as the phase milestone_split.py
# gives such a task by default.
PLAIN = {"M0": "M0b", "M1": "M1a", "M2": "M2", "M3": "M3", "M4": "M4"}
MILESTONE = r"M(?:[01][ab]|[0-4])"
MILESTONE_RE = re.compile(rf"\b({MILESTONE})\b")
# Header names as written in the epic files, mapped to record fields.
COLUMNS = {
    "id": "id",
    "task": "task",
    "milestone": "milestone",
    "depends on": "depends",
    "acceptance criteria (testable)": "acceptance",
    "acceptance criteria": "acceptance",
    "implements": "implements",
}
REQUIRED = ("id", "task", "milestone", "depends", "acceptance", "implements")


def split_row(line):
    """Split a Markdown table row on unescaped pipes; return stripped cells."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells, cur, i = [], [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def phase_index(name):
    """Position in PHASES of a milestone name; plain M0 and M1 as in PLAIN."""
    return PHASES.index(PLAIN.get(name, name))


def phase_name(index):
    return PHASES[index]


def major(index):
    """The milestone number (0-4) of a position in PHASES."""
    return int(PHASES[index][1])


def milestones_of(text):
    """Every milestone named in a Milestone cell, as sorted PHASES indices."""
    found = sorted({phase_index(m) for m in MILESTONE_RE.findall(text)})
    # A range such as "M0–M2" also covers the milestones between its ends.
    if re.search(rf"{MILESTONE}\s*[–-]\s*{MILESTONE}", text) and found:
        found = list(range(found[0], found[-1] + 1))
    return found


def id_key(task_id):
    m = ID_RE.match(task_id)
    if not m:
        return (task_id, 0, 0)
    return (AREAS.index(m["area"]) if m["area"] in AREAS else 99,
            int(m["epic"]), int(m["task"]))


def load(epics_dir=EPICS_DIR):
    """Return (records, problems).

    records: list of dicts with id, area, epic, task_no, milestone (as
    written), milestones (sorted PHASES indices), first and last (PHASES
    indices), phase (the name of the first, such as "M0b"), task, depends,
    acceptance, implements, file, line.
    problems: list of dicts with kind (MALFORMED, DUPLICATE, NOFILE), file,
    line, id and detail.
    """
    records, problems, seen = [], [], {}
    for area in AREAS:
        path = Path(epics_dir) / f"{area}.md"
        rel = f"epics/{area}.md"
        if not path.exists():
            problems.append({"kind": "NOFILE", "file": rel, "line": 0,
                             "id": "", "detail": "epic file missing"})
            continue
        header = None
        for lineno, line in enumerate(path.read_text(encoding="utf-8")
                                      .splitlines(), 1):
            if not line.lstrip().startswith("|"):
                header = None if not line.strip() else header
                continue
            cells = split_row(line)
            low = [c.lower() for c in cells]
            if low and low[0] == "id":
                header = [COLUMNS.get(c) for c in low]
                continue
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
                continue
            first = cells[0] if cells else ""
            if not re.match(r"^[A-Z]+-E\d+-T\d+", first):
                continue
            if header is None or "depends" not in header:
                problems.append({"kind": "MALFORMED", "file": rel,
                                 "line": lineno, "id": first,
                                 "detail": "task row outside a task table"})
                continue
            if len(cells) != len(header):
                problems.append({"kind": "MALFORMED", "file": rel,
                                 "line": lineno, "id": first,
                                 "detail": f"{len(cells)} cells, header has "
                                           f"{len(header)} (unescaped pipe?)"})
                continue
            rec = {f: cells[i] for i, f in enumerate(header) if f}
            m = ID_RE.match(rec["id"])
            if not m:
                problems.append({"kind": "MALFORMED", "file": rel,
                                 "line": lineno, "id": first,
                                 "detail": "ID cell is not a bare task ID"})
                continue
            if m["area"] != area:
                problems.append({"kind": "MALFORMED", "file": rel,
                                 "line": lineno, "id": rec["id"],
                                 "detail": f"area {m['area']} in {area}.md"})
            missing = [f for f in REQUIRED if f not in rec]
            if missing:
                problems.append({"kind": "MALFORMED", "file": rel,
                                 "line": lineno, "id": rec["id"],
                                 "detail": "missing columns: "
                                           + ", ".join(missing)})
                continue
            ms = milestones_of(rec["milestone"])
            if not ms:
                problems.append({"kind": "MALFORMED", "file": rel,
                                 "line": lineno, "id": rec["id"],
                                 "detail": "no milestone (M0a–M4) in "
                                           f"'{rec['milestone']}'"})
                continue
            if rec["id"] in seen:
                prev = seen[rec["id"]]
                problems.append({"kind": "DUPLICATE", "file": rel,
                                 "line": lineno, "id": rec["id"],
                                 "detail": f"also at {prev['file']}:"
                                           f"{prev['line']}"})
                continue
            rec.update({
                "area": m["area"],
                "epic": f"{m['area']}-E{m['epic']}",
                "task_no": int(m["task"]),
                "milestones": ms,
                "first": ms[0],
                "last": ms[-1],
                "phase": PHASES[ms[0]],
                "file": rel,
                "line": lineno,
            })
            seen[rec["id"]] = rec
            records.append(rec)
    return records, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true",
                    help="print every record as JSON")
    ap.add_argument("--epics", default=str(EPICS_DIR),
                    help="directory holding the epic files")
    args = ap.parse_args()
    records, problems = load(args.epics)
    if args.json:
        json.dump(records, sys.stdout, indent=1, ensure_ascii=False)
        print()
        return 1 if problems else 0
    by_file = {}
    for r in records:
        by_file.setdefault(r["file"], []).append(r)
    for area in AREAS:
        rel = f"epics/{area}.md"
        rs = by_file.get(rel, [])
        per_m = {}
        for r in rs:
            per_m[r["first"]] = per_m.get(r["first"], 0) + 1
        dist = ", ".join(f"{PHASES[k]}: {per_m[k]}" for k in sorted(per_m))
        epics = len({r["epic"] for r in rs})
        print(f"{rel}: {len(rs)} tasks in {epics} epics ({dist})")
    print(f"total: {len(records)} tasks")
    for p in problems:
        print(f"{p['kind']}: {p['file']}:{p['line']} {p['id']}: {p['detail']}")
    print(f"problems: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
