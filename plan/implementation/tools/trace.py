#!/usr/bin/env python3
"""Write the generated sections of plan/implementation/traceability.md.

Three matrices, each built from the epic task rows:

    ADRs      every ADR in plan/decisions/: the RFC sections that cite it
              and the tasks whose Implements cell names it
    SC        every success criterion in 00-vision-and-scope.md §7: the
              tasks that deliver it and the tasks that verify it
    Design    every numbered subsection under a domain RFC's "Design"
              heading: the tasks whose Implements cell cites it

Usage:

    python3 trace.py            # rewrite the generated sections
    python3 trace.py --check    # exit 1 if the file is out of date
    python3 trace.py --gaps     # print the gaps only; write nothing
    python3 trace.py --init     # rewrite the whole file from the skeleton
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import epics  # noqa: E402
import gendoc as g  # noqa: E402

OUT = g.IMPL_DIR / "traceability.md"
RFC0001 = "RFC-0001"

TOKEN = re.compile(
    r"(?P<rfc>RFC-\d{4})"
    r"|(?P<adr>ADR-\d{4})(?:\s*(?:–|to)\s*(?:ADR-)?(?P<adr2>\d{4}))?"
    r"|§\s*(?P<sec>\d+)(?:\.\d+)*(?:\s*[–-]\s*§?\s*(?P<sec2>\d+)(?:\.\d+)*)?")


# ---------------------------------------------------------------- inputs

def load_adrs():
    out = []
    for path in sorted((g.PLAN_DIR / "decisions").glob("ADR-*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        aid = re.match(r"ADR-\d{4}", path.name)[0]
        title = re.sub(r"^#\s*", "", lines[0]) if lines else ""
        title = re.sub(rf"^{aid}:?\s*", "", title)
        status = ""
        for line in lines[:15]:
            m = re.match(r"^\s*-?\s*\**Status:?\**:?\s*(.*)$", line, re.I)
            if m:
                status = m[1].strip()
                break
        word = status.split()[0].rstrip(".,;") if status else "?"
        short = word
        m = re.match(r"Superseded by (ADR-\d{4})", status, re.I)
        if m:
            short = f"Superseded by {m[1]}"
        out.append({"id": aid, "title": title, "status": short,
                    "superseded": word.lower() == "superseded",
                    "file": f"../decisions/{path.name}"})
    return out


def rfc_sections(path):
    """Split an RFC into labelled sections.

    Returns (labels, design): labels maps label -> text; design lists
    (n, title) for every `### n.` heading under `## Design`. Numbered
    `## n.` sections (RFC-0001) are labelled `§n`; Design subsections `§n`;
    other `##` sections by their heading; text before the first `##` is
    `header`.
    """
    labels, design = defaultdict(list), []
    label, in_design, fence = "header", False, False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence:
            m = re.match(r"^## (?:(\d+)\.\s*)?(.+?)\s*$", line)
            if m:
                in_design = m[2].strip().lower() == "design" and not m[1]
                label = f"§{m[1]}" if m[1] else m[2].strip()
                if in_design:
                    label = "Design"
            elif in_design:
                m = re.match(r"^### (\d+)\.\s*(.+?)\s*$", line)
                if m:
                    label = f"§{m[1]}"
                    design.append((int(m[1]), m[2].strip()))
        labels[label].append(line)
    return {k: "\n".join(v) for k, v in labels.items()}, design


def load_rfcs():
    rfcs = {}
    for path in sorted((g.PLAN_DIR / "rfcs").glob("RFC-*.md")):
        rid = re.match(r"RFC-\d{4}", path.name)[0]
        labels, design = rfc_sections(path)
        rfcs[rid] = {"path": path, "labels": labels, "design": design,
                     "text": path.read_text(encoding="utf-8")}
    return rfcs


def implements_refs(cell, fallback_rfc, adr_ids):
    """Parse an Implements cell into (rfc_secs, adrs, bad).

    rfc_secs: set of (RFC-ID, top-level section number). A § reference
    belongs to the RFC or ADR named most recently before it in the cell;
    with none, to `fallback_rfc`. References that follow an ADR are skipped.
    adrs: ADR IDs named, with ranges expanded.
    """
    secs, adrs = set(), set()
    ctx = fallback_rfc
    for m in TOKEN.finditer(cell or ""):
        if m["rfc"]:
            ctx = m["rfc"]
        elif m["adr"]:
            ctx = m["adr"]
            adrs.add(m["adr"])
            if m["adr2"]:
                lo, hi = int(m["adr"][4:]), int(m["adr2"])
                adrs.update(a for a in adr_ids if lo <= int(a[4:]) <= hi)
        elif m["sec"] and ctx and ctx.startswith("RFC-"):
            lo = int(m["sec"])
            hi = int(m["sec2"]) if m["sec2"] else lo
            if hi < lo or hi - lo > 20:
                hi = lo
            for n in range(lo, hi + 1):
                secs.add((ctx, n))
    return secs, adrs


# ------------------------------------------------------------- analysis

def analyse():
    records, problems = epics.load()
    if problems:
        sys.exit("epics.py reports problems; fix them first "
                 "(python3 epics.py)")
    tasks = {r["id"]: r for r in records}
    info = g.epic_info()
    adrs = load_adrs()
    adr_ids = [a["id"] for a in adrs]
    rfcs = load_rfcs()
    scs = g.load_scs()
    gates = g.sc_gates(rfcs[RFC0001]["text"], set(scs))

    impl_secs = defaultdict(set)    # (rfc, n) -> tasks
    impl_adr = defaultdict(set)     # adr -> tasks
    dangling = []                   # (task, text)
    deliver = defaultdict(set)
    verify_qual = defaultdict(set)
    verify_ac = defaultdict(set)
    for r in records:
        fallback = (info.get(r["epic"], {}).get("rfcs") or [None])[0]
        secs, named = implements_refs(r["implements"], fallback, adr_ids)
        for rid, n in secs:
            impl_secs[(rid, n)].add(r["id"])
            if rid not in rfcs:
                dangling.append((r["id"], f"{rid} (no such RFC)"))
            elif rid != RFC0001 and \
                    n not in {d[0] for d in rfcs[rid]["design"]}:
                dangling.append((r["id"], f"{rid} §{n} (not a Design "
                                          "subsection)"))
        for a in named:
            if a in adr_ids:
                impl_adr[a].add(r["id"])
            else:
                dangling.append((r["id"], f"{a} (no such ADR)"))
        named_sc = (g.sc_refs(r["task"], scs)
                    | g.sc_refs(r["implements"], scs))
        ac_sc = g.sc_refs(r["acceptance"], scs)
        for n in named_sc | ac_sc:
            if n not in scs:
                dangling.append((r["id"], f"SC-{n} (not in vision §7)"))
        if r["area"] == "QUAL":
            for n in named_sc | ac_sc:
                verify_qual[n].add(r["id"])
        else:
            for n in named_sc:
                deliver[n].add(r["id"])
            for n in ac_sc:
                verify_ac[n].add(r["id"])

    # ADR citations in the RFCs, by section label.
    cites = defaultdict(lambda: defaultdict(list))  # adr -> rfc -> labels
    for rid, rfc in rfcs.items():
        for label, text in rfc["labels"].items():
            for a in sorted(set(re.findall(r"\bADR-\d{4}\b", text))):
                cites[a][rid].append(label)

    return {"tasks": tasks, "records": records, "adrs": adrs, "rfcs": rfcs,
            "scs": scs, "gates": gates, "impl_secs": impl_secs,
            "impl_adr": impl_adr, "dangling": sorted(set(dangling),
                                                     key=lambda d: (
                                                         epics.id_key(d[0]),
                                                         d[1])),
            "deliver": deliver, "verify_qual": verify_qual,
            "verify_ac": verify_ac, "cites": cites, "epics": info}


def label_key(label):
    m = re.match(r"§(\d+)", label)
    order = ["header", "Summary", "Motivation and goals / non-goals",
             "Design"]
    if label in order:
        return (0, order.index(label), "")
    if m:
        return (1, int(m[1]), "")
    return (2, 0, label)


def first_milestone(ids, tasks):
    return min((tasks[t]["first"] for t in ids), default=None)


# ------------------------------------------------------------- sections

def build(res):
    tasks = res["tasks"]
    gaps = []          # (kind, item, area, why)
    late = []          # (sc, gate, what, first milestone)

    # ADRs
    adr_rows = []
    for a in res["adrs"]:
        per_rfc = res["cites"].get(a["id"], {})
        parts = []
        for rid in sorted(per_rfc):
            labs = sorted(set(per_rfc[rid]), key=label_key)
            parts.append(f"{rid} " + ", ".join(labs))
        domain = [rid for rid in per_rfc if rid != RFC0001]
        impl = res["impl_adr"].get(a["id"], set())
        flags = []
        if not a["superseded"]:
            if not domain:
                flags.append("**GAP (RFC)**: no domain RFC cites it")
                gaps.append(("ADR (RFC)", a["id"], area_of_adr(a["id"]),
                             "no domain RFC section cites it"))
            if not impl:
                flags.append("**GAP (task)**: no Implements cell names it")
                gaps.append(("ADR (task)", a["id"], area_of_adr(a["id"]),
                             "no task's Implements cell names it"))
        flag = "; ".join(flags) if flags else (
            "superseded, exempt" if a["superseded"] else "ok")
        adr_rows.append([f"[{a['id']}]({a['file']})", g.cell(a["title"]),
                         a["status"], g.cell("; ".join(parts)) or "–",
                         g.counted(impl), flag])
    adr_md = g.table(["ADR", "Title", "Status", "RFC sections citing it",
                      "Tasks (Implements column)", "Flag"], adr_rows)

    # Success criteria
    sc_rows = []
    for n in sorted(res["scs"]):
        d = res["deliver"].get(n, set())
        vq = res["verify_qual"].get(n, set())
        va = res["verify_ac"].get(n, set())
        gate = res["gates"].get(n)
        flags = []
        if not d:
            flags.append("**GAP (delivery)**")
            gaps.append(("SC (delivery)", f"SC-{n}", area_of_sc(va, vq),
                         "no non-QUAL task names it in its Task or "
                         "Implements column"))
        if not (vq or va):
            flags.append("**GAP (verification)**")
            gaps.append(("SC (verification)", f"SC-{n}", "QUAL",
                         "no QUAL task and no acceptance criterion names it"))
        if gate is not None:
            fd = first_milestone(d, tasks)
            fv = first_milestone(vq | va, tasks)
            if fd is not None and fd > gate:
                late.append((n, gate, "delivery", fd))
                flags.append(f"late: first delivery in {g.label(fd)}")
            if fv is not None and fv > gate:
                late.append((n, gate, "verification", fv))
                flags.append(f"late: first verification in {g.label(fv)}")
        gate_txt = g.label(gate) if gate is not None else "–"
        if gate is not None and g.label(gate) == "M4":
            gate_txt = "M4 (every SC)"
        sc_rows.append([f"SC-{n}", g.cell(g.abridge(res["scs"][n])),
                        gate_txt, g.counted(d), g.counted(vq), g.counted(va),
                        "; ".join(flags) or "ok"])
    sc_md = g.table(["SC", "Target (abridged)", "First exit (RFC-0001 §16)",
                     "Delivered by", "Verified by QUAL tasks",
                     "Verified in acceptance criteria of", "Flag"], sc_rows)

    # Design subsections
    design_rows, design_count, single = [], 0, 0
    for rid in sorted(res["rfcs"]):
        if rid == RFC0001:
            continue
        for n, title in res["rfcs"][rid]["design"]:
            design_count += 1
            ids = res["impl_secs"].get((rid, n), set())
            if not ids:
                flag = "**GAP**"
                gaps.append(("Design", f"{rid} §{n}", area_of_rfc(rid),
                             f"no task cites \"{title}\""))
            else:
                flag = "ok"
                if len(ids) == 1:
                    single += 1
            design_rows.append([rid, f"§{n}", g.cell(title),
                                g.counted(ids), flag])
    design_md = g.table(["RFC", "§", "Design subsection",
                         "Tasks (Implements column)", "Flag"], design_rows)

    # Gaps
    if gaps:
        gaps_md = g.table(["Kind", "Item", "Area", "Why"],
                          [list(x) for x in gaps])
    else:
        gaps_md = "No GAPs."
    if late:
        gaps_md += ("\n\nSuccess criteria whose first delivering or "
                    "verifying task sits in a later milestone than the first "
                    "exit that names the SC:\n\n")
        gaps_md += g.table(
            ["SC", "First exit", "Missing by then", "Earliest task"],
            [[f"SC-{n}", g.label(gt), what, g.label(fm)]
             for n, gt, what, fm in late])
    if res["dangling"]:
        gaps_md += ("\n\nImplements, Task or acceptance cells that cite "
                    "something that does not exist:\n\n")
        gaps_md += g.table(["Task", "Reference"],
                           [[t, g.cell(x)] for t, x in res["dangling"]])

    # Summary
    n_adr = len(res["adrs"])
    n_sup = sum(1 for a in res["adrs"] if a["superseded"])
    n_dom = sum(1 for rid in res["rfcs"] if rid != RFC0001)
    kinds = defaultdict(int)
    for k, *_ in gaps:
        kinds[k] += 1
    sc_ids = sorted(res["scs"])
    summary = [
        f"Generated by `tools/trace.py` from {len(res['records'])} tasks in "
        f"{len(res['epics'])} epics, {n_adr} ADRs ({n_sup} superseded), "
        f"{len(sc_ids)} success criteria (SC-{sc_ids[0]} to "
        f"SC-{sc_ids[-1]}) and {design_count} Design subsections in "
        f"{n_dom} domain RFCs.",
        "",
        g.table(["Check", "GAPs"], [
            ["ADR with no domain RFC section citing it",
             kinds["ADR (RFC)"]],
            ["ADR with no task naming it in Implements", kinds["ADR (task)"]],
            ["SC with no delivering task", kinds["SC (delivery)"]],
            ["SC with no verifying task", kinds["SC (verification)"]],
            ["Design subsection with no task", kinds["Design"]],
        ]),
        "",
        f"Not GAPs, listed under Gaps for the owning area: {len(late)} SC "
        f"milestone mismatches and {len(res['dangling'])} references to "
        f"items that do not exist. {single} of the {design_count} Design "
        "subsections are cited by exactly one task.",
    ]
    return {"trace-summary": "\n".join(summary), "trace-gaps": gaps_md,
            "trace-adr": adr_md, "trace-sc": sc_md,
            "trace-design": design_md}, gaps, late, res["dangling"]


def area_of_adr(aid):
    n = int(aid[4:])
    if n <= 99:
        return "ARCH"
    return {1: "PLC", 2: "TIER", 3: "DIST", 4: "COMP", 5: "PROD",
            6: "QUAL"}.get(n // 100, "ARCH")


def area_of_rfc(rid):
    n = int(rid[4:])
    return {1: "PLC", 2: "TIER", 3: "DIST", 4: "COMP", 5: "PROD",
            6: "QUAL"}.get(n // 10, "ARCH")


def area_of_sc(*task_sets):
    areas = defaultdict(int)
    for s in task_sets:
        for t in s:
            area = t.split("-")[0]
            if area != "QUAL":
                areas[area] += 1
    if not areas:
        return "QUAL"
    return max(sorted(areas), key=lambda a: areas[a])


SKELETON = f"""# Traceability: ADRs, success criteria and RFC Design subsections to tasks

This file maps the decisions and targets in the plan to the tasks in `epics/`. Everything between the generated markers comes from `tools/trace.py`. Re-run it after any change to an epic file, an ADR, an RFC's Design headings or `00-vision-and-scope.md` §7:

```sh
python3 plan/implementation/tools/trace.py           # rewrite the generated sections
python3 plan/implementation/tools/trace.py --gaps    # list the gaps only
python3 plan/implementation/tools/trace.py --check   # exit 1 if this file is stale
```

How the matrices are built:
- **ADRs.** Every file in `plan/decisions/`, with its title from line 1 and its status from the header. "RFC sections citing it" lists each RFC whose text names the ADR ID, by section: `§n` for a numbered section (RFC-0001's top-level sections and every domain RFC's Design subsections), the heading for other sections, and `header` for the lines above the first section. Ranges in RFC text, such as "ADR-0400 to ADR-0404", are not expanded. "Tasks" are those whose Implements cell names the ADR; a range there, such as `ADR-0100–ADR-0107`, is expanded.
- **Success criteria.** Every SC row in `00-vision-and-scope.md` §7. A non-QUAL task that names the SC in its Task or Implements column *delivers* it. A QUAL task that names it in any column *verifies* it, and so does any task whose acceptance criteria name it. Ranges such as `SC-1 to SC-11` are expanded. "First exit" is the first milestone whose `Exit:` sentence in RFC-0001 §16 names the SC; M4's exit requires every SC.
- **Design subsections.** Every `### n.` heading under the `## Design` heading of a domain RFC (RFC-0010 to RFC-0062). A task implements subsection n when its Implements cell cites `RFC-NNNN §n` or any `§n.m` below it. A § reference belongs to the RFC or ADR named most recently before it in the same cell, and one that follows an ADR is skipped. A cell that names neither before its first § falls back to the first RFC on the epic's "RFC sections implemented" line.

**GAP** marks a non-superseded ADR that no domain RFC section cites or no task implements, an SC with no delivering or no verifying task, and a Design subsection that no task cites. Superseded ADRs are exempt. Two other findings are listed with the gaps but not counted as GAPs: an SC whose first delivering or verifying task comes after the first exit that names it, and a cell that cites an ADR, SC or RFC section that does not exist.

## Summary

{g.marker("trace-summary")}

## Gaps

{g.marker("trace-gaps")}

## ADRs

{g.marker("trace-adr")}

## Success criteria

{g.marker("trace-sc")}

## RFC Design subsections

{g.marker("trace-design")}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if traceability.md is out of date")
    ap.add_argument("--gaps", action="store_true",
                    help="print the gaps and write nothing")
    ap.add_argument("--init", action="store_true",
                    help="rewrite the whole file from the skeleton")
    args = ap.parse_args()
    res = analyse()
    sections, gaps, late, dangling = build(res)
    if args.gaps:
        for k, item, area, why in gaps:
            print(f"GAP {k}: {item} ({area}): {why}")
        for n, gt, what, fm in late:
            print(f"LATE SC-{n}: first exit {g.label(gt)}, first {what} task "
                  f"in {g.label(fm)}")
        for t, x in dangling:
            print(f"DANGLING {t}: {x}")
        print(f"GAPs: {len(gaps)}  late: {len(late)}  "
              f"dangling: {len(dangling)}")
        return 0
    return g.write_sections(OUT, SKELETON, sections, init=args.init,
                            check=args.check)


if __name__ == "__main__":
    sys.exit(main())
