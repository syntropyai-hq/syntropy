"""Helpers shared by the document generators (trace.py, roadmap.py,
deps_report.py).

A generated section sits between two markers:

    <!-- BEGIN GENERATED: name -->
    ...
    <!-- END GENERATED -->

`write_sections()` replaces the text between each named pair and leaves
everything outside the markers untouched. A file that does not exist yet,
or a run with `init=True`, is written from the generator's skeleton first.
"""

import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep __pycache__ out of plan/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import epics  # noqa: E402

TOOLS_DIR = Path(__file__).resolve().parent
IMPL_DIR = TOOLS_DIR.parent
PLAN_DIR = IMPL_DIR.parent
AREAS = epics.AREAS
# Milestones as positions in epics.PHASES (M0a, M0b, M1a, M1b, M2, M3, M4).
MILESTONES = tuple(range(len(epics.PHASES)))


def label(n):
    """The milestone name of a position in epics.PHASES, such as "M1a"."""
    return epics.PHASES[n]

SC_RE = re.compile(r"\bSC-(\d+)(?:\s*(?:–|to)\s*SC-(\d+)|\s*–\s*(\d+))?")

BEGIN = "<!-- BEGIN GENERATED: {} -->"
END = "<!-- END GENERATED -->"
BLOCK_RE = re.compile(r"(<!-- BEGIN GENERATED: (?P<name>[\w.-]+) -->\n)"
                      r"(?P<body>.*?)"
                      r"(\n?<!-- END GENERATED(?:: [\w.-]+)? -->)", re.S)


def marker(name):
    """The empty marker pair a skeleton uses for one generated section."""
    return f"{BEGIN.format(name)}\n{END}"


def fill(text, sections):
    """Replace each named generated block in `text`; return (new, missing)."""
    seen = set()

    def repl(m):
        name = m["name"]
        if name not in sections:
            return m[0]
        seen.add(name)
        body = sections[name].rstrip("\n")
        return f"{BEGIN.format(name)}\n{body}\n{END}"

    out = BLOCK_RE.sub(repl, text)
    return out, sorted(set(sections) - seen)


def write_sections(path, skeleton, sections, init=False, check=False):
    """Rewrite the generated sections of `path`.

    Returns 0 when the file is current or was written, 1 when `check` finds
    it out of date, and exits when a section's markers are missing.
    """
    path = Path(path)
    if init or not path.exists():
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        base = skeleton
    else:
        old = path.read_text(encoding="utf-8")
        base = old
    new, missing = fill(base, sections)
    if missing:
        sys.exit(f"{path}: no markers for generated section(s) "
                 + ", ".join(missing) + "; run with --init to rewrite the "
                 "whole file from the skeleton")
    if not new.endswith("\n"):
        new += "\n"
    if check:
        if new != old:
            print(f"{path.name}: out of date")
            return 1
        print(f"{path.name}: current")
        return 0
    if new != old:
        path.write_text(new, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(f"{path.name}: unchanged")
    return 0


def sort_ids(ids):
    return sorted(set(ids), key=epics.id_key)


def compress(ids):
    """'PLC-E2-T01, PLC-E2-T02, PLC-E2-T03' -> 'PLC-E2-T01–T03'."""
    out, run = [], []

    def flush():
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
        else:
            last = run[-1].rsplit("-T", 1)[1]
            out.append(f"{run[0]}–T{last}")

    for tid in sort_ids(ids):
        if run:
            pe, pn = run[-1].rsplit("-T", 1)
            ce, cn = tid.rsplit("-T", 1)
            if ce == pe and int(cn) == int(pn) + 1:
                run.append(tid)
                continue
            flush()
        run = [tid]
    flush()
    return ", ".join(out)


def counted(ids):
    """'3: A, B–C', or '–' when empty."""
    ids = sort_ids(ids)
    if not ids:
        return "–"
    return f"{len(ids)}: {compress(ids)}"


def cell(text):
    """Make text safe inside a Markdown table cell."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def abridge(text, limit=110):
    """`text` if short, else its first sentence, else a cut at `limit`."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    m = re.match(r"(.+?)\.(?=\s+[A-Z`(]|$)", text)
    if m and len(m[1]) <= limit and m[1].count("`") % 2 == 0:
        return m[1]
    cut = text[:limit]
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    # Never end inside an inline code span.
    if cut.count("`") % 2:
        cut = cut[:cut.rfind("`")]
    return cut.rstrip(" ,;:(") + "…"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def milestones_text(rfc0001_text):
    """RFC-0001 §16 without its heading, or "" if it is missing."""
    m = re.search(r"^## 16\. Milestones\n(.*?)(?=^## )", rfc0001_text,
                  re.S | re.M)
    return m[1] if m else ""


def milestone_section(rfc0001_text, n):
    """The title and body of RFC-0001 §16's `### <milestone>:` subsection,
    where n is a position in epics.PHASES or a name such as "M0a"."""
    name = n if isinstance(n, str) else label(n)
    m = re.search(rf"^### {name}: ?(?P<title>[^\n]*)\n(?P<body>.*?)"
                  r"(?=^### |\Z)", milestones_text(rfc0001_text), re.S | re.M)
    if not m:
        return None, ""
    return m["title"].strip(), m["body"].strip()


def rfc0001_subsection(rfc0001_text, title):
    """The body of the RFC-0001 §16 subsection headed `### <title>`."""
    m = re.search(rf"^### {re.escape(title)}\n(?P<body>.*?)(?=^### |\Z)",
                  milestones_text(rfc0001_text), re.S | re.M)
    return m["body"].strip() if m else None


def milestone_bullets(body):
    """The top-level `- ` items of a milestone body except its `Exit:` item,
    each with its indented continuation lines folded in."""
    items = []
    for line in body.splitlines():
        if line.startswith("- "):
            items.append(line[2:].strip())
        elif line.startswith("  ") and items and line.strip():
            items[-1] += " " + line.strip()
        elif line.strip():
            items.append(None)  # a paragraph ends the list
    return [i for i in items if i and not i.startswith("Exit:")]


def rfc_path(rfc_id):
    hits = sorted((PLAN_DIR / "rfcs").glob(f"{rfc_id}-*.md"))
    return hits[0] if hits else None


def parse_md_table(lines, start):
    """Parse the Markdown table that begins at lines[start]; return
    (header, rows) where rows are lists of cells."""
    header = epics.split_row(lines[start])
    rows = []
    i = start + 2
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        rows.append(epics.split_row(lines[i]))
        i += 1
    return header, rows


def epic_info():
    """Every epic heading in the epic files: {epic_id: {title, goal, rfcs}}.

    `rfcs` lists the RFC IDs on the epic's "RFC sections implemented" line,
    in order of first appearance.
    """
    out = {}
    for area in AREAS:
        path = epics.EPICS_DIR / f"{area}.md"
        if not path.exists():
            continue
        cur = None
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^## Epic (?P<id>[A-Z]+-E\d+): ?(?P<title>.*)$",
                         line)
            if m:
                cur = {"title": m["title"].strip(), "goal": "", "rfcs": []}
                out[m["id"]] = cur
                continue
            if cur is None:
                continue
            m = re.match(r"^- (?:\*\*)?Goal[:.](?:\*\*)? ?(.*)$", line)
            if m:
                cur["goal"] = m[1].strip()
            m = re.match(r"^- (?:\*\*)?RFC sections implemented:(?:\*\*)? ?"
                         r"(.*)$", line)
            if m:
                for rid in re.findall(r"RFC-\d{4}", m[1]):
                    if rid not in cur["rfcs"]:
                        cur["rfcs"].append(rid)
    return out


def open_questions():
    """Open OQ rows and Settled rows from plan/open-questions.md.

    Returns (oqs, settled): lists of dicts keyed by lower-cased header
    names, plus 'section'. Rows of any other section are ignored.
    """
    path = PLAN_DIR / "open-questions.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    oqs, settled = [], []
    section = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            section = line[3:].strip()
        if line.lstrip().startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\|\s*:?-{3,}", lines[i + 1]):
            header, rows = parse_md_table(lines, i)
            keys = [h.strip().lower() for h in header]
            for r in rows:
                if len(r) != len(keys):
                    continue
                d = dict(zip(keys, r))
                d["section"] = section
                if section == "Settled":
                    settled.append(d)
                elif re.match(r"OQ-[A-Z]+-\d+$", d.get("id", "")):
                    oqs.append(d)
            i += len(rows) + 2
            continue
        i += 1
    return oqs, settled


def load_scs():
    text = (PLAN_DIR / "00-vision-and-scope.md").read_text(encoding="utf-8")
    m = re.search(r"^## 7\. .*?(?=^## )", text, re.S | re.M)
    scs = {}
    for line in (m[0] if m else "").splitlines():
        if re.match(r"^\|\s*SC-\d+\s*\|", line):
            cells = epics.split_row(line)
            n = int(cells[0].split("-")[1])
            scs[n] = cells[1]
    return scs


def sc_refs(text, known):
    """SC numbers named in text, with ranges such as 'SC-1 to SC-11'."""
    out = set()
    for m in SC_RE.finditer(text or ""):
        a = int(m[1])
        b = m[2] or m[3]
        if b and int(b) > a:
            out.update(n for n in range(a, int(b) + 1) if n in known)
        else:
            out.add(a)
    return out


def sc_gates(rfc0001_text, known):
    """First milestone whose RFC-0001 §16 'Exit:' sentence names each SC."""
    gate = {}
    for n in MILESTONES:
        _, body = milestone_section(rfc0001_text, n)
        m = re.search(r"^- Exit:\s*(.*)$", body, re.M)
        if not m:
            continue
        first = re.match(r"(.+?\.)(?=\s+[A-Z]|\s*$)", m[1])
        sentence = first[1] if first else m[1]
        for sc in sorted(sc_refs(sentence, known)):
            gate.setdefault(sc, n)
        if label(n) == "M4" and re.search(r"\bevery SC\b", sentence):
            for sc in known:
                gate.setdefault(sc, n)
    return gate


def _oq_numbers(text):
    """Open-question numbers cited in text: 'open questions 1, 2 and 7'."""
    out = set()
    for m in re.finditer(r"[Oo]pen questions? ((?:\d+(?:, | and |–| to )?)+)",
                         text or ""):
        out.update(int(n) for n in re.findall(r"\d+", m[1]))
    return out


def oq_task_links(oq, records, info):
    """Tasks that cite the RFC open question an OQ row names.

    An OQ row names its source as 'RFC-NNNN OQn' in the Resolution, Question
    or Why column. A task matches when its Task, Implements or acceptance
    cell cites open question n and RFC-NNNN appears in its Implements cell or
    on its epic's "RFC sections implemented" line.
    """
    text = " ".join(oq.get(k, "") for k in ("resolution", "question",
                                            "why it matters"))
    refs = {(rid, int(n)) for rid, n in
            re.findall(r"(RFC-\d{4}) OQ ?(\d+)", text)}
    hits = []
    for rid, n in sorted(refs):
        for r in records:
            ctx = set(re.findall(r"RFC-\d{4}", r["implements"]))
            ctx |= set(info.get(r["epic"], {}).get("rfcs", []))
            if rid not in ctx:
                continue
            cells = " ".join((r["task"], r["implements"], r["acceptance"]))
            if n in _oq_numbers(cells):
                hits.append(r["id"])
    return sort_ids(hits)
