"""Mirror the plan's epics and tasks into GitHub issues and a project board.

The epic files are the source of truth. Each epic becomes an issue of type
Epic, each task an issue of type Task and a sub-issue of its epic, and each
task's dependencies, the edges depcheck.py reads from its "Depends on" cell,
become native "blocked by" links; a link between two synced issues that the
plan doesn't have is removed, including one a person added by hand. All of
them go on one public project board linked to the repository. Running the
script again overwrites the title, body, labels, milestone and type of any
issue whose generated content changed, and leaves alone status, assignees,
comments and open or closed state.

The script adopts an issue only if all three hold: its body ends with the
task marker; its author is an owner, member or collaborator of the
repository (GitHub's authorAssociation), whoever runs the sync; and it is an
item of the organisation's project number PROJECT_NUMBER, which only accounts
with write access to that project can add to. Markers on other issues are
counted and ignored. Before the first write of any phase, setup
included, the run stops if two adopted issues carry the same ID, or if it
adopted nothing although the repository already has marked issues or more
issues than half the plan: the board exists, and creating it again would
duplicate it. Synced issues whose task or epic has left the plan are listed
as orphans and left open.

A run that writes first fetches origin's main branch, and refuses to start
if the plan inputs (PLAN_INPUTS: the epic files and the tools that turn them
into issue text, labels and links) differ from it. A run from an out-of-date
checkout would otherwise undo newer issue text and links. Only those inputs
are compared, not HEAD or this script, so a branch that changes only this
script can still run. --dry-run skips that check.

    python3 gh_sync.py --dry-run                       # what would change
    python3 gh_sync.py --banned FILE --phase setup     # labels, milestones, issue type, project, its README, views
    python3 gh_sync.py --banned FILE --phase issues    # epics, then tasks in dependency order
    python3 gh_sync.py --banned FILE --phase deps      # "blocked by" links
    python3 gh_sync.py --banned FILE --phase all

It needs the `gh` CLI, logged in with the `repo` and `project` scopes. Epics
use the org's "Epic" issue type if one exists, and "Feature" otherwise. Every
title and body is checked against the D5 banned-terms file (--banned), which
is kept outside the repository, and the run stops on any match. GitHub allows
about 500 content-creating requests an hour, so the script waits --interval
seconds between writes (default 9) and backs off when a call fails with a
rate limit. A write that fails with a server or network error may still have
landed, so the run pauses and re-reads what exists before it resumes. Every
run starts by reading what already exists, so an interrupted run can simply
be started again.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import depcheck  # noqa: E402
import epics  # noqa: E402
import roadmap  # noqa: E402

TOOLS_DIR = Path(__file__).resolve().parent
PLAN_DIR = TOOLS_DIR.parent.parent
REPO_ROOT = PLAN_DIR.parent
# What a run's output depends on: the epic files, and the code that turns
# them into issue text, labels and links. epics.py parses the rows,
# depcheck.py the dependency edges, roadmap.py's KIND_RE the spike label, and
# sizing.py the size and maintainer labels. check_fresh() compares these with
# origin/main. gh_sync.py itself is left out on purpose, so a branch that
# changes only this script can still run against the live board. Paths are
# relative to the repository root.
PLAN_INPUTS = ("plan/implementation/epics",
               *(f"plan/implementation/tools/{f}"
                 for f in ("epics.py", "depcheck.py", "roadmap.py", "sizing.py")))
TRUSTED = ("OWNER", "MEMBER", "COLLABORATOR")

PROJECT_TITLE = "Syntropy"
# The board is the organisation's project with this number. Adoption and
# setup() match it by number and owner, not by title, because any member who
# can create projects could make another one with the same title.
PROJECT_NUMBER = 1
MARKER = "syntropy-task"
MILESTONES = ("M0a", "M0b", "M1a", "M1b", "M2", "M3", "M4")

# Two lanes, by area (AGENTS.md, "Lanes").
LANES = {"COMP": "engine", "PLC": "engine", "TIER": "engine",
         "QUAL": "harness-surface", "PROD": "harness-surface",
         "DIST": "harness-surface"}
AREA_NAMES = {"PLC": "placement and scheduling", "TIER": "tiers, I/O and KV",
              "DIST": "distributed, network and security",
              "COMP": "compute, models and fidelity",
              "PROD": "product surface", "QUAL": "quality, CI and release"}

LABELS = {
    **{f"area:{a}": ("1d76db", f"Area {a}: {n}") for a, n in AREA_NAMES.items()},
    "lane:engine": ("5319e7", "Lane: the COMP, PLC and TIER areas"),
    "lane:harness-surface": ("0e8a16", "Lane: the QUAL, PROD and DIST areas"),
    **{f"size:{c}": ("c5def5", f"Estimated size class {c} (sizing.py, [estimate])")
       for c in ("S", "M", "L", "XL")},
    "spike": ("fbca04", "Spike: a timeboxed harness that answers a question"),
    "maintainer-action": ("d93f0b", "Needs a person: hardware, accounts, keys or other human-only work"),
    "maintainer-approval": ("b60205", "Touches a class of change that needs a maintainer's approval"),
    "epic": ("3e4b9e", "An epic: its tasks are its sub-issues"),
}

VIEWS = (
    ("Board", "BOARD_LAYOUT", ""),
    ("Engine lane", "BOARD_LAYOUT", 'label:"lane:engine"'),
    ("Harness and surface lane", "BOARD_LAYOUT", 'label:"lane:harness-surface"'),
    ("M0a", "BOARD_LAYOUT", "milestone:M0a"),
    ("Maintainer actions", "TABLE_LAYOUT", "label:maintainer-action"),
    ("Epics", "TABLE_LAYOUT", "label:epic"),
    ("All tasks", "TABLE_LAYOUT", "-label:epic"),
)

PROJECT_README = """Every epic and task in the Syntropy plan, as issues.

The epic files under `plan/implementation/epics/` are the source of truth. The issues are generated from them by `plan/implementation/tools/gh_sync.py`, so change the plan, not an issue's description.

- **Pick ready work.** A task is ready when every task in its plan "Depends on" cell is done; `python3 plan/implementation/tools/depcheck.py --task <ID>` lists them, and each issue's "blocked by" links mirror that cell. The Engine and Harness lanes split the work by area (AGENTS.md, "Lanes").
- **Claim a task** before you start. With write access, assign yourself on its issue and move it to In Progress. Without it, say in a comment that you're taking it, and a maintainer assigns you. Skip tasks someone else is assigned to.
- **Close it** through the pull request that meets its acceptance criteria: put `Closes #<issue>` in the PR description.
- **Maintainer action** tasks need a person, and **maintainer-approval** tasks touch a change class that needs a maintainer's approval.
"""


class Interrupted(RuntimeError):
    """A rate limit, a network or server failure, or a write whose result is
    unknown. The phase pauses, re-reads what exists and resumes."""


LIMIT_TEXT = re.compile(r"secondary rate limit|rate limit exceeded|"
                        r"submitted too quickly|abuse detection", re.I)
NETWORK_TEXT = re.compile(r"HTTP 50[0234]\b|error connecting|connection (?:reset|refused)|"
                          r"i/o timeout|TLS handshake timeout|unexpected EOF|no such host",
                          re.I)


def error_text(out):
    """GitHub's own error fields in a failed call's output: the GraphQL
    `errors[].type` and `message`, or the REST `message`. Never its data."""
    try:
        doc = json.loads(out)
    except ValueError:
        return ""
    if not isinstance(doc, dict):
        return ""
    parts = [str(doc.get("message", ""))]
    for e in doc.get("errors") or []:
        if isinstance(e, dict):
            parts += [str(e.get("type", "")), str(e.get("message", ""))]
    return " ".join(p for p in parts if p)


def rate_limited(stderr, out):
    """Whether a failed call was refused for a rate limit: a GraphQL error of
    type RATE_LIMITED, HTTP 429, or GitHub's rate-limit wording in gh's error
    message or the response's error fields."""
    errors = error_text(out)
    return ("RATE_LIMITED" in errors.split() or "HTTP 429" in stderr
            or bool(LIMIT_TEXT.search(stderr + " " + errors)))


class GitHub:
    """A thin client over `gh api` that paces writes and retries on limits."""

    def __init__(self, repo, interval, dry_run):
        self.owner, self.name = repo.split("/")
        self.interval = interval
        self.dry_run = dry_run
        self.last_write = 0.0
        self.writes = 0

    def _run(self, args, payload=None, write=False):
        # Only a failed call is inspected. A successful read returns issue
        # bodies, and their text must never decide whether to back off.
        for attempt in range(8):
            proc = subprocess.run(["gh", "api", *args], input=payload,
                                  capture_output=True, text=True)
            out, err = proc.stdout.strip(), proc.stderr.strip()
            if proc.returncode == 0:
                return json.loads(out) if out else {}
            what = f"gh api {' '.join(args[:2])}: {err[:300]}"
            wait = min(60 * 2 ** attempt, 900)
            if rate_limited(err, out):
                # GitHub refused the request, so it is safe to repeat.
                print(f"  rate limited; waiting {wait} s: {err[:120]}", flush=True)
                time.sleep(wait)
                continue
            if NETWORK_TEXT.search(err + " " + error_text(out)):
                if write:
                    # The write may have landed. Re-read before resuming,
                    # so a create is never repeated.
                    raise Interrupted(what)
                print(f"  network or server error; waiting {wait} s: {err[:120]}", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"{what} {error_text(out)[:300]}".strip())
        raise Interrupted("gave up after repeated rate limits or network errors")

    def query(self, query, write=False, **variables):
        data = self._run(["graphql", "--input", "-"],
                         json.dumps({"query": query, "variables": variables}), write)
        if data.get("errors"):
            raise RuntimeError(json.dumps(data["errors"])[:600])
        return data["data"]

    def mutate(self, what, query, **variables):
        """A content-creating write: paced, and skipped in a dry run."""
        if self.dry_run:
            print(f"  would {what}")
            return None
        wait = self.interval - (time.monotonic() - self.last_write)
        if wait > 0:
            time.sleep(wait)
        self.last_write = time.monotonic()
        self.writes += 1
        return self.query(query, write=True, **variables)

    def rest(self, path):
        sep = "&" if "?" in path else "?"
        return self._run([f"{path}{sep}per_page=100"])

    def rest_write(self, what, path, **fields):
        """A paced REST POST, skipped in a dry run."""
        if self.dry_run:
            print(f"  would {what}")
            return None
        wait = self.interval - (time.monotonic() - self.last_write)
        if wait > 0:
            time.sleep(wait)
        self.last_write = time.monotonic()
        self.writes += 1
        args = ["-X", "POST", path]
        for k, v in fields.items():
            args += ["-f", f"{k}={v}"]
        return self._run(args, write=True)


# ---------------------------------------------------------------- the plan

def slug(heading):
    s = re.sub(r"[^\w\- ]", "", heading.lower())
    return s.replace(" ", "-")


def load_epics():
    """Each epic's title, goal and header lines, from the epic files."""
    out = {}
    for area in epics.AREAS:
        path = epics.EPICS_DIR / f"{area}.md"
        cur = None
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"^## (Epic ([A-Z]+-E\d+): (.+))$", line)
            if m:
                cur = {"id": m[2], "title": m[3].strip(), "area": area,
                       "file": f"epics/{area}.md", "anchor": slug(m[1]),
                       "line": lineno, "meta": []}
                out[cur["id"]] = cur
            elif cur and line.startswith("### Tasks"):
                cur = None
            elif cur and line.startswith("- "):
                cur["meta"].append(re.sub(r"^\*\*(.+?)[:.]\*\*\s*", r"\1: ", line[2:].strip()))
    return out


def check_fresh(root=REPO_ROOT):
    """Stop unless the plan inputs in this checkout match a freshly fetched
    origin/main. The working tree is compared, so uncommitted edits count."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)

    fetch = git("fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main")
    if fetch.returncode:
        raise SystemExit("could not fetch origin's main branch, so the plan inputs can't "
                         f"be checked; nothing was written: {fetch.stderr.strip()[:300]}")
    diff = git("diff", "--name-only", "origin/main", "--", *PLAN_INPUTS)
    if diff.returncode:
        raise SystemExit(f"git diff against origin/main failed: {diff.stderr.strip()[:300]}")
    changed = [line for line in diff.stdout.splitlines() if line]
    if changed:
        raise SystemExit(f"{len(changed)} plan input(s) differ from origin/main: "
                         f"{', '.join(changed)}. Nothing was written. Run from an up-to-date "
                         f"main, or use --dry-run to see what this checkout would change.")
    print("plan inputs match origin/main")


def load_sizes():
    proc = subprocess.run([sys.executable, "-B", str(TOOLS_DIR / "sizing.py"), "--json"],
                          capture_output=True, text=True, check=True)
    return {r["id"]: r for r in json.loads(proc.stdout)}


def plan_graph():
    """The tasks and each task's dependencies, from depcheck.py's edges.

    depcheck expands ranges such as PLC-E2-T01–T05 and skips task IDs inside
    a note, so the "blocked by" links match the plan's "Depends on" cells
    exactly. The run stops if depcheck reports any problem."""
    res = depcheck.analyse()
    if res["problems"]:
        raise SystemExit(f"{len(res['problems'])} task-graph problems; "
                         f"run depcheck.py, which must exit 0")
    deps = {t: set() for t in res["tasks"]}
    for e in res["edges"]:
        deps[e["from"]].add(e["to"])
    records = sorted(res["tasks"].values(), key=lambda r: epics.id_key(r["id"]))
    return records, {t: sorted(d, key=epics.id_key) for t, d in deps.items()}


def topo_order(tasks, deps):
    """Tasks with their dependencies first, ties broken by task ID."""
    ids = {t["id"] for t in tasks}
    done, order = set(), []

    def visit(t, stack=()):
        if t in done:
            return
        if t in stack:
            raise SystemExit(f"dependency cycle through {t}; run depcheck.py")
        for d in deps[t]:
            visit(d, stack + (t,))
        done.add(t)
        order.append(t)

    for t in sorted(ids, key=epics.id_key):
        visit(t)
    return order


def short_title(task):
    """The task cell's first sentence, cut at a word boundary to fit a title."""
    text = re.sub(r"\*\*|__", "", task).strip()
    head = text
    for m in re.finditer(r"(?<=[a-z0-9`)\]])[.;]\s", text):
        cand = text[:m.start()]
        if cand.count("(") == cand.count(")") and cand.count("`") % 2 == 0:
            head = cand
            break
    if len(head) > 110:
        head = head[:107].rsplit(" ", 1)[0].rstrip(",;:(") + "…"
    return head


def blob(base, rel, anchor=None):
    url = f"{base}/blob/main/plan/implementation/{rel}"
    return f"{url}#{anchor}" if anchor else url


# ----------------------------------------------------------- issue content

def split_list(text):
    """Split a cell on top-level semicolons (outside brackets and code)."""
    parts, cur, depth, code = [], "", 0, False
    for ch in text:
        if ch == "`":
            code = not code
        elif not code and ch in "([":
            depth += 1
        elif not code and ch in ")]":
            depth = max(depth - 1, 0)
        if ch == ";" and depth == 0 and not code:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


def checklist(text):
    items = split_list(text)
    return "\n".join(f"- [ ] {i[0].upper() + i[1:]}" for i in items) if items else "None."


def epic_issue(e, base):
    goal = next((m[len("Goal:"):].strip() for m in e["meta"] if m.startswith("Goal:")), "")
    rest = "\n".join(f"- {m}" for m in e["meta"] if not m.startswith("Goal:"))
    goal = goal[0].upper() + goal[1:] if goal else "See the epic file."
    body = (f"### Description\n\n{goal}\n\n"
            f"### Scope\n\n- Area: {e['area']} ({AREA_NAMES[e['area']]}), "
            f"lane `{LANES[e['area']]}`\n{rest}\n\n"
            f"### Acceptance criteria\n\n- [ ] Every sub-issue below is closed, each "
            f"against its own acceptance criteria.\n\n"
            f"Source: [{e['file']}, epic {e['id']}]({blob(base, e['file'], e['anchor'])}). "
            f"The epic file is the source of truth; edit the plan, not this description.")
    labels = sorted([f"area:{e['area']}", f"lane:{LANES[e['area']]}", "epic"])
    return {"id": e["id"], "title": f"[{e['id']}] {e['title']}", "body": body,
            "labels": labels, "milestone": None, "type": "Epic", "parent": None}


SPLIT_PREFIX = re.compile(r"^Windows and macOS part of [A-Z]+-E\d+-T\d+:\s*")


def is_spike(task):
    """roadmap.py's spike kind (KIND_RE), read at the start of the Task cell,
    after a split row's "Windows and macOS part of <ID>:" prefix, or a leading
    label that ends in "spike", as in "Platform facts spike: …". A spike that
    the cell only mentions, such as "the S-04 spike settled", doesn't count."""
    text = SPLIT_PREFIX.sub("", task)
    m = roadmap.KIND_RE.match(text)
    if m:
        return m["kind"].lower().startswith("spike")
    return bool(re.match(r"[\w ]*\bspikes?\b(?::|$)", text, re.I))


def task_issue(t, epic, size, dep_ids, numbers, base):
    deps = "\n".join(f"- #{numbers[d]} {d}" if d in numbers else f"- {d}"
                     for d in dep_ids) or "None."
    cell = t["depends"].strip()
    note = "" if not cell or cell.lower() in ("none", "—", "-") else \
        f"\n\nThe plan's dependency cell: {cell}"
    size_line = f"\n- Size class: {size['class']} [estimate]" if size else ""
    rigs = f"\n- Rigs: {', '.join(size['rigs'])}" if size and size.get("rigs") else ""
    body = (f"### Description\n\n{t['task']}\n\n"
            f"### Acceptance criteria\n\n{checklist(t['acceptance'])}\n\n"
            f"### Implements\n\n{t['implements']}\n\n"
            f"### Depends on\n\n{deps}{note}\n\n"
            f"### Details\n\n- Task: {t['id']}\n- Milestone: {t['milestone']}\n"
            f"- Epic: [{epic['id']}: {epic['title']}]({blob(base, epic['file'], epic['anchor'])})\n"
            f"- Lane: `{LANES[t['area']]}`{size_line}{rigs}\n"
            f"- Source: [{t['file']}, line {t['line']}]({blob(base, t['file'])}#L{t['line']}). "
            f"The epic file is the source of truth; this description is regenerated "
            f"from it by `plan/implementation/tools/gh_sync.py`.")
    labels = [f"area:{t['area']}", f"lane:{LANES[t['area']]}"]
    if size:
        labels.append(f"size:{size['class']}")
        if size.get("maintainer"):
            labels.append("maintainer-action")
        if size.get("approval"):
            labels.append("maintainer-approval")
    if is_spike(t["task"]):
        labels.append("spike")
    return {"id": t["id"], "title": f"[{t['id']}] {short_title(t['task'])}",
            "body": body, "labels": sorted(labels), "milestone": t["phase"],
            "type": "Task", "parent": epic["id"]}


def digest(item):
    key = json.dumps([item["title"], item["body"], item["labels"],
                      item["milestone"], item["type"], item["parent"]])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def with_marker(item):
    return f"{item['body']}\n\n<!-- {MARKER}: {item['id']} {digest(item)} -->"


def check_banned(items, banned):
    pats = [re.compile(p, re.I) for p in banned.read_text().splitlines() if p.strip()]
    bad = [(i["id"], p.pattern) for i in items for p in pats
           if p.search(i["title"]) or p.search(i["body"])]
    if bad:
        raise SystemExit(f"D5 check failed on {len(bad)} item(s), first: {bad[0][0]}")


# ------------------------------------------------------------ GitHub state

ISSUES_Q = """query($o:String!,$n:String!,$after:String){ repository(owner:$o,name:$n){
  issues(first:100, after:$after, states:[OPEN,CLOSED]){ totalCount
    pageInfo{hasNextPage endCursor}
    nodes{ id number body authorAssociation blockedBy(first:50){ nodes{ id } }
      projectItems(first:20){ nodes{ project{ number owner{ ... on Organization{ login } } } } }
    } } } }"""


def existing_issues(gh):
    """The synced issues by task or epic ID, how many markers were ignored,
    and how many issues the repository has.

    Anyone can open an issue on a public repository and copy a marker into
    it, and an organisation member may have read access only. So an issue is
    adopted only if its body ends with the marker, its author is an owner,
    member or collaborator, and it is an item of the board (the org's project
    PROJECT_NUMBER, owner checked), which needs write access to the project.
    The sync creates every issue straight into the board. Two adopted issues
    with the same ID stop the run."""
    found, ignored, clashes, after, total = {}, 0, [], None, 0
    while True:
        r = gh.query(ISSUES_Q, o=gh.owner, n=gh.name, after=after)["repository"]["issues"]
        total = r["totalCount"]
        for n in r["nodes"]:
            m = re.search(rf"<!-- {MARKER}: (\S+) (\w+) -->\s*\Z", n["body"] or "")
            if not m:
                continue
            on_board = any(i["project"]["number"] == PROJECT_NUMBER
                           and (i["project"]["owner"] or {}).get("login") == gh.owner
                           for i in n["projectItems"]["nodes"])
            if n["authorAssociation"] not in TRUSTED or not on_board:
                ignored += 1
                continue
            if m[1] in found:
                clashes.append(f"{m[1]} on #{found[m[1]]['number']} and #{n['number']}")
                continue
            found[m[1]] = {"id": n["id"], "number": n["number"], "digest": m[2],
                           "blocked_by": {b["id"] for b in n["blockedBy"]["nodes"]}}
        if not r["pageInfo"]["hasNextPage"]:
            break
        after = r["pageInfo"]["endCursor"]
    if clashes:
        raise SystemExit(f"{len(clashes)} ID(s) carried by two adopted issues: "
                         f"{'; '.join(clashes[:10])}. Stopped before the next write; "
                         f"close the extra issue or remove its marker, then run again.")
    return found, ignored, total


def check_adoption(existing, ignored, total, planned):
    """Stop before the first write if the board exists but nothing was adopted.

    A run that adopts nothing creates the whole board, which is right only
    on a repository the sync has never written to. If the repository already
    has marked issues, or more issues than half of the `planned` epics and
    tasks, the board exists and adoption has failed: its author may have left
    the organisation, or the marker or the token may be wrong. Creating it
    again would duplicate it. An interrupted first run adopted something, so
    it still resumes."""
    if not existing and (ignored or total > planned // 2):
        raise SystemExit(f"adopted no issues, but the repository has {total} issues and "
                         f"{ignored} task marker(s) on issues that weren't adopted (author "
                         f"not an owner, member or collaborator, or issue not on project "
                         f"#{PROJECT_NUMBER}). Creating the board again would duplicate it, "
                         f"so nothing was written.")


def setup(gh, ctx):
    repo = gh.query("""query($o:String!,$n:String!){ repository(owner:$o,name:$n){ id
        owner{ id ... on Organization { issueTypes(first:50){ nodes{ id name } }
        projectsV2(first:50){ nodes{ id title number url readme } } } } } }""",
                    o=gh.owner, n=gh.name)["repository"]
    ctx["repo_id"], ctx["owner_id"] = repo["id"], repo["owner"]["id"]

    have = {l["name"] for l in gh.rest(f"repos/{gh.owner}/{gh.name}/labels")}
    for name, (color, desc) in LABELS.items():
        if name not in have:
            gh.rest_write(f"create label {name}", f"repos/{gh.owner}/{gh.name}/labels",
                          name=name, color=color, description=desc)
    have = {m["title"] for m in gh.rest(f"repos/{gh.owner}/{gh.name}/milestones?state=all")}
    for title in MILESTONES:
        if title not in have:
            gh.rest_write(f"create milestone {title}", f"repos/{gh.owner}/{gh.name}/milestones",
                          title=title,
                          description=f"Milestone {title}. Exit criteria: RFC-0001 §16.")

    types = {t["name"]: t["id"] for t in repo["owner"]["issueTypes"]["nodes"]}
    # Creating an issue type needs the broad admin:org scope, so the script
    # uses the org's types as they are: "Epic" if a maintainer has added it,
    # otherwise "Feature", with the `epic` label on every epic either way.
    if "Epic" not in types and "Feature" in types:
        types["Epic"] = types["Feature"]
    ctx["types"] = types

    project = next((p for p in repo["owner"]["projectsV2"]["nodes"]
                    if p["number"] == PROJECT_NUMBER), None)
    if not project:
        r = gh.mutate("create the project", """mutation($o:ID!,$r:ID!,$t:String!){
            createProjectV2(input:{ownerId:$o, repositoryId:$r, title:$t}){
            projectV2{ id number url } } }""", o=ctx["owner_id"], r=ctx["repo_id"],
                      t=PROJECT_TITLE)
        project = r["createProjectV2"]["projectV2"] if r else {"id": "PROJECT", "url": "(new)"}
        if r and project["number"] != PROJECT_NUMBER:
            raise SystemExit(f"created the project as #{project['number']}; set PROJECT_NUMBER "
                             f"to {project['number']} in gh_sync.py and run again")
        gh.mutate("make the project public", """mutation($p:ID!,$d:String!,$r:String!){
            updateProjectV2(input:{projectId:$p, public:true, shortDescription:$d, readme:$r}){
            projectV2{ id } } }""", p=project["id"],
                  d="Every epic and task in the Syntropy plan", r=PROJECT_README)
    elif (project.get("readme") or "").strip() != PROJECT_README.strip():
        gh.mutate("update the project README", """mutation($p:ID!,$r:String!){
            updateProjectV2(input:{projectId:$p, readme:$r}){ projectV2{ id } } }""",
                  p=project["id"], r=PROJECT_README)
    ctx["project"] = project
    print(f"project {project.get('url')}")
    if project["id"] != "PROJECT":
        views = gh.query("""query($p:ID!){ node(id:$p){ ... on ProjectV2 {
            views(first:50){ nodes{ id name } } } } }""", p=project["id"])
        names = {v["name"]: v["id"] for v in views["node"]["views"]["nodes"]}
    else:
        names = {}
    if "View 1" in names and VIEWS[0][0] not in names:
        gh.mutate("rename View 1", """mutation($v:ID!,$n:String!){
            updateProjectV2View(input:{viewId:$v, name:$n}){ projectV2View{ id } } }""",
                  v=names.pop("View 1"), n="Everything")
    for name, layout, flt in VIEWS:
        if name in names:
            continue
        r = gh.mutate(f"create view {name}", """mutation($p:ID!,$n:String!,$l:ProjectV2ViewLayout!){
            createProjectV2View(input:{projectId:$p, name:$n, layout:$l}){
            projectV2View{ id } } }""", p=project["id"], n=name, l=layout)
        if r and flt:
            gh.mutate(f"filter view {name}", """mutation($v:ID!,$f:String!){
                updateProjectV2View(input:{viewId:$v, filter:$f}){ projectV2View{ id } } }""",
                      v=r["createProjectV2View"]["projectV2View"]["id"], f=flt)


def lookup(gh, ctx):
    """Label, milestone and issue-type IDs that issue writes need."""
    r = gh.query("""query($o:String!,$n:String!){ repository(owner:$o,name:$n){
        labels(first:100){ nodes{ id name } } milestones(first:50, states:[OPEN,CLOSED]){
        nodes{ id title } } } }""", o=gh.owner, n=gh.name)["repository"]
    ctx["label_ids"] = {l["name"]: l["id"] for l in r["labels"]["nodes"]}
    ctx["milestone_ids"] = {m["title"]: m["id"] for m in r["milestones"]["nodes"]}


CREATE_Q = """mutation($r:ID!,$t:String!,$b:String!,$l:[ID!],$m:ID,$ty:ID,$p:[ID!],$par:ID){
  createIssue(input:{repositoryId:$r, title:$t, body:$b, labelIds:$l, milestoneId:$m,
  issueTypeId:$ty, projectV2Ids:$p, parentIssueId:$par}){ issue{ id number } } }"""
UPDATE_Q = """mutation($i:ID!,$t:String!,$b:String!,$l:[ID!],$m:ID,$ty:ID){
  updateIssue(input:{id:$i, title:$t, body:$b, labelIds:$l, milestoneId:$m,
  issueTypeId:$ty}){ issue{ id } } }"""


def write_issue(gh, ctx, item, existing):
    v = dict(t=item["title"], b=with_marker(item),
             l=[ctx["label_ids"].get(x, x) for x in item["labels"]],
             m=ctx["milestone_ids"].get(item["milestone"]) if item["milestone"] else None,
             ty=ctx["types"].get(item["type"]))
    old = existing.get(item["id"])
    if old and old["digest"] == digest(item):
        return old, False
    if old:
        gh.mutate(f"update #{old['number']} {item['id']}", UPDATE_Q, i=old["id"], **v)
        return old, True
    parent = existing.get(item["parent"], {}).get("id") if item["parent"] else None
    r = gh.mutate(f"create {item['id']}", CREATE_Q, r=ctx["repo_id"],
                  p=[ctx["project"]["id"]], par=parent, **v)
    if r is not None and not (r.get("createIssue") or {}).get("issue"):
        # GitHub answers an over-limit write with an empty payload; the caller
        # pauses, re-reads what exists and resumes, so nothing is duplicated.
        raise Interrupted(f"empty response creating {item['id']}")
    issue = r["createIssue"]["issue"] if r else {"id": f"NEW-{item['id']}", "number": 0}
    rec = {"id": issue["id"], "number": issue["number"], "digest": digest(item),
           "blocked_by": set()}
    existing[item["id"]] = rec
    return rec, True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default="syntropyai-hq/syntropy")
    ap.add_argument("--phase", choices=("setup", "issues", "deps", "all"), default="all")
    ap.add_argument("--banned", type=Path, help="the D5 banned-terms file (required to write)")
    ap.add_argument("--interval", type=float, default=9.0,
                    help="seconds between writes (default 9, about 400 an hour, "
                         "under GitHub's 500 with room for other activity)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not (args.banned and args.banned.is_file()):
        ap.error("--banned FILE is required unless --dry-run")
    if not args.dry_run:
        check_fresh()

    records, deps = plan_graph()
    epic_map, sizes = load_epics(), load_sizes()
    order = topo_order(records, deps)
    by_id = {r["id"]: r for r in records}
    base = f"https://github.com/{args.repo}"

    gh = GitHub(args.repo, args.interval, args.dry_run)
    login = gh.query("query{ viewer{ login } }")["viewer"]["login"]
    # Every stop check runs here, before setup(), whose label, milestone,
    # project and view writes are the first writes of a run.
    existing, ignored, total = existing_issues(gh)
    print(f"run as {login}: {len(records)} tasks, {len(epic_map)} epics, "
          f"{sum(len(d) for d in deps.values())} dependency links; "
          f"{len(existing)} already on GitHub; "
          f"{ignored} marker(s) ignored (author not trusted, or issue not on "
          f"project #{PROJECT_NUMBER})")
    orphans = sorted((v["number"], k) for k, v in existing.items()
                     if k not in by_id and k not in epic_map)
    print(f"{len(orphans)} orphan(s), synced issues whose task or epic is no longer "
          f"in the plan, left open" + (": " if orphans else "")
          + ", ".join(f"#{n} {k}" for n, k in orphans))
    check_adoption(existing, ignored, total, len(records) + len(epic_map))

    epic_items = [epic_issue(e, base) for e in epic_map.values()]
    if args.banned:
        numbers = {k: v["number"] for k, v in existing.items()}
        check_banned(epic_items + [task_issue(by_id[t], epic_map[by_id[t]["epic"]],
                     sizes.get(t), deps[t], numbers, base) for t in order], args.banned)

    ctx = {}
    phases = ("setup", "issues", "deps") if args.phase == "all" else (args.phase,)
    if "setup" in phases or args.dry_run:
        setup(gh, ctx)
    else:
        setup(GitHub(args.repo, 0, True), ctx)  # read-only: fills in the IDs
    lookup(gh, ctx)

    def resilient(phase_fn):
        """Run a phase; on an interruption (a rate limit, a network or server
        error, or an empty create response), pause, re-read GitHub and resume.
        Any other error is a bug or a refusal, and fails at once."""
        nonlocal existing
        for attempt in range(30):
            try:
                if attempt:
                    existing, _, _ = existing_issues(gh)
                return phase_fn()
            except Interrupted as err:
                if gh.dry_run:
                    raise
                print(f"interrupted ({str(err)[:160]}); resuming in 5 minutes", flush=True)
                time.sleep(300)
        raise SystemExit("gave up after 30 interruptions")

    def issues_phase():
        changed = 0
        for item in epic_items:
            changed += write_issue(gh, ctx, item, existing)[1]
        for n, t in enumerate(order, 1):
            numbers = {k: v["number"] for k, v in existing.items()}
            item = task_issue(by_id[t], epic_map[by_id[t]["epic"]], sizes.get(t),
                              deps[t], numbers, base)
            changed += write_issue(gh, ctx, item, existing)[1]
            if n % 50 == 0:
                print(f"tasks {n}/{len(order)}, {gh.writes} writes so far", flush=True)
        print(f"issues: {changed} created or updated")

    def deps_phase():
        added = removed = 0
        synced = {v["id"]: k for k, v in existing.items()}
        # Earliest milestone first, so the work people start on is linked first;
        # the sort is stable, so each milestone keeps its dependency order.
        for t in sorted(order, key=lambda t: epics.phase_index(by_id[t]["phase"])):
            if t not in existing:
                continue
            for d in deps[t]:
                if d in existing and existing[d]["id"] not in existing[t]["blocked_by"]:
                    gh.mutate(f"mark {t} blocked by {d}", """mutation($i:ID!,$b:ID!){
                        addBlockedBy(input:{issueId:$i, blockingIssueId:$b}){ issue{ id } } }""",
                              i=existing[t]["id"], b=existing[d]["id"])
                    existing[t]["blocked_by"].add(existing[d]["id"])
                    added += 1
                    if added % 100 == 0:
                        print(f"dependency links {added}", flush=True)
            # A link between two synced issues that the plan no longer has is
            # stale: an edge was removed or reversed in an epic file. Links to
            # issues the script doesn't manage are left alone.
            want = {existing[d]["id"] for d in deps[t] if d in existing}
            for b in sorted(existing[t]["blocked_by"] - want):
                if b in synced:
                    gh.mutate(f"unmark {t} blocked by {synced[b]}", """mutation($i:ID!,$b:ID!){
                        removeBlockedBy(input:{issueId:$i, blockingIssueId:$b}){ issue{ id } } }""",
                              i=existing[t]["id"], b=b)
                    existing[t]["blocked_by"].discard(b)
                    removed += 1
        print(f"dependency links: {added} added, {removed} removed")

    if "issues" in phases:
        resilient(issues_phase)
    if "deps" in phases:
        resilient(deps_phase)
    print(f"done: {gh.writes} writes")


if __name__ == "__main__":
    main()
