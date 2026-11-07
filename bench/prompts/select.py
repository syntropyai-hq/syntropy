# SPDX-License-Identifier: MIT OR Apache-2.0
"""Selects the items of `bench-prompts-v1` from the parity corpus sources (QUAL-E12-T05).

RFC-0061 §3.1: the benchmark prompt set is drawn from the same licensed sources as the parity
corpus (`testdata/corpus/SOURCES.toml`). This script applies the set's selection rules to the
sources at their pinned revisions and writes the item manifest, `bench-prompts-v1.toml`: each
item's source, locator, licence and the SHA-256 of its text. With `--out DIR` it also writes
each item's text to `DIR/<sha256>`, the layout `syntropy_bench::prompts::DirStore` reads.

The rules are deterministic, so running the script again gives the same manifest; the
benchmark harness refuses item text whose SHA-256 differs from the manifest's.

Rules:
- `chat`: Taskmaster-1 `self-dialogs.json`, dialogs in file order, the first 24 with at least 6
  turns once consecutive turns by one speaker are joined with a newline (SOURCES.toml's rule).
  Rendered one turn per paragraph as `User: ...` or `Assistant: ...`.
- `agent`: Taskmaster-1 `woz-dialogs.json`, the first 8 dialogs with at least 8 turns, rendered
  as for `chat`; the manifest records each turn's byte length, so the harness can replay the
  conversation turn by turn with a shared prefix (vision S4).
- `code`: for each of the ten code sources, the first file in path order under its first
  include path whose extension is the source's language and whose size is 4–64 KiB.
- `long`: none in this version. PG-19's death-year rule needs the catalogue records that the
  corpus import reads (QUAL-E5-T22); QUAL-E12-T20 adds `long` items from that import.

Standard library only. Fetches from GitHub: raw files, and the git tree API, which allows 60
unauthenticated requests an hour; `--gh` lists trees through the GitHub CLI instead, and
`GITHUB_TOKEN` is sent when set.
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tomllib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCES = ROOT / "testdata/corpus/SOURCES.toml"
UA = "syntropy-bench-prompts/1 (https://github.com/syntropyai-hq/syntropy)"

CODE_EXT = {
    "code-rust-regex": ".rs",
    "code-c-sqlite": ".c",
    "code-go": ".go",
    "code-python-flask": ".py",
    "code-java-guava": ".java",
    "code-cpp-googletest": ".cc",
    "code-typescript": ".ts",
    "code-js-express": ".js",
    "code-ruby-sinatra": ".rb",
    "code-csharp-newtonsoft-json": ".cs",
}
CODE_MIN, CODE_MAX = 4 * 1024, 64 * 1024
CHAT_ITEMS, CHAT_MIN_TURNS = 24, 6
AGENT_ITEMS, AGENT_MIN_TURNS = 8, 8


def fetch(url, accept=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if accept:
        req.add_header("Accept", accept)
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def repo_path(url):
    return url.removeprefix("https://github.com/").rstrip("/")


def raw(source, path):
    return fetch(f"https://raw.githubusercontent.com/{repo_path(source['url'])}/{source['revision']}/{path}")


USE_GH = False


def tree(src, sha, recursive):
    q = "?recursive=1" if recursive else ""
    path = f"repos/{repo_path(src['url'])}/git/trees/{sha}{q}"
    if USE_GH:
        # The GitHub CLI's own login, for its higher API rate limit.
        return json.loads(subprocess.run(["gh", "api", path], check=True, capture_output=True).stdout)
    return json.loads(fetch(f"https://api.github.com/{path}"))


def subtree(src, prefix):
    """Every entry under `prefix` (a directory path ending in `/`), with full paths.

    The include directory's own tree is listed recursively, which stays under the API's
    truncation limit where the whole repository's would not.
    """
    sha = src["revision"]
    for part in prefix.strip("/").split("/"):
        listing = tree(src, sha, False)
        sha = next(e["sha"] for e in listing["tree"] if e["path"] == part and e["type"] == "tree")
    listing = tree(src, sha, True)
    if listing.get("truncated"):
        sys.exit(f"{src['id']}: the listing of {prefix} is truncated")
    for e in listing["tree"]:
        e["path"] = prefix + e["path"]
    return listing["tree"]


def turns(dialog, user, assistant):
    """Consecutive utterances by one speaker joined with a newline, as (role, text)."""
    out = []
    for u in dialog["utterances"]:
        role = {user: "User", assistant: "Assistant"}.get(u["speaker"])
        if role is None:
            continue
        text = u["text"]
        if out and out[-1][0] == role:
            out[-1] = (role, out[-1][1] + "\n" + text)
        else:
            out.append((role, text))
    return out


def render(ts):
    parts = [f"{role}: {text}" for role, text in ts]
    return parts


def dialogs(source, path):
    entry = next(f for f in source["files"] if f["path"] == path)
    data = raw(source, path)
    if git_blob(data) != entry["git_blob"]:
        sys.exit(f"{path}: git blob differs from SOURCES.toml")
    return json.loads(data), entry["git_blob"]


def item(category, source, locator, text, extra=None):
    body = text.encode("utf-8")
    it = {
        "id": f"{category}-{locator['key']}",
        "category": category,
        "source": source["id"],
        "licence": source["licence"],
        "locator": {k: v for k, v in locator.items() if k != "key"},
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
    }
    if extra:
        it.update(extra)
    return it, body


def select():
    sources = {s["id"]: s for s in tomllib.loads(SOURCES.read_text("utf-8"))["source"]}
    items = []

    tm1 = sources["taskmaster-1"]
    data, blob = dialogs(tm1, "TM-1-2019/self-dialogs.json")
    for d in data:
        ts = turns(d, "USER", "ASSISTANT")
        if len(ts) >= CHAT_MIN_TURNS:
            text = "\n\n".join(render(ts))
            loc = {"key": d["conversation_id"], "file": "TM-1-2019/self-dialogs.json",
                   "file_git_blob": blob, "dialog": d["conversation_id"]}
            items.append(item("chat", tm1, loc, text))
            if sum(i[0]["category"] == "chat" for i in items) == CHAT_ITEMS:
                break

    data, blob = dialogs(tm1, "TM-1-2019/woz-dialogs.json")
    n = 0
    for d in data:
        ts = turns(d, "USER", "ASSISTANT")
        if len(ts) >= AGENT_MIN_TURNS:
            parts = render(ts)
            # Each turn after the first is preceded by the paragraph break, so the text is the
            # turns' bytes in order and the turn lengths add up to the item's length.
            pieces = [parts[0]] + ["\n\n" + p for p in parts[1:]]
            text = "".join(pieces)
            loc = {"key": d["conversation_id"], "file": "TM-1-2019/woz-dialogs.json",
                   "file_git_blob": blob, "dialog": d["conversation_id"]}
            lens = [len(p.encode("utf-8")) for p in pieces]
            items.append(item("agent", tm1, loc, text, {"turn_bytes": lens}))
            n += 1
            if n == AGENT_ITEMS:
                break

    for sid, ext in CODE_EXT.items():
        src = sources[sid]
        first = src["include"][0]
        entries = subtree(src, first)
        cands = sorted(
            (e for e in entries
             if e["type"] == "blob" and e["path"].endswith(ext)
             and CODE_MIN <= e.get("size", 0) <= CODE_MAX),
            key=lambda e: e["path"],
        )
        if not cands:
            sys.exit(f"{sid}: no file matches the rule")
        e = cands[0]
        data = raw(src, e["path"])
        if git_blob(data) != e["sha"]:
            sys.exit(f"{sid}: {e['path']} git blob differs from the tree")
        text = data.decode("utf-8")
        loc = {"key": sid.removeprefix("code-"), "file": e["path"], "file_git_blob": e["sha"]}
        items.append(item("code", src, loc, text))
    return items


def toml_str(s):
    return json.dumps(s, ensure_ascii=False)


def write_manifest(items, path):
    lines = [
        "# SPDX-License-Identifier: MIT OR Apache-2.0",
        "#",
        "# bench-prompts-v1 (RFC-0061 §3.1; QUAL-E12-T05): the benchmark prompt set's items, written",
        "# by `select.py` from the parity corpus sources at their pinned revisions. The item text",
        "# is not in git: `select.py --out DIR` writes it to DIR/<sha256>, and the harness refuses",
        "# text whose SHA-256 differs. Licences are the sources' (testdata/corpus/SOURCES.toml); CC-BY",
        "# sources keep their attribution files under testdata/corpus/attribution/.",
        "",
        "schema = 1",
        'set = "bench-prompts-v1"',
        'sources = "testdata/corpus/SOURCES.toml"',
        '# Items of a category are joined with this separator when a prompt is built.',
        'separator = "\\n\\n"',
        "# The per-repetition nonce prepended in `Cold` scenarios, in tokens (RFC-0061 §3.1).",
        "nonce_tokens = 8",
        "",
    ]
    for it in items:
        lines.append("[[item]]")
        for k in ("id", "category", "source", "licence"):
            lines.append(f"{k} = {toml_str(it[k])}")
        loc = ", ".join(f"{k} = {toml_str(v)}" for k, v in it["locator"].items())
        lines.append(f"locator = {{ {loc} }}")
        lines.append(f"sha256 = {toml_str(it['sha256'])}")
        lines.append(f"bytes = {it['bytes']}")
        if "turn_bytes" in it:
            lines.append(f"turn_bytes = {it['turn_bytes']}")
        lines.append("")
    path.write_text("\n".join(lines), "utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", metavar="DIR", help="write each item's text to DIR/<sha256>")
    p.add_argument("--check", action="store_true", help="fail if the manifest would change")
    p.add_argument("--gh", action="store_true", help="list git trees through the GitHub CLI")
    args = p.parse_args()
    global USE_GH
    USE_GH = args.gh
    selected = select()
    items = [i for i, _ in selected]
    target = HERE / "bench-prompts-v1.toml"
    if args.check:
        tmp = HERE / ".check.toml"
        write_manifest(items, tmp)
        same = tmp.read_bytes() == target.read_bytes()
        tmp.unlink()
        if not same:
            sys.exit("bench-prompts-v1.toml differs from what select.py writes")
    else:
        write_manifest(items, target)
    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for it, body in selected:
            (out / it["sha256"]).write_bytes(body)
    counts = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    print(f"{len(items)} items: {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
