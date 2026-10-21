#!/usr/bin/env python3
"""Check text against the D5 banned-terms list (AGENTS.md, "Naming (D5)").

The list names the third-party projects that code, comments, docs and plan
text must not name. It is kept outside the repository, so this script never
prints it, and never prints the text a term matched. A finding is only a
location: `path:line` for a file, `commit message:line` for a message.

    python3 naming_check.py --staged              # files staged for commit (the pre-commit hook)
    python3 naming_check.py --base origin/main    # files this branch changed, as committed at HEAD
    python3 naming_check.py --message FILE        # a commit message file (the commit-msg hook)
    python3 naming_check.py --stdin [--label L]   # text on standard input
    python3 naming_check.py PATH...               # files on disk

The list is the first of these that is set: --banned FILE, the
SYNTROPY_BANNED_TERMS environment variable, `git config syntropy.bannedTerms`.
If none is set, the check is skipped with one notice line and exits 0, unless
--require is given. A list that is set but cannot be read is an error. The list
is read by gh_sync.load_banned, as gh_sync.py --banned reads it: one
case-insensitive regular expression per non-blank line, matched against each
line of text.

Whole files are checked, not only changed lines, as AGENTS.md asks of every
changed file. A file containing a NUL byte is treated as binary and skipped.
In a message file, comment lines are skipped and everything below git's
scissors line is ignored, so `git commit -v` diffs are not checked twice.
Lines may end in LF or CRLF, as an editor on Windows saves them.

Exit status: 0 when nothing matches or the check is skipped, 1 when something
matches, 2 on an error.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gh_sync import load_banned  # noqa: E402

NAME = "naming check"
ENV = "SYNTROPY_BANNED_TERMS"
GIT_KEY = "syntropy.bannedTerms"
# git's scissors line, after the comment character: everything below it is ignored.
SCISSORS = re.compile(r"^. -{24} >8 -{24}$")

PASS, MATCH, ERROR = 0, 1, 2


class CheckError(Exception):
    """A problem that stops the check: exit status 2."""


def git(*args, stdin=None):
    """`git <args>`'s standard output, as bytes. Raises CheckError on failure."""
    try:
        done = subprocess.run(["git", *args], input=stdin, capture_output=True, check=False)
    except FileNotFoundError:
        raise CheckError("git is not on PATH") from None
    if done.returncode != 0:
        detail = done.stderr.decode(errors="replace").strip().splitlines()
        raise CheckError(f"`git {' '.join(args)}` failed: {detail[-1] if detail else done.returncode}")
    return done.stdout


def list_path(banned):
    """Where the list is, and which setting named it; None when nothing is set."""
    if banned:
        return Path(banned), "--banned"
    if os.environ.get(ENV):
        return Path(os.environ[ENV]), ENV
    try:
        configured = git("config", "--path", "--get", GIT_KEY).decode().strip()
    except CheckError:
        configured = ""  # git exits 1 when the key is unset
    if configured:
        return Path(configured), f"git config {GIT_KEY}"
    return None


def matching_lines(text, patterns, skip=None):
    """The 1-based numbers of the lines of `text` that a pattern matches.
    `skip(line)` returning "skip" leaves a line out; returning "stop" ends the
    scan there."""
    found = []
    # Split on "\n" only, so line numbers agree with grep and editors, and drop the "\r" of a
    # CRLF line ending, so a line reads the same however the file was saved.
    for number, line in enumerate(text.split("\n"), 1):
        if line.endswith("\r"):
            line = line[:-1]
        verdict = skip(line) if skip else None
        if verdict == "stop":
            break
        if verdict == "skip":
            continue
        if any(p.search(line) for p in patterns):
            found.append(number)
    return found


def decode(content):
    """`content` as text, or None if it looks binary."""
    if b"\0" in content:
        return None
    return content.decode("utf-8", errors="replace")


def changed_paths(*diff_args):
    """Paths `git diff --name-only` lists for `diff_args`, deleted files left out."""
    out = git("diff", "--name-only", "-z", "--diff-filter=d", *diff_args)
    return [p for p in out.decode("utf-8", errors="surrogateescape").split("\0") if p]


def blobs(revision, paths):
    """(path, content) for each path at `revision`, where `:` is the index,
    read through one `git cat-file --batch`. Entries that are not blobs, such
    as submodules, are skipped."""
    for path in paths:
        if "\n" in path:
            raise CheckError(f"cannot check a path that contains a newline: {path!r}")
    request = "".join(f"{revision}{path}\n" for path in paths).encode("utf-8", "surrogateescape")
    out = git("cat-file", "--batch", stdin=request)
    at = 0
    for path in paths:
        header_end = out.index(b"\n", at)
        header = out[at:header_end].split()
        at = header_end + 1
        if header[-1] == b"missing":
            continue
        kind, size = header[1], int(header[2])
        content = out[at:at + size]
        at += size + 1  # the content is followed by a newline
        if kind == b"blob":
            yield path, content


def sources(args):
    """(label, text, skip) for everything to check, and how many were binary."""
    if args.staged:
        items = blobs(":", changed_paths("--cached"))
    elif args.base:
        items = blobs("HEAD:", changed_paths(f"{args.base}...HEAD"))
    elif args.message:
        return [message_source(args.message)], 0
    elif args.stdin:
        return [(args.label or "stdin", sys.stdin.buffer.read().decode("utf-8", errors="replace"), None)], 0
    else:
        items = []
        for path in args.paths:
            try:
                items.append((path, Path(path).read_bytes()))
            except OSError as error:
                raise CheckError(f"cannot read {path}: {error.strerror}") from None
    texts, binary = [], 0
    for label, content in items:
        text = decode(content)
        if text is None:
            binary += 1
        else:
            texts.append((label, text, None))
    return texts, binary


def message_source(path):
    """A commit message file, with git's comment lines skipped and the scissors honoured."""
    try:
        text = Path(path).read_bytes().decode("utf-8", errors="replace")
    except OSError as error:
        raise CheckError(f"cannot read the commit message {path}: {error.strerror}") from None
    try:
        comment = git("config", "--get", "core.commentChar").decode().strip() or "#"
    except CheckError:
        comment = "#"
    if comment == "auto":  # git picks the character per message; assume the default
        comment = "#"

    def skip(line):
        if SCISSORS.match(line):
            return "stop"
        if line.startswith(comment):
            return "skip"
        return None

    return "commit message", text, skip


def run(argv):
    parser = argparse.ArgumentParser(
        description="Check text against the D5 banned-terms list, printing locations only.")
    parser.add_argument("--banned", help="the banned-terms list (else $%s, else git config %s)" % (ENV, GIT_KEY))
    parser.add_argument("--require", action="store_true",
                        help="fail instead of skipping when no list is set")
    what = parser.add_mutually_exclusive_group()
    what.add_argument("--staged", action="store_true", help="the files staged for commit")
    what.add_argument("--base", metavar="REF", help="the files changed in REF...HEAD, as at HEAD")
    what.add_argument("--message", metavar="FILE", help="a commit message file")
    what.add_argument("--stdin", action="store_true", help="text on standard input")
    parser.add_argument("--label", help="the name findings give for --stdin text")
    parser.add_argument("paths", nargs="*", help="files on disk")
    args = parser.parse_args(argv)
    if args.paths and (args.staged or args.base or args.message or args.stdin):
        parser.error("give paths, or one of --staged, --base, --message and --stdin, not both")
    if not (args.paths or args.staged or args.base or args.message or args.stdin):
        parser.error("nothing to check: give paths, --staged, --base REF, --message FILE or --stdin")

    located = list_path(args.banned)
    if located is None:
        if args.require:
            raise CheckError(f"no banned-terms list: give --banned, set {ENV} or git config {GIT_KEY}")
        print(f"{NAME}: skipped, no banned-terms list is set ({ENV} or git config {GIT_KEY})")
        return PASS
    path, where = located
    try:
        patterns = load_banned(path)
    except OSError as error:
        raise CheckError(f"cannot read the banned-terms list named by {where} ({path}): {error.strerror}") from None
    except re.error:
        raise CheckError(f"the banned-terms list named by {where} has a line that is not a valid regular expression") from None

    texts, binary = sources(args)
    if not texts and not binary:
        print(f"{NAME}: skipped, nothing to check")
        return PASS
    findings = 0
    files = 0
    for label, text, skip in texts:
        lines = matching_lines(text, patterns, skip)
        if lines:
            files += 1
            findings += len(lines)
            for number in lines:
                print(f"{label}:{number}: matches a banned term (D5)")
    skipped = f", {binary} binary skipped" if binary else ""
    if findings:
        print(f"{NAME}: FAIL: {findings} line(s) in {files} of {len(texts)} checked match the "
              f"banned-terms list{skipped}; the terms are not printed")
        return MATCH
    print(f"{NAME}: PASS: {len(texts)} checked{skipped}")
    return PASS


def main(argv=None):
    try:
        return run(sys.argv[1:] if argv is None else argv)
    except CheckError as error:
        print(f"{NAME}: error: {error}", file=sys.stderr)
        return ERROR


if __name__ == "__main__":
    sys.exit(main())
