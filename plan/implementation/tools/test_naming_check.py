#!/usr/bin/env python3
"""Tests for naming_check.py, and for the commit-msg hook that runs it.

    python3 plan/implementation/tools/test_naming_check.py

Each test writes its own throwaway banned-terms list. None reads the real list,
which is kept outside the repository. Every test that runs the script asserts
that its output never contains a term or a pattern from the list.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import naming_check  # noqa: E402
from gh_sync import load_banned  # noqa: E402

SCRIPT = TOOLS / "naming_check.py"
REPO = TOOLS.parent.parent.parent
# Throwaway terms that mean nothing; the second is a pattern with a character class.
TERMS = ["zorblax", "quux[- ]?frob"]
SECRET_TEXT = ["zorblax", "quux", "frob"]


class NamingCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.banned = self.dir / "terms.txt"
        self.banned.write_text("\n".join(TERMS) + "\n\n", encoding="utf-8")
        # git config that ignores the user's own settings, so syntropy.bannedTerms is unset
        # unless a test sets it.
        empty = self.dir / "empty.gitconfig"
        empty.write_text("", encoding="utf-8")
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith("GIT_") and k != naming_check.ENV}
        self.env.update({
            "GIT_CONFIG_GLOBAL": str(empty),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        })

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, *args, stdin="", cwd=None, env=None):
        done = subprocess.run(
            [sys.executable, str(SCRIPT), *args], input=stdin.encode(), capture_output=True,
            cwd=cwd or self.dir, env=env or self.env, check=False)
        output = (done.stdout + done.stderr).decode()
        for secret in SECRET_TEXT + TERMS:
            self.assertNotIn(secret.lower(), output.lower(), "the output names a term")
        return done.returncode, output

    def git(self, repo, *args):
        subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, env=self.env,
                       check=True, capture_output=True)

    def repo(self):
        repo = self.dir / "repo"
        repo.mkdir()
        self.git(repo, "init", "-q", "-b", "main")
        (repo / "base.txt").write_text("clean\n", encoding="utf-8")
        self.git(repo, "add", "base.txt")
        self.git(repo, "commit", "-q", "-m", "base")
        return repo

    # The list and the matcher

    def test_the_list_is_read_by_gh_syncs_loader(self):
        patterns = load_banned(self.banned)
        self.assertEqual(len(patterns), 2, "blank lines are not patterns")
        self.assertTrue(patterns[0].search("a ZorBlax b"), "matching ignores case")
        self.assertTrue(patterns[1].search("quux-frob"))
        self.assertTrue(patterns[1].search("quuxfrob"))

    def test_matching_lines_numbers_lines_from_one_and_splits_on_newlines_only(self):
        patterns = load_banned(self.banned)
        text = "fine\nzorblax\x0cstill line two\nfine\nQUUX FROB\n"
        self.assertEqual(naming_check.matching_lines(text, patterns), [2, 4])

    def test_matching_lines_reads_crlf_lines_as_lf_lines(self):
        anchored = self.dir / "anchored.txt"
        anchored.write_text("^zorblax$\n", encoding="utf-8")
        patterns = load_banned(anchored)
        self.assertEqual(naming_check.matching_lines("fine\r\nzorblax\r\n", patterns), [2])
        self.assertEqual(naming_check.matching_lines("fine\nzorblax\n", patterns), [2])

    # What the script reports

    def test_a_match_on_stdin_is_reported_by_label_and_line(self):
        code, output = self.run_check("--banned", str(self.banned), "--stdin", "--label", "note",
                                      stdin="one\nzorblax two\nthree\nquux frob\n")
        self.assertEqual(code, 1)
        self.assertIn("note:2: matches a banned term (D5)", output)
        self.assertIn("note:4: matches a banned term (D5)", output)
        self.assertIn("naming check: FAIL: 2 line(s)", output)

    def test_clean_text_passes(self):
        code, output = self.run_check("--banned", str(self.banned), "--stdin", stdin="all clean\n")
        self.assertEqual(code, 0)
        self.assertIn("naming check: PASS: 1 checked", output)

    def test_files_on_disk_are_reported_by_path_and_binary_files_are_skipped(self):
        (self.dir / "a.md").write_text("ok\nok\nzorblax\n", encoding="utf-8")
        (self.dir / "b.md").write_text("ok\n", encoding="utf-8")
        (self.dir / "c.bin").write_bytes(b"\0zorblax")
        code, output = self.run_check("--banned", str(self.banned), "a.md", "b.md", "c.bin")
        self.assertEqual(code, 1)
        self.assertIn("a.md:3: matches", output)
        self.assertNotIn("b.md:", output)
        self.assertNotIn("c.bin:", output)
        self.assertIn("1 binary skipped", output)

    def check_message(self, newline):
        """A message with a term in its body, in a comment and below the scissors line, saved
        with `newline` endings: only the body's line is reported."""
        message = self.dir / "COMMIT_EDITMSG"
        message.write_bytes(newline.join([
            "feat: a thing", "", "quux frob in the body", "# zorblax in a comment",
            "# ------------------------ >8 ------------------------",
            "+zorblax in a diff below the scissors", "",
        ]).encode("utf-8"))
        code, output = self.run_check("--banned", str(self.banned), "--message", str(message))
        self.assertEqual(code, 1)
        self.assertEqual(output.count("matches a banned term"), 1, output)
        self.assertIn("commit message:3: matches", output)

    def test_a_commit_message_skips_comments_and_stops_at_the_scissors(self):
        self.check_message("\n")

    def test_a_crlf_commit_message_skips_comments_and_stops_at_the_scissors(self):
        # An editor on Windows can save the message file with CRLF endings.
        self.check_message("\r\n")

    def test_staged_files_are_read_from_the_index(self):
        repo = self.repo()
        (repo / "doc.md").write_text("line\nzorblax\n", encoding="utf-8")
        (repo / "clean.md").write_text("fine\n", encoding="utf-8")
        self.git(repo, "add", "doc.md", "clean.md")
        # The working tree no longer has the term; the index, which is committed, still does.
        (repo / "doc.md").write_text("line\nfine now\n", encoding="utf-8")
        code, output = self.run_check("--banned", str(self.banned), "--staged", cwd=repo)
        self.assertEqual(code, 1)
        self.assertIn("doc.md:2: matches", output)
        self.assertIn("1 of 2 checked", output)

    def test_files_changed_against_a_base_are_read_at_head(self):
        repo = self.repo()
        self.git(repo, "switch", "-q", "-c", "topic")
        (repo / "sub").mkdir()
        (repo / "sub" / "new file.md").write_text("zorblax\n", encoding="utf-8")
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-q", "-m", "add")
        code, output = self.run_check("--banned", str(self.banned), "--base", "main", cwd=repo)
        self.assertEqual(code, 1)
        self.assertIn("sub/new file.md:1: matches", output)

    def test_nothing_staged_is_skipped_with_a_notice(self):
        repo = self.repo()
        code, output = self.run_check("--banned", str(self.banned), "--staged", cwd=repo)
        self.assertEqual(code, 0)
        self.assertIn("naming check: skipped, nothing to check", output)

    # Where the list comes from

    def test_without_a_list_the_check_is_skipped_with_one_notice(self):
        code, output = self.run_check("--stdin", stdin="zorblax\n")
        self.assertEqual(code, 0)
        self.assertEqual(output.strip().count("\n"), 0, "one line")
        self.assertIn("naming check: skipped, no banned-terms list is set", output)

    def test_require_turns_a_missing_setting_into_an_error(self):
        code, output = self.run_check("--require", "--stdin", stdin="text\n")
        self.assertEqual(code, 2)
        self.assertIn("no banned-terms list", output)

    def test_the_environment_variable_names_the_list(self):
        env = dict(self.env, **{naming_check.ENV: str(self.banned)})
        code, output = self.run_check("--stdin", stdin="zorblax\n", env=env)
        self.assertEqual(code, 1)
        self.assertIn("stdin:1: matches", output)

    def test_git_config_names_the_list(self):
        repo = self.repo()
        self.git(repo, "config", naming_check.GIT_KEY, str(self.banned))
        code, output = self.run_check("--stdin", stdin="zorblax\n", cwd=repo)
        self.assertEqual(code, 1)

    def test_banned_comes_before_the_environment_before_git_config(self):
        repo = self.repo()
        clean_list = self.dir / "other.txt"
        clean_list.write_text("nothinglikethis\n", encoding="utf-8")
        self.git(repo, "config", naming_check.GIT_KEY, str(clean_list))
        env = dict(self.env, **{naming_check.ENV: str(self.banned)})
        code, _ = self.run_check("--stdin", stdin="zorblax\n", cwd=repo, env=env)
        self.assertEqual(code, 1, "the environment beats git config")
        code, _ = self.run_check("--banned", str(clean_list), "--stdin", stdin="zorblax\n",
                                 cwd=repo, env=env)
        self.assertEqual(code, 0, "--banned beats the environment")

    def test_a_list_that_is_set_but_missing_is_an_error(self):
        code, output = self.run_check("--banned", str(self.dir / "nope.txt"), "--stdin",
                                      stdin="text\n")
        self.assertEqual(code, 2)
        self.assertIn("cannot read the banned-terms list named by --banned", output)

    def test_an_invalid_pattern_is_an_error_that_does_not_print_it(self):
        bad = self.dir / "bad.txt"
        bad.write_text("zorblax[\n", encoding="utf-8")
        code, output = self.run_check("--banned", str(bad), "--stdin", stdin="text\n")
        self.assertEqual(code, 2)
        self.assertIn("not a valid regular expression", output)

    def test_usage_errors_exit_two(self):
        code, _ = self.run_check("--banned", str(self.banned))
        self.assertEqual(code, 2)
        code, _ = self.run_check("--banned", str(self.banned), "--stdin", "a.md")
        self.assertEqual(code, 2)


@unittest.skipUnless(shutil.which("sh"), "needs a POSIX sh on PATH")
class HookTest(unittest.TestCase):
    """The hooks in .githooks/: commit-msg on messages saved with CRLF endings, as an editor on
    Windows may save them, and both hooks run for a repository that is not this one, as when
    core.hooksPath points at a .githooks elsewhere. They run with a throwaway list and author."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        terms = self.dir / "terms.txt"
        terms.write_text("zorblax\n", encoding="utf-8")
        empty = self.dir / "empty.gitconfig"
        empty.write_text("", encoding="utf-8")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update({
            naming_check.ENV: str(terms),
            "GIT_CONFIG_GLOBAL": str(empty),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
        })

    def tearDown(self):
        self.tmp.cleanup()

    def run_hook(self, lines, cwd=REPO, hook="commit-msg"):
        message = self.dir / "COMMIT_EDITMSG"
        message.write_bytes("\r\n".join(lines + [""]).encode("utf-8"))
        args = [message.as_posix()] if hook == "commit-msg" else []
        done = subprocess.run(
            ["sh", (REPO / ".githooks" / hook).as_posix(), *args],
            cwd=cwd, env=self.env, capture_output=True, check=False)
        output = (done.stdout + done.stderr).decode()
        self.assertNotIn("zorblax", output.lower(), "the output names a term")
        return done.returncode, output

    def other_repo(self):
        """A repository with no .githooks and no plan tools, with one file staged."""
        repo = self.dir / "other"
        repo.mkdir()
        git = ["git", "-c", "init.defaultBranch=main"]
        subprocess.run([*git, "init", "-q"], cwd=repo, env=self.env, check=True,
                       capture_output=True)
        (repo / "readme.txt").write_text("hello\n", encoding="utf-8")
        subprocess.run([*git, "add", "readme.txt"], cwd=repo, env=self.env, check=True,
                       capture_output=True)
        return repo

    def test_commit_msg_runs_for_a_repository_without_the_hooks_or_tools(self):
        code, output = self.run_hook(
            ["docs: a message", "", "Signed-off-by: Test <test@example.invalid>"],
            cwd=self.other_repo())
        self.assertEqual(code, 0, output)
        self.assertIn("commit-msg: subject: ok", output)
        self.assertIn("commit-msg: sign-off: ok", output)
        self.assertIn("naming check: skipped, this repository has no", output)

    def test_pre_commit_runs_for_a_repository_without_the_hooks_or_tools(self):
        code, output = self.run_hook([], cwd=self.other_repo(), hook="pre-commit")
        self.assertEqual(code, 0, output)
        self.assertIn("pre-commit: rustfmt: skipped, no Rust files staged", output)
        self.assertIn("naming check: skipped, this repository has no", output)

    def message(self, *body):
        return ["docs: a message saved with CRLF endings", "", *body,
                "# a comment that says zorblax",
                "# ------------------------ >8 ------------------------",
                "diff --git a/x b/x", "+zorblax below the scissors"]

    def test_the_scissors_comments_and_sign_off_are_read_through_crlf(self):
        code, output = self.run_hook(self.message("Signed-off-by: Test <test@example.invalid>"))
        self.assertEqual(code, 0, output)
        self.assertIn("commit-msg: subject: ok", output)
        self.assertIn("commit-msg: sign-off: ok", output)
        self.assertIn("naming check: PASS", output)

    def test_an_empty_crlf_message_is_left_to_git(self):
        # Only comments, the scissors line and the diff: the hook must drop all of it, and leave
        # git to abort the empty message, instead of reading the diff as a subject.
        code, output = self.run_hook([
            "# Please enter the commit message for your changes.",
            "# ------------------------ >8 ------------------------",
            "diff --git a/x b/x", "+a line",
        ])
        self.assertEqual(code, 0, output)
        self.assertEqual(output, "")

    def test_a_crlf_message_still_needs_a_sign_off_and_a_clean_body(self):
        code, output = self.run_hook(self.message("The body says zorblax."))
        self.assertEqual(code, 1, output)
        self.assertIn("commit-msg: sign-off: FAIL", output)
        self.assertIn("commit message:3: matches a banned term", output)


if __name__ == "__main__":
    unittest.main()
