#!/usr/bin/env python3
"""Tests for depcheck.py's COUNT problem: the open-question register's stated count.

    python3 plan/implementation/tools/test_depcheck.py

The fixture registers are written to a temporary directory. The repository's own register
is read, never written.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import depcheck  # noqa: E402

SCRIPT = TOOLS / "depcheck.py"


def register(stated, open_rows, settled_rows=()):
    """A minimal register: the intro sentence, an open table and a Settled table."""
    lines = ["# Open questions", "",
             f"This register lists the questions. {stated} questions are open.", "",
             "## Process, lab and release", "",
             "| ID | Question | Why it matters | RFC | Resolution | Task | Needed by |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {oq} | q | w | RFC-0001 | decision | — | M0a |" for oq in open_rows]
    lines += ["", "## Settled", "",
              "| ID | Question | Raised in | Settled by | Answer | Still open |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {oq} | q | this register | RFC-0001 | a | — |" for oq in settled_rows]
    return "\n".join(lines) + "\n"


class RegisterCount(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "open-questions.md"

    def tearDown(self):
        self.tmp.cleanup()

    def problems(self, text):
        self.path.write_text(text, encoding="utf-8")
        return depcheck.register_problems(self.path)

    def test_repository_register_is_consistent(self):
        self.assertEqual(depcheck.register_problems(depcheck.REGISTER), [])

    def test_matching_count_passes(self):
        self.assertEqual(self.problems(register(2, ["OQ-PRC-01", "OQ-PRC-02"], ["OQ-PRC-03"])), [])

    def test_count_off_by_one_names_both_numbers(self):
        probs = self.problems(register(3, ["OQ-PRC-01", "OQ-PRC-02"]))
        self.assertEqual(len(probs), 1)
        self.assertEqual(probs[0]["kind"], "COUNT")
        self.assertIn("says 3", probs[0]["detail"])
        self.assertIn("but 2", probs[0]["detail"])
        self.assertEqual(probs[0]["line"], 3)

    def test_settled_rows_do_not_count(self):
        probs = self.problems(register(3, ["OQ-PRC-01", "OQ-PRC-02"], ["OQ-PRC-03"]))
        self.assertEqual([p["kind"] for p in probs], ["COUNT"])

    def test_open_id_listed_twice(self):
        probs = self.problems(register(3, ["OQ-PRC-01", "OQ-PRC-02", "OQ-PRC-01"]))
        self.assertEqual(len(probs), 1)
        self.assertEqual(probs[0]["task"], "OQ-PRC-01")
        self.assertIn("listed twice", probs[0]["detail"])

    def test_open_id_also_settled(self):
        probs = self.problems(register(2, ["OQ-PRC-01", "OQ-PRC-02"], ["OQ-PRC-02"]))
        self.assertEqual(len(probs), 1)
        self.assertEqual(probs[0]["task"], "OQ-PRC-02")
        self.assertIn("also listed under Settled", probs[0]["detail"])

    def test_missing_sentence(self):
        text = register(1, ["OQ-PRC-01"]).replace("1 questions are open", "some are open")
        probs = self.problems(text)
        self.assertEqual(len(probs), 1)
        self.assertIn("no \"N questions are open\" sentence", probs[0]["detail"])

    def test_cli_exits_1_and_lists_count(self):
        self.path.write_text(register(5, ["OQ-PRC-01"]), encoding="utf-8")
        run = subprocess.run([sys.executable, str(SCRIPT), "--register", str(self.path)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn("COUNT: open-questions.md:3", run.stdout)
        self.assertIn("COUNT: 1", run.stdout)

    def test_cli_exits_0_on_a_consistent_register(self):
        self.path.write_text(register(1, ["OQ-PRC-01"]), encoding="utf-8")
        run = subprocess.run([sys.executable, str(SCRIPT), "--quiet", "--register", str(self.path)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("COUNT: 0", run.stdout)


if __name__ == "__main__":
    unittest.main()
