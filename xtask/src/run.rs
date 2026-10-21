// SPDX-License-Identifier: MIT OR Apache-2.0

//! Running a job or suite and reaching its verdict.

use std::io::{self, Write};
use std::path::Path;
use std::process::Command;

use crate::cli::Kind;
use crate::registry::{Entry, Step};

/// Runs one step. The real implementation is [`System`]; tests substitute a fake.
pub trait Exec {
    /// Runs `step` with `root` as the working directory. `Err` carries why it did not pass.
    fn exec(&mut self, step: &Step, root: &Path) -> Result<(), String>;
}

/// Runs each step as a child process that inherits stdout and stderr.
pub struct System;

impl Exec for System {
    fn exec(&mut self, step: &Step, root: &Path) -> Result<(), String> {
        let mut command = step.action.command();
        command.current_dir(root);
        outcome(&mut command, &step.action.display())
    }
}

/// Runs `command` to completion. A zero exit status passes; anything else, including a program
/// that cannot start, fails with the reason. `shown` is the command as progress lines print it.
fn outcome(command: &mut Command, shown: &str) -> Result<(), String> {
    match command.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(status.to_string()),
        Err(error) => Err(format!("could not start `{shown}`: {error}")),
    }
}

/// The outcome of a job or suite.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    /// The names of the steps that failed, in order.
    Fail(Vec<&'static str>),
}

/// Runs every step of `entry` from `root` and returns the verdict. A failed step does not stop
/// the steps after it. Progress lines and the verdict go to `log`.
pub fn run(
    kind: Kind,
    entry: &Entry,
    root: &Path,
    exec: &mut dyn Exec,
    log: &mut dyn Write,
) -> io::Result<Verdict> {
    let label = format!("xtask {} {}", kind.command(), entry.name);
    let total = entry.steps.len();
    let mut failed = Vec::new();
    for (index, step) in entry.steps.iter().enumerate() {
        writeln!(
            log,
            "{label}: [{}/{total}] {}: {}",
            index + 1,
            step.name,
            step.action.display()
        )?;
        // Flush before the child writes, so the lines appear in order.
        log.flush()?;
        match exec.exec(step, root) {
            Ok(()) => writeln!(log, "{label}: {} passed", step.name)?,
            Err(reason) => {
                writeln!(log, "{label}: {} failed: {reason}", step.name)?;
                failed.push(step.name);
            }
        }
    }
    let ran: Vec<&str> = entry.steps.iter().map(|s| s.name).collect();
    if failed.is_empty() {
        writeln!(log, "{label}: PASS ({})", ran.join(", "))?;
    } else {
        writeln!(
            log,
            "{label}: FAIL ({} of {total} failed: {})",
            failed.len(),
            failed.join(", ")
        )?;
    }
    if !entry.pending.is_empty() {
        writeln!(log, "{label}: not in this {} yet:", kind.noun())?;
        for pending in entry.pending {
            writeln!(log, "  {} ({})", pending.check, pending.task)?;
        }
    }
    log.flush()?;
    Ok(if failed.is_empty() {
        Verdict::Pass
    } else {
        Verdict::Fail(failed)
    })
}

/// Writes one line per entry, its name and summary, for `--list`.
pub fn list(entries: &[Entry], out: &mut dyn Write) -> io::Result<()> {
    let width = entries.iter().map(|e| e.name.len()).max().unwrap_or(0);
    for entry in entries {
        writeln!(out, "{:width$}  {}", entry.name, entry.summary)?;
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Action, JOBS, Pending, find};

    /// Records each step it is asked to run, and fails the steps named in `fail`.
    struct Fake {
        fail: &'static [&'static str],
        ran: Vec<&'static str>,
    }

    impl Exec for Fake {
        fn exec(&mut self, step: &Step, _root: &Path) -> Result<(), String> {
            self.ran.push(step.name);
            if self.fail.contains(&step.name) {
                Err("exit status: 1".into())
            } else {
                Ok(())
            }
        }
    }

    const THREE: Entry = Entry {
        name: "three",
        summary: "three steps",
        steps: &[
            Step {
                name: "one",
                action: Action::Cargo(&["one"]),
            },
            Step {
                name: "two",
                action: Action::Cargo(&["two"]),
            },
            Step {
                name: "three",
                action: Action::Cargo(&["three"]),
            },
        ],
        pending: &[Pending {
            check: "a later check",
            task: "QUAL-E1-T99",
        }],
    };

    fn run_fake(
        entry: &Entry,
        fail: &'static [&'static str],
    ) -> (Verdict, Vec<&'static str>, String) {
        let mut fake = Fake {
            fail,
            ran: Vec::new(),
        };
        let mut log = Vec::new();
        let verdict = run(Kind::Job, entry, Path::new("."), &mut fake, &mut log).unwrap();
        (verdict, fake.ran, String::from_utf8(log).unwrap())
    }

    #[test]
    fn every_step_passing_is_a_pass() {
        let (verdict, ran, log) = run_fake(&THREE, &[]);
        assert_eq!(verdict, Verdict::Pass);
        assert_eq!(ran, ["one", "two", "three"]);
        assert!(
            log.contains("xtask ci three: [1/3] one: cargo one\n"),
            "{log}"
        );
        assert!(
            log.contains("xtask ci three: PASS (one, two, three)\n"),
            "{log}"
        );
    }

    #[test]
    fn a_failed_step_fails_the_job_and_the_rest_still_run() {
        let (verdict, ran, log) = run_fake(&THREE, &["two"]);
        assert_eq!(verdict, Verdict::Fail(vec!["two"]));
        assert_eq!(ran, ["one", "two", "three"]);
        assert!(
            log.contains("xtask ci three: two failed: exit status: 1\n"),
            "{log}"
        );
        assert!(
            log.contains("xtask ci three: FAIL (1 of 3 failed: two)\n"),
            "{log}"
        );
    }

    #[test]
    fn every_failure_is_named() {
        let (verdict, _, log) = run_fake(&THREE, &["one", "three"]);
        assert_eq!(verdict, Verdict::Fail(vec!["one", "three"]));
        assert!(log.contains("FAIL (2 of 3 failed: one, three)"), "{log}");
    }

    #[test]
    fn pending_checks_follow_the_verdict() {
        let (_, _, log) = run_fake(&THREE, &[]);
        let verdict_at = log.find("PASS").unwrap();
        let pending_at = log.find("not in this job yet:").unwrap();
        assert!(verdict_at < pending_at, "{log}");
        assert!(log.contains("  a later check (QUAL-E1-T99)\n"), "{log}");
    }

    #[test]
    fn lint_reports_its_pending_checks() {
        let lint = find(JOBS, "lint").unwrap();
        let (verdict, ran, log) = run_fake(lint, &[]);
        assert_eq!(verdict, Verdict::Pass);
        assert_eq!(ran, ["fmt", "clippy"]);
        for pending in lint.pending {
            assert!(log.contains(pending.task), "{log}");
        }
    }

    #[test]
    fn list_aligns_names_and_summaries() {
        let mut out = Vec::new();
        list(&[THREE], &mut out).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "three  three steps\n");
    }

    #[test]
    fn system_passes_a_step_that_exits_zero() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let version = Step {
            name: "version",
            action: Action::Cargo(&["--version"]),
        };
        assert_eq!(System.exec(&version, root), Ok(()));
    }

    #[test]
    fn a_non_zero_exit_fails_with_its_status() {
        let mut command = Action::Cargo(&["syntropy-xtask-no-such-subcommand"]).command();
        command.stderr(std::process::Stdio::null());
        let reason = outcome(&mut command, "cargo syntropy-xtask-no-such-subcommand").unwrap_err();
        assert!(reason.contains("exit status"), "{reason}");
    }

    #[test]
    fn a_program_that_cannot_start_fails_with_a_reason() {
        let mut command = Command::new("syntropy-xtask-no-such-program");
        let reason = outcome(&mut command, "syntropy-xtask-no-such-program").unwrap_err();
        assert!(
            reason.starts_with("could not start `syntropy-xtask-no-such-program`: "),
            "{reason}"
        );
    }
}
