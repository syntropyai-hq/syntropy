// SPDX-License-Identifier: MIT OR Apache-2.0

//! Running a job or suite and reaching its verdict.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;

use crate::cli::{Args, Kind};
use crate::registry::{Action, Entry, Step};

/// What every step of one run sees.
#[derive(Debug)]
pub struct Context {
    /// The workspace root of the checkout xtask was started from. Every step runs there.
    pub root: PathBuf,
    /// The entry's option values from the command line.
    pub args: Args,
}

/// Runs one step. The real implementation is [`System`]; tests substitute a fake.
pub trait Exec {
    /// Runs `step` in `context`. `Err` carries why it did not pass.
    fn exec(&mut self, step: &Step, context: &Context) -> Result<(), String>;
}

/// Runs a `cargo` step as a child process from the workspace root, inheriting stdout and
/// stderr, and a builtin step inside this process.
pub struct System;

impl Exec for System {
    fn exec(&mut self, step: &Step, context: &Context) -> Result<(), String> {
        match (&step.action, step.action.command()) {
            (Action::Builtin { run, .. }, _) => run(context),
            (_, Some(mut command)) => {
                command.current_dir(&context.root);
                outcome(&mut command, &step.action.display())
            }
            (_, None) => Err(format!("`{}` starts no process", step.action.display())),
        }
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

/// Runs every step of `entry` in `context` and returns the verdict. A failed step does not stop
/// the steps after it. Progress lines and the verdict go to `log`.
pub fn run(
    kind: Kind,
    entry: &Entry,
    context: &Context,
    exec: &mut dyn Exec,
    log: &mut dyn Write,
) -> io::Result<Verdict> {
    let label = format!("xtask {} {}{}", kind.command(), entry.name, context.args);
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
        match exec.exec(step, context) {
            Ok(()) => writeln!(log, "{label}: {} passed", step.name)?,
            Err(reason) if reason.contains('\n') => {
                // A check that found several problems: one indented line each.
                writeln!(log, "{label}: {} failed:", step.name)?;
                for line in reason.lines() {
                    writeln!(log, "  {line}")?;
                }
                failed.push(step.name);
            }
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
    // `pending` covers what `defined_in` names, not every check later tasks add.
    if !entry.pending.is_empty() {
        writeln!(
            log,
            "{label}: {} checks not in this {} yet (other task rows add more):",
            entry.defined_in,
            kind.noun()
        )?;
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

/// Writes each entry's name, summary and definition for `--list`, then a line per option.
pub fn list(entries: &[Entry], out: &mut dyn Write) -> io::Result<()> {
    let width = entries.iter().map(|e| e.name.len()).max().unwrap_or(0);
    for entry in entries {
        writeln!(
            out,
            "{:width$}  {} ({})",
            entry.name, entry.summary, entry.defined_in
        )?;
        for param in entry.params {
            writeln!(
                out,
                "{:width$}    {} {}  {}",
                "", param.name, param.value, param.help
            )?;
        }
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{self, Command as Cli};
    use crate::registry::{JOBS, Param, Pending, find};

    /// Records each step it is asked to run, and fails the steps named in `fail`.
    struct Fake {
        fail: &'static [&'static str],
        ran: Vec<&'static str>,
    }

    impl Exec for Fake {
        fn exec(&mut self, step: &Step, _context: &Context) -> Result<(), String> {
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
        defined_in: "RFC-0062 §2.3",
        params: &[],
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
        let verdict = run(Kind::Job, entry, &here(), &mut fake, &mut log).unwrap();
        (verdict, fake.ran, String::from_utf8(log).unwrap())
    }

    /// A context rooted at this crate's directory, with no options.
    fn here() -> Context {
        Context {
            root: PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap()),
            args: Args::default(),
        }
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
    fn a_multi_line_reason_prints_one_indented_line_each() {
        struct Two;
        impl Exec for Two {
            fn exec(&mut self, _step: &Step, _context: &Context) -> Result<(), String> {
                Err("first problem\nsecond problem".into())
            }
        }
        let mut log = Vec::new();
        run(Kind::Job, &THREE, &here(), &mut Two, &mut log).unwrap();
        let log = String::from_utf8(log).unwrap();
        assert!(
            log.contains("xtask ci three: one failed:\n  first problem\n  second problem\n"),
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
        let pending_at = log
            .find("xtask ci three: RFC-0062 §2.3 checks not in this job yet (other task rows add more):\n")
            .unwrap();
        assert!(verdict_at < pending_at, "{log}");
        assert!(log.contains("  a later check (QUAL-E1-T99)\n"), "{log}");
    }

    #[test]
    fn lint_reports_its_pending_checks() {
        let lint = find(JOBS, "lint").unwrap();
        let (verdict, ran, log) = run_fake(lint, &[]);
        assert_eq!(verdict, Verdict::Pass);
        assert_eq!(
            ran,
            [
                "toolchain",
                "fmt",
                "clippy",
                "rustdoc",
                "determinism",
                "hot-threads",
                "orphan-rule",
                "check-deps",
                "properties",
                "codeowners",
                "agent-instructions",
                "rigs",
                "specs",
                "deny",
                "advisories",
                "spdx",
                "third-party",
                "notices",
                "workflows",
                "actionlint",
                "scenarios",
                "sim-presets",
                "fixtures",
                "toolchains",
                "jobs",
                "plan",
            ]
        );
        for pending in lint.pending {
            assert!(log.contains(pending.task), "{log}");
        }
    }

    #[test]
    fn list_aligns_names_and_summaries() {
        let mut out = Vec::new();
        list(&[THREE], &mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "three  three steps (RFC-0062 §2.3)\n"
        );
    }

    #[test]
    fn list_shows_each_option() {
        const WITH_OPTION: Entry = Entry {
            name: "changes",
            summary: "the affected-job set",
            defined_in: "RFC-0062 §2.2",
            params: &[Param {
                name: "--base",
                value: "<ref>",
                help: "the merge base",
                required: false,
            }],
            steps: &[],
            pending: &[],
        };
        let mut out = Vec::new();
        list(&[THREE, WITH_OPTION], &mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "three    three steps (RFC-0062 §2.3)\n\
             changes  the affected-job set (RFC-0062 §2.2)\n\
             \x20          --base <ref>  the merge base\n"
        );
    }

    #[test]
    fn the_label_carries_the_options() {
        const WITH_OPTION: &[Entry] = &[Entry {
            name: "one",
            summary: "one step",
            defined_in: "RFC-0062 §2.2",
            params: &[Param {
                name: "--base",
                value: "<ref>",
                help: "the merge base",
                required: false,
            }],
            steps: &[Step {
                name: "only",
                action: Action::Cargo(&["only"]),
            }],
            pending: &[],
        }];
        let line: Vec<String> = ["ci", "one", "--base=main"].map(String::from).into();
        let Ok(Cli::Run { kind, entry, args }) = cli::parse(&line, WITH_OPTION, &[]) else {
            panic!("did not parse");
        };
        let context = Context { args, ..here() };
        let mut fake = Fake {
            fail: &[],
            ran: Vec::new(),
        };
        let mut log = Vec::new();
        run(kind, entry, &context, &mut fake, &mut log).unwrap();
        let log = String::from_utf8(log).unwrap();
        assert!(
            log.starts_with("xtask ci one --base main: [1/1] only: cargo only\n"),
            "{log}"
        );
        assert!(
            log.contains("xtask ci one --base main: PASS (only)\n"),
            "{log}"
        );
    }

    #[test]
    fn system_passes_a_step_that_exits_zero() {
        let version = Step {
            name: "version",
            action: Action::Cargo(&["--version"]),
        };
        assert_eq!(System.exec(&version, &here()), Ok(()));
    }

    #[test]
    fn a_non_zero_exit_fails_with_its_status() {
        let mut command = Action::Cargo(&["syntropy-xtask-no-such-subcommand"])
            .command()
            .unwrap();
        command.stderr(std::process::Stdio::null());
        let reason = outcome(&mut command, "cargo syntropy-xtask-no-such-subcommand").unwrap_err();
        // `ExitStatus` prints as "exit status: N" on Unix and "exit code: N" on Windows.
        assert!(
            reason.contains("exit status") || reason.contains("exit code"),
            "{reason}"
        );
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
