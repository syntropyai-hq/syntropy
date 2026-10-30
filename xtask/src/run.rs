// SPDX-License-Identifier: MIT OR Apache-2.0

//! Running a job or suite and reaching its verdict.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;

use crate::cli::{Args, Kind};
use crate::flake::{self, Class, End, Outcome};
use crate::registry::{Action, Entry, Step};

/// What every step of one run sees.
#[derive(Debug)]
pub struct Context {
    /// The workspace root of the checkout xtask was started from. Every step runs there.
    pub root: PathBuf,
    /// The entry's option values from the command line.
    pub args: Args,
}

/// Why a step did not pass, and whether the failure is the code's or the infrastructure's
/// (RFC-0060 §13).
#[derive(Debug, PartialEq, Eq)]
pub struct Failure {
    pub class: Class,
    pub reason: String,
}

impl Failure {
    /// A failure of the infrastructure, not of the code under test.
    pub fn infra(reason: String) -> Failure {
        Failure {
            class: Class::Infra,
            reason,
        }
    }
}

/// A check's own failure is a `test` failure.
impl From<String> for Failure {
    fn from(reason: String) -> Failure {
        Failure {
            class: Class::Test,
            reason,
        }
    }
}

impl From<&str> for Failure {
    fn from(reason: &str) -> Failure {
        Failure::from(reason.to_owned())
    }
}

/// Runs one step. The real implementation is [`System`]; tests substitute a fake.
pub trait Exec {
    /// Runs `step` in `context`. `Err` carries why it did not pass.
    fn exec(&mut self, step: &Step, context: &Context) -> Result<(), Failure>;
}

/// Runs a `cargo` step as a child process from the workspace root, inheriting stdout and
/// stderr, and a builtin step inside this process.
pub struct System;

impl Exec for System {
    fn exec(&mut self, step: &Step, context: &Context) -> Result<(), Failure> {
        match (&step.action, step.action.command()) {
            (Action::Builtin { run, .. }, _) => run(context).map_err(Failure::from),
            (action, Some(mut command)) => {
                command.current_dir(&context.root);
                // Only xtask's own exit codes carry its classes; cargo's do not.
                let ours = matches!(action, Action::Xtask(_));
                outcome(&mut command, &step.action.display(), ours)
            }
            (_, None) => Err(format!("`{}` starts no process", step.action.display()).into()),
        }
    }
}

/// Runs `command` to completion. A zero exit status passes; anything else fails with the reason.
/// A program that cannot start is an `infra` failure, and so is a child xtask (`ours`) that
/// exits with [`flake::EXIT_INFRA`]; every other failure is a `test` failure. `shown` is the
/// command as progress lines print it.
fn outcome(command: &mut Command, shown: &str, ours: bool) -> Result<(), Failure> {
    match command.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status)
            if ours
                && flake::classify(status.code().map_or(End::Signalled, End::Exited))
                    == Outcome::Fail(Class::Infra) =>
        {
            Err(Failure::infra(status.to_string()))
        }
        Ok(status) => Err(status.to_string().into()),
        Err(error) => Err(Failure::infra(format!(
            "could not start `{shown}`: {error}"
        ))),
    }
}

/// The outcome of a job or suite.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    /// The names of the steps that failed, in order. At least one was a `test` failure.
    Fail(Vec<&'static str>),
    /// The names of the steps that failed, in order, every one an `infra` failure
    /// (RFC-0060 §13).
    Infra(Vec<&'static str>),
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
    let mut class = Class::Infra;
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
            Err(Failure {
                class: this,
                reason,
            }) => {
                if this == Class::Test {
                    class = Class::Test;
                }
                let infra = if this == Class::Infra { " (infra)" } else { "" };
                failed.push(step.name);
                if reason.contains('\n') {
                    // A check that found several problems: one indented line each.
                    writeln!(log, "{label}: {} failed{infra}:", step.name)?;
                    for line in reason.lines() {
                        writeln!(log, "  {line}")?;
                    }
                } else {
                    writeln!(log, "{label}: {} failed{infra}: {reason}", step.name)?;
                }
            }
        }
    }
    let ran: Vec<&str> = entry.steps.iter().map(|s| s.name).collect();
    if failed.is_empty() {
        writeln!(log, "{label}: PASS ({})", ran.join(", "))?;
    } else {
        let infra = if class == Class::Infra { "infra: " } else { "" };
        writeln!(
            log,
            "{label}: FAIL ({infra}{} of {total} failed: {})",
            failed.len(),
            failed.join(", ")
        )?;
        // A failed suite says whether it is retried, so the job log shows it (RFC-0060 §13).
        // xtask itself never retries.
        if kind == Kind::Suite {
            writeln!(log, "{label}: {}", flake::decision_line(entry.name, class))?;
        }
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
    } else if class == Class::Infra {
        Verdict::Infra(failed)
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
        fn exec(&mut self, step: &Step, _context: &Context) -> Result<(), Failure> {
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
    fn a_deterministic_suite_failure_is_run_once_and_the_log_says_it_is_not_retried() {
        let unit = find(crate::registry::SUITES, "unit").unwrap();
        let mut fake = Fake {
            fail: &["test"],
            ran: Vec::new(),
        };
        let mut log = Vec::new();
        let verdict = run(Kind::Suite, unit, &here(), &mut fake, &mut log).unwrap();
        let log = String::from_utf8(log).unwrap();
        assert_eq!(verdict, Verdict::Fail(vec!["test"]));
        // Each step ran once: nothing ran the failed step again.
        assert_eq!(fake.ran, ["test", "fault-injection", "ipc-fault-injection"]);
        assert!(
            log.contains(
                "xtask test unit: FAIL (1 of 3 failed: test)\n\
                 xtask test unit: not retried: `unit` is a deterministic suite, and a failure in \
                 it is a failure (RFC-0060 §13)\n"
            ),
            "{log}"
        );
    }

    #[test]
    fn a_job_failure_carries_no_retry_line() {
        let (_, _, log) = run_fake(&THREE, &["two"]);
        assert!(!log.contains("retried"), "{log}");
    }

    /// Fails the steps in `infra` as infrastructure failures and those in `test` as test failures.
    struct Mixed {
        infra: &'static [&'static str],
        test: &'static [&'static str],
    }

    impl Exec for Mixed {
        fn exec(&mut self, step: &Step, _context: &Context) -> Result<(), Failure> {
            if self.infra.contains(&step.name) {
                Err(Failure::infra("could not start `cargo`: not found".into()))
            } else if self.test.contains(&step.name) {
                Err("exit status: 1".into())
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn only_infra_failures_make_an_infra_verdict() {
        let unit = find(crate::registry::SUITES, "unit").unwrap();
        let mut log = Vec::new();
        let mut infra = Mixed {
            infra: &["test"],
            test: &[],
        };
        let verdict = run(Kind::Suite, unit, &here(), &mut infra, &mut log).unwrap();
        let log = String::from_utf8(log).unwrap();
        assert_eq!(verdict, Verdict::Infra(vec!["test"]));
        assert!(
            log.contains("xtask test unit: test failed (infra): could not start"),
            "{log}"
        );
        assert!(
            log.contains("xtask test unit: FAIL (infra: 1 of 3 failed: test)\n"),
            "{log}"
        );
        assert!(
            log.contains("xtask test unit: infra failure (exit 3): "),
            "{log}"
        );
        // One real failure makes the whole run a test failure, which is never retried.
        let mut log = Vec::new();
        let mut both = Mixed {
            infra: &["test"],
            test: &["fault-injection"],
        };
        let verdict = run(Kind::Suite, unit, &here(), &mut both, &mut log).unwrap();
        let log = String::from_utf8(log).unwrap();
        assert_eq!(verdict, Verdict::Fail(vec!["test", "fault-injection"]));
        assert!(log.contains("xtask test unit: not retried: "), "{log}");
    }

    #[test]
    fn a_child_xtask_infra_exit_stays_infra_and_cargo_codes_do_not() {
        // A shell stands in for the child: exit 3 from an xtask step is infra, from cargo it is
        // not.
        if cfg!(windows) {
            return;
        }
        let mut command = Command::new("sh");
        command.args(["-c", "exit 3"]);
        let failure = outcome(&mut command, "cargo xtask x", true).unwrap_err();
        assert_eq!(failure.class, Class::Infra);
        let mut command = Command::new("sh");
        command.args(["-c", "exit 3"]);
        let failure = outcome(&mut command, "cargo x", false).unwrap_err();
        assert_eq!(failure.class, Class::Test);
    }

    #[test]
    fn a_multi_line_reason_prints_one_indented_line_each() {
        struct Two;
        impl Exec for Two {
            fn exec(&mut self, _step: &Step, _context: &Context) -> Result<(), Failure> {
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
                "toolchains",
                "jobs",
                "retries",
                "quarantine",
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
        let failure = outcome(
            &mut command,
            "cargo syntropy-xtask-no-such-subcommand",
            false,
        )
        .unwrap_err();
        assert_eq!(failure.class, Class::Test);
        let reason = failure.reason;
        // `ExitStatus` prints as "exit status: N" on Unix and "exit code: N" on Windows.
        assert!(
            reason.contains("exit status") || reason.contains("exit code"),
            "{reason}"
        );
    }

    #[test]
    fn a_program_that_cannot_start_fails_with_a_reason() {
        let mut command = Command::new("syntropy-xtask-no-such-program");
        let failure = outcome(&mut command, "syntropy-xtask-no-such-program", false).unwrap_err();
        // The runner could not run the check at all: that is the infrastructure's failure.
        assert_eq!(failure.class, Class::Infra);
        let reason = failure.reason;
        assert!(
            reason.starts_with("could not start `syntropy-xtask-no-such-program`: "),
            "{reason}"
        );
    }
}
