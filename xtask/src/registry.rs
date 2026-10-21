// SPDX-License-Identifier: MIT OR Apache-2.0

//! The registry of CI jobs and test suites.
//!
//! A job is what `cargo xtask ci <job>` runs and a suite is what `cargo xtask test <suite>` runs.
//! Each is an [`Entry`]: a name, the options it takes and an ordered list of steps. CI calls these
//! names and nothing else, so a job gives the same verdict locally and in CI on the same commit
//! (RFC-0062 §1).
//!
//! To extend the registry:
//! - add a job to [`JOBS`] or a suite to [`SUITES`];
//! - give an entry options, such as `--base <ref>` for `ci changes` (RFC-0062 §2.2), by listing
//!   them in its `params`; the parser accepts exactly those, and the steps see the values in
//!   [`crate::run::Context`];
//! - add a check to a job by appending a [`Step`] to its list, and drop the matching row from its
//!   `pending` list, if it has one;
//! - add a kind of step, such as a check that runs inside this binary, as a variant of [`Action`].

use std::ffi::OsString;
use std::process::Command;

/// A named job or suite: the steps it runs, in order.
#[derive(Debug, PartialEq, Eq)]
pub struct Entry {
    /// The name on the command line. Lowercase words joined by `-`.
    pub name: &'static str,
    /// One line for `--list`.
    pub summary: &'static str,
    /// The plan section that defines this entry, such as `RFC-0062 §2.3`.
    pub defined_in: &'static str,
    /// The options the entry takes after its name. Anything else is a usage error.
    pub params: &'static [Param],
    /// Run in order. Every step runs, even after one fails, so one run reports every failure.
    pub steps: &'static [Step],
    /// Checks that `defined_in` puts in this entry and that no merged task has added yet, each
    /// with the task that adds it. Printed after the verdict, so a pass never claims more than the
    /// steps that ran. The list covers `defined_in` only: other task rows can add further checks.
    pub pending: &'static [Pending],
}

/// An option an entry takes: `--name <value>` or `--name=<value>`, at most once.
#[derive(Debug, PartialEq, Eq)]
pub struct Param {
    /// The option as typed, such as `--base`.
    pub name: &'static str,
    /// A placeholder for its value in `--list`, such as `<ref>`.
    pub value: &'static str,
    /// One line for `--list`.
    pub help: &'static str,
}

/// One step of an entry.
#[derive(Debug, PartialEq, Eq)]
pub struct Step {
    /// A short label for progress lines and the verdict, such as `fmt`.
    pub name: &'static str,
    pub action: Action,
}

/// What a step does.
#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    /// `cargo <args>`, with the cargo that started xtask (`$CARGO`), so the toolchain pinned in
    /// `rust-toolchain.toml` runs every step.
    Cargo(&'static [&'static str]),
}

/// A check the plan assigns to an entry, with the task that adds it.
#[derive(Debug, PartialEq, Eq)]
pub struct Pending {
    pub check: &'static str,
    /// The task ID, such as `QUAL-E1-T03`.
    pub task: &'static str,
}

impl Action {
    /// The process this action starts. The caller sets the working directory.
    pub fn command(&self) -> Command {
        match self {
            Action::Cargo(args) => {
                let mut command = Command::new(cargo());
                command.args(*args);
                command
            }
        }
    }

    /// The command as a user would type it, for progress lines.
    pub fn display(&self) -> String {
        match self {
            Action::Cargo(args) => {
                let mut line = String::from("cargo");
                for arg in *args {
                    line.push(' ');
                    line.push_str(arg);
                }
                line
            }
        }
    }
}

/// The cargo binary that started xtask. Cargo sets `CARGO` for `cargo run`.
fn cargo() -> OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

/// The CI jobs, called by `cargo xtask ci <job>`. RFC-0062 §2.3 lists the jobs the plan expects.
pub const JOBS: &[Entry] = &[Entry {
    name: "lint",
    summary: "the L0 static checks: rustfmt and clippy over the workspace",
    defined_in: "RFC-0062 §2.3 and §12.4",
    params: &[],
    steps: &[
        Step {
            name: "fmt",
            action: Action::Cargo(&["fmt", "--all", "--check"]),
        },
        Step {
            name: "clippy",
            action: Action::Cargo(&[
                "clippy",
                "--workspace",
                "--all-targets",
                "--locked",
                "--",
                "-D",
                "warnings",
            ]),
        },
    ],
    // The rest of RFC-0062 §2.3's `lint` row and the §12.4 workflow checks. Other task rows add
    // further checks to `lint`, such as the `ci/jobs.toml` goldens (QUAL-E1-T07).
    pending: &[
        Pending {
            check: "check-deps: RFC-0001 §4.2 rules, `forbid(unsafe_code)` placement, release features",
            task: "QUAL-E1-T03",
        },
        Pending {
            check: "cargo-deny, SPDX headers and the licence manifest",
            task: "QUAL-E1-T04",
        },
        Pending {
            check: "DCO and the PR title",
            task: "QUAL-E1-T05",
        },
        Pending {
            check: "actionlint and the workflow security lint",
            task: "QUAL-E1-T09",
        },
    ],
}];

/// The test suites, called by `cargo xtask test <suite>`. RFC-0060 §1.1 defines the layers.
pub const SUITES: &[Entry] = &[Entry {
    name: "unit",
    summary: "L1: the crate-local unit tests and doctests of every workspace crate",
    defined_in: "RFC-0060 §1.1",
    params: &[],
    steps: &[Step {
        name: "test",
        action: Action::Cargo(&["test", "--workspace", "--locked"]),
    }],
    pending: &[],
}];

/// The entry called `name`, if the registry has one.
pub fn find<'a>(entries: &'a [Entry], name: &str) -> Option<&'a Entry> {
    entries.iter().find(|entry| entry.name == name)
}

/// Every name in `entries`, in registry order.
pub fn names(entries: &[Entry]) -> Vec<&'static str> {
    entries.iter().map(|entry| entry.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn is_kebab_case(name: &str) -> bool {
        !name.is_empty()
            && !name.starts_with('-')
            && !name.ends_with('-')
            && !name.contains("--")
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    }

    /// `QUAL-E1-T03` and the like: an area, an epic number and a two-digit task number.
    fn is_task_id(id: &str) -> bool {
        let parts: Vec<&str> = id.split('-').collect();
        let [area, epic, task] = parts.as_slice() else {
            return false;
        };
        let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
        ["PLC", "TIER", "DIST", "COMP", "PROD", "QUAL"].contains(area)
            && epic.strip_prefix('E').is_some_and(digits)
            && task
                .strip_prefix('T')
                .is_some_and(|n| n.len() == 2 && digits(n))
    }

    fn check_entries(entries: &[Entry]) {
        let mut seen = HashSet::new();
        for entry in entries {
            assert!(is_kebab_case(entry.name), "bad name `{}`", entry.name);
            assert!(seen.insert(entry.name), "duplicate name `{}`", entry.name);
            assert!(!entry.summary.is_empty(), "`{}` has no summary", entry.name);
            assert!(
                entry.defined_in.starts_with("RFC-") || entry.defined_in.starts_with("ADR-"),
                "`{}` does not cite its definition",
                entry.name
            );
            let mut params = HashSet::new();
            for param in entry.params {
                let bare = param.name.strip_prefix("--").unwrap_or("");
                assert!(is_kebab_case(bare), "bad option `{}`", param.name);
                assert!(
                    params.insert(param.name),
                    "`{}` has two options called `{}`",
                    entry.name,
                    param.name
                );
                assert!(!param.value.is_empty() && !param.help.is_empty());
            }
            assert!(!entry.steps.is_empty(), "`{}` has no steps", entry.name);
            let mut steps = HashSet::new();
            for step in entry.steps {
                assert!(is_kebab_case(step.name), "bad step name `{}`", step.name);
                assert!(
                    steps.insert(step.name),
                    "`{}` has two steps called `{}`",
                    entry.name,
                    step.name
                );
            }
            for pending in entry.pending {
                assert!(
                    is_task_id(pending.task),
                    "`{}` names `{}`, which is not a task ID",
                    entry.name,
                    pending.task
                );
            }
        }
    }

    #[test]
    fn jobs_are_well_formed() {
        check_entries(JOBS);
    }

    #[test]
    fn suites_are_well_formed() {
        check_entries(SUITES);
    }

    #[test]
    fn lint_runs_rustfmt_and_clippy_as_rfc_0062_defines_them() {
        let lint = find(JOBS, "lint").expect("lint is registered");
        let commands: Vec<String> = lint.steps.iter().map(|s| s.action.display()).collect();
        assert_eq!(
            commands,
            [
                "cargo fmt --all --check",
                "cargo clippy --workspace --all-targets --locked -- -D warnings",
            ]
        );
    }

    #[test]
    fn lint_lists_the_checks_later_tasks_add() {
        let lint = find(JOBS, "lint").expect("lint is registered");
        let tasks: Vec<&str> = lint.pending.iter().map(|p| p.task).collect();
        assert_eq!(
            tasks,
            ["QUAL-E1-T03", "QUAL-E1-T04", "QUAL-E1-T05", "QUAL-E1-T09"]
        );
    }

    #[test]
    fn unit_suite_tests_the_whole_workspace() {
        let unit = find(SUITES, "unit").expect("unit is registered");
        assert_eq!(unit.steps.len(), 1);
        assert_eq!(
            unit.steps[0].action.display(),
            "cargo test --workspace --locked"
        );
    }

    #[test]
    fn find_matches_whole_names_only() {
        assert!(find(JOBS, "lint").is_some());
        assert!(find(JOBS, "lin").is_none());
        assert!(find(JOBS, "LINT").is_none());
        assert!(find(JOBS, "").is_none());
        assert!(find(SUITES, "lint").is_none());
    }

    #[test]
    fn names_keeps_registry_order() {
        assert_eq!(names(JOBS), JOBS.iter().map(|e| e.name).collect::<Vec<_>>());
    }

    #[test]
    fn cargo_action_passes_its_arguments_through() {
        let action = Action::Cargo(&["fmt", "--all", "--check"]);
        let command = action.command();
        let args: Vec<&std::ffi::OsStr> = command.get_args().collect();
        assert_eq!(args, ["fmt", "--all", "--check"]);
    }

    #[test]
    fn name_and_task_id_checks_reject_bad_forms() {
        for bad in [
            "", "-lint", "lint-", "li--nt", "Lint", "lint job", "lint_job",
        ] {
            assert!(!is_kebab_case(bad), "`{bad}` passed");
        }
        assert!(is_kebab_case("hosted-test-linux"));
        for bad in [
            "QUAL-E1-T3",
            "QUAL-1-T03",
            "XYZ-E1-T03",
            "QUAL-E1-T03-x",
            "QUAL-E-T03",
        ] {
            assert!(!is_task_id(bad), "`{bad}` passed");
        }
        assert!(is_task_id("COMP-E11-T03"));
    }
}
