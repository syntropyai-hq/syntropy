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
//! - add a kind of step as a variant of [`Action`]. A check that lives in this binary runs in this
//!   process as an [`Action::Builtin`] step, such as `codeowners`, or as a child process as an
//!   [`Action::Xtask`] step, such as `check-deps`.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;

use crate::run::Context;
use crate::{codeowners, contrib};

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
    /// Leaving a required option out is a usage error.
    pub required: bool,
}

/// One step of an entry.
#[derive(Debug, PartialEq, Eq)]
pub struct Step {
    /// A short label for progress lines and the verdict, such as `fmt`.
    pub name: &'static str,
    pub action: Action,
}

/// What a step does.
#[derive(Debug)]
pub enum Action {
    /// `cargo <args>`, with the cargo that started xtask (`$CARGO`), so the toolchain pinned in
    /// `rust-toolchain.toml` runs every step.
    Cargo(&'static [&'static str]),
    /// `cargo xtask <args>`: this binary again, as a child process, for a check that lives in
    /// xtask. It inherits the environment, so it checks the same checkout.
    Xtask(&'static [&'static str]),
    /// A check that runs inside xtask. `shown` describes it in progress lines; `run` returns
    /// `Err` with every problem it found.
    Builtin { shown: &'static str, run: Builtin },
}

/// A check that runs inside xtask.
pub type Builtin = fn(&Context) -> Result<(), String>;

/// Builtins compare by what they say they do: function addresses are not stable enough to compare.
impl PartialEq for Action {
    fn eq(&self, other: &Action) -> bool {
        match (self, other) {
            (Action::Cargo(a), Action::Cargo(b)) => a == b,
            (Action::Xtask(a), Action::Xtask(b)) => a == b,
            (Action::Builtin { shown: a, .. }, Action::Builtin { shown: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl Eq for Action {}

/// A check the plan assigns to an entry, with the task that adds it.
#[derive(Debug, PartialEq, Eq)]
pub struct Pending {
    pub check: &'static str,
    /// The task ID, such as `QUAL-E1-T03`.
    pub task: &'static str,
}

impl Action {
    /// The process this action starts, if it starts one. The caller sets the working directory.
    pub fn command(&self) -> Option<Command> {
        match self {
            Action::Cargo(args) => {
                let mut command = Command::new(cargo());
                command.args(*args);
                Some(command)
            }
            Action::Xtask(args) => {
                // If the path is unknown, the step fails with "could not start".
                let program = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("xtask"));
                let mut command = Command::new(program);
                command.args(*args);
                Some(command)
            }
            Action::Builtin { .. } => None,
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
            Action::Xtask(args) => {
                let mut line = String::from("cargo xtask");
                for arg in *args {
                    line.push(' ');
                    line.push_str(arg);
                }
                line
            }
            Action::Builtin { shown, .. } => format!("xtask: {shown}"),
        }
    }
}

/// The cargo binary that started xtask. Cargo sets `CARGO` for `cargo run`.
pub fn cargo() -> OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

/// The CI jobs, called by `cargo xtask ci <job>`. RFC-0062 §2.3 lists the jobs the plan expects.
pub const JOBS: &[Entry] = &[
    Entry {
        name: "lint",
        summary: "the L0 static checks: rustfmt, clippy, the crate dependency rules, CODEOWNERS and \
                  the licence policy",
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
            Step {
                name: "check-deps",
                action: Action::Xtask(&["check-deps"]),
            },
            Step {
                name: "codeowners",
                action: Action::Builtin {
                    shown: "CODEOWNERS names only maintainers and covers RFC-0062 §12.3's paths",
                    run: codeowners::step,
                },
            },
            // The rig inventory of RFC-0061 §6.3 (QUAL-E15-T01).
            Step {
                name: "rigs",
                action: Action::Cargo(&[
                    "run",
                    "--quiet",
                    "--locked",
                    "--package",
                    "syntropy-bench",
                    "--bin",
                    "syntropy-bench",
                    "--",
                    "rigs",
                    "check",
                ]),
            },
            // The licence policy of ADR-0018 (RFC-0062 §11.2, §12.2; QUAL-E1-T04). `advisories` is
            // the only step that needs the network, and it fails closed without it.
            Step {
                name: "deny",
                action: Action::Xtask(&["deny"]),
            },
            Step {
                name: "advisories",
                action: Action::Xtask(&["deny", "--advisories"]),
            },
            Step {
                name: "spdx",
                action: Action::Xtask(&["spdx"]),
            },
            Step {
                name: "third-party",
                action: Action::Xtask(&["third-party"]),
            },
            Step {
                name: "notices",
                action: Action::Xtask(&["notices"]),
            },
        ],
        // The rest of RFC-0062 §2.3's `lint` row and the §12.4 workflow checks. Other task rows add
        // further checks to `lint`, such as the `ci/jobs.toml` goldens (QUAL-E1-T07). The row's DCO
        // and PR-title checks need a PR's data, so they are the `dco` and `pr-title` jobs, which are
        // separate required checks (RFC-0062 §2.5).
        pending: &[Pending {
            check: "actionlint and the workflow security lint",
            task: "QUAL-E1-T09",
        }],
    },
    Entry {
        name: "dco",
        summary: "every commit carries a Signed-off-by that matches its author",
        defined_in: "RFC-0062 §8.3 and §12.3",
        params: &[
            Param {
                name: "--event",
                value: "<file>",
                help: "the pull_request event payload, or GET /repos/{owner}/{repo}/pulls/{n}",
                required: true,
            },
            Param {
                name: "--commits",
                value: "<file>",
                help: "every commit, as GET /repos/{owner}/{repo}/pulls/{n}/commits returns them",
                required: true,
            },
        ],
        steps: &[Step {
            name: "dco",
            action: Action::Builtin {
                shown: "every commit is signed off by its author",
                run: contrib::dco_step,
            },
        }],
        pending: &[],
    },
    Entry {
        name: "pr-title",
        summary: "the PR title is a conventional commit, and a breaking change has a maintainer's label",
        defined_in: "RFC-0062 §8.3",
        params: &[
            Param {
                name: "--event",
                value: "<file>",
                help: "the pull_request event payload, or GET /repos/{owner}/{repo}/pulls/{n}",
                required: true,
            },
            Param {
                name: "--commits",
                value: "<file>",
                help: "the PR's commits, as GET /repos/{owner}/{repo}/pulls/{n}/commits returns them",
                required: true,
            },
            Param {
                name: "--label-events",
                value: "<file>",
                help: "the label history, as GET /repos/{owner}/{repo}/issues/{n}/events returns it",
                required: true,
            },
            Param {
                name: "--roster",
                value: "<file>",
                help: "the maintainer roster to trust, such as the base branch's governance/maintainers.toml",
                required: true,
            },
            Param {
                name: "--version-file",
                value: "<file>",
                help: "the product version to compare Release-As with: the base branch's version.txt",
                required: true,
            },
        ],
        steps: &[Step {
            name: "pr-title",
            action: Action::Builtin {
                shown: "the title, and who applied semver-major-approved",
                run: contrib::pr_title_step,
            },
        }],
        pending: &[],
    },
];

/// The test suites, called by `cargo xtask test <suite>`. RFC-0060 §1.1 defines the layers.
pub const SUITES: &[Entry] = &[
    Entry {
        name: "unit",
        summary: "L1: the crate-local unit tests and doctests of every workspace crate",
        defined_in: "RFC-0060 §1.1",
        params: &[],
        steps: &[Step {
            name: "test",
            action: Action::Cargo(&["test", "--workspace", "--locked"]),
        }],
        pending: &[],
    },
    Entry {
        name: "pairing-grind",
        summary: "L2: the full grinding-relay runs of the pairing model (DIST-E5-T14): 20 first-draft \
                  trials, ten million honest-reveal attempts and 20 cheating reveals, \
                  release-optimised",
        defined_in: "RFC-0034 Testing and validation",
        params: &[],
        // The default tests run a seeded prefix of each; the dev profile is too slow for the
        // full runs. `test-release` is RFC-0060 §1.3's release-optimised test profile.
        steps: &[Step {
            name: "test",
            action: Action::Cargo(&[
                "test",
                "--profile",
                "test-release",
                "--locked",
                "--package",
                "syntropy-cluster",
                "--test",
                "pairing",
                "--",
                "--ignored",
            ]),
        }],
        pending: &[],
    },
];

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
    fn lint_runs_the_checks_rfc_0062_defines_for_it() {
        let lint = find(JOBS, "lint").expect("lint is registered");
        let commands: Vec<String> = lint.steps.iter().map(|s| s.action.display()).collect();
        assert_eq!(
            commands,
            [
                "cargo fmt --all --check",
                "cargo clippy --workspace --all-targets --locked -- -D warnings",
                "cargo xtask check-deps",
                "xtask: CODEOWNERS names only maintainers and covers RFC-0062 §12.3's paths",
                "cargo run --quiet --locked --package syntropy-bench --bin syntropy-bench -- rigs \
                 check",
                "cargo xtask deny",
                "cargo xtask deny --advisories",
                "cargo xtask spdx",
                "cargo xtask third-party",
                "cargo xtask notices",
            ]
        );
    }

    #[test]
    fn the_contribution_checks_are_jobs_with_required_inputs() {
        let dco = find(JOBS, "dco").expect("dco is registered");
        let required: Vec<&str> = dco
            .params
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name)
            .collect();
        assert_eq!(required, ["--event", "--commits"]);
        let title = find(JOBS, "pr-title").expect("pr-title is registered");
        let required: Vec<&str> = title
            .params
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name)
            .collect();
        assert_eq!(
            required,
            [
                "--event",
                "--commits",
                "--label-events",
                "--roster",
                "--version-file"
            ]
        );
    }

    #[test]
    fn lint_lists_the_checks_later_tasks_add() {
        let lint = find(JOBS, "lint").expect("lint is registered");
        let tasks: Vec<&str> = lint.pending.iter().map(|p| p.task).collect();
        assert_eq!(tasks, ["QUAL-E1-T09"]);
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
    fn pairing_grind_runs_the_ignored_pairing_tests_release_optimised() {
        let suite = find(SUITES, "pairing-grind").expect("pairing-grind is registered");
        assert_eq!(suite.steps.len(), 1);
        assert_eq!(
            suite.steps[0].action.display(),
            "cargo test --profile test-release --locked --package syntropy-cluster --test pairing \
             -- --ignored"
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
        let command = action.command().unwrap();
        let args: Vec<&std::ffi::OsStr> = command.get_args().collect();
        assert_eq!(args, ["fmt", "--all", "--check"]);
    }

    #[test]
    fn xtask_action_runs_this_binary_with_its_arguments() {
        let action = Action::Xtask(&["check-deps"]);
        let command = action.command().unwrap();
        assert_eq!(
            Some(command.get_program()),
            std::env::current_exe()
                .ok()
                .as_deref()
                .map(|p| p.as_os_str())
        );
        let args: Vec<&std::ffi::OsStr> = command.get_args().collect();
        assert_eq!(args, ["check-deps"]);
        assert_eq!(action.display(), "cargo xtask check-deps");
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
