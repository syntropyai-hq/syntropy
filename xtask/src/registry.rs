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
use crate::{
    ci_gate, codeowners, contrib, dep_diff, determinism, instructions, properties, repo_settings,
    sim_presets, toolchain,
};

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
        summary: "the L0 static checks: rustfmt, clippy with the determinism lints, the crate \
                  dependency rules, CODEOWNERS and the licence policy",
        defined_in: "RFC-0062 §2.3 and §12.4",
        params: &[],
        steps: &[
            // First, so a run on a cargo that ignored `rust-toolchain.toml` says so up front.
            Step {
                name: "toolchain",
                action: Action::Builtin {
                    shown: "cargo and rustc are the release rust-toolchain.toml pins",
                    run: toolchain::step,
                },
            },
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
            // What clippy cannot check about the determinism lints of RFC-0060 §2.1 and §2.4,
            // which the `clippy` step applies (QUAL-E3-T03).
            Step {
                name: "determinism",
                action: Action::Builtin {
                    shown: "the covered crates keep ci/clippy-determinism.toml and lift it only \
                            where listed",
                    run: determinism::step,
                },
            },
            Step {
                name: "check-deps",
                action: Action::Xtask(&["check-deps"]),
            },
            // The property registry of RFC-0060 §3.2 (QUAL-E3-T05).
            Step {
                name: "properties",
                action: Action::Builtin {
                    shown: "ci/properties.toml matches the property tests, RFC-0060 and the epics",
                    run: properties::step,
                },
            },
            Step {
                name: "codeowners",
                action: Action::Builtin {
                    shown: "CODEOWNERS names only maintainers and covers RFC-0062 §12.3's paths",
                    run: codeowners::step,
                },
            },
            Step {
                name: "agent-instructions",
                action: Action::Builtin {
                    shown: "ci/agent-instructions.toml is well formed and CODEOWNERS owns every \
                            file it lists",
                    run: instructions::step,
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
            // The spec database's source lint (RFC-0061 §5.3; QUAL-E12-T11).
            Step {
                name: "specs",
                action: Action::Cargo(&[
                    "run",
                    "--quiet",
                    "--locked",
                    "--package",
                    "syntropy-bench",
                    "--bin",
                    "syntropy-bench",
                    "--",
                    "specs",
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
            // Scenario files, the SC map and RFC-0061 §4.2's table (QUAL-E13-T01, RFC-0061 §4).
            Step {
                name: "scenarios",
                action: Action::Cargo(&[
                    "run",
                    "--locked",
                    "--quiet",
                    "--package",
                    "syntropy-bench",
                    "--bin",
                    "syntropy-bench",
                    "--",
                    "scenarios",
                    "check",
                ]),
            },
            // The simulator's device and drive presets cite their sources (QUAL-E8-T03, RFC-0060
            // §8.5).
            Step {
                name: "sim-presets",
                action: Action::Builtin {
                    shown: "every simulator preset cites its sources, and labels and sources each figure",
                    run: sim_presets::step,
                },
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
    Entry {
        name: "dep-diff",
        summary: "a new direct dependency or vendored component has a maintainer's approval, and \
                  each new crate resolves on crates.io",
        defined_in: "RFC-0062 §11.4 and §12.3",
        params: &[
            Param {
                name: "--base",
                value: "<dir>",
                help: "a checkout of the PR's base commit",
                required: true,
            },
            Param {
                name: "--head",
                value: "<dir>",
                help: "a checkout of the PR merged into its base",
                required: true,
            },
            Param {
                name: "--event",
                value: "<file>",
                help: "the pull_request event payload, for the head commit",
                required: false,
            },
            Param {
                name: "--comments",
                value: "<file>",
                help: "every comment, as GET /repos/{owner}/{repo}/issues/{n}/comments returns them",
                required: false,
            },
            Param {
                name: "--roster",
                value: "<file>",
                help: "the maintainer roster to trust: the base branch's governance/maintainers.toml",
                required: false,
            },
            Param {
                name: "--registry",
                value: "<dir>",
                help: "recorded crates.io answers to read instead of asking crates.io",
                required: false,
            },
        ],
        steps: &[
            Step {
                name: "dep-diff",
                action: Action::Builtin {
                    shown: "every new direct dependency has a maintainer's approval",
                    run: dep_diff::step,
                },
            },
            Step {
                name: "registry",
                action: Action::Builtin {
                    shown: "every new crate resolves on crates.io, with its publisher printed",
                    run: dep_diff::registry_step,
                },
            },
        ],
        pending: &[],
    },
    // The removal half of the property registry (QUAL-E3-T05). It reads the PR description, so
    // it runs in `pr-meta.yml` with the base branch's xtask, like `dco` and `pr-title`.
    Entry {
        name: "properties-removal",
        summary: "a property removed from ci/properties.toml has a reason in the PR description",
        defined_in: "RFC-0060 §3.2",
        params: &[
            Param {
                name: "--base",
                value: "<dir>",
                help: "a checkout of the PR's base commit",
                required: true,
            },
            Param {
                name: "--head",
                value: "<dir>",
                help: "a checkout of the PR merged into its base",
                required: true,
            },
            Param {
                name: "--event",
                value: "<file>",
                help: "the pull_request event payload, or GET /repos/{owner}/{repo}/pulls/{n}",
                required: true,
            },
        ],
        steps: &[Step {
            name: "properties-removal",
            action: Action::Builtin {
                shown: "each removed property has a `Removes <ID>: <reason>` line",
                run: properties::removal_step,
            },
        }],
        pending: &[],
    },
    Entry {
        name: "build",
        summary: "`build-<os>` in `pr.yml`: the PR build of the workspace on each Tier 1 OS",
        defined_in: "RFC-0062 §2.3 and §4.4",
        params: &[],
        // Until the `pr` profile exists, the PR build is the `dev` profile, which the hosted
        // tests use too.
        steps: &[Step {
            name: "dev",
            action: Action::Xtask(&["build", "--profile", "dev"]),
        }],
        pending: &[Pending {
            check: "the `pr` profile build, in place of the `dev` build",
            task: "QUAL-E16-T21",
        }],
    },
    Entry {
        name: "build-test-release",
        summary: "the Linux `test-release` build for L4, in `build-linux`",
        defined_in: "RFC-0062 §2.3 and §4.4",
        params: &[],
        steps: &[Step {
            name: "test-release",
            action: Action::Xtask(&["build", "--profile", "test-release"]),
        }],
        pending: &[],
    },
    Entry {
        name: "ci-gate",
        summary: "the required check over `pr.yml`: every job `changes` listed passed, and no job \
                  failed or was cancelled",
        defined_in: "RFC-0062 §2.3 and §2.5",
        params: &[Param {
            name: "--needs",
            value: "<file>",
            help: "the `needs` context of the `ci-gate` job, as `${{ toJSON(needs) }}` writes it",
            required: true,
        }],
        steps: &[Step {
            name: "ci-gate",
            action: Action::Builtin {
                shown: "every listed job passed, and none failed or was cancelled",
                run: ci_gate::step,
            },
        }],
        pending: &[Pending {
            check: "a passing `hw/<job>` status for each job in `hw_required`, by event",
            task: "QUAL-E1-T07",
        }],
    },
    Entry {
        name: "repo-settings",
        summary: "the repository settings and branch rulesets match `ci/repo-settings/`, read \
                  through the REST API",
        defined_in: "RFC-0062 §3.3 and §12.6",
        params: &[
            Param {
                name: "--repo",
                value: "<owner/name>",
                help: "the repository; else $GITHUB_REPOSITORY, else this checkout's",
                required: false,
            },
            Param {
                name: "--from",
                value: "<dir>",
                help: "read recorded responses from <dir> instead of the API",
                required: false,
            },
            Param {
                name: "--save",
                value: "<dir>",
                help: "also write each API response to <dir>, to record a fixture",
                required: false,
            },
        ],
        steps: &[Step {
            name: "repo-settings",
            action: Action::Builtin {
                shown: "the settings and rulesets declared in ci/repo-settings/ are in force",
                run: repo_settings::step,
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
        steps: &[
            Step {
                name: "test",
                action: Action::Cargo(&["test", "--workspace", "--locked"]),
            },
            // `--workspace` builds `obs` without its test-only feature, so the fault plans,
            // triggers and counters of RFC-0060 §9.2 are tested in a build of their own
            // (QUAL-E9-T01).
            Step {
                name: "fault-injection",
                action: Action::Cargo(&[
                    "test",
                    "--package",
                    "syntropy-obs",
                    "--features",
                    "fault-injection",
                    "--locked",
                ]),
            },
        ],
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
    Entry {
        name: "election-model",
        summary: "L2: the full model check of coordinator election, lease, replication and heal \
                  (DIST-E10-T08) on 3 and 4 nodes with partitions and message loss, \
                  release-optimised",
        defined_in: "RFC-0032 Testing and validation",
        params: &[],
        // The default tests check the bounded 3-node configuration; the dev profile is too slow
        // for the full one. `test-release` is RFC-0060 §1.3's release-optimised test profile.
        // `--nocapture` prints each scenario's state count against its cap.
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
                "election",
                "--",
                "--ignored",
                "--nocapture",
            ]),
        }],
        pending: &[],
    },
    Entry {
        name: "properties",
        summary: "L2: the named properties of ci/properties.toml at a given case count",
        defined_in: "RFC-0060 §1.1 and §3.1",
        params: &[Param {
            name: "--cases",
            value: "<n>",
            help: "cases per property, as PROPTEST_CASES (default 256, a PR run; nightly 16384)",
            required: false,
        }],
        steps: &[Step {
            name: "properties",
            action: Action::Builtin {
                shown: "each listed property's tests with PROPTEST_CASES set",
                run: properties::suite_step,
            },
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
                "xtask: cargo and rustc are the release rust-toolchain.toml pins",
                "cargo fmt --all --check",
                "cargo clippy --workspace --all-targets --locked -- -D warnings",
                "xtask: the covered crates keep ci/clippy-determinism.toml and lift it only where \
                 listed",
                "cargo xtask check-deps",
                "xtask: ci/properties.toml matches the property tests, RFC-0060 and the epics",
                "xtask: CODEOWNERS names only maintainers and covers RFC-0062 §12.3's paths",
                "xtask: ci/agent-instructions.toml is well formed and CODEOWNERS owns every file it \
                 lists",
                "cargo run --quiet --locked --package syntropy-bench --bin syntropy-bench -- rigs \
                 check",
                "cargo run --quiet --locked --package syntropy-bench --bin syntropy-bench -- specs \
                 check",
                "cargo xtask deny",
                "cargo xtask deny --advisories",
                "cargo xtask spdx",
                "cargo xtask third-party",
                "cargo xtask notices",
                "cargo run --locked --quiet --package syntropy-bench --bin syntropy-bench -- \
                 scenarios check",
                "xtask: every simulator preset cites its sources, and labels and sources each \
                 figure",
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
    fn the_build_jobs_build_profiles_that_exist() {
        for (job, profile) in [("build", "dev"), ("build-test-release", "test-release")] {
            let entry = find(JOBS, job).expect("the build jobs are registered");
            let commands: Vec<String> = entry.steps.iter().map(|s| s.action.display()).collect();
            assert_eq!(
                commands,
                [format!("cargo xtask build --profile {profile}")],
                "{job}"
            );
            assert!(
                crate::build::find(profile).is_some(),
                "{job}: no profile `{profile}`"
            );
        }
        // `build` names the profile that replaces `dev`.
        let build = find(JOBS, "build").unwrap();
        let tasks: Vec<&str> = build.pending.iter().map(|p| p.task).collect();
        assert_eq!(tasks, ["QUAL-E16-T21"]);
    }

    #[test]
    fn ci_gate_needs_the_needs_context_and_repo_settings_needs_nothing() {
        let gate = find(JOBS, "ci-gate").expect("ci-gate is registered");
        let params: Vec<(&str, bool)> = gate.params.iter().map(|p| (p.name, p.required)).collect();
        assert_eq!(params, [("--needs", true)]);
        let settings = find(JOBS, "repo-settings").expect("repo-settings is registered");
        let params: Vec<(&str, bool)> = settings
            .params
            .iter()
            .map(|p| (p.name, p.required))
            .collect();
        assert_eq!(
            params,
            [("--repo", false), ("--from", false), ("--save", false)]
        );
    }

    #[test]
    fn unit_suite_tests_the_whole_workspace_then_the_fault_points() {
        let unit = find(SUITES, "unit").expect("unit is registered");
        let steps: Vec<String> = unit
            .steps
            .iter()
            .map(|step| step.action.display())
            .collect();
        assert_eq!(
            steps,
            [
                "cargo test --workspace --locked",
                "cargo test --package syntropy-obs --features fault-injection --locked",
            ]
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
    fn election_model_runs_the_ignored_election_tests_release_optimised() {
        let suite = find(SUITES, "election-model").expect("election-model is registered");
        assert_eq!(suite.steps.len(), 1);
        assert_eq!(
            suite.steps[0].action.display(),
            "cargo test --profile test-release --locked --package syntropy-cluster --test election \
             -- --ignored --nocapture"
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
