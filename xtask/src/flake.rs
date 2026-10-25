// SPDX-License-Identifier: MIT OR Apache-2.0

//! The retry half of the flake policy of RFC-0060 §13 (QUAL-E3-T14): the harness exit codes that
//! separate an `infra` failure from a `test` failure, the retry decision for each, and the
//! `retries` step of `lint`, which checks that nothing configures a retry.
//!
//! - **Exit codes.** xtask exits [`EXIT_PASS`], [`EXIT_TEST`], [`EXIT_USAGE`] or [`EXIT_INFRA`].
//!   A runner reads the class from the code with [`classify`], never from the log. Anything it
//!   cannot place, a process killed by a signal among them, is a `test` failure, so an unknown
//!   failure is never retried.
//! - **Retry decision.** [`decide`] is the whole policy. A `test` failure in a deterministic suite
//!   (L1–L3, L6, and L5 on deterministic kernels) is never retried. The `stat-props` suite
//!   (QUAL-E3-T15) re-runs once with fresh seeds. A non-deterministic suite (L7, L8) is not
//!   retried either: a test that fails and then passes is a quarantine candidate
//!   ([`crate::quarantine`]), not a retry. An `infra` failure is retried once on another rig of
//!   the same class, then reported to the maintainers.
//! - **Who retries.** xtask never retries a step. A failed suite's log ends with the decision for
//!   it, so the job log shows that a deterministic failure was not retried. Acting on an `infra`
//!   decision needs the hardware dispatcher (QUAL-E2-T01), which picks the other rig; on a hosted
//!   runner nothing retries it.

use std::path::Path;

use crate::registry::SUITES;
use crate::run::Context;

/// Every step passed, or help or a list was printed.
pub const EXIT_PASS: u8 = 0;
/// A `test` failure: a check or assertion failed, or xtask could not write its log.
pub const EXIT_TEST: u8 = 1;
/// The command line was not understood, or xtask was not started by `cargo xtask`. Another rig
/// would not understand it either, so this is a `test` failure.
pub const EXIT_USAGE: u8 = 2;
/// An `infra` failure and no `test` failure: the run could not test anything, because a program
/// it needs could not start, a fixture could not be fetched, a device was missing from
/// enumeration, or the lab's PDU or switch API failed.
pub const EXIT_INFRA: u8 = 3;

/// The class of a failure (RFC-0060 §13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// A real assertion or check failure. Never retried, except `stat-props` once.
    Test,
    /// The runner, a fixture, a device or the lab failed, not the code under test.
    Infra,
}

/// How a run ended, as the runner that started it sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// The harness exited with this code.
    Exited(i32),
    /// The harness was killed by a signal, so it set no code. A crash in the code under test
    /// ends this way.
    Signalled,
    /// The runner stopped answering before the harness ended. Only the dispatcher sees this.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the lab dispatcher (QUAL-E2-T01) reports it")
    )]
    RunnerLost,
}

/// The outcome of a run, from how it ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Pass,
    Fail(Class),
}

/// The outcome of a run from its harness's exit code. Only [`EXIT_INFRA`] and a lost runner are
/// `infra`; every other failure, including a code xtask never uses, is `test`.
pub fn classify(end: End) -> Outcome {
    match end {
        End::Exited(0) => Outcome::Pass,
        End::Exited(code) if code == i32::from(EXIT_INFRA) => Outcome::Fail(Class::Infra),
        End::RunnerLost => Outcome::Fail(Class::Infra),
        End::Exited(_) | End::Signalled => Outcome::Fail(Class::Test),
    }
}

/// Whether a suite's tests are reproducible by construction (RFC-0060 §13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// L1–L3, L6, and L5 on deterministic kernels: the same commit gives the same verdict.
    Deterministic,
    /// The `stat-props` suite (RFC-0060 §3.1): tests with a stated false-positive rate.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no suite has it until QUAL-E3-T15 adds `stat-props`"
        )
    )]
    StatProps,
    /// L7 and L8: real devices and networks.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "no suite has it until the first L7 or L8 suite lands"
        )
    )]
    NonDeterministic,
}

/// The kind of each suite in [`SUITES`]. The `retries` step fails on a suite missing here, so a
/// new suite states its kind before it can run in CI.
const SUITE_KINDS: &[(&str, Kind)] = &[
    ("unit", Kind::Deterministic),            // L1
    ("loom", Kind::Deterministic),            // L3
    ("pairing-grind", Kind::Deterministic),   // L2, seeded
    ("election-model", Kind::Deterministic),  // L2, exhaustive
    ("properties", Kind::Deterministic),      // L2, seeded proptest
    ("runner-build-id", Kind::Deterministic), // local builds and one ignored test
    ("cache-hit", Kind::Deterministic),       // two local builds, cargo's JSON compared
];

/// The kind of the suite called `name`, if it has one.
pub fn suite_kind(name: &str) -> Option<Kind> {
    SUITE_KINDS
        .iter()
        .find(|(suite, _)| *suite == name)
        .map(|&(_, kind)| kind)
}

/// What to do about a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// It passed.
    Pass,
    /// Report the failure as it is. No retry.
    Fail,
    /// Run the suite once more on the same rig with fresh seeds; a second rejection fails.
    RerunWithFreshSeeds,
    /// Run the job once more on another rig of the same class.
    RetryOnAnotherRig,
    /// The retry failed on infrastructure too: report it to the maintainers, since fixing a rig
    /// is a Maintainer action (ADR-0021).
    ReportToMaintainers,
}

/// The retry decision for a run of a suite of `kind` that ended with `outcome` on its
/// `attempt`th try, counting from 1 (RFC-0060 §13).
pub fn decide(kind: Kind, outcome: Outcome, attempt: u32) -> Decision {
    match (outcome, kind) {
        (Outcome::Pass, _) => Decision::Pass,
        (Outcome::Fail(Class::Infra), _) if attempt <= 1 => Decision::RetryOnAnotherRig,
        (Outcome::Fail(Class::Infra), _) => Decision::ReportToMaintainers,
        (Outcome::Fail(Class::Test), Kind::StatProps) if attempt <= 1 => {
            Decision::RerunWithFreshSeeds
        }
        (Outcome::Fail(Class::Test), _) => Decision::Fail,
    }
}

/// The line a failed run of suite `name` ends its log with: the decision and why.
pub fn decision_line(name: &str, class: Class) -> String {
    let Some(kind) = suite_kind(name) else {
        return format!(
            "not retried: `{name}` has no kind in xtask/src/flake.rs, so it is treated as a \
             failure (RFC-0060 §13)"
        );
    };
    match decide(kind, Outcome::Fail(class), 1) {
        Decision::Fail => match kind {
            Kind::Deterministic => format!(
                "not retried: `{name}` is a deterministic suite, and a failure in it is a \
                 failure (RFC-0060 §13)"
            ),
            _ => format!("not retried: a test failure in `{name}` (RFC-0060 §13)"),
        },
        Decision::RerunWithFreshSeeds => format!(
            "`{name}` re-runs once with fresh seeds; only a repeated rejection counts \
             (RFC-0060 §13)"
        ),
        Decision::RetryOnAnotherRig => format!(
            "infra failure (exit {EXIT_INFRA}): the lab dispatcher retries `{name}` once on \
             another rig of the same class (RFC-0060 §13, QUAL-E2-T01); xtask and hosted runners \
             do not retry it"
        ),
        Decision::Pass | Decision::ReportToMaintainers => {
            unreachable!("a first failed attempt neither passes nor exhausts its retry")
        }
    }
}

/// nextest's configuration, relative to the workspace root, if the workspace ever adds it.
const NEXTEST: &str = ".config/nextest.toml";
/// The workflows, relative to the workspace root.
const WORKFLOWS: &str = ".github/workflows";

/// The `retries` step of `cargo xtask ci lint`: every suite has a kind, and nothing configures
/// a retry of a suite.
pub fn step(context: &Context) -> Result<(), String> {
    let mut problems = unclassified(SUITES.iter().map(|entry| entry.name));
    let nextest = context.root.join(NEXTEST);
    if nextest.is_file() {
        let text = std::fs::read_to_string(&nextest).map_err(|e| format!("{NEXTEST}: {e}"))?;
        problems.extend(nextest_retries(&text));
    }
    for (name, text) in workflows(&context.root.join(WORKFLOWS))? {
        problems.extend(
            workflow_retries(&text)
                .into_iter()
                .map(|problem| format!("{WORKFLOWS}/{name}: {problem}")),
        );
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// A problem for each suite name with no kind.
fn unclassified<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    names
        .filter(|name| suite_kind(name).is_none())
        .map(|name| {
            format!(
                "suite `{name}` has no kind in xtask/src/flake.rs: say whether it is \
                 deterministic (RFC-0060 §13)"
            )
        })
        .collect()
}

/// Every workflow file in `dir`, by name, in name order.
fn workflows(dir: &Path) -> Result<Vec<(String, String)>, String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(Vec::new());
    };
    let mut files = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".yml") || name.ends_with(".yaml") {
            let text = std::fs::read_to_string(entry.path())
                .map_err(|e| format!("{WORKFLOWS}/{name}: {e}"))?;
            files.push((name, text));
        }
    }
    files.sort();
    Ok(files)
}

/// A problem for each non-zero `retries` setting anywhere in nextest's configuration.
fn nextest_retries(text: &str) -> Vec<String> {
    let value: toml::Value = match toml::from_str(text) {
        Ok(value) => value,
        Err(error) => return vec![format!("{NEXTEST}: not TOML: {error}")],
    };
    let mut problems = Vec::new();
    find_retries(&value, &mut Vec::new(), &mut problems);
    problems
}

fn find_retries(value: &toml::Value, path: &mut Vec<String>, problems: &mut Vec<String>) {
    match value {
        toml::Value::Table(table) => {
            for (key, inner) in table {
                path.push(key.clone());
                if key == "retries" && !is_zero_retries(inner) {
                    problems.push(format!(
                        "{NEXTEST}: `{}` configures a retry; deterministic suites get none \
                         (RFC-0060 §13)",
                        path.join(".")
                    ));
                } else {
                    find_retries(inner, path, problems);
                }
                path.pop();
            }
        }
        toml::Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                path.push(index.to_string());
                find_retries(item, path, problems);
                path.pop();
            }
        }
        _ => {}
    }
}

/// `retries = 0`, or `retries = { count = 0, … }`.
fn is_zero_retries(value: &toml::Value) -> bool {
    match value {
        toml::Value::Integer(0) => true,
        toml::Value::Table(table) => table.get("count") == Some(&toml::Value::Integer(0)),
        _ => false,
    }
}

/// A problem for each line of a workflow that retries a step: a retry action, or a test command
/// run again after it fails.
fn workflow_retries(text: &str) -> Vec<String> {
    let mut problems = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let code = line.split(" #").next().unwrap_or(line).trim();
        if code.starts_with('#') {
            continue;
        }
        let lower = code.to_ascii_lowercase();
        let uses = lower
            .strip_prefix("- uses:")
            .or_else(|| lower.strip_prefix("uses:"));
        if uses.is_some_and(|action| action.contains("retry")) {
            problems.push(format!(
                "line {}: a retry action; CI retries no suite (RFC-0060 §13)",
                index + 1
            ));
        } else if lower.contains("cargo xtask test") && lower.contains("||") {
            problems.push(format!(
                "line {}: a suite run again after it fails; CI retries no suite (RFC-0060 §13)",
                index + 1
            ));
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_separate_infra_from_test() {
        assert_eq!(classify(End::Exited(0)), Outcome::Pass);
        assert_eq!(classify(End::Exited(1)), Outcome::Fail(Class::Test));
        assert_eq!(classify(End::Exited(2)), Outcome::Fail(Class::Test));
        assert_eq!(classify(End::Exited(3)), Outcome::Fail(Class::Infra));
        assert_eq!(classify(End::RunnerLost), Outcome::Fail(Class::Infra));
    }

    #[test]
    fn a_failure_xtask_cannot_place_is_a_test_failure() {
        // A crash in the code under test, cargo's own codes and the test harness's panic code
        // are never taken for infrastructure.
        for end in [
            End::Signalled,
            End::Exited(-1),
            End::Exited(4),
            End::Exited(101),
            End::Exited(255),
        ] {
            assert_eq!(classify(end), Outcome::Fail(Class::Test), "{end:?}");
        }
    }

    #[test]
    fn a_deterministic_suite_failure_is_never_retried() {
        for attempt in 1..=3 {
            assert_eq!(
                decide(Kind::Deterministic, Outcome::Fail(Class::Test), attempt),
                Decision::Fail
            );
        }
        // Every suite in the registry today is deterministic, so none of them is ever retried.
        for suite in SUITES {
            assert_eq!(
                suite_kind(suite.name),
                Some(Kind::Deterministic),
                "{}",
                suite.name
            );
            assert_eq!(
                decide(
                    suite_kind(suite.name).unwrap(),
                    Outcome::Fail(Class::Test),
                    1
                ),
                Decision::Fail,
                "{}",
                suite.name
            );
        }
    }

    #[test]
    fn a_non_deterministic_test_failure_is_not_retried_either() {
        assert_eq!(
            decide(Kind::NonDeterministic, Outcome::Fail(Class::Test), 1),
            Decision::Fail
        );
    }

    #[test]
    fn stat_props_re_runs_once_with_fresh_seeds() {
        assert_eq!(
            decide(Kind::StatProps, Outcome::Fail(Class::Test), 1),
            Decision::RerunWithFreshSeeds
        );
        assert_eq!(
            decide(Kind::StatProps, Outcome::Fail(Class::Test), 2),
            Decision::Fail
        );
    }

    #[test]
    fn infra_is_retried_once_on_another_rig_then_reported() {
        for kind in [Kind::Deterministic, Kind::StatProps, Kind::NonDeterministic] {
            assert_eq!(
                decide(kind, Outcome::Fail(Class::Infra), 1),
                Decision::RetryOnAnotherRig,
                "{kind:?}"
            );
            assert_eq!(
                decide(kind, Outcome::Fail(Class::Infra), 2),
                Decision::ReportToMaintainers,
                "{kind:?}"
            );
        }
    }

    #[test]
    fn a_pass_needs_nothing() {
        assert_eq!(
            decide(Kind::Deterministic, Outcome::Pass, 1),
            Decision::Pass
        );
    }

    #[test]
    fn decision_lines_say_why() {
        assert_eq!(
            decision_line("unit", Class::Test),
            "not retried: `unit` is a deterministic suite, and a failure in it is a failure \
             (RFC-0060 §13)"
        );
        assert!(decision_line("unit", Class::Infra).starts_with("infra failure (exit 3): "));
        assert!(decision_line("no-such-suite", Class::Test).starts_with("not retried: "));
    }

    #[test]
    fn a_suite_without_a_kind_fails_the_step() {
        assert!(unclassified(SUITES.iter().map(|e| e.name)).is_empty());
        let problems = unclassified(["unit", "lab-soak"].into_iter());
        assert_eq!(problems.len(), 1);
        assert!(
            problems[0].starts_with("suite `lab-soak` has no kind"),
            "{problems:?}"
        );
    }

    #[test]
    fn nextest_retries_must_be_zero() {
        assert!(nextest_retries("[profile.default]\nretries = 0\n").is_empty());
        assert!(
            nextest_retries("[profile.ci]\nretries = { backoff = \"fixed\", count = 0 }\n")
                .is_empty()
        );
        let problems = nextest_retries("[profile.ci]\nretries = 2\n");
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("`profile.ci.retries`"), "{problems:?}");
        let problems = nextest_retries(
            "[[profile.default.overrides]]\nfilter = \"test(x)\"\nretries = { count = 1 }\n",
        );
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].contains("profile.default.overrides.0.retries"));
    }

    #[test]
    fn a_retry_action_or_a_re_run_in_a_workflow_fails() {
        let clean = "jobs:\n  t:\n    steps:\n      - uses: actions/checkout@abc # v4\n      \
                     - run: cargo xtask test unit\n      # - uses: some/retry-action@v1\n";
        assert!(workflow_retries(clean).is_empty());
        let retry = "    steps:\n      - uses: some-org/retry-action@0123 # v3\n";
        let problems = workflow_retries(retry);
        assert_eq!(problems.len(), 1);
        assert!(
            problems[0].starts_with("line 2: a retry action"),
            "{problems:?}"
        );
        let rerun = "      - run: cargo xtask test unit || cargo xtask test unit\n";
        assert_eq!(workflow_retries(rerun).len(), 1);
    }

    #[test]
    fn the_repository_configures_no_retry() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let context = Context {
            root: root.to_path_buf(),
            args: crate::cli::Args::default(),
        };
        assert_eq!(step(&context), Ok(()));
    }
}
