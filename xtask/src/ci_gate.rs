// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci ci-gate --needs <file>`: the one required check that stands for every
//! conditional job of `pr.yml` (RFC-0062 §2.3 and §2.5, ADR-0620 decision 3).
//!
//! The `ci-gate` job of `pr.yml` waits for every other job and writes its `needs` context to a
//! file with `${{ toJSON(needs) }}`: an object with one member per job, each holding the job's
//! `result` (`success`, `failure`, `cancelled` or `skipped`) and its `outputs`. The `changes` job
//! has a `jobs` output: a JSON array of the IDs of the jobs this run has to run.
//!
//! The gate passes when all of these hold:
//! - `changes` succeeded, and its `jobs` output is an array of IDs of jobs `ci-gate` waits for;
//! - every job `changes` listed succeeded;
//! - every other job succeeded or was skipped.
//!
//! So a failed or cancelled job fails the gate, and a skipped job passes only when `changes` did
//! not list it. A listed job that was skipped, such as one whose `if:` is wrong, fails, and so does
//! a failed `changes`, even though every job after it was skipped. A job that `changes` lists but
//! `ci-gate` does not wait for fails too, so a new job cannot be left out of the gate.
//!
//! The hardware clause, a passing `hw/<job>` status for each job in `hw_required` with its rules
//! per event, is QUAL-E1-T07's.

use serde_json::{Map, Value};

use crate::run::Context;

/// The job that decides which jobs run (RFC-0062 §2.2).
pub const CHANGES: &str = "changes";

/// The output of [`CHANGES`] that lists them.
pub const JOBS_OUTPUT: &str = "jobs";

/// A job's result, as the `needs` context reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Success,
    Failure,
    Cancelled,
    Skipped,
}

impl Outcome {
    fn parse(result: &str) -> Option<Outcome> {
        match result {
            "success" => Some(Outcome::Success),
            "failure" => Some(Outcome::Failure),
            "cancelled" => Some(Outcome::Cancelled),
            "skipped" => Some(Outcome::Skipped),
            _ => None,
        }
    }
}

/// What the gate found: one report line per job, and every reason it fails.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Evaluation {
    pub lines: Vec<String>,
    pub problems: Vec<String>,
}

/// Evaluates the `needs` context in `text`, as `${{ toJSON(needs) }}` writes it.
pub fn evaluate(text: &str) -> Evaluation {
    let mut evaluation = Evaluation::default();
    let needs: Map<String, Value> = match serde_json::from_str(text) {
        Ok(Value::Object(needs)) => needs,
        Ok(_) => {
            evaluation
                .problems
                .push("the `needs` context is not a JSON object".to_owned());
            return evaluation;
        }
        Err(error) => {
            evaluation
                .problems
                .push(format!("the `needs` context is not JSON: {error}"));
            return evaluation;
        }
    };

    let mut outcomes = Vec::new();
    for (job, value) in &needs {
        let result = value.get("result").and_then(Value::as_str);
        match result.and_then(Outcome::parse) {
            Some(outcome) => outcomes.push((job.as_str(), outcome)),
            None => evaluation.problems.push(format!(
                "`{job}` has the result {}, not success, failure, cancelled or skipped",
                result.map_or("none".to_owned(), |r| format!("`{r}`"))
            )),
        }
    }

    // By name, so the report reads the same whatever order the map keeps.
    outcomes.sort_by_key(|(job, _)| *job);
    let listed = listed_jobs(&needs, &outcomes, &mut evaluation.problems);
    for (job, outcome) in outcomes {
        if job == CHANGES {
            continue;
        }
        let is_listed = listed.as_ref().is_some_and(|listed| listed.contains(&job));
        let line = match (outcome, is_listed) {
            (Outcome::Success, true) => format!("{job}: success"),
            (Outcome::Success, false) => format!("{job}: success (not listed by `{CHANGES}`)"),
            (Outcome::Skipped, false) => format!("{job}: skipped, not listed by `{CHANGES}`"),
            (Outcome::Skipped, true) => {
                evaluation.problems.push(format!(
                    "`{job}` was skipped, but `{CHANGES}` listed it, so it had to run"
                ));
                format!("{job}: skipped, though listed")
            }
            (Outcome::Failure, _) => {
                evaluation.problems.push(format!("`{job}` failed"));
                format!("{job}: failure")
            }
            (Outcome::Cancelled, _) => {
                evaluation.problems.push(format!("`{job}` was cancelled"));
                format!("{job}: cancelled")
            }
        };
        evaluation.lines.push(line);
    }
    evaluation
}

/// The jobs `changes` listed, or `None` when that list cannot be trusted, with the reason added
/// to `problems`. Every listed job must be one `ci-gate` waits for.
fn listed_jobs<'a>(
    needs: &'a Map<String, Value>,
    outcomes: &[(&str, Outcome)],
    problems: &mut Vec<String>,
) -> Option<Vec<&'a str>> {
    let Some(changes) = needs.get(CHANGES) else {
        problems.push(format!(
            "`ci-gate` does not wait for `{CHANGES}`, so it cannot tell which jobs had to run"
        ));
        return None;
    };
    match outcomes.iter().find(|(job, _)| *job == CHANGES) {
        Some((_, Outcome::Success)) => {}
        Some((_, outcome)) => {
            problems.push(format!(
                "`{CHANGES}` did not succeed ({}), so which jobs had to run is unknown",
                match outcome {
                    Outcome::Failure => "it failed",
                    Outcome::Cancelled => "it was cancelled",
                    _ => "it was skipped",
                }
            ));
            return None;
        }
        // Its result was malformed, which is already a problem.
        None => return None,
    }
    let Some(output) = changes
        .get("outputs")
        .and_then(|outputs| outputs.get(JOBS_OUTPUT))
        .and_then(Value::as_str)
    else {
        problems.push(format!("`{CHANGES}` has no `{JOBS_OUTPUT}` output"));
        return None;
    };
    let names: Vec<String> = match serde_json::from_str(output) {
        Ok(names) => names,
        Err(_) => {
            problems.push(format!(
                "the `{JOBS_OUTPUT}` output of `{CHANGES}` is not a JSON array of job IDs: \
                 `{output}`"
            ));
            return None;
        }
    };
    // A list with a bad entry still names jobs that had to run, so they are held to it.
    let mut listed: Vec<&'a str> = Vec::new();
    for name in &names {
        match needs.get_key_value(name.as_str()) {
            Some((job, _)) if job == CHANGES => {
                problems.push(format!("`{CHANGES}` lists itself as a job to run"));
            }
            Some((job, _)) => {
                if listed.contains(&job.as_str()) {
                    problems.push(format!("`{CHANGES}` lists `{job}` twice"));
                } else {
                    listed.push(job.as_str());
                }
            }
            None => problems.push(format!(
                "`{CHANGES}` lists `{name}`, which `ci-gate` does not wait for; add it to the \
                 `needs` of `ci-gate`"
            )),
        }
    }
    Some(listed)
}

/// `cargo xtask ci ci-gate --needs <file>`. Prints one line per job, then fails with every
/// problem found.
pub fn step(context: &Context) -> Result<(), String> {
    let path = context
        .args
        .get("--needs")
        .ok_or_else(|| "`--needs` is required".to_owned())?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("--needs {path}: {e}"))?;
    let evaluation = evaluate(&text);
    for line in &evaluation.lines {
        eprintln!("ci-gate: {line}");
    }
    if evaluation.problems.is_empty() {
        Ok(())
    } else {
        Err(evaluation.problems.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `needs` context with `changes` listing `listed`, then each of `jobs` with its result.
    fn needs(changes: &str, listed: &str, jobs: &[(&str, &str)]) -> String {
        let mut needs = Map::new();
        needs.insert(
            CHANGES.to_owned(),
            serde_json::json!({ "result": changes, "outputs": { JOBS_OUTPUT: listed } }),
        );
        for (job, result) in jobs {
            needs.insert(
                (*job).to_owned(),
                serde_json::json!({ "result": result, "outputs": {} }),
            );
        }
        Value::Object(needs).to_string()
    }

    const ALL: &str = r#"["lint","build","hosted-test"]"#;

    #[test]
    fn every_listed_job_passing_and_the_rest_skipped_passes() {
        let evaluation = evaluate(&needs(
            "success",
            ALL,
            &[
                ("lint", "success"),
                ("build", "success"),
                ("hosted-test", "success"),
                ("hosted-l3", "skipped"),
                ("hosted-parity-nano", "skipped"),
            ],
        ));
        assert_eq!(evaluation.problems, Vec::<String>::new());
        assert_eq!(
            evaluation.lines,
            [
                "build: success",
                "hosted-l3: skipped, not listed by `changes`",
                "hosted-parity-nano: skipped, not listed by `changes`",
                "hosted-test: success",
                "lint: success",
            ]
        );
    }

    #[test]
    fn a_failed_job_fails_the_gate() {
        let evaluation = evaluate(&needs(
            "success",
            ALL,
            &[
                ("lint", "success"),
                ("build", "failure"),
                ("hosted-test", "success"),
            ],
        ));
        assert_eq!(evaluation.problems, ["`build` failed"]);
    }

    #[test]
    fn a_cancelled_job_fails_the_gate() {
        let evaluation = evaluate(&needs(
            "success",
            ALL,
            &[
                ("lint", "cancelled"),
                ("build", "success"),
                ("hosted-test", "success"),
            ],
        ));
        assert_eq!(evaluation.problems, ["`lint` was cancelled"]);
    }

    #[test]
    fn an_unlisted_job_that_ran_and_failed_still_fails_the_gate() {
        let evaluation = evaluate(&needs(
            "success",
            r#"["lint"]"#,
            &[("lint", "success"), ("hosted-l3", "failure")],
        ));
        assert_eq!(evaluation.problems, ["`hosted-l3` failed"]);
    }

    #[test]
    fn a_listed_job_that_was_skipped_fails_the_gate() {
        let evaluation = evaluate(&needs(
            "success",
            ALL,
            &[
                ("lint", "success"),
                ("build", "skipped"),
                ("hosted-test", "success"),
            ],
        ));
        assert_eq!(
            evaluation.problems,
            ["`build` was skipped, but `changes` listed it, so it had to run"]
        );
    }

    #[test]
    fn a_failed_changes_fails_the_gate_though_everything_after_it_was_skipped() {
        for result in ["failure", "cancelled", "skipped"] {
            let evaluation = evaluate(&needs(
                result,
                "",
                &[
                    ("lint", "skipped"),
                    ("build", "skipped"),
                    ("hosted-test", "skipped"),
                ],
            ));
            assert_eq!(evaluation.problems.len(), 1, "{result}: {evaluation:?}");
            assert!(
                evaluation.problems[0].starts_with("`changes` did not succeed"),
                "{result}: {evaluation:?}"
            );
        }
    }

    #[test]
    fn a_gate_that_does_not_wait_for_changes_fails() {
        let text = serde_json::json!({ "lint": { "result": "success", "outputs": {} } });
        let evaluation = evaluate(&text.to_string());
        assert_eq!(
            evaluation.problems,
            ["`ci-gate` does not wait for `changes`, so it cannot tell which jobs had to run"]
        );
        assert_eq!(evaluate("{}").problems.len(), 1);
    }

    #[test]
    fn a_listed_job_the_gate_does_not_wait_for_fails() {
        let evaluation = evaluate(&needs(
            "success",
            r#"["lint","sim-pr"]"#,
            &[("lint", "success")],
        ));
        assert_eq!(
            evaluation.problems,
            [
                "`changes` lists `sim-pr`, which `ci-gate` does not wait for; add it to the \
                 `needs` of `ci-gate`"
            ]
        );
        let evaluation = evaluate(&needs("success", r#"["changes"]"#, &[]));
        assert_eq!(
            evaluation.problems,
            ["`changes` lists itself as a job to run"]
        );
    }

    #[test]
    fn a_malformed_job_list_fails() {
        for output in ["", "lint", r#"{"lint":true}"#, "[1]", r#"["lint",null]"#] {
            let evaluation = evaluate(&needs("success", output, &[("lint", "success")]));
            assert_eq!(evaluation.problems.len(), 1, "`{output}`: {evaluation:?}");
            assert!(
                evaluation.problems[0].contains("is not a JSON array of job IDs"),
                "`{output}`: {evaluation:?}"
            );
        }
        let text = serde_json::json!({ "changes": { "result": "success", "outputs": {} } });
        assert_eq!(
            evaluate(&text.to_string()).problems,
            ["`changes` has no `jobs` output"]
        );
    }

    #[test]
    fn a_job_listed_twice_fails() {
        let evaluation = evaluate(&needs(
            "success",
            r#"["lint","lint"]"#,
            &[("lint", "success")],
        ));
        assert_eq!(evaluation.problems, ["`changes` lists `lint` twice"]);
    }

    #[test]
    fn an_unknown_or_missing_result_fails() {
        let text = serde_json::json!({
            "changes": { "result": "success", "outputs": { "jobs": "[\"lint\",\"build\"]" } },
            "lint": { "result": "neutral", "outputs": {} },
            "build": { "outputs": {} },
        });
        let evaluation = evaluate(&text.to_string());
        assert_eq!(
            evaluation.problems,
            [
                "`build` has the result none, not success, failure, cancelled or skipped",
                "`lint` has the result `neutral`, not success, failure, cancelled or skipped",
            ]
        );
    }

    /// A `needs` context recorded from a real run: of `pr.yml` on PR #1577, or of a fixture
    /// workflow on the `ci-evidence/qual-e1-t06-gate` branch, on 2026-10-21.
    macro_rules! recorded {
        ($name:literal) => {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/fixtures/ci-gate/needs-",
                $name,
                ".json"
            ))
        };
    }

    #[test]
    fn recorded_runs_get_the_verdicts_they_got_in_ci() {
        // Fixture runs: a skipped unlisted job passes; a failed or timed-out job fails.
        assert_eq!(
            evaluate(recorded!("fixture-skipped")).problems,
            Vec::<String>::new()
        );
        assert_eq!(
            evaluate(recorded!("fixture-failed")).problems,
            ["`build` failed"]
        );
        assert_eq!(
            evaluate(recorded!("fixture-cancelled")).problems,
            ["`hosted-test` was cancelled"]
        );
        // `pr.yml` runs: one cancelled by a second push, one with its Windows jobs failing.
        assert_eq!(
            evaluate(recorded!("pr-cancelled-run")).problems,
            [
                "`build` was cancelled",
                "`hosted-test` was cancelled",
                "`lint` was cancelled"
            ]
        );
        assert_eq!(
            evaluate(recorded!("pr-windows-failed")).problems,
            ["`build` failed", "`hosted-test` failed"]
        );
    }

    #[test]
    fn input_that_is_not_a_needs_object_fails() {
        assert_eq!(
            evaluate("[]").problems,
            ["the `needs` context is not a JSON object"]
        );
        assert!(evaluate("not json").problems[0].starts_with("the `needs` context is not JSON"));
    }

    #[test]
    fn every_problem_is_reported_in_one_run() {
        let evaluation = evaluate(&needs(
            "success",
            r#"["lint","build","hosted-test"]"#,
            &[
                ("lint", "failure"),
                ("build", "cancelled"),
                ("hosted-test", "skipped"),
            ],
        ));
        assert_eq!(evaluation.problems.len(), 3, "{evaluation:?}");
    }
}
