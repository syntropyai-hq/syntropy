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
//! `pr.yml` runs this command from a checkout of the base branch, built with no cache, so a PR
//! cannot change how its own jobs are judged (RFC-0062 §2.2, note 2026-10-22). A change here takes
//! effect when it merges.
//!
//! **The hardware clause** (RFC-0062 §2.5's per-event table, ADR-0620 decision 3; QUAL-E1-T07).
//! `changes` also outputs `hw_required`, the hardware jobs whose direct rules the change meets,
//! and `--event` is the run's event payload:
//! - on `pull_request`, each job in `hw_required` needs a passing `hw/<job>` commit status on the
//!   PR's head commit, posted by an account `ci/jobs.toml` names in `hardware.status_creators`,
//!   the lab dispatcher's. The latest such status counts. A fork PR waits for every job in the
//!   set (ADR-0620 decision 4). A same-repository PR is not held for a job `ci/jobs.toml` marks
//!   `planned`, one whose rig and `lab.yml` job do not exist yet; the summary names the task that
//!   brings it, which removes the mark. `--statuses` holds `GET /commits/{head}/statuses`, every
//!   page;
//! - on `merge_group`, the gate evaluates hosted jobs only: hardware was required on each member
//!   PR's head. QUAL-E1-T11 adds the `hw:queue` exception.
//!
//! The planned hosted jobs, whose suites later tasks add, are `ci/jobs.toml`'s too: the summary
//! names each one's task beside its skip.
//!
//! **Paused jobs** (ADR-0626; QUAL-E1-T37). A hosted job `ci/jobs.toml` marks `paused` never
//! runs: `changes` does not list it, so `pr.yml` skips it, and `ci-gate` does not wait for it.
//! During M0a these are the macOS and Windows jobs. The base branch's rules decide which jobs are
//! paused, so a PR cannot pause its own failing job, and a listed job that `ci-gate` does not wait
//! for still fails it.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::changes::{Config, Runner};
use crate::run::Context;

/// The job that decides which jobs run (RFC-0062 §2.2).
pub const CHANGES: &str = "changes";

/// The output of [`CHANGES`] that lists them.
pub const JOBS_OUTPUT: &str = "jobs";

/// The output of [`CHANGES`] that lists the hardware jobs the change requires.
pub const HW_OUTPUT: &str = "hw_required";

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

/// The run's event, from its payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    PullRequest {
        /// The PR's head commit, which hardware statuses are posted on.
        head: String,
        /// Whether the head is in another repository, or in one since deleted.
        fork: bool,
    },
    MergeGroup,
}

/// Reads a `pull_request` or `merge_group` event payload.
pub fn parse_event(text: &str) -> Result<Event, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| format!("event: not JSON: {e}"))?;
    if value.get("merge_group").is_some_and(Value::is_object) {
        return Ok(Event::MergeGroup);
    }
    let pr = value
        .get("pull_request")
        .filter(|pr| pr.is_object())
        .ok_or_else(|| "event: neither a `pull_request` nor a `merge_group` payload".to_owned())?;
    let head = pr
        .pointer("/head/sha")
        .and_then(Value::as_str)
        .filter(|sha| sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit()))
        .ok_or_else(|| "event: `pull_request.head.sha` is not a commit SHA".to_owned())?;
    let base_repo = pr
        .pointer("/base/repo/full_name")
        .and_then(Value::as_str)
        .ok_or_else(|| "event: `pull_request.base.repo.full_name` is missing".to_owned())?;
    let head_repo = pr.pointer("/head/repo/full_name").and_then(Value::as_str);
    Ok(Event::PullRequest {
        head: head.to_owned(),
        fork: head_repo != Some(base_repo),
    })
}

/// One commit status, from `GET /repos/{owner}/{repo}/commits/{ref}/statuses`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Status {
    pub id: u64,
    pub context: String,
    /// `success`, `failure`, `error` or `pending`.
    pub state: String,
    /// The account that posted it.
    pub creator: Option<String>,
    /// The commit it is on, from its `url`.
    pub sha: String,
}

/// Reads the statuses `gh api --paginate` writes: one JSON array per page, back to back, or a
/// single array of pages.
pub fn parse_statuses(text: &str) -> Result<Vec<Status>, String> {
    let mut statuses = Vec::new();
    for page in serde_json::Deserializer::from_str(text).into_iter::<Value>() {
        let page = page.map_err(|e| format!("statuses: not JSON: {e}"))?;
        let items = page
            .as_array()
            .ok_or_else(|| "statuses: a page is not a JSON array".to_owned())?;
        for item in items {
            if item.is_array() {
                // `--slurp` wraps the pages in one outer array.
                statuses.extend(parse_statuses(&item.to_string())?);
                continue;
            }
            let field = |name: &str| item.get(name).and_then(Value::as_str);
            let url = field("url").ok_or_else(|| "statuses: a status has no `url`".to_owned())?;
            statuses.push(Status {
                id: item
                    .get("id")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| "statuses: a status has no numeric `id`".to_owned())?,
                context: field("context")
                    .ok_or_else(|| "statuses: a status has no `context`".to_owned())?
                    .to_owned(),
                state: field("state")
                    .ok_or_else(|| "statuses: a status has no `state`".to_owned())?
                    .to_owned(),
                creator: item
                    .pointer("/creator/login")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                sha: url.rsplit('/').next().unwrap_or_default().to_owned(),
            });
        }
    }
    Ok(statuses)
}

/// Evaluates the `needs` context in `text`, as `${{ toJSON(needs) }}` writes it, for a run on
/// `event` with the rules of `config`. `statuses` are the head commit's, on a `pull_request`.
pub fn evaluate(
    text: &str,
    config: &Config,
    event: &Event,
    statuses: Option<&[Status]>,
) -> Evaluation {
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
    // By name, so the report reads the same whatever order the map keeps: `serde_json`'s
    // `preserve_order` keeps document order (RFC-0060 §2.4 rule T4).
    let mut jobs: Vec<(&String, &Value)> = needs.iter().collect();
    jobs.sort_by_key(|(job, _)| *job);
    for (job, value) in jobs {
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
            (Outcome::Skipped, false) => {
                let rule = config.job(job);
                match (
                    rule.and_then(|j| j.planned.as_ref()),
                    rule.and_then(|j| j.paused.as_ref()),
                ) {
                    (Some(tasks), _) => format!("{job}: skipped, planned: {tasks}"),
                    (None, Some(adr)) => format!("{job}: skipped, paused: {adr}"),
                    (None, None) => format!("{job}: skipped, not listed by `{CHANGES}`"),
                }
            }
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
    if listed.is_some() {
        hardware(&needs, config, event, statuses, &mut evaluation);
    }
    evaluation
}

/// The hardware clause: each job in `hw_required` and its `hw/<job>` status, by event.
fn hardware(
    needs: &Map<String, Value>,
    config: &Config,
    event: &Event,
    statuses: Option<&[Status]>,
    evaluation: &mut Evaluation,
) {
    let Some(output) = needs
        .get(CHANGES)
        .and_then(|changes| changes.get("outputs"))
        .and_then(|outputs| outputs.get(HW_OUTPUT))
        .and_then(Value::as_str)
    else {
        evaluation
            .problems
            .push(format!("`{CHANGES}` has no `{HW_OUTPUT}` output"));
        return;
    };
    let Ok(names) = serde_json::from_str::<Vec<String>>(output) else {
        evaluation.problems.push(format!(
            "the `{HW_OUTPUT}` output of `{CHANGES}` is not a JSON array of job IDs: `{output}`"
        ));
        return;
    };
    let mut required = BTreeSet::new();
    for name in &names {
        match config.job(name) {
            Some(job) if job.runner == Runner::Lab => {
                if !required.insert(name.as_str()) {
                    evaluation
                        .problems
                        .push(format!("`{CHANGES}` lists `{name}` twice in `{HW_OUTPUT}`"));
                }
            }
            _ => evaluation.problems.push(format!(
                "`{CHANGES}` requires `{name}`, which is not a hardware job of {}",
                crate::changes::CONFIG
            )),
        }
    }
    let (head, fork) = match event {
        Event::MergeGroup => {
            for name in &required {
                evaluation.lines.push(format!(
                    "{name}: not evaluated on a merge group; it was required on each member's \
                     head (RFC-0062 §2.5)"
                ));
            }
            return;
        }
        Event::PullRequest { head, fork } => (head.as_str(), *fork),
    };
    let short = &head[..12];
    let Some(statuses) = statuses else {
        if !required.is_empty() {
            evaluation.problems.push(
                "`--statuses` is required on a pull_request whose `hw_required` is not empty"
                    .to_owned(),
            );
        }
        return;
    };
    if let Some(other) = statuses.iter().find(|s| s.sha != head) {
        evaluation.problems.push(format!(
            "the statuses are for commit `{}`, not the PR's head `{short}`",
            other.sha
        ));
        return;
    }
    for name in required {
        let planned = config.job(name).and_then(|job| job.planned.as_deref());
        if let (Some(task), false) = (planned, fork) {
            evaluation.lines.push(format!(
                "{name}: required, but planned ({task}): a same-repository PR is not held for it \
                 until its rig runs it"
            ));
            continue;
        }
        let context = format!("hw/{name}");
        let posted: Vec<&Status> = statuses.iter().filter(|s| s.context == context).collect();
        let trusted = posted
            .iter()
            .filter(|s| {
                s.creator
                    .as_ref()
                    .is_some_and(|c| config.status_creators.contains(c))
            })
            .max_by_key(|s| s.id);
        match trusted {
            Some(status) if status.state == "success" => {
                evaluation
                    .lines
                    .push(format!("{name}: `{context}` passed on {short}"));
            }
            Some(status) => {
                evaluation.lines.push(format!("{name}: {}", status.state));
                evaluation.problems.push(format!(
                    "hardware run required: the latest `{context}` status on {short} is `{}`",
                    status.state
                ));
            }
            None => {
                evaluation
                    .lines
                    .push(format!("{name}: no `{context}` status"));
                let why = if config.status_creators.is_empty() {
                    format!(
                        "; {} names no lab dispatcher in `hardware.status_creators` yet",
                        crate::changes::CONFIG
                    )
                } else if posted.is_empty() {
                    String::new()
                } else {
                    format!(
                        "; {} posted by accounts {} does not name were ignored",
                        posted.len(),
                        crate::changes::CONFIG
                    )
                };
                evaluation.problems.push(format!(
                    "hardware run required: no passing `{context}` status on {short}{why}"
                ));
            }
        }
    }
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

/// `cargo xtask ci ci-gate --needs <file> --event <file> [--statuses <file>]`. Prints one line per
/// job, then fails with every problem found. The rules come from the checkout's `ci/jobs.toml`.
pub fn step(context: &Context) -> Result<(), String> {
    let read = |option: &str| -> Result<String, String> {
        let path = context
            .args
            .get(option)
            .ok_or_else(|| format!("`{option}` is required"))?;
        std::fs::read_to_string(path).map_err(|e| format!("{option} {path}: {e}"))
    };
    let text = read("--needs")?;
    let event = parse_event(&read("--event")?)?;
    let statuses = match context.args.get("--statuses") {
        Some(_) => Some(parse_statuses(&read("--statuses")?)?),
        None => None,
    };
    let config = Config::load(&context.root)?;
    let evaluation = evaluate(&text, &config, &event, statuses.as_deref());
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

    /// The rules the tests judge by: the jobs `pr.yml` waits for, one of them planned, two paused
    /// jobs it does not wait for, and two hardware jobs, one planned, whose statuses
    /// `lab-bot[bot]` posts.
    const CONFIG: &str = r#"
[paths]
workspace = ["Cargo.lock"]
no_crate = ["docs/**"]

[hardware]
status_creators = ["lab-bot[bot]"]

[job.lint]
runner = "hosted"
when = "always"

[job.build]
runner = "hosted"
when = "closure"

[job.hosted-test]
runner = "hosted"
when = "closure"

[job.hosted-test-macos-windows]
runner = "hosted"
when = "closure"
paused = "ADR-0626"

[job.toolchains-macos]
runner = "hosted"
when = "direct"
paths = ["ci/toolchains.toml"]
paused = "ADR-0626"

[job.hosted-l3]
runner = "hosted"
when = "closure"
crates = ["net"]

[job.hosted-parity-nano]
runner = "hosted"
when = "closure"
planned = "QUAL-E5-T06"

[job.hw-cuda-ada]
runner = "lab"
when = "direct"
paths = ["crates/kernels-*/**"]

[job.hw-metal-floor]
runner = "lab"
when = "direct"
paths = ["crates/kernels-*/**"]
planned = "QUAL-E2-T99"
"#;

    fn config() -> Config {
        Config::parse(CONFIG).expect("the test rules parse")
    }

    const HEAD: &str = "0123456789abcdef0123456789abcdef01234567";

    fn same_repo() -> Event {
        Event::PullRequest {
            head: HEAD.to_owned(),
            fork: false,
        }
    }

    fn fork() -> Event {
        Event::PullRequest {
            head: HEAD.to_owned(),
            fork: true,
        }
    }

    /// The hosted verdict: a same-repository PR that requires no hardware.
    fn hosted(text: &str) -> Evaluation {
        evaluate(text, &config(), &same_repo(), Some(&[]))
    }

    /// The problem a `needs` context recorded before `changes` had a `hw_required` output gives.
    const NO_HW_OUTPUT: &str = "`changes` has no `hw_required` output";

    /// A `needs` context with `changes` listing `listed`, then each of `jobs` with its result.
    fn needs(changes: &str, listed: &str, jobs: &[(&str, &str)]) -> String {
        needs_hw(changes, listed, "[]", jobs)
    }

    /// As [`needs`], with `hw` as the `hw_required` output.
    fn needs_hw(changes: &str, listed: &str, hw: &str, jobs: &[(&str, &str)]) -> String {
        let mut needs = Map::new();
        needs.insert(
            CHANGES.to_owned(),
            serde_json::json!({
                "result": changes,
                "outputs": { JOBS_OUTPUT: listed, HW_OUTPUT: hw },
            }),
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
        let evaluation = hosted(&needs(
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
                "hosted-parity-nano: skipped, planned: QUAL-E5-T06",
                "hosted-test: success",
                "lint: success",
            ]
        );
    }

    #[test]
    fn a_failed_job_fails_the_gate() {
        let evaluation = hosted(&needs(
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
        let evaluation = hosted(&needs(
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
        let evaluation = hosted(&needs(
            "success",
            r#"["lint"]"#,
            &[("lint", "success"), ("hosted-l3", "failure")],
        ));
        assert_eq!(evaluation.problems, ["`hosted-l3` failed"]);
    }

    #[test]
    fn a_listed_job_that_was_skipped_fails_the_gate() {
        let evaluation = hosted(&needs(
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
            let evaluation = hosted(&needs(
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
        let evaluation = hosted(&text.to_string());
        assert_eq!(
            evaluation.problems,
            ["`ci-gate` does not wait for `changes`, so it cannot tell which jobs had to run"]
        );
        assert_eq!(hosted("{}").problems.len(), 1);
    }

    #[test]
    fn a_listed_job_the_gate_does_not_wait_for_fails() {
        let evaluation = hosted(&needs(
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
        let evaluation = hosted(&needs("success", r#"["changes"]"#, &[]));
        assert_eq!(
            evaluation.problems,
            ["`changes` lists itself as a job to run"]
        );
    }

    #[test]
    fn the_paused_macos_and_windows_jobs_do_not_hold_the_gate_but_a_linux_failure_fails_it() {
        let listed = r#"["lint","hosted-test"]"#;
        // Not waited for at all, as `pr.yml` has it.
        let evaluation = hosted(&needs(
            "success",
            listed,
            &[("lint", "success"), ("hosted-test", "success")],
        ));
        assert_eq!(evaluation.problems, Vec::<String>::new());
        // Waited for and skipped, as they would be if `ci-gate` named them.
        let evaluation = hosted(&needs(
            "success",
            listed,
            &[
                ("lint", "success"),
                ("hosted-test", "success"),
                ("hosted-test-macos-windows", "skipped"),
                ("toolchains-macos", "skipped"),
            ],
        ));
        assert_eq!(evaluation.problems, Vec::<String>::new());
        assert!(
            evaluation
                .lines
                .contains(&"hosted-test-macos-windows: skipped, paused: ADR-0626".to_owned()),
            "{evaluation:?}"
        );
        // `hosted-test` is the Linux leg, which still gates, on a PR and on a merge group.
        let failing = needs(
            "success",
            listed,
            &[("lint", "success"), ("hosted-test", "failure")],
        );
        assert_eq!(hosted(&failing).problems, ["`hosted-test` failed"]);
        assert_eq!(
            evaluate(&failing, &config(), &Event::MergeGroup, None).problems,
            ["`hosted-test` failed"]
        );
    }

    #[test]
    fn a_paused_job_that_ran_and_failed_still_fails_the_gate() {
        // `changes` never lists a paused job, so it can run only if a PR's workflow runs it
        // anyway; a failure it reports is not hidden.
        let evaluation = hosted(&needs(
            "success",
            r#"["lint"]"#,
            &[("lint", "success"), ("toolchains-macos", "failure")],
        ));
        assert_eq!(evaluation.problems, ["`toolchains-macos` failed"]);
        // Nor can `changes` list one that `ci-gate` does not wait for.
        let evaluation = hosted(&needs(
            "success",
            r#"["lint","hosted-test-macos-windows"]"#,
            &[("lint", "success")],
        ));
        assert_eq!(
            evaluation.problems,
            [
                "`changes` lists `hosted-test-macos-windows`, which `ci-gate` does not wait for; \
              add it to the `needs` of `ci-gate`"
            ]
        );
    }

    #[test]
    fn on_the_real_rules_only_the_macos_and_windows_jobs_are_paused() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let rules = Config::load(root).expect("ci/jobs.toml parses");
        let paused: Vec<(&str, &str)> = rules
            .jobs
            .iter()
            .filter_map(|job| Some((job.name.as_str(), job.paused.as_deref()?)))
            .collect();
        assert_eq!(
            paused,
            [
                ("build-macos-windows", "ADR-0626"),
                ("hosted-test-macos-windows", "ADR-0626"),
                ("toolchains-macos", "ADR-0626"),
                ("toolchains-windows", "ADR-0626"),
            ]
        );
    }

    #[test]
    fn a_malformed_job_list_fails() {
        for output in ["", "lint", r#"{"lint":true}"#, "[1]", r#"["lint",null]"#] {
            let evaluation = hosted(&needs("success", output, &[("lint", "success")]));
            assert_eq!(evaluation.problems.len(), 1, "`{output}`: {evaluation:?}");
            assert!(
                evaluation.problems[0].contains("is not a JSON array of job IDs"),
                "`{output}`: {evaluation:?}"
            );
        }
        let text = serde_json::json!({ "changes": { "result": "success", "outputs": {} } });
        assert_eq!(
            hosted(&text.to_string()).problems,
            ["`changes` has no `jobs` output"]
        );
    }

    #[test]
    fn a_job_listed_twice_fails() {
        let evaluation = hosted(&needs(
            "success",
            r#"["lint","lint"]"#,
            &[("lint", "success")],
        ));
        assert_eq!(evaluation.problems, ["`changes` lists `lint` twice"]);
    }

    #[test]
    fn an_unknown_or_missing_result_fails() {
        let text = serde_json::json!({
            "changes": {
                "result": "success",
                "outputs": { "jobs": "[\"lint\",\"build\"]", "hw_required": "[]" },
            },
            "lint": { "result": "neutral", "outputs": {} },
            "build": { "outputs": {} },
        });
        let evaluation = hosted(&text.to_string());
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
        // The runs predate `hw_required`, so each also fails for its absence: a `changes` that
        // does not say which hardware a change needs cannot pass the gate.
        // Fixture runs: a skipped unlisted job passes; a failed or timed-out job fails.
        assert_eq!(
            hosted(recorded!("fixture-skipped")).problems,
            [NO_HW_OUTPUT]
        );
        assert_eq!(
            hosted(recorded!("fixture-failed")).problems,
            ["`build` failed", NO_HW_OUTPUT]
        );
        assert_eq!(
            hosted(recorded!("fixture-cancelled")).problems,
            ["`hosted-test` was cancelled", NO_HW_OUTPUT]
        );
        // `pr.yml` runs: one cancelled by a second push, one with its Windows jobs failing.
        assert_eq!(
            hosted(recorded!("pr-cancelled-run")).problems,
            [
                "`build` was cancelled",
                "`hosted-test` was cancelled",
                "`lint` was cancelled",
                NO_HW_OUTPUT,
            ]
        );
        assert_eq!(
            hosted(recorded!("pr-windows-failed")).problems,
            ["`build` failed", "`hosted-test` failed", NO_HW_OUTPUT]
        );
    }

    #[test]
    fn input_that_is_not_a_needs_object_fails() {
        assert_eq!(
            hosted("[]").problems,
            ["the `needs` context is not a JSON object"]
        );
        assert!(hosted("not json").problems[0].starts_with("the `needs` context is not JSON"));
    }

    #[test]
    fn every_problem_is_reported_in_one_run() {
        let evaluation = hosted(&needs(
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

    /// A status as the REST API returns it, on `sha`.
    fn status(id: u64, job: &str, state: &str, creator: &str, sha: &str) -> Value {
        serde_json::json!({
            "id": id,
            "url": format!("https://api.github.com/repos/syntropyai-hq/syntropy/statuses/{sha}"),
            "state": state,
            "context": format!("hw/{job}"),
            "creator": { "login": creator, "type": "Bot" },
        })
    }

    fn statuses(items: &[Value]) -> Vec<Status> {
        parse_statuses(&Value::Array(items.to_vec()).to_string()).expect("statuses parse")
    }

    /// Every hosted job passed, and `changes` required `hw`.
    fn passed_requiring(hw: &str) -> String {
        needs_hw(
            "success",
            ALL,
            hw,
            &[
                ("lint", "success"),
                ("build", "success"),
                ("hosted-test", "success"),
                ("hosted-l3", "skipped"),
                ("hosted-parity-nano", "skipped"),
            ],
        )
    }

    const CUDA: &str = r#"["hw-cuda-ada"]"#;
    const BOTH: &str = r#"["hw-cuda-ada","hw-metal-floor"]"#;

    #[test]
    fn a_pull_request_needs_a_passing_status_for_each_required_hardware_job() {
        let text = passed_requiring(CUDA);
        let none = evaluate(&text, &config(), &same_repo(), Some(&[]));
        assert_eq!(
            none.problems,
            ["hardware run required: no passing `hw/hw-cuda-ada` status on 0123456789ab"]
        );
        assert!(
            none.lines
                .contains(&"hw-cuda-ada: no `hw/hw-cuda-ada` status".to_owned())
        );
        let passed = statuses(&[status(7, "hw-cuda-ada", "success", "lab-bot[bot]", HEAD)]);
        let ok = evaluate(&text, &config(), &same_repo(), Some(&passed));
        assert_eq!(ok.problems, Vec::<String>::new());
        assert!(
            ok.lines
                .contains(&"hw-cuda-ada: `hw/hw-cuda-ada` passed on 0123456789ab".to_owned())
        );
    }

    #[test]
    fn only_the_latest_status_from_the_dispatcher_counts() {
        let text = passed_requiring(CUDA);
        let judge = |items: &[Value]| {
            evaluate(&text, &config(), &same_repo(), Some(&statuses(items))).problems
        };
        // A failure after a success fails; a success after a failure passes.
        assert_eq!(
            judge(&[
                status(1, "hw-cuda-ada", "success", "lab-bot[bot]", HEAD),
                status(2, "hw-cuda-ada", "failure", "lab-bot[bot]", HEAD),
            ]),
            [
                "hardware run required: the latest `hw/hw-cuda-ada` status on 0123456789ab is \
              `failure`"
            ]
        );
        assert_eq!(
            judge(&[
                status(2, "hw-cuda-ada", "success", "lab-bot[bot]", HEAD),
                status(1, "hw-cuda-ada", "failure", "lab-bot[bot]", HEAD),
            ]),
            Vec::<String>::new()
        );
        // A run still going does not pass.
        assert_eq!(
            judge(&[status(3, "hw-cuda-ada", "pending", "lab-bot[bot]", HEAD)]),
            [
                "hardware run required: the latest `hw/hw-cuda-ada` status on 0123456789ab is \
              `pending`"
            ]
        );
        // A status from anyone else is ignored, even a later passing one.
        assert_eq!(
            judge(&[status(9, "hw-cuda-ada", "success", "someone", HEAD)]),
            [
                "hardware run required: no passing `hw/hw-cuda-ada` status on 0123456789ab; 1 \
              posted by accounts ci/jobs.toml does not name were ignored"
            ]
        );
        // Another job's status does not stand in.
        assert_eq!(
            judge(&[status(4, "hw-metal-floor", "success", "lab-bot[bot]", HEAD)]),
            ["hardware run required: no passing `hw/hw-cuda-ada` status on 0123456789ab"]
        );
    }

    #[test]
    fn statuses_for_another_commit_fail() {
        let text = passed_requiring(CUDA);
        let other = "fedcba9876543210fedcba9876543210fedcba98";
        let problems = evaluate(
            &text,
            &config(),
            &same_repo(),
            Some(&statuses(&[status(
                1,
                "hw-cuda-ada",
                "success",
                "lab-bot[bot]",
                other,
            )])),
        )
        .problems;
        assert_eq!(
            problems,
            [format!(
                "the statuses are for commit `{other}`, not the PR's head `0123456789ab`"
            )]
        );
    }

    #[test]
    fn a_same_repository_pr_is_not_held_for_a_planned_hardware_job_but_a_fork_is() {
        let text = passed_requiring(BOTH);
        let cuda = statuses(&[status(1, "hw-cuda-ada", "success", "lab-bot[bot]", HEAD)]);
        let same = evaluate(&text, &config(), &same_repo(), Some(&cuda));
        assert_eq!(same.problems, Vec::<String>::new());
        assert!(same.lines.contains(
            &"hw-metal-floor: required, but planned (QUAL-E2-T99): a same-repository PR is not \
              held for it until its rig runs it"
                .to_owned()
        ));
        let forked = evaluate(&text, &config(), &fork(), Some(&cuda));
        assert_eq!(
            forked.problems,
            ["hardware run required: no passing `hw/hw-metal-floor` status on 0123456789ab"]
        );
    }

    #[test]
    fn a_merge_group_evaluates_hosted_jobs_only() {
        let text = passed_requiring(BOTH);
        let group = evaluate(&text, &config(), &Event::MergeGroup, None);
        assert_eq!(group.problems, Vec::<String>::new());
        assert!(
            group.lines.contains(
                &"hw-cuda-ada: not evaluated on a merge group; it was required on each member's \
              head (RFC-0062 §2.5)"
                    .to_owned()
            )
        );
        // A hosted failure still fails it.
        let failed = needs_hw(
            "success",
            ALL,
            BOTH,
            &[
                ("lint", "success"),
                ("build", "failure"),
                ("hosted-test", "success"),
            ],
        );
        assert_eq!(
            evaluate(&failed, &config(), &Event::MergeGroup, None).problems,
            ["`build` failed"]
        );
    }

    #[test]
    fn a_pull_request_that_requires_hardware_needs_its_statuses() {
        assert_eq!(
            evaluate(&passed_requiring(CUDA), &config(), &same_repo(), None).problems,
            ["`--statuses` is required on a pull_request whose `hw_required` is not empty"]
        );
        // Nothing required, nothing to read.
        assert_eq!(
            evaluate(&passed_requiring("[]"), &config(), &same_repo(), None).problems,
            Vec::<String>::new()
        );
    }

    #[test]
    fn hw_required_names_hardware_jobs_each_once() {
        let judge = |hw: &str| hosted(&passed_requiring(hw)).problems;
        assert_eq!(
            judge(r#"["hw-nowhere","lint"]"#),
            [
                "`changes` requires `hw-nowhere`, which is not a hardware job of ci/jobs.toml",
                "`changes` requires `lint`, which is not a hardware job of ci/jobs.toml",
            ]
        );
        assert_eq!(
            judge(r#"["hw-metal-floor","hw-metal-floor"]"#),
            ["`changes` lists `hw-metal-floor` twice in `hw_required`"]
        );
        assert_eq!(
            judge("hw-cuda-ada"),
            ["the `hw_required` output of `changes` is not a JSON array of job IDs: `hw-cuda-ada`"]
        );
    }

    #[test]
    fn with_no_dispatcher_named_no_status_counts() {
        let mut rules = config();
        rules.status_creators.clear();
        let cuda = statuses(&[status(1, "hw-cuda-ada", "success", "lab-bot[bot]", HEAD)]);
        assert_eq!(
            evaluate(&passed_requiring(CUDA), &rules, &fork(), Some(&cuda)).problems,
            [
                "hardware run required: no passing `hw/hw-cuda-ada` status on 0123456789ab; \
              ci/jobs.toml names no lab dispatcher in `hardware.status_creators` yet"
            ]
        );
    }

    /// The criterion of QUAL-E1-T07 on the real rules: a fork PR whose diff touches
    /// `crates/kernels-cuda/` cannot pass before a passing `hw/hw-cuda-ada` status on its head,
    /// and the same diff in a merge group passes on hosted jobs alone.
    ///
    /// The two payloads are constructed to GitHub's documented webhook shape, with only the
    /// fields `parse_event` reads and a few beside them: the repository has had no fork PR and
    /// has no merge queue yet (QUAL-E1-T11), so there is no run to record them from.
    #[test]
    fn on_the_real_rules_a_fork_kernel_change_waits_for_hw_cuda_ada() {
        use crate::changes::{Files, Workspace, affected, json_list};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let mut rules = Config::load(root).expect("ci/jobs.toml parses");
        let workspace = Workspace::read(&Files(root)).expect("the workspace reads");
        let paths = ["crates/kernels-cuda/src/lib.rs".to_owned()];
        let found = affected(&rules, &workspace, &paths, &BTreeSet::new());
        assert!(
            found.hw_required.contains(&"hw-cuda-ada".to_owned()),
            "{found:?}"
        );
        let text = needs_hw(
            "success",
            &json_list(&found.jobs),
            &json_list(&found.hw_required),
            &[
                ("lint", "success"),
                ("build", "success"),
                ("hosted-test", "success"),
                ("hosted-l3", "success"),
                ("hosted-parity-nano", "skipped"),
            ],
        );
        let event = parse_event(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/ci-gate/event-pull-request-fork.json"
        )))
        .expect("the fork payload parses");
        let Event::PullRequest { head, fork: true } = &event else {
            panic!("not a fork PR: {event:?}");
        };
        // No dispatcher is named yet, so nothing it posts can count.
        let ada = |creator: &str| statuses(&[status(5, "hw-cuda-ada", "success", creator, head)]);
        let before = evaluate(&text, &rules, &event, Some(&ada("lab-bot[bot]"))).problems;
        assert!(
            before
                .iter()
                .any(|p| p.starts_with("hardware run required: no passing `hw/hw-cuda-ada`")),
            "{before:?}"
        );
        // Once the dispatcher is named, its passing status on the head clears hw-cuda-ada, and
        // only hw-cuda-ada: the other rigs' jobs still hold the fork PR.
        rules.status_creators.insert("lab-bot[bot]".to_owned());
        let after = evaluate(&text, &rules, &event, Some(&ada("lab-bot[bot]"))).problems;
        assert!(
            !after.iter().any(|p| p.contains("`hw/hw-cuda-ada`")),
            "{after:?}"
        );
        assert_eq!(after.len(), found.hw_required.len() - 1, "{after:?}");
        // The same diff in a merge group passes on its hosted jobs, with no status at all.
        let group = parse_event(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/ci-gate/event-merge-group.json"
        )))
        .expect("the merge group payload parses");
        assert_eq!(group, Event::MergeGroup);
        assert_eq!(
            evaluate(&text, &rules, &group, None).problems,
            Vec::<String>::new()
        );
    }

    #[test]
    fn events_read_the_head_and_whether_it_is_a_fork() {
        let pr = |head_repo: Value| {
            serde_json::json!({
                "action": "synchronize",
                "pull_request": {
                    "head": { "sha": HEAD, "repo": head_repo },
                    "base": { "repo": { "full_name": "syntropyai-hq/syntropy" } },
                },
            })
            .to_string()
        };
        assert_eq!(
            parse_event(&pr(
                serde_json::json!({ "full_name": "syntropyai-hq/syntropy" })
            )),
            Ok(same_repo())
        );
        assert_eq!(
            parse_event(&pr(serde_json::json!({ "full_name": "someone/syntropy" }))),
            Ok(fork())
        );
        // A deleted fork has no head repository.
        assert_eq!(parse_event(&pr(Value::Null)), Ok(fork()));
        assert!(parse_event(r#"{"action":"opened"}"#).is_err());
        assert!(
            parse_event(r#"{"pull_request":{"head":{"sha":"abc"}}}"#)
                .unwrap_err()
                .contains("not a commit SHA")
        );
    }

    #[test]
    fn statuses_read_pages_back_to_back_or_slurped() {
        let page = |id: u64| Value::Array(vec![status(id, "hw-cuda-ada", "success", "a", HEAD)]);
        let back_to_back = format!("{}\n{}", page(1), page(2));
        let slurped = Value::Array(vec![page(1), page(2)]).to_string();
        for text in [back_to_back, slurped] {
            let read = parse_statuses(&text).expect("parses");
            assert_eq!(
                read.iter().map(|s| s.id).collect::<Vec<_>>(),
                [1, 2],
                "{text}"
            );
            assert_eq!(read[0].sha, HEAD);
            assert_eq!(read[0].creator.as_deref(), Some("a"));
        }
        assert_eq!(parse_statuses(""), Ok(Vec::new()));
        assert!(parse_statuses(r#"{"id":1}"#).is_err());
        assert!(parse_statuses(r#"[{"id":1,"state":"success","context":"x"}]"#).is_err());
    }
}
