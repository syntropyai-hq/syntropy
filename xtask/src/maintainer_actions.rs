// SPDX-License-Identifier: MIT OR Apache-2.0

//! The maintainer-action queue (ADR-0021 decision 4, RFC-0062 §12.3; QUAL-E1-T17).
//!
//! The epic files hold the tasks, one table row each. A task whose task cell opens a human-only
//! step with `**Maintainer action**:`, `**Maintainer action:**` or `**Maintainer action, ...**`
//! is on the queue. A cell that only mentions the phrase, as in `is a **Maintainer action**
//! (QUAL-E1-T26)`, is not. The state of each queued task comes from the completion records in
//! `governance/maintainer-actions.toml`:
//!
//! ```toml
//! [[completed]]
//! task = "QUAL-E1-T26"
//! by = "alice"             # a maintainer in governance/maintainers.toml
//! date = "2026-11-02"      # YYYY-MM-DD
//! evidence = "ceremony record or link"
//! ```
//!
//! `cargo xtask maintainer-actions` prints the queue. The `lint` step refuses a record without
//! evidence, from a non-maintainer, for a task that is not on the queue, or a second record for
//! the same task.

use std::io::{self, Write};
use std::path::Path;

use serde::Deserialize;

use crate::codeowners::ROSTER;
use crate::delegations::Date;
use crate::roster::Roster;
use crate::run::Context;

/// Where the completion records live, relative to the workspace root.
pub const RECORDS: &str = "governance/maintainer-actions.toml";

/// The epic files, relative to the workspace root.
pub const EPICS: &str = "plan/implementation/epics";

/// How much of a task's text the queue shows.
const SUMMARY_CHARS: usize = 100;

/// One task on the queue.
#[derive(Debug, PartialEq, Eq)]
pub struct Action {
    pub task: String,
    pub summary: String,
}

/// A maintainer's record that a task's human-only step is done.
#[derive(Debug, PartialEq, Eq)]
pub struct Completion {
    pub task: String,
    pub by: String,
    pub date: Date,
    pub evidence: String,
}

/// Whether a cell carries the marker that opens a human-only step.
pub fn marks_action(task_cell: &str) -> bool {
    const MARKER: &str = "**Maintainer action";
    task_cell.match_indices(MARKER).any(|(at, _)| {
        let rest = &task_cell[at + MARKER.len()..];
        rest.starts_with("**:") || rest.starts_with(":**") || rest.starts_with(',')
    })
}

/// Whether `text` is shaped like a task ID, such as `COMP-E13-T01`.
fn is_task_id(text: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let mut parts = text.split('-');
    let (Some(area), Some(epic), Some(task), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    !area.is_empty()
        && area.bytes().all(|b| b.is_ascii_uppercase())
        && epic.strip_prefix('E').is_some_and(digits)
        && task.strip_prefix('T').is_some_and(digits)
}

/// The queued tasks of the epic texts, in the order given and row order.
pub fn queue(epics: &[&str]) -> Vec<Action> {
    let mut actions = Vec::new();
    for text in epics {
        for line in text.lines() {
            let Some(row) = line.strip_prefix("| ") else {
                continue;
            };
            let mut cells = row.split(" | ");
            let (Some(id), Some(task)) = (cells.next(), cells.next()) else {
                continue;
            };
            // The marker can sit in any cell: a few rows open their human-only step in the
            // acceptance criteria.
            if is_task_id(id.trim()) && marks_action(row) {
                actions.push(Action {
                    task: id.trim().to_owned(),
                    summary: summary(task),
                });
            }
        }
    }
    actions
}

fn summary(task_cell: &str) -> String {
    let plain = task_cell.replace("**", "");
    match plain.char_indices().nth(SUMMARY_CHARS) {
        Some((end, _)) => format!("{}...", &plain[..end]),
        None => plain,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecords {
    #[serde(default)]
    completed: Vec<RawCompletion>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCompletion {
    task: Option<String>,
    by: Option<String>,
    date: Option<String>,
    evidence: Option<String>,
}

/// Parses the completion records. `Err` holds every problem found.
pub fn parse_records(text: &str, roster: &Roster) -> Result<Vec<Completion>, Vec<String>> {
    let raw: RawRecords = toml::from_str(text).map_err(|error| vec![error.to_string()])?;
    let mut problems = Vec::new();
    let mut records = Vec::new();
    for (index, record) in raw.completed.into_iter().enumerate() {
        let at = |message: String| format!("record {}: {message}", index + 1);
        let found = problems.len();
        let task = record.task.filter(|t| !t.trim().is_empty());
        let by = record.by.filter(|b| !b.trim().is_empty());
        let evidence = record.evidence.filter(|e| !e.trim().is_empty());
        if task.is_none() {
            problems.push(at("missing `task`".into()));
        }
        if by.is_none() {
            problems.push(at("missing `by`".into()));
        }
        if evidence.is_none() {
            problems.push(at(
                "a completion record needs evidence: `evidence` is missing or empty".into(),
            ));
        }
        if let Some(task) = task.as_deref().filter(|t| !is_task_id(t)) {
            problems.push(at(format!("`{task}` is not a task ID")));
        }
        if let Some(by) = by.as_deref().filter(|b| !roster.is_maintainer(b)) {
            problems.push(at(format!("`{by}` is not a maintainer in {ROSTER}")));
        }
        let date = match record.date.as_deref() {
            None => {
                problems.push(at("missing `date`".into()));
                None
            }
            Some(text) => match Date::parse(text) {
                Ok(date) => Some(date),
                Err(why) => {
                    problems.push(at(format!("`date` {why}")));
                    None
                }
            },
        };
        if let (Some(task), Some(by), Some(date), Some(evidence)) = (task, by, date, evidence)
            && problems.len() == found
        {
            records.push(Completion {
                task,
                by,
                date,
                evidence,
            });
        }
    }
    if problems.is_empty() {
        Ok(records)
    } else {
        Err(problems)
    }
}

/// Problems between the records and the queue: a record for a task that is not queued, or two
/// records for one task.
pub fn check_records(records: &[Completion], queue: &[Action]) -> Vec<String> {
    let mut problems = Vec::new();
    for (index, record) in records.iter().enumerate() {
        if !queue.iter().any(|a| a.task == record.task) {
            problems.push(format!(
                "{} is recorded but is not a Maintainer action in the epic files",
                record.task
            ));
        }
        if records[..index].iter().any(|r| r.task == record.task) {
            problems.push(format!("{} is recorded twice", record.task));
        }
    }
    problems
}

/// The queue as text: one line per task, the evidence of a done one beneath it, then a total.
pub fn render(queue: &[Action], records: &[Completion]) -> String {
    let mut out = String::new();
    let mut done = 0;
    for action in queue {
        let record = records.iter().find(|r| r.task == action.task);
        let state = if record.is_some() { "done" } else { "open" };
        out.push_str(&format!("{}  {state}  {}\n", action.task, action.summary));
        if let Some(r) = record {
            done += 1;
            out.push_str(&format!(
                "    by {} on {}, evidence: {}\n",
                r.by, r.date, r.evidence
            ));
        }
    }
    out.push_str(&format!(
        "{} maintainer actions, {done} done\n",
        queue.len()
    ));
    out
}

/// The `.md` files of `root`'s epics directory, sorted by file name, as texts.
pub fn read_epics(root: &Path) -> Result<Vec<String>, String> {
    let dir = root.join(EPICS);
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .map_err(|error| format!("{EPICS}: {error}"))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|path| {
            std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
        })
        .collect()
}

/// Reads the queue and the checked records of the workspace at `root`.
fn load(root: &Path) -> Result<(Vec<Action>, Vec<Completion>), Vec<String>> {
    let read = |relative: &str| {
        std::fs::read_to_string(root.join(relative))
            .map_err(|error| vec![format!("{relative}: {error}")])
    };
    let roster =
        Roster::parse(&read(ROSTER)?).map_err(|error| vec![format!("{ROSTER}: {error}")])?;
    let records = parse_records(&read(RECORDS)?, &roster).map_err(|problems| {
        problems
            .into_iter()
            .map(|p| format!("{RECORDS}: {p}"))
            .collect::<Vec<_>>()
    })?;
    let epics = read_epics(root).map_err(|error| vec![error])?;
    let queue = queue(&epics.iter().map(String::as_str).collect::<Vec<_>>());
    let problems = check_records(&records, &queue);
    if problems.is_empty() {
        Ok((queue, records))
    } else {
        Err(problems
            .into_iter()
            .map(|p| format!("{RECORDS}: {p}"))
            .collect())
    }
}

/// The `lint` step: the records are well formed and match the queue.
pub fn step(context: &Context) -> Result<(), String> {
    load(&context.root)
        .map(|_| ())
        .map_err(|problems| problems.join("\n"))
}

/// `cargo xtask maintainer-actions`: prints the queue. Returns false, after printing the
/// problems instead, when the records are invalid.
pub fn command(root: &Path, out: &mut impl Write) -> io::Result<bool> {
    match load(root) {
        Ok((queue, records)) => {
            write!(out, "{}", render(&queue, &records))?;
            Ok(true)
        }
        Err(problems) => {
            for problem in problems {
                writeln!(out, "error: {problem}")?;
            }
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roster() -> Roster {
        Roster::parse("maintainers = [\"alice\", \"bob\"]\nagents = []").unwrap()
    }

    const EPIC: &str = "\
# Epics

| ID | Task | Milestone |
|---|---|---|
| QUAL-E1-T01 | **Maintainer action**: record the keys | M0a | none | ok | RFC |
| QUAL-E1-T02 | Do a thing (**Maintainer action**: approving the plan) | M0a | none | ok | RFC |
| QUAL-E1-T03 | **Maintainer action, late in M0: key custody set-up.** Provision | M0a | none | ok | RFC |
| QUAL-E1-T04 | Registry. Recording is a **Maintainer action** (QUAL-E1-T01) | M0a | none | ok | RFC |
| QUAL-E1-T05 | Plain task | M0a | none | the ceremonies are **Maintainer action** (see T01) | RFC |
| QUAL-E1-T07 | Plain | M0a | none | ends with **Maintainer action**: creating the App | RFC |
| QUAL-E1-T06 | Mentions **Maintainer actions** | M0a | none | ok | RFC |
Some prose with **Maintainer action**: here.
";

    fn ids(actions: &[Action]) -> Vec<&str> {
        actions.iter().map(|a| a.task.as_str()).collect()
    }

    fn valid(task: &str) -> String {
        format!(
            "[[completed]]\ntask = \"{task}\"\nby = \"alice\"\ndate = \"2026-11-02\"\n\
             evidence = \"ceremony record\"\n"
        )
    }

    #[test]
    fn the_queue_is_the_rows_that_open_a_step() {
        let actions = queue(&[EPIC]);
        assert_eq!(
            ids(&actions),
            ["QUAL-E1-T01", "QUAL-E1-T02", "QUAL-E1-T03", "QUAL-E1-T07"]
        );
    }

    #[test]
    fn summaries_are_cut() {
        let long = format!(
            "| QUAL-E1-T09 | **Maintainer action**: {} | M0a |",
            "x".repeat(200)
        );
        let actions = queue(&[&long]);
        assert_eq!(actions[0].summary.chars().count(), SUMMARY_CHARS + 3);
        assert!(actions[0].summary.ends_with("..."));
        assert!(!actions[0].summary.contains("**"));
    }

    #[test]
    fn records_parse_or_name_every_problem() {
        assert_eq!(
            parse_records(&valid("QUAL-E1-T01"), &roster())
                .unwrap()
                .len(),
            1
        );
        assert!(
            parse_records("# nothing yet\n", &roster())
                .unwrap()
                .is_empty()
        );
        let no_evidence = valid("QUAL-E1-T01").replace("evidence = \"ceremony record\"\n", "");
        let blank = valid("QUAL-E1-T01").replace("ceremony record", "  ");
        for text in [no_evidence, blank] {
            let problems = parse_records(&text, &roster()).unwrap_err();
            assert!(problems[0].contains("needs evidence"), "{problems:?}");
        }
        let stranger = valid("QUAL-E1-T01").replace("alice", "mallory");
        assert!(parse_records(&stranger, &roster()).unwrap_err()[0].contains("not a maintainer"));
        let bad_date = valid("QUAL-E1-T01").replace("2026-11-02", "soon");
        assert!(parse_records(&bad_date, &roster()).is_err());
        let unknown = format!("{}extra = 1\n", valid("QUAL-E1-T01"));
        assert!(parse_records(&unknown, &roster()).is_err());
    }

    #[test]
    fn records_must_match_the_queue() {
        let actions = queue(&[EPIC]);
        let ok = parse_records(&valid("QUAL-E1-T01"), &roster()).unwrap();
        assert!(check_records(&ok, &actions).is_empty());
        let stray = parse_records(&valid("QUAL-E1-T05"), &roster()).unwrap();
        assert!(check_records(&stray, &actions)[0].contains("not a Maintainer action"));
        let twice = parse_records(
            &format!("{}{}", valid("QUAL-E1-T01"), valid("QUAL-E1-T01")),
            &roster(),
        )
        .unwrap();
        assert!(check_records(&twice, &actions)[0].contains("recorded twice"));
    }

    #[test]
    fn the_queue_renders_with_its_state() {
        let epic = "| QUAL-E1-T01 | **Maintainer action**: a | M0a |\n\
                    | QUAL-E1-T02 | **Maintainer action**: b | M0a |\n";
        let records = parse_records(&valid("QUAL-E1-T01"), &roster()).unwrap();
        assert_eq!(
            render(&queue(&[epic]), &records),
            "QUAL-E1-T01  done  Maintainer action: a\n    by alice on 2026-11-02, evidence: \
             ceremony record\nQUAL-E1-T02  open  Maintainer action: b\n2 maintainer actions, 1 done\n"
        );
    }

    #[test]
    fn the_real_epics_hold_the_registry_tasks() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let epics = read_epics(&root).unwrap();
        let actions = queue(&epics.iter().map(String::as_str).collect::<Vec<_>>());
        let ids = ids(&actions);
        assert!(ids.contains(&"QUAL-E1-T26") && ids.contains(&"QUAL-E1-T27"));
        assert!(!ids.contains(&"QUAL-E1-T17"));
    }

    #[test]
    fn the_real_records_pass() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut out = Vec::new();
        assert!(
            command(&root, &mut out).unwrap(),
            "{}",
            String::from_utf8_lossy(&out)
        );
    }
}
