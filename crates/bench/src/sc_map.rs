// SPDX-License-Identifier: MIT OR Apache-2.0

//! `bench/sc-map.toml`: every success criterion mapped to its scenarios, rigs, metric and pass
//! rule (RFC-0061 §4.2).
//!
//! The map holds one `[[sc]]` row per criterion of `00-vision-and-scope.md` §7. Each row has the
//! four cells of RFC-0061 §4.2's table as text, and the scenario-ID patterns those cells name.
//! [`render`] turns the map into the table, and RFC-0061 §4.2 holds that table as a generated
//! section, so the RFC and the map can't drift.
//!
//! A row's Status column says where its measurement stands, from what exists in
//! `bench/scenarios/`, and names its owners. The owners are derived, not typed ([`owners`]):
//! every task whose acceptance criteria in `plan/implementation/epics/` name the criterion,
//! plus the row's `added_owners`, minus the map's `[[not_owner]]` entries. So when an epic edit
//! starts or stops citing a criterion, the table goes stale, and `check` names the row and the
//! task until the table is rendered again or the map records why the task is not an owner.
//!
//! The map never claims a measurement that doesn't exist.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::scenario::{check_pattern, pattern_matches};

/// The map, relative to the repository root.
pub const SC_MAP: &str = "bench/sc-map.toml";
/// The success criteria the map must cover: the `| SC-n |` rows of §7.
pub const VISION: &str = "plan/00-vision-and-scope.md";
/// The RFC whose §4.2 holds the rendered table.
pub const RFC_0061: &str = "plan/rfcs/RFC-0061-benchmarking-and-performance-regression.md";
/// The epic files: the tasks, and the acceptance criteria that make a task an owner.
pub const EPICS: &str = "plan/implementation/epics";
/// The generated section's name in RFC-0061.
pub const SECTION: &str = "sc-table";

const BEGIN: &str = "<!-- BEGIN GENERATED: sc-table -->\n";
const END: &str = "<!-- END GENERATED -->";
const HEADER: &str = "| SC | Scenario(s) | Rig(s) | Measured | Pass | Status |\n\
                      |---|---|---|---|---|---|\n";

/// The milestones in delivery order (RFC-0001 §16), as `plan/implementation/tools/epics.py`
/// orders them. A plain `M0` or `M1` reads as `M0b` or `M1a`, as there.
const PHASES: [&str; 7] = ["M0a", "M0b", "M1a", "M1b", "M2", "M3", "M4"];

/// One success criterion's row.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// `SC-<n>`.
    pub id: String,
    /// The Scenario(s) cell, as Markdown.
    pub scenarios: String,
    /// The scenario-ID patterns the cell names ([`check_pattern`]). Empty for a criterion that
    /// no scenario file measures, such as an RFC-0060 property, chaos or parity result.
    #[serde(default)]
    pub patterns: Vec<String>,
    /// The Rig(s) cell.
    pub rigs: String,
    /// The Measured cell.
    pub measured: String,
    /// The Pass cell: the pass rule, with thresholds from `00-vision-and-scope.md` §7.
    pub pass: String,
    /// Owners the acceptance criteria don't show: a scenario task whose row names the criterion
    /// in its task or Implements cell but not in its acceptance criteria.
    #[serde(default)]
    pub added_owners: Vec<String>,
}

/// A task whose acceptance criteria name criteria it does not own, with the reason.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotOwner {
    /// The task.
    pub task: String,
    /// The criteria, as `SC-<n>`.
    pub sc: Vec<String>,
    /// Why the citation is not ownership, such as "lists these rows as not applicable".
    pub reason: String,
}

/// `bench/sc-map.toml`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Map {
    /// One row per success criterion, in SC order.
    pub sc: Vec<Row>,
    /// The citations that are not ownership.
    #[serde(default)]
    pub not_owner: Vec<NotOwner>,
}

/// `text` with every CRLF line ending as LF. A Windows checkout with no `.gitattributes` has
/// CRLF in every text file while the committed blobs keep LF, so every file this crate reads
/// goes through here, and everything it renders is LF.
pub fn to_lf(text: &str) -> String {
    text.replace("\r\n", "\n")
}

/// Parses the map.
pub fn parse(text: &str) -> Result<Map, String> {
    toml::from_str::<Map>(text).map_err(|e| e.to_string())
}

/// One task row of the epic files, as far as the owner rule reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    /// The task ID, such as `QUAL-E13-T02`.
    pub id: String,
    /// The earliest milestone the Milestone cell names, as an index into the delivery order;
    /// `PHASES.len()` when it names none.
    pub phase: usize,
    /// The Acceptance criteria cell.
    pub acceptance: String,
    /// The whole row, every cell joined.
    pub row: String,
}

impl Task {
    /// Sorts owners by milestone, then by area, epic and task number.
    fn order(&self) -> (usize, String, u32, u32) {
        let mut parts = self.id.split('-');
        let area = parts.next().unwrap_or_default().to_string();
        let number = |p: Option<&str>| p.and_then(|s| s[1..].parse().ok()).unwrap_or(0);
        let epic = number(parts.next());
        let task = number(parts.next());
        (self.phase, area, epic, task)
    }
}

/// The criterion numbers of `00-vision-and-scope.md`: every table row that starts `| SC-<n> |`.
pub fn vision_ids(vision: &str) -> Vec<u32> {
    vision
        .lines()
        .filter_map(|line| line.strip_prefix("| SC-"))
        .filter_map(|rest| rest.split_once(' ').map(|(n, _)| n))
        .filter_map(|n| n.parse().ok())
        .collect()
}

/// Every task row of the epic files, by ID. Rows are read the way
/// `plan/implementation/tools/epics.py` reads them: cells split on unescaped `|`, and each
/// table's header row names its columns.
pub fn tasks(epics_dir: &Path) -> BTreeMap<String, Task> {
    let mut tasks = BTreeMap::new();
    let Ok(entries) = fs::read_dir(epics_dir) else {
        return tasks;
    };
    let mut paths: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        tasks.extend(
            tasks_in(&to_lf(&text))
                .into_iter()
                .map(|t| (t.id.clone(), t)),
        );
    }
    tasks
}

/// The task rows of one epic file's text.
pub fn tasks_in(text: &str) -> Vec<Task> {
    let mut tasks = Vec::new();
    let mut header: Option<(usize, usize)> = None;
    for line in text.lines() {
        if !line.trim_start().starts_with('|') {
            if line.trim().is_empty() {
                header = None;
            }
            continue;
        }
        let cells = split_row(line);
        let low: Vec<String> = cells.iter().map(|c| c.to_lowercase()).collect();
        if low.first().is_some_and(|c| c == "id") {
            let milestone = low.iter().position(|c| c == "milestone");
            let acceptance = low
                .iter()
                .position(|c| c.starts_with("acceptance criteria"));
            header = milestone.zip(acceptance);
            continue;
        }
        let (Some((milestone, acceptance)), Some(id)) = (header, cells.first()) else {
            continue;
        };
        if !is_task_id(id) {
            continue;
        }
        tasks.push(Task {
            id: id.clone(),
            phase: phase_of(cells.get(milestone).map_or("", String::as_str)),
            acceptance: cells.get(acceptance).cloned().unwrap_or_default(),
            row: cells.join(" | "),
        });
    }
    tasks
}

/// Splits a Markdown table row on unescaped `|`, as `epics.py`'s `split_row` does.
fn split_row(line: &str) -> Vec<String> {
    let mut body = line.trim();
    body = body.strip_prefix('|').unwrap_or(body);
    if body.ends_with('|') && !body.ends_with("\\|") {
        body = &body[..body.len() - 1];
    }
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek() == Some(&'|') => {
                cell.push('|');
                chars.next();
            }
            '|' => cells.push(std::mem::take(&mut cell).trim().to_string()),
            c => cell.push(c),
        }
    }
    cells.push(cell.trim().to_string());
    cells
}

/// The earliest milestone a Milestone cell names.
fn phase_of(cell: &str) -> usize {
    cell.split(|c: char| !c.is_ascii_alphanumeric())
        .filter_map(|word| {
            let word = match word {
                "M0" => "M0b",
                "M1" => "M1a",
                w => w,
            };
            PHASES.iter().position(|p| *p == word)
        })
        .min()
        .unwrap_or(PHASES.len())
}

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

/// The success criteria a text names: every `SC-<n>`, and every number in a range written
/// `SC-<a>–SC-<b>` or `SC-<a>–<b>` (en dash or hyphen).
pub fn cited(text: &str) -> BTreeSet<u32> {
    let mut found = BTreeSet::new();
    let mut rest = text;
    while let Some(at) = rest.find("SC-") {
        rest = &rest[at + "SC-".len()..];
        let Some((a, after)) = number(rest) else {
            continue;
        };
        found.insert(a);
        // `SC-a–SC-b`, with optional spaces around the dash, or `SC-a–b` with none.
        if let Some(tail) = dash(after.trim_start())
            && let Some((b, _)) = tail.trim_start().strip_prefix("SC-").and_then(number)
        {
            found.extend(a..=b);
        }
        if let Some((b, end)) = dash(after).and_then(number)
            && !end.starts_with(|c: char| c.is_alphanumeric() || c == '_')
        {
            found.extend(a..=b);
        }
    }
    found
}

/// `s` after a leading en dash or hyphen.
fn dash(s: &str) -> Option<&str> {
    s.strip_prefix('–').or_else(|| s.strip_prefix('-'))
}

/// The leading decimal number of `s`, and the rest.
fn number(s: &str) -> Option<(u32, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    s[..end].parse().ok().map(|n| (n, &s[end..]))
}

fn sc_number(id: &str) -> Option<u32> {
    id.strip_prefix("SC-").and_then(|n| n.parse().ok())
}

/// A row's owners, in milestone order: every task whose acceptance criteria name the criterion,
/// except where a `[[not_owner]]` entry says otherwise, plus the row's `added_owners`.
pub fn owners(row: &Row, map: &Map, tasks: &BTreeMap<String, Task>) -> Vec<String> {
    let Some(n) = sc_number(&row.id) else {
        return Vec::new();
    };
    let excluded = |id: &str| {
        map.not_owner
            .iter()
            .any(|x| x.task == id && x.sc.iter().any(|s| sc_number(s) == Some(n)))
    };
    let mut owners: Vec<&Task> = tasks
        .values()
        .filter(|t| cited(&t.acceptance).contains(&n) && !excluded(&t.id))
        .collect();
    for added in &row.added_owners {
        if let Some(task) = tasks.get(added)
            && !owners.iter().any(|t| t.id == task.id)
        {
            owners.push(task);
        }
    }
    owners.sort_by_key(|t| t.order());
    owners.into_iter().map(|t| t.id.clone()).collect()
}

/// Checks the map: it covers exactly the criteria of §7, once each and in order; every row is
/// well formed; and the added owners and the exclusions still match the epics. Returns every
/// problem found.
pub fn check(map: &Map, vision: &[u32], tasks: &BTreeMap<String, Task>) -> Vec<String> {
    let mut problems = Vec::new();
    let mut numbers = Vec::new();
    for row in &map.sc {
        let Some(n) = sc_number(&row.id) else {
            problems.push(format!("`{}` is not an `SC-<n>` ID", row.id));
            continue;
        };
        numbers.push(n);
        for (cell, text) in [
            ("scenarios", &row.scenarios),
            ("rigs", &row.rigs),
            ("measured", &row.measured),
            ("pass", &row.pass),
        ] {
            if text.trim().is_empty() {
                problems.push(format!("{}: `{cell}` is empty", row.id));
            } else if text.contains('\n') {
                problems.push(format!(
                    "{}: `{cell}` must be one line, so it fits a table cell",
                    row.id
                ));
            }
        }
        let mut seen = BTreeSet::new();
        for added in &row.added_owners {
            if !seen.insert(added) {
                problems.push(format!("{}: added owner `{added}` is listed twice", row.id));
            }
            let Some(task) = tasks.get(added) else {
                problems.push(format!(
                    "{}: added owner `{added}` is not a task in {EPICS}/",
                    row.id
                ));
                continue;
            };
            if cited(&task.acceptance).contains(&n) {
                problems.push(format!(
                    "{}: `{added}`'s acceptance criteria name SC-{n}, so it is an owner \
                     already; remove it from `added_owners`",
                    row.id
                ));
            } else if !cited(&task.row).contains(&n) {
                problems.push(format!(
                    "{}: added owner `{added}`'s row doesn't name SC-{n}",
                    row.id
                ));
            }
        }
        if owners(row, map, tasks).is_empty() {
            problems.push(format!(
                "{}: no owner: no task's acceptance criteria name SC-{n}, and `added_owners` \
                 is empty",
                row.id
            ));
        }
        let mut seen = BTreeSet::new();
        for pattern in &row.patterns {
            if let Err(why) = check_pattern(pattern) {
                problems.push(format!("{}: {why}", row.id));
            }
            if !seen.insert(pattern) {
                problems.push(format!("{}: pattern `{pattern}` is listed twice", row.id));
            }
        }
    }
    for exclusion in &map.not_owner {
        let task = exclusion.task.as_str();
        if exclusion.reason.trim().is_empty() {
            problems.push(format!("not_owner `{task}`: give a `reason`"));
        }
        let Some(row) = tasks.get(task) else {
            problems.push(format!("not_owner `{task}` is not a task in {EPICS}/"));
            continue;
        };
        for sc in &exclusion.sc {
            match sc_number(sc) {
                None => problems.push(format!("not_owner `{task}`: `{sc}` is not an `SC-<n>` ID")),
                Some(n) if !cited(&row.acceptance).contains(&n) => problems.push(format!(
                    "not_owner `{task}`: its acceptance criteria no longer name {sc}; remove \
                     {sc} from the entry"
                )),
                Some(_) => {}
            }
        }
    }
    let mut seen = BTreeSet::new();
    for n in &numbers {
        if !seen.insert(*n) {
            problems.push(format!("SC-{n} is mapped twice"));
        }
    }
    if numbers.windows(2).any(|w| w[0] > w[1]) {
        problems.push(String::from("rows must be in SC order"));
    }
    let wanted: BTreeSet<u32> = vision.iter().copied().collect();
    for n in wanted.difference(&seen) {
        problems.push(format!(
            "SC-{n} is a success criterion in {VISION} §7 but has no row in {SC_MAP}"
        ));
    }
    for n in seen.difference(&wanted) {
        problems.push(format!(
            "SC-{n} has a row in {SC_MAP} but is not a success criterion in {VISION} §7"
        ));
    }
    problems
}

/// The Status cell of a row: where the measurement stands, from the IDs of the scenario files
/// that exist, then every owner.
pub fn status(row: &Row, owners: &[String], scenario_ids: &[String]) -> String {
    let owners = owners.join(", ");
    if row.patterns.is_empty() {
        return format!("not a scenario file; {owners}");
    }
    let found = scenario_ids
        .iter()
        .filter(|id| row.patterns.iter().any(|p| pattern_matches(p, id)))
        .count();
    match found {
        0 => format!("pending; {owners}"),
        1 => format!("1 scenario; {owners}"),
        n => format!("{n} scenarios; {owners}"),
    }
}

/// RFC-0061 §4.2's table, from the map, the epics and the scenario files that exist. A `|` in
/// a cell is written `\|`, which a Markdown table reads as a literal `|`, inside a code span
/// too.
pub fn render(map: &Map, tasks: &BTreeMap<String, Task>, scenario_ids: &[String]) -> String {
    let cell = |text: &str| text.replace('|', "\\|");
    let mut table = String::from(HEADER);
    for row in &map.sc {
        table.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            row.id,
            cell(&row.scenarios),
            cell(&row.rigs),
            cell(&row.measured),
            cell(&row.pass),
            status(row, &owners(row, map, tasks), scenario_ids)
        ));
    }
    table
}

/// How a rendered table's owners differ from the owners the epics give now: one line per task
/// missing from, or extra in, a row of `table`, naming the row and the task.
pub fn owner_drift(table: &str, map: &Map, tasks: &BTreeMap<String, Task>) -> Vec<String> {
    let listed: BTreeMap<&str, BTreeSet<&str>> = table
        .lines()
        .filter(|line| line.starts_with("| SC-"))
        .filter_map(|line| {
            let id = line[2..].split(' ').next()?;
            let status = line.trim_end().strip_suffix('|')?.rsplit(" | ").next()?;
            let owners = status.split_once("; ").map_or("", |(_, o)| o.trim());
            Some((id, owners.split(", ").filter(|o| !o.is_empty()).collect()))
        })
        .collect();
    let mut drift = Vec::new();
    for row in &map.sc {
        let now = owners(row, map, tasks);
        let then = listed.get(row.id.as_str()).cloned().unwrap_or_default();
        for task in now.iter().filter(|t| !then.contains(t.as_str())) {
            drift.push(format!(
                "{}: `{task}` is an owner now (its acceptance criteria name {}), but the table \
                 doesn't list it",
                row.id, row.id
            ));
        }
        for task in then.iter().filter(|t| !now.iter().any(|o| o == *t)) {
            drift.push(format!(
                "{}: the table lists `{task}`, which is no longer an owner",
                row.id
            ));
        }
    }
    drift
}

/// The body of the generated `sc-table` section in `text`, without its markers.
pub fn section(text: &str) -> Option<&str> {
    let start = text.find(BEGIN)? + BEGIN.len();
    let end = start + text[start..].find(END)?;
    Some(&text[start..end])
}

/// `text` with the generated `sc-table` section's body replaced by `body`.
pub fn replace_section(text: &str, body: &str) -> Option<String> {
    let start = text.find(BEGIN)? + BEGIN.len();
    let end = start + text[start..].find(END)?;
    Some(format!("{}{}{}", &text[..start], body, &text[end..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPIC: &str = "\
| ID | Task | Milestone | Depends on | Acceptance criteria | Implements |
|---|---|---|---|---|---|
| QUAL-E13-T03 | M1a scenarios | M1a | none | SC-1–SC-3 reported on Linux | RFC-0061 §4.2 |
| COMP-E5-T24 | SC-1 delivery | M0b | none | decode \\| prefill meets SC-1 | SC-1 |
| QUAL-E13-T02 | M0 scenarios for SC-1 and SC-2 | M0 | none | `pr-quick` fits in 10 min | SC-1; SC-2 |
| QUAL-E13-T20 | M1b scenarios | M1b | none | SC-1; SC-3 is not applicable here | SC-1 |
";

    fn tasks() -> BTreeMap<String, Task> {
        tasks_in(EPIC)
            .into_iter()
            .map(|t| (t.id.clone(), t))
            .collect()
    }

    fn row(n: u32) -> Row {
        Row {
            id: format!("SC-{n}"),
            scenarios: String::from("`decode.single.*`"),
            patterns: vec![String::from("decode.single.*")],
            rigs: String::from("R-ADA"),
            measured: String::from("`t_tok`"),
            pass: String::from("≤ 1.05×"),
            added_owners: Vec::new(),
        }
    }

    fn map(rows: Vec<Row>) -> Map {
        Map {
            sc: rows,
            not_owner: Vec::new(),
        }
    }

    #[test]
    fn epic_rows_read_like_epics_py() {
        let tasks = tasks();
        assert_eq!(tasks.len(), 4);
        let comp = &tasks["COMP-E5-T24"];
        assert_eq!(comp.phase, 1);
        // An escaped pipe stays inside its cell.
        assert_eq!(comp.acceptance, "decode | prefill meets SC-1");
        // A plain `M0` is `M0b`.
        assert_eq!(tasks["QUAL-E13-T02"].phase, 1);
        assert_eq!(tasks["QUAL-E13-T03"].phase, 2);
    }

    #[test]
    fn cited_expands_ranges() {
        let all = |t: &str| cited(t).into_iter().collect::<Vec<_>>();
        assert_eq!(all("SC-1–SC-4"), [1, 2, 3, 4]);
        assert_eq!(
            all("SC-12 - SC-14, SC-22–25 and SC-27"),
            [12, 13, 14, 22, 23, 24, 25, 27]
        );
        assert_eq!(all("SC-5–11"), [5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(all("SC-4's rigs; SC-x; SC-"), [4]);
        assert_eq!(all("SC-3-way"), [3]);
    }

    #[test]
    fn owners_come_from_acceptance_criteria_in_milestone_order() {
        let tasks = tasks();
        let m = map(vec![row(1)]);
        // COMP-E5-T24 (M0b) before QUAL-E13-T03 (M1a) before QUAL-E13-T20 (M1b); QUAL-E13-T02's
        // criteria don't name SC-1, so it is not an owner until it is added.
        assert_eq!(
            owners(&m.sc[0], &m, &tasks),
            ["COMP-E5-T24", "QUAL-E13-T03", "QUAL-E13-T20"]
        );
        let mut added = row(1);
        added.added_owners = vec![String::from("QUAL-E13-T02")];
        let m = map(vec![added]);
        assert_eq!(
            owners(&m.sc[0], &m, &tasks),
            [
                "COMP-E5-T24",
                "QUAL-E13-T02",
                "QUAL-E13-T03",
                "QUAL-E13-T20"
            ]
        );
    }

    #[test]
    fn an_exclusion_removes_a_citation_and_goes_stale_with_it() {
        let tasks = tasks();
        let mut m = map(vec![row(3)]);
        assert_eq!(
            owners(&m.sc[0], &m, &tasks),
            ["QUAL-E13-T03", "QUAL-E13-T20"]
        );
        m.not_owner.push(NotOwner {
            task: String::from("QUAL-E13-T20"),
            sc: vec![String::from("SC-3"), String::from("SC-2")],
            reason: String::from("its criteria list SC-3 as not applicable"),
        });
        assert_eq!(owners(&m.sc[0], &m, &tasks), ["QUAL-E13-T03"]);
        assert_eq!(
            check(&m, &[3], &tasks),
            [
                "not_owner `QUAL-E13-T20`: its acceptance criteria no longer name SC-2; remove \
              SC-2 from the entry"
            ]
        );
    }

    #[test]
    fn added_owners_must_need_adding() {
        let tasks = tasks();
        let mut r = row(1);
        r.added_owners = vec![
            String::from("QUAL-E13-T02"),
            String::from("QUAL-E13-T03"),
            String::from("QUAL-E99-T01"),
        ];
        let mut r2 = row(2);
        r2.added_owners = vec![String::from("COMP-E5-T24")];
        let problems = check(&map(vec![r, r2]), &[1, 2], &tasks);
        for want in [
            "SC-1: `QUAL-E13-T03`'s acceptance criteria name SC-1, so it is an owner already",
            "SC-1: added owner `QUAL-E99-T01` is not a task",
            "SC-2: added owner `COMP-E5-T24`'s row doesn't name SC-2",
        ] {
            assert!(
                problems.iter().any(|p| p.starts_with(want)),
                "no `{want}` in {problems:?}"
            );
        }
    }

    #[test]
    fn a_complete_map_passes_and_an_unmapped_criterion_fails() {
        let tasks = tasks();
        let rows: Vec<Row> = (1..=3).map(row).collect();
        assert_eq!(check(&map(rows), &[1, 2, 3], &tasks), Vec::<String>::new());
        let rows: Vec<Row> = [1, 3].into_iter().map(row).collect();
        let problems = check(&map(rows), &[1, 2, 3], &tasks);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].starts_with("SC-2 is a success criterion"));
    }

    #[test]
    fn a_row_with_no_owner_fails() {
        let problems = check(&map(vec![row(9)]), &[9], &tasks());
        assert_eq!(
            problems,
            [
                "SC-9: no owner: no task's acceptance criteria name SC-9, and `added_owners` is empty"
            ]
        );
    }

    #[test]
    fn duplicates_extras_order_and_bad_rows_fail() {
        let tasks = tasks();
        let check_rows = |rows: Vec<Row>, vision: &[u32]| check(&map(rows), vision, &tasks);
        assert!(
            check_rows([1, 1, 2].into_iter().map(row).collect(), &[1, 2])
                .iter()
                .any(|p| p == "SC-1 is mapped twice")
        );
        assert!(
            check_rows([2, 1].into_iter().map(row).collect(), &[1, 2])
                .iter()
                .any(|p| p.contains("SC order"))
        );
        assert!(
            check_rows([1, 2, 9].into_iter().map(row).collect(), &[1, 2])
                .iter()
                .any(|p| p.starts_with("SC-9 has a row"))
        );
        let mut bad = row(1);
        bad.pass = String::new();
        bad.rigs = String::from("a\nb");
        bad.patterns = vec![String::from("Decode.x"), String::from("Decode.x")];
        let problems = check_rows(vec![bad], &[1]);
        for want in [
            "`pass` is empty",
            "one line",
            "not a scenario class",
            "twice",
        ] {
            assert!(
                problems.iter().any(|p| p.contains(want)),
                "no `{want}` in {problems:?}"
            );
        }
    }

    #[test]
    fn vision_ids_read_the_table_rows_only() {
        let vision = "| SC-1 | a | b |\nsee SC-7 here\n| SC-12 | c |\n| SC-x | d |\n";
        assert_eq!(vision_ids(vision), [1, 12]);
    }

    #[test]
    fn status_says_pending_until_a_scenario_matches() {
        let r = row(1);
        let owners = [String::from("QUAL-E13-T02"), String::from("QUAL-E13-T20")];
        assert_eq!(
            status(&r, &owners, &[]),
            "pending; QUAL-E13-T02, QUAL-E13-T20"
        );
        let ids = vec![
            String::from("decode.single.llama8b.q4km"),
            String::from("prefill.single.llama8b.q4km"),
        ];
        assert_eq!(status(&r, &owners[..1], &ids), "1 scenario; QUAL-E13-T02");
        let mut outside = row(2);
        outside.patterns.clear();
        assert_eq!(
            status(&outside, &owners[..1], &ids),
            "not a scenario file; QUAL-E13-T02"
        );
    }

    #[test]
    fn render_names_the_derived_owners_and_drift_names_the_row() {
        let tasks = tasks();
        let m = map(vec![row(2)]);
        let table = render(&m, &tasks, &[]);
        assert_eq!(
            table,
            "| SC | Scenario(s) | Rig(s) | Measured | Pass | Status |\n|---|---|---|---|---|---|\n\
             | SC-2 | `decode.single.*` | R-ADA | `t_tok` | ≤ 1.05× | pending; QUAL-E13-T03 |\n"
        );
        assert_eq!(owner_drift(&table, &m, &tasks), Vec::<String>::new());
        // An epic edit makes QUAL-E13-T20 cite SC-2 and QUAL-E13-T03 stop citing it.
        let edited = EPIC
            .replace("SC-1–SC-3 reported", "SC-1 and SC-3 reported")
            .replace("SC-1; SC-3 is not", "SC-1, SC-2; SC-3 is not");
        let now: BTreeMap<String, Task> = tasks_in(&edited)
            .into_iter()
            .map(|t| (t.id.clone(), t))
            .collect();
        assert_eq!(
            owner_drift(&table, &m, &now),
            [
                "SC-2: `QUAL-E13-T20` is an owner now (its acceptance criteria name SC-2), but \
                 the table doesn't list it",
                "SC-2: the table lists `QUAL-E13-T03`, which is no longer an owner",
            ]
        );
    }

    #[test]
    fn a_pipe_in_a_cell_is_escaped() {
        let mut r = row(3);
        r.measured = String::from("`|t_pred − t_meas| / t_meas`");
        let m = map(vec![r]);
        assert!(check(&m, &[3], &tasks()).is_empty());
        let table = render(&m, &tasks(), &[]);
        assert!(
            table.contains("| `\\|t_pred − t_meas\\| / t_meas` |"),
            "{table}"
        );
    }

    #[test]
    fn render_and_replace_round_trip_through_the_section() {
        let table = render(&map(vec![row(1)]), &tasks(), &[]);
        let doc = format!("intro\n{BEGIN}old\n{END}\noutro\n");
        assert_eq!(section(&doc), Some("old\n"));
        let new = replace_section(&doc, &table).expect("has markers");
        assert_eq!(section(&new), Some(table.as_str()));
        assert!(new.starts_with("intro\n") && new.ends_with(&format!("{END}\noutro\n")));
        assert_eq!(section("no markers"), None);
    }

    #[test]
    fn map_parse_rejects_unknown_fields() {
        let text = "[[sc]]\nid = \"SC-1\"\nscenarios = \"a\"\nrigs = \"b\"\nmeasured = \"c\"\n\
                    pass = \"d\"\nowners = [\"QUAL-E13-T02\"]\n";
        assert!(parse(text).unwrap_err().contains("owners"));
    }
}
