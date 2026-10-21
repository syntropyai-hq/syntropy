// SPDX-License-Identifier: MIT OR Apache-2.0

//! `bench/sc-map.toml`: every success criterion mapped to its scenarios, rigs, metric and pass
//! rule (RFC-0061 §4.2).
//!
//! The map holds one `[[sc]]` row per criterion of `00-vision-and-scope.md` §7. Each row has the
//! four cells of RFC-0061 §4.2's table as text, the scenario-ID patterns those cells name, and
//! every task that delivers the measurement. [`render`] turns the map into the table, and
//! RFC-0061 §4.2 holds that table as a generated section, so the RFC and the map can't drift.
//!
//! A row's status column says where its measurement stands, from what exists in
//! `bench/scenarios/`. A row whose patterns match no scenario file yet is *pending* on its owner
//! tasks. The map never claims a measurement that doesn't exist.

use std::collections::BTreeSet;
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
/// The epic files, which define every task an owner may be.
pub const EPICS: &str = "plan/implementation/epics";
/// The generated section's name in RFC-0061.
pub const SECTION: &str = "sc-table";

const BEGIN: &str = "<!-- BEGIN GENERATED: sc-table -->\n";
const END: &str = "<!-- END GENERATED -->";
const HEADER: &str = "| SC | Scenario(s) | Rig(s) | Measured | Pass | Status |\n\
                      |---|---|---|---|---|---|\n";

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
    /// Every task that delivers the measurement, such as one per milestone or set of rigs, in
    /// plan order.
    pub owners: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMap {
    sc: Vec<Row>,
}

/// Parses the map.
pub fn parse(text: &str) -> Result<Vec<Row>, String> {
    toml::from_str::<RawMap>(text)
        .map(|map| map.sc)
        .map_err(|e| e.to_string())
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

/// Every task ID the epic files define: the first cell of each `| <ID> |` task row.
pub fn task_ids(epics_dir: &Path) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    let Ok(entries) = fs::read_dir(epics_dir) else {
        return ids;
    };
    for entry in entries.flatten() {
        let Ok(text) = fs::read_to_string(entry.path()) else {
            continue;
        };
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("| ") else {
                continue;
            };
            let Some((id, _)) = rest.split_once(" |") else {
                continue;
            };
            if is_task_id(id) {
                ids.insert(id.to_string());
            }
        }
    }
    ids
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

/// Checks the map: it covers exactly the criteria of §7, once each and in order, and every row
/// is well formed. Returns every problem found.
pub fn check(rows: &[Row], vision: &[u32], tasks: &BTreeSet<String>) -> Vec<String> {
    let mut problems = Vec::new();
    let mut numbers = Vec::new();
    for row in rows {
        let Some(n) = row
            .id
            .strip_prefix("SC-")
            .and_then(|n| n.parse::<u32>().ok())
        else {
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
        if row.owners.is_empty() {
            problems.push(format!("{}: `owners` is empty", row.id));
        }
        let mut owners = BTreeSet::new();
        for owner in &row.owners {
            if !tasks.contains(owner) {
                problems.push(format!(
                    "{}: owner `{owner}` is not a task in plan/implementation/epics/",
                    row.id
                ));
            }
            if !owners.insert(owner) {
                problems.push(format!("{}: owner `{owner}` is listed twice", row.id));
            }
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

/// The Status cell of a row, given the IDs of the scenario files that exist: where the
/// measurement stands, then every owner task.
pub fn status(row: &Row, scenario_ids: &[String]) -> String {
    let owners = row.owners.join(", ");
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

/// RFC-0061 §4.2's table, from the map and the scenario files that exist. A `|` in a cell is
/// written `\|`, which a Markdown table reads as a literal `|`, inside a code span too.
pub fn render(rows: &[Row], scenario_ids: &[String]) -> String {
    let cell = |text: &str| text.replace('|', "\\|");
    let mut table = String::from(HEADER);
    for row in rows {
        table.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            row.id,
            cell(&row.scenarios),
            cell(&row.rigs),
            cell(&row.measured),
            cell(&row.pass),
            status(row, scenario_ids)
        ));
    }
    table
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

    fn row(n: u32) -> Row {
        Row {
            id: format!("SC-{n}"),
            scenarios: String::from("`decode.single.*`"),
            patterns: vec![String::from("decode.single.*")],
            rigs: String::from("R-ADA"),
            measured: String::from("`t_tok`"),
            pass: String::from("≤ 1.05×"),
            owners: vec![String::from("QUAL-E13-T02")],
        }
    }

    fn tasks() -> BTreeSet<String> {
        BTreeSet::from([String::from("QUAL-E13-T02"), String::from("QUAL-E13-T20")])
    }

    #[test]
    fn a_complete_map_passes() {
        let rows: Vec<Row> = (1..=3).map(row).collect();
        assert_eq!(check(&rows, &[1, 2, 3], &tasks()), Vec::<String>::new());
    }

    #[test]
    fn an_unmapped_criterion_fails() {
        let rows: Vec<Row> = [1, 3].into_iter().map(row).collect();
        let problems = check(&rows, &[1, 2, 3], &tasks());
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(problems[0].starts_with("SC-2 is a success criterion"));
    }

    #[test]
    fn duplicates_extras_order_and_bad_rows_fail() {
        let mut rows: Vec<Row> = [1, 1, 2].into_iter().map(row).collect();
        assert!(
            check(&rows, &[1, 2], &tasks())
                .iter()
                .any(|p| p == "SC-1 is mapped twice")
        );
        rows = [2, 1].into_iter().map(row).collect();
        assert!(
            check(&rows, &[1, 2], &tasks())
                .iter()
                .any(|p| p.contains("SC order"))
        );
        rows = [1, 2, 9].into_iter().map(row).collect();
        assert!(
            check(&rows, &[1, 2], &tasks())
                .iter()
                .any(|p| p.starts_with("SC-9 has a row"))
        );
        let mut bad = row(1);
        bad.owners = vec![String::from("QUAL-E99-T01")];
        bad.pass = String::new();
        bad.rigs = String::from("a\nb");
        bad.patterns = vec![String::from("Decode.x"), String::from("Decode.x")];
        let problems = check(&[bad], &[1], &tasks());
        for want in [
            "owner",
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
        assert_eq!(status(&r, &[]), "pending; QUAL-E13-T02");
        let ids = vec![
            String::from("decode.single.llama8b.q4km"),
            String::from("prefill.single.llama8b.q4km"),
        ];
        assert_eq!(status(&r, &ids), "1 scenario; QUAL-E13-T02");
        let mut outside = row(2);
        outside.patterns.clear();
        assert_eq!(status(&outside, &ids), "not a scenario file; QUAL-E13-T02");
        let mut two = row(3);
        two.owners.push(String::from("QUAL-E13-T20"));
        assert_eq!(status(&two, &[]), "pending; QUAL-E13-T02, QUAL-E13-T20");
    }

    #[test]
    fn owners_are_tasks_listed_once() {
        let mut r = row(1);
        assert_eq!(
            check(std::slice::from_ref(&r), &[1], &tasks()),
            Vec::<String>::new()
        );
        r.owners = vec![String::from("QUAL-E13-T20"), String::from("QUAL-E13-T20")];
        assert_eq!(
            check(std::slice::from_ref(&r), &[1], &tasks()),
            ["SC-1: owner `QUAL-E13-T20` is listed twice"]
        );
        r.owners.clear();
        assert_eq!(check(&[r], &[1], &tasks()), ["SC-1: `owners` is empty"]);
    }

    #[test]
    fn render_and_replace_round_trip_through_the_section() {
        let rows = vec![row(1)];
        let table = render(&rows, &[]);
        assert_eq!(
            table,
            "| SC | Scenario(s) | Rig(s) | Measured | Pass | Status |\n|---|---|---|---|---|---|\n\
             | SC-1 | `decode.single.*` | R-ADA | `t_tok` | ≤ 1.05× | pending; QUAL-E13-T02 |\n"
        );
        let doc = format!("intro\n{BEGIN}old\n{END}\noutro\n");
        assert_eq!(section(&doc), Some("old\n"));
        let new = replace_section(&doc, &table).expect("has markers");
        assert_eq!(section(&new), Some(table.as_str()));
        assert!(new.starts_with("intro\n") && new.ends_with(&format!("{END}\noutro\n")));
        assert_eq!(section("no markers"), None);
    }

    #[test]
    fn a_pipe_in_a_cell_is_escaped() {
        let mut r = row(1);
        r.measured = String::from("`|t_pred − t_meas| / t_meas`");
        assert!(check(std::slice::from_ref(&r), &[1], &tasks()).is_empty());
        let table = render(&[r], &[]);
        assert!(
            table.contains("| `\\|t_pred − t_meas\\| / t_meas` |"),
            "{table}"
        );
    }

    #[test]
    fn map_parse_rejects_unknown_fields() {
        let text = "[[sc]]\nid = \"SC-1\"\nscenarios = \"a\"\nrigs = \"b\"\nmeasured = \"c\"\n\
                    pass = \"d\"\nowners = [\"QUAL-E13-T02\"]\nnotes = \"x\"\n";
        assert!(parse(text).unwrap_err().contains("notes"));
    }
}
