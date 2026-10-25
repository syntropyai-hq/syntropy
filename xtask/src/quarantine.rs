// SPDX-License-Identifier: MIT OR Apache-2.0

//! The quarantine half of the flake policy of RFC-0060 §13 (QUAL-E3-T14): `ci/quarantine.toml`,
//! the `quarantine` step of `lint` that checks it, and the `quarantine` job that lists its
//! entries for the release checklist.
//!
//! A non-deterministic test (L7, L8) that fails and then passes on the same commit may be
//! quarantined by a maintainer: taken out of the gates while it still runs and reports. Each
//! entry needs:
//! - `test`, the test's full name as its suite reports it, and `suite`, the suite or job that
//!   runs it;
//! - `layer`, `L7` or `L8`: a deterministic test is never quarantined, since a failure in it is a
//!   failure;
//! - `issue`, the tracking issue, a link to an issue of this repository;
//! - `added` and `expires`, `YYYY-MM-DD` dates at most 14 days apart. After `expires` the test is
//!   back in the required set ([`active`] leaves it out), and `lint` fails until the entry is
//!   removed or a maintainer renews it with a new date;
//! - `approved_by`, the maintainer who approved it, from `governance/maintainers.toml`; an agent
//!   may propose an entry but not apply it (ADR-0021). `/ci/` is code-owned, so a change to the
//!   file also needs a maintainer's review in branch protection;
//! - `reason`, one line.
//!
//! A parity gate or an SC checker can never be quarantined. An entry is refused when its test or
//! suite names parity, names a success criterion (`SC-25`, `sc25_join_detect`), is a checker
//! whose source in RFC-0060 §8.6's table is a success criterion, or is a scenario that
//! `bench/sc-map.toml` maps to one.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::roster::Roster;
use crate::run::Context;

/// The quarantine file, relative to the workspace root.
pub const FILE: &str = "ci/quarantine.toml";
/// The maintainer roster, relative to the workspace root.
const ROSTER: &str = "governance/maintainers.toml";
/// RFC-0060, whose §8.6 table names the checkers and their sources.
const RFC: &str = "plan/rfcs/RFC-0060-testing-and-correctness.md";
/// The success-criterion map, whose patterns name the scenarios that measure each criterion.
const SC_MAP: &str = "bench/sc-map.toml";
/// The repository whose issues track a quarantine.
const ISSUES: &str = "https://github.com/syntropyai-hq/syntropy/issues/";
/// The longest a quarantine lasts (RFC-0060 §13).
pub const MAX_DAYS: i64 = 14;
/// The layers whose tests may be quarantined (RFC-0060 §13).
const LAYERS: [&str; 2] = ["L7", "L8"];

/// `ci/quarantine.toml`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    quarantine: Vec<Entry>,
}

/// One quarantined test.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub test: String,
    pub suite: String,
    pub layer: String,
    pub issue: String,
    pub added: String,
    pub expires: String,
    pub approved_by: String,
    pub reason: String,
}

/// What an entry may not name: the parity gates and SC checkers of the repository.
#[derive(Debug, Default)]
pub struct Protected {
    /// Checkers whose source in RFC-0060 §8.6 is a success criterion.
    checkers: BTreeSet<String>,
    /// Scenario ID patterns from `bench/sc-map.toml`: `*` is one segment, or one or more as the
    /// last segment.
    scenarios: Vec<String>,
}

/// Parses the quarantine file.
fn parse(text: &str) -> Result<Vec<Entry>, String> {
    toml::from_str::<File>(text)
        .map(|file| file.quarantine)
        .map_err(|error| format!("{FILE}: not a quarantine file: {error}"))
}

/// Every problem with `entries` on day `today` (days since 1970-01-01, UTC).
pub fn problems(
    entries: &[Entry],
    roster: &Roster,
    protected: &Protected,
    today: i64,
) -> Vec<String> {
    let mut problems = Vec::new();
    let mut seen = BTreeSet::new();
    for entry in entries {
        let at = format!("{FILE}: `{}`", entry.test);
        let mut say = |problem: String| problems.push(format!("{at}: {problem}"));
        if entry.test.trim().is_empty() || entry.suite.trim().is_empty() {
            say("`test` and `suite` must name the test and the suite that runs it".into());
        }
        if !seen.insert((entry.suite.clone(), entry.test.clone())) {
            say("listed twice".into());
        }
        if !LAYERS.contains(&entry.layer.as_str()) {
            say(format!(
                "layer `{}`: only a non-deterministic test (L7, L8) may be quarantined \
                 (RFC-0060 §13)",
                entry.layer
            ));
        }
        if let Some(why) = protected.refuses(entry) {
            say(format!("{why}, which cannot be quarantined (RFC-0060 §13)"));
        }
        if !is_issue_link(&entry.issue) {
            say(format!(
                "issue `{}` is not a tracking issue link `{ISSUES}<n>` (RFC-0060 §13)",
                entry.issue
            ));
        }
        if !roster.is_maintainer(&entry.approved_by) {
            say(format!(
                "approved_by `{}` is not a maintainer in {ROSTER}; an agent may propose a \
                 quarantine but only a maintainer approves it (RFC-0060 §13, ADR-0021)",
                entry.approved_by
            ));
        }
        if entry.reason.trim().is_empty() {
            say("`reason` is empty".into());
        }
        let added = days(&entry.added);
        let expires = days(&entry.expires);
        if added.is_none() {
            say(format!("added `{}` is not a YYYY-MM-DD date", entry.added));
        }
        if expires.is_none() {
            say(format!(
                "expires `{}` is not a YYYY-MM-DD date",
                entry.expires
            ));
        }
        if let (Some(added), Some(expires)) = (added, expires) {
            if expires < added || expires - added > MAX_DAYS {
                say(format!(
                    "expires {} is not within {MAX_DAYS} days after added {} (RFC-0060 §13)",
                    entry.expires, entry.added
                ));
            }
            if added > today {
                say(format!("added {} is in the future", entry.added));
            }
            if today > expires {
                say(format!(
                    "expired on {}: the test is back in the required set; remove the entry, or \
                     a maintainer renews it (RFC-0060 §13)",
                    entry.expires
                ));
            }
        }
    }
    problems
}

/// The entries in force on day `today`: those not yet expired. An expired entry no longer takes
/// its test out of the gates, even before `lint` makes someone remove it.
pub fn active(entries: &[Entry], today: i64) -> Vec<&Entry> {
    entries
        .iter()
        .filter(|entry| days(&entry.expires).is_some_and(|expires| today <= expires))
        .collect()
}

impl Protected {
    /// Reads the protected names from RFC-0060 §8.6 and `bench/sc-map.toml`.
    fn load(root: &Path) -> Result<Protected, String> {
        let rfc = read(root, RFC)?;
        let checkers = sc_checkers(&rfc);
        if checkers.is_empty() {
            return Err(format!(
                "{RFC}: §8.6 names no checker with a success criterion as its source"
            ));
        }
        let map: toml::Value =
            toml::from_str(&read(root, SC_MAP)?).map_err(|e| format!("{SC_MAP}: {e}"))?;
        let scenarios = map
            .get("sc")
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|row| row.get("patterns").and_then(toml::Value::as_array))
            .flatten()
            .filter_map(toml::Value::as_str)
            .map(str::to_owned)
            .collect();
        Ok(Protected {
            checkers,
            scenarios,
        })
    }

    /// Why `entry` names a parity gate or an SC checker, if it does.
    fn refuses(&self, entry: &Entry) -> Option<String> {
        for name in [&entry.suite, &entry.test] {
            let lower = name.to_ascii_lowercase();
            if lower.contains("parity") {
                return Some(format!("`{name}` is a parity gate"));
            }
            if names_criterion(&lower) {
                return Some(format!("`{name}` is an SC checker"));
            }
            for segment in segments(&lower) {
                if self.checkers.contains(segment) {
                    return Some(format!(
                        "`{name}` runs `{segment}`, an SC checker (RFC-0060 §8.6)"
                    ));
                }
            }
            if let Some(pattern) = self.scenarios.iter().find(|p| scenario_matches(p, name)) {
                return Some(format!(
                    "`{name}` is a scenario that measures a success criterion ({SC_MAP}: \
                     `{pattern}`)"
                ));
            }
        }
        None
    }
}

/// The checkers of RFC-0060 §8.6's table whose source cell names a success criterion.
fn sc_checkers(rfc: &str) -> BTreeSet<String> {
    rfc.lines()
        .skip_while(|line| !line.starts_with("#### 8.6 "))
        .skip(1)
        .take_while(|line| !line.starts_with("#### "))
        .filter_map(|line| {
            let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
            let name = cells.first()?.trim().strip_prefix('`')?.strip_suffix('`')?;
            cells
                .last()
                .is_some_and(|source| source.contains("SC-"))
                .then(|| name.to_owned())
        })
        .collect()
}

/// `sc25`, `sc-25` or `sc_25` as a word of `lower`.
fn names_criterion(lower: &str) -> bool {
    let bytes = lower.as_bytes();
    (0..bytes.len()).any(|at| {
        let starts_word = at == 0 || !bytes[at - 1].is_ascii_alphanumeric();
        let rest = &lower[at..];
        let digits = rest
            .strip_prefix("sc-")
            .or_else(|| rest.strip_prefix("sc_"))
            .or_else(|| rest.strip_prefix("sc"));
        starts_word && digits.is_some_and(|d| d.starts_with(|c: char| c.is_ascii_digit()))
    })
}

/// The words of a test name: split at `::`, `.`, `/` and whitespace.
fn segments(lower: &str) -> impl Iterator<Item = &str> {
    lower
        .split(|c: char| c == ':' || c == '.' || c == '/' || c.is_whitespace())
        .filter(|segment| !segment.is_empty())
}

/// Whether scenario ID `id` matches an `sc-map.toml` pattern.
fn scenario_matches(pattern: &str, id: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('.').collect();
    let id: Vec<&str> = id.split('.').collect();
    match pattern.split_last() {
        Some((&"*", head)) => {
            id.len() > head.len() && head.iter().zip(&id).all(|(p, s)| *p == "*" || p == s)
        }
        _ => pattern.len() == id.len() && pattern.iter().zip(&id).all(|(p, s)| *p == "*" || p == s),
    }
}

fn is_issue_link(link: &str) -> bool {
    link.strip_prefix(ISSUES).is_some_and(|number| {
        !number.is_empty() && !number.starts_with('0') && number.bytes().all(|b| b.is_ascii_digit())
    })
}

/// Days since 1970-01-01 of a `YYYY-MM-DD` date, if it is a real date.
pub fn days(date: &str) -> Option<i64> {
    let bytes = date.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let number = |range: std::ops::Range<usize>| -> Option<i64> {
        let text = &date[range];
        text.bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| text.parse().ok())
            .flatten()
    };
    let (year, month, day) = (number(0..4)?, number(5..7)?, number(8..10)?);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=12).contains(&month) || day < 1 || day > month_days[(month - 1) as usize] {
        return None;
    }
    // The civil-to-days conversion of the Gregorian calendar, in 400-year eras.
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let year_of_era = y - era * 400;
    let month_index = (month + 9) % 12;
    let day_of_year = (153 * month_index + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some(era * 146_097 + day_of_era - 719_468)
}

/// Today in UTC, as days since 1970-01-01.
fn today() -> i64 {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    i64::try_from(seconds / 86_400).unwrap_or(0)
}

fn read(root: &Path, relative: &str) -> Result<String, String> {
    std::fs::read_to_string(root.join(relative)).map_err(|e| format!("{relative}: {e}"))
}

/// Reads and checks the quarantine file of the checkout at `root` on day `today`.
fn load(root: &Path, today: i64) -> Result<Vec<Entry>, String> {
    let entries = parse(&read(root, FILE)?)?;
    let roster = Roster::parse(&read(root, ROSTER)?).map_err(|e| format!("{ROSTER}: {}", e.0))?;
    let protected = Protected::load(root)?;
    let found = problems(&entries, &roster, &protected, today);
    if found.is_empty() {
        Ok(entries)
    } else {
        Err(found.join("\n"))
    }
}

/// The `quarantine` step of `cargo xtask ci lint`: every entry is well formed, approved by a
/// maintainer, tracked by an issue, not expired, and names no parity gate or SC checker.
pub fn step(context: &Context) -> Result<(), String> {
    load(&context.root, today()).map(|_| ())
}

/// The lines the release checklist and release notes carry for the quarantine on day `today`
/// (RFC-0060 §13; RFC-0062's release notes row). QUAL-E17-T05's `cargo xtask release check`
/// takes them from here.
pub fn checklist(entries: &[Entry], today: i64) -> Vec<String> {
    let active = active(entries, today);
    if active.is_empty() {
        return vec!["Quarantined tests (RFC-0060 §13): none".to_owned()];
    }
    let mut lines = vec![format!(
        "Quarantined tests (RFC-0060 §13): {}",
        active.len()
    )];
    lines.extend(active.iter().map(|entry| {
        format!(
            "- `{}` in `{}` ({}): {}, approved by {}, expires {}",
            entry.test, entry.suite, entry.layer, entry.issue, entry.approved_by, entry.expires
        )
    }));
    lines
}

/// The step of `cargo xtask ci quarantine`: checks the file as `lint` does, then prints the
/// release checklist's lines to stdout.
pub fn report_step(context: &Context) -> Result<(), String> {
    let today = today();
    let entries = load(&context.root, today)?;
    for line in checklist(&entries, today) {
        println!("{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TODAY: &str = "2026-10-25";

    fn today() -> i64 {
        days(TODAY).unwrap()
    }

    fn roster() -> Roster {
        Roster::parse("maintainers = [\"alice\"]\nagents = [\"review-bot\"]\n").unwrap()
    }

    fn protected() -> Protected {
        Protected {
            checkers: ["capacity", "sc27_coord_change"].map(str::to_owned).into(),
            scenarios: vec!["decode.single.*".into(), "kern.gemv.dense.*".into()],
        }
    }

    fn entry() -> Entry {
        Entry {
            test: "lab::soak::two_node_failover".into(),
            suite: "lab-soak".into(),
            layer: "L8".into(),
            issue: "https://github.com/syntropyai-hq/syntropy/issues/1351".into(),
            added: "2026-10-20".into(),
            expires: "2026-11-03".into(),
            approved_by: "alice".into(),
            reason: "fails then passes on the same commit; see the issue".into(),
        }
    }

    fn check(entry: Entry) -> Vec<String> {
        problems(&[entry], &roster(), &protected(), today())
    }

    #[test]
    fn a_well_formed_entry_passes() {
        assert_eq!(check(entry()), Vec::<String>::new());
    }

    #[test]
    fn an_expired_entry_fails() {
        let expired = Entry {
            added: "2026-10-10".into(),
            expires: "2026-10-24".into(),
            ..entry()
        };
        let found = check(expired.clone());
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("expired on 2026-10-24"), "{found:?}");
        // And it no longer takes its test out of the gates.
        assert!(active(&[expired], today()).is_empty());
        // The last day still counts.
        let last_day = Entry {
            added: "2026-10-11".into(),
            expires: TODAY.into(),
            ..entry()
        };
        assert!(check(last_day.clone()).is_empty());
        assert_eq!(active(&[last_day], today()).len(), 1);
    }

    #[test]
    fn an_entry_longer_than_14_days_fails() {
        let found = check(Entry {
            expires: "2026-11-04".into(),
            ..entry()
        });
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("not within 14 days"), "{found:?}");
        assert!(
            !check(Entry {
                expires: "2026-10-19".into(),
                ..entry()
            })
            .is_empty()
        );
    }

    #[test]
    fn an_entry_without_an_issue_link_fails() {
        for issue in [
            "",
            "#1351",
            "1351",
            "https://github.com/syntropyai-hq/syntropy/issues/",
            "https://github.com/syntropyai-hq/syntropy/pull/1351",
            "https://github.com/someone-else/syntropy/issues/1351",
            "https://github.com/syntropyai-hq/syntropy/issues/12x",
        ] {
            let found = check(Entry {
                issue: issue.into(),
                ..entry()
            });
            assert_eq!(found.len(), 1, "`{issue}`: {found:?}");
            assert!(
                found[0].contains("is not a tracking issue link"),
                "{found:?}"
            );
        }
    }

    #[test]
    fn the_file_refuses_an_entry_missing_a_field() {
        let error = parse(
            "[[quarantine]]\ntest = \"t\"\nsuite = \"s\"\nlayer = \"L8\"\nadded = \"2026-10-20\"\n\
             expires = \"2026-11-03\"\napproved_by = \"alice\"\nreason = \"r\"\n",
        )
        .unwrap_err();
        assert!(error.contains("missing field `issue`"), "{error}");
        assert_eq!(parse("").unwrap(), Vec::new());
    }

    #[test]
    fn a_parity_gate_is_rejected() {
        for (suite, test) in [
            ("hosted-parity-nano", "llama_nano::prefill"),
            ("lab-soak", "testkit::parity::gate::qwen_moe"),
            ("Parity-Smoke", "x"),
        ] {
            let found = check(Entry {
                suite: suite.into(),
                test: test.into(),
                ..entry()
            });
            assert_eq!(found.len(), 1, "{suite} {test}: {found:?}");
            assert!(found[0].contains("is a parity gate"), "{found:?}");
        }
    }

    #[test]
    fn an_sc_checker_is_rejected() {
        for test in [
            "sim::pp3_stage_loss::sc28_resume",
            "SC-25 join detection",
            "sim::churn::capacity",
            "sim::coord::sc27_coord_change",
            "decode.single.llama8b.q4km",
            "kern.gemv.dense.ada.f16",
        ] {
            let found = check(Entry {
                test: test.into(),
                ..entry()
            });
            assert_eq!(found.len(), 1, "`{test}`: {found:?}");
            assert!(found[0].contains("cannot be quarantined"), "{found:?}");
        }
        // A word that only starts like one is not.
        assert!(
            check(Entry {
                test: "lab::scan::scheduler".into(),
                ..entry()
            })
            .is_empty()
        );
    }

    #[test]
    fn a_deterministic_test_is_rejected() {
        for layer in ["L1", "L2", "L3", "L5", "L6", "l8", ""] {
            let found = check(Entry {
                layer: layer.into(),
                ..entry()
            });
            assert_eq!(found.len(), 1, "`{layer}`: {found:?}");
            assert!(
                found[0].contains("only a non-deterministic test"),
                "{found:?}"
            );
        }
    }

    #[test]
    fn only_a_maintainer_approves() {
        for login in ["mallory", "review-bot", "alice[bot]", ""] {
            let found = check(Entry {
                approved_by: login.into(),
                ..entry()
            });
            assert_eq!(found.len(), 1, "`{login}`: {found:?}");
            assert!(found[0].contains("is not a maintainer"), "{found:?}");
        }
    }

    #[test]
    fn duplicates_bad_dates_and_future_entries_fail() {
        let found = problems(&[entry(), entry()], &roster(), &protected(), today());
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("listed twice"));
        for date in ["2026-02-29", "2026-13-01", "26-10-20", "2026-1-20"] {
            assert!(
                !check(Entry {
                    added: date.into(),
                    ..entry()
                })
                .is_empty(),
                "{date}"
            );
        }
        let found = check(Entry {
            added: "2026-10-26".into(),
            expires: "2026-11-01".into(),
            ..entry()
        });
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("in the future"));
    }

    #[test]
    fn days_count_from_1970() {
        assert_eq!(days("1970-01-01"), Some(0));
        assert_eq!(days("2024-01-01"), Some(19_723));
        assert_eq!(days("2024-02-29"), Some(19_782));
        assert_eq!(days("2000-03-01"), Some(11_017));
        assert_eq!(
            days("2026-11-08").unwrap() - days("2026-10-25").unwrap(),
            14
        );
    }

    #[test]
    fn the_rfc_table_names_the_sc_checkers() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let protected = Protected::load(root).unwrap();
        for checker in [
            "capacity",
            "sc25_join_detect",
            "sc27_coord_change",
            "sc28_resume",
            "predicted_vs_simulated",
        ] {
            assert!(
                protected.checkers.contains(checker),
                "{checker}: {protected:?}"
            );
        }
        assert!(!protected.checkers.contains("log_prefix"));
        assert!(!protected.scenarios.is_empty());
    }

    #[test]
    fn the_checklist_lists_the_active_entries() {
        assert_eq!(
            checklist(&[], today()),
            ["Quarantined tests (RFC-0060 §13): none"]
        );
        let expired = Entry {
            test: "old".into(),
            added: "2026-10-01".into(),
            expires: "2026-10-15".into(),
            ..entry()
        };
        assert_eq!(
            checklist(&[entry(), expired], today()),
            [
                "Quarantined tests (RFC-0060 §13): 1",
                "- `lab::soak::two_node_failover` in `lab-soak` (L8): \
                 https://github.com/syntropyai-hq/syntropy/issues/1351, approved by alice, \
                 expires 2026-11-03",
            ]
        );
    }

    #[test]
    fn the_repository_file_passes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        assert!(load(root, super::today()).is_ok());
    }
}
