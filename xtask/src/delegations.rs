// SPDX-License-Identifier: MIT OR Apache-2.0

//! The delegation registry, `governance/delegations.toml`: which maintainer may act for which
//! scope, from which date to which date (RFC-0062 §12.3, ADR-0021 decision 4).
//!
//! One `[[delegation]]` table per entry:
//!
//! ```toml
//! [[delegation]]
//! delegate = "alice"            # a GitHub login in governance/maintainers.toml
//! scope = "tuf-root-custody"    # one of SCOPES
//! granted = "2026-11-01"        # YYYY-MM-DD
//! revoked = "2027-01-01"        # optional; the first day the delegation no longer holds
//! ```
//!
//! A file of comments and no tables is valid: nothing is delegated yet. Every problem the parser
//! finds is reported, one per line, so a registry is never half read. [`step`] is the `lint` step;
//! [`Registry::permits`] is the resolver the checks of QUAL-E1-T18 and QUAL-E2-T03 will read.

use std::collections::BTreeSet;
use std::fmt;

use serde::Deserialize;

use crate::codeowners::ROSTER;
use crate::roster::Roster;
use crate::run::Context;

/// Where the registry lives, relative to the workspace root.
pub const REGISTRY: &str = "governance/delegations.toml";

/// Every scope a delegation can name. The names after `security-deputy` are provisional:
/// QUAL-E1-T18 and QUAL-E2-T03 read them and may rename them.
pub const SCOPES: &[&str] = &[
    "tuf-root-custody",
    "tuf-targets-custody",
    "org-owner",
    "aws-break-glass",
    "security-deputy",
    "/hw run",
    "gate-labels",
    "rc:start",
    "promote",
];

/// RFC-0062 §7.2: three root-token holders.
pub const MAX_ROOT_CUSTODY: usize = 3;

/// RFC-0062 §12.6: at most two organisation owners.
pub const MAX_ORG_OWNERS: usize = 2;

/// A calendar date, `YYYY-MM-DD`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    year: u16,
    month: u8,
    day: u8,
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl Date {
    /// Reads `YYYY-MM-DD` strictly: four digits, two, two, a date that really exists, and a year
    /// from 2000 to 2999.
    pub fn parse(text: &str) -> Result<Date, String> {
        let digits = |bytes: &[u8]| bytes.iter().all(|b| b.is_ascii_digit());
        let bytes = text.as_bytes();
        if bytes.len() != 10
            || !digits(&bytes[..4])
            || bytes[4] != b'-'
            || !digits(&bytes[5..7])
            || bytes[7] != b'-'
            || !digits(&bytes[8..10])
        {
            return Err(format!("`{text}` is not a date written `YYYY-MM-DD`"));
        }
        let year = value(&bytes[..4]);
        let month = value(&bytes[5..7]) as u8;
        let day = value(&bytes[8..10]) as u8;
        if !(2000..=2999).contains(&year) {
            return Err(format!("`{text}` is outside the years 2000 to 2999"));
        }
        let days = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 if year.is_multiple_of(400)
                || (year.is_multiple_of(4) && !year.is_multiple_of(100)) =>
            {
                29
            }
            2 => 28,
            _ => return Err(format!("`{text}` is not a real date: month {month}")),
        };
        if day == 0 || day > days {
            return Err(format!(
                "`{text}` is not a real date: month {month} has days 1 to {days}"
            ));
        }
        Ok(Date { year, month, day })
    }
}

/// Four or two decimal digits as a number.
fn value(bytes: &[u8]) -> u16 {
    bytes
        .iter()
        .fold(0, |number, byte| number * 10 + (byte - b'0') as u16)
}

/// One delegation.
#[derive(Debug, PartialEq, Eq)]
pub struct Delegation {
    pub delegate: String,
    pub scope: String,
    pub granted: Date,
    pub revoked: Option<Date>,
}

impl Delegation {
    /// Whether the delegation holds on `on`: granted on or before it, and not yet revoked,
    /// `revoked` being the first day the delegation no longer holds.
    pub fn active_on(&self, on: Date) -> bool {
        self.granted <= on && self.revoked.is_none_or(|revoked| on < revoked)
    }
}

/// The registry as read.
#[derive(Debug, PartialEq, Eq)]
pub struct Registry {
    pub delegations: Vec<Delegation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRegistry {
    #[serde(default)]
    delegation: Vec<RawDelegation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDelegation {
    delegate: Option<String>,
    scope: Option<String>,
    granted: Option<String>,
    revoked: Option<String>,
}

impl Registry {
    /// Parses the registry's text against the maintainer `roster`, with every problem found.
    pub fn parse(text: &str, roster: &Roster) -> Result<Registry, Vec<String>> {
        let raw: RawRegistry = match toml::from_str(text) {
            Ok(raw) => raw,
            Err(error) => return Err(vec![format!("the registry is not valid TOML: {error}")]),
        };
        let mut problems = Vec::new();
        let mut entries = Vec::new();
        for (index, raw) in raw.delegation.into_iter().enumerate() {
            let number = index + 1;
            let at = |why: String| format!("delegation {number}: {why}");
            for (key, given) in [
                ("delegate", &raw.delegate),
                ("scope", &raw.scope),
                ("granted", &raw.granted),
            ] {
                if given.is_none() {
                    problems.push(at(format!("missing `{key}`")));
                }
            }
            let (Some(delegate), Some(scope), Some(started)) =
                (raw.delegate, raw.scope, raw.granted)
            else {
                continue;
            };
            let found = problems.len();
            if !SCOPES.contains(&scope.as_str()) {
                problems.push(at(format!(
                    "scope `{scope}` is not one of {}",
                    SCOPES.join(", ")
                )));
            }
            let granted = match Date::parse(&started) {
                Ok(date) => Some(date),
                Err(why) => {
                    problems.push(at(format!("`granted` {why}")));
                    None
                }
            };
            let revoked = match raw.revoked.as_ref() {
                None => None,
                Some(text) => match Date::parse(text) {
                    Ok(date) => {
                        if let Some(granted) = granted.filter(|granted| date <= *granted) {
                            problems.push(at(format!(
                                "`revoked` {date} is on or before `granted` {granted}"
                            )));
                        }
                        Some(date)
                    }
                    Err(why) => {
                        problems.push(at(format!("`revoked` {why}")));
                        None
                    }
                },
            };
            if !roster.is_maintainer(&delegate) {
                problems.push(at(format!(
                    "`{delegate}` is not a maintainer in {ROSTER}; only a maintainer may hold a \
                     delegation (RFC-0062 §12.3)"
                )));
            }
            if let Some(granted) = granted.filter(|_| problems.len() == found) {
                entries.push(Delegation {
                    delegate,
                    scope,
                    granted,
                    revoked,
                });
            }
        }
        for (index, entry) in entries.iter().enumerate() {
            for (other, earlier) in entries[..index].iter().enumerate() {
                if earlier.delegate.eq_ignore_ascii_case(&entry.delegate)
                    && earlier.scope == entry.scope
                    && earlier.granted == entry.granted
                {
                    problems.push(format!(
                        "delegation {}: `{}` is delegated `{}` from {} twice, as in delegation {}",
                        index + 1,
                        entry.delegate,
                        entry.scope,
                        entry.granted,
                        other + 1
                    ));
                }
            }
        }
        // The holders at once peak on a day some delegation begins, so count the distinct
        // holders active on each grant date. A delegation that has ended holds nothing then.
        for (scope, limit, rule) in [
            ("tuf-root-custody", MAX_ROOT_CUSTODY, "RFC-0062 §7.2"),
            ("org-owner", MAX_ORG_OWNERS, "RFC-0062 §12.6"),
        ] {
            let worst = entries
                .iter()
                .filter(|entry| entry.scope == scope)
                .map(|begin| {
                    entries
                        .iter()
                        .filter(|entry| entry.scope == scope && entry.active_on(begin.granted))
                        .map(|entry| entry.delegate.to_ascii_lowercase())
                        .collect::<BTreeSet<_>>()
                        .len()
                })
                .max()
                .unwrap_or(0);
            if worst > limit {
                problems.push(format!(
                    "the registry delegates `{scope}` to {worst} delegates at once; only {limit} \
                     may hold it ({rule})"
                ));
            }
        }
        if problems.is_empty() {
            Ok(Registry {
                delegations: entries,
            })
        } else {
            Err(problems)
        }
    }

    /// Whether `login` may act for `scope` on `on`. Logins compare ignoring ASCII case, as
    /// GitHub's do.
    #[cfg_attr(not(test), allow(dead_code))] // read by QUAL-E1-T18 and QUAL-E2-T03
    pub fn permits(&self, login: &str, scope: &str, on: Date) -> bool {
        self.delegations.iter().any(|entry| {
            entry.delegate.eq_ignore_ascii_case(login)
                && entry.scope == scope
                && entry.active_on(on)
        })
    }
}

/// The `delegations` step of `lint`: the registry reads, and every delegate is a maintainer.
pub fn step(context: &Context) -> Result<(), String> {
    let read = |relative: &str| {
        std::fs::read_to_string(context.root.join(relative))
            .map_err(|error| format!("{relative}: {error}"))
    };
    let roster = Roster::parse(&read(ROSTER)?).map_err(|error| format!("{ROSTER}: {error}"))?;
    Registry::parse(&read(REGISTRY)?, &roster).map_err(|problems| {
        problems
            .into_iter()
            .map(|problem| format!("{REGISTRY}: {problem}"))
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Args;

    const MAINTAINERS: &str =
        "maintainers = [\"alice\", \"bob\", \"carol\", \"dave\"]\nagents = [\"review-agent\"]";

    fn roster() -> Roster {
        Roster::parse(MAINTAINERS).unwrap()
    }

    /// One entry as the file holds it.
    fn entry(delegate: &str, scope: &str, granted: &str, revoked: Option<&str>) -> String {
        let mut text = format!("[[delegation]]\ndelegate = \"{delegate}\"\n");
        text.push_str(&format!("scope = \"{scope}\"\n"));
        if let Some(date) = revoked {
            text.push_str(&format!("revoked = \"{date}\"\n"));
        }
        text.push_str(&format!("granted = \"{granted}\"\n"));
        text
    }

    /// A granted entry with no end, `revoked` on the given day.
    fn given(delegate: &str, scope: &str) -> String {
        entry(delegate, scope, "2026-11-01", None)
    }

    /// `given` with the line naming `key` removed.
    fn without(key: &str) -> String {
        given("alice", "promote")
            .lines()
            .filter(|line| !line.starts_with(key))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    #[test]
    fn entries_and_an_empty_registry_parse() {
        let text = String::from("# Delegations.\n")
            + &given("alice", "tuf-root-custody")
            + &entry("bob", "/hw run", "2026-11-01", Some("2027-01-01"));
        let registry = Registry::parse(&text, &roster()).unwrap();
        assert_eq!(registry.delegations.len(), 2);
        assert_eq!(registry.delegations[1].scope, "/hw run");
        assert_eq!(
            registry.delegations[1].revoked,
            Some(Date::parse("2027-01-01").unwrap())
        );
        assert!(
            Registry::parse("# Nothing delegated yet.\n", &roster())
                .unwrap()
                .delegations
                .is_empty()
        );
    }

    #[test]
    fn every_bad_entry_is_named_by_its_number() {
        let bad: Vec<(String, &str)> = vec![
            (without("scope"), "missing `scope`"),
            (without("granted"), "missing `granted`"),
            (without("delegate"), "missing `delegate`"),
            (given("alice", "kernel"), "not one of tuf-root-custody"),
            (
                entry("alice", "promote", "2026-02-30", None),
                "not a real date",
            ),
            (
                entry("alice", "promote", "2026-13-01", None),
                "not a real date",
            ),
            (
                entry("alice", "promote", "tomorrow", None),
                "not a date written",
            ),
            (
                given("alice", "promote") + &given("ALICE", "promote"),
                "twice",
            ),
            (
                entry("alice", "promote", "2026-11-01", Some("2026-10-01")),
                "on or before",
            ),
            (given("mallory", "promote"), "is not a maintainer"),
            (given("review-agent", "promote"), "is not a maintainer"),
            ("delegate = \"alice\"".into(), "not valid TOML"),
        ];
        for (text, wanted) in bad {
            let problems = Registry::parse(&text, &roster()).unwrap_err();
            assert_eq!(problems.len(), 1, "{text}: {problems:?}");
            assert!(problems[0].contains(wanted), "{text}: {}", problems[0]);
        }
    }

    #[test]
    fn custody_and_ownership_have_limits() {
        let held = |scope: &str, count: usize| {
            let text = ["alice", "bob", "carol", "dave"]
                .iter()
                .take(count)
                .map(|name| given(name, scope))
                .collect::<Vec<_>>()
                .join("");
            Registry::parse(&text, &roster())
        };
        assert!(held("tuf-root-custody", 3).is_ok());
        let error = held("tuf-root-custody", 4).unwrap_err();
        assert!(error[0].contains("only 3 may hold it"), "{error:?}");
        // Four overlapping holders whose `revoked` dates all lie ahead still breach the limit.
        let future = |n: usize| {
            (0..n)
                .map(|i| {
                    format!(
                        "[[delegation]]\ndelegate = \"{}\"\nscope = \"tuf-root-custody\"\n\
                         granted = \"2026-11-0{}\"\nrevoked = \"2099-01-01\"\n",
                        ["alice", "bob", "carol", "dave"][i],
                        i + 1
                    )
                })
                .collect::<String>()
        };
        assert!(Registry::parse(&future(4), &roster()).is_err());
        assert!(Registry::parse(&future(3), &roster()).is_ok());
        // A fourth holder who starts after another one ends is fine.
        let handover = future(3).replace("revoked = \"2099-01-01\"", "revoked = \"2026-11-04\"")
            + "[[delegation]]\ndelegate = \"dave\"\nscope = \"tuf-root-custody\"\ngranted = \"2026-11-04\"\n";
        assert!(Registry::parse(&handover, &roster()).is_ok(), "{handover}");
        assert!(held("org-owner", 2).is_ok());
        let error = held("org-owner", 3).unwrap_err();
        assert!(
            error[0].contains("org-owner") && error[0].contains("only 2"),
            "{error:?}"
        );
        // A delegation that has ended holds nothing.
        let text = ["alice", "bob", "carol"]
            .iter()
            .map(|name| given(name, "tuf-root-custody"))
            .collect::<Vec<_>>()
            .join("")
            + &entry("dave", "tuf-root-custody", "2026-10-01", Some("2026-11-01"));
        assert!(Registry::parse(&text, &roster()).is_ok());
    }

    #[test]
    fn the_resolver_reads_grants_and_ends() {
        let registry = Registry::parse(&given("alice", "/hw run"), &roster()).unwrap();
        let day = |text: &str| Date::parse(text).unwrap();
        assert!(registry.permits("alice", "/hw run", day("2026-11-02")));
        assert!(registry.permits("ALICE", "/hw run", day("2026-11-02")));
        assert!(registry.permits("alice", "/hw run", day("2026-11-01")));
        assert!(!registry.permits("alice", "promote", day("2026-11-02")));
        assert!(!registry.permits("bob", "/hw run", day("2026-11-02")));
        assert!(!registry.permits("alice", "/hw run", day("2026-10-31")));
        let ended = Registry::parse(
            &entry("alice", "/hw run", "2026-11-01", Some("2026-12-01")),
            &roster(),
        )
        .unwrap();
        assert!(ended.permits("alice", "/hw run", day("2026-11-30")));
        assert!(!ended.permits("alice", "/hw run", day("2026-12-01")));
        assert!(!ended.permits("alice", "/hw run", day("2027-01-01")));
    }

    #[test]
    fn dates_are_read_strictly() {
        assert!(Date::parse("2028-02-29").is_ok());
        assert!(Date::parse("2100-02-29").is_err());
        assert!(Date::parse("2027-02-29").is_err());
        assert!(Date::parse("2026-01-1").is_err());
        assert!(Date::parse("1999-12-31").is_err());
        assert_eq!(Date::parse("2026-11-01").unwrap().to_string(), "2026-11-01");
    }

    #[test]
    fn the_step_reads_both_files_under_the_root() {
        let dir = std::env::temp_dir().join(format!("xtask-delegations-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("governance")).unwrap();
        std::fs::write(dir.join("governance/maintainers.toml"), MAINTAINERS).unwrap();
        std::fs::write(dir.join(REGISTRY), "# Nothing delegated yet.\n").unwrap();
        let context = Context {
            root: dir.clone(),
            args: Args::default(),
        };
        step(&context).unwrap();
        std::fs::write(dir.join(REGISTRY), given("mallory", "promote")).unwrap();
        let failure = step(&context).unwrap_err();
        assert!(
            failure.starts_with("governance/delegations.toml: delegation 1:"),
            "{failure}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
