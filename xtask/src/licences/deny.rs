// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask deny`: cargo-deny with the checkout's `deny.toml` (ADR-0018 §3, RFC-0062 §11.2).
//!
//! Two `lint` steps call it:
//!
//! - `cargo xtask deny` runs the `licenses`, `bans` and `sources` checks. They read `Cargo.lock`
//!   and the crate sources cargo has already fetched, so they give the same verdict offline.
//! - `cargo xtask deny --advisories` runs the `advisories` check, which first fetches the RustSec
//!   database into `$CARGO_HOME/advisory-dbs`. Without network access the fetch fails and so does
//!   the step, even with a database cached from an earlier run: the check fails closed rather
//!   than pass on stale data. It never runs with `--offline`, which would use a stale cache.
//!
//! The version of cargo-deny is pinned, because a new release can change what a check rejects.
//! A missing or different cargo-deny fails the step with the command that installs the pinned
//! one.
//!
//! The `deny` step also checks what cargo-deny does not: each entry of `[advisories] ignore`
//! carries an issue link and an expiry date that has not passed (RFC-0062 §11.2). It reads the
//! file, so it runs offline too.

use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use super::DENY_TOML;
use super::toml::strip_comment;

/// The cargo-deny release `lint` runs. CI installs exactly this version (QUAL-E1-T06).
pub const VERSION: &str = "0.20.2";

/// Which cargo-deny checks to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checks {
    /// `licenses`, `bans` and `sources`: no network needed.
    Policy,
    /// `advisories`: fetches the RustSec database.
    Advisories,
}

impl Checks {
    fn names(self) -> &'static [&'static str] {
        match self {
            Checks::Policy => &["licenses", "bans", "sources"],
            Checks::Advisories => &["advisories"],
        }
    }
}

/// The command that installs the pinned cargo-deny.
pub fn install_hint() -> String {
    format!("cargo install --locked cargo-deny@{VERSION}")
}

/// The installed cargo-deny version, from `cargo deny --version`, or why it cannot be read.
pub fn installed_version() -> Result<String, String> {
    let output = Command::new(crate::registry::cargo())
        .args(["deny", "--version"])
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("could not start `cargo deny`: {error}"))?;
    if !output.status.success() {
        return Err("cargo-deny is not installed".to_owned());
    }
    parse_version(&String::from_utf8_lossy(&output.stdout))
        .ok_or_else(|| "`cargo deny --version` printed no version".to_owned())
}

/// The version in `cargo-deny 0.20.2`.
fn parse_version(stdout: &str) -> Option<String> {
    let version = stdout.trim().strip_prefix("cargo-deny ")?;
    let version = version.split_whitespace().next()?;
    (!version.is_empty()).then(|| version.to_owned())
}

/// Every problem with the `ignore` list of `[advisories]` in `text`, a `deny.toml`, on `today`
/// (`YYYY-MM-DD`). Each entry must be on its own line, and a comment on that line or on the
/// comment lines just above it must give an issue link (`https://.../issues/<n>`) and an expiry
/// date (`expires YYYY-MM-DD`) no earlier than `today`.
///
/// The check fails closed. Table headers and keys are read as TOML reads them, so `[ advisories ]`
/// and a quoted `"ignore"` are the same list. Any other spelling of `advisories.ignore`, such as a
/// dotted key at the top level, an inline `advisories = { … }` table or an `[advisories.…]` table,
/// is a problem, because its entries cannot be checked. So is a quoted key with an escape, and an
/// advisory ID (`RUSTSEC-…`, `GHSA-…`) anywhere outside the one list the check reads.
pub fn ignore_problems(text: &str, today: &str) -> Vec<String> {
    const ADVISORIES: &str = "advisories";
    const IGNORE: &str = "ignore";
    let mut problems = Vec::new();
    let mut table: Vec<String> = Vec::new();
    let mut in_ignore = false;
    let mut notes: Vec<&str> = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let number = index + 1;
        let at = format!("{DENY_TOML}:{number}");
        let code = strip_comment(raw).trim();
        let comment = raw.trim()[code.len()..]
            .trim()
            .trim_start_matches('#')
            .trim();
        if !in_ignore {
            if let Some(id) = advisory_id(code) {
                problems.push(format!(
                    "{at}: the advisory `{id}` appears outside the `ignore = [` list of \
                     `[advisories]`, where its issue link and expiry cannot be checked \
                     (RFC-0062 §11.2)"
                ));
            }
            if code.starts_with('[') {
                let inner = code.trim_start_matches('[').trim_end_matches(']');
                match key_path(inner) {
                    Ok(Some(path)) => {
                        if path.first().is_some_and(|p| p == ADVISORIES) && path.len() > 1 {
                            problems.push(format!(
                                "{at}: `{code}` is not read; `[advisories]` holds `ignore` as an \
                                 array (RFC-0062 §11.2)"
                            ));
                        }
                        table = path;
                    }
                    Ok(None) => {}
                    Err(error) => problems.push(format!("{at}: {error}")),
                }
                continue;
            }
            let Some((key, value)) = split_key_value(code) else {
                continue;
            };
            let key = match key_path(key) {
                Ok(Some(key)) => key,
                Ok(None) => continue,
                Err(error) => {
                    problems.push(format!("{at}: {error}"));
                    continue;
                }
            };
            let full: Vec<&str> = table.iter().chain(&key).map(String::as_str).collect();
            let value = value.trim();
            if full == [ADVISORIES, IGNORE] && table == [ADVISORIES] {
                match value {
                    "[]" => {}
                    "[" => {
                        in_ignore = true;
                        notes.clear();
                    }
                    _ => problems.push(format!(
                        "{at}: write `ignore` as `[`, one entry per line with its comment, then \
                         `]` (RFC-0062 §11.2)"
                    )),
                }
            } else if full.starts_with(&[ADVISORIES, IGNORE]) || full == [ADVISORIES] {
                problems.push(format!(
                    "{at}: `{}` sets the advisories' `ignore` list in a form this check does not \
                     read; write it as `ignore = [` inside `[advisories]` (RFC-0062 §11.2)",
                    key.join(".")
                ));
            }
            continue;
        }
        if code.starts_with(']') {
            in_ignore = false;
            continue;
        }
        if code.is_empty() {
            if !comment.is_empty() {
                notes.push(comment);
            }
            continue;
        }
        let id = code
            .split('"')
            .nth(1)
            .filter(|id| !id.is_empty())
            .unwrap_or(code);
        let mut note = notes.join(" ");
        note.push(' ');
        note.push_str(comment);
        notes.clear();
        let at = format!("{at}: the ignored advisory `{id}`");
        if !has_issue_link(&note) {
            problems.push(format!(
                "{at} has no issue link (`https://.../issues/<n>`) in its comment (RFC-0062 §11.2)"
            ));
        }
        match expiry(&note) {
            None => problems.push(format!(
                "{at} has no expiry date (`expires YYYY-MM-DD`) in its comment (RFC-0062 §11.2)"
            )),
            Some(date) if date.as_str() < today => {
                problems.push(format!("{at} expired on {date}; fix it or renew the entry"));
            }
            Some(_) => {}
        }
    }
    if in_ignore {
        problems.push(format!(
            "{DENY_TOML}: the `ignore` array has no closing `]`"
        ));
    }
    problems
}

/// The parts of the TOML key `text`, such as `["advisories", "ignore"]` for
/// `advisories . "ignore"`. `Ok(None)` when `text` is not a key, such as the inside of an inline
/// table on a continuation line. A quoted part with an escape is an error, because this check
/// does not decode escapes and must not misread a key.
fn key_path(text: &str) -> Result<Option<Vec<String>>, String> {
    let mut parts = Vec::new();
    let mut rest = text.trim();
    loop {
        let (part, after) = if let Some(body) = rest.strip_prefix('"') {
            let Some(end) = body.find('"') else {
                return Ok(None);
            };
            if body[..end].contains('\\') {
                return Err(format!(
                    "the quoted key `\"{}\"` has an escape, which this check does not read; \
                     write the key plainly",
                    &body[..end]
                ));
            }
            (&body[..end], &body[end + 1..])
        } else if let Some(body) = rest.strip_prefix('\'') {
            let Some(end) = body.find('\'') else {
                return Ok(None);
            };
            (&body[..end], &body[end + 1..])
        } else {
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
                .unwrap_or(rest.len());
            if end == 0 {
                return Ok(None);
            }
            (&rest[..end], &rest[end..])
        };
        parts.push(part.to_owned());
        let after = after.trim_start();
        if after.is_empty() {
            return Ok(Some(parts));
        }
        let Some(next) = after.strip_prefix('.') else {
            return Ok(None);
        };
        rest = next.trim_start();
    }
}

/// `code` split at its first `=` outside a string: the key and the value.
fn split_key_value(code: &str) -> Option<(&str, &str)> {
    let mut quote = None;
    for (at, c) in code.char_indices() {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(q), c) if c == q => quote = None,
            (None, '=') => return Some((&code[..at], &code[at + 1..])),
            _ => {}
        }
    }
    None
}

/// The first advisory ID in `code`: `RUSTSEC-YYYY-NNNN` or `GHSA-xxxx-xxxx-xxxx`.
fn advisory_id(code: &str) -> Option<&str> {
    let start = ["RUSTSEC-", "GHSA-"]
        .iter()
        .filter_map(|prefix| code.find(prefix))
        .min()?;
    let rest = &code[start..];
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Whether `note` holds an `https://` link to an issue: `/issues/` followed by its number.
fn has_issue_link(note: &str) -> bool {
    note.split_whitespace().any(|word| {
        word.starts_with("https://")
            && word.split_once("/issues/").is_some_and(|(_, rest)| {
                let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
                !digits.is_empty()
            })
    })
}

/// The date after `expires ` in `note`, if it is a valid `YYYY-MM-DD`.
fn expiry(note: &str) -> Option<String> {
    let (_, rest) = note.split_once("expires ")?;
    let date: String = rest.chars().take(10).collect();
    let bytes = date.as_bytes();
    let digits = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    if !(digits(0..4) && digits(5..7) && digits(8..10)) {
        return None;
    }
    let month: u32 = date[5..7].parse().ok()?;
    let day: u32 = date[8..10].parse().ok()?;
    ((1..=12).contains(&month) && (1..=31).contains(&day)).then_some(date)
}

/// Today's date in UTC, as `YYYY-MM-DD`.
pub fn today() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    civil_date(i64::try_from(seconds / 86_400).unwrap_or(0))
}

/// The proleptic Gregorian date `days` after 1970-01-01, as `YYYY-MM-DD`.
fn civil_date(days: i64) -> String {
    // The days-to-civil conversion of the Gregorian calendar, in 400-year eras.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Runs `checks` on the workspace of `manifest` with the `deny.toml` of `checkout`, and writes
/// the verdict to `log`. cargo-deny's own report goes to the inherited stderr. Returns whether
/// the checks passed.
pub fn run(
    checkout: &Path,
    manifest: &Path,
    checks: Checks,
    log: &mut dyn Write,
) -> io::Result<bool> {
    let label = match checks {
        Checks::Policy => "deny",
        Checks::Advisories => "deny --advisories",
    };
    let names = checks.names().join(", ");
    match installed_version() {
        Ok(version) if version == VERSION => {}
        Ok(version) => {
            writeln!(
                log,
                "{label}: FAIL: lint pins cargo-deny {VERSION}, and {version} is installed; \
                 run `{}`",
                install_hint()
            )?;
            return Ok(false);
        }
        Err(error) => {
            writeln!(
                log,
                "{label}: FAIL: {error}; lint pins cargo-deny {VERSION}: run `{}`",
                install_hint()
            )?;
            return Ok(false);
        }
    }
    let config = checkout.join(DENY_TOML);
    // RFC-0062 §11.2's rule for ignored advisories, which cargo-deny does not check.
    let mut ignored_ok = true;
    if checks == Checks::Policy {
        let problems = match std::fs::read_to_string(&config) {
            Ok(text) => ignore_problems(&text, &today()),
            Err(error) => vec![format!("cannot read {}: {error}", config.display())],
        };
        for problem in &problems {
            writeln!(log, "{label}: {problem}")?;
        }
        ignored_ok = problems.is_empty();
    }
    let manifest = match std::fs::canonicalize(manifest) {
        Ok(manifest) => manifest,
        Err(error) => {
            writeln!(
                log,
                "{label}: FAIL: no manifest at {}: {error}",
                manifest.display()
            )?;
            return Ok(false);
        }
    };
    // Run from the workspace's own directory, so cargo reads its `.cargo/config.toml`.
    let dir = manifest.parent().unwrap_or(checkout);
    writeln!(
        log,
        "{label}: cargo-deny {VERSION}, {names}, with {} on {}",
        config.display(),
        manifest.display()
    )?;
    log.flush()?;
    let status = Command::new(crate::registry::cargo())
        .args(["deny", "--locked", "--color", "never"])
        .arg("--config")
        .arg(&config)
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("check")
        .args(checks.names())
        .current_dir(dir)
        .stdin(Stdio::null())
        .status();
    let pass = match status {
        Ok(status) if status.success() && !ignored_ok => {
            writeln!(
                log,
                "{label}: FAIL: {names} passed, but an ignored advisory breaks RFC-0062 §11.2 \
                 (above)"
            )?;
            false
        }
        Ok(status) if status.success() => {
            writeln!(log, "{label}: PASS: {names} (ADR-0018 §3, RFC-0062 §11.2)")?;
            true
        }
        Ok(status) => {
            writeln!(log, "{label}: FAIL: {names}: cargo-deny {status}")?;
            if checks == Checks::Advisories {
                writeln!(
                    log,
                    "{label}: either an advisory matched (above), or the RustSec database could \
                     not be fetched; this check needs network access and fails closed without it"
                )?;
            }
            false
        }
        Err(error) => {
            writeln!(log, "{label}: FAIL: could not start `cargo deny`: {error}")?;
            false
        }
    };
    log.flush()?;
    Ok(pass)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_version_line() {
        assert_eq!(
            parse_version("cargo-deny 0.20.2\n"),
            Some("0.20.2".to_owned())
        );
        assert_eq!(
            parse_version("cargo-deny 0.21.0 (abc 2026-01-01)"),
            Some("0.21.0".to_owned())
        );
        assert_eq!(parse_version("error: no such command: `deny`"), None);
        assert_eq!(parse_version("cargo-deny "), None);
    }

    #[test]
    fn the_policy_checks_need_no_network_and_advisories_run_alone() {
        assert_eq!(Checks::Policy.names(), ["licenses", "bans", "sources"]);
        assert_eq!(Checks::Advisories.names(), ["advisories"]);
    }

    #[test]
    fn civil_dates_count_from_1970() {
        assert_eq!(civil_date(0), "1970-01-01");
        assert_eq!(civil_date(19_723), "2024-01-01");
        assert_eq!(civil_date(19_782), "2024-02-29");
        assert_eq!(civil_date(20_747), "2026-10-21");
        assert_eq!(civil_date(-1), "1969-12-31");
        assert_eq!(today().len(), 10);
    }

    const TODAY: &str = "2026-10-21";

    fn advisories(ignore: &str) -> String {
        format!("[graph]\nignore = [\"x\"]\n\n[advisories]\n{ignore}\nyanked = \"deny\"\n")
    }

    #[test]
    fn an_empty_ignore_list_passes() {
        assert_eq!(
            ignore_problems(&advisories("ignore = []"), TODAY),
            Vec::<String>::new()
        );
        assert_eq!(
            ignore_problems("[advisories]\n", TODAY),
            Vec::<String>::new()
        );
    }

    #[test]
    fn an_entry_with_an_issue_and_a_future_expiry_passes() {
        let text = advisories(
            "ignore = [\n    # https://github.com/syntropyai-hq/syntropy/issues/12, expires 2026-10-21\n    \
             { id = \"RUSTSEC-2026-0001\", reason = \"not reachable\" },\n    \
             \"RUSTSEC-2026-0002\", # https://github.com/o/r/issues/3 expires 2027-01-01\n]",
        );
        assert_eq!(ignore_problems(&text, TODAY), Vec::<String>::new());
    }

    #[test]
    fn an_entry_without_an_issue_or_a_date_or_past_its_date_fails() {
        let text = advisories(
            "ignore = [\n    # expires 2027-01-01\n    \"RUSTSEC-2026-0001\",\n    \
             # https://github.com/o/r/issues/4\n    \"RUSTSEC-2026-0002\",\n    \
             # https://github.com/o/r/issues/5 expires 2026-10-20\n    \"RUSTSEC-2026-0003\",\n    \
             # https://github.com/o/r/pull/6 expires 2026-13-01\n    \"RUSTSEC-2026-0004\",\n]",
        );
        let found = ignore_problems(&text, TODAY);
        assert_eq!(found.len(), 5, "{found:#?}");
        assert!(
            found[0].contains("`RUSTSEC-2026-0001` has no issue link"),
            "{found:#?}"
        );
        assert!(
            found[1].contains("`RUSTSEC-2026-0002` has no expiry date"),
            "{found:#?}"
        );
        assert!(
            found[2].contains("`RUSTSEC-2026-0003` expired on 2026-10-20"),
            "{found:#?}"
        );
        assert!(
            found[3].contains("`RUSTSEC-2026-0004` has no issue link"),
            "{found:#?}"
        );
        assert!(
            found[4].contains("`RUSTSEC-2026-0004` has no expiry date"),
            "{found:#?}"
        );
    }

    #[test]
    fn entries_on_one_line_fail() {
        // The one-line list is unread, so its ID also counts as outside the list.
        let found = ignore_problems(&advisories("ignore = [\"RUSTSEC-2026-0001\"]"), TODAY);
        assert_eq!(found.len(), 2, "{found:#?}");
        assert!(
            found[0].contains("appears outside the `ignore = [` list"),
            "{found:#?}"
        );
        assert!(found[1].contains("one entry per line"), "{found:#?}");
    }

    /// The review's three spellings of `advisories.ignore`, and the other forms TOML allows, each
    /// with one advisory and no comment.
    #[test]
    fn every_spelling_of_the_ignore_list_is_checked_or_refused() {
        for (text, expected) in [
            // Read as the canonical list, so the entry's missing comment is found.
            (
                "[ advisories ]\nignore = [\n    \"RUSTSEC-2020-0071\",\n]\n",
                "`RUSTSEC-2020-0071` has no issue link",
            ),
            (
                "[\"advisories\"]\nignore = [\n    \"RUSTSEC-2020-0071\",\n]\n",
                "`RUSTSEC-2020-0071` has no issue link",
            ),
            (
                "[advisories]\n\"ignore\" = [\n    \"RUSTSEC-2020-0071\",\n]\n",
                "`RUSTSEC-2020-0071` has no issue link",
            ),
            (
                "[advisories]\n'ignore' = [\n    \"RUSTSEC-2020-0071\",\n]\n",
                "`RUSTSEC-2020-0071` has no issue link",
            ),
            // Refused: forms the check does not read.
            (
                "advisories.ignore = [\n    \"RUSTSEC-2020-0071\",\n]\n",
                "`advisories.ignore` sets the advisories' `ignore` list in a form",
            ),
            (
                "advisories . \"ignore\" = []\n",
                "`advisories.ignore` sets the advisories' `ignore` list in a form",
            ),
            (
                "advisories = { ignore = [\"RUSTSEC-2020-0071\"] }\n",
                "`advisories` sets the advisories' `ignore` list in a form",
            ),
            ("[advisories.ignore]\n", "`[advisories.ignore]` is not read"),
            (
                "[[advisories.ignore]]\nid = \"RUSTSEC-2020-0071\"\n",
                "`[[advisories.ignore]]` is not read",
            ),
            (
                "[advisories]\n\"ign\\u006fre\" = []\n",
                "has an escape, which this check does not read",
            ),
        ] {
            let found = ignore_problems(text, TODAY);
            assert!(
                found.iter().any(|p| p.contains(expected)),
                "{text:?}: expected `{expected}` in {found:#?}"
            );
        }
    }

    #[test]
    fn an_advisory_id_outside_the_ignore_list_fails() {
        for text in [
            "[graph]\nexclude = [\"RUSTSEC-2020-0071\"]\n",
            "[advisories]\nignored = [\"GHSA-abcd-efgh-ijkl\"]\nignore = []\n",
            "[advisories]\nignore = [] # but RUSTSEC-2020-0071 in a comment is fine\nx = \"RUSTSEC-2020-0071\"\n",
        ] {
            let found = ignore_problems(text, TODAY);
            assert_eq!(found.len(), 1, "{text:?}: {found:#?}");
            assert!(
                found[0].contains("appears outside the `ignore = [` list"),
                "{found:#?}"
            );
        }
        assert_eq!(
            ignore_problems(
                "# RUSTSEC-2020-0071 only in a comment\n[advisories]\nignore = []\n",
                TODAY
            ),
            Vec::<String>::new()
        );
    }

    #[test]
    fn keys_are_read_as_toml_reads_them() {
        let key = |text: &str| key_path(text).unwrap();
        assert_eq!(key("ignore"), Some(vec!["ignore".to_owned()]));
        assert_eq!(key(" \"ignore\" "), Some(vec!["ignore".to_owned()]));
        assert_eq!(
            key("advisories . 'ignore'"),
            Some(vec!["advisories".to_owned(), "ignore".to_owned()])
        );
        assert_eq!(key("{ crate"), None);
        assert_eq!(key("a b"), None);
        assert!(key_path("\"a\\tb\"").is_err());
        assert_eq!(split_key_value("a = \"x=y\""), Some(("a ", " \"x=y\"")));
        assert_eq!(split_key_value("\"a=b\","), None);
        assert_eq!(
            advisory_id("x = [\"RUSTSEC-2020-0071\"]"),
            Some("RUSTSEC-2020-0071")
        );
        assert_eq!(advisory_id("allow = [\"MIT\"]"), None);
    }

    #[test]
    fn the_checkout_ignores_no_advisory_without_its_issue_and_date() {
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let text = std::fs::read_to_string(checkout.join(DENY_TOML)).unwrap();
        assert_eq!(ignore_problems(&text, &today()), Vec::<String>::new());
    }

    #[test]
    fn the_hint_installs_the_pinned_version() {
        assert_eq!(
            install_hint(),
            format!("cargo install --locked cargo-deny@{VERSION}")
        );
    }
}
