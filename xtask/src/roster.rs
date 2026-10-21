// SPDX-License-Identifier: MIT OR Apache-2.0

//! The maintainer roster, `governance/maintainers.toml`: who counts as a maintainer, and which
//! accounts agents act through (RFC-0062 §8.3, §12.3).
//!
//! The file holds two arrays of GitHub logins:
//!
//! ```toml
//! maintainers = ["alice", "bob"]
//! agents = ["project-review-agent"]
//! ```
//!
//! The parser accepts exactly that TOML subset: `#` comments, and the two keys, each an array of
//! double-quoted strings on one or more lines. Anything else is an error, so the checks never act
//! on a file they only half read. QUAL-E1-T17's delegation registry can take over this file.

use std::fmt;

/// Who is a maintainer and who is an agent. Logins compare without regard to ASCII case, as
/// GitHub's do.
#[derive(Debug, PartialEq, Eq)]
pub struct Roster {
    pub maintainers: Vec<String>,
    pub agents: Vec<String>,
}

/// Why a roster file was refused.
#[derive(Debug, PartialEq, Eq)]
pub struct RosterError(pub String);

impl fmt::Display for RosterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Roster {
    /// Parses and validates the text of a roster file.
    pub fn parse(text: &str) -> Result<Roster, RosterError> {
        let mut maintainers = None;
        let mut agents = None;
        let mut lines = text.lines().enumerate();
        while let Some((index, line)) = lines.next() {
            let line = strip_comment(line).trim();
            if line.is_empty() {
                continue;
            }
            let at = |message: &str| RosterError(format!("line {}: {message}", index + 1));
            let Some((key, value)) = line.split_once('=') else {
                return Err(at("expected `key = [ ... ]`"));
            };
            let slot = match key.trim() {
                "maintainers" => &mut maintainers,
                "agents" => &mut agents,
                other => return Err(at(&format!("unknown key `{other}`"))),
            };
            if slot.is_some() {
                return Err(at(&format!("`{}` is given twice", key.trim())));
            }
            // Gather the array, which may continue over the following lines.
            let mut array = value.trim().to_owned();
            while !array.contains(']') {
                let Some((_, next)) = lines.next() else {
                    return Err(at("the array is not closed with `]`"));
                };
                array.push(' ');
                array.push_str(strip_comment(next).trim());
            }
            *slot = Some(parse_array(&array).map_err(|message| at(&message))?);
        }
        let roster = Roster {
            maintainers: maintainers
                .ok_or_else(|| RosterError("`maintainers` is missing".into()))?,
            agents: agents.ok_or_else(|| RosterError("`agents` is missing".into()))?,
        };
        roster.validate()?;
        Ok(roster)
    }

    fn validate(&self) -> Result<(), RosterError> {
        if self.maintainers.is_empty() {
            return Err(RosterError("`maintainers` is empty".into()));
        }
        for (list, logins) in [("maintainers", &self.maintainers), ("agents", &self.agents)] {
            for (i, login) in logins.iter().enumerate() {
                if !is_login(login) {
                    return Err(RosterError(format!(
                        "`{login}` in `{list}` is not a GitHub login"
                    )));
                }
                if logins[..i].iter().any(|l| l.eq_ignore_ascii_case(login)) {
                    return Err(RosterError(format!(
                        "`{login}` is listed twice in `{list}`"
                    )));
                }
            }
        }
        for login in &self.maintainers {
            if self.is_agent(login) {
                return Err(RosterError(format!(
                    "`{login}` is listed as a maintainer but is an agent identity"
                )));
            }
        }
        Ok(())
    }

    /// An agent identity: a GitHub App account (`…[bot]`) or a login in `agents`.
    pub fn is_agent(&self, login: &str) -> bool {
        is_app_account(login) || self.agents.iter().any(|a| a.eq_ignore_ascii_case(login))
    }

    /// A maintainer: a login in `maintainers` that is not an agent identity.
    pub fn is_maintainer(&self, login: &str) -> bool {
        !self.is_agent(login)
            && self
                .maintainers
                .iter()
                .any(|m| m.eq_ignore_ascii_case(login))
    }
}

/// GitHub App accounts end in `[bot]`.
pub fn is_app_account(login: &str) -> bool {
    login.to_ascii_lowercase().ends_with("[bot]")
}

/// A GitHub login: letters, digits and single hyphens, optionally an App's `[bot]` suffix.
pub fn is_login(login: &str) -> bool {
    let name = login
        .strip_suffix("[bot]")
        .or_else(|| login.strip_suffix("[BOT]"))
        .unwrap_or(login);
    !name.is_empty()
        && name.len() <= 39
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--")
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

/// The line without a `#` comment. Strings in the roster never contain `#`.
fn strip_comment(line: &str) -> &str {
    line.split_once('#').map_or(line, |(before, _)| before)
}

/// `["a", "b"]` → `["a", "b"]`. A trailing comma is allowed; escapes are not.
fn parse_array(text: &str) -> Result<Vec<String>, String> {
    let inner = text
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or_else(|| format!("expected an array, got `{text}`"))?;
    let mut items = Vec::new();
    for item in inner.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let value = item
            .strip_prefix('"')
            .and_then(|rest| rest.strip_suffix('"'))
            .filter(|value| !value.contains('"') && !value.contains('\\'))
            .ok_or_else(|| format!("expected a plain double-quoted string, got `{item}`"))?;
        items.push(value.to_owned());
    }
    // `["a" "b"]` has no comma, so it arrives as one item and fails above; a stray `,,` gives an
    // empty item, which is skipped.
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "\
# Maintainers.
maintainers = [\"alice\", \"Bob\"]  # trailing comment
agents = [
    \"project-review-agent\",   # the review agent
]
";

    fn roster() -> Roster {
        Roster::parse(FIXTURE).unwrap()
    }

    #[test]
    fn parses_both_lists() {
        assert_eq!(
            roster(),
            Roster {
                maintainers: vec!["alice".into(), "Bob".into()],
                agents: vec!["project-review-agent".into()],
            }
        );
    }

    #[test]
    fn maintainers_and_agents_compare_without_case() {
        let roster = roster();
        assert!(roster.is_maintainer("alice"));
        assert!(roster.is_maintainer("ALICE"));
        assert!(roster.is_maintainer("bob"));
        assert!(!roster.is_maintainer("carol"));
        assert!(roster.is_agent("Project-Review-Agent"));
        assert!(!roster.is_maintainer("project-review-agent"));
    }

    #[test]
    fn app_accounts_are_agents_without_being_listed() {
        let roster = roster();
        assert!(roster.is_agent("renovate[bot]"));
        assert!(roster.is_agent("some-app[BOT]"));
        assert!(!roster.is_maintainer("alice[bot]"));
    }

    #[test]
    fn the_real_roster_parses() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../governance/maintainers.toml");
        let text = std::fs::read_to_string(path).unwrap();
        let roster = Roster::parse(&text).unwrap();
        assert!(!roster.maintainers.is_empty());
    }

    #[test]
    fn malformed_rosters_are_refused() {
        for (text, wanted) in [
            ("agents = []", "`maintainers` is missing"),
            ("maintainers = [\"a\"]", "`agents` is missing"),
            ("maintainers = []\nagents = []", "`maintainers` is empty"),
            (
                "maintainers = [\"a\"]\nagents = []\nowners = []",
                "unknown key `owners`",
            ),
            (
                "maintainers = [\"a\"]\nmaintainers = [\"b\"]",
                "given twice",
            ),
            (
                "maintainers = [\"a\"\nagents = []",
                "expected a plain double-quoted string",
            ),
            ("maintainers = [\"a\"", "not closed"),
            ("maintainers = \"a\"\nagents = []", "expected an array"),
            (
                "maintainers = [a]\nagents = []",
                "plain double-quoted string",
            ),
            (
                "maintainers = [\"a\" \"b\"]\nagents = []",
                "plain double-quoted string",
            ),
            (
                "maintainers = [\"a\\\"b\"]\nagents = []",
                "plain double-quoted string",
            ),
            ("maintainers = [\"a b\"]\nagents = []", "not a GitHub login"),
            ("maintainers = [\"a\", \"A\"]\nagents = []", "listed twice"),
            (
                "maintainers = [\"x\"]\nagents = [\"x\"]",
                "is an agent identity",
            ),
            (
                "maintainers = [\"x[bot]\"]\nagents = []",
                "is an agent identity",
            ),
            ("maintainers\nagents = []", "expected `key = [ ... ]`"),
        ] {
            let error = Roster::parse(text).unwrap_err().to_string();
            assert!(error.contains(wanted), "{text:?}: {error}");
        }
    }

    #[test]
    fn logins_follow_github_rules() {
        for good in ["a", "a-b", "A1", "renovate[bot]"] {
            assert!(is_login(good), "{good}");
        }
        for bad in ["", "-a", "a-", "a--b", "a_b", "a.b", &"a".repeat(40)] {
            assert!(!is_login(bad), "{bad}");
        }
    }
}
