// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `codeowners` step of `lint`: `.github/CODEOWNERS` names only maintainers, never an agent
//! identity, and covers every path class of RFC-0062 §12.3 and nothing else.
//!
//! The parser accepts the pattern subset the file uses, so the coverage check never evaluates
//! syntax it does not model:
//! - `/dir/` (a directory, anchored at the repository root) and `/path/file` (a file, or a
//!   directory and everything in it);
//! - `*` inside a segment, as in `/crates/*/src/keys.rs`;
//! - a bare name with no `/`, such as `AGENTS.md`, which matches at any depth.
//!
//! `**`, `?`, `[…]`, `!` and `\` escapes are refused. As on GitHub, the last matching rule wins,
//! and a rule with no owners leaves its paths unowned.

use std::path::Path;

use crate::roster::{Roster, is_login};
use crate::run::Context;

/// Where the file lives, relative to the workspace root.
pub const CODEOWNERS: &str = ".github/CODEOWNERS";
/// The maintainer roster, relative to the workspace root.
pub const ROSTER: &str = "governance/maintainers.toml";

/// One RFC-0062 §12.3 path class and paths that must fall in it.
pub struct PathClass {
    pub name: &'static str,
    pub paths: &'static [&'static str],
}

/// The fixture tree for the coverage check: for each RFC-0062 §12.3 class that is a path, sample
/// paths a maintainer must review. Three items on §12.3's list are not paths, so CODEOWNERS
/// cannot express them: a waived gate and a change no gate covers are enforced by labels
/// (QUAL-E1-T18), and a new crate dependency by the `dep-diff` check (QUAL-E1-T19).
///
/// Security-critical code the plan has not yet placed is owned by module name: `keys`,
/// `signing` and `updater` in any crate, and `admission` and `cluster_key` in `net` and
/// `cluster`. The tasks that write that code put it in modules with those names.
pub const CLASSES: &[PathClass] = &[
    PathClass {
        name: "security-critical: key handling and signing",
        paths: &[
            "crates/daemon/src/keys.rs",
            "crates/net/src/keys/store.rs",
            "crates/cli/src/signing.rs",
            "crates/core/src/signing/verify.rs",
        ],
    },
    PathClass {
        name: "security-critical: cluster admission and the cluster key in net and cluster",
        paths: &[
            "crates/net/src/admission.rs",
            "crates/net/src/admission/cert.rs",
            "crates/cluster/src/admission.rs",
            "crates/cluster/src/cluster_key.rs",
            "crates/net/src/cluster_key/rotate.rs",
        ],
    },
    PathClass {
        name: "security-critical: the update client",
        paths: &[
            "crates/cli/src/updater.rs",
            "crates/daemon/src/updater/tuf.rs",
        ],
    },
    PathClass {
        name: "security-critical: the hardware dispatcher and lab configuration",
        paths: &[
            "bench/lab/rigs.toml",
            "lab/agent/main.rs",
            "lab/dispatcher/config.toml",
        ],
    },
    PathClass {
        name: "CI and release configuration",
        paths: &[
            ".github/workflows/pr.yml",
            ".github/actions/setup-rust/action.yml",
            ".github/CODEOWNERS",
            "ci/jobs.toml",
            "xtask/src/main.rs",
            "xtask/src/registry.rs",
            ".cargo/config.toml",
            "release-please-config.json",
            "deny.toml",
            "supply-chain/audits.toml",
            "governance/delegations.toml",
            "governance/maintainers.toml",
            "version.txt",
            ".release-please-manifest.json",
        ],
    },
    PathClass {
        name: "registering a new model family in the graph registry",
        paths: &[
            "crates/graph/src/families.rs",
            "crates/graph/src/families/mod.rs",
        ],
    },
    PathClass {
        name: "agent instruction and configuration files",
        paths: &[
            "AGENTS.md",
            "crates/net/AGENTS.md",
            "ci/agent-instructions.toml",
            "ci/agent-tooling.lock",
        ],
    },
    PathClass {
        name: "a new vendored component in the licence manifest",
        paths: &["third_party/licences.toml"],
    },
];

/// Paths that merge on the gates and the review agent alone (RFC-0062 §12.3: "Kernels, the
/// solver and the rest of `net` and `cluster`"), so no rule may give them an owner.
pub const UNOWNED: &[&str] = &[
    "crates/kernels-cuda/src/lib.rs",
    "crates/solver/src/lib.rs",
    "crates/net/src/lib.rs",
    "crates/net/src/quic/stream.rs",
    "crates/cluster/src/apply.rs",
    "crates/graph/src/families/llama.rs",
    "crates/core/src/lib.rs",
    "crates/core/Cargo.toml",
    "Cargo.toml",
    "Cargo.lock",
    "docs/HOW_IT_WORKS.md",
    "plan/rfcs/RFC-0062-ci-cd-and-release-engineering.md",
    "README.md",
];

/// One parsed line: a pattern and its owners, with the line number for messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    pub line: usize,
    pattern: Pattern,
    pub owners: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Pattern {
    /// The pattern as written.
    text: String,
    /// Path segments, each a glob where `*` matches any run of characters.
    segments: Vec<String>,
    /// `/…`: matched from the root. Otherwise a single segment matched at any depth.
    anchored: bool,
    /// `…/`: matches only what lies inside the directory.
    directory: bool,
}

/// Parses CODEOWNERS text. Every problem is reported, one per line.
pub fn parse(text: &str) -> Result<Vec<Rule>, Vec<String>> {
    let mut rules = Vec::new();
    let mut problems = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let content = strip_comment(raw).trim();
        if content.is_empty() {
            continue;
        }
        let mut fields = content.split_whitespace();
        let pattern = fields.next().unwrap_or_default();
        match parse_pattern(pattern) {
            Ok(pattern) => rules.push(Rule {
                line,
                pattern,
                owners: fields.map(str::to_owned).collect(),
            }),
            Err(why) => problems.push(format!("CODEOWNERS line {line}: `{pattern}`: {why}")),
        }
    }
    if problems.is_empty() {
        Ok(rules)
    } else {
        Err(problems)
    }
}

/// The line before its comment. As on GitHub, `#` starts a comment only at the start of the line
/// or after whitespace; a `#` inside a token stays part of it, and the owner check refuses it.
fn strip_comment(line: &str) -> &str {
    let mut previous = None;
    for (at, c) in line.char_indices() {
        if c == '#' && previous.is_none_or(char::is_whitespace) {
            return &line[..at];
        }
        previous = Some(c);
    }
    line
}

fn parse_pattern(text: &str) -> Result<Pattern, String> {
    for (bad, what) in [
        ("**", "`**` is not supported"),
        ("?", "`?` is not supported"),
        ("[", "`[…]` is not supported"),
        ("]", "`[…]` is not supported"),
        ("\\", "escapes are not supported"),
    ] {
        if text.contains(bad) {
            return Err(what.into());
        }
    }
    if text.starts_with('!') {
        return Err("negation is not supported".into());
    }
    let anchored = text.starts_with('/');
    let directory = text.ends_with('/');
    let body = text.trim_start_matches('/').trim_end_matches('/');
    if body.is_empty() {
        return Err("the pattern names no path".into());
    }
    if !anchored && body.contains('/') {
        return Err("a pattern with a `/` inside must start with `/`".into());
    }
    let segments: Vec<String> = body.split('/').map(str::to_owned).collect();
    if segments.iter().any(String::is_empty) {
        return Err("empty path segment".into());
    }
    Ok(Pattern {
        text: text.to_owned(),
        segments,
        anchored,
        directory,
    })
}

impl Pattern {
    /// Whether the pattern covers `path`, a file path relative to the root with `/` separators.
    fn matches(&self, path: &str) -> bool {
        let parts: Vec<&str> = path.split('/').collect();
        if self.anchored {
            self.matches_at(&parts, 0)
        } else {
            (0..parts.len()).any(|start| self.matches_at(&parts, start))
        }
    }

    /// The pattern's segments match `parts[start..]`'s first segments. A directory pattern needs
    /// at least one part after them; a file pattern matches the path itself or what lies inside.
    fn matches_at(&self, parts: &[&str], start: usize) -> bool {
        let end = start + self.segments.len();
        if end > parts.len() || (self.directory && end == parts.len()) {
            return false;
        }
        // As on GitHub, `/docs/*` covers the files directly in `docs/`, not deeper ones: a file
        // pattern whose last segment has a `*` matches whole paths only. A bare `*` matches every
        // path.
        let glob_last = self.segments.last().is_some_and(|s| s.contains('*'));
        let bare_star = !self.anchored && self.segments == ["*"];
        if glob_last && !bare_star && !self.directory && end != parts.len() {
            return false;
        }
        self.segments
            .iter()
            .zip(&parts[start..end])
            .all(|(glob, part)| glob_matches(glob, part))
    }
}

/// `*` matches any run of characters within one segment; everything else matches itself.
fn glob_matches(glob: &str, text: &str) -> bool {
    let pieces: Vec<&str> = glob.split('*').collect();
    if pieces.len() == 1 {
        return glob == text;
    }
    let (first, last) = (pieces[0], pieces[pieces.len() - 1]);
    if !text.starts_with(first) || !text[first.len()..].ends_with(last) {
        return false;
    }
    // `first` and `last` are in place; the middle pieces must appear, in order, between them.
    let mut rest = &text[first.len()..text.len() - last.len()];
    for piece in &pieces[1..pieces.len() - 1] {
        match rest.find(piece) {
            Some(at) => rest = &rest[at + piece.len()..],
            None => return false,
        }
    }
    true
}

/// The rule GitHub applies to `path`: the last one that matches.
pub fn owning_rule<'a>(rules: &'a [Rule], path: &str) -> Option<&'a Rule> {
    rules.iter().rev().find(|rule| rule.pattern.matches(path))
}

/// Every problem with `rules` against `roster`: an owner who is not a maintainer or who is an
/// agent identity, a class path no maintainer owns, and an owned path that must be unowned.
pub fn check(
    rules: &[Rule],
    roster: &Roster,
    classes: &[PathClass],
    unowned: &[&str],
) -> Vec<String> {
    let mut problems = Vec::new();
    for rule in rules {
        for owner in &rule.owners {
            let at = format!("CODEOWNERS line {}: `{owner}`", rule.line);
            let Some(login) = owner.strip_prefix('@') else {
                problems.push(format!(
                    "{at} is not a GitHub account; name maintainers by `@login`"
                ));
                continue;
            };
            if login.contains('/') {
                problems.push(format!(
                    "{at} is a team; name maintainers individually, so the check can verify them"
                ));
            } else if !is_login(login) {
                problems.push(format!("{at} is not a well-formed GitHub login"));
            } else if roster.is_agent(login) {
                problems.push(format!(
                    "{at} is an agent identity; agents are never code owners (RFC-0062 §12.3)"
                ));
            } else if !roster.is_maintainer(login) {
                problems.push(format!("{at} is not a maintainer in {ROSTER}"));
            }
        }
    }
    for class in classes {
        for path in class.paths {
            match owning_rule(rules, path) {
                None => problems.push(format!(
                    "no CODEOWNERS rule covers `{path}` ({})",
                    class.name
                )),
                Some(rule) if !rule.owners.iter().any(|o| is_maintainer_owner(o, roster)) => {
                    problems.push(format!(
                        "`{path}` ({}) falls to CODEOWNERS line {} (`{}`), which names no maintainer",
                        class.name, rule.line, rule.pattern.text
                    ))
                }
                Some(_) => {}
            }
        }
    }
    for path in unowned {
        if let Some(rule) = owning_rule(rules, path)
            && !rule.owners.is_empty()
        {
            problems.push(format!(
                "`{path}` must merge without a code owner (RFC-0062 §12.3), but CODEOWNERS line {} (`{}`) owns it",
                rule.line, rule.pattern.text
            ));
        }
    }
    problems
}

fn is_maintainer_owner(owner: &str, roster: &Roster) -> bool {
    owner
        .strip_prefix('@')
        .is_some_and(|login| roster.is_maintainer(login))
}

/// The `lint` step: checks the workspace's CODEOWNERS against its roster.
pub fn step(context: &Context) -> Result<(), String> {
    let codeowners = read(&context.root, CODEOWNERS)?;
    let roster = Roster::parse(&read(&context.root, ROSTER)?)
        .map_err(|error| format!("{ROSTER}: {error}"))?;
    let rules = parse(&codeowners).map_err(|problems| problems.join("\n"))?;
    let problems = check(&rules, &roster, CLASSES, UNOWNED);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

fn read(root: &Path, relative: &str) -> Result<String, String> {
    std::fs::read_to_string(root.join(relative)).map_err(|error| format!("{relative}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roster() -> Roster {
        Roster::parse("maintainers = [\"alice\", \"bob\"]\nagents = [\"project-review-agent\"]")
            .unwrap()
    }

    fn rules(text: &str) -> Vec<Rule> {
        parse(text).unwrap()
    }

    /// The repository's own CODEOWNERS and roster.
    fn real() -> (Vec<Rule>, Roster) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let rules = parse(&read(&root, CODEOWNERS).unwrap()).unwrap();
        let roster = Roster::parse(&read(&root, ROSTER).unwrap()).unwrap();
        (rules, roster)
    }

    const ONE_CLASS: &[PathClass] = &[PathClass {
        name: "CI",
        paths: &[".github/workflows/pr.yml", "ci/jobs.toml"],
    }];

    #[test]
    fn the_repository_codeowners_passes() {
        let (rules, roster) = real();
        assert_eq!(
            check(&rules, &roster, CLASSES, UNOWNED),
            Vec::<String>::new()
        );
    }

    #[test]
    fn every_class_has_sample_paths() {
        for class in CLASSES {
            assert!(!class.paths.is_empty(), "{}", class.name);
        }
    }

    #[test]
    fn an_agent_identity_as_owner_fails() {
        for agent in [
            "@project-review-agent",
            "@renovate[bot]",
            "@PROJECT-REVIEW-AGENT",
        ] {
            let text = format!("/.github/ @alice {agent}\n/ci/ @alice\n");
            let problems = check(&rules(&text), &roster(), ONE_CLASS, &[]);
            assert_eq!(problems.len(), 1, "{problems:?}");
            assert!(problems[0].contains("is an agent identity"), "{problems:?}");
            assert!(
                problems[0].starts_with("CODEOWNERS line 1:"),
                "{problems:?}"
            );
        }
    }

    #[test]
    fn owners_must_be_maintainers_named_individually() {
        let text = "/.github/ @alice @carol @org/maintainers someone@example.org\n/ci/ @bob\n";
        let problems = check(&rules(text), &roster(), ONE_CLASS, &[]);
        assert_eq!(problems.len(), 3, "{problems:?}");
        assert!(problems[0].contains("`@carol` is not a maintainer"));
        assert!(problems[1].contains("is a team"));
        assert!(problems[2].contains("is not a GitHub account"));
    }

    #[test]
    fn a_class_path_without_a_rule_fails_and_names_its_class() {
        let problems = check(&rules("/.github/ @alice\n"), &roster(), ONE_CLASS, &[]);
        assert_eq!(problems, ["no CODEOWNERS rule covers `ci/jobs.toml` (CI)"]);
    }

    #[test]
    fn a_later_rule_without_owners_uncovers_a_path() {
        let text = "/.github/ @alice\n/ci/ @alice\n/ci/jobs.toml\n";
        let problems = check(&rules(text), &roster(), ONE_CLASS, &[]);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].contains("falls to CODEOWNERS line 3"),
            "{problems:?}"
        );
    }

    #[test]
    fn a_class_owned_only_by_non_maintainers_fails() {
        let text = "/.github/ @alice\n/ci/ @carol\n";
        let problems = check(&rules(text), &roster(), ONE_CLASS, &[]);
        assert!(
            problems.iter().any(|p| p.contains("names no maintainer")),
            "{problems:?}"
        );
    }

    #[test]
    fn an_owner_on_a_path_that_must_be_unowned_fails() {
        let text = "* @alice\n";
        let problems = check(&rules(text), &roster(), &[], &["crates/solver/src/lib.rs"]);
        assert_eq!(problems.len(), 1, "{problems:?}");
        assert!(
            problems[0].contains("must merge without a code owner"),
            "{problems:?}"
        );
    }

    #[test]
    fn every_class_fails_when_its_rules_are_removed() {
        // The coverage check over the fixture tree: drop the rules for one class at a time
        // from the real file, and that class's paths are reported.
        let (rules, roster) = real();
        for class in CLASSES {
            let kept: Vec<Rule> = rules
                .iter()
                .filter(|rule| !class.paths.iter().any(|p| rule.pattern.matches(p)))
                .cloned()
                .collect();
            let problems = check(&kept, &roster, CLASSES, UNOWNED);
            for path in class.paths {
                assert!(
                    problems.iter().any(|p| p.contains(&format!("`{path}`"))),
                    "removing {}'s rules did not report {path}: {problems:?}",
                    class.name
                );
            }
        }
    }

    #[test]
    fn patterns_match_like_github() {
        let cases: &[(&str, &str, bool)] = &[
            ("/.github/", ".github/workflows/pr.yml", true),
            ("/.github/", ".github", false),
            ("/.github/", "x/.github/a", false),
            ("/deny.toml", "deny.toml", true),
            ("/deny.toml", "crates/x/deny.toml", false),
            ("/lab/", "lab/agent/main.rs", true),
            ("/lab/", "bench/lab/rigs.toml", false),
            ("/bench/lab/", "bench/lab/rigs.toml", true),
            ("/crates/*/src/keys.rs", "crates/daemon/src/keys.rs", true),
            (
                "/crates/*/src/keys.rs",
                "crates/daemon/src/keys_cache.rs",
                false,
            ),
            ("/crates/*/src/keys/", "crates/net/src/keys/store.rs", true),
            ("/crates/*/src/keys/", "crates/net/src/keys.rs", false),
            ("/crates/*/src/keys/", "crates/net/src/a/keys/x.rs", false),
            (
                "/crates/graph/src/families/mod.rs",
                "crates/graph/src/families/llama.rs",
                false,
            ),
            ("AGENTS.md", "AGENTS.md", true),
            ("AGENTS.md", "crates/net/AGENTS.md", true),
            ("AGENTS.md", "docs/AGENTS.md.bak", false),
            ("*", "anything/at/all.rs", true),
            ("*.toml", "crates/core/Cargo.toml", true),
            ("/.github/*", ".github/CODEOWNERS", true),
            ("/.github/*", ".github/workflows/pr.yml", false),
            ("/crates/*", "crates/core/src/lib.rs", false),
            ("/crates/*/src/keys.rs", "crates/core/src/keys.rs", true),
            ("/ci", "ci/jobs.toml", true),
            ("/ci", "cid/jobs.toml", false),
        ];
        for (pattern, path, wanted) in cases {
            let parsed = parse_pattern(pattern).unwrap();
            assert_eq!(parsed.matches(path), *wanted, "`{pattern}` on `{path}`");
        }
    }

    #[test]
    fn a_hash_inside_a_token_is_not_a_comment() {
        // GitHub reads `@alice#x` as one, invalid, owner; so does the check.
        let rules = rules("/ci/ @alice#x\n/.github/ @alice # a comment\n#whole line\n");
        assert_eq!(rules[0].owners, ["@alice#x"]);
        assert_eq!(rules[1].owners, ["@alice"]);
        let problems = check(&rules, &roster(), ONE_CLASS, &[]);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("`@alice#x` is not a well-formed GitHub login")),
            "{problems:?}"
        );
        assert!(
            problems.iter().any(|p| p.contains("names no maintainer")),
            "{problems:?}"
        );
    }

    #[test]
    fn the_last_matching_rule_wins() {
        let rules = rules("/ci/ @alice\n/ci/jobs.toml @bob\n");
        assert_eq!(
            owning_rule(&rules, "ci/jobs.toml").unwrap().owners,
            ["@bob"]
        );
        assert_eq!(
            owning_rule(&rules, "ci/cost.toml").unwrap().owners,
            ["@alice"]
        );
        assert!(owning_rule(&rules, "README.md").is_none());
    }

    #[test]
    fn unsupported_syntax_is_refused_with_its_line() {
        let text = "# comment\n\n/ok/ @alice\n/a/**/b @alice\n!x @alice\n/a/?.rs @alice\n\
                    a/b @alice\n/[ab]/ @alice\n/a\\ b @alice\n/ @alice\n";
        let problems = parse(text).unwrap_err();
        let lines: Vec<&str> = problems
            .iter()
            .map(|p| p.split(':').next().unwrap())
            .collect();
        assert_eq!(
            lines,
            [
                "CODEOWNERS line 4",
                "CODEOWNERS line 5",
                "CODEOWNERS line 6",
                "CODEOWNERS line 7",
                "CODEOWNERS line 8",
                "CODEOWNERS line 9",
                "CODEOWNERS line 10",
            ]
        );
    }

    #[test]
    fn globs_match_within_a_segment() {
        for (glob, text, wanted) in [
            ("*", "", true),
            ("*", "abc", true),
            ("a*", "abc", true),
            ("*c", "abc", true),
            ("a*c", "ac", true),
            ("a*c", "abc", true),
            ("a*b*c", "aXbYc", true),
            ("a*b*c", "acb", false),
            ("ab*ba", "aba", false),
            ("abc", "abc", true),
            ("abc", "abd", false),
        ] {
            assert_eq!(glob_matches(glob, text), wanted, "`{glob}` on `{text}`");
        }
    }
}
