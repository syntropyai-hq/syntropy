// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci repo-settings`: checks the repository's settings and branch rulesets against
//! the state declared in `ci/repo-settings/` (QUAL-E1-T06).
//!
//! Each declared file is the body of the REST request that `ci/repo-settings/apply.sh` sends, so
//! the script and the check read one description of the state:
//!
//! | File | Read back from `/repos/{repo}…` | Required by |
//! |---|---|---|
//! | `repository.json` | (the repository) | squash merges titled by the PR, RFC-0062 §8.3 and §12.3; secret scanning with push protection, ADR-0620 decision 4 |
//! | `private-vulnerability-reporting.json` | `/private-vulnerability-reporting` | RFC-0062 §13; ADR-0620 decision 4 |
//! | `actions-permissions.json` | `/actions/permissions` | full-length SHA pins, ADR-0620 decision 4 |
//! | `actions-selected-actions.json` | `/actions/permissions/selected-actions` | the actions allowed and blocked, ADR-0620 decision 4 |
//! | `actions-workflow-permissions.json` | `/actions/permissions/workflow` | a read-only default token that cannot approve PRs, for a workflow with no `permissions:` block, RFC-0062 §12.4 |
//! | `fork-pr-contributor-approval.json` | `/actions/permissions/fork-pr-contributor-approval` | approval for all outside contributors, RFC-0062 §3.3 item 1 |
//! | `ruleset-*.json` | `/rules/branches/{branch}` | RFC-0062 §2.5, §12.3 and §12.6; ADR-0621 decision 2 |
//!
//! A ruleset file's `rules` must be in force on each branch its `conditions.ref_name.include`
//! names. `/rules/branches/{branch}` lists the rules of every active ruleset that applies to the
//! branch, the organisation's included. A `*` in a pattern is read as `probe`, so
//! `refs/heads/release/*` is checked on `release/probe`, which need not exist. Bypass actors are
//! not shown there; the organisation audit checks them (QUAL-E1-T23).
//!
//! The live state passes when it contains the declared state:
//! - an object, when each declared member is present and passes;
//! - a list of objects, such as rules or required checks, when each declared item matches a live
//!   item, because an extra rule or check only adds protection;
//! - a list of strings or numbers, such as allowed action patterns or merge methods, when it holds
//!   exactly the declared items in any order, because an extra item could loosen it;
//! - anything else, when it is equal.
//!
//! Options:
//! - `--repo <owner/name>`: the repository, else `$GITHUB_REPOSITORY`, else the one `gh repo view`
//!   names for this checkout;
//! - `--from <dir>`: read recorded responses from `<dir>` instead of the API, one file per
//!   endpoint, named as in [`SETTINGS`] and `rules-<branch>.json` with `/` as `-`;
//! - `--save <dir>`: also write each response there under those names, to record a fixture.
//!
//! The API calls go through `gh api`. Reading `security_and_analysis`, private vulnerability
//! reporting and the Actions policy needs a token with administration read access, such as an
//! organisation owner's; with less, those settings read as missing and the check fails.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use crate::run::Context;

/// The directory of declared state, under the workspace root.
pub const DECLARED: &str = "ci/repo-settings";

/// The branch a `*` in a ruleset's branch pattern is checked on.
pub const PROBE: &str = "probe";

/// A setting read from one endpoint.
#[derive(Debug, PartialEq, Eq)]
pub struct Setting {
    /// The declared file under [`DECLARED`], and the response file name under `--from`.
    pub file: &'static str,
    /// The endpoint under `/repos/{repo}`; empty for the repository itself.
    pub endpoint: &'static str,
    /// The plan text that requires it.
    pub required_by: &'static str,
}

/// The settings read from one endpoint each, in the order `apply.sh` sets them.
pub const SETTINGS: &[Setting] = &[
    Setting {
        file: "repository.json",
        endpoint: "",
        required_by: "RFC-0062 §8.3, §12.3; ADR-0620 decision 4",
    },
    Setting {
        file: "private-vulnerability-reporting.json",
        endpoint: "/private-vulnerability-reporting",
        required_by: "RFC-0062 §13; ADR-0620 decision 4",
    },
    Setting {
        file: "actions-permissions.json",
        endpoint: "/actions/permissions",
        required_by: "ADR-0620 decision 4",
    },
    Setting {
        file: "actions-selected-actions.json",
        endpoint: "/actions/permissions/selected-actions",
        required_by: "ADR-0620 decision 4",
    },
    Setting {
        file: "actions-workflow-permissions.json",
        endpoint: "/actions/permissions/workflow",
        required_by: "RFC-0062 §12.4",
    },
    Setting {
        file: "fork-pr-contributor-approval.json",
        endpoint: "/actions/permissions/fork-pr-contributor-approval",
        required_by: "RFC-0062 §3.3 item 1; ADR-0620 decision 4",
    },
];

/// What the rulesets are required by.
const RULESETS_REQUIRED_BY: &str = "RFC-0062 §2.5, §12.3, §12.6; ADR-0621 decision 2";

/// Where the live state comes from.
pub trait Responses {
    /// The body of `GET /repos/{repo}<endpoint>`, all pages of it when `paginate`, saved or read
    /// under `file`.
    fn get(&mut self, endpoint: &str, file: &str, paginate: bool) -> Result<String, String>;
}

/// Recorded responses in a directory.
pub struct Recorded(pub PathBuf);

impl Responses for Recorded {
    fn get(&mut self, _endpoint: &str, file: &str, _paginate: bool) -> Result<String, String> {
        let path = self.0.join(file);
        std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// The REST API through `gh api`, saving each response under `save` when it is set.
pub struct Gh {
    pub repo: String,
    pub save: Option<PathBuf>,
}

impl Responses for Gh {
    fn get(&mut self, endpoint: &str, file: &str, paginate: bool) -> Result<String, String> {
        let path = format!("repos/{}{endpoint}", self.repo);
        let mut command = Command::new("gh");
        command.args([
            "api",
            "--method",
            "GET",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "X-GitHub-Api-Version: 2022-11-28",
        ]);
        if paginate {
            command.arg("--paginate");
        }
        command.arg(&path);
        let output = command
            .output()
            .map_err(|e| format!("GET /{path}: could not start `gh`: {e}"))?;
        let body = String::from_utf8_lossy(&output.stdout).into_owned();
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let reason = stderr.lines().next().unwrap_or("").trim();
            return Err(format!("GET /{path}: `gh api` failed: {reason}"));
        }
        if let Some(dir) = &self.save {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            let saved = dir.join(file);
            std::fs::write(&saved, &body).map_err(|e| format!("{}: {e}", saved.display()))?;
        }
        Ok(body)
    }
}

/// The response file for the rules on `branch`.
pub fn rules_file(branch: &str) -> String {
    format!("rules-{}.json", branch.replace('/', "-"))
}

/// The branch a ruleset's `refs/heads/…` pattern is checked on, or why it cannot be.
fn probe_branch(pattern: &str) -> Result<String, String> {
    let Some(branch) = pattern.strip_prefix("refs/heads/") else {
        return Err(format!(
            "the pattern `{pattern}` is not a `refs/heads/` branch pattern this check can read"
        ));
    };
    if branch.is_empty() || branch.contains(['?', '[', ']', '~']) {
        return Err(format!(
            "the pattern `{pattern}` is not a branch name with at most `*` wildcards"
        ));
    }
    Ok(branch.replace("**", PROBE).replace('*', PROBE))
}

/// Checks every declared file in `declared` against `responses`. Returns one line per setting
/// that passed, and every problem found.
pub fn check(declared: &Path, responses: &mut dyn Responses) -> (Vec<String>, Vec<String>) {
    let mut passed = Vec::new();
    let mut problems = Vec::new();

    for setting in SETTINGS {
        let label = format!("`{}` ({})", setting.file, setting.required_by);
        let want = match read_json(&declared.join(setting.file)) {
            Ok(want) => want,
            Err(error) => {
                problems.push(format!("{label}: {error}"));
                continue;
            }
        };
        let live = match responses.get(setting.endpoint, setting.file, false) {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(live) => live,
                Err(error) => {
                    problems.push(format!("{label}: the response is not JSON: {error}"));
                    continue;
                }
            },
            Err(error) => {
                problems.push(format!("{label}: {error}"));
                continue;
            }
        };
        let mut found = Vec::new();
        compare("", &want, Some(&live), &mut found);
        if found.is_empty() {
            passed.push(format!("{}: matches", setting.file));
        }
        problems.extend(found.into_iter().map(|p| format!("{label}: {p}")));
    }

    let rulesets = match ruleset_files(declared) {
        Ok(rulesets) => rulesets,
        Err(error) => {
            problems.push(error);
            Vec::new()
        }
    };
    if rulesets.is_empty() {
        problems.push(format!(
            "`{DECLARED}` declares no `ruleset-*.json`, and RFC-0062 §12.6 requires rulesets on \
             `main` and `release/*`"
        ));
    }
    // Each branch's rules are read once, however many rulesets name it.
    let mut live_rules: BTreeMap<String, Result<Vec<Value>, String>> = BTreeMap::new();
    for file in rulesets {
        let label = format!("`{file}` ({RULESETS_REQUIRED_BY})");
        let ruleset = match read_json(&declared.join(&file)) {
            Ok(ruleset) => ruleset,
            Err(error) => {
                problems.push(format!("{label}: {error}"));
                continue;
            }
        };
        let (patterns, rules) = match ruleset_parts(&ruleset) {
            Ok(parts) => parts,
            Err(error) => {
                problems.push(format!("{label}: {error}"));
                continue;
            }
        };
        let mut found = Vec::new();
        for pattern in patterns {
            let branch = match probe_branch(pattern) {
                Ok(branch) => branch,
                Err(error) => {
                    found.push(error);
                    continue;
                }
            };
            let live = live_rules.entry(branch.clone()).or_insert_with(|| {
                responses
                    .get(
                        &format!("/rules/branches/{branch}"),
                        &rules_file(&branch),
                        true,
                    )
                    .and_then(|text| json_pages(&text))
            });
            match live {
                Ok(live) => {
                    let live = Value::Array(live.clone());
                    compare(
                        &format!("rules on `{branch}`"),
                        &Value::Array(rules.to_vec()),
                        Some(&live),
                        &mut found,
                    );
                }
                Err(error) => found.push(error.clone()),
            }
        }
        if found.is_empty() {
            passed.push(format!("{file}: its rules are in force"));
        }
        problems.extend(found.into_iter().map(|p| format!("{label}: {p}")));
    }
    (passed, problems)
}

/// The declared `ruleset-*.json` files, sorted.
fn ruleset_files(declared: &Path) -> Result<Vec<String>, String> {
    let entries =
        std::fs::read_dir(declared).map_err(|e| format!("{}: {e}", declared.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", declared.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("ruleset-") && name.ends_with(".json") {
            files.push(name);
        }
    }
    files.sort();
    Ok(files)
}

/// A ruleset's branch patterns and rules, or why it is not one this check can read.
fn ruleset_parts(ruleset: &Value) -> Result<(Vec<&str>, &[Value]), String> {
    if ruleset.get("target").and_then(Value::as_str) != Some("branch") {
        return Err("its `target` is not `branch`".to_owned());
    }
    if ruleset.get("enforcement").and_then(Value::as_str) != Some("active") {
        return Err("its `enforcement` is not `active`, so it would enforce nothing".to_owned());
    }
    let patterns: Vec<&str> = ruleset
        .pointer("/conditions/ref_name/include")
        .and_then(Value::as_array)
        .map(|include| include.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if patterns.is_empty() {
        return Err("it names no branch in `conditions.ref_name.include`".to_owned());
    }
    let rules = ruleset
        .get("rules")
        .and_then(Value::as_array)
        .filter(|rules| !rules.is_empty())
        .ok_or_else(|| "it has no `rules`".to_owned())?;
    Ok((patterns, rules))
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: not JSON: {e}", path.display()))
}

/// The items of one or more JSON arrays, one after another, as `gh api --paginate` prints pages.
fn json_pages(text: &str) -> Result<Vec<Value>, String> {
    let mut items = Vec::new();
    for page in serde_json::Deserializer::from_str(text).into_iter::<Value>() {
        match page {
            Ok(Value::Array(page)) => items.extend(page),
            Ok(_) => return Err("the response is not a JSON array".to_owned()),
            Err(error) => return Err(format!("the response is not JSON: {error}")),
        }
    }
    Ok(items)
}

/// `name` appended to the path `path`, for messages.
fn member(path: &str, name: &str) -> String {
    if path.is_empty() {
        format!("`{name}`")
    } else {
        format!("{path}.`{name}`")
    }
}

/// How a value reads in a message.
fn shown(value: &Value) -> String {
    format!("`{value}`")
}

fn is_scalar(value: &Value) -> bool {
    !matches!(value, Value::Array(_) | Value::Object(_))
}

/// A key that names a list item, such as a rule's `type` or a required check's `context`.
fn identity(item: &Value) -> Option<(&'static str, &str)> {
    ["type", "context"].into_iter().find_map(|key| {
        item.get(key)
            .and_then(Value::as_str)
            .map(|value| (key, value))
    })
}

/// Adds to `problems` every way `live` does not contain `want`. `path` names the value.
fn compare(path: &str, want: &Value, live: Option<&Value>, problems: &mut Vec<String>) {
    let shown_path = if path.is_empty() {
        "the response".to_owned()
    } else {
        path.to_owned()
    };
    let Some(live) = live.filter(|live| !live.is_null() || want.is_null()) else {
        problems.push(format!("{shown_path} is missing; want {}", shown(want)));
        return;
    };
    match want {
        Value::Object(want) => {
            let Some(live) = live.as_object() else {
                problems.push(format!("{shown_path} is {}, not an object", shown(live)));
                return;
            };
            for (name, want) in want {
                compare(&member(path, name), want, live.get(name), problems);
            }
        }
        Value::Array(want) => {
            let Some(live) = live.as_array() else {
                problems.push(format!("{shown_path} is {}, not a list", shown(live)));
                return;
            };
            if want.iter().all(is_scalar) {
                for item in want {
                    if !live.contains(item) {
                        problems.push(format!("{shown_path} lacks {}", shown(item)));
                    }
                }
                for item in live {
                    if !want.contains(item) {
                        problems.push(format!(
                            "{shown_path} has {}, which is not declared",
                            shown(item)
                        ));
                    }
                }
            } else {
                for item in want {
                    compare_item(&shown_path, item, live, problems);
                }
            }
        }
        _ => {
            if live != want {
                problems.push(format!(
                    "{shown_path} is {}; want {}",
                    shown(live),
                    shown(want)
                ));
            }
        }
    }
}

/// Adds a problem unless some item of `live` contains `want`.
fn compare_item(path: &str, want: &Value, live: &[Value], problems: &mut Vec<String>) {
    let attempt = |candidate: &Value| {
        let mut found = Vec::new();
        compare("", want, Some(candidate), &mut found);
        found
    };
    let candidates: Vec<&Value> = match identity(want) {
        Some((key, name)) => live
            .iter()
            .filter(|item| item.get(key).and_then(Value::as_str) == Some(name))
            .collect(),
        None => live.iter().collect(),
    };
    if candidates
        .iter()
        .any(|candidate| attempt(candidate).is_empty())
    {
        return;
    }
    let what = match identity(want) {
        Some(("type", name)) => format!("the `{name}` rule"),
        Some((_, name)) => format!("`{name}`"),
        None => shown(want),
    };
    // The nearest candidate says what differs; with none, the item is missing.
    let nearest = candidates
        .iter()
        .map(|candidate| attempt(candidate))
        .min_by_key(Vec::len);
    match nearest {
        None => problems.push(format!("{path}: {what} is missing")),
        Some(found) => {
            for problem in found {
                problems.push(format!("{path}: {what}: {problem}"));
            }
        }
    }
}

/// The repository to read: `--repo`, else `$GITHUB_REPOSITORY`, else this checkout's, from
/// `gh repo view`.
fn repository(context: &Context) -> Result<String, String> {
    let repo = match context.args.get("--repo") {
        Some(repo) => repo.to_owned(),
        None => match std::env::var("GITHUB_REPOSITORY") {
            Ok(repo) if !repo.is_empty() => repo,
            _ => {
                let output = Command::new("gh")
                    .args([
                        "repo",
                        "view",
                        "--json",
                        "nameWithOwner",
                        "--jq",
                        ".nameWithOwner",
                    ])
                    .current_dir(&context.root)
                    .output()
                    .map_err(|e| format!("could not start `gh` to find the repository: {e}"))?;
                if !output.status.success() {
                    return Err(
                        "`gh repo view` could not name this checkout's repository; pass \
                         `--repo <owner/name>`"
                            .to_owned(),
                    );
                }
                String::from_utf8_lossy(&output.stdout).trim().to_owned()
            }
        },
    };
    checked_repo(repo)
}

/// `repo` if it is `owner/name` with characters GitHub allows, so it is safe in a URL path.
fn checked_repo(repo: String) -> Result<String, String> {
    let valid = |part: &str| {
        !part.is_empty()
            && part != "."
            && part != ".."
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    match repo.split_once('/') {
        Some((owner, name)) if valid(owner) && valid(name) => Ok(repo),
        _ => Err(format!(
            "`{repo}` is not a repository name of the form owner/name"
        )),
    }
}

/// `cargo xtask ci repo-settings [--repo <owner/name>] [--from <dir>] [--save <dir>]`.
pub fn step(context: &Context) -> Result<(), String> {
    let from = context.args.get("--from");
    let save = context.args.get("--save");
    let mut responses: Box<dyn Responses> = match (from, save) {
        (Some(_), Some(_)) => {
            return Err("`--from` reads recorded responses, so `--save` has none to save".into());
        }
        (Some(dir), None) => Box::new(Recorded(PathBuf::from(dir))),
        (None, save) => {
            let repo = repository(context)?;
            eprintln!("repo-settings: reading {repo} through the REST API");
            Box::new(Gh {
                repo,
                save: save.map(PathBuf::from),
            })
        }
    };
    let (passed, problems) = check(&context.root.join(DECLARED), responses.as_mut());
    for line in passed {
        eprintln!("repo-settings: {line}");
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

#[cfg(test)]
mod tests;
