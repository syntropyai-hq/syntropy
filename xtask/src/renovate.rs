// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `renovate` step of `lint`: `renovate.json5` carries the policy of RFC-0062 §11.1
//! (QUAL-E1-T10).
//!
//! The file is strict JSON, which JSON5 accepts, so this step reads it with `serde_json` and the
//! workspace takes on no JSON5 parser. It holds no comments: the `description` fields say why.
//!
//! The step checks the settings §11.1 names:
//!
//! - the schedule is one Monday window that ends by 9am UTC, and `minimumReleaseAge` is
//!   `3 days`;
//! - `vulnerabilityAlerts` are on, run at any time, wait for no release age, and neither turn on
//!   auto-merge nor add labels: Renovate applies them with `force` after every package rule, so
//!   they would override the never-auto-merge rules;
//! - nothing auto-merges unless a rule says so, and GitHub Actions and CI images are pinned by
//!   digest;
//! - `extends` is empty, so no rule comes from outside the repository, and no setting that lets
//!   an auto-merge skip its checks or the merge queue (`ignoreTests`, `automergeType`,
//!   `automergeStrategy`) appears anywhere;
//! - Renovate manages exactly the repository's own package files: the root manifest and the
//!   workspace members' for cargo, `.github/` for GitHub Actions and `ci/images/` for the CI
//!   images. Every other tracked file a manager would read, such as the xtask test fixtures,
//!   must be covered by `ignorePaths`, and no managed file may be. So a new fixture directory
//!   cannot become a dependency PR;
//! - every sample package of [`SAMPLES`] gets the update behaviour §11.1 gives its class. The
//!   samples include crates that bind a GPU runtime, ggml, HiGHS, QUIC and TLS, RDMA, float
//!   formats and tokenizers, and the toolchains. Each critical sample must come out carrying
//!   `full-validate` and not auto-merging, whatever its update type.
//!
//! The samples run through [`resolve`], a small model of how Renovate applies `packageRules`:
//! rules apply in order, a later scalar setting replaces an earlier one, and `addLabels`
//! accumulate. [`resolve`] understands only the `match*` keys in [`KNOWN_MATCHERS`]. A rule with
//! another key fails the step, because the model could not say what the rule does. For the same
//! reason a `matchPackageNames` entry that Renovate reads as a regex (`/…/` or `!/…/`) fails it,
//! as does a list with no positive entry, which Renovate reads as "every name but these".
//!
//! This is a structural check, not Renovate's own validator. `renovate-config-validator` is an
//! npm tool and a pinned install of it is a new external tool that needs a maintainer's approval,
//! so that check is a follow-up (QUAL-E1-T34). The label `full-validate` is a no-op until
//! QUAL-E1-T29 makes `ci-gate` act on it.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

use crate::run::Context;

/// The config file, relative to the workspace root.
pub const FILE: &str = "renovate.json5";

/// The label that requests every hardware job and the nightly suites (RFC-0062 §11.1).
pub const FULL_VALIDATE: &str = "full-validate";

/// The `match*` keys [`resolve`] models.
const KNOWN_MATCHERS: [&str; 3] = ["matchManagers", "matchUpdateTypes", "matchPackageNames"];

/// Settings that let an auto-merge skip its checks or the merge queue. None may appear.
const BYPASS_SETTINGS: [&str; 3] = ["ignoreTests", "automergeType", "automergeStrategy"];

/// The update types a sample is tried with.
const UPDATE_TYPES: [&str; 3] = ["major", "minor", "patch"];

/// What a class of package must come out as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    /// Never auto-merges, and carries `full-validate`, for every update type.
    Critical,
    /// Minor and patch auto-merge without the label; a major is held for dashboard approval.
    Ordinary,
    /// Minor and patch auto-merge without the label, and the dependency is pinned by digest.
    Pinned,
}

/// A package the policy is checked against: manager, name and expected class.
pub const SAMPLES: &[(&str, &str, Expect)] = &[
    ("cargo", "serde", Expect::Ordinary),
    ("cargo", "tracing", Expect::Ordinary),
    ("cargo", "clap", Expect::Ordinary),
    ("cargo", "bytes", Expect::Ordinary),
    ("cargo", "tokio", Expect::Ordinary),
    ("cargo", "tokio-util", Expect::Ordinary),
    ("cargo", "cudarc", Expect::Critical),
    ("cargo", "hip-sys", Expect::Critical),
    ("cargo", "objc2-metal", Expect::Critical),
    ("cargo", "ash", Expect::Critical),
    ("cargo", "ggml-sys", Expect::Critical),
    ("cargo", "highs-sys", Expect::Critical),
    ("cargo", "iroh", Expect::Critical),
    ("cargo", "noq-proto", Expect::Critical),
    ("cargo", "rustls", Expect::Critical),
    ("cargo", "tokio-rustls", Expect::Critical),
    ("cargo", "rdma-sys", Expect::Critical),
    ("cargo", "half", Expect::Critical),
    ("cargo", "tokenizers", Expect::Critical),
    ("custom.regex", "rust-lang/rust", Expect::Critical),
    ("dockerfile", "nvidia/cuda", Expect::Critical),
    ("github-actions", "actions/checkout", Expect::Pinned),
];

/// The settings the rules leave for one dependency and update type.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Resolved {
    /// Whether the update merges without a person.
    pub automerge: bool,
    /// The labels the rules add.
    pub labels: BTreeSet<String>,
    /// Whether a person must approve the PR on the dependency dashboard first.
    pub dashboard_approval: bool,
    /// Whether the dependency is pinned by digest.
    pub pin_digests: bool,
}

/// The `lint` step.
pub fn step(context: &Context) -> Result<(), String> {
    let path = context.root.join(FILE);
    let text = fs::read_to_string(&path).map_err(|error| format!("{FILE}: {error}"))?;
    let mut problems = check(&text);
    if problems.is_empty() {
        let tracked = tracked_files(&context.root)?;
        let members = workspace_members(&context.root)?;
        let config: Value = serde_json::from_str(&text).map_err(|e| format!("{FILE}: {e}"))?;
        check_paths(&config, &tracked, &members, &mut problems);
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// Every problem in the config `text`, one line each.
pub fn check(text: &str) -> Vec<String> {
    let config: Value = match serde_json::from_str(text) {
        Ok(config) => config,
        Err(error) => {
            return vec![format!(
                "{FILE}: not strict JSON (no comments, no trailing commas): {error}"
            )];
        }
    };
    let mut problems = Vec::new();
    check_settings(&config, &mut problems);
    check_samples(&config, &mut problems);
    problems
}

/// Whether `window` is `before Nam on monday` with N from 1 to 9: one Monday window, early UTC.
fn early_monday(window: &str) -> bool {
    window
        .strip_prefix("before ")
        .and_then(|rest| rest.strip_suffix("am on monday"))
        .is_some_and(|hour| matches!(hour, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"))
}

fn check_settings(config: &Value, problems: &mut Vec<String>) {
    let mut found = Vec::new();
    find_bypass_settings(config, "", &mut found);
    let mut expect = |ok: bool, what: &str| {
        if !ok {
            problems.push(format!("{FILE}: {what}"));
        }
    };
    for at in &found {
        expect(
            false,
            &format!(
                "`{at}` lets an auto-merge skip its checks or the merge queue (RFC-0062 §2.5)"
            ),
        );
    }
    expect(
        config["extends"] == serde_json::json!([]),
        "`extends` must be `[]`: every rule lives in this file",
    );
    let schedule_ok = config["schedule"]
        .as_array()
        .is_some_and(|items| items.len() == 1 && items[0].as_str().is_some_and(early_monday));
    expect(
        schedule_ok,
        "`schedule` must be one Monday, early-UTC window such as `before 6am on monday` (RFC-0062 §11.1)",
    );
    expect(
        config["timezone"] == "UTC",
        "`timezone` must be `UTC` (RFC-0062 §11.1)",
    );
    expect(
        config["minimumReleaseAge"] == "3 days",
        "`minimumReleaseAge` must be `3 days` (RFC-0062 §11.1)",
    );
    expect(
        config["automerge"] == false,
        "top-level `automerge` must be false: a rule turns it on for a class (RFC-0062 §11.1)",
    );
    expect(
        config["platformAutomerge"] == true,
        "`platformAutomerge` must be true, so an auto-merge goes through the merge queue (RFC-0062 §2.5)",
    );
    let alerts = &config["vulnerabilityAlerts"];
    expect(
        alerts["enabled"] == true
            && alerts["schedule"] == serde_json::json!(["at any time"])
            && alerts["minimumReleaseAge"] == "0 days",
        "`vulnerabilityAlerts` must be enabled, run `at any time` and set `minimumReleaseAge` to `0 days` (RFC-0062 §11.1)",
    );
    for key in ["automerge", "addLabels", "packageRules"] {
        expect(
            alerts.get(key).is_none(),
            &format!(
                "`vulnerabilityAlerts` must not set `{key}`: Renovate forces it over every package rule, \
                 so a critical crate's security PR would escape the never-auto-merge rules"
            ),
        );
    }
}

/// The path of every key in `value` that [`BYPASS_SETTINGS`] names, however deeply nested.
fn find_bypass_settings(value: &Value, at: &str, found: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, item) in map {
                let path = if at.is_empty() {
                    key.clone()
                } else {
                    format!("{at}.{key}")
                };
                if BYPASS_SETTINGS.contains(&key.as_str()) {
                    found.push(path.clone());
                }
                find_bypass_settings(item, &path, found);
            }
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                find_bypass_settings(item, &format!("{at}[{i}]"), found);
            }
        }
        _ => {}
    }
}

/// Which manager reads the tracked file `path`, by Renovate's default file patterns for the
/// managers in `enabledManagers`.
fn manager_of(path: &str) -> Option<&'static str> {
    let name = path.rsplit('/').next().unwrap_or(path);
    let yaml = name.ends_with(".yml") || name.ends_with(".yaml");
    if name == "Cargo.toml" {
        Some("cargo")
    } else if yaml
        && (name.starts_with("action.")
            || path.contains(".github/workflows/")
            || path.contains(".github/actions/"))
    {
        Some("github-actions")
    } else if name.starts_with("Dockerfile")
        || name.starts_with("Containerfile")
        || name.ends_with(".dockerfile")
    {
        Some("dockerfile")
    } else {
        None
    }
}

/// Whether Renovate should manage `path`, read by `manager`: the repository's own package files.
fn managed(manager: &str, path: &str, members: &[String]) -> bool {
    match manager {
        "cargo" => {
            path == "Cargo.toml"
                || members
                    .iter()
                    .any(|m| path.strip_prefix(m.as_str()) == Some("/Cargo.toml"))
        }
        "github-actions" => path.starts_with(".github/"),
        _ => path.starts_with("ci/images/"),
    }
}

/// `ignorePaths` as Renovate applies it: a pattern with `*` is a glob over `/`-separated
/// segments, in which `**` spans any number of segments; any other entry matches a path that
/// contains it.
fn ignored(patterns: &[&str], path: &str) -> bool {
    patterns.iter().any(|pattern| {
        if pattern.contains('*') {
            let pattern: Vec<&str> = pattern.split('/').collect();
            let path: Vec<&str> = path.split('/').collect();
            path_glob(&pattern, &path)
        } else {
            path.contains(pattern)
        }
    })
}

fn path_glob(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((&"**", rest)) => (0..=path.len()).any(|skip| path_glob(rest, &path[skip..])),
        Some((segment, rest)) => path
            .split_first()
            .is_some_and(|(first, tail)| glob(segment, first) && path_glob(rest, tail)),
    }
}

/// `ignorePaths` must cover every tracked package file outside the repository's own, and no
/// file of its own. `tracked` is every tracked path, `/`-separated; `members` the workspace
/// members' directories.
pub fn check_paths(
    config: &Value,
    tracked: &[String],
    members: &[String],
    problems: &mut Vec<String>,
) {
    let mut patterns = Vec::new();
    for item in config["ignorePaths"].as_array().into_iter().flatten() {
        match item.as_str() {
            Some(p) if !p.is_empty() && !p.contains(['?', '[', ']', '{', '}', '!']) => {
                patterns.push(p)
            }
            _ => problems.push(format!(
                "{FILE}: `ignorePaths` entry {item} is not a plain path or a `*`/`**` glob, which \
                 is all the lint model reads"
            )),
        }
    }
    for path in tracked {
        let Some(manager) = manager_of(path) else {
            continue;
        };
        match (managed(manager, path, members), ignored(&patterns, path)) {
            (true, true) => problems.push(format!(
                "{FILE}: `ignorePaths` hides `{path}`, which Renovate must manage ({manager})"
            )),
            (false, false) => problems.push(format!(
                "{FILE}: `{path}` is not one of the repository's own {manager} files, so \
                 `ignorePaths` must cover it (a test fixture is not a dependency)"
            )),
            _ => {}
        }
    }
}

/// Every tracked file of the checkout at `root`, `/`-separated from the root.
fn tracked_files(root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z"])
        .output()
        .map_err(|e| format!("could not start git: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "`git ls-files` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|_| "`git ls-files`: a path is not UTF-8".to_owned())
        .map(|text| {
            text.split('\0')
                .filter(|p| !p.is_empty())
                .map(str::to_owned)
                .collect()
        })
}

/// The workspace members' directories, from the root `Cargo.toml`.
fn workspace_members(root: &Path) -> Result<Vec<String>, String> {
    let text =
        fs::read_to_string(root.join("Cargo.toml")).map_err(|e| format!("Cargo.toml: {e}"))?;
    let manifest: toml::Table = toml::from_str(&text).map_err(|e| format!("Cargo.toml: {e}"))?;
    manifest
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| "Cargo.toml: no `workspace.members` array".to_owned())?
        .iter()
        .map(|m| {
            m.as_str()
                .map(|m| m.trim_end_matches('/').to_owned())
                .ok_or_else(|| "Cargo.toml: a workspace member is not a string".to_owned())
        })
        .collect()
}

fn check_samples(config: &Value, problems: &mut Vec<String>) {
    let rules = match config["packageRules"].as_array() {
        Some(rules) if !rules.is_empty() => rules,
        _ => {
            problems.push(format!("{FILE}: `packageRules` must be a non-empty list"));
            return;
        }
    };
    for (manager, name, expect) in SAMPLES {
        for update in UPDATE_TYPES {
            let resolved = match resolve(rules, manager, name, update) {
                Ok(resolved) => resolved,
                Err(error) => {
                    problems.push(format!("{FILE}: {error}"));
                    return;
                }
            };
            let label = format!("{manager} `{name}` {update}");
            let labelled = resolved.labels.contains(FULL_VALIDATE);
            match expect {
                Expect::Critical => {
                    if resolved.automerge {
                        problems.push(format!("{FILE}: {label} must never auto-merge"));
                    }
                    if !labelled {
                        problems.push(format!("{FILE}: {label} must carry `{FULL_VALIDATE}`"));
                    }
                }
                Expect::Ordinary | Expect::Pinned => {
                    if labelled {
                        problems.push(format!("{FILE}: {label} must not carry `{FULL_VALIDATE}`"));
                    }
                    let merges = update != "major";
                    if resolved.automerge != merges {
                        problems.push(format!(
                            "{FILE}: {label} must {}auto-merge",
                            if merges { "" } else { "not " }
                        ));
                    }
                    if *expect == Expect::Ordinary
                        && update == "major"
                        && !resolved.dashboard_approval
                    {
                        problems.push(format!("{FILE}: {label} must wait for dashboard approval"));
                    }
                    if *expect == Expect::Pinned && !resolved.pin_digests {
                        problems.push(format!("{FILE}: {label} must be pinned by digest"));
                    }
                }
            }
        }
    }
}

/// What the `packageRules` leave for the dependency `name` of `manager`, at `update`.
pub fn resolve(
    rules: &[Value],
    manager: &str,
    name: &str,
    update: &str,
) -> Result<Resolved, String> {
    let mut out = Resolved::default();
    for (at, rule) in rules.iter().enumerate() {
        let Some(rule) = rule.as_object() else {
            return Err(format!("packageRules[{at}] is not an object"));
        };
        let mut applies = true;
        for (key, value) in rule {
            if key.starts_with("match") || key.starts_with("exclude") {
                if !KNOWN_MATCHERS.contains(&key.as_str()) {
                    return Err(format!(
                        "packageRules[{at}] uses `{key}`, which the lint model does not understand; \
                         extend xtask/src/renovate.rs with it"
                    ));
                }
                let Some(items) = value.as_array() else {
                    return Err(format!("packageRules[{at}] `{key}` is not a list"));
                };
                let items: Vec<&str> = items.iter().filter_map(Value::as_str).collect();
                if key == "matchPackageNames" {
                    if let Some(regex) = items
                        .iter()
                        .find(|i| i.starts_with('/') || i.starts_with("!/"))
                    {
                        return Err(format!(
                            "packageRules[{at}] `matchPackageNames` entry `{regex}` is a regex to \
                             Renovate, which the lint model does not read; use a glob"
                        ));
                    }
                    if !items.iter().any(|i| !i.starts_with('!')) {
                        return Err(format!(
                            "packageRules[{at}] `matchPackageNames` has no positive entry, which \
                             Renovate reads as every name but the negated ones; add `*` to say so"
                        ));
                    }
                }
                applies &= match key.as_str() {
                    "matchManagers" => items.contains(&manager),
                    "matchUpdateTypes" => items.contains(&update),
                    _ => matches_names(&items, name),
                };
            }
        }
        if !applies {
            continue;
        }
        if let Some(on) = rule.get("automerge").and_then(Value::as_bool) {
            out.automerge = on;
        }
        if let Some(on) = rule
            .get("dependencyDashboardApproval")
            .and_then(Value::as_bool)
        {
            out.dashboard_approval = on;
        }
        if let Some(on) = rule.get("pinDigests").and_then(Value::as_bool) {
            out.pin_digests = on;
        }
        if let Some(labels) = rule.get("addLabels").and_then(Value::as_array) {
            out.labels
                .extend(labels.iter().filter_map(Value::as_str).map(str::to_owned));
        }
    }
    Ok(out)
}

/// Renovate's `matchPackageNames`: a glob with `*`, or a `!glob` that excludes. The list matches
/// when some positive glob matches and no negated one does. [`resolve`] refuses regex entries
/// and a list with no positive entry before this runs.
fn matches_names(patterns: &[&str], name: &str) -> bool {
    let mut positive = false;
    for pattern in patterns {
        match pattern.strip_prefix('!') {
            Some(negated) if glob(negated, name) => return false,
            Some(_) => {}
            None => positive |= glob(pattern, name),
        }
    }
    positive
}

/// `*` matches any run of characters, and nothing else is special.
fn glob(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let first = parts[0];
    let last = parts[parts.len() - 1];
    if text.len() < first.len() + last.len() || !text.starts_with(first) || !text.ends_with(last) {
        return false;
    }
    let mut rest = &text[first.len()..text.len() - last.len()];
    for part in &parts[1..parts.len() - 1] {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real() -> String {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(FILE);
        fs::read_to_string(root).expect("renovate.json5 exists")
    }

    fn rules(text: &str) -> Vec<Value> {
        let config: Value = serde_json::from_str(text).unwrap();
        config["packageRules"].as_array().unwrap().clone()
    }

    #[test]
    fn the_checked_in_config_passes() {
        assert_eq!(check(&real()), Vec::<String>::new());
    }

    #[test]
    fn a_gpu_binding_update_carries_full_validate_and_no_auto_merge() {
        let rules = rules(&real());
        for update in UPDATE_TYPES {
            let resolved = resolve(&rules, "cargo", "cudarc", update).unwrap();
            assert!(!resolved.automerge, "{update}");
            assert!(resolved.labels.contains(FULL_VALIDATE), "{update}");
        }
    }

    #[test]
    fn an_ordinary_minor_update_auto_merges_and_a_major_is_held() {
        let rules = rules(&real());
        let minor = resolve(&rules, "cargo", "serde", "minor").unwrap();
        assert!(minor.automerge && minor.labels.is_empty());
        let major = resolve(&rules, "cargo", "serde", "major").unwrap();
        assert!(!major.automerge && major.dashboard_approval);
    }

    #[test]
    fn a_critical_rule_placed_before_the_ordinary_one_is_caught() {
        let mut config: Value = serde_json::from_str(&real()).unwrap();
        let list = config["packageRules"].as_array_mut().unwrap();
        let critical = list.remove(list.len() - 2);
        list.insert(0, critical);
        let problems = check(&config.to_string());
        assert!(
            problems
                .iter()
                .any(|p| p.contains("`cudarc`") && p.contains("never auto-merge")),
            "{problems:?}"
        );
    }

    #[test]
    fn dropping_the_label_or_the_release_age_is_caught() {
        let text = real().replace("\"full-validate\"", "\"other\"");
        assert!(
            check(&text)
                .iter()
                .any(|p| p.contains("must carry `full-validate`"))
        );
        let text = real().replace("\"3 days\"", "\"0 days\"");
        assert!(check(&text).iter().any(|p| p.contains("minimumReleaseAge")));
    }

    #[test]
    fn a_rule_with_a_matcher_the_model_lacks_fails_closed() {
        let text = real().replace(
            "\"matchManagers\": [\"dockerfile\"]",
            "\"matchFileNames\": [\"x\"]",
        );
        assert!(check(&text).iter().any(|p| p.contains("matchFileNames")));
    }

    fn tracked() -> Vec<String> {
        [
            "Cargo.toml",
            "crates/core/Cargo.toml",
            "xtask/Cargo.toml",
            ".github/workflows/pr.yml",
            ".github/actions/setup-rust/action.yml",
            "ci/images/cuda-13/Dockerfile",
            "xtask/tests/fixtures/licences/deny-openssl/crates/scratch/Cargo.toml",
            "xtask/tests/fixtures/workflows/bad/.github/workflows/x.yml",
            "README.md",
        ]
        .map(str::to_owned)
        .to_vec()
    }

    fn members() -> Vec<String> {
        vec!["crates/core".to_owned(), "xtask".to_owned()]
    }

    fn path_problems(config: &Value, tracked: &[String]) -> Vec<String> {
        let mut problems = Vec::new();
        check_paths(config, tracked, &members(), &mut problems);
        problems
    }

    #[test]
    fn the_checked_in_ignore_paths_cover_the_fixtures_and_nothing_managed() {
        let config: Value = serde_json::from_str(&real()).unwrap();
        assert_eq!(path_problems(&config, &tracked()), Vec::<String>::new());
    }

    #[test]
    fn the_repository_package_files_pass() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let config: Value = serde_json::from_str(&real()).unwrap();
        let tracked = tracked_files(&root).expect("git ls-files");
        let members = workspace_members(&root).expect("workspace members");
        let mut problems = Vec::new();
        check_paths(&config, &tracked, &members, &mut problems);
        assert_eq!(problems, Vec::<String>::new());
    }

    #[test]
    fn a_fixture_manifest_outside_ignore_paths_is_caught() {
        let mut config: Value = serde_json::from_str(&real()).unwrap();
        config.as_object_mut().unwrap().remove("ignorePaths");
        let problems = path_problems(&config, &tracked());
        assert!(
            problems.iter().any(|p| p.contains("deny-openssl")),
            "{problems:?}"
        );
        assert!(
            problems.iter().any(|p| p.contains("bad/.github")),
            "{problems:?}"
        );
        // A new fixture root elsewhere is caught too.
        let config: Value = serde_json::from_str(&real()).unwrap();
        let mut more = tracked();
        more.push("bench/testdata/scratch/Cargo.toml".to_owned());
        let problems = path_problems(&config, &more);
        assert!(
            problems.iter().any(|p| p.contains("bench/testdata")),
            "{problems:?}"
        );
    }

    #[test]
    fn ignore_paths_that_hide_a_managed_file_are_caught() {
        let mut config: Value = serde_json::from_str(&real()).unwrap();
        config["ignorePaths"] = serde_json::json!(["**"]);
        let problems = path_problems(&config, &tracked());
        assert!(
            problems.iter().any(|p| p.contains("hides `Cargo.toml`")),
            "{problems:?}"
        );
        config["ignorePaths"] = serde_json::json!(["crates/"]);
        let problems = path_problems(&config, &tracked());
        assert!(
            problems
                .iter()
                .any(|p| p.contains("hides `crates/core/Cargo.toml`")),
            "{problems:?}"
        );
        config["ignorePaths"] = serde_json::json!(["**/fixtures/{a,b}/**"]);
        let problems = path_problems(&config, &tracked());
        assert!(
            problems.iter().any(|p| p.contains("not a plain path")),
            "{problems:?}"
        );
    }

    #[test]
    fn path_globs() {
        assert!(ignored(
            &["**/tests/fixtures/**"],
            "xtask/tests/fixtures/a/Cargo.toml"
        ));
        assert!(ignored(
            &["**/tests/fixtures/**"],
            "tests/fixtures/Cargo.toml"
        ));
        assert!(!ignored(&["**/tests/fixtures/**"], "xtask/Cargo.toml"));
        assert!(ignored(&["xtask/*/Cargo.toml"], "xtask/a/Cargo.toml"));
        assert!(!ignored(&["xtask/*/Cargo.toml"], "xtask/a/b/Cargo.toml"));
    }

    #[test]
    fn regex_or_negation_only_package_names_fail_closed() {
        for entry in [r#"["/^cud/"]"#, r#"["!serde"]"#, r#"["!/^serde/", "x"]"#] {
            let rule = format!(r#"{{"matchPackageNames": {entry}, "automerge": true}}"#);
            let text = real().replacen(
                "\"packageRules\": [",
                &format!("\"packageRules\": [{rule},"),
                1,
            );
            let problems = check(&text);
            assert!(
                problems.iter().any(|p| p.contains("matchPackageNames")),
                "{entry}: {problems:?}"
            );
        }
    }

    #[test]
    fn alerts_that_auto_merge_or_settings_that_bypass_checks_are_caught() {
        let mut config: Value = serde_json::from_str(&real()).unwrap();
        config["vulnerabilityAlerts"]["automerge"] = Value::Bool(true);
        assert!(
            check(&config.to_string())
                .iter()
                .any(|p| p.contains("must not set `automerge`"))
        );
        let mut config: Value = serde_json::from_str(&real()).unwrap();
        config["ignoreTests"] = Value::Bool(true);
        config["packageRules"][0]["automergeType"] = Value::from("branch");
        let problems = check(&config.to_string());
        assert!(
            problems.iter().any(|p| p.contains("`ignoreTests`")),
            "{problems:?}"
        );
        assert!(
            problems
                .iter()
                .any(|p| p.contains("`packageRules[0].automergeType`")),
            "{problems:?}"
        );
        let mut config: Value = serde_json::from_str(&real()).unwrap();
        config["extends"] = serde_json::json!(["github>someone/presets"]);
        assert!(
            check(&config.to_string())
                .iter()
                .any(|p| p.contains("`extends`"))
        );
    }

    #[test]
    fn the_schedule_must_end_early_on_monday() {
        assert!(early_monday("before 6am on monday"));
        assert!(!early_monday("before 11pm on monday"));
        assert!(!early_monday("before 10am on monday"));
        assert!(!early_monday("before 6am on tuesday"));
    }

    #[test]
    fn comments_are_refused() {
        let problems = check("// note\n{}");
        assert!(problems[0].contains("not strict JSON"), "{problems:?}");
    }

    #[test]
    fn globs_and_negations() {
        assert!(glob("ggml*", "ggml-sys"));
        assert!(glob("*-rdma", "foo-rdma"));
        assert!(glob("a*c*e", "abcde"));
        assert!(!glob("iroh", "iroh-base"));
        assert!(matches_names(&["*", "!iroh"], "serde"));
        assert!(!matches_names(&["*", "!iroh"], "iroh"));
    }
}
