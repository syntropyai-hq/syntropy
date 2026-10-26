// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `renovate` step of `lint`: `renovate.json5` carries the policy of RFC-0062 §11.1
//! (QUAL-E1-T10).
//!
//! The file is strict JSON, which JSON5 accepts, so this step reads it with `serde_json` and the
//! workspace takes on no JSON5 parser. It holds no comments: the `description` fields say why.
//!
//! The step checks the settings §11.1 names:
//!
//! - the schedule is Monday, early UTC, and `minimumReleaseAge` is `3 days`;
//! - `vulnerabilityAlerts` are on, run at any time, and wait for no release age;
//! - nothing auto-merges unless a rule says so, and GitHub Actions and CI images are pinned by
//!   digest;
//! - every sample package of [`SAMPLES`] gets the update behaviour §11.1 gives its class. The
//!   samples include crates that bind a GPU runtime, ggml, HiGHS, QUIC and TLS, RDMA, float
//!   formats and tokenizers, and the toolchains. Each critical sample must come out carrying
//!   `full-validate` and not auto-merging, whatever its update type.
//!
//! The samples run through [`resolve`], a small model of how Renovate applies `packageRules`:
//! rules apply in order, a later scalar setting replaces an earlier one, and `addLabels`
//! accumulate. [`resolve`] understands only the `match*` keys in [`KNOWN_MATCHERS`]. A rule with
//! another key fails the step, because the model could not say what the rule does.
//!
//! This is a structural check, not Renovate's own validator. `renovate-config-validator` is an
//! npm tool and a pinned install of it is a new external tool that needs a maintainer's approval,
//! so that check is a follow-up (QUAL-E1-T34). The label `full-validate` is a no-op until
//! QUAL-E1-T29 makes `ci-gate` act on it.

use std::collections::BTreeSet;
use std::fs;

use serde_json::Value;

use crate::run::Context;

/// The config file, relative to the workspace root.
pub const FILE: &str = "renovate.json5";

/// The label that requests every hardware job and the nightly suites (RFC-0062 §11.1).
pub const FULL_VALIDATE: &str = "full-validate";

/// The `match*` keys [`resolve`] models.
const KNOWN_MATCHERS: [&str; 3] = ["matchManagers", "matchUpdateTypes", "matchPackageNames"];

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
    let problems = check(&text);
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

fn check_settings(config: &Value, problems: &mut Vec<String>) {
    let mut expect = |ok: bool, what: &str| {
        if !ok {
            problems.push(format!("{FILE}: {what}"));
        }
    };
    let schedule_ok = config["schedule"].as_array().is_some_and(|items| {
        items.len() == 1
            && items[0]
                .as_str()
                .is_some_and(|s| s.contains("monday") && s.contains("before"))
    });
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
/// when some positive glob matches and no negated one does.
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
