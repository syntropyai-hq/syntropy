// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the settings check over recorded responses. `fixtures/repo-settings/complete/` is a
//! repository in the declared state; `defaults/` is a public repository on GitHub's defaults.
//! The mutation test removes or changes each declared setting of `complete` in turn.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;

use super::*;
use crate::cli::{self, Command as Cli};
use crate::registry;

/// The workspace's own declared state.
fn declared() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(DECLARED)
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/repo-settings")
        .join(name)
}

/// Responses held in memory, by file name.
struct Memory(BTreeMap<String, String>);

impl Memory {
    fn load(name: &str) -> Memory {
        let mut files = BTreeMap::new();
        for entry in std::fs::read_dir(fixture(name)).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().into_owned();
            files.insert(name, std::fs::read_to_string(entry.path()).unwrap());
        }
        Memory(files)
    }
}

impl Responses for Memory {
    fn get(&mut self, _endpoint: &str, file: &str, _paginate: bool) -> Result<String, String> {
        self.0
            .get(file)
            .cloned()
            .ok_or_else(|| format!("no recorded response `{file}`"))
    }
}

#[test]
fn the_declared_state_passes_on_a_repository_in_that_state() {
    let (passed, problems) = check(&declared(), &mut Memory::load("complete"));
    assert_eq!(problems, Vec::<String>::new());
    assert_eq!(
        passed,
        [
            "repository.json: matches",
            "private-vulnerability-reporting.json: matches",
            "actions-permissions.json: matches",
            "actions-selected-actions.json: matches",
            "actions-workflow-permissions.json: matches",
            "fork-pr-contributor-approval.json: matches",
            "ruleset-main-code-owners.json: its rules are in force",
            "ruleset-main.json: its rules are in force",
            "ruleset-release.json: its rules are in force",
            "ruleset-tags.json: its rules are in force and its bypass actors are the declared ones",
        ]
    );
}

#[test]
fn recorded_responses_are_read_from_a_directory() {
    let (_, problems) = check(&declared(), &mut Recorded(fixture("complete")));
    assert_eq!(problems, Vec::<String>::new());
}

#[test]
fn a_repository_on_githubs_defaults_fails_on_every_setting() {
    let (passed, problems) = check(&declared(), &mut Memory::load("defaults"));
    assert_eq!(passed, Vec::<String>::new());
    let repo = "`repository.json` (RFC-0062 §8.3, §12.3; ADR-0620 decision 4)";
    let pvr = "`private-vulnerability-reporting.json` (RFC-0062 §13; ADR-0620 decision 4)";
    let actions = "`actions-permissions.json` (ADR-0620 decision 4)";
    let selected = "`actions-selected-actions.json` (ADR-0620 decision 4)";
    let fork = "`fork-pr-contributor-approval.json` (RFC-0062 §3.3 item 1; ADR-0620 decision 4)";
    let workflow = "`actions-workflow-permissions.json` (RFC-0062 §12.4)";
    let rules = "(RFC-0062 §2.5, §12.3, §12.6; ADR-0621 decision 2)";
    let expected = [
        format!("{repo}: `allow_merge_commit` is `true`; want `false`"),
        format!("{repo}: `allow_rebase_merge` is `true`; want `false`"),
        format!(
            "{repo}: `security_and_analysis`.`secret_scanning`.`status` is `\"disabled\"`; \
             want `\"enabled\"`"
        ),
        format!(
            "{repo}: `security_and_analysis`.`secret_scanning_push_protection`.`status` is \
             `\"disabled\"`; want `\"enabled\"`"
        ),
        format!(
            "{repo}: `squash_merge_commit_message` is `\"COMMIT_MESSAGES\"`; want `\"PR_BODY\"`"
        ),
        format!(
            "{repo}: `squash_merge_commit_title` is `\"COMMIT_OR_PR_TITLE\"`; want `\"PR_TITLE\"`"
        ),
        format!("{pvr}: `enabled` is `false`; want `true`"),
        format!("{actions}: `allowed_actions` is `\"all\"`; want `\"selected\"`"),
        format!("{actions}: `sha_pinning_required` is `false`; want `true`"),
        format!("{selected}: no recorded response `actions-selected-actions.json`"),
        format!("{workflow}: `can_approve_pull_request_reviews` is `true`; want `false`"),
        format!("{workflow}: `default_workflow_permissions` is `\"write\"`; want `\"read\"`"),
        format!(
            "{fork}: `approval_policy` is `\"first_time_contributors\"`; \
             want `\"all_external_contributors\"`"
        ),
        format!(
            "`ruleset-main-code-owners.json` {rules}: rules on `main`: the `pull_request` rule is missing"
        ),
        format!("`ruleset-main.json` {rules}: rules on `main`: the `deletion` rule is missing"),
        format!(
            "`ruleset-main.json` {rules}: rules on `main`: the `non_fast_forward` rule is missing"
        ),
        format!("`ruleset-main.json` {rules}: rules on `main`: the `pull_request` rule is missing"),
        format!(
            "`ruleset-main.json` {rules}: rules on `main`: the `required_status_checks` rule is \
             missing"
        ),
        format!(
            "`ruleset-release.json` {rules}: rules on `release/probe`: the `deletion` rule is missing"
        ),
        format!(
            "`ruleset-release.json` {rules}: rules on `release/probe`: the `non_fast_forward` rule \
             is missing"
        ),
        "`ruleset-tags.json` (RFC-0062 §8.3, §12.6; ADR-0621 decision 2): the repository has no \
         ruleset named `release-tags`"
            .to_owned(),
    ];
    assert_eq!(problems, expected);
}

/// Every way to break `live` at a path `want` declares: drop a member, drop or add a list item,
/// change a value. Each comes with a description.
fn mutations(want: &Value, live: &Value) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    match want {
        Value::Object(want) => {
            let live = live
                .as_object()
                .expect("the fixture has the declared object");
            for (name, want_member) in want {
                let mut dropped = live.clone();
                dropped.remove(name);
                out.push((format!("without `{name}`"), Value::Object(dropped)));
                let live_member = live
                    .get(name)
                    .expect("the fixture has every declared member");
                for (what, changed) in mutations(want_member, live_member) {
                    let mut copy = live.clone();
                    copy.insert(name.clone(), changed);
                    out.push((format!("`{name}` {what}"), Value::Object(copy)));
                }
            }
        }
        Value::Array(want) if want.iter().all(is_scalar) => {
            let live = live.as_array().expect("the fixture has the declared list");
            for index in 0..live.len() {
                let mut shorter = live.clone();
                let removed = shorter.remove(index);
                out.push((format!("without {removed}"), Value::Array(shorter)));
            }
            let mut longer = live.clone();
            longer.push(Value::String("extra-item".to_owned()));
            out.push(("with an extra item".to_owned(), Value::Array(longer)));
        }
        Value::Array(want) => {
            let live = live.as_array().expect("the fixture has the declared list");
            for item in want {
                let matches = |candidate: &Value| {
                    let mut found = Vec::new();
                    compare("", item, Some(candidate), &mut found);
                    found.is_empty()
                };
                let first = live
                    .iter()
                    .position(matches)
                    .expect("the fixture has every declared item");
                let without: Vec<Value> = live.iter().filter(|c| !matches(c)).cloned().collect();
                out.push((format!("without the item {item}"), Value::Array(without)));
                for (what, changed) in mutations(item, &live[first]) {
                    let mut copy = live.clone();
                    copy[first] = changed;
                    out.push((format!("item {item}: {what}"), Value::Array(copy)));
                }
            }
        }
        Value::Bool(b) => out.push(("flipped".to_owned(), Value::Bool(!b))),
        Value::Number(n) => out.push((
            "changed".to_owned(),
            Value::from(n.as_i64().expect("declared numbers are integers") + 1),
        )),
        Value::String(s) => out.push(("changed".to_owned(), Value::String(format!("{s}-changed")))),
        Value::Null => out.push(("set".to_owned(), Value::Bool(true))),
    }
    out
}

/// Checks the declared state against `complete` with `file` replaced by `text`.
fn problems_with(file: &str, text: String) -> Vec<String> {
    let mut responses = Memory::load("complete");
    responses.0.insert(file.to_owned(), text);
    check(&declared(), &mut responses).1
}

#[test]
fn removing_or_changing_any_declared_setting_fails() {
    let mut cases = 0;
    for setting in SETTINGS {
        let want = read_json(&declared().join(setting.file)).unwrap();
        let live: Value = serde_json::from_str(
            &std::fs::read_to_string(fixture("complete").join(setting.file)).unwrap(),
        )
        .unwrap();
        for (what, changed) in mutations(&want, &live) {
            let problems = problems_with(setting.file, changed.to_string());
            assert!(
                problems
                    .iter()
                    .any(|p| p.starts_with(&format!("`{}`", setting.file))),
                "{} {what}: passed, or failed without naming it: {problems:?}",
                setting.file
            );
            cases += 1;
        }
        let problems = problems_with(setting.file, "not json".to_owned());
        assert_eq!(problems.len(), 1, "{problems:?}");
    }
    assert!(cases >= 30, "only {cases} cases");
}

#[test]
fn removing_or_changing_any_declared_rule_fails() {
    let mut cases = 0;
    for file in ruleset_files(&declared()).unwrap() {
        let ruleset = read_json(&declared().join(&file)).unwrap();
        // A tag ruleset is read whole; `the_tag_ruleset_is_read_whole` mutates it.
        if tags::is_tag_ruleset(&ruleset) {
            continue;
        }
        let (patterns, rules) = ruleset_parts(&ruleset).unwrap();
        for pattern in patterns {
            let branch = probe_branch(pattern).unwrap();
            let response = rules_file(&branch);
            let live: Value = serde_json::from_str(
                &std::fs::read_to_string(fixture("complete").join(&response)).unwrap(),
            )
            .unwrap();
            for (what, changed) in mutations(&Value::Array(rules.to_vec()), &live) {
                let problems = problems_with(&response, changed.to_string());
                assert!(
                    problems.iter().any(|p| p.starts_with(&format!("`{file}`"))),
                    "{file} on {branch}, {what}: passed, or failed without naming it: {problems:?}"
                );
                cases += 1;
            }
            // The whole ruleset gone: no rule of it in force on the branch.
            let mut problems = problems_with(&response, "[]".to_owned());
            problems.retain(|p| p.starts_with(&format!("`{file}`")));
            assert_eq!(problems.len(), rules.len(), "{file}: {problems:?}");
        }
    }
    assert!(cases >= 40, "only {cases} cases");
}

#[test]
fn a_missing_response_fails_and_says_which() {
    let mut responses = Memory::load("complete");
    responses.0.remove("rules-release-probe.json");
    let (_, problems) = check(&declared(), &mut responses);
    assert_eq!(
        problems,
        [
            "`ruleset-release.json` (RFC-0062 §2.5, §12.3, §12.6; ADR-0621 decision 2): no \
             recorded response `rules-release-probe.json`"
        ]
    );
}

#[test]
fn settings_hidden_from_a_weaker_token_read_as_missing() {
    // Without administration read access, `GET /repos/{repo}` leaves out `security_and_analysis`.
    let mut live: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture("complete").join("repository.json")).unwrap(),
    )
    .unwrap();
    live.as_object_mut()
        .unwrap()
        .remove("security_and_analysis");
    let problems = problems_with("repository.json", live.to_string());
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("`security_and_analysis` is missing"),
        "{problems:?}"
    );
}

#[test]
fn a_rule_from_another_ruleset_of_the_same_type_does_not_stand_in() {
    // The code-owner rule is in force, but the main ruleset's own pull_request rule is gone.
    let live: Vec<Value> = serde_json::from_str(
        &std::fs::read_to_string(fixture("complete").join("rules-main.json")).unwrap(),
    )
    .unwrap();
    let kept: Vec<&Value> = live
        .iter()
        .filter(|rule| {
            rule["type"] != "pull_request"
                || rule["parameters"]["require_code_owner_review"] == true
        })
        .collect();
    let problems = problems_with("rules-main.json", serde_json::to_string(&kept).unwrap());
    assert_eq!(
        problems,
        [
            "`ruleset-main.json` (RFC-0062 §2.5, §12.3, §12.6; ADR-0621 decision 2): rules on \
             `main`: the `pull_request` rule: `parameters`.`dismiss_stale_reviews_on_push` is \
             `true`; want `false`",
            "`ruleset-main.json` (RFC-0062 §2.5, §12.3, §12.6; ADR-0621 decision 2): rules on \
             `main`: the `pull_request` rule: `parameters`.`require_code_owner_review` is `true`; \
             want `false`",
        ]
    );
}

#[test]
fn a_required_check_from_another_app_does_not_count() {
    // A commit status anyone with write access can post would satisfy a check with no app.
    let text = std::fs::read_to_string(fixture("complete").join("rules-main.json"))
        .unwrap()
        .replace(
            r#"{"context":"dco","integration_id":15368}"#,
            r#"{"context":"dco"}"#,
        );
    let problems = problems_with("rules-main.json", text);
    assert_eq!(
        problems,
        [
            "`ruleset-main.json` (RFC-0062 §2.5, §12.3, §12.6; ADR-0621 decision 2): rules on \
             `main`: the `required_status_checks` rule: `parameters`.`required_status_checks`: \
             `dco`: `integration_id` is missing; want `15368`"
        ]
    );
}

#[test]
fn pages_of_rules_are_read_as_one_list() {
    let pages = r#"[{"type":"deletion"}]
[{"type":"non_fast_forward"}][]"#;
    let items = json_pages(pages).unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(
        json_pages(r#"{"message":"Not Found"}"#).unwrap_err(),
        "the response is not a JSON array"
    );
    assert!(
        json_pages("[")
            .unwrap_err()
            .starts_with("the response is not JSON")
    );
}

#[test]
fn branch_patterns_are_checked_on_a_probe_branch() {
    assert_eq!(probe_branch("refs/heads/main").unwrap(), "main");
    assert_eq!(
        probe_branch("refs/heads/release/*").unwrap(),
        "release/probe"
    );
    assert_eq!(
        probe_branch("refs/heads/release/**").unwrap(),
        "release/probe"
    );
    for bad in [
        "~DEFAULT_BRANCH",
        "refs/tags/v*",
        "refs/heads/",
        "refs/heads/v[0-9]",
    ] {
        assert!(probe_branch(bad).is_err(), "`{bad}` was accepted");
    }
    assert_eq!(rules_file("release/probe"), "rules-release-probe.json");
}

#[test]
fn a_ruleset_this_check_cannot_read_is_refused() {
    let ruleset = |target: &str, enforcement: &str, include: &str, rules: &str| -> Value {
        serde_json::from_str(&format!(
            r#"{{"target":"{target}","enforcement":"{enforcement}",
                "conditions":{{"ref_name":{{"include":{include}}}}},"rules":{rules}}}"#
        ))
        .unwrap()
    };
    let good = ruleset(
        "branch",
        "active",
        r#"["refs/heads/main"]"#,
        r#"[{"type":"deletion"}]"#,
    );
    assert!(ruleset_parts(&good).is_ok());
    for (bad, why) in [
        (
            ruleset(
                "tag",
                "active",
                r#"["refs/tags/v*"]"#,
                r#"[{"type":"deletion"}]"#,
            ),
            "its `target` is not `branch`",
        ),
        (
            ruleset(
                "branch",
                "evaluate",
                r#"["refs/heads/main"]"#,
                r#"[{"type":"deletion"}]"#,
            ),
            "its `enforcement` is not `active`, so it would enforce nothing",
        ),
        (
            ruleset("branch", "active", "[]", r#"[{"type":"deletion"}]"#),
            "it names no branch in `conditions.ref_name.include`",
        ),
        (
            ruleset("branch", "active", r#"["refs/heads/main"]"#, "[]"),
            "it has no `rules`",
        ),
    ] {
        assert_eq!(ruleset_parts(&bad).unwrap_err(), why);
    }
}

/// The tag ruleset as `complete` records it, changed by `edit`, checked against the declared one.
fn tag_problems(edit: impl FnOnce(&mut Value)) -> Vec<String> {
    let file = tags::ruleset_file(7000004);
    let mut live: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("complete").join(&file)).unwrap())
            .unwrap();
    edit(&mut live);
    let mut responses = Memory::load("complete");
    responses.0.insert(file, live.to_string());
    let declared = read_json(&declared().join(tags::TAG_RULESET)).unwrap();
    tags::check_ruleset(&declared, &mut responses)
}

/// The task's criterion (QUAL-E17-T01): the weekly check fails on a fixture ruleset with a second
/// bypass actor, and names it.
#[test]
fn a_second_tag_bypass_actor_fails_the_weekly_check() {
    let declared = read_json(&declared().join(tags::TAG_RULESET)).unwrap();
    let problems = tags::check_ruleset(&declared, &mut Recorded(fixture("tag-second-actor")));
    assert_eq!(
        problems,
        [
            "ruleset `release-tags`: `bypass_actors` has RepositoryRole 5 (always), which is not \
             declared: only the release App may bypass the tag ruleset (RFC-0062 §8.3)"
        ]
    );
    assert!(tags::check_ruleset(&declared, &mut Recorded(fixture("complete"))).is_empty());
}

#[test]
fn the_tag_ruleset_is_read_whole() {
    assert_eq!(tag_problems(|_| {}), Vec::<String>::new());
    // Another App in place of the release App.
    let problems = tag_problems(|live| live["bypass_actors"][0]["actor_id"] = Value::from(99));
    assert_eq!(
        problems,
        [
            "ruleset `release-tags`: `bypass_actors` lacks Integration 0 (always); the declared \
             ID 0 stands for the release App until a maintainer records its App ID in the file",
            "ruleset `release-tags`: `bypass_actors` has Integration 99 (always), which is not \
             declared: only the release App may bypass the tag ruleset (RFC-0062 §8.3)",
        ]
    );
    // A token that cannot edit the ruleset does not see its bypass actors.
    let problems = tag_problems(|live| {
        live.as_object_mut().unwrap().remove("bypass_actors");
    });
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("does not show `bypass_actors`"),
        "{problems:?}"
    );
    // Each rule, the enforcement and the tag pattern count.
    for rule in 0..3 {
        let problems = tag_problems(|live| {
            live["rules"].as_array_mut().unwrap().remove(rule);
        });
        assert_eq!(problems.len(), 1, "rule {rule}: {problems:?}");
        assert!(problems[0].contains("rule is missing"), "{problems:?}");
    }
    let problems = tag_problems(|live| live["enforcement"] = Value::from("evaluate"));
    assert_eq!(
        problems,
        ["ruleset `release-tags`.`enforcement` is `\"evaluate\"`; want `\"active\"`"]
    );
    let problems = tag_problems(|live| {
        live["conditions"]["ref_name"]["exclude"] = serde_json::json!(["refs/tags/v9*"]);
    });
    assert_eq!(problems.len(), 1, "{problems:?}");
    assert!(
        problems[0].contains("which is not declared"),
        "{problems:?}"
    );
}

#[test]
fn the_tag_ruleset_is_found_by_name() {
    let mut responses = Memory::load("complete");
    responses.0.insert(
        tags::RULESETS_FILE.to_owned(),
        r#"[{"id":1,"name":"release-tags","source_type":"Organization"}]"#.to_owned(),
    );
    let declared = read_json(&declared().join(tags::TAG_RULESET)).unwrap();
    assert_eq!(
        tags::check_ruleset(&declared, &mut responses),
        ["the repository has no ruleset named `release-tags`"]
    );
    responses.0.insert(
        tags::RULESETS_FILE.to_owned(),
        r#"[{"id":1,"name":"release-tags","source_type":"Repository"},
            {"id":2,"name":"release-tags","source_type":"Repository"}]"#
            .to_owned(),
    );
    assert_eq!(
        tags::check_ruleset(&declared, &mut responses),
        [
            "the repository has 2 rulesets named `release-tags`, so the declared one cannot be \
             told apart"
        ]
    );
}

#[test]
fn a_tag_ruleset_this_check_cannot_read_is_refused() {
    let base = read_json(&declared().join(tags::TAG_RULESET)).unwrap();
    for (edit, why) in [
        (
            serde_json::json!({"enforcement": "disabled"}),
            "its `enforcement` is not `active`, so it would enforce nothing",
        ),
        (
            serde_json::json!({"name": ""}),
            "it has no `name` to find it by",
        ),
        (
            serde_json::json!({"conditions": {"ref_name": {"include": ["refs/heads/v*"]}}}),
            "the pattern `refs/heads/v*` is not a `refs/tags/` pattern",
        ),
        (serde_json::json!({"rules": []}), "it has no `rules`"),
        (
            serde_json::json!({"bypass_actors": [{"actor_id": "x"}]}),
            "the bypass actor `{\"actor_id\":\"x\"}` lacks a numeric `actor_id`, an \
             `actor_type` or a `bypass_mode`",
        ),
    ] {
        let mut ruleset = base.clone();
        for (key, value) in edit.as_object().unwrap() {
            ruleset[key] = value.clone();
        }
        let problems = tags::check_ruleset(&ruleset, &mut Memory::load("complete"));
        assert_eq!(problems, [why]);
    }
}

#[test]
fn the_tag_ruleset_protects_v_tags_from_everyone_but_the_release_app() {
    // RFC-0062 §8.3: creations, updates and deletions of `v*` tags restricted, one bypass actor.
    let ruleset = read_json(&declared().join(tags::TAG_RULESET)).unwrap();
    assert_eq!(ruleset["target"], "tag");
    assert_eq!(
        ruleset["conditions"]["ref_name"]["include"],
        serde_json::json!(["refs/tags/v*"])
    );
    let rules: Vec<&str> = ruleset["rules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|rule| rule["type"].as_str().unwrap())
        .collect();
    assert_eq!(rules, ["creation", "update", "deletion"]);
    let actors = ruleset["bypass_actors"].as_array().unwrap();
    assert_eq!(actors.len(), 1);
    assert_eq!(actors[0]["actor_type"], "Integration");
    assert_eq!(actors[0]["bypass_mode"], "always");
}

#[test]
fn comparison_rules() {
    let run = |want: &str, live: &str| {
        let mut found = Vec::new();
        compare(
            "",
            &serde_json::from_str(want).unwrap(),
            Some(&serde_json::from_str(live).unwrap()),
            &mut found,
        );
        found
    };
    // Extra members and extra objects in a list pass.
    assert!(run(r#"{"a":1}"#, r#"{"a":1,"b":2}"#).is_empty());
    assert!(run(r#"[{"type":"x"}]"#, r#"[{"type":"y"},{"type":"x","p":1}]"#).is_empty());
    // A list of names must match exactly, in any order.
    assert!(run(r#"["a","b"]"#, r#"["b","a"]"#).is_empty());
    assert_eq!(
        run(r#"["a"]"#, r#"["a","b"]"#),
        ["the response has `\"b\"`, which is not declared"]
    );
    assert_eq!(
        run(r#"[]"#, r#"["a"]"#),
        ["the response has `\"a\"`, which is not declared"]
    );
    assert_eq!(run(r#"["a"]"#, r#"[]"#), ["the response lacks `\"a\"`"]);
    // Null reads as missing, and types must agree.
    assert_eq!(
        run(r#"{"a":1}"#, r#"{"a":null}"#),
        ["`a` is missing; want `1`"]
    );
    assert_eq!(
        run(r#"{"a":{"b":1}}"#, r#"{"a":[]}"#),
        ["`a` is `[]`, not an object"]
    );
    assert_eq!(
        run(r#"{"a":[1]}"#, r#"{"a":{}}"#),
        ["`a` is `{}`, not a list"]
    );
    // An object item with no identity key is shown whole.
    assert_eq!(
        run(r#"[{"p":1}]"#, r#"[]"#),
        ["the response: `{\"p\":1}` is missing"]
    );
}

#[test]
fn repository_names_are_checked_before_they_reach_a_url() {
    for good in ["syntropyai-hq/syntropy", "a/b.c", "a_b/c-d"] {
        assert_eq!(checked_repo(good.to_owned()).unwrap(), good);
    }
    for bad in [
        "", "a", "a/", "/b", "a/b/c", "a/..", "a b/c", "a/b?x=1", "../b",
    ] {
        assert!(
            checked_repo(bad.to_owned()).is_err(),
            "`{bad}` was accepted"
        );
    }
}

#[test]
fn recorded_responses_cannot_be_saved_again() {
    let line: Vec<String> = ["ci", "repo-settings", "--from", "x", "--save", "y"]
        .map(String::from)
        .into();
    let Ok(Cli::Run { args, .. }) = cli::parse(&line, registry::JOBS, registry::SUITES) else {
        panic!("did not parse");
    };
    let context = Context {
        root: declared().parent().unwrap().parent().unwrap().to_path_buf(),
        args,
    };
    assert_eq!(
        step(&context).unwrap_err(),
        "`--from` reads recorded responses, so `--save` has none to save"
    );
}

#[test]
fn the_declared_state_meets_the_plan() {
    // The values the plan names, so a change to a declared file that drops one fails here too.
    let read = |file: &str| read_json(&declared().join(file)).unwrap();
    let repo = read("repository.json");
    assert_eq!(repo["allow_merge_commit"], false);
    assert_eq!(repo["allow_rebase_merge"], false);
    assert_eq!(repo["squash_merge_commit_title"], "PR_TITLE");
    assert_eq!(
        repo["security_and_analysis"]["secret_scanning"]["status"],
        "enabled"
    );
    assert_eq!(
        repo["security_and_analysis"]["secret_scanning_push_protection"]["status"],
        "enabled"
    );
    assert_eq!(
        read("private-vulnerability-reporting.json")["enabled"],
        true
    );
    assert_eq!(
        read("actions-permissions.json")["sha_pinning_required"],
        true
    );
    assert_eq!(
        read("fork-pr-contributor-approval.json")["approval_policy"],
        "all_external_contributors"
    );
    let main = read("ruleset-main.json");
    assert_eq!(main["bypass_actors"], Value::Array(Vec::new()));
    let types: Vec<&str> = main["rules"]
        .as_array()
        .unwrap()
        .iter()
        .map(|rule| rule["type"].as_str().unwrap())
        .collect();
    for needed in ["deletion", "non_fast_forward", "required_status_checks"] {
        assert!(types.contains(&needed), "ruleset-main.json lacks {needed}");
    }
    let checks: Vec<(&str, u64)> = main["rules"]
        .as_array()
        .unwrap()
        .iter()
        .find(|rule| rule["type"] == "required_status_checks")
        .unwrap()["parameters"]["required_status_checks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["context"].as_str().unwrap(),
                c["integration_id"].as_u64().unwrap(),
            )
        })
        .collect();
    // 15368 is the GitHub Actions app, so only a workflow job can satisfy each check.
    assert_eq!(
        checks,
        [
            ("ci-gate", 15368),
            ("dco", 15368),
            ("pr-title", 15368),
            ("dep-diff", 15368),
            ("properties-removal", 15368)
        ]
    );
    let release = read("ruleset-release.json");
    assert_eq!(
        release["conditions"]["ref_name"]["include"],
        serde_json::json!(["refs/heads/release/*"])
    );
    assert_eq!(release["bypass_actors"], Value::Array(Vec::new()));
}
