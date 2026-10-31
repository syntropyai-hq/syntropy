// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tag rulesets, and `cargo xtask ci tag-ruleset`: the weekly check that the `v*` tag ruleset is
//! in force and that the release App is its only bypass actor (QUAL-E17-T01; RFC-0062 §8.3 and
//! "Testing and validation" item 9; ADR-0621 decision 2).
//!
//! GitHub has no endpoint that lists the rules in force on a tag, as `/rules/branches/{branch}`
//! does for a branch. A tag ruleset is therefore read whole: `/rulesets?includes_parents=false`
//! names the repository's rulesets, the one with the declared `name` is found there, and
//! `/rulesets/{id}` returns it. The declared `target`, `enforcement`, `conditions` and `rules`
//! must be in it, compared as [`super::check`] compares a setting, and its `bypass_actors` must
//! be exactly the declared ones: an extra actor is drift, because it could create, move or delete
//! a release tag.
//!
//! GitHub returns `bypass_actors` only to a caller that can edit the ruleset, which for a GitHub
//! App means the repository's Administration write permission. With less, the field is absent and
//! the check fails, saying so: it never passes a bypass list it could not read.
//!
//! The declared file names the release App by its numeric App ID. Until a maintainer records it,
//! the ID is `0`, which no live actor has, so the check fails and says what to record.

use serde_json::{Map, Value};

use super::{Gh, Recorded, Responses, compare, json_pages, read_json, repository};
use crate::run::Context;

/// The declared `v*` tag ruleset, under [`super::DECLARED`].
pub const TAG_RULESET: &str = "ruleset-tags.json";

/// What the tag ruleset is required by.
pub const REQUIRED_BY: &str = "RFC-0062 §8.3, §12.6; ADR-0621 decision 2";

/// The response file for the repository's list of rulesets.
pub const RULESETS_FILE: &str = "rulesets.json";

/// The response file for one ruleset, by its ID.
pub fn ruleset_file(id: u64) -> String {
    format!("ruleset-{id}.json")
}

/// The members of a declared tag ruleset that the live one must contain.
const COMPARED: &[&str] = &["target", "enforcement", "conditions", "rules"];

/// Whether a declared ruleset targets tags, so this module checks it.
pub fn is_tag_ruleset(ruleset: &Value) -> bool {
    ruleset.get("target").and_then(Value::as_str) == Some("tag")
}

/// A declared tag ruleset's name and bypass actors, or why it is not one this check can read.
fn declared_parts(ruleset: &Value) -> Result<(&str, &[Value]), String> {
    if !is_tag_ruleset(ruleset) {
        return Err("its `target` is not `tag`".to_owned());
    }
    if ruleset.get("enforcement").and_then(Value::as_str) != Some("active") {
        return Err("its `enforcement` is not `active`, so it would enforce nothing".to_owned());
    }
    let name = ruleset
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "it has no `name` to find it by".to_owned())?;
    let patterns: Vec<&str> = ruleset
        .pointer("/conditions/ref_name/include")
        .and_then(Value::as_array)
        .map(|include| include.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if patterns.is_empty() {
        return Err("it names no tag in `conditions.ref_name.include`".to_owned());
    }
    if let Some(pattern) = patterns.iter().find(|p| !p.starts_with("refs/tags/")) {
        return Err(format!(
            "the pattern `{pattern}` is not a `refs/tags/` pattern"
        ));
    }
    if ruleset
        .get("rules")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        return Err("it has no `rules`".to_owned());
    }
    let actors = ruleset
        .get("bypass_actors")
        .and_then(Value::as_array)
        .ok_or_else(|| "it has no `bypass_actors` list".to_owned())?;
    for actor in actors {
        if actor.get("actor_id").and_then(Value::as_u64).is_none()
            || actor.get("actor_type").and_then(Value::as_str).is_none()
            || actor.get("bypass_mode").and_then(Value::as_str).is_none()
        {
            return Err(format!(
                "the bypass actor `{actor}` lacks a numeric `actor_id`, an `actor_type` or a \
                 `bypass_mode`"
            ));
        }
    }
    Ok((name, actors))
}

/// A bypass actor as the check compares it: its ID, type and mode.
fn actor_key(actor: &Value) -> (Option<u64>, Option<&str>, Option<&str>) {
    (
        actor.get("actor_id").and_then(Value::as_u64),
        actor.get("actor_type").and_then(Value::as_str),
        actor.get("bypass_mode").and_then(Value::as_str),
    )
}

fn shown_actor(actor: &Value) -> String {
    match actor_key(actor) {
        (Some(id), Some(kind), Some(mode)) => format!("{kind} {id} ({mode})"),
        _ => format!("`{actor}`"),
    }
}

/// Every way the live `bypass_actors` differ from the declared ones.
fn compare_actors(declared: &[Value], live: Option<&Value>) -> Vec<String> {
    let Some(live) = live else {
        return vec![
            "the response does not show `bypass_actors`: GitHub shows them only to a token that \
             can edit the ruleset (Administration write), so they could not be checked"
                .to_owned(),
        ];
    };
    let Some(live) = live.as_array() else {
        return vec![format!("`bypass_actors` is `{live}`, not a list")];
    };
    let mut problems = Vec::new();
    let declared_keys: Vec<_> = declared.iter().map(actor_key).collect();
    let live_keys: Vec<_> = live.iter().map(actor_key).collect();
    for actor in declared {
        if !live_keys.contains(&actor_key(actor)) {
            let mut problem = format!("`bypass_actors` lacks {}", shown_actor(actor));
            if actor_key(actor).0 == Some(0) {
                problem.push_str(
                    "; the declared ID 0 stands for the release App until a maintainer records \
                     its App ID in the file",
                );
            }
            problems.push(problem);
        }
    }
    for actor in live {
        if !declared_keys.contains(&actor_key(actor)) {
            problems.push(format!(
                "`bypass_actors` has {}, which is not declared: only the release App may \
                 bypass the tag ruleset (RFC-0062 §8.3)",
                shown_actor(actor)
            ));
        }
    }
    problems
}

/// Checks one declared tag ruleset against `responses`. Returns every problem found.
pub fn check_ruleset(ruleset: &Value, responses: &mut dyn Responses) -> Vec<String> {
    let (name, actors) = match declared_parts(ruleset) {
        Ok(parts) => parts,
        Err(error) => return vec![error],
    };
    let listed = match responses
        .get("/rulesets?includes_parents=false", RULESETS_FILE, true)
        .and_then(|text| json_pages(&text))
    {
        Ok(listed) => listed,
        Err(error) => return vec![error],
    };
    let ids: Vec<u64> = listed
        .iter()
        .filter(|item| {
            item.get("name").and_then(Value::as_str) == Some(name)
                && item.get("source_type").and_then(Value::as_str) == Some("Repository")
        })
        .filter_map(|item| item.get("id").and_then(Value::as_u64))
        .collect();
    let id = match ids.as_slice() {
        [id] => *id,
        [] => return vec![format!("the repository has no ruleset named `{name}`")],
        _ => {
            return vec![format!(
                "the repository has {} rulesets named `{name}`, so the declared one cannot be \
                 told apart",
                ids.len()
            )];
        }
    };
    let live = match responses
        .get(&format!("/rulesets/{id}"), &ruleset_file(id), false)
        .and_then(|text| {
            serde_json::from_str::<Value>(&text)
                .map_err(|error| format!("the response is not JSON: {error}"))
        }) {
        Ok(live) => live,
        Err(error) => return vec![error],
    };
    let want: Map<String, Value> = COMPARED
        .iter()
        .filter_map(|key| ruleset.get(*key).map(|v| ((*key).to_owned(), v.clone())))
        .collect();
    let mut problems = Vec::new();
    compare(
        &format!("ruleset `{name}`"),
        &Value::Object(want),
        Some(&live),
        &mut problems,
    );
    problems.extend(
        compare_actors(actors, live.get("bypass_actors"))
            .into_iter()
            .map(|p| format!("ruleset `{name}`: {p}")),
    );
    problems
}

/// `cargo xtask ci tag-ruleset [--repo <owner/name>] [--from <dir>]`.
pub fn step(context: &Context) -> Result<(), String> {
    let mut responses: Box<dyn Responses> = match context.args.get("--from") {
        Some(dir) => Box::new(Recorded(dir.into())),
        None => {
            let repo = repository(context)?;
            eprintln!("tag-ruleset: reading {repo} through the REST API");
            Box::new(Gh { repo, save: None })
        }
    };
    let declared = context.root.join(super::DECLARED).join(TAG_RULESET);
    let label = format!("`{TAG_RULESET}` ({REQUIRED_BY})");
    let ruleset = read_json(&declared).map_err(|error| format!("{label}: {error}"))?;
    let problems = check_ruleset(&ruleset, responses.as_mut());
    if problems.is_empty() {
        eprintln!(
            "tag-ruleset: {TAG_RULESET}: its rules are in force and its only bypass actors are \
             the declared ones"
        );
        Ok(())
    } else {
        Err(problems
            .into_iter()
            .map(|p| format!("{label}: {p}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}
