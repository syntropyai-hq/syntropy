// SPDX-License-Identifier: MIT OR Apache-2.0

//! The contribution checks of RFC-0062 §8.3 and §12.3, over recorded GitHub REST payloads.
//!
//! - `dco`: every commit carries a `Signed-off-by` that matches its author (ADR-0018 §2). It is
//!   one rule for every commit: nothing here looks at who or what wrote the commit.
//! - `pr-title`: the title is `type(scope): subject` or `type: subject`, with an allowed type and,
//!   when there is one, an allowed scope. The acceptance criteria use `feat!: x`, so the scope is
//!   optional (as in conventional commits), but a scope that is given must be known. A breaking
//!   marker, `!` in the title or in any commit's subject, or a `BREAKING CHANGE:` footer in the
//!   PR body or a commit message, needs the `semver-major-approved` label. The label counts only
//!   when a maintainer applied it: an agent identity, an App or any other account applying it
//!   does not count. A PR of one commit must also have a valid title as that commit's subject,
//!   because with the squash setting `COMMIT_OR_PR_TITLE` such a PR lands on `main` under it.
//!
//! Inputs are the payloads the REST API returns, saved to files:
//! - the PR: the `pull_request` event payload (`$GITHUB_EVENT_PATH`), or `GET /pulls/{n}`;
//! - its commits: `GET /pulls/{n}/commits`;
//! - its label history: `GET /issues/{n}/events`.
//!
//! A file may hold several JSON arrays one after another, as `gh api --paginate` prints pages,
//! and their items are read as one list. The label history must be recorded with
//! `gh api --include --paginate`: nothing else shows that its last page was fetched. A commits
//! list is shown complete by the PR's commit count.

use std::path::Path;

use serde_json::Value;

use crate::codeowners::ROSTER;
use crate::roster::Roster;
use crate::run::Context;

/// The commit types RFC-0062 §8.3 allows.
pub const TYPES: &[&str] = &[
    "feat", "fix", "perf", "refactor", "docs", "test", "build", "ci", "chore", "revert",
];

/// Area scopes: the domains of the code (AGENTS.md, "Writing style").
pub const AREAS: &[&str] = &[
    "placement",
    "tiers",
    "distributed",
    "compute",
    "product",
    "quality",
];

/// Scopes that are neither a crate nor an area: the design documents, dependency updates and
/// release PRs.
pub const OTHER_SCOPES: &[&str] = &["plan", "deps", "release"];

/// The label a breaking marker needs (RFC-0062 §8.3).
pub const BREAKING_LABEL: &str = "semver-major-approved";

/// A commit as `GET /pulls/{n}/commits` returns it, reduced to what `dco` reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commit {
    pub sha: String,
    pub author_name: String,
    pub author_email: String,
    pub message: String,
}

/// A pull request, from an event payload or `GET /pulls/{n}`.
#[derive(Debug, PartialEq, Eq)]
pub struct PullRequest {
    pub title: String,
    pub body: String,
    pub labels: Vec<String>,
    /// How many commits GitHub says the PR has. The commits file must hold exactly this many.
    pub commits: u64,
}

/// `GET /pulls/{n}/commits` lists at most this many commits, however it is paginated.
pub const API_COMMIT_LIMIT: u64 = 250;

/// The commits file must hold every commit of the PR, or a check would pass on a partial list:
/// a missing `--paginate` gives 30, and the API never gives more than 250.
pub fn check_commit_count(pr: &PullRequest, commits: &[Commit]) -> Result<(), String> {
    if pr.commits >= API_COMMIT_LIMIT {
        return Err(format!(
            "the PR has {} commits; GET /pulls/{{n}}/commits lists at most {API_COMMIT_LIMIT}, \
             so the list cannot be checked complete",
            pr.commits
        ));
    }
    if commits.len() as u64 != pr.commits {
        return Err(format!(
            "the commits file holds {} commits, but the PR has {}; fetch every page \
             (`gh api --paginate`)",
            commits.len(),
            pr.commits
        ));
    }
    Ok(())
}

/// A `labeled` or `unlabeled` issue event.
#[derive(Debug, PartialEq, Eq)]
pub struct LabelEvent {
    pub applied: bool,
    pub label: String,
    pub actor: String,
    /// The event was performed through a GitHub App, so no person applied it directly.
    pub via_app: bool,
    pub created_at: String,
}

// ---- dco ----

/// Every commit's DCO problem, one line each. No commits at all is a problem too: a pull request
/// always has one, so an empty list means the input is wrong.
pub fn dco(commits: &[Commit]) -> Vec<String> {
    if commits.is_empty() {
        return vec!["no commits to check; the commits file is empty".into()];
    }
    let mut problems = Vec::new();
    for commit in commits {
        let short = short(&commit.sha);
        let author = format!("{} <{}>", commit.author_name, commit.author_email);
        let sign_offs = sign_offs(&commit.message);
        if sign_offs.is_empty() {
            problems.push(format!(
                "commit {short} by {author} has no `Signed-off-by:` line"
            ));
        } else if !sign_offs.iter().any(|(name, email)| {
            name.trim().eq_ignore_ascii_case(commit.author_name.trim())
                && email
                    .trim()
                    .eq_ignore_ascii_case(commit.author_email.trim())
        }) {
            let found: Vec<String> = sign_offs
                .iter()
                .map(|(name, email)| format!("{name} <{email}>"))
                .collect();
            problems.push(format!(
                "commit {short} is signed off by {}, which does not match its author {author}",
                found.join(", ")
            ));
        }
    }
    problems
}

/// The `Signed-off-by: Name <email>` lines of a commit message, as (name, email).
fn sign_offs(message: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for line in message.lines() {
        let Some((key, value)) = line.trim().split_once(':') else {
            continue;
        };
        if !key.trim().eq_ignore_ascii_case("signed-off-by") {
            continue;
        }
        let value = value.trim();
        if let Some((name, rest)) = value.rsplit_once('<')
            && let Some(email) = rest.strip_suffix('>')
            && !name.trim().is_empty()
            && !email.trim().is_empty()
        {
            found.push((name.trim().to_owned(), email.trim().to_owned()));
        }
    }
    found
}

// ---- pr-title ----

/// The parts of a valid title.
#[derive(Debug, PartialEq, Eq)]
pub struct Title<'t> {
    pub kind: &'t str,
    pub scope: Option<&'t str>,
    pub breaking: bool,
}

/// Parses `type(scope)!: subject`, where `(scope)` and `!` are optional, and checks the type and
/// the scope.
pub fn parse_title<'t>(title: &'t str, scopes: &[String]) -> Result<Title<'t>, String> {
    const FORM: &str = "the PR title must be `type(scope): subject` or `type: subject`";
    let Some((head, subject)) = title.split_once(": ") else {
        return Err(format!("{FORM}; `{title}` has no `: ` after the type"));
    };
    if subject.trim().is_empty() || subject.starts_with(char::is_whitespace) {
        return Err(format!(
            "{FORM}; the subject after `: ` is empty or starts with a space"
        ));
    }
    if subject.ends_with(char::is_whitespace) {
        return Err(format!("{FORM}; the subject ends with a space"));
    }
    let (head, breaking) = match head.strip_suffix('!') {
        Some(head) => (head, true),
        None => (head, false),
    };
    let (kind, scope) = match head.split_once('(') {
        None => (head, None),
        Some((kind, rest)) => match rest.strip_suffix(')') {
            Some(scope) => (kind, Some(scope)),
            None => return Err(format!("{FORM}; `{head}` does not close its `(scope)`")),
        },
    };
    if !TYPES.contains(&kind) {
        return Err(format!(
            "`{kind}` is not an allowed type; allowed types: {}",
            TYPES.join(", ")
        ));
    }
    if let Some(scope) = scope
        && !scopes.iter().any(|s| s == scope)
    {
        return Err(format!(
            "`{scope}` is not an allowed scope; allowed scopes: {}",
            scopes.join(", ")
        ));
    }
    Ok(Title {
        kind,
        scope,
        breaking,
    })
}

/// `type!:` or `type(scope)!:`: a `!` just before the first `:`.
pub fn has_breaking_bang(title: &str) -> bool {
    title
        .split_once(':')
        .is_some_and(|(head, _)| head.ends_with('!'))
}

/// A commit message's first line.
fn subject(message: &str) -> &str {
    message.lines().next().unwrap_or("")
}

/// The first 12 characters of a commit SHA, for messages.
fn short(sha: &str) -> &str {
    &sha[..sha.len().min(12)]
}

/// A `BREAKING CHANGE:` or `BREAKING-CHANGE:` footer line, as conventional commits define them.
pub fn has_breaking_footer(text: &str) -> bool {
    text.lines()
        .any(|line| line.starts_with("BREAKING CHANGE:") || line.starts_with("BREAKING-CHANGE:"))
}

/// Whether `label` is on the PR because a maintainer applied it. `Ok` names the maintainer;
/// `Err` says why the label does not count. Events are replayed in time order: the label's state
/// is set by its last `labeled` or `unlabeled` event.
pub fn applied_by_maintainer(
    label: &str,
    current: &[String],
    events: &[LabelEvent],
    roster: &Roster,
) -> Result<String, String> {
    if !current.iter().any(|l| l == label) {
        return Err(format!("the PR does not carry `{label}`"));
    }
    let mut ordered: Vec<&LabelEvent> = events.iter().filter(|e| e.label == label).collect();
    ordered.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    match ordered.last() {
        None => Err(format!(
            "the label history has no event applying `{label}`, so who applied it is unknown"
        )),
        Some(last) if !last.applied => Err(format!(
            "the label history ends with `{label}` removed by `{}`",
            last.actor
        )),
        Some(last) if last.via_app => Err(format!(
            "`{label}` was applied by `{}` through a GitHub App; only a maintainer's own account counts",
            last.actor
        )),
        Some(last) if roster.is_agent(&last.actor) => Err(format!(
            "`{label}` was applied by `{}`, an agent identity; only a maintainer counts",
            last.actor
        )),
        Some(last) if !roster.is_maintainer(&last.actor) => Err(format!(
            "`{label}` was applied by `{}`, who is not a maintainer in {ROSTER}",
            last.actor
        )),
        Some(last) => Ok(last.actor.clone()),
    }
}

/// Every `pr-title` problem.
pub fn pr_title(
    pr: &PullRequest,
    commits: &[Commit],
    events: &[LabelEvent],
    roster: &Roster,
    scopes: &[String],
) -> Vec<String> {
    let mut problems = Vec::new();
    if let Err(problem) = parse_title(&pr.title, scopes) {
        problems.push(problem);
    }
    // With the squash title setting `COMMIT_OR_PR_TITLE`, a PR of one commit lands on `main`
    // under that commit's subject, not the PR title, so the subject must pass as a title too.
    if let [only] = commits
        && let Err(problem) = parse_title(subject(&only.message), scopes)
    {
        problems.push(format!(
            "the PR's only commit, {}, lands on `main` under its own subject, so the subject \
             must follow the title rule too: {problem}",
            short(&only.sha)
        ));
    }
    let mut markers = Vec::new();
    // A malformed title or subject is still read for a marker, so it cannot hide one.
    if has_breaking_bang(&pr.title) {
        markers.push("`!` in the title".to_owned());
    }
    if has_breaking_footer(&pr.body) {
        markers.push("a `BREAKING CHANGE:` footer in the PR body".to_owned());
    }
    for commit in commits {
        if has_breaking_bang(subject(&commit.message)) {
            markers.push(format!(
                "`!` in the subject of commit {}",
                short(&commit.sha)
            ));
        }
        if has_breaking_footer(&commit.message) {
            markers.push(format!(
                "a `BREAKING CHANGE:` footer in commit {}",
                short(&commit.sha)
            ));
        }
    }
    if !markers.is_empty()
        && let Err(why) = applied_by_maintainer(BREAKING_LABEL, &pr.labels, events, roster)
    {
        problems.push(format!(
            "the PR is marked breaking ({}), which needs `{BREAKING_LABEL}` applied by a maintainer: {why}",
            markers.join("; ")
        ));
    }
    problems
}

/// The allowed scopes: every crate under `crates/` and `xtask`, the areas, and the others.
pub fn scopes(root: &Path) -> Result<Vec<String>, String> {
    let crates = root.join("crates");
    let entries = std::fs::read_dir(&crates).map_err(|e| format!("{}: {e}", crates.display()))?;
    let mut scopes = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", crates.display()))?;
        if entry.path().join("Cargo.toml").is_file()
            && let Some(name) = entry.file_name().to_str()
        {
            scopes.push(name.to_owned());
        }
    }
    scopes.sort();
    scopes.push("xtask".into());
    scopes.extend(AREAS.iter().chain(OTHER_SCOPES).map(|s| (*s).to_owned()));
    Ok(scopes)
}

// ---- reading payloads ----

/// One HTTP response that `gh api --include` recorded: its status, whether its `Link` header
/// points to a next page, and its body.
struct Page {
    status: u16,
    next: bool,
    body: String,
}

/// The pages of a file `gh api --include` wrote, or `None` for a file of bare JSON. Each page is
/// an `HTTP/` status line, headers, a blank line and the body.
fn recorded_pages(text: &str, what: &str) -> Result<Option<Vec<Page>>, String> {
    if !text.trim_start().starts_with("HTTP/") {
        return Ok(None);
    }
    let mut pages = Vec::new();
    let mut lines = text
        .lines()
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .peekable();
    while let Some(status_line) = lines.next() {
        if status_line.trim().is_empty() {
            continue;
        }
        let number = pages.len() + 1;
        let status = status_line
            .strip_prefix("HTTP/")
            .and_then(|rest| rest.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok())
            .ok_or_else(|| format!("{what}: page {number} has no readable `HTTP/` status line"))?;
        let mut next = false;
        loop {
            match lines.next() {
                None => return Err(format!("{what}: page {number} ends inside its headers")),
                Some("") => break,
                Some(header) => {
                    if let Some((name, value)) = header.split_once(':')
                        && name.trim().eq_ignore_ascii_case("link")
                    {
                        next = value.split(',').any(|link| link.contains("rel=\"next\""));
                    }
                }
            }
        }
        let mut body = String::new();
        while let Some(line) = lines.next_if(|line| !line.starts_with("HTTP/")) {
            body.push_str(line);
            body.push('\n');
        }
        pages.push(Page { status, next, body });
    }
    Ok(Some(pages))
}

/// The items of a list endpoint's response. A recorded file must show every page: each one a
/// success, the last with no `rel="next"`. `require_recorded` refuses bare JSON, for a list whose
/// completeness nothing else can show.
fn list_items(text: &str, what: &str, require_recorded: bool) -> Result<Vec<Value>, String> {
    let Some(pages) = recorded_pages(text, what)? else {
        if require_recorded {
            return Err(format!(
                "{what}: record the list with `gh api --include --paginate`, so the check can \
                 see that its last page was fetched"
            ));
        }
        return array_items(text, what);
    };
    let Some(last) = pages.last() else {
        return Err(format!("{what}: no pages"));
    };
    if last.next {
        return Err(format!(
            "{what}: the last recorded page links to a next one (`rel=\"next\"`), so the list \
             is incomplete; fetch every page with `--paginate`"
        ));
    }
    let mut items = Vec::new();
    for (i, page) in pages.iter().enumerate() {
        if !(200..300).contains(&page.status) {
            return Err(format!(
                "{what}: page {} has HTTP status {}",
                i + 1,
                page.status
            ));
        }
        items.extend(array_items(&page.body, what)?);
    }
    Ok(items)
}

/// The items of every JSON array in `text`, in order. A file may hold several arrays, one per
/// page, as `gh api --paginate` prints them.
fn array_items(text: &str, what: &str) -> Result<Vec<Value>, String> {
    let mut items = Vec::new();
    for value in serde_json::Deserializer::from_str(text).into_iter::<Value>() {
        match value.map_err(|e| format!("{what}: not JSON: {e}"))? {
            Value::Array(page) => items.extend(page),
            other => {
                return Err(format!(
                    "{what}: expected a JSON array, got {}",
                    kind_of(&other)
                ));
            }
        }
    }
    Ok(items)
}

fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// The string at `path` in `value`, such as `commit.author.name`.
fn string_at<'v>(value: &'v Value, path: &str, what: &str) -> Result<&'v str, String> {
    let mut at = value;
    for key in path.split('.') {
        at = at
            .get(key)
            .ok_or_else(|| format!("{what}: `{path}` is missing"))?;
    }
    at.as_str()
        .ok_or_else(|| format!("{what}: `{path}` is {}, not a string", kind_of(at)))
}

/// A commits list, bare or recorded with `--include`. `check_commit_count` shows it complete.
pub fn parse_commits(text: &str) -> Result<Vec<Commit>, String> {
    list_items(text, "commits", false)?
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let what = format!("commits[{i}]");
            Ok(Commit {
                sha: string_at(item, "sha", &what)?.to_owned(),
                author_name: string_at(item, "commit.author.name", &what)?.to_owned(),
                author_email: string_at(item, "commit.author.email", &what)?.to_owned(),
                message: string_at(item, "commit.message", &what)?.to_owned(),
            })
        })
        .collect()
}

/// A `pull_request` event payload, or the PR object itself.
pub fn parse_pull_request(text: &str) -> Result<PullRequest, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| format!("event: not JSON: {e}"))?;
    let pr = value.get("pull_request").unwrap_or(&value);
    let labels = pr
        .get("labels")
        .and_then(Value::as_array)
        .ok_or("event: `labels` is missing or not an array")?
        .iter()
        .enumerate()
        .map(|(i, label)| {
            string_at(label, "name", &format!("event: labels[{i}]")).map(str::to_owned)
        })
        .collect::<Result<_, _>>()?;
    Ok(PullRequest {
        title: string_at(pr, "title", "event")?.to_owned(),
        // GitHub sends `null` for an empty body.
        body: pr
            .get("body")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        labels,
        commits: pr
            .get("commits")
            .and_then(Value::as_u64)
            .ok_or("event: `commits`, the PR's commit count, is missing or not a number")?,
    })
}

/// The `labeled` and `unlabeled` events of `GET /issues/{n}/events`; other events are skipped.
pub fn parse_label_events(text: &str) -> Result<Vec<LabelEvent>, String> {
    let mut events = Vec::new();
    // Nothing counts a PR's events, so only the recorded `Link` headers can show the history
    // complete; a truncated history could otherwise end on an earlier maintainer's label.
    for (i, item) in list_items(text, "label events", true)?.iter().enumerate() {
        let what = format!("label events[{i}]");
        let applied = match string_at(item, "event", &what)? {
            "labeled" => true,
            "unlabeled" => false,
            _ => continue,
        };
        events.push(LabelEvent {
            applied,
            label: string_at(item, "label.name", &what)?.to_owned(),
            actor: string_at(item, "actor.login", &what)?.to_owned(),
            via_app: item
                .get("performed_via_github_app")
                .is_some_and(|app| !app.is_null()),
            created_at: string_at(item, "created_at", &what)?.to_owned(),
        });
    }
    Ok(events)
}

// ---- the job steps ----

fn read_arg(context: &Context, option: &str) -> Result<String, String> {
    let path = context
        .args
        .get(option)
        .ok_or_else(|| format!("`{option}` is required"))?;
    std::fs::read_to_string(path).map_err(|e| format!("{option} {path}: {e}"))
}

fn report(problems: Vec<String>) -> Result<(), String> {
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// `cargo xtask ci dco --event <file> --commits <file>`.
pub fn dco_step(context: &Context) -> Result<(), String> {
    let pr = parse_pull_request(&read_arg(context, "--event")?)?;
    let commits = parse_commits(&read_arg(context, "--commits")?)?;
    check_commit_count(&pr, &commits)?;
    report(dco(&commits))
}

/// `cargo xtask ci pr-title --event <file> --commits <file> --label-events <file>
/// --roster <file>`. The roster decides whose label counts, so the caller names a trusted copy,
/// such as the base branch's, never one the PR under check could have changed.
pub fn pr_title_step(context: &Context) -> Result<(), String> {
    let pr = parse_pull_request(&read_arg(context, "--event")?)?;
    let commits = parse_commits(&read_arg(context, "--commits")?)?;
    check_commit_count(&pr, &commits)?;
    let events = parse_label_events(&read_arg(context, "--label-events")?)?;
    let roster =
        Roster::parse(&read_arg(context, "--roster")?).map_err(|e| format!("roster: {e}"))?;
    let scopes = scopes(&context.root)?;
    report(pr_title(&pr, &commits, &events, &roster, &scopes))
}

#[cfg(test)]
mod tests;
