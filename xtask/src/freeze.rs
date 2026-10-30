// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci freeze --events <file> [--pr <file>] [--release-app <login>]
//! [--durations-out <file>]`: the release freeze of RFC-0062 §9.2 step 3 (QUAL-E17-T04, ADR-0621
//! decision 4).
//!
//! `rc:start` sets the freeze on `main`. While it is set, a PR merges only if it carries the
//! `rc:fix` label, or if it is the release PR itself: §9.2 step 3 has the maintainer merge the
//! release PR under the freeze, and step 5 does the same for the `X.Y.(Z+1)` release PR, so
//! holding it would stop every release before its tag. A publish or an abort lifts the freeze. A failed RC does not lift it: the next patch's
//! `rc:fix` PRs merge under the same freeze (§9.2 step 5).
//!
//! This module is the rule, over data. `--events` is the freeze's event log, a JSON array in time
//! order of `{"at": <unix seconds>, "kind": "set" | "lift", "version": "X.Y.Z"}`; a `lift` also
//! has `"reason": "publish" | "abort"`. `--pr` is
//! `{"number": <n>, "base": "<branch>", "labels": ["rc:fix", ...]}`, with optional `"head"` (the
//! head branch) and `"author"` (the PR author's login). A base given as `refs/heads/<branch>`, as
//! a `merge_group` payload gives it, is read as `<branch>`. The freeze is on `main` only: a PR to
//! another branch, such as a backport to `release/vX.Y` (§9.2 step 8), is not held.
//!
//! The release PR is recognised by two properties of the PR payload, never by a label, since
//! anyone with triage can label: its head branch is release-please's
//! `release-please--branches--main`, and its author is the release App (§8.2), whose login
//! `--release-app` names. Without `--release-app` no PR is treated as the release PR.
//! With `--pr` the command fails when the freeze holds that PR. With `--durations-out` it writes
//! each freeze window as JSON, for the CI cost report (QUAL-E1-T13).
//!
//! What writes the log, and what calls this on a PR, is not here: `release.yml` and `rc:start`
//! are QUAL-E17-T05 and the merge-queue hold is QUAL-E17-T17. `ci-gate` does not consult it yet.
//!
//! The freeze targets 48 h `[target]`, RFC-0060's RC budget (RFC-0062 §9.2).

use serde_json::{Value, json};

use crate::run::Context;

/// The branch the freeze is on.
pub const FROZEN_BRANCH: &str = "main";

/// The label that lets a PR merge during a freeze.
pub const FIX_LABEL: &str = "rc:fix";

/// The head branch of release-please's release PR on [`FROZEN_BRANCH`].
pub const RELEASE_HEAD: &str = "release-please--branches--main";

/// Why a freeze was lifted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Publish,
    Abort,
}

impl Reason {
    fn as_str(self) -> &'static str {
        match self {
            Reason::Publish => "publish",
            Reason::Abort => "abort",
        }
    }
}

/// One entry of the freeze's event log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Set { at: u64, version: String },
    Lift { at: u64, reason: Reason },
}

impl Event {
    fn at(&self) -> u64 {
        match self {
            Event::Set { at, .. } | Event::Lift { at, .. } => *at,
        }
    }
}

/// One freeze, from `rc:start` to its lift. Times are unix seconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    /// The version of the `set` that started the window.
    pub version: String,
    /// The versions of later `set` events under the same freeze, one per patch RC.
    pub continued_by: Vec<String>,
    pub start: u64,
    /// `None` while the freeze is still set.
    pub end: Option<u64>,
    pub lifted_by: Option<Reason>,
}

impl Window {
    /// The window's length, once it has ended.
    pub fn seconds(&self) -> Option<u64> {
        self.end.map(|end| end - self.start)
    }

    fn to_json(&self) -> Value {
        json!({
            "version": self.version,
            "continued_by": self.continued_by,
            "start": self.start,
            "end": self.end,
            "seconds": self.seconds(),
            "lifted_by": self.lifted_by.map(Reason::as_str),
        })
    }
}

/// Reads the event log.
pub fn parse_events(text: &str) -> Result<Vec<Event>, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| format!("events: not JSON: {e}"))?;
    let items = value
        .as_array()
        .ok_or_else(|| "events: not a JSON array".to_owned())?;
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let at = item
                .get("at")
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("events[{i}]: `at` is not unix seconds"))?;
            match item.get("kind").and_then(Value::as_str) {
                Some("set") => {
                    let version = item
                        .get("version")
                        .and_then(Value::as_str)
                        .ok_or_else(|| format!("events[{i}]: a `set` has no `version`"))?;
                    Ok(Event::Set {
                        at,
                        version: version.to_owned(),
                    })
                }
                Some("lift") => match item.get("reason").and_then(Value::as_str) {
                    Some("publish") => Ok(Event::Lift {
                        at,
                        reason: Reason::Publish,
                    }),
                    Some("abort") => Ok(Event::Lift {
                        at,
                        reason: Reason::Abort,
                    }),
                    _ => Err(format!(
                        "events[{i}]: a `lift` needs `reason` publish or abort"
                    )),
                },
                _ => Err(format!("events[{i}]: `kind` is not `set` or `lift`")),
            }
        })
        .collect()
}

/// Every freeze window in `events`, in order. A `set` while frozen continues the window (the next
/// patch's RC runs under the same freeze). Events out of time order, or a `lift` with no freeze
/// set, are errors.
pub fn durations(events: &[Event]) -> Result<Vec<Window>, String> {
    let mut windows: Vec<Window> = Vec::new();
    let mut last = 0;
    for (i, event) in events.iter().enumerate() {
        if event.at() < last {
            return Err(format!("events[{i}]: earlier than the event before it"));
        }
        last = event.at();
        let open = windows.last_mut().filter(|w| w.end.is_none());
        match (event, open) {
            (Event::Set { at, version }, None) => windows.push(Window {
                version: version.clone(),
                continued_by: Vec::new(),
                start: *at,
                end: None,
                lifted_by: None,
            }),
            (Event::Set { version, .. }, Some(window)) => window.continued_by.push(version.clone()),
            (Event::Lift { at, reason }, Some(window)) => {
                window.end = Some(*at);
                window.lifted_by = Some(*reason);
            }
            (Event::Lift { .. }, None) => {
                return Err(format!("events[{i}]: a `lift` while no freeze is set"));
            }
        }
    }
    Ok(windows)
}

/// Whether the freeze is set after `events`.
pub fn is_frozen(events: &[Event]) -> Result<bool, String> {
    Ok(durations(events)?.last().is_some_and(|w| w.end.is_none()))
}

/// Whether `pr` is the release PR on [`FROZEN_BRANCH`]: release-please's head branch, opened by
/// the release App `release_app`. Both must match; with no `release_app` nothing does.
pub fn is_release_pr(pr: &Pr, release_app: Option<&str>) -> bool {
    let Some(app) = release_app.filter(|a| !a.is_empty()) else {
        return false;
    };
    pr.base == FROZEN_BRANCH
        && pr.head.as_deref() == Some(RELEASE_HEAD)
        && pr.author.as_deref() == Some(app)
}

/// The freeze's verdict on one PR: `Ok` if it may merge, else why it is held.
pub fn evaluate(frozen: bool, pr: &Pr, release_app: Option<&str>) -> Result<(), String> {
    if frozen
        && pr.base == FROZEN_BRANCH
        && !pr.labels.iter().any(|l| l == FIX_LABEL)
        && !is_release_pr(pr, release_app)
    {
        return Err(format!(
            "held: a release freeze is set on `{FROZEN_BRANCH}` and the PR is neither labelled \
             `{FIX_LABEL}` nor the release PR"
        ));
    }
    Ok(())
}

/// A PR as `--pr` gives it.
#[derive(Debug, PartialEq, Eq)]
pub struct Pr {
    pub number: u64,
    pub base: String,
    pub labels: Vec<String>,
    /// The head branch, if the file gives it.
    pub head: Option<String>,
    /// The author's login, if the file gives it.
    pub author: Option<String>,
}

/// Reads a PR file: `{"number": <n>, "base": "<branch>", "labels": [...]}`, with optional
/// `head` and `author` strings. A missing `base` is an error, so a PR is never let through for
/// want of it. A `refs/heads/` prefix on `base` is dropped, so a full ref compares as its branch.
pub fn parse_pr(text: &str) -> Result<Pr, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| format!("pr: not JSON: {e}"))?;
    let number = value
        .get("number")
        .and_then(Value::as_u64)
        .ok_or_else(|| "pr: no numeric `number`".to_owned())?;
    let base = value
        .get("base")
        .and_then(Value::as_str)
        .ok_or_else(|| "pr: no `base` branch".to_owned())?;
    let base = base.strip_prefix("refs/heads/").unwrap_or(base).to_owned();
    let optional = |key: &str| -> Result<Option<String>, String> {
        match value.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(_) => Err(format!("pr: `{key}` is not a string")),
        }
    };
    let head = optional("head")?;
    let author = optional("author")?;
    let labels = value
        .get("labels")
        .and_then(Value::as_array)
        .ok_or_else(|| "pr: no `labels` array".to_owned())?
        .iter()
        .map(|l| {
            l.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "pr: a label is not a string".to_owned())
        })
        .collect::<Result<_, _>>()?;
    Ok(Pr {
        number,
        base,
        labels,
        head,
        author,
    })
}

/// The `ci freeze` step.
pub fn step(context: &Context) -> Result<(), String> {
    let read = |option: &str| -> Result<String, String> {
        let path = context
            .args
            .get(option)
            .ok_or_else(|| format!("`{option}` is required"))?;
        std::fs::read_to_string(path).map_err(|e| format!("{option} {path}: {e}"))
    };
    let events = parse_events(&read("--events")?)?;
    let windows = durations(&events)?;
    if let Some(path) = context.args.get("--durations-out") {
        let out: Vec<Value> = windows.iter().map(Window::to_json).collect();
        std::fs::write(path, Value::Array(out).to_string())
            .map_err(|e| format!("--durations-out {path}: {e}"))?;
    }
    let frozen = is_frozen(&events)?;
    eprintln!("freeze: {}", if frozen { "set" } else { "not set" });
    if context.args.get("--pr").is_none() {
        return Ok(());
    }
    let pr = parse_pr(&read("--pr")?)?;
    let release_app = context.args.get("--release-app");
    let verdict = evaluate(frozen, &pr, release_app);
    let number = pr.number;
    match &verdict {
        Ok(()) if frozen && is_release_pr(&pr, release_app) => {
            eprintln!("freeze: PR #{number} is the release PR and may merge")
        }
        Ok(()) => eprintln!("freeze: PR #{number} may merge"),
        Err(why) => eprintln!("freeze: PR #{number} {why}"),
    }
    verdict
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = r#"[
        {"at": 1000, "kind": "set", "version": "0.1.0"},
        {"at": 1500, "kind": "set", "version": "0.1.1"},
        {"at": 4600, "kind": "lift", "reason": "publish"},
        {"at": 9000, "kind": "set", "version": "0.2.0"},
        {"at": 9300, "kind": "lift", "reason": "abort"},
        {"at": 20000, "kind": "set", "version": "0.3.0"}
    ]"#;

    fn labels(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    fn events(text: &str) -> Vec<Event> {
        parse_events(text).expect("events parse")
    }

    fn pr(base: &str, items: &[&str]) -> Pr {
        Pr {
            number: 1,
            base: base.to_owned(),
            labels: labels(items),
            head: None,
            author: None,
        }
    }

    const APP: &str = "syntropy-release[bot]";

    fn release_pr(head: &str, author: &str) -> Pr {
        Pr {
            head: Some(head.to_owned()),
            author: Some(author.to_owned()),
            ..pr("main", &["autorelease: pending"])
        }
    }

    #[test]
    fn an_unlabelled_pr_is_held_while_frozen_and_an_rc_fix_merges() {
        assert!(evaluate(true, &pr("main", &["area:ci"]), None).is_err());
        assert!(evaluate(true, &pr("main", &[]), None).is_err());
        assert_eq!(
            evaluate(true, &pr("main", &["rc:fix", "area:ci"]), None),
            Ok(())
        );
    }

    #[test]
    fn a_frozen_main_does_not_hold_a_backport_to_a_release_branch() {
        assert_eq!(
            evaluate(true, &pr("release/v0.1", &["backport:v0.1"]), None),
            Ok(())
        );
    }

    #[test]
    fn every_pr_merges_when_no_freeze_is_set() {
        assert_eq!(evaluate(false, &pr("main", &[]), None), Ok(()));
    }

    #[test]
    fn the_release_pr_merges_while_frozen() {
        let release = release_pr(RELEASE_HEAD, APP);
        assert_eq!(evaluate(true, &release, Some(APP)), Ok(()));
    }

    #[test]
    fn the_release_pr_needs_both_its_branch_and_the_release_app() {
        // Another author on release-please's branch, or the App on another branch, is held.
        assert!(evaluate(true, &release_pr(RELEASE_HEAD, "someone"), Some(APP)).is_err());
        assert!(evaluate(true, &release_pr("feature", APP), Some(APP)).is_err());
        // A PR file without head or author is never the release PR.
        assert!(evaluate(true, &pr("main", &["autorelease: pending"]), Some(APP)).is_err());
        // Without `--release-app`, or with an empty one, nothing is the release PR.
        let release = release_pr(RELEASE_HEAD, APP);
        assert!(evaluate(true, &release, None).is_err());
        assert!(evaluate(true, &release_pr(RELEASE_HEAD, ""), Some("")).is_err());
        // A label alone does not make a PR the release PR.
        assert!(
            evaluate(
                true,
                &pr("main", &["autorelease: pending", "rc:start"]),
                Some(APP)
            )
            .is_err()
        );
    }

    #[test]
    fn a_publish_or_an_abort_lifts_the_freeze() {
        let all = events(LOG);
        assert_eq!(is_frozen(&all[..1]), Ok(true));
        assert_eq!(is_frozen(&all[..3]), Ok(false));
        assert_eq!(is_frozen(&all[..4]), Ok(true));
        assert_eq!(is_frozen(&all[..5]), Ok(false));
        assert_eq!(is_frozen(&all), Ok(true));
    }

    #[test]
    fn the_head_moves_only_for_rc_fix_prs_while_frozen() {
        let queue = [
            ("feat", labels(&[])),
            ("fix-a", labels(&["rc:fix"])),
            ("chore", labels(&["area:ci"])),
            ("fix-b", labels(&["rc:fix"])),
        ];
        let merged: Vec<&str> = queue
            .iter()
            .filter(|(_, l)| {
                let l: Vec<&str> = l.iter().map(String::as_str).collect();
                evaluate(true, &pr("main", &l), None).is_ok()
            })
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(merged, ["fix-a", "fix-b"]);
    }

    #[test]
    fn durations_list_each_window_and_a_second_set_continues_it() {
        let windows = durations(&events(LOG)).expect("durations");
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].version, "0.1.0");
        assert_eq!(windows[0].continued_by, ["0.1.1"]);
        assert_eq!(windows[0].seconds(), Some(3600));
        assert_eq!(windows[0].lifted_by, Some(Reason::Publish));
        assert_eq!(windows[1].seconds(), Some(300));
        assert_eq!(windows[1].lifted_by, Some(Reason::Abort));
        assert_eq!(windows[2].end, None);
        assert_eq!(windows[2].seconds(), None);
    }

    #[test]
    fn bad_logs_are_refused() {
        assert!(durations(&events(r#"[{"at":5,"kind":"lift","reason":"abort"}]"#)).is_err());
        let backwards =
            r#"[{"at":9,"kind":"set","version":"1"},{"at":3,"kind":"lift","reason":"abort"}]"#;
        assert!(durations(&events(backwards)).is_err());
        assert!(parse_events(r#"[{"at":1,"kind":"lift"}]"#).is_err());
        assert!(parse_events(r#"[{"at":1,"kind":"thaw"}]"#).is_err());
        assert!(parse_events(r#"{"at":1}"#).is_err());
    }

    #[test]
    fn pr_files_parse() {
        assert_eq!(
            parse_pr(r#"{"number": 7, "base": "main", "labels": ["rc:fix"]}"#),
            Ok(Pr {
                number: 7,
                base: "main".to_owned(),
                labels: labels(&["rc:fix"]),
                head: None,
                author: None,
            })
        );
        let full = parse_pr(
            r#"{"number": 8, "base": "refs/heads/main", "labels": [],
                "head": "release-please--branches--main", "author": "app[bot]"}"#,
        )
        .expect("parses");
        assert_eq!(full.base, "main");
        assert_eq!(full.head.as_deref(), Some(RELEASE_HEAD));
        assert_eq!(full.author.as_deref(), Some("app[bot]"));
        assert!(parse_pr(r#"{"number": 7, "base": "main", "labels": [], "head": 3}"#).is_err());
        assert!(parse_pr(r#"{"number": 7, "labels": []}"#).is_err());
        assert!(parse_pr(r#"{"number": 7, "base": "main"}"#).is_err());
    }
}
