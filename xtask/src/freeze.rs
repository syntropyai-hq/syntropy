// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask ci freeze --events <file> [--pr <file>] [--durations-out <file>]`: the release
//! freeze of RFC-0062 §9.2 step 3 (QUAL-E17-T04, ADR-0621 decision 4).
//!
//! `rc:start` sets the freeze on `main`. While it is set, a PR merges only if it carries the
//! `rc:fix` label. A publish or an abort lifts it. A failed RC does not lift it: the next patch's
//! `rc:fix` PRs merge under the same freeze (§9.2 step 5).
//!
//! This module is the rule, over data. `--events` is the freeze's event log, a JSON array in time
//! order of `{"at": <unix seconds>, "kind": "set" | "lift", "version": "X.Y.Z"}`; a `lift` also
//! has `"reason": "publish" | "abort"`. `--pr` is `{"number": <n>, "labels": ["rc:fix", ...]}`.
//! With `--pr` the command fails when the freeze holds that PR. With `--durations-out` it writes
//! each freeze window as JSON, for the CI cost report (QUAL-E1-T13).
//!
//! What writes the log, and what calls this on a PR, is not here: `release.yml` and `rc:start`
//! are QUAL-E17-T05 and the merge-queue hold is QUAL-E17-T17. `ci-gate` does not consult it yet.
//!
//! The freeze targets 48 h `[target]`, RFC-0060's RC budget (RFC-0062 §9.2).

use serde_json::{Value, json};

use crate::run::Context;

/// The label that lets a PR merge during a freeze.
pub const FIX_LABEL: &str = "rc:fix";

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
    pub version: String,
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
                start: *at,
                end: None,
                lifted_by: None,
            }),
            (Event::Set { .. }, Some(_)) => {}
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

/// The freeze's verdict on one PR: `Ok` if it may merge, else why it is held.
pub fn evaluate(frozen: bool, labels: &[String]) -> Result<(), String> {
    if frozen && !labels.iter().any(|l| l == FIX_LABEL) {
        return Err(format!(
            "held: a release freeze is set and the PR is not labelled `{FIX_LABEL}`"
        ));
    }
    Ok(())
}

/// Reads a PR file: `{"number": <n>, "labels": [...]}`. Returns the number and the labels.
pub fn parse_pr(text: &str) -> Result<(u64, Vec<String>), String> {
    let value: Value = serde_json::from_str(text).map_err(|e| format!("pr: not JSON: {e}"))?;
    let number = value
        .get("number")
        .and_then(Value::as_u64)
        .ok_or_else(|| "pr: no numeric `number`".to_owned())?;
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
    Ok((number, labels))
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
    let (number, labels) = parse_pr(&read("--pr")?)?;
    let verdict = evaluate(frozen, &labels);
    match &verdict {
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

    #[test]
    fn an_unlabelled_pr_is_held_while_frozen_and_an_rc_fix_merges() {
        assert!(evaluate(true, &labels(&["area:ci"])).is_err());
        assert!(evaluate(true, &[]).is_err());
        assert_eq!(evaluate(true, &labels(&["rc:fix", "area:ci"])), Ok(()));
    }

    #[test]
    fn every_pr_merges_when_no_freeze_is_set() {
        assert_eq!(evaluate(false, &[]), Ok(()));
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
            .filter(|(_, l)| evaluate(true, l).is_ok())
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(merged, ["fix-a", "fix-b"]);
    }

    #[test]
    fn durations_list_each_window_and_a_second_set_continues_it() {
        let windows = durations(&events(LOG)).expect("durations");
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].version, "0.1.0");
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
            parse_pr(r#"{"number": 7, "labels": ["rc:fix"]}"#),
            Ok((7, labels(&["rc:fix"])))
        );
        assert!(parse_pr(r#"{"number": 7}"#).is_err());
    }
}
