// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repeat suppression (RFC-0053 §4.1): identical records within 10 s collapse into one.
//!
//! Records are identical when they share a callsite, a level and their key fields' values. The
//! first of a run is written at once and opens a 10 s window. Repeats inside the window are
//! counted, not written. When the window closes, one record stands for them all: the last
//! repeat, with `repeated = N`, the number of records it replaces. A repeat after the window
//! opens a new one. A flapping link therefore writes two records each 10 s, however fast it
//! flaps.
//!
//! Time is the record's own `at`, on the node's `Clock` timeline, and [`Suppressor::due`] is
//! driven by the logger's `Clock`: nothing here reads a clock, so a test of 1,000 warnings in
//! 10 s is exact.

use std::collections::BTreeMap;

use syntropy_core::{MonoTime, Nanos};

use super::event::LogEvent;

/// The window (RFC-0053 §4.1): 10 s \[target\].
pub const WINDOW: Nanos = Nanos(10_000_000_000);

/// The most runs tracked at once. When a new run would exceed it, the oldest run's summary is
/// written early, so memory stays bounded whatever the records.
pub const MAX_RUNS: usize = 4_096;

/// What identifies a run: the callsite, the level and the key fields' values.
type Key = (String, u8, Vec<String>);

/// One run of identical records.
struct Run {
    opened: MonoTime,
    repeats: u64,
    last: Option<LogEvent>,
}

/// The suppressor's state.
#[derive(Default)]
pub struct Suppressor {
    runs: BTreeMap<Key, Run>,
}

impl std::fmt::Debug for Suppressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Suppressor")
            .field("runs", &self.runs.len())
            .finish()
    }
}

fn key(event: &LogEvent) -> Key {
    let count = event.key_fields.unwrap_or(event.fields.len());
    (
        event.callsite.to_string(),
        event.level as u8,
        event
            .fields
            .iter()
            .take(count)
            .map(|field| format!("{}={}", field.name, field.value))
            .collect(),
    )
}

impl Suppressor {
    /// Takes `event`; the records to write now: the summaries of runs that closed before it, then
    /// the event itself if it opens a run.
    pub fn admit(&mut self, event: LogEvent) -> Vec<LogEvent> {
        let mut out = self.due(event.at);
        let key = key(&event);
        if let Some(run) = self.runs.get_mut(&key) {
            run.repeats += 1;
            run.last = Some(event);
            return out;
        }
        if self.runs.len() >= MAX_RUNS
            && let Some(oldest) = self
                .runs
                .iter()
                .min_by_key(|(_, run)| run.opened)
                .map(|(key, _)| key.clone())
            && let Some(run) = self.runs.remove(&oldest)
        {
            out.extend(summary(run));
        }
        self.runs.insert(
            key,
            Run {
                opened: event.at,
                repeats: 0,
                last: None,
            },
        );
        out.push(event);
        out
    }

    /// The summaries of the runs whose window has closed by `now`.
    pub fn due(&mut self, now: MonoTime) -> Vec<LogEvent> {
        let closed: Vec<Key> = self
            .runs
            .iter()
            .filter(|(_, run)| now.saturating_duration_since(run.opened) >= WINDOW)
            .map(|(key, _)| key.clone())
            .collect();
        let mut out: Vec<LogEvent> = closed
            .into_iter()
            .filter_map(|key| self.runs.remove(&key))
            .filter_map(summary)
            .collect();
        out.sort_by_key(|event| event.at);
        out
    }

    /// Every pending summary, at shutdown.
    pub fn flush(&mut self) -> Vec<LogEvent> {
        let mut out: Vec<LogEvent> = std::mem::take(&mut self.runs)
            .into_values()
            .filter_map(summary)
            .collect();
        out.sort_by_key(|event| event.at);
        out
    }

    /// Runs being tracked.
    pub fn runs(&self) -> usize {
        self.runs.len()
    }
}

/// The record that stands for a run's repeats, if it had any.
fn summary(run: Run) -> Option<LogEvent> {
    let mut last = run.last?;
    last.repeated = Some(run.repeats);
    Some(last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::event::{Field, Value};
    use syntropy_obs::log::Level;

    fn warning(at_ms: u64, link: &str) -> LogEvent {
        LogEvent::new(
            MonoTime(at_ms * 1_000_000),
            Level::Warn,
            "syntropy_net::link",
            "link flapped",
        )
        .with(Field::new("link", Value::Str(link.to_owned())))
    }

    /// The criterion "1,000 identical warnings in 10 s become one record with `repeated`": the
    /// first is written, and the other 999 become one record with `repeated = 999` when the
    /// window closes.
    #[test]
    fn a_thousand_warnings_in_ten_seconds_collapse() {
        let mut s = Suppressor::default();
        let mut written = Vec::new();
        for n in 0..1_000u64 {
            written.extend(s.admit(warning(n * 9, "eth0")));
        }
        assert_eq!(written.len(), 1, "only the first is written in the window");
        assert_eq!(written[0].repeated, None);
        // 8,991 ms: still inside. 10 s after the first, the window closes.
        assert!(s.due(MonoTime(9_999_000_000)).is_empty());
        let closed = s.due(MonoTime(10_000_000_000));
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0].repeated, Some(999));
        assert_eq!(closed[0].at, MonoTime(8_991_000_000), "the last repeat");
        assert_eq!(s.runs(), 0);
    }

    #[test]
    fn different_keys_levels_and_windows_do_not_collapse() {
        let mut s = Suppressor::default();
        let mut written = s.admit(warning(0, "eth0"));
        written.extend(s.admit(warning(1, "eth1")));
        let mut error = warning(2, "eth0");
        error.level = Level::Error;
        written.extend(s.admit(error));
        assert_eq!(written.len(), 3);
        // A repeat after the window: the summary first, then a new run.
        written.extend(s.admit(warning(5, "eth0")));
        let later = s.admit(warning(20_000, "eth0"));
        assert_eq!(later.len(), 2);
        assert_eq!(later[0].repeated, Some(1));
        assert_eq!(later[1].repeated, None);
        assert_eq!(s.flush().len(), 0, "no repeats are pending");
    }

    #[test]
    fn key_fields_limit_what_must_match() {
        let mut s = Suppressor::default();
        let a = warning(0, "eth0").with(Field::new("micros", Value::U64(10)));
        let mut b = warning(1, "eth0").with(Field::new("micros", Value::U64(99)));
        let mut a = a;
        a.key_fields = Some(1);
        b.key_fields = Some(1);
        assert_eq!(s.admit(a).len(), 1);
        assert!(s.admit(b).is_empty(), "only `link` is a key field");
        assert_eq!(s.flush()[0].repeated, Some(1));
    }
}
