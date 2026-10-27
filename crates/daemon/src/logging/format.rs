// SPDX-License-Identifier: MIT OR Apache-2.0

//! How a record is written (RFC-0053 §4.1, §4.2): its UTC timestamp, human text, JSON Lines and
//! journald's native fields. Each takes a record whose content the pipeline has already removed.

use std::fmt::Write as _;

use syntropy_core::MonoTime;
use syntropy_obs::log::Level;

use super::event::{LogEvent, Value};

/// Turns node-timeline instants into UTC microseconds: one sample of the `Clock`'s wall time
/// and monotonic time, taken when the logger starts, then monotonic differences. Every
/// timestamp comes from the injected `Clock`, never from the OS directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WallClock {
    origin_unix_us: i128,
    origin: MonoTime,
}

impl WallClock {
    /// The wall clock whose instant `origin` is `unix_ms` milliseconds after the Unix epoch.
    pub fn new(unix_ms: u64, origin: MonoTime) -> WallClock {
        WallClock {
            origin_unix_us: i128::from(unix_ms) * 1_000,
            origin,
        }
    }

    /// `at` in microseconds since the Unix epoch, UTC.
    pub fn unix_us(&self, at: MonoTime) -> i128 {
        self.origin_unix_us + (i128::from(at.0) - i128::from(self.origin.0)) / 1_000
    }
}

/// `unix_us` as RFC 3339 UTC with microseconds: `2026-10-27T18:00:00.123456Z`.
pub fn utc(unix_us: i128) -> String {
    let seconds = unix_us.div_euclid(1_000_000);
    let micros = unix_us.rem_euclid(1_000_000);
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{micros:06}Z",
        of_day / 3_600,
        of_day / 60 % 60,
        of_day % 60
    )
}

/// The proleptic Gregorian date `days` after 1970-01-01.
fn civil(days: i128) -> (i128, i128, i128) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i128::from(month <= 2);
    (year, month, day)
}

/// Every field a record shows, its context first, then its own, then `repeated`.
fn shown_fields(event: &LogEvent) -> Vec<(String, String)> {
    let c = &event.context;
    let mut out = Vec::new();
    let mut push = |name: &str, value: Option<String>| {
        if let Some(value) = value {
            out.push((name.to_owned(), value));
        }
    };
    push("node", c.node.clone());
    push("instance", c.instance.clone());
    push("stage", c.stage.map(|s| s.to_string()));
    push("pid", c.pid.map(|p| p.to_string()));
    push("seq", c.seq.clone());
    push("trace", c.trace.clone());
    push("code", c.code.clone());
    for field in &event.fields {
        out.push((field.name.to_string(), field.value.to_string()));
    }
    if let Some(repeated) = event.repeated {
        out.push(("repeated".to_owned(), repeated.to_string()));
    }
    out
}

/// The level as text writes it, padded to five.
fn level_word(level: Level) -> &'static str {
    match level {
        Level::Error => "ERROR",
        Level::Warn => "WARN ",
        Level::Info => "INFO ",
        Level::Debug => "DEBUG",
        Level::Trace => "TRACE",
    }
}

/// The ANSI colour of a level on a terminal.
fn colour(level: Level) -> &'static str {
    match level {
        Level::Error => "\x1b[31m",
        Level::Warn => "\x1b[33m",
        Level::Info => "\x1b[32m",
        Level::Debug => "\x1b[34m",
        Level::Trace => "\x1b[35m",
    }
}

/// `text` with every control character escaped (`\n`, `\r`, `\u{1b}`, …), so no record can move
/// a terminal's cursor, colour its output or forge a line.
pub fn escape_controls(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_control() {
            out.extend(c.escape_default());
        } else {
            out.push(c);
        }
    }
    out
}

/// A value as text shows it: bare when it is a plain word, quoted otherwise. Quoting escapes
/// every control character.
fn quoted(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._:-/@".contains(c))
    {
        value.to_owned()
    } else {
        format!("{value:?}")
    }
}

/// Human text: `<UTC> <LEVEL> <target>: <message> key=value…`, one line, no newline. With
/// `stamp = false` the timestamp is left off, for a sink that stamps records itself.
pub fn text(event: &LogEvent, wall: &WallClock, stamp: bool, coloured: bool) -> String {
    let mut line = String::new();
    if stamp {
        line.push_str(&utc(wall.unix_us(event.at)));
        line.push(' ');
    }
    if coloured {
        let _ = write!(
            line,
            "{}{}\x1b[0m ",
            colour(event.level),
            level_word(event.level)
        );
    } else {
        line.push_str(level_word(event.level));
        line.push(' ');
    }
    let _ = write!(
        line,
        "{}: {}",
        event.target,
        escape_controls(&event.message)
    );
    for (name, value) in shown_fields(event) {
        let _ = write!(line, " {name}={}", quoted(&value));
    }
    line
}

/// One JSON object, no newline: `ts`, `level`, `target`, `message`, then the fields.
pub fn json(event: &LogEvent, wall: &WallClock) -> String {
    let mut object = serde_json::Map::new();
    object.insert("ts".to_owned(), utc(wall.unix_us(event.at)).into());
    object.insert("level".to_owned(), event.level.name().into());
    object.insert("target".to_owned(), event.target.to_string().into());
    object.insert("message".to_owned(), event.message.clone().into());
    let c = &event.context;
    let mut put = |name: &str, value: serde_json::Value| {
        object.insert(name.to_owned(), value);
    };
    if let Some(v) = &c.node {
        put("node", v.clone().into());
    }
    if let Some(v) = &c.instance {
        put("instance", v.clone().into());
    }
    if let Some(v) = c.stage {
        put("stage", v.into());
    }
    if let Some(v) = c.pid {
        put("pid", v.into());
    }
    if let Some(v) = &c.seq {
        put("seq", v.clone().into());
    }
    if let Some(v) = &c.trace {
        put("trace", v.clone().into());
    }
    if let Some(v) = &c.code {
        put("code", v.clone().into());
    }
    for field in &event.fields {
        let value = match &field.value {
            Value::U64(v) => (*v).into(),
            Value::I64(v) => (*v).into(),
            Value::F64(v) => serde_json::Number::from_f64(*v)
                .map_or_else(|| v.to_string().into(), serde_json::Value::Number),
            Value::Bool(v) => (*v).into(),
            Value::Str(v) => v.clone().into(),
        };
        put(&field.name, value);
    }
    if let Some(repeated) = event.repeated {
        put("repeated", repeated.into());
    }
    serde_json::Value::Object(object).to_string()
}

/// The syslog priority journald files a level under.
pub fn priority(level: Level) -> u8 {
    match level {
        Level::Error => 3,
        Level::Warn => 4,
        Level::Info => 6,
        Level::Debug | Level::Trace => 7,
    }
}

/// A field name as journald takes it: upper case, `A–Z`, `0–9` and `_`, not starting with `_`
/// or a digit.
fn journal_name(name: &str) -> String {
    let upper: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("SYNTROPY_{}", upper.trim_start_matches('_'))
}

/// Appends one journald field: `NAME=value\n`, or the binary form for a value holding a newline.
fn journal_field(out: &mut Vec<u8>, name: &str, value: &str) {
    out.extend_from_slice(name.as_bytes());
    if value.contains('\n') {
        out.push(b'\n');
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    } else {
        out.push(b'=');
        out.extend_from_slice(value.as_bytes());
    }
    out.push(b'\n');
}

/// The record in journald's native protocol: `MESSAGE`, `PRIORITY`, `SYSLOG_IDENTIFIER`, the
/// target and every field as `SYNTROPY_<NAME>`, structured, as RFC-0053 §4.2 asks.
pub fn journal(event: &LogEvent, wall: &WallClock) -> Vec<u8> {
    let mut out = Vec::new();
    journal_field(&mut out, "MESSAGE", &event.message);
    journal_field(&mut out, "PRIORITY", &priority(event.level).to_string());
    journal_field(&mut out, "SYSLOG_IDENTIFIER", "syntropy");
    journal_field(&mut out, "SYNTROPY_TARGET", &event.target);
    journal_field(&mut out, "SYNTROPY_TIMESTAMP", &utc(wall.unix_us(event.at)));
    for (name, value) in shown_fields(event) {
        journal_field(&mut out, &journal_name(&name), &value);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::event::Field;

    #[test]
    fn utc_is_rfc_3339_with_microseconds() {
        assert_eq!(utc(0), "1970-01-01T00:00:00.000000Z");
        // 2026-10-27T18:30:15.123456Z.
        assert_eq!(utc(1_793_125_815_123_456), "2026-10-27T18:30:15.123456Z");
        assert_eq!(utc(951_782_400_000_000), "2000-02-29T00:00:00.000000Z");
        assert_eq!(utc(-1), "1969-12-31T23:59:59.999999Z");
    }

    #[test]
    fn the_wall_clock_follows_monotonic_time() {
        let wall = WallClock::new(1_000, MonoTime(5_000_000));
        assert_eq!(wall.unix_us(MonoTime(5_000_000)), 1_000_000);
        assert_eq!(wall.unix_us(MonoTime(5_002_500)), 1_000_002);
        assert_eq!(wall.unix_us(MonoTime(4_000_000)), 999_000);
    }

    fn event() -> LogEvent {
        LogEvent::new(
            MonoTime(1_000_000),
            Level::Warn,
            "syntropy_net::link",
            "link degraded",
        )
        .with(Field::new("link", Value::Str("eth0".to_owned())))
        .with(Field::new("loss", Value::F64(0.25)))
        .with(Field::new("why", Value::Str("two words".to_owned())))
    }

    #[test]
    fn text_and_json_carry_every_field() {
        let wall = WallClock::new(0, MonoTime(0));
        let mut e = event();
        e.context.node = Some("n1".to_owned());
        e.repeated = Some(7);
        assert_eq!(
            text(&e, &wall, true, false),
            "1970-01-01T00:00:00.001000Z WARN  syntropy_net::link: link degraded node=n1 \
             link=eth0 loss=0.25 why=\"two words\" repeated=7"
        );
        let object: serde_json::Value = serde_json::from_str(&json(&e, &wall)).unwrap();
        assert_eq!(object["level"], "warn");
        assert_eq!(object["link"], "eth0");
        assert_eq!(object["repeated"], 7);
        assert_eq!(object["ts"], "1970-01-01T00:00:00.001000Z");
    }

    /// A message cannot carry a terminal escape or a forged line into text.
    #[test]
    fn control_characters_are_escaped() {
        let wall = WallClock::new(0, MonoTime(0));
        let mut e = event();
        e.message = "a\x1b[31mred\nINFO forged\r".to_owned();
        e.fields.clear();
        let line = text(&e, &wall, false, false);
        assert!(!line.chars().any(char::is_control), "{line:?}");
        assert!(line.ends_with("a\\u{1b}[31mred\\nINFO forged\\r"), "{line}");
        e.fields
            .push(Field::new("v", Value::Str("x\x07y".to_owned())));
        assert!(!text(&e, &wall, false, false).chars().any(char::is_control));
    }

    #[test]
    fn journal_fields_are_structured() {
        let wall = WallClock::new(0, MonoTime(0));
        let mut e = event();
        e.message = "two\nlines".to_owned();
        let bytes = journal(&e, &wall);
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("PRIORITY=4\n"));
        assert!(text.contains("SYNTROPY_LINK=eth0\n"));
        assert!(text.contains("SYNTROPY_TARGET=syntropy_net::link\n"));
        // The newline forces the binary form: the name, a newline, a length, the bytes.
        let mut expected = b"MESSAGE\n".to_vec();
        expected.extend_from_slice(&9u64.to_le_bytes());
        expected.extend_from_slice(b"two\nlines\n");
        assert!(bytes.starts_with(&expected));
    }
}
