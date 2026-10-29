// SPDX-License-Identifier: MIT OR Apache-2.0

//! The daemon's logger: every record, from `tracing` or a runner's `log` ring, passes the level
//! filter, the content gate and repeat suppression, then goes to every sink.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use syntropy_config::logging::{LogLevel, LoggingSection};
use syntropy_core::Clock;
use syntropy_obs::log::{Callsite, Level, LogRecord};

use super::event::{Context, Field, LogEvent, Value};
use super::format::WallClock;
use super::sink::Sink;
use super::suppress::Suppressor;
use crate::crash::LogTail;

/// What the `[logging]` section sets, as the pipeline applies it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Settings {
    level: Level,
    /// `logging.filter`'s directives, longest target first.
    filter: Vec<(String, Level)>,
    include_content: bool,
}

fn level(level: LogLevel) -> Level {
    match level {
        LogLevel::Error => Level::Error,
        LogLevel::Warn => Level::Warn,
        LogLevel::Info => Level::Info,
        LogLevel::Debug => Level::Debug,
        LogLevel::Trace => Level::Trace,
    }
}

fn parse_level(word: &str) -> Option<Level> {
    match word.trim().to_ascii_lowercase().as_str() {
        "error" => Some(Level::Error),
        "warn" => Some(Level::Warn),
        "info" => Some(Level::Info),
        "debug" => Some(Level::Debug),
        "trace" => Some(Level::Trace),
        _ => None,
    }
}

impl Settings {
    /// The settings `section` gives. A filter directive is `target=level`; a bare level sets the
    /// default, as `logging.level` does; a directive that does not parse is ignored.
    fn from_section(section: &LoggingSection) -> Settings {
        let mut level = level(section.level);
        let mut filter = Vec::new();
        for directive in section
            .filter
            .split(',')
            .map(str::trim)
            .filter(|d| !d.is_empty())
        {
            match directive.split_once('=') {
                Some((target, word)) => {
                    if let Some(at) = parse_level(word) {
                        filter.push((target.trim().to_owned(), at));
                    }
                }
                None => {
                    if let Some(at) = parse_level(directive) {
                        level = at;
                    }
                }
            }
        }
        filter.sort_by_key(|(target, _)| std::cmp::Reverse(target.len()));
        Settings {
            level,
            filter,
            include_content: section.include_content,
        }
    }

    /// The most detailed level logged for `target`.
    fn level_for(&self, target: &str) -> Level {
        self.filter
            .iter()
            .find(|(prefix, _)| target == prefix || target.starts_with(&format!("{prefix}::")))
            .map_or(self.level, |&(_, at)| at)
    }

    /// Whether content is written in a record at `at`: only with `logging.include_content`, and
    /// only at `debug` or below (RFC-0053 §4.3).
    fn shows_content(&self, at: Level) -> bool {
        self.include_content && at >= Level::Debug
    }
}

struct Inner {
    settings: Settings,
    suppressor: Suppressor,
    sinks: Vec<Sink>,
}

/// The daemon's logger. Shared by the `tracing` subscriber, the runner-ring readers and the
/// reload path.
pub struct Logger {
    clock: Arc<dyn Clock>,
    wall: WallClock,
    inner: Mutex<Inner>,
    /// The last lines written, with no content, for crash records (RFC-0053 §6).
    tail: Arc<LogTail>,
}

impl std::fmt::Debug for Logger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Logger")
            .field("wall", &self.wall)
            .finish_non_exhaustive()
    }
}

/// The warning content logging takes at start and on every reload (RFC-0053 §4.3).
pub const CONTENT_WARNING: &str = "logging.include_content is on: prompts, outputs and token IDs \
                                   are written to the logs at debug";

impl Logger {
    /// A logger writing to `sinks`, with `section`'s settings and `clock`'s time. With
    /// `logging.include_content` on, it warns at once.
    pub fn new(clock: Arc<dyn Clock>, section: &LoggingSection, sinks: Vec<Sink>) -> Logger {
        let wall = WallClock::new(clock.unix_ms(), clock.now());
        let logger = Logger {
            clock,
            wall,
            inner: Mutex::new(Inner {
                settings: Settings::from_section(section),
                suppressor: Suppressor::default(),
                sinks,
            }),
            tail: Arc::new(LogTail::new()),
        };
        logger.warn_if_content();
        logger
    }

    /// Applies a reloaded `[logging]` section. With `logging.include_content` on, it warns again.
    pub fn apply(&self, section: &LoggingSection) {
        self.inner().settings = Settings::from_section(section);
        self.warn_if_content();
    }

    fn warn_if_content(&self) {
        if self.inner().settings.include_content {
            self.log(LogEvent::new(
                self.clock.now(),
                Level::Warn,
                "syntropy_daemon::logging",
                CONTENT_WARNING,
            ));
        }
    }

    fn inner(&self) -> MutexGuard<'_, Inner> {
        // A sink that panicked mid-write leaves nothing half-done that a later write relies on.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The wall clock records are stamped with.
    pub fn wall(&self) -> WallClock {
        self.wall
    }

    /// The logger's clock.
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.clock
    }

    /// The last [`TAIL_LINES`](crate::crash::TAIL_LINES) records written, as text with no content
    /// whatever `logging.include_content` says, for crash records.
    pub fn tail(&self) -> &Arc<LogTail> {
        &self.tail
    }

    /// Whether a record at `at` from `target` would be written.
    pub fn enabled(&self, at: Level, target: &str) -> bool {
        at <= self.inner().settings.level_for(target)
    }

    /// Writes `event` to every sink, if the filter lets it through: without its content unless
    /// content is on and the record is at `debug` or below, and collapsed with its repeats.
    pub fn log(&self, mut event: LogEvent) {
        let mut inner = self.inner();
        if event.level > inner.settings.level_for(&event.target) {
            return;
        }
        if !inner.settings.shows_content(event.level) {
            event.fields.retain(|field| !field.content);
            if let Some(count) = event.key_fields {
                event.key_fields = Some(count.min(event.fields.len()));
            }
        }
        let out = inner.suppressor.admit(event);
        let wall = self.wall;
        for event in &out {
            for sink in &mut inner.sinks {
                sink.write(event, &wall);
            }
            self.tail.push(event, &wall);
        }
    }

    /// Writes the summaries of the repeat windows that have closed by now. The daemon calls it
    /// about once a second.
    pub fn tick(&self) {
        let now = self.clock.now();
        let mut inner = self.inner();
        let out = inner.suppressor.due(now);
        let wall = self.wall;
        for event in &out {
            for sink in &mut inner.sinks {
                sink.write(event, &wall);
            }
            self.tail.push(event, &wall);
        }
    }

    /// Writes every pending summary and flushes the sinks, at shutdown.
    pub fn flush(&self) {
        let mut inner = self.inner();
        let out = inner.suppressor.flush();
        let wall = self.wall;
        for event in &out {
            for sink in &mut inner.sinks {
                sink.write(event, &wall);
            }
            self.tail.push(event, &wall);
        }
        for sink in &mut inner.sinks {
            sink.flush();
        }
    }

    /// Runs `f` over the sinks, for tests that read what a memory sink kept.
    pub fn with_sinks<R>(&self, f: impl FnOnce(&mut [Sink]) -> R) -> R {
        f(&mut self.inner().sinks)
    }
}

/// The record a runner's `LogRecord` stands for, from the build's `callsites` (RFC-0053 §4.1):
/// its message with each `{name}` replaced by that argument, and each argument a field. A content
/// argument shows as `[content]` in the message, never its value, and is a content field.
/// `None` when the table has no such callsite.
pub fn from_record(
    record: &LogRecord,
    callsites: &[Callsite],
    context: Context,
) -> Option<LogEvent> {
    let callsite = callsites.iter().find(|c| c.id == record.callsite)?;
    let mut message = callsite.format.to_owned();
    let mut fields = Vec::new();
    for (index, name) in callsite.fields.iter().enumerate().take(4) {
        let arg = record.args[index];
        let content = callsite.is_content(index);
        let shown = if content {
            "[content]".to_owned()
        } else {
            arg.to_string()
        };
        message = message.replace(&format!("{{{name}}}"), &shown);
        fields.push(Field {
            name: (*name).into(),
            value: Value::U64(arg),
            content,
        });
    }
    let mut event = LogEvent::new(record.at, record.level, callsite.target, message);
    event.callsite = format!("{}#{}", callsite.target, callsite.id.0).into();
    event.fields = fields;
    event.key_fields = Some(usize::from(callsite.key_fields));
    event.context = context;
    Some(event)
}
