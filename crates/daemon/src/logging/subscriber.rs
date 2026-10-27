// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `tracing` subscriber of the daemon and of runners' non-hot code (RFC-0053 §4.1): each
//! `tracing` event becomes a [`LogEvent`] in the daemon's [`Logger`].
//!
//! Spans are not recorded here: RFC-0053 §3's spans are PROD-E8-T07's. A field named
//! `content.<name>` is content, written only with `logging.include_content` and only at `debug`
//! (RFC-0053 §4.3), so code that logs a prompt, an output or token IDs puts them there and never
//! in the message.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_obs::log::Level;
use tracing::field::{Field as TracingField, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::{Event, Metadata, Subscriber};

use super::event::{CONTENT_PREFIX, Field, LogEvent, Value};
use super::pipeline::Logger;

/// A `tracing` subscriber that writes events through a [`Logger`].
#[derive(Debug)]
pub struct LogSubscriber {
    logger: Arc<Logger>,
    next_span: AtomicU64,
}

impl LogSubscriber {
    /// The subscriber over `logger`.
    pub fn new(logger: Arc<Logger>) -> LogSubscriber {
        LogSubscriber {
            logger,
            next_span: AtomicU64::new(1),
        }
    }
}

fn level(level: &tracing::Level) -> Level {
    match *level {
        tracing::Level::ERROR => Level::Error,
        tracing::Level::WARN => Level::Warn,
        tracing::Level::INFO => Level::Info,
        tracing::Level::DEBUG => Level::Debug,
        _ => Level::Trace,
    }
}

/// Collects an event's fields: `message` into the message, `content.*` as content.
#[derive(Default)]
struct Fields {
    message: String,
    fields: Vec<Field>,
}

impl Fields {
    fn push(&mut self, field: &TracingField, value: Value) {
        let name = field.name();
        if name == "message" {
            self.message = value.to_string();
        } else if name.starts_with(CONTENT_PREFIX) {
            self.fields.push(Field::content(name, value));
        } else {
            self.fields.push(Field::new(name, value));
        }
    }
}

impl Visit for Fields {
    fn record_u64(&mut self, field: &TracingField, value: u64) {
        self.push(field, Value::U64(value));
    }

    fn record_i64(&mut self, field: &TracingField, value: i64) {
        self.push(field, Value::I64(value));
    }

    fn record_f64(&mut self, field: &TracingField, value: f64) {
        self.push(field, Value::F64(value));
    }

    fn record_bool(&mut self, field: &TracingField, value: bool) {
        self.push(field, Value::Bool(value));
    }

    fn record_str(&mut self, field: &TracingField, value: &str) {
        self.push(field, Value::Str(value.to_owned()));
    }

    fn record_debug(&mut self, field: &TracingField, value: &dyn fmt::Debug) {
        self.push(field, Value::Str(format!("{value:?}")));
    }
}

impl Subscriber for LogSubscriber {
    fn register_callsite(&self, _: &'static Metadata<'static>) -> Interest {
        // `logging.level` and `logging.filter` reload hot, so every callsite is asked each time.
        Interest::sometimes()
    }

    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.is_span()
            || self
                .logger
                .enabled(level(metadata.level()), metadata.target())
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(self.next_span.fetch_add(1, Ordering::Relaxed))
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let metadata = event.metadata();
        let mut fields = Fields::default();
        event.record(&mut fields);
        let mut record = LogEvent::new(
            self.logger.clock().now(),
            level(metadata.level()),
            metadata.target().to_owned(),
            fields.message,
        );
        record.callsite = format!(
            "{}:{}",
            metadata.file().unwrap_or(metadata.target()),
            metadata.line().unwrap_or(0)
        )
        .into();
        record.fields = fields.fields;
        self.logger.log(record);
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}
