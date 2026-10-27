// SPDX-License-Identifier: MIT OR Apache-2.0

//! One log record as the daemon handles it, whatever produced it (RFC-0053 §4.1).

use std::borrow::Cow;
use std::fmt;

use syntropy_core::MonoTime;
use syntropy_obs::log::Level;

/// A field's value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// An unsigned integer.
    U64(u64),
    /// A signed integer.
    I64(i64),
    /// A float.
    F64(f64),
    /// A boolean.
    Bool(bool),
    /// Text.
    Str(String),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::Str(v) => f.write_str(v),
        }
    }
}

/// One field of a record.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    /// Its name.
    pub name: Cow<'static, str>,
    /// Its value.
    pub value: Value,
    /// Whether it is content: a prompt, an output, token IDs or anything derived from them. Such
    /// a field is written only with `logging.include_content`, and only at `debug` or below.
    pub content: bool,
}

impl Field {
    /// A field that is not content.
    pub fn new(name: impl Into<Cow<'static, str>>, value: Value) -> Field {
        Field {
            name: name.into(),
            value,
            content: false,
        }
    }

    /// A content field.
    pub fn content(name: impl Into<Cow<'static, str>>, value: Value) -> Field {
        Field {
            name: name.into(),
            value,
            content: true,
        }
    }
}

/// The prefix of a `tracing` field that holds content, such as `content.prompt`.
pub const CONTENT_PREFIX: &str = "content.";

/// What every record carries where it applies (RFC-0053 §4.1): who and what it is about. The
/// sequence is an opaque short ID, never a full `SequenceId`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Context {
    /// The node.
    pub node: Option<String>,
    /// The instance.
    pub instance: Option<String>,
    /// The stage.
    pub stage: Option<u16>,
    /// The runner's process ID, for a runner's record.
    pub pid: Option<u32>,
    /// The sequence, as an opaque short ID.
    pub seq: Option<String>,
    /// The trace ID.
    pub trace: Option<String>,
    /// The error code, as `domain.code`.
    pub code: Option<String>,
}

/// One record.
#[derive(Clone, Debug, PartialEq)]
pub struct LogEvent {
    /// When, on the node's `Clock` timeline.
    pub at: MonoTime,
    /// How severe.
    pub level: Level,
    /// The crate and module, such as `syntropy_net::link`.
    pub target: Cow<'static, str>,
    /// What produced it, for repeat suppression: a callsite's identity. Records from one
    /// callsite with one level and equal key fields collapse (RFC-0053 §4.1).
    pub callsite: Cow<'static, str>,
    /// The message. Never content: content goes in content fields.
    pub message: String,
    /// The fields.
    pub fields: Vec<Field>,
    /// How many of the leading fields identify the record for repeat suppression; `None` for
    /// all of them.
    pub key_fields: Option<usize>,
    /// The context.
    pub context: Context,
    /// Set on a record that stands for this many suppressed repeats of it (RFC-0053 §4.1).
    pub repeated: Option<u64>,
}

impl LogEvent {
    /// A record from `target` with `message` and no fields; its callsite is its target and
    /// message.
    pub fn new(
        at: MonoTime,
        level: Level,
        target: impl Into<Cow<'static, str>>,
        message: impl Into<String>,
    ) -> LogEvent {
        let target = target.into();
        let message = message.into();
        LogEvent {
            at,
            level,
            callsite: Cow::Owned(format!("{target}:{message}")),
            target,
            message,
            fields: Vec::new(),
            key_fields: None,
            context: Context::default(),
            repeated: None,
        }
    }

    /// The record with `field` added.
    #[must_use]
    pub fn with(mut self, field: Field) -> LogEvent {
        self.fields.push(field);
        self
    }

    /// The crate the record comes from: its target up to the first `::`.
    pub fn crate_name(&self) -> &str {
        self.target.split("::").next().unwrap_or(&self.target)
    }
}
