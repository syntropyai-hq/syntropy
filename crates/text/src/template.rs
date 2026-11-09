// SPDX-License-Identifier: MIT OR Apache-2.0

//! Chat-template rendering into template and content spans (RFC-0050 §6.1, §6.4; ADR-0431;
//! PROD-E3-T10).
//!
//! A model ships its chat template as Jinja. [`ChatTemplate::new`] loads it into `minijinja`
//! configured as the reference engine runs chat templates: `trim_blocks` and `lstrip_blocks`
//! on, undefined values lenient, and the helpers templates call: `raise_exception`,
//! `strftime_now` (from the request's date, never a clock) and a `tojson` that writes JSON as
//! Python's `json.dumps` does. Python's string methods that templates use (`startswith`,
//! `split`, `strip`, …) are provided too.
//!
//! [`ChatTemplate::render`] passes `messages` (no `tools` yet: PROD-E3-T01 adds them),
//! `add_generation_prompt`, and the optional `enable_thinking`, `bos_token` and `eos_token`.
//! It returns the prompt as [`Span`]s: what the template wrote is [`SpanKind::Template`], and
//! every message's text, a request value, is [`SpanKind::Content`]. The helpers keep the
//! marking through concatenation, slicing, splitting and stripping, so a template cannot turn
//! content into template text by handling it. [`crate::tokenizer::SpanEncoder`] then tokenizes
//! content literally (RFC-0043 §8.7). The spans' text, joined, is the reference engine's
//! rendering byte for byte.
//!
//! # Untrusted templates
//!
//! A template is text from a model file, so it is data. Rendering is bounded:
//!
//! - **Fuel.** Every engine instruction costs fuel, and a render stops with
//!   [`TemplateError::OutOfFuel`] when [`Limits::fuel`] runs out: an endless or huge loop ends.
//! - **Recursion.** Nesting deeper than [`Limits::recursion`] stops with
//!   [`TemplateError::Recursion`].
//! - **Output.** A rendering longer than [`Limits::max_bytes`] is refused with
//!   [`TemplateError::TooLarge`].
//! - **No panic.** A panic inside the engine is caught and returned as
//!   [`TemplateError::Engine`].
//! - **Features.** Macros, `include`, `import` and `extends` are not built in, so a template
//!   that uses them fails to load, and no template can reach a file or another template.
//!
//! Fuel bounds instructions, not memory: one instruction can build a long string (`'x' * n`, a
//! string doubled on each of a few loop passes). The output limit refuses such a result only
//! after it was built. Templates come with the model a user chose, keyed by its
//! `TokenizerId`, and the per-model `chat_template` override replaces a broken one.

mod clock;
mod json;
mod markup;
mod strings;

use std::fmt;
use std::panic::{self, AssertUnwindSafe};

use minijinja::value::Value;
use minijinja::{Environment, Error, ErrorKind, UndefinedBehavior};
use serde::Serialize;
use syntropy_core::ErrorClass;

use crate::tokenizer::Span;
#[cfg(doc)]
use crate::tokenizer::SpanKind;

/// Who wrote a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// The system prompt.
    System,
    /// The user.
    User,
    /// The model, in an earlier turn.
    Assistant,
}

/// One message of a conversation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Message {
    /// Who wrote it.
    pub role: Role,
    /// Its text.
    pub content: String,
    /// The reasoning an assistant turn carried separately (`reasoning_content`), when it did.
    pub reasoning: Option<String>,
}

impl Message {
    /// A message without separate reasoning.
    pub fn new(role: Role, content: impl Into<String>) -> Message {
        Message {
            role,
            content: content.into(),
            reasoning: None,
        }
    }
}

/// A calendar date, for `strftime_now`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Date {
    /// The year.
    pub year: i32,
    /// The month, 1–12.
    pub month: u8,
    /// The day of the month, 1–31.
    pub day: u8,
}

/// What one rendering takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderRequest<'a> {
    /// The conversation.
    pub messages: &'a [Message],
    /// Whether to open the model's turn at the end.
    pub add_generation_prompt: bool,
    /// `enable_thinking`, for templates that read it; `None` leaves it undefined.
    pub enable_thinking: Option<bool>,
    /// The BOS token's text, for templates that write `bos_token`.
    pub bos_token: Option<&'a str>,
    /// The EOS token's text, for templates that write `eos_token`.
    pub eos_token: Option<&'a str>,
    /// Today's date, for `strftime_now`; a template that calls it without one fails.
    pub date: Option<Date>,
}

impl<'a> RenderRequest<'a> {
    /// A request for `messages` with the model's turn opened and nothing else set.
    pub fn new(messages: &'a [Message]) -> RenderRequest<'a> {
        RenderRequest {
            messages,
            add_generation_prompt: true,
            enable_thinking: None,
            bos_token: None,
            eos_token: None,
            date: None,
        }
    }
}

/// The bounds on one rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Engine instructions per render, plus [`fuel_per_message`](Self::fuel_per_message) for
    /// each message.
    pub fuel: u64,
    /// Fuel added for each message, so a long conversation is not refused.
    pub fuel_per_message: u64,
    /// The deepest nesting of blocks.
    pub recursion: usize,
    /// The longest rendering, in bytes.
    pub max_bytes: usize,
}

impl Default for Limits {
    /// 200,000 instructions plus 2,000 per message, nesting 100, output 64 MiB \[target\]. Every
    /// golden conversation, up to seven messages, renders through the Qwen3-shaped test template
    /// within 2,000 instructions \[measured: `limits_leave_the_m0_template_headroom` in this
    /// crate's tests, which pass with that fuel\], so the default leaves two orders of
    /// magnitude of headroom. 64 MiB is far above the text of the longest context any v1
    /// family takes \[estimate: 1M tokens at about 4 bytes each is 4 MiB\].
    fn default() -> Self {
        Limits {
            fuel: 200_000,
            fuel_per_message: 2_000,
            recursion: 100,
            max_bytes: 64 << 20,
        }
    }
}

/// Why a template did not load or render.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateError {
    /// The template does not parse, or uses a feature this engine does not build in.
    Syntax(String),
    /// The template ran out of fuel.
    OutOfFuel,
    /// The template nested deeper than the limit.
    Recursion,
    /// The rendering is longer than the limit.
    TooLarge {
        /// Its length.
        bytes: usize,
        /// The limit.
        max: usize,
    },
    /// The template called `raise_exception`: it refuses this conversation.
    Raised(String),
    /// The template wrote a content marker itself or broke one.
    Markup(String),
    /// The template failed in another way: an unknown method or filter, a bad operation.
    Render(String),
    /// The engine panicked.
    Engine(String),
}

impl TemplateError {
    /// The error class (RFC-0001 §10). A conversation the template refuses is the client's;
    /// a template that cannot be rendered, or exceeds a bound, is the model's.
    pub fn class(&self) -> ErrorClass {
        match self {
            TemplateError::Raised(_) => ErrorClass::InvalidRequest,
            TemplateError::OutOfFuel
            | TemplateError::Recursion
            | TemplateError::TooLarge { .. } => ErrorClass::Capacity,
            TemplateError::Syntax(_) | TemplateError::Markup(_) | TemplateError::Render(_) => {
                ErrorClass::Unsupported
            }
            TemplateError::Engine(_) => ErrorClass::Internal,
        }
    }
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateError::Syntax(e) => write!(f, "chat template does not load: {e}"),
            TemplateError::OutOfFuel => write!(f, "chat template ran out of fuel"),
            TemplateError::Recursion => write!(f, "chat template nests too deeply"),
            TemplateError::TooLarge { bytes, max } => {
                write!(
                    f,
                    "chat template rendered {bytes} bytes, over the {max}-byte limit"
                )
            }
            TemplateError::Raised(m) => write!(f, "chat template refused the conversation: {m}"),
            TemplateError::Markup(m) => write!(f, "chat template broke content marking: {m}"),
            TemplateError::Render(e) => write!(f, "chat template failed: {e}"),
            TemplateError::Engine(e) => write!(f, "chat template engine failed: {e}"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// The detail `raise_exception` errors carry, so they map to [`TemplateError::Raised`].
const RAISED: &str = "\u{FDD3}raise_exception\u{FDD3}";

fn strings_error(e: TemplateError) -> Error {
    Error::new(ErrorKind::InvalidOperation, e.to_string())
}

const NAME: &str = "chat_template";

/// A loaded chat template. Build one per model and share it.
#[derive(Clone)]
pub struct ChatTemplate {
    env: Environment<'static>,
    limits: Limits,
}

impl fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Loads `source` with `limits`.
    ///
    /// # Errors
    ///
    /// [`TemplateError::Syntax`] when it does not parse; [`TemplateError::Engine`] if the
    /// parser panicked.
    pub fn new(source: &str, limits: Limits) -> Result<ChatTemplate, TemplateError> {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        env.set_recursion_limit(limits.recursion);
        env.set_unknown_method_callback(strings::method);
        env.add_filter("tojson", json::tojson);
        env.add_filter("trim", strings::trim_filter);
        env.add_filter("length", strings::length_filter);
        env.add_filter("count", strings::length_filter);
        env.add_filter("upper", strings::upper_filter);
        env.add_filter("lower", strings::lower_filter);
        env.add_filter("replace", strings::replace_filter);
        env.add_function(
            "raise_exception",
            |msg: Option<Value>| -> Result<Value, Error> {
                let msg = msg.map(|m| match m.as_str() {
                    Some(s) => markup::Marked::parse(s).map_or_else(|_| s.to_owned(), |m| m.text),
                    None => m.to_string(),
                });
                Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("{RAISED}{}", msg.unwrap_or_default()),
                ))
            },
        );
        let source = source.to_owned();
        let loaded = panic::catch_unwind(AssertUnwindSafe(move || {
            env.add_template_owned(NAME, source).map(|()| env)
        }));
        match loaded {
            Ok(Ok(env)) => Ok(ChatTemplate { env, limits }),
            Ok(Err(e)) => Err(TemplateError::Syntax(describe(&e))),
            Err(p) => Err(TemplateError::Engine(panic_text(&p))),
        }
    }

    /// The limits it renders with.
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Renders `req` into spans.
    ///
    /// # Errors
    ///
    /// A [`TemplateError`]; nothing is returned in part.
    pub fn render(&self, req: &RenderRequest<'_>) -> Result<Vec<Span>, TemplateError> {
        let mut env = self.env.clone();
        let per_message = self
            .limits
            .fuel_per_message
            .saturating_mul(req.messages.len() as u64);
        env.set_fuel(Some(self.limits.fuel.saturating_add(per_message)));
        if let Some(date) = req.date.filter(|d| d.valid()) {
            env.add_function("strftime_now", move |format: String| {
                clock::strftime(date, &format)
                    .map_err(|e| Error::new(ErrorKind::InvalidOperation, e))
            });
        } else {
            env.add_function("strftime_now", |_: String| -> Result<String, Error> {
                Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "strftime_now: the request has no date",
                ))
            });
        }

        let ctx = context(req);
        let rendered = panic::catch_unwind(AssertUnwindSafe(|| {
            env.get_template(NAME).and_then(|t| t.render(&ctx))
        }));
        let text = match rendered {
            Ok(Ok(text)) => text,
            Ok(Err(e)) => return Err(classify(&e)),
            Err(p) => return Err(TemplateError::Engine(panic_text(&p))),
        };
        if text.len() > self.limits.max_bytes {
            return Err(TemplateError::TooLarge {
                bytes: text.len(),
                max: self.limits.max_bytes,
            });
        }
        Ok(markup::Marked::parse(&text)?.into_spans())
    }
}

/// One message as the template sees it: content marked, `reasoning_content` only when set.
#[derive(Serialize)]
struct MessageValue {
    role: Role,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

fn context(req: &RenderRequest<'_>) -> Value {
    let messages: Vec<MessageValue> = req
        .messages
        .iter()
        .map(|m| MessageValue {
            role: m.role,
            content: markup::mark(&m.content),
            reasoning_content: m.reasoning.as_deref().map(markup::mark),
        })
        .collect();
    let mut ctx = std::collections::BTreeMap::new();
    ctx.insert("messages", Value::from_serialize(&messages));
    // The reference engine passes these two as none when the request has none.
    ctx.insert("tools", Value::from(()));
    ctx.insert("documents", Value::from(()));
    ctx.insert(
        "add_generation_prompt",
        Value::from(req.add_generation_prompt),
    );
    if let Some(t) = req.enable_thinking {
        ctx.insert("enable_thinking", Value::from(t));
    }
    if let Some(t) = req.bos_token {
        ctx.insert("bos_token", Value::from(t));
    }
    if let Some(t) = req.eos_token {
        ctx.insert("eos_token", Value::from(t));
    }
    Value::from(ctx)
}

/// Maps an engine error, following its causes.
fn classify(e: &Error) -> TemplateError {
    let mut cur: Option<&Error> = Some(e);
    while let Some(err) = cur {
        if err.kind() == ErrorKind::OutOfFuel {
            return TemplateError::OutOfFuel;
        }
        if let Some(d) = err.detail() {
            if let Some(msg) = d.strip_prefix(RAISED) {
                return TemplateError::Raised(msg.to_owned());
            }
            if d.contains("recursion limit exceeded") {
                return TemplateError::Recursion;
            }
        }
        cur = std::error::Error::source(err).and_then(|s| s.downcast_ref::<Error>());
    }
    TemplateError::Render(describe(e))
}

fn describe(e: &Error) -> String {
    let mut s = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(cause) = src {
        s.push_str(": ");
        s.push_str(&cause.to_string());
        src = cause.source();
    }
    s
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic".to_owned())
}
