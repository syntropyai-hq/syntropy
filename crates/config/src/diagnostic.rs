// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a load finds wrong, and how it is printed (RFC-0054 §8).
//!
//! Layering ([`crate::ResolvedConfig::resolve`]) and the validation phases report every finding as
//! a [`Problem`]: the error code, where it is, the key it is about and what is wrong. The
//! messages never hold a value, which could be a secret. [`render`] prints a problem in §8's
//! format with its source line, and withholds every line or source that could hold a
//! secret-typed value (RFC-0054 §8 note of 2026-10-22).

use std::fmt;
use std::path::Path;

use crate::code::Code;
use crate::origin::Origin;
use crate::registry::Registry;
use crate::render::{self, At, Report};
use crate::resolve::ConfigFile;

pub use crate::render::Severity;

/// An error or warning found while loading: where it is, if anywhere, and what is wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// The `Config` error code, or `None` for a warning §8 gives no code, such as a deprecated
    /// key.
    pub code: Option<Code>,
    /// The file and line, variable or flag.
    pub at: Option<Origin>,
    /// The 1-based column, for a problem at a place in a file.
    pub column: Option<u32>,
    /// The caret's width at `column`, in characters.
    pub width: Option<u32>,
    /// The dotted key the problem is about, as written.
    pub key: Option<Box<str>>,
    /// What is wrong. Never a value, which could be a secret.
    pub message: String,
    /// A hint, such as "did you mean `ram_headroom`?".
    pub help: Option<Box<str>>,
}

impl Problem {
    /// A problem with `code` at `at`.
    pub(crate) fn new(code: Code, at: Option<Origin>, message: impl Into<String>) -> Self {
        Self {
            code: Some(code),
            at,
            column: None,
            width: None,
            key: None,
            message: message.into(),
            help: None,
        }
    }

    /// A warning that §8 gives no code.
    pub(crate) fn uncoded(at: Option<Origin>, message: impl Into<String>) -> Self {
        Self {
            code: None,
            ..Self::new(Code::Syntax, at, message)
        }
    }

    /// The same problem, about `key`.
    pub(crate) fn about(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into().into_boxed_str());
        self
    }

    /// The same problem, at `column` with a caret `width` wide.
    pub(crate) fn spanning(mut self, column: u32, width: u32) -> Self {
        self.column = Some(column);
        self.width = Some(width.max(1));
        self
    }

    /// The same problem, with a hint.
    pub(crate) fn hint(mut self, help: Option<String>) -> Self {
        self.help = help.map(String::into_boxed_str);
        self
    }
}

/// `where: message`, as one line. [`render`] prints the full form.
impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.at, self.column) {
            (Some(Origin::File { path, line }), Some(column)) => {
                write!(f, "{}:{line}:{column}: {}", path.display(), self.message)
            }
            (Some(at), _) => write!(f, "{at}: {}", self.message),
            (None, _) => f.write_str(&self.message),
        }
    }
}

/// Renders `problem` in RFC-0054 §8's format, with its source line from `files` where it has
/// one, withholding every line or source that could hold a secret-typed value of `registry`.
pub fn render(
    problem: &Problem,
    severity: Severity,
    files: &[ConfigFile],
    registry: &Registry,
) -> String {
    let place;
    let at = match &problem.at {
        Some(Origin::File { path, line }) => match file_text(files, path) {
            Some(text) => {
                let (column, width) = match problem.column {
                    Some(column) => (column, problem.width.unwrap_or(1)),
                    None => render::key_span(text, *line).unwrap_or((1, 1)),
                };
                At::File {
                    path,
                    text,
                    line: *line,
                    column,
                    width,
                }
            }
            None => {
                place = format!("{}:{line}", path.display());
                At::Elsewhere(&place)
            }
        },
        Some(origin) => {
            place = origin.to_string();
            At::Elsewhere(&place)
        }
        None => At::Nowhere,
    };
    render::render(
        &Report {
            severity,
            code: problem.code,
            message: &problem.message,
            at,
            help: problem.help.as_deref(),
        },
        registry,
    )
}

/// Renders every error, then every warning, each in §8's format.
pub fn render_all(
    errors: &[Problem],
    warnings: &[Problem],
    files: &[ConfigFile],
    registry: &Registry,
) -> String {
    let errors = errors
        .iter()
        .map(|p| render(p, Severity::Error, files, registry));
    let warnings = warnings
        .iter()
        .map(|p| render(p, Severity::Warning, files, registry));
    errors.chain(warnings).collect()
}

fn file_text<'a>(files: &'a [ConfigFile], path: &Path) -> Option<&'a str> {
    files
        .iter()
        .find(|f| f.path == path)
        .map(|f| f.text.as_str())
}
