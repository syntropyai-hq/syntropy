// SPDX-License-Identifier: MIT OR Apache-2.0

//! What a request can constrain its output to (RFC-0050 §7.1).

use serde::{Deserialize, Serialize};

/// The syntax of a [`OutputConstraint::Grammar`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarSyntax {
    /// Lark-style EBNF.
    Lark,
}

/// An output constraint (RFC-0050 §7.1).
///
/// RFC-0050's `ToolCall` kind, the family's tool-call syntax with each tool's schema spliced in,
/// is built internally from a request's tools, and comes with forced and strict tool calls
/// (PROD-E3-T06).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputConstraint {
    /// Any JSON object.
    JsonObject,
    /// JSON matching `schema`. With `strict`, a keyword outside RFC-0050 §7.2's coverage fails the
    /// request; without it, the keyword is ignored and named in a warning.
    JsonSchema {
        schema: serde_json::Value,
        strict: bool,
    },
    /// Text matching a regular expression (native API only).
    Regex(String),
    /// Text in a grammar (native API only).
    Grammar { syntax: GrammarSyntax, text: String },
    /// Exactly one of these strings (native API only).
    Choice(Vec<String>),
}
