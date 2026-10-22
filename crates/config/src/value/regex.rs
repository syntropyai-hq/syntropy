// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Regex`: a linear-time, size-limited regular expression over tensor names (RFC-0054 §3,
//! §6.3).

use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use super::{ValueError, serde_via_str};

/// The longest pattern accepted, in bytes \[target\]: a chosen budget, far above any tensor-name
/// override, that bounds parse time on hostile input.
pub const MAX_PATTERN_LEN: usize = 1024;

/// The largest compiled program accepted, in bytes \[target\]: a chosen budget. A tensor-name
/// pattern compiles to a few KiB; a pattern such as `(\w{100}){100}` compiles far above this
/// and is rejected (RFC-0054 testing table: "regex size limits enforced").
pub const SIZE_LIMIT: usize = 256 * 1024;

/// Nesting depth accepted \[target\]: a chosen budget against stack exhaustion while parsing.
const NEST_LIMIT: u32 = 64;

/// A regular expression, written as a TOML literal string such as
/// `'blk\.(\d+)\.ffn_.*_exps'` (RFC-0054 §3).
///
/// - It always matches the whole name: the pattern is compiled as `\A(?:pattern)\z`.
/// - Matching takes time linear in the name's length, with no backtracking.
/// - Patterns longer than [`MAX_PATTERN_LEN`] bytes, or compiling to more than [`SIZE_LIMIT`]
///   bytes, are rejected as `bad_regex`.
/// - It prints as the pattern as written, and two regexes are equal when their patterns are.
#[derive(Clone)]
pub struct Regex {
    pattern: String,
    compiled: regex::Regex,
}

impl Regex {
    /// Compiles `pattern` within the limits.
    pub fn new(pattern: &str) -> Result<Self, ValueError> {
        if pattern.len() > MAX_PATTERN_LEN {
            return Err(ValueError::bad_regex(format!(
                "the pattern is {} bytes; the limit is {MAX_PATTERN_LEN}",
                pattern.len()
            )));
        }
        // The pattern must parse on its own first. Unbalanced parentheses would otherwise close
        // the wrapper's group early: `a)|(b` becomes `\A(?:a)|(b)\z`, which matches any name
        // that starts with `a` or ends with `b`. Once the pattern parses alone, its groups are
        // balanced, so the wrapper anchors all of it.
        build(pattern, pattern)?;
        let compiled = build(&format!(r"\A(?:{pattern})\z"), pattern)?;
        Ok(Self {
            pattern: pattern.to_owned(),
            compiled,
        })
    }

    /// The pattern as written.
    pub fn as_str(&self) -> &str {
        &self.pattern
    }

    /// Whether the pattern matches all of `name`.
    pub fn is_full_match(&self, name: &str) -> bool {
        self.compiled.is_match(name)
    }

    /// The capture groups of a full match of `name`, if it matches: group 0 is `name`.
    pub fn captures<'n>(&self, name: &'n str) -> Option<regex::Captures<'n>> {
        self.compiled.captures(name)
    }
}

/// Compiles `text` within the limits, naming `pattern` in an error.
fn build(text: &str, pattern: &str) -> Result<regex::Regex, ValueError> {
    regex::RegexBuilder::new(text)
        .size_limit(SIZE_LIMIT)
        .dfa_size_limit(SIZE_LIMIT)
        .nest_limit(NEST_LIMIT)
        .build()
        .map_err(|e| match e {
            regex::Error::CompiledTooBig(limit) => ValueError::bad_regex(format!(
                "`{pattern}` compiles to more than {limit} bytes; simplify it"
            )),
            other => {
                // The parser's message spans lines: the pattern, a caret, then `error: <why>`.
                // One line keeps an error's header on one line.
                let text = other.to_string();
                let why = text
                    .lines()
                    .rev()
                    .find_map(|line| line.strip_prefix("error: "))
                    .unwrap_or_else(|| text.lines().last().unwrap_or("").trim());
                ValueError::bad_regex(format!("`{pattern}` does not compile: {why}"))
            }
        })
}

impl fmt::Debug for Regex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Regex").field(&self.pattern).finish()
    }
}

impl fmt::Display for Regex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.pattern)
    }
}

impl PartialEq for Regex {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
    }
}

impl Eq for Regex {}

impl Hash for Regex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pattern.hash(state);
    }
}

impl FromStr for Regex {
    type Err = ValueError;

    fn from_str(s: &str) -> Result<Self, ValueError> {
        Self::new(s)
    }
}

serde_via_str!(
    Regex,
    "a regular expression, written as a TOML literal string"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::ValueErrorKind;

    #[test]
    fn matches_whole_names_only() {
        let re = Regex::new(r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight").unwrap();
        assert!(re.is_full_match("blk.3.ffn_up_exps.weight"));
        assert!(!re.is_full_match("blk.3.ffn_up_exps.weight.extra"));
        assert!(!re.is_full_match("x.blk.3.ffn_up_exps.weight"));
        assert_eq!(
            &re.captures("blk.12.ffn_gate_exps.weight").unwrap()[1],
            "12"
        );
        // An alternation at the top level is still anchored as a whole.
        let re = Regex::new("a|b").unwrap();
        assert!(re.is_full_match("a") && re.is_full_match("b") && !re.is_full_match("ab"));
    }

    #[test]
    fn rejects_patterns_that_would_escape_the_anchors() {
        // Each is invalid alone. Wrapped unchecked, each compiled with the anchors on one
        // alternative only, and `blk\.0\.)|(.*` matched every name.
        for pattern in [r"a)|(b", r"blk\.0\.)|(.*", r"x)(?:y"] {
            let err = Regex::new(pattern).unwrap_err();
            assert_eq!(err.kind(), ValueErrorKind::BadRegex, "{pattern}");
            assert!(err.message().contains("does not compile"), "{err}");
        }
        // Balanced groups and top-level alternations stay anchored as a whole.
        let re = Regex::new(r"(a)|(b)").unwrap();
        assert!(re.is_full_match("a") && re.is_full_match("b"));
        assert!(!re.is_full_match("axyz") && !re.is_full_match("zzzb"));
        let re = Regex::new(r"blk\.0\..*|token_embd\.weight").unwrap();
        assert!(re.is_full_match("token_embd.weight"));
        assert!(!re.is_full_match("output.weight"));
    }

    #[test]
    fn rejects_patterns_above_the_size_limit() {
        let err = Regex::new(r"(\w{100}){100}").unwrap_err();
        assert_eq!(err.kind(), ValueErrorKind::BadRegex);
        assert!(err.message().contains("compiles to more than"), "{err}");

        let long = "a".repeat(MAX_PATTERN_LEN + 1);
        let err = Regex::new(&long).unwrap_err();
        assert_eq!(err.kind(), ValueErrorKind::BadRegex);
        assert!(Regex::new(&"a".repeat(MAX_PATTERN_LEN)).is_ok());
    }

    #[test]
    fn rejects_patterns_that_do_not_compile() {
        for pattern in ["(", "a{2,1}", "[z-a]", "(?<n>a)(?<n>b)"] {
            let err = Regex::new(pattern).unwrap_err();
            assert_eq!(err.kind(), ValueErrorKind::BadRegex, "{pattern}");
        }
        // Backreferences need backtracking, which this engine does not do.
        assert!(Regex::new(r"(a)\1").is_err());
        let deep = format!("{}a{}", "(".repeat(100), ")".repeat(100));
        assert!(Regex::new(&deep).is_err());
    }

    #[test]
    fn round_trips_through_its_text() {
        let re = Regex::new(r"blk\.(\d+)\..*").unwrap();
        assert_eq!(re.to_string().parse::<Regex>(), Ok(re.clone()));
        let json = serde_json::to_string(&re).unwrap();
        assert_eq!(serde_json::from_str::<Regex>(&json).unwrap(), re);
    }
}
