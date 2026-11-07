// SPDX-License-Identifier: MIT OR Apache-2.0

//! How content stays marked while the template engine handles it as a plain string.
//!
//! Every request value enters the template wrapped: [`OPEN`], the value with its own copies of
//! the three marker characters escaped, [`CLOSE`]. The markers are Unicode noncharacters
//! (U+FDD0–U+FDD2), which no text format assigns a meaning. Concatenation keeps them, and the
//! string helpers in `strings.rs` take a marked string apart with [`Marked::parse`], work on the
//! plain text with one content flag per byte, and put the result back with [`Marked::emit`].
//! So a slice of content stays content and text the template writes stays template text.
//!
//! An empty value enters as the empty string, with no markers: it adds no byte to the prompt,
//! and a template that tests it for truth must see it as empty, as the reference engine does.
//!
//! The rendered string is read back by [`Marked::parse`] and split into spans. A marker the
//! template itself wrote, an unbalanced marker or a bad escape is refused: content boundaries
//! come only from the values the renderer passed in.

use std::ops::Range;

use super::TemplateError;
use crate::tokenizer::Span;

/// Starts a content value.
pub(super) const OPEN: char = '\u{FDD0}';
/// Ends a content value.
pub(super) const CLOSE: char = '\u{FDD1}';
/// Inside content, escapes the next character: `o` for [`OPEN`], `c` for [`CLOSE`], `e` for
/// itself.
pub(super) const ESC: char = '\u{FDD2}';

/// A request value as the template receives it.
pub(super) fn mark(content: &str) -> String {
    if content.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(content.len() + 6);
    out.push(OPEN);
    push_escaped(&mut out, content);
    out.push(CLOSE);
    out
}

fn push_escaped(out: &mut String, text: &str) {
    for c in text.chars() {
        match c {
            OPEN => out.extend([ESC, 'o']),
            CLOSE => out.extend([ESC, 'c']),
            ESC => out.extend([ESC, 'e']),
            c => out.push(c),
        }
    }
}

/// Plain text with a content flag for each byte.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Marked {
    pub(super) text: String,
    pub(super) content: Vec<bool>,
}

impl Marked {
    /// Text the template wrote.
    pub(super) fn template(text: &str) -> Marked {
        Marked {
            text: text.to_owned(),
            content: vec![false; text.len()],
        }
    }

    /// Reads a marked string.
    ///
    /// # Errors
    ///
    /// [`TemplateError::Markup`] for a marker outside content, a nested or unclosed one, or a
    /// bad escape.
    pub(super) fn parse(s: &str) -> Result<Marked, TemplateError> {
        let bad = |what: &str| TemplateError::Markup(what.to_owned());
        let mut out = Marked {
            text: String::with_capacity(s.len()),
            content: Vec::with_capacity(s.len()),
        };
        let mut in_content = false;
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            let c = match (c, in_content) {
                (OPEN, false) => {
                    in_content = true;
                    continue;
                }
                (CLOSE, true) => {
                    in_content = false;
                    continue;
                }
                (ESC, true) => match chars.next() {
                    Some('o') => OPEN,
                    Some('c') => CLOSE,
                    Some('e') => ESC,
                    _ => return Err(bad("a bad escape inside content")),
                },
                (OPEN, true) => return Err(bad("a content marker inside content")),
                (CLOSE | ESC, false) => {
                    return Err(bad("a content marker outside content"));
                }
                (c, _) => c,
            };
            let start = out.text.len();
            out.text.push(c);
            out.content.resize(out.text.len(), in_content);
            debug_assert!(out.content[start..].iter().all(|&f| f == in_content));
        }
        if in_content {
            return Err(bad("content that is never closed"));
        }
        Ok(out)
    }

    /// Whether any byte is content.
    pub(super) fn has_content(&self) -> bool {
        self.content.iter().any(|&c| c)
    }

    /// The bytes in `range`, marked again. `range` lies on character boundaries.
    pub(super) fn emit(&self, range: Range<usize>) -> String {
        let mut out = String::with_capacity(range.len() + 6);
        for (start, end, content) in self.runs(range) {
            let text = &self.text[start..end];
            if content {
                out.push(OPEN);
                push_escaped(&mut out, text);
                out.push(CLOSE);
            } else {
                out.push_str(text);
            }
        }
        out
    }

    /// The whole value, marked again.
    pub(super) fn emit_all(&self) -> String {
        self.emit(0..self.text.len())
    }

    /// Appends `other`.
    pub(super) fn push(&mut self, other: &Marked) {
        self.text.push_str(&other.text);
        self.content.extend_from_slice(&other.content);
    }

    /// Maximal runs of one kind inside `range`, as `(start, end, content)`.
    fn runs(&self, range: Range<usize>) -> Vec<(usize, usize, bool)> {
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        for i in range {
            let c = self.content[i];
            match runs.last_mut() {
                Some(last) if last.2 == c && last.1 == i => last.1 = i + 1,
                _ => runs.push((i, i + 1, c)),
            }
        }
        runs
    }

    /// The spans, adjacent runs of one kind merged.
    pub(super) fn into_spans(self) -> Vec<Span> {
        self.runs(0..self.text.len())
            .into_iter()
            .map(|(start, end, content)| {
                let text = &self.text[start..end];
                if content {
                    Span::content(text)
                } else {
                    Span::template(text)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::SpanKind;

    #[test]
    fn marking_round_trips_values_that_hold_the_markers() {
        let nasty = format!("a{OPEN}b{CLOSE}c{ESC}d{ESC}o");
        let m = Marked::parse(&format!("t{}u", mark(&nasty))).unwrap();
        assert_eq!(m.text, format!("t{nasty}u"));
        let spans = m.into_spans();
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1], Span::content(nasty));
        assert_eq!(spans[0].kind, SpanKind::Template);
    }

    #[test]
    fn an_empty_value_is_unmarked() {
        assert_eq!(mark(""), "");
    }

    #[test]
    fn markers_the_template_wrote_are_refused() {
        for s in [
            format!("{CLOSE}"),
            format!("{ESC}o"),
            format!("{OPEN}x"),
            format!("{OPEN}{OPEN}x{CLOSE}"),
            format!("{OPEN}{ESC}z{CLOSE}"),
            format!("{OPEN}{ESC}"),
        ] {
            assert!(
                matches!(Marked::parse(&s), Err(TemplateError::Markup(_))),
                "{s:?}"
            );
        }
    }

    #[test]
    fn emit_keeps_each_byte_s_kind() {
        let m = Marked::parse(&format!("ab{}cd", mark("XYZ"))).unwrap();
        let part = Marked::parse(&m.emit(1..4)).unwrap();
        assert_eq!(part.text, "bXY");
        assert_eq!(part.content, [false, true, true]);
    }
}
