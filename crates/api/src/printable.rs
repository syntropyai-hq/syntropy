// SPDX-License-Identifier: MIT OR Apache-2.0

//! Text from elsewhere made safe to show on a terminal: what `doctor --fix`'s prompt shows of a
//! file, and what the CLI prints of a daemon's answer (RFC-0051 §2.2).
//!
//! [`printable`] escapes every control character but a tab, ESC among them (it starts the CSI and
//! OSC sequences that move the cursor, clear the screen or set the window title), and every
//! [`invisible`] formatting character, so the text cannot redraw the terminal, reverse what
//! follows it or hide part of itself. Each escape is Rust's `char::escape_default` form, such as
//! `\u{1b}` or `\n`.

use std::borrow::Cow;

/// `text` with every control character but a tab, and every [`invisible`] character, escaped.
#[must_use]
pub fn printable(text: &str) -> Cow<'_, str> {
    let unsafe_char = |c: char| (c.is_control() && c != '\t') || invisible(c);
    if !text.chars().any(unsafe_char) {
        return Cow::Borrowed(text);
    }
    let mut safe = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if unsafe_char(c) {
            safe.extend(c.escape_default());
        } else {
            safe.push(c);
        }
    }
    Cow::Owned(safe)
}

/// Whether `c` changes how the text around it shows without showing itself: the soft hyphen and
/// the combining grapheme joiner, the zero-width characters and joiners, the bidirectional
/// controls, the line and paragraph separators, the byte-order mark, the interlinear annotation
/// characters and the tag characters. The list is `syntropy-config`'s, whose printed TOML writes
/// them as escapes, so both agree.
#[must_use]
pub fn invisible(c: char) -> bool {
    syntropy_config::chars::invisible(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_borrowed_unchanged() {
        for text in ["plain text, ≈ ünïcode", "שלום", "tab\tstays", ""] {
            assert!(
                matches!(printable(text), Cow::Borrowed(t) if t == text),
                "{text:?}"
            );
        }
    }

    #[test]
    fn controls_and_escape_sequences_are_escaped() {
        assert_eq!(printable("a\u{1b}[2Jb"), "a\\u{1b}[2Jb");
        assert_eq!(
            printable("x\u{1b}]0;title\u{7}y"),
            "x\\u{1b}]0;title\\u{7}y"
        );
        assert_eq!(printable("c1\u{9b}31m"), "c1\\u{9b}31m");
        assert_eq!(printable("line\nbreak\r"), "line\\nbreak\\r");
        assert_eq!(printable("del\u{7f}"), "del\\u{7f}");
        assert_eq!(printable("nul\0"), "nul\\u{0}");
    }

    #[test]
    fn invisible_and_bidirectional_characters_are_escaped() {
        assert_eq!(printable("ok\u{202e}gnp.exe"), "ok\\u{202e}gnp.exe");
        for c in [
            '\u{00AD}',
            '\u{034F}',
            '\u{061C}',
            '\u{180E}',
            '\u{200B}',
            '\u{200C}',
            '\u{200D}',
            '\u{200E}',
            '\u{200F}',
            '\u{2028}',
            '\u{2029}',
            '\u{202A}',
            '\u{202D}',
            '\u{2060}',
            '\u{2066}',
            '\u{2069}',
            '\u{FEFF}',
            '\u{FFF9}',
            '\u{E0001}',
            '\u{E007F}',
        ] {
            let text = format!("a{c}b");
            let shown = printable(&text);
            assert!(!shown.contains(c), "{c:?}");
            assert_eq!(shown, format!("a{}b", c.escape_default()), "{c:?}");
        }
    }
}
