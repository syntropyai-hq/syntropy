// SPDX-License-Identifier: MIT OR Apache-2.0

//! Characters that must never reach a terminal raw, and that the TOML this crate prints writes
//! as escapes (RFC-0051 §2.2).

/// Whether `c` changes how the text around it shows without showing itself: the soft hyphen and
/// the combining grapheme joiner, the zero-width characters and joiners, the bidirectional
/// controls (marks, embeddings, overrides and isolates, Unicode UAX #9), which can reverse the
/// text after them, the line and paragraph separators, the byte-order mark, the interlinear
/// annotation characters and the tag characters.
#[must_use]
pub fn invisible(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}'
            | '\u{034F}'
            | '\u{061C}'
            | '\u{180E}'
            | '\u{200B}'..='\u{200F}'
            | '\u{2028}'..='\u{202E}'
            | '\u{2060}'..='\u{2069}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{E0000}'..='\u{E007F}'
    )
}
