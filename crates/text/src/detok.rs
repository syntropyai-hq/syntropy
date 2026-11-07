// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental detokenization (RFC-0050 §5.2).
//!
//! A [`Detokenizer`] turns one choice's emitted token IDs into text deltas as they arrive. It
//! keeps two offsets into the choice's token list, `prefix` and `read`:
//!
//! - `tokens[prefix..read]` is the context window: tokens whose text was released already, kept so
//!   that the next tokens decode in context. Decoders are not concatenative token by token: a
//!   SentencePiece decoder strips the space of the first token it sees, and a byte-level or
//!   byte-fallback decoder needs the bytes of one character together.
//! - `tokens[read..]` is the pending part, whose text is not released yet.
//!
//! On each push the window `tokens[prefix..]` is decoded and the window's released text,
//! `decode(tokens[prefix..read])`, is subtracted. The remainder is released only if it does not end
//! in U+FFFD, the replacement character the decoder writes for an incomplete UTF-8 sequence. So a
//! byte-fallback tokenizer that splits one character across tokens never shows a replacement
//! character mid-stream.
//!
//! The text is also held while the last token that decodes is spelled `<0xNN>`, a byte-fallback
//! token. The byte-fallback decoder joins a run of such tokens and, when the run's bytes are not
//! valid UTF-8, writes U+FFFD for every byte of the run. So a byte token that arrives later can
//! turn text the run already spelled into replacement characters, which no released delta could
//! take back. A token of another spelling ends the run. Special tokens skipped in decoding do not
//! end it, since the decoder never sees them.
//!
//! On release the window moves up: `prefix` takes `read`'s old value and
//! `read` the end, so a window holds the last released batch and the pending tokens, and the cost
//! per token stays constant.
//!
//! [`Detokenizer::finish`] releases what is still pending, replacement characters included, so
//! the concatenation of every delta and the final flush equals the decode of the whole list.

use crate::tokenizer::Tokenizer;

/// The character a decoder writes for bytes that are not valid UTF-8.
const REPLACEMENT: char = '\u{FFFD}';

/// Why a delta could not be produced.
#[derive(Debug)]
pub enum DetokError {
    /// The engine refused to decode the window (an ID outside the vocabulary, or a decoder
    /// error).
    Decode(String),
    /// The window's released text is not a prefix of the window's text, so the decoder is not
    /// prefix-stable on this input and no delta can be cut from it.
    Unstable,
}

impl std::fmt::Display for DetokError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "the tokenizer did not decode the window: {e}"),
            Self::Unstable => {
                f.write_str("the decoder's output for the window does not extend its released text")
            }
        }
    }
}

impl std::error::Error for DetokError {}

/// One choice's incremental detokenizer (RFC-0050 §5.2).
#[derive(Clone, Debug, Default)]
pub struct Detokenizer {
    tokens: Vec<u32>,
    prefix: usize,
    read: usize,
    skip_special: bool,
}

impl Detokenizer {
    /// A detokenizer for one choice. With `skip_special`, special added tokens decode to nothing,
    /// as in the engine's `decode`.
    #[must_use]
    pub fn new(skip_special: bool) -> Self {
        Self {
            skip_special,
            ..Self::default()
        }
    }

    /// Every token pushed so far.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The tokens whose text is held back, waiting for the rest of a character.
    #[must_use]
    pub fn pending(&self) -> &[u32] {
        &self.tokens[self.read..]
    }

    /// Appends `ids` and returns the text they release, which may be empty.
    ///
    /// # Errors
    ///
    /// [`DetokError`] when the engine cannot decode the window. The detokenizer keeps the tokens,
    /// so a later call or [`finish`](Self::finish) reports the same error.
    pub fn push(&mut self, tokenizer: &Tokenizer, ids: &[u32]) -> Result<String, DetokError> {
        self.tokens.extend_from_slice(ids);
        let delta = self.window_delta(tokenizer)?;
        if delta.is_empty() || delta.ends_with(REPLACEMENT) || self.ends_in_byte_run(tokenizer) {
            return Ok(String::new());
        }
        self.prefix = self.read;
        self.read = self.tokens.len();
        Ok(delta)
    }

    /// Releases the pending text, replacement characters included, and leaves nothing pending.
    ///
    /// # Errors
    ///
    /// [`DetokError`] when the engine cannot decode the window.
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Result<String, DetokError> {
        let delta = self.window_delta(tokenizer)?;
        self.prefix = self.read;
        self.read = self.tokens.len();
        Ok(delta)
    }

    /// Whether the last pending token the decoder sees is a byte-fallback token.
    fn ends_in_byte_run(&self, tokenizer: &Tokenizer) -> bool {
        let added = tokenizer.get_added_vocabulary();
        self.tokens[self.read..]
            .iter()
            .rev()
            .filter_map(|&id| tokenizer.id_to_token(id))
            .find(|t| !(self.skip_special && added.is_special_token(t)))
            .is_some_and(|t| is_byte_token(&t))
    }

    /// `decode(tokens[prefix..])` with `decode(tokens[prefix..read])` cut from its front.
    fn window_delta(&self, tokenizer: &Tokenizer) -> Result<String, DetokError> {
        let decode = |ids: &[u32]| {
            tokenizer
                .decode(ids, self.skip_special)
                .map_err(|e| DetokError::Decode(e.to_string()))
        };
        let released = decode(&self.tokens[self.prefix..self.read])?;
        let text = decode(&self.tokens[self.prefix..])?;
        text.strip_prefix(released.as_str())
            .map(str::to_owned)
            .ok_or(DetokError::Unstable)
    }
}

/// Whether `token` is spelled as a byte-fallback token, `<0xNN>`, the spelling the byte-fallback
/// decoder turns into a byte.
fn is_byte_token(token: &str) -> bool {
    token.len() == 6
        && token.starts_with("<0x")
        && token.ends_with('>')
        && u8::from_str_radix(&token[3..5], 16).is_ok()
}

#[cfg(test)]
mod tests;
