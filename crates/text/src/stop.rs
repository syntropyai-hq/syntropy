// SPDX-License-Identifier: MIT OR Apache-2.0

//! The stop-string matcher with hold-back (RFC-0050 §5.2).
//!
//! Stop strings can span tokens, so the matcher works on released text, after the
//! [`Detokenizer`](crate::detok::Detokenizer). It holds back the longest suffix of the text that is
//! a proper prefix of some stop string, which is at most `max_stop_len − 1` bytes, and releases the
//! rest. A stop string matches as soon as its last byte arrives. When two stop strings complete at
//! the same byte, the one that starts first, which is the longer, wins. On a match the matcher
//! releases the text before the stop string and nothing after it.
//!
//! The result does not depend on how the text is split into pushes: it is what a search of the
//! whole text for the first stop string to complete gives.

/// A completed match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StopMatch {
    /// The index of the stop string in the list [`StopMatcher::new`] was given.
    pub index: usize,
}

/// What one push releases.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Released {
    /// Text that can be delivered to the client.
    pub text: String,
    /// The stop string that matched, if one did. The choice then finishes with `StopSequence`,
    /// and the matcher releases nothing more.
    pub stop: Option<StopMatch>,
}

/// One choice's stop matcher (RFC-0050 §5.2).
#[derive(Clone, Debug)]
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
    stopped: bool,
}

impl StopMatcher {
    /// A matcher for `stops`. Empty strings never match and are dropped; the remaining strings
    /// keep their indices in `stops` for [`StopMatch::index`].
    #[must_use]
    pub fn new<S: AsRef<str>>(stops: &[S]) -> Self {
        Self {
            stops: stops.iter().map(|s| s.as_ref().to_owned()).collect(),
            held: String::new(),
            stopped: false,
        }
    }

    /// The length in bytes of the longest stop string.
    #[must_use]
    pub fn max_stop_len(&self) -> usize {
        self.stops.iter().map(String::len).max().unwrap_or(0)
    }

    /// The text held back as a possible start of a stop string.
    #[must_use]
    pub fn held(&self) -> &str {
        &self.held
    }

    /// Whether a stop string has matched.
    #[must_use]
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Feeds released text and returns what can be delivered.
    pub fn push(&mut self, text: &str) -> Released {
        if self.stopped {
            return Released::default();
        }
        let start = self.held.len();
        self.held.push_str(text);
        // A match must end in the new text: the held text holds none (it is a proper prefix of a
        // stop string), and every released byte is outside every partial match.
        if let Some((at, index)) = first_match(&self.stops, &self.held, start + 1) {
            self.held.truncate(at);
            self.stopped = true;
            return Released {
                text: std::mem::take(&mut self.held),
                stop: Some(StopMatch { index }),
            };
        }
        let keep = partial_suffix(&self.stops, &self.held);
        let rest = self.held.split_off(self.held.len() - keep);
        Released {
            text: std::mem::replace(&mut self.held, rest),
            stop: None,
        }
    }

    /// Releases the held text at the end of the choice, when no stop string matched.
    pub fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }
}

/// The first stop string to complete in `text` at a byte end of `min_end` or later: the smallest
/// end, then the earliest start. Returns the start in bytes and the stop's index.
fn first_match(stops: &[String], text: &str, min_end: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    (min_end.max(1)..=bytes.len()).find_map(|end| {
        stops
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_empty() && bytes[..end].ends_with(s.as_bytes()))
            .map(|(i, s)| (end - s.len(), i))
            .min()
    })
}

/// The length of the longest suffix of `text` that is a proper prefix of some stop string.
fn partial_suffix(stops: &[String], text: &str) -> usize {
    let bytes = text.as_bytes();
    stops
        .iter()
        .flat_map(|s| {
            let s = s.as_bytes();
            (1..s.len().min(bytes.len() + 1)).filter(move |&k| bytes.ends_with(&s[..k]))
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
