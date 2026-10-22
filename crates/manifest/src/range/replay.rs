// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recorded Hub responses, replayed as a [`RangeSource`] (RFC-0041 §2; COMP-E10-T01).
//!
//! A recording is the list of HTTP range responses a reader received for one file. Replaying
//! it serves the same bytes without a network, so a reader's tests and fuzz seeds cover the
//! Hub path. [`Recorder`] captures the requests a reader issues against any source, and
//! [`Recording::to_text`] writes them in the format below.
//!
//! **Format** (UTF-8 text, one item per line; `.hubrec` by convention):
//!
//! ```text
//! syntropy-range-recording 1
//! # Lines starting with '#' are comments; blank lines are ignored.
//! source <free text: the resolve URL it was recorded from, or "synthetic:<name>">
//! length <the file's length in bytes, decimal>
//! range <first>-<last> <body as lower-case hex>
//! range ...
//! ```
//!
//! - The first line is the magic and the version, exactly as shown.
//! - `source` and `length` come once each, before any `range`.
//! - Each `range` is one `206 Partial Content` response to `Range: bytes=<first>-<last>`.
//!   Both ends are inclusive, as in the HTTP header, and the body holds exactly
//!   `last − first + 1` bytes, inside `length`.
//! - Ranges may overlap, but where they do their bytes must agree.
//!
//! A read is served when the recorded ranges cover all of it. Any other read is
//! [`RangeError::NotRecorded`]: the fixture does not match the reader.

use std::cell::RefCell;
use std::fmt::Write as _;

use super::{RangeError, RangeSource, check_bounds, len_u64};

const MAGIC: &str = "syntropy-range-recording 1";

/// A recorded file: its source, its length and the range responses received.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Recording {
    /// Where it was recorded from.
    pub source: String,
    /// The file's length.
    pub length: u64,
    /// The responses, in the order received: (first byte, body).
    pub responses: Vec<(u64, Vec<u8>)>,
}

fn bad(line: usize, message: impl Into<String>) -> RangeError {
    RangeError::Recording {
        line,
        message: message.into(),
    }
}

fn parse_hex(line: usize, hex: &str) -> Result<Vec<u8>, RangeError> {
    if !hex.len().is_multiple_of(2) {
        return Err(bad(line, "odd number of hex digits"));
    }
    let digit = |c: u8| match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        _ => Err(bad(line, "body is not lower-case hex")),
    };
    hex.as_bytes()
        .chunks(2)
        .map(|p| Ok(digit(p[0])? << 4 | digit(p[1])?))
        .collect()
}

impl Recording {
    /// Parses a recording in the format of the module docs.
    ///
    /// # Errors
    ///
    /// [`RangeError::Recording`] naming the first malformed line.
    pub fn parse(text: &str) -> Result<Self, RangeError> {
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, l.trim_end()))
            .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'));
        match lines.next() {
            Some((_, MAGIC)) => {}
            Some((n, _)) => return Err(bad(n, format!("expected `{MAGIC}`"))),
            None => return Err(bad(1, "empty recording")),
        }
        let mut source = None;
        let mut length = None;
        let mut responses: Vec<(u64, Vec<u8>)> = Vec::new();
        for (n, line) in lines {
            let (key, rest) = line.split_once(' ').unwrap_or((line, ""));
            match key {
                "source" if source.is_none() && responses.is_empty() => {
                    source = Some(rest.to_owned());
                }
                "length" if length.is_none() && responses.is_empty() => {
                    length = Some(rest.parse::<u64>().map_err(|_| bad(n, "bad length"))?);
                }
                "range" => {
                    let len = length.ok_or_else(|| bad(n, "`range` before `length`"))?;
                    if source.is_none() {
                        return Err(bad(n, "`range` before `source`"));
                    }
                    let (span, hex) = rest
                        .split_once(' ')
                        .ok_or_else(|| bad(n, "expected `range <first>-<last> <hex>`"))?;
                    let (first, last) = span
                        .split_once('-')
                        .and_then(|(a, b)| Some((a.parse::<u64>().ok()?, b.parse::<u64>().ok()?)))
                        .ok_or_else(|| bad(n, "bad span"))?;
                    let body = parse_hex(n, hex)?;
                    if last < first || last - first + 1 != len_u64(&body) {
                        return Err(bad(n, "body length differs from the span"));
                    }
                    if last >= len {
                        return Err(bad(n, "span passes the file's length"));
                    }
                    for (start, prev) in &responses {
                        if let Some(conflict) = overlap_conflict(*start, prev, first, &body) {
                            return Err(bad(
                                n,
                                format!("disagrees with an earlier response at byte {conflict}"),
                            ));
                        }
                    }
                    responses.push((first, body));
                }
                _ => return Err(bad(n, format!("unexpected `{key}`"))),
            }
        }
        Ok(Self {
            source: source.ok_or_else(|| bad(1, "no `source`"))?,
            length: length.ok_or_else(|| bad(1, "no `length`"))?,
            responses,
        })
    }

    /// Writes the recording in the format of the module docs.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!("{MAGIC}\nsource {}\nlength {}\n", self.source, self.length);
        for (first, body) in &self.responses {
            let last = first + len_u64(body) - 1;
            let _ = write!(out, "range {first}-{last} ");
            for b in body {
                let _ = write!(out, "{b:02x}");
            }
            out.push('\n');
        }
        out
    }
}

/// The first byte where two responses overlap and differ.
fn overlap_conflict(a_start: u64, a: &[u8], b_start: u64, b: &[u8]) -> Option<u64> {
    let start = a_start.max(b_start);
    let end = (a_start + len_u64(a)).min(b_start + len_u64(b));
    (start..end).find(|&i| {
        let ia = usize::try_from(i - a_start).unwrap_or(usize::MAX);
        let ib = usize::try_from(i - b_start).unwrap_or(usize::MAX);
        a[ia] != b[ib]
    })
}

/// A [`Recording`] served as a [`RangeSource`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replay {
    recording: Recording,
}

impl Replay {
    /// Replays `recording`.
    #[must_use]
    pub fn new(recording: Recording) -> Self {
        Self { recording }
    }

    /// Parses and replays a recording.
    ///
    /// # Errors
    ///
    /// As [`Recording::parse`].
    pub fn parse(text: &str) -> Result<Self, RangeError> {
        Recording::parse(text).map(Self::new)
    }

    /// The recording.
    #[must_use]
    pub fn recording(&self) -> &Recording {
        &self.recording
    }
}

impl RangeSource for Replay {
    fn len(&self) -> u64 {
        self.recording.length
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        let len = len_u64(buf);
        check_bounds(offset, len, self.recording.length)?;
        let mut pos = offset;
        let end = offset + len;
        while pos < end {
            // The response covering `pos` that reaches furthest.
            let (start, body) = self
                .recording
                .responses
                .iter()
                .filter(|(s, b)| *s <= pos && pos < s + len_u64(b))
                .max_by_key(|(s, b)| s + len_u64(b))
                .ok_or(RangeError::NotRecorded { offset, len })?;
            let stop = (start + len_u64(body)).min(end);
            let from = usize::try_from(pos - start).unwrap_or(usize::MAX);
            let to = usize::try_from(stop - start).unwrap_or(usize::MAX);
            let dst = usize::try_from(pos - offset).unwrap_or(usize::MAX);
            buf[dst..dst + (to - from)].copy_from_slice(&body[from..to]);
            pos = stop;
        }
        Ok(())
    }
}

/// Wraps a source and records every successful read, to make a [`Recording`] of exactly the
/// requests a reader issues.
#[derive(Debug)]
pub struct Recorder<S> {
    inner: S,
    source: String,
    responses: RefCell<Vec<(u64, Vec<u8>)>>,
}

impl<S: RangeSource> Recorder<S> {
    /// Records reads of `inner`, naming it `source` in the recording.
    pub fn new(inner: S, source: impl Into<String>) -> Self {
        Self {
            inner,
            source: source.into(),
            responses: RefCell::new(Vec::new()),
        }
    }

    /// The recording so far.
    #[must_use]
    pub fn recording(&self) -> Recording {
        Recording {
            source: self.source.clone(),
            length: self.inner.len(),
            responses: self.responses.borrow().clone(),
        }
    }
}

impl<S: RangeSource> RangeSource for Recorder<S> {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        self.inner.read_at(offset, buf)?;
        if !buf.is_empty() {
            self.responses.borrow_mut().push((offset, buf.to_vec()));
        }
        Ok(())
    }
}
