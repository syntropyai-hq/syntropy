// SPDX-License-Identifier: MIT OR Apache-2.0

//! `RangeSource`: the byte source every reader parses through, so one parser reads a local
//! file, a sparse blob or a Hub file (RFC-0041 §2).
//!
//! - [`RangeSource`] is RFC-0041 §2's trait.
//! - [`FileSource`] reads a local file.
//! - [`replay::Replay`] serves a recorded Hub fixture, so tests and fuzz seeds read Hub
//!   responses without a network. The HTTP range source against the live Hub is COMP-E10-T14,
//!   which waits on the maintainers' choice of HTTPS client (OQ-MF-18).
//! - [`HeaderReader`] is the one header-read routine. It bounds every read by
//!   `store.header_max_bytes` ([`HeaderLimit`](RangeError::HeaderLimit), RFC-0041 §10 code 5)
//!   and coalesces the parser's many small reads into a few requests against the source: it
//!   keeps the prefix of the file read so far, and extends it by at least doubling, starting
//!   at [`HeaderReader::INITIAL_READ`] (RFC-0041 §7.2: "the first 4 MiB of a GGUF, extended as
//!   needed").
//! - [`coalesce`] merges a set of byte ranges into as few requests as a gap and a size bound
//!   allow, for callers that know their reads up front.

use std::fmt;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::Path;

use syntropy_core::ErrorClass;

pub mod replay;

/// `store.header_max_bytes`' default: 1 GiB (RFC-0041 §9).
pub const HEADER_MAX_BYTES_DEFAULT: u64 = 1 << 30;

/// A byte source a reader parses through (RFC-0041 §2).
pub trait RangeSource {
    /// The source's length in bytes.
    fn len(&self) -> u64;

    /// Whether the source is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads exactly `buf.len()` bytes at `offset`.
    ///
    /// # Errors
    ///
    /// [`RangeError::OutOfRange`] when `offset + buf.len()` passes [`len`](Self::len), and the
    /// source's own failures.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError>;
}

impl<S: RangeSource + ?Sized> RangeSource for &S {
    fn len(&self) -> u64 {
        (**self).len()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        (**self).read_at(offset, buf)
    }
}

/// A failed read (RFC-0041 §10 where it names a code).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RangeError {
    /// `HeaderLimit`, `(Formats, 5)`, class `Integrity`: a header read reaches past
    /// `store.header_max_bytes`. The file is treated as malformed.
    HeaderLimit {
        /// The end of the read.
        end: u64,
        /// The cap.
        cap: u64,
    },
    /// `HeaderInvalid`, `(Formats, 6)`, class `Integrity`: the read is outside the source.
    OutOfRange {
        /// The read's offset.
        offset: u64,
        /// The read's length.
        len: u64,
        /// The source's length.
        source_len: u64,
    },
    /// The OS failed a read of a local file. RFC-0041 §10 names no code for it; the file is
    /// treated as unreadable, class `Integrity`.
    Io {
        /// The OS error's kind.
        kind: io::ErrorKind,
        /// The OS error's message.
        message: String,
    },
    /// A replayed recording holds no response covering the read: the fixture does not match
    /// the reader. Class `Internal`.
    NotRecorded {
        /// The read's offset.
        offset: u64,
        /// The read's length.
        len: u64,
    },
    /// A recording is malformed. Class `Internal`.
    Recording {
        /// The 1-based line.
        line: usize,
        /// What is wrong.
        message: String,
    },
}

impl RangeError {
    /// The `Formats` code RFC-0041 §10 gives it, if any.
    #[must_use]
    pub fn code(&self) -> Option<u16> {
        match self {
            Self::HeaderLimit { .. } => Some(5),
            Self::OutOfRange { .. } => Some(6),
            _ => None,
        }
    }

    /// Its class (RFC-0001 §10).
    #[must_use]
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::HeaderLimit { .. } | Self::OutOfRange { .. } | Self::Io { .. } => {
                ErrorClass::Integrity
            }
            Self::NotRecorded { .. } | Self::Recording { .. } => ErrorClass::Internal,
        }
    }
}

impl fmt::Display for RangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderLimit { end, cap } => write!(
                f,
                "HeaderLimit (Formats 5): a header read reaches byte {end}, past store.header_max_bytes = {cap}"
            ),
            Self::OutOfRange {
                offset,
                len,
                source_len,
            } => write!(
                f,
                "HeaderInvalid (Formats 6): {len} bytes at {offset} pass the source's end at {source_len}"
            ),
            Self::Io { kind, message } => write!(f, "read failed ({kind:?}): {message}"),
            Self::NotRecorded { offset, len } => {
                write!(
                    f,
                    "the recording holds no response for {len} bytes at {offset}"
                )
            }
            Self::Recording { line, message } => {
                write!(f, "malformed recording, line {line}: {message}")
            }
        }
    }
}

impl std::error::Error for RangeError {}

/// Checks that `len` bytes at `offset` lie inside a source of `source_len` bytes.
///
/// # Errors
///
/// [`RangeError::OutOfRange`] when they do not.
pub fn check_bounds(offset: u64, len: u64, source_len: u64) -> Result<(), RangeError> {
    match offset.checked_add(len) {
        Some(end) if end <= source_len => Ok(()),
        _ => Err(RangeError::OutOfRange {
            offset,
            len,
            source_len,
        }),
    }
}

fn len_u64(buf: &[u8]) -> u64 {
    u64::try_from(buf.len()).unwrap_or(u64::MAX)
}

/// A local file (RFC-0041 §2).
#[derive(Debug)]
pub struct FileSource {
    file: File,
    len: u64,
}

impl FileSource {
    /// Opens `path` and records its length.
    ///
    /// # Errors
    ///
    /// [`RangeError::Io`] when the file cannot be opened or its length read.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RangeError> {
        let file = File::open(path).map_err(io_error)?;
        let len = file.metadata().map_err(io_error)?.len();
        Ok(Self { file, len })
    }
}

fn io_error(e: io::Error) -> RangeError {
    RangeError::Io {
        kind: e.kind(),
        message: e.to_string(),
    }
}

impl RangeSource for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        check_bounds(offset, len_u64(buf), self.len)?;
        read_exact_at(&self.file, offset, buf).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                // The file shrank since it was opened.
                RangeError::OutOfRange {
                    offset,
                    len: len_u64(buf),
                    source_len: self.len,
                }
            } else {
                io_error(e)
            }
        })
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let at = offset.saturating_add(u64::try_from(done).unwrap_or(u64::MAX));
        match file.seek_read(&mut buf[done..], at) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Merges `ranges` into as few requests as possible: sorted by start, a range joins the
/// request before it when it starts at most `max_gap` bytes after that request's end and the
/// merged request stays at most `max_len` bytes long. A range that overlaps a request it cannot
/// join is trimmed to the bytes not yet requested, so no byte is requested twice. A single
/// range longer than `max_len` stays one request. Empty ranges are dropped.
#[must_use]
pub fn coalesce(ranges: &[Range<u64>], max_gap: u64, max_len: u64) -> Vec<Range<u64>> {
    let mut sorted: Vec<Range<u64>> = ranges.iter().filter(|r| r.start < r.end).cloned().collect();
    sorted.sort_by_key(|r| (r.start, r.end));
    let mut out: Vec<Range<u64>> = Vec::with_capacity(sorted.len());
    for r in sorted {
        if let Some(last) = out.last_mut() {
            if r.end <= last.end {
                // Already covered.
                continue;
            }
            if r.start <= last.end.saturating_add(max_gap) && r.end - last.start <= max_len {
                last.end = r.end;
                continue;
            }
            // Too long to merge: request only the part not already covered.
            let start = r.start.max(last.end);
            out.push(start..r.end);
            continue;
        }
        out.push(r);
    }
    out
}

/// The one header-read routine (RFC-0041 §2, §7.2): every read is bounded by
/// `store.header_max_bytes`, and reads are served from the prefix of the source read so far,
/// which grows by coalesced requests.
#[derive(Debug)]
pub struct HeaderReader<S> {
    source: S,
    cap: u64,
    initial: u64,
    prefix: Vec<u8>,
    requests: u32,
}

impl<S: RangeSource> HeaderReader<S> {
    /// The first request's length: 4 MiB (RFC-0041 §7.2).
    pub const INITIAL_READ: u64 = 4 << 20;

    /// A reader over `source`, with `cap` the `store.header_max_bytes` in force.
    pub fn new(source: S, cap: u64) -> Self {
        Self {
            source,
            cap,
            initial: Self::INITIAL_READ,
            prefix: Vec::new(),
            requests: 0,
        }
    }

    /// Sets the first request's length (at least 1), in place of
    /// [`INITIAL_READ`](Self::INITIAL_READ).
    #[must_use]
    pub fn with_initial_read(mut self, bytes: u64) -> Self {
        self.initial = bytes.max(1);
        self
    }

    /// The source.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// How many requests the reader has issued to the source.
    #[must_use]
    pub fn requests(&self) -> u32 {
        self.requests
    }

    /// The bytes `[offset, offset + len)` of the header.
    ///
    /// # Errors
    ///
    /// [`RangeError::HeaderLimit`] when the read reaches past the cap,
    /// [`RangeError::OutOfRange`] when it passes the source's end, and the source's errors.
    pub fn read(&mut self, offset: u64, len: u64) -> Result<&[u8], RangeError> {
        let end = offset.checked_add(len).ok_or(RangeError::HeaderLimit {
            end: u64::MAX,
            cap: self.cap,
        })?;
        if end > self.cap {
            return Err(RangeError::HeaderLimit { end, cap: self.cap });
        }
        let source_len = self.source.len();
        check_bounds(offset, len, source_len)?;
        let have = len_u64(&self.prefix);
        if end > have {
            // Extend the prefix by at least doubling it, within the cap and the source.
            let target = end
                .max(have.saturating_mul(2))
                .max(self.initial)
                .min(self.cap)
                .min(source_len);
            let extra = usize::try_from(target - have).map_err(|_| RangeError::HeaderLimit {
                end: target,
                cap: self.cap,
            })?;
            let start = self.prefix.len();
            // Exactly the target: the doubling above already sets the growth, and an amortised
            // reserve on top could hold up to twice the source's length.
            self.prefix.reserve_exact(extra);
            self.prefix.resize(start + extra, 0);
            if let Err(e) = self.source.read_at(have, &mut self.prefix[start..]) {
                self.prefix.truncate(start);
                return Err(e);
            }
            self.requests += 1;
        }
        let from = usize::try_from(offset).unwrap_or(usize::MAX);
        let to = usize::try_from(end).unwrap_or(usize::MAX);
        Ok(&self.prefix[from..to])
    }

    /// Reads into `buf` at `offset`, as [`read`](Self::read).
    ///
    /// # Errors
    ///
    /// As [`read`](Self::read).
    pub fn read_into(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), RangeError> {
        let bytes = self.read(offset, len_u64(buf))?;
        buf.copy_from_slice(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
