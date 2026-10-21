// SPDX-License-Identifier: MIT OR Apache-2.0

//! Errors from the I/O interface: the `Io` domain codes of RFC-0021 §16.
//!
//! [`IoError`] is `Copy` and holds no heap data, so a failed read travels in a
//! [`ReadCompletion`](crate::ReadCompletion) through the preallocated buffers without
//! allocating. Converting it into the common `Error` of RFC-0001 §5.12, with the class each
//! code's documentation names, belongs with that type in `syntropy-core`.

use std::fmt;

/// A code in the `Io` error domain (RFC-0021 §16).
///
/// Each variant's number is its stable code. Code 9, `IO_GDS_FALLBACK`, was retired with the
/// GPUDirect Storage path (RFC-0021 §9) and stays reserved, so no variant uses it.
#[non_exhaustive]
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IoErrorCode {
    /// `IO_ENGINE_UNAVAILABLE`, class `Unsupported`: ring setup fails with `ENOSYS`/`EPERM`, or
    /// IoRing is missing. The next engine in the fallback chain is used.
    EngineUnavailable = 1,
    /// `IO_DIRECT_REFUSED`, class `Unsupported`: `O_DIRECT` or an unbuffered open or read is
    /// refused. The volume goes to flagged buffered mode (RFC-0021 §4.2).
    DirectRefused = 2,
    /// `IO_REGISTER_FAILED`, class `Capacity`: fixed-buffer or file registration fails.
    /// Operations continue unregistered.
    RegisterFailed = 3,
    /// `IO_SHORT_READ`, class `Transient`: a short read, `EIO` or `ENODATA`. Retried once, then
    /// `IO_EXTENT_BAD`.
    ShortRead = 4,
    /// `IO_EXTENT_BAD`, class `Integrity`: repeated read failure or a hash mismatch
    /// (RFC-0021 §14). The extent is marked bad, rebuilt and never served.
    ExtentBad = 5,
    /// `IO_NO_SPACE`, class `Capacity`: `ENOSPC` on a pack, spill or session write.
    NoSpace = 6,
    /// `IO_MISALIGNED`, class `Internal`: a request violates RFC-0021 §6. Rejected before
    /// submission and counted; a bug.
    Misaligned = 7,
    /// `IO_THROTTLED`, class `Drift`: the volume's rate fell under thermal throttling
    /// (RFC-0021 §11).
    Throttled = 8,
    /// `IO_CANCELLED`, class `Cancelled`: a prefetch cancelled or dropped as no longer useful.
    /// The completion is discarded.
    Cancelled = 10,
}

impl IoErrorCode {
    /// Every code, in numeric order.
    pub const ALL: [IoErrorCode; 9] = [
        IoErrorCode::EngineUnavailable,
        IoErrorCode::DirectRefused,
        IoErrorCode::RegisterFailed,
        IoErrorCode::ShortRead,
        IoErrorCode::ExtentBad,
        IoErrorCode::NoSpace,
        IoErrorCode::Misaligned,
        IoErrorCode::Throttled,
        IoErrorCode::Cancelled,
    ];

    /// The stable numeric code.
    #[must_use]
    pub const fn code(self) -> u16 {
        self as u16
    }

    /// The code's name as RFC-0021 §16 writes it, such as `IO_SHORT_READ`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            IoErrorCode::EngineUnavailable => "IO_ENGINE_UNAVAILABLE",
            IoErrorCode::DirectRefused => "IO_DIRECT_REFUSED",
            IoErrorCode::RegisterFailed => "IO_REGISTER_FAILED",
            IoErrorCode::ShortRead => "IO_SHORT_READ",
            IoErrorCode::ExtentBad => "IO_EXTENT_BAD",
            IoErrorCode::NoSpace => "IO_NO_SPACE",
            IoErrorCode::Misaligned => "IO_MISALIGNED",
            IoErrorCode::Throttled => "IO_THROTTLED",
            IoErrorCode::Cancelled => "IO_CANCELLED",
        }
    }

    /// The variant with this numeric code, or `None` for an unknown or reserved code.
    #[must_use]
    pub const fn from_code(code: u16) -> Option<Self> {
        Some(match code {
            1 => IoErrorCode::EngineUnavailable,
            2 => IoErrorCode::DirectRefused,
            3 => IoErrorCode::RegisterFailed,
            4 => IoErrorCode::ShortRead,
            5 => IoErrorCode::ExtentBad,
            6 => IoErrorCode::NoSpace,
            7 => IoErrorCode::Misaligned,
            8 => IoErrorCode::Throttled,
            10 => IoErrorCode::Cancelled,
            _ => return None,
        })
    }
}

/// An I/O failure: its `Io` domain code and, where the OS reported one, the OS error number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IoError {
    /// What went wrong.
    pub code: IoErrorCode,
    /// The OS error, as `std::io::Error::raw_os_error` reports it, when one caused this.
    pub os_error: Option<i32>,
}

impl IoError {
    /// An error with no OS error number.
    #[must_use]
    pub const fn new(code: IoErrorCode) -> Self {
        Self {
            code,
            os_error: None,
        }
    }

    /// An error caused by an OS error number.
    #[must_use]
    pub const fn with_os_error(code: IoErrorCode, os_error: i32) -> Self {
        Self {
            code,
            os_error: Some(os_error),
        }
    }
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.code.name(), self.code.code())?;
        if let Some(os) = self.os_error {
            write!(f, ", os error {os}")?;
        }
        Ok(())
    }
}

impl std::error::Error for IoError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC-0021 §16's table: name and number of every code.
    const TABLE: [(&str, u16); 9] = [
        ("IO_ENGINE_UNAVAILABLE", 1),
        ("IO_DIRECT_REFUSED", 2),
        ("IO_REGISTER_FAILED", 3),
        ("IO_SHORT_READ", 4),
        ("IO_EXTENT_BAD", 5),
        ("IO_NO_SPACE", 6),
        ("IO_MISALIGNED", 7),
        ("IO_THROTTLED", 8),
        ("IO_CANCELLED", 10),
    ];

    #[test]
    fn codes_match_rfc_0021_section_16() {
        for (code, (name, number)) in IoErrorCode::ALL.iter().zip(TABLE) {
            assert_eq!(code.name(), name);
            assert_eq!(code.code(), number);
            assert_eq!(IoErrorCode::from_code(number), Some(*code));
        }
    }

    #[test]
    fn reserved_and_unknown_codes_decode_to_none() {
        assert_eq!(IoErrorCode::from_code(0), None);
        assert_eq!(
            IoErrorCode::from_code(9),
            None,
            "IO_GDS_FALLBACK stays reserved"
        );
        assert_eq!(IoErrorCode::from_code(11), None);
        assert_eq!(IoErrorCode::from_code(u16::MAX), None);
    }

    #[test]
    fn display_names_the_code_and_the_os_error() {
        assert_eq!(
            IoError::new(IoErrorCode::Misaligned).to_string(),
            "IO_MISALIGNED (7)"
        );
        assert_eq!(
            IoError::with_os_error(IoErrorCode::ShortRead, 5).to_string(),
            "IO_SHORT_READ (4), os error 5"
        );
    }
}
