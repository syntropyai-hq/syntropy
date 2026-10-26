// SPDX-License-Identifier: MIT OR Apache-2.0

//! Enumeration errors.

use std::fmt;
use std::io;

/// Why enumeration failed.
///
/// The rule for files: a required file that is absent is [`EnumError::Missing`]; an optional
/// file that is absent means the feature is absent (no cgroup limit, no battery); any file that
/// exists but does not parse is [`EnumError::Malformed`]. Enumeration never guesses a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnumError {
    /// A file enumeration needs is absent.
    Missing {
        /// The path, relative to the filesystem root.
        path: String,
    },
    /// A file exists but could not be read.
    Io {
        /// The path, relative to the filesystem root.
        path: String,
        /// The I/O error's kind.
        kind: io::ErrorKind,
    },
    /// A file exists but its contents do not parse.
    Malformed {
        /// The path, relative to the filesystem root.
        path: String,
        /// What was wrong.
        what: String,
    },
    /// An x86-64 CPU below x86-64-v2, the baseline ADR-0020 supports.
    BelowBaseline {
        /// The x86-64-v2 features the CPU lacks, by `/proc/cpuinfo` name.
        missing: Vec<&'static str>,
    },
    /// The CPU architecture is not one Syntropy builds for.
    UnsupportedArch {
        /// What was found.
        found: String,
    },
    /// Live enumeration was asked for on an OS this module does not cover.
    UnsupportedOs,
    /// More devices or tiers than a `u16` ordinal can number.
    TooManyOrdinals,
    /// An OS call failed (Windows and macOS, whose E1 facts come from calls, not files).
    Os {
        /// The call, e.g. `GlobalMemoryStatusEx` or `sysctl hw.memsize`.
        call: String,
        /// The error's kind.
        kind: io::ErrorKind,
        /// The OS's message.
        message: String,
    },
    /// An OS call succeeded but its response lacks a value E1 needs or does not parse.
    BadResponse {
        /// The call.
        call: String,
        /// What was wrong.
        what: String,
    },
}

impl EnumError {
    /// Maps a read error on `path`: not-found becomes [`EnumError::Missing`].
    pub(crate) fn from_io(path: &str, err: &io::Error) -> Self {
        if err.kind() == io::ErrorKind::NotFound {
            Self::Missing { path: path.into() }
        } else {
            Self::Io {
                path: path.into(),
                kind: err.kind(),
            }
        }
    }

    /// Maps an OS call's error.
    pub(crate) fn os(call: impl Into<String>, err: &io::Error) -> Self {
        Self::Os {
            call: call.into(),
            kind: err.kind(),
            message: err.to_string(),
        }
    }

    pub(crate) fn bad(call: impl Into<String>, what: impl Into<String>) -> Self {
        Self::BadResponse {
            call: call.into(),
            what: what.into(),
        }
    }

    pub(crate) fn malformed(path: &str, what: impl Into<String>) -> Self {
        Self::Malformed {
            path: path.into(),
            what: what.into(),
        }
    }
}

impl fmt::Display for EnumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { path } => write!(f, "/{path} is missing"),
            Self::Io { path, kind } => write!(f, "cannot read /{path}: {kind}"),
            Self::Malformed { path, what } => write!(f, "/{path} is malformed: {what}"),
            Self::BelowBaseline { missing } => write!(
                f,
                "the CPU is below x86-64-v2; it lacks {}",
                missing.join(", ")
            ),
            Self::UnsupportedArch { found } => write!(f, "unsupported CPU architecture {found}"),
            Self::UnsupportedOs => f.write_str("host enumeration does not cover this OS"),
            Self::TooManyOrdinals => f.write_str("more than 65,536 devices or tiers on one node"),
            Self::Os { call, message, .. } => write!(f, "{call} failed: {message}"),
            Self::BadResponse { call, what } => write!(f, "{call} returned a bad response: {what}"),
        }
    }
}

impl std::error::Error for EnumError {}
