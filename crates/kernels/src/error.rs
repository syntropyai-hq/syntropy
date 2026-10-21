// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error the [`KernelProvider`](crate::KernelProvider) methods return.
//!
//! RFC-0001 §5.12 defines one `Error` struct in `core` for every fallible interface, with an
//! `ErrorCode` of `(Domain, u16)`. `core` has [`ErrorClass`] but not that struct yet, so this
//! module holds the smallest stand-in the trait needs: a class, a code in the `Kernels`
//! domain, and a message. It gives way to `core`'s `Error` when that type is built; no task
//! builds it yet, and issue #1525 (item 2) asks for one. Raising each code of RFC-0040 §13 is
//! COMP-E1-T08's work.

use std::fmt;

use serde::{Deserialize, Serialize};
use syntropy_core::ErrorClass;

/// A code in the `Kernels` domain: the `n` of RFC-0040 §13's `(Domain::Kernels, n)`. Codes
/// are never reused (RFC-0001 §10.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ErrorCode(pub u16);

impl ErrorCode {
    /// `OpUnsupported` (RFC-0040 §13 code 1): what [`unsupported`] carries.
    pub const OP_UNSUPPORTED: Self = Self(1);
    /// `WorkspaceExceeded` (RFC-0040 §13 code 14): a prepared op asks for more workspace than
    /// `query` declared ([`PreparedOp::check_workspace`](crate::PreparedOp::check_workspace)).
    pub const WORKSPACE_EXCEEDED: Self = Self(14);
}

/// A failure of a provider call (RFC-0001 §5.12; codes in RFC-0040 §13).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Error {
    /// What kind of failure it is; RFC-0001 §10 says what each class triggers.
    pub class: ErrorClass,
    /// Stable, machine-readable code in the `Kernels` domain.
    pub code: ErrorCode,
    /// For people; never parsed.
    pub message: String,
}

/// Shows the message, then the class and the code, e.g.
/// `not supported by this provider (Unsupported, Kernels 1)`.
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({:?}, Kernels {})",
            self.message, self.class, self.code.0
        )
    }
}

impl std::error::Error for Error {}

/// The error a default method body returns when the provider does not implement it:
/// class `Unsupported`, code `(Kernels, 1)` (RFC-0040 §2.1).
#[must_use]
pub fn unsupported() -> Error {
    Error {
        class: ErrorClass::Unsupported,
        code: ErrorCode::OP_UNSUPPORTED,
        message: String::from("not supported by this provider"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_shows_message_class_and_code() {
        assert_eq!(
            unsupported().to_string(),
            "not supported by this provider (Unsupported, Kernels 1)"
        );
        let boxed: Box<dyn std::error::Error> = Box::new(unsupported());
        assert!(boxed.source().is_none());
    }
}
