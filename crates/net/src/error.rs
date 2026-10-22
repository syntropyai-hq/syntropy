// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error `net`'s fallible interfaces return, [`Transport`](crate::transport::Transport)'s
//! among them (RFC-0001 §5.12; codes in RFC-0031 §12).
//!
//! RFC-0001 §5.12 defines one `Error` struct in `core` for every fallible interface, with an
//! `ErrorCode` of `(Domain, u16)`. `core` has [`ErrorClass`] but not that struct yet, so this
//! module holds the smallest stand-in `net` needs, as `kernels::Error` does for the kernel
//! providers: a class, a code in the `Net` domain, and a message. It gives way to `core`'s
//! `Error` when that type is built; no task builds it yet, and issue #1525 (item 2) asks for one.
//!
//! The code is optional. RFC-0031 §12 gives some refusals a class and no code of their own, such
//! as a wire dtype the build does not carry (`Unsupported`), and so does a backend that does not
//! implement an optional method.

use std::fmt;

use syntropy_core::ErrorClass;

use crate::frame::code;

/// A code in the `Net` domain, 0x0200–0x02FF: the `n` of RFC-0031 §12's `(Domain::Net, n)`. The
/// same values travel in `HelloAck::status` and `Close::code` ([`crate::frame::code`]). Codes are
/// never reused (RFC-0001 §10.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErrorCode(pub u16);

impl ErrorCode {
    pub const PROTO_MAJOR_MISMATCH: Self = Self(code::PROTO_MAJOR_MISMATCH);
    pub const CREDENTIAL_INVALID: Self = Self(code::CREDENTIAL_INVALID);
    pub const NOT_A_NEIGHBOUR: Self = Self(code::NOT_A_NEIGHBOUR);
    pub const UNKNOWN_FRAME_TYPE: Self = Self(code::UNKNOWN_FRAME_TYPE);
    pub const FRAME_TOO_LARGE: Self = Self(code::FRAME_TOO_LARGE);
    pub const WIRE_DTYPE_MISMATCH: Self = Self(code::WIRE_DTYPE_MISMATCH);
    pub const CRC_STORM: Self = Self(code::CRC_STORM);
    pub const ACK_TIMEOUT: Self = Self(code::ACK_TIMEOUT);
    pub const QP_ERROR: Self = Self(code::QP_ERROR);
    pub const CONNECT_TIMEOUT: Self = Self(code::CONNECT_TIMEOUT);
    pub const PEER_RESET: Self = Self(code::PEER_RESET);
    pub const EDGE_DEMOTED: Self = Self(code::EDGE_DEMOTED);
    pub const RDMA_NOT_ENABLED: Self = Self(code::RDMA_NOT_ENABLED);
    pub const RDMA_OS_BUILD_MISMATCH: Self = Self(code::RDMA_OS_BUILD_MISMATCH);
    pub const SHM_USER_MISMATCH: Self = Self(code::SHM_USER_MISMATCH);
    /// `BlobNotFound`, `BlobBusy` and `BlobThrottled` share this code.
    pub const BLOB_UNAVAILABLE: Self = Self(code::BLOB_UNAVAILABLE);
    pub const EPOCH_NOT_SERVED: Self = Self(code::EPOCH_NOT_SERVED);
    pub const PROTO_RANGE_DISJOINT: Self = Self(code::PROTO_RANGE_DISJOINT);
    pub const QP_BUDGET_EXHAUSTED: Self = Self(code::QP_BUDGET_EXHAUSTED);

    /// Every code of RFC-0031 §12, in code order.
    pub const ALL: [ErrorCode; 19] = [
        Self::PROTO_MAJOR_MISMATCH,
        Self::CREDENTIAL_INVALID,
        Self::NOT_A_NEIGHBOUR,
        Self::UNKNOWN_FRAME_TYPE,
        Self::FRAME_TOO_LARGE,
        Self::WIRE_DTYPE_MISMATCH,
        Self::CRC_STORM,
        Self::ACK_TIMEOUT,
        Self::QP_ERROR,
        Self::CONNECT_TIMEOUT,
        Self::PEER_RESET,
        Self::EDGE_DEMOTED,
        Self::RDMA_NOT_ENABLED,
        Self::RDMA_OS_BUILD_MISMATCH,
        Self::SHM_USER_MISMATCH,
        Self::BLOB_UNAVAILABLE,
        Self::EPOCH_NOT_SERVED,
        Self::PROTO_RANGE_DISJOINT,
        Self::QP_BUDGET_EXHAUSTED,
    ];

    /// The class RFC-0031 §12's table gives the code, with its revision notes of 2026-10-22;
    /// `None` for a value outside the table.
    pub fn class(self) -> Option<ErrorClass> {
        use ErrorClass::{Auth, Internal, Link, Transient, Unsupported};
        Some(match self.0 {
            code::PROTO_MAJOR_MISMATCH => Unsupported,
            code::CREDENTIAL_INVALID | code::NOT_A_NEIGHBOUR => Auth,
            code::UNKNOWN_FRAME_TYPE | code::WIRE_DTYPE_MISMATCH => Unsupported,
            code::FRAME_TOO_LARGE => Internal,
            code::CRC_STORM
            | code::ACK_TIMEOUT
            | code::QP_ERROR
            | code::CONNECT_TIMEOUT
            | code::PEER_RESET
            | code::EDGE_DEMOTED
            | code::QP_BUDGET_EXHAUSTED => Link,
            code::RDMA_NOT_ENABLED
            | code::RDMA_OS_BUILD_MISMATCH
            | code::SHM_USER_MISMATCH
            | code::PROTO_RANGE_DISJOINT => Unsupported,
            code::BLOB_UNAVAILABLE | code::EPOCH_NOT_SERVED => Transient,
            _ => return None,
        })
    }
}

/// A failure of a `net` call (RFC-0001 §5.12; codes in RFC-0031 §12).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// What kind of failure it is; RFC-0001 §10 says what each class triggers.
    pub class: ErrorClass,
    /// Stable, machine-readable code in the `Net` domain; `None` for a refusal RFC-0031 §12
    /// gives no code of its own.
    pub code: Option<ErrorCode>,
    /// For people; never parsed.
    pub message: String,
}

impl Error {
    /// An error with an RFC-0031 §12 code, classed as the table classes it.
    ///
    /// # Panics
    ///
    /// If `code` is not in the table: a code is a constant of [`ErrorCode`], so that is a bug.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Error {
        let class = code
            .class()
            .unwrap_or_else(|| panic!("{:#06x} is not an RFC-0031 §12 code", code.0));
        Error {
            class,
            code: Some(code),
            message: message.into(),
        }
    }

    /// An error with a class and no code of its own.
    pub fn uncoded(class: ErrorClass, message: impl Into<String>) -> Error {
        Error {
            class,
            code: None,
            message: message.into(),
        }
    }

    /// What an optional method returns when the backend does not implement it: class
    /// `Unsupported`, no code (RFC-0031 §2's default `reconnect`).
    pub fn unsupported() -> Error {
        Error::uncoded(
            ErrorClass::Unsupported,
            "not supported by this transport backend",
        )
    }
}

/// Shows the message, then the class and the code, e.g.
/// `connect timed out (Link, Net 0x020a)` or `not supported (Unsupported)`.
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(
                f,
                "{} ({:?}, Net {:#06x})",
                self.message, self.class, code.0
            ),
            None => write!(f, "{} ({:?})", self.message, self.class),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_has_its_rfc_0031_class() {
        let expected = [
            (0x0201, ErrorClass::Unsupported),
            (0x0202, ErrorClass::Auth),
            (0x0203, ErrorClass::Auth),
            (0x0204, ErrorClass::Unsupported),
            (0x0205, ErrorClass::Internal),
            (0x0206, ErrorClass::Unsupported),
            (0x0207, ErrorClass::Link),
            (0x0208, ErrorClass::Link),
            (0x0209, ErrorClass::Link),
            (0x020A, ErrorClass::Link),
            (0x020B, ErrorClass::Link),
            (0x020C, ErrorClass::Link),
            (0x020D, ErrorClass::Unsupported),
            (0x020E, ErrorClass::Unsupported),
            (0x020F, ErrorClass::Unsupported),
            (0x0210, ErrorClass::Transient),
            (0x0211, ErrorClass::Transient),
            (0x0212, ErrorClass::Unsupported),
            (0x0213, ErrorClass::Link),
        ];
        assert_eq!(ErrorCode::ALL.len(), expected.len());
        for (code, (value, class)) in ErrorCode::ALL.into_iter().zip(expected) {
            assert_eq!(code.0, value);
            assert_eq!(code.class(), Some(class), "{value:#06x}");
        }
        assert_eq!(ErrorCode(0x0200).class(), None);
        assert_eq!(ErrorCode(0x0214).class(), None);
    }

    #[test]
    fn display_names_the_class_and_the_code() {
        let timeout = Error::new(ErrorCode::CONNECT_TIMEOUT, "connect timed out");
        assert_eq!(timeout.class, ErrorClass::Link);
        assert_eq!(timeout.to_string(), "connect timed out (Link, Net 0x020a)");
        assert_eq!(
            Error::unsupported().to_string(),
            "not supported by this transport backend (Unsupported)"
        );
    }

    #[test]
    #[should_panic(expected = "is not an RFC-0031 §12 code")]
    fn a_code_outside_the_table_is_a_bug() {
        let _ = Error::new(ErrorCode(0x02FF), "no such code");
    }
}
