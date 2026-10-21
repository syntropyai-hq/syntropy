// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error the [`KernelProvider`](crate::KernelProvider) methods return.
//!
//! RFC-0001 §5.12 defines one `Error` struct in `core` for every fallible interface, with an
//! `ErrorCode` of `(Domain, u16)`. `core` has [`ErrorClass`] but not that struct yet, so this
//! module holds the smallest stand-in the trait needs: a class, a code in the `Kernels`
//! domain, and a message. It gives way to `core`'s `Error` when that type is built. Raising
//! each code of RFC-0040 §13 is COMP-E1-T08's work.

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
