// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error the [`KernelProvider`](crate::KernelProvider) methods return: `core`'s
//! [`Error`], RFC-0001 §5.12's one error type for every fallible interface (COMP-E13-T15).
//!
//! A provider's own failures carry codes of the `Kernels` domain, RFC-0040 §13's codes 1–17,
//! which [`codes`] lists. An `Error` may also carry a code of another domain, such as an
//! `Exec` or `Net` code; nothing here narrows it. Raising each code of RFC-0040 §13 is
//! COMP-E1-T08's work.

use syntropy_core::ErrorClass;
pub use syntropy_core::error::codes::kernels as codes;
pub use syntropy_core::{Error, ErrorCode};

/// The error a default method body returns when the provider does not implement it:
/// class `Unsupported`, code `(Kernels, 1)` (RFC-0040 §2.1).
#[must_use]
pub fn unsupported() -> Error {
    Error::new(
        ErrorClass::Unsupported,
        codes::OP_UNSUPPORTED,
        "not supported by this provider",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_shows_message_class_and_code() {
        assert_eq!(
            unsupported().to_string(),
            "not supported by this provider (Unsupported, Kernels 0x0001)"
        );
        let boxed: Box<dyn std::error::Error> = Box::new(unsupported());
        assert!(boxed.source().is_none());
    }

    /// `kernels`' `Error` and `ErrorCode` are `core`'s, not types of their own: a value of
    /// either crate's name passes where the other's is expected (COMP-E13-T15). A compile-time
    /// check; the assertion only uses the values.
    #[test]
    fn the_error_is_cores() {
        fn to_core(
            e: crate::Error,
            c: crate::ErrorCode,
        ) -> (syntropy_core::Error, syntropy_core::ErrorCode) {
            (e, c)
        }
        fn from_core(e: syntropy_core::Error) -> crate::Error {
            e
        }
        let (e, c) = to_core(unsupported(), codes::OP_UNSUPPORTED);
        assert_eq!(from_core(e).code, c);
    }
}
