// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`SynStatus`], the result of every vtable function, and its mapping to [`Error`].

use syntropy_core::{Domain, ErrorClass};

use crate::error::{Error, ErrorCode};

/// The result of a boundary call (RFC-0040 §10.4). `class` 0 is success. A failure carries the
/// error's class and its code, with the code's domain as its boundary number
/// ([`DOMAIN_CODES`](super::DOMAIN_CODES)); the message is fetched with `last_error`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynStatus {
    /// `ErrorClass` code ([`class_code`]), 0 for success.
    pub class: u32,
    /// The code's domain, as its boundary number ([`DOMAIN_CODES`](super::DOMAIN_CODES)).
    pub domain: u32,
    /// The code in the domain.
    pub code: u16,
    /// 1 when retrying the call may succeed.
    pub retry: u8,
}

impl SynStatus {
    /// Success.
    pub const OK: Self = Self {
        class: 0,
        domain: super::DOMAIN_KERNELS,
        code: 0,
        retry: 0,
    };

    /// Whether the call succeeded.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        self.class == 0
    }
}

/// The ABI number of an error class: 1 for `InvalidRequest` through 13 for `Internal`, in the
/// order of RFC-0001 §5.12. A class this build does not know maps to `Internal`.
#[must_use]
pub fn class_code(class: ErrorClass) -> u32 {
    match class {
        ErrorClass::InvalidRequest => 1,
        ErrorClass::Unsupported => 2,
        ErrorClass::Capacity => 3,
        ErrorClass::Transient => 4,
        ErrorClass::Link => 5,
        ErrorClass::Device => 6,
        ErrorClass::Integrity => 7,
        ErrorClass::Correctness => 8,
        ErrorClass::Drift => 9,
        ErrorClass::Timeout => 10,
        ErrorClass::Auth => 11,
        ErrorClass::Cancelled => 12,
        _ => 13,
    }
}

/// The class `code` names, or `None` for 0 (success). A code this build does not know is
/// `Internal`.
#[must_use]
pub fn class_of_code(code: u32) -> Option<ErrorClass> {
    Some(match code {
        0 => return None,
        1 => ErrorClass::InvalidRequest,
        2 => ErrorClass::Unsupported,
        3 => ErrorClass::Capacity,
        4 => ErrorClass::Transient,
        5 => ErrorClass::Link,
        6 => ErrorClass::Device,
        7 => ErrorClass::Integrity,
        8 => ErrorClass::Correctness,
        9 => ErrorClass::Drift,
        10 => ErrorClass::Timeout,
        11 => ErrorClass::Auth,
        12 => ErrorClass::Cancelled,
        _ => ErrorClass::Internal,
    })
}

/// The status that reports `error`.
#[must_use]
pub fn status_of_error(error: &Error) -> SynStatus {
    match super::domain_code(error.code.domain) {
        Some(domain) => SynStatus {
            class: class_code(error.class),
            domain,
            code: error.code.code,
            retry: u8::from(error.class == ErrorClass::Transient),
        },
        // Every `Domain` has a number (`tests/abi_domains.rs`); a domain without one would be a
        // build that skipped that test, and crosses as an internal error rather than a guess.
        None => SynStatus {
            class: class_code(ErrorClass::Internal),
            domain: super::DOMAIN_KERNELS,
            code: 0,
            retry: 0,
        },
    }
}

/// The error `status` reports, with `message` (fetched with `last_error`); `None` for success.
/// A domain number this build does not know is reported as an `Internal` error that names the
/// number and the code, as an unknown class is (RFC-0040 §10.4, note of 2026-10-30).
#[must_use]
pub fn error_of_status(status: SynStatus, message: String) -> Option<Error> {
    let class = class_of_code(status.class)?;
    Some(match super::domain_of_code(status.domain) {
        Some(domain) => Error::new(class, ErrorCode::new(domain, status.code), message),
        None => Error::new(
            ErrorClass::Internal,
            ErrorCode::new(Domain::Kernels, 0),
            format!(
                "a status of domain number {} this build does not know, code {}: {message}",
                status.domain, status.code
            ),
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_class_round_trips() {
        for class in [
            ErrorClass::InvalidRequest,
            ErrorClass::Unsupported,
            ErrorClass::Capacity,
            ErrorClass::Transient,
            ErrorClass::Link,
            ErrorClass::Device,
            ErrorClass::Integrity,
            ErrorClass::Correctness,
            ErrorClass::Drift,
            ErrorClass::Timeout,
            ErrorClass::Auth,
            ErrorClass::Cancelled,
            ErrorClass::Internal,
        ] {
            assert_eq!(class_of_code(class_code(class)), Some(class));
        }
    }

    #[test]
    fn success_and_unknown_codes() {
        assert!(SynStatus::OK.is_ok());
        assert_eq!(class_of_code(0), None);
        assert_eq!(class_of_code(999), Some(ErrorClass::Internal));
        assert!(error_of_status(SynStatus::OK, String::new()).is_none());
    }

    #[test]
    fn an_error_keeps_its_class_and_code() {
        let error = Error::new(
            ErrorClass::Capacity,
            crate::error::codes::ALLOC_FAILED,
            "no memory",
        );
        let status = status_of_error(&error);
        assert_eq!(
            (status.class, status.domain, status.code, status.retry),
            (3, 1, 9, 0)
        );
        assert_eq!(error_of_status(status, error.message.clone()), Some(error));
    }

    /// A code of another domain crosses with its domain: here a `Net` code and an `Exec` code.
    #[test]
    fn an_error_of_another_domain_keeps_its_domain() {
        for code in [
            ErrorCode::new(Domain::Net, 0x0207),
            syntropy_core::error::codes::exec::RING_CLOSED,
        ] {
            let error = Error::new(ErrorClass::Link, code, "elsewhere");
            let status = status_of_error(&error);
            assert_ne!(status.domain, super::super::DOMAIN_KERNELS);
            assert_eq!(error_of_status(status, error.message.clone()), Some(error));
        }
    }

    /// A domain number this build does not know decodes as `Internal`, naming the number.
    #[test]
    fn an_unknown_domain_number_is_internal() {
        let status = SynStatus {
            class: class_code(ErrorClass::Capacity),
            domain: 99,
            code: 0x0042,
            retry: 0,
        };
        let error = error_of_status(status, "far away".into()).expect("a failure");
        assert_eq!(error.class, ErrorClass::Internal);
        assert_eq!(error.code, ErrorCode::new(Domain::Kernels, 0));
        assert!(error.message.contains("domain number 99"), "{error}");
        assert!(error.message.contains("far away"), "{error}");
    }
}
