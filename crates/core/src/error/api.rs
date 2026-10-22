// SPDX-License-Identifier: MIT OR Apache-2.0

//! The map from an error code to its API error shape in RFC-0050 §12, as data. `api` builds
//! the bodies and headers from it.
//!
//! RFC-0050 §12.1 maps by condition and `ErrorClass`, not by domain code. A code is mapped
//! here only where the table covers every error of the code's class:
//! - `Link` and `Device`: the row "stage, link or device failure not recovered before the
//!   first token" (`instance_unavailable`);
//! - `Integrity`, `Drift` and `Transient`: the rule that one that escapes maps to 503
//!   `instance_unavailable`;
//! - `Timeout`: the request-deadline row (`timeout`);
//! - `Correctness`: the canary row (`no_conforming_instance`);
//! - `Internal`: the `Internal` row (`internal_error`);
//! - `Cancelled`: §12.2's normal finish, with `syntropy.warnings` naming the cause.
//!
//! For `Auth`, `Capacity`, `InvalidRequest` and `Unsupported`, every row of RFC-0050 §12.1
//! names a client-side condition, such as an API key or a request field. None of those is a
//! cluster, transport, recovery or security code, so those codes are [`ApiMapping::Unmapped`]
//! until RFC-0050 gives them rows. The same holds for Dist 0x0305, which has no class.

use super::codes::{self, CodeInfo};
use super::{ErrorClass, ErrorCode};

/// An error body of RFC-0050 §12.1 before the first token. After it, the shape's in-stream
/// error carries the same type and code (RFC-0050 §12.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApiError {
    /// The HTTP status in the OpenAI and native shapes.
    pub status: u16,
    /// The `Retry-After` value in seconds, where the row fixes one.
    pub retry_after: Option<u32>,
    /// The OpenAI `type`.
    pub openai_type: &'static str,
    /// The OpenAI `code`, which is also the `Api` domain code's name (RFC-0050 §12.3).
    pub openai_code: &'static str,
    /// The Anthropic `type`.
    pub anthropic_type: &'static str,
    /// The HTTP status in the Anthropic shape: 529 for overload, as the published API uses.
    pub anthropic_status: u16,
}

impl ApiError {
    /// 503 `instance_unavailable`, `Retry-After: 1`; Anthropic `overloaded_error` (529).
    pub const INSTANCE_UNAVAILABLE: Self = Self {
        status: 503,
        retry_after: Some(1),
        openai_type: "server_error",
        openai_code: "instance_unavailable",
        anthropic_type: "overloaded_error",
        anthropic_status: 529,
    };
    /// 504 `timeout`; Anthropic `timeout_error`.
    pub const TIMEOUT: Self = Self {
        status: 504,
        retry_after: None,
        openai_type: "server_error",
        openai_code: "timeout",
        anthropic_type: "timeout_error",
        anthropic_status: 504,
    };
    /// 503 `no_conforming_instance`; Anthropic `api_error` (500).
    pub const NO_CONFORMING_INSTANCE: Self = Self {
        status: 503,
        retry_after: None,
        openai_type: "server_error",
        openai_code: "no_conforming_instance",
        anthropic_type: "api_error",
        anthropic_status: 500,
    };
    /// 500 `internal_error`; Anthropic `api_error`.
    pub const INTERNAL_ERROR: Self = Self {
        status: 500,
        retry_after: None,
        openai_type: "server_error",
        openai_code: "internal_error",
        anthropic_type: "api_error",
        anthropic_status: 500,
    };
}

/// What a client sees for a code (RFC-0050 §12).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiMapping {
    /// This error body and status.
    Error(ApiError),
    /// Not a failure: the shape's normal finish, with `syntropy.warnings` naming the cause,
    /// unless the client itself disconnected (RFC-0050 §12.2).
    NormalFinish,
    /// RFC-0050 §12.1 has no row for the code yet.
    Unmapped,
}

/// The API shape for an error of class `class`, where RFC-0050 §12 gives one for the whole
/// class; [`ApiMapping::Unmapped`] otherwise.
pub const fn for_class(class: ErrorClass) -> ApiMapping {
    match class {
        ErrorClass::Link
        | ErrorClass::Device
        | ErrorClass::Integrity
        | ErrorClass::Drift
        | ErrorClass::Transient => ApiMapping::Error(ApiError::INSTANCE_UNAVAILABLE),
        ErrorClass::Timeout => ApiMapping::Error(ApiError::TIMEOUT),
        ErrorClass::Correctness => ApiMapping::Error(ApiError::NO_CONFORMING_INSTANCE),
        ErrorClass::Internal => ApiMapping::Error(ApiError::INTERNAL_ERROR),
        ErrorClass::Cancelled => ApiMapping::NormalFinish,
        ErrorClass::Auth
        | ErrorClass::Capacity
        | ErrorClass::InvalidRequest
        | ErrorClass::Unsupported => ApiMapping::Unmapped,
    }
}

/// The API shape for a published code, from its class; `None` when no table lists the code.
pub fn for_code(code: ErrorCode) -> Option<ApiMapping> {
    codes::lookup(code).map(for_info)
}

/// The API shape for a published code, from its class.
pub const fn for_info(info: &CodeInfo) -> ApiMapping {
    match info.class {
        Some(class) => for_class(class),
        None => ApiMapping::Unmapped,
    }
}
