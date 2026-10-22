// SPDX-License-Identifier: MIT OR Apache-2.0

//! The map from an error code to what an API client sees (RFC-0050 §12), as data. `api`
//! builds the bodies and headers from it.
//!
//! A code's mapping comes from one of two places, in this order:
//! 1. **The code's own RFC row**, where it states the client outcome. Such a row is an
//!    override, listed in `OVERRIDES` with the row it comes from. Examples are RFC-0033 §13's
//!    "Client sees" column and RFC-0031 §12's "informational" codes.
//! 2. **A class rule of RFC-0050 §12.1**, but only where the rule covers every error of the
//!    class:
//!    - `Internal`: the `Internal` row (`internal_error`);
//!    - `Integrity`, `Drift` and `Transient`: "if one escapes, it maps to 503
//!      `instance_unavailable`";
//!    - `Link` and `Device`: the row "stage, link or device failure not recovered before the
//!      first token, after one re-route" (`instance_unavailable`). Its condition names the
//!      whole class plus a "when", so it is read as a class rule. #1668 asks the maintainers
//!      to confirm that reading;
//!    - `Cancelled`: §12.2's normal finish.
//!
//! The `Timeout` row ("request deadline before the first token"), the `Correctness` row ("no
//! instance passes its canary") and every `Auth`, `Capacity`, `InvalidRequest` and
//! `Unsupported` row name a narrower condition than their class. A code of those classes
//! whose own row states no outcome is therefore [`ApiMapping::Unmapped`], and stays so until
//! RFC-0050 gives it a row (#1668).
//!
//! **Retry hints.** The [`ApiError::retry_after`] of a shape is the row's default. The
//! error's own retry hint (`Error::retry`, RFC-0001 §5.12) overrides it. When the hint is
//! `Never`, the response carries `x-should-retry: false` and no `Retry-After` (RFC-0050
//! §12.1, "Retry signalling"). That is the case for an escaped `Integrity` code such as
//! Security 0x050C–0x050E (a signature that does not verify), even though the shape is
//! `instance_unavailable`. (0x050F `SourceDistrusted` has an override and never reaches a
//! client.)

use super::codes::{self, CodeInfo, cluster, dist, net, recovery, security};
use super::{ErrorClass, ErrorCode};

/// An error body of RFC-0050 §12.1 before the first token. After it, the shape's in-stream
/// error carries the same type and code (RFC-0050 §12.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApiError {
    /// The HTTP status in the OpenAI and native shapes.
    pub status: u16,
    /// The `Retry-After` value in seconds, where the row fixes one. `Error::retry` overrides
    /// it: a `Never` hint sends no `Retry-After` (module docs).
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
    /// This error body and status before the first token; the shape's in-stream error with
    /// the same type and code after it.
    Error(ApiError),
    /// The shape's in-stream error of RFC-0050 §12.2, carrying the code itself. The code's
    /// RFC row says the client sees a stream error.
    StreamError {
        /// The row calls it retryable: the body carries `retry_after`, whose value comes from
        /// `Error::retry`.
        retryable: bool,
    },
    /// Not a failure: the shape's normal finish, with `syntropy.warnings` naming the cause,
    /// unless the client itself disconnected (RFC-0050 §12.2).
    NormalFinish,
    /// The code's RFC row says no client sees it: it is informational, not fatal, handled
    /// inside the cluster, or the client is gone. It surfaces in logs, metrics, `doctor` or
    /// the dashboard, not in an API response.
    NotClientVisible,
    /// Neither the code's RFC nor RFC-0050 §12.1 gives it a shape yet (#1668).
    Unmapped,
}

/// The codes whose own RFC row states the client outcome, with that row. They take
/// precedence over the class rule.
const OVERRIDES: &[(ErrorCode, ApiMapping, &str)] = &[
    (
        cluster::NODE_CONFLICT,
        ApiMapping::NotClientVisible,
        "RFC-0030 §10: both are excluded until one stops; `NodeConflict`; `doctor` explains",
    ),
    (
        cluster::DISCOVERY_SILENT,
        ApiMapping::NotClientVisible,
        "RFC-0030 §10: keep serving locally; `doctor`: `DiscoverySilent`",
    ),
    (
        net::EDGE_DEMOTED,
        ApiMapping::NotClientVisible,
        "RFC-0031 §12: informational; becomes a `Ready { degraded }` reason",
    ),
    (
        net::QP_BUDGET_EXHAUSTED,
        ApiMapping::NotClientVisible,
        "RFC-0031 §12: informational: channel opened on the next transport",
    ),
    (
        dist::DUPLICATE_COMMAND,
        ApiMapping::NotClientVisible,
        "RFC-0032 §10: reply carries the original result",
    ),
    (
        dist::ROUTER_MISMATCH,
        ApiMapping::NotClientVisible,
        "RFC-0032 §10: local routing disabled; not fatal",
    ),
    (
        recovery::RECOVERY_IN_PROGRESS,
        ApiMapping::StreamError { retryable: true },
        "RFC-0033 §13: keep-alives; on deadline, a retryable timeout",
    ),
    (
        recovery::REPLAY_NO_SOURCE,
        ApiMapping::StreamError { retryable: true },
        "RFC-0033 §13: stream error, retryable",
    ),
    (
        recovery::EPOCH_RETIRED,
        ApiMapping::StreamError { retryable: true },
        "RFC-0033 §13: stream error, `retry: OnNewPlan`",
    ),
    (
        recovery::INGRESS_LOST,
        ApiMapping::NotClientVisible,
        "RFC-0033 §13: — (connection gone)",
    ),
    (
        recovery::NO_REPAIR_CAPACITY,
        ApiMapping::StreamError { retryable: true },
        "RFC-0033 §13: stream error with `Retry-After`; instance unroutable",
    ),
    (
        recovery::RESTART_BUDGET_SPENT,
        ApiMapping::NotClientVisible,
        "RFC-0033 §13: — (handled by re-plan)",
    ),
    (
        recovery::HISTORY_DROPPED,
        ApiMapping::NotClientVisible,
        "RFC-0033 §13: — (informational; recovery cascades)",
    ),
    (
        recovery::KV_MIGRATION_FAILED,
        ApiMapping::NotClientVisible,
        "RFC-0033 §13: falls back to replay or re-prefill; nothing visible",
    ),
    (
        recovery::STATE_POSITION_MISMATCH,
        ApiMapping::NotClientVisible,
        "RFC-0033 §13: nothing (the handshake re-runs; §4.2.1)",
    ),
    (
        security::SOURCE_DISTRUSTED,
        ApiMapping::NotClientVisible,
        "RFC-0034 §10.1: refetched from another source; the node is flagged in the dashboard",
    ),
];

/// The codes whose own RFC row states the client outcome: the code, its mapping and the row
/// it comes from.
pub fn overrides() -> impl Iterator<Item = (ErrorCode, ApiMapping, &'static str)> {
    OVERRIDES.iter().copied()
}

/// The API shape for an error of class `class`, where an RFC-0050 §12 rule covers the whole
/// class; [`ApiMapping::Unmapped`] otherwise.
pub const fn for_class(class: ErrorClass) -> ApiMapping {
    match class {
        ErrorClass::Link
        | ErrorClass::Device
        | ErrorClass::Integrity
        | ErrorClass::Drift
        | ErrorClass::Transient => ApiMapping::Error(ApiError::INSTANCE_UNAVAILABLE),
        ErrorClass::Internal => ApiMapping::Error(ApiError::INTERNAL_ERROR),
        ErrorClass::Cancelled => ApiMapping::NormalFinish,
        ErrorClass::Timeout
        | ErrorClass::Correctness
        | ErrorClass::Auth
        | ErrorClass::Capacity
        | ErrorClass::InvalidRequest
        | ErrorClass::Unsupported => ApiMapping::Unmapped,
    }
}

/// The API shape for a published code; `None` when no table lists the code.
pub fn for_code(code: ErrorCode) -> Option<ApiMapping> {
    codes::lookup(code).map(for_info)
}

/// The API shape for a published code: its RFC row's override, else its class rule.
pub fn for_info(info: &CodeInfo) -> ApiMapping {
    if let Some(&(_, mapping, _)) = OVERRIDES.iter().find(|(code, ..)| *code == info.code) {
        return mapping;
    }
    match info.class {
        Some(class) => for_class(class),
        None => ApiMapping::Unmapped,
    }
}
