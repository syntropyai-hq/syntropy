// SPDX-License-Identifier: MIT OR Apache-2.0

//! The map from an error code to what an API client sees (RFC-0050 §12), as data. `api`
//! builds the bodies and headers from it.
//!
//! RFC-0050 §12.1 ("Distributed error codes", note 2026-10-22) gives every code of the
//! `Cluster`, `Net`, `Dist`, `Recovery` and `Security` domains one of four outcomes, and its
//! note of 2026-10-28 applies the same rules to the `Kernels` and `Exec` domains, and its note
//! of 2026-10-31 to the `Tier` domain. A code's
//! mapping is found in this order:
//! 1. **Its own §12.1 row.** A code that can reach an API client is named in a row of the
//!    §12.1 table. [`per_code`] lists those codes with the row each comes from. The rows for
//!    cluster codes are native routes only ([`ApiMapping::Native`]); `InstanceNotRoutable`
//!    and `NoRepairCapacity` have a row in every shape.
//! 2. **The exempt list.** A code that never leaves the cluster is on §12.1's exempt list
//!    ([`exempt`]). If one ever reaches an API response, it renders as 503
//!    `instance_unavailable`, whatever its class ([`ApiMapping::Exempt`]).
//! 3. **`DuplicateCommand` (Dist 0x0305)** never appears in `Error::code`
//!    ([`ApiMapping::NeverSurfaces`]). **`IngressLost` (Recovery 0x0404)**, the one `Cancelled`
//!    code, has no response to carry it ([`ApiMapping::NoResponse`]). §12.1's note names
//!    both, so [`per_code`] lists them too, and they never reach step 4.
//! 4. **A whole-class row** ([`for_class`]): `Link` and `Device` (`instance_unavailable`),
//!    `Timeout` (`timeout`), `Correctness` (`no_conforming_instance`), `Internal`
//!    (`internal_error`), and the escape rule for `Integrity`, `Drift` and `Transient` (503
//!    `instance_unavailable`). `Cancelled` takes §12.2's normal finish. The `Auth`,
//!    `Capacity`, `InvalidRequest` and `Unsupported` rows each name a narrower condition than
//!    their class, so they are no class rule: every distributed code of those classes is in
//!    step 1 or step 2.
//!
//! A code's own domain RFC can say more about where it surfaces: RFC-0031 §12 calls
//! `EdgeDemoted` and `QpBudgetExhausted` informational, and RFC-0033 §13's "Client sees"
//! column gives each recovery code's visible outcome. The shape here is what a response
//! carries if the code reaches one.
//!
//! **Retry hints.** The [`ApiError::retry_after`] of a shape is the row's default. The
//! error's own retry hint (`Error::retry`, RFC-0001 §5.12) overrides it. When the hint is
//! `Never`, the response carries `x-should-retry: false` and no `Retry-After` (RFC-0050
//! §12.1, "Retry signalling"). That is the case for an escaped `Integrity` code such as
//! Security 0x050C–0x050F (a signature or source that does not verify), even though the
//! shape is `instance_unavailable`.

use super::codes::{self, CodeInfo, cluster, dist, kernels, net, recovery, security, tier};
use super::{ErrorClass, ErrorCode};

/// The `Retry-After` header a row of RFC-0050 §12.1 sends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryAfter {
    /// The row sends no `Retry-After`.
    Absent,
    /// The row fixes the value, in seconds (`Retry-After: 1`).
    Seconds(u32),
    /// The row sends `Retry-After` and fixes no value (`503 + Retry-After`); the value comes
    /// from the error's retry hint.
    Unfixed,
}

/// An error body of RFC-0050 §12.1 in the OpenAI, Anthropic and native shapes, before the
/// first token. After it, the shape's in-stream error carries the same type and code
/// (RFC-0050 §12.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApiError {
    /// The HTTP status in the OpenAI and native shapes.
    pub status: u16,
    /// The row's `Retry-After`. `Error::retry` overrides it: a `Never` hint sends no
    /// `Retry-After` (module docs).
    pub retry_after: RetryAfter,
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
    /// The `Link`/`Device` row, and the escape rule for `Integrity`, `Drift` and `Transient`:
    /// 503 `instance_unavailable`, `Retry-After: 1`; Anthropic `overloaded_error` (529).
    pub const INSTANCE_UNAVAILABLE: Self = Self {
        status: 503,
        retry_after: RetryAfter::Seconds(1),
        openai_type: "server_error",
        openai_code: "instance_unavailable",
        anthropic_type: "overloaded_error",
        anthropic_status: 529,
    };
    /// The row for unroutable instances and infeasible repair (`Capacity`): 503
    /// `instance_unavailable` with a `Retry-After` the row does not fix; Anthropic
    /// `overloaded_error` (529).
    pub const INSTANCE_UNAVAILABLE_CAPACITY: Self = Self {
        retry_after: RetryAfter::Unfixed,
        ..Self::INSTANCE_UNAVAILABLE
    };
    /// The `Timeout` row: 504 `timeout`; Anthropic `timeout_error`.
    pub const TIMEOUT: Self = Self {
        status: 504,
        retry_after: RetryAfter::Absent,
        openai_type: "server_error",
        openai_code: "timeout",
        anthropic_type: "timeout_error",
        anthropic_status: 504,
    };
    /// The `Correctness` row: 503 `no_conforming_instance`; Anthropic `api_error` (500).
    pub const NO_CONFORMING_INSTANCE: Self = Self {
        status: 503,
        retry_after: RetryAfter::Absent,
        openai_type: "server_error",
        openai_code: "no_conforming_instance",
        anthropic_type: "api_error",
        anthropic_status: 500,
    };
    /// The `Internal` row: 500 `internal_error`; Anthropic `api_error`.
    pub const INTERNAL_ERROR: Self = Self {
        status: 500,
        retry_after: RetryAfter::Absent,
        openai_type: "server_error",
        openai_code: "internal_error",
        anthropic_type: "api_error",
        anthropic_status: 500,
    };
}

/// An error of a native route only (`/v1/syntropy/cluster…`, `/v1/syntropy/nodes…`,
/// `/v1/syntropy/config…`; RFC-0050 §1.2). The OpenAI and Anthropic shapes never serve these
/// routes, so the row gives them no shape. The body is the native error, which names the
/// domain code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeError {
    /// The HTTP status.
    pub status: u16,
    /// The `Api` domain code the row reports the error as (RFC-0050 §12.3), where it names
    /// one: `cluster_proposal_rejected` for a rejected configuration proposal.
    pub api_code: Option<&'static str>,
}

/// What a client sees for a code (RFC-0050 §12).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApiMapping {
    /// This error body and status before the first token; the shape's in-stream error with
    /// the same type and code after it.
    Error(ApiError),
    /// An error of a native route only; the OpenAI and Anthropic shapes never carry it.
    Native(NativeError),
    /// Not a failure: the shape's normal finish, with `syntropy.warnings` naming the cause,
    /// unless the client itself disconnected (RFC-0050 §12.2).
    NormalFinish,
    /// No response carries the code, because the client's connection is already gone
    /// (`IngressLost`; RFC-0050 §12.1, RFC-0033 §13).
    NoResponse,
    /// On RFC-0050 §12.1's exempt list: the code never leaves the cluster. It is raised
    /// between nodes or reported as a node diagnostic (`syntropy doctor`, the dashboard). If
    /// it ever reaches an API response, it renders as `if_escaped`, which is always
    /// [`ApiError::INSTANCE_UNAVAILABLE`], whatever the code's class.
    Exempt {
        /// The shape the code renders as if it escapes.
        if_escaped: ApiError,
    },
    /// The code never appears in `Error::code` (`DuplicateCommand`: the coordinator answers
    /// a repeated command with its original result; RFC-0032 §6.3).
    NeverSurfaces,
}

/// [`ApiMapping::Exempt`] with its one fallback shape.
const EXEMPT_MAPPING: ApiMapping = ApiMapping::Exempt {
    if_escaped: ApiError::INSTANCE_UNAVAILABLE,
};

const JOIN_REFUSED: ApiMapping = ApiMapping::Native(NativeError {
    status: 403,
    api_code: None,
});
const PROPOSAL_REJECTED: ApiMapping = ApiMapping::Native(NativeError {
    status: 409,
    api_code: Some("cluster_proposal_rejected"),
});

/// The codes named by a row of RFC-0050 §12.1, with that row, plus the two codes §12.1's
/// note names one by one (`DuplicateCommand`, `IngressLost`).
const PER_CODE: &[(ErrorCode, ApiMapping, &str)] = &[
    (
        cluster::IDENTITY_HOST_MISMATCH,
        JOIN_REFUSED,
        "RFC-0050 §12.1: a join the keyholder or the joining node refuses",
    ),
    (
        cluster::PROTO_MISMATCH,
        ApiMapping::Native(NativeError {
            status: 400,
            api_code: None,
        }),
        "RFC-0050 §12.1: a join refused because the two nodes' protocol majors differ",
    ),
    (
        cluster::TOKEN_MALFORMED,
        ApiMapping::Native(NativeError {
            status: 400,
            api_code: None,
        }),
        "RFC-0050 §12.1: `POST /v1/syntropy/cluster/join` with a malformed invite token",
    ),
    (
        cluster::TOKEN_EXPIRED,
        JOIN_REFUSED,
        "RFC-0050 §12.1: a join the keyholder or the joining node refuses",
    ),
    (
        cluster::TOKEN_EXHAUSTED,
        JOIN_REFUSED,
        "RFC-0050 §12.1: a join the keyholder or the joining node refuses",
    ),
    (
        cluster::PAIR_DENIED,
        JOIN_REFUSED,
        "RFC-0050 §12.1: a join the keyholder or the joining node refuses",
    ),
    (
        cluster::CLOCK_SKEW,
        JOIN_REFUSED,
        "RFC-0050 §12.1: a join the keyholder or the joining node refuses",
    ),
    (
        cluster::TOKEN_REVOKED,
        JOIN_REFUSED,
        "RFC-0050 §12.1: a join the keyholder or the joining node refuses",
    ),
    (
        dist::FEATURE_LEVEL_TOO_LOW,
        PROPOSAL_REJECTED,
        "RFC-0050 §12.1: a cluster configuration proposal rejected",
    ),
    (
        dist::DUPLICATE_COMMAND,
        ApiMapping::NeverSurfaces,
        "RFC-0050 §12.1: `DuplicateCommand` (0x0305) never appears in `Error::code`",
    ),
    (
        dist::INSTANCE_NOT_ROUTABLE,
        ApiMapping::Error(ApiError::INSTANCE_UNAVAILABLE_CAPACITY),
        "RFC-0050 §12.1: the only instances of the model are unroutable",
    ),
    (
        dist::CONFIG_CONFLICT,
        PROPOSAL_REJECTED,
        "RFC-0050 §12.1: a cluster configuration proposal rejected",
    ),
    (
        recovery::INGRESS_LOST,
        ApiMapping::NoResponse,
        "RFC-0050 §12.1: `IngressLost` (0x0404), has no response to carry it",
    ),
    (
        recovery::NO_REPAIR_CAPACITY,
        ApiMapping::Error(ApiError::INSTANCE_UNAVAILABLE_CAPACITY),
        "RFC-0050 §12.1: recovery found no feasible repair",
    ),
    (
        security::KEYHOLDER_REQUIRED,
        JOIN_REFUSED,
        "RFC-0050 §12.1: an operator action that needs a keyholder signature with no keyholder reachable",
    ),
];

/// RFC-0050 §12.1's exempt list: the codes that never leave the cluster, or for `Kernels`,
/// the node that raised them.
const EXEMPT: &[ErrorCode] = &[
    cluster::NODE_CONFLICT,
    cluster::LOCAL_NETWORK_DENIED,
    cluster::NETWORK_PROFILE_PUBLIC,
    net::PROTO_MAJOR_MISMATCH,
    net::CREDENTIAL_INVALID,
    net::NOT_A_NEIGHBOUR,
    net::UNKNOWN_FRAME_TYPE,
    net::FRAME_TOO_LARGE,
    net::WIRE_DTYPE_MISMATCH,
    net::RDMA_NOT_ENABLED,
    net::RDMA_OS_BUILD_MISMATCH,
    net::SHM_USER_MISMATCH,
    net::PROTO_RANGE_DISJOINT,
    dist::ENTRY_REJECTED,
    recovery::HISTORY_DROPPED,
    security::CERT_INVALID_SIGNATURE,
    security::CERT_EXPIRED,
    security::CERT_REVOKED,
    security::CERT_WRONG_CLUSTER,
    security::ROLE_INSUFFICIENT,
    security::CREDENTIAL_EXPIRED,
    security::ROTATION_WINDOW_CLOSED,
    security::UNTRUSTED_FABRIC,
    security::AUTHORITY_INVALID,
    security::ROTATION_CHAIN_INVALID,
    security::NOT_COORDINATOR_ELIGIBLE,
    kernels::OP_UNSUPPORTED,
    kernels::MODULE_MISSING,
    kernels::MODULE_ABI,
    kernels::DRIVER_TOO_OLD,
    kernels::CAPTURE_FAILED,
    kernels::ALLOC_FAILED,
    kernels::REGISTER_FAILED,
    kernels::GGML_IMPORT_FAILED,
    kernels::AMX_DENIED,
    kernels::NO_CERTIFIED_CANDIDATE,
    tier::LEDGER_EXCEEDED,
    tier::VRAM_SHORT,
    tier::HOST_RESERVE,
    tier::REGISTER_UNSUPPORTED,
    tier::VMM_UNSUPPORTED,
];

/// The codes RFC-0050 §12.1 names one by one: the code, its mapping and the row it comes
/// from, in domain and number order.
pub fn per_code() -> impl Iterator<Item = (ErrorCode, ApiMapping, &'static str)> {
    PER_CODE.iter().copied()
}

/// RFC-0050 §12.1's exempt list, in domain and number order.
pub fn exempt() -> impl Iterator<Item = ErrorCode> {
    EXEMPT.iter().copied()
}

/// The API shape for an error of class `class`, where a row of RFC-0050 §12 covers the
/// whole class; `None` for `Auth`, `Capacity`, `InvalidRequest` and `Unsupported`, whose
/// rows each name a narrower condition.
pub const fn for_class(class: ErrorClass) -> Option<ApiMapping> {
    match class {
        ErrorClass::Link
        | ErrorClass::Device
        | ErrorClass::Integrity
        | ErrorClass::Drift
        | ErrorClass::Transient => Some(ApiMapping::Error(ApiError::INSTANCE_UNAVAILABLE)),
        ErrorClass::Timeout => Some(ApiMapping::Error(ApiError::TIMEOUT)),
        ErrorClass::Correctness => Some(ApiMapping::Error(ApiError::NO_CONFORMING_INSTANCE)),
        ErrorClass::Internal => Some(ApiMapping::Error(ApiError::INTERNAL_ERROR)),
        ErrorClass::Cancelled => Some(ApiMapping::NormalFinish),
        ErrorClass::Auth
        | ErrorClass::Capacity
        | ErrorClass::InvalidRequest
        | ErrorClass::Unsupported => None,
    }
}

/// The API shape for a published code; `None` when no table lists the code, or when no rule
/// of RFC-0050 §12.1 covers it (no published code: `tests/error_codes.rs` checks).
pub fn for_code(code: ErrorCode) -> Option<ApiMapping> {
    codes::lookup(code).and_then(for_info)
}

/// The API shape for a published code: its §12.1 row, else the exempt list, else its
/// whole-class row (module docs).
pub fn for_info(info: &CodeInfo) -> Option<ApiMapping> {
    if let Some(&(_, mapping, _)) = PER_CODE.iter().find(|(code, ..)| *code == info.code) {
        return Some(mapping);
    }
    if EXEMPT.contains(&info.code) {
        return Some(EXEMPT_MAPPING);
    }
    info.class.and_then(for_class)
}
