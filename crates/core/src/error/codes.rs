// SPDX-License-Identifier: MIT OR Apache-2.0

//! The published error codes, one module per domain, with each code's names and class as its
//! RFC's table gives them.
//!
//! [`REGISTRY`] lists every code and [`RESERVED`] the numbers a table holds back. A code is
//! never reused or renumbered (RFC-0001 §10.4); `tests/error_codes.rs` pins every number.
//!
//! Two RFC rows give one code several names: Net 0x0210 (`BlobNotFound`, `BlobBusy`,
//! `BlobThrottled`) and Security 0x050E (`ModelSignatureInvalid`, `ModelUnsigned`). Each name
//! has its own constant, and the constants are equal. One row has no class: Dist 0x0305,
//! `DuplicateCommand`, whose reply carries the original result.

use super::{Domain, ErrorClass, ErrorCode};

/// One published code: its number, its names and its class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeInfo {
    /// The code.
    pub code: ErrorCode,
    /// The names its RFC's table gives it; usually one.
    pub names: &'static [&'static str],
    /// The class its RFC's table gives it. `None` where the table gives none.
    pub class: Option<ErrorClass>,
}

macro_rules! class {
    (-) => {
        None
    };
    ($class:ident) => {
        Some(ErrorClass::$class)
    };
}

/// Declares a domain's module: one constant per code and its `TABLE`. A row is
/// `CONST = number, [names], class;`, where `-` is "no class". The module takes doc comments
/// only, so no lint attribute can pass through the macro (RFC-0060 §2.4).
macro_rules! domain_codes {
    (
        $(#[doc = $doc:literal])*
        $module:ident: $domain:ident {
            $($konst:ident = $num:literal, [$first:literal $(, $alias:literal)*], $class:tt;)*
        }
        $($extra:item)*
    ) => {
        $(#[doc = $doc])*
        pub mod $module {
            use super::{CodeInfo, Domain, ErrorClass, ErrorCode};

            $(
                #[doc = concat!("`", $first, "`", $(", also `", $alias, "`",)* ": ",
                    stringify!($num), ", class `", stringify!($class), "`.")]
                pub const $konst: ErrorCode = ErrorCode::new(Domain::$domain, $num);
            )*

            $($extra)*

            /// Every code of the domain, in number order.
            pub(super) const TABLE: &[CodeInfo] = &[
                $(CodeInfo {
                    code: $konst,
                    names: &[$first $(, $alias)*],
                    class: class!($class),
                },)*
            ];
        }
    };
}

domain_codes! {
    /// `Domain::Cluster`, 0x0100–0x01FF (RFC-0030 §10).
    cluster: Cluster {
        IDENTITY_HOST_MISMATCH = 0x0101, ["IdentityHostMismatch"], Auth;
        NODE_CONFLICT = 0x0102, ["NodeConflict"], Auth;
        PROTO_MISMATCH = 0x0103, ["ProtoMismatch"], Unsupported;
        TOKEN_MALFORMED = 0x0104, ["TokenMalformed"], InvalidRequest;
        TOKEN_EXPIRED = 0x0105, ["TokenExpired"], Auth;
        TOKEN_EXHAUSTED = 0x0106, ["TokenExhausted"], Auth;
        PAIR_DENIED = 0x0107, ["PairDenied"], Auth;
        RENDEZVOUS_OFFLINE = 0x0108, ["RendezvousOffline"], Link;
        COORDINATOR_UNREACHABLE = 0x0109, ["CoordinatorUnreachable"], Transient;
        LOCAL_NETWORK_DENIED = 0x010A, ["LocalNetworkDenied"], Unsupported;
        NETWORK_PROFILE_PUBLIC = 0x010B, ["NetworkProfilePublic"], Unsupported;
        CLOCK_SKEW = 0x010C, ["ClockSkew"], Auth;
        DISCOVERY_SILENT = 0x010D, ["DiscoverySilent"], Link;
        RATE_LIMITED = 0x010E, ["RateLimited"], Transient;
        TOKEN_REVOKED = 0x010F, ["TokenRevoked"], Auth;
    }
}

domain_codes! {
    /// `Domain::Net`, 0x0200–0x02FF (RFC-0031 §12). These codes travel in `HelloAck::status`,
    /// in `Close::code` and in `Error::code`.
    net: Net {
        PROTO_MAJOR_MISMATCH = 0x0201, ["ProtoMajorMismatch"], Unsupported;
        CREDENTIAL_INVALID = 0x0202, ["CredentialInvalid"], Auth;
        NOT_A_NEIGHBOUR = 0x0203, ["NotANeighbour"], Auth;
        UNKNOWN_FRAME_TYPE = 0x0204, ["UnknownFrameType"], Unsupported;
        FRAME_TOO_LARGE = 0x0205, ["FrameTooLarge"], InvalidRequest;
        WIRE_DTYPE_MISMATCH = 0x0206, ["WireDTypeMismatch"], Unsupported;
        CRC_STORM = 0x0207, ["CrcStorm"], Link;
        ACK_TIMEOUT = 0x0208, ["AckTimeout"], Link;
        QP_ERROR = 0x0209, ["QpError"], Link;
        CONNECT_TIMEOUT = 0x020A, ["ConnectTimeout"], Link;
        PEER_RESET = 0x020B, ["PeerReset"], Link;
        EDGE_DEMOTED = 0x020C, ["EdgeDemoted"], Link;
        RDMA_NOT_ENABLED = 0x020D, ["RdmaNotEnabled"], Unsupported;
        RDMA_OS_BUILD_MISMATCH = 0x020E, ["RdmaOsBuildMismatch"], Unsupported;
        SHM_USER_MISMATCH = 0x020F, ["ShmUserMismatch"], Unsupported;
        BLOB_NOT_FOUND = 0x0210, ["BlobNotFound", "BlobBusy", "BlobThrottled"], Transient;
        EPOCH_NOT_SERVED = 0x0211, ["EpochNotServed"], Transient;
        PROTO_RANGE_DISJOINT = 0x0212, ["ProtoRangeDisjoint"], Unsupported;
        QP_BUDGET_EXHAUSTED = 0x0213, ["QpBudgetExhausted"], Link;
    }

    /// `BlobBusy`: the same code as [`BLOB_NOT_FOUND`] (RFC-0031 §12 gives one row three names).
    pub const BLOB_BUSY: ErrorCode = BLOB_NOT_FOUND;
    /// `BlobThrottled`: the same code as [`BLOB_NOT_FOUND`].
    pub const BLOB_THROTTLED: ErrorCode = BLOB_NOT_FOUND;
}

domain_codes! {
    /// `Domain::Dist`, 0x0300–0x03FF (RFC-0032 §10).
    dist: Dist {
        NOT_COORDINATOR = 0x0301, ["NotCoordinator"], Transient;
        STALE_TERM = 0x0302, ["StaleTerm"], Transient;
        FEATURE_LEVEL_TOO_LOW = 0x0303, ["FeatureLevelTooLow"], Unsupported;
        GRAPH_MOVED = 0x0304, ["GraphMoved"], Transient;
        DUPLICATE_COMMAND = 0x0305, ["DuplicateCommand"], -;
        LOG_DIVERGED = 0x0306, ["LogDiverged"], Transient;
        SNAPSHOT_UNAVAILABLE = 0x0307, ["SnapshotUnavailable"], Transient;
        INSTANCE_NOT_ROUTABLE = 0x0308, ["InstanceNotRoutable"], Capacity;
        ISLAND_BRINGUP_TIMEOUT = 0x0309, ["IslandBringupTimeout"], Timeout;
        ISLAND_GATE_FAILED = 0x030A, ["IslandGateFailed"], Drift;
        ROUTER_MISMATCH = 0x030B, ["RouterMismatch"], Correctness;
        STEP_OUT_OF_ORDER = 0x030C, ["StepOutOfOrder"], Internal;
        NO_INGRESS_PATH = 0x030D, ["NoIngressPath"], Link;
        ENTRY_REJECTED = 0x030E, ["EntryRejected"], Auth;
        CONFIG_CONFLICT = 0x030F, ["ConfigConflict"], InvalidRequest;
    }
}

domain_codes! {
    /// `Domain::Recovery`, 0x0400–0x04FF (RFC-0033 §13).
    recovery: Recovery {
        RECOVERY_IN_PROGRESS = 0x0401, ["RecoveryInProgress"], Transient;
        REPLAY_NO_SOURCE = 0x0402, ["ReplayNoSource"], Link;
        EPOCH_RETIRED = 0x0403, ["EpochRetired"], Transient;
        INGRESS_LOST = 0x0404, ["IngressLost"], Cancelled;
        NO_REPAIR_CAPACITY = 0x0405, ["NoRepairCapacity"], Capacity;
        RESTART_BUDGET_SPENT = 0x0406, ["RestartBudgetSpent"], Device;
        HISTORY_DROPPED = 0x0407, ["HistoryDropped"], Capacity;
        KV_MIGRATION_FAILED = 0x0408, ["KvMigrationFailed"], Link;
        STATE_POSITION_MISMATCH = 0x0409, ["StatePositionMismatch"], Correctness;
    }
}

domain_codes! {
    /// `Domain::Security`, 0x0500–0x05FF (RFC-0034 §15). 0x0509–0x050B are reserved
    /// ([`RESERVED`]); API authentication codes are RFC-0050's.
    security: Security {
        CERT_INVALID_SIGNATURE = 0x0501, ["CertInvalidSignature"], Auth;
        CERT_EXPIRED = 0x0502, ["CertExpired"], Auth;
        CERT_REVOKED = 0x0503, ["CertRevoked"], Auth;
        CERT_WRONG_CLUSTER = 0x0504, ["CertWrongCluster"], Auth;
        ROLE_INSUFFICIENT = 0x0505, ["RoleInsufficient"], Auth;
        CREDENTIAL_EXPIRED = 0x0506, ["CredentialExpired"], Auth;
        KEYHOLDER_REQUIRED = 0x0507, ["KeyholderRequired"], Auth;
        ROTATION_WINDOW_CLOSED = 0x0508, ["RotationWindowClosed"], Auth;
        MODULE_SIGNATURE_INVALID = 0x050C, ["ModuleSignatureInvalid"], Integrity;
        UPDATE_SIGNATURE_INVALID = 0x050D, ["UpdateSignatureInvalid"], Integrity;
        MODEL_SIGNATURE_INVALID = 0x050E, ["ModelSignatureInvalid", "ModelUnsigned"], Integrity;
        SOURCE_DISTRUSTED = 0x050F, ["SourceDistrusted"], Integrity;
        UNTRUSTED_FABRIC = 0x0510, ["UntrustedFabric"], Unsupported;
        AUTHORITY_INVALID = 0x0511, ["AuthorityInvalid"], Auth;
        ROTATION_CHAIN_INVALID = 0x0512, ["RotationChainInvalid"], Auth;
        NOT_COORDINATOR_ELIGIBLE = 0x0513, ["NotCoordinatorEligible"], Auth;
    }

    /// `ModelUnsigned` (when a signature is required): the same code as
    /// [`MODEL_SIGNATURE_INVALID`] (RFC-0034 §15 gives one row two names).
    pub const MODEL_UNSIGNED: ErrorCode = MODEL_SIGNATURE_INVALID;
}

domain_codes! {
    /// `Domain::Kernels`, 0x0001–0x00FF (RFC-0040 §13, whose table numbers its codes 1–17).
    kernels: Kernels {
        OP_UNSUPPORTED = 1, ["OpUnsupported"], Unsupported;
        MODULE_MISSING = 2, ["ModuleMissing"], Unsupported;
        MODULE_INTEGRITY = 3, ["ModuleIntegrity"], Integrity;
        MODULE_ABI = 4, ["ModuleAbi"], Unsupported;
        DRIVER_TOO_OLD = 5, ["DriverTooOld"], Unsupported;
        PARITY_FAILED = 6, ["ParityFailed"], Correctness;
        CAPTURE_FAILED = 7, ["CaptureFailed"], Unsupported;
        DEVICE_STICKY = 8, ["DeviceSticky"], Device;
        ALLOC_FAILED = 9, ["AllocFailed"], Capacity;
        REGISTER_FAILED = 10, ["RegisterFailed"], Capacity;
        DOORBELL_TIMEOUT = 11, ["DoorbellTimeout"], Timeout;
        GGML_IMPORT_FAILED = 12, ["GgmlImportFailed"], Unsupported;
        AMX_DENIED = 13, ["AmxDenied"], Unsupported;
        WORKSPACE_EXCEEDED = 14, ["WorkspaceExceeded"], Internal;
        NO_CERTIFIED_CANDIDATE = 15, ["NoCertifiedCandidate"], Unsupported;
        PROVIDER_PANICKED = 16, ["ProviderPanicked"], Internal;
        MANIFEST_INVALID = 17, ["ManifestInvalid"], Integrity;
    }
}

domain_codes! {
    /// `Domain::Exec`, 0x0600–0x06FF (RFC-0012 §13, note of 2026-10-28).
    exec: Exec {
        RING_CLOSED = 0x0601, ["RingClosed"], Link;
        STEP_STALLED = 0x0602, ["StepStalled"], Internal;
        POOL_BARRIER_MISSED = 0x0603, ["PoolBarrierMissed"], Internal;
    }
}

/// Every published code, by domain and then by number.
pub const REGISTRY: &[&[CodeInfo]] = &[
    cluster::TABLE,
    net::TABLE,
    dist::TABLE,
    recovery::TABLE,
    security::TABLE,
    kernels::TABLE,
    exec::TABLE,
];

/// Numbers a domain's table holds back. Never assigned (RFC-0034 §15: 0x0509–0x050B, because
/// API authentication codes are RFC-0050's).
pub const RESERVED: &[ErrorCode] = &[
    ErrorCode::new(Domain::Security, 0x0509),
    ErrorCode::new(Domain::Security, 0x050A),
    ErrorCode::new(Domain::Security, 0x050B),
];

/// Every published code, by domain and then by number.
pub fn all() -> impl Iterator<Item = &'static CodeInfo> {
    REGISTRY.iter().flat_map(|table| table.iter())
}

/// The published code `code`, or `None` when no table lists it.
pub fn lookup(code: ErrorCode) -> Option<&'static CodeInfo> {
    all().find(|info| info.code == code)
}
