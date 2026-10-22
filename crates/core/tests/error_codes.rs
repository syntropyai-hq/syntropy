// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error-code registry (DIST-E18-T02): every published code is pinned to its number, its
//! names and its class, numbers are unique and inside their domain's range, and every code
//! has an RFC-0050 §12 API shape or is on the open list.

use std::collections::BTreeSet;

use syntropy_core::error::api::{self, ApiError, ApiMapping};
use syntropy_core::error::codes::{self, CodeInfo};
use syntropy_core::{Domain, ErrorClass, ErrorCode};

use Domain::{Cluster, Dist, Net, Recovery, Security};
use ErrorClass::*;

/// Every code as its RFC's table writes it. A renumbered, renamed, reclassed, removed or
/// added code fails `registry_matches_the_rfc_tables`; adding a code means adding its row.
const PINNED: &[(Domain, u16, &[&str], Option<ErrorClass>)] = &[
    // RFC-0030 §10
    (Cluster, 0x0101, &["IdentityHostMismatch"], Some(Auth)),
    (Cluster, 0x0102, &["NodeConflict"], Some(Auth)),
    (Cluster, 0x0103, &["ProtoMismatch"], Some(Unsupported)),
    (Cluster, 0x0104, &["TokenMalformed"], Some(InvalidRequest)),
    (Cluster, 0x0105, &["TokenExpired"], Some(Auth)),
    (Cluster, 0x0106, &["TokenExhausted"], Some(Auth)),
    (Cluster, 0x0107, &["PairDenied"], Some(Auth)),
    (Cluster, 0x0108, &["RendezvousOffline"], Some(Link)),
    (
        Cluster,
        0x0109,
        &["CoordinatorUnreachable"],
        Some(Transient),
    ),
    (Cluster, 0x010A, &["LocalNetworkDenied"], Some(Unsupported)),
    (
        Cluster,
        0x010B,
        &["NetworkProfilePublic"],
        Some(Unsupported),
    ),
    (Cluster, 0x010C, &["ClockSkew"], Some(Auth)),
    (Cluster, 0x010D, &["DiscoverySilent"], Some(Link)),
    (Cluster, 0x010E, &["RateLimited"], Some(Transient)),
    // RFC-0031 §12
    (Net, 0x0201, &["ProtoMajorMismatch"], Some(Unsupported)),
    (Net, 0x0202, &["CredentialInvalid"], Some(Auth)),
    (Net, 0x0203, &["NotANeighbour"], Some(Auth)),
    (Net, 0x0204, &["UnknownFrameType"], Some(Unsupported)),
    (Net, 0x0205, &["FrameTooLarge"], Some(Internal)),
    (Net, 0x0206, &["WireDTypeMismatch"], Some(Internal)),
    (Net, 0x0207, &["CrcStorm"], Some(Link)),
    (Net, 0x0208, &["AckTimeout"], Some(Link)),
    (Net, 0x0209, &["QpError"], Some(Link)),
    (Net, 0x020A, &["ConnectTimeout"], Some(Link)),
    (Net, 0x020B, &["PeerReset"], Some(Link)),
    (Net, 0x020C, &["EdgeDemoted"], Some(Link)),
    (Net, 0x020D, &["RdmaNotEnabled"], Some(Unsupported)),
    (Net, 0x020E, &["RdmaOsBuildMismatch"], Some(Unsupported)),
    (Net, 0x020F, &["ShmUserMismatch"], Some(Unsupported)),
    (
        Net,
        0x0210,
        &["BlobNotFound", "BlobBusy", "BlobThrottled"],
        Some(Transient),
    ),
    (Net, 0x0211, &["EpochNotServed"], Some(Transient)),
    (Net, 0x0212, &["ProtoRangeDisjoint"], Some(Unsupported)),
    (Net, 0x0213, &["QpBudgetExhausted"], Some(Link)),
    // RFC-0032 §10
    (Dist, 0x0301, &["NotCoordinator"], Some(Transient)),
    (Dist, 0x0302, &["StaleTerm"], Some(Transient)),
    (Dist, 0x0303, &["FeatureLevelTooLow"], Some(Unsupported)),
    (Dist, 0x0304, &["GraphMoved"], Some(Transient)),
    (Dist, 0x0305, &["DuplicateCommand"], None),
    (Dist, 0x0306, &["LogDiverged"], Some(Transient)),
    (Dist, 0x0307, &["SnapshotUnavailable"], Some(Transient)),
    (Dist, 0x0308, &["InstanceNotRoutable"], Some(Capacity)),
    (Dist, 0x0309, &["IslandBringupTimeout"], Some(Timeout)),
    (Dist, 0x030A, &["IslandGateFailed"], Some(Drift)),
    (Dist, 0x030B, &["RouterMismatch"], Some(Correctness)),
    (Dist, 0x030C, &["StepOutOfOrder"], Some(Internal)),
    (Dist, 0x030D, &["NoIngressPath"], Some(Link)),
    (Dist, 0x030E, &["EntryRejected"], Some(Auth)),
    (Dist, 0x030F, &["ConfigConflict"], Some(InvalidRequest)),
    // RFC-0033 §13
    (Recovery, 0x0401, &["RecoveryInProgress"], Some(Transient)),
    (Recovery, 0x0402, &["ReplayNoSource"], Some(Link)),
    (Recovery, 0x0403, &["EpochRetired"], Some(Transient)),
    (Recovery, 0x0404, &["IngressLost"], Some(Cancelled)),
    (Recovery, 0x0405, &["NoRepairCapacity"], Some(Capacity)),
    (Recovery, 0x0406, &["RestartBudgetSpent"], Some(Device)),
    (Recovery, 0x0407, &["HistoryDropped"], Some(Capacity)),
    (Recovery, 0x0408, &["KvMigrationFailed"], Some(Link)),
    (
        Recovery,
        0x0409,
        &["StatePositionMismatch"],
        Some(Correctness),
    ),
    // RFC-0034 §15
    (Security, 0x0501, &["CertInvalidSignature"], Some(Auth)),
    (Security, 0x0502, &["CertExpired"], Some(Auth)),
    (Security, 0x0503, &["CertRevoked"], Some(Auth)),
    (Security, 0x0504, &["CertWrongCluster"], Some(Auth)),
    (Security, 0x0505, &["RoleInsufficient"], Some(Auth)),
    (Security, 0x0506, &["CredentialExpired"], Some(Auth)),
    (Security, 0x0507, &["KeyholderRequired"], Some(Auth)),
    (Security, 0x0508, &["RotationWindowClosed"], Some(Auth)),
    (
        Security,
        0x050C,
        &["ModuleSignatureInvalid"],
        Some(Integrity),
    ),
    (
        Security,
        0x050D,
        &["UpdateSignatureInvalid"],
        Some(Integrity),
    ),
    (
        Security,
        0x050E,
        &["ModelSignatureInvalid", "ModelUnsigned"],
        Some(Integrity),
    ),
    (Security, 0x050F, &["SourceDistrusted"], Some(Integrity)),
    (Security, 0x0510, &["UntrustedFabric"], Some(Unsupported)),
    (Security, 0x0511, &["AuthorityInvalid"], Some(Auth)),
    (Security, 0x0512, &["RotationChainInvalid"], Some(Auth)),
    (Security, 0x0513, &["NotCoordinatorEligible"], Some(Auth)),
];

/// The codes RFC-0050 §12.1 has no row for yet: the open question this task leaves for the
/// API RFC. A code leaves this list when its class or RFC-0050 gives it a shape.
const UNMAPPED: &[(Domain, u16)] = &[
    (Cluster, 0x0101),
    (Cluster, 0x0102),
    (Cluster, 0x0103),
    (Cluster, 0x0104),
    (Cluster, 0x0105),
    (Cluster, 0x0106),
    (Cluster, 0x0107),
    (Cluster, 0x010A),
    (Cluster, 0x010B),
    (Cluster, 0x010C),
    (Net, 0x0201),
    (Net, 0x0202),
    (Net, 0x0203),
    (Net, 0x0204),
    (Net, 0x020D),
    (Net, 0x020E),
    (Net, 0x020F),
    (Net, 0x0212),
    (Dist, 0x0303),
    (Dist, 0x0305),
    (Dist, 0x0308),
    (Dist, 0x030E),
    (Dist, 0x030F),
    (Recovery, 0x0405),
    (Recovery, 0x0407),
    (Security, 0x0501),
    (Security, 0x0502),
    (Security, 0x0503),
    (Security, 0x0504),
    (Security, 0x0505),
    (Security, 0x0506),
    (Security, 0x0507),
    (Security, 0x0508),
    (Security, 0x0510),
    (Security, 0x0511),
    (Security, 0x0512),
    (Security, 0x0513),
];

const DOMAINS: [Domain; 5] = [Cluster, Net, Dist, Recovery, Security];

fn registry() -> Vec<&'static CodeInfo> {
    codes::all().collect()
}

#[test]
fn registry_matches_the_rfc_tables() {
    let got: Vec<_> = registry()
        .into_iter()
        .map(|i| (i.code.domain, i.code.code, i.names, i.class))
        .collect();
    assert_eq!(got.len(), PINNED.len(), "a code was added or removed");
    for (got, want) in got.iter().zip(PINNED) {
        assert_eq!(got, want, "renumbered, renamed or reclassed");
    }
}

#[test]
fn numbers_are_unique_and_never_reuse_a_reserved_one() {
    let mut seen = BTreeSet::new();
    for info in registry() {
        assert!(seen.insert(info.code), "{} is used twice", info.code);
    }
    for reserved in codes::RESERVED {
        assert!(seen.insert(*reserved), "reserved {reserved} is assigned");
    }
    assert_eq!(
        codes::RESERVED,
        [0x0509, 0x050A, 0x050B].map(|n| ErrorCode::new(Security, n))
    );
}

#[test]
fn names_are_unique() {
    let mut seen = BTreeSet::new();
    for info in registry() {
        assert!(!info.names.is_empty(), "{} has no name", info.code);
        for name in info.names {
            assert!(seen.insert(*name), "{name} names two codes");
        }
    }
}

#[test]
fn every_code_is_in_its_domains_range() {
    assert_eq!(Cluster.range(), 0x0100..=0x01FF);
    assert_eq!(Net.range(), 0x0200..=0x02FF);
    assert_eq!(Dist.range(), 0x0300..=0x03FF);
    assert_eq!(Recovery.range(), 0x0400..=0x04FF);
    assert_eq!(Security.range(), 0x0500..=0x05FF);
    for info in registry()
        .into_iter()
        .map(|i| i.code)
        .chain(codes::RESERVED.iter().copied())
    {
        assert!(
            info.domain.range().contains(&info.code),
            "{info} is outside its domain's range"
        );
    }
    // The ranges do not overlap, so a number alone names its domain.
    for (i, a) in DOMAINS.iter().enumerate() {
        for b in &DOMAINS[i + 1..] {
            assert!(a.range().end() < b.range().start());
        }
    }
}

#[test]
fn registry_is_in_domain_then_number_order() {
    let codes: Vec<_> = registry().into_iter().map(|i| i.code).collect();
    assert!(codes.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn constants_name_their_codes() {
    use codes::{cluster, dist, net, recovery, security};
    assert_eq!(
        cluster::IDENTITY_HOST_MISMATCH,
        ErrorCode::new(Cluster, 0x0101)
    );
    assert_eq!(cluster::RATE_LIMITED, ErrorCode::new(Cluster, 0x010E));
    assert_eq!(net::UNKNOWN_FRAME_TYPE, ErrorCode::new(Net, 0x0204));
    assert_eq!(net::QP_BUDGET_EXHAUSTED, ErrorCode::new(Net, 0x0213));
    assert_eq!(dist::DUPLICATE_COMMAND, ErrorCode::new(Dist, 0x0305));
    assert_eq!(dist::CONFIG_CONFLICT, ErrorCode::new(Dist, 0x030F));
    assert_eq!(
        recovery::STATE_POSITION_MISMATCH,
        ErrorCode::new(Recovery, 0x0409)
    );
    assert_eq!(
        security::NOT_COORDINATOR_ELIGIBLE,
        ErrorCode::new(Security, 0x0513)
    );
    // One row, several names, one code.
    assert_eq!(net::BLOB_BUSY, net::BLOB_NOT_FOUND);
    assert_eq!(net::BLOB_THROTTLED, net::BLOB_NOT_FOUND);
    assert_eq!(security::MODEL_UNSIGNED, security::MODEL_SIGNATURE_INVALID);
    assert_eq!(codes::lookup(net::BLOB_BUSY).unwrap().names[1], "BlobBusy");
    assert_eq!(codes::lookup(ErrorCode::new(Security, 0x0509)), None);
}

#[test]
fn every_code_maps_to_an_api_shape_or_is_on_the_open_list() {
    let unmapped: BTreeSet<_> = UNMAPPED
        .iter()
        .map(|&(d, n)| ErrorCode::new(d, n))
        .collect();
    for info in registry() {
        let mapping = api::for_code(info.code).expect("a published code");
        assert_eq!(
            mapping == ApiMapping::Unmapped,
            unmapped.contains(&info.code),
            "{} ({}) maps to {mapping:?}",
            info.code,
            info.names[0]
        );
    }
    assert_eq!(api::for_code(ErrorCode::new(Security, 0x0509)), None);
    let mapped = registry().len() - UNMAPPED.len();
    assert_eq!((mapped, UNMAPPED.len()), (36, 37));
}

#[test]
fn class_shapes_match_rfc_0050() {
    let unavailable = ApiMapping::Error(ApiError {
        status: 503,
        retry_after: Some(1),
        openai_type: "server_error",
        openai_code: "instance_unavailable",
        anthropic_type: "overloaded_error",
        anthropic_status: 529,
    });
    for class in [Link, Device, Integrity, Drift, Transient] {
        assert_eq!(api::for_class(class), unavailable, "{class:?}");
    }
    assert_eq!(
        api::for_class(Timeout),
        ApiMapping::Error(ApiError {
            status: 504,
            retry_after: None,
            openai_type: "server_error",
            openai_code: "timeout",
            anthropic_type: "timeout_error",
            anthropic_status: 504,
        })
    );
    assert_eq!(
        api::for_class(Correctness),
        ApiMapping::Error(ApiError {
            status: 503,
            retry_after: None,
            openai_type: "server_error",
            openai_code: "no_conforming_instance",
            anthropic_type: "api_error",
            anthropic_status: 500,
        })
    );
    assert_eq!(
        api::for_class(Internal),
        ApiMapping::Error(ApiError {
            status: 500,
            retry_after: None,
            openai_type: "server_error",
            openai_code: "internal_error",
            anthropic_type: "api_error",
            anthropic_status: 500,
        })
    );
    assert_eq!(api::for_class(Cancelled), ApiMapping::NormalFinish);
    for class in [Auth, Capacity, InvalidRequest, Unsupported] {
        assert_eq!(api::for_class(class), ApiMapping::Unmapped, "{class:?}");
    }
}

#[test]
fn error_code_round_trips_and_displays() {
    let code = codes::net::CRC_STORM;
    let json = serde_json::to_string(&code).expect("serialize");
    assert_eq!(json, r#"{"domain":"Net","code":519}"#);
    assert_eq!(
        serde_json::from_str::<ErrorCode>(&json).expect("decode"),
        code
    );
    assert_eq!(code.to_string(), "Net 0x0207");
    assert_eq!(Recovery.source(), "RFC-0033 §13");
}
