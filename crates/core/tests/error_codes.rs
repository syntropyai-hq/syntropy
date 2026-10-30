// SPDX-License-Identifier: MIT OR Apache-2.0

//! The error-code registry (DIST-E18-T02): every published code is pinned to its number, its
//! names and its class, numbers are unique and inside their domain's range, and every code
//! maps to an RFC-0050 §12.1 shape or is on §12.1's exempt list.

use std::collections::BTreeSet;

use syntropy_core::error::api::{self, ApiError, ApiMapping, NativeError, RetryAfter};
use syntropy_core::error::codes::{self, CodeInfo};
use syntropy_core::{Domain, ErrorClass, ErrorCode};

use Domain::{Cluster, Dist, Exec, Kernels, Net, Recovery, Security};
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
    (Cluster, 0x010F, &["TokenRevoked"], Some(Auth)),
    // RFC-0031 §12
    (Net, 0x0201, &["ProtoMajorMismatch"], Some(Unsupported)),
    (Net, 0x0202, &["CredentialInvalid"], Some(Auth)),
    (Net, 0x0203, &["NotANeighbour"], Some(Auth)),
    (Net, 0x0204, &["UnknownFrameType"], Some(Unsupported)),
    (Net, 0x0205, &["FrameTooLarge"], Some(InvalidRequest)),
    (Net, 0x0206, &["WireDTypeMismatch"], Some(Unsupported)),
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
    // RFC-0040 §13
    (Kernels, 1, &["OpUnsupported"], Some(Unsupported)),
    (Kernels, 2, &["ModuleMissing"], Some(Unsupported)),
    (Kernels, 3, &["ModuleIntegrity"], Some(Integrity)),
    (Kernels, 4, &["ModuleAbi"], Some(Unsupported)),
    (Kernels, 5, &["DriverTooOld"], Some(Unsupported)),
    (Kernels, 6, &["ParityFailed"], Some(Correctness)),
    (Kernels, 7, &["CaptureFailed"], Some(Unsupported)),
    (Kernels, 8, &["DeviceSticky"], Some(Device)),
    (Kernels, 9, &["AllocFailed"], Some(Capacity)),
    (Kernels, 10, &["RegisterFailed"], Some(Capacity)),
    (Kernels, 11, &["DoorbellTimeout"], Some(Timeout)),
    (Kernels, 12, &["GgmlImportFailed"], Some(Unsupported)),
    (Kernels, 13, &["AmxDenied"], Some(Unsupported)),
    (Kernels, 14, &["WorkspaceExceeded"], Some(Internal)),
    (Kernels, 15, &["NoCertifiedCandidate"], Some(Unsupported)),
    (Kernels, 16, &["ProviderPanicked"], Some(Internal)),
    (Kernels, 17, &["ManifestInvalid"], Some(Integrity)),
    // RFC-0012 §13, note of 2026-10-28
    (Exec, 0x0601, &["RingClosed"], Some(Internal)),
    (Exec, 0x0602, &["StepStalled"], Some(Internal)),
    (Exec, 0x0603, &["PoolBarrierMissed"], Some(Internal)),
];

/// A 503 `instance_unavailable` whose `Retry-After` the row does not fix: RFC-0050 §12.1's
/// row for unroutable instances and infeasible repair.
const UNAVAILABLE_CAPACITY: ApiMapping = ApiMapping::Error(ApiError {
    status: 503,
    retry_after: RetryAfter::Unfixed,
    openai_type: "server_error",
    openai_code: "instance_unavailable",
    anthropic_type: "overloaded_error",
    anthropic_status: 529,
});
/// 503 `instance_unavailable`, `Retry-After: 1`: the `Link`/`Device` row, the escape rule,
/// and the shape an exempt code renders as if it escapes.
const UNAVAILABLE: ApiError = ApiError {
    status: 503,
    retry_after: RetryAfter::Seconds(1),
    openai_type: "server_error",
    openai_code: "instance_unavailable",
    anthropic_type: "overloaded_error",
    anthropic_status: 529,
};
const NATIVE_400: ApiMapping = ApiMapping::Native(NativeError {
    status: 400,
    api_code: None,
});
const NATIVE_403: ApiMapping = ApiMapping::Native(NativeError {
    status: 403,
    api_code: None,
});
const NATIVE_409: ApiMapping = ApiMapping::Native(NativeError {
    status: 409,
    api_code: Some("cluster_proposal_rejected"),
});

/// The codes RFC-0050 §12.1 names one by one, with words its text uses for them: the 13 codes
/// its table rows name, plus `DuplicateCommand` and `IngressLost` from the notes below the
/// table. `pinned_tables_match_the_rfc_text` checks this table against the RFC.
const PINNED_ROWS: &[(Domain, u16, ApiMapping, &str)] = &[
    (
        Cluster,
        0x0101,
        NATIVE_403,
        "a join the keyholder or the joining node refuses",
    ),
    (
        Cluster,
        0x0103,
        NATIVE_400,
        "a join refused because the two nodes' protocol majors differ",
    ),
    (
        Cluster,
        0x0104,
        NATIVE_400,
        "`POST /v1/syntropy/cluster/join` with a malformed invite token",
    ),
    (
        Cluster,
        0x0105,
        NATIVE_403,
        "a join the keyholder or the joining node refuses",
    ),
    (
        Cluster,
        0x0106,
        NATIVE_403,
        "a join the keyholder or the joining node refuses",
    ),
    (
        Cluster,
        0x0107,
        NATIVE_403,
        "a join the keyholder or the joining node refuses",
    ),
    (
        Cluster,
        0x010C,
        NATIVE_403,
        "a join the keyholder or the joining node refuses",
    ),
    (
        Cluster,
        0x010F,
        NATIVE_403,
        "a join the keyholder or the joining node refuses",
    ),
    (
        Dist,
        0x0303,
        NATIVE_409,
        "a cluster configuration proposal rejected",
    ),
    (
        Dist,
        0x0305,
        ApiMapping::NeverSurfaces,
        "`DuplicateCommand` (0x0305) never appears in `Error::code`",
    ),
    (
        Dist,
        0x0308,
        UNAVAILABLE_CAPACITY,
        "the only instances of the model are unroutable",
    ),
    (
        Dist,
        0x030F,
        NATIVE_409,
        "a cluster configuration proposal rejected",
    ),
    (
        Recovery,
        0x0404,
        ApiMapping::NoResponse,
        "`IngressLost` (0x0404), has no response to carry it",
    ),
    (
        Recovery,
        0x0405,
        UNAVAILABLE_CAPACITY,
        "recovery found no feasible repair",
    ),
    (
        Security,
        0x0507,
        NATIVE_403,
        "an operator action that needs a keyholder signature with no keyholder reachable",
    ),
];

/// RFC-0050 §12.1's exempt list: the codes that never leave the cluster.
const PINNED_EXEMPT: &[(Domain, u16)] = &[
    (Cluster, 0x0102),
    (Cluster, 0x010A),
    (Cluster, 0x010B),
    (Net, 0x0201),
    (Net, 0x0202),
    (Net, 0x0203),
    (Net, 0x0204),
    (Net, 0x0205),
    (Net, 0x0206),
    (Net, 0x020D),
    (Net, 0x020E),
    (Net, 0x020F),
    (Net, 0x0212),
    (Dist, 0x030E),
    (Recovery, 0x0407),
    (Security, 0x0501),
    (Security, 0x0502),
    (Security, 0x0503),
    (Security, 0x0504),
    (Security, 0x0505),
    (Security, 0x0506),
    (Security, 0x0508),
    (Security, 0x0510),
    (Security, 0x0511),
    (Security, 0x0512),
    (Security, 0x0513),
    (Kernels, 0x0001),
    (Kernels, 0x0002),
    (Kernels, 0x0004),
    (Kernels, 0x0005),
    (Kernels, 0x0007),
    (Kernels, 0x0009),
    (Kernels, 0x000A),
    (Kernels, 0x000C),
    (Kernels, 0x000D),
    (Kernels, 0x000F),
];

/// The whole-class rows of RFC-0050 §12.1 (and §12.2 for `Cancelled`), transcribed; `None`
/// where every row of the class names a narrower condition.
fn class_row(class: ErrorClass) -> Option<ApiMapping> {
    let server = |status, openai_code, anthropic_type, anthropic_status| {
        Some(ApiMapping::Error(ApiError {
            status,
            retry_after: RetryAfter::Absent,
            openai_type: "server_error",
            openai_code,
            anthropic_type,
            anthropic_status,
        }))
    };
    match class {
        Link | Device | Integrity | Drift | Transient => Some(ApiMapping::Error(UNAVAILABLE)),
        Timeout => server(504, "timeout", "timeout_error", 504),
        Correctness => server(503, "no_conforming_instance", "api_error", 500),
        Internal => server(500, "internal_error", "api_error", 500),
        Cancelled => Some(ApiMapping::NormalFinish),
        Auth | Capacity | InvalidRequest | Unsupported => None,
        other => panic!("{other:?} is a class this transcription predates"),
    }
}

fn is_note(mapping: ApiMapping) -> bool {
    matches!(mapping, ApiMapping::NeverSurfaces | ApiMapping::NoResponse)
}

const DOMAINS: [Domain; 7] = Domain::ALL;

/// Every `ErrorClass` of RFC-0001 §5.12.
const CLASSES: [ErrorClass; 13] = [
    Link,
    Device,
    Integrity,
    Drift,
    Transient,
    Timeout,
    Correctness,
    Internal,
    Cancelled,
    Auth,
    Capacity,
    InvalidRequest,
    Unsupported,
];

fn registry() -> Vec<&'static CodeInfo> {
    codes::all().collect()
}

/// `Domain::ALL` lists every variant once, in declaration order. Postcard writes a variant as
/// its index, so the indices that decode are exactly the variants: a variant added to `Domain`
/// and not to `ALL` fails here.
#[test]
fn domain_all_lists_every_variant() {
    let decoded: Vec<Domain> = (0u8..=127)
        .map_while(|i| postcard::from_bytes::<Domain>(&[i]).ok())
        .collect();
    assert_eq!(decoded, Domain::ALL);
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
    assert_eq!(Kernels.range(), 0x0001..=0x00FF);
    assert_eq!(Exec.range(), 0x0600..=0x06FF);
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
            assert!(
                a.range().end() < b.range().start() || b.range().end() < a.range().start(),
                "{a:?} and {b:?} overlap"
            );
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
    use codes::{cluster, dist, exec, kernels, net, recovery, security};
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
    assert_eq!(kernels::OP_UNSUPPORTED, ErrorCode::new(Kernels, 1));
    assert_eq!(kernels::MANIFEST_INVALID, ErrorCode::new(Kernels, 17));
    assert_eq!(exec::POOL_BARRIER_MISSED, ErrorCode::new(Exec, 0x0603));
}

#[test]
fn per_code_rows_follow_rfc_0050() {
    let got: Vec<_> = api::per_code().collect();
    let want: Vec<_> = PINNED_ROWS
        .iter()
        .map(|&(d, n, mapping, _)| (ErrorCode::new(d, n), mapping))
        .collect();
    assert_eq!(
        got.iter().map(|&(c, m, _)| (c, m)).collect::<Vec<_>>(),
        want
    );
    for ((code, mapping, row), &(.., words)) in got.into_iter().zip(PINNED_ROWS) {
        assert!(row.contains(words), "{code}: {row:?} quotes {words:?}");
        assert_eq!(api::for_code(code), Some(mapping), "{code}: {row}");
    }
}

#[test]
fn exempt_list_follows_rfc_0050() {
    let want: Vec<_> = PINNED_EXEMPT
        .iter()
        .map(|&(d, n)| ErrorCode::new(d, n))
        .collect();
    assert_eq!(api::exempt().collect::<Vec<_>>(), want);
    for code in want {
        assert_eq!(
            api::for_code(code),
            Some(ApiMapping::Exempt {
                if_escaped: UNAVAILABLE
            }),
            "{code}"
        );
    }
}

#[test]
fn every_code_maps_to_an_rfc_0050_shape_or_is_exempt() {
    let exempt: BTreeSet<_> = PINNED_EXEMPT
        .iter()
        .map(|&(d, n)| ErrorCode::new(d, n))
        .collect();
    let (mut by_row, mut by_class) = (0, 0);
    for info in registry() {
        let row = PINNED_ROWS
            .iter()
            .find(|&&(d, n, ..)| ErrorCode::new(d, n) == info.code);
        let want = match row {
            Some(&(.., mapping, _)) => {
                by_row += 1;
                assert!(!exempt.contains(&info.code), "{} is on both", info.code);
                Some(mapping)
            }
            None if exempt.contains(&info.code) => Some(ApiMapping::Exempt {
                if_escaped: UNAVAILABLE,
            }),
            None => {
                by_class += 1;
                let class = info.class.expect("only DuplicateCommand has no class");
                let shape = class_row(class);
                assert!(
                    shape.is_some(),
                    "{} ({}, {class:?}) has no §12.1 row, no whole-class row and is not exempt",
                    info.code,
                    info.names[0]
                );
                shape
            }
        };
        assert_eq!(
            api::for_code(info.code),
            want,
            "{} ({})",
            info.code,
            info.names[0]
        );
    }
    assert_eq!(api::for_code(ErrorCode::new(Security, 0x0509)), None);
    // 13 codes by table rows plus `DuplicateCommand` and `IngressLost`; 26 exempt in the
    // distributed domains and 10 in `Kernels`; the rest by class, 7 of them `Kernels` codes
    // and 3 `Exec` codes.
    assert_eq!((by_row, exempt.len(), by_class), (15, 36, 43));
}

#[test]
fn class_rows_match_rfc_0050() {
    for class in CLASSES {
        assert_eq!(api::for_class(class), class_row(class), "{class:?}");
    }
}

/// An RFC's text, read from the plan.
fn rfc(file: &str) -> String {
    let path = format!("{}/../../plan/rfcs/{file}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The file of the RFC that publishes `domain`'s codes (`Domain::source`).
fn code_table_rfc(domain: Domain) -> &'static str {
    match domain {
        Cluster => "RFC-0030-discovery-membership-invites.md",
        Net => "RFC-0031-transport-and-wire-protocol.md",
        Dist => "RFC-0032-distributed-execution.md",
        Recovery => "RFC-0033-fault-tolerance-and-elasticity.md",
        Security => "RFC-0034-security-and-trust.md",
        Kernels => "RFC-0040-kernel-providers-and-backends.md",
        Exec => "RFC-0012-execution-scheduler.md",
        other => panic!("{other:?} is a domain this test predates"),
    }
}

/// Whether `word` is one CamelCase identifier, such as a code's or a class's name.
fn is_camel_case(word: &str) -> bool {
    word.starts_with(|c: char| c.is_ascii_uppercase())
        && word.chars().all(|c| c.is_ascii_alphanumeric())
}

/// Reads every code row, `| 0x0NNN | \`Name\`… | \`Class\`… |` (`| N | …` in RFC-0040 §13),
/// out of the seven domain RFCs' tables and checks the registry against them, so a code an RFC adds, renames or reclasses
/// fails here even when `PINNED` was not updated.
#[test]
fn registry_matches_the_rfc_code_tables() {
    let mut from_rfcs = Vec::new();
    for domain in DOMAINS {
        let text = rfc(code_table_rfc(domain));
        // RFC-0040 and RFC-0012 have other tables, so their codes are read from §13 alone.
        // RFC-0040 numbers its codes in decimal.
        let text = match domain {
            Kernels | Exec => section_13(&text),
            _ => text,
        };
        let rows = text.lines().filter(|l| match domain {
            Kernels => l.starts_with("| ") && l[2..].starts_with(|c: char| c.is_ascii_digit()),
            _ => l.starts_with("| 0x0"),
        });
        for line in rows {
            let cells: Vec<_> = line.trim_matches('|').split(" | ").map(str::trim).collect();
            let number = match domain {
                Kernels => cells[0].parse::<u16>().ok(),
                _ => u16::from_str_radix(cells[0].trim_start_matches("0x"), 16).ok(),
            };
            let Some(number) = number else {
                continue; // a range such as the reserved 0x0509–0x050B
            };
            if !domain.range().contains(&number) {
                continue;
            }
            // The names: each backticked word that starts a CamelCase identifier, so a field
            // list (`FeatureLevelTooLow { needed, cluster }`) keeps only its name.
            let names: Vec<_> = backticked(cells[1])
                .filter_map(|w| w.split_whitespace().next())
                .filter(|w| is_camel_case(w))
                .map(str::to_owned)
                .collect();
            // The class: the first backticked word of the cell; "—" is no class.
            let class = backticked(cells[2]).next().map(|name| {
                *CLASSES
                    .iter()
                    .find(|c| format!("{c:?}") == name)
                    .unwrap_or_else(|| panic!("{domain:?} {number:#06X}: class {name:?}"))
            });
            from_rfcs.push((ErrorCode::new(domain, number), names, class));
        }
    }
    let registry: Vec<_> = registry()
        .into_iter()
        .map(|i| {
            (
                i.code,
                i.names.iter().map(|n| n.to_string()).collect(),
                i.class,
            )
        })
        .collect();
    assert_eq!(
        registry, from_rfcs,
        "the registry and the RFC tables differ"
    );
}

/// An RFC's §13, from its heading to the next `## ` or `### ` heading.
fn section_13(text: &str) -> String {
    let start = text.find("\n### 13.").expect("§13") + 1;
    let rest = &text[start + 4..];
    let end = rest
        .find("\n## ")
        .into_iter()
        .chain(rest.find("\n### "))
        .min()
        .unwrap_or(rest.len());
    text[start..start + 4 + end].to_owned()
}

/// RFC-0050 §12.1, read from the plan.
fn rfc_0050_section_12_1() -> String {
    let text = rfc("RFC-0050-api-surface.md");
    let start = text.find("#### 12.1").expect("§12.1");
    let end = start + text[start..].find("#### 12.2").expect("§12.2");
    text[start..end].to_owned()
}

/// The backticked words of `text`.
fn backticked(text: &str) -> impl Iterator<Item = &str> {
    text.split('`').skip(1).step_by(2)
}

fn code_named(name: &str) -> Option<ErrorCode> {
    registry()
        .into_iter()
        .find(|info| info.names.contains(&name))
        .map(|info| info.code)
}

/// Reads the codes, statuses and exempt list out of RFC-0050 §12.1's text and checks them
/// against `PINNED_ROWS` and `PINNED_EXEMPT`, so the pinned tables cannot drift from the RFC.
#[test]
fn pinned_tables_match_the_rfc_text() {
    let section = rfc_0050_section_12_1();

    // Every CamelCase name the section puts in backticks is a published code, a class, a
    // domain or one of a few other types it names. A code the RFC adds to a row or to the
    // exempt list without a registry entry fails here.
    const OTHER_TYPES: [&str; 6] = [
        "BenchControls",
        "Close",
        "CommandId",
        "Error",
        "ErrorClass",
        "Never",
    ];
    for name in backticked(&section).filter(|w| is_camel_case(w)) {
        assert!(
            code_named(name).is_some()
                || CLASSES.iter().any(|c| format!("{c:?}") == name)
                || DOMAINS.iter().any(|d| format!("{d:?}") == name)
                || OTHER_TYPES.contains(&name),
            "RFC-0050 §12.1 names `{name}`, which is no published code"
        );
    }

    // Every code a table row names in its condition cell.
    let mut named = BTreeSet::new();
    for line in section.lines().filter(|l| l.starts_with("| ")) {
        let cells: Vec<_> = line.split(" | ").collect();
        let status: Option<u16> = cells[2]
            .split(|c: char| !c.is_ascii_digit())
            .find(|s| !s.is_empty())
            .and_then(|s| s.parse().ok());
        let native = cells[3].contains("native route only");
        let proposal = cells[0].contains("`cluster_proposal_rejected`");
        for code in backticked(cells[0]).filter_map(code_named) {
            assert!(named.insert(code), "{code} is in two rows");
            let &(.., mapping, words) = PINNED_ROWS
                .iter()
                .find(|&&(d, n, ..)| ErrorCode::new(d, n) == code)
                .unwrap_or_else(|| panic!("{code} has a §12.1 row but is not pinned"));
            assert!(cells[0].contains(words), "{code}: {:?}", cells[0]);
            match mapping {
                ApiMapping::Native(e) => {
                    assert!(native, "{code}");
                    assert_eq!(Some(e.status), status, "{code}");
                    assert_eq!(e.api_code.is_some(), proposal, "{code}");
                }
                ApiMapping::Error(e) => {
                    assert!(!native, "{code}");
                    assert_eq!(Some(e.status), status, "{code}");
                }
                other => panic!("{code}: a table row gives {other:?}"),
            }
        }
    }
    let pinned_rows: BTreeSet<_> = PINNED_ROWS
        .iter()
        .filter(|&&(.., m, _)| !is_note(m))
        .map(|&(d, n, ..)| ErrorCode::new(d, n))
        .collect();
    assert_eq!(named, pinned_rows);

    // The notes below the table that name one code each.
    for &(.., words) in PINNED_ROWS.iter().filter(|&&(.., m, _)| is_note(m)) {
        assert!(section.contains(words), "{words}");
    }

    // The exempt list: the bullet lines "  - `Domain`: 0x0NNN `Name`, …".
    let mut exempt = Vec::new();
    for line in section
        .lines()
        .filter(|l| l.starts_with("  - `") && l.contains(" 0x0"))
    {
        let (_, items) = line.split_once(": ").expect("domain: codes");
        for item in items.split(", ") {
            let (number, name) = item.split_once(' ').expect("number and name");
            let number = u16::from_str_radix(number.trim_start_matches("0x"), 16).expect("hex");
            let name = backticked(name).next().expect("a backticked name");
            let code = code_named(name).expect("a published name");
            assert_eq!(code.code, number, "{name}");
            exempt.push(code);
        }
    }
    let pinned_exempt: Vec<_> = PINNED_EXEMPT
        .iter()
        .map(|&(d, n)| ErrorCode::new(d, n))
        .collect();
    assert_eq!(exempt, pinned_exempt);
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
