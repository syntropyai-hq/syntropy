// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serde properties of `NodeInfo` and its field types (DIST-E2-T10; RFC-0001 §5.2).
//!
//! The cluster carries `NodeInfo` in deterministic CBOR (ADR-0305), and nodes of adjacent builds
//! read each other's records (RFC-0031 §10's N±1 window). So every type round-trips through CBOR
//! and JSON, a record with fields this build does not know decodes, an `OsKind` this build does not
//! know decodes as `Other`, an inverted version range is refused, and absent role fields take
//! RFC-0054's defaults.

use std::collections::BTreeMap;
use std::fmt::Debug;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use syntropy_core::build::{BuildId, BuildProfile, TargetTriple};
use syntropy_core::resource::*;
use syntropy_core::{Blake3, NodeId};

fn to_cbor<T: Serialize>(value: &T) -> Vec<u8> {
    let mut bytes = Vec::new();
    ciborium::into_writer(value, &mut bytes).expect("CBOR encode");
    bytes
}

fn from_cbor<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    ciborium::from_reader(bytes).map_err(|e| e.to_string())
}

/// `value` round-trips through CBOR and JSON unchanged.
fn round_trips<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let back: T = from_cbor(&to_cbor(value)).expect("CBOR decode");
    assert_eq!(&back, value, "CBOR");
    let json = serde_json::to_string(value).unwrap();
    let back: T = serde_json::from_str(&json).expect("JSON decode");
    assert_eq!(&back, value, "JSON: {json}");
}

fn os_kind() -> impl Strategy<Value = OsKind> {
    prop_oneof![
        Just(OsKind::Linux),
        Just(OsKind::MacOs),
        Just(OsKind::Windows),
        Just(OsKind::Android),
        Just(OsKind::Ios),
        Just(OsKind::Other),
    ]
}

fn range_u8() -> impl Strategy<Value = VersionRange<u8>> {
    (any::<u8>(), any::<u8>()).prop_map(|(a, b)| VersionRange::new(a.min(b), a.max(b)).unwrap())
}

fn range_u32() -> impl Strategy<Value = VersionRange<u32>> {
    (any::<u32>(), any::<u32>()).prop_map(|(a, b)| VersionRange::new(a.min(b), a.max(b)).unwrap())
}

fn protocol() -> impl Strategy<Value = ProtocolRanges> {
    (any::<u16>(), range_u8(), range_u32(), any::<u64>()).prop_map(
        |(proto_major, wire_minor, feature_level, capabilities)| ProtocolRanges {
            proto_major,
            wire_minor,
            feature_level,
            capabilities,
        },
    )
}

fn roles() -> impl Strategy<Value = RolePrefs> {
    (any::<bool>(), any::<bool>(), any::<bool>()).prop_map(|(c, i, h)| RolePrefs {
        may_coordinate: c,
        may_ingress: i,
        may_host_stages: h,
    })
}

fn build() -> BuildId {
    BuildId {
        version: "0.4.1".parse().unwrap(),
        commit: core::array::from_fn(|i| i as u8),
        profile: BuildProfile::Release,
        target: TargetTriple("aarch64-unknown-linux-gnu".into()),
        toolchain: Blake3([7; 32]),
    }
}

fn node_info() -> impl Strategy<Value = NodeInfo> {
    (
        any::<[u8; 32]>(),
        any::<[u8; 32]>(),
        os_kind(),
        prop_oneof![Just(CpuArch::X86_64), Just(CpuArch::Aarch64)],
        protocol(),
        roles(),
        prop::collection::btree_map("[a-z]{1,4}", "[a-z0-9]{0,4}", 0..3),
        prop_oneof![
            Just(PowerSource::Mains),
            (0u8..=100).prop_map(|percent| PowerSource::Battery { percent })
        ],
    )
        .prop_map(
            |(id, host, os, arch, protocol, roles, labels, power)| NodeInfo {
                id: NodeId(id),
                host: HostFingerprint(Blake3(host)),
                os,
                arch,
                build: build(),
                protocol,
                power,
                labels,
                roles,
                inventory: NodeInventory::default(),
            },
        )
}

proptest! {
    #[test]
    fn node_info_round_trips(info in node_info()) {
        round_trips(&info);
    }

    #[test]
    fn field_types_round_trip(os in os_kind(), p in protocol(), r in roles(), h in any::<[u8; 32]>()) {
        round_trips(&os);
        round_trips(&p);
        round_trips(&r);
        round_trips(&HostFingerprint(Blake3(h)));
    }

    /// A record from a newer build, carrying fields this build does not know, decodes in CBOR
    /// and in JSON, and keeps every field it does know.
    #[test]
    fn a_record_with_future_fields_decodes(info in node_info()) {
        let mut json = serde_json::to_value(&info).unwrap();
        json.as_object_mut().unwrap().insert("from_a_newer_build".into(), serde_json::json!({"x": [1, 2]}));
        let back: NodeInfo = serde_json::from_value(json.clone()).unwrap();
        prop_assert_eq!(&back, &info);
        // The same record as CBOR, built from the struct itself (a `Blake3` is hex text in a
        // human-readable format and bytes in CBOR), with one more map entry.
        let mut value = ciborium::Value::serialized(&info).unwrap();
        match &mut value {
            ciborium::Value::Map(entries) => entries.push((
                ciborium::Value::Text("from_a_newer_build".into()),
                ciborium::Value::Array(vec![ciborium::Value::Integer(1.into())]),
            )),
            other => panic!("a NodeInfo is a CBOR map, not {other:?}"),
        }
        let back: NodeInfo = from_cbor(&to_cbor(&value)).unwrap();
        prop_assert_eq!(&back, &info);
    }
}

/// An `OsKind` from a newer build decodes as `Other`, in CBOR and JSON, so an older node still
/// reads the record.
#[test]
fn an_unknown_os_decodes_as_other() {
    let back: OsKind = from_cbor(&to_cbor(&"Fuchsia")).unwrap();
    assert_eq!(back, OsKind::Other);
    let back: OsKind = serde_json::from_str("\"Fuchsia\"").unwrap();
    assert_eq!(back, OsKind::Other);
    // A known one still decodes as itself.
    let back: OsKind = from_cbor(&to_cbor(&"Linux")).unwrap();
    assert_eq!(back, OsKind::Linux);
}

/// A range whose `min_supported` is above `current` is refused, when built and when decoded,
/// for the wire minor and the feature level.
#[test]
fn an_inverted_range_is_refused() {
    assert!(VersionRange::new(3u8, 2).is_err());
    assert!(VersionRange::new(9u32, 1).is_err());
    assert!(VersionRange::new(2u8, 2).is_ok());

    #[derive(Serialize)]
    struct Raw<T> {
        min_supported: T,
        current: T,
    }
    let wire = to_cbor(&Raw {
        min_supported: 5u8,
        current: 4u8,
    });
    let err = from_cbor::<VersionRange<u8>>(&wire).unwrap_err();
    assert!(err.contains("above its current"), "{err}");
    let level = to_cbor(&Raw {
        min_supported: 100u32,
        current: 7u32,
    });
    assert!(from_cbor::<VersionRange<u32>>(&level).is_err());
    assert!(
        serde_json::from_str::<VersionRange<u8>>(r#"{"min_supported":5,"current":4}"#).is_err()
    );

    // Inside a whole `ProtocolRanges`, for either range, against a valid control that decodes,
    // so a refusal is the range check and not a misnamed field.
    let protocol = |wire: (u8, u8), level: (u32, u32)| {
        serde_json::json!({
            "proto_major": 1,
            "wire_minor": {"min_supported": wire.0, "current": wire.1},
            "feature_level": {"min_supported": level.0, "current": level.1},
            "capabilities": 0
        })
    };
    let decodes = |value: &serde_json::Value| {
        let json = serde_json::from_value::<ProtocolRanges>(value.clone()).is_ok();
        let cbor = from_cbor::<ProtocolRanges>(&to_cbor(value)).is_ok();
        assert_eq!(json, cbor, "JSON and CBOR agree on {value}");
        json
    };
    assert!(decodes(&protocol((0, 3), (2, 9))), "the control decodes");
    assert!(
        !decodes(&protocol((4, 3), (2, 9))),
        "an inverted wire_minor"
    );
    assert!(
        !decodes(&protocol((0, 3), (9, 2))),
        "an inverted feature_level"
    );
}

/// Absent role fields take RFC-0054's defaults: every role allowed.
#[test]
fn absent_roles_take_their_defaults() {
    let empty: BTreeMap<String, bool> = BTreeMap::new();
    let back: RolePrefs = from_cbor(&to_cbor(&empty)).unwrap();
    assert_eq!(back, RolePrefs::default());
    assert_eq!(
        back,
        RolePrefs {
            may_coordinate: true,
            may_ingress: true,
            may_host_stages: true,
        }
    );
    let partial: RolePrefs = serde_json::from_str(r#"{"may_ingress": false}"#).unwrap();
    assert_eq!(
        partial,
        RolePrefs {
            may_coordinate: true,
            may_ingress: false,
            may_host_stages: true,
        }
    );
}

/// The zero digest means "not computed", and two uncomputed fingerprints never count as one
/// host.
#[test]
fn an_uncomputed_fingerprint_never_matches() {
    let zero = HostFingerprint::NOT_COMPUTED;
    assert!(!zero.is_computed());
    assert!(!zero.same_host(&HostFingerprint::default()));
    let a = HostFingerprint(Blake3([1; 32]));
    assert!(a.is_computed());
    assert!(a.same_host(&HostFingerprint(Blake3([1; 32]))));
    assert!(!a.same_host(&HostFingerprint(Blake3([2; 32]))));
    assert!(!a.same_host(&zero));
}

/// One `NodeInfo` drawn from the strategy with a fixed seed.
fn sample() -> NodeInfo {
    use proptest::strategy::ValueTree;
    node_info()
        .new_tree(&mut proptest::test_runner::TestRunner::deterministic())
        .unwrap()
        .current()
}

/// One value always encodes to the same bytes: encoding, decoding and encoding again gives the
/// same CBOR, and `labels` come out in key order whatever order they were inserted in.
#[test]
fn a_node_info_encodes_to_stable_bytes() {
    let mut info = sample();
    info.labels = [("zone", "b"), ("aa", "x"), ("b", "y")]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
    let first = to_cbor(&info);
    let again = to_cbor(&from_cbor::<NodeInfo>(&first).unwrap());
    assert_eq!(first, again);
    let keys: Vec<&String> = info.labels.keys().collect();
    assert_eq!(keys, ["aa", "b", "zone"]);
}

/// The fields with defaults may be absent: `labels`, `roles` and `inventory` take theirs. `power`
/// is required, since an absent value would rank as mains in the coordinator election.
#[test]
fn absent_optional_fields_take_defaults_and_power_is_required() {
    let info = sample();
    let mut json = serde_json::to_value(&info).unwrap();
    let object = json.as_object_mut().unwrap();
    for key in ["labels", "roles", "inventory"] {
        object.remove(key);
    }
    let back: NodeInfo = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(back.labels, BTreeMap::new());
    assert_eq!(back.roles, RolePrefs::default());
    assert_eq!(back.inventory, NodeInventory::default());
    assert_eq!(
        (back.id, back.os, back.power),
        (info.id, info.os, info.power)
    );
    json.as_object_mut().unwrap().remove("power");
    let err = serde_json::from_value::<NodeInfo>(json).unwrap_err();
    assert!(err.to_string().contains("power"), "{err}");
}
