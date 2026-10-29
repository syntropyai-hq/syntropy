// SPDX-License-Identifier: MIT OR Apache-2.0

//! `ReplanReason`, `DriftTerm` and `WorkloadTerm` round-trip through serde (DIST-E14-T21;
//! RFC-0033 §6.1, note of 2026-10-23; RFC-0010 §6.2, §7.1).
//!
//! Each strategy draws every variant with random fields. Each value round-trips through postcard,
//! the IPC's and the event log's encoding, which tags a variant by its index, and through JSON,
//! which tags it by name. The unit tests in `replan.rs` check by exhaustive matches that every
//! variant has a sample, and pin the postcard indices.

use std::fmt::Debug;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use smallvec::SmallVec;
use syntropy_core::policy::KeyPath;
use syntropy_core::resource::*;
use syntropy_core::*;

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let bytes = postcard::to_allocvec(value).expect("serialize");
    let back: T = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(&back, value, "postcard round trip through {bytes:?}");
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&back, value, "JSON round trip through {json}");
}

// ---- Strategies for the field types, as `resource_serde.rs` writes them ---------------------

fn node_id() -> impl Strategy<Value = NodeId> {
    any::<[u8; 32]>().prop_map(NodeId)
}

fn device_id() -> impl Strategy<Value = DeviceId> {
    (node_id(), any::<u16>()).prop_map(|(node, ordinal)| DeviceId { node, ordinal })
}

fn tier_id() -> impl Strategy<Value = TierId> {
    (node_id(), any::<u16>()).prop_map(|(node, ordinal)| TierId { node, ordinal })
}

fn rdma_kind() -> impl Strategy<Value = RdmaKind> {
    prop_oneof![
        Just(RdmaKind::RoceV2),
        Just(RdmaKind::InfiniBand),
        Just(RdmaKind::Thunderbolt5)
    ]
}

fn net_medium() -> impl Strategy<Value = NetMedium> {
    prop_oneof![
        Just(NetMedium::Ethernet),
        Just(NetMedium::WiFi),
        Just(NetMedium::ThunderboltIp),
        rdma_kind().prop_map(NetMedium::Rdma),
        Just(NetMedium::Overlay),
        Just(NetMedium::Loopback),
    ]
}

fn link_kind() -> impl Strategy<Value = LinkKind> {
    prop_oneof![
        Just(LinkKind::MemoryBus),
        Just(LinkKind::Pcie),
        Just(LinkKind::Storage),
        net_medium().prop_map(LinkKind::Network),
    ]
}

fn endpoint() -> impl Strategy<Value = Endpoint> {
    prop_oneof![
        device_id().prop_map(Endpoint::Device),
        tier_id().prop_map(Endpoint::Tier),
        (node_id(), any::<u16>()).prop_map(|(node, ordinal)| Endpoint::Nic { node, ordinal }),
    ]
}

fn link_id() -> impl Strategy<Value = LinkId> {
    (endpoint(), endpoint(), link_kind(), any::<u8>()).prop_map(|(src, dst, kind, path)| LinkId {
        src,
        dst,
        kind,
        path,
    })
}

fn transport_kind() -> impl Strategy<Value = TransportKind> {
    prop_oneof![
        Just(TransportKind::SharedMemory),
        rdma_kind().prop_map(TransportKind::Rdma),
        Just(TransportKind::Quic),
        Just(TransportKind::QuicRelay),
        Just(TransportKind::Tcp),
    ]
}

fn attn_kind() -> impl Strategy<Value = AttnKind> {
    prop_oneof![
        Just(AttnKind::Gqa),
        Just(AttnKind::Mla),
        any::<bool>().prop_map(|sinks| AttnKind::SlidingGqa { sinks }),
        Just(AttnKind::BlockSparse),
        Just(AttnKind::TopKSparse),
        Just(AttnKind::Compressed),
        Just(AttnKind::Bidirectional),
    ]
}

fn kernel_class() -> impl Strategy<Value = KernelClass> {
    prop_oneof![
        Just(KernelClass::DenseGemv),
        Just(KernelClass::ExpertGemv),
        attn_kind().prop_map(KernelClass::Attention),
        Just(KernelClass::RowGather),
        Just(KernelClass::Copy),
    ]
}

fn dtype() -> impl Strategy<Value = DType> {
    prop::sample::select(DType::ALL.to_vec())
}

fn scale_granularity() -> impl Strategy<Value = ScaleGranularity> {
    prop_oneof![
        Just(ScaleGranularity::Tensor),
        Just(ScaleGranularity::Channel),
        (any::<u16>(), any::<u16>())
            .prop_map(|(rows, cols)| ScaleGranularity::Block { rows, cols }),
        any::<u16>().prop_map(|k| ScaleGranularity::Group { k }),
    ]
}

fn quant_scheme() -> impl Strategy<Value = QuantScheme> {
    prop_oneof![
        dtype().prop_map(QuantScheme::Plain),
        prop::sample::select(GgmlType::ALL.to_vec()).prop_map(QuantScheme::Ggml),
        any::<u16>().prop_map(|block| QuantScheme::Mxfp4 { block }),
        (any::<u16>(), dtype()).prop_map(|(block, scale)| QuantScheme::Nvfp4 { block, scale }),
        (dtype(), scale_granularity())
            .prop_map(|(format, scale)| QuantScheme::Fp8 { format, scale }),
        (any::<u8>(), any::<u16>(), any::<bool>(), any::<bool>()).prop_map(
            |(bits, group, symmetric, zero_point)| {
                QuantScheme::CompressedTensors(CtScheme::PackedInt {
                    bits,
                    group,
                    symmetric,
                    zero_point,
                })
            }
        ),
        scale_granularity()
            .prop_map(|strategy| QuantScheme::CompressedTensors(CtScheme::Fp8 { strategy })),
    ]
}

/// A dotted configuration key, with the odd characters a hostile or future key could carry.
fn key_path() -> impl Strategy<Value = KeyPath> {
    "[a-z_]{1,12}(\\.[a-z0-9_\"\\\\ é-]{1,12}){0,3}".prop_map(KeyPath)
}

// ---- Strategies for the three types: every variant ------------------------------------------

fn drift_term() -> impl Strategy<Value = DriftTerm> {
    prop_oneof![
        (device_id(), tier_id(), kernel_class(), quant_scheme()).prop_map(
            |(device, tier, class, scheme)| DriftTerm::Bandwidth {
                device,
                tier,
                class,
                scheme,
            }
        ),
        device_id().prop_map(|device| DriftTerm::FixedTime { device }),
        tier_id().prop_map(|tier| DriftTerm::Storage { tier }),
        device_id().prop_map(|device| DriftTerm::HostLink { device }),
        (link_id(), transport_kind())
            .prop_map(|(link, transport)| DriftTerm::Hop { link, transport }),
    ]
}

fn workload_term() -> impl Strategy<Value = WorkloadTerm> {
    prop_oneof![
        Just(WorkloadTerm::ExpertMiss),
        tier_id().prop_map(|tier| WorkloadTerm::ExpertTierShortfall { tier }),
        Just(WorkloadTerm::SpeculativeAcceptance),
    ]
}

fn replan_reason() -> impl Strategy<Value = ReplanReason> {
    prop_oneof![
        (
            prop::collection::vec(any::<u16>().prop_map(StageIx), 0..5),
            prop::collection::vec(node_id(), 0..5),
        )
            .prop_map(|(stages, nodes)| ReplanReason::StageLost {
                stages: SmallVec::from_vec(stages),
                nodes: SmallVec::from_vec(nodes),
            }),
        (node_id(), prop::option::of(device_id()))
            .prop_map(|(node, device)| ReplanReason::RunnerFailed { node, device }),
        (node_id(), node_id()).prop_map(|(src, dst)| ReplanReason::EdgeLost { src, dst }),
        node_id().prop_map(|node| ReplanReason::NodeJoined { node }),
        (node_id(), drift_term()).prop_map(|(node, term)| ReplanReason::Drift { node, term }),
        node_id().prop_map(|node| ReplanReason::MemoryPressure { node }),
        workload_term().prop_map(|term| ReplanReason::WorkloadDrift { term }),
        prop::collection::vec(key_path(), 0..4).prop_map(|keys| ReplanReason::Config { keys }),
        Just(ReplanReason::User),
        Just(ReplanReason::HealOrphans),
        Just(ReplanReason::DegradedExit),
    ]
}

proptest! {
    #[test]
    fn replan_reason_round_trips(value in replan_reason()) {
        round_trip(&value);
    }

    #[test]
    fn drift_term_round_trips(value in drift_term()) {
        round_trip(&value);
    }

    #[test]
    fn workload_term_round_trips(value in workload_term()) {
        round_trip(&value);
    }
}

/// A demotion's reason names the planned transport, the figure the plan assumed and lost
/// (RFC-0033 §6.1, note of 2026-10-23), under the variant names.
#[test]
fn a_demotion_is_hop_drift_naming_the_planned_transport() {
    let device = DeviceId {
        node: NodeId([1; 32]),
        ordinal: 0,
    };
    let link = LinkId {
        src: Endpoint::Device(device),
        dst: Endpoint::Device(DeviceId {
            node: NodeId([2; 32]),
            ordinal: 0,
        }),
        kind: LinkKind::Network(NetMedium::Rdma(RdmaKind::RoceV2)),
        path: 0,
    };
    let reason = ReplanReason::Drift {
        node: NodeId([1; 32]),
        term: DriftTerm::Hop {
            link,
            transport: TransportKind::Rdma(RdmaKind::RoceV2),
        },
    };
    let json = serde_json::to_value(&reason).unwrap();
    assert_eq!(
        json["Drift"]["term"]["Hop"]["transport"],
        serde_json::json!({ "Rdma": "RoceV2" })
    );
    round_trip(&reason);
}
