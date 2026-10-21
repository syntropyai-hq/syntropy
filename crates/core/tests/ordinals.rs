// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ordinal rules of RFC-0010 §1.4 on fixture inventories (PLC-E1-T01).
//!
//! The fixtures are made-up machines shaped like the platform classes RFC-0010 §2.2
//! describes, not records of real hosts.

use std::path::PathBuf;

use proptest::prelude::*;
use syntropy_core::resource::ordinal::{
    DeviceCandidate, Enumerated, HostDomain, TierSlot, VolumeCandidate, assign,
};
use syntropy_core::resource::{DeviceKind, PciAddress};
use syntropy_core::{DeviceId, NodeId, TierId};

const NODE: NodeId = NodeId([0x5a; 32]);

fn pci(bus: u8) -> Option<PciAddress> {
    Some(PciAddress {
        domain: 0,
        bus,
        device: 0,
        function: 0,
    })
}

fn device(
    kind: DeviceKind,
    pci: Option<PciAddress>,
    numa: Option<u16>,
    name: &str,
) -> DeviceCandidate {
    DeviceCandidate {
        kind,
        pci,
        numa,
        uuid: None,
        name: name.to_owned(),
        unified: false,
    }
}

fn domain(numa: u16) -> HostDomain {
    HostDomain { numa, pinned: true }
}

fn volume(mount: &str, holds_store: bool) -> VolumeCandidate {
    VolumeCandidate {
        mount: PathBuf::from(mount),
        holds_store,
    }
}

/// A desktop: two discrete GPUs, one NUMA domain, two volumes.
fn desktop() -> Enumerated {
    Enumerated {
        devices: vec![
            device(DeviceKind::DiscreteGpu, pci(0x41), Some(0), "gpu-b"),
            device(DeviceKind::Cpu, None, Some(0), "cpu"),
            device(DeviceKind::DiscreteGpu, pci(0x01), Some(0), "gpu-a"),
        ],
        domains: vec![domain(0)],
        volumes: vec![volume("/mnt/extra", false), volume("/", true)],
    }
}

/// A dual-socket server: four discrete GPUs over two domains, two CPU devices, an NPU.
fn dual_socket() -> Enumerated {
    Enumerated {
        devices: vec![
            device(DeviceKind::Npu, pci(0x90), Some(1), "npu"),
            device(DeviceKind::DiscreteGpu, pci(0x81), Some(1), "gpu-2"),
            device(DeviceKind::Cpu, None, Some(1), "cpu-1"),
            device(DeviceKind::DiscreteGpu, pci(0x02), Some(0), "gpu-1"),
            device(DeviceKind::DiscreteGpu, pci(0xc1), Some(1), "gpu-3"),
            device(DeviceKind::Cpu, None, Some(0), "cpu-0"),
            device(DeviceKind::DiscreteGpu, pci(0x01), Some(0), "gpu-0"),
        ],
        domains: vec![domain(1), domain(0)],
        volumes: vec![
            volume("/data", true),
            volume("/scratch", false),
            volume("/archive", false),
        ],
    }
}

/// A unified-memory laptop: an integrated GPU sharing RAM, no pinned tier.
fn unified_laptop() -> Enumerated {
    Enumerated {
        devices: vec![
            device(DeviceKind::Cpu, None, Some(0), "cpu"),
            DeviceCandidate {
                unified: true,
                ..device(DeviceKind::IntegratedGpu, None, None, "igpu")
            },
        ],
        domains: vec![HostDomain {
            numa: 0,
            pinned: false,
        }],
        volumes: vec![volume("/", true)],
    }
}

/// A mixed box: a discrete GPU, a unified-memory integrated GPU, two identical GPUs that
/// differ only by UUID, a domain and a volume each listed twice.
fn mixed() -> Enumerated {
    let twin = |b: u8| DeviceCandidate {
        uuid: Some([b; 16]),
        ..device(DeviceKind::DiscreteGpu, None, None, "twin")
    };
    Enumerated {
        devices: vec![
            twin(2),
            device(DeviceKind::Cpu, None, Some(0), "cpu"),
            DeviceCandidate {
                unified: true,
                ..device(DeviceKind::IntegratedGpu, pci(0x05), Some(0), "igpu")
            },
            twin(1),
            device(DeviceKind::DiscreteGpu, pci(0x03), Some(0), "dgpu"),
        ],
        domains: vec![
            HostDomain {
                numa: 0,
                pinned: false,
            },
            domain(0),
        ],
        volumes: vec![volume("/", true), volume("/", true), volume("/fast", false)],
    }
}

fn fixtures() -> Vec<Enumerated> {
    vec![desktop(), dual_socket(), unified_laptop(), mixed()]
}

fn did(ordinal: u16) -> DeviceId {
    DeviceId {
        node: NODE,
        ordinal,
    }
}

fn tid(ordinal: u16) -> TierId {
    TierId {
        node: NODE,
        ordinal,
    }
}

#[test]
fn desktop_ordinals_follow_rfc_0010() {
    let o = assign(NODE, &desktop()).unwrap();
    let names: Vec<_> = o
        .devices
        .iter()
        .map(|(id, d)| (id.ordinal, d.name.as_str()))
        .collect();
    assert_eq!(names, [(0, "gpu-a"), (1, "gpu-b"), (2, "cpu")]);
    let tiers: Vec<_> = o
        .tiers
        .iter()
        .map(|(id, s)| (id.ordinal, s.clone()))
        .collect();
    assert_eq!(
        tiers,
        [
            (0, TierSlot::Vram { device: did(0) }),
            (1, TierSlot::Vram { device: did(1) }),
            (2, TierSlot::HostPinned { numa: 0 }),
            (3, TierSlot::HostPageable { numa: 0 }),
            (4, TierSlot::Nvme { mount: "/".into() }),
            (
                5,
                TierSlot::Nvme {
                    mount: "/mnt/extra".into()
                }
            ),
        ]
    );
}

#[test]
fn dual_socket_ordinals_follow_rfc_0010() {
    let o = assign(NODE, &dual_socket()).unwrap();
    let names: Vec<_> = o.devices.iter().map(|(_, d)| d.name.as_str()).collect();
    assert_eq!(
        names,
        ["gpu-0", "gpu-1", "gpu-2", "gpu-3", "cpu-0", "cpu-1", "npu"]
    );
    let slots: Vec<_> = o.tiers.iter().map(|(_, s)| s.clone()).collect();
    assert_eq!(
        slots,
        [
            TierSlot::Vram { device: did(0) },
            TierSlot::Vram { device: did(1) },
            TierSlot::Vram { device: did(2) },
            TierSlot::Vram { device: did(3) },
            TierSlot::HostPinned { numa: 0 },
            TierSlot::HostPageable { numa: 0 },
            TierSlot::HostPinned { numa: 1 },
            TierSlot::HostPageable { numa: 1 },
            TierSlot::Nvme {
                mount: "/data".into()
            },
            TierSlot::Nvme {
                mount: "/archive".into()
            },
            TierSlot::Nvme {
                mount: "/scratch".into()
            },
        ]
    );
    assert!(
        o.tiers
            .iter()
            .enumerate()
            .all(|(i, (id, _))| *id == tid(i as u16))
    );
}

#[test]
fn unified_laptop_has_one_unified_tier_and_no_pinned_tier() {
    let o = assign(NODE, &unified_laptop()).unwrap();
    let names: Vec<_> = o.devices.iter().map(|(_, d)| d.name.as_str()).collect();
    assert_eq!(names, ["igpu", "cpu"]);
    let slots: Vec<_> = o.tiers.iter().map(|(_, s)| s.clone()).collect();
    assert_eq!(
        slots,
        [
            TierSlot::Unified { device: did(0) },
            TierSlot::HostPageable { numa: 0 },
            TierSlot::Nvme { mount: "/".into() },
        ]
    );
}

#[test]
fn duplicates_count_once_and_twins_order_by_uuid() {
    let o = assign(NODE, &mixed()).unwrap();
    let devices: Vec<_> = o
        .devices
        .iter()
        .map(|(_, d)| (d.name.as_str(), d.uuid))
        .collect();
    assert_eq!(
        devices,
        [
            ("dgpu", None),
            ("twin", Some([1; 16])),
            ("twin", Some([2; 16])),
            ("igpu", None),
            ("cpu", None),
        ]
    );
    let slots: Vec<_> = o.tiers.iter().map(|(_, s)| s.clone()).collect();
    assert_eq!(
        slots,
        [
            TierSlot::Vram { device: did(0) },
            TierSlot::Vram { device: did(1) },
            TierSlot::Vram { device: did(2) },
            TierSlot::Unified { device: did(3) },
            TierSlot::HostPinned { numa: 0 },
            TierSlot::HostPageable { numa: 0 },
            TierSlot::Nvme { mount: "/".into() },
            TierSlot::Nvme {
                mount: "/fast".into()
            },
        ]
    );
}

/// One fixture with each of its lists shuffled.
fn shuffled_fixture() -> impl Strategy<Value = (usize, Enumerated)> {
    (0..fixtures().len()).prop_flat_map(|i| {
        let f = fixtures().swap_remove(i);
        (
            Just(i),
            Just(f.devices).prop_shuffle(),
            Just(f.domains).prop_shuffle(),
            Just(f.volumes).prop_shuffle(),
        )
            .prop_map(|(i, devices, domains, volumes)| {
                (
                    i,
                    Enumerated {
                        devices,
                        domains,
                        volumes,
                    },
                )
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Acceptance criterion 3: 1,000 shuffled fixture inventories give identical DeviceId
    /// and TierId assignments.
    #[test]
    fn ordinals_ignore_enumeration_order((i, shuffled) in shuffled_fixture()) {
        let expected = assign(NODE, &fixtures()[i]).unwrap();
        let got = assign(NODE, &shuffled).unwrap();
        prop_assert_eq!(got, expected);
    }
}
