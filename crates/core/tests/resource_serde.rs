// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serde properties of the resource graph records (PLC-E1-T01, RFC-0010 §1.1–§1.4).
//!
//! Every type round-trips through a serde format unchanged, and every record decodes when
//! it carries fields this build does not know. JSON stands in for any self-describing
//! format; the property is serde's, not the format's.

use std::fmt::Debug;
use std::path::PathBuf;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use smallvec::SmallVec;
use syntropy_core::resource::ordinal::{
    DeviceCandidate, Enumerated, HostDomain, Ordinals, TierSlot, VolumeCandidate,
};
use syntropy_core::resource::*;
use syntropy_core::*;

// ---- Strategies -------------------------------------------------------------------------

fn text() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop::sample::select(vec!['a', 'Z', '0', '/', ' ', 'é', '"']),
        0..8,
    )
    .prop_map(|c| c.into_iter().collect())
}

fn path() -> impl Strategy<Value = PathBuf> {
    text().prop_map(PathBuf::from)
}

fn f64s() -> impl Strategy<Value = f64> + Clone {
    -1.0e12f64..1.0e12
}

fn f32s() -> impl Strategy<Value = f32> + Clone {
    -1.0e6f32..1.0e6
}

fn node_id() -> impl Strategy<Value = NodeId> {
    any::<[u8; 32]>().prop_map(NodeId)
}

fn blake3() -> impl Strategy<Value = Blake3> {
    any::<[u8; 32]>().prop_map(Blake3)
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

fn bytes() -> impl Strategy<Value = Bytes> {
    any::<u64>().prop_map(Bytes)
}

fn nanos() -> impl Strategy<Value = Nanos> + Clone {
    any::<u64>().prop_map(Nanos)
}

fn bps() -> impl Strategy<Value = BytesPerSec> + Clone {
    f64s().prop_map(BytesPerSec)
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

fn measure_source() -> impl Strategy<Value = MeasureSource> {
    prop_oneof![
        Just(MeasureSource::Startup),
        Just(MeasureSource::Online),
        Just(MeasureSource::Prior),
        quant_scheme().prop_map(|scheme| MeasureSource::Derived {
            from: DerivedFrom { scheme }
        }),
    ]
}

fn measured<T: Debug + Clone + 'static>(
    value: impl Strategy<Value = T> + Clone,
) -> impl Strategy<Value = Measured<T>> {
    (
        value.clone(),
        value,
        any::<u32>(),
        any::<u64>(),
        measure_source(),
    )
        .prop_map(|(p50, p99, samples, at_unix_ms, source)| Measured {
            p50,
            p99,
            samples,
            at_unix_ms,
            source,
        })
}

fn m_nanos() -> impl Strategy<Value = Measured<Nanos>> {
    measured(nanos())
}

fn m_bps() -> impl Strategy<Value = Measured<BytesPerSec>> {
    measured(bps())
}

fn small<A: smallvec::Array + Debug + 'static>(
    s: impl Strategy<Value = A::Item>,
) -> impl Strategy<Value = SmallVec<A>>
where
    A::Item: Debug,
{
    // Up to two past the inline capacity, so both the inline and the spilled forms occur.
    prop::collection::vec(s, 0..A::size() + 3).prop_map(SmallVec::from_vec)
}

fn ggml_backend() -> impl Strategy<Value = GgmlBackend> {
    prop_oneof![
        Just(GgmlBackend::Cpu),
        Just(GgmlBackend::Cuda),
        Just(GgmlBackend::Hip),
        Just(GgmlBackend::Metal),
        Just(GgmlBackend::Vulkan),
        Just(GgmlBackend::Sycl),
    ]
}

fn backend() -> impl Strategy<Value = Backend> {
    prop_oneof![
        Just(Backend::Cuda),
        Just(Backend::Hip),
        Just(Backend::Metal),
        Just(Backend::Vulkan),
        Just(Backend::Cpu),
        ggml_backend().prop_map(Backend::Ggml),
    ]
}

fn device_kind() -> impl Strategy<Value = DeviceKind> {
    prop_oneof![
        Just(DeviceKind::DiscreteGpu),
        Just(DeviceKind::IntegratedGpu),
        Just(DeviceKind::Cpu),
        Just(DeviceKind::Npu),
    ]
}

fn device_features() -> impl Strategy<Value = DeviceFeatures> {
    any::<u64>().prop_map(DeviceFeatures::from_bits_truncate)
}

fn apple_family() -> impl Strategy<Value = AppleGpuFamily> {
    prop_oneof![
        Just(AppleGpuFamily::Apple7),
        Just(AppleGpuFamily::Apple8),
        Just(AppleGpuFamily::Apple9),
        Just(AppleGpuFamily::Apple10),
    ]
}

fn cpu_arch() -> impl Strategy<Value = CpuArch> {
    prop_oneof![Just(CpuArch::X86_64), Just(CpuArch::Aarch64)]
}

fn cpu_isa() -> impl Strategy<Value = CpuIsa> {
    prop_oneof![
        Just(CpuIsa::X86V2),
        Just(CpuIsa::X86V3),
        Just(CpuIsa::X86V4),
        Just(CpuIsa::X86V4Amx),
        Just(CpuIsa::Arm80),
        Just(CpuIsa::Arm82Dot),
        Just(CpuIsa::Arm86I8mm),
        Just(CpuIsa::Arm92Sme2),
    ]
}

fn api_ids() -> impl Strategy<Value = ApiIds> {
    (any::<u32>(), any::<u32>()).prop_map(|(vendor, device)| ApiIds { vendor, device })
}

fn device_class() -> impl Strategy<Value = DeviceClass> {
    prop_oneof![
        any::<u16>().prop_map(|sm| DeviceClass::Cuda { sm }),
        any::<u32>().prop_map(|gfx| DeviceClass::Hip { gfx }),
        (apple_family(), any::<bool>())
            .prop_map(|(family, metal4)| DeviceClass::Metal { family, metal4 }),
        (any::<u32>(), any::<bool>(), any::<u8>()).prop_map(|(vendor, coopmat, subgroup)| {
            DeviceClass::Vulkan {
                vendor,
                coopmat,
                subgroup,
            }
        }),
        cpu_isa().prop_map(|isa| DeviceClass::Cpu { isa }),
        (any::<u32>(), any::<u32>()).prop_map(|(vendor, model)| DeviceClass::Npu { vendor, model }),
    ]
}

fn driver_info() -> impl Strategy<Value = DriverInfo> {
    (
        text(),
        prop::option::of(text()),
        prop::option::of(path()),
        prop::option::of(any::<u32>().prop_map(VkDriverId)),
    )
        .prop_map(|(driver, runtime, loader, vk_driver)| DriverInfo {
            driver,
            runtime,
            loader,
            vk_driver,
        })
}

fn pci_address() -> impl Strategy<Value = PciAddress> {
    (any::<u32>(), any::<u8>(), any::<u8>(), any::<u8>()).prop_map(
        |(domain, bus, device, function)| PciAddress {
            domain,
            bus,
            device,
            function,
        },
    )
}

fn pcie_link() -> impl Strategy<Value = PcieLink> {
    (any::<u8>(), any::<u8>()).prop_map(|(g, width)| PcieLink { r#gen: g, width })
}

fn pcie_upstream() -> impl Strategy<Value = PcieUpstream> {
    prop_oneof![
        Just(PcieUpstream::CpuRootPort),
        Just(PcieUpstream::Chipset),
        Just(PcieUpstream::Switch),
        Just(PcieUpstream::ThunderboltTunnel),
        Just(PcieUpstream::Unknown),
    ]
}

fn pci_info() -> impl Strategy<Value = PciInfo> {
    (
        pci_address(),
        any::<u16>(),
        any::<u16>(),
        any::<u32>(),
        pcie_link(),
        pcie_link(),
        pcie_upstream(),
        any::<Option<bool>>(),
        any::<Option<u16>>(),
    )
        .prop_map(
            |(
                bdf,
                vendor,
                device,
                subsystem,
                link_cur,
                link_max,
                upstream,
                resizable_bar,
                numa,
            )| {
                PciInfo {
                    bdf,
                    vendor,
                    device,
                    subsystem,
                    link_cur,
                    link_max,
                    upstream,
                    resizable_bar,
                    numa,
                }
            },
        )
}

fn isa_flags() -> impl Strategy<Value = IsaFlags> {
    any::<u32>().prop_map(IsaFlags::from_bits_truncate)
}

fn cpu_set() -> impl Strategy<Value = CpuSet> {
    prop::collection::vec(any::<u64>(), 0..3).prop_map(CpuSet)
}

fn cpu_topology() -> impl Strategy<Value = CpuTopology> {
    (
        (
            any::<u16>(),
            any::<u16>(),
            any::<u16>(),
            any::<u16>(),
            any::<u16>(),
        ),
        (bytes(), any::<u8>(), isa_flags(), any::<bool>(), cpu_set()),
    )
        .prop_map(
            |(
                (numa, physical_cores, logical_cpus, perf_cores, efficiency_cores),
                (llc, ccds, isa, amx_permitted, reserved_cpus),
            )| CpuTopology {
                numa,
                physical_cores,
                logical_cpus,
                perf_cores,
                efficiency_cores,
                llc,
                ccds,
                isa,
                amx_permitted,
                reserved_cpus,
            },
        )
}

fn thermal_state() -> impl Strategy<Value = ThermalState> {
    prop_oneof![
        Just(ThermalState::Unknown),
        Just(ThermalState::Nominal),
        Just(ThermalState::Throttled),
        Just(ThermalState::Critical),
    ]
}

fn power_info() -> impl Strategy<Value = PowerInfo> {
    (
        prop::option::of(f32s()),
        prop::option::of(f32s()),
        thermal_state(),
    )
        .prop_map(|(limit_w, cap_w, thermal)| PowerInfo {
            limit_w,
            cap_w,
            thermal,
        })
}

fn error_class() -> impl Strategy<Value = ErrorClass> {
    prop::sample::select(vec![
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
    ])
}

fn probe_step() -> impl Strategy<Value = ProbeStep> {
    use ProbeStep::*;
    prop::sample::select(vec![
        E1, E2, E3, E4, E5, D1, D2, M1, M2, M3, M4, M5, M6, H1, H2, H3, H4, H5, S1, S2, L1, L2,
    ])
}

fn spec_quantity() -> impl Strategy<Value = SpecQuantity> {
    prop::sample::select(vec![
        SpecQuantity::MemoryBandwidth,
        SpecQuantity::Compute,
        SpecQuantity::PcieH2d,
        SpecQuantity::PcieD2h,
        SpecQuantity::StorageRead,
        SpecQuantity::LinkBandwidth,
    ])
}

fn exclusion_source() -> impl Strategy<Value = ExclusionSource> {
    prop_oneof![
        Just(ExclusionSource::UserConfig),
        Just(ExclusionSource::FeaturesDisable)
    ]
}

fn device_problem() -> impl Strategy<Value = DeviceProblem> {
    prop_oneof![
        (text(), text()).prop_map(|(have, need)| DeviceProblem::DriverTooOld { have, need }),
        Just(DeviceProblem::NoProvider),
        (probe_step(), error_class())
            .prop_map(|(suite, class)| DeviceProblem::ProbeCrashed { suite, class }),
        (pcie_link(), pcie_link()).prop_map(|(cur, max)| DeviceProblem::LinkDegraded { cur, max }),
        (spec_quantity(), f64s(), f64s()).prop_map(|(quantity, measured, spec)| {
            DeviceProblem::BelowSpec {
                quantity,
                measured,
                spec,
            }
        }),
        exclusion_source().prop_map(|by| DeviceProblem::Excluded { by }),
        api_ids().prop_map(|api| DeviceProblem::ApiIdsMismatch { api }),
    ]
}

fn device_info_ext() -> impl Strategy<Value = DeviceInfoExt> {
    (
        driver_info(),
        prop::option::of(pci_info()),
        any::<Option<[u8; 16]>>(),
        any::<Option<u64>>(),
        prop::option::of(cpu_topology()),
        prop::option::of(power_info()),
        small(device_problem()),
        any::<Option<u16>>(),
        prop::option::of(api_ids()),
    )
        .prop_map(
            |(driver, pci, uuid, luid, cpu, power, problems, gpu_cores, api_ids)| DeviceInfoExt {
                driver,
                pci,
                uuid,
                luid,
                cpu,
                power,
                problems,
                gpu_cores,
                api_ids,
            },
        )
}

fn device_info() -> impl Strategy<Value = DeviceInfo> {
    (
        (
            device_id(),
            device_kind(),
            text(),
            device_class(),
            small(backend()),
        ),
        (
            device_features(),
            small(tier_id()),
            any::<Option<u16>>(),
            blake3(),
            device_info_ext(),
        ),
    )
        .prop_map(
            |(
                (id, kind, name, class, backends),
                (features, local_tiers, numa, fingerprint, ext),
            )| {
                DeviceInfo {
                    id,
                    kind,
                    name,
                    class,
                    backends,
                    features,
                    local_tiers,
                    numa,
                    fingerprint,
                    ext,
                }
            },
        )
}

fn tier_kind() -> impl Strategy<Value = TierKind> {
    prop_oneof![
        Just(TierKind::Vram),
        Just(TierKind::Unified),
        Just(TierKind::HostPinned),
        Just(TierKind::HostPageable),
        Just(TierKind::Nvme),
    ]
}

fn budget_source() -> impl Strategy<Value = BudgetSource> {
    prop_oneof![
        Just(BudgetSource::AllocationProbe),
        Just(BudgetSource::DxgiBudget),
        Just(BudgetSource::VkMemoryBudget),
        Just(BudgetSource::MetalWorkingSet),
        bytes().prop_map(|reserve| BudgetSource::MemAvailableMinus { reserve }),
        Just(BudgetSource::HostRules),
        Just(BudgetSource::Statfs),
    ]
}

fn lock_accounting() -> impl Strategy<Value = LockAccounting> {
    (
        bytes(),
        bytes(),
        any::<Option<bool>>(),
        any::<Option<bool>>(),
        prop::option::of(bytes()),
        text(),
    )
        .prop_map(
            |(
                vmlck_before,
                vmpin_before,
                registration_charged,
                io_fixed_charged,
                lockable_measured,
                driver,
            )| LockAccounting {
                vmlck_before,
                vmpin_before,
                registration_charged,
                io_fixed_charged,
                lockable_measured,
                driver,
            },
        )
}

fn host_mem_info() -> impl Strategy<Value = HostMemInfo> {
    let ob = || prop::option::of(bytes());
    (
        (bytes(), bytes(), ob(), ob(), ob()),
        (
            ob(),
            any::<bool>(),
            ob(),
            ob(),
            prop::option::of(lock_accounting()),
        ),
    )
        .prop_map(
            |(
                (total, available, numa_free, cgroup_limit, memlock_limit),
                (windows_lockable, wsl2, macos_wired_limit, large_pages, lock),
            )| HostMemInfo {
                total,
                available,
                numa_free,
                cgroup_limit,
                memlock_limit,
                windows_lockable,
                wsl2,
                macos_wired_limit,
                large_pages,
                lock,
            },
        )
}

fn tier_info_ext() -> impl Strategy<Value = TierInfoExt> {
    (
        any::<Option<u16>>(),
        budget_source(),
        prop::option::of(host_mem_info()),
        prop::option::of(bytes()),
    )
        .prop_map(|(numa, budget_source, host, context_reserve)| TierInfoExt {
            numa,
            budget_source,
            host,
            context_reserve,
        })
}

fn fs_kind() -> impl Strategy<Value = FsKind> {
    use FsKind::*;
    prop::sample::select(vec![Ext4, Xfs, Btrfs, Zfs, Ntfs, Refs, Apfs, Other])
}

fn storage_bus() -> impl Strategy<Value = StorageBus> {
    use StorageBus::*;
    prop::sample::select(vec![Nvme, Sata, Usb, Virtual, Unknown])
}

fn direct_io() -> impl Strategy<Value = DirectIoSupport> {
    prop_oneof![
        (any::<u32>(), any::<u32>()).prop_map(|(mem_align, offset_align)| {
            DirectIoSupport::Supported {
                mem_align,
                offset_align,
            }
        }),
        Just(DirectIoSupport::Unsupported),
    ]
}

fn temp_source() -> impl Strategy<Value = TempSource> {
    prop_oneof![
        Just(TempSource::Hwmon),
        Just(TempSource::StorageQuery),
        Just(TempSource::None)
    ]
}

fn volume_info() -> impl Strategy<Value = VolumeInfo> {
    (
        (
            path(),
            fs_kind(),
            storage_bus(),
            prop::option::of(pci_info()),
            any::<bool>(),
        ),
        (
            direct_io(),
            any::<u32>(),
            any::<u32>(),
            any::<Option<bool>>(),
            bytes(),
        ),
        (temp_source(), text(), text()),
    )
        .prop_map(
            |(
                (mount, fs, bus, pci, shared_lane),
                (direct_io, logical_block, physical_block, encrypted, free),
                (temperature_source, model, firmware),
            )| VolumeInfo {
                mount,
                fs,
                bus,
                pci,
                shared_lane,
                direct_io,
                logical_block,
                physical_block,
                encrypted,
                free,
                temperature_source,
                model,
                firmware,
            },
        )
}

fn tier_info() -> impl Strategy<Value = TierInfo> {
    (
        (tier_id(), tier_kind(), bytes(), bytes()),
        (
            small(device_id()),
            prop::option::of(bytes()),
            prop::option::of(volume_info()),
            tier_info_ext(),
        ),
    )
        .prop_map(
            |((id, kind, capacity, usable), (affinity, pin_limit, volume, ext))| TierInfo {
                id,
                kind,
                capacity,
                usable,
                affinity,
                pin_limit,
                volume,
                ext,
            },
        )
}

fn host_pool_info() -> impl Strategy<Value = HostPoolInfo> {
    (any::<u16>(), tier_id(), tier_id(), bytes(), bytes()).prop_map(
        |(numa, pinned, pageable, usable, pin_limit)| HostPoolInfo {
            numa,
            tiers: [pinned, pageable],
            usable,
            pin_limit,
        },
    )
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

fn ip_net() -> impl Strategy<Value = IpNet> {
    prop_oneof![
        (any::<[u8; 4]>(), 0..=32u8)
            .prop_map(|(a, p)| IpNet::new(std::net::IpAddr::from(a), p).expect("valid")),
        (
            any::<[u8; 16]>(),
            0..=128u8,
            proptest::option::of(any::<u32>())
        )
            .prop_map(|(a, p, scope)| {
                let net = IpNet::new(std::net::IpAddr::from(a), p).expect("valid");
                match scope {
                    Some(scope) => net.with_scope(scope),
                    None => net,
                }
            }),
    ]
}

fn iface_flags() -> impl Strategy<Value = IfaceFlags> {
    any::<u32>().prop_map(IfaceFlags::from_bits_truncate)
}

fn rdma_link_layer() -> impl Strategy<Value = RdmaLinkLayer> {
    prop_oneof![
        Just(RdmaLinkLayer::Ethernet),
        Just(RdmaLinkLayer::InfiniBand),
        Just(RdmaLinkLayer::Thunderbolt),
    ]
}

fn rdma_port_info() -> impl Strategy<Value = RdmaPortInfo> {
    (
        text(),
        any::<u8>(),
        rdma_kind(),
        any::<u8>(),
        any::<bool>(),
        rdma_link_layer(),
    )
        .prop_map(
            |(device, port, kind, gid_index, active, link_layer)| RdmaPortInfo {
                device,
                port,
                kind,
                gid_index,
                active,
                link_layer,
            },
        )
}

fn iface_info() -> impl Strategy<Value = IfaceInfo> {
    (
        (any::<u16>(), text(), net_medium(), small(ip_net())),
        (
            any::<u32>(),
            proptest::option::of(any::<u32>()),
            iface_flags(),
            proptest::option::of(text()),
            proptest::option::of(rdma_port_info()),
            proptest::option::of(node_id()),
        ),
    )
        .prop_map(
            |((ordinal, name, medium, addrs), (mtu, speed_mbps, flags, bridge, rdma, tb_peer))| {
                IfaceInfo {
                    ordinal,
                    name,
                    medium,
                    addrs,
                    mtu,
                    speed_mbps,
                    flags,
                    bridge,
                    rdma,
                    tb_peer,
                }
            },
        )
}

fn link_info() -> impl Strategy<Value = LinkInfo> {
    (
        link_id(),
        small(transport_kind()),
        proptest::option::of(iface_info()),
        any::<bool>(),
    )
        .prop_map(|(id, transports, interface, metered)| LinkInfo {
            id,
            transports,
            interface,
            metered,
        })
}

fn link_caps() -> impl Strategy<Value = LinkCaps> {
    (
        link_id(),
        transport_kind(),
        m_nanos(),
        m_bps(),
        proptest::option::of(m_nanos()),
        proptest::option::of(f32s()),
    )
        .prop_map(|(link, transport, alpha, beta, alpha_eff, loss)| LinkCaps {
            link,
            transport,
            alpha,
            beta,
            alpha_eff,
            loss,
        })
}

fn demotion_cause() -> impl Strategy<Value = DemotionCause> {
    prop_oneof![
        Just(DemotionCause::ConnectFailed),
        Just(DemotionCause::RepeatedFaults),
    ]
}

fn demotion() -> impl Strategy<Value = Demotion> {
    (link_id(), transport_kind(), transport_kind()).prop_map(|(edge, from, to)| Demotion {
        edge,
        from,
        to,
    })
}

fn node_inventory() -> impl Strategy<Value = NodeInventory> {
    (
        prop::collection::vec(device_info(), 0..3),
        prop::collection::vec(tier_info(), 0..3),
        prop::collection::vec(link_info(), 0..3),
        prop::collection::vec(host_pool_info(), 0..3),
    )
        .prop_map(|(devices, tiers, links, host_pools)| NodeInventory {
            devices,
            tiers,
            links,
            host_pools,
        })
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

fn profile_status() -> impl Strategy<Value = ProfileStatus> {
    prop_oneof![
        Just(ProfileStatus::Quick),
        Just(ProfileStatus::Full),
        Just(ProfileStatus::FullWithSustained),
    ]
}

fn power_source() -> impl Strategy<Value = PowerSource> {
    prop_oneof![
        Just(PowerSource::Mains),
        any::<u8>().prop_map(|percent| PowerSource::Battery { percent }),
    ]
}

fn capability_profile_ext() -> impl Strategy<Value = CapabilityProfileExt> {
    (
        blake3(),
        profile_status(),
        power_source(),
        any::<bool>(),
        any::<u32>(),
        prop::option::of(node_inventory()),
        prop::option::of(m_bps()),
    )
        .prop_map(
            |(key, status, power, battery_present, suite, inventory, wan_down)| {
                CapabilityProfileExt {
                    key: ProfileKey(key),
                    status,
                    power,
                    battery_present,
                    suite: ProbeSuiteVersion(suite),
                    inventory,
                    wan_down,
                }
            },
        )
}

fn quant_bw() -> impl Strategy<Value = QuantBw> {
    (
        tier_id(),
        kernel_class(),
        quant_scheme(),
        small((any::<u16>(), m_bps())),
        bytes(),
        prop::option::of(m_bps()),
    )
        .prop_map(|(tier, class, scheme, curve, extent, contended)| QuantBw {
            tier,
            class,
            scheme,
            curve,
            extent,
            contended,
        })
}

fn launch_caps() -> impl Strategy<Value = LaunchCaps> {
    (m_nanos(), prop::option::of(m_nanos())).prop_map(|(eager, captured_node)| LaunchCaps {
        eager,
        captured_node,
    })
}

fn device_caps_ext() -> impl Strategy<Value = DeviceCapsExt> {
    (
        prop::collection::vec(quant_bw(), 0..3),
        prop::option::of(m_nanos()),
        prop::option::of(launch_caps()),
        prop::option::of(measured(f32s())),
        prop::option::of(m_nanos()),
        prop::option::of(m_nanos()),
    )
        .prop_map(
            |(quant_bw, t_switch, launch, dma_interference, wake, host_wait)| DeviceCapsExt {
                quant_bw,
                t_switch,
                launch,
                dma_interference,
                wake,
                host_wait,
            },
        )
}

fn host_link_caps() -> impl Strategy<Value = HostLinkCaps> {
    (
        link_id(),
        m_bps(),
        m_bps(),
        m_bps(),
        m_nanos(),
        prop::option::of(m_bps()),
        prop::collection::vec((device_id(), m_bps()), 0..3),
    )
        .prop_map(
            |(link, h2d_pinned, d2h_pinned, h2d_pageable, h2d_small, bidirectional, p2p)| {
                HostLinkCaps {
                    link,
                    h2d_pinned,
                    d2h_pinned,
                    h2d_pageable,
                    h2d_small,
                    bidirectional,
                    p2p,
                }
            },
        )
}

fn engine_kind() -> impl Strategy<Value = EngineKind> {
    use EngineKind::*;
    prop::sample::select(vec![
        UringDirect,
        PreadPool,
        IoRing,
        Iocp,
        MacNoCache,
        MacMapped,
    ])
}

fn dio_align() -> impl Strategy<Value = DioAlign> {
    (any::<u32>(), any::<u32>()).prop_map(|(mem, offset)| DioAlign { mem, offset })
}

fn direct_path_report() -> impl Strategy<Value = DirectPathReport> {
    (
        any::<bool>(),
        any::<bool>(),
        any::<Option<bool>>(),
        any::<bool>(),
    )
        .prop_map(
            |(cufile_loads, cufile_opens, cufile_p2p, directstorage)| DirectPathReport {
                cufile_loads,
                cufile_opens,
                cufile_p2p,
                directstorage,
            },
        )
}

fn io_point() -> impl Strategy<Value = IoPoint> {
    (
        engine_kind(),
        bytes(),
        any::<u16>(),
        prop::option::of(nanos()),
        m_bps(),
        m_nanos(),
    )
        .prop_map(|(engine, size, qd, cpu_per_gb, rate, latency)| IoPoint {
            engine,
            size,
            qd,
            cpu_per_gb,
            rate,
            latency,
        })
}

fn sustained_status() -> impl Strategy<Value = SustainedStatus> {
    prop_oneof![
        Just(SustainedStatus::Pending),
        any::<u64>().prop_map(|since_ms| SustainedStatus::Running { since_ms }),
        Just(SustainedStatus::Done),
    ]
}

fn throttle_info() -> impl Strategy<Value = ThrottleInfo> {
    (nanos(), f32s(), prop::option::of(f32s())).prop_map(|(onset_after, ratio, temp_at_onset_c)| {
        ThrottleInfo {
            onset_after,
            ratio,
            temp_at_onset_c,
        }
    })
}

fn block_reason() -> impl Strategy<Value = BlockReason> {
    use BlockReason::*;
    prop::sample::select(vec![Denied, SysctlDisabled, Unsupported, ConfigDisabled])
}

fn storage_caps_ext() -> impl Strategy<Value = StorageCapsExt> {
    (
        (
            engine_kind(),
            bytes(),
            dio_align(),
            any::<bool>(),
            any::<bool>(),
        ),
        (
            prop::option::of(f32s()),
            any::<bool>(),
            any::<u16>(),
            small((device_id(), direct_path_report())),
        ),
        (
            prop::collection::vec(io_point(), 0..3),
            sustained_status(),
            prop::option::of(throttle_info()),
            small((engine_kind(), block_reason())),
        ),
    )
        .prop_map(
            |(
                (engine, chunk, dio, buffered, sqpoll),
                (read_around_efficiency, shared_lanes, qd_target, direct_paths),
                (grid, sustained, throttle, engines_blocked),
            )| StorageCapsExt {
                engine,
                chunk,
                dio,
                buffered,
                sqpoll,
                read_around_efficiency,
                shared_lanes,
                qd_target,
                direct_paths,
                grid,
                sustained,
                throttle,
                engines_blocked,
            },
        )
}

fn hop_histogram() -> impl Strategy<Value = HopHistogram> {
    (
        nanos(),
        f32s(),
        prop::collection::vec(any::<u32>(), 0..6),
        any::<u64>(),
    )
        .prop_map(|(first, ratio, counts, samples)| HopHistogram {
            first,
            ratio,
            counts,
            samples,
        })
}

fn sync_kind() -> impl Strategy<Value = SyncKind> {
    prop_oneof![Just(SyncKind::Event), Just(SyncKind::Doorbell)]
}

fn device_path() -> impl Strategy<Value = DevicePath> {
    (device_id(), device_id(), bytes(), sync_kind()).prop_map(|(src, dst, payload, sync)| {
        DevicePath {
            src,
            dst,
            payload,
            sync,
        }
    })
}

fn link_caps_ext() -> impl Strategy<Value = LinkCapsExt> {
    (
        (
            small((bytes(), m_nanos())),
            prop::option::of(nanos()),
            prop::option::of(m_nanos()),
        ),
        (
            prop::option::of(hop_histogram()),
            any::<bool>(),
            prop::option::of(device_path()),
            any::<bool>(),
            net_medium(),
        ),
    )
        .prop_map(
            |(
                (grid, p999, mean),
                (hop_hist, periodic_stall, alpha_eff_path, alpha_eff_includes_transfer, medium),
            )| LinkCapsExt {
                grid,
                p999,
                mean,
                hop_hist,
                periodic_stall,
                alpha_eff_path,
                alpha_eff_includes_transfer,
                medium,
            },
        )
}

fn ledger_summary() -> impl Strategy<Value = LedgerSummary> {
    prop::collection::vec((tier_id(), bytes()), 0..3).prop_map(|held| LedgerSummary { held })
}

fn link_change() -> impl Strategy<Value = LinkChange> {
    prop_oneof![
        Just(LinkChange::Added),
        Just(LinkChange::Removed),
        Just(LinkChange::TransportsChanged),
    ]
}

fn graph_change_cause() -> impl Strategy<Value = GraphChangeCause> {
    use GraphChangeCause::*;
    prop::sample::select(vec![Membership, StartupProfile, ModelProbe, Drift, Ledger])
}

fn graph_delta() -> impl Strategy<Value = GraphDelta> {
    (
        prop::collection::vec(node_id(), 0..3),
        prop::collection::vec(node_id(), 0..3),
        prop::collection::vec((node_id(), any::<u64>(), blake3()), 0..3),
        prop::collection::vec((node_id(), ledger_summary()), 0..3),
        prop::collection::vec((link_id(), link_change()), 0..3),
        graph_change_cause(),
    )
        .prop_map(
            |(nodes_added, nodes_removed, profiles, ledgers, links, cause)| GraphDelta {
                nodes_added,
                nodes_removed,
                profiles,
                ledgers,
                links,
                cause,
            },
        )
}

fn device_candidate() -> impl Strategy<Value = DeviceCandidate> {
    (
        device_kind(),
        prop::option::of(pci_address()),
        any::<Option<u16>>(),
        any::<Option<[u8; 16]>>(),
        text(),
        any::<bool>(),
    )
        .prop_map(|(kind, pci, numa, uuid, name, unified)| DeviceCandidate {
            kind,
            pci,
            numa,
            uuid,
            name,
            unified,
        })
}

fn enumerated() -> impl Strategy<Value = Enumerated> {
    (
        prop::collection::vec(device_candidate(), 0..4),
        prop::collection::vec(
            (any::<u16>(), any::<bool>()).prop_map(|(numa, pinned)| HostDomain { numa, pinned }),
            0..3,
        ),
        prop::collection::vec(
            (path(), any::<bool>())
                .prop_map(|(mount, holds_store)| VolumeCandidate { mount, holds_store }),
            0..3,
        ),
    )
        .prop_map(|(devices, domains, volumes)| Enumerated {
            devices,
            domains,
            volumes,
        })
}

fn tier_slot() -> impl Strategy<Value = TierSlot> {
    prop_oneof![
        device_id().prop_map(|device| TierSlot::Vram { device }),
        device_id().prop_map(|device| TierSlot::Unified { device }),
        any::<u16>().prop_map(|numa| TierSlot::HostPinned { numa }),
        any::<u16>().prop_map(|numa| TierSlot::HostPageable { numa }),
        path().prop_map(|mount| TierSlot::Nvme { mount }),
    ]
}

fn ordinals() -> impl Strategy<Value = Ordinals> {
    (
        prop::collection::vec((device_id(), device_candidate()), 0..3),
        prop::collection::vec((tier_id(), tier_slot()), 0..3),
    )
        .prop_map(|(devices, tiers)| Ordinals { devices, tiers })
}

// ---- Round trips ------------------------------------------------------------------------

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&back, value, "round trip through {json}");
}

macro_rules! round_trip_tests {
    ($($name:ident: $strategy:expr;)*) => {
        proptest! {
            $(
                #[test]
                fn $name(value in $strategy) {
                    round_trip(&value);
                }
            )*
        }
    };
}

round_trip_tests! {
    rt_node_id: node_id();
    rt_device_id: device_id();
    rt_tier_id: tier_id();
    rt_link_id: link_id();
    rt_endpoint: endpoint();
    rt_link_kind: link_kind();
    rt_net_medium: net_medium();
    rt_rdma_kind: rdma_kind();
    rt_blake3: blake3();
    rt_graph_version: any::<u64>().prop_map(GraphVersion);
    rt_bytes: bytes();
    rt_bytes_per_sec: bps();
    rt_flops_per_sec: f64s().prop_map(FlopsPerSec);
    rt_nanos: nanos();
    rt_quant_scheme: quant_scheme();
    rt_measured_nanos: m_nanos();
    rt_measured_bps: m_bps();
    rt_measured_f32: measured(f32s());
    rt_measure_source: measure_source();
    rt_error_class: error_class();
    rt_device_info: device_info();
    rt_device_kind: device_kind();
    rt_backend: backend();
    rt_ggml_backend: ggml_backend();
    rt_device_features: device_features();
    rt_device_class: device_class();
    rt_apple_gpu_family: apple_family();
    rt_cpu_arch: cpu_arch();
    rt_cpu_isa: cpu_isa();
    rt_api_ids: api_ids();
    rt_device_info_ext: device_info_ext();
    rt_driver_info: driver_info();
    rt_pci_info: pci_info();
    rt_pci_address: pci_address();
    rt_pcie_link: pcie_link();
    rt_pcie_upstream: pcie_upstream();
    rt_cpu_topology: cpu_topology();
    rt_isa_flags: isa_flags();
    rt_cpu_set: cpu_set();
    rt_power_info: power_info();
    rt_thermal_state: thermal_state();
    rt_device_problem: device_problem();
    rt_probe_step: probe_step();
    rt_spec_quantity: spec_quantity();
    rt_exclusion_source: exclusion_source();
    rt_tier_info: tier_info();
    rt_tier_kind: tier_kind();
    rt_tier_info_ext: tier_info_ext();
    rt_budget_source: budget_source();
    rt_host_mem_info: host_mem_info();
    rt_lock_accounting: lock_accounting();
    rt_volume_info: volume_info();
    rt_fs_kind: fs_kind();
    rt_storage_bus: storage_bus();
    rt_direct_io_support: direct_io();
    rt_temp_source: temp_source();
    rt_host_pool_info: host_pool_info();
    rt_link_info: link_info();
    rt_link_caps: link_caps();
    rt_demotion: demotion();
    rt_demotion_cause: demotion_cause();
    rt_iface_info: iface_info();
    rt_iface_flags: iface_flags();
    rt_ip_net: ip_net();
    rt_rdma_port_info: rdma_port_info();
    rt_transport_kind: transport_kind();
    rt_kernel_class: kernel_class();
    rt_attn_kind: attn_kind();
    rt_capability_profile_ext: capability_profile_ext();
    rt_profile_status: profile_status();
    rt_power_source: power_source();
    rt_device_caps_ext: device_caps_ext();
    rt_quant_bw: quant_bw();
    rt_launch_caps: launch_caps();
    rt_host_link_caps: host_link_caps();
    rt_storage_caps_ext: storage_caps_ext();
    rt_engine_kind: engine_kind();
    rt_dio_align: dio_align();
    rt_direct_path_report: direct_path_report();
    rt_io_point: io_point();
    rt_sustained_status: sustained_status();
    rt_throttle_info: throttle_info();
    rt_block_reason: block_reason();
    rt_link_caps_ext: link_caps_ext();
    rt_hop_histogram: hop_histogram();
    rt_device_path: device_path();
    rt_sync_kind: sync_kind();
    rt_node_inventory: node_inventory();
    rt_graph_delta: graph_delta();
    rt_ledger_summary: ledger_summary();
    rt_link_change: link_change();
    rt_graph_change_cause: graph_change_cause();
    rt_enumerated: enumerated();
    rt_ordinals: ordinals();
}

// ---- Unknown future fields --------------------------------------------------------------

/// Encodes `value`, adds unknown fields to every object in it, and decodes. Objects that
/// stand for an enum variant (a single key naming the variant) are left alone, since a
/// variant tag is not a record.
fn decodes_with_future_fields<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let mut json = serde_json::to_value(value).expect("serialize");
    add_future_fields_to_records(&mut json);
    let back: T = serde_json::from_value(json.clone())
        .unwrap_or_else(|e| panic!("decode with unknown fields failed: {e}\n{json}"));
    assert_eq!(&back, value);
}

/// Adds an unknown field to every JSON object in `value`, recursively, so nested records see
/// one too. Externally tagged enum objects are skipped: an object with one key that starts
/// with an upper-case letter is a variant tag, and only its content is visited.
fn add_future_fields_to_records(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            let is_variant = map.len() == 1
                && map
                    .keys()
                    .next()
                    .is_some_and(|k| k.starts_with(|c: char| c.is_ascii_uppercase()));
            for v in map.values_mut() {
                add_future_fields_to_records(v);
            }
            if !is_variant {
                map.insert(
                    "field_from_a_future_build".to_owned(),
                    serde_json::json!({ "nested": [1, 2, 3], "flag": true }),
                );
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(add_future_fields_to_records),
        _ => {}
    }
}

macro_rules! future_field_tests {
    ($($name:ident: $strategy:expr;)*) => {
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]
            $(
                #[test]
                fn $name(value in $strategy) {
                    decodes_with_future_fields(&value);
                }
            )*
        }
    };
}

future_field_tests! {
    ff_device_info: device_info();
    ff_tier_info: tier_info();
    ff_link_info: link_info();
    ff_iface_info: iface_info();
    ff_rdma_port_info: rdma_port_info();
    ff_host_pool_info: host_pool_info();
    ff_node_inventory: node_inventory();
    ff_capability_profile_ext: capability_profile_ext();
    ff_device_caps_ext: device_caps_ext();
    ff_quant_bw: quant_bw();
    ff_host_link_caps: host_link_caps();
    ff_storage_caps_ext: storage_caps_ext();
    ff_link_caps_ext: link_caps_ext();
    ff_graph_delta: graph_delta();
    ff_enumerated: enumerated();
}

/// The helper reaches a record nested inside a variant, and leaves the variant tag alone.
#[test]
fn future_fields_reach_nested_records() {
    let mut v = serde_json::json!({ "a": { "Derived": { "from": { "scheme": 1 } } } });
    add_future_fields_to_records(&mut v);
    assert!(
        v["a"]["Derived"]["from"]
            .get("field_from_a_future_build")
            .is_some()
    );
    assert!(v["a"].get("field_from_a_future_build").is_none());
}

// ---- Serde defaults ---------------------------------------------------------------------

/// A record written before RFC-0010's fields existed decodes, and each added field takes
/// its default.
#[test]
fn records_without_added_fields_decode_to_defaults() {
    let ext: DeviceInfoExt = serde_json::from_str("{}").unwrap();
    assert_eq!(ext, DeviceInfoExt::default());
    let ext: TierInfoExt = serde_json::from_str("{}").unwrap();
    assert_eq!(ext, TierInfoExt::default());
    let ext: CapabilityProfileExt = serde_json::from_str("{}").unwrap();
    assert_eq!(ext, CapabilityProfileExt::default());
    let ext: DeviceCapsExt = serde_json::from_str("{}").unwrap();
    assert_eq!(ext, DeviceCapsExt::default());
    let ext: StorageCapsExt = serde_json::from_str("{}").unwrap();
    assert_eq!(ext, StorageCapsExt::default());
    let ext: LinkCapsExt = serde_json::from_str("{}").unwrap();
    assert_eq!(ext, LinkCapsExt::default());
    assert!(ext.alpha_eff_includes_transfer);
    let inv: NodeInventory = serde_json::from_str("{}").unwrap();
    assert_eq!(inv, NodeInventory::default());
    let delta: GraphDelta = serde_json::from_str("{}").unwrap();
    assert_eq!(delta, GraphDelta::default());
    let m: Measured<Nanos> = serde_json::from_str(
        r#"{ "p50": 1, "p99": 2, "samples": 3, "at_unix_ms": 4, "source": "Startup" }"#,
    )
    .unwrap();
    assert_eq!(m.source, MeasureSource::Startup);
    assert_eq!(Measured::<Nanos>::default().source, MeasureSource::Prior);
}

/// A `DeviceInfo` with only RFC-0001's fields decodes, with an empty `ext`.
#[test]
fn rfc0001_device_info_decodes() {
    let node = NodeId([7; 32]);
    let json = serde_json::json!({
        "id": { "node": node, "ordinal": 0 },
        "kind": "DiscreteGpu",
        "name": "GPU 0",
        "class": { "Cuda": { "sm": 89 } },
        "backends": ["Cuda", { "Ggml": "Cuda" }],
        "features": "FP8_TC | GRAPH_CAPTURE",
        "local_tiers": [{ "node": node, "ordinal": 0 }],
        "numa": 0,
        "fingerprint": Blake3([0; 32]),
    });
    let info: DeviceInfo = serde_json::from_value(json).unwrap();
    assert_eq!(info.class, DeviceClass::Cuda { sm: 89 });
    assert_eq!(
        info.features,
        DeviceFeatures::FP8_TC | DeviceFeatures::GRAPH_CAPTURE
    );
    assert_eq!(info.ext, DeviceInfoExt::default());
}

// ---- Field defaults ---------------------------------------------------------------------

/// Encodes `value`, removes `key` from the top-level JSON object, and decodes the rest.
fn decode_without<T: Serialize + DeserializeOwned>(value: &T, key: &str) -> T {
    let mut json = serde_json::to_value(value).expect("serialize");
    let removed = json.as_object_mut().expect("a record").remove(key);
    assert!(removed.is_some(), "the record has no field {key}");
    serde_json::from_value(json).unwrap_or_else(|e| panic!("decode without {key} failed: {e}"))
}

fn sample_tier_id(ordinal: u16) -> TierId {
    TierId {
        node: NodeId([3; 32]),
        ordinal,
    }
}

fn sample_link_id() -> LinkId {
    let device = DeviceId {
        node: NodeId([3; 32]),
        ordinal: 0,
    };
    LinkId {
        src: Endpoint::Tier(sample_tier_id(1)),
        dst: Endpoint::Device(device),
        kind: LinkKind::Pcie,
        path: 0,
    }
}

/// A `TierInfo` with only RFC-0001's fields decodes, with an empty `ext`.
#[test]
fn rfc0001_tier_info_decodes() {
    let tier = TierInfo {
        id: sample_tier_id(2),
        kind: TierKind::HostPinned,
        capacity: Bytes(64 << 30),
        usable: Bytes(32 << 30),
        affinity: SmallVec::new(),
        pin_limit: Some(Bytes(32 << 30)),
        volume: None,
        ext: TierInfoExt {
            numa: Some(1),
            budget_source: BudgetSource::HostRules,
            host: None,
            context_reserve: None,
        },
    };
    let back = decode_without(&tier, "ext");
    assert_eq!(back.ext, TierInfoExt::default());
    assert_eq!(back.id, tier.id);
}

/// `HostMemInfo` and `VolumeInfo` decode from an empty object, every field defaulted.
#[test]
fn host_mem_info_and_volume_info_decode_from_nothing() {
    let host: HostMemInfo = serde_json::from_str("{}").unwrap();
    assert_eq!(host, HostMemInfo::default());
    let volume: VolumeInfo = serde_json::from_str("{}").unwrap();
    assert_eq!(volume, VolumeInfo::default());
}

/// `HostPoolInfo` decodes without `usable`, which defaults to zero.
#[test]
fn host_pool_info_usable_defaults() {
    let pool = HostPoolInfo {
        numa: 1,
        tiers: [sample_tier_id(2), sample_tier_id(3)],
        usable: Bytes(48 << 30),
        pin_limit: Bytes(16 << 30),
    };
    let back = decode_without(&pool, "usable");
    assert_eq!(back.usable, Bytes(0));
    assert_eq!(back.pin_limit, pool.pin_limit);
}

/// `QuantBw` decodes without `contended`, as from a device that is not `Unified`.
#[test]
fn quant_bw_contended_defaults() {
    let bw = QuantBw {
        tier: sample_tier_id(0),
        class: KernelClass::DenseGemv,
        scheme: QuantScheme::Ggml(GgmlType::Q4_K),
        curve: SmallVec::new(),
        extent: Bytes(1 << 30),
        contended: Some(Measured::default()),
    };
    let back = decode_without(&bw, "contended");
    assert_eq!(back.contended, None);
    assert_eq!(back.extent, bw.extent);
}

/// `HostLinkCaps` decodes without `p2p`, which defaults to no verified peers.
#[test]
fn host_link_caps_p2p_defaults() {
    let caps = HostLinkCaps {
        link: sample_link_id(),
        h2d_pinned: Measured::default(),
        d2h_pinned: Measured::default(),
        h2d_pageable: Measured::default(),
        h2d_small: Measured::default(),
        bidirectional: None,
        p2p: vec![(
            DeviceId {
                node: NodeId([3; 32]),
                ordinal: 1,
            },
            Measured::default(),
        )],
    };
    let back = decode_without(&caps, "p2p");
    assert!(back.p2p.is_empty());
    assert_eq!(back.link, caps.link);
}

/// `LinkInfo` decodes without `metered`, which defaults to false.
#[test]
fn link_info_metered_defaults() {
    let link = LinkInfo {
        id: sample_link_id(),
        transports: SmallVec::new(),
        interface: None,
        metered: true,
    };
    let back = decode_without(&link, "metered");
    assert!(!back.metered);
    assert_eq!(back.id, link.id);
}

/// A `DeviceInfoExt` and a `DriverInfo` written before ADR-0405's fields existed decode, with
/// `gpu_cores`, `api_ids` and `vk_driver` unset.
#[test]
fn records_without_adr_0405_fields_decode() {
    let driver = DriverInfo {
        driver: String::from("580.65"),
        runtime: Some(String::from("13.0")),
        loader: None,
        vk_driver: Some(VkDriverId(4)),
    };
    let back = decode_without(&driver, "vk_driver");
    assert_eq!(back.vk_driver, None);
    assert_eq!(back.driver, driver.driver);

    let ext = DeviceInfoExt {
        driver,
        gpu_cores: Some(10),
        api_ids: Some(ApiIds {
            vendor: 0x10de,
            device: 0x2684,
        }),
        ..DeviceInfoExt::default()
    };
    let back = decode_without(&ext, "gpu_cores");
    assert_eq!(back.gpu_cores, None);
    assert_eq!(back.api_ids, ext.api_ids);
    let back = decode_without(&ext, "api_ids");
    assert_eq!(back.api_ids, None);
    assert_eq!(back.gpu_cores, Some(10));

    // An absent `api_ids` is `None`; an `ApiIds` record must carry both IDs, so a partial
    // one is refused rather than read as 0.
    let ext: DeviceInfoExt = serde_json::from_str(r#"{ "api_ids": null }"#).unwrap();
    assert_eq!(ext.api_ids, None);
    assert!(serde_json::from_str::<DeviceInfoExt>(r#"{ "api_ids": {} }"#).is_err());
    assert!(serde_json::from_str::<ApiIds>(r#"{ "vendor": 4318 }"#).is_err());
    assert!(serde_json::from_str::<ApiIds>("{}").is_err());

    // A pre-ADR-0405 driver record inside an ext record, as JSON.
    let ext: DeviceInfoExt = serde_json::from_str(
        r#"{ "driver": { "driver": "1.4", "runtime": null, "loader": null } }"#,
    )
    .unwrap();
    assert_eq!(ext.driver.vk_driver, None);
    assert_eq!(ext.gpu_cores, None);
}

/// The CPU class carries a `CpuIsa`, and every variant has the encoding its name gives.
#[test]
fn cpu_isa_and_npu_encodings() {
    let class = DeviceClass::Cpu {
        isa: CpuIsa::X86V4Amx,
    };
    assert_eq!(
        serde_json::to_value(class).unwrap(),
        serde_json::json!({ "Cpu": { "isa": "X86V4Amx" } })
    );
    let npu = DeviceClass::Npu {
        vendor: 1,
        model: 2,
    };
    assert_eq!(
        serde_json::to_value(npu).unwrap(),
        serde_json::json!({ "Npu": { "vendor": 1, "model": 2 } })
    );
    let problem = DeviceProblem::ApiIdsMismatch {
        api: ApiIds {
            vendor: 0x1002,
            device: 0x744c,
        },
    };
    assert_eq!(
        serde_json::to_value(&problem).unwrap(),
        serde_json::json!({ "ApiIdsMismatch": { "api": { "vendor": 0x1002, "device": 0x744c } } })
    );
}

/// A `LinkInfo` written before RFC-0030's `interface` field existed decodes with no interface,
/// and an `IfaceInfo` with only its ordinal and name decodes with every other field at its
/// default (RFC-0030 §7.1).
#[test]
fn records_without_interface_fields_decode() {
    let iface = IfaceInfo {
        ordinal: 2,
        name: String::from("en0"),
        medium: NetMedium::WiFi,
        addrs: SmallVec::from_vec(vec![
            IpNet::new("192.0.2.10".parse().unwrap(), 24).unwrap(),
            IpNet::new("fe80::1".parse().unwrap(), 64)
                .unwrap()
                .with_scope(4),
        ]),
        mtu: 1500,
        speed_mbps: Some(1200),
        flags: IfaceFlags::UP | IfaceFlags::MULTICAST | IfaceFlags::DEFAULT_ROUTE,
        bridge: None,
        rdma: None,
        tb_peer: None,
    };
    let link = LinkInfo {
        id: sample_link_id(),
        transports: SmallVec::new(),
        interface: Some(iface.clone()),
        metered: false,
    };
    let back = decode_without(&link, "interface");
    assert_eq!(back.interface, None);
    assert_eq!(back.id, link.id);

    let bare: IfaceInfo = serde_json::from_str(r#"{"ordinal":2,"name":"en0"}"#).expect("decodes");
    assert_eq!(bare.medium, NetMedium::Ethernet);
    assert!(bare.addrs.is_empty());
    assert_eq!((bare.mtu, bare.speed_mbps), (0, None));
    assert_eq!(bare.flags, IfaceFlags::empty());
    assert_eq!((bare.bridge, bare.rdma, bare.tb_peer), (None, None, None));
    round_trip(&iface);
}

/// A `LinkCaps` that names only its link and transport decodes as unmeasured: zero figures with
/// source `Prior`, no `alpha_eff`, no loss.
#[test]
fn link_caps_decodes_as_unmeasured_without_its_figures() {
    let full = LinkCaps {
        link: LinkId {
            src: Endpoint::Nic {
                node: NodeId([1; 32]),
                ordinal: 0,
            },
            dst: Endpoint::Nic {
                node: NodeId([2; 32]),
                ordinal: 0,
            },
            kind: LinkKind::Network(NetMedium::Ethernet),
            path: 0,
        },
        transport: TransportKind::Quic,
        alpha: Measured::default(),
        beta: Measured::default(),
        alpha_eff: None,
        loss: None,
    };
    let mut json: serde_json::Value = serde_json::to_value(&full).unwrap();
    let object = json.as_object_mut().unwrap();
    for field in ["alpha", "beta", "alpha_eff", "loss"] {
        object.remove(field);
    }
    let back: LinkCaps = serde_json::from_value(json).unwrap();
    assert_eq!(back, full);
    assert!(!back.is_measured());
}
