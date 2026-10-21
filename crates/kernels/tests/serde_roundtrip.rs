// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serde round trips of the provider data types (COMP-E1-T01, RFC-0040 §2.2).
//!
//! Every data type goes through JSON and back unchanged. JSON stands in for any
//! self-describing format; the property is serde's, not the format's. `ProviderId` and
//! `ProviderBuild` hold `&'static str`, so they decode only from input that lives for the
//! whole program and holds the name unescaped; the helper leaks each encoding to give it that
//! lifetime.

use std::fmt::Debug;

use proptest::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::resource::{AppleGpuFamily, DeviceKind, GgmlBackend, PciAddress};
use syntropy_core::{Blake3, Bytes, ErrorClass};
use syntropy_kernels::*;

fn round_trip<T>(value: &T) -> Result<(), TestCaseError>
where
    T: Serialize + Deserialize<'static> + PartialEq + Debug,
{
    let json: &'static str = String::leak(serde_json::to_string(value).expect("encodes"));
    let back: T = serde_json::from_str(json).expect("decodes");
    prop_assert_eq!(&back, value);
    Ok(())
}

// ---- Strategies -------------------------------------------------------------------------

fn text() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop::sample::select(vec!['a', 'Z', '0', '-', ' ', 'é', '"', '\\']),
        0..8,
    )
    .prop_map(|c| c.into_iter().collect())
}

/// Names as providers and toolchains spell them. A borrowed `&str` cannot hold a JSON escape
/// sequence, so these draw from characters JSON writes as themselves; a name with a quote or
/// a backslash does not decode (`static_names_with_escapes_do_not_decode`).
fn static_text() -> impl Strategy<Value = &'static str> {
    prop::collection::vec(
        prop::sample::select(vec!['a', 'Z', '0', '-', '.', '_', ' ', 'é']),
        0..12,
    )
    .prop_map(|c| &*String::leak(c.into_iter().collect()))
}

fn bytes() -> impl Strategy<Value = Bytes> {
    any::<u64>().prop_map(Bytes)
}

fn provider_id() -> impl Strategy<Value = ProviderId> {
    static_text().prop_map(ProviderId)
}

fn provider_build() -> impl Strategy<Value = ProviderBuild> {
    (
        any::<u32>(),
        proptest::option::of(any::<[u8; 32]>().prop_map(Blake3)),
        prop::collection::vec((static_text(), text()), 0..5),
        proptest::option::of(text()),
    )
        .prop_map(|(version, module, toolchain, driver)| ProviderBuild {
            version,
            module,
            toolchain: SmallVec::from_vec(toolchain),
            driver,
        })
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

fn memory_domain() -> impl Strategy<Value = MemoryDomain> {
    prop_oneof![
        Just(MemoryDomain::Cuda),
        Just(MemoryDomain::Hip),
        Just(MemoryDomain::Metal),
        Just(MemoryDomain::Vulkan),
        Just(MemoryDomain::Host),
        ggml_backend().prop_map(MemoryDomain::Ggml),
    ]
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

fn apple_family() -> impl Strategy<Value = AppleGpuFamily> {
    prop_oneof![
        Just(AppleGpuFamily::Apple7),
        Just(AppleGpuFamily::Apple8),
        Just(AppleGpuFamily::Apple9),
        Just(AppleGpuFamily::Apple10),
    ]
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
    ]
}

fn device_features() -> impl Strategy<Value = DeviceFeatures> {
    any::<u64>().prop_map(DeviceFeatures::from_bits_truncate)
}

fn device_kind() -> impl Strategy<Value = DeviceKind> {
    prop_oneof![
        Just(DeviceKind::DiscreteGpu),
        Just(DeviceKind::IntegratedGpu),
        Just(DeviceKind::Cpu),
        Just(DeviceKind::Npu),
    ]
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

fn device_memory_info() -> impl Strategy<Value = DeviceMemoryInfo> {
    (
        bytes(),
        bytes(),
        proptest::option::of(bytes()),
        proptest::option::of(bytes()),
        bytes(),
        any::<bool>(),
    )
        .prop_map(
            |(total, free, budget, host_visible_device_local, max_alloc, unified)| {
                DeviceMemoryInfo {
                    total,
                    free,
                    budget,
                    host_visible_device_local,
                    max_alloc,
                    unified,
                }
            },
        )
}

fn provider_device() -> impl Strategy<Value = ProviderDevice> {
    (
        any::<u32>(),
        proptest::option::of(any::<[u8; 16]>()),
        proptest::option::of(pci_address()),
        device_kind(),
        text(),
        device_class(),
        device_features(),
        device_memory_info(),
        proptest::option::of(any::<u16>()),
    )
        .prop_map(
            |(local, uuid, pci, kind, name, class, features, memory, numa)| ProviderDevice {
                local,
                uuid,
                pci,
                kind,
                name,
                class,
                features,
                memory,
                numa,
            },
        )
}

fn stream_plan() -> impl Strategy<Value = StreamPlan> {
    any::<[u8; 4]>().prop_map(|[compute, copy_in, copy_out, collective]| StreamPlan {
        compute,
        copy_in,
        copy_out,
        collective,
    })
}

fn power_hint() -> impl Strategy<Value = PowerHint> {
    prop_oneof![Just(PowerHint::Mains), Just(PowerHint::Battery)]
}

fn open_options() -> impl Strategy<Value = OpenOptions> {
    (
        stream_plan(),
        any::<u32>(),
        any::<bool>(),
        any::<bool>(),
        power_hint(),
    )
        .prop_map(
            |(streams, doorbell_slots, capture, deterministic, power)| OpenOptions {
                streams,
                doorbell_slots,
                capture,
                deterministic,
                power,
            },
        )
}

fn stream_priority() -> impl Strategy<Value = StreamPriority> {
    prop_oneof![
        Just(StreamPriority::Compute),
        Just(StreamPriority::CopyIn),
        Just(StreamPriority::CopyOut),
        Just(StreamPriority::Collective),
        Just(StreamPriority::Low),
    ]
}

fn wait_cmp() -> impl Strategy<Value = WaitCmp> {
    prop_oneof![Just(WaitCmp::Eq), Just(WaitCmp::Geq)]
}

fn host_flag() -> impl Strategy<Value = HostFlag> {
    (any::<u16>(), any::<u16>()).prop_map(|(page, slot)| HostFlag { page, slot })
}

fn wake_kind() -> impl Strategy<Value = WakeKind> {
    prop_oneof![
        Just(WakeKind::Futex),
        Just(WakeKind::EventFd),
        Just(WakeKind::WinEvent),
        Just(WakeKind::TimelineValue),
    ]
}

fn host_waker() -> impl Strategy<Value = HostWaker> {
    (wake_kind(), any::<u64>()).prop_map(|(kind, handle)| HostWaker { kind, handle })
}

fn throttle_reason() -> impl Strategy<Value = ThrottleReason> {
    prop_oneof![
        Just(ThrottleReason::Thermal),
        Just(ThrottleReason::Power),
        Just(ThrottleReason::Other),
    ]
}

fn error_code() -> impl Strategy<Value = ErrorCode> {
    any::<u16>().prop_map(ErrorCode)
}

fn device_health() -> impl Strategy<Value = DeviceHealth> {
    prop_oneof![
        Just(DeviceHealth::Ok),
        throttle_reason().prop_map(|reason| DeviceHealth::Throttled { reason }),
        error_code().prop_map(|code| DeviceHealth::Sticky { code }),
    ]
}

fn host_reg_flags() -> impl Strategy<Value = HostRegFlags> {
    any::<u8>().prop_map(HostRegFlags::from_bits_truncate)
}

fn segment() -> impl Strategy<Value = Segment> {
    (any::<u64>(), bytes(), bytes(), any::<u64>()).prop_map(
        |(reservation, offset, bytes, handle)| Segment {
            reservation,
            offset,
            bytes,
            handle,
        },
    )
}

fn mem_kind() -> impl Strategy<Value = MemKind> {
    prop_oneof![
        Just(MemKind::Device),
        Just(MemKind::HostMapped),
        Just(MemKind::Unified),
    ]
}

fn alloc_request() -> impl Strategy<Value = AllocRequest> {
    (bytes(), mem_kind(), bytes(), any::<bool>()).prop_map(|(bytes, kind, align, reserve_only)| {
        AllocRequest {
            bytes,
            kind,
            align,
            reserve_only,
        }
    })
}

fn host_region() -> impl Strategy<Value = HostRegion> {
    (any::<u64>(), bytes()).prop_map(|(addr, len)| HostRegion { addr, len })
}

fn error_class() -> impl Strategy<Value = ErrorClass> {
    prop_oneof![
        Just(ErrorClass::InvalidRequest),
        Just(ErrorClass::Unsupported),
        Just(ErrorClass::Capacity),
        Just(ErrorClass::Transient),
        Just(ErrorClass::Link),
        Just(ErrorClass::Device),
        Just(ErrorClass::Integrity),
        Just(ErrorClass::Correctness),
        Just(ErrorClass::Drift),
        Just(ErrorClass::Timeout),
        Just(ErrorClass::Auth),
        Just(ErrorClass::Cancelled),
        Just(ErrorClass::Internal),
    ]
}

fn error() -> impl Strategy<Value = Error> {
    (error_class(), error_code(), text()).prop_map(|(class, code, message)| Error {
        class,
        code,
        message,
    })
}

// ---- Properties -------------------------------------------------------------------------

proptest! {
    #[test]
    fn provider_id_round_trips(v in provider_id()) { round_trip(&v)?; }

    #[test]
    fn provider_build_round_trips(v in provider_build()) { round_trip(&v)?; }

    #[test]
    fn memory_domain_round_trips(v in memory_domain()) { round_trip(&v)?; }

    #[test]
    fn provider_device_round_trips(v in provider_device()) { round_trip(&v)?; }

    #[test]
    fn device_class_round_trips(v in device_class()) { round_trip(&v)?; }

    #[test]
    fn cpu_isa_round_trips(v in cpu_isa()) { round_trip(&v)?; }

    #[test]
    fn device_features_round_trip(v in device_features()) { round_trip(&v)?; }

    #[test]
    fn device_memory_info_round_trips(v in device_memory_info()) { round_trip(&v)?; }

    #[test]
    fn open_options_round_trip(v in open_options()) { round_trip(&v)?; }

    #[test]
    fn stream_plan_round_trips(v in stream_plan()) { round_trip(&v)?; }

    #[test]
    fn power_hint_round_trips(v in power_hint()) { round_trip(&v)?; }

    #[test]
    fn stream_priority_round_trips(v in stream_priority()) { round_trip(&v)?; }

    #[test]
    fn wait_cmp_round_trips(v in wait_cmp()) { round_trip(&v)?; }

    #[test]
    fn host_flag_round_trips(v in host_flag()) { round_trip(&v)?; }

    #[test]
    fn host_waker_round_trips(v in host_waker()) { round_trip(&v)?; }

    #[test]
    fn wake_kind_round_trips(v in wake_kind()) { round_trip(&v)?; }

    #[test]
    fn device_health_round_trips(v in device_health()) { round_trip(&v)?; }

    #[test]
    fn throttle_reason_round_trips(v in throttle_reason()) { round_trip(&v)?; }

    #[test]
    fn host_reg_flags_round_trip(v in host_reg_flags()) { round_trip(&v)?; }

    #[test]
    fn segment_round_trips(v in segment()) { round_trip(&v)?; }

    #[test]
    fn alloc_request_round_trips(v in alloc_request()) { round_trip(&v)?; }

    #[test]
    fn mem_kind_round_trips(v in mem_kind()) { round_trip(&v)?; }

    #[test]
    fn host_region_round_trips(v in host_region()) { round_trip(&v)?; }

    #[test]
    fn error_code_round_trips(v in error_code()) { round_trip(&v)?; }

    #[test]
    fn error_round_trips(v in error()) { round_trip(&v)?; }
}

/// `HostRegFlags` carries RFC-0040 §2.2's bit values, which the tier manager maps onto.
#[test]
fn host_reg_flag_bits() {
    assert_eq!(HostRegFlags::DMA.bits(), 1);
    assert_eq!(HostRegFlags::MAPPED.bits(), 2);
}

/// The limit of a `&'static str` name: an encoding that must escape it cannot be borrowed,
/// so decoding fails rather than producing a different name.
#[test]
fn static_names_with_escapes_do_not_decode() {
    let json = serde_json::to_string(&ProviderId("a\"b")).expect("encodes");
    let json: &'static str = String::leak(json);
    assert!(serde_json::from_str::<ProviderId>(json).is_err());
}
