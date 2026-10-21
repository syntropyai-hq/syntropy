// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device topology records: RFC-0001 §5.2's `DeviceInfo` and RFC-0010 §1.2's additions.

use std::path::PathBuf;

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::error::ErrorClass;
use crate::ids::{Blake3, DeviceId, TierId};
use crate::units::Bytes;

/// One compute engine on a node (RFC-0001 §5.2). Owner: RFC-0010.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// The device's identity; the ordinal follows RFC-0010 §1.4.
    pub id: DeviceId,
    /// What kind of engine it is.
    pub kind: DeviceKind,
    /// The name the driver reports.
    pub name: String,
    /// Backend plus architecture generation, e.g. CUDA sm_120, Metal Apple9, x86-64-v4.
    /// Part of `PackKey` (RFC-0001 §5.6).
    pub class: DeviceClass,
    /// Providers able to drive this device, in preference order (RFC-0040).
    pub backends: SmallVec<[Backend; 4]>,
    /// Capability bits detected at runtime (ADR-0014 rule 1).
    pub features: DeviceFeatures,
    /// Tiers this device executes against at memory-bus speed (execute-in-place sites).
    pub local_tiers: SmallVec<[TierId; 2]>,
    /// NUMA domain, where known.
    pub numa: Option<u16>,
    /// Detects hardware changes behind a stable ordinal.
    pub fingerprint: Blake3,
    /// RFC-0010 §1.2's additions.
    #[serde(default)]
    pub ext: DeviceInfoExt,
}

/// What kind of compute engine a device is (RFC-0001 §5.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeviceKind {
    /// A GPU with its own memory behind PCIe or a Thunderbolt tunnel.
    DiscreteGpu,
    /// A GPU that shares the host's memory.
    IntegratedGpu,
    /// The CPU cores of one NUMA domain.
    Cpu,
    /// A neural processing unit.
    Npu,
}

/// A provider family that can drive a device (RFC-0001 §5.2; RFC-0040 owns providers).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Backend {
    /// The native CUDA provider.
    Cuda,
    /// The native HIP provider.
    Hip,
    /// The native Metal provider.
    Metal,
    /// The native Vulkan provider.
    Vulkan,
    /// The native CPU provider.
    Cpu,
    /// The ggml provider, through one of its backends.
    Ggml(GgmlBackend),
}

/// The ggml backend a [`Backend::Ggml`] provider runs on.
///
/// RFC-0001 §5.2 names the type without variants. These are the backends RFC-0001 §4.2
/// rule 7 says ship (each GPU module's ggml backend, ggml SYCL, and ggml on the CPU).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GgmlBackend {
    /// ggml's CPU backend.
    Cpu,
    /// ggml's CUDA backend.
    Cuda,
    /// ggml's HIP backend.
    Hip,
    /// ggml's Metal backend.
    Metal,
    /// ggml's Vulkan backend.
    Vulkan,
    /// ggml's SYCL backend.
    Sycl,
}

bitflags! {
    /// Capability bits of a device (RFC-0040 §2). Every feature is detected at runtime and
    /// recorded here (ADR-0014 rule 1).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct DeviceFeatures: u64 {
        // GPU matrix and numeric paths.
        /// FP4 tensor cores.
        const FP4_TC = 1 << 0;
        /// FP8 tensor cores.
        const FP8_TC = 1 << 1;
        /// INT8 tensor cores.
        const INT8_TC = 1 << 2;
        /// BF16 matrix units.
        const BF16_MMA = 1 << 3;
        /// FP16 matrix units.
        const FP16_MMA = 1 << 4;
        /// Vulkan cooperative matrix.
        const COOP_MATRIX = 1 << 5;
        /// Vulkan cooperative matrix 2.
        const COOP_MATRIX2 = 1 << 6;
        /// Integer dot product.
        const INT_DOT = 1 << 7;
        /// Metal 4 tensor operations.
        const METAL4_TENSOR = 1 << 8;
        /// GPU neural accelerators.
        const NEURAL_ACCEL = 1 << 9;
        // GPU execution and memory.
        /// Graph capture.
        const GRAPH_CAPTURE = 1 << 16;
        /// Stream memory operations.
        const STREAM_MEM_OPS = 1 << 17;
        /// Virtual memory management.
        const VMM = 1 << 18;
        /// Host memory registration.
        const HOST_REGISTER = 1 << 19;
        /// Host memory import.
        const HOST_IMPORT = 1 << 20;
        /// Peer-to-peer copies.
        const P2P = 1 << 21;
        /// Unified memory.
        const UNIFIED = 1 << 22;
        /// Metal residency sets.
        const RESIDENCY_SETS = 1 << 23;
        /// A memory budget query.
        const MEMORY_BUDGET = 1 << 24;
        /// Batched asynchronous copies.
        const BATCHED_COPY = 1 << 25;
        /// Shared events.
        const SHARED_EVENTS = 1 << 26;
        /// Resizable BAR.
        const REBAR = 1 << 27;
        // CPU ISA.
        /// AVX2.
        const AVX2 = 1 << 32;
        /// AVX-512 foundation.
        const AVX512 = 1 << 33;
        /// AVX-512 VNNI.
        const AVX512_VNNI = 1 << 34;
        /// AVX-512 BF16.
        const AVX512_BF16 = 1 << 35;
        /// AVX-512 FP16.
        const AVX512_FP16 = 1 << 36;
        /// AVX-VNNI.
        const AVX_VNNI = 1 << 37;
        /// AMX INT8.
        const AMX_INT8 = 1 << 38;
        /// AMX BF16.
        const AMX_BF16 = 1 << 39;
        /// NEON dot product.
        const NEON_DOTPROD = 1 << 40;
        /// NEON int8 matrix multiply.
        const NEON_I8MM = 1 << 41;
        /// NEON BF16.
        const NEON_BF16 = 1 << 42;
        /// SVE2.
        const SVE2 = 1 << 43;
        /// SME2.
        const SME2 = 1 << 44;
        /// NEON FP16 arithmetic.
        const NEON_FP16 = 1 << 45;
    }
}

/// The pair that decides which kernels and pack layouts apply (RFC-0010 §1.2). RFC-0001
/// declares it as a `PackKey` component: two devices with equal `DeviceClass` share packs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeviceClass {
    /// An NVIDIA GPU by compute capability: 75, 86, 89, 90, 100, 120, 121, ...
    Cuda {
        /// Compute capability, major × 10 + minor.
        sm: u16,
    },
    /// An AMD GPU by `gfx` target: 0x1100, 0x1201, 0x942, ...
    Hip {
        /// The `gfx` target number.
        gfx: u32,
    },
    /// An Apple GPU.
    Metal {
        /// The Metal GPU family.
        family: AppleGpuFamily,
        /// GPU core count.
        cores: u16,
    },
    /// A GPU driven through Vulkan.
    Vulkan {
        /// PCI vendor ID.
        vendor: u32,
        /// PCI device ID.
        device: u32,
        /// The Vulkan driver.
        driver: VkDriverId,
    },
    /// The CPU cores of one NUMA domain.
    Cpu {
        /// Instruction set architecture.
        arch: CpuArch,
        /// Microarchitecture level.
        level: IsaLevel,
    },
    /// A neural processing unit.
    Npu {
        /// Vendor ID.
        vendor: u32,
        /// Model ID.
        model: u32,
    },
}

/// The Metal GPU family of an Apple GPU. RFC-0010 §1.2 names the type without variants;
/// these are the families ADR-0014 supports, Apple7 (M1, A14) and newer.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AppleGpuFamily {
    /// Apple7 (M1, A14): the oldest family ADR-0014 supports.
    Apple7,
    /// Apple8.
    Apple8,
    /// Apple9.
    Apple9,
    /// Apple10.
    Apple10,
}

/// A Vulkan `VkDriverId` value, as `VkPhysicalDeviceDriverProperties::driverID` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VkDriverId(pub u32);

/// A CPU instruction set architecture. RFC-0010 §1.2 names the type without variants.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CpuArch {
    /// x86-64.
    X86_64,
    /// AArch64.
    Aarch64,
}

/// A CPU microarchitecture level (RFC-0010 §1.2: x86-64-v2/v3/v4, aarch64-v8.0,
/// v8.2-dotprod, v9-sve2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum IsaLevel {
    /// x86-64-v2.
    X86_64V2,
    /// x86-64-v3.
    X86_64V3,
    /// x86-64-v4.
    X86_64V4,
    /// AArch64 v8.0.
    Aarch64V8_0,
    /// AArch64 v8.2 with the dot-product extension.
    Aarch64V8_2Dotprod,
    /// AArch64 v9 with SVE2.
    Aarch64V9Sve2,
}

/// RFC-0010 §1.2's additions to [`DeviceInfo`]. Every field has a serde default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceInfoExt {
    /// Vendor driver version, runtime version, loader path.
    pub driver: DriverInfo,
    /// `None` for Apple GPUs, CPU devices and most NPUs.
    pub pci: Option<PciInfo>,
    /// NVML, HIP or Vulkan device UUID; the dedupe key across providers.
    pub uuid: Option<[u8; 16]>,
    /// Windows adapter LUID (DXGI, CUDA and Vulkan match on it).
    pub luid: Option<u64>,
    /// CPU devices only.
    pub cpu: Option<CpuTopology>,
    /// Power limit, current cap, thermal state.
    pub power: Option<PowerInfo>,
    /// Why a present device is unusable or degraded.
    pub problems: SmallVec<[DeviceProblem; 2]>,
}

/// A device's driver stack. RFC-0010 §1.2 names the type and its three parts.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DriverInfo {
    /// The vendor driver's version string.
    pub driver: String,
    /// The vendor runtime's version string, where one is separate from the driver.
    pub runtime: Option<String>,
    /// The library the driver was loaded from, where one was.
    pub loader: Option<PathBuf>,
}

/// A PCI function's identity and link (RFC-0010 §1.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PciInfo {
    /// domain:bus:device.function.
    pub bdf: PciAddress,
    /// PCI vendor ID.
    pub vendor: u16,
    /// PCI device ID.
    pub device: u16,
    /// PCI subsystem ID.
    pub subsystem: u32,
    /// Current link, read at idle and under load.
    pub link_cur: PcieLink,
    /// Maximum link.
    pub link_max: PcieLink,
    /// What the function hangs off.
    pub upstream: PcieUpstream,
    /// BAR1 or CPU-visible VRAM equals VRAM size.
    pub resizable_bar: Option<bool>,
    /// NUMA domain of the slot, where known.
    pub numa: Option<u16>,
}

/// A PCI address, domain:bus:device.function. The derived order is the order RFC-0010 §1.4
/// numbers discrete GPUs in.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct PciAddress {
    /// PCI segment (domain).
    pub domain: u32,
    /// Bus number.
    pub bus: u8,
    /// Device number.
    pub device: u8,
    /// Function number.
    pub function: u8,
}

/// A PCIe link's generation and width (RFC-0010 §1.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct PcieLink {
    /// PCIe generation.
    pub r#gen: u8,
    /// Lane count.
    pub width: u8,
}

/// What a PCI function's link goes up to (RFC-0010 §1.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PcieUpstream {
    /// A CPU root port.
    CpuRootPort,
    /// A chipset, which may share its uplink.
    Chipset,
    /// A PCIe switch.
    Switch,
    /// A Thunderbolt PCIe tunnel.
    ThunderboltTunnel,
    /// Not determined. The serde default.
    #[default]
    Unknown,
}

/// CPU topology of one CPU device (RFC-0010 §1.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CpuTopology {
    /// NUMA domain.
    pub numa: u16,
    /// Physical cores.
    pub physical_cores: u16,
    /// Logical CPUs.
    pub logical_cpus: u16,
    /// Hybrid parts: P-cores; homogeneous parts: all cores.
    pub perf_cores: u16,
    /// Hybrid parts: E-cores.
    pub efficiency_cores: u16,
    /// Last-level cache per domain, summed over CCDs.
    pub llc: Bytes,
    /// Core complexes sharing one fabric link (0 = unknown).
    pub ccds: u8,
    /// Instruction set extensions.
    pub isa: IsaFlags,
    /// `arch_prctl(ARCH_REQ_XCOMP_PERM)` granted.
    pub amx_permitted: bool,
    /// Cores ADR-0012 reserves: dispatchers, I/O, net, user.
    pub reserved_cpus: CpuSet,
}

bitflags! {
    /// CPU instruction set extensions (RFC-0010 §1.2). RFC-0010 names the type and lists
    /// the extensions; the bit positions are this crate's.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct IsaFlags: u32 {
        /// AVX2.
        const AVX2 = 1 << 0;
        /// AVX-512 F.
        const AVX512F = 1 << 1;
        /// AVX-512 BW.
        const AVX512BW = 1 << 2;
        /// AVX-512 VL.
        const AVX512VL = 1 << 3;
        /// AVX-512 VNNI.
        const AVX512VNNI = 1 << 4;
        /// AVX-512 BF16.
        const AVX512BF16 = 1 << 5;
        /// AVX-512 FP16.
        const AVX512FP16 = 1 << 6;
        /// AMX BF16.
        const AMX_BF16 = 1 << 7;
        /// AMX INT8.
        const AMX_INT8 = 1 << 8;
        /// NEON dot product.
        const NEON_DOTPROD = 1 << 9;
        /// NEON int8 matrix multiply.
        const NEON_I8MM = 1 << 10;
        /// SVE2.
        const SVE2 = 1 << 11;
        /// SME2.
        const SME2 = 1 << 12;
    }
}

/// A set of logical CPUs: bit `i % 64` of word `i / 64` is logical CPU `i`. RFC-0010 §1.2
/// names the type without fields.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CpuSet(pub Vec<u64>);

impl CpuSet {
    /// Adds logical CPU `cpu`.
    pub fn insert(&mut self, cpu: u32) {
        let word = (cpu / 64) as usize;
        if self.0.len() <= word {
            self.0.resize(word + 1, 0);
        }
        self.0[word] |= 1 << (cpu % 64);
    }

    /// Whether logical CPU `cpu` is in the set.
    pub fn contains(&self, cpu: u32) -> bool {
        self.0
            .get((cpu / 64) as usize)
            .is_some_and(|w| w & (1 << (cpu % 64)) != 0)
    }
}

/// A device's power limit, current cap and thermal state (RFC-0010 §1.2, which names the type
/// and its three parts).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PowerInfo {
    /// The board power limit, in watts.
    pub limit_w: Option<f32>,
    /// The power cap in effect now, in watts.
    pub cap_w: Option<f32>,
    /// The thermal state the OS or driver reports.
    pub thermal: ThermalState,
}

/// A device's thermal state, coarsened to four levels.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThermalState {
    /// Not reported. The serde default.
    #[default]
    Unknown,
    /// No thermal limit in effect.
    Nominal,
    /// Clocks reduced for heat.
    Throttled,
    /// At or near shutdown temperature.
    Critical,
}

/// Why a present device is unusable or degraded (RFC-0010 §1.2).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DeviceProblem {
    /// The driver is older than the provider needs.
    DriverTooOld {
        /// The installed version.
        have: String,
        /// The oldest version that works.
        need: String,
    },
    /// Enumerated, but no `KernelProvider` drives it (e.g. most NPUs in v1).
    NoProvider,
    /// The probe runner crashed during a step.
    ProbeCrashed {
        /// The step that was running.
        suite: ProbeStep,
        /// The error's class.
        class: ErrorClass,
    },
    /// The PCIe link trained below its maximum, e.g. an x16 card at x4.
    LinkDegraded {
        /// The current link.
        cur: PcieLink,
        /// The maximum link.
        max: PcieLink,
    },
    /// A measured value below 0.5 × spec: a misconfiguration flag (RFC-0010 §3.3 rule 4).
    BelowSpec {
        /// What was measured.
        quantity: SpecQuantity,
        /// The measured value.
        measured: f64,
        /// The spec bound.
        spec: f64,
    },
    /// Excluded by user configuration or `features.disable`.
    Excluded {
        /// What excluded it.
        by: ExclusionSource,
    },
}

/// A step of the startup probe (RFC-0010 §2.1 E1–E5, §3.2 table). RFC-0010 §1.2 names the
/// type without variants.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProbeStep {
    /// E1 host enumeration.
    E1,
    /// E2 GPU inventory from management APIs.
    E2,
    /// E3 storage enumeration.
    E3,
    /// E4 network inventory.
    E4,
    /// E5 provider enumeration.
    E5,
    /// D1 open.
    D1,
    /// D2 capacity.
    D2,
    /// M1 memory W_eff.
    M1,
    /// M2 batch curve.
    M2,
    /// M3 fixed cost.
    M3,
    /// M4 compute F_eff.
    M4,
    /// M5 host-wait stack.
    M5,
    /// M6 CPU partial attention.
    M6,
    /// H1 PCIe.
    H1,
    /// H2 hand-off.
    H2,
    /// H3 DMA interference.
    H3,
    /// H4 peer copies.
    H4,
    /// H5 lock accounting.
    H5,
    /// S1 NVMe quick and engine selection.
    S1,
    /// S2 NVMe sustained.
    S2,
    /// L1 links.
    L1,
    /// L2 device path.
    L2,
}

/// A quantity with a spec bound that RFC-0010 §3.3 rule 4 checks. RFC-0010 §1.2 names the
/// type without variants; these are the quantities the §3.2 steps measure.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SpecQuantity {
    /// Weight bytes per second (M1).
    MemoryBandwidth,
    /// Compute throughput (M4).
    Compute,
    /// Host-to-device copy rate (H1).
    PcieH2d,
    /// Device-to-host copy rate (H1).
    PcieD2h,
    /// Storage read rate (S1, S2).
    StorageRead,
    /// Network goodput (L1).
    LinkBandwidth,
}

/// What excluded a device (RFC-0010 §1.2: user config, or `features.disable`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ExclusionSource {
    /// The user's configuration excludes the device.
    UserConfig,
    /// `features.disable` turns off a feature the device needs (ADR-0014 rule 4).
    FeaturesDisable,
}
