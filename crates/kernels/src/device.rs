// SPDX-License-Identifier: MIT OR Apache-2.0

//! A provider's identity and its view of the devices it drives (RFC-0040 §2.2): who the
//! provider is, which memory domain it belongs to, the devices it enumerates, the options it
//! opens a device with, the per-device context, and the health it reports.

use std::any::Any;
use std::fmt;
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use syntropy_core::resource::{AppleGpuFamily, DeviceKind, GgmlBackend, PciAddress};
use syntropy_core::{Blake3, Bytes};

use crate::error::ErrorCode;

pub use syntropy_core::resource::DeviceFeatures;

/// A provider's name: `"cuda"`, `"metal"`, `"cpu"`, `"hip"`, `"vulkan"`, `"ggml-cuda"`, ...
/// (RFC-0040 §2.2).
///
/// The name is a `&'static str`, as RFC-0040 §2.2 writes it. Decoding therefore borrows from
/// input that lives for the whole program (`'de: 'static`), and fails when the format had to
/// escape the name (in JSON, a quote, backslash or control character).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProviderId(pub &'static str);

/// A provider's build identity (RFC-0040 §2.1 `build`). Part of every selection-cache key
/// (RFC-0040 §6.5).
///
/// Like [`ProviderId`], the toolchain names are `&'static str`, so decoding borrows them from
/// input that lives for the whole program, with the same limit on escaped names.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
pub struct ProviderBuild {
    /// The provider's version.
    pub version: u32,
    /// BLAKE3 of the kernel module, for providers loaded from a provider library.
    pub module: Option<Blake3>,
    /// Compiler and toolchain versions, by tool name.
    pub toolchain: SmallVec<[(&'static str, String); 3]>,
    /// The driver or runtime version seen at load.
    pub driver: Option<String>,
}

/// A set of providers that share buffers, streams and events on one device without copies
/// (RFC-0040 §2.3). Selection never mixes domains on a device.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MemoryDomain {
    /// `cuda` and `ggml-cuda`: primary context, `CUdeviceptr`, `CUstream`, `CUevent`.
    Cuda,
    /// `hip` and `ggml-hip`: as CUDA.
    Hip,
    /// `metal` and `ggml-metal`: `MTLBuffer`, `MTLSharedEvent`, command queue.
    Metal,
    /// `vulkan` and `ggml-vulkan`: `VkBuffer` on one `VkDevice`, timeline semaphores.
    Vulkan,
    /// `cpu` and `ggml-cpu`: host pointers.
    Host,
    /// A ggml backend with no native provider (`ggml-sycl`, ...): ggml buffers only.
    Ggml(GgmlBackend),
}

/// One device as a provider sees it (RFC-0040 §2.2). The probe (RFC-0010) merges records
/// from several providers into one `DeviceInfo` by `uuid`, then PCI address, then
/// (vendor, name, memory).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderDevice {
    /// The provider-local index.
    pub local: u32,
    /// CUDA or HIP UUID, `VkPhysicalDeviceIDProperties`, or the hashed `MTLDevice` registry ID.
    pub uuid: Option<[u8; 16]>,
    /// The PCI address, where the device has one.
    pub pci: Option<PciAddress>,
    /// What kind of engine the device is.
    pub kind: DeviceKind,
    /// The name the driver reports.
    pub name: String,
    /// The device class as RFC-0040 §2.2 defines it.
    pub class: DeviceClass,
    /// Capability bits detected at runtime (ADR-0014 rule 1).
    pub features: DeviceFeatures,
    /// The device's memory when it was enumerated.
    pub memory: DeviceMemoryInfo,
    /// NUMA domain, where known.
    pub numa: Option<u16>,
}

/// A device's class as RFC-0040 §2.2 defines it: part of `PackKey` (RFC-0001 §5.6), so two
/// devices with equal classes accept the same pack.
///
/// **This is not `syntropy_core::resource::DeviceClass`.** RFC-0010 §1.2 defines a
/// `DeviceClass` with different fields (Metal by family and core count, Vulkan by vendor,
/// device and driver, CPU by architecture and ISA level, plus an `Npu` variant), and `core`
/// implements that one. The two RFCs conflict, and which definition `PackKey` uses is a
/// maintainer decision (issue #1499, item 1). Until it is made, this crate carries RFC-0040's
/// definition as COMP-E1-T01 lists it, and `core`'s type is unchanged.
///
/// RFC-0040 names the Metal family type `AppleFamily`; this uses `core`'s
/// [`AppleGpuFamily`], RFC-0010's type for the same Metal GPU family.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DeviceClass {
    /// An NVIDIA GPU by compute capability: 75, 80, 86, 89, 90, 100, 120, 121, ...
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
        /// Whether the device runs the Metal 4 language revision.
        metal4: bool,
    },
    /// A GPU driven through Vulkan.
    Vulkan {
        /// PCI vendor ID.
        vendor: u32,
        /// Whether cooperative matrix is available.
        coopmat: bool,
        /// Subgroup size.
        subgroup: u8,
    },
    /// The CPU cores of one NUMA domain.
    Cpu {
        /// The ISA level the CPU provider dispatches to (RFC-0040 §7.3).
        isa: CpuIsa,
    },
}

/// The CPU provider's ISA levels (RFC-0040 §7.3). Detection is COMP-E3-T01's work.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CpuIsa {
    /// SSE4.2 and POPCNT: the correctness floor.
    X86V2,
    /// AVX2, FMA and F16C.
    X86V3,
    /// AVX-512 F, BW, VL and DQ.
    X86V4,
    /// `X86V4` plus AMX-TILE and AMX-INT8.
    X86V4Amx,
    /// NEON.
    Arm80,
    /// NEON with SDOT/UDOT and FP16 arithmetic.
    Arm82Dot,
    /// `Arm82Dot` plus I8MM and BF16.
    Arm86I8mm,
    /// `Arm86I8mm` plus SME2.
    Arm92Sme2,
}

/// Free and budgeted device memory (RFC-0040 §2.2; `memory_info` in §2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceMemoryInfo {
    /// Total device memory.
    pub total: Bytes,
    /// Free device memory.
    pub free: Bytes,
    /// The OS-granted budget where one exists (DXGI, Vulkan, the Metal working set).
    pub budget: Option<Bytes>,
    /// The resizable-BAR window, where one exists.
    pub host_visible_device_local: Option<Bytes>,
    /// The largest single allocation.
    pub max_alloc: Bytes,
    /// Whether device memory is the host's memory.
    pub unified: bool,
}

/// How a device is opened (RFC-0040 §2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpenOptions {
    /// Streams to create, per priority (RFC-0040 §4.1).
    pub streams: StreamPlan,
    /// Host flags to pre-register: at least MoE layers × 2 + 8.
    pub doorbell_slots: u32,
    /// Whether graph capture may be used; false is the `graph_capture` kill switch.
    pub capture: bool,
    /// Deterministic kernel variants only (ADR-0019 decision 8).
    pub deterministic: bool,
    /// Mains or battery: spin budgets and clock hints.
    pub power: PowerHint,
}

/// How many streams `open` creates per priority (RFC-0040 §4.1). RFC-0040 §2.2 names the
/// type and its four priorities; the default plan is one each of compute, copy-in and
/// copy-out, plus one collective stream for `IntraNodeTp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamPlan {
    /// `Compute` streams.
    pub compute: u8,
    /// `CopyIn` streams (host to device).
    pub copy_in: u8,
    /// `CopyOut` streams (device to host).
    pub copy_out: u8,
    /// `Collective` streams.
    pub collective: u8,
}

/// The power hint `open` receives (RFC-0040 §2.2: `Mains | Battery`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerHint {
    /// On mains power.
    Mains,
    /// On battery power.
    Battery,
}

/// A provider's per-device state: the backend context, module handles, pipeline caches and
/// the doorbell page (RFC-0040 §2.2).
///
/// It exists only on the device's dispatcher thread (RFC-0040 §2.4, ADR-0012), so it is
/// neither `Send` nor `Sync`. Every [`KernelProvider`](crate::KernelProvider) method that
/// takes `&DeviceCtx` can therefore run only on that thread.
///
/// It cannot cross threads:
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<syntropy_kernels::DeviceCtx>();
/// ```
///
/// It cannot be shared between threads:
///
/// ```compile_fail
/// fn assert_sync<T: Sync>() {}
/// assert_sync::<syntropy_kernels::DeviceCtx>();
/// ```
///
/// The same helpers compile for a type that is both, and the path above resolves, so the two
/// tests above fail only for the missing bound:
///
/// ```
/// fn assert_send<T: Send>() {}
/// fn assert_sync<T: Sync>() {}
/// assert_send::<syntropy_kernels::HostFlag>();
/// assert_sync::<syntropy_kernels::HostFlag>();
/// let ctx = syntropy_kernels::DeviceCtx::new(7u32);
/// assert_eq!(ctx.downcast_ref::<u32>(), Some(&7));
/// ```
pub struct DeviceCtx {
    inner: Box<dyn Any>,
    _not_send: PhantomData<*const ()>,
}

impl DeviceCtx {
    /// Wraps a provider's per-device state.
    #[must_use]
    pub fn new<T: Any>(inner: T) -> Self {
        Self {
            inner: Box::new(inner),
            _not_send: PhantomData,
        }
    }

    /// The provider's state, if it is a `T`.
    #[must_use]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }

    /// The provider's state, mutably, if it is a `T`.
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.inner.downcast_mut()
    }
}

impl fmt::Debug for DeviceCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceCtx").finish_non_exhaustive()
    }
}

/// A device's health (RFC-0040 §2.2; `health` in RFC-0001 §5.7). A sticky error fails the
/// runner with class `Device`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceHealth {
    /// No problem.
    Ok,
    /// Running, but with clocks reduced.
    Throttled {
        /// Why the clocks are reduced.
        reason: ThrottleReason,
    },
    /// A sticky device error: the device is unusable until the runner restarts.
    Sticky {
        /// The error's code.
        code: ErrorCode,
    },
}

/// Why a device's clocks are reduced. RFC-0040 §2.2 names the type without variants; these
/// are the two causes drivers report, plus one for any other.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThrottleReason {
    /// Heat.
    Thermal,
    /// A power limit.
    Power,
    /// Any other cause the driver reports.
    Other,
}
