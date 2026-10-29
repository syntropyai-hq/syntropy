// SPDX-License-Identifier: MIT OR Apache-2.0

//! The codes and records that carry the trait's types across the boundary (RFC-0040 §10.4).
//!
//! Enums cross as `u32` codes. A code table only grows, and a value this build does not know
//! has no code (`None`), which the caller reports as an error rather than guessing. Records
//! that are not plain integers cross as JSON.

use serde::{Deserialize, Serialize};
use syntropy_core::resource::{Backend, GgmlBackend};
use syntropy_core::{
    AccumKind, ActQuant, Blake3, BufferView, Bytes, DType, DeviceId, Nanos, NodeId, Numerics,
    ViewKind,
};

use super::types::{
    SynBuffer, SynBufferView, SynDeviceId, SynHealth, SynHostRegion, SynMemoryInfo, SynOpenOptions,
    SynRegistration, SynScalar, SynSegment, SynSlice, SynTensorArg,
};
use crate::device::{
    DeviceHealth, DeviceMemoryInfo, MemoryDomain, OpenOptions, PowerHint, ProviderBuild,
    StreamPlan, ThrottleReason,
};
use crate::error::{Error, ErrorCode};
use crate::memory::{
    AllocRequest, BufferSlice, DeviceBuffer, HostRegion, MemKind, Registration, Segment,
};
use crate::op::{LayoutVersion, OpCaps, Scalar, TensorArg};
use crate::stream::{StreamPriority, WaitCmp, WakeKind};
use syntropy_core::ErrorClass;

const GGML_BASE: u32 = 16;

fn ggml_code(backend: GgmlBackend) -> Option<u32> {
    Some(match backend {
        GgmlBackend::Cpu => 0,
        GgmlBackend::Cuda => 1,
        GgmlBackend::Hip => 2,
        GgmlBackend::Metal => 3,
        GgmlBackend::Vulkan => 4,
        GgmlBackend::Sycl => 5,
        _ => return None,
    })
}

fn ggml_of_code(code: u32) -> Option<GgmlBackend> {
    Some(match code {
        0 => GgmlBackend::Cpu,
        1 => GgmlBackend::Cuda,
        2 => GgmlBackend::Hip,
        3 => GgmlBackend::Metal,
        4 => GgmlBackend::Vulkan,
        5 => GgmlBackend::Sycl,
        _ => return None,
    })
}

/// `s` as a `&'static str` the process keeps once: the first call with a given text allocates
/// it, and every later call returns that same allocation. A loaded module's provider ID and its
/// build's toolchain keys are `&'static str` in the types they fill (`ProviderId`,
/// `ProviderBuild`), so the loader interns them instead of leaking a copy per load: the set grows
/// only with the distinct texts, and every string in it stays reachable from a static, so a leak
/// checker reports none.
#[must_use]
pub fn intern(s: &str) -> &'static str {
    static INTERNED: std::sync::Mutex<std::collections::BTreeSet<&'static str>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    let mut set = INTERNED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&kept) = set.get(s) {
        return kept;
    }
    let kept: &'static str = Box::leak(s.to_owned().into_boxed_str());
    set.insert(kept);
    kept
}

/// The code of a [`Backend`]: 1 CUDA, 2 HIP, 3 Metal, 4 Vulkan, 5 CPU, 16 + n for ggml's
/// backend n.
#[must_use]
pub fn backend_code(backend: Backend) -> Option<u32> {
    Some(match backend {
        Backend::Cuda => 1,
        Backend::Hip => 2,
        Backend::Metal => 3,
        Backend::Vulkan => 4,
        Backend::Cpu => 5,
        Backend::Ggml(g) => GGML_BASE + ggml_code(g)?,
        _ => return None,
    })
}

/// The [`Backend`] `code` names.
#[must_use]
pub fn backend_of_code(code: u32) -> Option<Backend> {
    Some(match code {
        1 => Backend::Cuda,
        2 => Backend::Hip,
        3 => Backend::Metal,
        4 => Backend::Vulkan,
        5 => Backend::Cpu,
        c if c >= GGML_BASE => Backend::Ggml(ggml_of_code(c - GGML_BASE)?),
        _ => return None,
    })
}

/// The code of a [`MemoryDomain`]: 1 CUDA, 2 HIP, 3 Metal, 4 Vulkan, 5 host, 16 + n for ggml's
/// backend n.
#[must_use]
pub fn domain_code(domain: MemoryDomain) -> Option<u32> {
    Some(match domain {
        MemoryDomain::Cuda => 1,
        MemoryDomain::Hip => 2,
        MemoryDomain::Metal => 3,
        MemoryDomain::Vulkan => 4,
        MemoryDomain::Host => 5,
        MemoryDomain::Ggml(g) => GGML_BASE + ggml_code(g)?,
    })
}

/// The [`MemoryDomain`] `code` names.
#[must_use]
pub fn domain_of_code(code: u32) -> Option<MemoryDomain> {
    Some(match code {
        1 => MemoryDomain::Cuda,
        2 => MemoryDomain::Hip,
        3 => MemoryDomain::Metal,
        4 => MemoryDomain::Vulkan,
        5 => MemoryDomain::Host,
        c if c >= GGML_BASE => MemoryDomain::Ggml(ggml_of_code(c - GGML_BASE)?),
        _ => return None,
    })
}

/// A [`DType`]'s code: its position in [`DType::ALL`].
#[must_use]
pub fn dtype_code(dtype: DType) -> Option<u32> {
    DType::ALL
        .iter()
        .position(|&d| d == dtype)
        .and_then(|i| u32::try_from(i).ok())
}

/// The [`DType`] `code` names.
#[must_use]
pub fn dtype_of_code(code: u32) -> Option<DType> {
    DType::ALL.get(usize::try_from(code).ok()?).copied()
}

/// A [`StreamPriority`]'s code: 0 compute, 1 copy-in, 2 copy-out, 3 collective, 4 low.
#[must_use]
pub fn priority_code(prio: StreamPriority) -> Option<u32> {
    Some(match prio {
        StreamPriority::Compute => 0,
        StreamPriority::CopyIn => 1,
        StreamPriority::CopyOut => 2,
        StreamPriority::Collective => 3,
        StreamPriority::Low => 4,
    })
}

/// The [`StreamPriority`] `code` names.
#[must_use]
pub fn priority_of_code(code: u32) -> Option<StreamPriority> {
    Some(match code {
        0 => StreamPriority::Compute,
        1 => StreamPriority::CopyIn,
        2 => StreamPriority::CopyOut,
        3 => StreamPriority::Collective,
        4 => StreamPriority::Low,
        _ => return None,
    })
}

/// A [`MemKind`]'s code: 0 device, 1 host-mapped, 2 unified.
#[must_use]
pub fn mem_kind_code(kind: MemKind) -> Option<u32> {
    Some(match kind {
        MemKind::Device => 0,
        MemKind::HostMapped => 1,
        MemKind::Unified => 2,
    })
}

/// The [`MemKind`] `code` names.
#[must_use]
pub fn mem_kind_of_code(code: u32) -> Option<MemKind> {
    Some(match code {
        0 => MemKind::Device,
        1 => MemKind::HostMapped,
        2 => MemKind::Unified,
        _ => return None,
    })
}

/// A [`WakeKind`]'s code: 0 futex, 1 eventfd, 2 Windows event, 3 timeline value.
#[must_use]
pub fn wake_code(kind: WakeKind) -> Option<u32> {
    Some(match kind {
        WakeKind::Futex => 0,
        WakeKind::EventFd => 1,
        WakeKind::WinEvent => 2,
        WakeKind::TimelineValue => 3,
    })
}

/// The [`WakeKind`] `code` names.
#[must_use]
pub fn wake_of_code(code: u32) -> Option<WakeKind> {
    Some(match code {
        0 => WakeKind::Futex,
        1 => WakeKind::EventFd,
        2 => WakeKind::WinEvent,
        3 => WakeKind::TimelineValue,
        _ => return None,
    })
}

/// A [`WaitCmp`]'s code: 0 for `Eq`, 1 for `Geq`.
#[must_use]
pub fn cmp_code(cmp: WaitCmp) -> u32 {
    match cmp {
        WaitCmp::Eq => 0,
        WaitCmp::Geq => 1,
    }
}

/// The [`WaitCmp`] `code` names.
#[must_use]
pub fn cmp_of_code(code: u32) -> Option<WaitCmp> {
    match code {
        0 => Some(WaitCmp::Eq),
        1 => Some(WaitCmp::Geq),
        _ => None,
    }
}

// Records that are plain integers.

impl From<DeviceId> for SynDeviceId {
    fn from(id: DeviceId) -> Self {
        Self {
            node: id.node.0,
            ordinal: u32::from(id.ordinal),
        }
    }
}

impl SynDeviceId {
    /// The `DeviceId`, or `None` when the ordinal does not fit.
    #[must_use]
    pub fn to_device_id(self) -> Option<DeviceId> {
        Some(DeviceId {
            node: NodeId(self.node),
            ordinal: u16::try_from(self.ordinal).ok()?,
        })
    }
}

impl From<&OpenOptions> for SynOpenOptions {
    fn from(o: &OpenOptions) -> Self {
        Self {
            streams: [
                o.streams.compute,
                o.streams.copy_in,
                o.streams.copy_out,
                o.streams.collective,
            ],
            doorbell_slots: o.doorbell_slots,
            capture: u8::from(o.capture),
            deterministic: u8::from(o.deterministic),
            power: match o.power {
                PowerHint::Mains => 0,
                PowerHint::Battery => 1,
            },
            reserved: 0,
        }
    }
}

impl SynOpenOptions {
    /// The `OpenOptions`, or `None` for a power code this build does not know.
    #[must_use]
    pub fn to_options(&self) -> Option<OpenOptions> {
        Some(OpenOptions {
            streams: StreamPlan {
                compute: self.streams[0],
                copy_in: self.streams[1],
                copy_out: self.streams[2],
                collective: self.streams[3],
            },
            doorbell_slots: self.doorbell_slots,
            capture: self.capture != 0,
            deterministic: self.deterministic != 0,
            power: match self.power {
                0 => PowerHint::Mains,
                1 => PowerHint::Battery,
                _ => return None,
            },
        })
    }
}

impl From<&DeviceMemoryInfo> for SynMemoryInfo {
    fn from(m: &DeviceMemoryInfo) -> Self {
        Self {
            total: m.total.0,
            free: m.free.0,
            budget: m.budget.map_or(0, |b| b.0),
            host_visible: m.host_visible_device_local.map_or(0, |b| b.0),
            max_alloc: m.max_alloc.0,
            flags: u32::from(m.budget.is_some())
                | u32::from(m.host_visible_device_local.is_some()) << 1
                | u32::from(m.unified) << 2,
            reserved: 0,
        }
    }
}

impl From<&SynMemoryInfo> for DeviceMemoryInfo {
    fn from(m: &SynMemoryInfo) -> Self {
        Self {
            total: Bytes(m.total),
            free: Bytes(m.free),
            budget: (m.flags & 1 != 0).then_some(Bytes(m.budget)),
            host_visible_device_local: (m.flags & 2 != 0).then_some(Bytes(m.host_visible)),
            max_alloc: Bytes(m.max_alloc),
            unified: m.flags & 4 != 0,
        }
    }
}

impl SynBuffer {
    /// The wire form of `buf`, or `None` for a kind this build has no code for.
    #[must_use]
    pub fn of(buf: &DeviceBuffer) -> Option<Self> {
        Some(Self {
            addr: buf.addr,
            bytes: buf.bytes.0,
            handle: buf.handle,
            kind: mem_kind_code(buf.kind)?,
            reserved: 0,
        })
    }

    /// The `DeviceBuffer`, or `None` for an unknown kind.
    #[must_use]
    pub fn to_buffer(&self) -> Option<DeviceBuffer> {
        Some(DeviceBuffer {
            addr: self.addr,
            bytes: Bytes(self.bytes),
            kind: mem_kind_of_code(self.kind)?,
            handle: self.handle,
        })
    }
}

impl From<BufferSlice> for SynSlice {
    fn from(s: BufferSlice) -> Self {
        Self {
            buffer: s.buffer,
            offset: s.offset.0,
            bytes: s.bytes.0,
        }
    }
}

impl From<SynSlice> for BufferSlice {
    fn from(s: SynSlice) -> Self {
        Self {
            buffer: s.buffer,
            offset: Bytes(s.offset),
            bytes: Bytes(s.bytes),
        }
    }
}

impl From<&Segment> for SynSegment {
    fn from(s: &Segment) -> Self {
        Self {
            reservation: s.reservation,
            offset: s.offset.0,
            bytes: s.bytes.0,
            handle: s.handle,
        }
    }
}

impl From<&SynSegment> for Segment {
    fn from(s: &SynSegment) -> Self {
        Self {
            reservation: s.reservation,
            offset: Bytes(s.offset),
            bytes: Bytes(s.bytes),
            handle: s.handle,
        }
    }
}

impl From<HostRegion> for SynHostRegion {
    fn from(r: HostRegion) -> Self {
        Self {
            addr: r.addr,
            len: r.len.0,
        }
    }
}

impl From<SynHostRegion> for HostRegion {
    fn from(r: SynHostRegion) -> Self {
        Self {
            addr: r.addr,
            len: Bytes(r.len),
        }
    }
}

impl From<&Registration> for SynRegistration {
    fn from(r: &Registration) -> Self {
        Self {
            region: r.region.into(),
            handle: r.handle,
        }
    }
}

impl From<&SynRegistration> for Registration {
    fn from(r: &SynRegistration) -> Self {
        Self {
            region: r.region.into(),
            handle: r.handle,
        }
    }
}

impl From<&AllocRequest> for super::types::SynAllocRequest {
    fn from(r: &AllocRequest) -> Self {
        Self {
            bytes: r.bytes.0,
            align: r.align.0,
            // `AllocRequest::kind` is set by `MemKind`'s variants; an unknown one is refused by
            // the caller before this runs.
            kind: mem_kind_code(r.kind).unwrap_or(u32::MAX),
            reserve_only: u32::from(r.reserve_only),
        }
    }
}

impl super::types::SynAllocRequest {
    /// The `AllocRequest`, or `None` for an unknown kind.
    #[must_use]
    pub fn to_request(&self) -> Option<AllocRequest> {
        Some(AllocRequest {
            bytes: Bytes(self.bytes),
            kind: mem_kind_of_code(self.kind)?,
            align: Bytes(self.align),
            reserve_only: self.reserve_only != 0,
        })
    }
}

impl SynHealth {
    /// The wire form of `health`, or `None` for a variant this build has no code for.
    #[must_use]
    pub fn of(health: DeviceHealth) -> Option<Self> {
        Some(match health {
            DeviceHealth::Ok => Self { kind: 0, detail: 0 },
            DeviceHealth::Throttled { reason } => Self {
                kind: 1,
                detail: match reason {
                    ThrottleReason::Thermal => 0,
                    ThrottleReason::Power => 1,
                    ThrottleReason::Other => 2,
                },
            },
            DeviceHealth::Sticky { code } => Self {
                kind: 2,
                detail: u32::from(code.0),
            },
        })
    }

    /// The `DeviceHealth`, or `None` for an unknown code.
    #[must_use]
    pub fn to_health(&self) -> Option<DeviceHealth> {
        Some(match self.kind {
            0 => DeviceHealth::Ok,
            1 => DeviceHealth::Throttled {
                reason: match self.detail {
                    0 => ThrottleReason::Thermal,
                    1 => ThrottleReason::Power,
                    2 => ThrottleReason::Other,
                    _ => return None,
                },
            },
            2 => DeviceHealth::Sticky {
                code: ErrorCode(u16::try_from(self.detail).ok()?),
            },
            _ => return None,
        })
    }
}

impl SynTensorArg {
    /// The wire form of `t`, or `None` for a dtype this build has no code for.
    #[must_use]
    pub fn of(t: &TensorArg) -> Option<Self> {
        Some(Self {
            buf: t.buf.into(),
            dtype: dtype_code(t.dtype)?,
            reserved: 0,
            dims: t.dims,
            strides: t.strides,
        })
    }

    /// The `TensorArg`, or `None` for an unknown dtype.
    #[must_use]
    pub fn to_tensor(&self) -> Option<TensorArg> {
        Some(TensorArg {
            buf: self.buf.into(),
            dtype: dtype_of_code(self.dtype)?,
            dims: self.dims,
            strides: self.strides,
        })
    }
}

impl SynBufferView {
    /// The wire form of `v`, or `None` for a view kind this build has no code for.
    #[must_use]
    pub fn of(v: &BufferView) -> Option<Self> {
        let (kind, ordinal, node) = match v.kind {
            ViewKind::Host => (0, 0, [0; 32]),
            ViewKind::Device(d) => (1, u32::from(d.ordinal), d.node.0),
            ViewKind::Unified => (2, 0, [0; 32]),
            _ => return None,
        };
        Some(Self {
            kind,
            ordinal,
            node,
            addr: v.addr,
            len: v.len.0,
        })
    }

    /// The `BufferView`, or `None` for an unknown kind or an ordinal that does not fit.
    #[must_use]
    pub fn to_view(&self) -> Option<BufferView> {
        let kind = match self.kind {
            0 => ViewKind::Host,
            1 => ViewKind::Device(DeviceId {
                node: NodeId(self.node),
                ordinal: u16::try_from(self.ordinal).ok()?,
            }),
            2 => ViewKind::Unified,
            _ => return None,
        };
        Some(BufferView {
            kind,
            addr: self.addr,
            len: Bytes(self.len),
        })
    }
}

impl SynScalar {
    /// The wire form of `s`, or `None` for a variant this build has no code for.
    #[must_use]
    pub fn of(s: &Scalar) -> Option<Self> {
        Some(match *s {
            Scalar::F32(v) => Self {
                tag: 0,
                bits: v.to_bits(),
            },
            Scalar::I32(v) => Self {
                tag: 1,
                bits: v.cast_unsigned(),
            },
            Scalar::U32(v) => Self { tag: 2, bits: v },
        })
    }

    /// The `Scalar`, or `None` for an unknown tag.
    #[must_use]
    pub fn to_scalar(&self) -> Option<Scalar> {
        Some(match self.tag {
            0 => Scalar::F32(f32::from_bits(self.bits)),
            1 => Scalar::I32(self.bits.cast_signed()),
            2 => Scalar::U32(self.bits),
            _ => return None,
        })
    }
}

// Records that cross as JSON.

/// Encodes `value` as JSON.
///
/// # Errors
///
/// `Internal` if the value cannot be encoded.
pub fn to_json<T: Serialize>(value: &T) -> Result<Vec<u8>, Error> {
    serde_json::to_vec(value).map_err(|e| internal(format!("encoding a boundary record: {e}")))
}

/// Decodes a `T` from the JSON in `bytes`.
///
/// # Errors
///
/// `Internal` if the bytes are not a `T`.
pub fn from_json<T: for<'de> Deserialize<'de>>(bytes: &[u8]) -> Result<T, Error> {
    serde_json::from_slice(bytes).map_err(|e| internal(format!("decoding a boundary record: {e}")))
}

/// An `Internal` error: a bug in a provider, the loader or this ABI. RFC-0040 §13 numbers no
/// code for a boundary violation, so the code is 0; a caught panic is
/// [`ErrorCode::PROVIDER_PANICKED`] (see [`panicked`]).
#[must_use]
pub fn internal(message: String) -> Error {
    Error {
        class: ErrorClass::Internal,
        code: ErrorCode(0),
        message,
    }
}

/// The error a thunk returns after it caught a panic (RFC-0040 §13 code 16).
#[must_use]
pub fn panicked(message: String) -> Error {
    Error {
        class: ErrorClass::Internal,
        code: ErrorCode::PROVIDER_PANICKED,
        message,
    }
}

/// [`ProviderBuild`] as it crosses: the toolchain keys are owned strings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildWire {
    /// `ProviderBuild::version`.
    pub version: u32,
    /// `ProviderBuild::module`.
    pub module: Option<Blake3>,
    /// `ProviderBuild::toolchain`.
    pub toolchain: Vec<(String, String)>,
    /// `ProviderBuild::driver`.
    pub driver: Option<String>,
}

impl BuildWire {
    /// The wire form of `build`.
    #[must_use]
    pub fn of(build: &ProviderBuild) -> Self {
        Self {
            version: build.version,
            module: build.module,
            toolchain: build
                .toolchain
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
            driver: build.driver.clone(),
        }
    }

    /// The `ProviderBuild`. Its toolchain keys are `&'static str`, interned ([`intern`]), so
    /// each distinct key is allocated once for the process however many providers are loaded.
    #[must_use]
    pub fn into_build(self) -> ProviderBuild {
        ProviderBuild {
            version: self.version,
            module: self.module,
            toolchain: self
                .toolchain
                .into_iter()
                .map(|(k, v)| (intern(&k), v))
                .collect(),
            driver: self.driver,
        }
    }
}

/// [`Numerics`] as it crosses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NumericsWire {
    /// 0 F32, 1 I32 then F32, 2 F64.
    pub accumulate: u32,
    /// `max_block` of `I32ThenF32`, else 0.
    pub max_block: u32,
    /// 0 none, 1 BF16, 2 Q8 block 32, 3 Q8K 256, 4 FP8 dynamic, 5 MXFP8, 6 NVFP4.
    pub act_in: u32,
    /// `group` of `Fp8Dynamic`, else 0.
    pub group: u16,
    /// The output dtype's code.
    pub out: u32,
    /// `Numerics::softmax_f32`.
    pub softmax_f32: bool,
    /// `Numerics::state_f32`.
    pub state_f32: bool,
}

impl NumericsWire {
    fn of(n: &Numerics) -> Result<Self, Error> {
        let (accumulate, max_block) = match n.accumulate {
            AccumKind::F32 => (0, 0),
            AccumKind::I32ThenF32 { max_block } => (1, max_block),
            AccumKind::F64 => (2, 0),
            _ => {
                return Err(internal(
                    "an accumulation kind the ABI has no code for".into(),
                ));
            }
        };
        let (act_in, group) = match n.act_in {
            ActQuant::None => (0, 0),
            ActQuant::Bf16 => (1, 0),
            ActQuant::Q8Block32 => (2, 0),
            ActQuant::Q8K256 => (3, 0),
            ActQuant::Fp8Dynamic { group } => (4, group),
            ActQuant::Mxfp8 => (5, 0),
            ActQuant::Nvfp4 => (6, 0),
            _ => {
                return Err(internal(
                    "an activation quantisation the ABI has no code for".into(),
                ));
            }
        };
        Ok(Self {
            accumulate,
            max_block,
            act_in,
            group,
            out: dtype_code(n.out)
                .ok_or_else(|| internal("a dtype the ABI has no code for".into()))?,
            softmax_f32: n.softmax_f32,
            state_f32: n.state_f32,
        })
    }

    fn to_numerics(self) -> Result<Numerics, Error> {
        let unknown = |what: &str| internal(format!("an unknown {what} code"));
        Ok(Numerics {
            accumulate: match self.accumulate {
                0 => AccumKind::F32,
                1 => AccumKind::I32ThenF32 {
                    max_block: self.max_block,
                },
                2 => AccumKind::F64,
                _ => return Err(unknown("accumulation")),
            },
            act_in: match self.act_in {
                0 => ActQuant::None,
                1 => ActQuant::Bf16,
                2 => ActQuant::Q8Block32,
                3 => ActQuant::Q8K256,
                4 => ActQuant::Fp8Dynamic { group: self.group },
                5 => ActQuant::Mxfp8,
                6 => ActQuant::Nvfp4,
                _ => return Err(unknown("activation quantisation")),
            },
            out: dtype_of_code(self.out).ok_or_else(|| unknown("dtype"))?,
            softmax_f32: self.softmax_f32,
            state_f32: self.state_f32,
        })
    }
}

/// [`OpCaps`] as it crosses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapsWire {
    /// `OpCaps::workspace`.
    pub workspace: u64,
    /// `OpCaps::est`, in nanoseconds.
    pub est: Option<u64>,
    /// `OpCaps::layout`.
    pub layout: u16,
    /// `OpCaps::capture_safe`.
    pub capture_safe: bool,
    /// `OpCaps::deterministic`.
    pub deterministic: bool,
    /// `OpCaps::numerics`.
    pub numerics: NumericsWire,
    /// `OpCaps::kernel_hash`.
    pub kernel_hash: Blake3,
    /// `OpCaps::source_key`.
    pub source_key: Blake3,
}

impl CapsWire {
    /// The wire form of `caps`.
    ///
    /// # Errors
    ///
    /// `Internal` if `caps` holds a value this build has no code for.
    pub fn of(caps: &OpCaps) -> Result<Self, Error> {
        Ok(Self {
            workspace: caps.workspace.0,
            est: caps.est.map(|n| n.0),
            layout: caps.layout.0,
            capture_safe: caps.capture_safe,
            deterministic: caps.deterministic,
            numerics: NumericsWire::of(&caps.numerics)?,
            kernel_hash: caps.kernel_hash,
            source_key: caps.source_key,
        })
    }

    /// The `OpCaps`.
    ///
    /// # Errors
    ///
    /// `Internal` if a code is unknown to this build.
    pub fn into_caps(self) -> Result<OpCaps, Error> {
        Ok(OpCaps {
            workspace: Bytes(self.workspace),
            est: self.est.map(Nanos),
            layout: LayoutVersion(self.layout),
            capture_safe: self.capture_safe,
            deterministic: self.deterministic,
            numerics: self.numerics.to_numerics()?,
            kernel_hash: self.kernel_hash,
            source_key: self.source_key,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_tables_round_trip() {
        for backend in [
            Backend::Cuda,
            Backend::Hip,
            Backend::Metal,
            Backend::Vulkan,
            Backend::Cpu,
            Backend::Ggml(GgmlBackend::Sycl),
            Backend::Ggml(GgmlBackend::Cpu),
        ] {
            assert_eq!(
                backend_of_code(backend_code(backend).unwrap()),
                Some(backend)
            );
        }
        for domain in [
            MemoryDomain::Cuda,
            MemoryDomain::Host,
            MemoryDomain::Ggml(GgmlBackend::Vulkan),
        ] {
            assert_eq!(domain_of_code(domain_code(domain).unwrap()), Some(domain));
        }
        for dtype in DType::ALL {
            assert_eq!(dtype_of_code(dtype_code(dtype).unwrap()), Some(dtype));
        }
        for prio in [
            StreamPriority::Compute,
            StreamPriority::CopyIn,
            StreamPriority::Low,
        ] {
            assert_eq!(priority_of_code(priority_code(prio).unwrap()), Some(prio));
        }
        assert_eq!(backend_of_code(0), None);
        assert_eq!(backend_of_code(16 + 99), None);
        assert_eq!(dtype_of_code(10_000), None);
    }

    #[test]
    fn plain_records_round_trip() {
        let options = OpenOptions {
            streams: StreamPlan {
                compute: 2,
                copy_in: 1,
                copy_out: 1,
                collective: 0,
            },
            doorbell_slots: 64,
            capture: true,
            deterministic: false,
            power: PowerHint::Battery,
        };
        assert_eq!(SynOpenOptions::from(&options).to_options(), Some(options));

        let info = DeviceMemoryInfo {
            total: Bytes(100),
            free: Bytes(40),
            budget: Some(Bytes(90)),
            host_visible_device_local: None,
            max_alloc: Bytes(50),
            unified: true,
        };
        assert_eq!(DeviceMemoryInfo::from(&SynMemoryInfo::from(&info)), info);

        let tensor = TensorArg::packed(
            BufferSlice {
                buffer: 7,
                offset: Bytes(8),
                bytes: Bytes(64),
            },
            DType::F16,
            [1, 2, 3, 4],
        )
        .unwrap();
        assert_eq!(SynTensorArg::of(&tensor).unwrap().to_tensor(), Some(tensor));

        for view in [
            BufferView {
                kind: ViewKind::Host,
                addr: 5,
                len: Bytes(6),
            },
            BufferView {
                kind: ViewKind::Device(DeviceId {
                    node: NodeId([9; 32]),
                    ordinal: 3,
                }),
                addr: 1,
                len: Bytes(2),
            },
        ] {
            assert_eq!(SynBufferView::of(&view).unwrap().to_view(), Some(view));
        }
        for scalar in [Scalar::F32(-1.5), Scalar::I32(-7), Scalar::U32(u32::MAX)] {
            assert_eq!(SynScalar::of(&scalar).unwrap().to_scalar(), Some(scalar));
        }
        for health in [
            DeviceHealth::Ok,
            DeviceHealth::Throttled {
                reason: ThrottleReason::Power,
            },
            DeviceHealth::Sticky {
                code: ErrorCode::PROVIDER_PANICKED,
            },
        ] {
            assert_eq!(SynHealth::of(health).unwrap().to_health(), Some(health));
        }
    }

    #[test]
    fn caps_round_trip_through_json() {
        let caps = OpCaps {
            workspace: Bytes(4096),
            est: Some(Nanos(1200)),
            layout: LayoutVersion(3),
            capture_safe: true,
            deterministic: true,
            numerics: Numerics {
                accumulate: AccumKind::I32ThenF32 { max_block: 256 },
                act_in: ActQuant::Fp8Dynamic { group: 128 },
                out: DType::BF16,
                softmax_f32: true,
                state_f32: false,
            },
            kernel_hash: Blake3([1; 32]),
            source_key: Blake3([2; 32]),
        };
        let bytes = to_json(&CapsWire::of(&caps).unwrap()).unwrap();
        let back: CapsWire = from_json(&bytes).unwrap();
        assert_eq!(back.into_caps().unwrap(), caps);
    }
}
