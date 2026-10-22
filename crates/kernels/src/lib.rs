// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` trait and its supporting types (RFC-0001 §5.7; RFC-0040 §2.1, §2.2).
//!
//! It also holds the tolerance tables of the exactness contract ([`tolerances`], RFC-0042 §6).
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0040, with the numerics of RFC-0042. Its only
//! workspace dependency is `core`. It holds the op registry and the op types ([`op`],
//! RFC-0040 §3.1, §3.2), and the reference numerics' modes ([`reference`](mod@reference),
//! RFC-0060 §4.1). Per-device op selection, the argument types and the F32 reference semantics of
//! every op come with later tasks.

#![forbid(unsafe_code)]

pub mod device;
pub mod error;
pub mod memory;
pub mod op;
pub mod provider;
pub mod reference;
pub mod stream;
pub mod tolerances;

pub use device::{
    CpuIsa, DeviceClass, DeviceCtx, DeviceFeatures, DeviceHealth, DeviceMemoryInfo, MemoryDomain,
    OpenOptions, PowerHint, ProviderBuild, ProviderDevice, ProviderId, StreamPlan, ThrottleReason,
};
pub use error::{Error, ErrorCode, unsupported};
pub use memory::{
    AllocRequest, BufferSlice, DeviceBuffer, HostRegFlags, HostRegion, HostRegistrar, MemKind,
    Registration, Segment,
};
pub use op::{
    ActKind, AttnKind, Fusions, GatedAct, IndexKind, LayoutVersion, OpArgs, OpCaps, OpKey, OpKind,
    OpRegistry, PREFILL_LADDER, PoolKind, PreparedOp, RegistryError, RopeKind, RouteNorm, ScanKind,
    ShapeClass, ShapeSig, bucket_set,
};
pub use provider::{CollectiveOps, KernelProvider};
pub use reference::{RefDevice, RefMode};
pub use stream::{
    CapturedGraph, Event, HostFlag, HostWaker, Stream, StreamPriority, WaitCmp, WakeKind,
};
