// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` trait and its supporting types (RFC-0001 §5.7; RFC-0040 §2.1, §2.2).
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0040. It depends only on `core`. The op
//! registry, per-device op selection and the F32 reference semantics of every op come with
//! later tasks; the op types the trait names are placeholders until then ([`op`]).

#![forbid(unsafe_code)]

pub mod device;
pub mod error;
pub mod memory;
pub mod op;
pub mod provider;
pub mod stream;

pub use device::{
    CpuIsa, DeviceClass, DeviceCtx, DeviceFeatures, DeviceHealth, DeviceMemoryInfo, MemoryDomain,
    OpenOptions, PowerHint, ProviderBuild, ProviderDevice, ProviderId, StreamPlan, ThrottleReason,
};
pub use error::{Error, ErrorCode, unsupported};
pub use memory::{
    AllocRequest, BufferSlice, DeviceBuffer, HostRegFlags, HostRegion, HostRegistrar, MemKind,
    Registration, Segment,
};
pub use op::{LayoutVersion, OpArgs, OpCaps, OpKey, OpRegistry, PreparedOp, ShapeSig};
pub use provider::{CollectiveOps, KernelProvider};
pub use stream::{
    CapturedGraph, Event, HostFlag, HostWaker, Stream, StreamPriority, WaitCmp, WakeKind,
};
