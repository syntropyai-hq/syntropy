// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` trait and its supporting types (RFC-0001 §5.7; RFC-0040 §2.1, §2.2).
//!
//! It also holds the tolerance tables of the exactness contract ([`tolerances`], RFC-0042 §6),
//! and the accumulation contract: the static check of each kernel's `Numerics` and the
//! dynamic detector ([`numerics`], RFC-0042 §4).
//!
//! Layer 1 in RFC-0001 §4.1, designed in RFC-0040, with the numerics of RFC-0042. Its only
//! workspace dependency is `core`. It holds the op registry and the op types ([`op`],
//! RFC-0040 §3.1, §3.2), and the reference numerics' modes ([`reference`](mod@reference),
//! RFC-0060 §4.1). Per-device op selection, the argument types and the F32 reference semantics of
//! every op come with later tasks.
//!
//! # `unsafe`
//!
//! This crate declares, and never implements or calls, three `unsafe` trait methods:
//! [`KernelProvider::register_host`], [`KernelProvider::import`] and
//! [`HostRegistrar::register`]. Each takes a host range the provider then reads and writes by
//! address, so its caller must prove the range is valid. That needs an exception to RFC-0001
//! §4.2 rule 8, which §18.36 records for the maintainers to decide. The root denies
//! `unsafe_code` rather than forbidding it, and each declaration carries the only
//! `#[allow(unsafe_code)]`. `cargo xtask check-deps` refuses any other `unsafe` in the crate's
//! sources.

#![deny(unsafe_code)]

pub mod device;
pub mod error;
pub mod memory;
pub mod numerics;
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
