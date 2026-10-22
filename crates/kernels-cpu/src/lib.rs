// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` for CPUs, written in Rust with `std::arch`.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0040 §7.3. Built so far:
//!
//! - [`isa`]: the ISA level a CPU device runs at, decided by one pure function of recorded
//!   detection inputs and the `features.disable` kill switches, and the thin OS shim that
//!   records those inputs on the live host (ADR-0405 decision 3).
//! - [`dispatch`]: the per-op function-pointer table for one level, which the provider's
//!   `open` installs after [`isa::table_for_class`] checks the class.
//! - [`provider`]: [`CpuProvider`], the provider's shell (COMP-E3-T02): one device per NUMA
//!   domain, host memory, streams as work queues ([`stream`]), events as atomics, doorbells
//!   ([`doorbell`]), the device-triggered wake ([`wake`]), op-list capture and replay, and
//!   health. The private `device` module holds an open device's state.
//!
//! The kernels come with later tasks, starting with COMP-E3-T03.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]

mod device;
pub mod dispatch;
pub mod doorbell;
pub mod isa;
pub mod provider;
pub mod stream;
pub mod wake;

pub use dispatch::{DispatchTable, KernelFn};
pub use doorbell::{Doorbells, FLAGS_PER_PAGE};
pub use isa::{
    AmxPermission, ArmInputs, Detection, IsaInputs, KillSwitches, MacSysctls, X86Feature,
    X86Features, X86Inputs, detect, inputs, select, table_for_class,
};
pub use provider::CpuProvider;
pub use stream::{Parks, Run, STREAM_CAPACITY, StreamRunner};
pub use wake::{CpuWaker, Parked};
