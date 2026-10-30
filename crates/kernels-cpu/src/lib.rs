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
//! - [`sme2`]: the SME2 assembly unit (COMP-E3-T07): outer-product GEMM tiles in
//!   `src/sme2/*.S`, reachable only through an [`Sme2`] token that an `Arm92Sme2` table on an
//!   SME2 host holds (ADR-0402 item 3).
//!
//! The other kernels come with later tasks, starting with COMP-E3-T03.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]
// `core::Error` is RFC-0001 §5.12's struct with its fields inline: 344 B [measured: `size_of`,
// debug build, x86-64 Linux, `core` at ce2044ae, unchanged by COMP-E13-T15], over clippy's
// 128-byte limit for an `Err`. An error is the cold path, and the Ok value of a `Result` is
// written in place.
#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

mod device;
pub mod dispatch;
pub mod doorbell;
pub mod isa;
pub mod provider;
pub mod sme2;
pub mod stream;
pub mod wake;

pub use dispatch::{DispatchTable, KernelFn};
pub use doorbell::{Doorbells, FLAGS_PER_PAGE};
pub use isa::{
    AmxPermission, ArmInputs, Detection, IsaInputs, KillSwitches, MacSysctls, X86Feature,
    X86Features, X86Inputs, detect, inputs, select, table_for_class,
};
pub use provider::CpuProvider;
pub use sme2::Sme2;
pub use stream::{Parks, Run, STREAM_CAPACITY, StreamRunner};
pub use wake::{CpuWaker, Parked};
