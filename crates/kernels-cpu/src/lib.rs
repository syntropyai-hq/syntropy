// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` for CPUs, written in Rust with `std::arch`.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0040 §7.3. Built so far:
//!
//! - [`isa`]: the ISA level a CPU device runs at, decided by one pure function of recorded
//!   detection inputs and the `features.disable` kill switches, and the thin OS shim that
//!   records those inputs on the live host (ADR-0405 decision 3).
//! - [`dispatch`]: the per-op function-pointer table for one level, which the provider's
//!   `open` (COMP-E3-T02) installs after [`isa::table_for_class`] checks the class.
//!
//! The provider itself and the kernels come with later tasks.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]

pub mod dispatch;
pub mod isa;

pub use dispatch::{DispatchTable, KernelFn};
pub use isa::{
    AmxPermission, ArmInputs, Detection, IsaInputs, KillSwitches, MacSysctls, X86Feature,
    X86Features, X86Inputs, detect, inputs, select, table_for_class,
};
