// SPDX-License-Identifier: MIT OR Apache-2.0

//! The core side of the provider-library ABI: module loading, the `ModuleProvider` proxy, and
//! (with later tasks) the ABI and build-ID checks and the driver census.
//!
//! Layer 2 in RFC-0001 §4.1, designed in RFC-0040 §10.4, §10.5. A provider library is a `cdylib`
//! that exports one symbol (`syntropy_provider_entry_v1`) and returns a table of function
//! pointers, one per `KernelProvider` method ([`syntropy_kernels::abi`]). This crate opens the
//! library ([`Module::load`]) and wraps each table into a [`ModuleProvider`], which implements
//! `KernelProvider` for the core by calling through the pointers.
//!
//! The boundary keeps the rules of RFC-0040 §10.4:
//!
//! - Only `#[repr(C)]` records, pointers with lengths and `u64` handles cross. Nothing the
//!   module allocates reaches the core: results land in buffers the proxy supplies, and a
//!   failure's message is fetched with `last_error`.
//! - A failed call returns the error the module reported. A module that caught a panic reports
//!   `Internal` (code 16) and becomes `Sticky`; [`ModuleProvider::fault`] keeps the first
//!   `Internal` error for the methods whose signature cannot carry one (`free`, `query`,
//!   `repack`, `repacked_len`, `elapsed`, `host_registrar`), and `health` reports `Sticky`.
//! - A `DeviceCtx` is not `Send`, so a context is used on the thread that opened it, and the
//!   module refuses a handle used anywhere else.
//! - A loaded library is never unloaded.
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl
//! carries a `// SAFETY:` comment that states its invariant (ADR-0001 item 4). The platform
//! loader calls are in `sys`; the rest is the proxy over the module's tables.

#![deny(clippy::undocumented_unsafe_blocks)]
// `core::Error` is RFC-0001 §5.12's struct with its fields inline: 344 B [measured: `size_of`,
// debug build, x86-64 Linux, `core` at ce2044ae, unchanged by COMP-E13-T15], over clippy's
// 128-byte limit for an `Err`. An error is the cold path, and the Ok value of a `Result` is
// written in place.
#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

mod host;
mod module;
mod provider;
mod sys;

pub use host::HostServices;
pub use module::{Module, ModuleError};
pub use provider::ModuleProvider;
