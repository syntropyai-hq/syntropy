// SPDX-License-Identifier: MIT OR Apache-2.0

//! The provider-library ABI, version 1 (RFC-0040 §10.4; ADR-0400 decision 4).
//!
//! A provider library is a `cdylib` that exports one symbol, [`ENTRY_SYMBOL`]. The symbol takes
//! the core's [`SynHostApi`] and returns a [`SynModule`]: one [`SynProviderVtbl`] per provider,
//! with one function pointer per `KernelProvider` method. This module holds the module side:
//!
//! - [`types`]: the `#[repr(C)]` tables and records that cross the boundary;
//! - [`status`]: [`SynStatus`] and its mapping to and from [`Error`](crate::Error);
//! - [`wire`]: the codes and the JSON records for the values that are not plain integers;
//! - [`guest`]: the safe, generic bodies of the thunks, with the panic guard, the sticky flag,
//!   the per-thread `last_error` store and the context table;
//! - [`export!`](crate::abi::export), which expands the thunks and the entry symbol in the
//!   provider's own `kernels-*` crate.
//!
//! This crate is `#![forbid(unsafe_code)]`, so every raw-pointer dereference sits in the
//! expansion of `export!`, inside the provider crate, and every one of them carries a `SAFETY`
//! comment. The core side, which calls through the tables, is `syntropy-kernels-loader`.
//!
//! # What crosses the boundary
//!
//! Only `#[repr(C)]` structs, raw pointers with lengths and `u64` handles (RFC-0040 §10.4).
//! Records that are not plain integers, such as the device list, an `OpKey` or an `OpCaps`,
//! cross as UTF-8 JSON in a pointer and a length. Both sides are the same build in a release
//! (the build ID is checked at load), and a developer build that mixes compatible minors reads
//! JSON by field name, so a minor version can append fields. The per-step calls (streams,
//! events, copies, `launch`) use plain structs.
//!
//! Each side frees what it allocated. Nothing the module allocates is handed to the core: the
//! module writes into a buffer the core supplies ([`SynOutBuf`]), and the core retries with a
//! larger buffer if the module reports that the data did not fit.

pub mod guest;
pub mod status;
pub mod types;
pub mod wire;

mod export;

use syntropy_core::Domain;

#[doc(inline)]
pub use crate::__abi_export as export;

pub use status::{SynStatus, status_of_error};
pub use types::{
    SynAllocRequest, SynBuffer, SynBufferView, SynDeviceId, SynHealth, SynHostApi, SynHostRegion,
    SynMemoryInfo, SynModule, SynOpArgs, SynOpenOptions, SynOutBuf, SynPreparedOp, SynProviderVtbl,
    SynRegistrarVtbl, SynRegistration, SynScalar, SynSegment, SynSlice, SynSlicePair, SynTensorArg,
};

/// The ABI major version. It is part of the entry symbol's name, so a core never finds the
/// entry point of a library of another major version.
pub const ABI_MAJOR: u32 = 1;

/// The ABI minor version: a minor version only appends fields and vtable entries.
pub const ABI_MINOR: u32 = 0;

/// The first eight bytes of every [`SynHeader`](types::SynHeader).
pub const MAGIC: [u8; 8] = *b"SYNPROV1";

/// The one symbol a provider library exports.
pub const ENTRY_SYMBOL: &str = "syntropy_provider_entry_v1";

/// The ELF version node the Linux version script exports the entry symbol under:
/// `ci/link/provider.map`, which `ci/link/provider-cc` passes to the
/// linker in place of rustc's anonymous script (COMP-E9-T19).
pub const VERSION_NODE: &str = "SYNTROPY_PROVIDER_1";

/// The signature of the entry symbol.
pub type EntryFn = unsafe extern "C" fn(host: *const SynHostApi) -> *const SynModule;

/// The `Kernels` domain's number in a [`SynStatus`] and a [`SynHealth`].
pub const DOMAIN_KERNELS: u32 = 1;

/// Every error domain's number on the boundary, in a [`SynStatus`] and a `Sticky`
/// [`SynHealth`] (RFC-0040 §10.4, note of 2026-10-30). The table only grows: a domain keeps its
/// number, and a domain added to `core` takes the next one. `Kernels` keeps the 1 the ABI
/// published first. `tests/abi_domains.rs` fails unless every `Domain` has a number here.
pub const DOMAIN_CODES: [(Domain, u32); 8] = [
    (Domain::Kernels, DOMAIN_KERNELS),
    (Domain::Cluster, 2),
    (Domain::Net, 3),
    (Domain::Dist, 4),
    (Domain::Recovery, 5),
    (Domain::Security, 6),
    (Domain::Exec, 7),
    (Domain::Tier, 8),
];

/// The boundary number of `domain`, or `None` for a domain this build has no number for.
#[must_use]
pub fn domain_code(domain: Domain) -> Option<u32> {
    DOMAIN_CODES
        .iter()
        .find(|(d, _)| *d == domain)
        .map(|&(_, n)| n)
}

/// The domain numbered `code`, or `None` for a number this build does not know.
#[must_use]
pub fn domain_of_code(code: u32) -> Option<Domain> {
    DOMAIN_CODES
        .iter()
        .find(|(_, n)| *n == code)
        .map(|&(d, _)| d)
}
