// SPDX-License-Identifier: MIT OR Apache-2.0

//! The step descriptor, its parity mirrors and the handshake that guards their reuse
//! (RFC-0040 §3.5, §3.8, §4.2 rule 5; ADR-0404 items 1, 2 and 6).
//!
//! Every value that changes from step to step lives in the step descriptor, so one captured
//! graph serves every step. The host writes a descriptor into one of two pinned mirrors; the
//! graph instance of that mirror's parity copies it to the device, then raises the parity's
//! descriptor-consumed flag `DC[p]`, and at its end the step-done flag `O[p]`.
//!
//! - [`mirror`] holds the `#[repr(C)]` mirrors [`StepDesc`] and [`SamplerState`], generated
//!   from `include/syntropy/step_desc.h`, the layouts' one definition.
//! - [`desc`] sizes a descriptor from its limits ([`StepDescLayout`]) and writes and reads
//!   whole descriptors ([`StepDescriptor`]) in mirror bytes, with the KV write rule of
//!   `row_kv_write_from`.
//! - [`parity`] holds the host's side of the per-parity handshake ([`ParityHandshake`]).
//!
//! `kernels` forbids `unsafe`, so the mirrors are byte slices and the flags `AtomicU32`s here;
//! the provider crates map them onto pinned, host-mapped memory.

pub mod desc;
// Generated: the tests compare it with the generator's output, so rustfmt leaves it alone.
#[rustfmt::skip]
pub mod mirror;
pub mod parity;

pub use desc::{
    RowEntry, SeqEntry, StepDescError, StepDescLayout, StepDescLimits, StepDescriptor, StepFlags,
    TapFields, kv_writes,
};
pub use mirror::{SamplerState, StepDesc};
pub use parity::{HandshakeError, Parity, ParityFlags, ParityHandshake};

#[cfg(test)]
mod cgen;
#[cfg(test)]
mod tests;
