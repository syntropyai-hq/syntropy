// SPDX-License-Identifier: MIT OR Apache-2.0

//! The table-delta ring a device's dispatcher posts slot-table, copy-eligibility and KV
//! block-table changes through, and the entries the first node of every captured step applies
//! (RFC-0023 §5.4; RFC-0040 §3.4; ADR-0404 item 5).
//!
//! - [`mirror`] holds the `#[repr(C)]` mirrors of the ring header and its entries,
//!   [`DeltaRingHeader`] and [`TableDelta`], generated from
//!   `include/syntropy/table_delta_ring.h`, their one definition.
//! - [`ring`] sizes a ring for one capacity ([`DeltaRingLayout`]), posts and reads entries in
//!   its bytes, and writes and reads the device's echo.
//!
//! The dispatcher is the ring's only producer: it posts entries and release-publishes the head,
//! and a step applies the entries up to the `delta_head` its descriptor names, never the live
//! head, echoing the applied position and step (RFC-0040 §3.4, §3.5). `kernels` forbids
//! `unsafe`, so the ring's bytes are handled as slices here; the provider crates map them onto
//! host-mapped memory.

#[rustfmt::skip]
pub mod mirror;
pub mod ring;
#[cfg(test)]
mod tests;

pub use mirror::{DeltaRingHeader, TableDelta};
pub use ring::{DeltaRingLayout, RingError, applied, head};
