// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulated fabric, the deterministic simulator, the fake `KernelProvider`, `RefExecutor`, the
//! parity harness and golden traces.
//!
//! A dev-only crate in RFC-0001 §4.1, designed in RFC-0060: no production crate depends on it
//! (RFC-0001 §4.2 rule 9). So far it holds the hot-path guards of RFC-0060 §3.5,
//! [`guards::CountingAlloc`] and [`guards::alloc_count_this_thread`], the simulator's clock,
//! entropy and world core, in [`sim`], and the stubs of the reference interpreter's contract
//! ([`refexec`], RFC-0060 §4.1). The rest is planned.
//!
//! `unsafe` is denied in this crate, not forbidden, for one reason: a global allocator is an
//! `unsafe impl GlobalAlloc`. RFC-0001 §4.2 rule 8 (note 2026-10-22) lets [`guards`] hold that one
//! impl, which delegates to `std::alloc::System`, under the crate's only lint override. The
//! `check-deps` placement check refuses any other `unsafe` here.

#![deny(unsafe_code)]
#![deny(clippy::undocumented_unsafe_blocks)]

pub mod fuzz;
pub mod guards;
pub mod refexec;
pub mod sim;
