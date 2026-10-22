// SPDX-License-Identifier: MIT OR Apache-2.0

//! `testkit` in this fixture, with rule 8's module-scoped exception in the shape the check
//! accepts: `deny` at the root and one allowed `GlobalAlloc` impl in `src/guards.rs`.

#![deny(unsafe_code)]

pub mod guards;
