// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit newtypes for sizes and times.
//!
//! RFC-0001 §5.1 declares [`Bytes`] and [`Nanos`] in `syntropy-core`, which does not have them
//! yet. They are defined here so the I/O interface never takes a bare integer for a size or a
//! time (RFC-0001 §5.1). When `syntropy-core` gains them, these definitions become re-exports of
//! the core types, and paths through this crate keep working.

/// A size in bytes (RFC-0001 §5.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bytes(pub u64);

/// A duration in nanoseconds (RFC-0001 §5.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Nanos(pub u64);
