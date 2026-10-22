// SPDX-License-Identifier: MIT OR Apache-2.0

//! The deterministic simulator (RFC-0060 §8, ADR-0600).
//!
//! So far it holds the simulator's sources of time and randomness (QUAL-E3-T02): [`SimClock`],
//! virtual time advanced only by its owner, and [`SimEntropy`], a seeded ChaCha20 stream. They
//! implement `syntropy_core`'s [`Clock`](syntropy_core::Clock) and
//! [`Entropy`](syntropy_core::Entropy), which the code under rule T1 is given (RFC-0060 §2.1).
//! The world, the fabric, the fakes and the checkers are planned.

mod clock;
mod entropy;

pub use clock::{SIM_UNIX_MS_ORIGIN, SimClock, SimClockHandle};
pub use entropy::SimEntropy;
