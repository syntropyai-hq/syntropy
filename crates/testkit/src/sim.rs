// SPDX-License-Identifier: MIT OR Apache-2.0

//! The deterministic simulator (RFC-0060 §8, ADR-0600).
//!
//! So far it holds the simulator's sources of time and randomness (QUAL-E3-T02): [`SimClock`],
//! virtual time advanced only by its owner, and [`SimEntropy`], a seeded ChaCha20 stream. They
//! implement `syntropy_core`'s [`Clock`](syntropy_core::Clock) and
//! [`Entropy`](syntropy_core::Entropy), which the code under rule T1 is given (RFC-0060 §2.1).
//! [`SimTicks`] counts a `SimClock`'s virtual time as a
//! [`TickSource`](syntropy_core::TickSource), for hot-thread code in `exec` and `obs`
//! (QUAL-E3-T17). It also holds the world's core (QUAL-E8-T01): [`SimWorld`] and its event loop,
//! the [`Checker`] trait, and the [`TraceHasher`]. The scenario file and its device and drive
//! presets are [`scenario`] and [`presets`] (QUAL-E8-T03). The fabric, the fakes and the cluster
//! model are planned.

mod clock;
mod entropy;
pub mod presets;
pub mod scenario;
mod ticks;
mod trace;
mod world;

pub use clock::{SIM_UNIX_MS_ORIGIN, SimClock, SimClockHandle};
pub use entropy::SimEntropy;
pub use ticks::SimTicks;
pub use trace::{Hex, TraceEvent, TraceHasher};
pub use world::{Checker, Env, Found, Model, RunOutcome, Schedule, SimWorld, Violation};
