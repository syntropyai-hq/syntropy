// SPDX-License-Identifier: MIT OR Apache-2.0

//! The deterministic simulator (RFC-0060 §8, ADR-0600).
//!
//! So far it holds the simulator's sources of time and randomness (QUAL-E3-T02): [`SimClock`],
//! virtual time advanced only by its owner, and [`SimEntropy`], a seeded ChaCha20 stream. They
//! implement `syntropy_core`'s [`Clock`](syntropy_core::Clock) and
//! [`Entropy`](syntropy_core::Entropy), which the code under rule T1 is given (RFC-0060 §2.1).
//! It also holds the world's core (QUAL-E8-T01): [`SimWorld`] and its event loop, the
//! [`Checker`] trait, and the [`TraceHasher`]; and the simulated fabric (QUAL-E8-T02):
//! [`SimFabric`], its [`LinkModel`]s and their [`Preset`]s, and the delay distributions
//! [`Dist`]. The fakes and the cluster model are planned.

mod clock;
mod dist;
mod entropy;
mod fabric;
mod trace;
mod world;

pub use clock::{SIM_UNIX_MS_ORIGIN, SimClock, SimClockHandle};
pub use dist::Dist;
pub use entropy::SimEntropy;
pub use fabric::{
    INITIAL_RTT, LinkModel, MAX_ACK_DELAY, PTO_GRANULARITY, Preset, RttEstimate, Sent, SimFabric,
    Split,
};
pub use trace::{Hex, TraceEvent, TraceHasher};
pub use world::{Checker, Env, Found, Model, RunOutcome, Schedule, SimWorld, Violation};
