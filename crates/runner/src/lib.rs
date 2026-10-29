// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner process: its lifecycle state machine and the wiring of exec, sched, tier, io, net and
//! the kernel providers.
//!
//! Layer 5 in RFC-0001 §4.1, designed in RFC-0001 §3. Implemented so far: the lifecycle machine of
//! RFC-0001 §3.3 ([`lifecycle`]), and the runner process's command loop, heartbeat and watch on the
//! daemon ([`process`]).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod lifecycle;
pub mod process;
