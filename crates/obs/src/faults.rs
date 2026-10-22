// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fault points: named places where a test build injects a failure (RFC-0060 §9.2;
//! RFC-0001 §4.2 rule 10, which adopted RFC-0060's amendment A6).
//!
//! A call site names its point, `<crate>.<component>.<operation>`, and handles the effect the
//! point returns:
//!
//! ```
//! use syntropy_obs::faults::FaultEffect;
//!
//! fn submit() -> Result<(), &'static str> {
//!     match syntropy_obs::fault_point!("io.ring.submit") {
//!         FaultEffect::Error(_) => return Err("injected"),
//!         _ => {}
//!     }
//!     Ok(())
//! }
//! assert_eq!(submit(), Ok(()));
//! ```
//!
//! **Without the `fault-injection` feature** the macro expands to [`FaultEffect::None`] and
//! nothing else: no call, no state and no symbol. Only [`FaultEffect`] and [`Ctx`], plain types,
//! exist. Release artefacts are built without the feature. `cargo xtask check-deps` rule 10
//! checks that, and also reads the core binary's symbol table for any symbol of the
//! feature-only module `syntropy_obs::faults::inject`.
//!
//! **With the feature**, every point calls `faults::check`:
//! - While a `FaultPlan` is installed, it counts the hit. It fires the first rule whose point,
//!   filter and trigger all match, and counts the fire.
//! - It applies `Delay`, `Panic` and `Hang` itself and returns the effect. So the call site
//!   handles `Error` and `Corrupt`, and may ignore the rest.
//! - A plan comes from a TOML file named by `SYNTROPY_FAULT_PLAN` at process start
//!   (`faults::load_from_env`), or from a harness through `faults::install`. Pushing a plan over
//!   the daemon–runner IPC channel needs a message RFC-0001 §3.5 does not define yet: DIST-E1-T13
//!   adds it, and its handler calls `faults::install`.
//! - `faults::counters` and `faults::fired` report each point's hits and fires.
//!
//! **Determinism** (RFC-0060 §2.1, rule T1). `After` and `Window` read time only through the
//! [`Clock`](syntropy_core::Clock) passed to `install`. A `Probability` rule carries its seed,
//! which a harness draws from its [`Entropy`](syntropy_core::Entropy) or writes in the plan.
//! Whether the rule's k-th matching hit fires is a fixed function of (seed, point, k). So a seed
//! replays the same sequence of fires, and a run whose hits arrive in the same order replays
//! exactly. There is no unseeded source.

use serde::{Deserialize, Serialize};
use syntropy_core::{DeviceId, ErrorClass, LinkId, Nanos, NodeId};

/// What a fault point does when it fires (RFC-0060 §9.2).
///
/// In a plan's TOML, a unit variant is a string (`effect = "panic"`) and the others are tables
/// (`effect = { error = "Device" }`, `effect = { delay = 5000000 }`,
/// `effect = { corrupt = { bytes = 4 } }`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultEffect {
    /// Nothing happens: the point did not fire, or the build has no fault points.
    #[default]
    None,
    /// The call site fails with an error of this class.
    Error(ErrorClass),
    /// The point sleeps this long, then continues.
    Delay(Nanos),
    /// The call site corrupts this many bytes of its data.
    Corrupt {
        /// How many bytes to corrupt.
        bytes: u32,
    },
    /// The point panics.
    Panic,
    /// The point blocks its thread forever.
    Hang,
}

/// Where a fault point was reached, for a rule's `CtxFilter` (RFC-0060 §9.2).
///
/// `sequence` arrives with `core`'s `SequenceId` (RFC-0001 §5.13).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ctx {
    /// The node.
    pub node: Option<NodeId>,
    /// The device.
    pub device: Option<DeviceId>,
    /// The link.
    pub link: Option<LinkId>,
}

/// Names a fault point and returns its [`FaultEffect`] (RFC-0060 §9.2). The context is the
/// thread's current [`Ctx`] unless one is given.
#[cfg(feature = "fault-injection")]
#[macro_export]
macro_rules! fault_point {
    ($name:literal) => {
        $crate::faults::check($name, &$crate::faults::Ctx::here())
    };
    ($name:literal, $ctx:expr) => {
        $crate::faults::check($name, &$ctx)
    };
}

/// Names a fault point. Without the `fault-injection` feature it is always
/// [`FaultEffect::None`], and a given context expression is type-checked but never evaluated.
#[cfg(not(feature = "fault-injection"))]
#[macro_export]
macro_rules! fault_point {
    ($name:literal) => {
        $crate::faults::FaultEffect::None
    };
    ($name:literal, $ctx:expr) => {{
        let _ = || {
            let _ = &$ctx;
        };
        $crate::faults::FaultEffect::None
    }};
}

#[cfg(feature = "fault-injection")]
mod inject;

#[cfg(feature = "fault-injection")]
pub use inject::{
    CtxFilter, CtxGuard, FaultPlan, FaultRule, PLAN_ENV, PlanError, PointCounts, Trigger, check,
    clear, counters, fired, install, load_from_env, load_from_path, reset_counters,
};

#[cfg(test)]
mod tests;
