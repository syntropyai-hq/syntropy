// SPDX-License-Identifier: MIT OR Apache-2.0

//! `obs` in this fixture, with a stand-in for the fault-point module: always-compiled items, and a
//! feature-only module that the core binary's build leaves out.

#![forbid(unsafe_code)]

/// Fault points.
pub mod faults {
    /// Always compiled, like `syntropy_obs::faults::FaultEffect`; its symbols may appear.
    #[inline(never)]
    pub fn effect() -> u8 {
        0
    }

    /// Compiled only with `fault-injection`.
    #[cfg(feature = "fault-injection")]
    pub mod inject {
        /// The check every fault point calls.
        #[inline(never)]
        pub fn check(point: &'static str) -> usize {
            point.len()
        }
    }
}
