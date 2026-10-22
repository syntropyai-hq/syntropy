// SPDX-License-Identifier: MIT OR Apache-2.0

//! `obs` in this fixture, with a stand-in for the feature-only fault-point module.

#![forbid(unsafe_code)]

/// Fault points.
pub mod faults {
    /// Compiled only with `fault-injection`, like the real `syntropy_obs::faults::inject`.
    #[cfg(feature = "fault-injection")]
    pub mod inject {
        /// The check every fault point calls.
        #[inline(never)]
        pub fn check(point: &'static str) -> usize {
            point.len()
        }
    }
}
