// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory-pressure levels, the yield ladder and reacquisition (RFC-0020 §11). [`PressureLevel`]
//! lives in `syntropy-core`, because the daemon's `MemoryPressure` message carries it to a runner
//! (RFC-0001 §3.5, §18.28; DIST-E1-T15), and is re-exported here; the ladder comes in a later task.

#![forbid(unsafe_code)]

pub use syntropy_core::pressure::PressureLevel;

#[cfg(test)]
mod tests {
    use super::*;

    /// `tier`'s `PressureLevel` is `core`'s, not a copy: one value is both.
    #[test]
    fn tier_and_core_name_one_pressure_level() {
        let level: syntropy_core::PressureLevel = PressureLevel::High;
        assert_eq!(level, syntropy_core::pressure::PressureLevel::High);
        assert_eq!(PressureLevel::default(), PressureLevel::Normal);
        assert!(PressureLevel::Normal < PressureLevel::Moderate);
        assert!(PressureLevel::Moderate < PressureLevel::High);
        assert!(PressureLevel::High < PressureLevel::Critical);
    }
}
