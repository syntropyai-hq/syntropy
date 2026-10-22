// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hot-thread timing (RFC-0060 §2.1 rule T1; RFC-0001 §5.14). Owner: RFC-0060.
//!
//! `exec` and `obs` time their hot loops through a [`TickSource`], not a [`Clock`]: a raw tick
//! counter they read on every step, converted to nanoseconds only where a duration is needed. The
//! trait is statically dispatched. Code takes it as a generic parameter, so production code is
//! monomorphised to the hardware counter at no cost, and the simulator's to its virtual time.
//! A `dyn TickSource` does not compile, because the trait requires [`Clone`]:
//!
//! ```compile_fail,E0038
//! use syntropy_core::TickSource;
//!
//! fn record_step(ticks: &dyn TickSource) -> u64 {
//!     ticks.ticks()
//! }
//! ```
//!
//! The production source is `SystemTicks` in `syntropy_io::os`, beside `SystemClock`, because
//! reading the hardware counter takes `unsafe` (RFC-0001 §4.2 rule 8). The test source is
//! `SimTicks` in `testkit`, on a `SimClock`'s virtual time. [`TickRate`] converts ticks to
//! nanoseconds for both, and [`DeviceClockSync::to_ticks`] maps device timestamps onto a tick
//! timeline for both.
//!
//! [`Clock`]: crate::Clock

use serde::{Deserialize, Serialize};

use crate::clock::MonoTime;
use crate::units::Nanos;

/// Hot-thread time source for `exec` and `obs` (rule T1). Used only as a generic parameter,
/// never `dyn`.
pub trait TickSource: Clone + Send + Sync {
    /// Raw ticks: the invariant TSC on x86-64, CNTVCT_EL0 on arm64, or the OS monotonic counter
    /// where neither is usable. In the simulator, virtual time (RFC-0060 §8.4).
    fn ticks(&self) -> u64;

    /// A tick delta in nanoseconds, at the tick rate calibrated against `Clock` at startup.
    fn ticks_to_nanos(&self, ticks: u64) -> Nanos;

    /// The current instant on the `Clock` timeline, so step records and protocol events share
    /// one origin.
    fn now(&self) -> MonoTime;

    /// Maps a device timestamp (a CUDA or HIP event, a Metal GPU timestamp, a Vulkan timestamp
    /// query; RFC-0040 §3.7) onto the tick timeline, through the device's latest sync sample.
    fn device_to_ticks(&self, device_ts: u64, sync: &DeviceClockSync) -> u64;
}

/// One correlated reading of a device clock and the host tick counter, taken by the device's
/// provider, with the device clock's rate. A sample older than the provider's resync interval
/// is refreshed first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceClockSync {
    /// The device clock's reading at the sample.
    pub device_ts: u64,
    /// The host tick counter's reading at the same moment.
    pub host_ticks: u64,
    /// The device clock's rate, in ticks per second.
    pub device_hz: u64,
}

impl DeviceClockSync {
    /// Maps `device_ts` onto a host tick timeline counting `tick_hz` ticks per second. The
    /// result is the sample's host ticks, plus the device time elapsed since the sample's device
    /// reading, converted from `device_hz` to `tick_hz`. A timestamp before the sample maps
    /// before it, rounded towards earlier ticks.
    ///
    /// The result saturates at 0 and `u64::MAX`. A sample whose `device_hz` is 0 cannot convert,
    /// so every timestamp maps to its `host_ticks`.
    #[must_use]
    pub fn to_ticks(&self, device_ts: u64, tick_hz: u64) -> u64 {
        if self.device_hz == 0 {
            return self.host_ticks;
        }
        let delta = i128::from(device_ts) - i128::from(self.device_ts);
        let Some(scaled) = delta.checked_mul(i128::from(tick_hz)) else {
            // Only a delta beyond the whole range overflows, so the result is at an end.
            return if delta < 0 { 0 } else { u64::MAX };
        };
        let ticks = i128::from(self.host_ticks) + scaled.div_euclid(i128::from(self.device_hz));
        ticks.clamp(0, i128::from(u64::MAX)) as u64
    }
}

/// Nanoseconds in a second.
const NANOS_PER_SEC: u64 = 1_000_000_000;

/// The fixed-point position of [`TickRate`]'s multiplier.
const SHIFT: u32 = 32;

/// A tick counter's rate, and the conversion of tick counts to nanoseconds at that rate.
///
/// [`TickRate::ticks_to_nanos`] multiplies by a fixed-point factor, round(10⁹ × 2³² / `hz`), and
/// rounds to the nearest nanosecond, with no division on the hot path. Up to 10 GHz the factor
/// is at least 4.29 × 10⁸, so its own rounding adds at most 1.2 × 10⁻⁹ of the result, and the
/// final rounding at most half a nanosecond [derivation: 0.5 / 4.29 × 10⁸].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TickRate {
    hz: u64,
    /// round(10⁹ × 2³² / hz).
    factor: u64,
}

impl TickRate {
    /// One tick per nanosecond: the rate of a counter that already counts nanoseconds.
    pub const NANOS: TickRate = TickRate {
        hz: NANOS_PER_SEC,
        factor: 1 << SHIFT,
    };

    /// The rate of a counter that advances `hz` times a second. `None` for 0, and for a rate
    /// above about 8.6 × 10¹⁸ Hz, whose factor would round to 0.
    #[must_use]
    pub const fn from_hz(hz: u64) -> Option<TickRate> {
        if hz == 0 {
            return None;
        }
        let scaled = (NANOS_PER_SEC as u128) << SHIFT;
        // At most 10⁹ × 2³² ≈ 4.3 × 10¹⁸ (at 1 Hz), which fits in a u64.
        let factor = (scaled + hz as u128 / 2) / hz as u128;
        if factor == 0 {
            return None;
        }
        Some(TickRate {
            hz,
            factor: factor as u64,
        })
    }

    /// Ticks per second.
    #[must_use]
    pub const fn hz(&self) -> u64 {
        self.hz
    }

    /// `ticks` at this rate, in nanoseconds, rounded to the nearest and saturating at
    /// `u64::MAX`.
    #[must_use]
    #[inline]
    pub fn ticks_to_nanos(&self, ticks: u64) -> Nanos {
        let product = u128::from(ticks) * u128::from(self.factor) + (1 << (SHIFT - 1));
        Nanos(u64::try_from(product >> SHIFT).unwrap_or(u64::MAX))
    }

    /// The whole ticks this rate counts in `nanos`, rounded down and saturating at `u64::MAX`.
    #[must_use]
    pub fn nanos_to_ticks(&self, nanos: Nanos) -> u64 {
        let ticks = u128::from(nanos.0) * u128::from(self.hz) / u128::from(NANOS_PER_SEC);
        u64::try_from(ticks).unwrap_or(u64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nanosecond_counter_converts_exactly() {
        let rate = TickRate::NANOS;
        assert_eq!(TickRate::from_hz(1_000_000_000), Some(rate));
        for ticks in [0, 1, 999, 1_000_000_007, u64::MAX / 2, u64::MAX] {
            assert_eq!(rate.ticks_to_nanos(ticks), Nanos(ticks));
            assert_eq!(rate.nanos_to_ticks(Nanos(ticks)), ticks);
        }
    }

    /// The criterion "`ticks_to_nanos` round-trips a known rate", at rates from a 24 MHz
    /// CNTVCT_EL0 to a 5.2 GHz TSC: a second of ticks is a second, and converting either way
    /// and back loses at most one unit of the coarser side, a rounding, and the factor's error
    /// of at most 1.2 × 10⁻⁹ of the duration (bounded here by 1 / 8 × 10⁸).
    #[test]
    fn ticks_to_nanos_round_trips_known_rates() {
        for hz in [
            24_000_000u64,
            100_000_000,
            1_000_000_000,
            2_995_200_000,
            3_000_000_000,
            5_200_000_000,
        ] {
            let rate = TickRate::from_hz(hz).unwrap();
            assert_eq!(rate.hz(), hz);
            assert_eq!(rate.ticks_to_nanos(hz), Nanos(NANOS_PER_SEC), "{hz} Hz");
            assert_eq!(rate.nanos_to_ticks(Nanos(NANOS_PER_SEC)), hz, "{hz} Hz");
            assert_eq!(rate.ticks_to_nanos(0), Nanos(0));
            for nanos in [1u64, 37, 1_000, 999_999_937, 86_400 * NANOS_PER_SEC] {
                let ticks = rate.nanos_to_ticks(Nanos(nanos));
                let back = rate.ticks_to_nanos(ticks).0;
                let tick_ns = NANOS_PER_SEC.div_ceil(hz);
                let factor_ns = nanos / 800_000_000;
                assert!(
                    back.abs_diff(nanos) <= tick_ns + 1 + factor_ns,
                    "{hz} Hz: {nanos} ns → {ticks} ticks → {back} ns"
                );
            }
            for ticks in [1u64, 3, 1_000_003, hz * 3_600] {
                let back = rate.nanos_to_ticks(rate.ticks_to_nanos(ticks));
                let nano_ticks = hz.div_ceil(NANOS_PER_SEC);
                let factor_ns = rate.ticks_to_nanos(ticks).0 / 800_000_000;
                assert!(
                    back.abs_diff(ticks) <= nano_ticks * (1 + factor_ns) + 1,
                    "{hz} Hz: {ticks} ticks → {back}"
                );
            }
        }
    }

    #[test]
    fn conversions_are_monotonic_and_saturate() {
        let rate = TickRate::from_hz(2_995_200_000).unwrap();
        let mut last = Nanos(0);
        for ticks in (0..10_000u64).map(|t| t * 7) {
            let nanos = rate.ticks_to_nanos(ticks);
            assert!(nanos >= last);
            last = nanos;
        }
        let slow = TickRate::from_hz(1).unwrap();
        assert_eq!(slow.ticks_to_nanos(1), Nanos(NANOS_PER_SEC));
        assert_eq!(slow.ticks_to_nanos(u64::MAX), Nanos(u64::MAX));
        assert_eq!(
            slow.nanos_to_ticks(Nanos(u64::MAX)),
            u64::MAX / NANOS_PER_SEC
        );
        let fast = TickRate::from_hz(8_000_000_000_000_000_000).unwrap();
        assert_eq!(fast.nanos_to_ticks(Nanos(u64::MAX)), u64::MAX);
        assert_eq!(TickRate::from_hz(0), None);
        assert_eq!(TickRate::from_hz(u64::MAX), None);
    }

    /// A device clock at 100 MHz, sampled when it read 5,000 and the host counter read
    /// 1,000,000 at 3 GHz: each device tick is 30 host ticks.
    #[test]
    fn device_timestamps_map_through_the_sample() {
        let sync = DeviceClockSync {
            device_ts: 5_000,
            host_ticks: 1_000_000,
            device_hz: 100_000_000,
        };
        let hz = 3_000_000_000;
        assert_eq!(sync.to_ticks(5_000, hz), 1_000_000);
        assert_eq!(sync.to_ticks(5_010, hz), 1_000_300);
        assert_eq!(sync.to_ticks(4_990, hz), 999_700);
        // One second of device time is one second of host ticks.
        assert_eq!(sync.to_ticks(5_000 + 100_000_000, hz), 1_000_000 + hz);
        // Before the host counter's origin, and past the end of the range.
        assert_eq!(sync.to_ticks(0, hz), 1_000_000 - 150_000);
        assert_eq!(
            DeviceClockSync {
                host_ticks: 10,
                ..sync
            }
            .to_ticks(0, hz),
            0
        );
        assert_eq!(
            DeviceClockSync {
                host_ticks: u64::MAX - 10,
                ..sync
            }
            .to_ticks(6_000, hz),
            u64::MAX
        );
        assert_eq!(sync.to_ticks(u64::MAX, u64::MAX), u64::MAX);
        // A device rate that does not divide the host rate rounds towards earlier ticks: one
        // tick of a 7 Hz clock is 428,571,428.57 ticks of a 3 GHz counter.
        let odd = DeviceClockSync {
            device_hz: 7,
            host_ticks: 1_000_000_000,
            ..sync
        };
        assert_eq!(odd.to_ticks(5_001, hz), 1_000_000_000 + 428_571_428);
        assert_eq!(odd.to_ticks(4_999, hz), 1_000_000_000 - 428_571_429);
        // A sample with no rate cannot convert.
        let unrated = DeviceClockSync {
            device_hz: 0,
            ..sync
        };
        assert_eq!(unrated.to_ticks(9_999, hz), 1_000_000);
    }

    /// Hot-thread code takes the source as a generic parameter, as `exec` and `obs` will.
    #[derive(Clone)]
    struct Fixed(u64);

    impl TickSource for Fixed {
        fn ticks(&self) -> u64 {
            self.0
        }

        fn ticks_to_nanos(&self, ticks: u64) -> Nanos {
            TickRate::NANOS.ticks_to_nanos(ticks)
        }

        fn now(&self) -> MonoTime {
            MonoTime(self.0)
        }

        fn device_to_ticks(&self, device_ts: u64, sync: &DeviceClockSync) -> u64 {
            sync.to_ticks(device_ts, TickRate::NANOS.hz())
        }
    }

    fn step_nanos<T: TickSource>(source: &T, started: u64) -> Nanos {
        source.ticks_to_nanos(source.ticks() - started)
    }

    #[test]
    fn hot_code_takes_the_source_as_a_generic_parameter() {
        assert_eq!(step_nanos(&Fixed(1_500), 1_000), Nanos(500));
        let source = Fixed(42);
        assert_eq!(source.now(), MonoTime(42));
        let sync = DeviceClockSync {
            device_ts: 0,
            host_ticks: 100,
            device_hz: 1_000_000_000,
        };
        assert_eq!(source.device_to_ticks(25, &sync), 125);
    }

    #[test]
    fn a_device_sample_serializes_with_its_field_names() {
        let sync = DeviceClockSync {
            device_ts: 1,
            host_ticks: 2,
            device_hz: 3,
        };
        let json = serde_json::to_string(&sync).unwrap();
        assert_eq!(json, r#"{"device_ts":1,"host_ticks":2,"device_hz":3}"#);
        assert_eq!(
            serde_json::from_str::<DeviceClockSync>(&json).unwrap(),
            sync
        );
    }
}
