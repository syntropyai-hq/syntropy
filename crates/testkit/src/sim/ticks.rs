// SPDX-License-Identifier: MIT OR Apache-2.0

//! The test [`TickSource`] (RFC-0060 §2.1 rule T1; QUAL-E3-T17).
//!
//! A [`SimTicks`] counts ticks of a [`SimClock`]'s virtual time at a rate the test chooses, so
//! hot-thread code generic over [`TickSource`] runs on virtual time. Its ticks advance only when
//! the clock's owner advances the clock. The simulator's per-node device timelines (RFC-0060
//! §8.4) are QUAL-E3-T16's.
//!
//! [`SimClock`]: super::SimClock

use syntropy_core::{Clock, DeviceClockSync, MonoTime, Nanos, TickRate, TickSource};

use super::SimClockHandle;

/// A [`TickSource`] on a [`SimClock`](super::SimClock)'s virtual time: `ticks` is the virtual
/// time since `MonoTime(0)` counted at `rate`, rounded down, and `now` is the clock's instant.
/// Clones read the same clock.
#[derive(Clone, Debug)]
pub struct SimTicks {
    clock: SimClockHandle,
    rate: TickRate,
}

impl SimTicks {
    /// Ticks of `clock` at `rate`.
    #[must_use]
    pub fn new(clock: SimClockHandle, rate: TickRate) -> SimTicks {
        SimTicks { clock, rate }
    }

    /// The rate the ticks count at.
    #[must_use]
    pub fn rate(&self) -> TickRate {
        self.rate
    }
}

impl TickSource for SimTicks {
    fn ticks(&self) -> u64 {
        self.rate.nanos_to_ticks(Nanos(self.clock.now().0))
    }

    fn ticks_to_nanos(&self, ticks: u64) -> Nanos {
        self.rate.ticks_to_nanos(ticks)
    }

    fn now(&self) -> MonoTime {
        self.clock.now()
    }

    fn device_to_ticks(&self, device_ts: u64, sync: &DeviceClockSync) -> u64 {
        sync.to_ticks(device_ts, self.rate.hz())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SimClock;

    /// A 3 GHz counter.
    fn rate() -> TickRate {
        TickRate::from_hz(3_000_000_000).unwrap()
    }

    /// The criterion "the test implementation's ticks advance only with its `SimClock`".
    #[test]
    fn ticks_advance_only_with_the_clock() {
        let mut clock = SimClock::starting_at(MonoTime(1_000), 0);
        let ticks = SimTicks::new(clock.handle(), rate());
        let first = ticks.ticks();
        assert_eq!(first, 3_000);
        for _ in 0..1_000 {
            assert_eq!(ticks.ticks(), first, "no tick without an advance");
        }
        clock.advance(Nanos(1_000));
        assert_eq!(ticks.ticks(), first + 3_000);
        assert_eq!(ticks.ticks_to_nanos(ticks.ticks() - first), Nanos(1_000));
        assert_eq!(ticks.now(), MonoTime(2_000));
        // A clone reads the same clock.
        let copy = ticks.clone();
        clock.advance(Nanos(5));
        assert_eq!(copy.ticks(), ticks.ticks());
        assert_eq!(copy.now(), MonoTime(2_005));
    }

    #[test]
    fn ticks_to_nanos_round_trips_the_chosen_rate() {
        let clock = SimClock::new();
        let ticks = SimTicks::new(clock.handle(), rate());
        assert_eq!(ticks.rate(), rate());
        assert_eq!(ticks.ticks_to_nanos(3_000_000_000), Nanos(1_000_000_000));
        assert_eq!(ticks.ticks_to_nanos(3), Nanos(1));
        assert_eq!(ticks.ticks_to_nanos(0), Nanos(0));
    }

    #[test]
    fn device_timestamps_map_onto_the_virtual_ticks() {
        let mut clock = SimClock::new();
        let ticks = SimTicks::new(clock.handle(), rate());
        clock.advance(Nanos(10));
        // The provider samples its 1 GHz device clock at 500 when the host counter reads 30.
        let sync = DeviceClockSync {
            device_ts: 500,
            host_ticks: ticks.ticks(),
            device_hz: 1_000_000_000,
        };
        assert_eq!(sync.host_ticks, 30);
        assert_eq!(ticks.device_to_ticks(520, &sync), 30 + 60);
        assert_eq!(ticks.device_to_ticks(490, &sync), 0);
    }

    /// Hot-thread code generic over the source runs on virtual time.
    fn time_step<T: TickSource>(source: &T, work: impl FnOnce()) -> Nanos {
        let start = source.ticks();
        work();
        source.ticks_to_nanos(source.ticks() - start)
    }

    #[test]
    fn generic_code_runs_on_virtual_time() {
        let mut clock = SimClock::new();
        let ticks = SimTicks::new(clock.handle(), rate());
        let step = time_step(&ticks, || {
            clock.advance(Nanos(250));
        });
        assert_eq!(step, Nanos(250));
    }
}
