// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's clock (RFC-0060 §2.1 rule T1, §8.2).
//!
//! A [`SimClock`] holds virtual time. It never reads the host's clocks, so a run on it gives the
//! same instants on every host and in every process. Only the value that owns the `SimClock` can
//! move it forward, because advancing takes `&mut self` and the type is not `Clone`. In the
//! simulator that owner is the event loop (RFC-0060 §8.2). The code under test reads the same
//! time through a [`SimClockHandle`], which implements [`Clock`] and has no way to advance it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use syntropy_core::{Clock, MonoTime, Nanos};

/// Nanoseconds in a millisecond, for the wall clock.
const NANOS_PER_MS: u64 = 1_000_000;

/// The wall-clock reading of [`SimClock::new`] at its first instant: 2026-01-01T00:00:00Z in
/// milliseconds since the Unix epoch. Any fixed value would do; a fixed one keeps log stamps
/// identical from run to run.
pub const SIM_UNIX_MS_ORIGIN: u64 = 1_767_225_600_000;

/// Virtual time for one simulated timeline, advanced only by its owner.
///
/// It implements [`Clock`]: [`Clock::now`] is the current virtual instant, and [`Clock::unix_ms`]
/// is the wall-clock reading given at construction plus the virtual time elapsed since, rounded
/// down to the millisecond. Hand the code under test a [`SimClockHandle`] from
/// [`SimClock::handle`], or a `&SimClock` as `&dyn Clock`; neither can advance it.
///
/// A handle has no way to move the clock:
///
/// ```compile_fail,E0599
/// use syntropy_core::Nanos;
/// use syntropy_testkit::sim::SimClock;
///
/// let clock = SimClock::new();
/// let mut handle = clock.handle();
/// handle.advance(Nanos(1));
/// ```
///
/// and the clock cannot be copied into a second owner:
///
/// ```compile_fail,E0599
/// use syntropy_testkit::sim::SimClock;
///
/// let clock = SimClock::new();
/// let second = clock.clone();
/// ```
#[derive(Debug)]
pub struct SimClock {
    shared: Arc<Shared>,
}

/// A read-only view of a [`SimClock`], for the code under test. Clones share the clock.
#[derive(Clone, Debug)]
pub struct SimClockHandle {
    shared: Arc<Shared>,
}

/// What the owner and its handles share.
#[derive(Debug)]
struct Shared {
    /// The current virtual instant, in nanoseconds. Only `SimClock` stores to it.
    now: AtomicU64,
    /// The instant at construction, which `unix_ms_origin` corresponds to.
    mono_origin: MonoTime,
    /// The wall-clock reading at `mono_origin`, in milliseconds since the Unix epoch.
    unix_ms_origin: u64,
}

impl Shared {
    fn now(&self) -> MonoTime {
        MonoTime(self.now.load(Ordering::Acquire))
    }

    fn unix_ms(&self) -> u64 {
        let elapsed = self.now().saturating_duration_since(self.mono_origin);
        self.unix_ms_origin.saturating_add(elapsed.0 / NANOS_PER_MS)
    }
}

impl SimClock {
    /// A clock at `MonoTime(0)`, whose wall clock reads [`SIM_UNIX_MS_ORIGIN`] at that instant.
    pub fn new() -> SimClock {
        SimClock::starting_at(MonoTime(0), SIM_UNIX_MS_ORIGIN)
    }

    /// A clock at `now`, whose wall clock reads `unix_ms` at that instant.
    pub fn starting_at(now: MonoTime, unix_ms: u64) -> SimClock {
        SimClock {
            shared: Arc::new(Shared {
                now: AtomicU64::new(now.0),
                mono_origin: now,
                unix_ms_origin: unix_ms,
            }),
        }
    }

    /// A read-only view of this clock that sees every later advance.
    pub fn handle(&self) -> SimClockHandle {
        SimClockHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Moves the clock forward by `by` and returns the new instant.
    ///
    /// # Panics
    ///
    /// If the new instant is past the end of `MonoTime`'s range: a simulation that runs for 584
    /// years of virtual time has a bug.
    pub fn advance(&mut self, by: Nanos) -> MonoTime {
        let now = self.shared.now();
        let to = now
            .checked_add(by)
            .unwrap_or_else(|| panic!("SimClock overflow: {now:?} + {by:?}"));
        self.store(to);
        to
    }

    /// Moves the clock to `to`. Moving to the current instant changes nothing, so events
    /// scheduled at one instant can each advance the clock to it.
    ///
    /// # Panics
    ///
    /// If `to` is earlier than the current instant: the clock is monotonic, and an event loop
    /// that goes back in time has a bug.
    pub fn advance_to(&mut self, to: MonoTime) {
        let now = self.shared.now();
        assert!(to >= now, "SimClock cannot go back: from {now:?} to {to:?}");
        self.store(to);
    }

    fn store(&mut self, to: MonoTime) {
        self.shared.now.store(to.0, Ordering::Release);
    }
}

impl Default for SimClock {
    fn default() -> SimClock {
        SimClock::new()
    }
}

impl Clock for SimClock {
    fn now(&self) -> MonoTime {
        self.shared.now()
    }

    fn unix_ms(&self) -> u64 {
        self.shared.unix_ms()
    }
}

impl Clock for SimClockHandle {
    fn now(&self) -> MonoTime {
        self.shared.now()
    }

    fn unix_ms(&self) -> u64 {
        self.shared.unix_ms()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// Code under rule T1 takes its clock as a parameter.
    fn deadline(clock: &dyn Clock, after: Nanos) -> Option<MonoTime> {
        clock.now().checked_add(after)
    }

    #[test]
    fn a_new_clock_starts_at_zero_and_the_fixed_wall_clock() {
        let clock = SimClock::new();
        assert_eq!(clock.now(), MonoTime(0));
        assert_eq!(clock.unix_ms(), SIM_UNIX_MS_ORIGIN);
        assert_eq!(SimClock::default().now(), MonoTime(0));
        assert_eq!(SimClock::default().unix_ms(), SIM_UNIX_MS_ORIGIN);
    }

    #[test]
    fn the_fixed_wall_clock_is_the_first_instant_of_2026() {
        // 56 years from 1970 to 2026, 14 of them leap years (1972, 1976, ..., 2024).
        let days = 56 * 365 + 14;
        assert_eq!(SIM_UNIX_MS_ORIGIN, days * 86_400 * 1_000);
    }

    #[test]
    fn time_stands_still_until_the_owner_advances_it() {
        let mut clock = SimClock::starting_at(MonoTime(5_000), 1_000);
        let handle = clock.handle();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(clock.now(), MonoTime(5_000));
        assert_eq!(handle.now(), MonoTime(5_000));
        assert_eq!(handle.unix_ms(), 1_000);

        assert_eq!(clock.advance(Nanos(250)), MonoTime(5_250));
        assert_eq!(clock.now(), MonoTime(5_250));
        assert_eq!(handle.now(), MonoTime(5_250));
        clock.advance_to(MonoTime(9_000));
        assert_eq!(handle.now(), MonoTime(9_000));
        assert_eq!(deadline(&handle, Nanos(1)), Some(MonoTime(9_001)));
        assert_eq!(deadline(&clock, Nanos(2)), Some(MonoTime(9_002)));
    }

    #[test]
    fn every_handle_sees_the_same_time() {
        let mut clock = SimClock::new();
        let first = clock.handle();
        let second = first.clone();
        let third = clock.handle();
        clock.advance(Nanos(42));
        for handle in [&first, &second, &third] {
            assert_eq!(handle.now(), MonoTime(42));
        }
    }

    #[test]
    fn a_handle_reads_the_clock_from_another_thread() {
        let mut clock = SimClock::new();
        clock.advance(Nanos(7));
        let shared: Arc<dyn Clock> = Arc::new(clock.handle());
        let seen = std::thread::spawn(move || shared.now()).join().unwrap();
        assert_eq!(seen, MonoTime(7));
    }

    #[test]
    fn the_wall_clock_follows_virtual_time_in_whole_milliseconds() {
        let mut clock = SimClock::starting_at(MonoTime(123), 10_000);
        clock.advance(Nanos(999_999));
        assert_eq!(clock.unix_ms(), 10_000);
        clock.advance(Nanos(1));
        assert_eq!(clock.unix_ms(), 10_001);
        clock.advance(Nanos(2_500_000));
        assert_eq!(clock.unix_ms(), 10_003);
        assert_eq!(clock.handle().unix_ms(), 10_003);
    }

    #[test]
    fn the_wall_clock_saturates_instead_of_wrapping() {
        let mut clock = SimClock::starting_at(MonoTime(0), u64::MAX - 1);
        clock.advance(Nanos(5 * NANOS_PER_MS));
        assert_eq!(clock.unix_ms(), u64::MAX);
    }

    #[test]
    fn advancing_to_the_current_instant_changes_nothing() {
        let mut clock = SimClock::starting_at(MonoTime(10), SIM_UNIX_MS_ORIGIN);
        clock.advance_to(MonoTime(10));
        assert_eq!(clock.now(), MonoTime(10));
        assert_eq!(clock.advance(Nanos(0)), MonoTime(10));
    }

    #[test]
    #[should_panic(expected = "SimClock cannot go back")]
    fn the_clock_never_goes_back() {
        let mut clock = SimClock::starting_at(MonoTime(10), SIM_UNIX_MS_ORIGIN);
        clock.advance_to(MonoTime(9));
    }

    #[test]
    fn advancing_to_the_last_instant_is_allowed() {
        let mut clock = SimClock::starting_at(MonoTime(u64::MAX - 1), SIM_UNIX_MS_ORIGIN);
        assert_eq!(clock.advance(Nanos(1)), MonoTime(u64::MAX));
    }

    #[test]
    #[should_panic(expected = "SimClock overflow")]
    fn advancing_past_the_end_of_the_range_panics() {
        let mut clock = SimClock::starting_at(MonoTime(u64::MAX), SIM_UNIX_MS_ORIGIN);
        clock.advance(Nanos(1));
    }

    #[test]
    fn clocks_are_independent() {
        let mut one = SimClock::new();
        let two = SimClock::new();
        one.advance(Nanos(1));
        assert_eq!(one.now(), MonoTime(1));
        assert_eq!(two.now(), MonoTime(0));
    }

    #[test]
    fn the_clock_and_its_handles_are_send_and_sync() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<SimClock>();
        send_sync::<SimClockHandle>();
    }
}
