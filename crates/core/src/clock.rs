// SPDX-License-Identifier: MIT OR Apache-2.0

//! Injected time (RFC-0001 §5.14, RFC-0060 §2.1, ADR-0600 decision 2). Owner: RFC-0060.
//!
//! Decision-making code reads time only through a [`Clock`] it is given (rule T1), so the
//! simulator can run it on virtual time. The production clock and the simulator's clock implement
//! the trait elsewhere: the production `SystemClock` lives in `syntropy_io::os` (ADR-0022), and
//! the simulator's clock lives in `testkit`. Hot-thread timing in `exec` and
//! `obs` uses `TickSource` instead (RFC-0060 §2.1).

use serde::{Deserialize, Serialize};

use crate::units::Nanos;

/// Monotonic nanoseconds since an arbitrary origin, including time the host spent suspended, so
/// leases and failure timers see a suspend as elapsed time. Never compared across nodes.
///
/// Arithmetic is checked or saturating, never wrapping: a deadline past the end of the range is
/// `None` or the largest instant, not a time in the past.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MonoTime(pub u64);

impl MonoTime {
    /// The instant `after` later, or `None` if that is past the end of the range.
    pub fn checked_add(self, after: Nanos) -> Option<MonoTime> {
        self.0.checked_add(after.0).map(MonoTime)
    }

    /// The instant `after` later, or the largest instant if that is past the end of the range.
    pub fn saturating_add(self, after: Nanos) -> MonoTime {
        MonoTime(self.0.saturating_add(after.0))
    }

    /// The time elapsed from `earlier` to `self`, or `None` if `earlier` is later than `self`.
    pub fn checked_duration_since(self, earlier: MonoTime) -> Option<Nanos> {
        self.0.checked_sub(earlier.0).map(Nanos)
    }

    /// The time elapsed from `earlier` to `self`, or zero if `earlier` is later than `self`.
    pub fn saturating_duration_since(self, earlier: MonoTime) -> Nanos {
        Nanos(self.0.saturating_sub(earlier.0))
    }
}

/// A source of time, passed to the code that reads it (rule T1).
pub trait Clock: Send + Sync {
    /// The current instant on this node's monotonic, suspend-aware timeline.
    fn now(&self) -> MonoTime;

    /// Wall clock, for log stamps, lease expiry hints and `Measured::at_unix_ms` only.
    /// Protocol decisions never depend on it.
    fn unix_ms(&self) -> u64;
}

/// A reading of the clock that stops while the host is suspended, in nanoseconds since an
/// arbitrary origin (RFC-0030 §5.8). It is not on [`MonoTime`]'s timeline: compare only the
/// differences of two readings, against the differences of [`Clock::now`] taken around them.
/// The production reading is `syntropy_io::os::SystemClock::awake` (ADR-0022).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AwakeTime(pub u64);

impl AwakeTime {
    /// The awake time from `earlier` to `self`, or zero if `earlier` is later.
    #[must_use]
    pub fn saturating_duration_since(self, earlier: AwakeTime) -> Nanos {
        Nanos(self.0.saturating_sub(earlier.0))
    }
}

/// A [`Clock`] that can also read the clock that stops while the host is suspended (RFC-0030
/// §5.8; ADR-0022 decision 2). Code that detects a suspend takes one, so it runs on virtual time
/// in tests; only a binary's wiring passes the production one.
pub trait AwakeClock: Clock {
    /// The clock that stops while the host is suspended.
    fn awake(&self) -> AwakeTime;

    /// One tick of the coarser of the two clocks: a lower bound, not enough slack on its own.
    /// Each difference is off by up to a tick of its clock, so an interval's two differences can
    /// disagree by two ticks, plus the change in the time between reading `now` and reading
    /// `awake`, which includes any preemption between the two reads. See
    /// [`Interval::suspended`].
    fn tick(&self) -> Nanos;
}

/// The time between two observations, on both clocks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Interval {
    /// The time [`Clock::now`] advanced, suspends included.
    pub running: Nanos,
    /// The time [`AwakeClock::awake`] advanced, suspends excluded.
    pub awake: Nanos,
}

impl Interval {
    /// The time the host was suspended in the interval: `running − awake`, when that is at least
    /// `slack`, else zero, as RFC-0030 §5.8's "at least `swim.resume_detect`" reads.
    ///
    /// The two differences disagree by up to two ticks plus the change in the gap between the
    /// two reads ([`AwakeClock::tick`]), so a `slack` that must ignore that noise covers it. A
    /// threshold such as RFC-0030 §5.8's 5 s does; a slack of zero counts every gap, noise
    /// included.
    #[must_use]
    pub fn suspended(&self, slack: Nanos) -> Nanos {
        let gap = self.running.0.saturating_sub(self.awake.0);
        if gap >= slack.0 { Nanos(gap) } else { Nanos(0) }
    }

    /// How much later than `expected` the interval ended while the host was awake: `awake −
    /// expected`, when that is more than `slack`, else zero. A late tick on both clocks is a
    /// paused VM or a starved process (RFC-0030 §5.8).
    #[must_use]
    pub fn late(&self, expected: Nanos, slack: Nanos) -> Nanos {
        let late = self.awake.0.saturating_sub(expected.0);
        if late > slack.0 {
            Nanos(late)
        } else {
            Nanos(0)
        }
    }
}

/// Measures the interval between successive observations of the two clocks, from which a
/// caller finds the time the host spent suspended or paused (RFC-0030 §5.8). The runner
/// supervisor (DIST-E1-T06) and the membership shell's resume detection (DIST-E4-T05) each keep
/// one, with their own thresholds. It reads no clock: the caller passes both readings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SuspendDetector {
    last: Option<(MonoTime, AwakeTime)>,
}

impl SuspendDetector {
    /// A detector with no observation yet.
    #[must_use]
    pub const fn new() -> Self {
        Self { last: None }
    }

    /// Records the readings `now` and `awake`, taken together, and returns the interval since
    /// the previous observation. The first observation returns a zero interval. A reading earlier
    /// than the previous one counts as no time, and the later of the two stays the baseline, so
    /// a reading that went back cannot overstate the next interval.
    pub fn observe(&mut self, now: MonoTime, awake: AwakeTime) -> Interval {
        let (interval, baseline) = match self.last {
            None => (Interval::default(), (now, awake)),
            Some((then, then_awake)) => (
                Interval {
                    running: now.saturating_duration_since(then),
                    awake: awake.saturating_duration_since(then_awake),
                },
                (now.max(then), awake.max(then_awake)),
            ),
        };
        self.last = Some(baseline);
        interval
    }

    /// Reads both clocks of `clock` and observes them.
    pub fn observe_clock(&mut self, clock: &impl AwakeClock) -> Interval {
        let now = clock.now();
        let awake = clock.awake();
        self.observe(now, awake)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// A clock a test advances by hand.
    struct ManualClock {
        mono: AtomicU64,
        unix_ms: u64,
    }

    impl ManualClock {
        fn at(mono: u64) -> ManualClock {
            ManualClock {
                mono: AtomicU64::new(mono),
                unix_ms: 1_700_000_000_000,
            }
        }

        fn advance(&self, by: Nanos) {
            self.mono.fetch_add(by.0, Ordering::Relaxed);
        }
    }

    impl Clock for ManualClock {
        fn now(&self) -> MonoTime {
            MonoTime(self.mono.load(Ordering::Relaxed))
        }

        fn unix_ms(&self) -> u64 {
            self.unix_ms
        }
    }

    /// Code under rule T1 takes its clock as a parameter.
    fn lease_deadline(clock: &impl Clock, lease: Nanos) -> Option<MonoTime> {
        clock.now().checked_add(lease)
    }

    #[test]
    fn code_reads_the_time_it_is_given() {
        let clock = ManualClock::at(1_000);
        assert_eq!(lease_deadline(&clock, Nanos(500)), Some(MonoTime(1_500)));
        clock.advance(Nanos(2_000));
        assert_eq!(clock.now(), MonoTime(3_000));
        assert_eq!(lease_deadline(&clock, Nanos(500)), Some(MonoTime(3_500)));
        assert_eq!(clock.unix_ms(), 1_700_000_000_000);
    }

    #[test]
    fn a_clock_can_be_shared_as_a_trait_object_across_threads() {
        fn send_sync<T: Send + Sync + ?Sized>() {}
        send_sync::<dyn Clock>();
        let clock: std::sync::Arc<dyn Clock> = std::sync::Arc::new(ManualClock::at(7));
        let seen = std::thread::spawn({
            let clock = std::sync::Arc::clone(&clock);
            move || clock.now()
        })
        .join()
        .unwrap();
        assert_eq!(seen, MonoTime(7));
    }

    #[test]
    fn instants_order_by_time() {
        assert!(MonoTime(1) < MonoTime(2));
        assert_eq!(MonoTime::default(), MonoTime(0));
        assert_eq!(MonoTime(3).max(MonoTime(9)), MonoTime(9));
    }

    #[test]
    fn arithmetic_never_wraps() {
        let end = MonoTime(u64::MAX - 1);
        assert_eq!(end.checked_add(Nanos(1)), Some(MonoTime(u64::MAX)));
        assert_eq!(end.checked_add(Nanos(2)), None);
        assert_eq!(end.saturating_add(Nanos(10)), MonoTime(u64::MAX));
        assert_eq!(MonoTime(10).saturating_add(Nanos(5)), MonoTime(15));

        assert_eq!(
            MonoTime(10).checked_duration_since(MonoTime(4)),
            Some(Nanos(6))
        );
        assert_eq!(MonoTime(4).checked_duration_since(MonoTime(10)), None);
        assert_eq!(
            MonoTime(4).saturating_duration_since(MonoTime(10)),
            Nanos(0)
        );
        assert_eq!(MonoTime(4).saturating_duration_since(MonoTime(4)), Nanos(0));
    }

    #[test]
    fn serializes_as_a_bare_integer() {
        let json = serde_json::to_string(&MonoTime(42)).unwrap();
        assert_eq!(json, "42");
        assert_eq!(
            serde_json::from_str::<MonoTime>(&json).unwrap(),
            MonoTime(42)
        );
    }

    /// A suspend shows as running time with no awake time; a late tick as awake time past the
    /// expected interval; small differences within the slack are neither.
    #[test]
    fn the_detector_separates_suspends_from_late_ticks() {
        let mut d = SuspendDetector::new();
        assert_eq!(d.observe(MonoTime(0), AwakeTime(500)), Interval::default());
        let s = 1_000_000_000;
        // 60 s asleep inside a 61 s interval.
        let i = d.observe(MonoTime(61 * s), AwakeTime(500 + s));
        assert_eq!(
            i,
            Interval {
                running: Nanos(61 * s),
                awake: Nanos(s)
            }
        );
        assert_eq!(i.suspended(Nanos(1_000)), Nanos(60 * s));
        assert_eq!(i.late(Nanos(s), Nanos(0)), Nanos(0));
        // A tick 9 s late on both clocks: no suspend, 8 s late against a 1 s period.
        let i = d.observe(MonoTime(70 * s), AwakeTime(500 + 10 * s));
        assert_eq!(i.suspended(Nanos(1_000)), Nanos(0));
        assert_eq!(i.late(Nanos(s), Nanos(0)), Nanos(8 * s));
        assert_eq!(i.late(Nanos(s), Nanos(8 * s)), Nanos(0), "within the slack");
        // A gap below the slack is no suspend; one at least the slack is (RFC-0030 §5.8).
        let i = d.observe(MonoTime(71 * s + 10), AwakeTime(500 + 11 * s));
        assert_eq!(i.suspended(Nanos(11)), Nanos(0));
        assert_eq!(i.suspended(Nanos(10)), Nanos(10));
        // A reading that goes back counts as no time, never as a wrap, and the later reading
        // stays the baseline: one true second later on both clocks is one second, no suspend.
        let i = d.observe(MonoTime(5), AwakeTime(0));
        assert_eq!(i, Interval::default());
        let i = d.observe(MonoTime(72 * s + 10), AwakeTime(500 + 12 * s));
        assert_eq!(
            i,
            Interval {
                running: Nanos(s),
                awake: Nanos(s)
            }
        );
        assert_eq!(i.suspended(Nanos(1)), Nanos(0));
    }
}
