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
}
