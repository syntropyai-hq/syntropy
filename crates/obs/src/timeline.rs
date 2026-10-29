// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hot-thread time (RFC-0053 §2.1; RFC-0060 §2.1 rule T1). A hot-thread module: the lint of
//! `cargo xtask ci lint`'s `hot-threads` step rejects `tracing` here.
//!
//! `obs` reads no clock. All of its time goes through a [`TickSource`], taken as a generic
//! parameter and never as `dyn`, so production code is monomorphised to the hardware counter and
//! the simulator's to virtual time:
//!
//! - durations are tick deltas converted with [`TickSource::ticks_to_nanos`];
//! - record timestamps come from [`TickSource::now`], so they share one origin with protocol
//!   events;
//! - a tick reading is placed on that timeline through an origin pair, one tick reading and one
//!   `now()` reading, which [`Timeline::resample`] takes again after every resume from suspend;
//! - a device timestamp is mapped with [`TickSource::device_to_ticks`] through the provider's
//!   latest [`DeviceClockSync`], then placed like a tick reading.

use syntropy_core::{DeviceClockSync, MonoTime, Nanos, TickSource};

/// A thread's view of time: its [`TickSource`] and the origin pair that places tick readings on
/// the `Clock` timeline.
#[derive(Clone, Debug)]
pub struct Timeline<T: TickSource> {
    ticks: T,
    origin_ticks: u64,
    origin: MonoTime,
}

impl<T: TickSource> Timeline<T> {
    /// A timeline on `ticks`, with its origin pair taken now.
    pub fn new(ticks: T) -> Timeline<T> {
        let origin_ticks = ticks.ticks();
        let origin = ticks.now();
        Timeline {
            ticks,
            origin_ticks,
            origin,
        }
    }

    /// Takes the origin pair again: after every resume from suspend, when the tick counter may
    /// have stopped while the `Clock` went on (RFC-0053 §2.1).
    pub fn resample(&mut self) {
        self.origin_ticks = self.ticks.ticks();
        self.origin = self.ticks.now();
    }

    /// The tick source.
    pub fn ticks(&self) -> &T {
        &self.ticks
    }

    /// A tick reading, to time a duration from.
    #[inline]
    pub fn start(&self) -> u64 {
        self.ticks.ticks()
    }

    /// The time since `start`, a reading from [`Timeline::start`].
    #[inline]
    pub fn elapsed(&self, start: u64) -> Nanos {
        self.ticks
            .ticks_to_nanos(self.ticks.ticks().saturating_sub(start))
    }

    /// The current instant on the `Clock` timeline, for a record's timestamp.
    #[inline]
    pub fn now(&self) -> MonoTime {
        self.ticks.now()
    }

    /// Where tick reading `ticks` falls on the `Clock` timeline.
    #[inline]
    pub fn at(&self, ticks: u64) -> MonoTime {
        if ticks >= self.origin_ticks {
            self.origin
                .saturating_add(self.ticks.ticks_to_nanos(ticks - self.origin_ticks))
        } else {
            let before = self.ticks.ticks_to_nanos(self.origin_ticks - ticks);
            MonoTime(self.origin.0.saturating_sub(before.0))
        }
    }

    /// Where device timestamp `device_ts` falls on the `Clock` timeline, through `sync`.
    #[inline]
    pub fn device_at(&self, device_ts: u64, sync: &DeviceClockSync) -> MonoTime {
        self.at(self.ticks.device_to_ticks(device_ts, sync))
    }

    /// The time between two device timestamps of one device, through `sync`.
    #[inline]
    pub fn device_elapsed(&self, begin: u64, end: u64, sync: &DeviceClockSync) -> Nanos {
        let (begin, end) = (
            self.ticks.device_to_ticks(begin, sync),
            self.ticks.device_to_ticks(end, sync),
        );
        self.ticks.ticks_to_nanos(end.saturating_sub(begin))
    }
}
