// SPDX-License-Identifier: MIT OR Apache-2.0

//! `quic.lan_initial_window` as a congestion controller: the initial window on paths whose round
//! trip is under 2 ms (RFC-0031 §4.3 and its note of 2026-10-23, from DIST-E7-T21).
//!
//! The QUIC stack builds one controller per path from the connection's one factory, before the
//! path has a round-trip sample, so a larger initial window set on the factory would also reach a
//! relay path that multipath adds. [`LanWindow`] wraps the controller the setting
//! `quic.congestion` names, Cubic or BBR3, and decides on the path's first round-trip sample:
//! - under [`LAN_RTT`], it reports a window of at least the LAN window while the path is in slow
//!   start, so a 1–16 MiB prefill frame need not spend several round trips there on an idle LAN;
//! - at [`LAN_RTT`] or more, a relay path among them, it leaves the inner controller's default.
//!
//! It hands control back to the inner controller at the path's first congestion event, so loss on
//! the LAN still backs off as Cubic or BBR3 would.
//!
//! **Finding the first sample.** noq-proto calls `on_ack` for each packet an acknowledgement
//! covers before it updates the path's round-trip estimate with that acknowledgement
//! (`noq-proto src/connection/mod.rs`, `on_packet_acked` then `rtt.update`). The first `on_ack`
//! therefore still sees the configured initial round trip, not a measurement. [`LanWindow`] notes
//! that value and decides on the first later value that differs from it: the estimate after its
//! first update, which is the first sample.

use std::any::Any;
use std::sync::Arc;
use std::time::{Duration, Instant};

use noq_proto::RttEstimator;
use noq_proto::congestion::{
    Bbr3Config, Controller, ControllerFactory, ControllerMetrics, CubicConfig,
};

use crate::config::Congestion;

/// The round trip below which a path is a LAN path for the initial window (RFC-0031 §4.3).
pub const LAN_RTT: Duration = Duration::from_millis(2);

/// What a path's first round-trip sample decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// No sample yet: the inner controller's initial window applies. Holds the estimate the
    /// first `on_ack` saw, the configured initial round trip, once it has run.
    Unsampled(Option<Duration>),
    /// A LAN path still in slow start: the window is at least the LAN window.
    Lan,
    /// A path at [`LAN_RTT`] or more, or one past its first congestion event: the inner
    /// controller alone decides.
    Inner,
}

/// The controller factory a connection's transport configuration names (`QuicTransportConfig::
/// congestion_controller_factory`): each path gets a [`LanWindow`] over a fresh inner controller.
#[derive(Debug)]
pub struct LanWindowFactory {
    congestion: Congestion,
    lan_window: u64,
}

impl LanWindowFactory {
    /// Paths controlled by `congestion` (`quic.congestion`), with `lan_window` bytes
    /// (`quic.lan_initial_window`) as the window of a LAN path in slow start.
    pub fn new(congestion: Congestion, lan_window: u64) -> LanWindowFactory {
        LanWindowFactory {
            congestion,
            lan_window,
        }
    }

    /// The `quic.congestion` setting the inner controllers follow.
    pub fn congestion(&self) -> Congestion {
        self.congestion
    }

    /// The window of a LAN path in slow start, in bytes.
    pub fn lan_window(&self) -> u64 {
        self.lan_window
    }

    fn inner(&self, now: Instant, current_mtu: u16) -> Box<dyn Controller> {
        match self.congestion {
            Congestion::Cubic => Arc::new(CubicConfig::default()).build(now, current_mtu),
            Congestion::Bbr => Arc::new(Bbr3Config::default()).build(now, current_mtu),
        }
    }
}

impl ControllerFactory for LanWindowFactory {
    fn build(self: Arc<Self>, now: Instant, current_mtu: u16) -> Box<dyn Controller> {
        Box::new(LanWindow {
            inner: self.inner(now, current_mtu),
            lan_window: self.lan_window,
            path: Path::Unsampled(None),
        })
    }
}

/// One path's controller: the inner controller, with the LAN window while a LAN path is in slow
/// start. See the [module](self).
#[derive(Debug)]
pub struct LanWindow {
    inner: Box<dyn Controller>,
    lan_window: u64,
    path: Path,
}

impl LanWindow {
    /// Takes the path's first round-trip sample: a LAN path below [`LAN_RTT`], any other path
    /// at it or above. Later samples change nothing.
    pub fn on_first_rtt(&mut self, rtt: Duration) {
        if let Path::Unsampled(_) = self.path {
            self.path = if rtt < LAN_RTT {
                Path::Lan
            } else {
                Path::Inner
            };
        }
    }

    /// Takes the round-trip estimate an `on_ack` sees: the first is the configured initial round
    /// trip, and the first that differs from it is the first sample (see the [module](self)).
    pub fn observe_rtt(&mut self, estimate: Duration) {
        match self.path {
            Path::Unsampled(None) => self.path = Path::Unsampled(Some(estimate)),
            Path::Unsampled(Some(initial)) if estimate != initial => self.on_first_rtt(estimate),
            _ => {}
        }
    }

    /// Whether the path counts as a LAN path in slow start, so the LAN window applies.
    pub fn lan_window_applies(&self) -> bool {
        self.path == Path::Lan
    }

    /// The inner controller's own window.
    pub fn inner_window(&self) -> u64 {
        self.inner.window()
    }
}

impl Controller for LanWindow {
    fn on_sent(&mut self, now: Instant, bytes: u64, largest_pn: u64) {
        self.inner.on_sent(now, bytes, largest_pn);
    }

    fn on_packet_sent(&mut self, now: Instant, bytes: u16, pn: u64) {
        self.inner.on_packet_sent(now, bytes, pn);
    }

    fn on_cwnd_limited(&mut self) {
        self.inner.on_cwnd_limited();
    }

    fn on_ack(
        &mut self,
        now: Instant,
        sent: Instant,
        bytes: u64,
        pn: u64,
        app_limited: bool,
        rtt: &RttEstimator,
    ) {
        self.observe_rtt(rtt.get());
        self.inner.on_ack(now, sent, bytes, pn, app_limited, rtt);
    }

    fn on_end_acks(
        &mut self,
        now: Instant,
        in_flight: u64,
        app_limited: bool,
        largest_packet_num_acked: Option<u64>,
    ) {
        self.inner
            .on_end_acks(now, in_flight, app_limited, largest_packet_num_acked);
    }

    fn on_congestion_event(
        &mut self,
        now: Instant,
        sent: Instant,
        is_persistent_congestion: bool,
        is_ecn: bool,
        lost_bytes: u64,
        largest_lost_pn: u64,
    ) {
        // The first congestion event ends slow start: the inner controller decides from here.
        self.path = Path::Inner;
        self.inner.on_congestion_event(
            now,
            sent,
            is_persistent_congestion,
            is_ecn,
            lost_bytes,
            largest_lost_pn,
        );
    }

    fn on_packet_lost(&mut self, lost_bytes: u16, pn: u64, now: Instant) {
        self.inner.on_packet_lost(lost_bytes, pn, now);
    }

    fn on_spurious_congestion_event(&mut self) {
        self.inner.on_spurious_congestion_event();
    }

    fn on_mtu_update(&mut self, new_mtu: u16) {
        self.inner.on_mtu_update(new_mtu);
    }

    fn on_ack_frequency_update(
        &mut self,
        ack_eliciting_threshold: u64,
        requested_max_ack_delay: Duration,
    ) {
        self.inner
            .on_ack_frequency_update(ack_eliciting_threshold, requested_max_ack_delay);
    }

    fn window(&self) -> u64 {
        let inner = self.inner.window();
        if self.lan_window_applies() {
            inner.max(self.lan_window)
        } else {
            inner
        }
    }

    fn metrics(&self) -> ControllerMetrics {
        let mut metrics = self.inner.metrics();
        metrics.congestion_window = self.window();
        metrics
    }

    fn clone_box(&self) -> Box<dyn Controller> {
        Box::new(LanWindow {
            inner: self.inner.clone_box(),
            lan_window: self.lan_window,
            path: self.path,
        })
    }

    fn initial_window(&self) -> u64 {
        self.inner.initial_window()
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAN: u64 = 256 * 1024;
    const MTU: u16 = 1200;

    fn path(congestion: Congestion) -> LanWindow {
        let factory = Arc::new(LanWindowFactory::new(congestion, LAN));
        let controller = factory.build(Instant::now(), MTU);
        *controller
            .into_any()
            .downcast::<LanWindow>()
            .expect("the factory builds a LanWindow")
    }

    /// The default initial window noq-proto gives Cubic and BBR3 at a 1,200-byte MTU:
    /// `14720.clamp(2 × 1200, 10 × 1200)` = 12,000 bytes (RFC-0031 §4.3's note).
    const DEFAULT_WINDOW: u64 = 12_000;

    #[test]
    fn a_path_under_2_ms_reaches_the_lan_window_and_one_at_2_ms_or_more_keeps_the_default() {
        for congestion in [Congestion::Cubic, Congestion::Bbr] {
            // Before any sample, no path has the LAN window: a relay path that multipath adds
            // starts here too.
            let unsampled = path(congestion);
            assert_eq!(unsampled.window(), DEFAULT_WINDOW, "{congestion:?}");
            assert_eq!(unsampled.initial_window(), DEFAULT_WINDOW, "{congestion:?}");

            for rtt in [Duration::from_micros(80), Duration::from_micros(1_999)] {
                let mut lan = path(congestion);
                lan.on_first_rtt(rtt);
                assert!(lan.lan_window_applies(), "{congestion:?} at {rtt:?}");
                assert_eq!(lan.window(), LAN, "{congestion:?} at {rtt:?}");
                assert_eq!(lan.metrics().congestion_window, LAN);
            }

            // 2 ms and above, the typical relay path among them, keep the inner default.
            for rtt in [
                LAN_RTT,
                Duration::from_millis(12),
                Duration::from_millis(80),
            ] {
                let mut far = path(congestion);
                far.on_first_rtt(rtt);
                assert!(!far.lan_window_applies(), "{congestion:?} at {rtt:?}");
                assert_eq!(far.window(), DEFAULT_WINDOW, "{congestion:?} at {rtt:?}");
            }
        }
    }

    #[test]
    fn the_initial_estimate_is_not_a_sample() {
        // What noq-proto's ack processing shows the controller: the configured initial round
        // trip on the first `on_ack`, the first sample from the next one on.
        let initial = Duration::from_millis(333);
        let mut lan = path(Congestion::Cubic);
        lan.observe_rtt(initial);
        lan.observe_rtt(initial);
        assert!(!lan.lan_window_applies());
        assert_eq!(lan.window(), DEFAULT_WINDOW);
        lan.observe_rtt(Duration::from_micros(150));
        assert!(lan.lan_window_applies());
        assert_eq!(lan.window(), LAN);
        // Later estimates change nothing.
        lan.observe_rtt(Duration::from_millis(40));
        assert_eq!(lan.window(), LAN);

        let mut far = path(Congestion::Bbr);
        far.observe_rtt(initial);
        far.observe_rtt(Duration::from_millis(30));
        assert!(!far.lan_window_applies());
        assert_eq!(far.window(), DEFAULT_WINDOW);
    }

    #[test]
    fn only_the_first_sample_decides() {
        let mut far = path(Congestion::Cubic);
        far.on_first_rtt(Duration::from_millis(5));
        far.on_first_rtt(Duration::from_micros(100));
        assert_eq!(far.window(), DEFAULT_WINDOW);

        let mut lan = path(Congestion::Cubic);
        lan.on_first_rtt(Duration::from_micros(100));
        lan.on_first_rtt(Duration::from_millis(5));
        assert_eq!(lan.window(), LAN);
    }

    #[test]
    fn the_first_congestion_event_hands_control_back_to_the_inner_controller() {
        for congestion in [Congestion::Cubic, Congestion::Bbr] {
            let mut lan = path(congestion);
            lan.on_first_rtt(Duration::from_micros(300));
            assert_eq!(lan.window(), LAN);
            let now = Instant::now();
            lan.on_congestion_event(now, now, false, false, 1_200, 7);
            assert!(!lan.lan_window_applies(), "{congestion:?}");
            assert_eq!(lan.window(), lan.inner_window(), "{congestion:?}");
            assert!(lan.window() < LAN, "{congestion:?}");
        }
    }

    #[test]
    fn a_clone_keeps_the_decision() {
        let mut lan = path(Congestion::Cubic);
        lan.on_first_rtt(Duration::from_micros(300));
        let copy = lan.clone_box();
        assert_eq!(copy.window(), LAN);
    }

    #[test]
    fn a_lan_window_below_the_inner_window_changes_nothing() {
        let factory = Arc::new(LanWindowFactory::new(Congestion::Cubic, 4_096));
        let mut lan = *factory
            .build(Instant::now(), MTU)
            .into_any()
            .downcast::<LanWindow>()
            .expect("a LanWindow");
        lan.on_first_rtt(Duration::from_micros(300));
        assert_eq!(lan.window(), DEFAULT_WINDOW);
    }
}
