// SPDX-License-Identifier: MIT OR Apache-2.0

//! The QUIC transport configuration of RFC-0031 §4.3, as the pinned iroh reaches it (note of
//! 2026-10-23, DIST-E7-T21).
//!
//! [`QuicSettings`] holds the table's values, from the `net.*`, `link.*` and `quic.*` sections,
//! and [`QuicSettings::transport_config`] builds the configuration a data connection uses:
//!
//! | Row | Setting |
//! |---|---|
//! | Congestion controller, LAN initial window | a [`LanWindowFactory`] over Cubic or BBR3 |
//! | Stream receive window | `2 × net.max_frame`: twice the largest slot any channel can have, since iroh sets one value for every stream of a connection |
//! | Connection receive window | the credited sum, which the backend sets with `Connection::set_receive_window` at each channel open; the configuration starts it at the credited sum of no channel, [`QuicSettings::receive_window_floor`] |
//! | Keep-alive | 500 ms, connection-wide and per path |
//! | Idle timeout | `link.failover_timeout`, connection-wide and per path |
//! | MTU discovery | noq-proto's default: on, from 1,200 bytes |
//!
//! Keep-alive and the idle timeout go both connection-wide (`keep_alive_interval`,
//! `max_idle_timeout`) and per path (`default_path_keep_alive_interval`,
//! `default_path_max_idle_timeout`). With only the connection-wide values, iroh would keep its
//! per-path defaults of 5 s and 15 s, so a dead path would be abandoned after 15 s, not within
//! `link.failover_timeout` (DIST-E7-T03, plan note of 2026-10-23). iroh clamps a per-path idle
//! timeout to 15 s and ignores a per-path keep-alive above 5 s; the table's values lie inside.

use std::sync::Arc;
use std::time::Duration;

use iroh::endpoint::{IdleTimeout, QuicTransportConfig, VarInt};

use super::congestion::LanWindowFactory;
use crate::config::{Congestion, LinkSection, NetSection, QuicSection};
use crate::transport::core::{CONTROL_RECEIVES, MAX_CONTROL_FRAME};

/// The keep-alive while a channel of a `Ready` instance is open (RFC-0031 §4.3).
pub const KEEP_ALIVE: Duration = Duration::from_millis(500);

/// `link.connect_timeout` and `link.failover_timeout` on LAN links when no setting gives one
/// (RFC-0031 §3, §7.4, §11): 2 s.
pub const LAN_TIMEOUT: Duration = Duration::from_secs(2);

/// The same two timeouts on WAN links: 10 s.
pub const WAN_TIMEOUT: Duration = Duration::from_secs(10);

/// Which of `link.*`'s two defaults a link takes (RFC-0031 §11: "2 s LAN, 10 s WAN").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinkClass {
    Lan,
    Wan,
}

/// RFC-0031 §4.3's values for one endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuicSettings {
    /// `quic.congestion`.
    pub congestion: Congestion,
    /// `quic.lan_initial_window`, in bytes.
    pub lan_initial_window: u64,
    /// `quic.max_conn_window`, in bytes: a sanity bound on the connection receive window.
    pub max_conn_window: u64,
    /// `net.max_frame`, in bytes: the largest slot a channel may have.
    pub max_frame: u64,
    /// `link.connect_timeout`, or `None` for the 2 s LAN and 10 s WAN defaults.
    pub connect_timeout: Option<Duration>,
    /// `link.failover_timeout`, or `None` for the same defaults.
    pub failover_timeout: Option<Duration>,
}

impl Default for QuicSettings {
    /// The registered defaults: Cubic, a 256 KiB LAN window, a 64 MiB connection window cap,
    /// 64 MiB frames, and the LAN or WAN timeout by link.
    fn default() -> QuicSettings {
        QuicSettings {
            congestion: Congestion::Cubic,
            lan_initial_window: 256 * 1024,
            max_conn_window: 64 * 1024 * 1024,
            max_frame: 64 * 1024 * 1024,
            connect_timeout: None,
            failover_timeout: None,
        }
    }
}

impl QuicSettings {
    /// The settings from the resolved sections.
    pub fn from_sections(net: &NetSection, link: &LinkSection, quic: &QuicSection) -> QuicSettings {
        QuicSettings {
            congestion: quic.congestion,
            lan_initial_window: quic.lan_initial_window.bytes().0,
            max_conn_window: quic.max_conn_window.bytes().0,
            max_frame: net.max_frame.bytes().0,
            connect_timeout: link.connect_timeout.map(Duration::from),
            failover_timeout: link.failover_timeout.map(Duration::from),
        }
    }

    /// `link.connect_timeout` on a link of class `class`.
    pub fn connect_timeout(&self, class: LinkClass) -> Duration {
        self.connect_timeout.unwrap_or(default_timeout(class))
    }

    /// How long an accepting side holds a stream whose `Hello` names a channel not yet accepted,
    /// before it refuses it with `NotANeighbour`: half the shorter of the LAN and WAN connect
    /// timeouts. A connector on the same settings waits at least the longer, from before its
    /// `Hello` left, so it receives the refusal rather than timing out.
    pub fn early_hold(&self) -> Duration {
        self.connect_timeout(LinkClass::Lan)
            .min(self.connect_timeout(LinkClass::Wan))
            / 2
    }

    /// `link.failover_timeout` on a link of class `class`: the idle timeout (RFC-0031 §4.3).
    pub fn failover_timeout(&self, class: LinkClass) -> Duration {
        self.failover_timeout.unwrap_or(default_timeout(class))
    }

    /// The stream receive window: twice the largest slot any channel may have, so every stream
    /// of the connection holds two frames of its channel in flight (RFC-0031 §4.3).
    pub fn stream_receive_window(&self) -> u64 {
        self.max_frame.saturating_mul(2)
    }

    /// The connection receive window a connection starts with, before any channel opens on it:
    /// room for one channel's control frames.
    pub fn receive_window_floor() -> u64 {
        u64::from(CONTROL_RECEIVES) * MAX_CONTROL_FRAME as u64
    }

    /// The connection receive window for channels whose credited bytes sum to `credited`
    /// (RFC-0031 §4.3): never smaller than what credits allow, which wins over
    /// `quic.max_conn_window`; that cap only shows a sum above it, through
    /// [`ConnectionWindow::over_cap`].
    pub fn connection_window(&self, credited: u64) -> ConnectionWindow {
        let window = credited.max(QuicSettings::receive_window_floor());
        ConnectionWindow {
            window,
            over_cap: window > self.max_conn_window,
        }
    }

    /// The transport configuration of a data connection on a link of class `class`. See the
    /// [module](self) table.
    pub fn transport_config(&self, class: LinkClass) -> QuicTransportConfig {
        let idle = self.failover_timeout(class);
        QuicTransportConfig::builder()
            .congestion_controller_factory(Arc::new(LanWindowFactory::new(
                self.congestion,
                self.lan_initial_window,
            )))
            .stream_receive_window(varint(self.stream_receive_window()))
            .receive_window(varint(QuicSettings::receive_window_floor()))
            .keep_alive_interval(KEEP_ALIVE)
            .max_idle_timeout(IdleTimeout::try_from(idle).ok())
            .default_path_keep_alive_interval(KEEP_ALIVE)
            .default_path_max_idle_timeout(idle)
            .build()
    }
}

/// A connection receive window, and whether it passed `quic.max_conn_window`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectionWindow {
    pub window: u64,
    pub over_cap: bool,
}

fn default_timeout(class: LinkClass) -> Duration {
    match class {
        LinkClass::Lan => LAN_TIMEOUT,
        LinkClass::Wan => WAN_TIMEOUT,
    }
}

/// `value` as a QUIC variable-length integer, saturating at its largest value, 2^62 − 1.
pub(crate) fn varint(value: u64) -> VarInt {
    VarInt::from_u64(value).unwrap_or(VarInt::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field of noq-proto's `TransportConfig` as its `Debug` prints it: iroh's configuration
    /// has no getters, and its `Debug` shows the inner configuration field by field.
    fn field(config: &QuicTransportConfig, name: &str) -> String {
        let shown = format!("{config:?}");
        // A field follows `{ ` or `, `, so `receive_window` does not match inside
        // `stream_receive_window`.
        let start = [format!("{{ {name}: "), format!(", {name}: ")]
            .iter()
            .find_map(|label| shown.find(label.as_str()).map(|at| at + label.len()))
            .unwrap_or_else(|| panic!("no {name} in {shown}"));
        let rest = &shown[start..];
        let end = rest.find(", ").unwrap_or(rest.len());
        rest[..end].to_owned()
    }

    #[test]
    fn keep_alive_and_idle_timeout_are_set_per_path_and_connection_wide() {
        let settings = QuicSettings::default();
        for (class, idle) in [(LinkClass::Lan, "2s"), (LinkClass::Wan, "10s")] {
            let config = settings.transport_config(class);
            assert_eq!(field(&config, "keep_alive_interval"), "Some(500ms)");
            assert_eq!(
                field(&config, "default_path_keep_alive_interval"),
                "Some(500ms)"
            );
            assert_eq!(
                field(&config, "default_path_max_idle_timeout"),
                format!("Some({idle})")
            );
            // `IdleTimeout` prints as the milliseconds it holds.
            let millis = settings.failover_timeout(class).as_millis();
            assert_eq!(
                field(&config, "max_idle_timeout"),
                format!("Some({millis})"),
                "{class:?}"
            );
        }
    }

    #[test]
    fn a_failover_timeout_setting_reaches_both_idle_timeouts() {
        let settings = QuicSettings {
            failover_timeout: Some(Duration::from_millis(1_500)),
            ..QuicSettings::default()
        };
        let config = settings.transport_config(LinkClass::Wan);
        assert_eq!(
            field(&config, "default_path_max_idle_timeout"),
            "Some(1.5s)"
        );
        assert_eq!(field(&config, "max_idle_timeout"), "Some(1500)");
    }

    #[test]
    fn the_stream_window_holds_two_of_the_largest_frames() {
        let settings = QuicSettings::default();
        let config = settings.transport_config(LinkClass::Lan);
        assert_eq!(settings.stream_receive_window(), 128 * 1024 * 1024);
        assert_eq!(
            field(&config, "stream_receive_window"),
            (128u64 * 1024 * 1024).to_string()
        );
        assert_eq!(
            field(&config, "receive_window"),
            QuicSettings::receive_window_floor().to_string()
        );
    }

    #[test]
    fn the_connection_window_never_falls_below_the_credited_sum() {
        let settings = QuicSettings::default();
        let floor = QuicSettings::receive_window_floor();
        assert_eq!(
            settings.connection_window(0),
            ConnectionWindow {
                window: floor,
                over_cap: false
            }
        );
        assert_eq!(settings.connection_window(4 * 512).window, floor.max(2048));
        // Four 16 MiB prefill slots and one more frame pass the 64 MiB cap: the window still
        // covers them, and the cap only shows it.
        let credited = 4 * 16 * 1024 * 1024 + 4096;
        assert_eq!(
            settings.connection_window(credited),
            ConnectionWindow {
                window: credited,
                over_cap: true
            }
        );
    }

    #[test]
    fn an_early_stream_is_refused_well_within_any_connectors_wait() {
        let defaults = QuicSettings::default();
        assert_eq!(defaults.early_hold(), Duration::from_secs(1));
        for class in [LinkClass::Lan, LinkClass::Wan] {
            assert!(defaults.early_hold() * 2 <= defaults.connect_timeout(class));
        }
        let configured = QuicSettings {
            connect_timeout: Some(Duration::from_millis(700)),
            ..QuicSettings::default()
        };
        assert_eq!(configured.early_hold(), Duration::from_millis(350));
    }

    #[test]
    fn the_settings_come_from_the_sections() {
        use syntropy_config::{ByteSize, Duration as ConfigDuration};
        let net = NetSection {
            max_frame: ByteSize::mib(16),
            blob_frame: ByteSize::mib(1),
            bulk_p99_ratio: 1.2,
            bulk_floor: syntropy_config::form::parse("5%").expect("a fraction"),
        };
        let link = LinkSection {
            connect_timeout: None,
            failover_timeout: Some(ConfigDuration::from_millis(750)),
            demote_after: crate::config::CountInWindow {
                count: 3,
                window: ConfigDuration::from_secs(60),
            },
            ack_timeout_min: ConfigDuration::from_nanos(200_000),
            p99_weight: None,
        };
        let quic = QuicSection {
            congestion: Congestion::Bbr,
            lan_initial_window: ByteSize::kib(512),
            max_conn_window: ByteSize::mib(32),
        };
        let settings = QuicSettings::from_sections(&net, &link, &quic);
        assert_eq!(settings.congestion, Congestion::Bbr);
        assert_eq!(settings.lan_initial_window, 512 * 1024);
        assert_eq!(settings.max_conn_window, 32 * 1024 * 1024);
        assert_eq!(settings.max_frame, 16 * 1024 * 1024);
        assert_eq!(
            settings.failover_timeout(LinkClass::Wan),
            Duration::from_millis(750)
        );
        assert_eq!(settings.connect_timeout(LinkClass::Lan), LAN_TIMEOUT);
        assert_eq!(settings.connect_timeout(LinkClass::Wan), WAN_TIMEOUT);
    }
}
