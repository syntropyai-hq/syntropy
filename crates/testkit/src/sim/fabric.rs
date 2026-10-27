// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulated fabric: links between simulated nodes and what a message on one costs (RFC-0060
//! §8.3; ADR-0011).
//!
//! A [`SimFabric`] holds one [`LinkModel`] per directed link ([`LinkId`]) and answers one
//! question per message: when does it arrive, if at all?
//! - **A reliable frame** on a stream (one `Channel`, RFC-0001 §5.8) arrives at
//!   `max(s + S/β + ⌈S/mtu⌉ · per_packet_overhead + α + jitter, last delivery on the stream)`,
//!   where `s`, when it starts to go out, is the later of its send time and the end of whatever
//!   the link was already sending. So every stream on a link shares its β, in send order, and a
//!   bulk transfer delays the frames of a data channel beside it (RFC-0060 §8.3, note of
//!   2026-10-23). Each stream keeps FIFO order however the jitter falls. Each lost packet delays
//!   its frame by one QUIC probe timeout (`PTO`), as a QUIC stream recovers it.
//! - **A datagram** (membership gossip) is lost with probability `loss` and never sent again.
//!   With probability `reorder` it leaves the link's order and takes one more α draw.
//! - **A split** marks links down, in one direction or both. A link going down drops every
//!   message in flight on it: a [`Sent`] ticket from before the split never arrives, even after
//!   the link comes back.
//!
//! The fabric only answers. The world's model schedules the delivery events, asks
//! [`SimFabric::arrives`] when each one comes due, and turns a credit window and a downed link
//! into channel behaviour (`SimTransport`, QUAL-E8-T04).

use std::collections::BTreeMap;

use syntropy_core::ids::{Endpoint, LinkId};
use syntropy_core::{BytesPerSec, Entropy, MonoTime, Nanos, NodeId};

use super::dist::{Dist, unit};

/// How a link behaves (RFC-0060 §8.3).
#[derive(Clone, Debug, PartialEq)]
pub struct LinkModel {
    /// One-way latency per message.
    pub alpha: Dist,
    /// Goodput.
    pub beta: BytesPerSec,
    /// Added per message.
    pub jitter: Dist,
    /// Per packet: a datagram is lost outright, a reliable frame waits one PTO per lost packet.
    pub loss: f64,
    /// Datagrams only.
    pub reorder: f64,
    /// Bytes per packet.
    pub mtu: u32,
    pub up: bool,
    pub per_packet_overhead: Nanos,
}

/// QUIC's default `max_ack_delay` (RFC 9000 §18.2), the last term of the PTO.
pub const MAX_ACK_DELAY: Nanos = Nanos(25_000_000);

/// The floor of the PTO's variance term: `max(4·rttvar, 1 ms)` (RFC 9002 §6.2.1).
pub const PTO_GRANULARITY: Nanos = Nanos(1_000_000);

/// The round-trip time a link assumes before its first sample (RFC 9002 §6.2.2).
pub const INITIAL_RTT: Nanos = Nanos(333_000_000);

const MS: u64 = 1_000_000;
const US: u64 = 1_000;

/// A preset of RFC-0060 §8.3's table: test links built from the measured ranges of
/// `hardware-and-performance-model.md` §2.5. They are presets for tests, not Syntropy
/// measurements, and a scenario may override any field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    Lan2_5g,
    Lan10g,
    Wifi6Home,
    Tb5Rdma,
    TbIp,
    WanInvite,
    Shm,
}

impl Preset {
    /// Every preset, in the table's order.
    pub const ALL: [Preset; 7] = [
        Preset::Lan2_5g,
        Preset::Lan10g,
        Preset::Wifi6Home,
        Preset::Tb5Rdma,
        Preset::TbIp,
        Preset::WanInvite,
        Preset::Shm,
    ];

    /// The name a scenario uses (RFC-0060 §8.5).
    pub fn name(self) -> &'static str {
        match self {
            Preset::Lan2_5g => "lan_2_5g",
            Preset::Lan10g => "lan_10g",
            Preset::Wifi6Home => "wifi6_home",
            Preset::Tb5Rdma => "tb5_rdma",
            Preset::TbIp => "tb_ip",
            Preset::WanInvite => "wan_invite",
            Preset::Shm => "shm",
        }
    }

    pub fn from_name(name: &str) -> Option<Preset> {
        Preset::ALL.into_iter().find(|preset| preset.name() == name)
    }

    /// The preset's link.
    ///
    /// α, β, loss and the tail are RFC-0060 §8.3's. The table gives no other field, so the
    /// presets set them plainly: no jitter, no reorder, no per-packet overhead, an MTU of 1,500
    /// bytes on IP links and 4,096 on RDMA, and on shared memory a frame of any size as one
    /// packet. `wifi6_home`'s body has σ 0.92, the value that puts the mixture's p50, p90 and p99
    /// at the table's 5, 18 and 250 ms (RFC-0060 §8.3, note of 2026-10-23).
    pub fn model(self) -> LinkModel {
        let log_normal = |median_us: u64, sigma: f64| Dist::LogNormal {
            median: Nanos(median_us * US),
            sigma,
        };
        let (alpha, gigabits_per_s, loss, mtu) = match self {
            Preset::Lan2_5g => (log_normal(150, 0.3), 2.35, 0.0, 1_500),
            Preset::Lan10g => (log_normal(100, 0.3), 9.4, 0.0, 1_500),
            Preset::Wifi6Home => (
                Dist::Mixture {
                    p_tail: 0.02,
                    body: Box::new(log_normal(5_000, 0.92)),
                    tail: Box::new(Dist::Uniform {
                        lo: Nanos(200 * MS),
                        hi: Nanos(300 * MS),
                    }),
                },
                0.4,
                1e-3,
                1_500,
            ),
            // 2.5 GB/s, in bytes: 20 Gb/s.
            Preset::Tb5Rdma => (log_normal(15, 0.2), 20.0, 0.0, 4_096),
            Preset::TbIp => (log_normal(250, 0.3), 9.0, 0.0, 1_500),
            Preset::WanInvite => (
                Dist::Mixture {
                    p_tail: 0.01,
                    body: Box::new(log_normal(25_000, 0.5)),
                    tail: Box::new(Dist::Const(Nanos(4 * 25 * MS))),
                },
                0.05,
                1e-3,
                1_500,
            ),
            // 20 GB/s, in bytes: 160 Gb/s.
            Preset::Shm => (Dist::Const(Nanos(2 * US)), 160.0, 0.0, u32::MAX),
        };
        LinkModel {
            alpha,
            beta: BytesPerSec(gigabits_per_s * 1e9 / 8.0),
            jitter: Dist::Const(Nanos(0)),
            loss,
            reorder: 0.0,
            mtu,
            up: true,
            per_packet_overhead: Nanos(0),
        }
    }
}

impl LinkModel {
    /// The preset named `name`, as a scenario names it.
    pub fn preset(name: &str) -> Option<LinkModel> {
        Preset::from_name(name).map(Preset::model)
    }
}

/// QUIC's running round-trip estimate for a link (RFC 9002 §5.3), which sets the PTO. A message's
/// round-trip sample is twice its one-way delay, α plus jitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RttEstimate {
    pub srtt: Nanos,
    pub rttvar: Nanos,
    samples: u64,
}

impl RttEstimate {
    fn new() -> RttEstimate {
        RttEstimate {
            srtt: INITIAL_RTT,
            rttvar: Nanos(INITIAL_RTT.0 / 2),
            samples: 0,
        }
    }

    /// `PTO = srtt + max(4·rttvar, 1 ms) + max_ack_delay` (RFC 9002 §6.2.1).
    pub fn pto(&self) -> Nanos {
        let variance = self.rttvar.0.saturating_mul(4).max(PTO_GRANULARITY.0);
        Nanos(
            self.srtt
                .0
                .saturating_add(variance)
                .saturating_add(MAX_ACK_DELAY.0),
        )
    }

    fn sample(&mut self, rtt: Nanos) {
        if self.samples == 0 {
            self.srtt = rtt;
            self.rttvar = Nanos(rtt.0 / 2);
        } else {
            // In 128 bits: a century-scale α overflows `3·rttvar` and `7·srtt` in 64, and each
            // average is at most the larger of its terms, so it fits back.
            let deviation = self.srtt.0.abs_diff(rtt.0);
            let rttvar = (3 * u128::from(self.rttvar.0) + u128::from(deviation)) / 4;
            let srtt = (7 * u128::from(self.srtt.0) + u128::from(rtt.0)) / 8;
            self.rttvar = Nanos(u64::try_from(rttvar).expect("an average of u64 values"));
            self.srtt = Nanos(u64::try_from(srtt).expect("an average of u64 values"));
        }
        self.samples += 1;
    }
}

/// One link's model and running state.
#[derive(Clone, Debug)]
struct Link {
    model: LinkModel,
    /// Raised each time the link goes down, so messages from before never arrive.
    generation: u64,
    alpha_cursor: usize,
    jitter_cursor: usize,
    rtt: RttEstimate,
    /// The last delivery on each reliable stream.
    streams: BTreeMap<u64, MonoTime>,
    /// The last in-order datagram delivery.
    last_datagram: MonoTime,
    /// When the link finishes sending what it has been given: its serialisation queue.
    busy_until: MonoTime,
}

/// A message on its way: when it is due, and the link state it was sent under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sent {
    pub link: LinkId,
    pub at: MonoTime,
    /// The packets of a reliable frame lost and sent again, each a PTO.
    pub lost_packets: u64,
    generation: u64,
}

/// Which way a split cuts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// From the first node to the second only; the other way stays up.
    OneWay,
    /// Both ways.
    Both,
}

/// The simulated fabric: every directed link and its state (RFC-0060 §8.3).
#[derive(Clone, Debug, Default)]
pub struct SimFabric {
    links: BTreeMap<LinkId, Link>,
}

/// The node a link endpoint belongs to.
fn node_of(endpoint: &Endpoint) -> Option<NodeId> {
    match endpoint {
        Endpoint::Device(device) => Some(device.node),
        Endpoint::Tier(tier) => Some(tier.node),
        Endpoint::Nic { node, .. } => Some(*node),
        _ => None,
    }
}

impl SimFabric {
    pub fn new() -> SimFabric {
        SimFabric::default()
    }

    /// Adds `link` with `model`, or replaces its model and state.
    pub fn add_link(&mut self, link: LinkId, model: LinkModel) {
        self.links.insert(
            link,
            Link {
                model,
                generation: 0,
                alpha_cursor: 0,
                jitter_cursor: 0,
                rtt: RttEstimate::new(),
                streams: BTreeMap::new(),
                last_datagram: MonoTime(0),
                busy_until: MonoTime(0),
            },
        );
    }

    pub fn model(&self, link: &LinkId) -> Option<&LinkModel> {
        self.links.get(link).map(|state| &state.model)
    }

    /// The link's running round-trip estimate.
    pub fn rtt(&self, link: &LinkId) -> Option<RttEstimate> {
        self.links.get(link).map(|state| state.rtt)
    }

    pub fn is_up(&self, link: &LinkId) -> bool {
        self.links.get(link).is_some_and(|state| state.model.up)
    }

    /// Marks `link` up or down. Going down drops every message in flight on it, and empties its
    /// serialisation queue. Returns whether the link exists.
    pub fn set_up(&mut self, link: &LinkId, up: bool) -> bool {
        let Some(state) = self.links.get_mut(link) else {
            return false;
        };
        if state.model.up && !up {
            state.generation += 1;
            // What it was sending is dropped, so it is no longer busy sending it.
            state.busy_until = MonoTime(0);
        }
        state.model.up = up;
        true
    }

    /// Takes the links from `a` to `b` down, and with [`Split::Both`] those from `b` to `a`.
    /// Returns how many links it marked.
    pub fn split(&mut self, a: NodeId, b: NodeId, way: Split) -> usize {
        self.mark_between(a, b, way, false)
    }

    /// Brings the links between `a` and `b` back up, both ways. Returns how many it marked.
    pub fn heal(&mut self, a: NodeId, b: NodeId) -> usize {
        self.mark_between(a, b, Split::Both, true)
    }

    fn mark_between(&mut self, a: NodeId, b: NodeId, way: Split, up: bool) -> usize {
        let chosen: Vec<LinkId> = self
            .links
            .keys()
            .filter(|link| {
                let ends = (node_of(&link.src), node_of(&link.dst));
                ends == (Some(a), Some(b)) || (way == Split::Both && ends == (Some(b), Some(a)))
            })
            .copied()
            .collect();
        for link in &chosen {
            self.set_up(link, up);
        }
        chosen.len()
    }

    /// Sends a reliable frame of `bytes` on `stream` over `link` at `now`, and says when it
    /// arrives; `None` if the link is down or unknown, when the frame is dropped.
    pub fn send_reliable(
        &mut self,
        link: LinkId,
        stream: u64,
        now: MonoTime,
        bytes: u64,
        entropy: &dyn Entropy,
    ) -> Option<Sent> {
        let state = self.links.get_mut(&link).filter(|state| state.model.up)?;
        let one_way = state.one_way(entropy);
        let pto = state.rtt.pto();
        let packets = state.packets(bytes);
        let lost_packets = if state.model.loss > 0.0 {
            (0..packets)
                .filter(|_| unit(entropy) < state.model.loss)
                .count() as u64
        } else {
            0
        };
        let sent = state.queue(now, bytes, packets);
        let delay = one_way.saturating_add(lost_packets.saturating_mul(pto.0));
        let due = sent.saturating_add(Nanos(delay));
        let last = state.streams.entry(stream).or_insert(MonoTime(0));
        let at = due.max(*last);
        *last = at;
        state.rtt.sample(Nanos(one_way.saturating_mul(2)));
        Some(Sent {
            link,
            at,
            lost_packets,
            generation: state.generation,
        })
    }

    /// Sends a datagram of `bytes` over `link` at `now`, and says when it arrives; `None` if the
    /// link is down or unknown, or the datagram is lost.
    pub fn send_datagram(
        &mut self,
        link: LinkId,
        now: MonoTime,
        bytes: u64,
        entropy: &dyn Entropy,
    ) -> Option<Sent> {
        let state = self.links.get_mut(&link).filter(|state| state.model.up)?;
        let one_way = state.one_way(entropy);
        state.rtt.sample(Nanos(one_way.saturating_mul(2)));
        if state.model.loss > 0.0 && unit(entropy) < state.model.loss {
            return None;
        }
        let packets = state.packets(bytes);
        let due = state
            .queue(now, bytes, packets)
            .saturating_add(Nanos(one_way));
        let at = if state.model.reorder > 0.0 && unit(entropy) < state.model.reorder {
            // Out of order: one more α, and the link's order does not hold it back.
            let extra = state.model.alpha.sample(entropy, &mut state.alpha_cursor);
            due.saturating_add(extra)
        } else {
            let at = due.max(state.last_datagram);
            state.last_datagram = at;
            at
        };
        Some(Sent {
            link,
            at,
            lost_packets: 0,
            generation: state.generation,
        })
    }

    /// Whether a message sent as `sent` arrives now: its link is up and has not gone down since.
    pub fn arrives(&self, sent: &Sent) -> bool {
        self.links
            .get(&sent.link)
            .is_some_and(|state| state.model.up && state.generation == sent.generation)
    }
}

impl Link {
    /// α plus jitter for one message, in nanoseconds.
    fn one_way(&mut self, entropy: &dyn Entropy) -> u64 {
        let alpha = self.model.alpha.sample(entropy, &mut self.alpha_cursor);
        let jitter = self.model.jitter.sample(entropy, &mut self.jitter_cursor);
        alpha.0.saturating_add(jitter.0)
    }

    /// Puts a message of `bytes` in `packets` on the link's serialisation queue at `now`, and
    /// says when its last byte has gone out: after whatever the link was already sending.
    fn queue(&mut self, now: MonoTime, bytes: u64, packets: u64) -> MonoTime {
        let start = now.max(self.busy_until);
        let end = start.saturating_add(Nanos(self.serialisation(bytes, packets)));
        self.busy_until = end;
        end
    }

    /// Packets for a message of `bytes`: at least one.
    fn packets(&self, bytes: u64) -> u64 {
        bytes.div_ceil(u64::from(self.model.mtu.max(1))).max(1)
    }

    /// `⌈S/β⌉ + packets · per_packet_overhead`, in nanoseconds.
    fn serialisation(&self, bytes: u64, packets: u64) -> u64 {
        let beta = self.model.beta.0;
        let transfer = if beta > 0.0 {
            // A whole number of nanoseconds, rounded up; `as` saturates.
            (bytes as f64 * 1e9 / beta).ceil() as u64
        } else {
            u64::MAX
        };
        transfer.saturating_add(packets.saturating_mul(self.model.per_packet_overhead.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 9002 §5.3's averages hold for round trips up to `u64::MAX` nanoseconds, a century-scale
    /// α included, without overflow.
    #[test]
    fn the_rtt_estimate_takes_century_scale_samples() {
        let mut estimate = RttEstimate::new();
        estimate.sample(Nanos(u64::MAX));
        assert_eq!(estimate.srtt, Nanos(u64::MAX));
        estimate.sample(Nanos(u64::MAX));
        assert_eq!(estimate.srtt, Nanos(u64::MAX));
        let first = 3 * u128::from(u64::MAX / 2) / 4;
        assert_eq!(estimate.rttvar, Nanos(first as u64));
        estimate.sample(Nanos(0));
        let rttvar = (3 * first + u128::from(u64::MAX)) / 4;
        assert_eq!(estimate.rttvar, Nanos(rttvar as u64));
        assert_eq!(estimate.srtt, Nanos((7 * u128::from(u64::MAX) / 8) as u64));
        assert_eq!(estimate.pto(), Nanos(u64::MAX));
        // Small samples keep their exact averages.
        let mut small = RttEstimate::new();
        small.sample(Nanos(8));
        small.sample(Nanos(16));
        assert_eq!((small.srtt, small.rttvar), (Nanos(9), Nanos(5)));
    }
}
