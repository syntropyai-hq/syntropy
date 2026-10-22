// SPDX-License-Identifier: MIT OR Apache-2.0

//! Link topology records, and link selection: RFC-0001 §5.2's `LinkInfo` and `LinkCaps`, RFC-0031
//! §2's `TransportKind`, and RFC-0031 §3's per-frame cost, which the solver uses to pick a hop's
//! transport and `net`'s runtime fallback uses to pick the next one.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::iface::IfaceInfo;
use crate::ids::{LinkId, LinkKind, NetMedium, RdmaKind};
use crate::measured::{MeasureSource, Measured};
use crate::policy::Objective;
use crate::units::{Bytes, BytesPerSec, Nanos};

/// One directed link and the transports usable on it (RFC-0001 §5.2). Owner: RFC-0010.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkInfo {
    /// The link's identity.
    pub id: LinkId,
    /// Transports usable on this link, best first.
    #[serde(default)]
    pub transports: SmallVec<[TransportKind; 3]>,
    /// For a network link, the interface it leaves through: name, addresses, MTU
    /// (RFC-0030 §7.1). `None` for other links.
    #[serde(default)]
    pub interface: Option<IfaceInfo>,
    /// The link is metered.
    #[serde(default)]
    pub metered: bool,
}

/// A transport that carries frames over a link (RFC-0031 §2, which adds `QuicRelay` to
/// RFC-0001 §5.8's list). Owner: RFC-0031.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TransportKind {
    /// Shared memory between two nodes on one host.
    SharedMemory,
    /// Remote direct memory access.
    Rdma(RdmaKind),
    /// QUIC on a direct path.
    Quic,
    /// QUIC through a relay, kept apart from direct paths in plans, metrics and the dashboard.
    QuicRelay,
    /// TCP.
    Tcp,
}

/// One transport's probed costs on one directed link (RFC-0001 §5.2): one record per (link,
/// transport) pair. RFC-0010 §1.3's additions to it are [`LinkCapsExt`](super::LinkCapsExt).
/// Owner: RFC-0010.
///
/// `link` and `transport` identify the record and have no default. A measured field filled by
/// default is a zero with source [`MeasureSource::Prior`], so the record reads as unmeasured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkCaps {
    /// The link.
    pub link: LinkId,
    /// The transport these figures are for.
    pub transport: TransportKind,
    /// One-way small-message latency (RFC-0010's L1 probe). Its `p50` and `p99` are RFC-0031
    /// §3's α_p50 and α_p99.
    #[serde(default)]
    pub alpha: Measured<Nanos>,
    /// Goodput. RFC-0031 §3's β is its median, `p50`.
    #[serde(default)]
    pub beta: Measured<BytesPerSec>,
    /// The full one-way time of a collective-sized message through the real device path, device
    /// buffer to device buffer (RFC-0010's L2 probe and `Warming`). Used alone, never with S/β
    /// added again.
    #[serde(default)]
    pub alpha_eff: Option<Measured<Nanos>>,
    /// The loss rate, where measured.
    #[serde(default)]
    pub loss: Option<f32>,
}

impl LinkCaps {
    /// Whether the transport has a probe result on this link. One without is tried only after
    /// every measured candidate (RFC-0031 §3). Startup measures α on every transport a pair
    /// supports and β on the best-ranked one only (RFC-0010 §3), so α decides.
    #[must_use]
    pub fn is_measured(&self) -> bool {
        self.alpha.source != MeasureSource::Prior
    }
}

/// A hop moved off its transport (RFC-0031 §3, note of 2026-10-23): the data RFC-0033 §7.1's
/// `DegradeDetail::Demoted` carries. The runner reports it, with its [`DemotionCause`] and the new
/// transport's cost, in the command channel's `EdgeDemoted` (RFC-0001 §3.5). Owner: RFC-0031.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Demotion {
    /// The hop's link.
    pub edge: LinkId,
    /// The planned transport, which could not connect.
    pub from: TransportKind,
    /// The transport the hop connected on.
    pub to: TransportKind,
}

/// Why a hop moved off its transport: the `cause` of the command channel's `EdgeDemoted` and of
/// RFC-0033 §11's `ReportEdgeDemoted`. Owner: RFC-0031.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DemotionCause {
    /// The planned transport did not connect within `link.connect_timeout`, so the hop connected
    /// on the next candidate (RFC-0031 §3).
    ConnectFailed,
    /// The transport faulted 3 times within 60 s (RFC-0031 §7.4, ADR-0011 item 7).
    RepeatedFaults,
}

/// λ on Wi-Fi links and on relayed QUIC, in latency mode: the default of `link.p99_weight` there
/// [target: RFC-0031 §3, "cost Wi-Fi at p99"].
pub const P99_WEIGHT_TAIL_HEAVY: f64 = 1.0;

/// λ on every other link in latency mode: the default of `link.p99_weight` there [target:
/// RFC-0031 §3].
pub const P99_WEIGHT: f64 = 0.25;

/// RFC-0031 §3's z on IP paths (`Quic`, `Tcp`, `QuicRelay`) until an L2 probe gives
/// `alpha_eff` [estimate: the upper end of RFC-0031 §Performance's ≈15–25 µs with polling, for a
/// 32 KiB F32 decode frame].
pub const SOFTWARE_PATH_IP: Nanos = Nanos(25_000);

/// RFC-0031 §3's z on `Rdma` until an L2 probe gives `alpha_eff` [estimate: the upper end of
/// RFC-0031 §Performance's ≈12–20 µs, for a 32 KiB F32 decode frame].
pub const SOFTWARE_PATH_RDMA: Nanos = Nanos(20_000);

/// RFC-0031 §3's z on `SharedMemory` until an L2 probe gives `alpha_eff` [estimate: RFC-0031
/// §Performance gives no z for it; its per-hop table's shared-memory hop of ≈5–10 µs, less α of
/// ≈1–5 µs and S/β of ≈3 µs, leaves ≈1–2 µs. The same subtraction on its RDMA and 10 GbE rows
/// gives ≈12–22 µs and ≈17–22 µs, matching their z ranges].
pub const SOFTWARE_PATH_SHARED_MEMORY: Nanos = Nanos(2_000);

/// λ, the weight of a link's p99 in its cost (RFC-0031 §3). In latency mode it is
/// `link.p99_weight`, `configured`, and by default [`P99_WEIGHT_TAIL_HEAVY`] on a Wi-Fi link or
/// relayed QUIC and [`P99_WEIGHT`] otherwise. In throughput mode it is 0, because the tail is
/// amortised over batches.
#[must_use]
pub fn p99_weight(caps: &LinkCaps, objective: &Objective, configured: Option<f64>) -> f64 {
    match objective {
        Objective::Latency => configured.unwrap_or_else(|| {
            let wifi = caps.link.kind == LinkKind::Network(NetMedium::WiFi);
            if wifi || caps.transport == TransportKind::QuicRelay {
                P99_WEIGHT_TAIL_HEAVY
            } else {
                P99_WEIGHT
            }
        }),
        Objective::Throughput { .. } => 0.0,
    }
}

/// RFC-0031 §3's z_t: the backend's software path at the two ends. Where an L2 probe exists it is
/// `alpha_eff − α_p50`, at 0 when `alpha_eff` is the smaller. Otherwise it is the transport's
/// default, [`SOFTWARE_PATH_IP`], [`SOFTWARE_PATH_RDMA`] or [`SOFTWARE_PATH_SHARED_MEMORY`],
/// which does not scale with the frame size.
#[must_use]
pub fn software_path(caps: &LinkCaps) -> Nanos {
    match caps.alpha_eff {
        Some(eff) => Nanos(eff.p50.0.saturating_sub(caps.alpha.p50.0)),
        None => match caps.transport {
            TransportKind::Quic | TransportKind::Tcp | TransportKind::QuicRelay => SOFTWARE_PATH_IP,
            TransportKind::Rdma(_) => SOFTWARE_PATH_RDMA,
            TransportKind::SharedMemory => SOFTWARE_PATH_SHARED_MEMORY,
        },
    }
}

/// RFC-0031 §3's cost of one frame of `size` bytes on this link and transport:
///
/// `cost_t(S) = α_p50 + S/β + λ · (α_p99 − α_p50) + z_t(S)`
///
/// with λ from [`p99_weight`] (`p99_weight` is `link.p99_weight` as configured, `None` for its
/// default) and z from [`software_path`]. It ranks the transports of one edge: the solver picks
/// `HopPlan::transport` by it at the decode frame size and `prefill_transport` at the prefill
/// chunk size, and the runtime fallback picks the next candidate by it. The objective's hop term
/// is RFC-0011 §5's, not this (RFC-0031 §3, note of 2026-10-23).
///
/// The sum is rounded to the nearest nanosecond once. A tail below the median counts as 0, and
/// so does a negative or NaN λ. An infinite λ saturates: a link with a tail then costs
/// `Nanos(u64::MAX)`, and one without (α_p99 = α_p50) adds nothing for it. A link whose β is not
/// a positive rate cannot carry the frame: its cost is `Nanos(u64::MAX)`, as is any sum that
/// would exceed it.
#[must_use]
pub fn link_cost(
    caps: &LinkCaps,
    size: Bytes,
    objective: &Objective,
    p99_weight: Option<f64>,
) -> Nanos {
    let beta = caps.beta.p50.0;
    if !(beta > 0.0 && beta.is_finite()) {
        return Nanos(u64::MAX);
    }
    let lambda = self::p99_weight(caps, objective, p99_weight);
    let tail = caps.alpha.p99.0.saturating_sub(caps.alpha.p50.0);
    // Only a positive λ on a positive tail adds anything, so 0 · ∞ never makes a NaN.
    let tail_term = if lambda > 0.0 && tail > 0 {
        lambda * tail as f64
    } else {
        0.0
    };
    let total = caps.alpha.p50.0 as f64
        + size.0 as f64 * 1e9 / beta
        + tail_term
        + software_path(caps).0 as f64;
    // Every term is finite and non-negative, or +∞. A sum at or above 2^64, or an infinite one,
    // saturates to `u64::MAX` in the cast.
    debug_assert!(total >= 0.0, "cost_t is never NaN or negative: {total}");
    Nanos(total.round() as u64)
}

/// RFC-0031 §3's order for transports with no probe result: `SharedMemory`, `Rdma`, direct `Quic`,
/// `Tcp`, then `QuicRelay`. Lower ranks first.
#[must_use]
pub fn unmeasured_rank(transport: TransportKind) -> u8 {
    match transport {
        TransportKind::SharedMemory => 0,
        TransportKind::Rdma(_) => 1,
        TransportKind::Quic => 2,
        TransportKind::Tcp => 3,
        TransportKind::QuicRelay => 4,
    }
}

/// The order in which to try an edge's candidate transports (RFC-0031 §3): every measured
/// candidate by [`link_cost`] at `size`, cheapest first, then every unmeasured one in
/// [`unmeasured_rank`] order. Ties go by the unmeasured rank, then by the transport, so the order
/// is the same on every node.
#[must_use]
pub fn candidate_order<'a>(
    candidates: &'a [LinkCaps],
    size: Bytes,
    objective: &Objective,
    p99_weight: Option<f64>,
) -> Vec<&'a LinkCaps> {
    let mut ranked: Vec<&LinkCaps> = candidates.iter().collect();
    ranked.sort_by_key(|caps| {
        let measured = caps.is_measured();
        let cost = if measured {
            link_cost(caps, size, objective, p99_weight)
        } else {
            Nanos(0)
        };
        (
            !measured,
            cost,
            unmeasured_rank(caps.transport),
            caps.transport,
        )
    });
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{DeviceId, Endpoint, NodeId};

    fn link(medium: NetMedium) -> LinkId {
        let end = |n: u8| {
            Endpoint::Device(DeviceId {
                node: NodeId([n; 32]),
                ordinal: 0,
            })
        };
        LinkId {
            src: end(1),
            dst: end(2),
            kind: LinkKind::Network(medium),
            path: 0,
        }
    }

    fn measured<T>(p50: T, p99: T) -> Measured<T> {
        Measured {
            p50,
            p99,
            samples: 200,
            at_unix_ms: 0,
            source: MeasureSource::Startup,
        }
    }

    /// α 100 µs at p50 and 300 µs at p99, β 1 GB/s, no L2 probe.
    fn caps(medium: NetMedium, transport: TransportKind) -> LinkCaps {
        LinkCaps {
            link: link(medium),
            transport,
            alpha: measured(Nanos(100_000), Nanos(300_000)),
            beta: measured(BytesPerSec(1e9), BytesPerSec(1e9)),
            alpha_eff: None,
            loss: None,
        }
    }

    fn throughput() -> Objective {
        Objective::Throughput {
            per_sequence_floor_tps: 10.0,
            target_concurrency: None,
            ttft_ceiling: None,
        }
    }

    /// A 32 KiB decode frame on wired QUIC in latency mode: α_p50 + S/β + 0.25 · (α_p99 − α_p50)
    /// + 25 µs = 100,000 + 32,768 + 50,000 + 25,000 ns.
    #[test]
    fn the_cost_is_rfc_0031_s_formula() {
        let c = caps(NetMedium::Ethernet, TransportKind::Quic);
        assert_eq!(
            link_cost(&c, Bytes(32_768), &Objective::Latency, None),
            Nanos(100_000 + 32_768 + 50_000 + 25_000)
        );
    }

    /// λ is 1.0 on Wi-Fi and on relayed QUIC, 0.25 elsewhere, in latency mode.
    #[test]
    fn wifi_and_relayed_links_weigh_the_whole_tail() {
        let s = Bytes(32_768);
        let wifi = caps(NetMedium::WiFi, TransportKind::Quic);
        let relay = caps(NetMedium::Ethernet, TransportKind::QuicRelay);
        let tcp = caps(NetMedium::Ethernet, TransportKind::Tcp);
        assert_eq!(p99_weight(&wifi, &Objective::Latency, None), 1.0);
        assert_eq!(p99_weight(&relay, &Objective::Latency, None), 1.0);
        assert_eq!(p99_weight(&tcp, &Objective::Latency, None), 0.25);
        let full_tail = Nanos(100_000 + 32_768 + 200_000 + 25_000);
        assert_eq!(link_cost(&wifi, s, &Objective::Latency, None), full_tail);
        assert_eq!(link_cost(&relay, s, &Objective::Latency, None), full_tail);
    }

    /// In throughput mode λ is 0, whatever `link.p99_weight` says; in latency mode the
    /// configured value replaces the default.
    #[test]
    fn throughput_mode_ignores_the_tail_and_a_configured_weight_applies_in_latency_mode() {
        let s = Bytes(32_768);
        let wifi = caps(NetMedium::WiFi, TransportKind::Quic);
        let no_tail = Nanos(100_000 + 32_768 + 25_000);
        assert_eq!(link_cost(&wifi, s, &throughput(), None), no_tail);
        assert_eq!(link_cost(&wifi, s, &throughput(), Some(2.0)), no_tail);
        assert_eq!(
            link_cost(&wifi, s, &Objective::Latency, Some(0.5)),
            Nanos(100_000 + 32_768 + 100_000 + 25_000)
        );
        // A negative or NaN weight counts as 0.
        assert_eq!(
            link_cost(&wifi, s, &Objective::Latency, Some(-1.0)),
            no_tail
        );
        assert_eq!(
            link_cost(&wifi, s, &Objective::Latency, Some(f64::NAN)),
            no_tail
        );
    }

    /// z is `alpha_eff − α_p50` once an L2 probe exists, and the transport's default before.
    #[test]
    fn z_comes_from_alpha_eff_or_the_transport_default() {
        let mut rdma = caps(
            NetMedium::Rdma(RdmaKind::Thunderbolt5),
            TransportKind::Rdma(RdmaKind::Thunderbolt5),
        );
        assert_eq!(software_path(&rdma), Nanos(20_000));
        assert_eq!(
            software_path(&caps(NetMedium::Loopback, TransportKind::SharedMemory)),
            Nanos(2_000)
        );
        for ip in [
            TransportKind::Quic,
            TransportKind::Tcp,
            TransportKind::QuicRelay,
        ] {
            assert_eq!(
                software_path(&caps(NetMedium::Ethernet, ip)),
                SOFTWARE_PATH_IP
            );
        }
        rdma.alpha_eff = Some(measured(Nanos(112_000), Nanos(150_000)));
        assert_eq!(software_path(&rdma), Nanos(12_000));
        assert_eq!(
            link_cost(&rdma, Bytes(16_384), &Objective::Latency, None),
            Nanos(100_000 + 16_384 + 50_000 + 12_000)
        );
        // An `alpha_eff` below α_p50 gives z = 0.
        rdma.alpha_eff = Some(measured(Nanos(90_000), Nanos(95_000)));
        assert_eq!(software_path(&rdma), Nanos(0));
    }

    /// An infinite configured λ saturates: the cost of a link with a tail is the most, and a link
    /// without one is not made cheaper than its median, as a NaN sum cast to 0 would make it.
    #[test]
    fn an_infinite_weight_saturates_and_never_makes_a_link_free() {
        let s = Bytes(32_768);
        let mut flat = caps(NetMedium::Ethernet, TransportKind::Quic);
        flat.alpha = measured(Nanos(100_000), Nanos(100_000));
        assert_eq!(
            link_cost(&flat, s, &Objective::Latency, Some(f64::INFINITY)),
            Nanos(100_000 + 32_768 + 25_000)
        );
        let tailed = caps(NetMedium::Ethernet, TransportKind::Quic);
        assert_eq!(
            link_cost(&tailed, s, &Objective::Latency, Some(f64::INFINITY)),
            Nanos(u64::MAX)
        );
        // Ranked, the flat link comes first and the tailed one last.
        let order: Vec<Nanos> = candidate_order(
            &[tailed.clone(), flat.clone()],
            s,
            &Objective::Latency,
            Some(f64::INFINITY),
        )
        .into_iter()
        .map(|c| link_cost(c, s, &Objective::Latency, Some(f64::INFINITY)))
        .collect();
        assert_eq!(order, [Nanos(157_768), Nanos(u64::MAX)]);
    }

    /// S/β is rounded once, with the rest of the sum, to the nearest nanosecond.
    #[test]
    fn the_sum_rounds_once() {
        let mut c = caps(NetMedium::Ethernet, TransportKind::Tcp);
        c.beta = measured(BytesPerSec(3e9), BytesPerSec(3e9));
        // 1,000 B at 3 GB/s is 333.3 ns, and 2,000 B is 666.7 ns.
        assert_eq!(
            link_cost(&c, Bytes(1_000), &throughput(), None),
            Nanos(100_000 + 333 + 25_000)
        );
        assert_eq!(
            link_cost(&c, Bytes(2_000), &throughput(), None),
            Nanos(100_000 + 667 + 25_000)
        );
    }

    /// A link with no positive β cannot carry the frame, and a tail below the median counts as 0.
    #[test]
    fn a_link_without_a_rate_costs_the_most_and_a_negative_tail_counts_as_0() {
        let mut c = caps(NetMedium::Ethernet, TransportKind::Quic);
        c.beta = Measured::default();
        assert_eq!(
            link_cost(&c, Bytes(1), &Objective::Latency, None),
            Nanos(u64::MAX)
        );
        c.beta = measured(BytesPerSec(f64::NAN), BytesPerSec(f64::NAN));
        assert_eq!(
            link_cost(&c, Bytes(1), &Objective::Latency, None),
            Nanos(u64::MAX)
        );
        c.beta = measured(BytesPerSec(1e9), BytesPerSec(1e9));
        c.alpha = measured(Nanos(100_000), Nanos(50_000));
        assert_eq!(
            link_cost(&c, Bytes(0), &Objective::Latency, None),
            Nanos(100_000 + 25_000)
        );
        c.beta = measured(BytesPerSec(1e-300), BytesPerSec(1e-300));
        assert_eq!(
            link_cost(&c, Bytes(1), &Objective::Latency, None),
            Nanos(u64::MAX)
        );
    }

    #[test]
    fn the_unmeasured_rank_is_rfc_0031_s_order() {
        let order = [
            TransportKind::SharedMemory,
            TransportKind::Rdma(RdmaKind::RoceV2),
            TransportKind::Quic,
            TransportKind::Tcp,
            TransportKind::QuicRelay,
        ];
        for pair in order.windows(2) {
            assert!(
                unmeasured_rank(pair[0]) < unmeasured_rank(pair[1]),
                "{pair:?}"
            );
        }
    }

    /// Measured candidates go first, cheapest first; unmeasured ones follow in rank order, even
    /// when the rank would put them earlier.
    #[test]
    fn candidates_are_tried_measured_by_cost_then_unmeasured_by_rank() {
        let s = Bytes(32_768);
        let mut quic = caps(NetMedium::Ethernet, TransportKind::Quic);
        let mut tcp = caps(NetMedium::Ethernet, TransportKind::Tcp);
        tcp.alpha = measured(Nanos(80_000), Nanos(300_000));
        quic.alpha = measured(Nanos(90_000), Nanos(100_000));
        let mut rdma = caps(
            NetMedium::Rdma(RdmaKind::RoceV2),
            TransportKind::Rdma(RdmaKind::RoceV2),
        );
        rdma.alpha = Measured::default();
        let mut shm = caps(NetMedium::Loopback, TransportKind::SharedMemory);
        shm.alpha = Measured::default();
        let all = [rdma.clone(), tcp.clone(), shm.clone(), quic.clone()];
        let order: Vec<TransportKind> = candidate_order(&all, s, &Objective::Latency, None)
            .into_iter()
            .map(|c| c.transport)
            .collect();
        // QUIC: 90,000 + 32,768 + 0.25 · 10,000 + 25,000; TCP: 80,000 + 32,768 + 0.25 · 220,000
        // + 25,000, which is more.
        assert_eq!(
            order,
            [
                TransportKind::Quic,
                TransportKind::Tcp,
                TransportKind::SharedMemory,
                TransportKind::Rdma(RdmaKind::RoceV2),
            ]
        );
        // In throughput mode the tail no longer counts, and TCP's lower median wins.
        let order: Vec<TransportKind> = candidate_order(&all, s, &throughput(), None)
            .into_iter()
            .map(|c| c.transport)
            .collect();
        assert_eq!(order[..2], [TransportKind::Tcp, TransportKind::Quic]);
    }

    /// Two candidates at one cost go in rank order, whichever order they arrive in.
    #[test]
    fn equal_costs_go_by_rank() {
        let quic = caps(NetMedium::Ethernet, TransportKind::Quic);
        let tcp = caps(NetMedium::Ethernet, TransportKind::Tcp);
        for all in [[quic.clone(), tcp.clone()], [tcp.clone(), quic.clone()]] {
            let first = candidate_order(&all, Bytes(64), &throughput(), None)[0].transport;
            assert_eq!(first, TransportKind::Quic);
        }
    }
}
