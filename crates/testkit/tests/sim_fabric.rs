// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulated fabric (QUAL-E8-T02; RFC-0060 §8.3; ADR-0011).
//!
//! - Over 10⁵ samples, the `wifi6_home` preset gives p50 ≈ 5 ms, p90 ≈ 18 ms and p99 ≈ 250 ms,
//!   each within 10%.
//! - FIFO order holds on a stream under jitter.
//! - A lost packet delays its frame by exactly the PTO formula.
//! - A one-way split drops A→B and keeps B→A.

use std::sync::Arc;

use syntropy_core::ids::{Endpoint, LinkId, LinkKind, NetMedium};
use syntropy_core::{BytesPerSec, MonoTime, Nanos, NodeId};
use syntropy_testkit::sim::{
    Dist, INITIAL_RTT, LinkModel, MAX_ACK_DELAY, Preset, SimEntropy, SimFabric, Split,
};

const US: u64 = 1_000;
const MS: u64 = 1_000_000;

fn node(n: u8) -> NodeId {
    NodeId([n; 32])
}

fn link(from: u8, to: u8) -> LinkId {
    LinkId {
        src: Endpoint::Nic {
            node: node(from),
            ordinal: 0,
        },
        dst: Endpoint::Nic {
            node: node(to),
            ordinal: 0,
        },
        kind: LinkKind::Network(NetMedium::Ethernet),
        path: 0,
    }
}

/// A link with a fixed α, no jitter and no loss, at 1 GB/s.
fn plain(alpha: Nanos) -> LinkModel {
    LinkModel {
        alpha: Dist::Const(alpha),
        beta: BytesPerSec(1e9),
        jitter: Dist::Const(Nanos(0)),
        loss: 0.0,
        reorder: 0.0,
        mtu: 1_500,
        up: true,
        per_packet_overhead: Nanos(0),
    }
}

fn percentile(sorted: &[u64], q: f64) -> f64 {
    sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)] as f64
}

#[test]
fn wifi6_home_gives_the_tables_percentiles() {
    let model = Preset::Wifi6Home.model();
    let entropy = SimEntropy::new(2026);
    let mut cursor = 0;
    let mut draws: Vec<u64> = (0..100_000)
        .map(|_| model.alpha.sample(&entropy, &mut cursor).0)
        .collect();
    draws.sort_unstable();
    for (q, target_ms) in [(0.50, 5.0), (0.90, 18.0), (0.99, 250.0)] {
        let got_ms = percentile(&draws, q) / MS as f64;
        let error = got_ms / target_ms - 1.0;
        println!("p{}: {got_ms:.2} ms ({:+.1}%)", q * 100.0, error * 100.0);
        assert!(error.abs() < 0.10, "p{} is {got_ms:.2} ms", q * 100.0);
    }
}

/// The standard normal CDF, by Abramowitz and Stegun 7.1.26 (absolute error under 1.5e-7).
fn normal_cdf(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-x * x).exp();
    0.5 * (1.0 + if z < 0.0 { -erf } else { erf })
}

#[test]
fn wifi6_homes_parameters_put_its_exact_quantiles_on_the_targets() {
    // The mixture's CDF in closed form, from the preset's own parameters, and its quantiles by
    // bisection: no sampling, so the preset cannot drift from RFC-0060 §8.3's targets unseen.
    let Dist::Mixture { p_tail, body, tail } = Preset::Wifi6Home.model().alpha else {
        panic!("wifi6_home's α is a mixture");
    };
    let Dist::LogNormal { median, sigma } = *body else {
        panic!("a log-normal body");
    };
    let Dist::Uniform { lo, hi } = *tail else {
        panic!("a uniform tail");
    };
    let (median, lo, hi) = (median.0 as f64, lo.0 as f64, hi.0 as f64);
    let cdf = |x: f64| {
        let body = normal_cdf((x / median).ln() / sigma);
        let tail = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
        (1.0 - p_tail) * body + p_tail * tail
    };
    let quantile = |q: f64| {
        let (mut low, mut high) = (1.0, 1e12);
        for _ in 0..200 {
            let mid = (low + high) / 2.0;
            if cdf(mid) < q { low = mid } else { high = mid }
        }
        high / MS as f64
    };
    for (q, target_ms) in [(0.50, 5.0), (0.90, 18.0), (0.99, 250.0)] {
        let exact = quantile(q);
        let error = exact / target_ms - 1.0;
        println!(
            "exact p{}: {exact:.2} ms ({:+.1}%)",
            q * 100.0,
            error * 100.0
        );
        assert!(error.abs() < 0.05, "p{} is {exact:.2} ms", q * 100.0);
    }
    assert_eq!(sigma, 0.92);
}

#[test]
fn a_stream_keeps_fifo_order_under_jitter() {
    let mut fabric = SimFabric::new();
    let ab = link(1, 2);
    fabric.add_link(
        ab,
        LinkModel {
            jitter: Dist::Uniform {
                lo: Nanos(0),
                hi: Nanos(5 * MS),
            },
            ..plain(Nanos(MS))
        },
    );
    let entropy = SimEntropy::new(3);
    let mut previous = MonoTime(0);
    let mut held_back = 0;
    for i in 0..1_000u64 {
        let sent = fabric
            .send_reliable(ab, 7, MonoTime(i * 10 * US), 1_000, &entropy)
            .expect("the link is up");
        assert!(
            sent.at >= previous,
            "frame {i} arrived before frame {}",
            i - 1
        );
        // Jitter of up to 5 ms over sends 10 µs apart would reorder most frames; the stream
        // holds each such frame until the one before it has arrived.
        if sent.at == previous {
            held_back += 1;
        }
        previous = sent.at;
    }
    assert!(held_back > 100, "only {held_back} frames were held back");
    // Another stream on the same link has its own order: sent last, it can still arrive first.
    let other = fabric
        .send_reliable(ab, 8, MonoTime(10 * MS), 1_000, &entropy)
        .expect("the link is up");
    assert!(other.at < previous, "{other:?} after {previous:?}");
}

#[test]
fn a_frame_arrives_after_alpha_transfer_packets_and_jitter() {
    let mut fabric = SimFabric::new();
    let ab = link(1, 2);
    fabric.add_link(
        ab,
        LinkModel {
            mtu: 1_000,
            per_packet_overhead: Nanos(2 * US),
            ..plain(Nanos(100 * US))
        },
    );
    let entropy = SimEntropy::new(1);
    // 2,500 bytes: three packets, 2,500 ns at 1 GB/s.
    let sent = fabric
        .send_reliable(ab, 0, MonoTime(MS), 2_500, &entropy)
        .unwrap();
    assert_eq!(sent.at, MonoTime(MS + 100 * US + 2_500 + 3 * 2 * US));
    assert_eq!(sent.lost_packets, 0);
}

/// Every stream on a link shares its β, in send order (RFC-0060 §8.3, note of 2026-10-23): a
/// message sent while the link is still sending another starts after it, whatever its stream,
/// and a datagram queues the same way; another link is not delayed, and a drained link times a
/// message as an idle one.
#[test]
fn streams_on_one_link_share_its_beta_in_send_order() {
    let mut fabric = SimFabric::new();
    let (ab, ac) = (link(1, 2), link(1, 3));
    fabric.add_link(ab, plain(Nanos(100 * US)));
    fabric.add_link(ac, plain(Nanos(100 * US)));
    let entropy = SimEntropy::new(1);
    // A 1 MB bulk frame takes 1 ms at 1 GB/s.
    let bulk = fabric
        .send_reliable(ab, 1, MonoTime(0), 1_000_000, &entropy)
        .unwrap();
    assert_eq!(bulk.at, MonoTime(MS + 100 * US));
    // A 1 KB frame on another stream 10 µs later starts once the bulk frame has gone.
    let data = fabric
        .send_reliable(ab, 2, MonoTime(10 * US), 1_000, &entropy)
        .unwrap();
    assert_eq!(data.at, MonoTime(MS + 1_000 + 100 * US));
    // A datagram after them waits too.
    let gossip = fabric
        .send_datagram(ab, MonoTime(20 * US), 100, &entropy)
        .unwrap();
    assert_eq!(gossip.at, MonoTime(MS + 1_000 + 100 + 100 * US));
    // Another link is idle.
    let other = fabric
        .send_reliable(ac, 2, MonoTime(10 * US), 1_000, &entropy)
        .unwrap();
    assert_eq!(other.at, MonoTime(10 * US + 1_000 + 100 * US));
    // Once the link has drained, a message is timed as on an idle link.
    let later = fabric
        .send_reliable(ab, 2, MonoTime(5 * MS), 1_000, &entropy)
        .unwrap();
    assert_eq!(later.at, MonoTime(5 * MS + 1_000 + 100 * US));
}

#[test]
fn a_lost_packet_delays_its_frame_by_exactly_one_pto() {
    let mut fabric = SimFabric::new();
    let ab = link(1, 2);
    // Every packet is lost once; α is 1 ms, so each message's round-trip sample is 2 ms.
    fabric.add_link(
        ab,
        LinkModel {
            loss: 1.0,
            beta: BytesPerSec(1e12),
            ..plain(Nanos(MS))
        },
    );
    let entropy = SimEntropy::new(1);
    let pto = |srtt: u64, rttvar: u64| srtt + (4 * rttvar).max(MS) + MAX_ACK_DELAY.0;

    // Before any sample: srtt 333 ms, rttvar 166.5 ms, so PTO = 333 + 666 + 25 = 1,024 ms.
    assert_eq!(fabric.rtt(&ab).unwrap().pto(), Nanos(1_024 * MS));
    let first = fabric
        .send_reliable(ab, 0, MonoTime(0), 1_000, &entropy)
        .unwrap();
    assert_eq!(first.lost_packets, 1);
    assert_eq!(
        first.at,
        MonoTime(MS + 1 + pto(INITIAL_RTT.0, INITIAL_RTT.0 / 2))
    );

    // After one 2 ms sample: srtt 2 ms, rttvar 1 ms, so PTO = 2 + 4 + 25 = 31 ms.
    let rtt = fabric.rtt(&ab).unwrap();
    assert_eq!((rtt.srtt, rtt.rttvar), (Nanos(2 * MS), Nanos(MS)));
    let second = fabric
        .send_reliable(ab, 1, MonoTime(2_000 * MS), 1_000, &entropy)
        .unwrap();
    assert_eq!(second.at, MonoTime(2_000 * MS + MS + 1 + pto(2 * MS, MS)));
    assert_eq!(pto(2 * MS, MS), 31 * MS);

    // After a second 2 ms sample: rttvar (3·1 + 0)/4 = 0.75 ms, so PTO = 2 + 3 + 25 = 30 ms.
    // A 3,000-byte frame is two packets, both lost: two PTOs.
    let third = fabric
        .send_reliable(ab, 2, MonoTime(4_000 * MS), 3_000, &entropy)
        .unwrap();
    assert_eq!(third.lost_packets, 2);
    assert_eq!(
        third.at,
        MonoTime(4_000 * MS + MS + 3 + 2 * pto(2 * MS, 750 * US))
    );

    // With no loss, no PTO.
    fabric.add_link(ab, plain(Nanos(MS)));
    let clean = fabric
        .send_reliable(ab, 0, MonoTime(0), 1_000, &entropy)
        .unwrap();
    assert_eq!((clean.at, clean.lost_packets), (MonoTime(MS + 1_000), 0));
}

#[test]
fn a_one_way_split_drops_a_to_b_and_keeps_b_to_a() {
    let mut fabric = SimFabric::new();
    let (ab, ba) = (link(1, 2), link(2, 1));
    fabric.add_link(ab, Preset::Lan2_5g.model());
    fabric.add_link(ba, Preset::Lan2_5g.model());
    let entropy = SimEntropy::new(5);
    let before = fabric
        .send_reliable(ab, 0, MonoTime(0), 64, &entropy)
        .unwrap();

    assert_eq!(fabric.split(node(1), node(2), Split::OneWay), 1);
    assert!(!fabric.is_up(&ab) && fabric.is_up(&ba));
    assert_eq!(
        fabric.send_reliable(ab, 0, MonoTime(MS), 64, &entropy),
        None
    );
    assert_eq!(fabric.send_datagram(ab, MonoTime(MS), 64, &entropy), None);
    let back = fabric
        .send_reliable(ba, 0, MonoTime(MS), 64, &entropy)
        .expect("B to A stays up");
    assert!(fabric.arrives(&back));
    assert!(
        fabric
            .send_datagram(ba, MonoTime(MS), 64, &entropy)
            .is_some()
    );

    // Both ways, then healed.
    assert_eq!(fabric.split(node(1), node(2), Split::Both), 2);
    assert!(!fabric.is_up(&ba));
    assert!(
        !fabric.arrives(&back),
        "B to A went down with the frame in flight"
    );
    assert_eq!(fabric.heal(node(1), node(2)), 2);
    assert!(fabric.is_up(&ab) && fabric.is_up(&ba));
    // A frame in flight when its link went down never arrives, even once the link is back.
    assert!(!fabric.arrives(&before));
    let after = fabric
        .send_reliable(ab, 0, MonoTime(2 * MS), 64, &entropy)
        .unwrap();
    assert!(fabric.arrives(&after));
    assert_eq!(
        fabric.split(node(3), node(4), Split::Both),
        0,
        "no such links"
    );
}

#[test]
fn datagrams_are_lost_and_reordered_at_their_rates() {
    let entropy = SimEntropy::new(11);
    let jittery = LinkModel {
        alpha: Dist::Uniform {
            lo: Nanos(MS),
            hi: Nanos(2 * MS),
        },
        ..plain(Nanos(0))
    };
    let mut fabric = SimFabric::new();
    let ab = link(1, 2);

    // Loss.
    fabric.add_link(
        ab,
        LinkModel {
            loss: 0.25,
            ..jittery.clone()
        },
    );
    let sent = 20_000;
    let lost = (0..sent)
        .filter(|i| {
            fabric
                .send_datagram(ab, MonoTime(i * 10 * US), 100, &entropy)
                .is_none()
        })
        .count();
    let rate = lost as f64 / sent as f64;
    assert!((rate - 0.25).abs() < 0.01, "loss rate {rate}");

    // Order: none out of order without reorder, some with it.
    let inversions = |reorder: f64| {
        let mut fabric = SimFabric::new();
        fabric.add_link(
            ab,
            LinkModel {
                reorder,
                ..jittery.clone()
            },
        );
        let times: Vec<MonoTime> = (0..2_000u64)
            .map(|i| {
                fabric
                    .send_datagram(ab, MonoTime(i * 10 * US), 100, &entropy)
                    .unwrap()
                    .at
            })
            .collect();
        times.windows(2).filter(|pair| pair[1] < pair[0]).count()
    };
    assert_eq!(inversions(0.0), 0);
    let reordered = inversions(0.2);
    assert!(reordered > 100, "{reordered} datagrams out of order");
}

#[test]
fn the_presets_are_rfc_0060s_table() {
    let names: Vec<&str> = Preset::ALL.iter().map(|preset| preset.name()).collect();
    assert_eq!(
        names,
        [
            "lan_2_5g",
            "lan_10g",
            "wifi6_home",
            "tb5_rdma",
            "tb_ip",
            "wan_invite",
            "shm"
        ]
    );
    // β in bytes per second, and loss.
    let expected = [
        (2.35e9 / 8.0, 0.0),
        (9.4e9 / 8.0, 0.0),
        (400e6 / 8.0, 1e-3),
        (2.5e9, 0.0),
        (9e9 / 8.0, 0.0),
        (50e6 / 8.0, 1e-3),
        (20e9, 0.0),
    ];
    for (preset, (beta, loss)) in Preset::ALL.into_iter().zip(expected) {
        let model = LinkModel::preset(preset.name()).expect("every preset has a name");
        assert_eq!(model, preset.model());
        assert!((model.beta.0 / beta - 1.0).abs() < 1e-12, "{preset:?} β");
        assert_eq!(model.loss, loss, "{preset:?} loss");
        assert!(model.up);
    }
    assert_eq!(LinkModel::preset("lan_1g"), None);
    assert_eq!(Preset::Shm.model().alpha, Dist::Const(Nanos(2 * US)));
    // The tails: Wi-Fi's 2 % from 200 to 300 ms, the WAN's 1 % at 4 × its 25 ms median.
    match Preset::Wifi6Home.model().alpha {
        Dist::Mixture { p_tail, tail, .. } => {
            assert_eq!(p_tail, 0.02);
            assert_eq!(
                *tail,
                Dist::Uniform {
                    lo: Nanos(200 * MS),
                    hi: Nanos(300 * MS)
                }
            );
        }
        other => panic!("{other:?}"),
    }
    match Preset::WanInvite.model().alpha {
        Dist::Mixture { p_tail, tail, .. } => {
            assert_eq!(p_tail, 0.01);
            assert_eq!(*tail, Dist::Const(Nanos(100 * MS)));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn one_seed_gives_one_fabric() {
    let run = |seed| {
        let entropy = SimEntropy::new(seed);
        let mut fabric = SimFabric::new();
        fabric.add_link(link(1, 2), Preset::Wifi6Home.model());
        fabric.add_link(
            link(2, 1),
            LinkModel {
                alpha: Dist::Trace(Arc::from([Nanos(3), Nanos(1), Nanos(2)])),
                ..Preset::Wifi6Home.model()
            },
        );
        (0..500u64)
            .map(|i| {
                let ab = fabric.send_reliable(link(1, 2), i % 3, MonoTime(i * MS), 1_500, &entropy);
                let ba = fabric.send_datagram(link(2, 1), MonoTime(i * MS), 200, &entropy);
                (ab.map(|sent| sent.at), ba.map(|sent| sent.at))
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(8), run(8));
    assert_ne!(run(8), run(9));
}
