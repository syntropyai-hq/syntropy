// SPDX-License-Identifier: MIT OR Apache-2.0

//! `LedgerGrant` (RFC-0001 §3.5, RFC-0020 §3, §12; OQ-IO-27): serde round trips, the
//! daemon-to-runner hand-off, and the rules on tiers.
//!
//! JSON stands in for the `ipc` framing's encoding, which is DIST-E1-T03's: any serde format
//! that round-trips the record carries the same grant.

use std::sync::mpsc;
use std::thread;

use proptest::prelude::*;
use serde::{Deserialize, Serialize};
use syntropy_core::ledger::{LedgerGrant, LedgerGrantError};
use syntropy_core::{Bytes, NodeId, TierId};

fn tier(node: u8, ordinal: u16) -> TierId {
    TierId {
        node: NodeId([node; 32]),
        ordinal,
    }
}

/// Grant lines with distinct tiers, in random order.
fn lines() -> impl Strategy<Value = Vec<(TierId, Bytes)>> {
    prop::collection::btree_map((any::<u8>(), any::<u16>()), any::<u64>(), 0..12).prop_flat_map(
        |m| {
            let v: Vec<_> = m
                .into_iter()
                .map(|((n, o), b)| (tier(n, o), Bytes(b)))
                .collect();
            Just(v).prop_shuffle()
        },
    )
}

fn grant() -> impl Strategy<Value = LedgerGrant> {
    lines().prop_map(|l| LedgerGrant::new(l).unwrap())
}

proptest! {
    #[test]
    fn grant_round_trips(g in grant()) {
        let json = serde_json::to_string(&g).unwrap();
        let back: LedgerGrant = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(back, g);
    }

    /// Every line is found by `granted`, a tier with no line is granted 0, and the order the
    /// lines came in changes neither the grant nor its encoding.
    #[test]
    fn granted_reads_each_line(l in lines(), n in any::<u8>(), o in any::<u16>()) {
        let g = LedgerGrant::new(l.clone()).unwrap();
        prop_assert_eq!(g.len(), l.len());
        for &(t, b) in &l {
            prop_assert_eq!(g.granted(t), b);
        }
        let absent = tier(n, o);
        if l.iter().all(|&(t, _)| t != absent) {
            prop_assert_eq!(g.granted(absent), Bytes(0));
        }
        let mut reversed = l;
        reversed.reverse();
        let r = LedgerGrant::new(reversed).unwrap();
        prop_assert_eq!(&r, &g);
        prop_assert_eq!(serde_json::to_string(&r).unwrap(), serde_json::to_string(&g).unwrap());
        let tiers: Vec<_> = g.iter().map(|(t, _)| t).collect();
        prop_assert!(tiers.windows(2).all(|w| w[0] < w[1]));
    }
}

// ---- The daemon-to-runner hand-off --------------------------------------------------------

/// The part of RFC-0001 §3.5's `Init` this test needs. The real message is `ipc`'s.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct StubInit {
    build_id: String,
    ledger: LedgerGrant,
}

/// One length-prefixed frame, as the IPC command channel carries messages (DIST-E1-T03): a
/// little-endian `u32` length, then the encoded message.
fn frame(msg: &StubInit) -> Vec<u8> {
    let body = serde_json::to_vec(msg).unwrap();
    let mut out = u32::try_from(body.len()).unwrap().to_le_bytes().to_vec();
    out.extend(body);
    out
}

fn unframe(bytes: &[u8]) -> StubInit {
    let (len, body) = bytes.split_at(4);
    let len = u32::from_le_bytes(len.try_into().unwrap()) as usize;
    assert_eq!(body.len(), len, "one whole frame");
    serde_json::from_slice(body).unwrap()
}

/// The stub daemon's device ledger for one node: VRAM, pinned RAM and NVMe tiers, with the
/// runner's share of each [target: fixture figures].
fn daemon_grant() -> LedgerGrant {
    LedgerGrant::new([
        (tier(7, 2), Bytes(400 << 30)),
        (tier(7, 0), Bytes(22 << 30)),
        (tier(7, 1), Bytes(96 << 30)),
    ])
    .unwrap()
}

/// A grant the stub daemon builds is the grant the stub runner decodes, on another thread,
/// from the bytes of an `Init` frame.
#[test]
fn a_daemon_grant_decodes_identically_in_the_runner() {
    let sent = StubInit {
        build_id: "test-build".to_owned(),
        ledger: daemon_grant(),
    };
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let runner = thread::spawn(move || unframe(&rx.recv().unwrap()));
    tx.send(frame(&sent)).unwrap();
    let received = runner.join().unwrap();
    assert_eq!(received, sent);
    assert_eq!(received.ledger.granted(tier(7, 0)), Bytes(22 << 30));
    assert_eq!(received.ledger.granted(tier(7, 1)), Bytes(96 << 30));
    assert_eq!(received.ledger.granted(tier(7, 2)), Bytes(400 << 30));
    assert_eq!(received.ledger.granted(tier(8, 0)), Bytes(0));
}

proptest! {
    #[test]
    fn any_grant_survives_the_hand_off(g in grant()) {
        let sent = StubInit { build_id: String::new(), ledger: g };
        prop_assert_eq!(unframe(&frame(&sent)), sent);
    }
}

// ---- The encoding and its edges -----------------------------------------------------------

/// The encoding is a record whose `per_tier` lists `[tier, bytes]` pairs in `TierId` order.
#[test]
fn encoding_is_a_record_of_ordered_pairs() {
    let g = LedgerGrant::new([(tier(1, 3), Bytes(5)), (tier(1, 0), Bytes(u64::MAX))]).unwrap();
    let v = serde_json::to_value(&g).unwrap();
    let node = serde_json::to_value(NodeId([1; 32])).unwrap();
    assert_eq!(
        v,
        serde_json::json!({
            "per_tier": [
                [{ "node": node, "ordinal": 0 }, u64::MAX],
                [{ "node": node, "ordinal": 3 }, 5],
            ]
        })
    );
}

#[test]
fn duplicate_tiers_are_refused() {
    let t = tier(3, 1);
    // Same bytes or not.
    for second in [Bytes(10), Bytes(11)] {
        assert_eq!(
            LedgerGrant::new([(t, Bytes(10)), (tier(3, 0), Bytes(1)), (t, second)]),
            Err(LedgerGrantError::DuplicateTier(t))
        );
    }
    // Same ordinal on different nodes is two tiers.
    assert!(LedgerGrant::new([(tier(1, 0), Bytes(1)), (tier(2, 0), Bytes(1))]).is_ok());

    // On decode as well.
    let line = serde_json::json!([serde_json::to_value(t).unwrap(), 10]);
    let json = serde_json::json!({ "per_tier": [line.clone(), line] });
    let err = serde_json::from_value::<LedgerGrant>(json).unwrap_err();
    assert!(err.to_string().contains("granted more than once"), "{err}");
}

#[test]
fn duplicate_tier_message_names_the_tier() {
    let msg = LedgerGrantError::DuplicateTier(tier(0xab, 4)).to_string();
    assert_eq!(
        msg,
        "tier 4 of node abababababababab is granted more than once"
    );
}

#[test]
fn empty_and_zero_grants() {
    let empty = LedgerGrant::default();
    assert!(empty.is_empty());
    assert_eq!(empty, LedgerGrant::new([]).unwrap());
    assert_eq!(empty.granted(tier(0, 0)), Bytes(0));
    assert_eq!(serde_json::to_string(&empty).unwrap(), r#"{"per_tier":[]}"#);
    // A missing `per_tier` decodes as no lines.
    assert_eq!(serde_json::from_str::<LedgerGrant>("{}").unwrap(), empty);

    // A line of 0 bytes grants what no line grants, but it is still a line.
    let zero = LedgerGrant::new([(tier(0, 0), Bytes(0))]).unwrap();
    assert_eq!(zero.granted(tier(0, 0)), Bytes(0));
    assert_eq!(zero.len(), 1);
    assert_ne!(zero, empty);
}

/// A grant from a later build, with fields this build does not know, still decodes.
#[test]
fn decodes_with_future_fields() {
    let g = daemon_grant();
    let mut v = serde_json::to_value(&g).unwrap();
    v.as_object_mut().unwrap().insert(
        "field_from_a_future_build".to_owned(),
        serde_json::json!({ "cores": [0, 1, 2] }),
    );
    assert_eq!(serde_json::from_value::<LedgerGrant>(v).unwrap(), g);
}

#[test]
fn malformed_lines_are_refused() {
    for json in [
        r#"{"per_tier": 3}"#,
        r#"{"per_tier": [[1, 2]]}"#,
        r#"{"per_tier": [[{"node": [], "ordinal": 0}, 1]]}"#,
        r#"{"per_tier": [[{"node": [], "ordinal": 0}, -1]]}"#,
    ] {
        assert!(serde_json::from_str::<LedgerGrant>(json).is_err(), "{json}");
    }
}
