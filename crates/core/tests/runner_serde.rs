// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner lifecycle types the command channel carries (RFC-0001 §3.5, §5.11, §18.28;
//! DIST-E1-T15): `Deadline`, `DrainReason`, `SequenceOutcome`, `FetchPriority` and
//! `PressureLevel` round-trip through serde, and a `Deadline` holds no absolute time.
//!
//! Each value round-trips through postcard, the command channel's encoding, which tags a variant
//! by its index, and through JSON, which tags it by name. `ipc`'s own tests round-trip the
//! messages through the channel.

use std::fmt::Debug;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use syntropy_core::clock::MonoTime;
use syntropy_core::error::ErrorClass;
use syntropy_core::{
    Deadline, DrainReason, FetchPriority, Nanos, PlanEpoch, PressureLevel, SequenceOutcome,
};

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let bytes = postcard::to_allocvec(value).expect("serialize");
    let back: T = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(&back, value, "postcard round trip through {bytes:?}");
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&back, value, "JSON round trip through {json}");
}

fn epoch() -> impl Strategy<Value = PlanEpoch> {
    (any::<u64>(), any::<u64>()).prop_map(|(term, seq)| PlanEpoch { term, seq })
}

fn deadline() -> impl Strategy<Value = Deadline> {
    any::<u64>().prop_map(|n| Deadline::within(Nanos(n)))
}

fn drain_reason() -> impl Strategy<Value = DrainReason> {
    prop_oneof![
        epoch().prop_map(|successor| DrainReason::Replan { successor }),
        Just(DrainReason::Unload),
        Just(DrainReason::Shutdown),
        Just(DrainReason::NodeLeaving),
        Just(DrainReason::RestartBudgetSpent),
    ]
}

fn error_class() -> impl Strategy<Value = ErrorClass> {
    prop::sample::select(vec![
        ErrorClass::Unsupported,
        ErrorClass::Capacity,
        ErrorClass::Transient,
        ErrorClass::Link,
        ErrorClass::Device,
        ErrorClass::Integrity,
        ErrorClass::Correctness,
        ErrorClass::Drift,
        ErrorClass::Timeout,
        ErrorClass::Auth,
        ErrorClass::Cancelled,
        ErrorClass::Internal,
    ])
}

fn sequence_outcome() -> impl Strategy<Value = SequenceOutcome> {
    prop_oneof![
        Just(SequenceOutcome::Finished),
        error_class().prop_map(SequenceOutcome::Failed),
        epoch().prop_map(|to| SequenceOutcome::Migrated { to }),
    ]
}

fn fetch_priority() -> impl Strategy<Value = FetchPriority> {
    prop_oneof![
        Just(FetchPriority::Urgent),
        Just(FetchPriority::Normal),
        Just(FetchPriority::Background),
    ]
}

fn pressure_level() -> impl Strategy<Value = PressureLevel> {
    prop_oneof![
        Just(PressureLevel::Normal),
        Just(PressureLevel::Moderate),
        Just(PressureLevel::High),
        Just(PressureLevel::Critical),
    ]
}

proptest! {
    #[test]
    fn deadline_round_trips(value in deadline()) {
        round_trip(&value);
    }

    #[test]
    fn drain_reason_round_trips(value in drain_reason()) {
        round_trip(&value);
    }

    #[test]
    fn sequence_outcome_round_trips(value in sequence_outcome()) {
        round_trip(&value);
    }

    #[test]
    fn fetch_priority_round_trips(value in fetch_priority()) {
        round_trip(&value);
    }

    #[test]
    fn pressure_level_round_trips(value in pressure_level()) {
        round_trip(&value);
    }

    /// A deadline is a duration: each receiver counts it from its own receipt, so two receipts
    /// of one deadline expire `within` after each, whatever their clocks read.
    #[test]
    fn a_deadline_counts_from_each_receipt(
        within in 0u64..1 << 40,
        first in 0u64..1 << 60,
        second in 0u64..1 << 60,
    ) {
        let deadline = Deadline::within(Nanos(within));
        prop_assert_eq!(deadline.expires_at(MonoTime(first)), MonoTime(first + within));
        prop_assert_eq!(deadline.expires_at(MonoTime(second)), MonoTime(second + within));
    }
}

/// The encoding holds the duration and nothing else: no instant, no clock.
#[test]
fn a_deadline_encodes_only_its_duration() {
    let json = serde_json::to_value(Deadline::within(Nanos(30_000_000_000))).unwrap();
    assert_eq!(json, serde_json::json!({ "within": 30_000_000_000u64 }));
}

/// An expiry past the clock's end saturates rather than wrapping.
#[test]
fn a_deadline_saturates_at_the_clocks_end() {
    let deadline = Deadline::within(Nanos(10));
    assert_eq!(
        deadline.expires_at(MonoTime(u64::MAX - 3)),
        MonoTime(u64::MAX)
    );
}

/// The successor epoch travels in `DrainReason::Replan`, under the variant's own name.
#[test]
fn a_replan_drain_names_its_successor() {
    let reason = DrainReason::Replan {
        successor: PlanEpoch { term: 7, seq: 2 },
    };
    assert_eq!(
        serde_json::to_value(reason).unwrap(),
        serde_json::json!({ "Replan": { "successor": { "term": 7, "seq": 2 } } })
    );
}

/// `FetchPriority` sorts in service order with `Normal` as its default; `PressureLevel` sorts from
/// none to worst with `Normal` as its default.
#[test]
fn the_orders_and_defaults_are_the_rfcs() {
    assert!(FetchPriority::Urgent < FetchPriority::Normal);
    assert!(FetchPriority::Normal < FetchPriority::Background);
    assert_eq!(FetchPriority::default(), FetchPriority::Normal);
    assert!(PressureLevel::Normal < PressureLevel::Moderate);
    assert!(PressureLevel::Moderate < PressureLevel::High);
    assert!(PressureLevel::High < PressureLevel::Critical);
    assert_eq!(PressureLevel::default(), PressureLevel::Normal);
}
