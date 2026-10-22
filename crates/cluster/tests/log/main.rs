// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cluster event log (DIST-E9-T01; RFC-0032 §6.1–§6.3).
//!
//! The acceptance criteria this file checks:
//! - `apply` is total over arbitrary sequences of well-formed base events, a superset of every
//!   log `decide` can produce (property test);
//! - replaying a log yields an identical state hash on every platform: two folds of a random
//!   log agree, and the golden vectors in `vectors.txt` pin the chain, the events' bytes and the
//!   state hash of a fixed log on each OS CI runs;
//! - a follower one minor behind computes the same chain for an event with an unknown field
//!   (property test).

mod spec;
mod strategies;

use std::path::PathBuf;

use proptest::prelude::*;
use syntropy_cluster::log::payload::{
    ConfigPatch, NodeInfo, Opaque, PlanBody, PlanSummary, RunnerStateKind, WorkloadReport,
};
use syntropy_cluster::log::{
    ClusterEvent, ClusterState, CommandId, EntryHeader, LogEntry, MemberStatus, ReceiveError,
    apply, chain, genesis, state_hash,
};
use syntropy_core::policy::{ConfigValue, KeyPath};
use syntropy_core::{Blake3, Bytes, InstanceId, ModelId, NodeId, PlanDigest, PlanEpoch};
use syntropy_net::ctl::envelope;

fn vectors_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/log/vectors.txt")
}

/// The committed file equals the spec's rendering. `SYNTROPY_BLESS_GOLDEN=1` rewrites it, which
/// is a change to the chain or the encodings and needs review as one.
#[test]
fn the_vectors_file_matches_the_spec() {
    let text = spec::render(&spec::vectors());
    let path = vectors_path();
    if std::env::var("SYNTROPY_BLESS_GOLDEN").as_deref() == Ok("1") {
        std::fs::write(&path, &text).unwrap();
    }
    // `.gitattributes` keeps the file LF everywhere; a checkout that converts it anyway must
    // not fail the comparison.
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n");
    assert_eq!(committed, text, "{} is stale", path.display());
}

/// The library's chain, entries and fold agree with the formula, on the coordinator's side and
/// the follower's.
#[test]
fn the_library_derives_the_vectors() {
    let v = spec::vectors();
    assert_eq!(genesis(&v.cluster), Blake3(v.genesis));
    let mut prev = genesis(&v.cluster);
    let mut leader = ClusterState::new(v.cluster);
    let mut follower = ClusterState::new(v.cluster);
    for step in &v.steps {
        let entry = LogEntry::append(
            &prev,
            step.index,
            step.term,
            1_000 * step.index,
            step.cause,
            step.event.clone(),
        )
        .unwrap();
        assert_eq!(entry.event_bytes(), step.event_bytes, "[{}]", step.index);
        assert_eq!(entry.chain(), Blake3(step.chain), "[{}]", step.index);
        assert_eq!(
            chain(
                &prev,
                step.index,
                step.term,
                step.cause.as_ref(),
                &step.event_bytes
            ),
            Blake3(step.chain)
        );
        let received = LogEntry::receive(&prev, &entry.header(), step.event_bytes.clone()).unwrap();
        assert_eq!(received, entry);
        apply(&mut leader, &entry);
        apply(&mut follower, &received);
        prev = entry.chain();
    }
    assert_eq!(leader, follower);
    // `apply` never reads `at_ms`: the vectors fold with 0, and this with 1,000 per index.
    assert_eq!(state_hash(&leader).unwrap(), v.state_hash);
    assert_eq!(leader.applied, (10, 2));
    assert_eq!(leader.chain, Blake3(v.steps[9].chain));
}

/// A header survives its own encoding, as a log record stores it beside the event bytes.
#[test]
fn a_header_round_trips_through_the_envelope() {
    let entry = LogEntry::append(
        &genesis(&spec::cluster()),
        1,
        1,
        7,
        Some(CommandId {
            origin: NodeId([1; 32]),
            counter: 9,
        }),
        ClusterEvent::NodeFailed {
            node: NodeId([2; 32]),
        },
    )
    .unwrap();
    let bytes = envelope::to_cbor(&entry.header()).unwrap();
    let header: EntryHeader = envelope::from_cbor(&bytes).unwrap();
    assert_eq!(header, entry.header());
}

/// Appends `steps` after `state`'s position, on the coordinator's side.
fn entries(
    state: &ClusterState,
    steps: &[(u64, Option<CommandId>, ClusterEvent)],
) -> Vec<LogEntry> {
    let mut prev = state.chain;
    let mut term = state.applied.1.max(1);
    let mut out = Vec::new();
    for (i, (bump, cause, event)) in steps.iter().enumerate() {
        term += bump;
        let entry = LogEntry::append(
            &prev,
            state.applied.0 + i as u64 + 1,
            term,
            i as u64,
            *cause,
            event.clone(),
        )
        .unwrap();
        prev = entry.chain();
        out.push(entry);
    }
    out
}

/// Adds a field a later minor might add to an event's encoding: the event is a one-entry map
/// from its variant's name to its fields, and the field joins those.
fn with_unknown_field(event_bytes: &[u8], value: i64) -> Vec<u8> {
    let mut decoded: ciborium::Value = envelope::from_cbor(event_bytes).unwrap();
    let ciborium::Value::Map(variant) = &mut decoded else {
        panic!("an event is a map");
    };
    let ciborium::Value::Map(fields) = &mut variant[0].1 else {
        panic!("a base event's fields are a map");
    };
    fields.push((
        ciborium::Value::Text("added_in_a_later_minor".to_owned()),
        ciborium::Value::Integer(value.into()),
    ));
    envelope::to_cbor(&decoded).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `apply` is total: any sequence of well-formed base events folds without a panic, on
    /// both sides, and the two folds agree byte for byte.
    #[test]
    fn apply_is_total_and_every_fold_agrees(
        steps in prop::collection::vec(strategies::step(), 0..48),
    ) {
        let cluster = spec::cluster();
        let mut leader = ClusterState::new(cluster);
        let mut follower = ClusterState::new(cluster);
        let log = entries(&leader, &steps);
        let mut prev = leader.chain;
        for entry in &log {
            apply(&mut leader, entry);
            let received =
                LogEntry::receive(&prev, &entry.header(), entry.event_bytes().to_vec()).unwrap();
            apply(&mut follower, &received);
            prev = entry.chain();
            prop_assert_eq!(leader.applied, (entry.index, entry.term));
            prop_assert_eq!(leader.chain, entry.chain());
        }
        prop_assert_eq!(&leader, &follower);
        prop_assert_eq!(state_hash(&leader).unwrap(), state_hash(&follower).unwrap());
        // The state survives its own encoding, as a snapshot will carry it.
        let decoded: ClusterState =
            envelope::from_cbor(&envelope::to_cbor(&leader).unwrap()).unwrap();
        prop_assert_eq!(state_hash(&decoded).unwrap(), state_hash(&leader).unwrap());
    }

    /// A follower whose build lacks a field a later minor added still decodes the event, keeps
    /// the bytes it received, and computes the coordinator's chain. Re-encoding its decoded
    /// view would drop the field and break the chain, which is why the bytes are kept.
    #[test]
    fn a_follower_one_minor_behind_computes_the_same_chain(
        steps in prop::collection::vec(strategies::step(), 1..16),
        extra in any::<i64>(),
    ) {
        let cluster = spec::cluster();
        let mut prev = genesis(&cluster);
        let mut follower = ClusterState::new(cluster);
        for (i, (term, cause, event)) in steps.into_iter().enumerate() {
            let index = i as u64 + 1;
            let newer = with_unknown_field(&envelope::to_cbor(&event).unwrap(), extra);
            let coordinator_chain = chain(&prev, index, term, cause.as_ref(), &newer);
            let header = EntryHeader { index, term, at_ms: 0, cause, chain: coordinator_chain };
            let entry = LogEntry::receive(&prev, &header, newer.clone()).unwrap();
            prop_assert_eq!(&entry.event, &event);
            prop_assert_eq!(entry.event_bytes(), &newer[..]);
            prop_assert_eq!(entry.chain(), coordinator_chain);
            let reencoded = envelope::to_cbor(&entry.event).unwrap();
            prop_assert_ne!(&reencoded, &newer);
            let reencoded_chain = chain(&prev, index, term, cause.as_ref(), &reencoded);
            prop_assert_ne!(reencoded_chain, coordinator_chain);
            apply(&mut follower, &entry);
            prop_assert_eq!(follower.chain, coordinator_chain);
            prev = coordinator_chain;
        }
    }

    /// `receive` is total over its input: arbitrary bytes give an entry or a classified error.
    #[test]
    fn receive_is_total_over_arbitrary_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..256),
        index in any::<u64>(),
        term in any::<u64>(),
        cause in prop::option::of(strategies::command_id()),
    ) {
        let prev = genesis(&spec::cluster());
        let good = chain(&prev, index, term, cause.as_ref(), &bytes);
        let header = EntryHeader { index, term, at_ms: 0, cause, chain: good };
        match LogEntry::receive(&prev, &header, bytes.clone()) {
            Ok(entry) => prop_assert_eq!(entry.event_bytes(), &bytes[..]),
            Err(ReceiveError::Event(_)) => {}
            Err(other) => prop_assert!(false, "{other}"),
        }
        let wrong = EntryHeader { chain: Blake3([0; 32]), ..header };
        let is_mismatch = matches!(
            LogEntry::receive(&prev, &wrong, bytes),
            Err(ReceiveError::ChainMismatch { .. })
        );
        prop_assert!(is_mismatch);
    }
}

fn node(byte: u8) -> NodeId {
    NodeId([byte; 32])
}

/// Folds `events` after a fresh state, each in term 1 with no cause.
fn fold(events: Vec<ClusterEvent>) -> ClusterState {
    let mut state = ClusterState::new(spec::cluster());
    let steps: Vec<_> = events.into_iter().map(|e| (0, None, e)).collect();
    for entry in entries(&state.clone(), &steps) {
        apply(&mut state, &entry);
    }
    state
}

fn committed(instance: InstanceId, seq: u64) -> ClusterEvent {
    ClusterEvent::PlanCommitted {
        plan: PlanBody::ByDigest {
            digest: PlanDigest(Blake3([seq as u8; 32])),
            len: Bytes(1),
        },
        summary: PlanSummary {
            instance,
            epoch: PlanEpoch { term: 1, seq },
            stages: vec![],
        },
    }
}

#[test]
fn a_plan_that_served_drains_and_one_that_never_did_is_dropped() {
    let i = InstanceId(3);
    let model = ModelId(Blake3([1; 32]));
    let requested = ClusterEvent::InstanceRequested {
        instance: i,
        model,
        policy: Box::default(),
    };
    let e0 = PlanEpoch { term: 1, seq: 0 };
    // Plan 0 serves, plan 1 replaces it: plan 0 drains.
    let state = fold(vec![
        ClusterEvent::CoordinatorElected {
            node: node(1),
            term: 1,
            lease_until_ms: 0,
        },
        requested.clone(),
        committed(i, 0),
        ClusterEvent::InstanceRoutable {
            instance: i,
            epoch: e0,
            context_length: 4096,
        },
        committed(i, 1),
    ]);
    let record = &state.instances[&i];
    assert_eq!(record.current.as_ref().unwrap().epoch().seq, 1);
    assert_eq!(record.retiring.len(), 1);
    assert_eq!(record.retiring[0].epoch(), e0);
    assert_eq!(record.routable.unwrap().epoch, e0);
    assert_eq!(state.alloc.next_plan_seq, 2);
    // Plan 0 never serves: plan 1 drops it.
    let state = fold(vec![requested, committed(i, 0), committed(i, 1)]);
    assert!(state.instances[&i].retiring.is_empty());
}

#[test]
fn a_retired_plan_takes_its_runners_with_it() {
    let i = InstanceId(0);
    let e0 = PlanEpoch { term: 1, seq: 0 };
    let e1 = PlanEpoch { term: 1, seq: 1 };
    let runner = |epoch| ClusterEvent::RunnerStateChanged {
        instance: i,
        node: node(1),
        epoch,
        state: RunnerStateKind::Ready,
    };
    let state = fold(vec![
        ClusterEvent::InstanceRequested {
            instance: i,
            model: ModelId(Blake3([1; 32])),
            policy: Box::default(),
        },
        committed(i, 0),
        runner(e0),
        ClusterEvent::InstanceRoutable {
            instance: i,
            epoch: e0,
            context_length: 1,
        },
        committed(i, 1),
        runner(e1),
        ClusterEvent::PlanRetired {
            instance: i,
            epoch: e0,
        },
    ]);
    let record = &state.instances[&i];
    assert!(record.retiring.is_empty());
    assert_eq!(record.runners.len(), 1);
    assert!(record.runners.contains_key(&(node(1), e1)));
}

#[test]
fn an_election_resets_the_term_counters_but_not_instance_ids() {
    let state = fold(vec![
        ClusterEvent::CoordinatorElected {
            node: node(1),
            term: 1,
            lease_until_ms: 0,
        },
        ClusterEvent::InstanceRequested {
            instance: InstanceId(5),
            model: ModelId(Blake3([1; 32])),
            policy: Box::default(),
        },
        committed(InstanceId(5), 3),
        ClusterEvent::CoordinatorElected {
            node: node(2),
            term: 4,
            lease_until_ms: 0,
        },
    ]);
    assert_eq!(state.alloc.term, 4);
    assert_eq!(state.alloc.next_plan_seq, 0);
    assert_eq!(state.alloc.next_instance, 6);
    assert_eq!(state.max_term, 4);
    let coordinator = state.coordinator.unwrap();
    assert_eq!((coordinator.node, coordinator.term), (node(2), 4));
    assert_eq!(coordinator.elected_at_index, 4);
}

#[test]
fn a_config_patch_sets_then_unsets_and_records_where() {
    let key = KeyPath::from("allow_approximate");
    let other = KeyPath::from("network_scope");
    let patch = |set: Vec<(KeyPath, ConfigValue)>, unset: Vec<KeyPath>| {
        ClusterEvent::ClusterConfigChanged {
            patch: ConfigPatch {
                set,
                unset,
                origin: Opaque::null(),
            },
        }
    };
    let state = fold(vec![
        patch(
            vec![
                (key.clone(), ConfigValue::Bool(true)),
                (other.clone(), ConfigValue::String("lan".into())),
            ],
            vec![],
        ),
        patch(vec![], vec![other.clone()]),
    ]);
    assert_eq!(state.config.keys.len(), 1);
    assert_eq!(state.config.keys[&key].value, ConfigValue::Bool(true));
    assert_eq!(state.config.keys[&key].set_at, (1, 1));
}

#[test]
fn suspicion_is_informational_and_only_marks_an_alive_member() {
    let admitted = |b| ClusterEvent::NodeAdmitted {
        info: NodeInfo { id: node(b) },
    };
    let state = fold(vec![
        admitted(1),
        admitted(2),
        ClusterEvent::NodeSleeping {
            node: node(2),
            incarnation: 7,
        },
        ClusterEvent::NodeSuspected { node: node(1) },
        ClusterEvent::NodeSuspected { node: node(2) },
    ]);
    assert_eq!(state.members[&node(1)].status, MemberStatus::Suspected);
    assert_eq!(state.members[&node(2)].status, MemberStatus::Sleeping);
    assert_eq!(state.members[&node(2)].incarnation, 7);
    // Re-admission makes a member alive again and keeps its incarnation.
    let state = fold(vec![
        admitted(2),
        ClusterEvent::NodeSleeping {
            node: node(2),
            incarnation: 7,
        },
        ClusterEvent::NodeRevoked { node: node(2) },
        admitted(2),
    ]);
    assert_eq!(state.members[&node(2)].status, MemberStatus::Alive);
    assert_eq!(state.members[&node(2)].incarnation, 7);
}

#[test]
fn events_about_absent_subjects_change_only_the_position() {
    let empty = ClusterState::new(spec::cluster());
    let events = vec![
        ClusterEvent::NodeFailed { node: node(9) },
        ClusterEvent::NodeInfoChanged {
            node: node(9),
            info: NodeInfo { id: node(9) },
        },
        committed(InstanceId(9), 0),
        ClusterEvent::InstanceRemoved {
            instance: InstanceId(9),
        },
        ClusterEvent::WorkloadUpdated {
            model: ModelId(Blake3([9; 32])),
            instance: InstanceId(9),
            report: WorkloadReport::default(),
        },
    ];
    let state = fold(events);
    let mut expected = empty.clone();
    expected.applied = state.applied;
    expected.chain = state.chain;
    expected.max_term = state.max_term;
    // `committed` is in term 1, which the counters adopt only after an election.
    assert_eq!(state, expected);
    assert_eq!(state.applied, (5, 1));
}
