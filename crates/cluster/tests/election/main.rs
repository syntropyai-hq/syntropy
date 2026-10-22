// SPDX-License-Identifier: MIT OR Apache-2.0

//! DIST-E10-T08: the explicit-state model check of coordinator election, the lease, replication
//! and split heal, against RFC-0032 §7 (OQ-NET-08, correctness part).
//!
//! [`model`] is the protocol, [`check`] the 13 properties of RFC-0032 Testing's "Election safety
//! and liveness" row and the transition oracle, and [`scenarios`] the configurations. Every
//! property is checked in every scenario by stateright's depth-first search, which keeps a
//! fingerprint per visited state rather than the states themselves.
//!
//! The default tests check the bounded 3-node configuration. The full run on 3 and 4 nodes with
//! partitions and message loss is ignored here and runs release-optimised with
//! `cargo xtask test election-model`; see [`scenarios::full`] for its bounds and measured cost.

mod check;
mod model;
mod scenarios;

use stateright::{Checker, Expectation, Model};

use model::{Config, ElectionModel};

/// At most four threads: CI runners have few cores, and the host is shared with other jobs.
fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |p| p.get().min(4))
}

/// Checks every property in one scenario and returns the number of distinct states visited.
///
/// A scenario that expects a property to fail names the cause the plan predicts; the property
/// excuses only states where that cause occurred, and [`check::EXPECTED`] requires the expected
/// counterexample to occur with it.
fn check_scenario(cfg: Config, threads: usize) -> usize {
    let (name, n, cap) = (cfg.name, cfg.n, cfg.state_cap);
    let properties = ElectionModel { cfg: cfg.clone() }.properties();
    let checker = ElectionModel { cfg }
        .checker()
        .threads(threads)
        .target_state_count(cap)
        .spawn_dfs()
        .join();
    let states = checker.unique_state_count();
    eprintln!(
        "{name} ({n} nodes): {states} states, {} generated (cap {cap}), depth {}",
        checker.state_count(),
        checker.max_depth()
    );
    let mut failures = Vec::new();
    for p in properties {
        let found = checker.discovery(p.name);
        match p.expectation {
            Expectation::Sometimes if found.is_none() => {
                failures.push(format!("{name}: \"{}\" was never reached", p.name));
            }
            Expectation::Always | Expectation::Eventually => {
                if let Some(path) = found {
                    failures.push(format!(
                        "{name}: \"{}\" fails after {:#?}",
                        p.name,
                        path.into_actions()
                    ));
                }
            }
            Expectation::Sometimes => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    states
}

/// Checks each scenario in turn. `ELECTION_SCENARIO=<name>-<nodes>`, such as `split-heal-4`,
/// checks only that one. `ELECTION_HEAL_RULE` picks the heal comparison, to compare the rules
/// DIST-E10-T10 weighed: `proof` is §7.5 as written before OQ-NET-34, `acks` is option (a) with a
/// past election's support read from its `SignedElection`'s `acks` alone, and `chain`, the
/// default, is option (a) with a node's own endorsements counted too.
fn check_all(configs: Vec<Config>, threads: usize) {
    let only = std::env::var("ELECTION_SCENARIO").ok();
    let rule = std::env::var("ELECTION_HEAL_RULE").unwrap_or_else(|_| "chain".to_owned());
    assert!(
        ["proof", "acks", "chain"].contains(&rule.as_str()),
        "ELECTION_HEAL_RULE is proof, acks or chain, not {rule}"
    );
    let total: usize = configs
        .into_iter()
        .filter(|c| {
            only.as_ref()
                .is_none_or(|o| *o == format!("{}-{}", c.name, c.n))
        })
        .map(|mut cfg| {
            cfg.proof_only_heal = rule == "proof";
            cfg.own_endorsement_counts = rule == "chain";
            check_scenario(cfg, threads)
        })
        .sum();
    eprintln!("{total} states in all");
}

#[test]
fn the_bounded_three_node_configuration_satisfies_every_property() {
    // Every bounded scenario runs to exhaustion, so the thread count changes only the time.
    check_all(scenarios::bounded(), threads());
}

#[test]
#[ignore = "full run: cargo xtask test election-model"]
fn three_and_four_nodes_with_partitions_and_loss_satisfy_every_property() {
    // One thread: a scenario stopped at its cap then explores the same states in the same order
    // on every run, so its result is reproducible.
    check_all(scenarios::full(), 1);
}

#[test]
fn every_property_of_the_testing_row_is_a_named_case() {
    let names: Vec<&str> = check::PROPERTIES.iter().map(|(name, _)| *name).collect();
    assert_eq!(names.len(), 13);
    let model = ElectionModel {
        cfg: scenarios::split_heal(3, 1),
    };
    let registered: Vec<&str> = model.properties().iter().map(|p| p.name).collect();
    for name in names {
        assert!(registered.contains(&name), "{name}");
    }
}

/// An 8-step counterexample to "every command acknowledged with `replicated = true` … appears in
/// the converged log" under the reply rule before OQ-NET-33, when nodes are lost with
/// their disks: the originator's own ack made the reply `replicated = true`, then the coordinator
/// and the originator were both lost.
const ORIGINATOR_ACK_TRACE: [&str; 8] = [
    "Deliver(Command { from: 2, to: 0, id: 1 })",
    "Replicate(0, 2)",
    "Deliver(Append { from: 0, to: 2, term: Term { counter: 1, tag: 10 }, log: [Elected { node: 0, \
     term: Term { counter: 1, tag: 10 }, acks: 6 }, Command { id: 1 }], proof: Proof { node: 0, \
     term: Term { counter: 1, tag: 10 }, sig_ok: true, support: 6 } })",
    "Crash(0)",
    "Expire(1)",
    "Claim(1)",
    "Crash(2)",
    "Win(1)",
];

/// Replays `trace` in `cfg` and returns the model and its final state.
fn replay(cfg: Config, trace: &[&str]) -> (ElectionModel, model::State) {
    try_replay(cfg, trace).unwrap_or_else(|step| panic!("not enabled: {step}"))
}

/// Replays `trace` in `cfg`, or returns the first step that is not enabled.
fn try_replay<'t>(
    cfg: Config,
    trace: &[&'t str],
) -> Result<(ElectionModel, model::State), &'t str> {
    let model = ElectionModel { cfg };
    let mut state = model.initial();
    for step in trace {
        let mut actions = Vec::new();
        model.actions(&state, &mut actions);
        let Some(action) = actions.into_iter().find(|a| format!("{a:?}") == *step) else {
            return Err(step);
        };
        state = model
            .next_state(&state, action)
            .expect("a step changes the state");
    }
    Ok((model, state))
}

/// Replays [`ORIGINATOR_ACK_TRACE`] and returns the model and its final state.
fn replay_originator_ack_trace(originator_ack_counts: bool) -> (ElectionModel, model::State) {
    let mut cfg = scenarios::commands_lost(3);
    cfg.originator_ack_counts = originator_ack_counts;
    let (model, state) = replay(cfg, &ORIGINATOR_ACK_TRACE);
    // Nodes 0 and 2 are gone, and node 1 leads a settled cluster whose log lacks the command.
    assert!(!state.nodes[0].alive && !state.nodes[2].alive);
    assert!(check::settled(&model, &state));
    assert_eq!(check::converged(&model, &state), Some(1));
    assert!(!state.nodes[1].has_command(1));
    (model, state)
}

/// Why OQ-NET-33 chose option (a). Under the old rule the originator's own ack made the reply
/// `replicated = true`, and the command was lost with the coordinator's and the originator's
/// disks. Under the rule as revised, the same steps never answer `replicated = true`, so nothing
/// acknowledged is lost.
#[test]
fn the_originators_own_ack_no_longer_makes_a_command_replicated() {
    let (name, holds) = check::PROPERTIES[12];
    let (old_rule, state) = replay_originator_ack_trace(true);
    assert!(state.acked_replicated.contains_key(&1));
    assert!(!holds(&old_rule, &state), "old rule: {name}");
    let (new_rule, state) = replay_originator_ack_trace(false);
    assert!(state.acked_replicated.is_empty());
    assert!(holds(&new_rule, &state), "revised rule: {name}");
}

/// OQ-NET-34's 15-step trace in `stale-heal`, under §6.3 as revised by OQ-NET-33. A split cuts
/// coordinator 0 off alone before node 2 sends it its command. Node 2 wins at counter 2 with
/// node 1's ack, commits the command, and node 1's ack makes it `replicated = true`. Node 2 is
/// lost with its disk, node 1 claims at counter 3, the network heals, and node 1 wins alone,
/// unsupported. Node 0's append then carries its supported proof at counter 1, which wins under
/// §7.5's heal order, so node 1 rolls back past the command its own ack made `replicated = true`.
const STALE_HEAL_TRACE: [&str; 15] = [
    "Split(0)",
    "Expire(1)",
    "Expire(2)",
    "Claim(2)",
    "Deliver(Claim { from: 2, to: 1, term: Term { counter: 2, tag: 12 } })",
    "Win(2)",
    "Replicate(2, 1)",
    "Deliver(Append { from: 2, to: 1, term: Term { counter: 2, tag: 12 }, log: [Elected { node: 0, \
     term: Term { counter: 1, tag: 10 }, acks: 6 }, Elected { node: 2, term: Term { counter: 2, \
     tag: 12 }, acks: 2 }, Command { id: 1 }], proof: Proof { node: 2, term: Term { counter: 2, \
     tag: 12 }, sig_ok: true, support: 2 } })",
    "Crash(2)",
    "Expire(1)",
    "Claim(1)",
    "Heal",
    "Replicate(0, 1)",
    "Win(1)",
    "Deliver(Append { from: 0, to: 1, term: Term { counter: 1, tag: 10 }, log: [Elected { node: 0, \
     term: Term { counter: 1, tag: 10 }, acks: 6 }], proof: Proof { node: 0, term: Term { counter: \
     1, tag: 10 }, sig_ok: true, support: 6 } })",
];

/// The counterexample that opened OQ-NET-34, under §7.5's heal comparison as written before it:
/// the command is lost, through the rollback past node 1's own ack.
#[test]
fn a_stale_supported_proof_rolls_a_survivor_back_past_its_own_ack() {
    let (name, holds) = check::PROPERTIES[12];
    let mut cfg = scenarios::stale_heal(3);
    assert_eq!(cfg.expected_failure, None);
    cfg.proof_only_heal = true;
    let (model, state) = replay(cfg, &STALE_HEAL_TRACE);
    assert_eq!(state.acked_replicated.get(&1), Some(&model::bit(1)));
    assert!(!state.nodes[2].alive);
    assert!(
        state
            .nodes
            .iter()
            .filter(|n| n.alive)
            .all(|n| !n.has_command(1))
    );
    assert_ne!(state.flags & model::flag::ROLLBACK_PAST_OWN_ACK, 0);
    assert!(!holds(&model, &state), "{name}");
}

/// The same 15 steps under OQ-NET-34's option (a), which DIST-E10-T10 checks. Node 1's chain holds
/// node 2's election at counter 2, supported by node 1's ack, which beats node 0's proof at
/// counter 1, so node 0's append is refused and node 1 keeps the command. Its fencing reply
/// carries that election, so node 0 learns its own was succeeded, steps down and follows node 1.
#[test]
fn a_chain_with_a_newer_supported_election_keeps_its_acknowledged_command() {
    let (model, state) = replay(scenarios::stale_heal(3), &STALE_HEAL_TRACE);
    assert!(state.nodes[1].has_command(1));
    assert!(state.nodes[1].is_coordinator(1));
    assert_eq!(state.flags & model::flag::ROLLBACK_PAST_OWN_ACK, 0);
    let fence = state
        .net
        .iter()
        .find(|m| matches!(m, model::Msg::Fence { from: 1, to: 0, .. }))
        .expect("node 1 fences node 0's append")
        .clone();
    let model::Msg::Fence { chain, .. } = &fence else {
        unreachable!()
    };
    assert_eq!(
        chain.map(|c| (c.node, c.term.counter)),
        Some((2, 2)),
        "the fence carries node 2's supported election"
    );
    let next = model
        .next_state(&state, model::Action::Deliver(fence))
        .expect("the fence changes node 0");
    assert_eq!(next.nodes[0].coordinator, Some(1), "node 0 follows node 1");
    assert!(!next.nodes[0].is_coordinator(0));
    // Node 1's own election is unsupported and node 0's was supported, so the strict wording of
    // "an unsupported self-election never deposes a supported coordinator" would count this. The
    // property as restated for option (a) does not: node 1's chain holds a supported election
    // that beats node 0's.
    assert_eq!(next.flags & model::flag::UNSUPPORTED_DEPOSED_SUPPORTED, 0);
}

/// The probe's counterexample under option (a) with a past election's support read from its
/// `SignedElection`'s `acks` alone, in `stale-heal-late-support` (DIST-E10-T10). Node 1's ack of
/// node 2's claim is lost, so node 2 wins at counter 2 with no `acks`; node 1 endorses it on the
/// first append, which holds the command, and that ack makes the command `replicated = true`.
/// Later the log still shows node 2's election as unsupported, so nothing in node 1's chain beats
/// node 0's older supported proof, and the command is lost with node 2's disk.
const ACKS_ONLY_TRACE: [&str; 33] = [
    "Split(0)",
    "Expire(2)",
    "Claim(2)",
    "Drop(Claim { from: 2, to: 1, term: Term { counter: 2, tag: 12 } })",
    "Expire(1)",
    "Claim(1)",
    "Heal",
    "Resubmit(2, 1)",
    "Replicate(0, 2)",
    "Replicate(0, 1)",
    "Deliver(Claim { from: 1, to: 2, term: Term { counter: 2, tag: 11 } })",
    "Win(2)",
    "Replicate(2, 1)",
    "Replicate(2, 0)",
    "Deliver(Append { from: 2, to: 1, term: Term { counter: 2, tag: 12 }, log: [Elected { node: \
     0, term: Term { counter: 1, tag: 10 }, acks: 6 }, Elected { node: 2, term: Term { counter: \
     2, tag: 12 }, acks: 0 }, Command { id: 1 }], proof: Proof { node: 2, term: Term { counter: \
     2, tag: 12 }, sig_ok: true, support: 0 } })",
    "Replicate(2, 1)",
    "Deliver(Append { from: 2, to: 0, term: Term { counter: 2, tag: 12 }, log: [Elected { node: \
     0, term: Term { counter: 1, tag: 10 }, acks: 6 }, Elected { node: 2, term: Term { counter: \
     2, tag: 12 }, acks: 0 }, Command { id: 1 }], proof: Proof { node: 2, term: Term { counter: \
     2, tag: 12 }, sig_ok: true, support: 0 } })",
    "Replicate(2, 0)",
    "Deliver(Fence { from: 0, to: 2, proof: Some(Proof { node: 0, term: Term { counter: 1, tag: \
     10 }, sig_ok: true, support: 6 }), chain: Some(Proof { node: 0, term: Term { counter: 1, \
     tag: 10 }, sig_ok: true, support: 6 }) })",
    "Deliver(Append { from: 0, to: 2, term: Term { counter: 1, tag: 10 }, log: [Elected { node: \
     0, term: Term { counter: 1, tag: 10 }, acks: 6 }], proof: Proof { node: 0, term: Term { \
     counter: 1, tag: 10 }, sig_ok: true, support: 6 } })",
    "Crash(2)",
    "Expire(1)",
    "Deliver(Fence { from: 2, to: 0, proof: Some(Proof { node: 2, term: Term { counter: 2, tag: \
     12 }, sig_ok: true, support: 2 }), chain: Some(Proof { node: 2, term: Term { counter: 2, \
     tag: 12 }, sig_ok: true, support: 2 }) })",
    "Claim(0)",
    "Deliver(Command { from: 2, to: 0, id: 1 })",
    "Deliver(Append { from: 0, to: 1, term: Term { counter: 1, tag: 10 }, log: [Elected { node: \
     0, term: Term { counter: 1, tag: 10 }, acks: 6 }], proof: Proof { node: 0, term: Term { \
     counter: 1, tag: 10 }, sig_ok: true, support: 6 } })",
    "Expire(1)",
    "Deliver(Append { from: 2, to: 1, term: Term { counter: 2, tag: 12 }, log: [Elected { node: \
     0, term: Term { counter: 1, tag: 10 }, acks: 6 }, Elected { node: 2, term: Term { counter: \
     2, tag: 12 }, acks: 0 }, Command { id: 1 }], proof: Proof { node: 2, term: Term { counter: \
     2, tag: 12 }, sig_ok: true, support: 2 } })",
    "Deliver(Claim { from: 0, to: 1, term: Term { counter: 2, tag: 10 } })",
    "Win(0)",
    "Replicate(0, 1)",
    "Deliver(Append { from: 2, to: 0, term: Term { counter: 2, tag: 12 }, log: [Elected { node: \
     0, term: Term { counter: 1, tag: 10 }, acks: 6 }, Elected { node: 2, term: Term { counter: \
     2, tag: 12 }, acks: 0 }, Command { id: 1 }], proof: Proof { node: 2, term: Term { counter: \
     2, tag: 12 }, sig_ok: true, support: 2 } })",
    "Deliver(Append { from: 0, to: 1, term: Term { counter: 2, tag: 10 }, log: [Elected { node: \
     0, term: Term { counter: 1, tag: 10 }, acks: 6 }, Elected { node: 0, term: Term { counter: \
     2, tag: 10 }, acks: 2 }], proof: Proof { node: 0, term: Term { counter: 2, tag: 10 }, \
     sig_ok: true, support: 2 } })",
];

/// Why a node counts its own `Endorse`: under option (a) with support read from `acks` alone, the
/// probe's trace loses a command that node 1's own ack made `replicated = true`. With node 1's
/// endorsement counted, as the settled rule does, the same steps never get that far.
#[test]
fn a_late_endorsement_counts_as_support() {
    let (name, holds) = check::PROPERTIES[12];
    let mut acks_only = scenarios::stale_heal_late_support(3);
    acks_only.own_endorsement_counts = false;
    let (model, state) = replay(acks_only, &ACKS_ONLY_TRACE);
    assert!(!state.nodes[2].alive);
    assert_ne!(
        state.acked_replicated.get(&1).copied().unwrap_or(0) & model::bit(1),
        0,
        "node 1's ack made the command replicated = true"
    );
    assert!(!holds(&model, &state), "acks alone: {name}");
    match try_replay(scenarios::stale_heal_late_support(3), &ACKS_ONLY_TRACE) {
        Err(_) => {}
        Ok((model, state)) => assert!(holds(&model, &state), "with endorsements: {name}"),
    }
}
