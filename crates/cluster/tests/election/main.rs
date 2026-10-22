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
/// checks only that one.
fn check_all(configs: Vec<Config>, threads: usize) {
    let only = std::env::var("ELECTION_SCENARIO").ok();
    let total: usize = configs
        .into_iter()
        .filter(|c| {
            only.as_ref()
                .is_none_or(|o| *o == format!("{}-{}", c.name, c.n))
        })
        .map(|cfg| check_scenario(cfg, threads))
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
     term: Term { counter: 1, tag: 10 } }, Command { id: 1 }], proof: Proof { node: 0, term: Term \
     { counter: 1, tag: 10 }, sig_ok: true, support: 6 } })",
    "Crash(0)",
    "Expire(1)",
    "Claim(1)",
    "Crash(2)",
    "Win(1)",
];

/// Replays `trace` in `cfg` and returns the model and its final state.
fn replay(cfg: Config, trace: &[&str]) -> (ElectionModel, model::State) {
    let model = ElectionModel { cfg };
    let mut state = model.initial();
    for step in trace {
        let mut actions = Vec::new();
        model.actions(&state, &mut actions);
        let action = actions
            .into_iter()
            .find(|a| format!("{a:?}") == *step)
            .unwrap_or_else(|| panic!("not enabled: {step}"));
        state = model
            .next_state(&state, action)
            .expect("a step changes the state");
    }
    (model, state)
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
     term: Term { counter: 1, tag: 10 } }, Elected { node: 2, term: Term { counter: 2, tag: 12 } }, \
     Command { id: 1 }], proof: Proof { node: 2, term: Term { counter: 2, tag: 12 }, sig_ok: true, \
     support: 2 } })",
    "Crash(2)",
    "Expire(1)",
    "Claim(1)",
    "Heal",
    "Replicate(0, 1)",
    "Win(1)",
    "Deliver(Append { from: 0, to: 1, term: Term { counter: 1, tag: 10 }, log: [Elected { node: 0, \
     term: Term { counter: 1, tag: 10 } }], proof: Proof { node: 0, term: Term { counter: 1, tag: \
     10 }, sig_ok: true, support: 6 } })",
];

/// The counterexample OQ-NET-34 holds: the command is lost, and the cause is the one
/// `stale-heal` expects. DIST-E10-T10 checks whether the rule OQ-NET-34 recommends removes it.
#[test]
fn a_stale_supported_proof_rolls_a_survivor_back_past_its_own_ack() {
    let (name, holds) = check::PROPERTIES[12];
    // Without the scenario's expectation, so the property is judged strictly.
    let mut cfg = scenarios::stale_heal(3);
    let expected = cfg.expected_failure.take();
    assert_eq!(
        expected,
        Some((check::COMMANDS_SURVIVE, model::flag::ROLLBACK_PAST_OWN_ACK))
    );
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
