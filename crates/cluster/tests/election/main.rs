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
/// A scenario that names an expected counterexample must produce exactly that one: the property
/// fails as the plan predicts, pending the open question that records it.
fn check_scenario(cfg: Config) -> usize {
    let (name, n, cap, expected) = (cfg.name, cfg.n, cfg.state_cap, cfg.expected_failure);
    let properties = ElectionModel { cfg: cfg.clone() }.properties();
    let checker = ElectionModel { cfg }
        .checker()
        .threads(threads())
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
            Expectation::Always | Expectation::Eventually if Some(p.name) == expected => {
                if found.is_none() {
                    failures.push(format!(
                        "{name}: the expected counterexample to \"{}\" was not found",
                        p.name
                    ));
                }
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
fn check_all(configs: Vec<Config>) {
    let only = std::env::var("ELECTION_SCENARIO").ok();
    let total: usize = configs
        .into_iter()
        .filter(|c| {
            only.as_ref()
                .is_none_or(|o| *o == format!("{}-{}", c.name, c.n))
        })
        .map(check_scenario)
        .sum();
    eprintln!("{total} states in all");
}

#[test]
fn the_bounded_three_node_configuration_satisfies_every_property() {
    check_all(scenarios::bounded());
}

#[test]
#[ignore = "full run: cargo xtask test election-model"]
fn three_and_four_nodes_with_partitions_and_loss_satisfy_every_property() {
    check_all(scenarios::full());
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

/// The shortest counterexample to "every command acknowledged with `replicated = true` …
/// appears in the converged log" when nodes are lost with their disks (OQ-NET-33): the
/// originator's own ack makes the reply `replicated = true`, then the coordinator and the
/// originator are both lost.
const ORIGINATOR_ACK_TRACE: [&str; 11] = [
    "Deliver(Command { from: 2, to: 0, id: 1 })",
    "Replicate(0, 2)",
    "Expire(2)",
    "Claim(2)",
    "Deliver(Claim { from: 2, to: 1, term: Term { counter: 2, tag: 12 } })",
    "Deliver(Append { from: 0, to: 2, term: Term { counter: 1, tag: 10 }, log: [Elected { node: 0, \
     term: Term { counter: 1, tag: 10 } }, Command { id: 1 }], proof: Proof { node: 0, term: Term \
     { counter: 1, tag: 10 }, sig_ok: true, support: 6 } })",
    "Crash(0)",
    "Expire(1)",
    "Claim(1)",
    "Crash(2)",
    "Win(1)",
];

#[test]
fn the_originators_own_ack_loses_a_replicated_command_with_both_disks() {
    let model = ElectionModel {
        cfg: scenarios::commands_lost(3),
    };
    let mut state = model.initial();
    for step in ORIGINATOR_ACK_TRACE {
        let mut actions = Vec::new();
        model.actions(&state, &mut actions);
        let action = actions
            .into_iter()
            .find(|a| format!("{a:?}") == step)
            .unwrap_or_else(|| panic!("not enabled: {step}"));
        state = model
            .next_state(&state, action)
            .expect("a step changes the state");
    }
    // Node 0 replied `replicated = true` on node 2's ack; nodes 0 and 2 are gone; node 1 leads
    // a settled cluster whose log lacks the command.
    assert!(state.acked_replicated.contains(&1));
    assert!(!state.nodes[0].alive && !state.nodes[2].alive);
    assert!(check::settled(&model, &state));
    assert_eq!(check::converged(&model, &state), Some(1));
    assert!(!state.nodes[1].has_command(1));
    let (name, holds) = check::PROPERTIES[12];
    assert!(!holds(&model, &state), "{name}");
}
