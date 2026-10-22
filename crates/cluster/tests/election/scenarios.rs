// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scenarios the model checks. Each one exercises some of the 13 properties and must reach
//! the situations it names, so those properties are not vacuously true; every property is
//! checked in every scenario.

use crate::check::TWO_AT_ONE_TERM;
use crate::model::{Config, flag, seen, side};

fn base(name: &'static str, n: u8) -> Config {
    Config {
        name,
        n,
        role: vec![true; n as usize],
        keyholder: vec![false; n as usize],
        tags: (0..n).map(|i| 10 + i).collect(),
        byzantine: None,
        start_led: true,
        max_counter: 3,
        claims_per_node: 2,
        splits: Vec::new(),
        split_budget: 0,
        loss_budget: 0,
        crashable: Vec::new(),
        crash_budget: 0,
        restart: false,
        originator_ack_counts: false,
        proof_only_heal: false,
        own_endorsement_counts: true,
        commands: Vec::new(),
        revocation: None,
        forge_budget: 0,
        forge_authority: false,
        forge_terms: false,
        partial_partition: false,
        must_reach: 0,
        expected_failure: None,
        state_cap: 50_000_000,
    }
}

/// A split isolates one node, both sides may elect, then the network heals, with one message
/// lost and one command in flight.
pub fn split_heal(n: u8, losses: u8) -> Config {
    Config {
        splits: vec![side(n, &[0]), side(n, &[n - 1])],
        split_budget: 1,
        loss_budget: losses,
        commands: vec![(1, 1)],
        must_reach: seen::HEALED_AFTER_SPLIT_ELECTION | seen::SETTLED_AFTER_HEAL,
        ..base("split-heal", n)
    }
}

/// No coordinator yet; a split into two sides lets both elect at the first counter, then heals.
pub fn equal_counters(n: u8) -> Config {
    let half: Vec<u8> = (0..n / 2).collect();
    Config {
        start_led: false,
        splits: vec![side(n, &half)],
        split_budget: 1,
        max_counter: 2,
        must_reach: seen::SPLIT_ELECTED_EQUAL_COUNTERS
            | seen::HEALED_AFTER_SPLIT_ELECTION
            | seen::SETTLED_AFTER_HEAL,
        ..base("equal-counters", n)
    }
}

/// As [`equal_counters`], with two nodes on different sides sharing a 24-bit tag. Both sides can
/// then elect at one term, and at heal the lower node ID wins the tie, so a node switches
/// coordinators within that term: the first property's expected counterexample, which RFC-0032
/// §7.2 accepts at about 1.2 × 10⁻⁴ per simultaneous split election for 64 nodes.
pub fn tag_collision(n: u8) -> Config {
    let mut cfg = equal_counters(n);
    cfg.name = "tag-collision";
    cfg.tags[0] = cfg.tags[(n / 2) as usize];
    cfg.expected_failure = Some((TWO_AT_ONE_TERM, flag::TAG_COLLISION));
    cfg
}

/// One link to the coordinator is cut; the cut-off node still reaches the others. Nothing is
/// lost and nothing crashes.
pub fn partial_partition(n: u8) -> Config {
    Config {
        splits: vec![vec![(0, n - 1)]],
        split_budget: 1,
        partial_partition: true,
        must_reach: seen::CLAIM_ABORTED,
        ..base("partial-partition", n)
    }
}

/// Every node but one crashes, the coordinator first.
pub fn lone_survivor(n: u8) -> Config {
    Config {
        crashable: (0..n).filter(|&i| i != 1).collect(),
        crash_budget: n - 1,
        must_reach: seen::LONE_SURVIVOR_LEADS,
        ..base("lone-survivor", n)
    }
}

/// Lost beats let a claim gather an `Ack` from a voter whose lease ran out while another voter
/// still hears the coordinator and answers `LeaseLive`.
pub fn aborted_claim(n: u8) -> Config {
    Config {
        loss_budget: 2,
        claims_per_node: 1,
        must_reach: seen::CLAIM_ABORTED_AFTER_ACK,
        ..base("aborted-claim", n)
    }
}

/// The coordinator is byzantine: it appends a forged authority entry, and omits its own
/// revocation, which a keyholder signs and gossips.
pub fn byzantine_coordinator(n: u8) -> Config {
    let mut keyholder = vec![false; n as usize];
    keyholder[1] = true;
    Config {
        byzantine: Some(0),
        keyholder,
        revocation: Some((1, 0)),
        forge_budget: 1,
        forge_authority: true,
        must_reach: seen::FORGED_REFUSED | seen::REVOCATION_OMITTED_BUT_HELD,
        ..base("byzantine-coordinator", n)
    }
}

/// A byzantine member answers the coordinator and its followers with bare terms and forged
/// proofs at a counter above any real one.
pub fn forged_terms(n: u8) -> Config {
    Config {
        byzantine: Some(n - 1),
        forge_budget: 2,
        forge_terms: true,
        must_reach: seen::FORGED_TERM_IGNORED,
        ..base("forged-terms", n)
    }
}

/// A split leaves a lone honest node to elect itself; the byzantine coordinator, still
/// supported, forges an authority entry; at heal its proof wins and its snapshot must be
/// refused.
pub fn byzantine_snapshot(n: u8) -> Config {
    Config {
        byzantine: Some(0),
        splits: vec![side(n, &[n - 1])],
        split_budget: 1,
        forge_budget: 1,
        forge_authority: true,
        must_reach: seen::FORGED_SNAPSHOT_REFUSED,
        ..base("byzantine-snapshot", n)
    }
}

/// A byzantine member without `COORDINATOR` claims and elects itself once the coordinator
/// crashes.
pub fn no_role_claimant(n: u8) -> Config {
    let mut role = vec![true; n as usize];
    role[(n - 1) as usize] = false;
    Config {
        role,
        byzantine: Some(n - 1),
        crashable: vec![0],
        crash_budget: 1,
        must_reach: seen::NO_ROLE_CLAIM_IGNORED,
        ..base("no-role-claimant", n)
    }
}

/// A node cut off from everyone elects itself, unsupported, at higher counters; at heal the
/// supported coordinator wins.
pub fn isolated_self_election(n: u8) -> Config {
    Config {
        splits: vec![side(n, &[n - 1])],
        split_budget: 1,
        max_counter: 4,
        claims_per_node: 3,
        must_reach: seen::SUPPORTED_BEAT_UNSUPPORTED | seen::SETTLED_AFTER_HEAL,
        ..base("isolated-self-election", n)
    }
}

/// Commands from a node that may crash, acknowledged while the coordinator may crash and a
/// message may be lost. Crashed daemons restart with their disks.
pub fn commands_crash(n: u8) -> Config {
    Config {
        commands: vec![(1, n - 1)],
        crashable: vec![0, n - 1],
        crash_budget: 2,
        restart: true,
        loss_budget: 1,
        must_reach: seen::REPLICATED_ACK | seen::SETTLED_CONVERGED,
        ..base("commands-crash", n)
    }
}

/// As [`commands_crash`], but a crash loses the node and its disk for good. Under the reply rule
/// before OQ-NET-33, where the originator's own ack counted, this scenario lost acknowledged
/// commands; under the rule as revised it must not.
pub fn commands_lost(n: u8) -> Config {
    Config {
        restart: false,
        ..commands_crash(n)
    }
    .renamed("commands-lost")
}

/// The bounded 3-node configuration the default tests check: every scenario on 3 nodes, at
/// bounds the dev profile checks in about a minute. Every scenario here runs to exhaustion: about
/// 2.3 million states, 53 s in the dev profile and 8.1 s release-optimised, with a test-process
/// peak of 85 MB [measured, 2026-10-23, DIST-E10-T10's model, AMD Ryzen 9 5950X under WSL2,
/// toolchain 1.99.1, 4 checker threads, with other builds running].
pub fn bounded() -> Vec<Config> {
    vec![
        split_heal(3, 1).without_commands().claims(1).counter(2),
        equal_counters(3).claims(1),
        partial_partition(3),
        lone_survivor(3),
        aborted_claim(3).claims(1).counter(2),
        byzantine_coordinator(3).claims(1).counter(2),
        forged_terms(3).claims(1).counter(2),
        byzantine_snapshot(3).claims(1).counter(2),
        no_role_claimant(3),
        isolated_self_election(3).claims(1),
        commands_crash(3)
            .losses(0)
            .claims(1)
            .counter(2)
            .coordinator_crash_only(),
        commands_lost(3).claims(1).counter(2),
    ]
}

/// The full run: 3 and 4 nodes, with partitions and message loss. Each scenario stops at its
/// `state_cap` of generated states. Ten reach it, so their check is an exploration to that
/// bound, not an exhaustive one: `split-heal`, `equal-counters` and `commands-crash` on 3 and 4
/// nodes, `stale-heal` and `stale-heal-late-support` on 3, and `tag-collision` and
/// `byzantine-snapshot` on 4. The other 18 run to exhaustion. In all: 94 million states, 1061 s
/// release-optimised on one checker thread, with a test-process peak of 171 MB [measured,
/// 2026-10-23, DIST-E10-T10's model, AMD Ryzen 9 5950X under WSL2, toolchain 1.99.1, under
/// `ulimit -v 4000000`, with other builds running].
pub fn full() -> Vec<Config> {
    vec![
        split_heal(3, 0),
        split_heal(3, 2)
            .without_commands()
            .claims(1)
            .counter(2)
            .renamed("split-heal-lossy"),
        equal_counters(3),
        partial_partition(3),
        lone_survivor(3),
        aborted_claim(3),
        byzantine_coordinator(3),
        forged_terms(3),
        byzantine_snapshot(3),
        no_role_claimant(3),
        isolated_self_election(3),
        commands_crash(3).claims(1).counter(2),
        commands_lost(3),
        stale_heal(3),
        stale_heal_one_claim(3),
        stale_heal_late_support(3),
        split_heal(4, 0).without_commands().claims(1).counter(2),
        equal_counters(4).claims(1),
        tag_collision(4).claims(1),
        partial_partition(4),
        lone_survivor(4),
        aborted_claim(4).claims(1).counter(2),
        byzantine_coordinator(4).claims(1).counter(2),
        forged_terms(4).claims(1).counter(2),
        byzantine_snapshot(4).claims(1).counter(2),
        no_role_claimant(4),
        isolated_self_election(4).claims(1),
        commands_crash(4).losses(0).claims(1).counter(2),
    ]
}

impl Config {
    fn renamed(mut self, name: &'static str) -> Config {
        self.name = name;
        self
    }

    fn claims(mut self, claims: u8) -> Config {
        self.claims_per_node = claims;
        self
    }

    fn counter(mut self, max: u8) -> Config {
        self.max_counter = max;
        self
    }

    fn losses(mut self, losses: u8) -> Config {
        self.loss_budget = losses;
        self
    }

    /// Only the first coordinator, node 0, may crash.
    fn coordinator_crash_only(mut self) -> Config {
        self.crashable = vec![0];
        self.crash_budget = 1;
        self
    }

    fn without_commands(mut self) -> Config {
        self.commands.clear();
        self.must_reach &= !seen::REPLICATED_ACK;
        self
    }
}

/// A split cuts the coordinator off alone. The other side elects the command's originator,
/// which commits the command with the other member's ack, then is lost with its disk; the
/// survivor elects itself, unsupported, at the next counter, and the network heals. Under
/// RFC-0032 §7.5's comparison of proofs alone, the cut-off coordinator's older but supported
/// proof won, and the survivor rolled back past the command its own ack made
/// `replicated = true` (OQ-NET-34). Under option (a), which DIST-E10-T10 checks, the survivor's
/// chain holds the newer supported election, so it keeps the command: no failure is expected.
pub fn stale_heal(n: u8) -> Config {
    Config {
        splits: vec![side(n, &[0])],
        split_budget: 1,
        commands: vec![(1, n - 1)],
        crashable: vec![n - 1],
        crash_budget: 1,
        must_reach: seen::HEALED_AFTER_SPLIT_ELECTION
            | seen::REPLICATED_ACK
            | seen::SETTLED_AFTER_HEAL,
        ..base("stale-heal", n)
    }
}

/// As [`stale_heal`], with one claim per node, which is all OQ-NET-34's 15-step trace needs. It
/// runs to exhaustion, so it checks the heal rule completely at these bounds: under §7.5's proof
/// comparison from before OQ-NET-34 (`ELECTION_HEAL_RULE=proof`) it finds that trace's
/// rollback, and under the settled rule nothing. `stale-heal` stops at its cap first, and under
/// the old comparison it no longer reaches the trace before then. The cap is raised so that the
/// run stays exhaustive.
pub fn stale_heal_one_claim(n: u8) -> Config {
    Config {
        claims_per_node: 1,
        state_cap: 100_000_000,
        ..stale_heal(n)
    }
    .renamed("stale-heal-one-claim")
}

/// As [`stale_heal`], with one message loss and one claim per node. Losing node 1's ack lets
/// node 2 win at counter 2 unsupported, so its `SignedElection` carries no `acks`; node 1 then
/// endorses it on the first append, which also holds the command, and that ack makes the
/// command `replicated = true`. The election is supported only by that later endorsement, which
/// the log does not hold (DIST-E10-T10).
pub fn stale_heal_late_support(n: u8) -> Config {
    Config {
        loss_budget: 1,
        claims_per_node: 1,
        ..stale_heal(n)
    }
    .renamed("stale-heal-late-support")
}
