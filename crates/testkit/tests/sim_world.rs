// SPDX-License-Identifier: MIT OR Apache-2.0

//! The simulator's world core (QUAL-E8-T01; RFC-0060 §8.2, §8.6; ADR-0600 decisions 5–6).
//!
//! - One scenario and one seed give one trace hash, in this process and in a second one.
//! - Two seeds order three events scheduled at one instant differently.
//! - A failed check stops the run with the seed, the event index and the trace prefix.

use std::collections::BTreeSet;
use std::process::Command;

use syntropy_core::{Blake3, Clock, Entropy, MonoTime, Nanos};
use syntropy_testkit::sim::{
    Checker, Env, Found, Hex, Model, RunOutcome, Schedule, SimWorld, TraceEvent, Violation,
};

const MS: u64 = 1_000_000;

fn ms(n: u64) -> MonoTime {
    MonoTime(n * MS)
}

/// A ping between two of four nodes; each delivery sends up to two more after random delays of
/// 0 to 2 ms, so many land at one instant.
#[derive(Clone, Copy, Debug)]
struct Ping {
    from: u8,
    to: u8,
    hop: u32,
}

impl TraceEvent for Ping {
    fn trace_bytes(&self, out: &mut Vec<u8>) {
        out.push(self.from);
        out.push(self.to);
        out.extend_from_slice(&self.hop.to_le_bytes());
    }
}

const NODES: u8 = 4;
const HOPS: u32 = 6;

struct Gossip {
    env: Env,
    received: [u64; NODES as usize],
    /// (instant, from, to, hop) in processing order.
    log: Vec<(u64, u8, u8, u32)>,
}

impl Gossip {
    fn total(&self) -> u64 {
        self.received.iter().sum()
    }
}

impl Model for Gossip {
    type Event = Ping;

    fn handle(&mut self, now: MonoTime, ping: Ping, schedule: &mut Schedule<'_, Ping>) {
        assert_eq!(
            self.env.clock.now(),
            now,
            "the model sees the world's clock"
        );
        self.received[ping.to as usize] += 1;
        self.log.push((now.0, ping.from, ping.to, ping.hop));
        if ping.hop == HOPS {
            return;
        }
        for _ in 0..2 {
            let to = (schedule.entropy().u64() % u64::from(NODES)) as u8;
            let delay = Nanos(schedule.entropy().u64() % 3 * MS);
            schedule.after(
                delay,
                Ping {
                    from: ping.to,
                    to,
                    hop: ping.hop + 1,
                },
            );
        }
    }
}

/// The scenario: two pings at the start, run for 50 ms.
fn gossip(seed: u64) -> SimWorld<Gossip> {
    let mut world = SimWorld::new(seed, |env| Gossip {
        env,
        received: [0; NODES as usize],
        log: Vec::new(),
    });
    world.schedule(
        ms(0),
        Ping {
            from: 0,
            to: 1,
            hop: 0,
        },
    );
    world.schedule(
        ms(0),
        Ping {
            from: 0,
            to: 2,
            hop: 0,
        },
    );
    world
}

fn run_gossip(seed: u64) -> (Blake3, u64) {
    let mut world = gossip(seed);
    assert_eq!(
        world.run_until(ms(50)),
        RunOutcome::Reached {
            at: ms(50),
            events: world.events()
        }
    );
    assert!(
        world.events() > 100,
        "the scenario is busy: {}",
        world.events()
    );
    (world.trace_hash(), world.events())
}

const SEEDS: [u64; 3] = [1, 7, 0x5eed_5eed];

/// Set in the child process of the two-process test.
const CHILD: &str = "SYNTROPY_SIM_TRACE_CHILD";

#[test]
fn one_seed_gives_one_trace_hash_in_this_process() {
    for seed in SEEDS {
        assert_eq!(run_gossip(seed), run_gossip(seed), "seed {seed}");
    }
    let hashes: BTreeSet<Blake3> = SEEDS.iter().map(|&seed| run_gossip(seed).0).collect();
    assert_eq!(hashes.len(), SEEDS.len(), "different seeds, different runs");
}

/// The child half of the next test: prints each seed's trace hash. It does nothing in a normal
/// test run.
#[test]
fn child_prints_trace_hashes() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    for seed in SEEDS {
        let (hash, events) = run_gossip(seed);
        println!("sim-trace {seed} {events} {}", Hex(&hash));
    }
}

#[test]
fn one_seed_gives_one_trace_hash_in_two_processes() {
    let output = Command::new(std::env::current_exe().expect("the test binary"))
        .args([
            "--exact",
            "child_prints_trace_hashes",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(CHILD, "1")
        .output()
        .expect("the child process runs");
    assert!(output.status.success(), "child failed: {output:?}");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8");
    let lines: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.find("sim-trace ").map(|start| &line[start..]))
        .collect();
    assert_eq!(lines.len(), SEEDS.len(), "child output: {stdout}");
    for (seed, line) in SEEDS.into_iter().zip(lines) {
        let (hash, events) = run_gossip(seed);
        let here = format!("sim-trace {seed} {events} {}", Hex(&hash));
        println!("this process:  {here}\nchild process: {line}");
        assert_eq!(line, here, "seed {seed}");
    }
}

/// Records the labels of the events it handles, in order.
struct Order(Vec<char>);

#[derive(Clone, Copy, Debug)]
struct Label(char);

impl TraceEvent for Label {
    fn trace_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.0.to_string().as_bytes());
    }
}

impl Model for Order {
    type Event = Label;

    fn handle(&mut self, _now: MonoTime, label: Label, _schedule: &mut Schedule<'_, Label>) {
        self.0.push(label.0);
    }
}

/// `A`, `B` and `C` at 1 ms, with `D` before them and `E` after.
fn simultaneous(seed: u64) -> String {
    let mut world = SimWorld::new(seed, |_| Order(Vec::new()));
    world.schedule(ms(2), Label('E'));
    for label in ['A', 'B', 'C'] {
        world.schedule(ms(1), Label(label));
    }
    world.schedule(MonoTime(MS / 2), Label('D'));
    world.run_until(ms(3));
    world.model().0.iter().collect()
}

#[test]
fn seeds_order_simultaneous_events_differently() {
    let orders: Vec<String> = (0..64).map(simultaneous).collect();
    for (seed, order) in orders.iter().enumerate() {
        assert_eq!(*order, simultaneous(seed as u64), "seed {seed} replays");
        assert!(
            order.starts_with('D') && order.ends_with('E'),
            "instants come first: seed {seed} gave {order}"
        );
    }
    let first = &orders[0];
    let other = orders
        .iter()
        .position(|order| order != first)
        .expect("some seed orders A, B and C differently from seed 0");
    println!("seed 0: {first}, seed {other}: {}", orders[other]);
    let distinct: BTreeSet<&String> = orders.iter().collect();
    assert_eq!(
        distinct.len(),
        6,
        "64 seeds reach all six orders of A, B and C: {distinct:?}"
    );
}

/// Fails after the eleventh ping.
struct AtMostTenPings;

impl Checker<Gossip> for AtMostTenPings {
    fn name(&self) -> &str {
        "at_most_ten_pings"
    }

    fn after_event(&mut self, model: &Gossip, _now: MonoTime) -> Result<(), String> {
        match model.total() {
            total if total > 10 => Err(format!("{total} pings delivered")),
            _ => Ok(()),
        }
    }
}

fn violated(outcome: RunOutcome) -> Violation {
    match outcome {
        RunOutcome::Violated(violation) => violation,
        other => panic!("expected a violation, got {other:?}"),
    }
}

#[test]
fn a_safety_violation_stops_the_run_with_its_seed_index_and_trace_prefix() {
    let mut world = gossip(7);
    world.add_checker(AtMostTenPings);
    let violation = violated(world.run_until(ms(50)));
    assert_eq!(violation.checker, "at_most_ten_pings");
    assert_eq!(violation.message, "11 pings delivered");
    assert_eq!(violation.seed, 7);
    assert_eq!(violation.found, Found::AfterEvent { index: 10 });
    assert_eq!(violation.events, 11);
    // The run stopped at the event: nothing after it ran, and the clock stayed there.
    assert_eq!(world.events(), 11);
    assert_eq!(world.now(), violation.at);
    assert_eq!(
        world.model().log.last().map(|entry| entry.0),
        Some(violation.at.0)
    );
    assert_eq!(violation.trace_prefix, world.trace_hash());
    // The prefix is the trace of the first 11 events of the unchecked run.
    let mut unchecked = gossip(7);
    unchecked.run_until(ms(50));
    assert_eq!(&unchecked.model().log[..11], &world.model().log[..]);
    assert_ne!(violation.trace_prefix, unchecked.trace_hash());
    // A replay with the same seed stops at the same place with the same prefix.
    let mut replay = gossip(7);
    replay.add_checker(AtMostTenPings);
    assert_eq!(violated(replay.run_until(ms(50))), violation);
    // A stopped world stays stopped.
    assert_eq!(violated(world.run_until(ms(60))), violation);
    assert_eq!(world.events(), 11);
    let text = violation.to_string();
    for part in [
        "`at_most_ten_pings`",
        "after event 10",
        "seed 7",
        "11 events",
    ] {
        assert!(text.contains(part), "{text}");
    }
    assert!(
        text.contains(&Hex(&violation.trace_prefix).to_string()),
        "{text}"
    );
}

/// Wants every node to have received a ping by each milestone.
struct EveryNodeReached;

impl Checker<Gossip> for EveryNodeReached {
    fn name(&self) -> &str {
        "every_node_reached"
    }

    fn at_milestone(&mut self, model: &Gossip, now: MonoTime, label: &str) -> Result<(), String> {
        let missing: Vec<usize> = (0..NODES as usize)
            .filter(|&node| model.received[node] == 0)
            .collect();
        // Every event up to the milestone's instant has run, and none after it.
        assert!(model.log.iter().all(|entry| entry.0 <= now.0));
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!("{label}: nodes {missing:?} received nothing"))
        }
    }
}

#[test]
fn a_liveness_violation_stops_the_run_at_its_milestone() {
    // At 0 ms only the two starting pings have landed, so nodes 0 and 3 have nothing yet.
    let mut world = gossip(7);
    world.add_checker(EveryNodeReached);
    world.milestone(ms(0), "start");
    let violation = violated(world.run_until(ms(50)));
    assert_eq!(
        violation.found,
        Found::AtMilestone {
            label: "start".into()
        }
    );
    assert_eq!(violation.at, ms(0));
    let at_zero = world
        .model()
        .log
        .iter()
        .filter(|entry| entry.0 == 0)
        .count() as u64;
    assert_eq!(violation.events, at_zero, "every event at 0 ms ran first");
    assert!(violation.events >= 2);
    assert_eq!(violation.message, "start: nodes [0, 3] received nothing");
    assert!(violation.to_string().contains("at milestone `start`"));
    // By 50 ms every node has a ping, so a milestone there passes.
    let mut world = gossip(7);
    world.add_checker(EveryNodeReached);
    world.milestone(ms(50), "end");
    assert!(matches!(
        world.run_until(ms(60)),
        RunOutcome::Reached { .. }
    ));
}

#[test]
fn run_until_leaves_the_clock_at_its_end() {
    let mut world = SimWorld::new(1, |_| Order(Vec::new()));
    assert_eq!(world.now(), MonoTime(0));
    assert_eq!(
        world.run_until(ms(5)),
        RunOutcome::Reached {
            at: ms(5),
            events: 0
        }
    );
    assert_eq!(world.now(), ms(5));
    world.schedule(ms(5), Label('X'));
    world.schedule(ms(9), Label('Y'));
    assert_eq!(
        world.run_until(ms(7)),
        RunOutcome::Reached {
            at: ms(7),
            events: 1
        }
    );
    assert_eq!((world.pending(), world.now()), (1, ms(7)));
}

#[test]
#[should_panic(expected = "an event scheduled in the past")]
fn scheduling_into_the_past_is_a_bug() {
    let mut world = SimWorld::new(1, |_| Order(Vec::new()));
    world.run_until(ms(5));
    world.schedule(ms(4), Label('X'));
}
