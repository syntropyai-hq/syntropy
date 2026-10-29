// SPDX-License-Identifier: MIT OR Apache-2.0

//! The lifecycle machine's tests (DIST-E1-T05).
//!
//! - Over random event sequences, only RFC-0001 §3.3's edges occur, every transition returns
//!   exactly one `StateChanged`, and a refused event changes nothing
//!   (`random_event_sequences_take_only_section_3_3s_edges`). The same run takes every edge.
//! - The forward path, the plan delta, the drain and the restart, step by step.

use std::cell::RefCell;
use std::collections::BTreeSet;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use smallvec::smallvec;
use syntropy_core::{DegradeDetail, DegradeFlags, DeviceId, Nanos, NodeId};

use super::*;

const EPOCH: PlanEpoch = PlanEpoch { term: 3, seq: 1 };
const SHARD: Bytes = Bytes(4 << 30);

fn class() -> impl Strategy<Value = ErrorClass> {
    prop_oneof![
        Just(ErrorClass::Capacity),
        Just(ErrorClass::Link),
        Just(ErrorClass::Device),
        Just(ErrorClass::Integrity),
        Just(ErrorClass::Correctness),
        Just(ErrorClass::Timeout),
        Just(ErrorClass::Internal),
    ]
}

fn warm_step() -> impl Strategy<Value = WarmStep> {
    prop_oneof![
        Just(WarmStep::Prepare),
        Just(WarmStep::Capture),
        Just(WarmStep::CacheFill),
        Just(WarmStep::LinkProbe),
        Just(WarmStep::Canary),
    ]
}

fn reason() -> impl Strategy<Value = DrainReason> {
    prop_oneof![
        any::<u64>().prop_map(|seq| DrainReason::Replan {
            successor: PlanEpoch { term: 4, seq }
        }),
        Just(DrainReason::Unload),
        Just(DrainReason::Shutdown),
        Just(DrainReason::NodeLeaving),
    ]
}

/// Any event, with values that are sometimes out of range or out of order.
fn event() -> impl Strategy<Value = Event> {
    let bytes = || (0u64..=SHARD.0 + 1).prop_map(Bytes);
    prop_oneof![
        bytes().prop_map(|have| Event::FetchProgress { have }),
        bytes().prop_map(|pack| Event::ShardVerified { pack }),
        bytes().prop_map(|done| Event::PackProgress { done }),
        bytes().prop_map(|resident| Event::PackCommitted { resident }),
        bytes().prop_map(|done| Event::LoadProgress { done }),
        Just(Event::Resident),
        warm_step().prop_map(Event::WarmStep),
        Just(Event::WarmPassed),
        (0u32..4).prop_map(|live| Event::Sequences { live }),
        prop::option::of(Just(Degradation {
            flags: DegradeFlags::CACHE_REDUCED,
            detail: smallvec![DegradeDetail::Reduced {
                device: DeviceId {
                    node: NodeId([1; 32]),
                    ordinal: 0,
                },
                bytes: Bytes(1 << 20),
            }],
        }))
        .prop_map(Event::Degraded),
        (0u64..6).prop_map(|seq| Event::PlanDelta {
            epoch: PlanEpoch { term: 3, seq }
        }),
        (reason(), any::<u64>()).prop_map(|(reason, within)| Event::Drain {
            reason,
            deadline: Deadline::within(Nanos(within)),
        }),
        Just(Event::DeadlineReached),
        class().prop_map(|class| Event::Fault { class }),
        Just(Event::Restart),
        prop_oneof![Just(GiveUp::BudgetSpent), Just(GiveUp::Fatal)].prop_map(Event::GiveUp),
    ]
}

/// The event that moves `machine` along §3.3's forward path, or a restart from `Failed`. The
/// property test uses it for some steps, so that random runs reach the later states.
fn forward(machine: &Lifecycle) -> Option<Event> {
    Some(match machine.state() {
        RunnerState::Fetching { .. } => Event::ShardVerified { pack: Bytes(100) },
        RunnerState::Packing { .. } => Event::PackCommitted {
            resident: Bytes(50),
        },
        RunnerState::Loading { .. } => Event::Resident,
        RunnerState::Warming { .. } => Event::WarmPassed,
        RunnerState::Ready { .. } => Event::Drain {
            reason: DrainReason::Unload,
            deadline: Deadline::within(Nanos(1_000_000_000)),
        },
        RunnerState::Draining { .. } => Event::Sequences { live: 0 },
        RunnerState::Failed { .. } => Event::Restart,
        _ => return None,
    })
}

/// One run: the machine after `start`, then each step's event, either the generated one or, when
/// its flag is set, [`forward`]'s. Checks every step and returns the edges taken.
fn run(steps: &[(bool, Event)]) -> BTreeSet<(RunnerStateKind, RunnerStateKind)> {
    let mut edges = BTreeSet::new();
    let (mut machine, first) = Lifecycle::start(EPOCH, SHARD);
    assert_eq!(
        first,
        RunnerMsg::StateChanged {
            state: machine.state().clone(),
            epoch: EPOCH,
        }
    );
    assert_eq!(machine.state().kind(), RunnerStateKind::Fetching);
    let mut transitions = 0usize;
    let mut sent = 0usize;
    for (guided, generated) in steps {
        let event = match guided.then(|| forward(&machine)).flatten() {
            Some(event) => event,
            None => generated.clone(),
        };
        let before = machine.clone();
        let result = machine.apply(event.clone());
        let (from, to) = (before.state().kind(), machine.state().kind());
        // A transition is a change of kind, or `Ready → Ready` under a new epoch.
        let transition = from != to || before.epoch() != machine.epoch();
        transitions += usize::from(transition);
        match result {
            Ok(Some(msg)) => {
                sent += 1;
                assert!(
                    transition,
                    "{event:?} in {from:?} sent {msg:?} without moving"
                );
                assert!(from.can_become(to), "{from:?} -> {to:?} on {event:?}");
                assert_eq!(
                    msg,
                    RunnerMsg::StateChanged {
                        state: machine.state().clone(),
                        epoch: machine.epoch(),
                    }
                );
                edges.insert((from, to));
            }
            Ok(None) => {
                assert!(!transition, "{event:?} moved {from:?} -> {to:?} silently");
                assert_eq!(before.epoch(), machine.epoch());
                assert_eq!(before.restarts(), machine.restarts());
            }
            Err(refused) => {
                assert_eq!(machine, before, "{event:?} refused ({refused}) but changed");
            }
        }
        // The epoch never goes back, and `Stopped` is terminal.
        assert!(machine.epoch() >= before.epoch());
        if from == RunnerStateKind::Stopped {
            assert_eq!(machine, before);
        }
    }
    assert_eq!(sent, transitions, "one StateChanged per transition");
    edges
}

/// RFC-0001 §3.3's edges, and none other, over random event sequences, with every edge taken.
#[test]
fn random_event_sequences_take_only_section_3_3s_edges() {
    let mut runner = TestRunner::new_with_rng(
        Config {
            cases: 512,
            ..Config::default()
        },
        TestRng::from_seed(RngAlgorithm::ChaCha, &[5; 32]),
    );
    let steps = prop::collection::vec((any::<bool>(), event()), 0..64);
    let random = RefCell::new(BTreeSet::new());
    runner
        .run(&steps, |steps| {
            random.borrow_mut().extend(run(&steps));
            Ok(())
        })
        .unwrap();
    // Fixed runs as well, which take each edge whatever the seed.
    let random = random.into_inner();
    let mut taken = random.clone();
    for steps in fixed_runs() {
        taken.extend(run(&steps));
    }
    let every_edge: BTreeSet<_> = RunnerStateKind::ALL
        .into_iter()
        .flat_map(|from| RunnerStateKind::ALL.into_iter().map(move |to| (from, to)))
        .filter(|(from, to)| from.can_become(*to))
        .collect();
    assert_eq!(
        random, every_edge,
        "the random runs take every §3.3 edge, and no other"
    );
    assert_eq!(taken, every_edge, "the fixed runs take no other edge");
}

/// Runs that take each §3.3 edge, as `(false, event)` steps.
fn fixed_runs() -> Vec<Vec<(bool, Event)>> {
    let to_ready = vec![
        Event::ShardVerified { pack: Bytes(10) },
        Event::PackCommitted {
            resident: Bytes(10),
        },
        Event::Resident,
        Event::WarmPassed,
    ];
    let mut runs = vec![];
    // Failing from each working state, then a restart and a give-up.
    for depth in 0..=5 {
        let mut run = vec![];
        run.extend(to_ready.iter().take(depth.min(4)).cloned());
        if depth == 5 {
            run.extend(to_ready.iter().skip(4).cloned());
            run.push(Event::Drain {
                reason: DrainReason::Shutdown,
                deadline: Deadline::within(Nanos(5)),
            });
        }
        run.push(Event::Fault {
            class: ErrorClass::Device,
        });
        run.push(Event::Restart);
        run.push(Event::Fault {
            class: ErrorClass::Link,
        });
        run.push(Event::GiveUp(GiveUp::Fatal));
        runs.push(run);
    }
    // The whole forward path with a plan delta, ended by the deadline.
    let mut run = to_ready.clone();
    run.push(Event::PlanDelta {
        epoch: PlanEpoch { term: 3, seq: 2 },
    });
    run.push(Event::Drain {
        reason: DrainReason::Unload,
        deadline: Deadline::within(Nanos(5)),
    });
    run.push(Event::DeadlineReached);
    runs.push(run);
    runs.into_iter()
        .map(|run| run.into_iter().map(|event| (false, event)).collect())
        .collect()
}

/// The forward path reports each state once, with its progress fields at entry, and progress
/// inside a state sends nothing.
#[test]
fn the_forward_path_reports_each_state_once() {
    let (mut m, first) = Lifecycle::start(EPOCH, Bytes(100));
    assert!(matches!(
        first,
        RunnerMsg::StateChanged {
            state: RunnerState::Fetching {
                have: Bytes(0),
                need: Bytes(100)
            },
            epoch: EPOCH,
        }
    ));
    assert_eq!(m.apply(Event::FetchProgress { have: Bytes(60) }), Ok(None));
    assert_eq!(
        m.apply(Event::FetchProgress { have: Bytes(101) }),
        Err(Refused::PastTotal {
            progress: Bytes(101),
            total: Bytes(100)
        })
    );
    assert_eq!(
        m.state(),
        &RunnerState::Fetching {
            have: Bytes(60),
            need: Bytes(100)
        }
    );
    let packing = m.apply(Event::ShardVerified { pack: Bytes(80) }).unwrap();
    assert!(matches!(
        packing,
        Some(RunnerMsg::StateChanged {
            state: RunnerState::Packing {
                done: Bytes(0),
                total: Bytes(80)
            },
            ..
        })
    ));
    assert_eq!(m.apply(Event::PackProgress { done: Bytes(80) }), Ok(None));
    m.apply(Event::PackCommitted {
        resident: Bytes(70),
    })
    .unwrap()
    .unwrap();
    assert_eq!(m.apply(Event::LoadProgress { done: Bytes(1) }), Ok(None));
    m.apply(Event::Resident).unwrap().unwrap();
    assert_eq!(
        m.state(),
        &RunnerState::Warming {
            step: WarmStep::Prepare
        }
    );
    assert_eq!(m.apply(Event::WarmStep(WarmStep::Capture)), Ok(None));
    // Steps may be skipped, never repeated or taken back.
    assert_eq!(m.apply(Event::WarmStep(WarmStep::LinkProbe)), Ok(None));
    assert_eq!(
        m.apply(Event::WarmStep(WarmStep::LinkProbe)),
        Err(Refused::WarmStepBack {
            current: WarmStep::LinkProbe,
            reported: WarmStep::LinkProbe
        })
    );
    let ready = m.apply(Event::WarmPassed).unwrap();
    assert_eq!(
        ready,
        Some(RunnerMsg::StateChanged {
            state: RunnerState::Ready {
                live_sequences: 0,
                degraded: None
            },
            epoch: EPOCH,
        })
    );
    // A forward event out of its state is refused.
    assert_eq!(
        m.apply(Event::Resident),
        Err(Refused::WrongState {
            state: RunnerStateKind::Ready,
            event: "Resident"
        })
    );
}

/// `ApplyPlanDelta` in `Ready` is `Ready → Ready` under the new epoch, with one `StateChanged`;
/// a stale epoch, or a delta in another state, is refused.
#[test]
fn a_plan_delta_is_ready_to_ready_under_a_later_epoch() {
    let mut m = ready();
    assert_eq!(m.apply(Event::Sequences { live: 2 }), Ok(None));
    let later = PlanEpoch { term: 3, seq: 2 };
    assert_eq!(
        m.apply(Event::PlanDelta { epoch: later }),
        Ok(Some(RunnerMsg::StateChanged {
            state: RunnerState::Ready {
                live_sequences: 2,
                degraded: None
            },
            epoch: later,
        }))
    );
    assert_eq!(m.epoch(), later);
    for stale in [later, EPOCH] {
        assert_eq!(
            m.apply(Event::PlanDelta { epoch: stale }),
            Err(Refused::StaleEpoch {
                current: later,
                reported: stale
            })
        );
    }
    let (mut fetching, _) = Lifecycle::start(EPOCH, SHARD);
    assert_eq!(
        fetching.apply(Event::PlanDelta { epoch: later }),
        Err(Refused::WrongState {
            state: RunnerStateKind::Fetching,
            event: "PlanDelta"
        })
    );
}

/// A drain carries the live count, ends when it reaches zero or at the deadline, and stops for
/// the drain's reason.
#[test]
fn a_drain_ends_at_zero_live_sequences_or_the_deadline() {
    let deadline = Deadline::within(Nanos(30_000_000_000));
    for by_deadline in [false, true] {
        let mut m = ready();
        m.apply(Event::Sequences { live: 3 }).unwrap();
        let draining = m
            .apply(Event::Drain {
                reason: DrainReason::NodeLeaving,
                deadline,
            })
            .unwrap();
        assert!(matches!(
            draining,
            Some(RunnerMsg::StateChanged {
                state: RunnerState::Draining {
                    reason: DrainReason::NodeLeaving,
                    live_sequences: 3,
                    ..
                },
                ..
            })
        ));
        assert_eq!(m.apply(Event::Sequences { live: 1 }), Ok(None));
        let stop = if by_deadline {
            Event::DeadlineReached
        } else {
            Event::Sequences { live: 0 }
        };
        assert_eq!(
            m.apply(stop).unwrap(),
            Some(RunnerMsg::StateChanged {
                state: RunnerState::Stopped {
                    reason: StopReason::Drained(DrainReason::NodeLeaving)
                },
                epoch: EPOCH,
            })
        );
        // Terminal.
        for event in [
            Event::Restart,
            Event::Fault {
                class: ErrorClass::Internal,
            },
        ] {
            assert!(m.apply(event).is_err());
        }
    }
}

/// A fault records the state it happened in and the restarts so far; a restart re-enters
/// `Fetching` for the same shard; giving up stops for the budget or the last fault's class.
#[test]
fn a_fault_restarts_at_fetching_or_stops_for_the_supervisors_reason() {
    let mut m = ready();
    assert_eq!(
        m.apply(Event::Fault {
            class: ErrorClass::Device
        })
        .unwrap(),
        Some(RunnerMsg::StateChanged {
            state: RunnerState::Failed {
                class: ErrorClass::Device,
                during: RunnerStateKind::Ready,
                restarts: 0,
            },
            epoch: EPOCH,
        })
    );
    assert!(
        m.apply(Event::Fault {
            class: ErrorClass::Device
        })
        .is_err()
    );
    assert_eq!(
        m.apply(Event::Restart).unwrap(),
        Some(RunnerMsg::StateChanged {
            state: RunnerState::Fetching {
                have: Bytes(0),
                need: SHARD
            },
            epoch: EPOCH,
        })
    );
    m.apply(Event::ShardVerified { pack: Bytes(1) }).unwrap();
    m.apply(Event::Fault {
        class: ErrorClass::Integrity,
    })
    .unwrap();
    assert_eq!(
        m.state(),
        &RunnerState::Failed {
            class: ErrorClass::Integrity,
            during: RunnerStateKind::Packing,
            restarts: 1,
        }
    );
    let mut spent = m.clone();
    assert_eq!(
        spent.apply(Event::GiveUp(GiveUp::BudgetSpent)).unwrap(),
        Some(RunnerMsg::StateChanged {
            state: RunnerState::Stopped {
                reason: StopReason::Drained(DrainReason::RestartBudgetSpent)
            },
            epoch: EPOCH,
        })
    );
    m.apply(Event::GiveUp(GiveUp::Fatal)).unwrap();
    assert_eq!(
        m.state(),
        &RunnerState::Stopped {
            reason: StopReason::Fatal(ErrorClass::Integrity)
        }
    );
}

/// A machine in `Ready` at [`EPOCH`].
fn ready() -> Lifecycle {
    let (mut m, _) = Lifecycle::start(EPOCH, SHARD);
    for event in [
        Event::ShardVerified { pack: Bytes(10) },
        Event::PackCommitted {
            resident: Bytes(10),
        },
        Event::Resident,
        Event::WarmPassed,
    ] {
        m.apply(event).unwrap().unwrap();
    }
    m
}
