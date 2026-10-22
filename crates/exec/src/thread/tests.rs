// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use syntropy_core::NodeId;

use super::*;

fn core(id: u32, numa_node: u16) -> Core {
    Core {
        id: CoreId(id),
        numa_node,
    }
}

fn device(ordinal: u16) -> DeviceId {
    DeviceId {
        node: NodeId([7; 32]),
        ordinal,
    }
}

/// Two NUMA nodes of four cores each.
fn two_nodes() -> Cores {
    Cores::new((0..8).map(|id| core(id, (id / 4) as u16)).collect())
}

#[test]
fn a_device_gets_a_core_on_its_own_numa_node_first() {
    let mut cores = two_nodes();
    assert_eq!(cores.take(Some(1)), Some(core(4, 1)));
    assert_eq!(cores.take(Some(1)), Some(core(5, 1)));
    assert_eq!(cores.take(Some(0)), Some(core(0, 0)));
    // A device with no NUMA node gets the lowest free core.
    assert_eq!(cores.take(None), Some(core(1, 0)));
    assert_eq!(cores.len(), 4);
}

#[test]
fn a_full_node_falls_back_to_another_and_every_core_is_handed_out_once() {
    let mut cores = two_nodes().without(&[CoreId(0), CoreId(1), CoreId(2)]);
    assert_eq!(cores.take(Some(0)), Some(core(3, 0)));
    // Node 0 is now full, so its next device gets node 1's lowest core.
    assert_eq!(cores.take(Some(0)), Some(core(4, 1)));
    let rest: Vec<Core> = std::iter::from_fn(|| cores.take(Some(1))).collect();
    assert_eq!(rest, [core(5, 1), core(6, 1), core(7, 1)]);
    assert!(cores.is_empty());
    assert_eq!(cores.take(None), None);
}

#[test]
fn a_topology_with_a_repeated_core_lists_it_once() {
    let cores = Cores::new(vec![core(2, 0), core(1, 0), core(2, 0)]);
    assert_eq!(cores.len(), 2);
}

fn rings<In, Out>(capacity: usize) -> (DispatcherRings<In, Out>, Producer<In>, Consumer<Out>) {
    let (to_dispatcher, from_net) = syntropy_spsc::ring(capacity).unwrap();
    let (to_net, from_dispatcher) = syntropy_spsc::ring(capacity).unwrap();
    (
        DispatcherRings {
            from_net,
            to_net,
            io: None,
        },
        to_dispatcher,
        from_dispatcher,
    )
}

/// Criterion 2: a dispatcher reports its pinned core and that core's NUMA node, and runs there.
#[test]
fn a_dispatcher_reports_the_core_and_numa_node_it_runs_on() {
    let mut cores = Cores::detect().expect("the affinity mask");
    let given = cores.take(Some(0)).expect("a core");
    let (rings, _tx, _rx) = rings::<u32, u32>(4);
    let spec = DispatcherSpec {
        kind: DispatcherKind::Device,
        device: device(0),
        numa: Some(given.numa_node),
        core: Some(given),
        spin: syntropy_spsc::SPIN_DEFAULT,
    };
    let dispatcher = spawn_dispatcher(spec, rings, |_| {
        (0..1_000).map(|_| os::current_core()).collect::<Vec<_>>()
    })
    .unwrap();
    let report = dispatcher.report().clone();
    let seen = dispatcher.join().unwrap();
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        // Unpinned, on the one memory node (ADR-0012, revision of 2026-10-23).
        assert_eq!((report.core, report.numa_node), (None, Some(0)));
        assert!(report.unpinned.is_some(), "{report:?}");
        assert!(seen.iter().all(Option::is_none));
    } else {
        assert_eq!(report.core, Some(given.id.0), "{report:?}");
        assert_eq!(report.numa_node, Some(given.numa_node));
        assert_eq!(report.numa_local, Some(true));
        assert_eq!(report.unpinned, None);
        assert!(seen.iter().all(|c| *c == Some(given.id)), "{seen:?}");
    }
    assert_eq!(report.kind, DispatcherKind::Device);
    assert_eq!(report.device, device(0));
    assert_eq!(report.spin_budget_ns, 50_000);
}

#[test]
fn a_dispatcher_given_no_core_says_why_it_is_unpinned() {
    let (rings, _tx, _rx) = rings::<u32, u32>(4);
    let spec = DispatcherSpec {
        kind: DispatcherKind::Cpu,
        device: device(3),
        numa: Some(1),
        core: None,
        spin: syntropy_spsc::SPIN_ON_BATTERY,
    };
    let dispatcher = spawn_dispatcher(spec, rings, |ctx| ctx.spin()).unwrap();
    let report = dispatcher.report().clone();
    assert_eq!(dispatcher.join().unwrap(), Duration::from_micros(10));
    assert_eq!(
        (report.core, report.numa_node, report.numa_local),
        (None, None, None)
    );
    assert_eq!(
        report.unpinned.as_deref(),
        Some("no free core was given to it")
    );
    assert_eq!(report.spin_budget_ns, 10_000);
}

/// Between steps the dispatcher spins, then parks on its inbound ring; each frame the `net`
/// domain pushes wakes it, and it answers on its outbound ring.
#[test]
fn an_idle_dispatcher_wakes_for_each_frame_and_answers_on_its_ring() {
    let (rings, mut to_dispatcher, mut from_dispatcher) = rings::<u32, u32>(4);
    let spec = DispatcherSpec {
        kind: DispatcherKind::Device,
        device: device(1),
        numa: None,
        core: None,
        // No spin: every wait parks, so each frame is a wake-up.
        spin: Duration::ZERO,
    };
    let dispatcher = spawn_dispatcher(spec, rings, |mut ctx| {
        let mut answered = 0u32;
        loop {
            ctx.idle_wait();
            match ctx.rings.from_net.pop() {
                Ok(frame) => {
                    ctx.rings.to_net.push(frame * 2).unwrap();
                    answered += 1;
                }
                Err(syntropy_spsc::PopError::Closed) => return answered,
                Err(syntropy_spsc::PopError::Empty) => {}
            }
        }
    })
    .unwrap();
    for frame in 1..=20 {
        // Give the thread time to park before each frame.
        std::thread::sleep(Duration::from_millis(2));
        to_dispatcher.push(frame).unwrap();
        assert_eq!(
            from_dispatcher.pop_wait(Duration::from_millis(1)),
            Some(frame * 2)
        );
    }
    drop(to_dispatcher);
    assert_eq!(dispatcher.join().unwrap(), 20);
}

/// The debug state's record, as JSON.
#[test]
fn the_report_serializes_as_the_debug_state_lists_it() {
    let report = HotThreadReport {
        kind: DispatcherKind::Cpu,
        device: device(2),
        core: Some(5),
        numa_node: Some(1),
        numa_local: Some(true),
        priority: ThreadPriority::Nice(0),
        unpinned: None,
        spin_budget_ns: 50_000,
    };
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["kind"], "cpu");
    assert_eq!(json["core"], 5);
    assert_eq!(json["numa_node"], 1);
    assert_eq!(json["numa_local"], true);
    assert_eq!(json["priority"]["nice"], 0);
    assert_eq!(json["device"]["ordinal"], 2);
    assert!(json["unpinned"].is_null());
}
