// SPDX-License-Identifier: MIT OR Apache-2.0

//! The region in one process: layout checks, the header, ring ends by side, and opening checks.
//! `tests/cross_process.rs` runs the runner side in a second process.

use syntropy_spsc::shared::{SharedPopError, SharedPushError};

use super::*;

const PROTOCOL: IpcProtocol = IpcProtocol { major: 1, minor: 0 };
const BUILD: [u8; 32] = [7; 32];

fn ring(role: RingRole, slots: u32, slot_bytes: u32) -> RingSpec {
    RingSpec {
        role,
        slots,
        slot_bytes,
    }
}

/// Every role of RFC-0001 §3.5, with two dispatchers' step rings.
fn layout() -> ShmLayout {
    ShmLayout {
        rings: vec![
            ring(RingRole::Telemetry, 64, 256),
            ring(RingRole::Step { dispatcher: 0 }, 32, 128),
            ring(RingRole::Step { dispatcher: 1 }, 32, 128),
            ring(RingRole::LocalTokens, 128, 64),
            ring(RingRole::LocalSequences, 64, 64),
            ring(RingRole::Log, 64, 512),
        ],
    }
}

fn open(region: &Region, side: Side) -> Result<Region, ShmError> {
    Region::open(&region.handle(), side, &layout(), PROTOCOL)
}

#[test]
fn a_new_region_is_owner_only_and_reads_back_its_header() {
    let region = Region::create(&layout(), BUILD, PROTOCOL).unwrap();
    assert!(region.is_owner_only().unwrap());
    assert_eq!(region.side(), Side::Daemon);
    assert_eq!(region.build_id(), BUILD);
    assert_eq!(region.protocol(), PROTOCOL);
    assert_eq!((region.heartbeat(), region.peer_heartbeat()), (0, 0));
    assert_eq!((region.steps(), region.runner_state()), (0, 0));
    assert_eq!(region.bytes() % 128, 0);
}

/// The runner's side sees what the daemon's writes, and the other way round.
#[test]
fn both_sides_share_the_header() {
    let daemon = Region::create(&layout(), BUILD, PROTOCOL).unwrap();
    let runner = open(&daemon, Side::Runner).unwrap();
    assert_eq!(runner.build_id(), BUILD);
    runner.beat();
    runner.beat();
    runner.step();
    runner.set_runner_state(4);
    daemon.beat();
    assert_eq!(daemon.peer_heartbeat(), 2);
    assert_eq!(runner.peer_heartbeat(), 1);
    assert_eq!(daemon.steps(), 1);
    assert_eq!(daemon.runner_state(), 4);
}

#[test]
fn each_side_takes_its_own_end_of_each_ring_once() {
    let daemon = Region::create(&layout(), BUILD, PROTOCOL).unwrap();
    let runner = open(&daemon, Side::Runner).unwrap();
    let mut tokens_out = runner.producer(RingRole::LocalTokens).unwrap();
    let mut tokens_in = daemon.consumer(RingRole::LocalTokens).unwrap();
    let mut sequences_out = daemon.producer(RingRole::LocalSequences).unwrap();
    let mut sequences_in = runner.consumer(RingRole::LocalSequences).unwrap();
    tokens_out.push(b"token").unwrap();
    sequences_out.push(b"open").unwrap();
    let mut buffer = [0u8; 64];
    let len = tokens_in.pop_into(&mut buffer).unwrap();
    assert_eq!(&buffer[..len], b"token");
    let len = sequences_in.pop_into(&mut buffer).unwrap();
    assert_eq!(&buffer[..len], b"open");
    assert_eq!(
        runner.producer(RingRole::LocalTokens).unwrap_err(),
        EndError::Taken(RingRole::LocalTokens)
    );
    assert_eq!(
        daemon.producer(RingRole::Log).unwrap_err(),
        EndError::WrongSide {
            role: RingRole::Log,
            side: Side::Daemon,
        }
    );
    assert_eq!(
        runner.consumer(RingRole::Log).unwrap_err(),
        EndError::WrongSide {
            role: RingRole::Log,
            side: Side::Runner,
        }
    );
    assert_eq!(
        runner
            .producer(RingRole::Step { dispatcher: 9 })
            .unwrap_err(),
        EndError::NoSuchRing(RingRole::Step { dispatcher: 9 })
    );
    // A slot's size limits what the ring takes.
    assert_eq!(
        tokens_out.push(&[0; 65]),
        Err(SharedPushError::TooLong { len: 65, max: 64 })
    );
    assert_eq!(tokens_in.pop_into(&mut buffer), Err(SharedPopError::Empty));
}

/// Each step ring is separate, as RFC-0053 §2.5 has one per dispatcher.
#[test]
fn step_rings_are_separate_per_dispatcher() {
    let daemon = Region::create(&layout(), BUILD, PROTOCOL).unwrap();
    let runner = open(&daemon, Side::Runner).unwrap();
    let mut first = runner.producer(RingRole::Step { dispatcher: 0 }).unwrap();
    let mut second = runner.producer(RingRole::Step { dispatcher: 1 }).unwrap();
    first.push(b"0").unwrap();
    second.push(b"1").unwrap();
    let mut buffer = [0u8; 8];
    let mut read = |dispatcher| {
        let mut end = daemon.consumer(RingRole::Step { dispatcher }).unwrap();
        let len = end.pop_into(&mut buffer).unwrap();
        buffer[..len].to_vec()
    };
    assert_eq!(read(1), b"1");
    assert_eq!(read(0), b"0");
}

#[test]
fn opening_checks_the_protocol_the_size_and_the_ring_table() {
    let daemon = Region::create(&layout(), BUILD, PROTOCOL).unwrap();
    let other = IpcProtocol { major: 2, minor: 0 };
    assert!(matches!(
        Region::open(&daemon.handle(), Side::Runner, &layout(), other),
        Err(ShmError::Protocol { found, expected }) if found == PROTOCOL && expected == other
    ));
    let mut smaller = layout();
    smaller.rings.pop();
    assert!(matches!(
        Region::open(&daemon.handle(), Side::Runner, &smaller, PROTOCOL),
        Err(ShmError::Size { .. })
    ));
    // Same size, other roles: two rings swapped.
    let mut swapped = layout();
    swapped.rings.swap(1, 2);
    assert!(matches!(
        Region::open(&daemon.handle(), Side::Runner, &swapped, PROTOCOL),
        Err(ShmError::RingTable)
    ));
}

#[test]
fn layouts_are_checked() {
    let mut twice = layout();
    twice.rings.push(ring(RingRole::Log, 8, 8));
    assert!(matches!(
        Region::create(&twice, BUILD, PROTOCOL),
        Err(ShmError::Layout(LayoutProblem::DuplicateRole(
            RingRole::Log
        )))
    ));
    let bad = ShmLayout {
        rings: vec![ring(RingRole::Telemetry, 3, 8)],
    };
    assert!(matches!(
        Region::create(&bad, BUILD, PROTOCOL),
        Err(ShmError::Layout(LayoutProblem::Ring {
            role: RingRole::Telemetry,
            ..
        }))
    ));
    let many = ShmLayout {
        rings: (0..=MAX_RINGS as u16)
            .map(|dispatcher| ring(RingRole::Step { dispatcher }, 1, 1))
            .collect(),
    };
    assert!(matches!(
        Region::create(&many, BUILD, PROTOCOL),
        Err(ShmError::Layout(LayoutProblem::TooManyRings(257)))
    ));
}

#[test]
fn handles_round_trip_through_their_text() {
    for handle in [
        RegionHandle::Fd(3),
        RegionHandle::Name("/syntropy-1a2b-0".to_owned()),
        RegionHandle::Name("Local\\syntropy-99-1".to_owned()),
    ] {
        assert_eq!(handle.to_string().parse::<RegionHandle>().unwrap(), handle);
    }
    for bad in ["", "fd:", "fd:x", "name:", "pipe:3"] {
        assert!(bad.parse::<RegionHandle>().is_err(), "{bad}");
    }
}

#[test]
fn the_layout_travels_through_serde() {
    // `Init` carries the layout to the runner (RFC-0001 §3.5).
    let layout = layout();
    let text = serde_json::to_string(&layout).unwrap();
    assert_eq!(serde_json::from_str::<ShmLayout>(&text).unwrap(), layout);
    let text = serde_json::to_string(&PROTOCOL).unwrap();
    assert_eq!(
        serde_json::from_str::<IpcProtocol>(&text).unwrap(),
        PROTOCOL
    );
}

#[test]
fn a_region_can_move_between_threads() {
    fn is<T: Send + Sync>() {}
    is::<Region>();
}

#[test]
fn errors_name_what_happened() {
    assert_eq!(
        ShmError::Protocol {
            found: IpcProtocol { major: 2, minor: 1 },
            expected: PROTOCOL,
        }
        .to_string(),
        "the region is for ipc_protocol 2.1, and this build speaks 1.0"
    );
    assert_eq!(
        EndError::Taken(RingRole::Log).to_string(),
        "this region already handed out its Log end"
    );
}
