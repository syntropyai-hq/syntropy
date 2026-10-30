// SPDX-License-Identifier: MIT OR Apache-2.0

//! The command channel's tests (DIST-E1-T03).
//!
//! - Every message round-trips through a frame, and through a real channel.
//! - Decode is total: arbitrary bytes and damaged frames give an envelope or a classified error,
//!   never a panic, and a forged sequence length reserves nothing.
//! - The handshake refuses an `Init` from another build. (A daemon killing a runner of another
//!   build is `tests/cmd_cross_process.rs`.)
//! - No message skips a field on the way out (`skips`): postcard reads fields by position.
//! - The endpoint is owner-only, and a peer of another user is refused (`other_user`, which needs
//!   a second OS account).
//! - The async listener, the API's local socket, accepts this user with its peer and keeps its
//!   endpoint owner-only (`async_listener`; its other-user case is in `other_user`).

use std::fmt::Debug;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

use proptest::prelude::*;
use serde::de::{DeserializeOwned, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use smallvec::smallvec;
use syntropy_core::ledger::LedgerGrant;
use syntropy_core::resource::{AppleGpuFamily, CpuIsa, DeviceClass, TierKind, TransportKind};
use syntropy_core::runner::{
    Deadline, Degradation, DegradeDetail, DegradeFlags, DrainReason, RunnerState, RunnerStateKind,
    StopReason, WarmStep,
};
use syntropy_core::sequence::SequenceOutcome;
use syntropy_core::{
    Blake3, Bytes, CancelReason, ChunkId, DeviceId, ErrorClass, FetchPriority, InstanceId,
    LayoutVersion, LinkId, LinkKind, ModelId, Nanos, NetMedium, NodeId, PackKey, PlanEpoch,
    PressureLevel, SequenceId, ShardSpec, StageIx, TensorGroupId, TierId,
};

use super::*;
use crate::shm::{IpcProtocol, RingRole, RingSpec, ShmLayout};

// With the `async` feature, and not under loom, which tokio does not build under.
#[cfg(all(feature = "async", not(loom)))]
mod async_listener;
mod fixtures;
// Not under loom: `cargo xtask test loom` runs this crate's unit tests in a job with no second
// account, and the test needs real processes and sockets, which loom does not model.
#[cfg(all(not(loom), any(target_os = "linux", target_vendor = "apple")))]
mod other_user;
#[cfg(all(not(loom), windows))]
#[path = "tests/other_user_windows.rs"]
mod other_user;
mod skips;

const BUILD: [u8; 32] = [0x5a; 32];

/// Every daemon message: each variant at least once.
fn daemon_msgs() -> Vec<DaemonMsg> {
    let all = vec![
        DaemonMsg::Init(fixtures::init()),
        DaemonMsg::AssignPlan {
            plan: Arc::new(fixtures::plan()),
            stages: smallvec![StageIx(0)],
        },
        // Past the inline capacity of 2, and every optional field of the plan unset.
        DaemonMsg::AssignPlan {
            plan: Arc::new(fixtures::sparse_plan()),
            stages: smallvec![StageIx(0), StageIx(1), StageIx(2)],
        },
        DaemonMsg::ShardProgress {
            have: Bytes(1 << 30),
            need: Bytes(3 << 30),
            complete: false,
        },
        DaemonMsg::ApplyPlanDelta {
            epoch: PlanEpoch { term: 7, seq: 3 },
            delta: fixtures::delta(),
        },
        DaemonMsg::CancelSequence {
            seq: SequenceId {
                ingress: 0x5eed,
                boot: 3,
                n: 41,
            },
            reason: CancelReason::ClientGone,
        },
        DaemonMsg::CancelSequence {
            seq: SequenceId {
                ingress: u64::MAX,
                boot: u32::MAX,
                n: u32::MAX,
            },
            reason: CancelReason::Error {
                domain: 7,
                code: 1201,
            },
        },
        DaemonMsg::PeerRevoked {
            node: fixtures::node(9),
        },
        DaemonMsg::Ping,
        DaemonMsg::Drain {
            reason: DrainReason::Replan {
                successor: PlanEpoch { term: 8, seq: 1 },
            },
            deadline: Deadline::within(Nanos(30_000_000_000)),
        },
        DaemonMsg::Drain {
            reason: DrainReason::RestartBudgetSpent,
            deadline: Deadline::within(Nanos(u64::MAX)),
        },
        DaemonMsg::Shutdown {
            deadline: Deadline::within(Nanos(0)),
        },
        DaemonMsg::MemoryPressure {
            level: PressureLevel::Critical,
            reclaim: Bytes(2 << 30),
            tier: None,
        },
        DaemonMsg::MemoryPressure {
            level: PressureLevel::Moderate,
            reclaim: Bytes(512 << 20),
            tier: Some(TierId {
                node: fixtures::node(2),
                ordinal: 1,
            }),
        },
    ];
    // No wildcard: a new variant does not compile here until it has a fixture.
    let mut seen = [false; 10];
    for msg in &all {
        seen[match msg {
            DaemonMsg::Init(_) => 0,
            DaemonMsg::AssignPlan { .. } => 1,
            DaemonMsg::ShardProgress { .. } => 2,
            DaemonMsg::ApplyPlanDelta { .. } => 3,
            DaemonMsg::CancelSequence { .. } => 4,
            DaemonMsg::PeerRevoked { .. } => 5,
            DaemonMsg::Ping => 6,
            DaemonMsg::Drain { .. } => 7,
            DaemonMsg::Shutdown { .. } => 8,
            DaemonMsg::MemoryPressure { .. } => 9,
        }] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "a DaemonMsg variant has no fixture"
    );
    all
}

fn hello(build_id: [u8; 32]) -> Hello {
    Hello {
        pid: std::process::id(),
        build_id,
        ipc_protocol: PROTOCOL,
        obs_tables: Blake3([0x0b; 32]),
    }
}

/// One runner state of each kind, `Ready` both plain and with every kind of degradation detail.
fn runner_states() -> Vec<RunnerState> {
    let node = NodeId([0x3c; 32]);
    let device = DeviceId { node, ordinal: 1 };
    let all = vec![
        RunnerState::Fetching {
            have: Bytes(1 << 20),
            need: Bytes(3 << 30),
        },
        RunnerState::Packing {
            done: Bytes(0),
            total: Bytes(u64::MAX),
        },
        RunnerState::Loading {
            done: Bytes(7),
            total: Bytes(9),
        },
        RunnerState::Warming {
            step: WarmStep::LinkProbe,
        },
        RunnerState::Ready {
            live_sequences: 0,
            degraded: None,
        },
        RunnerState::Ready {
            live_sequences: u32::MAX,
            degraded: Some(Degradation {
                flags: DegradeFlags::ABSORBED_STAGE
                    | DegradeFlags::TRANSPORT_DEMOTED
                    | DegradeFlags::CACHE_REDUCED,
                detail: smallvec![
                    DegradeDetail::Absorbed {
                        stages: smallvec![StageIx(2), StageIx(3)],
                        tiers: smallvec![TierKind::HostPinned, TierKind::Nvme],
                    },
                    DegradeDetail::Demoted {
                        edge: LinkId {
                            src: syntropy_core::Endpoint::Device(device),
                            dst: syntropy_core::Endpoint::Nic { node, ordinal: 0 },
                            kind: LinkKind::Network(NetMedium::default()),
                            path: 1,
                        },
                        from: TransportKind::Quic,
                        to: TransportKind::Tcp,
                    },
                    DegradeDetail::Reduced {
                        device,
                        bytes: Bytes(512 << 20),
                    },
                ],
            }),
        },
        RunnerState::Draining {
            reason: DrainReason::Replan {
                successor: PlanEpoch { term: 7, seq: 3 },
            },
            deadline: Deadline::within(Nanos(30_000_000_000)),
            live_sequences: 4,
        },
        RunnerState::Failed {
            class: ErrorClass::Device,
            during: RunnerStateKind::Warming,
            restarts: 2,
        },
        RunnerState::Stopped {
            reason: StopReason::Drained(DrainReason::RestartBudgetSpent),
        },
        RunnerState::Stopped {
            reason: StopReason::Fatal(ErrorClass::Integrity),
        },
    ];
    for kind in RunnerStateKind::ALL {
        assert!(
            all.iter().any(|state| state.kind() == kind),
            "{kind:?} has no fixture"
        );
    }
    all
}

/// Every runner message: each variant at least once, and `StateChanged` with every state.
fn runner_msgs() -> Vec<RunnerMsg> {
    let mut all = vec![RunnerMsg::Hello(hello(BUILD))];
    all.extend(
        runner_states()
            .into_iter()
            .map(|state| RunnerMsg::StateChanged {
                state,
                epoch: PlanEpoch { term: 7, seq: 2 },
            }),
    );
    all.extend([
        RunnerMsg::Pong,
        RunnerMsg::SequenceEnded {
            seq: SequenceId {
                ingress: 0x5eed,
                boot: 3,
                n: 41,
            },
            outcome: SequenceOutcome::Finished,
        },
        RunnerMsg::SequenceEnded {
            seq: SequenceId {
                ingress: 1,
                boot: 0,
                n: u32::MAX,
            },
            outcome: SequenceOutcome::Failed(ErrorClass::Device),
        },
        RunnerMsg::SequenceEnded {
            seq: SequenceId {
                ingress: u64::MAX,
                boot: u32::MAX,
                n: 0,
            },
            outcome: SequenceOutcome::Migrated {
                to: PlanEpoch { term: 9, seq: 0 },
            },
        },
        RunnerMsg::FetchShard {
            spec: shard_spec(),
            priority: FetchPriority::Urgent,
        },
        RunnerMsg::FetchShard {
            spec: ShardSpec {
                model: ModelId(Blake3([0; 32])),
                node: NodeId([0; 32]),
                groups: vec![],
                chunks: vec![],
                digest: Blake3([0; 32]),
            },
            priority: FetchPriority::Background,
        },
        RunnerMsg::PackCommitted {
            key: PackKey::for_shard(
                &shard_spec(),
                DeviceClass::Metal {
                    family: AppleGpuFamily::Apple9,
                    metal4: true,
                },
                LayoutVersion(u16::MAX),
            ),
            sidecar: Blake3([0x5c; 32]),
        },
        RunnerMsg::ProbeResult {
            profile: probe_profile(),
        },
    ]);
    let mut seen = [false; 7];
    for msg in &all {
        seen[match msg {
            RunnerMsg::Hello(_) => 0,
            RunnerMsg::StateChanged { .. } => 1,
            RunnerMsg::Pong => 2,
            RunnerMsg::SequenceEnded { .. } => 3,
            RunnerMsg::FetchShard { .. } => 4,
            RunnerMsg::PackCommitted { .. } => 5,
            RunnerMsg::ProbeResult { .. } => 6,
        }] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "a RunnerMsg variant has no fixture"
    );
    all
}

/// A probe's profile as a probe-mode process sends it: `node` and `version` unset, for the
/// daemon to stamp, and no devices measured yet (PLC-E1-T07 adds them).
fn probe_profile() -> syntropy_core::resource::CapabilityProfile {
    syntropy_core::resource::CapabilityProfile {
        node: NodeId([0; 32]),
        version: 0,
        devices: vec![],
        storage: vec![],
        links: vec![],
        ext: Default::default(),
    }
}

/// A shard of three groups and two chunks, its digest a stand-in: `ipc` links no BLAKE3, and
/// the channel carries the digest as sent.
fn shard_spec() -> ShardSpec {
    ShardSpec {
        model: ModelId(Blake3([0x3d; 32])),
        node: NodeId([0x4e; 32]),
        groups: vec![TensorGroupId(0), TensorGroupId(7), TensorGroupId(u32::MAX)],
        chunks: vec![ChunkId(Blake3([1; 32])), ChunkId(Blake3([2; 32]))],
        digest: Blake3([0x6f; 32]),
    }
}

/// Frames `envelope`, decodes it back, and checks the decode took the whole frame and gave the
/// same envelope, which encodes to the same bytes.
fn frame_round_trip<M: Serialize + DeserializeOwned + PartialEq + Debug>(envelope: &Envelope<M>) {
    let frame = encode_frame(envelope).expect("encode");
    let len = u32::from_le_bytes(frame[..HEADER].try_into().unwrap()) as usize;
    assert_eq!(len, frame.len() - HEADER);
    let (back, used) = decode_frame::<M>(&frame).expect("decode");
    assert_eq!(used, frame.len());
    assert_eq!(&back, envelope);
    assert_eq!(encode_frame(&back).expect("encode again"), frame);
    // A frame followed by the next one decodes alone.
    let mut two = frame.clone();
    two.extend_from_slice(&frame);
    assert_eq!(
        decode_frame::<M>(&two).expect("decode first").1,
        frame.len()
    );
}

#[test]
fn every_message_round_trips_through_a_frame() {
    for (n, msg) in daemon_msgs().into_iter().enumerate() {
        frame_round_trip(&Envelope {
            id: RequestId(n as u64 + 1),
            reply_to: None,
            msg,
        });
    }
    for (n, msg) in runner_msgs().into_iter().enumerate() {
        frame_round_trip(&Envelope {
            id: RequestId(u64::MAX - n as u64),
            reply_to: Some(RequestId(n as u64)),
            msg,
        });
    }
}

#[test]
fn a_borrowed_message_frames_as_an_owned_one() {
    for msg in daemon_msgs() {
        let owned = encode_frame(&Envelope {
            id: RequestId(3),
            reply_to: Some(RequestId(2)),
            msg: msg.clone(),
        });
        let borrowed = encode_frame(&Envelope {
            id: RequestId(3),
            reply_to: Some(RequestId(2)),
            msg: &msg,
        });
        assert_eq!(owned, borrowed);
    }
}

/// Postcard writes a variant as its index, so the 1.0 messages keep theirs and DIST-E1-T15's
/// come after them; protocol 1.1 marked the addition (RFC-0001 §3.5; §5, note of 2026-10-22).
/// Protocol 2.0 is DIST-E1-T05's: `StateChanged`'s `state` became a `RunnerState`, a layout
/// change an older peer cannot read, so the major moved (RFC-0001 §3.5, note of 2026-10-28).
#[test]
fn the_lifecycle_messages_are_appended_and_bump_the_minor() {
    let index = |bytes: Vec<u8>| bytes[0];
    let daemon = |msg: &DaemonMsg| index(postcard::to_allocvec(msg).unwrap());
    let runner = |msg: &RunnerMsg| index(postcard::to_allocvec(msg).unwrap());
    let expected_daemon = [
        ("Init", 0),
        ("AssignPlan", 1),
        ("ShardProgress", 2),
        ("ApplyPlanDelta", 3),
        ("CancelSequence", 4),
        ("PeerRevoked", 5),
        ("Ping", 6),
        ("Drain", 7),
        ("Shutdown", 8),
        ("MemoryPressure", 9),
    ];
    for msg in daemon_msgs() {
        let name = format!("{msg:?}");
        let (_, want) = expected_daemon
            .iter()
            .find(|(n, _)| name.starts_with(n))
            .expect("a known variant");
        assert_eq!(daemon(&msg), *want, "{name}");
    }
    let expected_runner = [
        ("Hello", 0),
        ("StateChanged", 1),
        ("Pong", 2),
        ("SequenceEnded", 3),
        ("FetchShard", 4),
        ("PackCommitted", 5),
        ("ProbeResult", 6),
    ];
    for msg in runner_msgs() {
        let name = format!("{msg:?}");
        let (_, want) = expected_runner
            .iter()
            .find(|(n, _)| name.starts_with(n))
            .expect("a known variant");
        assert_eq!(runner(&msg), *want, "{name}");
    }
    assert_eq!(PROTOCOL, IpcProtocol { major: 6, minor: 0 });
}

/// Why 6.0 is a major bump: a 5.x daemon's `RunnerMsg` ends at `PackCommitted`, so it refuses
/// `ProbeResult`, which a 6.0 probe-mode process sends. The messages it knows still read as
/// themselves.
#[test]
fn a_5_x_daemon_cannot_read_the_message_6_0_appends() {
    #[derive(Debug, Deserialize)]
    #[allow(dead_code, reason = "only whether it decodes matters")]
    enum RunnerMsgV5 {
        Hello(Hello),
        StateChanged {
            state: RunnerState,
            epoch: PlanEpoch,
        },
        Pong,
        SequenceEnded {
            seq: SequenceId,
            outcome: SequenceOutcome,
        },
        FetchShard {
            spec: ShardSpec,
            priority: FetchPriority,
        },
        PackCommitted {
            key: PackKey,
            sidecar: Blake3,
        },
    }
    for msg in runner_msgs() {
        let bytes = postcard::to_allocvec(&msg).unwrap();
        let old = postcard::from_bytes::<RunnerMsgV5>(&bytes);
        let appended = matches!(msg, RunnerMsg::ProbeResult { .. });
        assert_eq!(old.is_err(), appended, "{msg:?}");
    }
    assert_ne!(PROTOCOL.major, 5);
}

/// Why 5.0 is a major bump: a 4.x daemon's `RunnerMsg` ends at `SequenceEnded`, so it refuses
/// `FetchShard` and `PackCommitted`, which a 5.0 runner sends in normal operation. The messages
/// it knows still read as themselves.
#[test]
fn a_4_x_daemon_cannot_read_the_messages_5_0_appends() {
    #[derive(Debug, Deserialize)]
    #[allow(dead_code, reason = "only whether it decodes matters")]
    enum RunnerMsgV4 {
        Hello(Hello),
        StateChanged {
            state: RunnerState,
            epoch: PlanEpoch,
        },
        Pong,
        SequenceEnded {
            seq: SequenceId,
            outcome: SequenceOutcome,
        },
    }
    for msg in runner_msgs() {
        let bytes = postcard::to_allocvec(&msg).unwrap();
        let old = postcard::from_bytes::<RunnerMsgV4>(&bytes);
        let appended = matches!(
            msg,
            RunnerMsg::FetchShard { .. }
                | RunnerMsg::PackCommitted { .. }
                | RunnerMsg::ProbeResult { .. }
        );
        assert_eq!(old.is_err(), appended, "{msg:?}");
    }
    assert_ne!(PROTOCOL.major, 4);
}

/// Why 2.0 is a major bump: a 1.1 `StateChanged` (variant 1, the state's `u32`, the epoch) does
/// not read as itself under 2.0. It fails to decode or decodes as some other value, so a 1.x
/// peer cannot be read and the handshake must refuse it by protocol.
#[test]
fn a_1_1_state_changed_does_not_read_as_itself_under_2_0() {
    for word in 0..RunnerStateKind::ALL.len() as u32 {
        for (term, seq) in [(0u64, 0u64), (7, 2), (u64::MAX, 1)] {
            let mut old = vec![1u8];
            for value in [u64::from(word), term, seq] {
                let mut v = value;
                loop {
                    let byte = (v & 0x7f) as u8;
                    v >>= 7;
                    if v == 0 {
                        old.push(byte);
                        break;
                    }
                    old.push(byte | 0x80);
                }
            }
            let read_as_itself = matches!(
                postcard::from_bytes::<RunnerMsg>(&old),
                Ok(RunnerMsg::StateChanged { state, epoch })
                    if state.kind().discriminant() == word && epoch == PlanEpoch { term, seq }
            );
            assert!(!read_as_itself, "{old:?}");
        }
    }
    assert_ne!(PROTOCOL.major, 1);
}

#[test]
fn hello_and_its_envelope_keep_their_layout() {
    // Frozen (see `RunnerMsg::Hello`): id 1, no reply, variant 0, pid 7 as a varint, the build
    // ID, protocol 1.0 as varints, the digest.
    let envelope = Envelope {
        id: RequestId(1),
        reply_to: None,
        msg: RunnerMsg::Hello(Hello {
            pid: 7,
            build_id: [0xaa; 32],
            ipc_protocol: IpcProtocol { major: 1, minor: 0 },
            obs_tables: Blake3([0xbb; 32]),
        }),
    };
    let mut expected = vec![1, 0, 0, 7];
    expected.extend([0xaa; 32]);
    expected.extend([1, 0]);
    expected.extend([0xbb; 32]);
    let mut frame = (expected.len() as u32).to_le_bytes().to_vec();
    frame.extend(&expected);
    assert_eq!(encode_frame(&envelope).unwrap(), frame);
}

fn tier_id() -> impl Strategy<Value = TierId> {
    (any::<[u8; 32]>(), any::<u16>()).prop_map(|(node, ordinal)| TierId {
        node: syntropy_core::NodeId(node),
        ordinal,
    })
}

fn ring_role() -> impl Strategy<Value = RingRole> {
    prop_oneof![
        Just(RingRole::Telemetry),
        any::<u16>().prop_map(|dispatcher| RingRole::Step { dispatcher }),
        Just(RingRole::LocalTokens),
        Just(RingRole::LocalSequences),
        Just(RingRole::Log),
    ]
}

fn init() -> impl Strategy<Value = Init> {
    let ledger = prop::collection::btree_map(tier_id(), any::<u64>(), 0..5)
        .prop_map(|lines| LedgerGrant::new(lines.into_iter().map(|(t, b)| (t, Bytes(b)))).unwrap());
    let ring =
        (ring_role(), any::<u32>(), any::<u32>()).prop_map(|(role, slots, slot_bytes)| RingSpec {
            role,
            slots,
            slot_bytes,
        });
    (
        any::<[u8; 32]>(),
        any::<u64>(),
        ledger,
        prop::collection::vec(ring, 0..6),
    )
        .prop_map(|(build_id, instance, ledger, rings)| Init {
            build_id,
            instance: InstanceId(instance),
            ledger,
            shm: ShmLayout { rings },
        })
}

fn epoch() -> impl Strategy<Value = PlanEpoch> {
    (any::<u64>(), any::<u64>()).prop_map(|(term, seq)| PlanEpoch { term, seq })
}

/// The daemon messages whose fields are small enough to generate; the plan and the delta are
/// fixtures above.
fn daemon_msg() -> impl Strategy<Value = DaemonMsg> {
    prop_oneof![
        init().prop_map(DaemonMsg::Init),
        prop::collection::vec(any::<u16>().prop_map(StageIx), 0..5).prop_map(|stages| {
            DaemonMsg::AssignPlan {
                plan: Arc::new(fixtures::plan()),
                stages: stages.into(),
            }
        }),
        (any::<u64>(), any::<u64>(), any::<bool>()).prop_map(|(have, need, complete)| {
            DaemonMsg::ShardProgress {
                have: Bytes(have),
                need: Bytes(need),
                complete,
            }
        }),
        epoch().prop_map(|epoch| DaemonMsg::ApplyPlanDelta {
            epoch,
            delta: fixtures::delta(),
        }),
        (
            any::<(u64, u32, u32)>(),
            prop_oneof![
                Just(CancelReason::ClientGone),
                Just(CancelReason::ClientRequest),
                Just(CancelReason::StopString),
                Just(CancelReason::Deadline),
                any::<(u16, u16)>().prop_map(|(domain, code)| CancelReason::Error { domain, code }),
            ]
        )
            .prop_map(|((ingress, boot, n), reason)| DaemonMsg::CancelSequence {
                seq: SequenceId { ingress, boot, n },
                reason,
            }),
        any::<[u8; 32]>().prop_map(|node| DaemonMsg::PeerRevoked {
            node: syntropy_core::NodeId(node)
        }),
        Just(DaemonMsg::Ping),
        (drain_reason(), any::<u64>()).prop_map(|(reason, within)| DaemonMsg::Drain {
            reason,
            deadline: Deadline::within(Nanos(within)),
        }),
        any::<u64>().prop_map(|within| DaemonMsg::Shutdown {
            deadline: Deadline::within(Nanos(within)),
        }),
        (
            prop_oneof![
                Just(PressureLevel::Normal),
                Just(PressureLevel::Moderate),
                Just(PressureLevel::High),
                Just(PressureLevel::Critical),
            ],
            any::<u64>(),
            prop::option::of(tier_id()),
        )
            .prop_map(|(level, reclaim, tier)| DaemonMsg::MemoryPressure {
                level,
                reclaim: Bytes(reclaim),
                tier,
            }),
    ]
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

fn sequence_outcome() -> impl Strategy<Value = SequenceOutcome> {
    prop_oneof![
        Just(SequenceOutcome::Finished),
        prop_oneof![
            Just(ErrorClass::InvalidRequest),
            Just(ErrorClass::Capacity),
            Just(ErrorClass::Device),
            Just(ErrorClass::Timeout),
            Just(ErrorClass::Cancelled),
            Just(ErrorClass::Internal),
        ]
        .prop_map(SequenceOutcome::Failed),
        epoch().prop_map(|to| SequenceOutcome::Migrated { to }),
    ]
}

fn error_class() -> impl Strategy<Value = ErrorClass> {
    prop_oneof![
        Just(ErrorClass::InvalidRequest),
        Just(ErrorClass::Unsupported),
        Just(ErrorClass::Capacity),
        Just(ErrorClass::Transient),
        Just(ErrorClass::Link),
        Just(ErrorClass::Device),
        Just(ErrorClass::Integrity),
        Just(ErrorClass::Correctness),
        Just(ErrorClass::Drift),
        Just(ErrorClass::Timeout),
        Just(ErrorClass::Auth),
        Just(ErrorClass::Cancelled),
        Just(ErrorClass::Internal),
    ]
}

fn degradation() -> impl Strategy<Value = Degradation> {
    let transport = || {
        prop_oneof![
            Just(TransportKind::SharedMemory),
            Just(TransportKind::Quic),
            Just(TransportKind::QuicRelay),
            Just(TransportKind::Tcp),
        ]
    };
    let detail = prop_oneof![
        (
            prop::collection::vec(any::<u16>().prop_map(StageIx), 0..4),
            prop::collection::vec(
                prop_oneof![
                    Just(TierKind::Vram),
                    Just(TierKind::Unified),
                    Just(TierKind::HostPinned),
                    Just(TierKind::HostPageable),
                    Just(TierKind::Nvme),
                ],
                0..4
            ),
        )
            .prop_map(|(stages, tiers)| DegradeDetail::Absorbed {
                stages: stages.into(),
                tiers: tiers.into(),
            }),
        (tier_id(), tier_id(), any::<u8>(), transport(), transport()).prop_map(
            |(src, dst, path, from, to)| DegradeDetail::Demoted {
                edge: LinkId {
                    src: syntropy_core::Endpoint::Tier(src),
                    dst: syntropy_core::Endpoint::Tier(dst),
                    kind: LinkKind::Pcie,
                    path,
                },
                from,
                to,
            }
        ),
        (tier_id(), any::<u64>()).prop_map(|(tier, bytes)| DegradeDetail::Reduced {
            device: DeviceId {
                node: tier.node,
                ordinal: tier.ordinal,
            },
            bytes: Bytes(bytes),
        }),
    ];
    (any::<u32>(), prop::collection::vec(detail, 0..4)).prop_map(|(bits, detail)| Degradation {
        flags: DegradeFlags::from_bits_truncate(bits),
        detail: detail.into(),
    })
}

fn runner_state() -> impl Strategy<Value = RunnerState> {
    let kind = (0..RunnerStateKind::ALL.len()).prop_map(|i| RunnerStateKind::ALL[i]);
    prop_oneof![
        any::<(u64, u64)>().prop_map(|(have, need)| RunnerState::Fetching {
            have: Bytes(have),
            need: Bytes(need),
        }),
        any::<(u64, u64)>().prop_map(|(done, total)| RunnerState::Packing {
            done: Bytes(done),
            total: Bytes(total),
        }),
        any::<(u64, u64)>().prop_map(|(done, total)| RunnerState::Loading {
            done: Bytes(done),
            total: Bytes(total),
        }),
        prop_oneof![
            Just(WarmStep::Prepare),
            Just(WarmStep::Capture),
            Just(WarmStep::CacheFill),
            Just(WarmStep::LinkProbe),
            Just(WarmStep::Canary),
        ]
        .prop_map(|step| RunnerState::Warming { step }),
        (any::<u32>(), prop::option::of(degradation())).prop_map(|(live_sequences, degraded)| {
            RunnerState::Ready {
                live_sequences,
                degraded,
            }
        }),
        (drain_reason(), any::<u64>(), any::<u32>()).prop_map(
            |(reason, within, live_sequences)| RunnerState::Draining {
                reason,
                deadline: Deadline::within(Nanos(within)),
                live_sequences,
            }
        ),
        (error_class(), kind, any::<u32>()).prop_map(|(class, during, restarts)| {
            RunnerState::Failed {
                class,
                during,
                restarts,
            }
        }),
        prop_oneof![
            drain_reason().prop_map(StopReason::Drained),
            error_class().prop_map(StopReason::Fatal),
        ]
        .prop_map(|reason| RunnerState::Stopped { reason }),
    ]
}

fn runner_msg() -> impl Strategy<Value = RunnerMsg> {
    prop_oneof![
        (
            any::<u32>(),
            any::<[u8; 32]>(),
            any::<(u16, u16)>(),
            any::<[u8; 32]>()
        )
            .prop_map(|(pid, build_id, (major, minor), digest)| {
                RunnerMsg::Hello(Hello {
                    pid,
                    build_id,
                    ipc_protocol: IpcProtocol { major, minor },
                    obs_tables: Blake3(digest),
                })
            }),
        (runner_state(), epoch())
            .prop_map(|(state, epoch)| RunnerMsg::StateChanged { state, epoch }),
        Just(RunnerMsg::Pong),
        (any::<(u64, u32, u32)>(), sequence_outcome()).prop_map(|((ingress, boot, n), outcome)| {
            RunnerMsg::SequenceEnded {
                seq: SequenceId { ingress, boot, n },
                outcome,
            }
        }),
        (generated_shard_spec(), fetch_priority())
            .prop_map(|(spec, priority)| RunnerMsg::FetchShard { spec, priority }),
        (
            generated_shard_spec(),
            device_class(),
            any::<u16>(),
            any::<[u8; 32]>()
        )
            .prop_map(|(spec, class, layout, sidecar)| RunnerMsg::PackCommitted {
                key: PackKey::for_shard(&spec, class, LayoutVersion(layout)),
                sidecar: Blake3(sidecar),
            }),
    ]
}

fn generated_shard_spec() -> impl Strategy<Value = ShardSpec> {
    (
        any::<[u8; 32]>(),
        any::<[u8; 32]>(),
        prop::collection::vec(any::<u32>(), 0..16),
        prop::collection::vec(any::<[u8; 32]>(), 0..16),
        any::<[u8; 32]>(),
    )
        .prop_map(|(model, node, groups, chunks, digest)| ShardSpec {
            model: ModelId(Blake3(model)),
            node: NodeId(node),
            groups: groups.into_iter().map(TensorGroupId).collect(),
            chunks: chunks.into_iter().map(|c| ChunkId(Blake3(c))).collect(),
            digest: Blake3(digest),
        })
}

fn fetch_priority() -> impl Strategy<Value = FetchPriority> {
    prop_oneof![
        Just(FetchPriority::Urgent),
        Just(FetchPriority::Normal),
        Just(FetchPriority::Background),
    ]
}

fn device_class() -> impl Strategy<Value = DeviceClass> {
    prop_oneof![
        any::<u16>().prop_map(|sm| DeviceClass::Cuda { sm }),
        any::<u32>().prop_map(|gfx| DeviceClass::Hip { gfx }),
        any::<bool>().prop_map(|metal4| DeviceClass::Metal {
            family: AppleGpuFamily::Apple8,
            metal4,
        }),
        (any::<u32>(), any::<bool>(), any::<u8>()).prop_map(|(vendor, coopmat, subgroup)| {
            DeviceClass::Vulkan {
                vendor,
                coopmat,
                subgroup,
            }
        }),
        Just(DeviceClass::Cpu {
            isa: CpuIsa::Arm82Dot
        }),
        (any::<u32>(), any::<u32>()).prop_map(|(vendor, model)| DeviceClass::Npu { vendor, model }),
    ]
}

fn envelope<M: Debug>(msg: impl Strategy<Value = M>) -> impl Strategy<Value = Envelope<M>> {
    (any::<u64>(), any::<Option<u64>>(), msg).prop_map(|(id, reply_to, msg)| Envelope {
        id: RequestId(id),
        reply_to: reply_to.map(RequestId),
        msg,
    })
}

/// Decodes `input` as each direction's frame. The decoder must return, and each result must be
/// consistent with the input.
fn check_total(input: &[u8]) {
    fn one<M: DeserializeOwned>(input: &[u8]) {
        match decode_frame::<M>(input) {
            Ok((_, used)) => assert!((HEADER..=input.len()).contains(&used)),
            Err(FrameError::Incomplete { needed }) => {
                assert!(needed > 0);
                assert!(input.len() + needed <= HEADER + MAX_FRAME);
            }
            Err(FrameError::TooLarge { len }) => assert!(len as usize > MAX_FRAME),
            Err(FrameError::Malformed(_)) => {}
            Err(FrameError::Trailing { unread }) => assert!(unread > 0),
        }
    }
    one::<DaemonMsg>(input);
    one::<RunnerMsg>(input);
}

/// One valid frame of each message fixture, both directions.
fn valid_frames() -> Vec<Vec<u8>> {
    let mut frames: Vec<Vec<u8>> = daemon_msgs()
        .into_iter()
        .map(|msg| {
            encode_frame(&Envelope {
                id: RequestId(1),
                reply_to: None,
                msg,
            })
            .unwrap()
        })
        .collect();
    frames.extend(runner_msgs().into_iter().map(|msg| {
        encode_frame(&Envelope {
            id: RequestId(1),
            reply_to: Some(RequestId(9)),
            msg,
        })
        .unwrap()
    }));
    frames
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn generated_daemon_messages_round_trip(env in envelope(daemon_msg())) {
        frame_round_trip(&env);
    }

    #[test]
    fn generated_runner_messages_round_trip(env in envelope(runner_msg())) {
        frame_round_trip(&env);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    /// Decode is total over arbitrary bytes.
    #[test]
    fn decode_is_total_over_arbitrary_bytes(input in prop::collection::vec(any::<u8>(), 0..512)) {
        check_total(&input);
    }

    /// Arbitrary bytes behind a length prefix that matches them, so they reach postcard.
    #[test]
    fn decode_is_total_over_arbitrary_payloads(payload in prop::collection::vec(any::<u8>(), 0..512)) {
        let mut input = (payload.len() as u32).to_le_bytes().to_vec();
        input.extend(&payload);
        check_total(&input);
    }

    /// Valid frames with bytes changed, then cut short or with the length prefix rewritten to
    /// match what is left, so the damage reaches every part of the decoder.
    #[test]
    fn decode_is_total_over_damaged_frames(
        which in any::<prop::sample::Index>(),
        edits in prop::collection::vec((any::<prop::sample::Index>(), 1..=255u8), 0..6),
        cut in any::<prop::sample::Index>(),
        mode in 0..3u8,
    ) {
        let frames = valid_frames();
        let mut frame = frames[which.index(frames.len())].clone();
        for (at, flip) in &edits {
            let at = HEADER + at.index(frame.len() - HEADER);
            frame[at] ^= flip;
        }
        match mode {
            0 => {}
            1 => frame.truncate(cut.index(frame.len() + 1)),
            _ => {
                frame.truncate(HEADER + cut.index(frame.len() - HEADER + 1));
                let len = (frame.len() - HEADER) as u32;
                frame[..HEADER].copy_from_slice(&len.to_le_bytes());
            }
        }
        check_total(&frame);
    }
}

#[test]
fn each_frame_error_is_classified() {
    let frame = encode_frame(&Envelope {
        id: RequestId(1),
        reply_to: None,
        msg: RunnerMsg::Pong,
    })
    .unwrap();
    assert_eq!(
        decode_frame::<RunnerMsg>(&frame[..2]).unwrap_err(),
        FrameError::Incomplete { needed: 2 }
    );
    assert_eq!(
        decode_frame::<RunnerMsg>(&frame[..frame.len() - 1]).unwrap_err(),
        FrameError::Incomplete { needed: 1 }
    );
    let too_large = (MAX_FRAME as u32 + 1).to_le_bytes();
    assert_eq!(
        decode_frame::<RunnerMsg>(&too_large).unwrap_err(),
        FrameError::TooLarge {
            len: MAX_FRAME as u32 + 1
        }
    );
    // A variant index past the last, which serde's derive refuses.
    let bad_variant = [3, 0, 0, 0, 1, 0, 99];
    assert!(matches!(
        decode_frame::<RunnerMsg>(&bad_variant).unwrap_err(),
        FrameError::Malformed(postcard::Error::SerdeDeCustom)
    ));
    // `Pong` and one byte more inside the frame.
    let mut trailing = frame.clone();
    trailing.push(0);
    trailing[0] += 1;
    assert_eq!(
        decode_frame::<RunnerMsg>(&trailing).unwrap_err(),
        FrameError::Trailing { unread: 1 }
    );
    // A tier granted twice: the ledger's own check, through postcard. The tuple has `Init`'s
    // layout, with the ledger as its list of lines.
    let mut init = fixtures::init();
    let line = init.ledger.iter().next().unwrap();
    let doubled = encode_frame(&Envelope {
        id: RequestId(1),
        reply_to: None,
        msg: (
            0u8,
            (
                init.build_id,
                init.instance,
                vec![line, line],
                std::mem::take(&mut init.shm),
            ),
        ),
    })
    .unwrap();
    assert!(matches!(
        decode_frame::<DaemonMsg>(&doubled).unwrap_err(),
        FrameError::Malformed(postcard::Error::SerdeDeCustom)
    ));
}

/// A sequence that records the size hint its deserializer gave.
struct HintProbe(Option<usize>);

impl<'de> Deserialize<'de> for HintProbe {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<HintProbe, D::Error> {
        struct Probe;
        impl<'de> Visitor<'de> for Probe {
            type Value = HintProbe;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a sequence of bytes")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<HintProbe, A::Error> {
                let hint = seq.size_hint();
                while seq.next_element::<u8>()?.is_some() {}
                Ok(HintProbe(hint))
            }
        }
        deserializer.deserialize_seq(Probe)
    }
}

#[test]
fn a_declared_length_is_no_size_hint() {
    // Three elements. Postcard's own slice flavor passes the 3 on; the frame decoder does not.
    let seq = [3, 1, 2, 3];
    assert_eq!(postcard::from_bytes::<HintProbe>(&seq).unwrap().0, Some(3));
    let frame = [6, 0, 0, 0, 1, 0, 3, 1, 2, 3];
    let (envelope, _) = decode_frame::<HintProbe>(&frame).unwrap();
    assert_eq!(envelope.msg.0, None);
}

#[test]
fn a_forged_stage_count_fails_fast() {
    // `AssignPlan` whose stage list claims 2^28 entries in a frame far too short for them.
    let mut frame = encode_frame(&Envelope {
        id: RequestId(1),
        reply_to: None,
        msg: DaemonMsg::AssignPlan {
            plan: Arc::new(fixtures::plan()),
            stages: smallvec![],
        },
    })
    .unwrap();
    // The empty stage list is the frame's last byte, a varint 0.
    assert_eq!(frame.pop(), Some(0));
    frame.extend([0x80, 0x80, 0x80, 0x80, 0x01]);
    let len = (frame.len() - HEADER) as u32;
    frame[..HEADER].copy_from_slice(&len.to_le_bytes());
    assert!(matches!(
        decode_frame::<DaemonMsg>(&frame).unwrap_err(),
        FrameError::Malformed(postcard::Error::DeserializeUnexpectedEnd)
    ));
}

#[test]
fn an_endpoint_round_trips_through_text() {
    for endpoint in [
        Endpoint::Socket("/run/user/1000/syntropy/daemon.sock".into()),
        Endpoint::Pipe("syntropy-daemon".into()),
    ] {
        assert_eq!(endpoint.to_string().parse::<Endpoint>(), Ok(endpoint));
    }
    for bad in [
        "",
        "socket:",
        "pipe:",
        "pipe:a\\b",
        "tcp:1.2.3.4:5",
        "/run/daemon.sock",
    ] {
        assert!(bad.parse::<Endpoint>().is_err(), "{bad}");
    }
}

// ---- The channel ----------------------------------------------------------------------------

/// A fresh endpoint for one test.
fn endpoint() -> Endpoint {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let name = format!(
        "syntropy-ipc-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    if cfg!(windows) {
        Endpoint::Pipe(name)
    } else {
        Endpoint::Socket(std::env::temp_dir().join(name).join("cmd.sock"))
    }
}

/// Removes a socket endpoint's directory once its listener is gone.
fn remove_dir(endpoint: &Endpoint) {
    if let Endpoint::Socket(path) = endpoint {
        let _ = std::fs::remove_dir(path.parent().unwrap());
    }
}

/// A connected pair: the daemon's end and the runner's.
fn pair() -> (DaemonChannel, RunnerChannel) {
    let endpoint = endpoint();
    let mut listener = Listener::bind(&endpoint).expect("bind");
    let client = {
        let endpoint = endpoint.clone();
        thread::spawn(move || connect(&endpoint).expect("connect"))
    };
    let daemon = listener.accept().expect("accept");
    let runner = client.join().unwrap();
    drop(listener);
    remove_dir(&endpoint);
    (Channel::new(daemon), Channel::new(runner))
}

/// Reads the `len` raw bytes the other end wrote.
fn raw(stream: &mut impl std::io::Read, len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    stream.read_exact(&mut bytes).unwrap();
    bytes
}

/// Both transports carry the same bytes: every message, sent through a channel over the OS
/// socket and over the in-memory pair, arrives as exactly `encode_frame`'s frame (DIST-E1-T11).
#[test]
fn the_framing_is_byte_identical_over_both_transports() {
    let (mut os_daemon, mut os_runner) = pair();
    let (mut mem_daemon, mut mem_runner) = memory_channel();
    for (n, msg) in daemon_msgs().into_iter().enumerate() {
        let want = encode_frame(&Envelope {
            id: RequestId(n as u64 + 1),
            reply_to: None,
            msg: &msg,
        })
        .unwrap();
        os_daemon.send(&msg).unwrap();
        mem_daemon.send(&msg).unwrap();
        assert_eq!(
            raw(os_runner.stream_mut(), want.len()),
            want,
            "{msg:?} over the socket"
        );
        assert_eq!(
            raw(mem_runner.stream_mut(), want.len()),
            want,
            "{msg:?} in memory"
        );
    }
    for (n, msg) in runner_msgs().into_iter().enumerate() {
        let want = encode_frame(&Envelope {
            id: RequestId(n as u64 + 1),
            reply_to: Some(RequestId(7)),
            msg: &msg,
        })
        .unwrap();
        os_runner.reply(RequestId(7), &msg).unwrap();
        mem_runner.reply(RequestId(7), &msg).unwrap();
        assert_eq!(
            raw(os_daemon.stream_mut(), want.len()),
            want,
            "{msg:?} over the socket"
        );
        assert_eq!(
            raw(mem_daemon.stream_mut(), want.len()),
            want,
            "{msg:?} in memory"
        );
    }
    assert_eq!(mem_daemon.stream().available(), 0);
}

/// The in-process handshake keeps the process handshake's checks: the build, the protocol, and
/// the process ID, which for a runner in this process is this process's.
#[test]
fn the_in_process_handshake_checks_build_protocol_and_process() {
    let hello = |pid, build_id, ipc_protocol| Hello {
        pid,
        build_id,
        ipc_protocol,
        obs_tables: Blake3([0; 32]),
    };
    let me = std::process::id();
    let cases = [
        (hello(me, BUILD, PROTOCOL), true),
        (hello(me.wrapping_add(1), BUILD, PROTOCOL), false),
        (hello(me, [0xee; 32], PROTOCOL), false),
        (
            hello(
                me,
                BUILD,
                IpcProtocol {
                    major: PROTOCOL.major + 1,
                    minor: 0,
                },
            ),
            false,
        ),
    ];
    for (hello, accepted) in cases {
        let (mut daemon, mut runner) = memory_channel();
        runner.send(&RunnerMsg::Hello(hello)).unwrap();
        let result = daemon.handshake_in_process(&BUILD);
        assert_eq!(result.is_ok(), accepted, "{hello:?}: {result:?}");
        if accepted {
            assert_eq!(result.unwrap(), hello);
        }
    }
    // A runner that sends anything else first is refused.
    let (mut daemon, mut runner) = memory_channel();
    runner.send(&RunnerMsg::Pong).unwrap();
    assert!(matches!(
        daemon.handshake_in_process(&BUILD),
        Err(HandshakeError::Unexpected { expected: "Hello" })
    ));
}

#[test]
fn every_message_crosses_a_channel_with_its_request_id() {
    let (mut daemon, mut runner) = pair();
    let to_runner = daemon_msgs();
    let to_daemon = runner_msgs();
    let sender = {
        let to_runner = to_runner.clone();
        thread::spawn(move || {
            for (n, msg) in to_runner.iter().enumerate() {
                assert_eq!(daemon.send(msg).unwrap(), RequestId(n as u64 + 1));
            }
            daemon
        })
    };
    for (n, msg) in to_runner.into_iter().enumerate() {
        let envelope = runner.recv().unwrap();
        assert_eq!(envelope.id, RequestId(n as u64 + 1));
        assert_eq!(envelope.reply_to, None);
        assert_eq!(envelope.msg, msg);
    }
    let mut daemon = sender.join().unwrap();
    for msg in &to_daemon {
        runner.send(msg).unwrap();
    }
    for (n, msg) in to_daemon.into_iter().enumerate() {
        let envelope = daemon.recv().unwrap();
        assert_eq!((envelope.id, envelope.msg), (RequestId(n as u64 + 1), msg));
    }
}

#[test]
fn pong_names_the_ping_it_answers() {
    let (mut daemon, mut runner) = pair();
    daemon.send(&DaemonMsg::Ping).unwrap();
    let ping = daemon.send(&DaemonMsg::Ping).unwrap();
    assert_eq!(ping, RequestId(2));
    runner.recv().unwrap();
    let received = runner.recv().unwrap();
    assert_eq!(received.msg, DaemonMsg::Ping);
    let pong = runner.reply(received.id, &RunnerMsg::Pong).unwrap();
    let answer = daemon.recv().unwrap();
    assert_eq!(answer.id, pong);
    assert_eq!(answer.reply_to, Some(ping));
    assert_eq!(answer.msg, RunnerMsg::Pong);
}

#[test]
fn recv_classifies_a_closed_or_broken_stream() {
    // Closed between frames.
    let (daemon, mut runner) = pair();
    drop(daemon);
    assert!(matches!(runner.recv(), Err(RecvError::Closed)));

    // Closed inside a frame.
    let (mut daemon, mut runner) = pair();
    let frame = encode_frame(&Envelope {
        id: RequestId(1),
        reply_to: None,
        msg: DaemonMsg::Ping,
    })
    .unwrap();
    daemon.stream.write_all(&frame[..frame.len() - 1]).unwrap();
    drop(daemon);
    assert!(matches!(
        runner.recv(),
        Err(RecvError::Io(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof
    ));

    // A length over the limit, refused before any payload is read.
    let (mut daemon, mut runner) = pair();
    daemon
        .stream
        .write_all(&(MAX_FRAME as u32 + 1).to_le_bytes())
        .unwrap();
    assert!(matches!(
        runner.recv(),
        Err(RecvError::Frame(FrameError::TooLarge { .. }))
    ));

    // A malformed frame leaves the stream at the next frame.
    let (mut daemon, mut runner) = pair();
    daemon.stream.write_all(&[3, 0, 0, 0, 1, 0, 99]).unwrap();
    daemon.send(&DaemonMsg::Ping).unwrap();
    assert!(matches!(
        runner.recv(),
        Err(RecvError::Frame(FrameError::Malformed(_)))
    ));
    assert_eq!(runner.recv().unwrap().msg, DaemonMsg::Ping);
}

#[test]
fn the_runner_handshake_accepts_its_own_build() {
    let (mut daemon, mut runner) = pair();
    let daemon_side = thread::spawn(move || {
        let envelope = daemon.recv().unwrap();
        assert_eq!(envelope.msg, RunnerMsg::Hello(hello(BUILD)));
        daemon.send(&DaemonMsg::Init(fixtures::init())).unwrap();
    });
    assert_eq!(runner.handshake(&hello(BUILD)).unwrap(), fixtures::init());
    daemon_side.join().unwrap();
}

#[test]
fn the_runner_handshake_refuses_another_builds_init() {
    let (mut daemon, mut runner) = pair();
    let daemon_side = thread::spawn(move || {
        daemon.recv().unwrap();
        let mut init = fixtures::init();
        init.build_id = [0x33; 32];
        daemon.send(&DaemonMsg::Init(init)).unwrap();
    });
    let error = runner.handshake(&hello(BUILD)).unwrap_err();
    assert!(
        matches!(error, HandshakeError::Build { theirs, ours } if theirs == [0x33; 32] && ours == BUILD),
        "{error}"
    );
    daemon_side.join().unwrap();
}

#[test]
fn the_runner_handshake_needs_init_first() {
    let (mut daemon, mut runner) = pair();
    let daemon_side = thread::spawn(move || {
        daemon.recv().unwrap();
        daemon.send(&DaemonMsg::Ping).unwrap();
    });
    assert!(matches!(
        runner.handshake(&hello(BUILD)),
        Err(HandshakeError::Unexpected { expected: "Init" })
    ));
    daemon_side.join().unwrap();
}

#[test]
fn an_endpoint_of_another_os_is_unsupported() {
    let other = if cfg!(windows) {
        Endpoint::Socket("/tmp/syntropy-ipc.sock".into())
    } else {
        Endpoint::Pipe("syntropy-ipc".into())
    };
    assert_eq!(
        Listener::bind(&other).unwrap_err().kind(),
        std::io::ErrorKind::Unsupported
    );
    assert_eq!(
        connect(&other).unwrap_err().kind(),
        std::io::ErrorKind::Unsupported
    );
}

#[cfg(windows)]
#[test]
fn a_second_listener_on_a_pipe_name_is_refused() {
    let endpoint = endpoint();
    let _first = Listener::bind(&endpoint).expect("bind");
    // `FILE_FLAG_FIRST_PIPE_INSTANCE`: the name is taken.
    assert!(Listener::bind(&endpoint).is_err());
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
mod unix {
    use std::any::Any;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
    use std::os::unix::net::{UnixListener, UnixStream};

    use super::*;

    /// A listener of either kind, as the stale-socket cases bind it.
    pub(super) type Bind<'a> = &'a dyn Fn(&Endpoint) -> std::io::Result<Box<dyn Any>>;

    /// Held while a test creates sockets whose liveness it asserts, and while a test spawns a
    /// process. On Apple platforms std sets close-on-exec on a new socket only after `socket()`
    /// returns, so a process another test thread spawns in that instant inherits the socket and
    /// keeps it open past its `exec`. A "stale" socket a child holds is a live listener, and the
    /// takeover rightly refuses it: the macOS failure of #1972's hosted run, job 123735904097.
    pub(super) static FORK_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Takes [`FORK_GUARD`], whether or not a test that held it panicked.
    pub(super) fn fork_guard() -> std::sync::MutexGuard<'static, ()> {
        FORK_GUARD
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The error of a bind that must fail; `what` names what a success would mean.
    fn refused(result: std::io::Result<Box<dyn Any>>, what: &str) -> std::io::Error {
        match result {
            Ok(_) => panic!("{what}"),
            Err(error) => error,
        }
    }

    /// What `bind` does with what a path already holds (B1 of the review of #1898):
    /// - a stale socket, as a daemon killed without its cleanup leaves, is replaced, and the new
    ///   listener accepts;
    /// - a live listener's socket keeps the bind failing with `AddrInUse`, and stays;
    /// - a regular file, and a link to a stale socket, are left in place and the bind fails.
    pub(super) fn stale_socket_cases(bind: Bind<'_>) {
        // No process is spawned while these sockets exist (`FORK_GUARD`).
        let _guard = fork_guard();
        // 1. A stale socket: a std listener dropped without removing its file.
        let endpoint = endpoint();
        let path = socket_path(&endpoint).to_owned();
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(path.parent().unwrap())
            .unwrap();
        drop(UnixListener::bind(&path).unwrap());
        assert!(path.exists(), "the stand-in for a crash left no socket");
        let listener = bind(&endpoint).expect("a stale socket is taken over");
        UnixStream::connect(&path).expect("the new listener accepts");

        // 2. A live listener's socket: the bind fails, and the socket stays.
        let error = refused(bind(&endpoint), "a live socket was taken over");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse, "{error}");
        assert!(path.exists());
        drop(listener);
        assert!(!path.exists());

        // 3. A regular file is not a socket: left in place.
        std::fs::write(&path, b"not a socket").unwrap();
        let error = refused(bind(&endpoint), "a regular file was replaced");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse, "{error}");
        assert_eq!(std::fs::read(&path).unwrap(), b"not a socket");
        std::fs::remove_file(&path).unwrap();

        // 4. A link to a stale socket: the link is not followed, and both stay.
        let target = path.with_extension("target");
        drop(UnixListener::bind(&target).unwrap());
        std::os::unix::fs::symlink(&target, &path).unwrap();
        let error = refused(bind(&endpoint), "a link was replaced");
        assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse, "{error}");
        assert!(
            std::fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(std::fs::symlink_metadata(&target).is_ok());
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&target).unwrap();
        remove_dir(&endpoint);
    }

    #[test]
    fn bind_takes_over_a_stale_socket_and_nothing_else() {
        stale_socket_cases(&|endpoint| {
            Listener::bind(endpoint).map(|l| Box::new(l) as Box<dyn Any>)
        });
    }

    pub(super) fn socket_path(endpoint: &Endpoint) -> &std::path::Path {
        match endpoint {
            Endpoint::Socket(path) => path,
            Endpoint::Pipe(_) => unreachable!(),
        }
    }

    #[test]
    fn bind_creates_the_directory_owner_only_and_removes_the_socket_on_drop() {
        let endpoint = endpoint();
        let path = socket_path(&endpoint).to_owned();
        let listener = Listener::bind(&endpoint).expect("bind");
        let dir = std::fs::metadata(path.parent().unwrap()).unwrap();
        assert_eq!(dir.mode() & 0o777, 0o700);
        assert!(path.exists());
        drop(listener);
        assert!(!path.exists());
        remove_dir(&endpoint);
    }

    #[test]
    fn bind_refuses_a_directory_others_can_enter() {
        let endpoint = endpoint();
        let path = socket_path(&endpoint);
        let dir = path.parent().unwrap();
        std::fs::create_dir(dir).unwrap();
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let error = Listener::bind(&endpoint).unwrap_err();
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied,
            "{error}"
        );
        assert!(!path.exists());
        remove_dir(&endpoint);
    }

    #[test]
    fn bind_refuses_a_link_to_a_directory() {
        let endpoint = endpoint();
        let path = socket_path(&endpoint);
        let dir = path.parent().unwrap();
        let target = dir.with_extension("target");
        std::fs::create_dir(&target).unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::os::unix::fs::symlink(&target, dir).unwrap();
        let error = Listener::bind(&endpoint).unwrap_err();
        assert_eq!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied,
            "{error}"
        );
        std::fs::remove_file(dir).unwrap();
        std::fs::remove_dir(&target).unwrap();
    }
}
