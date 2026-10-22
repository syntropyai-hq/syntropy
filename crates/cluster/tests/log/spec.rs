// SPDX-License-Identifier: MIT OR Apache-2.0

//! The golden hash-chain vectors. The chain is computed here from RFC-0032 §6.1's formula with
//! `blake3` directly, not with `syntropy_cluster::log::chain`. The event bytes are the
//! envelope's CBOR of a fixed log, and the state hash is the fold of that log, so the file also
//! pins the events' encoding and `ClusterState`'s.

use std::fmt::Write as _;

use syntropy_cluster::log::payload::{
    ConfigPatch, ManifestDigest, NodeInfo, Opaque, PlanBody, PlanSummary, RunnerStateKind,
    StageSummary,
};
use syntropy_cluster::log::{ClusterEvent, ClusterState, CommandId, LogEntry, apply, state_hash};
use syntropy_core::policy::{ConfigValue, KeyPath};
use syntropy_core::{
    Blake3, Bytes, InstanceId, LayerIx, ModelId, NodeId, PlanDigest, PlanEpoch, SessionId,
};
use syntropy_net::ctl::envelope;

/// The cluster key's public half in the vectors: bytes 0x00 to 0x1f.
pub fn cluster() -> [u8; 32] {
    std::array::from_fn(|i| i as u8)
}

/// One entry of the fixed log.
pub struct Step {
    pub index: u64,
    pub term: u64,
    pub cause: Option<CommandId>,
    pub event: ClusterEvent,
    pub event_bytes: Vec<u8>,
    pub chain: [u8; 32],
}

/// The fixed log, its chain and the state it folds to.
pub struct Vectors {
    pub cluster: [u8; 32],
    pub genesis: [u8; 32],
    pub steps: Vec<Step>,
    pub state_hash: Blake3,
}

/// chain(i) = BLAKE3(chain(i-1) ‖ u64_le(index) ‖ u64_le(term) ‖ cause ‖ event_bytes), with
/// cause = 00, or 01 ‖ origin ‖ u64_le(counter).
pub fn chain_input(
    prev: &[u8; 32],
    index: u64,
    term: u64,
    cause: Option<&CommandId>,
    event_bytes: &[u8],
) -> Vec<u8> {
    let mut input = Vec::new();
    input.extend_from_slice(prev);
    input.extend_from_slice(&index.to_le_bytes());
    input.extend_from_slice(&term.to_le_bytes());
    match cause {
        None => input.push(0),
        Some(cause) => {
            input.push(1);
            input.extend_from_slice(&cause.origin.0);
            input.extend_from_slice(&cause.counter.to_le_bytes());
        }
    }
    input.extend_from_slice(event_bytes);
    input
}

fn n(byte: u8) -> NodeId {
    NodeId([byte; 32])
}

fn cause(origin: u8, counter: u64) -> Option<CommandId> {
    Some(CommandId {
        origin: n(origin),
        counter,
    })
}

/// The fixed log: an election, an admission, a model brought up to routable with a session on
/// it, a configuration change, and a member leaving.
fn log() -> Vec<(u64, Option<CommandId>, ClusterEvent)> {
    let model = ModelId(Blake3([0x4d; 32]));
    let instance = InstanceId(1);
    let epoch = PlanEpoch { term: 1, seq: 0 };
    vec![
        (
            1,
            None,
            ClusterEvent::CoordinatorElected {
                node: n(1),
                term: 1,
                lease_until_ms: 10_000,
            },
        ),
        (
            1,
            cause(2, 1),
            ClusterEvent::NodeAdmitted {
                info: NodeInfo { id: n(2) },
            },
        ),
        (
            1,
            cause(1, 1),
            ClusterEvent::ModelRegistered {
                model,
                manifest: ManifestDigest(Blake3([0x6d; 32])),
            },
        ),
        (
            1,
            cause(1, 2),
            ClusterEvent::InstanceRequested {
                instance,
                model,
                policy: Box::default(),
            },
        ),
        (
            1,
            None,
            ClusterEvent::PlanCommitted {
                plan: PlanBody::ByDigest {
                    digest: PlanDigest(Blake3([0x70; 32])),
                    len: Bytes(70_000),
                },
                summary: PlanSummary {
                    instance,
                    epoch,
                    stages: vec![
                        StageSummary {
                            node: n(1),
                            layers: LayerIx(0)..LayerIx(16),
                        },
                        StageSummary {
                            node: n(2),
                            layers: LayerIx(16)..LayerIx(32),
                        },
                    ],
                },
            },
        ),
        (
            1,
            cause(1, 3),
            ClusterEvent::RunnerStateChanged {
                instance,
                node: n(1),
                epoch,
                state: RunnerStateKind::Ready,
            },
        ),
        (
            1,
            None,
            ClusterEvent::InstanceRoutable {
                instance,
                epoch,
                context_length: 32_768,
            },
        ),
        (
            1,
            cause(2, 2),
            ClusterEvent::SessionPinned {
                session: SessionId([0x53; 16]),
                instance,
                holders: vec![n(1)],
            },
        ),
        (
            1,
            cause(1, 4),
            ClusterEvent::ClusterConfigChanged {
                patch: ConfigPatch {
                    set: vec![(KeyPath::from("allow_approximate"), ConfigValue::Bool(true))],
                    unset: vec![],
                    origin: Opaque(ciborium::Value::Map(vec![(
                        ciborium::Value::Text("layer".to_owned()),
                        ciborium::Value::Text("file".to_owned()),
                    )])),
                },
            },
        ),
        (2, cause(2, 3), ClusterEvent::NodeLeft { node: n(2) }),
    ]
}

/// The committed vectors.
pub fn vectors() -> Vectors {
    let cluster = cluster();
    let genesis = *blake3::hash(&cluster).as_bytes();
    let mut prev = genesis;
    let mut state = ClusterState::new(cluster);
    let mut steps = Vec::new();
    for (i, (term, cause, event)) in log().into_iter().enumerate() {
        let index = i as u64 + 1;
        let event_bytes = envelope::to_cbor(&event).expect("the event encodes");
        let chain = *blake3::hash(&chain_input(
            &prev,
            index,
            term,
            cause.as_ref(),
            &event_bytes,
        ))
        .as_bytes();
        let entry = LogEntry::append(&Blake3(prev), index, term, 0, cause, event.clone())
            .expect("the event encodes");
        apply(&mut state, &entry);
        steps.push(Step {
            index,
            term,
            cause,
            event,
            event_bytes,
            chain,
        });
        prev = chain;
    }
    Vectors {
        cluster,
        genesis,
        steps,
        state_hash: state_hash(&state).expect("the state encodes"),
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn variant(event: &ClusterEvent) -> String {
    let debug = format!("{event:?}");
    debug
        .split([' ', '{', '('])
        .next()
        .unwrap_or_default()
        .to_owned()
}

/// The vectors file.
pub fn render(v: &Vectors) -> String {
    let mut text = String::from(
        "# Event log hash-chain vectors (RFC-0032 §6.1), frozen by DIST-E9-T01.\n\
         # Generated by crates/cluster/tests/log/spec.rs. A change to this file changes the\n\
         # chain, the base events' CBOR or ClusterState's CBOR.\n\
         #\n\
         # chain(0) = BLAKE3(cluster)\n\
         # chain(i) = BLAKE3(chain(i-1) ‖ u64_le(index) ‖ u64_le(term) ‖ cause ‖ event_bytes)\n\
         # cause    = 00, or 01 ‖ origin (32 bytes) ‖ u64_le(counter)\n\
         #\n\
         # cluster is the cluster key's public half. event_bytes are the event's CBOR as the\n\
         # coordinator encodes it; state_hash is BLAKE3 of the CBOR of the state the log folds\n\
         # to. Hex is lowercase, byte order as hashed.\n\n",
    );
    let _ = writeln!(text, "cluster     = {}", hex(&v.cluster));
    let _ = writeln!(text, "chain(0)    = {}", hex(&v.genesis));
    for step in &v.steps {
        let _ = writeln!(text, "\n[{}] {}", step.index, variant(&step.event));
        let _ = writeln!(text, "term        = {}", step.term);
        let cause = match &step.cause {
            None => "none".to_owned(),
            Some(c) => format!("{} {}", hex(&c.origin.0), c.counter),
        };
        let _ = writeln!(text, "cause       = {cause}");
        let _ = writeln!(text, "event_bytes = {}", hex(&step.event_bytes));
        let _ = writeln!(text, "chain       = {}", hex(&step.chain));
    }
    let _ = writeln!(text, "\nstate_hash  = {}", hex(&v.state_hash.0));
    text
}
