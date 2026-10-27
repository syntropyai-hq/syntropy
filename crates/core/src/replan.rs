// SPDX-License-Identifier: MIT OR Apache-2.0

//! Why a re-plan was asked for: [`ReplanReason`], with its term types [`DriftTerm`] and
//! [`WorkloadTerm`] (RFC-0033 §6.1, note of 2026-10-23; RFC-0010 §6.2, §7.1, §7.3; RFC-0001
//! §5.13, §18.33; DIST-E14-T21).
//!
//! Plain data with serde, in `core` so one definition serves every crate that requests, starts or
//! records a re-plan. `cluster` carries it in RFC-0001 §5.10's `RequestReplan`, and the solver
//! reads it in RFC-0011 §2's `SolveReason::Replan`. The solver is layer 2 and cannot depend on
//! `cluster` (RFC-0001 §4.2).
//!
//! Variants are only ever appended. Postcard, the encoding of the IPC and of the event log, tags
//! a variant by its index, so reordering or removing one changes what recorded bytes mean.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ids::{DeviceId, LinkId, NodeId, StageIx, TierId};
use crate::policy::KeyPath;
use crate::quant::QuantScheme;
use crate::resource::{KernelClass, TransportKind};

/// Why the coordinator asks the solver for a new plan (RFC-0033 §6.1). Owner: RFC-0033.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReplanReason {
    /// Stages lost with the nodes that held them (RFC-0033 §6.1). The solver gets
    /// `RepairConstraints` with it.
    StageLost {
        /// The lost stages.
        stages: SmallVec<[StageIx; 2]>,
        /// The nodes that held them.
        nodes: SmallVec<[NodeId; 2]>,
    },
    /// A runner failed on a live node.
    RunnerFailed {
        /// The runner's node.
        node: NodeId,
        /// The device it ran on, when it had one.
        device: Option<DeviceId>,
    },
    /// Both nodes are alive but the path between them is gone: relink over another transport,
    /// every shard kept (RFC-0033 §3.3; RFC-0001 §7.2).
    EdgeLost {
        /// The edge's source node.
        src: NodeId,
        /// The edge's destination node.
        dst: NodeId,
    },
    /// A node joined the cluster (RFC-0033 §10).
    NodeJoined {
        /// The node that joined.
        node: NodeId,
    },
    /// A profile quantity of `node` drifted past `replan.drift_threshold` (RFC-0010 §7.2), or one
    /// of its hops was demoted to another transport (RFC-0033 §6.1, note of 2026-10-23).
    Drift {
        /// The node whose quantity drifted.
        node: NodeId,
        /// The quantity.
        term: DriftTerm,
    },
    /// A node's memory budget fell below what its instances hold (RFC-0001 §10.3).
    MemoryPressure {
        /// The node under pressure.
        node: NodeId,
    },
    /// A workload term is off what the plan assumed (RFC-0010 §7.1, §7.3). The current curves
    /// arrive separately in `ReportWorkload` (RFC-0032 §6.5).
    WorkloadDrift {
        /// The term.
        term: WorkloadTerm,
    },
    /// A configuration change the solver reads (RFC-0054 amendment A4).
    Config {
        /// The changed keys, so the dashboard and logs can say which change caused the re-plan.
        keys: Vec<KeyPath>,
    },
    /// A user asked: a load, an unload or a policy change.
    User,
    /// After a partition heals, orphaned runners are adopted by a new plan or drained (RFC-0032
    /// §7.5).
    HealOrphans,
    /// A degraded mode ends (RFC-0033 §7.2).
    DegradedExit,
}

/// The profile quantity whose drift caused a re-plan: one variant per live estimator of RFC-0010
/// §6.2. A later estimator appends its own. Owner: RFC-0010 (the terms), RFC-0031
/// ([`DriftTerm::Hop`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DriftTerm {
    /// A device's W_eff from one tier for one kernel class and quant scheme, as when another app
    /// takes bandwidth or the power source changes. §6.2's estimator is keyed by all four.
    Bandwidth {
        /// The device.
        device: DeviceId,
        /// The tier it reads from.
        tier: TierId,
        /// The kernel class.
        class: KernelClass,
        /// The quant scheme.
        scheme: QuantScheme,
    },
    /// A device's `t_fixed`, the intercept of the per-step fit.
    FixedTime {
        /// The device.
        device: DeviceId,
    },
    /// A storage tier's sustained read, as under thermal throttling.
    Storage {
        /// The tier.
        tier: TierId,
    },
    /// The pinned host↔device rate over the device's PCIe link or tunnel, the figure
    /// `HostLinkCaps` holds, as when a link trains down.
    HostLink {
        /// The device.
        device: DeviceId,
    },
    /// A hop's α, β, wireless mean or stall rate, or its demotion to another transport (RFC-0031
    /// §3, §7.4).
    Hop {
        /// The hop's link.
        link: LinkId,
        /// The transport whose figures drifted. For a demotion it is the planned transport,
        /// `Demotion::from`: the figure the plan assumed and lost. The new transport's cost is in
        /// the graph the next solve reads.
        transport: TransportKind,
    },
}

/// A workload term off what the plan assumed: not a profile quantity, so the solver reads the
/// current curves when it re-solves (RFC-0010 §7.1, §7.3). Owner: RFC-0010.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadTerm {
    /// The MoE miss term is above its prediction: the hit rate is below the prior.
    ExpertMiss,
    /// A cache tier's measured hit rate is below the hit curve at its budget (RFC-0023 §13).
    ExpertTierShortfall {
        /// The tier.
        tier: TierId,
    },
    /// Tokens per verify window are below the prediction (RFC-0012).
    SpeculativeAcceptance,
}

#[cfg(test)]
mod tests {
    //! Every variant round-trips. The names below come from exhaustive matches, so a new variant
    //! fails to compile here until it has a sample, even on the `#[non_exhaustive]` enum.

    use std::collections::BTreeSet;
    use std::fmt::Debug;

    use serde::Serialize;
    use serde::de::DeserializeOwned;
    use smallvec::smallvec;

    use super::*;
    use crate::dtype::DType;
    use crate::ids::{Endpoint, LinkKind, NetMedium};

    fn name_of_reason(reason: &ReplanReason) -> &'static str {
        match reason {
            ReplanReason::StageLost { .. } => "StageLost",
            ReplanReason::RunnerFailed { .. } => "RunnerFailed",
            ReplanReason::EdgeLost { .. } => "EdgeLost",
            ReplanReason::NodeJoined { .. } => "NodeJoined",
            ReplanReason::Drift { .. } => "Drift",
            ReplanReason::MemoryPressure { .. } => "MemoryPressure",
            ReplanReason::WorkloadDrift { .. } => "WorkloadDrift",
            ReplanReason::Config { .. } => "Config",
            ReplanReason::User => "User",
            ReplanReason::HealOrphans => "HealOrphans",
            ReplanReason::DegradedExit => "DegradedExit",
        }
    }

    fn name_of_drift(term: &DriftTerm) -> &'static str {
        match term {
            DriftTerm::Bandwidth { .. } => "Bandwidth",
            DriftTerm::FixedTime { .. } => "FixedTime",
            DriftTerm::Storage { .. } => "Storage",
            DriftTerm::HostLink { .. } => "HostLink",
            DriftTerm::Hop { .. } => "Hop",
        }
    }

    fn name_of_workload(term: &WorkloadTerm) -> &'static str {
        match term {
            WorkloadTerm::ExpertMiss => "ExpertMiss",
            WorkloadTerm::ExpertTierShortfall { .. } => "ExpertTierShortfall",
            WorkloadTerm::SpeculativeAcceptance => "SpeculativeAcceptance",
        }
    }

    const REASONS: &[&str] = &[
        "StageLost",
        "RunnerFailed",
        "EdgeLost",
        "NodeJoined",
        "Drift",
        "MemoryPressure",
        "WorkloadDrift",
        "Config",
        "User",
        "HealOrphans",
        "DegradedExit",
    ];
    const DRIFTS: &[&str] = &["Bandwidth", "FixedTime", "Storage", "HostLink", "Hop"];
    const WORKLOADS: &[&str] = &["ExpertMiss", "ExpertTierShortfall", "SpeculativeAcceptance"];

    fn node(byte: u8) -> NodeId {
        NodeId([byte; 32])
    }

    fn device() -> DeviceId {
        DeviceId {
            node: node(1),
            ordinal: 2,
        }
    }

    fn tier() -> TierId {
        TierId {
            node: node(1),
            ordinal: 3,
        }
    }

    fn drift_terms() -> Vec<DriftTerm> {
        vec![
            DriftTerm::Bandwidth {
                device: device(),
                tier: tier(),
                class: KernelClass::ExpertGemv,
                scheme: QuantScheme::Plain(DType::BF16),
            },
            DriftTerm::FixedTime { device: device() },
            DriftTerm::Storage { tier: tier() },
            DriftTerm::HostLink { device: device() },
            DriftTerm::Hop {
                link: LinkId {
                    src: Endpoint::Device(device()),
                    dst: Endpoint::Nic {
                        node: node(4),
                        ordinal: 0,
                    },
                    kind: LinkKind::Network(NetMedium::WiFi),
                    path: 1,
                },
                transport: TransportKind::Quic,
            },
        ]
    }

    fn workload_terms() -> Vec<WorkloadTerm> {
        vec![
            WorkloadTerm::ExpertMiss,
            WorkloadTerm::ExpertTierShortfall { tier: tier() },
            WorkloadTerm::SpeculativeAcceptance,
        ]
    }

    fn reasons() -> Vec<ReplanReason> {
        let mut reasons = vec![
            ReplanReason::StageLost {
                stages: smallvec![StageIx(0), StageIx(3)],
                nodes: smallvec![node(5)],
            },
            ReplanReason::RunnerFailed {
                node: node(6),
                device: Some(device()),
            },
            ReplanReason::RunnerFailed {
                node: node(6),
                device: None,
            },
            ReplanReason::EdgeLost {
                src: node(7),
                dst: node(8),
            },
            ReplanReason::NodeJoined { node: node(9) },
            ReplanReason::MemoryPressure { node: node(10) },
            ReplanReason::Config {
                keys: vec![
                    KeyPath("replan.drift_threshold".into()),
                    KeyPath("node.labels".into()),
                ],
            },
            ReplanReason::Config { keys: Vec::new() },
            ReplanReason::User,
            ReplanReason::HealOrphans,
            ReplanReason::DegradedExit,
        ];
        reasons.extend(drift_terms().into_iter().map(|term| ReplanReason::Drift {
            node: node(11),
            term,
        }));
        reasons.extend(
            workload_terms()
                .into_iter()
                .map(|term| ReplanReason::WorkloadDrift { term }),
        );
        reasons
    }

    fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
        let bytes = postcard::to_allocvec(value).expect("serialize");
        let back: T = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(&back, value, "postcard round trip through {bytes:?}");
        let json = serde_json::to_string(value).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&back, value, "JSON round trip through {json}");
    }

    fn names<T>(values: &[T], name: fn(&T) -> &'static str) -> BTreeSet<&'static str> {
        values.iter().map(name).collect()
    }

    #[test]
    fn every_variant_round_trips_through_postcard_and_json() {
        let reasons = reasons();
        assert_eq!(
            names(&reasons, name_of_reason),
            REASONS.iter().copied().collect()
        );
        assert_eq!(
            names(&drift_terms(), name_of_drift),
            DRIFTS.iter().copied().collect()
        );
        assert_eq!(
            names(&workload_terms(), name_of_workload),
            WORKLOADS.iter().copied().collect()
        );
        for reason in &reasons {
            round_trip(reason);
        }
        for term in drift_terms() {
            round_trip(&term);
        }
        for term in workload_terms() {
            round_trip(&term);
        }
    }

    /// JSON names each variant, so a log or a dashboard reads which change caused the re-plan.
    #[test]
    fn json_names_the_variant_and_the_changed_keys() {
        let json = serde_json::to_value(ReplanReason::Config {
            keys: vec![KeyPath("replan.drift_threshold".into())],
        })
        .unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "Config": { "keys": ["replan.drift_threshold"] } })
        );
        let json = serde_json::to_value(ReplanReason::WorkloadDrift {
            term: WorkloadTerm::ExpertTierShortfall { tier: tier() },
        })
        .unwrap();
        assert_eq!(
            json["WorkloadDrift"]["term"]["ExpertTierShortfall"]["tier"]["ordinal"],
            3
        );
        assert_eq!(serde_json::to_value(ReplanReason::User).unwrap(), "User");
    }

    /// Postcard tags a variant by its index: the order is RFC-0033 §6.1's, and a variant is only
    /// appended. These pin the first byte of each, so a reorder fails here.
    #[test]
    fn postcard_indices_follow_the_rfc_order() {
        for (index, reason) in [
            ReplanReason::StageLost {
                stages: SmallVec::new(),
                nodes: SmallVec::new(),
            },
            ReplanReason::RunnerFailed {
                node: node(0),
                device: None,
            },
            ReplanReason::EdgeLost {
                src: node(0),
                dst: node(0),
            },
            ReplanReason::NodeJoined { node: node(0) },
            ReplanReason::Drift {
                node: node(0),
                term: DriftTerm::FixedTime { device: device() },
            },
            ReplanReason::MemoryPressure { node: node(0) },
            ReplanReason::WorkloadDrift {
                term: WorkloadTerm::ExpertMiss,
            },
            ReplanReason::Config { keys: Vec::new() },
            ReplanReason::User,
            ReplanReason::HealOrphans,
            ReplanReason::DegradedExit,
        ]
        .iter()
        .enumerate()
        {
            let bytes = postcard::to_allocvec(reason).unwrap();
            assert_eq!(usize::from(bytes[0]), index, "{}", name_of_reason(reason));
        }
        for (index, term) in drift_terms().iter().enumerate() {
            assert_eq!(usize::from(postcard::to_allocvec(term).unwrap()[0]), index);
        }
        for (index, term) in workload_terms().iter().enumerate() {
            assert_eq!(usize::from(postcard::to_allocvec(term).unwrap()[0]), index);
        }
    }
}
