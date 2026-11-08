// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serde properties of the placement plan types (PLC-E4-T17, RFC-0001 §5.4, RFC-0011 §2).
//!
//! Every type round-trips through a serde format unchanged, and every record decodes when it
//! carries fields this build does not know. JSON stands in for any self-describing format, and
//! postcard, which the IPC uses, for the formats that are not.

use std::fmt::Debug;

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use serde::Serialize;
use serde::de::DeserializeOwned;
use smallvec::SmallVec;
use syntropy_core::plan::*;
use syntropy_core::resource::{TierKind, TransportKind};
use syntropy_core::*;

// ---- Strategies -------------------------------------------------------------------------

fn f32s() -> impl Strategy<Value = f32> + Clone {
    -1.0e6f32..1.0e6
}

fn node_id() -> impl Strategy<Value = NodeId> {
    any::<[u8; 32]>().prop_map(NodeId)
}

fn blake3() -> impl Strategy<Value = Blake3> {
    any::<[u8; 32]>().prop_map(Blake3)
}

fn device_id() -> impl Strategy<Value = DeviceId> {
    (node_id(), any::<u16>()).prop_map(|(node, ordinal)| DeviceId { node, ordinal })
}

fn tier_id() -> impl Strategy<Value = TierId> {
    (node_id(), any::<u16>()).prop_map(|(node, ordinal)| TierId { node, ordinal })
}

fn rdma_kind() -> impl Strategy<Value = RdmaKind> {
    prop_oneof![
        Just(RdmaKind::RoceV2),
        Just(RdmaKind::InfiniBand),
        Just(RdmaKind::Thunderbolt5)
    ]
}

fn link_id() -> impl Strategy<Value = LinkId> {
    let endpoint = prop_oneof![
        device_id().prop_map(Endpoint::Device),
        tier_id().prop_map(Endpoint::Tier),
    ];
    let kind = prop_oneof![
        Just(LinkKind::MemoryBus),
        Just(LinkKind::Pcie),
        Just(LinkKind::Storage),
        Just(LinkKind::Network(NetMedium::WiFi)),
        rdma_kind().prop_map(|r| LinkKind::Network(NetMedium::Rdma(r))),
    ];
    (endpoint.clone(), endpoint, kind, any::<u8>()).prop_map(|(src, dst, kind, path)| LinkId {
        src,
        dst,
        kind,
        path,
    })
}

fn bytes() -> impl Strategy<Value = Bytes> {
    any::<u64>().prop_map(Bytes)
}

fn nanos() -> impl Strategy<Value = Nanos> {
    any::<u64>().prop_map(Nanos)
}

fn layer() -> impl Strategy<Value = LayerIx> {
    any::<u16>().prop_map(LayerIx)
}

fn stage_ix() -> impl Strategy<Value = StageIx> {
    any::<u16>().prop_map(StageIx)
}

fn group_id() -> impl Strategy<Value = TensorGroupId> {
    any::<u32>().prop_map(TensorGroupId)
}

fn small<A: smallvec::Array + Debug + 'static>(
    s: impl Strategy<Value = A::Item>,
) -> impl Strategy<Value = SmallVec<A>>
where
    A::Item: Debug,
{
    // Up to two past the inline capacity, so both the inline and the spilled forms occur.
    prop::collection::vec(s, 0..A::size() + 3).prop_map(SmallVec::from_vec)
}

fn plan_epoch() -> impl Strategy<Value = PlanEpoch> {
    (any::<u64>(), any::<u64>()).prop_map(|(term, seq)| PlanEpoch { term, seq })
}

fn objective() -> impl Strategy<Value = Objective> {
    prop_oneof![
        Just(Objective::Latency),
        (
            f32s(),
            proptest::option::of(any::<u16>()),
            proptest::option::of(any::<u64>().prop_map(Nanos))
        )
            .prop_map(
                |(per_sequence_floor_tps, target_concurrency, ttft_ceiling)| {
                    Objective::Throughput {
                        per_sequence_floor_tps,
                        target_concurrency,
                        ttft_ceiling,
                    }
                }
            ),
    ]
}

fn exactness() -> impl Strategy<Value = ExactnessPolicy> {
    prop_oneof![
        Just(ExactnessPolicy::Exact),
        any::<u32>().prop_map(|bits| ExactnessPolicy::Approximate {
            knobs: ApproxKnobs::from_bits_truncate(bits),
            params: ApproxParams::default(),
        }),
    ]
}

fn transport() -> impl Strategy<Value = TransportKind> {
    prop_oneof![
        Just(TransportKind::SharedMemory),
        rdma_kind().prop_map(TransportKind::Rdma),
        Just(TransportKind::Quic),
        Just(TransportKind::QuicRelay),
        Just(TransportKind::Tcp),
    ]
}

fn wire_dtype() -> impl Strategy<Value = WireDType> {
    prop_oneof![
        Just(WireDType::F32),
        Just(WireDType::BF16),
        Just(WireDType::F16),
        Just(WireDType::Q8B32),
        Just(WireDType::F8E4M3B32),
    ]
}

fn relay_via() -> impl Strategy<Value = RelayVia> {
    (node_id(), link_id(), link_id(), transport()).prop_map(|(node, first, second, transport)| {
        RelayVia {
            node,
            first,
            second,
            transport,
        }
    })
}

fn hop_plan() -> impl Strategy<Value = HopPlan> {
    (
        (link_id(), transport(), wire_dtype(), nanos()),
        (
            prop::option::of((link_id(), transport())),
            prop::option::of(relay_via()),
            nanos(),
        ),
    )
        .prop_map(
            |((link, transport, wire, predicted), (prefill_transport, relay, predicted_p99))| {
                HopPlan {
                    link,
                    transport,
                    wire,
                    predicted,
                    prefill_transport,
                    relay,
                    predicted_p99,
                }
            },
        )
}

fn tp_split() -> impl Strategy<Value = TpSplit> {
    (
        small(any::<u16>()),
        small(any::<u32>()),
        prop::option::of(small(any::<u32>())),
    )
        .prop_map(|(attn, ffn, experts)| TpSplit { attn, ffn, experts })
}

fn stage_member() -> impl Strategy<Value = StageMember> {
    (node_id(), small(device_id()), link_id()).prop_map(|(node, devices, link)| StageMember {
        node,
        devices,
        link,
    })
}

fn expert_owner_map() -> impl Strategy<Value = ExpertOwnerMap> {
    prop::collection::vec((layer(), prop::collection::vec(any::<u8>(), 0..4)), 0..3)
        .prop_map(|layers| ExpertOwnerMap { layers })
}

fn collective_algo() -> impl Strategy<Value = CollectiveAlgo> {
    prop_oneof![Just(CollectiveAlgo::OneShot), Just(CollectiveAlgo::Ring)]
}

fn parallelism_mode() -> impl Strategy<Value = ParallelismMode> {
    prop_oneof![
        Just(ParallelismMode::Local),
        (small(device_id()), tp_split())
            .prop_map(|(devices, split)| ParallelismMode::IntraNodeTp { devices, split }),
        (small(stage_member()), tp_split(), collective_algo()).prop_map(
            |(members, split, collective)| ParallelismMode::CrossNodeTp {
                members,
                split,
                collective
            }
        ),
        (small(stage_member()), expert_owner_map())
            .prop_map(|(members, owners)| ParallelismMode::CrossNodeEp { members, owners }),
    ]
}

fn stage_assignment() -> impl Strategy<Value = StageAssignment> {
    (
        stage_ix(),
        node_id(),
        layer(),
        layer(),
        any::<u8>(),
        parallelism_mode(),
        hop_plan(),
        nanos(),
    )
        .prop_map(
            |(stage, node, start, end, role, mode, next_hop, predicted_step)| StageAssignment {
                stage,
                node,
                layers: start..end,
                role: StageRole::from_bits_truncate(role),
                mode,
                next_hop,
                predicted_step,
            },
        )
}

fn tier_assignment() -> impl Strategy<Value = TierAssignment> {
    let stream =
        (device_id(), link_id(), any::<u32>()).prop_map(|(device, via, min_tokens)| StreamSite {
            device,
            via,
            min_tokens,
        });
    (
        group_id(),
        node_id(),
        tier_id(),
        device_id(),
        prop::option::of(stream),
        small(node_id()),
    )
        .prop_map(
            |(group, owner, home, decode, batched, replicas)| TierAssignment {
                group,
                owner,
                home,
                compute: ComputeSite { decode, batched },
                replicas,
            },
        )
}

fn cache_policy() -> impl Strategy<Value = CachePolicy> {
    prop_oneof![
        (f32s(), any::<u16>(), f32s(), any::<u16>(), any::<bool>()).prop_map(
            |(decay, adapt_every_steps, hysteresis, max_replace, guard)| {
                CachePolicy::DecayedLfu {
                    decay,
                    adapt_every_steps,
                    hysteresis,
                    max_replace,
                    guard,
                }
            }
        ),
        Just(CachePolicy::Lru),
        Just(CachePolicy::Static),
    ]
}

fn expert_cache_budget() -> impl Strategy<Value = ExpertCacheBudget> {
    (
        device_id(),
        tier_id(),
        bytes(),
        prop::option::of(prop::collection::vec(bytes(), 0..4)),
        cache_policy(),
        f32s(),
        prop_oneof![Just(MissSplitMode::Fitted), Just(MissSplitMode::Fixed)],
    )
        .prop_map(
            |(device, tier, bytes, per_layer, policy, copied_share, mode)| ExpertCacheBudget {
                device,
                tier,
                bytes,
                per_layer,
                policy,
                miss_split: MissSplit { copied_share, mode },
            },
        )
}

fn kv_budget() -> impl Strategy<Value = KvBudget> {
    let dtype = prop_oneof![
        Just(KvDType::Native),
        any::<u16>().prop_map(|block| KvDType::Q8 { block }),
        (any::<u16>(), any::<bool>()).prop_map(|(block, rotate)| KvDType::K8V4 { block, rotate }),
    ];
    let spill_compute = prop_oneof![
        any::<u16>().prop_map(|max_rows| SpillCompute::CpuSplit { max_rows }),
        Just(SpillCompute::GpuStream),
    ];
    (
        stage_ix(),
        tier_id(),
        bytes(),
        dtype,
        prop::option::of(tier_id()),
        spill_compute,
    )
        .prop_map(
            |(stage, tier, bytes, dtype, spill, spill_compute)| KvBudget {
                stage,
                tier,
                bytes,
                dtype,
                spill,
                spill_compute,
            },
        )
}

fn prefetch_window() -> impl Strategy<Value = PrefetchWindow> {
    (tier_id(), tier_id(), bytes(), any::<u16>()).prop_map(
        |(from, to, max_inflight, lookahead_layers)| PrefetchWindow {
            from,
            to,
            max_inflight,
            lookahead_layers,
        },
    )
}

fn speculative_config() -> impl Strategy<Value = SpeculativeConfig> {
    let source = prop_oneof![
        any::<u8>().prop_map(|heads| DraftSource::Mtp { heads }),
        group_id().prop_map(|group| DraftSource::DraftHead { group }),
        (blake3(), device_id()).prop_map(|(m, device)| DraftSource::DraftModel {
            model: ModelId(m),
            device
        }),
    ];
    let tree = prop::option::of(
        prop::collection::vec(any::<u16>(), 0..6).prop_map(|parents| TreeShape { parents }),
    );
    (source, any::<u16>(), tree).prop_map(|(source, max_window, tree)| SpeculativeConfig {
        source,
        max_window,
        tree,
        acceptance: Acceptance::Exact,
    })
}

fn batching_limits() -> impl Strategy<Value = BatchingLimits> {
    (
        any::<u32>(),
        any::<u32>(),
        any::<u16>(),
        small(any::<u32>()),
    )
        .prop_map(
            |(max_sequences, prefill_chunk, micro_batch, prefill_ladder)| BatchingLimits {
                max_sequences,
                prefill_chunk,
                micro_batch,
                prefill_ladder,
            },
        )
}

fn stage_term() -> impl Strategy<Value = StageTerm> {
    prop::sample::select(StageTerm::ALL.to_vec())
}

fn stage_cost() -> impl Strategy<Value = StageCost> {
    let purpose = prop::sample::select(vec![
        MemoryPurpose::ContextReserve,
        MemoryPurpose::Scratch,
        MemoryPurpose::DenseWeights,
        MemoryPurpose::Kv,
        MemoryPurpose::KvReserved,
        MemoryPurpose::ExpertCache,
        MemoryPurpose::Staging,
        MemoryPurpose::NetPools,
        MemoryPurpose::History,
        MemoryPurpose::Headroom,
    ]);
    let resident = (tier_id(), bytes(), bytes(), small((purpose, bytes()))).prop_map(
        |(tier, before, after, by_purpose)| ResidentBytes {
            tier,
            before,
            after,
            by_purpose,
        },
    );
    (
        (
            stage_ix(),
            small((stage_term(), nanos())),
            small((stage_term(), nanos())),
        ),
        (
            nanos(),
            small((tier_id(), bytes())),
            small(resident),
            nanos(),
        ),
    )
        .prop_map(
            |((stage, terms, overlapped), (cpu_layer_max, bytes, resident, prefill_per_chunk))| {
                StageCost {
                    stage,
                    terms,
                    overlapped,
                    cpu_layer_max,
                    bytes,
                    resident,
                    prefill_per_chunk,
                }
            },
        )
}

fn plan_ext() -> impl Strategy<Value = PlacementPlanExt> {
    let transition = prop::sample::select(vec![
        TransitionStrategy::MakeBeforeBreak,
        TransitionStrategy::BreakBeforeMake,
    ]);
    let path = prop::sample::select(vec![
        SolvePath::Milp,
        SolvePath::MilpIncumbent,
        SolvePath::Dp,
    ]);
    let solve = (path, f32s(), any::<u64>(), any::<u64>(), nanos()).prop_map(
        |(path, gap, mip_nodes, simplex_iterations, wall)| SolveStats {
            path,
            gap,
            mip_nodes,
            simplex_iterations,
            wall,
        },
    );
    let degraded = prop::option::of(prop::sample::select(vec![
        DegradedReason::DenseOnNvme,
        DegradedReason::MultiStageOnNode,
        DegradedReason::AbsorbedStage,
    ]));
    let status = prop::sample::select(vec![
        CertStatus::Certified,
        CertStatus::Derived,
        CertStatus::Uncertified,
    ]);
    let kind = prop::sample::select(vec![
        CertKind::Op,
        CertKind::Wire,
        CertKind::Family,
        CertKind::Path,
    ]);
    let tier_kind = prop::sample::select(vec![TierKind::HostPinned, TierKind::Nvme]);
    let runtime = (any::<bool>(), small(tier_kind), any::<u32>(), any::<u16>()).prop_map(
        |(prefix_reuse, kv_spill_tiers, max_concurrency, max_window)| CertifiedRuntime {
            prefix_reuse,
            kv_spill_tiers,
            max_concurrency,
            max_window,
        },
    );
    let certification = (
        status,
        runtime,
        prop::collection::vec(
            (kind, blake3()).prop_map(|(kind, id)| CertRef { kind, id }),
            0..3,
        ),
    )
        .prop_map(|(status, runtime, relies_on)| PlanCertification {
            status,
            runtime,
            relies_on,
        });
    (
        (
            transition,
            prop::option::of(blake3()),
            solve,
            degraded,
            bytes(),
        ),
        (
            certification,
            small((node_id(), any::<u16>())),
            prop::collection::vec(group_id(), 0..3),
            small((node_id(), any::<u8>())),
        ),
    )
        .prop_map(
            |(
                (transition, explanation, solve, degraded, move_bytes),
                (certification, rdma_qps, per_site_layout, wan_rate_class),
            )| PlacementPlanExt {
                transition,
                explanation,
                solve,
                degraded,
                move_bytes,
                certification,
                rdma_qps,
                per_site_layout,
                wan_rate_class,
            },
        )
}

fn predicted_cost() -> impl Strategy<Value = PredictedCost> {
    let jitter = (f32s(), nanos()).prop_map(|(stall_rate, t_tok_p99)| Jitter {
        stall_rate,
        t_tok_p99,
    });
    (
        (nanos(), nanos(), f32s(), small((any::<u32>(), nanos()))),
        ((any::<u32>(), nanos()), prop::option::of(nanos())),
        (prop::option::of(jitter), any::<u32>()),
        (
            small((any::<u16>(), nanos())),
            prop::option::of(f32s()),
            prop::collection::vec(stage_cost(), 0..3),
        ),
    )
        .prop_map(
            |(
                (t_tok, t_pass, tokens_per_pass, ttft),
                (ttft_ref, ttft_cap),
                (jitter, context_length),
                (pass_by_window, throughput, stages),
            )| PredictedCost {
                t_tok,
                t_pass,
                tokens_per_pass,
                ttft,
                ttft_ref,
                ttft_cap,
                jitter,
                context_length,
                pass_by_window,
                throughput,
                stages,
            },
        )
}

fn assumption() -> impl Strategy<Value = Assumption> {
    let input = prop_oneof![
        (device_id(), tier_id(), any::<u16>()).prop_map(|(device, tier, block)| {
            AssumedInput::QuantBw {
                device,
                tier,
                scheme: QuantScheme::Mxfp4 { block },
            }
        }),
        layer().prop_map(|layer| AssumedInput::HitCurve { layer }),
        link_id().prop_map(|link| AssumedInput::LinkP99 { link }),
    ];
    let source = prop_oneof![
        Just(MeasureSource::Prior),
        Just(MeasureSource::Startup),
        Just(MeasureSource::Online),
    ];
    (input, source, nanos()).prop_map(|(input, source, sensitivity)| Assumption {
        input,
        source,
        sensitivity,
    })
}

fn placement_plan() -> impl Strategy<Value = PlacementPlan> {
    (
        (
            plan_epoch(),
            any::<u64>(),
            blake3(),
            any::<u64>(),
            blake3(),
            objective(),
            exactness(),
        ),
        (
            prop::collection::vec(stage_assignment(), 0..3),
            prop::collection::vec(tier_assignment(), 0..4),
            prop::collection::vec(expert_cache_budget(), 0..2),
            prop::collection::vec(kv_budget(), 0..3),
            prop::collection::vec(prefetch_window(), 0..2),
            prop::collection::vec((device_id(), bytes()), 0..2),
        ),
        (
            prop::option::of(speculative_config()),
            batching_limits(),
            predicted_cost(),
            prop::collection::vec(assumption(), 0..2),
            plan_ext(),
        ),
    )
        .prop_map(
            |(
                (epoch, instance, model, graph, policy, objective, exactness),
                (stages, tensors, expert_cache, kv, prefetch, scratch),
                (speculative, batching, predicted, assumptions, ext),
            )| PlacementPlan {
                epoch,
                instance: InstanceId(instance),
                model: ModelId(model),
                graph: GraphVersion(graph),
                policy,
                objective,
                exactness,
                stages,
                tensors,
                expert_cache,
                kv,
                prefetch,
                scratch,
                speculative,
                batching,
                predicted,
                assumptions,
                ext,
            },
        )
}

fn delta_item() -> impl Strategy<Value = DeltaItem> {
    prop_oneof![
        (device_id(), expert_cache_budget())
            .prop_map(|(device, budget)| DeltaItem::ExpertCache { device, budget }),
        (stage_ix(), small(kv_budget()))
            .prop_map(|(stage, budgets)| DeltaItem::Kv { stage, budgets }),
        prefetch_window().prop_map(|window| DeltaItem::Prefetch { window }),
        prop::option::of(speculative_config()).prop_map(|config| DeltaItem::Speculative { config }),
        batching_limits().prop_map(|limits| DeltaItem::Batching { limits }),
        predicted_cost().prop_map(|cost| DeltaItem::Predicted { cost }),
        (stage_ix(), node_id(), node_id(), hop_plan()).prop_map(|(stage, prev, next, hop)| {
            DeltaItem::Relink {
                stage,
                prev,
                next,
                hop,
            }
        }),
        small(stage_ix()).prop_map(|stages| DeltaItem::ExtendShard { stages }),
    ]
}

fn plan_delta() -> impl Strategy<Value = PlanDelta> {
    (
        plan_epoch(),
        plan_epoch(),
        prop::collection::vec(delta_item(), 0..4),
    )
        .prop_map(|(from, to, changes)| PlanDelta { from, to, changes })
}

// ---- Properties -------------------------------------------------------------------------

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&back, value, "round trip through {json}");
    // Postcard too: the IPC carries plans in it (RFC-0001 §3.5), and it reads fields by
    // position, so a field skipped on the way out breaks it (RFC-0001 §18.20).
    let bytes = postcard::to_allocvec(value).expect("serialize to postcard");
    let back: T = postcard::from_bytes(&bytes).expect("deserialize from postcard");
    assert_eq!(&back, value, "round trip through postcard");
}

macro_rules! round_trip_tests {
    ($($name:ident: $strategy:expr;)*) => {
        proptest! {
            $(
                #[test]
                fn $name(value in $strategy) {
                    round_trip(&value);
                }
            )*
        }
    };
}

round_trip_tests! {
    rt_instance_id: any::<u64>().prop_map(InstanceId);
    rt_model_id: blake3().prop_map(ModelId);
    rt_layer_ix: layer();
    rt_expert_ix: any::<u16>().prop_map(ExpertIx);
    rt_stage_ix: stage_ix();
    rt_tensor_group_id: group_id();
    rt_plan_epoch: plan_epoch();
    rt_plan_digest: blake3().prop_map(PlanDigest);
    rt_objective: objective();
    rt_exactness_policy: exactness();
    rt_stage_assignment: stage_assignment();
    rt_parallelism_mode: parallelism_mode();
    rt_tp_split: tp_split();
    rt_collective_algo: collective_algo();
    rt_expert_owner_map: expert_owner_map();
    rt_hop_plan: hop_plan();
    rt_relay_via: relay_via();
    rt_tier_assignment: tier_assignment();
    rt_expert_cache_budget: expert_cache_budget();
    rt_kv_budget: kv_budget();
    rt_prefetch_window: prefetch_window();
    rt_speculative_config: speculative_config();
    rt_batching_limits: batching_limits();
    rt_predicted_cost: predicted_cost();
    rt_assumption: assumption();
    rt_stage_cost: stage_cost();
    rt_plan_ext: plan_ext();
    rt_placement_plan: placement_plan();
    rt_delta_item: delta_item();
    rt_plan_delta: plan_delta();
}

/// Decodes `value` after adding an unknown field to every record in its JSON form.
fn decodes_with_future_fields<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let mut json = serde_json::to_value(value).expect("serialize");
    add_future_fields_to_records(&mut json);
    let back: T = serde_json::from_value(json.clone())
        .unwrap_or_else(|e| panic!("decode with unknown fields failed: {e}\n{json}"));
    assert_eq!(&back, value);
}

/// Adds an unknown field to every JSON object in `value`, recursively. Externally tagged enum
/// objects (one key starting with an upper-case letter) are skipped, and so are the
/// `{start, end}` objects of a `Range`, whose serde form is std's and closed.
fn add_future_fields_to_records(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            let is_variant = map.len() == 1
                && map
                    .keys()
                    .next()
                    .is_some_and(|k| k.starts_with(|c: char| c.is_ascii_uppercase()));
            let is_range = map.len() == 2 && map.contains_key("start") && map.contains_key("end");
            for v in map.values_mut() {
                add_future_fields_to_records(v);
            }
            if !is_variant && !is_range {
                map.insert(
                    "field_from_a_future_build".to_owned(),
                    serde_json::json!({ "nested": [1, 2, 3], "flag": true }),
                );
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(add_future_fields_to_records),
        _ => {}
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn plan_decodes_with_future_fields(value in placement_plan()) {
        decodes_with_future_fields(&value);
    }

    #[test]
    fn delta_decodes_with_future_fields(value in plan_delta()) {
        decodes_with_future_fields(&value);
    }

    #[test]
    fn tp_split_decodes_with_future_fields(value in tp_split()) {
        decodes_with_future_fields(&value);
    }

    #[test]
    fn expert_owner_map_decodes_with_future_fields(value in expert_owner_map()) {
        decodes_with_future_fields(&value);
    }
}

#[test]
fn kv_budget_without_spill_compute_defaults_to_gpu_stream() {
    let budget = KvBudget {
        stage: StageIx(1),
        tier: TierId {
            node: NodeId([7; 32]),
            ordinal: 2,
        },
        bytes: Bytes(1 << 30),
        dtype: KvDType::Native,
        spill: None,
        spill_compute: SpillCompute::CpuSplit { max_rows: 5 },
    };
    let mut json = serde_json::to_value(budget).expect("serialize");
    json.as_object_mut()
        .expect("a record")
        .remove("spill_compute");
    let back: KvBudget = serde_json::from_value(json).expect("decode");
    assert_eq!(back.spill_compute, SpillCompute::GpuStream);
    assert_eq!(
        KvBudget {
            spill_compute: SpillCompute::GpuStream,
            ..budget
        },
        back
    );
}

fn sample_link(node: u8, path: u8) -> LinkId {
    LinkId {
        src: Endpoint::Device(DeviceId {
            node: NodeId([node; 32]),
            ordinal: 0,
        }),
        dst: Endpoint::Device(DeviceId {
            node: NodeId([node.wrapping_add(1); 32]),
            ordinal: 0,
        }),
        kind: LinkKind::Network(NetMedium::Rdma(RdmaKind::Thunderbolt5)),
        path,
    }
}

#[test]
fn hop_with_the_rfc_0032_fields_round_trips() {
    let hop = HopPlan {
        link: sample_link(1, 0),
        transport: TransportKind::Quic,
        wire: WireDType::F32,
        predicted: Nanos(150_000),
        prefill_transport: Some((
            sample_link(1, 1),
            TransportKind::Rdma(RdmaKind::Thunderbolt5),
        )),
        relay: Some(RelayVia {
            node: NodeId([9; 32]),
            first: sample_link(3, 0),
            second: sample_link(9, 0),
            transport: TransportKind::Rdma(RdmaKind::Thunderbolt5),
        }),
        predicted_p99: Nanos(275_000),
    };
    round_trip(&hop);
}

#[test]
fn hop_without_the_rfc_0032_fields_decodes_with_defaults() {
    let hop = HopPlan {
        link: sample_link(1, 0),
        transport: TransportKind::Tcp,
        wire: WireDType::F32,
        predicted: Nanos(150_000),
        prefill_transport: Some((sample_link(1, 1), TransportKind::Quic)),
        relay: Some(RelayVia {
            node: NodeId([9; 32]),
            first: sample_link(3, 0),
            second: sample_link(9, 0),
            transport: TransportKind::Quic,
        }),
        predicted_p99: Nanos(275_000),
    };
    let mut json = serde_json::to_value(hop).expect("serialize");
    let record = json.as_object_mut().expect("a record");
    for field in ["prefill_transport", "relay", "predicted_p99"] {
        record.remove(field).expect("the field was written");
    }
    let back: HopPlan = serde_json::from_value(json).expect("decode");
    assert_eq!(back.prefill_transport, None);
    assert_eq!(back.relay, None);
    assert_eq!(back.predicted_p99, Nanos(0));
    assert_eq!(
        HopPlan {
            prefill_transport: None,
            relay: None,
            predicted_p99: Nanos(0),
            ..hop
        },
        back
    );
}

#[test]
fn relay_via_carries_the_node_both_links_and_the_transport() {
    let relay = RelayVia {
        node: NodeId([2; 32]),
        first: sample_link(3, 0),
        second: sample_link(2, 0),
        transport: TransportKind::Rdma(RdmaKind::Thunderbolt5),
    };
    let json = serde_json::to_value(relay).expect("serialize");
    let mut fields: Vec<&str> = json
        .as_object()
        .expect("a record")
        .keys()
        .map(String::as_str)
        .collect();
    fields.sort_unstable();
    assert_eq!(fields, ["first", "node", "second", "transport"]);
    assert_eq!(relay.node, NodeId([2; 32]));
    assert_eq!(relay.first, sample_link(3, 0));
    assert_eq!(relay.second, sample_link(2, 0));
    assert_eq!(relay.transport, TransportKind::Rdma(RdmaKind::Thunderbolt5));
    assert_ne!(relay.first, relay.second);
}

#[test]
fn plan_epochs_order_by_term_then_seq() {
    let e = |term, seq| PlanEpoch { term, seq };
    assert!(e(1, 9) < e(2, 0));
    assert!(e(2, 0) < e(2, 1));
    assert_eq!(e(3, 3).cmp(&e(3, 3)), std::cmp::Ordering::Equal);
}

// ---- A 4-member island (RFC-0032 §5.1, DIST-E13-T13) ---------------------------------------
//
// The test inputs below are not measurements. Four members have `W_eff` in the ratio 3 : 2 : 2 : 1,
// and each split gives a member its share at the legal granularity: whole KV-head groups for
// attention, whole 128-column tiles for FFN and expert columns. Validating a split against a
// family's granularities is DIST-E13-T01's.

/// Shares of `total` units in the ratio `weights`, as whole units; the test picks totals the
/// weights divide.
fn weighted(total: u32, weights: [u32; 4]) -> [u32; 4] {
    let sum: u32 = weights.iter().sum();
    assert_eq!(total % sum, 0, "the test's totals divide by the weights");
    weights.map(|weight| total / sum * weight)
}

const WEIGHTS: [u32; 4] = [3, 2, 2, 1];

/// 8 KV-head groups, 14,336 FFN columns (112 tiles of 128) and 2,048 columns per expert (16 tiles),
/// split 3 : 2 : 2 : 1.
fn uneven_split() -> TpSplit {
    let attn = weighted(8, WEIGHTS).map(|groups| groups as u16);
    TpSplit {
        attn: SmallVec::from_slice(&attn),
        ffn: SmallVec::from_slice(&weighted(14_336 / 128, WEIGHTS)),
        experts: Some(SmallVec::from_slice(&weighted(2_048 / 128, WEIGHTS))),
    }
}

/// Three MoE layers of 16 experts each, owned 6 : 4 : 4 : 2 by the four members, laid out
/// differently in each layer.
fn four_member_owners() -> ExpertOwnerMap {
    let counts = weighted(16, WEIGHTS);
    let blocked: Vec<u8> = (0..4u8)
        .flat_map(|member| std::iter::repeat_n(member, counts[member as usize] as usize))
        .collect();
    let mut striped = Vec::with_capacity(16);
    let mut left = counts;
    while striped.len() < 16 {
        for member in 0..4u8 {
            if left[member as usize] > 0 {
                left[member as usize] -= 1;
                striped.push(member);
            }
        }
    }
    let reversed: Vec<u8> = blocked.iter().rev().copied().collect();
    ExpertOwnerMap {
        layers: vec![
            (LayerIx(1), blocked),
            (LayerIx(3), striped),
            (LayerIx(5), reversed),
        ],
    }
}

fn four_members() -> SmallVec<[StageMember; 4]> {
    (1u8..=4)
        .map(|node| StageMember {
            node: NodeId([node; 32]),
            devices: SmallVec::from_slice(&[DeviceId {
                node: NodeId([node; 32]),
                ordinal: 0,
            }]),
            link: sample_link(1, node),
        })
        .collect()
}

#[test]
fn an_uneven_weighted_split_round_trips_unchanged() {
    let split = uneven_split();
    assert_eq!(split.attn.as_slice(), [3, 2, 2, 1]);
    assert_eq!(split.ffn.as_slice(), [42, 28, 28, 14]);
    assert_eq!(split.experts.as_deref(), Some([6, 4, 4, 2].as_slice()));
    // Every unit is given to exactly one member.
    assert_eq!(split.attn.iter().map(|g| u32::from(*g)).sum::<u32>(), 8);
    assert_eq!(split.ffn.iter().sum::<u32>(), 112);
    round_trip(&split);
    round_trip(&ParallelismMode::CrossNodeTp {
        members: four_members(),
        split,
        collective: CollectiveAlgo::OneShot,
    });
}

#[test]
fn an_owner_map_for_a_four_member_island_round_trips_unchanged() {
    let owners = four_member_owners();
    for (layer, experts) in &owners.layers {
        let mut per_member = [0u32; 4];
        for owner in experts {
            per_member[*owner as usize] += 1;
        }
        assert_eq!(per_member, [6, 4, 4, 2], "layer {layer:?}");
    }
    round_trip(&owners);
    round_trip(&ParallelismMode::CrossNodeEp {
        members: four_members(),
        owners,
    });
}

#[test]
fn both_collective_algorithms_round_trip() {
    for algo in [CollectiveAlgo::OneShot, CollectiveAlgo::Ring] {
        round_trip(&algo);
    }
}

/// A plan serialized before RFC-0011 §2's additions, with no `stages` and no `ext`, loads in a
/// self-describing format, with each default the conservative one (PLC-E4-T01).
#[test]
fn a_plan_from_before_the_additions_loads_with_conservative_defaults() {
    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let plan = placement_plan()
        .new_tree(&mut runner)
        .expect("a plan")
        .current();
    let mut json = serde_json::to_value(&plan).expect("serialize");
    let object = json.as_object_mut().expect("a plan is a JSON object");
    object.remove("ext");
    object["predicted"]
        .as_object_mut()
        .expect("an object")
        .remove("stages");
    let back: PlacementPlan = serde_json::from_value(json).expect("an older plan loads");
    assert!(back.predicted.stages.is_empty());
    assert_eq!(back.ext, PlacementPlanExt::default());
    assert_eq!(back.ext.transition, TransitionStrategy::BreakBeforeMake);
    assert_eq!(back.ext.certification.status, CertStatus::Uncertified);
    assert_eq!(back.ext.certification.runtime, CertifiedRuntime::default());
    assert!(!back.ext.certification.runtime.prefix_reuse);
    assert_eq!(back.ext.solve.path, SolvePath::Dp);
    assert!(back.ext.degraded.is_none());
}

/// `StageTerm`'s codes are 0 to 17 in `ALL`'s order, and each code reads back as its term.
#[test]
fn stage_terms_are_numbered_in_order() {
    for (code, term) in StageTerm::ALL.iter().enumerate() {
        assert_eq!(usize::from(term.code()), code);
        assert_eq!(StageTerm::from_code(term.code()), Some(*term));
    }
    assert_eq!(StageTerm::from_code(18), None);
}

/// A stage's step is the sum of its critical-path terms; overlapped work is not in it.
#[test]
fn a_stage_step_sums_its_terms_and_not_the_overlapped_ones() {
    let cost = StageCost {
        stage: StageIx(0),
        terms: [
            (StageTerm::DenseVram, Nanos(30)),
            (StageTerm::HopWait, Nanos(12)),
        ]
        .into_iter()
        .collect(),
        overlapped: [(StageTerm::ExpertHitCpu, Nanos(1_000))]
            .into_iter()
            .collect(),
        cpu_layer_max: Nanos(0),
        bytes: SmallVec::new(),
        resident: SmallVec::new(),
        prefill_per_chunk: Nanos(0),
    };
    assert_eq!(cost.step(), Nanos(42));
}
