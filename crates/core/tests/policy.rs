// SPDX-License-Identifier: MIT OR Apache-2.0

//! The policy objects (PROD-E1-T04; RFC-0001 §11.3, RFC-0054 §4).
//!
//! - Every type round-trips through a serde format unchanged. JSON stands in for any
//!   self-describing format.
//! - `ApproxKnobs` keeps RFC-0001 §11.3's bit positions and RFC-0054 §4.2's scopes.
//! - `InstancePolicy::canonical_bytes`, the bytes its digest is computed over, is pinned to
//!   golden values. CI runs this file on x86_64 Linux and Windows and on arm64 macOS, so one set
//!   of values passing everywhere is the cross-target check.
//! - Records written before RFC-0054's fields decode with those fields at their defaults.
//! - `ApproxParams` and `Objective::Throughput` skip an unset field only in the canonical
//!   encoding, so their canonical bytes are the ones they had when they skipped in every format,
//!   and postcard, which the IPC uses and which cannot skip a field, round-trips them
//!   (RFC-0001 §18.20).

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::Range;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use syntropy_core::canon;
use syntropy_core::plan::{
    Acceptance, DraftSource, KvDType, SpeculativeConfig, StageRole, TreeShape, WireDType,
};
use syntropy_core::policy::*;
use syntropy_core::resource::device::DeviceKind;
use syntropy_core::resource::tier::TierKind;
use syntropy_core::*;

// ---- Strategies -------------------------------------------------------------------------

fn text() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop::sample::select(vec!['a', 'Z', '0', '.', '*', ' ', 'é', '"', '\\']),
        0..8,
    )
    .prop_map(|c| c.into_iter().collect())
}

fn f32s() -> impl Strategy<Value = f32> + Clone {
    -1.0e6f32..1.0e6
}

fn opt<T: Debug>(s: impl Strategy<Value = T>) -> impl Strategy<Value = Option<T>> {
    prop::option::of(s)
}

fn objective() -> impl Strategy<Value = Objective> {
    prop_oneof![
        Just(Objective::Latency),
        (f32s(), opt(any::<u16>()), opt(any::<u64>().prop_map(Nanos))).prop_map(
            |(per_sequence_floor_tps, target_concurrency, ttft_ceiling)| Objective::Throughput {
                per_sequence_floor_tps,
                target_concurrency,
                ttft_ceiling,
            }
        ),
    ]
}

fn knobs() -> impl Strategy<Value = ApproxKnobs> {
    any::<u32>().prop_map(ApproxKnobs::from_bits_truncate)
}

fn quant_scheme() -> impl Strategy<Value = QuantScheme> {
    prop_oneof![
        Just(QuantScheme::Plain(DType::F8E4M3)),
        Just(QuantScheme::Ggml(GgmlType::Q8_0)),
        Just(QuantScheme::Mxfp4 { block: 32 }),
    ]
}

fn kv_dtype() -> impl Strategy<Value = KvDType> {
    prop_oneof![
        Just(KvDType::Native),
        any::<u16>().prop_map(|block| KvDType::Q8 { block }),
        (any::<u16>(), any::<bool>()).prop_map(|(block, rotate)| KvDType::K8V4 { block, rotate }),
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

fn approx_params() -> impl Strategy<Value = ApproxParams> {
    (opt(quant_scheme()), opt(kv_dtype()), opt(wire_dtype())).prop_map(
        |(miss_low_precision, kv_lossy, wire_compress)| ApproxParams {
            miss_low_precision,
            kv_lossy,
            wire_compress,
        },
    )
}

fn exactness() -> impl Strategy<Value = ExactnessPolicy> {
    prop_oneof![
        Just(ExactnessPolicy::Exact),
        (knobs(), approx_params())
            .prop_map(|(knobs, params)| ExactnessPolicy::Approximate { knobs, params }),
    ]
}

fn fidelity() -> impl Strategy<Value = Fidelity> {
    prop_oneof![Just(Fidelity::Exact), Just(Fidelity::Approximate)]
}

fn fraction() -> impl Strategy<Value = Fraction> {
    (0.0f32..=1.0).prop_map(|v| Fraction::new(v).expect("in range"))
}

fn reserve() -> impl Strategy<Value = Reserve> {
    (any::<u64>(), 0.0f32..=1.0, opt(any::<u64>())).prop_map(|(min, fraction, max)| Reserve {
        min: Bytes(min),
        fraction,
        max: max.map(Bytes),
    })
}

fn node_selector() -> impl Strategy<Value = NodeSelector> {
    prop_oneof![
        Just(NodeSelector::Any),
        Just(NodeSelector::This),
        text().prop_map(NodeSelector::Name),
        text().prop_map(NodeSelector::IdPrefix),
        (text(), text()).prop_map(|(key, value)| NodeSelector::Label { key, value }),
    ]
}

fn device_api() -> impl Strategy<Value = DeviceApi> {
    prop_oneof![
        Just(DeviceApi::Cuda),
        Just(DeviceApi::Hip),
        Just(DeviceApi::Metal),
        Just(DeviceApi::Vulkan),
    ]
}

fn device_kind() -> impl Strategy<Value = DeviceKind> {
    prop_oneof![
        Just(DeviceKind::DiscreteGpu),
        Just(DeviceKind::IntegratedGpu),
        Just(DeviceKind::Cpu),
        Just(DeviceKind::Npu),
    ]
}

fn device_selector() -> impl Strategy<Value = DeviceSelector> {
    prop_oneof![
        Just(DeviceSelector::Any),
        device_kind().prop_map(DeviceSelector::Kind),
        (opt(text()), device_api(), any::<u16>())
            .prop_map(|(node, api, index)| DeviceSelector::Gpu { node, api, index }),
        (opt(text()), any::<u16>()).prop_map(|(node, numa)| DeviceSelector::Cpu { node, numa }),
    ]
}

fn tier_kind() -> impl Strategy<Value = TierKind> {
    prop_oneof![
        Just(TierKind::Vram),
        Just(TierKind::Unified),
        Just(TierKind::HostPinned),
        Just(TierKind::HostPageable),
        Just(TierKind::Nvme),
    ]
}

fn tier_selector() -> impl Strategy<Value = TierSelector> {
    (opt(text()), tier_kind(), opt(any::<u16>())).prop_map(|(node, kind, index)| TierSelector {
        node,
        kind,
        index,
    })
}

fn layer_range() -> impl Strategy<Value = LayerRange> {
    (opt(any::<u16>()), opt(any::<u16>())).prop_map(|(start, end)| LayerRange {
        start: start.map(LayerIx),
        end: end.map(LayerIx),
    })
}

fn expert_range() -> impl Strategy<Value = Range<ExpertIx>> {
    (any::<u16>(), any::<u16>()).prop_map(|(a, b)| ExpertIx(a)..ExpertIx(b))
}

fn key_path() -> impl Strategy<Value = KeyPath> {
    text().prop_map(KeyPath)
}

fn config_value() -> impl Strategy<Value = ConfigValue> {
    let leaf = prop_oneof![
        any::<bool>().prop_map(ConfigValue::Bool),
        any::<i64>().prop_map(ConfigValue::Integer),
        (-1.0e12f64..1.0e12).prop_map(ConfigValue::Float),
        text().prop_map(ConfigValue::String),
        text().prop_map(ConfigValue::Datetime),
    ];
    leaf.prop_recursive(3, 16, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(ConfigValue::Array),
            prop::collection::btree_map(text(), inner, 0..4).prop_map(ConfigValue::Table),
        ]
    })
}

fn memory_yield() -> impl Strategy<Value = MemoryYield> {
    prop_oneof![
        Just(MemoryYield::Never),
        Just(MemoryYield::UnderPressure),
        Just(MemoryYield::Aggressive),
    ]
}

fn power_policy() -> impl Strategy<Value = PowerPolicy> {
    prop_oneof![
        Just(PowerPolicy::Performance),
        Just(PowerPolicy::Balanced),
        Just(PowerPolicy::Battery),
    ]
}

fn network_scope() -> impl Strategy<Value = NetworkScope> {
    prop_oneof![Just(NetworkScope::Lan), Just(NetworkScope::LanAndInvites)]
}

fn usage_mode() -> impl Strategy<Value = UsageMode> {
    prop_oneof![Just(UsageMode::Personal), Just(UsageMode::Dedicated)]
}

fn device_filter() -> impl Strategy<Value = DeviceFilter> {
    (
        prop::collection::vec(device_selector(), 0..3),
        prop::collection::vec(device_selector(), 0..3),
    )
        .prop_map(|(allow, deny)| DeviceFilter { allow, deny })
}

fn resource_policy() -> impl Strategy<Value = ResourcePolicy> {
    (
        (
            memory_yield(),
            opt(0.0f32..=1.0),
            power_policy(),
            network_scope(),
        ),
        (usage_mode(), reserve(), reserve(), any::<u16>()),
        (any::<u8>(), device_filter(), any::<bool>()),
    )
        .prop_map(
            |(
                (memory_yield, max_pinned_fraction, power, network_scope),
                (mode, ram_headroom, vram_headroom, reserve_cores),
                (min_battery_percent, gpus, macos_raise_wired_limit),
            )| ResourcePolicy {
                memory_yield,
                max_pinned_fraction,
                power,
                network_scope,
                mode,
                ram_headroom,
                vram_headroom,
                reserve_cores,
                min_battery_percent,
                gpus,
                macos_raise_wired_limit,
            },
        )
}

fn resource_overlay() -> impl Strategy<Value = ResourceOverlay> {
    (
        (opt(memory_yield()), opt(0.0f32..=1.0), opt(power_policy())),
        (opt(network_scope()), opt(reserve()), opt(reserve())),
        (opt(any::<u16>()), opt(device_filter())),
    )
        .prop_map(
            |(
                (memory_yield, max_pinned_fraction, power),
                (network_scope, ram_headroom, vram_headroom),
                (reserve_cores, gpus),
            )| ResourceOverlay {
                memory_yield,
                max_pinned_fraction,
                power,
                network_scope,
                ram_headroom,
                vram_headroom,
                reserve_cores,
                gpus,
            },
        )
}

fn speculative_config() -> impl Strategy<Value = SpeculativeConfig> {
    let source = prop_oneof![
        any::<u8>().prop_map(|heads| DraftSource::Mtp { heads }),
        any::<u32>().prop_map(|g| DraftSource::DraftHead {
            group: TensorGroupId(g)
        }),
    ];
    let tree =
        opt(prop::collection::vec(any::<u16>(), 0..4).prop_map(|parents| TreeShape { parents }));
    (source, any::<u16>(), tree).prop_map(|(source, max_window, tree)| SpeculativeConfig {
        source,
        max_window,
        tree,
        acceptance: Acceptance::Exact,
    })
}

fn speculative_mode() -> impl Strategy<Value = SpeculativeMode> {
    prop_oneof![
        Just(SpeculativeMode::Auto),
        Just(SpeculativeMode::Off),
        speculative_config().prop_map(SpeculativeMode::Force),
    ]
}

fn context_policy() -> impl Strategy<Value = ContextPolicy> {
    (opt(any::<u32>()), opt(any::<u16>())).prop_map(|(context_length, max_sequences)| {
        ContextPolicy {
            context_length,
            max_sequences,
        }
    })
}

fn node_filter() -> impl Strategy<Value = NodeFilter> {
    (
        prop::collection::vec(node_selector(), 0..3),
        prop::collection::vec(node_selector(), 0..3),
    )
        .prop_map(|(include, exclude)| NodeFilter { include, exclude })
}

fn keep_alive() -> impl Strategy<Value = KeepAlive> {
    prop_oneof![
        Just(KeepAlive::Auto),
        any::<u64>().prop_map(|n| KeepAlive::For(Nanos(n))),
        Just(KeepAlive::Never),
    ]
}

fn dense_nvme() -> impl Strategy<Value = DenseNvmePolicy> {
    prop_oneof![
        Just(DenseNvmePolicy::Never),
        Just(DenseNvmePolicy::IfOnlyOption),
        Just(DenseNvmePolicy::Allow),
    ]
}

fn class_sel() -> impl Strategy<Value = ClassSel> {
    prop::sample::select(vec![
        ClassSel::Dense,
        ClassSel::Expert,
        ClassSel::Lookup,
        ClassSel::Head,
        ClassSel::Router,
        ClassSel::Attention,
        ClassSel::Ffn,
        ClassSel::All,
    ])
}

fn group_selector() -> impl Strategy<Value = GroupSelector> {
    (class_sel(), opt(layer_range()), opt(expert_range())).prop_map(|(class, layers, experts)| {
        GroupSelector {
            class,
            layers,
            experts,
        }
    })
}

fn stage_role() -> impl Strategy<Value = StageRole> {
    any::<u8>().prop_map(StageRole::from_bits_truncate)
}

fn exclude_target() -> impl Strategy<Value = ExcludeTarget> {
    prop_oneof![
        node_selector().prop_map(ExcludeTarget::Node),
        device_selector().prop_map(ExcludeTarget::Device),
    ]
}

fn pin_kind() -> impl Strategy<Value = PinKind> {
    prop_oneof![
        (layer_range(), node_selector())
            .prop_map(|(layers, node)| PinKind::Layers { layers, node }),
        (group_selector(), tier_selector())
            .prop_map(|(groups, tier)| PinKind::Tier { groups, tier }),
        (group_selector(), device_selector())
            .prop_map(|(groups, device)| PinKind::Compute { groups, device }),
        (stage_role(), node_selector()).prop_map(|(role, node)| PinKind::Role { role, node }),
        exclude_target().prop_map(|target| PinKind::Exclude { target }),
    ]
}

fn pin_strength() -> impl Strategy<Value = PinStrength> {
    prop_oneof![
        Just(PinStrength::Require),
        f32s().prop_map(|weight| PinStrength::Prefer { weight }),
    ]
}

fn placement_pin() -> impl Strategy<Value = PlacementPin> {
    (pin_kind(), pin_strength()).prop_map(|(kind, strength)| PlacementPin { kind, strength })
}

fn tensor_override() -> impl Strategy<Value = TensorOverride> {
    (
        text(),
        opt(layer_range()),
        opt(expert_range()),
        tier_selector(),
        opt(device_selector()),
    )
        .prop_map(|(pattern, layers, experts, tier, compute)| TensorOverride {
            pattern,
            layers,
            experts,
            tier,
            compute,
        })
}

fn instance_policy() -> impl Strategy<Value = InstancePolicy> {
    (
        (objective(), exactness(), fidelity(), resource_overlay()),
        (
            prop::collection::vec(placement_pin(), 0..3),
            prop::collection::vec(tensor_override(), 0..3),
            speculative_mode(),
            prop::collection::btree_map(key_path(), config_value(), 0..3),
        ),
        (context_policy(), node_filter(), keep_alive()),
        (any::<bool>(), any::<bool>(), dense_nvme()),
    )
        .prop_map(
            |(
                (objective, exactness, default_fidelity, resources),
                (pins, tensor_overrides, speculative, domain),
                (context, nodes, keep_alive),
                (preload, deterministic, dense_on_nvme),
            )| InstancePolicy {
                objective,
                exactness,
                default_fidelity,
                resources,
                pins,
                tensor_overrides,
                speculative,
                domain,
                context,
                nodes,
                keep_alive,
                preload,
                deterministic,
                dense_on_nvme,
            },
        )
}

// ---- Round trips ------------------------------------------------------------------------

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&back, value, "round trip through {json}");
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
    objective_round_trips: objective();
    exactness_round_trips: exactness();
    approx_knobs_round_trip: knobs();
    approx_params_round_trip: approx_params();
    fidelity_round_trips: fidelity();
    fraction_round_trips: fraction();
    reserve_round_trips: reserve();
    node_selector_round_trips: node_selector();
    device_api_round_trips: device_api();
    device_selector_round_trips: device_selector();
    tier_selector_round_trips: tier_selector();
    layer_range_round_trips: layer_range();
    key_path_round_trips: key_path();
    config_value_round_trips: config_value();
    memory_yield_round_trips: memory_yield();
    power_policy_round_trips: power_policy();
    network_scope_round_trips: network_scope();
    usage_mode_round_trips: usage_mode();
    device_filter_round_trips: device_filter();
    resource_policy_round_trips: resource_policy();
    resource_overlay_round_trips: resource_overlay();
    speculative_mode_round_trips: speculative_mode();
    context_policy_round_trips: context_policy();
    node_filter_round_trips: node_filter();
    keep_alive_round_trips: keep_alive();
    dense_nvme_round_trips: dense_nvme();
    class_sel_round_trips: class_sel();
    group_selector_round_trips: group_selector();
    exclude_target_round_trips: exclude_target();
    pin_kind_round_trips: pin_kind();
    pin_strength_round_trips: pin_strength();
    placement_pin_round_trips: placement_pin();
    tensor_override_round_trips: tensor_override();
    instance_policy_round_trips: instance_policy();
}

proptest! {
    /// The bytes are a function of the value: a decoded copy encodes the same.
    #[test]
    fn canonical_bytes_survive_a_round_trip(policy in instance_policy()) {
        let json = serde_json::to_string(&policy).expect("serialize");
        let back: InstancePolicy = serde_json::from_str(&json).expect("deserialize");
        prop_assert_eq!(back.canonical_bytes(), policy.canonical_bytes());
    }

    /// The order the `domain` map was filled in never reaches the canonical bytes.
    #[test]
    fn domain_insertion_order_does_not_matter(
        entries in prop::collection::btree_map(key_path(), config_value(), 0..6)
    ) {
        let mut forward = InstancePolicy::default();
        let mut backward = InstancePolicy::default();
        for (k, v) in entries.iter() {
            forward.domain.insert(k.clone(), v.clone());
        }
        for (k, v) in entries.iter().rev() {
            backward.domain.insert(k.clone(), v.clone());
        }
        prop_assert_eq!(forward.canonical_bytes(), backward.canonical_bytes());
    }
}

// ---- ApproxParams and Objective in the canonical encoding and in postcard -----------------

/// `ApproxParams` with the attributes it had before `canon::absent_if_none`: an unset field
/// skipped in every format. The canonical encoding writes a struct's field names, not its type
/// name, so the two must encode alike.
#[derive(Serialize)]
struct ApproxParamsBefore {
    #[serde(skip_serializing_if = "Option::is_none")]
    miss_low_precision: Option<QuantScheme>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_lossy: Option<KvDType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    wire_compress: Option<WireDType>,
}

/// `ExactnessPolicy` over [`ApproxParamsBefore`].
#[derive(Serialize)]
enum ExactnessPolicyBefore {
    #[allow(dead_code, reason = "the variant is only written")]
    Exact,
    Approximate {
        knobs: ApproxKnobs,
        params: ApproxParamsBefore,
    },
}

/// `Objective` with the attributes it had before `canon::absent_if_none`.
#[derive(Serialize)]
enum ObjectiveBefore {
    Latency,
    Throughput {
        per_sequence_floor_tps: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        target_concurrency: Option<u16>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ttft_ceiling: Option<Nanos>,
    },
}

fn objective_before(objective: &Objective) -> ObjectiveBefore {
    match *objective {
        Objective::Latency => ObjectiveBefore::Latency,
        Objective::Throughput {
            per_sequence_floor_tps,
            target_concurrency,
            ttft_ceiling,
        } => ObjectiveBefore::Throughput {
            per_sequence_floor_tps,
            target_concurrency,
            ttft_ceiling,
        },
        _ => unreachable!("a variant this test does not mirror"),
    }
}

fn before(params: &ApproxParams) -> ApproxParamsBefore {
    ApproxParamsBefore {
        miss_low_precision: params.miss_low_precision,
        kv_lossy: params.kv_lossy,
        wire_compress: params.wire_compress,
    }
}

fn postcard_round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let bytes = postcard::to_allocvec(value).expect("serialize");
    let back: T = postcard::from_bytes(&bytes).expect("deserialize");
    assert_eq!(&back, value);
}

proptest! {
    /// The canonical bytes, and so every policy digest, are what they were with
    /// `Option::is_none`, for every combination of set and unset parameters.
    #[test]
    fn approx_params_canonical_bytes_are_unchanged(params in approx_params(), bits in any::<u32>()) {
        prop_assert_eq!(
            canon::to_bytes(&params).unwrap(),
            canon::to_bytes(&before(&params)).unwrap()
        );
        let knobs = ApproxKnobs::from_bits_truncate(bits);
        prop_assert_eq!(
            canon::to_bytes(&ExactnessPolicy::Approximate { knobs, params }).unwrap(),
            canon::to_bytes(&ExactnessPolicyBefore::Approximate { knobs, params: before(&params) })
                .unwrap()
        );
    }

    /// The same for `Objective`.
    #[test]
    fn objective_canonical_bytes_are_unchanged(objective in objective()) {
        prop_assert_eq!(
            canon::to_bytes(&objective).unwrap(),
            canon::to_bytes(&objective_before(&objective)).unwrap()
        );
    }

    /// Postcard round-trips every parameter combination, and every objective and exactness.
    #[test]
    fn approx_params_and_objective_round_trip_through_postcard(
        params in approx_params(),
        exactness in exactness(),
        objective in objective(),
    ) {
        postcard_round_trip(&params);
        postcard_round_trip(&exactness);
        postcard_round_trip(&objective);
    }
}

#[test]
fn unset_approx_params_are_null_in_json() {
    let params = ApproxParams {
        kv_lossy: Some(KvDType::Q8 { block: 32 }),
        ..ApproxParams::default()
    };
    let json = serde_json::to_value(params).unwrap();
    assert_eq!(json["miss_low_precision"], serde_json::Value::Null);
    assert_eq!(json["wire_compress"], serde_json::Value::Null);
    // A record written while the fields were skipped still decodes.
    let old: ApproxParams = serde_json::from_str(r#"{"kv_lossy":{"Q8":{"block":32}}}"#).unwrap();
    assert_eq!(old, params);
}

// ---- ApproxKnobs ------------------------------------------------------------------------

#[test]
fn approx_knob_bits_are_rfc_0001_s11_3() {
    let expected = [
        ("EXPERT_SKIP", 0),
        ("EXPERT_DEFER", 1),
        ("MISS_LOW_PRECISION", 2),
        ("TOPK_REDUCE", 3),
        ("KV_LOSSY", 4),
        ("WIRE_COMPRESS", 5),
        ("APPROX_ATTENTION", 6),
        ("SPEC_LENIENT", 7),
    ];
    let actual: Vec<(&str, u32)> = ApproxKnobs::all()
        .iter_names()
        .map(|(name, knob)| (name, knob.bits().trailing_zeros()))
        .collect();
    assert_eq!(actual, expected);
    for (name, knob) in ApproxKnobs::all().iter_names() {
        assert_eq!(knob.bits().count_ones(), 1, "{name} is one bit");
    }
    assert_eq!(ApproxKnobs::EXPERT_SKIP.bits(), 1 << 0);
    assert_eq!(ApproxKnobs::EXPERT_DEFER.bits(), 1 << 1);
    assert_eq!(ApproxKnobs::MISS_LOW_PRECISION.bits(), 1 << 2);
    assert_eq!(ApproxKnobs::TOPK_REDUCE.bits(), 1 << 3);
    assert_eq!(ApproxKnobs::KV_LOSSY.bits(), 1 << 4);
    assert_eq!(ApproxKnobs::WIRE_COMPRESS.bits(), 1 << 5);
    assert_eq!(ApproxKnobs::APPROX_ATTENTION.bits(), 1 << 6);
    assert_eq!(ApproxKnobs::SPEC_LENIENT.bits(), 1 << 7);
}

#[test]
fn knob_scopes_are_rfc_0054_s4_2() {
    assert_eq!(
        ApproxKnobs::PER_SEQUENCE,
        ApproxKnobs::EXPERT_SKIP
            | ApproxKnobs::EXPERT_DEFER
            | ApproxKnobs::TOPK_REDUCE
            | ApproxKnobs::SPEC_LENIENT
    );
    assert_eq!(
        ApproxKnobs::PER_INSTANCE,
        ApproxKnobs::MISS_LOW_PRECISION
            | ApproxKnobs::KV_LOSSY
            | ApproxKnobs::WIRE_COMPRESS
            | ApproxKnobs::APPROX_ATTENTION
    );
    // Every knob has exactly one scope.
    assert!(
        ApproxKnobs::PER_SEQUENCE
            .intersection(ApproxKnobs::PER_INSTANCE)
            .is_empty()
    );
    assert_eq!(
        ApproxKnobs::PER_SEQUENCE | ApproxKnobs::PER_INSTANCE,
        ApproxKnobs::all()
    );
}

#[test]
fn approx_flags_is_approx_knobs() {
    let flags: ApproxFlags = ApproxKnobs::KV_LOSSY;
    assert_eq!(flags.bits(), 1 << 4);
}

// ---- Digests ----------------------------------------------------------------------------

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A policy that sets a value in every field, the way a `[policy."…"]` table can.
fn full_policy() -> InstancePolicy {
    InstancePolicy {
        objective: Objective::Throughput {
            per_sequence_floor_tps: 5.0,
            target_concurrency: Some(8),
            ttft_ceiling: Some(Nanos(2_000_000_000)),
        },
        exactness: ExactnessPolicy::Approximate {
            knobs: ApproxKnobs::TOPK_REDUCE | ApproxKnobs::KV_LOSSY,
            params: ApproxParams {
                kv_lossy: Some(KvDType::Q8 { block: 32 }),
                ..ApproxParams::default()
            },
        },
        default_fidelity: Fidelity::Approximate,
        resources: ResourceOverlay {
            network_scope: Some(NetworkScope::Lan),
            ram_headroom: Some(Reserve::bytes(Bytes(8 << 30))),
            ..ResourceOverlay::default()
        },
        pins: vec![
            PlacementPin {
                kind: PinKind::Layers {
                    layers: LayerRange {
                        start: Some(LayerIx(0)),
                        end: Some(LayerIx(20)),
                    },
                    node: NodeSelector::Name("pc".into()),
                },
                strength: PinStrength::Require,
            },
            PlacementPin {
                kind: PinKind::Role {
                    role: StageRole::TAIL,
                    node: NodeSelector::Name("pc".into()),
                },
                strength: PinStrength::Prefer { weight: 1.0 },
            },
        ],
        tensor_overrides: vec![TensorOverride {
            pattern: r"blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight".into(),
            layers: Some(LayerRange {
                start: Some(LayerIx(40)),
                end: None,
            }),
            experts: None,
            tier: TierSelector {
                node: None,
                kind: TierKind::Nvme,
                index: None,
            },
            compute: Some(DeviceSelector::Cpu {
                node: None,
                numa: 0,
            }),
        }],
        speculative: SpeculativeMode::Off,
        domain: BTreeMap::from([
            (KeyPath::from("recovery.history"), ConfigValue::Integer(4)),
            (
                KeyPath::from("solver.context_target"),
                ConfigValue::Integer(32768),
            ),
        ]),
        context: ContextPolicy {
            context_length: Some(65536),
            max_sequences: Some(4),
        },
        nodes: NodeFilter {
            include: vec![NodeSelector::Any],
            exclude: vec![NodeSelector::Name("macbook".into())],
        },
        keep_alive: KeepAlive::For(Nanos(3_600_000_000_000)),
        preload: true,
        deterministic: false,
        dense_on_nvme: DenseNvmePolicy::Never,
    }
}

/// Golden canonical bytes. They must not change: the digest `EnsureInstance` keys instances by
/// is BLAKE3 over them, and every node, whatever its architecture or build, must produce the
/// same bytes. A change to the policy types or the canonical encoding that moves them is a
/// breaking change to the cluster protocol.
#[test]
fn canonical_bytes_golden_vectors() {
    assert_eq!(
        hex(&InstancePolicy::default().canonical_bytes()),
        DEFAULT_BYTES
    );
    assert_eq!(hex(&full_policy().canonical_bytes()), FULL_BYTES);
}

const DEFAULT_BYTES: &str = concat!(
    "0b0e00000000000000070700000000000000636f6e746578740b000000000000000007100000000000000064",
    "656661756c745f666964656c6974790c070500000000000000457861637400070d0000000000000064656e73",
    "655f6f6e5f6e766d650c070c0000000000000049664f6e6c794f7074696f6e00070d00000000000000646574",
    "65726d696e69737469630200070600000000000000646f6d61696e0a00000000000000000709000000000000",
    "0065786163746e6573730c070500000000000000457861637400070a000000000000006b6565705f616c6976",
    "650c0704000000000000004175746f000705000000000000006e6f6465730b02000000000000000707000000",
    "000000006578636c756465090000000000000000070700000000000000696e636c7564650901000000000000",
    "000c070300000000000000416e79000709000000000000006f626a6563746976650c0707000000000000004c",
    "6174656e63790007040000000000000070696e730900000000000000000707000000000000007072656c6f61",
    "6402000709000000000000007265736f75726365730b0000000000000000070b000000000000007370656375",
    "6c61746976650c0704000000000000004175746f0007100000000000000074656e736f725f6f766572726964",
    "6573090000000000000000",
);

const FULL_BYTES: &str = concat!(
    "0b0e00000000000000070700000000000000636f6e746578740b0200000000000000070e0000000000000063",
    "6f6e746578745f6c656e67746801030000010000000000070d000000000000006d61785f73657175656e6365",
    "730103040000000000000007100000000000000064656661756c745f666964656c6974790c070b0000000000",
    "0000417070726f78696d61746500070d0000000000000064656e73655f6f6e5f6e766d650c07050000000000",
    "00004e6576657200070d0000000000000064657465726d696e69737469630200070600000000000000646f6d",
    "61696e0a02000000000000000710000000000000007265636f766572792e686973746f72790c070700000000",
    "000000496e7465676572040400000000000000071500000000000000736f6c7665722e636f6e746578745f74",
    "61726765740c070700000000000000496e746567657204008000000000000007090000000000000065786163",
    "746e6573730c070b00000000000000417070726f78696d6174650b0200000000000000070500000000000000",
    "6b6e6f6273031800000000000000070600000000000000706172616d730b0100000000000000070800000000",
    "0000006b765f6c6f737379010c07020000000000000051380b0100000000000000070500000000000000626c",
    "6f636b032000000000000000070a000000000000006b6565705f616c6976650c070300000000000000466f72",
    "0300a0b830460300000705000000000000006e6f6465730b0200000000000000070700000000000000657863",
    "6c7564650901000000000000000c0704000000000000004e616d650707000000000000006d6163626f6f6b07",
    "0700000000000000696e636c7564650901000000000000000c070300000000000000416e7900070900000000",
    "0000006f626a6563746976650c070a000000000000005468726f7567687075740b0300000000000000071600",
    "0000000000007065725f73657175656e63655f666c6f6f725f747073050000a0400712000000000000007461",
    "726765745f636f6e63757272656e637901030800000000000000070c00000000000000747466745f6365696c",
    "696e670103009435770000000007040000000000000070696e730902000000000000000b0200000000000000",
    "0704000000000000006b696e640c0706000000000000004c61796572730b0200000000000000070600000000",
    "0000006c61796572730b0200000000000000070300000000000000656e640103140000000000000007050000",
    "00000000007374617274010300000000000000000704000000000000006e6f64650c0704000000000000004e",
    "616d650702000000000000007063070800000000000000737472656e6774680c070700000000000000526571",
    "75697265000b02000000000000000704000000000000006b696e640c070400000000000000526f6c650b0200",
    "0000000000000704000000000000006e6f64650c0704000000000000004e616d650702000000000000007063",
    "070400000000000000726f6c65030200000000000000070800000000000000737472656e6774680c07060000",
    "00000000005072656665720b0100000000000000070600000000000000776569676874050000803f07070000",
    "00000000007072656c6f616402010709000000000000007265736f75726365730b0200000000000000070d00",
    "0000000000006e6574776f726b5f73636f7065010c0703000000000000004c616e00070c0000000000000072",
    "616d5f68656164726f6f6d010b03000000000000000708000000000000006672616374696f6e050000000007",
    "03000000000000006d6178000703000000000000006d696e030000000002000000070b000000000000007370",
    "6563756c61746976650c0703000000000000004f66660007100000000000000074656e736f725f6f76657272",
    "696465730901000000000000000b0400000000000000070700000000000000636f6d70757465010c07030000",
    "00000000004370750b02000000000000000704000000000000006e6f6465000704000000000000006e756d61",
    "0300000000000000000706000000000000006c6179657273010b020000000000000007030000000000000065",
    "6e64000705000000000000007374617274010328000000000000000707000000000000007061747465726e07",
    "2b00000000000000626c6b5c2e285c642b295c2e66666e5f2875707c676174657c646f776e295f657870735c",
    "2e776569676874070400000000000000746965720b0300000000000000070500000000000000696e64657800",
    "0704000000000000006b696e640c0704000000000000004e766d65000704000000000000006e6f646500",
);

#[test]
fn digest_with_hashes_exactly_the_canonical_bytes() {
    let policy = full_policy();
    let mut seen = None;
    let digest = policy.digest_with(|bytes| {
        seen = Some(bytes.to_vec());
        Blake3([7; 32])
    });
    assert_eq!(digest, Blake3([7; 32]));
    assert_eq!(seen, Some(policy.canonical_bytes()));
    assert_eq!(
        policy.canonical_bytes(),
        canon::to_bytes(&policy).expect("encodes")
    );
}

#[test]
fn default_fidelity_is_exact_and_enters_the_canonical_bytes() {
    let exact = InstancePolicy::default();
    assert_eq!(exact.default_fidelity, Fidelity::Exact);
    let approximate = InstancePolicy {
        default_fidelity: Fidelity::Approximate,
        ..InstancePolicy::default()
    };
    assert_ne!(exact.canonical_bytes(), approximate.canonical_bytes());
}

#[test]
fn only_explicitly_set_overlay_fields_enter_the_canonical_bytes() {
    // An empty overlay encodes as an empty struct: no field, not a field per `None`.
    assert_eq!(
        canon::to_bytes(&ResourceOverlay::default()).expect("encodes"),
        [0x0b, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    let base = InstancePolicy::default().canonical_bytes();
    let set: [ResourceOverlay; 8] = [
        ResourceOverlay {
            memory_yield: Some(MemoryYield::UnderPressure),
            ..Default::default()
        },
        ResourceOverlay {
            max_pinned_fraction: Some(0.5),
            ..Default::default()
        },
        ResourceOverlay {
            power: Some(PowerPolicy::Balanced),
            ..Default::default()
        },
        ResourceOverlay {
            network_scope: Some(NetworkScope::LanAndInvites),
            ..Default::default()
        },
        ResourceOverlay {
            ram_headroom: Some(Reserve::default()),
            ..Default::default()
        },
        ResourceOverlay {
            vram_headroom: Some(Reserve::default()),
            ..Default::default()
        },
        ResourceOverlay {
            reserve_cores: Some(0),
            ..Default::default()
        },
        ResourceOverlay {
            gpus: Some(DeviceFilter::default()),
            ..Default::default()
        },
    ];
    let mut seen = vec![base];
    for resources in set {
        // Setting a field, even to the value a node would default to, is an explicit choice
        // and changes the bytes; each field changes them differently.
        let bytes = InstancePolicy {
            resources,
            ..InstancePolicy::default()
        }
        .canonical_bytes();
        assert!(!seen.contains(&bytes));
        seen.push(bytes);
    }
}

#[test]
fn domain_keys_enter_the_canonical_bytes() {
    let mut policy = InstancePolicy::default();
    let before = policy.canonical_bytes();
    policy.domain.insert(
        KeyPath::from("ring.max_inflight_steps"),
        ConfigValue::Integer(2),
    );
    let one = policy.canonical_bytes();
    assert_ne!(before, one);
    policy.domain.insert(
        KeyPath::from("ring.max_inflight_steps"),
        ConfigValue::Integer(3),
    );
    assert_ne!(one, policy.canonical_bytes());
}

#[test]
fn every_field_enters_the_canonical_bytes() {
    let full = full_policy();
    let default = InstancePolicy::default();
    let d = full.canonical_bytes();
    let swaps: [(&str, InstancePolicy); 14] = [
        (
            "objective",
            InstancePolicy {
                objective: default.objective,
                ..full.clone()
            },
        ),
        (
            "exactness",
            InstancePolicy {
                exactness: default.exactness,
                ..full.clone()
            },
        ),
        (
            "default_fidelity",
            InstancePolicy {
                default_fidelity: default.default_fidelity,
                ..full.clone()
            },
        ),
        (
            "resources",
            InstancePolicy {
                resources: default.resources.clone(),
                ..full.clone()
            },
        ),
        (
            "pins",
            InstancePolicy {
                pins: Vec::new(),
                ..full.clone()
            },
        ),
        (
            "tensor_overrides",
            InstancePolicy {
                tensor_overrides: Vec::new(),
                ..full.clone()
            },
        ),
        (
            "speculative",
            InstancePolicy {
                speculative: SpeculativeMode::Auto,
                ..full.clone()
            },
        ),
        (
            "domain",
            InstancePolicy {
                domain: BTreeMap::new(),
                ..full.clone()
            },
        ),
        (
            "context",
            InstancePolicy {
                context: ContextPolicy::default(),
                ..full.clone()
            },
        ),
        (
            "nodes",
            InstancePolicy {
                nodes: NodeFilter::default(),
                ..full.clone()
            },
        ),
        (
            "keep_alive",
            InstancePolicy {
                keep_alive: KeepAlive::Auto,
                ..full.clone()
            },
        ),
        (
            "preload",
            InstancePolicy {
                preload: false,
                ..full.clone()
            },
        ),
        (
            "deterministic",
            InstancePolicy {
                deterministic: true,
                ..full.clone()
            },
        ),
        (
            "dense_on_nvme",
            InstancePolicy {
                dense_on_nvme: DenseNvmePolicy::IfOnlyOption,
                ..full.clone()
            },
        ),
    ];
    for (field, policy) in swaps {
        assert_ne!(
            policy.canonical_bytes(),
            d,
            "{field} does not reach the canonical bytes"
        );
    }
}

// ---- Defaults and older records ---------------------------------------------------------

#[test]
fn instance_policy_defaults() {
    let p = InstancePolicy::default();
    assert_eq!(p.objective, Objective::Latency);
    assert_eq!(p.exactness, ExactnessPolicy::Exact);
    assert_eq!(p.default_fidelity, Fidelity::Exact);
    assert_eq!(p.resources, ResourceOverlay::default());
    assert_eq!(p.speculative, SpeculativeMode::Auto);
    assert_eq!(p.nodes.include, [NodeSelector::Any]);
    assert!(p.nodes.exclude.is_empty());
    assert_eq!(p.keep_alive, KeepAlive::Auto);
    assert_eq!(p.dense_on_nvme, DenseNvmePolicy::IfOnlyOption);
    assert!(!p.preload && !p.deterministic);
}

#[test]
fn resource_policy_defaults_are_rfc_0054_personal() {
    let p = ResourcePolicy::default();
    assert_eq!(p.memory_yield, MemoryYield::UnderPressure);
    assert_eq!(p.max_pinned_fraction, None);
    assert_eq!(p.power, PowerPolicy::Balanced);
    assert_eq!(p.network_scope, NetworkScope::LanAndInvites);
    assert_eq!(p.mode, UsageMode::Personal);
    assert_eq!(p.min_battery_percent, 30);
    assert_eq!(p.gpus.allow, [DeviceSelector::Any]);
    assert!(!p.macos_raise_wired_limit);
    // RFC-0054 §4.3: a 128 GB desktop keeps 12.8 GB. The share is an `f32`, so 0.10 is
    // 0.100000001…, a few hundred bytes over.
    let kept = p.ram_headroom.resolve(Bytes(128_000_000_000)).0;
    assert!((12_800_000_000..12_800_001_000).contains(&kept), "{kept}");
    // A 16 GiB laptop keeps the 6 GiB floor.
    assert_eq!(p.ram_headroom.resolve(Bytes(16 << 30)), Bytes(6 << 30));
}

#[test]
fn rfc_0001_records_decode() {
    let policy: InstancePolicy = serde_json::from_str(
        r#"{"objective":{"Throughput":{"per_sequence_floor_tps":5.0}},
            "exactness":{"Approximate":{"knobs":"KV_LOSSY"}}}"#,
    )
    .expect("an RFC-0001 policy decodes");
    assert_eq!(
        policy.objective,
        Objective::Throughput {
            per_sequence_floor_tps: 5.0,
            target_concurrency: None,
            ttft_ceiling: None,
        }
    );
    assert_eq!(
        policy.exactness,
        ExactnessPolicy::Approximate {
            knobs: ApproxKnobs::KV_LOSSY,
            params: ApproxParams::default(),
        }
    );
    assert_eq!(policy.default_fidelity, Fidelity::Exact);
    let resources: ResourcePolicy = serde_json::from_str(
        r#"{"memory_yield":"Never","max_pinned_fraction":0.5,"power":"Battery","network_scope":"Lan"}"#,
    )
    .expect("an RFC-0001 resource policy decodes");
    assert_eq!(resources.memory_yield, MemoryYield::Never);
    assert_eq!(resources.min_battery_percent, 30);
}

#[test]
fn fraction_rejects_values_outside_0_to_1() {
    assert!(Fraction::new(1.5).is_none());
    assert!(Fraction::new(f32::NAN).is_none());
    assert_eq!(
        Fraction::new(-0.0).map(Fraction::get).map(f32::to_bits),
        Some(0)
    );
    assert!(serde_json::from_str::<Fraction>("1.5").is_err());
    assert!(serde_json::from_str::<Fraction>("-0.1").is_err());
    assert_eq!(
        serde_json::from_str::<Fraction>("0.25").ok(),
        Fraction::new(0.25)
    );
}

#[test]
fn reserve_validity_and_cap() {
    assert!(Reserve::min_fraction(Bytes(1), 0.5).is_valid());
    assert!(!Reserve::min_fraction(Bytes(1), 1.5).is_valid());
    let capped = Reserve {
        min: Bytes(2),
        fraction: 0.5,
        max: Some(Bytes(10)),
    };
    assert!(capped.is_valid());
    assert_eq!(capped.resolve(Bytes(100)), Bytes(10));
    assert!(
        !Reserve {
            max: Some(Bytes(1)),
            ..capped
        }
        .is_valid()
    );
}

/// -0.0 and 0.0 compare equal, so a policy that writes either must give one digest.
#[test]
fn negative_zero_floats_encode_as_zero() {
    fn with(z: f32, zf: f64) -> InstancePolicy {
        InstancePolicy {
            objective: Objective::Throughput {
                per_sequence_floor_tps: z,
                target_concurrency: None,
                ttft_ceiling: None,
            },
            resources: ResourceOverlay {
                max_pinned_fraction: Some(z),
                ram_headroom: Some(Reserve::min_fraction(Bytes(0), z)),
                ..ResourceOverlay::default()
            },
            pins: vec![PlacementPin {
                kind: PinKind::Exclude {
                    target: ExcludeTarget::Node(NodeSelector::Any),
                },
                strength: PinStrength::Prefer { weight: z },
            }],
            domain: BTreeMap::from([(KeyPath::from("x.y"), ConfigValue::Float(zf))]),
            ..InstancePolicy::default()
        }
    }
    assert_eq!(
        with(-0.0, -0.0).canonical_bytes(),
        with(0.0, 0.0).canonical_bytes()
    );
    // Each field on its own, too.
    let zero = with(0.0, 0.0);
    let one = |p: &mut InstancePolicy| {
        assert_eq!(p.canonical_bytes(), zero.canonical_bytes());
    };
    let mut p = zero.clone();
    p.objective = Objective::Throughput {
        per_sequence_floor_tps: -0.0,
        target_concurrency: None,
        ttft_ceiling: None,
    };
    one(&mut p);
    let mut p = zero.clone();
    p.resources.max_pinned_fraction = Some(-0.0);
    one(&mut p);
    let mut p = zero.clone();
    p.resources.ram_headroom = Some(Reserve::min_fraction(Bytes(0), -0.0));
    one(&mut p);
    let mut p = zero.clone();
    p.pins[0].strength = PinStrength::Prefer { weight: -0.0 };
    one(&mut p);
    let mut p = zero.clone();
    p.domain
        .insert(KeyPath::from("x.y"), ConfigValue::Float(-0.0));
    one(&mut p);
    // `ResourcePolicy`, a node's own record, encodes the same way.
    let a = ResourcePolicy {
        max_pinned_fraction: Some(-0.0),
        ..ResourcePolicy::default()
    };
    let b = ResourcePolicy {
        max_pinned_fraction: Some(0.0),
        ..ResourcePolicy::default()
    };
    assert_eq!(
        canon::to_bytes(&a).expect("encodes"),
        canon::to_bytes(&b).expect("encodes")
    );
}
