// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serde properties of the RFC-0020 §3 types (TIER-E1-T01).
//!
//! Every serializable type round-trips through a serde format unchanged, records decode with
//! fields this build does not know, and defaulted fields may be left out. JSON stands in for
//! any self-describing format.

use std::fmt::Debug;
use std::sync::Arc;

use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use smallvec::SmallVec;
use syntropy_core::plan::DeltaItem;
use syntropy_core::{Bytes, DeviceId, LayerIx, NodeId, PlanDelta, PlanEpoch, StageIx, TierId};
use syntropy_tier::*;

// ---- Strategies -------------------------------------------------------------------------

fn bytes() -> impl Strategy<Value = Bytes> {
    any::<u64>().prop_map(Bytes)
}

fn node_id() -> impl Strategy<Value = NodeId> {
    any::<[u8; 32]>().prop_map(NodeId)
}

fn device_id() -> impl Strategy<Value = DeviceId> {
    (node_id(), any::<u16>()).prop_map(|(node, ordinal)| DeviceId { node, ordinal })
}

fn tier_id() -> impl Strategy<Value = TierId> {
    (node_id(), any::<u16>()).prop_map(|(node, ordinal)| TierId { node, ordinal })
}

fn purpose() -> impl Strategy<Value = Purpose> {
    prop_oneof![
        Just(Purpose::Weights),
        Just(Purpose::ExpertCache),
        Just(Purpose::ExpertStaging),
        Just(Purpose::Kv),
        Just(Purpose::Scratch),
        Just(Purpose::Graphs),
        Just(Purpose::IoStaging),
        Just(Purpose::NetPool),
        Just(Purpose::RowCache),
        Just(Purpose::History),
    ]
}

fn pin_class() -> impl Strategy<Value = PinClass> {
    prop_oneof![
        Just(PinClass::DmaPinned),
        Just(PinClass::Locked),
        Just(PinClass::Pageable),
    ]
}

fn registrations() -> impl Strategy<Value = Registrations> {
    any::<u8>().prop_map(Registrations::from_bits_truncate)
}

fn slot_class() -> impl Strategy<Value = SlotClass> {
    (
        bytes(),
        any::<u32>(),
        proptest::option::of(any::<u16>().prop_map(LayerIx)),
    )
        .prop_map(|(size, count, layer)| SlotClass { size, count, layer })
}

fn slot_geometry() -> impl Strategy<Value = SlotGeometry> {
    prop_oneof![
        bytes().prop_map(|align| SlotGeometry::Extents { align }),
        prop::collection::vec(slot_class(), 0..70)
            .prop_map(|v| SlotGeometry::Classes(SmallVec::from_vec(v))),
        (bytes(), any::<u32>()).prop_map(|(page, count)| SlotGeometry::Pages { page, count }),
        (bytes(), any::<u32>()).prop_map(|(slot, count)| SlotGeometry::Ring { slot, count }),
        Just(SlotGeometry::Bump),
    ]
}

fn segment_spec() -> impl Strategy<Value = SegmentSpec> {
    (bytes(), any::<u32>()).prop_map(|(segment, min_segments)| SegmentSpec {
        segment,
        min_segments,
    })
}

fn arena_spec() -> impl Strategy<Value = ArenaSpec> {
    (
        (tier_id(), purpose(), bytes(), bytes(), slot_geometry()),
        (
            pin_class(),
            proptest::option::of(any::<u16>()),
            registrations(),
            prop::collection::vec(device_id(), 0..6),
            any::<u16>(),
            proptest::option::of(segment_spec()),
        ),
    )
        .prop_map(
            |(
                (tier, purpose, bytes, max_bytes, geometry),
                (pin, numa, registrations, dma, pin_priority, elastic),
            )| ArenaSpec {
                tier,
                purpose,
                bytes,
                max_bytes,
                geometry,
                pin,
                numa,
                registrations,
                dma_devices: SmallVec::from_vec(dma),
                pin_priority,
                elastic,
            },
        )
}

fn tier_budget_line() -> impl Strategy<Value = TierBudgetLine> {
    (
        bytes(),
        prop::collection::vec((purpose(), bytes()), 0..12),
        bytes(),
    )
        .prop_map(|(reserved, by, headroom)| TierBudgetLine {
            reserved,
            by_purpose: SmallVec::from_vec(by),
            headroom,
        })
}

fn arena_plan() -> impl Strategy<Value = ArenaPlan> {
    (
        prop::collection::vec(arena_spec(), 0..4),
        prop::collection::vec((tier_id(), tier_budget_line()), 0..4),
    )
        .prop_map(|(arenas, per_tier)| ArenaPlan { arenas, per_tier })
}

fn arena_ref() -> impl Strategy<Value = ArenaRef> {
    (any::<u16>(), any::<u8>()).prop_map(|(shard, arena)| ArenaRef { shard, arena })
}

fn step_fence() -> impl Strategy<Value = StepFence> {
    (any::<u16>(), any::<u64>()).prop_map(|(device, step)| StepFence { device, step })
}

fn view_kind() -> impl Strategy<Value = ViewKind> {
    prop_oneof![
        Just(ViewKind::Host),
        device_id().prop_map(ViewKind::Device),
        Just(ViewKind::Unified),
    ]
}

fn buffer_view() -> impl Strategy<Value = BufferView> {
    (view_kind(), any::<u64>(), bytes()).prop_map(|(kind, addr, len)| BufferView {
        kind,
        addr,
        len,
    })
}

fn pressure_level() -> impl Strategy<Value = PressureLevel> {
    prop_oneof![
        Just(PressureLevel::Normal),
        Just(PressureLevel::Moderate),
        Just(PressureLevel::High),
        Just(PressureLevel::Critical),
    ]
}

fn pin_state() -> impl Strategy<Value = PinState> {
    prop_oneof![
        Just(PinState::Running),
        Just(PinState::Done),
        bytes().prop_map(|underpinned| PinState::GaveUp { underpinned }),
    ]
}

fn pin_progress() -> impl Strategy<Value = PinProgress> {
    (
        bytes(),
        bytes(),
        pin_state(),
        proptest::option::of(any::<u64>()),
    )
        .prop_map(|(done, target, state, eta_ms)| PinProgress {
            done,
            target,
            state,
            eta_ms,
        })
}

fn plan_epoch() -> impl Strategy<Value = PlanEpoch> {
    (any::<u64>(), any::<u64>()).prop_map(|(term, seq)| PlanEpoch { term, seq })
}

fn plan_delta() -> impl Strategy<Value = PlanDelta> {
    let item = prop::collection::vec(any::<u16>().prop_map(StageIx), 0..3).prop_map(|s| {
        DeltaItem::ExtendShard {
            stages: SmallVec::from_vec(s),
        }
    });
    (
        plan_epoch(),
        plan_epoch(),
        prop::collection::vec(item, 0..3),
    )
        .prop_map(|(from, to, changes)| PlanDelta { from, to, changes })
}

fn range() -> impl Strategy<Value = std::ops::Range<u64>> {
    (any::<u64>(), any::<u64>()).prop_map(|(a, b)| a.min(b)..a.max(b))
}

fn tier_command() -> impl Strategy<Value = TierCommand> {
    prop_oneof![
        (arena_ref(), range()).prop_map(|(arena, range)| TierCommand::PinStep { arena, range }),
        (arena_ref(), range()).prop_map(|(arena, range)| TierCommand::Unpin { arena, range }),
        (arena_ref(), any::<u32>())
            .prop_map(|(arena, segments)| TierCommand::ShrinkElastic { arena, segments }),
        (arena_ref(), any::<u32>())
            .prop_map(|(arena, segments)| TierCommand::GrowElastic { arena, segments }),
        (pressure_level(), bytes())
            .prop_map(|(level, reclaim)| TierCommand::Yield { level, reclaim }),
        plan_delta().prop_map(|d| TierCommand::ApplyDelta { delta: Arc::new(d) }),
    ]
}

// ---- Helpers ----------------------------------------------------------------------------

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&back, value, "round trip changed the value; json: {json}");
}

/// Adds an unknown field to the top-level JSON object. Nested objects are left alone: an
/// externally tagged enum is an object with exactly one key.
fn add_unknown_fields(value: &mut serde_json::Value) {
    value
        .as_object_mut()
        .expect("a struct serializes as an object")
        .insert("field_from_a_newer_build".into(), serde_json::json!(1));
}

fn tolerates_unknown_fields<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let mut json = serde_json::to_value(value).expect("serialize");
    add_unknown_fields(&mut json);
    let back: T = serde_json::from_value(json.clone()).expect("deserialize with unknown fields");
    assert_eq!(
        &back, value,
        "unknown fields changed the value; json: {json}"
    );
}

// ---- Properties -------------------------------------------------------------------------

proptest! {
    #[test]
    fn purpose_round_trips(v in purpose()) { round_trip(&v); }

    #[test]
    fn pin_class_round_trips(v in pin_class()) { round_trip(&v); }

    #[test]
    fn registrations_round_trip(v in registrations()) { round_trip(&v); }

    #[test]
    fn slot_geometry_round_trips(v in slot_geometry()) { round_trip(&v); }

    #[test]
    fn arena_spec_round_trips(v in arena_spec()) {
        round_trip(&v);
        tolerates_unknown_fields(&v);
    }

    #[test]
    fn arena_plan_round_trips(v in arena_plan()) {
        round_trip(&v);
        tolerates_unknown_fields(&v);
    }

    #[test]
    fn tier_budget_line_round_trips(v in tier_budget_line()) {
        round_trip(&v);
        tolerates_unknown_fields(&v);
    }

    #[test]
    fn step_fence_round_trips(v in step_fence()) { round_trip(&v); tolerates_unknown_fields(&v); }

    #[test]
    fn buffer_view_round_trips(v in buffer_view()) { round_trip(&v); tolerates_unknown_fields(&v); }

    #[test]
    fn tier_command_round_trips(v in tier_command()) { round_trip(&v); }

    #[test]
    fn arena_ref_round_trips(v in arena_ref()) { round_trip(&v); }

    #[test]
    fn pin_progress_round_trips(v in pin_progress()) { round_trip(&v); tolerates_unknown_fields(&v); }

    #[test]
    fn pressure_level_round_trips(v in pressure_level()) { round_trip(&v); }
}

// ---- Serde defaults ---------------------------------------------------------------------

/// Removes `fields` from a JSON object.
fn without(mut json: serde_json::Value, fields: &[&str]) -> serde_json::Value {
    let map = json.as_object_mut().expect("object");
    for f in fields {
        assert!(map.remove(*f).is_some(), "fixture has no field {f}");
    }
    json
}

fn fixture_arena() -> ArenaSpec {
    ArenaSpec {
        tier: TierId {
            node: NodeId([7; 32]),
            ordinal: 3,
        },
        purpose: Purpose::ExpertCache,
        bytes: Bytes(1 << 30),
        max_bytes: Bytes(4 << 30),
        geometry: SlotGeometry::Classes(SmallVec::from_slice(&[SlotClass {
            size: Bytes(1 << 20),
            count: 16,
            layer: Some(LayerIx(2)),
        }])),
        pin: PinClass::DmaPinned,
        numa: Some(1),
        registrations: Registrations::GPU_DMA | Registrations::IO_FIXED,
        dma_devices: SmallVec::from_slice(&[DeviceId {
            node: NodeId([7; 32]),
            ordinal: 0,
        }]),
        pin_priority: 1,
        elastic: Some(SegmentSpec {
            segment: Bytes(256 << 20),
            min_segments: 2,
        }),
    }
}

#[test]
fn arena_spec_defaults_its_optional_fields() {
    let full = fixture_arena();
    let json = without(
        serde_json::to_value(&full).unwrap(),
        &["numa", "registrations", "dma_devices", "elastic"],
    );
    let back: ArenaSpec = serde_json::from_value(json).unwrap();
    assert_eq!(back.numa, None);
    assert_eq!(back.registrations, Registrations::empty());
    assert!(back.dma_devices.is_empty());
    assert_eq!(back.elastic, None);
    assert_eq!(
        ArenaSpec {
            numa: Some(1),
            registrations: full.registrations,
            dma_devices: full.dma_devices.clone(),
            elastic: full.elastic,
            ..back
        },
        full
    );
}

#[test]
fn arena_spec_requires_its_sizing_fields() {
    let full = serde_json::to_value(fixture_arena()).unwrap();
    for field in [
        "tier",
        "purpose",
        "bytes",
        "max_bytes",
        "geometry",
        "pin",
        "pin_priority",
    ] {
        let json = without(full.clone(), &[field]);
        assert!(
            serde_json::from_value::<ArenaSpec>(json).is_err(),
            "{field} decoded without a value"
        );
    }
}

#[test]
fn slot_class_layer_defaults_to_none() {
    let back: SlotClass = serde_json::from_str(r#"{"size":64,"count":3}"#).unwrap();
    assert_eq!(
        back,
        SlotClass {
            size: Bytes(64),
            count: 3,
            layer: None
        }
    );
}

#[test]
fn arena_plan_and_budget_line_default_every_field() {
    assert_eq!(
        serde_json::from_str::<ArenaPlan>("{}").unwrap(),
        ArenaPlan::default()
    );
    assert_eq!(
        serde_json::from_str::<TierBudgetLine>("{}").unwrap(),
        TierBudgetLine::default()
    );
}

#[test]
fn pin_progress_eta_defaults_to_none() {
    let back: PinProgress =
        serde_json::from_str(r#"{"done":1,"target":2,"state":"Running"}"#).unwrap();
    assert_eq!(back.eta_ms, None);
}

// ---- Serde defaults in sequence form ------------------------------------------------------
//
// A struct may also arrive as a sequence (a JSON array; compact formats always use one). There
// a missing trailing field is an `invalid_length` error unless the field has
// `#[serde(default)]`, even for an `Option`. These tests pin the defaults of the trailing
// fields; a defaulted field in the middle (`ArenaSpec::numa`) cannot be left out of a sequence.

#[test]
fn slot_class_short_sequence_defaults_layer() {
    let back: SlotClass = serde_json::from_str("[64,3]").unwrap();
    assert_eq!(
        back,
        SlotClass {
            size: Bytes(64),
            count: 3,
            layer: None
        }
    );
}

#[test]
fn pin_progress_short_sequence_defaults_eta() {
    let back: PinProgress = serde_json::from_str(r#"[1,2,"Running"]"#).unwrap();
    assert_eq!(
        back,
        PinProgress {
            done: Bytes(1),
            target: Bytes(2),
            state: PinState::Running,
            eta_ms: None
        }
    );
}

#[test]
fn arena_spec_short_sequence_defaults_elastic() {
    const ORDER: [&str; 10] = [
        "tier",
        "purpose",
        "bytes",
        "max_bytes",
        "geometry",
        "pin",
        "numa",
        "registrations",
        "dma_devices",
        "pin_priority",
    ];
    let full = fixture_arena();
    let map = serde_json::to_value(&full).unwrap();
    let seq: Vec<serde_json::Value> = ORDER.iter().map(|k| map[*k].clone()).collect();
    let back: ArenaSpec = serde_json::from_value(serde_json::Value::Array(seq)).unwrap();
    assert_eq!(back.elastic, None);
    assert_eq!(
        ArenaSpec {
            elastic: full.elastic,
            ..back
        },
        full
    );
}
