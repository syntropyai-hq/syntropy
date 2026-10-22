// SPDX-License-Identifier: MIT OR Apache-2.0

//! Registration units and Windows lock ranges (TIER-E2-T03; RFC-0020 §4.2, §6.1): no slot, and
//! so no DMA, spans two lock ranges. OS-independent.

use proptest::prelude::*;
use smallvec::smallvec;
use syntropy_core::Bytes;
use syntropy_tier::arena::{SlotClass, SlotGeometry};
use syntropy_tier::{LockRange, UnitSlot, lock_ranges, slot_layout};

const MIB: u64 = 1 << 20;

/// Every property the layout must hold, checked in one place.
fn check(slots: &[UnitSlot], step: Bytes, ranges: &[LockRange]) -> Result<(), TestCaseError> {
    let total: u64 = slots.iter().map(|s| s.len.0).sum();
    // The ranges cover the slots and the bytes once, in order, without gaps.
    let mut next_slot = 0;
    let mut next_byte = 0;
    for r in ranges {
        prop_assert_eq!(r.slots.start, next_slot);
        prop_assert_eq!(r.bytes.start, next_byte);
        prop_assert!(r.slots.start < r.slots.end, "range with no slots {:?}", r);
        prop_assert!(!r.is_empty(), "range with no bytes {:?}", r);
        next_slot = r.slots.end;
        next_byte = r.bytes.end;
        let sum: u64 = slots[r.slots.clone()].iter().map(|s| s.len.0).sum();
        prop_assert_eq!(sum, r.len().0, "range {:?} is not its slots", r);
        // At most pin_step, unless the range is one slot that alone exceeds it.
        let with_bytes: Vec<&UnitSlot> = slots[r.slots.clone()]
            .iter()
            .filter(|s| s.len.0 > 0)
            .collect();
        prop_assert!(
            r.len() <= step || (with_bytes.len() == 1 && with_bytes[0].len > step),
            "range {:?} exceeds {:?}",
            r,
            step
        );
        // One group per range, among the slots that hold bytes.
        let g = with_bytes[0].group;
        prop_assert!(with_bytes.iter().all(|s| s.group == g));
    }
    if total == 0 {
        prop_assert!(
            ranges.is_empty(),
            "ranges for a layout with no bytes: {:?}",
            ranges
        );
    } else {
        prop_assert_eq!(next_slot, slots.len());
        prop_assert_eq!(next_byte, total);
    }

    // No slot spans two ranges: each slot's bytes lie inside the range that holds it.
    let mut offset = 0;
    for (i, s) in slots.iter().enumerate() {
        let (start, end) = (offset, offset + s.len.0);
        offset = end;
        if s.len.0 == 0 {
            continue;
        }
        let overlapping: Vec<_> = ranges
            .iter()
            .filter(|r| r.bytes.start < end && start < r.bytes.end)
            .collect();
        prop_assert_eq!(
            overlapping.len(),
            1,
            "slot {} at {}..{} is in {:?}",
            i,
            start,
            end,
            overlapping
        );
        let r = overlapping[0];
        prop_assert!(r.bytes.start <= start && end <= r.bytes.end && r.slots.contains(&i));
    }

    // Greedy: a range ends only at a group change, at the step, or at the end.
    let first_with_bytes =
        |r: &LockRange| *slots[r.slots.clone()].iter().find(|s| s.len.0 > 0).unwrap();
    for w in ranges.windows(2) {
        let next = first_with_bytes(&w[1]);
        let same_group = first_with_bytes(&w[0]).group == next.group;
        prop_assert!(!same_group || w[0].len().0 + next.len.0 > step.0);
    }
    Ok(())
}

fn slots() -> impl Strategy<Value = Vec<UnitSlot>> {
    proptest::collection::vec(
        (0u32..4, prop_oneof![1 => Just(0u64), 7 => 0u64..96 * MIB]),
        0..64,
    )
    .prop_map(|mut v| {
        // Groups come in runs, as layers and slot classes do.
        v.sort_by_key(|(g, _)| *g);
        v.into_iter()
            .map(|(group, len)| UnitSlot {
                group,
                len: Bytes(len),
            })
            .collect()
    })
}

proptest! {
    #[test]
    fn no_slot_spans_two_lock_ranges(slots in slots(), step in (1u64..128).prop_map(|m| m * MIB)) {
        let ranges = lock_ranges(&slots, Bytes(step));
        check(&slots, Bytes(step), &ranges)?;
    }

    /// Slot-class geometry, as an expert cache lays it out: one class per MoE layer.
    #[test]
    fn slot_classes_never_straddle(
        classes in proptest::collection::vec((1u64..64, 1u32..40), 1..8),
        step in (1u64..256).prop_map(|m| m * MIB),
    ) {
        let geometry = SlotGeometry::Classes(
            classes
                .iter()
                .map(|&(mib, count)| SlotClass { size: Bytes(mib * MIB), count, layer: None })
                .collect(),
        );
        let slots = slot_layout(&geometry, &[]);
        let ranges = lock_ranges(&slots, Bytes(step));
        check(&slots, Bytes(step), &ranges)?;
    }
}

#[test]
fn extents_are_padded_and_cut_by_layer() {
    let geometry = SlotGeometry::Extents { align: Bytes(4096) };
    let slots = slot_layout(
        &geometry,
        &[(0, Bytes(1000)), (0, Bytes(5000)), (1, Bytes(4096))],
    );
    assert_eq!(
        slots.iter().map(|s| s.len.0).collect::<Vec<_>>(),
        [4096, 8192, 4096]
    );
    let ranges = lock_ranges(&slots, Bytes(1 << 30));
    assert_eq!(
        ranges,
        [
            LockRange {
                slots: 0..2,
                bytes: 0..12288
            },
            LockRange {
                slots: 2..3,
                bytes: 12288..16384
            },
        ]
    );
}

#[test]
fn an_oversized_slot_is_its_own_range() {
    let slots = [
        UnitSlot {
            group: 0,
            len: Bytes(MIB),
        },
        UnitSlot {
            group: 0,
            len: Bytes(3 * MIB),
        },
        UnitSlot {
            group: 0,
            len: Bytes(MIB),
        },
    ];
    let ranges = lock_ranges(&slots, Bytes(2 * MIB));
    assert_eq!(
        ranges.iter().map(|r| r.slots.clone()).collect::<Vec<_>>(),
        [0..1, 1..2, 2..3]
    );
}

#[test]
fn rings_and_bump_arenas_are_one_unit() {
    let ring = slot_layout(
        &SlotGeometry::Ring {
            slot: Bytes(MIB),
            count: 8,
        },
        &[],
    );
    assert_eq!(ring.len(), 1);
    assert_eq!(ring[0].len, Bytes(8 * MIB));
    let bump = slot_layout(&SlotGeometry::Bump, &[(0, Bytes(5 * MIB))]);
    assert_eq!(bump.len(), 1);
    let pages = slot_layout(
        &SlotGeometry::Pages {
            page: Bytes(MIB),
            count: 3,
        },
        &[],
    );
    assert_eq!(pages.len(), 3);
    assert_eq!(lock_ranges(&[], Bytes(MIB)), []);
    assert!(slot_layout(&SlotGeometry::Classes(smallvec![]), &[]).is_empty());
}

/// Zero-length slots never make an empty range: they join a neighbouring range, and a layout
/// with no bytes gives no ranges (review finding 6 of #1659).
#[test]
fn zero_length_slots_make_no_empty_range() {
    let z = |group| UnitSlot {
        group,
        len: Bytes(0),
    };
    let b = |group, mib| UnitSlot {
        group,
        len: Bytes(mib * MIB),
    };
    assert_eq!(lock_ranges(&[z(0), z(1)], Bytes(MIB)), []);
    assert_eq!(
        lock_ranges(&slot_layout(&SlotGeometry::Bump, &[]), Bytes(MIB)),
        []
    );
    let ranges = lock_ranges(&[z(0), b(0, 1), z(1), b(1, 1), z(1)], Bytes(4 * MIB));
    assert_eq!(
        ranges,
        [
            LockRange {
                slots: 0..3,
                bytes: 0..MIB
            },
            LockRange {
                slots: 3..5,
                bytes: MIB..2 * MIB
            },
        ]
    );
}
