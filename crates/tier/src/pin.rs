// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive pinning (RFC-0020 §6): the progress types, and the registration units of §6.1,
//! which on Windows are also the lock ranges (RFC-0020 §4.2). The pinning loop and its gate
//! come in TIER-E2-T07.

#![forbid(unsafe_code)]

use std::ops::Range;

use serde::{Deserialize, Serialize};
use syntropy_core::Bytes;

use crate::arena::SlotGeometry;

/// How far progressive pinning has got (RFC-0020 §3, supporting types).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PinProgress {
    /// Bytes pinned so far.
    pub done: Bytes,
    /// Bytes the plan asks to pin.
    pub target: Bytes,
    /// Where the loop is.
    pub state: PinState,
    /// Estimated time to finish, in milliseconds. Defaults to `None` (no estimate).
    #[serde(default)]
    pub eta_ms: Option<u64>,
}

/// The state of the pinning loop (RFC-0020 §3, §6).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinState {
    /// Pinning is in progress.
    Running,
    /// Every range is at its planned pin class.
    Done,
    /// Pressure or capacity failures lasted `tier.pin_give_up`; the remaining ranges are
    /// `Underpinned`.
    GaveUp {
        /// Bytes left below their planned pin class.
        underpinned: Bytes,
    },
}

// ---- Registration units and lock ranges (RFC-0020 §4.2, §6.1) -----------------------------

/// One slot of an arena's layout, in layout order: a DMA source is at most one slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitSlot {
    /// What the slot belongs to: the layer of a weight extent, the class of a cache slot. A
    /// lock range never mixes groups ("one per layer for weights, one per slot-class block for
    /// caches", RFC-0020 §4.2).
    pub group: u32,
    /// The slot's length, padding to the next slot's alignment included.
    pub len: Bytes,
}

/// One registration unit, which on Windows is also one lock range and one reservation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockRange {
    /// The slots it holds, as indices into the layout.
    pub slots: Range<usize>,
    /// The bytes it holds, as offsets into the arena's logical layout.
    pub bytes: Range<u64>,
}

impl LockRange {
    /// The range's length.
    pub fn len(&self) -> Bytes {
        Bytes(self.bytes.end - self.bytes.start)
    }

    /// Whether the range holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.start == self.bytes.end
    }
}

/// The slots of an arena in layout order (RFC-0020 §3).
///
/// - `Extents`: one slot per extent, `extents[i] = (layer, bytes)`, each padded to `align`.
/// - `Classes`: class `i`'s slots, all in group `i`, one class after another.
/// - `Pages`: one slot per page.
/// - `Ring` and `Bump`: the whole arena is one slot, so it is one unit (RFC-0020 §6.1);
///   `Bump` takes its length from `extents`' sum (normally one entry).
pub fn slot_layout(geometry: &SlotGeometry, extents: &[(u32, Bytes)]) -> Vec<UnitSlot> {
    match geometry {
        SlotGeometry::Extents { align } => extents
            .iter()
            .map(|&(group, len)| UnitSlot {
                group,
                len: Bytes(match align.0 {
                    0 | 1 => len.0,
                    a => len.0.div_ceil(a).saturating_mul(a),
                }),
            })
            .collect(),
        SlotGeometry::Classes(classes) => classes
            .iter()
            .enumerate()
            .flat_map(|(i, c)| {
                std::iter::repeat_n(
                    UnitSlot {
                        group: i as u32,
                        len: c.size,
                    },
                    c.count as usize,
                )
            })
            .collect(),
        SlotGeometry::Pages { page, count } => std::iter::repeat_n(
            UnitSlot {
                group: 0,
                len: *page,
            },
            *count as usize,
        )
        .collect(),
        SlotGeometry::Ring { slot, count } => vec![UnitSlot {
            group: 0,
            len: Bytes(slot.0.saturating_mul(u64::from(*count))),
        }],
        SlotGeometry::Bump => vec![UnitSlot {
            group: 0,
            len: Bytes(
                extents
                    .iter()
                    .fold(0u64, |s, &(_, l)| s.saturating_add(l.0)),
            ),
        }],
    }
}

/// Cuts a layout into registration units, which on Windows are the lock ranges (RFC-0020
/// §4.2, §6.1).
///
/// Units are cut only between slots, so no slot, and so no DMA, spans two of them. A unit
/// never mixes groups, and holds as many whole slots as fit in `pin_step` (`tier.pin_step`,
/// passed in until the config schema has the key, TIER-E1-T09). A slot longer than `pin_step`
/// cannot be split without a DMA crossing a boundary, so it is a unit of its own, the one case
/// where a unit exceeds `pin_step` (RFC-0020 §6.1, "that extent is its own unit").
pub fn lock_ranges(slots: &[UnitSlot], pin_step: Bytes) -> Vec<LockRange> {
    let mut out: Vec<LockRange> = Vec::new();
    let mut offset = 0u64;
    for (i, s) in slots.iter().enumerate() {
        let start = offset;
        offset = offset.saturating_add(s.len.0);
        let joins = out.last().is_some_and(|last| {
            last.slots.end == i
                && slots[last.slots.start].group == s.group
                && (last.bytes.end - last.bytes.start).saturating_add(s.len.0) <= pin_step.0
        });
        match out.last_mut() {
            Some(last) if joins => {
                last.slots.end = i + 1;
                last.bytes.end = offset;
            }
            _ => out.push(LockRange {
                slots: i..i + 1,
                bytes: start..offset,
            }),
        }
    }
    out
}
