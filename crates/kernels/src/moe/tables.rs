// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tables of one device cache that the MoE step reads (RFC-0040 §3.4; RFC-0023 §5.4), the
//! step's balance block, and the host sites' view of `copy_eligible` at the step's table
//! version (ADR-0404 item 5).
//!
//! The device arrays are flat and row-major: entry `(layer, expert)` of `slot_of` is at index
//! `layer · experts + expert`, which is the `index` a table delta names, and layer `l`'s
//! `copy_eligible` bitmap is `⌈experts / 32⌉` little-endian `u32` words, expert `e` at bit
//! `e % 32` of word `e / 32`. The table-delta ring's entry layout is TIER-E8-T14's; here a
//! delta is the change it carries and the ring position it was posted at.

use std::collections::VecDeque;
use std::fmt;

use super::mirror::{MOE_MISS, RouteSlot};
use crate::step::StepDescriptor;

/// Why a table operation was refused. The table is unchanged on error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableError {
    /// A layer, expert or slot outside the table.
    OutOfRange {
        what: &'static str,
        index: u64,
        len: u64,
    },
    /// A table larger than its `u32` indices can address.
    TooLarge,
    /// A delta posted at a ring position at or below one already posted, or below the
    /// mirror's version.
    PositionOutOfOrder { pos: u64, after: u64 },
    /// A step's `delta_head` below the mirror's version: the mirror never goes back.
    DeltaHeadRegressed { version: u64, delta_head: u64 },
    /// The mirror is not at the step's `delta_head`, so evaluating it would use another
    /// table version than the device's.
    VersionSkew { version: u64, delta_head: u64 },
    /// A step launched out of order: its number not above the last one's, or its
    /// `delta_head` below the last one's.
    StepOutOfOrder { step: u64, delta_head: u64 },
    /// A completion for a step the gate never saw launched, or saw complete already.
    UnknownStep(u64),
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::OutOfRange { what, index, len } => {
                write!(f, "{what} {index} is outside a table of {len}")
            }
            Self::TooLarge => write!(f, "table larger than u32 indices address"),
            Self::PositionOutOfOrder { pos, after } => {
                write!(
                    f,
                    "delta at ring position {pos} posted after position {after}"
                )
            }
            Self::DeltaHeadRegressed {
                version,
                delta_head,
            } => write!(
                f,
                "delta_head {delta_head} is below the mirror's version {version}"
            ),
            Self::VersionSkew {
                version,
                delta_head,
            } => write!(
                f,
                "mirror at version {version}, the step at delta_head {delta_head}"
            ),
            Self::StepOutOfOrder { step, delta_head } => {
                write!(
                    f,
                    "step {step} at delta_head {delta_head} launched out of order"
                )
            }
            Self::UnknownStep(s) => write!(f, "step {s} is not a launched, incomplete step"),
        }
    }
}

impl std::error::Error for TableError {}

fn check(what: &'static str, index: u32, len: u32) -> Result<(), TableError> {
    if index < len {
        Ok(())
    } else {
        Err(TableError::OutOfRange {
            what,
            index: u64::from(index),
            len: u64::from(len),
        })
    }
}

/// `slot_of[layer][expert] → u32`: an expert's slot in the device's cache, or `MOE_MISS`
/// (RFC-0040 §3.4). Changed only by table deltas.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SlotOf {
    layers: u32,
    experts: u32,
    slots: Vec<u32>,
}

impl SlotOf {
    /// A table of `layers × experts` entries, every one `MOE_MISS`.
    ///
    /// # Errors
    ///
    /// [`TableError::TooLarge`] when an entry's index does not fit in a `u32`.
    pub fn new(layers: u32, experts: u32) -> Result<Self, TableError> {
        let len = u64::from(layers) * u64::from(experts);
        u32::try_from(len).map_err(|_| TableError::TooLarge)?;
        Ok(Self {
            layers,
            experts,
            slots: vec![MOE_MISS; len as usize],
        })
    }

    /// The flat index of `(layer, expert)`: the `index` of its table delta.
    ///
    /// # Errors
    ///
    /// [`TableError::OutOfRange`].
    pub fn index(&self, layer: u32, expert: u32) -> Result<u32, TableError> {
        check("layer", layer, self.layers)?;
        check("expert", expert, self.experts)?;
        Ok(layer * self.experts + expert)
    }

    /// The expert's slot, or `None` when it is a miss.
    ///
    /// # Errors
    ///
    /// [`TableError::OutOfRange`].
    pub fn get(&self, layer: u32, expert: u32) -> Result<Option<u32>, TableError> {
        let slot = self.slots[self.index(layer, expert)? as usize];
        Ok((slot != MOE_MISS).then_some(slot))
    }

    /// Sets the expert's slot, or marks it a miss with `None`.
    ///
    /// # Errors
    ///
    /// [`TableError::OutOfRange`], also for `Some(MOE_MISS)`, which is not a slot.
    pub fn set(&mut self, layer: u32, expert: u32, slot: Option<u32>) -> Result<(), TableError> {
        let i = self.index(layer, expert)? as usize;
        let slot = match slot {
            Some(s) => {
                check("slot", s, MOE_MISS)?;
                s
            }
            None => MOE_MISS,
        };
        self.slots[i] = slot;
        Ok(())
    }

    /// The route entry the router writes for an expert `tokens` selected.
    ///
    /// # Errors
    ///
    /// [`TableError::OutOfRange`].
    pub fn route_slot(
        &self,
        layer: u32,
        expert: u16,
        tokens: u16,
    ) -> Result<RouteSlot, TableError> {
        let cache_slot = self.get(layer, u32::from(expert))?.unwrap_or(MOE_MISS);
        Ok(RouteSlot {
            expert,
            tokens,
            cache_slot,
        })
    }

    /// The device array.
    #[must_use]
    pub fn as_words(&self) -> &[u32] {
        &self.slots
    }
}

/// `slot_addr[slot] → u64`: each cache slot's device address, fixed for the plan epoch
/// (RFC-0040 §3.4; RFC-0020 §8).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SlotAddr(Vec<u64>);

impl SlotAddr {
    /// The epoch's slot addresses, slot `i` at `addrs[i]`.
    ///
    /// # Errors
    ///
    /// [`TableError::TooLarge`] when slot indices would reach `MOE_MISS`.
    pub fn new(addrs: Vec<u64>) -> Result<Self, TableError> {
        if addrs.len() as u64 >= u64::from(MOE_MISS) {
            return Err(TableError::TooLarge);
        }
        Ok(Self(addrs))
    }

    /// The slot's device address.
    #[must_use]
    pub fn addr(&self, slot: u32) -> Option<u64> {
        self.0.get(slot as usize).copied()
    }

    /// The device array.
    #[must_use]
    pub fn as_words(&self) -> &[u64] {
        &self.0
    }
}

/// `copy_src[layer][expert] → u64`: the host address of a missed expert's DMA-registered
/// copy, for the dispatcher's copies (RFC-0040 §3.4). Host only; 0 means none.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CopySrc {
    layers: u32,
    experts: u32,
    addrs: Vec<u64>,
}

impl CopySrc {
    /// A table of `layers × experts` entries, every one without a source.
    ///
    /// # Errors
    ///
    /// [`TableError::TooLarge`] when an entry's index does not fit in a `u32`.
    pub fn new(layers: u32, experts: u32) -> Result<Self, TableError> {
        let len = u64::from(layers) * u64::from(experts);
        u32::try_from(len).map_err(|_| TableError::TooLarge)?;
        Ok(Self {
            layers,
            experts,
            addrs: vec![0; len as usize],
        })
    }

    fn index(&self, layer: u32, expert: u32) -> Result<usize, TableError> {
        check("layer", layer, self.layers)?;
        check("expert", expert, self.experts)?;
        Ok((layer * self.experts + expert) as usize)
    }

    /// The expert's copy source.
    ///
    /// # Errors
    ///
    /// [`TableError::OutOfRange`].
    pub fn get(&self, layer: u32, expert: u32) -> Result<Option<u64>, TableError> {
        let a = self.addrs[self.index(layer, expert)?];
        Ok((a != 0).then_some(a))
    }

    /// Sets the expert's copy source, or clears it with `None` (or `Some(0)`).
    ///
    /// # Errors
    ///
    /// [`TableError::OutOfRange`].
    pub fn set(&mut self, layer: u32, expert: u32, addr: Option<u64>) -> Result<(), TableError> {
        let i = self.index(layer, expert)?;
        self.addrs[i] = addr.unwrap_or(0);
        Ok(())
    }
}

/// One layer's `copy_eligible` bitmap, as [`assign`](super::assign) reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EligibleBits<'a> {
    words: &'a [u32],
    experts: u32,
}

impl<'a> EligibleBits<'a> {
    /// `experts` bits in `words`, expert `e` at bit `e % 32` of word `e / 32`. Experts at or
    /// above `experts` are never eligible.
    #[must_use]
    pub fn new(words: &'a [u32], experts: u32) -> Self {
        Self { words, experts }
    }

    /// No eligible expert: a `Unified` device or a CPU-only stage (RFC-0040 §3.4).
    #[must_use]
    pub fn none() -> Self {
        Self::new(&[], 0)
    }

    /// The bitmap's words.
    #[must_use]
    pub fn words(&self) -> &'a [u32] {
        self.words
    }

    /// Experts the bitmap covers.
    #[must_use]
    pub fn experts(&self) -> u32 {
        self.experts
    }

    /// Whether `expert` is marked.
    #[must_use]
    pub fn get(&self, expert: u32) -> bool {
        expert < self.experts
            && self
                .words
                .get((expert / 32) as usize)
                .is_some_and(|w| (w >> (expert % 32)) & 1 == 1)
    }
}

/// `copy_eligible[layer]`: per layer, the experts whose current host copy is `DmaPinned` and
/// `Resident`, so the copy engine may fetch them (RFC-0040 §3.4; RFC-0020 §6). Changed only
/// by table deltas.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CopyEligible {
    layers: u32,
    experts: u32,
    words: Vec<u32>,
}

/// One change to `copy_eligible`, as a table delta carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EligibilityDelta {
    pub layer: u32,
    pub expert: u32,
    pub eligible: bool,
}

impl CopyEligible {
    /// A bitmap of `layers × experts` bits, none set.
    ///
    /// # Errors
    ///
    /// [`TableError::TooLarge`] when the words do not fit in `u32` indices.
    pub fn new(layers: u32, experts: u32) -> Result<Self, TableError> {
        let len = u64::from(layers) * u64::from(experts.div_ceil(32));
        u32::try_from(len).map_err(|_| TableError::TooLarge)?;
        Ok(Self {
            layers,
            experts,
            words: vec![0; len as usize],
        })
    }

    fn words_per_layer(&self) -> usize {
        self.experts.div_ceil(32) as usize
    }

    /// Layer `layer`'s bitmap.
    ///
    /// # Errors
    ///
    /// [`TableError::OutOfRange`].
    pub fn layer(&self, layer: u32) -> Result<EligibleBits<'_>, TableError> {
        check("layer", layer, self.layers)?;
        let w = self.words_per_layer();
        let start = layer as usize * w;
        Ok(EligibleBits::new(
            &self.words[start..start + w],
            self.experts,
        ))
    }

    /// Whether `(layer, expert)` is marked.
    ///
    /// # Errors
    ///
    /// [`TableError::OutOfRange`].
    pub fn get(&self, layer: u32, expert: u32) -> Result<bool, TableError> {
        check("expert", expert, self.experts)?;
        Ok(self.layer(layer)?.get(expert))
    }

    /// Applies one delta.
    ///
    /// # Errors
    ///
    /// [`TableError::OutOfRange`].
    pub fn apply(&mut self, d: EligibilityDelta) -> Result<(), TableError> {
        check("layer", d.layer, self.layers)?;
        check("expert", d.expert, self.experts)?;
        let word = d.layer as usize * self.words_per_layer() + (d.expert / 32) as usize;
        let bit = 1u32 << (d.expert % 32);
        if d.eligible {
            self.words[word] |= bit;
        } else {
            self.words[word] &= !bit;
        }
        Ok(())
    }

    /// The device array: every layer's words, layer 0 first.
    #[must_use]
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }
}

/// The host sites' mirror of `copy_eligible`, held at exactly one table version (RFC-0023
/// §5.4; RFC-0040 §3.4; ADR-0404 item 5).
///
/// The dispatcher posts each eligibility delta here at the ring position it posts it to the
/// device. A posted delta waits: [`advance_to`](Self::advance_to) applies, in order, exactly the
/// deltas at positions below a step's `delta_head`, as the device's `ApplyTableDeltas` does,
/// and [`at`](Self::at) refuses any other version. So a mirror that has seen deltas past the
/// step's version (ahead) or not yet been advanced to it (behind) still evaluates the step at
/// the device's version, or not at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EligibilityMirror {
    table: CopyEligible,
    version: u64,
    pending: VecDeque<(u64, EligibilityDelta)>,
}

impl EligibilityMirror {
    /// A mirror of `table` at version `version`: every delta below `version` is in `table`.
    #[must_use]
    pub fn new(table: CopyEligible, version: u64) -> Self {
        Self {
            table,
            version,
            pending: VecDeque::new(),
        }
    }

    /// The table version the mirror holds: the `delta_head` it was last advanced to.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Records a delta the dispatcher posted at ring position `pos`.
    ///
    /// # Errors
    ///
    /// [`TableError::PositionOutOfOrder`] when `pos` is below the version or not above the last
    /// posted position, and [`TableError::OutOfRange`].
    pub fn post(&mut self, pos: u64, d: EligibilityDelta) -> Result<(), TableError> {
        let after = self.pending.back().map(|&(p, _)| p);
        if pos < self.version || after.is_some_and(|a| pos <= a) {
            return Err(TableError::PositionOutOfOrder {
                pos,
                after: after.unwrap_or(self.version),
            });
        }
        check("layer", d.layer, self.table.layers)?;
        check("expert", d.expert, self.table.experts)?;
        self.pending.push_back((pos, d));
        Ok(())
    }

    /// Applies every posted delta below `delta_head`, in order, and holds the mirror there.
    ///
    /// # Errors
    ///
    /// [`TableError::DeltaHeadRegressed`] when `delta_head` is below the version.
    pub fn advance_to(&mut self, delta_head: u64) -> Result<&CopyEligible, TableError> {
        if delta_head < self.version {
            return Err(TableError::DeltaHeadRegressed {
                version: self.version,
                delta_head,
            });
        }
        while let Some(&(pos, d)) = self.pending.front() {
            if pos >= delta_head {
                break;
            }
            self.table.apply(d)?;
            self.pending.pop_front();
        }
        self.version = delta_head;
        Ok(&self.table)
    }

    /// The table at `delta_head`, which must be the mirror's version.
    ///
    /// # Errors
    ///
    /// [`TableError::VersionSkew`] otherwise.
    pub fn at(&self, delta_head: u64) -> Result<&CopyEligible, TableError> {
        if delta_head != self.version {
            return Err(TableError::VersionSkew {
                version: self.version,
                delta_head,
            });
        }
        Ok(&self.table)
    }
}

/// Keeps a copy source until the last step that saw it eligible has completed (RFC-0023
/// §5.4; RFC-0040 §3.4; RFC-0020 §7.4).
///
/// Step `s` saw every delta below its `delta_head`. So when the delta that clears an expert's
/// eligibility sits at ring position `p`, the steps that may have seen the old source eligible
/// are the ones with `delta_head ≤ p`. Its source may be retired once a launched step has a
/// `delta_head` above `p`, so no later step can see it (heads never fall), and every step with
/// `delta_head ≤ p` has completed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RetireGate {
    /// Launched steps not yet known complete, with all earlier ones, as (step, `delta_head`,
    /// completed), in launch order.
    inflight: VecDeque<(u64, u64, bool)>,
    last: Option<(u64, u64)>,
}

impl RetireGate {
    /// An empty gate.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the launch of `step` at `delta_head`.
    ///
    /// # Errors
    ///
    /// [`TableError::StepOutOfOrder`] when `step` is not above the last launched step, or
    /// `delta_head` is below its head.
    pub fn launch(&mut self, step: u64, delta_head: u64) -> Result<(), TableError> {
        if self.last.is_some_and(|(s, h)| step <= s || delta_head < h) {
            return Err(TableError::StepOutOfOrder { step, delta_head });
        }
        self.last = Some((step, delta_head));
        self.inflight.push_back((step, delta_head, false));
        Ok(())
    }

    /// Records that `step`'s fence has passed.
    ///
    /// # Errors
    ///
    /// [`TableError::UnknownStep`] for a step not launched, or completed already.
    pub fn complete(&mut self, step: u64) -> Result<(), TableError> {
        let entry = self
            .inflight
            .iter_mut()
            .find(|e| e.0 == step && !e.2)
            .ok_or(TableError::UnknownStep(step))?;
        entry.2 = true;
        while self.inflight.front().is_some_and(|e| e.2) {
            self.inflight.pop_front();
        }
        Ok(())
    }

    /// The step whose fence the retirement of a source cleared at ring position `clear_pos`
    /// waits on: the last launched step with `delta_head ≤ clear_pos` that is not known
    /// complete with every step before it. `None` when there is none.
    #[must_use]
    pub fn waits_on(&self, clear_pos: u64) -> Option<u64> {
        self.inflight
            .iter()
            .rev()
            .find(|e| e.1 <= clear_pos && !e.2)
            .map(|e| e.0)
    }

    /// Whether a source whose clearing delta sits at `clear_pos` may be retired now.
    #[must_use]
    pub fn may_retire(&self, clear_pos: u64) -> bool {
        self.last.is_some_and(|(_, h)| h > clear_pos)
            && self
                .inflight
                .iter()
                .all(|&(_, h, done)| done || h > clear_pos)
    }
}

/// The step's balance block: per MoE layer of the stage, `B_copy(l)` in bytes (RFC-0040 §3.4,
/// §3.5; RFC-0023 §8), the descriptor's `moe_balance` array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BalanceBlock<'a>(&'a [u32]);

impl<'a> BalanceBlock<'a> {
    /// The balance block of entries `b_copy`, layer 0 first.
    #[must_use]
    pub fn new(b_copy: &'a [u32]) -> Self {
        Self(b_copy)
    }

    /// The balance block of a step descriptor.
    #[must_use]
    pub fn of(step: &'a StepDescriptor) -> Self {
        Self(&step.moe_balance)
    }

    /// `B_copy(l)` of the stage's MoE layer `layer`.
    #[must_use]
    pub fn b_copy(&self, layer: usize) -> Option<u32> {
        self.0.get(layer).copied()
    }

    /// The split inputs of MoE layer `layer`, with the epoch's expert size `s_l` and the
    /// layer's copy-slot count.
    #[must_use]
    pub fn split(&self, layer: usize, expert_bytes: u64, copy_slots: u32) -> Option<LayerSplit> {
        Some(LayerSplit {
            b_copy: self.b_copy(layer)?,
            expert_bytes,
            copy_slots,
        })
    }
}

/// The per-layer inputs of [`assign`](super::assign) besides the doorbell and the bitmap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LayerSplit {
    /// `B_copy(l)`: the copied-miss byte budget for this step, from the balance block.
    pub b_copy: u32,
    /// `s_l`: one expert's bytes, uniform within the layer.
    pub expert_bytes: u64,
    /// Copy slots in the layer's half of the double-buffered copy slots (RFC-0023 §4, §6).
    pub copy_slots: u32,
}
