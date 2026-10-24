// SPDX-License-Identifier: MIT OR Apache-2.0

//! The MoE contract between the router op, the dispatcher, the CPU pools and the grouped
//! expert kernels (RFC-0040 §3.4; RFC-0023 §2, §5.4, §6; ADR-0404 items 3 to 5).
//!
//! - [`mirror`] holds the `#[repr(C)]` mirrors of the doorbell, [`DoorbellHeader`],
//!   [`RouteSlot`] and [`PredSlot`], generated from `include/syntropy/moe_doorbell.h`, their
//!   one definition, with the codes [`assign`] writes.
//! - [`doorbell`] sizes a doorbell layer for the epoch's `MAX_UNION` and `MAX_PRED`
//!   ([`DoorbellLayout`]) and writes and reads whole layers in its bytes ([`DoorbellLayer`]).
//! - [`assign`] is the split rule every site evaluates from the same doorbell. Its Rust version
//!   and its C version, `include/syntropy/moe_assign.h`, are printed from one specification,
//!   and the tests check that both are current and that they give identical output.
//! - [`tables`] holds the per-cache tables the kernels read (`slot_of`, `slot_addr`,
//!   `copy_eligible`), the dispatcher's `copy_src`, the step's balance block, the host mirror of
//!   `copy_eligible` that the host sites hold at exactly the step's `delta_head`, and the gate
//!   that keeps a copy source until the last step that saw it eligible has completed.
//!
//! `kernels` forbids `unsafe`, so a doorbell is a byte slice here; the provider crates map it
//! onto host-mapped memory.

pub mod doorbell;
// Generated: the tests compare it with the generator's output, so rustfmt leaves it alone.
#[rustfmt::skip]
pub mod mirror;
// Generated from `spec`, like `mirror`.
#[rustfmt::skip]
mod split;
pub mod tables;

use std::fmt;

pub use doorbell::{DoorbellError, DoorbellLayer, DoorbellLayout};
pub use mirror::{DoorbellHeader, PredSlot, RouteSlot};
pub use tables::{
    BalanceBlock, CopyEligible, CopySrc, EligibilityDelta, EligibilityMirror, EligibleBits,
    LayerSplit, RetireGate, SlotAddr, SlotOf, TableError,
};

use mirror::{MOE_CPU, MOE_HIT};

/// Where one routed expert of a layer is computed this step: the output of [`assign`] for one
/// route entry (RFC-0040 §3.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Assignment {
    /// Resident in the device's cache: the hit kernel computes it from its slot.
    Hit,
    /// Copied into copy slot `i` of the layer's half of the copy slots, then computed by the
    /// GPU's second expert kernel.
    Copy(u32),
    /// Computed in place by the CPU pools.
    Cpu,
}

impl Assignment {
    /// The code the generated versions write: `MOE_HIT`, `MOE_CPU`, or the copy-slot index.
    #[must_use]
    pub fn code(self) -> u32 {
        match self {
            Self::Hit => MOE_HIT,
            Self::Copy(i) => i,
            Self::Cpu => MOE_CPU,
        }
    }

    /// The assignment a code stands for. Every code below `MOE_CPU` is a copy-slot index.
    #[must_use]
    pub fn from_code(code: u32) -> Self {
        match code {
            MOE_HIT => Self::Hit,
            MOE_CPU => Self::Cpu,
            i => Self::Copy(i),
        }
    }
}

/// Why [`assign`] refused its inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssignError {
    /// More route entries than a doorbell layer holds (`n_route` is a `u16`).
    TooManyRoutes(usize),
    /// `out` has fewer entries than `route`.
    OutputTooShort { need: usize, have: usize },
    /// The eligibility bitmap has fewer words than its expert count needs.
    BitmapTooShort { need: usize, have: usize },
}

impl fmt::Display for AssignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyRoutes(n) => {
                write!(f, "{n} route entries; a doorbell holds at most 65535")
            }
            Self::OutputTooShort { need, have } => {
                write!(f, "output holds {have} codes for {need} route entries")
            }
            Self::BitmapTooShort { need, have } => {
                write!(
                    f,
                    "eligibility bitmap holds {have} words; its experts need {need}"
                )
            }
        }
    }
}

impl std::error::Error for AssignError {}

/// The split rule (RFC-0023 §6; RFC-0040 §3.4; ADR-0404 item 4): writes into `out[i]` the code
/// of where `route[i]` is computed this step, as [`Assignment::code`] gives it.
///
/// - [`Assignment::Hit`] for an entry whose `cache_slot` is not `MOE_MISS`.
/// - The misses are walked in order of `tokens` descending, then `expert` ascending, then
///   entry position (a well-formed doorbell never repeats an expert). A miss marked in
///   `eligible` at the step's table version becomes [`Assignment::Copy`] with the next copy-slot
///   index while that index is below `split.copy_slots` and the copied bytes, `split.expert_bytes`
///   per expert, stay at or below `split.b_copy`.
/// - [`Assignment::Cpu`] for every other miss, an expert outside `eligible` among them.
///
/// The function is pure and total over its inputs, so every site that evaluates it from the
/// same doorbell, bitmap version and balance block gets the same codes. Entries of `out` past
/// `route.len()` are left as they were.
///
/// # Errors
///
/// [`AssignError`] when `route` is longer than a doorbell layer can be, or `out` or the bitmap
/// is too short; `out` is unchanged then.
pub fn assign_into(
    route: &[RouteSlot],
    eligible: EligibleBits<'_>,
    split: LayerSplit,
    out: &mut [u32],
) -> Result<(), AssignError> {
    let n = route.len();
    if n > usize::from(u16::MAX) {
        return Err(AssignError::TooManyRoutes(n));
    }
    if out.len() < n {
        return Err(AssignError::OutputTooShort {
            need: n,
            have: out.len(),
        });
    }
    let need = eligible.experts().div_ceil(32) as usize;
    if eligible.words().len() < need {
        return Err(AssignError::BitmapTooShort {
            need,
            have: eligible.words().len(),
        });
    }
    split::assign_codes(
        route,
        n as u32,
        eligible.words(),
        eligible.experts(),
        split.b_copy,
        split.expert_bytes,
        split.copy_slots,
        out,
    );
    Ok(())
}

/// [`assign_into`], returning one [`Assignment`] per route entry.
///
/// # Errors
///
/// As [`assign_into`].
pub fn assign(
    route: &[RouteSlot],
    eligible: EligibleBits<'_>,
    split: LayerSplit,
) -> Result<Vec<Assignment>, AssignError> {
    let mut out = vec![0; route.len()];
    assign_into(route, eligible, split, &mut out)?;
    Ok(out.into_iter().map(Assignment::from_code).collect())
}

#[cfg(test)]
mod spec;
#[cfg(test)]
mod tests;
