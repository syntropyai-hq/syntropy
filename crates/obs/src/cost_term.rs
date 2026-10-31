// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cost terms that prediction and measurement share.
//!
//! RFC-0053 §2.2 defines [`CostTerm`] and RFC-0001 §5.13 places it in `obs`. RFC-0011's
//! `StageCost` keys its predicted terms by it, and RFC-0053 §2.5's `StepRecord` keys its measured
//! terms by it, so the solver's prediction and a runner's measurement compare term by term.

use serde::{Deserialize, Serialize};
use syntropy_core::plan::StageTerm;

/// One term of a stage step's time: the key space for prediction (RFC-0011 `PredictedCost`) and
/// measurement (RFC-0053 `StepRecord`).
///
/// Owner: RFC-0053 (keys), RFC-0011 (how each term is predicted), RFC-0012 (where each term is
/// measured).
///
/// **Discriminants.** The enum is `repr(u8)`. Variants are declared in the order RFC-0053 §2.2
/// lists them, and each discriminant is its position in that list, counting from 0. `StepRecord`
/// stores the discriminant as a byte in `term_ids`, so a discriminant never changes once
/// released. [`CostTerm::ALL`] holds the variants in discriminant order.
///
/// **Serde.** A term serializes as its variant name, such as `"DenseVram"`.
#[repr(u8)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CostTerm {
    /// Dense layer groups computed by a GPU from the `Vram` tier.
    DenseVram = 0,
    /// Dense layer groups computed by an integrated GPU from the `Unified` tier.
    DenseUnified = 1,
    /// Dense layer groups computed by the CPU from RAM.
    DenseHostCpu = 2,
    /// Expert-cache hits computed on a GPU, from a `Vram` or `Unified` slot.
    ExpertHitGpu = 3,
    /// Experts homed in RAM and computed in place by the CPU, including the in-place share of
    /// misses.
    ExpertHitCpu = 4,
    /// Missed experts copied over PCIe: the copied share of the `MissSplit`.
    ExpertMissCopied = 5,
    /// Time a step waits on NVMe demand reads.
    NvmeDemandWait = 6,
    /// KV reads from every tier. `StepRecord.bytes` carries the bytes per tier.
    KvRead = 7,
    /// Row lookups from large tables.
    RowLookup = 8,
    /// Device hand-offs inside a stage: `t_switch` times the number of hand-offs.
    Switch = 9,
    /// Cross-node or intra-node tensor-parallel and expert-parallel collectives (RFC-0032).
    Collective = 10,
    /// Serialising and sending the outbound frame.
    HopSend = 11,
    /// Waiting for the inbound frame beyond the previous stage's compute.
    HopWait = 12,
    /// Stalls for lack of send credit (RFC-0001 §8.4 rule 1).
    CreditStall = 13,
    /// `t_fixed`: launch, synchronisation and framework overhead.
    Fixed = 14,
    /// Sampling at the ring's tail.
    Sampler = 15,
    /// Applying a constraint mask before sampling.
    ConstraintMask = 16,
    /// Drafting tokens for speculative decoding.
    Draft = 17,
}

impl CostTerm {
    /// The number of variants. RFC-0053 §2.2 fixes it at 18, and its `TermVector` holds one
    /// entry per variant.
    pub const COUNT: usize = 18;

    /// Every variant, in discriminant order: `ALL[i] as u8 == i`.
    pub const ALL: [CostTerm; CostTerm::COUNT] = [
        CostTerm::DenseVram,
        CostTerm::DenseUnified,
        CostTerm::DenseHostCpu,
        CostTerm::ExpertHitGpu,
        CostTerm::ExpertHitCpu,
        CostTerm::ExpertMissCopied,
        CostTerm::NvmeDemandWait,
        CostTerm::KvRead,
        CostTerm::RowLookup,
        CostTerm::Switch,
        CostTerm::Collective,
        CostTerm::HopSend,
        CostTerm::HopWait,
        CostTerm::CreditStall,
        CostTerm::Fixed,
        CostTerm::Sampler,
        CostTerm::ConstraintMask,
        CostTerm::Draft,
    ];

    /// The stable one-byte discriminant, as `StepRecord.term_ids` stores it.
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// The term with discriminant `v`, or `None` when no variant has it.
    pub const fn from_u8(v: u8) -> Option<CostTerm> {
        if (v as usize) < CostTerm::COUNT {
            Some(CostTerm::ALL[v as usize])
        } else {
            None
        }
    }

    /// The variant's position in [`CostTerm::ALL`]. The match lists every variant with no
    /// wildcard, so adding a variant fails to compile here until `COUNT` and `ALL` change too.
    const fn index(self) -> usize {
        match self {
            CostTerm::DenseVram => 0,
            CostTerm::DenseUnified => 1,
            CostTerm::DenseHostCpu => 2,
            CostTerm::ExpertHitGpu => 3,
            CostTerm::ExpertHitCpu => 4,
            CostTerm::ExpertMissCopied => 5,
            CostTerm::NvmeDemandWait => 6,
            CostTerm::KvRead => 7,
            CostTerm::RowLookup => 8,
            CostTerm::Switch => 9,
            CostTerm::Collective => 10,
            CostTerm::HopSend => 11,
            CostTerm::HopWait => 12,
            CostTerm::CreditStall => 13,
            CostTerm::Fixed => 14,
            CostTerm::Sampler => 15,
            CostTerm::ConstraintMask => 16,
            CostTerm::Draft => 17,
        }
    }
}

impl CostTerm {
    /// The term in `core`'s mirror, which plans carry in `StageCost` (RFC-0011 §2, PLC-E4-T01):
    /// the variant of the same name and discriminant. The match is exhaustive, so a variant added
    /// here fails to compile until the mirror has it too.
    #[must_use]
    pub const fn stage_term(self) -> StageTerm {
        match self {
            CostTerm::DenseVram => StageTerm::DenseVram,
            CostTerm::DenseUnified => StageTerm::DenseUnified,
            CostTerm::DenseHostCpu => StageTerm::DenseHostCpu,
            CostTerm::ExpertHitGpu => StageTerm::ExpertHitGpu,
            CostTerm::ExpertHitCpu => StageTerm::ExpertHitCpu,
            CostTerm::ExpertMissCopied => StageTerm::ExpertMissCopied,
            CostTerm::NvmeDemandWait => StageTerm::NvmeDemandWait,
            CostTerm::KvRead => StageTerm::KvRead,
            CostTerm::RowLookup => StageTerm::RowLookup,
            CostTerm::Switch => StageTerm::Switch,
            CostTerm::Collective => StageTerm::Collective,
            CostTerm::HopSend => StageTerm::HopSend,
            CostTerm::HopWait => StageTerm::HopWait,
            CostTerm::CreditStall => StageTerm::CreditStall,
            CostTerm::Fixed => StageTerm::Fixed,
            CostTerm::Sampler => StageTerm::Sampler,
            CostTerm::ConstraintMask => StageTerm::ConstraintMask,
            CostTerm::Draft => StageTerm::Draft,
        }
    }

    /// The term a plan's [`StageTerm`] names: the one with its discriminant, or `None` for a
    /// mirror variant this build does not know.
    #[must_use]
    pub fn from_stage_term(term: StageTerm) -> Option<CostTerm> {
        CostTerm::from_u8(term.code())
    }
}

impl From<CostTerm> for StageTerm {
    fn from(term: CostTerm) -> Self {
        term.stage_term()
    }
}

// Compile-time checks (PROD-E8-T12): the enum is one byte and has exactly 18 variants, and `ALL`
// lists them in discriminant order 0..18. With the exhaustive match in `index`, a 19th variant
// cannot be added without one of these failing.
const _: () = {
    assert!(core::mem::size_of::<CostTerm>() == 1);
    assert!(CostTerm::COUNT == 18);
    let mut i = 0;
    while i < CostTerm::COUNT {
        assert!(CostTerm::ALL[i] as usize == i);
        assert!(CostTerm::ALL[i].index() == i);
        i += 1;
    }
};

#[cfg(test)]
mod tests {
    use super::CostTerm;

    /// The golden list: each variant's name and discriminant, in RFC-0053 §2.2's order. A change
    /// here is a change to the `StepRecord` encoding.
    const GOLDEN: [(&str, u8); 18] = [
        ("DenseVram", 0),
        ("DenseUnified", 1),
        ("DenseHostCpu", 2),
        ("ExpertHitGpu", 3),
        ("ExpertHitCpu", 4),
        ("ExpertMissCopied", 5),
        ("NvmeDemandWait", 6),
        ("KvRead", 7),
        ("RowLookup", 8),
        ("Switch", 9),
        ("Collective", 10),
        ("HopSend", 11),
        ("HopWait", 12),
        ("CreditStall", 13),
        ("Fixed", 14),
        ("Sampler", 15),
        ("ConstraintMask", 16),
        ("Draft", 17),
    ];

    #[test]
    fn discriminants_match_golden_list() {
        assert_eq!(CostTerm::ALL.len(), GOLDEN.len());
        for (term, (name, disc)) in CostTerm::ALL.iter().zip(GOLDEN) {
            assert_eq!(format!("{term:?}"), name);
            assert_eq!(term.as_u8(), disc, "{name}");
            assert_eq!(*term as u8, disc, "{name}");
        }
    }

    #[test]
    fn every_variant_round_trips_through_serde() {
        for (term, (name, _)) in CostTerm::ALL.iter().zip(GOLDEN) {
            let json = serde_json::to_string(term).unwrap();
            assert_eq!(json, format!("\"{name}\""));
            let back: CostTerm = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *term);
        }
    }

    /// Every `CostTerm` maps to exactly one `StageTerm`, of the same name and discriminant, and
    /// back; the mirror has no variant the key space lacks (RFC-0011 §2, PLC-E4-T01).
    #[test]
    fn every_term_maps_to_exactly_one_stage_term() {
        use syntropy_core::plan::StageTerm;
        assert_eq!(StageTerm::ALL.len(), CostTerm::COUNT);
        let mut seen = std::collections::BTreeSet::new();
        for term in CostTerm::ALL {
            let mirror = term.stage_term();
            assert_eq!(mirror.code(), term.as_u8());
            assert_eq!(format!("{mirror:?}"), format!("{term:?}"));
            assert_eq!(CostTerm::from_stage_term(mirror), Some(term));
            assert!(seen.insert(mirror), "{term:?} shares its mirror");
        }
        for mirror in StageTerm::ALL {
            assert!(seen.contains(&mirror), "{mirror:?} has no CostTerm");
        }
    }

    #[test]
    fn from_u8_inverts_as_u8() {
        for term in CostTerm::ALL {
            assert_eq!(CostTerm::from_u8(term.as_u8()), Some(term));
        }
        for v in CostTerm::COUNT as u8..=u8::MAX {
            assert_eq!(CostTerm::from_u8(v), None);
        }
    }

    #[test]
    fn unknown_name_is_rejected() {
        assert!(serde_json::from_str::<CostTerm>("\"Hop\"").is_err());
    }
}
