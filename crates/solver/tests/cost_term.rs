// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every term a `StageCost` can carry maps to exactly one of RFC-0053 §2.2's 18 `CostTerm`
//! variants, so the solver's prediction and a runner's measurement compare term by term
//! (RFC-0011 §2; PLC-E4-T01).

use std::collections::BTreeMap;

use smallvec::SmallVec;
use syntropy_core::{Nanos, StageIx};
use syntropy_obs::cost_term::CostTerm;
use syntropy_solver::{StageCost, StageTerm};

/// A stage that spends `code + 1` ns under each term, with every term also overlapped.
fn every_term() -> StageCost {
    let terms: SmallVec<[(StageTerm, Nanos); 18]> = StageTerm::ALL
        .iter()
        .map(|&term| (term, Nanos(u64::from(term.code()) + 1)))
        .collect();
    StageCost {
        stage: StageIx(0),
        overlapped: terms.iter().copied().collect(),
        terms,
        cpu_layer_max: Nanos(0),
        bytes: SmallVec::new(),
        resident: SmallVec::new(),
        prefill_per_chunk: Nanos(0),
    }
}

#[test]
fn every_stage_cost_term_maps_to_exactly_one_cost_term() {
    let stage = every_term();
    let mut by_cost_term = BTreeMap::new();
    for &(term, nanos) in stage.terms.iter().chain(&stage.overlapped) {
        let cost_term =
            CostTerm::from_stage_term(term).unwrap_or_else(|| panic!("{term:?} has no CostTerm"));
        // The same term, by name and discriminant, and back.
        assert_eq!(cost_term.as_u8(), term.code());
        assert_eq!(format!("{cost_term:?}"), format!("{term:?}"));
        assert_eq!(StageTerm::from(cost_term), term);
        // One term per CostTerm: no two terms share one.
        let previous = by_cost_term.insert(cost_term, term);
        assert!(
            previous.is_none() || previous == Some(term),
            "{term:?} shares {cost_term:?}"
        );
        assert_eq!(nanos, Nanos(u64::from(cost_term.as_u8()) + 1));
    }
    // All 18 CostTerms are reached, so a StageCost can carry every term a runner measures.
    assert_eq!(by_cost_term.len(), CostTerm::COUNT);
    assert_eq!(CostTerm::COUNT, 18);
    for cost_term in CostTerm::ALL {
        assert!(
            by_cost_term.contains_key(&cost_term),
            "{cost_term:?} unreached"
        );
    }
}

/// A `StageCost` keyed by the mirror reads back keyed by RFC-0053's `CostTerm`, term for term,
/// and its step time is the sum of its terms.
#[test]
fn a_stage_cost_reads_back_as_cost_terms() {
    let stage = every_term();
    let measured: Vec<(CostTerm, Nanos)> = stage
        .terms
        .iter()
        .map(|&(term, nanos)| (CostTerm::from_stage_term(term).unwrap(), nanos))
        .collect();
    let mirrored: Vec<(StageTerm, Nanos)> = measured
        .iter()
        .map(|&(term, nanos)| (term.stage_term(), nanos))
        .collect();
    assert_eq!(mirrored.as_slice(), stage.terms.as_slice());
    assert_eq!(stage.step(), Nanos((1..=18).sum()));
}
