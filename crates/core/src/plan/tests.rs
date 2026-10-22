// SPDX-License-Identifier: MIT OR Apache-2.0

//! P-core-1 (RFC-0060 §3.2): `PlanEpoch` order is total, and a higher `term` dominates any `seq`.
//!
//! A named property (QUAL-E3-T05): it runs on proptest's default configuration, so
//! `PROPTEST_CASES` sets its case count, 256 when unset, and a failing case is saved to
//! `crates/core/proptest-regressions/plan/tests.txt`. `ci/properties.toml` registers it.

use std::cmp::Ordering;

use proptest::prelude::*;

use super::PlanEpoch;

/// A term or sequence number: often one of a few small values, so that many pairs share a
/// term and the order has to fall through to `seq`, and otherwise anywhere in the range.
fn part() -> impl Strategy<Value = u64> {
    prop_oneof![0u64..4, Just(u64::MAX), any::<u64>()]
}

fn epoch() -> impl Strategy<Value = PlanEpoch> {
    (part(), part()).prop_map(|(term, seq)| PlanEpoch { term, seq })
}

/// The oracle: `term` first, then `seq` (RFC-0001 §5.4).
fn lexicographic(a: PlanEpoch, b: PlanEpoch) -> Ordering {
    match a.term.cmp(&b.term) {
        Ordering::Equal => a.seq.cmp(&b.seq),
        unequal => unequal,
    }
}

proptest! {
    /// P-core-1: the order is the lexicographic definition and a total order.
    #[test]
    fn p_core_1_plan_epoch_order_is_total_and_lexicographic(
        a in epoch(),
        b in epoch(),
        c in epoch(),
    ) {
        prop_assert_eq!(a.cmp(&b), lexicographic(a, b));
        prop_assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
        // Exactly one of <, == and > holds, and swapping the sides reverses the order.
        prop_assert_eq!([a < b, a == b, a > b].iter().filter(|&&holds| holds).count(), 1);
        prop_assert_eq!(b.cmp(&a), a.cmp(&b).reverse());
        prop_assert_eq!(a == b, a.term == b.term && a.seq == b.seq);
        if a <= b && b <= c {
            prop_assert!(a <= c, "{a:?} <= {b:?} <= {c:?}");
        }
    }

    /// P-core-1: a plan from a later term supersedes every plan of an earlier term, whatever
    /// the sequence numbers.
    #[test]
    fn p_core_1_a_higher_term_dominates_any_seq(
        term in 0u64..u64::MAX,
        later in 1u64..=u64::MAX,
        seq_old in any::<u64>(),
        seq_new in any::<u64>(),
    ) {
        let old = PlanEpoch { term, seq: seq_old };
        let new = PlanEpoch { term: term.saturating_add(later), seq: seq_new };
        prop_assert!(new > old, "{new:?} > {old:?}");
        prop_assert!(old < new);
    }
}

#[test]
fn the_highest_seq_of_a_term_is_below_the_lowest_of_the_next() {
    let last = PlanEpoch {
        term: 7,
        seq: u64::MAX,
    };
    let first = PlanEpoch { term: 8, seq: 0 };
    assert!(last < first);
    assert_eq!(PlanEpoch::default(), PlanEpoch { term: 0, seq: 0 });
}
