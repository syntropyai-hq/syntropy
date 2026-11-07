// SPDX-License-Identifier: MIT OR Apache-2.0

use proptest::prelude::*;
use syntropy_core::ErrorClass;
use syntropy_core::runner::DegradeFlags;

use super::*;
use crate::error::unsupported;

#[test]
fn a_step_of_capture_safe_ops_is_one_segment() {
    assert_eq!(segments([true; 5]), [Segment::Captured(0..5)]);
    assert_eq!(segments([]), []);
}

#[test]
fn an_unsafe_op_breaks_the_capture_around_itself() {
    use Segment::{Captured, Direct};
    assert_eq!(
        segments([true, true, false, true]),
        [Captured(0..2), Direct(2), Captured(3..4)]
    );
    assert_eq!(segments([false, true]), [Direct(0), Captured(1..2)]);
    assert_eq!(segments([true, false]), [Captured(0..1), Direct(1)]);
    assert_eq!(segments([false, false]), [Direct(0), Direct(1)]);
}

proptest! {
    /// The parts cover every op once, in order; a direct part is exactly an unsafe op; no two
    /// captured parts touch; and the submissions are one per part, so each break costs one
    /// more than the unbroken step (RFC-0040 §4.4 rule 5).
    #[test]
    fn segments_cover_the_step_in_order(flags in proptest::collection::vec(any::<bool>(), 0..40)) {
        let parts = segments(flags.iter().copied());
        let mut next = 0;
        let mut last_captured = false;
        for p in &parts {
            match p {
                Segment::Captured(r) => {
                    prop_assert_eq!(r.start, next);
                    prop_assert!(r.end > r.start);
                    prop_assert!(flags[r.clone()].iter().all(|&f| f));
                    prop_assert!(!last_captured);
                    next = r.end;
                    last_captured = true;
                }
                Segment::Direct(i) => {
                    prop_assert_eq!(*i, next);
                    prop_assert!(!flags[*i]);
                    next = i + 1;
                    last_captured = false;
                }
            }
        }
        prop_assert_eq!(next, flags.len());
        let unsafe_ops = flags.iter().filter(|&&f| !f).count();
        prop_assert!(parts.len() <= 2 * unsafe_ops + 1);
    }
}

#[test]
fn only_epoch_health_buckets_and_scratch_invalidate() {
    use GraphChange::*;
    for c in [NewEpoch, HealthChanged, BucketSet, ScratchSize] {
        assert!(c.invalidates(), "{c:?}");
    }
    for c in [
        ExpertCacheContents,
        SlotTableWords,
        KvPageContents,
        StepDescriptor,
        ExpertCacheTailShrink,
    ] {
        assert!(!c.invalidates(), "{c:?}");
    }
}

#[test]
fn capture_failed_degrades_to_capture_off_and_nothing_else_does() {
    let e = capture_failed("instantiate failed");
    assert_eq!(
        (e.code, e.class),
        (codes::CAPTURE_FAILED, ErrorClass::Unsupported)
    );
    assert_eq!(capture_degrade(&e), Some(DegradeFlags::CAPTURE_OFF));
    // A provider without capture is not degraded: the op list is its normal path.
    assert_eq!(capture_degrade(&unsupported()), None);
    let sticky = Error::new(ErrorClass::Device, codes::DEVICE_STICKY, "sticky");
    assert_eq!(capture_degrade(&sticky), None);
}

#[test]
fn the_instance_count_is_rfc_0040s() {
    // RFC-0012 §2's latency-mode stage: 7 buckets × 2 parities, 2 profiled variants.
    assert_eq!(instances(7, 2, false), 16);
    assert_eq!(instances(7, 2, true), 17);
    assert_eq!(instances(0, 0, false), 0);
}

fn key(bucket: ShapeClass, parity: Parity) -> GraphKey {
    GraphKey {
        bucket,
        parity,
        variant: GraphVariant::Plain,
    }
}

#[test]
fn a_staged_graph_goes_live_only_between_steps() {
    let k = key(ShapeClass::Decode, Parity::Even);
    let mut set = GraphSet::new();
    set.stage(k, 1);
    assert_eq!(set.get(&k), None, "live before the swap");
    assert!(set.has_staged());
    assert_eq!(set.between_steps(), Vec::<i32>::new());
    assert_eq!(set.get(&k), Some(&1));

    // A re-instantiated graph replaces the old one at the next swap; the old one comes back.
    set.stage(k, 2);
    assert_eq!(set.get(&k), Some(&1));
    assert_eq!(set.between_steps(), [1]);
    assert_eq!(set.get(&k), Some(&2));
    assert!(!set.has_staged());
}

#[test]
fn the_later_of_two_staged_graphs_wins() {
    let k = key(ShapeClass::Verify { window: 4 }, Parity::Odd);
    let mut set = GraphSet::new();
    set.stage(k, 1);
    set.stage(k, 2);
    assert_eq!(set.between_steps(), [1]);
    assert_eq!(set.get(&k), Some(&2));
}

#[test]
fn an_invalidating_change_drops_live_and_staged_graphs() {
    let mut set = GraphSet::new();
    for (i, p) in Parity::BOTH.into_iter().enumerate() {
        set.stage(key(ShapeClass::Decode, p), i);
    }
    set.between_steps();
    set.stage(key(ShapeClass::Batch { rows: 2 }, Parity::Even), 9);
    assert!(set.apply(GraphChange::KvPageContents).is_empty());
    assert_eq!(set.len(), 2);
    let mut dropped = set.apply(GraphChange::NewEpoch);
    dropped.sort_unstable();
    assert_eq!(dropped, [0, 1, 9]);
    assert!(set.is_empty());
    assert!(!set.has_staged());
}
