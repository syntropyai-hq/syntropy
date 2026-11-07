// SPDX-License-Identifier: MIT OR Apache-2.0

//! The KV store allocates nothing per step (RFC-0022, Testing: "No allocation or lock per step";
//! TIER-E6-T02): every pool, block table, ring tag and retire list is sized in `KvStore::new`,
//! so the step loop's `append`, `note_read`, `truncate`, `flush` and `fence`, and the sequence
//! churn of `open`, `fork` and `release`, reuse it.

use syntropy_core::plan::KvDType;
use syntropy_core::{DType, LayerIx, SequenceId, StageIx};
use syntropy_manifest::kv::{KvFormula, LayerKv};
use syntropy_testkit::guards::{CountingAlloc, alloc_count_this_thread};
use syntropy_tier::experts::stats_ring;
use syntropy_tier::kv::{KvError, KvStore, KvStoreConfig, PAGE_TOKENS};

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

fn seq(n: u32) -> SequenceId {
    SequenceId {
        ingress: 1,
        boot: 1,
        n,
    }
}

/// The allocator counts at all: without that, a zero below would prove nothing.
#[test]
fn the_allocator_counts() {
    let before = alloc_count_this_thread();
    let boxed = Box::new(1_u64);
    assert_eq!(alloc_count_this_thread() - before, 1);
    drop(boxed);
}

#[test]
fn decode_steps_and_sequence_churn_allocate_nothing() {
    // Full, windowed and state classes.
    let formula = KvFormula::new(
        vec![
            LayerKv::gqa(8, 128, None),
            LayerKv::gqa(8, 128, Some(100)),
            LayerKv::gqa(8, 128, None),
        ],
        DType::F16,
    );
    let cfg = KvStoreConfig::from_formula(
        &formula,
        StageIx(0),
        LayerIx(0)..LayerIx(3),
        KvDType::Native,
        0,
        PAGE_TOKENS,
        8,
        4096,
        256,
    )
    .unwrap();
    let mut s = KvStore::new(cfg);
    let (mut tx, mut rx) = stats_ring(64, 1).unwrap();

    let before = alloc_count_this_thread();
    let mut forks = 0;
    for round in 0..50u32 {
        let mut a = s.open(seq(round), StageIx(0), None).unwrap();
        // Prefill in chunks the window ring takes, then decode.
        for start in (0..120).step_by(40) {
            s.append(&mut a, start..start + 40).unwrap();
        }
        let mut b = s.fork(&a, 100).unwrap();
        forks += 1;
        for step in 0..300u64 {
            for h in [&mut a, &mut b] {
                let from = h.tokens;
                let slots = s.append(h, from..from + 1).unwrap();
                assert!(slots.copies.len() <= 3);
                s.note_read(h).unwrap();
            }
            if step % 50 == 49 {
                // Rejected speculative positions.
                let to = b.tokens - 3;
                s.truncate(&mut b, to).unwrap();
            }
            s.flush(step, &mut tx);
            s.fence();
            if step % 16 == 0 {
                rx.drain(|_| {});
            }
        }
        s.release(a).map_err(|(_, e)| e).unwrap();
        s.release(b).map_err(|(_, e)| e).unwrap();
        s.fence();
        // A full pool is reported without allocating either.
        let mut c = s.open(seq(1000 + round), StageIx(0), None).unwrap();
        let e = s.append(&mut c, 0..4096).unwrap_err();
        assert!(matches!(
            e,
            KvError::BadPosition { .. } | KvError::PoolFull { .. }
        ));
        s.release(c).map_err(|(_, e)| e).unwrap();
    }
    assert_eq!(alloc_count_this_thread() - before, 0);
    assert_eq!(forks, 50);
}
