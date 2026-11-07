// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the KV store (TIER-E6-T02): the classes a formula gives, the page state machine,
//! copy-on-write forks, window rings, the byte counters against `KvFormula`, and a model-based
//! property test (RFC-0022, Testing: "Page tables and pools"). The zero-allocation check is
//! `tests/kv_alloc.rs`, which needs its own global allocator.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use proptest::prelude::*;
use syntropy_core::plan::KvDType;
use syntropy_core::{DType, LayerIx, SequenceId, StageIx};
use syntropy_manifest::kv::{AuxKind, AuxKv, KvFormula, LayerKv};

use super::*;
use crate::experts::{StatRecord, stats_ring};

const STAGE: StageIx = StageIx(0);
const P: u32 = PAGE_TOKENS;

fn seq(n: u32) -> SequenceId {
    SequenceId {
        ingress: 1,
        boot: 1,
        n,
    }
}

/// A dense formula: `n` GQA layers, 8 KV heads of 128, F16.
fn dense(n: u16) -> KvFormula {
    KvFormula::new(vec![LayerKv::gqa(8, 128, None); usize::from(n)], DType::F16)
}

/// A hybrid formula: two full layers, two sliding-window layers of 100 tokens, one recurrent.
fn hybrid() -> KvFormula {
    KvFormula::new(
        vec![
            LayerKv::gqa(8, 128, None),
            LayerKv::gqa(8, 128, Some(100)),
            LayerKv::Fixed {
                elems_per_seq: 1 << 16,
            },
            LayerKv::gqa(8, 128, Some(100)),
            LayerKv::gqa(4, 64, None),
        ],
        DType::F16,
    )
}

fn layers(n: u16) -> std::ops::Range<LayerIx> {
    LayerIx(0)..LayerIx(n)
}

fn store(formula: &KvFormula, n: u16, seqs: u32, ctx: u32, full: u32) -> KvStore {
    KvStore::new(
        KvStoreConfig::from_formula(
            formula,
            STAGE,
            layers(n),
            KvDType::Native,
            0,
            P,
            seqs,
            ctx,
            full,
        )
        .unwrap(),
    )
}

fn full_pages(s: &KvStore, h: &KvHandle) -> Vec<PageId> {
    s.table(h.pages).classes[0]
        .pages
        .iter()
        .map(|r| r.page)
        .collect()
}

// ---- classes and layout ------------------------------------------------------------------------

#[test]
fn a_formula_gives_full_windowed_and_state_classes_sized_for_the_plan() {
    let cfg = KvStoreConfig::from_formula(
        &hybrid(),
        STAGE,
        layers(5),
        KvDType::Native,
        0,
        P,
        4,
        4096,
        100,
    )
    .unwrap();
    let classes: Vec<_> = cfg
        .classes
        .iter()
        .map(|c| (c.class, c.layers.clone(), c.capacity))
        .collect();
    assert_eq!(
        classes,
        [
            (PageClass::Full, vec![LayerIx(0), LayerIx(4)], 100),
            // A ring of ⌈100 / 64⌉ + 1 = 3 pages per sequence.
            (
                PageClass::Windowed { window: 100 },
                vec![LayerIx(1), LayerIx(3)],
                12
            ),
            (PageClass::State, vec![LayerIx(2)], 4),
        ]
    );
    // F16: 2 · 8 · 128 elements of 2 bytes per token, and 2 · 4 · 64 for layer 4.
    assert_eq!(cfg.classes[0].unit_bytes.0, 4096 + 1024);
    assert_eq!(
        cfg.classes[0].layer_page_bytes,
        [
            syntropy_core::Bytes(4096 * 64),
            syntropy_core::Bytes(1024 * 64)
        ]
    );
    assert_eq!(cfg.classes[2].unit_bytes.0, 2 << 16);
}

#[test]
fn compressed_layers_aux_caches_and_odd_page_sizes_are_refused() {
    let mut f = dense(2);
    f.layers[1] = LayerKv::Compressed {
        elems_per_entry: 512,
        tokens_per_entry: 4,
        window: Some(128),
        selected: None,
        latent: true,
    };
    let refused = |f: &KvFormula, p: u32| {
        KvStoreConfig::from_formula(f, STAGE, layers(2), KvDType::Native, 0, p, 1, 64, 1)
            .unwrap_err()
    };
    assert!(matches!(refused(&f, P), KvError::Unsupported(_)));
    let mut f = dense(2);
    f.aux.push(AuxKv {
        layer: LayerIx(1),
        kind: AuxKind::SparseIndexer,
        elems_per_token: 128,
        dtype: DType::F16,
    });
    assert!(matches!(refused(&f, P), KvError::Unsupported(_)));
    for p in [0, 8, 48, 512] {
        assert_eq!(
            refused(&dense(2), p).class(),
            ErrorClass::Unsupported,
            "P = {p}"
        );
    }
    // The other page sizes ADR-0205 allows are accepted.
    for p in [16, 32, 128, 256] {
        KvStoreConfig::from_formula(&dense(2), STAGE, layers(2), KvDType::Native, 0, p, 1, 64, 1)
            .unwrap();
    }
}

#[test]
fn the_arena_is_layer_first_with_aligned_disjoint_arrays() {
    let s = store(&hybrid(), 5, 4, 4096, 100);
    let mut ranges = Vec::new();
    for c in &s.config().classes {
        for (l, b) in c.layers.iter().zip(&c.layer_page_bytes) {
            let first = s
                .page_offset(c.class, *l, PageId { tier: 0, index: 0 })
                .unwrap();
            let last = s
                .page_offset(
                    c.class,
                    *l,
                    PageId {
                        tier: 0,
                        index: c.capacity - 1,
                    },
                )
                .unwrap();
            assert_eq!(first.0 % LAYER_ALIGN, 0);
            assert_eq!(last.0 - first.0, b.0 * u64::from(c.capacity - 1));
            ranges.push((first.0, last.0 + b.0));
        }
    }
    ranges.sort_unstable();
    for w in ranges.windows(2) {
        assert!(w[0].1 <= w[1].0, "{w:?} overlap");
    }
    assert_eq!(ranges.last().unwrap().1, s.arena_bytes().0);
    let out = PageId {
        tier: 0,
        index: 100,
    };
    assert_eq!(s.page_offset(PageClass::Full, LayerIx(0), out), None);
    assert_eq!(
        s.page_offset(PageClass::Full, LayerIx(1), PageId { tier: 0, index: 0 }),
        None,
        "layer 1 is windowed"
    );
}

// ---- append, seal, truncate, release -----------------------------------------------------------

#[test]
fn appends_fill_pages_in_order_and_seal_each_one_that_fills() {
    let mut s = store(&dense(2), 2, 2, 1024, 16);
    let mut h = s.open(seq(1), STAGE, None).unwrap();
    let w = s.append(&mut h, 0..100).unwrap();
    assert_eq!((w.positions, w.table, w.copies.len()), (0..100, h.pages, 0));
    assert_eq!(h.tokens, 100);
    let pages = full_pages(&s, &h);
    assert_eq!(pages.len(), 2);
    assert_eq!(
        s.page_state(PageClass::Full, pages[0]),
        Some(PageState::Sealed)
    );
    assert_eq!(
        s.page_state(PageClass::Full, pages[1]),
        Some(PageState::Writable)
    );
    s.append(&mut h, 100..128).unwrap();
    assert_eq!(
        s.page_state(PageClass::Full, pages[1]),
        Some(PageState::Sealed)
    );
    assert_eq!(h.residency.vram, 2);
    assert_eq!(s.headroom().tokens, [14 * P, 0, 0]);
}

#[test]
fn an_append_that_does_not_follow_the_sequence_is_refused() {
    let mut s = store(&dense(1), 1, 1, 256, 8);
    let mut h = s.open(seq(1), STAGE, None).unwrap();
    s.append(&mut h, 0..10).unwrap();
    for bad in [0..5, 11..12, 9..20, 10..257] {
        let e = s.append(&mut h, bad.clone()).unwrap_err();
        assert!(matches!(e, KvError::BadPosition { .. }), "{bad:?}: {e}");
    }
    assert_eq!(h.tokens, 10);
    // An empty append is fine and changes nothing.
    assert!(s.append(&mut h, 10..10).unwrap().copies.is_empty());
}

#[test]
fn a_full_pool_refuses_the_append_and_changes_nothing() {
    let mut s = store(&dense(1), 1, 2, 4096, 3);
    let mut a = s.open(seq(1), STAGE, None).unwrap();
    s.append(&mut a, 0..130).unwrap(); // 3 pages
    let mut b = s.open(seq(2), STAGE, None).unwrap();
    let e = s.append(&mut b, 0..1).unwrap_err();
    assert_eq!(
        e,
        KvError::PoolFull {
            class: PageClass::Full,
            tier: 0
        }
    );
    assert_eq!((e.class(), e.kv_code()), (ErrorClass::Capacity, Some(1)));
    assert!(e.to_string().starts_with("KV_POOL_FULL"));
    assert_eq!(b.tokens, 0);
    assert!(s.table(b.pages).classes[0].pages.is_empty());
    // A partial page still takes positions without a new page.
    s.append(&mut a, 130..192).unwrap();
    assert!(s.append(&mut a, 192..193).is_err());
    assert_eq!(a.tokens, 192);
}

#[test]
fn truncation_releases_whole_pages_which_the_fence_frees() {
    let mut s = store(&dense(1), 1, 1, 1024, 8);
    let mut h = s.open(seq(1), STAGE, None).unwrap();
    s.append(&mut h, 0..300).unwrap(); // 5 pages
    s.truncate(&mut h, 70).unwrap();
    assert_eq!(h.tokens, 70);
    assert_eq!(full_pages(&s, &h).len(), 2);
    // Released pages wait for the step fence.
    assert_eq!(s.free_pages(PageClass::Full), Some(3));
    assert_eq!(s.fence(), 3);
    assert_eq!(s.free_pages(PageClass::Full), Some(6));
    // The cut page was sealed; the next append reopens it, with no copy.
    let tail = full_pages(&s, &h)[1];
    assert_eq!(s.page_state(PageClass::Full, tail), Some(PageState::Sealed));
    let w = s.append(&mut h, 70..80).unwrap();
    assert!(w.copies.is_empty());
    assert_eq!(full_pages(&s, &h)[1], tail);
    assert_eq!(
        s.page_state(PageClass::Full, tail),
        Some(PageState::Writable)
    );
    // Truncating at or past the end does nothing.
    s.truncate(&mut h, 80).unwrap();
    s.truncate(&mut h, 500).unwrap();
    assert_eq!(h.tokens, 80);
}

#[test]
fn release_returns_every_page_and_the_slot() {
    let mut s = store(&hybrid(), 5, 1, 4096, 16);
    let mut h = s.open(seq(1), STAGE, None).unwrap();
    for start in (0..200).step_by(50) {
        s.append(&mut h, start..start + 50).unwrap();
    }
    assert!(matches!(s.open(seq(2), STAGE, None), Err(KvError::NoSlot)));
    s.release(h).unwrap();
    s.fence();
    for c in [
        PageClass::Full,
        PageClass::Windowed { window: 100 },
        PageClass::State,
    ] {
        assert_eq!(s.live_pages(c), Some(0), "{c:?}");
    }
    let h = s.open(seq(2), STAGE, None).unwrap();
    assert_eq!(h.tokens, 0);
}

#[test]
fn a_handle_for_another_stage_or_slot_is_refused() {
    let mut s = store(&dense(1), 1, 2, 256, 4);
    assert!(matches!(
        s.open(seq(1), StageIx(3), None),
        Err(KvError::WrongStage { .. })
    ));
    let h = s.open(seq(1), STAGE, None).unwrap();
    let mut forged = KvHandle {
        pages: PageTableId(1),
        ..h
    };
    assert!(matches!(
        s.append(&mut forged, 0..1),
        Err(KvError::BadTable { .. })
    ));
}

// ---- fork and copy-on-write --------------------------------------------------------------------

#[test]
fn a_fork_shares_pages_and_whichever_side_writes_first_copies_the_cut_page() {
    let mut s = store(&dense(1), 1, 2, 1024, 8);
    let mut a = s.open(seq(1), STAGE, None).unwrap();
    s.append(&mut a, 0..100).unwrap();
    let mut b = s.fork(&a, 80).unwrap();
    assert_eq!(b.tokens, 80);
    let (pa, pb) = (full_pages(&s, &a), full_pages(&s, &b));
    assert_eq!(pa, pb, "both share pages 0 and 1");
    assert!(s.table(b.pages).classes[0].pages.iter().all(|r| r.shared));
    assert_eq!(s.page_refs(PageClass::Full, pa[1]), Some(2));
    // The cut page is sealed, so neither writes it in place.
    assert_eq!(
        s.page_state(PageClass::Full, pa[1]),
        Some(PageState::Sealed)
    );

    let w = s.append(&mut b, 80..90).unwrap();
    let copy = w.copies[0];
    assert_eq!(
        (copy.class, copy.from, copy.tokens),
        (PageClass::Full, pa[1], 16)
    );
    assert_ne!(copy.to, pa[1]);
    assert_eq!(full_pages(&s, &b)[1], copy.to);
    assert_eq!(s.page_refs(PageClass::Full, pa[1]), Some(1));

    // The source is now the page's only holder: it reopens it, with no copy.
    let w = s.append(&mut a, 100..110).unwrap();
    assert!(w.copies.is_empty());
    assert_eq!(full_pages(&s, &a)[1], pa[1]);
    // Page 0 stays shared and sealed.
    assert_eq!(s.page_refs(PageClass::Full, pa[0]), Some(2));
}

#[test]
fn a_fork_at_a_page_boundary_copies_nothing() {
    let mut s = store(&dense(1), 1, 2, 1024, 8);
    let mut a = s.open(seq(1), STAGE, None).unwrap();
    s.append(&mut a, 0..100).unwrap();
    let mut b = s.fork(&a, 64).unwrap();
    assert_eq!(full_pages(&s, &b).len(), 1);
    assert!(s.append(&mut b, 64..70).unwrap().copies.is_empty());
    // The source's tail page was not shared, so it stays writable.
    assert_eq!(
        s.page_state(PageClass::Full, full_pages(&s, &a)[1]),
        Some(PageState::Writable)
    );
}

#[test]
fn state_cannot_be_forked_or_cut_back_without_snapshots() {
    let mut s = store(&hybrid(), 5, 2, 4096, 16);
    let mut a = s.open(seq(1), STAGE, None).unwrap();
    s.append(&mut a, 0..50).unwrap();
    assert!(matches!(s.fork(&a, 40), Err(KvError::Unsupported(_))));
    assert!(matches!(
        s.truncate(&mut a, 40),
        Err(KvError::Unsupported(_))
    ));
    // At the sequence's end, the fork shares the state page; each side copies it on write.
    let mut b = s.fork(&a, 50).unwrap();
    let w = s.append(&mut b, 50..51).unwrap();
    let state = w
        .copies
        .iter()
        .find(|c| c.class == PageClass::State)
        .unwrap();
    assert_eq!(state.tokens, 0, "the whole state");
    assert!(
        s.append(&mut a, 50..51)
            .unwrap()
            .copies
            .iter()
            .all(|c| c.class != PageClass::State)
    );
}

// ---- window rings ------------------------------------------------------------------------------

#[test]
fn a_window_ring_reuses_its_oldest_page_in_place() {
    let mut s = store(&hybrid(), 5, 1, 8192, 200);
    let win = PageClass::Windowed { window: 100 };
    let mut h = s.open(seq(1), STAGE, None).unwrap();
    for start in (0..1000).step_by(50) {
        s.append(&mut h, start..start + 50).unwrap();
    }
    let ring = &s.table(h.pages).classes[1].pages;
    assert_eq!(ring.len(), 3);
    // Three pages in all, ever: the ring never takes a fourth.
    assert_eq!(s.live_pages(win), Some(3));
    // A chunk longer than the ring holds beside its first page is refused.
    let e = s.append(&mut h, 1000..1000 + 2 * P).unwrap_err();
    assert!(matches!(e, KvError::BadPosition { .. }));
}

#[test]
fn a_window_ring_cut_back_past_what_it_holds_is_refused() {
    // No State class: only full and windowed layers.
    let f = KvFormula::new(
        vec![LayerKv::gqa(8, 128, None), LayerKv::gqa(8, 128, Some(100))],
        DType::F16,
    );
    let mut s = store(&f, 2, 2, 8192, 200);
    let mut h = s.open(seq(1), STAGE, None).unwrap();
    for start in (0..300).step_by(50) {
        s.append(&mut h, start..start + 50).unwrap();
    }
    // The ring holds token pages 2..=4 (positions 128..300). A cut to 250 keeps 150..250.
    s.truncate(&mut h, 250).unwrap();
    // A cut to 200 needs positions 100..200, and 100..128 left the ring.
    assert!(matches!(
        s.truncate(&mut h, 200),
        Err(KvError::BadPosition { .. })
    ));
    assert_eq!(h.tokens, 250);
    // The emptied slot fills again, and the ring then holds token pages 2..=4 once more.
    s.append(&mut h, 250..300).unwrap();
    s.truncate(&mut h, 250).unwrap();
    // A cut that vacates the slot reused by token page 4 cannot bring token page 1 back.
    s.append(&mut h, 250..300).unwrap();
    assert!(matches!(s.fork(&h, 220), Err(KvError::BadPosition { .. })));
    s.fence();
    assert_eq!(s.live_pages(PageClass::Windowed { window: 100 }), Some(3));
}

// ---- moving pages ------------------------------------------------------------------------------

#[test]
fn a_moving_page_is_never_read_written_or_released() {
    let mut s = store(&dense(1), 1, 2, 1024, 8);
    let mut h = s.open(seq(1), STAGE, None).unwrap();
    s.append(&mut h, 0..64).unwrap();
    let page = full_pages(&s, &h)[0];
    let ticket = MoveTicket(7);
    // Only a sealed page moves.
    let mut h2 = s.open(seq(2), STAGE, None).unwrap();
    s.append(&mut h2, 0..10).unwrap();
    let writable = full_pages(&s, &h2)[0];
    assert!(matches!(
        s.begin_move(PageClass::Full, writable, 2, ticket),
        Err(KvError::BadState { .. })
    ));
    s.begin_move(PageClass::Full, page, 2, ticket).unwrap();
    assert_eq!(
        s.page_state(PageClass::Full, page),
        Some(PageState::Moving { to: 2, ticket })
    );
    let e = s.note_read(&h).unwrap_err();
    assert_eq!(
        (e, e.class()),
        (KvError::Moving { page }, ErrorClass::Transient)
    );
    // A second page, then a cut that would release the moving page.
    s.append(&mut h, 64..70).unwrap();
    assert!(matches!(s.truncate(&mut h, 0), Err(KvError::Moving { .. })));
    // A cut back into the moving page keeps it, but no append may then write it.
    s.truncate(&mut h, 10).unwrap();
    assert_eq!(
        s.append(&mut h, 10..11).unwrap_err(),
        KvError::Moving { page }
    );
    assert_eq!(h.tokens, 10);
    let (back, e) = s.release(h).unwrap_err();
    assert_eq!(e, KvError::Moving { page });
    let mut h = back;
    // A wrong ticket does not complete it.
    assert!(s.finish_move(PageClass::Full, page, MoveTicket(8)).is_err());
    s.finish_move(PageClass::Full, page, ticket).unwrap();
    s.note_read(&h).unwrap();
    s.append(&mut h, 10..11).unwrap();
    s.release(h).unwrap();
}

// ---- byte counters -----------------------------------------------------------------------------

#[test]
fn the_kv_byte_counters_equal_the_formula_for_a_scripted_decode() {
    for (formula, n) in [(dense(4), 4u16), (hybrid(), 5)] {
        let mut s = store(&formula, n, 1, 4096, 64);
        let (mut tx, mut rx) = stats_ring(1024, 1).unwrap();
        let mut h = s.open(seq(1), STAGE, None).unwrap();
        // A prefill of 60 positions in two chunks, then 200 decode steps.
        let mut steps = vec![30u32, 30];
        steps.extend(std::iter::repeat_n(1, 200));
        let mut step = 0;
        let mut written_full = 0;
        let mut appended = 0u32;
        for chunk in steps {
            let from = h.tokens;
            s.append(&mut h, from..from + chunk).unwrap();
            appended += chunk;
            s.note_read(&h).unwrap();
            let read: u64 = s.counters().iter().map(|c| c.read).sum();
            let expected = formula.read_per_step(layers(n), u64::from(h.tokens), KvDType::Native);
            assert_eq!(read, expected.0, "step {step} at {} tokens", h.tokens);
            written_full += s.counters()[0].written;
            s.flush(step, &mut tx);
            step += 1;
        }
        // The full class wrote exactly what the formula stores for every appended position.
        let full: Vec<LayerIx> = s.config().classes[0].layers.clone();
        let stored: u64 = full
            .iter()
            .map(|l| {
                formula
                    .stored(*l..LayerIx(l.0 + 1), u64::from(appended), KvDType::Native)
                    .0
            })
            .sum();
        assert_eq!(written_full, stored);
        assert_eq!(appended, 260);
        // Every step's records reached the ring, one per class.
        let mut records = 0;
        rx.drain(|r| {
            if let StatRecord::KvBytes(k) = r {
                assert_eq!(k.stage, STAGE);
                records += 1;
            }
        });
        assert_eq!(records, step as usize * s.config().classes.len());
        assert!(s.counters().iter().all(|c| c.read == 0 && c.written == 0));
    }
}

// ---- the model-based property test -------------------------------------------------------------

/// One operation of the property test.
#[derive(Clone, Debug)]
enum Op {
    Open,
    Append { seq: usize, n: u32 },
    Truncate { seq: usize, back: u32 },
    Fork { seq: usize, back: u32 },
    Release { seq: usize },
    Read { seq: usize },
    Move { seq: usize, page: usize },
    Finish,
    Fence,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        2 => Just(Op::Open),
        6 => (0usize..8, 1u32..90).prop_map(|(seq, n)| Op::Append { seq, n }),
        2 => (0usize..8, 0u32..80).prop_map(|(seq, back)| Op::Truncate { seq, back }),
        2 => (0usize..8, 0u32..80).prop_map(|(seq, back)| Op::Fork { seq, back }),
        1 => (0usize..8).prop_map(|seq| Op::Release { seq }),
        2 => (0usize..8).prop_map(|seq| Op::Read { seq }),
        1 => (0usize..8, 0usize..8).prop_map(|(seq, page)| Op::Move { seq, page }),
        1 => Just(Op::Finish),
        2 => Just(Op::Fence),
    ]
}

/// The device memory the test pretends to have: the value at each (class, page, offset), written
/// by appends and copies exactly as the slots say. Every live sequence must read back, through
/// its block tables, the values the model says it appended.
#[derive(Default)]
struct Memory(BTreeMap<(usize, PageId, u32), u64>);

struct Live {
    h: KvHandle,
    /// The value at each position: what the sequence appended or inherited.
    values: Vec<u64>,
}

fn class_tables(s: &KvStore, h: &KvHandle) -> Vec<(usize, PageClass, Vec<PageRef>)> {
    s.table(h.pages)
        .classes
        .iter()
        .enumerate()
        .map(|(i, t)| (i, t.class, t.pages.clone()))
        .collect()
}

/// The page and offset of position `pos` in class `class`, from a block table.
fn locate(class: PageClass, pages: &[PageRef], pos: u32) -> Option<(PageId, u32)> {
    let t = pos / P;
    let i = match class {
        PageClass::Windowed { window } => t as usize % ring_pages(window, P) as usize,
        _ => t as usize,
    };
    pages.get(i).map(|r| (r.page, pos % P))
}

fn check(s: &KvStore, live: &[Live], mem: &Memory, moving: &[(PageClass, PageId)]) {
    // References: what the tables hold is exactly what the pools count.
    let mut held: BTreeMap<(usize, PageId), u16> = BTreeMap::new();
    for l in live {
        for (i, _, pages) in class_tables(s, &l.h) {
            for r in pages.iter().filter(|r| r.page.tier != u8::MAX) {
                *held.entry((i, r.page)).or_default() += 1;
            }
        }
    }
    for (ix, c) in s.config().classes.iter().enumerate() {
        let mut in_tables = 0;
        for index in 0..c.capacity {
            let page = PageId { tier: 0, index };
            let refs = s.page_refs(c.class, page).unwrap();
            let want = held.get(&(ix, page)).copied().unwrap_or(0);
            assert_eq!(refs, want, "{:?} {page:?}: refs", c.class);
            let state = s.page_state(c.class, page).unwrap();
            if want == 0 {
                assert!(
                    matches!(state, PageState::Free | PageState::Retired),
                    "{:?} {page:?} leaked: {state:?}",
                    c.class
                );
            } else {
                in_tables += 1;
                assert!(
                    !matches!(state, PageState::Free | PageState::Retired),
                    "{:?} {page:?} held but {state:?}",
                    c.class
                );
                // A shared page is never writable.
                if want > 1 {
                    assert!(
                        !matches!(state, PageState::Writable),
                        "{page:?} shared and writable"
                    );
                }
            }
        }
        assert_eq!(s.live_pages(c.class), Some(in_tables));
    }
    for (class, page) in moving {
        assert!(matches!(
            s.page_state(*class, *page),
            Some(PageState::Moving { .. })
        ));
    }
    // Contents: every full-class position and every windowed position still in the window.
    for l in live {
        for (i, class, pages) in class_tables(s, &l.h) {
            let lo = match class {
                PageClass::Windowed { window } => l.h.tokens.saturating_sub(window),
                PageClass::State => continue,
                _ => 0,
            };
            for pos in lo..l.h.tokens {
                let (page, off) = locate(class, &pages, pos).expect("a page for every position");
                assert_eq!(
                    mem.0.get(&(i, page, off)),
                    Some(&l.values[pos as usize]),
                    "{class:?} position {pos} of {:?}",
                    l.h.pages
                );
            }
        }
    }
}

fn run(ops: Vec<Op>, formula: &KvFormula, n: u16) {
    let mut s = store(formula, n, 6, 2048, 40);
    let mut live: Vec<Live> = Vec::new();
    let mut mem = Memory::default();
    let mut moving: Vec<(PageClass, PageId)> = Vec::new();
    let mut next_value = 1u64;
    let mut tickets = 0u64;
    let has_state = s
        .config()
        .classes
        .iter()
        .any(|c| c.class == PageClass::State);
    let moving_in = |s: &KvStore, h: &KvHandle, moving: &[(PageClass, PageId)]| {
        class_tables(s, h).iter().any(|(_, c, pages)| {
            pages
                .iter()
                .any(|r| moving.iter().any(|(mc, mp)| mc == c && *mp == r.page))
        })
    };
    for op in ops {
        match op {
            Op::Open => match s.open(seq(0), STAGE, None) {
                Ok(h) => live.push(Live {
                    h,
                    values: Vec::new(),
                }),
                Err(KvError::NoSlot | KvError::PoolFull { .. }) => {}
                Err(e) => panic!("open: {e}"),
            },
            Op::Append { seq, n } if !live.is_empty() => {
                let i = seq % live.len();
                let from = live[i].h.tokens;
                let tables_before = class_tables(&s, &live[i].h);
                match s.append(&mut live[i].h, from..from + n) {
                    Ok(w) => {
                        // Copies first, as the caller must issue them, then the writes.
                        for c in &w.copies {
                            let ix = s
                                .config()
                                .classes
                                .iter()
                                .position(|k| k.class == c.class)
                                .unwrap();
                            for off in 0..c.tokens {
                                let v = mem.0.get(&(ix, c.from, off)).copied();
                                if let Some(v) = v {
                                    mem.0.insert((ix, c.to, off), v);
                                }
                            }
                        }
                        for (ix, class, pages) in class_tables(&s, &live[i].h) {
                            if class == PageClass::State {
                                continue;
                            }
                            for pos in from..from + n {
                                let (page, off) = locate(class, &pages, pos).unwrap();
                                // Never written while another table holds it.
                                assert_eq!(s.page_refs(class, page), Some(1), "{page:?} shared");
                                mem.0
                                    .insert((ix, page, off), next_value + u64::from(pos - from));
                            }
                        }
                        live[i]
                            .values
                            .extend((0..u64::from(n)).map(|k| next_value + k));
                        next_value += u64::from(n);
                    }
                    Err(KvError::PoolFull { .. } | KvError::Moving { .. }) => {
                        assert_eq!(live[i].h.tokens, from, "a failed append changed the length");
                        assert_eq!(class_tables(&s, &live[i].h), tables_before);
                    }
                    Err(KvError::BadPosition { .. }) => {
                        // Past the slot's context, or longer than a window ring takes beside
                        // the chunk's first page.
                        let ring_limit = s.config().classes.iter().find_map(|c| match c.class {
                            PageClass::Windowed { window } => {
                                Some((ring_pages(window, P) - 1) * P - from % P)
                            }
                            _ => None,
                        });
                        assert!(from + n > 2048 || ring_limit.is_some_and(|l| n > l));
                    }
                    Err(e) => panic!("append: {e}"),
                }
            }
            Op::Truncate { seq, back } if !live.is_empty() => {
                let i = seq % live.len();
                let to = live[i].h.tokens.saturating_sub(back);
                match s.truncate(&mut live[i].h, to) {
                    Ok(()) => live[i].values.truncate(to as usize),
                    Err(KvError::Unsupported(_)) => assert!(has_state),
                    Err(KvError::BadPosition { .. } | KvError::Moving { .. }) => {}
                    Err(e) => panic!("truncate: {e}"),
                }
            }
            Op::Fork { seq, back } if !live.is_empty() => {
                let i = seq % live.len();
                let at = live[i].h.tokens.saturating_sub(back);
                match s.fork(&live[i].h, at) {
                    Ok(h) => {
                        let values = live[i].values[..at as usize].to_vec();
                        live.push(Live { h, values });
                    }
                    Err(KvError::Unsupported(_)) => assert!(has_state),
                    Err(KvError::NoSlot | KvError::BadPosition { .. } | KvError::Moving { .. }) => {
                    }
                    Err(e) => panic!("fork: {e}"),
                }
            }
            Op::Release { seq } if !live.is_empty() => {
                let i = seq % live.len();
                let l = live.swap_remove(i);
                let was_moving = moving_in(&s, &l.h, &moving);
                match s.release(l.h) {
                    Ok(()) => assert!(!was_moving),
                    Err((h, KvError::Moving { .. })) => {
                        assert!(was_moving);
                        live.push(Live {
                            h,
                            values: l.values,
                        });
                    }
                    Err((_, e)) => panic!("release: {e}"),
                }
            }
            Op::Read { seq } if !live.is_empty() => {
                let i = seq % live.len();
                let was_moving = moving_in(&s, &live[i].h, &moving);
                match s.note_read(&live[i].h) {
                    Ok(()) => assert!(!was_moving, "a moving page was read"),
                    Err(KvError::Moving { .. }) => assert!(was_moving),
                    Err(e) => panic!("read: {e}"),
                }
            }
            Op::Move { seq, page } if !live.is_empty() => {
                let i = seq % live.len();
                let tables = class_tables(&s, &live[i].h);
                let (_, class, pages) = &tables[0];
                if let Some(r) = pages.get(page)
                    && s.page_state(*class, r.page) == Some(PageState::Sealed)
                {
                    tickets += 1;
                    s.begin_move(*class, r.page, 2, MoveTicket(tickets))
                        .unwrap();
                    moving.push((*class, r.page));
                }
            }
            Op::Finish => {
                if let Some((class, page)) = moving.pop() {
                    let Some(PageState::Moving { ticket, .. }) = s.page_state(class, page) else {
                        panic!("{page:?} stopped moving");
                    };
                    s.finish_move(class, page, ticket).unwrap();
                }
            }
            Op::Fence => {
                s.fence();
            }
            _ => {}
        }
        check(&s, &live, &mem, &moving);
    }
    // Draining everything leaves every pool whole.
    while let Some((class, page)) = moving.pop() {
        let Some(PageState::Moving { ticket, .. }) = s.page_state(class, page) else {
            unreachable!()
        };
        s.finish_move(class, page, ticket).unwrap();
    }
    for l in live.drain(..) {
        s.release(l.h).map_err(|(_, e)| e).unwrap();
    }
    s.fence();
    for c in s.config().classes.clone() {
        assert_eq!(s.free_pages(c.class), Some(c.capacity), "{:?}", c.class);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// RFC-0022's "Page tables and pools" row on full layers: no page leaked, double-freed,
    /// read while moving or written while shared, and every sequence reads back its own values.
    #[test]
    fn full_class_pools_follow_the_model(ops in proptest::collection::vec(op(), 1..120)) {
        run(ops, &dense(2), 2);
    }

    /// The same on a formula with full, windowed and state classes.
    #[test]
    fn mixed_class_pools_follow_the_model(ops in proptest::collection::vec(op(), 1..120)) {
        run(ops, &hybrid(), 5);
    }

    /// The same on full and windowed classes, where truncation and forks are allowed.
    #[test]
    fn windowed_class_pools_follow_the_model(ops in proptest::collection::vec(op(), 1..120)) {
        let f = KvFormula::new(
            vec![LayerKv::gqa(8, 128, None), LayerKv::gqa(2, 64, Some(100))],
            DType::F16,
        );
        run(ops, &f, 2);
    }
}
