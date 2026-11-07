// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for `KvWrite(Gqa)`, `AttnDecode(Gqa)` and `AttnPrefill(Gqa)`, and
//! property tests for `rows_active = 0`, masked positions and NaN propagation.

use proptest::prelude::*;

use super::*;
use crate::reference::rope::{Rope, RopePairs, RopeScaling};
use crate::reference::trig;

const SENTINEL: f32 = -12345.0;
const FILL: u8 = 0xAB;

fn layout(n_kv: usize, d: usize, p: usize, dtype: DType) -> KvLayout {
    KvLayout::new(n_kv, d, p, dtype).unwrap()
}

fn pool(l: &KvLayout, pages: usize) -> Vec<u8> {
    vec![FILL; pages * l.page_bytes()]
}

fn plain_rope(d_rot: usize) -> RopeTable {
    RopeTable::new(&Rope {
        scaling: RopeScaling::Plain,
        theta: 10_000.0,
        d_rot,
        pairs: RopePairs::Halves,
    })
    .unwrap()
}

/// The rows of one sequence (index `seq`) at positions `from .. from + n`, all writing.
fn rows(seq: u32, from: u32, n: u32) -> Vec<Row> {
    (from..from + n)
        .map(|pos| Row {
            seq,
            pos,
            kv_write_from: pos,
        })
        .collect()
}

/// Stores `k` and `v` (one row per position from 0) for sequence 0 with page table `pages`.
fn fill(
    l: &KvLayout,
    rope: Option<&RopeTable>,
    k: &[f32],
    v: &[f32],
    pages: &[u32],
    pool: &mut [u8],
) {
    let n = k.len() / (l.n_kv_heads() * l.d_head());
    let seqs = [Seq {
        pages,
        kv_len: n as u32,
    }];
    kv_write(l, rope, k, v, &rows(0, 0, n as u32), n, &seqs, pool).unwrap();
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

fn u16_at(pool: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([pool[at], pool[at + 1]])
}

#[test]
fn kv_write_rounds_once_after_rope() {
    // One KV head of 2 values, 2 positions per page, F16; θ = 10000 over both dims, so the
    // frequency is 1 and position 1 rotates by 1 radian.
    let l = layout(1, 2, 2, DType::F16);
    let t = plain_rope(2);
    let mut p = pool(&l, 2);
    let k = [1.0, 0.0];
    // 1 + 2^-11 is a tie between F16 neighbours and rounds to even (1.0); with 2^-20 more it
    // rounds up to 1 + 2^-10.
    let v = [1.0 + 2f32.powi(-11), 1.0 + 2f32.powi(-11) + 2f32.powi(-20)];
    let seqs = [Seq {
        pages: &[1],
        kv_len: 2,
    }];
    kv_write(&l, Some(&t), &k, &v, &rows(0, 1, 1), 1, &seqs, &mut p).unwrap();
    // Page 1, slot 1: K at element (0·1 + 0)·2 + 1 = 1 of the page's K plane, V after the K
    // plane's 4 elements.
    let page = l.page_bytes();
    let (s, c) = trig::sin_cos(1.0);
    assert_eq!(u16_at(&p, page + 2 * 2), f16_bits(c));
    assert_eq!(u16_at(&p, page + 2 * 2 + 2), f16_bits(s));
    // Halves pairing over d_rot = 2 is the pair (0, 1): [1, 0] → [cos 1, sin 1].
    assert_eq!(u16_at(&p, page + 6 * 2), 0x3C00);
    assert_eq!(u16_at(&p, page + 6 * 2 + 2), 0x3C01);
    // Nothing else changed: page 0, and page 1's slot 0.
    for (i, &b) in p.iter().enumerate() {
        let written = [4, 5, 6, 7, 12, 13, 14, 15].contains(&(i.wrapping_sub(page)));
        if !written {
            assert_eq!(b, FILL, "byte {i}");
        }
    }
}

#[test]
fn kv_write_bf16_and_f32() {
    let v = [1.0 + 2f32.powi(-8), 1.0 + 2f32.powi(-8) + 2f32.powi(-16)];
    let l = layout(1, 2, 1, DType::BF16);
    let mut p = pool(&l, 1);
    fill(&l, None, &[0.1, -0.1], &v, &[0], &mut p);
    assert_eq!(u16_at(&p, 0), round_bf16(0.1));
    assert_eq!(u16_at(&p, 2), round_bf16(-0.1));
    assert_eq!(u16_at(&p, 4), 0x3F80);
    assert_eq!(u16_at(&p, 6), 0x3F81);
    let l = layout(1, 2, 1, DType::F32);
    let mut p = pool(&l, 1);
    fill(&l, None, &[0.1, f32::NAN], &v, &[0], &mut p);
    assert_eq!(&p[..4], &0.1f32.to_le_bytes());
    assert_eq!(&p[4..8], &f32::NAN.to_le_bytes());
    assert_eq!(&p[8..12], &v[0].to_le_bytes());
}

#[test]
fn kv_write_skips_replay_rows_and_follows_the_page_table() {
    // Two KV heads of 1 value, 2 positions per page; the sequence's pages are [3, 1].
    let l = layout(2, 1, 2, DType::F32);
    let mut p = pool(&l, 4);
    let seqs = [Seq {
        pages: &[3, 1],
        kv_len: 4,
    }];
    let rs = [
        // A replay row: position 1 is below its kv_write_from, so it writes nothing.
        Row {
            seq: 0,
            pos: 1,
            kv_write_from: 3,
        },
        Row {
            seq: 0,
            pos: 3,
            kv_write_from: 3,
        },
    ];
    kv_write(
        &l,
        None,
        &[1.0, 2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0, 8.0],
        &rs,
        2,
        &seqs,
        &mut p,
    )
    .unwrap();
    // Position 3 is page 1, slot 1. Each plane is [head][slot]: K head 0 slot 1 is element 1,
    // K head 1 slot 1 element 3, V head 0 slot 1 element 5, V head 1 slot 1 element 7.
    let f = |page: usize, e: usize| {
        let at = page * l.page_bytes() + 4 * e;
        f32::from_le_bytes(p[at..at + 4].try_into().unwrap())
    };
    assert_eq!([f(1, 1), f(1, 3), f(1, 5), f(1, 7)], [3.0, 4.0, 7.0, 8.0]);
    let written: Vec<usize> = [1, 3, 5, 7]
        .iter()
        .map(|e| l.page_bytes() + 4 * e)
        .collect();
    for (i, &b) in p.iter().enumerate() {
        if !written.iter().any(|&w| (w..w + 4).contains(&i)) {
            assert_eq!(b, FILL, "byte {i}");
        }
    }
}

/// Attention of the rows `rs` of sequence 0 (kv_len `kv_len`, pages `pages`).
#[allow(clippy::too_many_arguments)]
fn attend_seq(
    gqa: &Gqa,
    l: &KvLayout,
    rope: Option<&RopeTable>,
    q: &[f32],
    rs: &[Row],
    pages: &[u32],
    kv_len: u32,
    pool: &[u8],
) -> Vec<f32> {
    let width = gqa.n_q_heads * l.d_head();
    let mut out = vec![SENTINEL; rs.len() * width + 3];
    let seqs = [Seq { pages, kv_len }];
    attn_decode(gqa, l, rope, q, rs, rs.len(), &seqs, pool, &mut out).unwrap();
    let mut prefill = vec![SENTINEL; out.len()];
    attn_prefill(gqa, l, rope, q, rs, rs.len(), &seqs, pool, &mut prefill).unwrap();
    assert_eq!(bits(&out), bits(&prefill), "decode and prefill differ");
    assert_eq!(&out[rs.len() * width..], &[SENTINEL; 3], "padding written");
    out.truncate(rs.len() * width);
    out
}

#[test]
fn equal_scores_average_the_values() {
    // q = 0 makes every score 0, so each key weighs 1 and the output is the mean of the values.
    let l = layout(1, 2, 4, DType::F32);
    let mut p = pool(&l, 1);
    fill(
        &l,
        None,
        &[1.0, 2.0, -3.0, 4.0],
        &[1.0, 3.0, 3.0, 5.0],
        &[0],
        &mut p,
    );
    let gqa = Gqa {
        n_q_heads: 1,
        scale: None,
    };
    let out = attend_seq(&gqa, &l, None, &[0.0, 0.0], &rows(0, 1, 1), &[0], 2, &p);
    assert_eq!(out, [2.0, 4.0]);
}

#[test]
fn two_keys_by_hand() {
    // scale 1, q = [1, 0], k0 = [1, 0], k1 = [0, 0]: s = [1, 0], m = 1, e = [1, e^-1],
    // l = 1 + e^-1, out = [1 / l, e^-1 / l] = [0.731058578, 0.268941421].
    let l = layout(1, 2, 2, DType::F32);
    let mut p = pool(&l, 1);
    fill(
        &l,
        None,
        &[1.0, 0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0, 1.0],
        &[0],
        &mut p,
    );
    let gqa = Gqa {
        n_q_heads: 1,
        scale: Some(1.0),
    };
    let out = attend_seq(&gqa, &l, None, &[1.0, 0.0], &rows(0, 1, 1), &[0], 2, &p);
    let e1 = math::exp(-1.0);
    let lsum = 1.0 + e1;
    assert_eq!(bits(&out), bits(&[1.0 / lsum, e1 / lsum]));
    assert!((out[0] - 0.731_058_6).abs() <= f32::EPSILON);
    assert!((out[1] - 0.268_941_43).abs() <= f32::EPSILON);
    // The row at position 0 sees only key 0: out = v0 exactly.
    let out = attend_seq(&gqa, &l, None, &[1.0, 0.0], &rows(0, 0, 1), &[0], 2, &p);
    assert_eq!(out, [1.0, 0.0]);
}

#[test]
fn a_dominant_score_selects_its_value() {
    // Default scale 1/√2: scores ±70.7, so e^-141.4 underflows to 0 and the output is v0.
    let l = layout(1, 2, 2, DType::F16);
    let mut p = pool(&l, 1);
    fill(
        &l,
        None,
        &[1.0, 0.0, -1.0, 0.0],
        &[0.5, -2.0, 9.0, 9.0],
        &[0],
        &mut p,
    );
    let gqa = Gqa {
        n_q_heads: 1,
        scale: None,
    };
    let out = attend_seq(&gqa, &l, None, &[100.0, 0.0], &rows(0, 1, 1), &[0], 2, &p);
    assert_eq!(out, [0.5, -2.0]);
}

#[test]
fn query_heads_share_their_kv_head() {
    // 4 query heads on 2 KV heads of 1 value: heads 0 and 1 read KV head 0, heads 2 and 3 head 1.
    let l = layout(2, 1, 1, DType::F32);
    let mut p = pool(&l, 1);
    fill(&l, None, &[0.0, 0.0], &[10.0, 20.0], &[0], &mut p);
    let gqa = Gqa {
        n_q_heads: 4,
        scale: None,
    };
    let out = attend_seq(
        &gqa,
        &l,
        None,
        &[1.0, 2.0, 3.0, 4.0],
        &rows(0, 0, 1),
        &[0],
        1,
        &p,
    );
    assert_eq!(out, [10.0, 10.0, 20.0, 20.0]);
}

#[test]
fn rope_on_q_equals_a_rotated_query() {
    // The op rotates q at the row's position; passing q already rotated, without RoPE, gives
    // the same bits.
    let l = layout(1, 4, 2, DType::F32);
    let t = plain_rope(4);
    let mut p = pool(&l, 2);
    let k = [
        0.3, -1.0, 2.0, 0.5, 1.0, 1.0, -1.0, 0.25, -0.5, 0.75, 0.0, 2.0,
    ];
    let v = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    fill(&l, Some(&t), &k, &v, &[1, 0], &mut p);
    let gqa = Gqa {
        n_q_heads: 1,
        scale: None,
    };
    let q = [0.9, -0.2, 0.4, 1.1];
    let with = attend_seq(&gqa, &l, Some(&t), &q, &rows(0, 2, 1), &[1, 0], 3, &p);
    let mut rotated = q;
    t.rotate(2, &mut rotated);
    let without = attend_seq(&gqa, &l, None, &rotated, &rows(0, 2, 1), &[1, 0], 3, &p);
    assert_eq!(bits(&with), bits(&without));
}

#[test]
fn refusals_write_nothing() {
    let l = layout(1, 2, 2, DType::F32);
    let p = pool(&l, 2);
    let gqa = Gqa {
        n_q_heads: 1,
        scale: None,
    };
    let q = [1.0; 4];
    let check = |gqa: &Gqa, rs: &[Row], seqs: &[Seq<'_>], pool: &[u8]| {
        let mut out = [SENTINEL; 4];
        let e = attn_decode(gqa, &l, None, &q, rs, rs.len(), seqs, pool, &mut out).unwrap_err();
        assert_eq!(out, [SENTINEL; 4]);
        e
    };
    let seq = |pages, kv_len| [Seq { pages, kv_len }];
    let pos_err = |e: OpError| matches!(e, OpError::Position { .. });
    // Position 2 not below a KV length of 2.
    assert!(pos_err(check(
        &gqa,
        &rows(0, 0, 2)
            .iter()
            .map(|r| Row {
                pos: r.pos + 1,
                ..*r
            })
            .collect::<Vec<_>>(),
        &seq(&[0, 1], 2),
        &p
    )));
    // Position 2 past a one-page table.
    assert!(pos_err(check(&gqa, &rows(0, 2, 1), &seq(&[0], 3), &p)));
    // A page past the pool.
    assert!(pos_err(check(&gqa, &rows(0, 0, 1), &seq(&[2], 1), &p)));
    // A sequence past the step's sequences.
    assert!(pos_err(check(&gqa, &rows(1, 0, 1), &seq(&[0], 1), &p)));
    // A pool that is not whole pages.
    let e = check(&gqa, &rows(0, 0, 1), &seq(&[0], 1), &p[..p.len() - 1]);
    assert!(matches!(e, OpError::Param { .. }), "{e}");
    // 3 query heads on 2 KV heads.
    let l2 = layout(2, 1, 2, DType::F32);
    let mut out = [SENTINEL; 3];
    let e = attn_decode(
        &Gqa {
            n_q_heads: 3,
            scale: None,
        },
        &l2,
        None,
        &[1.0; 3],
        &rows(0, 0, 1),
        1,
        &seq(&[0], 1),
        &pool(&l2, 1),
        &mut out,
    )
    .unwrap_err();
    assert!(matches!(e, OpError::Param { .. }), "{e}");
    // An FP8 cache is not defined here.
    let e = KvLayout::new(1, 2, 2, DType::F8E4M3).unwrap_err();
    assert!(matches!(e, OpError::Unsupported { .. }), "{e}");
    // KvWrite into a page past the pool writes nothing.
    let mut p2 = p.clone();
    let e = kv_write(
        &l,
        None,
        &[1.0; 2],
        &[1.0; 2],
        &rows(0, 0, 1),
        1,
        &seq(&[5], 1),
        &mut p2,
    )
    .unwrap_err();
    assert!(pos_err(e));
    assert_eq!(p2, p);
}

/// A random sequence: K and V for `n` positions, `n_kv` heads of `d` values, in [-2, 2].
fn kv_values(n: usize, n_kv: usize, d: usize) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    let len = n * n_kv * d;
    (
        prop::collection::vec(-2.0f32..2.0, len),
        prop::collection::vec(-2.0f32..2.0, len),
    )
}

const N_KV: usize = 2;
const N_Q: usize = 4;
const D: usize = 4;
const P: usize = 3;
const N: usize = 7;
const GQA: Gqa = Gqa {
    n_q_heads: N_Q,
    scale: None,
};
/// Three pages, out of order, for up to 9 positions.
const PAGES: [u32; 3] = [2, 0, 3];

fn setup(k: &[f32], v: &[f32], dtype: DType) -> (KvLayout, Vec<u8>) {
    let l = layout(N_KV, D, P, dtype);
    let mut p = pool(&l, 4);
    fill(&l, Some(&plain_rope(D)), k, v, &PAGES, &mut p);
    (l, p)
}

fn dtype() -> impl Strategy<Value = DType> {
    prop_oneof![Just(DType::F32), Just(DType::F16), Just(DType::BF16)]
}

proptest! {
    #[test]
    fn rows_active_zero_reads_and_writes_nothing(
        q in prop::collection::vec(any::<f32>(), 0..32),
    ) {
        // No rows and no sequences: nothing is read or written, whatever the buffers hold.
        let l = layout(N_KV, D, P, DType::F16);
        let mut p = vec![FILL; l.page_bytes()];
        kv_write(&l, None, &q, &q, &[], 0, &[], &mut p).unwrap();
        prop_assert!(p.iter().all(|&b| b == FILL));
        let mut out = vec![SENTINEL; 32];
        attn_decode(&GQA, &l, None, &q, &[], 0, &[], &p, &mut out).unwrap();
        attn_prefill(&GQA, &l, None, &q, &[], 0, &[], &p, &mut out).unwrap();
        prop_assert!(out.iter().all(|&v| v == SENTINEL));
    }

    #[test]
    fn page_size_and_order_do_not_change_the_output(
        (k, v) in kv_values(N, N_KV, D),
        q in prop::collection::vec(-2.0f32..2.0, 2 * N_Q * D),
        dt in dtype(),
    ) {
        // The same KV stored with 3-position pages out of order and with one 8-position page.
        let t = plain_rope(D);
        let (l, p) = setup(&k, &v, dt);
        let l8 = layout(N_KV, D, 8, dt);
        let mut p8 = pool(&l8, 2);
        fill(&l8, Some(&t), &k, &v, &[1], &mut p8);
        let rs = rows(0, 5, 2);
        let a = attend_seq(&GQA, &l, Some(&t), &q, &rs, &PAGES, N as u32, &p);
        let b = attend_seq(&GQA, &l8, Some(&t), &q, &rs, &[1], N as u32, &p8);
        prop_assert_eq!(bits(&a), bits(&b));
        prop_assert!(a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn positions_after_the_row_are_masked(
        (k, v) in kv_values(N, N_KV, D),
        q in prop::collection::vec(-2.0f32..2.0, N_Q * D),
        pos in 0u32..(N as u32 - 1),
        dt in dtype(),
    ) {
        // The causal mask: the sequence has N positions, the row sits at `pos`, and every
        // later position holds NaN in K and V. The output is the clean one, and finite.
        let t = plain_rope(D);
        let (l, clean) = setup(&k, &v, dt);
        let mut dirty = clean.clone();
        let w = N_KV * D;
        let (mut kn, mut vn) = (k.clone(), v.clone());
        kn[(pos as usize + 1) * w..].fill(f32::NAN);
        vn[(pos as usize + 1) * w..].fill(f32::NAN);
        fill(&l, Some(&t), &kn, &vn, &PAGES, &mut dirty);
        let rs = rows(0, pos, 1);
        let a = attend_seq(&GQA, &l, Some(&t), &q, &rs, &PAGES, N as u32, &clean);
        let b = attend_seq(&GQA, &l, Some(&t), &q, &rs, &PAGES, N as u32, &dirty);
        prop_assert_eq!(bits(&a), bits(&b));
        prop_assert!(a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn slots_past_the_kv_length_are_masked(
        (k, v) in kv_values(N, N_KV, D),
        q in prop::collection::vec(-2.0f32..2.0, N_Q * D),
        garbage in any::<u8>(),
        dt in dtype(),
    ) {
        // The sequence has N = 7 positions in three 3-position pages, so slots 7 and 8 of the
        // last page are past its KV length. Filled with NaN or any byte, they change nothing.
        let t = plain_rope(D);
        let (l, clean) = setup(&k, &v, dt);
        let mut dirty = clean.clone();
        let nan = vec![f32::NAN; 2 * N_KV * D];
        let seqs = [Seq { pages: &PAGES, kv_len: 9 }];
        kv_write(&l, None, &nan, &nan, &rows(0, 7, 2), 2, &seqs, &mut dirty).unwrap();
        let mut bytes = clean.clone();
        let last = PAGES[2] as usize * l.page_bytes();
        for (i, b) in bytes[last..last + l.page_bytes()].iter_mut().enumerate() {
            // Slot 2 of every plane and head (positions 8) gets the garbage byte.
            let elem = i / l.elem_bytes();
            if elem % (P * D) >= 2 * D {
                *b = garbage;
            }
        }
        let rs = rows(0, 6, 1);
        let a = attend_seq(&GQA, &l, Some(&t), &q, &rs, &PAGES, N as u32, &clean);
        let b = attend_seq(&GQA, &l, Some(&t), &q, &rs, &PAGES, N as u32, &dirty);
        let c = attend_seq(&GQA, &l, Some(&t), &q, &rs, &PAGES, N as u32, &bytes);
        prop_assert_eq!(bits(&a), bits(&b));
        prop_assert_eq!(bits(&a), bits(&c));
        prop_assert!(a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn nan_propagates_to_the_heads_that_read_it(
        (k, v) in kv_values(N, N_KV, D),
        q in prop::collection::vec(-2.0f32..2.0, N_Q * D),
        j in 0usize..N,
        kvh in 0usize..N_KV,
        dd in 0usize..D,
        qh in 0usize..N_Q,
        dt in dtype(),
    ) {
        let t = plain_rope(D);
        let rs = rows(0, N as u32 - 1, 1);
        let group = N_Q / N_KV;
        let at = j * N_KV * D + kvh * D + dd;
        // A NaN in a read key: every output of the query heads on that KV head is NaN.
        let mut kn = k.clone();
        kn[at] = f32::NAN;
        let (l, p) = setup(&kn, &v, dt);
        let out = attend_seq(&GQA, &l, Some(&t), &q, &rs, &PAGES, N as u32, &p);
        for h in 0..N_Q {
            let nan = h / group == kvh;
            prop_assert!(out[h * D..(h + 1) * D].iter().all(|x| x.is_nan() == nan));
        }
        // A NaN in a read value: that dim of those heads only.
        let mut vn = v.clone();
        vn[at] = f32::NAN;
        let (l, p) = setup(&k, &vn, dt);
        let out = attend_seq(&GQA, &l, Some(&t), &q, &rs, &PAGES, N as u32, &p);
        for h in 0..N_Q {
            for e in 0..D {
                prop_assert_eq!(out[h * D + e].is_nan(), h / group == kvh && e == dd);
            }
        }
        // A NaN in the query: that head only.
        let (l, p) = setup(&k, &v, dt);
        let mut qn = q.clone();
        qn[qh * D + dd] = f32::NAN;
        let out = attend_seq(&GQA, &l, Some(&t), &qn, &rs, &PAGES, N as u32, &p);
        for h in 0..N_Q {
            prop_assert!(out[h * D..(h + 1) * D].iter().all(|x| x.is_nan() == (h == qh)));
        }
    }

    #[test]
    fn rows_of_two_sequences(
        (k, v) in kv_values(N, N_KV, D),
        q in prop::collection::vec(-2.0f32..2.0, 3 * N_Q * D),
    ) {
        // Two sequences in one pool; each row's output equals attending its sequence alone.
        let t = plain_rope(D);
        let l = layout(N_KV, D, P, DType::F16);
        let mut p = pool(&l, 6);
        fill(&l, Some(&t), &k, &v, &PAGES, &mut p);
        let other: [u32; 2] = [5, 1];
        let k2: Vec<f32> = k[..4 * N_KV * D].iter().map(|x| -x).collect();
        let v2: Vec<f32> = v[..4 * N_KV * D].iter().map(|x| x * 0.5).collect();
        fill(&l, Some(&t), &k2, &v2, &other, &mut p);
        let seqs = [Seq { pages: &PAGES, kv_len: N as u32 }, Seq { pages: &other, kv_len: 4 }];
        let rs = [rows(0, 6, 1)[0], rows(1, 3, 1)[0], rows(0, 2, 1)[0]];
        let width = N_Q * D;
        let mut out = vec![SENTINEL; 3 * width];
        attn_prefill(&GQA, &l, Some(&t), &q, &rs, 3, &seqs, &p, &mut out).unwrap();
        let a = attend_seq(&GQA, &l, Some(&t), &q[..width], &rows(0, 6, 1), &PAGES, N as u32, &p);
        let b = attend_seq(&GQA, &l, Some(&t), &q[width..2 * width], &rows(0, 3, 1), &other, 4, &p);
        let c = attend_seq(&GQA, &l, Some(&t), &q[2 * width..], &rows(0, 2, 1), &PAGES, N as u32, &p);
        prop_assert_eq!(bits(&out), bits(&[a, b, c].concat()));
    }
}
