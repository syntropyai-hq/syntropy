// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for the sampling ops, their tie rule and failure markers, and property
//! tests for `rows_active = 0`, the grammar mask and NaN propagation.

use proptest::prelude::*;

use super::*;

const SENTINEL: f32 = -12345.0;
const NEG: f32 = f32::NEG_INFINITY;

/// One row processed with `p`.
fn process(p: Process<'_>, x: &[f32]) -> Vec<f32> {
    let mut y = vec![SENTINEL; x.len()];
    logits_process(&[p], x.len(), x, 1, &mut y).unwrap();
    y
}

#[test]
fn logits_process_by_hand() {
    let x = [2.0, -1.0, 0.5, 4.0, 3.0];
    // Bias: two entries for token 1 are both added, in order.
    let p = Process {
        bias: &[(1, 0.5), (4, -3.0), (1, 0.25)],
        ..Process::NONE
    };
    assert_eq!(process(p, &x), [2.0, -0.25, 0.5, 4.0, 0.0]);
    // Repetition 2 on tokens 0 (positive, halved) and 1 (negative, doubled); frequency 0.5
    // and presence 1 on token 3 seen twice: (4 − 2·0.5) − 1 = 2.
    let context = [1, 3, 0, 0, 0];
    let emitted = [0, 0, 0, 2, 0];
    let pen = Penalties {
        repetition: 2.0,
        presence: 1.0,
        frequency: 0.5,
        context: &context,
        emitted: &emitted,
    };
    let p = Process {
        penalties: Some(pen),
        ..Process::NONE
    };
    assert_eq!(process(p, &x), [1.0, -2.0, 0.5, 2.0, 3.0]);
    // The mask allows tokens 0, 2 and 4 (bits 0b10101).
    let mask = [0b10101u32];
    let p = Process {
        mask: Some(&mask),
        ..Process::NONE
    };
    assert_eq!(process(p, &x), [2.0, NEG, 0.5, NEG, 3.0]);
    // Temperature 0.5 doubles every value; 0 (greedy) leaves them.
    let p = Process {
        temperature: 0.5,
        ..Process::NONE
    };
    assert_eq!(process(p, &x), [4.0, -2.0, 1.0, 8.0, 6.0]);
    let p = Process {
        temperature: 0.0,
        ..Process::NONE
    };
    assert_eq!(process(p, &x), x);
    // Everything at once, in the documented order: bias, repetition, penalties, mask,
    // temperature. Token 0: (2 + 2) / 2 = 2, then / 0.5 = 4.
    let p = Process {
        temperature: 0.5,
        penalties: Some(pen),
        bias: &[(0, 2.0)],
        mask: Some(&[0b11101]),
    };
    assert_eq!(process(p, &x), [4.0, NEG, 1.0, 4.0, 6.0]);
}

#[test]
fn logits_process_refusals_write_nothing() {
    let x = [1.0f32; 4];
    let mut y = [SENTINEL; 4];
    for (p, want) in [
        (
            Process {
                temperature: -1.0,
                ..Process::NONE
            },
            "invalid",
        ),
        (
            Process {
                temperature: f32::NAN,
                ..Process::NONE
            },
            "invalid",
        ),
        (
            Process {
                bias: &[(4, 1.0)],
                ..Process::NONE
            },
            "range",
        ),
        (
            Process {
                mask: Some(&[]),
                ..Process::NONE
            },
            "len",
        ),
    ] {
        let e = logits_process(&[p], 4, &x, 1, &mut y).unwrap_err();
        let ok = match want {
            "invalid" => matches!(e, OpError::Invalid { .. }),
            "range" => matches!(e, OpError::RowOutOfRange { id: 4, rows: 4 }),
            _ => matches!(e, OpError::Len { arg: "mask", .. }),
        };
        assert!(ok, "{e:?}");
    }
    let counts = [0u32; 3];
    let pen = Penalties {
        repetition: 0.0,
        presence: 0.0,
        frequency: 0.0,
        context: &counts,
        emitted: &counts,
    };
    let p = Process {
        penalties: Some(pen),
        ..Process::NONE
    };
    assert!(matches!(
        logits_process(&[p], 4, &x, 1, &mut y),
        Err(OpError::Invalid { .. })
    ));
    let pen = Penalties {
        repetition: 1.0,
        ..pen
    };
    let p = Process {
        penalties: Some(pen),
        ..Process::NONE
    };
    assert!(matches!(
        logits_process(&[p], 4, &x, 1, &mut y),
        Err(OpError::Len { arg: "context", .. })
    ));
    assert_eq!(y, [SENTINEL; 4]);
}

/// One row filtered with `f`.
fn filter(f: Filter, x: &[f32]) -> Vec<f32> {
    let mut y = vec![SENTINEL; x.len()];
    top_k_top_p(&[f], x.len(), x, 1, &mut y).unwrap();
    y
}

#[test]
fn top_k_top_p_by_hand() {
    let x = [1.0, 3.0, 3.0, 0.0, 2.0];
    // Top-2 of a tie for first: both 3s survive; top-1 keeps the lower ID.
    let k2 = Filter {
        top_k: 2,
        ..Filter::NONE
    };
    assert_eq!(filter(k2, &x), [NEG, 3.0, 3.0, NEG, NEG]);
    let k1 = Filter {
        top_k: 1,
        ..Filter::NONE
    };
    assert_eq!(filter(k1, &x), [NEG, 3.0, NEG, NEG, NEG]);
    // k at or above the vocabulary, and 0, are off.
    for top_k in [0, 5, 9] {
        let f = Filter {
            top_k,
            ..Filter::NONE
        };
        assert_eq!(filter(f, &x), x);
    }
    // ln 3 and 0: probabilities 0.75 and 0.25.
    let y = [3f32.ln(), 0.0];
    let p = |top_p| Filter {
        top_p,
        ..Filter::NONE
    };
    assert_eq!(filter(p(0.7), &y), [y[0], NEG]);
    assert_eq!(filter(p(0.8), &y), y);
    // Top-p never removes everything.
    assert_eq!(filter(p(0.0), &y), [y[0], NEG]);
    // Min-p 0.5: 0.25 < 0.5 · 0.75 is removed; min-p 0.3 keeps it.
    let mp = |min_p| Filter {
        min_p,
        ..Filter::NONE
    };
    assert_eq!(filter(mp(0.5), &y), [y[0], NEG]);
    assert_eq!(filter(mp(0.3), &y), y);
    // The filters intersect: top-k 3 then top-p over the three survivors.
    let z = [0.0, 10.0, 9.0, 8.0, -1.0];
    let f = Filter {
        top_k: 3,
        top_p: 0.5,
        min_p: 0.0,
    };
    assert_eq!(filter(f, &z), [NEG, 10.0, NEG, NEG, NEG]);
    // NaN and +∞ make the row NaN; an all −∞ row stays.
    let out = filter(Filter::NONE, &[1.0, f32::NAN]);
    assert!(out.iter().all(|v| v.is_nan()));
    let out = filter(k1, &[1.0, f32::INFINITY]);
    assert!(out.iter().all(|v| v.is_nan()));
    assert_eq!(filter(k1, &[NEG, NEG]), [NEG, NEG]);
    // A NaN top_p and a min_p outside [0, 1] are refused.
    let mut buf = [0.0; 2];
    assert!(top_k_top_p(&[p(f32::NAN)], 2, &y, 1, &mut buf).is_err());
    assert!(top_k_top_p(&[mp(1.5)], 2, &y, 1, &mut buf).is_err());
}

fn sample(x: &[f32], u: f32) -> u32 {
    let mut out = [7u32];
    sample_seeded(x.len(), x, &[u], 1, &mut out).unwrap();
    out[0]
}

#[test]
fn sample_seeded_by_hand() {
    // ln 3 and 0: masses 1 and 1/3, Z = 4/3. u < 0.75 picks token 0.
    let x = [3f32.ln(), 0.0];
    assert_eq!(sample(&x, 0.0), 0);
    assert_eq!(sample(&x, 0.7), 0);
    assert_eq!(sample(&x, 0.8), 1);
    assert_eq!(sample(&x, 0.999_999_9), 1);
    // A masked token is never drawn, and a lone survivor always is.
    let m = [NEG, 5.0, NEG];
    for u in [0.0, 0.5, 0.999_999_9] {
        assert_eq!(sample(&m, u), 1);
    }
    // Equal logits: the uniform draw walks the IDs in order.
    let e = [1.0; 4];
    assert_eq!(sample(&e, 0.1), 0);
    assert_eq!(sample(&e, 0.3), 1);
    assert_eq!(sample(&e, 0.6), 2);
    assert_eq!(sample(&e, 0.9), 3);
    // Markers.
    assert_eq!(sample(&[1.0, f32::NAN], 0.5), NAN_ROW);
    assert_eq!(sample(&[1.0, f32::INFINITY], 0.5), NAN_ROW);
    assert_eq!(sample(&[NEG, NEG], 0.5), EMPTY_ROW);
    // Draws outside [0, 1) are refused.
    let mut out = [0u32];
    for u in [1.0, -0.1, f32::NAN] {
        assert!(matches!(
            sample_seeded(2, &x, &[u], 1, &mut out),
            Err(OpError::Invalid { .. })
        ));
    }
}

#[test]
fn argmax_by_hand() {
    let x = [
        1.0,
        5.0,
        5.0,
        NEG,
        NEG,
        NEG,
        1.0,
        f32::NAN,
        0.0,
        0.0,
        f32::INFINITY,
        2.0,
    ];
    let mut out = [0u32; 4];
    argmax(3, &x, 4, &mut out).unwrap();
    // Ties to the lower ID; all −∞ is empty; NaN is the NaN marker; +∞ is a value.
    assert_eq!(out, [1, EMPTY_ROW, NAN_ROW, 1]);
}

#[test]
fn verify_accept_greedy() {
    // Rows' argmaxes: 2, 0, 1.
    let rows = [0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 0.0, 9.0, 0.0];
    let g = |d: &[u32]| verify_accept(Acceptance::Greedy, 3, &rows, d, &[]).unwrap();
    assert_eq!(
        g(&[2, 0]),
        Verdict {
            accepted: 2,
            next: 1,
            draws: 0
        }
    );
    assert_eq!(
        g(&[2, 1]),
        Verdict {
            accepted: 1,
            next: 0,
            draws: 0
        }
    );
    assert_eq!(
        g(&[1, 0]),
        Verdict {
            accepted: 0,
            next: 2,
            draws: 0
        }
    );
    assert_eq!(
        g(&[]),
        Verdict {
            accepted: 0,
            next: 2,
            draws: 0
        }
    );
    // A NaN row stops the chain with the marker.
    let bad = [0.0, 1.0, 2.0, f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(
        verify_accept(Acceptance::Greedy, 3, &bad, &[2, 0], &[]).unwrap(),
        Verdict {
            accepted: 1,
            next: NAN_ROW,
            draws: 0
        }
    );
}

#[test]
fn verify_accept_sampled() {
    // Row 0: p = [0.75, 0.25] (ln 3, 0). Row 1: p = [0.25, 0.75]. Row 2: bonus row, all on 1.
    let l3 = 3f32.ln();
    let rows = [l3, 0.0, 0.0, l3, NEG, 0.0];
    let s = |d: &[u32], u: &[f32]| verify_accept(Acceptance::Sampled, 2, &rows, d, u);
    // Both accepted (0.5 < 0.75 twice), then the bonus draw.
    assert_eq!(
        s(&[0, 1], &[0.5, 0.5, 0.3]).unwrap(),
        Verdict {
            accepted: 2,
            next: 1,
            draws: 3
        }
    );
    // Draft 1 at row 0 has p = 0.25: u = 0.5 rejects it, and the residual is token 0 alone.
    assert_eq!(
        s(&[1, 1], &[0.5, 0.9]).unwrap(),
        Verdict {
            accepted: 0,
            next: 0,
            draws: 2
        }
    );
    // Accept at row 0, reject at row 1 (u = 0.8 ≥ 0.75): the residual is token 0.
    assert_eq!(
        s(&[0, 1], &[0.1, 0.8, 0.0]).unwrap(),
        Verdict {
            accepted: 1,
            next: 0,
            draws: 3
        }
    );
    // Running out of draws is refused, and so is a draft past the vocabulary.
    assert!(matches!(
        s(&[0, 1], &[0.1, 0.1]),
        Err(OpError::Len { arg: "u", .. })
    ));
    assert!(matches!(
        s(&[0, 2], &[0.1; 3]),
        Err(OpError::RowOutOfRange { id: 2, rows: 2 })
    ));
}

/// Rows of values with NaN, infinities and repeats drawn from a small set, so ties happen.
fn rows(n: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(
        prop_oneof![
            8 => (-4i32..4).prop_map(|v| v as f32 * 0.5),
            1 => Just(f32::NEG_INFINITY),
        ],
        n,
    )
}

proptest! {
    /// `rows_active = 0` writes nothing, whatever the buffers hold.
    #[test]
    fn rows_active_zero_writes_nothing(vocab in 1usize..8, x in rows(16)) {
        let x = &x[..vocab];
        let mut y = vec![SENTINEL; vocab];
        logits_process(&[], vocab, x, 0, &mut y).unwrap();
        top_k_top_p(&[], vocab, x, 0, &mut y).unwrap();
        prop_assert!(y.iter().all(|&v| v == SENTINEL));
        let mut out = [7u32; 2];
        sample_seeded(vocab, x, &[], 0, &mut out).unwrap();
        argmax(vocab, x, 0, &mut out).unwrap();
        prop_assert_eq!(out, [7, 7]);
    }

    /// Rows past `rows_active` are neither read nor written.
    #[test]
    fn padded_rows_are_untouched(vocab in 1usize..8, x in rows(8), u in 0.0f32..1.0) {
        let mut xs = x[..vocab].to_vec();
        xs.extend(std::iter::repeat_n(f32::NAN, vocab));
        let mut y = vec![SENTINEL; 2 * vocab];
        logits_process(&[Process::NONE], vocab, &xs, 1, &mut y).unwrap();
        prop_assert!(y[vocab..].iter().all(|&v| v == SENTINEL));
        let mut out = [7u32; 2];
        sample_seeded(vocab, &xs, &[u], 1, &mut out).unwrap();
        prop_assert_eq!(out[1], 7);
        prop_assert_ne!(out[0], NAN_ROW);
    }

    /// The grammar mask: a masked token is never sampled, never the argmax, and never survives
    /// `TopKTopP`; a fully masked row gives the empty marker.
    #[test]
    fn masked_tokens_are_never_chosen(x in rows(40), mask in any::<u64>(), u in 0.0f32..1.0) {
        let vocab = 40;
        let words = [mask as u32, (mask >> 32) as u32];
        let allowed = |i: usize| words[i / 32] & (1 << (i % 32)) != 0;
        let p = Process { mask: Some(&words), ..Process::NONE };
        let y = process(p, &x);
        let t = filter(Filter { top_k: 5, top_p: 0.9, min_p: 0.0 }, &y);
        for i in 0..vocab {
            if !allowed(i) {
                prop_assert_eq!(y[i], NEG);
                prop_assert_eq!(t[i], NEG);
            }
        }
        let any_left = (0..vocab).any(|i| allowed(i) && x[i] > NEG);
        for (tok, what) in [(sample(&t, u), "sample"), ({
            let mut o = [0u32];
            argmax(vocab, &y, 1, &mut o).unwrap();
            o[0]
        }, "argmax")] {
            if any_left {
                prop_assert!((tok as usize) < vocab && allowed(tok as usize), "{what} {tok}");
            } else {
                prop_assert_eq!(tok, EMPTY_ROW, "{}", what);
            }
        }
    }

    /// A NaN anywhere in a row survives every op and ends as the NaN marker, even on a masked
    /// token.
    #[test]
    fn nan_propagates(x in rows(12), at in 0usize..12, u in 0.0f32..1.0) {
        let mut x = x;
        x[at] = f32::NAN;
        let mask = [0u32];
        let p = Process { mask: Some(&mask), temperature: 0.7, ..Process::NONE };
        let y = process(p, &x);
        prop_assert!(y[at].is_nan());
        let t = filter(Filter { top_k: 3, top_p: 0.5, min_p: 0.1 }, &y);
        prop_assert!(t.iter().all(|v| v.is_nan()));
        prop_assert_eq!(sample(&t, u), NAN_ROW);
        let mut o = [0u32];
        argmax(12, &y, 1, &mut o).unwrap();
        prop_assert_eq!(o[0], NAN_ROW);
    }

    /// Ties go to the lower token ID: on a row of equal maxima, argmax and top-1 pick the first.
    #[test]
    fn ties_go_to_the_lower_id(vocab in 2usize..20, first in 0usize..20, v in -3.0f32..3.0) {
        let first = first % vocab;
        let mut x = vec![v - 1.0; vocab];
        for xi in &mut x[first..] {
            *xi = v;
        }
        let mut o = [0u32];
        argmax(vocab, &x, 1, &mut o).unwrap();
        prop_assert_eq!(o[0] as usize, first);
        let t = filter(Filter { top_k: 1, ..Filter::NONE }, &x);
        prop_assert_eq!(t.iter().position(|&y| y > NEG), Some(first));
    }

    /// Sampling picks only tokens with positive mass, and is monotone in the draw.
    #[test]
    fn sampling_is_monotone_in_the_draw(x in rows(16), u in 0.0f32..1.0, w in 0.0f32..1.0) {
        let (a, b) = (sample(&x, u.min(w)), sample(&x, u.max(w)));
        if x.iter().all(|&v| v == NEG) {
            prop_assert_eq!((a, b), (EMPTY_ROW, EMPTY_ROW));
        } else {
            prop_assert!(a <= b);
            prop_assert!(x[a as usize] > NEG && x[b as usize] > NEG);
        }
    }

    /// `VerifyAccept` consumes draws in RFC-0012 §8.2's order: one per draft tried, then one
    /// for the residual or bonus; greedy takes none and stops at the first draft that is not
    /// its row's argmax.
    #[test]
    fn verify_accept_draw_count(x in rows(15), d in prop::collection::vec(0u32..3, 4), u in prop::collection::vec(0.0f32..1.0, 6)) {
        let drafts = &d[..];
        let v = verify_accept(Acceptance::Sampled, 3, &x, &drafts[..4], &u).unwrap();
        let want = if v.accepted == 4 { 5 } else { v.accepted + 2 };
        prop_assert_eq!(v.draws, want);
        let g = verify_accept(Acceptance::Greedy, 3, &x, drafts, &[]).unwrap();
        prop_assert_eq!(g.draws, 0);
        for i in 0..g.accepted {
            prop_assert_eq!(argmax_of(&x[3 * i..3 * i + 3]), drafts[i]);
        }
        prop_assert_eq!(g.next, argmax_of(&x[3 * g.accepted..3 * g.accepted + 3]));
    }
}
