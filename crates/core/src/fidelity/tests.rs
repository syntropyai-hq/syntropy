// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests for the fidelity metrics (COMP-E14-T01's acceptance criteria).

use super::*;

use proptest::prelude::*;

/// F64 rounding allowance for comparisons between quantities computed through different
/// operation orders. The vocabularies here have at most a few thousand entries and logits
/// within ±40, so each sum's rounding error is below `V · 2⁻⁵² · 80 ≈ 4e−11` in the worst
/// case and far smaller with compensated summation. Test parameter.
const ROUNDING: f64 = 1e-10;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// SplitMix64: a small deterministic generator for the tail draws, so the test needs no RNG
/// crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 53 random bits.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Builds KL_K's inputs from full logits with no tail sample.
fn kl_k_from_full(r: &[f32], c: &[f32], k: usize) -> f64 {
    let sr = StoredRef::from_logits(r, k, &[]).unwrap();
    let gc = GatheredCand::from_logits(c, &sr.view()).unwrap();
    kl_k(&sr.view(), &gc.view()).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Criterion 1: unit tests on known distributions
// ---------------------------------------------------------------------------------------------

#[test]
fn rel_l2_and_max_abs_scaled_known_values() {
    // r = (3, 4), y − r = (0, 5): ‖y − r‖ = 5, ‖r‖ = 5.
    let r = [3.0f64, 4.0];
    let y = [3.0f32, 9.0];
    assert_eq!(rel_l2(&y, &r).unwrap(), 1.0);
    // max |y − r| = 5, rms(r) = √12.5.
    let expect = 5.0 / (12.5f64.sqrt() + 1e-30);
    assert_eq!(max_abs_scaled(&y, &r).unwrap(), expect);
    // Identical inputs.
    assert_eq!(rel_l2(&r, &r).unwrap(), 0.0);
    assert_eq!(max_abs_scaled(&r, &r).unwrap(), 0.0);
}

#[test]
fn rel_l2_zero_reference() {
    assert_eq!(rel_l2(&[0.0f32, 0.0], &[0.0f32, 0.0]).unwrap(), 0.0);
    assert_eq!(
        rel_l2(&[0.0f32, 1e-3], &[0.0f32, 0.0]).unwrap(),
        f64::INFINITY
    );
    // max_abs_scaled divides by rms + 1e-30.
    assert_eq!(
        max_abs_scaled(&[0.0f32, 1.0], &[0.0f32, 0.0]).unwrap(),
        1.0 / 1e-30
    );
}

#[test]
fn op_metrics_reject_bad_inputs() {
    assert_eq!(
        rel_l2(&[1.0f32], &[1.0f32, 2.0]),
        Err(MetricError::LengthMismatch {
            expected: 2,
            found: 1
        })
    );
    assert_eq!(
        rel_l2::<f32, f32>(&[], &[]),
        Err(MetricError::Empty { what: "y" })
    );
    assert_eq!(
        max_abs_scaled(&[f32::NAN], &[1.0f32]),
        Err(MetricError::NonFinite {
            what: "y",
            index: Some(0)
        })
    );
}

#[test]
fn kl_two_token_closed_form() {
    // P_ref = (1/4, 3/4), P_cand = (1/2, 1/2).
    let r = [0.0f64, 3.0f64.ln()];
    let c = [0.0f64, 0.0];
    let expect = 0.25 * (0.25f64 / 0.5).ln() + 0.75 * (0.75f64 / 0.5).ln();
    assert!(close(kl_full(&r, &c).unwrap(), expect, 1e-15));
}

#[test]
fn kl_four_token_closed_form_and_shift_invariance() {
    // P_ref uniform over 4; P_cand = (1, 1, 1, 2) / 5.
    let r = [0.0f64; 4];
    let c = [0.0f64, 0.0, 0.0, 2.0f64.ln()];
    let expect = 0.75 * (1.25f64).ln() + 0.25 * (0.625f64).ln();
    assert!(close(kl_full(&r, &c).unwrap(), expect, 1e-15));
    // Adding a constant to either row leaves both distributions unchanged.
    let c_shift: Vec<f64> = c.iter().map(|v| v + 17.5).collect();
    let r_shift: Vec<f64> = r.iter().map(|v| v - 3.25).collect();
    assert!(close(kl_full(&r_shift, &c_shift).unwrap(), expect, 1e-14));
    // Equal rows.
    assert!(close(kl_full(&c, &c).unwrap(), 0.0, 1e-16));
}

#[test]
fn kl_handles_underflowing_reference_probability() {
    // exp(−800) underflows to 0 in F64; the term contributes 0, the result stays finite.
    let r = [0.0f64, -800.0];
    let c = [0.0f64, 0.0];
    let expect = (2.0f64).ln(); // P_ref ≈ (1, 0), P_cand = (1/2, 1/2)
    assert!(close(kl_full(&r, &c).unwrap(), expect, 1e-15));
}

#[test]
fn kl_rejects_non_finite_logits() {
    assert_eq!(
        kl_full(&[0.0f32, f32::NEG_INFINITY], &[0.0f32, 0.0]),
        Err(MetricError::NonFinite {
            what: "logits",
            index: Some(1)
        })
    );
}

#[test]
fn argmax_ties_to_lower_id() {
    assert_eq!(argmax(&[1.0f32, 3.0, 3.0, 2.0]).unwrap(), 1);
    assert_eq!(argmax(&[5.0f32, 5.0]).unwrap(), 0);
    assert!(top1_agrees(&[1.0f32, 3.0, 3.0], &[0.0f32, 2.0, 2.0]).unwrap());
    assert!(!top1_agrees(&[1.0f32, 3.0, 3.0], &[0.0f32, 2.0, 2.5]).unwrap());
}

#[test]
fn stored_ref_orders_top_k_and_computes_log_r() {
    let z = [1.0f32, 3.0, 3.0, 0.5, 2.0];
    let sr = StoredRef::from_logits(&z, 3, &[0, 3, 0]).unwrap();
    let ids: Vec<u32> = sr.top_k.iter().map(|e| e.id).collect();
    assert_eq!(ids, vec![1, 2, 4]);
    let total: f64 = z.iter().map(|&v| f64::from(v).exp()).sum();
    let tail: f64 = [z[0], z[3]].iter().map(|&v| f64::from(v).exp()).sum();
    assert!(close(sr.lse, total.ln(), 1e-15));
    assert!(close(sr.log_r, (tail / total).ln(), 1e-15));
    assert_eq!(
        StoredRef::from_logits(&z, 3, &[2]),
        Err(MetricError::TailIdInTopK { id: 2 })
    );
    assert_eq!(
        StoredRef::from_logits(&z, 3, &[5]),
        Err(MetricError::IdOutOfRange { id: 5, vocab: 5 })
    );
    assert_eq!(StoredRef::from_logits(&z, 0, &[]), Err(MetricError::ZeroK));
}

#[test]
fn kl_k_known_coarse_graining() {
    // P_ref = (1/2, 1/4, 1/4), P_cand = (1/4, 1/4, 1/2); K = 1 keeps token 0 and merges the
    // rest: coarse P_ref = (1/2, 1/2), coarse P_cand = (1/4, 3/4).
    let r = [2.0f64.ln() as f32, 0.0, 0.0];
    let c = [0.0f32, 0.0, 2.0f64.ln() as f32];
    let lr = log_softmax(&r).unwrap();
    let lc = log_softmax(&c).unwrap();
    let (pr0, pc0) = (lr[0].exp(), lc[0].exp());
    let expect = pr0 * (pr0 / pc0).ln() + (1.0 - pr0) * ((1.0 - pr0) / (1.0 - pc0)).ln();
    let got = kl_k_from_full(&r, &c, 1);
    assert!(close(got, expect, 1e-12), "{got} vs {expect}");
    assert!(got < kl_full(&r, &c).unwrap());
}

#[test]
fn kl_k_with_whole_vocabulary_is_full_kl_and_tail_is_zero() {
    let r = [0.3f32, -1.0, 2.0, 0.0];
    let c = [0.0f32, 0.5, 1.5, -0.25];
    let sr = StoredRef::from_logits(&r, 10, &[]).unwrap();
    assert_eq!(sr.log_r, f64::NEG_INFINITY);
    let gc = GatheredCand::from_logits(&c, &sr.view()).unwrap();
    assert_eq!(gc.log_s, f64::NEG_INFINITY);
    let k = kl_k(&sr.view(), &gc.view()).unwrap();
    assert!(close(k, kl_full(&r, &c).unwrap(), ROUNDING));
    assert_eq!(kl_tail(&sr.view(), &gc.view()).unwrap(), 0.0);
}

#[test]
fn kl_tail_single_tail_token_is_exact() {
    // V∖T has one token, so every draw is that token and KL_K + KL_tail = KL exactly.
    let r = [1.0f32, 0.5, -0.5];
    let c = [0.25f32, 1.0, 0.0];
    let sr = StoredRef::from_logits(&r, 2, &[2, 2, 2]).unwrap();
    let gc = GatheredCand::from_logits(&c, &sr.view()).unwrap();
    let sum = kl_k(&sr.view(), &gc.view()).unwrap() + kl_tail(&sr.view(), &gc.view()).unwrap();
    assert!(close(sum, kl_full(&r, &c).unwrap(), ROUNDING));
}

#[test]
fn stored_reference_estimators_reject_harness_errors() {
    let top = [IdLogit { id: 0, logit: 0.0 }];
    let tail = [IdLogit { id: 1, logit: -1.0 }];
    let reference = RefView {
        top_k: &top,
        tail_sample: &tail,
        lse: 0.5,
        log_r: -1.0,
    };
    let cand = |log_s| CandView {
        top_k: &[0.0],
        tail_sample: &[-1.0],
        lse: 0.5,
        log_s,
    };
    assert_eq!(
        kl_k(&reference, &cand(f64::NAN)),
        Err(MetricError::NonFinite {
            what: "log_s",
            index: None
        })
    );
    assert_eq!(
        kl_k(&reference, &cand(f64::NEG_INFINITY)),
        Err(MetricError::TailBucketMismatch)
    );
    let no_tail = RefView {
        tail_sample: &[],
        ..reference
    };
    let cand_no_tail = CandView {
        tail_sample: &[],
        ..cand(-1.0)
    };
    assert_eq!(
        kl_tail(&no_tail, &cand_no_tail),
        Err(MetricError::EmptyTailSample)
    );
    let short = CandView {
        top_k: &[],
        ..cand(-1.0)
    };
    assert_eq!(
        kl_k(&reference, &short),
        Err(MetricError::LengthMismatch {
            expected: 1,
            found: 0
        })
    );
}

#[test]
fn nearest_rank_percentiles() {
    let v: Vec<f64> = (1..=100).map(f64::from).collect();
    assert_eq!(p99_nearest_rank(&v).unwrap(), 99.0);
    let v: Vec<f64> = (1..=200).rev().map(f64::from).collect();
    assert_eq!(p99_nearest_rank(&v).unwrap(), 198.0);
    let v: Vec<f64> = (1..=50).map(f64::from).collect();
    assert_eq!(p99_nearest_rank(&v).unwrap(), 50.0);
    assert_eq!(p99_nearest_rank(&[7.0]).unwrap(), 7.0);
    assert_eq!(percentile_nearest_rank(&[1.0, 2.0], 50.0).unwrap(), 1.0);
    assert_eq!(
        percentile_nearest_rank(&[1.0], 0.0),
        Err(MetricError::InvalidPercentile)
    );
    assert_eq!(mean(&[1.0, 2.0, 6.0]).unwrap(), 3.0);
    assert_eq!(top1_disagreement(&[true, false, true, true]).unwrap(), 0.25);
}

#[test]
fn pooled_and_per_category_aggregates() {
    let s = |kl, top1| PositionScore {
        kl,
        kl_tail: Some(kl / 10.0),
        top1,
    };
    let rows = vec![
        ("chat", s(0.1, true)),
        ("code", s(0.4, false)),
        ("chat", s(0.3, true)),
        ("code", s(0.2, true)),
    ];
    let a = aggregate_by_category(&rows).unwrap();
    assert_eq!(a.pooled.positions, 4);
    assert!(close(a.pooled.kl_mean, 0.25, 1e-15));
    assert_eq!(a.pooled.kl_p99, 0.4);
    assert_eq!(a.pooled.top1_disagreement, 0.25);
    assert!(close(a.pooled.kl_tail_mean.unwrap(), 0.025, 1e-15));
    let chat = a.by_category["chat"];
    assert!(close(chat.kl_mean, 0.2, 1e-15));
    assert_eq!(chat.kl_p99, 0.3);
    assert_eq!(chat.top1_disagreement, 0.0);
    let code = a.by_category["code"];
    assert_eq!(code.kl_p99, 0.4);
    assert_eq!(code.top1_disagreement, 0.5);
    let mixed = [
        s(0.1, true),
        PositionScore {
            kl_tail: None,
            ..s(0.2, true)
        },
    ];
    assert_eq!(aggregate(&mixed), Err(MetricError::MixedTail));
}

// ---------------------------------------------------------------------------------------------
// Criterion 2: property test that KL_K ≤ KL
// ---------------------------------------------------------------------------------------------

fn logits_and_k() -> impl Strategy<Value = (Vec<f32>, Vec<f32>, usize)> {
    (2usize..96).prop_flat_map(|v| {
        (
            prop::collection::vec(-40.0f32..40.0, v),
            prop::collection::vec(-40.0f32..40.0, v),
            1usize..=v,
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2048, ..ProptestConfig::default() })]

    /// `KL_K` is the KL of the coarse-grained distributions, so by the data-processing
    /// inequality `0 ≤ KL_K ≤ KL` (RFC-0060 §5.4). `ROUNDING` allows for F64 rounding only.
    #[test]
    fn kl_k_is_a_lower_bound((r, c, k) in logits_and_k()) {
        let kl = kl_full(&r, &c).unwrap();
        let klk = kl_k_from_full(&r, &c, k);
        prop_assert!(klk <= kl + ROUNDING, "KL_K {klk} > KL {kl} at K = {k}");
        prop_assert!(klk >= -ROUNDING, "KL_K {klk} < 0");
    }
}

// ---------------------------------------------------------------------------------------------
// Criterion 3: the mean of KL_K + KL_tail over repeated tail samples converges to KL
// ---------------------------------------------------------------------------------------------

/// Test parameters for criterion 3.
const VOCAB: usize = 4096;
const K: usize = 32;
const M: usize = 64;
const REPEATS: usize = 20_000;
const SEED: u64 = 0x5EED_0E14_0001;
/// The run's mean must lie within this many standard errors of the exact KL. The standard
/// error is computed analytically from the full distributions, so the bound is exact for
/// the estimator's true variance; by the central limit theorem a correct estimator falls
/// outside 4 standard errors with probability ≈ 6e−5, and the seed is fixed.
const Z: f64 = 4.0;

/// A Zipf-like reference over `VOCAB` tokens.
fn zipf_reference() -> Vec<f32> {
    (0..VOCAB)
        .map(|i| (-1.1 * ((i + 1) as f64).ln() + 0.3 * ((i * 7) as f64).sin()) as f32)
        .collect()
}

/// Runs the estimator `REPEATS` times and checks the mean against the exact KL.
fn check_convergence(r: &[f32], c: &[f32]) {
    let kl = kl_full(r, c).unwrap();
    let base = StoredRef::from_logits(r, K, &[]).unwrap();
    let mut in_top = vec![false; r.len()];
    for e in &base.top_k {
        in_top[e.id as usize] = true;
    }
    // The tail distribution p_j / r over V∖T, and its CDF for inverse-CDF sampling.
    let lp = log_softmax(r).unwrap();
    let lq = log_softmax(c).unwrap();
    let r_mass = base.log_r.exp();
    let tail_ids: Vec<usize> = (0..r.len()).filter(|&i| !in_top[i]).collect();
    let mut cdf = Vec::with_capacity(tail_ids.len());
    let mut acc = 0.0;
    let (mut e1, mut e2) = (0.0, 0.0);
    for &j in &tail_ids {
        let w = lp[j].exp() / r_mass;
        acc += w;
        cdf.push(acc);
        let d = lp[j] - lq[j];
        e1 += w * d;
        e2 += w * d * d;
    }
    // Var(KL_tail) = r² / M · Var_{j ∝ p}(log p_j − log q_j); KL_K is fixed per position.
    let var = r_mass * r_mass / M as f64 * (e2 - e1 * e1);
    let se = (var / REPEATS as f64).sqrt();

    // The top-K side and lse values do not depend on the draw, so they are built once and
    // only the tail sample changes per repeat.
    let gathered = GatheredCand::from_logits(c, &base.view()).unwrap();
    let klk = kl_k(&base.view(), &gathered.view()).unwrap();
    let mut rng = SplitMix64(SEED);
    let mut sum = 0.0;
    let mut ref_tail = vec![IdLogit { id: 0, logit: 0.0 }; M];
    let mut cand_tail = vec![0.0f32; M];
    for _ in 0..REPEATS {
        for (rt, ct) in ref_tail.iter_mut().zip(cand_tail.iter_mut()) {
            let u = rng.next_f64() * acc;
            let j = tail_ids[cdf.partition_point(|&x| x <= u).min(tail_ids.len() - 1)];
            *rt = IdLogit {
                id: j as u32,
                logit: r[j],
            };
            *ct = c[j];
        }
        let rv = RefView {
            tail_sample: &ref_tail,
            ..base.view()
        };
        let cv = CandView {
            tail_sample: &cand_tail,
            ..gathered.view()
        };
        sum += kl_k(&rv, &cv).unwrap() + kl_tail(&rv, &cv).unwrap();
    }
    let est = sum / REPEATS as f64;
    // KL_K alone must be visibly short of KL, or the test shows nothing.
    assert!(kl - klk > 10.0 * Z * se, "KL {kl}, KL_K {klk}, se {se}");
    assert!(
        (est - kl).abs() <= Z * se,
        "mean {est} vs KL {kl}: |diff| {} > {Z} · se {se}",
        (est - kl).abs()
    );
}

#[test]
fn kl_k_plus_tail_converges_when_candidate_inflates_tail() {
    let r = zipf_reference();
    // The candidate raises every tail-ranked logit by a position-dependent offset.
    let c: Vec<f32> = r
        .iter()
        .enumerate()
        .map(|(i, &z)| {
            let bump = if i >= K {
                0.8 + 0.4 * ((i as f64) * 0.37).cos()
            } else {
                0.0
            };
            (f64::from(z) + bump) as f32
        })
        .collect();
    check_convergence(&r, &c);
}

#[test]
fn kl_k_plus_tail_converges_when_candidate_reshapes_tail() {
    let r = zipf_reference();
    // Within the tail, the candidate's ranking differs from the reference's.
    let c: Vec<f32> = r
        .iter()
        .enumerate()
        .map(|(i, &z)| (f64::from(z) + 1.5 * ((i as f64) * 0.011).sin()) as f32)
        .collect();
    check_convergence(&r, &c);
}

// ---------------------------------------------------------------------------------------------
// Criterion 4: the tightness check either side of its bound, including KL → 0
// ---------------------------------------------------------------------------------------------

#[test]
fn tightness_bound_values() {
    // Relative term dominates.
    assert_eq!(tightness_bound(1.0, 0.5), 0.01);
    // Absolute term dominates as KL → 0.
    assert_eq!(tightness_bound(1e-9, 0.02), 0.01 * 0.02);
    assert_eq!(tightness_bound(0.0, 0.02), 0.01 * 0.02);
}

#[test]
fn tightness_relative_side() {
    // KL = 1, C = 0.5: bound = 0.01.
    let pass = TightnessCheck::new(1.0, 1.0 - 0.0099, 0.5).unwrap();
    assert!(pass.pass, "{pass:?}");
    let fail = TightnessCheck::new(1.0, 1.0 - 0.0101, 0.5).unwrap();
    assert!(!fail.pass, "{fail:?}");
    // The bound is |KL − KL_K|, so a KL_K above KL by too much also fails.
    assert!(!TightnessCheck::new(1.0, 1.0101, 0.5).unwrap().pass);
}

#[test]
fn tightness_absolute_side_as_kl_goes_to_zero() {
    // C_kl_mean = 0.02: absolute bound 2e-4.
    let c = 0.02;
    // KL = 1e-9 with KL_K = 0: relative-only would fail (1e-9 > 1e-11); the absolute term
    // passes it.
    assert!(TightnessCheck::new(1e-9, 0.0, c).unwrap().pass);
    // KL = 1.9e-4, KL_K = 0: just inside.
    assert!(TightnessCheck::new(1.9e-4, 0.0, c).unwrap().pass);
    // KL = 2.1e-4, KL_K = 0: just outside (bound max(2.1e-6, 2e-4) = 2e-4).
    assert!(!TightnessCheck::new(2.1e-4, 0.0, c).unwrap().pass);
    // KL = 0 exactly and C = 0: bound 0, equal values pass, any gap fails.
    assert!(TightnessCheck::new(0.0, 0.0, 0.0).unwrap().pass);
    assert!(!TightnessCheck::new(1e-300, 0.0, 0.0).unwrap().pass);
    assert_eq!(
        TightnessCheck::new(0.0, 0.0, -1.0),
        Err(MetricError::InvalidCeiling)
    );
    assert_eq!(
        TightnessCheck::new(f64::NAN, 0.0, 0.1),
        Err(MetricError::NonFinite {
            what: "kl",
            index: None
        })
    );
}

#[test]
fn tightness_checks_mean_and_p99_separately() {
    // 100 positions. KL_K equals KL everywhere except the top 2 positions, so the means are
    // close but the p99 (rank 99) differs.
    let kl: Vec<f64> = (1..=100).map(|i| f64::from(i) / 100.0).collect();
    let mut kl_k = kl.clone();
    kl_k[98] = 0.5; // the p99 position, KL = 0.99
    kl_k[99] = 0.5; // the maximum, KL = 1.0
    let t = check_tightness(&kl, &kl_k, 0.1).unwrap();
    // p99 KL = 0.99; p99 KL_K = 0.97, rank 99 of the sorted KL_K (0.01 … 0.98 and two 0.5s):
    // diff 0.02 > 0.0099.
    assert_eq!(t.p99.kl, 0.99);
    assert_eq!(t.p99.kl_k, 0.97);
    assert!(!t.p99.pass, "{:?}", t.p99);
    // Mean gap = (0.49 + 0.5) / 100 = 0.0099; bound = max(0.01 · 0.505, 0.001) = 0.00505.
    assert!(!t.mean.pass);
    assert!(!t.pass());

    // Shrinking the gap at those two positions passes both.
    let mut kl_k = kl.clone();
    kl_k[98] = 0.99 - 0.002;
    kl_k[99] = 1.0 - 0.002;
    let t = check_tightness(&kl, &kl_k, 0.1).unwrap();
    assert!(t.pass(), "{t:?}");
}

#[test]
fn tightness_per_category() {
    let mut rows = Vec::new();
    for i in 0..200 {
        let kl = 0.05 + f64::from(i) * 1e-4;
        rows.push(("chat", kl, kl * 0.995)); // within 1 %
        rows.push(("code", kl, kl * 0.95)); // 5 % short
    }
    let t = check_tightness_by_category(&rows, 0.05).unwrap();
    assert!(t["chat"].pass());
    assert!(!t["code"].pass());
    assert!(!t["code"].mean.pass && !t["code"].p99.pass);
}

#[test]
fn tightness_from_real_distributions() {
    // Full logits on both sides: with K covering the head of a sharply peaked reference, KL_K
    // is tight; with K = 1 on a flat reference and a reshaped tail it is not.
    let peaked: Vec<f32> = (0..256).map(|i| -3.0 * (i as f32)).collect();
    let peaked_c: Vec<f32> = peaked
        .iter()
        .enumerate()
        .map(|(i, &z)| z + 0.05 * ((i as f32) * 0.3).sin())
        .collect();
    let kl = kl_full(&peaked, &peaked_c).unwrap();
    let klk = kl_k_from_full(&peaked, &peaked_c, 16);
    assert!(TightnessCheck::new(kl, klk, 0.01).unwrap().pass);

    let flat = zipf_reference();
    let reshaped: Vec<f32> = flat
        .iter()
        .enumerate()
        .map(|(i, &z)| (f64::from(z) + 1.5 * ((i as f64) * 0.011).sin()) as f32)
        .collect();
    let kl = kl_full(&flat, &reshaped).unwrap();
    let klk = kl_k_from_full(&flat, &reshaped, 1);
    assert!(!TightnessCheck::new(kl, klk, 0.01).unwrap().pass);
}
