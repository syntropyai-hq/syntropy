// SPDX-License-Identifier: MIT OR Apache-2.0

//! The streamed `F32Fixed` sum against the level-by-level loop written out, and hand-computed
//! cases whose rounding shows the order.

use proptest::prelude::*;

use super::*;

/// RFC-0060 §4.1's order written out: each block summed in index order from `+0.0`, then the
/// block sums added in neighbouring pairs, level by level, an odd last one carried up.
fn explicit(xs: &[f32]) -> f32 {
    let mut level: Vec<f32> = xs
        .chunks(BLOCK)
        .map(|block| {
            let mut acc = 0.0f32;
            for &x in block {
                acc += x;
            }
            acc
        })
        .collect();
    if level.is_empty() {
        return 0.0;
    }
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| {
                if pair.len() == 2 {
                    pair[0] + pair[1]
                } else {
                    pair[0]
                }
            })
            .collect();
    }
    level[0]
}

fn bits(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn the_empty_sum_is_positive_zero() {
    assert_eq!(bits(sum(&[])), 0);
    assert_eq!(bits(dot(&[], &[])), 0);
    assert_eq!(sum_f64_by(0, |_| unreachable!()).to_bits(), 0);
}

#[test]
fn a_sum_of_negative_zeros_is_positive_zero() {
    assert_eq!(bits(sum(&[-0.0; 5])), 0);
    assert_eq!(bits(sum(&[-0.0; 70])), 0);
}

#[test]
fn a_block_is_summed_in_index_order() {
    // 2^24 + 1 rounds to 2^24 (ties to even), so sequential order loses each 1 that follows
    // 2^24, where a pairwise tree inside the block would keep 1 + 1 = 2.
    let mut xs = vec![1.0f32; 32];
    xs[0] = 16_777_216.0;
    assert_eq!(sum(&xs), 16_777_216.0);
    // The same terms with the big one last: the 31 ones are summed exactly first.
    xs.rotate_left(1);
    assert_eq!(sum(&xs), 16_777_216.0 + 32.0);
}

#[test]
fn blocks_are_added_pairwise() {
    // Three blocks: 2^24 in block 0, then 32 ones in block 1, then 1 alone in block 2.
    // Level 1 adds blocks 0 and 1 (2^24 + 32, exact) and carries block 2; level 2 adds 1 to it,
    // which rounds to even: 2^24 + 32 + 1 lies halfway between neighbours 2 apart.
    let mut xs = vec![0.0f32; 65];
    xs[0] = 16_777_216.0;
    xs[32..64].fill(1.0);
    xs[64] = 1.0;
    assert_eq!(sum(&xs), 16_777_248.0);
    // Block sums [2^24, 32, 1, 1]: level 1 gives (2^24 + 32) and (1 + 1) = 2, exactly; a
    // sequential sum of the four would round each lone 1 away.
    let mut xs = vec![0.0f32; 97];
    xs[0] = 16_777_216.0;
    xs[32..64].fill(1.0);
    xs[64] = 1.0;
    xs[96] = 1.0;
    assert_eq!(sum(&xs), 16_777_250.0);
    assert_eq!(sum(&xs), explicit(&xs));
}

#[test]
fn the_odd_block_is_carried_to_the_top() {
    // Five blocks [1, 1, 1, 1, 2^24]: level 1 gives [2, 2, 2^24], level 2 gives [4, 2^24], and
    // the root 4 + 2^24 is exact. Adding 2^24 earlier would round the ones away.
    let mut xs = vec![0.0f32; 129];
    for b in 0..4 {
        xs[b * 32] = 1.0;
    }
    xs[128] = 16_777_216.0;
    assert_eq!(sum(&xs), 16_777_220.0);
}

#[test]
fn a_product_is_rounded_before_it_is_added() {
    // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11 (2^-24 is half an ulp at 1,
    // ties to even), so with -(1 + 2^-11) added the dot is 0; a fused multiply-add keeps 2^-24.
    let a = 1.0 + 2f32.powi(-12);
    let d = dot(&[a, -1.0], &[a, 1.0 + 2f32.powi(-11)]);
    assert_eq!(bits(d), 0);
}

#[test]
fn nan_and_opposite_infinities_propagate() {
    let mut xs = vec![1.0f32; 100];
    xs[77] = f32::NAN;
    assert!(sum(&xs).is_nan());
    let mut xs = vec![0.0f32; 100];
    xs[3] = f32::INFINITY;
    assert_eq!(sum(&xs), f32::INFINITY);
    xs[90] = f32::NEG_INFINITY;
    assert!(sum(&xs).is_nan());
    assert!(dot(&[0.0], &[f32::INFINITY]).is_nan());
}

#[test]
fn neumaier_keeps_what_plain_summation_loses() {
    let xs = [1.0f64, 1e100, 1.0, -1e100];
    assert_eq!(sum_f64_by(xs.len(), |i| xs[i]), 2.0);
    assert!(sum_f64_by(2, |i| [f64::INFINITY, 1.0][i]).is_infinite());
    assert!(sum_f64_by(2, |i| [f64::NAN, 1.0][i]).is_nan());
}

#[test]
#[should_panic(expected = "unequal lengths")]
fn a_dot_of_unequal_lengths_panics() {
    let _ = dot(&[1.0], &[1.0, 2.0]);
}

/// Values whose sums round often: wide exponents, both signs.
fn term() -> impl Strategy<Value = f32> {
    (any::<bool>(), -30i32..30, 1u32..(1 << 23)).prop_map(|(neg, e, m)| {
        let x = (1.0 + m as f32 / (1u32 << 23) as f32) * 2f32.powi(e);
        if neg { -x } else { x }
    })
}

proptest! {
    #[test]
    fn the_streamed_sum_is_the_level_by_level_loop(xs in prop::collection::vec(term(), 0..2100)) {
        prop_assert_eq!(bits(sum(&xs)), bits(explicit(&xs)));
    }

    #[test]
    fn every_block_count_up_to_70_matches(blocks in 0usize..70, tail in 0usize..32, seed in any::<u64>()) {
        // Cover every shape of the tree, including the partial last block.
        let n = blocks * BLOCK + tail;
        let mut s = seed | 1;
        let xs: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 * 2f32.powi((s % 41) as i32 - 20) * if s & 1 == 0 { 1.0 } else { -1.0 }
            })
            .collect();
        prop_assert_eq!(bits(sum(&xs)), bits(explicit(&xs)));
    }

    #[test]
    fn dot_sums_the_rounded_products(pairs in prop::collection::vec((term(), term()), 0..300)) {
        let (a, b): (Vec<f32>, Vec<f32>) = pairs.into_iter().unzip();
        let products: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        prop_assert_eq!(bits(dot(&a, &b)), bits(explicit(&products)));
    }

    #[test]
    fn each_term_is_read_once_in_index_order(n in 0usize..500) {
        let mut seen = Vec::new();
        let _ = sum_by(n, |i| {
            seen.push(i);
            1.0
        });
        prop_assert_eq!(seen, (0..n).collect::<Vec<_>>());
    }
}
