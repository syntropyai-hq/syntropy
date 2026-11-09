// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-computed cases for the RoPE tables and `Rope`, and property tests for
//! `rows_active = 0`, padded rows and NaN propagation.

use proptest::prelude::*;

use super::*;
use crate::reference::trig;

const SENTINEL: f32 = -12345.0;

fn table(scaling: RopeScaling, theta: f32, d_rot: usize, pairs: RopePairs) -> RopeTable {
    RopeTable::new(&Rope {
        scaling,
        theta,
        d_rot,
        pairs,
    })
    .unwrap()
}

fn plain(theta: f32, d_rot: usize) -> RopeTable {
    table(RopeScaling::Plain, theta, d_rot, RopePairs::Adjacent)
}

/// `θ^(−2i/d)` from the platform's F64 `powf`, an independent oracle for the tables.
fn oracle(theta: f32, d_rot: usize, i: usize) -> f64 {
    f64::from(theta).powf(-((2 * i) as f64) / d_rot as f64)
}

/// The F32 ulps between two values of one sign.
fn ulp_diff(a: f32, b: f32) -> u32 {
    a.to_bits().abs_diff(b.to_bits())
}

#[test]
fn plain_frequencies_by_hand() {
    // θ = 10000, d_rot = 4: θ^0 = 1 and θ^(-1/2) = 0.01.
    assert_eq!(plain(10_000.0, 4).freqs(), &[1.0, 0.01]);
    // The table against powf at Qwen3's θ and head dim: equal or 1 ulp apart.
    let t = plain(1_000_000.0, 128);
    assert_eq!(t.freqs().len(), 64);
    assert_eq!(t.attn_factor(), 1.0);
    for (i, &f) in t.freqs().iter().enumerate() {
        assert!(
            ulp_diff(f, oracle(1e6, 128, i) as f32) <= 1,
            "freq[{i}] = {f:e}"
        );
    }
}

#[test]
fn software_ln_and_exp_against_f64() {
    for x in [1e-300, 0.1, 0.5, 1.0, 2.0, 10_000.0, 500_000.0, 1e6, 1e300] {
        let (got, want) = (ln64(x), x.ln());
        assert!(
            (got - want).abs() <= 4.0 * f64::EPSILON * want.abs().max(1.0),
            "ln({x})"
        );
    }
    assert_eq!(ln64(1.0), 0.0);
    for y in [-700.0, -20.7, -1.0, -1e-9, 0.0, 0.3, 1.0, 13.8, 700.0] {
        let (got, want) = (exp64(y), y.exp());
        assert!((got - want).abs() <= 4.0 * f64::EPSILON * want, "exp({y})");
    }
}

#[test]
fn rotation_by_hand() {
    let t = plain(10_000.0, 4);
    let (s1, c1) = trig::sin_cos(1.0);
    let (s2, c2) = trig::sin_cos(0.01);
    // Adjacent pairs (0, 1), (2, 3): [1, 0] rotates to [cos a, sin a].
    let mut y = [SENTINEL; 4];
    rope(&t, 1, 4, &[1.0, 0.0, 1.0, 0.0], &[1], 1, &mut y).unwrap();
    assert_eq!(y, [c1, s1, c2, s2]);
    // cos 1 = 0.540302306, sin 1 = 0.841470985, cos 0.01 = 0.999950000, sin 0.01 = 0.009999833.
    for (got, want) in y
        .iter()
        .zip([0.540_302_3, 0.841_470_96, 0.999_95, 0.009_999_833])
    {
        assert!(ulp_diff(*got, want) <= 1, "{got} vs {want}");
    }
    // Halves pairs (0, 2), (1, 3).
    let h = table(RopeScaling::Plain, 10_000.0, 4, RopePairs::Halves);
    rope(&h, 1, 4, &[1.0, 1.0, 0.0, 0.0], &[1], 1, &mut y).unwrap();
    assert_eq!(y, [c1, c2, s1, s2]);
    // [u, w] = [2, 3] at angle 1: u' = 2c − 3s, w' = 3c + 2s, each product rounded once.
    rope(&t, 1, 4, &[2.0, 3.0, 0.0, 0.0], &[1], 1, &mut y).unwrap();
    assert_eq!(y[0].to_bits(), (2.0 * c1 - 3.0 * s1).to_bits());
    assert_eq!(y[1].to_bits(), (3.0 * c1 + 2.0 * s1).to_bits());
}

#[test]
fn position_zero_and_unrotated_dims() {
    // Position 0 leaves finite values unchanged; d_rot = 2 of d_head = 4 copies dims 2 and 3
    // even when they hold NaN.
    let t = plain(10_000.0, 2);
    let x = [1.5, -2.0, f32::NAN, 7.0, 0.25, 8.0, 9.0, -0.0];
    let mut y = [SENTINEL; 10];
    rope(&t, 2, 4, &x, &[0], 1, &mut y).unwrap();
    assert_eq!(&y[..2], &[1.5, -2.0]);
    assert!(y[2].is_nan());
    assert_eq!(&y[3..8], &[7.0, 0.25, 8.0, 9.0, -0.0]);
    assert_eq!(y[7].to_bits(), (-0.0f32).to_bits());
    assert_eq!(&y[8..], &[SENTINEL; 2]);
    // d_rot = 0: a layer without RoPE copies every value.
    let none = plain(10_000.0, 0);
    let mut z = [SENTINEL; 8];
    rope(&none, 2, 4, &x, &[77], 1, &mut z).unwrap();
    assert_eq!(
        z.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        x.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn linear_divides_the_frequencies() {
    let t = table(
        RopeScaling::Linear { factor: 4.0 },
        10_000.0,
        4,
        RopePairs::Adjacent,
    );
    assert_eq!(t.freqs(), &[0.25, 0.0025]);
    assert_eq!(t.kind(), RopeKind::Linear);
}

#[test]
fn llama3_bands() {
    // Llama 3.1's parameters: factor 8, low 1, high 4, original context 8192, θ = 500000.
    let (factor, lo, hi, ctx) = (8.0f64, 1.0f64, 4.0f64, 8192.0f64);
    let t = table(
        RopeScaling::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_ctx: 8192,
        },
        500_000.0,
        128,
        RopePairs::Halves,
    );
    let mut bands = [0; 3];
    for (i, &f) in t.freqs().iter().enumerate() {
        let p = oracle(500_000.0, 128, i);
        let wavelen = 2.0 * std::f64::consts::PI / p;
        let want = if wavelen < ctx / hi {
            bands[0] += 1;
            p
        } else if wavelen > ctx / lo {
            bands[2] += 1;
            p / factor
        } else {
            bands[1] += 1;
            let s = (ctx / wavelen - lo) / (hi - lo);
            (1.0 - s) * p / factor + s * p
        };
        assert!(
            ulp_diff(f, want as f32) <= 1,
            "freq[{i}] = {f:e}, want {want:e}"
        );
    }
    // Every band is exercised: kept, interpolated and divided.
    assert!(bands.iter().all(|&n| n > 0), "{bands:?}");
    assert_eq!(t.freqs()[0], 1.0);
}

#[test]
fn yarn_ramp_and_attention_factor() {
    let yarn = |attn_factor| {
        table(
            RopeScaling::Yarn {
                factor: 4.0,
                original_ctx: 32_768,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attn_factor,
            },
            1_000_000.0,
            128,
            RopePairs::Halves,
        )
    };
    let t = yarn(None);
    // 0.1 · ln 4 + 1 = 1.138629436…
    assert!(ulp_diff(t.attn_factor(), 1.138_629_4) <= 1);
    // c(32) = 128 · ln(32768 / (64π)) / (2 ln 1e6) = 23.2…, so lo = 23; c(1) = 39.2…, so hi = 40.
    // Below lo the frequency is the plain one, above hi the plain one divided by 4.
    for i in 0..=23 {
        assert!(
            ulp_diff(t.freqs()[i], oracle(1e6, 128, i) as f32) <= 1,
            "{i}"
        );
    }
    for i in 40..64 {
        assert!(
            ulp_diff(t.freqs()[i], (oracle(1e6, 128, i) / 4.0) as f32) <= 1,
            "{i}"
        );
    }
    // Between, the ramp mixes them: (i − 23) / 17 of the divided frequency.
    let r = 6.0 / 17.0;
    let p = oracle(1e6, 128, 29);
    assert!(ulp_diff(t.freqs()[29], (p / 4.0 * r + p * (1.0 - r)) as f32) <= 1);
    // The sines and cosines carry the factor.
    let (s, c) = t.sin_cos(3, 0);
    let (s0, c0) = trig::sin_cos(3.0);
    assert_eq!((s, c), (s0 * t.attn_factor(), c0 * t.attn_factor()));
    // A stated attention factor wins.
    assert_eq!(yarn(Some(0.5)).attn_factor(), 0.5);
}

#[test]
fn angles_above_two_to_the_twenty_are_f64() {
    let t = plain(10_000.0, 8);
    for i in 0..4 {
        let f = t.freqs()[i];
        let p = F32_ANGLE_MAX;
        assert_eq!(t.sin_cos(p, i), trig::sin_cos(p as f32 * f));
        let p = F32_ANGLE_MAX + 1;
        assert_eq!(
            t.sin_cos(p, i),
            trig::sin_cos_f64(f64::from(p) * f64::from(f))
        );
        let p = u32::MAX;
        assert_eq!(
            t.sin_cos(p, i),
            trig::sin_cos_f64(f64::from(p) * f64::from(f))
        );
    }
}

#[test]
fn bad_parameters_are_refused() {
    let bad = |scaling, theta, d_rot| {
        RopeTable::new(&Rope {
            scaling,
            theta,
            d_rot,
            pairs: RopePairs::Adjacent,
        })
        .unwrap_err()
    };
    let p = RopeScaling::Plain;
    for e in [
        bad(p, 10_000.0, 3),
        bad(p, 1.0, 4),
        bad(p, f32::NAN, 4),
        bad(RopeScaling::Linear { factor: 0.0 }, 10_000.0, 4),
        bad(
            RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 4.0,
                high_freq_factor: 4.0,
                original_ctx: 8192,
            },
            10_000.0,
            4,
        ),
        bad(
            RopeScaling::Yarn {
                factor: 4.0,
                original_ctx: 0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attn_factor: None,
            },
            10_000.0,
            4,
        ),
        bad(
            RopeScaling::Yarn {
                factor: 4.0,
                original_ctx: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attn_factor: Some(f32::INFINITY),
            },
            10_000.0,
            4,
        ),
    ] {
        assert!(matches!(e, OpError::Param { op: "Rope", .. }), "{e}");
    }
    // d_rot past d_head, and short buffers, write nothing.
    let t = plain(10_000.0, 8);
    let mut y = [SENTINEL; 4];
    let e = rope(&t, 1, 4, &[1.0; 4], &[1], 1, &mut y).unwrap_err();
    assert!(matches!(e, OpError::Param { .. }), "{e}");
    let t = plain(10_000.0, 4);
    let e = rope(&t, 1, 4, &[1.0; 4], &[], 1, &mut y).unwrap_err();
    assert!(
        matches!(
            e,
            OpError::Len {
                arg: "positions",
                ..
            }
        ),
        "{e}"
    );
    assert_eq!(y, [SENTINEL; 4]);
}

proptest! {
    #[test]
    fn rows_active_zero_writes_nothing(
        x in prop::collection::vec(any::<f32>(), 0..32),
        pos in prop::collection::vec(any::<u32>(), 0..4),
    ) {
        let t = plain(10_000.0, 4);
        let mut y = vec![SENTINEL; 32];
        rope(&t, 2, 4, &x, &pos, 0, &mut y).unwrap();
        prop_assert!(y.iter().all(|&v| v == SENTINEL));
    }

    #[test]
    fn nan_stays_in_its_pair(
        x in prop::collection::vec(-100.0f32..100.0, 16),
        positions in prop::collection::vec(1u32..5_000_000, 2),
        at in 0usize..16,
        halves in any::<bool>(),
    ) {
        // Two rows of two heads of 4 values; d_rot = 4, so every value has a partner.
        let pairs = if halves { RopePairs::Halves } else { RopePairs::Adjacent };
        let t = table(RopeScaling::Plain, 10_000.0, 4, pairs);
        let mut x = x;
        x[at] = f32::NAN;
        let mut y = vec![SENTINEL; 20];
        rope(&t, 2, 4, &x, &positions, 2, &mut y).unwrap();
        let partner = |i: usize| {
            let (head, d) = (i - i % 4, i % 4);
            head + if halves { (d + 2) % 4 } else { d ^ 1 }
        };
        for (i, &v) in y[..16].iter().enumerate() {
            let in_pair = i == at || i == partner(at);
            prop_assert_eq!(v.is_nan(), in_pair, "{} {:?}", i, y);
        }
        prop_assert!(y[16..].iter().all(|&v| v == SENTINEL));
    }
}
