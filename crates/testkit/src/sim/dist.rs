// SPDX-License-Identifier: MIT OR Apache-2.0

//! Delay distributions for the simulated fabric (RFC-0060 §8.3), and the arithmetic that samples
//! them the same way on every host.
//!
//! Every draw comes from the run's seeded [`Entropy`]. The floating-point work uses only
//! addition, subtraction, multiplication, division and square root, which IEEE 754 rounds
//! correctly and so give the same bits everywhere. `f64::exp` and `f64::ln` call the platform's
//! maths library, which is not required to round the same way on Linux, macOS and Windows; a
//! seed must replay one run on any laptop (RFC-0060 §8), so this module computes both itself.

use std::sync::Arc;

use syntropy_core::{Entropy, Nanos};

/// A distribution of delays (RFC-0060 §8.3).
#[derive(Clone, Debug, PartialEq)]
pub enum Dist {
    /// Always this delay.
    Const(Nanos),
    /// Uniform over `lo..=hi` nanoseconds.
    Uniform { lo: Nanos, hi: Nanos },
    /// Log-normal: `median × e^(σ·z)` for a standard normal `z`.
    LogNormal { median: Nanos, sigma: f64 },
    /// A body with a rare heavy tail, such as home Wi-Fi: each draw comes from `tail` with
    /// probability `p_tail`, else from `body`.
    Mixture {
        p_tail: f64,
        body: Box<Dist>,
        tail: Box<Dist>,
    },
    /// A recorded series, replayed in order and from the start again after its end. Empty, it
    /// gives 0.
    Trace(Arc<[Nanos]>),
}

impl Dist {
    /// One delay. `cursor` is the position in any [`Dist::Trace`] this draw reaches; the caller
    /// keeps one per stream of draws, so a trace replays in order.
    pub fn sample(&self, entropy: &dyn Entropy, cursor: &mut usize) -> Nanos {
        match self {
            Dist::Const(delay) => *delay,
            Dist::Uniform { lo, hi } => {
                let (lo, hi) = if lo <= hi { (lo.0, hi.0) } else { (hi.0, lo.0) };
                match (hi - lo).checked_add(1) {
                    Some(span) => Nanos(lo + entropy.u64() % span),
                    None => Nanos(entropy.u64()),
                }
            }
            Dist::LogNormal { median, sigma } => {
                let z = standard_normal(entropy);
                to_nanos(median.0 as f64 * exp(sigma * z))
            }
            Dist::Mixture { p_tail, body, tail } => {
                if unit(entropy) < *p_tail {
                    tail.sample(entropy, cursor)
                } else {
                    body.sample(entropy, cursor)
                }
            }
            Dist::Trace(series) => {
                if series.is_empty() {
                    return Nanos(0);
                }
                let delay = series[*cursor % series.len()];
                *cursor = cursor.wrapping_add(1);
                delay
            }
        }
    }
}

/// A delay in nanoseconds, rounded to the nearest, from a non-negative finite value; a value past
/// `u64`'s range saturates.
fn to_nanos(value: f64) -> Nanos {
    if value.is_nan() || value <= 0.0 {
        Nanos(0)
    } else {
        // `as` saturates at `u64::MAX`; rounding adds a half and truncates.
        Nanos((value + 0.5) as u64)
    }
}

/// A uniform draw from `[0, 1)`: the top 53 bits of a `u64`.
pub(crate) fn unit(entropy: &dyn Entropy) -> f64 {
    (entropy.u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// A standard normal draw, by the Marsaglia polar method, which needs only `ln` and a square root.
fn standard_normal(entropy: &dyn Entropy) -> f64 {
    loop {
        let u = 2.0 * unit(entropy) - 1.0;
        let v = 2.0 * unit(entropy) - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * ln(s) / s).sqrt();
        }
    }
}

/// ln 2, split so that `k × LN_2_HI` is exact for the `k` that `exp` meets: `LN_2_HI` has its
/// low 21 bits zero, and `LN_2_HI + LN_2_LO` is ln 2 to well past double precision. Given by
/// their bits, 0.693147180369123816490 and 1.90821492927058770002e-10.
const LN_2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN_2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);

/// `e^x` from basic IEEE operations: `x = k·ln 2 + r` with `|r| ≤ ln 2 / 2`, `e^r` by its Taylor
/// series, then `× 2^k`. Accurate to about 1e-15 relative.
pub(crate) fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.78 {
        return f64::INFINITY;
    }
    if x < -745.2 {
        return 0.0;
    }
    let k = (x / std::f64::consts::LN_2).round();
    let r = (x - k * LN_2_HI) - k * LN_2_LO;
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..=24 {
        term *= r / f64::from(i);
        sum += term;
    }
    scale_by_power_of_two(sum, k as i32)
}

/// `value × 2^k` by building the power of two from its bits, in two steps where `2^k` alone
/// would be subnormal or overflow.
fn scale_by_power_of_two(value: f64, k: i32) -> f64 {
    let power = |k: i32| f64::from_bits(((k + 1023) as u64) << 52);
    if k > 1023 {
        value * power(1023) * power(k - 1023)
    } else if k < -1022 {
        value * power(-1022) * power(k + 1022)
    } else {
        value * power(k)
    }
}

/// `ln x` for a positive finite `x`, from basic IEEE operations: `x = m·2^e` with
/// `√½ ≤ m < √2`, then `ln m = 2·atanh((m − 1)/(m + 1))` by its series. Accurate to about 1e-15
/// relative.
///
/// # Panics
///
/// If `x` is not positive and finite.
pub(crate) fn ln(x: f64) -> f64 {
    assert!(x > 0.0 && x.is_finite(), "ln of {x}");
    let (mut m, mut e) = {
        let bits = x.to_bits();
        let biased = ((bits >> 52) & 0x7ff) as i32;
        if biased == 0 {
            // Subnormal: scale into the normal range first.
            let scaled = (x * f64::from_bits(((54 + 1023) as u64) << 52)).to_bits();
            (
                f64::from_bits((scaled & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000),
                ((scaled >> 52) & 0x7ff) as i32 - 1023 - 54,
            )
        } else {
            (
                f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000),
                biased - 1023,
            )
        }
    };
    if m > std::f64::consts::SQRT_2 {
        m /= 2.0;
        e += 1;
    }
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = s;
    let mut odd = 1.0;
    for _ in 0..24 {
        term *= s2;
        odd += 2.0;
        sum += term / odd;
    }
    let e = f64::from(e);
    (2.0 * sum + e * LN_2_LO) + e * LN_2_HI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SimEntropy;

    fn relative(a: f64, b: f64) -> f64 {
        if a == b { 0.0 } else { ((a - b) / b).abs() }
    }

    #[test]
    fn exp_and_ln_match_std_closely() {
        let mut worst = (0.0f64, 0.0f64);
        let mut x = -700.0;
        while x < 700.0 {
            worst.0 = worst.0.max(relative(exp(x), x.exp()));
            x += 0.137;
        }
        let mut x = 1e-300;
        while x < 1e300 {
            worst.1 = worst.1.max(relative(ln(x), x.ln()));
            x *= 1.731;
        }
        for x in [
            0.5,
            0.999_999,
            1.000_001,
            2.0,
            std::f64::consts::SQRT_2,
            5e-324,
            f64::MAX,
        ] {
            worst.1 = worst.1.max(relative(ln(x), x.ln()));
        }
        assert!(worst.0 < 1e-13, "exp: worst relative error {:e}", worst.0);
        assert!(worst.1 < 1e-13, "ln: worst relative error {:e}", worst.1);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(exp(800.0), f64::INFINITY);
        assert_eq!(exp(-800.0), 0.0);
    }

    #[test]
    #[should_panic(expected = "ln of 0")]
    fn ln_of_zero_is_a_bug() {
        let _ = ln(0.0);
    }

    #[test]
    fn draws_depend_only_on_the_seed() {
        let dists = [
            Dist::Uniform {
                lo: Nanos(10),
                hi: Nanos(20),
            },
            Dist::LogNormal {
                median: Nanos(5_000_000),
                sigma: 0.92,
            },
        ];
        for dist in &dists {
            let draws = |seed| {
                let entropy = SimEntropy::new(seed);
                let mut cursor = 0;
                (0..100)
                    .map(|_| dist.sample(&entropy, &mut cursor).0)
                    .collect::<Vec<u64>>()
            };
            assert_eq!(draws(4), draws(4));
            assert_ne!(draws(4), draws(5));
        }
    }

    #[test]
    fn const_uniform_and_trace_draw_what_they_say() {
        let entropy = SimEntropy::new(1);
        let mut cursor = 0;
        assert_eq!(
            Dist::Const(Nanos(7)).sample(&entropy, &mut cursor),
            Nanos(7)
        );
        let uniform = Dist::Uniform {
            lo: Nanos(10),
            hi: Nanos(12),
        };
        let mut seen = [false; 3];
        for _ in 0..200 {
            let draw = uniform.sample(&entropy, &mut cursor).0;
            assert!((10..=12).contains(&draw), "{draw}");
            seen[(draw - 10) as usize] = true;
        }
        assert_eq!(seen, [true; 3], "both ends are reachable");
        let trace = Dist::Trace(Arc::from([Nanos(1), Nanos(2), Nanos(3)]));
        let replay: Vec<u64> = (0..7)
            .map(|_| trace.sample(&entropy, &mut cursor).0)
            .collect();
        assert_eq!(replay, [1, 2, 3, 1, 2, 3, 1]);
        assert_eq!(
            Dist::Trace(Arc::from([])).sample(&entropy, &mut cursor),
            Nanos(0)
        );
    }

    #[test]
    fn a_log_normal_draw_centres_on_its_median() {
        let entropy = SimEntropy::new(9);
        let mut cursor = 0;
        let dist = Dist::LogNormal {
            median: Nanos(1_000_000),
            sigma: 0.5,
        };
        let mut draws: Vec<u64> = (0..100_000)
            .map(|_| dist.sample(&entropy, &mut cursor).0)
            .collect();
        draws.sort_unstable();
        let median = draws[draws.len() / 2] as f64;
        assert!((median / 1e6 - 1.0).abs() < 0.02, "median {median}");
    }

    #[test]
    fn exp_and_ln_are_within_1e_15_over_the_presets_ranges() {
        // exp sees σ·z: |σ| ≤ 1 and |z| < 8 in practice. ln sees the polar method's s in (0, 1).
        let mut worst = (0.0f64, 0.0f64);
        let mut x = -8.0;
        while x <= 8.0 {
            worst.0 = worst.0.max(relative(exp(x), x.exp()));
            x += 1e-4;
        }
        let mut x = 1e-12;
        while x < 1.0 {
            worst.1 = worst.1.max(relative(ln(x), x.ln()));
            x *= 1.0001;
        }
        assert!(worst.0 < 1e-15, "exp: worst relative error {:e}", worst.0);
        assert!(worst.1 < 1e-15, "ln: worst relative error {:e}", worst.1);
    }

    /// Exact bits, taken at this commit on Linux x86-64. The arithmetic is basic IEEE operations
    /// only, so every platform must give the same bits; a difference means something in it stopped
    /// being portable, and a seed no longer replays one run everywhere.
    #[test]
    fn golden_values_pin_the_arithmetic_on_every_platform() {
        let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<u64>>();
        assert_eq!(
            bits(&[exp(1.0), exp(-3.7), exp(0.92 * 2.326)]),
            [
                0x4005_bf0a_8b14_5768,
                0x3f99_511f_c687_1045,
                0x4020_ff5d_2b5d_79cf
            ]
        );
        assert_eq!(
            bits(&[ln(2.5), ln(1e-10), ln(0.3)]),
            [
                0x3fed_5240_f0e0_e078,
                0xc037_069e_2aa2_aa5b,
                0xbff3_4378_fcbd_a721
            ]
        );
        let entropy = SimEntropy::new(42);
        let units: Vec<f64> = (0..3).map(|_| unit(&entropy)).collect();
        assert_eq!(
            bits(&units),
            [
                0x3fda_b8c2_9449_b95c,
                0x3fe6_9d6f_eb2b_916b,
                0x3fe6_7136_a7b9_c580
            ]
        );
        let entropy = SimEntropy::new(42);
        let normals: Vec<f64> = (0..3).map(|_| standard_normal(&entropy)).collect();
        assert_eq!(
            bits(&normals),
            [
                0xbfe5_561b_cbbd_7cb0,
                0x3fdd_821b_772d_25f2,
                0x3ff0_9df4_9149_9fe9
            ]
        );
        let entropy = SimEntropy::new(42);
        let mut cursor = 0;
        let body = Dist::LogNormal {
            median: Nanos(5_000_000),
            sigma: 0.92,
        };
        let draws: Vec<u64> = (0..6)
            .map(|_| body.sample(&entropy, &mut cursor).0)
            .collect();
        assert_eq!(
            draws,
            [
                2_707_478, 7_641_690, 12_999_568, 2_779_518, 5_721_031, 63_824_458
            ]
        );
    }
}
