// SPDX-License-Identifier: MIT OR Apache-2.0

//! Standard normal draws that give the same bits on every platform (RFC-0060 §12.1, rule T1).
//!
//! The draws come from [`SimEntropy`], whose bytes are defined by ChaCha20. The transform is
//! Marsaglia's polar method, which needs a square root and a natural logarithm. IEEE 754 rounds
//! `sqrt`, `+`, `-`, `*` and `/` correctly, so they agree everywhere, and Rust never fuses a
//! multiply and an add on its own. The platform's `ln` is not specified to the last bit, so
//! [`ln`] is computed here from those operations alone.

use syntropy_core::Entropy;

use crate::sim::SimEntropy;

/// Bytes taken from the stream at a time.
const CHUNK: usize = 4096;

/// A stream of standard normal draws.
pub struct Normals {
    entropy: SimEntropy,
    buf: Box<[u8; CHUNK]>,
    used: usize,
    spare: Option<f64>,
}

impl Normals {
    /// The draws for `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            entropy: SimEntropy::new(seed),
            buf: Box::new([0; CHUNK]),
            used: CHUNK,
            spare: None,
        }
    }

    fn u64(&mut self) -> u64 {
        if self.used == CHUNK {
            self.entropy.fill(&mut self.buf[..]);
            self.used = 0;
        }
        let mut word = [0u8; 8];
        word.copy_from_slice(&self.buf[self.used..self.used + 8]);
        self.used += 8;
        u64::from_le_bytes(word)
    }

    /// A uniform draw in [-1, 1), on a grid of 2⁻⁵².
    fn signed_unit(&mut self) -> f64 {
        let u = (self.u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        u * 2.0 - 1.0
    }

    /// The next standard normal draw.
    pub fn draw(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let (u, v) = (self.signed_unit(), self.signed_unit());
            let s = u * u + v * v;
            if s == 0.0 || s >= 1.0 {
                continue;
            }
            let k = (-2.0 * ln(s) / s).sqrt();
            self.spare = Some(v * k);
            return u * k;
        }
    }
}

/// The natural logarithm of a positive normal `x`, from correctly rounded operations only.
///
/// `x = m · 2^e` with `m` in [√½, √2), and `ln m = 2 atanh t` with `t = (m − 1) / (m + 1)`, so
/// `|t| < 0.172`. Fourteen terms of the odd series leave an error below 10⁻¹⁹, under half an
/// ulp, and the evaluation order is fixed.
pub fn ln(x: f64) -> f64 {
    debug_assert!(x.is_normal() && x > 0.0);
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52));
    if m > std::f64::consts::SQRT_2 {
        m /= 2.0;
        e += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let mut sum = 0.0;
    for k in (0..14).rev() {
        sum = sum * t2 + 1.0 / f64::from(2 * k + 1);
    }
    2.0 * t * sum + e as f64 * std::f64::consts::LN_2
}
