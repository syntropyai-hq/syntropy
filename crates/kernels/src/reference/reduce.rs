// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reference's reductions: one implementation of `F32Fixed`'s order, for every op in
//! [`reference`](super) and for `graph::reference`, which `testkit`'s `RefExecutor` drives
//! (RFC-0060 §4.1; ADR-0019 decision 1; COMP-E1-T03).
//!
//! **The `F32Fixed` order.** RFC-0060 §4.1 says every reduction is "pairwise over fixed
//! 32-element blocks in index order". Its note of 2026-10-31 reads that as two levels:
//!
//! 1. The terms are cut into blocks of [`BLOCK`] = 32 in index order; the last block holds the
//!    remainder, 1 to 32 terms. Each block is summed sequentially in index order into an F32
//!    accumulator that starts at `+0.0`: `((0 + t₀) + t₁) + … + t₃₁`.
//! 2. The block sums are added pairwise, level by level: neighbours `(s₀ + s₁), (s₂ + s₃), …`,
//!    with an odd last sum carried to the next level unchanged, until one sum is left.
//!
//! An empty reduction is `+0.0`. Because the accumulator starts at `+0.0`, a sum is never `−0.0`:
//! in round-to-nearest `+0.0 + −0.0 = +0.0`. Each addition is one IEEE F32 addition, so a NaN
//! term makes the sum NaN, and `+∞` with `−∞` makes it NaN. Nothing is contracted (Rust never
//! fuses a multiply and an add), so a product term is rounded once before it is added.
//!
//! [`sum_by`] streams the level-by-level tree with a stack of at most 64 partial sums, one per
//! bit of the block count, so it allocates nothing and reads each term once; its unit tests check
//! it against the level-by-level loop written out. [`sum`] and [`dot`] are the two forms the ops
//! use. A GPU running `F32Fixed` (`RefDevice::Gpu`, COMP-E14-T23) uses the same tree.
//!
//! **The `F64` mode** sums with Neumaier's compensated summation in index order
//! ([`sum_f64_by`]), as RFC-0060 §4.1 defines it.

/// Terms per block of the `F32Fixed` order.
pub const BLOCK: usize = 32;

/// The `F32Fixed` sum of the `n` terms `term(0), term(1), …, term(n − 1)` (module docs). Each
/// term is read once, in index order.
#[allow(
    clippy::assign_op_pattern,
    reason = "keeps the lower-index sum as the left operand, as the F32Fixed definition writes it"
)]
pub fn sum_by(n: usize, mut term: impl FnMut(usize) -> f32) -> f32 {
    // The pending sums of the level-by-level tree: `sums[i]` covers `1 << level[i]` blocks, and
    // the levels strictly decrease from the bottom of the stack. Pushing a block sum and
    // merging equal levels is a binary counter, and merging what is left from the top down
    // adds each odd carried sum where the level-by-level loop adds it.
    let mut sums = [0.0f32; usize::BITS as usize];
    let mut levels = [0u32; usize::BITS as usize];
    let mut depth = 0;
    let mut start = 0;
    while start < n {
        let end = n.min(start + BLOCK);
        let mut acc = 0.0f32;
        for i in start..end {
            acc += term(i);
        }
        let mut level = 0;
        while depth > 0 && levels[depth - 1] == level {
            depth -= 1;
            acc = sums[depth] + acc;
            level += 1;
        }
        sums[depth] = acc;
        levels[depth] = level;
        depth += 1;
        start = end;
    }
    let Some(top) = depth.checked_sub(1) else {
        return 0.0;
    };
    let mut acc = sums[top];
    for &s in sums[..top].iter().rev() {
        acc = s + acc;
    }
    acc
}

/// The `F32Fixed` sum of `xs` (module docs).
#[must_use]
pub fn sum(xs: &[f32]) -> f32 {
    sum_by(xs.len(), |i| xs[i])
}

/// The `F32Fixed` dot product of `a` and `b`: the sum of the rounded products `a[i] · b[i]`
/// (module docs).
///
/// # Panics
///
/// When `a` and `b` differ in length.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot of unequal lengths");
    sum_by(a.len(), |i| a[i] * b[i])
}

/// The `F64` mode's sum of `term(0), …, term(n − 1)`: Neumaier's compensated summation in index
/// order, the compensation added once at the end. An empty sum is `+0.0`. A non-finite running
/// sum makes the compensation meaningless, so the plain sum is returned then, and NaN and
/// infinities propagate as in an uncompensated sum.
pub fn sum_f64_by(n: usize, mut term: impl FnMut(usize) -> f64) -> f64 {
    let mut s = 0.0f64;
    let mut c = 0.0f64;
    for i in 0..n {
        let x = term(i);
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    if s.is_finite() { s + c } else { s }
}

#[cfg(test)]
mod tests;
