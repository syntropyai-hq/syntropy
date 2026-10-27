// SPDX-License-Identifier: MIT OR Apache-2.0

//! Log-linear histogram buckets (RFC-0053 §2.1). A hot-thread module: the lint of `cargo xtask
//! ci lint`'s `hot-threads` step rejects `tracing` here.
//!
//! A histogram has [`BUCKETS`] = 256 buckets: 32 power-of-two ranges above its scale's base,
//! each split into 8 linear sub-buckets, so a bucket is at most 12.5% wider than its lower
//! bound. Values below the base fall in bucket 0, and values at or above the top range's end in
//! bucket 255. The bucket of a value is found from its leading zeros and the three bits after
//! its top bit, with no loop, division or table.

use serde::{Deserialize, Serialize};

use super::desc::Unit;

/// Buckets per histogram.
pub const BUCKETS: usize = 256;

/// Power-of-two ranges per histogram.
const RANGES: u32 = 32;

/// Linear sub-buckets per range, as a power of two.
const SUB_BITS: u32 = 3;

/// A histogram's range: the power of two its first range starts at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HistScale {
    /// Times: 2⁶ ns (64 ns) to 2³⁸ ns (≈ 275 s).
    Nanos,
    /// Sizes: 2¹⁰ B (1 KiB) to 2⁴² B (4 TiB).
    Bytes,
}

impl HistScale {
    /// The scale for a histogram of `unit`, or `None` for a unit no histogram takes: RFC-0053
    /// §2.1 gives histograms of times and of sizes only.
    #[must_use]
    pub const fn of(unit: Unit) -> Option<HistScale> {
        match unit {
            Unit::Nanos => Some(HistScale::Nanos),
            Unit::Bytes => Some(HistScale::Bytes),
            Unit::Count | Unit::Ratio => None,
        }
    }

    /// The exponent of the first range's start.
    #[must_use]
    pub const fn base(self) -> u32 {
        match self {
            HistScale::Nanos => 6,
            HistScale::Bytes => 10,
        }
    }

    /// The bucket `value` falls in.
    #[inline]
    #[must_use]
    pub const fn bucket(self, value: u64) -> usize {
        let base = self.base();
        if value < (1 << base) {
            return 0;
        }
        let top = 63 - value.leading_zeros();
        let range = top - base;
        if range >= RANGES {
            return BUCKETS - 1;
        }
        // `top` is at least the base, 6 or more, so the shift is in range.
        let sub = (value >> (top - SUB_BITS)) & 7;
        (range * 8 + sub as u32) as usize
    }

    /// The smallest value bucket `index` holds; 0 for bucket 0, which holds everything below the
    /// base too.
    #[must_use]
    pub const fn lower_bound(self, index: usize) -> u64 {
        if index == 0 {
            return 0;
        }
        let range = (index / 8) as u32;
        let sub = (index % 8) as u64;
        // The base is 6 or more, so the shift is in range.
        (8 + sub) << (self.base() + range - SUB_BITS)
    }

    /// One past the largest value bucket `index` holds; `u64::MAX` for the last bucket, which
    /// holds everything above the top range too.
    #[must_use]
    pub const fn upper_bound(self, index: usize) -> u64 {
        if index + 1 >= BUCKETS {
            u64::MAX
        } else {
            self.lower_bound(index + 1)
        }
    }

    /// The upper bound of the bucket that holds quantile `q` (0 to 1) of `counts`, or `None`
    /// when the histogram is empty. Computed at read time from bucket counts (RFC-0053 §2.1), so
    /// it overstates by at most one bucket's width.
    #[must_use]
    pub fn quantile(self, counts: &[u64; BUCKETS], q: f64) -> Option<u64> {
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return None;
        }
        // The rank of the quantile's value, 1-based, at least the first value.
        let rank = ((q.clamp(0.0, 1.0) * total as f64).ceil() as u64).max(1);
        let mut seen = 0;
        for (index, &count) in counts.iter().enumerate() {
            seen += count;
            if seen >= rank {
                return Some(self.upper_bound(index));
            }
        }
        Some(u64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every bucket's lower bound falls in it, the value just below it in the bucket before, and
    /// the bounds grow.
    #[test]
    fn bounds_and_buckets_agree() {
        for scale in [HistScale::Nanos, HistScale::Bytes] {
            let mut last = 0;
            for index in 1..BUCKETS {
                let lower = scale.lower_bound(index);
                assert!(lower > last, "{scale:?} bucket {index}");
                assert_eq!(scale.bucket(lower), index, "{scale:?} {lower}");
                assert_eq!(scale.bucket(lower - 1), index - 1, "{scale:?} {lower}");
                last = lower;
            }
        }
    }

    /// RFC-0053 §2.1's ranges: times from 64 ns to 2³⁸ ns, sizes from 1 KiB to 4 TiB, smaller
    /// values in the first bucket and larger in the last.
    #[test]
    fn the_ranges_are_the_rfcs() {
        let nanos = HistScale::Nanos;
        assert_eq!(nanos.bucket(0), 0);
        assert_eq!(nanos.bucket(63), 0);
        assert_eq!(nanos.bucket(64), 0);
        assert_eq!(nanos.bucket(72), 1);
        assert_eq!(nanos.lower_bound(255), (15 << 34));
        assert_eq!(nanos.bucket((1 << 38) - 1), 255);
        assert_eq!(nanos.bucket(1 << 38), 255);
        assert_eq!(nanos.bucket(u64::MAX), 255);
        let bytes = HistScale::Bytes;
        assert_eq!(bytes.bucket(1023), 0);
        assert_eq!(bytes.bucket(1024), 0);
        assert_eq!(bytes.bucket(1 << 20), 80);
        assert_eq!(bytes.bucket(1 << 42), 255);
        assert_eq!(HistScale::of(Unit::Count), None);
    }

    /// A bucket above the first is at most 12.5% wider than its lower bound.
    #[test]
    fn buckets_are_at_most_an_eighth_wide() {
        for scale in [HistScale::Nanos, HistScale::Bytes] {
            for index in 1..BUCKETS - 1 {
                let (lower, upper) = (scale.lower_bound(index), scale.upper_bound(index));
                assert!((upper - lower) * 8 <= lower, "{scale:?} {index}");
            }
        }
    }

    #[test]
    fn quantiles_read_the_bucket_counts() {
        let scale = HistScale::Nanos;
        let mut counts = [0; BUCKETS];
        assert_eq!(scale.quantile(&counts, 0.5), None);
        counts[scale.bucket(1_000)] = 90;
        counts[scale.bucket(1_000_000)] = 10;
        let median = scale.quantile(&counts, 0.5).unwrap();
        assert!((1_000..=1_125).contains(&median), "{median}");
        let p99 = scale.quantile(&counts, 0.99).unwrap();
        assert!((1_000_000..=1_125_000).contains(&p99), "{p99}");
    }
}
