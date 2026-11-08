// SPDX-License-Identifier: MIT OR Apache-2.0

//! RTT summaries and histograms.

use serde::{Deserialize, Serialize};

use crate::report::Bucket;

/// The histogram's bucket bounds in µs, a 1-2-5 series from 100 µs to 10 s. An echo later than
/// the timeout counts as lost, and the timeout is at most 10 s, so the last bucket holds the rest.
pub const BUCKETS_US: [u32; 16] = [
    100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000, 10_000_000,
];

/// Nearest-rank percentiles of a set of RTTs, in µs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RttSummary {
    /// How many RTTs.
    pub samples: u64,
    /// The smallest.
    pub min: u32,
    /// The median.
    pub p50: u32,
    /// The 99th percentile.
    pub p99: u32,
    /// The 99.9th percentile.
    pub p999: u32,
    /// The largest.
    pub max: u32,
}

impl RttSummary {
    /// The summary of `samples`, or `None` if there are none. Sorts `samples`.
    pub fn of(samples: &mut [u32]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        samples.sort_unstable();
        Some(Self {
            samples: samples.len() as u64,
            min: samples[0],
            p50: rank(samples, 500),
            p99: rank(samples, 990),
            p999: rank(samples, 999),
            max: samples[samples.len() - 1],
        })
    }
}

/// The nearest-rank percentile `per_mille` / 10 of sorted `samples`: the smallest value with at
/// least that share of the samples at or below it.
fn rank(sorted: &[u32], per_mille: u64) -> u32 {
    let n = sorted.len() as u64;
    let rank = (per_mille * n).div_ceil(1_000).max(1);
    sorted[(rank - 1) as usize]
}

/// The histogram of `samples` over [`BUCKETS_US`]. A sample above the last bound counts in the
/// last bucket.
pub fn histogram(samples: &[u32]) -> Vec<Bucket> {
    let mut counts = [0u64; BUCKETS_US.len()];
    for &sample in samples {
        let index = BUCKETS_US
            .iter()
            .position(|&bound| sample <= bound)
            .unwrap_or(BUCKETS_US.len() - 1);
        counts[index] += 1;
    }
    BUCKETS_US
        .iter()
        .zip(counts)
        .map(|(&le_us, count)| Bucket { le_us, count })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_are_nearest_rank() {
        let mut samples: Vec<u32> = (1..=1_000).collect();
        let summary = RttSummary::of(&mut samples).unwrap();
        assert_eq!(
            summary,
            RttSummary {
                samples: 1_000,
                min: 1,
                p50: 500,
                p99: 990,
                p999: 999,
                max: 1_000,
            }
        );
        // With ten samples the 99th and 99.9th percentiles are the largest.
        let mut samples = vec![9, 1, 8, 2, 7, 3, 6, 4, 5, 10];
        let summary = RttSummary::of(&mut samples).unwrap();
        assert_eq!((summary.p50, summary.p99, summary.p999), (5, 10, 10));
        assert_eq!(RttSummary::of(&mut [42]).unwrap().p50, 42);
        assert_eq!(RttSummary::of(&mut []), None);
    }

    #[test]
    fn the_histogram_counts_each_sample_once() {
        let buckets = histogram(&[100, 101, 200, 999_999, 10_000_000, 20_000_000]);
        let counts: Vec<(u32, u64)> = buckets
            .iter()
            .filter(|bucket| bucket.count > 0)
            .map(|bucket| (bucket.le_us, bucket.count))
            .collect();
        assert_eq!(
            counts,
            [(100, 1), (200, 2), (1_000_000, 1), (10_000_000, 2)]
        );
        assert_eq!(buckets.len(), BUCKETS_US.len());
    }
}
