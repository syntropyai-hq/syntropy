// SPDX-License-Identifier: MIT OR Apache-2.0

//! The thread-count harness: runs one op's output ranges on real threads and checks that the
//! result does not depend on how many threads computed it (COMP-E3-T09; RFC-0040 §3.3).
//!
//! An op that splits its work across a pool hands each thread a contiguous range of output
//! rows. [`run_split`] does that with scoped threads. Each thread writes into a buffer of its
//! own, so no thread can see another's writes, and the harness then copies each thread's
//! range into one output. [`check_thread_counts`] runs it at 1, 2, 3, 4 and every core the
//! host offers, plus more threads than output rows, and compares the outputs with the
//! one-thread run bit for bit, NaN payloads included.
//!
//! A kernel task adds its op by giving the harness its partition and its range kernel.

use std::ops::Range;
use std::thread;

/// The thread counts every op is checked at: 1, 2, 3, 4, the host's cores, and `n_out + 3`,
/// which leaves some threads an empty range.
pub fn thread_counts(n_out: usize) -> Vec<usize> {
    let cores = thread::available_parallelism().map_or(1, usize::from);
    let mut v = vec![1, 2, 3, 4, cores, n_out + 3];
    v.sort_unstable();
    v.dedup();
    v
}

/// Runs `kernel` over the ranges `partition(parts)` gives, one thread per range, and returns
/// the `rows` × `n_out` output. `kernel(range, y)` writes the outputs in `range` of every row
/// of `y`, which is `rows · n_out` values long, and must leave the rest alone.
///
/// Each thread's buffer starts filled with a signalling-NaN pattern, so an output no range
/// covered shows up as that pattern and fails the comparison.
pub fn run_split<P, K>(rows: usize, n_out: usize, parts: usize, partition: P, kernel: K) -> Vec<f32>
where
    P: Fn(usize) -> Vec<Range<usize>>,
    K: Fn(Range<usize>, &mut [f32]) + Sync,
{
    const UNWRITTEN: u32 = 0x7FA0_0001;
    let ranges = partition(parts);
    assert_eq!(ranges.len(), parts.max(1), "one range per thread");
    let kernel = &kernel;
    let parts_out: Vec<(Range<usize>, Vec<f32>)> = thread::scope(|s| {
        let handles: Vec<_> = ranges
            .into_iter()
            .map(|range| {
                s.spawn(move || {
                    let mut y = vec![f32::from_bits(UNWRITTEN); rows * n_out];
                    kernel(range.clone(), &mut y);
                    (range, y)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("the kernel thread panicked"))
            .collect()
    });
    let mut y = vec![f32::from_bits(UNWRITTEN); rows * n_out];
    for (range, part) in parts_out {
        for r in 0..rows {
            let row = r * n_out;
            y[row + range.start..row + range.end]
                .copy_from_slice(&part[row + range.start..row + range.end]);
        }
    }
    y
}

/// Checks that `kernel` gives the same bits at every count of [`thread_counts`], and that two
/// runs at one thread agree. Returns the one-thread output.
pub fn check_thread_counts<P, K>(
    what: &str,
    rows: usize,
    n_out: usize,
    partition: P,
    kernel: K,
) -> Vec<f32>
where
    P: Fn(usize) -> Vec<Range<usize>>,
    K: Fn(Range<usize>, &mut [f32]) + Sync,
{
    let one = run_split(rows, n_out, 1, &partition, &kernel);
    let again = run_split(rows, n_out, 1, &partition, &kernel);
    assert_eq!(
        bits(&one),
        bits(&again),
        "{what}: two one-thread runs differ"
    );
    for parts in thread_counts(n_out) {
        let y = run_split(rows, n_out, parts, &partition, &kernel);
        let first = bits(&y).iter().zip(bits(&one)).position(|(a, b)| *a != b);
        assert!(
            first.is_none(),
            "{what}: {parts} threads differ from 1 at output {first:?}"
        );
    }
    one
}

/// The values' bit patterns.
pub fn bits(y: &[f32]) -> Vec<u32> {
    y.iter().map(|v| v.to_bits()).collect()
}
