// SPDX-License-Identifier: MIT OR Apache-2.0

//! Microbenchmarks of `obs`'s hot-thread recording against RFC-0053's budgets (PROD-E8-T01):
//! a counter add at most 5 ns and a histogram record at most 20 ns [targets].
//!
//! Ignored timing tests, not part of any suite: RFC-0061's microbenchmark harness (QUAL-E13-T08)
//! does not exist yet, and a timing on a shared CI runner proves nothing. Run them on a quiet
//! machine with an optimised build, `--ignored --nocapture`, under a memory cap.
//!
//! Each prints its median over 9 runs of 10⁷ operations and fails above its budget. The figures
//! are measurements of the machine they ran on; they go in a PR, labelled `[measured]` with the
//! commit and rig, never in the docs. `bench` reads the wall clock here because it is outside the
//! determinism lints, which keep it out of `obs` (RFC-0060 §2.1 rule T1).

use std::hint::black_box;
use std::time::Instant;

use syntropy_obs::metrics::{
    LabelKey, LocalMetrics, MetricDesc, MetricId, MetricKind, Registry, ThreadKind, Unit,
};

const OPS: u64 = 10_000_000;
const RUNS: usize = 9;

const TOKENS: MetricDesc = MetricDesc {
    id: MetricId(400),
    name: "syntropy_exec_tokens_total",
    kind: MetricKind::Counter,
    unit: Unit::Count,
    labels: &[LabelKey::STAGE],
    help: "Tokens processed.",
    owner: "exec",
};

const STEP_TIME: MetricDesc = MetricDesc {
    id: MetricId(401),
    name: "syntropy_exec_step_seconds",
    kind: MetricKind::Histogram,
    unit: Unit::Nanos,
    labels: &[LabelKey::STAGE],
    help: "Stage step time.",
    owner: "exec",
};

/// The median of `RUNS` timings of `op` over `OPS` iterations, in nanoseconds per operation.
fn median_ns(mut op: impl FnMut(u64)) -> f64 {
    let mut runs: Vec<f64> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            for n in 0..OPS {
                op(black_box(n));
            }
            start.elapsed().as_nanos() as f64 / OPS as f64
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    runs[RUNS / 2]
}

#[test]
#[ignore = "a timing: run by hand, optimised, on a quiet machine"]
fn counter_add_is_within_5_ns() {
    let mut registry = Registry::new();
    registry.register(&TOKENS).unwrap();
    let tokens = registry.counter(TOKENS.id, &["0"]).unwrap();
    let local = LocalMetrics::new(&registry.freeze(), ThreadKind::Dispatcher(0));
    // The cells and the handle go through `black_box` too, so no iteration folds into another.
    let ns = median_ns(|n| black_box(&*local).add(black_box(tokens), n & 3));
    println!("counter add: {ns:.2} ns/op (median of {RUNS} x {OPS})");
    assert!(
        ns <= 5.0,
        "counter add took {ns:.2} ns, over the 5 ns budget"
    );
}

#[test]
#[ignore = "a timing: run by hand, optimised, on a quiet machine"]
fn histogram_record_is_within_20_ns() {
    let mut registry = Registry::new();
    registry.register(&STEP_TIME).unwrap();
    let step_time = registry.histogram(STEP_TIME.id, &["0"]).unwrap();
    let local = LocalMetrics::new(&registry.freeze(), ThreadKind::Dispatcher(0));
    // Values spread over the whole range, so the bucket is not predictable.
    let values: Vec<u64> = (0..4096u64)
        .map(|n| (1u64 << (6 + n % 32)) | (n.wrapping_mul(0x9e37_79b9) & 0xffff))
        .collect();
    let ns = median_ns(|n| {
        black_box(&*local).record(black_box(step_time), values[(n & 4095) as usize]);
    });
    println!("histogram record: {ns:.2} ns/op (median of {RUNS} x {OPS})");
    assert!(
        ns <= 20.0,
        "histogram record took {ns:.2} ns, over the 20 ns budget"
    );
}
