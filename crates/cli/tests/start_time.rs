// SPDX-License-Identifier: MIT OR Apache-2.0

//! PROD-E6-T01's start-time benchmark: process start to first output of a client command,
//! against RFC-0051's budget of 50 ms [target].
//!
//! Ignored by default. Timing belongs on a lab rig, never a hosted runner (RFC-0061; the
//! microbenchmark harness, QUAL-E13-T08, runs it there once it exists). Run it with
//! `cargo test -p syntropy-cli --release --test start_time -- --ignored --nocapture`. It runs
//! `syntropy version --json` against a daemon served in-process `RUNS` times, measures from
//! spawn to the first byte on stdout, prints the median and the slowest, and fails if the median
//! is over the budget.

mod common;

use std::io::Read;
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::{BIN, Daemon};

/// How many runs the figures come from.
const RUNS: usize = 41;

/// RFC-0051's budget [target].
const BUDGET: Duration = Duration::from_millis(50);

#[test]
#[ignore = "timing: run on a lab rig with --release (RFC-0061)"]
fn process_start_to_first_output_of_version() {
    let daemon = Daemon::start(None);
    let host = format!("http://{}", daemon.addr);
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        let mut child = std::process::Command::new(BIN)
            .args(["version", "--json"])
            .env("SYNTROPY_HOST", &host)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut first = [0u8; 1];
        let read = child.stdout.as_mut().unwrap().read(&mut first).unwrap();
        let elapsed = start.elapsed();
        assert_eq!(read, 1, "no output");
        assert!(child.wait().unwrap().success());
        times.push(elapsed);
    }
    times.sort();
    let median = times[RUNS / 2];
    let slowest = times[RUNS - 1];
    println!(
        "start to first output of `syntropy version --json` over {RUNS} runs: median {median:?}, \
         fastest {:?}, slowest {slowest:?}; budget {BUDGET:?} [target]",
        times[0]
    );
    assert!(
        median <= BUDGET,
        "median {median:?} is over the {BUDGET:?} budget"
    );
}
