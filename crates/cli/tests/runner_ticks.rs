// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runner mode reads the tick source's rate before it starts a thread (DIST-E1-T04), so the
//! calibration of up to `TSC_CALIBRATION` (20 ms) never falls on a thread the runner starts.
//! `syntropy_runner::process::serve` converts one tick first, which needs the rate; its unit test
//! `serve_calibrates_the_tick_source_first` shows it does so before anything else. This test shows
//! that with the production tick source, that conversion leaves a new thread's first read fast.
//!
//! This file holds one test, so its process has read no tick source before it.

use std::time::{Duration, Instant};

use syntropy_core::TickSource;
use syntropy_io::os::SystemTicks;

#[test]
fn a_thread_started_after_calibration_reads_ticks_at_once() {
    // What `serve` does first.
    let _ = SystemTicks::new().ticks_to_nanos(1);
    let first_read = std::thread::spawn(|| {
        let start = Instant::now();
        let ticks = SystemTicks::new().ticks();
        (start.elapsed(), ticks)
    })
    .join()
    .unwrap();
    assert!(
        first_read.0 < Duration::from_millis(1),
        "the first ticks() read on a new thread took {:?}",
        first_read.0
    );
}
