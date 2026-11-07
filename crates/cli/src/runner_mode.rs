// SPDX-License-Identifier: MIT OR Apache-2.0

//! `syntropy __runner --ipc <handle> --region <handle>`: runner mode, the process the daemon
//! spawns for each model instance by re-executing this binary (RFC-0001 §3.1; ADR-0015 decision 1;
//! DIST-E1-T04). With `--probe` and no region it is probe mode instead: it sends `Hello` and one
//! `ProbeResult`, and exits 0 (`syntropy_runner::probe`; DIST-E1-T17).
//!
//! This is the runner's wiring, the one place its production sources are built (RFC-0060 §2.1,
//! rule T1). It takes the inherited command channel ([`syntropy_ipc::cmd::inherited`]) and runs
//! [`syntropy_runner::process::serve`] with this build's ID, the region the command line names,
//! `io`'s `SystemTicks`, whose rate `serve` reads before it starts a thread, `io`'s `SystemClock`
//! for the daemon watch, and `runner.daemon_silence`'s default.
//!
//! The runner exits 0 when the daemon drains it, shuts it down or is gone: end-of-file on the
//! channel, or a still heartbeat, after which the beat thread ends the process (RFC-0052 §6.5).
//! It exits 1 when the channel, the handshake or the region fails. The daemon's supervisor reads
//! no exit code today (`Input::Exited` carries none); it classifies an exit by the runner's state
//! and reports (RFC-0001 §3.4). It sets no parent-death signal, exit watch or kill-on-close
//! job.

use syntropy_core::{Blake3, Nanos};
use syntropy_io::os::{SystemClock, SystemTicks};
use syntropy_ipc::cmd::{Hello, PROTOCOL, RunnerChannel, inherited};
use syntropy_ipc::shm::{Region, ShmLayout, Side};
use syntropy_runner::probe::{probe, unmeasured};
use syntropy_runner::process::{DAEMON_SILENCE, Ended, Start, serve};

use crate::args::RunnerArgs;
use crate::build_id;
use crate::error::{CliError, Exit};

/// Runs runner mode (module docs).
///
/// # Errors
///
/// [`CliError`] (exit 1) when the inherited channel cannot be taken, or the handshake, the region
/// or the channel fails.
pub fn run(args: &RunnerArgs) -> Result<(), CliError> {
    let failed = |message: String| CliError::cli(Exit::Failure, "runner", message);
    let stream = inherited(&args.ipc).map_err(|error| failed(error.to_string()))?;
    let hello = Hello {
        pid: std::process::id(),
        build_id: build_id::id(),
        ipc_protocol: PROTOCOL,
        // RFC-0053 §4.1's log-callsite, metric and span-name tables have no builder yet.
        obs_tables: Blake3([0; 32]),
    };
    let Some(region) = args.region.clone() else {
        // Probe mode: no region, no plan. The enumeration is PLC-E1-T07's.
        return probe(RunnerChannel::new(stream), &hello, unmeasured)
            .map_err(|error| failed(error.to_string()));
    };
    let start = Start {
        ticks: SystemTicks::new(),
        hello,
        open_region: move |layout: &ShmLayout| {
            Region::open(&region, Side::Runner, layout, PROTOCOL)
        },
        clock: SystemClock::new(),
        silence: DAEMON_SILENCE,
        on_silent: |still| daemon_silent(still),
    };
    match serve(RunnerChannel::new(stream), start) {
        Ok(Ended::DaemonClosed | Ended::Drained(_) | Ended::ShutDown) => Ok(()),
        Err(error) => Err(failed(error.to_string())),
    }
}

/// The daemon's heartbeat has been still for `still`: the runner holds no sequence to drain yet,
/// so it exits.
fn daemon_silent(still: Nanos) -> ! {
    eprintln!(
        "syntropy runner {}: the daemon's heartbeat has been still for {} ms; the runner exits \
         (RFC-0052 §6.5)",
        std::process::id(),
        still.0 / 1_000_000
    );
    std::process::exit(i32::from(Exit::Success.code()))
}
