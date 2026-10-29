// SPDX-License-Identifier: MIT OR Apache-2.0

//! Probe mode: `syntropy __runner --probe --ipc <handle>` (RFC-0001 §3.1; DIST-E1-T17).
//!
//! The daemon spawns a probe process through the same process host as a runner
//! (`daemon::runner_host::probe_process`), for the work that needs a driver context: device
//! enumeration and the startup microbenchmarks (RFC-0010). The process sends `Hello`, then one
//! `ProbeResult`, and exits. It takes no shared-memory region and waits for no `Init`, so it has
//! no plan, loads no model and admits no sequence.
//!
//! What it measures is the caller's: this module only carries the result. Until PLC-E1-T07 runs
//! its enumeration here, the profile names no device; PLC-E2-T01 adds `ProbeRequest` and its
//! suites. The profile's `node` and `version` are the daemon's to stamp.

use std::fmt;

use syntropy_core::resource::CapabilityProfile;
use syntropy_ipc::cmd::{Hello, RunnerChannel, RunnerMsg, SendError, Transport};

/// Why a probe could not report.
#[derive(Debug)]
pub enum ProbeError {
    /// `Hello` could not be sent.
    Hello(SendError),
    /// `ProbeResult` could not be sent.
    Result(SendError),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbeError::Hello(e) => write!(f, "the probe's Hello failed: {e}"),
            ProbeError::Result(e) => write!(f, "the probe's ProbeResult failed: {e}"),
        }
    }
}

impl std::error::Error for ProbeError {}

/// Runs a probe over `channel`: sends `hello`, measures with `measure`, sends the profile as
/// `ProbeResult`, and returns, after which the process exits (module docs).
///
/// # Errors
///
/// [`ProbeError`] when either message cannot be sent, as when the daemon is gone.
pub fn probe<T: Transport>(
    mut channel: RunnerChannel<T>,
    hello: &Hello,
    measure: impl FnOnce() -> CapabilityProfile,
) -> Result<(), ProbeError> {
    channel
        .send(&RunnerMsg::Hello(*hello))
        .map_err(ProbeError::Hello)?;
    let profile = measure();
    channel
        .send(&RunnerMsg::ProbeResult { profile })
        .map_err(ProbeError::Result)?;
    Ok(())
}

/// The profile a probe sends before any measurement exists: no devices, no storage, no links,
/// `node` and `version` left for the daemon (module docs).
#[must_use]
pub fn unmeasured() -> CapabilityProfile {
    CapabilityProfile {
        node: syntropy_core::NodeId([0; 32]),
        version: 0,
        devices: Vec::new(),
        storage: Vec::new(),
        links: Vec::new(),
        ext: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use syntropy_core::Blake3;
    use syntropy_ipc::cmd::{PROTOCOL, RecvError, memory_channel};

    use super::*;

    /// A probe sends `Hello`, then `ProbeResult`, and nothing else; it returns without waiting for
    /// `Init` or anything else from the daemon, so it never holds a plan.
    #[test]
    fn a_probe_sends_hello_then_its_result_and_nothing_more() {
        let (mut daemon, runner) = memory_channel();
        let hello = Hello {
            pid: 7,
            build_id: [3; 32],
            ipc_protocol: PROTOCOL,
            obs_tables: Blake3([0; 32]),
        };
        let mut measured = false;
        probe(runner, &hello, || {
            measured = true;
            let mut profile = unmeasured();
            profile.version = 9;
            profile
        })
        .unwrap();
        assert!(measured);
        assert_eq!(daemon.recv().unwrap().msg, RunnerMsg::Hello(hello));
        let RunnerMsg::ProbeResult { profile } = daemon.recv().unwrap().msg else {
            panic!("ProbeResult follows Hello");
        };
        assert_eq!(profile.version, 9);
        assert!(profile.devices.is_empty());
        // The runner's end is gone: the probe sent nothing more and waited for nothing.
        assert!(matches!(daemon.recv(), Err(RecvError::Closed)));
    }
}
