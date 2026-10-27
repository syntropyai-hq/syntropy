// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where a runner runs (RFC-0001 §3.1; ADR-0015 items 1 and 7; DIST-E1-T11).
//!
//! A runner is normally a process the daemon spawns, re-executing its own binary (DIST-E1-T04).
//! Where that is impossible, on iOS and in some Android contexts, and in the simulator, the
//! runner runs in the daemon's process on threads of its own: [`RunnerHost::InProcess`]. It
//! speaks the same IPC protocol, with the same framing and encoding, over an in-memory transport
//! ([`syntropy_ipc::cmd::memory`]) instead of a socket or a named pipe. That gives up crash
//! isolation and nothing else (ADR-0015 item 7).
//!
//! The shared-memory region is the same too: the host creates it on the daemon's side and opens
//! it on the runner's, in one process. So the header's words, the heartbeats, the step counter,
//! the state word and the live-sequence count, work unchanged, and the supervisor's
//! `RunnerLink::sample_region` reads an in-process runner exactly as it reads a process.
//!
//! The runner's code is in `runner`, which `daemon` cannot depend on (both are layer 5, RFC-0001
//! §4.1), so the host takes the runner's entry point from the binary's wiring:
//!
//! - [`connect_in_process`] builds both ends and hands back the runner's, for a caller that
//!   drives the runner itself, as the simulator does on one thread (QUAL-E8-T06);
//! - [`start_in_process`] runs a given entry point on a thread of its own, as an embedded node
//!   does.

use std::thread::{self, JoinHandle};

use syntropy_core::Entropy;
use syntropy_ipc::cmd::memory::MemoryStream;
use syntropy_ipc::cmd::{DaemonChannel, PROTOCOL, RunnerChannel, memory_channel};
use syntropy_ipc::shm::{Region, ShmError, ShmLayout, Side};

/// How a node hosts its runners.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunnerHost {
    /// On the daemon's own threads, over an in-memory transport. No crash isolation: the node
    /// reports `isolation: none` (ADR-0015 item 7; DIST-E1-T09). The process host, which
    /// re-executes the binary, comes with DIST-E1-T04.
    InProcess,
}

/// The daemon's end of an in-process runner: its command channel and its side of the region.
#[derive(Debug)]
pub struct DaemonEnd {
    /// The command channel to the runner.
    pub channel: DaemonChannel<MemoryStream>,
    /// The daemon's side of the runner's region.
    pub region: Region,
}

/// The runner's end: what the runner's code is given in place of `--ipc <handle>`.
#[derive(Debug)]
pub struct RunnerEnd {
    /// The command channel to the daemon.
    pub channel: RunnerChannel<MemoryStream>,
    /// The runner's side of its region.
    pub region: Region,
}

/// Builds an in-process runner's two ends: a connected command channel, and a region of
/// `layout` created on the daemon's side and opened on the runner's.
///
/// # Errors
///
/// [`ShmError`] when the region cannot be created or opened.
pub fn connect_in_process(
    layout: &ShmLayout,
    build_id: [u8; 32],
    entropy: &dyn Entropy,
) -> Result<(DaemonEnd, RunnerEnd), ShmError> {
    let region = Region::create(layout, build_id, PROTOCOL, entropy)?;
    let runner_region = Region::open(&region.handle(), Side::Runner, layout, PROTOCOL)?;
    let (daemon, runner) = memory_channel();
    Ok((
        DaemonEnd {
            channel: daemon,
            region,
        },
        RunnerEnd {
            channel: runner,
            region: runner_region,
        },
    ))
}

/// A runner running in-process on a thread of its own.
#[derive(Debug)]
pub struct InProcessRunner {
    /// The daemon's end.
    pub end: DaemonEnd,
    thread: JoinHandle<()>,
}

impl InProcessRunner {
    /// Waits for the runner's thread to end. A runner that panicked is the in-process form of a
    /// crash, reported as the panic.
    ///
    /// # Errors
    ///
    /// The panic's payload, if the runner panicked.
    pub fn join(self) -> thread::Result<()> {
        let InProcessRunner { end, thread } = self;
        // Closing the daemon's end first lets a runner blocked on the channel see end-of-file.
        drop(end);
        thread.join()
    }

    /// Whether the runner's thread has ended.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.thread.is_finished()
    }
}

/// Starts `runner`, the runner's entry point, on a thread named `syntropy-runner`, with the
/// runner's end of a new in-process connection.
///
/// # Errors
///
/// [`ShmError`] when the region cannot be built, and the I/O error as
/// [`ShmError::Os`] when the thread cannot be started.
pub fn start_in_process(
    layout: &ShmLayout,
    build_id: [u8; 32],
    entropy: &dyn Entropy,
    runner: impl FnOnce(RunnerEnd) + Send + 'static,
) -> Result<InProcessRunner, ShmError> {
    let (end, runner_end) = connect_in_process(layout, build_id, entropy)?;
    let thread = thread::Builder::new()
        .name(String::from("syntropy-runner"))
        .spawn(move || runner(runner_end))
        .map_err(ShmError::Os)?;
    Ok(InProcessRunner { end, thread })
}

#[cfg(test)]
mod tests {
    use syntropy_ipc::cmd::{DaemonMsg, RunnerMsg};

    use super::*;

    struct Fixed;

    impl Entropy for Fixed {
        fn fill(&self, buf: &mut [u8]) {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(31) ^ 0xa5;
            }
        }
    }

    /// A ping crosses to a runner thread and its pong comes back; the header word the runner
    /// writes is the one the daemon reads; the runner sees end-of-file when the daemon closes.
    #[test]
    fn an_in_process_runner_shares_the_channel_and_the_region() {
        let runner = start_in_process(&ShmLayout::default(), [4; 32], &Fixed, |mut end| {
            let ping = end.channel.recv().unwrap();
            assert_eq!(ping.msg, DaemonMsg::Ping);
            end.region.beat();
            end.region.set_live_sequences(3);
            end.channel.reply(ping.id, &RunnerMsg::Pong).unwrap();
            assert!(matches!(
                end.channel.recv(),
                Err(syntropy_ipc::cmd::RecvError::Closed)
            ));
        })
        .unwrap();
        let mut end_channel = runner;
        let id = end_channel.end.channel.send(&DaemonMsg::Ping).unwrap();
        let pong = end_channel.end.channel.recv().unwrap();
        assert_eq!((pong.reply_to, pong.msg), (Some(id), RunnerMsg::Pong));
        assert_eq!(end_channel.end.region.peer_heartbeat(), 1);
        assert_eq!(end_channel.end.region.live_sequences(), 3);
        assert_eq!(end_channel.end.region.build_id(), [4; 32]);
        end_channel.join().expect("the runner thread ends cleanly");
        assert_eq!(RunnerHost::InProcess, RunnerHost::InProcess);
    }
}
