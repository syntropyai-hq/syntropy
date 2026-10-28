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
use std::time::Duration;

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

/// How long dropping an [`InProcessRunner`] waits for its thread after closing the channel:
/// [`DROP_WAIT_STEPS`] waits of [`DROP_WAIT_STEP`], 2 s in all \[chosen\]. The wait counts sleeps
/// rather than reading a clock (RFC-0060 §2.1, rule T1).
pub const DROP_WAIT_STEP: Duration = Duration::from_millis(10);

/// See [`DROP_WAIT_STEP`].
pub const DROP_WAIT_STEPS: u32 = 200;

/// A runner running in-process on a thread of its own.
///
/// **Dropping it ends the runner.** The drop closes the daemon's end, so a runner reading its
/// channel sees end-of-file and returns. It then waits for the thread for at most
/// [`DROP_WAIT_STEPS`] × [`DROP_WAIT_STEP`]. A runner that has not ended by then is logged and
/// detached, so a drop never hangs. A runner that panicked is logged; [`join`](Self::join)
/// returns the panic instead.
#[derive(Debug)]
pub struct InProcessRunner {
    /// The daemon's end; `None` once closed.
    end: Option<DaemonEnd>,
    /// The runner's thread; `None` once joined or detached.
    thread: Option<JoinHandle<()>>,
}

impl InProcessRunner {
    /// The daemon's end.
    #[must_use]
    pub fn end(&self) -> &DaemonEnd {
        self.end
            .as_ref()
            .expect("the end is open until the runner is joined or dropped")
    }

    /// The daemon's end, to send and receive.
    pub fn end_mut(&mut self) -> &mut DaemonEnd {
        self.end
            .as_mut()
            .expect("the end is open until the runner is joined or dropped")
    }

    /// Closes the daemon's end, then waits for the runner's thread to end, for as long as it
    /// takes. A runner that panicked is the in-process form of a crash, reported as its panic.
    ///
    /// # Errors
    ///
    /// The panic's payload, if the runner panicked.
    pub fn join(mut self) -> thread::Result<()> {
        // Closing the daemon's end first lets a runner blocked on the channel see end-of-file.
        drop(self.end.take());
        match self.thread.take() {
            Some(thread) => thread.join(),
            None => Ok(()),
        }
    }

    /// Whether the runner's thread has ended.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.thread.as_ref().is_none_or(JoinHandle::is_finished)
    }
}

impl Drop for InProcessRunner {
    fn drop(&mut self) {
        drop(self.end.take());
        let Some(thread) = self.thread.take() else {
            return;
        };
        for _ in 0..DROP_WAIT_STEPS {
            if thread.is_finished() {
                break;
            }
            thread::sleep(DROP_WAIT_STEP);
        }
        if !thread.is_finished() {
            tracing::warn!(
                waited_ms = u64::from(DROP_WAIT_STEPS)
                    * u64::try_from(DROP_WAIT_STEP.as_millis()).unwrap_or(u64::MAX),
                "the in-process runner did not end after its channel closed; its thread is detached"
            );
            return;
        }
        if let Err(panic) = thread.join() {
            tracing::error!(
                panic = panic_message(&panic),
                "the in-process runner panicked"
            );
        }
    }
}

/// A panic payload as text, when it is one.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a panic with no message")
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
    Ok(InProcessRunner {
        end: Some(end),
        thread: Some(thread),
    })
}

#[cfg(test)]
mod tests {
    use syntropy_ipc::cmd::{DaemonMsg, RunnerMsg};

    use super::*;

    /// Bytes that differ on every call, in every test and process: this process's ID and a
    /// counter, spread by a mixing step, as `ipc`'s own tests name their segments. Segment names on
    /// Windows and Apple platforms are global to the session, so tests that run in parallel, or two
    /// test processes, must never draw the same name. Production passes the OS entropy source.
    struct Varying;

    static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    impl Entropy for Varying {
        fn fill(&self, buf: &mut [u8]) {
            let n = DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut x =
                (u64::from(std::process::id()) << 32 | n).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            for byte in buf {
                x ^= x >> 29;
                x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                *byte = (x >> 56) as u8;
            }
        }
    }

    /// A ping crosses to a runner thread and its pong comes back; the header word the runner
    /// writes is the one the daemon reads; the runner sees end-of-file when the daemon closes.
    #[test]
    fn an_in_process_runner_shares_the_channel_and_the_region() {
        let runner = start_in_process(&ShmLayout::default(), [4; 32], &Varying, |mut end| {
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
        let id = end_channel
            .end_mut()
            .channel
            .send(&DaemonMsg::Ping)
            .unwrap();
        let pong = end_channel.end_mut().channel.recv().unwrap();
        assert_eq!((pong.reply_to, pong.msg), (Some(id), RunnerMsg::Pong));
        assert_eq!(end_channel.end().region.peer_heartbeat(), 1);
        assert_eq!(end_channel.end().region.live_sequences(), 3);
        assert_eq!(end_channel.end().region.build_id(), [4; 32]);
        end_channel.join().expect("the runner thread ends cleanly");
        assert_eq!(RunnerHost::InProcess, RunnerHost::InProcess);
    }

    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn start(runner: impl FnOnce(RunnerEnd) + Send + 'static) -> InProcessRunner {
        start_in_process(&ShmLayout::default(), [4; 32], &Varying, runner).unwrap()
    }

    /// Dropping a host whose runner has already returned joins it at once.
    #[test]
    fn a_drop_after_the_runner_exits_joins_it() {
        let ended = Arc::new(AtomicBool::new(false));
        let host = start({
            let ended = Arc::clone(&ended);
            move |_end| ended.store(true, Ordering::Release)
        });
        while !host.is_finished() {
            thread::sleep(DROP_WAIT_STEP);
        }
        drop(host);
        assert!(ended.load(Ordering::Acquire));
    }

    /// Dropping a host whose runner is blocked reading its channel closes the channel: the runner
    /// reads end-of-file, returns, and the drop joins it. The runner takes a moment after the
    /// end-of-file before it records it, so only a drop that waits for the thread sees the record:
    /// a drop that closed the channel late, or detached the thread, returns first.
    #[test]
    fn a_drop_while_the_runner_reads_closes_its_channel_and_joins_it() {
        let saw_eof = Arc::new(AtomicBool::new(false));
        let host = start({
            let saw_eof = Arc::clone(&saw_eof);
            move |mut end| {
                let read = end.channel.recv();
                thread::sleep(DROP_WAIT_STEP * 20);
                saw_eof.store(
                    matches!(read, Err(syntropy_ipc::cmd::RecvError::Closed)),
                    Ordering::Release,
                );
            }
        });
        drop(host);
        assert!(
            saw_eof.load(Ordering::Acquire),
            "the runner saw end-of-file, and the drop joined it"
        );
    }

    /// A runner that panics: `join` returns its panic. A drop logs it instead of raising it.
    #[test]
    fn a_runner_panic_surfaces_through_join() {
        let host = start(|_end| panic!("the runner failed"));
        let panic = host.join().expect_err("the panic");
        assert_eq!(panic_message(&*panic), "the runner failed");
        let host = start(|_end| panic!("another failure"));
        drop(host);
    }

    /// A runner that ignores end-of-file is detached once the drop's bounded wait is over: the
    /// drop returns while the runner still runs.
    #[test]
    fn a_drop_never_hangs_on_a_runner_that_does_not_end() {
        let stop = Arc::new(AtomicBool::new(false));
        let ended = Arc::new(AtomicBool::new(false));
        let host = start({
            let (stop, ended) = (Arc::clone(&stop), Arc::clone(&ended));
            move |_end| {
                while !stop.load(Ordering::Acquire) {
                    thread::sleep(DROP_WAIT_STEP);
                }
                ended.store(true, Ordering::Release);
            }
        });
        drop(host);
        assert!(
            !ended.load(Ordering::Acquire),
            "the drop returned while the runner ran"
        );
        stop.store(true, Ordering::Release);
    }
}
