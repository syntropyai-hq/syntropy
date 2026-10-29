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
//! **The process host** ([`RunnerHost::Process`], DIST-E1-T04). [`spawn_process`] re-executes the
//! binary as `__runner --ipc <handle> --region <handle>`, with both handles inherited: the runner's
//! end of a connected pair ([`syntropy_ipc::cmd::spawn_pair`]) and the region's descriptor, or its
//! name where the OS shares regions by name. It then beats the daemon's heartbeat in the region's
//! header from a dedicated thread, `syntropy-daemon-beat`, which never touches the `ctl` runtime,
//! so a busy control plane never looks dead to the runner (RFC-0052 §6.5). The runner needs no
//! parent-death signal: it reads end-of-file when the daemon's process ends, however it ends, and
//! a still heartbeat when the daemon is frozen. When the runner exits, the host removes the
//! region's name ([`Region::unlink`]).
//!
//! The runner's code is in `runner`, which `daemon` cannot depend on (both are layer 5, RFC-0001
//! §4.1), so the in-process host takes the runner's entry point from the binary's wiring:
//!
//! - [`connect_in_process`] builds both ends and hands back the runner's, for a caller that
//!   drives the runner itself, as the simulator does on one thread (QUAL-E8-T06);
//! - [`start_in_process`] runs a given entry point on a thread of its own, as an embedded node
//!   does.

use std::io;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use syntropy_core::Entropy;

use crate::crash::{CrashSink, HostKill};
use syntropy_ipc::cmd::memory::MemoryStream;
use syntropy_ipc::cmd::{
    DaemonChannel, Hello, PROTOCOL, Refused, RunnerChannel, Stream, memory_channel, spawn_pair,
};
use syntropy_ipc::shm::{Region, ShmError, ShmLayout, Side};

/// How a node hosts its runners.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunnerHost {
    /// A process of its own, re-executing the daemon's binary ([`spawn_process`]). The default
    /// wherever the OS lets the daemon spawn a process (ADR-0015 item 1).
    Process,
    /// On the daemon's own threads, over an in-memory transport. No crash isolation: the node
    /// reports `isolation: none` (ADR-0015 item 7; DIST-E1-T09).
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

/// How often the daemon beats its heartbeat in a runner's region: 250 ms \[chosen\], eight beats
/// per `runner.daemon_silence` (2 s, RFC-0033 §8.2), so a beat or two lost to scheduling never
/// reads as a lost daemon.
pub const DAEMON_BEAT: Duration = Duration::from_millis(250);

/// The daemon's heartbeat in one runner's region, beaten from a thread of its own. Dropping it
/// stops the thread and waits for it, which takes at most one wake-up.
#[derive(Debug)]
struct DaemonBeat {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl DaemonBeat {
    fn start(region: Arc<Region>) -> io::Result<DaemonBeat> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread = thread::Builder::new()
            .name(String::from("syntropy-daemon-beat"))
            .spawn({
                let stop = Arc::clone(&stop);
                move || {
                    while !stop.load(Ordering::Acquire) {
                        region.beat();
                        thread::park_timeout(DAEMON_BEAT);
                    }
                }
            })?;
        Ok(DaemonBeat {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for DaemonBeat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            // A panic on the beat thread has already been reported by the panic hook.
            let _ = thread.join();
        }
    }
}

/// A runner in a process of its own, which [`spawn_process`] started.
///
/// **Dropping it ends the runner without a signal.** The drop closes the daemon's end of the
/// channel, so the runner reads end-of-file and exits, and stops the daemon's heartbeat. It then
/// waits for the process for at most [`DROP_WAIT_STEPS`] × [`DROP_WAIT_STEP`], blocking its thread
/// for up to 2 s, and removes the region's name. A runner that has not exited by then is logged and
/// left to finish on its own, and a detached thread reaps it when it does; the drop never kills it.
#[derive(Debug)]
pub struct ProcessRunner {
    /// The process; `None` only once the drop has handed it to a reaper thread.
    child: Option<Child>,
    pid: u32,
    /// The command channel; `None` once closed.
    channel: Option<DaemonChannel<Stream>>,
    region: Arc<Region>,
    /// Whether the region's name has been removed.
    unlinked: bool,
    /// The daemon's heartbeat; `None` once the runner has exited.
    beat: Option<DaemonBeat>,
    build_id: [u8; 32],
    /// The exit status, once reaped.
    exited: Option<ExitStatus>,
    /// Where a crash exit is recorded, when the host was given one (RFC-0053 §6).
    crashes: Option<CrashSink>,
    /// Why the host killed the runner, once it has (PROD-E9-T12).
    killed: Option<HostKill>,
}

impl ProcessRunner {
    /// The runner's process ID.
    #[must_use]
    pub fn id(&self) -> u32 {
        self.pid
    }

    /// The daemon's side of the runner's region.
    #[must_use]
    pub fn region(&self) -> &Region {
        &self.region
    }

    /// Whether the host has removed the region's name ([`Region::unlink`]), which it does when it
    /// reaps the runner.
    #[must_use]
    pub fn region_unlinked(&self) -> bool {
        self.unlinked
    }

    /// The command channel to the runner.
    ///
    /// # Panics
    ///
    /// After [`close`](Self::close).
    pub fn channel(&mut self) -> &mut DaemonChannel<Stream> {
        self.channel
            .as_mut()
            .expect("the channel is open until it is closed")
    }

    fn child(&mut self) -> &mut Child {
        self.child
            .as_mut()
            .expect("the process is held until the drop")
    }

    /// Waits for the runner's `Hello`, and checks its build, protocol and process ID
    /// ([`DaemonChannel::handshake`]). On a refusal the runner is killed and reaped, and the
    /// region's name removed. The daemon sends `Init` next.
    ///
    /// # Errors
    ///
    /// [`Refused`], with the cause and the kill's outcome.
    pub fn handshake(&mut self) -> Result<Hello, Refused> {
        let build_id = self.build_id;
        let child = self
            .child
            .as_mut()
            .expect("the process is held until the drop");
        let channel = self
            .channel
            .as_mut()
            .expect("the channel is open until it is closed");
        let result = channel.handshake(child, &build_id);
        if let Err(refused) = &result
            && let Ok(status) = &refused.exit
        {
            // The refusal killed the runner (RFC-0001 §3.4: a spawned runner of another build is
            // killed), so its exit is the host's kill.
            self.killed = Some(HostKill::Refused);
            self.reaped(*status);
        }
        result
    }

    /// Closes the daemon's end of the channel: the runner reads end-of-file and exits.
    pub fn close(&mut self) {
        drop(self.channel.take());
    }

    /// Kills the runner's process for `reason`, as the supervisor does for a hang or a stall
    /// (RFC-0001 §3.4; `HostKill::Supervisor`, which DIST-E1-T16 wires from the supervisor
    /// shell's `Effects::kill`). Reap it with [`wait`](Self::wait) or
    /// [`try_wait`](Self::try_wait). With a [`CrashSink`], the reap records the kill with its
    /// reason, on every OS (PROD-E9-T12). A second kill keeps the first reason.
    ///
    /// # Errors
    ///
    /// The OS error. Killing a process that has already exited is not one.
    pub fn kill(&mut self, reason: HostKill) -> io::Result<()> {
        if self.exited.is_some() {
            return Ok(());
        }
        self.killed.get_or_insert(reason);
        self.child().kill()
    }

    /// Whether the runner has exited, reaping it if it has.
    ///
    /// # Errors
    ///
    /// The OS error.
    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.exited {
            return Ok(Some(status));
        }
        let status = self.child().try_wait()?;
        if let Some(status) = status {
            self.reaped(status);
        }
        Ok(status)
    }

    /// Waits for the runner to exit, and reaps it.
    ///
    /// # Errors
    ///
    /// The OS error.
    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        if let Some(status) = self.exited {
            return Ok(status);
        }
        let status = self.child().wait()?;
        self.reaped(status);
        Ok(status)
    }

    /// Records the runner's exit in `sink` when it is reaped and the exit is a crash: a kill the
    /// host made ([`kill`](Self::kill), or a refused handshake), with its reason, on every OS;
    /// otherwise a signal on Unix or a system `NTSTATUS` error on Windows (RFC-0053 §6). A clean
    /// exit records nothing. The record goes in the state directory's `crashes/`, which `doctor`
    /// and the diagnostic bundle read.
    pub fn record_crashes(&mut self, sink: CrashSink) {
        self.crashes = Some(sink);
    }

    /// The runner has exited: record a crash exit, stop its heartbeat and remove the region's
    /// name.
    fn reaped(&mut self, status: ExitStatus) {
        self.exited = Some(status);
        if let Some(sink) = &self.crashes {
            // A kill the host made is recorded with its reason on every OS; any other exit only
            // when its status shows a crash.
            let written = match self.killed {
                Some(reason) => sink.record_host_kill(self.pid, reason, &status).map(Some),
                None => sink.record_exit(self.pid, &status),
            };
            if let Err(error) = written {
                tracing::warn!(%error, pid = self.pid, "writing a runner's crash record failed");
            }
        }
        drop(self.beat.take());
        self.unlink();
    }

    fn unlink(&mut self) {
        match self.region.unlink() {
            Ok(()) => self.unlinked = true,
            Err(error) => tracing::warn!(%error, "removing a runner's region name failed"),
        }
    }
}

impl Drop for ProcessRunner {
    fn drop(&mut self) {
        self.close();
        drop(self.beat.take());
        for _ in 0..DROP_WAIT_STEPS {
            match self.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => thread::sleep(DROP_WAIT_STEP),
                Err(error) => {
                    tracing::warn!(%error, pid = self.pid, "waiting for a runner failed");
                    break;
                }
            }
        }
        if self.exited.is_some() {
            return;
        }
        self.unlink();
        tracing::warn!(
            pid = self.pid,
            "the runner did not exit after its channel closed; it is left to exit on its own"
        );
        // Reap it when it exits, so it never stays a zombie.
        if let Some(mut child) = self.child.take() {
            let reaper = thread::Builder::new()
                .name(String::from("syntropy-runner-reaper"))
                .spawn(move || {
                    let _ = child.wait();
                });
            if let Err(error) = reaper {
                tracing::warn!(%error, pid = self.pid, "the runner's reaper did not start");
            }
        }
    }
}

/// Windows' `CREATE_NO_WINDOW` process-creation flag: a runner gets no console window of its own
/// when the daemon has none, as under the GUI-subsystem launcher (RFC-0001 §3.1).
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Spawns a runner by executing `program`, the daemon's own binary, as `program __runner --ipc
/// <handle> --region <handle>` (module docs). The binary's wiring passes
/// `std::env::current_exe()`. The region is built with `layout`, `build_id` and [`PROTOCOL`], its
/// name and the pipe's name, where the OS names them, drawn from `entropy`.
///
/// The runner's standard input is null; its standard output and error are the daemon's, until its
/// records go through its rings (RFC-0053 §4). On Windows it gets no console window.
///
/// The caller then calls [`ProcessRunner::handshake`] and sends `Init`.
///
/// # Errors
///
/// [`ShmError`] when the region cannot be built, and the I/O error as [`ShmError::Os`] when the
/// channel, the heartbeat's thread or the process cannot be started. On every error after the
/// region is built, its name is removed.
pub fn spawn_process(
    program: &Path,
    layout: &ShmLayout,
    build_id: [u8; 32],
    entropy: &dyn Entropy,
) -> Result<ProcessRunner, ShmError> {
    let region = Region::create(layout, build_id, PROTOCOL, entropy)?;
    let spawned = spawn_with(program, &region, entropy);
    let (child, daemon_end) = match spawned {
        Ok(spawned) => spawned,
        Err(error) => {
            if let Err(unlink) = region.unlink() {
                tracing::warn!(error = %unlink, "removing an unspawned runner's region name failed");
            }
            return Err(ShmError::Os(error));
        }
    };
    let region = Arc::new(region);
    let mut runner = ProcessRunner {
        pid: child.id(),
        child: Some(child),
        channel: Some(DaemonChannel::new(daemon_end)),
        region: Arc::clone(&region),
        unlinked: false,
        beat: None,
        build_id,
        exited: None,
        crashes: None,
        killed: None,
    };
    // On an error the drop closes the channel, so the runner exits on its own, and removes the
    // region's name.
    runner.beat = Some(DaemonBeat::start(region).map_err(ShmError::Os)?);
    Ok(runner)
}

/// Builds the channel, and spawns `program` with the runner's end and `region`.
fn spawn_with(
    program: &Path,
    region: &Region,
    entropy: &dyn Entropy,
) -> io::Result<(Child, Stream)> {
    let mut command = Command::new(program);
    command.arg("__runner").stdin(Stdio::null());
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut command, CREATE_NO_WINDOW);
    let (daemon_end, child_end) = spawn_pair(&mut command, entropy)?;
    let region_handle = region.configure_spawn(&mut command)?;
    command
        .arg("--ipc")
        .arg(child_end.handle().to_string())
        .arg("--region")
        .arg(region_handle.to_string());
    let spawned = command.spawn();
    // The command goes first, with the hook that names the child's end by number, then the
    // daemon's copy of the runner's end, so the runner's exit closes the pair.
    drop(command);
    drop(child_end);
    Ok((spawned?, daemon_end))
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
