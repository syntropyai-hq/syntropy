// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runner process: its command loop, its heartbeat, and its watch on the daemon
//! (RFC-0001 §3.1, §3.4; RFC-0052 §6.5; DIST-E1-T04).
//!
//! The daemon spawns a runner by re-executing its own binary, `syntropy __runner --ipc <handle>
//! --region <handle>`, and the binary's wiring hands [`serve`] the channel and a way to open the
//! region. [`serve`] then:
//!
//! 1. reads the tick source's rate, which calibrates it on the process's first use, so the
//!    calibration of up to `TSC_CALIBRATION` (20 ms) never falls on a thread the runner starts;
//! 2. starts the runner's one background thread, `syntropy-runner-beat`, which watches the daemon
//!    ([`DaemonWatch`]) from the first moment;
//! 3. sends `Hello` and takes the daemon's `Init`, which names the region's layout;
//! 4. opens the region and hands it to the beat thread, which from then on, every [`HEARTBEAT`]
//!    (`runner.heartbeat`), beats the runner's heartbeat in the region's header and reads the
//!    daemon's;
//! 5. answers the daemon's commands until a `Drain` or a `Shutdown`, or until the channel reaches
//!    end-of-file.
//!
//! **Daemon loss** (RFC-0052 §6.5, orphan rule step 1). The runner sets no parent-death signal,
//! no exit watch and no kill-on-close job. It finds the daemon gone in one of two ways:
//!
//! - **End-of-file** on the command channel: the daemon's process ended, however it ended, and the
//!   OS closed its end. [`serve`] returns [`Ended::DaemonClosed`]. A connection reset or a broken
//!   pipe, which a read or a reply can meet instead when the daemon dies, counts the same.
//! - **Silence**: the daemon's heartbeat has not advanced for `runner.daemon_silence`
//!   ([`DAEMON_SILENCE`]) of awake time, as when the daemon is frozen (`SIGSTOP`) with its end
//!   still open. Before `Init`, when the runner has no region yet, the wait for `Init` counts as
//!   silence, so a daemon frozen between `Hello` and `Init` is found too. That bounds the time from
//!   the runner's start to its `Init` by the same silence: the daemon sends `Init` as soon as it has
//!   checked `Hello`, so only a daemon stalled for that long misses it, and the runner then exits as
//!   for any lost daemon. The beat thread calls the
//!   wiring's `on_silent`, which ends the process.
//!
//! Until orphan mode lands (DIST-E1-T08), either way the runner drains and exits. It holds no
//! sequence before a plan is assigned, and plans arrive with later tasks, so today the drain has
//! nothing to wait for.
//!
//! The silence is counted in awake time ([`AwakeClock::awake`]), which stops while the host is
//! suspended, so a laptop's sleep, which freezes the daemon and the runner alike, never reads as a
//! lost daemon (RFC-0033 §8.2: "suspend-aware").
//!
//! **The state word.** [`begin`] and [`advance`] run the runner's [`Lifecycle`] and write the
//! `RunnerState` word in the region's header on each transition (RFC-0001 §3.5), so the daemon
//! reads the runner's state from the header as well as from its `StateChanged` messages. [`serve`]
//! does not call them yet: the lifecycle starts with a plan (`AssignPlan`), which later tasks
//! handle.
//!
//! The runner's code reads no production clock or tick source itself (RFC-0060 §2.1, rule T1): the
//! wiring passes both in.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use syntropy_core::{AwakeClock, AwakeTime, Bytes, DrainReason, Nanos, PlanEpoch, TickSource};
use syntropy_ipc::cmd::{
    DaemonMsg, HandshakeError, Hello, RecvError, RunnerChannel, RunnerMsg, SendError, Transport,
};
use syntropy_ipc::shm::{Region, ShmError, ShmLayout};

use crate::lifecycle::{Event, Lifecycle, Refused};

/// `runner.daemon_silence`'s default, 2 s (RFC-0033 §8.2): how long the daemon's heartbeat may
/// stay still, in awake time, before the runner counts the daemon lost. `Init` carries no runner
/// configuration until PROD-E1-T07, so the wiring passes this default.
pub const DAEMON_SILENCE: Nanos = Nanos(2_000_000_000);

/// `runner.heartbeat`'s default, 100 ms (RFC-0033 §8.2): how often the beat thread beats the
/// runner's heartbeat and reads the daemon's. So a silence is seen within 100 ms of reaching
/// [`DAEMON_SILENCE`], and the supervisor's `runner.liveness_timeout` (3 s) spans thirty beats.
pub const HEARTBEAT: Duration = Duration::from_millis(100);

/// The watch on the daemon's heartbeat: sans-I/O, fed the heartbeat's value and the awake time it
/// was read at (RFC-0052 §6.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DaemonWatch {
    silence: Nanos,
    /// The last value seen, and the awake time it was first seen at.
    last: Option<(u64, AwakeTime)>,
}

impl DaemonWatch {
    /// A watch that counts the daemon lost after `silence` of awake time with no beat.
    #[must_use]
    pub const fn new(silence: Nanos) -> DaemonWatch {
        DaemonWatch {
            silence,
            last: None,
        }
    }

    /// Records the daemon's heartbeat `beat`, read at awake time `awake`. Returns how long the
    /// heartbeat has been still, once that reaches the watch's silence, and `None` before.
    ///
    /// Silence is counted from the first reading of the current value, so the first reading
    /// starts the count: a daemon that never beats is lost one silence after the runner starts
    /// watching. An awake time earlier than the last counts as no time.
    pub fn observe(&mut self, beat: u64, awake: AwakeTime) -> Option<Nanos> {
        match self.last {
            Some((seen, since)) if seen == beat => {
                let still = awake.saturating_duration_since(since);
                (still.0 >= self.silence.0).then_some(still)
            }
            _ => {
                self.last = Some((beat, awake));
                None
            }
        }
    }
}

/// Starts the runner's [`Lifecycle`] in `Fetching` for plan `epoch` and a shard of `shard` bytes,
/// and writes its state word in `region`'s header. Returns the machine and the `StateChanged` to
/// send.
pub fn begin(region: &Region, epoch: PlanEpoch, shard: Bytes) -> (Lifecycle, RunnerMsg) {
    let (machine, msg) = Lifecycle::start(epoch, shard);
    region.set_runner_state(machine.state().kind().discriminant());
    (machine, msg)
}

/// Applies `event` to `machine`. On a transition, writes the new state word in `region`'s header
/// and returns the `StateChanged` to send.
///
/// # Errors
///
/// [`Refused`] for an event the machine's state does not take; the header is left as it was.
pub fn advance(
    machine: &mut Lifecycle,
    event: Event,
    region: &Region,
) -> Result<Option<RunnerMsg>, Refused> {
    let msg = machine.apply(event)?;
    if msg.is_some() {
        region.set_runner_state(machine.state().kind().discriminant());
    }
    Ok(msg)
}

/// How [`serve`] ended. The runner exits after each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    /// The command channel reached end-of-file, or was reset: the daemon is gone (RFC-0052 §6.5).
    DaemonClosed,
    /// The daemon sent `Drain`.
    Drained(DrainReason),
    /// The daemon sent `Shutdown`.
    ShutDown,
}

/// Why [`serve`] failed.
#[derive(Debug)]
pub enum ServeError {
    /// The handshake failed: no `Init`, or an `Init` of another build.
    Handshake(HandshakeError),
    /// The region named by the command line could not be opened with `Init`'s layout.
    Region(ShmError),
    /// The beat thread could not be started.
    Thread(std::io::Error),
    /// A message could not be read.
    Recv(RecvError),
    /// A reply could not be sent.
    Send(SendError),
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServeError::Handshake(error) => {
                write!(f, "the handshake with the daemon failed: {error}")
            }
            ServeError::Region(error) => {
                write!(f, "the runner's region could not be opened: {error}")
            }
            ServeError::Thread(error) => write!(f, "the beat thread could not be started: {error}"),
            ServeError::Recv(error) => write!(f, "reading the command channel failed: {error}"),
            ServeError::Send(error) => write!(f, "writing the command channel failed: {error}"),
        }
    }
}

impl std::error::Error for ServeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ServeError::Handshake(error) => Some(error),
            ServeError::Region(error) => Some(error),
            ServeError::Thread(error) => Some(error),
            ServeError::Recv(error) => Some(error),
            ServeError::Send(error) => Some(error),
        }
    }
}

/// What the wiring gives [`serve`] besides the channel.
pub struct Start<K, C, O, S> {
    /// The tick source the runner's hot threads will read. [`serve`] reads its rate first.
    pub ticks: K,
    /// The runner's `Hello`: its process ID, build ID, protocol and observability-table digest.
    pub hello: Hello,
    /// Opens the runner's side of its region with `Init`'s layout.
    pub open_region: O,
    /// The clock the beat thread reads its awake time from.
    pub clock: C,
    /// `runner.daemon_silence`.
    pub silence: Nanos,
    /// Called once, on the beat thread, with the silence's length, when the daemon's heartbeat has
    /// been still for `silence`. The wiring drains and ends the process.
    pub on_silent: S,
}

/// Runs the runner's side of the command channel (module docs): the handshake, the beat thread,
/// and the commands, until the daemon drains the runner, shuts it down, or is gone.
///
/// # Errors
///
/// [`ServeError`] when the handshake, the region, the beat thread or the channel fails.
pub fn serve<T, K, C, O, S>(
    mut channel: RunnerChannel<T>,
    start: Start<K, C, O, S>,
) -> Result<Ended, ServeError>
where
    T: Transport,
    K: TickSource,
    C: AwakeClock + Send + 'static,
    O: FnOnce(&ShmLayout) -> Result<Region, ShmError>,
    S: FnOnce(Nanos) + Send + 'static,
{
    // The first use of the tick source calibrates it; converting a tick needs the rate.
    let _ = start.ticks.ticks_to_nanos(1);
    let beat =
        Beat::start(start.clock, start.silence, start.on_silent).map_err(ServeError::Thread)?;
    let init = channel
        .handshake(&start.hello)
        .map_err(ServeError::Handshake)?;
    let region = Arc::new((start.open_region)(&init.shm).map_err(ServeError::Region)?);
    beat.watch(region);
    loop {
        let envelope = match channel.recv() {
            Ok(envelope) => envelope,
            Err(RecvError::Closed) => return Ok(Ended::DaemonClosed),
            Err(RecvError::Io(error)) if gone(&error) => return Ok(Ended::DaemonClosed),
            Err(error) => return Err(ServeError::Recv(error)),
        };
        match envelope.msg {
            DaemonMsg::Ping => match channel.reply(envelope.id, &RunnerMsg::Pong) {
                Ok(_) => {}
                Err(SendError::Io(error)) if gone(&error) => return Ok(Ended::DaemonClosed),
                Err(error) => return Err(ServeError::Send(error)),
            },
            DaemonMsg::Drain { reason, .. } => return Ok(Ended::Drained(reason)),
            DaemonMsg::Shutdown { .. } => return Ok(Ended::ShutDown),
            other => tracing::warn!(
                message = message_name(&other),
                "this runner build does not handle the message yet; it is ignored"
            ),
        }
    }
}

/// Whether `error` means the daemon's end of the channel is gone: a reset, an abort or a broken
/// pipe, as a read or a write can meet when the daemon's process dies with data in flight, and on
/// Windows `ERROR_PIPE_NOT_CONNECTED` (233).
fn gone(error: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    matches!(
        error.kind(),
        ErrorKind::ConnectionReset
            | ErrorKind::ConnectionAborted
            | ErrorKind::BrokenPipe
            | ErrorKind::UnexpectedEof
    ) || (cfg!(windows) && error.raw_os_error() == Some(233))
}

/// A daemon message's name, for a log record.
fn message_name(msg: &DaemonMsg) -> &'static str {
    match msg {
        DaemonMsg::Init(_) => "Init",
        DaemonMsg::AssignPlan { .. } => "AssignPlan",
        DaemonMsg::ShardProgress { .. } => "ShardProgress",
        DaemonMsg::ApplyPlanDelta { .. } => "ApplyPlanDelta",
        DaemonMsg::CancelSequence { .. } => "CancelSequence",
        DaemonMsg::PeerRevoked { .. } => "PeerRevoked",
        DaemonMsg::Ping => "Ping",
        DaemonMsg::Drain { .. } => "Drain",
        DaemonMsg::Shutdown { .. } => "Shutdown",
        DaemonMsg::MemoryPressure { .. } => "MemoryPressure",
    }
}

/// The beat thread: the watch on the daemon, and once the region is open, the runner's heartbeat.
/// Dropping it stops the thread and waits for it, which takes at most one wake-up.
struct Beat {
    stop: Arc<AtomicBool>,
    /// The runner's region, set once `Init` has named its layout.
    region: Arc<OnceLock<Arc<Region>>>,
    thread: Option<JoinHandle<()>>,
}

/// The value the watch is fed before the region is open, in place of the daemon's heartbeat: a
/// counter that starts at zero and is bumped by one per beat never reaches it, so the first real
/// reading restarts the count.
const NO_REGION: u64 = u64::MAX;

impl Beat {
    fn start<C, S>(clock: C, silence: Nanos, on_silent: S) -> std::io::Result<Beat>
    where
        C: AwakeClock + Send + 'static,
        S: FnOnce(Nanos) + Send + 'static,
    {
        let stop = Arc::new(AtomicBool::new(false));
        let region = Arc::new(OnceLock::<Arc<Region>>::new());
        let thread = thread::Builder::new()
            .name(String::from("syntropy-runner-beat"))
            .spawn({
                let stop = Arc::clone(&stop);
                let region = Arc::clone(&region);
                move || {
                    let mut watch = DaemonWatch::new(silence);
                    while !stop.load(Ordering::Acquire) {
                        let daemon = match region.get() {
                            Some(region) => {
                                region.beat();
                                region.peer_heartbeat()
                            }
                            None => NO_REGION,
                        };
                        if let Some(still) = watch.observe(daemon, clock.awake()) {
                            on_silent(still);
                            return;
                        }
                        thread::park_timeout(HEARTBEAT);
                    }
                }
            })?;
        Ok(Beat {
            stop,
            region,
            thread: Some(thread),
        })
    }

    /// Hands the thread the open region: from its next wake-up it beats the runner's heartbeat and
    /// watches the daemon's.
    fn watch(&self, region: Arc<Region>) {
        let _ = self.region.set(region);
        if let Some(thread) = &self.thread {
            thread.thread().unpark();
        }
    }
}

impl Drop for Beat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            // A panic on the beat thread has already been reported by the panic hook.
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests;
