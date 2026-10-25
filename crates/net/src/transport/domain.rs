// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `net` domain: the single-threaded tokio runtime the QUIC data plane runs on, on a thread
//! of its own (RFC-0031 §4.4; ADR-0012 items 1, 2 and 5; ADR-0308 item 5).
//!
//! - **One thread, one runtime.** [`NetDomain::start`] spawns the thread and runs a
//!   current-thread runtime on it, so control traffic, which runs on the process's other runtime,
//!   never shares it. Reserving and pinning the thread's core is the runner's, as it is for the
//!   dispatchers (RFC-0001 §8.1): the thread's name lets it find the thread.
//! - **Hand-off.** Dispatchers reach channels only through `syntropy-spsc` rings, never a lock
//!   ([`stream`](super::stream)). A ring into the domain signals its consumer's wake handle,
//!   which the channel's task awaits, so a parked channel wakes without polling (ADR-0012 item 4).
//! - **Polling** (RFC-0031 §4.4). While the runner has a `Ready` instance with live sequences
//!   ([`NetDomain::set_live`]), a channel's task does not park after its last frame: it keeps
//!   polling its rings for the spin budget, [`NetDomain::set_spin`] (ADR-0012 item 5: 50 µs, 10 µs
//!   on battery), then parks. A dispatcher that knows when its next frame is due takes a
//!   [`Prewaker`] and calls it at the predicted send time minus `sched.prewake_us`, so the task is
//!   polling, not parked, when the frame comes: the hand-off then costs a cache-line transfer, not
//!   a cold wake.
//!
//! On Windows the runtime cannot wait on a ring's event handle, so a parked task looks at its
//! rings every [`PARKED_POLL`] instead; it still polls continuously while it spins.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use syntropy_spsc::{Consumer, Producer, SPIN_DEFAULT};
use tokio::runtime::Handle;
use tokio::sync::Notify;

/// How often a parked channel looks at its rings where the runtime cannot wait on their wake
/// handles (Windows): 1 ms, the runtime timer's resolution.
pub const PARKED_POLL: Duration = Duration::from_millis(1);

/// The domain state channels and dispatchers share: atomics only, so a dispatcher never takes a
/// lock (ADR-0012 item 4). `notify` is used only on the domain's own thread.
#[derive(Debug)]
pub(crate) struct DomainState {
    /// When the domain started; times below are nanoseconds since then.
    epoch: Instant,
    live: AtomicBool,
    spin_ns: AtomicU64,
    /// A dispatcher's pre-wake keeps every channel polling until this time.
    prewake_until: AtomicU64,
    /// Wakes parked channel tasks when a pre-wake arrives.
    notify: Notify,
}

impl DomainState {
    fn new() -> DomainState {
        DomainState {
            epoch: Instant::now(),
            live: AtomicBool::new(false),
            spin_ns: AtomicU64::new(SPIN_DEFAULT.as_nanos() as u64),
            prewake_until: AtomicU64::new(0),
            notify: Notify::new(),
        }
    }

    fn since_epoch(&self, at: Instant) -> u64 {
        at.saturating_duration_since(self.epoch).as_nanos() as u64
    }

    /// Whether a channel whose last frame moved at `last_busy` keeps polling at `now` rather
    /// than park: within the spin budget of that frame while sequences are live, or within a
    /// pre-wake.
    pub(crate) fn spinning(&self, last_busy: Instant, now: Instant) -> bool {
        if self.since_epoch(now) < self.prewake_until.load(Ordering::Acquire) {
            return true;
        }
        self.live.load(Ordering::Acquire)
            && now.saturating_duration_since(last_busy).as_nanos() as u64
                <= self.spin_ns.load(Ordering::Relaxed)
    }

    /// Resolves when a pre-wake arrives, for a parked channel's task.
    pub(crate) async fn prewoken(&self) {
        self.notify.notified().await;
    }
}

/// The `net` domain's runtime and its polling policy. Dropping it stops the runtime: tasks still
/// running end, and their channels report the link as gone.
#[derive(Debug)]
pub struct NetDomain {
    handle: DomainHandle,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

/// A handle that runs tasks in a [`NetDomain`]. Cloning it is cheap.
#[derive(Clone, Debug)]
pub struct DomainHandle {
    runtime: Handle,
    pub(crate) state: Arc<DomainState>,
}

impl NetDomain {
    /// Starts the domain on a new thread named `name`, such as `syntropy-net`.
    ///
    /// # Errors
    ///
    /// The OS error if the thread or the runtime cannot be made.
    pub fn start(name: &str) -> io::Result<NetDomain> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let handle = DomainHandle {
            runtime: runtime.handle().clone(),
            state: Arc::new(DomainState::new()),
        };
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let thread = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                // The runtime runs until the domain is dropped; a dropped sender ends it too.
                let _ = runtime.block_on(stopped);
                runtime.shutdown_timeout(Duration::from_millis(100));
            })?;
        Ok(NetDomain {
            handle,
            stop: Some(stop),
            thread: Some(thread),
        })
    }

    /// The handle tasks and transports use.
    pub fn handle(&self) -> &DomainHandle {
        &self.handle
    }

    /// Whether the runner has a `Ready` instance with live sequences: while it does, channels
    /// keep polling for the spin budget after each frame (RFC-0031 §4.4).
    pub fn set_live(&self, live: bool) {
        self.handle.state.live.store(live, Ordering::Release);
    }

    /// The spin budget: [`SPIN_DEFAULT`] (50 µs), or [`syntropy_spsc::SPIN_ON_BATTERY`] (10 µs)
    /// on battery (ADR-0012 item 5).
    pub fn set_spin(&self, spin: Duration) {
        self.handle
            .state
            .spin_ns
            .store(spin.as_nanos() as u64, Ordering::Relaxed);
    }

    /// A pre-wake for one dispatcher. Each dispatcher takes its own, since its doorbell is a
    /// single-producer ring.
    ///
    /// # Errors
    ///
    /// The OS error if the doorbell's wake handle cannot be made.
    pub fn prewaker(&self) -> io::Result<Prewaker> {
        let (tx, rx) = doorbell()?;
        let state = Arc::clone(&self.handle.state);
        let waker_state = Arc::clone(&state);
        self.handle.runtime.spawn(ring_doorbell(rx, waker_state));
        Ok(Prewaker { tx, state })
    }
}

impl Drop for NetDomain {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl DomainHandle {
    /// Runs `task` on the domain's thread.
    pub fn spawn<F>(&self, task: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.runtime.spawn(task)
    }
}

/// One dispatcher's pre-wake (RFC-0031 §4.4): it asks the domain's channels to poll, not park,
/// for a while. Call it at the predicted send time minus `sched.prewake_us`.
pub struct Prewaker {
    tx: Producer<()>,
    state: Arc<DomainState>,
}

impl std::fmt::Debug for Prewaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prewaker").finish_non_exhaustive()
    }
}

impl Prewaker {
    /// Keeps every channel of the domain polling for `window`, waking any that are parked. A
    /// window that a later pre-wake already covers changes nothing. Never blocks; it makes a
    /// system call only to wake a parked domain.
    pub fn prewake(&mut self, window: Duration) {
        let state = &self.state;
        let until = state.since_epoch(Instant::now()) + window.as_nanos() as u64;
        let before = state.prewake_until.fetch_max(until, Ordering::AcqRel);
        if before < until {
            // A full doorbell already holds a ring for the domain to answer.
            let _ = self.tx.push(());
        }
    }
}

/// A doorbell ring whose consumer the domain can await.
fn doorbell() -> io::Result<(Producer<()>, Consumer<()>)> {
    wake::ring(2)
}

/// Answers a dispatcher's doorbell: wakes the domain's parked channels for each ring.
async fn ring_doorbell(mut rx: Consumer<()>, state: Arc<DomainState>) {
    let mut ready = wake::Ready::new(&rx);
    loop {
        let mut closed = false;
        loop {
            match rx.pop() {
                Ok(()) => {}
                Err(syntropy_spsc::PopError::Empty) => break,
                Err(syntropy_spsc::PopError::Closed) => {
                    closed = true;
                    break;
                }
            }
        }
        state.notify.notify_waiters();
        if closed {
            return;
        }
        ready.wait(&mut rx).await;
    }
}

/// Awaiting a ring from the domain: its wake handle where the runtime can wait on it, a short
/// timer otherwise.
pub(crate) mod wake {
    use std::io;

    use syntropy_spsc::{Consumer, Producer};

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    mod imp {
        use std::io;
        use std::os::fd::{AsFd, OwnedFd};

        use syntropy_spsc::{Arm, Consumer, Producer};
        use tokio::io::unix::AsyncFd;

        pub(crate) fn ring<T>(capacity: usize) -> io::Result<(Producer<T>, Consumer<T>)> {
            syntropy_spsc::ring_with_wake(capacity)
        }

        /// Waits until a ring may hold values: on its wake handle, through the runtime.
        pub(crate) struct Ready {
            /// A duplicate of the wake handle's descriptor, which the runtime watches: an eventfd,
            /// or a kqueue holding one user event, readable while signalled. `None` if it could
            /// not be made, and then the wait falls back to the timer.
            fd: Option<AsyncFd<OwnedFd>>,
        }

        impl Ready {
            /// Must run in the domain's runtime, which the descriptor registers with.
            pub(crate) fn new<T>(rx: &Consumer<T>) -> Ready {
                let fd = rx
                    .wake_handle()
                    .and_then(|handle| handle.as_fd().try_clone_to_owned().ok())
                    // SAFETY: the `OwnedFd` owns the duplicated descriptor and the `AsyncFd` owns
                    // the `OwnedFd`, so the descriptor stays open, and names the same file
                    // description, until the `AsyncFd` is dropped; `OwnedFd::as_raw_fd` always
                    // returns it.
                    .and_then(|owned| unsafe { AsyncFd::register(owned) }.ok());
                Ready { fd }
            }

            /// Returns when the producer pushed or closed since the consumer last popped, or at
            /// once if a value is ready. It may also return with nothing to pop.
            pub(crate) async fn wait<T>(&mut self, rx: &mut Consumer<T>) {
                if let Arm::Ready = rx.arm() {
                    return;
                }
                match (&self.fd, rx.wake_handle()) {
                    (Some(fd), Some(handle)) => {
                        if let Ok(mut guard) = fd.readable().await {
                            // Reset before popping, so a push during the pops stays signalled.
                            let _ = handle.reset();
                            guard.clear_ready();
                        }
                    }
                    _ => tokio::time::sleep(super::super::PARKED_POLL).await,
                }
            }
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    mod imp {
        use std::io;

        use syntropy_spsc::{Arm, Consumer, Producer};

        pub(crate) fn ring<T>(capacity: usize) -> io::Result<(Producer<T>, Consumer<T>)> {
            syntropy_spsc::ring(capacity)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))
        }

        /// Waits until a ring may hold values: by looking every
        /// [`PARKED_POLL`](super::super::PARKED_POLL), since the runtime cannot wait on the
        /// ring's event handle here.
        pub(crate) struct Ready;

        impl Ready {
            pub(crate) fn new<T>(_rx: &Consumer<T>) -> Ready {
                Ready
            }

            pub(crate) async fn wait<T>(&mut self, rx: &mut Consumer<T>) {
                if let Arm::Ready = rx.arm() {
                    return;
                }
                tokio::time::sleep(super::super::PARKED_POLL).await;
            }
        }
    }

    pub(crate) use imp::Ready;

    /// A ring of `capacity` slots whose consumer is a task in the domain.
    pub(crate) fn ring<T>(capacity: usize) -> io::Result<(Producer<T>, Consumer<T>)> {
        imp::ring(capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(
        miri,
        ignore = "starts a tokio runtime, whose I/O driver calls the OS (FFI)"
    )]
    fn a_channel_spins_while_live_within_the_budget_and_after_a_prewake() {
        let domain = NetDomain::start("syntropy-net-test").expect("a domain");
        let state = Arc::clone(&domain.handle().state);
        let now = Instant::now();
        // Not live: parks at once.
        assert!(!state.spinning(now, now));
        domain.set_live(true);
        assert!(state.spinning(now, now));
        assert!(state.spinning(now, now + Duration::from_micros(40)));
        assert!(!state.spinning(now, now + Duration::from_micros(60)));
        domain.set_spin(syntropy_spsc::SPIN_ON_BATTERY);
        assert!(!state.spinning(now, now + Duration::from_micros(20)));
        domain.set_live(false);
        let mut prewaker = domain.prewaker().expect("a prewaker");
        prewaker.prewake(Duration::from_secs(60));
        assert!(state.spinning(now, Instant::now()));
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "starts a tokio runtime, whose I/O driver calls the OS (FFI)"
    )]
    fn a_prewake_wakes_a_parked_task() {
        let domain = NetDomain::start("syntropy-net-test").expect("a domain");
        let state = Arc::clone(&domain.handle().state);
        let (woke_tx, woke_rx) = std::sync::mpsc::channel();
        let parked = Arc::clone(&state);
        domain.handle().spawn(async move {
            parked.prewoken().await;
            let _ = woke_tx.send(());
        });
        let mut prewaker = domain.prewaker().expect("a prewaker");
        // The parked task may not have registered yet: ring until it answers.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            prewaker.prewake(Duration::from_millis(1));
            // A new window each time, so each pre-wake rings the doorbell.
            std::thread::sleep(Duration::from_millis(2));
            if woke_rx.try_recv().is_ok() {
                break;
            }
            assert!(Instant::now() < deadline, "the parked task never woke");
        }
    }
}
