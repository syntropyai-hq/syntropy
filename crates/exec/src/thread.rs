// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dispatcher threads: one per GPU or NPU device and one per CPU device, each pinned to a core
//! on its device's NUMA node and run at the highest priority the OS grants without elevated
//! privileges (RFC-0012 §1.1; ADR-0012 item 2; PLC-E7-T01).
//!
//! - [`Cores`] hands out the cores a runner gives its hot threads: a core on the device's NUMA
//!   node when one is free, else any free core, so each dispatcher owns one core (ADR-0012
//!   item 3).
//! - [`spawn_dispatcher`] starts the thread, pins it, raises its priority, and returns a
//!   [`DispatcherThread`] whose [`HotThreadReport`] says where it runs: the record the runner's
//!   debug state lists (RFC-0050 §13.6, note of 2026-10-23).
//! - [`DispatcherRings`] are the thread's rings to the runner's `net` domain and to the I/O
//!   engine, all `syntropy-spsc` rings, with no lock and no tokio (ADR-0012 items 4 and 6).
//! - The thread's body is normally a [`Dispatcher`](crate::Dispatcher)'s production loop,
//!   [`Dispatcher::run`](crate::Dispatcher::run). It spins and parks as RFC-0012 §3.3 says,
//!   between steps on every inbound ring (PLC-E7-T17). [`DispatcherCtx::idle_wait`] is the
//!   simpler wait, for a body with one inbound ring: spin for the spin budget, then park on the
//!   ring until a frame arrives (ADR-0012 item 5).
//! - The runner sets the spin budget from `sched.spin_us` (50 µs, or 10 µs on battery). `sched`
//!   declares that key; `exec` sits below `sched` (RFC-0001 §4.1) and takes the value.
//!
//! The OS calls are `syntropy_io::os`'s, because `exec` holds no `unsafe` (RFC-0001 §4.2
//! rule 8). On macOS and iOS a thread cannot be bound to a core, so a dispatcher runs unpinned at
//! QoS `USER_INTERACTIVE` and its report says so (ADR-0012, revision of 2026-10-23).

use std::io;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde::Serialize;
use syntropy_core::DeviceId;
use syntropy_io::IoClient;
use syntropy_io::os::{self, Core, CoreId, PinError, Priority};
use syntropy_spsc::{Consumer, Producer};

/// What a dispatcher drives (RFC-0012 §1.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatcherKind {
    /// A GPU or NPU device.
    Device,
    /// A CPU device: one NUMA domain's cores.
    Cpu,
}

/// The cores a runner gives its hot threads, each handed out once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cores {
    free: Vec<Core>,
}

impl Cores {
    /// The cores of `topology`.
    #[must_use]
    pub fn new(mut topology: Vec<Core>) -> Cores {
        topology.sort();
        topology.dedup_by_key(|core| core.id);
        Cores { free: topology }
    }

    /// The cores this process may run on ([`os::cpu_topology`]). On Linux that is the main
    /// thread's mask, so detect before pinning the main thread, if it is ever pinned.
    ///
    /// # Errors
    ///
    /// If the OS cannot report the process's affinity mask.
    pub fn detect() -> io::Result<Cores> {
        os::cpu_topology().map(Cores::new)
    }

    /// The same cores without `reserved`, such as the `net` domain's and the I/O threads' cores
    /// (ADR-0012 items 2 and 3).
    #[must_use]
    pub fn without(mut self, reserved: &[CoreId]) -> Cores {
        self.free.retain(|core| !reserved.contains(&core.id));
        self
    }

    /// A free core on `numa`, the device's NUMA node, or, when the device has no node or none of
    /// its cores is free, any free core: the lowest-numbered either way. `None` when every core
    /// is taken.
    pub fn take(&mut self, numa: Option<u16>) -> Option<Core> {
        let local = numa.and_then(|node| self.free.iter().position(|c| c.numa_node == node));
        let at = local.or(if self.free.is_empty() { None } else { Some(0) })?;
        Some(self.free.remove(at))
    }

    /// How many cores are free.
    #[must_use]
    pub fn len(&self) -> usize {
        self.free.len()
    }

    /// Whether every core is taken.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.free.is_empty()
    }
}

/// The scheduling priority a hot thread got, as its report shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadPriority {
    /// A Linux or Android nice value.
    Nice(i32),
    /// A Windows thread priority level.
    Level(i32),
    /// The Apple QoS class `USER_INTERACTIVE`.
    UserInteractive,
    /// The OS kept the priority the thread started with.
    Unchanged,
}

impl From<Priority> for ThreadPriority {
    fn from(priority: Priority) -> ThreadPriority {
        match priority {
            Priority::Nice(nice) => ThreadPriority::Nice(nice),
            Priority::Level(level) => ThreadPriority::Level(level),
            Priority::UserInteractive => ThreadPriority::UserInteractive,
            Priority::Unchanged => ThreadPriority::Unchanged,
        }
    }
}

/// Where a dispatcher thread runs, and how the OS schedules it: the runner's debug state lists
/// one per hot thread (RFC-0050 §13.6, note of 2026-10-23).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HotThreadReport {
    /// What the thread drives.
    pub kind: DispatcherKind,
    /// Its device.
    pub device: DeviceId,
    /// The core it is pinned to, or `None` when it is not pinned.
    pub core: Option<u32>,
    /// The NUMA node it runs on: its core's, or node 0 where the OS cannot pin and has one node,
    /// as on Apple platforms. `None` when it was given no core, or pinning failed.
    pub numa_node: Option<u16>,
    /// Whether that node is the device's own, when both are known.
    pub numa_local: Option<bool>,
    /// The priority it got.
    pub priority: ThreadPriority,
    /// Why it is not pinned, when it is not.
    pub unpinned: Option<String>,
    /// Its spin budget before it parks, in nanoseconds (`sched.spin_us`).
    pub spin_budget_ns: u64,
}

/// What to start: the device, its NUMA node, the core [`Cores::take`] gave it, and its spin
/// budget.
#[derive(Clone, Debug)]
pub struct DispatcherSpec {
    /// What the thread drives.
    pub kind: DispatcherKind,
    /// Its device.
    pub device: DeviceId,
    /// The device's NUMA node, if it has one.
    pub numa: Option<u16>,
    /// The core to pin the thread to, or `None` to leave it unpinned.
    pub core: Option<Core>,
    /// How long it spins before it parks: `sched.spin_us`, 50 µs or 10 µs on battery
    /// ([`syntropy_spsc::SPIN_DEFAULT`], [`syntropy_spsc::SPIN_ON_BATTERY`]).
    pub spin: Duration,
}

/// A dispatcher's rings: frames from and to the `net` domain, and its client of the I/O engine
/// (RFC-0001 §8.3; RFC-0021 §3.1).
pub struct DispatcherRings<In, Out> {
    /// Frames the `net` domain receives for this stage.
    pub from_net: Consumer<In>,
    /// Frames this stage sends.
    pub to_net: Producer<Out>,
    /// The thread's rings to an I/O thread, if its device reads from NVMe tiers.
    pub io: Option<IoClient>,
}

/// What a dispatcher thread's body gets: its rings, its spin budget and its report.
pub struct DispatcherCtx<In, Out> {
    /// The thread's rings.
    pub rings: DispatcherRings<In, Out>,
    spin: Duration,
    report: HotThreadReport,
}

impl<In, Out> DispatcherCtx<In, Out> {
    /// Waits between steps: spins for the spin budget, then parks until the `net` domain pushes a
    /// frame or closes the ring (RFC-0012 §3.3). Returns at once if a frame is waiting.
    pub fn idle_wait(&mut self) {
        self.rings.from_net.wait(self.spin);
    }

    /// The spin budget.
    #[must_use]
    pub fn spin(&self) -> Duration {
        self.spin
    }

    /// Where the thread runs.
    #[must_use]
    pub fn report(&self) -> &HotThreadReport {
        &self.report
    }
}

/// A running dispatcher thread.
#[derive(Debug)]
pub struct DispatcherThread<R> {
    report: HotThreadReport,
    handle: JoinHandle<R>,
}

impl<R> DispatcherThread<R> {
    /// Where the thread runs, as it reported once it had pinned itself.
    #[must_use]
    pub fn report(&self) -> &HotThreadReport {
        &self.report
    }

    /// Waits for the thread's body to return.
    ///
    /// # Errors
    ///
    /// The body's panic, if it panicked.
    pub fn join(self) -> thread::Result<R> {
        self.handle.join()
    }
}

/// Starts a dispatcher thread for `spec`: pins it to `spec.core`, raises its priority, and runs
/// `body` with its rings, normally a [`Dispatcher`](crate::Dispatcher)'s production loop.
/// Returns once the thread has reported where it runs.
///
/// # Errors
///
/// If the OS cannot start the thread, or the thread ends before it reports.
pub fn spawn_dispatcher<In, Out, R>(
    spec: DispatcherSpec,
    rings: DispatcherRings<In, Out>,
    body: impl FnOnce(DispatcherCtx<In, Out>) -> R + Send + 'static,
) -> io::Result<DispatcherThread<R>>
where
    In: Send + 'static,
    Out: Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = mpsc::sync_channel(1);
    let name = format!("dispatcher-{}", spec.device.ordinal);
    let handle = thread::Builder::new().name(name).spawn(move || {
        // `node`: the NUMA node the thread runs on, when known. A pinned thread's is its core's.
        // Where the OS cannot pin, as on Apple platforms, the given core's node still holds,
        // because the topology there has one node (ADR-0012, revision of 2026-10-23).
        let (core, node, unpinned) = match spec.core {
            Some(core) => match os::pin_current_thread(core.id) {
                Ok(()) => (Some(core), Some(core.numa_node), None),
                Err(error @ PinError::Unsupported) => {
                    (None, Some(core.numa_node), Some(error.to_string()))
                }
                Err(error) => (None, None, Some(error.to_string())),
            },
            None => (None, None, Some("no free core was given to it".to_owned())),
        };
        let report = HotThreadReport {
            kind: spec.kind,
            device: spec.device,
            core: core.map(|c| c.id.0),
            numa_node: node,
            numa_local: node.zip(spec.numa).map(|(on, wanted)| on == wanted),
            priority: os::raise_current_thread_priority().into(),
            unpinned,
            spin_budget_ns: u64::try_from(spec.spin.as_nanos()).unwrap_or(u64::MAX),
        };
        // The receiver waits for this one message; if it has gone, the body still runs.
        let _ = tx.send(report.clone());
        body(DispatcherCtx {
            rings,
            spin: spec.spin,
            report,
        })
    })?;
    let report = rx.recv().map_err(|_| {
        io::Error::other("the dispatcher thread ended before it reported where it runs")
    })?;
    Ok(DispatcherThread { report, handle })
}

#[cfg(test)]
mod tests;
