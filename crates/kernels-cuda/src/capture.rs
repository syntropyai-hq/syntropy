// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph capture and doorbell operations (RFC-0040 §4.2, §4.4; ADR-0404 items 1, 2, 7;
//! COMP-E4-T05).
//!
//! | Trait method | Driver calls |
//! |---|---|
//! | `begin_capture` | `cuStreamBeginCapture_v2` in `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` |
//! | `end_capture` | `cuStreamEndCapture`; `cuGraphInstantiateWithFlags`; `cuGraphDestroy`; `cuGraphUpload` on the captured stream |
//! | `replay` | `cuGraphLaunch`: one submission |
//! | `signal_host` | `cuStreamWriteValue32_v2` of the given value (1 under RFC-0040 §4.2's handshake) |
//! | `wait_host` | `cuStreamWaitValue32_v2` (`EQ` or `GEQ`), then `cuStreamWriteValue32_v2` of `reset` when one is given |
//!
//! **Thread-local mode** (ADR-0404 item 7). A capture in thread-local mode is invalidated only
//! by an unsafe call made on the capturing thread, so the housekeeping thread may register host
//! memory (RFC-0020 §6) while the dispatcher captures. Inside a capture the stream calls of
//! [`stream`](crate::stream) and the doorbell operations here become graph nodes: copies become
//! memcpy nodes, and `signal_host` and `wait_host` become memory-operation nodes whose values
//! are the constants they were given (RFC-0040 §4.2).
//!
//! **Instantiation on a helper thread** (RFC-0012 §7.1). `end_capture` instantiates on the
//! calling thread. A runner that instantiates elsewhere calls [`CudaProvider::end_capture_pending`](crate::CudaProvider::end_capture_pending)
//! instead, sends the [`PendingGraph`] to a helper thread holding an [`Instantiator`] (both are
//! `Send`: the instantiator holds its own retain of the primary context), and hands the
//! [`ReadyGraph`] back to the dispatcher, which [`adopt`](crate::CudaProvider::adopt)s it between
//! steps. [`syntropy_kernels::capture::GraphSet`] keeps the swap to step boundaries.
//!
//! **The kill switch.** A device opened with `OpenOptions::capture = false`, the
//! `graph_capture` entry of `features.disable` (RFC-0040 §12), refuses `begin_capture` with
//! `CaptureFailed`, `(Kernels, 7)`. The executor then issues the same operations directly, in
//! the same order on the same streams, so the output is the same bit for bit, and the runner
//! reports `Ready { degraded: CaptureOff }` ([`syntropy_kernels::capture::capture_degrade`]).
//! A capture or instantiation the driver refuses raises the same code, and takes the same path.
//!
//! **What is counted.** Every replay, and every doorbell operation outside a capture, counts as
//! a submission on its stream ([`DeviceStreams::submissions`]). [`GraphStats`] lists the graph
//! instances and the device memory each took: the drop in `cuMemGetInfo_v2`'s free bytes across
//! instantiation and upload. The reading is device-wide, so another process's allocation in that
//! window shows up in it; RFC-0040 §4.4 has graph memory measured at `Warming`, not predicted.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use syntropy_core::{ErrorClass, ErrorCode};
use syntropy_kernels::capture::capture_failed;
use syntropy_kernels::{CapturedGraph, Error, Stream, WaitCmp};

use crate::driver::{Driver, PrimaryContext, SystemDriver};
use crate::stream::{DeviceStreams, STICKY, StreamFault, stream_error};

mod system;

pub use system::SystemCapture;

/// A [`Driver`] whose open contexts can capture graphs and run stream memory operations.
pub trait CaptureDriver: Driver {
    /// The capture calls of the device whose context is `ctx`. The value keeps the context
    /// retained for as long as it lives, so a helper thread may hold it past the device.
    fn capture_api(&self, ctx: &Self::Context) -> Result<Arc<dyn CaptureApi>, StreamFault>;
}

impl CaptureDriver for SystemDriver {
    fn capture_api(&self, ctx: &PrimaryContext) -> Result<Arc<dyn CaptureApi>, StreamFault> {
        Ok(Arc::new(SystemCapture::new(ctx)?))
    }
}

/// The capture, graph and stream memory-operation calls of one device's context, over raw
/// driver handles. `Send` and `Sync`, so an [`Instantiator`] can call it from a helper thread.
///
/// The `unsafe` methods take handles, which must come from this device (streams from its
/// [`DeviceStreams`], graphs and executables from this object) and must not be destroyed yet.
pub trait CaptureApi: Send + Sync {
    /// `cuStreamBeginCapture_v2` on `s` in thread-local mode.
    ///
    /// # Safety
    ///
    /// `s` is a live stream of this device, not being captured.
    unsafe fn begin_capture(&self, s: u64) -> Result<(), StreamFault>;
    /// `cuStreamEndCapture` on `s`: the captured graph.
    ///
    /// # Safety
    ///
    /// `s` is a live stream of this device being captured, by the calling thread.
    unsafe fn end_capture(&self, s: u64) -> Result<u64, StreamFault>;
    /// `cuGraphInstantiateWithFlags` with no flags: the executable.
    ///
    /// # Safety
    ///
    /// `graph` is a live graph from [`end_capture`](Self::end_capture).
    unsafe fn instantiate(&self, graph: u64) -> Result<u64, StreamFault>;
    /// `cuGraphDestroy`.
    ///
    /// # Safety
    ///
    /// `graph` is a live graph; it is not used again.
    unsafe fn destroy_graph(&self, graph: u64) -> Result<(), StreamFault>;
    /// `cuGraphUpload` of `exec` on `s`.
    ///
    /// # Safety
    ///
    /// `exec` is a live executable and `s` a live stream of this device.
    unsafe fn upload(&self, exec: u64, s: u64) -> Result<(), StreamFault>;
    /// `cuGraphLaunch` of `exec` on `s`.
    ///
    /// # Safety
    ///
    /// As [`upload`](Self::upload); every buffer the graph references is still live.
    unsafe fn launch(&self, exec: u64, s: u64) -> Result<(), StreamFault>;
    /// `cuGraphExecDestroy`. An executable still running is freed when it completes.
    ///
    /// # Safety
    ///
    /// `exec` is a live executable; it is not used again.
    unsafe fn destroy_exec(&self, exec: u64) -> Result<(), StreamFault>;
    /// `cuMemGetInfo_v2`'s free bytes on the device.
    fn free_bytes(&self) -> Result<u64, StreamFault>;
    /// `cuStreamWriteValue32_v2`: writes `value` to the 4 bytes at device address `addr`
    /// when `s` reaches this point.
    ///
    /// # Safety
    ///
    /// `s` is a live stream; `addr` is a device address in this context of 4 writable,
    /// 4-byte-aligned bytes that stay valid until the write completes (every replay, when
    /// captured).
    unsafe fn write_value32(&self, s: u64, addr: u64, value: u32) -> Result<(), StreamFault>;
    /// `cuStreamWaitValue32_v2`: later work on `s` waits until the 4 bytes at `addr` compare
    /// `cmp` against `value`.
    ///
    /// # Safety
    ///
    /// As [`write_value32`](Self::write_value32), for reading.
    unsafe fn wait_value32(
        &self,
        s: u64,
        addr: u64,
        value: u32,
        cmp: WaitCmp,
    ) -> Result<(), StreamFault>;
}

/// The error for a failed capture or instantiation call (RFC-0040 §13): a sticky error is
/// `DeviceSticky`, as for every driver call; anything else is `CaptureFailed`, `(Kernels, 7)`,
/// class `Unsupported`, whose fallback is per-op launches.
#[must_use]
pub fn capture_error(f: &StreamFault) -> Error {
    if STICKY.contains(&f.status) {
        stream_error(f)
    } else {
        capture_failed(format!(
            "graph capture failed: CUDA driver call {} failed: {}",
            f.call, f.error
        ))
    }
}

/// A provider fault: class `Internal`, code `(Kernels, 0)`.
fn internal(message: String) -> Error {
    Error::new(
        ErrorClass::Internal,
        ErrorCode::new(syntropy_core::Domain::Kernels, 0),
        message,
    )
}

/// Tags each [`DeviceGraphs`], so a graph of another device is refused.
static NEXT_TAG: AtomicU32 = AtomicU32::new(1);

/// One live executable: the driver's handle and the device memory it took.
#[derive(Debug)]
struct Slot {
    exec: u64,
    bytes: u64,
}

/// The graph instances of a device and the device memory each took (RFC-0040 §4.4, "Graph
/// count and memory"). [`CudaProvider::graph_stats`](crate::CudaProvider::graph_stats) returns it; the executor reports it at
/// `Warming`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphStats {
    /// Live graph instances.
    pub instances: usize,
    /// The device memory each live instance took, in bytes, in the order they were made.
    pub per_instance: Vec<u64>,
    /// Their sum.
    pub device_bytes: u64,
}

/// A captured graph not yet instantiated: `Send`, so a helper thread can instantiate it. Dropped
/// without instantiation, it destroys the graph.
pub struct PendingGraph {
    graph: u64,
    api: Arc<dyn CaptureApi>,
    tag: u32,
}

impl std::fmt::Debug for PendingGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingGraph")
            .field("graph", &self.graph)
            .field("tag", &self.tag)
            .finish_non_exhaustive()
    }
}

impl Drop for PendingGraph {
    fn drop(&mut self) {
        // SAFETY: `graph` came from `end_capture` and is destroyed once, here.
        let _ = unsafe { self.api.destroy_graph(self.graph) };
    }
}

/// An instantiated graph not yet adopted by its device: `Send`, so it travels back from the
/// helper thread. Dropped without adoption, it destroys the executable.
pub struct ReadyGraph {
    exec: Option<u64>,
    bytes: u64,
    api: Arc<dyn CaptureApi>,
    tag: u32,
}

impl std::fmt::Debug for ReadyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadyGraph")
            .field("exec", &self.exec)
            .field("bytes", &self.bytes)
            .field("tag", &self.tag)
            .finish_non_exhaustive()
    }
}

impl ReadyGraph {
    /// The device memory instantiation took, in bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for ReadyGraph {
    fn drop(&mut self) {
        if let Some(exec) = self.exec.take() {
            // SAFETY: `exec` came from `instantiate`, was never adopted, and is destroyed once.
            let _ = unsafe { self.api.destroy_exec(exec) };
        }
    }
}

/// Instantiates a device's captured graphs on any thread (RFC-0040 §4.4, CUDA row; RFC-0012
/// §7.1). It holds its own retain of the device's primary context, and makes it current for
/// each call. Instantiation submits no work to the device.
#[derive(Clone)]
pub struct Instantiator {
    api: Arc<dyn CaptureApi>,
    tag: u32,
}

impl std::fmt::Debug for Instantiator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instantiator")
            .field("tag", &self.tag)
            .finish_non_exhaustive()
    }
}

impl Instantiator {
    /// Instantiates `g`, a graph of this instantiator's device, and destroys the graph. The
    /// result records how far the device's free memory fell.
    pub fn instantiate(&self, g: PendingGraph) -> Result<ReadyGraph, Error> {
        if g.tag != self.tag {
            return Err(internal("a graph captured on another device".into()));
        }
        let before = self.api.free_bytes().map_err(|f| stream_error(&f))?;
        // SAFETY: `g.graph` is a live graph of this device (same tag); `g` destroys it after.
        let exec = unsafe { self.api.instantiate(g.graph) }.map_err(|f| capture_error(&f))?;
        let mut ready = ReadyGraph {
            exec: Some(exec),
            bytes: 0,
            api: Arc::clone(&self.api),
            tag: self.tag,
        };
        // On an error `ready` drops, which destroys the executable.
        let after = self.api.free_bytes().map_err(|f| stream_error(&f))?;
        ready.bytes = before.saturating_sub(after);
        drop(g);
        Ok(ready)
    }
}

/// One device's graph executables and its capture switch. Lives in the device's context, on
/// its dispatcher thread; the [`Instantiator`] is the part that may leave it.
pub struct DeviceGraphs {
    api: Arc<dyn CaptureApi>,
    tag: u32,
    /// `OpenOptions::capture`: false is the `graph_capture` kill switch.
    enabled: bool,
    slots: RefCell<Vec<Option<Slot>>>,
}

impl std::fmt::Debug for DeviceGraphs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceGraphs")
            .field("tag", &self.tag)
            .field("enabled", &self.enabled)
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}

impl DeviceGraphs {
    /// The graphs of a device whose capture calls are `api`; `enabled` is
    /// `OpenOptions::capture`.
    #[must_use]
    pub fn new(api: Arc<dyn CaptureApi>, enabled: bool) -> Self {
        Self {
            api,
            tag: NEXT_TAG.fetch_add(1, Ordering::Relaxed),
            enabled,
            slots: RefCell::default(),
        }
    }

    /// Whether capture is on for the device (`OpenOptions::capture`).
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    fn handle(&self, index: usize) -> u64 {
        (u64::from(self.tag) << 32) | index as u64
    }

    fn exec(&self, g: &CapturedGraph) -> Result<u64, Error> {
        (g.handle >> 32 == u64::from(self.tag))
            .then_some((g.handle & 0xFFFF_FFFF) as usize)
            .and_then(|i| self.slots.borrow().get(i)?.as_ref().map(|s| s.exec))
            .ok_or_else(|| internal(format!("no graph {:#x} on this device", g.handle)))
    }

    /// Starts a thread-local capture of `s`.
    pub fn begin(&self, streams: &DeviceStreams, s: &Stream) -> Result<(), Error> {
        if !self.enabled {
            return Err(capture_failed(
                "graph capture is off for this device (`features.disable` holds `graph_capture`)",
            ));
        }
        if streams.capturing(s)? {
            return Err(internal(format!(
                "stream {:#x} is already being captured",
                s.handle
            )));
        }
        let raw = streams.raw_stream(s)?;
        // SAFETY: `raw` is a live stream of this device, not being captured (checked above).
        unsafe { self.api.begin_capture(raw) }.map_err(|f| capture_error(&f))?;
        streams.set_capturing(s, true)
    }

    /// Ends the capture of `s`, leaving the graph to instantiate. The stream leaves capture
    /// mode whether or not the driver gives a graph.
    pub fn end(&self, streams: &DeviceStreams, s: &Stream) -> Result<PendingGraph, Error> {
        if !streams.capturing(s)? {
            return Err(internal(format!(
                "stream {:#x} is not being captured",
                s.handle
            )));
        }
        let raw = streams.raw_stream(s)?;
        streams.set_capturing(s, false)?;
        // SAFETY: `raw` is a live stream this object started capturing; `end_capture` runs on
        // the device's dispatcher thread, which began it (the provider is not `Sync` per
        // device: `DeviceCtx` is used from one thread).
        let graph = unsafe { self.api.end_capture(raw) }.map_err(|f| capture_error(&f))?;
        Ok(PendingGraph {
            graph,
            api: Arc::clone(&self.api),
            tag: self.tag,
        })
    }

    /// An instantiator for this device's graphs, for a helper thread.
    #[must_use]
    pub fn instantiator(&self) -> Instantiator {
        Instantiator {
            api: Arc::clone(&self.api),
            tag: self.tag,
        }
    }

    /// Takes `g` into the device's table: uploads it on `s` (one submission, between steps) and
    /// returns its handle for `replay`.
    pub fn adopt(
        &self,
        streams: &DeviceStreams,
        s: &Stream,
        mut g: ReadyGraph,
    ) -> Result<CapturedGraph, Error> {
        if g.tag != self.tag {
            return Err(internal("a graph instantiated for another device".into()));
        }
        let raw = streams.raw_stream(s)?;
        let exec = g.exec.ok_or_else(|| internal("an adopted graph".into()))?;
        let before = self.api.free_bytes().map_err(|f| stream_error(&f))?;
        // SAFETY: `exec` is a live executable of this device (same tag); `raw` a live stream.
        unsafe { self.api.upload(exec, raw) }.map_err(|f| capture_error(&f))?;
        streams.submitted(s, 1);
        let after = self.api.free_bytes().map_err(|f| stream_error(&f))?;
        g.exec = None;
        let mut slots = self.slots.borrow_mut();
        slots.push(Some(Slot {
            exec,
            bytes: g.bytes + before.saturating_sub(after),
        }));
        Ok(CapturedGraph {
            handle: self.handle(slots.len() - 1),
        })
    }

    /// Enqueues `g` on `s`: one `cuGraphLaunch`, one submission.
    pub fn replay(
        &self,
        streams: &DeviceStreams,
        s: &Stream,
        g: &CapturedGraph,
    ) -> Result<(), Error> {
        let exec = self.exec(g)?;
        let raw = streams.raw_stream(s)?;
        // SAFETY: `exec` is live in this device's table and `raw` is a live stream; the graph
        // references only buffers fixed for the epoch (RFC-0040 §4.4 rule 1), which the tier
        // manager keeps until the graph is released.
        unsafe { self.api.launch(exec, raw) }.map_err(|f| stream_error(&f))?;
        streams.submitted(s, 1);
        Ok(())
    }

    /// Destroys `g`: after an invalidating change, or when a re-instantiated graph replaces it.
    /// A replay still running finishes first.
    pub fn release(&self, g: CapturedGraph) -> Result<(), Error> {
        self.exec(&g)?;
        let slot = self.slots.borrow_mut()[(g.handle & 0xFFFF_FFFF) as usize].take();
        if let Some(slot) = slot {
            // SAFETY: the executable was live in the table; its entry is gone, so it is
            // destroyed once and never launched again.
            unsafe { self.api.destroy_exec(slot.exec) }.map_err(|f| stream_error(&f))?;
        }
        Ok(())
    }

    /// The live instances and their device memory.
    #[must_use]
    pub fn stats(&self) -> GraphStats {
        let per_instance: Vec<u64> = self
            .slots
            .borrow()
            .iter()
            .flatten()
            .map(|s| s.bytes)
            .collect();
        GraphStats {
            instances: per_instance.len(),
            device_bytes: per_instance.iter().sum(),
            per_instance,
        }
    }

    /// `signal_host` at device address `addr`: one memory operation on `s`.
    pub fn signal(
        &self,
        streams: &DeviceStreams,
        s: &Stream,
        addr: u64,
        value: u32,
    ) -> Result<(), Error> {
        let raw = streams.raw_stream(s)?;
        // SAFETY: `raw` is a live stream; the caller resolved `addr` from a live doorbell page
        // of this device, which the tier manager keeps for the epoch.
        unsafe { self.api.write_value32(raw, addr, value) }.map_err(|f| stream_error(&f))?;
        streams.submitted(s, 1);
        Ok(())
    }

    /// `wait_host` at device address `addr`: the wait, then the consumer reset when `reset`
    /// is given (RFC-0040 §4.2).
    pub fn wait(
        &self,
        streams: &DeviceStreams,
        s: &Stream,
        addr: u64,
        value: u32,
        cmp: WaitCmp,
        reset: Option<u32>,
    ) -> Result<(), Error> {
        let raw = streams.raw_stream(s)?;
        // SAFETY: as in `signal`.
        unsafe { self.api.wait_value32(raw, addr, value, cmp) }.map_err(|f| stream_error(&f))?;
        streams.submitted(s, 1);
        if let Some(r) = reset {
            // SAFETY: as in `signal`.
            unsafe { self.api.write_value32(raw, addr, r) }.map_err(|f| stream_error(&f))?;
            streams.submitted(s, 1);
        }
        Ok(())
    }
}

impl Drop for DeviceGraphs {
    /// Destroys every live executable. [`CudaDevice`](crate::CudaDevice) declares its graphs
    /// after its streams, which drain first, and before its memory and its context.
    fn drop(&mut self) {
        for slot in std::mem::take(self.slots.get_mut()).into_iter().flatten() {
            // SAFETY: a live executable whose entry is gone; destroyed once.
            let _ = unsafe { self.api.destroy_exec(slot.exec) };
        }
    }
}

#[cfg(test)]
pub(crate) mod fake;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod gpu_tests;
