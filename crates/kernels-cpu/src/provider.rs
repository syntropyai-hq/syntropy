// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU provider's shell (RFC-0040 §7.3; COMP-E3-T02): the [`KernelProvider`] contract over
//! host memory, without kernels yet.
//!
//! - **Devices.** One CPU device per NUMA domain (RFC-0001 §5.1), from the cores this process
//!   may run on (`syntropy_io::os::cpu_topology`). Each reports the ISA level detection gives as
//!   its class (ADR-0405).
//! - **Memory.** `alloc` gives `HostMapped` host memory, zeroed and aligned to at least 64
//!   bytes. `register_host` and `import` take memory the tier manager owns as it is, since the
//!   CPU sees it directly. `Device` and `Unified` memory, and `reserve_only`, are not the CPU's
//!   (RFC-0040 §5.1, the CPU column).
//! - **Streams and events.** A stream is a work queue ([`crate::stream`]) and an event is an
//!   atomic. A call that enqueues returns at once, and the stream's runner does the work.
//! - **Doorbells.** `signal_host` is a release store of the value. `wait_host` is an acquire
//!   wait for it, then the consumer's reset. `arm_wake` wakes a [`CpuWaker`] the waiter
//!   registered, after the stores before it (RFC-0040 §4.2, the CPU row).
//! - **Capture.** It records the op list that a stream's calls would push, and `replay` pushes
//!   it again (RFC-0040 §4.4, the CPU row). A replay allocates nothing.
//! - **Kernels.** `prepare` looks the op's kernel up in the ISA level's dispatch table
//!   (COMP-E3-T01), and `launch` queues it. The table is empty until the kernel tasks fill it, so
//!   every op is `Unsupported` today, except stand-ins a test adds with
//!   [`CpuProvider::with_kernel`]. `OpArgs` is a placeholder until COMP-E1-T05, so a kernel runs
//!   with [`OpArgs::none`](syntropy_kernels::OpArgs::none).
//!
//! The calls the dispatcher makes inside a step allocate nothing, take no lock and make no
//! system call: `replay`, `launch`, `record`, `wait`, `poll`, `signal_host`, `wait_host`,
//! `copy`, `copy_batch` and `health` (RFC-0040 §2.4).

use std::alloc::Layout;
use std::collections::BTreeSet;
use std::sync::Arc;

use smallvec::SmallVec;
use syntropy_core::resource::{Backend, DeviceKind};
use syntropy_core::{Bytes, DeviceId, ErrorClass, HostRegion};
use syntropy_kernels::{
    AllocRequest, BufferSlice, CapturedGraph, DeviceBuffer, DeviceClass, DeviceCtx, DeviceFeatures,
    DeviceHealth, DeviceMemoryInfo, Error, ErrorCode, Event, HostFlag, HostWaker, KernelProvider,
    LayoutVersion, MemKind, MemoryDomain, OpArgs, OpCaps, OpKey, OpKind, OpRegistry, OpenOptions,
    PreparedOp, ProviderBuild, ProviderDevice, ProviderId, Registration, ShapeSig, Stream,
    StreamPriority, WaitCmp, WakeKind, unsupported,
};

use crate::device::{CpuDevice, HostBuf, internal};
use crate::dispatch::KernelFn;
use crate::doorbell::Doorbells;
use crate::isa::{self, KillSwitches};
use crate::stream::{Item, Run, StreamRunner, sticky};
use crate::wake::CpuWaker;

/// The capabilities every CPU device has, on top of its ISA's: op lists replay as a captured
/// graph, `wait_host` and `signal_host` are native, and host memory is registered and
/// imported as it is.
const SHELL_FEATURES: DeviceFeatures = DeviceFeatures::GRAPH_CAPTURE
    .union(DeviceFeatures::STREAM_MEM_OPS)
    .union(DeviceFeatures::HOST_REGISTER)
    .union(DeviceFeatures::HOST_IMPORT);

/// The smallest alignment `alloc` gives: a cache line, so no two buffers share one.
const MIN_ALIGN: u64 = 64;

/// The provider for CPUs.
#[derive(Debug)]
pub struct CpuProvider {
    kill: KillSwitches,
    /// Stand-in kernels added to every opened device's table, for tests.
    extra: Vec<(OpKind, KernelFn)>,
    registry: OpRegistry,
}

impl CpuProvider {
    /// The provider, with the `features.disable` switches `kill` (RFC-0040 §12).
    #[must_use]
    pub fn new(kill: KillSwitches) -> CpuProvider {
        CpuProvider {
            kill,
            extra: Vec::new(),
            registry: OpRegistry::new(),
        }
    }

    /// The same provider with `kernel` as the kernel for `kind` on every device it opens: a
    /// stand-in, for tests, until the kernel tasks fill the dispatch tables.
    #[must_use]
    pub fn with_kernel(mut self, kind: OpKind, kernel: KernelFn) -> CpuProvider {
        self.extra.push((kind, kernel));
        self
    }

    fn device<'a>(&self, ctx: &'a DeviceCtx) -> Result<&'a CpuDevice, Error> {
        ctx.downcast_ref::<CpuDevice>()
            .ok_or_else(|| internal("the context is not a CPU device's".to_owned()))
    }

    fn live<'a>(&self, ctx: &'a DeviceCtx) -> Result<&'a CpuDevice, Error> {
        let device = self.device(ctx)?;
        match device.health.sticky() {
            Some(code) => Err(sticky(code)),
            None => Ok(device),
        }
    }

    /// Takes stream `s`'s runner, for the thread that will execute it: a CPU pool thread
    /// (PLC-E7-T07). `None` if a thread already took it. Called between steps.
    ///
    /// # Errors
    ///
    /// The context is not a CPU device's.
    pub fn take_runner(&self, ctx: &DeviceCtx, s: &Stream) -> Result<Option<StreamRunner>, Error> {
        let device = self.device(ctx)?;
        let mut streams = device.streams.borrow_mut();
        Ok(usize::try_from(s.handle)
            .ok()
            .and_then(|i| streams.get_mut(i))
            .and_then(|stream| stream.runner.take()))
    }

    /// Runs stream `s`'s queued items on the calling thread, as [`StreamRunner::run`] does, while
    /// no thread has taken its runner. This is the inline path the simulator takes through
    /// `ComputePool::run_inline` (RFC-0060 §2.3). It never spins or parks.
    ///
    /// # Errors
    ///
    /// A kernel's error, a sticky device error, or a stream whose runner a thread has taken.
    pub fn run(&self, ctx: &DeviceCtx, s: &Stream) -> Result<Run, Error> {
        let device = self.device(ctx)?;
        // The runner is taken out while it runs and put back after: it touches only its own
        // ring, doorbells, events and health, never the device's tables.
        let mut runner = self
            .take_runner(ctx, s)?
            .ok_or_else(|| internal(format!("stream {} is run by another thread", s.handle)))?;
        let result = runner.run();
        device.streams.borrow_mut()[s.handle as usize].runner = Some(runner);
        result
    }

    /// How many freed allocations wait for the streams to drain before they are deallocated.
    ///
    /// # Errors
    ///
    /// The context is not a CPU device's.
    pub fn deferred_frees(&self, ctx: &DeviceCtx) -> Result<usize, Error> {
        Ok(self.device(ctx)?.deferred.borrow().len())
    }

    /// The NUMA node the device is.
    ///
    /// # Errors
    ///
    /// The context is not a CPU device's.
    pub fn numa_node(&self, ctx: &DeviceCtx) -> Result<u16, Error> {
        Ok(self.device(ctx)?.node)
    }

    /// The device's doorbell flags, for the host threads on the other side of each doorbell.
    ///
    /// # Errors
    ///
    /// The context is not a CPU device's.
    pub fn doorbells(&self, ctx: &DeviceCtx) -> Result<Doorbells, Error> {
        Ok(self.device(ctx)?.doorbells.clone())
    }

    /// Registers `waker` with the device, so `arm_wake` can wake it: the waiter's side of the
    /// CPU's device-triggered wake. Called at `Warming`; the handle is fixed for the epoch.
    ///
    /// # Errors
    ///
    /// The context is not a CPU device's.
    pub fn register_waker(
        &self,
        ctx: &DeviceCtx,
        waker: Arc<CpuWaker>,
    ) -> Result<HostWaker, Error> {
        let device = self.device(ctx)?;
        let mut wakers = device.wakers.borrow_mut();
        wakers.push(waker);
        Ok(HostWaker {
            kind: WakeKind::Futex,
            handle: (wakers.len() - 1) as u64,
        })
    }

    fn flag_slot(device: &CpuDevice, flag: HostFlag) -> Result<u32, Error> {
        device
            .doorbells
            .index(flag)
            .ok_or_else(|| internal(format!("no doorbell {flag:?} on this device")))
    }

    fn event_state(device: &CpuDevice, e: &Event) -> Result<Arc<crate::stream::EventState>, Error> {
        usize::try_from(e.handle)
            .ok()
            .and_then(|i| device.events.borrow().get(i).cloned())
            .ok_or_else(|| internal(format!("no event {} on this device", e.handle)))
    }

    /// The NUMA nodes of the cores this process may run on, in order.
    fn nodes() -> Result<Vec<u16>, Error> {
        let cores = syntropy_io::os::cpu_topology().map_err(|error| Error {
            class: ErrorClass::Unsupported,
            code: ErrorCode::OP_UNSUPPORTED,
            message: format!("the cores this process may run on: {error}"),
        })?;
        let nodes: BTreeSet<u16> = cores.iter().map(|core| core.numa_node).collect();
        Ok(nodes.into_iter().collect())
    }
}

impl KernelProvider for CpuProvider {
    fn id(&self) -> ProviderId {
        ProviderId("cpu")
    }

    fn backend(&self) -> Backend {
        Backend::Cpu
    }

    fn build(&self) -> ProviderBuild {
        ProviderBuild {
            version: 1,
            module: None,
            toolchain: SmallVec::new(),
            driver: None,
        }
    }

    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Host
    }

    /// One device per NUMA domain. An x86-64 CPU without `X86V2`'s SSE4.2 and POPCNT has no
    /// device: the provider has no kernels below that level.
    fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error> {
        let Some(detection) = isa::detect(&self.kill) else {
            return Ok(Vec::new());
        };
        Ok(Self::nodes()?
            .into_iter()
            .enumerate()
            .map(|(local, node)| ProviderDevice {
                local: u32::try_from(local).unwrap_or(u32::MAX),
                uuid: None,
                pci: None,
                kind: DeviceKind::Cpu,
                name: format!("cpu{node}"),
                class: DeviceClass::Cpu { isa: detection.isa },
                features: detection.features | SHELL_FEATURES,
                // The tier manager budgets RAM (RFC-0020 §5); the CPU provider reports none.
                memory: DeviceMemoryInfo {
                    total: Bytes(0),
                    free: Bytes(0),
                    budget: None,
                    host_visible_device_local: None,
                    max_alloc: Bytes(u64::MAX),
                    unified: false,
                },
                numa: Some(node),
            })
            .collect())
    }

    /// Opens device `dev.ordinal`. It runs detection again, as RFC-0040 §7.3 says, installs the
    /// level's dispatch table, allocates the doorbell pages and creates the planned streams.
    fn open(&self, dev: DeviceId, opts: &OpenOptions) -> Result<DeviceCtx, Error> {
        let nodes = Self::nodes()?;
        let node = *nodes.get(usize::from(dev.ordinal)).ok_or_else(|| Error {
            class: ErrorClass::Unsupported,
            code: ErrorCode::OP_UNSUPPORTED,
            message: format!("no CPU device {} on this host", dev.ordinal),
        })?;
        let detection = isa::detect(&self.kill).ok_or_else(|| Error {
            class: ErrorClass::Unsupported,
            code: ErrorCode::OP_UNSUPPORTED,
            message: "this CPU is below X86V2, the provider's lowest level".to_owned(),
        })?;
        let mut table = isa::table_for_class(detection.isa, &detection)?;
        for (kind, kernel) in &self.extra {
            table.insert(*kind, *kernel);
        }
        let device = CpuDevice::new(node, table, Doorbells::new(opts.doorbell_slots));
        let plan = opts.streams;
        for (count, prio) in [
            (plan.compute, StreamPriority::Compute),
            (plan.copy_in, StreamPriority::CopyIn),
            (plan.copy_out, StreamPriority::CopyOut),
            (plan.collective, StreamPriority::Collective),
        ] {
            for _ in 0..count {
                device.add_stream(prio)?;
            }
        }
        Ok(DeviceCtx::new(device))
    }

    fn ops(&self) -> &OpRegistry {
        &self.registry
    }

    /// `None` for every op: each kernel task declares its op's capabilities with its kernel.
    fn query(&self, _ctx: &DeviceCtx, _op: &OpKey, _shape: &ShapeSig) -> Option<OpCaps> {
        None
    }

    fn alloc(&self, ctx: &DeviceCtx, req: AllocRequest) -> Result<DeviceBuffer, Error> {
        let device = self.live(ctx)?;
        if req.kind != MemKind::HostMapped || req.reserve_only {
            return Err(unsupported());
        }
        let align = req.align.0.max(MIN_ALIGN).next_power_of_two();
        let layout = usize::try_from(req.bytes.0)
            .ok()
            .filter(|&bytes| bytes > 0)
            .zip(usize::try_from(align).ok())
            .and_then(|(bytes, align)| Layout::from_size_align(bytes, align).ok())
            .ok_or_else(|| {
                alloc_failed(format!("{} bytes at {align}-byte alignment", req.bytes.0))
            })?;
        // SAFETY: `layout` has a non-zero size, checked above.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(alloc_failed(format!(
                "{} bytes of host memory",
                req.bytes.0
            )));
        }
        let handle = device.handle();
        device.buffers.borrow_mut().insert(
            handle,
            HostBuf {
                addr: ptr as u64,
                bytes: req.bytes.0,
                owned: Some(layout),
            },
        );
        Ok(DeviceBuffer {
            addr: ptr as u64,
            bytes: req.bytes,
            kind: MemKind::HostMapped,
            handle,
        })
    }

    fn free(&self, ctx: &DeviceCtx, buf: DeviceBuffer) {
        let Ok(device) = self.device(ctx) else {
            return;
        };
        let removed = device.buffers.borrow_mut().remove(&buf.handle);
        if let Some(HostBuf {
            addr,
            owned: Some(layout),
            ..
        }) = removed
        {
            // A queued copy may still hold the address, so the memory goes only once every
            // stream has drained.
            device.release(addr, layout);
        } else {
            device.release_deferred();
        }
    }

    /// The CPU sees host memory directly, so a registration only records the region.
    fn register_host(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<Registration, Error> {
        let device = self.live(ctx)?;
        Ok(Registration {
            region,
            handle: device.handle(),
        })
    }

    /// Wraps the tier manager's memory as a buffer, without copying. The tier manager keeps it,
    /// so `free` forgets the buffer and leaves the memory.
    fn import(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<DeviceBuffer, Error> {
        let device = self.live(ctx)?;
        if region.addr == 0 || region.len.0 == 0 {
            return Err(alloc_failed("an empty or null region".to_owned()));
        }
        let handle = device.handle();
        device.buffers.borrow_mut().insert(
            handle,
            HostBuf {
                addr: region.addr,
                bytes: region.len.0,
                owned: None,
            },
        );
        Ok(DeviceBuffer {
            addr: region.addr,
            bytes: region.len,
            kind: MemKind::HostMapped,
            handle,
        })
    }

    fn repack(
        &self,
        _op: &OpKey,
        _src: &[u8],
        _dst: &mut [u8],
    ) -> Option<Result<LayoutVersion, Error>> {
        None
    }

    fn repacked_len(&self, _op: &OpKey, _src: Bytes) -> Option<Bytes> {
        None
    }

    /// A stream `open` created for `prio` that is not handed out yet, or a new one. Creating a
    /// stream allocates, so it happens between steps.
    fn stream(&self, ctx: &DeviceCtx, prio: StreamPriority) -> Result<Stream, Error> {
        let device = self.live(ctx)?;
        let planned = {
            let mut streams = device.streams.borrow_mut();
            streams
                .iter_mut()
                .enumerate()
                .find(|(_, s)| s.prio == prio && !s.handed_out)
                .map(|(i, s)| {
                    s.handed_out = true;
                    i
                })
        };
        let index = match planned {
            Some(index) => index,
            None => {
                let index = device.add_stream(prio)?;
                device.streams.borrow_mut()[index].handed_out = true;
                index
            }
        };
        Ok(Stream {
            handle: index as u64,
        })
    }

    fn event(&self, ctx: &DeviceCtx) -> Result<Event, Error> {
        let device = self.live(ctx)?;
        let mut events = device.events.borrow_mut();
        events.push(Arc::default());
        Ok(Event {
            handle: (events.len() - 1) as u64,
        })
    }

    fn record(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error> {
        let device = self.live(ctx)?;
        let event = Self::event_state(device, e)?;
        device.push(
            s.handle,
            Item::Record {
                event,
                generation: 0,
            },
        )
    }

    fn wait(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error> {
        let device = self.live(ctx)?;
        let event = Self::event_state(device, e)?;
        device.push(
            s.handle,
            Item::WaitEvent {
                event,
                generation: 0,
            },
        )
    }

    fn signal_host(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        flag: HostFlag,
        value: u32,
    ) -> Result<(), Error> {
        let device = self.live(ctx)?;
        let slot = Self::flag_slot(device, flag)?;
        device.push(s.handle, Item::Signal { slot, value })
    }

    fn wait_host(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        flag: HostFlag,
        value: u32,
        cmp: WaitCmp,
        reset: Option<u32>,
    ) -> Result<(), Error> {
        let device = self.live(ctx)?;
        let slot = Self::flag_slot(device, flag)?;
        device.push(
            s.handle,
            Item::WaitHost {
                slot,
                value,
                cmp,
                reset,
            },
        )
    }

    /// Queues the writer's wake of a [`CpuWaker`] registered with
    /// [`CpuProvider::register_waker`]. It runs after the items before it, normally the
    /// doorbell's release store.
    fn arm_wake(&self, ctx: &DeviceCtx, s: &Stream, waker: &HostWaker) -> Result<(), Error> {
        let device = self.live(ctx)?;
        if waker.kind != WakeKind::Futex {
            return Err(unsupported());
        }
        let target = usize::try_from(waker.handle)
            .ok()
            .and_then(|i| device.wakers.borrow().get(i).cloned())
            .ok_or_else(|| internal(format!("no waker {} on this device", waker.handle)))?;
        device.push(s.handle, Item::Wake(target))
    }

    fn poll(&self, ctx: &DeviceCtx, e: &Event) -> Result<bool, Error> {
        let device = self.live(ctx)?;
        device.release_deferred();
        Ok(Self::event_state(device, e)?.complete())
    }

    /// Drains `s` on the calling thread if its runner is here, spinning on any wait; otherwise
    /// spins until the thread that took the runner has run everything pushed. Allowed only
    /// outside `Ready`.
    fn synchronize(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error> {
        let device = self.device(ctx)?;
        let index = usize::try_from(s.handle)
            .map_err(|_| internal(format!("no stream {} on this device", s.handle)))?;
        if let Some(mut runner) = self.take_runner(ctx, s)? {
            let result = runner.drain(None).map(|_| ());
            device.streams.borrow_mut()[index].runner = Some(runner);
            device.release_deferred();
            return result;
        }
        while !device.drained(index) {
            if let Some(code) = device.health.sticky() {
                return Err(sticky(code));
            }
            std::hint::spin_loop();
        }
        device.release_deferred();
        Ok(())
    }

    fn copy(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        src: BufferSlice,
        dst: BufferSlice,
    ) -> Result<(), Error> {
        let device = self.live(ctx)?;
        if src.bytes != dst.bytes {
            return Err(internal(format!(
                "a copy of {} bytes into {} bytes",
                src.bytes.0, dst.bytes.0
            )));
        }
        // Checked now, and resolved to addresses when pushed onto the ring: at once here, or at
        // each replay of a capture, so a buffer freed since the capture is never written.
        device.range(src.buffer, src.offset, src.bytes)?;
        device.range(dst.buffer, dst.offset, dst.bytes)?;
        device.push(s.handle, Item::CopyRef { src, dst })
    }

    fn prepare(&self, ctx: &DeviceCtx, op: &OpKey, _shape: &ShapeSig) -> Result<PreparedOp, Error> {
        let device = self.live(ctx)?;
        let kernel = device.table.get(&op.op).ok_or_else(unsupported)?;
        let mut prepared = device.prepared.borrow_mut();
        prepared.push(kernel);
        Ok(PreparedOp::with_handle(
            Bytes(0),
            (prepared.len() - 1) as u64,
        ))
    }

    /// Queues the prepared kernel. `args` is the placeholder of COMP-E1-T05, so the kernel runs
    /// with [`OpArgs::none`].
    fn launch(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        op: &PreparedOp,
        _args: &OpArgs<'_>,
    ) -> Result<(), Error> {
        let device = self.live(ctx)?;
        let kernel = usize::try_from(op.handle)
            .ok()
            .and_then(|i| device.prepared.borrow().get(i).copied())
            .ok_or_else(|| internal(format!("no prepared op {} on this device", op.handle)))?;
        device.push(s.handle, Item::Launch(kernel))
    }

    /// Starts recording the items `s`'s calls would push, instead of pushing them.
    fn begin_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error> {
        let device = self.live(ctx)?;
        let mut streams = device.streams.borrow_mut();
        let stream = usize::try_from(s.handle)
            .ok()
            .and_then(|i| streams.get_mut(i))
            .ok_or_else(|| internal(format!("no stream {} on this device", s.handle)))?;
        if stream.capture.is_some() {
            return Err(internal(format!(
                "stream {} is already capturing",
                s.handle
            )));
        }
        stream.capture = Some(Vec::new());
        Ok(())
    }

    fn end_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<CapturedGraph, Error> {
        let device = self.live(ctx)?;
        let list = {
            let mut streams = device.streams.borrow_mut();
            usize::try_from(s.handle)
                .ok()
                .and_then(|i| streams.get_mut(i))
                .and_then(|stream| stream.capture.take())
                .ok_or_else(|| internal(format!("stream {} is not capturing", s.handle)))?
        };
        let mut graphs = device.graphs.borrow_mut();
        graphs.push(list);
        Ok(CapturedGraph {
            handle: (graphs.len() - 1) as u64,
        })
    }

    /// Pushes the captured op list onto `s` again. Each record and wait takes its generation
    /// now. No allocation: the items go onto the preallocated ring.
    fn replay(&self, ctx: &DeviceCtx, s: &Stream, g: &CapturedGraph) -> Result<(), Error> {
        let device = self.live(ctx)?;
        let graphs = device.graphs.borrow();
        let list = usize::try_from(g.handle)
            .ok()
            .and_then(|i| graphs.get(i))
            .ok_or_else(|| internal(format!("no captured graph {} on this device", g.handle)))?;
        for item in list {
            device.push(s.handle, item.clone())?;
        }
        Ok(())
    }

    fn health(&self, ctx: &DeviceCtx) -> DeviceHealth {
        match self.device(ctx).ok().and_then(|d| d.health.sticky()) {
            None => DeviceHealth::Ok,
            Some(code) => DeviceHealth::Sticky {
                code: ErrorCode(code),
            },
        }
    }
}

fn alloc_failed(what: String) -> Error {
    Error {
        class: ErrorClass::Capacity,
        code: ErrorCode::ALLOC_FAILED,
        message: format!("cannot allocate {what}"),
    }
}
