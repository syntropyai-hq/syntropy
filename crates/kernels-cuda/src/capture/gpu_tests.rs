// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capture on a real CUDA device (COMP-E4-T05's hardware legs). Each test runs with
//! `--ignored`, and on a host without a CUDA device it returns after saying so. The buffers are
//! small (under 1 MiB of device memory in all), so the tests fit beside other work on the GPU.
//!
//! The synthetic decode step is what the kernels of later tasks will sit between: the descriptor
//! copy from its pinned mirror, a wait for the host's flag `F` with the consumer reset, the
//! step's work (here a copy of the descriptor into a host-mapped output, since no kernel exists
//! yet), and the `DC` signal to the host (RFC-0040 §4.2, §4.4).

use std::ffi::{c_uint, c_void};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use cudarc::driver::sys::{CU_MEMHOSTREGISTER_PORTABLE, CUcontext, CUgraph, CUstream};
use syntropy_core::{Bytes, DeviceId, NodeId};
use syntropy_kernels::error::codes;
use syntropy_kernels::{
    AllocRequest, BufferSlice, CapturedGraph, DeviceBuffer, DeviceCtx, HostFlag, KernelProvider,
    MemKind, OpenOptions, PowerHint, Stream, StreamPlan, StreamPriority, WaitCmp,
};

use crate::driver::{PrimaryContext, Status};
use crate::{CudaDevice, CudaProvider};

/// Descriptor bytes per step.
const DESC: u64 = 64 << 10;
/// Steps replayed per test.
const STEPS: u32 = 200;

fn opts(capture: bool) -> OpenOptions {
    OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 8,
        capture,
        deterministic: false,
        power: PowerHint::Mains,
    }
}

fn dev0() -> DeviceId {
    DeviceId {
        node: NodeId([0; 32]),
        ordinal: 0,
    }
}

fn whole(b: &DeviceBuffer) -> BufferSlice {
    BufferSlice {
        buffer: b.handle,
        offset: Bytes(0),
        bytes: b.bytes,
    }
}

/// The provider, or `None` after saying why there is no CUDA device.
fn provider() -> Option<CudaProvider> {
    let p = CudaProvider::new();
    match p.enumerate() {
        Ok(d) if !d.is_empty() => {
            eprintln!(
                "device: {} ({:?}); build: {:?}",
                d[0].name,
                d[0].class,
                p.build()
            );
            Some(p)
        }
        other => {
            eprintln!("no CUDA device ({other:?}): skipped");
            None
        }
    }
}

/// One opened device with the step's buffers.
struct Rig<'p> {
    p: &'p CudaProvider,
    ctx: DeviceCtx,
    compute: Stream,
    mirror: DeviceBuffer,
    desc: DeviceBuffer,
    out: DeviceBuffer,
    bell: DeviceBuffer,
    f: HostFlag,
    dc: HostFlag,
}

impl<'p> Rig<'p> {
    fn new(p: &'p CudaProvider, capture: bool) -> Self {
        let ctx = p.open(dev0(), &opts(capture)).unwrap();
        let compute = p.stream(&ctx, StreamPriority::Compute).unwrap();
        let alloc = |kind, bytes| {
            p.alloc(
                &ctx,
                AllocRequest {
                    bytes: Bytes(bytes),
                    kind,
                    align: Bytes(4096),
                    reserve_only: false,
                },
            )
            .unwrap()
        };
        let mirror = alloc(MemKind::HostMapped, DESC);
        let desc = alloc(MemKind::Device, DESC);
        let out = alloc(MemKind::HostMapped, DESC);
        let bell = alloc(MemKind::HostMapped, 4096);
        let f = p.host_flag(&ctx, &bell, 0).unwrap();
        let dc = p.host_flag(&ctx, &bell, 64).unwrap();
        Self {
            p,
            ctx,
            compute,
            mirror,
            desc,
            out,
            bell,
            f,
            dc,
        }
    }

    fn host(&self, b: &DeviceBuffer) -> *mut u8 {
        self.p.resolve(&self.ctx, whole(b)).unwrap().host.unwrap() as *mut u8
    }

    fn flag(&self, f: HostFlag) -> &AtomicU32 {
        let off = (u64::from(f.page) - u64::from(self.f.page)) * 4096 + u64::from(f.slot) * 4;
        // SAFETY: the flag is a 4-byte-aligned slot of the doorbell page `bell`, live while the
        // rig is; the device writes it only through stream memory operations, and the host
        // touches it only through this atomic.
        unsafe { &*self.host(&self.bell).add(off as usize).cast::<AtomicU32>() }
    }

    /// The step's operations on `s`, in order.
    fn step(&self, s: &Stream) {
        let p = self.p;
        p.copy(&self.ctx, s, whole(&self.mirror), whole(&self.desc))
            .unwrap();
        p.wait_host(&self.ctx, s, self.f, 1, WaitCmp::Eq, Some(0))
            .unwrap();
        p.copy(&self.ctx, s, whole(&self.desc), whole(&self.out))
            .unwrap();
        p.signal_host(&self.ctx, s, self.dc, 1).unwrap();
    }

    /// Writes step `i`'s descriptor into the mirror.
    fn write_desc(&self, i: u32) {
        // SAFETY: the mirror is `DESC` bytes of host-mapped memory, live while the rig is; no
        // device work reads it now (the previous step's `DC` was seen).
        let m = unsafe { std::slice::from_raw_parts_mut(self.host(&self.mirror), DESC as usize) };
        for (j, b) in m.iter_mut().enumerate() {
            *b = (j as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(i)
                .to_le_bytes()[j % 4];
        }
    }

    /// Releases the step: sets `F`, runs `issue`, then spins until the device raises `DC`, and
    /// consumes it. Returns the output's bytes.
    fn run(&self, issue: impl FnOnce()) -> Vec<u8> {
        self.flag(self.f).store(1, Ordering::Release);
        issue();
        let t0 = Instant::now();
        while self.flag(self.dc).load(Ordering::Acquire) != 1 {
            assert!(t0.elapsed() < Duration::from_secs(10), "DC never raised");
            std::hint::spin_loop();
        }
        self.flag(self.dc).store(0, Ordering::Release);
        assert_eq!(self.flag(self.f).load(Ordering::Acquire), 0, "F not reset");
        // SAFETY: the step that wrote `out` is done (its `DC` was raised after the copy).
        unsafe { std::slice::from_raw_parts(self.host(&self.out), DESC as usize) }.to_vec()
    }

    fn capture(&self) -> CapturedGraph {
        self.p.begin_capture(&self.ctx, &self.compute).unwrap();
        self.step(&self.compute);
        self.p.end_capture(&self.ctx, &self.compute).unwrap()
    }

    fn submissions(&self) -> u64 {
        self.p.submissions(&self.ctx, &self.compute).unwrap()
    }

    fn context(&self) -> PrimaryContext {
        let dev = self
            .ctx
            .downcast_ref::<CudaDevice<PrimaryContext>>()
            .unwrap();
        dev._context.share().unwrap()
    }
}

impl Drop for Rig<'_> {
    fn drop(&mut self) {
        let _ = self.p.synchronize(&self.ctx, &self.compute);
    }
}

/// AC1 and AC4: a captured step is one compute-stream submission per decode step, with the step
/// run `STEPS` times and each step's output checked; the graph count and memory are reported.
#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn one_compute_submission_per_decode_step() {
    let Some(p) = provider() else { return };
    let r = Rig::new(&p, true);
    let copy = p.stream(&r.ctx, StreamPriority::CopyIn).unwrap();
    let g = r.capture();
    let stats = p.graph_stats(&r.ctx).unwrap();
    eprintln!("graph stats after capture: {stats:?}");
    assert_eq!(stats.instances, 1);
    let mut per_step = Vec::new();
    for i in 0..STEPS {
        r.write_desc(i);
        let before = r.submissions();
        let out = r.run(|| p.replay(&r.ctx, &r.compute, &g).unwrap());
        per_step.push(r.submissions() - before);
        // SAFETY: as in `run`; the mirror is not written until the next step.
        let want = unsafe { std::slice::from_raw_parts(r.host(&r.mirror), DESC as usize) };
        assert_eq!(out, want, "step {i}");
    }
    assert!(per_step.iter().all(|&n| n == 1), "{per_step:?}");
    let copies = p.submissions(&r.ctx, &copy).unwrap();
    eprintln!(
        "compute-stream submissions per step over {STEPS} steps: min {} max {}; copy stream: {copies}",
        per_step.iter().min().unwrap(),
        per_step.iter().max().unwrap()
    );
    let by_stream = p.device(&r.ctx).unwrap().streams.submissions_by_stream();
    eprintln!("submissions by stream (including the one upload): {by_stream:?}");
}

/// AC3: with the kill switch the same step runs as per-op launches, and its output equals the
/// captured graph's byte for byte on every step.
#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn the_kill_switch_gives_per_op_launches_with_identical_bytes() {
    let Some(p) = provider() else { return };
    let captured = Rig::new(&p, true);
    let g = captured.capture();
    let eager = Rig::new(&p, false);
    let e = p.begin_capture(&eager.ctx, &eager.compute).unwrap_err();
    assert_eq!(e.code, codes::CAPTURE_FAILED, "{e}");
    eprintln!("kill switch: begin_capture -> {e}");
    let mut eager_subs = Vec::new();
    for i in 0..STEPS {
        captured.write_desc(i);
        eager.write_desc(i);
        let a = captured.run(|| p.replay(&captured.ctx, &captured.compute, &g).unwrap());
        let before = eager.submissions();
        let b = eager.run(|| eager.step(&eager.compute));
        eager_subs.push(eager.submissions() - before);
        assert_eq!(a, b, "step {i}");
    }
    eprintln!(
        "{STEPS} steps byte-identical ({DESC} bytes each); per-op submissions per step: {:?}",
        eager_subs.first()
    );
    assert!(eager_subs.iter().all(|&n| n == 5));
    assert_eq!(p.graph_stats(&eager.ctx).unwrap().instances, 0);
}

/// The driver calls the registering thread makes, as plain function pointers (which are `Send`).
#[derive(Clone, Copy)]
struct Register {
    set_current: unsafe extern "C" fn(CUcontext) -> Status,
    register: unsafe extern "C" fn(*mut c_void, usize, c_uint) -> Status,
    unregister: unsafe extern "C" fn(*mut c_void) -> Status,
}

impl Register {
    fn new(ctx: &PrimaryContext) -> Self {
        // SAFETY: the signatures are the CUDA 13 driver API's (`cuda.h`).
        unsafe {
            Self {
                set_current: ctx.api().symbol("cuCtxSetCurrent").unwrap(),
                register: ctx.api().symbol("cuMemHostRegister_v2").unwrap(),
                unregister: ctx.api().symbol("cuMemHostUnregister").unwrap(),
            }
        }
    }

    /// On a new thread with `ctx` current: `cuMemHostRegister_v2` of 16 MiB of fresh host
    /// memory, then `cuMemHostUnregister`. Returns the two statuses.
    fn on_another_thread(self, ctx: CUcontext) -> (Status, Status) {
        let ctx = ctx as usize;
        std::thread::spawn(move || {
            let layout = std::alloc::Layout::from_size_align(16 << 20, 4096).unwrap();
            // SAFETY: a non-zero size.
            let mem = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!mem.is_null());
            // SAFETY: `ctx` is a primary context the test keeps retained; setting it current
            // changes only this thread. `mem` is `layout.size()` bytes this thread owns until
            // the dealloc below, after the unregister.
            let statuses = unsafe {
                assert_eq!((self.set_current)(ctx as CUcontext), 0);
                let reg = (self.register)(mem.cast(), layout.size(), CU_MEMHOSTREGISTER_PORTABLE);
                let unreg = if reg == 0 {
                    (self.unregister)(mem.cast())
                } else {
                    0
                };
                (self.set_current)(std::ptr::null_mut());
                (reg, unreg)
            };
            // SAFETY: allocated above with `layout`, freed once.
            unsafe { std::alloc::dealloc(mem, layout) };
            statuses
        })
        .join()
        .unwrap()
    }
}

/// AC2: another thread's `cuMemHostRegister` during a thread-local capture does not fail it:
/// capture, instantiation and replay succeed and the step's output is right. The control
/// captures in global mode, where the same registration is refused by the driver.
#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn a_registration_on_another_thread_does_not_fail_a_capture() {
    let Some(p) = provider() else { return };
    let r = Rig::new(&p, true);
    let ctx = r.context();
    let reg = Register::new(&ctx);

    p.begin_capture(&r.ctx, &r.compute).unwrap();
    p.copy(&r.ctx, &r.compute, whole(&r.mirror), whole(&r.desc))
        .unwrap();
    let (st_reg, st_unreg) = reg.on_another_thread(ctx.handle());
    eprintln!(
        "thread-local capture: other thread's register -> {st_reg}, unregister -> {st_unreg}"
    );
    assert_eq!((st_reg, st_unreg), (0, 0));
    p.wait_host(&r.ctx, &r.compute, r.f, 1, WaitCmp::Eq, Some(0))
        .unwrap();
    p.copy(&r.ctx, &r.compute, whole(&r.desc), whole(&r.out))
        .unwrap();
    p.signal_host(&r.ctx, &r.compute, r.dc, 1).unwrap();
    let g = p.end_capture(&r.ctx, &r.compute).unwrap();
    for i in 0..10 {
        r.write_desc(i);
        let out = r.run(|| p.replay(&r.ctx, &r.compute, &g).unwrap());
        // SAFETY: as in `run`.
        let want = unsafe { std::slice::from_raw_parts(r.host(&r.mirror), DESC as usize) };
        assert_eq!(out, want, "step {i}");
    }

    // The control: a global-mode capture of the same stream, begun directly.
    let raw = p
        .device(&r.ctx)
        .unwrap()
        .streams
        .raw_stream(&r.compute)
        .unwrap();
    // SAFETY: the CUDA 13 driver API's signatures.
    let (begin, end, destroy): (
        unsafe extern "C" fn(CUstream, c_uint) -> Status,
        unsafe extern "C" fn(CUstream, *mut CUgraph) -> Status,
        unsafe extern "C" fn(CUgraph) -> Status,
    ) = unsafe {
        (
            ctx.api().symbol("cuStreamBeginCapture_v2").unwrap(),
            ctx.api().symbol("cuStreamEndCapture").unwrap(),
            ctx.api().symbol("cuGraphDestroy").unwrap(),
        )
    };
    ctx.make_current().unwrap();
    let stream = std::ptr::with_exposed_provenance_mut(raw as usize);
    // SAFETY: `stream` is the device's live compute stream, idle (synchronized below by the
    // last step's `DC`), and not being captured. Mode 0 is `CU_STREAM_CAPTURE_MODE_GLOBAL`.
    assert_eq!(unsafe { begin(stream, 0) }, 0);
    let (g_reg, g_unreg) = reg.on_another_thread(ctx.handle());
    let mut graph: CUgraph = std::ptr::null_mut();
    // SAFETY: ends the capture this thread began; writes one handle to a live local.
    let st_end = unsafe { end(stream, &raw mut graph) };
    if !graph.is_null() {
        // SAFETY: the graph `end` returned, destroyed once.
        unsafe { destroy(graph) };
    }
    eprintln!(
        "control, global capture: other thread's register -> {g_reg}, unregister -> {g_unreg}; end capture -> {st_end}"
    );
    assert_ne!(
        g_reg, 0,
        "the global-mode control did not refuse the registration"
    );
}

/// RFC-0012 §7.1: a helper thread instantiates the graph with the device's context current; the
/// dispatcher adopts the executable and replays it.
#[test]
#[ignore = "needs the NVIDIA driver and a GPU"]
fn a_helper_thread_instantiates_a_graph_the_dispatcher_replays() {
    let Some(p) = provider() else { return };
    let r = Rig::new(&p, true);
    p.begin_capture(&r.ctx, &r.compute).unwrap();
    r.step(&r.compute);
    let pending = p.end_capture_pending(&r.ctx, &r.compute).unwrap();
    let inst = p.instantiator(&r.ctx).unwrap();
    let ready = std::thread::spawn(move || inst.instantiate(pending))
        .join()
        .unwrap()
        .unwrap();
    eprintln!("instantiated on a helper thread: {ready:?}");
    let g = p.adopt(&r.ctx, &r.compute, ready).unwrap();
    for i in 0..10 {
        r.write_desc(i);
        let before = r.submissions();
        let out = r.run(|| p.replay(&r.ctx, &r.compute, &g).unwrap());
        assert_eq!(r.submissions() - before, 1);
        // SAFETY: as in `run`.
        let want = unsafe { std::slice::from_raw_parts(r.host(&r.mirror), DESC as usize) };
        assert_eq!(out, want, "step {i}");
    }
    eprintln!("graph stats: {:?}", p.graph_stats(&r.ctx).unwrap());
}
