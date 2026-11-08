// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`CaptureApi`] over the installed driver, in one device's primary context.

use std::ffi::{c_uint, c_ulonglong};

use cudarc::driver::sys::{
    CUcontext, CUdeviceptr, CUgraph, CUgraph_st, CUgraphExec, CUgraphExec_st, CUstream,
    CUstream_st, CUstreamCaptureMode, CUstreamWaitValue_flags,
};
use syntropy_kernels::WaitCmp;

use super::CaptureApi;
use crate::driver::{PrimaryContext, SUCCESS, Status};
use crate::stream::StreamFault;
use crate::stream::system::switched;

/// The entry points, with their CUDA 13 C signatures as `cudarc` declares them, enums passed as
/// `c_uint` and the status returned as `c_uint`.
#[derive(Clone, Copy)]
struct Calls {
    get_current: unsafe extern "C" fn(*mut CUcontext) -> Status,
    set_current: unsafe extern "C" fn(CUcontext) -> Status,
    begin: unsafe extern "C" fn(CUstream, c_uint) -> Status,
    end: unsafe extern "C" fn(CUstream, *mut CUgraph) -> Status,
    instantiate: unsafe extern "C" fn(*mut CUgraphExec, CUgraph, c_ulonglong) -> Status,
    graph_destroy: unsafe extern "C" fn(CUgraph) -> Status,
    upload: unsafe extern "C" fn(CUgraphExec, CUstream) -> Status,
    launch: unsafe extern "C" fn(CUgraphExec, CUstream) -> Status,
    exec_destroy: unsafe extern "C" fn(CUgraphExec) -> Status,
    mem_info: unsafe extern "C" fn(*mut usize, *mut usize) -> Status,
    write32: unsafe extern "C" fn(CUstream, CUdeviceptr, u32, c_uint) -> Status,
    wait32: unsafe extern "C" fn(CUstream, CUdeviceptr, u32, c_uint) -> Status,
}

/// The capture calls of one device's primary context, which it keeps retained.
pub struct SystemCapture {
    calls: Calls,
    /// A retain of the device's primary context of this object's own, so an [`Instantiator`]
    /// on a helper thread keeps the context and the library alive.
    ///
    /// [`Instantiator`]: super::Instantiator
    ctx: PrimaryContext,
}

// SAFETY: a CUDA context handle and the driver's entry points may be used from any thread: the
// driver API is thread-safe, and every call here makes the context current on the calling thread
// for its duration and puts back the previous one (`run`). The retain is released once, on drop.
unsafe impl Send for SystemCapture {}
// SAFETY: as for `Send`; no method takes `&mut self`, and the object holds no Rust-side state
// that a call changes.
unsafe impl Sync for SystemCapture {}

/// A handle as the `u64` the tables keep. Rust never dereferences it.
fn to_u64<T>(p: *mut T) -> u64 {
    p.expose_provenance() as u64
}

/// The driver handle a `u64` from [`to_u64`] stands for.
fn from_u64<T>(h: u64) -> *mut T {
    std::ptr::with_exposed_provenance_mut(usize::try_from(h).unwrap_or(0))
}

impl SystemCapture {
    /// Resolves the entry points and retains `ctx` once more.
    pub fn new(ctx: &PrimaryContext) -> Result<Self, StreamFault> {
        let fault = |f: crate::driver::Fault| StreamFault {
            call: f.call,
            status: 0,
            error: f.error,
        };
        let api = ctx.api();
        // SAFETY: each type is the entry point's C signature from the CUDA 13 driver API, as
        // `cudarc` declares it, with enum arguments as `c_uint` (C enums of non-negative values
        // with `unsigned int`'s ABI) and the status as `Status`.
        let calls = unsafe {
            Calls {
                get_current: api.symbol("cuCtxGetCurrent").map_err(fault)?,
                set_current: api.symbol("cuCtxSetCurrent").map_err(fault)?,
                begin: api
                    .symbol("cuStreamBeginCapture_v2")
                    .or_else(|_| api.symbol("cuStreamBeginCapture"))
                    .map_err(fault)?,
                end: api.symbol("cuStreamEndCapture").map_err(fault)?,
                instantiate: api.symbol("cuGraphInstantiateWithFlags").map_err(fault)?,
                graph_destroy: api.symbol("cuGraphDestroy").map_err(fault)?,
                upload: api.symbol("cuGraphUpload").map_err(fault)?,
                launch: api.symbol("cuGraphLaunch").map_err(fault)?,
                exec_destroy: api.symbol("cuGraphExecDestroy").map_err(fault)?,
                mem_info: api.symbol("cuMemGetInfo_v2").map_err(fault)?,
                write32: api
                    .symbol("cuStreamWriteValue32_v2")
                    .or_else(|_| api.symbol("cuStreamWriteValue32"))
                    .map_err(fault)?,
                wait32: api
                    .symbol("cuStreamWaitValue32_v2")
                    .or_else(|_| api.symbol("cuStreamWaitValue32"))
                    .map_err(fault)?,
            }
        };
        Ok(Self {
            calls,
            ctx: ctx.share().map_err(fault)?,
        })
    }

    /// Runs `f`, a driver call named `call`, with this device's context current, as
    /// [`SystemStreams`](crate::stream::SystemStreams) does.
    fn run(&self, call: &'static str, f: impl FnOnce() -> Status) -> Result<(), StreamFault> {
        let get = |previous: &mut CUcontext| {
            // SAFETY: writes one handle through a pointer to a live local.
            unsafe { (self.calls.get_current)(previous) }
        };
        let set = |ctx: CUcontext| {
            // SAFETY: `ctx` is this device's primary context, retained by `self.ctx`, or the
            // context that was current on this thread a moment ago, or null. Only this thread's
            // current context changes.
            unsafe { (self.calls.set_current)(ctx) }
        };
        let status = switched(self.ctx.handle(), get, set, f)
            .map_err(|(call, status)| self.fault(call, status))?;
        if status == SUCCESS {
            Ok(())
        } else {
            Err(self.fault(call, status))
        }
    }

    fn fault(&self, call: &'static str, status: Status) -> StreamFault {
        StreamFault {
            call,
            status,
            error: self
                .ctx
                .api()
                .check(call, status)
                .err()
                .map_or_else(|| format!("CUDA error {status}"), |f| f.error),
        }
    }
}

impl CaptureApi for SystemCapture {
    unsafe fn begin_capture(&self, s: u64) -> Result<(), StreamFault> {
        let mode = CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL as c_uint;
        // SAFETY: the caller's contract: a live stream of this context, not being captured.
        self.run("cuStreamBeginCapture_v2", || unsafe {
            (self.calls.begin)(from_u64::<CUstream_st>(s), mode)
        })
    }

    unsafe fn end_capture(&self, s: u64) -> Result<u64, StreamFault> {
        let mut g: CUgraph = std::ptr::null_mut();
        // SAFETY: the caller's contract: a live stream this thread is capturing; writes one
        // handle through a pointer to a live local.
        self.run("cuStreamEndCapture", || unsafe {
            (self.calls.end)(from_u64::<CUstream_st>(s), &raw mut g)
        })
        .map(|()| to_u64(g))
    }

    unsafe fn instantiate(&self, graph: u64) -> Result<u64, StreamFault> {
        let mut exec: CUgraphExec = std::ptr::null_mut();
        // SAFETY: the caller's contract: a live graph; writes one handle to a live local.
        self.run("cuGraphInstantiateWithFlags", || unsafe {
            (self.calls.instantiate)(&raw mut exec, from_u64::<CUgraph_st>(graph), 0)
        })
        .map(|()| to_u64(exec))
    }

    unsafe fn destroy_graph(&self, graph: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: a live graph, not used again.
        self.run("cuGraphDestroy", || unsafe {
            (self.calls.graph_destroy)(from_u64::<CUgraph_st>(graph))
        })
    }

    unsafe fn upload(&self, exec: u64, s: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: a live executable and stream of this context.
        self.run("cuGraphUpload", || unsafe {
            (self.calls.upload)(from_u64::<CUgraphExec_st>(exec), from_u64::<CUstream_st>(s))
        })
    }

    unsafe fn launch(&self, exec: u64, s: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: a live executable and stream of this context, and
        // every buffer the graph references live.
        self.run("cuGraphLaunch", || unsafe {
            (self.calls.launch)(from_u64::<CUgraphExec_st>(exec), from_u64::<CUstream_st>(s))
        })
    }

    unsafe fn destroy_exec(&self, exec: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: a live executable, not used again.
        self.run("cuGraphExecDestroy", || unsafe {
            (self.calls.exec_destroy)(from_u64::<CUgraphExec_st>(exec))
        })
    }

    fn free_bytes(&self) -> Result<u64, StreamFault> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: writes two sizes through pointers to live locals.
        self.run("cuMemGetInfo_v2", || unsafe {
            (self.calls.mem_info)(&raw mut free, &raw mut total)
        })
        .map(|()| free as u64)
    }

    unsafe fn write_value32(&self, s: u64, addr: u64, value: u32) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: a live stream, and 4 writable aligned bytes at `addr`
        // in this context. Flag 0 is `CU_STREAM_WRITE_VALUE_DEFAULT`, which orders the write
        // after the work before it.
        self.run("cuStreamWriteValue32_v2", || unsafe {
            (self.calls.write32)(from_u64::<CUstream_st>(s), addr, value, 0)
        })
    }

    unsafe fn wait_value32(
        &self,
        s: u64,
        addr: u64,
        value: u32,
        cmp: WaitCmp,
    ) -> Result<(), StreamFault> {
        let flags = match cmp {
            WaitCmp::Eq => CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_EQ,
            WaitCmp::Geq => CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_GEQ,
        } as c_uint;
        // SAFETY: the caller's contract: a live stream, and 4 readable aligned bytes at `addr`
        // in this context.
        self.run("cuStreamWaitValue32_v2", || unsafe {
            (self.calls.wait32)(from_u64::<CUstream_st>(s), addr, value, flags)
        })
    }
}
