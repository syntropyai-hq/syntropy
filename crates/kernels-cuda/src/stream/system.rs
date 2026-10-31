// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`StreamApi`] over the installed driver: the entry points resolved from the library
//! [`SystemDriver`](crate::driver::SystemDriver) opened, in one device's primary context.

use std::ffi::{c_int, c_uint};
use std::sync::Arc;

use cudarc::driver::sys::{
    CUcontext, CUdeviceptr, CUevent, CUevent_flags, CUevent_st, CUmemcpyAttributes,
    CUmemcpySrcAccessOrder, CUstream, CUstream_flags, CUstream_st,
};

use super::{StreamApi, StreamFault};
use crate::driver::{Api, PrimaryContext, SUCCESS, Status};

type Batch = unsafe extern "C" fn(
    *mut CUdeviceptr,
    *mut CUdeviceptr,
    *mut usize,
    usize,
    *mut CUmemcpyAttributes,
    *mut usize,
    usize,
    CUstream,
) -> Status;

/// The entry points, with their CUDA 13 C signatures as `cudarc` declares them and the status
/// returned as `c_uint`.
#[derive(Clone, Copy)]
struct Calls {
    get_current: unsafe extern "C" fn(*mut CUcontext) -> Status,
    set_current: unsafe extern "C" fn(CUcontext) -> Status,
    priority_range: unsafe extern "C" fn(*mut c_int, *mut c_int) -> Status,
    stream_create: unsafe extern "C" fn(*mut CUstream, c_uint, c_int) -> Status,
    stream_destroy: unsafe extern "C" fn(CUstream) -> Status,
    stream_wait: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> Status,
    stream_sync: unsafe extern "C" fn(CUstream) -> Status,
    event_create: unsafe extern "C" fn(*mut CUevent, c_uint) -> Status,
    event_destroy: unsafe extern "C" fn(CUevent) -> Status,
    event_record: unsafe extern "C" fn(CUevent, CUstream) -> Status,
    event_query: unsafe extern "C" fn(CUevent) -> Status,
    event_elapsed: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> Status,
    memcpy: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> Status,
}

/// The stream calls of one device's primary context. The function pointers stay valid while
/// `api` holds the library open.
pub struct SystemStreams {
    api: Arc<Api>,
    calls: Calls,
    /// The device's primary context, retained by the device for as long as this lives.
    ctx: CUcontext,
    /// `cuMemcpyBatchAsync_v2`, CUDA 13's batched copy, where the driver exports it.
    batch: Option<Batch>,
}

/// A handle as the `u64` the tables keep. The handle is opaque: Rust never dereferences it.
fn to_u64<T>(p: *mut T) -> u64 {
    p.expose_provenance() as u64
}

/// The driver handle a `u64` from [`to_u64`] stands for.
fn from_u64<T>(h: u64) -> *mut T {
    std::ptr::with_exposed_provenance_mut(usize::try_from(h).unwrap_or(0))
}

impl SystemStreams {
    /// Resolves the entry points for the device whose primary context is `ctx`.
    pub fn new(ctx: &PrimaryContext) -> Result<Self, StreamFault> {
        let api = Arc::clone(ctx.api());
        let missing = |f: crate::driver::Fault| StreamFault {
            call: f.call,
            status: 0,
            error: f.error,
        };
        // SAFETY: each type is the entry point's C signature from the CUDA 13 driver API, as
        // `cudarc` declares it, with the status returned as `Status`.
        let calls = unsafe {
            Calls {
                get_current: api.symbol("cuCtxGetCurrent").map_err(missing)?,
                set_current: api.symbol("cuCtxSetCurrent").map_err(missing)?,
                priority_range: api.symbol("cuCtxGetStreamPriorityRange").map_err(missing)?,
                stream_create: api.symbol("cuStreamCreateWithPriority").map_err(missing)?,
                stream_destroy: api.symbol("cuStreamDestroy_v2").map_err(missing)?,
                stream_wait: api.symbol("cuStreamWaitEvent").map_err(missing)?,
                stream_sync: api.symbol("cuStreamSynchronize").map_err(missing)?,
                event_create: api.symbol("cuEventCreate").map_err(missing)?,
                event_destroy: api.symbol("cuEventDestroy_v2").map_err(missing)?,
                event_record: api.symbol("cuEventRecord").map_err(missing)?,
                event_query: api.symbol("cuEventQuery").map_err(missing)?,
                event_elapsed: api
                    .symbol("cuEventElapsedTime_v2")
                    .or_else(|_| api.symbol("cuEventElapsedTime"))
                    .map_err(missing)?,
                memcpy: api.symbol("cuMemcpyAsync").map_err(missing)?,
            }
        };
        // SAFETY: as above.
        let batch = unsafe { api.symbol::<Batch>("cuMemcpyBatchAsync_v2") }.ok();
        Ok(Self {
            api,
            calls,
            ctx: ctx.handle(),
            batch,
        })
    }

    /// Runs `f`, a driver call, with this device's context current on the calling thread, and
    /// then puts back the context that was current before. When this device's context is
    /// already current, as on its own dispatcher thread, that is one `cuCtxGetCurrent` and no
    /// switch. No call assumes the thread's current context: a thread that touched another
    /// device's context in between (a registration dropped there, a second device opened) gets
    /// its context back as it left it, and the call still runs in this device's.
    fn in_ctx(&self, f: impl FnOnce() -> Status) -> Result<Status, StreamFault> {
        let get = |previous: &mut CUcontext| {
            // SAFETY: writes one handle through a pointer to a live local.
            unsafe { (self.calls.get_current)(previous) }
        };
        let set = |ctx: CUcontext| {
            // SAFETY: `ctx` is this device's primary context, which the device keeps retained
            // while this lives, or the context that was current on this thread a moment ago, or
            // null, which leaves no context current. Only this thread's current context changes.
            unsafe { (self.calls.set_current)(ctx) }
        };
        switched(self.ctx, get, set, f).map_err(|(call, status)| {
            self.check(call, status)
                .expect_err("`switched` reports only failed statuses")
        })
    }

    /// `f`'s status, run as [`in_ctx`](Self::in_ctx) runs it, as a result for `call`.
    fn run(&self, call: &'static str, f: impl FnOnce() -> Status) -> Result<(), StreamFault> {
        let status = self.in_ctx(f)?;
        self.check(call, status)
    }

    fn check(&self, call: &'static str, status: Status) -> Result<(), StreamFault> {
        if status == SUCCESS {
            return Ok(());
        }
        Err(StreamFault {
            call,
            status,
            error: self
                .api
                .check(call, status)
                .err()
                .map_or_else(|| format!("CUDA error {status}"), |f| f.error),
        })
    }
}

/// [`SystemStreams::in_ctx`]'s context switch over the driver's `cuCtxGetCurrent` (`get`) and
/// `cuCtxSetCurrent` (`set`): runs `f` with `ctx` current and puts back the previous context.
/// Returns `f`'s status, or the failed context call and its status. A failed restore is
/// reported unless `f` itself really failed, in which case `f`'s fault is the one the caller
/// sees: a sticky error fails both, and `f` names what the caller asked for. `NOT_READY` is not
/// a failure: it is `cuEventQuery`'s answer for unfinished work, so a poll still reports a
/// failed restore.
pub(super) fn switched(
    ctx: CUcontext,
    get: impl FnOnce(&mut CUcontext) -> Status,
    mut set: impl FnMut(CUcontext) -> Status,
    f: impl FnOnce() -> Status,
) -> Result<Status, (&'static str, Status)> {
    let mut previous: CUcontext = std::ptr::null_mut();
    let s = get(&mut previous);
    if s != SUCCESS {
        return Err(("cuCtxGetCurrent", s));
    }
    if previous == ctx {
        return Ok(f());
    }
    let s = set(ctx);
    if s != SUCCESS {
        return Err(("cuCtxSetCurrent", s));
    }
    let status = f();
    let s = set(previous);
    if s != SUCCESS && (status == SUCCESS || status == super::NOT_READY) {
        return Err(("cuCtxSetCurrent", s));
    }
    Ok(status)
}

impl StreamApi for SystemStreams {
    fn priority_range(&self) -> Result<(i32, i32), StreamFault> {
        let (mut least, mut greatest): (c_int, c_int) = (0, 0);
        // SAFETY: writes two integers through pointers to live locals.
        self.run("cuCtxGetStreamPriorityRange", || unsafe {
            (self.calls.priority_range)(&raw mut least, &raw mut greatest)
        })
        .map(|()| (least, greatest))
    }

    fn create_stream(&self, priority: i32) -> Result<u64, StreamFault> {
        let mut s: CUstream = std::ptr::null_mut();
        let flags = CUstream_flags::CU_STREAM_NON_BLOCKING as c_uint;
        // SAFETY: writes one handle through a pointer to a live local.
        self.run("cuStreamCreateWithPriority", || unsafe {
            (self.calls.stream_create)(&raw mut s, flags, priority)
        })
        .map(|()| to_u64(s))
    }

    fn create_event(&self, timing: bool) -> Result<u64, StreamFault> {
        let mut e: CUevent = std::ptr::null_mut();
        let flags = if timing {
            CUevent_flags::CU_EVENT_DEFAULT
        } else {
            CUevent_flags::CU_EVENT_DISABLE_TIMING
        } as c_uint;
        // SAFETY: writes one handle through a pointer to a live local.
        self.run("cuEventCreate", || unsafe {
            (self.calls.event_create)(&raw mut e, flags)
        })
        .map(|()| to_u64(e))
    }

    unsafe fn destroy_stream(&self, s: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: `s` is a live stream, not used again.
        self.run("cuStreamDestroy_v2", || unsafe {
            (self.calls.stream_destroy)(from_u64::<CUstream_st>(s))
        })
    }

    unsafe fn destroy_event(&self, e: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: `e` is a live event, not used again.
        self.run("cuEventDestroy_v2", || unsafe {
            (self.calls.event_destroy)(from_u64::<CUevent_st>(e))
        })
    }

    unsafe fn record(&self, e: u64, s: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: live handles of this context.
        self.run("cuEventRecord", || unsafe {
            (self.calls.event_record)(from_u64::<CUevent_st>(e), from_u64::<CUstream_st>(s))
        })
    }

    unsafe fn wait(&self, s: u64, e: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: live handles of this context. Flag 0 is
        // `CU_EVENT_WAIT_DEFAULT`.
        self.run("cuStreamWaitEvent", || unsafe {
            (self.calls.stream_wait)(from_u64::<CUstream_st>(s), from_u64::<CUevent_st>(e), 0)
        })
    }

    unsafe fn query(&self, e: u64) -> Result<bool, StreamFault> {
        // SAFETY: the caller's contract: a live event of this context.
        let st = self.in_ctx(|| unsafe { (self.calls.event_query)(from_u64::<CUevent_st>(e)) })?;
        if st == super::NOT_READY {
            return Ok(false);
        }
        self.check("cuEventQuery", st).map(|()| true)
    }

    unsafe fn elapsed_ms(&self, start: u64, end: u64) -> Result<f32, StreamFault> {
        let mut ms = 0f32;
        // SAFETY: the caller's contract: live events; writes one float to a live local.
        self.run("cuEventElapsedTime", || unsafe {
            (self.calls.event_elapsed)(
                &raw mut ms,
                from_u64::<CUevent_st>(start),
                from_u64::<CUevent_st>(end),
            )
        })
        .map(|()| ms)
    }

    unsafe fn synchronize(&self, s: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: a live stream of this context.
        self.run("cuStreamSynchronize", || unsafe {
            (self.calls.stream_sync)(from_u64::<CUstream_st>(s))
        })
    }

    unsafe fn copy(&self, s: u64, dst: u64, src: u64, bytes: usize) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: a live stream, and addresses valid for `bytes` in this
        // context until the copy completes.
        self.run("cuMemcpyAsync", || unsafe {
            (self.calls.memcpy)(dst, src, bytes, from_u64::<CUstream_st>(s))
        })
    }

    fn batched(&self) -> bool {
        self.batch.is_some()
    }

    unsafe fn copy_batch(
        &self,
        s: u64,
        dsts: &[u64],
        srcs: &[u64],
        sizes: &[usize],
    ) -> Result<(), StreamFault> {
        let Some(batch) = self.batch else {
            return Err(StreamFault {
                call: "cuMemcpyBatchAsync_v2",
                status: 0,
                error: "the driver library lacks this entry point".into(),
            });
        };
        debug_assert!(dsts.len() == srcs.len() && srcs.len() == sizes.len());
        // SAFETY: an all-zero `CUmemcpyAttributes` is a valid value: every field is an integer,
        // an enum whose zero is a declared variant (`..._INVALID`), or a union of integers.
        let mut attr: CUmemcpyAttributes = unsafe { std::mem::zeroed() };
        // Each copy reads its source in stream order, after the work enqueued before it, as
        // `cuMemcpyAsync` does.
        attr.srcAccessOrder = CUmemcpySrcAccessOrder::CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
        let mut first: usize = 0;
        // SAFETY: the caller's contract: a live stream and valid addresses for every pair. The
        // three arrays hold `dsts.len()` entries, which the driver reads and does not write; one
        // attribute applies from index 0 to every copy.
        self.run("cuMemcpyBatchAsync_v2", || unsafe {
            batch(
                dsts.as_ptr().cast_mut(),
                srcs.as_ptr().cast_mut(),
                sizes.as_ptr().cast_mut(),
                dsts.len(),
                &raw mut attr,
                &raw mut first,
                1,
                from_u64::<CUstream_st>(s),
            )
        })
    }
}
