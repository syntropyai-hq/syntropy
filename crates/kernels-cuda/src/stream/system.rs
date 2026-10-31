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
use crate::driver::{Api, PrimaryContext, SUCCESS, Status, sym};

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
    /// Resolves the entry points and makes `ctx` current on the calling thread, the device's
    /// dispatcher thread, which every later call runs on (RFC-0040 §2.4).
    pub fn new(ctx: &PrimaryContext) -> Result<Self, StreamFault> {
        let api = Arc::clone(ctx.api());
        let lib = api.lib();
        let missing = |call: &'static str| StreamFault {
            call,
            status: 0,
            error: "the driver library lacks this entry point".into(),
        };
        let calls = (|| {
            Ok(Calls {
                set_current: sym(lib, "cuCtxSetCurrent")?,
                priority_range: sym(lib, "cuCtxGetStreamPriorityRange")?,
                stream_create: sym(lib, "cuStreamCreateWithPriority")?,
                stream_destroy: sym(lib, "cuStreamDestroy_v2")?,
                stream_wait: sym(lib, "cuStreamWaitEvent")?,
                stream_sync: sym(lib, "cuStreamSynchronize")?,
                event_create: sym(lib, "cuEventCreate")?,
                event_destroy: sym(lib, "cuEventDestroy_v2")?,
                event_record: sym(lib, "cuEventRecord")?,
                event_query: sym(lib, "cuEventQuery")?,
                event_elapsed: sym(lib, "cuEventElapsedTime_v2")
                    .or_else(|_| sym(lib, "cuEventElapsedTime"))?,
                memcpy: sym(lib, "cuMemcpyAsync")?,
            })
        })()
        .map_err(missing)?;
        let batch = sym::<Batch>(lib, "cuMemcpyBatchAsync_v2").ok();
        let this = Self {
            api: Arc::clone(&api),
            calls,
            ctx: ctx.handle(),
            batch,
        };
        this.make_current()?;
        Ok(this)
    }

    /// Makes the device's context current on the calling thread. Streams and events are created
    /// in the current context, so each creation makes sure it is this device's, even when the
    /// thread opened another device since.
    fn make_current(&self) -> Result<(), StreamFault> {
        // SAFETY: `ctx` is a primary context the device keeps retained while this lives; making
        // it current changes only this thread's current context.
        let s = unsafe { (self.calls.set_current)(self.ctx) };
        self.check("cuCtxSetCurrent", s)
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
                .error_name(status)
                .unwrap_or_else(|| format!("CUDA error {status}")),
        })
    }
}

impl StreamApi for SystemStreams {
    fn priority_range(&self) -> Result<(i32, i32), StreamFault> {
        self.make_current()?;
        let (mut least, mut greatest): (c_int, c_int) = (0, 0);
        // SAFETY: writes two integers through pointers to live locals; the context is current.
        let s = unsafe { (self.calls.priority_range)(&raw mut least, &raw mut greatest) };
        self.check("cuCtxGetStreamPriorityRange", s)
            .map(|()| (least, greatest))
    }

    fn create_stream(&self, priority: i32) -> Result<u64, StreamFault> {
        self.make_current()?;
        let mut s: CUstream = std::ptr::null_mut();
        let flags = CUstream_flags::CU_STREAM_NON_BLOCKING as c_uint;
        // SAFETY: writes one handle through a pointer to a live local; the context is current.
        let st = unsafe { (self.calls.stream_create)(&raw mut s, flags, priority) };
        self.check("cuStreamCreateWithPriority", st)
            .map(|()| to_u64(s))
    }

    fn create_event(&self, timing: bool) -> Result<u64, StreamFault> {
        self.make_current()?;
        let mut e: CUevent = std::ptr::null_mut();
        let flags = if timing {
            CUevent_flags::CU_EVENT_DEFAULT
        } else {
            CUevent_flags::CU_EVENT_DISABLE_TIMING
        } as c_uint;
        // SAFETY: writes one handle through a pointer to a live local; the context is current.
        let st = unsafe { (self.calls.event_create)(&raw mut e, flags) };
        self.check("cuEventCreate", st).map(|()| to_u64(e))
    }

    unsafe fn destroy_stream(&self, s: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: `s` is a live stream, not used again.
        let st = unsafe { (self.calls.stream_destroy)(from_u64::<CUstream_st>(s)) };
        self.check("cuStreamDestroy_v2", st)
    }

    unsafe fn destroy_event(&self, e: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: `e` is a live event, not used again.
        let st = unsafe { (self.calls.event_destroy)(from_u64::<CUevent_st>(e)) };
        self.check("cuEventDestroy_v2", st)
    }

    unsafe fn record(&self, e: u64, s: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: live handles of this context.
        let st = unsafe {
            (self.calls.event_record)(from_u64::<CUevent_st>(e), from_u64::<CUstream_st>(s))
        };
        self.check("cuEventRecord", st)
    }

    unsafe fn wait(&self, s: u64, e: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: live handles of this context. Flag 0 is
        // `CU_EVENT_WAIT_DEFAULT`.
        let st = unsafe {
            (self.calls.stream_wait)(from_u64::<CUstream_st>(s), from_u64::<CUevent_st>(e), 0)
        };
        self.check("cuStreamWaitEvent", st)
    }

    unsafe fn query(&self, e: u64) -> Result<bool, StreamFault> {
        // SAFETY: the caller's contract: a live event of this context.
        let st = unsafe { (self.calls.event_query)(from_u64::<CUevent_st>(e)) };
        if st == super::NOT_READY {
            return Ok(false);
        }
        self.check("cuEventQuery", st).map(|()| true)
    }

    unsafe fn elapsed_ms(&self, start: u64, end: u64) -> Result<f32, StreamFault> {
        let mut ms = 0f32;
        // SAFETY: the caller's contract: live events; writes one float to a live local.
        let st = unsafe {
            (self.calls.event_elapsed)(
                &raw mut ms,
                from_u64::<CUevent_st>(start),
                from_u64::<CUevent_st>(end),
            )
        };
        self.check("cuEventElapsedTime", st).map(|()| ms)
    }

    unsafe fn synchronize(&self, s: u64) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: a live stream of this context.
        let st = unsafe { (self.calls.stream_sync)(from_u64::<CUstream_st>(s)) };
        self.check("cuStreamSynchronize", st)
    }

    unsafe fn copy(&self, s: u64, dst: u64, src: u64, bytes: usize) -> Result<(), StreamFault> {
        // SAFETY: the caller's contract: a live stream, and addresses valid for `bytes` in this
        // context until the copy completes.
        let st = unsafe { (self.calls.memcpy)(dst, src, bytes, from_u64::<CUstream_st>(s)) };
        self.check("cuMemcpyAsync", st)
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
        let st = unsafe {
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
        };
        self.check("cuMemcpyBatchAsync_v2", st)
    }
}
