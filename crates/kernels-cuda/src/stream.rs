// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streams, events and copies (RFC-0040 §4.1, §4.3, §7.7; COMP-E4-T04).
//!
//! `open` creates the streams of [`OpenOptions::streams`](syntropy_kernels::OpenOptions), each
//! with the CUDA priority [`stream_priority`] gives its [`StreamPriority`]: compute highest
//! (RFC-0040 §4.1). Every stream is non-blocking, so it never waits on the legacy default stream
//! that code using the CUDA runtime may submit to.
//!
//! [`DeviceStreams`] owns the device's streams and events. A [`Stream`] or [`Event`] the trait
//! hands out is an index into its tables, tagged with the table's identity, so a handle from
//! another device, or a made-up one, is an error and never reaches the driver. The trait has no
//! method that destroys a stream or an event: they live until the device is closed. Then
//! [`DeviceStreams`] drains every stream, destroys the events and then the streams, and only
//! after that does the device release its primary context (field order of
//! [`CudaDevice`](crate::CudaDevice)).
//!
//! The steady-state calls, `record`, `wait`, `poll`, `copy` and `copy_batch`, borrow the tables
//! and make one driver call each (one per [`BATCH_CHUNK`] pairs for `copy_batch`): no allocation,
//! no lock and no synchronising call (RFC-0040 §2.4). Events from `event` are created with
//! timing disabled; [`DeviceStreams::event`] with `timing` makes the events `elapsed` reads,
//! which only microbenchmarks and sampled telemetry use (RFC-0040 §4.3).
//!
//! `copy_batch` is `cuMemcpyBatchAsync` where the driver has CUDA 13's entry point, and a loop of
//! `cuMemcpyAsync` otherwise (RFC-0040 §7.7; ADR-0014).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

use syntropy_core::{ErrorClass, ErrorCode, Nanos};
use syntropy_kernels::error::codes;
use syntropy_kernels::{Error, Event, Stream, StreamPlan, StreamPriority};

use crate::driver::{Driver, PrimaryContext, SystemDriver};

mod system;

pub use system::SystemStreams;

/// A [`Driver`] whose open contexts have streams and events.
pub trait StreamDriver: Driver {
    /// The stream calls of `ctx`, made current on the calling thread.
    fn stream_api(&self, ctx: &Self::Context) -> Result<Box<dyn StreamApi>, StreamFault>;
}

impl StreamDriver for SystemDriver {
    fn stream_api(&self, ctx: &PrimaryContext) -> Result<Box<dyn StreamApi>, StreamFault> {
        Ok(Box::new(SystemStreams::new(ctx)?))
    }
}

/// A failed stream, event or copy call: the entry point, the driver's status and its name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamFault {
    /// The driver entry point.
    pub call: &'static str,
    /// The status the driver returned, as a plain integer.
    pub status: u32,
    /// `cuGetErrorName`'s text, or the raw code.
    pub error: String,
}

/// `CUDA_ERROR_NOT_READY`: `cuEventQuery`'s answer for an event whose work is still running.
pub const NOT_READY: u32 = 600;

/// `CUDA_ERROR_OUT_OF_MEMORY`.
pub const OUT_OF_MEMORY: u32 = 2;

/// The driver's sticky errors: after one, the context is unusable and every later call fails
/// (RFC-0040 §7.1, health). `ILLEGAL_ADDRESS`, `LAUNCH_TIMEOUT`, `ASSERT`,
/// `HARDWARE_STACK_ERROR`, `ILLEGAL_INSTRUCTION`, `MISALIGNED_ADDRESS`, `INVALID_ADDRESS_SPACE`,
/// `INVALID_PC`, `LAUNCH_FAILED` and `ECC_UNCORRECTABLE`.
pub const STICKY: [u32; 10] = [700, 702, 710, 714, 715, 716, 717, 718, 719, 214];

/// The most pairs one `cuMemcpyBatchAsync` call takes. The address and size arrays live on the
/// stack, so a batch never allocates; a longer batch is several calls on the same stream.
pub const BATCH_CHUNK: usize = 128;

/// The error for a failed driver call (RFC-0040 §13): a sticky error is `DeviceSticky`,
/// `(Kernels, 8)`, class `Device`, so the runner exits and restarts; out of memory is
/// `AllocFailed`, `(Kernels, 9)`, class `Capacity`; anything else is a provider fault, class
/// `Internal`.
#[must_use]
pub fn stream_error(f: &StreamFault) -> Error {
    let message = format!("CUDA driver call {} failed: {}", f.call, f.error);
    if STICKY.contains(&f.status) {
        Error::new(ErrorClass::Device, codes::DEVICE_STICKY, message)
    } else if f.status == OUT_OF_MEMORY {
        Error::new(ErrorClass::Capacity, codes::ALLOC_FAILED, message)
    } else {
        internal(message)
    }
}

/// A provider fault: class `Internal`, code `(Kernels, 0)`, as the CPU provider raises for a
/// handle it did not issue.
fn internal(message: String) -> Error {
    Error::new(
        ErrorClass::Internal,
        ErrorCode::new(syntropy_core::Domain::Kernels, 0),
        message,
    )
}

/// The CUDA stream priority for `prio`, given the context's range from
/// `cuCtxGetStreamPriorityRange`: `least` and `greatest`, where a numerically lower value is a
/// higher priority (`greatest <= least`).
///
/// `Compute` takes `greatest`, the highest (RFC-0040 §4.1). Then come `Collective`, whose
/// all-reduce sits inside the step, `CopyIn`, which brings the step's copied misses, and
/// `CopyOut`, whose spills and hand-offs can lag, one level apart while the range has levels
/// left. `Low` takes `least`. A device with fewer levels gives the lower classes the same
/// level, never one above `Compute`.
#[must_use]
pub fn stream_priority(prio: StreamPriority, least: i32, greatest: i32) -> i32 {
    let (least, greatest) = (least.max(greatest), least.min(greatest));
    let rank = match prio {
        StreamPriority::Compute => 0,
        StreamPriority::Collective => 1,
        StreamPriority::CopyIn => 2,
        StreamPriority::CopyOut => 3,
        _ => return least,
    };
    greatest.saturating_add(rank).min(least)
}

/// The stream, event and copy calls of one device's context, over raw driver handles.
///
/// Only [`DeviceStreams`] calls it. The `unsafe` methods take handles, which must come from this
/// object's `create_*` and must not be destroyed yet; addresses must be valid for the copy's
/// length in the device's context.
pub trait StreamApi {
    /// `cuCtxGetStreamPriorityRange`: `(least, greatest)`.
    fn priority_range(&self) -> Result<(i32, i32), StreamFault>;
    /// `cuStreamCreateWithPriority` with `CU_STREAM_NON_BLOCKING`.
    fn create_stream(&self, priority: i32) -> Result<u64, StreamFault>;
    /// `cuEventCreate`, with `CU_EVENT_DISABLE_TIMING` unless `timing`.
    fn create_event(&self, timing: bool) -> Result<u64, StreamFault>;
    /// `cuStreamDestroy_v2`.
    ///
    /// # Safety
    ///
    /// `s` is a live stream of this object; it is not used again.
    unsafe fn destroy_stream(&self, s: u64) -> Result<(), StreamFault>;
    /// `cuEventDestroy_v2`.
    ///
    /// # Safety
    ///
    /// `e` is a live event of this object; it is not used again.
    unsafe fn destroy_event(&self, e: u64) -> Result<(), StreamFault>;
    /// `cuEventRecord`.
    ///
    /// # Safety
    ///
    /// `e` and `s` are live handles of this object.
    unsafe fn record(&self, e: u64, s: u64) -> Result<(), StreamFault>;
    /// `cuStreamWaitEvent`.
    ///
    /// # Safety
    ///
    /// `s` and `e` are live handles of this object.
    unsafe fn wait(&self, s: u64, e: u64) -> Result<(), StreamFault>;
    /// `cuEventQuery`: `true` when the event's work is done, `false` for `NOT_READY`.
    ///
    /// # Safety
    ///
    /// `e` is a live event of this object.
    unsafe fn query(&self, e: u64) -> Result<bool, StreamFault>;
    /// `cuEventElapsedTime`, in milliseconds.
    ///
    /// # Safety
    ///
    /// `start` and `end` are live events of this object.
    unsafe fn elapsed_ms(&self, start: u64, end: u64) -> Result<f32, StreamFault>;
    /// `cuStreamSynchronize`.
    ///
    /// # Safety
    ///
    /// `s` is a live stream of this object.
    unsafe fn synchronize(&self, s: u64) -> Result<(), StreamFault>;
    /// `cuMemcpyAsync` of `bytes` from `src` to `dst` on `s`.
    ///
    /// # Safety
    ///
    /// `s` is a live stream of this object, and `src` and `dst` are addresses in the device's
    /// context (device or registered host memory) valid for `bytes` until the copy completes.
    unsafe fn copy(&self, s: u64, dst: u64, src: u64, bytes: usize) -> Result<(), StreamFault>;
    /// Whether [`copy_batch`](Self::copy_batch) is one driver call (`cuMemcpyBatchAsync`).
    fn batched(&self) -> bool;
    /// `cuMemcpyBatchAsync` of the pairs on `s`, in stream order. At most [`BATCH_CHUNK`] pairs,
    /// none of zero length, and the three slices of equal length. Only called when
    /// [`batched`](Self::batched) holds.
    ///
    /// # Safety
    ///
    /// As [`copy`](Self::copy), for every pair.
    unsafe fn copy_batch(
        &self,
        s: u64,
        dsts: &[u64],
        srcs: &[u64],
        sizes: &[usize],
    ) -> Result<(), StreamFault>;
}

/// A stream of the device: the driver's handle and the priority it was made for.
#[derive(Debug)]
struct StreamSlot {
    raw: u64,
    prio: StreamPriority,
    handed_out: bool,
}

/// An event of the device: the driver's handle and whether it records timestamps.
#[derive(Debug)]
struct EventSlot {
    raw: u64,
    timing: bool,
}

/// Tags the tables of each [`DeviceStreams`], so a handle from another device is refused.
static NEXT_TAG: AtomicU32 = AtomicU32::new(1);

/// One device's streams and events (RFC-0040 §4.1, §4.3). Lives in the device's context, on its
/// dispatcher thread only.
pub struct DeviceStreams {
    api: Box<dyn StreamApi>,
    tag: u32,
    /// `(least, greatest)` from `cuCtxGetStreamPriorityRange`.
    range: (i32, i32),
    streams: RefCell<Vec<StreamSlot>>,
    events: RefCell<Vec<EventSlot>>,
}

impl std::fmt::Debug for DeviceStreams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceStreams")
            .field("tag", &self.tag)
            .field("range", &self.range)
            .field("streams", &self.streams)
            .field("events", &self.events)
            .finish_non_exhaustive()
    }
}

/// The priorities a [`StreamPlan`] names, one entry per stream, in creation order.
#[must_use]
pub fn planned(plan: &StreamPlan) -> Vec<StreamPriority> {
    [
        (StreamPriority::Compute, plan.compute),
        (StreamPriority::CopyIn, plan.copy_in),
        (StreamPriority::CopyOut, plan.copy_out),
        (StreamPriority::Collective, plan.collective),
    ]
    .into_iter()
    .flat_map(|(p, n)| std::iter::repeat_n(p, usize::from(n)))
    .collect()
}

impl DeviceStreams {
    /// Creates the plan's streams over `api`. If one fails, those already made are destroyed.
    pub fn open(api: Box<dyn StreamApi>, plan: &StreamPlan) -> Result<Self, Error> {
        let range = api.priority_range().map_err(|f| stream_error(&f))?;
        let this = Self {
            api,
            tag: NEXT_TAG.fetch_add(1, Ordering::Relaxed),
            range,
            streams: RefCell::default(),
            events: RefCell::default(),
        };
        for prio in planned(plan) {
            // On an error `this` drops, which destroys the streams made so far.
            this.add_stream(prio)?;
        }
        Ok(this)
    }

    /// The handle for entry `index` of a table.
    fn handle(&self, index: usize) -> u64 {
        (u64::from(self.tag) << 32) | index as u64
    }

    /// The table index `handle` names, if it is one of this device's handles.
    fn index(&self, handle: u64) -> Option<usize> {
        (handle >> 32 == u64::from(self.tag)).then_some((handle & 0xFFFF_FFFF) as usize)
    }

    fn add_stream(&self, prio: StreamPriority) -> Result<usize, Error> {
        let priority = stream_priority(prio, self.range.0, self.range.1);
        let raw = self
            .api
            .create_stream(priority)
            .map_err(|f| stream_error(&f))?;
        let mut streams = self.streams.borrow_mut();
        streams.push(StreamSlot {
            raw,
            prio,
            handed_out: false,
        });
        Ok(streams.len() - 1)
    }

    /// The CUDA priority of each stream, in creation order, with its class.
    #[must_use]
    pub fn priorities(&self) -> Vec<(StreamPriority, i32)> {
        self.streams
            .borrow()
            .iter()
            .map(|s| (s.prio, stream_priority(s.prio, self.range.0, self.range.1)))
            .collect()
    }

    /// A planned stream of `prio` not yet handed out, or a new one. Creating a stream allocates,
    /// so it happens between steps.
    pub fn stream(&self, prio: StreamPriority) -> Result<Stream, Error> {
        let planned = self
            .streams
            .borrow_mut()
            .iter_mut()
            .enumerate()
            .find(|(_, s)| s.prio == prio && !s.handed_out)
            .map(|(i, s)| {
                s.handed_out = true;
                i
            });
        let index = match planned {
            Some(i) => i,
            None => {
                let i = self.add_stream(prio)?;
                self.streams.borrow_mut()[i].handed_out = true;
                i
            }
        };
        Ok(Stream {
            handle: self.handle(index),
        })
    }

    /// A new event. Only microbenchmarks and sampled telemetry ask for `timing` (RFC-0040 §4.3).
    pub fn event(&self, timing: bool) -> Result<Event, Error> {
        let raw = self
            .api
            .create_event(timing)
            .map_err(|f| stream_error(&f))?;
        let mut events = self.events.borrow_mut();
        events.push(EventSlot { raw, timing });
        Ok(Event {
            handle: self.handle(events.len() - 1),
        })
    }

    fn raw_stream(&self, s: &Stream) -> Result<u64, Error> {
        self.index(s.handle)
            .and_then(|i| self.streams.borrow().get(i).map(|s| s.raw))
            .ok_or_else(|| internal(format!("no stream {:#x} on this device", s.handle)))
    }

    fn raw_event(&self, e: &Event) -> Result<(u64, bool), Error> {
        self.index(e.handle)
            .and_then(|i| self.events.borrow().get(i).map(|e| (e.raw, e.timing)))
            .ok_or_else(|| internal(format!("no event {:#x} on this device", e.handle)))
    }

    /// Records `e` at the current end of `s`.
    pub fn record(&self, s: &Stream, e: &Event) -> Result<(), Error> {
        let (s, (e, _)) = (self.raw_stream(s)?, self.raw_event(e)?);
        // SAFETY: both handles come from this device's tables, whose entries stay live until
        // `drop`.
        unsafe { self.api.record(e, s) }.map_err(|f| stream_error(&f))
    }

    /// Makes later work on `s` wait for the work `e` last recorded. `e` may be on another stream.
    pub fn wait(&self, s: &Stream, e: &Event) -> Result<(), Error> {
        let (s, (e, _)) = (self.raw_stream(s)?, self.raw_event(e)?);
        // SAFETY: as in `record`.
        unsafe { self.api.wait(s, e) }.map_err(|f| stream_error(&f))
    }

    /// Whether the work `e` last recorded is done, without blocking.
    pub fn poll(&self, e: &Event) -> Result<bool, Error> {
        let (e, _) = self.raw_event(e)?;
        // SAFETY: as in `record`.
        unsafe { self.api.query(e) }.map_err(|f| stream_error(&f))
    }

    /// Device time from `start` to `end`, when both record timestamps and both are complete.
    #[must_use]
    pub fn elapsed(&self, start: &Event, end: &Event) -> Option<Nanos> {
        let (start, t0) = self.raw_event(start).ok()?;
        let (end, t1) = self.raw_event(end).ok()?;
        if !(t0 && t1) {
            return None;
        }
        // SAFETY: as in `record`.
        let ms = unsafe { self.api.elapsed_ms(start, end) }.ok()?;
        nanos(ms)
    }

    /// Blocks until every item on `s` is done. Never called in `Ready` (RFC-0040 §2.1).
    pub fn synchronize(&self, s: &Stream) -> Result<(), Error> {
        let s = self.raw_stream(s)?;
        // SAFETY: as in `record`.
        unsafe { self.api.synchronize(s) }.map_err(|f| stream_error(&f))
    }

    /// Copies `bytes` from `src` to `dst` on `s`.
    ///
    /// # Safety
    ///
    /// `src` and `dst` are addresses in the device's context valid for `bytes` until the copy
    /// completes.
    pub unsafe fn copy(&self, s: &Stream, dst: u64, src: u64, bytes: u64) -> Result<(), Error> {
        let s = self.raw_stream(s)?;
        let bytes = length(bytes)?;
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `s` is live (as in `record`); the caller vouches for the addresses.
        unsafe { self.api.copy(s, dst, src, bytes) }.map_err(|f| stream_error(&f))
    }

    /// Copies each `(dst, src, bytes)` on `s`, in order: one `cuMemcpyBatchAsync` per
    /// [`BATCH_CHUNK`] pairs where the driver has it, else one `cuMemcpyAsync` per pair. Pairs of
    /// zero length are skipped.
    ///
    /// # Safety
    ///
    /// As [`copy`](Self::copy), for every pair.
    pub unsafe fn copy_batch(
        &self,
        s: &Stream,
        pairs: impl IntoIterator<Item = Result<(u64, u64, u64), Error>>,
    ) -> Result<(), Error> {
        let s = self.raw_stream(s)?;
        if !self.api.batched() {
            for pair in pairs {
                let (dst, src, bytes) = pair?;
                let bytes = length(bytes)?;
                if bytes > 0 {
                    // SAFETY: as in `copy`.
                    unsafe { self.api.copy(s, dst, src, bytes) }.map_err(|f| stream_error(&f))?;
                }
            }
            return Ok(());
        }
        let mut dsts = [0u64; BATCH_CHUNK];
        let mut srcs = [0u64; BATCH_CHUNK];
        let mut sizes = [0usize; BATCH_CHUNK];
        let mut n = 0;
        let flush = |n: usize, dsts: &[u64], srcs: &[u64], sizes: &[usize]| {
            if n == 0 {
                return Ok(());
            }
            // SAFETY: as in `copy`; at most `BATCH_CHUNK` pairs, none empty.
            unsafe { self.api.copy_batch(s, &dsts[..n], &srcs[..n], &sizes[..n]) }
                .map_err(|f| stream_error(&f))
        };
        for pair in pairs {
            let (dst, src, bytes) = pair?;
            let bytes = length(bytes)?;
            if bytes == 0 {
                continue;
            }
            (dsts[n], srcs[n], sizes[n]) = (dst, src, bytes);
            n += 1;
            if n == BATCH_CHUNK {
                flush(n, &dsts, &srcs, &sizes)?;
                n = 0;
            }
        }
        flush(n, &dsts, &srcs, &sizes)
    }

    /// Whether `copy_batch` uses `cuMemcpyBatchAsync`.
    #[must_use]
    pub fn batched(&self) -> bool {
        self.api.batched()
    }
}

/// A copy length as the driver takes it.
fn length(bytes: u64) -> Result<usize, Error> {
    usize::try_from(bytes).map_err(|_| internal(format!("a copy of {bytes} bytes")))
}

/// Milliseconds from `cuEventElapsedTime` as nanoseconds, rounded; `None` for a negative or
/// non-finite value.
fn nanos(ms: f32) -> Option<Nanos> {
    let ns = (f64::from(ms) * 1e6).round();
    (ns.is_finite() && ns >= 0.0 && ns <= u64::MAX as f64).then_some(Nanos(ns as u64))
}

impl Drop for DeviceStreams {
    /// Drains every stream, then destroys the events and then the streams. The device's primary
    /// context is still retained here: [`CudaDevice`](crate::CudaDevice) declares its streams
    /// before its context, and fields drop in declaration order.
    fn drop(&mut self) {
        let streams = std::mem::take(self.streams.get_mut());
        let events = std::mem::take(self.events.get_mut());
        for s in &streams {
            // SAFETY: a live stream of this object. Errors are ignored: a sticky error has
            // already been reported, and the handles are destroyed regardless.
            let _ = unsafe { self.api.synchronize(s.raw) };
        }
        for e in events {
            // SAFETY: a live event; its table entry is gone, so it is not used again.
            let _ = unsafe { self.api.destroy_event(e.raw) };
        }
        for s in streams {
            // SAFETY: a live stream, drained above; its table entry is gone.
            let _ = unsafe { self.api.destroy_stream(s.raw) };
        }
    }
}

#[cfg(test)]
pub(crate) mod fake;

#[cfg(test)]
mod tests;
