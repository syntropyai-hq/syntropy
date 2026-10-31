// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `KernelProvider` for NVIDIA GPUs (RFC-0040 §7.1): binding and enumeration
//! (COMP-E4-T01).
//!
//! The provider opens the CUDA driver, `libcuda.so.1` or `nvcuda.dll`, at run time and links no
//! NVIDIA library, so a host without the driver builds, loads it and sees no CUDA device. It
//! needs a driver with CUDA 13.0's API, driver branch 580 or newer (ADR-0014); an older driver
//! fails `enumerate` with `DriverTooOld`, `(Kernels, 5)`, and the backend is skipped. `open`
//! retains the device's primary context, so code that uses the CUDA runtime later shares it.
//!
//! `open` also creates the streams of the stream plan; `stream`, `event`, `record`, `wait`, `poll`,
//! `elapsed`, `synchronize`, `copy` and `copy_batch` are [`stream`]'s (COMP-E4-T04).
//!
//! Memory, capture and the kernels are later tasks (COMP-E4-T02, COMP-E4-T05 onwards); until then
//! those methods return [`unsupported`].
//!
//! `unsafe` is allowed in this crate (RFC-0001 §4.2 rule 8). Every `unsafe` block and impl carries
//! a `// SAFETY:` comment that states its invariant (ADR-0001 item 4).

#![deny(clippy::undocumented_unsafe_blocks)]
// `core::Error` is RFC-0001 §5.12's struct with its fields inline, over clippy's 128-byte limit for
// an `Err`, as in `syntropy-kernels-cpu`. An error is the cold path.
#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

pub mod describe;
pub mod driver;
pub mod stream;

use syntropy_core::resource::Backend;
use syntropy_core::{Bytes, DeviceId, ErrorClass, ErrorCode, HostRegion, Nanos};
use syntropy_kernels::error::codes;
use syntropy_kernels::{
    AllocRequest, BufferSlice, CapturedGraph, DeviceBuffer, DeviceCtx, DeviceHealth, Error, Event,
    HostFlag, KernelProvider, LayoutVersion, MemoryDomain, OpArgs, OpCaps, OpKey, OpRegistry,
    OpenOptions, PreparedOp, ProviderBuild, ProviderDevice, ProviderId, Registration, ShapeSig,
    Stream, StreamPriority, unsupported,
};

use crate::describe::{check_driver, describe, driver_text};
use crate::driver::{Driver, Fault, RawDevice, SystemDriver};
use crate::stream::{DeviceStreams, StreamDriver, stream_error};

/// The CUDA provider, over a [`Driver`]: the real one by default, a fake in tests.
#[derive(Debug)]
pub struct CudaProvider<D: Driver = SystemDriver> {
    driver: D,
    ops: OpRegistry,
}

impl CudaProvider<SystemDriver> {
    /// The provider over the system's CUDA driver.
    #[must_use]
    pub fn new() -> Self {
        Self::with_driver(SystemDriver::default())
    }
}

impl Default for CudaProvider<SystemDriver> {
    fn default() -> Self {
        Self::new()
    }
}

/// A driver failure other than "no driver" or "no device": the backend offers no devices, and the
/// message names the call.
fn driver_error(f: &Fault) -> Error {
    Error::new(
        ErrorClass::Unsupported,
        codes::OP_UNSUPPORTED,
        format!("CUDA driver call {} failed: {}", f.call, f.error),
    )
}

/// The device a provider-local index names, with the driver's ordinal.
struct Claimed {
    raw: RawDevice,
    record: ProviderDevice,
}

impl<D: Driver> CudaProvider<D> {
    /// The provider over `driver`.
    pub fn with_driver(driver: D) -> Self {
        Self {
            driver,
            ops: OpRegistry::new(),
        }
    }

    /// The devices this provider claims, in provider-local order. No driver library, or a driver
    /// that reports no device, gives none.
    fn claimed(&self) -> Result<Vec<Claimed>, Error> {
        if !self.driver.present() {
            return Ok(Vec::new());
        }
        // The version first: every driver exports `cuDriverGetVersion`, and no newer call runs on
        // a driver too old for it.
        check_driver(self.driver.version().map_err(|f| driver_error(&f))?)?;
        match self.driver.init() {
            Err(f) if f.no_device => return Ok(Vec::new()),
            Err(f) => return Err(driver_error(&f)),
            Ok(()) => {}
        }
        let raw = self.driver.devices().map_err(|f| driver_error(&f))?;
        let mut out = Vec::with_capacity(raw.len());
        for device in raw {
            let local = u32::try_from(out.len()).unwrap_or(u32::MAX);
            if let Some(record) = describe(local, &device) {
                out.push(Claimed {
                    raw: device,
                    record,
                });
            }
        }
        Ok(out)
    }
}

/// An open device: the driver's ordinal, its streams and events, and the retained primary
/// context, released on drop.
///
/// Fields drop in declaration order: the streams are drained and destroyed, with their events,
/// before the context is released.
#[derive(Debug)]
pub struct CudaDevice<C> {
    /// The driver's ordinal.
    pub ordinal: i32,
    /// The device's record from `enumerate`.
    pub record: ProviderDevice,
    /// The device's streams and events.
    pub streams: DeviceStreams,
    _context: C,
}

impl<D: StreamDriver> CudaProvider<D>
where
    D::Context: 'static,
{
    /// The open device behind `ctx`.
    fn device<'a>(&self, ctx: &'a DeviceCtx) -> Result<&'a CudaDevice<D::Context>, Error> {
        ctx.downcast_ref().ok_or_else(|| {
            Error::new(
                ErrorClass::Internal,
                ErrorCode::new(syntropy_core::Domain::Kernels, 0),
                "the device context is not a CUDA device's",
            )
        })
    }

    /// An event that records timestamps, for [`KernelProvider::elapsed`]. Only microbenchmarks
    /// and sampled telemetry make these; the steady-state graph uses `event`'s, which do not
    /// (RFC-0040 §4.3).
    pub fn timing_event(&self, ctx: &DeviceCtx) -> Result<Event, Error> {
        self.device(ctx)?.streams.event(true)
    }
}

impl<D: StreamDriver> KernelProvider for CudaProvider<D>
where
    D: Send + Sync,
    D::Context: 'static,
{
    fn id(&self) -> ProviderId {
        ProviderId::new("cuda")
    }

    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn build(&self) -> ProviderBuild {
        let driver = if self.driver.present() {
            self.driver
                .version()
                .ok()
                .map(|v| format!("CUDA {}", driver_text(v)))
        } else {
            None
        };
        ProviderBuild {
            version: 1,
            module: None,
            toolchain: Default::default(),
            driver,
        }
    }

    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Cuda
    }

    fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error> {
        Ok(self.claimed()?.into_iter().map(|c| c.record).collect())
    }

    /// Opens the device at provider-local index `dev.ordinal`, retains its primary context and
    /// creates the streams of `opts.streams`. Every stream and event call makes the device's
    /// context current for its duration and restores the thread's previous one, so no call
    /// depends on what the thread did in between.
    fn open(&self, dev: DeviceId, opts: &OpenOptions) -> Result<DeviceCtx, Error> {
        let claimed = self
            .claimed()?
            .into_iter()
            .nth(usize::from(dev.ordinal))
            .ok_or_else(|| {
                Error::new(
                    ErrorClass::Unsupported,
                    codes::OP_UNSUPPORTED,
                    format!("no CUDA device {} on this host", dev.ordinal),
                )
            })?;
        let context = self
            .driver
            .retain(claimed.raw.ordinal)
            .map_err(|f| driver_error(&f))?;
        let api = self
            .driver
            .stream_api(&context)
            .map_err(|f| stream_error(&f))?;
        let streams = DeviceStreams::open(api, &opts.streams)?;
        Ok(DeviceCtx::new(CudaDevice {
            ordinal: claimed.raw.ordinal,
            record: claimed.record,
            streams,
            _context: context,
        }))
    }

    fn ops(&self) -> &OpRegistry {
        &self.ops
    }

    fn query(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Option<OpCaps> {
        None
    }

    fn alloc(&self, _: &DeviceCtx, _: AllocRequest) -> Result<DeviceBuffer, Error> {
        Err(unsupported())
    }

    fn free(&self, _: &DeviceCtx, _: DeviceBuffer) {}

    fn register_host(&self, _: &DeviceCtx, _: HostRegion) -> Result<Registration, Error> {
        Err(unsupported())
    }

    fn import(&self, _: &DeviceCtx, _: HostRegion) -> Result<DeviceBuffer, Error> {
        Err(unsupported())
    }

    fn repack(&self, _: &OpKey, _: &[u8], _: &mut [u8]) -> Option<Result<LayoutVersion, Error>> {
        None
    }

    fn repacked_len(&self, _: &OpKey, _: Bytes) -> Option<Bytes> {
        None
    }

    /// A stream of the plan with priority `prio` not handed out yet, or a new one.
    fn stream(&self, ctx: &DeviceCtx, prio: StreamPriority) -> Result<Stream, Error> {
        self.device(ctx)?.streams.stream(prio)
    }

    /// An event without timestamps; [`CudaProvider::timing_event`] makes one with them.
    fn event(&self, ctx: &DeviceCtx) -> Result<Event, Error> {
        self.device(ctx)?.streams.event(false)
    }

    fn record(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error> {
        self.device(ctx)?.streams.record(s, e)
    }

    fn wait(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error> {
        self.device(ctx)?.streams.wait(s, e)
    }

    fn signal_host(&self, _: &DeviceCtx, _: &Stream, _: HostFlag, _: u32) -> Result<(), Error> {
        Err(unsupported())
    }

    fn poll(&self, ctx: &DeviceCtx, e: &Event) -> Result<bool, Error> {
        self.device(ctx)?.streams.poll(e)
    }

    /// `None` unless both events come from [`CudaProvider::timing_event`] and both are complete.
    fn elapsed(&self, ctx: &DeviceCtx, start: &Event, end: &Event) -> Option<Nanos> {
        self.device(ctx).ok()?.streams.elapsed(start, end)
    }

    fn synchronize(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error> {
        self.device(ctx)?.streams.synchronize(s)
    }

    fn copy(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        src: BufferSlice,
        dst: BufferSlice,
    ) -> Result<(), Error> {
        let dev = self.device(ctx)?;
        let (dst, src, bytes) = copy_addrs(dev, &src, &dst)?;
        // SAFETY: `copy_addrs` checked both ranges lie inside buffers this device allocated or
        // imported, which the tier manager keeps until the copy's stream has passed it.
        unsafe { dev.streams.copy(s, dst, src, bytes) }
    }

    /// One `cuMemcpyBatchAsync` per [`stream::BATCH_CHUNK`] pairs on a CUDA 13 driver that
    /// exports it, else a loop of `cuMemcpyAsync` (RFC-0040 §7.7).
    fn copy_batch(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        pairs: &[(BufferSlice, BufferSlice)],
    ) -> Result<(), Error> {
        let dev = self.device(ctx)?;
        let pairs = pairs.iter().map(|(src, dst)| copy_addrs(dev, src, dst));
        // SAFETY: as in `copy`, for every pair.
        unsafe { dev.streams.copy_batch(s, pairs) }
    }

    fn prepare(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Result<PreparedOp, Error> {
        Err(unsupported())
    }

    fn launch(
        &self,
        _: &DeviceCtx,
        _: &Stream,
        _: &PreparedOp,
        _: &OpArgs<'_>,
    ) -> Result<(), Error> {
        Err(unsupported())
    }

    fn begin_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> {
        Err(unsupported())
    }

    fn end_capture(&self, _: &DeviceCtx, _: &Stream) -> Result<CapturedGraph, Error> {
        Err(unsupported())
    }

    fn replay(&self, _: &DeviceCtx, _: &Stream, _: &CapturedGraph) -> Result<(), Error> {
        Err(unsupported())
    }

    fn health(&self, _: &DeviceCtx) -> DeviceHealth {
        DeviceHealth::Ok
    }
}

/// The device addresses and length of a copy from `src` to `dst`: `(dst, src, bytes)`. Both
/// slices must lie inside buffers of this device and have the same length.
fn copy_addrs<C>(
    dev: &CudaDevice<C>,
    src: &BufferSlice,
    dst: &BufferSlice,
) -> Result<(u64, u64, u64), Error> {
    if src.bytes != dst.bytes {
        return Err(Error::new(
            ErrorClass::Internal,
            ErrorCode::new(syntropy_core::Domain::Kernels, 0),
            format!(
                "a copy of {} bytes into a slice of {}",
                src.bytes.0, dst.bytes.0
            ),
        ));
    }
    Ok((slice_addr(dev, dst)?, slice_addr(dev, src)?, src.bytes.0))
}

/// The device address of `s`'s first byte, when `s` lies inside a buffer of this device.
fn slice_addr<C>(_dev: &CudaDevice<C>, s: &BufferSlice) -> Result<u64, Error> {
    Err(Error::new(
        ErrorClass::Internal,
        ErrorCode::new(syntropy_core::Domain::Kernels, 0),
        format!("no buffer {:#x} on this device", s.buffer),
    ))
}

#[cfg(test)]
mod tests;
