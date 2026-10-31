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
//! [`memory`] allocates device slabs and host-mapped pages, and registers and imports the tier
//! manager's host memory (COMP-E4-T02). Streams, capture and the kernels are later tasks
//! (COMP-E4-T04 onwards); until then those methods return [`unsupported`].
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
pub mod memory;

use syntropy_core::resource::Backend;
use syntropy_core::{Bytes, DeviceId, ErrorClass, HostRegion};
use syntropy_kernels::error::codes;
use syntropy_kernels::{
    AllocRequest, BufferSlice, CapturedGraph, DeviceBuffer, DeviceCtx, DeviceHealth, Error, Event,
    HostFlag, KernelProvider, LayoutVersion, MemoryDomain, OpArgs, OpCaps, OpKey, OpRegistry,
    OpenOptions, PreparedOp, ProviderBuild, ProviderDevice, ProviderId, Registration, ShapeSig,
    Stream, StreamPriority, unsupported,
};

use crate::describe::{check_driver, describe, driver_text};
use crate::driver::{Driver, Fault, RawDevice, SystemDriver};
use crate::memory::{AllocHook, DeviceMemory, HostRanges, MemoryDriver};

/// The CUDA provider, over a [`Driver`]: the real one by default, a fake in tests.
pub struct CudaProvider<D: Driver = SystemDriver> {
    driver: D,
    ops: OpRegistry,
    hook: Option<AllocHook>,
    /// Host ranges registered with the driver, shared by every device this provider opens.
    host_ranges: std::sync::Arc<HostRanges>,
}

impl<D: Driver + std::fmt::Debug> std::fmt::Debug for CudaProvider<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaProvider")
            .field("driver", &self.driver)
            .field("ops", &self.ops)
            .field("hook", &self.hook.is_some())
            .finish()
    }
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
            hook: None,
            host_ranges: std::sync::Arc::default(),
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

/// An open device: the driver's ordinal and the retained primary context, released on drop.
#[derive(Debug)]
pub struct CudaDevice<C> {
    /// The driver's ordinal.
    pub ordinal: i32,
    /// The device's record from `enumerate`.
    pub record: ProviderDevice,
    /// What the tier manager allocated. Declared before the context, so whatever is left is
    /// freed before the context is released.
    pub(crate) memory: DeviceMemory,
    _context: C,
}

impl<D: MemoryDriver> KernelProvider for CudaProvider<D>
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

    /// Opens the device at provider-local index `dev.ordinal` and retains its primary context.
    fn open(&self, dev: DeviceId, _opts: &OpenOptions) -> Result<DeviceCtx, Error> {
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
        let memory = self.driver.memory(&context).map_err(|f| driver_error(&f))?;
        Ok(DeviceCtx::new(CudaDevice {
            ordinal: claimed.raw.ordinal,
            record: claimed.record,
            memory: DeviceMemory::new(
                memory,
                self.hook.clone(),
                std::sync::Arc::clone(&self.host_ranges),
            ),
            _context: context,
        }))
    }

    fn ops(&self) -> &OpRegistry {
        &self.ops
    }

    fn query(&self, _: &DeviceCtx, _: &OpKey, _: &ShapeSig) -> Option<OpCaps> {
        None
    }

    /// A `Device` fixed slab or `HostMapped` doorbell pages (RFC-0040 §5.1; [`memory`]).
    fn alloc(&self, ctx: &DeviceCtx, req: AllocRequest) -> Result<DeviceBuffer, Error> {
        self.memory(ctx)?.alloc(req)
    }

    fn free(&self, ctx: &DeviceCtx, buf: DeviceBuffer) {
        if let Ok(m) = self.memory(ctx) {
            m.free(&buf);
        }
    }

    /// One RFC-0020 §6.1 unit of whole host pages (RFC-0040 §5.2; [`memory`]).
    fn register_host(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<Registration, Error> {
        self.memory(ctx)?.register_host(region)
    }

    fn import(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<DeviceBuffer, Error> {
        self.memory(ctx)?.import(region)
    }

    fn repack(&self, _: &OpKey, _: &[u8], _: &mut [u8]) -> Option<Result<LayoutVersion, Error>> {
        None
    }

    fn repacked_len(&self, _: &OpKey, _: Bytes) -> Option<Bytes> {
        None
    }

    fn stream(&self, _: &DeviceCtx, _: StreamPriority) -> Result<Stream, Error> {
        Err(unsupported())
    }

    fn event(&self, _: &DeviceCtx) -> Result<Event, Error> {
        Err(unsupported())
    }

    fn record(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> {
        Err(unsupported())
    }

    fn wait(&self, _: &DeviceCtx, _: &Stream, _: &Event) -> Result<(), Error> {
        Err(unsupported())
    }

    fn signal_host(&self, _: &DeviceCtx, _: &Stream, _: HostFlag, _: u32) -> Result<(), Error> {
        Err(unsupported())
    }

    fn poll(&self, _: &DeviceCtx, _: &Event) -> Result<bool, Error> {
        Err(unsupported())
    }

    fn synchronize(&self, _: &DeviceCtx, _: &Stream) -> Result<(), Error> {
        Err(unsupported())
    }

    fn copy(&self, _: &DeviceCtx, _: &Stream, _: BufferSlice, _: BufferSlice) -> Result<(), Error> {
        Err(unsupported())
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

#[cfg(test)]
mod tests;
