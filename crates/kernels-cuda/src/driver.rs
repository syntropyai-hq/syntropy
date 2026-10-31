// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CUDA driver, behind [`Driver`] so the tests can stand in a driver of any version and any
//! set of devices (RFC-0040 §7.1).
//!
//! [`Cudarc`] loads `libcuda.so.1` (or `nvcuda.dll`) with `cudarc`'s dynamic loading. `cudarc`
//! panics when the library or one of its symbols is missing, so [`Cudarc`] asks whether the
//! library is present before any call, and reads the driver's version, which every driver
//! exports, before any call a newer driver added.

use std::ffi::c_int;

use cudarc::driver::{result, sys};

/// What one CUDA device reports, before [`crate::describe::describe()`] turns it into a `ProviderDevice`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawDevice {
    /// The driver's ordinal.
    pub ordinal: i32,
    /// `cuDeviceGetName`.
    pub name: String,
    /// `cuDeviceGetUuid`.
    pub uuid: [u8; 16],
    /// Compute capability, major and minor.
    pub cc: (i32, i32),
    /// PCI domain, bus and device numbers.
    pub pci: (u32, u32, u32),
    /// `cuDeviceTotalMem`, in bytes.
    pub total: u64,
    /// `cuMemGetInfo`'s free bytes, read in the device's primary context.
    pub free: u64,
    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED`.
    pub integrated: bool,
    /// `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`.
    pub vmm: bool,
    /// `CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED`.
    pub host_register: bool,
    /// Whether `cuDeviceCanAccessPeer` holds for some other device.
    pub peer: bool,
    /// `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`, where the device reports one.
    pub numa: Option<i32>,
}

/// A failed driver call: the call's name and the driver's error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fault {
    /// The driver entry point.
    pub call: &'static str,
    /// `cuGetErrorName`'s text, or the raw code.
    pub error: String,
    /// Whether the driver reported `CUDA_ERROR_NO_DEVICE`.
    pub no_device: bool,
}

/// The driver calls enumeration needs.
pub trait Driver {
    /// Whether the driver library can be loaded at all.
    fn present(&self) -> bool;
    /// `cuInit(0)`.
    fn init(&self) -> Result<(), Fault>;
    /// `cuDriverGetVersion`: 1000 × major + 10 × minor, e.g. 13000 for CUDA 13.0.
    fn version(&self) -> Result<i32, Fault>;
    /// Every device the driver reports, in ordinal order.
    fn devices(&self) -> Result<Vec<RawDevice>, Fault>;
    /// A retained primary context of the device at `ordinal`, released when dropped.
    type Context;
    /// `cuDevicePrimaryCtxRetain` on the device at `ordinal`.
    fn retain(&self, ordinal: i32) -> Result<Self::Context, Fault>;
}

/// A retained primary context (`cuDevicePrimaryCtxRetain`), released on drop. Code that uses the
/// CUDA runtime on the same device works in this context (RFC-0040 §7.1).
#[derive(Debug)]
pub struct PrimaryContext {
    dev: sys::CUdevice,
    ctx: sys::CUcontext,
}

impl PrimaryContext {
    /// The context handle.
    #[must_use]
    pub fn handle(&self) -> sys::CUcontext {
        self.ctx
    }
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        // SAFETY: `dev` is the device whose primary context `retain` retained once; this is the
        // matching release, run once.
        let _ = unsafe { result::primary_ctx::release(self.dev) };
    }
}

/// The real driver, through `cudarc`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cudarc;

fn fault(call: &'static str, e: result::DriverError) -> Fault {
    Fault {
        call,
        error: e.error_name().map_or_else(
            |_| format!("{:?}", e.0),
            |n| n.to_string_lossy().into_owned(),
        ),
        no_device: e.0 == sys::cudaError_enum::CUDA_ERROR_NO_DEVICE,
    }
}

impl Cudarc {
    fn attribute(dev: sys::CUdevice, attr: sys::CUdevice_attribute) -> Result<i32, Fault> {
        // SAFETY: `dev` came from `cuDeviceGet` after a successful `cuInit`, and the attribute is
        // one the CUDA 13 driver defines; the call only writes the returned integer.
        unsafe { result::device::get_attribute(dev, attr) }
            .map_err(|e| fault("cuDeviceGetAttribute", e))
    }

    /// Free memory, read in the device's primary context, which is retained and released here.
    fn free(dev: sys::CUdevice) -> Result<u64, Fault> {
        // SAFETY: `dev` is a valid device handle. The primary context retained here is made
        // current only for the `cuMemGetInfo` call and released before returning, so this
        // thread's context stack is restored and the retain count stays balanced.
        unsafe {
            let previous = result::ctx::get_current().map_err(|e| fault("cuCtxGetCurrent", e))?;
            let ctx = result::primary_ctx::retain(dev)
                .map_err(|e| fault("cuDevicePrimaryCtxRetain", e))?;
            let info = result::ctx::set_current(ctx)
                .map_err(|e| fault("cuCtxSetCurrent", e))
                .and_then(|()| result::mem_get_info().map_err(|e| fault("cuMemGetInfo", e)));
            let restored = result::ctx::set_current(previous.unwrap_or(std::ptr::null_mut()))
                .map_err(|e| fault("cuCtxSetCurrent", e));
            let released = result::primary_ctx::release(dev)
                .map_err(|e| fault("cuDevicePrimaryCtxRelease", e));
            let (free, _total) = info?;
            restored?;
            released?;
            Ok(free as u64)
        }
    }
}

impl Driver for Cudarc {
    type Context = PrimaryContext;

    fn retain(&self, ordinal: i32) -> Result<PrimaryContext, Fault> {
        let dev = result::device::get(ordinal).map_err(|e| fault("cuDeviceGet", e))?;
        // SAFETY: `dev` came from `cuDeviceGet`; the retained context is released by
        // `PrimaryContext`'s drop.
        let ctx = unsafe { result::primary_ctx::retain(dev) }
            .map_err(|e| fault("cuDevicePrimaryCtxRetain", e))?;
        Ok(PrimaryContext { dev, ctx })
    }

    fn present(&self) -> bool {
        // SAFETY: only tries to load the library; it calls nothing in it.
        unsafe { sys::is_culib_present() }
    }

    fn init(&self) -> Result<(), Fault> {
        result::init().map_err(|e| fault("cuInit", e))
    }

    fn version(&self) -> Result<i32, Fault> {
        let mut version: c_int = 0;
        // SAFETY: `cuDriverGetVersion` exists in every driver and writes one integer through the
        // pointer, which points at a live local.
        let status = unsafe { sys::cuDriverGetVersion(&raw mut version) };
        status
            .result()
            .map(|()| version)
            .map_err(|e| fault("cuDriverGetVersion", e))
    }

    fn devices(&self) -> Result<Vec<RawDevice>, Fault> {
        use sys::CUdevice_attribute as A;
        let count = result::device::get_count().map_err(|e| fault("cuDeviceGetCount", e))?;
        let handles = (0..count)
            .map(|i| result::device::get(i).map_err(|e| fault("cuDeviceGet", e)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = Vec::with_capacity(handles.len());
        for (ordinal, &dev) in (0..count).zip(&handles) {
            let mut peer = false;
            for &other in handles.iter().filter(|&&o| o != dev) {
                let mut can: c_int = 0;
                // SAFETY: both handles came from `cuDeviceGet`; the call writes one integer
                // through a pointer to a live local.
                let status = unsafe { sys::cuDeviceCanAccessPeer(&raw mut can, dev, other) };
                status
                    .result()
                    .map_err(|e| fault("cuDeviceCanAccessPeer", e))?;
                peer |= can != 0;
            }
            // SAFETY: `dev` is a valid device handle; the call writes one size.
            let total = unsafe { result::device::total_mem(dev) }
                .map_err(|e| fault("cuDeviceTotalMem", e))?;
            let uuid = result::device::get_uuid(dev).map_err(|e| fault("cuDeviceGetUuid", e))?;
            let numa = Self::attribute(dev, A::CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID)?;
            out.push(RawDevice {
                ordinal,
                name: result::device::get_name(dev).map_err(|e| fault("cuDeviceGetName", e))?,
                uuid: uuid.bytes.map(|b| b as u8),
                cc: (
                    Self::attribute(dev, A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                    Self::attribute(dev, A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
                ),
                pci: (
                    u32::try_from(Self::attribute(dev, A::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?)
                        .unwrap_or(0),
                    u32::try_from(Self::attribute(dev, A::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?)
                        .unwrap_or(0),
                    u32::try_from(Self::attribute(dev, A::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?)
                        .unwrap_or(0),
                ),
                total: total as u64,
                free: Self::free(dev)?,
                integrated: Self::attribute(dev, A::CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0,
                vmm: Self::attribute(
                    dev,
                    A::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                )? != 0,
                host_register: Self::attribute(
                    dev,
                    A::CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED,
                )? != 0,
                peer,
                numa: (numa >= 0).then_some(numa),
            });
        }
        Ok(out)
    }
}
