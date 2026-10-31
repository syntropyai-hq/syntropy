// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CUDA driver, behind [`Driver`] so the tests can stand in a driver of any version and any
//! set of devices (RFC-0040 §7.1).
//!
//! [`SystemDriver`] opens exactly [`DRIVER_LIBRARY`], `libcuda.so.1` or `nvcuda.dll`, with
//! `libloading`, and resolves the entry points enumeration needs itself. It never asks for the
//! unversioned `libcuda.so`: that name is a development link, which the loader may resolve to a
//! different file from the installed driver. On one WSL2 host it named a native-Linux driver
//! that cannot reach the GPU, so `cuInit` reported no device (RFC-0040 §7.1, note 2026-10-31).
//! `cudarc` supplies the driver API's types only; its own loader tries `libcuda.so` first.
//!
//! A missing library means no device. The driver's version, which every driver exports, is read
//! before any entry point a newer driver added is used.

use std::ffi::{CStr, c_char, c_int, c_uint};
use std::mem::MaybeUninit;
use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::{CUcontext, CUdevice, CUdevice_attribute, CUresult, CUuuid};

/// A driver call's status, as the driver returns it: a plain integer. It is never read as
/// `cudarc`'s `CUresult` enum, which lists only the codes CUDA 13.4 defines, so a newer driver's
/// code is a value, not an invalid enum.
pub type Status = c_uint;

/// `CUDA_SUCCESS`.
pub const SUCCESS: Status = CUresult::CUDA_SUCCESS as Status;

/// `CUDA_ERROR_NO_DEVICE`.
pub const NO_DEVICE: Status = CUresult::CUDA_ERROR_NO_DEVICE as Status;

/// The driver library the provider opens, and the only name it asks the loader for.
pub const DRIVER_LIBRARY: &str = if cfg!(windows) {
    "nvcuda.dll"
} else {
    "libcuda.so.1"
};

/// What one CUDA device reports, before [`crate::describe::describe()`] turns it into a
/// `ProviderDevice`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawDevice {
    /// The driver's ordinal.
    pub ordinal: i32,
    /// `cuDeviceGetName`.
    pub name: String,
    /// `cuDeviceGetUuid_v2`.
    pub uuid: [u8; 16],
    /// Compute capability, major and minor.
    pub cc: (i32, i32),
    /// PCI domain, bus and device numbers.
    pub pci: (u32, u32, u32),
    /// `cuDeviceTotalMem_v2`, in bytes.
    pub total: u64,
    /// `cuMemGetInfo_v2`'s free bytes, read in the device's primary context.
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
    /// A retained primary context of one device, released when dropped.
    type Context;
    /// Whether the driver library can be opened at all.
    fn present(&self) -> bool;
    /// `cuInit(0)`.
    fn init(&self) -> Result<(), Fault>;
    /// `cuDriverGetVersion`: 1000 × major + 10 × minor, e.g. 13000 for CUDA 13.0.
    fn version(&self) -> Result<i32, Fault>;
    /// Every device the driver reports, in ordinal order.
    fn devices(&self) -> Result<Vec<RawDevice>, Fault>;
    /// `cuDevicePrimaryCtxRetain` on the device at `ordinal`.
    fn retain(&self, ordinal: i32) -> Result<Self::Context, Fault>;
}

/// The entry points used after the version check.
#[derive(Clone, Copy)]
struct Calls {
    init: unsafe extern "C" fn(c_uint) -> Status,
    count: unsafe extern "C" fn(*mut c_int) -> Status,
    get: unsafe extern "C" fn(*mut CUdevice, c_int) -> Status,
    name: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> Status,
    uuid: unsafe extern "C" fn(*mut CUuuid, CUdevice) -> Status,
    total: unsafe extern "C" fn(*mut usize, CUdevice) -> Status,
    attribute: unsafe extern "C" fn(*mut c_int, CUdevice_attribute, CUdevice) -> Status,
    can_access_peer: unsafe extern "C" fn(*mut c_int, CUdevice, CUdevice) -> Status,
    retain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> Status,
    release: unsafe extern "C" fn(CUdevice) -> Status,
    get_current: unsafe extern "C" fn(*mut CUcontext) -> Status,
    set_current: unsafe extern "C" fn(CUcontext) -> Status,
    mem_info: unsafe extern "C" fn(*mut usize, *mut usize) -> Status,
}

/// The open driver library and its entry points. The pointers stay valid while `_lib` lives,
/// and every holder of a pointer holds the `Api` through an `Arc`.
pub(crate) struct Api {
    _lib: libloading::Library,
    version: unsafe extern "C" fn(*mut c_int) -> Status,
    error_name: Option<unsafe extern "C" fn(Status, *mut *const c_char) -> Status>,
    /// The other entry points, or the first one the library lacks.
    calls: Result<Calls, &'static str>,
}

/// One entry point of `lib`, as a plain function pointer of type `F`.
pub(crate) fn sym<F: Copy>(
    lib: &libloading::Library,
    name: &'static str,
) -> Result<F, &'static str> {
    // SAFETY: `F` is the entry point's C signature from the CUDA 13 driver API, as `cudarc`'s
    // bindings declare it, with the status returned as `c_uint` (`CUresult` is `#[repr(u32)]`, so
    // the ABI is the same); the pointer is used only while `lib` is loaded (see `Api`).
    unsafe { lib.get::<F>(name.as_bytes()) }
        .map(|s| *s)
        .map_err(|_| name)
}

/// Opens [`DRIVER_LIBRARY`] through `open`, the only name it passes, and resolves the entry
/// points. `None` when the library or `cuDriverGetVersion` is missing.
fn load<E>(open: &mut dyn FnMut(&str) -> Result<libloading::Library, E>) -> Option<Api> {
    let lib = open(DRIVER_LIBRARY).ok()?;
    let version = sym(&lib, "cuDriverGetVersion").ok()?;
    let error_name = sym(&lib, "cuGetErrorName").ok();
    let calls = (|| {
        Ok(Calls {
            init: sym(&lib, "cuInit")?,
            count: sym(&lib, "cuDeviceGetCount")?,
            get: sym(&lib, "cuDeviceGet")?,
            name: sym(&lib, "cuDeviceGetName")?,
            uuid: sym(&lib, "cuDeviceGetUuid_v2")?,
            total: sym(&lib, "cuDeviceTotalMem_v2")?,
            attribute: sym(&lib, "cuDeviceGetAttribute")?,
            can_access_peer: sym(&lib, "cuDeviceCanAccessPeer")?,
            retain: sym(&lib, "cuDevicePrimaryCtxRetain")?,
            release: sym(&lib, "cuDevicePrimaryCtxRelease_v2")?,
            get_current: sym(&lib, "cuCtxGetCurrent")?,
            set_current: sym(&lib, "cuCtxSetCurrent")?,
            mem_info: sym(&lib, "cuMemGetInfo_v2")?,
        })
    })();
    Some(Api {
        _lib: lib,
        version,
        error_name,
        calls,
    })
}

/// The fault for a failed `call`: the driver's name for `status` where it has one, else the raw
/// number.
fn fault(call: &'static str, status: Status, name: Option<String>) -> Fault {
    Fault {
        call,
        error: name.unwrap_or_else(|| format!("CUDA error {status}")),
        no_device: status == NO_DEVICE,
    }
}

impl Api {
    /// `Ok` for `CUDA_SUCCESS`, else the fault for `call`.
    fn check(&self, call: &'static str, status: Status) -> Result<(), Fault> {
        if status == SUCCESS {
            return Ok(());
        }
        Err(fault(call, status, self.error_name(status)))
    }

    /// `cuGetErrorName`'s text for `status`, when the driver knows the code.
    pub(crate) fn error_name(&self, status: Status) -> Option<String> {
        let f = self.error_name?;
        let mut text: *const c_char = std::ptr::null();
        // SAFETY: `cuGetErrorName` takes any integer code and writes a pointer to a static
        // string, or returns an error for a code it does not know.
        if unsafe { f(status, &raw mut text) } != SUCCESS || text.is_null() {
            return None;
        }
        // SAFETY: on success the driver returned a NUL-terminated static string.
        Some(
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    /// The open driver library, for modules that resolve entry points of their own.
    pub(crate) fn lib(&self) -> &libloading::Library {
        &self._lib
    }

    fn calls(&self) -> Result<&Calls, Fault> {
        self.calls.as_ref().map_err(|&missing| Fault {
            call: missing,
            error: "the driver library lacks this entry point".into(),
            no_device: false,
        })
    }
}

/// Opens the driver library `name`. On Windows only `System32` is searched
/// (`LOAD_LIBRARY_SEARCH_SYSTEM32`), where the display driver installs `nvcuda.dll`, so a copy in
/// the working or application directory is never loaded. Elsewhere the dynamic linker's search
/// order applies to the soname.
///
/// # Safety
///
/// Loading runs the library's initialisers, as any `dlopen` or `LoadLibraryExW` does.
unsafe fn open_driver(name: &str) -> Result<libloading::Library, libloading::Error> {
    #[cfg(windows)]
    {
        use libloading::os::windows::{LOAD_LIBRARY_SEARCH_SYSTEM32, Library};
        // SAFETY: the caller's contract: the driver's initialisers run.
        unsafe { Library::load_with_flags(name, LOAD_LIBRARY_SEARCH_SYSTEM32) }.map(Into::into)
    }
    #[cfg(not(windows))]
    {
        // SAFETY: the caller's contract: the driver's initialisers run.
        unsafe { libloading::Library::new(name) }
    }
}

/// The installed driver, opened once on first use.
#[derive(Default)]
pub struct SystemDriver {
    api: OnceLock<Option<Arc<Api>>>,
}

impl std::fmt::Debug for SystemDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemDriver")
            .field("library", &DRIVER_LIBRARY)
            .finish_non_exhaustive()
    }
}

impl SystemDriver {
    fn api(&self) -> Option<&Arc<Api>> {
        self.api
            .get_or_init(|| {
                // SAFETY: opening the driver runs its initialisers, which the driver supports;
                // the library stays open while `Api` lives.
                load(&mut |name| unsafe { open_driver(name) }).map(Arc::new)
            })
            .as_ref()
    }

    fn need(&self) -> Result<&Arc<Api>, Fault> {
        self.api().ok_or_else(|| Fault {
            call: "dlopen",
            error: format!("{DRIVER_LIBRARY} is not installed"),
            no_device: true,
        })
    }

    fn attribute(api: &Api, dev: CUdevice, attr: CUdevice_attribute) -> Result<i32, Fault> {
        let mut v: c_int = 0;
        // SAFETY: `dev` came from `cuDeviceGet` after `cuInit`; the call writes one integer
        // through a pointer to a live local.
        let s = unsafe { (api.calls()?.attribute)(&raw mut v, attr, dev) };
        api.check("cuDeviceGetAttribute", s).map(|()| v)
    }

    /// Free memory, read in the device's primary context, which is retained and released here.
    fn free(api: &Api, dev: CUdevice) -> Result<u64, Fault> {
        let c = api.calls()?;
        let mut previous: CUcontext = std::ptr::null_mut();
        let mut ctx: CUcontext = std::ptr::null_mut();
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: `dev` is a valid device. The retained primary context is current only for the
        // `cuMemGetInfo_v2` call; the previous context is restored and the retain is released
        // before returning, so this thread's context and the retain count are unchanged.
        unsafe {
            api.check("cuCtxGetCurrent", (c.get_current)(&raw mut previous))?;
            api.check("cuDevicePrimaryCtxRetain", (c.retain)(&raw mut ctx, dev))?;
            let info = api
                .check("cuCtxSetCurrent", (c.set_current)(ctx))
                .and_then(|()| {
                    api.check(
                        "cuMemGetInfo_v2",
                        (c.mem_info)(&raw mut free, &raw mut total),
                    )
                });
            let restored = api.check("cuCtxSetCurrent", (c.set_current)(previous));
            let released = api.check("cuDevicePrimaryCtxRelease_v2", (c.release)(dev));
            info?;
            restored?;
            released?;
        }
        Ok(free as u64)
    }
}

/// A retained primary context (`cuDevicePrimaryCtxRetain`), released on drop. Code that uses the
/// CUDA runtime on the same device works in this context (RFC-0040 §7.1).
pub struct PrimaryContext {
    api: Arc<Api>,
    dev: CUdevice,
    ctx: CUcontext,
}

impl std::fmt::Debug for PrimaryContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimaryContext")
            .field("dev", &self.dev)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl PrimaryContext {
    /// The context handle.
    #[must_use]
    pub fn handle(&self) -> CUcontext {
        self.ctx
    }

    /// The driver library the context was retained through.
    pub(crate) fn api(&self) -> &Arc<Api> {
        &self.api
    }
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        if let Ok(c) = self.api.calls() {
            // SAFETY: `dev` is the device whose primary context `retain` retained once; this is
            // the matching release, run once, while the library is loaded (`api` holds it).
            let _ = unsafe { (c.release)(self.dev) };
        }
    }
}

impl Driver for SystemDriver {
    type Context = PrimaryContext;

    fn present(&self) -> bool {
        self.api().is_some()
    }

    fn init(&self) -> Result<(), Fault> {
        let api = self.need()?;
        // SAFETY: `cuInit(0)` takes no pointers; 0 is the only defined flag value.
        let s = unsafe { (api.calls()?.init)(0) };
        api.check("cuInit", s)
    }

    fn version(&self) -> Result<i32, Fault> {
        let api = self.need()?;
        let mut v: c_int = 0;
        // SAFETY: `cuDriverGetVersion` needs no `cuInit` and writes one integer through a
        // pointer to a live local.
        let s = unsafe { (api.version)(&raw mut v) };
        api.check("cuDriverGetVersion", s).map(|()| v)
    }

    fn devices(&self) -> Result<Vec<RawDevice>, Fault> {
        use CUdevice_attribute as A;
        let api = self.need()?;
        let c = *api.calls()?;
        let mut count: c_int = 0;
        // SAFETY: writes one integer through a pointer to a live local.
        api.check("cuDeviceGetCount", unsafe { (c.count)(&raw mut count) })?;
        let mut handles = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        for i in 0..count {
            let mut dev = MaybeUninit::<CUdevice>::uninit();
            // SAFETY: `i` is below the count the driver reported; the call writes one handle.
            api.check("cuDeviceGet", unsafe { (c.get)(dev.as_mut_ptr(), i) })?;
            // SAFETY: `cuDeviceGet` succeeded, so it wrote the handle.
            handles.push(unsafe { dev.assume_init() });
        }
        let mut out = Vec::with_capacity(handles.len());
        for (ordinal, &dev) in (0..count).zip(&handles) {
            let mut peer = false;
            for &other in handles.iter().filter(|&&o| o != dev) {
                let mut can: c_int = 0;
                // SAFETY: both handles came from `cuDeviceGet`; writes one integer.
                let s = unsafe { (c.can_access_peer)(&raw mut can, dev, other) };
                api.check("cuDeviceCanAccessPeer", s)?;
                peer |= can != 0;
            }
            let mut name = [0 as c_char; 256];
            // SAFETY: the buffer holds 256 bytes and the driver writes at most that many,
            // NUL-terminated.
            let s = unsafe { (c.name)(name.as_mut_ptr(), 256, dev) };
            api.check("cuDeviceGetName", s)?;
            // SAFETY: the driver NUL-terminated the name inside the buffer.
            let name = unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let mut uuid = MaybeUninit::<CUuuid>::uninit();
            // SAFETY: writes one `CUuuid` through a pointer to a live local.
            let s = unsafe { (c.uuid)(uuid.as_mut_ptr(), dev) };
            api.check("cuDeviceGetUuid_v2", s)?;
            // SAFETY: the call succeeded, so the 16 bytes are written.
            let uuid = unsafe { uuid.assume_init() }.bytes.map(|b| b as u8);
            let mut total: usize = 0;
            // SAFETY: writes one size through a pointer to a live local.
            let s = unsafe { (c.total)(&raw mut total, dev) };
            api.check("cuDeviceTotalMem_v2", s)?;
            let attr = |a| Self::attribute(api, dev, a);
            let numa = attr(A::CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID)?;
            out.push(RawDevice {
                ordinal,
                name,
                uuid,
                cc: (
                    attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                    attr(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
                ),
                pci: (
                    u32::try_from(attr(A::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?).unwrap_or(0),
                    u32::try_from(attr(A::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?).unwrap_or(0),
                    u32::try_from(attr(A::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?).unwrap_or(0),
                ),
                total: total as u64,
                free: Self::free(api, dev)?,
                integrated: attr(A::CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0,
                vmm: attr(A::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED)? != 0,
                host_register: attr(A::CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED)? != 0,
                peer,
                numa: (numa >= 0).then_some(numa),
            });
        }
        Ok(out)
    }

    fn retain(&self, ordinal: i32) -> Result<PrimaryContext, Fault> {
        let api = self.need()?;
        let c = api.calls()?;
        let mut dev = MaybeUninit::<CUdevice>::uninit();
        // SAFETY: writes one handle; an out-of-range ordinal fails the call.
        api.check("cuDeviceGet", unsafe { (c.get)(dev.as_mut_ptr(), ordinal) })?;
        // SAFETY: `cuDeviceGet` succeeded, so it wrote the handle.
        let dev = unsafe { dev.assume_init() };
        let mut ctx: CUcontext = std::ptr::null_mut();
        // SAFETY: `dev` is valid; the retained context is released by `PrimaryContext`'s drop.
        let s = unsafe { (c.retain)(&raw mut ctx, dev) };
        api.check("cuDevicePrimaryCtxRetain", s)?;
        Ok(PrimaryContext {
            api: Arc::clone(api),
            dev,
            ctx,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_driver_library_is_the_versioned_soname() {
        if cfg!(windows) {
            assert_eq!(DRIVER_LIBRARY, "nvcuda.dll");
        } else {
            assert_eq!(DRIVER_LIBRARY, "libcuda.so.1");
        }
    }

    /// The loader asks for exactly one name, the versioned driver library, and never the
    /// unversioned development link `libcuda.so` (RFC-0040 §7.1, note 2026-10-31).
    #[test]
    fn the_loader_asks_only_for_the_driver_library() {
        let mut asked = Vec::new();
        let api = load::<&str>(&mut |name| {
            asked.push(name.to_owned());
            Err("not installed")
        });
        assert!(api.is_none());
        assert_eq!(asked, [DRIVER_LIBRARY]);
        assert!(asked.iter().all(|n| n != "libcuda.so"));
    }

    /// A code newer than the CUDA 13.4 list is kept as a number: it is never turned into
    /// `cudarc`'s enum, where it would be an invalid value.
    #[test]
    fn an_unknown_status_code_is_reported_by_number() {
        let f = fault("cuInit", 0xFFFF, None);
        assert_eq!(f.error, "CUDA error 65535");
        assert!(!f.no_device);
        let f = fault("cuInit", NO_DEVICE, Some("CUDA_ERROR_NO_DEVICE".into()));
        assert_eq!(f.error, "CUDA_ERROR_NO_DEVICE");
        assert!(f.no_device);
        assert_eq!((SUCCESS, NO_DEVICE), (0, 100));
    }

    #[test]
    fn a_missing_library_means_no_driver() {
        let d = SystemDriver::default();
        // Fix the cache to "no library", as `load` gives when `open` fails.
        let _ = d.api.set(None);
        assert!(!d.present());
        let f = d.version().unwrap_err();
        assert!(f.no_device, "{f:?}");
    }
}
