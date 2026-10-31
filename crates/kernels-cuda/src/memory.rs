// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory (RFC-0040 §5.1, §5.2; COMP-E4-T02): `alloc` for `Device` as fixed slabs and for
//! `HostMapped` doorbell pages, `register_host` per RFC-0020 §6.1 unit, and `import`.
//!
//! | Request | Driver calls | `free` |
//! |---|---|---|
//! | `alloc`, `Device` | `cuMemAlloc_v2`, one fixed slab per request | `cuMemFree_v2` |
//! | `alloc`, `HostMapped` | `cuMemHostAlloc` (portable, device-mapped), zeroed; `cuMemHostGetDevicePointer_v2` | `cuMemFreeHost` |
//! | `register_host` | `cuMemHostRegister_v2` (portable, device-mapped) over the tier manager's range, once per provider | [`CudaProvider::unregister_host`]: `cuMemHostUnregister` after the last device's release |
//! | `import` | as `register_host`; then `cuMemHostGetDevicePointer_v2` in the importing device's context | as `register_host` |
//!
//! `reserve_only` and VMM segments are COMP-E4-T13's, and `Unified` is COMP-E4-T08's; both
//! return `Unsupported` here.
//!
//! **Only the tier manager's requests allocate** (RFC-0001 §9.1). Every driver call that takes
//! memory runs in [`DeviceMemory`], from `alloc`, `register_host` or `import`, and nowhere else:
//! `enumerate`, `open` and every other method allocate nothing. An [`AllocHook`] installed with
//! [`CudaProvider::with_alloc_hook`] sees each such call as it succeeds, which is how a test
//! checks that every provider allocation answers a request.
//!
//! **Registration units.** The tier manager registers one RFC-0020 §6.1 unit per call, aligned
//! to extent, slot and elastic-segment boundaries, once for each device in the arena's
//! `dma_devices` (RFC-0020 §6). The provider holds each unit to whole host pages
//! ([`host_page`]). On one device it refuses a range that overlaps one the device already
//! holds; across devices, a unit must be exactly the range another device registered, or not
//! overlap it at all. So no host page, and therefore no DMA source, belongs to two
//! registrations. [`CudaProvider::registration_for`] names the one registration that holds a
//! whole source range, and gives `None` for a range that crosses two.
//!
//! **One driver registration per range.** Registrations are portable, so the driver pins a
//! range once for every context, and a second `cuMemHostRegister_v2` of it would fail. The
//! provider's [`HostRanges`] therefore registers a range with the driver for its first device,
//! counts the devices that hold it, and unregisters it when the last one releases. Each device
//! still gets a handle of its own, and one device's release or close never tears down another's.
//! The entry keeps the first device's memory calls, with their retain of its primary context,
//! until the driver unregisters the range.
//!
//! **Lifetimes.** Handles are unique across every device of the process and never reused, so a
//! stale or foreign handle names nothing and `free` ignores it: no driver allocation is freed
//! twice. Whatever the tier manager has not freed when the device closes is freed then, before
//! the primary context is released.
//!
//! `free` and `unregister_host` run between steps: `cuMemFree_v2` may synchronise the device,
//! and the stream rules of RFC-0040 §2.4 keep it off the hot path.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{c_uint, c_void};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys::{
    CU_MEMHOSTALLOC_DEVICEMAP, CU_MEMHOSTALLOC_PORTABLE, CU_MEMHOSTREGISTER_DEVICEMAP,
    CU_MEMHOSTREGISTER_PORTABLE, CUdeviceptr,
};
use syntropy_core::{Bytes, ErrorClass, ErrorCode, HostRegion};
use syntropy_kernels::error::codes;
use syntropy_kernels::{
    AllocRequest, BufferSlice, DeviceBuffer, DeviceCtx, Error, MemKind, Registration, unsupported,
};

use crate::driver::{Driver, Fault, PrimaryContext, Status, SystemDriver};
use crate::{CudaDevice, CudaProvider};

/// The alignment `cuMemAlloc_v2` gives every allocation: at least 256 bytes (CUDA C++
/// Programming Guide, "Device Memory Accesses"). A larger `align` is met by padding the slab.
pub const DEVICE_ALIGN: u64 = 256;

/// The smallest host page the provider assumes: 4 KiB.
pub const MIN_HOST_PAGE: u64 = 4096;

/// The host's page size: `cuMemHostAlloc` returns page-aligned memory, and a registered or
/// imported range must start and end on a page boundary, so no page is pinned by two
/// registrations. On Linux it is the kernel's `AT_PAGESZ` from `/proc/self/auxv`, which is
/// 64 KiB on some arm64 kernels; elsewhere, and if that file cannot be read, [`MIN_HOST_PAGE`].
pub fn host_page() -> u64 {
    static PAGE: OnceLock<u64> = OnceLock::new();
    *PAGE.get_or_init(|| {
        std::fs::read("/proc/self/auxv")
            .ok()
            .and_then(|auxv| auxv_page_size(&auxv))
            .map_or(MIN_HOST_PAGE, |p| p.max(MIN_HOST_PAGE))
    })
}

/// `AT_PAGESZ` (6) from an auxiliary vector of native-endian, word-sized (type, value) pairs;
/// `None` when it is absent or not a power of two.
fn auxv_page_size(auxv: &[u8]) -> Option<u64> {
    const AT_PAGESZ: u64 = 6;
    const WORD: usize = std::mem::size_of::<usize>();
    let words = auxv.as_chunks::<WORD>().0;
    words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[t, v]| {
            (
                usize::from_ne_bytes(*t) as u64,
                usize::from_ne_bytes(*v) as u64,
            )
        })
        .find(|&(t, _)| t == AT_PAGESZ)
        .map(|(_, v)| v)
        .filter(|v| v.is_power_of_two())
}

/// What the tier manager's request made the provider take from the driver: what an
/// [`AllocHook`] sees, once per successful driver call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemEvent {
    /// `alloc`: a `Device` slab or `HostMapped` pages, with the bytes taken from the driver,
    /// padding for alignment included.
    Alloc {
        /// The kind requested.
        kind: MemKind,
        /// Bytes taken from the driver.
        bytes: u64,
        /// The handle of the returned buffer.
        handle: u64,
    },
    /// `register_host`.
    Register {
        /// The registered range.
        region: HostRegion,
        /// The registration's handle.
        handle: u64,
    },
    /// `import`.
    Import {
        /// The imported range.
        region: HostRegion,
        /// The buffer's handle.
        handle: u64,
    },
    /// `free`, `unregister_host`, or the device closing: the memory behind `handle` went back
    /// to the driver.
    Release {
        /// The released handle.
        handle: u64,
    },
}

/// Called with every [`MemEvent`]; installed with [`CudaProvider::with_alloc_hook`].
pub type AllocHook = Arc<dyn Fn(&MemEvent) + Send + Sync>;

/// The driver's memory calls, made in one device's primary context. The real one makes the
/// context current on the calling thread before each call; a test stands in a recorder. `Send`
/// and `Sync`, because [`HostRanges`], which every device's dispatcher thread reaches, keeps the
/// calls of the device that registered a range until the range is unregistered.
pub trait MemoryOps: Send + Sync {
    /// `cuMemAlloc_v2`: device memory of `bytes`, aligned to at least [`DEVICE_ALIGN`].
    fn device_alloc(&self, bytes: u64) -> Result<u64, Fault>;
    /// `cuMemFree_v2` of an address `device_alloc` returned.
    fn device_free(&self, addr: u64) -> Result<(), Fault>;
    /// `cuMemHostAlloc`, portable and device-mapped, zeroed: page-aligned host memory. Returns
    /// the host address.
    fn host_alloc(&self, bytes: u64) -> Result<u64, Fault>;
    /// `cuMemFreeHost` of an address `host_alloc` returned.
    fn host_free(&self, host: u64) -> Result<(), Fault>;
    /// `cuMemHostRegister_v2`, portable and device-mapped.
    fn host_register(&self, host: u64, bytes: u64) -> Result<(), Fault>;
    /// `cuMemHostUnregister` of an address `host_register` took.
    fn host_unregister(&self, host: u64) -> Result<(), Fault>;
    /// `cuMemHostGetDevicePointer_v2`: the device address of mapped host memory.
    fn device_pointer(&self, host: u64) -> Result<u64, Fault>;
}

/// A driver whose devices can allocate: [`Driver`] plus the memory calls, bound to a retained
/// primary context at `open`.
pub trait MemoryDriver: Driver {
    /// The memory calls in `ctx`'s device. The result must keep the context alive itself.
    fn memory(&self, ctx: &Self::Context) -> Result<Arc<dyn MemoryOps>, Fault>;
}

/// The memory entry points, resolved once per device at `open`.
#[derive(Clone, Copy)]
struct MemCalls {
    alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> Status,
    free: unsafe extern "C" fn(CUdeviceptr) -> Status,
    host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> Status,
    free_host: unsafe extern "C" fn(*mut c_void) -> Status,
    register: unsafe extern "C" fn(*mut c_void, usize, c_uint) -> Status,
    unregister: unsafe extern "C" fn(*mut c_void) -> Status,
    device_pointer: unsafe extern "C" fn(*mut CUdeviceptr, *mut c_void, c_uint) -> Status,
}

/// [`MemoryOps`] over the installed driver. It holds its own retain of the primary context, so
/// the context outlives every allocation it frees.
struct SystemMemory {
    ctx: PrimaryContext,
    calls: MemCalls,
}

// SAFETY: the driver API is thread-safe, and a `CUcontext` is a handle any thread may make
// current; every call here makes the context current on the calling thread first, and the
// primary context stays retained while `SystemMemory` lives.
unsafe impl Send for SystemMemory {}
// SAFETY: as for `Send`: `&SystemMemory` only reads the handle and the entry points.
unsafe impl Sync for SystemMemory {}

impl MemoryDriver for SystemDriver {
    fn memory(&self, ctx: &PrimaryContext) -> Result<Arc<dyn MemoryOps>, Fault> {
        let api = ctx.api();
        // SAFETY: each type is the entry point's C signature in the CUDA 13 driver API
        // (`cuda.h`), with `CUresult` as `Status` and `CUdeviceptr` as `u64`.
        let calls = unsafe {
            MemCalls {
                alloc: api.symbol("cuMemAlloc_v2")?,
                free: api.symbol("cuMemFree_v2")?,
                host_alloc: api.symbol("cuMemHostAlloc")?,
                free_host: api.symbol("cuMemFreeHost")?,
                register: api.symbol("cuMemHostRegister_v2")?,
                unregister: api.symbol("cuMemHostUnregister")?,
                device_pointer: api.symbol("cuMemHostGetDevicePointer_v2")?,
            }
        };
        Ok(Arc::new(SystemMemory {
            ctx: ctx.share()?,
            calls,
        }))
    }
}

/// A byte count as the driver's `size_t`.
fn size(bytes: u64) -> Result<usize, Fault> {
    usize::try_from(bytes).map_err(|_| Fault {
        call: "size_t",
        error: format!("{bytes} bytes do not fit this host's address space"),
        no_device: false,
    })
}

impl SystemMemory {
    fn check(&self, call: &'static str, status: Status) -> Result<(), Fault> {
        self.ctx.api().check(call, status)
    }
}

impl MemoryOps for SystemMemory {
    fn device_alloc(&self, bytes: u64) -> Result<u64, Fault> {
        let len = size(bytes)?;
        self.ctx.make_current()?;
        let mut addr: CUdeviceptr = 0;
        // SAFETY: the device's context is current on this thread; the call writes one address
        // through a pointer to a live local.
        let s = unsafe { (self.calls.alloc)(&raw mut addr, len) };
        self.check("cuMemAlloc_v2", s).map(|()| addr)
    }

    fn device_free(&self, addr: u64) -> Result<(), Fault> {
        self.ctx.make_current()?;
        // SAFETY: `addr` came from `cuMemAlloc_v2` in this context, and `DeviceMemory` frees
        // each address once.
        let s = unsafe { (self.calls.free)(addr) };
        self.check("cuMemFree_v2", s)
    }

    fn host_alloc(&self, bytes: u64) -> Result<u64, Fault> {
        let len = size(bytes)?;
        self.ctx.make_current()?;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: the device's context is current; the call writes one pointer through a
        // pointer to a live local.
        let s = unsafe {
            (self.calls.host_alloc)(
                &raw mut host,
                len,
                CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP,
            )
        };
        self.check("cuMemHostAlloc", s)?;
        // SAFETY: the driver returned `len` bytes of writable host memory at `host`, which
        // nothing else references yet. Zeroed, so every doorbell flag starts at 0.
        unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, len) };
        Ok(host as u64)
    }

    fn host_free(&self, host: u64) -> Result<(), Fault> {
        self.ctx.make_current()?;
        // SAFETY: `host` came from `cuMemHostAlloc`, and `DeviceMemory` frees each address once.
        let s = unsafe { (self.calls.free_host)(host as *mut c_void) };
        self.check("cuMemFreeHost", s)
    }

    fn host_register(&self, host: u64, bytes: u64) -> Result<(), Fault> {
        let len = size(bytes)?;
        self.ctx.make_current()?;
        let flags = CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP;
        // SAFETY: the tier manager owns `bytes` of host memory at `host` for at least as long
        // as the registration lives (RFC-0020 §6); the driver only pins and maps the pages, and
        // fails the call for a range it cannot pin.
        let s = unsafe { (self.calls.register)(host as *mut c_void, len, flags) };
        self.check("cuMemHostRegister_v2", s)
    }

    fn host_unregister(&self, host: u64) -> Result<(), Fault> {
        self.ctx.make_current()?;
        // SAFETY: `host` is the start of a range `cuMemHostRegister_v2` registered, and
        // `DeviceMemory` unregisters each range once.
        let s = unsafe { (self.calls.unregister)(host as *mut c_void) };
        self.check("cuMemHostUnregister", s)
    }

    fn device_pointer(&self, host: u64) -> Result<u64, Fault> {
        self.ctx.make_current()?;
        let mut addr: CUdeviceptr = 0;
        // SAFETY: `host` is device-mapped host memory of this context; flags must be 0; the
        // call writes one address through a pointer to a live local.
        let s = unsafe { (self.calls.device_pointer)(&raw mut addr, host as *mut c_void, 0) };
        self.check("cuMemHostGetDevicePointer_v2", s).map(|()| addr)
    }
}

/// What a handle holds, and how it goes back to the driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// A `Device` slab: `cuMemFree_v2` of `base`.
    Device,
    /// `HostMapped` pages: `cuMemFreeHost` of `host`.
    HostPages,
    /// An `import`: `cuMemHostUnregister` of `host`.
    Imported,
    /// A `register_host` unit: `cuMemHostUnregister` of `host`.
    Registered,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    held: Held,
    /// The address the driver returned for a `Device` slab, before alignment padding.
    base: u64,
    /// The device address of the first byte the caller sees; 0 for a registration.
    addr: u64,
    /// The host address of the first byte; 0 for a `Device` slab.
    host: u64,
    /// The bytes the caller asked for.
    bytes: u64,
}

/// Handles are unique across the process, so a buffer of one device names nothing on another.
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

/// The address of a [`BufferSlice`]'s first byte, from [`CudaProvider::resolve`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolved {
    /// The device address.
    pub device: u64,
    /// The host address, for `HostMapped` memory; `None` for device memory.
    pub host: Option<u64>,
}

/// Bytes a device holds for the tier manager, from [`CudaProvider::memory_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemStats {
    /// Live `Device` buffers.
    pub device_buffers: u32,
    /// Bytes of them, as requested.
    pub device_bytes: u64,
    /// Live `HostMapped` allocations and imports.
    pub host_buffers: u32,
    /// Bytes of them.
    pub host_bytes: u64,
    /// Live registrations.
    pub registrations: u32,
    /// Bytes they cover.
    pub registered_bytes: u64,
}

/// A host range registered with the driver, and the devices that hold it.
struct Pinned {
    end: u64,
    users: u32,
    /// The calls of the device that registered it, which unregister it.
    owner: Arc<dyn MemoryOps>,
}

/// Why [`HostRanges::acquire`] failed.
enum Refused {
    /// The range overlaps a registered one without being exactly it.
    Overlap(String),
    /// The driver refused the registration.
    Driver(Fault),
}

/// The host ranges registered with the driver, shared by every device one provider opens: one
/// driver registration per range, counted per device (see the module documentation).
#[derive(Default)]
pub struct HostRanges {
    map: Mutex<BTreeMap<u64, Pinned>>,
}

impl fmt::Debug for HostRanges {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostRanges")
            .field("ranges", &self.lock().len())
            .finish()
    }
}

impl HostRanges {
    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, Pinned>> {
        // A panic while the lock was held left the map consistent: every change to it is a
        // single insert, remove or count update.
        self.map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Takes `start..end` for one more device: registers it with the driver through `ops` if no
    /// device holds it, or counts one more user of exactly that range.
    fn acquire(&self, start: u64, end: u64, ops: &Arc<dyn MemoryOps>) -> Result<(), Refused> {
        let mut map = self.lock();
        if let Some((&s, pinned)) = map.range_mut(..end).next_back()
            && pinned.end > start
        {
            if s == start && pinned.end == end {
                pinned.users += 1;
                return Ok(());
            }
            return Err(Refused::Overlap(format!(
                "host range {start:#x}..{end:#x} overlaps {s:#x}..{:#x}, which another device \
                 registered",
                pinned.end
            )));
        }
        ops.host_register(start, end - start)
            .map_err(Refused::Driver)?;
        map.insert(
            start,
            Pinned {
                end,
                users: 1,
                owner: Arc::clone(ops),
            },
        );
        Ok(())
    }

    /// Gives up one device's hold on the range at `start`; the last release unregisters it.
    fn release(&self, start: u64) -> Result<(), Fault> {
        let mut map = self.lock();
        let Some(pinned) = map.get_mut(&start) else {
            return Ok(());
        };
        pinned.users -= 1;
        if pinned.users > 0 {
            return Ok(());
        }
        map.remove(&start)
            .map_or(Ok(()), |pinned| pinned.owner.host_unregister(start))
    }

    /// How many devices hold the range at `start`; 0 when the driver has it unregistered.
    pub fn users(&self, start: u64) -> u32 {
        self.lock().get(&start).map_or(0, |p| p.users)
    }
}

/// One device's memory: the driver calls and the table of what the tier manager holds.
pub struct DeviceMemory {
    ops: Arc<dyn MemoryOps>,
    host_ranges: Arc<HostRanges>,
    hook: Option<AllocHook>,
    entries: RefCell<BTreeMap<u64, Entry>>,
    /// Registered and imported host ranges: start → (end, handle). They never overlap.
    ranges: RefCell<BTreeMap<u64, (u64, u64)>>,
}

impl fmt::Debug for DeviceMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceMemory")
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

/// `AllocFailed`, `(Kernels, 9)`, class `Capacity` (RFC-0040 §13).
fn alloc_failed(what: String) -> Error {
    Error::new(ErrorClass::Capacity, codes::ALLOC_FAILED, what)
}

/// `RegisterFailed`, `(Kernels, 10)`, class `Capacity` (RFC-0040 §13).
fn register_failed(what: String) -> Error {
    Error::new(ErrorClass::Capacity, codes::REGISTER_FAILED, what)
}

/// A caller's error no RFC-0040 §13 code names, such as a range that overlaps a registration
/// or a handle of another device: class `Internal`, code 0, as the CPU provider reports it.
fn internal(what: String) -> Error {
    Error::new(
        ErrorClass::Internal,
        ErrorCode::new(syntropy_core::Domain::Kernels, 0),
        what,
    )
}

/// `n` rounded up to a multiple of the power of two `align`; `None` on overflow.
fn round_up(n: u64, align: u64) -> Option<u64> {
    Some(n.checked_add(align - 1)? & !(align - 1))
}

/// The bytes to take from the driver for `bytes` at `align`, when the driver's own alignment
/// is `base`: the padding that lets an aligned start be found inside. `None` for a size of
/// zero, an alignment that is not a power of two (0 means 1), or an overflow.
pub fn padded(bytes: u64, align: u64, base: u64) -> Option<u64> {
    let align = align.max(1);
    if bytes == 0 || !align.is_power_of_two() {
        return None;
    }
    bytes.checked_add(align.saturating_sub(base))
}

impl DeviceMemory {
    /// The memory of a device whose driver calls are `ops`.
    pub fn new(ops: Arc<dyn MemoryOps>, hook: Option<AllocHook>, host: Arc<HostRanges>) -> Self {
        Self {
            ops,
            host_ranges: host,
            hook,
            entries: RefCell::new(BTreeMap::new()),
            ranges: RefCell::new(BTreeMap::new()),
        }
    }

    fn event(&self, e: MemEvent) {
        if let Some(hook) = &self.hook {
            hook(&e);
        }
    }

    fn insert(&self, entry: Entry) -> u64 {
        let handle = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
        self.entries.borrow_mut().insert(handle, entry);
        handle
    }

    /// `alloc` (RFC-0040 §5.1).
    pub fn alloc(&self, req: AllocRequest) -> Result<DeviceBuffer, Error> {
        if req.reserve_only {
            // Address reservation and VMM segments are COMP-E4-T13's.
            return Err(unsupported());
        }
        let bytes = req.bytes.0;
        let align = req.align.0.max(1);
        if !align.is_power_of_two() {
            // The tier manager's alignments are powers of two (RFC-0020 §3): a caller's error.
            return Err(internal(format!("alignment {align} is not a power of two")));
        }
        match req.kind {
            MemKind::Device => {
                let take = padded(bytes, align, DEVICE_ALIGN).ok_or_else(|| {
                    alloc_failed(format!(
                        "cannot allocate {bytes} bytes of device memory at {align}-byte alignment"
                    ))
                })?;
                let base = self.ops.device_alloc(take).map_err(|f| {
                    alloc_failed(format!(
                        "cannot allocate {take} bytes of device memory: {} failed: {}",
                        f.call, f.error
                    ))
                })?;
                // The aligned start must leave `bytes` inside the slab: true for the alignment
                // the driver documents, and checked rather than assumed.
                let fits = |a: u64| a - base + bytes <= take;
                let Some(addr) = round_up(base, align).filter(|&a| fits(a)) else {
                    let _ = self.ops.device_free(base);
                    return Err(alloc_failed(format!(
                        "device address {base:#x} cannot be aligned to {align}"
                    )));
                };
                let handle = self.insert(Entry {
                    held: Held::Device,
                    base,
                    addr,
                    host: 0,
                    bytes,
                });
                self.event(MemEvent::Alloc {
                    kind: MemKind::Device,
                    bytes: take,
                    handle,
                });
                Ok(DeviceBuffer {
                    addr,
                    bytes: Bytes(bytes),
                    kind: MemKind::Device,
                    handle,
                })
            }
            MemKind::HostMapped => {
                // Doorbell pages: aligned to at least the host page, which `cuMemHostAlloc`
                // gives; a larger alignment pads, a smaller one is met.
                let page = host_page();
                let take = padded(bytes, align, page)
                    .and_then(|t| round_up(t, page))
                    .ok_or_else(|| {
                        alloc_failed(format!(
                            "cannot allocate {bytes} bytes of host-mapped memory at \
                             {align}-byte alignment"
                        ))
                    })?;
                let fail = |f: Fault| {
                    alloc_failed(format!(
                        "cannot allocate {take} bytes of host-mapped memory: {} failed: {}",
                        f.call, f.error
                    ))
                };
                let base = self.ops.host_alloc(take).map_err(fail)?;
                let mapped = self.ops.device_pointer(base).map_err(fail);
                let fits = |h: u64| h - base + bytes <= take;
                let (host, device) = match (mapped, round_up(base, align).filter(|&h| fits(h))) {
                    (Ok(device), Some(host)) => (host, device + (host - base)),
                    (mapped, _) => {
                        let _ = self.ops.host_free(base);
                        return Err(mapped.err().unwrap_or_else(|| {
                            alloc_failed(format!(
                                "host address {base:#x} cannot be aligned to {align}"
                            ))
                        }));
                    }
                };
                let handle = self.insert(Entry {
                    held: Held::HostPages,
                    base,
                    addr: device,
                    host,
                    bytes,
                });
                self.event(MemEvent::Alloc {
                    kind: MemKind::HostMapped,
                    bytes: take,
                    handle,
                });
                Ok(DeviceBuffer {
                    addr: device,
                    bytes: Bytes(bytes),
                    kind: MemKind::HostMapped,
                    handle,
                })
            }
            // `Unified` on GB10-class devices is COMP-E4-T08's.
            _ => Err(unsupported()),
        }
    }

    /// Checks that `region` is whole host pages and overlaps no range already registered or
    /// imported; the message names the problem.
    fn claimable(&self, region: HostRegion) -> Result<u64, String> {
        let (start, len) = (region.addr, region.len.0);
        let page = host_page();
        if start == 0 || len == 0 || start % page != 0 || len % page != 0 {
            return Err(format!(
                "host range {start:#x}+{len} is not whole {page}-byte pages"
            ));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| format!("host range {start:#x}+{len} overflows"))?;
        let ranges = self.ranges.borrow();
        if let Some((&s, &(e, h))) = ranges.range(..end).next_back()
            && e > start
        {
            return Err(format!(
                "host range {start:#x}..{end:#x} overlaps {s:#x}..{e:#x}, held by handle {h}"
            ));
        }
        Ok(end)
    }

    /// `register_host` (RFC-0040 §5.2): one RFC-0020 §6.1 unit.
    pub fn register_host(&self, region: HostRegion) -> Result<Registration, Error> {
        let end = self.claimable(region).map_err(internal)?;
        self.host_ranges
            .acquire(region.addr, end, &self.ops)
            .map_err(|r| match r {
                Refused::Overlap(m) => internal(m),
                Refused::Driver(f) => register_failed(format!(
                    "cannot register {} bytes at {:#x}: {} failed: {}",
                    region.len.0, region.addr, f.call, f.error
                )),
            })?;
        let handle = self.insert(Entry {
            held: Held::Registered,
            base: region.addr,
            addr: 0,
            host: region.addr,
            bytes: region.len.0,
        });
        self.ranges.borrow_mut().insert(region.addr, (end, handle));
        self.event(MemEvent::Register { region, handle });
        Ok(Registration { region, handle })
    }

    /// `import` (RFC-0040 §5.2): the tier manager's memory as a device-mapped buffer.
    pub fn import(&self, region: HostRegion) -> Result<DeviceBuffer, Error> {
        let end = self.claimable(region).map_err(internal)?;
        let fail = |f: Fault| {
            alloc_failed(format!(
                "cannot import {} bytes at {:#x}: {} failed: {}",
                region.len.0, region.addr, f.call, f.error
            ))
        };
        self.host_ranges
            .acquire(region.addr, end, &self.ops)
            .map_err(|r| match r {
                Refused::Overlap(m) => internal(m),
                Refused::Driver(f) => fail(f),
            })?;
        // The device's own address of the range, read in this device's context.
        let device = match self.ops.device_pointer(region.addr) {
            Ok(d) => d,
            Err(f) => {
                let _ = self.host_ranges.release(region.addr);
                return Err(fail(f));
            }
        };
        let handle = self.insert(Entry {
            held: Held::Imported,
            base: region.addr,
            addr: device,
            host: region.addr,
            bytes: region.len.0,
        });
        self.ranges.borrow_mut().insert(region.addr, (end, handle));
        self.event(MemEvent::Import { region, handle });
        Ok(DeviceBuffer {
            addr: device,
            bytes: region.len,
            kind: MemKind::HostMapped,
            handle,
        })
    }

    /// Returns `entry` to the driver. A failure cannot be reported, since `free` returns
    /// nothing; the handle is gone either way, so it is never released twice.
    fn release(&self, handle: u64, entry: Entry) {
        let _ = match entry.held {
            Held::Device => self.ops.device_free(entry.base),
            Held::HostPages => self.ops.host_free(entry.base),
            Held::Imported | Held::Registered => {
                self.ranges.borrow_mut().remove(&entry.host);
                self.host_ranges.release(entry.host)
            }
        };
        self.event(MemEvent::Release { handle });
    }

    /// `free`: releases a buffer from `alloc` or `import`. A handle this device does not hold,
    /// or a registration's, is ignored.
    pub fn free(&self, buf: &DeviceBuffer) {
        let removed = {
            let mut entries = self.entries.borrow_mut();
            match entries.get(&buf.handle) {
                Some(e) if e.held != Held::Registered && e.addr == buf.addr => {
                    entries.remove(&buf.handle)
                }
                _ => None,
            }
        };
        if let Some(entry) = removed {
            self.release(buf.handle, entry);
        }
    }

    /// Releases a registration from `register_host`.
    pub fn unregister(&self, reg: &Registration) -> Result<(), Error> {
        let removed = {
            let mut entries = self.entries.borrow_mut();
            match entries.get(&reg.handle) {
                Some(e) if e.held == Held::Registered && e.host == reg.region.addr => {
                    entries.remove(&reg.handle)
                }
                _ => None,
            }
        };
        let entry = removed.ok_or_else(|| {
            internal(format!(
                "handle {} is not a registration of this device",
                reg.handle
            ))
        })?;
        self.ranges.borrow_mut().remove(&entry.host);
        let result = self.host_ranges.release(entry.host).map_err(|f| {
            register_failed(format!(
                "cannot unregister {:#x}: {} failed: {}",
                entry.host, f.call, f.error
            ))
        });
        self.event(MemEvent::Release { handle: reg.handle });
        result
    }

    /// The registration or import that holds all of `len` bytes at host address `addr`, the
    /// test of a DMA source; `None` when no single one does, which includes a range that
    /// crosses from one registration into the next.
    pub fn registration_for(&self, addr: u64, len: u64) -> Option<u64> {
        let end = addr.checked_add(len)?;
        let ranges = self.ranges.borrow();
        let (_, &(e, handle)) = ranges.range(..=addr).next_back()?;
        (end <= e).then_some(handle)
    }

    /// The addresses of `slice`'s first byte; `None` for a handle this device does not hold, a
    /// registration, or a range past the buffer's end.
    pub fn resolve(&self, slice: BufferSlice) -> Option<Resolved> {
        let entries = self.entries.borrow();
        let e = entries.get(&slice.buffer)?;
        let end = slice.offset.0.checked_add(slice.bytes.0)?;
        if e.held == Held::Registered || end > e.bytes {
            return None;
        }
        Some(Resolved {
            device: e.addr + slice.offset.0,
            host: (e.held != Held::Device).then(|| e.host + slice.offset.0),
        })
    }

    /// What the device holds now.
    pub fn stats(&self) -> MemStats {
        let mut s = MemStats::default();
        for e in self.entries.borrow().values() {
            match e.held {
                Held::Device => {
                    s.device_buffers += 1;
                    s.device_bytes += e.bytes;
                }
                Held::HostPages | Held::Imported => {
                    s.host_buffers += 1;
                    s.host_bytes += e.bytes;
                }
                Held::Registered => {
                    s.registrations += 1;
                    s.registered_bytes += e.bytes;
                }
            }
        }
        s
    }
}

impl Drop for DeviceMemory {
    /// Releases whatever the tier manager has not, so closing a device leaks nothing. Runs
    /// before the device's primary context is released (`CudaDevice`'s field order).
    fn drop(&mut self) {
        let entries = std::mem::take(self.entries.get_mut());
        for (handle, entry) in entries {
            self.release(handle, entry);
        }
    }
}

impl<D: Driver> CudaProvider<D>
where
    D::Context: 'static,
{
    /// The memory of the device `ctx` opened.
    pub(crate) fn memory<'a>(&self, ctx: &'a DeviceCtx) -> Result<&'a DeviceMemory, Error> {
        ctx.downcast_ref::<CudaDevice<D::Context>>()
            .map(|d| &d.memory)
            .ok_or_else(|| internal("the context is not a CUDA device's".to_owned()))
    }

    /// Installs `hook`, which sees every allocation, registration, import and release the
    /// devices opened afterwards make (RFC-0001 §9.1's rule that only the tier manager
    /// allocates, checked from outside).
    #[must_use]
    pub fn with_alloc_hook(mut self, hook: AllocHook) -> Self {
        self.hook = Some(hook);
        self
    }

    /// Releases a registration from `register_host`, as a shrinking arena does with whole
    /// units (RFC-0040 §5.2). The trait has no such method; COMP-E4-T03's `HostRegistrar`
    /// carries `unregister` for registrations made off the dispatcher.
    ///
    /// # Errors
    ///
    /// `Internal` for a registration this device does not hold, which includes one already
    /// released; `RegisterFailed` when the driver refuses.
    pub fn unregister_host(&self, ctx: &DeviceCtx, reg: Registration) -> Result<(), Error> {
        self.memory(ctx)?.unregister(&reg)
    }

    /// The registration that holds all of `len` bytes at host address `addr`; see
    /// [`DeviceMemory::registration_for`].
    pub fn registration_for(&self, ctx: &DeviceCtx, addr: u64, len: u64) -> Option<u64> {
        self.memory(ctx).ok()?.registration_for(addr, len)
    }

    /// The addresses of a slice of a buffer this device holds, for `copy`.
    pub fn resolve(&self, ctx: &DeviceCtx, slice: BufferSlice) -> Option<Resolved> {
        self.memory(ctx).ok()?.resolve(slice)
    }

    /// What the device holds for the tier manager now.
    ///
    /// # Errors
    ///
    /// `Internal` when `ctx` is not a CUDA device's.
    pub fn memory_stats(&self, ctx: &DeviceCtx) -> Result<MemStats, Error> {
        Ok(self.memory(ctx)?.stats())
    }
}

#[cfg(test)]
pub(crate) mod tests;
