// SPDX-License-Identifier: MIT OR Apache-2.0

//! NVIDIA GPUs through NVML (RFC-0010 §2.2's "NVIDIA GPU" row): UUID, PCI v3, memory v2, BAR1,
//! the current and maximum PCIe link, P2P status, NVLink state, power and temperature.
//!
//! The library is `libnvidia-ml.so.1` on Linux and WSL2 and `nvml.dll` on Windows, opened by
//! name through a [`VendorLoader`]. What it gives:
//!
//! - **Absent.** No library, or a library whose `nvmlInit_v2` says no driver is loaded: no GPU,
//!   and no error (RFC-0010 §2.1).
//! - **Unusable.** A library without one of the functions every NVML since the 2012 `_v2` API
//!   has: no GPU, with the reason kept for `syntropy doctor`.
//! - **A driver older than [`MIN_DRIVER_MAJOR`],** or one whose version NVML does not report.
//!   Every GPU is listed with
//!   `DeviceProblem::DriverTooOld`, so `doctor` can say why it is idle (RFC-0010 §2.1). A
//!   versioned function such an old library lacks, as `nvmlDeviceGetPciInfo_v3` or
//!   `nvmlDeviceGetMemoryInfo_v2`, leaves its fields `None`.
//! - Otherwise every GPU, each with its PCI address, UUID, VRAM total and budget.
//!
//! A getter that fails for one GPU, as `NOT_SUPPORTED` does for NVLink on a GPU without it,
//! leaves that field `None`; the GPU is still listed. A GPU whose handle NVML refuses, as for a
//! GPU this user may not see, is skipped, and its index and code are kept.
//!
//! The PCI slot's NUMA domain comes from sysfs on Linux ([`sysfs_numa`]), not from NVML.

use std::ffi::{c_char, c_uint, c_void};

use syntropy_core::resource::{
    DeviceInfoExt, DeviceKind, DeviceProblem, DriverInfo, PciAddress, PciInfo, PcieLink, PowerInfo,
};
use syntropy_core::units::Bytes;

use super::{GpuRecord, PeerAccess, VendorLibrary, VendorLoader};

/// The NVML library's file name on this OS.
pub const LIBRARY: &str = if cfg!(windows) {
    "nvml.dll"
} else {
    "libnvidia-ml.so.1"
};

/// The oldest NVIDIA driver major version Syntropy runs on: the CUDA driver floor of RFC-0040's
/// `DriverTooOld` (code 5) and ADR-0020.
pub const MIN_DRIVER_MAJOR: u32 = 580;

/// The most NVLink links NVML numbers (`NVML_NVLINK_MAX_LINKS`).
const MAX_NVLINKS: c_uint = 18;

/// NVML's return codes this module tells apart.
pub mod code {
    /// `NVML_SUCCESS`.
    pub const SUCCESS: i32 = 0;
    /// `NVML_ERROR_NOT_SUPPORTED`.
    pub const NOT_SUPPORTED: i32 = 3;
    /// `NVML_ERROR_DRIVER_NOT_LOADED`.
    pub const DRIVER_NOT_LOADED: i32 = 9;
}

type Device = *mut c_void;
type Ret = i32;

/// `nvmlPciInfo_t`, filled by `nvmlDeviceGetPciInfo_v3`.
#[repr(C)]
#[derive(Clone, Copy)]
struct RawPci {
    bus_id_legacy: [c_char; 16],
    domain: c_uint,
    bus: c_uint,
    device: c_uint,
    /// Device ID in the upper 16 bits, vendor ID in the lower.
    pci_device_id: c_uint,
    pci_sub_system_id: c_uint,
    bus_id: [c_char; 32],
}

/// `nvmlMemory_v2_t`, filled by `nvmlDeviceGetMemoryInfo_v2`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawMemory {
    version: c_uint,
    total: u64,
    reserved: u64,
    free: u64,
    used: u64,
}

/// `nvmlMemory_v2`: the struct's size with version 2 in the top byte.
const MEMORY_V2: c_uint = size_of::<RawMemory>() as c_uint | (2 << 24);

/// `nvmlBAR1Memory_t`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawBar1 {
    total: u64,
    free: u64,
    used: u64,
}

/// What NVML gave on this host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// No NVML library, or no NVIDIA driver loaded: no GPU, not an error.
    Absent,
    /// The library lacks a function every supported NVML has, or failed to start: no GPU.
    Unusable {
        /// Why, for `syntropy doctor`.
        reason: String,
    },
    /// The GPUs were read.
    Read {
        /// The driver version NVML reported.
        driver: String,
        /// GPUs NVML counted but refused a handle for: (index, NVML code).
        skipped: Vec<(u32, i32)>,
    },
}

/// NVML's result: its status and the GPUs it listed.
#[derive(Clone, Debug, PartialEq)]
pub struct Inventory {
    /// What happened.
    pub status: Status,
    /// One record per GPU, in NVML's index order.
    pub gpus: Vec<GpuRecord>,
}

impl Inventory {
    fn none(status: Status) -> Inventory {
        Inventory {
            status,
            gpus: Vec::new(),
        }
    }
}

/// The NVML functions this module calls, resolved once.
struct Api {
    init: unsafe extern "C" fn() -> Ret,
    shutdown: unsafe extern "C" fn() -> Ret,
    driver_version: unsafe extern "C" fn(*mut c_char, c_uint) -> Ret,
    count: unsafe extern "C" fn(*mut c_uint) -> Ret,
    handle: unsafe extern "C" fn(c_uint, *mut Device) -> Ret,
    uuid: unsafe extern "C" fn(Device, *mut c_char, c_uint) -> Ret,
    name: unsafe extern "C" fn(Device, *mut c_char, c_uint) -> Ret,
    pci: Option<unsafe extern "C" fn(Device, *mut RawPci) -> Ret>,
    memory: Option<unsafe extern "C" fn(Device, *mut RawMemory) -> Ret>,
    bar1: Option<unsafe extern "C" fn(Device, *mut RawBar1) -> Ret>,
    link_gen_cur: Option<unsafe extern "C" fn(Device, *mut c_uint) -> Ret>,
    link_gen_max: Option<unsafe extern "C" fn(Device, *mut c_uint) -> Ret>,
    link_width_cur: Option<unsafe extern "C" fn(Device, *mut c_uint) -> Ret>,
    link_width_max: Option<unsafe extern "C" fn(Device, *mut c_uint) -> Ret>,
    p2p: Option<unsafe extern "C" fn(Device, Device, c_uint, *mut c_uint) -> Ret>,
    nvlink: Option<unsafe extern "C" fn(Device, c_uint, *mut c_uint) -> Ret>,
    power_limit: Option<unsafe extern "C" fn(Device, *mut c_uint) -> Ret>,
    power_cap: Option<unsafe extern "C" fn(Device, *mut c_uint) -> Ret>,
    temperature: Option<unsafe extern "C" fn(Device, c_uint, *mut c_uint) -> Ret>,
}

/// Resolves `name` as the function type `F`.
///
/// # Safety
///
/// `F` must be the C signature NVML documents for `name`, which [`VendorLibrary`]'s contract
/// then makes the symbol's.
unsafe fn resolve<F: Copy>(lib: &dyn VendorLibrary, name: &str) -> Option<F> {
    const { assert!(size_of::<F>() == size_of::<*mut c_void>()) };
    let address = lib.symbol(name)?;
    if address.is_null() {
        return None;
    }
    // SAFETY: `F` is a function pointer type of a pointer's size (the caller's contract), and
    // the address is that function's.
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&address) })
}

impl Api {
    /// The functions, or the name of the first required one the library lacks.
    fn load(lib: &dyn VendorLibrary) -> Result<Api, String> {
        macro_rules! required {
            ($name:literal) => {
                // SAFETY: the field's type is NVML's documented signature for `$name`.
                unsafe { resolve(lib, $name) }.ok_or(concat!("no ", $name))?
            };
        }
        macro_rules! optional {
            ($name:literal) => {
                // SAFETY: as above.
                unsafe { resolve(lib, $name) }
            };
        }
        Ok(Api {
            init: required!("nvmlInit_v2"),
            shutdown: required!("nvmlShutdown"),
            driver_version: required!("nvmlSystemGetDriverVersion"),
            count: required!("nvmlDeviceGetCount_v2"),
            handle: required!("nvmlDeviceGetHandleByIndex_v2"),
            uuid: required!("nvmlDeviceGetUUID"),
            name: required!("nvmlDeviceGetName"),
            pci: optional!("nvmlDeviceGetPciInfo_v3"),
            memory: optional!("nvmlDeviceGetMemoryInfo_v2"),
            bar1: optional!("nvmlDeviceGetBAR1MemoryInfo"),
            link_gen_cur: optional!("nvmlDeviceGetCurrPcieLinkGeneration"),
            link_gen_max: optional!("nvmlDeviceGetMaxPcieLinkGeneration"),
            link_width_cur: optional!("nvmlDeviceGetCurrPcieLinkWidth"),
            link_width_max: optional!("nvmlDeviceGetMaxPcieLinkWidth"),
            p2p: optional!("nvmlDeviceGetP2PStatus"),
            nvlink: optional!("nvmlDeviceGetNvLinkState"),
            power_limit: optional!("nvmlDeviceGetPowerManagementLimit"),
            power_cap: optional!("nvmlDeviceGetEnforcedPowerLimit"),
            temperature: optional!("nvmlDeviceGetTemperature"),
        })
    }
}

/// A NUL-terminated string NVML wrote into `buf`.
fn text(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// NVML's UUID text, `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (`MIG-` for a MIG instance), as
/// its 16 bytes.
#[must_use]
pub fn parse_uuid(text: &str) -> Option<[u8; 16]> {
    let hex = text
        .strip_prefix("GPU-")
        .or_else(|| text.strip_prefix("MIG-"))?;
    let dashes: Vec<usize> = hex.match_indices('-').map(|(i, _)| i).collect();
    if hex.len() != 36 || dashes != [8, 13, 18, 23] {
        return None;
    }
    let digits: Vec<u8> = hex.bytes().filter(|b| *b != b'-').collect();
    let mut out = [0u8; 16];
    for (i, pair) in digits.as_chunks::<2>().0.iter().enumerate() {
        let pair = std::str::from_utf8(pair).ok()?;
        out[i] = u8::from_str_radix(pair, 16).ok()?;
    }
    Some(out)
}

/// The major version of a driver version such as `595.91.07`.
#[must_use]
pub fn driver_major(version: &str) -> Option<u32> {
    version.split('.').next()?.trim().parse().ok()
}

/// One call that fills `*out`, as `Some(value)` on `NVML_SUCCESS`.
fn get<T: Default>(f: impl FnOnce(*mut T) -> Ret) -> Option<T> {
    let mut out = T::default();
    (f(&raw mut out) == code::SUCCESS).then_some(out)
}

/// Lists the NVIDIA GPUs through the NVML library `loader` opens, with `numa` giving a PCI
/// slot's NUMA domain. Never fails (module docs).
pub fn enumerate(loader: &dyn VendorLoader, numa: &dyn Fn(PciAddress) -> Option<u16>) -> Inventory {
    let Some(lib) = loader.open(LIBRARY) else {
        return Inventory::none(Status::Absent);
    };
    let api = match Api::load(lib.as_ref()) {
        Ok(api) => api,
        Err(reason) => return Inventory::none(Status::Unusable { reason }),
    };
    // SAFETY: `nvmlInit_v2` takes no arguments.
    match unsafe { (api.init)() } {
        code::SUCCESS => {}
        code::DRIVER_NOT_LOADED => return Inventory::none(Status::Absent),
        other => {
            return Inventory::none(Status::Unusable {
                reason: format!("nvmlInit_v2 returned {other}"),
            });
        }
    }
    let inventory = read(&api, numa);
    // SAFETY: NVML was initialised above; every handle is dropped by now.
    unsafe { (api.shutdown)() };
    inventory
}

/// Reads every GPU of an initialised NVML.
fn read(api: &Api, numa: &dyn Fn(PciAddress) -> Option<u16>) -> Inventory {
    let mut version = [0 as c_char; 80];
    // SAFETY: the buffer and its length match.
    let driver = if unsafe { (api.driver_version)(version.as_mut_ptr(), 80) } == code::SUCCESS {
        text(&version)
    } else {
        String::new()
    };
    // SAFETY: `count` writes one `c_uint`.
    let Some(count) = get(|out| unsafe { (api.count)(out) }) else {
        return Inventory::none(Status::Unusable {
            reason: "nvmlDeviceGetCount_v2 failed".to_owned(),
        });
    };
    let too_old = driver_major(&driver).is_none_or(|major| major < MIN_DRIVER_MAJOR);
    let mut handles = Vec::new();
    let mut skipped = Vec::new();
    for index in 0..count {
        let mut device: Device = std::ptr::null_mut();
        // SAFETY: `handle` writes one device handle.
        match unsafe { (api.handle)(index, &raw mut device) } {
            code::SUCCESS => handles.push(device),
            other => skipped.push((index, other)),
        }
    }
    let mut gpus: Vec<GpuRecord> = handles
        .iter()
        .map(|&device| gpu(api, device, &driver, too_old, numa))
        .collect();
    if let Some(p2p) = api.p2p {
        for (i, &a) in handles.iter().enumerate() {
            for (j, &b) in handles.iter().enumerate() {
                if i == j {
                    continue;
                }
                // `nvmlGpuP2PCapsIndex_t`: READ 0, WRITE 1, NVLINK 2, ATOMICS 3. Status 0 is OK.
                let cap = |index: c_uint| {
                    // SAFETY: both handles are live; `p2p` writes one status.
                    get(|out| unsafe { p2p(a, b, index, out) }) == Some(0)
                };
                let access = PeerAccess {
                    peer: j,
                    read: cap(0),
                    write: cap(1),
                    nvlink: cap(2),
                    atomics: cap(3),
                };
                if access.read || access.write || access.nvlink || access.atomics {
                    gpus[i].peers.push(access);
                }
            }
        }
    }
    Inventory {
        status: Status::Read { driver, skipped },
        gpus,
    }
}

/// One GPU's record.
fn gpu(
    api: &Api,
    device: Device,
    driver: &str,
    too_old: bool,
    numa: &dyn Fn(PciAddress) -> Option<u16>,
) -> GpuRecord {
    let mut buf = [0 as c_char; 96];
    // SAFETY: the buffer and its length match.
    let uuid = (unsafe { (api.uuid)(device, buf.as_mut_ptr(), 96) } == code::SUCCESS)
        .then(|| parse_uuid(&text(&buf)))
        .flatten();
    buf.fill(0);
    // SAFETY: as above.
    let name = if unsafe { (api.name)(device, buf.as_mut_ptr(), 96) } == code::SUCCESS {
        text(&buf)
    } else {
        String::new()
    };
    let uint = |f: Option<unsafe extern "C" fn(Device, *mut c_uint) -> Ret>| {
        // SAFETY: `f` writes one `c_uint` for a live handle.
        f.and_then(|f| get(|out| unsafe { f(device, out) }))
    };
    let link = |r#gen, width| match (uint(r#gen), uint(width)) {
        (Some(r#gen), Some(width)) => PcieLink {
            r#gen: r#gen.try_into().unwrap_or(u8::MAX),
            width: width.try_into().unwrap_or(u8::MAX),
        },
        _ => PcieLink::default(),
    };
    let raw_pci = api.pci.and_then(|f| {
        let mut out = RawPci {
            bus_id_legacy: [0; 16],
            domain: 0,
            bus: 0,
            device: 0,
            pci_device_id: 0,
            pci_sub_system_id: 0,
            bus_id: [0; 32],
        };
        // SAFETY: `f` fills one `nvmlPciInfo_t`.
        (unsafe { f(device, &raw mut out) } == code::SUCCESS).then_some(out)
    });
    let memory = api.memory.and_then(|f| {
        let mut out = RawMemory {
            version: MEMORY_V2,
            ..RawMemory::default()
        };
        // SAFETY: `f` fills one `nvmlMemory_v2_t` whose `version` is set.
        (unsafe { f(device, &raw mut out) } == code::SUCCESS).then_some(out)
    });
    // SAFETY: `f` fills one `nvmlBAR1Memory_t`.
    let bar1 = api
        .bar1
        .and_then(|f| get(|out| unsafe { f(device, out) }))
        .map(|b: RawBar1| b.total);
    let pci = raw_pci.map(|raw| {
        let bdf = PciAddress {
            domain: raw.domain,
            bus: raw.bus.try_into().unwrap_or(u8::MAX),
            device: raw.device.try_into().unwrap_or(u8::MAX),
            function: 0,
        };
        PciInfo {
            bdf,
            vendor: (raw.pci_device_id & 0xFFFF) as u16,
            device: (raw.pci_device_id >> 16) as u16,
            subsystem: raw.pci_sub_system_id,
            link_cur: link(api.link_gen_cur, api.link_width_cur),
            link_max: link(api.link_gen_max, api.link_width_max),
            resizable_bar: bar1.zip(memory).map(|(bar1, memory)| bar1 >= memory.total),
            numa: numa(bdf),
            ..PciInfo::default()
        }
    });
    let nvlinks = api.nvlink.map_or(0, |f| {
        (0..MAX_NVLINKS)
            // SAFETY: `f` writes one `nvmlEnableState_t` for a live handle; 1 is enabled.
            .filter(|&link| get(|out| unsafe { f(device, link, out) }) == Some(1))
            .count() as u8
    });
    let watts = |mw: Option<c_uint>| mw.map(|mw| mw as f32 / 1000.0);
    let power = PowerInfo {
        limit_w: watts(uint(api.power_limit)),
        cap_w: watts(uint(api.power_cap)),
        ..PowerInfo::default()
    };
    let mut ext = DeviceInfoExt {
        driver: DriverInfo {
            driver: driver.to_owned(),
            ..DriverInfo::default()
        },
        pci,
        uuid,
        power: Some(power),
        ..DeviceInfoExt::default()
    };
    if too_old {
        ext.problems.push(DeviceProblem::DriverTooOld {
            have: driver.to_owned(),
            need: MIN_DRIVER_MAJOR.to_string(),
        });
    }
    GpuRecord {
        kind: DeviceKind::DiscreteGpu,
        name,
        unified: false,
        numa: pci.and_then(|pci| pci.numa),
        vram_total: memory.map(|m| Bytes(m.total)),
        vram_budget: memory.map(|m| Bytes(m.free)),
        bar1_total: bar1.map(Bytes),
        // `NVML_TEMPERATURE_GPU` is sensor 0.
        // SAFETY: `f` writes one `c_uint` for a live handle.
        temperature_c: api
            .temperature
            .and_then(|f| get(|out| unsafe { f(device, 0, out) })),
        nvlinks,
        peers: Vec::new(),
        ext,
    }
}

/// A PCI slot's NUMA domain from `/sys/bus/pci/devices/<bdf>/numa_node` on Linux; `None`
/// elsewhere, and where the file says -1 (no affinity) or is missing.
#[must_use]
pub fn sysfs_numa(bdf: PciAddress) -> Option<u16> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let path = format!(
        "/sys/bus/pci/devices/{:04x}:{:02x}:{:02x}.{:x}/numa_node",
        bdf.domain, bdf.bus, bdf.device, bdf.function
    );
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}
