// SPDX-License-Identifier: MIT OR Apache-2.0

//! PLC-E1-T04: NVML through a fake `dlopen`. The shim's symbols are Rust functions with NVML's C
//! signatures, answering from a per-thread fake host, so present, absent, old-driver and broken
//! libraries run the same code a real `libnvidia-ml.so.1` does. The last test runs the real
//! system loader on whatever host runs it.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_uint, c_void};

use syntropy_core::resource::{DeviceKind, DeviceProblem, PciAddress, PcieLink};
use syntropy_core::units::Bytes;
use syntropy_kernels_loader::gpu::nvml::{
    self, Inventory, LIBRARY, MIN_DRIVER_MAJOR, Status, code, driver_major, parse_uuid,
};
use syntropy_kernels_loader::gpu::{PeerAccess, VendorLibrary, VendorLoader};

const GIB: u64 = 1 << 30;
const NOT_PERMITTED: i32 = 4;

/// One fake GPU.
#[derive(Clone)]
struct FakeGpu {
    uuid: &'static str,
    name: &'static str,
    domain: u32,
    bus: u32,
    /// Device ID << 16 | vendor ID.
    pci_device_id: u32,
    subsystem: u32,
    total: u64,
    free: u64,
    /// Driver-reserved VRAM, neither free nor used by processes (`nvmlMemory_v2_t::reserved`).
    reserved: u64,
    bar1: u64,
    link_cur: (u32, u32),
    link_max: (u32, u32),
    /// Links reported enabled; `None` answers `NOT_SUPPORTED`.
    nvlinks: Option<Vec<u32>>,
    power_limit_mw: u32,
    power_cap_mw: u32,
    temperature: Option<u32>,
    /// `nvmlDeviceGetHandleByIndex_v2` refuses this GPU with the code.
    refused: Option<i32>,
}

/// The fake host the shim answers from.
#[derive(Clone)]
struct FakeHost {
    init: i32,
    driver: &'static str,
    gpus: Vec<FakeGpu>,
    /// (a, b) → status per caps index; missing pairs answer status 1 (not supported).
    p2p: BTreeMap<(usize, usize), [u32; 4]>,
    /// The `version` field the last `nvmlDeviceGetMemoryInfo_v2` call was given.
    memory_version: Option<u32>,
    shut_down: bool,
}

thread_local! {
    static HOST: RefCell<Option<FakeHost>> = const { RefCell::new(None) };
}

fn with<T>(f: impl FnOnce(&mut FakeHost) -> T) -> T {
    HOST.with(|h| f(h.borrow_mut().as_mut().expect("a fake host is installed")))
}

fn index(device: *mut c_void) -> usize {
    device as usize - 1
}

fn gpu_of(device: *mut c_void) -> FakeGpu {
    with(|h| h.gpus[index(device)].clone())
}

/// Writes `text` NUL-terminated into `buf` of `len`.
///
/// # Safety
///
/// `buf` is writable for `len` bytes.
unsafe fn put(text: &str, buf: *mut c_char, len: c_uint) -> i32 {
    if text.len() + 1 > len as usize {
        return 7; // NVML_ERROR_INSUFFICIENT_SIZE
    }
    for (i, b) in text.bytes().chain([0]).enumerate() {
        // SAFETY: `i < len`, checked above.
        unsafe { *buf.add(i) = b as c_char };
    }
    code::SUCCESS
}

#[repr(C)]
struct RawPci {
    bus_id_legacy: [c_char; 16],
    domain: c_uint,
    bus: c_uint,
    device: c_uint,
    pci_device_id: c_uint,
    pci_sub_system_id: c_uint,
    bus_id: [c_char; 32],
}

#[repr(C)]
struct RawMemory {
    version: c_uint,
    total: u64,
    reserved: u64,
    free: u64,
    used: u64,
}

#[repr(C)]
struct RawBar1 {
    total: u64,
    free: u64,
    used: u64,
}

unsafe extern "C" fn init() -> i32 {
    with(|h| h.init)
}
unsafe extern "C" fn shutdown() -> i32 {
    with(|h| h.shut_down = true);
    code::SUCCESS
}
unsafe extern "C" fn driver_version(buf: *mut c_char, len: c_uint) -> i32 {
    // SAFETY: NVML's contract: `buf` is writable for `len`.
    unsafe { put(with(|h| h.driver), buf, len) }
}
unsafe extern "C" fn count(out: *mut c_uint) -> i32 {
    // SAFETY: `out` is writable.
    unsafe { *out = with(|h| h.gpus.len()) as c_uint };
    code::SUCCESS
}
unsafe extern "C" fn handle(i: c_uint, out: *mut *mut c_void) -> i32 {
    if let Some(code) = with(|h| h.gpus[i as usize].refused) {
        return code;
    }
    // SAFETY: `out` is writable.
    unsafe { *out = (i as usize + 1) as *mut c_void };
    code::SUCCESS
}
unsafe extern "C" fn uuid(d: *mut c_void, buf: *mut c_char, len: c_uint) -> i32 {
    // SAFETY: as `driver_version`.
    unsafe { put(gpu_of(d).uuid, buf, len) }
}
unsafe extern "C" fn name(d: *mut c_void, buf: *mut c_char, len: c_uint) -> i32 {
    // SAFETY: as `driver_version`.
    unsafe { put(gpu_of(d).name, buf, len) }
}
unsafe extern "C" fn pci(d: *mut c_void, out: *mut RawPci) -> i32 {
    let g = gpu_of(d);
    // SAFETY: `out` is writable.
    unsafe {
        *out = RawPci {
            bus_id_legacy: [0; 16],
            domain: g.domain,
            bus: g.bus,
            device: 0,
            pci_device_id: g.pci_device_id,
            pci_sub_system_id: g.subsystem,
            bus_id: [0; 32],
        }
    };
    code::SUCCESS
}
unsafe extern "C" fn memory(d: *mut c_void, out: *mut RawMemory) -> i32 {
    let g = gpu_of(d);
    // SAFETY: `out` is readable and writable.
    unsafe {
        with(|h| h.memory_version = Some((*out).version));
        (*out).total = g.total;
        (*out).free = g.free;
        (*out).used = g.total - g.free - g.reserved;
        (*out).reserved = g.reserved;
    }
    code::SUCCESS
}
unsafe extern "C" fn bar1(d: *mut c_void, out: *mut RawBar1) -> i32 {
    let g = gpu_of(d);
    // SAFETY: `out` is writable.
    unsafe {
        *out = RawBar1 {
            total: g.bar1,
            free: g.bar1,
            used: 0,
        }
    };
    code::SUCCESS
}
macro_rules! uint_getter {
    ($name:ident, $field:expr) => {
        unsafe extern "C" fn $name(d: *mut c_void, out: *mut c_uint) -> i32 {
            let g = gpu_of(d);
            let value: u32 = $field(&g);
            // SAFETY: `out` is writable.
            unsafe { *out = value };
            code::SUCCESS
        }
    };
}
uint_getter!(gen_cur, |g: &FakeGpu| g.link_cur.0);
uint_getter!(gen_max, |g: &FakeGpu| g.link_max.0);
uint_getter!(width_cur, |g: &FakeGpu| g.link_cur.1);
uint_getter!(width_max, |g: &FakeGpu| g.link_max.1);
uint_getter!(power_limit, |g: &FakeGpu| g.power_limit_mw);
uint_getter!(power_cap, |g: &FakeGpu| g.power_cap_mw);
unsafe extern "C" fn p2p(a: *mut c_void, b: *mut c_void, cap: c_uint, out: *mut c_uint) -> i32 {
    let status = with(|h| {
        h.p2p
            .get(&(index(a), index(b)))
            .map_or(1, |caps| caps[cap as usize])
    });
    // SAFETY: `out` is writable.
    unsafe { *out = status };
    code::SUCCESS
}
unsafe extern "C" fn nvlink(d: *mut c_void, link: c_uint, out: *mut c_uint) -> i32 {
    let Some(links) = gpu_of(d).nvlinks else {
        return code::NOT_SUPPORTED;
    };
    // SAFETY: `out` is writable.
    unsafe { *out = u32::from(links.contains(&link)) };
    code::SUCCESS
}
unsafe extern "C" fn temperature(d: *mut c_void, sensor: c_uint, out: *mut c_uint) -> i32 {
    assert_eq!(sensor, 0, "NVML_TEMPERATURE_GPU");
    let Some(t) = gpu_of(d).temperature else {
        return code::NOT_SUPPORTED;
    };
    // SAFETY: `out` is writable.
    unsafe { *out = t };
    code::SUCCESS
}

/// Every symbol a current NVML exports that the module calls.
fn symbols() -> BTreeMap<&'static str, *mut c_void> {
    let mut s: BTreeMap<&'static str, *mut c_void> = BTreeMap::new();
    s.insert("nvmlInit_v2", init as *mut c_void);
    s.insert("nvmlShutdown", shutdown as *mut c_void);
    s.insert("nvmlSystemGetDriverVersion", driver_version as *mut c_void);
    s.insert("nvmlDeviceGetCount_v2", count as *mut c_void);
    s.insert("nvmlDeviceGetHandleByIndex_v2", handle as *mut c_void);
    s.insert("nvmlDeviceGetUUID", uuid as *mut c_void);
    s.insert("nvmlDeviceGetName", name as *mut c_void);
    s.insert("nvmlDeviceGetPciInfo_v3", pci as *mut c_void);
    s.insert("nvmlDeviceGetMemoryInfo_v2", memory as *mut c_void);
    s.insert("nvmlDeviceGetBAR1MemoryInfo", bar1 as *mut c_void);
    s.insert(
        "nvmlDeviceGetCurrPcieLinkGeneration",
        gen_cur as *mut c_void,
    );
    s.insert("nvmlDeviceGetMaxPcieLinkGeneration", gen_max as *mut c_void);
    s.insert("nvmlDeviceGetCurrPcieLinkWidth", width_cur as *mut c_void);
    s.insert("nvmlDeviceGetMaxPcieLinkWidth", width_max as *mut c_void);
    s.insert("nvmlDeviceGetP2PStatus", p2p as *mut c_void);
    s.insert("nvmlDeviceGetNvLinkState", nvlink as *mut c_void);
    s.insert(
        "nvmlDeviceGetPowerManagementLimit",
        power_limit as *mut c_void,
    );
    s.insert("nvmlDeviceGetEnforcedPowerLimit", power_cap as *mut c_void);
    s.insert("nvmlDeviceGetTemperature", temperature as *mut c_void);
    s
}

struct FakeLib(BTreeMap<&'static str, *mut c_void>);

// SAFETY: every symbol is a function above with NVML's C signature for its name.
unsafe impl VendorLibrary for FakeLib {
    fn symbol(&self, name: &str) -> Option<*mut c_void> {
        self.0.get(name).copied()
    }
}

/// The fake `dlopen`: the shim library under NVML's file name, or nothing.
struct FakeLoader {
    library: Option<BTreeMap<&'static str, *mut c_void>>,
    asked: RefCell<Vec<String>>,
}

impl FakeLoader {
    fn with(symbols: BTreeMap<&'static str, *mut c_void>) -> Self {
        Self {
            library: Some(symbols),
            asked: RefCell::new(Vec::new()),
        }
    }

    fn absent() -> Self {
        Self {
            library: None,
            asked: RefCell::new(Vec::new()),
        }
    }
}

impl VendorLoader for FakeLoader {
    fn open(&self, name: &str) -> Option<Box<dyn VendorLibrary>> {
        self.asked.borrow_mut().push(name.to_owned());
        (name == LIBRARY)
            .then(|| self.library.clone())
            .flatten()
            .map(|s| Box::new(FakeLib(s)) as Box<dyn VendorLibrary>)
    }
}

fn ada() -> FakeGpu {
    FakeGpu {
        uuid: "GPU-8c6e1c2a-5d4f-1b3e-9a07-6f2d4e1b0c93",
        name: "NVIDIA GeForce RTX 4090",
        domain: 0,
        bus: 0x01,
        pci_device_id: 0x2684_10DE,
        subsystem: 0x1675_10DE,
        total: 24 * GIB,
        free: 23 * GIB,
        reserved: GIB / 2,
        bar1: 24 * GIB,
        link_cur: (4, 16),
        link_max: (4, 16),
        nvlinks: None,
        power_limit_mw: 450_000,
        power_cap_mw: 400_000,
        temperature: Some(41),
        refused: None,
    }
}

fn second() -> FakeGpu {
    FakeGpu {
        uuid: "GPU-00112233-4455-6677-8899-aabbccddeeff",
        name: "NVIDIA RTX A6000",
        domain: 1,
        bus: 0x81,
        pci_device_id: 0x2230_10DE,
        subsystem: 0x1459_10DE,
        total: 48 * GIB,
        free: 47 * GIB,
        reserved: 0,
        bar1: 256 * 1024 * 1024,
        link_cur: (3, 8),
        link_max: (4, 16),
        nvlinks: Some(vec![0, 2]),
        power_limit_mw: 300_000,
        power_cap_mw: 300_000,
        temperature: None,
        refused: None,
    }
}

fn host(driver: &'static str, gpus: Vec<FakeGpu>) -> FakeHost {
    FakeHost {
        init: code::SUCCESS,
        driver,
        gpus,
        p2p: BTreeMap::new(),
        memory_version: None,
        shut_down: false,
    }
}

/// Runs NVML's enumeration over `loader` with `fake` installed; bus 0x81 is NUMA node 1.
fn run(fake: FakeHost, loader: &FakeLoader) -> (Inventory, FakeHost) {
    HOST.with(|h| *h.borrow_mut() = Some(fake));
    let numa = |bdf: PciAddress| (bdf.bus == 0x81).then_some(1);
    let inventory = nvml::enumerate(loader, &numa);
    let after = HOST.with(|h| h.borrow_mut().take().unwrap());
    (inventory, after)
}

/// Criterion "present": with a current driver every GPU is listed with its PCI address, UUID,
/// VRAM total and budget, and the other NVML facts, and NVML is shut down after.
#[test]
fn a_present_library_lists_every_gpu() {
    let loader = FakeLoader::with(symbols());
    let mut fake = host("595.91.07", vec![ada(), second()]);
    // GPU 0 reads, writes and atomically accesses GPU 1 over PCIe; GPU 1 reads GPU 0 only.
    fake.p2p.insert((0, 1), [0, 0, 1, 0]);
    fake.p2p.insert((1, 0), [0, 1, 1, 1]);
    let (inventory, after) = run(fake, &loader);
    assert_eq!(loader.asked.borrow().as_slice(), [LIBRARY]);
    assert!(after.shut_down);
    assert_eq!(after.memory_version, Some(40 | (2 << 24)), "nvmlMemory_v2");
    assert_eq!(
        inventory.status,
        Status::Read {
            driver: "595.91.07".into(),
            skipped: vec![]
        }
    );
    let [a, b] = inventory.gpus.as_slice() else {
        panic!("{:?}", inventory.gpus);
    };

    assert_eq!(a.kind, DeviceKind::DiscreteGpu);
    assert!(!a.unified);
    assert_eq!(a.name, "NVIDIA GeForce RTX 4090");
    assert_eq!(
        a.ext.uuid,
        Some([
            0x8c, 0x6e, 0x1c, 0x2a, 0x5d, 0x4f, 0x1b, 0x3e, 0x9a, 0x07, 0x6f, 0x2d, 0x4e, 0x1b,
            0x0c, 0x93
        ])
    );
    let pci = a.ext.pci.unwrap();
    assert_eq!(
        pci.bdf,
        PciAddress {
            domain: 0,
            bus: 0x01,
            device: 0,
            function: 0
        }
    );
    assert_eq!(
        (pci.vendor, pci.device, pci.subsystem),
        (0x10DE, 0x2684, 0x1675_10DE)
    );
    assert_eq!(
        pci.link_cur,
        PcieLink {
            r#gen: 4,
            width: 16
        }
    );
    assert_eq!(
        pci.link_max,
        PcieLink {
            r#gen: 4,
            width: 16
        }
    );
    assert_eq!(pci.resizable_bar, Some(true), "BAR1 24 GiB = VRAM 24 GiB");
    assert_eq!((pci.numa, a.numa), (None, None));
    assert_eq!(a.vram_total, Some(Bytes(24 * GIB)));
    // The budget is NVML's free figure, not total − used, which would count the 512 MiB the
    // driver reserves as free.
    assert_eq!(a.vram_budget, Some(Bytes(23 * GIB)));
    assert_eq!(a.bar1_total, Some(Bytes(24 * GIB)));
    assert_eq!(a.temperature_c, Some(41));
    assert_eq!(a.nvlinks, 0, "NOT_SUPPORTED is no link");
    let power = a.ext.power.unwrap();
    assert_eq!((power.limit_w, power.cap_w), (Some(450.0), Some(400.0)));
    assert_eq!(a.ext.driver.driver, "595.91.07");
    assert!(a.ext.problems.is_empty());
    assert_eq!(
        a.peers,
        vec![PeerAccess {
            peer: 1,
            read: true,
            write: true,
            nvlink: false,
            atomics: true
        }]
    );

    let pci = b.ext.pci.unwrap();
    assert_eq!((pci.bdf.domain, pci.bdf.bus), (1, 0x81));
    assert_eq!(pci.link_cur, PcieLink { r#gen: 3, width: 8 });
    assert_eq!(pci.resizable_bar, Some(false));
    assert_eq!((pci.numa, b.numa), (Some(1), Some(1)), "sysfs numa_node");
    assert_eq!(b.vram_budget, Some(Bytes(47 * GIB)));
    assert_eq!(b.temperature_c, None, "NOT_SUPPORTED leaves it unknown");
    assert_eq!(b.nvlinks, 2);
    assert_eq!(
        b.peers,
        vec![PeerAccess {
            peer: 0,
            read: true,
            write: false,
            nvlink: false,
            atomics: false
        }]
    );
    // The ordinal input carries what RFC-0010 §1.4 sorts and dedupes on.
    let candidate = b.candidate();
    assert_eq!(candidate.pci, Some(pci.bdf));
    assert_eq!(candidate.uuid, b.ext.uuid);
    assert_eq!(candidate.numa, Some(1));
}

/// Criterion "absent": no library, or a library with no driver loaded, is no GPU and no error,
/// so the host enumerates CPU-only.
#[test]
fn an_absent_library_or_driver_is_no_gpu() {
    let (inventory, _) = run(host("595.91.07", vec![ada()]), &FakeLoader::absent());
    assert_eq!(inventory.status, Status::Absent);
    assert!(inventory.gpus.is_empty());

    let mut fake = host("595.91.07", vec![ada()]);
    fake.init = code::DRIVER_NOT_LOADED;
    let (inventory, after) = run(fake, &FakeLoader::with(symbols()));
    assert_eq!(inventory.status, Status::Absent);
    assert!(inventory.gpus.is_empty());
    assert!(!after.shut_down, "never initialised, so never shut down");
}

/// Criterion "old driver": a driver below 580 still lists every GPU, each marked
/// `DriverTooOld`; the versioned functions such a library lacks leave their fields unknown.
#[test]
fn an_old_driver_lists_its_gpus_as_too_old() {
    let mut old = symbols();
    old.remove("nvmlDeviceGetPciInfo_v3");
    old.remove("nvmlDeviceGetMemoryInfo_v2");
    let (inventory, _) = run(host("470.256.02", vec![ada()]), &FakeLoader::with(old));
    let [gpu] = inventory.gpus.as_slice() else {
        panic!("{:?}", inventory.gpus);
    };
    assert_eq!(
        gpu.ext.problems.as_slice(),
        [DeviceProblem::DriverTooOld {
            have: "470.256.02".into(),
            need: MIN_DRIVER_MAJOR.to_string(),
        }]
    );
    assert!(gpu.ext.uuid.is_some());
    assert_eq!(
        (gpu.ext.pci, gpu.vram_total, gpu.vram_budget),
        (None, None, None)
    );

    // The version alone decides: a current library with an old driver is still too old, and
    // 580 itself is not.
    for (driver, old) in [("575.64.05", true), ("580.65.06", false), ("", true)] {
        let (inventory, _) = run(host(driver, vec![ada()]), &FakeLoader::with(symbols()));
        assert_eq!(
            !inventory.gpus[0].ext.problems.is_empty(),
            old,
            "driver {driver:?}"
        );
        assert!(inventory.gpus[0].ext.pci.is_some());
    }
}

/// A library without a function every supported NVML has is unusable: no GPU, with the reason.
#[test]
fn a_broken_library_is_unusable_not_an_error() {
    let mut broken = symbols();
    broken.remove("nvmlDeviceGetUUID");
    let (inventory, _) = run(host("595.91.07", vec![ada()]), &FakeLoader::with(broken));
    assert_eq!(
        inventory.status,
        Status::Unusable {
            reason: "no nvmlDeviceGetUUID".into()
        }
    );
    assert!(inventory.gpus.is_empty());

    let mut fake = host("595.91.07", vec![ada()]);
    fake.init = 999;
    let (inventory, _) = run(fake, &FakeLoader::with(symbols()));
    assert!(matches!(inventory.status, Status::Unusable { .. }));
}

/// A GPU whose handle NVML refuses is skipped with its index and code; the others are listed.
#[test]
fn a_refused_gpu_is_skipped() {
    let mut hidden = second();
    hidden.refused = Some(NOT_PERMITTED);
    let (inventory, _) = run(
        host("595.91.07", vec![hidden, ada()]),
        &FakeLoader::with(symbols()),
    );
    assert_eq!(
        inventory.status,
        Status::Read {
            driver: "595.91.07".into(),
            skipped: vec![(0, NOT_PERMITTED)]
        }
    );
    assert_eq!(inventory.gpus.len(), 1);
    assert_eq!(inventory.gpus[0].name, "NVIDIA GeForce RTX 4090");
}

#[test]
fn uuids_and_driver_versions_parse() {
    assert_eq!(
        parse_uuid("GPU-00112233-4455-6677-8899-aabbccddeeff"),
        Some([
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff
        ])
    );
    assert!(parse_uuid("MIG-00112233-4455-6677-8899-aabbccddeeff").is_some());
    for bad in [
        "00112233-4455-6677-8899-aabbccddeeff",
        "GPU-00112233-4455-6677-8899-aabbccddeef",
        "GPU-0011223344556677-8899-aabbccddeeff0",
        "GPU-zz112233-4455-6677-8899-aabbccddeeff",
        "GPU-0011223-34455-6677-8899-aabbccddeeff",
    ] {
        assert_eq!(parse_uuid(bad), None, "{bad}");
    }
    assert_eq!(driver_major("595.91.07"), Some(595));
    assert_eq!(driver_major("580"), Some(580));
    assert_eq!(driver_major("abc"), None);
}

/// The real loader on this host. Without an NVIDIA driver (the hosted runners, a Mac) NVML is
/// absent and E2 lists no NVIDIA GPU, with no error. With one, every GPU it lists carries its
/// PCI address, UUID and budget. `SYNTROPY_PRINT_GPUS=1` prints the records.
#[test]
fn the_live_host_enumerates_without_error() {
    let inventory = nvml::enumerate(
        &syntropy_kernels_loader::gpu::SystemLoader,
        &nvml::sysfs_numa,
    );
    if std::env::var_os("SYNTROPY_PRINT_GPUS").is_some() {
        println!("{:#?}", inventory);
    }
    match &inventory.status {
        Status::Absent => assert!(inventory.gpus.is_empty()),
        Status::Unusable { reason } => panic!("NVML is installed but unusable: {reason}"),
        Status::Read { .. } => {
            for gpu in &inventory.gpus {
                assert!(gpu.ext.pci.is_some(), "{gpu:?}");
                assert!(gpu.ext.uuid.is_some(), "{gpu:?}");
                assert!(gpu.vram_budget.is_some_and(|b| b.0 > 0), "{gpu:?}");
                assert!(gpu.vram_total >= gpu.vram_budget, "{gpu:?}");
            }
        }
    }
}
