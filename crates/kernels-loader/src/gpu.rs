// SPDX-License-Identifier: MIT OR Apache-2.0

//! E2, GPU inventory from the management APIs (RFC-0010 §2.1, §2.2): NVML for NVIDIA GPUs
//! ([`nvml`]) and IOKit's `IOAccelerator` for Apple GPUs ([`iokit`]).
//!
//! E2 runs in the runner's probe mode, in this crate, because the vendor libraries are loaded
//! with `dlopen` and called through `unsafe` that the daemon may not hold (RFC-0001 §4.2 rule 8,
//! note of 2026-10-22; OQ-PRC-26). It opens no GPU context. Its output is one [`GpuRecord`] per
//! GPU: a provisional device with its PCI address, UUID, VRAM total and OS budget. PLC-E1-T07
//! sends the records with the probe's result, and the daemon merges them with E5's devices by
//! UUID, LUID or PCI address and numbers them (RFC-0010 §1.4).
//!
//! A missing library means no such device, never an error (RFC-0010 §2.1; RFC-0001 §4.2 rule 7),
//! so a host without a GPU driver enumerates no GPU here and runs on its CPU devices.
//!
//! Vendor libraries are reached through [`VendorLoader`], so tests replace `dlopen` with a shim
//! whose symbols are Rust functions ([`SystemLoader`] is the real one).

pub mod iokit;
pub mod nvml;

use std::ffi::c_void;

use syntropy_core::resource::ordinal::DeviceCandidate;
use syntropy_core::resource::{DeviceInfoExt, DeviceKind};
use syntropy_core::units::Bytes;

use crate::sys::RawLibrary;

/// A loaded vendor library: its symbols by name.
///
/// # Safety
///
/// A symbol returned for a name must be the function that name has in the vendor's published
/// API, with its C signature, because the caller calls it as one. The library must stay loaded
/// for as long as the value lives.
pub unsafe trait VendorLibrary {
    /// The address of `name`, or `None` when the library does not export it, as an older driver
    /// does not export a newer function.
    fn symbol(&self, name: &str) -> Option<*mut c_void>;
}

/// Opens vendor libraries by file name: the system loader in production ([`SystemLoader`]), a
/// shim in tests.
pub trait VendorLoader {
    /// The library `name`, or `None` when none is installed.
    fn open(&self, name: &str) -> Option<Box<dyn VendorLibrary>>;
}

/// The system loader: `dlopen` by name on Unix, the system directory on Windows.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemLoader;

// SAFETY: the library was opened by its vendor's file name, and a loaded library is never
// unloaded (`RawLibrary`), so a symbol is the vendor's function for as long as it is used.
unsafe impl VendorLibrary for RawLibrary {
    fn symbol(&self, name: &str) -> Option<*mut c_void> {
        RawLibrary::symbol(self, name).ok()
    }
}

impl VendorLoader for SystemLoader {
    fn open(&self, name: &str) -> Option<Box<dyn VendorLibrary>> {
        RawLibrary::open_system(name)
            .ok()
            .map(|lib| Box::new(lib) as Box<dyn VendorLibrary>)
    }
}

/// Whether one GPU can reach another directly, as the driver reports it (NVML's P2P status).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PeerAccess {
    /// The other GPU, by its index in the same inventory.
    pub peer: usize,
    /// Peer reads are supported.
    pub read: bool,
    /// Peer writes are supported.
    pub write: bool,
    /// The two are joined by NVLink.
    pub nvlink: bool,
    /// Peer atomics are supported.
    pub atomics: bool,
}

/// One GPU as E2 found it: a provisional device, before the provider confirms it (E5) and the
/// daemon numbers it.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuRecord {
    /// Discrete or integrated.
    pub kind: DeviceKind,
    /// The name the management API reports.
    pub name: String,
    /// The GPU computes on memory it shares with the CPU (Apple), which becomes a `Unified`
    /// tier rather than `Vram` (RFC-0010 §2.2).
    pub unified: bool,
    /// The NUMA domain of its PCI slot, where known: sysfs `numa_node` on Linux.
    pub numa: Option<u16>,
    /// Total VRAM. `None` for a unified GPU and where the driver did not say.
    pub vram_total: Option<Bytes>,
    /// The OS budget: the VRAM free now, which RFC-0020 §5 starts the `Vram` tier's `usable`
    /// from before D2 confirms it by allocation. `None` where the driver did not say.
    pub vram_budget: Option<Bytes>,
    /// The CPU-visible BAR1 aperture.
    pub bar1_total: Option<Bytes>,
    /// GPU core temperature in °C.
    pub temperature_c: Option<u32>,
    /// Active NVLink links.
    pub nvlinks: u8,
    /// Direct access to every other GPU of the inventory that reports it.
    pub peers: Vec<PeerAccess>,
    /// Driver, PCI, UUID, power, Apple core count and problems, as `DeviceInfo` carries them.
    pub ext: DeviceInfoExt,
}

impl GpuRecord {
    /// The record's input to RFC-0010 §1.4's ordinal rules.
    #[must_use]
    pub fn candidate(&self) -> DeviceCandidate {
        DeviceCandidate {
            kind: self.kind,
            pci: self.ext.pci.map(|pci| pci.bdf),
            numa: self.numa,
            uuid: self.ext.uuid,
            name: self.name.clone(),
            unified: self.unified,
        }
    }
}

/// E2 on this host: the NVIDIA GPUs NVML lists, and on macOS the Apple GPU IOKit lists. Never
/// fails: a library that is missing or unusable contributes no GPU.
#[must_use]
pub fn enumerate() -> Vec<GpuRecord> {
    let mut gpus = nvml::enumerate(&SystemLoader, &nvml::sysfs_numa).gpus;
    #[cfg(target_os = "macos")]
    if let Ok(accelerators) = iokit::live_accelerators() {
        gpus.extend(iokit::apple_gpus(&accelerators));
    }
    gpus
}
