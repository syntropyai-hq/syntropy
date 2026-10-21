// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tier topology records: RFC-0001 §5.2's `TierInfo`, RFC-0010 §1.2's additions and
//! `VolumeInfo`, and RFC-0010 §2.2's host pool record.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::ids::{DeviceId, TierId};
use crate::units::Bytes;

/// One memory or storage level on a node (RFC-0001 §5.2). Owner: RFC-0010 (discovery),
/// RFC-0020 (semantics).
///
/// Each NUMA domain contributes one `HostPinned` and one `HostPageable` tier. The two share
/// one pool of RAM, which [`HostPoolInfo`] bounds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TierInfo {
    /// The tier's identity; the ordinal follows RFC-0010 §1.4.
    pub id: TierId,
    /// What kind of memory or storage it is.
    pub kind: TierKind,
    /// Physical size.
    pub capacity: Bytes,
    /// After OS reserve, driver reserve, wired and pinning limits, and user headroom
    /// (RFC-0001 §9.2). `HostPinned`: `pin_limit_n`. `HostPageable`: `host_usable_n`, shared
    /// with `HostPinned`.
    pub usable: Bytes,
    /// Devices that compute on bytes in this tier without a copy.
    pub affinity: SmallVec<[DeviceId; 4]>,
    /// `HostPinned`: the OS's DMA-registrable ceiling.
    pub pin_limit: Option<Bytes>,
    /// `Nvme`: the volume's filesystem, direct-I/O support and shared-lane flag.
    pub volume: Option<VolumeInfo>,
    /// RFC-0010 §1.2's additions.
    #[serde(default)]
    pub ext: TierInfoExt,
}

/// What kind of memory or storage a tier is (RFC-0001 §5.2). Owner: RFC-0020.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TierKind {
    /// Device-local memory of a discrete GPU.
    Vram,
    /// Memory shared by CPU and GPU (Apple, APUs, GB10-class); no PCIe term.
    Unified,
    /// Host RAM registered for DMA (ADR-0200): locked, and registered with each provider, NIC
    /// or I/O ring that copies from it.
    HostPinned,
    /// Anonymous host RAM, never file-backed, that only the CPU reads.
    HostPageable,
    /// A storage volume.
    Nvme,
}

/// RFC-0010 §1.2's additions to [`TierInfo`]. Every field has a serde default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TierInfoExt {
    /// NUMA domain, where known.
    pub numa: Option<u16>,
    /// Where `usable` came from, so the dashboard can explain a low number.
    pub budget_source: BudgetSource,
    /// Raw OS inputs for `HostPinned` and `HostPageable` tiers. RFC-0020's pure rule
    /// `budget::host_usable(&HostMemInfo, &ResourcePolicy)` turns them into `usable`.
    pub host: Option<HostMemInfo>,
    /// Context reserve measured at device open (RFC-0001 §9.3 item 1). `Vram` and `Unified`
    /// only.
    pub context_reserve: Option<Bytes>,
}

/// Where a tier's `usable` came from (RFC-0010 §1.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BudgetSource {
    /// Discrete VRAM, confirmed by reserve-and-map. The serde default, as the first
    /// listed variant.
    #[default]
    AllocationProbe,
    /// `IDXGIAdapter3::QueryVideoMemoryInfo`.
    DxgiBudget,
    /// `VK_EXT_memory_budget` `heapBudget`.
    VkMemoryBudget,
    /// `recommendedMaxWorkingSetSize`.
    MetalWorkingSet,
    /// GB10-class: `MemAvailable` − 6 GiB (ADR-0014).
    MemAvailableMinus {
        /// The amount subtracted.
        reserve: Bytes,
    },
    /// RFC-0020's rule over [`HostMemInfo`].
    HostRules,
    /// `Nvme`: free bytes on the volume, capped by `store.max_bytes`.
    Statfs,
}

/// Raw OS memory inputs for a host tier (RFC-0010 §1.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HostMemInfo {
    /// Physical RAM.
    pub total: Bytes,
    /// `MemAvailable` / `ullAvailPhys` / `vm_stat` free + inactive + purgeable.
    pub available: Bytes,
    /// Free memory in this tier's NUMA domain.
    pub numa_free: Option<Bytes>,
    /// cgroup v2 `memory.max` (v1 `memory.limit_in_bytes`) − current.
    pub cgroup_limit: Option<Bytes>,
    /// `RLIMIT_MEMLOCK`; `None` = unlimited or `CAP_IPC_LOCK` held.
    pub memlock_limit: Option<Bytes>,
    /// Working-set ceiling obtainable without privilege (Windows).
    pub windows_lockable: Option<Bytes>,
    /// `/dev/dxg` present: the `cudaHostAlloc` cap applies (RFC-0001 §9.2).
    pub wsl2: bool,
    /// `iogpu.wired_limit_mb` if set, else `None` (the default cap).
    pub macos_wired_limit: Option<Bytes>,
    /// Huge or large page size usable without privilege.
    pub large_pages: Option<Bytes>,
    /// Lock accounting measured by step H5 (RFC-0010 §3.2), the inputs of RFC-0020 §5's
    /// `lockable` and `io_fixed` formulas.
    pub lock: Option<LockAccounting>,
}

/// How locking and registration are charged on this host, from step H5 (RFC-0010 §1.2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LockAccounting {
    /// `VmLck` from `/proc/self/status` before the test registrations.
    pub vmlck_before: Bytes,
    /// `VmPin` from `/proc/self/status` before the test registrations.
    pub vmpin_before: Bytes,
    /// Linux: whether a 64 MiB `register_host` raised `VmLck` or `VmPin`, i.e. whether
    /// provider host registration is charged to `RLIMIT_MEMLOCK` on this driver. `None` where
    /// not testable.
    pub registration_charged: Option<bool>,
    /// Linux: whether a 64 MiB io_uring fixed-buffer registration raised `VmPin`.
    pub io_fixed_charged: Option<bool>,
    /// Windows: the largest lock-and-register total reached in 1 GiB ranges up to
    /// `max_pinned_fraction` before the first failure (RFC-0020 §5 `lockable_measured`).
    pub lockable_measured: Option<Bytes>,
    /// The driver version the result is valid for.
    pub driver: String,
}

/// The volume behind an `Nvme` tier (RFC-0010 §1.2; RFC-0001 names the type without fields).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VolumeInfo {
    /// The store directory's volume.
    pub mount: PathBuf,
    /// Filesystem.
    pub fs: FsKind,
    /// Storage bus.
    pub bus: StorageBus,
    /// The controller; `upstream` says CPU root port or chipset.
    pub pci: Option<super::device::PciInfo>,
    /// Shares an uplink with other storage or NICs (e.g. an AM5 chipset).
    pub shared_lane: bool,
    /// Direct-I/O support and alignment.
    pub direct_io: DirectIoSupport,
    /// Logical block size.
    pub logical_block: u32,
    /// Physical block size.
    pub physical_block: u32,
    /// FileVault, BitLocker or dm-crypt, when detectable.
    pub encrypted: Option<bool>,
    /// Free bytes.
    pub free: Bytes,
    /// Where temperature readings come from.
    pub temperature_source: TempSource,
    /// Drive model.
    pub model: String,
    /// Drive firmware version.
    pub firmware: String,
}

/// A volume's filesystem (RFC-0010 §1.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FsKind {
    /// ext4.
    Ext4,
    /// XFS.
    Xfs,
    /// Btrfs.
    Btrfs,
    /// ZFS.
    Zfs,
    /// NTFS.
    Ntfs,
    /// ReFS.
    Refs,
    /// APFS.
    Apfs,
    /// Any other filesystem. The serde default.
    #[default]
    Other,
}

/// A volume's storage bus (RFC-0010 §1.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageBus {
    /// NVMe.
    Nvme,
    /// SATA.
    Sata,
    /// USB.
    Usb,
    /// A virtual disk.
    Virtual,
    /// Not determined. The serde default.
    #[default]
    Unknown,
}

/// Whether a volume accepts direct I/O, and at what alignment (RFC-0010 §1.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DirectIoSupport {
    /// Direct I/O works at these alignments.
    Supported {
        /// Memory alignment, in bytes.
        mem_align: u32,
        /// File offset alignment, in bytes.
        offset_align: u32,
    },
    /// Direct I/O is refused. The serde default.
    #[default]
    Unsupported,
}

/// Where a volume's temperature comes from (RFC-0010 §1.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TempSource {
    /// The Linux NVMe hwmon.
    Hwmon,
    /// Windows `StorageDeviceTemperatureProperty`.
    StorageQuery,
    /// No interface (macOS has no public one). The serde default.
    #[default]
    None,
}

/// One NUMA domain's host RAM pool, linking its `HostPinned` and `HostPageable` tiers
/// (RFC-0010 §2.2). `HostPinned_n.usable = pin_limit_n`, `HostPageable_n.usable =
/// host_usable_n`, and this record bounds their sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostPoolInfo {
    /// The NUMA domain.
    #[serde(default)]
    pub numa: u16,
    /// `[HostPinned_n, HostPageable_n]`.
    pub tiers: [TierId; 2],
    /// `host_usable_n`: both tiers together.
    #[serde(default)]
    pub usable: Bytes,
    /// `pin_limit_n`: `HostPinned` alone.
    #[serde(default)]
    pub pin_limit: Bytes,
}
