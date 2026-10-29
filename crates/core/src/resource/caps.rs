// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capability records, the probe's measurement output: RFC-0001 §5.2's `CapabilityProfile`,
//! `DeviceCaps` and `StorageCaps` ([`CapabilityProfile`], [`DeviceCaps`], [`StorageCaps`];
//! PLC-E1-T17), RFC-0010 §1.3's additions to them and to `LinkCaps`, and the types they use.
//!
//! Each base record holds the fields RFC-0010 adds as one `ext` field with serde defaults, as
//! [`DeviceInfo`](super::DeviceInfo) holds `DeviceInfoExt`, so a record written before an
//! addition still decodes. `LinkCaps` is in the link module.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::dtype::DType;
use crate::ids::{Blake3, DeviceId, LinkId, NetMedium, NodeId, TierId};
use crate::measured::Measured;
use crate::quant::QuantScheme;
use crate::units::{Bytes, BytesPerSec, FlopsPerSec, Nanos};

use super::inventory::NodeInventory;
use super::link::LinkCaps;

/// Measured numbers for one node's devices, storage and outbound links (RFC-0001 §5.2): what
/// the startup probes produce and live timings refine. `ProbeResult` carries one from a probe
/// process to its daemon. Owner: RFC-0010.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilityProfile {
    /// The node it describes.
    pub node: NodeId,
    /// Bumped on each change; the coordinator takes a `ReportProfile` only if newer
    /// (RFC-0010 §1.4).
    pub version: u64,
    /// One record per device.
    pub devices: Vec<DeviceCaps>,
    /// One record per storage tier.
    pub storage: Vec<StorageCaps>,
    /// The links whose source is on this node.
    pub links: Vec<LinkCaps>,
    /// RFC-0010 §1.3's additions.
    #[serde(default)]
    pub ext: CapabilityProfileExt,
}

/// One device's measured rates (RFC-0001 §5.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceCaps {
    /// The device.
    pub device: DeviceId,
    /// W_eff per (tier, kernel class): weight bytes consumed per second by the kernel that will
    /// run, not copy bandwidth (cost model §3.2, §5.3). A per-class summary: `ext.quant_bw`,
    /// keyed by quant scheme too, is the solver's source.
    pub tier_bw: Vec<(TierId, KernelClass, Measured<BytesPerSec>)>,
    /// F_eff per (op class, dtype), for prefill, verify windows and n*.
    pub throughput: Vec<(OpClass, DType, Measured<FlopsPerSec>)>,
    /// The backend's per-step fixed cost on this device (launch, sync), fitted as an intercept
    /// (cost model §3.4).
    pub t_fixed: Measured<Nanos>,
    /// Pinned and pageable host transfers over PCIe or a Thunderbolt tunnel; `None` on a
    /// unified-memory device.
    pub host_link: Option<HostLinkCaps>,
    /// Capacity per tier that allocation probing confirmed; may be below what the driver
    /// reports.
    pub usable: Vec<(TierId, Bytes)>,
    /// RFC-0010 §1.3's additions.
    #[serde(default)]
    pub ext: DeviceCapsExt,
}

/// One storage tier's measured rates (RFC-0001 §5.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StorageCaps {
    /// The tier.
    pub tier: TierId,
    /// Sustained, thermally throttled read rate at the pack's extent size and queue depth: the
    /// rate the solver uses (cost model §2.4, §3.3).
    pub sustained_read: Measured<BytesPerSec>,
    /// The burst read rate.
    pub burst_read: Measured<BytesPerSec>,
    /// The extent size the rates were measured at.
    pub extent: Bytes,
    /// The queue depth the rates were measured at.
    pub queue_depth: u16,
    /// 4 KiB read latency at queue depth 1.
    pub small_read: Measured<Nanos>,
    /// The volume's temperature, where it reports one.
    pub temperature_c: Option<Measured<f32>>,
    /// RFC-0010 §1.3's and RFC-0021 §10's additions.
    #[serde(default)]
    pub ext: StorageCapsExt,
}

/// An op class: a group of kernels with the same performance shape, the key of
/// [`DeviceCaps::throughput`] (F_eff per op class and dtype; glossary "Op class"). RFC-0001
/// §5.2 and RFC-0060's per-device-class selection map name it.
///
/// The variants are option (a) of PLC-E1-T17, proposed for the maintainers in RFC-0010 §1.3:
/// the glossary's op classes plus step M4's GEMM (RFC-0010 §3.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OpClass {
    /// Matrix-matrix products: prefill and verify-window projections, in a tensor-core dtype
    /// or through the dequant path of a quantized scheme (step M4's GEMM).
    Gemm,
    /// Quantized matrix-vector products: decode projections over quantized weights.
    QuantGemv,
    /// Grouped expert matrix-vector products of a mixture-of-experts layer.
    ExpertGemv,
    /// Attention of one kind (step M4 measures each kind the provider implements).
    Attention(AttnKind),
    /// The scan of a linear-attention layer.
    LinearScan,
}

/// A kernel class the probe measures W_eff for (RFC-0001 §5.2).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum KernelClass {
    /// Dense matrix-vector products.
    DenseGemv,
    /// Routed-expert matrix-vector products.
    ExpertGemv,
    /// Attention of one kind.
    Attention(AttnKind),
    /// Row gathers (embedding lookup).
    RowGather,
    /// Copies and conversions.
    Copy,
}

/// An attention kind (RFC-0040 §3.1). Owner: RFC-0040.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AttnKind {
    /// Grouped-query attention.
    Gqa,
    /// Multi-head latent attention.
    Mla,
    /// Sliding-window grouped-query attention.
    SlidingGqa {
        /// The window keeps attention-sink tokens.
        sinks: bool,
    },
    /// Block-sparse attention.
    BlockSparse,
    /// Top-k sparse attention.
    TopKSparse,
    /// Attention over compressed entries.
    Compressed,
    /// Encoder-only models: every row sees every key of its sequence (RFC-0043 §5.4).
    Bidirectional,
}

/// RFC-0010 §1.3's additions to `CapabilityProfile`. Every field has a serde default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CapabilityProfileExt {
    /// Profile cache key (RFC-0010 §5).
    pub key: ProfileKey,
    /// How complete the profile is.
    pub status: ProfileStatus,
    /// The power state this profile was measured in.
    pub power: PowerSource,
    /// The host has a battery: a laptop-class node, departure-prone for RFC-0011 §4's margin
    /// whether or not it is on mains now.
    pub battery_present: bool,
    /// Bumped when a probe's method changes.
    pub suite: ProbeSuiteVersion,
    /// Set only when devices, tiers or links changed after admission (RFC-0010 §1.4).
    pub inventory: Option<NodeInventory>,
    /// Origin download rate from completed origin fetches (RFC-0010 §6.1). RFC-0011 turns it
    /// into the plan's WAN rate class.
    pub wan_down: Option<Measured<BytesPerSec>>,
}

/// The profile cache key: BLAKE3 over the canonical encoding of the inputs RFC-0010 §5.1
/// lists.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ProfileKey(pub Blake3);

/// How complete a capability profile is (RFC-0010 §1.3).
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ProfileStatus {
    /// The confirmation subset only. The serde default.
    #[default]
    Quick,
    /// Every startup step.
    Full,
    /// Every startup step and the sustained NVMe run.
    FullWithSustained,
}

/// Mains or battery power (RFC-0001 §5.2 `NodeInfo::power`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerSource {
    /// Mains power. The serde default.
    #[default]
    Mains,
    /// Battery power.
    Battery {
        /// Charge left, 0–100.
        percent: u8,
    },
}

/// Version of the probe suite's methods; bumped when a probe's method changes, which
/// invalidates cached profiles (RFC-0010 §5.1).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ProbeSuiteVersion(pub u32);

/// RFC-0010 §1.3's additions to `DeviceCaps`. Every field has a serde default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceCapsExt {
    /// W_eff per (tier, kernel class, quant scheme) with its batch curve. RFC-0001's
    /// `tier_bw` keeps one entry per (tier, class): the value for the scheme with the most
    /// bytes in the probe set.
    pub quant_bw: Vec<QuantBw>,
    /// CPU↔GPU activation hand-off through host-mapped memory and a doorbell: one direction,
    /// 16 KiB payload, device-side wait. Cost model §3.2 `t_switch`.
    pub t_switch: Option<Measured<Nanos>>,
    /// Launch cost of one small kernel outside a captured graph, and per node inside one.
    pub launch: Option<LaunchCaps>,
    /// The CPU pool's slowdown per GB/s of concurrent PCIe DMA reading the same DRAM: the
    /// `d` coefficient of RFC-0023's three-site balance.
    pub dma_interference: Option<Measured<f32>>,
    /// Doorbell-to-running latency through the backend's device-triggered wake (RFC-0012
    /// §3.3); sets `exec.inflight_park_threshold`.
    pub wake: Option<Measured<Nanos>>,
    /// Per host wait on backends without whole-step capture of host waits (M5, RFC-0012
    /// §7.5).
    pub host_wait: Option<Measured<Nanos>>,
}

/// W_eff for one (tier, kernel class, quant scheme), with its batch curve (RFC-0010 §1.3).
/// `tier`, `class` and `scheme` identify the record and have no default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantBw {
    /// The tier the weights are read from.
    pub tier: TierId,
    /// The kernel class.
    pub class: KernelClass,
    /// The quant scheme.
    pub scheme: QuantScheme,
    /// Weight bytes consumed per second at n tokens per pass, for n in {1, 2, 4, 8, 16, 32}.
    /// Point 0 (n = 1) is W_eff (cost model §3.1, §3.2, §3.10).
    #[serde(default)]
    pub curve: SmallVec<[(u16, Measured<BytesPerSec>); 6]>,
    /// Buffer size used: > 2 × LLC, ≥ 1 GiB.
    #[serde(default)]
    pub extent: Bytes,
    /// `Unified` devices only: the n = 1 rate while the I/O engine reads at the volume's S1
    /// rate (RFC-0020 §9).
    #[serde(default)]
    pub contended: Option<Measured<BytesPerSec>>,
}

/// Kernel launch cost (RFC-0010 §1.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LaunchCaps {
    /// One small kernel outside a captured graph.
    pub eager: Measured<Nanos>,
    /// Per node inside a captured graph, where capture works.
    pub captured_node: Option<Measured<Nanos>>,
}

/// Host↔device copy rates over one PCIe link or Thunderbolt tunnel (RFC-0010 §1.3; RFC-0001
/// names the type). `link` identifies the record and has no default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostLinkCaps {
    /// The `Pcie` link (or Thunderbolt tunnel).
    pub link: LinkId,
    /// 64 MiB copies from the registered arena.
    #[serde(default)]
    pub h2d_pinned: Measured<BytesPerSec>,
    /// 64 MiB copies into the registered arena.
    #[serde(default)]
    pub d2h_pinned: Measured<BytesPerSec>,
    /// For diagnosis only; streaming never uses it.
    #[serde(default)]
    pub h2d_pageable: Measured<BytesPerSec>,
    /// 16 KiB copy latency, submit to completion event.
    #[serde(default)]
    pub h2d_small: Measured<Nanos>,
    /// H2D and D2H at once, summed.
    #[serde(default)]
    pub bidirectional: Option<Measured<BytesPerSec>>,
    /// Verified peer copies (ADR-0006 item 5).
    #[serde(default)]
    pub p2p: Vec<(DeviceId, Measured<BytesPerSec>)>,
}

/// RFC-0010 §1.3's additions to `StorageCaps`, including the fields RFC-0021 §10 defines.
/// Every field has a serde default.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageCapsExt {
    /// The engine S1 chose (RFC-0021 §4, §5, §7).
    pub engine: EngineKind,
    /// Best of 1, 2, 4, 8 MiB at `qd_target`.
    pub chunk: Bytes,
    /// Memory and offset alignment for direct I/O.
    pub dio: DioAlign,
    /// RFC-0021 §4.2 flagged (non-direct) mode.
    pub buffered: bool,
    /// SQPOLL pays on this volume (RFC-0021 §4.1).
    pub sqpoll: bool,
    /// RFC-0021 §6.
    pub read_around_efficiency: Option<f32>,
    /// Copied from `VolumeInfo::shared_lane` (cost model §2.4).
    pub shared_lanes: bool,
    /// Smallest queue depth reaching ≥ 95% of the best rate at the dominant request size
    /// (RFC-0021 §3.2).
    pub qd_target: u16,
    /// Read-only report of direct-to-GPU paths per GPU (RFC-0021 §9; none is built in v1).
    pub direct_paths: SmallVec<[(DeviceId, DirectPathReport); 2]>,
    /// The measured grid: engine × size × queue depth → throughput and latency spread.
    pub grid: Vec<IoPoint>,
    /// Progress of the sustained run (S2).
    pub sustained: SustainedStatus,
    /// Thermal throttling, once seen.
    pub throttle: Option<ThrottleInfo>,
    /// Engines this host refuses, and why (e.g. io_uring under seccomp).
    pub engines_blocked: SmallVec<[(EngineKind, BlockReason); 2]>,
}

/// An I/O engine (RFC-0021 §2). Owner: RFC-0021.
#[non_exhaustive]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EngineKind {
    /// io_uring with `O_DIRECT`.
    UringDirect,
    /// A `pread` thread pool. The serde default: the engine that works in every sandbox
    /// (RFC-0021 goal 3).
    #[default]
    PreadPool,
    /// Windows IoRing.
    IoRing,
    /// Windows IOCP.
    Iocp,
    /// macOS `F_NOCACHE` reads.
    MacNoCache,
    /// macOS mapped reads.
    MacMapped,
}

/// Direct-I/O alignment (RFC-0021 §2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct DioAlign {
    /// Memory alignment, in bytes.
    pub mem: u32,
    /// File offset alignment, in bytes.
    pub offset: u32,
}

/// What a host offers for direct storage-to-GPU reads on one GPU (RFC-0010 §1.3: cuFile
/// loads, opens, P2P or compatibility mode; DirectStorage runtime present). Report only;
/// none is built in v1 (RFC-0021 §9).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct DirectPathReport {
    /// The cuFile library loads.
    pub cufile_loads: bool,
    /// cuFile opens a handle on the store volume.
    pub cufile_opens: bool,
    /// cuFile reads peer-to-peer rather than in compatibility mode. `None` when it does not
    /// open.
    pub cufile_p2p: Option<bool>,
    /// The DirectStorage runtime is present.
    pub directstorage: bool,
}

/// One point of the storage grid (RFC-0010 §1.3). `engine`, `size` and `qd` identify it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct IoPoint {
    /// The engine.
    pub engine: EngineKind,
    /// Request size.
    pub size: Bytes,
    /// Queue depth.
    pub qd: u16,
    /// I/O-thread CPU time per GB (RFC-0021 §10 tie-break).
    pub cpu_per_gb: Option<Nanos>,
    /// Throughput.
    pub rate: Measured<BytesPerSec>,
    /// Latency per request.
    pub latency: Measured<Nanos>,
}

/// Progress of the sustained NVMe run, S2 (RFC-0010 §1.3).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SustainedStatus {
    /// Not started. The serde default.
    #[default]
    Pending,
    /// Running.
    Running {
        /// Start time, in Unix milliseconds.
        since_ms: u64,
    },
    /// Finished.
    Done,
}

/// Thermal throttling seen on a volume (RFC-0010 §1.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThrottleInfo {
    /// Time under load before the rate fell.
    pub onset_after: Nanos,
    /// Sustained / burst.
    pub ratio: f32,
    /// Temperature when the rate fell, where readable.
    pub temp_at_onset_c: Option<f32>,
}

/// Why an I/O engine is unavailable on this host. RFC-0010 §1.3 names the type without
/// variants; these are the causes RFC-0021 §4 lists.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BlockReason {
    /// Setup failed with `ENOSYS` or `EPERM` (a seccomp profile, Android, ChromeOS).
    Denied,
    /// The `io_uring_disabled` sysctl forbids it.
    SysctlDisabled,
    /// The OS version lacks it.
    Unsupported,
    /// Turned off by configuration.
    ConfigDisabled,
}

/// RFC-0010 §1.3's additions to `LinkCaps`. One `LinkCaps` exists per (link, transport)
/// pair. Every field has a serde default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LinkCapsExt {
    /// Ping-pong grid: 64 B, 16 KiB, 1 MiB, 64 MiB. α is fitted from the small sizes, β from
    /// the large.
    pub grid: SmallVec<[(Bytes, Measured<Nanos>); 4]>,
    /// From the online histogram once it has ≥ 10,000 samples.
    pub p999: Option<Nanos>,
    /// Mean one-way hop time at the decode frame size: the value the solver costs `WiFi`
    /// and `Overlay` hops at (RFC-0011 §5).
    pub mean: Option<Measured<Nanos>>,
    /// The hop-time histogram, compacted, for C14's stall rate (RFC-0011 §6).
    pub hop_hist: Option<HopHistogram>,
    /// Hop-time excess shows a periodic pattern (autocorrelation peak at a lag near 1 s).
    pub periodic_stall: bool,
    /// α_eff provenance: which buffers, which sync method.
    pub alpha_eff_path: Option<DevicePath>,
    /// `LinkCaps::alpha_eff` includes the transfer time at the partial size. Always true;
    /// recorded so readers need not guess, and true by default.
    pub alpha_eff_includes_transfer: bool,
    /// Duplicated from `LinkId::kind` for quick filtering.
    pub medium: NetMedium,
}

impl Default for LinkCapsExt {
    fn default() -> Self {
        Self {
            grid: SmallVec::new(),
            p999: None,
            mean: None,
            hop_hist: None,
            periodic_stall: false,
            alpha_eff_path: None,
            alpha_eff_includes_transfer: true,
            medium: NetMedium::default(),
        }
    }
}

/// Compacted hop-time histogram for one (link, transport): log buckets (ratio 1.02) over
/// `probe.link.window`, merged with the decayed all-time histogram (RFC-0010 §6.2).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HopHistogram {
    /// Lower edge of the first bucket.
    pub first: Nanos,
    /// Ratio between consecutive bucket edges.
    pub ratio: f32,
    /// Count per bucket.
    pub counts: Vec<u32>,
    /// Total samples.
    pub samples: u64,
}

/// The device path an α_eff was measured through (RFC-0010 §1.3, §3.5). `src` and `dst`
/// identify it and have no default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DevicePath {
    /// The local device.
    pub src: DeviceId,
    /// The remote device.
    pub dst: DeviceId,
    /// Payload size.
    #[serde(default)]
    pub payload: Bytes,
    /// How each end synchronises.
    #[serde(default)]
    pub sync: SyncKind,
}

/// How a device path synchronises on each end (RFC-0010 §3.2 L2: "the doorbell or event
/// sync on both ends"). RFC-0010 names the type without variants.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SyncKind {
    /// A device event. The serde default.
    #[default]
    Event,
    /// A doorbell in host-mapped memory.
    Doorbell,
}
