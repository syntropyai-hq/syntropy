// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`FakeKernelProvider`] in `Numeric` mode, and the [`FakeDeviceSpec`] that times it
//! (RFC-0060 §8.4; ADR-0600 decision 4; QUAL-E3-T12).
//!
//! **Numerics.** The fake is the CPU provider underneath. Every buffer is real host memory that
//! [`CpuProvider`] allocates or imports, every stream, event, doorbell and capture is the CPU
//! provider shell's, and every op runs the CPU provider's kernel for it. The fake opens each CPU
//! device with `OpenOptions::deterministic` set, whatever the caller asks, so `prepare` takes
//! only kernels that declare a reduction order independent of the thread count (RFC-0060 §2.5;
//! RFC-0040 §3.3) and refuses the rest with `OpUnsupported`. Two fake devices given the same
//! inputs therefore compute the same bits, which is what makes the simulator's token stream a
//! strict oracle (RFC-0060 §8.6).
//!
//! **Devices.** The fake enumerates the devices it was built with, one per [`FakeDeviceSpec`],
//! not the host's NUMA domains. Every fake device opens CPU device 0, so a scenario of eight
//! nodes runs on one host. A device reports the host CPU's class and features, since those are
//! the kernels it runs.
//!
//! **Timing.** Each fake device keeps a virtual timeline, in nanoseconds from 0 at `open`. A
//! launch advances it by RFC-0060 §8.4's
//!
//! ```text
//! t_launch = Σ_tiers bytes_read / W_eff(device, tier, kernel class)
//!          + flops / F_eff(device)
//!          + t_fixed share
//! ```
//!
//! times a jitter factor drawn from LogNormal(1, σ): a median of 1, and exactly 1 when σ is 0.
//! The time goes on the timeline when the launch is queued, not when the CPU runs it, so the
//! timeline does not depend on how the host schedules the work.
//!
//! - **Bytes read** are the input tensors' spans and the weight views. Each counts at the W_eff
//!   of the tier its buffer was placed in ([`FakeKernelProvider::alloc_in`],
//!   [`FakeKernelProvider::import_in`]), with the op's kernel class override where the tier has
//!   one ([`FakeTierSpec::with_class`]). Outputs, the step descriptor and the workspace are not
//!   counted: the formula is over bytes read.
//! - **Flops** are [`flops`]: two per multiply-add of the weight-bound linear ops, one per
//!   output element for every other op.
//! - **The `t_fixed` share.** A launch queued directly pays the device's whole `t_fixed`. A
//!   replay of a captured graph pays one `t_fixed` for the graph, split equally over its
//!   launches and copies, which is what capture saves.
//!
//! A `copy` advances the timeline by its bytes at the source tier's W_eff (the `Copy` class
//! override where there is one) plus the `t_fixed` share, with the same jitter. A `record`
//! stamps its event with the timeline's time, and [`KernelProvider::elapsed`] returns the
//! difference of two stamps, so a probe that times launches between events measures the
//! spec's figures. In a capture, the stamps and the advances happen at each replay.
//!
//! **Capacity.** Each tier has a capacity. `alloc` and `import` count against their tier's and
//! fail with `AllocFailed`, class `Capacity`, past it; `free` gives the bytes back. A request
//! goes to the tier its `MemKind` names: `Device` to `Vram`, `Unified` to `Unified`, and
//! `HostMapped` (and every import) to `HostPinned`, else `HostPageable`. A device whose spec
//! has no such tier refuses the request as `Unsupported`, as a real provider without that
//! memory does.
//!
//! The `Digest` mode of RFC-0060 §8.4 is QUAL-E8-T07's.

// The fake returns the provider trait's `core::Error`, RFC-0001 §5.12's struct with its fields
// inline, over clippy's 128-byte limit for an `Err`, as `kernels-cpu` does.
#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fmt;

use syntropy_core::resource::{Backend, KernelClass, TierKind};
use syntropy_core::{Bytes, DeviceId, Domain, ErrorClass, HostRegion, Nanos};
use syntropy_kernels::{
    AllocRequest, BufferSlice, CapturedGraph, DeviceBuffer, DeviceCtx, DeviceHealth,
    DeviceMemoryInfo, Error, ErrorCode, Event, HostFlag, HostRegistrar, HostWaker, KernelProvider,
    LayoutVersion, MemKind, MemoryDomain, OpArgs, OpCaps, OpKey, OpKind, OpRegistry, OpenOptions,
    PreparedOp, ProviderBuild, ProviderDevice, ProviderId, Registration, Segment, ShapeSig, Stream,
    StreamPriority, WaitCmp, unsupported,
};
use syntropy_kernels_cpu::{CpuProvider, KernelFn, KillSwitches, Run};

use super::SimEntropy;
use super::dist::{exp, standard_normal};

/// One memory tier of a fake device (RFC-0060 §8.4).
#[derive(Clone, Debug, PartialEq)]
pub struct FakeTierSpec {
    /// The tier kind.
    pub tier: TierKind,
    /// W_eff for a kernel class with no override, in bytes per second.
    pub w_eff: f64,
    /// W_eff for a kernel class, in bytes per second, where it differs from [`Self::w_eff`]:
    /// RFC-0060 §8.4's `W_eff(device, tier, kernel class)`.
    pub w_eff_by_class: BTreeMap<KernelClass, f64>,
    /// What the tier holds.
    pub capacity: Bytes,
}

impl FakeTierSpec {
    /// A tier of kind `tier`, read at `w_eff` bytes per second by every kernel class, holding
    /// `capacity`.
    #[must_use]
    pub fn new(tier: TierKind, w_eff: f64, capacity: Bytes) -> FakeTierSpec {
        FakeTierSpec {
            tier,
            w_eff,
            w_eff_by_class: BTreeMap::new(),
            capacity,
        }
    }

    /// The same tier with kernel class `class` reading it at `w_eff` bytes per second.
    #[must_use]
    pub fn with_class(mut self, class: KernelClass, w_eff: f64) -> FakeTierSpec {
        self.w_eff_by_class.insert(class, w_eff);
        self
    }

    /// The W_eff a kernel of `class` reads this tier at, in bytes per second.
    #[must_use]
    pub fn w(&self, class: Option<KernelClass>) -> f64 {
        class
            .and_then(|c| self.w_eff_by_class.get(&c).copied())
            .unwrap_or(self.w_eff)
    }
}

/// The figures that time one fake device (RFC-0060 §8.4): per-tier W_eff and capacity, F_eff,
/// `t_fixed` and the jitter σ. They are the figures `FakeProbe` reports, with noise.
#[derive(Clone, Debug, PartialEq)]
pub struct FakeDeviceSpec {
    /// The name `enumerate` reports.
    pub name: String,
    /// The tiers the device's kernels read, each kind at most once.
    pub tiers: Vec<FakeTierSpec>,
    /// F_eff, in flops per second.
    pub f_eff: f64,
    /// The fixed cost of one launch queued directly.
    pub t_fixed: Nanos,
    /// σ of the LogNormal(1, σ) jitter factor; 0 turns jitter off.
    pub sigma: f64,
    /// The seed of the device's jitter stream.
    pub seed: u64,
}

/// Why a [`FakeDeviceSpec`] was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FakeSpecError {
    /// The spec lists no tier.
    NoTiers { device: String },
    /// The spec lists a tier kind twice.
    DuplicateTier { device: String, tier: TierKind },
    /// A rate, `f_eff` or a W_eff, that is not finite and positive.
    BadRate { device: String, what: String },
    /// A σ that is negative or not finite.
    BadSigma { device: String },
}

impl fmt::Display for FakeSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FakeSpecError::NoTiers { device } => write!(f, "fake device {device} has no tier"),
            FakeSpecError::DuplicateTier { device, tier } => {
                write!(f, "fake device {device} lists tier {tier:?} twice")
            }
            FakeSpecError::BadRate { device, what } => write!(
                f,
                "fake device {device}: {what} must be finite and greater than 0"
            ),
            FakeSpecError::BadSigma { device } => write!(
                f,
                "fake device {device}: sigma must be finite and not negative"
            ),
        }
    }
}

impl std::error::Error for FakeSpecError {}

impl FakeDeviceSpec {
    /// Checks the spec: at least one tier, no tier kind twice, every rate finite and positive,
    /// and σ finite and not negative.
    ///
    /// # Errors
    ///
    /// The first problem found.
    pub fn check(&self) -> Result<(), FakeSpecError> {
        let device = || self.name.clone();
        let rate = |value: f64, what: String| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(FakeSpecError::BadRate {
                    device: device(),
                    what,
                })
            }
        };
        if self.tiers.is_empty() {
            return Err(FakeSpecError::NoTiers { device: device() });
        }
        for (i, tier) in self.tiers.iter().enumerate() {
            if self.tiers[..i].iter().any(|t| t.tier == tier.tier) {
                return Err(FakeSpecError::DuplicateTier {
                    device: device(),
                    tier: tier.tier,
                });
            }
            rate(tier.w_eff, format!("w_eff of {:?}", tier.tier))?;
            for (class, w) in &tier.w_eff_by_class {
                rate(*w, format!("w_eff of {:?} for {class:?}", tier.tier))?;
            }
        }
        rate(self.f_eff, "f_eff".to_owned())?;
        if !(self.sigma.is_finite() && self.sigma >= 0.0) {
            return Err(FakeSpecError::BadSigma { device: device() });
        }
        Ok(())
    }

    /// The index of the tier of kind `tier`, if the device has one.
    #[must_use]
    pub fn tier_index(&self, tier: TierKind) -> Option<usize> {
        self.tiers.iter().position(|t| t.tier == tier)
    }

    /// The tier a request for `kind` goes to (see the [module](self) docs).
    fn tier_for(&self, kind: MemKind) -> Option<usize> {
        match kind {
            MemKind::Device => self.tier_index(TierKind::Vram),
            MemKind::Unified => self.tier_index(TierKind::Unified),
            _ => self.host_tier(),
        }
    }

    /// `HostPinned`, else `HostPageable`.
    fn host_tier(&self) -> Option<usize> {
        self.tier_index(TierKind::HostPinned)
            .or_else(|| self.tier_index(TierKind::HostPageable))
    }

    /// The device's own memory: its `Vram` or `Unified` tier.
    fn local_tier(&self) -> Option<usize> {
        self.tier_index(TierKind::Vram)
            .or_else(|| self.tier_index(TierKind::Unified))
    }
}

/// How a [`FakeKernelProvider`] computes (RFC-0060 §8.4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FakeMode {
    /// Real host buffers, and every op on the CPU provider's deterministic kernels.
    Numeric,
}

/// The fake `KernelProvider` of RFC-0060 §8.4 (see the [module](self) docs).
#[derive(Debug)]
pub struct FakeKernelProvider {
    cpu: CpuProvider,
    devices: Vec<FakeDeviceSpec>,
    mode: FakeMode,
}

/// A buffer the fake placed in a tier.
#[derive(Clone, Copy, Debug)]
struct Placed {
    addr: u64,
    bytes: u64,
    tier: usize,
}

/// What a captured stream call does to the timeline at each replay.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// A launch or copy: its bytes and flops term, in nanoseconds, without `t_fixed`.
    Work(f64),
    /// A `record` of this event.
    Stamp(u64),
}

/// The fake's state for one open device, inside its `DeviceCtx`.
struct FakeDevice {
    cpu: DeviceCtx,
    spec: FakeDeviceSpec,
    jitter: SimEntropy,
    /// The virtual timeline, in nanoseconds since `open`.
    now: Cell<u64>,
    used: RefCell<Vec<u64>>,
    /// Placed buffers by handle, and their handles by address, for the weight views.
    buffers: RefCell<BTreeMap<u64, Placed>>,
    by_addr: RefCell<BTreeMap<u64, u64>>,
    /// Each prepared op's kind and `k_in`, by the CPU provider's handle.
    prepared: RefCell<BTreeMap<u64, (OpKind, ShapeSig)>>,
    /// Streams capturing, with the steps captured so far.
    captures: RefCell<BTreeMap<u64, Vec<Step>>>,
    /// Captured graphs' steps, by the CPU provider's graph handle.
    graphs: RefCell<BTreeMap<u64, Vec<Step>>>,
    /// Each event's last stamp, by handle; `None` until it is first recorded.
    stamps: RefCell<BTreeMap<u64, Option<u64>>>,
}

impl FakeDevice {
    /// The tier of the buffer with `handle`; the host tier, or the first, for one the fake did
    /// not place.
    fn tier_of_handle(&self, handle: u64) -> usize {
        self.buffers
            .borrow()
            .get(&handle)
            .map_or_else(|| self.default_tier(), |p| p.tier)
    }

    /// The tier of the placed buffer holding `addr`, or the default.
    fn tier_of_addr(&self, addr: u64) -> usize {
        let by_addr = self.by_addr.borrow();
        let buffers = self.buffers.borrow();
        by_addr
            .range(..=addr)
            .next_back()
            .and_then(|(_, handle)| buffers.get(handle))
            .filter(|p| addr < p.addr.saturating_add(p.bytes))
            .map_or_else(|| self.default_tier(), |p| p.tier)
    }

    fn default_tier(&self) -> usize {
        self.spec.host_tier().unwrap_or(0)
    }

    /// The bytes and flops term of a launch, in nanoseconds.
    fn launch_ns(&self, kind: &OpKind, shape: &ShapeSig, args: &OpArgs<'_>) -> f64 {
        let class = kind.kernel_class();
        let mut ns = 0.0;
        for input in args.inputs {
            let tier = &self.spec.tiers[self.tier_of_handle(input.buf.buffer)];
            let bytes = input.span_bytes().min(input.buf.bytes.0);
            ns += bytes as f64 * 1e9 / tier.w(class);
        }
        for weight in args.weights {
            let tier = &self.spec.tiers[self.tier_of_addr(weight.addr)];
            ns += weight.len.0 as f64 * 1e9 / tier.w(class);
        }
        ns + flops(kind, shape, args) * 1e9 / self.spec.f_eff
    }

    /// The jitter factor: LogNormal(1, σ), exactly 1 when σ is 0.
    fn jitter(&self) -> f64 {
        if self.spec.sigma == 0.0 {
            1.0
        } else {
            exp(self.spec.sigma * standard_normal(&self.jitter))
        }
    }

    /// Advances the timeline by `(work + fixed) × jitter`.
    fn advance(&self, work: f64, fixed: f64) {
        let t = (work + fixed) * self.jitter();
        self.now.set(self.now.get().saturating_add(round(t)));
    }

    /// A launch or copy on `s`: captured if `s` is capturing, else on the timeline now with the
    /// whole `t_fixed`.
    fn work(&self, s: &Stream, ns: f64) {
        if let Some(steps) = self.captures.borrow_mut().get_mut(&s.handle) {
            steps.push(Step::Work(ns));
            return;
        }
        self.advance(ns, self.spec.t_fixed.0 as f64);
    }

    fn place(&self, buf: &DeviceBuffer, tier: usize) {
        self.buffers.borrow_mut().insert(
            buf.handle,
            Placed {
                addr: buf.addr,
                bytes: buf.bytes.0,
                tier,
            },
        );
        self.by_addr.borrow_mut().insert(buf.addr, buf.handle);
        self.used.borrow_mut()[tier] += buf.bytes.0;
    }

    /// Refuses `bytes` more in `tier` past its capacity.
    fn reserve(&self, tier: usize, bytes: u64) -> Result<(), Error> {
        let spec = &self.spec.tiers[tier];
        let used = self.used.borrow()[tier];
        if used.saturating_add(bytes) > spec.capacity.0 {
            return Err(Error::new(
                ErrorClass::Capacity,
                syntropy_kernels::error::codes::ALLOC_FAILED,
                format!(
                    "fake device {}: {bytes} bytes more in {:?} would pass its capacity of {} \
                     ({used} used)",
                    self.spec.name, spec.tier, spec.capacity.0
                ),
            ));
        }
        Ok(())
    }
}

/// Nanoseconds to the nearest whole one; 0 for a value that is not positive, saturating.
fn round(ns: f64) -> u64 {
    if ns.is_nan() || ns <= 0.0 {
        0
    } else {
        // `as` saturates at `u64::MAX`.
        (ns + 0.5) as u64
    }
}

/// The flops the fake counts for one launch of `kind` at `shape` with `args`.
///
/// The weight-bound linear ops count two flops per multiply-add: `Gemv`, `Gemm` and `Logits`
/// `2 × k_in` per output element, and `GateUpAct`, two products, `4 × k_in`. Every other op
/// counts one flop per output element. `k_in` is the shape's second dim (RFC-0040 §3.1's
/// `n_out, k_in` and `n_vocab, hidden`), and the output elements are those of every output
/// tensor, the whole bucket.
#[must_use]
pub fn flops(kind: &OpKind, shape: &ShapeSig, args: &OpArgs<'_>) -> f64 {
    let out: f64 = args
        .outputs
        .iter()
        .map(|t| t.dims.iter().map(|&d| f64::from(d)).product::<f64>())
        .sum();
    let k_in = shape.dims.get(1).map_or(1.0, |&k| f64::from(k));
    match kind {
        OpKind::Gemv | OpKind::Gemm | OpKind::Logits => 2.0 * k_in * out,
        OpKind::GateUpAct { .. } => 4.0 * k_in * out,
        _ => out,
    }
}

fn internal(message: String) -> Error {
    Error::new(
        ErrorClass::Internal,
        ErrorCode::new(Domain::Kernels, 0),
        message,
    )
}

impl FakeKernelProvider {
    /// A fake in `Numeric` mode with one device per spec, in order.
    ///
    /// # Errors
    ///
    /// A spec that fails [`FakeDeviceSpec::check`].
    pub fn numeric(devices: Vec<FakeDeviceSpec>) -> Result<FakeKernelProvider, FakeSpecError> {
        for spec in &devices {
            spec.check()?;
        }
        Ok(FakeKernelProvider {
            cpu: CpuProvider::new(KillSwitches::NONE),
            devices,
            mode: FakeMode::Numeric,
        })
    }

    /// The same fake with `kernel` as the CPU provider's stand-in for `kind`
    /// ([`CpuProvider::with_kernel`]), for tests. A stand-in declares no capabilities, so a fake
    /// device, always opened deterministic, refuses it at `prepare`.
    #[must_use]
    pub fn with_kernel(mut self, kind: OpKind, kernel: KernelFn) -> FakeKernelProvider {
        self.cpu = self.cpu.with_kernel(kind, kernel);
        self
    }

    /// The mode the fake computes in.
    #[must_use]
    pub fn mode(&self) -> FakeMode {
        self.mode
    }

    /// The CPU provider underneath.
    #[must_use]
    pub fn cpu(&self) -> &CpuProvider {
        &self.cpu
    }

    fn device<'a>(&self, ctx: &'a DeviceCtx) -> Result<&'a FakeDevice, Error> {
        ctx.downcast_ref::<FakeDevice>()
            .ok_or_else(|| internal("the context is not a fake device's".to_owned()))
    }

    /// The CPU provider's context under fake device `ctx`, for the CPU provider's own calls:
    /// [`CpuProvider::doorbells`], [`CpuProvider::register_waker`], [`CpuProvider::take_runner`].
    ///
    /// # Errors
    ///
    /// The context is not a fake device's.
    pub fn cpu_ctx<'a>(&self, ctx: &'a DeviceCtx) -> Result<&'a DeviceCtx, Error> {
        Ok(&self.device(ctx)?.cpu)
    }

    /// Runs stream `s`'s queued work on the calling thread: [`CpuProvider::run`], the path the
    /// simulator takes (RFC-0060 §2.3).
    ///
    /// # Errors
    ///
    /// As [`CpuProvider::run`].
    pub fn run(&self, ctx: &DeviceCtx, s: &Stream) -> Result<Run, Error> {
        self.cpu.run(self.cpu_ctx(ctx)?, s)
    }

    /// The device's virtual time: nanoseconds since `open`.
    ///
    /// # Errors
    ///
    /// The context is not a fake device's.
    pub fn now(&self, ctx: &DeviceCtx) -> Result<Nanos, Error> {
        Ok(Nanos(self.device(ctx)?.now.get()))
    }

    /// The spec the device was opened with.
    ///
    /// # Errors
    ///
    /// The context is not a fake device's.
    pub fn spec<'a>(&self, ctx: &'a DeviceCtx) -> Result<&'a FakeDeviceSpec, Error> {
        Ok(&self.device(ctx)?.spec)
    }

    /// The bytes allocated and imported in the device's tier of kind `tier`; `None` if it has
    /// none.
    ///
    /// # Errors
    ///
    /// The context is not a fake device's.
    pub fn used(&self, ctx: &DeviceCtx, tier: TierKind) -> Result<Option<Bytes>, Error> {
        let device = self.device(ctx)?;
        Ok(device
            .spec
            .tier_index(tier)
            .map(|i| Bytes(device.used.borrow()[i])))
    }

    fn tier(device: &FakeDevice, tier: TierKind) -> Result<usize, Error> {
        device.spec.tier_index(tier).ok_or_else(unsupported)
    }

    /// `alloc`, in the device's tier of kind `tier` rather than the one `req.kind` names. The
    /// memory is host memory from the CPU provider either way.
    ///
    /// # Errors
    ///
    /// `Unsupported` if the device has no such tier or `req` is `reserve_only`; `AllocFailed`
    /// past the tier's capacity, or as the CPU provider's `alloc`.
    pub fn alloc_in(
        &self,
        ctx: &DeviceCtx,
        req: AllocRequest,
        tier: TierKind,
    ) -> Result<DeviceBuffer, Error> {
        let device = self.device(ctx)?;
        let tier = Self::tier(device, tier)?;
        self.alloc_at(device, req, tier)
    }

    fn alloc_at(
        &self,
        device: &FakeDevice,
        req: AllocRequest,
        tier: usize,
    ) -> Result<DeviceBuffer, Error> {
        if req.reserve_only {
            return Err(unsupported());
        }
        device.reserve(tier, req.bytes.0)?;
        let host = AllocRequest {
            kind: MemKind::HostMapped,
            ..req
        };
        let mut buf = self.cpu.alloc(&device.cpu, host)?;
        buf.kind = req.kind;
        device.place(&buf, tier);
        Ok(buf)
    }

    /// `import`, into the device's tier of kind `tier`.
    ///
    /// # Errors
    ///
    /// `Unsupported` if the device has no such tier; `AllocFailed` past the tier's capacity, or
    /// as the CPU provider's `import`.
    pub fn import_in(
        &self,
        ctx: &DeviceCtx,
        region: HostRegion,
        tier: TierKind,
    ) -> Result<DeviceBuffer, Error> {
        let device = self.device(ctx)?;
        let tier = Self::tier(device, tier)?;
        self.import_at(device, region, tier)
    }

    fn import_at(
        &self,
        device: &FakeDevice,
        region: HostRegion,
        tier: usize,
    ) -> Result<DeviceBuffer, Error> {
        device.reserve(tier, region.len.0)?;
        let buf = self.cpu.import(&device.cpu, region)?;
        device.place(&buf, tier);
        Ok(buf)
    }
}

impl KernelProvider for FakeKernelProvider {
    fn id(&self) -> ProviderId {
        ProviderId::new("fake")
    }

    /// The kernels are the CPU provider's.
    fn backend(&self) -> Backend {
        Backend::Cpu
    }

    fn build(&self) -> ProviderBuild {
        self.cpu.build()
    }

    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Host
    }

    /// One device per spec, with the host CPU's class and features.
    fn enumerate(&self) -> Result<Vec<ProviderDevice>, Error> {
        let Some(template) = self.cpu.enumerate()?.into_iter().next() else {
            return Ok(Vec::new());
        };
        Ok(self
            .devices
            .iter()
            .enumerate()
            .map(|(local, spec)| {
                let local_tier = spec.local_tier().map(|i| &spec.tiers[i]);
                let total = local_tier.map_or(Bytes(0), |t| t.capacity);
                ProviderDevice {
                    local: u32::try_from(local).unwrap_or(u32::MAX),
                    name: spec.name.clone(),
                    memory: DeviceMemoryInfo {
                        total,
                        free: total,
                        budget: None,
                        host_visible_device_local: None,
                        max_alloc: if local_tier.is_some() {
                            total
                        } else {
                            Bytes(u64::MAX)
                        },
                        unified: spec.tier_index(TierKind::Unified).is_some(),
                    },
                    ..template.clone()
                }
            })
            .collect())
    }

    /// Opens fake device `dev.ordinal` on CPU device 0, always deterministic.
    fn open(&self, dev: DeviceId, opts: &OpenOptions) -> Result<DeviceCtx, Error> {
        let spec = self.devices.get(usize::from(dev.ordinal)).ok_or_else(|| {
            Error::new(
                ErrorClass::Unsupported,
                syntropy_kernels::error::codes::OP_UNSUPPORTED,
                format!("no fake device {}", dev.ordinal),
            )
        })?;
        let cpu = self.cpu.open(
            DeviceId { ordinal: 0, ..dev },
            &OpenOptions {
                deterministic: true,
                ..*opts
            },
        )?;
        Ok(DeviceCtx::new(FakeDevice {
            cpu,
            jitter: SimEntropy::new(spec.seed),
            now: Cell::new(0),
            used: RefCell::new(vec![0; spec.tiers.len()]),
            buffers: RefCell::new(BTreeMap::new()),
            by_addr: RefCell::new(BTreeMap::new()),
            prepared: RefCell::new(BTreeMap::new()),
            captures: RefCell::new(BTreeMap::new()),
            graphs: RefCell::new(BTreeMap::new()),
            stamps: RefCell::new(BTreeMap::new()),
            spec: spec.clone(),
        }))
    }

    /// The device's own tier, `Vram` or `Unified`: its capacity and what is left. A device with
    /// neither has no memory of its own, as the CPU provider's do not.
    fn memory_info(&self, ctx: &DeviceCtx) -> Result<DeviceMemoryInfo, Error> {
        let device = self.device(ctx)?;
        let i = device.spec.local_tier().ok_or_else(unsupported)?;
        let total = device.spec.tiers[i].capacity;
        Ok(DeviceMemoryInfo {
            total,
            free: Bytes(total.0.saturating_sub(device.used.borrow()[i])),
            budget: None,
            host_visible_device_local: None,
            max_alloc: total,
            unified: device.spec.tiers[i].tier == TierKind::Unified,
        })
    }

    fn ops(&self) -> &OpRegistry {
        self.cpu.ops()
    }

    fn query(&self, ctx: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Option<OpCaps> {
        self.cpu.query(&self.device(ctx).ok()?.cpu, op, shape)
    }

    /// Host memory in the tier `req.kind` names (see the [module](self) docs).
    fn alloc(&self, ctx: &DeviceCtx, req: AllocRequest) -> Result<DeviceBuffer, Error> {
        let device = self.device(ctx)?;
        let tier = device.spec.tier_for(req.kind).ok_or_else(unsupported)?;
        self.alloc_at(device, req, tier)
    }

    fn free(&self, ctx: &DeviceCtx, buf: DeviceBuffer) {
        let Ok(device) = self.device(ctx) else {
            return;
        };
        if let Some(placed) = device.buffers.borrow_mut().remove(&buf.handle) {
            device.by_addr.borrow_mut().remove(&placed.addr);
            let mut used = device.used.borrow_mut();
            used[placed.tier] = used[placed.tier].saturating_sub(placed.bytes);
        }
        self.cpu.free(&device.cpu, buf);
    }

    fn map_segment(
        &self,
        ctx: &DeviceCtx,
        reservation: &DeviceBuffer,
        offset: Bytes,
        bytes: Bytes,
    ) -> Result<Segment, Error> {
        self.cpu
            .map_segment(self.cpu_ctx(ctx)?, reservation, offset, bytes)
    }

    fn unmap_segment(&self, ctx: &DeviceCtx, seg: Segment) -> Result<(), Error> {
        self.cpu.unmap_segment(self.cpu_ctx(ctx)?, seg)
    }

    fn register_host(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<Registration, Error> {
        self.cpu.register_host(self.cpu_ctx(ctx)?, region)
    }

    fn host_registrar(&self, ctx: &DeviceCtx) -> Option<Box<dyn HostRegistrar>> {
        self.cpu.host_registrar(self.cpu_ctx(ctx).ok()?)
    }

    /// The tier manager's memory, as it is, in the host tier.
    fn import(&self, ctx: &DeviceCtx, region: HostRegion) -> Result<DeviceBuffer, Error> {
        let device = self.device(ctx)?;
        let tier = device.spec.host_tier().ok_or_else(unsupported)?;
        self.import_at(device, region, tier)
    }

    fn set_resident(
        &self,
        ctx: &DeviceCtx,
        bufs: &[&DeviceBuffer],
        resident: bool,
    ) -> Result<(), Error> {
        self.cpu.set_resident(self.cpu_ctx(ctx)?, bufs, resident)
    }

    fn repack(
        &self,
        op: &OpKey,
        src: &[u8],
        dst: &mut [u8],
    ) -> Option<Result<LayoutVersion, Error>> {
        self.cpu.repack(op, src, dst)
    }

    fn repacked_len(&self, op: &OpKey, src: Bytes) -> Option<Bytes> {
        self.cpu.repacked_len(op, src)
    }

    fn stream(&self, ctx: &DeviceCtx, prio: StreamPriority) -> Result<Stream, Error> {
        self.cpu.stream(self.cpu_ctx(ctx)?, prio)
    }

    fn event(&self, ctx: &DeviceCtx) -> Result<Event, Error> {
        let device = self.device(ctx)?;
        let e = self.cpu.event(&device.cpu)?;
        // Inserted now, so a `record` in a step replaces a value and allocates nothing.
        device.stamps.borrow_mut().insert(e.handle, None);
        Ok(e)
    }

    /// Records `e` on the CPU stream, and stamps it with the timeline's time: now, or at each
    /// replay of a capture.
    fn record(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error> {
        let device = self.device(ctx)?;
        self.cpu.record(&device.cpu, s, e)?;
        if let Some(steps) = device.captures.borrow_mut().get_mut(&s.handle) {
            steps.push(Step::Stamp(e.handle));
            return Ok(());
        }
        device
            .stamps
            .borrow_mut()
            .insert(e.handle, Some(device.now.get()));
        Ok(())
    }

    fn wait(&self, ctx: &DeviceCtx, s: &Stream, e: &Event) -> Result<(), Error> {
        self.cpu.wait(self.cpu_ctx(ctx)?, s, e)
    }

    fn signal_host(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        flag: HostFlag,
        value: u32,
    ) -> Result<(), Error> {
        self.cpu.signal_host(self.cpu_ctx(ctx)?, s, flag, value)
    }

    fn wait_host(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        flag: HostFlag,
        value: u32,
        cmp: WaitCmp,
        reset: Option<u32>,
    ) -> Result<(), Error> {
        self.cpu
            .wait_host(self.cpu_ctx(ctx)?, s, flag, value, cmp, reset)
    }

    fn arm_wake(&self, ctx: &DeviceCtx, s: &Stream, waker: &HostWaker) -> Result<(), Error> {
        self.cpu.arm_wake(self.cpu_ctx(ctx)?, s, waker)
    }

    fn poll(&self, ctx: &DeviceCtx, e: &Event) -> Result<bool, Error> {
        self.cpu.poll(self.cpu_ctx(ctx)?, e)
    }

    /// The virtual time between the stamps of `start` and `end`; `None` until both are
    /// recorded or if `end` is stamped before `start`.
    fn elapsed(&self, ctx: &DeviceCtx, start: &Event, end: &Event) -> Option<Nanos> {
        let device = self.device(ctx).ok()?;
        let stamps = device.stamps.borrow();
        let (a, b) = ((*stamps.get(&start.handle)?)?, (*stamps.get(&end.handle)?)?);
        b.checked_sub(a).map(Nanos)
    }

    fn synchronize(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error> {
        self.cpu.synchronize(self.cpu_ctx(ctx)?, s)
    }

    /// The CPU provider's copy, then the timeline: the bytes at the source tier's W_eff.
    fn copy(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        src: BufferSlice,
        dst: BufferSlice,
    ) -> Result<(), Error> {
        let device = self.device(ctx)?;
        self.cpu.copy(&device.cpu, s, src, dst)?;
        let tier = &device.spec.tiers[device.tier_of_handle(src.buffer)];
        let ns = src.bytes.0 as f64 * 1e9 / tier.w(Some(KernelClass::Copy));
        device.work(s, ns);
        Ok(())
    }

    /// The CPU provider's `prepare` on a device opened deterministic, so a kernel that does not
    /// declare itself deterministic is refused.
    fn prepare(&self, ctx: &DeviceCtx, op: &OpKey, shape: &ShapeSig) -> Result<PreparedOp, Error> {
        let device = self.device(ctx)?;
        let prepared = self.cpu.prepare(&device.cpu, op, shape)?;
        device
            .prepared
            .borrow_mut()
            .insert(prepared.handle, (op.op, shape.clone()));
        Ok(prepared)
    }

    /// The CPU provider's launch, then the timeline (see the [module](self) docs).
    fn launch(
        &self,
        ctx: &DeviceCtx,
        s: &Stream,
        op: &PreparedOp,
        args: &OpArgs<'_>,
    ) -> Result<(), Error> {
        let device = self.device(ctx)?;
        self.cpu.launch(&device.cpu, s, op, args)?;
        let ns = {
            let prepared = device.prepared.borrow();
            let (kind, shape) = prepared
                .get(&op.handle)
                .ok_or_else(|| internal(format!("no prepared op {} on this device", op.handle)))?;
            device.launch_ns(kind, shape, args)
        };
        device.work(s, ns);
        Ok(())
    }

    fn begin_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<(), Error> {
        let device = self.device(ctx)?;
        self.cpu.begin_capture(&device.cpu, s)?;
        device.captures.borrow_mut().insert(s.handle, Vec::new());
        Ok(())
    }

    fn end_capture(&self, ctx: &DeviceCtx, s: &Stream) -> Result<CapturedGraph, Error> {
        let device = self.device(ctx)?;
        let graph = self.cpu.end_capture(&device.cpu, s)?;
        let steps = device
            .captures
            .borrow_mut()
            .remove(&s.handle)
            .unwrap_or_default();
        device.graphs.borrow_mut().insert(graph.handle, steps);
        Ok(graph)
    }

    /// The CPU provider's replay, then the graph's steps on the timeline: each launch and copy
    /// with an equal share of one `t_fixed`, each record stamped as it is reached.
    fn replay(&self, ctx: &DeviceCtx, s: &Stream, g: &CapturedGraph) -> Result<(), Error> {
        let device = self.device(ctx)?;
        self.cpu.replay(&device.cpu, s, g)?;
        let graphs = device.graphs.borrow();
        let Some(steps) = graphs.get(&g.handle) else {
            return Ok(());
        };
        let works = steps.iter().filter(|s| matches!(s, Step::Work(_))).count();
        let share = if works == 0 {
            0.0
        } else {
            device.spec.t_fixed.0 as f64 / works as f64
        };
        for step in steps {
            match *step {
                Step::Work(ns) => device.advance(ns, share),
                Step::Stamp(e) => {
                    device.stamps.borrow_mut().insert(e, Some(device.now.get()));
                }
            }
        }
        Ok(())
    }

    fn health(&self, ctx: &DeviceCtx) -> DeviceHealth {
        match self.device(ctx) {
            Ok(device) => self.cpu.health(&device.cpu),
            Err(_) => DeviceHealth::Ok,
        }
    }
}
