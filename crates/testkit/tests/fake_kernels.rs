// SPDX-License-Identifier: MIT OR Apache-2.0

//! QUAL-E3-T12: `FakeKernelProvider` in `Numeric` mode and `FakeDeviceSpec` (RFC-0060 §8.4;
//! ADR-0600 decision 4).
//!
//! - The fake passes the same RFC-0040 §2 contract checks as the CPU provider: one generic
//!   check, run on both ([`contract`]).
//! - `Numeric` mode runs on the CPU provider shell's streams, buffers and capture: a `Gemv`
//!   through the fake, launched directly and replayed, equals the CPU provider's bit for bit, on
//!   two fake devices alike, and a kernel that does not declare itself deterministic is refused.
//! - A launch advances the timeline by `bytes/W + flops/F + t_fixed share` within 1% with jitter
//!   off, directly, replayed, and as a probe measures it between two events.

#![allow(
    clippy::result_large_err,
    reason = "core::Error is RFC-0001 §5.12's struct, 344 B inline [measured: size_of, debug, x86-64 Linux, ce2044ae] (COMP-E13-T15)"
)]

use std::sync::atomic::Ordering;

use smallvec::smallvec;
use syntropy_core::resource::{KernelClass, TierKind};
use syntropy_core::{
    BufferView, Bytes, DType, DeviceId, ErrorClass, GgmlType, HostRegion, Nanos, NodeId,
    QuantScheme, ViewKind,
};
use syntropy_kernels::step::StepDesc;
use syntropy_kernels::{
    AllocRequest, BufferSlice, DeviceCtx, DeviceHealth, Error, KernelProvider, LayoutVersion,
    MemKind, OpArgs, OpKey, OpKind, OpenOptions, PowerHint, PreparedOp, ShapeClass, ShapeSig,
    Stream, StreamPlan, StreamPriority, TensorArg, WaitCmp, unsupported,
};
use syntropy_kernels_cpu::{CpuProvider, Doorbells, KillSwitches, Launch, Run};
use syntropy_testkit::sim::fake_kernels::flops;
use syntropy_testkit::sim::{
    FakeDeviceSpec, FakeKernelProvider, FakeMode, FakeSpecError, FakeTierSpec,
};

const GIB: u64 = 1 << 30;

/// A CPU-like device: pinned and pageable host memory, no memory of its own. The figures are
/// chosen for the tests [target], not taken from any part.
fn cpu_spec(name: &str, sigma: f64, seed: u64) -> FakeDeviceSpec {
    FakeDeviceSpec {
        name: name.to_owned(),
        tiers: vec![
            FakeTierSpec::new(TierKind::HostPinned, 20e9, Bytes(GIB))
                .with_class(KernelClass::DenseGemv, 16e9),
            FakeTierSpec::new(TierKind::HostPageable, 10e9, Bytes(GIB)),
        ],
        f_eff: 0.5e12,
        t_fixed: Nanos(5_000),
        sigma,
        seed,
    }
}

/// A discrete-GPU-like device: VRAM of 64 MiB and pinned host memory.
fn gpu_spec() -> FakeDeviceSpec {
    FakeDeviceSpec {
        name: "fake-gpu".to_owned(),
        tiers: vec![
            FakeTierSpec::new(TierKind::Vram, 900e9, Bytes(64 << 20)),
            FakeTierSpec::new(TierKind::HostPinned, 25e9, Bytes(GIB)),
        ],
        f_eff: 80e12,
        t_fixed: Nanos(8_000),
        sigma: 0.0,
        seed: 7,
    }
}

fn options(deterministic: bool) -> OpenOptions {
    OpenOptions {
        streams: StreamPlan {
            compute: 1,
            copy_in: 1,
            copy_out: 0,
            collective: 0,
        },
        doorbell_slots: 8,
        capture: true,
        deterministic,
        power: PowerHint::Mains,
    }
}

fn device(ordinal: u16) -> DeviceId {
    DeviceId {
        node: NodeId([4; 32]),
        ordinal,
    }
}

fn fake(specs: Vec<FakeDeviceSpec>) -> FakeKernelProvider {
    FakeKernelProvider::numeric(specs).expect("valid specs")
}

/// What the contract checks need beyond the trait: running a stream inline, and the doorbells.
trait Shell: KernelProvider {
    fn run_inline(&self, ctx: &DeviceCtx, s: &Stream) -> Result<Run, Error>;
    fn bells(&self, ctx: &DeviceCtx) -> Doorbells;
}

impl Shell for CpuProvider {
    fn run_inline(&self, ctx: &DeviceCtx, s: &Stream) -> Result<Run, Error> {
        self.run(ctx, s)
    }
    fn bells(&self, ctx: &DeviceCtx) -> Doorbells {
        self.doorbells(ctx).unwrap()
    }
}

impl Shell for FakeKernelProvider {
    fn run_inline(&self, ctx: &DeviceCtx, s: &Stream) -> Result<Run, Error> {
        self.run(ctx, s)
    }
    fn bells(&self, ctx: &DeviceCtx) -> Doorbells {
        self.cpu().doorbells(self.cpu_ctx(ctx).unwrap()).unwrap()
    }
}

/// Host memory the test owns, as 8-byte words, so it is aligned for every dtype without
/// `unsafe`.
struct Mem {
    words: Vec<u64>,
}

impl Mem {
    fn zeroed(bytes: usize) -> Mem {
        Mem {
            words: vec![0; bytes.div_ceil(8).max(1)],
        }
    }

    fn from_bytes(bytes: &[u8]) -> Mem {
        let mut mem = Mem::zeroed(bytes.len());
        mem.write(0, bytes);
        mem
    }

    /// Writes `bytes` at byte offset `at`, before a launch reads them.
    fn write(&mut self, at: usize, bytes: &[u8]) {
        for (i, b) in bytes.iter().enumerate() {
            let (word, shift) = ((at + i) / 8, ((at + i) % 8) * 8);
            self.words[word] = (self.words[word] & !(0xFF << shift)) | (u64::from(*b) << shift);
        }
    }

    fn bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn region(&mut self) -> HostRegion {
        HostRegion {
            addr: self.words.as_mut_ptr() as u64,
            len: Bytes((self.words.len() * 8) as u64),
        }
    }
}

fn slice(buffer: u64, offset: u64, bytes: u64) -> BufferSlice {
    BufferSlice {
        buffer,
        offset: Bytes(offset),
        bytes: Bytes(bytes),
    }
}

/// RFC-0040 §2's contract over host buffers, as the CPU provider shell's tests check it
/// (`kernels-cpu/tests/provider_shell.rs`), written once over the trait.
fn contract<P: Shell>(p: &P, ctx: &DeviceCtx) {
    // Memory: host memory allocated, imported and registered as it is.
    let buf = p
        .alloc(
            ctx,
            AllocRequest {
                bytes: Bytes(4096),
                kind: MemKind::HostMapped,
                align: Bytes(0),
                reserve_only: false,
            },
        )
        .unwrap();
    assert_eq!(buf.addr % 64, 0, "at least cache-line aligned");
    assert_eq!((buf.bytes, buf.kind), (Bytes(4096), MemKind::HostMapped));
    for kind in [MemKind::Device, MemKind::Unified] {
        let request = AllocRequest {
            bytes: Bytes(64),
            kind,
            align: Bytes(64),
            reserve_only: false,
        };
        assert_eq!(
            p.alloc(ctx, request).unwrap_err(),
            unsupported(),
            "{kind:?}"
        );
    }
    let reserve = AllocRequest {
        bytes: Bytes(64),
        kind: MemKind::HostMapped,
        align: Bytes(64),
        reserve_only: true,
    };
    assert_eq!(p.alloc(ctx, reserve).unwrap_err(), unsupported());
    let empty = AllocRequest {
        bytes: Bytes(0),
        ..reserve
    };
    let empty = p
        .alloc(
            ctx,
            AllocRequest {
                reserve_only: false,
                ..empty
            },
        )
        .unwrap_err();
    assert_eq!(
        (empty.class, empty.code),
        (
            ErrorClass::Capacity,
            syntropy_kernels::error::codes::ALLOC_FAILED
        )
    );
    let mut owned = Mem::from_bytes(&[7u8; 128]);
    let region = owned.region();
    let imported = p.import(ctx, region).unwrap();
    assert_eq!((imported.addr, imported.bytes), (region.addr, region.len));
    assert_eq!(p.register_host(ctx, region).unwrap().region, region);
    p.free(ctx, imported);
    p.free(ctx, buf);
    assert_eq!(
        owned.bytes(),
        vec![7u8; 128],
        "freeing an import leaves the memory"
    );

    // A copy runs in stream order; out of bounds and mismatched lengths are refused.
    let s = p.stream(ctx, StreamPriority::CopyIn).unwrap();
    let src_bytes: Vec<u8> = (0..64).collect();
    let mut src = Mem::from_bytes(&src_bytes);
    let mut dst = Mem::zeroed(64);
    let a = p.import(ctx, src.region()).unwrap().handle;
    let b = p.import(ctx, dst.region()).unwrap().handle;
    p.copy(ctx, &s, slice(a, 16, 32), slice(b, 0, 32)).unwrap();
    assert_eq!(p.run_inline(ctx, &s).unwrap(), Run::Idle);
    assert_eq!(&dst.bytes()[..32], &src_bytes[16..48]);
    assert!(dst.bytes()[32..].iter().all(|&x| x == 0));
    assert!(p.copy(ctx, &s, slice(a, 60, 8), slice(b, 0, 8)).is_err());
    assert!(p.copy(ctx, &s, slice(a, 0, 8), slice(b, 0, 4)).is_err());

    // Events: recorded on one stream, waited on by another, polled.
    let (s1, s2) = (
        p.stream(ctx, StreamPriority::Compute).unwrap(),
        p.stream(ctx, StreamPriority::CopyIn).unwrap(),
    );
    let e = p.event(ctx).unwrap();
    p.record(ctx, &s1, &e).unwrap();
    p.wait(ctx, &s2, &e).unwrap();
    assert!(!p.poll(ctx, &e).unwrap(), "recorded, not reached");
    assert_eq!(p.run_inline(ctx, &s2).unwrap(), Run::WaitingForEvent);
    assert_eq!(p.run_inline(ctx, &s1).unwrap(), Run::Idle);
    assert!(p.poll(ctx, &e).unwrap());
    assert_eq!(p.run_inline(ctx, &s2).unwrap(), Run::Idle);

    // A doorbell wait passes on its value, then the consumer resets it.
    let bells = p.bells(ctx);
    let (incoming, outgoing) = (bells.flag_at(3), bells.flag_at(4));
    p.wait_host(ctx, &s1, incoming, 1, WaitCmp::Eq, Some(0))
        .unwrap();
    p.signal_host(ctx, &s1, outgoing, 1).unwrap();
    assert_eq!(
        p.run_inline(ctx, &s1).unwrap(),
        Run::WaitingForFlag(incoming)
    );
    assert_eq!(bells.flag(outgoing).unwrap().load(Ordering::Acquire), 0);
    bells.flag(incoming).unwrap().store(1, Ordering::Release);
    assert_eq!(p.run_inline(ctx, &s1).unwrap(), Run::Idle);
    assert_eq!(
        bells.flag(incoming).unwrap().load(Ordering::Acquire),
        0,
        "consumer reset"
    );
    assert_eq!(bells.flag(outgoing).unwrap().swap(0, Ordering::AcqRel), 1);

    // Capture records an op list, runs nothing, and each replay records afresh.
    let flag = bells.flag_at(0);
    let e = p.event(ctx).unwrap();
    p.begin_capture(ctx, &s1).unwrap();
    assert!(p.begin_capture(ctx, &s1).is_err(), "one capture at a time");
    p.signal_host(ctx, &s1, flag, 1).unwrap();
    p.record(ctx, &s1, &e).unwrap();
    let graph = p.end_capture(ctx, &s1).unwrap();
    assert_eq!(p.run_inline(ctx, &s1).unwrap(), Run::Idle);
    assert_eq!(bells.flag(flag).unwrap().load(Ordering::Acquire), 0);
    for _ in 0..3 {
        p.replay(ctx, &s1, &graph).unwrap();
        assert!(!p.poll(ctx, &e).unwrap(), "each replay records afresh");
        assert_eq!(p.run_inline(ctx, &s1).unwrap(), Run::Idle);
        assert!(p.poll(ctx, &e).unwrap());
        assert_eq!(bells.flag(flag).unwrap().swap(0, Ordering::AcqRel), 1);
    }

    // `synchronize` drains a stream here, and the device is healthy.
    p.signal_host(ctx, &s1, flag, 2).unwrap();
    p.synchronize(ctx, &s1).unwrap();
    assert_eq!(bells.flag(flag).unwrap().swap(0, Ordering::AcqRel), 2);
    assert_eq!(p.health(ctx), DeviceHealth::Ok);
    drop((src, dst));
}

#[test]
fn the_fake_passes_the_contract_checks_the_cpu_provider_passes() {
    let cpu = CpuProvider::new(KillSwitches::NONE);
    let ctx = cpu.open(device(0), &options(true)).unwrap();
    contract(&cpu, &ctx);

    let f = fake(vec![cpu_spec("fake0", 0.0, 1)]);
    assert_eq!(f.mode(), FakeMode::Numeric);
    let ctx = f.open(device(0), &options(true)).unwrap();
    contract(&f, &ctx);
}

#[test]
fn the_fake_enumerates_its_specs_not_the_host() {
    let f = fake(vec![
        cpu_spec("fake0", 0.0, 1),
        gpu_spec(),
        cpu_spec("fake2", 0.0, 3),
    ]);
    let host = CpuProvider::new(KillSwitches::NONE).enumerate().unwrap();
    let devices = f.enumerate().unwrap();
    let names: Vec<&str> = devices.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, ["fake0", "fake-gpu", "fake2"]);
    for (i, d) in devices.iter().enumerate() {
        assert_eq!(d.local as usize, i);
        assert_eq!(d.class, host[0].class, "the host CPU's kernels");
    }
    assert_eq!(devices[1].memory.total, Bytes(64 << 20));
    assert_eq!(devices[0].memory.total, Bytes(0));
    let missing = f.open(device(3), &options(true)).unwrap_err();
    assert_eq!(missing.class, ErrorClass::Unsupported);
    assert_eq!(f.id().0, "fake");
}

#[test]
fn bad_specs_are_refused() {
    let mut no_tiers = cpu_spec("a", 0.0, 1);
    no_tiers.tiers.clear();
    let mut twice = cpu_spec("b", 0.0, 1);
    twice.tiers[1].tier = TierKind::HostPinned;
    let mut slow = cpu_spec("c", 0.0, 1);
    slow.f_eff = 0.0;
    let mut class = cpu_spec("d", 0.0, 1);
    class.tiers[0] = class.tiers[0]
        .clone()
        .with_class(KernelClass::Copy, f64::NAN);
    let mut sigma = cpu_spec("e", 0.0, 1);
    sigma.sigma = -0.1;
    let refused = |spec: FakeDeviceSpec| FakeKernelProvider::numeric(vec![spec]).unwrap_err();
    assert!(matches!(refused(no_tiers), FakeSpecError::NoTiers { .. }));
    assert!(matches!(
        refused(twice),
        FakeSpecError::DuplicateTier {
            tier: TierKind::HostPinned,
            ..
        }
    ));
    assert!(matches!(refused(slow), FakeSpecError::BadRate { .. }));
    assert!(matches!(refused(class), FakeSpecError::BadRate { .. }));
    assert!(matches!(refused(sigma), FakeSpecError::BadSigma { .. }));
}

#[test]
fn capacity_is_counted_per_tier_and_given_back() {
    let f = fake(vec![gpu_spec()]);
    let ctx = f.open(device(0), &options(true)).unwrap();
    let request = |bytes: u64, kind| AllocRequest {
        bytes: Bytes(bytes),
        kind,
        align: Bytes(64),
        reserve_only: false,
    };
    // `Device` memory goes to VRAM: real host memory, reported as the kind asked for.
    let a = f.alloc(&ctx, request(48 << 20, MemKind::Device)).unwrap();
    assert_eq!(a.kind, MemKind::Device);
    assert_eq!(f.used(&ctx, TierKind::Vram).unwrap(), Some(Bytes(48 << 20)));
    let info = f.memory_info(&ctx).unwrap();
    assert_eq!((info.total, info.free), (Bytes(64 << 20), Bytes(16 << 20)));
    let over = f
        .alloc(&ctx, request(32 << 20, MemKind::Device))
        .unwrap_err();
    assert_eq!(
        (over.class, over.code),
        (
            ErrorClass::Capacity,
            syntropy_kernels::error::codes::ALLOC_FAILED
        )
    );
    // Host memory goes to the pinned tier, and an explicit tier wins over the kind.
    let h = f
        .alloc(&ctx, request(1 << 20, MemKind::HostMapped))
        .unwrap();
    assert_eq!(
        f.used(&ctx, TierKind::HostPinned).unwrap(),
        Some(Bytes(1 << 20))
    );
    let v = f
        .alloc_in(&ctx, request(16 << 20, MemKind::HostMapped), TierKind::Vram)
        .unwrap();
    assert_eq!(f.used(&ctx, TierKind::Vram).unwrap(), Some(Bytes(64 << 20)));
    assert_eq!(f.used(&ctx, TierKind::Unified).unwrap(), None);
    assert_eq!(
        f.alloc(&ctx, request(64, MemKind::Unified)).unwrap_err(),
        unsupported(),
        "no unified tier"
    );
    for buf in [a, h, v] {
        f.free(&ctx, buf);
    }
    assert_eq!(f.used(&ctx, TierKind::Vram).unwrap(), Some(Bytes(0)));
    assert_eq!(f.used(&ctx, TierKind::HostPinned).unwrap(), Some(Bytes(0)));
    // An import counts too.
    let mut big = Mem::zeroed(4096);
    let mut tight = gpu_spec();
    tight.tiers[1].capacity = Bytes(1024);
    let f = fake(vec![tight]);
    let ctx = f.open(device(0), &options(true)).unwrap();
    assert_eq!(
        f.import(&ctx, big.region()).unwrap_err().class,
        ErrorClass::Capacity
    );
}

fn stand_in(_: &Launch<'_>) -> Result<(), Error> {
    Ok(())
}

/// The fake opens every device deterministic: a kernel that declares no deterministic variant
/// is refused even when the caller did not ask, where the CPU provider opened non-deterministic
/// takes it.
#[test]
fn a_kernel_not_declared_deterministic_is_refused() {
    let key = OpKey {
        op: OpKind::Gemm,
        schemes: smallvec![QuantScheme::Plain(DType::F32)],
        layout: LayoutVersion::SOURCE,
        variant: 0,
    };
    let shape = ShapeSig {
        class: ShapeClass::Decode,
        rows: 1,
        dims: smallvec![64],
    };
    let cpu = CpuProvider::new(KillSwitches::NONE).with_kernel(OpKind::Gemm, stand_in);
    let ctx = cpu.open(device(0), &options(false)).unwrap();
    cpu.prepare(&ctx, &key, &shape).unwrap();

    let f = fake(vec![cpu_spec("fake0", 0.0, 1)]).with_kernel(OpKind::Gemm, stand_in);
    let ctx = f.open(device(0), &options(false)).unwrap();
    let error = f.prepare(&ctx, &key, &shape).unwrap_err();
    assert_eq!(
        (error.class, error.code),
        (
            ErrorClass::Unsupported,
            syntropy_kernels::error::codes::OP_UNSUPPORTED
        )
    );
}

/// Deterministic test values (SplitMix64).
struct Gen(u64);

impl Gen {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn f16(v: f32) -> [u8; 2] {
    syntropy_kernels::reference::act_quant::f16_bits(v).to_le_bytes()
}

/// One `Gemv` problem: weights, activations and the shape.
struct Problem {
    scheme: QuantScheme,
    rows: u32,
    n_out: u32,
    k: u32,
    weights: Vec<u8>,
    x: Vec<u8>,
}

impl Problem {
    fn new(scheme: QuantScheme, rows: u32, n_out: u32, k: u32, seed: u64) -> Problem {
        let mut g = Gen(seed);
        let mut weights = Vec::new();
        match scheme {
            QuantScheme::Plain(DType::F16) => {
                for _ in 0..n_out * k {
                    weights.extend(f16(g.unit()));
                }
            }
            QuantScheme::Ggml(GgmlType::Q8_0) => {
                for _ in 0..n_out * k / 32 {
                    weights.extend(f16(0.01 + g.unit().abs() * 0.01));
                    weights.extend((0..32).map(|_| g.next() as u8));
                }
            }
            other => panic!("no weights for {other:?}"),
        }
        let x = (0..rows * k)
            .flat_map(|_| (g.unit() * 4.0).to_le_bytes())
            .collect();
        Problem {
            scheme,
            rows,
            n_out,
            k,
            weights,
            x,
        }
    }

    fn key(&self) -> OpKey {
        OpKey {
            op: OpKind::Gemv,
            schemes: smallvec![
                self.scheme,
                QuantScheme::Plain(DType::F32),
                QuantScheme::Plain(DType::F32)
            ],
            layout: LayoutVersion::SOURCE,
            variant: 0,
        }
    }

    fn shape(&self) -> ShapeSig {
        ShapeSig {
            class: ShapeClass::Decode,
            rows: self.rows,
            dims: smallvec![self.n_out, self.k],
        }
    }

    fn out_bytes(&self) -> usize {
        (self.rows * self.n_out * 4) as usize
    }
}

/// Which operand a buffer holds, for placing it in a tier.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Weights,
    Input,
    Other,
}

/// A `Gemv` made ready on one device: its operands imported, its op prepared. The memory is
/// kept here until the test ends, since the device holds its addresses.
struct Bound {
    op: PreparedOp,
    mems: Vec<Mem>,
    /// Weights, activations, output, step descriptor, workspace, zeros of the output's size.
    handles: [u64; 6],
}

impl Bound {
    fn new(
        p: &impl KernelProvider,
        ctx: &DeviceCtx,
        problem: &Problem,
        import: &dyn Fn(Role, HostRegion) -> u64,
    ) -> Bound {
        let op = p.prepare(ctx, &problem.key(), &problem.shape()).unwrap();
        let mut step = Mem::zeroed(size_of::<StepDesc>());
        step.write(
            std::mem::offset_of!(StepDesc, rows_active),
            &problem.rows.to_le_bytes(),
        );
        let mut mems = vec![
            Mem::from_bytes(&problem.weights),
            Mem::from_bytes(&problem.x),
            Mem::zeroed(problem.out_bytes()),
            step,
            Mem::zeroed(op.workspace.0 as usize),
            Mem::zeroed(problem.out_bytes()),
        ];
        let roles = [
            Role::Weights,
            Role::Input,
            Role::Other,
            Role::Other,
            Role::Other,
            Role::Other,
        ];
        let mut handles = [0u64; 6];
        for (i, mem) in mems.iter_mut().enumerate() {
            handles[i] = import(roles[i], mem.region());
        }
        Bound { op, mems, handles }
    }

    fn launch(&self, p: &impl KernelProvider, ctx: &DeviceCtx, s: &Stream, problem: &Problem) {
        let (rows, n_out, k) = (problem.rows, problem.n_out, problem.k);
        let inputs = [TensorArg::packed(
            slice(self.handles[1], 0, u64::from(rows * k * 4)),
            DType::F32,
            [rows, k, 1, 1],
        )
        .unwrap()];
        let outputs = [TensorArg::packed(
            slice(self.handles[2], 0, problem.out_bytes() as u64),
            DType::F32,
            [rows, n_out, 1, 1],
        )
        .unwrap()];
        let weights = [BufferView {
            kind: ViewKind::Host,
            addr: self.mems[0].words.as_ptr() as u64,
            len: Bytes(problem.weights.len() as u64),
        }];
        let args = OpArgs {
            inputs: &inputs,
            outputs: &outputs,
            weights: &weights,
            step: slice(self.handles[3], 0, size_of::<StepDesc>() as u64),
            workspace: (self.op.workspace.0 > 0)
                .then(|| slice(self.handles[4], 0, self.op.workspace.0)),
            statics: &[],
        };
        p.launch(ctx, s, &self.op, &args).unwrap();
    }

    /// Queues a copy of zeros over the output.
    fn clear(&self, p: &impl KernelProvider, ctx: &DeviceCtx, s: &Stream, problem: &Problem) {
        let bytes = problem.out_bytes() as u64;
        p.copy(
            ctx,
            s,
            slice(self.handles[5], 0, bytes),
            slice(self.handles[2], 0, bytes),
        )
        .unwrap();
    }

    fn out_bits(&self, problem: &Problem) -> Vec<u32> {
        self.mems[2].bytes()[..problem.out_bytes()]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }
}

/// Runs `problem` on `p`'s device `ctx` directly, then replayed from a capture twice, each run
/// into a cleared output; returns the outputs of the three runs.
fn gemv_runs<P: Shell>(p: &P, ctx: &DeviceCtx, problem: &Problem) -> Vec<Vec<u32>> {
    let s = p.stream(ctx, StreamPriority::Compute).unwrap();
    let bound = Bound::new(p, ctx, problem, &|_, region| {
        p.import(ctx, region).unwrap().handle
    });
    let mut runs = Vec::new();
    bound.launch(p, ctx, &s, problem);
    assert_eq!(p.run_inline(ctx, &s).unwrap(), Run::Idle);
    runs.push(bound.out_bits(problem));
    p.begin_capture(ctx, &s).unwrap();
    bound.launch(p, ctx, &s, problem);
    let graph = p.end_capture(ctx, &s).unwrap();
    for _ in 0..2 {
        bound.clear(p, ctx, &s, problem);
        assert_eq!(p.run_inline(ctx, &s).unwrap(), Run::Idle);
        assert!(bound.out_bits(problem).iter().all(|&b| b == 0));
        p.replay(ctx, &s, &graph).unwrap();
        assert_eq!(p.run_inline(ctx, &s).unwrap(), Run::Idle);
        runs.push(bound.out_bits(problem));
    }
    runs
}

fn problems() -> [Problem; 2] {
    [
        Problem::new(QuantScheme::Plain(DType::F16), 4, 24, 96, 11),
        Problem::new(QuantScheme::Ggml(GgmlType::Q8_0), 3, 16, 128, 12),
    ]
}

/// `Numeric` mode is the CPU provider's arithmetic on the CPU provider's streams, buffers and
/// capture: the fake's outputs equal a deterministic CPU device's bit for bit, directly and
/// replayed, and two fake devices give the same bits.
#[test]
fn numeric_mode_computes_the_cpu_providers_bits() {
    let cpu = CpuProvider::new(KillSwitches::NONE);
    let cpu_ctx = cpu.open(device(0), &options(true)).unwrap();
    let f = fake(vec![cpu_spec("fake0", 0.0, 1), gpu_spec()]);
    let ctx0 = f.open(device(0), &options(false)).unwrap();
    let ctx1 = f.open(device(1), &options(true)).unwrap();
    for problem in problems() {
        let want = gemv_runs(&cpu, &cpu_ctx, &problem);
        assert!(want[0].iter().any(|&b| b != 0), "the kernel wrote");
        assert_eq!(want[0], want[1]);
        assert_eq!(want[0], want[2]);
        assert_eq!(gemv_runs(&f, &ctx0, &problem), want, "{:?}", problem.scheme);
        assert_eq!(gemv_runs(&f, &ctx1, &problem), want, "{:?}", problem.scheme);
    }
}

/// The work term of one `Gemv` on [`cpu_spec`]'s device with weights pinned and activations
/// pageable, computed from the spec's figures by hand, in nanoseconds.
fn by_hand(problem: &Problem) -> f64 {
    let weights = problem.weights.len() as f64 / 16e9; // pinned, DenseGemv override
    let x = f64::from(problem.rows * problem.k * 4) / 10e9; // pageable, no override
    let flops = 2.0 * f64::from(problem.k) * f64::from(problem.rows * problem.n_out);
    (weights + x + flops / 0.5e12) * 1e9
}

fn within_1_percent(got: Nanos, want: f64) {
    let rel = (got.0 as f64 - want).abs() / want;
    assert!(
        rel < 0.01,
        "{} ns against {want} ns: {:.4}%",
        got.0,
        rel * 100.0
    );
}

/// Binds `problem` on the fake with weights in `HostPinned`, activations in `HostPageable`.
fn bind_placed(f: &FakeKernelProvider, ctx: &DeviceCtx, problem: &Problem) -> Bound {
    Bound::new(f, ctx, problem, &|role, region| {
        let tier = match role {
            Role::Weights => TierKind::HostPinned,
            Role::Input => TierKind::HostPageable,
            Role::Other => TierKind::HostPinned,
        };
        f.import_in(ctx, region, tier).unwrap().handle
    })
}

#[test]
fn a_launch_advances_the_timeline_by_the_cost_formula() {
    let f = fake(vec![cpu_spec("fake0", 0.0, 1)]);
    let ctx = f.open(device(0), &options(true)).unwrap();
    let s = f.stream(&ctx, StreamPriority::Compute).unwrap();
    let problem = Problem::new(QuantScheme::Plain(DType::F16), 4, 256, 1024, 21);
    let bound = bind_placed(&f, &ctx, &problem);
    let work = by_hand(&problem);
    let t_fixed = 5_000.0;
    assert_eq!(f.now(&ctx).unwrap(), Nanos(0));

    // Directly: the work term and the whole t_fixed. Queued time, not run time.
    bound.launch(&f, &ctx, &s, &problem);
    let one = f.now(&ctx).unwrap();
    within_1_percent(one, work + t_fixed);
    assert_eq!(f.run(&ctx, &s).unwrap(), Run::Idle);
    assert_eq!(
        f.now(&ctx).unwrap(),
        one,
        "running the CPU work moves nothing"
    );

    // As a probe measures it: between two events, over 10 launches.
    let (e0, e1) = (f.event(&ctx).unwrap(), f.event(&ctx).unwrap());
    assert_eq!(f.elapsed(&ctx, &e0, &e1), None, "not recorded yet");
    f.record(&ctx, &s, &e0).unwrap();
    for _ in 0..10 {
        bound.launch(&f, &ctx, &s, &problem);
    }
    f.record(&ctx, &s, &e1).unwrap();
    let ten = f.elapsed(&ctx, &e0, &e1).unwrap();
    within_1_percent(ten, 10.0 * (work + t_fixed));
    assert_eq!(f.elapsed(&ctx, &e1, &e0), None, "end before start");
    assert_eq!(f.run(&ctx, &s).unwrap(), Run::Idle);

    // Replayed: four launches share one t_fixed; the records inside stamp at each replay.
    let (r0, r1) = (f.event(&ctx).unwrap(), f.event(&ctx).unwrap());
    f.begin_capture(&ctx, &s).unwrap();
    f.record(&ctx, &s, &r0).unwrap();
    for _ in 0..4 {
        bound.launch(&f, &ctx, &s, &problem);
    }
    f.record(&ctx, &s, &r1).unwrap();
    let graph = f.end_capture(&ctx, &s).unwrap();
    let before = f.now(&ctx).unwrap();
    assert_eq!(f.elapsed(&ctx, &r0, &r1), None, "a capture stamps nothing");
    for _ in 0..3 {
        let start = f.now(&ctx).unwrap();
        f.replay(&ctx, &s, &graph).unwrap();
        let replay = Nanos(f.now(&ctx).unwrap().0 - start.0);
        within_1_percent(replay, 4.0 * work + t_fixed);
        assert_eq!(f.elapsed(&ctx, &r0, &r1), Some(replay));
        assert_eq!(f.run(&ctx, &s).unwrap(), Run::Idle);
    }
    assert!(f.now(&ctx).unwrap() > before);

    // A copy: its bytes at the source tier's W_eff, plus t_fixed.
    let start = f.now(&ctx).unwrap();
    let bytes = problem.out_bytes() as u64;
    f.copy(
        &ctx,
        &s,
        slice(bound.handles[1], 0, bytes),
        slice(bound.handles[2], 0, bytes),
    )
    .unwrap();
    within_1_percent(
        Nanos(f.now(&ctx).unwrap().0 - start.0),
        bytes as f64 / 10e9 * 1e9 + t_fixed,
    );
    assert_eq!(f.run(&ctx, &s).unwrap(), Run::Idle);
}

#[test]
fn flops_follow_the_documented_model() {
    let problem = Problem::new(QuantScheme::Plain(DType::F16), 2, 8, 64, 1);
    let y = TensorArg::packed(slice(1, 0, 64), DType::F32, [2, 8, 1, 1]).unwrap();
    let args = OpArgs {
        outputs: &[y],
        ..OpArgs::none()
    };
    assert_eq!(
        flops(&OpKind::Gemv, &problem.shape(), &args),
        2.0 * 64.0 * 16.0
    );
    assert_eq!(
        flops(
            &OpKind::GateUpAct {
                act: syntropy_kernels::GatedAct::Silu
            },
            &problem.shape(),
            &args
        ),
        4.0 * 64.0 * 16.0
    );
    assert_eq!(flops(&OpKind::Add, &problem.shape(), &args), 16.0);
}

/// With σ > 0 each launch's time is scaled by a LogNormal(1, σ) draw from the device's seed:
/// the same seed replays the same timeline, another seed gives another, and the median factor
/// is near 1.
#[test]
fn jitter_is_seeded_and_centred_on_the_formula() {
    let timeline = |seed: u64| {
        let f = fake(vec![cpu_spec("fake0", 0.2, seed)]);
        let ctx = f.open(device(0), &options(true)).unwrap();
        let s = f.stream(&ctx, StreamPriority::Compute).unwrap();
        let problem = Problem::new(QuantScheme::Plain(DType::F16), 1, 64, 256, 5);
        let bound = bind_placed(&f, &ctx, &problem);
        let mut factors = Vec::new();
        for _ in 0..1000 {
            let start = f.now(&ctx).unwrap();
            bound.launch(&f, &ctx, &s, &problem);
            let t = f.now(&ctx).unwrap().0 - start.0;
            factors.push(t as f64 / (by_hand(&problem) + 5_000.0));
            if factors.len() % 256 == 0 {
                assert_eq!(f.run(&ctx, &s).unwrap(), Run::Idle);
            }
        }
        assert_eq!(f.run(&ctx, &s).unwrap(), Run::Idle);
        factors
    };
    let a = timeline(42);
    assert_eq!(a, timeline(42), "one seed, one timeline");
    assert_ne!(a, timeline(43));
    let mut sorted = a.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    // σ = 0.2 over 1,000 draws: the sample median's standard error is about
    // 1.2533 × 0.2 / √1000 ≈ 0.008 in log space [estimate], so 0.05 is over six of them.
    assert!((median - 1.0).abs() < 0.05, "median factor {median}");
    assert!(sorted[0] < 0.9 && sorted[999] > 1.1, "the factors spread");
}
